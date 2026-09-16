"""Port a real HuggingFace GPT-2 checkpoint into NNX and verify nnx_save.

Consumes the directory produced by ``port_hf_reference.py`` (a real HF
``model.safetensors`` plus torch reference logits) and, in a JAX-only
environment:

1. builds the NNX model from the checkpoint's own config,
2. maps every HF tensor onto its NNX path (the actual PyTorch -> JAX port),
3. checks the ported logits against the torch reference,
4. runs an nnx_save save/load round-trip and checks the logits again,
5. repeats in bfloat16, the dtype a TPU inference port would use.

Example (on victoria):
  /home/tensor/nnx_save_test/.venv/bin/python tests/port_hf_check.py \
      --dir /home/tensor/nnx_save_test/hf_tiny
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import load_file as load_np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GPT2Config, TinyGPT2, jax_path_for  # noqa: E402
from util import flatten, pure_dict  # noqa: E402

from nnx_save import load_model, save_model  # noqa: E402


def nested(flat_paths: dict[str, np.ndarray]) -> dict:
    out: dict = {}
    for path, value in flat_paths.items():
        cur = out
        parts = path.split("/")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def port(model: TinyGPT2, checkpoint: dict[str, np.ndarray], cfg: GPT2Config) -> tuple[int, list[str]]:
    """Map HF tensors onto the model's real paths (tied embeddings included)."""
    expected = set(flatten(pure_dict(model)))
    mapped, unknown, dropped = {}, [], []
    for key, value in checkpoint.items():
        target = jax_path_for(key, n_embd=cfg.n_embd)
        if target is None:
            unknown.append(key)
            continue
        # A shared embedding Module is recorded under one attribute only, and
        # which one is an NNX graph-traversal detail; resolve it here.
        if target not in expected and target == "wte/embedding" and "lm_head/embedding" in expected:
            target = "lm_head/embedding"
        if target not in expected:
            dropped.append(target)
            continue
        mapped[target] = value
    if dropped:
        print(f"WARNING: checkpoint tensors with no model path: {dropped}")
    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, nested(mapped))
    nnx.update(model, state)
    return len(mapped), unknown


def cast(model: TinyGPT2, dtype) -> TinyGPT2:
    import jax

    nnx.update(
        model,
        jax.tree.map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(model),
        ),
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None, help="where to write the nnx_save checkpoint")
    args = ap.parse_args()

    ckpt = load_np(os.path.join(args.dir, "model.safetensors"))
    ref = np.load(os.path.join(args.dir, "reference.npz"), allow_pickle=False)
    raw = json.loads(str(ref["config"]))
    ids = jnp.asarray(ref["ids"])
    ref_logits = ref["logits"]

    cfg = GPT2Config(
        vocab_size=raw["vocab_size"],
        n_positions=raw["n_positions"] if "n_positions" in raw else raw["max_position_embeddings"],
        n_embd=raw["n_embd"] if "n_embd" in raw else raw["hidden_size"],
        n_layer=raw["n_layer"] if "n_layer" in raw else raw["num_hidden_layers"],
        n_head=raw["n_head"] if "n_head" in raw else raw["num_attention_heads"],
        layer_norm_epsilon=raw.get("layer_norm_epsilon", 1e-5),
    )
    print(f"config: {cfg}")
    print(f"checkpoint tensors: {len(ckpt)}, tied={raw.get('tie_word_embeddings')}")

    model = TinyGPT2(cfg, tie_embeddings=bool(raw.get("tie_word_embeddings", True)), rngs=nnx.Rngs(0))
    n_mapped, unknown = port(model, ckpt, cfg)
    expected_paths = set(flatten(pure_dict(model)))
    print(f"ported {n_mapped} tensors, {len(unknown)} unknown keys: {unknown[:4]}")

    logits = np.asarray(model(ids))
    diff = np.abs(logits - ref_logits)
    print(f"[port]   max|jax-torch| = {diff.max():.3e}  (mean {diff.mean():.3e})")
    np.testing.assert_allclose(logits, ref_logits, rtol=1e-4, atol=1e-4)
    print("[port]   OK")

    out = args.out or os.path.join(args.dir, "nnx_saved.safetensors")
    t0 = time.perf_counter()
    save_model(model, out)
    t1 = time.perf_counter()
    print(f"[save]   {os.path.getsize(out)/1e6:.1f} MB in {t1-t0:.2f}s -> {out}")

    fresh = TinyGPT2(cfg, tie_embeddings=bool(raw.get("tie_word_embeddings", True)), rngs=nnx.Rngs(99))
    loaded, _ = load_model(fresh, out)
    t2 = time.perf_counter()
    print(f"[load]   {t2-t1:.2f}s")
    after = np.asarray(loaded(ids))
    print(f"[after]  max|after-before| = {np.abs(after - logits).max():.3e}")
    np.testing.assert_array_equal(after, logits)
    np.testing.assert_allclose(after, ref_logits, rtol=1e-4, atol=1e-4)
    assert loaded.lm_head is loaded.wte, "tied embedding lost by load_model"
    print("[after]  OK: logits identical and still match torch")

    # --- bf16, the dtype a TPU inference port uses -------------------------
    bf16 = TinyGPT2(cfg, tie_embeddings=bool(raw.get("tie_word_embeddings", True)), rngs=nnx.Rngs(0))
    port(bf16, ckpt, cfg)
    cast(bf16, jnp.bfloat16)
    bf16_path = out.replace(".safetensors", "_bf16.safetensors")
    save_model(bf16, bf16_path)
    print(f"[bf16]   saved {os.path.getsize(bf16_path)/1e6:.1f} MB")
    # A TPU inference port declares bf16 in the model itself, so build the
    # fresh model in bf16 (nnx_save adopts the model's dtype on load).
    fresh = cast(
        TinyGPT2(cfg, tie_embeddings=bool(raw.get("tie_word_embeddings", True)), rngs=nnx.Rngs(7)),
        jnp.bfloat16,
    )
    loaded_bf16, _ = load_model(fresh, bf16_path)
    dtypes = {np.dtype(v.dtype).name for v in flatten(pure_dict(loaded_bf16)).values()}
    assert dtypes == {"bfloat16"}, f"dtype changed: {dtypes}"
    bf16_after = np.asarray(loaded_bf16(ids))
    np.testing.assert_array_equal(bf16_after, np.asarray(bf16(ids)))
    rel = np.abs(bf16_after - ref_logits).max() / max(np.abs(ref_logits).max(), 1e-6)
    print(f"[bf16]   dtypes preserved, max relative deviation from fp32 torch = {rel:.3e}")
    print("[bf16]   OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
