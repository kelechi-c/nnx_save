"""Port a real HuggingFace checkpoint into NNX and verify nnx_save.

Consumes the directory produced by ``port_hf_reference.py`` (a real HF
``model.safetensors`` plus torch reference logits) and, in a JAX-only
environment:

1. builds the NNX model from the checkpoint's own config,
2. maps every HF tensor onto its NNX path (the actual PyTorch -> JAX port),
3. checks the ported logits against the torch reference,
4. runs an nnx_save save/load round-trip and checks the logits again,
5. repeats in bfloat16, the dtype a TPU inference port would use.

The architecture is taken from the checkpoint's own ``model_type``, so both
GPT-2 and Llama-style checkpoints work; the file layout is the same for both.

Example (on victoria):
  /home/tensor/nnx_save_test/.venv/bin/python tests/port_hf_check.py \
      --dir /home/tensor/nnx_save_test/hf_tiny
  /home/tensor/nnx_save_test/.venv/bin/python tests/port_hf_check.py \
      --dir /tmp/hf_tiny_llama
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import load_file as load_np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (  # noqa: E402
    GPT2Config,
    LlamaConfig,
    TinyGPT2,
    TinyLlama,
    jax_path_for,
    llama_jax_path_for,
)
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


def no_transpose(path: str, tensor: np.ndarray) -> np.ndarray:
    """HF GPT-2 stores Conv1D weights (in, out), i.e. already NNX layout."""
    return tensor


def transpose_kernels(path: str, tensor: np.ndarray) -> np.ndarray:
    """HF ``nn.Linear`` is (out, in); an NNX kernel is (in, out)."""
    if path.endswith("/kernel") and tensor.ndim == 2:
        return tensor.T
    return tensor


@dataclass
class Arch:
    """The architecture-specific pieces the shared flow needs."""

    name: str
    cfg: object
    build: callable  # zero-argument builder -> NNX module
    mapper: callable  # hf key -> nnx path | None
    transform: callable  # (nnx path, tensor) -> tensor
    aliases: dict = field(default_factory=dict)  # path -> fallback path
    tied: bool = False  # lm_head shares the token embedding
    embed_paths: tuple = ()  # the two attributes of a shared embedding
    warnings: tuple = ()


def gpt2_arch(raw: dict) -> Arch:
    cfg = GPT2Config(
        vocab_size=raw["vocab_size"],
        n_positions=raw["n_positions"] if "n_positions" in raw else raw["max_position_embeddings"],
        n_embd=raw["n_embd"] if "n_embd" in raw else raw["hidden_size"],
        n_layer=raw["n_layer"] if "n_layer" in raw else raw["num_hidden_layers"],
        n_head=raw["n_head"] if "n_head" in raw else raw["num_attention_heads"],
        layer_norm_epsilon=raw.get("layer_norm_epsilon", 1e-5),
    )
    tied = bool(raw.get("tie_word_embeddings", True))
    return Arch(
        name="gpt2",
        cfg=cfg,
        build=lambda: TinyGPT2(cfg, tie_embeddings=tied, rngs=nnx.Rngs(0)),
        mapper=lambda key: jax_path_for(key, n_embd=cfg.n_embd),
        transform=no_transpose,
        # A shared embedding Module is recorded under one attribute only, and
        # which one is an NNX graph-traversal detail; resolve it here.
        aliases={"wte/embedding": "lm_head/embedding"},
        tied=tied,
        embed_paths=("wte/embedding", "lm_head/embedding"),
    )


def llama_arch(raw: dict) -> Arch:
    cfg = LlamaConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        intermediate_size=raw["intermediate_size"],
        max_position_embeddings=raw.get("max_position_embeddings", 2048),
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        rope_theta=raw.get("rope_theta", 10000.0),
    )
    warned = ()
    if raw.get("rope_scaling"):
        warned = (
            f"config has rope_scaling={raw['rope_scaling']}; this port only implements "
            "the default RoPE, so the logits check is expected to fail",
        )
    return Arch(
        name="llama",
        cfg=cfg,
        build=lambda: TinyLlama(cfg, rngs=nnx.Rngs(0)),
        mapper=llama_jax_path_for,
        transform=transpose_kernels,
        tied=False,
        embed_paths=("embed_tokens/embedding", "lm_head/kernel"),
        warnings=warned,
    )


ARCHES = {"gpt2": gpt2_arch, "llama": llama_arch}


def port(model, checkpoint: dict[str, np.ndarray], arch: Arch) -> tuple[int, list[str]]:
    """Map HF tensors onto the model's real paths (tied embeddings included)."""
    expected = set(flatten(pure_dict(model)))
    mapped, unknown, dropped = {}, [], []
    for key, value in checkpoint.items():
        target = arch.mapper(key)
        if target is None:
            unknown.append(key)
            continue
        if target not in expected and target in arch.aliases:
            target = arch.aliases[target]
        if target not in expected:
            dropped.append(target)
            continue
        mapped[target] = arch.transform(target, np.asarray(value))
    if dropped:
        print(f"WARNING: checkpoint tensors with no model path: {dropped}")
    missing = expected - set(mapped)
    if missing:
        print(f"WARNING: {len(missing)} model parameters got no checkpoint tensor: {sorted(missing)[:4]}")
    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, nested(mapped))
    nnx.update(model, state)
    return len(mapped), unknown


def cast(model, dtype):
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

    arch_name = str(raw.get("model_type", "gpt2"))
    if arch_name not in ARCHES:
        raise SystemExit(f"port_hf_check.py does not know model_type={arch_name!r}; known: {sorted(ARCHES)}")
    arch = ARCHES[arch_name](raw)
    print(f"arch: {arch_name}, config: {arch.cfg}")
    for warning in arch.warnings:
        print(f"WARNING: {warning}")
    print(f"checkpoint tensors: {len(ckpt)}, tied={raw.get('tie_word_embeddings')}")

    model = arch.build()
    n_mapped, unknown = port(model, ckpt, arch)
    print(f"ported {n_mapped} tensors onto {len(flatten(pure_dict(model)))} parameters, "
          f"{len(unknown)} unknown keys: {unknown[:4]}")

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

    fresh = arch.build()
    loaded, _ = load_model(fresh, out)
    t2 = time.perf_counter()
    print(f"[load]   {t2-t1:.2f}s")
    after = np.asarray(loaded(ids))
    print(f"[after]  max|after-before| = {np.abs(after - logits).max():.3e}")
    np.testing.assert_array_equal(after, logits)
    np.testing.assert_allclose(after, ref_logits, rtol=1e-4, atol=1e-4)
    first, second = arch.embed_paths
    if arch.tied:
        assert getattr(loaded, first.split("/")[0]) is getattr(loaded, second.split("/")[0]), (
            "tied embedding lost by load_model"
        )
    else:
        assert getattr(loaded, first.split("/")[0]) is not getattr(loaded, second.split("/")[0]), (
            "untied lm_head was tied by load_model"
        )
    print("[after]  OK: logits identical and still match torch")

    # --- bf16, the dtype a TPU inference port uses -------------------------
    bf16 = arch.build()
    port(bf16, ckpt, arch)
    cast(bf16, jnp.bfloat16)
    bf16_path = out.replace(".safetensors", "_bf16.safetensors")
    save_model(bf16, bf16_path)
    print(f"[bf16]   saved {os.path.getsize(bf16_path)/1e6:.1f} MB")
    # A TPU inference port declares bf16 in the model itself, so build the
    # fresh model in bf16 (nnx_save adopts the model's dtype on load).
    fresh = cast(arch.build(), jnp.bfloat16)
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
