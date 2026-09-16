"""End-to-end PyTorch -> JAX/NNX port, then an nnx_save round-trip.

This is the workflow nnx_save was written for: a checkpoint produced by
PyTorch (HF GPT-2 naming) is ported into an NNX model, and the ported model
has to be saveable and loadable without changing its outputs.

Run locally (CPU):      .venv/bin/python -m pytest tests/test_ported_gpt2.py -q
Run on the 3050 (CUDA): NNX_SAVE_TORCH_DEVICE=cuda .venv/bin/python -m pytest tests/test_ported_gpt2.py -q
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from safetensors.numpy import load_file as load_np
from safetensors.torch import save_file as save_torch

from nnx_save import load_model, save_model

from models import GPT2Config, TinyGPT2
from torch_ref import TorchGPT2, hf_style_state_dict, jax_path_for
from util import all_equal, dtype_names, first_mismatches, flatten, pure_dict

TORCH_DEVICE = torch.device(os.environ.get("NNX_SAVE_TORCH_DEVICE", "cpu"))


def _nested(flat_paths: dict[str, np.ndarray]) -> dict:
    """{'a/b/c': array} -> {'a': {'b': {'c': array}}}"""
    out: dict = {}
    for path, value in flat_paths.items():
        parts = path.split("/")
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def _port(model: TinyGPT2, checkpoint: dict[str, np.ndarray]) -> TinyGPT2:
    """The actual port: rename/reshape HF tensors into the NNX state.

    Note the tied-embedding subtlety: NNX records a shared Module's variable
    under whichever attribute wins graph traversal, so the token embedding can
    live at ``wte/embedding`` or ``lm_head/embedding``. A port must resolve
    against the model's real state, not against a hard-coded path.
    """
    expected = set(flatten(pure_dict(model)))
    mapped, dropped = {}, []
    for key, value in checkpoint.items():
        target = jax_path_for(key, n_embd=model.cfg.n_embd)
        if target is None:
            continue
        if target not in expected and target == "wte/embedding" and "lm_head/embedding" in expected:
            target = "lm_head/embedding"
        if target not in expected:
            dropped.append(target)
            continue
        mapped[target] = value
    assert not dropped, f"checkpoint tensors with no home in the model: {dropped}"
    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, _nested(mapped))
    nnx.update(model, state)
    return model


def _cast_params(model: TinyGPT2, dtype) -> TinyGPT2:
    state = nnx.state(model)
    nnx.update(
        model,
        jax.tree.map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, state
        ),
    )
    return model


@pytest.fixture(scope="module")
def ported(tmp_path_factory):
    """Torch reference model + checkpoint on disk + ported NNX model."""
    cfg = GPT2Config()
    torch.manual_seed(0)
    torch_model = TorchGPT2(cfg).to(TORCH_DEVICE).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device=TORCH_DEVICE)
    with torch.no_grad():
        ref_logits = torch_model(ids).detach().cpu().numpy()

    ckpt_dir = tmp_path_factory.mktemp("port")
    ckpt_path = ckpt_dir / "hf_model.safetensors"
    save_torch(
        {k: v.detach().contiguous() for k, v in hf_style_state_dict(torch_model).items()},
        str(ckpt_path),
    )
    checkpoint = load_np(str(ckpt_path))

    jax_model = TinyGPT2(cfg, tie_embeddings=True, rngs=nnx.Rngs(0))
    _port(jax_model, checkpoint)
    return {
        "cfg": cfg,
        "torch_model": torch_model,
        "ids": ids.detach().cpu().numpy(),
        "ref_logits": ref_logits,
        "ckpt_path": ckpt_path,
        "checkpoint": checkpoint,
        "jax_model": jax_model,
    }


def test_port_matches_torch_before_any_saving(ported):
    """Sanity: the port itself is correct, so later failures are nnx_save's."""
    logits = np.asarray(ported["jax_model"](jnp.asarray(ported["ids"])))
    np.testing.assert_allclose(logits, ported["ref_logits"], rtol=1e-4, atol=1e-4)


def test_ported_checkpoint_covers_every_parameter(ported):
    """A real port must land every checkpoint tensor on a model parameter."""
    leaves = flatten(pure_dict(ported["jax_model"]))
    checked = 0
    for key, tensor in ported["checkpoint"].items():
        target = jax_path_for(key, n_embd=ported["cfg"].n_embd)
        if target is None:
            continue
        if target not in leaves and target == "wte/embedding":
            target = "lm_head/embedding"
        assert target in leaves, f"{key} -> {target} has no home in the model"
        np.testing.assert_array_equal(np.asarray(leaves[target]), tensor, err_msg=f"{key} -> {target}")
        checked += 1
    assert checked == len(ported["checkpoint"]), "some checkpoint tensors were never mapped"


def test_ported_model_roundtrip_preserves_logits(tmp_path, ported):
    """The nnx_save round-trip must not change the ported model's outputs."""
    c = ported
    path = tmp_path / "ported.safetensors"
    x = jnp.asarray(c["ids"])
    before = np.asarray(c["jax_model"](x))

    saved = save_model(c["jax_model"], str(path))
    # The tied token embedding is recorded under one of its two attribute paths.
    assert {"wte/embedding", "lm_head/embedding"} & set(saved), list(saved)[:5]

    fresh = TinyGPT2(c["cfg"], tie_embeddings=True, rngs=nnx.Rngs(1234))
    loaded, _ = load_model(fresh, str(path))

    assert all_equal(c["jax_model"], loaded), first_mismatches(c["jax_model"], loaded)
    after = np.asarray(loaded(x))
    np.testing.assert_array_equal(after, before)
    # ... and still inside the PyTorch tolerance after the round-trip.
    np.testing.assert_allclose(after, c["ref_logits"], rtol=1e-4, atol=1e-4)
    # Tied embeddings must survive (one shared Variable in the ported state).
    assert loaded.lm_head is loaded.wte


def test_ported_bfloat16_model_roundtrip(tmp_path, ported):
    """TPU inference ports ship bf16 weights; dtype must survive saving."""
    c = ported
    model = TinyGPT2(c["cfg"], tie_embeddings=True, rngs=nnx.Rngs(0))
    _port(model, c["checkpoint"])
    _cast_params(model, jnp.bfloat16)
    x = jnp.asarray(c["ids"])

    path = tmp_path / "ported_bf16.safetensors"
    save_model(model, str(path))
    assert dtype_names(model) == {"bfloat16"}, dtype_names(model)

    fresh = _cast_params(
        TinyGPT2(c["cfg"], tie_embeddings=True, rngs=nnx.Rngs(9)), jnp.bfloat16
    )
    loaded, _ = load_model(fresh, str(path))
    assert dtype_names(loaded) == {"bfloat16"}, f"dtype changed on load: {dtype_names(loaded)}"
    np.testing.assert_array_equal(np.asarray(loaded(x)), np.asarray(model(x)))


def test_float16_port_roundtrip(tmp_path, ported):
    """fp16 is the other common inference dtype."""
    c = ported
    model = TinyGPT2(c["cfg"], tie_embeddings=True, rngs=nnx.Rngs(0))
    _port(model, c["checkpoint"])
    _cast_params(model, jnp.float16)
    x = jnp.asarray(c["ids"])
    path = tmp_path / "ported_fp16.safetensors"
    save_model(model, str(path))
    fresh = _cast_params(
        TinyGPT2(c["cfg"], tie_embeddings=True, rngs=nnx.Rngs(5)), jnp.float16
    )
    loaded, _ = load_model(fresh, str(path))
    assert dtype_names(loaded) == {"float16"}, f"dtype changed on load: {dtype_names(loaded)}"
    np.testing.assert_array_equal(np.asarray(loaded(x)), np.asarray(model(x)))
