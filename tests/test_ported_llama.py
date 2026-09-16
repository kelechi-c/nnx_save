"""End-to-end PyTorch -> JAX/NNX port of a Llama-style decoder, then round-trips.

Llama is the most common inference port, and the parts that go wrong are not
the ones GPT-2 exercises: RMSNorm (no mean subtraction, no bias), RoPE on q/k,
GQA (4 query heads, 2 KV heads here), SwiGLU, no biases anywhere, and a
separate untied ``lm_head``.

The torch side is plain ``torch.nn`` -- no transformers install is needed --
but it uses the genuine HF ``LlamaForCausalLM`` parameter names and the math of
HF's ``modeling_llama``/``modeling_rope_utils``; the key set is the one a real
``hf-internal-testing/tiny-random-LlamaForCausalLM`` checkpoint has (21
tensors).

Run locally (CPU):      .venv/bin/python -m pytest tests/test_ported_llama.py -q
Run on the 3050 (CUDA): NNX_SAVE_TORCH_DEVICE=cuda .venv/bin/python -m pytest tests/test_ported_llama.py -q
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from safetensors.numpy import load_file as load_np
from safetensors.torch import save_file as save_torch

from nnx_save import load_model, save_model

from models import LlamaConfig, TinyLlama, llama_jax_path_for
from torch_ref import TorchLlama, hf_style_llama_state_dict
from util import (
    all_equal,
    cast_params,
    dtype_names,
    first_mismatches,
    flatten,
    port_into,
    pure_dict,
)

TORCH_DEVICE = torch.device(os.environ.get("NNX_SAVE_TORCH_DEVICE", "cpu"))


def _expected_hf_keys(n_layer: int) -> set[str]:
    """The key set of a real ``LlamaForCausalLM`` checkpoint, verbatim.

    Taken from ``hf-internal-testing/tiny-random-LlamaForCausalLM`` (21
    tensors at 2 layers).  Any Llama checkpoint has exactly these names: no
    biases, no ``rotary_emb`` buffer in the saved file, untied ``lm_head``.
    """
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(n_layer):
        keys |= {
            f"model.layers.{i}.{name}"
            for name in (
                "input_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        }
    return keys


@pytest.fixture(scope="module")
def ported(tmp_path_factory):
    """Torch reference model + a real safetensors checkpoint + the NNX port."""
    cfg = LlamaConfig()
    torch.manual_seed(0)
    torch_model = TorchLlama(cfg).to(TORCH_DEVICE).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device=TORCH_DEVICE)
    with torch.no_grad():
        ref_logits = torch_model(ids).detach().cpu().numpy()

    ckpt_dir = tmp_path_factory.mktemp("port_llama")
    ckpt_path = ckpt_dir / "hf_llama.safetensors"
    save_torch(
        {k: v.detach().contiguous() for k, v in hf_style_llama_state_dict(torch_model).items()},
        str(ckpt_path),
    )
    checkpoint = load_np(str(ckpt_path))

    jax_model = TinyLlama(cfg, rngs=nnx.Rngs(0))
    _, dropped = port_into(jax_model, checkpoint, llama_jax_path_for)
    return {
        "cfg": cfg,
        "torch_model": torch_model,
        "ids": ids.detach().cpu().numpy(),
        "ref_logits": ref_logits,
        "ckpt_path": ckpt_path,
        "checkpoint": checkpoint,
        "jax_model": jax_model,
        "dropped": dropped,
    }


def test_port_matches_torch_before_any_saving(ported):
    """(a) RoPE angles, GQA head order and SwiGLU all have to agree with HF."""
    logits = np.asarray(ported["jax_model"](jnp.asarray(ported["ids"])))
    diff = np.abs(logits - ported["ref_logits"])
    print(f"\n[llama] max|jax-torch| = {diff.max():.3e} (mean {diff.mean():.3e})")
    np.testing.assert_allclose(logits, ported["ref_logits"], rtol=1e-4, atol=1e-4)


def test_checkpoint_uses_real_hf_llama_key_names(ported):
    """The fixture has to be a real Llama checkpoint, not a convenient one."""
    cfg = ported["cfg"]
    assert set(ported["checkpoint"]) == _expected_hf_keys(cfg.num_hidden_layers)
    assert len(ported["checkpoint"]) == 21, len(ported["checkpoint"])


def test_the_port_really_exercises_gqa_rope_and_no_bias(ported):
    """Guards the geometry, so the port tests cannot quietly stop covering it."""
    cfg = ported["cfg"]
    assert cfg.num_key_value_heads < cfg.num_attention_heads, "GQA is the point here"

    leaves = flatten(pure_dict(ported["jax_model"]))
    assert not [p for p in leaves if p.endswith("/bias")], "Llama has no biases anywhere"
    assert leaves["layers/0/self_attn/q_proj/kernel"].shape == (
        cfg.hidden_size,
        cfg.num_attention_heads * cfg.head_dim,
    )
    assert leaves["layers/0/self_attn/k_proj/kernel"].shape == (
        cfg.hidden_size,
        cfg.num_key_value_heads * cfg.head_dim,
    )
    model = ported["jax_model"]
    assert model.lm_head is not model.embed_tokens, "lm_head must be untied"


def test_ported_checkpoint_covers_every_parameter(ported):
    """(b) Every checkpoint tensor lands on a parameter, in the right layout.

    The expected layout is spelled out here independently of the port helper:
    a 2-D torch ``.weight`` is (out, in) and must arrive as the NNX kernel
    (in, out).  A missing transpose of a square projection (q_proj/o_proj)
    shows up here as a value mismatch, not as a shape error.
    """
    leaves = flatten(pure_dict(ported["jax_model"]))
    assert not ported["dropped"], f"checkpoint tensors with no home: {ported['dropped']}"
    checked = 0
    for key, tensor in ported["checkpoint"].items():
        target = llama_jax_path_for(key)
        assert target is not None, f"{key} is not mapped at all"
        assert target in leaves, f"{key} -> {target} has no home in the model"
        expected = tensor.T if target.endswith("/kernel") else tensor
        np.testing.assert_array_equal(np.asarray(leaves[target]), expected, err_msg=f"{key} -> {target}")
        checked += 1
    assert checked == len(ported["checkpoint"]), "some checkpoint tensors were never mapped"
    # ... and the other direction: no parameter is left without a checkpoint
    # tensor (all 21 tensors, nothing initialised randomly).
    assert checked == len(leaves), f"{len(leaves)} parameters but {checked} checkpoint tensors"


def test_ported_model_roundtrip_preserves_logits(tmp_path, ported):
    """(c) fp32: the nnx_save round-trip is bit-identical and still matches torch."""
    c = ported
    path = tmp_path / "ported_llama.safetensors"
    x = jnp.asarray(c["ids"])
    before = np.asarray(c["jax_model"](x))

    saved = save_model(c["jax_model"], str(path))
    # Untied: both matrices are in the file, under their own paths.
    assert "lm_head/kernel" in saved and "embed_tokens/embedding" in saved

    fresh = TinyLlama(c["cfg"], rngs=nnx.Rngs(1234))
    loaded, _ = load_model(fresh, str(path))

    assert all_equal(c["jax_model"], loaded), first_mismatches(c["jax_model"], loaded)
    after = np.asarray(loaded(x))
    np.testing.assert_array_equal(after, before)
    # ... and still inside the PyTorch tolerance after the round-trip.
    np.testing.assert_allclose(after, c["ref_logits"], rtol=1e-4, atol=1e-4)


def test_ported_bfloat16_model_roundtrip(tmp_path, ported):
    """(c) bf16: TPU inference ports ship bf16 weights; dtype must survive."""
    c = ported
    model = TinyLlama(c["cfg"], rngs=nnx.Rngs(0))
    port_into(model, c["checkpoint"], llama_jax_path_for)
    cast_params(model, jnp.bfloat16)
    x = jnp.asarray(c["ids"])

    path = tmp_path / "ported_llama_bf16.safetensors"
    save_model(model, str(path))
    assert dtype_names(model) == {"bfloat16"}, dtype_names(model)

    fresh = cast_params(TinyLlama(c["cfg"], rngs=nnx.Rngs(9)), jnp.bfloat16)
    loaded, _ = load_model(fresh, str(path))
    assert dtype_names(loaded) == {"bfloat16"}, f"dtype changed on load: {dtype_names(loaded)}"

    before = np.asarray(model(x))
    after = np.asarray(loaded(x))
    np.testing.assert_array_equal(after, before)
    # Still the same function it was ported to, to bf16 resolution.  The
    # reference is the fp32 torch model, so this is a real cross-framework
    # check, not just the round-trip identity above.
    diff = np.abs(after - c["ref_logits"])
    rel = diff.max() / max(np.abs(c["ref_logits"]).max(), 1e-6)
    print(f"\n[llama/bf16] max|bf16-fp32 torch| = {diff.max():.3e} (relative {rel:.3e})")
    np.testing.assert_allclose(after, c["ref_logits"], rtol=2e-2, atol=2e-2)


def test_load_model_through_a_builder(tmp_path, ported):
    """(d) A zero-argument builder is built with nnx.eval_shape and filled."""
    c = ported
    path = tmp_path / "ported_llama_builder.safetensors"
    save_model(c["jax_model"], str(path))
    x = jnp.asarray(c["ids"])

    # strict=True: the file has to match the abstract skeleton exactly -- no
    # missing, extra, wrong-shaped or cast tensors.
    loaded, _ = load_model(lambda: TinyLlama(c["cfg"], rngs=nnx.Rngs(7)), str(path), strict=True)

    np.testing.assert_array_equal(np.asarray(loaded(x)), np.asarray(c["jax_model"](x)))
    np.testing.assert_allclose(np.asarray(loaded(x)), c["ref_logits"], rtol=1e-4, atol=1e-4)
    assert loaded.lm_head is not loaded.embed_tokens, "lm_head must stay untied"
