"""PyTorch -> JAX/NNX port of a ViT-style encoder, then nnx_save round-trips.

Covers the other half of the parameter-tree space: a patch-embedding
convolution, a class token, learned position embeddings, pre-LN attention
blocks with biases everywhere, an MLP, a final layer norm and a classification
head.  The torch reference is plain ``torch.nn`` built in-process (ViT exists
here to cover the encoder/conv shape of the tree), but it uses the parameter
names of ``transformers.ViTModel`` / ``ViTForImageClassification``, verified
against ``hf-internal-testing/tiny-random-vit`` -- including the fact that the
classification head has no ``vit.pooler.*`` keys.

Run locally (CPU):      .venv/bin/python -m pytest tests/test_ported_vit.py -q
Run on the 3050 (CUDA): NNX_SAVE_TORCH_DEVICE=cuda .venv/bin/python -m pytest tests/test_ported_vit.py -q
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

from models import TinyViT, ViTConfig, vit_jax_path_for
from torch_ref import TorchViT, hf_style_vit_state_dict
from util import (
    all_equal,
    cast_params,
    dtype_names,
    first_mismatches,
    flatten,
    port_into,
    pure_dict,
    transpose_linear_kernels,
)

TORCH_DEVICE = torch.device(os.environ.get("NNX_SAVE_TORCH_DEVICE", "cpu"))


def _expected_hf_keys(n_layer: int) -> set[str]:
    """The key set of a real HF ViT checkpoint (``tiny-random-vit`` names)."""
    keys = {
        "vit.embeddings.cls_token",
        "vit.embeddings.position_embeddings",
        "vit.embeddings.patch_embeddings.projection.weight",
        "vit.embeddings.patch_embeddings.projection.bias",
        "vit.layernorm.weight",
        "vit.layernorm.bias",
        "classifier.weight",
        "classifier.bias",
    }
    for i in range(n_layer):
        keys |= {
            f"vit.encoder.layer.{i}.{name}"
            for name in (
                "layernorm_before.weight",
                "layernorm_before.bias",
                "attention.attention.query.weight",
                "attention.attention.query.bias",
                "attention.attention.key.weight",
                "attention.attention.key.bias",
                "attention.attention.value.weight",
                "attention.attention.value.bias",
                "attention.output.dense.weight",
                "attention.output.dense.bias",
                "layernorm_after.weight",
                "layernorm_after.bias",
                "intermediate.dense.weight",
                "intermediate.dense.bias",
                "output.dense.weight",
                "output.dense.bias",
            )
        }
    return keys


def _vit_transform(path: str, tensor: np.ndarray) -> np.ndarray:
    """HF ViT tensor -> NNX layout.

    Linears are (out, in) in torch and (in, out) in NNX; the patch conv is
    ``nn.Conv2d``'s (out, in, kh, kw) and NNX's kernel is (kh, kw, in, out).
    """
    if path.endswith("/kernel"):
        if tensor.ndim == 4:
            return tensor.transpose(2, 3, 1, 0)
        return transpose_linear_kernels(path, tensor)
    return tensor


@pytest.fixture(scope="module")
def ported(tmp_path_factory):
    """Torch reference model + a real safetensors checkpoint + the NNX port."""
    cfg = ViTConfig()
    torch.manual_seed(0)
    torch_model = TorchViT(cfg).to(TORCH_DEVICE).eval()
    pixels = torch.randn(2, cfg.num_channels, cfg.image_size, cfg.image_size, device=TORCH_DEVICE)
    with torch.no_grad():
        ref_logits = torch_model(pixels).detach().cpu().numpy()

    ckpt_dir = tmp_path_factory.mktemp("port_vit")
    ckpt_path = ckpt_dir / "hf_vit.safetensors"
    save_torch(
        {k: v.detach().contiguous() for k, v in hf_style_vit_state_dict(torch_model).items()},
        str(ckpt_path),
    )
    checkpoint = load_np(str(ckpt_path))

    jax_model = TinyViT(cfg, rngs=nnx.Rngs(0))
    _, dropped = port_into(jax_model, checkpoint, vit_jax_path_for, transform=_vit_transform)
    return {
        "cfg": cfg,
        "torch_model": torch_model,
        # NHWC for jax, NCHW for torch: the same images either way.
        "pixels": pixels.detach().cpu().numpy().transpose(0, 2, 3, 1),
        "ref_logits": ref_logits,
        "ckpt_path": ckpt_path,
        "checkpoint": checkpoint,
        "jax_model": jax_model,
        "dropped": dropped,
    }


def test_port_matches_torch_before_any_saving(ported):
    """(a) Patch conv, class token, position embeddings and pre-LN all agree."""
    logits = np.asarray(ported["jax_model"](jnp.asarray(ported["pixels"])))
    diff = np.abs(logits - ported["ref_logits"])
    print(f"\n[vit] max|jax-torch| = {diff.max():.3e} (mean {diff.mean():.3e})")
    np.testing.assert_allclose(logits, ported["ref_logits"], rtol=1e-4, atol=1e-4)


def test_checkpoint_uses_real_hf_vit_key_names(ported):
    """Names have to be HF's ViT names, pooler-free classification head included."""
    cfg = ported["cfg"]
    assert set(ported["checkpoint"]) == _expected_hf_keys(cfg.num_hidden_layers)
    assert len(ported["checkpoint"]) == 40 == 8 + 16 * cfg.num_hidden_layers


def test_the_port_really_exercises_the_patch_conv(ported):
    """Guards the geometry, so the conv/grid part cannot quietly stop being used."""
    cfg = ported["cfg"]
    leaves = flatten(pure_dict(ported["jax_model"]))
    kernel = leaves["vit/embeddings/patch_embeddings/projection/kernel"]
    # NNX conv kernel (kh, kw, in, out) from HF's (out, in, kh, kw).
    assert kernel.shape == (cfg.patch_size, cfg.patch_size, cfg.num_channels, cfg.hidden_size)
    assert leaves["vit/embeddings/position_embeddings"].shape == (1, cfg.num_patches + 1, cfg.hidden_size)
    assert leaves["vit/embeddings/cls_token"].shape == (1, 1, cfg.hidden_size)
    assert [p for p in leaves if p.endswith("/bias")], "ViT blocks do have biases"
    # The head consumes one pooled token, not the whole grid.
    assert leaves["classifier/kernel"].shape == (cfg.hidden_size, cfg.num_labels)


def test_ported_checkpoint_covers_every_parameter(ported):
    """(b) Every checkpoint tensor lands on a parameter, in the right layout.

    The expected layout is written out here independently of ``_vit_transform``
    used by the port, so a wrong transpose is a value mismatch on a square
    weight rather than a shape error.
    """
    leaves = flatten(pure_dict(ported["jax_model"]))
    assert not ported["dropped"], f"checkpoint tensors with no home: {ported['dropped']}"
    checked = 0
    for key, tensor in ported["checkpoint"].items():
        target = vit_jax_path_for(key)
        assert target is not None, f"{key} is not mapped at all"
        assert target in leaves, f"{key} -> {target} has no home in the model"
        if target.endswith("/kernel"):
            expected = tensor.transpose(2, 3, 1, 0) if tensor.ndim == 4 else tensor.T
        else:
            expected = tensor
        np.testing.assert_array_equal(np.asarray(leaves[target]), expected, err_msg=f"{key} -> {target}")
        checked += 1
    assert checked == len(ported["checkpoint"]), "some checkpoint tensors were never mapped"
    assert checked == len(leaves), f"{len(leaves)} parameters but {checked} checkpoint tensors"


def test_ported_model_roundtrip_preserves_logits(tmp_path, ported):
    """(c) fp32: the nnx_save round-trip is bit-identical and still matches torch."""
    c = ported
    path = tmp_path / "ported_vit.safetensors"
    x = jnp.asarray(c["pixels"])
    before = np.asarray(c["jax_model"](x))

    saved = save_model(c["jax_model"], str(path))
    # Conv kernels are 4-D: safetensors has to keep that shape, not flatten it.
    assert tuple(saved["vit/embeddings/patch_embeddings/projection/kernel"].shape) == (
        c["cfg"].patch_size,
        c["cfg"].patch_size,
        c["cfg"].num_channels,
        c["cfg"].hidden_size,
    )

    fresh = TinyViT(c["cfg"], rngs=nnx.Rngs(1234))
    loaded, _ = load_model(fresh, str(path))

    assert all_equal(c["jax_model"], loaded), first_mismatches(c["jax_model"], loaded)
    after = np.asarray(loaded(x))
    np.testing.assert_array_equal(after, before)
    np.testing.assert_allclose(after, c["ref_logits"], rtol=1e-4, atol=1e-4)


def test_ported_bfloat16_model_roundtrip(tmp_path, ported):
    """(c) bf16: the dtype must survive and the logits must stay identical."""
    c = ported
    model = TinyViT(c["cfg"], rngs=nnx.Rngs(0))
    port_into(model, c["checkpoint"], vit_jax_path_for, transform=_vit_transform)
    cast_params(model, jnp.bfloat16)
    x = jnp.asarray(c["pixels"])

    path = tmp_path / "ported_vit_bf16.safetensors"
    save_model(model, str(path))
    assert dtype_names(model) == {"bfloat16"}, dtype_names(model)

    fresh = cast_params(TinyViT(c["cfg"], rngs=nnx.Rngs(9)), jnp.bfloat16)
    loaded, _ = load_model(fresh, str(path))
    assert dtype_names(loaded) == {"bfloat16"}, f"dtype changed on load: {dtype_names(loaded)}"

    before = np.asarray(model(x))
    after = np.asarray(loaded(x))
    np.testing.assert_array_equal(after, before)
    diff = np.abs(after - c["ref_logits"])
    rel = diff.max() / max(np.abs(c["ref_logits"]).max(), 1e-6)
    print(f"\n[vit/bf16] max|bf16-fp32 torch| = {diff.max():.3e} (relative {rel:.3e})")
    np.testing.assert_allclose(after, c["ref_logits"], rtol=2e-2, atol=2e-2)


def test_load_model_through_a_builder(tmp_path, ported):
    """(d) A zero-argument builder is built with nnx.eval_shape and filled."""
    c = ported
    path = tmp_path / "ported_vit_builder.safetensors"
    save_model(c["jax_model"], str(path))
    x = jnp.asarray(c["pixels"])

    loaded, _ = load_model(lambda: TinyViT(c["cfg"], rngs=nnx.Rngs(7)), str(path), strict=True)

    np.testing.assert_array_equal(np.asarray(loaded(x)), np.asarray(c["jax_model"](x)))
    np.testing.assert_allclose(np.asarray(loaded(x)), c["ref_logits"], rtol=1e-4, atol=1e-4)
