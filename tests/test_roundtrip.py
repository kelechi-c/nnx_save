"""Core round-trip tests: things nnx_save claims to do in its README.

Run:  .venv/bin/python -m pytest tests/test_roundtrip.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nnx_save import load_model, save_model

from models import (
    ConvAudio,
    Deep,
    DTypeZoo,
    MLP,
    TiedHead,
    WithBatchNorm,
    WithRngs,
    DotNamed,
)
from util import all_equal, compare, first_mismatches, flatten, pure_dict


def _poison(model):
    """Overwrite every float param so a later load must actually restore them."""
    state = nnx.state(model)
    poisoned = jax.tree.map(
        lambda x: jnp.full_like(x, 7.0) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        state,
    )
    nnx.update(model, poisoned)


def test_mlp_roundtrip_is_exact(tmp_path):
    """README scenario: save an MLP, poison it, load it back exactly."""
    path = tmp_path / "mlp.safetensors"
    model = MLP(8, 16, 4, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 8))
    expected_out = np.asarray(model(x))
    before = {k: np.asarray(v) for k, v in flatten(pure_dict(model)).items()}

    saved = save_model(model, str(path))
    assert path.exists() and path.stat().st_size > 0
    assert set(saved) == {"l1/kernel", "l1/bias", "l2/kernel", "l2/bias"}

    _poison(model)  # so a no-op load cannot pass this test
    loaded, state = load_model(model, str(path))
    assert isinstance(state, nnx.State)
    after = flatten(pure_dict(loaded))
    for key, value in before.items():
        np.testing.assert_array_equal(np.asarray(after[key]), value, err_msg=key)

    # The graphdef/structure is not stored, so the caller re-uses the same
    # model object here; check the forward pass matches the pre-save output.
    np.testing.assert_allclose(np.asarray(loaded(x)), expected_out, rtol=0, atol=0)


def test_load_into_fresh_model(tmp_path):
    """A freshly initialised model must end up with the saved weights."""
    path = tmp_path / "mlp.safetensors"
    original = MLP(8, 16, 4, rngs=nnx.Rngs(0))
    save_model(original, str(path))

    fresh = MLP(8, 16, 4, rngs=nnx.Rngs(1234))  # different init
    assert not all_equal(original, fresh), "test setup: models should differ"
    loaded, state = load_model(fresh, str(path))
    assert all_equal(original, loaded), first_mismatches(original, loaded)
    assert isinstance(state, nnx.State)


def test_load_restores_weights_not_the_callers_init(tmp_path):
    """Saving model A and loading into model B must yield A's weights."""
    path = tmp_path / "a.safetensors"
    a = MLP(8, 16, 4, rngs=nnx.Rngs(0))
    save_model(a, str(path))
    b = MLP(8, 16, 4, rngs=nnx.Rngs(99))
    b, _ = load_model(b, str(path))
    assert all_equal(a, b)


def test_nnx_list_of_layers_roundtrip(tmp_path):
    """nnx.List / nnx.Sequential containers are how ported stacks are built."""
    path = tmp_path / "deep.safetensors"
    model = Deep(8, 6, rngs=nnx.Rngs(0))
    save_model(model, str(path))
    fresh = Deep(8, 6, rngs=nnx.Rngs(7))
    loaded, _ = load_model(fresh, str(path))
    assert all_equal(model, loaded), first_mismatches(model, loaded)


def test_dtype_zoo_roundtrip(tmp_path):
    """bf16/f16/i8/u32/bool params must survive bit-exactly."""
    path = tmp_path / "dtypes.safetensors"
    model = DTypeZoo(rngs=nnx.Rngs(0))
    save_model(model, str(path))
    fresh = DTypeZoo(rngs=nnx.Rngs(1))
    _poison(fresh)
    loaded, _ = load_model(fresh, str(path))
    assert all_equal(model, loaded), first_mismatches(model, loaded)


def test_rngs_state_roundtrip(tmp_path):
    """Models with nnx.Rngs (dropout) expose rng keys in their state."""
    path = tmp_path / "rngs.safetensors"
    model = WithRngs(rngs=nnx.Rngs(0))
    save_model(model, str(path))
    fresh = WithRngs(rngs=nnx.Rngs(3))
    loaded, _ = load_model(fresh, str(path))
    assert all_equal(model, loaded), first_mismatches(model, loaded)


def test_batchnorm_running_stats_roundtrip(tmp_path):
    """BatchStat buffers (running mean/var) must survive."""
    path = tmp_path / "bn.safetensors"
    model = WithBatchNorm(rngs=nnx.Rngs(0))
    model(jnp.ones((4, 4)))  # populate running statistics
    model.eval()
    save_model(model, str(path))
    fresh = WithBatchNorm(rngs=nnx.Rngs(5))
    fresh.eval()
    loaded, _ = load_model(fresh, str(path))
    assert all_equal(model, loaded), first_mismatches(model, loaded)


def test_tied_embeddings_stay_tied(tmp_path):
    """Sharing one Param between two attributes is common for LLM heads."""
    path = tmp_path / "tied.safetensors"
    model = TiedHead(16, 8, rngs=nnx.Rngs(0))
    saved = save_model(model, str(path))
    assert list(saved) == ["embedding"], f"tied param should be stored once: {list(saved)}"
    fresh = TiedHead(16, 8, rngs=nnx.Rngs(4))
    loaded, _ = load_model(fresh, str(path))
    assert loaded.head is loaded.embedding, "weight tying was broken by load_model"
    assert all_equal(model, loaded), first_mismatches(model, loaded)


def test_dotted_attribute_names_roundtrip(tmp_path):
    """Ports often keep HF names verbatim (``blocks.0``); '.' is not the sep."""
    path = tmp_path / "dots.safetensors"
    model = DotNamed(rngs=nnx.Rngs(0))
    saved = save_model(model, str(path))
    assert set(saved) == {"blocks.0/kernel", "blocks.0/bias", "blocks.1/kernel", "blocks.1/bias"}
    fresh = DotNamed(rngs=nnx.Rngs(2))
    loaded, _ = load_model(fresh, str(path))
    assert all_equal(model, loaded), first_mismatches(model, loaded)


def test_conv_audio_model_roundtrip(tmp_path):
    """Audio ports (demucs-like) are conv stacks, not just Linear layers."""
    path = tmp_path / "conv.safetensors"
    model = ConvAudio(2, 8, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 128, 2))
    expected = np.asarray(model(x))
    save_model(model, str(path))
    fresh = ConvAudio(2, 8, rngs=nnx.Rngs(3))
    loaded, _ = load_model(fresh, str(path))
    assert all_equal(model, loaded), first_mismatches(model, loaded)
    np.testing.assert_array_equal(np.asarray(loaded(x)), expected)


def test_save_returns_flat_dict(tmp_path):
    """The README documents save_model's return value."""
    path = tmp_path / "mlp.safetensors"
    model = MLP(4, 8, 2, rngs=nnx.Rngs(0))
    flat = save_model(model, str(path))
    assert isinstance(flat, dict)
    assert all(isinstance(k, str) for k in flat)
    leaves = compare(model, model)
    assert set(flat) == set(leaves)
