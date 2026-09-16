"""Streaming save/load: same values as the classic path, far less host RAM.

`save_model`/`load_model` default to the streaming path. These tests pin the
format (the official safetensors reader must accept it), the equivalence with
the classic whole-dictionary path, and the abstract-builder entry point.

Run: .venv/bin/python -m pytest tests/test_streaming.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from safetensors import safe_open
from safetensors.numpy import load_file

from nnx_save import load_model, save_model

from models import MLP, DTypeZoo, ScalarState, TiedHead, WithRngs
from util import all_equal, dtype_names, first_mismatches, flatten, pure_dict

pytestmark = pytest.mark.streaming


def stored(value):
    """The representation a checkpoint file holds for one leaf."""
    if isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        return np.asarray(jax.random.key_data(value))
    return np.asarray(value)


@pytest.mark.parametrize(
    "build",
    [
        lambda: MLP(8, 16, 4, rngs=nnx.Rngs(0)),
        lambda: DTypeZoo(rngs=nnx.Rngs(0)),
        lambda: WithRngs(rngs=nnx.Rngs(0)),
        lambda: TiedHead(16, 8, rngs=nnx.Rngs(0)),
        lambda: ScalarState(rngs=nnx.Rngs(0)),
    ],
)
def test_streaming_file_is_readable_by_the_official_loader(tmp_path, build):
    """A hand-written streaming writer must produce real safetensors files."""
    model = build()
    path = tmp_path / "streamed.safetensors"
    saved = save_model(model, str(path))

    with safe_open(str(path), framework="np") as f:
        assert sorted(f.keys()) == sorted(saved)
        for key in f.keys():
            array = f.get_tensor(key)
            assert array.shape == stored(saved[key]).shape, key
            assert array.dtype == stored(saved[key]).dtype, key

    official = load_file(str(path))
    assert sorted(official) == sorted(saved)
    for key, value in official.items():
        np.testing.assert_array_equal(value, stored(saved[key]), err_msg=key)


@pytest.mark.parametrize(
    "build",
    [
        lambda: MLP(8, 16, 4, rngs=nnx.Rngs(0)),
        lambda: DTypeZoo(rngs=nnx.Rngs(0)),
        lambda: WithRngs(rngs=nnx.Rngs(0)),
        lambda: ScalarState(rngs=nnx.Rngs(0)),
    ],
)
def test_stream_and_classic_paths_agree(tmp_path, build):
    """The two writers/readers must be interchangeable."""
    model = build()
    stream_path = tmp_path / "stream.safetensors"
    classic_path = tmp_path / "classic.safetensors"
    save_model(model, str(stream_path), stream=True)
    save_model(model, str(classic_path), stream=False)

    fresh = build()
    _poison(fresh)
    loaded_stream, _ = load_model(fresh, str(stream_path), stream=True)
    fresh2 = build()
    _poison(fresh2)
    loaded_classic, _ = load_model(fresh2, str(classic_path), stream=False)

    assert all_equal(model, loaded_stream), first_mismatches(model, loaded_stream)
    assert all_equal(model, loaded_classic), first_mismatches(model, loaded_classic)
    assert all_equal(loaded_stream, loaded_classic), first_mismatches(loaded_stream, loaded_classic)
    assert dtype_names(loaded_stream) == dtype_names(model)


def test_builder_is_used_instead_of_a_random_model(tmp_path):
    """load_model accepts a builder and builds it with nnx.eval_shape."""
    path = tmp_path / "mlp.safetensors"
    model = MLP(8, 16, 4, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 8))
    expected = np.asarray(model(x))
    save_model(model, str(path))

    loaded, state = load_model(lambda: MLP(8, 16, 4, rngs=nnx.Rngs(1234)), str(path))
    assert all_equal(model, loaded), first_mismatches(model, loaded)
    np.testing.assert_array_equal(np.asarray(loaded(x)), expected)
    leaves = flatten(pure_dict(loaded))
    assert all(isinstance(v, jax.Array) for v in leaves.values()), leaves


def test_builder_path_with_scalars_and_rngs(tmp_path):
    """The abstract path keeps arrays concrete; scalar variables become 0-dim."""
    path = tmp_path / "scalars.safetensors"
    model = ScalarState(rngs=nnx.Rngs(0))
    save_model(model, str(path))
    loaded, _ = load_model(lambda: ScalarState(rngs=nnx.Rngs(9)), str(path))
    # The builder cannot know a python int from a 0-dim array; document it.
    assert int(loaded.step.value) == int(model.step.value)
    assert float(loaded.scale.value) == float(model.scale.value)
    assert bool(loaded.flag.value) == bool(model.flag.value)

    rng_path = tmp_path / "rngs.safetensors"
    rng_model = WithRngs(rngs=nnx.Rngs(0))
    save_model(rng_model, str(rng_path))
    loaded_rng, _ = load_model(lambda: WithRngs(rngs=nnx.Rngs(3)), str(rng_path))
    assert all_equal(rng_model, loaded_rng), first_mismatches(rng_model, loaded_rng)


def test_headers_are_aligned_and_sorted(tmp_path):
    """The safetensors header must be 8-byte aligned and the data contiguous."""
    path = tmp_path / "mlp.safetensors"
    save_model(MLP(8, 16, 4, rngs=nnx.Rngs(0)), str(path))
    raw = path.read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    assert (8 + header_len) % 8 == 0, "header must be padded to 8 bytes"
    import json

    header = json.loads(raw[8 : 8 + header_len])
    offsets = [tuple(v["data_offsets"]) for v in header.values()]
    assert offsets == sorted(offsets), "data offsets must follow the header order"
    end = max(o[1] for o in offsets)
    assert 8 + header_len + end == len(raw), "file size must match the declared offsets"


def test_strict_mode_still_applies_to_streaming(tmp_path):
    """Streaming shares the reporting path: strict must raise on a missing key."""
    from safetensors.numpy import save_file

    path = tmp_path / "broken.safetensors"
    save_model(MLP(8, 16, 4, rngs=nnx.Rngs(0)), str(path))
    tensors = dict(load_file(str(path)))
    tensors.pop("l2/kernel")
    save_file(tensors, str(path))

    with pytest.raises(ValueError, match="not in"):
        load_model(MLP(8, 16, 4, rngs=nnx.Rngs(1)), str(path), strict=True)
    with pytest.warns(UserWarning, match="not in"):
        load_model(MLP(8, 16, 4, rngs=nnx.Rngs(1)), str(path))


def _poison(model):
    """Overwrite float leaves so a no-op load cannot pass, scalars untouched."""

    def poison(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.full_like(x, 7.0)
        return x

    nnx.update(model, jax.tree.map(poison, nnx.state(model)))
