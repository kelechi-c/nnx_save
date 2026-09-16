"""Failure-mode tests: the checkpoint loader must never fail silently.

Each test asserts the *safe* behaviour. They are the specification for the
protections added on top of the original implementation: a save/load that
cannot be applied exactly either raises (`strict=True`) or warns and leaves the
affected variable alone.

Run: .venv/bin/python -m pytest tests/test_hazards.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from safetensors.numpy import load_file, save_file

from nnx_save import load_model, save_model

from models import Deep, MLP, PythonScalarState, ScalarState
from util import all_equal, dtype_names, first_mismatches, flatten, pure_dict

pytestmark = pytest.mark.hazard


def _rewrite(path, drop=(), extra=None):
    """Copy a safetensors file with keys removed/added (a damaged checkpoint)."""
    tensors = dict(load_file(str(path)))
    for key in drop:
        tensors.pop(key, None)
    if extra:
        tensors.update(extra)
    save_file(tensors, str(path))


# --- values that are not arrays -------------------------------------------


def test_non_array_variables_roundtrip(tmp_path):
    """Scalar ``nnx.Variable`` state (step counters) must survive a round-trip.

    The value *type* matters: a counter that comes back as a 0-dim array
    instead of an int silently changes downstream arithmetic.
    """
    path = tmp_path / "scalars.safetensors"
    model = ScalarState(rngs=nnx.Rngs(0))
    save_model(model, str(path))
    fresh = ScalarState(rngs=nnx.Rngs(1))
    loaded, _ = load_model(fresh, str(path))
    assert all_equal(model, loaded), first_mismatches(model, loaded)
    assert isinstance(loaded.step.value, int), type(loaded.step.value)
    assert isinstance(loaded.scale.value, float), type(loaded.scale.value)
    assert isinstance(loaded.flag.value, bool), type(loaded.flag.value)


def test_string_variables_fail_loudly(tmp_path):
    """A string variable cannot be stored; the error must name the path."""
    model = PythonScalarState(rngs=nnx.Rngs(0))
    with pytest.raises(TypeError, match="name"):
        save_model(model, str(tmp_path / "strings.safetensors"))


def test_separator_in_attribute_names_fails_loudly(tmp_path):
    """HAZARD: a '/' in an attribute name is unrecoverable, so save must refuse.

    Flat safetensors keys join the path with '/', so 'blocks/0' and a nested
    'blocks' -> '0' are indistinguishable in the file.
    """
    model = MLP(4, 8, 2, rngs=nnx.Rngs(0))
    try:
        setattr(model, "blocks/0", nnx.Linear(4, 4, rngs=nnx.Rngs(0)))
    except Exception as exc:  # NNX refuses the name outright: nothing to test
        pytest.skip(f"nnx rejects '/' in attribute names: {exc!r}")

    with pytest.raises(ValueError, match="separator"):
        save_model(model, str(tmp_path / "slash.safetensors"))


# --- checkpoints that do not line up with the model -----------------------


def test_missing_keys_warn_by_default(tmp_path):
    """A checkpoint missing parameters must say so, not load quietly."""
    path = tmp_path / "mlp.safetensors"
    save_model(MLP(8, 16, 4, rngs=nnx.Rngs(0)), str(path))
    _rewrite(path, drop=("l2/kernel",))

    fresh = MLP(8, 16, 4, rngs=nnx.Rngs(1))
    with pytest.warns(UserWarning, match="not in"):
        load_model(fresh, str(path))


def test_missing_keys_raise_in_strict_mode(tmp_path):
    path = tmp_path / "mlp.safetensors"
    save_model(MLP(8, 16, 4, rngs=nnx.Rngs(0)), str(path))
    _rewrite(path, drop=("l2/kernel",))

    with pytest.raises(ValueError, match="not in"):
        load_model(MLP(8, 16, 4, rngs=nnx.Rngs(1)), str(path), strict=True)


def test_incomplete_checkpoint_never_claims_success(tmp_path):
    """HAZARD: overlapping-but-incomplete checkpoints must be reported.

    A checkpoint for a shallower model leaves the deeper model's extra layers
    at their random init, which must not pass silently.
    """
    path = tmp_path / "deep3.safetensors"
    save_model(Deep(8, 3, rngs=nnx.Rngs(0)), str(path))
    deeper = Deep(8, 6, rngs=nnx.Rngs(1))  # layers/0..2 load, 3..5 have no source

    with pytest.warns(UserWarning, match="not in"):
        load_model(deeper, str(path))

    # The opposite direction (a deeper checkpoint in a shallower model) is not
    # an error, but the unused keys are reported.
    path6 = tmp_path / "deep6.safetensors"
    save_model(Deep(8, 6, rngs=nnx.Rngs(0)), str(path6))
    shallow = Deep(8, 3, rngs=nnx.Rngs(1))
    with pytest.warns(UserWarning, match="not used"):
        load_model(shallow, str(path6))


def test_wrong_shape_is_skipped_not_installed(tmp_path):
    """HAZARD: a checkpoint for a differently shaped model must not install.

    The parameter keeps its initialised value instead of silently becoming a
    wrong-shaped array.
    """
    path = tmp_path / "wide.safetensors"
    save_model(MLP(8, 32, 4, rngs=nnx.Rngs(0)), str(path))  # l1 kernel is (8, 32)
    fresh = MLP(8, 16, 4, rngs=nnx.Rngs(1))  # l1 kernel is (8, 16)
    before = np.asarray(fresh.l1.kernel.value)

    with pytest.warns(UserWarning, match="shape"):
        loaded, _ = load_model(fresh, str(path))
    assert np.asarray(loaded.l1.kernel.value).shape == (8, 16)
    np.testing.assert_array_equal(np.asarray(loaded.l1.kernel.value), before)


def test_wrong_shape_raises_in_strict_mode(tmp_path):
    path = tmp_path / "wide.safetensors"
    save_model(MLP(8, 32, 4, rngs=nnx.Rngs(0)), str(path))
    with pytest.raises(ValueError, match="shape"):
        load_model(MLP(8, 16, 4, rngs=nnx.Rngs(1)), str(path), strict=True)


def test_dtype_mismatch_does_not_change_the_model_dtype(tmp_path):
    """HAZARD: an f32 checkpoint must not silently un-bfloat16 a TPU model.

    The values are cast to the dtype the model declares for that parameter.
    """
    path = tmp_path / "f32.safetensors"
    save_model(MLP(8, 16, 4, rngs=nnx.Rngs(0)), str(path))

    fresh = MLP(8, 16, 4, rngs=nnx.Rngs(1))
    nnx.update(
        fresh,
        jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(fresh),
        ),
    )
    with pytest.warns(UserWarning, match="cast"):
        loaded, _ = load_model(fresh, str(path))
    assert dtype_names(loaded) == {"bfloat16"}, f"load changed parameter dtype: {dtype_names(loaded)}"


def test_extra_keys_warn(tmp_path):
    """Checkpoint keys the model does not use are a porting smell: warn."""
    path = tmp_path / "mlp.safetensors"
    save_model(MLP(8, 16, 4, rngs=nnx.Rngs(0)), str(path))
    _rewrite(path, extra={"unused/kernel": np.ones((2, 2), np.float32)})

    with pytest.warns(UserWarning, match="not used"):
        load_model(MLP(8, 16, 4, rngs=nnx.Rngs(1)), str(path))


# --- placement and argument handling --------------------------------------


def test_loaded_arrays_stay_on_accelerator(tmp_path):
    """HAZARD: a load must not move every parameter to host numpy.

    On TPU/GPU, host numpy parameters mean host<->device copies on every use
    and a single-device (unsharded) compilation.
    """
    path = tmp_path / "mlp.safetensors"
    save_model(MLP(8, 16, 4, rngs=nnx.Rngs(0)), str(path))
    loaded, _ = load_model(MLP(8, 16, 4, rngs=nnx.Rngs(1)), str(path))
    leaves = flatten(pure_dict(loaded))
    host_only = {k: type(v).__name__ for k, v in leaves.items() if isinstance(v, np.ndarray)}
    assert not host_only, f"loaded parameters are host numpy arrays: {host_only}"


def test_pathlib_path_is_accepted(tmp_path):
    """A Path is the natural argument; the API should not demand str()."""
    model = MLP(4, 8, 2, rngs=nnx.Rngs(0))
    path = tmp_path / "pathlib.safetensors"
    save_model(model, path)
    loaded, _ = load_model(model, path)
    assert all_equal(model, loaded)
