"""Reading one .safetensors file as a sharded model, per-local-shard.

The single-file reader uses the target model's shardings to read only the byte
ranges this process owns, then assembles the global arrays with
`jax.make_array_from_single_device_arrays`. On one process with 8 simulated
devices every shard is local, so this verifies the indexing, slicing and
assembly; the cross-host saving is what it enables.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from nnx_save import load_model, load_sharded, save_model

from util import all_equal, dtype_names, first_mismatches, flatten, pure_dict

pytestmark = pytest.mark.sharded


@pytest.fixture(scope="module")
def mesh():
    devices = jax.devices()
    assert len(devices) >= 8
    return Mesh(np.asarray(devices), axis_names=("x",))


def _sharded_mlp(width=64, hidden=128, out=32, mesh=None):
    model = nnx.Linear(width, hidden, rngs=nnx.Rngs(0))
    model.b = nnx.Linear(hidden, out, rngs=nnx.Rngs(1))
    state = nnx.state(model)
    nnx.update(model, jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P("x"))), state))
    return model


def test_single_file_loads_through_the_sharded_reader(mesh, tmp_path):
    model = _sharded_mlp(mesh=mesh)
    path = tmp_path / "sharded.safetensors"
    save_model(model, str(path), stream=True)

    target = _sharded_mlp(mesh=mesh)
    loaded, state = load_sharded(target, str(path))

    assert all_equal(model, loaded), first_mismatches(model, loaded)
    specs = {
        k: tuple(getattr(getattr(v, "sharding", None), "spec", ()) or ())
        for k, v in flatten(pure_dict(loaded)).items()
    }
    assert all(spec == ("x",) for spec in specs.values()), specs

    x = jnp.ones((2, 64))
    np.testing.assert_array_equal(np.asarray(loaded(x)), np.asarray(model(x)))


def test_single_file_reader_matches_the_unsharded_loader(mesh, tmp_path):
    model = _sharded_mlp(mesh=mesh)
    path = tmp_path / "sharded.safetensors"
    save_model(model, str(path), stream=True)

    plain, _ = load_model(_sharded_mlp(mesh=mesh), str(path))
    sharded, _ = load_sharded(_sharded_mlp(mesh=mesh), str(path))
    assert all_equal(plain, sharded), first_mismatches(plain, sharded)


def test_single_file_reader_reports_damage(tmp_path, mesh):
    from safetensors.numpy import load_file, save_file

    model = _sharded_mlp(mesh=mesh)
    path = tmp_path / "sharded.safetensors"
    save_model(model, str(path), stream=True)
    tensors = dict(load_file(str(path)))
    tensors.pop("b/kernel")
    save_file(tensors, str(path))

    with pytest.raises(ValueError, match="not in"):
        load_sharded(_sharded_mlp(mesh=mesh), str(path), strict=True)
    with pytest.warns(UserWarning, match="not in"):
        load_sharded(_sharded_mlp(mesh=mesh), str(path))


def test_bf16_sharded_model_single_file(mesh, tmp_path):
    model = _sharded_mlp(mesh=mesh)
    nnx.update(
        model,
        jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(model),
        ),
    )
    path = tmp_path / "bf16.safetensors"
    save_model(model, str(path), stream=True)
    target = _sharded_mlp(mesh=mesh)  # the model declares bf16, so it stays bf16
    nnx.update(
        target,
        jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(target),
        ),
    )
    loaded, _ = load_sharded(target, str(path))
    assert dtype_names(loaded) == {"bfloat16"}, dtype_names(loaded)
    assert all_equal(model, loaded), first_mismatches(model, loaded)


def test_single_file_save_refuses_a_non_local_sharding():
    """A pod-global sharding must not be written as if it were whole."""
    from nnx_save.checkpointer import _assert_fully_addressable

    class GlobalSharding:
        is_fully_addressable = False

        def __repr__(self):
            return "global-sharding"

    class FakeParam:
        sharding = GlobalSharding()

    with pytest.raises(ValueError, match="save_sharded"):
        _assert_fully_addressable({"w": FakeParam()}, "ckpt.safetensors")
    _assert_fully_addressable({"w": jnp.zeros(4)}, "ckpt.safetensors")
