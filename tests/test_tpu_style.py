"""TPU-style tests: sharded, bfloat16 parameters on an 8-device mesh.

conftest.py forces XLA to expose 8 host devices, so these run without a TPU.
They model the actual inference port: parameters sharded along a mesh axis,
bf16 weights, and a jitted forward pass after a checkpoint load.

Run: .venv/bin/python -m pytest tests/test_tpu_style.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from nnx_save import load_model, save_model

from models import GPT2Config, MLP, TinyGPT2, shard_state
from util import all_equal, dtype_names, first_mismatches, flatten, pure_dict

pytestmark = pytest.mark.tpu_style


@pytest.fixture(scope="module")
def mesh():
    devices = jax.devices()
    assert len(devices) >= 8, f"expected the 8 simulated devices, got {devices}"
    return Mesh(np.asarray(devices), axis_names=("x",))


def _shard_map(model):
    """{path: device count} for every parameter, to see what is really sharded."""
    return {
        k: len(v.sharding.device_set) if isinstance(v, jax.Array) else None
        for k, v in flatten(pure_dict(model)).items()
    }


def test_parameters_are_actually_sharded(mesh):
    """Test setup check: the model under test is genuinely sharded."""
    model = shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(0)), mesh)
    counts = _shard_map(model)
    assert counts and all(c == 8 for c in counts.values()), counts


def test_sharded_model_roundtrip_values(tmp_path, mesh):
    """Saving a sharded model must gather correctly, not corrupt values."""
    path = tmp_path / "sharded.safetensors"
    model = shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(0)), mesh)
    reference = jax.tree.map(np.asarray, pure_dict(model))

    save_model(model, str(path))
    fresh = shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(1)), mesh)
    loaded, _ = load_model(fresh, str(path))

    got = flatten(pure_dict(loaded))
    for key, expected in flatten(reference).items():
        np.testing.assert_array_equal(np.asarray(got[key]), expected, err_msg=key)


def test_sharding_survives_load(tmp_path, mesh):
    """HAZARD: after load, parameters should still be sharded across devices.

    This is the property a TPU inference port depends on: a model that does
    not fit on one chip must come back sharded, not as host numpy arrays.
    """
    path = tmp_path / "sharded.safetensors"
    model = shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(0)), mesh)
    save_model(model, str(path))

    fresh = shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(1)), mesh)
    loaded, _ = load_model(fresh, str(path))
    counts = _shard_map(loaded)
    assert all(c == 8 for c in counts.values()), f"sharding lost on load: {counts}"


def test_replicated_and_sharded_mix_roundtrip(tmp_path, mesh):
    """Real ports mix replicated (norms) and sharded (matmuls) parameters."""
    path = tmp_path / "mixed.safetensors"
    model = MLP(64, 128, 32, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    sharded = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P("x"))), state
    )
    nnx.update(model, sharded)
    model.l2.bias.value = jax.device_put(model.l2.bias.value, NamedSharding(mesh, P()))

    reference = jax.tree.map(np.asarray, pure_dict(model))
    save_model(model, str(path))
    loaded, _ = load_model(model, str(path))
    for key, expected in flatten(reference).items():
        np.testing.assert_array_equal(np.asarray(flatten(pure_dict(loaded))[key]), expected, err_msg=key)


def test_bf16_sharded_ported_model_roundtrip(tmp_path, mesh):
    """The full shape of the target use-case: bf16 + sharded + ported model."""
    path = tmp_path / "bf16_sharded.safetensors"
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=4)
    model = TinyGPT2(cfg, tie_embeddings=True, rngs=nnx.Rngs(0))
    nnx.update(
        model,
        jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(model),
        ),
    )
    shard_state(model, mesh)
    ids = jnp.arange(16, dtype=jnp.int32)[None]

    save_model(model, str(path))
    fresh = TinyGPT2(cfg, tie_embeddings=True, rngs=nnx.Rngs(1))
    nnx.update(
        fresh,
        jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(fresh),
        ),
    )
    loaded, _ = load_model(fresh, str(path))

    assert dtype_names(loaded) == {"bfloat16"}, f"dtype changed on load: {dtype_names(loaded)}"
    # Sharded and replicated compilations of the same bf16 model may differ in
    # the last f32 accumulation bits, so compare with a bf16-sized tolerance.
    np.testing.assert_allclose(
        np.asarray(loaded(ids)), np.asarray(model(ids)), rtol=1e-3, atol=1e-5
    )


def test_loaded_model_runs_under_nnx_jit(tmp_path):
    """The practical test: can a loaded checkpoint be used for inference?

    A host-numpy state still has to work inside ``nnx.jit`` (or at least fail
    loudly); silently compiling a single-device, host-fed model is the failure
    mode this test is looking for.
    """
    path = tmp_path / "mlp.safetensors"
    model = MLP(64, 128, 32, rngs=nnx.Rngs(0))
    x = jnp.ones((4, 64))
    expected = np.asarray(model(x))
    save_model(model, str(path))

    loaded, _ = load_model(MLP(64, 128, 32, rngs=nnx.Rngs(1)), str(path))

    @nnx.jit
    def forward(m, batch):
        return m(batch)

    got = np.asarray(forward(loaded, x))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
