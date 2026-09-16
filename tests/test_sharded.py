"""Sharded (pod-style) checkpoints: one shard file per process.

Uses the 8 simulated CPU devices from conftest. A single process owns all 8
devices, so this verifies the layout, the per-leaf shard slicing and the
assembled values; cross-process coordination (barriers, sidecars) is written
for a real pod but is not exercised here.
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from nnx_save import load_model, load_sharded, save_model, save_sharded

from models import GPT2Config, MLP, TinyGPT2, shard_state
from util import all_equal, dtype_names, first_mismatches, flatten, pure_dict

pytestmark = pytest.mark.sharded


@pytest.fixture(scope="module")
def mesh():
    devices = jax.devices()
    assert len(devices) >= 8, f"expected the 8 simulated devices, got {devices}"
    return Mesh(np.asarray(devices), axis_names=("x",))


def _specs(model):
    """PartitionSpec per leaf: P('x') is sharded, P() is replicated."""
    out = {}
    for key, value in flatten(pure_dict(model)).items():
        spec = getattr(getattr(value, "sharding", None), "spec", None)
        out[key] = None if spec is None else tuple(spec)
    return out


def test_sharded_roundtrip_matches_the_single_file(mesh, tmp_path):
    model = shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(0)), mesh)
    directory = tmp_path / "ckpt"
    manifest = save_sharded(model, str(directory), mesh=mesh)

    single = tmp_path / "single.safetensors"
    save_model(model, str(single), stream=True)
    from_single, _ = load_model(shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(1)), mesh), str(single))
    from_shards, _ = load_sharded(shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(2)), mesh), str(directory))

    assert all_equal(from_single, from_shards), first_mismatches(from_single, from_shards)
    assert all_equal(model, from_shards), first_mismatches(model, from_shards)
    # The layout that matters on a pod: every parameter is sharded again.
    specs = _specs(from_shards)
    assert all(spec == ("x",) for spec in specs.values()), specs
    assert manifest["format"] == "nnx_save.sharded.v1"
    assert manifest["mesh"]["shape"] == [8]


def test_shard_files_and_manifest_are_consistent(mesh, tmp_path):
    model = shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(0)), mesh)
    directory = tmp_path / "ckpt"
    save_sharded(model, str(directory), mesh=mesh)

    with open(directory / "manifest.json") as f:
        manifest = json.load(f)
    shard_file = directory / "shard_00000.bin"
    assert shard_file.exists()
    size = os.path.getsize(shard_file)
    expected = sum(
        int(np.prod(record["shards"]["0"]["shape"])) * np.dtype(record["shards"]["0"]["shape"] and _dtype(record)).itemsize
        for record in manifest["tensors"].values()
    )
    assert size == expected, f"{size} bytes on disk vs {expected} declared"
    for key, record in manifest["tensors"].items():
        shard = record["shards"]["0"]
        assert shard["offset"] + shard["nbytes"] <= size, key
        model_shape = tuple(record["shape"])
        assert int(np.prod(shard["shape"])) * _dtype(record).itemsize == shard["nbytes"], key
        if record["spec"] == [None]:
            assert tuple(shard["shape"]) == model_shape, key


def _dtype(record):
    return np.dtype(
        {
            "F32": "float32", "F64": "float64", "F16": "float16", "BF16": "bfloat16",
            "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
            "U64": "uint64", "U32": "uint32", "U16": "uint16", "U8": "uint8",
            "BOOL": "bool", "C64": "complex64",
        }[record["dtype"]]
    )


def test_replicated_and_sharded_mix(mesh, tmp_path):
    model = MLP(64, 128, 32, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    nnx.update(model, jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P("x"))), state))
    model.l2.bias.value = jax.device_put(model.l2.bias.value, NamedSharding(mesh, P()))

    directory = tmp_path / "ckpt"
    save_sharded(model, str(directory), mesh=mesh)
    loaded, _ = load_sharded(shard_state(MLP(64, 128, 32, rngs=nnx.Rngs(1)), mesh), str(directory))
    assert all_equal(model, loaded), first_mismatches(model, loaded)
    specs = _specs(loaded)
    assert specs["l2/bias"] == (), f"expected a replicated bias, got {specs['l2/bias']}"
    assert specs["l1/kernel"] == ("x",), specs["l1/kernel"]


def test_sharded_builder_and_bf16(mesh, tmp_path):
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
    directory = tmp_path / "ckpt"
    save_sharded(model, str(directory), mesh=mesh)

    def build():
        """A plain bf16 model: the checkpoint's own specs decide the layout."""
        fresh = TinyGPT2(cfg, tie_embeddings=True, rngs=nnx.Rngs(1))
        nnx.update(
            fresh,
            jax.tree.map(
                lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                nnx.state(fresh),
            ),
        )
        return fresh

    loaded, _ = load_sharded(build, str(directory))
    assert dtype_names(loaded) == {"bfloat16"}, dtype_names(loaded)
    assert all_equal(model, loaded), first_mismatches(model, loaded)
    assert loaded.lm_head is loaded.wte


def test_unsharded_model_uses_the_same_layout(tmp_path):
    """With no mesh every process stores the whole state; the format still works."""
    model = MLP(8, 16, 4, rngs=nnx.Rngs(0))
    directory = tmp_path / "ckpt"
    save_sharded(model, str(directory))
    loaded, _ = load_sharded(MLP(8, 16, 4, rngs=nnx.Rngs(1)), str(directory))
    assert all_equal(model, loaded), first_mismatches(model, loaded)
    with open(directory / "manifest.json") as f:
        assert json.load(f)["mesh"] is None


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_sharded(MLP(8, 16, 4, rngs=nnx.Rngs(0)), str(tmp_path / "nope"))


def test_single_file_load_refuses_a_non_local_sharding():
    """A pod-global sharding must fail fast, pointing at load_sharded."""
    from nnx_save.checkpointer import _check_addressable

    class GlobalSharding:
        is_fully_addressable = False

        def __repr__(self):
            return "global-sharding"

    _check_addressable(None, ("w",))  # local by construction
    with pytest.raises(ValueError, match="load_sharded"):
        _check_addressable(GlobalSharding(), ("w",))
    # A named sharding over this process's own devices is accepted.
    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), axis_names=("x",))
    _check_addressable(NamedSharding(mesh, P("x")), ("w",))
