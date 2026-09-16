"""TPU/pod-aware sharded checkpointing for NNX models.

A single-file checkpoint forces one host to hold the whole model: on a TPU pod
every process would have to gather every other process's shards. This module
writes a **directory** instead, in the layout pod checkpointers use:

    <directory>/
      manifest.json      # key -> {dtype, shape, spec, per-process shard placement}
      shard_00000.bin    # process 0's local shards, concatenated
      shard_00000.json   # process 0's sidecar (merged into the manifest)
      shard_00001.bin    # process 1's ...

Every process writes only its own file and reads only its own file, so the host
memory held at any moment is one tensor per process, not the whole model. The
global arrays are assembled with
`jax.experimental.multihost_utils.host_local_array_to_global_array`, which is
the supported way to build a sharded array from per-process buffers; the
inverse (`global_array_to_host_local_array`) is used when saving, because
`np.asarray(global_array)` gathers *all* hosts' shards into one process.

Verified in this repository on 8 simulated CPU devices inside one process,
which exercises the layout and the array plumbing but **not** cross-process
coordination; the barriers, sidecars and per-process files are written for a
real pod but have not been run on one.
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
from flax import nnx

from .checkpointer import (
    _SAFETENSORS_DTYPES,
    _buffer_bytes,
    _emit_report,
    _is_prng_key,
    _leaf_name,
    _stored_dtype_shape,
    _to_pure_dict,
    flatten_dict,
    unflatten_dict,
)

FORMAT = "nnx_save.sharded.v1"
MANIFEST = "manifest.json"
_SINGLE_FILE_SUFFIXES = (".safetensors",)


def _shard_base(directory: str, process_index: int) -> str:
    return os.path.join(directory, f"shard_{process_index:05d}")


def _spec_of(value) -> P:
    """PartitionSpec of an array's sharding, or P() when it is not named."""
    spec = getattr(getattr(value, "sharding", None), "spec", None)
    return P() if spec is None else spec


def _mesh_of(flat_state):
    for value in flat_state.values():
        mesh = getattr(getattr(value, "sharding", None), "mesh", None)
        if mesh is not None:
            return mesh
    return None


def _spec_to_json(spec: P):
    return [None if axis is None else str(axis) for axis in spec]


def _spec_from_json(raw) -> P:
    return P(*[None if axis is None else axis for axis in raw])


def _host_array(value) -> np.ndarray:
    """A C-contiguous host buffer for one local shard."""
    array = value if isinstance(value, np.ndarray) else np.asarray(value)
    if _is_prng_key(array):
        array = np.asarray(jax.random.key_data(array))
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array


def save_sharded(model, directory, mesh=None, barrier=True):
    """Write `model` into `directory`, one shard file per process.

    Returns the manifest on process 0 (None elsewhere). Every process must call
    this with the same model structure and a `directory` on a filesystem they
    all see.
    """
    os.makedirs(directory, exist_ok=True)
    flat_state = {
        "/".join(str(part) for part in key): value
        for key, value in flatten_dict(_to_pure_dict(nnx.state(model))).items()
    }
    process_index = jax.process_index()
    spec_tree = {key: _spec_of(value) for key, value in flat_state.items()}
    if mesh is None:
        mesh = _mesh_of(flat_state)

    entries = {}
    offset = 0
    base = _shard_base(directory, process_index)
    with open(base + ".bin", "wb") as f:
        for key in sorted(flat_state):
            value = flat_state[key]
            spec = spec_tree[key]
            sharding = getattr(value, "sharding", None)
            if mesh is not None and getattr(sharding, "mesh", None) is not None:
                # This process's slice of a global (possibly multi-host) array.
                local = multihost_utils.global_array_to_host_local_array(
                    value, mesh, spec
                )
            else:
                local = value
            array = _host_array(local)
            del local
            f.write(_buffer_bytes(array))
            entries[key] = {
                "dtype": _stored_dtype_shape(array, key)[0],
                "shape": list(array.shape),
                "global_shape": list(flat_state[key].shape),
                "offset": offset,
                "nbytes": int(array.nbytes),
                "spec": _spec_to_json(spec),
            }
            offset += int(array.nbytes)
            del array
    with open(base + ".json", "w") as f:
        json.dump(entries, f, sort_keys=True)

    if barrier and jax.process_count() > 1:
        multihost_utils.sync_global_devices("nnx_save_shards_written")

    manifest_path = os.path.join(directory, MANIFEST)
    if process_index == 0:
        tensors: dict = {}
        for index in range(jax.process_count()):
            sidecar = _shard_base(directory, index) + ".json"
            if not os.path.exists(sidecar):
                raise FileNotFoundError(
                    f"process {index} did not write {sidecar}; every process must call save_sharded"
                )
            with open(sidecar) as f:
                for key, entry in json.load(f).items():
                    record = tensors.setdefault(
                        key,
                        {
                            "dtype": entry["dtype"],
                            "shape": entry["global_shape"],
                            "spec": entry["spec"],
                            "shards": {},
                        },
                    )
                    record["shards"][str(index)] = {
                        "shape": entry["shape"],
                        "file": os.path.basename(_shard_base(directory, index)) + ".bin",
                        "offset": entry["offset"],
                        "nbytes": entry["nbytes"],
                    }
        manifest = {
            "format": FORMAT,
            "process_count": jax.process_count(),
            "mesh": None
            if mesh is None
            else {
                "shape": list(mesh.shape.values()),
                "axis_names": list(mesh.axis_names),
            },
            "tensors": tensors,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
        return manifest


def _place(array, existing, path, report):
    """Check/convert an assembled global array against the model's variable."""
    if _is_prng_key(existing):
        return jax.random.wrap_key_data(
            array.astype(jnp.uint32), impl=getattr(existing, "impl", None)
        )
    if isinstance(existing, (bool, int, float)) and array.ndim == 0:
        return type(existing)(array.item())
    if hasattr(existing, "shape") and tuple(existing.shape) != tuple(array.shape):
        report["wrong_shape"].append((path, tuple(array.shape), tuple(existing.shape)))
        return None
    if hasattr(existing, "dtype") and jnp.dtype(existing.dtype) != jnp.dtype(array.dtype):
        report["cast"].append((path, str(array.dtype), str(existing.dtype)))
        array = array.astype(existing.dtype)
    return array


def _load_single_file_shards(model, path, strict):
    """Load a single .safetensors file, reading only this process's shards.

    This is the "one file, many hosts" case: the target model carries the
    shardings (build it with `nnx.with_partitioning`, or shard it explicitly),
    and for every tensor each local device reads just its own index range with
    safetensors' slice API, then the global array is assembled with
    `jax.make_array_from_single_device_arrays`. No host reads another host's
    bytes and nothing is gathered. Same idea as orbax v1's `SafetensorsLayout`.
    """
    from safetensors import safe_open

    if not isinstance(model, nnx.Module) and callable(model):
        model = nnx.eval_shape(model)

    graphdef, abstract_state = nnx.split(model)
    flat_expected = {
        "/".join(str(p) for p in key): (key, value)
        for key, value in flatten_dict(_to_pure_dict(abstract_state)).items()
    }
    report = {"missing": [], "wrong_shape": [], "cast": [], "extra": []}
    matched = {}
    seen = set()
    local_devices = {str(d): d for d in jax.local_devices()}

    with safe_open(path, framework="np", backend="pread") as f:
        file_keys = set(f.keys())
        for file_key, (key, existing) in flat_expected.items():
            if file_key not in file_keys:
                report["missing"].append(key)
                continue
            seen.add(file_key)
            sharding = getattr(existing, "sharding", None)
            shape = getattr(existing, "shape", None)
            spec = getattr(sharding, "spec", None)
            if sharding is None or spec is None or shape is None:
                # No target sharding: this tensor is read whole.
                value = f.get_tensor(file_key)
                converted = _leaf_from_host(value, existing, key, report)
                del value
                if converted is not None:
                    matched[key] = converted
                continue

            slices = f.get_slice(file_key)
            pieces: dict[tuple, object] = {}
            shards = []
            for device, index in sharding.devices_indices_map(tuple(shape)).items():
                if str(device) not in local_devices:
                    continue
                token = repr(index)
                if token not in pieces:
                    pieces[token] = jax.device_put(slices[index], device)
                shards.append(pieces[token])
            if not shards:
                report["missing"].append(key)
                continue
            assembled = jax.make_array_from_single_device_arrays(
                tuple(shape), sharding, shards
            )
            converted = _place(assembled, existing, key, report)
            del assembled, shards, pieces
            if converted is not None:
                matched[key] = converted

        for file_key in file_keys - seen:
            report["extra"].append(tuple(file_key.split("/")))

    _emit_report(report, path, strict)
    nnx.replace_by_pure_dict(abstract_state, unflatten_dict(matched))
    model = nnx.merge(graphdef, abstract_state)
    return model, nnx.state(model)


def load_sharded(model, directory, mesh=None, strict=False, barrier=True):
    """Read a checkpoint written by `save_sharded`, or a single file.

    `directory` may be the directory layout written by `save_sharded`, or a
    single `.safetensors` file whose tensors are sharded according to the target
    model (each process then reads only its own shard bytes). `mesh` is used for
    the directory layout's global assembly; the single-file path takes its
    shardings from the model itself.
    """
    if os.path.isfile(directory):
        return _load_single_file_shards(model, directory, strict)

    if not isinstance(model, nnx.Module) and callable(model):
        model = nnx.eval_shape(model)

    manifest_path = os.path.join(directory, MANIFEST)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found: save_sharded writes it, and every process must see it"
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"unexpected sharded checkpoint format: {manifest.get('format')!r}")

    process_index = jax.process_index()
    process_count = manifest["process_count"]
    shard_path = _shard_base(directory, process_index) + ".bin"
    if not os.path.exists(shard_path):
        raise FileNotFoundError(
            f"{shard_path} not found: process {process_index} has no shard file for this checkpoint"
        )

    if mesh is None and manifest.get("mesh"):
        shape = tuple(manifest["mesh"]["shape"])
        axis_names = tuple(manifest["mesh"]["axis_names"])
        devices = np.asarray(jax.devices())
        if devices.size == int(np.prod(shape)):
            mesh = Mesh(devices.reshape(shape), axis_names=axis_names)

    graphdef, abstract_state = nnx.split(model)
    flat_expected = {
        "/".join(str(p) for p in key): (key, value)
        for key, value in flatten_dict(_to_pure_dict(abstract_state)).items()
    }
    report = {"missing": [], "wrong_shape": [], "cast": [], "extra": []}

    matched = {}
    seen = set()
    dtype_names = {v: k for k, v in _SAFETENSORS_DTYPES.items()}

    with open(shard_path, "rb") as f:
        for key in sorted(manifest["tensors"]):
            record = manifest["tensors"][key]
            entry = flat_expected.get(key)
            if entry is None:
                report["extra"].append(tuple(key.split("/")))
                continue
            expected_key, existing = entry
            shard_info = record["shards"].get(str(process_index))
            if shard_info is None:
                report["missing"].append(expected_key)
                continue
            seen.add(key)

            f.seek(shard_info["offset"])
            buffer = f.read(shard_info["nbytes"])
            dtype = np.dtype(dtype_names[record["dtype"]])
            local = np.frombuffer(buffer, dtype=dtype).reshape(shard_info["shape"])

            if mesh is None:
                assembled = jnp.asarray(local)
            else:
                assembled = multihost_utils.host_local_array_to_global_array(
                    local, mesh, _spec_from_json(record["spec"])
                )
            del local, buffer
            converted = _place(assembled, existing, expected_key, report)
            del assembled
            if converted is not None:
                matched[expected_key] = converted

    for key, (expected_key, _) in flat_expected.items():
        if key not in seen:
            report["missing"].append(expected_key)

    if barrier and process_count > 1:
        multihost_utils.sync_global_devices("nnx_save_shards_read")

    _emit_report(report, directory, strict)
    nnx.replace_by_pure_dict(abstract_state, unflatten_dict(matched))
    model = nnx.merge(graphdef, abstract_state)
    return model, nnx.state(model)
