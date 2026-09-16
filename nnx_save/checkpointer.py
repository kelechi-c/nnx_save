"""Save and load Flax NNX model state as a single safetensors file.

Two paths are provided:

* the classic path (`stream=False`) converts every leaf to numpy up front and
  hands the whole dictionary to ``safetensors.numpy``;
* the streaming path (default) writes/reads one tensor at a time and places
  loaded values straight onto their target device and sharding, so the host
  never holds a second full copy of the model.

See ``readme.md`` and ``tests/REPORT.md`` for the measured memory difference.
"""

import json
import struct
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file as load_flax, save_file as save_flax
from flax import nnx

_PRNG_KEY_DTYPE = getattr(jax.dtypes, "prng_key", None)

# safetensors dtype names, as written into the file header.
_SAFETENSORS_DTYPES = {
    "bool": "BOOL",
    "uint8": "U8",
    "int8": "I8",
    "int16": "I16",
    "uint16": "U16",
    "float16": "F16",
    "bfloat16": "BF16",
    "int32": "I32",
    "uint32": "U32",
    "float32": "F32",
    "float64": "F64",
    "int64": "I64",
    "uint64": "U64",
    "complex64": "C64",
}


def _is_prng_key(value):
    """True for typed PRNG keys (jax.random.key), which numpy cannot hold."""
    dtype = getattr(value, "dtype", None)
    return (
        _PRNG_KEY_DTYPE is not None
        and dtype is not None
        and jax.dtypes.issubdtype(dtype, _PRNG_KEY_DTYPE)
    )


def _to_pure_dict(state):
    """`nnx.to_pure_dict(state)`, tolerating the older `State.to_pure_dict`."""
    if hasattr(nnx, "to_pure_dict"):
        return nnx.to_pure_dict(state)
    return state.to_pure_dict()


def _leaf_name(path):
    """Human-readable path for messages (tuples from load, str keys from save)."""
    if isinstance(path, (tuple, list)):
        return "/".join(str(p) for p in path)
    return str(path)


def _preview(paths, limit=5):
    shown = ", ".join(_leaf_name(p) for p in paths[:limit])
    return shown + (f" (+{len(paths) - limit} more)" if len(paths) > limit else "")


def flatten_dict(d, parent_key=(), sep="/"):
    """Flatten a nested dictionary; keys become tuples."""
    items = {}
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    
    return items


def unflatten_dict(flat, sep="/"):
    """Convert a flattened dictionary with tuple keys back to nested form."""
    nested = {}
    for key_tuple, v in flat.items():
        current = nested
        for part in key_tuple[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[key_tuple[-1]] = v
    return nested


def _nest_loaded(loaded_state):
    """Turn flat safetensors keys ('a/b/c') back into a nested dictionary."""
    nested = {}
    for key, value in loaded_state.items():
        parts = key.split("/")
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------


def _leaf_to_numpy(value, key):
    """safetensors only accepts real numpy arrays, not jax arrays."""
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"nnx_save cannot store {_leaf_name(key)}={value!r}: safetensors has no "
            "string dtype. Keep non-numeric values out of the saved state."
        )
    if _is_prng_key(value):
        # safetensors has no key<fry> dtype; store the underlying uint32 bits.
        return np.asarray(jax.random.key_data(value))
    return np.asarray(value)


def _buffer_bytes(array) -> bytes:
    """Bytes of a host array without a needless copy when the buffer allows it."""
    try:
        # Zero-copy view for the common dtypes.
        return memoryview(array).cast("B")
    except (TypeError, ValueError):
        # ml_dtypes (bfloat16) exports a buffer format Python's memoryview
        # refuses; one copy of one tensor is still bounded.
        return array.tobytes(order="C")


def _stored_dtype_shape(value, key):
    """Header entry (dtype name, shape) for a leaf, without materialising it."""
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"nnx_save cannot store {_leaf_name(key)}={value!r}: safetensors has no "
            "string dtype. Keep non-numeric values out of the saved state."
        )
    if _is_prng_key(value):
        value = jax.random.key_data(value)
    dtype = np.dtype(getattr(value, "dtype", np.asarray(value).dtype))
    shape = tuple(getattr(value, "shape", np.shape(value)))
    try:
        name = _SAFETENSORS_DTYPES[dtype.name]
    except KeyError:  # pragma: no cover - unreachable for jax/numpy dtypes
        raise TypeError(f"nnx_save cannot store {_leaf_name(key)}: unsupported dtype {dtype}")
    return name, shape, dtype


def _write_header(f, specs):
    """Write the 8-byte length + JSON header; return nothing (file stays open)."""
    header = {}
    offset = 0
    for key, dtype_name, shape, itemsize in specs:
        nbytes = int(np.prod(shape, dtype=np.int64)) * itemsize if shape else itemsize
        header[key] = {
            "dtype": dtype_name,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((8 - (len(payload) + 8) % 8) % 8)
    f.write(struct.pack("<Q", len(payload)))
    f.write(payload)


def _save_streaming(model_file, flat_dict):
    """Write flat_dict one tensor at a time, so only one host copy is live."""
    keys = sorted(flat_dict)
    specs = []
    for key in keys:
        dtype_name, shape, dtype = _stored_dtype_shape(flat_dict[key], key)
        specs.append((key, dtype_name, shape, dtype.itemsize))

    with open(model_file, "wb") as f:
        _write_header(f, specs)
        for key, dtype_name, shape, itemsize in specs:
            array = _leaf_to_numpy(flat_dict[key], key)
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)
            f.write(_buffer_bytes(array))
            del array


def _flatten_for_save(state, sep="/"):
    """Turn pure state into flat 'a/b/c' keys, rejecting '/' inside a name."""

    def flatten(d, parent_key=""):
        items = []
        for k, v in d.items():
            if isinstance(k, str) and sep in k:
                raise ValueError(
                    f"nnx_save cannot store the attribute {k!r}: '/' in a name "
                    "collides with the checkpoint key separator"
                )
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten(_to_pure_dict(state))


def save_model(model, model_file='./model.safetensors', stream=True):
    """Write the model's state to `model_file`.

    `stream=True` (the default) writes tensor by tensor straight from the
    device, so peak host memory is one tensor rather than the whole model;
    `stream=False` keeps the original whole-dictionary behaviour.
    """
    flat_dict = _flatten_for_save(nnx.state(model))

    if stream:
        _save_streaming(model_file, flat_dict)
    else:
        tensor_dict = {k: _leaf_to_numpy(v, k) for k, v in flat_dict.items()}
        save_flax(tensor_dict, model_file)

    return flat_dict


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _check_addressable(sharding, path):
    """Refuse a sharding this process cannot address.

    `jax.device_put(full_array, global_sharding)` on a pod requires every host
    to hold the whole array (jax asserts the inputs are equal across hosts and
    defers the shard), so a single-file load cannot produce a multi-host
    sharding. `load_sharded` reads only this process's bytes instead.
    """
    if sharding is None or getattr(sharding, "is_fully_addressable", True):
        return
    raise ValueError(
        f"nnx_save: {_leaf_name(path)} targets a sharding this process cannot address "
        f"({sharding}). A single-file load reads whole tensors on one host; for a "
        "multi-host checkpoint use nnx_save.load_sharded, or load into a model whose "
        "parameters are sharded only across this process's devices."
    )


def _leaf_from_numpy(value, existing, path, report):
    """Rebuild one variable value from a numpy leaf (classic path)."""
    array = jnp.asarray(value)

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

    sharding = getattr(existing, "sharding", None)
    _check_addressable(sharding, path)
    if sharding is not None:
        # Keep the target model's layout: sharded stays sharded, replicated
        # stays replicated.
        return jax.device_put(array, sharding)
    return array


def _leaf_from_host(value, existing, path, report):
    """Rebuild one variable value from a host numpy leaf (streaming path).

    Casting happens in numpy before the transfer, and `jax.device_put` is given
    the numpy buffer directly, so no intermediate jax host array is allocated.
    """
    if _is_prng_key(existing):
        return jax.random.wrap_key_data(
            jnp.asarray(value).astype(jnp.uint32), impl=getattr(existing, "impl", None)
        )

    if isinstance(existing, (bool, int, float)) and value.ndim == 0:
        return type(existing)(value.item())

    if hasattr(existing, "shape") and tuple(existing.shape) != tuple(value.shape):
        report["wrong_shape"].append((path, tuple(value.shape), tuple(existing.shape)))
        return None

    if hasattr(existing, "dtype") and np.dtype(existing.dtype) != value.dtype:
        report["cast"].append((path, str(value.dtype), str(np.dtype(existing.dtype))))
        value = value.astype(np.dtype(existing.dtype), copy=False)

    sharding = getattr(existing, "sharding", None)
    _check_addressable(sharding, path)
    if sharding is not None:
        return jax.device_put(value, sharding)
    return jnp.asarray(value)


def _match_to_model(expected, loaded, report, path=()):
    """Collect the saved values that line up with the model's own state.

    The walk is driven by the *model's* structure, not by the checkpoint's, so
    keys that are not strings still line up: `nnx.List` indices are integers
    while safetensors keys are always strings.
    """
    matched = {}
    expected_str_keys = {str(k) for k in expected}
    for loaded_key in loaded:
        if str(loaded_key) not in expected_str_keys:
            report["extra"].append(path + (loaded_key,))

    for key, expected_value in expected.items():
        child_path = path + (key,)
        loaded_value = loaded.get(key, loaded.get(str(key)))
        if loaded_value is None:
            report["missing"].append(child_path)
            continue

        if isinstance(expected_value, dict):
            if not isinstance(loaded_value, dict):
                report["missing"].append(child_path)
                continue
            nested = _match_to_model(expected_value, loaded_value, report, child_path)
            if nested:
                matched[key] = nested
            continue

        converted = _leaf_from_numpy(loaded_value, expected_value, child_path, report)
        if converted is not None:
            matched[key] = converted
    return matched


def _classic_load(model_file, expected, report):
    """Whole-file load: every tensor is materialised on the host at once."""
    loaded_nested = _nest_loaded(load_flax(model_file))
    return _match_to_model(expected, loaded_nested, report)


def _stream_load(model_file, expected, report):
    """Per-tensor load: one host buffer and one transfer at a time."""
    from safetensors import safe_open

    flat_expected = {
        "/".join(str(p) for p in key): (key, value)
        for key, value in flatten_dict(expected).items()
    }

    matched = {}
    seen = set()
    with safe_open(model_file, framework="np", backend="pread") as f:
        for file_key in f.keys():
            entry = flat_expected.get(file_key)
            if entry is None:
                report["extra"].append(tuple(file_key.split("/")))
                continue
            key, existing = entry
            seen.add(file_key)
            value = f.get_tensor(file_key)  # one tensor, read on demand
            converted = _leaf_from_host(value, existing, key, report)
            del value
            if converted is not None:
                matched[key] = converted

    for file_key, (key, _) in flat_expected.items():
        if file_key not in seen:
            report["missing"].append(key)

    return unflatten_dict(matched)


def _emit_report(report, model_file, strict):
    def issue(kind, message):
        if strict:
            raise ValueError(message)
        warnings.warn(f"{message} ({_preview(report[kind])})", UserWarning, stacklevel=3)

    if report["missing"]:
        issue(
            "missing",
            f"nnx_save: {len(report['missing'])} parameter(s) in this model are not in "
            f"{model_file}; they keep their initialised values",
        )
    if report["wrong_shape"]:
        issue(
            "wrong_shape",
            f"nnx_save: {len(report['wrong_shape'])} saved parameter(s) have a different "
            f"shape than this model and were skipped (saved vs model: "
            + ", ".join(f"{_leaf_name(p)} {s} vs {e}" for p, s, e in report["wrong_shape"][:3])
            + ")",
        )
    if report["cast"]:
        issue(
            "cast",
            f"nnx_save: {len(report['cast'])} saved parameter(s) were cast to the model's "
            f"dtype (saved->model: "
            + ", ".join(f"{_leaf_name(p)} {s}->{e}" for p, s, e in report["cast"][:3])
            + ")",
        )
    if report["extra"] and not strict:
        warnings.warn(
            f"nnx_save: {len(report['extra'])} saved parameter(s) are not used by this "
            f"model (check the mapping if this was a port) ({_preview(report['extra'])})",
            UserWarning,
            stacklevel=2,
        )


def load_model(model, model_file="./model.safetensors", strict=False, stream=True):
    """Restore parameters from a safetensors file into `model`.

    Only the state is stored, so `model` must be built with the same structure
    as the model that was saved (the graph definition is not in the file).

    `model` may also be a zero-argument callable returning the module; it is
    then built with `nnx.eval_shape`, so the abstract (zero-byte) skeleton is
    created instead of a randomly initialised copy of every parameter.

    With `stream=True` (the default) tensors are read and placed one at a time,
    adopting each variable's dtype and sharding; the host never holds more than
    one tensor at a time. Anything that cannot be applied safely is reported:
    with `strict=True` it raises, otherwise it warns and the affected variable
    keeps its initialised value.
    """
    if not isinstance(model, nnx.Module) and callable(model):
        # A builder (not a module instance): build the abstract skeleton with
        # nnx.eval_shape so no randomly initialised copy is ever allocated.
        model = nnx.eval_shape(model)

    # Obtain the abstract state from the model.
    graphdef, abstract_state = nnx.split(model)
    expected_state = _to_pure_dict(abstract_state)

    report = {"missing": [], "wrong_shape": [], "cast": [], "extra": []}
    if stream:
        filtered_state = _stream_load(model_file, expected_state, report)
    else:
        filtered_state = _classic_load(model_file, expected_state, report)
    _emit_report(report, model_file, strict)

    # Update the abstract state with the filtered state.
    nnx.replace_by_pure_dict(abstract_state, filtered_state)
    
    # Merge the updated state with the graph definition.
    model = nnx.merge(graphdef, abstract_state)

    return model, nnx.state(model)
