import warnings

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file as load_flax, save_file as save_flax
from flax import nnx

_PRNG_KEY_DTYPE = getattr(jax.dtypes, "prng_key", None)


def _is_prng_key(value):
    """True for typed PRNG keys (jax.random.key), which numpy cannot hold."""
    return (
        _PRNG_KEY_DTYPE is not None
        and isinstance(value, jax.Array)
        and jax.dtypes.issubdtype(value.dtype, _PRNG_KEY_DTYPE)
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


def _leaf_to_numpy(value, key):
    """safetensors.numpy only accepts real numpy arrays, not jax arrays."""
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"nnx_save cannot store {_leaf_name(key)}={value!r}: safetensors has no "
            "string dtype. Keep non-numeric values out of the saved state."
        )
    if _is_prng_key(value):
        # safetensors has no key<fry> dtype; store the underlying uint32 bits.
        return np.asarray(jax.random.key_data(value))
    return np.asarray(value)


def _leaf_from_numpy(value, existing, path, report):
    """Rebuild one variable value from the checkpoint.

    The value adopts the dtype and the sharding of the variable it is replacing,
    which is what keeps a sharded bfloat16 inference model sharded after a load.
    A shape that does not match the model is never installed: it is reported
    and skipped.
    """
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
    if sharding is not None:
        # Keep the target model's layout: sharded stays sharded, replicated
        # stays replicated.
        return jax.device_put(array, sharding)
    return array


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


def _report_issue(report, kind, message, strict):
    if strict:
        raise ValueError(message)
    warnings.warn(f"{message} ({_preview(report[kind])})", UserWarning, stacklevel=3)


def load_model(model, model_file="./model.safetensors", strict=False):
    """Restore parameters from a safetensors file into `model`.

    Only the state is stored, so `model` must be built with the same structure
    as the model that was saved (the graph definition is not in the file).

    Values are moved back onto the accelerator and adopt each variable's dtype
    and sharding. Anything that cannot be applied safely is reported: with
    `strict=True` it raises, otherwise it warns and the affected variable keeps
    its initialised value.
    """
    
    # Load the flattened checkpoint dictionary.
    loaded_state = load_flax(model_file)
    loaded_nested = _nest_loaded(loaded_state)

    # Obtain the abstract state from the model.
    graphdef, abstract_state = nnx.split(model)
    # Convert the abstract state to a pure dictionary.
    expected_state = _to_pure_dict(abstract_state)

    report = {"missing": [], "wrong_shape": [], "cast": [], "extra": []}
    filtered_state = _match_to_model(expected_state, loaded_nested, report)

    if report["missing"]:
        _report_issue(
            report,
            "missing",
            f"nnx_save: {len(report['missing'])} parameter(s) in this model are not in "
            f"{model_file}; they keep their initialised values",
            strict,
        )
    if report["wrong_shape"]:
        _report_issue(
            report,
            "wrong_shape",
            f"nnx_save: {len(report['wrong_shape'])} saved parameter(s) have a different "
            f"shape than this model and were skipped (saved vs model: "
            + ", ".join(
                f"{_leaf_name(p)} {s} vs {e}" for p, s, e in report["wrong_shape"][:3]
            )
            + ")",
            strict,
        )
    if report["cast"]:
        _report_issue(
            report,
            "cast",
            f"nnx_save: {len(report['cast'])} saved parameter(s) were cast to the model's "
            f"dtype (saved->model: "
            + ", ".join(f"{_leaf_name(p)} {s}->{e}" for p, s, e in report["cast"][:3])
            + ")",
            strict,
        )
    if report["extra"] and not strict:
        warnings.warn(
            f"nnx_save: {len(report['extra'])} saved parameter(s) are not used by this "
            f"model (check the mapping if this was a port) ({_preview(report['extra'])})",
            UserWarning,
            stacklevel=2,
        )

    # Update the abstract state with the filtered state.
    nnx.replace_by_pure_dict(abstract_state, filtered_state)
    
    # Merge the updated state with the graph definition.
    model = nnx.merge(graphdef, abstract_state)

    return model, nnx.state(model)


def save_model(model, model_file='./model.safetensors'):
    
    def flatten_dict(d, parent_key='', sep='/'):
        items = []
        for k, v in d.items():
            if isinstance(k, str) and sep in k:
                raise ValueError(
                    f"nnx_save cannot store the attribute {k!r}: '/' in a name "
                    "collides with the checkpoint key separator"
                )
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    state = _to_pure_dict(nnx.state(model))
    flat_dict = flatten_dict(state)

    # safetensors.numpy requires numpy arrays; jax arrays (including bfloat16
    # parameters and PRNG keys) have to be materialised on the host first.
    tensor_dict = {k: _leaf_to_numpy(v, k) for k, v in flat_dict.items()}
    save_flax(tensor_dict, model_file)

    return flat_dict
