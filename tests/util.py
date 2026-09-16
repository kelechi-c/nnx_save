"""Small helpers shared by the nnx_save verification tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def _comparable(value):
    """numpy view of a leaf; typed PRNG keys need key_data() first."""
    if isinstance(value, jax.Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    ):
        return np.asarray(jax.random.key_data(value))
    return value


def pure_dict(obj) -> dict:
    """``nnx.to_pure_dict`` with a fallback for older/newer flax spellings."""
    state = obj if isinstance(obj, nnx.State) else nnx.state(obj)
    if hasattr(nnx, "to_pure_dict"):
        return nnx.to_pure_dict(state)
    return state.to_pure_dict()


def as_numpy(tree):
    """Convert every array leaf of a nested dict to numpy, keep others as-is."""
    return jax.tree.map(
        lambda x: np.asarray(x) if hasattr(x, "dtype") else x,
        tree,
        is_leaf=lambda x: isinstance(x, np.ndarray),
    )


def flatten(tree, prefix: str = "") -> dict:
    """Flatten a nested dict of arrays; keys joined with '/' (like nnx_save)."""
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(flatten(v, f"{prefix}/{k}" if prefix else str(k)))
    else:
        out[prefix] = tree
    return out


def describe(leaf) -> str:
    kind = type(leaf).__name__
    if hasattr(leaf, "dtype"):
        return f"{kind}(dtype={leaf.dtype}, shape={tuple(leaf.shape)})"
    return f"{kind}({leaf!r})"


def compare(model_a, model_b) -> dict:
    """Per-path comparison of two models' state.

    Returns ``{path: {"a": ..., "b": ..., "equal": bool, "same_dtype": bool}}``
    for every path in either model's state.
    """
    flat_a, flat_b = flatten(pure_dict(model_a)), flatten(pure_dict(model_b))
    report = {}
    for key in sorted(set(flat_a) | set(flat_b)):
        a, b = flat_a.get(key), flat_b.get(key)
        entry = {"a": describe(a) if key in flat_a else None,
                 "b": describe(b) if key in flat_b else None}
        if key in flat_a and key in flat_b:
            same_dtype = getattr(a, "dtype", None) == getattr(b, "dtype", None)
            try:
                equal = bool(
                    np.array_equal(
                        np.asarray(_comparable(a)), np.asarray(_comparable(b))
                    )
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                equal = False
                entry["error"] = repr(exc)
            entry.update(equal=equal, same_dtype=same_dtype)
        report[key] = entry
    return report


def first_mismatches(model_a, model_b, limit: int = 5) -> list[str]:
    lines = []
    for key, entry in compare(model_a, model_b).items():
        if not entry.get("equal", False) or not entry.get("same_dtype", False):
            lines.append(f"{key}: a={entry['a']} b={entry['b']} equal={entry.get('equal')}")
    return lines[:limit]


def all_equal(model_a, model_b) -> bool:
    return all(
        e.get("equal", False) and e.get("same_dtype", False)
        for e in compare(model_a, model_b).values()
    )


def dtype_names(model) -> set[str]:
    """Dtype names of every leaf, normalized ('bfloat16', not a class).

    PRNG keys report the dtype of the uint32 data they are stored as.
    """
    out = set()
    for value in flatten(pure_dict(model)).values():
        if isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            out.add(np.dtype(jax.random.key_data(value).dtype).name)
        elif hasattr(value, "dtype"):
            out.add(np.dtype(value.dtype).name)
        else:
            out.add(np.asarray(value).dtype.name)
    return out


def leaf_array(model, path: str):
    """One leaf as a numpy array, addressed by its '/'-joined path."""
    return np.asarray(flatten(pure_dict(model))[path])


def nested(flat_paths: dict[str, np.ndarray]) -> dict:
    """``{'a/b/c': array}`` -> ``{'a': {'b': {'c': array}}}``."""
    out: dict = {}
    for path, value in flat_paths.items():
        parts = path.split("/")
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def transpose_linear_kernels(path: str, tensor: np.ndarray) -> np.ndarray:
    """PyTorch ``(out, in)`` -> NNX ``(in, out)`` for 2-D linear weights.

    This is the standard port trap: ``nn.Linear`` stores the weight
    transposed relative to ``nnx.Linear``'s kernel, and a square projection
    (q_proj, o_proj, a pooler) has the same shape either way, so only the
    logits can catch it.
    """
    if path.endswith("/kernel") and tensor.ndim == 2:
        return tensor.T
    return tensor


def port_into(model, checkpoint: dict[str, np.ndarray], mapper, *, transform=None):
    """The actual PyTorch -> NNX port: put checkpoint tensors onto the state.

    ``mapper(hf_key)`` returns the NNX path for a checkpoint key (or ``None``
    for keys this model does not have).  ``transform(path, tensor)`` adapts the
    tensor layout and defaults to :func:`transpose_linear_kernels`.

    Returns ``(mapped, dropped)``, where ``dropped`` lists ``(key, target)``
    pairs whose target path has no home in the model -- that is a port bug, so
    the tests assert it is empty rather than warning.
    """
    if transform is None:
        transform = transpose_linear_kernels
    expected = set(flatten(pure_dict(model)))
    mapped, dropped = {}, []
    for key, value in checkpoint.items():
        target = mapper(key)
        if target is None:
            continue
        if target not in expected:
            dropped.append((key, target))
            continue
        mapped[target] = transform(target, np.asarray(value))
    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, nested(mapped))
    nnx.update(model, state)
    return mapped, dropped


def cast_params(model, dtype):
    """Cast every floating parameter of ``model`` to ``dtype``, in place."""
    nnx.update(
        model,
        jax.tree.map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(model),
        ),
    )
    return model
