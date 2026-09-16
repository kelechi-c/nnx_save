"""Peak host RAM of each load strategy, measured in isolated processes.

`ru_maxrss` is a process high-water mark, so every variant runs in its own
child process and the driver compares peaks. The variants isolate one effect
each:

  current          load_file -> nested dict -> jax copy -> replace -> merge
  current_meta     same, but the model starts as nnx.eval_shape (no random init)
  stream_mmap      safe_open(mmap) + one tensor at a time
  stream_pread     safe_open(pread) + one tensor at a time
  stream_meta      meta-init model + pread streaming (the proposed default)

Run: .venv/bin/python tests/bench_memory_attribution.py --params-millions 125
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnx_save import save_model  # noqa: E402
from nnx_save.checkpointer import (  # noqa: E402
    _leaf_from_numpy,
    _match_to_model,
    _nest_loaded,
    _to_pure_dict,
)

from bench_scale import ParamFarm  # noqa: E402


class ShapeFarm(nnx.Module):
    """One parameter per tensor of a real checkpoint's header.

    Used with `--shapes-from` so the memory numbers follow the real tensor-size
    distribution (many small tensors, a few large ones) instead of a uniform
    stack of square matrices.
    """

    def __init__(self, shapes, dtype):
        self.params = nnx.List([nnx.Param(jnp.zeros(tuple(s), dtype)) for s in shapes])


def load_shapes(path, dtype, build=True):
    """(model or None, parameter count) for a real checkpoint's header shapes.

    Counting must not build the model: on a 2.27 GB checkpoint that would
    allocate the whole thing just to add up sizes, and the freed memory shows up
    in the process high-water mark.
    """
    import json

    with open(path) as f:
        raw = json.load(f)
    shapes = [shape for _dtype, shape in raw.values()]
    total = sum(int(np.prod(s)) for s in shapes)
    return (ShapeFarm(shapes, dtype) if build else None), total

# Only the public API is exercised, so these are the numbers a caller sees.
VARIANTS = ("classic_random", "stream_random", "stream_meta")


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    return 0.0


def build_model(n_layers, dim, dtype, *, rngs_seed=1, meta=False, shapes_from=None):
    def build():
        if shapes_from:
            return load_shapes(shapes_from, dtype)[0]
        return ParamFarm(n_layers, dim, dtype, rngs=nnx.Rngs(rngs_seed))

    return nnx.eval_shape(build) if meta else build()


def run_variant(name: str, path: str, n_layers: int, dim: int, dtype, json_out: str | None,
                shapes_from: str | None = None):
    """Each variant drives nnx_save's public API only."""
    from nnx_save import load_model

    trace = []

    def mark(label):
        trace.append((label, round(rss_mb(), 1), round(peak_mb(), 1)))

    meta = name.endswith("meta")
    stream = name.startswith("stream")
    mark("start")
    target = (
        (lambda: build_model(n_layers, dim, dtype, shapes_from=shapes_from))
        if meta
        else build_model(n_layers, dim, dtype, shapes_from=shapes_from)
    )
    mark("model built" + (" (abstract)" if meta else " (random)"))

    t0 = time.perf_counter()
    loaded, _ = load_model(target, path, stream=stream)
    elapsed = time.perf_counter() - t0
    mark("loaded")

    # Light verification that must not allocate a second full copy: compare two
    # tensors read one at a time (a full re-read here would pollute the peak).
    from safetensors import safe_open

    from util import flatten, pure_dict

    by_name = flatten(pure_dict(loaded))  # keys are already "a/b/c" strings
    with safe_open(path, framework="np", backend="pread") as f:
        keys = list(f.keys())
        probes = [keys[0], keys[len(keys) // 2], keys[-1]]
        ok = all(
            key in by_name
            and bool(jnp.array_equal(by_name[key], jnp.asarray(f.get_tensor(key))))
            for key in probes
        )
    mark("loaded + spot-checked")
    result = {
        "variant": name,
        "peak_mb": round(peak_mb(), 1),
        "final_rss_mb": round(rss_mb(), 1),
        "seconds": round(elapsed, 2),
        "values_ok": ok,
        "trace": trace,
    }
    if json_out:
        with open(json_out, "w") as f:
            json.dump(result, f)
    else:
        print(json.dumps(result, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-millions", type=float, default=125)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--variant", choices=VARIANTS, default=None)
    ap.add_argument("--shapes-from", default=None,
                    help="JSON of {name: [dtype, shape]} from a real checkpoint header")
    ap.add_argument("--only", nargs="*", choices=VARIANTS, default=None,
                    help="driver mode: measure only these variants")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    n_layers = max(1, round(args.params_millions * 1e6 / args.dim**2))
    if args.shapes_from:
        n_params = load_shapes(args.shapes_from, dtype, build=False)[1]
        model_mb = n_params * jnp.dtype(dtype).itemsize / 1e6
        path = "/tmp/attrib_shapes.safetensors"
    else:
        model_mb = n_layers * args.dim * args.dim * jnp.dtype(dtype).itemsize / 1e6
        path = f"/tmp/attrib_{args.dtype}.safetensors"

    if args.variant:  # child process: measure one variant
        if not os.path.exists(path):
            save_model(build_model(n_layers, args.dim, dtype, shapes_from=args.shapes_from), path)
        run_variant(args.variant, path, n_layers, args.dim, dtype, args.json_out,
                    shapes_from=args.shapes_from)
        return

    # driver: save once (streaming writer), then measure each variant alone
    save_model(build_model(n_layers, args.dim, dtype, shapes_from=args.shapes_from), path, stream=True)
    gc.collect()
    label = (
        f"{len(json.load(open(args.shapes_from)))} tensors from {args.shapes_from}"
        if args.shapes_from
        else f"{n_layers} x ({args.dim}, {args.dim})"
    )
    print(f"model: {label} {args.dtype} = {model_mb:.0f} MB\n")
    print(f"{'variant':<16} {'peak RSS':>10} {'x model':>8} {'load s':>8}  values")
    print("-" * 54)
    for variant in (args.only or VARIANTS):
        out = f"/tmp/attrib_{variant}.json"
        subprocess.run(
            [sys.executable, __file__, "--params-millions", str(args.params_millions),
             "--dim", str(args.dim), "--dtype", args.dtype, "--variant", variant,
             "--json-out", out]
            + (["--shapes-from", args.shapes_from] if args.shapes_from else []),
            check=True, capture_output=True,
        )
        with open(out) as f:
            r = json.load(f)
        print(
            f"{r['variant']:<16} {r['peak_mb']:9.0f}M {r['peak_mb'] / model_mb:7.2f}x "
            f"{r['seconds']:7.2f}s  {'ok' if r['values_ok'] else 'MISMATCH'}"
        )
    print("\nper-phase RSS (MB):")
    for variant in (args.only or ("classic_random", "stream_meta")):
        with open(f"/tmp/attrib_{variant}.json") as f:
            r = json.load(f)
        print(f"  {variant}: " + " -> ".join(f"{lbl} {rss:.0f}" for lbl, rss, _ in r["trace"]))


if __name__ == "__main__":
    main()
