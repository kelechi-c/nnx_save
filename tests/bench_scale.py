"""Measure nnx_save's time and peak host RAM on a realistically sized model.

The TPU-port failure this is aimed at: a 1.4B-parameter checkpoint that loads
fine on paper but dies on a small host because the state is materialised as
numpy plus a copy of the safetensors bytes.

Run:  .venv/bin/python tests/bench_scale.py --params-millions 125
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnx_save import load_model, save_model  # noqa: E402


class ParamFarm(nnx.Module):
    """A model of the same *weight volume* as a transformer, minus the compute."""

    def __init__(self, n_layers: int, dim: int, dtype, *, rngs: nnx.Rngs):
        self.layers = nnx.List(
            [nnx.Param(jnp.zeros((dim, dim), dtype=dtype)) for _ in range(n_layers)]
        )


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-millions", type=float, default=125)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[args.dtype]
    n_layers = max(1, round(args.params_millions * 1e6 / args.dim**2))
    model_bytes = n_layers * args.dim * args.dim * jnp.dtype(dtype).itemsize
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(f"model: {n_layers} x ({args.dim}, {args.dim}) {args.dtype} = "
          f"{n_layers * args.dim**2 / 1e6:.1f}M params, {model_bytes / 1e6:.1f} MB")

    baseline = peak_mb()
    model = ParamFarm(n_layers, args.dim, dtype, rngs=nnx.Rngs(0))
    after_build = peak_mb()

    out = args.out or f"/tmp/nnx_save_bench_{args.dtype}.safetensors"
    t0 = time.perf_counter()
    save_model(model, out)
    t1 = time.perf_counter()
    after_save = peak_mb()
    size = os.path.getsize(out)

    fresh = ParamFarm(n_layers, args.dim, dtype, rngs=nnx.Rngs(1))
    t2 = time.perf_counter()
    loaded, _ = load_model(fresh, out)
    t3 = time.perf_counter()
    after_load = peak_mb()

    def ratio(x):
        return f"{x / (model_bytes / 1e6):.2f}x"

    print(f"file size      : {size / 1e6:.1f} MB")
    print(f"save time      : {t1 - t0:.2f} s")
    print(f"load time      : {t3 - t2:.2f} s")
    print(f"peak RSS build : {after_build:.0f} MB (baseline {baseline:.0f} MB, {ratio(after_build - baseline)} model)")
    print(f"peak RSS save  : {after_save:.0f} MB (baseline {baseline:.0f} MB, {ratio(after_save - baseline)} model)")
    print(f"peak RSS load  : {after_load:.0f} MB (baseline {baseline:.0f} MB, {ratio(after_load - baseline)} model)")
    leaves = jax.tree.leaves(nnx.state(loaded))
    kinds = {type(x).__name__ for x in leaves}
    print(f"loaded leaf types: {kinds}")
    print(f"total host RAM available: "
          f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES') / 1e6:.0f} MB free")


if __name__ == "__main__":
    main()
