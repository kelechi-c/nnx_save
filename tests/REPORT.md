# nnx_save verification report

- Date: 2026-09-16
- Library under test: `/home/tensor/Shared/code/ml/nnx_save` @ `a23d9f3` (plus the fixes below)
- Test suite: [`tests/`](.) — round-trip, failure-mode, sharding, streaming, pod-layout and PyTorch-port tests, plus scripts for a real checkpoint port and the memory benchmark
- Verdict in one line: **the published code could not save a single model on the current stack; after four blocking fixes the save/load round-trip is exact — including a real PyTorch→JAX port at bf16 — and the loader now reports anything it cannot apply instead of silently keeping random weights.**

## 1. Environment

| Component | Version |
| --- | --- |
| Python | 3.13.15 |
| jax / jaxlib | 0.11.1 (CPU; 8 host devices simulated for sharding tests) |
| flax | 0.12.9 |
| safetensors | 0.8.0 |
| numpy | 2.5.3 |
| torch (porting reference) | 2.14.0+cpu |
| Hardware | 8 cores, 31 GB RAM, no GPU; `victoria` (RTX 3050 6 GB, over Tailscale) ran the PyTorch reference on its GPU plus the whole JAX suite on CPU |

The library was written in March 2025; this is roughly a year of drift in jax, flax and
safetensors, which is where the failures come from.

## 2. Baseline: what the original code does

```
$ .venv/bin/python -m pytest tests/ -q
27 failed, 1 passed, 2 errors
```

Every failure is the same line:

```
safetensors/numpy.py:22: AttributeError: 'jaxlib._jax.ArrayImpl' object has no attribute 'ctypes'
```

A five-line reproduction, using the committed `checkpointer.py` unchanged:

```python
from flax import nnx
from checkpointer import save_model

model = nnx.Linear(4, 2, rngs=nnx.Rngs(0))
save_model(model, "model.safetensors")
# AttributeError: 'jaxlib._jax.ArrayImpl' object has no attribute 'ctypes'
```

`safetensors.numpy.save_file` reads `tensor.ctypes.data` and `tensor.dtype.byteorder`
directly, so it only accepts real `np.ndarray` objects. `nnx.to_pure_dict` returns **jax**
arrays, so `save_model` dies on its first tensor for every model. Nothing about the
library's design is wrong here — one conversion is missing.

Fix that one line and the next failure is silent and much worse:

```
--- with only the numpy conversion patched in ---
plain Linear restored exactly: True
per-layer weights restored: [False, False, False]
save/load reported no error and no warning: a deep stack loads with random weights
```

A model built with `nnx.List` — the standard way to build a transformer stack in NNX —
saves as `layers/0/kernel`, but `nnx.List` keys are **integers** in the state while
safetensors keys are always strings. The loader intersects checkpoint keys with state keys,
finds no match, and drops every layer without a word. The ported model comes back with
random weights and reports success. That is the failure mode this report cares about most:
it is exactly how a ported checkpoint silently produces garbage on a TPU.

## 3. Fixes applied

Four defects made the library unusable; these are the changes in
[`nnx_save/checkpointer.py`](../nnx_save/checkpointer.py):

| # | Defect | Fix |
| --- | --- | --- |
| F1 | `save_model` raised `AttributeError` for every model (jax arrays are not numpy arrays) | convert each leaf with `np.asarray` before `save_file` |
| F2 | `nnx.state(model).to_pure_dict()` is deprecated on current flax | `nnx.to_pure_dict(state)` when available, old spelling otherwise |
| F3 | `nnx.List` parameters were silently never restored (int keys vs string keys) | match checkpoint values by walking the **model's** structure, so `0` and `"0"` line up |
| F4 | Models holding `nnx.Rngs` (dropout, sampling) could not be saved at all — typed PRNG keys cannot be converted to numpy | store `jax.random.key_data(key)` and rebuild with `jax.random.wrap_key_data` |

Then five silent-failure modes were closed, since a checkpointing library that lies is
worse than one that fails:

| # | Hazard | Behaviour now |
| --- | --- | --- |
| F5 | Checkpoint missing parameters → random weights kept, no message | warning naming the paths; `strict=True` raises |
| F6 | Shape mismatch → wrong-shaped array installed silently | skipped with a warning; `strict=True` raises |
| F7 | f32 checkpoint into a bf16 model → model silently becomes f32 | values cast to the dtype the model declares (PyTorch's `load_state_dict` behaviour), with a warning |
| F8 | Load returned host numpy → parameters off-device, sharding gone | values are `jax.device_put` with the target variable's sharding, so a sharded model stays sharded |
| F9 | `nnx.Variable(0)` came back as a 0-dim array | python scalar types are restored from the model's own variable types |
| F10 | String variables died inside safetensors with a cryptic error | `TypeError` naming the offending path, at save time |
| F11 | `'/'` in an attribute name made the file ambiguous | `ValueError` at save time |
| F12 | Unused checkpoint keys ignored | warning listing them (this is how a wrong port mapping surfaces) |

`load_model(model, path, strict=False)` is API-compatible with the original; `strict=True`
turns every warning into an error. `save_model` still returns the flat dictionary.

## 4. Results after the fixes

```
$ .venv/bin/python -m pytest tests/ -q
34 passed
```

Coverage that matters for the original use case:

| Scenario | Result |
| --- | --- |
| README example (nnx.Linear / MLP), poisoned-then-loaded | exact, bit-identical |
| `nnx.List` stack (6 layers) | exact |
| Conv/GroupNorm audio stack | exact |
| dtypes f32, bf16, f16, i32, i8, u32, bool | exact, dtype preserved |
| `nnx.Rngs` keys + counts | exact |
| BatchNorm running statistics | exact |
| Tied embeddings (one `Param`, two attributes) | stays tied, stored once |
| HF-style dotted attribute names | exact |
| Python scalar variables (int/float/bool) | exact, types restored |
| 8-device sharded params, sharded + replicated mix | values exact, **sharding preserved** |
| bf16 + sharded + ported model | values exact, dtype preserved |
| loaded model under `nnx.jit` | matches the pre-save forward pass |
| damaged checkpoints (missing / wrong shape / wrong dtype / extra keys) | warn or raise, never silent |

The same suite (minus the five torch-dependent port tests) and both scripts were then re-run
**on victoria**, against its own CPU jax: `29 passed`, the real-checkpoint port check passed
again, and the scale benchmark reproduced the numbers below. GPU-side JAX was not exercised
there — see §6.

### Real checkpoint, real port

`hf-internal-testing/tiny-random-gpt2` was downloaded on victoria and its logits computed
with PyTorch (transformers 5.17.0, `torch 2.13.0+cu130`); the NNX side then ported the
checkpoint manually (HF GPT-2 `Conv1D` weights are already `(in, out)`, so kernels map
directly) and ran the round-trip:

```
config: GPT2Config(vocab_size=1000, n_positions=512, n_embd=32, n_layer=5, n_head=4, layer_norm_epsilon=1e-05)
checkpoint tensors: 64, tied=True
ported 64 tensors, 0 unknown keys: []
[port]   max|jax-torch| = 3.787e-06  (mean 6.161e-07)
[port]   OK
[save]   0.5 MB in 0.01s -> /tmp/hf_tiny/nnx_saved.safetensors
[load]   0.33s
[after]  max|after-before| = 0.000e+00
[after]  OK: logits identical and still match torch
[bf16]   saved 0.2 MB
[bf16]   dtypes preserved, max relative deviation from fp32 torch = 3.805e-03
[bf16]   OK
ALL CHECKS PASSED
```

The same port is exercised as a normal test (`tests/test_ported_gpt2.py`) against a
torch reference built locally, in fp32, bf16 and fp16.

## 5. Cost at scale

`tests/bench_scale.py`, 125.8M parameters on CPU (503 MB fp32 / 252 MB bf16):

| machine | dtype | file | save | load | peak RSS, load |
| --- | --- | --- | --- | --- | --- |
| 8-core CPU box | fp32 | 503 MB | 0.63 s | 1.18 s | **3.91x model** (2.16 GB) |
| 8-core CPU box | bf16 | 252 MB | 0.19 s | 0.62 s | **4.00x model** (1.20 GB) |
| victoria | fp32 | 503 MB | 0.25 s | 0.29 s | **4.04x model** (2.22 GB) |
| victoria | bf16 | 252 MB | 0.16 s | 0.17 s | **4.00x model** (1.19 GB) |

Both machines ran JAX on CPU: `np.asarray` on a CPU jax array is zero-copy, so the save
column there is a lower bound — on an accelerator it includes a full device→host copy.

Saving is cheap on CPU because `np.asarray` on a CPU jax array is zero-copy; on an
accelerator it is a genuine device→host transfer, so the save figure there includes one
full host copy.

**Loading costs about 4x the checkpoint size in host RAM** (mmap'd file, numpy views, the
jax copy, and the freshly built model that is being replaced). For the 1.4B-parameter
`medium` checkpoint that is ~5.6 GB of host RAM on top of the model — consistent with the
host-RAM kill that `medium` hit on victoria. Saving in bf16 halves it, and loading into a
bf16 model (F7 makes that automatic) halves it again.

## 6. Memory: where the ~4x came from, and the streaming redesign

### 6.1 Attribution

`tests/bench_memory_attribution.py` measures each strategy in its own process
(`ru_maxrss` is a process high-water mark) and traces `VmRSS` per phase. On a
125.8M-parameter fp32 model (503 MB):

| strategy | peak RSS | x model | load |
| --- | ---: | ---: | ---: |
| `load_file` -> nest -> `jnp.asarray` -> replace -> merge, random model (the original path) | 1680 MB | 3.34x | 1.00 s |
| same, model built with `nnx.eval_shape` | 1148 MB | 2.28x | 1.13 s |
| per-tensor `safe_open` (`mmap`), random model | 1838 MB | 3.65x | 0.53 s |
| per-tensor `safe_open` (`pread`), random model | 1359 MB | 2.70x | 0.42 s |
| per-tensor `pread` + `nnx.eval_shape` (the new default) | 794 MB | 1.58x | 0.51 s |

The phase trace shows the three terms: the randomly initialised model (+533 MB),
`safetensors.numpy.load_file` materialising the whole file into anonymous host
RAM (+480 MB for a 503 MB file — the numpy binding copies even with
`backend="mmap"`, unlike the torch binding), and the jax host copies (+452 MB).
Dropping the random model removes one, reading one tensor at a time removes the
second, and casting on the host before `jax.device_put` keeps the transfer to a
single temporary.

Above the interpreter baseline (~190 MB), the new default needs **~1.2x** the
model in host RAM where the original needed **~3.0x**.

### 6.2 At Stable Audio 3 *small* scale

567.6M parameters fp32 = **2.27 GB** — the real `model.safetensors` of
`stabilityai/stable-audio-3-small-music` (measured from its header on victoria:
684 tensors, all F32; the separate T5Gemma text encoder is another 1.18 GB).
Synthetic parameters of exactly that volume, so no model download is involved:

| load path | 8-core CPU box | victoria (RTX 3050 host, 7.5 GB RAM) |
| --- | ---: | ---: |
| `stream=False`, random model (original) | **7.27 GB peak, 13.3 s** | not runnable (exceeds the box) |
| `stream=True`, random model | 4.71 GB, 2.3 s | 4.29 GB, 2.2 s |
| `stream=True` + builder (new default) | **2.82 GB, 1.7 s** | **2.46 GB, 2.0 s** |

In other words: with the original code, loading the full SA3-small checkpoint
(plus the 1.18 GB text encoder) does not fit on the 7.5 GB host that has been
running SA3 inference; with the streaming path it uses about 2.5 GB and loads
~6x faster because the file is no longer read and copied twice. bf16 halves the
weights again. The same arithmetic on the 9.22 GB `medium` checkpoint (metadata
only, not downloaded) puts the original path near 30 GB of host RAM and the
streaming path near 11 GB.

### 6.3 The API

* `save_model(model, path, stream=True)` — writes the header, then one tensor at
  a time (`save()`/`numpy.save` would add a full `bytes()` copy, and
  `serialize_file` requires every source buffer alive at once).
* `load_model(model_or_builder, path, strict=False, stream=True)` — `stream=False`
  keeps the old whole-dictionary path for comparison; a zero-argument builder is
  built with `nnx.eval_shape`, so no random parameters are allocated.
* `load_model` refuses a sharding this process cannot address, with a message
  pointing at `load_sharded` — because `jax.device_put(full_array, global_sharding)`
  on a pod requires *every* host to hold the whole array.

## 7. TPU-resident checkpoints: `save_sharded` / `load_sharded`

A single file cannot be loaded by a pod without one host gathering everything,
so the library now also writes the layout pod checkpointers use — one shard file
per process plus a manifest:

```
ckpt_dir/
  manifest.json      # per tensor: dtype, global shape, PartitionSpec, per-process offset/shape
  shard_00000.bin    # process 0's local shards, concatenated
  shard_00000.json   # process 0's sidecar (merged into the manifest)
  shard_00001.bin
```

* **save**: each process takes *its own* slice of every global array with
  `jax.experimental.multihost_utils.global_array_to_host_local_array` — never
  `np.asarray(global_array)`, which gathers every host's shards — writes those
  bytes and a sidecar, then process 0 merges the sidecars into the manifest after
  a `sync_global_devices` barrier. Replicated parameters are written once per
  process, sharded ones only as the local slice.
* **load**: each process reads only its own file and assembles global arrays with
  `host_local_array_to_global_array`, per tensor, so the host holds one shard
  buffer at a time. The manifest's `PartitionSpec` (captured at save time)
  decides the layout, so a plain builder can be loaded into a sharded model.

Verified on the 8 simulated CPU devices in `tests/test_sharded.py`: values equal
the single-file path, every parameter comes back sharded, a replicated bias stays
replicated, a bf16 tied-embedding GPT-2 round-trips, and a model with no mesh
uses the same layout. **Not verified: cross-process coordination.** The barriers,
sidecars and per-process files are written for a real pod but no TPU pod was
available; the layout, the slicing and the sharding are exercised, the
multi-process handshake is not.

### 7.1 How the established libraries solve the same problem

(Read from the installed sources — `orbax-checkpoint 0.12.4`, `jax 0.11.1`,
`safetensors 0.8.0` — records in
[`../agent-hub/wiki/nnx-checkpointing-pitfalls.md`](/home/tensor/Shared/code/agent-hub/wiki/nnx-checkpointing-pitfalls.md).)

| technique | who uses it | what it saves |
| --- | --- | --- |
| one tensor at a time (`safe_open` + `get_tensor`), never `load_file` | orbax, HF, this library now | ~1x (the numpy binding copies the whole file) |
| abstract/meta model init (`nnx.eval_shape`, `torch.device("meta")`) | orbax v1 (`ShapeDtypeStruct` restore target), transformers (unconditional since v4.51) | ~1x |
| read only this process's shards, assemble with `make_array_from_single_device_arrays` / `host_local_array_to_global_array` | orbax `ArrayHandler`, this library | scales as 1/n_hosts |
| bound in-flight bytes (`restore_concurrent_bytes`, v1 `MemoryOptions.read_concurrent_bytes`, default 2 GiB / 128 MiB chunks) | orbax | O(budget) instead of O(model) |
| cast on the host before H2D ("avoid 2 copies on device") | orbax `_read_shard` | one device temporary |
| bind the parameter instead of copying into it (`assign=True`, `setattr`) | HF/accelerate/PyTorch | one destination allocation |
| per-process shard directories (`ocdbt.process_N`) | orbax | standard pod layout |
| single-file ranged reads driven by the target sharding | orbax v1 `SafetensorsLayout` | per-host bytes, no cross-process traffic |

Two findings from that pass changed this library's defaults: the numpy
`load_file` binding is *eager* (so per-tensor reads are the whole win, not an
optimisation), and `jax.device_put(full_array, global_sharding)` on a pod demands
the full array on every host (hence the guard in §6.3).

Known gaps against that table: no in-flight byte budget (a 100 GB checkpoint
would stream fine but without a cap), no single-file ranged reader for pods
(orbax v1 `SafetensorsLayout` is the reference implementation: map each local
shard's index domain to byte runs and `pread` exactly those), and no async
overlap of the device-to-host copy with the file write.

## 8. Remaining limitations

1. **State only.** The graph definition is not stored, so the caller must rebuild the same
   structure first. Since the fixes, a structural mistake shows up as missing/extra-key
   warnings rather than silence.
2. **Sharding is adopted, not stored.** Loading into a model that was built sharded keeps
   that sharding; loading into an unsharded model gives replicated parameters. Shard the
   model (or `jax.device_put` the state) before or after the load, deliberately.
3. **Multi-host / TPU-pod saving is untested.** `save_sharded`/`load_sharded` now
   implement the per-process layout and refuse to gather other hosts' shards, and
   `load_model` rejects a non-local sharding — but the cross-process handshake has
   never run on a pod. A single-file pod reader (byte ranges per shard, as in orbax
   v1's `SafetensorsLayout`) is not implemented.
4. **Strings and `/` in names** are unsupported and now fail loudly.
5. **No atomic write.** A crash mid-`save_file` leaves a partial checkpoint; write to a
   temporary path and `os.replace` if that matters.
6. **GPU-side JAX is untested.** On victoria, `jax-cuda13-plugin` installed cleanly but
   never registered a backend (`Backend 'cuda' is not in the list of known backends:
   ['cpu', 'tpu']`), and installing the full `jax[cuda13]` extra would have pulled ~3.5 GB of
   `nvidia-*` wheels over a measured ~1 MB/s link. The GPU was therefore used for the torch
   reference logits, not for a jax run. The save/load code paths it would have exercised
   (`np.asarray` on a device array, `jax.device_put` with a sharding) are backend-agnostic,
   but that is an argument, not a measurement.
7. On flax 0.12 the `variable.value` accessor used in test models is itself deprecated in
   favour of `variable[...]`/`get_value()`; that is NNX drift, not an nnx_save issue.

## 7. Reproducing

```bash
# CPU: whole suite (34 tests, ~70 s)
.venv/bin/python -m pytest tests/ -q

# real HF checkpoint: torch side, then jax side (two processes, two venvs)
/home/tensor/code/ml/hummingbird/.venv/bin/python tests/port_hf_reference.py \
    --model hf-internal-testing/tiny-random-gpt2 --out /tmp/hf_tiny
.venv/bin/python tests/port_hf_check.py --dir /tmp/hf_tiny

# cost at scale
.venv/bin/python tests/bench_scale.py --params-millions 125 --dtype bfloat16
```

See [`tests/README.md`](README.md) for the layout and
[`../agent-hub/tasks/20260916-nnx-save-verification.md`](/home/tensor/Shared/code/agent-hub/tasks/20260916-nnx-save-verification.md)
for the task record.
