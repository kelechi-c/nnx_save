# nnx_save verification report

- Date: 2026-09-16
- Library under test: `/home/tensor/Shared/code/ml/nnx_save` @ `a23d9f3` (plus the fixes below)
- Test suite: [`tests/`](.) — 34 tests, plus two scripts for a real checkpoint port
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
| Hardware | 8 cores, 31 GB RAM, no GPU (the CUDA half ran on `victoria`, an RTX 3050 6 GB, over Tailscale) |

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

| dtype | file | save | load | peak RSS, save | peak RSS, load |
| --- | --- | --- | --- | --- | --- |
| fp32 | 503 MB | 0.63 s | 1.18 s | 1.05x model | **3.91x model** (2.16 GB) |
| bf16 | 252 MB | 0.19 s | 0.62 s | 1.14x model | **4.00x model** (1.20 GB) |

Saving is cheap on CPU because `np.asarray` on a CPU jax array is zero-copy; on an
accelerator it is a genuine device→host transfer, so the save figure there includes one
full host copy.

**Loading costs about 4x the checkpoint size in host RAM** (mmap'd file, numpy views, the
jax copy, and the freshly built model that is being replaced). For the 1.4B-parameter
`medium` checkpoint that is ~5.6 GB of host RAM on top of the model — consistent with the
host-RAM kill that `medium` hit on victoria. Saving in bf16 halves it, and loading into a
bf16 model (F7 makes that automatic) halves it again.

## 6. Remaining limitations

1. **State only.** The graph definition is not stored, so the caller must rebuild the same
   structure first. Since the fixes, a structural mistake shows up as missing/extra-key
   warnings rather than silence.
2. **Sharding is adopted, not stored.** Loading into a model that was built sharded keeps
   that sharding; loading into an unsharded model gives replicated parameters. Shard the
   model (or `jax.device_put` the state) before or after the load, deliberately.
3. **Multi-host / TPU-pod saving is untested.** `np.asarray` gathers the *addressable*
   devices only; on a multi-controller pod each process would write its own copy. A
   per-shard writer is the correct fix and is not implemented.
4. **Strings and `/` in names** are unsupported and now fail loudly.
5. **No atomic write.** A crash mid-`save_file` leaves a partial checkpoint; write to a
   temporary path and `os.replace` if that matters.
6. On flax 0.12 the `variable.value` accessor used in test models is itself deprecated in
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
