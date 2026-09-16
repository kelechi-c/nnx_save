# nnx_save verification suite

Tests for [`nnx_save`](../nnx_save/checkpointer.py) written to answer one
question: **does a Flax NNX model — in particular one ported from PyTorch for
inference — survive a save/load round-trip unchanged?**

## Layout

| File | What it covers |
| --- | --- |
| `test_roundtrip.py` | The README's claim, across model shapes: MLP, `nnx.List` stacks, conv/audio stack, every dtype (bf16/f16/i8/u32/bool), `nnx.Rngs`, BatchNorm running stats, tied embeddings, dotted attribute names |
| `test_hazards.py` | Silent-failure modes. Each test asserts the *safe* behaviour, so a failure is a bug report |
| `test_tpu_style.py` | TPU-style: 8 simulated devices, sharded and replicated parameters, bf16, `nnx.jit` inference after load |
| `test_ported_gpt2.py` | Full PyTorch→NNX port: an HF-named GPT-2 checkpoint is ported into NNX, checked against torch logits, then round-tripped through `nnx_save` in fp32/bf16/fp16 |
| `port_hf_reference.py` | *(needs torch+transformers)* Downloads a real HF checkpoint and dumps torch reference logits |
| `port_hf_check.py` | *(needs jax only)* Ports that real checkpoint into NNX, checks the logits match, then round-trips it through `nnx_save` in fp32 and bf16 |
| `bench_scale.py` | Save/load wall time and peak host RAM for a ~125M-parameter model |
| `models.py`, `torch_ref.py`, `util.py` | Fixtures: NNX models, the torch reference implementation, comparison helpers |

`conftest.py` sets `XLA_FLAGS=--xla_force_host_platform_device_count=8`, so the
sharding tests run on CPU without a TPU.

## Running

```bash
# CPU (this repo's .venv)
.venv/bin/python -m pytest tests/ -q

# On the RTX 3050 (victoria): same suite with the CUDA jax, torch on GPU
NNX_SAVE_TORCH_DEVICE=cuda .venv/bin/python -m pytest tests/ -q

# Real HF checkpoint, two processes: torch side then jax side
/home/tensor/code/ml/hummingbird/.venv/bin/python tests/port_hf_reference.py \
    --model hf-internal-testing/tiny-random-gpt2 --out /tmp/hf_tiny
.venv/bin/python tests/port_hf_check.py --dir /tmp/hf_tiny

# Cost at scale
.venv/bin/python tests/bench_scale.py --params-millions 125
```

Findings and their severity are in [`REPORT.md`](REPORT.md).
