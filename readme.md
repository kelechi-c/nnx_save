## nnx_save
_might think of a better name...no this works_

a simple library for **saving Flax models** (the new **nnx** api specifically), saves it to a single `.safetensors` file, and also load it from the file.

I know it's kinda pointless; bloat, can be left as functions, isn't ok for production, doesn't have a lot of features, etc.
But I don't wanna have to write all that hacked together stuff just to save and load my JAX/Flax models to/from a safetensor file. I did it for myself, esp for my **snowpark** project. Might help someone out there too.

**PS**: Standard checkpointing library for Flax models is `orbax`. 

#### usage
- Install with ...
```bash
pip install git+https://github.com/kelechi-c/nnx_save
```
- saving a model
```python
from nnx_save import save_model

# instantiate model
model = nnx.Linear(768, 64, rngs=nnx.Rngs(0)) 

#..training happens here maybe..

state_dict = save_model(model, 'model.safetensors')
```

- loading a model
```python
from nnx_save import load_model

model = nnx.Linear(768, 64, rngs=nnx.Rngs(0)) # must have same structure/attributes as the intially saved one

updated_model, new_state = load_model(model, 'model.safetensors')

nnx.display(model) # optional visualization
```

- loading without ever building a random model (much less host RAM)
```python
# pass a builder instead of a model: it is built with nnx.eval_shape, so the
# parameters exist as shapes only until the checkpoint fills them in
updated_model, new_state = load_model(
    lambda: nnx.Linear(768, 64, rngs=nnx.Rngs(0)), 'model.safetensors'
)
```

- many devices / TPU pod (one shard file per process, nothing gathers the model)
```python
from nnx_save import save_sharded, load_sharded

save_sharded(model, 'ckpt_dir')          # every process writes its own shards
updated_model, state = load_sharded(model, 'ckpt_dir')
```
#### implementation details
- **saving**: get the model state, convert to pure dictionary, flatten, save as a .safetensor file (with the official `safetensors` library of course). Tensor by tensor by default, so only one host copy is alive at a time.
- **loading**: retrieve the model state mapping/pytree from file, unfold to original state, get/replace initialized model state, return updated model. One tensor is read and placed at a time, with the host buffer released before the next.

#### memory
Peak host RAM for a load, in a fresh process, on the **real tensor shapes** of
`stabilityai/stable-audio-3-small-music`: 685 tensors, 567.6M parameters, 2.27 GB fp32.

| load path | 8-core CPU box | RTX 3050 host (7.5 GB RAM) |
| --- | --- | --- |
| whole file into memory (`stream=False`) | **3.03x** the model (6.9 GB), 5.7 s | does not fit |
| streaming, random-initialised model | 2.09x (4.7 GB), 4.3 s | — |
| streaming + builder (`nnx.eval_shape`) | **1.13x** (2.6 GB), 4.0 s | **1.10x** (2.5 GB), 1.9 s |

Three things buy that: `safe_open(...).get_tensor()` reads one tensor instead of the whole file (`safetensors.numpy.load_file` materialises every tensor at once), the builder means no randomly initialised copy of the model is ever allocated, and casting happens on the host before the transfer (one device temporary instead of two). Loading into a bf16 model halves the weights again. On an accelerator the result lives in device memory, so the host figure is smaller still; on a pod `save_sharded`/`load_sharded` keep only this process's shards on this host.

#### things worth knowing
- **state only**: the graph definition is not stored, so `load_model` needs a model built with the same structure as the one that was saved. If the structure does not line up, the load says so instead of quietly keeping random weights.
- **safety checks while loading**: anything that cannot be applied exactly is reported as a warning (`strict=True` raises instead).
  - parameters missing from the file → warning, they keep their initialised values;
  - wrong shape → skipped with a warning (never installed);
  - dtype mismatch → values are cast to the dtype the model declares, so a bf16 model stays bf16;
  - extra keys in the file → warning (useful when checking a ported checkpoint's mapping).
- **sharding and placement**: loaded values go back on the accelerator with the sharding of the variable they replace, so a model built sharded and bf16 (the TPU inference case) stays that way. That path is for shardings this process can address; for a multi-host sharding use `load_sharded`, which reads only the local bytes instead of asking every host for the whole tensor.
- **scalar variables and rngs**: python scalar variables come back as the type they went in as (when loading into a model, not a builder — an abstract build cannot tell an `int` from a 0-dim array), and `nnx.Rngs` keys/counts are saved as their `uint32` key data and rebuilt as typed keys.
- **not supported**: string values in the state (safetensors has no string dtype) and `/` in attribute names (it collides with the checkpoint key separator). Both fail at save time with a message naming the offending parameter.
- **cost on save**: tensor-at-a-time, so the device→host staging is one tensor rather than the whole model; the file is a normal `.safetensors` that any other tool can read.

#### verified against
`jax 0.11.1`, `flax 0.12.9`, `safetensors 0.8.0` — 73 tests covering plain/nested/`nnx.List`/conv models, dtypes from f32 to bf16 and i8, rngs, batchnorm statistics, tied embeddings, 8-device sharding, streaming/classic equivalence, the pod-style shard layout, the single-file shard-range reader, and three PyTorch→NNX ports checked against torch logits (GPT-2, a Llama-style decoder with RMSNorm/RoPE/GQA/SwiGLU, and a ViT-style encoder). See [`tests/README.md`](tests/README.md) and [`tests/REPORT.md`](tests/REPORT.md).