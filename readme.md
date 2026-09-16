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
#### implementation details
- **saving**: get the model state, convert to pure dictionary, flatten, save as a .safetensor file (with the official `safetensors` library of course).

- **loading**: retrieve the model state mapping/pytree from file, unfold to original state, get/replace initialized model state, return updated model.

#### things worth knowing
- **state only**: the graph definition is not stored, so `load_model` needs a model built with the same structure as the one that was saved. If the structure does not line up, the load says so instead of quietly keeping random weights.
- **safety checks while loading**: anything that cannot be applied exactly is reported as a warning (`strict=True` raises instead).
  - parameters missing from the file → warning, they keep their initialised values;
  - wrong shape → skipped with a warning (never installed);
  - dtype mismatch → values are cast to the dtype the model declares, so a bf16 model stays bf16;
  - extra keys in the file → warning (useful when checking a ported checkpoint's mapping).
- **sharding and placement**: loaded values go back on the accelerator with the sharding of the variable they replace, so a model built sharded and bf16 (the TPU inference case) stays that way.
- **rng state**: `nnx.Rngs` keys and counts are saved and restored too (typed PRNG keys are stored as their `uint32` key data).
- **not supported**: string values in the state (safetensors has no string dtype) and `/` in attribute names (it collides with the checkpoint key separator). Both fail at save time with a message naming the offending parameter.

#### verified against
`jax 0.11.1`, `flax 0.12.9`, `safetensors 0.8.0` — 34 tests covering plain/nested/`nnx.List`/conv models, dtypes from f32 to bf16 and i8, rngs, batchnorm statistics, tied embeddings, 8-device sharding, and a PyTorch→NNX GPT-2 port checked against torch logits. See [`tests/README.md`](tests/README.md) and [`tests/REPORT.md`](tests/REPORT.md).