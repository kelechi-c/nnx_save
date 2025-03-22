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