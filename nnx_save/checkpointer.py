from safetensors.numpy import load_file as load_flax, save_file as save_flax
from flax import nnx


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


def load_model(model, model_file="./model.safetensors"):
    
    # Load the flattened checkpoint dictionary.
    loaded_state = load_flax(model_file)
    
    loaded_nested = {}
    # First, unflatten the keys (they are strings with the sep character)
    for k, v in loaded_state.items():
        keys = tuple(k.split("/"))
        current = loaded_nested
        for part in keys[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[keys[-1]] = v

    # Obtain the abstract state from the model.
    graphdef, abstract_state = nnx.split(model)
    # Convert the abstract state to a pure dictionary.
    expected_state = nnx.to_pure_dict(abstract_state)

    # Flatten both dictionaries.
    flat_loaded = flatten_dict(loaded_nested)
    flat_expected = flatten_dict(expected_state)

    # Filter the loaded flat dictionary to include only keys that exist in the expected state.
    filtered_flat = {k: v for k, v in flat_loaded.items() if k in flat_expected}

    # Unflatten back to the nested dictionary structure.
    filtered_state = unflatten_dict(filtered_flat)

    # Update the abstract state with the filtered state.
    nnx.replace_by_pure_dict(abstract_state, filtered_state)
    
    # Merge the updated state with the graph definition.
    model = nnx.merge(graphdef, abstract_state)

    return model, nnx.state(model)


def save_model(model, model_file='./model.safetensors'):

    def flatten_dict(d, parent_key='', sep='/'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    state = nnx.state(model).to_pure_dict()
    flat_dict = flatten_dict(state)

    tensor_dict = flat_dict
    save_flax(tensor_dict, model_file)

    return tensor_dict
