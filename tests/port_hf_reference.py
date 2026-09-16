"""Dump a real HuggingFace checkpoint + PyTorch reference logits.

Runs in an environment that has torch + transformers (on victoria that is the
hummingbird venv).  The jax side of the port then runs in a JAX-only venv, so
the two are genuinely independent.

Both architectures ``port_hf_check.py`` understands can be dumped here; the
checkpoint's own ``config.json`` (inside ``reference.npz`` as well) is what the
jax side dispatches on.

Examples (on victoria):
  /home/tensor/code/ml/hummingbird/.venv/bin/python tests/port_hf_reference.py \
      --model hf-internal-testing/tiny-random-gpt2 --out /home/tensor/nnx_save_test/hf_tiny
  /home/tensor/code/ml/hummingbird/.venv/bin/python tests/port_hf_reference.py \
      --arch llama --out /tmp/hf_tiny_llama
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

DEFAULT_MODELS = {
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpt2", choices=sorted(DEFAULT_MODELS),
                    help="which tiny-random checkpoint --model defaults to")
    ap.add_argument("--model", default=None, help="HF model id (default: the --arch tiny model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seq", type=int, default=16)
    args = ap.parse_args()

    model_id = args.model or DEFAULT_MODELS[args.arch]
    os.makedirs(args.out, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:  # transformers < 5
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()

    ckpt_path = os.path.join(args.out, "model.safetensors")
    model.save_pretrained(args.out, safe_serialization=True)
    print(f"saved {model.config.model_type} checkpoint:", ckpt_path)

    torch.manual_seed(0)
    ids = torch.randint(1, model.config.vocab_size, (2, args.seq), dtype=torch.long)
    with torch.no_grad():
        logits = model(ids).logits

    np.savez(
        os.path.join(args.out, "reference.npz"),
        ids=ids.numpy().astype(np.int32),
        logits=logits.detach().float().cpu().numpy(),
        config=json.dumps(model.config.to_dict()),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"reference logits {tuple(logits.shape)} from {n_params/1e6:.2f}M params ({args.dtype})")


if __name__ == "__main__":
    main()
