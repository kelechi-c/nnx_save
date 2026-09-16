"""A dependency-light PyTorch GPT-2 used as the porting source.

Only ``torch`` is required (no transformers): the point is to produce a
checkpoint with **HuggingFace GPT-2 parameter names** and a reference forward
pass, so the JAX/NNX port can be checked numerically before and after an
``nnx_save`` round-trip.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import GPT2Config, jax_path_for  # noqa: F401  (re-exported for the torch tests)


class Conv1D(nn.Module):
    """HF GPT-2's Conv1D: weight is (in_features, out_features), like NNX."""

    def __init__(self, nf_out: int, nf_in: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nf_in, nf_out))
        self.bias = nn.Parameter(torch.zeros(nf_out))
        self.nf = nf_out
        # Match nn.Linear's default init scale closely enough for a test.
        nn.init.normal_(self.weight, std=(1.0 / nf_in) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn_c_attn = Conv1D(3 * cfg.n_embd, cfg.n_embd)
        self.attn_c_proj = Conv1D(cfg.n_embd, cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp_c_fc = Conv1D(4 * cfg.n_embd, cfg.n_embd)
        self.mlp_c_proj = Conv1D(cfg.n_embd, 4 * cfg.n_embd)


class TorchGPT2(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied, like HF GPT-2

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, t = ids.shape
        head_dim = cfg.n_embd // cfg.n_head
        pos = torch.arange(t, device=ids.device)
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=ids.device))

        def split_heads(x):
            return x.view(b, t, cfg.n_head, head_dim).transpose(1, 2)

        x = self.wte(ids) + self.wpe(pos)[None]
        for block in self.h:
            h = block.ln_1(x)
            q, k, v = block.attn_c_attn(h).split(cfg.n_embd, dim=-1)
            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            logits = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
            logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
            ctx = (F.softmax(logits, dim=-1) @ v).transpose(1, 2).reshape(b, t, cfg.n_embd)
            x = x + block.attn_c_proj(ctx)
            x = x + block.mlp_c_proj(F.gelu(block.mlp_c_fc(block.ln_2(x)), approximate="none"))
        x = self.ln_f(x)
        return x @ self.wte.weight.T


def hf_style_state_dict(model: TorchGPT2) -> dict[str, torch.Tensor]:
    """Rename a TorchGPT2 into the names a real HF GPT-2 checkpoint uses.

    Tied ``lm_head.weight`` is deliberately omitted, exactly as HF stores it
    for GPT-2 (``tie_word_embeddings: true``).
    """
    out = {
        "transformer.wte.weight": model.wte.weight,
        "transformer.wpe.weight": model.wpe.weight,
        "transformer.ln_f.weight": model.ln_f.weight,
        "transformer.ln_f.bias": model.ln_f.bias,
    }
    for i, block in enumerate(model.h):
        p = f"transformer.h.{i}"
        out[f"{p}.ln_1.weight"] = block.ln_1.weight
        out[f"{p}.ln_1.bias"] = block.ln_1.bias
        out[f"{p}.attn.c_attn.weight"] = block.attn_c_attn.weight
        out[f"{p}.attn.c_attn.bias"] = block.attn_c_attn.bias
        out[f"{p}.attn.c_proj.weight"] = block.attn_c_proj.weight
        out[f"{p}.attn.c_proj.bias"] = block.attn_c_proj.bias
        out[f"{p}.ln_2.weight"] = block.ln_2.weight
        out[f"{p}.ln_2.bias"] = block.ln_2.bias
        out[f"{p}.mlp.c_fc.weight"] = block.mlp_c_fc.weight
        out[f"{p}.mlp.c_fc.bias"] = block.mlp_c_fc.bias
        out[f"{p}.mlp.c_proj.weight"] = block.mlp_c_proj.weight
        out[f"{p}.mlp.c_proj.bias"] = block.mlp_c_proj.bias
    return out


