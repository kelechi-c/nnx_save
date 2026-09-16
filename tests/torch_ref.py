"""Dependency-light PyTorch references used as the porting source.

Only ``torch`` is required (no transformers): the point is to produce
checkpoints with **HuggingFace parameter names** and a reference forward pass,
so the JAX/NNX ports can be checked numerically before and after an
``nnx_save`` round-trip.  Three architectures are covered: GPT-2
(``TorchGPT2``), a Llama-style decoder (``TorchLlama``) and a ViT-style
encoder (``TorchViT``).  The parameter names and the math follow the
corresponding HF ``modeling_*`` implementations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (  # noqa: F401  (re-exported for the torch tests)
    GPT2Config,
    LlamaConfig,
    ViTConfig,
    jax_path_for,
    llama_jax_path_for,
    vit_jax_path_for,
)


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


# --------------------------------------------------------------------------
# Llama: RMSNorm, RoPE on q/k, GQA, SwiGLU, no biases, untied lm_head.
#
# Mirrors HF ``LlamaForCausalLM``.  Its parameter names match a real
# ``hf-internal-testing/tiny-random-LlamaForCausalLM`` checkpoint exactly
# (21 tensors for a 2-layer model), which is what makes the port test a real
# port and not a renamed paraphrase.
# --------------------------------------------------------------------------


class TorchLlamaRMSNorm(nn.Module):
    """HF ``LlamaRMSNorm``: fp32 statistics, weight only, no bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


def _llama_rope(head_dim: int, positions: torch.Tensor, theta: float):
    """HF Llama angles: ``inv_freq = 1/theta**(arange(0, d, 2)/d)``, duplicated.

    Returns ``(cos, sin)`` of shape ``(len(positions), head_dim)``.  The
    duplication (not interleaving) must match the JAX side and ``rotate_half``.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim)
    )
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF ``rotate_half``: ``cat(-x2, x1)`` along the last axis."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """HF ``repeat_kv``: (b, n_kv, t, d) -> (b, n_kv * n_rep, t, d)."""
    if n_rep == 1:
        return hidden_states
    batch, n_kv, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, n_kv, n_rep, slen, head_dim)
        .reshape(batch, n_kv * n_rep, slen, head_dim)
    )


class TorchLlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, causal: torch.Tensor):
        cfg = self.cfg
        b, t, _ = x.shape

        def heads(y, n):
            return y.view(b, t, n, cfg.head_dim).transpose(1, 2)

        q = heads(self.q_proj(x), cfg.num_attention_heads)
        k = heads(self.k_proj(x), cfg.num_key_value_heads)
        v = heads(self.v_proj(x), cfg.num_key_value_heads)

        # HF casts the angles to the activation dtype before applying them.
        q_cos, q_sin = cos.to(q.dtype)[None, None], sin.to(q.dtype)[None, None]
        q = q * q_cos + _rotate_half(q) * q_sin
        k = k * q_cos + _rotate_half(k) * q_sin

        k = _repeat_kv(k, cfg.num_key_value_groups)
        v = _repeat_kv(v, cfg.num_key_value_groups)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        # HF upcasts the softmax to fp32 and casts the probabilities back.
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        ctx = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(ctx)


class TorchLlamaMLP(nn.Module):
    """SwiGLU: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TorchLlamaLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.input_layernorm = TorchLlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = TorchLlamaAttention(cfg)
        self.post_attention_layernorm = TorchLlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = TorchLlamaMLP(cfg)

    def forward(self, x, cos, sin, causal):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, causal)
        return h + self.mlp(self.post_attention_layernorm(h))


class TorchLlama(nn.Module):
    """HF ``LlamaForCausalLM`` with plain ``torch.nn`` modules."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()  # HF's LlamaModel, so keys start with 'model.'
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList([TorchLlamaLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.model.norm = TorchLlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self, std: float = 0.02):
        """HF's Llama init: N(0, 0.02) embeddings/linears, norm weights = 1."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, TorchLlamaRMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, t = ids.shape
        cos, sin = _llama_rope(self.cfg.head_dim, torch.arange(t, device=ids.device), self.cfg.rope_theta)
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=ids.device))
        x = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            x = layer(x, cos, sin, causal)
        return self.lm_head(self.model.norm(x))


def hf_style_llama_state_dict(model: TorchLlama) -> dict[str, torch.Tensor]:
    """Rename a TorchLlama into the names a real HF Llama checkpoint uses.

    ``lm_head.weight`` is present: Llama ships untied
    (``tie_word_embeddings: false``), and there are no bias tensors anywhere.
    """
    out = {
        "model.embed_tokens.weight": model.model.embed_tokens.weight,
        "model.norm.weight": model.model.norm.weight,
        "lm_head.weight": model.lm_head.weight,
    }
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}"
        out[f"{p}.input_layernorm.weight"] = layer.input_layernorm.weight
        out[f"{p}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight
        out[f"{p}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight
        out[f"{p}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight
        out[f"{p}.self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight
        out[f"{p}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        out[f"{p}.mlp.gate_proj.weight"] = layer.mlp.gate_proj.weight
        out[f"{p}.mlp.up_proj.weight"] = layer.mlp.up_proj.weight
        out[f"{p}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight
    return out


# --------------------------------------------------------------------------
# ViT: patch-embedding convolution, class token, pre-LN blocks, final norm
# and a classification head.  Names mirror ``transformers.ViTModel`` /
# ``ViTForImageClassification`` (no pooler: HF does not build it for the
# classification head).
# --------------------------------------------------------------------------


class TorchViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            cfg.num_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels).flatten(2).transpose(1, 2)


class TorchViTEmbeddings(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.patch_embeddings = TorchViTPatchEmbeddings(cfg)
        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.hidden_size))
        self.position_embeddings = nn.Parameter(torch.empty(1, cfg.num_patches + 1, cfg.hidden_size))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(pixels)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls, x), dim=1) + self.position_embeddings


class TorchViTSelfAttention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, t, _ = x.shape

        def heads(y):
            return y.view(b, t, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(cfg.head_dim)
        attn = torch.softmax(scores, dim=-1)  # ViT does not upcast the softmax
        return (attn @ v).transpose(1, 2).reshape(b, t, -1)


class TorchViTSelfOutput(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class TorchViTAttention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.attention = TorchViTSelfAttention(cfg)
        self.output = TorchViTSelfOutput(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(x))


class TorchViTIntermediate(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class TorchViTOutput(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class TorchViTLayer(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attention = TorchViTAttention(cfg)
        self.layernorm_after = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.intermediate = TorchViTIntermediate(cfg)
        self.output = TorchViTOutput(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.layernorm_before(x)) + x
        mlp = self.output(F.gelu(self.intermediate(self.layernorm_after(x)), approximate="none"))
        return mlp + x


class TorchViT(nn.Module):
    """HF ``ViTForImageClassification`` with plain ``torch.nn`` modules."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.vit = nn.Module()  # HF's ViTModel
        self.vit.embeddings = TorchViTEmbeddings(cfg)
        self.vit.encoder = nn.Module()
        self.vit.encoder.layer = nn.ModuleList(
            [TorchViTLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.vit.layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)
        self._init_weights()

    def _init_weights(self, std: float = 0.02):
        """HF's ViT init: N(0, 0.02) convs/linears/embeddings, biases zero."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.vit.embeddings.cls_token, mean=0.0, std=std)
        nn.init.normal_(self.vit.embeddings.position_embeddings, mean=0.0, std=std)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.vit.embeddings(pixels)
        for layer in self.vit.encoder.layer:
            x = layer(x)
        return self.classifier(self.vit.layernorm(x)[:, 0])


def hf_style_vit_state_dict(model: TorchViT) -> dict[str, torch.Tensor]:
    """Rename a TorchViT into the names a real HF ViT checkpoint uses.

    Same keys as ``hf-internal-testing/tiny-random-vit``: the patch conv is
    ``(out, in, kh, kw)``, every block has q/k/v + output biases, and the head
    is a ``classifier`` (no ``vit.pooler.*``).
    """
    out = {
        "vit.embeddings.cls_token": model.vit.embeddings.cls_token,
        "vit.embeddings.position_embeddings": model.vit.embeddings.position_embeddings,
        "vit.embeddings.patch_embeddings.projection.weight": model.vit.embeddings.patch_embeddings.projection.weight,
        "vit.embeddings.patch_embeddings.projection.bias": model.vit.embeddings.patch_embeddings.projection.bias,
        "vit.layernorm.weight": model.vit.layernorm.weight,
        "vit.layernorm.bias": model.vit.layernorm.bias,
        "classifier.weight": model.classifier.weight,
        "classifier.bias": model.classifier.bias,
    }
    for i, layer in enumerate(model.vit.encoder.layer):
        p = f"vit.encoder.layer.{i}"
        out[f"{p}.layernorm_before.weight"] = layer.layernorm_before.weight
        out[f"{p}.layernorm_before.bias"] = layer.layernorm_before.bias
        out[f"{p}.attention.attention.query.weight"] = layer.attention.attention.query.weight
        out[f"{p}.attention.attention.query.bias"] = layer.attention.attention.query.bias
        out[f"{p}.attention.attention.key.weight"] = layer.attention.attention.key.weight
        out[f"{p}.attention.attention.key.bias"] = layer.attention.attention.key.bias
        out[f"{p}.attention.attention.value.weight"] = layer.attention.attention.value.weight
        out[f"{p}.attention.attention.value.bias"] = layer.attention.attention.value.bias
        out[f"{p}.attention.output.dense.weight"] = layer.attention.output.dense.weight
        out[f"{p}.attention.output.dense.bias"] = layer.attention.output.dense.bias
        out[f"{p}.layernorm_after.weight"] = layer.layernorm_after.weight
        out[f"{p}.layernorm_after.bias"] = layer.layernorm_after.bias
        out[f"{p}.intermediate.dense.weight"] = layer.intermediate.dense.weight
        out[f"{p}.intermediate.dense.bias"] = layer.intermediate.dense.bias
        out[f"{p}.output.dense.weight"] = layer.output.dense.weight
        out[f"{p}.output.dense.bias"] = layer.output.dense.bias
    return out


