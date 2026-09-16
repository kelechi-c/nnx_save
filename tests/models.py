"""Small NNX models used to exercise ``nnx_save``.

Every model here mirrors a structure that shows up in real ports from
PyTorch: plain linears, mixed dtypes (bf16 for TPU inference), rng/buffer
state, containers (``nnx.List``), tied embeddings, batch-norm running
statistics and a GPT-2 shaped transformer.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@dataclass
class GPT2Config:
    """Model geometry; kept here so the NNX side never needs torch."""

    vocab_size: int = 97
    n_positions: int = 32
    n_embd: int = 32
    n_layer: int = 2
    n_head: int = 4
    layer_norm_epsilon: float = 1e-5


class MLP(nnx.Module):
    """Plain nested modules: the baseline round-trip case."""

    def __init__(self, din: int = 8, dmid: int = 16, dout: int = 4, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(din, dmid, rngs=rngs)
        self.l2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))


class Deep(nnx.Module):
    """Depth 6 so loading into a differently-shaped model is detectable."""

    def __init__(self, width: int = 8, depth: int = 6, *, rngs: nnx.Rngs):
        self.layers = nnx.List([nnx.Linear(width, width, rngs=rngs) for _ in range(depth)])

    def __call__(self, x):
        for layer in self.layers:
            x = nnx.relu(layer(x))
        return x


class DTypeZoo(nnx.Module):
    """Params covering every dtype a TPU inference port realistically holds."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.f32 = nnx.Param(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
        self.bf16 = nnx.Param(jnp.arange(6, dtype=jnp.float32).reshape(2, 3).astype(jnp.bfloat16))
        self.f16 = nnx.Param(jnp.arange(6, dtype=jnp.float32).reshape(2, 3).astype(jnp.float16))
        self.i32 = nnx.Param(jnp.arange(6, dtype=jnp.int32).reshape(2, 3))
        self.i8 = nnx.Param(jnp.arange(6, dtype=jnp.int8).reshape(2, 3))
        self.u32 = nnx.Param(jnp.arange(6, dtype=jnp.uint32).reshape(2, 3))
        self.boolean = nnx.Param(jnp.array([[True, False], [False, True]]))


class ScalarState(nnx.Module):
    """Non-array variables (step counters, scales) sitting next to Params."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.step = nnx.Variable(0)
        self.scale = nnx.Variable(0.5)
        self.flag = nnx.Variable(True)
        self.w = nnx.Param(jnp.ones((2, 2)))

    def __call__(self, x):
        return x * self.scale.value + self.w.value.sum() * 0.0


class PythonScalarState(nnx.Module):
    """Adds a string variable, which safetensors cannot represent at all."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.name = nnx.Variable("block")
        self.w = nnx.Param(jnp.ones((2, 2)))

    def __call__(self, x):
        return x * 0.0 + self.w.value.sum() * 0.0


class WithRngs(nnx.Module):
    """Dropout keeps an ``nnx.Rngs`` in the state — a training-mode model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.lin = nnx.Linear(4, 4, rngs=rngs)
        self.drop = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x):
        return self.drop(self.lin(x))


class WithBatchNorm(nnx.Module):
    """BatchNorm carries ``nnx.BatchStat`` running statistics."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.bn = nnx.BatchNorm(4, rngs=rngs)
        self.lin = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x):
        return self.lin(self.bn(x))


class TiedHead(nnx.Module):
    """Weight tying: one Param object reachable from two attributes."""

    def __init__(self, vocab: int = 16, dim: int = 8, *, rngs: nnx.Rngs):
        self.embedding = nnx.Param(jax.random.normal(rngs.params(), (vocab, dim)))
        # Same Variable object -> shared in the NNX graph.
        self.head = self.embedding

    def __call__(self, ids):
        return self.embedding.value[ids] @ self.head.value.T


class DotNamed(nnx.Module):
    """Attribute names taken verbatim from a PyTorch state dict (dots)."""

    def __init__(self, *, rngs: nnx.Rngs):
        setattr(self, "blocks.0", nnx.Linear(4, 4, rngs=rngs))
        setattr(self, "blocks.1", nnx.Linear(4, 4, rngs=rngs))

    def __call__(self, x):
        return getattr(self, "blocks.1")(getattr(self, "blocks.0")(x))


# --------------------------------------------------------------------------
# GPT-2 shaped transformer used for the PyTorch -> JAX port test.
# --------------------------------------------------------------------------


class GPT2Block(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        eps = cfg.layer_norm_epsilon
        self.ln_1 = nnx.LayerNorm(cfg.n_embd, epsilon=eps, rngs=rngs)
        # nnx.Linear's bias defaults to True, matching HF Conv1D bias.
        self.attn_c_attn = nnx.Linear(cfg.n_embd, 3 * cfg.n_embd, rngs=rngs)
        self.attn_c_proj = nnx.Linear(cfg.n_embd, cfg.n_embd, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(cfg.n_embd, epsilon=eps, rngs=rngs)
        self.mlp_c_fc = nnx.Linear(cfg.n_embd, 4 * cfg.n_embd, rngs=rngs)
        self.mlp_c_proj = nnx.Linear(4 * cfg.n_embd, cfg.n_embd, rngs=rngs)


class TinyGPT2(nnx.Module):
    """Minimal GPT-2 in NNX with HF-compatible parameter paths.

    ``tie_embeddings`` shares the token embedding with the lm head, which is
    how most ported checkpoints ship (GPT-2, Qwen, Llama...).
    """

    def __init__(self, cfg, *, tie_embeddings: bool = True, rngs: nnx.Rngs):
        self.cfg = cfg
        self.wte = nnx.Embed(cfg.vocab_size, cfg.n_embd, rngs=rngs)
        self.wpe = nnx.Embed(cfg.n_positions, cfg.n_embd, rngs=rngs)
        self.blocks = nnx.List([GPT2Block(cfg, rngs=rngs) for _ in range(cfg.n_layer)])
        self.ln_f = nnx.LayerNorm(cfg.n_embd, epsilon=cfg.layer_norm_epsilon, rngs=rngs)
        self.tie_embeddings = tie_embeddings
        if tie_embeddings:
            self.lm_head = self.wte  # tied: one shared Variable
        else:
            self.lm_head = nnx.Linear(cfg.n_embd, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, ids):
        """HF GPT-2 forward: causal multi-head attention, exact gelu."""
        b, t = ids.shape
        n_head = self.cfg.n_head
        head_dim = self.cfg.n_embd // n_head
        pos = jnp.arange(t)
        causal = jnp.tril(jnp.ones((t, t), dtype=bool))

        def split_heads(x):
            return x.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)

        x = self.wte(ids) + self.wpe(pos)[None]
        for block in self.blocks:
            h = block.ln_1(x)
            q, k, v = jnp.split(block.attn_c_attn(h), 3, axis=-1)
            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            logits = q @ k.swapaxes(-1, -2) / jnp.sqrt(jnp.asarray(head_dim, q.dtype))
            logits = jnp.where(causal, logits, jnp.finfo(logits.dtype).min)
            attn = jax.nn.softmax(logits, axis=-1)
            ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, self.cfg.n_embd)
            x = x + block.attn_c_proj(ctx)
            x = x + block.mlp_c_proj(jax.nn.gelu(block.mlp_c_fc(block.ln_2(x)), approximate=False))
        x = self.ln_f(x)
        if self.tie_embeddings:
            return x @ self.wte.embedding.value.T
        return self.lm_head(x)


class ConvAudio(nnx.Module):
    """Convolution stack: the shape most audio ports (demucs-likes) have."""

    def __init__(self, in_ch: int = 2, width: int = 8, *, rngs: nnx.Rngs):
        self.conv_in = nnx.Conv(in_ch, width, kernel_size=(7,), padding="SAME", rngs=rngs)
        self.norm = nnx.GroupNorm(width, num_groups=2, rngs=rngs)
        self.conv_mid = nnx.Conv(width, width, kernel_size=(5,), padding="SAME", rngs=rngs)
        self.conv_out = nnx.Conv(width, in_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.norm(self.conv_in(x)))
        x = nnx.relu(self.conv_mid(x))
        return self.conv_out(x)


def jax_path_for(hf_key: str, *, n_embd: int) -> str | None:
    """Map an HF GPT-2 key onto the NNX model's path (the actual port)."""
    if hf_key == "transformer.wte.weight":
        return "wte/embedding"
    if hf_key == "transformer.wpe.weight":
        return "wpe/embedding"
    if hf_key == "transformer.ln_f.weight":
        return "ln_f/scale"
    if hf_key == "transformer.ln_f.bias":
        return "ln_f/bias"
    if hf_key.startswith("transformer.h."):
        _, _, idx, rest = hf_key.split(".", 3)
        base = f"blocks/{idx}/"
        table = {
            "ln_1.weight": "ln_1/scale",
            "ln_1.bias": "ln_1/bias",
            "attn.c_attn.weight": "attn_c_attn/kernel",
            "attn.c_attn.bias": "attn_c_attn/bias",
            "attn.c_proj.weight": "attn_c_proj/kernel",
            "attn.c_proj.bias": "attn_c_proj/bias",
            "ln_2.weight": "ln_2/scale",
            "ln_2.bias": "ln_2/bias",
            "mlp.c_fc.weight": "mlp_c_fc/kernel",
            "mlp.c_fc.bias": "mlp_c_fc/bias",
            "mlp.c_proj.weight": "mlp_c_proj/kernel",
            "mlp.c_proj.bias": "mlp_c_proj/bias",
        }
        if rest not in table:
            return None
        return base + table[rest]
    return None  # lm_head.weight is absent in tied checkpoints


def shard_state(model, mesh: Mesh, spec=P("x")):
    """Shard every parameter of ``model`` along ``spec`` (TPU-style)."""
    sharding = NamedSharding(mesh, spec)
    state = nnx.state(model)
    sharded = jax.tree.map(lambda x: jax.device_put(x, sharding), state)
    nnx.update(model, sharded)
    return model


# --------------------------------------------------------------------------
# Llama-style decoder: the other transformer most inference ports ship.
#
# Geometry and math follow HF's ``LlamaForCausalLM``: RMSNorm with no mean
# subtraction and no bias, RoPE on q/k, grouped-query attention (more query
# heads than KV heads), a SwiGLU MLP, no biases anywhere, and a separate
# (untied) ``lm_head``.
# --------------------------------------------------------------------------


@dataclass
class LlamaConfig:
    """HF ``LlamaConfig`` field names, so a real config.json maps straight on."""

    vocab_size: int = 128
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2  # < heads on purpose: GQA is the hard part
    intermediate_size: int = 64
    max_position_embeddings: int = 32
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class LlamaRMSNorm(nnx.Module):
    """HF ``LlamaRMSNorm``: weight only, no mean subtraction, no bias.

    HF computes the statistics in fp32 and multiplies by the weight in the
    activation dtype; doing the same keeps a bf16 port close to a bf16
    transformers model.
    """

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(jnp.ones((dim,), jnp.float32))
        self.eps = eps

    def __call__(self, x):
        dtype = x.dtype
        h = jnp.asarray(x, jnp.float32)
        variance = jnp.mean(jnp.square(h), axis=-1, keepdims=True)
        h = h * jax.lax.rsqrt(variance + self.eps)
        return self.scale.value * h.astype(dtype)


def llama_rope(head_dim: int, positions, theta: float):
    """HF Llama RoPE angles, ``(cos, sin)`` of shape ``(len(positions), head_dim)``.

    Exactly HF's convention (``modeling_rope_utils._compute_default_rope_parameters``):

        inv_freq = 1 / theta ** (arange(0, head_dim, 2) / head_dim)
        emb      = outer(position, inv_freq), then duplicated: cat(emb, emb)

    The duplication (rather than interleaving) is what makes ``rotate_half``
    below the matching rotation, so both sides must use this one and not the
    GPT-J interleaved variant.
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.outer(jnp.asarray(positions, jnp.float32), inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x):
    """HF ``rotate_half``: ``cat(-x2, x1)``, i.e. rotate halves, not pairs."""
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class LlamaAttention(nnx.Module):
    """GQA attention with RoPE on q/k and no bias on any projection."""

    def __init__(self, cfg: LlamaConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = nnx.Linear(cfg.hidden_size, q_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(cfg.hidden_size, kv_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(cfg.hidden_size, kv_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(q_dim, cfg.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x, cos, sin, causal):
        cfg = self.cfg
        b, t, _ = x.shape

        def heads(y, n):
            return y.reshape(b, t, n, cfg.head_dim).transpose(0, 2, 1, 3)

        q = heads(self.q_proj(x), cfg.num_attention_heads)
        k = heads(self.k_proj(x), cfg.num_key_value_heads)
        v = heads(self.v_proj(x), cfg.num_key_value_heads)

        # HF casts the angles to the activation dtype before applying them.
        cos, sin = cos.astype(q.dtype)[None, None], sin.astype(q.dtype)[None, None]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # GQA: HF's repeat_kv expands (b, n_kv, t, d) to (b, n_kv * n_rep, t, d),
        # i.e. heads [kv0, kv0, kv1, kv1]. jnp.repeat(axis=1) is exactly that, so
        # the simplest exact-match approach is an explicit repeat: no need for
        # jax.nn.dot_product_attention(enable_gqa=True), and the softmax/mask
        # arithmetic stays identical to HF's eager path.
        n_rep = cfg.num_key_value_groups
        if n_rep > 1:
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)

        scores = q @ k.swapaxes(-1, -2) / jnp.sqrt(jnp.asarray(cfg.head_dim, q.dtype))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        # HF upcasts the softmax to fp32 and casts the probabilities back.
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self.o_proj(ctx)


class LlamaMLP(nnx.Module):
    """SwiGLU: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""

    def __init__(self, cfg: LlamaConfig, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(cfg.hidden_size, cfg.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(cfg.hidden_size, cfg.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(cfg.intermediate_size, cfg.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nnx.Module):
    """Pre-norm block: attention then SwiGLU MLP, both residual."""

    def __init__(self, cfg: LlamaConfig, *, rngs: nnx.Rngs):
        self.input_layernorm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)
        self.self_attn = LlamaAttention(cfg, rngs=rngs)
        self.post_attention_layernorm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)
        self.mlp = LlamaMLP(cfg, rngs=rngs)

    def __call__(self, x, cos, sin, causal):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, causal)
        return h + self.mlp(self.post_attention_layernorm(h))


class TinyLlama(nnx.Module):
    """HF ``LlamaForCausalLM`` in NNX with real HF parameter paths.

    ``lm_head`` is a separate Linear: Llama checkpoints are untied
    (``tie_word_embeddings: false``), unlike the GPT-2 fixture.
    """

    def __init__(self, cfg: LlamaConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embed_tokens = nnx.Embed(cfg.vocab_size, cfg.hidden_size, rngs=rngs)
        self.layers = nnx.List([LlamaBlock(cfg, rngs=rngs) for _ in range(cfg.num_hidden_layers)])
        self.norm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)
        self.lm_head = nnx.Linear(cfg.hidden_size, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, ids):
        _, t = ids.shape
        # One set of angles for every layer, as HF computes them once up front.
        cos, sin = llama_rope(self.cfg.head_dim, jnp.arange(t), self.cfg.rope_theta)
        causal = jnp.tril(jnp.ones((t, t), dtype=bool))
        x = self.embed_tokens(ids)
        for block in self.layers:
            x = block(x, cos, sin, causal)
        return self.lm_head(self.norm(x))


def llama_jax_path_for(hf_key: str) -> str | None:
    """Map a real HF Llama checkpoint key onto the NNX model's path.

    The names are those of any ``LlamaForCausalLM`` checkpoint; they were
    checked against ``hf-internal-testing/tiny-random-LlamaForCausalLM``
    (21 tensors, no biases, untied ``lm_head``). Older checkpoints may also
    carry a non-persistent ``...self_attn.rotary_emb.inv_freq`` buffer, which
    has no NNX home and maps to ``None``.
    """
    if hf_key == "model.embed_tokens.weight":
        return "embed_tokens/embedding"
    if hf_key == "model.norm.weight":
        return "norm/scale"
    if hf_key == "lm_head.weight":
        return "lm_head/kernel"
    if hf_key.startswith("model.layers."):
        _, _, idx, rest = hf_key.split(".", 3)
        table = {
            "input_layernorm.weight": "input_layernorm/scale",
            "post_attention_layernorm.weight": "post_attention_layernorm/scale",
            "self_attn.q_proj.weight": "self_attn/q_proj/kernel",
            "self_attn.k_proj.weight": "self_attn/k_proj/kernel",
            "self_attn.v_proj.weight": "self_attn/v_proj/kernel",
            "self_attn.o_proj.weight": "self_attn/o_proj/kernel",
            "mlp.gate_proj.weight": "mlp/gate_proj/kernel",
            "mlp.up_proj.weight": "mlp/up_proj/kernel",
            "mlp.down_proj.weight": "mlp/down_proj/kernel",
        }
        if rest not in table:
            return None
        return f"layers/{idx}/" + table[rest]
    return None


# --------------------------------------------------------------------------
# ViT-style encoder: patch-embedding convolution, class token, pre-LN
# attention blocks, MLP, final norm and a classification head.  Parameter
# names follow ``transformers.ViTForImageClassification``.
# --------------------------------------------------------------------------


@dataclass
class ViTConfig:
    """HF ``ViTConfig`` field names (a tiny local geometry)."""

    image_size: int = 8
    patch_size: int = 4
    num_channels: int = 3
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 64
    num_labels: int = 5
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu"

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class ViTPatchEmbeddings(nnx.Module):
    """The patch conv: NHWC pixels in, ``(batch, num_patches, hidden)`` out."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.projection = nnx.Conv(
            cfg.num_channels,
            cfg.hidden_size,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=(cfg.patch_size, cfg.patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, pixels):
        # (b, grid, grid, hidden) -> (b, grid * grid, hidden); jax's spatial
        # order is row-major, like HF's flatten(2).transpose(1, 2).
        x = self.projection(pixels)
        return x.reshape(x.shape[0], -1, x.shape[-1])


class ViTEmbeddings(nnx.Module):
    """Patch embeddings + [CLS] token + learned position embeddings."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.patch_embeddings = ViTPatchEmbeddings(cfg, rngs=rngs)
        self.cls_token = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (1, 1, cfg.hidden_size))
        )
        self.position_embeddings = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (1, cfg.num_patches + 1, cfg.hidden_size))
        )

    def __call__(self, pixels):
        x = self.patch_embeddings(pixels)
        cls = jnp.broadcast_to(self.cls_token.value, (x.shape[0], 1, x.shape[-1]))
        return jnp.concatenate([cls, x], axis=1) + self.position_embeddings.value


class ViTSelfAttention(nnx.Module):
    """Unmasked multi-head attention; ViT normalizes in the activation dtype."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.query = nnx.Linear(cfg.hidden_size, cfg.hidden_size, rngs=rngs)
        self.key = nnx.Linear(cfg.hidden_size, cfg.hidden_size, rngs=rngs)
        self.value = nnx.Linear(cfg.hidden_size, cfg.hidden_size, rngs=rngs)

    def __call__(self, x):
        cfg = self.cfg
        b, t, _ = x.shape

        def heads(y):
            return y.reshape(b, t, cfg.num_attention_heads, cfg.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        scores = q @ k.swapaxes(-1, -2) / jnp.sqrt(jnp.asarray(cfg.head_dim, q.dtype))
        attn = jax.nn.softmax(scores, axis=-1)
        return (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, -1)


class ViTSelfOutput(nnx.Module):
    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(cfg.hidden_size, cfg.hidden_size, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)


class ViTAttention(nnx.Module):
    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.attention = ViTSelfAttention(cfg, rngs=rngs)
        self.output = ViTSelfOutput(cfg, rngs=rngs)

    def __call__(self, x):
        return self.output(self.attention(x))


class ViTIntermediate(nnx.Module):
    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(cfg.hidden_size, cfg.intermediate_size, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)


class ViTOutput(nnx.Module):
    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(cfg.intermediate_size, cfg.hidden_size, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)


class ViTLayer(nnx.Module):
    """Pre-LN block: ``attention(ln_before(x)) + x``, then ``mlp(ln_after) + x``."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.layernorm_before = nnx.LayerNorm(cfg.hidden_size, epsilon=cfg.layer_norm_eps, rngs=rngs)
        self.attention = ViTAttention(cfg, rngs=rngs)
        self.layernorm_after = nnx.LayerNorm(cfg.hidden_size, epsilon=cfg.layer_norm_eps, rngs=rngs)
        self.intermediate = ViTIntermediate(cfg, rngs=rngs)
        self.output = ViTOutput(cfg, rngs=rngs)

    def __call__(self, x):
        h = x + self.attention(self.layernorm_before(x))
        mlp = self.output(jax.nn.gelu(self.intermediate(self.layernorm_after(h)), approximate=False))
        return h + mlp


class ViTEncoder(nnx.Module):
    """HF's ``vit.encoder``: a ``layer.N`` block list."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.layer = nnx.List([ViTLayer(cfg, rngs=rngs) for _ in range(cfg.num_hidden_layers)])

    def __call__(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class ViTModel(nnx.Module):
    """The ``vit`` sub-tree: embeddings, encoder blocks, final layer norm."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.embeddings = ViTEmbeddings(cfg, rngs=rngs)
        self.encoder = ViTEncoder(cfg, rngs=rngs)
        self.layernorm = nnx.LayerNorm(cfg.hidden_size, epsilon=cfg.layer_norm_eps, rngs=rngs)

    def __call__(self, pixels):
        return self.layernorm(self.encoder(self.embeddings(pixels)))


class TinyViT(nnx.Module):
    """``ViTForImageClassification`` in NNX: ``vit.*`` encoder + ``classifier``.

    HF classifies the final-norm [CLS] token and does not build the pooler, so
    the checkpoint has no ``vit.pooler.*`` keys.
    """

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.vit = ViTModel(cfg, rngs=rngs)
        self.classifier = nnx.Linear(cfg.hidden_size, cfg.num_labels, rngs=rngs)

    def __call__(self, pixels):
        return self.classifier(self.vit(pixels)[:, 0])


def vit_jax_path_for(hf_key: str) -> str | None:
    """Map a real HF ViT checkpoint key onto the NNX model's path.

    Names follow ``transformers.ViTForImageClassification`` and were checked
    against ``hf-internal-testing/tiny-random-vit``: ``vit.encoder.layer.N.*``
    for the blocks, ``vit.layernorm`` after them, and a ``classifier`` head
    (no pooler).
    """
    simple = {
        "vit.embeddings.cls_token": "vit/embeddings/cls_token",
        "vit.embeddings.position_embeddings": "vit/embeddings/position_embeddings",
        # nnx.Conv's kernel is (kh, kw, in, out); the HF conv weight is
        # (out, in, kh, kw) and is permuted by the port, not here.
        "vit.embeddings.patch_embeddings.projection.weight": "vit/embeddings/patch_embeddings/projection/kernel",
        "vit.embeddings.patch_embeddings.projection.bias": "vit/embeddings/patch_embeddings/projection/bias",
        "vit.layernorm.weight": "vit/layernorm/scale",
        "vit.layernorm.bias": "vit/layernorm/bias",
        "classifier.weight": "classifier/kernel",
        "classifier.bias": "classifier/bias",
    }
    if hf_key in simple:
        return simple[hf_key]
    if hf_key.startswith("vit.encoder.layer."):
        _, _, _, idx, rest = hf_key.split(".", 4)
        table = {
            "layernorm_before.weight": "layernorm_before/scale",
            "layernorm_before.bias": "layernorm_before/bias",
            "layernorm_after.weight": "layernorm_after/scale",
            "layernorm_after.bias": "layernorm_after/bias",
            "attention.attention.query.weight": "attention/attention/query/kernel",
            "attention.attention.query.bias": "attention/attention/query/bias",
            "attention.attention.key.weight": "attention/attention/key/kernel",
            "attention.attention.key.bias": "attention/attention/key/bias",
            "attention.attention.value.weight": "attention/attention/value/kernel",
            "attention.attention.value.bias": "attention/attention/value/bias",
            "attention.output.dense.weight": "attention/output/dense/kernel",
            "attention.output.dense.bias": "attention/output/dense/bias",
            "intermediate.dense.weight": "intermediate/dense/kernel",
            "intermediate.dense.bias": "intermediate/dense/bias",
            "output.dense.weight": "output/dense/kernel",
            "output.dense.bias": "output/dense/bias",
        }
        if rest not in table:
            return None
        return f"vit/encoder/layer/{idx}/" + table[rest]
    return None
