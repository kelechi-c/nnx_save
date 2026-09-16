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
