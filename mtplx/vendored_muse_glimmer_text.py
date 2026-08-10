# Copyright © 2026 MTPLX contributors.
"""Vendored MLX model definition for the Muse-Glimmer text tower.

Registered as ``mlx_lm.models.muse_glimmer_text`` by ``muse_glimmer_patch`` so
``mlx_lm.utils.load`` (and therefore ``mtplx serve``) can build the model; no
released mlx-lm ships it.

Ported 1:1 from llama.cpp ``src/models/muse-glimmer.cpp`` (the authoritative
reference — transformers has no muse_glimmer modeling code). Deviations from a
plain Gemma-3 text model:

  * Sigmoid **gated attention** — ``o_proj(sdpa_out * sigmoid(gate_proj(x)))``.
  * **Parameter-free QK-norm**: RMSNorm(q,k) with no learnable weight; the
    ``qk_scale_factor`` (3.87) is multiplied onto Q (llama.cpp synthesizes this
    into ``attn_q_norm``; the HF checkpoint has no q/k-norm weights).
  * **NoPE on the global (full-attention) layers**; RoPE (theta=500000) only on
    the sliding-window layers (3 local : 1 global).
  * Gemma-style ``(1 + weight)`` sandwich norms; post-attn/post-FFN norms use
    eps ``1e-8`` (``post_norm_eps``) vs ``1e-5`` for the pre-norms.
  * Embeddings **RMS-normalized** (no weight), NOT scaled by ``sqrt(hidden)``.
  * Output: ``logit_scale`` (0.196) then tanh softcap (20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

# Per-attention cache of the fused q/k/v/gate projection (built lazily from the
# loaded per-projection weights), keyed by id() since nn.Module is unhashable.
# Bit-exact vs the 4 separate matmuls; fusing them into one quantized_matmul
# cuts 4 kernel launches/layer -> 1 and measured +4.8% B=1 decode on the q4
# checkpoint. Kept off the nn.Module parameter tree so load/save are unaffected.
_QKVG_FUSED: dict = {}

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    intermediate_size: int = 19968
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 202048
    sliding_window: int = 2048
    sliding_window_pattern: int = 4
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    qk_scale_factor: float = 3.87
    logit_scale: float = 0.1961161345243454
    final_logit_softcapping: float = 20.0
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False
    layer_types: Optional[List[str]] = None

    def is_global(self, layer_idx: int) -> bool:
        if self.layer_types is not None and layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx] == "full_attention"
        return (layer_idx + 1) % self.sliding_window_pattern == 0


def _rms(x: mx.array, eps: float) -> mx.array:
    dt = x.dtype
    x = x.astype(mx.float32)
    x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x.astype(dt)


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm applying ``(1 + weight)`` (weights stored raw in HF)."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_global: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.qk_eps = args.rms_norm_eps
        self.qk_scale_factor = args.qk_scale_factor
        self.scale = self.head_dim**-0.5

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        if is_global:
            self.rope = None
        else:
            self.rope = initialize_rope(
                self.head_dim, args.rope_theta, False, None, args.max_position_embeddings
            )

    def _fused_qkvg(self, x):
        """One matmul for q/k/v/gate (bit-exact vs 4 separate); split the output."""
        f = _QKVG_FUSED.get(id(self))
        if f is None:
            ms = (self.q_proj, self.k_proj, self.v_proj, self.gate_proj)
            if all(hasattr(m, "scales") for m in ms):  # quantized
                f = ("q",
                     mx.concatenate([m.weight for m in ms], axis=0),
                     mx.concatenate([m.scales for m in ms], axis=0),
                     mx.concatenate([m.biases for m in ms], axis=0),
                     int(self.q_proj.group_size), int(self.q_proj.bits))
            else:  # bf16 dense (attention_bias=False -> no bias)
                f = ("d", mx.concatenate([m.weight for m in ms], axis=0))
            _QKVG_FUSED[id(self)] = f
        if f[0] == "q":
            t = mx.quantized_matmul(x, f[1], f[2], f[3], transpose=True, group_size=f[4], bits=f[5])
        else:
            t = x @ f[1].T
        qw = self.n_heads * self.head_dim
        kw = self.n_kv_heads * self.head_dim
        return t[..., :qw], t[..., qw:qw + kw], t[..., qw + kw:qw + 2 * kw], t[..., qw + 2 * kw:]

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values, gate = self._fused_qkvg(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Parameter-free QK-norm; qk_scale_factor folded onto Q.
        queries = _rms(queries, self.qk_eps) * self.qk_scale_factor
        keys = _rms(keys, self.qk_eps)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = output * mx.sigmoid(gate)
        return self.o_proj(output)


# Optional decode-only fused dense-SwiGLU kernel (+5.2% decode, quality-parity).
# Enabled by MG_MLP_KERNEL=1; the kernel module path is MG_MLP_KERNEL_PATH.
# If either is unset or the module can't load, we fall back to stock qmm — so a
# fresh checkout without the external kernel just runs the (bit-exact) default.
_MLP_KERNEL_ENABLED = __import__("os").environ.get("MG_MLP_KERNEL") == "1"
_dense_swiglu_qmv = None
_dense_swiglu_tried = False


def _get_dense_swiglu():
    global _dense_swiglu_qmv, _dense_swiglu_tried
    if _dense_swiglu_tried:
        return _dense_swiglu_qmv
    _dense_swiglu_tried = True
    import importlib.util
    import os
    path = os.environ.get("MG_MLP_KERNEL_PATH")
    if not path or not os.path.isfile(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_mtplx_dense_mlp", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _dense_swiglu_qmv = m.dense_swiglu_qmv
    except Exception:
        _dense_swiglu_qmv = None
    return _dense_swiglu_qmv


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self._hidden = dim
        self._intermediate = hidden_dim

    def __call__(self, x) -> mx.array:
        # Optional shape-optimized fused dense-SwiGLU kernel (+5.2% decode; NOT
        # bit-exact vs stock qmm ~5.9e-3, so env-gated pending a quality gate).
        # DECODE-ONLY: it's a row-owned qmv tuned for M=1; at prefill (M>1) the
        # compute-bound large-T stock qmm wins (measured −3.9% at L=512), so gate
        # on a single flattened row and fall through to stock qmm otherwise.
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        f = (_get_dense_swiglu()
             if _MLP_KERNEL_ENABLED and x2.shape[0] == 1 and hasattr(self.gate_proj, "scales")
             else None)
        if f is not None:
            out = f(
                x2,
                self.gate_proj.weight, self.gate_proj.scales, self.gate_proj.biases,
                self.up_proj.weight, self.up_proj.scales, self.up_proj.biases,
                self.down_proj.weight, self.down_proj.scales, self.down_proj.biases,
                hidden=self._hidden, intermediate=self._intermediate,
                gate_up_bits=int(self.gate_proj.bits), down_bits=int(self.down_proj.bits),
                group_size=int(self.gate_proj.group_size))
            return out.reshape(*lead, out.shape[-1])
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, is_global=args.is_global(layer_idx))
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.post_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.post_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class MuseGlimmerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Generic residual-stream tap for external drafters (dflash): when a
        # backend sets ``_tap_layers`` to a set of layer indices, ``_taps`` is
        # populated with the residual stream at the OUTPUT of each such layer on
        # every forward. ``None`` => zero overhead, no behavior change.
        self._tap_layers = None
        self._taps: dict[int, mx.array] = {}

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        h = _rms(h, self.args.rms_norm_eps)  # RMS-normed embeddings (not ×√hidden)

        if cache is None:
            cache = [None] * len(self.layers)

        pattern = self.args.sliding_window_pattern
        global_mask = create_attention_mask(h, cache[pattern - 1])
        sliding_mask = create_attention_mask(h, cache[0], window_size=self.args.sliding_window)

        taps = self._tap_layers
        if taps is not None:
            self._taps = {}
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = global_mask if self.args.is_global(i) else sliding_mask
            h = layer(h, mask, c)
            if taps is not None and i in taps:
                self._taps[i] = h

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)

        logits = logits * self.args.logit_scale
        cap = self.args.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits

    def sanitize(self, weights):
        """Strip the multimodal wrapper: keep the text tower, drop the vision
        stack, and remap ``model.language_model.*`` -> ``model.*``."""
        out = {}
        for k, v in weights.items():
            if (
                k.startswith("model.vision_tower")
                or k.startswith("model.vision_adapter")
                or k.startswith("model.vision_projection")
            ):
                continue
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            out[k] = v
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if self.args.is_global(i):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
