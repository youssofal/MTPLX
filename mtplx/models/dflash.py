"""Generic dflash block-diffusion drafter (MLX).

A *config-driven* drafter for MTPLX speculative decoding. A "dflash" drafter is a
small Qwen3-style transformer that, given a few taps of the target model's
residual stream, proposes a whole block of `block_size` draft tokens in one
non-autoregressive forward — then MTPLX verifies them against the target.

The mechanism (authoritative reference: llama.cpp ``src/models/dflash.cpp``,
simple/Qwen3 variant — no DSpark/Markov head):

  encode:  fused = enc_norm( fc( concat of target hidden taps @ target_layers ) )
  inject:  for each drafter layer, K/V = rope(k_norm(k_proj(fused))) / v_proj(fused)
           become the *context* the block attends to (no Q, no output).
  decode:  a block of `block_size` MASK tokens (embedded via the TARGET's token
           embedding) runs through the drafter layers with *non-causal*
           attention over [injected-context ++ block], then the TARGET's
           lm_head projects the final hidden → argmax → `block_size` draft ids.

Nothing here is Muse-Glimmer specific. A future dflash drafter is added by
dropping its weights + a ``config.json`` (this shape) + a pair manifest naming
its target — zero code. Everything model-specific lives in :class:`DFlashConfig`:
the tap layer indices, the block size, GQA shape, rope, and the block-seed
token. The drafter borrows the *target's* ``tok_embd`` and ``lm_head`` (passed
in at proposal time) so it never carries a vocab-sized table of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    block_size: int = 8
    # which TARGET layer residual streams are tapped and stacked into `fc`.
    target_layers: List[int] = field(default_factory=list)
    # Glimmer's converted config stores hidden-state indices (layer output is
    # index-1); NVIDIA stores zero-based model-layer ids directly.
    target_layer_offset: int = -1
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    # encoder input width; must equal len(target_layers) * hidden_size.
    n_embd_inp_enc: int = 0
    sliding_window: int = 2048
    # token id used to seed the MASK block. ``None`` => seed each block position
    # with the last committed token (a driver-side convention; the acceptance
    # bench validates which the checkpoint was trained with).
    mask_token_id: Optional[int] = None
    max_position_embeddings: int = 1048576
    rope_scaling: Optional[dict[str, Any]] = None
    has_embed_tokens: bool = False
    causal: bool = False
    vocab_size: int = 0
    quantization: Optional[dict[str, Any]] = None
    model_type: str = "dflash"

    def __post_init__(self):
        if not self.n_embd_inp_enc:
            self.n_embd_inp_enc = len(self.target_layers) * self.hidden_size

    @classmethod
    def from_dict(cls, d: dict) -> "DFlashConfig":
        d = dict(d)
        dflash = d.get("dflash_config") or {}
        rope = dict(d.get("rope_parameters") or d.get("rope_scaling") or {})
        if "rope_type" in rope and "type" not in rope:
            rope["rope_type"] = rope["rope_type"]
        d.setdefault("target_layers", d.get("target_layer_ids") or dflash.get("target_layer_ids") or [])
        if "target_layer_ids" in d or "eagle_aux_hidden_state_layer_ids" in d:
            d.setdefault("target_layer_offset", 0)
        d.setdefault("mask_token_id", dflash.get("mask_token_id"))
        d.setdefault("causal", bool(dflash.get("causal", False)))
        d.setdefault("rope_theta", rope.pop("rope_theta", d.get("rope_theta", 10000.0)))
        d.setdefault("rope_scaling", rope or None)
        keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in keys})


class _RMSNorm(nn.Module):
    """Standard (Qwen3) RMSNorm — learned weight, NOT Gemma (1+w)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class _Attention(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        d, H, KV, hd = (cfg.hidden_size, cfg.num_attention_heads,
                        cfg.num_key_value_heads, cfg.head_dim)
        self.n_heads, self.n_kv, self.head_dim = H, KV, hd
        self.repeat = H // KV
        self.scale = hd ** -0.5
        self.q_proj = nn.Linear(d, H * hd, bias=False)
        self.k_proj = nn.Linear(d, KV * hd, bias=False)
        self.v_proj = nn.Linear(d, KV * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, d, bias=False)
        self.q_norm = _RMSNorm(hd, cfg.rms_norm_eps)
        self.k_norm = _RMSNorm(hd, cfg.rms_norm_eps)
        self.rope = initialize_rope(
            hd,
            cfg.rope_theta,
            False,
            scaling_config=cfg.rope_scaling,
            max_position_embeddings=cfg.max_position_embeddings,
        )

    def _kv(self, feats: mx.array, offset: int) -> tuple[mx.array, mx.array]:
        """Project `feats` [T, d] → per-head K/V with k_norm + rope. [1,KV,T,hd]."""
        T = feats.shape[0]
        k = self.k_norm(self.k_proj(feats).reshape(T, self.n_kv, self.head_dim))
        v = self.v_proj(feats).reshape(T, self.n_kv, self.head_dim)
        k = k.transpose(1, 0, 2)[None]  # [1, KV, T, hd]
        v = v.transpose(1, 0, 2)[None]
        k = self.rope(k, offset=offset)
        return k, v

    def __call__(self, h: mx.array, ctx_k: mx.array, ctx_v: mx.array,
                 ctx_len: int, block_offset: int) -> mx.array:
        """`h` [Tblk, d] is the normed block. Attends non-causally over
        [injected context ++ block]. Returns o_proj output [Tblk, d]."""
        Tb = h.shape[0]
        q = self.q_norm(self.q_proj(h).reshape(Tb, self.n_heads, self.head_dim))
        q = q.transpose(1, 0, 2)[None]  # [1, H, Tb, hd]
        q = self.rope(q, offset=block_offset)
        bk, bv = self._kv(h, block_offset)                     # block's own K/V
        k = mx.concatenate([ctx_k, bk], axis=2)                # [1,KV,ctx+Tb,hd]
        v = mx.concatenate([ctx_v, bv], axis=2)
        # full (non-causal) attention: block attends to all context + all block.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        out = out[0].transpose(1, 0, 2).reshape(Tb, self.n_heads * self.head_dim)
        return self.o_proj(out)


class _MLP(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.self_attn = _Attention(cfg)
        self.mlp = _MLP(cfg)
        self.attn_norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def inject(self, fused: mx.array, offset: int) -> tuple[mx.array, mx.array]:
        """Precompute this layer's injected context K/V from the fused feature."""
        return self.self_attn._kv(fused, offset)

    def __call__(self, x: mx.array, ctx_k, ctx_v, ctx_len, block_offset) -> mx.array:
        r = self.self_attn(self.attn_norm(x), ctx_k, ctx_v, ctx_len, block_offset)
        x = x + r
        return x + self.mlp(self.ffn_norm(x))


class DFlashDrafter(nn.Module):
    """Config-driven dflash drafter. Weights map 1:1 to the checkpoint keys
    (fc, enc_norm, layers.N.{attn_norm,ffn_norm,self_attn.*,mlp.*}, norm)."""

    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.n_embd_inp_enc, cfg.hidden_size, bias=False)
        if cfg.has_embed_tokens:
            if cfg.vocab_size <= 0:
                raise ValueError("dflash has_embed_tokens requires vocab_size")
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.enc_norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = [_Layer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    # --- encode: stacked target taps -> fused context feature ---------------
    def encode(self, taps: List[mx.array]) -> mx.array:
        """`taps` is a list of target residual-stream tensors (one per
        config.target_layers), each [T, hidden]. Returns fused [T, hidden]."""
        if len(taps) != len(self.cfg.target_layers):
            raise ValueError(
                f"dflash expects {len(self.cfg.target_layers)} taps "
                f"(target_layers={self.cfg.target_layers}), got {len(taps)}")
        stacked = mx.concatenate([t.astype(mx.float32) for t in taps], axis=-1)
        return self.enc_norm(self.fc(stacked.astype(self.fc.weight.dtype)))

    # --- block logits: one non-causal block-decode forward -------------------
    def block_logits(
        self,
        fused: mx.array,
        target_tok_embd: Callable[[mx.array], mx.array],
        target_lm_head: Callable[[mx.array], mx.array],
        block_ids: mx.array,
        embed_scale: float = 1.0,
    ) -> mx.array:
        """One forward over `block_ids` [Tblk], attending non-causally over
        [context ++ block]. Cache-relative positions (matching the reference):
        the fused context sits at offset 0, the block at offset Tctx. Returns
        TARGET-projected logits [Tblk, vocab]."""
        Tctx = fused.shape[0]
        ctx_kv = [layer.inject(fused, 0) for layer in self.layers]   # context @ 0
        embed = self.embed_tokens if self.cfg.has_embed_tokens else target_tok_embd
        x = embed(block_ids).astype(mx.float32) * embed_scale
        x = x.astype(self.norm.weight.dtype)
        for layer, (ck, cv) in zip(self.layers, ctx_kv):
            x = layer(x, ck, cv, Tctx, Tctx)                          # block @ Tctx
        return target_lm_head(self.norm(x))

    # --- propose: dflash single-forward block draft --------------------------
    def propose_block(
        self,
        fused: mx.array,
        target_tok_embd: Callable[[mx.array], mx.array],
        target_lm_head: Callable[[mx.array], mx.array],
        primary_token_id: int,
        mask_token_id: int,
        block_size: Optional[int] = None,
        embed_scale: float = 1.0,
    ) -> mx.array:
        """dflash proposal (matches bstnxbt/dflash-mlx ``draft_greedy``): seed the
        block with the KNOWN primary token at position 0 and MASK for the rest,
        run ONE non-autoregressive forward, and read the TARGET-projected argmax
        at positions 1: — the ``block_size - 1`` speculative tokens after the
        primary. Not iterative; the "diffusion" is this single masked pass."""
        k = block_size or self.cfg.block_size
        block_ids = mx.concatenate([
            mx.array([int(primary_token_id)], dtype=mx.int32),
            mx.full((k - 1,), int(mask_token_id), dtype=mx.int32),
        ])
        logits = self.block_logits(fused, target_tok_embd, target_lm_head,
                                   block_ids, embed_scale)
        return mx.argmax(logits[1:], axis=-1).astype(mx.int32)        # drop pos 0

    # ---- incremental context cache (avoids re-encoding every round) ---------
    # The drafter's context is the projected+injected K/V of every committed
    # position. Re-encoding all of it each round is O(context) and dominates the
    # per-round cost. Instead cache per-layer (K,V) and only inject the new
    # committed positions — "lockstep" drafting with parallel weight reads.
    def init_context_cache(self) -> list:
        return [[None, None] for _ in self.layers]

    def extend_context(self, ctx_cache: list, new_taps: List[mx.array], offset: int) -> list:
        """Inject `new_taps` (list per target_layer, each [new, hidden]) at rope
        `offset` into the per-layer context K/V cache. Returns the cache."""
        fused = self.encode(new_taps)                     # [new, hidden]
        for i, layer in enumerate(self.layers):
            nk, nv = layer.self_attn._kv(fused, offset)   # [1, KV, new, hd] roped @offset
            ck, cv = ctx_cache[i]
            if ck is None:
                ctx_cache[i] = [nk, nv]
            else:
                ctx_cache[i] = [mx.concatenate([ck, nk], axis=2),
                                mx.concatenate([cv, nv], axis=2)]
        return ctx_cache

    def propose_block_cached(
        self,
        ctx_cache: list,
        ctx_len: int,
        target_tok_embd: Callable[[mx.array], mx.array],
        target_lm_head: Callable[[mx.array], mx.array],
        primary_token_id: int,
        mask_token_id: int,
        block_size: Optional[int] = None,
        embed_scale: float = 1.0,
    ) -> mx.array:
        """Same proposal as :meth:`propose_block` but attends to the cached
        context K/V (block at offset ``ctx_len``). Bit-identical to the
        re-encoding path — it only skips recomputing the context."""
        k = block_size or self.cfg.block_size
        block_ids = mx.concatenate([
            mx.array([int(primary_token_id)], dtype=mx.int32),
            mx.full((k - 1,), int(mask_token_id), dtype=mx.int32),
        ])
        embed = self.embed_tokens if self.cfg.has_embed_tokens else target_tok_embd
        x = embed(block_ids).astype(mx.float32) * embed_scale
        x = x.astype(self.norm.weight.dtype)
        for layer, (ck, cv) in zip(self.layers, ctx_cache):
            x = layer(x, ck, cv, ctx_len, ctx_len)
        logits = target_lm_head(self.norm(x))
        return mx.argmax(logits[1:], axis=-1).astype(mx.int32)


def normalize_dflash_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map NVIDIA's DFlash module names onto the original Glimmer adapter tree."""
    replacements = (
        ("hidden_norm.", "enc_norm."),
        (".input_layernorm.", ".attn_norm."),
        (".post_attention_layernorm.", ".ffn_norm."),
    )
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        normalized = key
        for old, new in replacements:
            normalized = normalized.replace(old, new)
        out[normalized] = value
    return out


def load_dflash(path: str) -> tuple[DFlashDrafter, DFlashConfig]:
    """Load a dflash drafter from a directory holding config.json +
    model.safetensors (weights keyed to match the module tree)."""
    import json
    import os

    raw_config = json.load(open(os.path.join(path, "config.json")))
    cfg = DFlashConfig.from_dict(raw_config)
    model = DFlashDrafter(cfg)
    quantization = raw_config.get("quantization")
    if isinstance(quantization, dict):
        def quant_predicate(module_path, _module):
            return quantization.get(module_path, False)

        nn.quantize(
            model,
            group_size=64,
            bits=4,
            mode="affine",
            class_predicate=quant_predicate,
        )
    weights = normalize_dflash_weights(mx.load(os.path.join(path, "model.safetensors")))
    model.load_weights(list(weights.items()))
    model.eval()
    return model, cfg
