# Copyright © 2026 MTPLX.
#
# Qwen4-Exp (Qwen3.8-Flash-Next) — MTPLX-owned MLX backend.
#
# The pinned mlx-lm has no implementation for model_type "qwen4_exp"
# (Qwen4ExpForConditionalGeneration). This module implements the text trunk
# natively, reusing the pinned mlx-lm building blocks where the architecture
# genuinely overlaps (GatedDeltaNet, the qwen3_next MoE block) and adding the
# four genuinely new pieces:
#
#   * Gated Residual ("hyper-connections"): hc_count widened residual streams
#     with a learned low-rank read mix and per-stream scalar write gates.
#     There are NO input/post-attention layernorms and no final model.norm in
#     this family — the per-block hc_norm and the final hyper_connection_mixer
#     play those roles.
#   * QSA (Qwen Sparse Attention): standard gated GQA whose causal mask is
#     intersected with a per-query token selection produced by a
#     DeepSeek-V3.2-class indexer (relu-scored mean-pooled key blocks,
#     top-(budget/ratio) blocks + the incomplete tail block).
#   * PLE (Per-Layer Embedding): a hashed n-gram lookup memory (~51B params,
#     320M rows x 160) injected on one early linear-attention layer through a
#     per-stream sigmoid gate and a dilated depthwise convolution. The table
#     is deliberately NEVER materialized: it stays an SSD-resident sidecar
#     (ngram-table.safetensors) gathered row-wise through numpy memmaps, so
#     the OS page cache is the hot-row cache.
#   * mrope carried by the family config; for text-only serving with equal
#     t/h/w positions the interleaved mrope is numerically identical to the
#     standard partial rotary embedding, which is what this module applies
#     (same treatment the pinned mlx-lm gives qwen3_5).
#
# Reference: transformers' modular_qwen4_exp.py (read 2026-08-26, T+9h after
# the weight drop). Norm convention: the Qwen4ExpTextRMSNorm family is stored
# zero-centered ((1+w) convention) in HF checkpoints and shifted by +1.0 in
# sanitize; the GDN gated norm is stored one-centered and is NOT shifted.

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx

from mtplx.progress_heartbeat import tick as _owner_progress_tick
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.attention_context import vision_rope_state
from mlx_lm.models.qwen3_5 import GatedDeltaNet as _Qwen3_5GatedDeltaNet
from mlx_lm.models.qwen3_next import (
    Qwen3NextSparseMoeBlock as _Qwen3NextSparseMoeBlock,
)

from mtplx.attention_context import current_attention_phase
from mtplx.runtime_options import qwen4_opdiet_enabled, qwen4_verify_glue_enabled


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4

    # GatedDeltaNet (names shared with qwen3_5 so the mlx-lm module reads them)
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"

    # MoE (names shared with qwen3_next's SparseMoeBlock)
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1

    # Gated Residual / hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320

    # QSA indexer
    indexer_n_heads: Optional[int] = 4
    indexer_kv_heads: Optional[int] = 1
    indexer_head_dim: Optional[int] = 128
    indexer_budget: Optional[int] = 2048
    indexer_compress_ratio: Optional[int] = 4

    # PLE / n-gram embedding
    ple_layer_ids: Optional[List[int]] = None  # ONE-indexed, per the HF config
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    eos_token_id: Union[int, List[int], None] = None
    # True in MTPLX packs: the table ships as ngram-table.safetensors and is
    # gathered lazily from SSD — no weight parameter is ever constructed.
    ngram_sidecar: bool = False

    # Rope
    rope_parameters: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    # M-RoPE contract (vision): per-axis frequency counts over the rotary
    # pairs and the interleaved layout flag. Text-only requests are exactly
    # plain rope under it (equal t/h/w axes), so these only change behavior
    # when a vision request supplies a position table.
    mrope_section: Optional[list] = None
    mrope_interleaved: bool = False

    def __post_init__(self):
        if self.rope_parameters:
            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            section = self.rope_parameters.get("mrope_section")
            if isinstance(section, (list, tuple)) and section:
                self.mrope_section = [int(x) for x in section]
            self.mrope_interleaved = bool(
                self.rope_parameters.get("mrope_interleaved", self.mrope_interleaved)
            )
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (i + 1) % self.full_attention_interval
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        # The shipped config says "full_attention"; those layers carry the
        # indexer whenever the QSA fields are set.
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def eos_id(self) -> int:
        eos = self.eos_token_id
        if isinstance(eos, list):
            return int(eos[0])
        return int(eos if eos is not None else 0)


def _rope_inv_freq_and_scaling(args: TextArgs) -> tuple[mx.array, float]:
    """Build the exact Transformers RoPE parameters used by Qwen4-Exp.

    The released checkpoint is native at 262,144 tokens.  Qwen's documented
    one-million-token configuration switches ``rope_type`` to static YaRN.
    Static matters here: the same corrected frequencies and attention scale
    apply at every position, including short rows in a long-context run.
    """

    rotary_dim = int(args.rotary_dim)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be a positive even integer, got {rotary_dim}")

    parameters = args.rope_parameters or {}
    rope_type = str(parameters.get("rope_type") or "default").strip().lower()
    base = float(parameters.get("rope_theta", args.rope_theta))
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError(f"rope_theta must be finite and greater than 1, got {base}")

    positions = mx.arange(0, rotary_dim, 2, dtype=mx.float32)
    position_frequencies = base ** (positions / rotary_dim)
    if rope_type == "default":
        return 1.0 / position_frequencies, 1.0
    if rope_type != "yarn":
        raise ValueError(
            "Qwen4-Exp supports rope_type 'default' and static 'yarn'; "
            f"got {rope_type!r}"
        )

    try:
        factor = float(parameters["factor"])
        original_max = int(parameters["original_max_position_embeddings"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "static YaRN requires numeric factor and "
            "original_max_position_embeddings"
        ) from exc
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError(f"YaRN factor must be finite and >= 1, got {factor}")
    if original_max <= 0:
        raise ValueError(
            "YaRN original_max_position_embeddings must be positive, "
            f"got {original_max}"
        )

    def mscale(scale: float, multiplier: float = 1.0) -> float:
        if scale <= 1.0:
            return 1.0
        return 0.1 * multiplier * math.log(scale) + 1.0

    configured_attention_scale = parameters.get("attention_factor")
    if configured_attention_scale is None:
        mscale_value = parameters.get("mscale")
        mscale_all_dim = parameters.get("mscale_all_dim")
        if mscale_value and mscale_all_dim:
            attention_scaling = mscale(factor, float(mscale_value)) / mscale(
                factor,
                float(mscale_all_dim),
            )
        else:
            attention_scaling = mscale(factor)
    else:
        attention_scaling = float(configured_attention_scale)
    if not math.isfinite(attention_scaling) or attention_scaling <= 0.0:
        raise ValueError(
            "YaRN attention_factor must be finite and positive, "
            f"got {attention_scaling}"
        )

    beta_fast = float(parameters.get("beta_fast") or 32.0)
    beta_slow = float(parameters.get("beta_slow") or 1.0)
    if beta_fast < beta_slow:
        raise ValueError(
            f"YaRN beta_fast must be >= beta_slow, got {beta_fast} < {beta_slow}"
        )

    def correction_dimension(rotations: float) -> float:
        return (
            rotary_dim
            * math.log(original_max / (rotations * 2.0 * math.pi))
            / (2.0 * math.log(base))
        )

    low = correction_dimension(beta_fast)
    high = correction_dimension(beta_slow)
    if bool(parameters.get("truncate", True)):
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0.0)
    high = min(high, float(rotary_dim - 1))
    if low == high:
        high += 0.001

    ramp = mx.clip(
        (mx.arange(rotary_dim // 2, dtype=mx.float32) - low) / (high - low),
        0.0,
        1.0,
    )
    inv_freq_extrapolation = 1.0 / position_frequencies
    inv_freq_interpolation = inv_freq_extrapolation / factor
    extrapolation_factor = 1.0 - ramp
    inv_freq = (
        inv_freq_interpolation * (1.0 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    return inv_freq.astype(mx.float32), float(attention_scaling)


def _rope_cos_sin(
    positions: mx.array,
    inv_freq: mx.array,
    attention_scaling: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Non-interleaved RoPE tables, including static-YaRN amplitude scaling."""

    angles = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([angles, angles], axis=-1)
    cosine = mx.cos(emb)
    sine = mx.sin(emb)
    if attention_scaling != 1.0:
        cosine = cosine * float(attention_scaling)
        sine = sine * float(attention_scaling)
    return cosine, sine


def _build_mrope_axes(section: list, interleaved: bool) -> list[int]:
    """Per-frequency axis assignment (0=t, 1=h, 2=w) over the rotary pairs.

    Interleaved layout is round-robin t,h,w while each axis has budget left
    (Qwen3.8-Flash-Next [11,11,10] -> t@0,3..30, h@1,4..31, w@2,5..29);
    non-interleaved is contiguous section blocks. Matches the reference
    Qwen-VL family layout (mlx-vlm / transformers).
    """
    remaining = [int(x) for x in section]
    axes: list[int] = []
    if interleaved:
        axis = 0
        while sum(remaining) > 0:
            if remaining[axis] > 0:
                axes.append(axis)
                remaining[axis] -= 1
            axis = (axis + 1) % len(remaining)
    else:
        for axis, count in enumerate(remaining):
            axes.extend([axis] * count)
    return axes


def _mrope_cos_sin(
    positions3: mx.array, inv_freq: mx.array, axes: mx.array
) -> tuple[mx.array, mx.array]:
    """Rope tables for 3-axis (t, h, w) positions.

    ``positions3`` is [3, S] int32; ``axes`` maps each of the len(inv_freq)
    frequencies to its position axis. With equal axes this reduces exactly
    to ``_rope_cos_sin`` (the text case), which is why text-only serving
    never needs it.
    """
    pos = mx.take(positions3.astype(mx.float32), axes, axis=0)  # [F, S]
    angles = pos.transpose(1, 0) * inv_freq[None, :]  # [S, F]
    emb = mx.concatenate([angles, angles], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _apply_partial_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the first `2 * inv_freq.size` features of the last axis of
    x[..., S, H, D] with per-position tables cos/sin of shape [S, rot]."""
    rot = cos.shape[-1]
    x_rope = x[..., :rot]
    x_pass = x[..., rot:]
    half = rot // 2
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    x_rope = (x_rope.astype(mx.float32) * cos + rotated.astype(mx.float32) * sin).astype(
        x.dtype
    )
    return mx.concatenate([x_rope, x_pass], axis=-1)


# ---------------------------------------------------------------------------
# MTPLX_QWEN4_OPDIET - exact-preserving op diet for the compiled verifier.
#
# Every helper below is a VALUE-IDENTICAL twin of the expression it replaces:
# same arithmetic on the same operands in the same order, only the op graph is
# smaller. Nothing here is reachable with the flag off.
# ---------------------------------------------------------------------------


def _rope_cos_sin_half(
    positions: mx.array,
    inv_freq: mx.array,
    attention_scaling: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """``_rope_cos_sin`` without the duplicated half.

    ``_rope_cos_sin`` builds ``emb = concatenate([angles, angles])`` and takes
    cos/sin of the doubled table. Both halves are therefore bit-identical
    (elementwise cos of the same numbers), so the second half is pure copy +
    transcendental work: two concatenate copies and 2x the cos/sin width per
    call. ``_apply_partial_rope_half`` consumes the [S, rot // 2] table
    directly.
    """

    angles = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    cosine = mx.cos(angles)
    sine = mx.sin(angles)
    if attention_scaling != 1.0:
        cosine = cosine * float(attention_scaling)
        sine = sine * float(attention_scaling)
    return cosine, sine


def _apply_partial_rope_half(
    x: mx.array, cos_h: mx.array, sin_h: mx.array
) -> mx.array:
    """``_apply_partial_rope`` on half-width tables, bitwise-identically.

    The stock form materializes ``rotated = concatenate([-x2, x1])`` (a
    standalone negate plus two copies) so the rotation can be written as one
    full-width multiply-add. Splitting the multiply-add per half removes both:
    the negate folds into the fused elementwise kernel and the rotated buffer
    never exists. Per element the arithmetic is unchanged --
    ``lo = x1*cos + (-x2)*sin`` and ``hi = x2*cos + x1*sin`` are exactly the
    two halves the full-width expression computes, and ``cos``/``sin`` repeat
    across the halves.
    """

    half = cos_h.shape[-1]
    rot = 2 * half
    x_rope = x[..., :rot]
    x_pass = x[..., rot:]
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]
    cos_h = cos_h[:, None, :]
    sin_h = sin_h[:, None, :]
    lo = (x1.astype(mx.float32) * cos_h + (-x2).astype(mx.float32) * sin_h).astype(
        x.dtype
    )
    hi = (x2.astype(mx.float32) * cos_h + x1.astype(mx.float32) * sin_h).astype(
        x.dtype
    )
    if x_pass.shape[-1] == 0:
        return mx.concatenate([lo, hi], axis=-1)
    return mx.concatenate([lo, hi, x_pass], axis=-1)


#: Per-forward RoPE table memo. The tables depend only on (pos_start, S,
#: inv_freq, scaling); a QSA layer builds the same one twice (indexer queries
#: and attention q/k) and, whenever ``pos_start`` is a plain int, every QSA
#: layer of the forward builds the same one again. Keyed on OBJECT IDENTITY of
#: the two array operands (never on values, which would need a device sync)
#: and the memo keeps them alive so an id can never be recycled underneath it.
_ROPE_TABLE_MEMO: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "mtplx_rope_table_memo", default=None
)


@contextlib.contextmanager
def _rope_table_scope():
    """Scope one forward's RoPE table memo (trace-time only under compile)."""

    token = _ROPE_TABLE_MEMO.set({})
    try:
        yield
    finally:
        _ROPE_TABLE_MEMO.reset(token)


def _shared_rope_cos_sin_half(
    pos_start,
    length: int,
    inv_freq: mx.array,
    attention_scaling: float,
) -> tuple[mx.array, mx.array]:
    """Half-width RoPE tables for ``pos_start + arange(length)``, memoized."""

    memo = _ROPE_TABLE_MEMO.get()
    if isinstance(pos_start, mx.array):
        start_key = ("array", id(pos_start))
    else:
        start_key = ("int", int(pos_start))
    key = (start_key, int(length), id(inv_freq), float(attention_scaling))
    if memo is not None:
        hit = memo.get(key)
        if hit is not None:
            return hit[0]
    positions = pos_start + mx.arange(length, dtype=mx.int32)
    tables = _rope_cos_sin_half(positions, inv_freq, attention_scaling)
    if memo is not None:
        # Hold the keyed objects so their ids stay unique for the scope.
        memo[key] = (tables, pos_start, inv_freq)
    return tables


#: One ``_rope_inv_freq_and_scaling`` result per TextArgs object. Sharing the
#: array OBJECT across the indexer and the attention of every layer is what
#: lets the identity-keyed table memo hit; the values were already identical
#: (same pure function, same args), so nothing numeric changes.
_INV_FREQ_MEMO: dict[int, tuple[Any, tuple[mx.array, float]]] = {}


def _rope_inv_freq_and_scaling_shared(args: TextArgs) -> tuple[mx.array, float]:
    cached = _INV_FREQ_MEMO.get(id(args))
    if cached is not None and cached[0] is args:
        return cached[1]
    value = _rope_inv_freq_and_scaling(args)
    _INV_FREQ_MEMO[id(args)] = (args, value)
    return value


def _rope_inv_freq_and_scaling_for(args: TextArgs) -> tuple[mx.array, float]:
    if qwen4_opdiet_enabled("rope"):
        return _rope_inv_freq_and_scaling_shared(args)
    return _rope_inv_freq_and_scaling(args)


def _hyper_residual_write(
    hyper: mx.array, block_out: mx.array, inject: mx.array
) -> mx.array:
    """``hyper + (block_out[..., None, :] * inject[..., :, None])``.

    The stock spelling reshapes the broadcast product back to ``hyper``'s
    flat [.., hc * hidden] layout before adding, and that reshape sits between
    two elementwise ops, so mx.compile cannot fuse them: the product is
    materialized (one kernel) and then added (a second kernel). Adding on the
    [.., hc, hidden] VIEW of ``hyper`` instead -- a free reshape of a
    contiguous array on both sides -- lets the multiply and the add fuse into
    one kernel. Same operands, same order, same result.
    """

    if not qwen4_opdiet_enabled("resid"):
        return hyper + (block_out[..., None, :] * inject[..., :, None]).reshape(
            *hyper.shape
        )
    grouped = hyper.reshape(
        *hyper.shape[:-1], inject.shape[-1], block_out.shape[-1]
    )
    return (
        grouped + block_out[..., None, :] * inject[..., :, None]
    ).reshape(*hyper.shape)


class GroupedRMSNorm(nn.Module):
    """RMSNorm normalized per contiguous group of `group_size` features, with a
    full-width weight. Used by every hc_norm and the PLE norms (weight arrives
    +1-shifted from sanitize)."""

    def __init__(self, dims: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        if dims % group_size:
            raise ValueError(f"dims ({dims}) not divisible by group_size ({group_size})")
        self.weight = mx.ones((dims,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        grouped = x.reshape(*shape[:-1], -1, self.group_size)
        normed = mx.fast.rms_norm(grouped, None, self.eps)
        return normed.reshape(shape) * self.weight


class SigmoidRMSNormGated(nn.Module):
    """GDN output norm with a sigmoid (not silu) gate — output_gate_type of
    this family. Stored one-centered; NOT +1-shifted in sanitize."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array, gate: Optional[mx.array] = None):
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is None:
            return x.astype(hidden_states.dtype)
        g = mx.sigmoid(gate.astype(mx.float32))
        return (g * x.astype(mx.float32)).astype(hidden_states.dtype)


class GatedDeltaNet(_Qwen3_5GatedDeltaNet):
    """qwen3_5's GDN with the family's output gate activation (sigmoid) and
    the reference q/k normalization.

    mlx-lm folds the attention scale through mx.fast.rms_norm, whose eps sits
    on mean(x²) — an effective d²·1e-6 on Σx² versus the reference FLA
    l2norm's d·1e-6 (transformers qwen3_5 l2norm: x·rsqrt(Σx²+1e-6)). At
    d=128 that skew is a measured, systematic ~1e-4-class divergence per
    layer (pinned by CPU-exact stage bisection, 2026-08-26), so this forward
    is mlx-lm's verbatim except the two q/k lines reproduce l2norm exactly.
    """

    def __init__(self, args: TextArgs):
        super().__init__(args)
        if getattr(args, "output_gate_type", "sigmoid") == "sigmoid":
            self.norm = SigmoidRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        from mlx_lm.models.gated_delta import gated_delta_update

        B, S, _ = inputs.shape

        fused_in = getattr(self, "in_proj_fused", None)
        if fused_in is not None:
            qkv, z, b, a = fused_in(inputs)
            z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        else:
            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        if self._fused_step_applies(B, S, mask, cache):
            # One-dispatch GDN step: conv+silu+l2norm + g/beta + delta +
            # gated norm in a single kernel between the two library GEMVs.
            # Verify rows (S>1) never take this branch, so capture-commit's
            # stash contract below is untouched.
            from mtplx.kernels.gdn_step_fused import fused_gdn_step

            y, new_conv, new_delta = fused_gdn_step(
                qkv.reshape(-1),
                z.reshape(-1),
                a.reshape(-1),
                b.reshape(-1),
                conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                self.conv1d.weight,
                self.A_log,
                self.dt_bias,
                cache[1],
                self.norm.weight,
            )
            cache[0] = new_conv.reshape(B, self.conv_kernel_size - 1, self.conv_dim)
            cache[1] = new_delta.reshape(
                B, self.num_v_heads, self.head_v_dim, self.head_k_dim
            )
            cache.advance(S)
            return self.out_proj(y.reshape(B, S, -1))
        if self._fused_conv_norm_applies(B, S, mask, cache):
            from mtplx.kernels.gdn_conv_norm import fused_gdn_conv_norm

            q_f, k_f, v_f, new_state = fused_gdn_conv_norm(
                qkv.reshape(-1),
                conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                self.conv1d.weight,
            )
            cache[0] = new_state.reshape(B, self.conv_kernel_size - 1, self.conv_dim)
            q = q_f.reshape(B, S, self.num_k_heads, self.head_k_dim)
            k = k_f.reshape(B, S, self.num_k_heads, self.head_k_dim)
            v = v_f.reshape(B, S, self.num_v_heads, self.head_v_dim)
            state = cache[1] if cache else None
        elif self._fused_conv_norm_rows_applies(B, S, mask, cache):
            from mtplx.kernels.gdn_conv_norm import fused_gdn_conv_norm_rows

            q_f, k_f, v_f, new_state = fused_gdn_conv_norm_rows(
                qkv.reshape(S, -1),
                conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                self.conv1d.weight,
            )
            cache[0] = new_state.reshape(B, self.conv_kernel_size - 1, self.conv_dim)
            q = q_f.reshape(B, S, self.num_k_heads, self.head_k_dim)
            k = k_f.reshape(B, S, self.num_k_heads, self.head_k_dim)
            v = v_f.reshape(B, S, self.num_v_heads, self.head_v_dim)
            state = cache[1] if cache else None
        else:
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if cache is not None:
                n_keep = self.conv_kernel_size - 1
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, S)
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
                else:
                    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
            conv_out = nn.silu(self.conv1d(conv_input))

            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                )
            ]

            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5

            def _l2norm(x: mx.array) -> mx.array:
                xf = x.astype(mx.float32)
                return (
                    xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) + 1e-6)
                ).astype(x.dtype)

            q = inv_scale * _l2norm(q)
            k = _l2norm(k)

        if cache is not None and _VERIFY_CAPTURE.get():
            # Family capture-commit: retain the exact rows gated_delta_update
            # consumed (plus the pre-conv stream for the conv-state tail) so a
            # rejected speculative window commits by replaying ONLY this
            # recurrence from the pre-verify state — no trunk re-forward.
            # These references are already materialized by this forward; at
            # mx.compile trace time they are tracers the compiled step
            # surfaces as extra outputs.
            cache._mtplx_verify_rows = (qkv, q, k, v, a, b)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        if self._fused_out_applies(B, S):
            from mtplx.kernels.gdn_out_fused import fused_gdn_out

            proj = self.out_proj
            y = fused_gdn_out(
                out.reshape(-1),
                z.reshape(-1),
                self.norm.weight,
                proj.weight,
                proj.scales,
                proj.biases,
                group_size=int(proj.group_size),
            )
            return y.reshape(B, S, -1)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    def _fused_conv_norm_applies(self, B, S, mask, cache) -> bool:
        # Fused conv+silu+l2norm (MTPLX_FUSED_GDN_CONVNORM): the decode-row
        # chain between the input GEMV and gated_delta_update. Family
        # geometry only (conv_dim 10240 / key_dim 2048 / heads of 128 — the
        # kernel's TG alignment depends on 2*key_dim being a threadgroup
        # multiple), dense rows, no conv bias, no ragged lengths. bf16
        # rounding happens after the norm instead of before (tolerance
        # class, same as the fallback re-forward's own noise).
        if B != 1 or S != 1 or mask is not None or cache is None:
            return False
        if not _fused_gdn_conv_norm_enabled() or self.training:
            return False
        if getattr(cache, "lengths", None) is not None:
            return False
        if self.conv_dim != 10240 or self.key_dim != 2048:
            return False
        if self.conv_kernel_size != 4 or self.head_k_dim != 128:
            return False
        if getattr(self.conv1d, "bias", None) is not None:
            return False
        from mtplx.kernels.gdn_conv_norm import device_supports_gdn_conv_norm

        # G14-class GPUs cap this 1024-thread pipeline below 1024 and the
        # dispatch raises — pack contracts arm the env, so the device gate
        # must sit here (issue #400). Cached one-shot probe.
        return device_supports_gdn_conv_norm()

    def _fused_conv_norm_rows_applies(self, B, S, mask, cache) -> bool:
        # Verify-width conv+silu+l2norm (MTPLX_FUSED_CONVNORM_VERIFY): the
        # same chain the S=1 kernel replaces, for speculative verify blocks
        # of 2..6 sequential rows. Deliberately ALLOWED under the capture
        # scope — the kernel produces exactly the q/k/v rows the
        # capture-commit stash retains, in the S=1 kernel's tolerance class.
        # The recurrence stays in the library gated_delta_update dispatch.
        if B != 1 or S < 2 or S > 6 or mask is not None or cache is None:
            return False
        if not _fused_conv_norm_rows_enabled() or self.training:
            return False
        if getattr(cache, "lengths", None) is not None:
            return False
        if self.conv_dim != 10240 or self.key_dim != 2048:
            return False
        if self.conv_kernel_size != 4 or self.head_k_dim != 128:
            return False
        if getattr(self.conv1d, "bias", None) is not None:
            return False
        from mtplx.kernels.gdn_conv_norm import (
            device_supports_gdn_conv_norm_rows,
        )

        # Same G14 device gate as the S=1 kernel (issue #400).
        return device_supports_gdn_conv_norm_rows()

    def _fused_step_applies(self, B, S, mask, cache) -> bool:
        # One-dispatch GDN step (MTPLX_FUSED_GDN_STEP): decode rows only,
        # family geometry, sigmoid-gated norm, live fp32 delta state. Mirrors
        # _fused_conv_norm_applies plus the recurrence/epilogue requirements;
        # anything else runs the staged chain.
        if B != 1 or S != 1 or mask is not None or cache is None:
            return False
        if not _fused_gdn_step_enabled() or self.training:
            return False
        if _VERIFY_CAPTURE.get():
            return False
        if getattr(cache, "lengths", None) is not None:
            return False
        if cache[1] is None:
            return False
        if self.conv_dim != 10240 or self.key_dim != 2048:
            return False
        if self.conv_kernel_size != 4 or self.head_k_dim != 128:
            return False
        if self.num_v_heads != 48 or self.head_v_dim != 128 or self.num_k_heads != 16:
            return False
        if getattr(self.conv1d, "bias", None) is not None:
            return False
        if not isinstance(self.norm, SigmoidRMSNormGated):
            return False
        return cache[1].dtype == mx.float32

    def _fused_out_applies(self, B: int, S: int) -> bool:
        # Fused norm+gate+out_proj (MTPLX_FUSED_GDN_OUT): decode rows only,
        # family geometry (48x128 values -> 2560), 4-bit affine out_proj at a
        # shipped forge group size, sigmoid-gated norm. Anything else runs
        # the stock chain. The capture-commit stash is upstream of this
        # boundary (it retains the gated_delta_update INPUTS), so the fused
        # output path is invisible to replay.
        if B * S != 1 or not _fused_gdn_out_enabled():
            return False
        if self.num_v_heads != 48 or self.head_v_dim != 128:
            return False
        if not isinstance(self.norm, SigmoidRMSNormGated):
            return False
        proj = self.out_proj
        return (
            getattr(proj, "bits", None) == 4
            and getattr(proj, "group_size", None) in (32, 64)
            and getattr(proj, "weight", None) is not None
            and proj.weight.dtype == mx.uint32
        )


class GatedResidual(nn.Module):
    """The Gated Residual read/write mixer (hyper-connections)."""

    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = GroupedRMSNorm(hc_hidden, args.hidden_size, eps=args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden, self.hc_count, bias=False)

    def _fused_read_applies(self, hyper_input: mx.array) -> bool:
        # The fused kernel hardcodes the family geometry and reads bf16
        # module weights directly; anything else (quantized hc mixes, other
        # dims, prefill widths) stays on the eager chain.
        if not _fused_hc_enabled():
            return False
        if self.hc_count != 4 or self.hidden_size != 2560:
            return False
        down = self.input_mix_weight_down
        if hasattr(down, "scales") or down.weight.shape[0] != 320:
            return False
        if down.weight.dtype != hyper_input.dtype:
            return False
        rows = 1
        for s in hyper_input.shape[:-1]:
            rows *= s
        return 1 <= rows <= 8

    def _v3_read_applies(self, hyper_input: mx.array) -> bool:
        # v3 (two-dispatch, kernel-private 8-bit pack): single-row decode
        # reads on the combine variant with bf16 module weights. Verify
        # widths (rows 2..8) and prefill stay on the eager chain.
        if not _fused_hc_v3_enabled():
            return False
        if self.hc_count != 4 or self.hidden_size != 2560:
            return False
        if "block_inject_weight" not in self:
            return False
        if hasattr(self.input_mix_weight_down, "scales"):
            return False
        rows = 1
        for s in hyper_input.shape[:-1]:
            rows *= s
        if rows != 1:
            return False
        from mtplx.kernels.hyper_connection_v3 import (
            device_supports_hyper_v3,
            prepare_v3_pack,
        )

        # G14 device gate before paying for the pack (issue #400).
        if not device_supports_hyper_v3():
            return False
        if getattr(self, "_v3_pack", None) is None:
            self._v3_pack = prepare_v3_pack(self)
        return True

    def __call__(self, hyper_input: mx.array):
        if self._v3_read_applies(hyper_input):
            from mtplx.kernels.hyper_connection_v3 import fused_hyper_read_v3

            x2 = hyper_input.reshape(-1)
            mixed, inject = fused_hyper_read_v3(
                x2, self.hc_norm.weight, self._v3_pack
            )
            mixed = mixed.reshape(*hyper_input.shape[:-1], self.hidden_size)
            inject = inject.reshape(*hyper_input.shape[:-1], self.hc_count)
            return mixed, hyper_input, inject
        if self._fused_read_applies(hyper_input):
            from mtplx.kernels.hyper_connection import fused_hyper_read

            combine = "block_inject_weight" in self
            x2 = hyper_input.reshape(-1, self.hc_count * self.hidden_size)
            mixed, inject = fused_hyper_read(
                x2,
                self.hc_norm.weight,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.block_inject_weight.weight if combine else None,
            )
            mixed = mixed.reshape(*hyper_input.shape[:-1], self.hidden_size)
            if not combine:
                return mixed
            inject = inject.reshape(*hyper_input.shape[:-1], self.hc_count)
            return mixed, hyper_input, inject
        normed = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        grouped = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed_input = mx.mean(mix * grouped, axis=-2)
        if "block_inject_weight" not in self:
            return mixed_input
        inject = 2.0 * mx.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        return mixed_input, hyper_input, inject



def _named_gated_residuals(owner: Any):
    """``(attribute name, module)`` for every GatedResidual ``owner`` holds."""

    for name in dir(owner):
        if name.startswith("__"):
            continue
        try:
            value = getattr(owner, name)
        except Exception:  # pragma: no cover - defensive: properties may raise
            continue
        if isinstance(value, GatedResidual):
            yield name, value


class SparseMoeBlock(_Qwen3NextSparseMoeBlock):
    def __call__(self, x: mx.array) -> mx.array:
        # Fused decode path (MTPLX_FUSED_MOE_DECODE=1 + sanitize-fused gu
        # weights): collapses gate_up -> GLU -> down -> weighted-sum into two
        # dispatches. Requires 4-bit affine at a shipped forge group size
        # (32 or 64 — the 2026-08-27 01:35 reforge moved the pack to g64);
        # anything else runs the stock chain.
        sw = self.switch_mlp
        if (
            x.shape[-2] == 1
            and x.size == x.shape[-1]  # B*S == 1
            and _fused_moe_decode_enabled()
            and isinstance(sw, _FusedGateUpSwitchGLU)
            and sw.bits == 4
            and sw.group_size in (32, 64)
            and getattr(sw.down_proj, "bits", None) == 4
            and getattr(sw.down_proj, "group_size", None) in (32, 64)
        ):
            from mtplx.kernels.moe_glu_decode import moe_glu_decode

            flat = x.reshape(-1)
            # Routing math mirrors the parent exactly: softmax over ALL
            # experts first, then top-k of the probabilities.
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            idx = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
            w = mx.take_along_axis(gates, idx, axis=-1)
            if self.norm_topk_prob:
                w = w / w.sum(axis=-1, keepdims=True)
            dn = sw.down_proj
            y = moe_glu_decode(
                flat,
                sw.gu_weight,
                sw.gu_scales,
                sw.gu_biases,
                dn.weight,
                dn.scales,
                dn.biases,
                idx.reshape(-1).astype(mx.uint32),
                w.reshape(-1).astype(mx.float32),
                gu_group_size=int(sw.group_size),
                dn_group_size=int(dn.group_size),
            ).reshape(x.shape)
            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
            return (y + shared).astype(x.dtype)
        if (
            # Fused verify path (MTPLX_FUSED_MOE_VERIFY=1, dark): the M=2..4
            # MTP verify forward pays the same per-layer dependency-gap
            # serialization the M=1 pair removed from AR decode, inside the
            # 23 ms verify_hidden_eval wall (round anatomy 2026-08-31). Same
            # kernels, M-batched grid: each token's rows are bit-identical
            # to the M=1 kernel on that row alone.
            x.ndim >= 2
            and 2 <= x.shape[-2] <= 4
            and x.size == x.shape[-2] * x.shape[-1]  # B == 1
            and _fused_moe_verify_enabled()
            and isinstance(sw, _FusedGateUpSwitchGLU)
            and sw.bits == 4
            and sw.group_size in (32, 64)
            and getattr(sw.down_proj, "bits", None) == 4
            and getattr(sw.down_proj, "group_size", None) in (32, 64)
        ):
            from mtplx.kernels.moe_glu_decode import moe_glu_verify

            x2 = x.reshape(-1, x.shape[-1])
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            idx = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
            w = mx.take_along_axis(gates, idx, axis=-1)
            if self.norm_topk_prob:
                w = w / w.sum(axis=-1, keepdims=True)
            dn = sw.down_proj
            y = moe_glu_verify(
                x2,
                sw.gu_weight,
                sw.gu_scales,
                sw.gu_biases,
                dn.weight,
                dn.scales,
                dn.biases,
                idx.reshape(-1).astype(mx.uint32),
                w.reshape(-1).astype(mx.float32),
                gu_group_size=int(sw.group_size),
                dn_group_size=int(dn.group_size),
            ).reshape(x.shape)
            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
            return (y + shared).astype(x.dtype)
        return super().__call__(x)


class _FusedGateUpSwitchGLU(nn.Module):
    """SwitchGLU with gate_proj and up_proj concatenated into ONE
    gather_qmm (N=2*moe_intermediate) for the small-M decode/verify regime.

    Rationale (2026-08-26 attribution campaign): at qL=1 the MoE runs three
    gather_qmm dispatches per layer at N=640 — grids too small to fill the
    M5's 40 cores. Concatenating gate+up along the output-rows axis halves
    the large dispatches and doubles rows in flight. Per-row dot products
    are unchanged, so results match the split path up to within-row
    accumulation order. Large-M (prefill) calls fall through to the original
    SwitchGLU, keeping its expert-sorted access pattern."""

    def __init__(self, down_proj, gu_weight, gu_scales, gu_biases, group_size, bits, mode):
        super().__init__()
        self.gu_weight = gu_weight
        self.gu_scales = gu_scales
        self.gu_biases = gu_biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.down_proj = down_proj
        # Built ONLY at sanitize time with placeholder params that strict
        # load_weights replaces by the lazily concatenated pack tensors —
        # the per-projection originals never materialize. A mid-session
        # module swap cannot reclaim their memory (freed tensors keep their
        # multi-GB safetensors shard buffers pinned via siblings; measured
        # +0.31G per fused module straight into a Metal OOM).

    def _gu(self, x, idx, sorted_indices=False):
        gu = mx.gather_qmm(
            x,
            self.gu_weight,
            self.gu_scales,
            self.gu_biases,
            rhs_indices=idx,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        return mx.split(gu, 2, axis=-1)

    def __call__(self, x, indices) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate, up = self._gu(x, idx, sorted_indices=do_sort)
        x = self.down_proj(nn.silu(gate) * up, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class _FusedGateUpMLP(nn.Module):
    """Shared-expert MLP with gate_proj+up_proj as one quantized matmul
    (same fusion rationale and build-time contract as
    _FusedGateUpSwitchGLU; N=640 -> 1280)."""

    def __init__(self, down_proj, gu_weight, gu_scales, gu_biases, group_size, bits, mode):
        super().__init__()
        self.gu_weight = gu_weight
        self.gu_scales = gu_scales
        self.gu_biases = gu_biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.down_proj = down_proj

    def __call__(self, x) -> mx.array:
        gu = mx.quantized_matmul(
            x,
            self.gu_weight,
            self.gu_scales,
            self.gu_biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        gate, up = mx.split(gu, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class _FusedGDNInProj(nn.Module):
    """GDN qkv/z/b/a input projections as ONE quantized matmul.

    All four share the layer input row; at qL=1 they are four separate GEMV
    dispatches per GDN layer (35 layers = 140 dispatches/step). Row-axis
    concat of quantized packs is bit-exact per output row — each row's dot
    and its quant groups are unchanged — so the fused output just splits at
    the recorded row offsets. Same placeholder-at-build/load-fills contract
    as _FusedGateUpSwitchGLU."""

    def __init__(self, weight, scales, biases, group_size, bits, mode, splits):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._splits = list(splits)  # cumulative row offsets: qkv|z|b|a

    def __call__(self, x):
        y = mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return mx.split(y, self._splits, axis=-1)


_LAYER_GDN_RE = re.compile(
    r"^(.*\.layers\.(\d+)\.linear_attn)\.in_proj_qkv\.weight$"
)
_GDN_IN_PROJS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def _fuse_gdn_in_proj_sanitize(model, out: dict) -> dict:
    """Sanitize-time GDN input fusion (MTPLX_FUSED_GDN_INPROJ=1).

    Concatenates the four quantized input projections of every GDN layer
    along the output-rows axis on the LAZY weight dict (originals never
    materialize) and swaps in a _FusedGDNInProj child. Fuses only when all
    four are affine-quantized at one (group_size, bits); anything else keeps
    the stock modules."""
    if not _fused_gdn_in_proj_enabled():
        return out
    fused = 0
    hits = [
        (m.group(1), int(m.group(2)))
        for m in (_LAYER_GDN_RE.match(k) for k in list(out))
        if m is not None
    ]
    for prefix, idx in hits:
        parts = []
        for sub in _GDN_IN_PROJS:
            w = out.get(f"{prefix}.{sub}.weight")
            s = out.get(f"{prefix}.{sub}.scales")
            b = out.get(f"{prefix}.{sub}.biases")
            if w is None or s is None or b is None:
                parts = None
                break
            parts.append((w, s, b))
        if parts is None:
            continue
        k_words = parts[0][0].shape[-1]
        n_groups = parts[0][1].shape[-1]
        if any(w.shape[-1] != k_words or s.shape[-1] != n_groups for w, s, _ in parts):
            continue  # mixed packing: stay stock
        gdn = model.layers[idx].linear_attn
        k_in = 2560  # in_proj input dim = hidden (same contract as gate_up)
        group_size = k_in // n_groups
        bits = (k_words * 32) // k_in
        if bits not in (4, 8):
            continue
        rows = [w.shape[0] for w, _, _ in parts]
        splits = [rows[0], rows[0] + rows[1], rows[0] + rows[1] + rows[2]]
        f_w = mx.concatenate([w for w, _, _ in parts], axis=0)
        f_s = mx.concatenate([s for _, s, _ in parts], axis=0)
        f_b = mx.concatenate([b for _, _, b in parts], axis=0)
        gdn.in_proj_fused = _FusedGDNInProj(
            mx.zeros(f_w.shape, dtype=f_w.dtype),
            mx.zeros(f_s.shape, dtype=f_s.dtype),
            mx.zeros(f_b.shape, dtype=f_b.dtype),
            group_size,
            bits,
            "affine",
            splits,
        )
        for sub in _GDN_IN_PROJS:
            gdn.pop(sub, None)
        out[f"{prefix}.in_proj_fused.weight"] = f_w
        out[f"{prefix}.in_proj_fused.scales"] = f_s
        out[f"{prefix}.in_proj_fused.biases"] = f_b
        for sub in _GDN_IN_PROJS:
            for part in ("weight", "scales", "biases"):
                out.pop(f"{prefix}.{sub}.{part}", None)
        fused += 1
    if fused:
        print(f"[qwen4_exp] sanitize fused GDN in_proj: {fused} layers", flush=True)
    return out


_LAYER_ATTN_RE = re.compile(r"^(.*\.layers\.(\d+)\.self_attn)\.q_proj\.weight$")
_QSA_QKV_PROJS = ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")


def _fuse_qsa_qkv_sanitize(model, out: dict) -> dict:
    """Sanitize-time QSA attention input fusion (MTPLX_FUSED_QSA_QKV).

    q/k/v and the indexer's qk projection all consume the layer input row;
    row-axis concat of the quantized packs is bit-exact per output row (same
    contract as the GDN in_proj fusion), so the 13 attention layers run one
    shared-input GEMV instead of four. Biased checkpoints and mixed packings
    keep the stock chain."""
    if not _fused_qsa_qkv_enabled():
        return out
    fused = 0
    hits = [
        (m.group(1), int(m.group(2)))
        for m in (_LAYER_ATTN_RE.match(k) for k in list(out))
        if m is not None
    ]
    for prefix, idx in hits:
        if any(f"{prefix}.{p}.bias" in out for p in _QSA_QKV_PROJS):
            continue
        parts = []
        fused_subs = []
        for sub in _QSA_QKV_PROJS:
            w = out.get(f"{prefix}.{sub}.weight")
            s = out.get(f"{prefix}.{sub}.scales")
            b = out.get(f"{prefix}.{sub}.biases")
            if w is None or s is None or b is None:
                if sub == "indexer.index_qk_proj":
                    continue  # optional member
                parts = None
                break
            parts.append((w, s, b))
            fused_subs.append(sub)
        if parts is None:
            continue
        k_words = parts[0][0].shape[-1]
        n_groups = parts[0][1].shape[-1]
        if any(
            w.shape[-1] != k_words or s.shape[-1] != n_groups
            for w, s, _ in parts[:3]
        ):
            continue
        # Include the indexer projection only when its actual pack geometry
        # matches q/k/v; never infer this from names or config.  The v2.10
        # production artifact is 4-bit group-64 here, despite the stale older
        # 8-bit artifact assumption.
        if len(parts) == 4 and (
            parts[3][0].shape[-1] != k_words or parts[3][1].shape[-1] != n_groups
        ):
            parts = parts[:3]
            fused_subs = fused_subs[:3]
        attn = model.layers[idx].self_attn
        if getattr(attn, "indexer", None) is None:
            continue
        k_in = 2560  # attention input dim = hidden (same contract as gate_up)
        group_size = k_in // n_groups
        bits = (k_words * 32) // k_in
        if bits not in (4, 8):
            continue
        rows = [w.shape[0] for w, _, _ in parts]
        splits = [sum(rows[: i + 1]) for i in range(len(rows) - 1)]
        f_w = mx.concatenate([w for w, _, _ in parts], axis=0)
        f_s = mx.concatenate([s for _, s, _ in parts], axis=0)
        f_b = mx.concatenate([b for _, _, b in parts], axis=0)
        attn.qkv_fused = _FusedGDNInProj(
            mx.zeros(f_w.shape, dtype=f_w.dtype),
            mx.zeros(f_s.shape, dtype=f_s.dtype),
            mx.zeros(f_b.shape, dtype=f_b.dtype),
            group_size,
            bits,
            "affine",
            splits,
        )
        for name in ("q_proj", "k_proj", "v_proj"):
            attn.pop(name, None)
        if "indexer.index_qk_proj" in fused_subs:
            attn.indexer.pop("index_qk_proj", None)
        out[f"{prefix}.qkv_fused.weight"] = f_w
        out[f"{prefix}.qkv_fused.scales"] = f_s
        out[f"{prefix}.qkv_fused.biases"] = f_b
        for sub in fused_subs:
            for part in ("weight", "scales", "biases"):
                out.pop(f"{prefix}.{sub}.{part}", None)
        fused += 1
    if fused:
        print(f"[qwen4_exp] sanitize fused QSA qkv: {fused} layers", flush=True)
    return out


def _fused_gate_up_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_GATE_UP") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_qsa_qkv_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_QSA_QKV") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_gdn_in_proj_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_GDN_INPROJ") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_gdn_out_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_GDN_OUT") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_gdn_conv_norm_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_GDN_CONVNORM") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_gdn_step_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_GDN_STEP") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_conv_norm_rows_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_CONVNORM_VERIFY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _qsa_gather_enabled() -> bool:
    raw = (os.environ.get("MTPLX_QSA_GATHER") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _qsa_gather_decode_enabled() -> bool:
    raw = (os.environ.get("MTPLX_QSA_GATHER_DECODE") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _qsa_gather_min_context() -> int:
    """Rows-gather engages only at/past this KV length. Below it the fused
    dense SDPA over full KV is cheaper (the gather trades one shared O(T)
    read for per-row O(S*K) materialized copies, and the S=1 lane's own
    receipt measured the copy overhead at -5.25% on short contexts). The
    break-even for verify widths lands in the tens of thousands of tokens,
    where the dense mask chain is also the within-request growth term."""
    try:
        return max(0, int(os.environ.get("MTPLX_QSA_GATHER_MIN_CONTEXT") or 16384))
    except ValueError:
        return 16384


def _qsa_gather_max_rows() -> int:
    """Rows-gather serves query widths 2..this (default 8: AR pipelining and
    batched-verify widths). Copy-block rounds (25+ rows) multiply the
    gathered working set by S and stay on the dense path until measured."""
    try:
        return max(2, int(os.environ.get("MTPLX_QSA_GATHER_MAX_ROWS") or 8))
    except ValueError:
        return 8


def _qsa_flash_enabled() -> bool:
    raw = (os.environ.get("MTPLX_QSA_FLASH") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _qsa_score_tile_rows() -> int:
    """Prefill indexer scoring tile (MTPLX_QSA_SCORE_TILE_ROWS, 0 = off).

    The whole-chunk scores matmul stages [1, S, H, nb] fp32 plus its relu
    twin — ~2.1 GB per layer per 2048-chunk at 262K context, the dominant
    term in the #393 prefill transient. Tiling the query rows caps the live
    fp32 at tile/S of that with per-row-identical selection math (each
    row's dot, relu-sum, mask and top-k never see other rows). Opt-in
    until the per-tile eval sync cost is measured on GPU (no default flip
    without a receipt)."""
    try:
        return max(0, int(os.environ.get("MTPLX_QSA_SCORE_TILE_ROWS") or 0))
    except ValueError:
        return 0


def _fused_qsa_indexer_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_QSA_INDEXER") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _compiled_qsa_indexer_enabled() -> bool:
    raw = (os.environ.get("MTPLX_COMPILED_QSA_INDEXER") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def qsa_prefill_lane_auto_supported() -> bool:
    """Device gate for the auto default: the lane's fast consumer must exist.

    The producer only emits the ``("flash_prefill", ids, valid)`` tuple when
    this returns True, so it must name EVERY fast consumer of that tuple.
    Two exist:

    * Metal 4 TensorOps (NAX) machines take ``qsa_prefill_flash`` (M4/M5).
    * M3-class machines have no G17 tensor units and can never take that
      kernel, but they can take the vendored Steel sparse-GQA kernel when it
      is built and probed.

    Machines with neither would ride the eager selector into the dense-mask
    reconstruction — pure tax — so auto stays off there until the portable
    gather tier carries its own receipts on that hardware class.

    ``MTPLX_QSA_PREFILL=0`` remains the master kill switch above this, and
    ``MTPLX_QSA_PREFILL_DIRECT=0`` (honored inside
    ``qsa_prefill_direct_ready``) removes the direct consumer from this
    answer, so killing the direct lane on an M3 also disarms the producer
    instead of leaving it selecting for nobody.
    """

    try:
        import mlx.core as _mx

        if not _mx.metal.is_available() or _mx.default_device() != _mx.gpu:
            return False
        from mtplx.kernels.qsa_indexer_select import (
            qsa_indexer_select_nax_available,
        )

        if bool(qsa_indexer_select_nax_available()):
            return True
        from mtplx.kernels.qsa_prefill_direct import qsa_prefill_direct_ready

        return bool(qsa_prefill_direct_ready())
    except Exception:
        return False


def _qsa_prefill_enabled() -> bool:
    """Large-S score -> top-k -> sparse-attention pipeline resolution.

    Explicit env wins both ways. Unset resolves AUTO: on where the NAX flash
    kernel is supported (2026-08-30 ABBA receipts, Flash-Next M5 Max 128GB:
    flat at/below the 32K crossover both orders, +34.8% paired at 98K,
    810 tok/s at 131K, and 262K cold prefill completing at 87.4 GB peak on
    the machine class that previously wedged — issue #393), off elsewhere.
    """

    raw = (os.environ.get("MTPLX_QSA_PREFILL") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        # Even an explicit ON must fail closed to dense when the Metal SDK
        # cannot compile the MPP pipelines (macOS 27, issue #404): honoring
        # the env verbatim there turns into a guaranteed mid-request 500.
        # The probe prints its own diagnostic when it says no.
        return _qsa_prefill_producer_compile_ok()
    return qsa_prefill_lane_auto_supported() and _qsa_prefill_producer_compile_ok()


def _qsa_prefill_producer_compile_ok() -> bool:
    from mtplx.kernels.qsa_indexer_select import qsa_indexer_select_nax_available

    # The portable producer uses bounded MLX score tiles when NAX is absent
    # (qsa_indexer_prefill_metal). It never dispatches the MPP score kernel.
    # Requiring that unrelated kernel to compile disables the Steel consumer
    # on M1-M4 even after a valid native extension has been installed.
    return not qsa_indexer_select_nax_available() or _qsa_prefill_mpp_compile_ok()


def _qsa_prefill_mpp_compile_ok() -> bool:
    try:
        from mtplx.kernels.qsa_prefill_probe import (
            qsa_prefill_mpp_compile_supported,
        )

        return bool(qsa_prefill_mpp_compile_supported())
    except Exception:
        return False


def _qsa_prefill_min_rows() -> int:
    """Keep decode/short verify lanes separate from matrix-shaped prefill."""

    try:
        return max(2, int(os.environ.get("MTPLX_QSA_PREFILL_MIN_ROWS") or 32))
    except ValueError:
        return 32


def _qsa_prefill_min_context() -> int:
    """History crossover for the matrix indexer/selector pipeline.

    The exact Metal selector has a fixed dispatch/radix cost.  Production A/B
    on this machine showed that cost does not amortize against v2.10's eager
    matmul/argpartition path until roughly 32K tokens, so chunks whose earliest
    query is below that history remain entirely on the stock path.
    """

    try:
        return max(
            2049,
            int(os.environ.get("MTPLX_QSA_PREFILL_MIN_CONTEXT") or 32768),
        )
    except ValueError:
        return 32768


def _qsa_prefill_flash_min_context() -> int:
    """Crossover for the direct block-sparse attention consumer.

    32768 matches the selector crossover: the 2026-08-30 ABBA battery ran the
    flash consumer from 32K history and measured flat at the 32K rung (both
    orders) with the full win from there up, so the earlier conservative
    65536 default gave away the 32-64K span for nothing.
    """

    try:
        return max(
            2049,
            int(os.environ.get("MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT") or 32768),
        )
    except ValueError:
        return 32768


def _qsa_prefill_direct_min_context() -> int:
    """Crossover for the vendored Steel sparse-GQA consumer (M3 lane).

    Defaults to the flash consumer's 32768 so the two direct consumers share
    one story until an operator turns the knob. oMLX measured +11.7% prefill
    on M3 already at 16K; an M3 Ultra 256GB sequential A/B of this port
    (2026-09-01) saw the 32k TTFT win with
    ``MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT=2049``.
    """

    try:
        return max(
            2049,
            int(os.environ.get("MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT") or 32768),
        )
    except ValueError:
        return 32768


def _qsa_prefill_score_workspace_bytes() -> int:
    """Byte budget for per-head float32 logits plus the reduced score plane."""

    try:
        mib = max(1, int(os.environ.get("MTPLX_QSA_PREFILL_SCORE_MB") or 128))
    except ValueError:
        mib = 128
    return mib * 1024 * 1024


def _qsa_prefill_compile_rows() -> int:
    """Canonical large prefill width captured by the graph bank.

    Normal prefill uses 2,048-token chunks.  Restricting graph capture to one
    operator-selected width prevents every arbitrary final/restored suffix
    from creating another shape-specialized ``mx.compile`` trace.  The Metal
    prefill path still serves non-canonical tails without graph capture.
    """

    try:
        return max(2, int(os.environ.get("MTPLX_QSA_PREFILL_COMPILE_ROWS") or 2048))
    except ValueError:
        return 2048


def _qsa_large_prefill_enabled(rows: int, total_tokens: int) -> bool:
    # S>1 is not sufficient: MTP target verification also uses multiple rows.
    # The request-scoped phase signal keeps speculative verify/rollback on its
    # existing exact cache path and reserves this matrix-shaped lane for the
    # prompt/SSD-restored prefill it was designed to accelerate.
    return (
        current_attention_phase() == "prefill"
        and int(rows) >= _qsa_prefill_min_rows()
        # Gate on the earliest query in the chunk, not its final T.  A large
        # restored/SSD chunk may straddle the crossover; routing it by final T
        # would make its early rows pay the exact fixed-cost pathology this
        # guard exists to avoid.
        and int(total_tokens) - int(rows) >= _qsa_prefill_min_context()
        # Capability/pipeline resolution is only useful for eligible prefill
        # chunks. Never pay its imports and native readiness checks on every
        # AR or speculative decode layer, especially on portable consumers.
        and _qsa_prefill_enabled()
    )


def _qsa_prefill_flash_attention_enabled(rows: int, total_tokens: int) -> bool:
    """Whether compact block selections should bypass stock dense SDPA."""

    return (
        _qsa_large_prefill_enabled(rows, total_tokens)
        and int(total_tokens) - int(rows) >= _qsa_prefill_flash_min_context()
    )


def _qsa_prefill_direct_attention_enabled(rows: int, total_tokens: int) -> bool:
    """Whether the vendored Steel sparse-GQA consumer may serve this chunk."""

    return (
        _qsa_large_prefill_enabled(rows, total_tokens)
        and int(total_tokens) - int(rows) >= _qsa_prefill_direct_min_context()
    )


_QSA_PREFILL_COUNTS: Dict[str, int] = {}
_QSA_PREFILL_DEBUG_ARMED = False


def _qsa_prefill_count(lane: str) -> None:
    """Engagement receipt for the large-prefill lanes (A/B law: never read a
    benchmark without proof the arm's code actually ran)."""

    global _QSA_PREFILL_DEBUG_ARMED
    _QSA_PREFILL_COUNTS[lane] = _QSA_PREFILL_COUNTS.get(lane, 0) + 1
    receipt = (os.environ.get("MTPLX_QSA_PREFILL_ENGAGEMENT_FILE") or "").strip()
    if receipt:
        try:
            Path(receipt).write_text(
                json.dumps(dict(_QSA_PREFILL_COUNTS), sort_keys=True) + "\n"
            )
        except OSError:
            pass
    if not _QSA_PREFILL_DEBUG_ARMED:
        _QSA_PREFILL_DEBUG_ARMED = True
        raw = (os.environ.get("MTPLX_QSA_PREFILL_DEBUG") or "0").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            import atexit
            import sys

            atexit.register(
                lambda: print(
                    f"qsa_prefill_engagement={_QSA_PREFILL_COUNTS}",
                    file=sys.stderr,
                    flush=True,
                )
            )


def qsa_prefill_engagement() -> Dict[str, int]:
    """Snapshot of per-lane large-prefill engagement counters."""

    return dict(_QSA_PREFILL_COUNTS)


def _qsa_prefill_dispatch_tier(
    *,
    flash_supported,
    flash_call,
    direct_supported,
    direct_call,
    gather_enabled: bool,
    gather_call,
):
    """Pick the one consumer of the ``("flash_prefill", ids, valid)`` tuple.

    Order is flash (M4/M5 MPP) -> direct (Steel, the M3 lane) -> gather
    (portable) -> dense. Each predicate is called at most once and only
    until one answers True. Returns the attention output, or ``None`` to
    mean "no tier dispatched; rebuild the dense mask". There is no retry:
    once a tier is entered its failure propagates.
    """

    if flash_supported():
        out = flash_call()
        _qsa_prefill_count("flash_kernel")
        return out
    if direct_supported():
        out = direct_call()
        _qsa_prefill_count("direct_kernel")
        return out
    if gather_enabled:
        out = gather_call()
        _qsa_prefill_count("gather_tier")
        return out
    _qsa_prefill_count("dense_fallback")
    return None


def _qsa_prefill_gather_enabled() -> bool:
    """Portable gathered-attention tier for the flash_prefill block contract.

    Serves the same compact per-row block selections as the NAX flash kernel,
    on any Metal device, by gathering the bounded visible set (top-k blocks +
    causal tail, <= budget + ratio - 1 rows per query) and running a bounded
    fp32 softmax over it — the oMLX PR #3244 portable-lane approach. Without
    this tier a non-NAX machine that armed MTPLX_QSA_PREFILL would fall back
    to reconstructing the dense [S, T] mask, paying the whale the lane exists
    to remove."""

    raw = (os.environ.get("MTPLX_QSA_PREFILL_GATHER") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _qsa_prefill_gather_tile_rows() -> int:
    """Query-row tile for the portable gathered tier (bounds gathered K/V)."""

    try:
        return max(8, int(os.environ.get("MTPLX_QSA_PREFILL_GATHER_TILE") or 64))
    except ValueError:
        return 64


def _fused_hc_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_HC") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_hc_v3_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_HC_V3") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_moe_decode_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_MOE_DECODE") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fused_moe_verify_enabled() -> bool:
    raw = (os.environ.get("MTPLX_FUSED_MOE_VERIFY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_LAYER_MLP_RE = re.compile(r"^(.*\.layers\.(\d+)\.mlp)\.switch_mlp\.gate_proj\.weight$")


def _fuse_gate_up_sanitize(model, out: dict) -> dict:
    """Sanitize-time MoE gate+up fusion (MTPLX_FUSED_GATE_UP=1).

    Runs on the LAZY, file-backed weight dict before quantize/load: swaps
    each layer's switch_mlp and shared_expert modules for the fused
    variants (placeholder params), moves the pack tensors into the dict as
    lazy concatenations under the fused names, and drops the per-projection
    keys. Materialization then only ever builds fused buffers — the split
    originals never come off the shards. down_proj children keep their
    stock modules and tree paths, so the quantize predicate and strict
    load_weights treat them exactly as before."""
    if not _fused_gate_up_enabled():
        return out
    fused = 0
    layer_hits = [
        (m.group(1), int(m.group(2)))
        for m in (_LAYER_MLP_RE.match(k) for k in list(out))
        if m is not None
    ]
    for prefix, idx in layer_hits:
        layer = model.layers[idx]
        for sub, cat_axis in (("switch_mlp", 1), ("shared_expert", 0)):
            base = f"{prefix}.{sub}"
            gw = out.get(f"{base}.gate_proj.weight")
            uw = out.get(f"{base}.up_proj.weight")
            gs = out.get(f"{base}.gate_proj.scales")
            us = out.get(f"{base}.up_proj.scales")
            gb = out.get(f"{base}.gate_proj.biases")
            ub = out.get(f"{base}.up_proj.biases")
            if gw is None or uw is None or gs is None or us is None:
                continue  # bf16/tiny checkpoints: stock path
            if (gb is None) != (ub is None):
                continue
            k_in = 2560  # gate/up input dim = hidden for both blocks
            group_size = k_in // gs.shape[-1]
            bits = (gw.shape[-1] * 32) // k_in
            if gb is None:
                continue  # non-affine packing: unknown mode, stay stock
            mod = getattr(layer.mlp, sub)
            cls = _FusedGateUpSwitchGLU if sub == "switch_mlp" else _FusedGateUpMLP
            gu_w = mx.concatenate([gw, uw], axis=cat_axis)
            gu_s = mx.concatenate([gs, us], axis=cat_axis)
            gu_b = mx.concatenate([gb, ub], axis=cat_axis)
            setattr(
                layer.mlp,
                sub,
                cls(
                    mod.down_proj,
                    mx.zeros(gu_w.shape, dtype=gu_w.dtype),
                    mx.zeros(gu_s.shape, dtype=gu_s.dtype),
                    mx.zeros(gu_b.shape, dtype=gu_b.dtype),
                    group_size,
                    bits,
                    "affine",
                ),
            )
            out[f"{base}.gu_weight"] = gu_w
            out[f"{base}.gu_scales"] = gu_s
            out[f"{base}.gu_biases"] = gu_b
            for proj in ("gate_proj", "up_proj"):
                for part in ("weight", "scales", "biases"):
                    out.pop(f"{base}.{proj}.{part}", None)
            fused += 1
    if fused:
        print(f"[qwen4_exp] sanitize fused gate+up: {fused} modules", flush=True)
    return out


# Armed around a speculative verify forward: GDN layers retain the exact
# recurrence rows so a rejected window commits by replaying only the
# gated-delta recurrence (see Qwen4ExpTextModel.commit_verified_window).
_VERIFY_CAPTURE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "qwen4_exp_verify_capture", default=False
)
_COMPILED_VERIFY_PLE: contextvars.ContextVar[Optional[mx.array]] = (
    contextvars.ContextVar("qwen4_exp_compiled_verify_ple", default=None)
)


@contextlib.contextmanager
def verify_capture_scope():
    token = _VERIFY_CAPTURE.set(True)
    try:
        yield
    finally:
        _VERIFY_CAPTURE.reset(token)


@contextlib.contextmanager
def compiled_verify_ple_scope(embedding: Optional[mx.array]):
    token = _COMPILED_VERIFY_PLE.set(embedding)
    try:
        yield
    finally:
        _COMPILED_VERIFY_PLE.reset(token)


def _qsa_stock_rows_gather_kv(
    keys: mx.array, values: mx.array, token_idx: mx.array
) -> tuple[mx.array, mx.array]:
    """Materialize per-row QSA K/V with the two stock gather dispatches."""

    h_kv = int(keys.shape[1])
    rows, selected = token_idx.shape
    head_dim = int(keys.shape[-1])
    flat = token_idx.reshape(-1)
    return (
        mx.take(keys, flat, axis=2).reshape(
            1, h_kv, int(rows), int(selected), head_dim
        ),
        mx.take(values, flat, axis=2).reshape(
            1, h_kv, int(rows), int(selected), head_dim
        ),
    )


def _qsa_rows_gather_kv_route(cache: Any, rows: int) -> Any:
    """Select the construction-bound gather only for its physical M=4 lane."""

    if int(rows) == 4:
        return cache.rows_gather_kv_m4
    return _qsa_stock_rows_gather_kv


class QSACache:
    """Cache for one QSA layer: the attention KV plus the indexer's raw key
    stream and the incrementally maintained pooled (mean->norm->rope) block
    keys. Single-sequence.

    The raw/pooled streams are POSITIONAL buffers keyed to ``kv.offset``, not
    append-only logs: every write lands at the absolute row range of the
    tokens being forwarded and the valid lengths derive from ``kv.offset``.
    That keeps the indexer streams in lockstep with the KV through every
    mutation the runtime performs — per-round speculative rollback,
    verified-window trims, and session-bank ``state`` round-trips — which
    also makes the layer trimmable like a plain ``KVCache`` instead of being
    deep-cloned every verify round. (Append-only streams desynced from the
    KV on the first rollback and, past the indexer's engage threshold, built
    a selection mask longer than the KV: the 2026-08-27 OpenCode
    ``broadcast_shapes (1,1,4,3719) vs (1,24,4,3715)`` crash.)"""

    step = 256

    def __init__(self, compress_ratio: int = 4):
        self.kv = KVCache()
        self.ratio = max(1, int(compress_ratio))
        self.rows_gather_kv_m4 = _qsa_stock_rows_gather_kv
        self.raw_keys: Optional[mx.array] = None  # [1, cap, index_head_dim]
        self.pooled: Optional[mx.array] = None  # [1, cap_blocks, index_head_dim]
        self.pooled_len = 0  # valid pooled blocks
        # fp32-transposed mirror of ``pooled`` [1, 1, D, cap_blocks], kept in
        # lockstep by write_pooled. The indexer used to upcast + transpose the
        # ENTIRE pooled table on every forward of every QSA layer — 33.5 MB
        # allocated and freed per layer per decoded token at 262K context
        # (#393 audit). Same values (astype of the same bf16 blocks), so
        # selection is bit-identical; this is allocation hygiene only.
        self.pooled_f32_t: Optional[mx.array] = None
        # Host-planned graph buckets can be reserved before the first array
        # write, when dtype/head width are not known yet.  The next write
        # materializes the pending capacity with the actual projected dtype.
        self._reserved_raw_capacity = 0
        self._reserved_pooled_capacity = 0

    @property
    def offset(self) -> int:
        return self.kv.offset

    @staticmethod
    def _grown_cap(end: int, current: int, step: int) -> int:
        """Geometric (doubling) growth, step-aligned.

        The previous fixed +``step`` growth full-copied the buffer every 256
        rows: Θ(N²) memcpy — ~34 GB of pure copy traffic per QSA layer over
        a 262K decode, times 13 caches (#393 audit). Doubling bounds total
        copy traffic at O(N) with at most 2x capacity overshoot; ``nbytes``
        keeps reporting real capacity so memory accounting stays honest."""
        cap = ((end + step - 1) // step) * step
        return max(cap, 2 * current)

    def write_raw(self, keys: mx.array) -> None:
        """Store this forward's indexer keys at their absolute positions.

        Called before ``kv.update_and_fetch`` advances the offset, so
        ``kv.offset`` IS the absolute position of ``keys[:, 0]``. After a
        trim the same positions are simply overwritten."""
        start = self.kv.offset
        end = start + keys.shape[1]
        if self.raw_keys is None or end > self.raw_keys.shape[1]:
            current = 0 if self.raw_keys is None else self.raw_keys.shape[1]
            # Geometric growth bounds copy traffic; a staged host reservation
            # (compiled-indexer graph buckets) may demand a wider backing.
            cap = max(self._grown_cap(end, current, self.step), self._reserved_raw_capacity)
            grown = mx.zeros((1, cap, keys.shape[2]), keys.dtype)
            if self.raw_keys is not None:
                grown[:, : self.raw_keys.shape[1], :] = self.raw_keys
            self.raw_keys = grown
        self.raw_keys[:, start:end, :] = keys

    def write_pooled(self, blocks: mx.array, nb_start: int, nb_total: int) -> None:
        if self.pooled is None or nb_total > self.pooled.shape[1]:
            current = 0 if self.pooled is None else self.pooled.shape[1]
            cap = max(
                self._grown_cap(nb_total, current, self.step),
                self._reserved_pooled_capacity,
            )
            grown = mx.zeros((1, cap, blocks.shape[2]), blocks.dtype)
            if self.pooled is not None:
                grown[:, : self.pooled.shape[1], :] = self.pooled
            self.pooled = grown
        self.pooled[:, nb_start:nb_total, :] = blocks
        # Keep the fp32-transposed mirror in lockstep (same capacity). When
        # the mirror is absent (fresh cache, or dropped by a state restore)
        # it must seed from the pooled buffer's CONTENT — zeros would blank
        # every previously valid block's scores (caught by the state
        # round-trip gate).
        cap_blocks = self.pooled.shape[1]
        if self.pooled_f32_t is None:
            self.pooled_f32_t = mx.swapaxes(self.pooled.astype(mx.float32), 1, 2)[
                :, None
            ]
        elif self.pooled_f32_t.shape[3] < cap_blocks:
            grown_t = mx.zeros((1, 1, blocks.shape[2], cap_blocks), mx.float32)
            grown_t[..., : self.pooled_f32_t.shape[3]] = self.pooled_f32_t
            self.pooled_f32_t = grown_t
            self.pooled_f32_t[..., nb_start:nb_total] = mx.swapaxes(
                blocks.astype(mx.float32), 1, 2
            )[:, None]
        else:
            self.pooled_f32_t[..., nb_start:nb_total] = mx.swapaxes(
                blocks.astype(mx.float32), 1, 2
            )[:, None]
        self.pooled_len = nb_total

    def pooled_f32_view(self, nb: int) -> mx.array:
        """[1, 1, D, nb] fp32 view of the valid pooled blocks.

        Rebuilds the mirror from ``pooled`` after a state restore (setter
        drops it) or a compiled-indexer commit (which replaces ``pooled``
        wholesale and nulls the mirror); otherwise a zero-copy slice of the
        maintained buffer."""
        if self.pooled_f32_t is None or self.pooled_f32_t.shape[3] < nb:
            self.pooled_f32_t = mx.swapaxes(
                self.pooled.astype(mx.float32), 1, 2
            )[:, None]
        return self.pooled_f32_t[..., :nb]

    def reserve_indexer_capacity(
        self,
        *,
        raw_capacity: int,
        pooled_capacity: int,
    ) -> None:
        """Reserve fixed backing shapes before an indexer graph is traced.

        The MTP replay planner calls this on the host.  Existing allocations
        grow immediately and retain their active prefix; a pristine cache
        records the request until its first projected rows establish dtype and
        head width.  No reservation may truncate a live logical frontier.
        """

        raw_requested = int(raw_capacity)
        pooled_requested = int(pooled_capacity)
        if raw_requested < 0 or pooled_requested < 0:
            raise ValueError(
                "QSA reserved capacities must be non-negative; got "
                f"raw={raw_requested}, pooled={pooled_requested}"
            )

        raw_existing = 0 if self.raw_keys is None else int(self.raw_keys.shape[1])
        pooled_existing = 0 if self.pooled is None else int(self.pooled.shape[1])
        raw_target = max(
            raw_requested,
            raw_existing,
            self._reserved_raw_capacity,
        )
        pooled_target = max(
            pooled_requested,
            pooled_existing,
            self._reserved_pooled_capacity,
        )
        if raw_target < self.offset:
            raise ValueError(
                f"raw capacity {raw_target} cannot cover QSA offset {self.offset}"
            )
        if pooled_target < self.pooled_len:
            raise ValueError(
                "pooled capacity cannot truncate the valid QSA frontier: "
                f"{pooled_target} < {self.pooled_len}"
            )

        self._reserved_raw_capacity = raw_target
        self._reserved_pooled_capacity = pooled_target
        if self.raw_keys is not None and raw_target > raw_existing:
            grown = mx.zeros(
                (1, raw_target, self.raw_keys.shape[2]),
                self.raw_keys.dtype,
            )
            grown[:, :raw_existing, :] = self.raw_keys
            self.raw_keys = grown
        if self.pooled is not None and pooled_target > pooled_existing:
            grown = mx.zeros(
                (1, pooled_target, self.pooled.shape[2]),
                self.pooled.dtype,
            )
            grown[:, :pooled_existing, :] = self.pooled
            self.pooled = grown

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        trimmed = self.kv.trim(n)
        # Pooled blocks past the new frontier were built from now-rejected
        # rows. The raw buffer needs no touch: future writes land at the same
        # absolute positions and overwrite.
        self.pooled_len = min(self.pooled_len, self.kv.offset // self.ratio)
        return trimmed

    @property
    def nbytes(self) -> int:
        total = self.kv.nbytes
        if self.raw_keys is not None:
            total += self.raw_keys.nbytes
        if self.pooled is not None:
            total += self.pooled.nbytes
        return total

    @property
    def state(self):
        off = self.kv.offset
        nb = min(self.pooled_len, off // self.ratio)
        raw = None if self.raw_keys is None else self.raw_keys[:, :off, :]
        pooled = None if self.pooled is None or nb == 0 else self.pooled[:, :nb, :]
        return (*self.kv.state, raw, pooled)

    @state.setter
    def state(self, v):
        if len(v) != 4:
            raise ValueError(
                "QSACache.state expects (keys, values, raw_keys, pooled); got "
                f"{len(v)} leaves — a session snapshot from an older build; "
                "drop it and re-prefill"
            )
        keys, values, raw, pooled = v
        self.kv.state = (keys, values)
        self.raw_keys = raw
        self.pooled = pooled
        self.pooled_len = 0 if pooled is None else pooled.shape[1]
        # Derived mirror: rebuilt lazily on the first pooled_f32_view read.
        # Restored snapshots stay 4-leaf — the state contract is unchanged.
        self.pooled_f32_t = None
        self._reserved_raw_capacity = 0 if raw is None else int(raw.shape[1])
        self._reserved_pooled_capacity = 0 if pooled is None else int(pooled.shape[1])


class QSAIndexer(nn.Module):
    """Vectorized exact port of the reference indexer for the single-sequence
    causal case (B=1, no padding): every query selects its top
    (budget/compress_ratio) complete key blocks by relu-scored pooled keys,
    plus the visible incomplete tail."""

    # The selector carries one private float32 score row per pooled backing
    # block. Bound that hidden output per Metal dispatch without imposing any
    # cap on logical history length. An irreducible one-row dispatch can exceed
    # the target at extreme capacities; MLX may also retain multiple lazy
    # chunks until their concatenated consumer is evaluated, so this is not a
    # claim about graph-wide peak memory.
    _fused_score_scratch_bytes = 32 * 1024 * 1024

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.budget = args.indexer_budget
        self.ratio = args.indexer_compress_ratio
        self.block_topk = self.budget // self.ratio
        self.rms_norm_eps = float(args.rms_norm_eps)
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self._inv_freq, self._rope_attention_scaling = (
            _rope_inv_freq_and_scaling_for(args)
        )
        # Kept outside the nn.Module parameter tree.  The graph bank is built
        # lazily on the first eligible inference call, after checkpoint load
        # and sanitize-time projection fusion have finalized every weight.
        object.__setattr__(self, "_compiled_indexer_core", None)
        object.__setattr__(self, "_compiled_indexer_parameter_signature", None)

    def _pool_keys_eager(
        self,
        fresh: mx.array,
        nb_old: int,
        nb_total: int,
    ) -> mx.array:
        """Stock completed-block preparation kept as the numeric oracle."""

        fresh = fresh.reshape(1, nb_total - nb_old, self.ratio, self.head_dim)
        pooled = mx.mean(fresh.astype(mx.float32), axis=2).astype(fresh.dtype)
        pooled = self.k_layernorm(pooled)
        starts = mx.arange(nb_old, nb_total, dtype=mx.int32) * self.ratio
        cos, sin = _rope_cos_sin(
            starts,
            self._inv_freq,
            self._rope_attention_scaling,
        )
        return _apply_partial_rope(pooled[:, :, None, :], cos, sin)[:, :, 0, :]

    def _prepare_kernel_supported(
        self,
        values: mx.array,
        norm_weight: mx.array,
        *,
        expected_ndim: int,
    ) -> bool:
        if not _fused_qsa_indexer_enabled():
            return False
        from mtplx.kernels.qsa_indexer_prepare import (
            qsa_indexer_prepare_supported,
        )

        return qsa_indexer_prepare_supported(
            values,
            norm_weight,
            self._inv_freq,
            expected_ndim=expected_ndim,
        )

    def _extend_pooled(self, cache: QSACache, total: int) -> Optional[mx.array]:
        if getattr(cache, "fixed_capacity", False):
            return self._extend_pooled_fixed(cache, total)
        nb_total = total // self.ratio
        nb_old = min(cache.pooled_len, nb_total)
        if nb_total > nb_old:
            fresh = cache.raw_keys[:, nb_old * self.ratio : nb_total * self.ratio, :]
            if self._prepare_kernel_supported(
                fresh,
                self.k_layernorm.weight,
                expected_ndim=3,
            ):
                from mtplx.kernels.qsa_indexer_prepare import (
                    qsa_indexer_pool_keys_metal,
                )

                pooled = qsa_indexer_pool_keys_metal(
                    fresh,
                    self.k_layernorm.weight,
                    self._inv_freq,
                    block_start=nb_old,
                    compress_ratio=self.ratio,
                    eps=self.rms_norm_eps,
                    attention_scaling=self._rope_attention_scaling,
                )
            else:
                pooled = self._pool_keys_eager(fresh, nb_old, nb_total)
            cache.write_pooled(pooled, nb_old, nb_total)
        if nb_total == 0:
            return None
        return cache.pooled[:, :nb_total, :]

    def _extend_pooled_fixed(self, cache: QSACache, total) -> mx.array:
        """Update only newly completed blocks in a fixed QSA bank.

        Verify width is static at trace time.  At the production M4/ratio-4
        shape at most one block completes, so this is one gather, one pooled
        projection, and one conditional fixed-shape slice update.
        """
        step_rows = int(getattr(cache, "_last_write_rows", 1))
        nb_old = cache.offset // self.ratio
        nb_total = total // self.ratio
        max_new = max(1, (step_rows + self.ratio - 1) // self.ratio)
        pooled = cache.pooled
        pooled_capacity = int(pooled.shape[1])
        for rel in range(max_new):
            block = nb_old + rel
            safe_block = mx.minimum(
                block, mx.array(pooled_capacity - 1, dtype=block.dtype)
            )
            start = safe_block * self.ratio
            fresh = mx.slice(
                cache.raw_keys,
                start,
                axes=(1,),
                slice_size=(1, self.ratio, self.head_dim),
            )
            fresh = fresh.reshape(1, 1, self.ratio, self.head_dim)
            candidate = mx.mean(fresh.astype(mx.float32), axis=2).astype(fresh.dtype)
            candidate = self.k_layernorm(candidate)
            starts = safe_block.reshape(1).astype(mx.int32) * self.ratio
            if qwen4_opdiet_enabled("rope"):
                cos, sin = _rope_cos_sin_half(
                    starts,
                    self._inv_freq,
                    self._rope_attention_scaling,
                )
                candidate = _apply_partial_rope_half(
                    candidate[:, :, None, :], cos, sin
                )[:, :, 0, :]
            else:
                cos, sin = _rope_cos_sin(
                    starts,
                    self._inv_freq,
                    self._rope_attention_scaling,
                )
                candidate = _apply_partial_rope(
                    candidate[:, :, None, :], cos, sin
                )[:, :, 0, :]
            if qwen4_opdiet_enabled("bank"):
                # One conditional pass over the bank instead of two, and that
                # pass stays a CONTIGUOUS copy.
                #
                # The stock pair rewrites the whole fixed bank twice to store
                # one block row: mx.slice_update copies it (the leaf is held
                # by the caller, so MLX cannot donate it) and mx.where then
                # reads both copies to pick one. Resolving the condition on
                # the touched ROW instead leaves exactly one full-bank pass --
                # the slice_update copy -- and makes the conditional work
                # head_dim wide instead of capacity x head_dim.
                #
                # Selecting over the whole bank on a row-id mask also removes
                # a pass, and measured -20% against stock; but both of its
                # operands broadcast, so MLX emits a general (strided) select
                # whose per-element index arithmetic gives most of the win
                # back. This spelling measured -49%
                # (the PR #391 harness micro_opdiet.py, compiled lane, 2026-09-01:
                # 0.492 -> 0.392 -> 0.253 ms per 12 QSA layers), which is why
                # it ships despite issuing MORE dispatches than either.
                #
                # mx.slice_update casts the update to the bank dtype; mx.where
                # would PROMOTE instead. Cast first so the two spellings agree
                # even when a caller's norm weights widen the candidate (a
                # no-op, and free, whenever they already match).
                old_row = mx.slice(
                    pooled,
                    safe_block,
                    axes=(1,),
                    slice_size=(1, 1, pooled.shape[2]),
                )
                merged = mx.where(
                    nb_total > block, candidate.astype(pooled.dtype), old_row
                )
                pooled = mx.slice_update(pooled, merged, safe_block, axes=(1,))
            else:
                updated = mx.slice_update(pooled, candidate, safe_block, axes=(1,))
                pooled = mx.where(nb_total > block, updated, pooled)
        cache.pooled = pooled
        return pooled

    def _tiled_topk(
        self,
        q: mx.array,
        pooled_t: mx.array,
        nb_q: mx.array,
        blk: mx.array,
        neg: mx.array,
        k_eff: int,
        nb_total: int,
        tile: int,
    ) -> mx.array:
        """Per-tile scoring + top-k; concatenated [S, k_eff] indices.

        Row math is identical to the whole-chunk path (each row's dot,
        relu-sum, validity mask, tie-break and top-k involve no other row);
        the per-tile mx.eval is the point — it retires each tile's fp32
        score intermediates before the next tile is built, so the live
        transient is bounded by ONE tile instead of the whole chunk."""
        parts = []
        S = q.shape[1]
        tie = blk.astype(mx.float32)[None, :] * 1e-12
        for s0 in range(0, S, tile):
            s1 = min(s0 + tile, S)
            sc = mx.matmul(q[:, s0:s1].astype(mx.float32), pooled_t)
            sc = mx.maximum(sc, 0.0).sum(axis=2) / math.sqrt(self.head_dim)
            sc = sc[0]  # [s1-s0, nb]
            valid_t = blk[None, :] < nb_q[s0:s1, None]
            masked_t = mx.where(valid_t, sc, neg) - tie
            top_t = mx.argpartition(masked_t, kth=nb_total - k_eff, axis=-1)[
                :, nb_total - k_eff :
            ]
            mx.eval(top_t)
            # Each settled tile is owner progress; a 100k+ sparse prefill
            # otherwise looks frozen to the stream stall watchdog (#448).
            _owner_progress_tick()
            parts.append(top_t)
        return mx.concatenate(parts, axis=0)

    def _select_eager(
        self,
        q: mx.array,
        pos_start: int,
        cache: QSACache,
        pooled: mx.array,
        total: int,
    ):
        """Stock MLX selector kept as the correctness oracle and kill-switch."""

        S = q.shape[1]
        nb_total = pooled.shape[1]
        # Fixed compiled-verify bank (PR #391 step 2): ``pos_start``/``total``
        # are graph tensors, so every host comparison on them is skipped and
        # the lane decisions come from the bank's promotion-time flags.
        fixed_capacity = bool(getattr(cache, "fixed_capacity", False))
        # Cached fp32-transposed pooled view: same values as the old per-call
        # astype+swapaxes of the whole table (astype of the same bf16 blocks
        # -> bit-identical scores), without re-materializing 33.5 MB per
        # layer per token at 262K (#393).
        pooled_t = cache.pooled_f32_view(nb_total)  # [1,1,D,nb]

        qpos = pos_start + mx.arange(S, dtype=mx.int32)  # abs position
        nb_q = (qpos + 1) // self.ratio  # complete blocks visible per query [S]
        blk = mx.arange(nb_total, dtype=mx.int32)
        valid = blk[None, :] < nb_q[:, None]  # [S, nb]
        neg = mx.array(-mx.inf, dtype=mx.float32)
        k_eff = min(self.block_topk, nb_total)

        tile = _qsa_score_tile_rows()
        if S > 1 and not fixed_capacity and 0 < tile < S:
            # Tiled scoring (see _qsa_score_tile_rows): bounds the live fp32
            # score transient at one tile; per-row selection math identical.
            top_idx = self._tiled_topk(
                q, pooled_t, nb_q, blk, neg, k_eff, nb_total, tile
            )
        else:
            scores = mx.matmul(q.astype(mx.float32), pooled_t)  # [1,S,H,nb]
            scores = (
                mx.maximum(scores, 0.0).sum(axis=2) / math.sqrt(self.head_dim)
            )
            scores = scores[0]  # [S, nb]
            masked_scores = mx.where(valid, scores, neg)
            # torch.topk tie-break (lowest index wins). Exact ties are
            # common: a block whose every head-dot is negative relu-scores
            # exactly 0.0.
            masked_scores = masked_scores - blk.astype(mx.float32)[None, :] * 1e-12
            top_idx = mx.argpartition(masked_scores, kth=nb_total - k_eff, axis=-1)[
                :, nb_total - k_eff :
            ]

        if S > 1 and not fixed_capacity and _qsa_large_prefill_enabled(S, total):
            # Preserve the eager score/top-k expression as an independently
            # selectable oracle while handing attention the compact block set.
            # IDs are chronological so the online-softmax consumer has a
            # deterministic traversal; validity distinguishes the padded
            # top-k slots on early rows whose complete prefix has < K blocks.
            block_ids = mx.sort(top_idx.astype(mx.int32), axis=-1)
            block_valid = mx.take_along_axis(
                valid,
                block_ids.astype(mx.int64),
                axis=-1,
            )
            block_ids = mx.where(
                block_valid,
                block_ids,
                mx.array(0, dtype=mx.int32),
            )
            _qsa_prefill_count("eager_selector")
            return ("flash_prefill", block_ids, block_valid)

        selected = mx.zeros((S, nb_total), dtype=mx.bool_)
        selected = mx.put_along_axis(
            selected, top_idx.astype(mx.int64), mx.array(True), axis=-1
        )
        selected = selected & valid  # -inf padding rows never select

        if (
            S == 1
            and not fixed_capacity
            and _qsa_flash_enabled()
        ):
            # Flash-skip lane (MTPLX_QSA_FLASH): hand attention the sorted
            # selected BLOCK ids + host-side tail bounds; the block-sparse
            # flash kernel iterates exactly that visible set in place — no
            # dense mask staged, no gathered copies (both measured slower:
            # dense = full-context reads, gather = -5.25% from two
            # materialized copies per layer per token, d6171d2c).
            _qsa_prefill_count("decode_flash_skip")
            blk_idx = mx.sort(top_idx[0].astype(mx.int32))
            tail_start = ((pos_start + 1) // self.ratio) * self.ratio
            return ("flash", blk_idx, tail_start)

        if (
            S == 1
            and not fixed_capacity
            and _qsa_gather_decode_enabled()
        ):
            # Decode gather lane (MTPLX_QSA_GATHER_DECODE, dormant opt-in —
            # FALSIFIED d6171d2c, clean A/B/A -5.25% at 22.9k, so the
            # rows-gather family default must never arm it): return the
            # selected TOKEN INDICES instead of a dense [T] mask so
            # attention reads only budget+tail keys/values. Every returned
            # token is visible by construction (complete selected blocks
            # are < the tail start; the tail runs to the current position),
            # so the gathered SDPA needs no mask — identical math to the
            # masked dense product over the same visible set.
            blk_idx = mx.sort(top_idx[0].astype(mx.int32))
            tok_from_blocks = (
                blk_idx[:, None] * self.ratio + mx.arange(self.ratio, dtype=mx.int32)
            ).reshape(-1)
            # Host-side int (no .item() sync — a per-layer eval would stall
            # the AR pipeline): for the single decode row qpos == pos_start,
            # so the visible-complete-block count is (pos_start+1) // ratio.
            _qsa_prefill_count("decode_gather")
            tail_start = ((pos_start + 1) // self.ratio) * self.ratio
            tail_ids = mx.arange(tail_start, total, dtype=mx.int32)
            return mx.concatenate([tok_from_blocks, tail_ids])

        rows_gather = (
            # The fixed bank decided its lane once at promotion (host ints);
            # inside the trace ``total`` is a tensor and must not be compared.
            bool(getattr(cache, "fixed_rows_gather", False))
            if fixed_capacity
            else (
                _qsa_gather_enabled()
                and S <= _qsa_gather_max_rows()
                and total >= _qsa_gather_min_context()
            )
        )
        if (
            S > 1
            and not (0 < tile < S)  # tiled branch produced no top_idx
            and rows_gather
        ):
            # Rows-gather lane (MTPLX_QSA_GATHER at S>1), adapting the
            # per-query gather + GQA-broadcast attention from community PR
            # #380 by @maceip. Every S>1 forward previously staged a dense
            # [S, T] bool mask and read the FULL KV through fused SDPA in
            # each of the 12 QSA layers, an O(T)-per-round chain that grows
            # with the generation. Here each row hands attention its own
            # token list instead: k_eff selected blocks plus its visible
            # tail, padded to one constant width (k_eff*ratio + ratio) so
            # gather shapes stay stable across the whole generation.
            # Selected blocks are complete blocks strictly below each row's
            # tail start, so the lists never double-count a token; invalid
            # slots carry valid=False and are re-pointed at token 0 for the
            # take.
            blk_ok = mx.take_along_axis(valid, top_idx.astype(mx.int64), axis=-1)
            tok_blocks = (
                top_idx.astype(mx.int32)[:, :, None] * self.ratio
                + mx.arange(self.ratio, dtype=mx.int32)
            ).reshape(S, -1)
            blocks_ok = mx.repeat(blk_ok, self.ratio, axis=1)
            tail_tok = nb_q[:, None] * self.ratio + mx.arange(
                self.ratio, dtype=mx.int32
            )
            tail_ok = tail_tok <= qpos[:, None]
            token_idx = mx.concatenate([tok_blocks, tail_tok], axis=1)
            token_ok = mx.concatenate([blocks_ok, tail_ok], axis=1)
            token_idx = mx.where(token_ok, token_idx, mx.array(0, dtype=mx.int32))
            return ("gather_rows", token_idx, token_ok)

        # Blocks -> tokens, plus the visible tail, intersected with causal.
        if S == 1:
            _qsa_prefill_count("decode_dense_mask")
        tok_sel = mx.repeat(selected, self.ratio, axis=1)
        if fixed_capacity:
            # Fixed bank: the mask spans the static raw-key capacity so the
            # compiled graph keeps one shape; ``causal`` below hides every
            # column past the live frontier, so the math matches the stock
            # mask over the visible set.
            mask_width = int(cache.raw_keys.shape[1])
        else:
            if nb_total * self.ratio < total:
                pad = mx.zeros((S, total - nb_total * self.ratio), dtype=mx.bool_)
                tok_sel = mx.concatenate([tok_sel, pad], axis=1)
            mask_width = total
        tpos = mx.arange(mask_width, dtype=mx.int32)
        tail = tpos[None, :] >= (nb_q[:, None] * self.ratio)
        causal = tpos[None, :] <= qpos[:, None]
        mask = (tok_sel | tail) & causal
        return mask[None, None]

    def _fused_selector_supported(self, q: mx.array, pooled: mx.array) -> bool:
        """Static fail-closed eligibility; kernel failures remain visible."""

        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
        supported_dtypes = (mx.float16, mx.bfloat16, mx.float32)
        if q.dtype not in supported_dtypes or pooled.dtype not in supported_dtypes:
            return False
        if q.ndim != 4 or pooled.ndim != 3:
            return False
        if q.shape[0] != 1 or pooled.shape[0] != 1:
            return False
        if q.shape[1] <= 0 or q.shape[2] <= 0 or q.shape[3] <= 0:
            return False
        if pooled.shape[1] <= 0 or pooled.shape[2] != q.shape[3]:
            return False
        if not (1 <= self.block_topk <= 512) or self.ratio <= 0:
            return False
        if q.dtype == mx.float32 or pooled.dtype == mx.float32:
            from mtplx.kernels.qsa_indexer_select import (
                qsa_indexer_select_nax_available,
            )

            if not qsa_indexer_select_nax_available():
                return False
        return True

    def _prefill_selector_supported(self, q: mx.array, pooled: mx.array) -> bool:
        """Fail-closed eligibility for the vectorized large-S selector."""

        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
        supported_dtypes = (mx.float16, mx.bfloat16, mx.float32)
        if q.dtype not in supported_dtypes or pooled.dtype not in supported_dtypes:
            return False
        if q.ndim != 4 or pooled.ndim != 3:
            return False
        if int(q.shape[0]) != 1 or int(pooled.shape[0]) != 1:
            return False
        if int(q.shape[1]) <= 1 or int(q.shape[2]) <= 0 or int(q.shape[3]) <= 0:
            return False
        if int(pooled.shape[1]) <= 0 or int(pooled.shape[2]) != int(q.shape[3]):
            return False
        return 1 <= self.block_topk <= 512 and self.ratio > 0

    def _fused_query_chunk_rows(self, rows: int, backing_blocks: int) -> int:
        scratch_per_row = max(1, int(backing_blocks)) * 4
        return min(
            int(rows),
            max(1, self._fused_score_scratch_bytes // scratch_per_row),
        )

    def _select_fused(
        self,
        q: mx.array,
        pos_start: int,
        pooled_backing: mx.array,
        logical_blocks: int,
        total: int,
        mode: str,
    ):
        """Dispatch the selector, chunking query rows but never history."""

        from mtplx.kernels.qsa_indexer_select import (
            qsa_indexer_select_blocks_metal,
            qsa_indexer_select_dense_mask_metal,
            qsa_indexer_select_row_tokens_metal,
        )

        rows = int(q.shape[1])
        chunk_rows = self._fused_query_chunk_rows(rows, pooled_backing.shape[1])
        # Keep the custom-kernel output specialization stable while a pooled
        # cache allocation remains in place. A logical prefix can occupy all
        # backing blocks and still have up to ratio-1 visible tail tokens, so
        # one extra ratio-sized block is the smallest safe capacity.
        dense_output_capacity = (
            (int(pooled_backing.shape[1]) + 1) * self.ratio
            if mode == "dense_mask"
            else None
        )
        chunks = []
        for row_start in range(0, rows, chunk_rows):
            q_chunk = q[:, row_start : row_start + chunk_rows]
            kwargs = {
                "pos_start": pos_start + row_start,
                "total_tokens": total,
                "block_topk": self.block_topk,
                "compress_ratio": self.ratio,
                "logical_blocks": logical_blocks,
            }
            if mode == "blocks":
                chunk = qsa_indexer_select_blocks_metal(
                    q_chunk, pooled_backing, **kwargs
                )
            elif mode == "row_tokens":
                chunk = qsa_indexer_select_row_tokens_metal(
                    q_chunk, pooled_backing, **kwargs
                )
            elif mode == "dense_mask":
                chunk = qsa_indexer_select_dense_mask_metal(
                    q_chunk,
                    pooled_backing,
                    output_total_tokens=dense_output_capacity,
                    **kwargs,
                )
            else:
                raise ValueError(f"unknown fused QSA selector mode {mode!r}")
            chunks.append(chunk)

        if mode == "dense_mask":
            mask = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)
            return mask[..., :total]
        if len(chunks) == 1:
            return chunks[0]
        return tuple(
            mx.concatenate([chunk[leaf] for chunk in chunks], axis=0)
            for leaf in range(len(chunks[0]))
        )

    def _verify_glue_rope_idx(self) -> bool:
        """True when ``MTPLX_QWEN4_VERIFY_GLUE``'s ``qsa_rope_idx`` serves.

        Host-only, and read from the verdict the install probe recorded at
        model build outside any trace, so two traces of the same compiled
        verify graph cannot disagree about which preparation they contain.
        A False here is routing (the item is not armed, or its probe
        disabled it); a contract miss raised at install and never gets here.
        """

        if not qwen4_verify_glue_enabled("qsa_rope_idx"):
            return False
        from mtplx import qwen4_verify_glue as _glue

        if not _glue.qsa_rope_idx_installed():
            return False
        _glue.note_prep_call()
        return True

    def _prepare_queries_m4(self, q: mx.array, pos_start) -> mx.array:
        """RMSNorm + partial RoPE in one dispatch, instead of twelve.

        This is the SHIPPED ``qsa_indexer_prepare_queries_metal``, whose
        bit-exactness against ``_prepare_queries_eager`` is pinned for
        bf16/f16 in tests/test_qsa_indexer_prepare_metal.py.  The fixed lane
        skipped it only because ``_prepare_queries`` gates on
        MTPLX_QSA_FUSED_INDEXER; the kernel itself already accepts a tensor
        ``pos_start``, which is exactly what a tensor-offset cache needs.
        """

        from mtplx.kernels.qsa_indexer_prepare import (
            qsa_indexer_prepare_queries_metal,
        )

        return qsa_indexer_prepare_queries_metal(
            q,
            self.q_layernorm.weight,
            self._inv_freq,
            pos_start=pos_start,
            eps=self.rms_norm_eps,
            attention_scaling=self._rope_attention_scaling,
        )

    def _prepare_queries_eager(self, q: mx.array, pos_start: int) -> mx.array:
        """Stock query preparation kept as the numeric oracle."""

        q = self.q_layernorm(q)
        if qwen4_opdiet_enabled("rope"):
            cos, sin = _shared_rope_cos_sin_half(
                pos_start,
                int(q.shape[1]),
                self._inv_freq,
                self._rope_attention_scaling,
            )
            return _apply_partial_rope_half(q, cos, sin)
        positions = pos_start + mx.arange(q.shape[1], dtype=mx.int32)
        cos, sin = _rope_cos_sin(
            positions,
            self._inv_freq,
            self._rope_attention_scaling,
        )
        return _apply_partial_rope(q, cos, sin)

    def _prepare_queries(self, q: mx.array, pos_start: int) -> mx.array:
        if not self._prepare_kernel_supported(
            q,
            self.q_layernorm.weight,
            expected_ndim=4,
        ):
            return self._prepare_queries_eager(q, pos_start)

        from mtplx.kernels.qsa_indexer_prepare import (
            qsa_indexer_prepare_queries_metal,
        )

        return qsa_indexer_prepare_queries_metal(
            q,
            self.q_layernorm.weight,
            self._inv_freq,
            pos_start=pos_start,
            eps=self.rms_norm_eps,
            attention_scaling=self._rope_attention_scaling,
        )

    def _compiled_mode(
        self,
        *,
        decode: bool,
        rows: int,
        total: int,
        last_nb: int,
    ) -> str:
        """Choose one fixed-shape graph/output contract on the host."""

        if last_nb <= self.block_topk:
            # Cache maintenance still belongs to the captured indexer even
            # while sparse selection is mathematically the dense causal mask.
            return "update_only"
        if decode and _qsa_flash_enabled():
            return "blocks"
        if not decode and _qsa_large_prefill_enabled(rows, total):
            return "prefill_blocks"
        if (
            not decode
            and _qsa_gather_enabled()
            and rows <= _qsa_gather_max_rows()
            and total >= _qsa_gather_min_context()
        ):
            return "row_tokens"
        return "dense_mask"

    def _projection_output_matches_dtype(self, dtype: mx.Dtype) -> bool:
        """Fail closed when a hidden-source projection may promote dtype."""

        projection = getattr(self, "index_qk_proj", None)
        if projection is None or not callable(projection):
            return False
        weight = getattr(projection, "weight", None)
        if not isinstance(weight, mx.array):
            return False
        expected_width = (self.n_heads + self.kv_heads) * self.head_dim
        if weight.ndim != 2 or int(weight.shape[0]) != expected_width:
            return False

        # QuantizedLinear's U32 packed weight does not determine the output
        # dtype; its scales/biases do.  Dense Linear uses weight dtype.
        scales = getattr(projection, "scales", None)
        if isinstance(scales, mx.array):
            if scales.dtype != dtype:
                return False
            biases = getattr(projection, "biases", None)
            return not isinstance(biases, mx.array) or biases.dtype == dtype
        return weight.dtype == dtype

    def _compiled_route_supported(
        self,
        source: mx.array,
        cache: QSACache,
        *,
        pos_start: int,
        qk_rows_supplied: bool,
        decode: bool,
        mode: str,
    ) -> bool:
        """Static eligibility check; dispatched graph failures are not hidden."""

        if not (_fused_qsa_indexer_enabled() and _compiled_qsa_indexer_enabled()):
            return False
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
        if source.ndim != 3 or int(source.shape[0]) != 1:
            return False
        if int(source.shape[1]) <= 0 or self.kv_heads != 1:
            return False
        rows = int(source.shape[1])
        # Capturing cache maintenance at the dense==sparse boundary created a
        # fresh graph per layer/capacity bucket without doing any selection.
        # Keep prefill update-only work eager; decode retains its established
        # compiled cache lane.
        if (
            not decode
            and mode == "update_only"
            and current_attention_phase() == "prefill"
        ):
            return False
        # Never send a matrix-shaped prefill through the legacy
        # one-threadgroup-per-row scorer.  The dedicated mode is captured only
        # at the canonical chunk width; arbitrary suffix tails use the same
        # Metal prefill selector outside mx.compile, avoiding a new trace for
        # every tail shape.
        if (
            not decode
            and rows >= _qsa_prefill_min_rows()
            and (
                mode not in ("prefill_blocks", "update_only")
                or rows != _qsa_prefill_compile_rows()
            )
        ):
            return False
        if cache.ratio != self.ratio or pos_start != cache.offset:
            return False
        supported_dtypes = (mx.float16, mx.bfloat16, mx.float32)
        if source.dtype not in supported_dtypes:
            return False
        if not (0 < self.head_dim <= 128):
            return False
        rotary_dim = 2 * int(self._inv_freq.shape[0])
        if (
            self._inv_freq.ndim != 1
            or self._inv_freq.dtype != mx.float32
            or rotary_dim <= 0
            or rotary_dim > self.head_dim
            or rotary_dim % 2
        ):
            return False
        if not (1 <= self.block_topk <= 512) or self.ratio <= 0:
            return False
        if self.q_layernorm.weight.dtype != source.dtype:
            return False
        if self.k_layernorm.weight.dtype != source.dtype:
            return False
        if tuple(self.q_layernorm.weight.shape) != (self.head_dim,):
            return False
        if tuple(self.k_layernorm.weight.shape) != (self.head_dim,):
            return False

        if qk_rows_supplied:
            expected_width = (self.n_heads + self.kv_heads) * self.head_dim
            if int(source.shape[2]) != expected_width:
                return False
        elif not self._projection_output_matches_dtype(source.dtype):
            return False

        for backing in (cache.raw_keys, cache.pooled):
            if backing is None:
                continue
            if (
                backing.ndim != 3
                or int(backing.shape[0]) != 1
                or int(backing.shape[2]) != self.head_dim
                or backing.dtype != source.dtype
            ):
                return False
        if pos_start > 0 and (
            cache.raw_keys is None or int(cache.raw_keys.shape[1]) < pos_start
        ):
            return False
        if cache.pooled is None:
            if cache.pooled_len != 0:
                return False
        elif not 0 <= cache.pooled_len <= int(cache.pooled.shape[1]):
            return False
        logical_blocks = (pos_start + int(source.shape[1])) // self.ratio
        pooled_frontier = min(cache.pooled_len, logical_blocks)
        max_new_blocks = (int(source.shape[1]) + self.ratio - 1) // self.ratio
        if logical_blocks - pooled_frontier > max_new_blocks:
            return False

        # Preserve the variable-width decode-gather experiment in the eager
        # oracle.  When flash is also enabled it wins first and its fixed block
        # output is safe for the compiled path.
        if (
            decode
            and mode != "update_only"
            and mode != "blocks"
            and _qsa_gather_decode_enabled()
        ):
            return False

        if mode not in ("update_only", "prefill_blocks") and source.dtype == mx.float32:
            from mtplx.kernels.qsa_indexer_select import (
                qsa_indexer_select_nax_available,
            )

            if not qsa_indexer_select_nax_available():
                return False
        return True

    def _ensure_compiled_backings(
        self,
        cache: QSACache,
        *,
        dtype: mx.Dtype,
        pos_start: int,
        rows: int,
    ) -> tuple[mx.array, mx.array]:
        """Reserve/materialize shape-stable raw and pooled cache leaves."""

        from mtplx.qsa_mtp_precompute import (
            precompute_qsa_replay_capacity,
            qsa_indexer_capacity_bucket,
        )

        raw_existing = 0 if cache.raw_keys is None else int(cache.raw_keys.shape[1])
        pooled_existing = 0 if cache.pooled is None else int(cache.pooled.shape[1])
        # Phase-3 staging can reserve a wider pristine cache before dtype/head
        # width are known.  Include those pending extents rather than
        # accidentally materializing the smaller immediate-call plan.
        raw_existing = max(raw_existing, cache._reserved_raw_capacity)
        pooled_existing = max(pooled_existing, cache._reserved_pooled_capacity)
        plan = precompute_qsa_replay_capacity(
            start_offset=pos_start,
            window_tokens=rows,
            compress_ratio=self.ratio,
            allocation_step=cache.step,
            current_raw_capacity=raw_existing,
            current_pooled_capacity=pooled_existing,
        )
        # The compiled pool stage has one fixed ceil(S/ratio)-block window.
        # For an unaligned first prefill that can be one row wider than the
        # logical complete-block frontier (for example S=1025, ratio=4:
        # logical=256 but staging=257). Bucket the physical requirement itself,
        # rather than taking max() after bucketing 256, which would stay 256.
        max_new_blocks = (rows + self.ratio - 1) // self.ratio
        pooled_capacity = qsa_indexer_capacity_bucket(
            max(
                1,
                plan.complete_blocks,
                pooled_existing,
                max_new_blocks,
            ),
            minimum=cache.step,
        )
        cache.reserve_indexer_capacity(
            raw_capacity=plan.raw_capacity,
            pooled_capacity=pooled_capacity,
        )
        if cache.raw_keys is None:
            cache.raw_keys = mx.zeros(
                (1, cache._reserved_raw_capacity, self.head_dim),
                dtype=dtype,
            )
        if cache.pooled is None:
            cache.pooled = mx.zeros(
                (1, cache._reserved_pooled_capacity, self.head_dim),
                dtype=dtype,
            )
        return cache.raw_keys, cache.pooled

    def _compiled_parameter_signature(self) -> tuple[int, ...]:
        """Identity seal for arrays captured by the lazy graph manager."""

        projection = getattr(self, "index_qk_proj", None)
        leaves = [
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            self._inv_freq,
            getattr(projection, "weight", None),
            getattr(projection, "scales", None),
            getattr(projection, "biases", None),
        ]
        return tuple(id(leaf) for leaf in leaves if isinstance(leaf, mx.array))

    def _get_compiled_indexer_core(self):
        """Build the graph bank only after final checkpoint weights exist."""

        signature = self._compiled_parameter_signature()
        core = self._compiled_indexer_core
        if core is not None and signature == self._compiled_indexer_parameter_signature:
            return core

        from mtplx.kernels.qsa_indexer_compile import QSACompiledIndexerCore

        projection = getattr(self, "index_qk_proj", None)
        core = QSACompiledIndexerCore(
            n_heads=self.n_heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            block_topk=self.block_topk,
            compress_ratio=self.ratio,
            q_norm_weight=self.q_layernorm.weight,
            k_norm_weight=self.k_layernorm.weight,
            inv_freq=self._inv_freq,
            rms_norm_eps=self.rms_norm_eps,
            rope_attention_scaling=self._rope_attention_scaling,
            project_qk=projection if callable(projection) else None,
            selector_scratch_bytes=self._fused_score_scratch_bytes,
            prefill_score_workspace_bytes=_qsa_prefill_score_workspace_bytes(),
        )
        object.__setattr__(self, "_compiled_indexer_core", core)
        object.__setattr__(
            self,
            "_compiled_indexer_parameter_signature",
            signature,
        )
        return core

    def _call_rows_compiled(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: Optional[mx.array],
        *,
        mode: str,
    ):
        """Run and commit one pure explicit-state compiled indexer graph."""

        rows = int(hidden.shape[1])
        total = pos_start + rows
        logical_blocks = total // self.ratio
        source = hidden if qk_rows is None else qk_rows
        raw_keys, pooled = self._ensure_compiled_backings(
            cache,
            dtype=source.dtype,
            pos_start=pos_start,
            rows=rows,
        )
        core = self._get_compiled_indexer_core()
        kwargs = {
            "pos_start": pos_start,
            "total_tokens": total,
            "logical_blocks": logical_blocks,
            "pooled_len": min(cache.pooled_len, logical_blocks),
            "mode": mode,
        }
        if qk_rows is None:
            result = core.select_hidden(hidden, raw_keys, pooled, **kwargs)
        else:
            result = core.select_qk_rows(qk_rows, raw_keys, pooled, **kwargs)

        # The graph is pure: updated cache arrays are explicit outputs.  The
        # host already knows the logical frontier, so committing it needs no
        # .item() synchronization. Attention advances cache.kv.offset later.
        cache.raw_keys = result.raw_keys
        cache.pooled = result.pooled
        cache.pooled_len = logical_blocks
        # The compiled graph rebuilt ``pooled`` wholesale, so the eager lane's
        # fp32-transposed mirror is stale; drop it and let pooled_f32_view
        # rebuild lazily on the next eager read.
        cache.pooled_f32_t = None

        if mode == "update_only":
            return None
        if mode == "blocks":
            block_ids, _, _ = result.selection
            tail_start = ((pos_start + 1) // self.ratio) * self.ratio
            return ("flash", block_ids[0], tail_start)
        if mode == "prefill_blocks":
            block_ids, block_valid, _ = result.selection
            _qsa_prefill_count("compiled_selector")
            return ("flash_prefill", block_ids, block_valid)
        if mode == "row_tokens":
            token_idx, token_ok = result.selection
            return ("gather_rows", token_idx, token_ok)
        if mode == "dense_mask":
            return result.selection[..., :total]
        raise ValueError(f"unknown compiled QSA indexer mode {mode!r}")

    def _call_rows(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: Optional[mx.array],
        *,
        decode: bool,
    ) -> Optional[mx.array]:
        """Shared arithmetic behind the explicit prefill/decode entry points."""

        B, S, _ = hidden.shape
        if decode != (S == 1):
            raise ValueError(
                f"QSA decode route requires S=1 and prefill requires S>1; got S={S}"
            )
        T = pos_start + S  # == the KV length after this forward's update
        last_nb = T // self.ratio
        # Fixed compiled-verify bank (PR #391 step 2, TensorOffsetQSACache):
        # the offset is a graph tensor, so the host-planned compiled indexer,
        # the Metal preparation/selectors and the dense==sparse shortcut
        # cannot run; the stock eager arithmetic is what the trace records.
        fixed_capacity = bool(getattr(cache, "fixed_capacity", False))
        if not fixed_capacity:
            compiled_mode = self._compiled_mode(
                decode=decode,
                rows=S,
                total=T,
                last_nb=last_nb,
            )
            compiled_source = hidden if qk_rows is None else qk_rows
            if self._compiled_route_supported(
                compiled_source,
                cache,
                pos_start=pos_start,
                qk_rows_supplied=qk_rows is not None,
                decode=decode,
                mode=compiled_mode,
            ):
                return self._call_rows_compiled(
                    hidden,
                    pos_start,
                    cache,
                    qk_rows,
                    mode=compiled_mode,
                )

        # qk_rows: the layer's fused shared-input GEMV already produced this
        # projection (MTPLX_FUSED_QSA_QKV) — same rows bit-exactly.  Keeping
        # projection outside the Metal preparation kernel is also the vLLM
        # boundary and preserves packed/quantized Linear dispatches.
        qk = self.index_qk_proj(hidden) if qk_rows is None else qk_rows
        q, k = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        k = k.reshape(B, S, self.head_dim)
        if fixed_capacity:
            if self._verify_glue_rope_idx():
                # MTPLX_QWEN4_VERIFY_GLUE item 'qsa_rope_idx': RMSNorm and
                # partial RoPE through the shipped fused preparation kernel.
                q = self._prepare_queries_m4(q, pos_start)
            else:
                q = self._prepare_queries_eager(q, pos_start)
            cache.write_raw(k)
            cache._last_write_rows = int(S)
            pooled = self._extend_pooled(cache, T)
            return self._select_eager(q, pos_start, cache, pooled, T)
        q = self._prepare_queries(q, pos_start)

        cache.write_raw(k)
        pooled = self._extend_pooled(cache, T)
        nb_total = 0 if pooled is None else pooled.shape[1]

        # Per-query complete-block counts. If every visible prefix fits inside
        # the budget the selection is the full causal mask — skip the work.
        if last_nb <= self.block_topk:
            return None  # dense == sparse in this regime

        pooled_backing = cache.pooled
        large_prefill = not decode and _qsa_large_prefill_enabled(S, T)
        if large_prefill and self._prefill_selector_supported(q, pooled):
            from mtplx.kernels.qsa_indexer_prefill import (
                qsa_indexer_prefill_blocks_metal,
            )

            block_ids, block_valid, _ = qsa_indexer_prefill_blocks_metal(
                q,
                pooled,
                pos_start=pos_start,
                total_tokens=T,
                block_topk=self.block_topk,
                compress_ratio=self.ratio,
                logical_blocks=nb_total,
                score_workspace_bytes=_qsa_prefill_score_workspace_bytes(),
            )
            _qsa_prefill_count("metal_selector")
            return ("flash_prefill", block_ids, block_valid)

        # The original custom selector scores a row serially inside one
        # threadgroup.  Keep it for decode/small verify only; large-S prefill
        # either takes the tiled scorer above or the stock vectorized oracle.
        legacy_fused = (
            _fused_qsa_indexer_enabled()
            and pooled_backing is not None
            and (decode or S < _qsa_prefill_min_rows())
            # Preserve the dormant variable-width decode-gather lane only in
            # the eager oracle. Flash still wins there when both knobs are on.
            and not (decode and _qsa_gather_decode_enabled())
            and self._fused_selector_supported(q, pooled_backing)
        )
        if not legacy_fused:
            return self._select_eager(q, pos_start, cache, pooled, T)

        if decode and _qsa_flash_enabled():
            block_ids, _, _ = self._select_fused(
                q,
                pos_start,
                pooled_backing,
                nb_total,
                T,
                "blocks",
            )
            tail_start = ((pos_start + 1) // self.ratio) * self.ratio
            return ("flash", block_ids[0], tail_start)

        if (
            not decode
            and _qsa_gather_enabled()
            and S <= _qsa_gather_max_rows()
            and T >= _qsa_gather_min_context()
        ):
            token_idx, token_ok = self._select_fused(
                q,
                pos_start,
                pooled_backing,
                nb_total,
                T,
                "row_tokens",
            )
            return ("gather_rows", token_idx, token_ok)

        return self._select_fused(
            q,
            pos_start,
            pooled_backing,
            nb_total,
            T,
            "dense_mask",
        )

    def _call_decode(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: Optional[mx.array],
    ) -> Optional[mx.array]:
        return self._call_rows(hidden, pos_start, cache, qk_rows, decode=True)

    def _call_prefill(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: Optional[mx.array],
    ) -> Optional[mx.array]:
        return self._call_rows(hidden, pos_start, cache, qk_rows, decode=False)

    def __call__(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        B, S, _ = hidden.shape
        if B != 1:
            raise NotImplementedError("qwen4_exp QSA serves single sequences (B=1)")
        if S == 1:
            return self._call_decode(hidden, pos_start, cache, qk_rows)
        return self._call_prefill(hidden, pos_start, cache, qk_rows)


def _qsa_rows_gather_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    token_idx: mx.array,
    token_ok: mx.array,
    scale: float,
    gather_kv: Any,
) -> mx.array:
    """Attention over per-row gathered tokens (rows-gather lane).

    Adapting the per-query single-take gather + GQA head-group broadcast
    from community PR #380 by @maceip, minus its mask re-gather (the
    selection itself carries validity here, so no second gathered copy is
    ever built — the S=1 lane's receipt priced each extra copy at -5.25%).

    q is [1, H, S, D]; k/v are the cache's [1, H_kv, T, D] slices;
    token_idx/token_ok are [S, K]. Keys differ per row, so fused SDPA over
    a shared KV sequence cannot serve this; the whole S stays in-graph as
    one broadcast GEMM pair, and the 12x-repeated K/V working set is never
    materialized: q is viewed [1, H_kv, rep, S, 1, D] against
    [1, H_kv, 1, S, D, K]. Invalid slots score -inf before the fp32
    softmax, identical math to the dense bool-mask product over the same
    visible set.
    """
    B, H, S, D = q.shape
    H_kv = int(k.shape[1])
    K = int(token_idx.shape[-1])
    k_sel, v_sel = gather_kv(k, v, token_idx)
    neg = mx.array(-mx.inf, dtype=mx.float32)
    if H != H_kv:
        rep = H // H_kv
        q_view = q.reshape(1, H_kv, rep, S, 1, D)
        k_view = k_sel.swapaxes(-1, -2).reshape(1, H_kv, 1, S, D, K)
        scores = mx.matmul(q_view, k_view).squeeze(-2).astype(mx.float32) * scale
        scores = mx.where(token_ok[None, None, None], scores, neg)
        probs = mx.softmax(scores, axis=-1).astype(q.dtype)
        v_view = v_sel.reshape(1, H_kv, 1, S, K, D)
        out = mx.matmul(probs[..., None, :], v_view).squeeze(-2)
        return out.reshape(1, H, S, D)
    scores = (
        mx.matmul(q[..., None, :], k_sel.swapaxes(-1, -2)).squeeze(-2).astype(mx.float32)
        * scale
    )
    scores = mx.where(token_ok[None, None], scores, neg)
    probs = mx.softmax(scores, axis=-1).astype(q.dtype)
    return mx.matmul(probs[..., None, :], v_sel).squeeze(-2)


def _qsa_prefill_gather_attention(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    compress_ratio: int,
    scale: float,
    tile_rows: int,
) -> mx.array:
    """Portable bounded attention over the flash_prefill block contract.

    The universal tier between the NAX flash kernel and the dense-mask
    reconstruction (approach from oMLX PR #3244's portable lane, Apache-2.0,
    reimplemented on our block/validity contract): each query row attends to
    its selected complete blocks plus its visible tail — a constant
    ``topk*ratio + ratio`` token width — instead of the full [S, T] masked
    context. Row tiles bound the gathered K/V working set; a per-tile eval
    retires each tile's gathered copies before the next tile is built.

    q is [1, H, S, D]; keys/values are the FULL cache backings
    [1, H_kv, cap, D] (never sliced-contiguous — gathers index absolute
    rows, all < total_tokens). Selection semantics match the dense mask
    exactly: selected blocks are complete (< each row's tail start), the
    tail runs to the row's own position, and invalid slots score -inf.
    """

    S = int(q.shape[2])
    ratio = int(compress_ratio)
    arange_ratio = mx.arange(ratio, dtype=mx.int32)
    outputs = []
    for r0 in range(0, S, tile_rows):
        r1 = min(r0 + tile_rows, S)
        qpos = mx.arange(pos_start + r0, pos_start + r1, dtype=mx.int32)
        nb_q = (qpos + 1) // ratio
        ids_t = block_ids[r0:r1]
        ok_t = block_valid[r0:r1]
        tok_blocks = (
            ids_t.astype(mx.int32)[:, :, None] * ratio + arange_ratio
        ).reshape(r1 - r0, -1)
        blocks_ok = mx.repeat(ok_t, ratio, axis=1)
        tail_tok = nb_q[:, None] * ratio + arange_ratio
        tail_ok = tail_tok <= qpos[:, None]
        token_idx = mx.concatenate([tok_blocks, tail_tok], axis=1)
        token_ok = mx.concatenate([blocks_ok, tail_ok], axis=1)
        token_idx = mx.where(token_ok, token_idx, mx.array(0, dtype=mx.int32))
        out_t = _qsa_rows_gather_attention(
            q[:, :, r0:r1],
            keys,
            values,
            token_idx,
            token_ok,
            scale,
            _qsa_stock_rows_gather_kv,
        )
        mx.eval(out_t)
        _owner_progress_tick()
        outputs.append(out_t)
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def _qsa_blocks_to_dense_mask(
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    compress_ratio: int,
) -> mx.array:
    """Reconstruct the exact dense QSA mask for an unsupported consumer.

    The normal large-prefill route never calls this: its sparse attention
    kernel consumes block IDs directly.  Keeping the reconstruction here makes
    unsupported geometry/dtype combinations correctness-preserving without
    hiding a Metal dispatch failure.  A sentinel column absorbs every invalid
    top-k slot so no padded entry can accidentally select block zero.
    """

    rows = int(block_ids.shape[0])
    ratio = int(compress_ratio)
    logical_blocks = int(total_tokens) // ratio
    qpos = mx.arange(pos_start, pos_start + rows, dtype=mx.int32)
    complete_for_row = (qpos + 1) // ratio
    in_range = (
        (block_ids >= 0)
        & (block_ids < logical_blocks)
        & (block_ids < complete_for_row[:, None])
    )
    valid = block_valid & in_range
    sentinel = mx.array(logical_blocks, dtype=mx.int32)
    safe_ids = mx.where(valid, block_ids, sentinel)
    selected = mx.zeros((rows, logical_blocks + 1), dtype=mx.bool_)
    selected = mx.put_along_axis(
        selected,
        safe_ids.astype(mx.int64),
        mx.array(True),
        axis=-1,
    )[:, :logical_blocks]
    token_selected = mx.repeat(selected, ratio, axis=-1)
    complete_token_count = logical_blocks * ratio
    if complete_token_count < int(total_tokens):
        token_selected = mx.concatenate(
            [
                token_selected,
                mx.zeros(
                    (rows, int(total_tokens) - complete_token_count),
                    dtype=mx.bool_,
                ),
            ],
            axis=-1,
        )

    tail_start = complete_for_row * ratio
    tpos = mx.arange(total_tokens, dtype=mx.int32)
    tail = (tpos[None, :] >= tail_start[:, None]) & (tpos[None, :] <= qpos[:, None])
    causal = tpos[None, :] <= qpos[:, None]
    return ((token_selected | tail) & causal)[None, None]


class Attention(nn.Module):
    """Gated GQA (qwen3_5 style: double-width q_proj, sigmoid output gate,
    per-head q/k RMSNorm, partial rotary) masked by the QSA indexer."""

    # The QSA indexer mask is part of this module's semantics (and __call__
    # takes (x, cache)): any generic dense-SDPA rewrite that replaces
    # __call__ would silently drop the sparse selection. attention_split
    # honors this and never hooks the class.
    _mtplx_generic_sdpa_rewrites_unsupported = True

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim * 2, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args) if args.indexer_n_heads else None
        self._inv_freq, self._rope_attention_scaling = (
            _rope_inv_freq_and_scaling_for(args)
        )
        self._mrope_axes = (
            mx.array(
                _build_mrope_axes(args.mrope_section, args.mrope_interleaved),
                dtype=mx.int32,
            )
            if args.mrope_section
            and sum(args.mrope_section) == int(args.rotary_dim) // 2
            else None
        )

    def _verify_glue_rope(self, rows: int) -> bool:
        """True when ``MTPLX_QWEN4_VERIFY_GLUE``'s ``qsa_rope`` serves this call.

        Host-only, from the verdict the install probe recorded at model build
        outside any trace.  Width is a NARROWING, not a failure: the kernel is
        a latency play on the 1..8-row decode/verify window and prefill is a
        regime nothing here has measured, so prefill keeps the stock chain.
        """

        if not qwen4_verify_glue_enabled("qsa_rope"):
            return False
        from mtplx import qwen4_verify_glue as _glue

        if not _glue.serves_rows(rows):
            return False
        return _glue.qsa_rope_installed()

    def __call__(self, x: mx.array, cache: QSACache) -> mx.array:
        B, S, _ = x.shape
        pos_start = cache.offset
        vrope = vision_rope_state()

        fused = getattr(self, "qkv_fused", None)
        if fused is not None:
            # One shared-input GEMV replaces q/k/v (+ indexer qk when its
            # pack precision matches; the v2.10 artifact's 4-bit group-64
            # indexer can therefore join this dispatch). Row-concat is
            # bit-exact per row; MTPLX_FUSED_QSA_QKV sanitize fusion.
            outs = fused(x)
            if len(outs) == 4:
                q, k, v, idx_rows = outs
            else:
                q, k, v = outs
                idx_rows = None
            sel_mask = (
                self.indexer(x, pos_start, cache, qk_rows=idx_rows)
                if self.indexer is not None
                else None
            )
        else:
            sel_mask = None
            if self.indexer is not None:
                sel_mask = self.indexer(x, pos_start, cache)
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q, gate = mx.split(q.reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        k = k.reshape(B, S, self.n_kv_heads, -1)
        v = v.reshape(B, S, self.n_kv_heads, -1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if vrope is not None and self._mrope_axes is not None:
            # Vision request: image tokens rope at (t, h, w) grid positions
            # from the request's M-RoPE table; spans past the table (decode
            # continuation) are equal-axes at sequence_index + delta, which
            # is plain rope shifted by delta. Table and delta are derived
            # per request from content — nothing new rides cache state.
            table, delta = vrope
            end = pos_start + S
            if table is not None and end <= int(table.shape[1]):
                cos, sin = _mrope_cos_sin(
                    table[:, pos_start:end], self._inv_freq, self._mrope_axes
                )
            else:
                positions = mx.arange(
                    pos_start + delta, pos_start + delta + S, dtype=mx.int32
                )
                cos, sin = _rope_cos_sin(
                    positions, self._inv_freq, self._rope_attention_scaling
                )
        elif vrope is None and self._verify_glue_rope(int(S)):
            # MTPLX_QWEN4_VERIFY_GLUE item 'qsa_rope': the table build and
            # both rotations as ONE dispatch. Same arithmetic, same order --
            # the install probe proved it bit-exact against whichever stock
            # spelling this process armed.
            from mtplx.kernels.qwen4_m4_rope import rope_qk

            rotated_q, k = rope_qk(
                q,
                k,
                self._inv_freq,
                pos_start=pos_start,
                attention_scaling=self._rope_attention_scaling,
            )
            q = rotated_q
            cos = sin = None
        elif qwen4_opdiet_enabled("rope"):
            # Text rope: one half-width table per (pos_start, S) instead of a
            # full-width table per consumer. The indexer above already asked
            # for this exact table, so this is a memo hit inside the layer.
            cos, sin = _shared_rope_cos_sin_half(
                pos_start, int(S), self._inv_freq, self._rope_attention_scaling
            )
            q = _apply_partial_rope_half(q, cos, sin)
            k = _apply_partial_rope_half(k, cos, sin)
            cos = sin = None
        else:
            # ``pos_start`` may be a graph tensor (fixed compiled verify bank).
            positions = pos_start + mx.arange(S, dtype=mx.int32)
            cos, sin = _rope_cos_sin(
                positions, self._inv_freq, self._rope_attention_scaling
            )
        if cos is not None:
            q = _apply_partial_rope(q, cos, sin)
            k = _apply_partial_rope(k, cos, sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        k, v = cache.kv.update_and_fetch(k, v)
        T = k.shape[2]

        if vrope is not None:
            # Vision request: QSA sparse selection is bypassed and attention
            # runs dense-causal — the reference qwen4_exp implementation
            # serves multimodal exactly this way (its sparse fast paths
            # exclude M-RoPE). The indexer above still ran, so the QSA cache
            # streams (raw/pooled) stay byte-identical with text serving and
            # bank state keeps one format. Masks key on sequence order,
            # which remains correct under M-RoPE; only rope reads the axes.
            sel_mask = None

        if isinstance(sel_mask, tuple) and sel_mask and sel_mask[0] == "flash":
            # Block-sparse flash attention over the indexer's exact visible
            # set. Reads the cache BACKING arrays in place at their
            # allocation stride — the :T slice above is non-contiguous, and
            # forcing it contiguous would copy the entire KV.
            from mtplx.kernels.qsa_flash_skip import qsa_flash_skip

            _, blk_idx, tail_start = sel_mask
            out = qsa_flash_skip(
                q.reshape(self.n_heads, self.head_dim),
                cache.kv.keys,
                cache.kv.values,
                blk_idx,
                T,
                tail_start,
                self.scale,
            )
            out = out.reshape(B, S, -1)
            return self.o_proj(out * mx.sigmoid(gate))

        if isinstance(sel_mask, tuple) and sel_mask and sel_mask[0] == "flash_prefill":
            # Large-S prefill consumes compact per-row block selections
            # directly from the full cache backing.  This is the point of the
            # prefill port: no [S,T] selection expansion and no per-row K/V
            # gather on the supported production geometry.
            from mtplx.kernels.qsa_prefill_flash import (
                qsa_prefill_flash,
                qsa_prefill_flash_supported,
            )

            _, block_ids, block_valid = sel_mask

            def _qsa_flash_supported() -> bool:
                return bool(
                    _qsa_prefill_flash_attention_enabled(S, T)
                ) and bool(
                    qsa_prefill_flash_supported(
                        q,
                        cache.kv.keys,
                        cache.kv.values,
                        block_ids,
                        block_valid,
                        pos_start=pos_start,
                        total_tokens=T,
                        scale=self.scale,
                    )
                )

            def _qsa_flash_call():
                return qsa_prefill_flash(
                    q,
                    cache.kv.keys,
                    cache.kv.values,
                    block_ids,
                    block_valid,
                    pos_start=pos_start,
                    total_tokens=T,
                    scale=self.scale,
                )

            def _qsa_direct_supported() -> bool:
                if not _qsa_prefill_direct_attention_enabled(S, T):
                    return False
                from mtplx.kernels.qsa_prefill_direct import (
                    qsa_prefill_direct_supported,
                )

                return bool(
                    qsa_prefill_direct_supported(
                        q,
                        k,
                        v,
                        block_ids,
                        block_valid,
                        pos_start=pos_start,
                        total_tokens=T,
                        scale=self.scale,
                        compress_ratio=self.indexer.ratio,
                        block_topk=self.indexer.block_topk,
                    )
                )

            def _qsa_direct_call():
                from mtplx.kernels.qsa_prefill_direct import qsa_prefill_direct

                out = qsa_prefill_direct(
                    q,
                    k,
                    v,
                    block_ids,
                    block_valid,
                    pos_start=pos_start,
                    total_tokens=T,
                    scale=self.scale,
                    compress_ratio=self.indexer.ratio,
                    block_topk=self.indexer.block_topk,
                )
                return out

            def _qsa_gather_call():
                return _qsa_prefill_gather_attention(
                    q,
                    cache.kv.keys,
                    cache.kv.values,
                    block_ids,
                    block_valid,
                    pos_start=pos_start,
                    total_tokens=T,
                    compress_ratio=self.indexer.ratio,
                    scale=self.scale,
                    tile_rows=_qsa_prefill_gather_tile_rows(),
                )

            out = _qsa_prefill_dispatch_tier(
                flash_supported=_qsa_flash_supported,
                flash_call=_qsa_flash_call,
                direct_supported=_qsa_direct_supported,
                direct_call=_qsa_direct_call,
                gather_enabled=_qsa_prefill_gather_enabled(),
                gather_call=_qsa_gather_call,
            )
            if out is not None:
                out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
                return self.o_proj(out * mx.sigmoid(gate))

            # Static unsupported geometry falls back exactly.  Once the
            # supported kernel is dispatched, failures propagate instead of
            # being hidden behind a dense retry.
            sel_mask = _qsa_blocks_to_dense_mask(
                block_ids,
                block_valid,
                pos_start=pos_start,
                total_tokens=T,
                compress_ratio=self.indexer.ratio,
            )

        if isinstance(sel_mask, tuple) and sel_mask and sel_mask[0] == "gather_rows":
            # Rows-gather lane (S>1): each verify/pipeline row reads only
            # its own selected blocks + tail instead of the full context
            # through a dense [S, T] mask. See _qsa_rows_gather_attention
            # (adapting community PR #380 by @maceip).
            _, tok_idx, tok_ok = sel_mask
            out = _qsa_rows_gather_attention(
                q,
                k,
                v,
                tok_idx,
                tok_ok,
                self.scale,
                _qsa_rows_gather_kv_route(cache, S),
            )
            out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
            return self.o_proj(out * mx.sigmoid(gate))

        if sel_mask is not None and sel_mask.ndim == 1:
            # QSA gather lane (decode): the indexer returned the selected
            # token indices — attention reads budget+tail keys/values
            # instead of the full context through a dense bool mask. All
            # gathered tokens are visible, so no mask is needed; the
            # softmax over the same visible set is identical math.
            k = mx.take(k, sel_mask, axis=2)
            v = mx.take(v, sel_mask, axis=2)
            mask = None
        elif sel_mask is not None:
            mask = sel_mask
        elif S > 1:
            qpos = pos_start + mx.arange(S, dtype=mx.int32)
            tpos = mx.arange(T, dtype=mx.int32)
            mask = (tpos[None, :] <= qpos[:, None])[None, None]
        else:
            mask = None

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(vocab: int, ngram_size: int, ple_index: int, seed: int):
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(vocab, 1)) // 2)
    base_seed = seed + _PRIME_1 * ple_index
    out = []
    for i in range(ngram_size):
        v = (base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64
        out.append(2 * (_splitmix64(v) % half_bound) + 1)
    return out


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def _head_vocab_layout(base: int, heads: int, ple_index: int):
    sizes, offsets, total = [], [], 0
    prime = base - 1
    # global head index runs across PLE layers; sizes are consecutive primes
    for h in range(ple_index * heads + heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        if h >= ple_index * heads:
            sizes.append(prime)
            offsets.append(total)
            total += prime
    return sizes, offsets, total


class NGramTable(nn.Module):
    """The hashed n-gram embedding. Two modes:

    * materialized (tiny/test configs): a QuantizedEmbedding-shaped or plain
      `weight` parameter, gathered with mx.take.
    * sidecar (the real 51B table): `attach_sidecar()` points row gathers at
      numpy memmaps over ngram-table.safetensors. Rows are dequantized after
      the gather; only touched pages ever become resident. No parameter is
      registered, so the table never counts as loadable weight.
    """

    def __init__(self, rows: int, dim: int, sidecar: bool = False):
        super().__init__()
        self.rows = rows
        self.dim = dim
        self._sidecar_mode = sidecar
        if not sidecar:
            self.weight = mx.zeros((max(rows, 1), dim))
        self._sidecar = None

    def attach_sidecar(self, path: Path):
        self.pop("weight", None)
        header, data_start = _read_safetensors_header(path)
        meta = header.get("__metadata__", {})
        entries = {}
        names = ("weight",) if int(meta.get("ngram_bits", 4)) == 0 else ("weight", "scales", "biases")
        for name in names:
            info = header[f"ngram.{name}"]
            entries[name] = (info, data_start)
        self._sidecar = _SidecarGather(
            path,
            entries,
            bits=int(meta.get("ngram_bits", 4)),
            group_size=int(meta.get("ngram_group_size", 32)),
        )

    def attach_resident(self, path: Path) -> bool:
        """Materialize the table as RESIDENT mx arrays for in-graph gathers.

        mx.load's lazy arrays are NOT page-granular — the first eval reads
        the whole tensor (measured: a mid-request ~32G materialization train
        collapsed decode to 8 t/s and trips the GPU watchdog in a bare
        process). So residency is paid ONCE, up front, at model load — and
        only on machines whose memory plan can afford it (the pipelined AR
        lane needs in-graph gathers on LAZY ids; smaller machines keep the
        pread sidecar + staged classic loop, which SSD serves at zero cost).
        """
        try:
            header, _ = _read_safetensors_header(path)
            meta = header.get("__metadata__", {})
            bits = int(meta.get("ngram_bits", 4))
            started = time.perf_counter()
            raw = mx.load(str(path))
            if bits == 0:
                parts = (raw["ngram.weight"], None, None)
                mx.eval(parts[0])
                nbytes = parts[0].nbytes
            else:
                parts = (
                    raw["ngram.weight"],
                    raw["ngram.scales"],
                    raw["ngram.biases"],
                )
                mx.eval(*parts)
                nbytes = sum(p.nbytes for p in parts)
            self._lazy_parts = parts
            self._lazy_bits = bits
            self._lazy_group = int(meta.get("ngram_group_size", 32))
            self.prefer_lazy = False
            print(
                f"[qwen4_exp] ngram table resident: {nbytes / 2**30:.1f}G in "
                f"{time.perf_counter() - started:.1f}s (pipelined-AR lane armed)",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[qwen4_exp] ngram resident bind failed: {exc!r}", flush=True)
            self._lazy_parts = None
            return False

    def _lazy_gather(self, ids: mx.array) -> mx.array:
        w, s, b = self._lazy_parts
        rows_w = w[ids]
        if self._lazy_bits == 0:
            return rows_w
        return mx.dequantize(
            rows_w, s[ids], b[ids], group_size=self._lazy_group, bits=self._lazy_bits
        )

    def __call__(self, ids: mx.array) -> mx.array:
        if getattr(self, "prefer_lazy", False) and getattr(self, "_lazy_parts", None) is not None:
            return self._lazy_gather(ids)
        if self._sidecar is not None:
            return self._sidecar(ids, self.dim)
        if self._sidecar_mode:
            raise RuntimeError(
                "qwen4_exp n-gram table sidecar was never attached — "
                "ngram-table.safetensors is missing from the model directory"
            )
        return self.weight[ids]


class _SidecarGather:
    """Row gather over the SSD-resident table.

    The row ids are a pure function of the token ids, so every gather is
    known before it is needed. Cold mmap faults are serial (~60us each on
    M5-class NVMe) while os.pread releases the GIL — so a threaded pread
    warm-up pass first (QD16, ~12.6us effective) then the numpy fancy-index
    hits page-cache-warm rows (~1.2us). Measured 2026-08-26: without this,
    a cold 100k-token prefill pays ~36s of serial faults; with it, ~7.5s of
    parallel IO that overlaps page-cache warming. MTPLX_NGRAM_PREFETCH=0
    disables it (A/B arm); prefetch_batches is the engagement receipt.
    """

    # Decode-step gathers are a few dozen rows; the LRU serves those. A
    # prefill gather is millions of ids — per-row cache bookkeeping there
    # would cost more than the IO, so big gathers bypass to the vectorized
    # memmap path (which still warms the page cache for later decode).
    _HOT_PATH_MAX_ROWS = 4096

    def __init__(self, path: Path, entries, bits: int, group_size: int):
        import numpy as np
        from collections import OrderedDict

        from mtplx.ple_row_gather import madvise_choice

        self.bits = bits
        self.group_size = group_size
        self._maps = {}
        self._row_meta = []
        self.vectorized_gathers = 0
        self.pread_gathers = 0
        self.madvise_applied, _advice_value = madvise_choice()
        self._fd = os.open(str(path), os.O_RDONLY)
        for name, (info, data_start) in entries.items():
            dtype = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16}[info["dtype"]]
            shape = tuple(info["shape"])
            offset = data_start + info["data_offsets"][0]
            mm = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)
            try:
                # Default (flag off): MADV_RANDOM -- row ids are
                # hash-scattered, so readahead around a mapping fault is
                # wasted IO.  Under MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY the
                # mapping faults are the ascending pre-touch and the
                # vectorised gather's residual misses instead, which readahead
                # helps, so `madvise_choice` flips it; MTPLX_QWEN4_NGRAM_MADVISE
                # overrides either way.  (pread(2) never consulted this advice
                # at all, so the pread pool's behaviour is unchanged.)
                mm._mmap.madvise(_advice_value)
            except (AttributeError, OSError, ValueError):
                self.madvise_applied = "unavailable"
            self._maps[name] = (mm, info["dtype"])
            itemsize = 4 if info["dtype"] == "U32" else 2
            self._row_meta.append((offset, int(shape[1]) * itemsize))
        self._pool = None
        self.prefetch_batches = 0
        self.lookahead_batches = 0
        if os.environ.get("MTPLX_NGRAM_PREFETCH", "1") != "0":
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(
                max_workers=16, thread_name_prefix="ngram-prefetch"
            )
        # Hot-row LRU: raw row bytes per map, keyed by row id. Row
        # popularity is Zipf (common n-grams recur constantly) even though
        # row PLACEMENT is hash-uniform, so a bounded RAM-held hot set
        # serves most decode gathers with zero pread and zero memmap touch
        # — and keeps its speed when macOS reclaims the file's page cache
        # under memory pressure. All access is on the single generation
        # thread (stage()/forward); the pread pool only warms pages and
        # never touches this dict. MTPLX_NGRAM_HOT_MB sizes it
        # (default 1024; 0 disables).
        self._hot = OrderedDict()
        self._hot_row_bytes = max(1, sum(rb for _, rb in self._row_meta))
        try:
            hot_mb = int(os.environ.get("MTPLX_NGRAM_HOT_MB", "1024"))
        except ValueError:
            hot_mb = 1024
        self._hot_cap_rows = (max(0, hot_mb) * 2**20) // self._hot_row_bytes
        self.hot_hits = 0
        self.hot_misses = 0
        self.prewarm_at_load = self._prewarm_table(path)

    def _prewarm_table(self, path: Path) -> dict:
        """Pre-read as much of the table as the budget allows (MTPLX_NGRAM_PREWARM).

        Last in construction on purpose: the budget is measured against FREE
        memory at this instant, which is only meaningful once the weights are
        mapped, and the hotness order needs the row geometry and the pread
        pool that the lines above build.

        On by default in ``auto`` mode.  Production serves at whatever the
        page cache happens to hold, and the ~30 GiB table's residency is the
        single largest source of first-chunk variance (1.9 s vs 4.4 s, w22,
        concordant with the pre-read's own throughput) as well as 56 vs 68.8
        tok/s on decode.  A benchmark harness reads the table itself
        (--prewarm-ngram-table); the daemon had no equivalent.

        Never raises: a pre-read is an optimisation, and it must not be the
        reason a model fails to load.
        """

        from mtplx.ple_row_gather import (
            format_prewarm_plan,
            format_prewarm_result,
            record_prewarm,
            run_prewarm,
        )

        receipt = run_prewarm(
            table_path=path,
            row_meta=tuple(self._row_meta),
            fd=self._fd,
            submit=None if self._pool is None else self._pool.submit,
        )
        # Two lines on the server's startup log: what was decided, and what it
        # cost.  Guarded -- a closed stdout (app-launched daemon, redirected
        # child) must not fail a load.
        try:
            print(format_prewarm_plan(receipt), flush=True)
            print(format_prewarm_result(receipt), flush=True)
        except (OSError, ValueError):
            pass
        return record_prewarm(
            receipt,
            enabled=bool(receipt.get("budget_bytes")),
            source=str(receipt.get("source", "default")),
        )

    def clear_hot_cache(self) -> int:
        """Drop every RAM-held row between independent benchmark runs."""

        cleared = len(self._hot)
        self._hot.clear()
        self.hot_hits = 0
        self.hot_misses = 0
        return cleared

    def _warm(self, rows, *, counted: bool = True) -> None:
        for future in self._submit_warm(rows, counted=counted):
            future.result()

    def submit_warm(self, rows):
        """Submit page-warming reads without waiting for their completion."""
        return self._submit_warm(rows, counted=True)

    def _submit_warm(self, rows, *, counted: bool):
        """The warm pass.  ``counted`` keeps the lookahead worker's batches out
        of ``prefetch_batches``: that counter is the decode-lane engagement
        receipt, and a worker thread incrementing it would both race the owner
        thread and change what the existing receipts mean."""
        fd = self._fd
        metas = self._row_meta

        def touch(chunk):
            for r in chunk:
                for base, rb in metas:
                    os.pread(fd, rb, base + int(r) * rb)

        step = max(1, min(64, (len(rows) + 31) // 32))
        chunks = [rows[i : i + step] for i in range(0, len(rows), step)]
        futures = tuple(self._pool.submit(touch, chunk) for chunk in chunks)
        if counted:
            self.prefetch_batches += 1
        else:
            self.lookahead_batches += 1
        return futures

    def prepare_rows_np(
        self,
        flat,
        names=("weight", "scales", "biases"),
        *,
        vectorized: bool | None = None,
        record=None,
    ):
        """Worker-thread half of the big-gather branch of `_rows_matrices`.

        Returns ``(unique_count, matrices)`` -- exactly what `_rows_matrices`
        would return for these ids -- or ``None`` when the ids would take the
        hot-row LRU instead.  That LRU is owner-thread-only state (`stage()` /
        `forward`), so the worker must never reach it; every real 2,048-token
        prefill chunk is far above `_HOT_PATH_MAX_ROWS`, but this checks it
        rather than assuming it.

        Safe off the owner thread: the memmaps are read-only, `flat` is the
        caller's private array, and the only shared mutation is the separate
        `lookahead_batches` counter.  Bit-identical by construction -- it is
        the same expression over the same maps and the same ids.
        """

        import numpy as np

        from mtplx.ple_row_gather import gather_matrices, warm_decision

        uniq, inverse = np.unique(flat, return_inverse=True)
        if 0 < len(uniq) <= self._HOT_PATH_MAX_ROWS and self._hot_cap_rows:
            return None
        maps = {name: self._maps[name][0] for name in names}
        path = "pread"
        fraction = None
        if vectorized is None:
            from mtplx.ple_row_gather import enabled as _vectorized_enabled

            vectorized = _vectorized_enabled()
        if vectorized and len(uniq) > self._HOT_PATH_MAX_ROWS:
            # MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY: the warm pass is ~165 ms of
            # GIL-contended pread per 32,768 rows and the fancy index behind
            # it is 0.44 ms, so on a page-warm table the warm pass IS the
            # gather.  Skip it only when mincore says the rows this gather
            # will read are already in core -- a demand-faulted mmap is flat
            # at 1.40 GiB/s against pooled pread's 12.9, so guessing warm on a
            # cold table would stall the generation thread with the GIL held.
            path, fraction = warm_decision(list(maps.values()), uniq)
        if path == "vectorized":
            self.vectorized_gathers += 1
        else:
            self.pread_gathers += 1
            if self._pool is not None and len(uniq):
                self._warm(uniq, counted=False)
        if record is not None:
            record["path"] = path
            record["rows"] = int(len(uniq))
            record["resident_fraction"] = fraction
        return (int(len(uniq)), gather_matrices(maps, uniq, inverse, names))

    def _rows_matrices(self, flat, names):
        """Raw row matrices (one per map, flat order) — through the hot-row
        LRU for decode-sized gathers, straight off the memmaps otherwise.
        Values are identical either way: the cache stores the same raw row
        bytes the maps would return."""
        import numpy as np

        uniq, inverse = np.unique(flat, return_inverse=True)
        if not (0 < len(uniq) <= self._HOT_PATH_MAX_ROWS and self._hot_cap_rows):
            from mtplx.ple_row_gather import (
                enabled as _vectorized_enabled,
                gather_matrices,
                warm_decision,
            )

            maps = {name: self._maps[name][0] for name in names}
            path = "pread"
            # The probe costs ~0.5 ms; the warm pass it decides costs ~165 ms
            # per 32,768 rows.  Below the sidecar's own hot-row threshold the
            # ratio inverts (with MTPLX_NGRAM_HOT_MB=0 every decode gather
            # lands here), so small gathers keep the shipped path outright.
            if (
                _vectorized_enabled()
                and len(uniq) > self._HOT_PATH_MAX_ROWS
            ):
                # The same measured choice the worker makes.  This branch is
                # the OWNER thread -- a lookahead miss, an inert lane, or a
                # verify-width gather -- so the ~165 ms per 32,768 rows the
                # warm pass costs lands directly on the generation loop here.
                path, _fraction = warm_decision(list(maps.values()), uniq)
            if path == "vectorized":
                self.vectorized_gathers += 1
            else:
                self.pread_gathers += 1
                if self._pool is not None and len(uniq):
                    self._warm(uniq)
            return gather_matrices(maps, uniq, inverse, names)
        hot = self._hot
        miss = [int(r) for r in uniq if int(r) not in hot]
        if miss:
            miss_np = np.asarray(miss, dtype=np.int64)
            if self._pool is not None:
                self._warm(miss_np)
            fetched = {
                name: np.ascontiguousarray(self._maps[name][0][miss_np])
                for name in names
            }
            for i, r in enumerate(miss):
                hot[r] = tuple(fetched[name][i] for name in names)
        self.hot_hits += len(uniq) - len(miss)
        self.hot_misses += len(miss)
        rows = []
        for r in uniq:
            key = int(r)
            rows.append(hot[key])
            hot.move_to_end(key)
        while len(hot) > self._hot_cap_rows:
            hot.popitem(last=False)
        return {
            name: np.stack([row[j] for row in rows])[inverse]
            for j, name in enumerate(names)
        }

    def gather_raw_np(self, flat) -> tuple[mx.array, mx.array, mx.array]:
        """Gather the exact packed q4 row payload without dequantizing it."""

        names = ("weight", "scales", "biases")
        mats = self._rows_matrices(flat, names)
        parts = []
        for name in names:
            dt = self._maps[name][1]
            rows = mx.array(mats[name])
            if dt == "BF16":
                rows = rows.view(mx.bfloat16)
            elif dt == "F16":
                rows = rows.view(mx.float16)
            parts.append(rows)
        return tuple(parts)

    def gather_np(self, flat, prepared=None) -> mx.array:
        """Gather+dequantize rows for MATERIALIZED numpy int64 ids — the
        staged fast path: no graph tensor is evaluated here, so calling
        this before the step's graph is built costs no GPU sync.

        ``prepared`` is a row-matrix dict produced earlier by
        :meth:`prepare_rows_np` for these exact ids (the prefill lookahead).
        Only the MLX array construction below then runs on this thread; the
        bytes are the same either way."""
        if self.bits == 0:  # raw bf16 rows, no dequantize
            mats = (
                self._rows_matrices(flat, ("weight",))
                if prepared is None
                else prepared
            )
            dt = self._maps["weight"][1]
            rows = mx.array(mats["weight"])
            return rows.view(mx.bfloat16 if dt == "BF16" else mx.float16)
        names = ("weight", "scales", "biases")
        mats = (
            self._rows_matrices(flat, names) if prepared is None else prepared
        )
        parts = []
        for name in names:
            dt = self._maps[name][1]
            rows = mx.array(mats[name])
            if dt == "BF16":
                rows = rows.view(mx.bfloat16)
            elif dt == "F16":
                rows = rows.view(mx.float16)
            parts.append(rows)
        w, s, b = parts
        return mx.dequantize(w, s, b, group_size=self.group_size, bits=self.bits)

    def __call__(self, ids: mx.array, dim: int) -> mx.array:
        import numpy as np

        flat = np.asarray(ids.reshape(-1), dtype=np.int64)
        return self.gather_np(flat).reshape(*ids.shape, dim)


def _read_safetensors_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def _ngram_resident_policy() -> bool:
    """Should the n-gram table go RAM-resident (arming the pipelined AR
    lane)? Delegates to memory_plan.ngram_table_resident_policy — THE
    single source: the server's Metal floor and the memory plan consult
    the same function, so gather behavior and memory accounting can never
    disagree. (History: resident on 128G wired ~99G and kernel-panicked
    the machine twice, 2026-08-26 — auto arms only at >=160G.)"""
    from mtplx.memory_plan import ngram_table_resident_policy

    return ngram_table_resident_policy()


#: Cumulative host seconds and call count inside `NGramEmbedding.stage` --
#: the per-chunk PLE gather the census measures as 8 host-late stalls totalling
#: 2,313 ms.  One float and one int, bumped once per stage call; the prefill
#: loop snapshots them per chunk so the receipt can show WHERE the run-to-run
#: prefill variance lives.  Cumulative on purpose: deltas compose, a reset
#: shared between the generation loop and the model would not.
_PLE_STAGE_SECONDS = [0.0]
_PLE_STAGE_CALLS = [0]


def ple_stage_seconds() -> float:
    """Cumulative host seconds spent in the PLE n-gram stage gather."""

    return float(_PLE_STAGE_SECONDS[0])


def ple_stage_calls() -> int:
    """Number of PLE n-gram stage gathers so far."""

    return int(_PLE_STAGE_CALLS[0])


def _ngram_rows_np(
    ids_np,
    prev_np,
    *,
    mult,
    sizes,
    offs,
    eos: int,
    ngram_size: int,
    heads_per_ngram: int,
):
    """Exact host row arithmetic shared by staged and fixed-M4 paths."""
    import numpy as np

    hist = np.concatenate([prev_np, ids_np], axis=1)

    def shift(h, s):
        if s == 0:
            return h
        b, ln = h.shape
        pos = np.arange(ln, dtype=np.int64)[None, :]
        eos_pos = np.where(h == eos, pos, np.int64(-1))
        prev_incl = np.maximum.accumulate(eos_pos, axis=1)
        prev = np.concatenate(
            [np.full((b, 1), -1, dtype=np.int64), prev_incl[:, :-1]], axis=1
        )
        pos_in_seg = pos - (prev + 1)
        src = np.maximum(pos - s, 0)
        shifted = np.take_along_axis(h, src, axis=1)
        valid = (pos_in_seg >= s) & (pos - s >= 0)
        return np.where(valid, shifted, np.int64(eos))

    shifted = [shift(hist, s) for s in range(ngram_size)]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * mult[0]
        for p in range(1, ngram):
            mixed = mixed ^ (shifted[p] * mult[p])
        blocks.append(mixed[..., None] % sizes[start:end] + offs[start:end])
    S = ids_np.shape[1]
    rows = np.concatenate(blocks, axis=-1)[:, -S:]
    return rows, hist[:, -(ngram_size - 1) :]


class NGramEmbedding(nn.Module):
    def __init__(self, args: TextArgs, ple_index: int):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        self.eos_id = args.eos_id
        head_dim = args.ple_embed_dim // self.ngram_heads

        sizes, offsets, total = _head_vocab_layout(
            args.ngram_vocab_size_base, self.ngram_heads, ple_index
        )
        div = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / div) * div
        # Checkpoint buffers overwrite these derived values on load.
        self.layer_multipliers = mx.array(
            _build_layer_multipliers(args.vocab_size, args.ngram_size, ple_index, args.seed),
            dtype=mx.int64,
        )
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self.ngram_embedding = NGramTable(
            padded, head_dim, sidecar=getattr(args, "ngram_sidecar", False)
        )

    def _shift_ignore_eos(self, ids: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return ids
        B, L = ids.shape
        pos = mx.arange(L, dtype=mx.int64)[None, :]
        eos_pos = mx.where(ids == self.eos_id, pos, mx.array(-1, dtype=mx.int64))
        prev_incl = mx.cummax(eos_pos, axis=1)
        prev = mx.concatenate(
            [mx.full((B, 1), -1, dtype=mx.int64), prev_incl[:, :-1]], axis=1
        )
        seg_start = prev + 1
        pos_in_seg = pos - seg_start
        src = pos - shift
        gather = mx.maximum(src, 0)
        shifted = mx.take_along_axis(ids, gather, axis=1)
        valid = (pos_in_seg >= shift) & (src >= 0)
        return mx.where(valid, shifted, mx.array(self.eos_id, dtype=mx.int64))

    # ---- staged fast path -------------------------------------------------
    # The row ids are a pure function of the token ids, so they can be
    # computed in numpy BEFORE the step's graph exists. The in-graph path
    # below forces a mid-forward GPU sync at layer 1 of every step
    # (np.asarray on a graph tensor + CPU gather while the pipeline
    # stalls); staging moves all of that to the top of Model.__call__,
    # where the previous step is already evaluated and the sync is free.
    # MTPLX_NGRAM_STAGE=0 disables; MTPLX_NGRAM_STAGE_VERIFY=1 also runs
    # the graph path and asserts equality (QA mode).

    def _np_consts(self):
        import numpy as np

        c = getattr(self, "_np_consts_cache", None)
        if c is None:
            c = (
                np.array(self.layer_multipliers, dtype=np.int64),
                np.array(self.ngram_heads_vocab_sizes, dtype=np.int64),
                np.array(self.ngram_heads_offsets, dtype=np.int64),
            )
            self._np_consts_cache = c
        return c

    def _rows_np(self, ids_np, prev_np):
        mult, sizes, offs = self._np_consts()
        return _ngram_rows_np(
            ids_np,
            prev_np,
            mult=mult,
            sizes=sizes,
            offs=offs,
            eos=self.eos_id,
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
        )

    def _take_prefill_lookahead(self, ids_np, flat):
        """Worker-prepared rows for this prefill chunk, then queue the next.

        Returns the raw row matrices a worker thread already gathered for
        exactly these ids, or None (every miss is counted, none is silent).
        The order matters: `take` frees the one slot BEFORE `submit` fills it
        with the following chunk, so the worker runs during this chunk's
        forward instead of after it.
        """

        import numpy as np

        from mtplx import ple_prefill_lookahead as lookahead_mod

        lookahead = lookahead_mod.active_lookahead()
        if lookahead is None:
            # No lookahead to consume: either the flag is off, or the prefill
            # is one chunk and the lane is inert by construction (nothing to
            # look ahead FROM).  The first-gather-early lane exists exactly
            # for that second case -- it is the whole win on a short prompt --
            # so it is consumed here rather than through the lookahead.
            return self._take_first_gather_early(ids_np, flat)
        index = lookahead.span_index_of(ids_np)
        if index is None:
            # Not a chunk of the planned prompt (an MTP verify width, a
            # restored suffix, a spliced prompt): take the ordinary path and
            # leave the pending slot alone.
            lookahead_mod.count("miss_unknown_span")
            return None
        payload = lookahead.take(index)
        lookahead.submit(lookahead.next_index(index))
        if payload is None:
            return None
        worker_flat, matrices = payload
        if not np.array_equal(worker_flat, flat):
            # The worker derives the chunk's PLE history from the plan; the
            # owner derives it from the live cache. They agree on a plain
            # chunked prefill and this proves it per chunk rather than
            # assuming it -- a prefix-cache restore that shifts the history
            # lands here and pays the ordinary gather, exactly.
            lookahead_mod.count("miss_row_mismatch")
            return None
        lookahead_mod.count("consumed_rows", int(flat.shape[0]))
        return matrices

    def _take_first_gather_early(self, ids_np, flat):
        """Worker-prepared rows for a chunk the lookahead never armed for."""

        import numpy as np

        from mtplx import ple_prefill_lookahead as lookahead_mod

        early = lookahead_mod.active_early_first_gather()
        if early is None:
            return None
        payload = early.take(ids_np)
        if payload is None:
            return None
        worker_flat, matrices = payload
        if not np.array_equal(worker_flat, flat):
            # Same proof the lookahead takes per chunk: the worker derived the
            # history from the plan, the owner from the live cache.  A restore
            # that shifts the history lands here and pays the ordinary gather,
            # exactly.
            lookahead_mod.count("early_row_mismatch")
            return None
        lookahead_mod.count("early_consumed_rows", int(flat.shape[0]))
        return matrices

    def _prefill_span_rows(self, plan_ids, start: int, end: int):
        """Sidecar row ids for one PLANNED prompt span -- worker thread.

        Pure NumPy: no MLX array is created or touched here.  The history is
        reconstructed from the plan (EOS-padded at the prompt head) exactly as
        `stage` reconstructs it from the PLE state cache.
        """

        import numpy as np

        ids = np.ascontiguousarray(plan_ids[start:end]).reshape(1, -1)
        context = self.context_len
        head = plan_ids[max(0, start - context) : start]
        if len(head) < context:
            head = np.concatenate(
                [
                    np.full(context - len(head), self.eos_id, dtype=np.int64),
                    head,
                ]
            )
        prev = np.ascontiguousarray(head).reshape(1, -1)
        rows, _ = self._rows_np(ids, prev)
        return np.ascontiguousarray(rows.reshape(-1))

    def _sidecar_map_names(self, sidecar):
        return ("weight",) if sidecar.bits == 0 else (
            "weight",
            "scales",
            "biases",
        )

    def prefill_lookahead_prepare(
        self,
        plan_ids,
        start: int,
        end: int,
        *,
        vectorized: bool | None = None,
        record=None,
    ):
        """Hash one PLANNED prompt span and gather its rows -- worker thread."""

        sidecar = self.ngram_embedding._sidecar
        if sidecar is None:
            return None
        flat = self._prefill_span_rows(plan_ids, start, end)
        prepared = sidecar.prepare_rows_np(
            flat,
            self._sidecar_map_names(sidecar),
            vectorized=vectorized,
            record=record,
        )
        if prepared is None:
            return None
        _unique, matrices = prepared
        return (flat, matrices)

    def first_gather_early_prepare(self, plan_ids, start: int, end: int, record):
        """The first chunk's gather, started at request arrival.

        Same expression, same maps, same ids as `prefill_lookahead_prepare`;
        it exists only to match the early lane's submit signature and to carry
        the receipt dict the worker fills in.
        """

        return self.prefill_lookahead_prepare(
            plan_ids, start, end, vectorized=True, record=record
        )

    def first_gather_prefetch_rest(self, plan_ids, start: int, record):
        """Page-warm the REST of the prompt's rows -- worker thread, no wait.

        Chained off the first chunk's future rather than merely submitted
        after it, so it can never delay the take whatever the pool's worker
        count.  It is the madvise(WILLNEED)
        equivalent for a hash-scattered row set: the rows are hashed span by
        span (bounded memory), unioned, and then either read in ascending order
        straight off the memmaps when they are already in core, or handed to
        the sidecar's own 16-thread pread pool when they are not -- the same
        pool, and the same GIL-releasing reads, that each chunk's gather would
        otherwise issue for itself one chunk at a time.
        """

        import numpy as np

        from mtplx.ple_row_gather import touch_rows, warm_decision

        sidecar = self.ngram_embedding._sidecar
        if sidecar is None:
            return 0
        total = int(np.asarray(plan_ids).reshape(-1).shape[0])
        width = max(1, int(start))
        pieces = []
        for begin in range(int(start), total, width):
            pieces.append(
                np.unique(
                    self._prefill_span_rows(plan_ids, begin, min(total, begin + width))
                )
            )
        if not pieces:
            return 0
        rows = np.unique(np.concatenate(pieces))
        names = self._sidecar_map_names(sidecar)
        maps = [sidecar._maps[name][0] for name in names]
        path, _fraction = warm_decision(maps, rows)
        if path == "vectorized":
            touched = touch_rows(maps, rows)
        elif sidecar._pool is not None:
            sidecar._submit_warm(rows, counted=False)
            touched = int(rows.shape[0])
        else:
            # Cold rows and MTPLX_NGRAM_PREFETCH=0: reading them here would be
            # serial demand faults with the GIL held, which is the generation
            # thread's problem, not this task's.  Leave them to the chunk that
            # needs them.
            path, touched = "skipped", 0
        if record is not None:
            record["prefetch_rest_rows"] = touched
            record["prefetch_rest_path"] = path
        return touched

    def stage(self, input_ids: mx.array, cache: Optional[ArraysCache], state_idx: int):
        """Precompute this step's rows before any graph is built."""
        started = time.perf_counter()
        try:
            self._stage_body(input_ids, cache, state_idx)
        finally:
            _PLE_STAGE_SECONDS[0] += time.perf_counter() - started
            _PLE_STAGE_CALLS[0] += 1

    def _stage_body(self, input_ids, cache, state_idx):
        import numpy as np

        sidecar = self.ngram_embedding._sidecar
        if sidecar is None or os.environ.get("MTPLX_NGRAM_STAGE", "1") == "0":
            return
        if getattr(self, "_stage_disabled", False):
            # Pipelined AR lane: input ids are LAZY — np.asarray below would
            # force a graph sync and collapse the pipeline. The lane gathers
            # in-graph via the table's mmap-lazy binding instead.
            return
        try:
            ids_np = np.asarray(input_ids, dtype=np.int64)
            B, S = ids_np.shape
            if cache is not None and cache[state_idx] is not None:
                prev_np = np.asarray(cache[state_idx], dtype=np.int64)
            else:
                prev_np = np.full((B, self.context_len), self.eos_id, dtype=np.int64)
            rows, new_hist = self._rows_np(ids_np, prev_np)
            flat = rows.reshape(-1)
            emb = sidecar.gather_np(
                flat, prepared=self._take_prefill_lookahead(ids_np, flat)
            )
            emb = emb.reshape(B, S, -1)
            self._staged = (B, S, emb, mx.array(new_hist), mx.array(prev_np))
        except Exception as exc:  # exact graph fallback stays available
            if not getattr(self, "_stage_warned", False):
                self._stage_warned = True
                print(f"[qwen4_exp] ngram staging disabled after error: {exc!r}",
                      flush=True)
            self._staged = None

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache], state_idx: int):
        compiled = _COMPILED_VERIFY_PLE.get()
        if compiled is not None:
            ids = input_ids.astype(mx.int64)
            B, _S = ids.shape
            if cache is not None and cache[state_idx] is not None:
                prev = cache[state_idx]
            else:
                prev = mx.full(
                    (B, self.context_len), self.eos_id, dtype=mx.int64
                )
            if cache is not None:
                cache[state_idx] = mx.concatenate([prev, ids], axis=1)[
                    :, -self.context_len :
                ]
            return compiled
        staged = getattr(self, "_staged", None)
        if staged is not None:
            self._staged = None
            sB, sS, emb, new_hist, prev = staged
            B, S = input_ids.shape
            if sB == B and sS == S:
                if os.environ.get("MTPLX_NGRAM_STAGE_VERIFY", "0") == "1":
                    ref = self._graph_path(input_ids, None, state_idx, prev=prev)
                    ok = bool(
                        mx.allclose(
                            ref.astype(mx.float32), emb.astype(mx.float32)
                        )
                    )
                    if not ok:
                        raise RuntimeError(
                            "ngram staged/graph mismatch — staging math broke"
                        )
                if cache is not None:
                    cache[state_idx] = new_hist
                self._stage_consumed = getattr(self, "_stage_consumed", 0) + 1
                self._stage_census()
                return emb
            # staged rows were computed for a different call shape — count it,
            # a silent fall-through here is exactly what an A/B cannot survive
            self._stage_bypassed = getattr(self, "_stage_bypassed", 0) + 1
        self._graph_calls = getattr(self, "_graph_calls", 0) + 1
        self._stage_census()
        return self._graph_path(input_ids, cache, state_idx)

    def _stage_census(self):
        if os.environ.get("MTPLX_NGRAM_STAGE_DEBUG", "0") != "1":
            return
        n = getattr(self, "_stage_consumed", 0) + getattr(self, "_graph_calls", 0)
        if n in (8, 64) or n % 512 == 0:
            sidecar = self.ngram_embedding._sidecar
            hot = (
                f" hot={sidecar.hot_hits}/{sidecar.hot_hits + sidecar.hot_misses}"
                if sidecar is not None
                else ""
            )
            print(
                f"[qwen4_exp] ngram path census: staged={getattr(self, '_stage_consumed', 0)} "
                f"graph={getattr(self, '_graph_calls', 0)} "
                f"stale-shape={getattr(self, '_stage_bypassed', 0)}{hot}",
                flush=True,
            )

    def _graph_path(self, input_ids, cache, state_idx, prev=None):
        ids = input_ids.astype(mx.int64)
        B, S = ids.shape
        if prev is None:
            if cache is not None and cache[state_idx] is not None:
                prev = cache[state_idx]
            else:
                prev = mx.full((B, self.context_len), self.eos_id, dtype=mx.int64)
        history = mx.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[state_idx] = history[:, -self.context_len :]

        shifted = [self._shift_ignore_eos(history, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for p in range(1, ngram):
                mixed = mx.bitwise_xor(mixed, shifted[p] * self.layer_multipliers[p])
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            head_ids = mx.remainder(mixed[..., None], sizes.reshape(1, 1, -1))
            blocks.append(head_ids + offsets.reshape(1, 1, -1))
        ngram_ids = mx.concatenate(blocks, axis=-1)[:, -S:]
        emb = self.ngram_embedding(ngram_ids)
        return emb.reshape(B, S, -1)


class PLELayer(nn.Module):
    """Per-Layer Embedding injection (runs on one linear-attention layer,
    before its hyper-connections). Cache slots: state_idx 2 = conv state,
    state_idx 3 = n-gram context ids."""

    CONV_IDX = 2
    NGRAM_IDX = 3

    def __init__(self, args: TextArgs, ple_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hidden_size * args.hc_count
        self.ple_embedding = NGramEmbedding(args, ple_index)
        self.conv_kernel_size = args.ple_conv_kernel_size
        self.conv_dilation = args.ngram_size
        self.conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(hc_hidden, args.hidden_size, eps=args.rms_norm_eps)
        self.norm_query = GroupedRMSNorm(hc_hidden, args.hidden_size, eps=args.rms_norm_eps)
        self.norm_conv = GroupedRMSNorm(hc_hidden, args.hidden_size, eps=args.rms_norm_eps)
        # Depthwise dilated conv, stored [channels, kernel, 1] (mlx layout).
        self.conv_weight = mx.zeros((hc_hidden, self.conv_kernel_size, 1))

    def _short_conv(self, x: mx.array, cache: Optional[ArraysCache]) -> mx.array:
        B, S, C = x.shape
        if cache is not None and cache[self.CONV_IDX] is not None:
            state = cache[self.CONV_IDX]
        else:
            state = mx.zeros((B, self.conv_state_len, C), dtype=x.dtype)
        window = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[self.CONV_IDX] = window[:, -self.conv_state_len :, :]
        out = mx.conv1d(
            window,
            self.conv_weight,
            stride=1,
            padding=0,
            dilation=self.conv_dilation,
            groups=C,
        )
        return nn.silu(out[:, -S:, :])

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache) -> mx.array:
        emb = self.ple_embedding(input_ids, cache, self.NGRAM_IDX)
        emb = emb.astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc_count, self.hidden_size)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc_count, self.hidden_size)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*hidden.shape)
        return gated + self._short_conv(self.norm_conv(gated), cache)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)
        if (layer_idx + 1) in args.ple_layer_ids:
            self.ple = PLELayer(args, args.ple_layer_ids.index(layer_idx + 1))
        self._hc = args.hc_count

    def __call__(self, hidden, *, input_ids, ssm_mask, cache):
        if "ple" in self:
            hidden = hidden + self.ple(hidden, input_ids, cache)

        mixed, hyper, inject = self.attn_hyper_connection(hidden)
        if self.is_linear:
            block_out = self.linear_attn(mixed, ssm_mask, cache)
        else:
            block_out = self.self_attn(mixed, cache)
        hidden = _hyper_residual_write(hyper, block_out, inject)

        mixed, hyper, inject = self.mlp_hyper_connection(hidden)
        block_out = self.mlp(mixed)
        hidden = _hyper_residual_write(hyper, block_out, inject)
        return hidden


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = (
            args.layer_types.index("linear_attention")
            if "linear_attention" in args.layer_types
            else 0
        )
        self._ple_stage_idx = next(
            (i for i, l in enumerate(self.layers) if getattr(l, "ple", None) is not None),
            None,
        )
        self.fa_idx = next(
            (i for i, t in enumerate(args.layer_types) if t != "linear_attention"),
            self.ssm_idx,
        )
        # Dual-alias env for the compiled GDN decode lane (PR #395 by maceip):
        # MTPLX_QWEN4EXP_COMPILE is the family-named alias of
        # MTPLX_COMPILED_GDN. An explicit falsy value on EITHER name is a kill
        # switch that wins over everything, including set_ar_pipeline_mode
        # re-arming the lane (upstream bug: the lane ignored an operator's
        # explicit 0 once the AR pipeline engaged).
        gdn_env = os.environ.get("MTPLX_COMPILED_GDN")
        qwen4_env = os.environ.get("MTPLX_QWEN4EXP_COMPILE")
        truthy_env = {"1", "true", "yes", "on"}
        gdn_disabled = gdn_env is not None and gdn_env.strip().lower() not in truthy_env
        qwen4_disabled = (
            qwen4_env is not None and qwen4_env.strip().lower() not in truthy_env
        )
        self._gdn_compile_explicit_off = gdn_disabled or qwen4_disabled
        if self._gdn_compile_explicit_off:
            self._gdn_compiled_env = False
        else:
            self._gdn_compiled_env = (
                gdn_env is not None and gdn_env.strip().lower() in truthy_env
            ) or (qwen4_env is not None and qwen4_env.strip().lower() in truthy_env)
        self._gdn_compiled_lane = False
        self._decode_runs = None
        self._decode_run_fns = {}

    def ple_prefill_lookahead(self, token_ids, spans):
        """Request-scoped PLE n-gram lookahead for a chunked prefill.

        Returns a ``PrefillLookahead`` when MTPLX_QWEN4_PLE_PREFILL_LOOKAHEAD
        is armed and this model can serve it, otherwise None.  Construction
        time is the only eligibility decision; the flag is read once.

        Arming it on a model whose PLE table never attached, or with staging
        turned off, raises: those are the two ways the lane would quietly
        become the control while wearing the candidate's label.
        """

        from mtplx import ple_prefill_lookahead as lookahead_mod

        if not lookahead_mod.enabled():
            return None
        if self._ple_stage_idx is None:
            lookahead_mod.count("no_ple_stage")
            return None
        embedding = self.layers[self._ple_stage_idx].ple.ple_embedding
        if embedding.ngram_embedding._sidecar is None:
            raise RuntimeError(
                f"{lookahead_mod.ENV_FLAG}=1 needs the SSD-resident n-gram "
                "sidecar, which never attached for this model"
            )
        if os.environ.get("MTPLX_NGRAM_STAGE", "1") == "0":
            raise RuntimeError(
                f"{lookahead_mod.ENV_FLAG}=1 prepares rows for the STAGED "
                "gather, but MTPLX_NGRAM_STAGE=0 routes them in-graph"
            )
        if getattr(embedding, "_stage_disabled", False):
            raise RuntimeError(
                f"{lookahead_mod.ENV_FLAG}=1 is incompatible with the "
                "pipelined AR lane, whose input ids are lazy"
            )
        # Build the shared NumPy hash constants on THIS thread so the worker
        # never races the lazy cache in `_np_consts`.
        embedding._np_consts()
        sidecar = embedding.ngram_embedding._sidecar
        vectorized = lookahead_mod.early_enabled()
        lookahead = lookahead_mod.PrefillLookahead(
            token_ids,
            spans,
            prepare=lambda start, end: embedding.prefill_lookahead_prepare(
                lookahead.token_ids, start, end, vectorized=vectorized
            ),
            # Which spans the worker is designed to serve, stated in the
            # sidecar's own terms rather than restated in the lane: one span
            # hashes to `tokens * ngram_heads` rows, and `prepare_rows_np`
            # declines everything at or below `_HOT_PATH_MAX_ROWS` because
            # that is the owner-thread-only hot-row LRU. With the LRU off
            # (`MTPLX_NGRAM_HOT_MB=0`) it declines nothing, so nothing is
            # exempt. Without this the lane read its own by-design declines
            # as non-engagement and 500ed every short prompt.
            rows_per_token=int(embedding.ngram_heads),
            min_servable_rows=(
                int(sidecar._HOT_PATH_MAX_ROWS) if sidecar._hot_cap_rows else 0
            ),
        )
        return lookahead

    def ple_first_gather_early(self, token_ids, span):
        """Start the FIRST prefill chunk's PLE gather now -- request arrival.

        Returns an ``EarlyFirstGather`` when MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY
        is armed and this model can serve it, otherwise None.  The eligibility
        rules are the lookahead's, for the same reasons: an armed flag on a
        model whose sidecar never attached, or with staging routed in-graph,
        would quietly measure the control while wearing the candidate's label.

        ``span`` is the caller's prediction of the prefill's first chunk.  It
        is never trusted: the payload is accepted only after its span's token
        ids compare equal to the ids `stage` was actually called with.
        """

        from mtplx import ple_prefill_lookahead as lookahead_mod

        if not lookahead_mod.early_enabled():
            return None
        if self._ple_stage_idx is None:
            lookahead_mod.count("early_no_ple_stage")
            return None
        embedding = self.layers[self._ple_stage_idx].ple.ple_embedding
        sidecar = embedding.ngram_embedding._sidecar
        if sidecar is None:
            raise RuntimeError(
                f"{lookahead_mod.EARLY_ENV_FLAG}=1 needs the SSD-resident "
                "n-gram sidecar, which never attached for this model"
            )
        if os.environ.get("MTPLX_NGRAM_STAGE", "1") == "0":
            raise RuntimeError(
                f"{lookahead_mod.EARLY_ENV_FLAG}=1 prepares rows for the "
                "STAGED gather, but MTPLX_NGRAM_STAGE=0 routes them in-graph"
            )
        if getattr(embedding, "_stage_disabled", False):
            raise RuntimeError(
                f"{lookahead_mod.EARLY_ENV_FLAG}=1 is incompatible with the "
                "pipelined AR lane, whose input ids are lazy"
            )
        start, end = int(span[0]), int(span[1])
        # The sidecar's own servability rule (aa20bf11), restated nowhere: a
        # span at or below the hot-row threshold belongs to the owner-thread
        # LRU, which the worker is forbidden to touch, so starting a worker
        # for it would buy a decline.
        min_rows = int(sidecar._HOT_PATH_MAX_ROWS) if sidecar._hot_cap_rows else 0
        if (end - start) * int(embedding.ngram_heads) <= min_rows:
            lookahead_mod.count("early_span_not_servable")
            return None
        # Build the shared NumPy hash constants on THIS thread so the worker
        # never races the lazy cache in `_np_consts`.
        embedding._np_consts()
        return lookahead_mod.EarlyFirstGather(
            token_ids,
            (start, end),
            prepare=lambda ids, a, b, record: embedding.first_gather_early_prepare(
                ids, a, b, record
            ),
            prefetch_rest=lambda ids, a, record: embedding.first_gather_prefetch_rest(
                ids, a, record
            ),
        )

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if qwen4_opdiet_enabled("rope"):
            with _rope_table_scope():
                return self._forward(inputs, cache, input_embeddings)
        return self._forward(inputs, cache, input_embeddings)

    def _forward(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        if self._ple_stage_idx is not None and _COMPILED_VERIFY_PLE.get() is None:
            ple = self.layers[self._ple_stage_idx].ple
            ple.ple_embedding.stage(inputs, cache[self._ple_stage_idx], ple.NGRAM_IDX)
        h = mx.tile(h, (1, 1, self.args.hc_count))
        if (
            # S<=4 covers AR decode (S=1) and MTP verify widths (S=2..4,
            # depth ceiling 3): mx.compile keys its trace cache on input
            # shapes, so each S gets one retrace then C++ replay, and the
            # GDN states are S-invariant so the same run fns serve all
            # widths. Prefill and masked/padded forwards stay eager.
            1 <= h.shape[1] <= 4
            and ssm_mask is None
            and not self._gdn_compile_explicit_off
            and (self._gdn_compiled_env or self._gdn_compiled_lane)
            and cache[self.ssm_idx] is not None
        ):
            h = self._decode_layers_compiled(h, inputs, cache)
        else:
            capture = _VERIFY_CAPTURE.get()
            for layer, c in zip(self.layers, cache):
                if (
                    capture
                    and c is not None
                    and getattr(layer, "ple", None) is not None
                ):
                    c._mtplx_verify_ple = (h, inputs)
                h = layer(h, input_ids=inputs, ssm_mask=ssm_mask, cache=c)
        # The MTP head consumes the pre-mixer widened stream; keep the last
        # one reachable (lazy ref, freed on the next step).
        self._last_widened = h
        return self.hyper_connection_mixer(h)

    # ---- compiled GDN decode runs ----------------------------------------
    # The qL=1 decode step is CPU-dispatch-bound: ~20.8ms of Python graph
    # construction per token against <=14ms of GPU work (measured 2026-08-27,
    # ar-lane census: build=20.79ms wait=0.00ms). GDN layers have FIXED state
    # shapes at decode (conv tape + SSM state), so contiguous non-PLE GDN
    # runs compile once and replay in C++. QSA layers grow their caches every
    # step (KV slab + raw-key concat) and the PLE layer consumes token ids —
    # both stay eager until the slab/graphbank arc.

    def _build_decode_runs(self):
        runs = []
        cur = []
        for i, layer in enumerate(self.layers):
            if layer.is_linear and "ple" not in layer:
                cur.append(i)
            else:
                if cur:
                    runs.append(("run", tuple(cur)))
                    cur = []
                runs.append(("eager", i))
        if cur:
            runs.append(("run", tuple(cur)))
        return runs

    def _compiled_run_fn(self, idxs, capture: bool = False):
        layers = [self.layers[i] for i in idxs]

        def step(h, *flat):
            out_states = []
            rows = []
            k = 0
            for layer in layers:
                c = ArraysCache(size=2)
                c[0], c[1] = flat[k], flat[k + 1]
                k += 2
                h = layer(h, input_ids=None, ssm_mask=None, cache=c)
                out_states.extend((c[0], c[1]))
                if capture:
                    # __call__ ran under the capture scope during THIS trace,
                    # so the temp cache carries the tracer rows — surface
                    # them as compiled outputs.
                    rows.extend(c._mtplx_verify_rows)
            return (h, *out_states, *rows)

        return mx.compile(step)

    def _get_run_fn(self, idxs, capture: bool):
        key = (idxs, bool(capture))
        fn = self._decode_run_fns.get(key)
        if fn is None:
            fn = self._compiled_run_fn(idxs, capture=capture)
            self._decode_run_fns[key] = fn
        return fn

    def _decode_layers_compiled(self, h, inputs, cache):
        if self._decode_runs is None:
            self._decode_runs = self._build_decode_runs()
        capture = _VERIFY_CAPTURE.get()
        for kind, payload in self._decode_runs:
            if kind == "eager":
                i = payload
                if capture and getattr(self.layers[i], "ple", None) is not None:
                    cache[i]._mtplx_verify_ple = (h, inputs)
                h = self.layers[i](
                    h, input_ids=inputs, ssm_mask=None, cache=cache[i]
                )
                continue
            idxs = payload
            flat = []
            usable = True
            for i in idxs:
                s0, s1 = cache[i][0], cache[i][1]
                if s0 is None or s1 is None:
                    usable = False
                    break
                flat.extend((s0, s1))
            if not usable:
                for i in idxs:
                    h = self.layers[i](
                        h, input_ids=inputs, ssm_mask=None, cache=cache[i]
                    )
                continue
            out = self._get_run_fn(idxs, capture)(h, *flat)
            h = out[0]
            k = 1
            for i in idxs:
                cache[i][0] = out[k]
                cache[i][1] = out[k + 1]
                k += 2
            if capture:
                for i in idxs:
                    cache[i]._mtplx_verify_rows = tuple(out[k : k + 6])
                    k += 6
        return h

    def clear_verify_capture(self, cache) -> None:
        for entry in cache:
            if entry is None:
                continue
            for attr in ("_mtplx_verify_rows", "_mtplx_verify_ple"):
                if getattr(entry, attr, None) is not None:
                    setattr(entry, attr, None)

    def _refuse_commit(self, layer_index: int, reason: str) -> bool:
        """A refused capture-commit silently falls back to the rollback +
        trunk re-forward. Silence hid a whole battery of fallbacks on
        2026-08-27 — print the first few reasons per process."""
        count = getattr(self, "_commit_refusals", 0)
        self._commit_refusals = count + 1
        if count < 3:
            print(
                f"[qwen4_exp] capture-commit refused (layer {layer_index}: "
                f"{reason}) — falling back to re-forward",
                flush=True,
            )
        return False

    def commit_verified_window(
        self,
        cache,
        snapshot_states,
        *,
        keep_tokens: int,
        verified_tokens: int,
    ) -> bool:
        """Repair-free commit of a speculative verify window.

        Trimmable entries (QSA attention) trim their uncommitted tail; each
        pure-GDN layer replays ONLY its gated-delta recurrence over the kept
        rows from the pre-verify snapshot state; the single PLE-carrying
        layer replays its full (cheap, <=window-rows) layer forward from its
        snapshot slots. Everything is lazy — no eval here; the next round's
        eval pulls the replay. Validates every entry before mutating any so
        a refusal leaves the cache intact for the rollback+re-forward
        fallback. Returns True when the commit landed.
        """
        from mlx_lm.models.gated_delta import gated_delta_update

        keep_tokens = int(keep_tokens)
        verified_tokens = int(verified_tokens)
        trim_n = verified_tokens - keep_tokens
        if keep_tokens < 1 or trim_n < 0 or len(cache) != len(self.layers):
            return False
        if trim_n == 0:
            # Full accept: the verify forward already left every entry in
            # the post-window state (the normal round relies on exactly this
            # and never commits a full accept). Replaying the 36 GDN
            # recurrences and the PLE layer over the same rows would only
            # recompute what is there and land in the next round's eval,
            # which is the freeze after every fully accepted copy block
            # (2026-09-02 receipt: 138 of 181 blocks fully accepted on a
            # 3.5k re-emission turn). Drop the stashed rows and return.
            for entry in cache:
                if entry is None:
                    continue
                if getattr(entry, "_mtplx_verify_rows", None) is not None:
                    entry._mtplx_verify_rows = None
                if getattr(entry, "_mtplx_verify_ple", None) is not None:
                    entry._mtplx_verify_ple = None
            return True

        plan = []
        for i, (layer, entry) in enumerate(zip(self.layers, cache)):
            if entry is None:
                return self._refuse_commit(i, "entry_missing")
            if callable(getattr(entry, "is_trimmable", None)) and entry.is_trimmable():
                plan.append(("trim", i, None))
                continue
            pre = snapshot_states[i] if snapshot_states is not None else None
            if pre is None:
                return self._refuse_commit(i, "snapshot_missing")
            if getattr(layer, "ple", None) is not None:
                cap = getattr(entry, "_mtplx_verify_ple", None)
                if cap is None:
                    return self._refuse_commit(i, "ple_rows_missing")
                if cap[0].shape[1] != verified_tokens:
                    return self._refuse_commit(
                        i, f"ple_rows_width_{cap[0].shape[1]}_vs_{verified_tokens}"
                    )
                if len(pre) < 4:
                    return self._refuse_commit(i, "ple_snapshot_short")
                plan.append(("ple", i, cap))
                continue
            rows = getattr(entry, "_mtplx_verify_rows", None)
            if rows is None:
                return self._refuse_commit(i, "gdn_rows_missing")
            if rows[0].shape[1] != verified_tokens:
                return self._refuse_commit(
                    i, f"gdn_rows_width_{rows[0].shape[1]}_vs_{verified_tokens}"
                )
            if len(pre) < 2 or pre[1] is None:
                return self._refuse_commit(i, "gdn_snapshot_short")
            plan.append(("gdn", i, rows))

        for kind, i, payload in plan:
            entry = cache[i]
            layer = self.layers[i]
            if kind == "trim":
                if trim_n:
                    entry.trim(trim_n)
                continue
            pre = snapshot_states[i]
            if kind == "gdn":
                qkv, q, k, v, a, b = payload
                gdn = layer.linear_attn
                conv_pre = pre[0]
                if conv_pre is None:
                    conv_pre = mx.zeros(
                        (qkv.shape[0], gdn.conv_kernel_size - 1, qkv.shape[2]),
                        dtype=qkv.dtype,
                    )
                _, new_state = gated_delta_update(
                    q[:, :keep_tokens],
                    k[:, :keep_tokens],
                    v[:, :keep_tokens],
                    a[:, :keep_tokens],
                    b[:, :keep_tokens],
                    gdn.A_log,
                    gdn.dt_bias,
                    pre[1],
                    None,
                    use_kernel=not gdn.training,
                )
                conv_input = mx.concatenate(
                    [conv_pre, qkv[:, :keep_tokens]], axis=1
                )
                entry[0] = mx.contiguous(
                    conv_input[:, -(gdn.conv_kernel_size - 1) :, :]
                )
                entry[1] = new_state
                entry._mtplx_verify_rows = None
            else:  # ple
                h_in, ids = payload
                for j in range(len(pre)):
                    entry[j] = pre[j]
                ple = layer.ple
                kept_ids = ids[:, :keep_tokens]
                ple.ple_embedding.stage(kept_ids, entry, ple.NGRAM_IDX)
                layer(
                    h_in[:, :keep_tokens],
                    input_ids=kept_ids,
                    ssm_mask=None,
                    cache=entry,
                )
                entry._mtplx_verify_ple = None
        return True


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model = Qwen4ExpTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        object.__setattr__(self, "_mtp_draft_head_logits", self._head_logits)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int = 0,
    ):
        # hidden_variant is accepted for the runtime contract's sake but this
        # family has exactly one draft input: the pre-mixer WIDENED stream.
        # emit_logits/logits_keep are the sustained-prefill contract: a
        # cache-only chunk skips the [1, S, 248320] head matmul entirely
        # (~1.02 GB per 2048-token chunk that used to be built and thrown
        # away 128 times per 262K cold prefill — the #393 audit receipt).
        out = self.model(inputs, cache, input_embeddings)
        if not emit_logits:
            return (None, self.model._last_widened) if return_hidden else None
        if logits_keep:
            out = out[:, -max(1, int(logits_keep)) :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        if return_hidden:
            return logits, self.model._last_widened
        return logits

    def _head_logits(self, h):
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def _mtplx_bind_draft_lm_head(self, head) -> None:
        """Bind the native MTP proposal head once at construction."""

        object.__setattr__(self, "_mtp_draft_head_logits", head.__call__)

    def _mtplx_native_mtp_draft_head(self):
        """Return the unmodified head used by the native MTP route."""

        if self.args.tie_word_embeddings:
            return None
        return self.lm_head

    # ---- runtime draft surface (validate_mtp_support shape) ---------------

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
    ):
        """Draft logits from the trunk's widened stream + next token ids.

        ``hidden_states`` is the pre-mixer widened stream [B,S,hc*d] on the
        first depth and this head's own pre-mixer output on deeper recursion
        steps (returned via ``return_hidden``). concat_order / hidden_variant
        have no meaning here (no concat, single variant); positions come from
        the QSA cache offset (contract mtp_position_mode="cache").
        """
        emb = self.model.embed_tokens(next_token_ids)
        h = self.mtp.fuse_and_run(hidden_states, emb, mtp_cache)
        logits = self._mtp_draft_head_logits(self.mtp.hyper_connection_mixer(h))
        if return_hidden:
            return logits, h
        return logits

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
        input_embeddings=None,
    ):
        """Append committed history to the head's cache (no lm_head cost).

        ``input_embeddings`` (vision splice) supplies exact embedding rows in
        place of ``embed_tokens(next_token_ids)`` when provided.
        """
        if input_embeddings is not None:
            emb = input_embeddings
        else:
            emb = self.model.embed_tokens(next_token_ids)
        return self.mtp.fuse_and_run_history(hidden_states, emb, mtp_cache)

    def make_mtp_cache(self):
        return [QSACache(self.model.args.indexer_compress_ratio or 4)]

    def make_cache(self):
        ratio = self.model.args.indexer_compress_ratio or 4
        caches = []
        for i, layer in enumerate(self.model.layers):
            if not layer.is_linear:
                caches.append(QSACache(ratio))
            elif "ple" in layer:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(ArraysCache(size=2))
        return caches


class Qwen4ExpMTP(nn.Module):
    """Flash-Next MTP head, reconstructed from the shipped tensors (no public
    reference implements it — transformers ships only the trunk).

    Wiring (the only reading consistent with every tensor shape): the trunk's
    pre-mixer WIDENED stream [B,S,hc*d] is RMS-normed at full width
    (pre_fc_norm_hidden is [hc*d]); each 2560-wide substream goes through the
    SHARED fc_hidden [d,d]; the normed+projected token embedding
    (pre_fc_norm_embedding -> fc_embedding, both [d]-sized) is broadcast-added
    into every substream; the fused widened stream runs ONE full-attention
    DecoderLayer (QSA + MoE + hyper-connections, its own tensors) and this
    head's own mixer collapses back to d for the SHARED trunk lm_head.

    Correctness is graded by measured acceptance — the probability-ratio
    verify contract keeps outputs exact for ANY draft head, so a mis-wiring
    can only cost speed, never quality.
    """

    def __init__(self, args: TextArgs):
        super().__init__()
        d = args.hidden_size
        self.pre_fc_norm_embedding = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(d * args.hc_count, eps=args.rms_norm_eps)
        self.fc_embedding = nn.Linear(d, d, bias=False)
        self.fc_hidden = nn.Linear(d, d, bias=False)
        fa_idx = next(
            i for i, t in enumerate(args.layer_types)
            if t != "linear_attention" and (i + 1) not in args.ple_layer_ids
        )
        self.layers = [DecoderLayer(args, fa_idx)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self._hc = args.hc_count
        object.__setattr__(self, "_mtp_prepare_inputs_impl", self._prepare_inputs_eager)

    def _prepare_inputs_eager(
        self,
        widened: mx.array,
        tok_emb: mx.array,
    ) -> mx.array:
        B, S, W = widened.shape
        hn = self.pre_fc_norm_hidden(widened).reshape(B, S, self._hc, -1)
        en = self.fc_embedding(self.pre_fc_norm_embedding(tok_emb))
        return (self.fc_hidden(hn) + en[:, :, None, :]).reshape(B, S, W)

    def install_compiled_prepare(self) -> dict[str, Any]:
        """Install the exact fixed-B1/S1 stateless draft preparation graph."""

        width = int(self.pre_fc_norm_hidden.weight.shape[0])
        embedding_width = int(self.pre_fc_norm_embedding.weight.shape[0])
        dtype = self.pre_fc_norm_hidden.weight.dtype
        widened = (
            (mx.arange(width, dtype=mx.float32) % 257) * (1.0 / 257.0)
        ).reshape(1, 1, width).astype(dtype)
        tok_emb = (
            (mx.arange(embedding_width, dtype=mx.float32) % 127) * (1.0 / 127.0)
        ).reshape(1, 1, embedding_width).astype(dtype)

        expected = self._prepare_inputs_eager(widened, tok_emb)
        mx.eval(expected)
        compiled = mx.compile(self._prepare_inputs_eager)
        actual = compiled(widened, tok_emb)
        mx.eval(actual)
        if not bool(mx.array_equal(actual, expected).item()):
            raise RuntimeError(
                "compiled Qwen4 MTP preparation failed exact construction parity"
            )
        object.__setattr__(self, "_mtp_prepare_inputs_impl", compiled)
        return {
            "installed": True,
            "shape": [1, 1, width],
            "dtype": str(dtype),
        }

    def fuse_and_run(self, widened: mx.array, tok_emb: mx.array, cache) -> mx.array:
        """Fuse (widened, token embedding) and run the head's layer; returns
        the PRE-mixer widened output — the recursion state for deeper drafts."""
        h = self._mtp_prepare_inputs_impl(widened, tok_emb)
        # cache is the make_mtp_cache() list (runtime convention); the single
        # layer consumes its own QSACache entry
        layer_cache = cache[0] if cache is not None else None
        return self.layers[0](h, input_ids=None, ssm_mask=None, cache=layer_cache)

    def fuse_and_run_history(
        self,
        widened: mx.array,
        tok_emb: mx.array,
        cache,
    ) -> mx.array:
        """History/prefill phase route; it may carry S>1 and stays eager."""

        h = self._prepare_inputs_eager(widened, tok_emb)
        layer_cache = cache[0] if cache is not None else None
        return self.layers[0](h, input_ids=None, ssm_mask=None, cache=layer_cache)

    def __call__(self, widened: mx.array, tok_emb: mx.array, cache) -> mx.array:
        return self.hyper_connection_mixer(self.fuse_and_run(widened, tok_emb, cache))


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params.get("model_type", "qwen4_exp"), text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    """House-shaped wrapper: language_model.{model,lm_head}.

    Vision serving: the tower (model-vision.safetensors) is constructed
    out-of-band by mtplx.vision.load_vision_tower — lazily, on the first
    image, digest-LRU-cached — never here, so text-only sessions pay
    nothing for it. Image rows enter through ``input_embeddings`` and the
    attention layers read the request's M-RoPE table via
    ``vision_rope_state()`` (2026-08-29); ``sanitize`` therefore still
    filters ``vision_tower.*`` keys from the LM load."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_config = dict(args.text_config)
        # NOTE: no top-level->text_config merge happens here; the shipped
        # packs carry eos_token_id inside text_config, which is what
        # TextArgs reads.
        self.language_model = TextModel(TextArgs.from_dict(text_config))

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int = 0,
    ):
        # Explicit params only (no **kwargs): the runtime capability probe
        # reads this signature; emit_logits/logits_keep are now genuinely
        # implemented (cache-only prefill chunks skip the vocab head).
        return self.language_model(
            inputs,
            cache,
            input_embeddings,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            emit_logits=emit_logits,
            logits_keep=logits_keep,
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    # ---- MTP attach -------------------------------------------------------

    def attach_mtp(self, model_path) -> bool:
        """Build + load the MTP head from mtp.safetensors. Per-module quant
        recipes are inferred from the packed tensor shapes (bits from the
        u32 column count vs the module's in_features, group from the scales
        columns) — the sidecar is self-describing."""
        path = Path(model_path) / "mtp.safetensors"
        if not path.exists():
            return False
        raw = mx.load(str(path))
        args = self.language_model.args
        mtp = Qwen4ExpMTP(args)

        flat = dict(raw)
        stripped = {}
        for name, v in flat.items():
            if not name.startswith("mtp."):
                continue
            stripped[name[len("mtp."):]] = v
        # The converter's +1 norm shift covers the trunk-suffix norms only;
        # the head's pre_fc norms ship raw zero-centered ((1+w) convention).
        for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
            if n in stripped and stripped[n].ndim == 1:
                stripped[n] = (stripped[n].astype(mx.float32) + 1.0).astype(
                    stripped[n].dtype
                )

        from mlx.utils import tree_flatten as _tf

        module_in = {}
        for pth, arr in _tf(mtp.parameters()):
            if pth.endswith(".weight") and arr.ndim >= 2:
                module_in[pth[: -len(".weight")]] = int(arr.shape[-1])

        qmap = {}
        for mod, in_f in module_in.items():
            w = stripped.get(f"{mod}.weight")
            s = stripped.get(f"{mod}.scales")
            if w is None or s is None or w.dtype != mx.uint32:
                continue
            bits = int(w.shape[-1]) * 32 // in_f
            group = in_f // int(s.shape[-1])
            qmap[mod] = {"bits": bits, "group_size": group}

        def predicate(pth, module):
            cfg = qmap.get(pth)
            return cfg if cfg else False

        nn.quantize(mtp, group_size=64, bits=8, class_predicate=predicate)
        mtp.load_weights(list(stripped.items()), strict=True)
        mtp.eval()
        # publish on the text model — mtp_patch._text_model() resolves
        # language_model, and registering the module on BOTH trees would
        # double-count its parameters
        self.language_model.mtp = mtp
        return True

    @property
    def mtp(self):
        return getattr(self.language_model, "mtp", None)

    # The runtime drives the draft surface on the wrapper (self.model.*)
    # while validate_mtp_support reads it off language_model — both resolve
    # to the same TextModel implementation.
    def mtp_forward(self, hidden_states, next_token_ids, **kwargs):
        return self.language_model.mtp_forward(hidden_states, next_token_ids, **kwargs)

    def mtp_update_cache(self, hidden_states, next_token_ids, **kwargs):
        return self.language_model.mtp_update_cache(hidden_states, next_token_ids, **kwargs)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_draft_logits(self, widened: mx.array, tok_emb: mx.array, cache) -> mx.array:
        """Draft logits from the trunk's widened stream + the next token's
        embedding, through the shared trunk lm_head."""
        lm = self.language_model
        return lm._head_logits(lm.mtp(widened, tok_emb, cache))

    def post_weight_load(self, model_path) -> None:
        """Attach the SSD-resident n-gram sidecar after weights load.
        (Gate+up fusion happens earlier, at sanitize time — see
        _fuse_gate_up_sanitize.)"""
        path = Path(model_path) / "ngram-table.safetensors"
        if not path.exists():
            return
        resident = _ngram_resident_policy()
        sidecar = None
        for layer in self.layers:
            if "ple" in layer:
                table = layer.ple.ple_embedding.ngram_embedding
                table.attach_sidecar(path)
                sidecar = table._sidecar
                if resident:
                    table.attach_resident(path)
        if sidecar is not None and not resident:
            hot_mb = (sidecar._hot_cap_rows * sidecar._hot_row_bytes) // 2**20
            print(
                "[qwen4_exp] ngram table: streamed from SSD (file-backed "
                f"pages, hot cache {hot_mb}M, prefetch "
                f"{'on' if sidecar._pool is not None else 'off'}) — "
                "resident only at >=160G RAM or MTPLX_NGRAM_RESIDENT=1",
                flush=True,
            )

    def set_ar_pipeline_mode(self, enabled: bool) -> bool:
        """Flip the family into (or out of) the pipelined-AR decode contract:
        n-gram staging off + in-graph mmap-lazy gathers, so a forward built
        on LAZY token ids records no host sync. Returns False when the lazy
        table binding is unavailable (lane must not engage)."""
        ready = True
        for layer in self.layers:
            if "ple" not in layer:
                continue
            emb = layer.ple.ple_embedding
            table = emb.ngram_embedding
            if enabled and getattr(table, "_lazy_parts", None) is None:
                ready = False
                continue
            emb._stage_disabled = bool(enabled)
            table.prefer_lazy = bool(enabled)
        if ready:
            model = self.language_model.model
            if not getattr(model, "_gdn_compile_explicit_off", False):
                model._gdn_compiled_lane = bool(enabled)
            else:
                model._gdn_compiled_lane = False
        return ready

    # -- family capture-commit (repair-free verify rollback) ----------------

    def verify_capture_scope(self):
        return verify_capture_scope()

    def clear_verify_capture(self, cache) -> None:
        self.language_model.model.clear_verify_capture(cache)

    def commit_verified_window(
        self, cache, snapshot_states, *, keep_tokens: int, verified_tokens: int
    ) -> bool:
        return self.language_model.model.commit_verified_window(
            cache,
            snapshot_states,
            keep_tokens=keep_tokens,
            verified_tokens=verified_tokens,
        )

    # -- weight plumbing ---------------------------------------------------

    _HF_NORM_SHIFT_SUFFIXES = (
        ".q_norm.weight",
        ".k_norm.weight",
        ".q_layernorm.weight",
        ".k_layernorm.weight",
        ".hc_norm.weight",
        ".norm_key.weight",
        ".norm_query.weight",
        ".norm_conv.weight",
    )

    def sanitize(self, weights):
        # Raw-HF discriminator: unconverted checkpoints carry HF conv layout
        # ([ch, 1, k]) and the model.language_model prefix.
        raw = any(
            k.endswith("conv1d.weight") and v.shape[-1] != 1 for k, v in weights.items()
        ) or any(k.startswith("model.language_model.") for k in weights)

        out = {}
        stacked: dict[str, dict[int, mx.array]] = {}
        for k, v in weights.items():
            if k.startswith("model.visual.") or k.startswith("vision_tower."):
                continue
            if k.startswith("mtp."):
                continue
            if k.startswith("model.language_model."):
                k = k.replace("model.language_model.", "language_model.model.", 1)
            elif k == "lm_head.weight":
                k = "language_model.lm_head.weight"
            elif not k.startswith("language_model."):
                k = "language_model." + k

            if raw:
                # numbered per-expert tensors -> stacked switch_mlp
                if ".mlp.experts." in k and ".weight" in k and "scale_inv" not in k:
                    prefix, rest = k.split(".mlp.experts.", 1)
                    idx_s, proj_rest = rest.split(".", 1)
                    proj = proj_rest.rsplit(".weight", 1)[0]
                    dest = f"{prefix}.mlp.switch_mlp.{proj}.weight"
                    stacked.setdefault(dest, {})[int(idx_s)] = v
                    continue
                if ".mlp.experts.gate_up_proj" in k or ".mlp.experts.down_proj" in k:
                    # Two packed layouts exist. transformers save_pretrained
                    # writes the runtime bmm orientation (gate_up [E, hidden,
                    # 2*inter], down [E, inter, hidden]); the hub bf16 repo
                    # ships Linear [out, in] halves (gate_up [E, 2*inter,
                    # hidden], down [E, hidden, inter]). Keyed on which axis
                    # equals hidden; square (test-config) tensors resolve to
                    # the transformers branch, which parity validated.
                    prefix = k.split(".mlp.experts.", 1)[0]
                    hid = self.language_model.args.hidden_size
                    if k.endswith("gate_up_proj"):
                        if v.shape[1] == hid:
                            gate, up = mx.split(v, 2, axis=-1)
                            gate = gate.swapaxes(1, 2)
                            up = up.swapaxes(1, 2)
                        else:
                            gate, up = mx.split(v, 2, axis=1)
                        out[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
                        out[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
                    else:
                        if v.shape[2] == hid:
                            v = v.swapaxes(1, 2)
                        out[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = v
                    continue
                if k.endswith("ple.conv1d.weight"):
                    out[k.replace("ple.conv1d.weight", "ple.conv_weight")] = v.moveaxis(2, 1)
                    continue
                if k.endswith("linear_attn.conv1d.weight") and v.shape[-1] != 1:
                    v = v.moveaxis(2, 1)
                if v.ndim == 1 and any(k.endswith(s) for s in self._HF_NORM_SHIFT_SUFFIXES):
                    v = v + 1.0
            else:
                if k.endswith("ple.conv1d.weight"):
                    k = k.replace("ple.conv1d.weight", "ple.conv_weight")

            out[k] = v

        for dest, parts in stacked.items():
            out[dest] = mx.stack([parts[i] for i in range(len(parts))])

        out = _fuse_gate_up_sanitize(self, out)
        out = _fuse_gdn_in_proj_sanitize(self, out)
        out = _fuse_qsa_qkv_sanitize(self, out)

        # The 51B table never loads as a parameter; sidecar rows are gathered
        # lazily. Materialized shard concat is only accepted for tiny test
        # configs (parity harnesses) where the table actually fits.
        table_keys = [k for k in out if ".ngram_embedding.shard_" in k]
        if table_keys:
            shards = {}
            for k in table_keys:
                idx = int(k.rsplit("shard_", 1)[1].split(".")[0])
                shards[idx] = out.pop(k)
            table = mx.concatenate([shards[i] for i in range(len(shards))], axis=0)
            key = next(
                (
                    k
                    for k in _tree_keys(self)
                    if k.endswith("ple.ple_embedding.ngram_embedding.weight")
                ),
                None,
            )
            if key is not None:
                want = _tree_get(self, key).shape[0]
                if table.shape[0] < want:
                    pad = mx.zeros(
                        (want - table.shape[0], table.shape[1]), dtype=table.dtype
                    )
                    table = mx.concatenate([table, pad], axis=0)
                out[key] = table

        return out

    @property
    def cast_predicate(self):
        def predicate(path: str) -> bool:
            if path.endswith("A_log"):
                return False
            if "ngram" in path or path.endswith("layer_multipliers"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        """Convert-time recipe (Optimized Speed): 4-bit/g32 base; 8-bit/g64
        embeddings, lm_head, GDN out_proj, router gates, shared expert and the
        QSA indexer projection; the structural small stuff stays bf16."""

        def predicate(path: str, module) -> Union[bool, dict]:
            if not hasattr(module, "to_quantized"):
                return False
            eight = (
                "embed_tokens",
                "lm_head",
                "linear_attn.out_proj",
                "mlp.gate",
                "shared_expert_gate",
                "shared_expert.gate_proj",
                "shared_expert.up_proj",
                "shared_expert.down_proj",
                "indexer.index_qk_proj",
            )
            keep = (
                "input_mix_weight_down",
                "input_mix_weight_up",
                "block_inject_weight",
                "ple.key_proj",
                "ple.value_proj",
                "ngram_embedding",
            )
            if any(path.endswith(s) or s in path for s in keep):
                return False
            if any(path.endswith(s) for s in eight):
                return {"bits": 8, "group_size": 64}
            return True

        return predicate


def _tree_keys(module: nn.Module):
    from mlx.utils import tree_flatten

    return [k for k, _ in tree_flatten(module.parameters())]


def _tree_get(module: nn.Module, dotted: str):
    from mlx.utils import tree_flatten

    for k, v in tree_flatten(module.parameters()):
        if k == dotted:
            return v
    raise KeyError(dotted)


def is_qwen4_exp_mtp_config(config: dict) -> bool:
    """Does this artifact belong to the Flash-Next family?

    The family always exports its draft head as an ``mtp.safetensors``
    sidecar and the shipped configs carry no usable declaration field
    (``mtp``/``mtp_num_hidden_layers`` arrive null), so weight presence is
    decided by :meth:`Model.attach_mtp` at inject time — mirroring the
    DeepSeek-V4 "weight presence is decided later" convention.
    """
    cfg = config or {}
    mt = str(cfg.get("model_type") or "").lower()
    tc = cfg.get("text_config") or {}
    tmt = str(tc.get("model_type") or "").lower()
    return "qwen4_exp" in (mt, tmt) or "qwen4_exp_text" in (mt, tmt)


def inject_qwen4_exp_mtp_support(
    model,
    path=None,
    config: dict | None = None,
    contract=None,
) -> bool:
    """Enable the speculative lane on an already-loaded Flash-Next model.

    Nothing to graft: :meth:`Model.attach_mtp` builds the head from the
    pack's self-describing ``mtp.safetensors`` (per-module quant inferred
    from packed shapes) and publishes it on ``language_model`` where
    ``mtplx.mtp_patch.validate_mtp_support`` looks. Returns False for a
    pack that ships no head — the degrade-to-autoregressive signal.
    """
    if not is_qwen4_exp_mtp_config(config or {}):
        return False
    if path is None:
        return False
    attach = getattr(model, "attach_mtp", None)
    if not callable(attach):
        return False
    return bool(attach(path))
