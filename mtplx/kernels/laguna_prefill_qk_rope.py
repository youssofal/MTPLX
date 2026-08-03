"""Fused q/k RMSNorm + rope for the Laguna S-2.1 PREFILL attention block.

Prefill twin of the decode ``fused_qk_norm_rope`` in ``laguna_decode.py``.  It
is the SAME per-(head, position) math the decode kernel is bit-exact against,
applied over ``T > 1`` positions instead of the single decode token, with a
per-position offset so every row of the sequence gets a distinct rotation.

Ported from the challenge's own prefill QK-norm+rope fusions
(``laguna_full_qk_norm_yarn_bf16_128_v4`` /
``laguna_sliding_qk_norm_rope_bf16_128_v1`` in
``Sources/MLXFastModel/LagunaRuntimeModel.swift``), re-expressed for S-2.1's two
attention families and its exact rope constants.  Both families in ONE kernel
family (specialized by the spec), one dispatch per layer:

  FULL attention layers  (48 q heads, 8 kv heads): partial YaRN rope over the
      first ``rot_dims = 64`` dims (head_dim * partial_rotary 0.5), theta
      500000, factor 128, mscale 1.4852030263919618.  The rope frequencies are
      the interpolated ``YarnRoPE._freqs`` buffer, captured into the spec so the
      kernel can only ever see the same floats the stock path uses.  The tail
      dims 64..127 are RMSNorm output with no mscale and no rotation, exactly
      what ``mx.fast.rope`` produces for a partial rotary.

  SLIDING attention layers (72 q heads, 8 kv heads): full-rotary rope over all
      128 dims, theta 10000, no mscale (the ``nn.RoPE`` base form).

Exactness follows the decode kernel link for link (it documents the derivation):
RMSNorm reproduces ``rms_single_row`` (``w * static_cast<T>(x * inv)``,
``precise::rsqrt(acc/128 + eps)``); the YaRN pre-scale rounds mscale to the
tensor dtype and multiplies in that dtype on the rotary dims only; the rotation
reproduces ``mx.fast.rope`` (``theta = position / freqs[p]`` for the freqs form
or ``position * base^(-2p/dims)`` for the base form, ``metal::fast::cos/sin``,
pairs ``(p, p + dims/2)``).

BATCHED-ROPE OFFSET TRAP.  MLX 0.31.2's ``mx.fast.rope`` only writes row 0 when
handed a length-1 sequence with a scalar offset (the decode corruption
``_rope_offset`` fixes).  This kernel sidesteps it structurally: it takes a
length-T int32 ``positions`` vector (``positions[t] = base_offset + t``) and
computes ``L = float(positions[t])`` per token, so every one of the T rows gets
its own rotation by construction.  The CPU check asserts all rows are distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import log2
from typing import Optional

import mlx.core as mx

_QK_HEAD_DIM = 128
_QK_LANES = 32


def _on_metal_device() -> bool:
    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


@dataclass(frozen=True)
class QkRopePrefillSpec:
    """Per-layer constants for the fused prefill q/k norm+rope kernel.

    Captured from the layer's own rope module so the kernel can only see the
    exact frequencies, base and mscale the stock path uses.  Exactly one of
    ``freqs`` (the YaRN interpolated periods) or ``base`` (the plain rope theta)
    is set: ``freqs`` for the FULL/YaRN family, ``base`` for the SLIDING family.
    """

    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    rot_dims: int
    freqs: Optional[mx.array]  # float32 [rot_dims // 2], YaRN interpolated
    base: Optional[float]  # rope theta when freqs is None (base form)
    mscale: Optional[float]  # YaRN attention factor, None/1.0 when absent


def is_qk_norm_rope_prefill_eligible(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    spec: "QkRopePrefillSpec | None",
) -> bool:
    if spec is None or not _on_metal_device():
        return False
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return False
    if keys.dtype != queries.dtype:
        return False
    if q_weight.dtype != queries.dtype or k_weight.dtype != queries.dtype:
        return False
    if spec.head_dim != _QK_HEAD_DIM or spec.rot_dims not in (64, 128):
        return False
    if spec.freqs is None and spec.base is None:
        return False
    if spec.freqs is not None:
        if spec.freqs.dtype != mx.float32:
            return False
        if int(spec.freqs.size) != spec.rot_dims // 2:
            return False
    if queries.ndim != 3 or keys.ndim != 3:
        return False
    if int(queries.shape[1]) != int(keys.shape[1]):  # same T
        return False
    if int(queries.shape[-1]) != spec.n_q_heads * spec.head_dim:
        return False
    if int(keys.shape[-1]) != spec.n_kv_heads * spec.head_dim:
        return False
    if int(q_weight.size) != spec.head_dim or int(k_weight.size) != spec.head_dim:
        return False
    return int(queries.shape[0]) == int(keys.shape[0])


@lru_cache(maxsize=None)
def _qk_norm_rope_prefill_kernel(
    n_q_heads: int,
    n_kv_heads: int,
    rot_dims: int,
    use_freqs: bool,
    base_log2: float,
    mscale: float,
):
    has_mscale = mscale != 1.0
    header = f"""
        using namespace metal;
        constant constexpr int HQ = {n_q_heads};
        constant constexpr int HKV = {n_kv_heads};
        constant constexpr int HEAD_DIM = {_QK_HEAD_DIM};
        constant constexpr int ROT_DIMS = {rot_dims};
        constant constexpr int HALF_ROT = ROT_DIMS / 2;
        constant constexpr bool USE_FREQS = {"true" if use_freqs else "false"};
        constant constexpr bool HAS_MSCALE = {"true" if has_mscale else "false"};
        constant constexpr float MSCALE_F = {mscale!r}f;
        constant constexpr float BASE_LOG2 = {base_log2!r}f;
    """

    # One threadgroup (32 lanes) per (batch, position, head).  The head axis
    # covers q heads then k heads, so a single dispatch norms+ropes both.  T
    # (the sequence length) is a runtime scalar so one compiled variant serves
    # every prefill length; the per-position rope angle comes from positions[t].
    source = """
        uint tg = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        constexpr int TOTAL_HEADS = HQ + HKV;

        uint SEQ = uint(seq_len);
        uint per_b = SEQ * uint(TOTAL_HEADS);
        uint b = tg / per_b;
        uint rem = tg - b * per_b;
        uint t = rem / uint(TOTAL_HEADS);
        uint hg = rem - t * uint(TOTAL_HEADS);
        bool is_q = hg < uint(HQ);
        uint h = is_q ? hg : (hg - uint(HQ));
        uint H_this = is_q ? uint(HQ) : uint(HKV);

        // in  [B, T, H_this*HEAD_DIM] : (b,t,h,:) -> ((b*T + t)*H_this + h)*D
        // out [B, H_this, T, HEAD_DIM]: (b,h,t,:) -> ((b*H_this + h)*T + t)*D
        size_t in_base =
            ((size_t)(b * SEQ + t) * (size_t)H_this + (size_t)h) * (size_t)HEAD_DIM;
        size_t out_base =
            ((size_t)(b * H_this + h) * (size_t)SEQ + (size_t)t) * (size_t)HEAD_DIM;
        const device T* src = (is_q ? q_in : k_in) + in_base;
        const device T* w = is_q ? q_w : k_w;
        device T* dst = (is_q ? q_out : k_out) + out_base;

        // RMS statistic, exactly as MLX's rms_single_row lays it out for a
        // 128-wide axis: 32 lanes x 4 sequential float squares, one simd_sum.
        float acc = 0.0f;
        uint sbase = lane * 4;
        for (int i = 0; i < 4; ++i) {
            float xi = static_cast<float>(src[sbase + i]);
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        float inv = metal::precise::rsqrt(acc / float(HEAD_DIM) + eps);

        // Per-position rope angle: L = base_offset + t, delivered as positions[t]
        // so every row of the sequence rotates by a distinct amount.
        float L = float(positions[t]);

        if (ROT_DIMS == HEAD_DIM) {
            for (uint p = lane; p < uint(HALF_ROT); p += 32u) {
                float inv_freq;
                if (USE_FREQS) {
                    inv_freq = 1.0 / (freqs[p]);
                } else {
                    float d = float(p) / float(HALF_ROT);
                    inv_freq = metal::exp2(-d * BASE_LOG2);
                }
                float theta = L * inv_freq;
                float costheta = metal::fast::cos(theta);
                float sintheta = metal::fast::sin(theta);
                T v1 = w[p] * static_cast<T>(src[p] * inv);
                T v2 = w[p + uint(HALF_ROT)] *
                    static_cast<T>(src[p + uint(HALF_ROT)] * inv);
                if (HAS_MSCALE) {
                    v1 = static_cast<T>(MSCALE_F) * v1;
                    v2 = static_cast<T>(MSCALE_F) * v2;
                }
                float x1 = static_cast<float>(v1);
                float x2 = static_cast<float>(v2);
                dst[p] = static_cast<T>(x1 * costheta - x2 * sintheta);
                dst[p + uint(HALF_ROT)] =
                    static_cast<T>(x1 * sintheta + x2 * costheta);
            }
        } else {
            // Partial rotary: rotate pairs (p, p + HALF_ROT) inside the first
            // ROT_DIMS dims; the tail is normed output with NO mscale and NO
            // rotation, exactly what the stock partial rope produces.
            if (lane < uint(HALF_ROT)) {
                uint p = lane;
                float inv_freq;
                if (USE_FREQS) {
                    inv_freq = 1.0 / (freqs[p]);
                } else {
                    float d = float(p) / float(HALF_ROT);
                    inv_freq = metal::exp2(-d * BASE_LOG2);
                }
                float theta = L * inv_freq;
                float costheta = metal::fast::cos(theta);
                float sintheta = metal::fast::sin(theta);
                T v1 = w[p] * static_cast<T>(src[p] * inv);
                T v2 = w[p + uint(HALF_ROT)] *
                    static_cast<T>(src[p + uint(HALF_ROT)] * inv);
                if (HAS_MSCALE) {
                    v1 = static_cast<T>(MSCALE_F) * v1;
                    v2 = static_cast<T>(MSCALE_F) * v2;
                }
                float x1 = static_cast<float>(v1);
                float x2 = static_cast<float>(v2);
                dst[p] = static_cast<T>(x1 * costheta - x2 * sintheta);
                dst[p + uint(HALF_ROT)] =
                    static_cast<T>(x1 * sintheta + x2 * costheta);
            }
            constexpr int TAIL = HEAD_DIM - ROT_DIMS;
            constexpr int PER_LANE = TAIL / 32;
            for (int i = 0; i < PER_LANE; ++i) {
                uint tt = uint(ROT_DIMS) + lane * uint(PER_LANE) + uint(i);
                dst[tt] = w[tt] * static_cast<T>(src[tt] * inv);
            }
        }
    """

    name = (
        f"mtplx_laguna_prefill_qk_rope_hq{n_q_heads}_hkv{n_kv_heads}"
        f"_r{rot_dims}_{'freqs' if use_freqs else 'base'}"
        f"{'_ms' if has_mscale else ''}"
    )
    return mx.fast.metal_kernel(
        name=name,
        input_names=["q_in", "k_in", "q_w", "k_w", "freqs", "eps", "positions", "seq_len"],
        output_names=["q_out", "k_out"],
        header=header,
        source=source,
    )


_DUMMY_FREQS: Optional[mx.array] = None


def _positions_vector(offset, length: int) -> mx.array:
    """A length-T int32 position vector ``offset + [0, 1, ..., T-1]``.

    Takes an int offset or an int32 scalar/1-element array (never an int() on a
    graph leaf, which would sync the stream under compile).
    """

    base = mx.arange(length, dtype=mx.int32)
    if isinstance(offset, mx.array):
        return base + offset.astype(mx.int32).reshape(())
    return base + mx.array(int(offset), dtype=mx.int32)


def _stock_qk_norm_rope_prefill(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    eps: float,
    offset,
    spec: QkRopePrefillSpec,
) -> tuple[mx.array, mx.array]:
    """The shipped op chain: RMSNorm -> transpose -> (YaRN pre-scale) -> rope.

    Reproduces the model's Attention path (``mx.fast.rms_norm`` then the rope
    module) using only the spec fields, so it is a faithful stock reference for
    both the fallback and the checks.
    """

    def one(x_in: mx.array, w: mx.array, n_heads: int) -> mx.array:
        batch, length, _ = x_in.shape
        normed = mx.fast.rms_norm(
            x_in.reshape(batch, length, n_heads, spec.head_dim), w, eps
        ).transpose(0, 2, 1, 3)
        rot = spec.rot_dims
        if spec.mscale is not None and spec.mscale != 1.0:
            scaled = mx.array(spec.mscale).astype(normed.dtype) * normed[..., :rot]
            normed = mx.concatenate([scaled, normed[..., rot:]], axis=-1)
        base = None if spec.freqs is not None else float(spec.base)
        return mx.fast.rope(
            normed,
            rot,
            traditional=False,
            base=base,
            scale=1.0,
            offset=offset,
            freqs=spec.freqs,
        )

    return one(queries, q_weight, spec.n_q_heads), one(keys, k_weight, spec.n_kv_heads)


def fused_qk_norm_rope_prefill(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    eps: float,
    offset,
    spec: QkRopePrefillSpec,
) -> tuple[mx.array, mx.array]:
    """RMSNorm + (partial/YaRN) rope for q and k over T positions, one dispatch.

    ``queries`` is ``[B, T, n_q_heads*head_dim]`` and ``keys`` is
    ``[B, T, n_kv_heads*head_dim]`` -- the raw projection outputs, before the
    stock reshape/norm/transpose.  Returns ``(q, k)`` shaped
    ``[B, n_q_heads, T, head_dim]`` / ``[B, n_kv_heads, T, head_dim]``, the
    layout attention reads.  ``offset`` is the base rope position (cache offset);
    per-position offsets ``offset + t`` are built into a length-T vector.

    Falls back to the stock chain on any shape the kernel does not cover, so
    callers can switch it on without owning a correctness branch.
    """

    if not is_qk_norm_rope_prefill_eligible(queries, keys, q_weight, k_weight, spec):
        return _stock_qk_norm_rope_prefill(
            queries, keys, q_weight, k_weight, eps, offset, spec
        )

    global _DUMMY_FREQS
    batch = int(queries.shape[0])
    length = int(queries.shape[1])
    freqs = spec.freqs
    if freqs is None:
        if _DUMMY_FREQS is None:
            _DUMMY_FREQS = mx.ones((1,), dtype=mx.float32)
        freqs = _DUMMY_FREQS

    base_log2 = log2(float(spec.base)) if spec.base is not None else 0.0
    kernel = _qk_norm_rope_prefill_kernel(
        spec.n_q_heads,
        spec.n_kv_heads,
        spec.rot_dims,
        spec.freqs is not None,
        base_log2,
        float(spec.mscale) if spec.mscale is not None else 1.0,
    )
    total_heads = spec.n_q_heads + spec.n_kv_heads
    positions = _positions_vector(offset, length)
    q_out, k_out = kernel(
        inputs=[
            queries,
            keys,
            q_weight,
            k_weight,
            freqs,
            float(eps),
            positions,
            int(length),
        ],
        template=[("T", queries.dtype)],
        grid=(_QK_LANES * batch * length * total_heads, 1, 1),
        threadgroup=(_QK_LANES, 1, 1),
        output_shapes=[
            (batch, spec.n_q_heads, length, spec.head_dim),
            (batch, spec.n_kv_heads, length, spec.head_dim),
        ],
        output_dtypes=[queries.dtype, queries.dtype],
    )
    # Fake-speedup guard: exactly the transposed head-major attention layout.
    assert tuple(q_out.shape) == (batch, spec.n_q_heads, length, spec.head_dim), (
        f"q_out {tuple(q_out.shape)} != "
        f"{(batch, spec.n_q_heads, length, spec.head_dim)}"
    )
    assert tuple(k_out.shape) == (batch, spec.n_kv_heads, length, spec.head_dim), (
        f"k_out {tuple(k_out.shape)} != "
        f"{(batch, spec.n_kv_heads, length, spec.head_dim)}"
    )
    return q_out, k_out


def qk_norm_rope_prefill_reference(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    eps: float,
    offset,
    spec: QkRopePrefillSpec,
) -> tuple[mx.array, mx.array]:
    """Pure-mx reference: explicit RMSNorm + explicit rotation over T positions.

    A from-scratch re-derivation (not ``mx.fast``) that mirrors the kernel's own
    arithmetic, so the CPU check cross-validates the math and the all-rows-
    distinct property independently of the shipped op chain.
    """

    bf = queries.dtype
    half = spec.rot_dims // 2
    if spec.freqs is not None:
        inv_freq = (1.0 / spec.freqs).astype(mx.float32)  # [half]
    else:
        # base^(-2p/dims): the plain rope inv-freq the base-form kernel builds
        # via exp2(-(p/half) * log2(base)).
        p = mx.arange(half, dtype=mx.float32)
        inv_freq = mx.power(mx.array(float(spec.base)), -(2.0 * p / spec.rot_dims))

    def one(x_in: mx.array, w: mx.array, n_heads: int) -> mx.array:
        batch, length, _ = x_in.shape
        x = x_in.reshape(batch, length, n_heads, spec.head_dim).astype(mx.float32)
        inv = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
        normed = w * (x * inv).astype(bf)  # [B,T,H,D], bf16
        normed = normed.transpose(0, 2, 1, 3)  # [B,H,T,D]
        rot = spec.rot_dims
        r = normed
        if spec.mscale is not None and spec.mscale != 1.0:
            scaled = mx.array(spec.mscale).astype(bf) * r[..., :rot]
            r = mx.concatenate([scaled, r[..., rot:]], axis=-1)
        xp = r[..., :half].astype(mx.float32)
        xq = r[..., half:rot].astype(mx.float32)
        pos = _positions_vector(offset, length).astype(mx.float32)  # [T]
        theta = pos[:, None] * inv_freq[None, :]  # [T, half]
        cos = mx.cos(theta)[None, None]
        sin = mx.sin(theta)[None, None]
        rotated = mx.concatenate([xp * cos - xq * sin, xp * sin + xq * cos], axis=-1)
        rotated = rotated.astype(bf)
        if rot < spec.head_dim:
            return mx.concatenate([rotated, normed[..., rot:]], axis=-1)
        return rotated

    return one(queries, q_weight, spec.n_q_heads), one(keys, k_weight, spec.n_kv_heads)
