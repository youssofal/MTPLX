"""Fused Q/K RMSNorm + plain RoPE for the Laguna S-2.1 sliding decode step.

Ported from the mlx.fast **Laguna XS2.1** challenge kernel
``laguna_sliding_qk_norm_rope_bf16_128_v1`` (``lagunaSlidingQKNormRoPEKernel`` /
``lagunaSlidingQKNormRoPE`` in Sources/MLXFastModel/LagunaRuntimeModel.swift),
re-expressed as a Python ``mx.fast.metal_kernel`` and *adapted* to Laguna S-2.1:

    sliding-attention heads   XS2.1 64   ->  S2.1 72   (**changed**)
    kv heads                  8          ->  8         (unchanged)
    head_dim                  128        ->  128       (unchanged)
    rotary dims               128 (full) ->  128 (full, 64 pairs)
    rope theta                10000      ->  10000     (unchanged)

The 30 sliding layers carry PLAIN RoPE: the whole 128-element head rotates
(``partial_rotary_factor 1.0``), the angle scale is one, and there is NO YaRN
mscale — mlx-lm builds ``nn.RoPE(dims=128, base=10000)`` for them (see
``models/laguna.py`` ``_rope_for`` with ``swa_rope_parameters`` rope_type
``default``, theta 10000).  The one S-2.1 shape change is the query head count:
72 sliding heads vs the challenge's 64, so the kernel dispatches ``(72 + 8) * 32``
threads instead of ``(64 + 8) * 32``.

## The single dispatch it replaces

The stock chain per sliding decode layer is four dispatches — ``q_norm``
(RMSNorm), ``k_norm`` (RMSNorm), ``RoPE(q)``, ``RoPE(k)`` — over 72x128 and
8x128 elements.  This kernel does all of it in ONE dispatch, one 32-lane
simdgroup per head (80 = 72 + 8 heads), writing the transposed
``[1, heads, 1, 128]`` layout attention consumes directly.

## Bit-exactness, link for link

* **RMSNorm** mirrors ``rms_single_row`` at a 128-wide axis: 32 lanes x 4 FP32
  squares, one ``simd_sum`` (total returned to every lane -> local
  ``precise::rsqrt``, the barrier-elided form), then ``w[i] * bfloat(float(x[i])
  * inv)`` — the same double rounding ``mx.fast.rms_norm`` writes.

* **The rotary angles** (cos/sin) are supplied as a precomputed ``angles`` table
  (length 128: 64 cos then 64 sin) rather than re-derived:
  :func:`build_sliding_rope_angles` runs the layer's own ``mx.fast.rope`` over a
  ``[ones(64), zeros(64)]`` seed at the current offset, so the table holds the
  EXACT cos/sin bits ``mx.fast.rope`` uses — the rotation is bitwise the rope's,
  not a re-derivation.  Full rotary, so pair ``p`` couples elements ``p`` and
  ``p + 64``.

Callers gate on :func:`is_qk_rope_sliding_eligible` first; the public helper
falls back to the stock ``q_norm``/``k_norm`` -> transpose -> ``RoPE`` chain on
any shape it does not cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import mlx.core as mx


# S-2.1 sliding-attention decode shape, baked.
_N_Q_HEADS = 72
_N_KV_HEADS = 8
_HEAD_DIM = 128
_ROT_DIMS = 128  # full rotary
_ROT_PAIRS = _ROT_DIMS // 2  # 64
_SIMD = 32
_ROPE_THETA = 10000.0


@dataclass(frozen=True)
class SlidingRopeSpec:
    """Shape + rotary geometry for the sliding plain-rope qk-norm+rope kernel."""

    n_q_heads: int = _N_Q_HEADS
    n_kv_heads: int = _N_KV_HEADS
    head_dim: int = _HEAD_DIM
    rot_dims: int = _ROT_DIMS
    base: float = _ROPE_THETA
    eps: float = 1e-6

    @property
    def total_heads(self) -> int:
        return self.n_q_heads + self.n_kv_heads

    @property
    def rot_pairs(self) -> int:
        return self.rot_dims // 2


def build_sliding_rope_angles(
    offset: int | mx.array, spec: SlidingRopeSpec = SlidingRopeSpec()
) -> mx.array:
    """Exact cos/sin table for the plain rotation at ``offset``.

    Runs ``mx.fast.rope`` (base ``theta``, full rotary) over a
    ``[ones(64), zeros(64)]`` seed, so it returns exactly
    ``[cos_0..cos_63, sin_0..sin_63]`` — the floats ``mx.fast.rope`` uses,
    making the kernel's rotation bitwise the rope's.
    """

    half = spec.rot_pairs
    seed = mx.concatenate(
        [mx.ones((half,), dtype=mx.float32), mx.zeros((half,), dtype=mx.float32)]
    ).reshape(1, 1, 1, spec.rot_dims)
    angles = mx.fast.rope(
        seed,
        spec.rot_dims,
        traditional=False,
        base=spec.base,
        scale=1.0,
        offset=offset,
    )
    return angles.reshape(1, 1, 1, spec.rot_dims)


def is_qk_rope_sliding_eligible(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    angles: mx.array,
    spec: SlidingRopeSpec,
) -> bool:
    """Whether the fused kernel covers this exact sliding decode shape."""

    if not mx.metal.is_available():
        return False
    try:
        if mx.default_device() != mx.gpu:
            return False
    except Exception:
        return False
    if queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16:
        return False
    if q_weight.dtype != mx.bfloat16 or k_weight.dtype != mx.bfloat16:
        return False
    if angles.dtype != mx.float32:
        return False
    if spec.head_dim != _HEAD_DIM or spec.rot_dims != _ROT_DIMS:
        return False
    if queries.ndim != 3 or keys.ndim != 3:
        return False
    if int(queries.shape[0]) != 1 or int(queries.shape[1]) != 1:
        return False
    if int(keys.shape[0]) != 1 or int(keys.shape[1]) != 1:
        return False
    if int(queries.shape[-1]) != spec.n_q_heads * spec.head_dim:
        return False
    if int(keys.shape[-1]) != spec.n_kv_heads * spec.head_dim:
        return False
    if int(q_weight.size) != spec.head_dim or int(k_weight.size) != spec.head_dim:
        return False
    if int(angles.size) != spec.rot_dims:
        return False
    return True


@lru_cache(maxsize=None)
def _qk_rope_sliding_kernel(
    n_q_heads: int, n_kv_heads: int, rot_dims: int, eps: float
):
    header = f"""
        using namespace metal;
        constant constexpr uint HEAD_DIM = {_HEAD_DIM};
        constant constexpr uint ROT_DIMS = {rot_dims};
        constant constexpr uint ROT_PAIRS = {rot_dims // 2};
        constant constexpr uint QUERY_HEADS = {n_q_heads};
        constant constexpr float RMS_EPS = {eps!r}f;
    """

    # One 32-lane simdgroup per head; lane `l` owns [4l, 4l+4).  simd_sum returns
    # the RMS statistic to every lane, so no threadgroup slot or barrier.
    source = """
        uint head = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;

        const device T* input;
        const device T* weight;
        if (head < QUERY_HEADS) {
            input = raw_queries + head * HEAD_DIM;
            weight = query_weight;
        } else {
            input = raw_keys + (head - QUERY_HEADS) * HEAD_DIM;
            weight = key_weight;
        }

        uint base = lane * 4;
        thread T normalized[4];
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float value = float(input[base + i]);
            sum += value * value;
        }
        sum = simd_sum(sum);
        float inverse_rms = metal::precise::rsqrt(sum / float(HEAD_DIM) + RMS_EPS);

        for (uint i = 0; i < 4; ++i) {
            normalized[i] =
                weight[base + i] *
                static_cast<T>(float(input[base + i]) * inverse_rms);
        }

        // Full rotary: element `p + 64`, the partner of pair `p`, is 16 lanes
        // away (base == lane*4, ROT_PAIRS == 64).
        thread float paired[4];
        for (uint i = 0; i < 4; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
        }

        device T* output =
            head < QUERY_HEADS
            ? queries + head * HEAD_DIM
            : keys + (head - QUERY_HEADS) * HEAD_DIM;

        // Every element rotates: lower 16 lanes own all 64 pairs [0, 64) and
        // write both halves of each.
        if (lane < HEAD_DIM / 8u) {
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first = float(normalized[i]);
                float second = paired[i];
                float cosine = angles[pair];
                float sine = angles[pair + ROT_PAIRS];
                output[pair] = static_cast<T>(first * cosine - second * sine);
                output[pair + ROT_PAIRS] =
                    static_cast<T>(first * sine + second * cosine);
            }
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_laguna_qk_rope_sliding_hq{n_q_heads}_hkv{n_kv_heads}"
            f"_r{rot_dims}_v1"
        ),
        input_names=["raw_queries", "raw_keys", "query_weight", "key_weight", "angles"],
        output_names=["queries", "keys"],
        header=header,
        source=source,
    )


def _stock_qk_rope_sliding(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    offset: int | mx.array,
    spec: SlidingRopeSpec,
) -> tuple[mx.array, mx.array]:
    """Stock chain: q_norm/k_norm -> transpose -> plain RoPE, for q and k.

    Reproduces ``Attention.__call__`` at T == 1 with mlx-lm ``nn.RoPE`` (base
    theta, full rotary): RMSNorm over head_dim, transpose to head-major, then
    ``mx.fast.rope(dims=128, base=theta)``.
    """

    n_q, n_kv, hd, rd = (
        spec.n_q_heads,
        spec.n_kv_heads,
        spec.head_dim,
        spec.rot_dims,
    )

    def one(x, weight, n_heads):
        normed = mx.fast.rms_norm(
            x.reshape(1, 1, n_heads, hd), weight, spec.eps
        ).transpose(0, 2, 1, 3)  # [1, n_heads, 1, hd]
        return mx.fast.rope(
            normed,
            rd,
            traditional=False,
            base=spec.base,
            scale=1.0,
            offset=offset,
        )

    return one(queries, q_weight, n_q), one(keys, k_weight, n_kv)


def fused_qk_rope_sliding_reference(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    angles: mx.array,
    spec: SlidingRopeSpec,
) -> tuple[mx.array, mx.array]:
    """Pure-mx reference implementing the exact math the metal kernel computes.

    RMSNorm (== ``mx.fast.rms_norm``) then a full rotation with the pure cos/sin
    from ``angles`` (== ``mx.fast.rope``).  No mscale (plain rope).
    """

    n_q, n_kv, hd, rd = (
        spec.n_q_heads,
        spec.n_kv_heads,
        spec.head_dim,
        spec.rot_dims,
    )
    half = rd // 2
    cos = angles.reshape(rd)[:half].reshape(1, 1, 1, half)
    sin = angles.reshape(rd)[half:].reshape(1, 1, 1, half)

    def one(x, weight, n_heads):
        normed = mx.fast.rms_norm(
            x.reshape(1, 1, n_heads, hd), weight, spec.eps
        ).transpose(0, 2, 1, 3)  # [1, n_heads, 1, hd], bf16
        v1 = normed[..., :half].astype(mx.float32)
        v2 = normed[..., half:rd].astype(mx.float32)
        rot_lo = (v1 * cos - v2 * sin).astype(mx.bfloat16)
        rot_hi = (v1 * sin + v2 * cos).astype(mx.bfloat16)
        return mx.concatenate([rot_lo, rot_hi], axis=-1)

    return one(queries, q_weight, n_q), one(keys, k_weight, n_kv)


def fused_qk_rope_sliding(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    angles: mx.array,
    spec: SlidingRopeSpec,
    *,
    offset: int | mx.array = 0,
) -> tuple[mx.array, mx.array]:
    """Fused q/k RMSNorm + plain RoPE for one sliding decode row.

    Returns ``(queries, keys)`` shaped ``[1, n_q_heads, 1, head_dim]`` and
    ``[1, n_kv_heads, 1, head_dim]``.  ``angles`` is from
    :func:`build_sliding_rope_angles`.  Falls back to the stock ``RoPE`` chain on
    any shape the kernel does not cover; the fallback needs ``offset`` to rope.
    """

    if not is_qk_rope_sliding_eligible(
        queries, keys, q_weight, k_weight, angles, spec
    ):
        q_out, k_out = _stock_qk_rope_sliding(
            queries, keys, q_weight, k_weight, offset, spec
        )
    else:
        kernel = _qk_rope_sliding_kernel(
            spec.n_q_heads, spec.n_kv_heads, spec.rot_dims, float(spec.eps)
        )
        q_out, k_out = kernel(
            inputs=[queries, keys, q_weight, k_weight, angles],
            template=[("T", queries.dtype)],
            grid=(_SIMD * spec.total_heads, 1, 1),
            threadgroup=(_SIMD, 1, 1),
            output_shapes=[
                (1, spec.n_q_heads, 1, spec.head_dim),
                (1, spec.n_kv_heads, 1, spec.head_dim),
            ],
            output_dtypes=[queries.dtype, queries.dtype],
        )

    assert tuple(q_out.shape) == (1, spec.n_q_heads, 1, spec.head_dim), q_out.shape
    assert tuple(k_out.shape) == (1, spec.n_kv_heads, 1, spec.head_dim), k_out.shape
    return q_out, k_out
