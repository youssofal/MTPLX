"""Fused Mia compressor finalization for fixed DeepSeek V4 cache records.

The two dense projections remain MLX GEMMs, matching the source's
``fused_wkv_wgate`` boundary.  These kernels own everything after those GEMMs:
per-dimension window softmax, gated reduction, RMS normalization, compressor
RoPE, quantization, and construction of the exact cache record.  Consequently
the Mia route never materializes a dense compressed history merely to repack it
into the paged cache.  The pinned image patch keeps pooling, RMSNorm, and RoPE
in FP32, then crosses one BF16 boundary before either stock432 or Mia132 record
quantization.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from mtplx.deepseek_v4_nvfp4_kv import (
    MIA_NVFP4_RECORD_BYTES,
    _NVFP4_HEADER,
)
from mtplx.deepseek_v4_paged_indexer import (
    INDEXER_RECORD_BYTES,
    _FP8_HEADER,
)


_REDUCTION_HEADER = r"""
    using namespace metal;

    inline float mtplx_bf16_roundtrip(float value) {
        return float(bfloat(value));
    }
"""


# Mia's Triton compressor classifies sign with ``value < 0``.  The ordinary
# target/draft stock432 writer keeps the base header's IEEE sign-bit contract.
_MIA_COMPRESSOR_NVFP4_HEADER = _NVFP4_HEADER + r"""
    inline uchar mtplx_mia_compressor_e2m1_encode(float value) {
        uchar code = mtplx_e2m1_encode(value);
        uint sign = value < 0.0f ? 1u : 0u;
        return uchar((code & uchar(0x07)) | uchar(sign << 3));
    }
"""


@lru_cache(maxsize=2)
def _nvfp4_finalize_kernel(compress_ratio: int):
    ratio = int(compress_ratio)
    if ratio not in (4, 128):
        raise ValueError("Mia attention compressor ratio must be 4 or 128")
    overlap = ratio == 4
    header = _MIA_COMPRESSOR_NVFP4_HEADER + _REDUCTION_HEADER + f"""
        constant constexpr uint MTPLX_COMPRESS_RATIO = {ratio}u;
        constant constexpr bool MTPLX_COMPRESS_OVERLAP = {'true' if overlap else 'false'};
        constant constexpr uint MTPLX_COMPRESS_HEAD = 512u;
        constant constexpr uint MTPLX_COMPRESS_ROPE = 64u;
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint out_row = threadgroup_position_in_grid.x;
        uint current_window = out_row;

        threadgroup float normed[512];
        threadgroup float record_values[512];
        threadgroup float simd_sums[16];
        threadgroup float rrms_shared;
        threadgroup float group_scales[32];

        const device T* current_kv = kv_windows
            + size_t(current_window) * MTPLX_COMPRESS_RATIO
                * (MTPLX_COMPRESS_OVERLAP ? 1024u : 512u);
        const device T* current_score = score_windows
            + size_t(current_window) * MTPLX_COMPRESS_RATIO
                * (MTPLX_COMPRESS_OVERLAP ? 1024u : 512u);

        float max_score = -INFINITY;
        if (MTPLX_COMPRESS_OVERLAP && (has_previous || current_window > 0u)) {
            const device T* previous_score = has_previous && current_window == 0u
                ? previous_score_window
                : score_windows
                    + size_t(current_window - 1u) * MTPLX_COMPRESS_RATIO * 1024u;
            for (uint slot = 0u; slot < MTPLX_COMPRESS_RATIO; ++slot) {
                max_score = max(
                    max_score,
                    float(previous_score[size_t(slot) * 1024u + tid])
                );
            }
        }
        uint current_half = MTPLX_COMPRESS_OVERLAP ? 512u : 0u;
        for (uint slot = 0u; slot < MTPLX_COMPRESS_RATIO; ++slot) {
            max_score = max(
                max_score,
                float(current_score[size_t(slot)
                    * (MTPLX_COMPRESS_OVERLAP ? 1024u : 512u)
                    + current_half + tid])
            );
        }

        float denominator = 0.0f;
        float numerator = 0.0f;
        if (MTPLX_COMPRESS_OVERLAP && (has_previous || current_window > 0u)) {
            const device T* previous_kv = has_previous && current_window == 0u
                ? previous_kv_window
                : kv_windows
                    + size_t(current_window - 1u) * MTPLX_COMPRESS_RATIO * 1024u;
            const device T* previous_score = has_previous && current_window == 0u
                ? previous_score_window
                : score_windows
                    + size_t(current_window - 1u) * MTPLX_COMPRESS_RATIO * 1024u;
            for (uint slot = 0u; slot < MTPLX_COMPRESS_RATIO; ++slot) {
                float probability = exp(
                    float(previous_score[size_t(slot) * 1024u + tid]) - max_score
                );
                denominator += probability;
                numerator += probability
                    * float(previous_kv[size_t(slot) * 1024u + tid]);
            }
        }
        for (uint slot = 0u; slot < MTPLX_COMPRESS_RATIO; ++slot) {
            size_t offset = size_t(slot)
                * (MTPLX_COMPRESS_OVERLAP ? 1024u : 512u)
                + current_half + tid;
            float probability = exp(float(current_score[offset]) - max_score);
            denominator += probability;
            numerator += probability * float(current_kv[offset]);
        }
        float pooled = numerator / denominator;
        float local_sq = pooled * pooled;
        float simd_sq = simd_sum(local_sq);
        if (lane == 0u) {
            simd_sums[sg] = simd_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total = 0.0f;
            for (uint i = 0u; i < 16u; ++i) {
                total += simd_sums[i];
            }
            rrms_shared = rsqrt(total / 512.0f + float(rms_eps));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float normalized = pooled * rrms_shared * float(norm_weight[tid]);
        normed[tid] = normalized;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rotated = normed[tid];
        if (tid >= 448u) {
            uint rope_dim = tid - 448u;
            uint pair = rope_dim / 2u;
            uint even_dim = 448u + pair * 2u;
            float even = normed[even_dim];
            float odd = normed[even_dim + 1u];
            float c = float(rope_cos[size_t(out_row) * 32u + pair]);
            float s = float(rope_sin[size_t(out_row) * 32u + pair]);
            rotated = (rope_dim & 1u) == 0u
                ? even * c - odd * s
                : odd * c + even * s;
        }
        float record_value = mtplx_bf16_roundtrip(rotated);
        record_values[tid] = record_value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint group = tid / 16u;
        float group_max = abs(record_value);
        group_max = max(group_max, simd_shuffle_down(group_max, 8u));
        group_max = max(group_max, simd_shuffle_down(group_max, 4u));
        group_max = max(group_max, simd_shuffle_down(group_max, 2u));
        group_max = max(group_max, simd_shuffle_down(group_max, 1u));
        if ((lane & 15u) == 0u) {
            uchar scale_byte = mtplx_e4m3_encode_positive(group_max / 6.0f);
            group_scales[group] = mtplx_e4m3_decode(scale_byte);
            records[size_t(out_row) * 432u + 256u + group] = scale_byte;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device uchar* record = records + size_t(out_row) * 432u;
        if ((tid & 1u) == 0u) {
            float scale = group_scales[group];
            float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
            uchar low = mtplx_mia_compressor_e2m1_encode(record_value * inverse);
            uchar high = mtplx_mia_compressor_e2m1_encode(
                record_values[tid + 1u] * inverse
            );
            record[tid / 2u] = uchar(low | uchar(high << 4));
        }
        if (tid < 16u) {
            record[288u + tid] = uchar(0);
        }

        if (tid >= 448u) {
            uint rope_dim = tid - 448u;
            bfloat stored = bfloat(record_value);
            ushort bits = as_type<ushort>(stored);
            uint byte = rope_dim * 2u;
            record[304u + byte] = uchar(bits & 0xffu);
            record[304u + byte + 1u] = uchar(bits >> 8u);
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_mia_fused_compress_nvfp4_r{ratio}",
        input_names=[
            "kv_windows",
            "score_windows",
            "previous_kv_window",
            "previous_score_window",
            "norm_weight",
            "rope_cos",
            "rope_sin",
            "has_previous",
            "rms_eps",
        ],
        output_names=["records"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _indexer_finalize_kernel():
    header = _FP8_HEADER + _REDUCTION_HEADER
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint out_row = threadgroup_position_in_grid.x;
        uint current_window = out_row;

        threadgroup float normed[128];
        threadgroup float simd_sums[4];
        threadgroup float rrms_shared;
        threadgroup float scale_shared;

        const device T* current_kv = kv_windows
            + size_t(current_window) * 4u * 256u;
        const device T* current_score = score_windows
            + size_t(current_window) * 4u * 256u;

        float max_score = -INFINITY;
        if (has_previous || current_window > 0u) {
            const device T* previous_score = has_previous && current_window == 0u
                ? previous_score_window
                : score_windows + size_t(current_window - 1u) * 4u * 256u;
            for (uint slot = 0u; slot < 4u; ++slot) {
                max_score = max(
                    max_score,
                    float(previous_score[size_t(slot) * 256u + tid])
                );
            }
        }
        for (uint slot = 0u; slot < 4u; ++slot) {
            max_score = max(
                max_score,
                float(current_score[size_t(slot) * 256u + 128u + tid])
            );
        }

        float denominator = 0.0f;
        float numerator = 0.0f;
        if (has_previous || current_window > 0u) {
            const device T* previous_kv = has_previous && current_window == 0u
                ? previous_kv_window
                : kv_windows + size_t(current_window - 1u) * 4u * 256u;
            const device T* previous_score = has_previous && current_window == 0u
                ? previous_score_window
                : score_windows + size_t(current_window - 1u) * 4u * 256u;
            for (uint slot = 0u; slot < 4u; ++slot) {
                float probability = exp(
                    float(previous_score[size_t(slot) * 256u + tid]) - max_score
                );
                denominator += probability;
                numerator += probability
                    * float(previous_kv[size_t(slot) * 256u + tid]);
            }
        }
        for (uint slot = 0u; slot < 4u; ++slot) {
            size_t offset = size_t(slot) * 256u + 128u + tid;
            float probability = exp(float(current_score[offset]) - max_score);
            denominator += probability;
            numerator += probability * float(current_kv[offset]);
        }

        float pooled = numerator / denominator;
        float simd_sq = simd_sum(pooled * pooled);
        if (lane == 0u) {
            simd_sums[sg] = simd_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total = simd_sums[0] + simd_sums[1]
                + simd_sums[2] + simd_sums[3];
            rrms_shared = rsqrt(total / 128.0f + float(rms_eps));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float normalized = pooled * rrms_shared * float(norm_weight[tid]);
        uint pair = (tid >= 64u ? tid - 64u : 0u) / 2u;
        normed[tid] = normalized;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= 64u) {
            uint even_dim = 64u + pair * 2u;
            float even = normed[even_dim];
            float odd = normed[even_dim + 1u];
            float c = float(rope_cos[size_t(out_row) * 32u + pair]);
            float s = float(rope_sin[size_t(out_row) * 32u + pair]);
            normalized = (tid & 1u) == 0u
                ? even * c - odd * s
                : odd * c + even * s;
        }
        normalized = mtplx_bf16_roundtrip(normalized);
        normed[tid] = normalized;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = abs(normed[tid]);
        float simd_maximum = simd_max(local_max);
        if (lane == 0u) {
            simd_sums[sg] = simd_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float amax = max(max(simd_sums[0], simd_sums[1]),
                max(simd_sums[2], simd_sums[3]));
            scale_shared = exp2(ceil(log2(max(amax, 1.0e-4f) / 448.0f)));
            uint bits = as_type<uint>(scale_shared);
            device uchar* record = records + size_t(out_row) * 132u;
            record[128u] = uchar(bits & 0xffu);
            record[129u] = uchar((bits >> 8u) & 0xffu);
            record[130u] = uchar((bits >> 16u) & 0xffu);
            record[131u] = uchar(bits >> 24u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        records[size_t(out_row) * 132u + tid] = mtplx_indexer_e4m3_encode(
            normed[tid] / scale_shared
        );
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_compress_indexer_r4",
        input_names=[
            "kv_windows",
            "score_windows",
            "previous_kv_window",
            "previous_score_window",
            "norm_weight",
            "rope_cos",
            "rope_sin",
            "has_previous",
            "rms_eps",
        ],
        output_names=["records"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def install_nvfp4_record_kernel(*, compress_ratio: int):
    """Construct and return the fixed-ratio stock432 finalizer once."""
    ratio = int(compress_ratio)
    if ratio not in (4, 128):
        raise ValueError(f"unsupported Mia NVFP4 compressor ratio {ratio}")
    return _nvfp4_finalize_kernel(ratio)


def install_indexer_record_kernel():
    """Construct and return the fixed ratio-4 Mia132 finalizer once."""
    return _indexer_finalize_kernel()


def fused_nvfp4_records(
    kv_windows: mx.array,
    score_windows: mx.array,
    previous_kv_window: mx.array,
    previous_score_window: mx.array,
    norm_weight: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    *,
    kernel,
    has_previous: bool,
    output_rows: int,
    rms_eps: float,
) -> mx.array:
    """Return exact stock432 records from projected compressor windows."""
    rows = int(output_rows)
    if rows == 0:
        return mx.zeros((1, 0, MIA_NVFP4_RECORD_BYTES), dtype=mx.uint8)
    records = kernel(
        inputs=[
            mx.contiguous(kv_windows),
            mx.contiguous(score_windows),
            mx.contiguous(previous_kv_window),
            mx.contiguous(previous_score_window),
            mx.contiguous(norm_weight),
            mx.contiguous(rope_cos),
            mx.contiguous(rope_sin),
            bool(has_previous),
            float(rms_eps),
        ],
        template=[("T", kv_windows.dtype)],
        grid=(rows * 512, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[(1, rows, MIA_NVFP4_RECORD_BYTES)],
        output_dtypes=[mx.uint8],
    )[0]
    return records


def fused_indexer_records(
    kv_windows: mx.array,
    score_windows: mx.array,
    previous_kv_window: mx.array,
    previous_score_window: mx.array,
    norm_weight: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    *,
    kernel,
    has_previous: bool,
    output_rows: int,
    rms_eps: float,
) -> mx.array:
    """Return Mia132 FP8 records from projected ratio-4 indexer windows."""
    rows = int(output_rows)
    if rows == 0:
        return mx.zeros((1, 0, INDEXER_RECORD_BYTES), dtype=mx.uint8)
    records = kernel(
        inputs=[
            mx.contiguous(kv_windows),
            mx.contiguous(score_windows),
            mx.contiguous(previous_kv_window),
            mx.contiguous(previous_score_window),
            mx.contiguous(norm_weight),
            mx.contiguous(rope_cos),
            mx.contiguous(rope_sin),
            bool(has_previous),
            float(rms_eps),
        ],
        template=[("T", kv_windows.dtype)],
        grid=(rows * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, rows, INDEXER_RECORD_BYTES)],
        output_dtypes=[mx.uint8],
    )[0]
    return records
