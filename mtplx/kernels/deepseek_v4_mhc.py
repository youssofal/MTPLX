"""SparkInfer-style carried mHC execution for DeepSeek V4 on Metal.

The installed route keeps the source state machine: one initial pre, fused
post-pre boundaries with fused RMSNorm, one final post, and a fused head
collapse to the source BF16 boundary.  The model-owned final RMSNorm remains a
separate operation after that boundary, matching the pinned DSpark/target
execution order.  Block projection and residual Gram partials are produced
together so their normalizers never launch a second hidden-width reduction.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_HIDDEN = 4096
_HC = 4
_MIXES = 24
_SOURCE_TILE = 128
_SOURCE_SPLITS = _HIDDEN // _SOURCE_TILE
_PARTIALS = 1 + _MIXES + 10
_HEAD_PARTIALS = 1 + _HC
_PREFILL_MIN_ROWS = 384
_PREFILL_BLOCK_M = 64
_PREFILL_THREADS = 256
_PREFILL_SIMDGROUPS = _PREFILL_THREADS // 32
_PREFILL_STATS = 11


class _MiaMHCWeightBinding:
    """Plain owner so derived BF16 views never become model parameters."""

    __slots__ = ("fn_bf16", "fn_broadcast")

    def __init__(self, fn_bf16: mx.array, fn_broadcast: mx.array | None = None):
        self.fn_bf16 = fn_bf16
        self.fn_broadcast = fn_broadcast

_MHC_HEADER = rf"""
    using namespace metal;
    constant constexpr uint MTPLX_HIDDEN = {_HIDDEN}u;
    constant constexpr uint MTPLX_HC = {_HC}u;
    constant constexpr uint MTPLX_MIXES = {_MIXES}u;
    constant constexpr uint MTPLX_SOURCE_TILE = {_SOURCE_TILE}u;
    constant constexpr uint MTPLX_SOURCE_SPLITS = {_SOURCE_SPLITS}u;
    constant constexpr uint MTPLX_PARTIALS = {_PARTIALS}u;
    constant constexpr float MTPLX_HC_EPS = 1.0e-6f;

    inline void mtplx_store_partials(
        device float* partials,
        uint row,
        uint split,
        uint lane,
        thread float* sums,
        thread float* gram
    ) {{
        size_t base = (size_t(row) * MTPLX_SOURCE_SPLITS + split)
            * MTPLX_PARTIALS;
        float value = simd_sum(sums[0]);
        if (lane == 0u) partials[base] = value;
        for (uint mix = 0u; mix < MTPLX_MIXES; ++mix) {{
            value = simd_sum(sums[mix + 1u]);
            if (lane == 0u) partials[base + 1u + mix] = value;
        }}
        for (uint pair = 0u; pair < 10u; ++pair) {{
            value = simd_sum(gram[pair]);
            if (lane == 0u) partials[base + 25u + pair] = value;
        }}
    }}

    inline float mtplx_sigmoid(float value) {{
        return 1.0f / (1.0f + exp(-value));
    }}
"""


@lru_cache(maxsize=1)
def _prefill_post_pre_gram_kernel():
    """SparkInfer large-M POST plus compact Gram, with BF16 carried output."""

    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simdgroup = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.z;
        threadgroup float reductions[8u * 11u];
        float local[11];
        for (uint slot = 0u; slot < 11u; ++slot) local[slot] = 0.0f;

        for (uint h = tid; h < MTPLX_HIDDEN; h += 256u) {
            float activation = float(x[size_t(row) * MTPLX_HIDDEN + h]);
            float value[4];
            for (uint dst = 0u; dst < 4u; ++dst) {
                float mixed = float(prev_post[size_t(row) * 4u + dst])
                    * activation;
                for (uint src = 0u; src < 4u; ++src) {
                    mixed += float(
                        prev_comb[(size_t(row) * 4u + src) * 4u + dst]
                    ) * float(
                        prev_residual[(size_t(row) * 4u + src)
                            * MTPLX_HIDDEN + h]
                    );
                }
                T rounded = T(mixed);
                residual[(size_t(row) * 4u + dst) * MTPLX_HIDDEN + h] = rounded;
                value[dst] = float(rounded);
                local[0] += value[dst] * value[dst];
            }
            local[1] += value[0] * value[0];
            local[2] += value[1] * value[1];
            local[3] += value[2] * value[2];
            local[4] += value[3] * value[3];
            local[5] += value[0] * value[1];
            local[6] += value[0] * value[2];
            local[7] += value[0] * value[3];
            local[8] += value[1] * value[2];
            local[9] += value[1] * value[3];
            local[10] += value[2] * value[3];
        }
        for (uint slot = 0u; slot < 11u; ++slot) {
            float value = simd_sum(local[slot]);
            if (lane == 0u) reductions[simdgroup * 11u + slot] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 11u) {
            float total = 0.0f;
            for (uint group = 0u; group < 8u; ++group) {
                total += reductions[group * 11u + tid];
            }
            stats[size_t(row) * 11u + tid] = total;
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_prefill_post_pre_gram_h4096",
        input_names=["x", "prev_residual", "prev_post", "prev_comb"],
        output_names=["residual", "stats"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _prefill_project_kernel():
    """BM64 BF16 simdgroup-matrix projection with FP32 accumulation."""

    header = _MHC_HEADER + f"""
        constant constexpr uint MTPLX_PROJECT_K = 16384u;
        constant constexpr uint MTPLX_PROJECT_N = {_MIXES}u;
        constant constexpr uint MTPLX_PROJECT_BM = 64u;
        constant constexpr uint MTPLX_PROJECT_BN = 32u;
        constant constexpr uint MTPLX_PROJECT_BK = 32u;
        constant constexpr uint MTPLX_PROJECT_THREADS = 512u;
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint simdgroup = simdgroup_index_in_threadgroup;
        uint row0 = threadgroup_position_in_grid.z * MTPLX_PROJECT_BM;
        uint sg_m = simdgroup / 2u;
        uint sg_n = (simdgroup & 1u) * 16u;

        threadgroup T a_tile[MTPLX_PROJECT_BM * MTPLX_PROJECT_BK];
        threadgroup T b_tile[MTPLX_PROJECT_BK * MTPLX_PROJECT_BN];
        threadgroup float c_tile[MTPLX_PROJECT_BM * MTPLX_PROJECT_BN];
        simdgroup_matrix<T, 8, 8> a, b_left, b_right;
        simdgroup_matrix<float, 8, 8> c_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_right =
            simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0u; k0 < MTPLX_PROJECT_K; k0 += MTPLX_PROJECT_BK) {
            for (uint index = tid;
                 index < MTPLX_PROJECT_BM * MTPLX_PROJECT_BK;
                 index += MTPLX_PROJECT_THREADS) {
                uint local_row = index / MTPLX_PROJECT_BK;
                uint local_k = index - local_row * MTPLX_PROJECT_BK;
                uint row = row0 + local_row;
                a_tile[index] = row < uint(rows)
                    ? residual[size_t(row) * MTPLX_PROJECT_K + k0 + local_k]
                    : T(0.0f);
            }
            for (uint index = tid;
                 index < MTPLX_PROJECT_BK * MTPLX_PROJECT_BN;
                 index += MTPLX_PROJECT_THREADS) {
                uint local_k = index / MTPLX_PROJECT_BN;
                uint local_n = index - local_k * MTPLX_PROJECT_BN;
                b_tile[index] = local_n < MTPLX_PROJECT_N
                    ? fn_bf16[size_t(local_n) * MTPLX_PROJECT_K + k0 + local_k]
                    : T(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0u; ks < MTPLX_PROJECT_BK; ks += 8u) {
                simdgroup_load(
                    a,
                    a_tile + sg_m * 8u * MTPLX_PROJECT_BK + ks,
                    MTPLX_PROJECT_BK
                );
                simdgroup_load(
                    b_left,
                    b_tile + ks * MTPLX_PROJECT_BN + sg_n,
                    MTPLX_PROJECT_BN
                );
                simdgroup_load(
                    b_right,
                    b_tile + ks * MTPLX_PROJECT_BN + sg_n + 8u,
                    MTPLX_PROJECT_BN
                );
                simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
                simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(
            c_left,
            c_tile + sg_m * 8u * MTPLX_PROJECT_BN + sg_n,
            MTPLX_PROJECT_BN
        );
        simdgroup_store(
            c_right,
            c_tile + sg_m * 8u * MTPLX_PROJECT_BN + sg_n + 8u,
            MTPLX_PROJECT_BN
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid;
             index < MTPLX_PROJECT_BM * MTPLX_PROJECT_BN;
             index += MTPLX_PROJECT_THREADS) {
            uint local_row = index / MTPLX_PROJECT_BN;
            uint local_n = index - local_row * MTPLX_PROJECT_BN;
            uint row = row0 + local_row;
            if (row < uint(rows) && local_n < MTPLX_PROJECT_N) {
                projected[size_t(row) * MTPLX_PROJECT_N + local_n] = c_tile[index];
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_prefill_bf16_fp32_k16384_n24_bm64",
        input_names=["residual", "fn_bf16", "rows"],
        output_names=["projected"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _prefill_finalize_kernel():
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.z;
        threadgroup float pre[4];
        threadgroup float norm_rms;

        if (tid == 0u) {
            float inv_rms = rsqrt(
                stats[size_t(row) * 11u] / 16384.0f + MTPLX_HC_EPS
            );
            float values[16];
            for (uint stream = 0u; stream < 4u; ++stream) {
                pre[stream] = mtplx_sigmoid(
                    projected[size_t(row) * 24u + stream] * inv_rms
                        * float(scale[0]) + float(base[stream])
                ) + MTPLX_HC_EPS;
                post[size_t(row) * 4u + stream] = 2.0f * mtplx_sigmoid(
                    projected[size_t(row) * 24u + 4u + stream] * inv_rms
                        * float(scale[1]) + float(base[4u + stream])
                );
            }
            for (uint index = 0u; index < 16u; ++index) {
                values[index] = projected[size_t(row) * 24u + 8u + index]
                    * inv_rms * float(scale[2]) + float(base[8u + index]);
            }
            for (uint r = 0u; r < 4u; ++r) {
                float maximum = values[r * 4u];
                for (uint c = 1u; c < 4u; ++c) {
                    maximum = max(maximum, values[r * 4u + c]);
                }
                float denominator = 0.0f;
                for (uint c = 0u; c < 4u; ++c) {
                    values[r * 4u + c] = exp(values[r * 4u + c] - maximum);
                    denominator += values[r * 4u + c];
                }
                for (uint c = 0u; c < 4u; ++c) {
                    values[r * 4u + c] = values[r * 4u + c] / denominator
                        + MTPLX_HC_EPS;
                }
            }
            for (uint c = 0u; c < 4u; ++c) {
                float denominator = MTPLX_HC_EPS;
                for (uint r = 0u; r < 4u; ++r) denominator += values[r * 4u + c];
                for (uint r = 0u; r < 4u; ++r) values[r * 4u + c] /= denominator;
            }
            for (uint iteration = 1u; iteration < 20u; ++iteration) {
                for (uint r = 0u; r < 4u; ++r) {
                    float denominator = MTPLX_HC_EPS;
                    for (uint c = 0u; c < 4u; ++c) denominator += values[r * 4u + c];
                    for (uint c = 0u; c < 4u; ++c) values[r * 4u + c] /= denominator;
                }
                for (uint c = 0u; c < 4u; ++c) {
                    float denominator = MTPLX_HC_EPS;
                    for (uint r = 0u; r < 4u; ++r) denominator += values[r * 4u + c];
                    for (uint r = 0u; r < 4u; ++r) values[r * 4u + c] /= denominator;
                }
            }
            for (uint index = 0u; index < 16u; ++index) {
                comb[size_t(row) * 16u + index] = values[index];
            }

            device const float* gram = stats + size_t(row) * 11u + 1u;
            float sy2 = pre[0] * pre[0] * gram[0]
                + pre[1] * pre[1] * gram[1]
                + pre[2] * pre[2] * gram[2]
                + pre[3] * pre[3] * gram[3]
                + 2.0f * (
                    pre[0] * pre[1] * gram[4]
                    + pre[0] * pre[2] * gram[5]
                    + pre[0] * pre[3] * gram[6]
                    + pre[1] * pre[2] * gram[7]
                    + pre[1] * pre[3] * gram[8]
                    + pre[2] * pre[3] * gram[9]
                );
            norm_rms = rsqrt(sy2 / 4096.0f + float(norm_eps));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = tid; h < MTPLX_HIDDEN; h += 256u) {
            float collapsed = 0.0f;
            for (uint stream = 0u; stream < 4u; ++stream) {
                collapsed += pre[stream] * float(
                    residual[(size_t(row) * 4u + stream) * MTPLX_HIDDEN + h]
                );
            }
            bfloat rounded = bfloat(collapsed);
            y[size_t(row) * MTPLX_HIDDEN + h] = T(
                float(rounded) * norm_rms * float(norm_weight[h])
            );
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_prefill_finalize_gram_norm_h4096",
        input_names=[
            "residual", "stats", "projected", "scale", "base",
            "norm_weight", "norm_eps",
        ],
        output_names=["y", "post", "comb"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _broadcast_partial_kernel():
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint split = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.z;
        uint first_h = split * MTPLX_SOURCE_TILE + lane * 4u;
        float sums[25];
        float gram[10];
        for (uint i = 0u; i < 25u; ++i) sums[i] = 0.0f;
        for (uint i = 0u; i < 10u; ++i) gram[i] = 0.0f;

        for (uint offset = 0u; offset < 4u; ++offset) {
            uint h = first_h + offset;
            float value = float(x[size_t(row) * MTPLX_HIDDEN + h]);
            for (uint stream = 0u; stream < MTPLX_HC; ++stream) {
                residual[(size_t(row) * MTPLX_HC + stream) * MTPLX_HIDDEN + h]
                    = T(value);
            }
            sums[0] += 4.0f * value * value;
            for (uint mix = 0u; mix < MTPLX_MIXES; ++mix) {
                sums[mix + 1u] += value * float(
                    fn_broadcast[size_t(mix) * MTPLX_HIDDEN + h]
                );
            }
            float square = value * value;
            for (uint pair = 0u; pair < 10u; ++pair) gram[pair] += square;
        }
        mtplx_store_partials(partials, row, split, lane, sums, gram);
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_broadcast_partial_h4096",
        input_names=["x", "fn_broadcast"],
        output_names=["residual", "partials"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _state_partial_kernel():
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint split = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.z;
        uint first_h = split * MTPLX_SOURCE_TILE + lane * 4u;
        float sums[25];
        float gram[10];
        for (uint i = 0u; i < 25u; ++i) sums[i] = 0.0f;
        for (uint i = 0u; i < 10u; ++i) gram[i] = 0.0f;

        for (uint offset = 0u; offset < 4u; ++offset) {
            uint h = first_h + offset;
            float value[4];
            for (uint stream = 0u; stream < MTPLX_HC; ++stream) {
                value[stream] = float(
                    residual[(size_t(row) * MTPLX_HC + stream)
                        * MTPLX_HIDDEN + h]
                );
                sums[0] += value[stream] * value[stream];
            }
            for (uint mix = 0u; mix < MTPLX_MIXES; ++mix) {
                for (uint stream = 0u; stream < MTPLX_HC; ++stream) {
                    sums[mix + 1u] += value[stream] * float(
                        fn[(size_t(mix) * MTPLX_HC + stream)
                            * MTPLX_HIDDEN + h]
                    );
                }
            }
            gram[0] += value[0] * value[0];
            gram[1] += value[1] * value[1];
            gram[2] += value[2] * value[2];
            gram[3] += value[3] * value[3];
            gram[4] += value[0] * value[1];
            gram[5] += value[0] * value[2];
            gram[6] += value[0] * value[3];
            gram[7] += value[1] * value[2];
            gram[8] += value[1] * value[3];
            gram[9] += value[2] * value[3];
        }
        mtplx_store_partials(partials, row, split, lane, sums, gram);
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_state_partial_h4096",
        input_names=["residual", "fn"],
        output_names=["partials"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _post_pre_partial_kernel():
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint split = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.z;
        uint first_h = split * MTPLX_SOURCE_TILE + lane * 4u;
        float sums[25];
        float gram[10];
        for (uint i = 0u; i < 25u; ++i) sums[i] = 0.0f;
        for (uint i = 0u; i < 10u; ++i) gram[i] = 0.0f;

        for (uint offset = 0u; offset < 4u; ++offset) {
            uint h = first_h + offset;
            float activation = float(x[size_t(row) * MTPLX_HIDDEN + h]);
            float value[4];
            for (uint dst = 0u; dst < MTPLX_HC; ++dst) {
                float mixed = float(prev_post[size_t(row) * MTPLX_HC + dst])
                    * activation;
                for (uint src = 0u; src < MTPLX_HC; ++src) {
                    mixed += float(
                        prev_comb[(size_t(row) * MTPLX_HC + src)
                            * MTPLX_HC + dst]
                    ) * float(
                        prev_residual[(size_t(row) * MTPLX_HC + src)
                            * MTPLX_HIDDEN + h]
                    );
                }
                value[dst] = float(T(mixed));
                residual[(size_t(row) * MTPLX_HC + dst) * MTPLX_HIDDEN + h]
                    = T(mixed);
                sums[0] += value[dst] * value[dst];
            }
            for (uint mix = 0u; mix < MTPLX_MIXES; ++mix) {
                for (uint stream = 0u; stream < MTPLX_HC; ++stream) {
                    sums[mix + 1u] += value[stream] * float(
                        fn[(size_t(mix) * MTPLX_HC + stream)
                            * MTPLX_HIDDEN + h]
                    );
                }
            }
            gram[0] += value[0] * value[0];
            gram[1] += value[1] * value[1];
            gram[2] += value[2] * value[2];
            gram[3] += value[3] * value[3];
            gram[4] += value[0] * value[1];
            gram[5] += value[0] * value[2];
            gram[6] += value[0] * value[3];
            gram[7] += value[1] * value[2];
            gram[8] += value[1] * value[3];
            gram[9] += value[2] * value[3];
        }
        mtplx_store_partials(partials, row, split, lane, sums, gram);
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_post_pre_partial_h4096",
        input_names=["x", "prev_residual", "prev_post", "prev_comb", "fn"],
        output_names=["residual", "partials"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _finalize_kernel():
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.z;
        threadgroup float pre[4];
        threadgroup float norm_rms;

        if (tid == 0u) {
            float reduced[35];
            for (uint column = 0u; column < 35u; ++column) {
                float total = 0.0f;
                for (uint split = 0u; split < MTPLX_SOURCE_SPLITS; ++split) {
                    total += partials[
                        (size_t(row) * MTPLX_SOURCE_SPLITS + split)
                            * MTPLX_PARTIALS + column
                    ];
                }
                reduced[column] = total;
            }
            float inv_rms = rsqrt(reduced[0] / 16384.0f + MTPLX_HC_EPS);
            float mixes[24];
            for (uint mix = 0u; mix < 24u; ++mix) {
                mixes[mix] = reduced[mix + 1u] * inv_rms;
            }
            for (uint stream = 0u; stream < 4u; ++stream) {
                pre[stream] = mtplx_sigmoid(
                    mixes[stream] * float(scale[0]) + float(base[stream])
                ) + MTPLX_HC_EPS;
                post[size_t(row) * 4u + stream] = 2.0f * mtplx_sigmoid(
                    mixes[4u + stream] * float(scale[1])
                        + float(base[4u + stream])
                );
            }

            float values[16];
            for (uint i = 0u; i < 16u; ++i) {
                values[i] = mixes[8u + i] * float(scale[2])
                    + float(base[8u + i]);
            }
            for (uint r = 0u; r < 4u; ++r) {
                float maximum = values[r * 4u];
                for (uint c = 1u; c < 4u; ++c) {
                    maximum = max(maximum, values[r * 4u + c]);
                }
                float denominator = 0.0f;
                for (uint c = 0u; c < 4u; ++c) {
                    values[r * 4u + c] = exp(values[r * 4u + c] - maximum);
                    denominator += values[r * 4u + c];
                }
                for (uint c = 0u; c < 4u; ++c) {
                    values[r * 4u + c] = values[r * 4u + c] / denominator
                        + MTPLX_HC_EPS;
                }
            }
            for (uint c = 0u; c < 4u; ++c) {
                float denominator = MTPLX_HC_EPS;
                for (uint r = 0u; r < 4u; ++r) denominator += values[r * 4u + c];
                for (uint r = 0u; r < 4u; ++r) values[r * 4u + c] /= denominator;
            }
            for (uint iteration = 1u; iteration < 20u; ++iteration) {
                for (uint r = 0u; r < 4u; ++r) {
                    float denominator = MTPLX_HC_EPS;
                    for (uint c = 0u; c < 4u; ++c) denominator += values[r * 4u + c];
                    for (uint c = 0u; c < 4u; ++c) values[r * 4u + c] /= denominator;
                }
                for (uint c = 0u; c < 4u; ++c) {
                    float denominator = MTPLX_HC_EPS;
                    for (uint r = 0u; r < 4u; ++r) denominator += values[r * 4u + c];
                    for (uint r = 0u; r < 4u; ++r) values[r * 4u + c] /= denominator;
                }
            }
            for (uint i = 0u; i < 16u; ++i) {
                comb[size_t(row) * 16u + i] = values[i];
            }

            float sy2 = pre[0] * pre[0] * reduced[25]
                + pre[1] * pre[1] * reduced[26]
                + pre[2] * pre[2] * reduced[27]
                + pre[3] * pre[3] * reduced[28]
                + 2.0f * (
                    pre[0] * pre[1] * reduced[29]
                    + pre[0] * pre[2] * reduced[30]
                    + pre[0] * pre[3] * reduced[31]
                    + pre[1] * pre[2] * reduced[32]
                    + pre[1] * pre[3] * reduced[33]
                    + pre[2] * pre[3] * reduced[34]
                );
            norm_rms = rsqrt(sy2 / 4096.0f + float(norm_eps));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint h = tid; h < MTPLX_HIDDEN; h += 256u) {
            float collapsed = 0.0f;
            for (uint stream = 0u; stream < 4u; ++stream) {
                collapsed += pre[stream] * float(
                    residual[(size_t(row) * 4u + stream) * MTPLX_HIDDEN + h]
                );
            }
            bfloat rounded = bfloat(collapsed);
            y[size_t(row) * MTPLX_HIDDEN + h] = T(
                float(rounded) * norm_rms * float(norm_weight[h])
            );
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_finalize_gram_norm_h4096",
        input_names=["residual", "partials", "scale", "base", "norm_weight", "norm_eps"],
        output_names=["y", "post", "comb"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _post_kernel():
    source = r"""
        uint element = thread_position_in_grid.x;
        uint total = uint(rows) * 4u * MTPLX_HIDDEN;
        if (element >= total) return;
        uint row = element / (4u * MTPLX_HIDDEN);
        uint within = element % (4u * MTPLX_HIDDEN);
        uint dst = within / MTPLX_HIDDEN;
        uint h = within % MTPLX_HIDDEN;
        float mixed = float(prev_post[size_t(row) * 4u + dst])
            * float(x[size_t(row) * MTPLX_HIDDEN + h]);
        for (uint src = 0u; src < 4u; ++src) {
            mixed += float(prev_comb[(size_t(row) * 4u + src) * 4u + dst])
                * float(prev_residual[(size_t(row) * 4u + src) * MTPLX_HIDDEN + h]);
        }
        residual[element] = T(mixed);
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_final_post_h4096",
        input_names=["x", "prev_residual", "prev_post", "prev_comb", "rows"],
        output_names=["residual"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _head_partial_kernel():
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint split = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.z;
        uint first_h = split * MTPLX_SOURCE_TILE + lane * 4u;
        float sums[5];
        for (uint i = 0u; i < 5u; ++i) sums[i] = 0.0f;
        for (uint offset = 0u; offset < 4u; ++offset) {
            uint h = first_h + offset;
            float value[4];
            for (uint stream = 0u; stream < 4u; ++stream) {
                value[stream] = float(
                    residual[(size_t(row) * 4u + stream) * MTPLX_HIDDEN + h]
                );
                sums[0] += value[stream] * value[stream];
            }
            for (uint mix = 0u; mix < 4u; ++mix) {
                for (uint stream = 0u; stream < 4u; ++stream) {
                    sums[mix + 1u] += value[stream] * float(
                        fn[(size_t(mix) * 4u + stream) * MTPLX_HIDDEN + h]
                    );
                }
            }
        }
        size_t base_out = (size_t(row) * MTPLX_SOURCE_SPLITS + split) * 5u;
        for (uint column = 0u; column < 5u; ++column) {
            float value = simd_sum(sums[column]);
            if (lane == 0u) partials[base_out + column] = value;
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_head_partial_h4096",
        input_names=["residual", "fn"],
        output_names=["partials"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _head_finalize_kernel():
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.z;
        threadgroup float pre[4];
        if (tid == 0u) {
            float reduced[5];
            for (uint column = 0u; column < 5u; ++column) {
                float total = 0.0f;
                for (uint split = 0u; split < MTPLX_SOURCE_SPLITS; ++split) {
                    total += partials[(size_t(row) * MTPLX_SOURCE_SPLITS + split)
                        * 5u + column];
                }
                reduced[column] = total;
            }
            float inv_rms = rsqrt(reduced[0] / 16384.0f + MTPLX_HC_EPS);
            for (uint stream = 0u; stream < 4u; ++stream) {
                pre[stream] = mtplx_sigmoid(
                    reduced[stream + 1u] * inv_rms * float(scale[0])
                        + float(base[stream])
                ) + MTPLX_HC_EPS;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = tid; h < MTPLX_HIDDEN; h += 256u) {
            float collapsed = 0.0f;
            for (uint stream = 0u; stream < 4u; ++stream) {
                collapsed += pre[stream] * float(
                    residual[(size_t(row) * 4u + stream) * MTPLX_HIDDEN + h]
                );
            }
            y[size_t(row) * MTPLX_HIDDEN + h] = T(collapsed);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mhc_head_bf16_h4096",
        input_names=["residual", "partials", "scale", "base"],
        output_names=["y"],
        header=_MHC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


class MiaMHCPlan:
    """Construction-time owner for the fixed hc=4, hidden=4096 pipeline."""

    def __init__(self, *, max_tokens: int, rms_eps: float, hc_eps: float, iters: int):
        if int(max_tokens) < 1:
            raise ValueError("Mia mHC max_tokens must be positive")
        if (float(rms_eps), float(hc_eps), int(iters)) != (1.0e-6, 1.0e-6, 20):
            raise ValueError(
                "Mia mHC requires rms_eps=hc_eps=1e-6 and 20 Sinkhorn iterations"
            )
        self.max_tokens = int(max_tokens)
        self._broadcast_partial = _broadcast_partial_kernel()
        self._state_partial = _state_partial_kernel()
        self._post_pre_partial = _post_pre_partial_kernel()
        self._finalize = _finalize_kernel()
        self._post = _post_kernel()
        self._head_partial = _head_partial_kernel()
        self._head_finalize = _head_finalize_kernel()
        self.prefill_min_rows = _PREFILL_MIN_ROWS
        self.prefill_block_m = _PREFILL_BLOCK_M
        self._prefill_post_pre_gram = _prefill_post_pre_gram_kernel()
        self._prefill_project = _prefill_project_kernel()
        self._prefill_finalize = _prefill_finalize_kernel()
        self.route_contract = (
            "broadcast_fn_fp32",
            "attention_post_pre_fn_fp32",
            "ffn_tiny_post_pre_fn_bf16_split32_fp32",
            "ffn_prefill_post_pre_fn_bf16_mma_bm64_fp32",
            "compact_gram_finalize",
            "head_bf16_then_rmsnorm",
        )
        self.bound_hyper_connections = 0

    def install_modules(self, *, hyper_connections, broadcast_connection) -> None:
        """Materialize the source ``fn_bf16`` views once after weight loading."""

        hyper_connections = tuple(hyper_connections)
        if not hyper_connections or broadcast_connection is not hyper_connections[0]:
            raise ValueError(
                "Mia mHC broadcast binding must be the first attention connection"
            )
        for index, module in enumerate(hyper_connections):
            if tuple(module.fn.shape) != (_MIXES, _HC * _HIDDEN):
                raise ValueError(
                    f"Mia mHC connection {index} fn shape changed: {module.fn.shape}"
                )
            if module.fn.dtype != mx.float32:
                raise ValueError("Mia mHC routing weights must remain FP32")
        installed = tuple(
            mx.contiguous(module.fn.astype(mx.bfloat16))
            for module in hyper_connections
        )
        fn_broadcast = mx.contiguous(
            mx.sum(
                broadcast_connection.fn.reshape(_MIXES, _HC, _HIDDEN),
                axis=1,
            )
        )
        mx.eval(*installed, fn_broadcast)
        for module, fn_bf16 in zip(hyper_connections, installed, strict=True):
            module._mia_mhc_weight = _MiaMHCWeightBinding(fn_bf16)
        broadcast_connection._mia_mhc_weight.fn_broadcast = fn_broadcast
        self._hyper_connections = hyper_connections
        self.bound_hyper_connections = len(hyper_connections)

    @staticmethod
    def _rows(value: mx.array) -> int:
        return int(value.size) // int(value.shape[-1])

    def _finish(self, residual, partials, hc, norm):
        rows = self._rows(residual) // _HC
        y, post, comb = self._finalize(
            inputs=[
                mx.contiguous(residual),
                partials,
                hc.scale,
                hc.base,
                norm.weight,
                float(norm.eps),
            ],
            template=[("T", residual.dtype)],
            grid=(256, 1, rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, _HIDDEN), (rows, _HC), (rows, _HC, _HC)],
            output_dtypes=[residual.dtype, mx.float32, mx.float32],
        )
        return residual, post, comb, y

    def _project_prefill(self, residual, fn_bf16, rows: int):
        return self._prefill_project(
            inputs=[residual, fn_bf16, rows],
            template=[("T", residual.dtype)],
            grid=(512, 1, (rows + _PREFILL_BLOCK_M - 1) // _PREFILL_BLOCK_M),
            threadgroup=(512, 1, 1),
            output_shapes=[(rows, _MIXES)],
            output_dtypes=[mx.float32],
        )[0]

    def _finish_prefill(self, residual, stats, projected, hc, norm):
        rows = self._rows(residual) // _HC
        y, post, comb = self._prefill_finalize(
            inputs=[
                residual,
                stats,
                projected,
                hc.scale,
                hc.base,
                norm.weight,
                float(norm.eps),
            ],
            template=[("T", residual.dtype)],
            grid=(_PREFILL_THREADS, 1, rows),
            threadgroup=(_PREFILL_THREADS, 1, 1),
            output_shapes=[(rows, _HIDDEN), (rows, _HC), (rows, _HC, _HC)],
            output_dtypes=[residual.dtype, mx.float32, mx.float32],
        )
        return residual, post, comb, y

    def pre_broadcast(self, x: mx.array, hc, norm):
        rows = self._rows(x)
        x = mx.contiguous(x.reshape(rows, _HIDDEN))
        residual, partials = self._broadcast_partial(
            inputs=[x, hc._mia_mhc_weight.fn_broadcast],
            template=[("T", x.dtype)],
            grid=(32, _SOURCE_SPLITS, rows),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (rows, _HC, _HIDDEN),
                (rows, _SOURCE_SPLITS, _PARTIALS),
            ],
            output_dtypes=[x.dtype, mx.float32],
        )
        return self._finish(residual, partials, hc, norm)

    def pre_state(self, residual: mx.array, hc, norm):
        rows = self._rows(residual) // _HC
        residual = mx.contiguous(residual.reshape(rows, _HC, _HIDDEN))
        (partials,) = self._state_partial(
            inputs=[residual, hc.fn],
            grid=(32, _SOURCE_SPLITS, rows),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, _SOURCE_SPLITS, _PARTIALS)],
            output_dtypes=[mx.float32],
        )
        return self._finish(residual, partials, hc, norm)

    def _post_pre_connection(
        self,
        x,
        residual,
        post,
        comb,
        hc,
        norm,
        *,
        projection_weight,
    ):
        rows = self._rows(x)
        x = mx.contiguous(x.reshape(rows, _HIDDEN))
        residual = mx.contiguous(residual.reshape(rows, _HC, _HIDDEN))
        post = mx.contiguous(post.reshape(rows, _HC))
        comb = mx.contiguous(comb.reshape(rows, _HC, _HC))
        residual, partials = self._post_pre_partial(
            inputs=[
                x,
                residual,
                post,
                comb,
                projection_weight,
            ],
            template=[("T", x.dtype)],
            grid=(32, _SOURCE_SPLITS, rows),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (rows, _HC, _HIDDEN),
                (rows, _SOURCE_SPLITS, _PARTIALS),
            ],
            output_dtypes=[x.dtype, mx.float32],
        )
        return self._finish(residual, partials, hc, norm)

    def _post_pre_ffn_prefill(self, x, residual, post, comb, hc, norm):
        rows = self._rows(x)
        x = mx.contiguous(x.reshape(rows, _HIDDEN))
        residual = mx.contiguous(residual.reshape(rows, _HC, _HIDDEN))
        post = mx.contiguous(post.reshape(rows, _HC))
        comb = mx.contiguous(comb.reshape(rows, _HC, _HC))
        residual, stats = self._prefill_post_pre_gram(
            inputs=[x, residual, post, comb],
            template=[("T", x.dtype)],
            grid=(_PREFILL_THREADS, 1, rows),
            threadgroup=(_PREFILL_THREADS, 1, 1),
            output_shapes=[
                (rows, _HC, _HIDDEN),
                (rows, _PREFILL_STATS),
            ],
            output_dtypes=[x.dtype, mx.float32],
        )
        projected = self._project_prefill(
            residual, hc._mia_mhc_weight.fn_bf16, rows
        )
        return self._finish_prefill(residual, stats, projected, hc, norm)

    def post_pre_attn(self, x, residual, post, comb, hc, norm):
        """Run the source attention connection with its installed FP32 projection."""

        return self._post_pre_connection(
            x,
            residual,
            post,
            comb,
            hc,
            norm,
            projection_weight=hc.fn,
        )

    def post_pre_ffn(self, x, residual, post, comb, hc, norm):
        """Run the source FFN connection with its installed BF16 projection."""

        if self._rows(x) >= self.prefill_min_rows:
            return self._post_pre_ffn_prefill(x, residual, post, comb, hc, norm)
        return self._post_pre_connection(
            x,
            residual,
            post,
            comb,
            hc,
            norm,
            projection_weight=hc._mia_mhc_weight.fn_bf16,
        )

    def post(self, x, residual, post, comb):
        rows = self._rows(x)
        elements = rows * _HC * _HIDDEN
        (out,) = self._post(
            inputs=[
                mx.contiguous(x.reshape(rows, _HIDDEN)),
                mx.contiguous(residual.reshape(rows, _HC, _HIDDEN)),
                mx.contiguous(post.reshape(rows, _HC)),
                mx.contiguous(comb.reshape(rows, _HC, _HC)),
                rows,
            ],
            template=[("T", x.dtype)],
            grid=((elements + 255) // 256 * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, _HC, _HIDDEN)],
            output_dtypes=[x.dtype],
        )
        return out

    def head(self, residual, head):
        rows = self._rows(residual) // _HC
        residual = mx.contiguous(residual.reshape(rows, _HC, _HIDDEN))
        (partials,) = self._head_partial(
            inputs=[residual, head.fn],
            grid=(32, _SOURCE_SPLITS, rows),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, _SOURCE_SPLITS, _HEAD_PARTIALS)],
            output_dtypes=[mx.float32],
        )
        (y,) = self._head_finalize(
            inputs=[
                residual,
                partials,
                head.scale,
                head.base,
            ],
            template=[("T", residual.dtype)],
            grid=(256, 1, rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, _HIDDEN)],
            output_dtypes=[residual.dtype],
        )
        return y
