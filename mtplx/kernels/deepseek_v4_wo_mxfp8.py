"""Construction-bound TP1 B12X WO projection for the Mia DeepSeek V4 lane.

The pinned source keeps the inverse-RoPE attention output in MXFP8 scratch. It
crosses one BF16 boundary after grouped WO-A except for Spark's exact M16 route,
which quantizes the FP32 WO-A accumulators directly for WO-B. This module owns
exactly that H64/D512/G8/R1024/H4096 contract. It does not construct a BF16
``[M, 64 * 512]`` inverse-RoPE tensor and it has no stock or eligibility
fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import mlx.core as mx


_HEADS = 64
_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_GROUPS = 8
_HEADS_PER_GROUP = 8
_GROUP_WIDTH = _HEADS_PER_GROUP * _HEAD_DIM
_RANK = 1024
_HIDDEN = 4096
_WO_B_K = _GROUPS * _RANK
_SCALE_GROUP = 32
_BN = 32
_BK = 32
_WO_B_DECODE_BM = 8
_WO_B_DECODE_BN = 64
_WO_B_DECODE_BK = 32


def mia_e4m3_encode_byte(value: float) -> int:
    """Scalar byte oracle for the source's saturating E4M3FN cast."""

    value = float(value)
    sign = 0x80 if math.copysign(1.0, value) < 0.0 else 0
    magnitude = min(abs(value), 448.0)
    if not magnitude > 0.0:
        code = 0
    elif magnitude < 0.015625:
        mantissa = round(magnitude / 0.001953125)
        code = 0x08 if mantissa >= 8 else int(mantissa)
    else:
        exponent = math.floor(math.log2(magnitude))
        step = 2.0 ** (exponent - 3)
        significand = round(magnitude / step)
        if significand >= 16:
            exponent += 1
            significand = 8
        stored_exponent = exponent + 7
        if stored_exponent >= 15:
            stored_exponent = 15
            significand = min(significand, 14)
        code = (stored_exponent << 3) | (significand - 8)
    return int(code | sign)


def mia_e4m3_decode_byte(raw: int) -> float:
    """Decode one E4M3FN byte without changing its stored representation."""

    raw = int(raw) & 0xFF
    exponent = (raw >> 3) & 0x0F
    mantissa = raw & 0x07
    magnitude = (
        mantissa * 0.001953125
        if exponent == 0
        else (1.0 + mantissa * 0.125) * (2.0 ** (exponent - 7))
    )
    return -magnitude if raw & 0x80 else magnitude


def mia_ceil_ue8m0_scale_byte(max_abs: float) -> int:
    """Return the source's ceil-power-of-two UE8M0 scale byte."""

    max_abs = float(max_abs)
    quant_scale = max_abs / 448.0 if max_abs > 0.0 else 1.0
    exponent = max(-127, min(127, math.ceil(math.log2(quant_scale))))
    return int(exponent + 127)


_MXFP8_HEADER = r"""
    using namespace metal;

    inline uchar mia_e4m3_encode(float value) {
        uint sign = as_type<uint>(value) >> 31;
        float magnitude = min(abs(value), 448.0f);
        uchar code;
        constexpr float MIN_NORMAL = 0.015625f;
        constexpr float SUB_STEP = 0.001953125f;
        if (!(magnitude > 0.0f)) {
            code = uchar(0);
        } else if (magnitude < MIN_NORMAL) {
            uint mantissa = uint(rint(magnitude / SUB_STEP));
            code = mantissa >= 8u ? uchar(0x08) : uchar(mantissa);
        } else {
            int exponent = int(floor(log2(magnitude)));
            float step = exp2(float(exponent - 3));
            uint significand = uint(rint(magnitude / step));
            if (significand >= 16u) {
                exponent += 1;
                significand = 8u;
            }
            uint stored_exponent = uint(exponent + 7);
            if (stored_exponent >= 15u) {
                stored_exponent = 15u;
                significand = min(significand, 14u);
            }
            code = uchar((stored_exponent << 3) | (significand - 8u));
        }
        return uchar(code | uchar(sign << 7));
    }

    inline float mia_e4m3_decode(uchar raw) {
        uint exponent = (uint(raw) >> 3) & 0x0fu;
        uint mantissa = uint(raw) & 0x07u;
        float magnitude = exponent == 0u
            ? float(mantissa) * 0.001953125f
            : (1.0f + float(mantissa) * 0.125f)
                * exp2(float(int(exponent) - 7));
        return (uint(raw) & 0x80u) != 0u ? -magnitude : magnitude;
    }

    inline int mia_ceil_ue8m0_exponent(float max_abs) {
        float quant_scale = max_abs > 0.0f ? max_abs / 448.0f : 1.0f;
        return clamp(int(ceil(log2(quant_scale))), -127, 127);
    }

    inline float mia_ue8m0_decode(uchar raw) {
        return exp2(float(int(raw) - 127));
    }
"""


_INVERSE_ROPE_QUANT_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint chunk = threadgroup_position_in_grid.y;
    uint group_row = threadgroup_position_in_grid.z;
    uint group = group_row % 8u;
    uint row = group_row / 8u;
    uint local_dim = chunk * 32u + lane;
    uint head_in_group = local_dim / 512u;
    uint head_dim = local_dim - head_in_group * 512u;
    uint head = group * 8u + head_in_group;

    size_t input_offset = (size_t(row) * 64u + head) * 512u + head_dim;
    float value = float(o[input_offset]);
    if (head_dim >= 448u) {
        uint rope_local = head_dim - 448u;
        uint pair = rope_local >> 1;
        uint partner_dim = 448u + (rope_local ^ 1u);
        float partner = float(
            o[(size_t(row) * 64u + head) * 512u + partner_dim]
        );
        float c = float(cos[size_t(row) * 32u + pair]);
        float s = float(sin[size_t(row) * 32u + pair]);
        value = (rope_local & 1u) == 0u
            ? value * c + partner * s
            : value * c - partner * s;
    }

    float max_abs = simd_max(abs(value));
    threadgroup float scale_shared;
    threadgroup uchar scale_byte_shared;
    if (lane == 0u) {
        int exponent = mia_ceil_ue8m0_exponent(max_abs);
        scale_shared = exp2(float(exponent));
        scale_byte_shared = uchar(exponent + 127);
        scales[(size_t(group) * uint(rows) + row) * 128u + chunk] =
            scale_byte_shared;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    quantized[(size_t(group) * uint(rows) + row) * 4096u + local_dim] =
        mia_e4m3_encode(value / scale_shared);
"""


_GROUP_MAJOR_QUANT_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint chunk = threadgroup_position_in_grid.y;
    uint row = threadgroup_position_in_grid.z;
    uint k = chunk * 32u + lane;
    float value = float(tmp[size_t(row) * 8192u + k]);
    float max_abs = simd_max(abs(value));
    threadgroup float scale_shared;
    if (lane == 0u) {
        int exponent = mia_ceil_ue8m0_exponent(max_abs);
        scale_shared = exp2(float(exponent));
        scales[size_t(row) * 256u + chunk] = uchar(exponent + 127);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    quantized[size_t(row) * 8192u + k] =
        mia_e4m3_encode(value / scale_shared);
"""


_WO_A_M16_QUANTIZED_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint group = threadgroup_position_in_grid.z;
    uint n0 = threadgroup_position_in_grid.y * 32u;
    uint sg_m = sg / 2u;
    uint sg_n = (sg & 1u) * 16u;

    threadgroup T a_tile[16u * 32u];
    threadgroup T b_tile[32u * 32u];
    threadgroup float c_tile[16u * 32u];
    threadgroup float scale_shared[16u];
    simdgroup_matrix<T, 8, 8> a, b_left, b_right;
    simdgroup_matrix<float, 8, 8> c_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_right =
        simdgroup_matrix<float, 8, 8>(0.0f);

    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight);
    for (uint k0 = 0u; k0 < 4096u; k0 += 32u) {
        for (uint index = tid; index < 16u * 32u; index += 128u) {
            uint local_row = index / 32u;
            uint local_k = index - local_row * 32u;
            uint k = k0 + local_k;
            size_t activation_row =
                (size_t(group) * 16u + local_row) * 4096u;
            size_t scale_row =
                (size_t(group) * 16u + local_row) * 128u;
            float scale = mia_ue8m0_decode(
                activation_scales[scale_row + k / 32u]
            );
            a_tile[index] = T(
                mia_e4m3_decode(activations[activation_row + k]) * scale
            );
        }
        for (uint index = tid; index < 32u * 32u; index += 128u) {
            uint local_k = index / 32u;
            uint local_n = index - local_k * 32u;
            uint k = k0 + local_k;
            uint n = n0 + local_n;
            size_t weight_row = (size_t(group) * 1024u + n) * 4096u;
            size_t scale_row = (size_t(group) * 1024u + n) * 128u;
            float scale = mia_ue8m0_decode(
                weight_scales[scale_row + k / 32u]
            );
            b_tile[index] = T(
                mia_e4m3_decode(weight_bytes[weight_row + k]) * scale
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ks = 0u; ks < 32u; ks += 8u) {
            simdgroup_load(a, a_tile + sg_m * 8u * 32u + ks, 32u);
            simdgroup_load(b_left, b_tile + ks * 32u + sg_n, 32u);
            simdgroup_load(b_right, b_tile + ks * 32u + sg_n + 8u, 32u);
            simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
            simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c_left, c_tile + sg_m * 8u * 32u + sg_n, 32u);
    simdgroup_store(c_right, c_tile + sg_m * 8u * 32u + sg_n + 8u, 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 16u) {
        float max_abs = 0.0f;
        for (uint local_n = 0u; local_n < 32u; ++local_n) {
            max_abs = max(max_abs, abs(c_tile[tid * 32u + local_n]));
        }
        int exponent = mia_ceil_ue8m0_exponent(max_abs);
        scale_shared[tid] = exp2(float(exponent));
        scales[
            size_t(tid) * 256u + group * 32u + n0 / 32u
        ] = uchar(exponent + 127);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < 16u * 32u; index += 128u) {
        uint local_row = index / 32u;
        uint local_n = index - local_row * 32u;
        uint n = n0 + local_n;
        quantized[
            size_t(local_row) * 8192u + group * 1024u + n
        ] = mia_e4m3_encode(c_tile[index] / scale_shared[local_row]);
    }
"""


_MXFP8_MMA_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint packed_tile = threadgroup_position_in_grid.z;
    uint group = packed_tile % GROUPS;
    uint row0 = (packed_tile / GROUPS) * BM;
    uint n0 = threadgroup_position_in_grid.y * BN;
    uint sg_m = sg / 2u;
    uint sg_n = (sg & 1u) * 16u;

    threadgroup T a_tile[BM * BK];
    threadgroup T b_tile[BK * BN];
    threadgroup float c_tile[BM * BN];
    simdgroup_matrix<T, 8, 8> a, b_left, b_right;
    simdgroup_matrix<float, 8, 8> c_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_right =
        simdgroup_matrix<float, 8, 8>(0.0f);

    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight);
    for (uint k0 = 0u; k0 < K; k0 += BK) {
        for (uint index = tid; index < BM * BK; index += THREADS) {
            uint local_row = index / BK;
            uint local_k = index - local_row * BK;
            uint row = row0 + local_row;
            uint k = k0 + local_k;
            if (row < uint(rows)) {
                size_t activation_row =
                    (size_t(group) * uint(rows) + row) * K;
                size_t scale_row =
                    (size_t(group) * uint(rows) + row) * (K / 32u);
                float scale = mia_ue8m0_decode(
                    activation_scales[scale_row + k / 32u]
                );
                a_tile[index] = T(
                    mia_e4m3_decode(activations[activation_row + k]) * scale
                );
            } else {
                a_tile[index] = T(0.0f);
            }
        }
        for (uint index = tid; index < BK * BN; index += THREADS) {
            uint local_k = index / BN;
            uint local_n = index - local_k * BN;
            uint k = k0 + local_k;
            uint n = n0 + local_n;
            size_t weight_row = (size_t(group) * N + n) * K;
            size_t scale_row =
                (size_t(group) * N + n) * (K / 32u);
            float scale = mia_ue8m0_decode(
                weight_scales[scale_row + k / 32u]
            );
            b_tile[index] = T(
                mia_e4m3_decode(weight_bytes[weight_row + k]) * scale
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ks = 0u; ks < BK; ks += 8u) {
            simdgroup_load(a, a_tile + sg_m * 8u * BK + ks, BK);
            simdgroup_load(b_left, b_tile + ks * BN + sg_n, BN);
            simdgroup_load(b_right, b_tile + ks * BN + sg_n + 8u, BN);
            simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
            simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(
        c_left, c_tile + sg_m * 8u * BN + sg_n, BN
    );
    simdgroup_store(
        c_right, c_tile + sg_m * 8u * BN + sg_n + 8u, BN
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < BM * BN; index += THREADS) {
        uint local_row = index / BN;
        uint local_n = index - local_row * BN;
        uint row = row0 + local_row;
        uint n = n0 + local_n;
        if (row < uint(rows)) {
            output[(size_t(row) * GROUPS + group) * N + n] =
                T(c_tile[index]);
        }
    }
"""


_WO_B_DECODE_FUSED_QUANT_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint row0 = threadgroup_position_in_grid.z * BM;
    uint n0 = threadgroup_position_in_grid.y * BN;
    uint sg_n = sg * 16u;

    threadgroup T a_tile[BM * BK];
    threadgroup T b_tile[BK * BN];
    threadgroup float c_tile[BM * BN];
    threadgroup float quant_scales[BM];
    simdgroup_matrix<T, 8, 8> a, b_left, b_right;
    simdgroup_matrix<float, 8, 8> c_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_right =
        simdgroup_matrix<float, 8, 8>(0.0f);
    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight);

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        if (tid < BM) {
            uint row = row0 + tid;
            float max_abs = 0.0f;
            if (row < uint(rows)) {
                for (uint local_k = 0u; local_k < BK; ++local_k) {
                    max_abs = max(
                        max_abs,
                        abs(float(tmp[size_t(row) * K + k0 + local_k]))
                    );
                }
            }
            int exponent = mia_ceil_ue8m0_exponent(max_abs);
            quant_scales[tid] = exp2(float(exponent));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < BM * BK; index += THREADS) {
            uint local_row = index / BK;
            uint local_k = index - local_row * BK;
            uint row = row0 + local_row;
            float value = row < uint(rows)
                ? float(tmp[size_t(row) * K + k0 + local_k])
                : 0.0f;
            float scale = quant_scales[local_row];
            uchar quantized = mia_e4m3_encode(value / scale);
            a_tile[index] = T(mia_e4m3_decode(quantized) * scale);
        }
        for (uint index = tid; index < BK * BN; index += THREADS) {
            uint local_k = index / BN;
            uint local_n = index - local_k * BN;
            uint k = k0 + local_k;
            uint n = n0 + local_n;
            float scale = mia_ue8m0_decode(
                weight_scales[size_t(n) * (K / 32u) + k / 32u]
            );
            b_tile[index] = T(
                mia_e4m3_decode(weight_bytes[size_t(n) * K + k]) * scale
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ks = 0u; ks < BK; ks += 8u) {
            simdgroup_load(a, a_tile + ks, BK);
            simdgroup_load(b_left, b_tile + ks * BN + sg_n, BN);
            simdgroup_load(b_right, b_tile + ks * BN + sg_n + 8u, BN);
            simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
            simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c_left, c_tile + sg_n, BN);
    simdgroup_store(c_right, c_tile + sg_n + 8u, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < BM * BN; index += THREADS) {
        uint local_row = index / BN;
        uint local_n = index - local_row * BN;
        uint row = row0 + local_row;
        if (row < uint(rows)) {
            output[size_t(row) * N + n0 + local_n] = T(c_tile[index]);
        }
    }
"""


_WO_B_DECODE_QUANTIZED_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint row0 = threadgroup_position_in_grid.z * BM;
    uint n0 = threadgroup_position_in_grid.y * BN;
    uint sg_n = sg * 16u;

    threadgroup T a_tile[BM * BK];
    threadgroup T b_tile[BK * BN];
    threadgroup float c_tile[BM * BN];
    simdgroup_matrix<T, 8, 8> a, b_left, b_right;
    simdgroup_matrix<float, 8, 8> c_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_right =
        simdgroup_matrix<float, 8, 8>(0.0f);
    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight);

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        for (uint index = tid; index < BM * BK; index += THREADS) {
            uint local_row = index / BK;
            uint local_k = index - local_row * BK;
            uint row = row0 + local_row;
            uint k = k0 + local_k;
            if (row < uint(rows)) {
                float scale = mia_ue8m0_decode(
                    activation_scales[size_t(row) * (K / 32u) + k / 32u]
                );
                a_tile[index] = T(
                    mia_e4m3_decode(activations[size_t(row) * K + k]) * scale
                );
            } else {
                a_tile[index] = T(0.0f);
            }
        }
        for (uint index = tid; index < BK * BN; index += THREADS) {
            uint local_k = index / BN;
            uint local_n = index - local_k * BN;
            uint k = k0 + local_k;
            uint n = n0 + local_n;
            float scale = mia_ue8m0_decode(
                weight_scales[size_t(n) * (K / 32u) + k / 32u]
            );
            b_tile[index] = T(
                mia_e4m3_decode(weight_bytes[size_t(n) * K + k]) * scale
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ks = 0u; ks < BK; ks += 8u) {
            simdgroup_load(a, a_tile + ks, BK);
            simdgroup_load(b_left, b_tile + ks * BN + sg_n, BN);
            simdgroup_load(b_right, b_tile + ks * BN + sg_n + 8u, BN);
            simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
            simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c_left, c_tile + sg_n, BN);
    simdgroup_store(c_right, c_tile + sg_n + 8u, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < BM * BN; index += THREADS) {
        uint local_row = index / BN;
        uint local_n = index - local_row * BN;
        uint row = row0 + local_row;
        if (row < uint(rows)) {
            output[size_t(row) * N + n0 + local_n] = T(c_tile[index]);
        }
    }
"""


@lru_cache(maxsize=1)
def _inverse_rope_quant_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_wo_inv_rope_mxfp8_tp1",
        input_names=["o", "cos", "sin", "rows"],
        output_names=["quantized", "scales"],
        header=_MXFP8_HEADER,
        source=_INVERSE_ROPE_QUANT_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _group_major_quant_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_wo_b_prefill_quant_tp1",
        input_names=["tmp", "rows"],
        output_names=["quantized", "scales"],
        header=_MXFP8_HEADER,
        source=_GROUP_MAJOR_QUANT_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _wo_a_m16_quantized_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_wo_a_mxfp8_quantized_m16_tp1",
        input_names=[
            "activations",
            "activation_scales",
            "weight",
            "weight_scales",
        ],
        output_names=["quantized", "scales"],
        header=_MXFP8_HEADER,
        source=_WO_A_M16_QUANTIZED_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _mxfp8_mma_kernel(stage: str, block_m: int):
    stage = str(stage)
    block_m = int(block_m)
    if (stage, block_m) not in {
        ("wo_a", 8),
        ("wo_a", 64),
        ("wo_b", 64),
    }:
        raise ValueError(f"unsupported Mia TP1 WO MMA route: {(stage, block_m)!r}")
    if stage == "wo_a":
        size_k, size_n, groups = _GROUP_WIDTH, _RANK, _GROUPS
    else:
        size_k, size_n, groups = _WO_B_K, _HIDDEN, 1
    threads = block_m * 2 * 4
    header = _MXFP8_HEADER + f"""
        constant constexpr uint K = {size_k}u;
        constant constexpr uint N = {size_n}u;
        constant constexpr uint GROUPS = {groups}u;
        constant constexpr uint BM = {block_m}u;
        constant constexpr uint BN = {_BN}u;
        constant constexpr uint BK = {_BK}u;
        constant constexpr uint THREADS = {threads}u;
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_mia_{stage}_mxfp8_tp1_bm{block_m}",
        input_names=[
            "activations",
            "activation_scales",
            "weight",
            "weight_scales",
            "rows",
        ],
        output_names=["output"],
        header=header,
        source=_MXFP8_MMA_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=2)
def _wo_b_decode_fused_quant_kernel(block_n: int):
    block_n = int(block_n)
    if block_n not in {32, 64}:
        raise ValueError(f"unsupported Mia TP1 WO-B decode BN: {block_n}")
    threads = (block_n // 16) * 32
    header = _MXFP8_HEADER + f"""
        constant constexpr uint K = {_WO_B_K}u;
        constant constexpr uint N = {_HIDDEN}u;
        constant constexpr uint BM = {_WO_B_DECODE_BM}u;
        constant constexpr uint BN = {block_n}u;
        constant constexpr uint BK = {_WO_B_DECODE_BK}u;
        constant constexpr uint THREADS = {threads}u;
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_mia_wo_b_mxfp8_decode_bm8_bn{block_n}_tp1",
        input_names=["tmp", "weight", "weight_scales", "rows"],
        output_names=["output"],
        header=header,
        source=_WO_B_DECODE_FUSED_QUANT_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=2)
def _wo_b_decode_quantized_kernel(block_n: int):
    block_n = int(block_n)
    if block_n not in {32, 64}:
        raise ValueError(f"unsupported Mia TP1 quantized WO-B decode BN: {block_n}")
    threads = (block_n // 16) * 32
    header = _MXFP8_HEADER + f"""
        constant constexpr uint K = {_WO_B_K}u;
        constant constexpr uint N = {_HIDDEN}u;
        constant constexpr uint BM = {_WO_B_DECODE_BM}u;
        constant constexpr uint BN = {block_n}u;
        constant constexpr uint BK = {_WO_B_DECODE_BK}u;
        constant constexpr uint THREADS = {threads}u;
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_mia_wo_b_mxfp8_quantized_bm8_bn{block_n}_tp1",
        input_names=[
            "activations",
            "activation_scales",
            "weight",
            "weight_scales",
            "rows",
        ],
        output_names=["output"],
        header=header,
        source=_WO_B_DECODE_QUANTIZED_SOURCE,
        ensure_row_contiguous=True,
    )


def _exact_quantized_mxfp8_wo_b(weight, scales):
    """Bind the exact standalone-quantized decode WO-B implementation."""

    kernel = _wo_b_decode_quantized_kernel(_WO_B_DECODE_BN)
    threads = (_WO_B_DECODE_BN // 16) * 32

    def run(values, activation_scales):
        rows = int(values.shape[0])
        return kernel(
            inputs=[values, activation_scales, weight, scales, rows],
            template=[("T", mx.bfloat16)],
            grid=(
                threads,
                _HIDDEN // _WO_B_DECODE_BN,
                (rows + _WO_B_DECODE_BM - 1) // _WO_B_DECODE_BM,
            ),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, _HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )[0]

    return run


def _native_quantized_mxfp8_wo_b(weight, scales):
    """Bind the target-only native MXFP8 decode WO-B implementation."""

    def run(values, activation_scales):
        reconstructed = mx.dequantize(
            values.view(mx.uint32),
            activation_scales,
            biases=None,
            group_size=_SCALE_GROUP,
            bits=8,
            mode="mxfp8",
        ).astype(mx.bfloat16)
        return mx.quantized_matmul(
            reconstructed,
            weight,
            scales=scales,
            biases=None,
            transpose=True,
            group_size=_SCALE_GROUP,
            bits=8,
            mode="mxfp8",
        )

    return run


def _require_array(name: str, value, shape: tuple[int, ...], dtype) -> None:
    observed_shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    observed_dtype = getattr(value, "dtype", None)
    if observed_shape != shape or observed_dtype != dtype:
        raise ValueError(
            f"Mia TP1 WO {name} must be {shape}/{dtype}; "
            f"got {observed_shape}/{observed_dtype}"
        )


@dataclass(frozen=True, slots=True)
class MiaTP1WOMXFP8Plan:
    """Prebound H64/G8 TP1 inverse-RoPE and two-stage MXFP8 projection."""

    wo_a_weight: object
    wo_a_scales: object
    wo_b_weight: object
    wo_b_scales: object
    owner_role: str
    max_prefill_rows: int
    inverse_rope_quant: object
    group_major_quant: object
    wo_a_m16_quantized: object
    wo_a_bm8: object
    wo_a_bm64: object
    wo_b_decode: object
    wo_b_bm64: object

    def __call__(self, o, cos, sin):
        prefix = tuple(int(dim) for dim in o.shape[:-2])
        rows = math.prod(prefix)
        quantized, activation_scales = self.inverse_rope_quant(
            inputs=[o, cos, sin, rows],
            template=[("T", mx.bfloat16)],
            grid=(32, _GROUP_WIDTH // _SCALE_GROUP, rows * _GROUPS),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (_GROUPS, rows, _GROUP_WIDTH),
                (_GROUPS, rows, _GROUP_WIDTH // _SCALE_GROUP),
            ],
            output_dtypes=[mx.uint8, mx.uint8],
        )
        if rows == 16:
            tmp_quantized, tmp_scales = self.wo_a_m16_quantized(
                inputs=[
                    quantized,
                    activation_scales,
                    self.wo_a_weight,
                    self.wo_a_scales,
                ],
                template=[("T", mx.bfloat16)],
                grid=(128, _RANK // _BN, _GROUPS),
                threadgroup=(128, 1, 1),
                output_shapes=[
                    (rows, _WO_B_K),
                    (rows, _WO_B_K // _SCALE_GROUP),
                ],
                output_dtypes=[mx.uint8, mx.uint8],
            )
        else:
            wo_a = self.wo_a_bm8 if rows <= 8 else self.wo_a_bm64
            block_m = 8 if rows <= 8 else 64
            threads = block_m * 8
            (tmp,) = wo_a(
                inputs=[
                    quantized,
                    activation_scales,
                    self.wo_a_weight,
                    self.wo_a_scales,
                    rows,
                ],
                template=[("T", mx.bfloat16)],
                grid=(
                    threads,
                    _RANK // _BN,
                    _GROUPS * ((rows + block_m - 1) // block_m),
                ),
                threadgroup=(threads, 1, 1),
                output_shapes=[(rows, _GROUPS, _RANK)],
                output_dtypes=[mx.bfloat16],
            )
        if rows <= 8:
            tmp_quantized, tmp_scales = self.group_major_quant(
                inputs=[tmp, rows],
                template=[("T", mx.bfloat16)],
                grid=(32, _WO_B_K // _SCALE_GROUP, rows),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    (rows, _WO_B_K),
                    (rows, _WO_B_K // _SCALE_GROUP),
                ],
                output_dtypes=[mx.uint8, mx.uint8],
            )
            output = self.wo_b_decode(tmp_quantized, tmp_scales)
        elif rows != 16:
            tmp_quantized, tmp_scales = self.group_major_quant(
                inputs=[tmp, rows],
                template=[("T", mx.bfloat16)],
                grid=(32, _WO_B_K // _SCALE_GROUP, rows),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    (rows, _WO_B_K),
                    (rows, _WO_B_K // _SCALE_GROUP),
                ],
                output_dtypes=[mx.uint8, mx.uint8],
            )
        if rows > 8:
            threads = 64 * 8
            (output,) = self.wo_b_bm64(
                inputs=[
                    tmp_quantized,
                    tmp_scales,
                    self.wo_b_weight,
                    self.wo_b_scales,
                    rows,
                ],
                template=[("T", mx.bfloat16)],
                grid=(threads, _HIDDEN // _BN, (rows + 63) // 64),
                threadgroup=(threads, 1, 1),
                output_shapes=[(rows, _HIDDEN)],
                output_dtypes=[mx.bfloat16],
            )
        return output.reshape(*prefix, _HIDDEN)


def install_mia_tp1_wo_mxfp8(
    *,
    wo_a_weight,
    wo_a_scales,
    wo_b_weight,
    wo_b_scales,
    owner_role: str,
    max_prefill_rows: int,
) -> MiaTP1WOMXFP8Plan:
    """Validate native MXFP8 storage once and bind the finite TP1 routes."""

    _require_array(
        "wo_a_weight",
        wo_a_weight,
        (_GROUPS * _RANK, _GROUP_WIDTH // 4),
        mx.uint32,
    )
    _require_array(
        "wo_a_scales",
        wo_a_scales,
        (_GROUPS * _RANK, _GROUP_WIDTH // _SCALE_GROUP),
        mx.uint8,
    )
    _require_array(
        "wo_b_weight",
        wo_b_weight,
        (_HIDDEN, _WO_B_K // 4),
        mx.uint32,
    )
    _require_array(
        "wo_b_scales",
        wo_b_scales,
        (_HIDDEN, _WO_B_K // _SCALE_GROUP),
        mx.uint8,
    )
    max_prefill_rows = int(max_prefill_rows)
    if max_prefill_rows <= 8:
        raise ValueError("Mia TP1 WO max_prefill_rows must exceed the M1-M8 decode band")
    owner_role = str(owner_role)
    if owner_role not in {"target", "draft"}:
        raise ValueError(f"unsupported Mia TP1 WO owner role: {owner_role!r}")
    group_major_quant = _group_major_quant_kernel()
    decode_wo_b = (
        _native_quantized_mxfp8_wo_b(wo_b_weight, wo_b_scales)
        if owner_role == "target"
        else _exact_quantized_mxfp8_wo_b(wo_b_weight, wo_b_scales)
    )
    return MiaTP1WOMXFP8Plan(
        wo_a_weight=wo_a_weight,
        wo_a_scales=wo_a_scales,
        wo_b_weight=wo_b_weight,
        wo_b_scales=wo_b_scales,
        owner_role=owner_role,
        max_prefill_rows=max_prefill_rows,
        inverse_rope_quant=_inverse_rope_quant_kernel(),
        group_major_quant=group_major_quant,
        wo_a_m16_quantized=_wo_a_m16_quantized_kernel(),
        wo_a_bm8=_mxfp8_mma_kernel("wo_a", 8),
        wo_a_bm64=_mxfp8_mma_kernel("wo_a", 64),
        wo_b_decode=decode_wo_b,
        wo_b_bm64=_mxfp8_mma_kernel("wo_b", 64),
    )


__all__ = [
    "MiaTP1WOMXFP8Plan",
    "install_mia_tp1_wo_mxfp8",
    "mia_ceil_ue8m0_scale_byte",
    "mia_e4m3_decode_byte",
    "mia_e4m3_encode_byte",
]
