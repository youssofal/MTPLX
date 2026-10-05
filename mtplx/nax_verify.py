"""NAX (Metal 4 tensor-ops) verify-shaped quantized matmul kernels.

Ported from bstnxbt/dflash-mlx `dflash_mlx/verify_qmm.py` (Apache-2.0), which is
based on DFlash (arXiv:2602.06036). Two kernels are kept:

- m16 NAX ktmpl: BM=16 tile via MetalPerformancePrimitives matmul2d
  (Apple G17 / M5-class NAX units, macOS >= 26.2). 4-bit affine weights,
  K % 256 == 0, N % 32 == 0.
- m4 K-split: plain SIMD kernel for exact M=4 rows, 4-bit affine weights,
  K % 32 == 0, N % 4 == 0.

MTPLX additions: multi-row lanes that read each weight tile once for 5 to 32
verify rows (an exact 5-row split-K lane, a pipelined m16 tile and a two-tile
32-row variant; default-off switches, ``MTPLX_MULTIROW_QMM=1`` turns all three
on, see ``m5_rows_lane_enabled``, ``m16_pipelined_enabled`` and
``m32_tile_enabled``), M-padding dispatch (verify rows 2..16 pad to the 16-row NAX
tile; weight streaming dominates so padded rows are nearly free), env gating,
and availability probes. Exactness vs stock mx.quantized_matmul is enforced by
the capture-commit/R1b gates before any product use.
"""

from __future__ import annotations

import os

import mlx.core as mx

_VERIFY_KERNEL_CACHE: dict[tuple, object] = {}


def nax_env_enabled() -> bool:
    return str(os.environ.get("MTPLX_NAX_VERIFY", "")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


# The detector lives in ``mtplx.nax_detect`` (one rule for every lane: GPU
# generation 17 or newer, 18 for the phone class, macOS 26.2 or newer, and
# the MTPLX_FORCE_GPU_FAMILY_FALLBACK rehearsal switch read per call). This
# module used to match the exact prefix ``applegpu_g17``, so a generation-18
# GPU would have lost every lane gated here while the Flash-Next prefill
# gate, which parsed the generation, kept its lanes.
from mtplx.nax_detect import (  # noqa: E402
    nax_available,
    nax_hardware_available as _nax_hardware_available,
)


def _build_kernel_m16_nax_ktmpl(k_val: int, group_size: int, dtype: mx.Dtype):
    key = ("m16_nax_ktmpl", int(k_val), group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 16;
        constexpr int NSG = 8;
        constexpr int GS = {group_size};
        constexpr int K = KCONST;
        constexpr int K_by_8 = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid = thread_position_in_threadgroup.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;
        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup T B_tile[NSG][BK * BN];
        threadgroup float partial[NSG][BM * BN];

        constexpr auto desc = matmul2d_descriptor(
            16,
            32,
            16,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, metal::execution_simdgroup> op;

        tensor<device T, dextents<int, 2>, tensor_inline> A(
            (device T*)x,
            dextents<int, 2>{{K, BM}},
            array<int, 2>{{1, K}});
        tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
            B_tile[sg_id],
            dextents<int, 2>{{BN, BK}},
            array<int, 2>{{1, BN}});
        tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
            partial[sg_id],
            dextents<int, 2>{{BN, BM}},
            array<int, 2>{{1, BN}});

        auto ct_c = op.template get_destination_cooperative_tensor<
            tensor<device T, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
            float>();
        _Pragma("unroll")
        for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {{
            ct_c[i] = 0.0f;
        }}

        int n_global = n0 + int(lane);
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            uint32_t p0 = w_q[n_global * K_by_8 + ((k0 + 0) >> 3)];
            uint32_t p1 = w_q[n_global * K_by_8 + ((k0 + 8) >> 3)];
            float s0 = float(scales[n_global * K_by_gs + ((k0 + 0) / GS)]);
            float s1 = float(scales[n_global * K_by_gs + ((k0 + 8) / GS)]);
            float b0 = float(biases[n_global * K_by_gs + ((k0 + 0) / GS)]);
            float b1 = float(biases[n_global * K_by_gs + ((k0 + 8) / GS)]);

            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                uint32_t nib = (p0 >> (ki * 4)) & 0xFu;
                B_tile[sg_id][ki * BN + int(lane)] = T(float(nib) * s0 + b0);
            }}
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                uint32_t nib = (p1 >> (ki * 4)) & 0xFu;
                B_tile[sg_id][(8 + ki) * BN + int(lane)] = T(float(nib) * s1 + b1);
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);

            auto tA = A.template slice<16, 16>(k0, 0);
            auto tB = B.template slice<32, 16>(0, 0);
            op.run(tA, tB, ct_c);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        auto tC = C.template slice<32, 16>(0, 0);
        ct_c.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc01 = partial[0][off] + partial[1][off];
            float acc23 = partial[2][off] + partial[3][off];
            float acc45 = partial[4][off] + partial[5][off];
            float acc67 = partial[6][off] + partial[7][off];
            float acc = (acc01 + acc23) + (acc45 + acc67);
            int row = off / BN;
            int col = off - row * BN;
            y[row * N + n0 + col] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_verify_m16_nax_k{int(k_val)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        header="""
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        """,
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m16_nax_pipelined(
    k_val: int, group_size: int, dtype: mx.Dtype, row_tiles: int = 1
):
    """m16 NAX tile with a register prefetch and ping-pong B tiles.

    Same arithmetic and summation order as ``_build_kernel_m16_nax_ktmpl`` (the
    output is bit-identical), different schedule. Per simdgroup the K chunk is
    walked in two-step units of 32 K: the weights and scales of the next unit
    are loaded into registers while the current unit runs, and the dequantized
    B tile of step i+1 is written into the second tile right after the matmul
    of step i is issued, so the tile writes no longer wait for the matmul.
    Receipts (2026-10-04, M5 Pro, dependent chain, weights cycled through
    1.5 GB, M=16): gate/up 5120x17408 0.292 -> 0.237 ms, down 17408x5120
    0.321 -> 0.255 ms, 5120x6144 0.115 -> 0.095, 6144x5120 0.114 -> 0.093,
    5120x10240 0.178 -> 0.146, 5120x12288 0.213 -> 0.175 (-17 to -21%). A
    prefetch of one unit (0.265) or four units (0.280, register pressure) is
    slower than two, and NSG=4 or 16 loses to NSG=8. Needs K % 512 == 0 (the
    K chunk of one simdgroup is a whole number of two-step units).

    ``row_tiles=2`` stacks two 16-row tiles (32 rows): the B tile is dequantized
    once and feeds both matmuls, so a 17 to 32 row verify still reads each
    weight once. Stock qmm needs 0.42 ms (gate/up) and 0.98 ms (down) for the
    same rows; the 32-row tile 0.30 and 0.34 ms.
    """
    key = ("m16_nax_pipelined", int(k_val), group_size, dtype, int(row_tiles))
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int NR = {int(row_tiles)};
        constexpr int BM = 16 * NR;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int NSG = 8;
        constexpr int NU = 2;
        constexpr int LU = NU * BK;
        constexpr int GS = {group_size};
        constexpr int K = KCONST;
        constexpr int K_by_8 = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;
        constexpr int NG = (LU / GS) > 0 ? (LU / GS) : 1;

        uint tid = thread_position_in_threadgroup.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;
        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        // Two B tiles per simdgroup; after the loop the same memory holds the
        // eight fp32 partial tiles (16 KB, or all 32 KB at two row tiles).
        threadgroup T B_tile[NSG][2][BK * BN];
        threadgroup float* partial_base = (threadgroup float*)B_tile;

        constexpr auto desc = matmul2d_descriptor(
            16,
            BN,
            BK,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, metal::execution_simdgroup> op;

        tensor<device T, dextents<int, 2>, tensor_inline> A(
            (device T*)x,
            dextents<int, 2>{{K, BM}},
            array<int, 2>{{1, K}});
        tensor<threadgroup T, dextents<int, 2>, tensor_inline> B0(
            B_tile[sg_id][0],
            dextents<int, 2>{{BN, BK}},
            array<int, 2>{{1, BN}});
        tensor<threadgroup T, dextents<int, 2>, tensor_inline> B1(
            B_tile[sg_id][1],
            dextents<int, 2>{{BN, BK}},
            array<int, 2>{{1, BN}});

        auto ct_c = op.template get_destination_cooperative_tensor<
            tensor<device T, extents<int, BK, 16>, tensor_inline>,
            tensor<threadgroup T, extents<int, BN, BK>, tensor_inline>,
            float>();
        auto ct_c1 = op.template get_destination_cooperative_tensor<
            tensor<device T, extents<int, BK, 16>, tensor_inline>,
            tensor<threadgroup T, extents<int, BN, BK>, tensor_inline>,
            float>();
        _Pragma("unroll")
        for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {{
            ct_c[i] = 0.0f;
            if constexpr (NR > 1) {{
                ct_c1[i] = 0.0f;
            }}
        }}

        int n_global = n0 + int(lane);
        const device uint4* wr = (const device uint4*)(w_q + (long)n_global * K_by_8);
        const device T* srow = scales + (long)n_global * K_by_gs;
        const device T* brow = biases + (long)n_global * K_by_gs;

        uint4 cur[NU];
        uint4 nxt[NU];
        float scur[NG];
        float bcur[NG];
        float snxt[NG];
        float bnxt[NG];
        _Pragma("unroll")
        for (int u = 0; u < NU; ++u) {{
            cur[u] = wr[k_begin / 32 + u];
        }}
        _Pragma("unroll")
        for (int g = 0; g < NG; ++g) {{
            scur[g] = float(srow[k_begin / GS + g]);
            bcur[g] = float(brow[k_begin / GS + g]);
        }}

        auto dequant = [&](threadgroup T* dst, uint4 q, float ss, float bb) {{
            uint32_t p[4] = {{q.x, q.y, q.z, q.w}};
            _Pragma("unroll")
            for (int j = 0; j < 4; ++j) {{
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (p[j] >> (ki * 4)) & 0xFu;
                    dst[(j * 8 + ki) * BN + int(lane)] = T(float(nib) * ss + bb);
                }}
            }}
        }};

        dequant(B_tile[sg_id][0], cur[0], scur[0], bcur[0]);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (int k1 = k_begin; k1 < k_end; k1 += LU) {{
            bool more = (k1 + LU) < k_end;
            if (more) {{
                _Pragma("unroll")
                for (int u = 0; u < NU; ++u) {{
                    nxt[u] = wr[(k1 + LU) / 32 + u];
                }}
                _Pragma("unroll")
                for (int g = 0; g < NG; ++g) {{
                    snxt[g] = float(srow[(k1 + LU) / GS + g]);
                    bnxt[g] = float(brow[(k1 + LU) / GS + g]);
                }}
            }}
            _Pragma("unroll")
            for (int u = 0; u < NU; ++u) {{
                int k0 = k1 + u * BK;
                auto tA = A.template slice<BK, 16>(k0, 0);
                if ((u & 1) == 0) {{
                    auto tB = B0.template slice<BN, BK>(0, 0);
                    op.run(tA, tB, ct_c);
                    if constexpr (NR > 1) {{
                        auto tA1 = A.template slice<BK, 16>(k0, 16);
                        op.run(tA1, tB, ct_c1);
                    }}
                }} else {{
                    auto tB = B1.template slice<BN, BK>(0, 0);
                    op.run(tA, tB, ct_c);
                    if constexpr (NR > 1) {{
                        auto tA1 = A.template slice<BK, 16>(k0, 16);
                        op.run(tA1, tB, ct_c1);
                    }}
                }}
                if (u + 1 < NU) {{
                    dequant(
                        B_tile[sg_id][(u + 1) & 1],
                        cur[u + 1],
                        scur[((u + 1) * 32) / GS],
                        bcur[((u + 1) * 32) / GS]);
                }} else if (more) {{
                    dequant(B_tile[sg_id][(u + 1) & 1], nxt[0], snxt[0], bnxt[0]);
                }}
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }}
            if (more) {{
                _Pragma("unroll")
                for (int u = 0; u < NU; ++u) {{
                    cur[u] = nxt[u];
                }}
                _Pragma("unroll")
                for (int g = 0; g < NG; ++g) {{
                    scur[g] = snxt[g];
                    bcur[g] = bnxt[g];
                }}
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* part = partial_base + sg_id * (BM * BN);
        tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
            part,
            dextents<int, 2>{{BN, BM}},
            array<int, 2>{{1, BN}});
        auto tC = C.template slice<BN, 16>(0, 0);
        ct_c.store(tC);
        if constexpr (NR > 1) {{
            auto tC1 = C.template slice<BN, 16>(0, 16);
            ct_c1.store(tC1);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc01 = partial_base[0 * BM * BN + off] + partial_base[1 * BM * BN + off];
            float acc23 = partial_base[2 * BM * BN + off] + partial_base[3 * BM * BN + off];
            float acc45 = partial_base[4 * BM * BN + off] + partial_base[5 * BM * BN + off];
            float acc67 = partial_base[6 * BM * BN + off] + partial_base[7 * BM * BN + off];
            float acc = (acc01 + acc23) + (acc45 + acc67);
            int row = off / BN;
            int col = off - row * BN;
            y[row * N + n0 + col] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_verify_m16_nax_pp_k{int(k_val)}_gs{group_size}_r{int(row_tiles)}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        header="""
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        """,
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m4_ksplit_np(group_size: int, dtype: mx.Dtype, *, k_parts: int = 4):
    key = ("m4_ksplit_np", group_size, dtype, int(k_parts))
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int M = 4;
        constexpr int BN = 4;
        constexpr int K_PARTS = {int(k_parts)};
        constexpr int GS = {group_size};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int packs_per_part = K_by_8 / K_PARTS;
        int pack_start = int(part) * packs_per_part;
        int pack_end = (int(part) == K_PARTS - 1) ? K_by_8 : pack_start + packs_per_part;

        float acc[BN * M];
        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {{
            int k_base = pack * 8;
            Vec8 v0 = xv[(0 * K + k_base) / 8];
            Vec8 v1 = xv[(1 * K + k_base) / 8];
            Vec8 v2 = xv[(2 * K + k_base) / 8];
            Vec8 v3 = xv[(3 * K + k_base) / 8];
            uint32_t p0 = w_q[(n0 + 0) * K_by_8 + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_8 + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_8 + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_8 + pack];
            float s0 = float(scales[(n0 + 0) * K_by_gs + (k_base / GS)]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + (k_base / GS)]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + (k_base / GS)]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + (k_base / GS)]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + (k_base / GS)]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + (k_base / GS)]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + (k_base / GS)]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + (k_base / GS)]);

            {{
                uint32_t packed = p0;
                float s = s0;
                float b = b0;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[0 * M + 0] += float(v0[ki]) * wv;
                    acc[0 * M + 1] += float(v1[ki]) * wv;
                    acc[0 * M + 2] += float(v2[ki]) * wv;
                    acc[0 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p1;
                float s = s1;
                float b = b1;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[1 * M + 0] += float(v0[ki]) * wv;
                    acc[1 * M + 1] += float(v1[ki]) * wv;
                    acc[1 * M + 2] += float(v2[ki]) * wv;
                    acc[1 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p2;
                float s = s2;
                float b = b2;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[2 * M + 0] += float(v0[ki]) * wv;
                    acc[2 * M + 1] += float(v1[ki]) * wv;
                    acc[2 * M + 2] += float(v2[ki]) * wv;
                    acc[2 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p3;
                float s = s3;
                float b = b3;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[3 * M + 0] += float(v0[ki]) * wv;
                    acc[3 * M + 1] += float(v1[ki]) * wv;
                    acc[3 * M + 2] += float(v2[ki]) * wv;
                    acc[3 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
        }}

        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partial[K_PARTS * BN * M];
        if (lane == 0) {{
            for (int i = 0; i < BN * M; ++i) {{
                partial[int(part) * BN * M + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < BN * M) {{
            float total = 0.0f;
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partial[p * BN * M + int(lane)];
            }}
            int j = int(lane) / M;
            int row = int(lane) - j * M;
            int n_global = n0 + j;
            if (n_global < N) {{
                y[row * N + n_global] = T(total);
            }}
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_verify_m4_ksplit_kp{int(k_parts)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m4_kp1(group_size: int, dtype: mx.Dtype):
    """MTPLX rewrite of the m4 kernel (2026-06-12 evening, not from dflash):
    single simdgroup per BN=4 tile — no K split, no threadgroup memory, no
    barrier, no partial reduction; simd_sum goes straight to the output.

    Chained-lazy microbench vs the ported K-split m4 (M5 Max, fans pinned):
    gate_up +5.3% (1.17x of the weight-stream floor), down +11%, qkvz/ba/
    gdn_out at parity or better, lm_head chained-context +27%. The K-split
    barrier/partial machinery cost more than its parallel reduction bought
    at these K sizes. Evidence: outputs/m4-rewrite-20260612/ (research repo).
    """
    key = ("m4_kp1", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int M = 4;
        constexpr int BN = 4;
        constexpr int GS = {group_size};

        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        float acc[BN * M];
        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = int(lane); pack < K_by_8; pack += 32) {{
            int k_base = pack * 8;
            int gi = k_base / GS;
            Vec8 v0 = xv[(0 * K + k_base) / 8];
            Vec8 v1 = xv[(1 * K + k_base) / 8];
            Vec8 v2 = xv[(2 * K + k_base) / 8];
            Vec8 v3 = xv[(3 * K + k_base) / 8];
            uint32_t p0 = w_q[(n0 + 0) * K_by_8 + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_8 + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_8 + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_8 + pack];
            float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                float w0 = float((p0 >> (ki * 4)) & 0xFu) * s0 + b0;
                float w1 = float((p1 >> (ki * 4)) & 0xFu) * s1 + b1;
                float w2 = float((p2 >> (ki * 4)) & 0xFu) * s2 + b2;
                float w3 = float((p3 >> (ki * 4)) & 0xFu) * s3 + b3;
                acc[0 * M + 0] += float(v0[ki]) * w0;
                acc[0 * M + 1] += float(v1[ki]) * w0;
                acc[0 * M + 2] += float(v2[ki]) * w0;
                acc[0 * M + 3] += float(v3[ki]) * w0;
                acc[1 * M + 0] += float(v0[ki]) * w1;
                acc[1 * M + 1] += float(v1[ki]) * w1;
                acc[1 * M + 2] += float(v2[ki]) * w1;
                acc[1 * M + 3] += float(v3[ki]) * w1;
                acc[2 * M + 0] += float(v0[ki]) * w2;
                acc[2 * M + 1] += float(v1[ki]) * w2;
                acc[2 * M + 2] += float(v2[ki]) * w2;
                acc[2 * M + 3] += float(v3[ki]) * w2;
                acc[3 * M + 0] += float(v0[ki]) * w3;
                acc[3 * M + 1] += float(v1[ki]) * w3;
                acc[3 * M + 2] += float(v2[ki]) * w3;
                acc[3 * M + 3] += float(v3[ki]) * w3;
            }}
        }}

        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        if (lane < BN * M) {{
            int j = int(lane) / M;
            int row = int(lane) - j * M;
            int n_global = n0 + j;
            if (n_global < N) {{
                y[row * N + n_global] = T(acc[int(lane)]);
            }}
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_verify_m4_kp1_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m4_bn6(group_size: int, dtype: mx.Dtype):
    """MTPLX rewrite, wide-tile variant: BN=6 columns per simdgroup (24 named
    accumulators — the proven m6 register ceiling), no K split, no barrier.
    1.5x fewer threadgroups; wins on deep-K (mlp_down +19%) and huge-N
    (lm_head chained-context +39%) shapes where the BN=4 grid is
    scheduler-hostile. Ragged N handled by clamped loads + guarded writes.
    Generated source: six identical column blocks (see m6 kernel for why
    named scalars + forced unrolls are load-bearing)."""
    key = ("m4_bn6", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    bn = 6
    decl = "\n        ".join(
        f"float a{j}_0 = 0.0f, a{j}_1 = 0.0f, a{j}_2 = 0.0f, a{j}_3 = 0.0f;"
        for j in range(bn)
    )
    loads = "\n            ".join(
        f"int nr{j} = min(n0 + {j}, N - 1);\n"
        f"            uint32_t p{j} = w_q[nr{j} * K_by_8 + pack];\n"
        f"            float s{j} = float(scales[nr{j} * K_by_gs + gi]);\n"
        f"            float b{j} = float(biases[nr{j} * K_by_gs + gi]);"
        for j in range(bn)
    )
    fmas = "\n                ".join(
        f"float w{j} = float((p{j} >> (ki * 4)) & 0xFu) * s{j} + b{j};\n"
        f"                a{j}_0 += float(v0[ki]) * w{j};\n"
        f"                a{j}_1 += float(v1[ki]) * w{j};\n"
        f"                a{j}_2 += float(v2[ki]) * w{j};\n"
        f"                a{j}_3 += float(v3[ki]) * w{j};"
        for j in range(bn)
    )
    sums = "\n        ".join(
        f"a{j}_0 = simd_sum(a{j}_0); a{j}_1 = simd_sum(a{j}_1); "
        f"a{j}_2 = simd_sum(a{j}_2); a{j}_3 = simd_sum(a{j}_3);"
        for j in range(bn)
    )
    writes = "\n        ".join(
        f"if (lane == {j} && n0 + {j} < N) {{\n"
        f"            y[0 * N + n0 + {j}] = T(a{j}_0);\n"
        f"            y[1 * N + n0 + {j}] = T(a{j}_1);\n"
        f"            y[2 * N + n0 + {j}] = T(a{j}_2);\n"
        f"            y[3 * N + n0 + {j}] = T(a{j}_3);\n"
        f"        }}"
        for j in range(bn)
    )
    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int BN = {bn};

        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        {decl}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = int(lane); pack < K_by_8; pack += 32) {{
            int k_base = pack * 8;
            int gi = k_base / GS;
            Vec8 v0 = xv[(0 * K + k_base) / 8];
            Vec8 v1 = xv[(1 * K + k_base) / 8];
            Vec8 v2 = xv[(2 * K + k_base) / 8];
            Vec8 v3 = xv[(3 * K + k_base) / 8];
            {loads}
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                {fmas}
            }}
        }}

        {sums}

        {writes}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_verify_m4_bn6_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m6_ksplit_np(group_size: int, dtype: mx.Dtype, *, k_parts: int = 2):
    """6-row K-split variant (24 accumulators/thread, scalar row registers).

    Beats both stock qmm (1.14-1.80x) and the m16 NAX tile (1.03-1.15x) on all
    live shapes at M=5..6 (2026-06-12 microbench), and unlike the NAX tile it
    is plain SIMD — no G17/macOS-26.2 gate. Covers the D4/D5 verify shapes.
    Note: an earlier un-unrolled probe measured 0.08-0.16x — explicit unrolls
    and scalar v0..v5 registers are load-bearing, not style.
    """
    key = ("m6_ksplit_np", group_size, dtype, int(k_parts))
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int M = 6;
        constexpr int BN = 4;
        constexpr int K_PARTS = {int(k_parts)};
        constexpr int GS = {group_size};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int packs_per_part = K_by_8 / K_PARTS;
        int pack_start = int(part) * packs_per_part;
        int pack_end = (int(part) == K_PARTS - 1) ? K_by_8 : pack_start + packs_per_part;

        float acc[BN * M];
        _Pragma("unroll")
        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {{
            int k_base = pack * 8;
            Vec8 v0 = xv[(0 * K + k_base) / 8];
            Vec8 v1 = xv[(1 * K + k_base) / 8];
            Vec8 v2 = xv[(2 * K + k_base) / 8];
            Vec8 v3 = xv[(3 * K + k_base) / 8];
            Vec8 v4 = xv[(4 * K + k_base) / 8];
            Vec8 v5 = xv[(5 * K + k_base) / 8];
            _Pragma("unroll")
            for (int j = 0; j < BN; ++j) {{
                uint32_t packed = w_q[(n0 + j) * K_by_8 + pack];
                float s = float(scales[(n0 + j) * K_by_gs + (k_base / GS)]);
                float b = float(biases[(n0 + j) * K_by_gs + (k_base / GS)]);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[j * M + 0] += float(v0[ki]) * wv;
                    acc[j * M + 1] += float(v1[ki]) * wv;
                    acc[j * M + 2] += float(v2[ki]) * wv;
                    acc[j * M + 3] += float(v3[ki]) * wv;
                    acc[j * M + 4] += float(v4[ki]) * wv;
                    acc[j * M + 5] += float(v5[ki]) * wv;
                }}
            }}
        }}

        _Pragma("unroll")
        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partial[K_PARTS * BN * M];
        if (lane == 0) {{
            _Pragma("unroll")
            for (int i = 0; i < BN * M; ++i) {{
                partial[int(part) * BN * M + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < BN * M) {{
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partial[p * BN * M + int(lane)];
            }}
            int j = int(lane) / M;
            int row = int(lane) - j * M;
            int n_global = n0 + j;
            if (n_global < N) {{
                y[row * N + n_global] = T(total);
            }}
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_verify_m6_ksplit_kp{int(k_parts)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def m6_ksplit_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    return (
        int(bits) == 4
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and 5 <= int(M) <= 6
        and int(K) % 32 == 0
        and int(N) % 4 == 0
    )


def nax_qmm_m6(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Run the 6-row K-split verify matmul. Pads M=5 to 6 rows."""
    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    if M < 6:
        pad = mx.zeros((6 - M, K), dtype=x2.dtype)
        x6 = mx.contiguous(mx.concatenate([x2, pad], axis=0))
    else:
        x6 = mx.contiguous(x2)
    kernel = _build_kernel_m6_ksplit_np(group_size, x2.dtype, k_parts=2)
    (y,) = kernel(
        inputs=[x6, w_q, scales, biases, K, N],
        template=[("T", x2.dtype)],
        grid=(64, N // 4, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(6, N)],
        output_dtypes=[x2.dtype],
    )
    if M < 6:
        return y[:M, :]
    return y


# CLOSED BRANCH (2026-06-12, removed 2026-09-06 for issue #322): an 8-row
# K-split verify matmul (BN=4, 32 partials) was built and microbenched at
# 0.51-0.87x of stock on every live shape - eight Vec8 row loads plus 32
# accumulators cost more occupancy than the wider rows buy. The dispatcher
# never routed it. M in 5..16 goes to the m16 NAX tile below; do not
# rebuild the m8 lane without a receipt that beats both.


def m16_nax_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    return (
        int(bits) == 4
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and 1 <= int(M) <= 16
        and int(K) % 256 == 0
        and int(N) % 32 == 0
        and nax_available()
    )


def m4_ksplit_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    return (
        int(bits) == 4
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and int(M) == 4
        and int(K) % 32 == 0
        and int(N) % 4 == 0
    )


def m16_pipelined_eligible(
    M: int, K: int, N: int, bits: int, group_size: int, dtype
) -> bool:
    """The pipelined m16 tile walks two-step units of 32 K per simdgroup."""
    return m16_nax_eligible(M, K, N, bits, group_size, dtype) and int(K) % 512 == 0


def m32_nax_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    """17 to 32 rows on the two-row-tile pipelined kernel (NAX hardware, K % 512)."""
    return (
        17 <= int(M) <= 32
        and m16_pipelined_eligible(16, K, N, bits, group_size, dtype)
    )


def m5_rows_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    return (
        int(bits) == 4
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and int(M) == 5
        and int(K) % 64 == 0
        and int(N) % 4 == 0
    )


def nax_qmm_m16(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Run the 16-row NAX verify matmul. x2 must be (M<=16, K); pads M to 16."""
    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    if M < 16:
        pad = mx.zeros((16 - M, K), dtype=x2.dtype)
        x16 = mx.contiguous(mx.concatenate([x2, pad], axis=0))
    else:
        x16 = mx.contiguous(x2)
    if m16_pipelined_enabled() and m16_pipelined_eligible(
        16, K, N, 4, group_size, x2.dtype
    ):
        kernel = _build_kernel_m16_nax_pipelined(K, group_size, x2.dtype)
    else:
        kernel = _build_kernel_m16_nax_ktmpl(K, group_size, x2.dtype)
    (y,) = kernel(
        inputs=[x16, w_q, scales, biases, N],
        template=[("T", x2.dtype), ("KCONST", K)],
        grid=(256, N // 32, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(16, N)],
        output_dtypes=[x2.dtype],
    )
    if M < 16:
        return y[:M, :]
    return y


def nax_qmm_m32(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Run the 32-row (two 16-row tiles) pipelined NAX matmul. x2 is (17..32, K); pads to 32."""
    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    if M < 32:
        pad = mx.zeros((32 - M, K), dtype=x2.dtype)
        x32 = mx.contiguous(mx.concatenate([x2, pad], axis=0))
    else:
        x32 = mx.contiguous(x2)
    kernel = _build_kernel_m16_nax_pipelined(K, group_size, x2.dtype, row_tiles=2)
    (y,) = kernel(
        inputs=[x32, w_q, scales, biases, N],
        template=[("T", x2.dtype), ("KCONST", K)],
        grid=(256, N // 32, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(32, N)],
        output_dtypes=[x2.dtype],
    )
    if M < 32:
        return y[:M, :]
    return y


def nax_qmm_m5(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Run the exact 5-row split-K verify matmul. x2 must be (5, K).

    The MTPLX verify_kernels split-K kernel at M=5 (20 accumulators per thread,
    K as a compile-time constant), without padding to 6 rows. Each weight tile
    is read once and serves all five rows.
    """
    from .verify_kernels import _run_ksplit

    return _run_ksplit(
        5,
        x2,
        w_q,
        scales,
        biases,
        bits=4,
        group_size=group_size,
        dual=False,
        kconst=True,
    )


_QLINEAR_PATCH: dict[str, object] = {"installed": False, "original": None}

# F23b (2026-08-16): verify-shaped QuantizedLinear calls that entered the
# patched fast-path window (bits in {4,6,8}, decode/verify phase, M in the
# verify range) but fell through every kernel gate back to stock. Keyed
# "b{bits}_m{M}" — the shape class tells which lane silently declined
# (lane_disabled kill switches, eligibility geometry, small-N floors).
# Import-stable surface for /health; increments happen ONLY on the bail
# path (calls the kernels serve never touch this dict).
nax_qlinear_fallback_counts: dict[str, int] = {}


def _count_qlinear_fallback(bits: int, m: int) -> None:
    key = f"b{int(bits)}_m{int(m)}"
    nax_qlinear_fallback_counts[key] = nax_qlinear_fallback_counts.get(key, 0) + 1


def install_nax_qlinear_patch() -> dict[str, object]:
    """Route verify-shaped (M in 4..16) 4-bit QuantizedLinear calls through the
    NAX/m4 verify kernels. Decode (M=1..3) and prefill (M>16) stay stock.

    Returns a report dict. Idempotent.
    """
    import mlx.nn as nn

    if _QLINEAR_PATCH["installed"]:
        return {"installed": True, "already": True, "nax_available": nax_available()}

    original = nn.QuantizedLinear.__call__

    from .attention_context import current_attention_phase, exact_verify_required
    from .kernel_selfcheck import lane_disabled

    def patched(self, x: mx.array) -> mx.array:  # type: ignore[no-untyped-def]
        if exact_verify_required():
            # Greedy (t<=0) verify forwards demand bit-exact stock matmuls:
            # the vk/nax lanes trade ~6e-3 accumulation-order drift for speed,
            # which flips argmax at near-tie logit rows and breaks the
            # MTP==AR greedy identity (the product's exactness promise).
            nax_qlinear_fallback_counts["exact_t0"] = (
                nax_qlinear_fallback_counts.get("exact_t0", 0) + 1
            )
            return original(self, x)
        bits = int(getattr(self, "bits", 0) or 0)
        group_size = int(getattr(self, "group_size", 0) or 0)
        if bits == 8 and x.ndim >= 2 and current_attention_phase() != "prefill":
            # 8-bit affine (Optimized-Quality): MTPLX verify_kernels family.
            from .verify_kernels import (
                vk_eligible_ksplit,
                vk_eligible_m4,
                vk_eligible_m6,
                vk_qmm_m4,
                vk_qmm_m4_ksplit,
                vk_qmm_m6,
                vk_qmm_m6_ksplit,
            )

            m = 1
            for d in x.shape[:-1]:
                m *= int(d)
            if 4 <= m <= 6:
                w_q = self["weight"]
                k = int(x.shape[-1])
                n = int(w_q.shape[0])
                y = None
                huge_n = n >= 100000
                if (
                    m == 4
                    and huge_n
                    and not lane_disabled("qmm_m4_wide")
                    and vk_eligible_m4(m, k, n, bits, group_size, x.dtype)
                ):
                    # lm_head-class shapes: the wide msg tile (few big TGs)
                    # wins isolated 1.34x while split-K thrashes (0.66x) in
                    # the 62k-tiny-threadgroup regime.
                    y = vk_qmm_m4(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        bits=8, group_size=group_size,
                    )
                elif (
                    m == 4
                    and not lane_disabled("qmm_m4")
                    and vk_eligible_ksplit(m, k, n, bits, group_size, x.dtype)
                ):
                    # Split-K morphology: the in-context winner (msg geometry
                    # loses its isolated 1.3-1.5x to co-residency on the
                    # layer shapes).
                    y = vk_qmm_m4_ksplit(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        bits=8, group_size=group_size,
                    )
                elif (
                    huge_n
                    and not lane_disabled("qmm_m6_wide")
                    and vk_eligible_m6(m, k, n, bits, group_size, x.dtype)
                ):
                    y = vk_qmm_m6(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        bits=8, group_size=group_size,
                    )
                elif (
                    not lane_disabled("qmm_m6")
                    and vk_eligible_ksplit(m, k, n, bits, group_size, x.dtype)
                ):
                    y = vk_qmm_m6_ksplit(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        bits=8, group_size=group_size,
                    )
                if y is not None:
                    y = y.reshape(*x.shape[:-1], n)
                    if "bias" in self:
                        y = y + self["bias"]
                    return y
                _count_qlinear_fallback(bits, m)
        if bits == 6 and x.ndim >= 2 and current_attention_phase() != "prefill":
            # 6-bit affine (9B tier), added 2026-07-07: split-K hexpack
            # kernels, exactness-gated vs stock across {bf16,fp16} x
            # gs{32,64,128} (54 cases, dmax <= 0.027 bf16 / 0.003 fp16).
            # Microbench on the 9B hot shapes: 1.1-2.4x vs stock; the tiny
            # kv projection (N=1024) measured 0.92x at m4/bf16, so small-N
            # stays stock via the floor below.
            from .verify_kernels import (
                vk_eligible_ksplit,
                vk_qmm_m4_ksplit,
                vk_qmm_m6_ksplit,
            )

            m = 1
            for d in x.shape[:-1]:
                m *= int(d)
            if 4 <= m <= 6:
                w_q = self["weight"]
                k = int(x.shape[-1])
                n = int(w_q.shape[0])
                y = None
                if (
                    m == 4
                    and n >= 2048
                    and not lane_disabled("qmm_m4")
                    and vk_eligible_ksplit(m, k, n, bits, group_size, x.dtype)
                ):
                    y = vk_qmm_m4_ksplit(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        bits=6, group_size=group_size,
                    )
                elif (
                    5 <= m <= 6
                    and n >= 2048
                    and not lane_disabled("qmm_m6")
                    and vk_eligible_ksplit(m, k, n, bits, group_size, x.dtype)
                ):
                    y = vk_qmm_m6_ksplit(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        bits=6, group_size=group_size,
                    )
                if y is not None:
                    y = y.reshape(*x.shape[:-1], n)
                    if "bias" in self:
                        y = y + self["bias"]
                    return y
                _count_qlinear_fallback(bits, m)
        if bits == 4 and x.ndim >= 2 and current_attention_phase() != "prefill":
            m = 1
            for d in x.shape[:-1]:
                m *= int(d)
            if 4 <= m <= (32 if m32_tile_enabled() else 16):
                w_q = self["weight"]
                k = int(x.shape[-1])
                n = int(w_q.shape[0])
                y = None
                if (
                    m > 16
                    and not lane_disabled("qmm_m32_nax")
                    and m32_nax_eligible(m, k, n, bits, group_size, x.dtype)
                ):
                    # 17 to 32 rows (context-copy blocks): two 16-row tiles on
                    # one dequantized B tile, opt-in (MTPLX_NAX_M32_TILE=1).
                    y = nax_qmm_m32(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        group_size=group_size,
                    )
                elif (
                    m == 4
                    and not lane_disabled("qmm_m4")
                    and m4_ksplit_eligible(m, k, n, bits, group_size, x.dtype)
                ):
                    # Plain SIMD K-split kernel: no NAX hardware requirement.
                    y = nax_qmm_m4(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        group_size=group_size,
                    )
                elif (
                    m == 5
                    and m5_rows_lane_enabled()
                    and not lane_disabled("qmm_m5")
                    and m5_rows_eligible(m, k, n, bits, group_size, x.dtype)
                ):
                    # Exact 5-row split-K lane, opt-in (MTPLX_M5_ROWS_LANE=1):
                    # reaches M=4 speed where stock qmv_wide and the padded
                    # lanes lose 10 to 15%.
                    y = nax_qmm_m5(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        group_size=group_size,
                    )
                elif (
                    # M=5 falls through to stock by default: three same-
                    # process micro sessions (2026-08-22, MEASUREMENTS 22:0x)
                    # put the padded-m6 lane +3..13% and the m16 tile worst
                    # at every M=5 4-bit shape, while stock ~ties the best.
                    # MTPLX_M5_PADDED_LANE=1 restores the old routing.
                    (m == 6 or (m == 5 and _m5_padded_lane()))
                    and not lane_disabled("qmm_m6")
                    and m6_ksplit_eligible(m, k, n, bits, group_size, x.dtype)
                ):
                    y = nax_qmm_m6(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        group_size=group_size,
                    )
                elif (
                    (m != 5 or _m5_padded_lane())
                    and not lane_disabled("qmm_m16_nax")
                    and m16_nax_eligible(m, k, n, bits, group_size, x.dtype)
                ):
                    y = nax_qmm_m16(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        group_size=group_size,
                    )
                if y is not None:
                    y = y.reshape(*x.shape[:-1], n)
                    if "bias" in self:
                        y = y + self["bias"]
                    return y
                _count_qlinear_fallback(bits, m)
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _QLINEAR_PATCH["installed"] = True
    _QLINEAR_PATCH["original"] = original
    return {"installed": True, "already": False, "nax_available": nax_available()}


def uninstall_nax_qlinear_patch() -> None:
    import mlx.nn as nn

    if _QLINEAR_PATCH["installed"] and _QLINEAR_PATCH["original"] is not None:
        nn.QuantizedLinear.__call__ = _QLINEAR_PATCH["original"]
        _QLINEAR_PATCH["installed"] = False
        _QLINEAR_PATCH["original"] = None


# Default is the ported K-split kernel: the kp1/bn6 rewrites win isolated
# microbenches and tie the 192 tune lane, but measure ~15% SLOWER per verify
# call on the deferred serve path at long context (2026-06-12 flappy pair:
# hidden-eval 48.8 -> 57.2 ms/call, acceptance matched) — small threadgroups
# lose under mixed co-residency with attention/GDN kernels. Promotion bar for
# any m4 variant: the serve-path long-form A/B, not the tune lane.
def _m4_impl() -> str:
    # Read per call, not at import: the turbo profile exports
    # MTPLX_NAX_M4_IMPL=vk_k while the server boots, and an import-time
    # snapshot silently pinned whichever value happened to be set first.
    return str(os.environ.get("MTPLX_NAX_M4_IMPL", "legacy")).strip().lower()


def _m5_padded_lane() -> bool:
    """Opt back into padded custom lanes for M=5 (default: stock).

    2026-08-22 three-session micro receipts at the production 4-bit g32
    shapes: padded-m6 pays +3..13% over stock and the m16 tile is worst at
    every M=5 cell; crossrow ~ties stock across sessions. Stock is the
    only lane that never loses at M=5, so it is the default; this env
    restores the previous padded routing for A/Bs.
    """
    return str(os.environ.get("MTPLX_M5_PADDED_LANE", "")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "on", "yes"}


def _multirow_umbrella() -> bool:
    """MTPLX_MULTIROW_QMM=1 turns on all three multi-row lanes at once."""
    return _env_flag("MTPLX_MULTIROW_QMM")


def m16_pipelined_enabled() -> bool:
    """Opt in to the pipelined m16 tile (default off, read per call).

    Bit-identical to the default m16 kernel, 17 to 21% faster per matmul on the
    Qwen3.8-27B shapes (see ``_build_kernel_m16_nax_pipelined``).
    """
    return _env_flag("MTPLX_NAX_M16_PIPELINED") or _multirow_umbrella()


def m32_tile_enabled() -> bool:
    """Opt in to the 32-row tile for 17 to 32 verify rows (default off, read per call)."""
    return _env_flag("MTPLX_NAX_M32_TILE") or _multirow_umbrella()


def m5_rows_lane_enabled() -> bool:
    """Opt in to the exact 5-row split-K lane for M=5 (default off, read per call)."""
    return _env_flag("MTPLX_M5_ROWS_LANE") or _multirow_umbrella()


def nax_qmm_m4(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Run the exact 4-row verify matmul. x2 must be (4, K).

    Implementation is selected by MTPLX_NAX_M4_IMPL:
      auto (default)  MTPLX rewrite family — bn6 wide tile for deep-K
                      (K >= 12288) and huge-N (N >= 100000) shapes,
                      kp1 single-simdgroup tile elsewhere.
      v3 / v4         force kp1 / bn6 everywhere (diagnostics).
      legacy          the original ported dflash K-split kernel.
    """
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    impl = _m4_impl()
    if impl in ("oct", "twin", "vk", "vk_u2", "vk_k", "vk_hybrid"):
        # MTPLX verify_kernels family (original implementations, 2026-07-02).
        from .verify_kernels import (
            vk_eligible_ksplit,
            vk_eligible_m4,
            vk_qmm_m4_impl,
        )

        if impl == "twin":
            eligible = (
                int(K) % 64 == 0
                and int(N) % 8 == 0
                and x2.dtype in (mx.bfloat16, mx.float16)
            )
        elif impl == "oct":
            eligible = vk_eligible_m4(4, K, N, 4, group_size, x2.dtype)
        else:
            eligible = vk_eligible_ksplit(4, K, N, 4, group_size, x2.dtype)
        if eligible:
            return vk_qmm_m4_impl(
                impl, x2, w_q, scales, biases, bits=4, group_size=group_size
            )
        impl = "legacy"
    if impl == "legacy":
        k_parts = 2 if N >= 4096 else 4
        kernel = _build_kernel_m4_ksplit_np(group_size, x2.dtype, k_parts=k_parts)
        grid = (32 * k_parts, N // 4, 1)
        tg = (32 * k_parts, 1, 1)
    elif impl == "v4" or (impl != "v3" and (N >= 100000 or K >= 12288)):
        kernel = _build_kernel_m4_bn6(group_size, x2.dtype)
        grid = (32, (N + 5) // 6, 1)
        tg = (32, 1, 1)
    else:
        kernel = _build_kernel_m4_kp1(group_size, x2.dtype)
        grid = (32, N // 4, 1)
        tg = (32, 1, 1)
    (y,) = kernel(
        inputs=[mx.contiguous(x2), w_q, scales, biases, K, N],
        template=[("T", x2.dtype)],
        grid=grid,
        threadgroup=tg,
        output_shapes=[(4, N)],
        output_dtypes=[x2.dtype],
    )
    return y
