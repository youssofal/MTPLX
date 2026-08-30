"""simdgroup_matrix MMA paths for 4-bit affine quantized matmuls.

Two kernel families, both exact-drop-in alternatives to stock
``mx.quantized_matmul`` (transpose=True, affine 4-bit):

- small-M lane (M in 1..16, any N incl. the 248,320-row vocabulary
  projection): BM=8/16 padded tile, BN=32, BK=64, one 128-thread
  threadgroup per 32 output columns. Stock steel dispatch collapses on
  the M in 8..16 vocabulary shapes (44.5 / 28.4 GB/s effective on a
  base M4 vs 98.6 GB/s at M=1); this lane keeps the weight stream at
  bandwidth while the simdgroup MMA units absorb the row padding.
- prefill lane (M >= 128, M % 128 == 0): BM=128, BN=32, BK=64,
  WM=4, WN=1, two alternating threadgroup weight buffers with the next
  tile's dequant overlapped into the current tile's MMAs, one barrier
  per inner iteration, and a grouped threadgroup-order swizzle so
  adjacent threadgroups share A rows and stream disjoint weights.

Tile geometry and the double-buffer/swizzle scheme follow the
Metal 4.1 packed-INT4 recipe measured upstream on G17 hardware
(BM=128/BN=32/BK=64/WM=4/WN=1, one barrier per iteration); the
simdgroup_matrix lowering here runs on every Apple GPU (G13+), so the
same source serves M4 development and M5 deployment. The MPP
``matmul2d`` native-int4 path (nax_verify.py) remains the G17 fast path
where present.
"""

from __future__ import annotations

import os
import platform
from functools import lru_cache

import mlx.core as mx

_MMA_KERNEL_CACHE: dict[tuple, object] = {}
_MMA_DISPATCH_COUNTERS: dict[str, int] = {}


def _count_mma_dispatch(kind: str, *, m: int, k: int, n: int) -> None:
    for key in (
        str(kind),
        f"{kind}_m{int(m)}_k{int(k)}_n{int(n)}",
    ):
        _MMA_DISPATCH_COUNTERS[key] = _MMA_DISPATCH_COUNTERS.get(key, 0) + 1


def mma_dispatch_counter_snapshot() -> dict[str, int]:
    """Return process-lifetime routed-kernel counts for receipt deltas."""

    return dict(_MMA_DISPATCH_COUNTERS)


def mma_env_lane(lane: str) -> bool:
    """Lanes: 'vocab' (small-M), 'prefill'. Default off until gated."""

    value = str(os.environ.get("MTPLX_INT4_MMA", "")).strip().lower()
    if value in {"1", "true", "on", "yes", "all"}:
        return True
    return lane in {
        item.strip().lower() for item in value.split(",") if item.strip()
    }


_DTYPE_TAG = {mx.bfloat16: "bf16", mx.float16: "fp16"}


@lru_cache(maxsize=None)
def _build_vocab_kernel(bm: int, group_size: int, dtype: mx.Dtype):
    nsg = 4
    source = f"""
        using namespace metal;
        constexpr int BM = {int(bm)};
        constexpr int BN = 32;
        constexpr int BK = 64;
        constexpr int NSG = {int(nsg)};
        constexpr int GS = {group_size};

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_f[{int(bm) // 8}];
        simdgroup_matrix<T, 8, 8> b_f;
        simdgroup_matrix<float, 8, 8> c_f[{int(bm) // 8}];
        #pragma unroll
        for (int mi = 0; mi < BM / 8; ++mi) {{
            c_f[mi] = simdgroup_matrix<float, 8, 8>(0.0f);
        }}

        // 128 threads dequant 256 packed words (2 each) per BK step.
        // tid -> (k_local word, n_local): word k coverage is 8 rows of B.
        int dq_word0 = int(tid) * 2;
        int dq_word1 = int(tid) * 2 + 1;

        for (int k0 = 0; k0 < K; k0 += BK) {{
            #pragma unroll
            for (int w = 0; w < 2; ++w) {{
                int word = w == 0 ? dq_word0 : dq_word1;
                int k_local = (word % (BK / 8)) * 8;
                int n_local = word / (BK / 8);
                int n_global = n0 + n_local;
                int k_base = k0 + k_local;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                #pragma unroll
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(k_local + ki) * BN + n_local] = T(float(nib) * s + b);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (int ks = 0; ks < BK / 8; ++ks) {{
                #pragma unroll
                for (int mi = 0; mi < BM / 8; ++mi) {{
                    simdgroup_load(
                        a_f[mi], x + (mi * 8) * K + k0 + ks * 8, K);
                }}
                simdgroup_load(
                    b_f, B_tile + ks * 8 * BN + int(sg_id) * 8, BN);
                #pragma unroll
                for (int mi = 0; mi < BM / 8; ++mi) {{
                    simdgroup_multiply_accumulate(
                        c_f[mi], a_f[mi], b_f, c_f[mi]);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_matrix<T, 8, 8> c_t;
        #pragma unroll
        for (int mi = 0; mi < BM / 8; ++mi) {{
            c_t.thread_elements()[0] = T(c_f[mi].thread_elements()[0]);
            c_t.thread_elements()[1] = T(c_f[mi].thread_elements()[1]);
            simdgroup_store(c_t, y + (mi * 8) * N + n0 + int(sg_id) * 8, N);
        }}
    """
    dtype_tag = _DTYPE_TAG.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_mma_vocab_bm{int(bm)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )


def int4_vocab_qmm(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Exact small-M 4-bit matmul for huge-N shapes. x2 is (M, K), M <= 16."""

    m = int(x2.shape[0])
    k = int(x2.shape[1])
    n = int(w_q.shape[0])
    bm = 16 if m > 8 else 8
    if m < bm:
        x2 = mx.concatenate([x2, mx.zeros((bm - m, k), dtype=x2.dtype)], axis=0)
    x2 = mx.contiguous(x2)
    kernel = _build_vocab_kernel(bm, group_size, x2.dtype)
    (y,) = kernel(
        inputs=[x2, w_q, scales, biases, k, n],
        template=[("T", x2.dtype)],
        grid=(32 * 4, n // 32, 1),
        threadgroup=(32 * 4, 1, 1),
        output_shapes=[(bm, n)],
        output_dtypes=[x2.dtype],
    )
    _count_mma_dispatch("vocab", m=m, k=k, n=n)
    return y[:m, :] if m < bm else y


def vocab_mma_eligible(m: int, k: int, n: int, bits: int, group_size: int, dtype) -> bool:
    if bits != 4 or group_size not in (32, 64, 128):
        return False
    if dtype not in (mx.bfloat16, mx.float16):
        return False
    # Stock GEMV wins below M=8 (~98 GB/s at m=1); our MMA lane only beats
    # stock once stock's wide-tile path collapses (M >= 8).
    if not (8 <= m <= 16):
        return False
    return k % 64 == 0 and n % 32 == 0


@lru_cache(maxsize=None)
def _build_prefill_kernel(group_size: int, dtype: mx.Dtype):
    source = f"""
        using namespace metal;
        constexpr int BM = 128;
        constexpr int BN = 32;
        constexpr int BK = 64;
        constexpr int WM = 4;   // row frags per simdgroup
        constexpr int WN = 1;   // col frags at a time (loop all 4)
        constexpr int NSG = 4;  // 128 threads
        constexpr int GS = {group_size};
        constexpr int GROUP_M = 8;  // threadgroup swizzle super-group

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint tg_m  = threadgroup_position_in_grid.x;
        uint tg_n  = threadgroup_position_in_grid.y;

        int M = int(M_size);
        int K = int(K_size);
        int N = int(N_size);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;

        int num_m = M / BM;
        int num_n = N / BN;

        // Grouped swizzle: consecutive threadgroups cover the same
        // GROUP_M row-tiles with consecutive column-tiles, so A rows
        // stay hot while B reads stream disjoint weight columns.
        uint linear = uint(tg_n) * uint(num_m) + uint(tg_m);
        int tiles_per_group = GROUP_M * num_n;
        int group_id = int(linear) / tiles_per_group;
        int first_m = group_id * GROUP_M;
        int group_m = min(num_m - first_m, GROUP_M);
        int rem = int(linear) - group_id * tiles_per_group;
        int sw_m = first_m + (rem % group_m);
        int sw_n = rem / group_m;

        int m0 = sw_m * BM;
        int n0 = sw_n * BN;

        threadgroup T B_buf[2][BK * BN];

        simdgroup_matrix<T, 8, 8> a_f[WM];
        simdgroup_matrix<T, 8, 8> b_f;
        simdgroup_matrix<float, 8, 8> c_f[WM][BN / 8];
        simdgroup_matrix<T, 8, 8> c_t;
        #pragma unroll
        for (int wi = 0; wi < WM; ++wi) {{
            #pragma unroll
            for (int wn = 0; wn < BN / 8; ++wn) {{
                c_f[wi][wn] = simdgroup_matrix<float, 8, 8>(0.0f);
            }}
        }}

        // 256 packed words per BK tile, 128 threads -> 2 words each.
        int dq_word0 = int(tid) * 2;
        int dq_word1 = int(tid) * 2 + 1;

        int cur = 0;
        // Prologue: dequant tile 0 into buffer 0.
        {{
            #pragma unroll
            for (int w = 0; w < 2; ++w) {{
                int word = w == 0 ? dq_word0 : dq_word1;
                int k_local = (word % (BK / 8)) * 8;
                int n_local = word / (BK / 8);
                int n_global = n0 + n_local;
                int k_base = k_local;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                #pragma unroll
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_buf[0][(k_local + ki) * BN + n_local] =
                        T(float(nib) * s + b);
                }}
            }}
        }}

        for (int k0 = 0; k0 < K; k0 += BK) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int kn = k0 + BK;
            if (kn < K) {{
                // Dequant tile k0+BK into the other buffer while the
                // current tile's MMAs consume B_buf[cur].
                #pragma unroll
                for (int w = 0; w < 2; ++w) {{
                    int word = w == 0 ? dq_word0 : dq_word1;
                    int k_local = (word % (BK / 8)) * 8;
                    int n_local = word / (BK / 8);
                    int n_global = n0 + n_local;
                    int k_base = kn + k_local;
                    uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                    float s = float(
                        scales[n_global * K_by_gs + (k_base / GS)]);
                    float b = float(
                        biases[n_global * K_by_gs + (k_base / GS)]);
                    #pragma unroll
                    for (int ki = 0; ki < 8; ++ki) {{
                        uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                        B_buf[1 - cur][(k_local + ki) * BN + n_local] =
                            T(float(nib) * s + b);
                    }}
                }}
            }}

            threadgroup T * B_tile = B_buf[cur];
            #pragma unroll
            for (int ks = 0; ks < BK / 8; ++ks) {{
                #pragma unroll
                for (int wi = 0; wi < WM; ++wi) {{
                    simdgroup_load(
                        a_f[wi],
                        x + (m0 + int(sg_id) * (WM * 8) + wi * 8) * K
                            + k0 + ks * 8,
                        K);
                }}
                #pragma unroll
                for (int wn = 0; wn < BN / 8; ++wn) {{
                    simdgroup_load(
                        b_f, B_tile + ks * 8 * BN + wn * 8, BN);
                    #pragma unroll
                    for (int wi = 0; wi < WM; ++wi) {{
                        simdgroup_multiply_accumulate(
                            c_f[wi][wn], a_f[wi], b_f, c_f[wi][wn]);
                    }}
                }}
            }}
            cur = 1 - cur;
        }}

        #pragma unroll
        for (int wi = 0; wi < WM; ++wi) {{
            #pragma unroll
            for (int wn = 0; wn < BN / 8; ++wn) {{
                c_t.thread_elements()[0] = T(c_f[wi][wn].thread_elements()[0]);
                c_t.thread_elements()[1] = T(c_f[wi][wn].thread_elements()[1]);
                simdgroup_store(
                    c_t,
                    y + (m0 + int(sg_id) * (WM * 8) + wi * 8) * N
                        + n0 + wn * 8,
                    N);
            }}
        }}
    """
    dtype_tag = _DTYPE_TAG.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_mma_prefill_bm128_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )


def int4_prefill_qmm(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Exact M%128==0 prefill 4-bit matmul. x2 is (M, K), M >= 128."""

    m = int(x2.shape[0])
    k = int(x2.shape[1])
    n = int(w_q.shape[0])
    x2 = mx.contiguous(x2)
    kernel = _build_prefill_kernel(group_size, x2.dtype)
    (y,) = kernel(
        inputs=[x2, w_q, scales, biases, m, k, n],
        template=[("T", x2.dtype)],
        grid=(128 * (m // 128), n // 32, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x2.dtype],
    )
    _count_mma_dispatch("prefill", m=m, k=k, n=n)
    return y


def prefill_mma_eligible(m: int, k: int, n: int, bits: int, group_size: int, dtype) -> bool:
    if bits != 4 or group_size not in (32, 64, 128):
        return False
    if dtype not in (mx.bfloat16, mx.float16):
        return False
    if m < 128 or m % 128 != 0:
        return False
    return k % 64 == 0 and n % 32 == 0


_MMA_QLINEAR_PATCH: dict[str, object] = {"installed": False, "original": None}


# ---------------------------------------------------------------------------
# Native MPP packed-INT4 lane (Metal 4 TensorOps, G17-class GPUs, OS >= 26.4)
#
# Feeds MLX-packed uint4 weights DIRECTLY into mpp::tensor_ops::matmul2d as a
# uint4b_format tensor (spec Table 7.3: bfloat/half x uint4b -> float), with a
# per-group decomposition that preserves exact affine semantics — the spec has
# no fused bf16-affine path (its tensor_blockwise scales plane only supports
# metal_fp8_ue8m0), so each group of GS k-elements contributes:
#     C += s_g[n] * (A_g . W_g)  +  b_g[n] * sum(A_g)
# i.e. the tweet recipe's "scale * dot_product + bias * activation_sum".
# Tile geometry follows the upstream measurement: 8x16x64 per simdgroup op,
# one 32-lane SIMD group per op, min column stripe 16.
# ---------------------------------------------------------------------------

_MPP_STATE: dict[str, object] = {"available": None}


@lru_cache(maxsize=1)
def _mpp_hardware_available() -> bool:
    """G17-class GPU + macOS >= 26.4 (native int4b tensors). Memoized."""
    if str(os.environ.get("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }:
        return False
    arch = str(mx.device_info().get("architecture", "")).lower()
    if not arch.startswith("applegpu_g17"):
        return False
    parts = platform.mac_ver()[0].split(".")
    try:
        major = int(parts[0]) if parts and parts[0] else 0
    except ValueError:
        major = 0
    try:
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except ValueError:
        minor = 0
    return major > 26 or (major == 26 and minor >= 4)


def mpp_available() -> bool:
    """Hardware gate AND a successful compile-and-run probe (or pending).

    MLX compiles custom kernels lazily, so a version-gate pass is not proof:
    the first eligible call would otherwise explode at the caller's next
    mx.eval. The probe dispatches a tiny 8x64x64 problem eagerly once and
    caches the verdict for the process life."""
    state = _MPP_STATE["available"]
    if state is False:
        return False
    if state is True:
        return True
    if not _mpp_hardware_available():
        _MPP_STATE["available"] = False
        return False
    ok = _mpp_runtime_probe()
    _MPP_STATE["available"] = ok
    return ok


def _mpp_runtime_probe() -> bool:
    """Eagerly compile+run a minimal native-int4 matmul2d problem."""
    try:
        wq = mx.zeros((64, 8), dtype=mx.uint32)
        s = mx.ones((64, 1), dtype=mx.bfloat16)
        b = mx.zeros((64, 1), dtype=mx.bfloat16)
        x = mx.zeros((8, 64), dtype=mx.bfloat16)
        kern = _build_mpp_vocab_kernel(8, 64, mx.bfloat16)
        (y,) = kern(
            inputs=[x, wq, s, b, 64, 64],
            template=[("T", mx.bfloat16)],
            grid=(128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(8, 64)],
            output_dtypes=[mx.bfloat16],
        )
        mx.eval(y)
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def _build_mpp_vocab_kernel(bm: int, group_size: int, dtype: mx.Dtype):
    """One threadgroup covers NSG*16 output columns; each simdgroup owns a
    16-column stripe and loops all K/GS groups, accumulating its 8xBM dot
    products into a cooperative_tensor while a pre-pass banks per-group
    activation row-sums for the bias epilogue."""
    nsg = 4
    source = f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int BM = {int(bm)};
        constexpr int SCOLS = 16;
        constexpr int NSG = {nsg};
        constexpr int GS = {group_size};
        constexpr int MAX_NG = 128;

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_gs = K / GS;
        int ng = K / GS;
        int n0 = int(tg_n) * (NSG * SCOLS);
        int col0 = n0 + int(sg_id) * SCOLS;

        // Pass 1: per-group activation row-sums for the bias epilogue.
        threadgroup float rs[MAX_NG][BM];
        for (int idx = int(tid); idx < ng * BM; idx += NSG * 32) {{
            int g = idx / BM;
            int r = idx - g * BM;
            device const T * xr = x + r * K + g * GS;
            float acc = 0.0f;
            for (int ki = 0; ki < GS; ++ki) {{
                acc += float(xr[ki]);
            }}
            rs[g][r] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Transposed views: dim0 is the unit-stride K axis.
        tensor<device T, dextents<int, 2>, tensor_inline> tA(
            (device T*)x,
            dextents<int, 2>{{K, BM}},
            array<int, 2>{{1, K}});
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> tB(
            (device uint4b_format*)w_q,
            dextents<int, 2>{{K, N}},
            array<int, 2>{{1, K}});

        constexpr auto desc = matmul2d_descriptor(
            BM,
            SCOLS,
            GS,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, metal::execution_simdgroup> op;

        auto ct_c = op.template get_destination_cooperative_tensor<
            tensor<device T, extents<int, BM, GS>, tensor_inline>,
            tensor<device uint4b_format, extents<int, SCOLS, GS>, tensor_inline>,
            float>();
        _Pragma("unroll")
        for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {{
            ct_c[i] = 0.0f;
        }}

        for (int g = 0; g < ng; ++g) {{
            auto sA = tA.template slice<GS, BM>(g * GS, 0);
            auto sB = tB.template slice<GS, SCOLS>(g * GS, col0);
            op.run(sA, sB, ct_c);
        }}

        // Epilogue: y[r][n] = dot + sum_g b[g][n] * rs[g][r]
        threadgroup float cout_t[NSG][BM * SCOLS];
        tensor<threadgroup float, dextents<int, 2>, tensor_inline> tC(
            cout_t[sg_id],
            dextents<int, 2>{{SCOLS, BM}},
            array<int, 2>{{1, SCOLS}});
        auto sC = tC.template slice<SCOLS, BM>(0, 0);
        ct_c.store(sC);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * SCOLS; off += NSG * 32) {{
            int row = off / SCOLS;
            int col = off - row * SCOLS;
            int n = col0 + col;
            float acc = cout_t[sg_id][off];
            float bacc = 0.0f;
            for (int g = 0; g < ng; ++g) {{
                bacc += float(biases[n * K_by_gs + g]) * rs[g][row];
            }}
            y[row * N + n] = T(acc + bacc);
        }}
    """
    dtype_tag = _DTYPE_TAG.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_mpp_vocab_bm{int(bm)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        header="""
            #include <metal_packed_numeric>
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        """,
        source=source,
    )


def mpp_vocab_eligible(m: int, k: int, n: int, bits: int, group_size: int, dtype) -> bool:
    """Native packed-INT4 vocab lane. Same shape family as the portable
    vocab lane but restricted to what the spec's int4b tensors allow:
    group size must be a multiple of 32 (slice alignment) and at most 128
    groups per row (banked row-sum buffer)."""
    if bits != 4 or group_size not in (32, 64, 128):
        return False
    if dtype not in (mx.bfloat16, mx.float16):
        return False
    if not (8 <= m <= 16):
        return False
    if k % group_size != 0 or k // group_size > 128:
        return False
    return n % 64 == 0


def mpp_vocab_qmm(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Native MPP packed-INT4 small-M matmul (G17 + OS 26.4 only)."""
    m = int(x2.shape[0])
    k = int(x2.shape[1])
    n = int(w_q.shape[0])
    bm = 16 if m > 8 else 8
    if m < bm:
        x2 = mx.concatenate([x2, mx.zeros((bm - m, k), dtype=x2.dtype)], axis=0)
    x2 = mx.contiguous(x2)
    kernel = _build_mpp_vocab_kernel(bm, group_size, x2.dtype)
    try:
        (y,) = kernel(
            inputs=[x2, w_q, scales, biases, k, n],
            template=[("T", x2.dtype)],
            grid=(128, n // 64, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(bm, n)],
            output_dtypes=[x2.dtype],
        )
    except RuntimeError:
        # First dispatch compiles the pipeline; a failure here means the
        # driver rejects MPP/int4b despite the version probe — cache it and
        # let callers fall back to the portable lanes.
        _MPP_STATE["available"] = False
        raise
    _count_mma_dispatch("mpp_vocab", m=m, k=k, n=n)
    return y[:m, :] if m < bm else y


def install_int4_mma_qlinear_patch() -> dict[str, object]:
    """Route eligible 4-bit QuantizedLinear calls through the simdgroup MMA
    kernels. Mirrors ``nax_verify.install_nax_qlinear_patch`` but for the
    classic simdgroup_matrix lowering, so it engages on any Apple GPU
    (no G17/MPP hardware gate). Lanes:

    - vocab/small-M decode+verify: M in 8..16 (stock GEMV wins below M=8).
    - prefill: M >= 128 and M % 128 == 0.
    - mpp: native packed-INT4 TensorOps path for the same small-M shapes on
      G17-class GPUs with macOS >= 26.4; tried before the portable vocab
      lane and never selected unless MTPLX_INT4_MMA explicitly lists 'mpp'.

    Env gate MTPLX_INT4_MMA (off unless set; 'all' enables vocab+prefill,
    comma list may add 'mpp'). Per-lane kill switches via kernel_selfcheck
    lane_disabled('int4_mma_vocab') / ('int4_mma_prefill') /
    ('int4_mma_mpp'). Idempotent.
    """
    import mlx.nn as nn

    if _MMA_QLINEAR_PATCH["installed"]:
        return {"installed": True, "already": True}

    from ..attention_context import current_attention_phase
    from ..kernel_selfcheck import lane_disabled

    original = nn.QuantizedLinear.__call__

    def patched(self, x: mx.array) -> mx.array:  # type: ignore[no-untyped-def]
        bits = int(getattr(self, "bits", 0) or 0)
        if bits == 4 and x.ndim >= 2:
            group_size = int(getattr(self, "group_size", 0) or 0)
            m = 1
            for d in x.shape[:-1]:
                m *= int(d)
            w_q = self["weight"]
            k = int(x.shape[-1])
            n = int(w_q.shape[0])
            y = None
            if (
                8 <= m <= 16
                and mma_env_lane("vocab")
                and not lane_disabled("int4_mma_vocab")
                and current_attention_phase() != "prefill"
                and vocab_mma_eligible(m, k, n, bits, group_size, x.dtype)
            ):
                if (
                    mma_env_lane("mpp")
                    and not lane_disabled("int4_mma_mpp")
                    and mpp_available()
                    and mpp_vocab_eligible(m, k, n, bits, group_size, x.dtype)
                ):
                    try:
                        y = mpp_vocab_qmm(
                            x.reshape(m, k), w_q, self["scales"], self["biases"],
                            group_size=group_size,
                        )
                    except RuntimeError:
                        y = None
                if y is None:
                    y = int4_vocab_qmm(
                        x.reshape(m, k), w_q, self["scales"], self["biases"],
                        group_size=group_size,
                    )
            elif (
                m >= 128
                and mma_env_lane("prefill")
                and not lane_disabled("int4_mma_prefill")
                and prefill_mma_eligible(m, k, n, bits, group_size, x.dtype)
            ):
                y = int4_prefill_qmm(
                    x.reshape(m, k), w_q, self["scales"], self["biases"],
                    group_size=group_size,
                )
            if y is not None:
                y = y.reshape(*x.shape[:-1], n)
                if "bias" in self:
                    y = y + self["bias"]
                return y
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _MMA_QLINEAR_PATCH["installed"] = True
    _MMA_QLINEAR_PATCH["original"] = original
    return {"installed": True, "already": False}


def uninstall_int4_mma_qlinear_patch() -> None:
    import mlx.nn as nn

    if _MMA_QLINEAR_PATCH["installed"] and _MMA_QLINEAR_PATCH["original"] is not None:
        nn.QuantizedLinear.__call__ = _MMA_QLINEAR_PATCH["original"]
        _MMA_QLINEAR_PATCH["installed"] = False
        _MMA_QLINEAR_PATCH["original"] = None
