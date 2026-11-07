"""Fixed-M4 MoE routing scaffold as two kernels (MTPLX_QWEN4_ROUTE_KERNEL).

WHAT IT REPLACES
----------------
Every one of the 48 MoE blocks runs the same ten-dispatch routing head between
the hyper read and the retained gather kernels.  From the Step-8 census
(``G-opdiet-census.md`` §2.1, rows 35-50; ``qwen4_m4_stage3.py:89-97`` and
``:127``):

  35  affine_qmv_wide_gs_64_b_8        [1,64,1]   block.gate(x)          q8/g64
  36  block_softmax_precise_bfloat16   [512,1,1]  mx.softmax(precise)
  37  arangeuint32                     [4,1,1]    argpartition scaffold
  38  carg_block_sort_bf16_u32_bn128_tn4 [1,4,1]  mx.argpartition(kth=-10)
  42  arangeuint32                     [40,1,1]   take_along_axis scaffold
  43  gather_axisbfloat16uint32        [1,10,4]   mx.take_along_axis
  45  affine_qmv_wide_gs_64_b_8        [1,1,1]    block.shared_expert_gate(x)
  46  row_reduce_small_1_reduce_sum    [4,1,1]    route_scores.sum(-1)
  49  v_Sigmoidbfloat16                [4,1,1]    mx.sigmoid(shared logit)
  50  CV2IBroadcastBD..Divide          [10,4,1]   route_scores / sum

Ten dispatches per layer, 480 per verify cycle, to produce 44 numbers and 40
indices.  This module emits the same
``(expert_ids[4,10] uint32, route_scores[4,10] bf16, shared_factor[4] bf16)``
tuple -- exactly what ``qwen4_m4_routed_glu`` and ``qwen4_m4_routed_down``
consume -- in TWO dispatches: **-8 launches/layer, -384/cycle**.

WHY TWO AND NOT ONE
-------------------
The softmax normaliser and the top-10 both reduce over all 512 experts of a
row, so a single dispatch would have to hold the whole row in one threadgroup.
That threadgroup would then also own the router GEMV -- 1.393 MB of q8 weights
read by 4 threadgroups (one per verifier row) instead of 64.  A single
threadgroup sustains order 20-40 GB/s on this part, so the fused-into-one
arrangement lands near 100 GB/s against the 294 GB/s the split GEMV measures
today: roughly 3x SLOWER on the byte-moving half to save one 0.8 us dispatch.
The split is K1 = the GEMV at MLX's own occupancy or better, K2 = the whole
44-number tail in one 4-threadgroup kernel.

BIT-EXACTNESS -- WHAT IS CLAIMED, AND FROM WHAT
-----------------------------------------------
Every boundary below is transcribed from the MLX 0.32.2 headers shipped in the
interpreter's own site-packages
(``mlx/include/mlx/backend/metal/kernels/{quantized,softmax,sort,unary_ops,
reduction/reduce_row}.h``), not from memory.  The claim is bit-identity, not a
rounding class -- a flipped near-tie changes the visible expert set, which is
not a rounding-class difference.

1. ROUTER GEMV -- ``qmv_wide_impl<bfloat16_t, 64, 8, vecs_per_tg, k_lanes>``
   (quantized.h:989).  The trace grid ``[1,64,1]`` is 64 threadgroups over
   512 output rows = 8 rows/threadgroup; the impl fixes
   ``num_simdgroups = 2`` and ``results_per_simdgroup = 32 / k_lanes``, so
   ``2 * 32 / k_lanes == 8`` pins **k_lanes = 8**.  Per output row and vector:

     lane k_lane owns groups g = k_lane, k_lane+8, ... < 40 (5 of the 40)
     per group: scale/bias widened to fp32 once, then 8 sub-chunks of 8
     per sub-chunk: w_dq[i] = fp32(scale * q[i] + bias)     (dequantize<>, bits==8)
                    acc = sum_{i=0..7} fp32(x[i]) * w_dq[i] (sequential)
                    result += acc                           (fp32)
     then a shuffle ladder down(4), down(2), down(1) -- NOT simd_sum
     then one static_cast<bfloat> on the store.

   ``vecs_per_tg`` never enters the arithmetic: each vector accumulates
   independently.  That is what lets ``VEC_LANES`` below add threads without
   moving a single rounding boundary.

2. SHARED-EXPERT GATE -- the same ``affine_qmv_wide`` instantiation at
   ``out_vec_size = 1`` (trace grid ``[1,1,1]``), folded into K1 as virtual
   output row 512 (the folded-inject trick from ``qwen4_m4_hyper_read``) and
   written to a separate ``[4]`` output so the ``[4,512]`` logits stay
   contiguous for K2.

   This is the one place the k_lanes = 8 reading is not pinned by a grid:
   ``ceil(1 / rows_per_tg) == 1`` for any rows_per_tg, so ``[1,1,1]`` says
   nothing. The supporting evidence is that EVERY other ``affine_qmv_wide``
   dispatch in the census runs 8 output rows per threadgroup regardless of N
   and of the quantization -- ``[1,320,1]`` at N=2560 (out_proj and shared
   down), ``[1,160,1]`` at N=1280 (shared gu), ``[1,2060,1]`` at N=16480
   (fused GDN in_proj, q4/g32) -- i.e. ``results_per_simdgroup`` is a constant
   of the instantiation, not a function of the output width. If that is wrong
   for N == 1, the folded row's accumulation order differs and the install
   gate's ``array_equal`` on ``shared_factor`` fails loudly on layer 0.

3. SOFTMAX -- ``softmax_single_row<bfloat16_t, float, SOFTMAX_N_READS=4>``
   (softmax.h), dispatched as 4 threadgroups x 128 threads (trace grid
   ``[512,1,1]`` is threads; 512/4 = 128 threads/row).  K2 reproduces the
   layout thread-for-thread so ``simd_max``/``simd_sum`` see the same values in
   the same lanes: per-thread max over 4 contiguous elements, simd_max, a
   32-slot threadgroup array pre-filled with -inf, a second simd_max in
   simdgroup 0; then ``fast::exp`` (softmax.h's ``softmax_exp``, explicitly the
   fast flavour), a sequential 4-term fp32 partial, simd_sum, the same
   threadgroup relay, and ``bfloat(ld[i] * (1/normalizer))``.

4. TOP-10 -- ``mx.argpartition`` is routed to the sort kernel on Metal; the
   trace kernel is ``carg_block_sort_bfloat16_uint32_bn128_tn4``, i.e. a FULL
   argsort of the 512-wide axis (128 threads x 4 per thread = 512 = one block,
   no multi-block merge).  ``BlockMergeSort`` (sort.h) is STABLE ascending:
   ``ThreadSort`` swaps only on strict ``op(vals[j+1], vals[j])``,
   ``merge_step`` takes from B only when ``op(b, a)`` is strict, and
   ``merge_partition`` is the matching stable merge-path search.  Indices enter
   as ``tgp_idxs[i] = i``.

   So ``argpartition(gates, kth=-10)[..., -10:]`` is exactly
   ``argsort(gates)[..., -10:]``: the ten greatest under the lexicographic key
   (value, index), emitted in ASCENDING (value, index) order.  Ties resolve to
   the HIGHER index -- slot 9 is the largest value and, among equal largest,
   the largest index.

   K2 encodes that key directly: ``(order_preserving_u16(gate) << 16) | index``
   in a uint32, then ten rounds of threadgroup max-extract writing slot
   ``9 - s``.  The map is exact for every non-negative finite bfloat, which is
   every softmax output.  (For a NEGATIVE NaN it would disagree with sort.h's
   NaN-last rule; softmax cannot emit one, and the install-time self-check
   compares against the stock scaffold on the real weights.)

   NOTE FOR TESTS: this is the METAL rule.  MLX's CPU ``argpartition`` is an
   unstable partial selection -- same SET, different slot ORDER (proved in
   tests/test_qwen4_route_kernel.py).  ``mx.argsort`` is stable on both
   backends, so the CPU-side reference for this kernel is argsort, never
   argpartition.

5. RENORMALISE -- ``row_reduce_small_1_reduce_sumbfloat16`` instantiates
   T == U == bfloat16 (reduce_row.h ``thread_reduce``: N_READS=4 blocks then
   the remainder, strictly ascending index order), so the 10-term sum
   accumulates IN bf16 and rounds at every add.  Then ``Divide`` (binary_ops.h
   ``return x / y`` at T = bfloat) widens to fp32 for the divide and rounds
   once on the store.  Both are pinned by CPU tests.

6. SHARED GATE SIGMOID -- ``unary_ops.h`` ``Sigmoid`` at T = bfloat16_t:

       auto y = 1 / (1 + metal::exp(metal::abs(x)));
       return (x < 0) ? y : 1 - y;

   ``metal::abs``/``metal::exp`` on bfloat are the bf16_math.h overloads that
   round the result back to bfloat, so ``exp`` rounds to bf16 BEFORE the
   reciprocal; the ``1 +`` and ``1 -`` then run in fp32 (bfloat promotes) and
   round once at the return.  An fp32 sigmoid rounded once disagrees on ~13%
   of bf16 inputs (measured on the CPU backend, test in the suite).  This is
   the same idiom the retained ``qwen4_m4_routed_glu`` kernel already ships.

RESIDUAL RISK (cannot be closed without a GPU)
----------------------------------------------
``metal::exp`` and ``1.0f/x`` resolve to ``__METAL_MAYBE_FAST_MATH__``; the
kernel therefore uses the SAME spelling MLX uses (plain ``metal::exp``, plain
``fast::exp`` inside the softmax) so both compile under whatever flag
``mx.fast.metal_kernel`` and the shipped metallib share.  Metal's ``simd_sum``
and ``simd_max`` reduction trees are reproduced by CALLING the same intrinsics
with the same lane assignment rather than by modelling them.  The gate is
construction-bound: ``install`` compares the kernel's whole tuple against the
stock scaffold on the real per-layer weights with ``mx.array_equal`` and raises
on the first mismatch, so an untrue assumption fails loudly at model build
instead of silently changing which experts run.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx

from mtplx.compile_state import compiled_step_body

ROWS = 4
TOP_K = 10
HIDDEN = 2560
NUM_EXPERTS = 512
GROUP_SIZE = 64
BITS = 8
GROUPS_PER_ROW = HIDDEN // GROUP_SIZE  # 40
BYTES_PER_ROW = HIDDEN * BITS // 8  # 2560

#: Fixed by bit-exactness, not by tuning.  ``qmv_wide_impl`` hardcodes
#: ``num_simdgroups = 2`` and derives ``results_per_simdgroup`` from
#: ``k_lanes``; the trace's 8 output rows per threadgroup leave k_lanes = 8 as
#: the only solution, and k_lanes is what fixes which of the 40 groups each
#: lane accumulates.  Changing it changes the reduction order.
K_LANES = 8
ROWS_PER_TG = 8
NTOT = NUM_EXPERTS + 1  # + the folded shared-expert gate row

#: Threads that share one output row's k-stripe set.  ``1`` is MLX's own
#: layout (all four verifier vectors live in one lane's registers, 4,160
#: threads); ``4`` gives each vector its own lane octet (16,640 threads) for
#: the same per-vector accumulation order -- bit-identical, four times the
#: threads, and the weight octet each vector lane loads is the same address in
#: the same threadgroup on the same cycle, so DRAM traffic is unchanged.
#: Which one is faster is a measurement (the PR #391 harness micro_route_kernel.py);
#: neither can change a number.
VEC_LANES_CHOICES = (1, 4)
DEFAULT_VEC_LANES = 4

#: Softmax geometry, transcribed from softmax_single_row.
SOFTMAX_N_READS = 4
TAIL_THREADS = NUM_EXPERTS // SOFTMAX_N_READS  # 128

#: Dispatches this module removes, for the microbench's launch column.
EAGER_DISPATCHES_PER_LAYER = 10
FUSED_DISPATCHES_PER_LAYER = 2

_LOGITS_KERNEL: Any | None = None
_TAIL_KERNEL: Any | None = None
_TAIL_FP16_KERNEL: Any | None = None


def router_bytes_per_layer() -> int:
    """q8 router pack + the folded shared gate, in bytes.

    1,392,640 for the router (the census's ``router 1.393 MB``) plus 2,720 for
    the shared-expert gate.  This is the DRAM floor K1 reads once per layer at
    any ``VEC_LANES``.
    """

    router = (
        NUM_EXPERTS * BYTES_PER_ROW + 2 * NUM_EXPERTS * GROUPS_PER_ROW * 2
    )
    shared = BYTES_PER_ROW + 2 * GROUPS_PER_ROW * 2
    return router + shared


_LOGITS_HEADER = """
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;
"""


# ---------------------------------------------------------------------------
# K1 -- router GEMV + folded shared-expert-gate GEMV.
#
# Transcribes qmv_wide_impl<bfloat16_t, 64, 8, ., k_lanes = 8>.  VEC_LANES
# only decides how many of the four verifier vectors a lane keeps in
# registers; the per-(row, vector) accumulation is byte-for-byte the same
# sequence in both arms.
# ---------------------------------------------------------------------------
_LOGITS_SOURCE = """
    constexpr int RK = 2560;
    constexpr int RGS = 64;
    constexpr int RNG = RK / RGS;                // 40
    constexpr int RSUB = 8;
    constexpr int RNEXP = 512;
    constexpr int RNTOT = RNEXP + 1;
    constexpr int RKLANES = 8;
    constexpr int RROWS_PER_TG = 8;
    constexpr int RVECS = 4;

    constexpr int VL = VEC_LANES;
    constexpr int RV = RVECS / VL;               // vectors per lane
    constexpr int RPR = RKLANES * VL;            // threads per output row

    constexpr int RPS = 32 / RPR;                // output rows per simdgroup

    // Indexed off the simdgroup attributes, not off a linear thread id, so the
    // eight k-lanes that feed one shuffle ladder are provably inside one
    // simdgroup. At VL == 1 this is qmv_wide_impl's own mapping verbatim
    // (k_lane = simd_lid % 8, sg_row = simd_lid / 8, four rows per simdgroup,
    // two simdgroups per threadgroup).
    const uint blk = threadgroup_position_in_grid.x;
    const uint sgid = simdgroup_index_in_threadgroup;
    const uint slid = thread_index_in_simdgroup;

    const int row_in_tg = int(sgid) * RPS + int(slid) / RPR;
    const int rem = int(slid) % RPR;
    const int v_lane = rem / RKLANES;
    const int k_lane = rem % RKLANES;

    const int out_row = int(blk) * RROWS_PER_TG + row_in_tg;
    // qmv_wide_impl clamps with min(out_row, out_vec_size - 1) and drops the
    // store; the clamped lanes read a valid row so the loop stays branch-free.
    const int row = out_row < RNTOT ? out_row : (RNTOT - 1);
    const bool is_shared = (row == RNEXP);

    const device uchar* wrow = is_shared
        ? ((const device uchar*)gate_weights)
        : ((const device uchar*)router_weights + (size_t)row * RK);
    const device bfloat* srow = is_shared
        ? gate_scales
        : (router_scales + (size_t)row * RNG);
    const device bfloat* brow = is_shared
        ? gate_biases
        : (router_biases + (size_t)row * RNG);

    float result[RV];
    #pragma clang loop unroll(full)
    for (int j = 0; j < RV; ++j) result[j] = 0.0f;

    for (int g = k_lane; g < RNG; g += RKLANES) {
        const float scale = float(srow[g]);
        const float bias = float(brow[g]);
        #pragma clang loop unroll(full)
        for (int sc = 0; sc < RGS / RSUB; ++sc) {
            const int k0 = g * RGS + sc * RSUB;
            const device uchar* wc = wrow + k0;
            float w_dq[RSUB];
            #pragma clang loop unroll(full)
            for (int i = 0; i < RSUB; ++i) {
                // dequantize<float, 8, 8>: static_cast<U>(s * w[i] + b)
                w_dq[i] = float(scale * wc[i] + bias);
            }
            #pragma clang loop unroll(full)
            for (int j = 0; j < RV; ++j) {
                const int v = v_lane * RV + j;
                const device bfloat* xc = x + (size_t)v * RK + k0;
                float acc = 0.0f;
                #pragma clang loop unroll(full)
                for (int i = 0; i < RSUB; ++i) {
                    acc += float(xc[i]) * w_dq[i];
                }
                result[j] += acc;
            }
        }
    }

    // The k_lanes == 8 arm of qmv_wide_impl's shuffle ladder, verbatim.
    // simd_sum would mix the four output rows a simdgroup spans.
    #pragma clang loop unroll(full)
    for (int j = 0; j < RV; ++j) {
        result[j] += simd_shuffle_down(result[j], 4);
        result[j] += simd_shuffle_down(result[j], 2);
        result[j] += simd_shuffle_down(result[j], 1);
    }

    if (k_lane == 0 && out_row < RNTOT) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < RV; ++j) {
            const int v = v_lane * RV + j;
            if (is_shared) {
                shared_logits[v] = bfloat(result[j]);
            } else {
                logits[(size_t)v * RNEXP + out_row] = bfloat(result[j]);
            }
        }
    }
"""


_TAIL_HEADER = """
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    // Order-preserving map from a bfloat to a uint16 so that
    //   key(a) < key(b)  <=>  a < b
    // for every non-NaN bfloat.  Softmax outputs are finite and >= 0, so the
    // sign branch is only exercised by the tests' constructed inputs.
    inline uint route_sort_key(bfloat v, uint idx) {
        const ushort u = as_type<ushort>(v);
        const ushort k = (u & 0x8000) ? (ushort)(~u) : (ushort)(u | 0x8000);
        return ((uint)k << 16) | idx;
    }
"""


# ---------------------------------------------------------------------------
# K2 -- softmax -> stable top-10 -> gather -> bf16 renormalise -> shared
# sigmoid, one threadgroup per verifier row.
# ---------------------------------------------------------------------------
_TAIL_SOURCE = """
    constexpr int TNEXP = 512;
    constexpr int TTOPK = 10;
    constexpr int TNR = 4;
    constexpr int TNT = 128;
    constexpr int TNSG = TNT / 32;

    const uint gid = threadgroup_position_in_grid.x;      // verifier row
    const uint lid = thread_position_in_threadgroup.x;
    const uint slid = thread_index_in_simdgroup;
    const uint sgid = simdgroup_index_in_threadgroup;

    threadgroup float local_max[32];
    threadgroup float local_norm[32];
    threadgroup uint keys[TNEXP];
    threadgroup bfloat gate_values[TNEXP];
    threadgroup uint partial[TNSG];
    threadgroup uint selected[TTOPK];

    const device bfloat* in = logits + (size_t)gid * TNEXP + lid * TNR;

    // ---- softmax_single_row<bfloat16_t, float, 4>, layout-for-layout -----
    float ld[TNR];
    #pragma clang loop unroll(full)
    for (int i = 0; i < TNR; ++i) ld[i] = float(in[i]);

    if (sgid == 0) {
        local_max[slid] = -metal::numeric_limits<float>::infinity();
        local_norm[slid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Limits<AccT>::finite_min, exactly as softmax_single_row seeds it.
    float maxval = -metal::numeric_limits<float>::max();
    #pragma clang loop unroll(full)
    for (int i = 0; i < TNR; ++i) maxval = (maxval < ld[i]) ? ld[i] : maxval;
    maxval = simd_max(maxval);
    if (slid == 0) local_max[sgid] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        maxval = simd_max(local_max[slid]);
        if (slid == 0) local_max[0] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];

    float normalizer = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < TNR; ++i) {
        // softmax.h softmax_exp() is fast::exp, explicitly.
        const float e = fast::exp(ld[i] - maxval);
        ld[i] = e;
        normalizer += e;
    }
    normalizer = simd_sum(normalizer);
    if (slid == 0) local_norm[sgid] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        normalizer = simd_sum(local_norm[slid]);
        if (slid == 0) local_norm[0] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1.0f / local_norm[0];

    #pragma clang loop unroll(full)
    for (int i = 0; i < TNR; ++i) {
        const int idx = int(lid) * TNR + i;
        const bfloat g = bfloat(ld[i] * normalizer);
        gate_values[idx] = g;
        keys[idx] = route_sort_key(g, (uint)idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- top-10 under the stable ascending sort's (value, index) key ----
    // Slot 9 is written first: the stock path slices the LAST ten of an
    // ascending argsort, so slot j holds the (10 - j)-th largest.
    for (int s = 0; s < TTOPK; ++s) {
        uint best = 0u;
        #pragma clang loop unroll(full)
        for (int i = 0; i < TNR; ++i) {
            best = max(best, keys[lid * TNR + i]);
        }
        best = simd_max(best);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (slid == 0) partial[sgid] = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint winner = partial[0];
        for (int g = 1; g < TNSG; ++g) winner = max(winner, partial[g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            selected[TTOPK - 1 - s] = winner;
            // Keys are unique in their low 16 bits, and every real key has the
            // order-map's sign bit set, so 0 can never win a later round.
            keys[winner & 0xFFFFu] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- gather, bf16 renormalise, shared-gate sigmoid ------------------
    if (lid == 0) {
        bfloat picked[TTOPK];
        #pragma clang loop unroll(full)
        for (int j = 0; j < TTOPK; ++j) {
            const uint idx = selected[j] & 0xFFFFu;
            expert_ids[gid * TTOPK + j] = idx;
            picked[j] = gate_values[idx];
        }
        // row_reduce_small instantiates T == U == bfloat: Sum::init, then one
        // rounded bf16 add per element in ascending slot order.
        bfloat total = bfloat(0.0f);
        #pragma clang loop unroll(full)
        for (int j = 0; j < TTOPK; ++j) total = picked[j] + total;
        #pragma clang loop unroll(full)
        for (int j = 0; j < TTOPK; ++j) {
            route_scores[gid * TTOPK + j] = picked[j] / total;
        }
#ifndef ROUTE_TAIL_ONLY
        // unary_ops.h Sigmoid at T = bfloat16_t, exactly as spelled there.
        const bfloat sx = shared_logits[gid];
        auto sigmoid_y = 1 / (1 + metal::exp(metal::abs(sx)));
        shared_factor[gid] = sx < bfloat(0.0f)
            ? bfloat(sigmoid_y)
            : bfloat(1 - sigmoid_y);
#endif
    }
"""


def logits_source() -> str:
    """Return the exact folded router/shared-gate q8 GEMV source."""

    return _LOGITS_HEADER + _LOGITS_SOURCE


def tail_source() -> str:
    """Return the exact softmax/top-10/renormalise/sigmoid tail source."""

    return _TAIL_HEADER + _TAIL_SOURCE


def logits_geometry(
    vec_lanes: int = DEFAULT_VEC_LANES,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return ``(grid, threadgroup)`` in THREADS for K1."""

    if vec_lanes not in VEC_LANES_CHOICES:
        raise ValueError(
            f"route kernel vec_lanes={vec_lanes}: want one of {VEC_LANES_CHOICES}"
        )
    threads = ROWS_PER_TG * K_LANES * vec_lanes
    blocks = (NTOT + ROWS_PER_TG - 1) // ROWS_PER_TG
    return ((threads * blocks, 1, 1), (threads, 1, 1))


def tail_geometry() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return ``(grid, threadgroup)`` in THREADS for K2."""

    return ((TAIL_THREADS * ROWS, 1, 1), (TAIL_THREADS, 1, 1))


def _projection_fields(projection: Any, name: str) -> tuple[Any, Any, Any]:
    weight = getattr(projection, "weight", None)
    scales = getattr(projection, "scales", None)
    biases = getattr(projection, "biases", None)
    if weight is None or scales is None or biases is None:
        raise ValueError(
            f"MTPLX_QWEN4_ROUTE_KERNEL: {name} is not an affine-quantized "
            "projection; the kernel reads a q8/g64 pack directly"
        )
    return weight, scales, biases


def _check_routing(block: Any, label: str) -> None:
    if int(getattr(block, "num_experts", -1)) != NUM_EXPERTS:
        raise ValueError(
            f"{label}: requires {NUM_EXPERTS} experts, got "
            f"{getattr(block, 'num_experts', None)}"
        )
    if int(getattr(block, "top_k", -1)) != TOP_K:
        raise ValueError(
            f"{label}: requires exact top-k {TOP_K}, got "
            f"{getattr(block, 'top_k', None)}"
        )
    if not bool(getattr(block, "norm_topk_prob", False)):
        raise ValueError(
            f"{label}: requires top-k probability normalization; the kernel "
            "always renormalises"
        )


def check_contract(block: Any, *, index: int) -> None:
    """Validate the router and shared-gate packs. Raises -- never returns False.

    Everything the two kernels hardcode is checked here: expert count, top-k,
    the ``norm_topk_prob`` renormalise, and both q8/g64 packs down to dtype and
    shape. An armed flag on a pack the kernel cannot serve fails at model build
    with the offending field named, never mid-forward and never by silently
    reverting to the ten-dispatch scaffold.
    """

    label = f"MTPLX_QWEN4_ROUTE_KERNEL layer {index}"
    _check_routing(block, label)
    specs = (
        ("mlp.gate", block.gate, (NUM_EXPERTS, BYTES_PER_ROW // 4)),
        (
            "mlp.shared_expert_gate",
            block.shared_expert_gate,
            (1, BYTES_PER_ROW // 4),
        ),
    )
    for name, projection, weight_shape in specs:
        if int(getattr(projection, "bits", -1)) != BITS:
            raise ValueError(
                f"{label}: {name} bits={getattr(projection, 'bits', None)} "
                f"(want {BITS})"
            )
        if int(getattr(projection, "group_size", -1)) != GROUP_SIZE:
            raise ValueError(
                f"{label}: {name} group_size="
                f"{getattr(projection, 'group_size', None)} (want {GROUP_SIZE})"
            )
        if str(getattr(projection, "mode", "")) != "affine":
            raise ValueError(
                f"{label}: {name} mode={getattr(projection, 'mode', None)!r} "
                "(want 'affine')"
            )
        weight, scales, biases = _projection_fields(projection, f"{label}: {name}")
        meta_shape = (weight_shape[0], GROUPS_PER_ROW)
        for field, arr, want, dtype in (
            ("weight", weight, weight_shape, mx.uint32),
            ("scales", scales, meta_shape, mx.bfloat16),
            ("biases", biases, meta_shape, mx.bfloat16),
        ):
            if tuple(arr.shape) != want:
                raise ValueError(
                    f"{label}: {name}.{field} shape {tuple(arr.shape)} "
                    f"(want {want})"
                )
            if arr.dtype != dtype:
                raise ValueError(
                    f"{label}: {name}.{field} dtype {arr.dtype} (want {dtype})"
                )


def _check_tail_contract(block: Any, *, index: int) -> None:
    label = f"MTPLX_QWEN4_ROUTE_KERNEL_FP16 layer {index}"
    _check_routing(block, label)
    scales = getattr(block.gate, "scales", None)
    if scales is None or scales.dtype != mx.float16:
        raise ValueError(
            f"{label}: mlp.gate is not an FP16-scaled quantized projection"
        )


def check_input(x: mx.array) -> None:
    """Validate the hidden-state view handed to the route kernel."""

    rows = 1
    for s in x.shape[:-1]:
        rows *= int(s)
    if x.shape[-1] != HIDDEN or rows != ROWS:
        raise ValueError(
            f"MTPLX_QWEN4_ROUTE_KERNEL is wired for exactly [{ROWS}, {HIDDEN}]; "
            f"got {tuple(x.shape)}"
        )
    if x.dtype != mx.bfloat16:
        raise ValueError(
            f"MTPLX_QWEN4_ROUTE_KERNEL: hidden dtype {x.dtype} (want bfloat16)"
        )


def bind(vec_lanes: int = DEFAULT_VEC_LANES) -> Callable[..., tuple]:
    """Bind the two-dispatch route head.

    The returned callable takes the MoE block's own hidden view and the two
    quantized packs and returns the tuple the retained kernels consume:
    ``(expert_ids [4,10] uint32, route_scores [4,10] bf16,
    shared_factor [4] bf16)``.
    """

    if vec_lanes not in VEC_LANES_CHOICES:
        raise ValueError(
            f"route kernel vec_lanes={vec_lanes}: want one of {VEC_LANES_CHOICES}"
        )

    global _LOGITS_KERNEL, _TAIL_KERNEL
    if _LOGITS_KERNEL is None:
        _LOGITS_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_route_logits",
            input_names=[
                "x",
                "router_weights",
                "router_scales",
                "router_biases",
                "gate_weights",
                "gate_scales",
                "gate_biases",
            ],
            output_names=["logits", "shared_logits"],
            header=_LOGITS_HEADER,
            source=_LOGITS_SOURCE,
            ensure_row_contiguous=True,
        )
    if _TAIL_KERNEL is None:
        _TAIL_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_route_tail",
            input_names=["logits", "shared_logits"],
            output_names=["expert_ids", "route_scores", "shared_factor"],
            header=_TAIL_HEADER,
            source=_TAIL_SOURCE,
            ensure_row_contiguous=True,
        )
    logits_kernel = _LOGITS_KERNEL
    tail_kernel = _TAIL_KERNEL
    logits_grid, logits_threadgroup = logits_geometry(vec_lanes)
    tail_grid, tail_threadgroup = tail_geometry()

    def route(
        x,
        router_weights,
        router_scales,
        router_biases,
        gate_weights,
        gate_scales,
        gate_biases,
    ):
        logits, shared_logits = logits_kernel(
            inputs=[
                x,
                router_weights,
                router_scales,
                router_biases,
                gate_weights,
                gate_scales,
                gate_biases,
            ],
            template=[("VEC_LANES", vec_lanes)],
            grid=logits_grid,
            threadgroup=logits_threadgroup,
            output_shapes=[(ROWS, NUM_EXPERTS), (ROWS,)],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
        expert_ids, route_scores, shared_factor = tail_kernel(
            inputs=[logits, shared_logits],
            grid=tail_grid,
            threadgroup=tail_threadgroup,
            output_shapes=[(ROWS, TOP_K), (ROWS, TOP_K), (ROWS,)],
            output_dtypes=[mx.uint32, mx.bfloat16, mx.bfloat16],
        )
        return expert_ids, route_scores, shared_factor

    return route


def _bind_tail_fp16() -> Callable[[mx.array], tuple]:
    global _TAIL_FP16_KERNEL
    if _TAIL_FP16_KERNEL is None:
        _TAIL_FP16_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_route_tail_fp16",
            input_names=["logits"],
            output_names=["expert_ids", "route_scores"],
            header="#define ROUTE_TAIL_ONLY\n" + _TAIL_HEADER.replace("bfloat", "half"),
            source=_TAIL_SOURCE.replace("bfloat", "half"),
            ensure_row_contiguous=True,
        )
    tail_grid, tail_threadgroup = tail_geometry()

    def tail(logits):
        return _TAIL_FP16_KERNEL(
            inputs=[logits],
            grid=tail_grid,
            threadgroup=tail_threadgroup,
            output_shapes=[(ROWS, TOP_K), (ROWS, TOP_K)],
            output_dtypes=[mx.uint32, mx.float16],
        )

    return tail


def _check_tail_exact(block: Any, tail: Callable, probes: list, *, index: int) -> None:
    from mtplx.qwen4_m4_stage3 import _same_bits

    stock = mx.compile(lambda h: block._verify_route(h))
    fused = mx.compile(lambda h: block._verify_route(h, tail))
    for x in probes:
        with compiled_step_body():
            checks = [_same_bits(want, got) for want, got in zip(stock(x), fused(x))]
        mx.eval(checks)
        for name, ok in zip(("expert_ids", "route_scores"), checks):
            if not ok.item():
                raise ValueError(
                    f"MTPLX_QWEN4_ROUTE_KERNEL_FP16 layer {index}: {name} is "
                    "not bit-exact with the stock routing scaffold"
                )


def install_tail_fp16(inner: Any) -> None:
    blocks = [layer.mlp for layer in inner.layers]
    for index, block in enumerate(blocks):
        _check_tail_contract(block, index=index)
    tail = _bind_tail_fp16()
    width = int(inner.args.hidden_size)
    sample = mx.sin(mx.arange(4 * width, dtype=mx.float32) * 0.0009765625)
    sample = sample.reshape(1, 4, width)
    noise = mx.random.normal(sample.shape, key=mx.random.key(0))
    probes = [
        rows.astype(mx.float16)
        for rows in (sample, noise, noise * 0.001, mx.zeros_like(sample))
    ]
    for index, block in enumerate(blocks):
        _check_tail_exact(block, tail, probes, index=index)
    for block in blocks:
        block._mtplx_route_tail = tail


def stock_route(block: Any, x: mx.array) -> tuple:
    """The ten-dispatch scaffold, isolated, as the kernel's reference.

    Byte-for-byte the head of ``_m4_paired_routed_glu_residual_tail_forward``
    (``qwen4_m4_stage3.py:89-97``, ``:127``). Kept here so the install-time
    self-check and the microbench compare against ONE definition.
    """

    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(ROWS)
    return (
        expert_ids.reshape(ROWS, TOP_K),
        route_scores.reshape(ROWS, TOP_K).astype(mx.bfloat16),
        shared_factor.astype(mx.bfloat16),
    )


def metal_order_reference(block: Any, x: mx.array) -> tuple:
    """``stock_route`` with ``argsort`` in place of ``argpartition``.

    On Metal the two are the SAME kernel (``carg_block_sort``), so this is
    identical to ``stock_route`` there. On the CPU stream it is not:
    ``argpartition`` is an unstable partial selection, so only this form
    reproduces the slot order the retained gather kernels are validated
    against. Tests compare the kernel's semantics against this.
    """

    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argsort(gates, axis=-1)[..., -block.top_k :]
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(ROWS)
    return (
        expert_ids.reshape(ROWS, TOP_K).astype(mx.uint32),
        route_scores.reshape(ROWS, TOP_K).astype(mx.bfloat16),
        shared_factor.astype(mx.bfloat16),
    )


__all__ = [
    "DEFAULT_VEC_LANES",
    "EAGER_DISPATCHES_PER_LAYER",
    "FUSED_DISPATCHES_PER_LAYER",
    "K_LANES",
    "NUM_EXPERTS",
    "ROWS",
    "TOP_K",
    "VEC_LANES_CHOICES",
    "bind",
    "check_contract",
    "check_input",
    "install_tail_fp16",
    "logits_geometry",
    "logits_source",
    "metal_order_reference",
    "router_bytes_per_layer",
    "stock_route",
    "tail_geometry",
    "tail_source",
]
