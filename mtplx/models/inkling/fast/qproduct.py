"""Row-stable products of affine-coded matrices (the Bare Speed pack's output projection and
vocabulary head; any 4- or 8-bit matrix in MLX's layout), one launch for a panel of rows.

``affine_product(x, weight)``: ``x`` ``(..., K)`` bf16, float16 or float32 with ``W`` rows (the head takes
float32, as ``head.vocabulary_logits`` feeds it; the output projection bf16), ``weight`` an affine
weight (``precision.AffineWeight``: uint32 codes ``(N, K * bits / 32)``, bf16 scales and biases
``(N, K / group)``). Returns ``(..., N)`` in ``x``'s dtype whose every row equals, bit for bit, MLX
0.32.2's one-row ``mx.quantized_matmul`` of that row (for float32 inputs MLX casts the scales and
biases to float32 first, two extra launches; here they are read in their stored type, bf16 or float16 (the
FP16 sibling packs), the same values). MLX's own product of 2 to 24 rows does not
(``affine_qmv_wide`` from M3 on, another order), so a verify panel through MLX differs from the
one-row step; through this launch it does not. One row takes this launch too (``_ONE_ROW``), proven
like every width; where it refuses, MLX's product (the reference at one row), counted.

Exactness: each output row repeats the one-row kernel MLX picks for the matrix's shape (its
``dispatch_qmv`` rule: ``qmv_fast`` when N is a multiple of 8 and K of 512 at 4 bits or 256 at 8
bits, else ``qmv``), lane for lane: lane ``l`` takes ``VPT`` consecutive values of each block of
``32 VPT`` columns (4 bits: 8 values in ``qmv``, 16 in ``qmv_fast``; 8 bits: 4 and 8), with the
reference's per-lane arithmetic as ``affine_lane.AFFINE_LANE`` writes it (scaled inputs and their
total, the codes in place, one term a block), the terms added in block order, then summed across
the simdgroup. The codes of an output row are unpacked once per block and shared by every panel
row. A first use on each shape proves the result against MLX's one-row product on the real
operands; a disagreement refuses the route (counted) and the caller keeps MLX's product.

``QMV_ROWS`` is the same arithmetic as a Metal device function for another lane's launch (the
fused attention output): ``qmv_rows<W, NROW, K, BITS, GROUP, VPT>(codes, scales, biases, x, x_stride,
row0, lane, result)`` leaves ``result[r][ro]`` (output row ``row0 + ro``, panel row ``r``) in every
lane of the calling simdgroup; ``qmv_constants(weight)`` gives its template values.
"""

from __future__ import annotations

from functools import lru_cache
import struct

import mlx.core as mx

from .affine_lane import AFFINE_LANE

#: Threads of one threadgroup (simdgroups x 4 output rows each), by output rows: a small matrix (an
#: output projection, 2,304 rows) 128, a large one (the head, 201,024 rows) 256. YSPLIT (the panel's
#: rows split over threadgroup rows) measured slower on the output projection (qproduct-bench-2:
#: 58.6 against 40.7 us at 4 rows), so it stays off by default.
_THREADS_SMALL = 128
_THREADS_LARGE = 256
_SMALL_ROWS = 16384
#: Panel rows one pass accumulates (each pass decodes the codes again): 8 for a small matrix, 4 for
#: the head (qproduct-bench-2, arms interleaved: head 856 us at 4 rows and 3,093 at 8, against
#: 1,474 and 3,796 for the one-row products and 803 and 4,514 for MLX's own product of the rows).
_RCH_SMALL = 8
_RCH_LARGE = 4
#: Output rows one simdgroup computes (each row's arithmetic does not depend on it). A small matrix
#: (the output projection, 2,304 rows) takes 2: twice the simdgroups in flight, measured on the engine
#: (GPU time of a 48-visit dependent chain, receipts/ebench-oproj-1.json, M5 Max): 2 rows 19.2 -> 10.3
#: us, 3 rows 27.2 -> 15.8, 4 rows 41.4 -> 23.7, 6 rows 61.2 -> 32.5, 8 rows 80.3 -> 42.6.
_NR_SMALL = 2
_NR_LARGE = 4
#: A single row through this launch too (it repeats MLX's one-row arithmetic; proven per shape like
#: every width) instead of MLX's product, which for a float32 input (the head) first casts the scales
#: and biases to float32 (two more launches). Engine-timed, receipts/ebench-head-1.json (M5 Max, head
#: of 201,024 rows, 1 row): 1,171 -> 894 us; output projection 9.5 -> 9.0 us (ebench-oproj-1).
_ONE_ROW = True
#: Input totals staged once per threadgroup in threadgroup memory (SUMS) for a small matrix only. The
#: head computes them in each simdgroup: engine-timed, receipts/ebench-head-2.json (M5 Max), 2 rows
#: 520 -> 464 us, 3 rows 1,260 -> 1,273, 4 rows 1,779 -> 1,472, 8 rows 2,890 -> 2,424.
_SUMS_LARGE = False

QMV_ROWS = AFFINE_LANE + """
// MLX's one-row affine product for output rows [row0, row0 + NROW) and W panel rows of x (row r at
// x + r * x_stride; bf16 or float, as the reference product takes it), every lane of the calling
// simdgroup together, through affine_lane. result[r][ro] ends in every lane.
template <int W, int NROW, int K, int BITS, int GROUP, int VPT, typename T>
inline void qmv_rows(const device uint32_t* codes, const device bfloat16_t* scales,
                     const device bfloat16_t* biases, const device T* x, int x_stride, int row0,
                     uint lane, thread float (&result)[W][NROW]) {
    constexpr int BLOCK = 32 * VPT, NB = K / BLOCK, WPR = K * BITS / 32, GPR = K / GROUP;
    for (int r = 0; r < W; ++r)
        for (int ro = 0; ro < NROW; ++ro) result[r][ro] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const int col = b * BLOCK + int(lane) * VPT;
        float wf[NROW][VPT], sc[NROW], bi[NROW];
        for (int ro = 0; ro < NROW; ++ro) {
            const size_t row = size_t(row0 + ro);
            sc[ro] = static_cast<float>(scales[row * GPR + col / GROUP]);
            bi[ro] = static_cast<float>(biases[row * GPR + col / GROUP]);
            affine_lane::unpack<VPT, BITS>(codes + row * WPR + col * BITS / 32, wf[ro]);
        }
        for (int r = 0; r < W; ++r) {
            const device T* xs = x + size_t(r) * x_stride + col;
            float xt[VPT];
            const float tot = affine_lane::take<VPT, BITS>(xs, xt);
            for (int ro = 0; ro < NROW; ++ro)
                result[r][ro] += affine_lane::term<VPT, BITS>(xt, wf[ro], sc[ro], bi[ro], tot);
        }
    }
    for (int r = 0; r < W; ++r)
        for (int ro = 0; ro < NROW; ++ro) result[r][ro] = simd_sum(result[r][ro]);
}
"""

_SOURCE = """
    constexpr uint NSG = THREADS / 32;
    constexpr int BLOCK = VPT * 32;
    constexpr int NB = K / BLOCK;                // blocks a row (K is a multiple of BLOCK)
    constexpr int WPR = K * BITS / 32;           // code words a row
    constexpr int GPR = K / G;                   // groups a row
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = threadgroup_position_in_grid.x * (NSG * NR) + sg * NR;   // NR output rows a simdgroup
    // YSPLIT: each threadgroup row of the grid takes one chunk of RCH panel rows (more threadgroups
    // for a small matrix); otherwise one threadgroup loops over the chunks.
    const uint rc_begin = YSPLIT ? threadgroup_position_in_grid.y * RCH : 0;
    const uint rc_end = YSPLIT ? min(rc_begin + RCH, uint(W)) : uint(W);
    constexpr uint LROWS = YSPLIT ? RCH : W;

    // The inputs' totals (affine_lane::total) for this threadgroup's panel rows, every block and lane.
    threadgroup float lsum[SUMS ? LROWS * NB * 32 : 1];
    if (SUMS) {
        for (uint idx = thread_index_in_threadgroup; idx < LROWS * NB * 32; idx += THREADS) {
            const uint r = rc_begin + idx / (NB * 32), b = (idx / 32) % NB, l = idx % 32;
            if (r < rc_end) lsum[idx] = affine_lane::total<VPT, BITS>(x + size_t(r) * K + b * BLOCK + l * VPT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 >= OUTN) return;                // rows past the output's width are not written

    for (uint rc = rc_begin; rc < rc_end; rc += RCH) {
        float acc[NR][RCH];
        for (uint ro = 0; ro < NR; ++ro)
            for (uint r = 0; r < RCH; ++r) acc[ro][r] = 0.0f;
        for (int b = 0; b < NB; ++b) {
            const int col = b * BLOCK + int(lane) * VPT;
            float wf[NR][VPT], sc[NR], bi[NR];
            for (uint ro = 0; ro < NR; ++ro) {
                const size_t row = min(row0 + ro, uint(N - 1));
                sc[ro] = static_cast<float>(scales[row * GPR + col / G]);
                bi[ro] = static_cast<float>(biases[row * GPR + col / G]);
                affine_lane::unpack<VPT, BITS>(codes + row * WPR + col * BITS / 32, wf[ro]);
            }
            for (uint r = 0; r < RCH; ++r) {
                if (rc + r >= rc_end) break;
                const device T* xs = x + size_t(rc + r) * K + col;
                float xt[VPT];
                float tot;
                if (SUMS) {
                    affine_lane::scaled<VPT, BITS>(xs, xt);
                    tot = lsum[((rc + r - rc_begin) * NB + b) * 32 + lane];
                } else {
                    tot = affine_lane::take<VPT, BITS>(xs, xt);
                }
                for (uint ro = 0; ro < NR; ++ro)
                    acc[ro][r] += affine_lane::term<VPT, BITS>(xt, wf[ro], sc[ro], bi[ro], tot);
            }
        }
        for (uint ro = 0; ro < NR; ++ro)
            for (uint r = 0; r < RCH; ++r) acc[ro][r] = simd_sum(acc[ro][r]);
        if (lane == 0) {
            for (uint r = 0; r < RCH && rc + r < rc_end; ++r)
                for (uint ro = 0; ro < NR; ++ro)
                    if (row0 + ro < OUTN) out[size_t(rc + r) * OUTN + row0 + ro] = static_cast<T>(FINISH(acc[ro][r]));
        }
    }
"""

_PROBE = """
    // qmv_rows as another lane calls it: each simdgroup takes NROW output rows for the whole panel.
    const uint lane = thread_index_in_simdgroup;
    const int row0 = int(threadgroup_position_in_grid.x * (THREADS / 32) + simdgroup_index_in_threadgroup) * NROW;
    if (row0 + NROW > N) return;
    float result[W][NROW];
    qmv_rows<W, NROW, K, BITS, G, VPT>(codes, scales, biases, x, K, row0, lane, result);
    if (lane == 0)
        for (int r = 0; r < W; ++r)
            for (int ro = 0; ro < NROW; ++ro) out[size_t(r) * N + row0 + ro] = static_cast<T>(result[r][ro]);
"""


@lru_cache(maxsize=None)
def _kernel(source):
    if source == "capped":
        # The soft cap as the library's own elementwise operations (its Divide, Tanh and Multiply,
        # read from the installed headers), applied to each finished value before it is written.
        from . import library_gemm

        ops, why = library_gemm.library("elementwise")
        if ops is None:
            raise RuntimeError(f"no soft-capped product here: {why}")
        finish = ("#define FINISH(v) Multiply()(Tanh()(Divide()((v), as_type<float>(uint(CAP)))), "
                  "as_type<float>(uint(CAP)))\n")
        return mx.fast.metal_kernel(name="mtplx_inkling_fast_qproduct_capped",
                                    input_names=["x", "codes", "scales", "biases"], output_names=["out"],
                                    source=_SOURCE, header=ops + QMV_ROWS + finish)
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_qproduct" if source == "product" else "mtplx_inkling_fast_qmv_rows_probe",
        input_names=["x", "codes", "scales", "biases"],
        output_names=["out"],
        source=_SOURCE if source == "product" else _PROBE,
        header=QMV_ROWS + "#define FINISH(v) (v)\n",
    )


def lane_map(bits, rows, cols):
    """(values a lane takes per block, why not served) for MLX 0.32.2's one-row choice."""
    if bits not in (4, 8):
        return 0, f"bits_{bits}"
    pack = 32 // bits
    fast = rows % 8 == 0 and cols % (pack * 2 * 32) == 0
    vpt = pack * 2 if fast else pack
    if cols % (vpt * 32):
        return 0, f"cols_{cols}_not_whole_blocks"     # a partial last block (MLX's tail path) is not served
    return vpt, ""


def qmv_constants(weight):
    """``QMV_ROWS`` template values for ``weight`` (``K``, ``BITS``, ``GROUP``, ``VPT``); raises
    ``RuntimeError`` naming why a stored form is not served."""
    bits, group = int(weight.bits), int(weight.group_size)
    n, words = weight.codes.shape
    k = words * 32 // bits
    vpt, why = lane_map(bits, n, k)
    if not why and (group not in (32, 64, 128) or k % group):
        why = f"group_{group}"
    if not why and (weight.scales.dtype not in (mx.bfloat16, mx.float16) or weight.biases.dtype != weight.scales.dtype):
        why = "scales_not_bf16_or_fp16"
    if why:
        raise RuntimeError(f"qmv_rows does not serve this matrix: {why}")
    return {"K": k, "BITS": bits, "GROUP": group, "VPT": vpt}


_verdicts: dict = {}


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


def _launch(x2, weight, rows, n, k, bits, group, vpt, *, threads=None, rch=None, sums=None, ysplit=None, nr=None,
            width=None, cap=0.0):
    nb = k // (vpt * 32)
    small = n <= _SMALL_ROWS
    threads = threads or (_THREADS_SMALL if small else _THREADS_LARGE)
    # An explicit chunk may exceed the panel (rows past it are skipped): odd chunk sizes are cliff suspects.
    rch = rch or min(rows, _RCH_SMALL if small else _RCH_LARGE)
    nr = int(nr or (_NR_SMALL if small else _NR_LARGE))    # a partial last group: reads clamp, writes skip
    ysplit = bool(ysplit)
    lrows = rch if ysplit else rows
    if sums is None:
        sums = (small or _SUMS_LARGE) and lrows * nb * 32 * 4 <= 24 * 1024
    sums = bool(sums)
    rows_per_group = threads // 32 * nr
    width = int(width or n)
    template = [("T", x2.dtype), ("W", rows), ("K", k), ("N", n), ("BITS", bits), ("G", group), ("VPT", vpt),
                ("RCH", rch), ("THREADS", threads), ("SUMS", sums), ("YSPLIT", ysplit), ("NR", nr), ("OUTN", width)]
    if cap:
        template.append(("CAP", int.from_bytes(struct.pack("<f", float(cap)), "little")))
    return _kernel("capped" if cap else "product")(
        inputs=[x2, weight.codes, weight.scales, weight.biases],
        template=template,
        grid=(-(-width // rows_per_group) * threads, -(-rows // rch) if ysplit else 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[x2.dtype],
    )[0]


def affine_product(x, weight, *, prove=True, **tune):
    """Every row of ``x`` (``(..., K)`` bf16, float16 or float32) times the affine matrix ``weight`` (transposed, as
    ``mx.quantized_matmul`` takes it), bit-identical to MLX's one-row product of that row.
    Raises ``RuntimeError`` (counted) for a stored form this launch does not twin, or when this
    Mac's first use disagrees with MLX's one-row product; a single row then takes MLX's product."""
    from .attn_input import count_route

    codes, scales, biases = weight.codes, weight.scales, weight.biases
    bits, group = int(weight.bits), int(weight.group_size)
    n, words = codes.shape
    k = words * 32 // bits
    lead = x.shape[:-1]
    rows = 1
    for d in lead:
        rows *= d
    one = tune.pop("one", None)
    reference = lambda: mx.quantized_matmul(x, codes, scales, biases, transpose=True, group_size=group, bits=bits)
    if rows == 1 and not (_ONE_ROW if one is None else one):
        return reference()
    try:
        return _proven_launch(x, weight, rows, lead, n, k, bits, group, prove, tune, count_route)
    except RuntimeError:
        if rows == 1:   # one row: MLX's product is the reference itself (the refusal is counted)
            return reference()
        raise


def _proven_launch(x, weight, rows, lead, n, k, bits, group, prove, tune, count_route):
    codes, scales, biases = weight.codes, weight.scales, weight.biases
    if x.shape[-1] != k or x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        count_route("refused:qproduct_input")
        raise RuntimeError("affine_product serves bf16, float16 or float32 inputs of the matrix's width")
    try:
        vpt = qmv_constants(weight)["VPT"]
    except RuntimeError as err:
        count_route("refused:qproduct_" + str(err).rsplit(": ", 1)[-1])
        raise
    x2 = x.reshape(rows, k)
    out = _launch(x2, weight, rows, n, k, bits, group, vpt, **tune)
    if prove:
        key = (n, k, bits, group, rows, x2.dtype)
        verdict = _verdicts.get(key)
        if verdict is None:
            picks = sorted({0, rows - 1})
            want = [mx.quantized_matmul(x2[r:r + 1], codes, scales, biases, transpose=True, group_size=group,
                                        bits=bits) for r in picks]
            mx.eval(out, want)
            verdict = all(_same(out[r:r + 1], w) for r, w in zip(picks, want))
            _verdicts[key] = verdict
            count_route(("proved:" if verdict else "refused:") + f"qproduct_{n}x{k}_b{bits}_w{rows}_{x2.dtype}")
        if not verdict:
            raise RuntimeError("affine_product disagreed with MLX's one-row product on this Mac")
    return out.reshape(*lead, n)


def affine_logits(x, weight, *, width=None, cap=0.0, prove=True, **tune):
    """``affine_product(x, weight)[..., :width]`` and then, with ``cap`` (the vocabulary head's soft
    cap), ``tanh(logits / cap) * cap``, all in the product's one launch: the cap is the library's own
    Divide, Tanh and Multiply applied to each finished value. Every value equals, bit for bit, MLX's
    one-row product of that row sliced and capped by MLX's launches (proven per shape on this Mac's
    first use). Raises ``RuntimeError`` (counted) where the launch refuses; ``(..., width)`` float32
    or bf16 as ``x``."""
    from .attn_input import count_route

    codes, scales, biases = weight.codes, weight.scales, weight.biases
    bits, group = int(weight.bits), int(weight.group_size)
    n, words = codes.shape
    k = words * 32 // bits
    width = int(width or n)
    lead = x.shape[:-1]
    rows = 1
    for d in lead:
        rows *= d
    if x.shape[-1] != k or x.dtype not in (mx.bfloat16, mx.float16, mx.float32) or not 0 < width <= n:
        count_route("refused:qlogits_input")
        raise RuntimeError("affine_logits serves bf16, float16 or float32 inputs of the matrix's width")
    try:
        vpt = qmv_constants(weight)["VPT"]
    except RuntimeError as err:
        count_route("refused:qlogits_" + str(err).rsplit(": ", 1)[-1])
        raise
    x2 = x.reshape(rows, k)
    out = _launch(x2, weight, rows, n, k, bits, group, vpt, width=width, cap=cap, **tune)
    if prove:
        key = ("logits", n, k, bits, group, rows, x2.dtype, width, float(cap))
        verdict = _verdicts.get(key)
        if verdict is None:
            picks = sorted({0, rows - 1})
            want = []
            for r in picks:
                y = mx.quantized_matmul(x2[r:r + 1], codes, scales, biases, transpose=True, group_size=group,
                                        bits=bits)[..., :width]
                want.append(mx.tanh(y / cap) * cap if cap else y)
            mx.eval(out, want)
            verdict = all(_same(out[r:r + 1], w) for r, w in zip(picks, want))
            _verdicts[key] = verdict
            count_route(("proved:" if verdict else "refused:") + f"qlogits_{n}x{k}_b{bits}_w{rows}_{width}")
        if not verdict:
            raise RuntimeError("affine_logits disagreed with MLX's launches on this Mac")
    return out.reshape(*lead, width)


def qmv_rows_probe(x, weight, rows_each=4, threads=64):
    """``QMV_ROWS`` called as another lane's launch would call it (each simdgroup ``rows_each`` output
    rows, the whole panel): for the bit checks of the device function itself."""
    c = qmv_constants(weight)
    n = weight.codes.shape[0]
    rows = x.shape[0]
    if n % rows_each:
        raise ValueError("probe takes output rows in whole groups of rows_each")
    per_group = threads // 32 * rows_each
    return _kernel("probe")(
        inputs=[x, weight.codes, weight.scales, weight.biases],
        template=[("T", x.dtype), ("W", rows), ("NROW", rows_each), ("K", c["K"]), ("N", n), ("BITS", c["BITS"]),
                  ("G", c["GROUP"]), ("VPT", c["VPT"]), ("THREADS", threads)],
        grid=(-(-n // per_group) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
