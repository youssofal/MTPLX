"""Row-stable products of affine-coded matrices (the Bare Speed pack's output projection and
vocabulary head; any 4- or 8-bit matrix in MLX's layout), one launch for a panel of rows.

``affine_product(x, weight)``: ``x`` ``(..., K)`` in the stream's 16-bit type (bf16, or float16 on
the FP16 packs) or float32 with ``W`` rows (the head takes float32, as ``head.vocabulary_logits``
feeds it; the output projection the stream's type), ``weight`` an affine weight
(``precision.AffineWeight``: uint32 codes ``(N, K * bits / 32)``, 16-bit scales and biases
``(N, K / group)`` of the stream's type). Returns ``(..., N)`` in ``x``'s dtype whose every row equals,
bit for bit, MLX 0.32.2's one-row ``mx.quantized_matmul`` of that row (for float32 inputs MLX casts the
scales and biases to float32 first, two extra launches; here they are read in their 16-bit type, the
same values). MLX's own product of 2 to 24 rows does not
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
fused attention output): ``qmv_rows<W, NROW, K, BITS, GROUP, VPT>(codes, scales, biases, panel, panel_stride,
row0, lane, result)`` leaves ``result[r][ro]`` (output row ``row0 + ro``, panel row ``r``) in every
lane of the calling simdgroup; ``qmv_constants(weight)`` gives its template values.
"""

from __future__ import annotations

from functools import lru_cache
import struct

import mlx.core as mx
from .. import launch_limits

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
// A lane's SPAN inputs widened to float, the values a float32 copy of the row holds (the product of a
// 16-bit row whose reference takes it as float32: the head).
template <int SPAN, typename P>
inline void widen(P v, thread float* out) {
    for (int j = 0; j < SPAN; ++j) out[j] = static_cast<float>(v[j]);
}

// MLX's one-row affine product for output rows [row0, row0 + NROW) and W panel rows (row r at
// panel + r * panel_stride; bf16, float16 or float, as the reference product takes it), every lane of
// the calling simdgroup together, through affine_lane; scales and biases bf16 or float16 (S).
// result[r][ro] ends in every lane.
template <int W, int NROW, int K, int BITS, int GROUP, int VPT, typename T, typename S>
inline void qmv_rows(const device uint32_t* codes, const device S* scales, const device S* biases,
                     const device T* panel, int panel_stride, int row0,
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
            const device T* xs = panel + size_t(r) * panel_stride + col;
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
            if (r >= rc_end) continue;
            const device T* xv = x + size_t(r) * K + b * BLOCK + l * VPT;
            if (WIDE) {
                float xw[VPT];
                widen<VPT>(xv, xw);
                lsum[idx] = affine_lane::total<VPT, BITS>(xw);
            } else {
                lsum[idx] = affine_lane::total<VPT, BITS>(xv);
            }
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
                if (WIDE) {
                    float xw[VPT];
                    widen<VPT>(xs, xw);
                    if (SUMS) {
                        affine_lane::scaled<VPT, BITS>(xw, xt);
                        tot = lsum[((rc + r - rc_begin) * NB + b) * 32 + lane];
                    } else {
                        tot = affine_lane::take<VPT, BITS>(xw, xt);
                    }
                } else if (SUMS) {
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
                    if (row0 + ro < OUTN) out[size_t(rc + r) * OUTN + row0 + ro] = static_cast<O>(FINISH(acc[ro][r]));
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
        return launch_limits.kernel(name="mtplx_inkling_fast_qproduct_capped",
                                    input_names=["x", "codes", "scales", "biases"], output_names=["out"],
                                    source=_SOURCE, header=ops + QMV_ROWS + finish)
    return launch_limits.kernel(
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
    if not why and not _sixteen_bit(weight):
        why = "scales_not_16bit"
    if why:
        raise RuntimeError(f"qmv_rows does not serve this matrix: {why}")
    return {"K": k, "BITS": bits, "GROUP": group, "VPT": vpt}


_verdicts: dict = {}
#: The 16-bit types a stream (and so a stored matrix's scales and biases) comes in: bf16, and float16
#: on the FP16 packs (M1 and M2 Macs).
STREAM_TYPES = (mx.bfloat16, mx.float16)


def _sixteen_bit(weight):
    """Scales and biases of one 16-bit stream type (the reference reads them as such; float32 inputs
    see them widened, the same values)."""
    return weight.scales.dtype in STREAM_TYPES and weight.biases.dtype == weight.scales.dtype


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


def _launch(x2, weight, rows, n, k, bits, group, vpt, *, threads=None, rch=None, sums=None, ysplit=None, nr=None,
            width=None, cap=0.0, wide=False):
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
    out_type = mx.float32 if wide else x2.dtype      # a widened 16-bit input gives the float32 product
    template = [("T", x2.dtype), ("O", out_type), ("WIDE", bool(wide)), ("W", rows), ("K", k), ("N", n),
                ("BITS", bits), ("G", group), ("VPT", vpt), ("RCH", rch), ("THREADS", threads), ("SUMS", sums),
                ("YSPLIT", ysplit), ("NR", nr), ("OUTN", width)]
    if cap:
        template.append(("CAP", int.from_bytes(struct.pack("<f", float(cap)), "little")))
    return _kernel("capped" if cap else "product")(
        inputs=[x2, weight.codes, weight.scales, weight.biases],
        template=template,
        grid=(-(-width // rows_per_group) * threads, -(-rows // rch) if ysplit else 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[out_type],
    )[0]


def affine_product(x, weight, *, prove=True, widen=False, one_row=None, **tune):
    """Every row of ``x`` (``(..., K)`` of the scales' 16-bit type, or float32) times the affine matrix
    ``weight`` (transposed, as ``mx.quantized_matmul`` takes it), bit-identical to MLX's one-row product
    of that row.

    ``widen``: ``x`` is 16-bit and the reference takes it as float32 (the head): each input is widened
    in the launch, the result is float32, every row equal to MLX's product of ``x.astype(float32)``
    (which also widens the 16-bit scales and biases to float32 in two launches of its own; here they
    are read in their 16-bit type, the same values). ``one_row``: whether a single row takes this launch
    too (proven on its first use like a panel) instead of MLX's product; ``None`` is ``_ONE_ROW``.
    Raises ``RuntimeError`` (counted) for a stored form this launch does not twin, or when this Mac's
    first use disagrees with MLX's one-row product; a single row then takes MLX's product."""
    from .attn_input import count_route

    codes, scales, biases = weight.codes, weight.scales, weight.biases
    bits, group = int(weight.bits), int(weight.group_size)
    n, words = codes.shape
    k = words * 32 // bits
    lead = x.shape[:-1]
    rows = 1
    for d in lead:
        rows *= d
    one = tune.pop("one", one_row)
    reference = lambda: mx.quantized_matmul(x.astype(mx.float32) if widen else x, codes, scales, biases,
                                            transpose=True, group_size=group, bits=bits)
    if rows == 1 and not (_ONE_ROW if one is None else one):
        return reference()
    try:
        return _proven_launch(x, weight, rows, lead, n, k, bits, group, prove, widen, tune, count_route)
    except RuntimeError:
        if rows == 1:   # one row: MLX's product is the reference itself (the refusal is counted)
            return reference()
        raise


def _proven_launch(x, weight, rows, lead, n, k, bits, group, prove, widen, tune, count_route):
    codes, scales, biases = weight.codes, weight.scales, weight.biases
    if (x.shape[-1] != k or x.dtype not in (scales.dtype, mx.float32) or x.dtype not in (*STREAM_TYPES, mx.float32)
            or (widen and x.dtype not in STREAM_TYPES)):
        count_route("refused:qproduct_input")
        raise RuntimeError("affine_product serves inputs of the matrix's width in its scales' type or float32")
    try:
        vpt = qmv_constants(weight)["VPT"]
    except RuntimeError as err:
        count_route("refused:qproduct_" + str(err).rsplit(": ", 1)[-1])
        raise
    x2 = x.reshape(rows, k)
    out = _launch(x2, weight, rows, n, k, bits, group, vpt, wide=widen, **tune)
    if prove:
        key = (n, k, bits, group, rows, x2.dtype, bool(widen))
        verdict = _verdicts.get(key)
        if verdict is None:
            picks = sorted({0, rows - 1})
            want = [mx.quantized_matmul(x2[r:r + 1].astype(mx.float32) if widen else x2[r:r + 1], codes, scales,
                                        biases, transpose=True, group_size=group, bits=bits) for r in picks]
            mx.eval(out, want)
            verdict = all(_same(out[r:r + 1], w) for r, w in zip(picks, want))
            _verdicts[key] = verdict
            count_route(("proved:" if verdict else "refused:")
                        + f"qproduct_{n}x{k}_b{bits}_w{rows}_{x2.dtype}{'_widened' if widen else ''}")
        if not verdict:
            raise RuntimeError("affine_product disagreed with MLX's one-row product on this Mac")
    return out.reshape(*lead, n)


def affine_logits(x, weight, *, width=None, cap=0.0, prove=True, widen=False, **tune):
    """``affine_product(x, weight)[..., :width]`` and then, with ``cap`` (the vocabulary head's soft
    cap), ``tanh(logits / cap) * cap``, all in the product's one launch: the cap is the library's own
    Divide, Tanh and Multiply applied to each finished value. Every value equals, bit for bit, MLX's
    one-row product of that row sliced and capped by MLX's launches (proven per shape on this Mac's
    first use). ``widen``: ``x`` is 16-bit and the reference takes it as float32 (the head's hidden):
    widened in the launch, as ``affine_product`` does. Raises ``RuntimeError`` (counted) where the launch
    refuses; ``(..., width)`` in ``x``'s type, float32 when widened."""
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
    if (x.shape[-1] != k or x.dtype not in (scales.dtype, mx.float32) or x.dtype not in (*STREAM_TYPES, mx.float32)
            or (widen and x.dtype not in STREAM_TYPES) or not 0 < width <= n):
        count_route("refused:qlogits_input")
        raise RuntimeError("affine_logits serves inputs of the matrix's width in its scales' type or float32")
    try:
        vpt = qmv_constants(weight)["VPT"]
    except RuntimeError as err:
        count_route("refused:qlogits_" + str(err).rsplit(": ", 1)[-1])
        raise
    x2 = x.reshape(rows, k)
    out = _launch(x2, weight, rows, n, k, bits, group, vpt, width=width, cap=cap, wide=widen, **tune)
    if prove:
        key = ("logits", n, k, bits, group, rows, x2.dtype, width, float(cap), bool(widen))
        verdict = _verdicts.get(key)
        if verdict is None:
            picks = sorted({0, rows - 1})
            want = []
            for r in picks:
                y = mx.quantized_matmul(x2[r:r + 1].astype(mx.float32) if widen else x2[r:r + 1], codes, scales,
                                        biases, transpose=True, group_size=group, bits=bits)[..., :width]
                want.append(mx.tanh(y / cap) * cap if cap else y)
            mx.eval(out, want)
            verdict = all(_same(out[r:r + 1], w) for r, w in zip(picks, want))
            _verdicts[key] = verdict
            count_route(("proved:" if verdict else "refused:")
                        + f"qlogits_{n}x{k}_b{bits}_w{rows}_{width}_{x2.dtype}{'_widened' if widen else ''}")
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

_OUTPUT = """
    // Stage C on stored codes in one launch (the step's output_projection): the output projection of
    // every panel row in MLX's one-row product order (affine_lane, as the product launch above),
    // rounded to the stream's 16-bit type T as that product returns it; the attention's static short
    // convolution over the committed tail with the reference kernel's arithmetic (short_conv, FORM as
    // proven for this width and type), cast to T as it casts; the residual in float32. A simdgroup
    // owns NR output channels for the whole panel (a channel's arithmetic does not depend on NR),
    // whose rows go in chunks of RCH.
    constexpr uint NSG = THREADS / 32;
    constexpr int BLOCK = VPT * 32;
    constexpr int NB = K / BLOCK;                // blocks a row (K is a multiple of BLOCK)
    constexpr int WPR = K * BITS / 32;           // code words a row
    constexpr int GPR = K / G;                   // groups a row
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = threadgroup_position_in_grid.x * (NSG * NR) + sg * NR;

    // The inputs' totals (affine_lane::total) of every panel row, block and lane, once a threadgroup.
    threadgroup float lsum[SUMS ? W * NB * 32 : 1];
    if (SUMS) {
        for (uint idx = thread_index_in_threadgroup; idx < uint(W * NB * 32); idx += THREADS) {
            const uint r = idx / (NB * 32), b = (idx / 32) % NB, l = idx % 32;
            lsum[idx] = affine_lane::total<VPT, BITS>(hg + size_t(r) * K + b * BLOCK + l * VPT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 >= uint(N)) return;

    for (uint rc = 0; rc < uint(W); rc += RCH) {
        float acc[NR][RCH];
        for (uint ro = 0; ro < NR; ++ro)
            for (uint r = 0; r < RCH; ++r) acc[ro][r] = 0.0f;
        for (int b = 0; b < NB; ++b) {
            const int col = b * BLOCK + int(lane) * VPT;
            if (NARROW) {
                // The chunk's inputs first, then one output channel's codes at a time (few rows: one
                // channel's codes live instead of NR).
                float xt[RCH][VPT], tot[RCH];
                for (uint r = 0; r < RCH; ++r) {
                    if (rc + r >= uint(W)) break;
                    const device T* xs = hg + size_t(rc + r) * K + col;
                    if (SUMS) {
                        affine_lane::scaled<VPT, BITS>(xs, xt[r]);
                        tot[r] = lsum[((rc + r) * NB + b) * 32 + lane];
                    } else {
                        tot[r] = affine_lane::take<VPT, BITS>(xs, xt[r]);
                    }
                }
                for (uint ro = 0; ro < NR; ++ro) {
                    const size_t row = min(row0 + ro, uint(N - 1));
                    const float sc = static_cast<float>(scales[row * GPR + col / G]);
                    const float bi = static_cast<float>(biases[row * GPR + col / G]);
                    float wf[VPT];
                    affine_lane::unpack<VPT, BITS>(codes + row * WPR + col * BITS / 32, wf);
                    for (uint r = 0; r < RCH; ++r) {
                        if (rc + r >= uint(W)) break;
                        acc[ro][r] += affine_lane::term<VPT, BITS>(xt[r], wf, sc, bi, tot[r]);
                    }
                }
            } else {
                // Each output channel's codes unpacked once for the whole chunk.
                float wf[NR][VPT], sc[NR], bi[NR];
                for (uint ro = 0; ro < NR; ++ro) {
                    const size_t row = min(row0 + ro, uint(N - 1));
                    sc[ro] = static_cast<float>(scales[row * GPR + col / G]);
                    bi[ro] = static_cast<float>(biases[row * GPR + col / G]);
                    affine_lane::unpack<VPT, BITS>(codes + row * WPR + col * BITS / 32, wf[ro]);
                }
                for (uint r = 0; r < RCH; ++r) {
                    if (rc + r >= uint(W)) break;
                    const device T* xs = hg + size_t(rc + r) * K + col;
                    float xt[VPT];
                    float tot;
                    if (SUMS) {
                        affine_lane::scaled<VPT, BITS>(xs, xt);
                        tot = lsum[((rc + r) * NB + b) * 32 + lane];
                    } else {
                        tot = affine_lane::take<VPT, BITS>(xs, xt);
                    }
                    for (uint ro = 0; ro < NR; ++ro)
                        acc[ro][r] += affine_lane::term<VPT, BITS>(xt, wf[ro], sc[ro], bi[ro], tot);
                }
            }
        }
        for (uint ro = 0; ro < NR; ++ro)
            for (uint r = 0; r < RCH; ++r) acc[ro][r] = simd_sum(acc[ro][r]);
        if (lane < NR && row0 + lane < uint(N)) {
            // Lane l convolves channel row0 + l over the chunk's rows: its three previous inputs, oldest
            // first, are the committed tail or rows this lane wrote before (attn), then the chunk's own.
            const uint c = row0 + lane;
            float seq[RCH + 3];
            for (uint t = 0; t < 3; ++t) {
                const int at = int(rc) - 3 + int(t);
                seq[t] = at >= 0 ? attn[size_t(at) * N + c] : (HAS_TAIL ? tail[uint(at + 3) * N + c] : 0.0f);
            }
            for (uint r = 0; r < RCH; ++r) {
                if (rc + r >= uint(W)) break;
                float v = acc[0][r];
                for (uint ro = 1; ro < NR; ++ro) v = (ro == lane) ? acc[ro][r] : v;
                const float a = static_cast<float>(static_cast<T>(v));
                seq[3 + r] = a;
                attn[size_t(rc + r) * N + c] = a;
            }
            for (uint r = 0; r < RCH; ++r) {
                if (rc + r >= uint(W)) break;
                float conv = 0.0f;
                for (uint j = 0; j < 4; ++j) {
                    const float w = cw[c * 4 + j];      // the stored (channels, taps) layout
                    if (FORM == 0) conv += seq[r + j] * w;
                    else conv = metal::fma(seq[r + j], w, conv);
                }
                const float total = seq[r + 3] + conv;
                const float value = total / as_type<float>(uint(DIVISOR));
                const float y = static_cast<float>(static_cast<T>(value));
                const size_t at = size_t(rc + r) * N + c;
                partial[at] = BOUNDARY ? y : y + static_cast<float>(x[at]);
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _output_kernel():
    return launch_limits.kernel(
        name="mtplx_inkling_fast_qoutput",
        input_names=["hg", "codes", "scales", "biases", "cw", "tail", "x"],
        output_names=["partial", "attn"],
        source=_OUTPUT,
        header=AFFINE_LANE,
    )


_output_verdicts: dict = {}
#: Panels of at most this many rows (by bits) take the narrow order (the chunk's inputs first, one output
#: channel's codes live at a time, four channels a simdgroup); wider panels share each channel's unpacked
#: codes across their rows, _OUTPUT_NR channels a simdgroup. Engine GPU clock, replays of a 24-visit chain
#: (x1/receipts/cnr-2.log, M5 Max, provisional), µs a visit at 4 bits, narrow with 4 channels against wide
#: with 2: 1 row 28.2 / 30.0 (best of 15), 2 rows 24.9 / 24.7, 3 rows 26.3 / 19.4, 4 rows 30.6 / 21.1,
#: 8 rows 54.3 (wide, 4 channels) / 33.5; on the 16 GB M4 (m4-out/m4-cnr-4bit-2.json) 47.6 / 47.5, 53.0 /
#: 49.0, 99.9 / 57.5, 129.4 / 65.4, 213.2 / 116.6, and 16 and 32 rows 404 / 229, 779 / 454 (wide, 4 / 2).
#: Every layout gives the same bits (each channel's arithmetic does not depend on it).
_OUTPUT_NARROW_ROWS = {4: 1, 8: 0}
#: Threads of one stage C threadgroup.
_OUTPUT_THREADS = 128
#: Output channels one simdgroup owns in the wide order, by bits. At 8 bits 4 stays ahead or level on
#: the M5 Max at 1 to 8 rows (cnr-2.log: 18.9 / 19.8 at one row, 31.8 / 32.3 at 8) and on the M4 past 8
#: rows (m4-cnr-8bit-wide.json: 212 / 223 at 16).
_OUTPUT_NR = {4: 2, 8: 4}
#: The inputs' totals computed once a threadgroup (shared memory and a barrier) measured slower than
#: each simdgroup taking its own (ceng-*-1: 4-bit 33.0 against 31.2 µs a visit at 4 rows, 63.4 against
#: 54.5 at 8; 8-bit 23.8 against 21.9 and 34.8 against 31.6), so stage C takes its own by default.
_OUTPUT_SUMS = False


def affine_output(hg, weight, conv, tail, x, boundary, *, form, prove=True, narrow=None, threads=None, sums=None,
                  rch=None, nr=None):
    """The step's stage C on a stored-code output projection (``fast.step.output_projection``) in
    one launch: ``(partial (1, W, N) float32, attn (1, W, N) float32)``, every bit as the step's
    three launches give it (the product, the short convolution, the residual add).

    ``hg`` ``(1, W, K)`` in the stream's 16-bit type (bf16, or float16 on the FP16 packs: the matrix's
    scales are of the same type); ``weight`` the affine output projection; ``conv`` the attention's
    static ``Convolution`` (size 4); ``tail`` its committed ``(1, 3, N)`` float32 rows or ``None``;
    ``x`` the stage input ``(1, W, N)`` of the stream's type (unused on a boundary layer); ``form`` the
    static convolution's proven form for this width and type (``attn_input._static_form``). Raises
    ``RuntimeError`` (counted) for a form it does not serve or when this Mac's first use disagrees."""
    from .attn_input import count_route, f32_bits

    if conv.dynamic is not None or conv.size != 4 or form not in ("float", "fused"):
        count_route("refused:qoutput_conv")
        raise RuntimeError("affine_output serves the static four-tap convolution")
    if hg.dtype not in STREAM_TYPES or x.dtype != hg.dtype or weight.scales.dtype != hg.dtype:
        count_route("refused:qoutput_dtype")
        raise RuntimeError("affine_output serves 16-bit streams whose type the matrix's scales share")
    try:
        c = qmv_constants(weight)
    except RuntimeError as err:
        count_route("refused:qoutput_" + str(err).rsplit(": ", 1)[-1])
        raise
    n = weight.codes.shape[0]
    rows = hg.shape[-2]
    k = c["K"]
    if rows > 32 or hg.shape[-1] != k:
        count_route("refused:qoutput_shape")
        raise RuntimeError("affine_output serves panels of up to 32 rows of the matrix's width")
    hg2, x2 = hg.reshape(rows, k), x.reshape(rows, n)
    narrow = (rows <= _OUTPUT_NARROW_ROWS.get(c["BITS"], 0)) if narrow is None else bool(narrow)
    threads = threads or _OUTPUT_THREADS
    nr = int(nr or (4 if narrow else _OUTPUT_NR.get(c["BITS"], 4)))
    rch = min(rows, rch or (rows if narrow else _RCH_SMALL))
    nb = k // (c["VPT"] * 32)
    sums = (_OUTPUT_SUMS and rows * nb * 32 * 4 <= 24 * 1024) if sums is None else bool(sums)
    partial, attn = _output_kernel()(
        inputs=[hg2, weight.codes, weight.scales, weight.biases, conv.weight.T,
                tail if tail is not None else conv.weight.T, x2],
        template=[("T", hg.dtype), ("W", rows), ("K", k), ("N", n), ("BITS", c["BITS"]), ("G", c["GROUP"]),
                  ("VPT", c["VPT"]),
                  ("THREADS", threads), ("NR", nr), ("RCH", rch), ("SUMS", sums), ("NARROW", narrow),
                  ("HAS_TAIL", tail is not None), ("BOUNDARY", bool(boundary)),
                  ("FORM", 0 if form == "float" else 1), ("DIVISOR", f32_bits(5 ** 0.5))],
        grid=(-(-n // (threads // 32 * nr)) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, n), (rows, n)],
        output_dtypes=[mx.float32, mx.float32],
    )
    partial, attn = partial.reshape(1, rows, n), attn.reshape(1, rows, n)
    if prove:
        # The layout is part of the key: each is proven on its own (a channel's arithmetic does not depend on
        # it, and the proof is what says so on this Mac).
        key = (n, k, c["BITS"], c["GROUP"], rows, tail is not None, bool(boundary), form, str(hg.dtype), nr, narrow,
               rch, sums, threads)
        verdict = _output_verdicts.get(key)
        if verdict is None:
            from .. import short_conv

            ref_attn = (mx.quantized_matmul(hg, weight.codes, weight.scales, weight.biases, transpose=True,
                                            group_size=c["GROUP"], bits=c["BITS"]) if rows == 1
                        else affine_product(hg, weight, prove=False))
            fused = short_conv.convolve(ref_attn, tail, conv.weight, None, False)
            if fused is None:
                verdict = False
            else:
                ref_partial = fused[0].astype(mx.float32)
                if not boundary:
                    ref_partial = ref_partial + x.astype(mx.float32)
                mx.eval(partial, attn, ref_partial, ref_attn)
                verdict = _same(partial, ref_partial) and _same(attn, ref_attn.astype(mx.float32))
            _output_verdicts[key] = verdict
            count_route(("proved:" if verdict else "refused:") + f"qoutput_{n}x{k}_b{c['BITS']}_w{rows}_{hg.dtype}")
        if not verdict:
            raise RuntimeError("affine_output disagreed with the step's three launches on this Mac")
    return partial, attn


_ROWS = """
    // Rows of an affine-coded table gathered and decoded in one launch: output (r, col) is entry col of
    // table row ids[r], its code times its group's scale plus its group's bias, rounded once to the
    // table's 16-bit type (FORM 0: the product and sum in float; FORM 1: in the 16-bit type, each
    // operation rounded on its own).
    const uint col = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    if (col >= uint(COLS)) return;
    constexpr uint PER = 32 / BITS;              // codes a word
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr uint WPR = COLS / PER;             // code words a row
    constexpr uint GPR = COLS / G;               // groups a row
    const size_t id = size_t(uint(ids[r]));
    const uint code = (codes[id * WPR + col / PER] >> (BITS * (col % PER))) & MASK;
    const T scale = scales[id * GPR + col / G];
    const T bias = biases[id * GPR + col / G];
    T value;
    if (FORM == 0) {
        value = static_cast<T>(static_cast<float>(scale) * static_cast<float>(code) + static_cast<float>(bias));
    } else {
        const T scaled = scale * static_cast<T>(code);
        value = scaled + bias;
    }
    out[size_t(r) * COLS + col] = value;
"""
#: Ways of writing the decode; a first use keeps the first whose rows equal MLX's own decode.
ROW_FORMS = (0, 1)
_rows_verdicts: dict = {}


@lru_cache(maxsize=1)
def _rows_kernel():
    return launch_limits.kernel(
        name="mtplx_inkling_fast_affine_rows",
        input_names=["ids", "codes", "scales", "biases"],
        output_names=["out"],
        source=_ROWS,
    )


def _rows_launch(ids, weight, form):
    bits, group = int(weight.bits), int(weight.group_size)
    vocab, words = weight.codes.shape
    cols = words * 32 // bits
    flat = ids.reshape(-1).astype(mx.int32)
    return _rows_kernel()(
        inputs=[flat, weight.codes, weight.scales, weight.biases],
        template=[("T", weight.scales.dtype), ("COLS", cols), ("BITS", bits), ("G", group), ("FORM", form)],
        grid=(cols, flat.size, 1),
        threadgroup=(min(256, cols), 1, 1),
        output_shapes=[(flat.size, cols)],
        output_dtypes=[weight.scales.dtype],
    )[0].reshape(*ids.shape, cols)


def rows_reason(weight):
    """Why ``affine_rows`` does not serve this stored table, or None: 4- or 8-bit codes in whole groups
    with 16-bit scales and biases of one type (bf16, float16 on the FP16 packs; the forms are proven
    against MLX's decode on a real table: Bare Speed's embedding, every value of all 201,024 rows)."""
    bits, group = int(weight.bits), int(weight.group_size)
    cols = weight.codes.shape[-1] * 32 // bits if bits else 0
    if bits not in (4, 8):
        return f"bits_{bits}"
    if group not in (32, 64, 128) or cols % group:
        return f"group_{group}"
    if not _sixteen_bit(weight):
        return "scales_not_16bit"
    return None


def affine_rows(ids, weight, *, prove=True, form=None):
    """``mx.dequantize(codes[ids], scales[ids], biases[ids])`` for an affine weight in one launch (MLX
    gathers the codes, the scales and the biases, then decodes: four launches), every value equal. The
    first use per stored form on this Mac compares a decode of sample rows (the first, the last and
    spread between) against MLX's own and keeps the first form that matches, or refuses (counted)."""
    from .attn_input import count_route

    bits, group = int(weight.bits), int(weight.group_size)
    vocab, words = weight.codes.shape
    cols = words * 32 // bits
    why = rows_reason(weight)
    if why is not None:
        count_route(f"refused:affine_rows_{why}")
        raise RuntimeError(f"affine_rows does not serve this table: {why}")
    if form is not None:
        return _rows_launch(ids, weight, form)
    key = (vocab, cols, bits, group, str(weight.scales.dtype))
    form = _rows_verdicts.get(key)
    if form is None and prove:
        sample = mx.array(sorted({0, vocab - 1, *range(1, vocab, max(1, vocab // 509))}), mx.int32)
        want = mx.dequantize(weight.codes[sample], weight.scales[sample], weight.biases[sample], bits=bits,
                             group_size=group)
        form = -1
        for candidate in ROW_FORMS:
            got = _rows_launch(sample, weight, candidate)
            mx.eval(got, want)
            if _same(got, want):
                form = candidate
                break
        _rows_verdicts[key] = form
        count_route(("proved:" if form >= 0 else "refused:")
                    + f"affine_rows_{vocab}x{cols}_b{bits}_{weight.scales.dtype}")
    if form is None or form < 0:
        raise RuntimeError("affine_rows did not match MLX's decode on this Mac")
    return _rows_launch(ids, weight, form)
