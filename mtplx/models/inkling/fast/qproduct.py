"""Row-stable products of affine-coded matrices (the Bare Speed pack's output projection and
vocabulary head; any 4- or 8-bit matrix in MLX's layout), one launch for a panel of rows.

``affine_product(x, weight)``: ``x`` ``(..., K)`` bf16 or float32 with ``W`` rows (the head takes
float32, as ``head.vocabulary_logits`` feeds it; the output projection bf16), ``weight`` an affine
weight (``precision.AffineWeight``: uint32 codes ``(N, K * bits / 32)``, bf16 scales and biases
``(N, K / group)``). Returns ``(..., N)`` in ``x``'s dtype whose every row equals, bit for bit, MLX
0.32.2's one-row ``mx.quantized_matmul`` of that row (for float32 inputs MLX casts the scales and
biases to float32 first, two extra launches; here they are read as bf16, the same values). MLX's own product of 2 to 24 rows does not
(``affine_qmv_wide`` from M3 on, another order), so a verify panel through MLX differs from the
one-row step; through this launch it does not. One row takes MLX's own product (the reference).

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
    const uint row0 = threadgroup_position_in_grid.x * (NSG * 4) + sg * 4;
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
    if (row0 >= N) return;

    for (uint rc = rc_begin; rc < rc_end; rc += RCH) {
        float acc[4][RCH];
        for (uint ro = 0; ro < 4; ++ro)
            for (uint r = 0; r < RCH; ++r) acc[ro][r] = 0.0f;
        for (int b = 0; b < NB; ++b) {
            const int col = b * BLOCK + int(lane) * VPT;
            float wf[4][VPT], sc[4], bi[4];
            for (uint ro = 0; ro < 4; ++ro) {
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
                for (uint ro = 0; ro < 4; ++ro)
                    acc[ro][r] += affine_lane::term<VPT, BITS>(xt, wf[ro], sc[ro], bi[ro], tot);
            }
        }
        for (uint ro = 0; ro < 4; ++ro)
            for (uint r = 0; r < RCH; ++r) acc[ro][r] = simd_sum(acc[ro][r]);
        if (lane == 0) {
            for (uint r = 0; r < RCH && rc + r < rc_end; ++r)
                for (uint ro = 0; ro < 4; ++ro)
                    if (row0 + ro < N) out[size_t(rc + r) * N + row0 + ro] = static_cast<T>(acc[ro][r]);
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


@lru_cache(maxsize=2)
def _kernel(source):
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_qproduct" if source == "product" else "mtplx_inkling_fast_qmv_rows_probe",
        input_names=["x", "codes", "scales", "biases"],
        output_names=["out"],
        source=_SOURCE if source == "product" else _PROBE,
        header=QMV_ROWS,
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
    if not why and (weight.scales.dtype != mx.bfloat16 or weight.biases.dtype != mx.bfloat16):
        why = "scales_not_bf16"
    if why:
        raise RuntimeError(f"qmv_rows does not serve this matrix: {why}")
    return {"K": k, "BITS": bits, "GROUP": group, "VPT": vpt}


_verdicts: dict = {}


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


def _launch(x2, weight, rows, n, k, bits, group, vpt, *, threads=None, rch=None, sums=None, ysplit=None):
    nb = k // (vpt * 32)
    small = n <= _SMALL_ROWS
    threads = threads or (_THREADS_SMALL if small else _THREADS_LARGE)
    rch = min(rows, rch or (_RCH_SMALL if small else _RCH_LARGE))
    ysplit = bool(ysplit)
    lrows = rch if ysplit else rows
    sums = (lrows * nb * 32 * 4 <= 24 * 1024) if sums is None else bool(sums)
    rows_per_group = threads // 32 * 4
    return _kernel("product")(
        inputs=[x2, weight.codes, weight.scales, weight.biases],
        template=[("T", x2.dtype), ("W", rows), ("K", k), ("N", n), ("BITS", bits), ("G", group), ("VPT", vpt),
                  ("RCH", rch), ("THREADS", threads), ("SUMS", sums), ("YSPLIT", ysplit)],
        grid=(-(-n // rows_per_group) * threads, -(-rows // rch) if ysplit else 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x2.dtype],
    )[0]


def affine_product(x, weight, *, prove=True, **tune):
    """Every row of ``x`` (``(..., K)`` bf16 or float32) times the affine matrix ``weight`` (transposed, as
    ``mx.quantized_matmul`` takes it), bit-identical to MLX's one-row product of that row.
    Raises ``RuntimeError`` (counted) for a stored form this launch does not twin, or when this
    Mac's first use disagrees with MLX's one-row product."""
    from .attn_input import count_route

    codes, scales, biases = weight.codes, weight.scales, weight.biases
    bits, group = int(weight.bits), int(weight.group_size)
    n, words = codes.shape
    k = words * 32 // bits
    lead = x.shape[:-1]
    rows = 1
    for d in lead:
        rows *= d
    if rows == 1:
        return mx.quantized_matmul(x, codes, scales, biases, transpose=True, group_size=group, bits=bits)
    if x.shape[-1] != k or x.dtype not in (mx.bfloat16, mx.float32):
        count_route("refused:qproduct_input")
        raise RuntimeError("affine_product serves bf16 or float32 inputs of the matrix's width")
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
    // Stage C on stored codes in one launch (the step's output_projection): the output projection
    // in MLX's one-row product order (QMV_ROWS), rounded to bf16 as that product returns it; the
    // attention's static short convolution over the committed tail, with the reference kernel's
    // arithmetic (short_conv, FORM as proven for this width); the residual in float32. One
    // simdgroup: output channels [n0, n0 + 4) for every panel row.
    constexpr uint NSG = THREADS / 32;
    const uint lane = thread_index_in_simdgroup;
    const int n0 = int(threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup) * 4;
    if (n0 >= N) return;
    float result[W][4];
    qmv_rows<W, 4, K, BITS, G, VPT>(codes, scales, biases, hg, K, n0, lane, result);
    if (lane < 4) {
        const uint n = uint(n0) + lane;
        float seq[W + 3];
        for (uint t = 0; t < 3; ++t) seq[t] = HAS_TAIL ? tail[t * N + n] : 0.0f;
        for (uint r = 0; r < W; ++r) {
            float v = result[r][0];
            for (uint tm = 1; tm < 4; ++tm) v = (tm == lane) ? result[r][tm] : v;
            const float a = static_cast<float>(static_cast<bfloat16_t>(v));
            seq[3 + r] = a;
            attn[size_t(r) * N + n] = a;
        }
        for (uint r = 0; r < W; ++r) {
            float acc = 0.0f;
            for (uint j = 0; j < 4; ++j) {
                const float w = cw[n * 4 + j];      // the stored (channels, taps) layout
                if (FORM == 0) acc += seq[r + j] * w;
                else acc = metal::fma(seq[r + j], w, acc);
            }
            const float total = seq[r + 3] + acc;
            const float value = total / as_type<float>(uint(DIVISOR));
            const float y = static_cast<float>(static_cast<bfloat16_t>(value));
            partial[size_t(r) * N + n] = BOUNDARY ? y : y + static_cast<float>(x[size_t(r) * N + n]);
        }
    }
"""


@lru_cache(maxsize=1)
def _output_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_qoutput",
        input_names=["hg", "codes", "scales", "biases", "cw", "tail", "x"],
        output_names=["partial", "attn"],
        source=_OUTPUT,
        header=QMV_ROWS,
    )


_output_verdicts: dict = {}


def affine_output(hg, weight, conv, tail, x, boundary, *, form, prove=True):
    """The step's stage C on a stored-code output projection (``fast.step.output_projection``) in
    one launch: ``(partial (1, W, N) float32, attn (1, W, N) float32)``, every bit as the step's
    three launches give it (the product, the short convolution, the residual add).

    ``hg`` ``(1, W, K)`` bf16; ``weight`` the affine output projection; ``conv`` the attention's
    static ``Convolution`` (size 4); ``tail`` its committed ``(1, 3, N)`` float32 rows or ``None``;
    ``x`` the stage input ``(1, W, N)`` bf16 (unused on a boundary layer); ``form`` the static
    convolution's proven form for this width (``attn_input._static_form``). Raises ``RuntimeError``
    (counted) for a form it does not serve or when this Mac's first use disagrees."""
    from .attn_input import count_route, f32_bits

    if conv.dynamic is not None or conv.size != 4 or form not in ("float", "fused"):
        count_route("refused:qoutput_conv")
        raise RuntimeError("affine_output serves the static four-tap convolution")
    if hg.dtype != mx.bfloat16 or x.dtype != mx.bfloat16:
        count_route("refused:qoutput_dtype")
        raise RuntimeError("affine_output serves bf16 streams")
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
    threads = 128
    partial, attn = _output_kernel()(
        inputs=[hg2, weight.codes, weight.scales, weight.biases, conv.weight.T,
                tail if tail is not None else conv.weight.T, x2],
        template=[("W", rows), ("K", k), ("N", n), ("BITS", c["BITS"]), ("G", c["GROUP"]), ("VPT", c["VPT"]),
                  ("THREADS", threads), ("HAS_TAIL", tail is not None), ("BOUNDARY", bool(boundary)),
                  ("FORM", 0 if form == "float" else 1), ("DIVISOR", f32_bits(5 ** 0.5))],
        grid=(-(-n // (threads // 32 * 4)) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, n), (rows, n)],
        output_dtypes=[mx.float32, mx.float32],
    )
    partial, attn = partial.reshape(1, rows, n), attn.reshape(1, rows, n)
    if prove:
        key = (n, k, c["BITS"], c["GROUP"], rows, tail is not None, bool(boundary), form)
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
            count_route(("proved:" if verdict else "refused:") + f"qoutput_{n}x{k}_b{c['BITS']}_w{rows}")
        if not verdict:
            raise RuntimeError("affine_output disagreed with the step's three launches on this Mac")
    return partial, attn
