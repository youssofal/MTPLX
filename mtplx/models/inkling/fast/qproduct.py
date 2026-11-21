"""Row-stable products of affine-coded matrices (the Bare Speed pack's output projection and
vocabulary head; any 4- or 8-bit matrix in MLX's layout), one launch for a panel of rows.

``affine_product(x, weight)``: ``x`` ``(..., K)`` bf16 with ``W`` rows, ``weight`` an affine weight
(``precision.AffineWeight``: uint32 codes ``(N, K * bits / 32)``, bf16 scales and biases
``(N, K / group)``). Returns ``(..., N)`` bf16 whose every row equals, bit for bit, MLX 0.32.2's
one-row ``mx.quantized_matmul`` of that row. MLX's own product of 2 to 24 rows does not
(``affine_qmv_wide`` from M3 on, another order), so a verify panel through MLX differs from the
one-row step; through this launch it does not. One row takes MLX's own product (the reference).

Exactness: each output row repeats the one-row kernel MLX picks for the matrix's shape (its
``dispatch_qmv`` rule: ``qmv_fast`` when N is a multiple of 8 and K of 512 at 4 bits or 256 at 8
bits, else ``qmv``), lane for lane: lane ``l`` takes ``VPT`` consecutive values of each block of
``32 VPT`` columns (4 bits: 8 values in ``qmv``, 16 in ``qmv_fast``; 8 bits: 4 and 8); per block
``load_vector`` (bf16 quartet sums, inputs pre-divided by 16, 256, 4,096 at 4 bits), ``qdot`` (masked
products in its expression tree), ``scale * accum + sum * bias`` added to the lane's total;
``simd_sum`` over the 32 lanes. The codes of an output row are decoded once per block and shared
by every panel row. A first use on each shape proves the result against MLX's one-row product on
the real operands; a disagreement refuses the route (counted) and the caller keeps MLX's product.

``QMV_ROWS`` is the same arithmetic as a Metal device function for another lane's launch (the
fused attention output): ``qmv_rows<W, TM, K, BITS, GROUP, VPT>(codes, scales, biases, x, x_stride,
row0, lane, result)`` leaves ``result[r][tm]`` (output row ``row0 + tm``, panel row ``r``) in every
lane of the calling simdgroup; ``qmv_constants(weight)`` gives its template values.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

#: Threads of one threadgroup (simdgroups x 4 output rows each), by output rows: a small matrix
#: (an output projection) gets more, smaller threadgroups so every GPU core has several.
_THREADS_SMALL = 64
_THREADS_LARGE = 256
_SMALL_ROWS = 16384
#: Panel rows a pass accumulates at once (wider panels loop; each pass decodes the codes again).
_RCH = 8

QMV_ROWS = """
// MLX 0.32.2's load_vector for one lane's VPT values of a block (bf16 quartet sums in its order at
// 4 bits, a running sum at 8; the 4-bit inputs pre-divided by 1, 16, 256, 4,096), into xt.
template <int VPT, int BITS>
inline float qmv_load(const device bfloat16_t* xs, thread float (&xt)[VPT]) {
    float sum = 0;
    if (BITS == 4) {
        for (int i = 0; i < VPT; i += 4) {
            sum += xs[i] + xs[i + 1] + xs[i + 2] + xs[i + 3];
            xt[i] = xs[i];
            xt[i + 1] = xs[i + 1] / 16.0f;
            xt[i + 2] = xs[i + 2] / 256.0f;
            xt[i + 3] = xs[i + 3] / 4096.0f;
        }
    } else {
        for (int i = 0; i < VPT; i++) {
            sum += xs[i];
            xt[i] = xs[i];
        }
    }
    return sum;
}

// The same pre-scaled inputs without the sum (the caller has it).
template <int VPT, int BITS>
inline void qmv_inputs(const device bfloat16_t* xs, thread float (&xt)[VPT]) {
    if (BITS == 4) {
        for (int i = 0; i < VPT; i += 4) {
            xt[i] = xs[i];
            xt[i + 1] = xs[i + 1] / 16.0f;
            xt[i + 2] = xs[i + 2] / 256.0f;
            xt[i + 3] = xs[i + 3] / 4096.0f;
        }
    } else {
        for (int i = 0; i < VPT; i++) xt[i] = xs[i];
    }
}

// One output row's VPT codes of a block as the floats qdot multiplies by (4 bits: the masked 16-bit
// words, exact in float; 8 bits: the bytes).
template <int VPT, int BITS>
inline void qmv_decode(const device uint32_t* cw, thread float (&wf)[VPT]) {
    if (BITS == 4) {
        for (int p = 0; p < VPT / 8; ++p) {
            const uint word = cw[p];
            const ushort w16[2] = {ushort(word & 0xffffu), ushort(word >> 16)};
            for (int h = 0; h < 2; ++h) {
                wf[8 * p + 4 * h] = float(w16[h] & 0x000f);
                wf[8 * p + 4 * h + 1] = float(w16[h] & 0x00f0);
                wf[8 * p + 4 * h + 2] = float(w16[h] & 0x0f00);
                wf[8 * p + 4 * h + 3] = float(w16[h] & 0xf000);
            }
        }
    } else {
        for (int i = 0; i < VPT; i++) wf[i] = float(uchar((cw[i / 4] >> (8 * (i % 4))) & 0xffu));
    }
}

// qdot on decoded codes, in its expression tree; returns scale * accum + sum * bias.
template <int VPT, int BITS>
inline float qmv_term(const thread float (&xt)[VPT], float sum, const thread float (&wf)[VPT], float scale,
                      float bias) {
    float accum = 0;
    if (BITS == 4) {
        for (int i = 0; i < VPT / 4; ++i) {
            accum +=
                (xt[4 * i] * wf[4 * i] +
                 xt[4 * i + 1] * wf[4 * i + 1] +
                 xt[4 * i + 2] * wf[4 * i + 2] +
                 xt[4 * i + 3] * wf[4 * i + 3]);
        }
    } else {
        for (int i = 0; i < VPT; i++) accum += xt[i] * wf[i];
    }
    return scale * accum + sum * bias;
}

// MLX's one-row affine product for output rows [row0, row0 + TM) and W panel rows of x (row r at
// x + r * x_stride), every lane of the calling simdgroup together. result[r][tm] ends in every lane.
template <int W, int TM, int K, int BITS, int GROUP, int VPT>
inline void qmv_rows(const device uint32_t* codes, const device bfloat16_t* scales,
                     const device bfloat16_t* biases, const device bfloat16_t* x, int x_stride, int row0,
                     uint lane, thread float (&result)[W][TM]) {
    constexpr int BLOCK = 32 * VPT, NB = K / BLOCK, WPR = K * BITS / 32, GPR = K / GROUP;
    for (int r = 0; r < W; ++r)
        for (int tm = 0; tm < TM; ++tm) result[r][tm] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const int col = b * BLOCK + int(lane) * VPT;
        float wf[TM][VPT], sc[TM], bi[TM];
        for (int tm = 0; tm < TM; ++tm) {
            const size_t row = size_t(row0 + tm);
            sc[tm] = static_cast<float>(scales[row * GPR + col / GROUP]);
            bi[tm] = static_cast<float>(biases[row * GPR + col / GROUP]);
            qmv_decode<VPT, BITS>(codes + row * WPR + col * BITS / 32, wf[tm]);
        }
        for (int r = 0; r < W; ++r) {
            float xt[VPT];
            const float sum = qmv_load<VPT, BITS>(x + size_t(r) * x_stride + col, xt);
            for (int tm = 0; tm < TM; ++tm) result[r][tm] += qmv_term<VPT, BITS>(xt, sum, wf[tm], sc[tm], bi[tm]);
        }
    }
    for (int r = 0; r < W; ++r)
        for (int tm = 0; tm < TM; ++tm) result[r][tm] = simd_sum(result[r][tm]);
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

    // load_vector's sum (MLX's order) for every panel row, block and lane, once per threadgroup.
    threadgroup float lsum[SUMS ? W * NB * 32 : 1];
    if (SUMS) {
        for (uint idx = thread_index_in_threadgroup; idx < W * NB * 32; idx += THREADS) {
            const uint r = idx / (NB * 32), b = (idx / 32) % NB, l = idx % 32;
            float xt[VPT];
            lsum[idx] = qmv_load<VPT, BITS>(x + size_t(r) * K + b * BLOCK + l * VPT, xt);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 >= N) return;

    for (uint rc = 0; rc < W; rc += RCH) {
        float acc[4][RCH];
        for (uint tm = 0; tm < 4; ++tm)
            for (uint r = 0; r < RCH; ++r) acc[tm][r] = 0.0f;
        for (int b = 0; b < NB; ++b) {
            const int col = b * BLOCK + int(lane) * VPT;
            float wf[4][VPT], sc[4], bi[4];
            for (uint tm = 0; tm < 4; ++tm) {
                const size_t row = min(row0 + tm, uint(N - 1));
                sc[tm] = static_cast<float>(scales[row * GPR + col / G]);
                bi[tm] = static_cast<float>(biases[row * GPR + col / G]);
                qmv_decode<VPT, BITS>(codes + row * WPR + col * BITS / 32, wf[tm]);
            }
            for (uint r = 0; r < RCH; ++r) {
                if (rc + r >= W) break;
                const device bfloat16_t* xs = x + size_t(rc + r) * K + col;
                float xt[VPT];
                float sum;
                if (SUMS) {
                    qmv_inputs<VPT, BITS>(xs, xt);
                    sum = lsum[((rc + r) * NB + b) * 32 + lane];
                } else {
                    sum = qmv_load<VPT, BITS>(xs, xt);
                }
                for (uint tm = 0; tm < 4; ++tm) acc[tm][r] += qmv_term<VPT, BITS>(xt, sum, wf[tm], sc[tm], bi[tm]);
            }
        }
        for (uint tm = 0; tm < 4; ++tm)
            for (uint r = 0; r < RCH; ++r) acc[tm][r] = simd_sum(acc[tm][r]);
        if (lane == 0) {
            for (uint r = 0; r < RCH && rc + r < W; ++r)
                for (uint tm = 0; tm < 4; ++tm)
                    if (row0 + tm < N) out[size_t(rc + r) * N + row0 + tm] = static_cast<bfloat16_t>(acc[tm][r]);
        }
    }
"""

_PROBE = """
    // qmv_rows as another lane calls it: each simdgroup takes TM output rows for the whole panel.
    const uint lane = thread_index_in_simdgroup;
    const int row0 = int(threadgroup_position_in_grid.x * (THREADS / 32) + simdgroup_index_in_threadgroup) * TM;
    if (row0 + TM > N) return;
    float result[W][TM];
    qmv_rows<W, TM, K, BITS, G, VPT>(codes, scales, biases, x, K, row0, lane, result);
    if (lane == 0)
        for (int r = 0; r < W; ++r)
            for (int tm = 0; tm < TM; ++tm) out[size_t(r) * N + row0 + tm] = static_cast<bfloat16_t>(result[r][tm]);
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
        return 0, f"cols_{cols}_not_whole_blocks"     # MLX's qmv tail (load_vector_safe) not served
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
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def _launch(x2, weight, rows, n, k, bits, group, vpt, *, threads=None, rch=None, sums=None):
    nb = k // (vpt * 32)
    threads = threads or (_THREADS_SMALL if n <= _SMALL_ROWS else _THREADS_LARGE)
    rch = min(rows, rch or _RCH)
    sums = (rows * nb * 32 * 4 <= 24 * 1024) if sums is None else bool(sums)
    rows_per_group = threads // 32 * 4
    return _kernel("product")(
        inputs=[x2, weight.codes, weight.scales, weight.biases],
        template=[("W", rows), ("K", k), ("N", n), ("BITS", bits), ("G", group), ("VPT", vpt),
                  ("RCH", rch), ("THREADS", threads), ("SUMS", sums)],
        grid=(-(-n // rows_per_group) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def affine_product(x, weight, *, prove=True, **tune):
    """Every row of ``x`` (``(..., K)`` bf16) times the affine matrix ``weight`` (transposed, as
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
    if x.shape[-1] != k or x.dtype != mx.bfloat16:
        count_route("refused:qproduct_input")
        raise RuntimeError("affine_product serves bf16 inputs of the matrix's width")
    try:
        vpt = qmv_constants(weight)["VPT"]
    except RuntimeError as err:
        count_route("refused:qproduct_" + str(err).rsplit(": ", 1)[-1])
        raise
    x2 = x.reshape(rows, k)
    out = _launch(x2, weight, rows, n, k, bits, group, vpt, **tune)
    if prove:
        key = (n, k, bits, group, rows)
        verdict = _verdicts.get(key)
        if verdict is None:
            picks = sorted({0, rows - 1})
            want = [mx.quantized_matmul(x2[r:r + 1], codes, scales, biases, transpose=True, group_size=group,
                                        bits=bits) for r in picks]
            mx.eval(out, want)
            verdict = all(_same(out[r:r + 1], w) for r, w in zip(picks, want))
            _verdicts[key] = verdict
            count_route(("proved:" if verdict else "refused:") + f"qproduct_{n}x{k}_b{bits}_w{rows}")
        if not verdict:
            raise RuntimeError("affine_product disagreed with MLX's one-row product on this Mac")
    return out.reshape(*lead, n)


def qmv_rows_probe(x, weight, tm=4, threads=64):
    """``QMV_ROWS`` called as another lane's launch would call it (each simdgroup ``tm`` output
    rows, the whole panel): for the bit checks of the device function itself."""
    c = qmv_constants(weight)
    n = weight.codes.shape[0]
    rows = x.shape[0]
    if n % tm:
        raise ValueError("probe takes output rows in whole groups of tm")
    per_group = threads // 32 * tm
    return _kernel("probe")(
        inputs=[x, weight.codes, weight.scales, weight.biases],
        template=[("W", rows), ("TM", tm), ("K", c["K"]), ("N", n), ("BITS", c["BITS"]), ("G", c["GROUP"]),
                  ("VPT", c["VPT"]), ("THREADS", threads)],
        grid=(-(-n // per_group) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
    )[0]
