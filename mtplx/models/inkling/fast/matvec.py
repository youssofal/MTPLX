"""The drafter's one-row products (its recurrent update and its folded vocabulary bias) in launches of
ours that run the library's own mat-vec, so every value is the library's.

``matvec(vec, mat, cols=, bias_col=None)``: ``vec`` ``(1, cols)`` times the first ``cols`` columns of
``mat`` ``(N, C)`` transposed, plus, with ``bias_col``, ``mat[:, bias_col]`` (``mx.addmm(mat[:, bias_col],
vec, mat[:, :cols].T)``); without it ``vec @ mat[:, :cols].T``. Returns ``(1, N)`` in ``vec``'s dtype.

The library computes these with its mat-vec kernel (``GEMVKernel``), its tiles chosen from the output
length and the input length; this launch reads that kernel's code from the installed headers
(``library_gemm``, part ``"matvec"``) and calls it with the same tiles, the same threadgroup shape
(8 or 4 simdgroups of 32 lanes, 4 output rows each, 4 inputs a lane a step) and the same arguments
(the matrix's row stride, the bias column and its stride, alpha and beta of one). The first use of
each shape on this Mac is proven against the library's own launch on the real operands (counted
``proved:`` / ``refused:matvec_*``); a refusal keeps the library's launch.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_SOURCE = """
    // One threadgroup of the library's mat-vec: BM simdgroups, each TM output rows over the whole input.
    using MatVec = GEMVKernel<T, BM, 1, 1, 32, TM, 4, AXPBY, float>;
    threadgroup float unused_reduce[1];           // one simdgroup across the input: no reduction
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const constant int& in_len = sizes[0];
    const constant int& out_len = sizes[1];
    const constant int& row_stride = sizes[2];
    const constant float& one = unit[0];
    const uint3 group_at = uint3(threadgroup_position_in_grid.x, 0, 0);
    const uint3 thread_at = uint3(lane, 0, sg);
    MatVec::run(mat, vec, mat + BIAS_AT, out, in_len, out_len, row_stride, one, one, row_stride,
                &unused_reduce[0], group_at, thread_at, uint(sg), uint(lane));
"""

_ON = os.environ.get("MTPLX_INKLING_DRAFT_MATVEC", "1") != "0"
_verdicts: dict = {}
_sizes: dict = {}


@lru_cache(maxsize=1)
def _kernel():
    from . import library_gemm

    text, why = library_gemm.library("matvec")
    if text is None:
        return None, why
    return mx.fast.metal_kernel(name="mtplx_inkling_fast_draft_matvec", input_names=["mat", "vec", "sizes", "unit"],
                                output_names=["out"], header=text, source=_SOURCE), None


def _tiles(out_len, in_len):
    """The library's mat-vec tiles for ``out = mat (out_len x in_len) . vec`` (non-transposed matrix):
    simdgroups a threadgroup and output rows a simdgroup, or ``None`` outside the shapes served here."""
    if out_len < 4 or in_len <= 64 or in_len >= 16 * out_len or in_len % 4:
        return None          # the library takes other tiles there (more simdgroups across the input)
    return (8 if out_len >= 4096 else 4), 4


def matvec(vec, mat, *, cols, bias_col=None):
    from .attn_input import count_route

    out_len, width = mat.shape

    def reference():
        if bias_col is None:
            return vec @ mat[:, :cols].T
        return mx.addmm(mat[:, bias_col], vec, mat[:, :cols].T)

    key = (out_len, width, cols, bias_col, str(vec.dtype), str(mat.dtype))
    verdict = _verdicts.get(key)
    if verdict is False or not _ON:
        return reference()
    tiles = _tiles(out_len, cols)
    kernel, why = _kernel()
    if (tiles is None or kernel is None or vec.shape != (1, cols) or vec.dtype != mat.dtype
            or mat.dtype not in (mx.bfloat16, mx.float16, mx.float32) or (bias_col is not None and bias_col < cols)):
        _verdicts[key] = False
        count_route("refused:matvec_" + "_".join(map(str, key[:4])))
        return reference()
    bm, tm = tiles
    sizes = _sizes.get((cols, out_len, width))
    if sizes is None:
        sizes = _sizes[(cols, out_len, width)] = (mx.array([cols, out_len, width], mx.int32),
                                                  mx.array([1.0], mx.float32))
    out = kernel(
        inputs=[mat, vec, *sizes],
        template=[("T", mat.dtype), ("BM", bm), ("TM", tm), ("AXPBY", bias_col is not None),
                  ("BIAS_AT", 0 if bias_col is None else bias_col)],
        grid=(-(-out_len // (bm * tm)) * bm * 32, 1, 1),
        threadgroup=(bm * 32, 1, 1),
        output_shapes=[(1, out_len)],
        output_dtypes=[mat.dtype],
    )[0]
    if verdict is None:
        try:
            want = reference()
            mx.eval(out, want)
            view = {2: mx.uint16, 4: mx.uint32}[out.dtype.size]
            verdict = out.shape == want.shape and bool(mx.array_equal(out.view(view), want.view(view)).item())
        except RuntimeError:
            verdict = False
        _verdicts[key] = verdict
        count_route(("proved:" if verdict else "refused:") + "matvec_" + "_".join(map(str, key[:4])))
        if not verdict:
            return reference()
    return out
