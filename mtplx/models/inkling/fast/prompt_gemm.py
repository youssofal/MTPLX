"""The prompt chunk's projections from one input as one launch of ours, with MLX's sums.

A prompt chunk's visit multiplies the same normed input by up to six stored 16-bit matrices (query, relative,
sink, gate, key, value) and each expert block multiplies its router input by the router: one MLX matrix
product each. ``project`` runs every matrix that shares an input in one launch: a threadgroup owns 64 rows by
128 columns of one matrix's output, and each of its eight simdgroups computes a 32 by 32 tile with the
installed MLX's own tensor-unit GEMM loop (read from its headers by ``mtplx/kernels/library_headers.py`` when the
launch is first built; nothing of it lives in this tree): the same 16-deep fragments in the same order into float32, rounded once to the operand
type as MLX's product stores it.

MLX's own product splits the reduction into parts on shapes with few output tiles (a narrow matrix at a short
chunk), and a split sum is not the same sum. So a shape (rows, columns, depth, type) is admitted only after the
launch has given MLX's product's bits twice: on operands built so that the order of the sum shows in its result
(``_ordered_operands``), and on its first call's own operands. The first alone is not enough: two orders of a
float32 sum usually round to the same 16-bit value, and on a small output every value can agree by chance (a
45-row key projection did so once and then differed on 22 of 28 later calls). A shape that does not match keeps
MLX's product, counted. ``MTPLX_INKLING_PROMPT_KERNELS=mlx`` keeps MLX's products everywhere.

Tensor-unit GPUs, bf16 or float32 operands, row-major matrices whose depth is a multiple of 32.

With the prompt read's ``invariant`` part (``fast/prompt_route.py``, off by default) every shape and every row
count takes the launch, without the comparison: each output element is then summed in the same order whatever
the rows around it, so a row's projection does not depend on the chunk width (MLX's split sums do).
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import logging
from threading import Lock

import mlx.core as mx

_log = logging.getLogger(__name__)
_lock = Lock()
_counts: Counter = Counter()
_admitted: dict = {}

MAX_MATRICES = 6
BLOCK_ROWS = 64
BLOCK_COLS = 128
THREADS = 256
PROMPT_MIN_ROWS = 33

PROJECT = """
    // T: the operands' type; MATS: matrices sharing the input; K: the depth; FULL_DEPTH: K is a multiple of 256.
    // A threadgroup: 8 simdgroups (2 down, 4 across) over a 64 x 128 block of one matrix's output.
    enum : int { SG_ROWS = 32, SG_COLS = 32, STEP = 32, DEPTH_BLOCK = 256 };
    const int block = int(threadgroup_position_in_grid.x);
    const int first_row = int(threadgroup_position_in_grid.y) * 64;
    const int rows = int(x_shape[0]);
    int which = 0;
    while (which + 1 < MATS && layout[2 * (which + 1)] <= block)
        ++which;
    const int width = layout[2 * which + 1];
    const int first_col = (block - layout[2 * which]) * 128;
    const device T* matrix = which == 0 ? w0 : which == 1 ? w1 : which == 2 ? w2 : which == 3 ? w3 : which == 4 ? w4 : w5;
    device T* product = which == 0 ? y0 : which == 1 ? y1 : which == 2 ? y2 : which == 3 ? y3 : which == 4 ? y4 : y5;
    const ushort group_id = ushort(simdgroup_index_in_threadgroup);
    const int my_row = first_row + SG_ROWS * int(group_id >> 2);
    const int my_col = first_col + SG_COLS * int(group_id & 3);
    const short live_rows = short(clamp(rows - my_row, 0, int(SG_ROWS)));
    const short live_cols = short(clamp(width - my_col, 0, int(SG_COLS)));
    const device T* lhs = x + size_t(max(min(my_row, rows - 1), 0)) * K;
    const device T* rhs = matrix + size_t(max(min(my_col, width - 1), 0)) * K;
    device T* to = product + size_t(my_row) * width + my_col;
    if (live_rows == SG_ROWS && live_cols == SG_COLS) {
        const auto sums = gemm_loop<T, SG_ROWS, SG_COLS, STEP, DEPTH_BLOCK, false, true, true, true, bool(FULL_DEPTH), float>(
            lhs, rhs, K, K, K, K / DEPTH_BLOCK, live_rows, live_cols);
        sums.store(to, width);
    } else {
        // every simdgroup runs the loop (its barriers are the threadgroup's); one with nothing to write skips it
        const auto sums = gemm_loop<T, SG_ROWS, SG_COLS, STEP, DEPTH_BLOCK, false, true, false, false, bool(FULL_DEPTH), float>(
            lhs, rhs, K, K, K, K / DEPTH_BLOCK, live_rows, live_cols);
        if (live_rows > 0 && live_cols > 0)
            sums.store_safe(to, width, short2(live_cols, live_rows));
    }
"""


def counts():
    with _lock:
        return dict(_counts)


def _count(key):
    with _lock:
        _counts[key] += 1


@lru_cache(maxsize=1)
def available():
    """(True, None) where the launch can run: tensor units and a readable GEMM header."""
    if not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        return False, "cpu"
    try:
        from mtplx import nax_detect

        if not nax_detect.nax_available():
            return False, "no_tensor_units"
    except Exception as exc:  # noqa: BLE001
        return False, f"nax_detect: {type(exc).__name__}"
    from mtplx.kernels.library_headers import text

    header, why = text("tensor_gemm")
    if header is None:
        _log.warning("Inkling prompt projections keep MLX's products: %s", why)
        return False, "headers"
    return True, None


@lru_cache(maxsize=1)
def _kernel():
    from mtplx.kernels.library_headers import text

    header, _ = text("tensor_gemm")
    return mx.fast.metal_kernel(
        name="mtplx_inkling_prompt_project",
        input_names=["x"] + ["w%d" % i for i in range(MAX_MATRICES)] + ["layout"],
        output_names=["y%d" % i for i in range(MAX_MATRICES)],
        source=PROJECT,
        header=header + "using namespace mlx::steel;\n",
    )


def _launch(x, matrices):
    rows, depth = x.shape
    widths = [w.shape[0] for w in matrices]
    firsts, total = [], 0
    for width in widths:
        firsts.append(total)
        total += -(-width // BLOCK_COLS)
    layout = []
    for i in range(MAX_MATRICES):
        layout += [firsts[i], widths[i]] if i < len(matrices) else [total, 0]
    stand = matrices + [matrices[0]] * (MAX_MATRICES - len(matrices))
    outs = _kernel()(
        inputs=[x, *stand, mx.array(layout, dtype=mx.int32)],
        template=[("T", x.dtype), ("MATS", len(matrices)), ("K", depth), ("FULL_DEPTH", depth % 256 == 0)],
        grid=(total * THREADS, -(-rows // BLOCK_ROWS), 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, w) for w in widths] + [(1,)] * (MAX_MATRICES - len(matrices)),
        output_dtypes=[x.dtype] * MAX_MATRICES,
    )
    return list(outs[: len(matrices)])


#: A value the float32 sum absorbs a 1 into (its spacing there is 2): +HUGE and -HUGE in one column of a matrix
#: cancel exactly, and the 1s summed while one of them is held are lost or rounded, so the result counts them in
#: the order and grouping the product sums them.
_HUGE = float(2**24)


def _ordered_operands(rows, depth, widths, dtype):
    """``x`` of ones and one matrix per width whose products reveal the reduction's grouping over the depth:
    each matrix row holds 1s at a random few percent of its columns and three +HUGE/-HUGE pairs at random
    columns of its own, so its exact sum is a small count of 1s and each grouping of the sum loses a different
    subset of them. Small integers, exact in 16-bit floats; two products that sum in different orders differ on
    many of the outputs."""
    key = mx.random.key(0x1A2B3C4D)
    x = mx.ones((rows, depth), dtype=dtype)
    signs = mx.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]) * _HUGE
    matrices = []
    for width in widths:
        k_ones, k_places, key = mx.random.split(key, 3)
        ones = (mx.random.uniform(shape=(width, depth), key=k_ones) < max(0.04, 16.0 / depth)).astype(mx.float32)
        places = mx.argsort(mx.random.uniform(shape=(width, depth), key=k_places), axis=1)[:, : min(6, depth)]
        huge = mx.broadcast_to(signs[: places.shape[1]], places.shape)
        matrices.append(mx.put_along_axis(ones, places, huge, axis=1).astype(dtype))
    return x, matrices


def _same(got, want, dtype):
    word = mx.uint32 if dtype == mx.float32 else mx.uint16
    return all(bool(mx.array_equal(a.view(word), b.view(word)).item()) for a, b in zip(got, want))


def _invariant():
    from .prompt_route import invariant_sums

    return invariant_sums()


def _unserved(x, matrices, invariant=False):
    import os

    if os.environ.get("MTPLX_INKLING_PROMPT_KERNELS", "ours").strip().lower() in ("mlx", "0", "off", "stock"):
        return "disabled"
    rows = x.size // x.shape[-1]
    if rows < (1 if invariant else PROMPT_MIN_ROWS):
        return "rows"
    if not available()[0]:
        return "unavailable"
    if not 1 <= len(matrices) <= MAX_MATRICES:
        return "matrices"
    if x.dtype not in (mx.bfloat16, mx.float32) or x.shape[-1] % 32:
        return "operand"
    for w in matrices:
        if not isinstance(w, mx.array) or w.ndim != 2 or w.dtype != x.dtype or w.shape[1] != x.shape[-1]:
            return "matrix"
    return None


def project(x, matrices, where):
    """``[x @ w.T for w in matrices]`` (each ``(…, N_i)`` in x's type), as one launch where every shape has been
    admitted, else as MLX's products (counted under ``where``). ``x``: ``(…, K)``; ``matrices``: ``(N_i, K)``."""
    invariant = _invariant()
    reason = _unserved(x, matrices, invariant)
    if reason is not None:
        _count(f"{where}.mlx.{reason}")
        return [x @ w.T for w in matrices]
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    if invariant:
        # one order for every row count: no comparison with MLX's product, whose order moves with the shape
        _count(f"{where}.ours.invariant")
        return [o.reshape(*lead, o.shape[-1]) for o in _launch(flat, matrices)]
    key = (flat.shape[0], flat.shape[1], str(x.dtype), tuple(w.shape[0] for w in matrices))
    with _lock:
        verdict = _admitted.get(key)
    if verdict is None:
        px, pm = _ordered_operands(flat.shape[0], flat.shape[1], [w.shape[0] for w in matrices], x.dtype)
        p_want = [px @ w.T for w in pm]
        p_got = _launch(px, pm)
        want = [flat @ w.T for w in matrices]
        got = _launch(flat, matrices)
        mx.eval(p_want, p_got, want, got)
        ordered = _same(p_got, p_want, x.dtype)
        verdict = ordered and _same(got, want, x.dtype)
        with _lock:
            _admitted[key] = verdict
        if not verdict:
            _log.info("Inkling prompt projection %s keeps MLX's product: its sums differ there (%s)", key,
                      "own operands" if ordered else "order")
        out = got if verdict else want
    elif verdict:
        out = _launch(flat, matrices)
    else:
        out = [flat @ w.T for w in matrices]
    _count(f"{where}.{'ours' if verdict else 'mlx.differs'}")
    return [o.reshape(*lead, o.shape[-1]) for o in out]
