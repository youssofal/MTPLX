"""The cross-pass message reads of a whole loop pass, read once for its twelve visits.

On loop passes two and three every visit reads a message from the pass's normalised stack of
prior states with its layer's four queries: scores ``q (4, D) @ stack^T``, RC6's scale and
softmax over the ``P`` prior states, then the softmax weights times the stack (``projection.py``
``stock_messages``; ``serial_messages`` row by row for a verify panel). The stack is built once at
the pass's start and every visit of the pass reads the same one, so nothing a visit computes
feeds another visit's message: the reads only wait on each other because each sits in its own
visit. As a visit's own chain they cost five launches each, among them a split-K product on two
threadgroups, 24 such reads a step.

Here a pass reads them together, with the same arithmetic per element:

* the scores of all twelve layers as one product, ``(12 * 4, D) @ stack^T`` with the panel's rows
  joined as columns (as ``read.scores`` joins them for one visit);
* RC6's scale and softmax per (visit, row, query) in one launch, written visit-major;
* the mix as one product batched over the visits and rows.

Each visit's rows are then a contiguous slice of the result, in the layout ``read.message_rows``
returns, handed to the visit's message (``attach``): stages A and D and the reference stages read
them as they read their own.

Exactness (class A against RC6). The scale and softmax repeat lane L1's ``probs`` kernel element
for element (``read._SHARE``, MLX's single-simdgroup softmax). Whether one product over the
twelve layers gives each layer's rows the bits of its own product depends on which kernel MLX
picks for the shapes on this GPU, so it is decided on the first call of each shape on this Mac,
against RC6's own message read of every visit (``stock_messages`` / ``cross_pass_message``) and
against lane L1's ``read.message_rows``: all twelve visits' rows must equal both bit for bit. A
shape that differs keeps the per-visit reads, counted (``fallback.pass_messages.<why>``).

``MTPLX_INKLING_PASS_MESSAGES=0`` turns it off (the per-visit reads, as before).
"""

from __future__ import annotations

from functools import lru_cache
import os
import struct

import mlx.core as mx

from .routes import count

#: Queries a visit's message read takes (attention, key, value and MLP controllers).
QUERIES = 4


def enabled():
    return os.environ.get("MTPLX_INKLING_PASS_MESSAGES", "1").strip().lower() not in ("0", "off", "false", "no")


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


_SOURCE = """
    // One simdgroup per (visit, panel row, query). The scores arrive as one (visits * 4, W * S)
    // array: visit k's query q is row 4 k + q, panel row r's prior states are columns r S .. r S + S - 1.
    // Each score is multiplied by the scale on its own (RC6 rounds that product), then the
    // single-simdgroup softmax runs over the row's S values; the weights are written visit-major,
    // (visits, W, 4, S), so a visit's rows are one contiguous block.
    const uint lane = thread_index_in_simdgroup;
    const uint g = threadgroup_position_in_grid.x;
    const uint q = g % NQ;
    const uint r = (g / NQ) % W;
    const uint k = g / (NQ * W);
    const uint from = (k * NQ + q) * (W * S) + r * S;
    float slot[4];
    for (uint i = 0; i < 4; ++i) {
        const uint s = lane * 4 + i;
        slot[i] = s < S ? scores[from + s] * as_type<float>(uint(SCALE)) : -INFINITY;
    }
    share_four(slot);
    const uint to = ((k * W + r) * NQ + q) * S;
    for (uint i = 0; i < 4; ++i) {
        const uint s = lane * 4 + i;
        if (s < S) weights[to + s] = slot[i];
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    from .read import _SHARE

    return mx.fast.metal_kernel(
        name="mtplx_inkling_pass_message_weights",
        input_names=["scores"],
        output_names=["weights"],
        header=_SHARE,
        source=_SOURCE,
    )


def batched_rows(stack, table):
    """Every visit's message rows from one scores product, one softmax launch and one mix product.

    ``stack`` ``(1, W, P, D)`` float32 (the pass's normalised prior states), ``table``
    ``(visits, 4, D)`` float32 (the loop layers' message queries, ``Model.message_queries``).
    Returns one ``(1, W, 4, D)`` float32 array a visit (contiguous slices of one result)."""
    rows, count_, width = stack.shape[1], stack.shape[2], stack.shape[3]
    visits = table.shape[0]
    if count_ > 128:
        raise ValueError("the pass softmax serves at most 128 prior states (one simdgroup a row)")

    def library():
        scores = table.reshape(visits * QUERIES, width) @ stack.reshape(rows * count_, width).T
        weights = _kernel()(
            inputs=[scores],
            template=[("W", rows), ("S", count_), ("NQ", QUERIES), ("SCALE", _f32_bits(width ** -0.5))],
            grid=(visits * rows * QUERIES * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(visits, rows, QUERIES, count_)],
            output_dtypes=[mx.float32],
        )[0]
        return weights @ stack  # (visits, W, 4, P) @ (1, W, P, D): batched over visits and rows

    # The scores, softmax and mix as our launches where proven (nax_reads.py), else the library's
    # two products around the softmax.
    from .nax_reads import message_read

    mixed = message_read(table, stack, library)
    return [mixed[k : k + 1] for k in range(visits)]


#: (W, P, D, visits) -> True when the batched rows equal every visit's own read on this Mac.
_VERDICT: dict = {}


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(mx.uint32), mx.contiguous(b).view(mx.uint32)).item())


def _proof(stack, table, verify):
    """The batched rows against RC6's message read and lane L1's, every visit, bit for bit.
    Evaluates on the first call of a shape only (before the engine records it)."""
    from .. import model as ref
    from ..projection import stock_messages
    from .read import message_rows

    width = stack.shape[-1]
    reference = ref.cross_pass_message if verify else stock_messages
    whole = batched_rows(stack, table)
    want = [reference(table[k], stack, width ** -0.5) for k in range(table.shape[0])]
    lane = [message_rows(stack, table[k]) for k in range(table.shape[0])]
    lane = [mx.concatenate(list(x), axis=1) if isinstance(x, (list, tuple)) else x for x in lane]
    mx.eval(whole, want, lane)
    if not all(_same(a, b) for a, b in zip(whole, want)):
        return "differs_from_reference"
    if not all(_same(a, b) for a, b in zip(whole, lane)):
        return "differs_from_lane_read"
    return None


def pass_rows(stack, table, verify):
    """Every visit's message rows of a loop pass (``batched_rows``) where proven on this Mac for
    this shape, else None (each visit reads its own, as before; counted)."""
    if not enabled():
        return None
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if stack.dtype != mx.float32 or table.dtype != mx.float32 or stack.ndim != 4 or table.ndim != 3:
        count("fallback.pass_messages.dtype_or_shape")
        return None
    rows, prior, width = stack.shape[1], stack.shape[2], stack.shape[3]
    if prior > 128 or table.shape[1] != QUERIES or table.shape[2] != width:
        count("fallback.pass_messages.dtype_or_shape")
        return None
    key = (rows, prior, width, table.shape[0], bool(verify))
    verdict = _VERDICT.get(key)
    if verdict is None:
        why = _proof(stack, table, verify)
        verdict = _VERDICT[key] = why is None
        from .attn_input import count_route

        count_route(("proved:" if verdict else f"refused:{why}:") + "pass_messages_w%d_p%d" % (rows, prior))
        if not verdict:
            _VERDICT[(key, "why")] = why
    if not verdict:
        count(f"fallback.pass_messages.{_VERDICT.get((key, 'why'), 'differs')}")
        return None
    return batched_rows(stack, table)


def attach(message, rows):
    """Give a visit's ``Message`` its rows: the fused stages read them through
    ``read.message_rows_of`` and the reference stages through ``Message.read``."""
    from . import read

    read._message_rows[message] = rows
    message._read = rows


__all__ = ["attach", "batched_rows", "enabled", "pass_rows"]
