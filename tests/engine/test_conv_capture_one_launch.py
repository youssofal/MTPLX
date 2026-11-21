"""On the engine: a verify's overlapping row states share the tape at every width, and replay bit for bit.

The stock convolution capture used to stack one slice of the tape for each verify row, a copy a row.
Recorded and replayed here with a depthwise convolution of the model's kind and no model loaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import gdn_capture
from mtplx.engine import runtime
from mtplx.engine.selftest import check, record

pytestmark = pytest.mark.skipif(
    not runtime.available(), reason=f"native engine unavailable: {runtime.unavailable()}"
)

CHANNELS, KEEP = 256, 3


def _gdn():
    mx.random.seed(11)
    conv = nn.Conv1d(CHANNELS, CHANNELS, kernel_size=KEEP + 1, groups=CHANNELS, bias=False)
    conv.set_dtype(mx.bfloat16)
    mx.eval(conv.parameters())
    return SimpleNamespace(conv1d=conv)


def _inputs(rows: int):
    def make(seed: int):
        qkv = mx.random.normal((1, rows, CHANNELS), key=mx.random.key(seed)).astype(mx.bfloat16)
        state = mx.random.normal((1, KEEP, CHANNELS), key=mx.random.key(seed + 100)).astype(mx.bfloat16)
        return [qkv, state]

    return make


def _capture(gdn):
    def fn(qkv, state):
        conv_out, conv_states = gdn_capture._stock_conv1d_capture(qkv, state, gdn)
        return [conv_out, conv_states, mx.contiguous(conv_states[:, -1, :, :])]

    return fn


def _stacked(gdn):
    def fn(qkv, state):
        rows = qkv.shape[1]
        tape = mx.concatenate([state, qkv], axis=1)
        conv_out = nn.silu(gdn.conv1d(tape))
        conv_states = mx.stack([tape[:, row + 1 : row + 1 + KEEP, :] for row in range(rows)], axis=1)
        return [conv_out, conv_states, mx.contiguous(conv_states[:, -1, :, :])]

    return fn


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7])
def test_the_row_views_replay_exactly_at_every_width(rows: int) -> None:
    engine = runtime.engine()
    gdn = _gdn()
    program, trace, _ = record(engine, _capture(gdn), _inputs(rows), name=f"conv-capture-{rows}")
    assert program.launches == trace.launches
    problem = check(engine, program, _capture(gdn), _inputs(rows))
    assert problem == "", problem
    # The same bits as the stacked form, on inputs the program was not recorded on.
    for seed in (5, 6):
        got = _capture(gdn)(*_inputs(rows)(seed))
        want = _stacked(gdn)(*_inputs(rows)(seed))
        mx.eval(got, want)
        assert all(mx.array_equal(a, b).item() for a, b in zip(got, want))
    stacked, _, _ = record(engine, _stacked(gdn), _inputs(rows), name=f"conv-capture-stacked-{rows}")
    assert stacked.launches - program.launches >= rows - 1, (stacked.launches, program.launches)
    # Isolate the capture itself from packing a particular consumer's output.
    def omit_row_views(qkv, state):
        out, _, last = _capture(gdn)(qkv, state)
        return [out, last]

    without_views, _, _ = record(engine, omit_row_views, _inputs(rows), name=f"last-state-{rows}")
    assert program.launches == without_views.launches



def test_the_launches_do_not_grow_with_the_width() -> None:
    engine = runtime.engine()
    gdn = _gdn()
    counts = {
        rows: record(engine, _capture(gdn), _inputs(rows), name=f"conv-capture-width-{rows}")[0].launches
        for rows in (2, 4, 7)
    }
    assert len(set(counts.values())) == 1, counts
