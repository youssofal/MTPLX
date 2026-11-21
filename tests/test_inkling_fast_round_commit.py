"""The commit inside Inkling's round program writes exactly what the stock commit writes.

Banks are compared up to the new committed length (the round writes all of its rows there and
the rows past the kept ones are not state).

``round_program._commit_target`` (rings masked by the kept count, banks at the committed
length, convolution tails through one kernel per visit) against the stock commit's own
statements (``fast/state.py`` ``DecodeState.commit``), on small random states: every kept
count from 0 to the panel's rows, rings that wrap, banks with room. Needs Metal (the kernels).
"""

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.inkling.fast import round_program
from mtplx.models.inkling.fast.state import VisitDelta

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

HEADS, WINDOW, DIM, CAP, KEEP = 2, 8, 4, 32, 3
WIDTHS = (4, 4, 6, 5)


class _State:
    """The few things ``_commit_target`` reads from a ``RoundState``."""

    def __init__(self, local, rings, banks, tails, newest, pos):
        self.local, self.window = local, WINDOW
        self._rings, self._banks, self._tails = rings, banks, tails
        self.newest = mx.array(newest, mx.uint32)
        self.pos = mx.array(pos, mx.uint32)

    def window_view(self, visit):
        k, v = self._rings[visit]
        return k, v, self.newest, mx.array(WINDOW, mx.uint32)

    def global_view(self, visit):
        k, v = self._banks[visit]
        return k, v, self.pos

    def tail(self, visit, slot):
        return self._tails[visit][slot]


def _f32(a):
    return np.array(a.astype(mx.float32))


def _stock(local, rings, banks, tails, deltas, kept, newest, pos):
    """``DecodeState.commit``'s statements on copies (window ``_idx`` = newest + 1)."""
    out = []
    for visit, is_local in enumerate(local):
        d = deltas[visit]
        if is_local:
            k, v = (_f32(a) for a in rings[visit])
            first = (newest + 1) % WINDOW
            for i in range(kept):
                k[:, :, (first + i) % WINDOW] = _f32(d.keys)[:, :, i]
                v[:, :, (first + i) % WINDOW] = _f32(d.values)[:, :, i]
        else:
            k, v = (_f32(a) for a in banks[visit])
            k[:, :, pos : pos + kept] = _f32(d.keys)[:, :, :kept]
            v[:, :, pos : pos + kept] = _f32(d.values)[:, :, :kept]
        new_tails = []
        for slot in range(4):
            tail, rows = np.array(tails[visit][slot]), np.array(d.taps[slot])
            if kept >= KEEP:
                new_tails.append(rows[:, kept - KEEP : kept])
            else:
                new_tails.append(np.concatenate([tail[:, kept:], rows[:, :kept]], axis=1))
        out += [k, v, *new_tails]
    return out


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("newest", [2, 6, 7])
def test_commit_inside_matches_the_stock_commit(rows, newest):
    key = mx.random.key(rows * 31 + newest)
    local = (True, False, True)
    rings, banks, tails, deltas = {}, {}, {}, []
    for visit, is_local in enumerate(local):
        key, *ks = mx.random.split(key, 13)
        if is_local:
            rings[visit] = (mx.random.normal((1, HEADS, WINDOW, DIM), key=ks[0]).astype(mx.bfloat16),
                            mx.random.normal((1, HEADS, WINDOW, DIM), key=ks[1]).astype(mx.bfloat16))
        else:
            banks[visit] = (mx.random.normal((1, HEADS, CAP, DIM), key=ks[0]).astype(mx.bfloat16),
                            mx.random.normal((1, HEADS, CAP, DIM), key=ks[1]).astype(mx.bfloat16))
        tails[visit] = [mx.random.normal((1, KEEP, w), key=ks[2 + s]) for s, w in enumerate(WIDTHS)]
        deltas.append(VisitDelta(
            mx.random.normal((1, HEADS, rows, DIM), key=ks[6]).astype(mx.bfloat16),
            mx.random.normal((1, HEADS, rows, DIM), key=ks[7]).astype(mx.bfloat16),
            tuple(mx.random.normal((1, rows, w), key=ks[8 + s]) for s, w in enumerate(WIDTHS)),
        ))
    pos = 11
    mx.eval(list(rings.values()), list(banks.values()), list(tails.values()), [d.arrays() for d in deltas])
    for kept in range(rows + 1):
        st = _State(local, rings, banks, tails, newest, pos)
        new, undo = round_program._commit_target(st, deltas, mx.array(kept, mx.uint32), rows)
        mx.eval(new, undo)
        expected = _stock(local, rings, banks, tails, deltas, kept, newest, pos)
        assert len(new) == len(expected)
        for got, want in zip(new, expected):
            got, want = _f32(got), want.astype(np.float32)
            if got.ndim == 4 and got.shape[2] == CAP:
                # a bank: the round writes every row at the committed length; the rows past the
                # kept ones lie past the new length, where nothing reads them
                got, want = got[:, :, : pos + kept], want[:, :, : pos + kept]
            assert np.array_equal(got, want), (rows, newest, kept)
        # the undo rows are the slots' rows before the round
        j = 0
        for visit, is_local in enumerate(local):
            if not is_local:
                continue
            for array, old in zip(rings[visit], undo[2 * j : 2 * j + 2]):
                slots = [(newest + 1 + i) % WINDOW for i in range(rows)]
                assert np.array_equal(_f32(old), _f32(array)[:, :, slots]), (rows, newest, kept)
            j += 1
