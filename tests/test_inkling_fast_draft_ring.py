"""A drafter window block's multi-row observe written straight into its ring (fast/draft_ring.py), GPU.

The reference is the generic path as the round program runs it: ``update_and_fetch`` of the rows, then,
at the next round, ``round_program._make_ring`` wherever the entry is no longer a ring. After every
observe the ring written here must equal that, bit for bit, with the same offset and ``_idx``: rings
not yet full, full, wrapped at every slot, observes that cross the window, and long mixed sequences of
one-row (generic, in place) and several-row observes.
"""

import random

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

H, D = 8, 128


def _bits(a, b):
    a, b = mx.contiguous(a), mx.contiguous(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def _rows(n, seed, dtype=mx.bfloat16):
    k1, k2 = mx.random.split(mx.random.key(seed))
    return (mx.random.normal((1, H, n, D), key=k1).astype(dtype), mx.random.normal((1, H, n, D), key=k2).astype(dtype))


def _settle(entry, window):
    """What the next round's draft_inputs does: a ring wherever the entry is not one."""
    from mtplx.models.inkling.fast.round_program import _is_ring, _make_ring

    if not _is_ring(entry, window):
        _make_ring(entry, window)


def _start(window, offset, seed):
    from mtplx.models.inkling.cache import WindowCache

    entry = WindowCache(max_size=window)
    if offset:
        k, v = _rows(offset, seed)
        entry.update_and_fetch(k, v)
        _settle(entry, window)
    mx.eval(entry.keys, entry.values)
    return entry


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    from mtplx.models.inkling.fast import draft_ring

    monkeypatch.setattr(draft_ring, "_VERDICT", {})
    monkeypatch.delenv("MTPLX_INKLING_DRAFT_RING", raising=False)


@needs
@pytest.mark.parametrize("window", [16, 1024])
@pytest.mark.parametrize("kept", [2, 3, 5, 8])
def test_one_observe_equals_the_generic_path(window, kept):
    from mtplx.models.inkling.fast.draft_ring import observe_rows

    offsets = sorted({1, 3, window - 9, window - kept, window - 1, window, window + 1, window + 5, 3 * window - 2})
    for offset in [o for o in offsets if o >= 1]:
        for singles in (0, 1, 4, window // 2 + 3):       # one-row observes after the ring was made: wraps
            ref = _start(window, offset, 7 * offset + kept)
            for i in range(singles):
                k1, v1 = _rows(1, 1000 + i)
                ref.update_and_fetch(k1, v1)
            mx.eval(ref.keys, ref.values)
            ours = _start(window, offset, 7 * offset + kept)
            for i in range(singles):
                k1, v1 = _rows(1, 1000 + i)
                ours.update_and_fetch(k1, v1)
            k, v = _rows(kept, 99 + offset)
            ref.update_and_fetch(k, v)
            _settle(ref, window)
            assert observe_rows(ours, k, v), "declined"
            mx.eval(ref.keys, ref.values, ours.keys, ours.values)
            where = f"window {window} offset {offset} singles {singles} kept {kept}"
            assert ours.offset == ref.offset and ours._idx == ref._idx, where
            assert _bits(ours.keys, ref.keys) and _bits(ours.values, ref.values), where


@needs
@pytest.mark.parametrize("window", [16, 64])
def test_long_mixed_sequences(window):
    from mtplx.models.inkling.fast.draft_ring import observe_rows

    rng = random.Random(window)
    ref, ours = _start(window, 5, 1), _start(window, 5, 1)
    for step in range(120):
        kept = rng.choice([1, 1, 2, 3, 4, 8])
        k, v = _rows(kept, 5000 + step)
        ref.update_and_fetch(k, v)
        _settle(ref, window)
        if kept == 1:
            ours.update_and_fetch(k, v)
        else:
            assert observe_rows(ours, k, v)
        _settle(ours, window)
        mx.eval(ref.keys, ref.values, ours.keys, ours.values)
        assert ours.offset == ref.offset and ours._idx == ref._idx, f"step {step}"
        assert _bits(ours.keys, ref.keys) and _bits(ours.values, ref.values), f"step {step}"


@needs
def test_declines_leave_the_entry_alone(monkeypatch):
    from mtplx.models.inkling.cache import WindowCache
    from mtplx.models.inkling.fast.draft_ring import observe_rows

    k, v = _rows(3, 1)
    fresh_entry = WindowCache(max_size=16)
    assert not observe_rows(fresh_entry, k, v)          # no ring yet
    entry = _start(16, 20, 2)
    one_k, one_v = _rows(1, 3)
    assert not observe_rows(entry, one_k, one_v)         # one row: the cache's own in-place write
    monkeypatch.setenv("MTPLX_INKLING_DRAFT_RING", "0")
    before = (entry.offset, entry._idx)
    assert not observe_rows(entry, k, v)
    assert (entry.offset, entry._idx) == before
