"""CPU checks of the fast decode state: the fixed layout on the reference cache objects,
chronological history, and commits of accepted rows (window wrap, bank growth, tails)."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.models.inkling.cache import WindowCache
from mtplx.models.inkling.fast.state import BANK_STEP, DecodeState, VisitDelta


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


WINDOW, HEADS, DIM, TAPS = 8, 2, 4, 4
WIDTHS = (HEADS * DIM, HEADS * DIM, 6, 6)


def fake_model(local=(True, False, True)):
    layers = [
        SimpleNamespace(sliding=s, convs=[SimpleNamespace(weight=mx.zeros((TAPS, w))) for w in WIDTHS])
        for s in local
    ]
    config = {"num_key_value_heads": HEADS, "head_dim": DIM, "sliding_window_size": WINDOW,
              "sconv_kernel_size": TAPS}
    return SimpleNamespace(config=config, layers=layers, visit_order=tuple(range(len(local))),
                           embed_tokens=SimpleNamespace(weight=mx.zeros((1,), mx.bfloat16)))


def kv_rows(start, count):
    value = (mx.arange(start, start + count, dtype=mx.float32)[:, None]
             + mx.arange(HEADS * DIM, dtype=mx.float32)[None, :] / 100)
    value = value.reshape(count, HEADS, DIM).transpose(1, 0, 2)[None]
    return value.astype(mx.bfloat16), (value + 0.5).astype(mx.bfloat16)


def tap_rows(start, count, width):
    return (mx.arange(start, start + count, dtype=mx.float32)[:, None] * 10
            + mx.arange(width, dtype=mx.float32)[None, :])[None]


def reference_cache(model, chunks):
    """Reference objects fed through their own update paths (concat for chunks, in place for 1)."""
    cache = []
    for layer in model.layers:
        cache.extend((WindowCache(max_size=WINDOW) if layer.sliding else KVCache(), ArraysCache(4)))
    done = 0
    for size in chunks:
        k, v = kv_rows(done, size)
        for visit in range(len(model.layers)):
            cache[2 * visit].update_and_fetch(k, v)
            for slot, width in enumerate(WIDTHS):
                cache[2 * visit + 1][slot] = tap_rows(done + size - 3, 3, width)
        done += size
    return cache


def chronological(kv):
    if isinstance(kv, WindowCache):
        n = min(kv.offset, kv.max_size)
        k = kv._temporal_order(kv.keys)
        return k[:, :, k.shape[2] - n:]
    return kv.keys[:, :, : kv.offset]


def delta(start, count):
    k, v = kv_rows(start, count)
    return VisitDelta(k, v, tuple(tap_rows(start, count, w) for w in WIDTHS))


@pytest.mark.parametrize("chunks", [(5,), (3, 1, 1, 1), (12,), (4, 1, 6, 1, 1), (20, 1), (1,) * 11])
def test_prepare_keeps_the_logical_state(chunks):
    model = fake_model()
    cache = reference_cache(model, chunks)
    before = [chronological(cache[2 * v]) for v in range(3)]
    state = DecodeState(model, cache)
    state.prepare()
    assert state.length == sum(chunks) and int(state.pos.item()) == sum(chunks)
    for visit in range(3):
        kv = cache[2 * visit]
        assert mx.array_equal(chronological(kv), before[visit]).item()
        if visit != 1:
            assert kv.keys.shape[2] == WINDOW and state._is_ring(kv)
    assert not state.prepare()  # already in layout: nothing changes


def test_history_is_the_reference_chronological_array():
    model = fake_model()
    cache = reference_cache(model, (12,))
    state = DecodeState(model, cache)
    state.prepare()
    k_rows, v_rows = kv_rows(100, 3)
    keys, _ = state.history(0, k_rows, v_rows)
    expected = mx.concatenate([kv_rows(12 - (WINDOW - 1), WINDOW - 1)[0], k_rows], axis=2)
    assert mx.array_equal(keys, expected).item()
    keys, _ = state.history(1, k_rows, v_rows)
    assert mx.array_equal(keys, mx.concatenate([kv_rows(0, 12)[0], k_rows], axis=2)).item()


def test_commit_of_accepted_rows_matches_the_reference_append():
    model = fake_model()
    cache = reference_cache(model, (6,))
    state = DecodeState(model, cache)
    state.prepare()
    # five commits that cross the window's physical end, with partial acceptance
    for start, rows, kept in ((6, 4, 3), (9, 2, 2), (11, 8, 1), (12, 5, 5), (17, 3, 0)):
        state.commit([delta(start, rows) for _ in range(3)], kept)
    assert state.length == 17
    ref = reference_cache(model, (6, 3, 2, 1, 5))
    for visit in range(3):
        assert mx.array_equal(chronological(cache[2 * visit]), chronological(ref[2 * visit])).item()
        assert cache[2 * visit].offset == 17
        for slot, width in enumerate(WIDTHS):
            assert mx.array_equal(cache[2 * visit + 1][slot], tap_rows(14, 3, width)).item()


def test_global_bank_grows_in_steps_and_keeps_history():
    model = fake_model((False,))
    cache = reference_cache(model, (5,))
    state = DecodeState(model, cache)
    state.prepare()
    kv = cache[0]
    state.length = kv.offset = kv.keys.shape[2] - 1
    end = state.length + 2
    state.commit([delta(state.length, 2)], 2)
    assert kv.keys.shape[2] == -(-end // BANK_STEP) * BANK_STEP and kv.offset == end
    assert mx.array_equal(kv.keys[:, :, end - 2:end], kv_rows(end - 2, 2)[0]).item()
    assert mx.array_equal(kv.keys[:, :, :5], kv_rows(0, 5)[0]).item()
    with pytest.raises(ValueError):
        state.commit([delta(0, 1)], 33)
