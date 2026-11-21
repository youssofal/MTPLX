"""CPU checks of the fast decode state: ring addressing, panel writes, growth, and exact
conversion to and from the reference cache objects."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.models.inkling.cache import WindowCache
from mtplx.models.inkling.fast.state import BANK_STEP, MAX_PANEL, FastState, slots


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


def rows(start, count, width, scale=1.0):
    return (mx.arange(start, start + count, dtype=mx.float32)[:, None] * scale
            + mx.arange(width, dtype=mx.float32)[None, :] / 100)


def kv_rows(start, count):
    value = rows(start, count, HEADS * DIM).reshape(count, HEADS, DIM).transpose(1, 0, 2)[None]
    return value.astype(mx.bfloat16), (value + 0.5).astype(mx.bfloat16)


def reference_cache(model, length, chunks):
    """Reference objects fed with `length` tokens in the given chunk sizes."""
    cache = []
    for layer in model.layers:
        kv = WindowCache(max_size=WINDOW) if layer.sliding else KVCache()
        conv = ArraysCache(4)
        cache.extend((kv, conv))
    done = 0
    for size in chunks:
        k, v = kv_rows(done, size)
        for visit in range(len(model.layers)):
            cache[2 * visit].update_and_fetch(k, v)
            conv = cache[2 * visit + 1]
            for slot, width in enumerate(WIDTHS):
                history = rows(done - 3, size + 3, width)
                conv[slot] = history[None, -3:]
        done += size
    assert done == length
    return cache


def test_slots_wrap_and_reach_back():
    assert slots(mx.array(5, mx.uint32), 4, 6).tolist() == [5, 0, 1, 2]
    assert slots(0, 3, 35, -3).tolist() == [32, 33, 34]
    assert slots(mx.array(40, mx.uint32), 3, 35, -3).tolist() == [2, 3, 4]


@pytest.mark.parametrize("chunks", [(5,), (3, 1, 1, 1), (12,), (4, 1, 6, 1, 1), (20, 1)])
def test_round_trip_is_exact(chunks):
    model = fake_model()
    length = sum(chunks)
    cache = reference_cache(model, length, chunks)
    state = FastState.from_reference(model, cache)
    assert state.length == length and int(state.pos.item()) == length
    back = state.to_reference(model)
    for visit, layer in enumerate(model.layers):
        a, b = cache[2 * visit], back[2 * visit]
        if layer.sliding:
            ka = a._temporal_order(a.keys)[:, :, -min(length, WINDOW):]
            kb = b._temporal_order(b.keys)
        else:
            ka, kb = a.keys[:, :, :length], b.keys[:, :, :length]
        assert mx.array_equal(ka, kb).item()
        assert b.offset == length
        for slot in range(4):
            assert mx.array_equal(cache[2 * visit + 1][slot], back[2 * visit + 1][slot]).item()


def test_reference_decode_continues_identically_after_round_trip():
    model = fake_model()
    cache = reference_cache(model, 11, (11,))
    back = FastState.from_reference(model, cache).to_reference(model)
    k, v = kv_rows(11, 1)
    for visit, layer in enumerate(model.layers):
        a, b = cache[2 * visit], back[2 * visit]
        ka, _ = a.update_and_fetch(k, v)
        kb, _ = b.update_and_fetch(k, v)
        if layer.sliding:
            ka, kb = a._temporal_order(ka), b._temporal_order(kb)
        assert mx.array_equal(ka, kb).item()


def test_panel_rows_ahead_of_pos_leave_committed_reads_alone():
    model = fake_model()
    state = FastState.from_reference(model, reference_cache(model, 10, (10,)))
    before = [state.history_keys(v, 0)[0] for v in range(3)]
    taps_before = [state.tap_history(v, 2) for v in range(3)]
    k, v = kv_rows(1000, 5)
    for visit in range(3):
        state.write_kv(visit, k, v)
        state.write_taps(visit, 2, rows(1000, 5, 6)[None])
    for visit in range(3):
        assert mx.array_equal(state.history_keys(visit, 0)[0], before[visit]).item()
        assert mx.array_equal(state.tap_history(visit, 2), taps_before[visit]).item()
    state.commit(2)
    assert state.length == 12
    keys, _ = state.history_keys(0, 0)
    assert mx.array_equal(keys[:, :, -2:], k[:, :, :2]).item()
    assert mx.array_equal(state.tap_history(0, 2)[:, -2:], rows(1000, 2, 6)[None]).item()


def test_growth_keeps_history_and_happens_in_steps():
    model = fake_model()
    state = FastState.allocate(model)
    assert state.cap == BANK_STEP
    k, v = kv_rows(0, 1)
    state.write_kv(1, k, v)
    state.commit(1)
    state.length = BANK_STEP - MAX_PANEL
    assert not state.grow(1) and state.cap == BANK_STEP
    state.length += 1
    assert state.grow(1) and state.cap == 2 * BANK_STEP
    assert mx.array_equal(state.keys[1][:, :, :1], k).item()
    with pytest.raises(ValueError):
        state.grow(MAX_PANEL + 1)
