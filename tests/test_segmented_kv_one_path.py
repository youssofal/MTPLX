"""With MTPLX_SEGMENTED_KV on, the full-attention layers are segment caches from the first token: one path, no context fence."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import mtplx.segmented_kv as segmented_kv_module
from mtplx.cache_state import (
    configure_tail_owned_attention_kv_cache,
    restore_cache,
    snapshot_cache,
)
from mtplx.segmented_kv import (
    SegmentedKVCache,
    SegmentedKVState,
    route_counts,
)

HEADS = 2
DIM = 8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MAX_SEGMENTS", "1000")
    # segmented_kv_enabled() needs a verdict on the loaded model (without one it stays off).
    monkeypatch.setattr(segmented_kv_module, "_MODEL_SUPPORT", {"supported": True, "reasons": []})


def _rows(seed: int, n: int):
    k = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)
    v = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed + 50_000)).astype(mx.bfloat16)
    return k, v


def _same(a, b) -> bool:
    ka, va = a
    kb, vb = b
    return bool(mx.array_equal(ka, kb).item() and mx.array_equal(va, vb).item())


def _target_cache(n_layers: int = 2) -> list:
    cache = [KVCache() for _ in range(n_layers)]
    configure_tail_owned_attention_kv_cache(cache)
    return cache


def test_segments_from_the_first_token_whatever_the_compiled_verify_settings(monkeypatch) -> None:
    for compiled in (None, "1"):
        if compiled:
            monkeypatch.setenv("MTPLX_COMPILED_VERIFY", compiled)
            monkeypatch.setenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "32768")
        cache = _target_cache()
        assert all(isinstance(c, SegmentedKVCache) for c in cache)
        assert all(c.offset == 0 for c in cache)


def test_a_layer_that_already_holds_rows_stays_stock() -> None:
    cache = [KVCache()]
    cache[0].update_and_fetch(*_rows(1, 60))
    configure_tail_owned_attention_kv_cache(cache)
    assert type(cache[0]) is KVCache


def test_segmented_snapshot_restores_as_a_reference() -> None:
    src = _target_cache(1)
    src[0].update_and_fetch(*_rows(1, 150))
    snap = snapshot_cache(src)
    assert isinstance(snap.states[0], SegmentedKVState)
    dest = _target_cache(1)
    restore_cache(dest, snap, clone_states=False)
    assert isinstance(dest[0], SegmentedKVCache)
    assert dest[0].attention_segments()[0][0] is src[0].attention_segments()[0][0]


def test_stock_snapshot_restores_into_a_segment_cache_and_extends_like_stock() -> None:
    control = KVCache()
    stock = [KVCache()]
    for seed, n in ((1, 90), (2, 90)):
        stock[0].update_and_fetch(*_rows(seed, n))
        control.update_and_fetch(*_rows(seed, n))
    dest = _target_cache(1)
    before = route_counts.get("restored_stock_as_segment", 0)
    restore_cache(dest, snapshot_cache(stock))
    assert route_counts.get("restored_stock_as_segment", 0) == before + 1
    assert isinstance(dest[0], SegmentedKVCache) and dest[0].offset == 180
    assert _same(dest[0].gather(), control.state)
    dest[0].update_and_fetch(*_rows(9, 17))
    control.update_and_fetch(*_rows(9, 17))
    assert _same(dest[0].gather(), control.state)
    dest[0].trim(5)
    control.trim(5)
    assert _same(dest[0].gather(), control.state)


def test_a_stock_buffer_restored_as_a_segment_is_stored_as_exact_rows() -> None:
    """A stock buffer has capacity beyond its rows; the segment must not keep a strided view
    (MLX copies a non-contiguous kernel input on every launch)."""
    stock = [KVCache()]
    stock[0].update_and_fetch(*_rows(1, 300))  # stock capacity: 512 rows
    assert stock[0].keys.shape[2] > 300
    dest = _target_cache(1)
    restore_cache(dest, snapshot_cache(stock))
    segment = dest[0].attention_segments()[0][0]
    assert segment.shape[2] == 300
    flat = mx.contiguous(segment)
    mx.eval(flat)
    assert mx.array_equal(flat, segment).item()


def test_a_segmented_snapshot_restores_as_rows_into_a_stock_cache() -> None:
    src = _target_cache(1)
    src[0].update_and_fetch(*_rows(1, 150))
    snap = snapshot_cache(src)
    dest = [KVCache()]  # an MTP history cache never went through the install
    restore_cache(dest, snap)
    assert type(dest[0]) is KVCache and dest[0].offset == 150
    assert _same(dest[0].state, src[0].gather())


def test_stock_caches_with_rows_are_never_converted() -> None:
    mtp = [KVCache()]
    mtp[0].update_and_fetch(*_rows(1, 150))
    snap = snapshot_cache(mtp)
    dest = [KVCache()]
    restore_cache(dest, snap)
    assert type(dest[0]) is KVCache
    configure_tail_owned_attention_kv_cache(mtp)
    assert type(mtp[0]) is KVCache


def test_snapshot_does_not_copy_the_history() -> None:
    rows = 4096
    cache = _target_cache(1)
    cache[0].update_and_fetch(*_rows(1, rows))
    mx.eval(cache[0].keys, cache[0].values)
    history = 2 * HEADS * rows * DIM * 2
    mx.synchronize()
    before = mx.get_active_memory()
    snap = snapshot_cache(cache)
    cache[0].append_rows(*_rows(2, 64))
    mx.eval(cache[0]._tail.keys, cache[0]._tail.values)
    grew = mx.get_active_memory() - before
    assert isinstance(snap.states[0], SegmentedKVState)
    assert grew < history // 4
