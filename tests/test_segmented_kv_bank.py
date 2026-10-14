"""Segmented KV in the session bank and the admission gate (MTPLX_SEGMENTED_KV).

Follow-up turns, a fork, a return to an earlier turn, memory that grows only with new tokens while
snapshots live, eviction, and the admission pricing. The same flow on stock KVCache objects is the
control that shows the duplicate this removes.
"""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import mtplx.segmented_kv as segmented_kv_module
from mtplx.segmented_kv import SegmentedKVCache, SegmentedKVState
from mtplx.session_bank import SessionBank, entry_has_segmented_kv

LAYERS, HEADS, DIM = 4, 4, 256
BASE, TURN = 4096, 512
ROW_BYTES = LAYERS * HEADS * DIM * 2 * 2  # K and V, bf16


@pytest.fixture(autouse=True)
def _model_checked(monkeypatch):
    # segmented_kv_enabled() needs a verdict on the loaded model (without one it stays off).
    monkeypatch.setattr(segmented_kv_module, "_MODEL_SUPPORT", {"supported": True, "reasons": []})


class Runtime:
    model_path = Path("models/example")
    mtp_enabled = False

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def make_cache(self):
        return [SegmentedKVCache() if self.kind == "segmented" else KVCache() for _ in range(LAYERS)]


def _rows(seed: int, n: int):
    k = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)
    v = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed + 7)).astype(mx.bfloat16)
    return k, v


def _write(cache, seed: int, n: int) -> None:
    for layer, entry in enumerate(cache):
        entry.update_and_fetch(*_rows(seed * 100 + layer, n))
    mx.eval([mx.array(0)])
    for entry in cache:
        k, v = entry.state if not isinstance(entry, SegmentedKVCache) else (entry.keys, entry.values)
        mx.eval(k, v)


def _contents(cache):
    out = []
    for entry in cache:
        k, v = (entry.keys, entry.values) if isinstance(entry, SegmentedKVCache) else entry.state
        out.append((k[..., : entry.offset, :], v[..., : entry.offset, :]))
    return out


def _bank(monkeypatch) -> SessionBank:
    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MAX_SEGMENTS", "1000")
    return SessionBank(max_entries=64, max_bytes=1 << 40, per_session_max_bytes=1 << 40)


def _turn(bank, runtime, tokens: list[int], new: int, seed: int, session: str, *, prev=None):
    """One turn: restore the previous prompt, write ``new`` rows, bank the result."""
    if prev is None:
        cache = runtime.make_cache()
    else:
        restored = bank.restore(runtime, prev, session_id=session)
        assert restored is not None
        cache = restored.cache
        assert cache[0].offset == len(prev)
    _write(cache, seed, new)
    full = list(tokens)
    entry = bank.put(runtime=runtime, token_ids=full, cache=cache, logits=None, hidden=None, session_id=session)
    assert entry is not None
    return cache, entry


def _growth(kind: str, monkeypatch, turns: int = 6) -> tuple[int, int, int]:
    """Active-memory growth and peak growth over ``turns`` follow-ups, every snapshot kept alive
    (the entries are held here); returns (growth, peak growth, bytes of the new tokens)."""
    bank = _bank(monkeypatch)
    runtime = Runtime(kind)
    tokens = list(range(BASE))
    cache, entry = _turn(bank, runtime, tokens, BASE, 1, "s")
    held = [entry]
    del cache
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    prev = tokens
    for t in range(turns):
        tokens = prev + [BASE * 10 + t * TURN + i for i in range(TURN)]
        cache, entry = _turn(bank, runtime, tokens, TURN, 10 + t, "s", prev=prev)
        held.append(entry)
        prev = tokens
        del cache
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory() - before, mx.get_peak_memory() - before, turns * TURN * ROW_BYTES


def test_six_follow_up_turns_grow_memory_only_with_the_new_tokens(monkeypatch) -> None:
    growth, peak, new_bytes = _growth("segmented", monkeypatch)
    slack = 8 * 1024 * 1024
    assert growth <= new_bytes * 1.35 + slack, (growth, new_bytes)
    assert peak <= new_bytes * 1.35 + 2 * TURN * ROW_BYTES + slack, (peak, new_bytes)


def test_the_stock_cache_duplicates_the_history_in_the_same_flow(monkeypatch) -> None:
    """Control: a stock KVCache bank copies the history on a follow-up while a snapshot aliases it."""
    _g, peak, _new = _growth("stock", monkeypatch, turns=2)
    assert peak > 0.9 * BASE * ROW_BYTES  # at least one full copy of the history at the peak


def test_follow_up_content_equals_a_cache_that_never_went_through_the_bank(monkeypatch) -> None:
    bank = _bank(monkeypatch)
    runtime = Runtime("segmented")
    control = [KVCache() for _ in range(LAYERS)]
    tokens = list(range(BASE))
    _, _ = _turn(bank, runtime, tokens, BASE, 1, "s")
    _write(control, 1, BASE)
    prev = tokens
    for t in range(6):
        tokens = prev + [90_000 + t * TURN + i for i in range(TURN)]
        cache, _ = _turn(bank, runtime, tokens, TURN, 10 + t, "s", prev=prev)
        _write(control, 10 + t, TURN)
        prev = tokens
    for (ck, cv), (kk, kv) in zip(_contents(cache), _contents(control)):
        assert mx.array_equal(ck, kk).item() and mx.array_equal(cv, kv).item()
    assert cache[0].segment_count == 7


def test_the_bank_entry_is_a_list_of_references_and_counts_shared_segments_once(monkeypatch) -> None:
    bank = _bank(monkeypatch)
    runtime = Runtime("segmented")
    tokens = list(range(BASE))
    _, first = _turn(bank, runtime, tokens, BASE, 1, "s")
    assert entry_has_segmented_kv(first)
    assert all(isinstance(state, SegmentedKVState) for state in first.cache_snapshot.states)
    tokens2 = tokens + list(range(BASE, BASE + TURN))
    _, second = _turn(bank, runtime, tokens2, TURN, 2, "s", prev=tokens)
    naive = first.nbytes + second.nbytes
    assert bank.total_nbytes < naive  # the history segments are shared
    assert bank.total_nbytes >= (BASE + TURN) * ROW_BYTES


def _borrow(bank, runtime, entry, matched: int):
    borrowed = bank.restore_entry_prefix_cache(runtime, entry, matched)
    assert borrowed is not None
    return borrowed[0]


def test_fork_shares_the_history_and_branches_do_not_see_each_other(monkeypatch) -> None:
    bank = _bank(monkeypatch)
    runtime = Runtime("segmented")
    tokens = list(range(BASE))
    cache, root = _turn(bank, runtime, tokens, BASE, 1, "main")
    del cache
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    a = _borrow(bank, runtime, root, BASE)
    b = _borrow(bank, runtime, root, BASE)
    fork_at = a[0].offset
    assert fork_at == b[0].offset and fork_at >= BASE - 8  # a seed slot may be left to re-forward
    _write(a, 2, TURN)
    _write(b, 3, TURN)
    gc.collect()
    mx.clear_cache()
    # two new tails (512 rows plus the 256-row growth step), no second copy of the 4096-row history
    assert mx.get_active_memory() - before <= 2 * (TURN + 256) * ROW_BYTES + 4 * 1024 * 1024
    assert not mx.array_equal(a[0].keys[..., fork_at:, :], b[0].keys[..., fork_at:, :]).item()
    assert mx.array_equal(a[0].keys[..., :fork_at, :], b[0].keys[..., :fork_at, :]).item()
    assert a[0]._sealed[0].segment is b[0]._sealed[0].segment  # one history buffer
    reference = [KVCache() for _ in range(LAYERS)]
    _write(reference, 1, BASE)
    for (ak, _), (rk, _) in zip(_contents(a), _contents(reference)):
        assert mx.array_equal(ak[..., :fork_at, :], rk[..., :fork_at, :]).item()


def test_return_to_an_earlier_turn_restores_a_prefix_reference_without_a_copy(monkeypatch) -> None:
    bank = _bank(monkeypatch)
    runtime = Runtime("segmented")
    t0 = list(range(BASE))
    _turn(bank, runtime, t0, BASE, 1, "s")
    t1 = t0 + [50_000 + i for i in range(TURN)]
    cache, _e1 = _turn(bank, runtime, t1, TURN, 2, "s", prev=t0)
    t2 = t1 + [60_000 + i for i in range(TURN)]
    cache, last = _turn(bank, runtime, t2, TURN, 3, "s", prev=t1)
    del cache
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    # go back to the end of turn 1 (a client edited its last message)
    cache = _borrow(bank, runtime, last, len(t1))
    back_at = cache[0].offset
    assert len(t1) - 8 <= back_at <= len(t1)
    assert cache[0]._sealed[-1].n <= TURN  # a (segment, n') reference into turn 2's segment
    assert mx.get_active_memory() - before <= 8 * 1024 * 1024  # no copy of the history
    reference = [KVCache() for _ in range(LAYERS)]
    _write(reference, 1, BASE)
    _write(reference, 2, TURN)
    for (ck, cv), (rk, rv) in zip(_contents(cache), _contents(reference)):
        assert mx.array_equal(ck, rk[..., :back_at, :]).item()
        assert mx.array_equal(cv, rv[..., :back_at, :]).item()


def test_eviction_releases_the_segments(monkeypatch) -> None:
    bank = _bank(monkeypatch)
    runtime = Runtime("segmented")
    tokens = list(range(BASE))
    cache, entry = _turn(bank, runtime, tokens, BASE, 1, "s")
    del cache
    gc.collect()
    mx.clear_cache()
    held = mx.get_active_memory()
    assert bank.clear(session_id="s") == 1
    del entry
    gc.collect()
    mx.clear_cache()
    assert held - mx.get_active_memory() >= BASE * ROW_BYTES * 0.9


def test_ssd_tier_skips_segmented_entries_with_the_ssd_switch_off(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_SSD", "0")
    class Cold:
        puts = 0

        def put_entry(self, *a, **k):
            Cold.puts += 1

    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    bank = SessionBank(cold_tier=Cold())
    timing: dict = {}
    for kind in ("segmented", "stock"):
        runtime = Runtime(kind)
        cache = runtime.make_cache()
        _write(cache, 1, 600)
        bank.put(
            runtime=runtime, token_ids=list(range(600)), cache=cache, logits=None, hidden=None,
            session_id=kind, timing_out=timing,
        )
        if kind == "segmented":
            assert timing["cold_enqueue"]["skip_reason"] == "segmented_kv"
            assert Cold.puts == 0
    assert Cold.puts == 1  # the stock entry still goes to the tier


def test_admission_prices_no_history_copy_for_a_segmented_entry(monkeypatch) -> None:
    from mtplx.server.openai import _admission_restore_copies_prefix

    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")  # segmented entries only exist with the switch on
    bank = _bank(monkeypatch)
    _, seg_entry = _turn(bank, Runtime("segmented"), list(range(BASE)), BASE, 1, "x")
    stock_bank = _bank(monkeypatch)
    _, stock_entry = _turn(stock_bank, Runtime("stock"), list(range(BASE)), BASE, 1, "y")
    assert _admission_restore_copies_prefix(seg_entry, "clone", "x") is False
    assert _admission_restore_copies_prefix(seg_entry, "reference", "x") is False
    assert _admission_restore_copies_prefix(stock_entry, "clone", "y") is True


def test_health_block_is_absent_with_the_switch_off_and_reports_segments_with_it_on(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_SEGMENTED_KV", raising=False)
    bank = _bank(monkeypatch)
    runtime = Runtime("segmented")
    tokens = list(range(BASE))
    _turn(bank, runtime, tokens, BASE, 1, "s")
    tokens2 = tokens + list(range(BASE * 10, BASE * 10 + TURN))
    _turn(bank, runtime, tokens2, TURN, 2, "s", prev=tokens)
    assert "segmented_kv" not in bank.to_dict()
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    block = bank.to_dict()["segmented_kv"]
    assert block["enabled"] is True and block["entries"]
    assert block["max_segments"] >= 2
    assert block["sealed_bytes_unique"] > 0
    assert {"route_counts", "ssd_counts", "merges", "seals"} <= set(block)
    assert all(e["segments"] >= 1 and e["sealed_bytes"] > 0 for e in block["entries"])
    import json

    json.dumps(block)  # /health serialises it
