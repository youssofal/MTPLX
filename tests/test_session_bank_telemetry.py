"""CPU tests for SessionBank telemetry truthfulness.

Covers:
- Live-reference / one-copy leases reporting non-zero ``nbytes`` and
  ``held_nbytes`` (including array-attribute fallback when ``.nbytes`` is
  absent) without materializing paged ``.state``.
- Cold-tier restores computing non-zero ``nbytes`` when a cold record omits
  or zeros its byte metadata.
- RAM and SSD hit accounting carried cleanly across same-session lease
  handoffs, near-prefix direct increments, ``mark_prefix_hit``, and SSD
  restores without double counting.
- Routine same-session ``superseded_session_lease`` handoffs incrementing
  ``lease_handoffs`` instead of spamming ``eviction_log``, while real
  memory-pressure/capacity evictions and unconsumed distinct-cache
  abandonments remain logged.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank, SessionBankEntry, _live_cache_nbytes

GIB = 1 << 30
MIB = 1 << 20


class OneCopyKV:
    """Minimal one-copy live KV cache for CPU tests."""

    one_copy_store = True

    def __init__(self, nbytes: int, offset: int = 0):
        self._nbytes = int(nbytes)
        self.offset = int(offset)
        self.state_reads = 0

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def state(self):
        self.state_reads += 1
        raise RuntimeError("one-copy live KV must not materialize state")

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(n), self.offset)
        self.offset -= n
        return n


class ArrayOnlyOneCopyKV:
    """One-copy cache exposing MLX tensor attributes without a ``.nbytes`` property."""

    one_copy_store = True

    def __init__(self, offset: int = 16):
        self.offset = int(offset)
        self.keys = mx.zeros((1, 4, offset, 32), dtype=mx.float16)
        self.values = mx.zeros((1, 4, offset, 32), dtype=mx.float16)

    @property
    def state(self):
        raise RuntimeError("state must not be materialized to measure live bytes")

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(n), self.offset)
        self.offset -= n
        return n


class DurableKV:
    """Small snapshottable KV cache for durable and SSD-restore tests."""

    def __init__(self, offset: int = 8, *, one_copy_store: bool = False):
        self.one_copy_store = one_copy_store
        self.offset = int(offset)
        self._k = mx.zeros((1, 2, offset, 16), dtype=mx.float16)
        self._v = mx.zeros((1, 2, offset, 16), dtype=mx.float16)

    @property
    def state(self):
        return (self._k, self._v)

    @state.setter
    def state(self, value):
        if value is not None:
            self._k, self._v = value
            self.offset = int(self._k.shape[2])

    @property
    def meta_state(self):
        return (self.offset,)

    @meta_state.setter
    def meta_state(self, value):
        if value:
            self.offset = int(value[0])

    @property
    def nbytes(self) -> int:
        return int(self._k.nbytes + self._v.nbytes)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(n), self.offset)
        self.offset -= n
        if n > 0:
            self._k = self._k[:, :, : self.offset, :]
            self._v = self._v[:, :, : self.offset, :]
        return n


class _FakeRuntime:
    model_path = Path("models/inkling-test")
    mtp_enabled = True

    def __init__(self, *, one_copy: bool = False):
        self._one_copy = one_copy

    def make_cache(self):
        return [DurableKV(offset=0, one_copy_store=self._one_copy)]

    def make_mtp_cache(self):
        return [DurableKV(offset=0, one_copy_store=self._one_copy)]


def _make_bank(**kwargs) -> SessionBank:
    defaults = dict(max_entries=16, max_bytes=32 * GIB, per_session_max_bytes=8 * GIB)
    defaults.update(kwargs)
    return SessionBank(**defaults)


def test_one_copy_live_lease_reports_non_zero_nbytes_and_array_fallback():
    rt = _FakeRuntime(one_copy=True)
    bank = _make_bank()

    kv_explicit = OneCopyKV(256 * MIB, offset=32)
    entry = bank.put(
        runtime=rt,
        token_ids=list(range(1, 33)),
        cache=[kv_explicit],
        logits=mx.zeros((1, 16), dtype=mx.float16),
        hidden=None,
        keep_live_ref=True,
        session_id="sess-explicit",
    )
    assert entry is not None
    assert entry.live_ref_only is True
    assert kv_explicit.state_reads == 0
    assert entry.held_nbytes >= 256 * MIB

    snap = bank.to_dict()
    assert snap["total_nbytes"] >= 256 * MIB
    assert snap["held_nbytes"] == snap["total_nbytes"]
    assert snap["lease_nbytes"] == snap["total_nbytes"]
    prefix = snap["prefixes"][0]
    assert prefix["nbytes"] == entry.held_nbytes
    assert prefix["held_nbytes"] == entry.held_nbytes
    assert prefix["snapshot_nbytes"] == 0

    # Array-attribute fallback when a custom cache omits `.nbytes` (and must
    # never touch `.state`).
    kv_array_only = ArrayOnlyOneCopyKV(offset=24)
    expected_kv_bytes = int(kv_array_only.keys.nbytes + kv_array_only.values.nbytes)
    assert _live_cache_nbytes([kv_array_only]) == expected_kv_bytes
    entry2 = bank.put(
        runtime=rt,
        token_ids=list(range(101, 125)),
        cache=[kv_array_only],
        logits=None,
        hidden=None,
        keep_live_ref=True,
        session_id="sess-array-only",
    )
    assert entry2 is not None
    assert entry2.live_ref_only is True
    assert entry2.held_nbytes == expected_kv_bytes
    snap2 = bank.to_dict()
    by_sid = {row["session_id"]: row for row in snap2["prefixes"]}
    assert by_sid["sess-array-only"]["nbytes"] == expected_kv_bytes
    assert by_sid["sess-array-only"]["held_nbytes"] == expected_kv_bytes


def test_warm_multi_turn_one_copy_session_tracks_hits_and_avoids_eviction_spam():
    """Simulate a 4-turn warm coding session on a one-copy live KV cache:
    prompt lease -> final lease -> restore -> next prompt lease -> final lease.

    Verifies:
    - Routine lease handoffs increment ``lease_handoffs`` and leave
      ``eviction_log`` / ``recent_evictions`` empty.
    - RAM restore hits accumulate on both the active prefix entry and the
      bank-wide snapshot (``hits``, ``cumulative_hits``, ``ram_hits``).
    - Direct ``entry.hits += 1`` (near-prefix callers) and ``mark_prefix_hit``
      are folded once with zero double-counting.
    """

    rt = _FakeRuntime(one_copy=True)
    bank = _make_bank()
    cache = [OneCopyKV(512 * MIB, offset=10)]

    # Turn 1: prompt lease (10 tok) -> generation-final lease (15 tok) on same cache.
    p1 = bank.put(
        runtime=rt,
        token_ids=list(range(1, 11)),
        cache=cache,
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        keep_live_ref=True,
        session_id="opencode-1",
    )
    assert p1 is not None
    cache[0].offset = 15
    g1 = bank.put(
        runtime=rt,
        token_ids=list(range(1, 16)),
        cache=cache,
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        keep_live_ref=True,
        session_id="opencode-1",
    )
    assert g1 is not None
    assert len(bank) == 1
    assert list(bank.eviction_log) == []
    assert bank.lease_handoffs == 1

    # Turn 2: restore consumes G1 (hit #1), prefills to 22, generates to 28.
    r2 = bank.restore(rt, list(range(1, 23)), mode="reference", session_id="opencode-1")
    assert r2 is not None and r2.cache is cache
    assert r2.restored_nbytes >= 512 * MIB
    cache[0].offset = 22
    bank.put(
        runtime=rt,
        token_ids=list(range(1, 23)),
        cache=cache,
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        keep_live_ref=True,
        session_id="opencode-1",
    )
    cache[0].offset = 28
    g2 = bank.put(
        runtime=rt,
        token_ids=list(range(1, 29)),
        cache=cache,
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        keep_live_ref=True,
        session_id="opencode-1",
    )
    assert g2 is not None
    assert g2.hits == 1

    # Turn 3: restore consumes G2 (hit #2), advances to 36.
    r3 = bank.restore(rt, list(range(1, 35)), mode="reference", session_id="opencode-1")
    assert r3 is not None
    cache[0].offset = 36
    g3 = bank.put(
        runtime=rt,
        token_ids=list(range(1, 37)),
        cache=cache,
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        keep_live_ref=True,
        session_id="opencode-1",
    )
    assert g3 is not None
    assert g3.hits == 2

    # Near-prefix direct increment (as generation.py does) + repeated to_dict()
    # reads must count the hit exactly once.
    g3.hits += 1
    snap_a = bank.to_dict()
    snap_b = bank.to_dict()
    assert snap_a["hits"] == 3
    assert snap_a["cumulative_hits"] == 3
    assert snap_a["ram_hits"] == 3
    assert snap_a["ssd_hits"] == 0
    assert snap_b["hits"] == 3
    assert snap_a["prefixes"][0]["hits"] == 3
    assert snap_a["prefixes"][0]["cumulative_hits"] == 3
    assert snap_a["prefixes"][0]["nbytes"] >= 512 * MIB
    assert snap_a["eviction_log"] == []
    assert snap_a["recent_evictions"] == []
    assert snap_a["lease_handoffs"] >= 4

    # Public helper mark_prefix_hit increments once and carries across next put.
    bank.mark_prefix_hit(g3, source="ram")
    cache[0].offset = 44
    g4 = bank.put(
        runtime=rt,
        token_ids=list(range(1, 45)),
        cache=cache,
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        keep_live_ref=True,
        session_id="opencode-1",
    )
    assert g4 is not None
    assert g4.hits == 4
    snap_final = bank.to_dict()
    assert snap_final["hits"] == 4
    assert snap_final["ram_hits"] == 4
    assert snap_final["eviction_log"] == []


def test_real_evictions_and_distinct_cache_abandonment_still_log():
    rt = _FakeRuntime(one_copy=True)
    bank = _make_bank(max_bytes=2 * GIB)

    # Diverged unconsumed lease on a distinct cache still logs as an eviction
    # and releases the old cache.
    cache_a = [OneCopyKV(600 * MIB, offset=10)]
    ref_a = weakref.ref(cache_a[0])
    bank.put(
        runtime=rt,
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        cache=cache_a,
        logits=None,
        hidden=None,
        keep_live_ref=True,
        session_id="diverged",
    )
    del cache_a
    cache_b = [OneCopyKV(600 * MIB, offset=10)]
    bank.put(
        runtime=rt,
        token_ids=[1, 2, 3, 99, 98, 97, 96, 95, 94, 93],
        cache=cache_b,
        logits=None,
        hidden=None,
        keep_live_ref=True,
        session_id="diverged",
    )
    gc.collect()
    assert ref_a() is None
    assert bank.eviction_log[-1]["reason"] == "superseded_session_lease"

    # Real memory-pressure eviction logs normally.
    assert bank.shrink_to_bytes(0, reason="memory_pressure") == 1
    assert bank.eviction_log[-1]["reason"] == "memory_pressure"
    snap = bank.to_dict()
    assert any(rec["reason"] == "memory_pressure" for rec in snap["eviction_log"])
    assert any(rec["reason"] == "memory_pressure" for rec in snap["recent_evictions"])


def test_ssd_exact_and_near_prefix_restores_record_hits_and_non_zero_nbytes():
    class FakeColdTier:
        def __init__(self, record):
            self._record = record

        def stats(self):
            return {"enabled": True, "restore_hits": 1}

        def lookup(self, token_ids, **_kwargs):
            return self._record

    tokens = tuple(range(1, 17))
    kv_state = (
        mx.zeros((1, 2, 15, 16), dtype=mx.float16),
        mx.zeros((1, 2, 15, 16), dtype=mx.float16),
    )
    expected_kv_bytes = int(kv_state[0].nbytes + kv_state[1].nbytes)
    cold_snapshot = CacheSnapshot(states=(kv_state,), meta_states=((15,),))
    cold_record = SimpleNamespace(
        token_ids=tokens,
        cache_snapshot=cold_snapshot,
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        mtp_history_snapshot=None,
        has_recurrent=False,
        gdn_boundaries=[],
        gdn_boundary_loader=None,
        nbytes=0,  # Simulate cold record with omitted/zero byte metadata.
        restore_s=0.05,
        metadata={
            "model_path": str(_FakeRuntime.model_path),
            "mtp_enabled": True,
            "session_id": "ssd-sess",
            "snapshot_epoch": len(tokens),
        },
    )

    # 1) One-copy SSD exact restore records ssd_hits and carries into next put.
    rt_oc = _FakeRuntime(one_copy=True)
    bank_oc = _make_bank(cold_tier=FakeColdTier(cold_record))
    restored = bank_oc.restore(rt_oc, list(tokens) + [99], session_id="ssd-sess")
    assert restored is not None
    assert restored.ssd_cache_hit is True
    assert restored.restored_nbytes >= expected_kv_bytes
    assert bank_oc.ssd_hits == 1
    assert bank_oc.ram_hits == 0
    assert bank_oc.total_hits == 1

    # Next turn commits a one-copy lease; entry inherits the session's SSD hit
    # without double-counting the bank total.
    live = [OneCopyKV(128 * MIB, offset=17)]
    next_entry = bank_oc.put(
        runtime=rt_oc,
        token_ids=list(tokens) + [99],
        cache=live,
        logits=None,
        hidden=None,
        keep_live_ref=True,
        session_id="ssd-sess",
    )
    assert next_entry is not None
    assert next_entry.hits == 1
    snap_oc = bank_oc.to_dict()
    assert snap_oc["hits"] == 1
    assert snap_oc["cumulative_hits"] == 1
    assert snap_oc["ssd_hits"] == 1
    assert snap_oc["ram_hits"] == 0
    assert snap_oc["prefixes"][0]["hits"] == 1
    assert snap_oc["prefixes"][0]["nbytes"] >= 128 * MIB

    # 2) Durable SSD exact restore after a prior RAM hit + eviction preserves
    # cumulative per-session hits (1 RAM + 1 SSD = 2) on the re-resident entry.
    rt_dur = _FakeRuntime(one_copy=False)
    bank_dur = _make_bank(cold_tier=FakeColdTier(cold_record))
    bank_dur.put(
        runtime=rt_dur,
        token_ids=list(tokens),
        cache=[DurableKV(offset=15)],
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        session_id="ssd-sess",
    )
    assert bank_dur.restore(rt_dur, list(tokens), session_id="ssd-sess") is not None
    assert bank_dur.ram_hits == 1
    assert bank_dur.shrink_to_bytes(0, reason="memory_pressure") == 1
    restored_dur = bank_dur.restore(rt_dur, list(tokens), session_id="ssd-sess")
    assert restored_dur is not None
    assert restored_dur.ssd_cache_hit is True
    assert restored_dur.entry.nbytes >= expected_kv_bytes
    assert restored_dur.entry.hits == 2
    snap_dur = bank_dur.to_dict()
    assert snap_dur["ram_hits"] == 1
    assert snap_dur["ssd_hits"] == 1
    assert snap_dur["hits"] == 2
    assert snap_dur["prefixes"][0]["hits"] == 2

    # 3) Ephemeral SSD near-prefix restore via restore_entry_prefix_cache
    # increments ssd_hits cleanly.
    bank_near = _make_bank()
    ephemeral = SessionBankEntry(
        token_ids=tokens,
        token_hash="ephem-ssd",
        model_path=str(rt_dur.model_path),
        mtp_enabled=True,
        hidden_variant=None,
        cache_snapshot=cold_snapshot,
        logits=None,
        hidden=None,
        nbytes=expected_kv_bytes,
        session_id="ssd-near",
    )
    setattr(ephemeral, "cache_source", "ssd")
    setattr(ephemeral, "ssd_cache_hit", True)
    setattr(ephemeral, "decoded_for_request", True)
    res = bank_near.restore_entry_prefix_cache(
        rt_dur, ephemeral, prefix_len=12, mode="clone", session_id="ssd-near"
    )
    assert res is not None
    assert bank_near.ssd_hits == 1
    assert bank_near.total_hits == 1
