"""Live-reference leases must be visible to every budget and releasable.

Issue #456 (and the memory half of #499, and the kernel panics cited by
PR #500). When a snapshot is over the per-session byte budget, ``put()`` keeps
a "live-reference lease" instead: the entry pins the turn's whole live cache
and used to record ``nbytes=0``. Every guard in the bank is byte based, so a
lease was invisible to all of them:

* ``total_nbytes`` read 0, so the server's admission shed (gated on
  ``bank_bytes_now > 0``) skipped shedding and reported "nothing sheddable";
* ``shrink_to_bytes`` loops on ``total_nbytes > target`` and never ran;
* the dynamic ceiling computes ``working = active - weights - bank bytes``, so
  the lease inflated the working set and the guard evicted the useful durable
  snapshots while the actual holder stayed;
* leases were exempt from supersede in both directions and from per-session
  retention, so a lease the next turn could not consume stayed for good while
  the turn allocated a fresh cache. One leaked cache per turn.

Measured by peterloron on a 64 GB M4 Max (27B, q8 paged KV): +3.7 GiB of
``active`` per turn with ``session_bank.entries: 4, total_nbytes: 0``.

These tests drive the real put/restore contract; nothing fabricates entries.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import mlx.core as mx

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank

GIB = 1 << 30


class LiveKV:
    """A live attention cache: reports its allocation, refuses to materialize.

    Every real cache class exposes ``nbytes`` (mlx-lm's ``KVCache`` and MTPLX's
    paged and tensor-offset caches alike), and the paged ones raise from
    ``state`` exactly like this, which is one of the ways a lease is created.
    """

    def __init__(self, nbytes: int, offset: int = 0):
        self._nbytes = int(nbytes)
        self.offset = int(offset)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def state(self):
        raise RuntimeError("Paged KV cache attempted to materialize active K/V arrays")

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        self.offset -= int(n)
        return int(n)


class Runtime:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return [LiveKV(0)]

    def make_mtp_cache(self):
        return [LiveKV(0)]


RUNTIME = Runtime()


def _bank(**kwargs) -> SessionBank:
    defaults = dict(max_entries=16, max_bytes=64 * GIB, per_session_max_bytes=2 * GIB)
    defaults.update(kwargs)
    return SessionBank(**defaults)


def _lease(bank: SessionBank, tokens, cache, *, session_id="s1", oversized=7 * GIB, **kwargs):
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=cache,
        logits=None,
        hidden=None,
        keep_live_ref=True,
        session_id=session_id,
        nbytes_override=int(oversized),
        **kwargs,
    )
    assert entry is not None and entry.live_ref_only is True
    return entry


def _released(ref: "weakref.ReferenceType") -> bool:
    gc.collect()
    return ref() is None


# --------------------------------------------------------------------------
# Visibility: a lease is memory the bank holds
# --------------------------------------------------------------------------


def test_a_lease_counts_toward_the_bank_total_and_shows_in_health():
    bank = _bank()
    _lease(bank, range(10), [LiveKV(6 * GIB, offset=10)])

    # Before the fix this read 0 with one entry: the #456 discriminator
    # ("entries non-zero while total_nbytes is 0").
    assert bank.total_nbytes >= 6 * GIB
    health = bank.to_dict()
    assert health["lease_entries"] == 1
    assert health["lease_nbytes"] == bank.total_nbytes
    assert health["held_nbytes"] == bank.total_nbytes
    assert health["prefixes"][0]["held_nbytes"] == bank.total_nbytes
    # Live-reference leases report their true held RAM footprint on ``nbytes``
    # so dashboard/app surfaces never render them as Zero KB, while keeping the
    # snapshot-only byte count visible under ``snapshot_nbytes``.
    assert health["prefixes"][0]["nbytes"] == bank.total_nbytes
    assert health["prefixes"][0]["snapshot_nbytes"] == 0


def test_the_refused_snapshot_size_is_the_floor_when_the_cache_cannot_report_bytes():
    class Opaque:
        state = None

        def is_trimmable(self) -> bool:
            return True

    bank = _bank()
    _lease(bank, range(10), [Opaque()], oversized=5 * GIB)

    assert bank.total_nbytes == 5 * GIB


def test_a_consumed_lease_stops_charging_the_cache_it_handed_over():
    bank = _bank()
    cache = [LiveKV(6 * GIB, offset=11)]
    _lease(bank, range(10), cache)

    restored = bank.restore(RUNTIME, list(range(10)), mode="reference", session_id="s1")

    assert restored is not None and restored.cache is cache
    # The running turn owns the cache now; charging it to the bank as well
    # would double count it against the working set.
    assert bank.total_nbytes == 0


# --------------------------------------------------------------------------
# The leak: a lease the next turn did not consume
# --------------------------------------------------------------------------


def test_a_newer_commit_of_the_same_session_releases_the_lease_it_left_behind():
    """Turn 1 leases cache A. Turn 2 could not use it (the client's prompt
    diverged), prefilled cold into cache B and leases B. A must go."""

    bank = _bank()
    cache_a = [LiveKV(6 * GIB, offset=10)]
    ref_a = weakref.ref(cache_a[0])
    _lease(bank, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cache_a)

    cache_b = [LiveKV(6 * GIB, offset=12)]
    _lease(bank, [1, 2, 3, 4, 5, 60, 70, 80, 90, 100, 110, 120], cache_b)
    del cache_a

    assert len(bank) == 1
    assert bank.total_nbytes < 8 * GIB
    assert _released(ref_a), "the abandoned turn's cache is still pinned by its lease"
    assert bank.eviction_log[-1]["reason"] == "superseded_session_lease"
    assert bank.eviction_log[-1]["live_ref_only"] is True


def test_five_cold_turns_hold_one_cache_not_five():
    """peterloron's five-turn table, in miniature."""

    bank = _bank()
    refs = []
    for turn in range(5):
        cache = [LiveKV(4 * GIB, offset=20 + turn)]
        refs.append(weakref.ref(cache[0]))
        # Each turn's transcript differs from the last one early on, so no
        # lease is ever a prefix of the next prompt.
        _lease(bank, [1, 2, 100 + turn, *range(3, 20 + turn)], cache, oversized=4 * GIB)
        del cache

    assert len(bank) == 1
    # One turn's cache, not five: before the fix this was 5 entries holding
    # 20 GiB and reporting 0.
    assert bank.total_nbytes == 4 * GIB
    assert [_released(ref) for ref in refs] == [True, True, True, True, False]


def test_a_durable_commit_also_releases_the_sessions_stale_lease():
    bank = _bank()
    cache_a = [LiveKV(6 * GIB, offset=10)]
    ref_a = weakref.ref(cache_a[0])
    _lease(bank, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cache_a)
    del cache_a

    durable = bank.put(
        runtime=RUNTIME,
        token_ids=[1, 2, 3, 40, 50],
        cache=[],
        logits=None,
        hidden=None,
        session_id="s1",
        nbytes_override=64,
    )

    assert durable is not None and durable.live_ref_only is False
    assert [entry.live_ref_only for entry in bank._entries.values()] == [False]
    assert _released(ref_a)


def test_another_sessions_lease_is_left_alone():
    bank = _bank()
    _lease(bank, [1, 2, 3, 4], [LiveKV(GIB, offset=4)], session_id="a")
    _lease(bank, [9, 8, 7, 6], [LiveKV(GIB, offset=4)], session_id="b")

    assert len(bank) == 2


def test_a_consumed_lease_still_hands_its_boundary_records_to_the_next_entry():
    """The fence. put() inherits recurrent boundary records from the longest
    stored prefix (#121), and between turns that prefix IS the consumed lease.
    Releasing the lease at consumption, before the next put, would strip the
    records and push the next divergent turn onto a cold prefill."""

    bank = _bank()
    boundary = (4, CacheSnapshot(states=(mx.zeros((64,), mx.uint8),), meta_states=(None,)), None)
    cache = [LiveKV(6 * GIB, offset=11)]
    _lease(bank, range(10), cache, gdn_boundaries=[boundary])

    restored = bank.restore(RUNTIME, list(range(10)), mode="reference", session_id="s1")
    assert restored is not None
    assert len(bank) == 1, "the consumed lease must survive until the next commit"

    cache[0].offset = 14
    newer = _lease(bank, range(14), cache)

    assert [int(record[0]) for record in newer.gdn_boundaries] == [4]
    assert len(bank) == 1


# --------------------------------------------------------------------------
# Pressure: every shedding path can reach a lease
# --------------------------------------------------------------------------


def test_memory_pressure_can_release_a_lease():
    bank = _bank()
    cache = [LiveKV(6 * GIB, offset=10)]
    ref = weakref.ref(cache[0])
    _lease(bank, range(10), cache)
    del cache

    # Before the fix: 0 evicted, because the loop condition read 0 bytes.
    assert bank.shrink_to_bytes(0, reason="memory_pressure") == 1
    assert bank.total_nbytes == 0
    assert _released(ref)


def test_the_bank_budget_evicts_an_older_lineages_lease():
    """An agent compaction mints a new session id, so the per-session rule
    cannot see the old lineage's lease. The byte budget must."""

    bank = _bank(max_bytes=8 * GIB)
    cache_old = [LiveKV(6 * GIB, offset=10)]
    ref_old = weakref.ref(cache_old[0])
    _lease(bank, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cache_old, session_id="anon-before")
    del cache_old

    _lease(bank, [7, 7, 7, 7], [LiveKV(6 * GIB, offset=4)], session_id="anon-after")

    assert [entry.session_id for entry in bank._entries.values()] == ["anon-after"]
    assert _released(ref_old)


def test_the_lease_being_stored_is_never_its_own_victim():
    bank = _bank(max_bytes=4 * GIB)
    _lease(bank, range(10), [LiveKV(6 * GIB, offset=10)])

    assert len(bank) == 1


def test_admission_shed_releases_a_stale_lease_but_not_the_one_the_prompt_restores_from():
    bank = _bank()
    mine = [LiveKV(5 * GIB, offset=8)]
    theirs = [LiveKV(5 * GIB, offset=8)]
    ref_theirs = weakref.ref(theirs[0])
    _lease(bank, [1, 2, 3, 4, 5, 6, 7, 8], mine, session_id="mine")
    _lease(bank, [9, 9, 9, 9, 9, 9, 9, 9], theirs, session_id="theirs")
    del theirs

    non_terminal, terminal = bank.shrink_for_admission(
        0, protect_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    assert (non_terminal, terminal) == (0, 1)
    assert [entry.session_id for entry in bank._entries.values()] == ["mine"]
    assert _released(ref_theirs)


def test_an_evicted_durable_entry_lets_go_of_its_live_reference_too():
    class SnapshottableKV:
        """A cache whose snapshot succeeds, so the put is a durable entry
        that ALSO keeps a live reference (keep_live_ref=True)."""

        state = None
        meta_state = None

        def is_trimmable(self) -> bool:
            return True

    bank = _bank()
    cache = [SnapshottableKV()]
    ref = weakref.ref(cache[0])
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=[1, 2, 3],
        cache=cache,
        logits=None,
        hidden=None,
        keep_live_ref=True,
        session_id="s1",
        nbytes_override=64,
    )
    assert entry is not None and entry.live_ref_only is False
    assert entry.cache_ref is cache
    del cache

    assert bank.shrink_to_bytes(0) == 1
    # Whoever still holds the entry object (a finished request's outcome
    # does) must not keep the cache alive through it.
    assert entry.cache_ref is None
    assert _released(ref)
