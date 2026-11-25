"""Releasing idle conversations before a prompt is refused.

The 2026-09-26 field report (M5 Max 128 GB, Flash-Next, the pi coding agent):
pi's compaction request arrives as a new anonymous session while the
114k-token conversation it summarizes waits for the answer. The admission
shed could not reach that conversation: the superseded clear targets the
incoming session id, the LRU pass protects sessions touched in the last 600 s,
and the chain walk never evicts a snapshot that also holds a live cache. 13
refusals in a row, and only a restart cleared them.

These tests pin the release step on the real SessionBank and
EngineSessionManager (synthetic entries, no model): which sessions and
entries it takes, in what order, what it never touches, how it owns a session
while it evicts, and what happens to the SSD copies. The 2026-09-27 review of
the first cut reproduced six defects; each has a test here.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.engine_session import EngineSessionManager
from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.session_bank import SessionBank, cold_persistence_key

RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


class _Tier:
    """Cold tier double: records encodes, publishes only what a test says."""

    # The SSD cache in its normal mode: it reads back what it published.
    restorable = True

    def __init__(self) -> None:
        self.encoded: list[tuple[int, ...]] = []
        self.published: set[tuple[int, ...]] = set()

    def put_entry(self, entry, capabilities=(), raise_on_yield=False):
        self.encoded.append(entry.token_ids)
        return True

    def is_published(self, entry) -> bool:
        return entry.token_ids in self.published


class _Lane:
    """The scheduler's idle persistence lane: newest job per key, cancel by key."""

    def __init__(self) -> None:
        self.pending: dict[str, object] = {}

    def dispatch(self, job) -> None:
        self.pending[job.coalesce_key] = job

    def cancel(self, key: str) -> int:
        return 1 if self.pending.pop(key, None) is not None else 0

    def run_all(self) -> None:
        jobs = list(self.pending.values())
        self.pending.clear()
        for job in jobs:
            job()


def _bank(tier=None, lane=None, **kwargs) -> SessionBank:
    defaults = dict(max_entries=32, max_bytes=10_000, per_session_max_bytes=10_000)
    defaults.update(kwargs)
    bank = SessionBank(cold_tier=tier, **defaults)
    if lane is not None:
        bank.cold_enqueue_dispatch = lane.dispatch
        bank.cold_enqueue_cancel = lane.cancel
    return bank


def _put(bank: SessionBank, tokens, *, session_id, nbytes, live_cache=False):
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=[],
        logits=None,
        hidden=None,
        session_id=session_id,
        nbytes_override=nbytes,
    )
    assert entry is not None
    if live_cache:
        # A generation-final commit of a coding-agent turn keeps the live
        # cache next to its snapshot (keep_live_ref).
        entry.cache_ref = object()
    return entry


def _keys(bank: SessionBank):
    return set(bank._entries)


def _conversation(bank: SessionBank, session_id="anon-conv"):
    """Julian's shape: generation-final and postcommit siblings that diverge
    at the assistant turn, each holding a live cache, plus an older sibling
    from an earlier turn (retained: not a strict prefix of either)."""

    older = _put(bank, (1, 2, 9), session_id=session_id, nbytes=300)
    final = _put(bank, (1, 2, 3, 4, 5, 60), session_id=session_id, nbytes=400, live_cache=True)
    post = _put(bank, (1, 2, 3, 4, 5, 70), session_id=session_id, nbytes=400, live_cache=True)
    return older, final, post


class TestBankRelease:
    def test_an_idle_conversation_is_released_whole(self):
        bank = _bank()
        _conversation(bank)
        receipt = bank.release_sessions(
            None, keep_session_ids={"anon-compaction"}, protect_tokens=(9, 9, 9)
        )
        assert _keys(bank) == set()
        assert receipt["entries"] == 3
        assert receipt["held_bytes"] == 1_100
        [row] = receipt["sessions"]
        assert row["session_id"] == "anon-conv"
        assert row["live_cache_refs"] == 2
        assert row["dropped_entries"] == 3
        assert all(
            record["reason"] == "idle_session_release"
            for record in list(bank.eviction_log)[-3:]
        )

    def test_the_chain_walk_alone_never_reaches_it(self):
        """The defect this step exists for: nothing in the old escalation
        takes a snapshot that also holds a live cache, so the walk stops
        with every byte of the conversation still resident."""

        bank = _bank()
        _conversation(bank)
        bank.shrink_for_admission(0, protect_tokens=(9, 9, 9))
        assert {(1, 2, 3, 4, 5, 60), (1, 2, 3, 4, 5, 70)} <= _keys(bank)

    def test_kept_sessions_are_never_touched(self):
        bank = _bank()
        _conversation(bank)
        receipt = bank.release_sessions(None, keep_session_ids={"anon-conv"})
        assert receipt["entries"] == 0
        assert len(bank._entries) == 3

    def test_the_restore_source_survives_and_its_siblings_go(self):
        bank = _bank()
        _older, final, _post = _conversation(bank)
        # The prompt extends the generation-final entry exactly.
        prompt = final.token_ids + (80, 81)
        receipt = bank.release_sessions(None, protect_tokens=prompt)
        assert _keys(bank) == {final.token_ids}
        assert receipt["protected_restore_source_tokens"] == len(final.token_ids)
        assert receipt["sessions"][0]["kept_restore_sources"] == 1

    def test_whole_session_release_does_not_stop_at_the_byte_target(self):
        """Review defect 4: the byte target was checked per entry, so a
        session could keep a small fallback while the entry behind its
        committed tokens went, and keep advertising the whole prefix."""

        bank = _bank()
        prompt = tuple(range(1_000))
        terminal = _put(bank, prompt, session_id="conv", nbytes=1_000)
        _put(bank, prompt[:3] + (7,), session_id="conv", nbytes=10)
        other = _put(bank, (5, 5, 5), session_id="other", nbytes=500)
        terminal.last_access_s = 1.0
        other.last_access_s = 2.0
        for entry in bank._entries.values():
            if entry.session_id == "conv":
                entry.last_access_s = 1.0
        assert bank.session_coverage_tokens("conv", prompt + (1,)) == 1_000
        receipt = bank.release_sessions(1_000)
        # The target was met by the terminal alone; the fallback went too,
        # and the next session in order was not needed.
        assert _keys(bank) == {other.token_ids}
        assert receipt["entries"] == 2
        assert bank.session_coverage_tokens("conv", prompt + (1,)) == 0

    def test_a_released_session_loses_its_activity_pin(self):
        bank = _bank()
        _conversation(bank)
        assert "anon-conv" in bank._active_session_ids()
        bank.release_sessions(None)
        assert "anon-conv" not in bank._active_session_ids()

    def test_least_recently_used_session_goes_first(self):
        bank = _bank()
        stale = _put(bank, (1, 1, 1), session_id="stale", nbytes=100)
        recent = _put(bank, (2, 2, 2), session_id="recent", nbytes=100)
        stale.last_access_s = 10.0
        recent.last_access_s = 20.0
        bank.release_sessions(100)
        assert _keys(bank) == {recent.token_ids}


class TestRestoreSource:
    """Review defect 3: protection must follow the restore's own lanes and
    gates, not the longest token overlap."""

    def test_a_shared_first_token_is_not_a_restore_source(self):
        bank = _bank()
        _conversation(bank)
        assert bank.restore_source_key((1, 99, 98, 97)) is None
        receipt = bank.release_sessions(None, protect_tokens=(1, 99, 98, 97))
        assert receipt["protected_restore_source_tokens"] is None
        assert _keys(bank) == set()

    def test_a_block_prefix_is_a_restore_source(self):
        bank = _bank(max_bytes=10**9, per_session_max_bytes=10**9)
        shared = tuple(range(1_000))
        entry = _put(bank, shared + (5, 5), session_id="conv", nbytes=100)
        # 1,000 shared tokens, two stored tokens after them: the tiny-gap
        # lane restores at 1,000.
        assert bank.restore_source_key(shared + (7, 7)) == entry.token_ids
        assert bank.restore_plan(shared + (7, 7))["reuse_tokens"] == 1_000
        # 300 shared tokens with 702 stored after them: the block lane needs
        # 512, so this entry serves nothing.
        assert bank.restore_source_key(shared[:300] + (7, 7)) is None

    def test_a_tiny_gap_overlap_is_a_restore_source(self):
        """The first cut rounded to 256-token blocks and missed this shape,
        which the near-prefix lane serves: 100 shared tokens, one stored
        token after them."""

        bank = _bank()
        shared = tuple(range(100))
        entry = _put(bank, shared + (1_000,), session_id="conv", nbytes=100)
        prompt = shared + tuple(range(2_000, 2_050))
        assert bank.restore_source_keys(prompt) == {entry.token_ids}

    def test_a_longer_recurrent_entry_without_a_boundary_does_not_displace_the_exact_prefix(
        self,
    ):
        """The review's reproduction: a valid 256-token exact prefix and a
        longer recurrent entry sharing 1,000 tokens with no usable recurrent
        boundary. The old rule protected the longer entry (the restore
        rejects it) and the release evicted the exact prefix."""

        bank = _bank(max_bytes=10**9, per_session_max_bytes=10**9)
        prompt = tuple(range(1_500))
        longer = _put(
            bank, prompt[:1_000] + tuple(range(9_000, 9_020)), session_id="conv", nbytes=900
        )
        # A hybrid entry without stored boundaries (put with a recurrent
        # cache): it cannot serve a sub-prefix, so it supersedes nothing.
        longer.has_recurrent = True
        longer.gdn_boundaries = []
        exact = _put(bank, prompt[:256], session_id="conv", nbytes=100)
        assert _keys(bank) == {longer.token_ids, exact.token_ids}
        assert bank.restore_source_keys(prompt) == {exact.token_ids}
        bank.release_sessions(None, protect_tokens=prompt)
        assert _keys(bank) == {exact.token_ids}

    def test_a_recurrent_entry_with_a_boundary_past_the_exact_prefix_is_kept(self):
        bank = _bank(max_bytes=10**9, per_session_max_bytes=10**9)
        prompt = tuple(range(1_500))
        longer = _put(
            bank, prompt[:1_000] + tuple(range(9_000, 9_020)), session_id="conv", nbytes=900
        )
        longer.has_recurrent = True
        longer.gdn_boundaries = [(768, None, None)]
        # Another session's exact prefix (a session's own contained prefix
        # would have been superseded by the entry with boundaries).
        exact = _put(bank, prompt[:256], session_id="other", nbytes=100)
        plan = bank.restore_plan(prompt)
        # The near lane beats the exact prefix: the restore lands at 768.
        assert plan["keys"] == {longer.token_ids, exact.token_ids}
        assert plan["reuse_tokens"] == 768
        assert plan["source"] is longer

    def test_an_identity_mismatch_is_not_a_source(self):
        bank = _bank()
        entry = _put(bank, tuple(range(10)), session_id="conv", nbytes=100)
        entry.template_hash = "other-template"
        plan = bank.restore_plan(tuple(range(20)), template_hash="this-template")
        assert plan["keys"] == set()

    def test_a_consumed_lease_is_not_a_source(self):
        bank = _bank()
        entry = _put(bank, tuple(range(10)), session_id="conv", nbytes=100)
        entry.live_ref_only = True
        entry.cache_ref = None
        assert bank.restore_plan(tuple(range(20)))["keys"] == set()

    def test_an_expired_entry_is_not_a_source(self):
        """A conversation idle past ``idle_ttl_s`` that shares a block
        prefix with the prompt: the restore purges it before matching, so
        the plan must not protect it."""

        bank = _bank(max_bytes=10**9, per_session_max_bytes=10**9, idle_ttl_s=3_600.0)
        shared = tuple(range(1_000))
        entry = _put(bank, shared + (5, 5), session_id="old", nbytes=900, live_cache=True)
        prompt = shared + (7, 7)
        assert bank.restore_source_keys(prompt) == {entry.token_ids}
        entry.last_access_s -= 3_601.0
        plan = bank.restore_plan(prompt)
        assert plan["keys"] == set()
        assert plan["reuse_tokens"] == 0
        assert plan["source"] is None
        # Planning is read-only; the release takes it like any idle entry.
        assert _keys(bank) == {entry.token_ids}
        receipt = bank.release_sessions(None, keep_session_ids={"new"}, protect_tokens=prompt)
        assert receipt["protected_restore_source_tokens"] is None
        assert _keys(bank) == set()

    def test_an_expired_near_source_leaves_the_live_exact_prefix(self):
        """The shape of the kept-recurrent test above, with the longer entry
        expired: the plan falls back to the live exact prefix alone."""

        bank = _bank(max_bytes=10**9, per_session_max_bytes=10**9, idle_ttl_s=3_600.0)
        prompt = tuple(range(1_500))
        longer = _put(
            bank, prompt[:1_000] + tuple(range(9_000, 9_020)), session_id="conv", nbytes=900
        )
        longer.has_recurrent = True
        longer.gdn_boundaries = [(768, None, None)]
        exact = _put(bank, prompt[:256], session_id="other", nbytes=100)
        assert bank.restore_plan(prompt)["source"] is longer
        longer.last_access_s -= 3_601.0
        plan = bank.restore_plan(prompt)
        assert plan["keys"] == {exact.token_ids}
        assert plan["source"] is exact
        assert plan["reuse_tokens"] == 256

    def test_an_expired_exact_prefix_is_not_a_source(self):
        bank = _bank(idle_ttl_s=3_600.0)
        entry = _put(bank, tuple(range(10)), session_id="conv", nbytes=100)
        assert bank.restore_plan(tuple(range(20)))["keys"] == {entry.token_ids}
        entry.last_access_s -= 3_601.0
        assert bank.restore_plan(tuple(range(20)))["keys"] == set()

    def test_the_admission_walk_no_longer_protects_a_one_token_overlap(self):
        """Old rule: the entry with the greatest common prefix was protected
        even when that prefix was one token; shrink_for_admission(0) then
        kept it. Now it is walked like any other terminal."""

        bank = _bank()
        _put(bank, (1, 50, 51), session_id="a", nbytes=100)
        assert bank.shrink_for_admission(0, protect_tokens=(1, 99, 98)) == (0, 1)
        assert _keys(bank) == set()


class TestDurability:
    """Review defects 5 and 6: "on SSD" means published, and cancelling a
    queued encode must be per entry."""

    def test_published_sessions_go_first(self):
        tier = _Tier()
        bank = _bank(tier=tier, lane=_Lane())
        older = _put(bank, (1, 1, 1), session_id="older", nbytes=100)
        newer = _put(bank, (2, 2, 2), session_id="newer", nbytes=100)
        older.last_access_s = 1.0
        newer.last_access_s = 2.0
        tier.published.add(newer.token_ids)
        receipt = bank.release_sessions(100)
        assert _keys(bank) == {older.token_ids}
        [row] = receipt["sessions"]
        assert row["session_id"] == "newer"
        assert row["on_ssd_entries"] == 1
        assert receipt["dropped_entries"] == 0

    def test_an_encoded_but_unpublished_entry_is_not_durable(self):
        """The first cut read cold_encode_completed_at, which is set when the
        encode lands in the writer's queue, before the manifest row."""

        tier = _Tier()
        lane = _Lane()
        bank = _bank(tier=tier, lane=lane)
        entry = _put(bank, (1, 1, 1), session_id="conv", nbytes=100)
        lane.run_all()
        assert tier.encoded == [entry.token_ids]
        assert entry.cold_encode_completed_at is not None
        assert not bank.entry_is_durable(entry)
        receipt = bank.release_sessions(None)
        assert receipt["dropped_entries"] == 1

    def test_a_released_entrys_queued_encode_is_cancelled(self):
        lane = _Lane()
        bank = _bank(tier=_Tier(), lane=lane)
        _put(bank, (1, 1, 1), session_id="conv", nbytes=100)
        assert list(lane.pending) == ["ssd_cold:conv"]
        receipt = bank.release_sessions(None)
        assert lane.pending == {}
        assert receipt["persistence_cancelled"] == 1

    def test_a_kept_siblings_queued_encode_is_left_alone(self):
        lane = _Lane()
        bank = _bank(tier=_Tier(), lane=lane)
        released = _put(bank, (1, 2, 9), session_id="conv", nbytes=100)
        kept = _put(bank, (1, 2, 3, 4), session_id="conv", nbytes=100)
        # Newest wins: the queued job is the kept entry's.
        receipt = bank.release_sessions(None, protect_tokens=kept.token_ids + (5,))
        assert _keys(bank) == {kept.token_ids}
        assert released.token_ids not in _keys(bank)
        assert receipt["persistence_cancelled"] == 0
        assert list(lane.pending) == ["ssd_cold:conv"]

    def test_a_kept_entry_whose_job_was_coalesced_away_is_filed_again(self):
        tier = _Tier()
        lane = _Lane()
        bank = _bank(tier=tier, lane=lane)
        kept = _put(bank, (1, 2, 3, 4), session_id="conv", nbytes=100)
        # A newer sibling's job replaced the kept entry's (newest wins).
        _put(bank, (1, 2, 9), session_id="conv", nbytes=100)
        receipt = bank.release_sessions(None, protect_tokens=kept.token_ids + (5,))
        assert _keys(bank) == {kept.token_ids}
        assert receipt["persistence_cancelled"] == 1
        assert receipt["persistence_redispatched"] == 1
        lane.run_all()
        assert tier.encoded == [kept.token_ids]


class TestColdTierPublication:
    """``is_published`` asks the manifest, the only thing a restore reads."""

    @staticmethod
    def _entry(tokens: int):
        import mlx.core as mx

        from mtplx.session_bank import CacheSnapshot

        state = mx.zeros((1, 1, 8, 4), dtype=mx.float16)
        return SimpleNamespace(
            token_ids=tuple(range(tokens)),
            nbytes=1024,
            cache_snapshot=CacheSnapshot(states=((state, state),), meta_states=(None,)),
            logits=None,
            hidden=None,
            mtp_history_snapshot=None,
            gdn_boundaries=[],
            has_recurrent=False,
            session_id="s",
            token_hash=f"h{tokens}",
            model_path="/m",
            mtp_enabled=False,
            hidden_variant=None,
            template_hash=None,
            mtp_history_policy=None,
            draft_head_identity=None,
            policy_fingerprint=None,
            snapshot_epoch=tokens,
            mtp_snapshot_epoch=None,
        )

    def test_queued_failed_and_published_writes(self, tmp_path):
        from mtplx.cache_bank.cold_tier import SessionBankColdTier

        tier = SessionBankColdTier(base_dir=tmp_path / "ssd", mode="on")
        try:
            published = self._entry(2048)
            assert not tier.is_published(published)
            assert tier.put_entry(published)
            assert tier.flush(timeout_s=20)
            assert tier.is_published(published)

            failed = self._entry(4096)
            gate = threading.Event()

            def failing_write(pending):
                gate.wait(5)
                raise OSError("disk full")

            tier._write_pending = failing_write
            assert tier.put_entry(failed)
            # Queued: the encode returned, nothing is on disk.
            assert not tier.is_published(failed)
            gate.set()
            assert tier.flush(timeout_s=20)
            # The write failed: still nothing a restore could read.
            assert not tier.is_published(failed)
        finally:
            tier.close()


class TestSchedulerCancel:
    def test_cancel_drops_the_pending_job_and_the_arrays_it_pins(self):
        scheduler = ModelWorkScheduler(name="test-release", idle_grace_s=3600)
        try:
            ran = threading.Event()

            class Snapshot:
                pass

            pinned = Snapshot()
            ref = weakref.ref(pinned)
            entry = SimpleNamespace(session_id="conv", token_hash="h")

            def job(_held=pinned):
                ran.set()

            key = cold_persistence_key(entry)
            future = scheduler.submit_idle_persistence(job, coalesce_key=key)
            del job, pinned
            assert scheduler.cancel_idle_persistence("ssd_cold:other") == 0
            assert scheduler.cancel_idle_persistence(key) == 1
            assert future.cancelled()
            gc.collect()
            assert ref() is None
            assert not ran.is_set()
            assert scheduler.stats()["persistence_cancelled"] == 1
        finally:
            scheduler.shutdown(wait=True, cancel_futures=True)

    def test_the_key_constructor_matches_every_dispatch_shape(self):
        assert cold_persistence_key(SimpleNamespace(session_id="s", token_hash="h")) == "ssd_cold:s"
        assert cold_persistence_key(SimpleNamespace(session_id=None, token_hash="h")) == "ssd_cold:hash:h"


class TestManagerRelease:
    """Review defect 2: eviction must be atomic with request ownership."""

    def _manager(self) -> EngineSessionManager:
        return EngineSessionManager(bank=_bank(), idle_ttl_s=3600)

    def test_an_in_flight_session_is_never_touched(self):
        manager = self._manager()
        busy = manager.get_or_create("busy")
        manager.get_or_create("idle")
        _put(manager.bank, (1, 1, 1), session_id="busy", nbytes=100)
        _put(manager.bank, (2, 2, 2), session_id="idle", nbytes=100)
        assert busy.try_begin_generation()
        try:
            receipt = manager.release_idle_sessions(None)
        finally:
            busy.end_generation()
        assert manager.bank.has_session_entries("busy")
        assert not manager.bank.has_session_entries("idle")
        assert "busy" in receipt["kept_sessions"]

    def test_a_request_that_takes_its_slot_after_the_snapshot_is_skipped(
        self, monkeypatch
    ):
        """The first cut read the busy set once and evicted later; a request
        that took its slot in between lost its entries."""

        manager = self._manager()
        session = manager.get_or_create("conv")
        _put(manager.bank, (1, 1, 1), session_id="conv", nbytes=100)
        # The snapshot says nothing is in flight ...
        monkeypatch.setattr(manager, "in_flight_session_ids", lambda: set())
        # ... and the request takes its slot before the eviction.
        assert session.try_begin_generation()
        try:
            receipt = manager.release_idle_sessions(None)
        finally:
            session.end_generation()
        assert manager.bank.has_session_entries("conv")
        assert receipt["skipped_busy_sessions"] == ["conv"]

    def test_records_are_kept_so_one_session_has_one_lock(self):
        manager = self._manager()
        session = manager.get_or_create("conv")
        session.commit(prompt_ids=[1, 2, 3], generated_ids=[4, 5], finish_reason="stop")
        _put(manager.bank, (1, 2, 3, 4, 5), session_id="conv", nbytes=100)
        manager.release_idle_sessions(None, keep_session_ids={"anon-compaction"})
        # The record (and its committed tokens, which canonicalization and
        # turn boundaries read) survives; the same object serves the next
        # request, so no second lock can exist for the id.
        assert manager.get_or_create("conv") is session
        assert session.committed_token_ids == (1, 2, 3, 4, 5)
        # What the bank no longer backs is what the admission reads.
        assert manager.bank.session_coverage_tokens("conv", [1, 2, 3, 4, 5, 6]) == 0

    def test_a_request_arriving_during_a_release_waits_instead_of_409(self):
        manager = self._manager()
        session = manager.get_or_create("conv")
        assert session.begin_release_hold()
        acquired: list[bool] = []
        started = threading.Event()

        def request():
            started.set()
            acquired.append(session.try_begin_generation())

        thread = threading.Thread(target=request)
        thread.start()
        started.wait(1)
        time.sleep(0.05)
        assert acquired == []  # waiting for the release, not refused
        session.end_release_hold()
        thread.join(2)
        assert acquired == [True]
        session.end_generation()

    def test_a_named_session_is_not_refused_because_of_a_release(self):
        manager = self._manager()
        session = manager.get_or_create("named")
        assert session.begin_release_hold()
        timer = threading.Timer(0.05, session.end_release_hold)
        timer.start()
        with manager.generation_slot(session, source="header.x-session-id") as held:
            assert held is session
        timer.join()

    def test_a_busy_generation_still_reads_busy(self):
        """Only a release makes a request wait: a generation holding the slot
        is refused (named) or forked (implicit) as before."""

        manager = self._manager()
        session = manager.get_or_create("named")
        assert session.try_begin_generation()
        try:
            started = time.monotonic()
            assert not session.try_begin_generation()
            assert time.monotonic() - started < 0.5
        finally:
            session.end_generation()

    def test_a_released_sessions_pending_postcommit_is_aborted(self):
        from concurrent.futures import Future

        manager = self._manager()
        session = manager.get_or_create("conv")
        _put(manager.bank, (1, 2, 3), session_id="conv", nbytes=100)
        record = session.set_pending_postcommit(Future())
        receipt = manager.release_idle_sessions(None)
        assert receipt["postcommits_aborted"] == 1
        assert record.abort_event.is_set()

    def test_a_session_without_a_record_is_released_under_the_registry_lock(self):
        manager = self._manager()
        _put(manager.bank, (1, 2, 3), session_id="recordless", nbytes=100)
        seen_locked: list[bool] = []
        original = manager.bank._evict_entry

        def evict(entry, *, reason):
            seen_locked.append(manager._lock.locked())
            return original(entry, reason=reason)

        manager.bank._evict_entry = evict
        manager.release_idle_sessions(None)
        assert seen_locked == [True]
        assert not manager._lock.locked()

    def test_the_callers_sessions_are_kept(self):
        manager = self._manager()
        manager.get_or_create("incoming")
        _put(manager.bank, (1, 1, 1), session_id="incoming", nbytes=100)
        receipt = manager.release_idle_sessions(None, keep_session_ids={"incoming"})
        assert receipt["entries"] == 0
        assert manager.bank.has_session_entries("incoming")


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    # The idle-lane settle job is off by default; keep it off here so the
    # lane double sees persistence jobs only.
    monkeypatch.delenv("MTPLX_SESSION_SNAPSHOT_SETTLE", raising=False)
