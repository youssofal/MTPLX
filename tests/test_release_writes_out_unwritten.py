"""An idle conversation the memory guard releases is written to SSD first.

Two clients taking turns on one server, each with a long conversation: every
commit files the conversation's SSD write on the idle lane, which runs only
in a quiet window. With one client's request always queued behind the
other's, that window does not come. When the next request does not fit, the
admission's last step releases the idle conversation and cancelled its
queued write: the conversation was in RAM only, and its next turn re-read
everything (a field log: seven idle sessions of 7.5 to 9.9 GB dropped with
``on_ssd_entries: 0``, each followed by a re-read of 80K to 103K tokens, 4 to
6 minutes, while the other client waited behind it).

The release now writes such an entry to the SSD cache before it lets it go,
streamed one tensor at a time and bounded in time; these tests drive the real
``SessionBank`` and ``SessionBankColdTier`` with the engine busy (the idle
lane never runs, and the tier's foreground signal is up).
"""

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.server.openai as srv
import mtplx.session_bank as session_bank_module
import mtplx.system_memory as sm
from mtplx.cache_bank import SessionBankColdTier
from mtplx.engine_session import EngineSessionManager
from mtplx.session_bank import SessionBank
from tests.test_memguard_admission import (  # noqa: F401
    COMPACTION,
    CONV,
    FN_ROW,
    GIB,
    _flash_next_state,
    _install,
    _Machine,
    _served_profile,
)


class _Runtime:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []


class _KV:
    """A real cache leaf: state, meta_state, trimmable."""

    def __init__(self, rows: int = 64, width: int = 8) -> None:
        self.state = (
            mx.arange(rows * width, dtype=mx.float32)
            .reshape(1, 1, rows, width)
            .astype(mx.float16)
        )
        self.meta_state = ("kv", str(rows))

    def is_trimmable(self) -> bool:
        return True


class _Lane:
    """The idle persistence lane of a busy engine: it keeps the newest job
    per key and never runs one; cancel by key."""

    def __init__(self) -> None:
        self.pending: dict[str, object] = {}

    def dispatch(self, job) -> None:
        self.pending[job.coalesce_key] = job

    def cancel(self, key: str) -> int:
        return 1 if self.pending.pop(key, None) is not None else 0


@pytest.fixture
def cold(tmp_path):
    tier = SessionBankColdTier(
        base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=2
    )
    # A request is queued or running: idle-lane encodes stand down.
    tier.foreground_busy = lambda: True
    try:
        yield tier
    finally:
        tier.close()


def _bank(cold, lane: _Lane, **kwargs) -> SessionBank:
    bank = SessionBank(cold_tier=cold, **kwargs)
    bank.cold_enqueue_dispatch = lane.dispatch
    bank.cold_enqueue_cancel = lane.cancel
    return bank


def _put(
    bank: SessionBank,
    tokens,
    *,
    session_id: str,
    nbytes: int | None = None,
    live_cache=False,
):
    entry = bank.put(
        runtime=_Runtime(),
        token_ids=list(tokens),
        cache=[_KV()],
        logits=mx.array([[0.5, 1.5]], dtype=mx.float16),
        hidden=mx.array([[[2.0, 3.0]]], dtype=mx.float16),
        session_id=session_id,
        nbytes_override=nbytes,
    )
    assert entry is not None and not entry.live_ref_only
    if live_cache:
        # A generation-final commit keeps the live cache next to its snapshot.
        entry.cache_ref = object()
    return entry


def _release(bank: SessionBank, target, **kwargs):
    """The release as the build under test takes it: with
    MTPLX_TEST_BASELINE=1 a keyword the older build does not have is left
    out, so the test fails there by what the release does, not by a
    TypeError."""

    if os.environ.get("MTPLX_TEST_BASELINE") == "1":
        params = inspect.signature(bank.release_sessions).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return bank.release_sessions(target, **kwargs)


class TestTheTier:
    def test_a_deadline_replaces_the_foreground_signal(self, cold):
        """The idle-lane spill yields at once while a request is queued; the
        release's spill is that request's own work and runs."""

        bank = SessionBank()
        entry = _put(bank, range(1, 9), session_id="s1")
        assert cold.spill_entry(entry, capabilities=["ar_insert"]) is False
        assert not cold.is_published(entry)
        assert cold.spill_entry(
            entry, capabilities=["ar_insert"], give_up_at_s=time.monotonic() + 60.0
        )
        assert cold.is_published(entry)

    def test_a_spill_past_its_deadline_leaves_nothing_restorable(self, cold):
        from mtplx.cache_bank.codec import ColdEncodeInterrupted

        bank = SessionBank()
        entry = _put(bank, range(1, 9), session_id="s1")
        with pytest.raises(ColdEncodeInterrupted):
            cold.spill_entry(
                entry, capabilities=["ar_insert"], raise_on_yield=True, give_up_at_s=0.0
            )
        assert not cold.is_published(entry)
        assert cold.stats().get("spill_writes_completed", 0) == 0


class TestTheRelease:
    def test_an_unwritten_entry_is_written_before_it_goes(self, cold):
        lane = _Lane()
        bank = _bank(cold, lane)
        entry = _put(bank, range(1, 9), session_id="idle", live_cache=True)
        assert "ssd_cold:idle" in lane.pending
        assert not cold.is_published(entry)
        receipt = _release(
            bank, None, keep_session_ids={"incoming"}, write_out_before_release=True
        )
        assert not bank.has_session_entries("idle")
        # On disk: the next turn restores it instead of re-reading it.
        assert cold.is_published(entry)
        [row] = receipt["sessions"]
        assert row["on_ssd_entries"] == 1
        assert row["dropped_entries"] == 0
        assert receipt["written_out_entries"] == 1
        # Its queued write is gone with the RAM copy: nothing holds the arrays.
        assert lane.pending == {}
        assert bank.queued_persistence_bytes == 0
        [record] = [r for r in bank.eviction_log if r["reason"] == "release_write_out"]
        assert record["outcome"] == "written"

    def test_an_entry_pushed_out_of_ram_with_its_write_queued_is_written_too(
        self, cold
    ):
        """The budget evicted the entry and kept its write; the release finds
        it by the queue (test_memguard_queued_persistence) and writes it."""

        lane = _Lane()
        bank = _bank(cold, lane, max_bytes=5_000)
        old = _put(bank, range(1, 9), session_id="old", nbytes=4_000)
        # Out of the activity pin that put() stamps, so the budget takes it.
        bank._session_last_active["old"] = time.monotonic() - 7_200.0
        _put(bank, range(100, 108), session_id="new", nbytes=4_000)
        assert old.token_ids not in bank._entries
        assert "ssd_cold:old" in lane.pending
        receipt = _release(
            bank, None, keep_session_ids={"new"}, write_out_before_release=True
        )
        assert cold.is_published(old)
        assert receipt["written_out_entries"] == 1
        assert "ssd_cold:old" not in lane.pending
        assert "ssd_cold:new" in lane.pending

    def test_out_of_time_the_entry_is_dropped_as_before(self, cold, monkeypatch):
        monkeypatch.setattr(session_bank_module, "RELEASE_WRITE_OUT_MAX_S", -1.0)
        lane = _Lane()
        bank = _bank(cold, lane)
        entry = _put(bank, range(1, 9), session_id="idle")
        receipt = _release(
            bank, None, keep_session_ids={"incoming"}, write_out_before_release=True
        )
        assert not bank.has_session_entries("idle")
        assert not cold.is_published(entry)
        assert receipt["written_out_entries"] == 0
        assert receipt["sessions"][0]["dropped_entries"] == 1
        assert lane.pending == {}
        [record] = [r for r in bank.eviction_log if r["reason"] == "release_write_out"]
        assert record["outcome"] == "deadline"

    def test_without_it_the_release_cancels_the_write(self, cold):
        """A caller that does not opt in (the pressure trim, the shed after
        an allocation failure: off the model-owner thread, or past the line
        already) keeps the cancel."""

        lane = _Lane()
        bank = _bank(cold, lane)
        entry = _put(bank, range(1, 9), session_id="idle")
        receipt = bank.release_sessions(None, keep_session_ids={"incoming"})
        assert not cold.is_published(entry)
        assert receipt["persistence_cancelled"] == 1
        assert receipt.get("written_out_entries", 0) == 0

    def test_an_entry_already_on_disk_is_not_written_again(self, cold):
        lane = _Lane()
        bank = _bank(cold, lane)
        entry = _put(bank, range(1, 9), session_id="idle")
        assert cold.spill_entry(
            entry, capabilities=["ar_insert"], give_up_at_s=time.monotonic() + 60
        )
        writes = cold.stats()["writes_completed"]
        receipt = _release(
            bank, None, keep_session_ids={"incoming"}, write_out_before_release=True
        )
        assert receipt["written_out_entries"] == 0
        assert receipt["sessions"][0]["on_ssd_entries"] == 1
        assert cold.stats()["writes_completed"] == writes

    def test_no_ssd_cache_changes_nothing(self):
        lane = _Lane()
        bank = SessionBank()
        bank.cold_enqueue_dispatch = lane.dispatch
        bank.cold_enqueue_cancel = lane.cancel
        _put(bank, range(1, 9), session_id="idle")
        receipt = _release(
            bank, None, keep_session_ids={"incoming"}, write_out_before_release=True
        )
        assert receipt["written_out_entries"] == 0
        assert receipt["sessions"][0]["dropped_entries"] == 1
        assert not [r for r in bank.eviction_log if r["reason"] == "release_write_out"]

    def test_a_session_that_starts_a_request_is_not_written(self, cold):
        """The write runs only while the release holds the session; a busy
        session is skipped whole, as before."""

        lane = _Lane()
        bank = _bank(cold, lane)
        entry = _put(bank, range(1, 9), session_id="idle")
        receipt = _release(
            bank,
            None,
            keep_session_ids={"incoming"},
            hold_session=lambda session_id: None,
            write_out_before_release=True,
        )
        assert receipt["skipped_busy_sessions"] == ["idle"]
        assert bank.has_session_entries("idle")
        assert not cold.is_published(entry)
        assert "ssd_cold:idle" in lane.pending


class TestTheAdmission:
    def test_the_released_conversation_restores_from_ssd(self, cold, monkeypatch):
        """The field shape: a request of one client does not fit while the
        other client's long conversation waits in RAM (generation-final and
        postcommit entries, each holding a live cache), its SSD write still
        queued behind the busy engine. The admission releases it; before
        this change the queued write was cancelled and nothing was on disk
        (``on_ssd_entries: 0``)."""

        lane = _Lane()
        bank = _bank(
            cold,
            lane,
            max_entries=64,
            max_bytes=60 * GIB,
            per_session_max_bytes=30 * GIB,
        )
        manager = EngineSessionManager(bank=bank, idle_ttl_s=3600)
        _put(
            bank,
            CONV,
            session_id="anon-conv",
            nbytes=len(CONV) * FN_ROW,
            live_cache=True,
        )
        newest = _put(
            bank,
            CONV[:-1] + (999_999,),
            session_id="anon-conv",
            nbytes=len(CONV) * FN_ROW,
            live_cache=True,
        )
        assert "ssd_cold:anon-conv" in lane.pending
        incoming = manager.get_or_create("anon-compaction")
        assert incoming.try_begin_generation()
        machine = _Machine(bank, base_gib=88.6, cache_gib=0.5, host_gib=6.0, lane=lane)
        _install(monkeypatch, machine)
        try:
            receipt = srv._prefill_admission_shed(
                _flash_next_state(manager),
                prompt_ids=COMPACTION,
                session_bank=bank,
                session_id="anon-compaction",
            )
        finally:
            incoming.end_generation()
        assert receipt is not None and receipt.get("refused") is not True
        released = receipt["idle_release"]
        assert [row["session_id"] for row in released["sessions"]] == ["anon-conv"]
        assert not bank.has_session_entries("anon-conv")
        # The conversation's newest state, the one its queued write was for,
        # is on disk; the memory came back all the same.
        assert cold.is_published(newest)
        assert released["written_out_entries"] == 1
        assert released["sessions"][0]["on_ssd_entries"] == 1
        assert lane.pending == {}
        assert machine.queued() == 0
        assert receipt["projected_bytes_after"] <= 96 * GIB


def _starved(at_s: float, compressor_gib: float):
    """A Mac losing to compression: free pages under the abort floor while
    the compressor grows fast (the guard's death signature)."""

    return sm.SystemMemory(
        available_bytes=20 * GIB,
        total_bytes=128 * GIB,
        level_percent=20,
        free_bytes=16 * GIB // 1024,
        file_backed_bytes=20 * GIB - 16 * GIB // 1024,
        wired_bytes=88 * GIB,
        compressor_bytes=int(compressor_gib * GIB),
        swap_used_bytes=0,
        monotonic_s=at_s,
    )


class TestTheSideRequest:
    """One session id, two prompts: a client's side call (a title or summary
    request) arrives under its conversation's session id while the
    conversation's newest state waits in RAM, its SSD write queued behind
    turns a few seconds apart. The field log, 2026-10-29 on a 64 GB Mac: an
    8K side request tripped the prefill's death signature; the shed gave back
    the conversation's three entries (9 GB, nothing on disk), and its next
    turn re-read 23K tokens, 69 s to the first token."""

    SIDE = list(range(500_000, 500_040))

    def test_its_own_conversation_is_written_before_the_shed_lets_it_go(
        self, cold
    ):
        lane = _Lane()
        bank = _bank(cold, lane)
        _put(bank, range(1, 9), session_id="conv")
        newest = _put(bank, list(range(1, 9)) + [9, 10], session_id="conv")
        assert "ssd_cold:conv" in lane.pending
        receipt = srv._release_own_unusable_entries(
            session_bank=bank,
            session_id="conv",
            prompt_ids=self.SIDE,
            vision_splice=None,
            restore_identity=None,
            reason="prefill_shed_before_abort_own_session",
        )
        assert not bank.has_session_entries("conv")
        # The next turn of the conversation restores from disk.
        assert cold.is_published(newest)
        assert receipt["written_out_entries"] >= 1
        assert receipt["sessions"][0]["dropped_entries"] == 0
        assert lane.pending == {}

    def test_the_prefill_shed_writes_out_idle_and_own_conversations(
        self, cold, monkeypatch
    ):
        lane = _Lane()
        bank = _bank(cold, lane)
        manager = EngineSessionManager(bank=bank, idle_ttl_s=3600)
        other = _put(bank, range(100, 108), session_id="other")
        own = _put(bank, range(1, 9), session_id="conv")
        incoming = manager.get_or_create("conv")
        assert incoming.try_begin_generation()
        readings = iter([_starved(0.0, 10), _starved(1.0, 12)])
        last = [None]

        def reader():
            last[0] = next(readings, last[0])
            return last[0]

        monkeypatch.setattr(sm, "_reader", reader)
        monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
        monkeypatch.setattr(
            srv,
            "_mlx_memory_stats_live",
            lambda: {"ok": True, "active_memory_bytes": 80 * GIB, "cache_memory_bytes": 0},
        )
        monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 0)
        state = SimpleNamespace(
            dashboard=SimpleNamespace(), allow_swap=False, sessions=manager
        )

        def own_session_shed(reason):
            return srv._release_own_unusable_entries(
                session_bank=bank,
                session_id="conv",
                prompt_ids=self.SIDE,
                vision_splice=None,
                restore_identity=None,
                reason=reason,
            )

        guard = srv._PrefillSystemGuard(state, own_session_shed=own_session_shed)
        try:
            assert guard() is False
            # The death signature: the shed runs, the Mac still reads starved.
            assert guard() is True
        finally:
            incoming.end_generation()
        assert guard.tripped["reason"] == "death_signature"
        shed = guard.shed
        assert shed["released_sessions"] == ["other"]
        assert shed["written_out_entries"] == 1
        assert shed["own_session_written_out_entries"] == 1
        assert cold.is_published(other)
        assert cold.is_published(own)
        assert not bank.has_session_entries("other")
        assert not bank.has_session_entries("conv")
        assert lane.pending == {}


class TestTheAdmissionsOwnSiblings:
    def test_a_side_request_does_not_cost_its_conversation_the_ssd_copy(
        self, cold, monkeypatch
    ):
        """The admission's step 7, with the conversation and the request
        under one session id: the release that makes room writes the
        conversation's newest state out first."""

        lane = _Lane()
        bank = _bank(
            cold,
            lane,
            max_entries=64,
            max_bytes=60 * GIB,
            per_session_max_bytes=30 * GIB,
        )
        manager = EngineSessionManager(bank=bank, idle_ttl_s=3600)
        _put(
            bank,
            CONV,
            session_id="anon-conv",
            nbytes=len(CONV) * FN_ROW,
            live_cache=True,
        )
        newest = _put(
            bank,
            CONV[:-1] + (999_999,),
            session_id="anon-conv",
            nbytes=len(CONV) * FN_ROW,
            live_cache=True,
        )
        assert "ssd_cold:anon-conv" in lane.pending
        incoming = manager.get_or_create("anon-conv")
        assert incoming.try_begin_generation()
        machine = _Machine(bank, base_gib=88.6, cache_gib=0.5, host_gib=6.0, lane=lane)
        _install(monkeypatch, machine)
        try:
            receipt = srv._prefill_admission_shed(
                _flash_next_state(manager),
                prompt_ids=COMPACTION,
                session_bank=bank,
                session_id="anon-conv",
            )
        finally:
            incoming.end_generation()
        assert receipt is not None and receipt.get("refused") is not True
        own = receipt["own_session_release"]
        # The newest state, the one its queued write was for, is on disk.
        assert own["written_out_entries"] == 1
        assert own["sessions"][0]["on_ssd_entries"] == 1
        assert not bank.has_session_entries("anon-conv")
        assert cold.is_published(newest)
        assert lane.pending == {}
