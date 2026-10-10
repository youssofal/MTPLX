"""A WARNING pressure trim keeps an idle conversation whose SSD write is queued.

Two clients taking turns on one server: every commit files the conversation's
SSD write on the idle lane, which runs only in a quiet window, and with one
client's request always queued behind the other's that window does not come.
The memory guard's pressure trim runs off the model-owner thread while a
request is running; at WARNING it waits up to 60 s for an idle engine, then
trims anyway. It took the idle conversation and cancelled its queued write,
so RAM had held the only copy: a 64K conversation was dropped during the other
client's cold prefill (the allocator at 0.997 of its limit), and its next turn
re-read all 63,881 tokens (173 s).

A WARNING trim now passes over such an entry and takes conversations already
on SSD, or that nothing will write; the receipt counts what it kept. CRITICAL
keeps taking everything idle, and says how many unwritten entries went. These
tests drive the real loop, ``SessionBank`` and ``SessionBankColdTier`` with
the engine busy (the idle lane never runs).
"""

from __future__ import annotations

import functools
import time
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.cache_bank import SessionBankColdTier
from mtplx.engine_session import EngineSessionManager
from mtplx.session_bank import SessionBank
from tests.test_memguard_admission import GIB, _run_loop
from tests.test_memguard_pressure_trim import _reading
from tests.test_release_writes_out_unwritten import _bank, _Lane, _put


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


@pytest.fixture(autouse=True)
def _a_busy_engine_past_its_deferral(monkeypatch):
    # A request is running and the WARNING has outlasted the 60 s the loop
    # waits for an idle engine (a zero deferral stands in for the minute).
    # The rest of the Mac is fine: the engine's own allocator raises the level.
    monkeypatch.setattr(srv, "_engine_busy_signal", lambda state: True)
    monkeypatch.setattr(
        srv,
        "_MemoryPressureGuard",
        functools.partial(srv._MemoryPressureGuard, warning_defer_max_s=0.0),
    )
    monkeypatch.setattr(sm, "_reader", lambda: _reading(60.0))


def _allocator_at(monkeypatch, level: int, fraction: float) -> None:
    monkeypatch.setattr(
        srv, "_allocator_pressure_level", lambda state: (level, fraction)
    )


def _state(manager):
    return SimpleNamespace(
        sessions=manager,
        dashboard=SimpleNamespace(last_memory_pressure_level=0),
    )


def _trim(state) -> dict:
    events = getattr(state.dashboard, "memory_guard_events", ()) or ()
    trims = [event for event in events if event.get("action") == "pressure_trim"]
    assert trims, "the loop did not trim"
    return trims[-1]


def _written(cold, entry) -> None:
    """The idle lane's write, run: the entry is published on disk."""

    assert cold.spill_entry(
        entry, capabilities=["ar_insert"], give_up_at_s=time.monotonic() + 60.0
    )
    assert cold.is_published(entry)


def _two_clients(cold, lane):
    """The field shape: client B is prefilling (in flight), client A's
    conversation is idle in RAM with its SSD write still queued."""

    bank = _bank(
        cold, lane, max_entries=64, max_bytes=60 * GIB, per_session_max_bytes=30 * GIB
    )
    manager = EngineSessionManager(bank=bank, idle_ttl_s=3600)
    a = _put(bank, range(1, 65), session_id="A", nbytes=5 * GIB, live_cache=True)
    b = _put(bank, range(1_000, 1_064), session_id="B", nbytes=8 * GIB)
    session = manager.get_or_create("B")
    assert session.try_begin_generation()
    assert "ssd_cold:A" in lane.pending and not cold.is_published(a)
    return bank, manager, session, a, b


class TestWarning:
    def test_it_keeps_the_idle_conversation_whose_write_is_queued(
        self, cold, monkeypatch
    ):
        """13 GiB resident, a 6.5 GiB target: before, A went with its queued
        write and its next turn re-read everything."""

        lane = _Lane()
        bank, manager, session, a, _b = _two_clients(cold, lane)
        _allocator_at(monkeypatch, 2, 0.997)
        state = _state(manager)
        try:
            _run_loop(state, monkeypatch, seconds=0.05)
        finally:
            session.end_generation()
        trim = _trim(state)
        assert trim["level"] == 2
        assert bank._entries.get(a.token_ids) is a
        assert "ssd_cold:A" in lane.pending
        assert trim["bank_entries_evicted"] == 0
        assert trim["bank_unwritten_kept"] == 1
        assert trim["bank_unwritten_dropped"] == 0

    def test_it_takes_a_conversation_already_on_ssd_instead(self, cold, monkeypatch):
        """A is the least recently used, so the LRU order alone took it; C is
        on disk and restores from there."""

        lane = _Lane()
        bank = _bank(cold, lane, max_bytes=60 * GIB, per_session_max_bytes=30 * GIB)
        manager = EngineSessionManager(bank=bank, idle_ttl_s=3600)
        a = _put(bank, range(1, 65), session_id="A", nbytes=4 * GIB)
        c = _put(bank, range(2_000, 2_064), session_id="C", nbytes=4 * GIB)
        a.last_access_s -= 3_600.0
        _written(cold, c)
        _allocator_at(monkeypatch, 2, 0.997)
        state = _state(manager)
        _run_loop(state, monkeypatch, seconds=0.05)
        trim = _trim(state)
        assert c.token_ids not in bank._entries
        assert bank._entries.get(a.token_ids) is a
        assert "ssd_cold:A" in lane.pending
        assert trim["bank_entries_evicted"] == 1
        assert trim["bank_unwritten_dropped"] == 0

    def test_once_its_write_has_run_a_later_trim_takes_it(self, cold, monkeypatch):
        lane = _Lane()
        bank, manager, session, a, b = _two_clients(cold, lane)
        _allocator_at(monkeypatch, 2, 0.997)
        try:
            _run_loop(_state(manager), monkeypatch, seconds=0.05)
            assert bank._entries.get(a.token_ids) is a
            # A quiet window came: the idle lane wrote it.
            _written(cold, a)
            state = _state(manager)
            _run_loop(state, monkeypatch, seconds=0.05)
        finally:
            session.end_generation()
        assert a.token_ids not in bank._entries
        assert _trim(state)["bank_entries_evicted"] == 1
        assert bank._entries.get(b.token_ids) is b

    def test_an_entry_nothing_will_write_is_taken_as_before(self, monkeypatch):
        """No SSD cache: keeping it saves nothing, so the trim takes it."""

        bank = SessionBank(max_bytes=60 * GIB, per_session_max_bytes=30 * GIB)
        manager = EngineSessionManager(bank=bank, idle_ttl_s=3600)
        a = _put(bank, range(1, 65), session_id="A", nbytes=5 * GIB)
        _allocator_at(monkeypatch, 2, 0.997)
        state = _state(manager)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert a.token_ids not in bank._entries
        assert _trim(state)["bank_unwritten_kept"] == 0


class TestCritical:
    def test_it_still_takes_everything_idle_and_counts_the_unwritten(
        self, cold, monkeypatch
    ):
        lane = _Lane()
        bank, manager, session, a, b = _two_clients(cold, lane)
        _allocator_at(monkeypatch, 4, 1.03)
        state = _state(manager)
        try:
            _run_loop(state, monkeypatch, seconds=0.05)
        finally:
            session.end_generation()
        trim = _trim(state)
        assert trim["level"] == 4
        assert a.token_ids not in bank._entries
        assert "ssd_cold:A" not in lane.pending
        assert bank._entries.get(b.token_ids) is b
        assert trim["bank_unwritten_dropped"] == 1
        assert trim["bank_unwritten_kept"] == 0
