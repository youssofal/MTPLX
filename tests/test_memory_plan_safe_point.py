from __future__ import annotations

from concurrent.futures import Future
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.server import openai as srv
from test_memory_pressure_guard import FakeDynamicBank, make_state, run_one_tick


@pytest.fixture
def state():
    result = make_state(FakeDynamicBank(8 << 30, 8 << 30, 2 << 30))
    result.model_scheduler = ModelWorkScheduler(
        name="safe-point-test", idle_grace_s=60, persistence_quiet_grace_s=60
    )
    try:
        yield result
    finally:
        result.model_scheduler.shutdown(wait=True, cancel_futures=True)


def reason(state):
    return list(state.dashboard.memory_guard_events)[-1]["reason"]


def test_samples_outside_lock_and_applies_inside_both_gates(state):
    bank = state.sessions.bank
    samples = []
    applied = []
    shrink = bank.shrink_to_bytes

    def sample():
        samples.append(state.lock.locked())
        return 2 << 30

    def apply(*args, **kwargs):
        applied.append(state.lock.locked())
        # A contender cannot enter the scheduler condition during apply.
        acquired = []

        def contend():
            got = state.model_scheduler._condition.acquire(blocking=False)
            acquired.append(got)
            if got:
                state.model_scheduler._condition.release()

        thread = Thread(target=contend)
        thread.start()
        thread.join(timeout=2)
        assert acquired == [False]
        return shrink(*args, **kwargs)

    bank.effective_max_bytes = sample
    bank.shrink_to_bytes = apply
    srv._apply_dynamic_bank_ceiling_at_safe_point(state)
    assert samples == [False]
    assert applied == [True]
    assert bank.calls == [(2 << 30, "dynamic_ceiling")]
    assert bank.protect_active_calls == [True]
    assert bank.max_bytes == 8 << 30
    assert not state.lock.locked()


def test_model_lock_contention_returns_without_mutation(state):
    with state.lock:
        srv._apply_dynamic_bank_ceiling_at_safe_point(state)
    assert state.sessions.bank.calls == []
    assert reason(state) == "model_lock_busy"


def test_foreground_arriving_during_sample_is_rechecked(state):
    def sample():
        state.foreground_count = lambda: 1
        return 2 << 30

    state.sessions.bank.effective_max_bytes = sample
    srv._apply_dynamic_bank_ceiling_at_safe_point(state)
    assert state.sessions.bank.calls == []
    assert reason(state) == "foreground_busy"


@pytest.mark.parametrize("probe", ["foreground", "requests", "session_ids"])
def test_failed_activity_probe_never_proves_idle(state, probe):
    def fail():
        raise RuntimeError("unavailable")

    if probe == "foreground":
        state.foreground_count = fail
    elif probe == "requests":
        state.dashboard.in_flight.count = fail
    else:
        state.dashboard.in_flight.session_ids = fail
    srv._apply_dynamic_bank_ceiling_at_safe_point(state)
    assert state.sessions.bank.calls == []
    assert reason(state) == "activity_probe_failed"
    assert not state.lock.locked()


def test_request_between_owner_items_defers(state):
    state.dashboard.in_flight.count = lambda: 1
    srv._apply_dynamic_bank_ceiling_at_safe_point(state)
    assert state.sessions.bank.calls == []
    assert reason(state) == "foreground_busy"


@pytest.mark.parametrize(
    "submit", ["submit_idle_postcommit", "submit_idle_persistence"]
)
def test_pending_background_work_defers_even_before_ready_time(state, submit):
    getattr(state.model_scheduler, submit)(lambda: None)
    srv._apply_dynamic_bank_ceiling_at_safe_point(state)
    assert state.sessions.bank.calls == []
    assert reason(state) == "scheduler_busy"


@pytest.mark.parametrize(
    "submit", ["submit_foreground", "submit_idle_postcommit", "submit_idle_persistence"]
)
def test_running_owner_work_defers(state, submit):
    scheduler = state.model_scheduler
    scheduler.idle_grace_s = 0
    scheduler.persistence_quiet_grace_s = 0
    started, release = Event(), Event()

    def work():
        started.set()
        assert release.wait(timeout=5)

    future = getattr(scheduler, submit)(work)
    try:
        assert started.wait(timeout=2)
        srv._apply_dynamic_bank_ceiling_at_safe_point(state)
        assert state.sessions.bank.calls == []
        assert reason(state) == "scheduler_busy"
    finally:
        release.set()
        future.result(timeout=2)


def test_dequeued_work_cannot_look_idle_before_start(state, monkeypatch):
    scheduler = state.model_scheduler
    # Interpose before the future reports running: queues are already empty
    # and active_kind is not yet set, but the owner has claimed this work.
    original = Future.set_running_or_notify_cancel
    claimed, release = Event(), Event()

    def pause(future):
        claimed.set()
        assert release.wait(timeout=5)
        return original(future)

    monkeypatch.setattr(Future, "set_running_or_notify_cancel", pause)
    future = scheduler.submit_foreground(lambda: None)
    try:
        assert claimed.wait(timeout=2)
        assert scheduler.stats()["active_kind"] is None
        assert scheduler.any_pending_or_active()
        srv._apply_dynamic_bank_ceiling_at_safe_point(state)
        assert state.sessions.bank.calls == []
        assert reason(state) == "scheduler_busy"
    finally:
        release.set()
        future.result(timeout=2)


def test_keepalive_work_cannot_look_idle(state):
    scheduler = state.model_scheduler
    started, release = Event(), Event()

    def beat():
        started.set()
        assert release.wait(timeout=5)

    scheduler.arm_idle_keepalive(beat, interval_s=0.001, attentive_s=10)
    try:
        assert started.wait(timeout=2)
        srv._apply_dynamic_bank_ceiling_at_safe_point(state)
        assert state.sessions.bank.calls == []
        assert reason(state) == "scheduler_busy"
    finally:
        scheduler.disarm_idle_keepalive()
        release.set()


def test_shrink_exception_releases_both_gates(state):
    def fail(*args, **kwargs):
        raise RuntimeError("bank failed")

    state.sessions.bank.shrink_to_bytes = fail
    srv._apply_dynamic_bank_ceiling_at_safe_point(state)
    assert reason(state) == "safe_point_apply_failed"
    assert not state.lock.locked()
    assert state.model_scheduler.run_idle_maintenance(lambda: None)


def test_missing_scheduler_gate_is_not_an_idle_fallback(state):
    scheduler = state.model_scheduler
    try:
        state.model_scheduler = SimpleNamespace()
        srv._apply_dynamic_bank_ceiling_at_safe_point(state)
        assert state.sessions.bank.calls == []
        assert reason(state) == "scheduler_gate_unavailable"
    finally:
        state.model_scheduler = scheduler


def test_bank_replacement_during_sample_is_not_trimmed(state):
    old = state.sessions.bank
    replacement = FakeDynamicBank(8 << 30, 8 << 30, 1 << 30)

    def sample():
        state.sessions.bank = replacement
        return 2 << 30

    old.effective_max_bytes = sample
    srv._apply_dynamic_bank_ceiling_at_safe_point(state)
    assert old.calls == replacement.calls == []
    assert reason(state) == "bank_changed"


def test_critical_shedding_still_acts_during_foreground(state, monkeypatch):
    state.foreground_count = lambda: 1
    with state.lock:
        run_one_tick(state, level=4, monkeypatch=monkeypatch)
    assert state.sessions.bank.calls == [(0, "memory_pressure_critical")]
    assert state.sessions.bank.protect_active_calls == [False]
