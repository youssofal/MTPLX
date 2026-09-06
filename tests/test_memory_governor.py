from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
from mtplx.memory_governor import (
    GIB,
    MemoryGovernorAction,
    MemoryGovernorConfig,
    MemoryPressureLevel,
    MemorySafePoint,
    MemorySample,
    RuntimeMemoryGovernor,
    sample_process_memory,
)


class FakeBank:
    def __init__(self, max_bytes=20 * GIB, per_session=8 * GIB):
        self.max_bytes = max_bytes
        self.per_session_max_bytes = per_session
        self._entries = {"a": object(), "b": object()}
        self.total_nbytes = 12 * GIB
        self.calls = []

    def rebalance_limits(self, *, max_bytes, per_session_max_bytes, reason):
        self.calls.append((max_bytes, per_session_max_bytes, reason))
        self.max_bytes = int(max_bytes)
        self.per_session_max_bytes = int(per_session_max_bytes)
        if self.total_nbytes > self.max_bytes:
            self._entries.pop("a", None)
            self.total_nbytes = self.max_bytes


def _sample(
    rss_gib: float,
    *,
    bank_gib: float = 12,
    total_gib: float = 100,
    safe: bool = True,
    timestamp: float = 10.0,
):
    return MemorySample(
        total_bytes=int(total_gib * GIB),
        rss_bytes=int(rss_gib * GIB),
        session_bank_bytes=int(bank_gib * GIB),
        model_bytes=60 * GIB,
        timestamp_s=timestamp,
        safe_point=(
            MemorySafePoint()
            if safe
            else MemorySafePoint(foreground_active=1)
        ),
    )


def _governor(**config_overrides):
    config = MemoryGovernorConfig(
        minimum_apply_interval_s=0.0,
        **config_overrides,
    )
    return RuntimeMemoryGovernor(
        initial_bank_max_bytes=20 * GIB,
        initial_per_session_max_bytes=8 * GIB,
        config=config,
    )


def test_critical_pressure_shrinks_immediately():
    governor = _governor()
    decision = governor.observe(_sample(95))
    assert decision.pressure == MemoryPressureLevel.CRITICAL
    assert decision.action == MemoryGovernorAction.SHRINK
    assert decision.target_bank_max_bytes < 20 * GIB


def test_high_pressure_requires_hysteresis_observations():
    governor = _governor(high_observations=2)
    first = governor.observe(_sample(87, timestamp=1))
    second = governor.observe(_sample(87, timestamp=2))
    assert first.action == MemoryGovernorAction.HOLD
    assert second.action == MemoryGovernorAction.SHRINK
    assert second.reason == "sustained_high_pressure"


def test_normal_observation_resets_pressure_streak():
    governor = _governor(high_observations=2)
    governor.observe(_sample(87, timestamp=1))
    governor.observe(_sample(80, timestamp=2))
    decision = governor.observe(_sample(87, timestamp=3))
    assert decision.action == MemoryGovernorAction.HOLD
    assert governor.high_streak == 1


def test_recovery_grows_only_after_hysteresis_and_never_past_startup_budget():
    governor = _governor(recovery_observations=3)
    governor.current_bank_max_bytes = 8 * GIB
    governor.current_per_session_max_bytes = 5 * GIB
    assert governor.observe(_sample(55, bank_gib=6, timestamp=1)).action == MemoryGovernorAction.HOLD
    assert governor.observe(_sample(55, bank_gib=6, timestamp=2)).action == MemoryGovernorAction.HOLD
    decision = governor.observe(_sample(55, bank_gib=6, timestamp=3))
    assert decision.action == MemoryGovernorAction.GROW
    assert 8 * GIB < decision.target_bank_max_bytes <= 20 * GIB
    assert decision.target_per_session_max_bytes <= decision.target_bank_max_bytes


def test_unknown_total_memory_holds_current_budget():
    governor = _governor()
    decision = governor.observe(
        MemorySample(
            total_bytes=None,
            rss_bytes=10 * GIB,
            session_bank_bytes=2 * GIB,
            timestamp_s=1,
        )
    )
    assert decision.pressure == MemoryPressureLevel.UNKNOWN
    assert decision.action == MemoryGovernorAction.HOLD


def test_unsafe_point_rejects_budget_mutation():
    governor = _governor()
    bank = FakeBank()
    decision = governor.observe(_sample(95, safe=False))
    receipt = governor.apply(decision, bank=bank)
    assert receipt.applied is False
    assert receipt.reason == "unsafe_point:foreground"
    assert bank.calls == []
    assert bank.max_bytes == 20 * GIB


def test_safe_point_applies_session_bank_limits():
    governor = _governor()
    bank = FakeBank()
    decision = governor.observe(_sample(95, safe=True))
    receipt = governor.apply(decision, bank=bank)
    assert receipt.applied is True
    assert bank.max_bytes == decision.target_bank_max_bytes
    assert bank.per_session_max_bytes == decision.target_per_session_max_bytes
    assert bank.calls[-1][2] == "runtime_memory_governor"
    assert receipt.evicted_entries == 1


def test_safe_point_supports_current_session_bank_eviction_primitive():
    class CurrentBankShape:
        def __init__(self):
            self.max_bytes = 20 * GIB
            self.per_session_max_bytes = 8 * GIB
            self.total_nbytes = 12 * GIB
            self._entries = {"a": object(), "b": object()}
            self.evict_calls = 0

        def _evict_if_needed(self):
            self.evict_calls += 1
            self._entries.pop("a", None)
            self.total_nbytes = self.max_bytes

    governor = _governor()
    bank = CurrentBankShape()
    decision = governor.observe(_sample(95, safe=True))
    receipt = governor.apply(decision, bank=bank)
    assert receipt.applied is True
    assert bank.max_bytes == decision.target_bank_max_bytes
    assert bank.per_session_max_bytes == decision.target_per_session_max_bytes
    assert bank.evict_calls == 1
    assert receipt.evicted_entries == 1


@pytest.mark.parametrize(
    ("safe_point", "blocker"),
    [
        (MemorySafePoint(foreground_active=1), "foreground"),
        (MemorySafePoint(scheduler_pending_or_active=True), "scheduler"),
        (MemorySafePoint(session_restore_active=True), "session_restore"),
        (MemorySafePoint(session_commit_active=True), "session_commit"),
        (MemorySafePoint(mtp_transaction_active=True), "mtp_transaction"),
        (MemorySafePoint(postcommit_active=True), "postcommit"),
    ],
)
def test_every_live_state_blocks_application(safe_point, blocker):
    governor = _governor()
    bank = FakeBank()
    decision = governor.observe(replace(_sample(95), safe_point=safe_point))
    receipt = governor.apply(decision, bank=bank)
    assert receipt.applied is False
    assert blocker in receipt.reason


@pytest.mark.parametrize(
    ("rss", "expected"),
    [
        (65, MemoryPressureLevel.LOW),
        (78, MemoryPressureLevel.NORMAL),
        (86, MemoryPressureLevel.HIGH),
        (93, MemoryPressureLevel.CRITICAL),
    ],
)
def test_pressure_matrix(rss, expected):
    governor = _governor(high_observations=1)
    assert governor.observe(_sample(rss)).pressure == expected


def test_sample_process_memory_respects_supplied_values_without_platform_probe():
    sample = sample_process_memory(
        session_bank_bytes=3 * GIB,
        model_bytes=50 * GIB,
        total_bytes=128 * GIB,
        rss_bytes=90 * GIB,
        safe_point=MemorySafePoint(),
    )
    assert sample.total_bytes == 128 * GIB
    assert sample.rss_bytes == 90 * GIB
    assert sample.session_bank_bytes == 3 * GIB
    assert sample.safe_point.is_safe
    assert sample.to_dict()["utilization"] == pytest.approx(90 / 128)


def test_metrics_include_last_decision_and_receipt():
    governor = _governor()
    bank = FakeBank()
    decision = governor.observe(_sample(95))
    governor.apply(decision, bank=bank)
    metrics = governor.to_metrics()
    assert metrics["memory_governor_last_decision"]["pressure"] == "critical"
    assert metrics["memory_governor_last_apply"]["applied"] is True


def test_server_constructs_governor_from_bank_limits_when_enabled(monkeypatch):
    bank = FakeBank(max_bytes=16 * GIB, per_session=6 * GIB)
    monkeypatch.setenv("MTPLX_MEMORY_GOVERNOR", "1")
    governor = srv._create_memory_governor(bank)
    assert governor is not None
    assert governor.initial_bank_max_bytes == 16 * GIB
    assert governor.initial_per_session_max_bytes == 6 * GIB

    monkeypatch.setenv("MTPLX_MEMORY_GOVERNOR", "0")
    assert srv._create_memory_governor(bank) is None


def test_server_safe_point_reports_scheduler_and_postcommit_activity():
    scheduler = SimpleNamespace(
        stats=lambda: {
            "foreground_pending": 0,
            "idle_pending": 1,
            "persistence_pending": 0,
            "active_kind": "idle_postcommit",
        }
    )
    state = SimpleNamespace(
        foreground_count=lambda: 0,
        model_scheduler=scheduler,
        lock=threading.Lock(),
    )
    safe_point = srv._memory_governor_safe_point(state)
    assert safe_point.is_safe is False
    assert safe_point.scheduler_pending_or_active is True
    assert safe_point.session_commit_active is True
    assert safe_point.postcommit_active is True


def test_server_tick_applies_critical_shrink_under_model_lock(monkeypatch):
    lock = threading.Lock()

    class LockCheckingBank(FakeBank):
        def rebalance_limits(self, *, max_bytes, per_session_max_bytes, reason):
            assert lock.locked()
            super().rebalance_limits(
                max_bytes=max_bytes,
                per_session_max_bytes=per_session_max_bytes,
                reason=reason,
            )

    bank = LockCheckingBank()
    governor = _governor()
    state = SimpleNamespace(
        memory_governor=governor,
        sessions=SimpleNamespace(bank=bank),
        lock=lock,
        foreground_count=lambda: 0,
        model_scheduler=SimpleNamespace(stats=lambda: {}),
        model_weights_bytes=60 * GIB,
    )

    def synthetic_sample(**kwargs):
        assert lock.locked() is False
        return replace(_sample(95), safe_point=kwargs["safe_point"])

    monkeypatch.setattr(srv, "sample_process_memory", synthetic_sample)
    monkeypatch.setattr(
        srv,
        "_machine_info",
        lambda: {"unified_memory_bytes": 100 * GIB},
    )
    result = srv._memory_governor_tick(state)
    assert result is not None
    assert result["decision"]["pressure"] == "critical"
    assert result["apply"]["applied"] is True
    assert bank.calls[-1][2] == "runtime_memory_governor"
    assert lock.locked() is False


def test_server_tick_fails_closed_when_model_lock_is_busy(monkeypatch):
    lock = threading.Lock()
    lock.acquire()
    bank = FakeBank()
    state = SimpleNamespace(
        memory_governor=_governor(),
        sessions=SimpleNamespace(bank=bank),
        lock=lock,
        model_weights_bytes=60 * GIB,
    )

    def synthetic_sample(**kwargs):
        return replace(_sample(95), safe_point=kwargs["safe_point"])

    monkeypatch.setattr(srv, "sample_process_memory", synthetic_sample)
    monkeypatch.setattr(
        srv,
        "_machine_info",
        lambda: {"unified_memory_bytes": 100 * GIB},
    )
    try:
        result = srv._memory_governor_tick(state)
    finally:
        lock.release()
    assert result is not None
    assert result["apply"]["applied"] is False
    assert "mtp_transaction" in result["apply"]["reason"]
    assert bank.calls == []


def test_memory_pressure_loop_runs_governor_without_legacy_guard(monkeypatch):
    monkeypatch.setenv("MTPLX_MEMORY_GOVERNOR", "1")
    monkeypatch.setenv("MTPLX_MEMORY_PRESSURE_GUARD", "0")
    monkeypatch.setattr(srv, "_memory_pressure_level", lambda: 1)
    calls = []
    monkeypatch.setattr(
        srv,
        "_memory_governor_tick",
        lambda state: calls.append(state),
    )
    state = SimpleNamespace(
        dashboard=SimpleNamespace(last_memory_pressure_level=0)
    )

    async def one_tick():
        task = asyncio.create_task(
            srv._memory_pressure_loop(state, interval_s=3600)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(one_tick())
    assert calls == [state]
    assert state.dashboard.last_memory_pressure_level == 1
