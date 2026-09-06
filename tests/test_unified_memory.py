from __future__ import annotations

from dataclasses import dataclass

from mtplx.unified_memory import (
    GIB,
    UnifiedMemoryConfig,
    UnifiedMemoryCoordinator,
    UnifiedMemorySample,
)


@dataclass
class FakeConsumer:
    name: str
    budget: int
    fail: bool = False

    def current_budget_bytes(self) -> int:
        return self.budget

    def apply_budget_bytes(self, value: int, *, reason: str) -> int:
        assert reason
        if self.fail:
            raise RuntimeError("boom")
        self.budget = int(value)
        return self.budget


def config(**kwargs):
    values = {
        "enabled": True,
        "reserve_bytes": 0,
        "minimum_available_bytes": 0,
        "target_utilization": 0.8,
        "warning_utilization": 0.9,
        "critical_utilization": 0.95,
        "minimum_apply_interval_s": 0.0,
        "hysteresis_ratio": 0.0,
        "minimum_session_bank_bytes": 1 * GIB,
        "minimum_expert_bytes": 1 * GIB,
        "minimum_kv_headroom_bytes": 1 * GIB,
    }
    values.update(kwargs)
    return UnifiedMemoryConfig(**values)


def sample(
    *, total=100 * GIB, process=50 * GIB, session=5 * GIB, expert=5 * GIB, kv=5 * GIB
):
    return UnifiedMemorySample(
        total_bytes=total,
        process_bytes=process,
        model_bytes=30 * GIB,
        session_bank_bytes=session,
        expert_bytes=expert,
        kv_bytes=kv,
    )


def test_plan_never_exceeds_managed_budget_and_preserves_kv_headroom():
    coordinator = UnifiedMemoryCoordinator(config())
    plan = coordinator.plan(sample(), safe=True, now_s=10.0)
    assert plan.eligible is True
    assert (
        plan.session_bank_budget_bytes
        + plan.expert_budget_bytes
        + plan.kv_headroom_bytes
        <= plan.managed_budget_bytes
    )
    assert plan.kv_headroom_bytes >= 1 * GIB


def test_critical_pressure_shrinks_elastic_budget():
    coordinator = UnifiedMemoryCoordinator(config())
    normal = coordinator.plan(sample(process=40 * GIB), safe=True, now_s=10.0)
    critical = coordinator.plan(sample(process=98 * GIB), safe=True, now_s=11.0)
    assert critical.pressure == "critical"
    assert critical.managed_budget_bytes < normal.managed_budget_bytes
    assert critical.expert_budget_bytes <= normal.expert_budget_bytes


def test_apply_rolls_back_prior_consumer_when_later_consumer_fails():
    coordinator = UnifiedMemoryCoordinator(config())
    plan = coordinator.plan(sample(), safe=True, now_s=10.0)
    session = FakeConsumer("session_bank", 10 * GIB)
    expert = FakeConsumer("expert_residency", 4 * GIB, fail=True)

    receipt = coordinator.apply(plan, [session, expert], safe=True)

    assert receipt.applied is False
    assert receipt.rolled_back is True
    assert session.budget == 10 * GIB
    assert receipt.reason == "apply_failed:RuntimeError"


def test_apply_is_blocked_outside_safe_point():
    coordinator = UnifiedMemoryCoordinator(config())
    plan = coordinator.plan(sample(), safe=True, now_s=10.0)
    session = FakeConsumer("session_bank", 10 * GIB)
    receipt = coordinator.apply(plan, [session], safe=False)
    assert receipt.applied is False
    assert receipt.reason == "unsafe_point"
    assert session.budget == 10 * GIB


def test_hysteresis_avoids_small_budget_churn():
    coordinator = UnifiedMemoryCoordinator(config(hysteresis_ratio=0.2))
    plan = coordinator.plan(sample(), safe=True, now_s=10.0)
    session = FakeConsumer("session_bank", plan.session_bank_budget_bytes + 1)
    expert = FakeConsumer("expert_residency", plan.expert_budget_bytes + 1)
    receipt = coordinator.apply(plan, [session, expert], safe=True)
    assert receipt.applied is True
    assert all(
        not item.applied and item.reason == "hysteresis" for item in receipt.mutations
    )


def test_disabled_or_unsafe_plan_is_ineligible():
    disabled = UnifiedMemoryCoordinator(config(enabled=False))
    assert disabled.plan(sample(), safe=True).reason == "disabled"
    enabled = UnifiedMemoryCoordinator(config())
    assert enabled.plan(sample(), safe=False).reason == "unsafe_point"


def test_apply_mutates_explicit_kv_headroom_consumer():
    coordinator = UnifiedMemoryCoordinator(config())
    plan = coordinator.plan(sample(), safe=True, now_s=10.0)
    session = FakeConsumer("session_bank", 10 * GIB)
    expert = FakeConsumer("expert_residency", 4 * GIB)
    kv = FakeConsumer("kv_headroom", 2 * GIB)
    receipt = coordinator.apply(plan, [session, expert, kv], safe=True)
    assert receipt.applied is True
    assert kv.budget == plan.kv_headroom_bytes
    assert any(item.consumer == "kv_headroom" for item in receipt.mutations)


def test_inactive_expert_partition_is_redistributed():
    coordinator = UnifiedMemoryCoordinator(config())
    all_partitions = coordinator.plan(sample(), safe=True, now_s=10.0)
    without_expert = coordinator.plan(
        sample(),
        safe=True,
        now_s=11.0,
        active_partitions={"session_bank", "kv_headroom"},
    )
    assert without_expert.expert_budget_bytes == 0
    assert "expert_residency" not in without_expert.active_partitions
    assert (
        without_expert.session_bank_budget_bytes + without_expert.kv_headroom_bytes
        == without_expert.managed_budget_bytes
    )
    assert (
        without_expert.session_bank_budget_bytes
        > all_partitions.session_bank_budget_bytes
    )
