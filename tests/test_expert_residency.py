from __future__ import annotations

from mtplx.expert_residency import (
    ExpertRef,
    ExpertResidencyConfig,
    ExpertResidencyController,
)


class FakeBackend:
    mode = "test_residency"

    def __init__(self, sizes: dict[ExpertRef, int], resident=()):
        self.sizes = sizes
        self.resident = set(resident)
        self.prefetch_calls = []
        self.evict_calls = []

    def resident_experts(self):
        return tuple(sorted(self.resident))

    def expert_nbytes(self, expert):
        return self.sizes[ExpertRef.coerce(expert)]

    def prefetch_experts(self, experts):
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self.prefetch_calls.append(values)
        self.resident.update(values)
        return values

    def evict_experts(self, experts):
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self.evict_calls.append(values)
        completed = tuple(item for item in values if item in self.resident)
        self.resident.difference_update(completed)
        return completed


def configured(**kwargs):
    values = {
        "enabled": True,
        "budget_bytes": 200,
        "minimum_observations": 1,
        "minimum_tick_interval_s": 0.0,
        "maximum_prefetch_per_tick": 8,
        "maximum_evict_per_tick": 8,
    }
    values.update(kwargs)
    return ExpertResidencyController(ExpertResidencyConfig(**values))


def test_plan_selects_highest_decayed_value_per_byte():
    refs = [ExpertRef(0, 0), ExpertRef(0, 1), ExpertRef(0, 2)]
    backend = FakeBackend({refs[0]: 100, refs[1]: 100, refs[2]: 100})
    controller = configured()
    controller.observe(0, [0], weights=[10], now_s=1.0)
    controller.observe(0, [1], weights=[8], now_s=1.0)
    controller.observe(0, [2], weights=[1], now_s=1.0)

    plan = controller.plan(backend, now_s=1.1)

    assert plan.eligible is True
    assert plan.target == (refs[0], refs[1])
    assert plan.prefetch == (refs[0], refs[1])
    assert plan.target_bytes == 200


def test_plan_keeps_resident_expert_with_hysteresis_and_evicts_cold_one():
    hot, warm, cold = ExpertRef(1, 1), ExpertRef(1, 2), ExpertRef(1, 3)
    backend = FakeBackend(
        {hot: 100, warm: 100, cold: 100},
        resident=(warm, cold),
    )
    controller = configured(hysteresis_ratio=0.25)
    controller.observe_refs([hot] * 10 + [warm] * 8 + [cold])

    plan = controller.plan(backend)

    assert hot in plan.target
    assert warm in plan.target
    assert cold in plan.evict


def test_apply_requires_safe_point_and_reports_backend_mode():
    ref = ExpertRef(2, 4)
    backend = FakeBackend({ref: 64})
    controller = configured(budget_bytes=64)
    controller.observe_refs([ref])
    plan = controller.plan(backend)

    blocked = controller.apply(plan, backend, safe=False)
    assert blocked.applied is False
    assert blocked.reason == "unsafe_point"
    assert backend.prefetch_calls == []

    receipt = controller.apply(plan, backend, safe=True)
    assert receipt.applied is True
    assert receipt.prefetched == (ref,)
    assert receipt.backend_mode == "test_residency"
    assert receipt.duration_ms >= 0


def test_timeout_interval_is_ineligible_without_changing_last_plan():
    ref = ExpertRef(0, 0)
    backend = FakeBackend({ref: 10})
    controller = configured(minimum_tick_interval_s=60.0)
    controller.observe_refs([ref])
    first = controller.plan(backend, now_s=100.0)
    second = controller.plan(backend, now_s=100.1)
    assert first.eligible is True
    assert second.eligible is False
    assert second.reason == "tick_interval"


def test_ingest_locality_snapshot_accepts_rows_and_keyed_counts():
    controller = configured()
    assert (
        controller.ingest_locality_snapshot(
            {
                "experts": [
                    {"layer": 1, "expert": 2, "count": 4},
                    {"layer": 3, "expert": 5, "weight": 2},
                ]
            }
        )
        == 2
    )
    assert controller.ingest_locality_snapshot({"expert_counts": {"4:7": 3}}) == 1
    snapshot = controller.snapshot()
    assert snapshot["tracked_experts"] == 3
    assert snapshot["router_mutation"] is False


def test_disabled_controller_never_applies():
    ref = ExpertRef(0, 0)
    backend = FakeBackend({ref: 10})
    controller = ExpertResidencyController(
        ExpertResidencyConfig(enabled=False, minimum_tick_interval_s=0)
    )
    controller.observe_refs([ref] * 20)
    plan = controller.plan(backend)
    receipt = controller.apply(plan, backend, safe=True)
    assert plan.reason == "disabled"
    assert receipt.applied is False


def test_ingest_phase_one_nested_layer_vectors():
    controller = configured()
    accepted = controller.ingest_locality_snapshot(
        {
            "layers": [
                {"layer_id": 2, "expert_counts": [0, 4, 1]},
                {
                    "layer": 3,
                    "experts": [
                        {"expert_id": 5, "observations": 7},
                        {"expert": 6, "frequency": 2},
                    ],
                },
            ]
        }
    )
    assert accepted == 4
    assert controller.snapshot()["tracked_experts"] == 4


class PartialBackend(FakeBackend):
    def __init__(self, sizes, resident=(), failed=()):
        super().__init__(sizes, resident=resident)
        self.failed = set(failed)

    def prefetch_experts(self, experts):
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self.prefetch_calls.append(values)
        completed = tuple(item for item in values if item not in self.failed)
        self.resident.update(completed)
        return completed


def test_failed_prefetch_does_not_evict_every_replacement_candidate():
    hot_a = ExpertRef(0, 0)
    hot_b = ExpertRef(0, 1)
    cold_a = ExpertRef(0, 2)
    cold_b = ExpertRef(0, 3)
    backend = PartialBackend(
        {item: 100 for item in (hot_a, hot_b, cold_a, cold_b)},
        resident=(cold_a, cold_b),
        failed=(hot_b,),
    )
    controller = configured(budget_bytes=200)
    controller.observe_refs([hot_a] * 10 + [hot_b] * 9)
    plan = controller.plan(backend)
    assert len(plan.prefetch) == 2
    assert len(plan.evict) == 2

    receipt = controller.apply(plan, backend, safe=True)

    assert receipt.prefetched == (hot_a,)
    assert receipt.failed == (hot_b,)
    assert len(receipt.evicted) == 1
    assert len(backend.resident) == 2
