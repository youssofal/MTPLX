from __future__ import annotations

import pytest

import mtplx.expert_locality as locality
from mtplx.expert_locality import (
    ExpertLocalityTracker,
    expert_locality_lane,
    record_expert_routes,
    reset_expert_locality_tracker,
)


class ExplodingArray:
    def tolist(self):
        raise AssertionError(
            "disabled instrumentation must not materialize router output"
        )


def test_disabled_global_helper_does_not_touch_lazy_array(monkeypatch):
    monkeypatch.delenv("MTPLX_EXPERT_LOCALITY", raising=False)
    reset_expert_locality_tracker()
    assert record_expert_routes(ExplodingArray(), layer_id=1) is False


def test_hot_workload_produces_small_working_set_and_high_lru_hit_rate():
    tracker = ExpertLocalityTracker(max_events=1000, cache_capacities=(2, 4, 8))
    for _ in range(100):
        tracker.record([[1, 2], [1, 2]], layer_id=4, lane="decode", num_experts=16)
    layer = tracker.snapshot()["layers"][0]
    assert layer["working_set_90"] == 2
    assert layer["lru_simulation"]["2"]["hit_rate"] > 0.95
    assert tracker.recommended_capacity(minimum_hit_rate=0.90, lane="decode") == 2


def test_uniform_workload_exposes_large_working_set():
    tracker = ExpertLocalityTracker(max_events=1000, cache_capacities=(2, 4, 8))
    for token in range(80):
        tracker.record(
            [[(2 * token) % 16, (2 * token + 1) % 16]],
            layer_id=7,
            lane="decode",
            num_experts=16,
        )
    layer = tracker.snapshot()["layers"][0]
    assert layer["working_set_90"] >= 14
    assert layer["lru_simulation"]["2"]["hit_rate"] < 0.1
    assert tracker.recommended_capacity(minimum_hit_rate=0.80) is None


def test_alternating_disjoint_routes_have_zero_consecutive_overlap():
    tracker = ExpertLocalityTracker(cache_capacities=(2, 4))
    for token in range(20):
        row = [0, 1] if token % 2 == 0 else [2, 3]
        tracker.record([row], layer_id="moe-1", lane="decode", num_experts=8)
    layer = tracker.snapshot()["layers"][0]
    # The first row is compared against the empty initial set. Every later
    # transition is disjoint, so the aggregate remains exactly zero.
    assert layer["consecutive_jaccard"] == 0.0


def test_decode_and_mtp_verify_lanes_are_separate():
    tracker = ExpertLocalityTracker(cache_capacities=(2, 4))
    for _ in range(10):
        tracker.record([[1, 2]], layer_id=3, lane="decode", num_experts=8)
        tracker.record([[5, 6]], layer_id=3, lane="mtp_verify", num_experts=8)
    rows = tracker.snapshot()["layers"]
    assert {(row["layer_id"], row["lane"]) for row in rows} == {
        ("3", "decode"),
        ("3", "mtp_verify"),
    }
    by_lane = {row["lane"]: row for row in rows}
    assert by_lane["decode"]["top_experts"][0][0] in {1, 2}
    assert by_lane["mtp_verify"]["top_experts"][0][0] in {5, 6}


def test_context_lane_is_used_by_global_helper(monkeypatch):
    monkeypatch.setenv("MTPLX_EXPERT_LOCALITY", "1")
    monkeypatch.setenv("MTPLX_EXPERT_LOCALITY_CACHE_SIZES", "2,4")
    monkeypatch.setenv("MTPLX_EXPERT_LOCALITY_SAMPLE_EVERY", "1")
    reset_expert_locality_tracker()
    with expert_locality_lane("prefill"):
        assert record_expert_routes([[1, 2]], layer_id=9, num_experts=16)
    snapshot = locality.get_expert_locality_tracker().snapshot()
    assert snapshot["layers"][0]["lane"] == "prefill"


def test_event_budget_is_bounded():
    tracker = ExpertLocalityTracker(max_events=3, cache_capacities=(2,))
    accepted = [
        tracker.record([[index]], layer_id=1, lane="decode", num_experts=32)
        for index in range(10)
    ]
    snapshot = tracker.snapshot()
    assert accepted.count(True) == 3
    assert snapshot["accepted_calls"] == 3
    assert snapshot["dropped_calls"] == 7


def test_sampling_reduces_materialization_frequency():
    tracker = ExpertLocalityTracker(
        max_events=10, sample_every=3, cache_capacities=(2,)
    )
    accepted = [
        tracker.record([[index]], layer_id=1, lane="decode", num_experts=32)
        for index in range(9)
    ]
    assert accepted == [False, False, True, False, False, True, False, False, True]


def test_global_tracker_defaults_to_one_in_sixteen_sampling(monkeypatch):
    monkeypatch.setenv("MTPLX_EXPERT_LOCALITY", "1")
    monkeypatch.delenv("MTPLX_EXPERT_LOCALITY_SAMPLE_EVERY", raising=False)
    reset_expert_locality_tracker()
    accepted = [
        record_expert_routes([[index % 8]], layer_id=1, num_experts=8)
        for index in range(16)
    ]
    assert accepted == ([False] * 15) + [True]
    assert locality.get_expert_locality_tracker().snapshot()["sample_every"] == 16


def test_invalid_experts_are_counted_but_never_enter_cache_simulation():
    tracker = ExpertLocalityTracker(cache_capacities=(2,))
    tracker.record([[-1, 1, 99]], layer_id=1, lane="decode", num_experts=8)
    layer = tracker.snapshot()["layers"][0]
    assert layer["invalid_assignments"] == 2
    assert layer["assignments"] == 1
    assert layer["unique_experts"] == 1


def test_reuse_distance_histogram_tracks_event_distance():
    tracker = ExpertLocalityTracker(cache_capacities=(2,))
    tracker.record([[1]], layer_id=1, lane="decode")
    tracker.record([[2]], layer_id=1, lane="decode")
    tracker.record([[1]], layer_id=1, lane="decode")
    layer = tracker.snapshot()["layers"][0]
    assert layer["reuse_distance_events"]["2-3"] == 1


@pytest.mark.parametrize(
    ("name", "rows", "expected_min_ws90", "expected_max_ws90"),
    [
        ("single-hot", [[1, 1]] * 32, 1, 1),
        ("two-hot", [[1, 2]] * 32, 2, 2),
        ("rotating-8", [[i % 8, (i + 1) % 8] for i in range(64)], 7, 8),
        ("rotating-32", [[i % 32, (i + 1) % 32] for i in range(128)], 28, 32),
    ],
)
def test_routing_workload_matrix(name, rows, expected_min_ws90, expected_max_ws90):
    tracker = ExpertLocalityTracker(max_events=1000, cache_capacities=(4, 8, 16, 32))
    for row in rows:
        tracker.record([row], layer_id=name, lane="decode", num_experts=64)
    layer = tracker.snapshot()["layers"][0]
    assert expected_min_ws90 <= layer["working_set_90"] <= expected_max_ws90


def test_reset_clears_all_state():
    tracker = ExpertLocalityTracker(cache_capacities=(2,))
    tracker.record([[1, 2]], layer_id=1, lane="decode")
    tracker.reset()
    assert tracker.snapshot()["layers"] == []
    assert tracker.snapshot()["calls"] == 0


class _FakeSwitch:
    def __init__(self):
        self.calls = 0

    def __call__(self, hidden, indices):
        self.calls += 1
        return hidden, indices


class _FakeBlock:
    num_experts = 8

    def __init__(self):
        self.switch_mlp = _FakeSwitch()


class _FakeModel:
    def __init__(self):
        self.block = _FakeBlock()

    def named_modules(self):
        return [
            ("", self),
            ("layers.0", self.block),
            ("layers.0.switch_mlp", self.block.switch_mlp),
        ]


def test_installer_is_disabled_without_touching_model(monkeypatch):
    monkeypatch.delenv("MTPLX_EXPERT_LOCALITY", raising=False)
    model = _FakeModel()
    original = model.block.switch_mlp.__class__
    report = locality.install_expert_locality_instrumentation(model)
    assert report["enabled"] is False
    assert report["installed"] is False
    assert model.block.switch_mlp.__class__ is original


def test_installer_records_exact_router_indices_and_is_idempotent(monkeypatch):
    monkeypatch.setenv("MTPLX_EXPERT_LOCALITY", "1")
    monkeypatch.setenv("MTPLX_EXPERT_LOCALITY_CACHE_SIZES", "2,4")
    monkeypatch.setenv("MTPLX_EXPERT_LOCALITY_SAMPLE_EVERY", "1")
    locality.reset_expert_locality_tracker()
    model = _FakeModel()
    report = locality.install_expert_locality_instrumentation(model)
    assert report["installed"] is True
    assert report["instrumented_modules"] == 1
    with locality.expert_locality_lane("decode"):
        output = model.block.switch_mlp("hidden", [[1, 2]])
    assert output == ("hidden", [[1, 2]])
    assert model.block.switch_mlp.calls == 1
    snapshot = locality.expert_locality_metrics()
    assert snapshot["layers"][0]["layer_id"] == "layers.0"
    assert snapshot["layers"][0]["unique_experts"] == 2
    second = locality.install_expert_locality_instrumentation(model)
    assert second["modules"][0]["status"] == "already_instrumented"
