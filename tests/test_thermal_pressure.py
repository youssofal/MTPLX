"""Thermal pressure in flight samples and receipts, from a cached sampler.

Thermal pressure is the kernel's throttle verdict (nominal, moderate, heavy,
trapping, sleeping), not the fan state. A background thread reads it every
couple of seconds; flight samples, receipts and /health only read the cache,
so nothing on the decode path ever calls the operating system.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from mtplx import thermal_pressure
from mtplx.thermal_pressure import (
    UNAVAILABLE,
    ThermalPressureSampler,
    notify_thermal_pressure_reader,
)


class _Reader:
    def __init__(self, levels):
        self.levels = list(levels)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        value = self.levels[min(self.calls - 1, len(self.levels) - 1)]
        if isinstance(value, BaseException):
            raise value
        return value


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def process_sampler(monkeypatch):
    """Install a process sampler whose source is a scripted reader."""

    def install(sampler):
        monkeypatch.setattr(thermal_pressure, "_process_sampler", sampler)
        return sampler

    return install


def test_reads_are_served_from_the_cache():
    reader = _Reader([2])
    sampler = ThermalPressureSampler(reader)
    assert sampler.current() == UNAVAILABLE  # nothing sampled yet
    assert sampler.sample_once() == "heavy"
    for _ in range(10_000):
        assert sampler.current() == "heavy"
        sampler.worst_since(0.0)
    assert reader.calls == 1


def test_an_unavailable_source_reports_unavailable_and_starts_no_thread():
    sampler = ThermalPressureSampler(None)
    assert sampler.available is False
    assert sampler.sample_once() == UNAVAILABLE
    assert sampler.start() is False
    assert sampler._thread is None
    assert sampler.current() == UNAVAILABLE
    assert sampler.worst_since(1.0) == UNAVAILABLE
    health = sampler.health_payload()
    assert health["available"] is False and health["level"] == UNAVAILABLE


def test_a_failing_read_keeps_the_last_level_and_is_counted():
    reader = _Reader([1, OSError("notify down"), OSError("notify down")])
    sampler = ThermalPressureSampler(reader)
    assert sampler.sample_once() == "moderate"
    assert sampler.sample_once() == "moderate"
    assert sampler.sample_once() == "moderate"
    health = sampler.health_payload()
    assert health["read_errors"] == 2
    assert "notify down" in health["last_error"]


def test_worst_level_over_a_request_follows_the_transitions():
    clock = _Clock()
    reader = _Reader([0, 2, 0])
    sampler = ThermalPressureSampler(reader, clock=clock)
    for when in (0.0, 10.0, 20.0):
        clock.t = when
        sampler.sample_once()
    # A request spanning the heavy stretch sees it.
    assert sampler.worst_since(5.0) == "heavy"
    # One that started inside it inherits the level in effect.
    assert sampler.worst_since(15.0) == "heavy"
    # One that started after it is back to nominal.
    assert sampler.worst_since(25.0) == "nominal"
    # A window that closed before the change never saw it.
    assert sampler.worst_since(1.0, now_s=8.0) == "nominal"
    # Levels are recorded as changes, not as every read.
    assert [level for _when, level in sampler._changes] == [0, 2, 0]


def test_the_background_thread_samples_and_stops():
    reader = _Reader([0])
    sampler = ThermalPressureSampler(reader, interval_s=0.01)
    assert sampler.start() is True
    deadline = time.time() + 5.0
    while reader.calls < 3 and time.time() < deadline:
        time.sleep(0.01)
    sampler.stop()
    assert reader.calls >= 3
    assert not sampler._thread.is_alive()
    assert sampler.current() == "nominal"


def test_the_receipt_carries_the_thermal_state(process_sampler):
    from test_server_openai import _fake_state
    from mtplx.server import openai

    clock = _Clock(100.0)
    sampler = process_sampler(
        ThermalPressureSampler(_Reader([0, 2, 1]), clock=clock)
    )
    for when in (100.0, 105.0, 110.0):
        clock.t = when
        sampler.sample_once()
    state = _fake_state()
    openai._record_request_metrics(
        state,
        {"request_id": "chatcmpl-hot", "request_received_monotonic_s": 102.0},
    )
    receipt = state.last_metrics[-1]
    assert receipt["thermal_pressure"] == "moderate"
    assert receipt["thermal_pressure_max"] == "heavy"


def test_the_receipt_says_whether_the_prompt_was_read_on_a_throttled_chip(process_sampler):
    from test_server_openai import _fake_state
    from mtplx.server import openai

    # Nominal while the prompt is read (102 to 104), heavy from 105 on: the
    # read was cool even though the request ended hot.
    clock = _Clock(100.0)
    sampler = process_sampler(ThermalPressureSampler(_Reader([0, 2]), clock=clock))
    for when in (100.0, 105.0):
        clock.t = when
        sampler.sample_once()
    state = _fake_state()
    openai._record_request_metrics(
        state,
        {"request_id": "chatcmpl-a", "request_received_monotonic_s": 102.0, "ttft_s": 2.0},
    )
    openai._record_request_metrics(
        state,
        {"request_id": "chatcmpl-b", "request_received_monotonic_s": 103.0, "ttft_s": 4.0},
    )
    cool, hot = state.last_metrics[-2:]
    assert cool["thermal_pressure_max"] == "heavy"
    assert cool["thermal_pressure_prefill_max"] == "nominal"
    assert hot["thermal_pressure_prefill_max"] == "heavy"
    # A request that never reached a first token has no read to judge.
    openai._record_request_metrics(
        state, {"request_id": "chatcmpl-c", "request_received_monotonic_s": 103.0}
    )
    assert "thermal_pressure_prefill_max" not in state.last_metrics[-1]


def test_heavy_and_worse_are_throttling():
    assert [thermal_pressure.is_throttling(level) for level in thermal_pressure.THERMAL_PRESSURE_LEVELS] == [
        False, False, True, True, True,
    ]
    assert not thermal_pressure.is_throttling(UNAVAILABLE)
    assert not thermal_pressure.is_throttling(None)


def test_the_session_prefill_speed_is_weighted_by_tokens_and_counts_throttled_reads(
    process_sampler,
):
    from test_server_openai import _fake_state
    from mtplx.server import openai
    from mtplx.server.dashboard_state import DashboardState

    clock = _Clock(100.0)
    sampler = process_sampler(ThermalPressureSampler(_Reader([0, 2]), clock=clock))
    for when in (100.0, 200.0):
        clock.t = when
        sampler.sample_once()
    state = _fake_state()
    state.dashboard = DashboardState()
    # His 12 November shape: a cold 50K read on a cool chip, then a warm
    # 6,843-token read at 868 tok/s under heavy pressure, and a cached turn
    # whose 19 new tokens are no speed at all.
    reads = [
        {"new_prefill_tokens": 50_000, "prefill_tok_s": 2_200.0, "request_received_monotonic_s": 110.0, "ttft_s": 23.0},
        {"new_prefill_tokens": 6_843, "prefill_tok_s": 868.0, "request_received_monotonic_s": 210.0, "ttft_s": 8.0},
        {"new_prefill_tokens": 19, "prefill_tok_s": 153.0, "request_received_monotonic_s": 220.0, "ttft_s": 0.3},
    ]
    for index, read in enumerate(reads):
        openai._dashboard_record_completion(
            state, envelope={"request_id": f"r{index}", **read}, stats={}
        )
    lifetime = state.dashboard.lifetime.snapshot()
    assert lifetime["prefill_reads_total"] == 2
    assert lifetime["prefill_throttled_reads_total"] == 1
    assert lifetime["prefill_read_tokens_total"] == 56_843
    assert lifetime["prefill_read_peak_tok_s"] == 2_200.0
    weighted = 56_843 / lifetime["prefill_read_seconds_total"]
    assert weighted == pytest.approx(56_843 / (50_000 / 2_200 + 6_843 / 868))
    # Weighted by tokens, the session reads at about 1,860 tok/s: neither the
    # throttled turn's 868 nor a plain mean of the two turns.
    assert 1_800 < weighted < 1_900


def test_the_flight_end_event_keeps_the_receipt_thermal_state(tmp_path):
    from mtplx.server.flight_recorder import FlightRecorder

    recorder = FlightRecorder(str(tmp_path / "flight.jsonl"))
    events: list[dict] = []
    recorder._emit = events.append
    recorder.begin("chatcmpl-e", session_id="s", model="m", prompt_tokens=10, stream=True)
    recorder.end(
        "chatcmpl-e",
        {"thermal_pressure": "moderate", "thermal_pressure_max": "heavy"},
    )
    (end,) = [event for event in events if event["ev"] == "end"]
    assert end["thermal_pressure"] == "moderate"
    assert end["thermal_pressure_max"] == "heavy"


def test_the_receipt_says_unavailable_when_nothing_was_sampled(process_sampler):
    from test_server_openai import _fake_state
    from mtplx.server import openai

    process_sampler(ThermalPressureSampler(None))
    state = _fake_state()
    openai._record_request_metrics(state, {"request_id": "chatcmpl-cold"})
    receipt = state.last_metrics[-1]
    assert receipt["thermal_pressure"] == UNAVAILABLE
    assert receipt["thermal_pressure_max"] == UNAVAILABLE


def test_flight_samples_carry_the_state_and_the_decode_path_never_reads_the_os(
    process_sampler, tmp_path
):
    from mtplx.server.flight_recorder import FlightRecorder

    reader = _Reader([2])
    sampler = process_sampler(ThermalPressureSampler(reader))
    sampler.sample_once()
    recorder = FlightRecorder(str(tmp_path / "flight.jsonl"))
    events: list[dict] = []
    recorder._emit = events.append
    recorder.begin("chatcmpl-f", session_id="s", model="m", prompt_tokens=10, stream=True)
    t0 = 1000.0
    # Ten thousand token batches over three seconds: the stream side of the
    # decode path. Samples come out at most once a second.
    for index in range(10_000):
        recorder.on_tokens("chatcmpl-f", 3, t0 + index * 0.0003)
    sink = recorder.live_depth_sink("chatcmpl-f")
    for _ in range(1_000):  # the model-owner thread's by-depth publisher
        sink({"generated_tokens": 30_000, "accepted_by_depth": [1]})
    samples = [event for event in events if event["ev"] == "s"]
    assert 1 <= len(samples) <= 5
    assert all(sample["thermal"] == "heavy" for sample in samples)
    # The only read of the OS is the sampler's own.
    assert reader.calls == 1


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS notification only")
def test_the_macos_source_reads_a_valid_level():
    reader = notify_thermal_pressure_reader()
    assert reader is not None
    assert 0 <= reader() <= 4


def test_health_reports_thermal_pressure(process_sampler):
    from fastapi.testclient import TestClient
    from test_server_openai import _fake_state
    from mtplx.server.openai import create_app

    sampler = process_sampler(ThermalPressureSampler(_Reader([1])))
    sampler.sample_once()
    payload = TestClient(create_app(_fake_state())).get("/health").json()
    assert payload["thermal_pressure"]["level"] == "moderate"
    assert payload["thermal_pressure"]["available"] is True
