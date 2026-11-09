"""Live dashboard progress for non-streaming requests.

The dashboard watches a request over its own SSE connection, so what it shows
must not depend on whether the client asked for a stream. Progress frames used
to come from the SSE drain loop only: a non-streaming request decoding for
25 s produced 0 progress events and no live TPS line until it completed. These
tests pin that both variants publish the same frames, with the same throttle,
without changing the response the client gets.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from threading import Event

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.dashboard_state import InFlightHandle, ProgressEventGate
from mtplx.server.flight_recorder import FlightRecorder
from mtplx.server.openai import create_app
from mtplx.server.stream_rate import DECODE_PHASE_ANSWER, PhaseRateMeter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_server_openai import _fake_state, _stream_payloads  # noqa: E402

DECODED_TOKENS = 300
# Token milestones the gate always publishes; with the interval rule switched
# off below these are the only frames, so every expectation is exact.
MILESTONE_FRAMES = [1, 32, 64, 128, 256]
TIMING_KEYS = {
    "dashboard_progress_decision_time_s",
    "dashboard_progress_registry_update_time_s",
    "dashboard_progress_rolling_update_time_s",
    "dashboard_progress_bus_publish_time_s",
}


@pytest.fixture(autouse=True)
def _milestones_only(monkeypatch):
    monkeypatch.setattr(ProgressEventGate, "MIN_INTERVAL_S", 3600.0)


def _one_token_per_step(_state, _prompt_ids, **kwargs):
    token_callback = kwargs.get("token_callback")
    tokens = [ord("a") + index % 26 for index in range(DECODED_TOKENS)]
    if token_callback is not None:
        for token in tokens:
            token_callback([token])
    return {
        "text": "".join(chr(token) for token in tokens),
        "tokens": tokens,
        "stats": {
            "generation_mode": kwargs["generation_mode"],
            "mtp_depth": kwargs["depth"],
            "completion_tokens": len(tokens),
        },
        "prompt_tokens": 3,
        "completion_tokens": len(tokens),
        "finish_reason": "length",
    }


def _chat(monkeypatch, state, *, stream: bool):
    events: list[dict] = []
    monkeypatch.setattr(state.dashboard.bus, "publish", events.append)
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", _one_token_per_step)
    client = TestClient(create_app(state))
    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Count."}],
            "enable_thinking": False,
            "stream": stream,
            "max_tokens": DECODED_TOKENS,
        },
    )
    assert response.status_code == 200
    progress = [event for event in events if event["kind"] == "progress"]
    return response, progress


def _without_timings(frame: dict) -> dict:
    return {key: value for key, value in frame.items() if key not in TIMING_KEYS}


def test_non_streaming_chat_publishes_live_progress(monkeypatch):
    state = _fake_state()

    response, progress = _chat(monkeypatch, state, stream=False)

    response_id = response.json()["id"]
    published_tokens = [event["progress"]["completion_tokens"] for event in progress]
    assert published_tokens == MILESTONE_FRAMES
    assert {event["request_id"] for event in progress} == {response_id}
    last = progress[-1]["progress"]
    assert last["dashboard_progress_published"] is True
    assert last["decode_tok_s"] > 0
    assert "phase_tok_s" in last
    live_history = state.dashboard.rolling.snapshot()["live_history"]
    assert live_history and live_history[-1]["session_id"] == response_id


def test_non_streaming_progress_matches_streaming(monkeypatch):
    stream_response, stream_progress = _chat(monkeypatch, _fake_state(), stream=True)
    _response, nonstream_progress = _chat(monkeypatch, _fake_state(), stream=False)

    def tokens(events):
        return [event["progress"]["completion_tokens"] for event in events]

    assert tokens(nonstream_progress) == tokens(stream_progress) == MILESTONE_FRAMES
    assert [set(event["progress"]) for event in nonstream_progress] == [
        set(event["progress"]) for event in stream_progress
    ]
    # The streaming client still gets each published frame inline.
    client_frames = [
        payload["mtplx_progress"]
        for payload in _stream_payloads(stream_response.text)
        if "mtplx_progress" in payload
        and not payload["mtplx_progress"].get("heartbeat")
    ]
    client_tokens = [frame["completion_tokens"] for frame in client_frames]
    assert client_tokens == MILESTONE_FRAMES


def test_non_streaming_response_is_unchanged_by_dashboard_progress(monkeypatch):
    with_progress, progress = _chat(monkeypatch, _fake_state(), stream=False)
    assert progress

    monkeypatch.setattr(openai, "_dashboard_publish_progress", lambda *_a, **_k: None)
    without_progress, silent = _chat(monkeypatch, _fake_state(), stream=False)
    assert silent == []

    published = with_progress.json()
    baseline = without_progress.json()
    assert set(published) == set(baseline)
    assert published["choices"] == baseline["choices"]
    assert published["usage"] == baseline["usage"]
    assert set(published["mtplx_stats"]) == set(baseline["mtplx_stats"])


def test_non_streaming_request_forgets_its_progress_gate(monkeypatch):
    state = _fake_state()

    response, progress = _chat(monkeypatch, state, stream=False)

    assert progress
    gate = state.dashboard.progress_events
    assert gate.stats_for(response.json()["id"]) == {}
    assert gate._last_publish_s == {}
    assert gate._last_publish_tokens == {}
    assert gate._stats == {}
    assert state.dashboard.in_flight.snapshot() == []


def test_decode_progress_publishes_what_the_stream_loop_published():
    """The frames equal the SSE drain loop's former inline build, batch by
    batch, including the phase the streaming path notes after each batch."""

    helper_state = _fake_state()
    inline_state = _fake_state()
    progress = openai._DashboardDecodeProgress(helper_state, "chatcmpl-same")
    meter = PhaseRateMeter()
    decode_started_s = None
    completion_tokens = 0
    t = 100.0
    for batch in range(1, 120):
        t += 0.02
        count = 1 + batch % 3
        frame = progress.on_tokens(count, t, session_id="sess-same")

        if decode_started_s is None:
            decode_started_s = t
        completion_tokens += count
        meter.observe(count, t)
        payload = openai._stream_progress_payload(
            completion_tokens=completion_tokens,
            decode_started_s=decode_started_s,
            now_s=t,
            phase_meter=meter,
        )
        payload["request_id"] = "chatcmpl-same"
        payload["session_id"] = "sess-same"
        expected = openai._dashboard_publish_progress(
            inline_state, request_id="chatcmpl-same", payload=payload
        )

        if expected is None:
            assert frame is None
        else:
            assert _without_timings(frame) == _without_timings(expected)
        progress.phase_meter.note_phase(DECODE_PHASE_ANSWER, t)
        meter.note_phase(DECODE_PHASE_ANSWER, t)
    assert progress.completion_tokens == completion_tokens
    assert progress.decode_started_s == pytest.approx(100.02)


def test_decode_progress_builds_no_frame_while_throttled(monkeypatch):
    built: list[int] = []
    real_payload = openai._stream_progress_payload

    def counting_payload(**kwargs):
        built.append(kwargs["completion_tokens"])
        return real_payload(**kwargs)

    monkeypatch.setattr(openai, "_stream_progress_payload", counting_payload)
    state = _fake_state()
    progress = openai._DashboardDecodeProgress(state, "chatcmpl-cheap")
    for step in range(1, 31):
        progress.on_tokens(1, 100.0 + step * 0.02, session_id=None)

    assert built == [1]
    stats = state.dashboard.progress_events.stats_for("chatcmpl-cheap")
    assert stats["dashboard_progress_published_events"] == 1
    assert stats["dashboard_progress_throttled_events"] == 29


def test_decode_progress_marks_decode_start_once_with_the_prefill_state():
    class RecordingFlight(FlightRecorder):
        def __init__(self) -> None:
            super().__init__(None)
            self.decode_started: list[tuple[str, object]] = []
            self.tokens: list[int] = []

        def note_decode_started(self, request_id, prefill_state):
            self.decode_started.append((request_id, prefill_state))

        def on_tokens(self, request_id, count, timestamp_s):
            self.tokens.append(count)

    state = _fake_state()
    state.flight = RecordingFlight()
    prefill_state = {"tokens_done": 512, "tokens_total": 512}
    state.dashboard.in_flight.register(
        InFlightHandle(
            request_id="chatcmpl-prefilled",
            cancel_event=Event(),
            started_s=time.time(),
            session_id=None,
            model="mtplx-test-model",
            prompt_preview="prompt",
            prompt_tokens=512,
            prefill_state=prefill_state,
        )
    )
    progress = openai._DashboardDecodeProgress(state, "chatcmpl-prefilled")

    progress.on_tokens(3, 10.0, session_id=None)
    progress.on_tokens(0, 10.1, session_id=None)
    progress.on_tokens(2, 10.2, session_id=None)

    assert state.flight.decode_started == [("chatcmpl-prefilled", prefill_state)]
    assert state.flight.tokens == [3, 2]
