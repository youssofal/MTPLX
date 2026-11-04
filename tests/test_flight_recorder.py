"""Flight recorder contracts: lifecycle events, sampling, live endpoint,
receipt-sink integration, and the 1 Hz live decode sink at its exact
_DecodeTrace call site (the E5 lesson: telemetry ships with a unit test at
the emitting seam, or it silently never fires)."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient

from mtplx.server.flight_recorder import FlightRecorder, resolve_flight_recorder

from test_server_openai import _fake_state  # noqa: E402


def _read_events(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _wait_for_writer(path, kinds, timeout_s=3.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            events = _read_events(path)
            if [e["ev"] for e in events] == kinds:
                return events
        time.sleep(0.05)
    raise AssertionError(
        f"writer never produced {kinds}; have "
        f"{[e['ev'] for e in _read_events(path)] if os.path.exists(path) else 'no file'}"
    )


def test_lifecycle_events_and_sampling(tmp_path):
    path = str(tmp_path / "flight-9999.jsonl")
    recorder = FlightRecorder(path, text_mode="abnormal")
    recorder.begin(
        "chatcmpl-t1", session_id="ses_a", model="m", prompt_tokens=100, stream=True
    )
    recorder.on_delta("chatcmpl-t1", "reasoning_content", "thinking about rooks")
    recorder.on_delta("chatcmpl-t1", "content", "done")
    t0 = time.perf_counter()
    recorder.note_decode_started(
        "chatcmpl-t1",
        {"tokens_done": 90, "tokens_total": 100, "cached_tokens": 60, "elapsed_s": 0.4},
    )
    recorder.on_tokens("chatcmpl-t1", 4, t0)  # immediate first sample
    recorder.on_tokens("chatcmpl-t1", 4, t0 + 0.5)  # inside interval: no sample
    recorder.on_tokens("chatcmpl-t1", 4, t0 + 1.4)  # second sample
    sink = recorder.live_depth_sink("chatcmpl-t1")
    sink({"accepted_by_depth": [3, 2, 1], "drafted_by_depth": [4, 4, 4]})
    recorder.on_tokens("chatcmpl-t1", 4, t0 + 2.6)  # third sample carries acc

    snapshot = recorder.snapshot()
    (active,) = snapshot["active"]
    assert active["phase"] == "decode"
    assert active["gen_tokens"] == 16
    assert active["reasoning_chars"] == len("thinking about rooks")
    assert active["content_chars"] == len("done")
    assert active["accepted_by_depth"] == [3, 2, 1]
    assert "rooks" in active["tail"]
    assert active["prefill"]["cached_tokens"] == 60

    recorder.end(
        "chatcmpl-t1",
        {
            "request_id": "chatcmpl-t1",
            "request_cancelled": True,
            "cancellation_reason": "client_disconnected",
            "completion_tokens": 16,
        },
    )
    events = _wait_for_writer(path, ["begin", "prefill", "s", "s", "s", "end"])
    begin, prefill, s1, _s2, s3, end = events
    assert begin["session_id"] == "ses_a" and begin["prompt_tokens"] == 100
    assert prefill["cached_tokens"] == 60
    assert s1["gen"] == 4 and "acc" not in s1
    assert s3["acc"] == [3, 2, 1] and s3["ctx"] == 100 + 16
    assert end["cancelled"] is True and end["gen"] == 16
    # Cancelled request persists its generated text (the class the client
    # zeroes out) ...
    assert "text_path" in end
    deadline = time.time() + 3.0
    while not os.path.exists(end["text_path"]) and time.time() < deadline:
        time.sleep(0.05)
    with open(end["text_path"], encoding="utf-8") as handle:
        assert "rooks" in handle.read()
    # ... and the registry is drained.
    assert recorder.snapshot()["active"] == []
    assert recorder.snapshot()["recent"][0]["rid"] == "chatcmpl-t1"


def test_normal_end_keeps_text_off_disk_in_abnormal_mode(tmp_path):
    path = str(tmp_path / "flight-9998.jsonl")
    recorder = FlightRecorder(path, text_mode="abnormal")
    recorder.begin("r", session_id=None, model=None, prompt_tokens=1, stream=True)
    recorder.on_delta("r", "content", "hello")
    recorder.end("r", {"request_id": "r", "finish_reason": "stop"})
    events = _wait_for_writer(path, ["begin", "end"])
    assert "text_path" not in events[-1]
    assert events[-1]["reason"] == "stop" and events[-1]["cancelled"] is False


def test_end_carries_the_waits_before_the_request_ran(tmp_path):
    """begin is written after the wait for the previous turn's commit, so the
    end event carries every wait span of the TTFT breakdown."""
    path = str(tmp_path / "flight-9994.jsonl")
    recorder = FlightRecorder(path, text_mode="off")
    recorder.begin("w", session_id="ses_w", model=None, prompt_tokens=9, stream=True)
    recorder.end(
        "w",
        {
            "request_id": "w",
            "ttft_s": 140.2,
            "ttft_spans": {
                "exclusive_s": {
                    "encode_s": 0.01,
                    "response_tail_wait_s": 135.0004,
                    "postcommit_wait_s": 0.002,
                    "scheduler_queue_s": 4.5,
                    "lock_wait_s": 0.1,
                    "engine_first_token_s": 0.6,
                }
            },
        },
    )
    end = _wait_for_writer(path, ["begin", "end"])[-1]
    assert end["response_tail_wait_s"] == 135.0
    assert end["postcommit_wait_s"] == 0.002
    assert end["scheduler_queue_s"] == 4.5
    assert end["lock_wait_s"] == 0.1
    assert "encode_s" not in end and "engine_first_token_s" not in end


def test_end_without_spans_adds_no_wait_fields(tmp_path):
    path = str(tmp_path / "flight-9993.jsonl")
    recorder = FlightRecorder(path, text_mode="off")
    recorder.begin("n", session_id=None, model=None, prompt_tokens=1, stream=False)
    recorder.end(
        "n",
        {"request_id": "n", "ttft_spans": {"exclusive_s": {"scheduler_queue_s": None}}},
    )
    end = _wait_for_writer(path, ["begin", "end"])[-1]
    assert not any(key.endswith("_wait_s") or key == "scheduler_queue_s" for key in end)


def test_sweep_writes_orphan_end_once(tmp_path):
    path = str(tmp_path / "flight-9997.jsonl")
    recorder = FlightRecorder(path, text_mode="off")
    recorder.begin("orph", session_id=None, model=None, prompt_tokens=5, stream=True)
    recorder.sweep("orph")
    recorder.sweep("orph")  # second sweep is a no-op
    events = _wait_for_writer(path, ["begin", "end"])
    assert events[-1]["reason"] == "orphaned" and events[-1]["cancelled"] is True


def test_inert_recorder_is_total_noop():
    recorder = FlightRecorder(None)
    recorder.begin("x", session_id=None, model=None, prompt_tokens=1, stream=True)
    recorder.on_delta("x", "content", "y")
    recorder.on_tokens("x", 1, 0.0)
    recorder.end("x", {"request_id": "x"})
    recorder.pc("s", {"stored": True})
    assert recorder.live_depth_sink("x") is None
    snapshot = recorder.snapshot()
    assert snapshot["enabled"] is False and snapshot["active"] == []


def test_pc_event_shape(tmp_path):
    path = str(tmp_path / "flight-9996.jsonl")
    recorder = FlightRecorder(path, text_mode="off")
    recorder.pc(
        "ses_b",
        {
            "stored": False,
            "mode": "abandoned_foreground_busy",
            "reason": "foreground_preempted_postcommit",
            "retry_scheduled": True,
            "ignored_key": "dropped",
        },
    )
    events = _wait_for_writer(path, ["pc"])
    (pc,) = events
    assert pc["session_id"] == "ses_b"
    assert pc["stored"] is False
    assert pc["mode"] == "abandoned_foreground_busy"
    assert pc["retry_scheduled"] is True
    assert "ignored_key" not in pc


def test_resolve_flight_recorder_matrix(tmp_path, monkeypatch):
    class Args:
        flight_recorder = None
        port = 8123

    monkeypatch.delenv("MTPLX_FLIGHT_RECORDER", raising=False)
    monkeypatch.delenv("MTPLX_FLIGHT_TEXT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    default = resolve_flight_recorder(Args())
    assert default.enabled
    assert default.path.endswith("metrics/flight-8123.jsonl")
    assert default.text_mode == "abnormal"

    monkeypatch.setenv("MTPLX_FLIGHT_RECORDER", "off")
    assert resolve_flight_recorder(Args()).enabled is False

    monkeypatch.setenv("MTPLX_FLIGHT_RECORDER", str(tmp_path / "custom.jsonl"))
    monkeypatch.setenv("MTPLX_FLIGHT_TEXT", "always")
    custom = resolve_flight_recorder(Args())
    assert custom.path == str(tmp_path / "custom.jsonl")
    assert custom.text_mode == "always"

    class ArgOverride:
        flight_recorder = "off"
        port = 8123

    monkeypatch.delenv("MTPLX_FLIGHT_RECORDER", raising=False)
    assert resolve_flight_recorder(ArgOverride()).enabled is False


def test_receipt_sink_emits_flight_end(tmp_path):
    """_record_request_metrics is the single terminal funnel: a receipt with a
    request_id must close the flight record even when the stream never did."""
    from mtplx.server import openai as server_openai

    state = _fake_state()
    path = str(tmp_path / "flight-9995.jsonl")
    state.flight = FlightRecorder(path, text_mode="off")
    state.flight.begin(
        "chatcmpl-sink", session_id="ses_c", model="m", prompt_tokens=7, stream=True
    )
    server_openai._record_request_metrics(
        state,
        {
            "request_id": "chatcmpl-sink",
            "completion_tokens": 3,
            "cached_tokens": 5,
            "finish_reason": "stop",
            "ttft_spans": {"exclusive_s": {"response_tail_wait_s": 12.5}},
        },
    )
    events = _wait_for_writer(path, ["begin", "end"])
    assert events[-1]["completion_tokens"] == 3 and events[-1]["cached_tokens"] == 5
    assert events[-1]["response_tail_wait_s"] == 12.5
    assert state.flight.snapshot()["active"] == []


def test_flight_endpoint_serves_snapshot(tmp_path):
    from mtplx.server.openai import create_app

    state = _fake_state()
    path = str(tmp_path / "flight-9994.jsonl")
    state.flight = FlightRecorder(path, text_mode="off")
    state.flight.begin(
        "chatcmpl-live", session_id="ses_d", model="m", prompt_tokens=9, stream=True
    )
    client = TestClient(create_app(state))
    payload = client.get("/v1/mtplx/flight").json()
    assert payload["enabled"] is True
    assert payload["active"][0]["rid"] == "chatcmpl-live"
    assert payload["active"][0]["phase"] == "prefill"


def test_flight_endpoint_inert_on_stub_state():
    from mtplx.server.openai import create_app

    state = _fake_state()
    if hasattr(state, "flight"):
        state.flight = FlightRecorder(None)
    client = TestClient(create_app(state))
    payload = client.get("/v1/mtplx/flight").json()
    assert payload["enabled"] is False and payload["active"] == []


def test_decode_trace_live_sink_publishes_at_call_site(monkeypatch):
    """The exact seam: _DecodeTrace.maybe_emit must publish by-depth totals to
    the installed sink even with file tracing disabled, throttle to ~1 Hz,
    honor force/final, and disarm a raising sink without propagating."""
    monkeypatch.delenv("MTPLX_DECODE_TRACE_JSONL", raising=False)
    from mtplx import generation as generation_mod

    received = []
    generation_mod.set_live_decode_sink(lambda payload: received.append(payload))
    try:
        trace = generation_mod._DecodeTrace(
            prompt_tokens=10,
            max_tokens=100,
            speculative_depth=3,
            sampler=None,
            verify_strategy="joint",
            verify_core="fused",
            mtp_history_policy="managed",
            mtp_cache_policy="paged",
            trace_label=None,
            trace_metadata=None,
        )
        assert trace.enabled is False  # file trace off; live sink still active
        totals = {
            "generated_tokens": 12,
            "accepted_by_depth": [5, 3, 1],
            "drafted_by_depth": [6, 6, 6],
            "verify_calls": 7,
        }
        kwargs = dict(
            cache=None,
            mtp_cache=None,
            mtp_history_materialize_every=0,
            mtp_history_materialize_events=0,
        )
        trace.maybe_emit(force=False, final=False, totals=totals, **kwargs)
        assert len(received) == 1  # first publish immediate
        assert received[0]["accepted_by_depth"] == [5, 3, 1]
        trace.maybe_emit(force=False, final=False, totals=totals, **kwargs)
        assert len(received) == 1  # throttled inside the 1s window
        trace.maybe_emit(force=True, final=False, totals=totals, **kwargs)
        assert len(received) == 2  # force bypasses the throttle

        def boom(_payload):
            raise RuntimeError("sink broke")

        trace.live_sink = boom
        trace.maybe_emit(force=True, final=False, totals=totals, **kwargs)
        assert trace.live_sink is None  # disarmed, decode untouched
    finally:
        generation_mod.set_live_decode_sink(None)


def test_decode_trace_without_sink_costs_nothing(monkeypatch):
    monkeypatch.delenv("MTPLX_DECODE_TRACE_JSONL", raising=False)
    from mtplx import generation as generation_mod

    generation_mod.set_live_decode_sink(None)
    trace = generation_mod._DecodeTrace(
        prompt_tokens=1,
        max_tokens=1,
        speculative_depth=2,
        sampler=None,
        verify_strategy="joint",
        verify_core="fused",
        mtp_history_policy="managed",
        mtp_cache_policy="paged",
        trace_label=None,
        trace_metadata=None,
    )
    assert trace.live_sink is None
    trace.maybe_emit(
        force=True,
        final=False,
        totals={"generated_tokens": 1},
        cache=None,
        mtp_cache=None,
        mtp_history_materialize_every=0,
        mtp_history_materialize_events=0,
    )  # no sink, file trace off: returns without touching anything


def test_non_streaming_requests_get_sink_driven_samples(tmp_path):
    """Non-streaming requests never enter the SSE drain (on_tokens never
    fires) — found 2026-08-22 when an Ivan-harness arm left 75 requests with
    begin/end and zero samples. The generation-side depth sink now emits the
    file sample itself when the stream side hasn't."""
    path = str(tmp_path / "flight-9998.jsonl")
    recorder = FlightRecorder(path, text_mode="off")
    recorder.begin(
        "chatcmpl-ns", session_id="ses_ns", model="m", prompt_tokens=500, stream=False
    )
    sink = recorder.live_depth_sink("chatcmpl-ns")
    assert sink is not None
    sink(
        {
            "generated_tokens": 40,
            "accepted_by_depth": [10, 6, 2],
            "drafted_by_depth": [16, 16, 16],
            "verify_time_s": 0.8,
            "draft_time_s": 0.1,
        }
    )
    recorder.end("chatcmpl-ns", {"request_id": "chatcmpl-ns", "completion_tokens": 40})
    assert _wait_for_writer(path, ["begin", "s", "end"])
    events = _read_events(path)
    sample = next(e for e in events if e["ev"] == "s")
    assert sample["gen"] == 40
    assert sample["acc"] == [10, 6, 2]
    assert sample["drf"] == [16, 16, 16]
    assert sample["ctx"] == 540
    assert "tps" not in sample, "no stream window -> no fabricated rate"


def test_non_stream_token_callback_feeds_the_flight_recorder():
    """A non-streaming chat request must drive the recorder's token hook the
    way the SSE drain loop does, or /v1/mtplx/flight reports it as a
    never-ending prefill."""
    import inspect
    import re

    from mtplx.server import openai as server

    source = inspect.getsource(server)
    start = source.index("def _nonstream_on_tokens(")
    body = source[start : source.index("def adopt_forked_session(", start)]
    assert re.search(r"_flight\(state\)\.on_tokens\(\s*response_id,", body)
