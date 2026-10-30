"""The TTFT span instrument: one clock from HTTP arrival, named spans that add
up to the time to first token, the owner-thread queue receipt, and the attempt
receipt on every response."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.server import openai, request_spans
from mtplx.server.openai import create_app

from test_server_openai import _fake_streaming_session_state


@pytest.fixture(autouse=True)
def _fresh_registry():
    request_spans.reset_for_tests()
    yield
    request_spans.reset_for_tests()


def test_spans_add_up_to_the_first_token_by_construction():
    clock = request_spans.RequestClock(100.0)
    clock.mark("http_parse", 100.004)
    clock.mark("policy", 100.010)
    clock.mark("encode", 100.050)
    clock.mark("postcommit_wait", 100.650)
    clock.mark("dispatch", 100.700)
    clock.mark("scheduler_queue", 101.900)
    clock.mark("engine_first_token", 102.500)
    clock.mark("first_delta_sent", 102.502)

    summary = clock.summary()

    assert summary["endpoint"] == "first_delta_sent"
    assert summary["ttft_s"] == pytest.approx(2.502)
    assert summary["sum_s"] == pytest.approx(summary["ttft_s"], abs=1e-5)
    assert list(summary["exclusive_s"]) == [
        "http_parse_s",
        "policy_s",
        "encode_s",
        "postcommit_wait_s",
        "dispatch_s",
        "scheduler_queue_s",
        "engine_first_token_s",
        "first_delta_sent_s",
    ]
    assert summary["exclusive_s"]["scheduler_queue_s"] == pytest.approx(1.2)
    assert summary["attempts"] == 1
    assert summary["discarded_attempt_wall_s"] == 0.0


def test_first_mark_of_a_name_wins():
    clock = request_spans.RequestClock(0.0)
    clock.mark("postcommit_wait", 1.0)
    clock.mark("postcommit_wait", 5.0)
    clock.mark("engine_first_token", 6.0)

    assert clock.summary()["exclusive_s"]["postcommit_wait_s"] == pytest.approx(1.0)


def test_discarded_attempt_before_any_delta_is_its_own_span():
    clock = request_spans.RequestClock(0.0)
    clock.mark("prologue", 0.1)
    clock.mark("dispatch", 0.2)
    clock.mark("scheduler_queue", 0.3)
    clock.mark("engine_first_token", 1.0)
    clock.discard_attempt("chat.stream.tool_fed_empty_retry", now=4.0)
    clock.mark("scheduler_queue", 4.1)
    clock.mark("engine_first_token", 4.6)
    clock.mark("first_delta_sent", 4.61)

    summary = clock.summary()

    assert summary["attempts"] == 2
    assert summary["retry_paths"] == ["chat.stream.tool_fed_empty_retry"]
    assert summary["discarded_attempt_wall_s"] == pytest.approx(3.8)
    assert summary["exclusive_s"]["discarded_attempt_s"] == pytest.approx(3.8)
    assert summary["exclusive_s"]["scheduler_queue_s"] == pytest.approx(0.1)
    assert summary["ttft_s"] == pytest.approx(4.61)
    assert summary["sum_s"] == pytest.approx(4.61, abs=1e-5)


def test_discard_after_the_client_saw_a_delta_keeps_the_ttft():
    clock = request_spans.RequestClock(0.0)
    clock.mark("dispatch", 0.1)
    clock.mark("engine_first_token", 0.5)
    clock.mark("first_delta_sent", 0.51)
    clock.discard_attempt("chat.stream.tool_fed_empty_retry", now=3.0)

    summary = clock.summary()

    assert summary["ttft_s"] == pytest.approx(0.51)
    assert summary["attempts"] == 2
    assert summary["discarded_attempt_wall_s"] == pytest.approx(2.9)
    assert "discarded_attempt_s" not in summary["exclusive_s"]


def test_health_summary_reports_percentiles_per_span():
    rows = []
    for queue_s in (0.0, 0.0, 0.0, 1.0, 2.0):
        clock = request_spans.RequestClock(0.0)
        clock.mark("scheduler_queue", queue_s)
        clock.mark("engine_first_token", queue_s + 0.5)
        rows.append(clock.summary())

    health = request_spans.health_summary(rows)

    assert health["count"] == 5
    assert health["spans"]["scheduler_queue_s"]["p50"] == pytest.approx(0.0)
    assert health["spans"]["scheduler_queue_s"]["p90"] == pytest.approx(1.6)
    assert health["ttft_p50_s"] == pytest.approx(0.5)


def test_arrival_middleware_stamps_before_the_handler():
    app = FastAPI()
    app.add_middleware(request_spans.RequestArrivalClock)
    seen: dict[str, float] = {}

    @app.post("/probe")
    async def probe(request: Request) -> dict[str, bool]:
        seen["handler"] = time.perf_counter()
        seen["arrival"] = request_spans.arrival_s(request)
        return {"ok": True}

    with TestClient(app) as client:
        assert client.post("/probe", json={"x": 1}).status_code == 200

    assert seen["arrival"] is not None
    assert seen["arrival"] <= seen["handler"]


def test_scheduler_receipt_names_the_item_the_request_waited_behind():
    scheduler = ModelWorkScheduler(idle_grace_s=0.0)
    try:
        release = threading.Event()
        started = threading.Event()

        def idle_history_reprefill() -> None:
            started.set()
            release.wait(5.0)

        scheduler.submit_idle_postcommit(
            idle_history_reprefill, batch_key="postcommit:sess-a"
        )
        assert started.wait(5.0)
        receipts: list[dict] = []
        future = scheduler.submit_foreground(
            lambda: receipts.append(scheduler.active_item_receipt()),
            batch_key="chat.stream",
        )
        time.sleep(0.2)
        release.set()
        future.result(timeout=5.0)
    finally:
        scheduler.shutdown(wait=True)

    receipt = receipts[0]
    assert receipt["running_kind"] == "idle_postcommit"
    assert receipt["running_batch_key"] == "postcommit"
    assert receipt["queue_wait_s"] >= 0.15
    assert future.queue_receipt["queue_wait_s"] == receipt["queue_wait_s"]
    # Off the owner thread there is no active item to describe.
    assert scheduler.active_item_receipt() is None


def _fake_engine(monkeypatch, texts: list[str]):
    calls = {"n": 0}

    def fake_generate_mtpk(_runtime, prompt_ids, **kwargs):
        text = texts[min(calls["n"], len(texts) - 1)]
        calls["n"] += 1
        tokens = [ord(ch) for ch in text]
        callback = kwargs.get("token_callback")
        if callback is not None:
            for token in tokens:
                callback([token])
        return SimpleNamespace(
            tokens=tokens,
            text=text,
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.001,
                    "cache_restore_time_s": 0.0005,
                    "generated_tokens": len(tokens),
                    "elapsed_s": 0.01,
                    "tok_s": 100.0,
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)
    return calls


def _stream_stats(text: str) -> dict:
    stats = None
    for line in text.splitlines():
        if not line.startswith("data: {"):
            continue
        payload = json.loads(line[len("data: ") :])
        if "mtplx_stats" in payload:
            stats = payload["mtplx_stats"]
    assert stats is not None, text
    return stats


def test_streamed_response_carries_spans_that_add_up_to_its_ttft(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    _fake_engine(monkeypatch, ["Hello there"])

    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 16,
            },
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()
        health = client.get("/health").json()

    stats = _stream_stats(body)
    spans = stats["ttft_spans"]
    assert spans["origin"] == "http_arrival"
    assert spans["endpoint"] == "first_delta_sent"
    for name in (
        "http_parse",
        "policy",
        "encode",
        "prologue",
        "dispatch",
        "scheduler_queue",
        "lock_wait",
        "admission",
        "engine_first_token",
        "first_delta_sent",
    ):
        assert f"{name}_s" in spans["exclusive_s"], (name, spans)
    assert spans["sum_s"] == pytest.approx(spans["ttft_s"], abs=1e-4)
    # The replay harness sums exclusive_s against the client's TTFT.
    assert sum(spans["exclusive_s"].values()) == pytest.approx(spans["ttft_s"], abs=1e-4)
    # The engine span is explained by the engine's own receipt.
    assert spans["details"]["engine"]["prompt_eval_time_s"] == pytest.approx(0.001)
    # The server TTFT now starts at arrival, and the stream's first delta
    # can only be written after the first token exists.
    assert stats["ttft_s"] <= spans["ttft_s"] + 1e-6
    assert stats["attempts"] == 1
    assert stats["discarded_attempt_wall_s"] == 0.0
    assert health["ttft_spans"]["count"] >= 1
    assert "scheduler_queue_s" in health["ttft_spans"]["spans"]


def test_nonstream_blank_retry_names_the_discarded_attempt(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    state.args.blank_retry_attempts = 1
    calls = _fake_engine(monkeypatch, ["", "OK"])
    clock = request_spans.open_clock("req-blank", arrival=None, handler_start=time.perf_counter())
    clock.mark("dispatch")

    out = openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=4,
        temperature=0.7,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode="mtp",
        depth=3,
        request_observability={"request_id": "req-blank"},
        streaming_response=False,
    )

    assert calls["n"] == 2
    stats = out["stats"]
    assert stats["attempts"] == 2
    assert stats["retry_path"] == "blank_retry"
    assert stats["discarded_attempt_wall_s"] > 0.0
    assert stats["ttft_spans"]["exclusive_s"]["discarded_attempt_s"] > 0.0


def test_streamed_repair_retry_names_the_discarded_attempt(monkeypatch):
    """Item 3: a stream repair path that throws away its first generation
    (here the tool-fed empty retry: orphaned tool-control markup after tool
    results) says so in the response: two attempts, the retry path, and the
    wall time of the first one."""

    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    texts = [
        "</think>\n\nparameter=limit>\n180\n</parameter>\n</function>\n</tool_call>",
        "</think>\n\nPart 2 has 93 lines.",
    ]
    calls: list[str] = []

    def fake_run_generation(_state, prompt_ids, **kwargs):
        text = texts[len(calls)]
        calls.append(text)
        time.sleep(0.05)
        tokens = [ord(char) for char in text]
        callback = kwargs.get("token_callback")
        if callback is not None:
            for token in tokens:
                callback([token])
        observability = kwargs.get("request_observability") or {}
        stats = {
            **observability,
            "generation_mode": kwargs["generation_mode"],
            "mtp_depth": kwargs["depth"],
            "completion_tokens": len(tokens),
        }
        # What the real generation does when it ends.
        request_spans.publish(
            request_spans.clock_from_observability(observability), stats
        )
        return {
            "text": text,
            "tokens": tokens,
            "stats": stats,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"x-mtplx-cache-mode": "bypass"},
            json={
                "messages": [
                    {"role": "user", "content": "Count the lines of part 2."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path": "notes.txt"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "93 lines"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "Read a file.",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    }
                ],
                "stream": True,
                "max_tokens": 128,
                "enable_thinking": True,
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert len(calls) == 2, calls
    stats = _stream_stats(body)
    assert stats["attempts"] == 2
    assert stats["retry_path"] == "chat.stream.tool_fed_empty_retry"
    assert stats["discarded_attempt_wall_s"] > 0.0


def test_a_tool_call_answer_ends_its_ttft_at_the_tool_call_delta(monkeypatch):
    """E2d: every warm agent turn (a tool call after tool results) reported
    its spans ending at the engine's first token, 0.36 to 0.38 s before the
    client's first delta: with tool results in the history the stream holds
    the call until its markup is whole and writes it outside the content
    loop, where first_delta_sent was marked. The first visible delta of any
    kind now ends the clock."""

    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    state.args.stream_interval = 1
    _fake_engine(
        monkeypatch,
        [
            "<tool_call>\n<function=count_lines>\n<parameter=part>\n2\n"
            "</parameter>\n</function>\n</tool_call>"
        ],
    )

    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Count the lines of each part."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "count_lines",
                                    "arguments": '{"part": 1}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "93 lines"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "count_lines",
                            "description": "Count the lines in one part.",
                            "parameters": {
                                "type": "object",
                                "properties": {"part": {"type": "integer"}},
                                "required": ["part"],
                            },
                        },
                    }
                ],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 64,
            },
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()

    assert '"tool_calls": [' in body
    stats = _stream_stats(body)
    spans = stats["ttft_spans"]
    assert spans["endpoint"] == "first_delta_sent"
    assert "first_delta_sent_s" in spans["exclusive_s"]
    assert spans["sum_s"] == pytest.approx(spans["ttft_s"], abs=1e-4)


@pytest.mark.parametrize(
    "delta, visible",
    [
        ({"role": "assistant"}, False),
        ({}, False),
        ({"content": ""}, False),
        ({"content": "O"}, True),
        ({"reasoning_content": "t"}, True),
        ({"tool_calls": [{"index": 0}]}, True),
    ],
)
def test_the_first_visible_delta_is_what_a_client_counts(delta, visible):
    chunk = (
        'data: {"id": "r", "object": "chat.completion.chunk", "created": 1, '
        '"model": "m", "choices": [{"index": 0, "delta": '
        + json.dumps(delta)
        + ', "finish_reason": null}]}\n\n'
    )
    assert openai._sse_chunk_carries_visible_delta(chunk) is visible


def test_a_response_shorter_than_the_stream_interval_ends_at_its_first_delta(
    monkeypatch,
):
    # The review of 4c9da1ba: a delta written by the final drain (every
    # token held under stream_interval) must end the spans too, not leave
    # them at the engine's first token.
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    state.args.stream_interval = 64
    _fake_engine(monkeypatch, ["Hi"])

    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hi"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 16,
            },
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()

    spans = _stream_stats(body)["ttft_spans"]
    assert spans["endpoint"] == "first_delta_sent"
    assert "first_delta_sent_s" in spans["exclusive_s"]
    assert spans["sum_s"] == pytest.approx(spans["ttft_s"], abs=1e-4)


def test_a_streamed_completion_carries_spans_that_end_at_its_first_text(monkeypatch):
    # /v1/completions opened no clock before (the review of 4c9da1ba).
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    _fake_engine(monkeypatch, ["Hello there"])

    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "Say hello", "stream": True, "max_tokens": 16},
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()
        nonstream = client.post(
            "/v1/completions",
            json={"prompt": "Say hello", "stream": False, "max_tokens": 16},
        )

    spans = _stream_stats(body)["ttft_spans"]
    assert spans["endpoint"] == "first_delta_sent"
    for name in (
        "encode",
        "policy",
        "prologue",
        "dispatch",
        "scheduler_queue",
        "engine_first_token",
        "first_delta_sent",
    ):
        assert f"{name}_s" in spans["exclusive_s"], (name, spans)
    assert spans["sum_s"] == pytest.approx(spans["ttft_s"], abs=1e-4)
    # A response that is not streamed has no first delta: its spans end at
    # the engine's first token.
    assert nonstream.status_code == 200, nonstream.text
    plain = nonstream.json()["mtplx_stats"]["ttft_spans"]
    assert plain["endpoint"] == "engine_first_token"
    assert plain["sum_s"] == pytest.approx(plain["ttft_s"], abs=1e-4)


def test_a_response_from_a_lane_that_does_not_publish_still_carries_spans(
    monkeypatch,
):
    # The batched lanes finalize outside _run_generation, whose envelope is
    # where the spans were published (the review of 4c9da1ba). A generation
    # result without ttft_spans is published by the handler before the
    # response is written.
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0

    def lane_result(_state, prompt_ids, **kwargs):
        observability = dict(kwargs["request_observability"])
        return {
            "text": "OK",
            "tokens": [ord("O"), ord("K")],
            "stats": {
                **observability,
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", lane_result)
    with TestClient(create_app(state)) as client:
        chat = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": False,
                "max_tokens": 4,
            },
        )
        completion = client.post(
            "/v1/completions",
            json={"prompt": "Say OK", "stream": False, "max_tokens": 4},
        )

    for response in (chat, completion):
        assert response.status_code == 200, response.text
        spans = response.json()["mtplx_stats"]["ttft_spans"]
        assert spans["origin"] == "http_arrival"
        for name in ("encode", "policy", "prologue", "dispatch"):
            assert f"{name}_s" in spans["exclusive_s"], (name, spans)


def test_scheduler_queue_wait_is_the_scheduler_queue_span():
    clock = request_spans.RequestClock(100.0)
    clock.mark("dispatch", 100.700)
    assert clock.scheduler_queue_wait_s() is None
    clock.mark("scheduler_queue", 101.900)
    clock.mark("engine_first_token", 102.500)

    assert clock.scheduler_queue_wait_s() == pytest.approx(1.2)
    assert clock.scheduler_queue_wait_s() == pytest.approx(
        clock.summary()["exclusive_s"]["scheduler_queue_s"]
    )


def test_scheduler_queue_wait_counts_only_the_attempt_that_runs():
    clock = request_spans.RequestClock(0.0)
    clock.mark("dispatch", 0.2)
    clock.mark("scheduler_queue", 3.0)
    clock.discard_attempt("retry", now=5.0)
    clock.mark("scheduler_queue", 5.4)

    assert clock.scheduler_queue_wait_s() == pytest.approx(0.4)


def test_scheduler_queue_wait_is_none_without_a_dispatch():
    clock = request_spans.RequestClock(0.0)
    clock.mark("scheduler_queue", 1.0)

    assert clock.scheduler_queue_wait_s() is None
