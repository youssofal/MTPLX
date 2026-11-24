"""Stream termination truth (F34 + F36, 2.8 charlatan-defensibility wave).

F34 — the commit wait between the last content chunk and the finish frame was
an unbounded ``await queue.get()``: a wedged model owner hung the stream
forever with heartbeats dead. It is now a bounded poll that keeps the client
heartbeat cadence alive and breaks with a visible stall-watchdog error finish.

F36 — explicit-cancel paths (POST /v1/mtplx/cancel/{id}) used to end the SSE
stream with no terminal frame while the transport was still up, and the
catch-all ``except BaseException: yield`` swallowed GeneratorExit (RuntimeError
noise + the disconnect metric mis-tagged "stream_cancelled").

The engine is always monkeypatched — no model loads, CPU only.
"""

from __future__ import annotations

import asyncio
import json
import time
from threading import Event

import pytest

pytest.importorskip("fastapi")
from fastapi.responses import Response
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app

from test_server_openai import (  # noqa: E402 - shared fixtures
    _fake_state,
    _fake_streaming_session_state,
    _stream_payloads,
)


def _frames(response_text: str) -> list[dict]:
    return _stream_payloads(response_text)


def _blocking_stream_generation(release, text: str = "OK"):
    """Fake engine: emits one token per char, then parks on a test-owned
    release event so the stream idles mid-generation until the test lets the
    worker unwind."""

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for char in text:
                token_callback([ord(char)])
        release.wait(15.0)
        raise openai._StreamCancelled("stream client disconnected")

    return fake_run_generation


# --- F34: bounded, watchdog-visible commit wait -----------------------------


def test_commit_wait_is_bounded_and_heartbeats_stay_alive(monkeypatch):
    """A wedged owner during the session postcommit must not hang the stream:
    heartbeats keep flowing at the stream cadence and the stall watchdog ends
    the stream with a visible error finish + [DONE] within the deadline."""

    # The frame after the commit (the order before aed64295, still the
    # order for sessions found by prompt inference and with the switch
    # off): the stream itself waits for the commit.
    monkeypatch.setenv("MTPLX_STREAM_TERMINAL_FRAME_BEFORE_COMMIT", "0")
    state = _fake_streaming_session_state()
    monkeypatch.setattr(openai, "STREAM_STALL_DEADLINE_S", 1.0)
    monkeypatch.setattr(openai, "STREAM_HEARTBEAT_INTERVAL_S", 0.05)

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            token_callback([ord("O")])
            token_callback([ord("K")])
        return {
            "text": "OK",
            "tokens": [ord("O"), ord("K")],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            # No _final_state: the worker takes the
            # _store_generation_final_history_snapshot commit branch.
        }

    release_wedged_owner = Event()

    def wedged_store(*_args, **_kwargs):
        # The model owner never answers the commit (wedged owner).
        release_wedged_owner.wait(30.0)
        return {"stored": False, "reason": "wedged_test_owner"}

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", wedged_store
    )

    started = time.monotonic()
    try:
        with TestClient(create_app(state)) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"x-mtplx-session-id": "wedged-owner-session"},
                json={
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "enable_thinking": False,
                    "stream": True,
                    "max_tokens": 4,
                },
            ) as response:
                assert response.status_code == 200
                body = response.read().decode()
    finally:
        release_wedged_owner.set()
    elapsed = time.monotonic() - started

    # Bounded: the old unbounded queue.get() hung here forever.
    assert elapsed < 20.0
    assert "data: [DONE]" in body
    frames = _frames(body)
    heartbeats = [
        frame
        for frame in frames
        if (frame.get("mtplx_progress") or {}).get("heartbeat")
    ]
    assert heartbeats, "client heartbeats must keep flowing during the commit wait"
    error_frames = [
        frame
        for frame in frames
        if frame.get("choices")
        and frame["choices"][0].get("finish_reason") == "error"
    ]
    assert error_frames, "the stall break must be a visible error finish"
    message = error_frames[-1]["error"]["message"]
    assert "stall watchdog" in message
    assert "committing the session" in message


# --- F36(a): explicit cancel gets a terminal frame --------------------------


async def _never_disconnected(self) -> bool:
    return False


def test_explicit_cancel_ends_chat_stream_with_terminal_frame(monkeypatch):
    state = _fake_state()
    state.requests_cancelled = 0
    release_worker = Event()
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai, "_run_generation", _blocking_stream_generation(release_worker)
    )
    # The captured generator runs outside a live transport; pin the request
    # "connected" so the explicit-cancel path (not the disconnect path) is
    # what ends the stream.
    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", _never_disconnected
    )
    captured = _capture_stream_generator(monkeypatch)

    try:
        with TestClient(create_app(state)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"x-mtplx-cache-mode": "bypass"},
                json={
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "stream": True,
                    "max_tokens": 8,
                },
            )
            assert response.status_code == 200
            assert response.text == "captured"

        generator = captured["generator"]
        chunks: list[str] = []

        async def drive() -> None:
            # Frame 1 (role) precedes registration; frame 2 (first token)
            # confirms the in-flight handle exists.
            chunks.append(await generator.__anext__())
            chunks.append(await generator.__anext__())
            request_id = json.loads(
                chunks[0].removeprefix("data: ")
            )["id"]
            # Same registry flip POST /v1/mtplx/cancel/{id} performs.
            assert state.dashboard.in_flight.cancel(request_id) is True
            while True:
                try:
                    chunks.append(
                        await asyncio.wait_for(generator.__anext__(), 10.0)
                    )
                except StopAsyncIteration:
                    break

        asyncio.run(drive())
    finally:
        release_worker.set()

    body = "".join(chunks)
    # The transport was up the whole time: the stream must end with a
    # visible cancel frame and [DONE], never a silent close.
    assert "data: [DONE]" in body
    frames = _frames(body)
    error_frames = [
        frame
        for frame in frames
        if frame.get("choices")
        and frame["choices"][0].get("finish_reason") == "error"
    ]
    assert error_frames
    assert "cancelled via POST /v1/mtplx/cancel" in error_frames[-1]["error"]["message"]
    # Truthful metric tag: an explicit cancel is NOT a client disconnect.
    cancel_records = [
        record
        for record in state.last_metrics
        if record.get("request_cancelled")
    ]
    assert cancel_records
    # #381 attribution: the registry flip IS the POST endpoint's path, so
    # the metric now names it instead of the generic stream_cancelled.
    assert cancel_records[-1]["cancellation_reason"] == "post_endpoint"
    assert cancel_records[-1]["stream_cancelled_by_client"] is False


def test_cancel_ack_ends_completions_stream_with_terminal_frame(monkeypatch):
    """The completions worker acknowledging a cancellation used to end the
    stream with a bare return — no terminal frame, no [DONE] — while the
    transport was still up."""

    state = _fake_state()
    state.runtime.tokenizer.encode = lambda _text, **_kwargs: [1, 2, 3]

    def cancelling_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            token_callback([ord("O")])
        raise openai._StreamCancelled("cancelled mid-flight")

    monkeypatch.setattr(
        openai, "_run_generation_dispatched", cancelling_generation
    )
    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", _never_disconnected
    )

    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 8},
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()

    assert "data: [DONE]" in body
    frames = _frames(body)
    error_frames = [
        frame
        for frame in frames
        if frame.get("choices")
        and frame["choices"][0].get("finish_reason") == "error"
    ]
    assert error_frames
    assert "cancelled" in error_frames[-1]["error"]["message"]


# --- F36(b): GeneratorExit is re-raised, not swallowed ----------------------


def _capture_stream_generator(monkeypatch):
    captured: dict[str, object] = {}

    def capture_streaming_response(content, **_kwargs):
        captured["generator"] = content
        return Response("captured")

    monkeypatch.setattr(openai, "StreamingResponse", capture_streaming_response)
    return captured


def test_chat_generator_exit_is_reraised_and_tags_client_disconnected(monkeypatch):
    """aclose() on the live chat stream generator (what Starlette does on a
    client disconnect) must complete cleanly — the pre-fix catch-all yielded
    after GeneratorExit ("async generator ignored GeneratorExit") — and the
    cancellation metric must say client_disconnected."""

    state = _fake_state()
    state.requests_cancelled = 0
    release_worker = Event()
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai, "_run_generation", _blocking_stream_generation(release_worker)
    )
    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", _never_disconnected
    )
    captured = _capture_stream_generator(monkeypatch)

    try:
        with TestClient(create_app(state)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"x-mtplx-cache-mode": "bypass"},
                json={
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "stream": True,
                    "max_tokens": 8,
                },
            )
            assert response.status_code == 200
            assert response.text == "captured"

        generator = captured["generator"]

        async def drive() -> None:
            # Frame 1 (role) is emitted before the guarded try; frame 2
            # comes from the token queue inside it — close there,
            # mid-generation.
            await generator.__anext__()
            await generator.__anext__()
            await generator.aclose()

        asyncio.run(drive())
    finally:
        release_worker.set()

    cancel_records = [
        record
        for record in state.last_metrics
        if record.get("request_cancelled")
    ]
    assert cancel_records
    assert cancel_records[-1]["cancellation_reason"] == "client_disconnected"
    assert cancel_records[-1]["stream_cancelled_by_client"] is True


def test_completions_generator_exit_is_reraised(monkeypatch):
    """Closing the /v1/completions stream generator mid-generation (client SSE
    disconnect -> GeneratorExit) must re-raise cleanly AND record both the
    lifetime cancellation count and cancellation telemetry receipt."""

    state = _fake_state()
    state.requests_cancelled = 0
    state.runtime.tokenizer.encode = lambda _text, **_kwargs: [1, 2, 3]
    release_worker = Event()
    monkeypatch.setattr(
        openai,
        "_run_generation_dispatched",
        _blocking_stream_generation(release_worker),
    )
    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", _never_disconnected
    )
    captured = _capture_stream_generator(monkeypatch)

    try:
        with TestClient(create_app(state)) as client:
            response = client.post(
                "/v1/completions",
                json={"prompt": "hello", "stream": True, "max_tokens": 8},
            )
            assert response.status_code == 200
            assert response.text == "captured"

        generator = captured["generator"]

        async def drive() -> None:
            await generator.__anext__()
            # Pre-fix this raised RuntimeError("async generator ignored
            # GeneratorExit") out of aclose().
            await generator.aclose()

        asyncio.run(drive())
    finally:
        release_worker.set()

    assert state.dashboard.lifetime.cancelled_total == 1
    assert state.requests_cancelled == 1
    cancel_records = [
        record
        for record in state.last_metrics
        if record.get("request_cancelled")
    ]
    assert cancel_records
    assert cancel_records[-1]["cancellation_reason"] == "client_disconnected"
    assert cancel_records[-1]["stream_cancelled_by_client"] is True


def test_completions_stream_worker_error_does_not_record_cancellation(monkeypatch):
    """A worker exception during /v1/completions streaming ends the stream with
    an error frame without misclassifying teardown as a cancellation."""

    state = _fake_state()
    state.requests_cancelled = 0
    state.runtime.tokenizer.encode = lambda _text, **_kwargs: [1, 2, 3]

    def failing_generation(_state, _prompt_ids, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(openai, "_run_generation_dispatched", failing_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 8},
        )
        assert response.status_code == 200

    frames = _frames(response.text)
    assert any(
        frame.get("choices") and frame["choices"][0].get("finish_reason") == "error"
        for frame in frames
    )
    assert state.dashboard.lifetime.cancelled_total == 0
    assert state.requests_cancelled == 0
    assert not any(record.get("request_cancelled") for record in state.last_metrics)


def test_a_wedged_commit_after_an_early_frame_refuses_the_next_turn(monkeypatch):
    """With the frame at the last token (aed64295), a wedged owner during the
    commit cannot hang the stream, which already ended; the next turn of the
    same session waits for that commit and, once the owner has made no
    progress for the stall deadline, is refused with 503 instead of hanging
    or reading the session before the commit."""

    monkeypatch.delenv("MTPLX_STREAM_TERMINAL_FRAME_BEFORE_COMMIT", raising=False)
    state = _fake_streaming_session_state()
    monkeypatch.setattr(openai, "STREAM_STALL_DEADLINE_S", 1.0)
    monkeypatch.setattr(openai, "STREAM_COMMIT_WAIT_MAX_S", 0.1)

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            token_callback([ord("O")])
            token_callback([ord("K")])
        return {
            "text": "OK",
            "tokens": [ord("O"), ord("K")],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
        }

    release_wedged_owner = Event()
    reads: list[str] = []

    def wedged_store(*_args, **_kwargs):
        release_wedged_owner.wait(30.0)
        return {"stored": False, "reason": "wedged_test_owner"}

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", wedged_store
    )
    from mtplx.engine_session import EngineSession

    original_resolve = EngineSession.resolve_pending_postcommit_for_request

    def observed_resolve(self):
        reads.append(self.session_id)
        return original_resolve(self)

    monkeypatch.setattr(
        EngineSession, "resolve_pending_postcommit_for_request", observed_resolve
    )
    headers = {"x-mtplx-session-id": "wedged-tail-session"}
    try:
        with TestClient(create_app(state)) as client:
            started = time.monotonic()
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "enable_thinking": False,
                    "stream": True,
                    "max_tokens": 4,
                },
            ) as response:
                assert response.status_code == 200
                first = response.read().decode()
            assert "data: [DONE]" in first
            assert time.monotonic() - started < 5.0
            reads.clear()
            second = client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "messages": [
                        {"role": "user", "content": "Say OK"},
                        {"role": "assistant", "content": "OK"},
                        {"role": "user", "content": "Again"},
                    ],
                    "enable_thinking": False,
                    "stream": False,
                    "max_tokens": 4,
                },
            )
            elapsed = time.monotonic() - started
    finally:
        release_wedged_owner.set()

    assert second.status_code == 503, second.text
    assert "session_commit_stalled" in second.text
    assert elapsed < 20.0
    # Refused before it read the session.
    assert reads == []
