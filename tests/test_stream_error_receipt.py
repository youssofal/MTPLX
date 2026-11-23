"""#415: engine failures must write honest receipts, not fake cancellations.

The shipped lie: a Pi 262K post-compaction prefill was pressure-aborted
with a structured 507, but the stream teardown sets the cancel event, so
the request log recorded ``request_cancelled=true`` /
``cancellation_reason="stream_cancelled"`` /
``stream_cancelled_by_client=false`` — the engine-health refusal was
invisible to request-log diagnostics. These tests pin the error
classifier, the honest error envelope, and the wiring that prefers the
error receipt over the cancellation receipt.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mtplx.server.openai as srv
import mtplx.server.responses as responses_api

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_server_openai import CaptureTokenizer, _fake_state  # noqa: E402


class TestStreamErrorKind:
    def test_http_507_is_memory_refusal(self):
        exc = HTTPException(status_code=507, detail="insufficient memory: ...")
        assert srv._stream_error_kind(exc) == ("memory_refusal", 507)

    def test_allocation_failure_marker_is_memory_refusal(self):
        exc = RuntimeError("[metal::malloc] Unable to allocate 16 GB")
        assert srv._stream_error_kind(exc) == ("memory_refusal", 507)

    def test_memory_error_is_memory_refusal(self):
        assert srv._stream_error_kind(MemoryError()) == ("memory_refusal", 507)

    def test_other_http_keeps_status(self):
        exc = HTTPException(status_code=409, detail="busy")
        assert srv._stream_error_kind(exc) == ("http_error", 409)

    def test_generic_exception_is_engine_error(self):
        assert srv._stream_error_kind(RuntimeError("boom")) == (
            "engine_error",
            None,
        )


class TestErrorEnvelope:
    def _capture(self, monkeypatch):
        rows: list[dict] = []
        monkeypatch.setattr(
            srv, "_record_request_metrics", lambda _state, row: rows.append(row)
        )
        published: list[dict] = []
        state = SimpleNamespace(
            dashboard=SimpleNamespace(
                bus=SimpleNamespace(publish=published.append)
            ),
        )
        return state, rows, published

    def test_memory_refusal_row_is_honest(self, monkeypatch):
        state, rows, published = self._capture(monkeypatch)
        error = HTTPException(
            status_code=507,
            detail=(
                "insufficient memory: the request exceeded available GPU "
                "memory (sustained critical memory pressure during prefill; "
                "aborted before the allocator wall)."
            ),
        )
        srv._record_stream_error_metric(
            state,
            response_id="chatcmpl-test-507",
            session_id="pi-session",
            prompt_tokens=38113,
            streamed_completion_tokens=0,
            stream_started_s=time.perf_counter() - 53.2,
            error=error,
            request_observability={"request_id": "chatcmpl-test-507"},
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["request_cancelled"] is False
        assert row["stream_error"] is True
        assert row["error_kind"] == "memory_refusal"
        assert row["error_status"] == 507
        assert row["memory_refusal"] is True
        assert "insufficient memory" in row["error_detail"]
        assert "cancellation_reason" not in row
        assert row["stream_cancelled_by_client"] is False
        assert row["prompt_tokens"] == 38113
        assert row["completion_tokens"] == 0
        assert published and published[0]["kind"] == "request_error"

    def test_nonstream_mode_stamped(self, monkeypatch):
        state, rows, _published = self._capture(monkeypatch)
        srv._record_stream_error_metric(
            state,
            response_id="chatcmpl-test-nonstream",
            session_id=None,
            prompt_tokens=10,
            streamed_completion_tokens=0,
            stream_started_s=time.perf_counter(),
            error=HTTPException(status_code=507, detail="insufficient memory"),
            request_observability={},
            mode="nonstream",
        )
        assert rows[0]["mode"] == "nonstream"


class TestWiring:
    def test_stream_finally_prefers_error_receipt(self):
        src = inspect.getsource(srv)
        cut = src.index("stream_error_item is not None and generated is None")
        # The error branch must run INSTEAD of the cancellation recorder,
        # not merely before it.
        follow = src[cut : cut + 4000]
        assert "_record_stream_error_metric" in follow
        assert "elif cancel_event.is_set() and generated is None:" in follow

    def test_error_queue_item_captured(self):
        src = inspect.getsource(srv)
        assert "stream_error_item = item" in src

    def test_nonstream_507_records_error_receipt(self):
        src = inspect.getsource(srv)
        cut = src.index('mode="nonstream"')
        lead = src[cut - 2500 : cut]
        assert "except HTTPException as exc:" in lead
        assert "== 507" in lead


class TestRefusalFieldsOnRow:
    def test_structured_507_carries_reason_and_retry_when(self, monkeypatch):
        rows: list[dict] = []
        monkeypatch.setattr(
            srv, "_record_request_metrics", lambda _state, row: rows.append(row)
        )
        state = SimpleNamespace(
            dashboard=SimpleNamespace(bus=SimpleNamespace(publish=lambda _e: None)),
        )
        error = HTTPException(
            status_code=507,
            detail={
                "message": "insufficient memory: ...",
                "code": "insufficient_memory",
                "memory": {
                    "refusal_reason": "system_memory_short_after_reclamation",
                    "retry_when": "after_other_apps_free_memory",
                    "retry_can_succeed": True,
                },
            },
        )
        srv._record_stream_error_metric(
            state,
            response_id="chatcmpl-test-reason",
            session_id=None,
            prompt_tokens=90005,
            streamed_completion_tokens=0,
            stream_started_s=time.perf_counter(),
            error=error,
            request_observability={},
            mode="nonstream",
        )
        row = rows[0]
        assert row["refusal_reason"] == "system_memory_short_after_reclamation"
        assert row["retry_when"] == "after_other_apps_free_memory"
        assert row["retry_can_succeed"] is True


class TestMidStreamLoggingAndTelemetry:
    def test_log_stream_error_includes_context_and_traceback(self, caplog):
        try:
            raise RuntimeError("watchdog boom")
        except RuntimeError as exc:
            err = exc

        with caplog.at_level(logging.ERROR, logger="mtplx.server"):
            srv._log_stream_error(
                err,
                request_id="chatcmpl-log-1",
                path="/v1/chat/completions",
                session_id="sess-1",
                streamed_tokens=7,
            )

        assert len(caplog.records) == 1
        rec = caplog.records[0]
        assert "unhandled streaming error" in rec.message
        assert "request_id=chatcmpl-log-1" in rec.message
        assert "path=/v1/chat/completions" in rec.message
        assert "session_id=sess-1" in rec.message
        assert "streamed_tokens=7" in rec.message
        assert "RuntimeError: watchdog boom" in rec.message
        assert rec.exc_info is not None and rec.exc_text and "Traceback" in rec.exc_text

    def test_record_stream_error_metric_emits_generation_event_unless_suppressed(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(srv, "_record_request_metrics", lambda _state, _row: None)
        state = SimpleNamespace(
            args=SimpleNamespace(server_console=False),
            dashboard=SimpleNamespace(bus=SimpleNamespace(publish=lambda _e: None)),
        )
        srv._record_stream_error_metric(
            state,
            response_id="chatcmpl-stdout-1",
            session_id="sess-2",
            prompt_tokens=123,
            streamed_completion_tokens=4,
            stream_started_s=time.perf_counter() - 0.1,
            error=RuntimeError("midstream failure"),
            request_observability={},
        )
        out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        events = [json.loads(line) for line in out_lines]
        gen_events = [e for e in events if e.get("event") == "mtplx_openai_generation"]
        assert len(gen_events) == 1
        event = gen_events[0]
        assert event["request_id"] == "chatcmpl-stdout-1"
        assert event["prompt_tokens"] == 123
        assert event["completion_tokens"] == 4
        assert event["finish_reason"] == "error"
        assert event["error_kind"] == "engine_error"
        assert "midstream failure" in event["error_detail"]

        # Warmup and server_console suppress the stdout JSON event.
        srv._record_stream_error_metric(
            state,
            response_id="chatcmpl-warmup",
            session_id=None,
            prompt_tokens=10,
            streamed_completion_tokens=0,
            stream_started_s=time.perf_counter(),
            error=RuntimeError("warmup err"),
            request_observability={"warmup": True},
        )
        state.args.server_console = True
        srv._record_stream_error_metric(
            state,
            response_id="chatcmpl-console",
            session_id=None,
            prompt_tokens=10,
            streamed_completion_tokens=0,
            stream_started_s=time.perf_counter(),
            error=RuntimeError("console err"),
            request_observability={},
        )
        assert capsys.readouterr().out.strip() == ""

    @staticmethod
    def _streaming_error_app(monkeypatch):
        rows: list[dict] = []
        monkeypatch.setattr(
            srv, "_record_request_metrics", lambda _state, row: rows.append(row)
        )

        def fail_mid_stream(_state, _prompt_ids, **kwargs):
            token_callback = kwargs.get("token_callback")
            if token_callback is not None:
                token_callback([ord("h"), ord("i")])
            raise RuntimeError(
                "[METAL] Command buffer execution failed: Impacting Interactivity"
            )

        monkeypatch.setattr(srv, "_run_generation", fail_mid_stream)
        state = _fake_state()
        state.runtime.tokenizer = CaptureTokenizer()
        state.args.stats_footer = False
        return TestClient(srv.create_app(state)), rows

    @staticmethod
    def _parse_sse(body: str) -> list[tuple[str | None, str]]:
        frames: list[tuple[str | None, str]] = []
        for raw_frame in body.split("\n\n"):
            if not raw_frame.strip():
                continue
            event_name = None
            data_parts: list[str] = []
            for line in raw_frame.splitlines():
                if line.startswith("event: "):
                    event_name = line[len("event: ") :].strip()
                elif line.startswith("data: "):
                    data_parts.append(line[len("data: ") :])
            if data_parts:
                frames.append((event_name, "\n".join(data_parts)))
        return frames

    def test_chat_completions_mid_stream_error_logs_and_records_telemetry(
        self, monkeypatch, caplog, capsys
    ):
        client, rows = self._streaming_error_app(monkeypatch)
        with caplog.at_level(logging.ERROR, logger="mtplx.server"):
            response = client.post(
                "/v1/chat/completions",
                headers={"x-mtplx-cache-mode": "bypass"},
                json={
                    "model": "mtplx-test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        frames = self._parse_sse(response.text)
        assert frames[-1][1] == "[DONE]"
        payloads = [json.loads(data) for _, data in frames[:-1]]
        assert payloads[-1]["choices"][0]["finish_reason"] == "error"
        assert "Impacting Interactivity" in payloads[-1]["error"]["message"]

        assert any(
            "unhandled streaming error" in r.message
            and "path=/v1/chat/completions" in r.message
            and "Traceback" in ((r.exc_text or "") + r.message)
            for r in caplog.records
        )
        assert len(rows) == 1
        assert rows[0]["stream_error"] is True
        assert rows[0]["error_kind"] == "engine_error"
        assert "Impacting Interactivity" in rows[0]["error_detail"]

        gen_events = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if '"mtplx_openai_generation"' in line
        ]
        assert len(gen_events) == 1
        assert gen_events[0]["finish_reason"] == "error"

    def test_completions_mid_stream_error_logs_and_records_telemetry(
        self, monkeypatch, caplog, capsys
    ):
        client, rows = self._streaming_error_app(monkeypatch)
        with caplog.at_level(logging.ERROR, logger="mtplx.server"):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "mtplx-test-model",
                    "prompt": "hello",
                    "stream": True,
                },
            )

        assert response.status_code == 200
        frames = self._parse_sse(response.text)
        assert frames[-1][1] == "[DONE]"
        payloads = [json.loads(data) for _, data in frames[:-1]]
        assert payloads[-1]["choices"][0]["finish_reason"] == "error"
        assert "Impacting Interactivity" in payloads[-1]["error"]["message"]

        assert any(
            "unhandled streaming error" in r.message
            and "path=/v1/completions" in r.message
            and "Traceback" in ((r.exc_text or "") + r.message)
            for r in caplog.records
        )
        assert len(rows) == 1
        assert rows[0]["stream_error"] is True
        assert rows[0]["error_kind"] == "engine_error"

        gen_events = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if '"mtplx_openai_generation"' in line
        ]
        assert len(gen_events) == 1
        assert gen_events[0]["finish_reason"] == "error"

    def test_messages_mid_stream_error_logs_and_records_telemetry(
        self, monkeypatch, caplog, capsys
    ):
        client, rows = self._streaming_error_app(monkeypatch)
        with caplog.at_level(logging.ERROR, logger="mtplx.server"):
            response = client.post(
                "/v1/messages",
                headers={"x-mtplx-cache-mode": "bypass"},
                json={
                    "model": "mtplx-test-model",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        frames = self._parse_sse(response.text)
        error_frames = [
            json.loads(data) for event, data in frames if event == "error"
        ]
        assert len(error_frames) == 1
        assert "Impacting Interactivity" in error_frames[0]["error"]["message"]

        log_hits = [
            r
            for r in caplog.records
            if "unhandled streaming error" in r.message and "path=/v1/messages" in r.message
        ]
        assert len(log_hits) == 1
        assert len(rows) == 1
        assert rows[0]["stream_error"] is True
        gen_events = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if '"mtplx_openai_generation"' in line
        ]
        assert len(gen_events) == 1
        assert gen_events[0]["finish_reason"] == "error"

    def test_responses_mid_stream_error_logs_and_records_telemetry(
        self, monkeypatch, caplog, capsys
    ):
        client, rows = self._streaming_error_app(monkeypatch)
        with caplog.at_level(logging.ERROR, logger="mtplx.server"):
            response = client.post(
                "/v1/responses",
                headers={"x-mtplx-cache-mode": "bypass"},
                json={
                    "model": "mtplx-test-model",
                    "input": "hello",
                    "stream": True,
                },
            )

        assert response.status_code == 200
        frames = self._parse_sse(response.text)
        error_frames = [
            json.loads(data) for event, data in frames if event == "error"
        ]
        failed_frames = [
            json.loads(data) for event, data in frames if event == "response.failed"
        ]
        assert len(error_frames) == 1
        assert len(failed_frames) == 1
        assert "Impacting Interactivity" in error_frames[0]["message"]

        log_hits = [
            r
            for r in caplog.records
            if "unhandled streaming error" in r.message
            and "path=/v1/responses" in r.message
        ]
        assert len(log_hits) == 1
        assert len(rows) == 1
        assert rows[0]["stream_error"] is True
        gen_events = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if '"mtplx_openai_generation"' in line
        ]
        assert len(gen_events) == 1
        assert gen_events[0]["finish_reason"] == "error"

    def test_stream_translator_local_exceptions_are_logged(self, caplog):
        async def bad_json_iter():
            yield b"data: {not-json\n\n"

        async def exploding_iter():
            yield b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            raise RuntimeError("iterator crashed")

        async def collect(ait):
            return [chunk async for chunk in ait]

        with caplog.at_level(logging.ERROR, logger="mtplx.server"):
            msg_bad = "".join(
                asyncio.run(
                    collect(
                        srv._anthropic_stream_from_openai_sse(
                            bad_json_iter(), model="m"
                        )
                    )
                )
            )
            msg_boom = "".join(
                asyncio.run(
                    collect(
                        srv._anthropic_stream_from_openai_sse(
                            exploding_iter(), model="m"
                        )
                    )
                )
            )
            resp_bad = "".join(
                asyncio.run(
                    collect(
                        responses_api.stream_from_chat_sse(
                            bad_json_iter(),
                            response_id="resp_1",
                            model="m",
                            response_fields={},
                            custom_tool_names=set(),
                            namespace_functions={},
                        )
                    )
                )
            )
            resp_boom = "".join(
                asyncio.run(
                    collect(
                        responses_api.stream_from_chat_sse(
                            exploding_iter(),
                            response_id="resp_2",
                            model="m",
                            response_fields={},
                            custom_tool_names=set(),
                            namespace_functions={},
                        )
                    )
                )
            )

        assert "event: error" in msg_bad and "failed to parse upstream SSE chunk" in msg_bad
        assert "event: error" in msg_boom and "iterator crashed" in msg_boom
        assert "event: error" in resp_bad and "event: response.failed" in resp_bad
        assert "event: error" in resp_boom and "iterator crashed" in resp_boom
        logged_paths = [r.message for r in caplog.records if "unhandled streaming error" in r.message]
        assert sum("path=/v1/messages" in m for m in logged_paths) == 2
        assert sum("path=/v1/responses" in m for m in logged_paths) == 2
