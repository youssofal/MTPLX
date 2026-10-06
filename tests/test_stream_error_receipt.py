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

import inspect
import time
from types import SimpleNamespace

from fastapi import HTTPException

import mtplx.server.openai as srv


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
