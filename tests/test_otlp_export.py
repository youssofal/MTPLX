from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mtplx.otlp_export import (
    OTLPExporter,
    OTLPExporterConfig,
    OTLPSpan,
    sanitize_attributes,
)


def test_sanitizer_redacts_content_and_secrets_but_preserves_metrics():
    value = sanitize_attributes(
        {
            "prompt": "private question",
            "messages": [{"role": "user", "content": "secret"}],
            "authorization": "Bearer abc",
            "prompt_tokens": 41,
            "completion_tokens": 12,
            "latency_ms": 8.5,
            "model": "local/model",
        }
    )
    assert value["authorization"] == "<redacted>"
    assert value["prompt"]["content_redacted"] is True
    assert value["messages"]["content_redacted"] is True
    assert value["prompt_tokens"] == 41
    assert value["completion_tokens"] == 12
    assert value["latency_ms"] == 8.5
    assert value["model"] == "local/model"


def test_long_non_content_string_is_bounded_and_hashed():
    value = sanitize_attributes(
        {"route_name": "x" * 1000},
        maximum_string_bytes=64,
    )
    row = value["route_name"]
    assert row["truncated"] is True
    assert row["bytes"] == 1000
    assert len(row["sha256"]) == 64


class _Collector(BaseHTTPRequestHandler):
    payloads = []

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        type(self).payloads.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, _format, *_args):
        return


def test_exporter_sends_valid_otlp_json_without_external_dependency():
    _Collector.payloads.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
    exporter = OTLPExporter(
        OTLPExporterConfig(
            endpoint=endpoint,
            enabled=True,
            batch_size=1,
            flush_interval_s=0.01,
            timeout_s=1.0,
        )
    )
    try:
        with exporter.span(
            "mtplx.test",
            attributes={"prompt": "do not export", "prompt_tokens": 5},
        ):
            pass
        assert exporter.flush(timeout_s=2.0)
    finally:
        exporter.shutdown(timeout_s=1.0)
        server.shutdown()
        thread.join(timeout=1.0)
    assert _Collector.payloads
    payload = _Collector.payloads[0]
    span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["name"] == "mtplx.test"
    attrs = {item["key"]: item["value"] for item in span["attributes"]}
    assert attrs["prompt_tokens"]["intValue"] == "5"
    assert "do not export" not in json.dumps(payload)


def test_export_failure_is_fail_open_and_counted():
    exporter = OTLPExporter(
        OTLPExporterConfig(
            endpoint="http://127.0.0.1:1/v1/traces",
            enabled=True,
            batch_size=1,
            flush_interval_s=0.01,
            timeout_s=0.05,
        )
    )
    try:
        with exporter.span("mtplx.failure"):
            pass
        exporter.flush(timeout_s=1.0)
        snapshot = exporter.snapshot()
    finally:
        exporter.shutdown(timeout_s=0.2)
    assert snapshot["failed_exports"] >= 1
    assert snapshot["failure_is_request_fatal"] is False


def test_bounded_queue_drops_instead_of_blocking():
    exporter = OTLPExporter(
        OTLPExporterConfig(
            endpoint="http://127.0.0.1:1/v1/traces",
            enabled=False,
            maximum_queue_size=1,
        )
    )
    # Directly enable without starting a worker to make the queue deterministic.
    exporter.config = OTLPExporterConfig(
        endpoint="http://127.0.0.1:1/v1/traces",
        enabled=True,
        maximum_queue_size=1,
        batch_size=32,
    )
    span = OTLPSpan(
        name="one",
        trace_id="0" * 32,
        span_id="1" * 16,
        parent_span_id=None,
        start_time_unix_nano=1,
        end_time_unix_nano=2,
        attributes={},
    )
    assert exporter.emit(span) is True
    assert exporter.emit(span) is False
    assert exporter.snapshot()["dropped_spans"] == 1


def test_invalid_endpoint_is_rejected():
    with pytest.raises(ValueError):
        OTLPExporterConfig(endpoint="file:///tmp/traces", enabled=True)
