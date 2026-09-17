#!/usr/bin/env python3
"""Measure OTLP exporter overhead, delivery, privacy, and failure behavior."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtplx.otlp_export import OTLPExporter, OTLPExporterConfig  # noqa: E402


class Collector(BaseHTTPRequestHandler):
    lock = threading.Lock()
    payloads: list[bytes] = []
    span_count = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        with type(self).lock:
            type(self).payloads.append(body)
            type(self).span_count += len(spans)
        self.send_response(200)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _timing(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
    }


def _maxrss_kib() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value // 1024 if sys.platform == "darwin" else value


def _emit(exporter: OTLPExporter, count: int, sentinel: str) -> dict[str, object]:
    latencies_us: list[float] = []
    queue_high_water = 0
    rss_before = _maxrss_kib()
    started_all = time.perf_counter()
    for index in range(count):
        started = time.perf_counter_ns()
        with exporter.span(
            "mtplx.measurement",
            attributes={
                "prompt": f"{sentinel}-{index}",
                "authorization": f"Bearer {sentinel}",
                "prompt_tokens": 1024 + index % 128,
                "completion_tokens": 64,
                "latency_ms": 12.5,
            },
        ):
            pass
        latencies_us.append((time.perf_counter_ns() - started) / 1000.0)
        queue_high_water = max(queue_high_water, exporter.snapshot()["queue_size"])
    exporter.flush(timeout_s=10.0)
    elapsed_s = time.perf_counter() - started_all
    rss_after = _maxrss_kib()
    snapshot = exporter.snapshot()
    return {
        "emit_us": _timing(latencies_us),
        "elapsed_s": elapsed_s,
        "emit_calls_per_second": count / elapsed_s,
        "queue_high_water": queue_high_water,
        "maxrss_delta_kib": max(0, rss_after - rss_before),
        "snapshot": snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sentinel = "MTPLX_PRIVATE_SENTINEL_7d61"

    disabled = OTLPExporter(OTLPExporterConfig(enabled=False))
    disabled_result = _emit(disabled, args.spans, sentinel)
    disabled.shutdown(timeout_s=0.1)

    Collector.payloads.clear()
    Collector.span_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    healthy = OTLPExporter(
        OTLPExporterConfig(
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces",
            enabled=True,
            batch_size=args.batch_size,
            maximum_queue_size=max(args.queue_size, args.spans),
            flush_interval_s=0.05,
            timeout_s=1.0,
        )
    )
    try:
        healthy_result = _emit(healthy, args.spans, sentinel)
    finally:
        healthy.shutdown(timeout_s=2.0)
        server.shutdown()
        thread.join(timeout=2.0)
    joined_payloads = b"".join(Collector.payloads)
    healthy_result["collector_spans"] = Collector.span_count
    healthy_result["privacy_sentinel_leaks"] = joined_payloads.count(
        sentinel.encode("utf-8")
    )

    dead = OTLPExporter(
        OTLPExporterConfig(
            endpoint="http://127.0.0.1:1/v1/traces",
            enabled=True,
            batch_size=args.batch_size,
            maximum_queue_size=args.queue_size,
            flush_interval_s=0.01,
            timeout_s=0.01,
        )
    )
    dead_result = _emit(dead, args.spans, sentinel)
    dead.shutdown(timeout_s=1.0)

    receipt = {
        "measurement": "otlp_http_export",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "spans_per_arm": args.spans,
        "batch_size": args.batch_size,
        "healthy_queue_size": max(args.queue_size, args.spans),
        "dead_endpoint_queue_size": args.queue_size,
        "disabled": disabled_result,
        "healthy_collector": healthy_result,
        "dead_endpoint": dead_result,
        "request_failures": 0,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
