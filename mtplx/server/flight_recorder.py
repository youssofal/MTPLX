"""Per-request flight recorder: durable second-by-second generation telemetry.

Motivated by the 2026-08-21 forensics runs: per-request receipts alone cannot
answer "what was the TPS curve of that 45-minute think?", "is the engine hung
or deriving right now?", or "what did the cancelled turn actually generate?".
The recorder writes a compact JSONL event stream per serve port:

    begin    request accepted (identity, prompt size)
    prefill  decode started (cached/new split, prefill timing)
    s        ~1 Hz while decoding: tokens, instantaneous+cumulative TPS,
             reasoning/content chars, live MTP accepted/drafted-by-depth
    end      ALWAYS written — completion, cancel, disconnect, and orphaned
             streams all land here with whatever was accumulated
    pc       postcommit (idle save) outcomes between requests

Default file: ~/.mtplx/metrics/flight-<port>.jsonl (rotation-cascaded like the
request log). Events are numeric telemetry plus a transient in-memory tail for
the live endpoint; full generated text is persisted separately under
~/.mtplx/metrics/gen/ only for abnormal endings by default (the one case where
the client also loses it), controlled by MTPLX_FLIGHT_TEXT=abnormal|always|off.

Threading: counters are updated from the HTTP event loop (delta/token hooks)
and the model-owner thread (live by-depth publish) as single-writer plain
attributes — readers tolerate tearing, mirroring progress_heartbeat. All disk
I/O happens on one daemon writer thread; hot paths only enqueue. The registry
mutations and snapshots share one lock, mirroring InFlightRegistry.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Callable

_FLIGHT_LOG_MAX_BYTES = 64 * 1024 * 1024
_FLIGHT_LOG_KEEP_GENERATIONS = 4
_TEXT_CAPTURE_MAX_CHARS = 4_000_000
_TAIL_CHARS = 400
_SAMPLE_INTERVAL_S = 1.0
_TPS_WINDOW = 48
_LIVE_FIELDS = (
    ("accepted_by_depth", "acc"), ("drafted_by_depth", "drf"),
    ("verify_calls", "vc"), ("verify_time_s", "vt"), ("draft_time_s", "dt"),
    ("verify_forward_time_s", "vft"), ("verify_logits_eval_time_s", "vlt"),
    ("verify_hidden_eval_time_s", "vht"),
    ("verify_target_distribution_time_s", "vdt"),
    ("verify_eval_unattributed_time_s", "vut"),
    ("accept_time_s", "at"), ("commit_time_s", "ct"), ("repair_time_s", "rt"),
    ("snapshot_time_s", "st"), ("bonus_time_s", "bt"),
    ("capture_commit_time_s", "cct"), ("active_memory_bytes", "mem_active"),
    ("cache_memory_bytes", "mem_cache"), ("peak_memory_bytes", "mem_peak"),
    ("verify_route", "route"), ("compiled_verify_calls", "cv"),
    ("eager_verify_calls", "evc"),
)


class FlightRecord:
    """Mutable per-request accumulator. Field writers: event loop (chars,
    tokens), model-owner thread (live_depth). Snapshot readers copy values."""

    __slots__ = (
        "request_id",
        "session_id",
        "model",
        "stream",
        "started_s",
        "prompt_tokens",
        "prefill",
        "decode_started_s",
        "last_token_s",
        "gen_tokens",
        "token_times",
        "reasoning_chars",
        "content_chars",
        "tail",
        "text_parts",
        "text_chars",
        "live_depth",
        "last_sample_s",
        "samples",
    )

    def __init__(
        self,
        request_id: str,
        *,
        session_id: str | None,
        model: str | None,
        prompt_tokens: int | None,
        stream: bool,
        capture_text: bool,
    ) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self.model = model
        self.stream = stream
        self.started_s = time.time()
        self.prompt_tokens = prompt_tokens
        self.prefill: dict[str, Any] | None = None
        self.decode_started_s: float | None = None
        self.last_token_s: float | None = None
        self.gen_tokens = 0
        self.token_times: deque[float] = deque(maxlen=_TPS_WINDOW)
        self.reasoning_chars = 0
        self.content_chars = 0
        self.tail: deque[str] = deque(maxlen=16)
        self.text_parts: list[str] | None = [] if capture_text else None
        self.text_chars = 0
        self.live_depth: dict[str, Any] | None = None
        self.last_sample_s = 0.0
        self.samples = 0

    def tps_window(self) -> float:
        times = list(self.token_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def tps_avg(self, now_s: float) -> float:
        if self.decode_started_s is None or self.gen_tokens < 2:
            return 0.0
        span = now_s - self.decode_started_s
        return (self.gen_tokens - 1) / span if span > 0 else 0.0

    def tail_text(self) -> str:
        return "".join(self.tail)[-_TAIL_CHARS:]


class FlightRecorder:
    """Registry + JSONL writer. A recorder with path=None is fully inert."""

    def __init__(self, path: str | None, *, text_mode: str = "abnormal") -> None:
        self.path = path
        self.enabled = bool(path)
        self.text_mode = text_mode if text_mode in {"abnormal", "always", "off"} else "abnormal"
        self._records: dict[str, FlightRecord] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=10)
        self._lock = threading.Lock()
        self._queue: "queue.SimpleQueue[tuple[str, Any] | None]" = queue.SimpleQueue()
        self._writer: threading.Thread | None = None
        self._lines_written = 0

    # -- writer thread ------------------------------------------------------

    def _ensure_writer(self) -> None:
        if self._writer is not None and self._writer.is_alive():
            return
        with self._lock:
            if self._writer is not None and self._writer.is_alive():
                return
            self._writer = threading.Thread(
                target=self._writer_loop, name="mtplx-flight-writer", daemon=True
            )
            self._writer.start()

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            # Batch-drain: group every queued line into ONE open/write/close so
            # the disk sees one append per burst, not one per event (SSD-wear
            # and syscall frugality; ordering across kinds is preserved).
            batch: list[tuple[str, Any]] = [item]
            try:
                while True:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            lines: list[dict[str, Any]] = []
            try:
                for kind, payload in batch:
                    if kind == "line":
                        lines.append(payload)
                        continue
                    if lines:
                        self._append_lines(lines)
                        lines = []
                    if kind == "text":
                        dest, body = payload
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with open(dest, "w", encoding="utf-8") as sink:
                            sink.write(body)
                if lines:
                    self._append_lines(lines)
            except Exception:
                # Telemetry must never take down its writer; drop and continue.
                continue

    def _append_lines(self, events: list[dict[str, Any]]) -> None:
        path = self.path
        if not path or not events:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if os.path.getsize(path) >= _FLIGHT_LOG_MAX_BYTES:
                for gen in range(_FLIGHT_LOG_KEEP_GENERATIONS - 1, 0, -1):
                    older = f"{path}.{gen}"
                    if os.path.exists(older):
                        os.replace(older, f"{path}.{gen + 1}")
                os.replace(path, f"{path}.1")
        except OSError:
            pass
        body = "".join(
            json.dumps(event, ensure_ascii=False, default=str) + "\n"
            for event in events
        )
        with open(path, "a", encoding="utf-8") as sink:
            sink.write(body)
        self._lines_written += len(events)

    def _emit(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._ensure_writer()
        self._queue.put(("line", event))

    # -- lifecycle hooks (event loop unless noted) --------------------------

    def begin(
        self,
        request_id: str,
        *,
        session_id: str | None,
        model: str | None,
        prompt_tokens: int | None,
        stream: bool,
    ) -> None:
        if not self.enabled or not request_id:
            return
        record = FlightRecord(
            request_id,
            session_id=session_id,
            model=model,
            prompt_tokens=prompt_tokens,
            stream=stream,
            capture_text=self.text_mode != "off",
        )
        with self._lock:
            self._records[request_id] = record
        self._emit(
            {
                "ev": "begin",
                "ts": record.started_s,
                "rid": request_id,
                "session_id": session_id,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "stream": stream,
            }
        )

    def note_decode_started(self, request_id: str, prefill_state: Any) -> None:
        record = self._records.get(request_id)
        if record is None or record.prefill is not None:
            return
        payload: dict[str, Any] = {}
        if isinstance(prefill_state, dict):
            payload = {
                key: prefill_state.get(key)
                for key in (
                    "tokens_done",
                    "tokens_total",
                    "cached_tokens",
                    "elapsed_s",
                    "prefill_tok_s",
                )
                if prefill_state.get(key) is not None
            }
        record.prefill = payload
        self._emit(
            {
                "ev": "prefill",
                "ts": time.time(),
                "rid": request_id,
                **payload,
            }
        )

    def on_delta(self, request_id: str, field: str, text: str) -> None:
        record = self._records.get(request_id)
        if record is None or not text:
            return
        if field == "reasoning_content":
            record.reasoning_chars += len(text)
        else:
            record.content_chars += len(text)
        record.tail.append(text)
        if record.text_parts is not None and record.text_chars < _TEXT_CAPTURE_MAX_CHARS:
            record.text_parts.append(text)
            record.text_chars += len(text)

    def on_tokens(self, request_id: str, count: int, timestamp_s: float) -> None:
        """Token-batch hook from the SSE drain loop. Also drives sampling: the
        1 Hz cadence rides the token stream itself, so there is no timer thread
        and an idle engine costs nothing."""
        record = self._records.get(request_id)
        if record is None or count <= 0:
            return
        if record.decode_started_s is None:
            record.decode_started_s = timestamp_s
        record.gen_tokens += count
        record.token_times.extend([timestamp_s] * min(count, _TPS_WINDOW))
        record.last_token_s = timestamp_s
        if timestamp_s - record.last_sample_s >= _SAMPLE_INTERVAL_S:
            record.last_sample_s = timestamp_s
            record.samples += 1
            sample: dict[str, Any] = {
                "ev": "s",
                "ts": time.time(),
                "rid": request_id,
                "gen": record.gen_tokens,
                "tps": round(record.tps_window(), 2),
                "tps_avg": round(record.tps_avg(timestamp_s), 2),
                "rc": record.reasoning_chars,
                "cc": record.content_chars,
                "ctx": (record.prompt_tokens or 0) + record.gen_tokens,
            }
            depth = record.live_depth
            if depth:
                for src, dst in _LIVE_FIELDS:
                    value = depth.get(src)
                    if value is not None:
                        sample[dst] = (
                            round(value, 3) if isinstance(value, float) else value
                        )
            self._emit(sample)

    def live_depth_snapshot(self, request_id: str) -> dict[str, Any]:
        """Host-only counters for the dashboard, scoped to this request."""
        record = self._records.get(request_id)
        return dict(record.live_depth or {}) if record is not None else {}

    def live_depth_sink(self, request_id: str) -> Callable[[dict[str, Any]], None] | None:
        """Returns the model-owner-thread publisher for by-depth totals, or
        None when the recorder is off. The callable assigns one dict ref —
        single writer, tear-tolerant readers, no lock (progress_heartbeat
        precedent)."""
        if not self.enabled:
            return None
        record = self._records.get(request_id)
        if record is None:
            return None

        def publish(payload: dict[str, Any]) -> None:
            record.live_depth = payload
            # Non-streaming requests never enter the SSE drain, so on_tokens
            # never fires and the file got begin/end with ZERO samples
            # (found 2026-08-22: an Ivan-harness arm — non-streaming by his
            # protocol — left 75 sample-less requests). The generation-side
            # sink already publishes at most ~1 Hz, so emitting the sample
            # here when the stream side hasn't costs the same one enqueue
            # per second; streamed requests keep their richer stream-side
            # samples (this only fires when on_tokens hasn't sampled).
            now = time.perf_counter()
            if now - record.last_sample_s < _SAMPLE_INTERVAL_S:
                return
            record.last_sample_s = now
            record.samples += 1
            generated = int(payload.get("generated_tokens") or 0)
            sample: dict[str, Any] = {
                "ev": "s",
                "ts": time.time(),
                "rid": request_id,
                "gen": max(generated, record.gen_tokens),
                "rc": record.reasoning_chars,
                "cc": record.content_chars,
                "ctx": (record.prompt_tokens or 0) + max(generated, record.gen_tokens),
            }
            if record.token_times:
                # Stream-fed rates only; a non-streaming request has no
                # token-time window and a fabricated 0.0 would read as a
                # stall. trace derives its curve from gen deltas regardless.
                sample["tps"] = round(record.tps_window(), 2)
                sample["tps_avg"] = round(record.tps_avg(now), 2)
            for src, dst in _LIVE_FIELDS:
                value = payload.get(src)
                if value is not None:
                    sample[dst] = round(value, 3) if isinstance(value, float) else value
            self._emit(sample)

        return publish

    def emit_route(self, event: dict[str, Any]) -> None:
        """Route Tape record (model-owner thread; enqueue only). The smallest
        public wrapper over _emit so mtplx.route_tape never touches the queue."""
        if not self.enabled:
            return
        self._emit(event)

    def pc(self, session_id: str | None, payload: dict[str, Any]) -> None:
        """Postcommit outcome event (model-owner thread; enqueue only)."""
        if not self.enabled:
            return
        event = {"ev": "pc", "ts": time.time(), "session_id": session_id}
        for key in (
            "action",
            "stored",
            "mode",
            "reason",
            "elapsed_s",
            "retry_scheduled",
            # generation_final receipts: how long the two renders were when
            # the O(1) snapshot was refused, and the committed prefix when
            # it was taken.
            "history_tokens",
            "generation_boundary_tokens",
            "history_suffix_tokens",
            "prefix_len",
            "divergence_token",
            "divergence_offset_in_turn",
        ):
            if key in payload:
                event[key] = payload[key]
        self._emit(event)

    def end(self, request_id: str | None, receipt: dict[str, Any]) -> None:
        """Terminal event — called from the single receipt sink so completion,
        cancellation, and disconnect all funnel here (any thread; enqueue only)."""
        if not self.enabled or not request_id:
            return
        with self._lock:
            record = self._records.pop(request_id, None)
        if record is None:
            return
        now = time.time()
        cancelled = bool(receipt.get("request_cancelled"))
        reason = (
            receipt.get("cancellation_reason")
            if cancelled
            else receipt.get("finish_reason") or "stop"
        )
        event: dict[str, Any] = {
            "ev": "end",
            "ts": now,
            "rid": request_id,
            "session_id": record.session_id,
            "reason": reason,
            "cancelled": cancelled,
            "elapsed_s": round(now - record.started_s, 3),
            "gen": record.gen_tokens,
            "rc": record.reasoning_chars,
            "cc": record.content_chars,
            "samples": record.samples,
        }
        for key in (
            "completion_tokens",
            "prompt_tokens",
            "cached_tokens",
            "new_prefill_tokens",
            "decode_tok_s",
            "ttft_s",
            # Fork-EV shadow aggregate (MTPLX_FORKEV_TELEMETRY); absent when
            # the instrument is off, so existing flight rows are unchanged.
            "forkev",
        ):
            if receipt.get(key) is not None:
                event[key] = receipt[key]
        text = "".join(record.text_parts) if record.text_parts else ""
        if text and (self.text_mode == "always" or (self.text_mode == "abnormal" and cancelled)):
            dest = os.path.join(
                os.path.dirname(self.path or ""), "gen", f"{request_id}.txt"
            )
            event["text_path"] = dest
            self._ensure_writer()
            self._queue.put(("text", (dest, text)))
        self._emit(event)
        with self._lock:
            self._recent.appendleft(
                {k: event[k] for k in event if k not in ("ev",)}
            )

    def sweep(self, request_id: str) -> None:
        """Stream teardown safety net: if the receipt sink never fired for this
        request (unexpected error path), still write an end event."""
        if not self.enabled:
            return
        if request_id in self._records:
            self.end(request_id, {"request_cancelled": True, "cancellation_reason": "orphaned"})

    # -- live snapshot (any thread) -----------------------------------------

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        now_perf = time.perf_counter()
        with self._lock:
            records = list(self._records.values())
            recent = list(self._recent)
        active = []
        for record in records:
            last_perf = record.last_token_s
            active.append(
                {
                    "rid": record.request_id,
                    "session_id": record.session_id,
                    "model": record.model,
                    "phase": "decode" if record.decode_started_s is not None else "prefill",
                    "started_at": record.started_s,
                    "elapsed_s": round(now - record.started_s, 1),
                    "prompt_tokens": record.prompt_tokens,
                    "prefill": record.prefill,
                    "gen_tokens": record.gen_tokens,
                    "tps_now": round(record.tps_window(), 2),
                    "tps_avg": round(record.tps_avg(now_perf), 2),
                    "reasoning_chars": record.reasoning_chars,
                    "content_chars": record.content_chars,
                    "accepted_by_depth": (record.live_depth or {}).get("accepted_by_depth"),
                    "drafted_by_depth": (record.live_depth or {}).get("drafted_by_depth"),
                    "stalled_s": (
                        round(now_perf - last_perf, 1) if last_perf is not None else None
                    ),
                    "tail": record.tail_text(),
                }
            )
        return {
            "enabled": self.enabled,
            "file": self.path,
            "text_mode": self.text_mode,
            "lines_written": self._lines_written,
            "active": active,
            "recent": recent,
        }


def resolve_flight_recorder(args: Any) -> FlightRecorder:
    """Build the process recorder from --flight-recorder / MTPLX_FLIGHT_RECORDER
    (off|on|<path>; default ON at ~/.mtplx/metrics/flight-<port>.jsonl) and
    MTPLX_FLIGHT_TEXT (abnormal|always|off; default abnormal)."""
    raw = getattr(args, "flight_recorder", None) or os.environ.get(
        "MTPLX_FLIGHT_RECORDER"
    )
    raw = str(raw or "").strip()
    text_mode = (os.environ.get("MTPLX_FLIGHT_TEXT") or "abnormal").strip().lower()
    if raw.lower() in {"0", "off", "false", "no", "none", "disabled"}:
        return FlightRecorder(None, text_mode=text_mode)
    if raw.lower() in {"1", "on", "true", "yes", "enabled"}:
        raw = ""
    if raw:
        return FlightRecorder(raw, text_mode=text_mode)
    try:
        port = int(getattr(args, "port", 0) or 0)
        metrics_dir = os.path.join(os.path.expanduser("~"), ".mtplx", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        return FlightRecorder(
            os.path.join(metrics_dir, f"flight-{port}.jsonl"), text_mode=text_mode
        )
    except Exception:
        return FlightRecorder(None, text_mode=text_mode)
