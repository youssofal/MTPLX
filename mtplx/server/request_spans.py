"""Where a request's time to first token goes, span by span.

One clock per request starts when the HTTP request reaches the server: an ASGI
middleware stamps the arrival before routing, the body read, the JSON parse and
the pydantic validation, which the handler's own clock never saw. The request
path then takes named marks (policy resolved, prompt encoded, session resolved,
the previous turn's postcommit waited for, the owner thread started the work,
the model lock taken, the memory admission passed, the first token produced,
the first delta written to the stream). A span is the time between one mark
and the mark before it, named by the later mark. The spans of a request
therefore add up to its time to first token by construction: a slow step can
only show up inside a named span, never between two instruments. The client's
TTFT minus the span sum is only what happens before the server sees the
request and after it writes the first delta (the socket and the client).

Durations that are not on the sequential path are reported beside the spans as
details: where the owner-thread queue wait came from (the item that was running
when this request was submitted), what the engine spent inside its span (the
restore, the SSD read, the prefill forwards, the first sample), and any attempt
a repair or retry path discarded before the one the client received.

Marks live in a small registry keyed by the request id, because the path
crosses the event loop, ``asyncio.to_thread`` workers and the model owner
thread, and the observability dict it could otherwise ride is copied and
serialized along the way.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Any, Iterable, Mapping

ARRIVAL_STATE_KEY = "mtplx_arrival_s"

# The marks, in path order: http_parse, policy, encode, session_resolve,
# postcommit_sweep, postcommit_wait, canonicalize, prologue, dispatch,
# scheduler_queue, lock_wait, admission, engine_first_token, first_delta_sent.
# ``dispatch`` covers the stream start, the worker thread and the session's
# generation slot; ``scheduler_queue`` the owner-thread queue. A mark a lane
# never takes merges its time into the next span that is taken; the sum does
# not change. Spans are ordered by time, not by this list.
#
# Marks one generation attempt takes; a discarded attempt drops them so the
# attempt that reaches the client re-marks its own.
_ATTEMPT_MARKS = frozenset(
    {"scheduler_queue", "lock_wait", "admission", "engine_first_token"}
)
_TTFT_ENDPOINTS = ("first_delta_sent", "engine_first_token")

_LIVE_MAX = 256
_WINDOW_MAX = 128


class RequestArrivalClock:
    """Pure ASGI middleware: stamp the moment an HTTP request arrives.

    Registered outermost so the stamp precedes every other middleware, the
    body read and validation. It only writes one float into the scope's state
    dict (what ``Request.state`` reads), so the response stream passes through
    untouched at zero per-chunk cost.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            state = scope.get("state")
            if state is None:
                state = {}
                scope["state"] = state
            state.setdefault(ARRIVAL_STATE_KEY, time.perf_counter())
        await self.app(scope, receive, send)


def arrival_s(raw_request: Any) -> float | None:
    """The middleware's arrival stamp for a Starlette request, if present."""

    scope = getattr(raw_request, "scope", None)
    if not isinstance(scope, Mapping):
        return None
    state = scope.get("state")
    if not isinstance(state, Mapping):
        return None
    value = state.get(ARRIVAL_STATE_KEY)
    return float(value) if isinstance(value, (int, float)) else None


class RequestClock:
    """Marks and details for one request. Thread-safe: marks arrive from the
    event loop, ``to_thread`` workers and the model owner thread."""

    def __init__(self, origin_s: float, *, origin: str = "http_arrival") -> None:
        self.origin_s = float(origin_s)
        self.origin = origin
        self._lock = threading.Lock()
        self._marks: dict[str, float] = {}
        self._details: dict[str, Any] = {}
        self._attempts = 1
        self._discarded_attempt_wall_s = 0.0
        self._retry_paths: list[str] = []
        self._attempt_anchor_s: float | None = None
        # The /health window row this clock last published, updated in place
        # when the stream refreshes the summary.
        self.window_row: dict[str, Any] | None = None

    def mark(self, name: str, now: float | None = None) -> None:
        """Record ``name`` at ``now`` (default: now). The first mark of a name
        wins: a later call on a second code path cannot move an earlier one."""

        stamp = time.perf_counter() if now is None else float(now)
        with self._lock:
            if name in self._marks:
                return
            self._marks[name] = stamp
            if name == "dispatch" and self._attempt_anchor_s is None:
                self._attempt_anchor_s = stamp

    def has_mark(self, name: str) -> bool:
        with self._lock:
            return name in self._marks

    def detail(self, name: str, value: Any) -> None:
        """Attach a JSON-safe detail (a duration or a small dict)."""

        with self._lock:
            self._details[name] = value

    def discard_attempt(self, path: str, now: float | None = None) -> None:
        """A repair or retry path threw away a whole generation attempt.

        Counts it, adds its wall time to ``discarded_attempt_wall_s``, and,
        when the client had not yet received a delta, replaces the attempt's
        own marks with one ``discarded_attempt`` span so the attempt that
        reaches the client is measured on its own.
        """

        stamp = time.perf_counter() if now is None else float(now)
        with self._lock:
            anchor = self._attempt_anchor_s
            if anchor is None:
                anchor = self._marks.get("dispatch", self.origin_s)
            self._attempts += 1
            self._discarded_attempt_wall_s += max(0.0, stamp - anchor)
            self._retry_paths.append(str(path))
            self._attempt_anchor_s = stamp
            if "first_delta_sent" not in self._marks:
                for name in _ATTEMPT_MARKS:
                    self._marks.pop(name, None)
                # Several discards accumulate into one span.
                self._marks.pop("discarded_attempt", None)
                self._marks["discarded_attempt"] = stamp

    def scheduler_queue_wait_s(self) -> float | None:
        """Time this attempt waited between dispatch and the owner thread
        taking it (the ``scheduler_queue`` span), or None before that mark."""

        with self._lock:
            end = self._marks.get("scheduler_queue")
            start = self._attempt_anchor_s
            if start is None:
                start = self._marks.get("dispatch")
        if end is None or start is None:
            return None
        return max(0.0, end - start)

    @property
    def attempts(self) -> int:
        with self._lock:
            return self._attempts

    def summary(self) -> dict[str, Any]:
        """The published breakdown: ordered spans up to the first token, the
        details, and the attempt receipt."""

        with self._lock:
            marks = dict(self._marks)
            details = dict(self._details)
            attempts = self._attempts
            discarded = self._discarded_attempt_wall_s
            paths = list(self._retry_paths)
        ordered = sorted(marks.items(), key=lambda item: item[1])
        endpoint_name = next(
            (name for name in _TTFT_ENDPOINTS if name in marks), None
        )
        endpoint_s = marks[endpoint_name] if endpoint_name is not None else None
        spans: dict[str, float] = {}
        after: dict[str, float] = {}
        previous = self.origin_s
        for name, stamp in ordered:
            segment = max(0.0, stamp - previous)
            previous = max(previous, stamp)
            if endpoint_s is None or stamp <= endpoint_s:
                spans[f"{name}_s"] = round(segment, 6)
            else:
                after[f"{name}_s"] = round(segment, 6)
        out: dict[str, Any] = {
            "origin": self.origin,
            "endpoint": endpoint_name,
            "ttft_s": (
                round(max(0.0, endpoint_s - self.origin_s), 6)
                if endpoint_s is not None
                else None
            ),
            # Exclusive: no two spans overlap and together they cover the
            # clock from arrival to the endpoint.
            "exclusive_s": spans,
            "sum_s": round(sum(spans.values()), 6),
            "attempts": attempts,
            "discarded_attempt_wall_s": round(discarded, 6),
            "retry_paths": paths,
        }
        if after:
            out["after_first_token"] = after
        if details:
            out["details"] = details
        return out


_LIVE: "OrderedDict[str, RequestClock]" = OrderedDict()
_LIVE_LOCK = threading.Lock()
_WINDOW: deque[dict[str, Any]] = deque(maxlen=_WINDOW_MAX)
_WINDOW_LOCK = threading.Lock()


def open_clock(
    request_id: str | None,
    *,
    arrival: float | None,
    handler_start: float,
) -> RequestClock:
    """Start a request's clock at its HTTP arrival (or at handler entry when
    the middleware did not run) and register it under ``request_id``."""

    if arrival is not None and arrival <= handler_start:
        clock = RequestClock(arrival, origin="http_arrival")
        clock.mark("http_parse", handler_start)
    else:
        clock = RequestClock(handler_start, origin="handler")
    if request_id:
        with _LIVE_LOCK:
            _LIVE[str(request_id)] = clock
            _LIVE.move_to_end(str(request_id))
            while len(_LIVE) > _LIVE_MAX:
                _LIVE.popitem(last=False)
    return clock


def clock_for(request_id: Any) -> RequestClock | None:
    if not request_id:
        return None
    with _LIVE_LOCK:
        return _LIVE.get(str(request_id))


def clock_from_observability(observability: Mapping[str, Any] | None) -> RequestClock | None:
    if not observability:
        return None
    return clock_for(observability.get("request_id"))


def close_clock(request_id: Any, clock: RequestClock | None = None) -> None:
    """Drop the registry entry (only if it is still ``clock``)."""

    if not request_id:
        return
    with _LIVE_LOCK:
        current = _LIVE.get(str(request_id))
        if current is not None and (clock is None or current is clock):
            del _LIVE[str(request_id)]


def mark(observability: Mapping[str, Any] | None, name: str, now: float | None = None) -> None:
    clock = clock_from_observability(observability)
    if clock is not None:
        clock.mark(name, now)


def engine_details(stats: Mapping[str, Any]) -> dict[str, float]:
    """What the engine reported for its span, from the generation stats."""

    out: dict[str, float] = {}
    for key in (
        "cache_restore_time_s",
        "ssd_restore_s",
        "prompt_eval_time_s",
        "prompt_mtp_history_time_s",
        "pre_first_token_setup_s",
        "first_primary_sample_time_s",
    ):
        value = stats.get(key)
        if isinstance(value, (int, float)) and value:
            out[key] = round(float(value), 6)
    return out


def publish(
    clock: RequestClock | None,
    envelope: dict[str, Any],
    *,
    stats: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Write the clock's summary into ``envelope["ttft_spans"]`` and add it to
    the rolling window /health reads."""

    if clock is None:
        return None
    if stats is not None:
        engine = engine_details(stats)
        if engine:
            clock.detail("engine", engine)
    summary = _write(clock, envelope)
    if not bool(envelope.get("warmup")):
        row = dict(summary)
        with _WINDOW_LOCK:
            if clock.window_row is not None:
                # A second attempt of the same request replaces its row.
                clock.window_row.clear()
                clock.window_row.update(row)
            else:
                clock.window_row = row
                _WINDOW.append(row)
    return summary


def refresh(clock: RequestClock | None, stats: dict[str, Any]) -> None:
    """Re-read the clock into a response just before it is written.

    The owner thread publishes when the generation ends; a stream writes its
    first delta on the event loop, which for a very short answer can land
    after that. The stream refreshes its final stats (and the /health row)
    just before it writes them. A response whose generation path did not
    publish (the batched lanes finalize outside ``_run_generation``) is
    published here.
    """

    if clock is None:
        return
    if "ttft_spans" not in stats:
        publish(clock, stats)
        return
    summary = _write(clock, stats)
    with _WINDOW_LOCK:
        if clock.window_row is not None:
            clock.window_row.clear()
            clock.window_row.update(summary)


def _write(clock: RequestClock, target: dict[str, Any]) -> dict[str, Any]:
    summary = clock.summary()
    target["ttft_spans"] = summary
    target["attempts"] = summary["attempts"]
    target["discarded_attempt_wall_s"] = summary["discarded_attempt_wall_s"]
    if summary["retry_paths"]:
        target["retry_path"] = summary["retry_paths"][-1]
    return summary


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(len(ordered) - 1, lo + 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def health_summary(window: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Median and p90 per span over the recent requests, for /health."""

    if window is None:
        with _WINDOW_LOCK:
            rows = list(_WINDOW)
    else:
        rows = list(window)
    if not rows:
        return {"count": 0}
    ttfts = [float(row["ttft_s"]) for row in rows if row.get("ttft_s") is not None]
    names: list[str] = []
    for row in rows:
        for name in row.get("exclusive_s") or {}:
            if name not in names:
                names.append(name)
    spans: dict[str, dict[str, float]] = {}
    for name in names:
        # A request that never took a mark spent nothing in that span.
        values = [
            float((row.get("exclusive_s") or {}).get(name, 0.0)) for row in rows
        ]
        spans[name] = {
            "p50": round(_percentile(values, 50.0), 6),
            "p90": round(_percentile(values, 90.0), 6),
        }
    retried = [row for row in rows if int(row.get("attempts") or 1) > 1]
    return {
        "count": len(rows),
        "ttft_p50_s": round(_percentile(ttfts, 50.0), 6) if ttfts else None,
        "ttft_p90_s": round(_percentile(ttfts, 90.0), 6) if ttfts else None,
        "spans": spans,
        "retried_requests": len(retried),
        "discarded_attempt_wall_s": round(
            sum(float(row.get("discarded_attempt_wall_s") or 0.0) for row in rows), 6
        ),
    }


def reset_for_tests() -> None:
    with _LIVE_LOCK:
        _LIVE.clear()
    with _WINDOW_LOCK:
        _WINDOW.clear()
