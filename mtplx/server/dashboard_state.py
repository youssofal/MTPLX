"""In-process pub/sub + in-flight + rolling-window primitives for the dashboard.

These types are deliberately decoupled from FastAPI and from MLX. They live
on ``ServerState`` and are written/read from both sync (generation worker
threads) and async (SSE coroutine) contexts. Thread-safety notes:

- ``MetricsBus.publish`` is non-blocking and safe to call from any thread.
  It uses ``loop.call_soon_threadsafe`` when an event loop is provided so
  asyncio queues receive events without locking the generation thread.
- ``InFlightRegistry`` and ``RollingMetrics`` use a single ``threading.Lock``
  for all mutations because they are mutated from the generation thread and
  read from the asyncio thread. The reads always copy out plain dicts/lists.
- ``LifetimeCounters`` uses atomic integer increments under the same lock.

The hot generation path must never block on the bus. Subscribers with full
queues drop events silently rather than back-pressuring the publisher; the
dashboard's 200ms snapshot cadence means a missed progress event is at
worst one frame of stale visualization, which is acceptable.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any


# ---- MetricsBus -----------------------------------------------------------


class MetricsBus:
    """Non-blocking pub/sub of dashboard events across threads.

    Event shape is a plain ``dict``: ``{"kind": "progress" | "completed" |
    "new_max_tps" | "cache_evict" | "thermal", ...}``. The bus owns the loop
    binding so workers in sync threads can publish without juggling loop
    references. Subscribers receive their own bounded queue and must call
    ``unsubscribe`` to release it.
    """

    def __init__(self, *, max_queue_size: int = 256) -> None:
        self._max_queue_size = int(max_queue_size)
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.dropped_events_total: int = 0

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio loop used to thread-safely deliver events.

        Called once during FastAPI startup. Subsequent calls update the loop
        binding (useful for tests that recreate the app)."""

        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue_size)
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def publish(self, event: dict[str, Any]) -> None:
        """Publish an event. Non-blocking. Drops on any subscriber's full queue."""

        with self._lock:
            subscribers = list(self._subscribers)
        if not subscribers:
            return
        loop = self._loop
        for queue in subscribers:
            if loop is None:
                # Pre-startup path: best-effort direct put. This only runs
                # before ``attach_loop`` is called, which in production never
                # happens because workers cannot start before the FastAPI
                # lifespan ``yield``s.
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    self.dropped_events_total += 1
                continue
            try:
                loop.call_soon_threadsafe(_safe_queue_put, queue, event, self)
            except RuntimeError:
                # Loop was already closed; treat as dropped.
                self.dropped_events_total += 1


def _safe_queue_put(
    queue: asyncio.Queue[dict[str, Any]],
    event: dict[str, Any],
    bus: MetricsBus,
) -> None:
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        bus.dropped_events_total += 1


# ---- InFlightRegistry -----------------------------------------------------


@dataclass
class InFlightHandle:
    """Per-request handle tracked by ``InFlightRegistry``.

    The ``cancel_event`` is the same per-request ``threading.Event`` the
    generation worker checks via ``_raise_if_stream_cancelled``; storing it
    here lets external clients (``POST /v1/mtplx/cancel/{id}``) trip the
    same flag without reaching into the request coroutine.

    ``prefill_state`` carries the in-progress prompt-eval status used by
    the dashboard's PrefillPanel: ``{phase, tokens_done, tokens_total,
    elapsed_s, prefill_tok_s, started_s}``. None means "no prefill seen
    yet" (request just started or prefill already completed and decode
    is running).
    """

    request_id: str
    cancel_event: threading.Event
    started_s: float
    session_id: str | None = None
    model: str | None = None
    prompt_preview: str = ""
    prompt_tokens: int | None = None
    last_progress: dict[str, Any] = field(default_factory=dict)
    prefill_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "started_s": self.started_s,
            "age_s": max(0.0, time.time() - self.started_s),
            "session_id": self.session_id,
            "model": self.model,
            "prompt_preview": self.prompt_preview,
            "prompt_tokens": self.prompt_tokens,
            "last_progress": dict(self.last_progress),
            "prefill_state": dict(self.prefill_state) if self.prefill_state else None,
            "cancelled": bool(self.cancel_event.is_set()),
        }


class InFlightRegistry:
    """Tracks active generation requests for cancellation + a live panel.

    The registry is thread-safe and exposes a stale-handle reaper so
    bugs that leak handles cannot starve the dashboard's request panel.
    """

    STALE_AFTER_S = 3600.0  # 1 hour

    def __init__(self) -> None:
        self._handles: dict[str, InFlightHandle] = {}
        self._lock = threading.Lock()

    def register(self, handle: InFlightHandle) -> None:
        with self._lock:
            self._handles[handle.request_id] = handle

    def deregister(self, request_id: str) -> None:
        with self._lock:
            self._handles.pop(request_id, None)

    def get(self, request_id: str) -> InFlightHandle | None:
        with self._lock:
            return self._handles.get(request_id)

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            handle = self._handles.get(request_id)
        if handle is None:
            return False
        # Duck-typed: attributed events record that THIS trip really came
        # from the POST /v1/mtplx/cancel endpoint, so the stream's terminal
        # frame can stop blaming the endpoint for internal cancels (#381).
        set_origin = getattr(handle.cancel_event, "set_origin", None)
        if set_origin is not None:
            set_origin("post_endpoint")
        else:
            handle.cancel_event.set()
        return True

    def update_progress(self, request_id: str, progress: dict[str, Any]) -> None:
        with self._lock:
            handle = self._handles.get(request_id)
            if handle is not None:
                handle.last_progress = dict(progress)

    def update_prefill(self, request_id: str, prefill: dict[str, Any] | None) -> None:
        with self._lock:
            handle = self._handles.get(request_id)
            if handle is not None:
                handle.prefill_state = (
                    dict(prefill) if prefill is not None else None
                )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            self._reap_stale_locked()
            return [handle.to_dict() for handle in self._handles.values()]

    def count(self) -> int:
        with self._lock:
            return len(self._handles)

    def session_ids(self) -> list[str]:
        """Session ids of live requests (memory guard: these sessions'
        bank entries must keep dynamic-ceiling protection even when one
        turn outlives the bank's activity-pin TTL)."""
        with self._lock:
            return [
                handle.session_id
                for handle in self._handles.values()
                if handle.session_id
            ]

    def _reap_stale_locked(self) -> None:
        now = time.time()
        stale = [
            rid
            for rid, handle in self._handles.items()
            if (now - handle.started_s) > self.STALE_AFTER_S
        ]
        for rid in stale:
            self._handles.pop(rid, None)


# ---- RollingMetrics -------------------------------------------------------


@dataclass
class _TPSPoint:
    when_s: float
    tok_s: float
    session_id: str | None


class RollingMetrics:
    """5-minute window of decode TPS with per-session and all-time maxes.

    ``append`` is called once per completed generation. ``observe_progress``
    is called from ``on_tokens`` mid-generation so the live TPS gauge has
    sub-second freshness without waiting for the request to finish. Progress
    samples intentionally do not update the sticky all-time max: early
    short-window bursts are useful for a live gauge, but they are not a
    truthful completed-request record.

    ``sticky_all_time_max`` survives session resets but resets when the
    server restarts; we accept that as the "live dashboard" contract.
    """

    WINDOW_S = 300.0  # 5 minutes
    LIVE_SAMPLE_MIN_INTERVAL_S = 0.75
    LIVE_HISTORY_MAX_POINTS = 240

    MAX_PER_SESSION_ENTRIES = 64

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: deque[_TPSPoint] = deque()
        self._live_points: deque[_TPSPoint] = deque()
        # LRU-bounded: agent clients mint fresh session ids freely, so an
        # unpruned per-session map grows for the daemon's lifetime and is
        # copied whole into every dashboard snapshot.
        self._max_per_session: OrderedDict[str, float] = OrderedDict()
        self._sticky_all_time_max: float = 0.0
        self._sticky_all_time_max_when_s: float = 0.0
        self._sticky_all_time_max_session_id: str | None = None
        self._all_time_min: float | None = None
        self._last_live_sample_s: float = 0.0
        self._last_live_sample_session_id: str | None = None

    def append(self, tok_s: float, session_id: str | None) -> bool:
        """Record a completed request's decode TPS. Returns True iff this
        sample set a new sticky-all-time max."""

        if tok_s is None or tok_s <= 0:
            return False
        is_new_max = False
        now = time.time()
        with self._lock:
            self._points.append(_TPSPoint(when_s=now, tok_s=float(tok_s), session_id=session_id))
            self._evict_old_locked(now)
            if session_id:
                prev = self._max_per_session.get(session_id, 0.0)
                if float(tok_s) > prev:
                    self._max_per_session[session_id] = float(tok_s)
                self._max_per_session.move_to_end(session_id)
                while len(self._max_per_session) > self.MAX_PER_SESSION_ENTRIES:
                    self._max_per_session.popitem(last=False)
            if float(tok_s) > self._sticky_all_time_max:
                self._sticky_all_time_max = float(tok_s)
                self._sticky_all_time_max_when_s = now
                self._sticky_all_time_max_session_id = session_id
                is_new_max = True
            if self._all_time_min is None or float(tok_s) < self._all_time_min:
                self._all_time_min = float(tok_s)
        return is_new_max

    def observe_progress(self, tok_s: float, session_id: str | None) -> bool:
        """Record a mid-generation TPS sample.

        These samples stay on the separate live deque so the dashboard can
        render a fresh gauge without polluting completed-request records or
        firing "new speed record" toasts from a transient early burst.
        """

        if tok_s is None or tok_s <= 0:
            return False
        now = time.time()
        with self._lock:
            if (
                self._live_points
                and session_id == self._last_live_sample_session_id
                and now - self._last_live_sample_s < self.LIVE_SAMPLE_MIN_INTERVAL_S
            ):
                return False
            self._live_points.append(_TPSPoint(when_s=now, tok_s=float(tok_s), session_id=session_id))
            self._last_live_sample_s = now
            self._last_live_sample_session_id = session_id
            # Live deque is much chattier than completed appends; keep a
            # bounded window so snapshots, Swift decoding, and Apple Charts
            # stay constant-cost across long benchmark runs.
            cutoff = now - self.WINDOW_S
            while self._live_points and self._live_points[0].when_s < cutoff:
                self._live_points.popleft()
            while len(self._live_points) > self.LIVE_HISTORY_MAX_POINTS:
                self._live_points.popleft()
        return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            self._evict_old_locked(now)
            values = [p.tok_s for p in self._points]
            history = [
                {"t": p.when_s, "tok_s": p.tok_s, "session_id": p.session_id}
                for p in self._points
            ]
            live_history = [
                {"t": p.when_s, "tok_s": p.tok_s, "session_id": p.session_id}
                for p in self._live_points
            ]
            return {
                "window_s": self.WINDOW_S,
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": (sum(values) / len(values)) if values else None,
                "p50": _percentile(values, 50.0),
                "p95": _percentile(values, 95.0),
                "history": history,
                "live_history": live_history,
                "max_per_session": dict(self._max_per_session),
                "sticky_all_time_max": self._sticky_all_time_max,
                "sticky_all_time_max_when_s": self._sticky_all_time_max_when_s,
                "sticky_all_time_max_session_id": self._sticky_all_time_max_session_id,
                "all_time_min": self._all_time_min,
            }

    def session_ids_recent(self, *, since_s: float = 3600.0) -> list[str]:
        """Distinct session ids that completed a request within ``since_s``."""

        cutoff = time.time() - since_s
        with self._lock:
            return sorted(
                {
                    p.session_id
                    for p in self._points
                    if p.session_id and p.when_s >= cutoff
                }
            )

    def _evict_old_locked(self, now: float) -> None:
        cutoff = now - self.WINDOW_S
        while self._points and self._points[0].when_s < cutoff:
            self._points.popleft()


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo_idx = int(rank)
    hi_idx = min(len(sorted_values) - 1, lo_idx + 1)
    frac = rank - lo_idx
    return sorted_values[lo_idx] * (1 - frac) + sorted_values[hi_idx] * frac


# ---- LifetimeCounters -----------------------------------------------------


class LifetimeCounters:
    """Process-lifetime token and request counters surfaced as a hero tile."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at_s: float = time.time()
        self.prompt_tokens_total: int = 0
        self.completion_tokens_total: int = 0
        self.cached_tokens_total: int = 0
        self.requests_total: int = 0
        self.cancelled_total: int = 0

    def record_completion(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
    ) -> None:
        with self._lock:
            self.prompt_tokens_total += int(prompt_tokens or 0)
            self.completion_tokens_total += int(completion_tokens or 0)
            self.cached_tokens_total += int(cached_tokens or 0)
            self.requests_total += 1

    def record_cancellation(self) -> None:
        with self._lock:
            self.cancelled_total += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at_s": self.started_at_s,
                "uptime_s": max(0.0, time.time() - self.started_at_s),
                "prompt_tokens_total": self.prompt_tokens_total,
                "completion_tokens_total": self.completion_tokens_total,
                "cached_tokens_total": self.cached_tokens_total,
                "tokens_total": self.prompt_tokens_total + self.completion_tokens_total,
                "requests_total": self.requests_total,
                "cancelled_total": self.cancelled_total,
            }


# ---- PrefillHistory -------------------------------------------------------


class PrefillHistory:
    """Bounded ring of recent prefill rows for the prefill TPS sparkline."""

    def __init__(self, *, capacity: int = 100) -> None:
        self._capacity = int(capacity)
        self._rows: deque[dict[str, Any]] = deque(maxlen=self._capacity)
        # Producer-side samples: polling/SSE consumers must not count the
        # same chunk again. Keep actual compute separate from restore/setup.
        self._chunks: deque[tuple[int, float]] = deque(maxlen=self._capacity)
        self._lock = threading.Lock()

    def record_chunk(self, tokens: int, elapsed_s: float) -> None:
        if tokens <= 0 or not math.isfinite(elapsed_s) or elapsed_s <= 0:
            return
        with self._lock:
            self._chunks.append((tokens, elapsed_s))

    def rates(self) -> dict[str, Any]:
        with self._lock:
            chunks = list(self._chunks)
        return {
            "tokens": sum(tokens for tokens, _ in chunks),
            "compute_time_s": sum(seconds for _, seconds in chunks),
            "peak_tok_s": max((tokens / seconds for tokens, seconds in chunks), default=None),
            "samples": len(chunks),
            "capacity": self._capacity,
        }

    def append(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._rows.append(dict(row))

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._rows)

    def capacity(self) -> int:
        return self._capacity


# ---- ProgressEventGate ----------------------------------------------------


@dataclass
class ProgressPublishStats:
    """Bounded per-request accounting for live progress overhead."""

    published_events: int = 0
    throttled_events: int = 0
    last_completion_tokens: int = 0
    decision_time_s: float = 0.0
    registry_update_time_s: float = 0.0
    rolling_update_time_s: float = 0.0
    bus_publish_time_s: float = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "dashboard_progress_published_events": self.published_events,
            "dashboard_progress_throttled_events": self.throttled_events,
            "dashboard_progress_last_completion_tokens": self.last_completion_tokens,
            "dashboard_progress_decision_time_s": self.decision_time_s,
            "dashboard_progress_registry_update_time_s": (
                self.registry_update_time_s
            ),
            "dashboard_progress_rolling_update_time_s": self.rolling_update_time_s,
            "dashboard_progress_bus_publish_time_s": self.bus_publish_time_s,
        }


class ProgressEventGate:
    """Per-request throttle for dashboard progress SSE events.

    Generation can produce token progress far faster than a native UI should
    redraw. Progress is published at UI-frame cadence plus early token
    milestones so AIME can prove question-start TPS without per-token work on
    the generation path.
    """

    MIN_INTERVAL_S = 0.20
    MILESTONE_TOKENS = (1, 32, 64, 128, 256)
    STEADY_MILESTONE_STRIDE = 256

    def __init__(self) -> None:
        self._last_publish_s: dict[str, float] = {}
        self._last_publish_tokens: dict[str, int] = {}
        self._stats: dict[str, ProgressPublishStats] = {}
        self._lock = threading.Lock()

    def should_publish(
        self,
        request_id: str,
        *,
        now: float | None = None,
        completion_tokens: int | None = None,
    ) -> bool:
        now_s = time.time() if now is None else float(now)
        tokens = max(0, int(completion_tokens or 0))
        with self._lock:
            last_s = self._last_publish_s.get(request_id)
            last_tokens = self._last_publish_tokens.get(request_id, 0)
            if last_s is None:
                self._last_publish_s[request_id] = now_s
                self._last_publish_tokens[request_id] = tokens
                return True
            if self._crossed_milestone(last_tokens, tokens):
                self._last_publish_s[request_id] = now_s
                self._last_publish_tokens[request_id] = tokens
                return True
            if now_s - last_s >= self.MIN_INTERVAL_S:
                self._last_publish_s[request_id] = now_s
                self._last_publish_tokens[request_id] = max(last_tokens, tokens)
                return True
            return False

    def forget(self, request_id: str) -> None:
        with self._lock:
            self._last_publish_s.pop(request_id, None)
            self._last_publish_tokens.pop(request_id, None)
            self._stats.pop(request_id, None)

    def record_overhead(
        self,
        request_id: str,
        *,
        published: bool,
        completion_tokens: int | None = None,
        decision_time_s: float = 0.0,
        registry_update_time_s: float = 0.0,
        rolling_update_time_s: float = 0.0,
        bus_publish_time_s: float = 0.0,
    ) -> None:
        tokens = max(0, int(completion_tokens or 0))
        with self._lock:
            stats = self._stats.setdefault(request_id, ProgressPublishStats())
            if published:
                stats.published_events += 1
            else:
                stats.throttled_events += 1
            stats.last_completion_tokens = max(stats.last_completion_tokens, tokens)
            stats.decision_time_s += max(0.0, float(decision_time_s or 0.0))
            stats.registry_update_time_s += max(
                0.0, float(registry_update_time_s or 0.0)
            )
            stats.rolling_update_time_s += max(
                0.0, float(rolling_update_time_s or 0.0)
            )
            stats.bus_publish_time_s += max(0.0, float(bus_publish_time_s or 0.0))

    def stats_for(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            stats = self._stats.get(request_id)
            return stats.to_public_dict() if stats is not None else {}

    @classmethod
    def _crossed_milestone(cls, last_tokens: int, current_tokens: int) -> bool:
        if current_tokens <= last_tokens:
            return False
        for milestone in cls.MILESTONE_TOKENS:
            if last_tokens < milestone <= current_tokens:
                return True
        return (
            current_tokens >= cls.STEADY_MILESTONE_STRIDE
            and current_tokens // cls.STEADY_MILESTONE_STRIDE
            > last_tokens // cls.STEADY_MILESTONE_STRIDE
        )


# ---- DashboardState (umbrella) -------------------------------------------


@dataclass
class DashboardState:
    """Bundle of dashboard primitives attached to ``ServerState``."""

    bus: MetricsBus = field(default_factory=MetricsBus)
    in_flight: InFlightRegistry = field(default_factory=InFlightRegistry)
    rolling: RollingMetrics = field(default_factory=RollingMetrics)
    lifetime: LifetimeCounters = field(default_factory=LifetimeCounters)
    prefill_history: PrefillHistory = field(default_factory=PrefillHistory)
    progress_events: ProgressEventGate = field(default_factory=ProgressEventGate)
    last_thermal: dict[str, Any] | None = None
    last_thermal_when_s: float = 0.0
    # macOS kern.memorystatus_vm_pressure_level: 1 normal, 2 warning,
    # 4 critical, 0 unknown. Written by the memory-pressure guard loop.
    last_memory_pressure_level: int = 0
    # Which signal produced the level above: "macos" (system-wide — often
    # another process allocating) or "allocator" (this engine's Metal
    # active+cache near its limit). Lets the app banner name the culprit
    # instead of implying the engine is misbehaving under an external storm.
    last_memory_pressure_source: str = "macos"
    # (active+cache)/metal-limit at the same tick, for the banner detail.
    last_allocator_fraction: float = 0.0
