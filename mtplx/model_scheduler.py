"""Single-owner model work scheduler for MTPLX serving.

The scheduler deliberately runs model work on one thread because MLX stream
state and live cache references are thread-affine on Apple Silicon. It still
keeps request admission explicit: foreground generation has priority over idle
maintenance work such as SessionBank postcommit snapshots.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, deque

from . import progress_heartbeat
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Thread, get_ident
import time
from typing import Any, Callable

_QOS_CLASSES = {
    "user_interactive": 0x21,
    "user_initiated": 0x19,
    "default": 0x15,
    "utility": 0x11,
    "background": 0x09,
}


def _pin_owner_thread_qos() -> str | None:
    """Raise the model owner thread's macOS QoS class (Darwin, best-effort).

    Python threads start at QOS_CLASS_DEFAULT, which the scheduler ranks
    below every user-interactive app thread — on a busy Mac (Electron
    renderers, WindowServer compositing) the decode loop gets preempted
    between Metal submissions and 32k decode drops from ~43 to ~27 tok/s
    (measured 2026-07-16, load 6.7 vs 13). USER_INITIATED marks in-flight
    generation as work the user is waiting on without competing with UI
    event handling the way USER_INTERACTIVE would.

    MTPLX_GENERATION_QOS: user_interactive | user_initiated (default) |
    default | utility | background | off.
    """
    raw = os.environ.get("MTPLX_GENERATION_QOS", "user_initiated").strip().lower()
    if raw in {"off", "none", "0", "false"}:
        return None
    qos = _QOS_CLASSES.get(raw)
    if qos is None or sys.platform != "darwin":
        return None
    try:
        import ctypes

        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        if int(libsystem.pthread_set_qos_class_self_np(qos, 0)) == 0:
            return raw
        return None
    except Exception:
        return None


@dataclass
class _WorkItem:
    kind: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: Future
    sequence: int
    batch_key: str | None = None
    queued_at_s: float = field(default_factory=time.monotonic)
    earliest_start_s: float = field(default_factory=time.monotonic)


def _batch_key_class(batch_key: str) -> str:
    """Stable telemetry class for a batch key: the prefix before the first ':'.

    Keys like ``postcommit:{session_id}`` carry a per-session suffix that is
    useful as the *active* diagnostic but would grow the started-by counter by
    one entry per session for the daemon's lifetime."""
    return batch_key.split(":", 1)[0]


class ModelWorkScheduler:
    """Priority admission scheduler for the single MLX/model owner thread."""

    def __init__(
        self,
        *,
        name: str = "mtplx-model",
        idle_grace_s: float = 0.025,
    ) -> None:
        self.name = str(name)
        self.idle_grace_s = max(0.0, float(idle_grace_s))
        self._condition = Condition()
        self._foreground: deque[_WorkItem] = deque()
        self._idle: deque[_WorkItem] = deque()
        self._sequence = 0
        self._shutdown = False
        self._active_kind: str | None = None
        self._owner_thread_id: int | None = None
        self._started = 0
        self._completed = 0
        self._cancelled_before_start = 0
        self._request_cancelled = 0
        self._completed_by_kind: Counter[str] = Counter()
        self._started_by_batch_key: Counter[str] = Counter()
        self._batch_histogram: Counter[int] = Counter()
        self._queue_wait_samples_s: deque[float] = deque(maxlen=256)
        self._run_duration_samples_s: deque[float] = deque(maxlen=256)
        self._cancellation_latency_samples_s: deque[float] = deque(maxlen=256)
        self._active_sequence: int | None = None
        self._active_batch_key: str | None = None
        self._active_started_at_s: float | None = None
        self._active_queue_wait_s: float | None = None
        self.owner_qos: str | None = None
        self._thread = Thread(
            target=self._run,
            name=f"{self.name}-owner",
            daemon=True,
        )
        self._thread.start()

    @property
    def owner_thread_id(self) -> int | None:
        return self._owner_thread_id

    def is_owner_thread(self) -> bool:
        return self._owner_thread_id == get_ident()

    def foreground_pending(self) -> int:
        with self._condition:
            return len(self._foreground)

    def has_foreground_pending(self) -> bool:
        return self.foreground_pending() > 0

    def foreground_pending_or_active(self) -> bool:
        with self._condition:
            return bool(self._foreground) or self._active_kind == "foreground"

    def any_pending_or_active(self) -> bool:
        """True while the owner thread is executing or has queued work of any
        kind (foreground or idle postcommit). Used as the model-activity
        signal for the smart-fan stale-lease reconciler."""
        with self._condition:
            return (
                bool(self._foreground)
                or bool(self._idle)
                or self._active_kind is not None
            )

    def stats(self) -> dict[str, Any]:
        with self._condition:
            active_run_s = (
                max(0.0, time.monotonic() - self._active_started_at_s)
                if self._active_started_at_s is not None
                else None
            )
            return {
                "foreground_pending": len(self._foreground),
                "idle_pending": len(self._idle),
                "active_kind": self._active_kind,
                "active_sequence": self._active_sequence,
                "active_batch_key": self._active_batch_key,
                "active_run_s": active_run_s,
                "active_queue_wait_s": self._active_queue_wait_s,
                "started": self._started,
                "completed": self._completed,
                "cancelled_before_start": self._cancelled_before_start,
                "request_cancelled": self._request_cancelled,
                "completed_by_kind": dict(self._completed_by_kind),
                "started_by_batch_key": dict(self._started_by_batch_key),
                "batch_histogram": {
                    str(size): count
                    for size, count in sorted(self._batch_histogram.items())
                },
                "queue_wait_s": _sample_summary(self._queue_wait_samples_s),
                "run_duration_s": _sample_summary(self._run_duration_samples_s),
                "cancellation_latency_s": _sample_summary(
                    self._cancellation_latency_samples_s
                ),
                "owner_thread_id": self._owner_thread_id,
                "shutdown": self._shutdown,
            }

    def record_request_cancelled(self, *, latency_s: float | None = None) -> None:
        """Record a user-facing cancellation signal.

        The in-flight registry owns the actual cancel event; this method keeps
        the scheduler telemetry envelope complete without coupling the two
        subsystems together.
        """

        with self._condition:
            self._request_cancelled += 1
            if latency_s is not None:
                self._cancellation_latency_samples_s.append(max(0.0, float(latency_s)))

    def record_batch_step(self, *, size: int, batch_key: str | None = None) -> None:
        """Record a model-owner microbatch executed inside a long-lived pump."""

        progress_heartbeat.tick()
        with self._condition:
            self._batch_histogram[max(1, int(size))] += 1
            if batch_key:
                self._started_by_batch_key[_batch_key_class(str(batch_key))] += 1

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """ThreadPoolExecutor-compatible foreground submit."""
        return self._submit(
            "foreground",
            fn,
            args=args,
            kwargs=kwargs,
            batch_key=None,
            earliest_start_s=time.monotonic(),
        )

    def submit_foreground(
        self,
        fn: Callable[..., Any],
        *args: Any,
        batch_key: str | None = None,
        **kwargs: Any,
    ) -> Future:
        return self._submit(
            "foreground",
            fn,
            args=args,
            kwargs=kwargs,
            batch_key=batch_key,
            earliest_start_s=time.monotonic(),
        )

    def submit_idle_postcommit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        batch_key: str | None = None,
        **kwargs: Any,
    ) -> Future:
        return self._submit(
            "idle_postcommit",
            fn,
            args=args,
            kwargs=kwargs,
            batch_key=batch_key,
            earliest_start_s=time.monotonic() + self.idle_grace_s,
        )

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._condition:
            self._shutdown = True
            if cancel_futures:
                for queue in (self._foreground, self._idle):
                    while queue:
                        item = queue.popleft()
                        item.future.cancel()
            self._condition.notify_all()
        if wait and self._thread.is_alive():
            self._thread.join()

    def _submit(
        self,
        kind: str,
        fn: Callable[..., Any],
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        batch_key: str | None,
        earliest_start_s: float,
    ) -> Future:
        future: Future = Future()
        with self._condition:
            if self._shutdown:
                future.set_exception(RuntimeError("model scheduler is shut down"))
                return future
            self._sequence += 1
            item = _WorkItem(
                kind=kind,
                fn=fn,
                args=args,
                kwargs=kwargs,
                future=future,
                sequence=self._sequence,
                batch_key=batch_key,
                earliest_start_s=earliest_start_s,
            )
            if kind == "foreground":
                self._foreground.append(item)
            else:
                self._idle.append(item)
            self._condition.notify_all()
        return future

    def _run(self) -> None:
        self._owner_thread_id = get_ident()
        self.owner_qos = _pin_owner_thread_qos()
        while True:
            item = self._take_next()
            if item is None:
                return
            if not item.future.set_running_or_notify_cancel():
                with self._condition:
                    self._cancelled_before_start += 1
                continue
            now = time.monotonic()
            queue_wait_s = max(0.0, now - item.queued_at_s)
            with self._condition:
                self._active_kind = (
                    "foreground" if item.kind == "foreground" else "idle_postcommit"
                )
                self._active_sequence = item.sequence
                self._active_batch_key = item.batch_key
                self._active_started_at_s = now
                self._active_queue_wait_s = queue_wait_s
                self._queue_wait_samples_s.append(queue_wait_s)
                self._started += 1
                self._started_by_batch_key[
                    _batch_key_class(item.batch_key) if item.batch_key else "none"
                ] += 1
            try:
                item.future.set_result(item.fn(*item.args, **item.kwargs))
            except BaseException as exc:
                item.future.set_exception(exc)
            finally:
                progress_heartbeat.tick()
                run_duration_s = max(0.0, time.monotonic() - now)
                with self._condition:
                    self._completed += 1
                    self._completed_by_kind[item.kind] += 1
                    self._batch_histogram[1] += 1
                    self._run_duration_samples_s.append(run_duration_s)
                    self._active_kind = None
                    self._active_sequence = None
                    self._active_batch_key = None
                    self._active_started_at_s = None
                    self._active_queue_wait_s = None
                    self._condition.notify_all()

    def _take_next(self) -> _WorkItem | None:
        with self._condition:
            while True:
                if self._shutdown and not self._foreground and not self._idle:
                    return None
                if self._foreground:
                    return self._foreground.popleft()
                if self._idle:
                    now = time.monotonic()
                    delay = self._idle[0].earliest_start_s - now
                    if delay <= 0:
                        return self._idle.popleft()
                    self._condition.wait(timeout=delay)
                    continue
                self._condition.wait()


def _sample_summary(samples: deque[float]) -> dict[str, float | int | None]:
    values = list(samples)
    if not values:
        return {
            "count": 0,
            "latest": None,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": len(values),
        "latest": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(len(ordered) - 1, lo + 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac
