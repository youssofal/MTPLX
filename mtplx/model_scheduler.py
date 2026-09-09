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
from threading import Condition, Event, Thread, get_ident
import time
from typing import Any, Callable

_QOS_CLASSES = {
    "user_interactive": 0x21,
    "user_initiated": 0x19,
    "default": 0x15,
    "utility": 0x11,
    "background": 0x09,
}


def _release_mlx_thread_state() -> None:
    """Destroy this thread's MLX streams + thread_local compile cache.

    mlx 0.32.1 (ml-explore/mlx#4248) deleted the GIL-safe pre-finalization
    hooks that used to clear the thread_local compile cache, so a worker
    thread exiting during Py_Finalize runs mlx's TLS destructor into
    _Py_Dealloc on a dead interpreter -> SIGSEGV/SIGTRAP (#303). Upstream
    closed #4327/#4347 WONTFIX: mx.clear_streams() at thread end is the
    permanent contract. ONE-WAY for this thread — only ever the LAST mlx
    action, or later evals raise "There is no Stream(cpu, N)".
    """
    try:
        import mlx.core as mx

        clear_streams = getattr(mx, "clear_streams", None)
        if clear_streams is not None:
            clear_streams()
    except Exception:
        pass


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
    coalesce_key: str | None = None


class _KeepaliveTurn:
    """Sentinel `_take_next` returns when the idle keepalive is due."""


_KEEPALIVE = _KeepaliveTurn()


def _batch_key_class(batch_key: str) -> str:
    """Stable telemetry class for a batch key: the prefix before the first ':'.

    Keys like ``postcommit:{session_id}`` carry a per-session suffix that is
    useful as the *active* diagnostic but would grow the started-by counter by
    one entry per session for the daemon's lifetime."""
    return batch_key.split(":", 1)[0]


class ModelWorkScheduler:
    """Priority admission scheduler for the single MLX/model owner thread.

    Three bands: foreground > idle_postcommit > idle_persistence. The
    persistence band exists for durability work (SSD cold encodes) that a
    latency-critical canonical postcommit must never queue behind — the
    2026-08-06 causal probe showed FIFO idle ordering ran 1-2s encodes
    ahead of the postcommit whose entry anchors the NEXT turn's restore,
    degrading every warm agent turn. Persistence eligibility carries a
    QUIET GRACE anchored to the most recent foreground/postcommit
    COMPLETION (not its own submission time — a grace measured at
    submission expires during a long generation and would release cold
    work in the few-ms gap before the server tail submits its
    postcommit). Running work is never preempted. There is deliberately NO
    max-defer bypass: any age-based valve reopens the race for foregrounds
    longer than the valve (the item is already "overdue" at completion and
    would dequeue in the tail gap). Eventual drain means after a real
    quiet window; continuous latency-critical work is allowed to defer
    background durability — foreground load already makes absolute
    eventuality impossible.
    """

    # Capability marker for server wiring: submit_idle_persistence exists.
    SUPPORTS_IDLE_PERSISTENCE = True

    def __init__(
        self,
        *,
        name: str = "mtplx-model",
        idle_grace_s: float | None = None,
        persistence_quiet_grace_s: float = 0.25,
    ) -> None:
        self.name = str(name)
        if idle_grace_s is None:
            # A serve request is not one scheduler item: restore and
            # prefill/generate arrive as separate foreground submissions
            # with 100-200 ms of handler python (16k-prompt tokenize)
            # between them. The original 25 ms grace let a pending
            # multi-GB SSD encode START inside that micro-gap; its
            # per-tensor abort fired as soon as the next item queued, but
            # the already-submitted GPU evals still had to drain ahead of
            # the request — a discrete ~0.8 s unattributed prompt-state
            # wall on turns with pending encodes (gate254 y-series +
            # native sample, 2026-08-07). 300 ms outlasts the inter-item
            # gap; idle work is seconds-scale, so the added latency to
            # background durability is noise.
            raw = os.environ.get("MTPLX_SCHEDULER_IDLE_GRACE_S", "").strip()
            try:
                idle_grace_s = float(raw) if raw else 0.3
            except ValueError:
                idle_grace_s = 0.3
        self.idle_grace_s = max(0.0, float(idle_grace_s))
        self.persistence_quiet_grace_s = max(0.0, float(persistence_quiet_grace_s))
        self._condition = Condition()
        self._foreground: deque[_WorkItem] = deque()
        self._idle: deque[_WorkItem] = deque()
        self._persistence: deque[_WorkItem] = deque()
        self._persistence_coalesced = 0
        # Idle-pump budget (issue #290): the persistence band is normally
        # reachable only while the idle deque is COMPLETELY empty, so any
        # self-chaining idle_postcommit occupant (each completion enqueues
        # its successor 'idle_grace_s' in the future — the 2.8.2 background
        # warm ladder ran exactly this shape for minutes after the last
        # request) starves durability forever: at every decision point the
        # idle deque is non-empty and every completion re-arms the quiet
        # anchor. The server arms this budget only after it has observed
        # REQUEST-level idleness for several seconds (no in-flight HTTP
        # requests, none recently), which is a window where no foreground
        # tail gap can exist — so an armed pump lets the persistence head
        # bridge the not-yet-ready gaps of a background chain without
        # reopening the 2026-08-06 tail-gap race. A ready idle item still
        # wins, foreground always wins, and any foreground submission
        # disarms the pump immediately.
        self._persistence_pump_budget = 0
        self._persistence_pumped = 0
        self._last_quiet_anchor_s = time.monotonic()
        # Idle keepalive (GPU residency): see arm_idle_keepalive.
        self._keepalive_fn: Callable[[], Any] | None = None
        self._keepalive_interval_s = 0.0
        self._keepalive_attentive_s = 0.0
        self._keepalive_attentive_anchor_s = 0.0
        self._last_owner_activity_s = time.monotonic()
        self._keepalive_beats = 0
        self._keepalive_errors = 0
        self._keepalive_consecutive_errors = 0
        self._keepalive_last_error: str | None = None
        self._keepalive_last_beat_s: float | None = None
        self._keepalive_last_duration_s: float | None = None
        self._sequence = 0
        self._shutdown = False
        self._park_on_exit = False
        self._active_kind: str | None = None
        # Covers the dequeue-to-start gap as well as keepalive work, neither
        # of which is represented by _active_kind throughout its lifetime.
        self._owner_work_claimed = False
        self._owner_thread_id: int | None = None
        self._started = 0
        self._completed = 0
        self._cancelled_before_start = 0
        self._request_cancelled = 0
        self._completed_by_kind: Counter[str] = Counter()
        self._started_by_batch_key: Counter[str] = Counter()
        self._batch_histogram: Counter[int] = Counter()
        # True while the active item has reported real microbatch sizes via
        # record_batch_step — its completion must not also stamp a size-1.
        self._active_self_reported = False
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

    def persistence_pending(self) -> int:
        """Queued (not running) durability items. Cheap: for the server's
        idle pump tick, which must not build the full stats() dict."""
        with self._condition:
            return len(self._persistence)

    def pump_persistence(self, *, budget: int = 1) -> bool:
        """Open up to ``budget`` persistence slots through a busy idle band.

        Issue #290 anti-starvation valve, deliberately NOT age-based (the
        rejected max-defer valve raced the foreground tail gap): the CALLER
        asserts server-level idleness — no in-flight requests and none for
        several seconds — a state in which no latency-critical tail can be
        pending. While armed, the persistence head may run in the gaps
        where the idle deque holds only not-yet-ready items (a chaining
        background occupant); a READY idle item still runs first, the
        completion-anchored quiet grace still applies, and any foreground
        submission disarms the remaining budget before it is admitted.
        Returns True when at least one slot was armed."""
        with self._condition:
            if self._shutdown or not self._persistence:
                return False
            self._persistence_pump_budget = max(
                self._persistence_pump_budget, min(int(budget), 8)
            )
            self._condition.notify_all()
            return True

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
                or bool(self._persistence)
                or self._active_kind is not None
                or self._owner_work_claimed
            )

    def run_idle_maintenance(self, fn: Callable[[], Any]) -> bool:
        """Try a short maintenance callback while all owner work is excluded.

        The caller must first acquire the model lock without blocking and
        recheck request activity inside ``fn``. This gate closes the race
        between an idle snapshot and a newly admitted postcommit, restore,
        persistence or keepalive item. It never waits for scheduler work.
        The callback runs on the caller's thread; it must not execute model
        work or wait for a scheduler future.
        """
        if not self._condition.acquire(blocking=False):
            return False
        try:
            if (
                self._shutdown
                or self._owner_work_claimed
                or self._foreground
                or self._idle
                or self._persistence
                or self._active_kind is not None
            ):
                return False
            fn()
            return True
        finally:
            self._condition.release()

    # MARK: idle keepalive

    def arm_idle_keepalive(
        self,
        fn: Callable[[], Any],
        *,
        interval_s: float,
        attentive_s: float,
    ) -> None:
        """Run ``fn`` on the owner thread whenever it has been idle for
        ``interval_s``, for ``attentive_s`` after the last foreground
        completion (and after arming).

        Purpose: keep the process's GPU working set resident between turns.
        macOS drops an idle process's Metal residency ~2.5 s after its last
        command buffer, and re-establishing it costs ~9 ms per GiB on the
        next submit (measured 2026-09-03, M5 Max: 8 GiB set 15 -> 87 ms;
        the 77 GiB Flash-Next weights ~0.75-1.0 s added to every prefill
        that followed >2.5 s of quiet — every chat turn, every agent
        tool-call round trip). A trivially small kernel on the model's queue
        inside that window holds the residency set, so the next request's
        prefill starts warm.

        The beat runs strictly on the owner thread between items, so it can
        never race model work; a foreground submission wakes the loop and
        wins immediately (the beat itself is sub-millisecond). Outside the
        attentive window the loop parks untimed as before — a daemon nobody
        is talking to costs no wakeups.
        """
        with self._condition:
            self._keepalive_fn = fn
            self._keepalive_interval_s = max(0.05, float(interval_s))
            self._keepalive_attentive_s = max(0.0, float(attentive_s))
            now = time.monotonic()
            self._keepalive_attentive_anchor_s = now
            self._last_owner_activity_s = now
            self._keepalive_consecutive_errors = 0
            self._condition.notify_all()

    def disarm_idle_keepalive(self) -> None:
        with self._condition:
            self._keepalive_fn = None
            self._condition.notify_all()

    def keepalive_state(self) -> dict[str, Any]:
        """Telemetry envelope for /health and the request record."""
        with self._condition:
            return self._keepalive_state_locked(time.monotonic())

    def _keepalive_state_locked(self, now: float) -> dict[str, Any]:
        armed = self._keepalive_fn is not None
        attentive = armed and (
            now - self._keepalive_attentive_anchor_s <= self._keepalive_attentive_s
        )
        return {
            "armed": armed,
            "attentive": attentive,
            "interval_s": self._keepalive_interval_s if armed else None,
            "attentive_s": self._keepalive_attentive_s if armed else None,
            "attentive_remaining_s": (
                max(
                    0.0,
                    self._keepalive_attentive_anchor_s
                    + self._keepalive_attentive_s
                    - now,
                )
                if attentive
                else 0.0
            ),
            "beats": self._keepalive_beats,
            "errors": self._keepalive_errors,
            "last_error": self._keepalive_last_error,
            "last_beat_age_s": (
                max(0.0, now - self._keepalive_last_beat_s)
                if self._keepalive_last_beat_s is not None
                else None
            ),
            "last_beat_duration_s": self._keepalive_last_duration_s,
            # Whether the GPU working set should still be resident right
            # now: the owner thread did something within one interval.
            "warm": armed
            and now - self._last_owner_activity_s <= self._keepalive_interval_s * 1.5,
        }

    def _keepalive_due_locked(self, now: float) -> float | None:
        """Monotonic time of the next beat, or None when no beat is owed."""
        if self._keepalive_fn is None or self._shutdown:
            return None
        if now - self._keepalive_attentive_anchor_s > self._keepalive_attentive_s:
            return None
        return self._last_owner_activity_s + self._keepalive_interval_s

    def _run_keepalive(self) -> None:
        fn = self._keepalive_fn
        if fn is None:
            return
        started = time.monotonic()
        error: str | None = None
        try:
            fn()
        except BaseException as exc:  # never let a beat take the owner thread down
            error = f"{type(exc).__name__}: {exc}"
        finished = time.monotonic()
        with self._condition:
            self._last_owner_activity_s = finished
            self._keepalive_last_beat_s = finished
            self._keepalive_last_duration_s = finished - started
            if error is None:
                self._keepalive_beats += 1
                self._keepalive_consecutive_errors = 0
            else:
                self._keepalive_errors += 1
                self._keepalive_consecutive_errors += 1
                self._keepalive_last_error = error
                if self._keepalive_consecutive_errors >= 3:
                    # A beat that keeps failing is not keeping anything
                    # warm; stop paying for it rather than loop on errors.
                    self._keepalive_fn = None

    def stats(self) -> dict[str, Any]:
        with self._condition:
            active_run_s = (
                max(0.0, time.monotonic() - self._active_started_at_s)
                if self._active_started_at_s is not None
                else None
            )
            return {
                "idle_keepalive": self._keepalive_state_locked(time.monotonic()),
                "foreground_pending": len(self._foreground),
                "idle_pending": len(self._idle),
                "persistence_pending": len(self._persistence),
                "persistence_coalesced": self._persistence_coalesced,
                "persistence_pump_budget": self._persistence_pump_budget,
                "persistence_pumped": self._persistence_pumped,
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
            self._active_self_reported = True
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

    def foreground_busy(self) -> bool:
        """True while a foreground item is queued or running.

        Cooperative signal for idle-band work that runs long inside a single
        work item (SSD cold-tier encode) or off-thread entirely (SSD writer
        file IO): both must stand down while latency-critical traffic is in
        flight. Cheap enough to poll per tensor / per blob write.
        """
        with self._condition:
            return bool(self._foreground) or self._active_kind == "foreground"

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

    def submit_idle_persistence(
        self,
        fn: Callable[..., Any],
        *args: Any,
        batch_key: str | None = None,
        coalesce_key: str | None = None,
        **kwargs: Any,
    ) -> Future:
        """Durability work: strictly below idle_postcommit, quiet-grace
        gated from the most recent foreground/postcommit completion. Drains
        after a real quiet window; deliberately no age-based bypass.

        coalesce_key (optional): newest-wins bound on PENDING work. Each
        queued persistence closure can pin GB-scale state (an SSD encode
        job holds its bank entry's snapshot arrays), and under continuous
        latency-critical load the quiet window may not arrive for many
        turns — unbounded pending closures grew active memory ~19% in the
        2026-08-06 product A/B. Submitting with a key cancels-and-releases
        any PENDING item with the same key (a RUNNING item is never
        cancelled), keeping at most one pending closure per key. The
        superseded future is cancelled; different keys drain
        independently."""
        return self._submit(
            "idle_persistence",
            fn,
            args=args,
            kwargs=kwargs,
            batch_key=batch_key,
            earliest_start_s=time.monotonic() + self.idle_grace_s,
            coalesce_key=coalesce_key,
        )

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
        park: bool = False,
    ) -> None:
        with self._condition:
            self._shutdown = True
            if park:
                # Process is exiting: the owner thread must never
                # pthread_exit (#303 — see _release_mlx_thread_state).
                self._park_on_exit = True
            if cancel_futures:
                for queue in (self._foreground, self._idle, self._persistence):
                    while queue:
                        item = queue.popleft()
                        item.future.cancel()
            self._condition.notify_all()
        if wait and not park and self._thread.is_alive():
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
        coalesce_key: str | None = None,
    ) -> Future:
        future: Future = Future()
        with self._condition:
            if self._shutdown:
                future.set_exception(RuntimeError("model scheduler is shut down"))
                return future
            if kind == "idle_persistence" and coalesce_key is not None:
                # Newest-wins: cancel-and-release any PENDING same-key item
                # before enqueueing. Only queued items are reachable here — a
                # running item was popped from the deque and is never
                # cancelled. Removing the item drops the closure (and the
                # GB-scale entry it pins); the superseded future cancels so
                # any waiter unblocks.
                for stale in list(self._persistence):
                    if stale.coalesce_key == coalesce_key:
                        self._persistence.remove(stale)
                        stale.future.cancel()
                        self._persistence_coalesced += 1
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
                coalesce_key=coalesce_key,
            )
            if kind == "foreground":
                # A request is arriving: its tail postcommit is imminent.
                # Any armed idle-pump slot must not survive into that
                # window (issue #290 valve; tail-gap fence intact).
                self._persistence_pump_budget = 0
                self._foreground.append(item)
            elif kind == "idle_persistence":
                self._persistence.append(item)
            else:
                self._idle.append(item)
            self._condition.notify_all()
        return future

    def _run(self) -> None:
        try:
            self._run_loop()
        finally:
            _release_mlx_thread_state()
            if self._park_on_exit:
                # Park, never exit: a clean thread exit runs pthread TSD
                # cleanup -> mlx thread_local dtors -> _Py_Dealloc during
                # interpreter finalization (#303). A never-set Event blocks
                # in pthread_cond_wait without re-entering the interpreter.
                Event().wait()

    def _run_loop(self) -> None:
        self._owner_thread_id = get_ident()
        self.owner_qos = _pin_owner_thread_qos()
        while True:
            item = self._take_next()
            if item is None:
                return
            if item is _KEEPALIVE:
                try:
                    self._run_keepalive()
                finally:
                    with self._condition:
                        self._owner_work_claimed = False
                        self._condition.notify_all()
                continue
            if not item.future.set_running_or_notify_cancel():
                with self._condition:
                    self._cancelled_before_start += 1
                    self._owner_work_claimed = False
                    self._condition.notify_all()
                # Same lifetime contract as the completed path below: the
                # loop is about to park in _take_next, so the canceled
                # item must not survive in this frame.
                del item
                continue
            now = time.monotonic()
            queue_wait_s = max(0.0, now - item.queued_at_s)
            with self._condition:
                self._active_kind = item.kind
                self._active_sequence = item.sequence
                self._active_batch_key = item.batch_key
                self._active_started_at_s = now
                self._active_queue_wait_s = queue_wait_s
                self._active_self_reported = False
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
                    if item.kind == "foreground" and not self._active_self_reported:
                        # One foreground item that never reported microbatch
                        # sizes is a single-request unit of owner work. Pumps
                        # report their true per-step sizes via
                        # record_batch_step, and idle/persistence bookkeeping
                        # is not a batch at all — stamping [1] for those
                        # polluted the histogram with phantom size-1 batches.
                        self._batch_histogram[1] += 1
                    self._run_duration_samples_s.append(run_duration_s)
                    # Any owner work is GPU activity: the keepalive clock
                    # restarts from here. Only a FOREGROUND completion
                    # extends the attentive window — a background warm
                    # rung or a postcommit snapshot is not a user talking
                    # to the model.
                    self._last_owner_activity_s = time.monotonic()
                    if item.kind == "foreground":
                        self._keepalive_attentive_anchor_s = self._last_owner_activity_s
                    if item.kind != "idle_persistence":
                        # Foreground AND postcommit completions re-arm the
                        # persistence quiet grace: cold work may only start
                        # after a full quiet window with no latency-critical
                        # completion — closing the race where a
                        # submission-time grace expires during a long
                        # generation and releases cold in the few-ms gap
                        # before the tail postcommit arrives.
                        self._last_quiet_anchor_s = time.monotonic()
                    self._active_kind = None
                    self._owner_work_claimed = False
                    self._active_sequence = None
                    self._active_batch_key = None
                    self._active_started_at_s = None
                    self._active_queue_wait_s = None
                    self._condition.notify_all()
                # Release the finished item before looping: _take_next can
                # park this frame indefinitely, and a bound local would pin
                # the completed closure (persistence items can reference
                # GB-scale snapshot views) across the entire idle period.
                del item

    def _take_next(self) -> _WorkItem | _KeepaliveTurn | None:
        with self._condition:
            while True:
                if (
                    self._shutdown
                    and not self._foreground
                    and not self._idle
                    and not self._persistence
                ):
                    return None
                if self._foreground:
                    self._owner_work_claimed = True
                    return self._foreground.popleft()
                now = time.monotonic()
                wait_until: float | None = None
                if self._idle:
                    if self._idle[0].earliest_start_s - now <= 0:
                        self._owner_work_claimed = True
                        return self._idle.popleft()
                    wait_until = self._idle[0].earliest_start_s
                if self._persistence and (
                    not self._idle or self._persistence_pump_budget > 0
                ):
                    # Persistence normally runs only with NO queued
                    # postcommit, after a quiet grace anchored to the last
                    # foreground/postcommit COMPLETION. No age-based bypass:
                    # an item queued during a foreground longer than any
                    # valve would already be "overdue" at completion and
                    # would dequeue in the few-ms gap before the tail
                    # postcommit arrives. The ONE exception is an armed
                    # idle pump (issue #290): the server observed seconds
                    # of request-level idleness — no tail gap can exist —
                    # so the head may bridge the not-yet-ready gaps of a
                    # self-chaining idle occupant (the 2.8.2 warm-ladder
                    # starvation). A READY idle item was already taken
                    # above; foreground submission zeroes the budget.
                    # Peek as a subexpression: binding the head to a local
                    # would keep a superseded item (and its snapshot
                    # closure) pinned by this parked frame until the next
                    # wake, making newest-wins release timing-dependent.
                    ready_at = max(
                        self._persistence[0].earliest_start_s,
                        self._last_quiet_anchor_s + self.persistence_quiet_grace_s,
                    )
                    if now >= ready_at:
                        if self._idle:
                            # Pump-bridged pop: consume one armed slot.
                            self._persistence_pump_budget -= 1
                            self._persistence_pumped += 1
                        self._owner_work_claimed = True
                        return self._persistence.popleft()
                    if wait_until is None:
                        wait_until = ready_at
                    else:
                        wait_until = min(wait_until, ready_at)
                # Idle keepalive: with no runnable work, the owner thread
                # beats once per interval while the engine is attentive.
                keepalive_at = self._keepalive_due_locked(now)
                if keepalive_at is not None:
                    if now >= keepalive_at:
                        self._owner_work_claimed = True
                        return _KEEPALIVE
                    wait_until = (
                        keepalive_at if wait_until is None else min(wait_until, keepalive_at)
                    )
                if wait_until is not None:
                    self._condition.wait(timeout=max(0.0, wait_until - now))
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
