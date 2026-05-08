"""Single-owner model work scheduler for MTPLX serving.

The scheduler deliberately runs model work on one thread because MLX stream
state and live cache references are thread-affine on Apple Silicon. It still
keeps request admission explicit: foreground generation has priority over idle
maintenance work such as SessionBank postcommit snapshots.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Thread, get_ident
import time
from typing import Any, Callable


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

    def stats(self) -> dict[str, Any]:
        with self._condition:
            return {
                "foreground_pending": len(self._foreground),
                "idle_pending": len(self._idle),
                "active_kind": self._active_kind,
                "started": self._started,
                "completed": self._completed,
                "owner_thread_id": self._owner_thread_id,
                "shutdown": self._shutdown,
            }

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
        while True:
            item = self._take_next()
            if item is None:
                return
            if not item.future.set_running_or_notify_cancel():
                continue
            with self._condition:
                self._active_kind = (
                    "foreground" if item.kind == "foreground" else "idle_postcommit"
                )
                self._started += 1
            try:
                item.future.set_result(item.fn(*item.args, **item.kwargs))
            except BaseException as exc:
                item.future.set_exception(exc)
            finally:
                with self._condition:
                    self._completed += 1
                    self._active_kind = None
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
