"""In-memory bookkeeping for supervised engine processes.

``EngineRegistry`` tracks one ``EngineRecord`` per model id: its lifecycle
state, port, pid, in-flight pin count and last-used time. It owns no
process and does no I/O; ``mtplx/supervisor/process.py`` spawns and probes
the actual child, and the proxy/service layers call back into this registry
to update state as things happen. Mirrors the LRU/idle/pin shape of
``RetrievalRegistry`` in ``mtplx/retrieval.py``, generalized from "resident
backends under a cap" to "engine processes under supervision".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("mtplx.supervisor")


class EngineState(str, Enum):
    """Lifecycle of one supervised engine, per DESIGN.md."""

    INSTALLED = "installed"
    LOADING = "loading"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class EngineSpec:
    """What to run: a model id, its pack path, and its memory estimate."""

    model_id: str
    path: Path
    resident_bytes: int


@dataclass
class EngineRecord:
    """Everything the supervisor tracks about one engine over its life."""

    spec: EngineSpec
    state: EngineState
    port: int | None = None
    pid: int | None = None
    pins: int = 0
    last_used: float = field(default_factory=time.time)
    failure_reason: str | None = None
    aliases: list[str] = field(default_factory=list)


class EngineRegistry:
    """Thread-safe table of engine records, keyed by model id.

    One lock guards every read and write. Callers that need to act on a
    consistent snapshot (eviction, admission math) should treat the lists
    returned by ``lru_unpinned``/``idle``/``records`` as a point-in-time
    copy: the registry can change again the instant the lock is released.
    """

    def __init__(self) -> None:
        self._records: dict[str, EngineRecord] = {}
        self._lock = threading.RLock()

    # -- registration -----------------------------------------------------

    def add(self, spec: EngineSpec) -> EngineRecord:
        """Register a pack as installed. Re-adding the same id replaces it."""
        with self._lock:
            record = EngineRecord(spec=spec, state=EngineState.INSTALLED)
            self._records[spec.model_id] = record
            return record

    def get(self, model_id: str) -> EngineRecord | None:
        with self._lock:
            return self._records.get(model_id)

    def records(self) -> list[EngineRecord]:
        """Every tracked record, in no particular order."""
        with self._lock:
            return list(self._records.values())

    # -- resolution ---------------------------------------------------------

    def resolve(
        self, requested: str | None, default_id: str, strict: bool
    ) -> tuple[EngineRecord | None, str]:
        """Resolve a requested model id (or alias) to a record, per
        DESIGN.md behavior 1.

        Returns ``(record, "loaded")`` when ``requested`` names a record
        (by model id or by a registered alias) that is READY or LOADING;
        ``(record, "draining")`` when it names a record currently DRAINING
        (a caller should treat this as unavailable, not loadable -- see
        C1); ``(record, "installed")`` when it names a record in any other
        known state (INSTALLED, STOPPED, FAILED); and, when ``requested``
        is ``None`` or names nothing registered, falls back to
        ``default_id`` as ``(record, "fallback")`` unless ``strict`` is
        set, in which case that case is ``(None, "unknown")``.
        """
        with self._lock:
            if requested:
                record = self._records.get(requested)
                if record is None:
                    for candidate in self._records.values():
                        if requested in candidate.aliases:
                            record = candidate
                            break
                if record is not None:
                    if record.state in (EngineState.READY, EngineState.LOADING):
                        return record, "loaded"
                    if record.state == EngineState.DRAINING:
                        return record, "draining"
                    return record, "installed"
            if strict:
                return None, "unknown"
            default = self._records.get(default_id)
            if default is not None:
                return default, "fallback"
            return None, "unknown"

    def add_alias(self, model_id: str, alias: str) -> None:
        """Register ``alias`` as another accepted name for ``model_id`` (R1:
        engines often answer with their own served model id, which differs
        from the pack directory name the supervisor routes by).

        Idempotent: re-adding the same alias to the same model is a no-op.
        An alias equal to an existing model id, or already registered as
        another record's alias, is ignored with a warning log rather than
        silently reassigning routing.
        """
        with self._lock:
            record = self._records.get(model_id)
            if record is None:
                raise KeyError(model_id)
            if alias == model_id or alias in record.aliases:
                return
            if alias in self._records:
                logger.warning(
                    "alias %r for model %r collides with an existing model id; ignored",
                    alias,
                    model_id,
                )
                return
            for other_id, other in self._records.items():
                if other_id != model_id and alias in other.aliases:
                    logger.warning(
                        "alias %r for model %r is already registered to %r; ignored",
                        alias,
                        model_id,
                        other_id,
                    )
                    return
            record.aliases.append(alias)

    # -- pins ---------------------------------------------------------------

    def pin(self, model_id: str) -> None:
        with self._lock:
            record = self._records.get(model_id)
            if record is None:
                raise KeyError(model_id)
            record.pins += 1

    def unpin(self, model_id: str) -> None:
        """Decrement a pin, floored at 0. Unknown ids are ignored."""
        with self._lock:
            record = self._records.get(model_id)
            if record is None:
                return
            record.pins = max(0, record.pins - 1)

    def touch(self, model_id: str) -> None:
        """Mark a record as used just now, for LRU and idle-TTL purposes."""
        with self._lock:
            record = self._records.get(model_id)
            if record is None:
                raise KeyError(model_id)
            record.last_used = time.time()

    # -- state ----------------------------------------------------------------

    def set_state(
        self,
        model_id: str,
        state: EngineState,
        *,
        port: int | None = None,
        pid: int | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Move a record to a new state, optionally updating port/pid.

        Entering READY always clears ``failure_reason`` (a healthy engine
        carries no stale failure). Pins are request-owned and are never
        touched here, including on a transition into READY.
        """
        with self._lock:
            record = self._records.get(model_id)
            if record is None:
                raise KeyError(model_id)
            record.state = state
            if port is not None:
                record.port = port
            if pid is not None:
                record.pid = pid
            if state == EngineState.READY:
                record.failure_reason = None
            elif failure_reason is not None:
                record.failure_reason = failure_reason

    # -- eviction candidates --------------------------------------------------

    def lru_unpinned(self, exclude: set[str]) -> list[EngineRecord]:
        """READY, unpinned records, oldest ``last_used`` first."""
        with self._lock:
            candidates = [
                record
                for model_id, record in self._records.items()
                if model_id not in exclude
                and record.state == EngineState.READY
                and record.pins == 0
            ]
            return sorted(candidates, key=lambda record: record.last_used)

    def idle(
        self, now: float, ttl_s: float, exclude: set[str]
    ) -> list[EngineRecord]:
        """READY, unpinned records idle longer than ``ttl_s``.

        ``ttl_s <= 0`` disables idle unload entirely and always returns [].
        """
        if ttl_s <= 0:
            return []
        with self._lock:
            return [
                record
                for model_id, record in self._records.items()
                if model_id not in exclude
                and record.state == EngineState.READY
                and record.pins == 0
                and (now - record.last_used) > ttl_s
            ]
