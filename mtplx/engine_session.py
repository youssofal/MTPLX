"""Serving-session scaffolding for long-context MTPLX chat.

This module keeps HTTP/OpenAI behavior out of the prefix cache. It is small on
purpose: the first production step is to make lifecycle, metrics, and admin
state explicit before the generation loop accepts warm prompt state directly.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any, Iterator, Mapping

from .session_bank import (
    CacheMissReason,
    DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS,
    DEFAULT_IDLE_TTL_S,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_MAX_BYTES,
    DEFAULT_PER_SESSION_MAX_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
    SessionBank,
    block_aligned_prefix_len,
)


logger = logging.getLogger(__name__)

_HIGH_MEMORY_SESSION_BANK_THRESHOLD_BYTES = 96 * 1024**3
_HIGH_MEMORY_PER_SESSION_MAX_BYTES = 24 * 1024**3
_HIGH_MEMORY_MAX_ENTRIES = 16


def _bank_bytes_from_env(name: str, default: int) -> int:
    """Read a SessionBank byte-cap override from the environment.

    Supports plain integers (interpreted as bytes) and the suffixes K, M, G,
    T (powers of 1024). Returns the default if unset, unparseable, or
    nonpositive.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    s = raw.strip().upper()
    if not s:
        return default
    try:
        suffixes = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
        if s and s[-1] in suffixes:
            value = int(float(s[:-1]) * suffixes[s[-1]])
        else:
            value = int(s)
    except (OverflowError, ValueError, IndexError):
        return default
    if value < 1:
        return default
    return value


def _bank_entries_from_env(name: str, default: int) -> int:
    """Read a SessionBank entry-count override from the environment.

    Parses a plain integer (no K/M/G/T suffixes - entry count is a count, not
    a byte size). Validates that the value is >= 1. On parse error or invalid
    value, logs a warning and returns the default.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning(
            "Invalid %s=%r (expected integer >= 1); falling back to default %d",
            name,
            raw,
            default,
        )
        return default
    if value < 1:
        logger.warning(
            "Invalid %s=%r (must be >= 1); falling back to default %d",
            name,
            raw,
            default,
        )
        return default
    return value


def _detect_total_ram_bytes_for_session_bank() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        total = int(str(output).strip())
    except Exception:
        return None
    return total if total > 0 else None


def _default_max_entries() -> int:
    total_ram = _detect_total_ram_bytes_for_session_bank()
    if (
        total_ram is not None
        and total_ram >= _HIGH_MEMORY_SESSION_BANK_THRESHOLD_BYTES
    ):
        return _HIGH_MEMORY_MAX_ENTRIES
    return DEFAULT_MAX_ENTRIES


def _session_bank_max_entries() -> int:
    raw = os.environ.get("MTPLX_SESSION_BANK_MAX_ENTRIES")
    default = _default_max_entries()
    if raw is None:
        return default
    return _bank_entries_from_env("MTPLX_SESSION_BANK_MAX_ENTRIES", default)


def _default_per_session_max_bytes() -> int:
    total_ram = _detect_total_ram_bytes_for_session_bank()
    if (
        total_ram is not None
        and total_ram >= _HIGH_MEMORY_SESSION_BANK_THRESHOLD_BYTES
    ):
        return _HIGH_MEMORY_PER_SESSION_MAX_BYTES
    return DEFAULT_PER_SESSION_MAX_BYTES


def _session_bank_per_session_max_bytes() -> int:
    raw = os.environ.get("MTPLX_SESSION_BANK_PER_SESSION_BYTES")
    default = _default_per_session_max_bytes()
    if raw is None:
        return default
    return _bank_bytes_from_env("MTPLX_SESSION_BANK_PER_SESSION_BYTES", default)


@dataclass
class BoundarySnapshot:
    kind: str
    token_len: int
    token_hash: str
    created_at_s: float = field(default_factory=time.time)
    bank_token_hash: str | None = None
    nbytes: int = 0
    snapshot_epoch: int = 0


@dataclass
class EngineSessionCommit:
    committed: bool
    reason: str
    prefix_len: int


class EngineSessionBusy(RuntimeError):
    """Raised when a foreground request tries to mutate an in-flight session."""


def token_hash_short(token_ids: list[int] | tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for token in token_ids:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()[:16]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


IMPLICIT_SESSION_SOURCES = frozenset({"longest_prefix", "pending_postcommit_near_prefix"})


def _new_anon_session_id() -> str:
    return f"anon-{secrets.token_hex(8)}"


def common_prefix_len(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def _message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def _message_content(message: Any) -> str:
    if isinstance(message, Mapping):
        value = message.get("content", "")
    else:
        value = getattr(message, "content", "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def system_prompt_hash(messages: list[Any]) -> str | None:
    for message in messages:
        if _message_role(message) in {"system", "developer"}:
            return hash_text(_message_content(message))
    return None


def is_no_history_shape(messages: list[Any]) -> bool:
    roles = [_message_role(message) for message in messages if _message_role(message)]
    return roles in (["system", "user"], ["developer", "user"])


def is_background_request(
    *,
    messages: list[Any],
    max_tokens: int | None,
    headers: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    main_system_hash: str | None = None,
) -> bool:
    if max_tokens is None or int(max_tokens) > 48:
        return False
    headers = headers or {}
    metadata = metadata or {}
    header_task = ""
    for key, value in headers.items():
        if key.lower() == "x-openwebui-task":
            header_task = str(value)
            break
    metadata_task = str(metadata.get("task") or metadata.get("openwebui_task") or "")
    current_system_hash = system_prompt_hash(messages)
    system_mismatch = (
        main_system_hash is not None
        and current_system_hash is not None
        and current_system_hash != main_system_hash
    )
    return bool(header_task or metadata_task or system_mismatch)


_DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S = 8.0
_DEFAULT_NEAR_PREFIX_MAX_TOKEN_GAP = 8
_DEFAULT_NEAR_PREFIX_MIN_MATCH_TOKENS = 64
_DEFAULT_PREFIX_BLOCK_SIZE = DEFAULT_PREFIX_BLOCK_SIZE
_DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS = DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS


def _postcommit_wait_timeout_s() -> float:
    """Read MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S from the environment.

    Defaults to 8s. Values <= 0 disable the wait (returns 0.0). Bad values
    fall back to the default so a typo does not leave the server hanging.
    """
    raw = os.environ.get("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S")
    if raw is None:
        return _DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S
    if value < 0:
        return 0.0
    return value


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(int(minimum), value)


def _near_prefix_max_token_gap() -> int:
    return _env_int(
        "MTPLX_SESSION_NEAR_PREFIX_MAX_TOKEN_GAP",
        _DEFAULT_NEAR_PREFIX_MAX_TOKEN_GAP,
        minimum=0,
    )


def _near_prefix_min_match_tokens() -> int:
    return _env_int(
        "MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS",
        _DEFAULT_NEAR_PREFIX_MIN_MATCH_TOKENS,
        minimum=1,
    )


def _block_prefix_restore_enabled() -> bool:
    raw = os.environ.get("MTPLX_SESSION_BLOCK_PREFIX_RESTORE")
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _prefix_block_size() -> int:
    return _env_int(
        "MTPLX_SESSION_PREFIX_BLOCK_SIZE",
        _DEFAULT_PREFIX_BLOCK_SIZE,
        minimum=1,
    )


def _block_prefix_min_match_tokens() -> int:
    return _env_int(
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS",
        _DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS,
        minimum=1,
    )


@dataclass
class PendingPostcommit:
    """Best-effort SessionBank maintenance currently tied to a session.

    The future remains the compatibility surface; this record adds the
    information needed to make idle maintenance cancellable and observable.
    """

    future: Any
    abort_event: Event = field(default_factory=Event)
    reason: str = "postcommit"
    token_count: int = 0
    created_at_s: float = field(default_factory=time.time)
    started_at_s: float | None = None
    finished_at_s: float | None = None
    last_outcome: dict[str, Any] | None = None
    last_abort_reason: str | None = None

    def mark_started(self) -> None:
        if self.started_at_s is None:
            self.started_at_s = time.time()

    def mark_finished(self, outcome: dict[str, Any] | None = None) -> None:
        self.finished_at_s = time.time()
        if outcome is not None:
            self.last_outcome = outcome

    def abort(self, reason: str) -> bool:
        self.last_abort_reason = str(reason)
        self.abort_event.set()
        cancel = getattr(self.future, "cancel", None)
        if callable(cancel):
            try:
                return bool(cancel())
            except BaseException:
                return False
        return False

    def update_token_count(self, token_count: int) -> None:
        try:
            self.token_count = max(0, int(token_count))
        except (TypeError, ValueError):
            self.token_count = 0

    def to_admin_dict(self) -> dict[str, Any]:
        now = time.time()
        future = self.future
        return {
            "active": bool(future is not None and not getattr(future, "done", lambda: False)()),
            "reason": self.reason,
            "token_count": int(self.token_count),
            "age_s": max(0.0, now - float(self.created_at_s)),
            "created_at_s": self.created_at_s,
            "started_at_s": self.started_at_s,
            "finished_at_s": self.finished_at_s,
            "abort_requested": self.abort_event.is_set(),
            "last_abort_reason": self.last_abort_reason,
            "last_outcome": self.last_outcome,
        }


class EngineSession:
    def __init__(self, session_id: str, *, idle_ttl_s: float = DEFAULT_IDLE_TTL_S) -> None:
        self.session_id = str(session_id)
        self.idle_ttl_s = float(idle_ttl_s)
        self.created_at_s = time.time()
        self.last_access_s = self.created_at_s
        self.committed_token_ids: tuple[int, ...] = ()
        self.boundaries: list[BoundarySnapshot] = []
        self.in_flight = False
        self.in_flight_started_s: float | None = None
        self.last_commit_s: float | None = None
        self.last_finish_reason: str | None = None
        self.last_cache_miss_reason: str | None = CacheMissReason.NEW_SESSION.value
        self.last_restore_mode: str = "cold"
        self.bytes_estimate = 0
        self.revision = 0
        self._lock = Lock()
        # Reference to the most recent postcommit work scheduled for this
        # session. The next request in this session waits briefly on this
        # before acquiring the session lock so the SessionBank entry is
        # available when its prefix lookup runs - avoiding the cold-prefill
        # cascade documented in PR #34. Always written/read while NOT holding
        # the session lock to preserve the no-deadlock ordering. Type kept as
        # Any to avoid pulling concurrent.futures into hot import paths.
        self._pending_postcommit: PendingPostcommit | None = None
        # Per-session lock guarding `pending_postcommit` reads/writes. This is
        # SEPARATE from `_lock` (which guards `in_flight_generation`) so that
        # `wait_for_pending_postcommit` can serialize access to the future
        # field without ever contending with the foreground/in-flight lock,
        # preserving the no-deadlock ordering callers rely on. The lock is
        # only held while reading or mutating `pending_postcommit` itself,
        # never while awaiting on the future.
        self._postcommit_lock = Lock()
        # Last wait outcome, exposed via to_admin_dict for the metrics endpoint.
        self.last_postcommit_wait: dict[str, Any] | None = None
        self.last_postcommit_outcome: dict[str, Any] | None = None

    @property
    def pending_postcommit(self) -> Any:
        record = self._pending_postcommit
        return None if record is None else record.future

    @pending_postcommit.setter
    def pending_postcommit(self, future: Any) -> None:
        if future is None:
            self._pending_postcommit = None
        elif isinstance(future, PendingPostcommit):
            self._pending_postcommit = future
        else:
            self._pending_postcommit = PendingPostcommit(future=future)

    @property
    def prefix_len(self) -> int:
        return len(self.committed_token_ids)

    def touch(self) -> None:
        self.last_access_s = time.time()

    def is_stale(self, *, now_s: float | None = None) -> bool:
        now = time.time() if now_s is None else float(now_s)
        return now - self.last_access_s > self.idle_ttl_s

    def set_pending_postcommit(
        self,
        future: Any,
        *,
        abort_event: Event | None = None,
        reason: str = "postcommit",
        token_count: int = 0,
    ) -> PendingPostcommit:
        """Record a reference to in-flight postcommit work for this session.

        The next request in this session calls wait_for_pending_postcommit()
        before acquiring the session lock so the prior turn's SessionBank
        entry is visible at lookup time. Older references are dropped on each
        new commit; only the most recent matters for the next turn's lookup.

        The write is guarded by `_postcommit_lock` so it cannot race with a
        concurrent `wait_for_pending_postcommit` reading the field.
        """
        record = (
            future
            if isinstance(future, PendingPostcommit)
            else PendingPostcommit(
                future=future,
                abort_event=abort_event or Event(),
                reason=str(reason or "postcommit"),
                token_count=max(0, int(token_count or 0)),
            )
        )
        with self._postcommit_lock:
            self._pending_postcommit = record
        return record

    def has_pending_postcommit(self) -> bool:
        with self._postcommit_lock:
            return self._pending_postcommit is not None

    def pending_postcommit_admin(self) -> dict[str, Any] | None:
        with self._postcommit_lock:
            record = self._pending_postcommit
            if record is None:
                return None
            return record.to_admin_dict()

    def abort_pending_postcommit(self, reason: str) -> dict[str, Any]:
        with self._postcommit_lock:
            record = self._pending_postcommit
        if record is None:
            return {"aborted": False, "reason": "no_pending"}
        cancelled = record.abort(reason)
        return {
            "aborted": True,
            "reason": str(reason),
            "future_cancelled": bool(cancelled),
        }

    def finish_pending_postcommit(
        self,
        record: PendingPostcommit,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        if record is None:
            return
        record.mark_finished(outcome)
        if outcome is not None:
            self.last_postcommit_outcome = outcome
        with self._postcommit_lock:
            if self._pending_postcommit is record:
                self._pending_postcommit = None

    def wait_for_pending_postcommit(
        self,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Bounded wait for the prior postcommit job to land.

        Returns a small telemetry dict the caller can attach to request stats:
            {"waited": bool, "elapsed_s": float, "outcome": str,
             "timeout_s": float}

        `outcome` is one of:
            - "no_pending"     : nothing was scheduled, no wait performed
            - "completed"      : the future resolved within the timeout
            - "timeout"        : the future did not resolve in time; the
                                 request should fall through to a cold
                                 prefill rather than hang
            - "error:<Type>"   : the future raised; we swallow it because the
                                 postcommit's job is best-effort caching, not
                                 correctness
            - "disabled"       : timeout_s <= 0; wait short-circuits

        CRITICAL: This must be called WITHOUT the session lock (`_lock`)
        held. The postcommit work runs on the model scheduler's owner thread;
        a foreground request that holds the session lock and then waits on
        scheduler-bound work risks priority inversion against other
        same-session commits queued behind it. Callers should invoke this
        before entering `in_flight_generation()`.

        Concurrency contract (the bug fix from PR #37 review):

        Two concurrent same-session waiters MUST observe the SAME active
        future and either both report "completed" (on resolve) or both
        report "timeout" (on timeout). Neither must ever observe
        "no_pending" while a real postcommit is still in flight.

        We achieve that by capturing the current future under
        `_postcommit_lock`, releasing the lock BEFORE awaiting on the
        future (so other same-session waiters can observe the same future),
        then re-acquiring the lock after the wait and clearing
        `pending_postcommit` ONLY if it is still the same future identity
        (`is` comparison). If a newer commit superseded the future while we
        were waiting, the newer reference belongs to the next caller and we
        leave it alone.
        """
        if timeout_s is None:
            timeout_s = _postcommit_wait_timeout_s()
        timeout_s = float(timeout_s)
        # Capture the current future under the lock so a concurrent
        # `set_pending_postcommit` cannot tear our read. The lock is released
        # immediately - we MUST NOT hold it while awaiting on the future, and
        # leaving it visible on the session is the whole point: a second
        # same-session waiter must observe the same future, not "no_pending".
        with self._postcommit_lock:
            record = self._pending_postcommit
        if record is None:
            outcome = {
                "waited": False,
                "elapsed_s": 0.0,
                "outcome": "no_pending",
                "timeout_s": timeout_s,
            }
            self.last_postcommit_wait = outcome
            return outcome
        future = record.future
        if not hasattr(future, "result"):
            # Defensive: anything stashed on the session that does not look
            # like a future is treated as "nothing to wait on". We do NOT
            # clear it - the field will be overwritten on the next legitimate
            # set_pending_postcommit call.
            outcome = {
                "waited": False,
                "elapsed_s": 0.0,
                "outcome": "no_pending",
                "timeout_s": timeout_s,
            }
            self.last_postcommit_wait = outcome
            return outcome
        if timeout_s <= 0.0:
            # Disabled mode: do NOT touch `pending_postcommit`. Operators
            # toggle this dynamically via MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S; if
            # they re-enable later we want the existing future still visible
            # so it is not silently dropped.
            outcome = {
                "waited": False,
                "elapsed_s": 0.0,
                "outcome": "disabled",
                "timeout_s": timeout_s,
            }
            self.last_postcommit_wait = outcome
            record.last_outcome = outcome
            return outcome
        t0 = time.monotonic()
        # We catch BaseException because any failure in the postcommit must
        # not propagate into the foreground request: the wait is a best-effort
        # cache warmup, not a correctness dependency. Timeout is the most
        # common non-success outcome and is reported distinctly so operators
        # can spot a stuck postcommit lane.
        try:
            future.result(timeout=timeout_s)
            outcome = {
                "waited": True,
                "elapsed_s": time.monotonic() - t0,
                "outcome": "completed",
                "timeout_s": timeout_s,
            }
        except BaseException as exc:
            exc_name = type(exc).__name__
            preempted_cancel = (
                exc_name == "CancelledError"
                and record.abort_event.is_set()
                and record.last_abort_reason == "foreground_preempted_postcommit"
            )
            label = (
                "timeout"
                if exc_name == "TimeoutError" or preempted_cancel
                else f"error:{exc_name}"
            )
            future_cancelled = False
            if label == "timeout":
                future_cancelled = record.abort("foreground_preempted_postcommit")
            outcome = {
                "waited": True,
                "elapsed_s": time.monotonic() - t0,
                "outcome": label,
                "timeout_s": timeout_s,
                "abort_requested": label == "timeout",
                "future_cancelled": bool(future_cancelled),
                "abort_reason": (
                    "foreground_preempted_postcommit" if label == "timeout" else None
                ),
            }
        # Clear the reference ONLY if it is still the same future we
        # observed. A concurrent same-session commit may have superseded it
        # while we were waiting; in that case the newer future belongs to
        # the next caller and we must not stomp it. Identity check (`is`)
        # is required: equality could collapse two distinct futures with
        # the same result.
        with self._postcommit_lock:
            if self._pending_postcommit is record:
                self._pending_postcommit = None
        self.last_postcommit_wait = outcome
        self.last_postcommit_outcome = outcome
        record.mark_finished(outcome)
        return outcome

    def try_begin_generation(self) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        self.in_flight = True
        self.in_flight_started_s = time.time()
        self.touch()
        return True

    def end_generation(self) -> None:
        self.in_flight = False
        self.in_flight_started_s = None
        self.touch()
        self._lock.release()

    @contextmanager
    def in_flight_generation(self) -> Iterator["EngineSession"]:
        if not self.try_begin_generation():
            raise EngineSessionBusy(f"session {self.session_id} is already in flight")
        try:
            yield self
        finally:
            self.end_generation()

    def commit(
        self,
        *,
        prompt_ids: list[int] | tuple[int, ...],
        generated_ids: list[int] | tuple[int, ...],
        finish_reason: str,
        boundary_kind: str = "assistant_end",
        nbytes: int = 0,
    ) -> EngineSessionCommit:
        if finish_reason not in {"stop", "length"}:
            return EngineSessionCommit(False, f"unsafe_finish:{finish_reason}", self.prefix_len)
        tokens = tuple(int(token) for token in prompt_ids) + tuple(int(token) for token in generated_ids)
        self.committed_token_ids = tokens
        self.last_commit_s = time.time()
        self.last_finish_reason = finish_reason
        self.bytes_estimate = int(nbytes)
        self.revision += 1
        self._record_interval_boundaries(tokens)
        self.add_boundary(boundary_kind, tokens, nbytes=nbytes)
        return EngineSessionCommit(True, "committed", self.prefix_len)

    def commit_prompt_prefix(
        self,
        *,
        prompt_ids: list[int] | tuple[int, ...],
        finish_reason: str,
        boundary_kind: str = "prompt_prefix",
    ) -> EngineSessionCommit:
        """Publish a safe foreground prompt prefix for async postcommit waiters.

        Streaming tool-call responses cannot commit the structured assistant
        history directly; that history is canonicalized by a low-priority
        retokenized postcommit. The next tool-result request still needs to
        resolve to this EngineSession so it can wait for that postcommit
        instead of cold-prefilling. Publishing the already-prefilled prompt
        prefix gives the resolver a stable session anchor without claiming the
        assistant/tool history has landed yet.
        """
        tokens = tuple(int(token) for token in prompt_ids)
        if not tokens:
            return EngineSessionCommit(False, "empty_prompt_prefix", self.prefix_len)
        current = self.committed_token_ids
        if current:
            if len(tokens) < len(current):
                return EngineSessionCommit(
                    False,
                    "prompt_prefix_older_than_session",
                    self.prefix_len,
                )
            if tokens[: len(current)] != current:
                return EngineSessionCommit(
                    False,
                    "prompt_prefix_not_extending_session",
                    self.prefix_len,
                )
            if len(tokens) == len(current):
                return EngineSessionCommit(False, "prompt_prefix_unchanged", self.prefix_len)
        self.committed_token_ids = tokens
        self.last_commit_s = time.time()
        self.last_finish_reason = str(finish_reason)
        self.revision += 1
        self._record_interval_boundaries(tokens)
        self.add_boundary(boundary_kind, tokens)
        return EngineSessionCommit(True, "committed_prompt_prefix", self.prefix_len)

    def commit_retokenized_prefix(
        self,
        *,
        token_ids: list[int] | tuple[int, ...],
        expected_revision: int | None = None,
        boundary_kind: str = "retokenized_history",
        nbytes: int = 0,
    ) -> EngineSessionCommit:
        """Publish canonical retokenized history after async postcommit.

        Streaming tool-call turns first publish the foreground prompt as a
        temporary same-session anchor. The idle postcommit later renders the
        canonical assistant/tool history. Publishing that canonical prefix
        prevents the next OpenCode turn from resolving by a stale prompt
        boundary that can differ by one chat-template token.
        """
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            return EngineSessionCommit(False, "empty_retokenized_prefix", self.prefix_len)
        if expected_revision is not None and self.revision != int(expected_revision):
            return EngineSessionCommit(False, "stale_session_revision", self.prefix_len)
        current = self.committed_token_ids
        commit_reason = "committed_retokenized_prefix"
        if current:
            if len(tokens) < len(current):
                return EngineSessionCommit(
                    False,
                    "retokenized_prefix_older_than_session",
                    self.prefix_len,
                )
            matched = common_prefix_len(current, tokens)
            block_boundary = block_aligned_prefix_len(
                matched,
                block_size=_prefix_block_size(),
            )
            block_min_match = max(_prefix_block_size(), _block_prefix_min_match_tokens())
            if (
                matched < len(current)
                and (len(current) - matched) > _near_prefix_max_token_gap()
                and (
                    not _block_prefix_restore_enabled()
                    or block_boundary < block_min_match
                )
            ):
                return EngineSessionCommit(
                    False,
                    "retokenized_prefix_not_extending_session",
                    self.prefix_len,
                )
            if (
                matched < len(current)
                and (len(current) - matched) > _near_prefix_max_token_gap()
            ):
                commit_reason = "committed_retokenized_prefix_after_block_overlap"
            if len(tokens) == len(current) and matched == len(current):
                return EngineSessionCommit(
                    False,
                    "retokenized_prefix_unchanged",
                    self.prefix_len,
                )
        self.committed_token_ids = tokens
        self.last_commit_s = time.time()
        self.last_finish_reason = "postcommit"
        self.bytes_estimate = int(nbytes)
        self.revision += 1
        self._record_interval_boundaries(tokens)
        self.add_boundary(boundary_kind, tokens, nbytes=nbytes)
        return EngineSessionCommit(True, commit_reason, self.prefix_len)

    def add_boundary(
        self,
        kind: str,
        token_ids: list[int] | tuple[int, ...],
        *,
        bank_token_hash: str | None = None,
        nbytes: int = 0,
        snapshot_epoch: int | None = None,
    ) -> BoundarySnapshot:
        epoch = len(self.boundaries) if snapshot_epoch is None else int(snapshot_epoch)
        boundary = BoundarySnapshot(
            kind=str(kind),
            token_len=len(token_ids),
            token_hash=token_hash_short(token_ids),
            bank_token_hash=bank_token_hash,
            nbytes=int(nbytes),
            snapshot_epoch=epoch,
        )
        self.boundaries.append(boundary)
        self.touch()
        return boundary

    def nearest_boundary_at_or_before(self, token_len: int) -> BoundarySnapshot | None:
        candidates = [boundary for boundary in self.boundaries if boundary.token_len <= token_len]
        if not candidates:
            self.last_cache_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
            return None
        return max(candidates, key=lambda boundary: boundary.token_len)

    def _record_interval_boundaries(self, token_ids: tuple[int, ...], *, every: int = 512) -> None:
        existing = {boundary.token_len for boundary in self.boundaries}
        for token_len in range(every, len(token_ids), every):
            if token_len not in existing:
                self.add_boundary("interval_512", token_ids[:token_len])

    def to_admin_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prefix_len": self.prefix_len,
            "bytes": self.bytes_estimate,
            "created_at_s": self.created_at_s,
            "last_access_s": self.last_access_s,
            "last_commit_s": self.last_commit_s,
            "last_finish_reason": self.last_finish_reason,
            "revision": self.revision,
            "in_flight": self.in_flight,
            "in_flight_started_s": self.in_flight_started_s,
            "last_cache_miss_reason": self.last_cache_miss_reason,
            "last_restore_mode": self.last_restore_mode,
            "last_postcommit_wait": self.last_postcommit_wait,
            "last_postcommit_outcome": self.last_postcommit_outcome,
            "pending_postcommit": bool(self.pending_postcommit is not None),
            "pending_postcommit_detail": self.pending_postcommit_admin(),
            "boundaries": [
                {
                    "kind": boundary.kind,
                    "token_len": boundary.token_len,
                    "token_hash": boundary.token_hash,
                    "bank_token_hash": boundary.bank_token_hash,
                    "nbytes": boundary.nbytes,
                    "snapshot_epoch": boundary.snapshot_epoch,
                    "created_at_s": boundary.created_at_s,
                }
                for boundary in self.boundaries[-32:]
            ],
        }


class EngineSessionManager:
    def __init__(
        self,
        *,
        bank: SessionBank | None = None,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
        cold_tier: Any | None = None,
    ) -> None:
        # Bank caps mostly default to the constants in session_bank.py. On
        # 96GB+ Apple Silicon machines, the per-session default rises to the
        # global 24GiB bank cap so exact 100k-token prefixes and retokenized
        # follow-up histories can stay hot in RAM together without increasing
        # the total RAM-cache budget. Operators can still override byte caps via
        # MTPLX_SESSION_BANK_PER_SESSION_BYTES (e.g. "16G") or
        # MTPLX_SESSION_BANK_MAX_BYTES. The entry-count cap is also overridable
        # via MTPLX_SESSION_BANK_MAX_ENTRIES (plain integer) for workloads
        # where ~2 GB-per-entry contexts make the default of 8 the binding
        # constraint well before the byte caps.
        self.bank = bank or SessionBank(
            max_entries=_session_bank_max_entries(),
            max_bytes=_bank_bytes_from_env(
                "MTPLX_SESSION_BANK_MAX_BYTES", DEFAULT_MAX_BYTES
            ),
            per_session_max_bytes=_session_bank_per_session_max_bytes(),
            idle_ttl_s=idle_ttl_s,
            cold_tier=cold_tier,
        )
        self.idle_ttl_s = float(idle_ttl_s)
        self._sessions: dict[str, EngineSession] = {}
        self._lock = Lock()
        self.last_prefix_diagnostic: dict[str, Any] | None = None

    def resolve_session_id(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        user: str | None = None,
        chat_id: str | None = None,
        conversation_id: str | None = None,
        prompt_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[str, str]:
        headers = headers or {}
        metadata = metadata or {}
        lowered_headers = {
            str(key).lower(): value
            for key, value in headers.items()
        }
        for key in (
            "x-mtplx-session-id",
            "x-openwebui-chat-id",
            "x-openwebui-user-id",
        ):
            value = lowered_headers.get(key)
            if str(value or "").strip():
                return str(value).strip(), f"header.{key}"
        for key in ("session_id", "mtplx_session_id", "chat_id", "conversation_id"):
            value = metadata.get(key)
            if value:
                return str(value), f"metadata.{key}"
        if user:
            return str(user), "user"
        if chat_id:
            return str(chat_id), "chat_id"
        if conversation_id:
            return str(conversation_id), "conversation_id"
        if prompt_ids:
            best = self.longest_prefix_session(prompt_ids)
            if best is not None:
                self.last_prefix_diagnostic = self._prefix_diagnostic(
                    prompt_ids,
                    selected=best,
                    exact=True,
                )
                return best.session_id, "longest_prefix"
            pending, matched = self.pending_near_prefix_session(prompt_ids)
            if pending is not None:
                diagnostic = self._prefix_diagnostic(prompt_ids)
                diagnostic.update(
                    {
                        "best_session_id": pending.session_id,
                        "best_prefix_len": len(pending.committed_token_ids),
                        "matched_prefix_len": int(matched),
                        "divergence_at_token": int(matched),
                        "best_token_hash": token_hash_short(
                            pending.committed_token_ids
                        ),
                        "reason": "pending_postcommit_near_prefix_match",
                        "near_prefix_gap": len(pending.committed_token_ids)
                        - int(matched),
                    }
                )
                self.last_prefix_diagnostic = diagnostic
                return pending.session_id, "pending_postcommit_near_prefix"
            self.last_prefix_diagnostic = self._prefix_diagnostic(prompt_ids)
        else:
            self.last_prefix_diagnostic = None
        return _new_anon_session_id(), "new"

    def get_or_create(self, session_id: str) -> EngineSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = EngineSession(session_id, idle_ttl_s=self.idle_ttl_s)
                self._sessions[session_id] = session
            session.touch()
            return session

    def _sessions_snapshot(self) -> list[EngineSession]:
        with self._lock:
            return list(self._sessions.values())

    @contextmanager
    def generation_slot(
        self,
        session: EngineSession,
        *,
        source: str | None = None,
    ) -> Iterator[EngineSession]:
        acquired = session
        if not session.try_begin_generation():
            if str(source or "") not in IMPLICIT_SESSION_SOURCES:
                raise EngineSessionBusy(
                    f"session {session.session_id} is already in flight"
                )
            while True:
                acquired = self.get_or_create(_new_anon_session_id())
                if acquired.try_begin_generation():
                    break
        try:
            yield acquired
        finally:
            acquired.end_generation()

    def longest_prefix_session(self, token_ids: list[int] | tuple[int, ...]) -> EngineSession | None:
        tokens = tuple(int(token) for token in token_ids)
        best: EngineSession | None = None
        for session in self._sessions_snapshot():
            prefix = session.committed_token_ids
            if not prefix:
                continue
            if len(prefix) > len(tokens):
                continue
            if tokens[: len(prefix)] != prefix:
                continue
            if best is None or len(prefix) > len(best.committed_token_ids):
                best = session
        return best

    def pending_near_prefix_session(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        max_token_gap: int | None = None,
        min_matched_tokens: int | None = None,
    ) -> tuple[EngineSession | None, int]:
        """Return a pending-postcommit session with a near-exact prompt prefix."""
        tokens = tuple(int(token) for token in token_ids)
        gap_limit = (
            _near_prefix_max_token_gap()
            if max_token_gap is None
            else max(0, int(max_token_gap))
        )
        min_match = (
            _near_prefix_min_match_tokens()
            if min_matched_tokens is None
            else max(1, int(min_matched_tokens))
        )
        best: EngineSession | None = None
        best_matched = 0
        block_size = _prefix_block_size()
        block_min_match = max(block_size, _block_prefix_min_match_tokens())
        for session in self._sessions_snapshot():
            if not session.has_pending_postcommit():
                continue
            prefix = session.committed_token_ids
            if not prefix:
                continue
            matched = common_prefix_len(tokens, prefix)
            gap = len(prefix) - matched
            required_match = min(min_match, max(1, len(prefix) - gap_limit))
            safe_block = min(
                block_aligned_prefix_len(matched, block_size=block_size),
                len(prefix),
                len(tokens),
            )
            near_match = gap >= 0 and gap <= gap_limit and matched >= required_match
            block_match = (
                _block_prefix_restore_enabled()
                and safe_block >= block_min_match
                and safe_block >= 2
            )
            if not near_match and not block_match:
                continue
            candidate_matched = int(matched if near_match else safe_block)
            if best is None or candidate_matched > best_matched or (
                candidate_matched == best_matched and len(prefix) > len(best.committed_token_ids)
            ):
                best = session
                best_matched = candidate_matched
        return best, best_matched

    def nearest_prefix_session(
        self,
        token_ids: list[int] | tuple[int, ...],
    ) -> tuple[EngineSession | None, int]:
        tokens = tuple(int(token) for token in token_ids)
        best: EngineSession | None = None
        best_len = 0
        for session in self._sessions_snapshot():
            prefix = session.committed_token_ids
            if not prefix:
                continue
            matched = common_prefix_len(tokens, prefix)
            if matched > best_len:
                best = session
                best_len = matched
        return best, best_len

    def _prefix_diagnostic(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        selected: EngineSession | None = None,
        exact: bool = False,
    ) -> dict[str, Any]:
        tokens = tuple(int(token) for token in token_ids)
        session, matched = (selected, len(selected.committed_token_ids)) if selected is not None else self.nearest_prefix_session(tokens)
        if session is None:
            return {
                "prompt_len": len(tokens),
                "exact_prefix_match": False,
                "best_session_id": None,
                "best_prefix_len": 0,
                "matched_prefix_len": 0,
                "divergence_at_token": None,
                "reason": "no_existing_session_prefix",
            }
        divergence_at = None if exact else matched
        boundary = block_aligned_prefix_len(
            matched,
            block_size=_prefix_block_size(),
        )
        return {
            "prompt_len": len(tokens),
            "exact_prefix_match": bool(exact),
            "best_session_id": session.session_id,
            "best_prefix_len": len(session.committed_token_ids),
            "matched_prefix_len": int(matched),
            "nearest_boundary_tokens": int(boundary),
            "new_prefill_tokens": max(0, len(tokens) - int(boundary)),
            "divergence_at_token": divergence_at,
            "best_token_hash": token_hash_short(session.committed_token_ids),
            "prompt_token_hash": token_hash_short(tokens),
            "reason": "exact_prefix_match" if exact else "prefix_divergence_at_token",
        }

    def evict_stale(self) -> int:
        now = time.time()
        with self._lock:
            stale_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.is_stale(now_s=now) and not session.in_flight
            ]
            for session_id in stale_ids:
                self._sessions.pop(session_id, None)
                self.bank.clear(session_id=session_id)
        return len(stale_ids)

    def clear_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        bank_entries = self.bank.clear(session_id=session_id)
        return {"session_id": session_id, "existed": existed, "bank_entries_cleared": bank_entries}

    def clear_all(self) -> dict[str, Any]:
        with self._lock:
            sessions = len(self._sessions)
            self._sessions.clear()
        bank_entries = self.bank.clear()
        return {"sessions_cleared": sessions, "bank_entries_cleared": bank_entries}

    def archive_cold_tier(self) -> dict[str, Any]:
        archive = getattr(self.bank, "archive_cold_tier", None)
        if not callable(archive):
            return {"archived": False, "reason": "ssd_cache_archive_unavailable"}
        return archive()

    def flush_cold_tier(self, *, timeout_s: float = 30.0) -> bool:
        flush = getattr(self.bank, "flush_cold_tier", None)
        if not callable(flush):
            return True
        return bool(flush(timeout_s=timeout_s))

    def list_sessions(self) -> dict[str, Any]:
        self.evict_stale()
        sessions = sorted(
            (session.to_admin_dict() for session in self._sessions_snapshot()),
            key=lambda row: row["last_access_s"],
            reverse=True,
        )
        return {
            "sessions": sessions,
            "count": len(sessions),
            "session_bank": self.bank.to_dict(),
            "last_prefix_diagnostic": self.last_prefix_diagnostic,
        }
