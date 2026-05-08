"""Serving-session scaffolding for long-context MTPLX chat.

This module keeps HTTP/OpenAI behavior out of the prefix cache. It is small on
purpose: the first production step is to make lifecycle, metrics, and admin
state explicit before the generation loop accepts warm prompt state directly.
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator, Mapping

from .session_bank import (
    CacheMissReason,
    DEFAULT_IDLE_TTL_S,
    DEFAULT_MAX_BYTES,
    DEFAULT_PER_SESSION_MAX_BYTES,
    SessionBank,
)


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
    return bool(
        header_task
        or metadata_task
        or system_mismatch
        or is_no_history_shape(messages)
    )


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

    @property
    def prefix_len(self) -> int:
        return len(self.committed_token_ids)

    def touch(self) -> None:
        self.last_access_s = time.time()

    def is_stale(self, *, now_s: float | None = None) -> bool:
        now = time.time() if now_s is None else float(now_s)
        return now - self.last_access_s > self.idle_ttl_s

    @contextmanager
    def in_flight_generation(self) -> Iterator["EngineSession"]:
        if not self._lock.acquire(blocking=False):
            raise EngineSessionBusy(f"session {self.session_id} is already in flight")
        self.in_flight = True
        self.in_flight_started_s = time.time()
        self.touch()
        try:
            yield self
        finally:
            self.in_flight = False
            self.in_flight_started_s = None
            self.touch()
            self._lock.release()

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
    ) -> None:
        # Bank caps default to the constants in session_bank.py. Operators
        # running into per-session eviction at large contexts can override
        # via MTPLX_SESSION_BANK_PER_SESSION_BYTES (e.g. "16G" for 16 GiB)
        # or MTPLX_SESSION_BANK_MAX_BYTES.
        self.bank = bank or SessionBank(
            max_bytes=_bank_bytes_from_env(
                "MTPLX_SESSION_BANK_MAX_BYTES", DEFAULT_MAX_BYTES
            ),
            per_session_max_bytes=_bank_bytes_from_env(
                "MTPLX_SESSION_BANK_PER_SESSION_BYTES", DEFAULT_PER_SESSION_MAX_BYTES
            ),
            idle_ttl_s=idle_ttl_s,
        )
        self.idle_ttl_s = float(idle_ttl_s)
        self._sessions: dict[str, EngineSession] = {}
        self._lock = Lock()

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
                return best.session_id, "longest_prefix"
        return f"anon-{hash_text(str(time.time_ns()))}", "new"

    def get_or_create(self, session_id: str) -> EngineSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = EngineSession(session_id, idle_ttl_s=self.idle_ttl_s)
                self._sessions[session_id] = session
            session.touch()
            return session

    def longest_prefix_session(self, token_ids: list[int] | tuple[int, ...]) -> EngineSession | None:
        tokens = tuple(int(token) for token in token_ids)
        best: EngineSession | None = None
        for session in self._sessions.values():
            prefix = session.committed_token_ids
            if len(prefix) > len(tokens):
                continue
            if tokens[: len(prefix)] != prefix:
                continue
            if best is None or len(prefix) > len(best.committed_token_ids):
                best = session
        return best

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

    def list_sessions(self) -> dict[str, Any]:
        self.evict_stale()
        sessions = sorted(
            (session.to_admin_dict() for session in self._sessions.values()),
            key=lambda row: row["last_access_s"],
            reverse=True,
        )
        return {
            "sessions": sessions,
            "count": len(sessions),
            "session_bank": self.bank.to_dict(),
        }
