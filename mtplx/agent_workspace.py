"""Portable state for the MTPLX agent desktop experience.

The model daemon remains the source of inference truth. This module owns the
user-facing state around a run: local workspace metadata, durable run events,
and explicit approval decisions. It intentionally does not execute tools.
Tool adapters can use the event and approval records without coupling the
desktop client to a particular UI or subprocess implementation.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


WORKSPACE_RUN_STATUSES = (
    "queued",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
)
APPROVAL_STATUSES = ("pending", "approved", "denied", "expired", "consumed")
POLICY_MODES = ("allow", "ask", "deny")
DEFAULT_TOOL_POLICY = {
    "read": "allow",
    "search": "allow",
    "write": "ask",
    "terminal": "ask",
    "browser": "ask",
    "network": "ask",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_id(value: str, *, fallback: str = "item") -> str:
    raw = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-")
    return cleaned[:160] or fallback


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def approval_arguments_sha256(arguments: Mapping[str, Any] | None) -> str:
    canonical = json.dumps(
        dict(arguments or {}),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    root_path: str
    created_at: str
    updated_at: str
    agent_profile: str = "mtplx-agent"
    model: str | None = None
    instructions: str = ""
    tool_policy: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TOOL_POLICY)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "root_path": self.root_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "agent_profile": self.agent_profile,
            "model": self.model,
            "instructions": self.instructions,
            "tool_policy": dict(self.tool_policy),
        }


@dataclass(frozen=True)
class AgentRun:
    id: str
    workspace_id: str
    session_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    model: str | None = None
    event_count: int = 0
    last_event_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "event_count": self.event_count,
            "last_event_at": self.last_event_at,
            "error": self.error,
        }


@dataclass(frozen=True)
class RunEvent:
    id: str
    run_id: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "payload": dict(self.payload),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    workspace_id: str
    run_id: str | None
    tool: str
    action: str
    description: str
    target: str | None
    risk: str
    status: str
    created_at: str
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_sha256: str | None = None
    expires_at: str | None = None
    resolved_at: str | None = None
    resolved_by: str | None = None
    reason: str | None = None
    consumed_at: str | None = None
    consumed_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "run_id": self.run_id,
            "tool": self.tool,
            "action": self.action,
            "description": self.description,
            "target": self.target,
            "risk": self.risk,
            "status": self.status,
            "created_at": self.created_at,
            "arguments": dict(self.arguments),
            "arguments_sha256": self.arguments_sha256,
            "expires_at": self.expires_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "reason": self.reason,
            "consumed_at": self.consumed_at,
            "consumed_by": self.consumed_by,
        }


class WorkspaceStoreError(ValueError):
    """Invalid workspace state or request."""


class WorkspaceNotFoundError(WorkspaceStoreError):
    pass


class WorkspaceConflictError(WorkspaceStoreError):
    pass


class WorkspaceStore:
    """Small file-backed store shared by the native app and HTTP clients."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        configured = root or os.environ.get("MTPLX_WORKSPACE_DIR") or "~/.mtplx/workspaces"
        self.root = Path(configured).expanduser().resolve()
        self._lock = threading.RLock()
        self._lock_path = self.root / ".lock"

    @classmethod
    def from_env(cls) -> "WorkspaceStore":
        return cls(os.environ.get("MTPLX_WORKSPACE_DIR"))

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("workspaces", "runs", "events", "approvals"):
            (self.root / name).mkdir(exist_ok=True)

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._lock:
            self.ensure_layout()
            handle = self._lock_path.open("a+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

    def _workspace_path(self, workspace_id: str) -> Path:
        return self.root / "workspaces" / f"{safe_id(workspace_id)}.json"

    def _run_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"{safe_id(run_id)}.json"

    def _events_path(self, run_id: str) -> Path:
        return self.root / "events" / f"{safe_id(run_id)}.jsonl"

    def _approval_path(self, approval_id: str) -> Path:
        return self.root / "approvals" / f"{safe_id(approval_id)}.json"

    @staticmethod
    def _decode_workspace(value: Mapping[str, Any]) -> Workspace:
        policy = value.get("tool_policy")
        return Workspace(
            id=str(value["id"]),
            name=str(value.get("name") or value["id"]),
            root_path=str(value.get("root_path") or ""),
            created_at=str(value.get("created_at") or utc_now()),
            updated_at=str(value.get("updated_at") or value.get("created_at") or utc_now()),
            agent_profile=str(value.get("agent_profile") or "mtplx-agent"),
            model=str(value["model"]) if value.get("model") else None,
            instructions=str(value.get("instructions") or ""),
            tool_policy={
                **DEFAULT_TOOL_POLICY,
                **(
                    {str(key): str(item) for key, item in policy.items()}
                    if isinstance(policy, Mapping)
                    else {}
                ),
            },
        )

    @staticmethod
    def _decode_run(value: Mapping[str, Any]) -> AgentRun:
        return AgentRun(
            id=str(value["id"]),
            workspace_id=str(value["workspace_id"]),
            session_id=str(value["session_id"]),
            title=str(value.get("title") or "Agent run"),
            status=str(value.get("status") or "queued"),
            created_at=str(value.get("created_at") or utc_now()),
            updated_at=str(value.get("updated_at") or value.get("created_at") or utc_now()),
            model=str(value["model"]) if value.get("model") else None,
            event_count=int(value.get("event_count") or 0),
            last_event_at=(str(value["last_event_at"]) if value.get("last_event_at") else None),
            error=(str(value["error"]) if value.get("error") else None),
        )

    @staticmethod
    def _decode_approval(value: Mapping[str, Any]) -> ApprovalRequest:
        return ApprovalRequest(
            id=str(value["id"]),
            workspace_id=str(value["workspace_id"]),
            run_id=str(value["run_id"]) if value.get("run_id") else None,
            tool=str(value.get("tool") or "unknown"),
            action=str(value.get("action") or "tool action"),
            description=str(value.get("description") or ""),
            target=str(value["target"]) if value.get("target") else None,
            risk=str(value.get("risk") or "medium"),
            status=str(value.get("status") or "pending"),
            created_at=str(value.get("created_at") or utc_now()),
            arguments=(
                dict(value.get("arguments") or {})
                if isinstance(value.get("arguments"), Mapping)
                else {}
            ),
            arguments_sha256=(
                str(value["arguments_sha256"])
                if value.get("arguments_sha256")
                else None
            ),
            expires_at=str(value["expires_at"]) if value.get("expires_at") else None,
            resolved_at=str(value["resolved_at"]) if value.get("resolved_at") else None,
            resolved_by=str(value["resolved_by"]) if value.get("resolved_by") else None,
            reason=str(value["reason"]) if value.get("reason") else None,
            consumed_at=str(value["consumed_at"]) if value.get("consumed_at") else None,
            consumed_by=str(value["consumed_by"]) if value.get("consumed_by") else None,
        )

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WorkspaceNotFoundError(path.stem) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise WorkspaceStoreError(f"invalid workspace state: {path}") from exc
        if not isinstance(value, dict):
            raise WorkspaceStoreError(f"invalid object state: {path}")
        return value

    def get_workspace(self, workspace_id: str) -> Workspace:
        with self._lock:
            return self._decode_workspace(self._read_json(self._workspace_path(workspace_id)))

    def list_workspaces(self, *, limit: int = 100) -> list[Workspace]:
        bounded = max(1, min(int(limit), 1000))
        with self._lock:
            self.ensure_layout()
            result: list[Workspace] = []
            for path in sorted(self.root.joinpath("workspaces").glob("*.json")):
                try:
                    result.append(self._decode_workspace(self._read_json(path)))
                except WorkspaceStoreError:
                    continue
                if len(result) >= bounded:
                    break
            result.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
            return result

    def create_workspace(
        self,
        name: str,
        root_path: str,
        *,
        workspace_id: str | None = None,
        agent_profile: str = "mtplx-agent",
        model: str | None = None,
        instructions: str = "",
        tool_policy: Mapping[str, str] | None = None,
    ) -> Workspace:
        clean_name = str(name or "").strip()
        clean_root = str(root_path or "").strip()
        if not clean_name:
            raise WorkspaceStoreError("workspace name is required")
        if not clean_root:
            raise WorkspaceStoreError("workspace root_path is required")
        resolved_root = Path(clean_root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise WorkspaceStoreError(f"workspace root_path is not a directory: {resolved_root}")
        normalized_policy = self._normalize_policy(tool_policy)
        identifier = safe_id(workspace_id or clean_name.lower().replace(" ", "-"), fallback="workspace")
        now = utc_now()
        workspace = Workspace(
            id=identifier,
            name=clean_name,
            root_path=str(resolved_root),
            created_at=now,
            updated_at=now,
            agent_profile=str(agent_profile or "mtplx-agent"),
            model=str(model) if model else None,
            instructions=str(instructions or ""),
            tool_policy=normalized_policy,
        )
        with self._exclusive():
            target = self._workspace_path(identifier)
            if target.exists():
                raise WorkspaceConflictError(f"workspace already exists: {identifier}")
            _atomic_write(target, json.dumps(workspace.to_dict(), indent=2, sort_keys=True) + "\n")
        return workspace

    def update_workspace(self, workspace_id: str, **changes: Any) -> Workspace:
        allowed = {"name", "root_path", "agent_profile", "model", "instructions", "tool_policy"}
        unknown = set(changes) - allowed
        if unknown:
            raise WorkspaceStoreError(f"unknown workspace fields: {', '.join(sorted(unknown))}")
        with self._exclusive():
            current = self._decode_workspace(self._read_json(self._workspace_path(workspace_id)))
            value = current.to_dict()
            for key, item in changes.items():
                if item is None and key == "model":
                    value[key] = None
                elif key == "root_path":
                    resolved_root = Path(str(item)).expanduser().resolve()
                    if not resolved_root.is_dir():
                        raise WorkspaceStoreError(
                            f"workspace root_path is not a directory: {resolved_root}"
                        )
                    value[key] = str(resolved_root)
                elif key == "tool_policy":
                    value[key] = self._normalize_policy(item, base=current.tool_policy)
                else:
                    value[key] = str(item)
            value["updated_at"] = utc_now()
            updated = self._decode_workspace(value)
            _atomic_write(
                self._workspace_path(updated.id),
                json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
            )
            return updated

    def create_run(
        self,
        workspace_id: str,
        *,
        session_id: str | None = None,
        title: str = "Agent run",
        model: str | None = None,
        run_id: str | None = None,
    ) -> AgentRun:
        workspace = self.get_workspace(workspace_id)
        now = utc_now()
        run = AgentRun(
            id=safe_id(run_id or _new_id("run"), fallback="run"),
            workspace_id=workspace.id,
            session_id=str(session_id or uuid.uuid4()),
            title=str(title or "Agent run"),
            status="queued",
            created_at=now,
            updated_at=now,
            model=str(model or workspace.model) if (model or workspace.model) else None,
        )
        with self._exclusive():
            target = self._run_path(run.id)
            if target.exists():
                raise WorkspaceConflictError(f"run already exists: {run.id}")
            _atomic_write(target, json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n")
        self.append_event(
            run.id,
            "run_created",
            {
                "workspace_id": workspace.id,
                "session_id": run.session_id,
                "title": run.title,
                "model": run.model,
            },
        )
        return run

    @staticmethod
    def _normalize_policy(
        policy: Mapping[str, str] | None,
        *,
        base: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        result = {**DEFAULT_TOOL_POLICY, **{str(k): str(v) for k, v in (base or {}).items()}}
        for key, value in (policy or {}).items():
            mode = str(value).strip().lower()
            if mode not in POLICY_MODES:
                raise WorkspaceStoreError(
                    f"unknown policy mode for {key}: {value}; use allow, ask, or deny"
                )
            result[str(key)] = mode
        return result

    def get_run(self, run_id: str) -> AgentRun:
        with self._lock:
            return self._decode_run(self._read_json(self._run_path(run_id)))

    def list_runs(self, workspace_id: str | None = None, *, limit: int = 100) -> list[AgentRun]:
        bounded = max(1, min(int(limit), 1000))
        with self._lock:
            self.ensure_layout()
            result: list[AgentRun] = []
            for path in self.root.joinpath("runs").glob("*.json"):
                try:
                    run = self._decode_run(self._read_json(path))
                except WorkspaceStoreError:
                    continue
                if workspace_id and run.workspace_id != workspace_id:
                    continue
                result.append(run)
            result.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
            return result[:bounded]

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        error: str | None = None,
    ) -> AgentRun:
        if status is not None and status not in WORKSPACE_RUN_STATUSES:
            raise WorkspaceStoreError(f"unknown run status: {status}")
        transition_kind: str | None = None
        transition_payload: dict[str, Any] = {}
        with self._exclusive():
            value = self._read_json(self._run_path(run_id))
            previous_status = str(value.get("status") or "queued")
            if status is not None:
                value["status"] = status
            if title is not None:
                value["title"] = str(title)
            if error is not None:
                value["error"] = str(error)
            value["updated_at"] = utc_now()
            run = self._decode_run(value)
            _atomic_write(self._run_path(run.id), json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n")
            if status is not None and status != previous_status:
                transition_kind = {
                    "running": "agent_started",
                    "paused": "run_paused",
                    "completed": "run_completed",
                    "failed": "run_failed",
                    "cancelled": "run_cancelled",
                }.get(status)
                transition_payload = {
                    "from": previous_status,
                    "to": status,
                }
                if error:
                    transition_payload["error"] = str(error)
        if transition_kind:
            self.append_event(run.id, transition_kind, transition_payload)
        return run

    def recover_incomplete_runs(self) -> list[AgentRun]:
        recovered: list[AgentRun] = []
        for run in self.list_runs(limit=5000):
            if run.status not in {"queued", "running"}:
                continue
            recovered.append(
                self.update_run(
                    run.id,
                    status="paused",
                    error="MTPLX restarted before this run reached a terminal state",
                )
            )
        return recovered

    def resume_run(self, run_id: str) -> AgentRun:
        """Move a restart-paused run back to the queue and record the resume."""
        run = self.get_run(run_id)
        if run.status not in {"paused", "queued"}:
            return run
        resumed = self.update_run(run_id, status="queued", error="")
        self.append_event(
            run_id,
            "run_resumed",
            {"from": run.status, "to": resumed.status},
        )
        return resumed

    def append_event(
        self,
        run_id: str,
        kind: str,
        payload: Mapping[str, Any] | None = None,
    ) -> RunEvent:
        clean_kind = str(kind or "event").strip()
        if not clean_kind:
            raise WorkspaceStoreError("event kind is required")
        with self._exclusive():
            run = self._decode_run(self._read_json(self._run_path(run_id)))
            created = utc_now()
            event = RunEvent(
                id=_new_id("evt"),
                run_id=run.id,
                sequence=run.event_count + 1,
                kind=clean_kind,
                payload=dict(payload or {}),
                created_at=created,
            )
            events_path = self._events_path(run.id)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            updated = AgentRun(
                **{
                    **run.to_dict(),
                    "event_count": event.sequence,
                    "last_event_at": created,
                    "updated_at": created,
                }
            )
            _atomic_write(self._run_path(run.id), json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n")
            return event

    def list_events(self, run_id: str, *, limit: int = 500, after: int = 0) -> list[RunEvent]:
        run = self.get_run(run_id)
        bounded = max(1, min(int(limit), 5000))
        events_path = self._events_path(run.id)
        if not events_path.exists():
            return []
        result: list[RunEvent] = []
        with self._lock, events_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                    if int(value.get("sequence") or 0) <= int(after):
                        continue
                    result.append(
                        RunEvent(
                            id=str(value["id"]),
                            run_id=str(value["run_id"]),
                            sequence=int(value["sequence"]),
                            kind=str(value["kind"]),
                            payload=dict(value.get("payload") or {}),
                            created_at=str(value["created_at"]),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if len(result) >= bounded:
                    break
        return result

    def create_approval(
        self,
        workspace_id: str,
        *,
        run_id: str | None,
        tool: str,
        action: str,
        description: str,
        target: str | None = None,
        risk: str = "medium",
        arguments: Mapping[str, Any] | None = None,
        expires_in_seconds: int = 600,
    ) -> ApprovalRequest:
        self.get_workspace(workspace_id)
        if run_id is not None and self.get_run(run_id).workspace_id != workspace_id:
            raise WorkspaceConflictError("run does not belong to workspace")
        approval = ApprovalRequest(
            id=_new_id("approval"),
            workspace_id=workspace_id,
            run_id=run_id,
            tool=str(tool or "unknown"),
            action=str(action or "tool action"),
            description=str(description or ""),
            target=str(target) if target else None,
            risk=str(risk or "medium"),
            status="pending",
            created_at=utc_now(),
            arguments=dict(arguments or {}),
            arguments_sha256=approval_arguments_sha256(arguments),
            expires_at=(
                datetime.fromtimestamp(
                    datetime.now(timezone.utc).timestamp()
                    + max(30, min(int(expires_in_seconds), 86_400)),
                    tz=timezone.utc,
                ).isoformat().replace("+00:00", "Z")
            ),
        )
        with self._exclusive():
            _atomic_write(
                self._approval_path(approval.id),
                json.dumps(approval.to_dict(), indent=2, sort_keys=True) + "\n",
            )
        if run_id:
            self.append_event(run_id, "approval_requested", approval.to_dict())
        return approval

    def get_approval(self, approval_id: str) -> ApprovalRequest:
        with self._lock:
            approval = self._decode_approval(self._read_json(self._approval_path(approval_id)))
        return self._expire_if_needed(approval)

    @staticmethod
    def _deadline_expired(approval: ApprovalRequest) -> bool:
        if not approval.expires_at:
            return False
        try:
            expires_at = datetime.fromisoformat(approval.expires_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        return expires_at <= datetime.now(timezone.utc)

    @classmethod
    def _is_expired(cls, approval: ApprovalRequest) -> bool:
        return approval.status == "pending" and cls._deadline_expired(approval)

    def _expire_if_needed(self, approval: ApprovalRequest) -> ApprovalRequest:
        if not self._is_expired(approval):
            return approval
        try:
            return self.resolve_approval(
                approval.id,
                "expired",
                resolved_by="system",
                reason="approval expired",
            )
        except WorkspaceConflictError:
            return self._decode_approval(self._read_json(self._approval_path(approval.id)))

    def list_approvals(
        self,
        *,
        workspace_id: str | None = None,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        bounded = max(1, min(int(limit), 1000))
        with self._lock:
            self.ensure_layout()
            candidates: list[ApprovalRequest] = []
            for path in self.root.joinpath("approvals").glob("*.json"):
                try:
                    approval = self._decode_approval(self._read_json(path))
                except WorkspaceStoreError:
                    continue
                if workspace_id and approval.workspace_id != workspace_id:
                    continue
                if run_id and approval.run_id != run_id:
                    continue
                candidates.append(approval)
        result = []
        for approval in candidates:
            current = self._expire_if_needed(approval)
            if status and current.status != status:
                continue
            result.append(current)
        result.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return result[:bounded]

    def resolve_approval(
        self,
        approval_id: str,
        decision: str,
        *,
        resolved_by: str = "user",
        reason: str | None = None,
    ) -> ApprovalRequest:
        if decision not in {"approved", "denied", "expired"}:
            raise WorkspaceStoreError("approval decision must be approved, denied, or expired")
        with self._exclusive():
            current = self._decode_approval(self._read_json(self._approval_path(approval_id)))
            if current.status != "pending":
                raise WorkspaceConflictError(f"approval is already {current.status}: {approval_id}")
            if self._is_expired(current):
                decision = "expired"
                reason = reason or "approval expired"
                resolved_by = "system"
            updated = ApprovalRequest(
                **{
                    **current.to_dict(),
                    "status": decision,
                    "resolved_at": utc_now(),
                    "resolved_by": str(resolved_by or "user"),
                    "reason": str(reason) if reason else None,
                }
            )
            _atomic_write(
                self._approval_path(updated.id),
                json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
            )
        if updated.run_id:
            self.append_event(updated.run_id, "approval_resolved", updated.to_dict())
        return updated

    def consume_approval(
        self,
        approval_id: str,
        *,
        workspace_id: str,
        run_id: str | None,
        tool: str,
        arguments: Mapping[str, Any] | None,
        consumed_by: str = "tool-executor",
    ) -> ApprovalRequest:
        """Atomically consume one exact, approved tool authorization."""
        actual_hash = approval_arguments_sha256(arguments)
        expired: ApprovalRequest | None = None
        with self._exclusive():
            current = self._decode_approval(
                self._read_json(self._approval_path(approval_id))
            )
            if self._deadline_expired(current):
                expired = ApprovalRequest(
                    **{
                        **current.to_dict(),
                        "status": "expired",
                        "resolved_at": utc_now(),
                        "resolved_by": "system",
                        "reason": "approval expired before execution",
                    }
                )
                _atomic_write(
                    self._approval_path(expired.id),
                    json.dumps(expired.to_dict(), indent=2, sort_keys=True) + "\n",
                )
            elif current.status != "approved":
                raise WorkspaceConflictError(
                    f"approval is not executable ({current.status}): {approval_id}"
                )
            elif current.workspace_id != workspace_id:
                raise WorkspaceConflictError("approval belongs to a different workspace")
            elif current.run_id != run_id:
                raise WorkspaceConflictError("approval belongs to a different run")
            elif current.tool != str(tool):
                raise WorkspaceConflictError("approval is bound to a different tool")
            elif not current.arguments_sha256:
                raise WorkspaceConflictError("approval is not bound to exact arguments")
            elif current.arguments_sha256 != actual_hash:
                raise WorkspaceConflictError("approval arguments do not match execution")
            else:
                updated = ApprovalRequest(
                    **{
                        **current.to_dict(),
                        "status": "consumed",
                        "consumed_at": utc_now(),
                        "consumed_by": str(consumed_by or "tool-executor"),
                    }
                )
                _atomic_write(
                    self._approval_path(updated.id),
                    json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
                )
        if expired is not None:
            if expired.run_id:
                self.append_event(expired.run_id, "approval_resolved", expired.to_dict())
            raise WorkspaceConflictError(f"approval is expired: {approval_id}")
        if updated.run_id:
            self.append_event(updated.run_id, "approval_consumed", updated.to_dict())
        return updated

    def status(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "exists": self.root.exists(),
            "workspaces": len(self.list_workspaces(limit=1000)),
            "runs": len(self.list_runs(limit=1000)),
            "pending_approvals": len(self.list_approvals(status="pending", limit=1000)),
        }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "APPROVAL_STATUSES",
    "AgentRun",
    "ApprovalRequest",
    "DEFAULT_TOOL_POLICY",
    "RunEvent",
    "WORKSPACE_RUN_STATUSES",
    "Workspace",
    "WorkspaceConflictError",
    "WorkspaceNotFoundError",
    "WorkspaceStore",
    "WorkspaceStoreError",
    "approval_arguments_sha256",
]
