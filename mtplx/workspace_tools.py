"""Policy-bound first-party coding tools for MTPLX workspaces.

This is the shared execution boundary for desktop, CLI, API, and delegated
agents. It validates every path against a registered workspace, binds risky
actions to one exact approval, filters child environments, and appends tool
evidence to the durable run log.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .agent_workspace import (
    ApprovalRequest,
    Workspace,
    WorkspaceConflictError,
    WorkspaceStore,
    WorkspaceStoreError,
    _atomic_write,
    approval_arguments_sha256,
    safe_id,
)


FIRST_PARTY_TOOL_NAMES: tuple[str, ...] = (
    "list_files",
    "read_file",
    "search_files",
    "inspect_repo",
    "git_status",
    "git_diff",
    "write_file",
    "apply_patch",
    "run_tests",
    "run_command",
)

TOOL_POLICY_KEY: dict[str, str] = {
    "list_files": "read",
    "read_file": "read",
    "search_files": "search",
    "inspect_repo": "read",
    "git_status": "read",
    "git_diff": "read",
    "write_file": "write",
    "apply_patch": "write",
    "run_tests": "terminal",
    "run_command": "terminal",
}

MUTATING_TOOLS = frozenset({"write_file", "apply_patch", "run_tests", "run_command"})
FILE_CHANGING_TOOLS = frozenset({"write_file", "apply_patch"})
EXTERNAL_ACTION_POLICY_KEYS: dict[str, tuple[str, ...]] = {
    "web_search": ("browser", "network"),
    "fetch_url": ("browser", "network"),
    "browser_open": ("browser", "network"),
    "external_write": ("write", "network"),
}

_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".build",
        ".swiftpm",
        "DerivedData",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mtplx",
    }
)
_SAFE_ENV_EXAMPLES = frozenset({".env.example", ".env.sample", ".env.template"})
_SECRET_FILE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "id_rsa",
        "id_ed25519",
        "credentials",
    }
)
_SECRET_DIRECTORIES = frozenset({".ssh", ".aws", ".gnupg"})
_MAX_READ_BYTES = 2 * 1024 * 1024
_MAX_READ_CHARACTERS = 40_000
_MAX_SEARCH_BYTES = 1 * 1024 * 1024
_MAX_RESULTS = 100
_MAX_WRITE_CHARACTERS = 1_000_000
_MAX_PATCH_CHARACTERS = 2_000_000
_MAX_COMMAND_CHARACTERS = 10_000
_MAX_STDOUT_CHARACTERS = 40_000
_MAX_STDERR_CHARACTERS = 20_000


class WorkspaceToolError(WorkspaceStoreError):
    pass


class WorkspaceToolPermissionError(WorkspaceToolError):
    pass


@dataclass(frozen=True)
class ToolPreview:
    workspace_id: str
    run_id: str | None
    tool: str
    arguments: dict[str, Any]
    arguments_sha256: str
    policy_keys: tuple[str, ...]
    policy_modes: dict[str, str]
    effective_mode: str
    action: str
    description: str
    target: str | None
    risk: str
    root_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "run_id": self.run_id,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "arguments_sha256": self.arguments_sha256,
            "policy_keys": list(self.policy_keys),
            "policy_modes": dict(self.policy_modes),
            "effective_mode": self.effective_mode,
            "action": self.action,
            "description": self.description,
            "target": self.target,
            "risk": self.risk,
            "root_path": self.root_path,
        }


def _schema(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def first_party_tool_definitions() -> list[dict[str, Any]]:
    def string(description: str) -> dict[str, str]:
        return {"type": "string", "description": description}

    def integer(description: str) -> dict[str, str]:
        return {"type": "integer", "description": description}

    def boolean(description: str) -> dict[str, str]:
        return {"type": "boolean", "description": description}

    return [
        _schema(
            "list_files",
            "List files and directories inside the selected MTPLX workspace.",
            {
                "path": string("Relative directory path. Defaults to the workspace root."),
                "depth": integer("Maximum directory depth, from 0 to 4."),
            },
        ),
        _schema(
            "read_file",
            "Read a UTF-8 text file inside the selected MTPLX workspace.",
            {
                "path": string("Relative file path inside the workspace."),
                "max_chars": integer("Maximum characters to return, capped at 40000."),
            },
            ("path",),
        ),
        _schema(
            "search_files",
            "Search text files inside the workspace and return matching lines.",
            {
                "query": string("Text to find."),
                "path": string("Optional relative directory to search."),
                "case_sensitive": boolean("Whether matching is case-sensitive."),
                "max_results": integer("Maximum matches, capped at 100."),
            },
            ("query",),
        ),
        _schema(
            "inspect_repo",
            "Inspect repository branch, project markers, and top-level layout.",
            {},
        ),
        _schema("git_status", "Read repository branch and changed-file status.", {}),
        _schema(
            "git_diff",
            "Read the working-tree or staged Git diff.",
            {
                "scope": string("unstaged, staged, or both."),
                "path": string("Optional relative path to limit the diff."),
            },
        ),
        _schema(
            "write_file",
            "Write complete UTF-8 content to one workspace file after approval.",
            {
                "path": string("Relative file path inside the workspace."),
                "content": string("Complete replacement file content."),
            },
            ("path", "content"),
        ),
        _schema(
            "apply_patch",
            "Validate and apply a unified Git patch after approval.",
            {"patch": string("Unified diff to validate and apply.")},
            ("patch",),
        ),
        _schema(
            "run_tests",
            "Run detected or explicit repository tests after approval.",
            {
                "command": string("Optional test command."),
                "network": boolean("Allow network access for this test command."),
                "timeout_seconds": integer("Timeout from 1 to 900 seconds."),
            },
        ),
        _schema(
            "run_command",
            "Run an exact shell command in the workspace after approval.",
            {
                "command": string("Exact command to run."),
                "network": boolean("Allow network access for this command."),
                "timeout_seconds": integer("Timeout from 1 to 300 seconds."),
            },
            ("command",),
        ),
    ]


class WorkspaceToolService:
    """Execute the MTPLX coding tools through one durable policy boundary."""

    def __init__(
        self,
        workspace_store: WorkspaceStore,
        *,
        sandbox_mode: str | None = None,
    ) -> None:
        self.workspace_store = workspace_store
        self.executions_root = workspace_store.root / "executions"
        self._execution_lock = threading.RLock()
        configured = sandbox_mode or os.environ.get("MTPLX_AGENT_SANDBOX") or "auto"
        self.sandbox_mode = str(configured).strip().lower()
        if self.sandbox_mode not in {"auto", "required", "off"}:
            raise WorkspaceToolError("sandbox mode must be auto, required, or off")

    def definitions(self, permissions: Iterable[str] | None = None) -> list[dict[str, Any]]:
        allowed = None if permissions is None else {str(item) for item in permissions}
        result = []
        for definition in first_party_tool_definitions():
            name = str(definition["function"]["name"])
            if allowed is None or self._permission_allows(allowed, name, {}):
                result.append(definition)
        return result

    def authorize_external_action(
        self,
        workspace_id: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        run_id: str | None = None,
        approval_id: str | None = None,
        executor_id: str = "desktop",
    ) -> dict[str, Any]:
        """Issue a one-use authorization receipt before client-side I/O."""
        name = str(tool or "").strip()
        policy_keys = EXTERNAL_ACTION_POLICY_KEYS.get(name)
        if policy_keys is None:
            raise WorkspaceToolError(f"unknown external action: {name or '<empty>'}")
        workspace = self.workspace_store.get_workspace(workspace_id)
        if run_id is not None:
            run = self.workspace_store.get_run(run_id)
            if run.workspace_id != workspace.id:
                raise WorkspaceConflictError("run does not belong to workspace")
        try:
            encoded = json.dumps(
                dict(arguments or {}),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            normalized = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkspaceToolError("external action arguments must be JSON-compatible") from exc
        if len(encoded) > 100_000:
            raise WorkspaceToolError("external action arguments exceed 100000 characters")
        modes = {key: workspace.tool_policy.get(key, "ask") for key in policy_keys}
        effective = (
            "deny"
            if "deny" in modes.values()
            else "ask"
            if "ask" in modes.values()
            else "allow"
        )
        target = str(normalized.get("url") or normalized.get("query") or name)[:2000]
        preview = ToolPreview(
            workspace_id=workspace.id,
            run_id=run_id,
            tool=name,
            arguments=normalized,
            arguments_sha256=approval_arguments_sha256(normalized),
            policy_keys=policy_keys,
            policy_modes=modes,
            effective_mode=effective,
            action=f"Authorize {name}",
            description=f"Authorize one exact {name} action against {target}.",
            target=target,
            risk="high" if name == "external_write" else "medium",
            root_path=workspace.root_path,
        )
        if effective == "deny":
            return {
                "ok": False,
                "status": "denied",
                "error": "workspace_policy_denied",
                "preview": preview.to_dict(),
            }
        if effective == "ask":
            if not approval_id:
                approval = self.request_approval(preview)
                return {
                    "ok": False,
                    "status": "approval_required",
                    "error": "approval_required",
                    "preview": preview.to_dict(),
                    "approval": approval.to_dict(),
                }
            try:
                self.workspace_store.consume_approval(
                    approval_id,
                    workspace_id=workspace.id,
                    run_id=run_id,
                    tool=name,
                    arguments=normalized,
                    consumed_by=executor_id,
                )
            except WorkspaceStoreError as exc:
                return {
                    "ok": False,
                    "status": "approval_invalid",
                    "error": str(exc),
                    "preview": preview.to_dict(),
                }
        receipt = {
            "tool": name,
            "arguments_sha256": preview.arguments_sha256,
            "policy": modes,
            "approval_id": approval_id,
            "executor_id": executor_id,
        }
        if run_id:
            self.workspace_store.append_event(
                run_id,
                "external_action_authorized",
                receipt,
            )
        return {
            "ok": True,
            "status": "authorized",
            "tool": name,
            "arguments_sha256": preview.arguments_sha256,
            "approval_id": approval_id,
            "result": {"authorized": True, **receipt},
        }

    def preview(
        self,
        workspace_id: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        run_id: str | None = None,
        root_override: str | os.PathLike[str] | None = None,
        permissions: Iterable[str] | None = None,
        policy_overrides: Mapping[str, str] | None = None,
    ) -> ToolPreview:
        name = str(tool or "").strip()
        if name not in FIRST_PARTY_TOOL_NAMES:
            raise WorkspaceToolError(f"unknown workspace tool: {name or '<empty>'}")
        workspace = self.workspace_store.get_workspace(workspace_id)
        if run_id is not None:
            run = self.workspace_store.get_run(run_id)
            if run.workspace_id != workspace.id:
                raise WorkspaceConflictError("run does not belong to workspace")
        root = self._root(workspace, root_override)
        normalized = self._normalize_arguments(name, arguments or {}, root)
        permission_set = None if permissions is None else {str(item) for item in permissions}
        if permission_set is not None and not self._permission_allows(
            permission_set, name, normalized
        ):
            raise WorkspaceToolPermissionError(f"agent profile denies tool: {name}")
        policy_keys = [TOOL_POLICY_KEY[name]]
        if name in {"run_tests", "run_command"} and bool(normalized.get("network")):
            policy_keys.append("network")
        rank = {"allow": 0, "ask": 1, "deny": 2}
        modes: dict[str, str] = {}
        for key in policy_keys:
            workspace_mode = workspace.tool_policy.get(key, "ask")
            override_mode = str((policy_overrides or {}).get(key) or "allow").lower()
            if override_mode not in rank:
                raise WorkspaceToolPermissionError(
                    f"invalid policy override for {key}: {override_mode}"
                )
            modes[key] = max((workspace_mode, override_mode), key=rank.__getitem__)
        if "deny" in modes.values():
            effective = "deny"
        elif "ask" in modes.values():
            effective = "ask"
        else:
            effective = "allow"
        action, description, target, risk = self._describe(name, normalized)
        return ToolPreview(
            workspace_id=workspace.id,
            run_id=run_id,
            tool=name,
            arguments=normalized,
            arguments_sha256=approval_arguments_sha256(normalized),
            policy_keys=tuple(policy_keys),
            policy_modes=modes,
            effective_mode=effective,
            action=action,
            description=description,
            target=target,
            risk=risk,
            root_path=str(root),
        )

    def request_approval(self, preview: ToolPreview) -> ApprovalRequest:
        if preview.effective_mode == "deny":
            raise WorkspaceToolPermissionError(
                f"workspace policy denies {preview.tool}: {preview.policy_modes}"
            )
        for approval in self.workspace_store.list_approvals(
            workspace_id=preview.workspace_id,
            run_id=preview.run_id,
            status="pending",
            limit=100,
        ):
            if (
                approval.tool == preview.tool
                and approval.arguments_sha256 == preview.arguments_sha256
            ):
                return approval
        return self.workspace_store.create_approval(
            preview.workspace_id,
            run_id=preview.run_id,
            tool=preview.tool,
            action=preview.action,
            description=preview.description,
            target=preview.target,
            risk=preview.risk,
            arguments=preview.arguments,
        )

    def execute(
        self,
        workspace_id: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        run_id: str | None = None,
        approval_id: str | None = None,
        root_override: str | os.PathLike[str] | None = None,
        permissions: Iterable[str] | None = None,
        executor_id: str = "primary",
        idempotency_key: str | None = None,
        policy_overrides: Mapping[str, str] | None = None,
        cancellation_event: threading.Event | None = None,
        execution_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        preview = self.preview(
            workspace_id,
            tool,
            arguments,
            run_id=run_id,
            root_override=root_override,
            permissions=permissions,
            policy_overrides=policy_overrides,
        )
        if preview.effective_mode == "deny":
            return {
                "ok": False,
                "status": "denied",
                "error": "workspace_policy_denied",
                "preview": preview.to_dict(),
            }
        replay = self._execution_replay(preview, idempotency_key)
        if replay is not None:
            return replay
        if preview.effective_mode == "ask":
            if not approval_id:
                approval = self.request_approval(preview)
                return {
                    "ok": False,
                    "status": "approval_required",
                    "error": "approval_required",
                    "preview": preview.to_dict(),
                    "approval": approval.to_dict(),
                }
            try:
                self.workspace_store.consume_approval(
                    approval_id,
                    workspace_id=preview.workspace_id,
                    run_id=preview.run_id,
                    tool=preview.tool,
                    arguments=preview.arguments,
                    consumed_by=executor_id,
                )
            except WorkspaceStoreError as exc:
                return {
                    "ok": False,
                    "status": "approval_invalid",
                    "error": str(exc),
                    "preview": preview.to_dict(),
                }
        claimed = self._claim_execution(preview, idempotency_key, executor_id)
        if claimed is not None:
            return claimed
        if run_id:
            self.workspace_store.append_event(
                run_id,
                "tool_call",
                {
                    "tool": preview.tool,
                    "arguments": preview.arguments,
                    "arguments_sha256": preview.arguments_sha256,
                    "approval_id": approval_id,
                    "executor_id": executor_id,
                    "root_path": preview.root_path,
                },
            )
            if preview.tool == "run_tests":
                self.workspace_store.append_event(
                    run_id,
                    "test_started",
                    {"command": preview.arguments["command"], "executor_id": executor_id},
                )
        started = time.monotonic()
        try:
            if cancellation_event is not None and cancellation_event.is_set():
                result = {"error": "cancelled", "cancelled": True}
            else:
                result = self._dispatch(
                    preview,
                    cancellation_event=cancellation_event,
                    execution_timeout_seconds=execution_timeout_seconds,
                )
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        elapsed_ms = int((time.monotonic() - started) * 1000)
        succeeded = self._succeeded(result)
        response = {
            "ok": succeeded,
            "status": "completed" if succeeded else "failed",
            "tool": preview.tool,
            "arguments_sha256": preview.arguments_sha256,
            "policy": preview.policy_modes,
            "approval_id": approval_id,
            "elapsed_ms": elapsed_ms,
            "result": result,
            "idempotency_key": idempotency_key,
        }
        if run_id:
            self.workspace_store.append_event(run_id, "tool_result", response)
            if preview.tool in FILE_CHANGING_TOOLS and succeeded:
                self.workspace_store.append_event(
                    run_id,
                    "file_changed",
                    {
                        "tool": preview.tool,
                        "target": preview.target,
                        "arguments_sha256": preview.arguments_sha256,
                        "executor_id": executor_id,
                    },
                )
            if preview.tool == "run_tests":
                self.workspace_store.append_event(
                    run_id,
                    "test_completed",
                    {
                        "command": preview.arguments["command"],
                        "passed": succeeded,
                        "exit_code": result.get("exit_code"),
                        "timed_out": bool(result.get("timed_out")),
                        "cancelled": bool(result.get("cancelled")),
                        "elapsed_ms": elapsed_ms,
                        "executor_id": executor_id,
                    },
                )
        self._complete_execution(preview, idempotency_key, response)
        return response

    def _execution_path(self, idempotency_key: str) -> Path:
        identifier = safe_id(idempotency_key, fallback="execution")
        digest = approval_arguments_sha256({"idempotency_key": idempotency_key})[:16]
        return self.executions_root / f"{identifier[:120]}-{digest}.json"

    def _execution_replay(
        self,
        preview: ToolPreview,
        idempotency_key: str | None,
    ) -> dict[str, Any] | None:
        if not idempotency_key:
            return None
        path = self._execution_path(idempotency_key)
        with self._execution_lock, self.workspace_store._exclusive():
            if not path.is_file():
                return None
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise WorkspaceToolError("invalid idempotent execution record") from exc
        expected = {
            "workspace_id": preview.workspace_id,
            "run_id": preview.run_id,
            "tool": preview.tool,
            "arguments_sha256": preview.arguments_sha256,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            return {
                "ok": False,
                "status": "idempotency_conflict",
                "error": "idempotency key is bound to another exact action",
                "preview": preview.to_dict(),
            }
        if record.get("status") == "completed" and isinstance(record.get("response"), Mapping):
            return {
                **dict(record["response"]),
                "replayed": True,
                "idempotency_key": idempotency_key,
            }
        return {
            "ok": False,
            "status": "execution_indeterminate",
            "error": (
                "an execution with this key started but did not record completion; "
                "MTPLX will not duplicate the side effect"
            ),
            "preview": preview.to_dict(),
            "idempotency_key": idempotency_key,
        }

    def _claim_execution(
        self,
        preview: ToolPreview,
        idempotency_key: str | None,
        executor_id: str,
    ) -> dict[str, Any] | None:
        if not idempotency_key:
            return None
        record = {
            "schema_version": 1,
            "idempotency_key": idempotency_key,
            "workspace_id": preview.workspace_id,
            "run_id": preview.run_id,
            "tool": preview.tool,
            "arguments_sha256": preview.arguments_sha256,
            "executor_id": executor_id,
            "status": "started",
            "started_at": time.time(),
        }
        path = self._execution_path(idempotency_key)
        with self._execution_lock, self.workspace_store._exclusive():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    current = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise WorkspaceToolError(
                        "invalid idempotent execution record"
                    ) from exc
                expected = {
                    "workspace_id": preview.workspace_id,
                    "run_id": preview.run_id,
                    "tool": preview.tool,
                    "arguments_sha256": preview.arguments_sha256,
                }
                if any(current.get(key) != value for key, value in expected.items()):
                    return {
                        "ok": False,
                        "status": "idempotency_conflict",
                        "error": "idempotency key is bound to another exact action",
                        "preview": preview.to_dict(),
                    }
                if current.get("status") == "completed" and isinstance(
                    current.get("response"), Mapping
                ):
                    return {
                        **dict(current["response"]),
                        "replayed": True,
                        "idempotency_key": idempotency_key,
                    }
                return {
                    "ok": False,
                    "status": "execution_indeterminate",
                    "error": (
                        "an execution with this key is already in progress or did not "
                        "record completion; MTPLX will not duplicate the side effect"
                    ),
                    "preview": preview.to_dict(),
                    "idempotency_key": idempotency_key,
                }
            _atomic_write(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
        return None

    def _complete_execution(
        self,
        preview: ToolPreview,
        idempotency_key: str | None,
        response: Mapping[str, Any],
    ) -> None:
        if not idempotency_key:
            return
        path = self._execution_path(idempotency_key)
        with self._execution_lock, self.workspace_store._exclusive():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                record = {
                    "schema_version": 1,
                    "workspace_id": preview.workspace_id,
                    "run_id": preview.run_id,
                    "tool": preview.tool,
                    "arguments_sha256": preview.arguments_sha256,
                }
            record.update(
                {
                    "status": "completed",
                    "completed_at": time.time(),
                    "response": dict(response),
                }
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, json.dumps(record, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _root(
        workspace: Workspace,
        root_override: str | os.PathLike[str] | None,
    ) -> Path:
        root = Path(root_override or workspace.root_path).expanduser().resolve()
        if not root.is_dir():
            raise WorkspaceToolError(f"workspace root is not a directory: {root}")
        return root

    @staticmethod
    def _permission_allows(
        permissions: set[str],
        tool: str,
        arguments: Mapping[str, Any],
    ) -> bool:
        category = TOOL_POLICY_KEY[tool]
        if "all" not in permissions and tool not in permissions and category not in permissions:
            return False
        if bool(arguments.get("network")) and not (
            "all" in permissions or "network" in permissions
        ):
            return False
        return True

    def _normalize_arguments(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        root: Path,
    ) -> dict[str, Any]:
        value = dict(arguments)
        if tool == "list_files":
            return {
                "path": str(value.get("path") or ""),
                "depth": max(0, min(int(value.get("depth") or 2), 4)),
            }
        if tool == "read_file":
            return {
                "path": str(value.get("path") or ""),
                "max_chars": max(
                    1,
                    min(int(value.get("max_chars") or _MAX_READ_CHARACTERS), _MAX_READ_CHARACTERS),
                ),
            }
        if tool == "search_files":
            return {
                "query": str(value.get("query") or ""),
                "path": str(value.get("path") or ""),
                "case_sensitive": bool(value.get("case_sensitive", False)),
                "max_results": max(
                    1,
                    min(int(value.get("max_results") or 50), _MAX_RESULTS),
                ),
            }
        if tool == "git_diff":
            return {
                "scope": str(value.get("scope") or "unstaged").lower(),
                "path": str(value.get("path") or ""),
            }
        if tool == "write_file":
            return {
                "path": str(value.get("path") or ""),
                "content": str(value.get("content") or ""),
            }
        if tool == "apply_patch":
            return {"patch": str(value.get("patch") or "")}
        if tool in {"run_tests", "run_command"}:
            requested = str(value.get("command") or "").strip()
            if tool == "run_tests" and not requested:
                requested = self._detected_test_command(root) or ""
            maximum = 900 if tool == "run_tests" else 300
            fallback = 300 if tool == "run_tests" else 60
            return {
                "command": requested,
                "network": bool(value.get("network", False)),
                "timeout_seconds": max(
                    1,
                    min(int(value.get("timeout_seconds") or fallback), maximum),
                ),
            }
        return {}

    @staticmethod
    def _describe(
        tool: str,
        arguments: Mapping[str, Any],
    ) -> tuple[str, str, str | None, str]:
        if tool == "write_file":
            path = str(arguments["path"])
            size = len(str(arguments["content"]).encode("utf-8"))
            return (
                f"Write {path}",
                f"Replace {path} with exactly {size} UTF-8 bytes.",
                path,
                "medium",
            )
        if tool == "apply_patch":
            digest = approval_arguments_sha256(arguments)
            return (
                "Apply Git patch",
                f"Apply the exact approved patch with argument hash {digest}.",
                "workspace diff",
                "medium",
            )
        if tool in {"run_tests", "run_command"}:
            command = str(arguments["command"])
            network = " with network access" if arguments.get("network") else " without network access"
            risk = "critical" if _looks_destructive(command) else "high"
            action = "Run tests" if tool == "run_tests" else "Run command"
            return action, f"Execute exactly: {command}{network}.", command, risk
        target = str(arguments.get("path") or ".") if arguments else "."
        return f"Run {tool}", f"Execute read-only tool {tool} on {target}.", target, "low"

    def _dispatch(
        self,
        preview: ToolPreview,
        *,
        cancellation_event: threading.Event | None = None,
        execution_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        root = Path(preview.root_path)
        arguments = preview.arguments
        if preview.tool == "run_tests":
            return self._run_tests(
                arguments,
                root,
                cancellation_event=cancellation_event,
            )
        if preview.tool == "run_command":
            return self._run_command(
                arguments,
                root,
                cancellation_event=cancellation_event,
            )
        if preview.tool == "apply_patch":
            return self._apply_patch(
                arguments,
                root,
                cancellation_event=cancellation_event,
                timeout_seconds=execution_timeout_seconds,
            )
        handlers = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "search_files": self._search_files,
            "inspect_repo": self._inspect_repo,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "write_file": self._write_file,
        }
        return handlers[preview.tool](arguments, root)

    def _resolve_path(self, root: Path, raw_path: str, *, sensitive: bool = True) -> Path:
        candidate = (root / (raw_path or ".")).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise WorkspaceToolError(f"path is outside workspace: {raw_path}") from exc
        if sensitive and self._is_sensitive(candidate, root):
            raise WorkspaceToolPermissionError(f"sensitive path is blocked: {raw_path}")
        return candidate

    @staticmethod
    def _is_sensitive(path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            return True
        names = relative.parts
        if any(item in _SECRET_DIRECTORIES for item in names):
            return True
        name = path.name
        if name in _SAFE_ENV_EXAMPLES:
            return False
        return name in _SECRET_FILE_NAMES or name.startswith(".env.")

    def _list_files(self, arguments: Mapping[str, Any], root: Path) -> dict[str, Any]:
        directory = self._resolve_path(root, str(arguments["path"]), sensitive=False)
        if not directory.is_dir():
            return {"error": "not_a_directory", "path": str(arguments["path"])}
        depth_limit = int(arguments["depth"])
        files: list[str] = []
        directories: list[str] = []
        base_depth = len(directory.relative_to(root).parts)
        for current, dirnames, filenames in os.walk(directory, followlinks=False):
            current_path = Path(current)
            relative_depth = len(current_path.relative_to(root).parts) - base_depth
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if name not in _EXCLUDED_DIRECTORIES
                and not (current_path / name).is_symlink()
                and relative_depth < depth_limit
            ]
            for name in dirnames:
                directories.append(str((current_path / name).relative_to(root)))
            for name in sorted(filenames):
                path = current_path / name
                if path.is_symlink() or self._is_sensitive(path, root):
                    continue
                files.append(str(path.relative_to(root)))
                if len(files) + len(directories) >= 500:
                    return {
                        "path": str(arguments["path"] or "."),
                        "files": files,
                        "directories": directories,
                        "truncated": True,
                    }
        return {
            "path": str(arguments["path"] or "."),
            "files": files,
            "directories": directories,
            "truncated": False,
        }

    def _read_file(self, arguments: Mapping[str, Any], root: Path) -> dict[str, Any]:
        path = self._resolve_path(root, str(arguments["path"]))
        if not path.is_file():
            return {"error": "not_a_file", "path": str(arguments["path"])}
        size = path.stat().st_size
        if size > _MAX_READ_BYTES:
            return {"error": "file_too_large", "max_bytes": _MAX_READ_BYTES}
        data = path.read_bytes()
        if b"\x00" in data[:8192]:
            return {"error": "binary_file", "path": str(arguments["path"])}
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "not_utf8", "path": str(arguments["path"])}
        limit = int(arguments["max_chars"])
        return {
            "path": str(arguments["path"]),
            "content": content[:limit],
            "truncated": len(content) > limit,
            "bytes": len(data),
        }

    def _search_files(self, arguments: Mapping[str, Any], root: Path) -> dict[str, Any]:
        query = str(arguments["query"])
        if not query:
            return {"error": "empty_query"}
        search_root = self._resolve_path(root, str(arguments["path"]), sensitive=False)
        if not search_root.is_dir():
            return {"error": "not_a_directory", "path": str(arguments["path"])}
        case_sensitive = bool(arguments["case_sensitive"])
        needle = query if case_sensitive else query.casefold()
        limit = int(arguments["max_results"])
        matches: list[dict[str, Any]] = []
        for current, dirnames, filenames in os.walk(search_root, followlinks=False):
            current_path = Path(current)
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if name not in _EXCLUDED_DIRECTORIES and not (current_path / name).is_symlink()
            ]
            for name in sorted(filenames):
                path = current_path / name
                if path.is_symlink() or self._is_sensitive(path, root):
                    continue
                try:
                    if path.stat().st_size > _MAX_SEARCH_BYTES:
                        continue
                    data = path.read_bytes()
                    if b"\x00" in data[:8192]:
                        continue
                    content = data.decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                for index, line in enumerate(content.splitlines(), start=1):
                    haystack = line if case_sensitive else line.casefold()
                    if needle not in haystack:
                        continue
                    matches.append(
                        {
                            "path": str(path.relative_to(root)),
                            "line": index,
                            "snippet": line[:500],
                        }
                    )
                    if len(matches) >= limit:
                        return {"query": query, "matches": matches, "truncated": True}
        return {"query": query, "matches": matches, "truncated": False}

    def _inspect_repo(self, _arguments: Mapping[str, Any], root: Path) -> dict[str, Any]:
        markers = (
            "Package.swift",
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "Makefile",
            ".gitignore",
        )
        branch = self._run_process(
            ["git", "branch", "--show-current"], root, timeout=10, sandbox=False
        )
        top_level = sorted(
            item.name
            for item in root.iterdir()
            if item.name not in _EXCLUDED_DIRECTORIES and not self._is_sensitive(item, root)
        )[:100]
        return {
            "root": str(root),
            "is_git_repository": (root / ".git").exists(),
            "branch": branch["stdout"].strip() if branch["exit_code"] == 0 else "",
            "project_markers": [name for name in markers if (root / name).exists()],
            "top_level": top_level,
            "git_error": branch["stderr"] if branch["exit_code"] else "",
        }

    def _git_status(self, _arguments: Mapping[str, Any], root: Path) -> dict[str, Any]:
        return self._run_process(
            ["git", "status", "--short", "--branch"], root, timeout=10, sandbox=False
        )

    def _git_diff(self, arguments: Mapping[str, Any], root: Path) -> dict[str, Any]:
        scope = str(arguments["scope"])
        if scope not in {"unstaged", "staged", "both"}:
            return {"error": "invalid_scope"}
        raw_path = str(arguments["path"])
        if raw_path:
            self._resolve_path(root, raw_path, sensitive=False)
        scopes = ("unstaged", "staged") if scope == "both" else (scope,)
        sections = []
        for item in scopes:
            command = ["git", "diff", "--no-ext-diff", "--unified=3"]
            if item == "staged":
                command.insert(2, "--cached")
            if raw_path:
                command.extend(["--", raw_path])
            result = self._run_process(command, root, timeout=20, sandbox=False)
            sections.append({"scope": item, **result})
        return {
            "path": raw_path,
            "scopes": sections,
            "ok": all(item["exit_code"] == 0 for item in sections),
        }

    def _write_file(self, arguments: Mapping[str, Any], root: Path) -> dict[str, Any]:
        path = self._resolve_path(root, str(arguments["path"]))
        content = str(arguments["content"])
        if not str(arguments["path"]).strip():
            return {"error": "empty_path"}
        if len(content) > _MAX_WRITE_CHARACTERS:
            return {"error": "content_too_large", "max_chars": _MAX_WRITE_CHARACTERS}
        _atomic_write(path, content)
        return {"path": str(arguments["path"]), "written": True, "bytes": len(content.encode())}

    def _apply_patch(
        self,
        arguments: Mapping[str, Any],
        root: Path,
        *,
        cancellation_event: threading.Event | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        patch = str(arguments["patch"])
        if not patch.strip():
            return {"error": "empty_patch"}
        if len(patch) > _MAX_PATCH_CHARACTERS:
            return {"error": "patch_too_large", "max_chars": _MAX_PATCH_CHARACTERS}
        descriptor, temporary = tempfile.mkstemp(prefix="mtplx-patch-", suffix=".diff")
        deadline = time.monotonic() + max(1, int(timeout_seconds or 60))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(patch)
                handle.flush()
                os.fsync(handle.fileno())
            check = self._run_process(
                ["git", "apply", "--check", "--whitespace=nowarn", "--", temporary],
                root,
                timeout=max(1, int(deadline - time.monotonic())),
                sandbox=False,
                cancellation_event=cancellation_event,
            )
            if check["exit_code"] != 0:
                return {"validated": False, "applied": False, **check}
            if time.monotonic() >= deadline:
                return {
                    "validated": True,
                    "applied": False,
                    "error": "apply_patch timeout exceeded",
                    "timed_out": True,
                }
            applied = self._run_process(
                ["git", "apply", "--whitespace=nowarn", "--", temporary],
                root,
                timeout=max(1, int(deadline - time.monotonic())),
                sandbox=False,
                cancellation_event=cancellation_event,
            )
            return {
                "validated": True,
                "applied": applied["exit_code"] == 0,
                **applied,
            }
        finally:
            try:
                os.unlink(temporary)
            except OSError:
                pass

    def _run_tests(
        self,
        arguments: Mapping[str, Any],
        root: Path,
        *,
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if not str(arguments["command"]):
            return {"error": "no_test_runner_detected"}
        return self._run_shell(
            arguments,
            root,
            cancellation_event=cancellation_event,
        )

    def _run_command(
        self,
        arguments: Mapping[str, Any],
        root: Path,
        *,
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if not str(arguments["command"]):
            return {"error": "empty_command"}
        return self._run_shell(
            arguments,
            root,
            cancellation_event=cancellation_event,
        )

    def _run_shell(
        self,
        arguments: Mapping[str, Any],
        root: Path,
        *,
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        command = str(arguments["command"])
        if "\x00" in command or len(command) > _MAX_COMMAND_CHARACTERS:
            return {"error": "invalid_command"}
        return self._run_process(
            ["/bin/zsh", "-lc", command],
            root,
            timeout=int(arguments["timeout_seconds"]),
            allow_network=bool(arguments["network"]),
            sandbox=True,
            display_command=command,
            cancellation_event=cancellation_event,
        )

    @staticmethod
    def _detected_test_command(root: Path) -> str | None:
        if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() or (root / "tests").exists():
            return "pytest -q"
        if (root / "Package.swift").exists():
            return "swift test"
        if (root / "package.json").exists():
            return "npm test"
        return None

    def _run_process(
        self,
        command: Sequence[str],
        root: Path,
        *,
        timeout: int,
        allow_network: bool = False,
        sandbox: bool,
        display_command: str | None = None,
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        argv = [str(item) for item in command]
        sandboxed = False
        with tempfile.TemporaryDirectory(prefix="mtplx-agent-") as temporary:
            temp_root = Path(temporary).resolve()
            if sandbox:
                argv, sandboxed = self._sandboxed_argv(
                    argv,
                    root=root,
                    temp_root=temp_root,
                    allow_network=allow_network,
                )
            executable_directory = str(Path(sys.executable).resolve().parent)
            path_entries = dict.fromkeys(
                (
                    executable_directory,
                    "/opt/homebrew/bin",
                    "/usr/local/bin",
                    "/usr/bin",
                    "/bin",
                    "/usr/sbin",
                    "/sbin",
                )
            )
            environment = {
                "PATH": ":".join(path_entries),
                "HOME": str(temp_root),
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
                "PWD": str(root),
                "TMPDIR": str(temp_root),
            }
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            except OSError as exc:
                return {
                    "command": display_command or shlex.join(command),
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": str(exc),
                    "timed_out": False,
                    "sandboxed": sandboxed,
                    "network_allowed": allow_network,
                }
            timed_out = False
            cancelled = False

            def terminate_process() -> tuple[str, str]:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    return process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    return process.communicate()

            deadline = time.monotonic() + max(1, int(timeout))
            while True:
                if cancellation_event is not None and cancellation_event.is_set():
                    cancelled = True
                    stdout, stderr = terminate_process()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    stdout, stderr = terminate_process()
                    break
                try:
                    stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            return {
                "command": display_command or shlex.join(command),
                "exit_code": int(process.returncode or 0),
                "stdout": _redact_output(stdout)[:_MAX_STDOUT_CHARACTERS],
                "stderr": _redact_output(stderr)[:_MAX_STDERR_CHARACTERS],
                "timed_out": timed_out,
                "cancelled": cancelled,
                "sandboxed": sandboxed,
                "network_allowed": allow_network,
            }

    def _sandboxed_argv(
        self,
        command: list[str],
        *,
        root: Path,
        temp_root: Path,
        allow_network: bool,
    ) -> tuple[list[str], bool]:
        executable = Path("/usr/bin/sandbox-exec")
        available = sys.platform == "darwin" and executable.exists()
        if self.sandbox_mode == "off":
            return command, False
        if not available:
            if self.sandbox_mode == "required":
                raise WorkspaceToolError("sandbox-exec is required but unavailable")
            return command, False
        root_literal = _sandbox_quote(str(root))
        temp_literal = _sandbox_quote(str(temp_root))
        network_rule = "(allow network*)" if allow_network else ""
        profile = " ".join(
            item
            for item in (
                "(version 1)",
                "(deny default)",
                "(allow process*)",
                "(allow signal)",
                "(allow sysctl-read)",
                "(allow mach-lookup)",
                "(allow ipc-posix*)",
                "(allow file-read*)",
                f'(allow file-write* (subpath "{root_literal}") (subpath "{temp_literal}"))',
                network_rule,
            )
            if item
        )
        return [str(executable), "-p", profile, *command], True

    @staticmethod
    def _succeeded(result: Mapping[str, Any]) -> bool:
        if result.get("cancelled"):
            return False
        if result.get("error"):
            return False
        if "exit_code" in result and int(result.get("exit_code") or 0) != 0:
            return False
        if result.get("applied") is False or result.get("validated") is False:
            return False
        if result.get("ok") is False:
            return False
        return True


def _looks_destructive(command: str) -> bool:
    lowered = " ".join(command.lower().split())
    patterns = (
        r"(^|[;&|]\s*)rm\s+-[^\n]*r",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+-[^\n]*f",
        r"\bgit\s+checkout\s+--\b",
        r"\bgit\s+push\b",
        r"\bsudo\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _redact_output(value: str) -> str:
    pattern = re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\b"
        r"(\s*[:=]\s*)([^\s,;]+)"
    )
    return pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)


def _sandbox_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "FIRST_PARTY_TOOL_NAMES",
    "FILE_CHANGING_TOOLS",
    "MUTATING_TOOLS",
    "TOOL_POLICY_KEY",
    "ToolPreview",
    "WorkspaceToolError",
    "WorkspaceToolPermissionError",
    "WorkspaceToolService",
    "first_party_tool_definitions",
]
