"""Built-in and user-defined MTPLX agent profiles."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .agent_workspace import WorkspaceConflictError, WorkspaceStoreError, _atomic_write, safe_id
from .workspace_tools import FIRST_PARTY_TOOL_NAMES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


PROFILE_PERMISSION_NAMES = frozenset(
    {
        "all",
        "read",
        "search",
        "write",
        "terminal",
        "browser",
        "network",
        "memory",
        *FIRST_PARTY_TOOL_NAMES,
    }
)


@dataclass(frozen=True)
class AgentProfile:
    id: str
    name: str
    description: str
    permissions: tuple[str, ...]
    instructions: str
    token_budget: int
    context_window: int
    model: str | None = None
    built_in: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": list(self.permissions),
            "instructions": self.instructions,
            "token_budget": self.token_budget,
            "context_window": self.context_window,
            "model": self.model,
            "built_in": self.built_in,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sha256": self.sha256,
        }


_BUILTIN_VALUES: tuple[dict[str, Any], ...] = (
    {
        "id": "planner",
        "name": "Planner",
        "description": "Turns the active goal into an ordered implementation plan.",
        "permissions": ["read", "search"],
        "instructions": "Inspect first. Produce a concrete numbered plan with verification gates.",
        "token_budget": 3000,
        "context_window": 65_536,
    },
    {
        "id": "implementer",
        "name": "Implementer",
        "description": "Applies approved repository changes inside an isolated worktree.",
        "permissions": ["read", "search", "write", "terminal"],
        "instructions": "Use tools to inspect, edit, and verify the assigned change. Leave a reviewable diff.",
        "token_budget": 8000,
        "context_window": 131_072,
    },
    {
        "id": "reviewer",
        "name": "Reviewer",
        "description": "Inspects a diff and returns blocking findings with evidence.",
        "permissions": ["read", "search"],
        "instructions": "Prioritize correctness and security findings. Cite files and exact evidence.",
        "token_budget": 5000,
        "context_window": 131_072,
    },
    {
        "id": "tester",
        "name": "Tester and verifier",
        "description": "Runs verification gates and records reproducible evidence.",
        "permissions": ["read", "search", "terminal"],
        "instructions": "Run the narrowest relevant gates, then broader checks when justified.",
        "token_budget": 5000,
        "context_window": 65_536,
    },
    {
        "id": "research",
        "name": "Researcher",
        "description": "Investigates repository and approved external reference context.",
        "permissions": ["read", "search", "browser", "network"],
        "instructions": "Distinguish source evidence from inference and preserve citations.",
        "token_budget": 5000,
        "context_window": 131_072,
    },
    {
        "id": "memory_curator",
        "name": "Memory curator",
        "description": "Extracts durable lessons and proposes scoped memory updates.",
        "permissions": ["read", "search", "memory"],
        "instructions": "Capture only durable, scoped facts with provenance and conflict awareness.",
        "token_budget": 3000,
        "context_window": 65_536,
    },
    {
        "id": "user_profile",
        "name": "User profile",
        "description": "Maintains explicit workflow preferences and project conventions.",
        "permissions": ["read", "search", "memory"],
        "instructions": "Preserve explicit preferences without inferring sensitive traits.",
        "token_budget": 3000,
        "context_window": 65_536,
    },
)


def _profile_hash(value: Mapping[str, Any]) -> str:
    canonical = {
        key: value.get(key)
        for key in (
            "id",
            "name",
            "description",
            "permissions",
            "instructions",
            "token_budget",
            "context_window",
            "model",
            "built_in",
        )
    }
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_profile(value: Mapping[str, Any], *, built_in: bool | None = None) -> AgentProfile:
    permissions = tuple(str(item) for item in (value.get("permissions") or []))
    profile = AgentProfile(
        id=safe_id(str(value.get("id") or "profile"), fallback="profile"),
        name=str(value.get("name") or value.get("id") or "Agent profile"),
        description=str(value.get("description") or ""),
        permissions=permissions,
        instructions=str(value.get("instructions") or ""),
        token_budget=max(256, min(int(value.get("token_budget") or 2400), 16_384)),
        context_window=max(1024, min(int(value.get("context_window") or 65_536), 1_048_576)),
        model=str(value["model"]) if value.get("model") else None,
        built_in=bool(value.get("built_in")) if built_in is None else built_in,
        created_at=str(value["created_at"]) if value.get("created_at") else None,
        updated_at=str(value["updated_at"]) if value.get("updated_at") else None,
    )
    return AgentProfile(**{**profile.to_dict(), "permissions": profile.permissions, "sha256": _profile_hash(profile.to_dict())})


BUILTIN_AGENT_PROFILES: tuple[AgentProfile, ...] = tuple(
    _decode_profile({**value, "built_in": True}, built_in=True) for value in _BUILTIN_VALUES
)


class AgentProfileStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve() / "profiles"
        self._lock = threading.RLock()

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, profile_id: str) -> Path:
        return self.root / f"{safe_id(profile_id, fallback='profile')}.json"

    def list(self) -> list[AgentProfile]:
        with self._lock:
            self.ensure_layout()
            custom: list[AgentProfile] = []
            for path in sorted(self.root.glob("*.json")):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, Mapping):
                        custom.append(_decode_profile(value, built_in=False))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return [*BUILTIN_AGENT_PROFILES, *sorted(custom, key=lambda item: item.id)]

    def get(self, profile_id: str) -> AgentProfile:
        clean_id = safe_id(profile_id, fallback="profile")
        builtin = next((item for item in BUILTIN_AGENT_PROFILES if item.id == clean_id), None)
        if builtin is not None:
            return builtin
        try:
            value = json.loads(self._path(clean_id).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkspaceStoreError(f"agent profile not found: {clean_id}") from exc
        if not isinstance(value, Mapping):
            raise WorkspaceStoreError(f"invalid agent profile: {clean_id}")
        return _decode_profile(value, built_in=False)

    def create(
        self,
        profile_id: str,
        *,
        name: str,
        description: str = "",
        permissions: Iterable[str] = ("read", "search"),
        instructions: str = "",
        token_budget: int = 2400,
        context_window: int = 65_536,
        model: str | None = None,
    ) -> AgentProfile:
        identifier = safe_id(profile_id, fallback="profile")
        if any(item.id == identifier for item in BUILTIN_AGENT_PROFILES):
            raise WorkspaceConflictError(f"built-in agent profile cannot be replaced: {identifier}")
        normalized_permissions = self._validate_permissions(permissions)
        if not str(name).strip():
            raise WorkspaceStoreError("agent profile name is required")
        now = utc_now()
        profile = _decode_profile(
            {
                "id": identifier,
                "name": str(name).strip(),
                "description": str(description or ""),
                "permissions": list(normalized_permissions),
                "instructions": str(instructions or ""),
                "token_budget": token_budget,
                "context_window": context_window,
                "model": model,
                "built_in": False,
                "created_at": now,
                "updated_at": now,
            },
            built_in=False,
        )
        with self._lock:
            self.ensure_layout()
            path = self._path(identifier)
            if path.exists():
                raise WorkspaceConflictError(f"agent profile already exists: {identifier}")
            _atomic_write(path, json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n")
        return profile

    def update(self, profile_id: str, **changes: Any) -> AgentProfile:
        current = self.get(profile_id)
        if current.built_in:
            raise WorkspaceConflictError(f"built-in agent profile cannot be edited: {current.id}")
        allowed = {
            "name",
            "description",
            "permissions",
            "instructions",
            "token_budget",
            "context_window",
            "model",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise WorkspaceStoreError(f"unknown agent profile fields: {', '.join(sorted(unknown))}")
        value = current.to_dict()
        value.update({key: item for key, item in changes.items() if item is not None})
        if "permissions" in changes:
            value["permissions"] = list(self._validate_permissions(changes["permissions"]))
        value["updated_at"] = utc_now()
        updated = _decode_profile(value, built_in=False)
        with self._lock:
            _atomic_write(
                self._path(updated.id),
                json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
            )
        return updated

    @staticmethod
    def _validate_permissions(permissions: Iterable[str]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(str(item).strip() for item in permissions if str(item).strip()))
        unknown = sorted(set(normalized) - PROFILE_PERMISSION_NAMES)
        if unknown:
            raise WorkspaceStoreError(f"unknown agent permissions: {', '.join(unknown)}")
        return normalized


__all__ = [
    "AgentProfile",
    "AgentProfileStore",
    "BUILTIN_AGENT_PROFILES",
    "PROFILE_PERMISSION_NAMES",
]
