"""Provider-neutral runtime system status registry.

The registry is an observability boundary. Runtime components may publish
JSON-compatible status without the registry importing or controlling them.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any


_SYSTEM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class RuntimeSystemsRegistry:
    """Thread-safe, bounded collection of runtime status snapshots."""

    def __init__(
        self,
        *,
        max_systems: int = 128,
        max_status_bytes: int = 64 * 1024,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_systems < 1:
            raise ValueError("max_systems must be positive")
        if max_status_bytes < 2:
            raise ValueError("max_status_bytes must be at least 2")
        self._max_systems = int(max_systems)
        self._max_status_bytes = int(max_status_bytes)
        self._clock = clock
        self._lock = threading.Lock()
        self._revision = 0
        self._updated_at_s = float(clock())
        self._systems: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _validate_name(name: str) -> str:
        normalized = str(name).strip()
        if not _SYSTEM_NAME.fullmatch(normalized):
            raise ValueError(
                "system name must be 1 to 128 characters and contain only "
                "letters, digits, dot, colon, underscore, or hyphen"
            )
        return normalized

    def _clone_status(self, status: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(status, Mapping):
            raise TypeError("status must be a mapping")
        try:
            encoded = json.dumps(
                dict(status),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("status must be JSON-compatible") from exc
        if len(encoded) > self._max_status_bytes:
            raise ValueError(
                f"status exceeds max_status_bytes={self._max_status_bytes}"
            )
        return json.loads(encoded)

    def update(self, name: str, status: Mapping[str, Any]) -> int:
        """Replace one system status and return the registry revision."""

        normalized = self._validate_name(name)
        cloned = self._clone_status(status)
        now = float(self._clock())
        with self._lock:
            if (
                normalized not in self._systems
                and len(self._systems) >= self._max_systems
            ):
                raise ValueError(f"registry is limited to {self._max_systems} systems")
            self._revision += 1
            self._systems[normalized] = {
                "revision": self._revision,
                "updated_at_s": now,
                "status": cloned,
            }
            self._updated_at_s = now
            return self._revision

    def remove(self, name: str) -> bool:
        """Remove one system status if present."""

        normalized = self._validate_name(name)
        with self._lock:
            if normalized not in self._systems:
                return False
            del self._systems[normalized]
            self._revision += 1
            self._updated_at_s = float(self._clock())
            return True

    def snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-compatible view of all published statuses."""

        with self._lock:
            revision = self._revision
            updated_at_s = self._updated_at_s
            systems = json.loads(json.dumps(self._systems, separators=(",", ":")))
        return {
            "ts": float(self._clock()),
            "revision": revision,
            "updated_at_s": updated_at_s,
            "system_count": len(systems),
            "systems": systems,
        }


def runtime_systems_snapshot(state: Any) -> dict[str, Any]:
    """Read a registry from a server state without requiring a concrete type."""

    registry = getattr(state, "runtime_systems", None)
    snapshot = getattr(registry, "snapshot", None)
    if callable(snapshot):
        payload = snapshot()
        if isinstance(payload, Mapping):
            return dict(payload)
    now = time.time()
    return {
        "ts": now,
        "revision": 0,
        "updated_at_s": now,
        "system_count": 0,
        "systems": {},
    }


def install_runtime_systems_endpoint(app: Any, state: Any) -> None:
    """Install the read-only runtime systems endpoint on a FastAPI app."""

    @app.get("/v1/mtplx/systems")
    def mtplx_runtime_systems() -> dict[str, Any]:
        return runtime_systems_snapshot(state)


__all__ = [
    "RuntimeSystemsRegistry",
    "install_runtime_systems_endpoint",
    "runtime_systems_snapshot",
]
