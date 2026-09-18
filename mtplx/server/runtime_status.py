"""Read-only serving-status provider for the generic runtime systems registry.

Only bounded aggregate scalars cross this boundary. No model lock, allocator,
cache mutation, configuration changes, prompt content, paths, or client IDs.
"""

from __future__ import annotations

import threading
from typing import Any


def _counter(value: Any) -> int | None:
    # Do not stringify or invoke arbitrary object conversion for telemetry.
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


class ServingStatusProvider:
    """Refresh from the existing server's CPU-side state, not a second owner."""

    def __init__(self, state: Any) -> None:
        self._state = state
        self._lock = threading.Lock()

    def publish(self) -> None:
        # Serialize collection/publication so an older read cannot replace a newer one.
        with self._lock:
            try:
                status = self._read()
            except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
                # Never leave a prior healthy snapshot visible after a failed refresh.
                # Error text may contain paths or credentials, so only a fixed code escapes.
                status = {
                    "available": False,
                    "enabled": False,
                    "wired": True,
                    "phase": "unavailable",
                    "reason": "status_read_failed",
                }
            self._state.runtime_systems.update("serving", status)

    def _read(self) -> dict[str, Any]:
        state = self._state
        runtime = getattr(state, "runtime", None)
        released = getattr(state, "aime_parent_runtime_released", False) is True
        available = runtime is not None and not released
        foreground = getattr(state, "foreground_count", None)
        active = _counter(
            foreground()
            if callable(foreground)
            else getattr(state, "foreground_active", None)
        )
        in_flight = getattr(getattr(state, "dashboard", None), "in_flight", None)
        count = getattr(in_flight, "count", None)
        if callable(count):
            dashboard_active = _counter(count())
            if dashboard_active is not None:
                active = max(active or 0, dashboard_active)
        mode = getattr(getattr(state, "args", None), "generation_mode", None)
        mode = mode if type(mode) is str and mode in {"ar", "mtp"} else None
        return {
            "available": available,
            "enabled": available,
            "wired": True,
            "phase": (
                "unavailable"
                if not available
                else "unknown"
                if active is None
                else "busy"
                if active
                else "idle"
            ),
            "generation_mode": mode,
            "active_requests": active,
            "requests_completed": _counter(getattr(state, "requests_completed", None)),
            "requests_cancelled": _counter(getattr(state, "requests_cancelled", None)),
            "mtp_enabled": getattr(runtime, "mtp_enabled", False) is True
            if available
            else False,
            "sample_scope": "aggregate_scalars_not_a_transactional_snapshot",
        }
