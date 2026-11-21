"""Which models the native Metal engine serves on this Mac.

A backend is listed when its product path installs an engine site: Inkling 2
Mini (``inkling2_loop``), whose decode step (verify panels, the pending anchor
and one-row decode) is a replay site (``mtplx/models/inkling/fast/engine_site.py``).

``status()`` asks the same functions the engine asks before it serves
(``runtime.mode()``, ``runtime.available()``): MLX and the compiled module
load and pass their start check. It loads no model.
"""

from __future__ import annotations

from typing import Any

from . import runtime as _runtime

# Backend id: the models it covers, as the CLI names them.
SERVED_BACKENDS: dict[str, str] = {
    "inkling2_loop": "Inkling 2 Mini",
}


def status() -> dict[str, Any]:
    """``serves`` lists the backends the engine serves on this Mac now;
    empty, with ``reason``, when it serves none."""

    mode = _runtime.mode()
    if mode != "auto":
        reason = "switched off (MTPLX_NATIVE_ENGINE=0)" if mode == "0" else "shadow mode compares, never serves"
        return {"mode": mode, "available": None if mode == "0" else _runtime.available(), "serves": [], "reason": reason}
    available = _runtime.available()
    return {
        "mode": mode,
        "available": available,
        "serves": list(SERVED_BACKENDS) if available else [],
        "reason": "" if available else _runtime.unavailable(),
        "module_version": _runtime.module_version(),
    }
