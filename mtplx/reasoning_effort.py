"""Canonical MTPLX reasoning-effort vocabulary shared by CLI and server code.

Each model family declares the subset it supports through its
``ReasoningCodec.effort_levels``, and that subset is exactly what the app's
effort picker renders. Every *writing* surface — serve/CLI argparse,
``mtplx config set``, and the live-settings POST — must therefore accept this
whole vocabulary and leave the per-family narrowing to the one place that
knows which model is loaded (``_reasoning_effort_for_state``).

Surfaces that inline their own list go stale the moment a family ships a new
level. That is how 2.7.0 shipped with Qwen 3.8's ``xhigh`` bouncing back to
``medium``: the request path knew the level, the live-settings coercer did
not, and because that POST is all-or-nothing the app's whole settings payload
was rejected.
"""

from __future__ import annotations

from typing import Any


# Ordered low to high.
REASONING_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# "auto" is not a level; it means "use the loaded family's default_effort".
REASONING_EFFORT_AUTO = "auto"
REASONING_EFFORT_CHOICES = (REASONING_EFFORT_AUTO, *REASONING_EFFORT_LEVELS)


def normalize_reasoning_effort(
    value: Any, *, default: str = REASONING_EFFORT_AUTO
) -> str:
    effort = str(value or default).strip().lower()
    if effort not in REASONING_EFFORT_CHOICES:
        raise ValueError(
            "reasoning_effort must be one of: " + ", ".join(REASONING_EFFORT_CHOICES)
        )
    return effort
