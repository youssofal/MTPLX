"""Model Supervisor: a separate process that owns and routes to N engines.

This package is built up wave by wave (see
``docs/features/model-supervisor/COMPONENT_DAG.md``). This module
re-exports the public names from the pieces implemented so far.
"""

from __future__ import annotations

from .budget import Admission, admit, estimate_resident_bytes, total_ram_bytes
from .registry import (
    EngineRecord,
    EngineRegistry,
    EngineSpec,
    EngineState,
)

__all__ = [
    "EngineState",
    "EngineSpec",
    "EngineRecord",
    "EngineRegistry",
    "Admission",
    "admit",
    "estimate_resident_bytes",
    "total_ram_bytes",
]
