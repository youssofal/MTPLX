"""The native executor: records the Metal launches MLX makes for one
evaluation and replays them from compiled code on a command queue of our own.

Importing this package loads nothing compiled.  ``runtime.available()`` loads
the module on first use and never raises.
"""

from __future__ import annotations

from .ir import ArrayIR, BufferIR, ProgramIR, StepIR
from .runtime import (
    BuildRefused,
    Engine,
    EngineError,
    ReplayLost,
    EngineUnavailable,
    RunRefused,
    available,
    dump_dir,
    enabled,
    engine,
    mode,
    module_version,
    unavailable,
)

__all__ = [
    "ArrayIR",
    "BufferIR",
    "BuildRefused",
    "Engine",
    "EngineError",
    "ReplayLost",
    "EngineUnavailable",
    "RunRefused",
    "ProgramIR",
    "StepIR",
    "available",
    "dump_dir",
    "enabled",
    "engine",
    "mode",
    "module_version",
    "unavailable",
]
