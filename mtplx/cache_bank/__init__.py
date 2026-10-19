"""Persistent SessionBank cold-tier primitives.

The cache-bank package deliberately sits below the serving layer and above raw
disk I/O. It serializes only committed SessionBank snapshots into immutable
bytes, then lets a single background writer persist them. No live MLX arrays
are ever handed to the writer thread.

The tier's names are resolved lazily (PEP 562): ``cold_tier`` imports the
tensor codec, which imports ``mlx.core`` at module scope, while
``reconcile`` (the manifest-vs-disk store walk behind ``mtplx gc`` and the
tier's own orphan cleanup) is pure SQLite + filesystem and must stay
importable on a machine whose MLX install is missing or broken.
"""

from __future__ import annotations

from typing import Any

from .reconcile import DEFAULT_COLD_TIER_DIR

_COLD_TIER_EXPORTS = frozenset(
    {
        "COLD_TIER_FORMAT_VERSION",
        "DEFAULT_COLD_TIER_MAX_BYTES",
        "DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS",
        "ServerIdentity",
        "SessionBankColdTier",
        "parse_size_bytes",
    }
)

__all__ = [
    "COLD_TIER_FORMAT_VERSION",
    "DEFAULT_COLD_TIER_DIR",
    "DEFAULT_COLD_TIER_MAX_BYTES",
    "DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS",
    "ServerIdentity",
    "SessionBankColdTier",
    "parse_size_bytes",
]


def __getattr__(name: str) -> Any:
    if name in _COLD_TIER_EXPORTS:
        from . import cold_tier

        return getattr(cold_tier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
