"""Memory budget and admission math for supervised engines.

Pure arithmetic and read-only filesystem probing: no spawning, no
unloading. ``admit`` tells the caller whether a spec fits and, if not,
what unloading would free; the caller (the proxy/service layer) is the one
that actually evicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mtplx.memory_plan import detect_total_ram_bytes
from mtplx.model_catalog import catalog_model_matching

from .registry import EngineRegistry, EngineSpec, EngineState

GIB = 1024**3

# D5: the n-gram sidecar streams from SSD at serve time (see docs/server.md)
# and is never fully resident, so it is excluded from the weights-size
# fallback below.
_NGRAM_SIDECAR_NAME = "ngram-table.safetensors"

# D5: honest estimate when no catalog figure exists, calibrated against the
# gap between "bytes on disk" and observed peak resident memory.
_WEIGHTS_SIZE_FALLBACK_FACTOR = 1.15

# Used only if every detection path (ctypes sysctl, psutil, os.sysconf)
# fails; matches the floor `usable_engine_bytes` treats as a sane minimum.
_TOTAL_RAM_FALLBACK_BYTES = 8 * GIB


def total_ram_bytes() -> int:
    """Physical RAM on this machine, always a positive int.

    Reuses ``mtplx.memory_plan.detect_total_ram_bytes`` (PATH-immune
    ``sysctlbyname`` on macOS, ``os.sysconf`` elsewhere) rather than
    duplicating detection. Falls back to ``psutil`` and then to a fixed
    floor if every other path fails, so callers never have to handle
    ``None``.
    """
    detected = detect_total_ram_bytes()
    if detected:
        return int(detected)
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass
    return _TOTAL_RAM_FALLBACK_BYTES


def _pack_model_id(path: Path) -> str | None:
    """Best-effort model id from a pack's ``mtplx_runtime.json``."""
    runtime_file = path / "mtplx_runtime.json"
    try:
        raw = runtime_file.read_text()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("public_model_id", "served_model_id", "model_id"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def estimate_resident_bytes(path: Path) -> int:
    """Estimate how much RAM a pack will hold resident once loaded.

    D5: use the catalog's measured ``peak_memory_gib`` when the pack
    matches a known official model (by directory name, or by the model id
    recorded in its ``mtplx_runtime.json``). Otherwise sum the on-disk size
    of every ``*.safetensors`` file under the pack -- recursively, since a
    sharded pack can nest weights under a subdirectory (M3) -- excluding
    any file named ``ngram-table.safetensors`` at any depth (that sidecar
    streams from SSD rather than residing in RAM; see docs/server.md), and
    scale by 1.15 to account for allocator overhead beyond the raw weight
    bytes.
    """
    path = Path(path)
    model = catalog_model_matching(path.name)
    if model is None:
        pack_id = _pack_model_id(path)
        if pack_id:
            model = catalog_model_matching(pack_id)
    if model is not None and model.peak_memory_gib:
        return int(model.peak_memory_gib * GIB)

    total = 0
    for file in path.rglob("*.safetensors"):
        if file.name == _NGRAM_SIDECAR_NAME:
            continue
        try:
            total += file.stat().st_size
        except OSError:
            continue
    return int(total * _WEIGHTS_SIZE_FALLBACK_FACTOR)


@dataclass
class Admission:
    """Verdict from ``admit``: whether a spec fits, and what to free."""

    ok: bool
    needed_bytes: int
    available_bytes: int
    would_free: list[str]
    reason: str


def _gib(num_bytes: int) -> str:
    return f"{num_bytes / GIB:.1f}"


_RESIDENT_STATES = (EngineState.READY, EngineState.LOADING, EngineState.DRAINING)


def admit(
    registry: EngineRegistry,
    spec: EngineSpec,
    budget_bytes: int,
    *,
    evict_to_fit: bool,
) -> Admission:
    """Decide whether ``spec`` fits in ``budget_bytes`` right now.

    ``available_bytes`` is the budget minus everything already resident or
    on its way there (READY, LOADING, DRAINING). When the spec does not
    fit and ``evict_to_fit`` is set, unpinned READY engines are considered
    for eviction oldest-first until it would fit; ``would_free`` then lists
    exactly the ids that must be unloaded, in that order. This function
    unloads nothing itself; the caller acts on ``would_free``.
    """
    resident_bytes = sum(
        record.spec.resident_bytes
        for record in registry.records()
        if record.state in _RESIDENT_STATES
    )
    available = budget_bytes - resident_bytes
    needed = spec.resident_bytes

    if needed <= available:
        return Admission(True, needed, available, [], "")

    candidates = registry.lru_unpinned(exclude={spec.model_id})

    if not evict_to_fit:
        would_free_total = sum(c.spec.resident_bytes for c in candidates)
        reason = (
            f"{spec.model_id} needs {_gib(needed)} GiB resident but "
            f"{_gib(available)} GiB is available"
        )
        if candidates:
            names = ", ".join(c.spec.model_id for c in candidates)
            reason += f"; unloading {names} would free {_gib(would_free_total)} GiB"
        return Admission(
            False, needed, available, [c.spec.model_id for c in candidates], reason
        )

    would_free: list[str] = []
    freed = 0
    for candidate in candidates:
        would_free.append(candidate.spec.model_id)
        freed += candidate.spec.resident_bytes
        if available + freed >= needed:
            return Admission(True, needed, available, would_free, "")

    reason = (
        f"{spec.model_id} needs {_gib(needed)} GiB resident but "
        f"{_gib(available)} GiB is available"
    )
    if would_free:
        reason += (
            f"; unloading {', '.join(would_free)} would free {_gib(freed)} GiB"
        )
    return Admission(False, needed, available, would_free, reason)
