"""Opt-in expert-routing locality instrumentation for Apple Silicon.

This module is diagnostic only. It never changes router outputs or expert
placement. When ``MTPLX_EXPERT_LOCALITY=1`` is unset, the public record helper
returns before touching an MLX array, preserving the normal lazy execution
path. Enabled runs intentionally materialize sampled router indices so the
result is measurement evidence, not a production cache policy.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


_LOCALITY_LANE: ContextVar[str] = ContextVar(
    "mtplx_expert_locality_lane",
    default="model_forward",
)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"", "0", "false", "off", "no"}


def expert_locality_enabled() -> bool:
    return _env_truthy("MTPLX_EXPERT_LOCALITY", False)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        return max(minimum, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(minimum, int(default))


def _cache_capacities() -> tuple[int, ...]:
    raw = str(os.environ.get("MTPLX_EXPERT_LOCALITY_CACHE_SIZES") or "16,32,64,96,128")
    values: set[int] = set()
    for item in raw.split(","):
        try:
            values.add(max(1, int(item.strip())))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(values or {16, 32, 64, 96, 128}))


def _flatten_python(value: Any) -> Iterator[int]:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield int(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_python(item)
        return
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            yield from _flatten_python(tolist())
        except Exception:
            return


def _rows_python(value: Any) -> list[tuple[int, ...]]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:
            return []
    if isinstance(value, int):
        return [(int(value),)]
    if not isinstance(value, (list, tuple)):
        return []
    if not value:
        return []
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return [tuple(int(item) for item in value)]
    rows: list[tuple[int, ...]] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            flattened = tuple(_flatten_python(item))
            if flattened:
                rows.append(flattened)
        elif isinstance(item, int) and not isinstance(item, bool):
            rows.append((int(item),))
    return rows


def _reuse_bucket(distance: int) -> str:
    value = max(0, int(distance))
    if value <= 1:
        return str(value)
    lower = 2
    upper = 3
    while value > upper:
        lower = upper + 1
        upper = upper * 2 + 1
    return f"{lower}-{upper}"


@dataclass
class _LRUSimulation:
    capacity: int
    entries: OrderedDict[int, None] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def observe(self, expert: int) -> None:
        key = int(expert)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return
        self.misses += 1
        self.entries[key] = None
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def to_dict(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "capacity": int(self.capacity),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "hit_rate": (self.hits / total) if total else 0.0,
            "resident": len(self.entries),
        }


@dataclass
class _LayerLaneStats:
    layer_id: str
    lane: str
    cache_capacities: tuple[int, ...]
    events: int = 0
    rows: int = 0
    assignments: int = 0
    invalid_assignments: int = 0
    expert_counts: Counter[int] = field(default_factory=Counter)
    last_seen_event: dict[int, int] = field(default_factory=dict)
    reuse_distance: Counter[str] = field(default_factory=Counter)
    consecutive_overlap_sum: float = 0.0
    consecutive_overlap_samples: int = 0
    previous_experts: set[int] = field(default_factory=set)
    lru: dict[int, _LRUSimulation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lru:
            self.lru = {
                capacity: _LRUSimulation(capacity) for capacity in self.cache_capacities
            }

    def observe_rows(
        self, rows: Sequence[Sequence[int]], num_experts: int | None
    ) -> None:
        self.events += 1
        event_index = self.events
        for row in rows:
            row_set: set[int] = set()
            self.rows += 1
            for raw in row:
                expert = int(raw)
                if expert < 0 or (
                    num_experts is not None and expert >= int(num_experts)
                ):
                    self.invalid_assignments += 1
                    continue
                self.assignments += 1
                self.expert_counts[expert] += 1
                row_set.add(expert)
                previous = self.last_seen_event.get(expert)
                if previous is not None:
                    self.reuse_distance[_reuse_bucket(event_index - previous)] += 1
                self.last_seen_event[expert] = event_index
                for simulation in self.lru.values():
                    simulation.observe(expert)
            if self.previous_experts or row_set:
                union = self.previous_experts | row_set
                overlap = (
                    len(self.previous_experts & row_set) / len(union) if union else 1.0
                )
                self.consecutive_overlap_sum += overlap
                self.consecutive_overlap_samples += 1
            self.previous_experts = row_set

    def _coverage_count(self, fraction: float) -> int:
        if self.assignments <= 0:
            return 0
        target = self.assignments * max(0.0, min(1.0, fraction))
        cumulative = 0
        for index, count in enumerate(
            sorted(self.expert_counts.values(), reverse=True),
            start=1,
        ):
            cumulative += count
            if cumulative >= target:
                return index
        return len(self.expert_counts)

    def to_dict(self) -> dict[str, Any]:
        top = self.expert_counts.most_common(16)
        return {
            "layer_id": self.layer_id,
            "lane": self.lane,
            "events": int(self.events),
            "rows": int(self.rows),
            "assignments": int(self.assignments),
            "invalid_assignments": int(self.invalid_assignments),
            "unique_experts": len(self.expert_counts),
            "consecutive_jaccard": (
                self.consecutive_overlap_sum / self.consecutive_overlap_samples
                if self.consecutive_overlap_samples
                else 0.0
            ),
            "working_set_50": self._coverage_count(0.50),
            "working_set_90": self._coverage_count(0.90),
            "working_set_99": self._coverage_count(0.99),
            "top_experts": [[int(expert), int(count)] for expert, count in top],
            "reuse_distance_events": dict(sorted(self.reuse_distance.items())),
            "lru_simulation": {
                str(capacity): simulation.to_dict()
                for capacity, simulation in sorted(self.lru.items())
            },
        }


class ExpertLocalityTracker:
    """Bounded per-layer/lane routing statistics and LRU simulations."""

    def __init__(
        self,
        *,
        max_events: int = 4096,
        sample_every: int = 1,
        cache_capacities: Sequence[int] | None = None,
    ) -> None:
        self.max_events = max(1, int(max_events))
        self.sample_every = max(1, int(sample_every))
        self.cache_capacities = tuple(
            sorted(
                {
                    max(1, int(value))
                    for value in (cache_capacities or _cache_capacities())
                }
            )
        )
        self._lock = threading.Lock()
        self._stats: dict[tuple[str, str], _LayerLaneStats] = {}
        self._calls = 0
        self._accepted_calls = 0
        self._dropped_calls = 0
        self._started_s = time.monotonic()

    def record(
        self,
        indices: Any,
        *,
        layer_id: int | str,
        lane: str | None = None,
        num_experts: int | None = None,
    ) -> bool:
        with self._lock:
            self._calls += 1
            call_index = self._calls
            if call_index % self.sample_every != 0:
                return False
            if self._accepted_calls >= self.max_events:
                self._dropped_calls += 1
                return False
        rows = _rows_python(indices)
        if not rows:
            return False
        key = (str(layer_id), str(lane or _LOCALITY_LANE.get()))
        with self._lock:
            stats = self._stats.get(key)
            if stats is None:
                stats = _LayerLaneStats(
                    layer_id=key[0],
                    lane=key[1],
                    cache_capacities=self.cache_capacities,
                )
                self._stats[key] = stats
            stats.observe_rows(rows, num_experts)
            self._accepted_calls += 1
        return True

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()
            self._calls = 0
            self._accepted_calls = 0
            self._dropped_calls = 0
            self._started_s = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = [
                stats.to_dict()
                for _, stats in sorted(self._stats.items(), key=lambda item: item[0])
            ]
            return {
                "enabled": True,
                "calls": int(self._calls),
                "accepted_calls": int(self._accepted_calls),
                "dropped_calls": int(self._dropped_calls),
                "max_events": int(self.max_events),
                "sample_every": int(self.sample_every),
                "cache_capacities": list(self.cache_capacities),
                "elapsed_s": max(0.0, time.monotonic() - self._started_s),
                "layers": rows,
            }

    def recommended_capacity(
        self,
        *,
        minimum_hit_rate: float = 0.60,
        lane: str | None = None,
    ) -> int | None:
        target = max(0.0, min(1.0, float(minimum_hit_rate)))
        with self._lock:
            candidates = [
                stats
                for stats in self._stats.values()
                if lane is None or stats.lane == lane
            ]
            if not candidates:
                return None
            for capacity in self.cache_capacities:
                hits = sum(stats.lru[capacity].hits for stats in candidates)
                misses = sum(stats.lru[capacity].misses for stats in candidates)
                if hits + misses and hits / (hits + misses) >= target:
                    return capacity
        return None


_GLOBAL_TRACKER: ExpertLocalityTracker | None = None
_GLOBAL_LOCK = threading.Lock()


def get_expert_locality_tracker() -> ExpertLocalityTracker:
    global _GLOBAL_TRACKER
    with _GLOBAL_LOCK:
        if _GLOBAL_TRACKER is None:
            _GLOBAL_TRACKER = ExpertLocalityTracker(
                max_events=_env_int("MTPLX_EXPERT_LOCALITY_MAX_EVENTS", 4096),
                sample_every=_env_int("MTPLX_EXPERT_LOCALITY_SAMPLE_EVERY", 16),
            )
        return _GLOBAL_TRACKER


def reset_expert_locality_tracker() -> None:
    global _GLOBAL_TRACKER
    with _GLOBAL_LOCK:
        _GLOBAL_TRACKER = None


def record_expert_routes(
    indices: Any,
    *,
    layer_id: int | str,
    lane: str | None = None,
    num_experts: int | None = None,
) -> bool:
    """Record one router output; return before array materialization when off."""

    if not expert_locality_enabled():
        return False
    return get_expert_locality_tracker().record(
        indices,
        layer_id=layer_id,
        lane=lane,
        num_experts=num_experts,
    )


@contextmanager
def expert_locality_lane(lane: str) -> Iterator[None]:
    token = _LOCALITY_LANE.set(str(lane))
    try:
        yield
    finally:
        _LOCALITY_LANE.reset(token)


def expert_locality_metrics() -> dict[str, Any]:
    if not expert_locality_enabled():
        return {"enabled": False}
    tracker = get_expert_locality_tracker()
    snapshot = tracker.snapshot()
    snapshot["recommended_capacity_60"] = tracker.recommended_capacity(
        minimum_hit_rate=0.60
    )
    snapshot["recommended_capacity_80"] = tracker.recommended_capacity(
        minimum_hit_rate=0.80
    )
    return snapshot


_INSTRUMENTED_CLASS_CACHE: dict[type, type] = {}
_INSTRUMENTED_CLASS_LOCK = threading.Lock()


def _current_runtime_lane() -> str:
    try:
        from .attention_context import (
            current_attention_phase,
            current_model_forward_kind,
        )

        phase = current_attention_phase()
        if phase == "prefill":
            return "prefill"
        if phase == "ar_decode":
            return "decode"
        if phase == "decode_verify":
            kind = current_model_forward_kind()
            return "mtp_repair" if kind == "repair" else "mtp_verify"
        if phase == "postcommit":
            return "postcommit"
        return phase or "model_forward"
    except Exception:
        return str(_LOCALITY_LANE.get())


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _shape_axis(value: Any, axis: int = 0) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return _positive_int(shape[axis])
    except (IndexError, TypeError):
        return None


def _infer_num_experts(parent: Any, switch_mlp: Any) -> int | None:
    for owner in (parent, switch_mlp, getattr(parent, "gate", None)):
        if owner is None:
            continue
        for name in (
            "num_experts",
            "n_experts",
            "n_routed_experts",
            "num_local_experts",
        ):
            value = _positive_int(getattr(owner, name, None))
            if value is not None:
                return value
    for name in ("gate_proj", "up_proj", "down_proj", "weight"):
        projection = getattr(switch_mlp, name, None)
        value = _shape_axis(getattr(projection, "weight", projection), 0)
        if value is not None:
            return value
    return None


def _instrumented_switch_class(base_class: type) -> type:
    with _INSTRUMENTED_CLASS_LOCK:
        cached = _INSTRUMENTED_CLASS_CACHE.get(base_class)
        if cached is not None:
            return cached

        def __call__(self: Any, *args: Any, **kwargs: Any) -> Any:
            indices = kwargs.get("indices")
            if indices is None and len(args) >= 2:
                indices = args[1]
            if indices is not None:
                record_expert_routes(
                    indices,
                    layer_id=getattr(
                        self, "_mtplx_expert_locality_layer_id", "unknown"
                    ),
                    lane=_current_runtime_lane(),
                    num_experts=getattr(
                        self, "_mtplx_expert_locality_num_experts", None
                    ),
                )
            return base_class.__call__(self, *args, **kwargs)

        instrumented = type(
            f"MTPLXExpertLocality{base_class.__name__}",
            (base_class,),
            {
                "__call__": __call__,
                "_mtplx_expert_locality_class": True,
            },
        )
        instrumented.__module__ = base_class.__module__
        _INSTRUMENTED_CLASS_CACHE[base_class] = instrumented
        return instrumented


def _set_instrumentation_attribute(target: Any, name: str, value: Any) -> None:
    try:
        object.__setattr__(target, name, value)
    except Exception:
        setattr(target, name, value)


def install_expert_locality_instrumentation(model_or_runtime: Any) -> dict[str, Any]:
    """Install read-only router-index taps on sparse MoE switch modules.

    Installation is startup-only and opt-in. The wrapped switch receives the
    exact router-selected indices already produced by the model and records a
    sampled copy before delegating byte-for-byte to the original implementation.
    """

    enabled = expert_locality_enabled()
    report: dict[str, Any] = {
        "enabled": enabled,
        "installed": False,
        "instrumented_modules": 0,
        "candidate_modules": 0,
        "modules": [],
        "skipped": [],
    }
    if not enabled:
        report["reason"] = "env_disabled"
        return report

    root = getattr(model_or_runtime, "model", model_or_runtime)
    named_modules = getattr(root, "named_modules", None)
    if not callable(named_modules):
        report["reason"] = "model_has_no_named_modules"
        return report

    try:
        modules = list(named_modules())
    except Exception as exc:
        report["reason"] = "named_modules_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    for raw_path, parent in modules:
        switch_mlp = getattr(parent, "switch_mlp", None)
        if switch_mlp is None:
            continue
        report["candidate_modules"] += 1
        path = str(raw_path or "root")
        if bool(getattr(switch_mlp, "_mtplx_expert_locality_instrumented", False)):
            report["modules"].append({"path": path, "status": "already_instrumented"})
            continue
        try:
            original_class = switch_mlp.__class__
            switch_mlp.__class__ = _instrumented_switch_class(original_class)
            num_experts = _infer_num_experts(parent, switch_mlp)
            _set_instrumentation_attribute(
                switch_mlp, "_mtplx_expert_locality_layer_id", path
            )
            _set_instrumentation_attribute(
                switch_mlp, "_mtplx_expert_locality_num_experts", num_experts
            )
            _set_instrumentation_attribute(
                switch_mlp, "_mtplx_expert_locality_instrumented", True
            )
        except Exception as exc:
            report["skipped"].append(
                {
                    "path": path,
                    "reason": "instrumentation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        report["instrumented_modules"] += 1
        report["modules"].append(
            {
                "path": path,
                "status": "instrumented",
                "num_experts": num_experts,
                "switch_class": original_class.__name__,
            }
        )

    report["installed"] = report["instrumented_modules"] > 0
    if not report["installed"]:
        report["reason"] = (
            "no_switch_mlp_modules"
            if report["candidate_modules"] == 0
            else "all_candidates_skipped"
        )
    return report


__all__ = [
    "ExpertLocalityTracker",
    "expert_locality_enabled",
    "expert_locality_lane",
    "expert_locality_metrics",
    "get_expert_locality_tracker",
    "install_expert_locality_instrumentation",
    "record_expert_routes",
    "reset_expert_locality_tracker",
]
