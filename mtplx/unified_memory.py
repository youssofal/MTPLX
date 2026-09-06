"""Shared Apple-unified-memory budgeting for native MTPLX systems.

The coordinator owns policy, not memory.  It computes one bounded plan for
SessionBank, expert warm-set state, and protected KV headroom, then applies
mutable budgets through explicit consumers at a proven runtime safe point.
All successful mutations are rolled back if a later consumer fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class UnifiedMemoryConfig:
    enabled: bool = False
    reserve_bytes: int = 4 * GIB
    minimum_available_bytes: int = 512 * MIB
    target_utilization: float = 0.88
    warning_utilization: float = 0.92
    critical_utilization: float = 0.96
    hysteresis_ratio: float = 0.04
    minimum_apply_interval_s: float = 1.0
    session_bank_weight: float = 0.58
    expert_weight: float = 0.27
    kv_headroom_weight: float = 0.15
    minimum_session_bank_bytes: int = 256 * MIB
    minimum_expert_bytes: int = 0
    minimum_kv_headroom_bytes: int = 256 * MIB
    maximum_session_bank_bytes: int | None = None
    maximum_expert_bytes: int | None = None
    maximum_kv_headroom_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.reserve_bytes < 0 or self.minimum_available_bytes < 0:
            raise ValueError("memory reserves must be non-negative")
        if not 0 < self.target_utilization < 1:
            raise ValueError("target_utilization must be in (0, 1)")
        if not self.target_utilization <= self.warning_utilization <= 1:
            raise ValueError("warning_utilization must be >= target_utilization")
        if not self.warning_utilization <= self.critical_utilization <= 1:
            raise ValueError("critical_utilization must be >= warning_utilization")
        if not 0 <= self.hysteresis_ratio < 1:
            raise ValueError("hysteresis_ratio must be in [0, 1)")
        if self.minimum_apply_interval_s < 0:
            raise ValueError("minimum_apply_interval_s must be non-negative")
        weights = (
            self.session_bank_weight,
            self.expert_weight,
            self.kv_headroom_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("partition weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one partition weight must be positive")
        minima = (
            self.minimum_session_bank_bytes,
            self.minimum_expert_bytes,
            self.minimum_kv_headroom_bytes,
        )
        if any(value < 0 for value in minima):
            raise ValueError("partition minima must be non-negative")


@dataclass(frozen=True)
class UnifiedMemorySample:
    total_bytes: int
    process_bytes: int
    model_bytes: int = 0
    session_bank_bytes: int = 0
    expert_bytes: int = 0
    kv_bytes: int = 0
    wired_bytes: int = 0
    compressed_bytes: int = 0
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.total_bytes,
            self.process_bytes,
            self.model_bytes,
            self.session_bank_bytes,
            self.expert_bytes,
            self.kv_bytes,
            self.wired_bytes,
            self.compressed_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("memory values must be non-negative")
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be positive")

    @property
    def utilization(self) -> float:
        return min(1.0, self.process_bytes / self.total_bytes)


@dataclass(frozen=True)
class UnifiedMemoryPlan:
    plan_id: str
    generated_at_s: float
    pressure: str
    total_bytes: int
    managed_budget_bytes: int
    session_bank_budget_bytes: int
    expert_budget_bytes: int
    kv_headroom_bytes: int
    reserve_bytes: int
    eligible: bool
    reason: str
    active_partitions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "generated_at_s": self.generated_at_s,
            "pressure": self.pressure,
            "total_bytes": self.total_bytes,
            "managed_budget_bytes": self.managed_budget_bytes,
            "session_bank_budget_bytes": self.session_bank_budget_bytes,
            "expert_budget_bytes": self.expert_budget_bytes,
            "kv_headroom_bytes": self.kv_headroom_bytes,
            "reserve_bytes": self.reserve_bytes,
            "eligible": self.eligible,
            "reason": self.reason,
            "active_partitions": list(self.active_partitions),
        }


@dataclass(frozen=True)
class BudgetMutation:
    consumer: str
    previous_bytes: int
    requested_bytes: int
    applied_bytes: int
    applied: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer,
            "previous_bytes": self.previous_bytes,
            "requested_bytes": self.requested_bytes,
            "applied_bytes": self.applied_bytes,
            "applied": self.applied,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class UnifiedMemoryReceipt:
    plan_id: str
    applied: bool
    rolled_back: bool
    mutations: tuple[BudgetMutation, ...]
    reason: str
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "applied": self.applied,
            "rolled_back": self.rolled_back,
            "mutations": [item.to_dict() for item in self.mutations],
            "reason": self.reason,
            "duration_ms": self.duration_ms,
        }


@runtime_checkable
class BudgetConsumer(Protocol):
    name: str

    def current_budget_bytes(self) -> int: ...

    def apply_budget_bytes(self, value: int, *, reason: str) -> int: ...


class SessionBankBudgetConsumer:
    name = "session_bank"

    def __init__(self, bank: Any) -> None:
        self.bank = bank

    def current_budget_bytes(self) -> int:
        return int(getattr(self.bank, "max_bytes", 0) or 0)

    def apply_budget_bytes(self, value: int, *, reason: str) -> int:
        requested = max(0, int(value))
        per_session = int(
            getattr(self.bank, "per_session_max_bytes", requested) or requested
        )
        per_session = min(per_session, requested) if requested else 0
        rebalance = getattr(self.bank, "rebalance_limits", None)
        if not callable(rebalance):
            raise RuntimeError("SessionBank does not expose rebalance_limits")
        rebalance(
            max_bytes=requested,
            per_session_max_bytes=per_session,
            reason=reason,
        )
        return int(getattr(self.bank, "max_bytes", requested) or requested)


class AttributeBudgetConsumer:
    """Small adapter for controllers exposing a mutable byte budget."""

    def __init__(
        self, target: Any, *, name: str, attribute: str = "budget_bytes"
    ) -> None:
        self.target = target
        self.name = name
        self.attribute = attribute

    def current_budget_bytes(self) -> int:
        config = getattr(self.target, "config", self.target)
        return int(getattr(config, self.attribute, 0) or 0)

    def apply_budget_bytes(self, value: int, *, reason: str) -> int:
        requested = max(0, int(value))
        config = getattr(self.target, "config", None)
        if config is not None:
            try:
                self.target.config = replace(config, **{self.attribute: requested})
                return requested
            except Exception:
                pass
        setter = getattr(self.target, "set_budget_bytes", None)
        if callable(setter):
            result = setter(requested, reason=reason)
            return requested if result is None else int(result)
        setattr(self.target, self.attribute, requested)
        return requested


class UnifiedMemoryCoordinator:
    """Compute and atomically apply managed-memory partitions."""

    _ORDER = ("session_bank", "expert_residency", "kv_headroom")

    def __init__(self, config: UnifiedMemoryConfig | None = None) -> None:
        self.config = config or UnifiedMemoryConfig()
        self._last_plan: UnifiedMemoryPlan | None = None
        self._last_receipt: UnifiedMemoryReceipt | None = None
        self._last_apply_s = 0.0
        self._lock = threading.RLock()

    def _pressure(self, sample: UnifiedMemorySample) -> str:
        utilization = sample.utilization
        if utilization >= self.config.critical_utilization:
            return "critical"
        if utilization >= self.config.warning_utilization:
            return "warning"
        if utilization >= self.config.target_utilization:
            return "elevated"
        return "normal"

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int | None) -> int:
        value = max(minimum, int(value))
        if maximum is not None:
            value = min(value, int(maximum))
        return value

    @staticmethod
    def _digest(payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def plan(
        self,
        sample: UnifiedMemorySample,
        *,
        safe: bool,
        now_s: float | None = None,
        active_partitions: Sequence[str] | None = None,
    ) -> UnifiedMemoryPlan:
        now = time.monotonic() if now_s is None else float(now_s)
        active = set(
            active_partitions
            if active_partitions is not None
            else ("session_bank", "expert_residency", "kv_headroom")
        )
        allowed_partitions = {"session_bank", "expert_residency", "kv_headroom"}
        unknown = active - allowed_partitions
        if unknown:
            raise ValueError(f"unknown memory partitions: {sorted(unknown)}")
        pressure = self._pressure(sample)
        target_process = int(sample.total_bytes * self.config.target_utilization)
        unmanaged = max(
            0,
            sample.process_bytes
            - sample.session_bank_bytes
            - sample.expert_bytes
            - sample.kv_bytes,
        )
        available = max(
            self.config.minimum_available_bytes,
            target_process - unmanaged - self.config.reserve_bytes,
        )
        # Under warning/critical pressure, do not preserve the current managed
        # footprint merely because it is already allocated.
        if pressure == "critical":
            available = int(available * 0.72)
        elif pressure == "warning":
            available = int(available * 0.84)
        elif pressure == "elevated":
            available = int(available * 0.94)

        minima = {
            "session_bank": (
                self.config.minimum_session_bank_bytes
                if "session_bank" in active
                else 0
            ),
            "expert_residency": (
                self.config.minimum_expert_bytes if "expert_residency" in active else 0
            ),
            "kv_headroom": (
                self.config.minimum_kv_headroom_bytes if "kv_headroom" in active else 0
            ),
        }
        weights = {
            "session_bank": (
                self.config.session_bank_weight if "session_bank" in active else 0.0
            ),
            "expert_residency": (
                self.config.expert_weight if "expert_residency" in active else 0.0
            ),
            "kv_headroom": (
                self.config.kv_headroom_weight if "kv_headroom" in active else 0.0
            ),
        }
        minimum_total = sum(minima.values())
        if available < minimum_total:
            # Preserve KV headroom first, then SessionBank; the expert warm set
            # is the first elastic partition to collapse.
            kv = min(available, minima["kv_headroom"])
            remaining = max(0, available - kv)
            session = min(remaining, minima["session_bank"])
            expert = max(0, available - kv - session)
        else:
            remainder = available - minimum_total
            total_weight = sum(weights.values())
            allocations = dict(minima)
            if total_weight > 0:
                distributed = 0
                ordered = ("session_bank", "expert_residency", "kv_headroom")
                active_ordered = [name for name in ordered if weights[name] > 0]
                for name in active_ordered[:-1]:
                    addition = int(remainder * weights[name] / total_weight)
                    allocations[name] += addition
                    distributed += addition
                if active_ordered:
                    allocations[active_ordered[-1]] += remainder - distributed
            session = allocations["session_bank"]
            expert = allocations["expert_residency"]
            kv = allocations["kv_headroom"]

        session = self._clamp(
            session,
            0,
            self.config.maximum_session_bank_bytes,
        )
        expert = self._clamp(expert, 0, self.config.maximum_expert_bytes)
        kv = self._clamp(kv, 0, self.config.maximum_kv_headroom_bytes)
        # Reconcile clamping without exceeding the managed budget.
        overflow = max(0, session + expert + kv - available)
        for name in ("expert", "session", "kv"):
            if overflow <= 0:
                break
            value = {"expert": expert, "session": session, "kv": kv}[name]
            floor = {
                "expert": 0,
                "session": min(session, self.config.minimum_session_bank_bytes),
                "kv": min(kv, self.config.minimum_kv_headroom_bytes),
            }[name]
            cut = min(overflow, max(0, value - floor))
            if name == "expert":
                expert -= cut
            elif name == "session":
                session -= cut
            else:
                kv -= cut
            overflow -= cut

        eligible = self.config.enabled and safe
        reason = (
            "ready"
            if eligible
            else ("disabled" if not self.config.enabled else "unsafe_point")
        )
        if now - self._last_apply_s < self.config.minimum_apply_interval_s:
            eligible = False
            reason = "apply_interval"
        payload = {
            "at": round(now, 6),
            "pressure": pressure,
            "managed": available,
            "session": session,
            "expert": expert,
            "kv": kv,
        }
        plan = UnifiedMemoryPlan(
            plan_id=self._digest(payload),
            generated_at_s=now,
            pressure=pressure,
            total_bytes=sample.total_bytes,
            managed_budget_bytes=available,
            session_bank_budget_bytes=session,
            expert_budget_bytes=expert,
            kv_headroom_bytes=kv,
            reserve_bytes=self.config.reserve_bytes,
            eligible=eligible,
            reason=reason,
            active_partitions=tuple(sorted(active)),
        )
        with self._lock:
            self._last_plan = plan
        return plan

    def apply(
        self,
        plan: UnifiedMemoryPlan,
        consumers: Sequence[BudgetConsumer],
        *,
        safe: bool,
    ) -> UnifiedMemoryReceipt:
        started = time.perf_counter()
        if not plan.eligible:
            receipt = UnifiedMemoryReceipt(
                plan_id=plan.plan_id,
                applied=False,
                rolled_back=False,
                mutations=(),
                reason=plan.reason,
            )
            with self._lock:
                self._last_receipt = receipt
            return receipt
        if not safe:
            receipt = UnifiedMemoryReceipt(
                plan_id=plan.plan_id,
                applied=False,
                rolled_back=False,
                mutations=(),
                reason="unsafe_point",
            )
            with self._lock:
                self._last_receipt = receipt
            return receipt

        targets = {
            "session_bank": plan.session_bank_budget_bytes,
            "expert_residency": plan.expert_budget_bytes,
            "kv_headroom": plan.kv_headroom_bytes,
        }
        by_name = {consumer.name: consumer for consumer in consumers}
        ordered = [by_name[name] for name in self._ORDER if name in by_name]
        previous: list[tuple[BudgetConsumer, int]] = []
        mutations: list[BudgetMutation] = []
        rolled_back = False
        try:
            for consumer in ordered:
                before = max(0, int(consumer.current_budget_bytes()))
                requested = max(0, int(targets[consumer.name]))
                if before:
                    delta = abs(requested - before) / before
                    if delta < self.config.hysteresis_ratio:
                        mutations.append(
                            BudgetMutation(
                                consumer=consumer.name,
                                previous_bytes=before,
                                requested_bytes=requested,
                                applied_bytes=before,
                                applied=False,
                                reason="hysteresis",
                            )
                        )
                        continue
                previous.append((consumer, before))
                actual = int(
                    consumer.apply_budget_bytes(
                        requested,
                        reason=f"unified_memory:{plan.plan_id}",
                    )
                )
                mutations.append(
                    BudgetMutation(
                        consumer=consumer.name,
                        previous_bytes=before,
                        requested_bytes=requested,
                        applied_bytes=actual,
                        applied=True,
                        reason="applied",
                    )
                )
        except Exception as exc:
            rolled_back = True
            for consumer, before in reversed(previous):
                try:
                    consumer.apply_budget_bytes(
                        before,
                        reason=f"unified_memory_rollback:{plan.plan_id}",
                    )
                except Exception:
                    pass
            receipt = UnifiedMemoryReceipt(
                plan_id=plan.plan_id,
                applied=False,
                rolled_back=rolled_back,
                mutations=tuple(mutations),
                reason=f"apply_failed:{type(exc).__name__}",
            )
        else:
            receipt = UnifiedMemoryReceipt(
                plan_id=plan.plan_id,
                applied=True,
                rolled_back=False,
                mutations=tuple(mutations),
                reason="applied",
            )
            with self._lock:
                self._last_apply_s = plan.generated_at_s
        receipt = replace(
            receipt,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        with self._lock:
            self._last_receipt = receipt
        return receipt

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": True,
                "enabled": self.config.enabled,
                "atomic_rollback": True,
                "last_plan": self._last_plan.to_dict() if self._last_plan else None,
                "last_receipt": self._last_receipt.to_dict()
                if self._last_receipt
                else None,
            }
