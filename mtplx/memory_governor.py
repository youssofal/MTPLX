"""Safe-point runtime memory governance for Apple Silicon serving.

The governor adjusts cache *budgets*, never live model/cache state.  A decision
may be applied only after the caller proves that no foreground generation,
SessionBank restore/commit, MTP transaction, or idle postcommit is active.  All
changes use hysteresis and are reversible toward the startup budget.
"""

from __future__ import annotations

import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


GIB = 1024**3
MIB = 1024**2


class MemoryPressureLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MemoryGovernorAction(str, Enum):
    HOLD = "hold"
    SHRINK = "shrink"
    GROW = "grow"


@dataclass(frozen=True)
class MemoryGovernorConfig:
    target_utilization: float = 0.78
    high_utilization: float = 0.85
    critical_utilization: float = 0.92
    recovery_utilization: float = 0.70
    high_observations: int = 2
    recovery_observations: int = 3
    minimum_apply_interval_s: float = 5.0
    minimum_bank_bytes: int = 512 * MIB
    shrink_fraction: float = 0.75
    critical_shrink_fraction: float = 0.50
    growth_fraction: float = 0.10
    per_session_fraction: float = 2.0 / 3.0
    minimum_change_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.recovery_utilization < self.target_utilization:
            raise ValueError("recovery_utilization must be below target_utilization")
        if not self.target_utilization < self.high_utilization:
            raise ValueError("target_utilization must be below high_utilization")
        if not self.high_utilization < self.critical_utilization <= 1.0:
            raise ValueError("critical_utilization must be above high_utilization")
        if self.high_observations < 1 or self.recovery_observations < 1:
            raise ValueError("observation counts must be positive")
        if self.minimum_bank_bytes < 1:
            raise ValueError("minimum_bank_bytes must be positive")


@dataclass(frozen=True)
class MemorySafePoint:
    foreground_active: int = 0
    scheduler_pending_or_active: bool = False
    session_restore_active: bool = False
    session_commit_active: bool = False
    mtp_transaction_active: bool = False
    postcommit_active: bool = False

    @property
    def is_safe(self) -> bool:
        return not (
            int(self.foreground_active) > 0
            or bool(self.scheduler_pending_or_active)
            or bool(self.session_restore_active)
            or bool(self.session_commit_active)
            or bool(self.mtp_transaction_active)
            or bool(self.postcommit_active)
        )

    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if int(self.foreground_active) > 0:
            blockers.append("foreground")
        if self.scheduler_pending_or_active:
            blockers.append("scheduler")
        if self.session_restore_active:
            blockers.append("session_restore")
        if self.session_commit_active:
            blockers.append("session_commit")
        if self.mtp_transaction_active:
            blockers.append("mtp_transaction")
        if self.postcommit_active:
            blockers.append("postcommit")
        return tuple(blockers)


@dataclass(frozen=True)
class MemorySample:
    total_bytes: int | None
    rss_bytes: int | None
    session_bank_bytes: int
    model_bytes: int | None = None
    mlx_active_bytes: int | None = None
    mlx_cache_bytes: int | None = None
    timestamp_s: float = 0.0
    safe_point: MemorySafePoint = MemorySafePoint()

    @property
    def utilization(self) -> float | None:
        if self.total_bytes is None or self.total_bytes <= 0:
            return None
        used = max(0, int(self.rss_bytes or 0))
        return min(2.0, used / float(self.total_bytes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "rss_bytes": self.rss_bytes,
            "session_bank_bytes": int(self.session_bank_bytes),
            "model_bytes": self.model_bytes,
            "mlx_active_bytes": self.mlx_active_bytes,
            "mlx_cache_bytes": self.mlx_cache_bytes,
            "timestamp_s": float(self.timestamp_s),
            "utilization": self.utilization,
            "safe": self.safe_point.is_safe,
            "blockers": list(self.safe_point.blockers()),
        }


@dataclass(frozen=True)
class MemoryGovernorDecision:
    action: MemoryGovernorAction
    pressure: MemoryPressureLevel
    current_bank_max_bytes: int
    target_bank_max_bytes: int
    target_per_session_max_bytes: int
    utilization: float | None
    reason: str
    sample: MemorySample | None = None

    @property
    def changes_budget(self) -> bool:
        return int(self.target_bank_max_bytes) != int(self.current_bank_max_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "pressure": self.pressure.value,
            "current_bank_max_bytes": int(self.current_bank_max_bytes),
            "target_bank_max_bytes": int(self.target_bank_max_bytes),
            "target_per_session_max_bytes": int(self.target_per_session_max_bytes),
            "utilization": self.utilization,
            "reason": self.reason,
            "safe": self.sample.safe_point.is_safe if self.sample else None,
        }


@dataclass(frozen=True)
class MemoryGovernorApplyReceipt:
    applied: bool
    reason: str
    decision: MemoryGovernorDecision
    previous_bank_max_bytes: int
    previous_per_session_max_bytes: int
    evicted_entries: int = 0
    evicted_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": bool(self.applied),
            "reason": self.reason,
            "previous_bank_max_bytes": int(self.previous_bank_max_bytes),
            "previous_per_session_max_bytes": int(self.previous_per_session_max_bytes),
            "evicted_entries": int(self.evicted_entries),
            "evicted_bytes": int(self.evicted_bytes),
            "decision": self.decision.to_dict(),
        }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def detect_total_memory_bytes() -> int | None:
    override = _positive_int(os.environ.get("MTPLX_MEMORY_BUDGET"))
    if override is not None:
        return override
    if sys.platform == "darwin":
        try:
            raw = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            return _positive_int(raw.strip())
        except Exception:
            return None
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return _positive_int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None


def detect_process_rss_bytes() -> int | None:
    if sys.platform == "darwin":
        try:
            raw = subprocess.check_output(
                ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            kib = _positive_int(raw.strip())
            return kib * 1024 if kib is not None else None
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            fields = open("/proc/self/statm", encoding="utf-8").read().split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            pass
    try:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss if sys.platform == "darwin" else rss * 1024
    except (ValueError, OSError):
        return None


def _mlx_memory_value(name: str) -> int | None:
    try:
        import mlx.core as mx
    except Exception:
        return None
    candidates = [getattr(mx, name, None)]
    metal = getattr(mx, "metal", None)
    if metal is not None:
        candidates.append(getattr(metal, name, None))
    for candidate in candidates:
        if not callable(candidate):
            continue
        try:
            return _positive_int(candidate()) or 0
        except Exception:
            continue
    return None


def sample_process_memory(
    *,
    session_bank_bytes: int,
    model_bytes: int | None = None,
    safe_point: MemorySafePoint | None = None,
    total_bytes: int | None = None,
    rss_bytes: int | None = None,
) -> MemorySample:
    return MemorySample(
        total_bytes=total_bytes if total_bytes is not None else detect_total_memory_bytes(),
        rss_bytes=rss_bytes if rss_bytes is not None else detect_process_rss_bytes(),
        session_bank_bytes=max(0, int(session_bank_bytes)),
        model_bytes=_positive_int(model_bytes),
        mlx_active_bytes=_mlx_memory_value("get_active_memory"),
        mlx_cache_bytes=_mlx_memory_value("get_cache_memory"),
        timestamp_s=time.monotonic(),
        safe_point=safe_point or MemorySafePoint(),
    )


class RuntimeMemoryGovernor:
    """Hysteretic cache-budget controller applied only at proven safe points."""

    def __init__(
        self,
        *,
        initial_bank_max_bytes: int,
        initial_per_session_max_bytes: int,
        config: MemoryGovernorConfig | None = None,
    ) -> None:
        self.config = config or MemoryGovernorConfig()
        self.initial_bank_max_bytes = max(
            self.config.minimum_bank_bytes,
            int(initial_bank_max_bytes),
        )
        self.initial_per_session_max_bytes = min(
            self.initial_bank_max_bytes,
            max(1, int(initial_per_session_max_bytes)),
        )
        self.current_bank_max_bytes = self.initial_bank_max_bytes
        self.current_per_session_max_bytes = self.initial_per_session_max_bytes
        self.high_streak = 0
        self.recovery_streak = 0
        self.last_apply_s = 0.0
        self.last_decision: MemoryGovernorDecision | None = None
        self.last_receipt: MemoryGovernorApplyReceipt | None = None

    def _pressure(self, utilization: float | None) -> MemoryPressureLevel:
        if utilization is None or not math.isfinite(utilization):
            return MemoryPressureLevel.UNKNOWN
        if utilization >= self.config.critical_utilization:
            return MemoryPressureLevel.CRITICAL
        if utilization >= self.config.high_utilization:
            return MemoryPressureLevel.HIGH
        if utilization <= self.config.recovery_utilization:
            return MemoryPressureLevel.LOW
        return MemoryPressureLevel.NORMAL

    def _safe_capacity(self, sample: MemorySample) -> int:
        if sample.total_bytes is None or sample.rss_bytes is None:
            return self.current_bank_max_bytes
        non_bank_rss = max(
            int(sample.model_bytes or 0),
            max(0, int(sample.rss_bytes) - int(sample.session_bank_bytes)),
        )
        target_process_bytes = int(
            float(sample.total_bytes) * self.config.target_utilization
        )
        return max(
            self.config.minimum_bank_bytes,
            min(
                self.initial_bank_max_bytes,
                max(0, target_process_bytes - non_bank_rss),
            ),
        )

    def _per_session_target(self, bank_target: int) -> int:
        proportional = int(bank_target * self.config.per_session_fraction)
        return max(
            1,
            min(
                bank_target,
                self.initial_per_session_max_bytes,
                proportional,
            ),
        )

    def observe(self, sample: MemorySample) -> MemoryGovernorDecision:
        utilization = sample.utilization
        pressure = self._pressure(utilization)
        now = float(sample.timestamp_s or time.monotonic())

        if pressure in {MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL}:
            self.high_streak += 1
            self.recovery_streak = 0
        elif pressure == MemoryPressureLevel.LOW:
            self.recovery_streak += 1
            self.high_streak = 0
        else:
            self.high_streak = 0
            self.recovery_streak = 0

        action = MemoryGovernorAction.HOLD
        target = self.current_bank_max_bytes
        reason = pressure.value
        safe_capacity = self._safe_capacity(sample)

        if pressure == MemoryPressureLevel.CRITICAL:
            action = MemoryGovernorAction.SHRINK
            target = min(
                safe_capacity,
                int(self.current_bank_max_bytes * self.config.critical_shrink_fraction),
            )
            reason = "critical_pressure"
        elif (
            pressure == MemoryPressureLevel.HIGH
            and self.high_streak >= self.config.high_observations
        ):
            action = MemoryGovernorAction.SHRINK
            target = min(
                safe_capacity,
                int(self.current_bank_max_bytes * self.config.shrink_fraction),
            )
            reason = "sustained_high_pressure"
        elif (
            pressure == MemoryPressureLevel.LOW
            and self.recovery_streak >= self.config.recovery_observations
            and self.current_bank_max_bytes < self.initial_bank_max_bytes
        ):
            action = MemoryGovernorAction.GROW
            increment = max(
                1,
                int(self.initial_bank_max_bytes * self.config.growth_fraction),
            )
            target = min(
                self.initial_bank_max_bytes,
                self.current_bank_max_bytes + increment,
                safe_capacity,
            )
            reason = "sustained_recovery_headroom"

        target = max(
            self.config.minimum_bank_bytes,
            min(self.initial_bank_max_bytes, int(target)),
        )
        change = abs(target - self.current_bank_max_bytes)
        minimum_change = max(
            1,
            int(self.current_bank_max_bytes * self.config.minimum_change_fraction),
        )
        if action != MemoryGovernorAction.HOLD and change < minimum_change:
            action = MemoryGovernorAction.HOLD
            target = self.current_bank_max_bytes
            reason = "inside_change_deadband"

        if (
            action != MemoryGovernorAction.HOLD
            and self.last_apply_s > 0
            and now - self.last_apply_s < self.config.minimum_apply_interval_s
        ):
            action = MemoryGovernorAction.HOLD
            target = self.current_bank_max_bytes
            reason = "minimum_apply_interval"

        decision = MemoryGovernorDecision(
            action=action,
            pressure=pressure,
            current_bank_max_bytes=self.current_bank_max_bytes,
            target_bank_max_bytes=target,
            target_per_session_max_bytes=self._per_session_target(target),
            utilization=utilization,
            reason=reason,
            sample=sample,
        )
        self.last_decision = decision
        return decision

    def apply(
        self,
        decision: MemoryGovernorDecision,
        *,
        bank: Any,
    ) -> MemoryGovernorApplyReceipt:
        previous_max = int(getattr(bank, "max_bytes", self.current_bank_max_bytes))
        previous_per_session = int(
            getattr(
                bank,
                "per_session_max_bytes",
                self.current_per_session_max_bytes,
            )
        )
        sample = decision.sample
        if decision.action == MemoryGovernorAction.HOLD or not decision.changes_budget:
            receipt = MemoryGovernorApplyReceipt(
                False,
                decision.reason,
                decision,
                previous_max,
                previous_per_session,
            )
            self.last_receipt = receipt
            return receipt
        if sample is None or not sample.safe_point.is_safe:
            blockers = sample.safe_point.blockers() if sample is not None else ("no_sample",)
            receipt = MemoryGovernorApplyReceipt(
                False,
                "unsafe_point:" + ",".join(blockers),
                decision,
                previous_max,
                previous_per_session,
            )
            self.last_receipt = receipt
            return receipt

        before_entries = len(getattr(bank, "_entries", {}) or {})
        before_bytes = int(getattr(bank, "total_nbytes", 0) or 0)
        rebalance = getattr(bank, "rebalance_limits", None)
        if callable(rebalance):
            rebalance(
                max_bytes=decision.target_bank_max_bytes,
                per_session_max_bytes=decision.target_per_session_max_bytes,
                reason="runtime_memory_governor",
            )
        else:
            bank.max_bytes = int(decision.target_bank_max_bytes)
            bank.per_session_max_bytes = int(decision.target_per_session_max_bytes)
            evict = getattr(bank, "_evict_if_needed", None)
            if callable(evict):
                evict()

        self.current_bank_max_bytes = int(decision.target_bank_max_bytes)
        self.current_per_session_max_bytes = int(
            decision.target_per_session_max_bytes
        )
        self.last_apply_s = float(sample.timestamp_s or time.monotonic())
        after_entries = len(getattr(bank, "_entries", {}) or {})
        after_bytes = int(getattr(bank, "total_nbytes", 0) or 0)
        receipt = MemoryGovernorApplyReceipt(
            True,
            decision.reason,
            decision,
            previous_max,
            previous_per_session,
            evicted_entries=max(0, before_entries - after_entries),
            evicted_bytes=max(0, before_bytes - after_bytes),
        )
        self.last_receipt = receipt
        return receipt

    def to_metrics(self) -> dict[str, Any]:
        return {
            "memory_governor_bank_max_bytes": int(self.current_bank_max_bytes),
            "memory_governor_per_session_max_bytes": int(
                self.current_per_session_max_bytes
            ),
            "memory_governor_initial_bank_max_bytes": int(
                self.initial_bank_max_bytes
            ),
            "memory_governor_high_streak": int(self.high_streak),
            "memory_governor_recovery_streak": int(self.recovery_streak),
            "memory_governor_last_decision": (
                self.last_decision.to_dict() if self.last_decision else None
            ),
            "memory_governor_last_apply": (
                self.last_receipt.to_dict() if self.last_receipt else None
            ),
        }


__all__ = [
    "MemoryGovernorAction",
    "MemoryGovernorApplyReceipt",
    "MemoryGovernorConfig",
    "MemoryGovernorDecision",
    "MemoryPressureLevel",
    "MemorySafePoint",
    "MemorySample",
    "RuntimeMemoryGovernor",
    "detect_process_rss_bytes",
    "detect_total_memory_bytes",
    "sample_process_memory",
]
