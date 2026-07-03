"""Small cooperative scheduler core for future stepable generation.

This is not a second model worker. It is a deterministic request-state machine
that is intended to run *inside* the existing single owner thread once the
generation primitives are fully stepable.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import time
from typing import Protocol

from .admission import AdmissionPolicy, AdmissionStatus, MemoryPressure
from .state import (
    RequestPhase,
    RequestState,
    SchedulerMode,
    SchedulerPreset,
    preset_config,
)

# Upper bound on how many finished RequestState objects the scheduler keeps
# around for deduplication. The monotonic ``finished_total`` counter tracks the
# true count, so a long-running server does not accumulate every finished
# request forever.
_DEFAULT_MAX_FINISHED_RETAINED = 4096


@dataclass(frozen=True)
class BatchSchedulerConfig:
    mode: SchedulerMode = SchedulerMode.SERIAL
    preset: SchedulerPreset = SchedulerPreset.LATENCY
    max_active_requests: int | None = None
    decode_batch_max: int | None = None
    batch_wait_ms: float | None = None
    prefill_chunk_tokens: int | None = None
    experimental_mtp_cohorts: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        mode: str = "serial",
        preset: str = "latency",
        max_active_requests: int | None = None,
        decode_batch_max: int | None = None,
        batch_wait_ms: float | None = None,
        prefill_chunk_tokens: int | None = None,
        experimental_mtp_cohorts: bool = False,
    ) -> "BatchSchedulerConfig":
        resolved_preset = SchedulerPreset(preset)
        preset_defaults = preset_config(resolved_preset)
        return cls(
            mode=SchedulerMode(mode),
            preset=resolved_preset,
            max_active_requests=max_active_requests,
            decode_batch_max=decode_batch_max,
            batch_wait_ms=(
                preset_defaults.batch_wait_ms
                if batch_wait_ms is None
                else max(0.0, float(batch_wait_ms))
            ),
            prefill_chunk_tokens=prefill_chunk_tokens,
            experimental_mtp_cohorts=bool(experimental_mtp_cohorts),
        )

    def to_dict(self) -> dict[str, object]:
        defaults = preset_config(self.preset)
        return {
            "mode": self.mode.value,
            "preset": self.preset.value,
            "max_active_requests": int(
                self.max_active_requests or defaults.max_active_requests
            ),
            "decode_batch_max": int(self.decode_batch_max or defaults.decode_batch_max),
            "batch_wait_ms": float(
                defaults.batch_wait_ms
                if self.batch_wait_ms is None
                else self.batch_wait_ms
            ),
            "prefill_chunk_tokens": int(
                self.prefill_chunk_tokens or defaults.prefill_chunk_tokens
            ),
            "experimental_mtp_cohorts": bool(self.experimental_mtp_cohorts),
        }


@dataclass(frozen=True)
class StepResult:
    next_phase: RequestPhase
    prompt_tokens_done: int = 0
    generated_tokens: int = 0
    finished: bool = False


class SchedulerHooks(Protocol):
    def prefill_step(self, request: RequestState, *, token_budget: int) -> StepResult:
        ...

    def decode_step(self, requests: list[RequestState]) -> list[StepResult]:
        ...

    def postcommit_step(self, request: RequestState) -> StepResult:
        ...


@dataclass
class BatchSchedulerStats:
    started_at_s: float = field(default_factory=time.monotonic)
    steps: int = 0
    admitted: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    batch_histogram: Counter[int] = field(default_factory=Counter)
    memory_pressure: MemoryPressure = MemoryPressure.NORMAL
    last_mtp_disabled_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "uptime_s": max(0.0, time.monotonic() - self.started_at_s),
            "steps": self.steps,
            "admitted": self.admitted,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "failed": self.failed,
            "batch_histogram": {
                str(size): count for size, count in sorted(self.batch_histogram.items())
            },
            "memory_pressure": self.memory_pressure.value,
            "last_mtp_disabled_reason": self.last_mtp_disabled_reason,
        }


class MTPContinuousScheduler:
    """Cooperative request scheduler ready for stepable generation hooks."""

    def __init__(
        self,
        *,
        config: BatchSchedulerConfig,
        hooks: SchedulerHooks,
        admission: AdmissionPolicy | None = None,
    ) -> None:
        self.config = config
        self.hooks = hooks
        self.admission = admission or AdmissionPolicy(
            preset=preset_config(config.preset),
            max_active_requests=config.max_active_requests,
            decode_batch_max=config.decode_batch_max,
            prefill_chunk_tokens=config.prefill_chunk_tokens,
        )
        self.waiting: deque[RequestState] = deque()
        self.active: dict[str, RequestState] = {}
        self.prefill: deque[RequestState] = deque()
        self.decode_ready: deque[RequestState] = deque()
        self.postcommit: deque[RequestState] = deque()
        # Keyed by request_id so deduplication is O(1) and retention is bounded
        # (see ``_record_finished``). ``finished_total`` keeps the true count.
        self.finished: dict[str, RequestState] = {}
        self.finished_total = 0
        self.max_finished_retained = _DEFAULT_MAX_FINISHED_RETAINED
        self.stats = BatchSchedulerStats()

    def submit(self, request: RequestState) -> None:
        request.mark_phase(RequestPhase.QUEUED)
        self.waiting.append(request)

    def cancel(self, request_id: str) -> bool:
        for request in list(self.waiting):
            if request.request_id == request_id:
                request.cancel()
                self.stats.cancelled += 1
                return True
        for request in list(self.active.values()):
            if request.request_id == request_id:
                request.cancel()
                self.stats.cancelled += 1
                self._finish(request)
                self._purge_terminal_queues()
                return True
        return False

    def step(self) -> bool:
        """Run one cooperative unit. Returns True if any work was done."""

        self.stats.steps += 1
        self._drain_cancelled()
        admitted = self._admit_waiting()
        if self.decode_ready and self._should_coalesce_decode_ready():
            self._run_prefill_step()
            return True
        if self.decode_ready:
            self._run_decode_batch()
            return True
        if self.prefill:
            self._run_prefill_step()
            return True
        if self.postcommit:
            self._run_postcommit_step()
            return True
        return admitted

    def run_until_idle(self, *, max_steps: int = 10_000) -> None:
        for _ in range(max_steps):
            if not self.step():
                return
        raise RuntimeError("cooperative scheduler did not become idle")

    def snapshot(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "queued": len(self.waiting),
            "active": len(self.active),
            "prefill": len(self.prefill),
            "decode_ready": len(self.decode_ready),
            "postcommit": len(self.postcommit),
            "finished": self.finished_total,
            "finished_retained": len(self.finished),
            "requests": [request.to_dict() for request in self.active.values()],
        }

    def _should_coalesce_decode_ready(self) -> bool:
        if not self.prefill:
            return False
        if self.config.mode == SchedulerMode.SERIAL:
            return False
        max_batch = self.admission.effective_limits(self.stats.memory_pressure)[1]
        return len(self.decode_ready) < max_batch

    def _admit_waiting(self) -> bool:
        admitted_any = False
        remaining: deque[RequestState] = deque()
        while self.waiting:
            request = self.waiting.popleft()
            if request.cancel_event.is_set():
                request.cancel()
                self._record_finished(request)
                continue
            decision = self.admission.decide(
                request,
                active_count=len(self.active),
            )
            self.stats.memory_pressure = decision.memory_pressure
            if decision.status != AdmissionStatus.ADMIT:
                remaining.append(request)
                continue
            request.mark_phase(RequestPhase.PREFILLING)
            self.active[request.request_id] = request
            self.prefill.append(request)
            self.stats.admitted += 1
            admitted_any = True
        self.waiting = remaining
        return admitted_any

    def _run_prefill_step(self) -> None:
        request = self.prefill.popleft()
        if request.cancel_event.is_set():
            request.cancel()
            self._finish(request)
            return
        limits = self.admission.effective_limits(self.stats.memory_pressure)
        token_budget = limits[2]
        request.mark_phase(RequestPhase.PREFILLING)
        result = self.hooks.prefill_step(request, token_budget=token_budget)
        request.prompt_tokens_done += int(result.prompt_tokens_done)
        request.mark_phase(result.next_phase)
        if result.next_phase == RequestPhase.DECODE_READY:
            self.decode_ready.append(request)
        elif result.finished:
            request.finish()
            self._finish(request)
        else:
            self.prefill.append(request)

    def _run_decode_batch(self) -> None:
        max_batch = self.admission.effective_limits(self.stats.memory_pressure)[1]
        cohort: list[RequestState] = []
        while self.decode_ready and len(cohort) < max_batch:
            request = self.decode_ready.popleft()
            if request.cancel_event.is_set():
                request.cancel()
                self._finish(request)
                continue
            request.mark_phase(RequestPhase.DECODING)
            cohort.append(request)
        if not cohort:
            return
        if len(cohort) > 1:
            self.stats.last_mtp_disabled_reason = "batch_size_gt_1"
        self.stats.batch_histogram[len(cohort)] += 1
        results = self.hooks.decode_step(cohort)
        for request, result in zip(cohort, results):
            request.tokens_generated += int(result.generated_tokens)
            request.mark_phase(result.next_phase)
            if result.finished or result.next_phase == RequestPhase.FINISHED:
                request.mark_phase(RequestPhase.POSTCOMMIT)
                self.postcommit.append(request)
            else:
                self.decode_ready.append(request)

    def _run_postcommit_step(self) -> None:
        request = self.postcommit.popleft()
        if request.cancel_event.is_set():
            request.cancel()
            self._finish(request)
            return
        result = self.hooks.postcommit_step(request)
        request.mark_phase(result.next_phase)
        if result.finished or result.next_phase == RequestPhase.FINISHED:
            request.finish()
        self._finish(request)

    def _drain_cancelled(self) -> None:
        for request in list(self.active.values()):
            if request.cancel_event.is_set() and not request.is_terminal:
                request.cancel()
                self._finish(request)
        self._purge_terminal_queues()

    def _finish(self, request: RequestState) -> None:
        self.active.pop(request.request_id, None)
        if request.phase == RequestPhase.CANCELLED:
            pass
        elif request.phase == RequestPhase.FAILED:
            self.stats.failed += 1
        else:
            self.stats.completed += 1
        self._record_finished(request)

    def _record_finished(self, request: RequestState) -> None:
        """Record a request as finished, exactly once.

        A request can reach a terminal state through either the normal finish
        path or cancellation draining, so this must be idempotent. Keying the
        finished set by ``request_id`` makes the dedup check O(1) instead of the
        former O(n) identity scan over every previously finished request (which
        made a long-running session O(n**2) in the number of requests). The set
        is bounded so retained state does not grow without limit; the monotonic
        ``finished_total`` counter still reflects the true count.
        """
        if request.request_id in self.finished:
            return
        self.finished_total += 1
        self.finished[request.request_id] = request
        while len(self.finished) > self.max_finished_retained:
            oldest_id = next(iter(self.finished))
            del self.finished[oldest_id]

    def _purge_terminal_queues(self) -> None:
        self.prefill = deque(request for request in self.prefill if not request.is_terminal)
        self.decode_ready = deque(
            request for request in self.decode_ready if not request.is_terminal
        )
        self.postcommit = deque(
            request for request in self.postcommit if not request.is_terminal
        )
