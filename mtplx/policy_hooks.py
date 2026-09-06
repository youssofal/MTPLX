"""Bounded, provider-neutral policy hooks for MTPLX request lifecycles.

Hooks are in-process callables registered explicitly by trusted application
code.  MTPLX does not import policy modules from environment variables or call
an external policy service.  Every invocation is isolated, timeout-bounded,
and governed by an explicit fail-open or fail-closed rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, TimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HookPhase(str, Enum):
    REQUEST = "request"
    STREAM_EVENT = "stream_event"
    RESPONSE = "response"
    ERROR = "error"


class PolicyAction(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    REWRITE = "rewrite"
    ANNOTATE = "annotate"


class FailureMode(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class PolicyContext:
    phase: HookPhase
    request_id: str | None = None
    model: str | None = None
    session_id: str | None = None
    route: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HookResult:
    action: PolicyAction = PolicyAction.ALLOW
    value: Any | None = None
    annotations: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    status_code: int = 403

    @classmethod
    def allow(cls, *, annotations: Mapping[str, Any] | None = None) -> HookResult:
        return cls(action=PolicyAction.ALLOW, annotations=dict(annotations or {}))

    @classmethod
    def reject(cls, reason: str, *, status_code: int = 403) -> HookResult:
        return cls(
            action=PolicyAction.REJECT, reason=str(reason), status_code=int(status_code)
        )

    @classmethod
    def rewrite(
        cls,
        value: Any,
        *,
        annotations: Mapping[str, Any] | None = None,
    ) -> HookResult:
        return cls(
            action=PolicyAction.REWRITE,
            value=value,
            annotations=dict(annotations or {}),
        )

    @classmethod
    def annotate(cls, annotations: Mapping[str, Any]) -> HookResult:
        return cls(action=PolicyAction.ANNOTATE, annotations=dict(annotations))


@dataclass(frozen=True)
class PolicyOutcome:
    allowed: bool
    value: Any
    annotations: Mapping[str, Any]
    reason: str
    status_code: int
    executed_hooks: tuple[str, ...]
    failed_hooks: tuple[str, ...]
    timed_out_hooks: tuple[str, ...]
    rewritten: bool

    def to_dict(self, *, include_value: bool = False) -> dict[str, Any]:
        payload = {
            "allowed": self.allowed,
            "annotations": dict(self.annotations),
            "reason": self.reason,
            "status_code": self.status_code,
            "executed_hooks": list(self.executed_hooks),
            "failed_hooks": list(self.failed_hooks),
            "timed_out_hooks": list(self.timed_out_hooks),
            "rewritten": self.rewritten,
        }
        if include_value:
            payload["value"] = self.value
        return payload


@dataclass(frozen=True)
class PolicyHookConfig:
    maximum_hooks: int = 32
    default_timeout_s: float = 0.05
    maximum_value_bytes: int = 2 * 1024 * 1024
    maximum_annotations: int = 64
    maximum_annotation_bytes: int = 4096
    maximum_workers: int = 4
    maximum_pending_tasks: int = 16
    default_failure_mode: FailureMode = FailureMode.OPEN

    def __post_init__(self) -> None:
        if self.maximum_hooks < 1:
            raise ValueError("maximum_hooks must be at least 1")
        if self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")
        if self.maximum_value_bytes < 1:
            raise ValueError("maximum_value_bytes must be positive")
        if self.maximum_annotations < 1 or self.maximum_annotation_bytes < 1:
            raise ValueError("annotation limits must be positive")
        if self.maximum_workers < 1 or self.maximum_pending_tasks < 1:
            raise ValueError("policy executor limits must be positive")


PolicyCallable = Callable[[Any, PolicyContext], HookResult | Mapping[str, Any] | None]


@dataclass(frozen=True)
class _Registration:
    name: str
    phases: frozenset[HookPhase]
    callback: PolicyCallable
    priority: int
    timeout_s: float
    failure_mode: FailureMode


class PolicyValidationError(ValueError):
    pass


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PolicyValidationError("non-finite policy value")
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    raise PolicyValidationError(f"unsupported policy value: {type(value).__name__}")


def _encoded_size(value: Any) -> int:
    return len(
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _coerce_result(value: HookResult | Mapping[str, Any] | None) -> HookResult:
    if value is None:
        return HookResult.allow()
    if isinstance(value, HookResult):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        action = PolicyAction(str(payload.pop("action", "allow")))
        return HookResult(action=action, **payload)
    raise PolicyValidationError(f"hook returned unsupported {type(value).__name__}")


class PolicyExecutorSaturated(RuntimeError):
    pass


class _BoundedHookExecutor:
    """Fixed daemon-worker executor; timed-out hooks cannot create new threads."""

    _STOP = object()

    def __init__(self, *, workers: int, pending: int) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(workers, pending))
        self._closed = False
        self._lock = threading.Lock()
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                name=f"mtplx-policy-{index}",
                daemon=True,
            )
            for index in range(workers)
        )
        for thread in self._threads:
            thread.start()

    def submit(self, callback: PolicyCallable, *args: Any) -> Future[Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("policy executor is closed")
        future: Future[Any] = Future()
        try:
            self._queue.put_nowait((future, callback, args))
        except queue.Full as exc:
            raise PolicyExecutorSaturated("policy executor queue is full") from exc
        return future

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            future, callback, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = callback(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is self._STOP:
                continue
            future, _callback, _args = item
            future.cancel()
        for _ in self._threads:
            self._queue.put_nowait(self._STOP)


class PolicyBus:
    """Thread-safe registry and bounded lifecycle dispatcher."""

    def __init__(self, config: PolicyHookConfig | None = None) -> None:
        self.config = config or PolicyHookConfig()
        self._hooks: dict[str, _Registration] = {}
        self._lock = threading.RLock()
        self._executor: _BoundedHookExecutor | None = None
        self._executions = 0
        self._rejections = 0
        self._rewrites = 0
        self._failures = 0
        self._timeouts = 0
        self._last_outcome: dict[str, Any] | None = None

    def register(
        self,
        name: str,
        callback: PolicyCallable,
        *,
        phases: Sequence[HookPhase | str] = tuple(HookPhase),
        priority: int = 100,
        timeout_s: float | None = None,
        failure_mode: FailureMode | str | None = None,
    ) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise PolicyValidationError("hook name must not be empty")
        if not callable(callback):
            raise PolicyValidationError("hook callback must be callable")
        phase_set = frozenset(HookPhase(item) for item in phases)
        if not phase_set:
            raise PolicyValidationError("hook must subscribe to at least one phase")
        timeout = (
            self.config.default_timeout_s if timeout_s is None else float(timeout_s)
        )
        if timeout <= 0:
            raise PolicyValidationError("hook timeout must be positive")
        mode = (
            self.config.default_failure_mode
            if failure_mode is None
            else FailureMode(failure_mode)
        )
        registration = _Registration(
            name=normalized,
            phases=phase_set,
            callback=callback,
            priority=int(priority),
            timeout_s=timeout,
            failure_mode=mode,
        )
        with self._lock:
            if (
                normalized not in self._hooks
                and len(self._hooks) >= self.config.maximum_hooks
            ):
                raise PolicyValidationError("maximum policy hook count reached")
            self._hooks[normalized] = registration

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._hooks.pop(str(name), None) is not None

    def clear(self) -> None:
        with self._lock:
            self._hooks.clear()

    def _registrations(self, phase: HookPhase) -> tuple[_Registration, ...]:
        with self._lock:
            rows = [item for item in self._hooks.values() if phase in item.phases]
        rows.sort(key=lambda item: (item.priority, item.name))
        return tuple(rows)

    def _bounded_annotations(self, annotations: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for index, (key, value) in enumerate(annotations.items()):
            if index >= self.config.maximum_annotations:
                break
            clean = _json_safe(value)
            if _encoded_size(clean) > self.config.maximum_annotation_bytes:
                digest = hashlib.sha256(
                    json.dumps(clean, sort_keys=True, default=str).encode()
                ).hexdigest()
                clean = {"redacted": True, "sha256": digest}
            result[str(key)] = clean
        return result

    def _executor_for_use(self) -> _BoundedHookExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = _BoundedHookExecutor(
                    workers=min(
                        self.config.maximum_workers,
                        self.config.maximum_hooks,
                    ),
                    pending=self.config.maximum_pending_tasks,
                )
            return self._executor

    def _run_one(
        self,
        registration: _Registration,
        value: Any,
        context: PolicyContext,
    ) -> HookResult:
        isolated_value = deepcopy(value)
        isolated_context = deepcopy(context)
        future = self._executor_for_use().submit(
            registration.callback,
            isolated_value,
            isolated_context,
        )
        try:
            raw = future.result(timeout=registration.timeout_s)
            return _coerce_result(raw)
        except TimeoutError:
            future.cancel()
            raise

    def execute(
        self,
        phase: HookPhase | str,
        value: Any,
        *,
        context: PolicyContext | None = None,
    ) -> PolicyOutcome:
        phase_value = HookPhase(phase)
        safe_value = _json_safe(value)
        if _encoded_size(safe_value) > self.config.maximum_value_bytes:
            raise PolicyValidationError("policy input exceeds maximum_value_bytes")
        current = deepcopy(safe_value)
        ctx = context or PolicyContext(phase=phase_value)
        if ctx.phase != phase_value:
            ctx = PolicyContext(
                phase=phase_value,
                request_id=ctx.request_id,
                model=ctx.model,
                session_id=ctx.session_id,
                route=ctx.route,
                metadata=ctx.metadata,
            )
        annotations: dict[str, Any] = {}
        executed: list[str] = []
        failed: list[str] = []
        timed_out: list[str] = []
        rewritten = False
        allowed = True
        reason = "allowed"
        status_code = 200

        for registration in self._registrations(phase_value):
            executed.append(registration.name)
            try:
                result = self._run_one(registration, current, ctx)
            except TimeoutError:
                timed_out.append(registration.name)
                with self._lock:
                    self._timeouts += 1
                if registration.failure_mode is FailureMode.CLOSED:
                    allowed = False
                    reason = f"policy_timeout:{registration.name}"
                    status_code = 503
                    break
                continue
            except Exception as exc:
                failed.append(registration.name)
                with self._lock:
                    self._failures += 1
                if registration.failure_mode is FailureMode.CLOSED:
                    allowed = False
                    reason = f"policy_failure:{registration.name}:{type(exc).__name__}"
                    status_code = 503
                    break
                continue

            annotations.update(self._bounded_annotations(result.annotations))
            if result.action is PolicyAction.REJECT:
                allowed = False
                reason = result.reason or f"rejected:{registration.name}"
                status_code = min(599, max(400, int(result.status_code)))
                break
            if result.action is PolicyAction.REWRITE:
                rewritten_value = _json_safe(result.value)
                if _encoded_size(rewritten_value) > self.config.maximum_value_bytes:
                    failed.append(registration.name)
                    if registration.failure_mode is FailureMode.CLOSED:
                        allowed = False
                        reason = f"rewrite_too_large:{registration.name}"
                        status_code = 503
                        break
                    continue
                current = deepcopy(rewritten_value)
                rewritten = True
            elif result.action not in {
                PolicyAction.ALLOW,
                PolicyAction.ANNOTATE,
            }:
                failed.append(registration.name)

        outcome = PolicyOutcome(
            allowed=allowed,
            value=current,
            annotations=annotations,
            reason=reason,
            status_code=status_code,
            executed_hooks=tuple(executed),
            failed_hooks=tuple(failed),
            timed_out_hooks=tuple(timed_out),
            rewritten=rewritten,
        )
        with self._lock:
            self._executions += 1
            if not allowed:
                self._rejections += 1
            if rewritten:
                self._rewrites += 1
            self._last_outcome = outcome.to_dict(include_value=False)
        return outcome

    def shutdown(self) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            phases = {
                phase.value: sum(phase in item.phases for item in self._hooks.values())
                for phase in HookPhase
            }
            return {
                "available": True,
                "enabled": bool(self._hooks),
                "registered_hooks": len(self._hooks),
                "hooks_by_phase": phases,
                "executions": self._executions,
                "rejections": self._rejections,
                "rewrites": self._rewrites,
                "failures": self._failures,
                "timeouts": self._timeouts,
                "default_failure_mode": self.config.default_failure_mode.value,
                "last_outcome": deepcopy(self._last_outcome),
                "external_policy_dependency": False,
                "executor_started": self._executor is not None,
            }


_DEFAULT_BUS: PolicyBus | None = None
_DEFAULT_LOCK = threading.Lock()


def default_policy_bus() -> PolicyBus:
    global _DEFAULT_BUS
    with _DEFAULT_LOCK:
        if _DEFAULT_BUS is None:
            _DEFAULT_BUS = PolicyBus()
        return _DEFAULT_BUS


def reset_default_policy_bus_for_tests() -> None:
    global _DEFAULT_BUS
    with _DEFAULT_LOCK:
        if _DEFAULT_BUS is not None:
            _DEFAULT_BUS.shutdown()
        _DEFAULT_BUS = None
