"""Deterministic, provider-neutral replay and promotion gates for MTPLX.

This module is an offline verifier boundary.  It does not choose a model,
change serving policy, call a network service, or promote anything itself.
Callers supply candidate and evaluator functions, then explicitly inspect the
returned :class:`PromotionDecision`.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

_SECRET_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "access_token",
    "refresh_token",
    "password",
    "passwd",
    "secret",
    "credential",
    "cookie",
)
_REDACTED = "<redacted>"


class ReplayValidationError(ValueError):
    """Raised before execution when a replay suite is structurally unsafe."""


class ReplayIsolationError(RuntimeError):
    """Raised when request/evaluation inputs cannot be deep-copied safely."""


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReplayValidationError("non-finite floats are not replay-safe")
        return value
    raise ReplayValidationError(
        f"unsupported replay value: {type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sensitive_key(key: Any, extra: Sequence[str] = ()) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in (*_SECRET_PARTS, *extra))


def redact_sensitive(value: Any, *, sensitive_keys: Sequence[str] = ()) -> Any:
    """Recursively redact credential-bearing mapping/dataclass fields."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): (
                _REDACTED
                if _sensitive_key(key, sensitive_keys)
                else redact_sensitive(item, sensitive_keys=sensitive_keys)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item, sensitive_keys=sensitive_keys) for item in value]
    return _json_value(value)


def public_request_fingerprint(
    request: Any,
    *,
    sensitive_keys: Sequence[str] = (),
) -> str:
    """Stable exportable fingerprint; credential values are redacted first."""

    return _digest(redact_sensitive(request, sensitive_keys=sensitive_keys))


def _private_execution_digest(request: Any) -> str:
    """Non-exported dedupe key that intentionally retains credential identity."""

    return _digest(request)


def _isolated(value: Any, *, label: str) -> Any:
    try:
        copied = deepcopy(value)
    except Exception as exc:
        raise ReplayIsolationError(f"unable to isolate {label}: {type(exc).__name__}") from exc
    # Validate after copying.  This prevents opaque live objects from entering
    # digests, reports, candidate calls, or evaluator calls.
    _json_value(copied)
    return copied


def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    request: Any
    baseline_output: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    name: str
    score: float
    passed: bool
    baseline_score: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ReplayValidationError("evaluation name must not be empty")
        if not math.isfinite(float(self.score)):
            raise ReplayValidationError("evaluation score must be finite")
        if self.baseline_score is not None and not math.isfinite(float(self.baseline_score)):
            raise ReplayValidationError("baseline score must be finite")


@dataclass(frozen=True)
class ReplayError:
    phase: str
    error_type: str
    message: str
    evaluation_name: str | None = None


@dataclass(frozen=True)
class ReplayResult:
    case_id: str
    request_fingerprint: str
    output: Any | None
    evaluations: tuple[Evaluation, ...]
    errors: tuple[ReplayError, ...]
    metadata: Mapping[str, Any]
    candidate_reused: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    violations: tuple[str, ...]
    metrics: Mapping[str, float | int]
    decision_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "violations": list(self.violations),
            "metrics": dict(self.metrics),
            "decision_id": self.decision_id,
        }


@dataclass(frozen=True)
class ReplayReport:
    results: tuple[ReplayResult, ...]
    candidate_name: str

    def metrics(self, *, score_tolerance: float = 0.0) -> dict[str, float | int]:
        evaluations = [item for result in self.results for item in result.evaluations]
        errors = [item for result in self.results for item in result.errors]
        regressions = [
            item
            for item in evaluations
            if item.baseline_score is not None
            and item.score < item.baseline_score - score_tolerance
        ]
        drops = [
            max(0.0, float(item.baseline_score) - float(item.score))
            for item in evaluations
            if item.baseline_score is not None
        ]
        return {
            "case_count": len(self.results),
            "evaluation_count": len(evaluations),
            "pass_rate": (
                sum(bool(item.passed) for item in evaluations) / len(evaluations)
                if evaluations
                else 0.0
            ),
            "error_rate": len(errors) / max(1, len(self.results)),
            "regression_rate": len(regressions) / max(1, len(evaluations)),
            "mean_score_drop": sum(drops) / len(drops) if drops else 0.0,
        }

    def to_dict(
        self,
        *,
        include_outputs: bool = False,
        include_metadata: bool = False,
        include_evaluation_details: bool = False,
        include_error_messages: bool = False,
        sensitive_keys: Sequence[str] = (),
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for result in self.results:
            row: dict[str, Any] = {
                "case_id": result.case_id,
                "request_fingerprint": result.request_fingerprint,
                "candidate_reused": result.candidate_reused,
                "ok": result.ok,
                "evaluations": [],
                "errors": [],
            }
            if include_outputs:
                row["output"] = redact_sensitive(
                    result.output,
                    sensitive_keys=sensitive_keys,
                )
            if include_metadata:
                row["metadata"] = redact_sensitive(
                    result.metadata,
                    sensitive_keys=sensitive_keys,
                )
            for evaluation in result.evaluations:
                evaluation_row = {
                    "name": evaluation.name,
                    "score": evaluation.score,
                    "passed": evaluation.passed,
                    "baseline_score": evaluation.baseline_score,
                }
                if include_evaluation_details:
                    evaluation_row["details"] = redact_sensitive(
                        evaluation.details,
                        sensitive_keys=sensitive_keys,
                    )
                row["evaluations"].append(evaluation_row)
            for error in result.errors:
                error_row = {
                    "phase": error.phase,
                    "error_type": error.error_type,
                    "evaluation_name": error.evaluation_name,
                }
                if include_error_messages:
                    error_row["message"] = _REDACTED
                row["errors"].append(error_row)
            rows.append(row)
        return {
            "schema_version": 1,
            "candidate_name": self.candidate_name,
            "results": rows,
            "metrics": self.metrics(),
        }


@dataclass(frozen=True)
class RegressionPolicy:
    minimum_cases: int = 1
    minimum_evaluations: int = 1
    minimum_pass_rate: float = 1.0
    maximum_error_rate: float = 0.0
    maximum_regression_rate: float = 0.0
    maximum_mean_score_drop: float = 0.0
    score_tolerance: float = 0.0

    def evaluate(self, report: ReplayReport) -> PromotionDecision:
        metrics = report.metrics(score_tolerance=self.score_tolerance)
        violations: list[str] = []
        checks = (
            (metrics["case_count"] < self.minimum_cases, "minimum_case_count"),
            (
                metrics["evaluation_count"] < self.minimum_evaluations,
                "minimum_evaluation_count",
            ),
            (metrics["pass_rate"] < self.minimum_pass_rate, "minimum_pass_rate"),
            (metrics["error_rate"] > self.maximum_error_rate, "maximum_error_rate"),
            (
                metrics["regression_rate"] > self.maximum_regression_rate,
                "maximum_regression_rate",
            ),
            (
                metrics["mean_score_drop"] > self.maximum_mean_score_drop,
                "maximum_mean_score_drop",
            ),
        )
        violations.extend(name for failed, name in checks if failed)
        decision_payload = {
            "candidate_name": report.candidate_name,
            "policy": asdict(self),
            "metrics": metrics,
            "violations": violations,
        }
        return PromotionDecision(
            promote=not violations,
            violations=tuple(violations),
            metrics=metrics,
            decision_id=_digest(decision_payload),
        )


Candidate = Callable[[Any], Any]
Evaluator = Callable[[ReplayCase, Any, Any | None], Evaluation | Mapping[str, Any] | Any]


class CounterfactualReplay:
    """Run held-out replay cases with bounded, deterministic execution."""

    def __init__(
        self,
        *,
        max_concurrency: int = 1,
        candidate_timeout_s: float | None = None,
        evaluator_timeout_s: float | None = None,
        deduplicate_requests: bool = True,
        sensitive_keys: Sequence[str] = (),
    ) -> None:
        if max_concurrency < 1:
            raise ReplayValidationError("max_concurrency must be at least 1")
        self.max_concurrency = int(max_concurrency)
        self.candidate_timeout_s = candidate_timeout_s
        self.evaluator_timeout_s = evaluator_timeout_s
        self.deduplicate_requests = bool(deduplicate_requests)
        self.sensitive_keys = tuple(str(key).lower() for key in sensitive_keys)

    @staticmethod
    def _validate_cases(cases: Sequence[ReplayCase]) -> None:
        if not cases:
            raise ReplayValidationError("replay suite must contain at least one case")
        ids = [str(case.case_id) for case in cases]
        if any(not case_id.strip() for case_id in ids):
            raise ReplayValidationError("case_id must not be empty")
        if len(set(ids)) != len(ids):
            raise ReplayValidationError("case_id values must be unique")

    @staticmethod
    def _validate_evaluators(evaluators: Mapping[str, Evaluator]) -> None:
        names = [str(name).strip() for name in evaluators]
        if not names or any(not name for name in names):
            raise ReplayValidationError("at least one named evaluator is required")
        if len(set(names)) != len(names):
            raise ReplayValidationError("evaluator names must be unique")

    def _candidate_task(self, candidate: Candidate, request: Any) -> Any:
        return _resolve(candidate(_isolated(request, label="candidate request")))

    @staticmethod
    def _evaluation_from_value(name: str, value: Any) -> Evaluation:
        if isinstance(value, Evaluation):
            if value.name != name:
                raise ReplayValidationError(
                    f"evaluator {name!r} returned evaluation named {value.name!r}"
                )
            return value
        if isinstance(value, Mapping):
            return Evaluation(name=name, **dict(value))
        raise ReplayValidationError(
            f"evaluator {name!r} returned unsupported {type(value).__name__}"
        )

    def run(
        self,
        cases: Iterable[ReplayCase],
        *,
        candidate: Candidate,
        evaluators: Mapping[str, Evaluator],
        candidate_name: str = "candidate",
    ) -> ReplayReport:
        ordered_cases = tuple(cases)
        self._validate_cases(ordered_cases)
        self._validate_evaluators(evaluators)

        prepared: list[tuple[ReplayCase, Any, str, str]] = []
        for case in ordered_cases:
            isolated_case = ReplayCase(
                case_id=str(case.case_id),
                request=_isolated(case.request, label=f"request for {case.case_id}"),
                baseline_output=_isolated(
                    case.baseline_output,
                    label=f"baseline for {case.case_id}",
                ),
                metadata=_isolated(case.metadata, label=f"metadata for {case.case_id}"),
            )
            prepared.append(
                (
                    isolated_case,
                    isolated_case.request,
                    _private_execution_digest(isolated_case.request),
                    public_request_fingerprint(
                        isolated_case.request,
                        sensitive_keys=self.sensitive_keys,
                    ),
                )
            )

        execution_keys: list[str] = []
        requests_by_key: dict[str, Any] = {}
        for _case, request, private_digest, _public_digest in prepared:
            key = private_digest if self.deduplicate_requests else f"{private_digest}:{len(execution_keys)}"
            execution_keys.append(key)
            requests_by_key.setdefault(key, request)

        candidate_values: dict[str, Any] = {}
        candidate_errors: dict[str, ReplayError] = {}
        candidate_pool = ThreadPoolExecutor(max_workers=self.max_concurrency)
        candidate_futures: dict[str, Future[Any]] = {
            key: candidate_pool.submit(self._candidate_task, candidate, request)
            for key, request in requests_by_key.items()
        }
        try:
            for key in requests_by_key:
                try:
                    candidate_values[key] = candidate_futures[key].result(
                        timeout=self.candidate_timeout_s
                    )
                except TimeoutError:
                    candidate_futures[key].cancel()
                    candidate_errors[key] = ReplayError(
                        phase="candidate",
                        error_type="TimeoutError",
                        message="candidate timeout",
                    )
                except Exception as exc:
                    candidate_errors[key] = ReplayError(
                        phase="candidate",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
        finally:
            # A Python thread cannot be force-killed safely. Do not let a timed-
            # out callback hold the replay caller open while it unwinds; queued
            # work is cancelled and any already-running callback is detached.
            candidate_pool.shutdown(wait=False, cancel_futures=True)

        results: list[ReplayResult] = []
        seen_keys: set[str] = set()
        for index, (case, _request, _private_digest, public_digest) in enumerate(prepared):
            execution_key = execution_keys[index]
            reused = execution_key in seen_keys
            seen_keys.add(execution_key)
            if execution_key in candidate_errors:
                results.append(
                    ReplayResult(
                        case_id=case.case_id,
                        request_fingerprint=public_digest,
                        output=None,
                        evaluations=(),
                        errors=(candidate_errors[execution_key],),
                        metadata=case.metadata,
                        candidate_reused=reused,
                    )
                )
                continue

            output = _isolated(
                candidate_values[execution_key],
                label=f"candidate output for {case.case_id}",
            )
            evaluation_rows: list[Evaluation] = []
            evaluation_errors: list[ReplayError] = []
            evaluator_pool = ThreadPoolExecutor(max_workers=self.max_concurrency)
            eval_futures: list[tuple[str, Future[Any]]] = []
            for name, evaluator in evaluators.items():
                eval_case = _isolated(case, label=f"case for evaluator {name}")
                eval_output = _isolated(output, label=f"output for evaluator {name}")
                eval_baseline = _isolated(
                    case.baseline_output,
                    label=f"baseline for evaluator {name}",
                )
                eval_futures.append(
                    (
                        name,
                        evaluator_pool.submit(
                            lambda fn=evaluator, c=eval_case, o=eval_output, b=eval_baseline: _resolve(
                                fn(c, o, b)
                            )
                        ),
                    )
                )
            try:
                for name, future in eval_futures:
                    try:
                        evaluation_rows.append(
                            self._evaluation_from_value(
                                name,
                                future.result(timeout=self.evaluator_timeout_s),
                            )
                        )
                    except TimeoutError:
                        future.cancel()
                        evaluation_errors.append(
                            ReplayError(
                                phase="evaluator",
                                evaluation_name=name,
                                error_type="TimeoutError",
                                message="evaluator timeout",
                            )
                        )
                    except Exception as exc:
                        evaluation_errors.append(
                            ReplayError(
                                phase="evaluator",
                                evaluation_name=name,
                                error_type=type(exc).__name__,
                                message=str(exc),
                            )
                        )
            finally:
                evaluator_pool.shutdown(wait=False, cancel_futures=True)
            results.append(
                ReplayResult(
                    case_id=case.case_id,
                    request_fingerprint=public_digest,
                    output=output,
                    evaluations=tuple(evaluation_rows),
                    errors=tuple(evaluation_errors),
                    metadata=case.metadata,
                    candidate_reused=reused,
                )
            )

        return ReplayReport(results=tuple(results), candidate_name=str(candidate_name))


__all__ = [
    "CounterfactualReplay",
    "Evaluation",
    "PromotionDecision",
    "RegressionPolicy",
    "ReplayCase",
    "ReplayError",
    "ReplayIsolationError",
    "ReplayReport",
    "ReplayResult",
    "ReplayValidationError",
    "public_request_fingerprint",
    "redact_sensitive",
]
