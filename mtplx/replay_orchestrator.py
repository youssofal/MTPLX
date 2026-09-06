"""Deterministic capture plan and replay receipt orchestration.

This module owns bounded capture discovery, stable plan selection, stale-plan
validation, and atomic receipt storage. It deliberately does not execute a
candidate or evaluate outputs. Callers can connect any replay engine through
the mapping-only receipt boundary without importing that implementation here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ReplayPlanError(ValueError):
    """Raised when a replay plan or receipt is invalid."""


class StaleReplayPlanError(ReplayPlanError):
    """Raised when capture evidence changes after plan creation."""


@dataclass(frozen=True)
class CaptureFilter:
    model: str | None = None
    session_id: str | None = None
    status: str | None = None
    require_error: bool | None = None
    minimum_prompt_tokens: int | None = None
    maximum_prompt_tokens: int | None = None
    created_after_s: float | None = None
    created_before_s: float | None = None


@dataclass(frozen=True)
class ReplayPlanConfig:
    maximum_scan_files: int = 5000
    maximum_cases: int = 128
    maximum_file_bytes: int = 4 * 1024 * 1024
    deterministic_seed: str = "mtplx"
    deduplicate_public_fingerprints: bool = True
    require_replayable_content: bool = False

    def __post_init__(self) -> None:
        if self.maximum_scan_files < 1:
            raise ValueError("maximum_scan_files must be at least 1")
        if self.maximum_cases < 1:
            raise ValueError("maximum_cases must be at least 1")
        if self.maximum_file_bytes < 256:
            raise ValueError("maximum_file_bytes must be at least 256")


@dataclass(frozen=True)
class CaptureEnvelope:
    capture_id: str
    path: str
    file_sha256: str
    public_fingerprint: str
    request: Any | None
    baseline_output: Any | None
    metadata: Mapping[str, Any]
    replayable: bool
    unavailable_reason: str | None

    def to_dict(self, *, include_request: bool = False) -> dict[str, Any]:
        payload = {
            "capture_id": self.capture_id,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "public_fingerprint": self.public_fingerprint,
            "metadata": dict(self.metadata),
            "replayable": self.replayable,
            "unavailable_reason": self.unavailable_reason,
        }
        if include_request:
            payload["request"] = self.request
            payload["baseline_output"] = self.baseline_output
        return payload


@dataclass(frozen=True)
class ReplayPlan:
    plan_id: str
    created_at_s: float
    capture_root: str
    cases: tuple[CaptureEnvelope, ...]
    skipped_count: int
    duplicate_count: int
    source_digest: str
    config: ReplayPlanConfig
    filters: CaptureFilter

    @property
    def replayable_cases(self) -> int:
        return sum(item.replayable for item in self.cases)

    def to_dict(self, *, include_requests: bool = False) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "created_at_s": self.created_at_s,
            "capture_root": self.capture_root,
            "cases": [
                item.to_dict(include_request=include_requests) for item in self.cases
            ],
            "case_count": len(self.cases),
            "replayable_cases": self.replayable_cases,
            "skipped_count": self.skipped_count,
            "duplicate_count": self.duplicate_count,
            "source_digest": self.source_digest,
            "config": asdict(self.config),
            "filters": asdict(self.filters),
            "promotion_is_automatic": False,
        }


@dataclass(frozen=True)
class ReplayReceipt:
    receipt_id: str
    plan_id: str
    source_digest: str
    candidate_name: str
    created_at_s: float
    report: Mapping[str, Any]
    decision: Mapping[str, Any]
    stale_check_passed: bool
    promotion_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "source_digest": self.source_digest,
            "candidate_name": self.candidate_name,
            "created_at_s": self.created_at_s,
            "report": dict(self.report),
            "decision": dict(self.decision),
            "stale_check_passed": self.stale_check_passed,
            "promotion_applied": self.promotion_applied,
        }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _first(mapping: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        value: Any = mapping
        found = True
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found:
            return value
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return None


def _public_fingerprint(record: Mapping[str, Any], file_sha256: str) -> str:
    value = _first(
        record,
        ("request_fingerprint",),
        ("request", "fingerprint"),
        ("request", "sha256"),
        ("prompt_sha256",),
        ("prompt", "sha256"),
    )
    if isinstance(value, str) and value:
        return value
    return file_sha256


def _extract_replay_request(record: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    candidates = (
        _first(record, ("request", "payload")),
        _first(record, ("request", "body")),
        _first(record, ("request_payload",)),
        _first(record, ("messages",)),
    )
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, Mapping) and value.get("content_redacted") is True:
            continue
        if isinstance(value, Mapping) and set(value) <= {
            "count",
            "bytes",
            "sha256",
            "content_bytes",
            "content_sha256",
            "content_redacted",
        }:
            continue
        return value, None
    return None, "request_content_not_captured"


def _extract_baseline(record: Mapping[str, Any]) -> Any | None:
    for value in (
        _first(record, ("outcome", "response")),
        _first(record, ("outcome", "body")),
        _first(record, ("response",)),
        _first(record, ("response_payload",)),
    ):
        if value is None:
            continue
        if isinstance(value, Mapping) and value.get("content_redacted") is True:
            continue
        return value
    return None


class ReplayOrchestrator:
    """Build stable plans and persist engine-neutral replay receipts."""

    def __init__(
        self,
        capture_root: str | os.PathLike[str],
        *,
        config: ReplayPlanConfig | None = None,
        receipt_directory: str | os.PathLike[str] | None = None,
    ) -> None:
        self.capture_root = Path(capture_root).expanduser().resolve()
        self.config = config or ReplayPlanConfig()
        self.receipt_directory = (
            Path(receipt_directory).expanduser().resolve()
            if receipt_directory is not None
            else self.capture_root / "replay_receipts"
        )
        self._last_plan: ReplayPlan | None = None
        self._last_receipt: ReplayReceipt | None = None
        self._lock = threading.RLock()

    def _capture_files(self) -> tuple[Path, ...]:
        if not self.capture_root.exists():
            return ()
        rows: list[Path] = []
        for path in self.capture_root.rglob("*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(self.capture_root)
            except (OSError, ValueError):
                continue
            if self.receipt_directory in resolved.parents or "pruned" in resolved.parts:
                continue
            rows.append(resolved)
        rows.sort(key=lambda path: path.as_posix())
        return tuple(rows[: self.config.maximum_scan_files])

    def _read(self, path: Path) -> CaptureEnvelope | None:
        try:
            stat = path.stat()
            if stat.st_size > self.config.maximum_file_bytes:
                return None
            raw = path.read_bytes()
            record = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(record, Mapping):
            return None
        file_sha = hashlib.sha256(raw).hexdigest()
        capture_id = str(
            _first(record, ("request_id",), ("capture_id",), ("id",)) or path.stem
        )
        request, unavailable = _extract_replay_request(record)
        metadata = {
            "model": _first(record, ("model",), ("request", "model")),
            "session_id": _first(record, ("session_id",), ("request", "session_id")),
            "status": _first(record, ("outcome", "status"), ("status",)),
            "error_type": _first(
                record,
                ("outcome", "error_type"),
                ("error", "type"),
                ("error_type",),
            ),
            "prompt_tokens": _safe_int(
                _first(
                    record,
                    ("prompt_tokens",),
                    ("request", "prompt_tokens"),
                    ("usage", "prompt_tokens"),
                )
            ),
            "created_at_s": _safe_float(
                _first(
                    record,
                    ("created_at_s",),
                    ("timestamp_s",),
                    ("created_at",),
                )
            ),
        }
        return CaptureEnvelope(
            capture_id=capture_id,
            path=str(path.relative_to(self.capture_root)),
            file_sha256=file_sha,
            public_fingerprint=_public_fingerprint(record, file_sha),
            request=request,
            baseline_output=_extract_baseline(record),
            metadata=metadata,
            replayable=request is not None,
            unavailable_reason=unavailable,
        )

    @staticmethod
    def _matches(envelope: CaptureEnvelope, filters: CaptureFilter) -> bool:
        metadata = envelope.metadata
        if filters.model is not None and metadata.get("model") != filters.model:
            return False
        if filters.session_id is not None and metadata.get("session_id") != filters.session_id:
            return False
        if filters.status is not None and str(metadata.get("status")) != filters.status:
            return False
        has_error = bool(metadata.get("error_type"))
        if filters.require_error is not None and has_error != filters.require_error:
            return False
        prompt_tokens = _safe_int(metadata.get("prompt_tokens"))
        if filters.minimum_prompt_tokens is not None and (
            prompt_tokens is None or prompt_tokens < filters.minimum_prompt_tokens
        ):
            return False
        if filters.maximum_prompt_tokens is not None and (
            prompt_tokens is None or prompt_tokens > filters.maximum_prompt_tokens
        ):
            return False
        created = _safe_float(metadata.get("created_at_s"))
        if filters.created_after_s is not None and (
            created is None or created < filters.created_after_s
        ):
            return False
        if filters.created_before_s is not None and (
            created is None or created > filters.created_before_s
        ):
            return False
        return True

    def _rank_key(self, envelope: CaptureEnvelope) -> str:
        return _digest(
            {
                "seed": self.config.deterministic_seed,
                "capture": envelope.capture_id,
                "fingerprint": envelope.public_fingerprint,
            }
        )

    @staticmethod
    def _source_digest(cases: Sequence[CaptureEnvelope]) -> str:
        return _digest(
            [
                {
                    "path": item.path,
                    "sha256": item.file_sha256,
                    "fingerprint": item.public_fingerprint,
                }
                for item in cases
            ]
        )

    def _select(
        self, filters: CaptureFilter
    ) -> tuple[tuple[CaptureEnvelope, ...], int, int]:
        skipped = 0
        duplicate_count = 0
        candidates: list[CaptureEnvelope] = []
        seen: set[str] = set()
        for path in self._capture_files():
            envelope = self._read(path)
            if envelope is None or not self._matches(envelope, filters):
                skipped += 1
                continue
            if self.config.require_replayable_content and not envelope.replayable:
                skipped += 1
                continue
            if (
                self.config.deduplicate_public_fingerprints
                and envelope.public_fingerprint in seen
            ):
                duplicate_count += 1
                continue
            seen.add(envelope.public_fingerprint)
            candidates.append(envelope)
        candidates.sort(key=lambda item: (self._rank_key(item), item.path))
        cases = tuple(candidates[: self.config.maximum_cases])
        return cases, skipped, duplicate_count

    def build_plan(self, filters: CaptureFilter | None = None) -> ReplayPlan:
        selected_filter = filters or CaptureFilter()
        cases, skipped, duplicate_count = self._select(selected_filter)
        source_digest = self._source_digest(cases)
        created = time.time()
        plan_id = _digest(
            {
                "created_at_s": created,
                "source_digest": source_digest,
                "config": asdict(self.config),
                "filters": asdict(selected_filter),
            }
        )[:24]
        plan = ReplayPlan(
            plan_id=plan_id,
            created_at_s=created,
            capture_root=str(self.capture_root),
            cases=cases,
            skipped_count=skipped,
            duplicate_count=duplicate_count,
            source_digest=source_digest,
            config=self.config,
            filters=selected_filter,
        )
        with self._lock:
            self._last_plan = plan
        return plan

    def assert_fresh(self, plan: ReplayPlan) -> None:
        if Path(plan.capture_root) != self.capture_root:
            raise ReplayPlanError("plan capture root does not match orchestrator")
        if plan.config != self.config:
            raise ReplayPlanError("plan configuration does not match orchestrator")
        current, _, _ = self._select(plan.filters)
        current_digest = self._source_digest(current)
        if current_digest != plan.source_digest:
            raise StaleReplayPlanError("capture set changed after replay plan creation")

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def record_receipt(
        self,
        plan: ReplayPlan,
        *,
        candidate_name: str,
        report: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> ReplayReceipt:
        """Validate source freshness and atomically persist an engine result."""
        if not candidate_name.strip():
            raise ReplayPlanError("candidate_name must not be empty")
        self.assert_fresh(plan)
        created = time.time()
        receipt_payload = {
            "plan_id": plan.plan_id,
            "source_digest": plan.source_digest,
            "candidate_name": candidate_name,
            "created_at_s": created,
            "report": dict(report),
            "decision": dict(decision),
            "stale_check_passed": True,
            "promotion_applied": False,
        }
        receipt = ReplayReceipt(
            receipt_id=_digest(receipt_payload)[:24],
            **receipt_payload,
        )
        self._atomic_json(
            self.receipt_directory / f"{receipt.receipt_id}.json",
            receipt.to_dict(),
        )
        with self._lock:
            self._last_receipt = receipt
        return receipt

    def list_receipts(self, *, limit: int = 20) -> tuple[dict[str, Any], ...]:
        if not self.receipt_directory.exists():
            return ()
        rows: list[dict[str, Any]] = []
        paths = sorted(
            self.receipt_directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths[: max(0, int(limit))]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return tuple(rows)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            receipt_count = 0
            if self.receipt_directory.exists():
                try:
                    receipt_count = sum(
                        1 for _ in self.receipt_directory.glob("*.json")
                    )
                except OSError:
                    receipt_count = 0
            return {
                "available": True,
                "enabled": self.capture_root.exists(),
                "capture_root_configured": bool(str(self.capture_root)),
                "maximum_cases": self.config.maximum_cases,
                "content_capture_required": False,
                "promotion_is_automatic": False,
                "receipt_count": receipt_count,
                "last_plan": self._last_plan.to_dict(include_requests=False)
                if self._last_plan
                else None,
                "last_receipt": self._last_receipt.to_dict()
                if self._last_receipt
                else None,
            }
