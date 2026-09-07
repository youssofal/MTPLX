"""Out-of-band memory curation for MTPLX.

Dreaming is intentionally reviewable and conservative. It verifies the
current store, organizes transcript observations, and creates proposals only
for facts explicitly marked for retention. Applying a run requires the source
snapshot to remain unchanged and writes through the memory store's hash
precondition.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from mtplx.memory import (
    MemoryConflictError,
    MemoryDocument,
    MemoryStore,
    MemoryStoreError,
    content_sha256,
    safe_component,
    utc_now,
)


_EXPLICIT_MEMORY_RE = re.compile(
    r"^\s*(?:memory|remember|durable memory|save)\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class DreamingRun:
    run_id: str
    status: str
    started_at: str
    completed_at: str | None
    source_digest: str
    transcript_count: int
    memory_count: int
    invalid_count: int
    proposals: list[dict[str, Any]]
    report_path: str
    applied: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "source_digest": self.source_digest,
            "transcript_count": self.transcript_count,
            "memory_count": self.memory_count,
            "invalid_count": self.invalid_count,
            "proposals": self.proposals,
            "report_path": self.report_path,
            "applied": self.applied,
            "error": self.error,
        }


class DreamingService:
    """Run memory curation outside model request execution."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or MemoryStore.from_env()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mtplx-dreaming")
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def start(self, *, max_transcripts: int = 100) -> dict[str, Any]:
        run_id = f"dream-{uuid.uuid4().hex[:12]}"
        with self._lock:
            future = self._executor.submit(
                self.run,
                run_id=run_id,
                max_transcripts=max_transcripts,
            )
            self._futures[run_id] = future
        return {
            "run_id": run_id,
            "status": "queued",
            "poll": f"/v1/mtplx/memory/dream/{run_id}",
        }

    def run(
        self,
        *,
        run_id: str | None = None,
        max_transcripts: int = 100,
    ) -> dict[str, Any]:
        started_at = utc_now()
        run_id = run_id or f"dream-{uuid.uuid4().hex[:12]}"
        try:
            documents = self.store.list_documents(scope="all", limit=5000)
            transcripts = [doc for doc in documents if doc.scope == "transcripts"][-max(1, int(max_transcripts)) :]
            memories = [doc for doc in documents if doc.scope in {"shared", "working"}]
            source_digest = _source_digest(documents)
            invalid_count = 0
            for document in documents:
                if document.content_sha256 != content_sha256(document.content):
                    invalid_count += 1
            proposals = _build_proposals(transcripts)
            report_path = f"dreaming/runs/{safe_component(run_id)}.md"
            report = _render_report(
                run_id=run_id,
                started_at=started_at,
                source_digest=source_digest,
                transcripts=transcripts,
                memories=memories,
                invalid_count=invalid_count,
                proposals=proposals,
            )
            result = DreamingRun(
                run_id=run_id,
                status="completed",
                started_at=started_at,
                completed_at=utc_now(),
                source_digest=source_digest,
                transcript_count=len(transcripts),
                memory_count=len(memories),
                invalid_count=invalid_count,
                proposals=proposals,
                report_path=report_path,
            )
            self.store.write(
                report_path,
                report,
                author="dreaming",
                metadata={"kind": "dreaming_run", "run": result.to_dict()},
            )
            return result.to_dict()
        except Exception as exc:
            result = DreamingRun(
                run_id=run_id,
                status="failed",
                started_at=started_at,
                completed_at=utc_now(),
                source_digest="",
                transcript_count=0,
                memory_count=0,
                invalid_count=0,
                proposals=[],
                report_path=f"dreaming/runs/{safe_component(run_id)}.md",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.store.write(
                result.report_path,
                f"# Dreaming run {run_id}\n\nFailed: {result.error}\n",
                author="dreaming",
                metadata={"kind": "dreaming_run", "run": result.to_dict()},
            )
            return result.to_dict()

    def get(self, run_id: str) -> dict[str, Any] | None:
        normalized = f"dreaming/runs/{safe_component(run_id)}.md"
        try:
            document = self.store.read(normalized)
        except MemoryStoreError:
            with self._lock:
                future = self._futures.get(run_id)
            if future is not None and future.done():
                return future.result()
            return {"run_id": run_id, "status": "queued"} if future else None
        run = document.metadata.get("run")
        return dict(run) if isinstance(run, dict) else None

    def apply(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        if not run:
            raise MemoryStoreError(f"dreaming run not found: {run_id}")
        if run.get("status") != "completed":
            raise MemoryStoreError(f"dreaming run is not applicable: {run.get('status')}")
        documents = self.store.list_documents(scope="all", limit=5000)
        current_digest = _source_digest(documents)
        if current_digest != run.get("source_digest"):
            raise MemoryConflictError(
                f"dreaming/{run_id}", str(run.get("source_digest")), current_digest
            )
        applied: list[str] = []
        for proposal in run.get("proposals") or []:
            path = str(proposal["path"])
            existing = None
            try:
                existing = self.store.read(path)
            except MemoryStoreError:
                pass
            current_content = existing.content if existing else ""
            additions = str(proposal.get("content") or "")
            if additions and additions not in current_content:
                new_content = current_content + ("\n" if current_content else "") + additions
                self.store.write(
                    path,
                    new_content,
                    expected_sha256=existing.content_sha256 if existing else None,
                    author="dreaming",
                    agent_id=proposal.get("agent_id"),
                    metadata={"kind": "dreaming_candidate", "source_run": run_id},
                )
                applied.append(path)
        run["status"] = "applied"
        run["applied"] = True
        run["applied_paths"] = applied
        report_path = str(run.get("report_path") or "")
        report = self.store.read(report_path)
        self.store.write(
            report_path,
            report.content,
            expected_sha256=report.content_sha256,
            author="dreaming",
            metadata={"kind": "dreaming_run", "run": run},
        )
        return {"run_id": run_id, "status": "applied", "paths": applied}

    def reap(self) -> None:
        with self._lock:
            finished = [run_id for run_id, future in self._futures.items() if future.done()]
            for run_id in finished:
                self._futures.pop(run_id, None)


def _source_digest(documents: list[MemoryDocument]) -> str:
    payload = [
        {"path": doc.path, "content_sha256": doc.content_sha256, "version": doc.version}
        for doc in sorted(documents, key=lambda item: item.path)
        if doc.scope in {"shared", "working", "transcripts"}
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _build_proposals(transcripts: list[MemoryDocument]) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for transcript in transcripts:
        agent_id = safe_component(str(transcript.metadata.get("agent_id") or "default"))
        for match in _EXPLICIT_MEMORY_RE.finditer(transcript.content):
            value = " ".join(match.group("value").split())
            if not value:
                continue
            key = (agent_id, value.casefold())
            entry = candidates.setdefault(
                key,
                {
                    "path": f"working/{agent_id}/dreaming-candidates.md",
                    "agent_id": agent_id,
                    "content": f"- {value}",
                    "reason": "explicit memory marker in a session transcript",
                    "source_paths": [],
                },
            )
            if transcript.path not in entry["source_paths"]:
                entry["source_paths"].append(transcript.path)
    return sorted(candidates.values(), key=lambda item: (item["path"], item["content"]))


def _render_report(
    *,
    run_id: str,
    started_at: str,
    source_digest: str,
    transcripts: list[MemoryDocument],
    memories: list[MemoryDocument],
    invalid_count: int,
    proposals: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Dreaming run {run_id}",
        "",
        f"Started: {started_at}",
        f"Source digest: `{source_digest}`",
        "",
        "## Verify",
        "",
        f"Inspected {len(memories)} durable memory documents and {len(transcripts)} session transcripts.",
        f"Content hash mismatches: {invalid_count}.",
        "",
        "## Organize",
        "",
        "Transcripts were grouped by session and agent. Existing shared and working documents were left unchanged.",
        "",
        "## Enrich",
        "",
    ]
    if proposals:
        lines.append("The following explicit memory markers are reviewable proposals:")
        lines.append("")
        for proposal in proposals:
            lines.append(f"- `{proposal['path']}`: {proposal['content']}")
            lines.append(f"  Sources: {', '.join(proposal['source_paths'])}")
    else:
        lines.append("No explicit memory markers were found. No durable memory was proposed.")
    lines.extend(
        (
            "",
            "Apply this run only after review. Applying requires the source digest to remain unchanged.",
            "",
        )
    )
    return "\n".join(lines)


__all__ = ["DreamingRun", "DreamingService"]
