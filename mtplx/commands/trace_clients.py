"""Normalize local harness transcripts without changing their history."""

from __future__ import annotations

import datetime
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def load_hermes_session(db: Path, session_id: str, log: Path | None = None) -> tuple[dict, list[dict]]:
    """Read Hermes' durable messages and API-call receipts, without modifying it.

    Hermes stores reasoning/tool arguments separately. Its log adds input/output
    counts and API latency; completion clocks plus both counts identify a request
    even when the engine assigned an anonymous, shared-prefix session id.
    """
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        if session_id == "latest":
            row = conn.execute("SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
            if row is None:
                raise ValueError("No Hermes sessions found")
            session_id = row["id"]
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"Hermes session not found: {session_id}")
        session = dict(row)
        records = [dict(r) for r in conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id", (session_id,)
        )]
    calls = []
    log = log or db.parent / "logs" / "agent.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if f"[{session_id}]" not in line:
                continue
            match = re.search(r"API call #\d+:.* in=(\d+) out=(\d+).* latency=([\d.]+)s", line)
            if match:
                calls.append((datetime.datetime.fromisoformat(line[:23]).timestamp(),
                              int(match[1]), int(match[2]), float(match[3])))
    results = {r["tool_call_id"]: r for r in records if r["role"] == "tool" and r["tool_call_id"]}
    messages = []
    previous = float(session["started_at"])
    for r in records:
        stamp = float(r["timestamp"])
        if r["role"] not in {"user", "assistant"}:
            previous = stamp
            continue
        parts = []
        if r.get("reasoning_content") or r.get("reasoning"):
            parts.append({"type": "reasoning", "text": r.get("reasoning_content") or r["reasoning"]})
        if r.get("content"):
            parts.append({"type": "text", "text": r["content"]})
        for tool in json.loads(r.get("tool_calls") or "[]"):
            fn = tool.get("function") or {}
            result = results.get(tool.get("id"))
            state = {"input": fn.get("arguments"), "status": "running"}
            if result:
                output = result.get("content") or ""
                cancelled = output.startswith("[Tool execution cancelled")
                state.update(output=output, status="error" if cancelled else "completed",
                             time={"start": stamp * 1000, "end": result["timestamp"] * 1000})
            parts.append({"type": "tool", "tool": fn.get("name"), "callID": tool.get("id"), "state": state})
        call = min(calls, key=lambda c: abs(c[0] - stamp), default=None) if r["role"] == "assistant" else None
        if call and abs(call[0] - stamp) > 2:
            call = None
        created = stamp - call[3] if call else previous
        messages.append({**r, "_id": str(r["id"]), "_parts": parts, "_client": "hermes",
                         "_time_created_s": created, "_hermes_api_counts": call[1:3] if call else None,
                         "time": {"created": created * 1000, "completed": stamp * 1000},
                         "tokens": {"input": call[1] if call else 0, "output": call[2] if call else 0},
                         "finish": r.get("finish_reason")})
        previous = stamp
    return {"id": session_id, "title": session.get("title"), "client": "hermes",
            "directory": session.get("cwd"), "transcript_path": str(db)}, messages


def load_pi_session(path: Path) -> tuple[dict, list[dict]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    header = next((r for r in records if r.get("type") == "session"), {})
    # Pi is an append-only tree. Follow the current leaf's ancestors, so a
    # fork or rewind does not quietly include abandoned turns in the report.
    entries = {r["id"]: r for r in records if r.get("id") and r.get("type") != "session"}
    lineage, seen = [], set()
    current = next((r for r in reversed(records) if r.get("id") in entries), None)
    while current and current["id"] not in seen:
        seen.add(current["id"])
        lineage.append(current)
        current = entries.get(current.get("parentId"))
    tool_results = {r["message"]["toolCallId"]: r["message"] for r in lineage
                    if r.get("message", {}).get("role") == "toolResult"}
    messages = []
    for record in reversed(lineage):
        if record.get("type") != "message":
            continue
        message = dict(record.get("message") or {})
        if message.get("role") not in {"user", "assistant"}:
            continue
        stamp = message.get("timestamp")
        if stamp is None:
            stamp = datetime.datetime.fromisoformat(record["timestamp"]).timestamp() * 1000
        usage = message.get("usage") or {}
        completed = (datetime.datetime.fromisoformat(record["timestamp"]).timestamp() * 1000
                     if message.get("role") == "assistant" and message.get("stopReason")
                     and record.get("timestamp") else None)
        content = message.get("content") or []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            kind = part.get("type")
            if kind == "thinking":
                parts.append({"type": "reasoning", "text": part.get("thinking", "")})
            elif kind == "toolCall":
                result = tool_results.get(part.get("id"))
                state = {"input": part.get("arguments")}
                if result is not None:
                    state.update(
                        status="error" if result.get("isError") else "completed",
                        output=result.get("content"),
                        metadata={"result_timestamp_ms": result.get("timestamp")},
                    )
                parts.append({"type": "tool", "tool": part.get("name"), "callID": part.get("id"),
                              "state": state})
            else:
                parts.append(part)
        messages.append({
            **message, "_id": record["id"], "_parts": parts, "_client": "pi",
            "_parent_entry_id": record.get("parentId"),
            "_time_created_s": float(stamp) / 1000,
            "time": {"created": stamp, "completed": completed},
            "tokens": {"input": usage.get("input", 0), "output": usage.get("output", 0),
                       "cache": {"read": usage.get("cacheRead", 0)}},
            "finish": message.get("stopReason"),
        })
    return {"id": header.get("id"), "directory": header.get("cwd"),
            "title": path.stem, "client": "pi", "transcript_path": str(path)}, messages
