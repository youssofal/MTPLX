"""Per-request capture for bit-exact failure replay (#196/#197 third layer).

The rare engine-side early stop only shows up on real agent turns; single-turn
probes never reproduce it (0 in 90 across MTP/AR/template cells, 2026-07-26).
This module persists every generation request's reproduction envelope at
DISPATCH TIME — before any token is generated — so a turn that hangs, dies, or
stops early still leaves everything needed to replay it: the exact post-encoding
prompt token ids, the resolved sampler, the requested seed, mode/depth, session
identity, and the template hash. The outcome (resolved seed, finish reason,
counts, text head/tail) is merged into the same file at completion.

Off by default. Enable with MTPLX_REQUEST_CAPTURE_DIR=<dir>; the ring keeps the
newest MTPLX_REQUEST_CAPTURE_KEEP files (default 200) and prunes older ones by
renaming into a ``pruned/`` subdirectory (house rule: never delete).

Files are ``req-<utcstamp>-<request-id>.json``, written atomically
(tmp + rename). Payloads are plain JSON with only primitive types.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_PATHS_BY_ID: dict[str, str] = {}


def capture_dir() -> str | None:
    raw = str(os.environ.get("MTPLX_REQUEST_CAPTURE_DIR", "")).strip()
    return raw or None


def _keep_count() -> int:
    try:
        return max(1, int(os.environ.get("MTPLX_REQUEST_CAPTURE_KEEP", "200")))
    except ValueError:
        return 200


def _atomic_write(path: str, payload: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _prune_locked(directory: str) -> None:
    try:
        entries = sorted(
            name
            for name in os.listdir(directory)
            if name.startswith("req-") and name.endswith(".json")
        )
    except OSError:
        return
    excess = len(entries) - _keep_count()
    if excess <= 0:
        return
    pruned_dir = os.path.join(directory, "pruned")
    os.makedirs(pruned_dir, exist_ok=True)
    for name in entries[:excess]:
        try:
            os.replace(
                os.path.join(directory, name), os.path.join(pruned_dir, name)
            )
        except OSError:
            pass


def capture_request(request_id: str | None, payload: dict[str, Any]) -> None:
    """Persist the reproduction envelope at dispatch time. Never raises."""
    directory = capture_dir()
    if not directory or not request_id:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        safe_id = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(request_id)
        )[:80]
        path = os.path.join(directory, f"req-{stamp}-{safe_id}.json")
        record = {
            "capture_version": 1,
            "captured_at_utc": stamp,
            "request_id": str(request_id),
            "phase": "dispatched",
            **payload,
        }
        with _LOCK:
            _atomic_write(path, record)
            _PATHS_BY_ID[str(request_id)] = path
            _prune_locked(directory)
    except Exception:
        pass


def capture_outcome(request_id: str | None, outcome: dict[str, Any]) -> None:
    """Merge the completion outcome into the request's capture file. Never raises."""
    if not capture_dir() or not request_id:
        return
    try:
        with _LOCK:
            path = _PATHS_BY_ID.get(str(request_id))
            if not path or not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["phase"] = "completed"
            record["outcome"] = outcome
            _atomic_write(path, record)
    except Exception:
        pass


def clip_text_head_tail(text: str, head: int = 2000, tail: int = 2000) -> dict[str, Any]:
    """Store enough text to diagnose early stops without unbounded files —
    the tail is where the failure signature lives."""
    text = str(text or "")
    if len(text) <= head + tail:
        return {"text": text, "text_clipped": False}
    return {
        "text_head": text[:head],
        "text_tail": text[-tail:],
        "text_chars": len(text),
        "text_clipped": True,
    }
