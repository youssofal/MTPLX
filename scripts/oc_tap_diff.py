#!/usr/bin/env python3
"""Diff consecutive chat requests captured by oc_tap.py for one client loop.

Answers: which message slots changed between request N and N+1 — appended
(normal agent-loop growth), mutated in place (client rewrote history), or
removed. Prints per-request one-liners plus a mutation report whenever a
previously-sent slot's content hash changed or shrank.

Usage: python3 scripts/oc_tap_diff.py [journal] [--full-on-mutation]
"""

from __future__ import annotations

import hashlib
import json
import sys


def norm_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(parts)
    return json.dumps(content, sort_keys=True) if content is not None else ""


def slot_key(message: dict) -> tuple:
    role = message.get("role")
    if role == "tool":
        return (role, message.get("tool_call_id"))
    if role == "assistant" and message.get("tool_calls"):
        ids = tuple(
            (c.get("id"), (c.get("function") or {}).get("name"))
            for c in message.get("tool_calls") or []
        )
        return (role, ids)
    return (role, None)


def summarize(message: dict) -> dict:
    content = norm_content(message.get("content"))
    reasoning = norm_content(
        message.get("reasoning_content") or message.get("reasoning") or ""
    )
    tool_calls = message.get("tool_calls") or []
    args_chars = sum(
        len(str((c.get("function") or {}).get("arguments") or ""))
        for c in tool_calls
    )
    return {
        "role": message.get("role"),
        "chars": len(content),
        "sha8": hashlib.sha256(content.encode()).hexdigest()[:8],
        "reasoning_chars": len(reasoning),
        "tool_calls": len(tool_calls),
        "args_chars": args_chars,
        "tool_call_id": message.get("tool_call_id"),
    }


def main() -> None:
    journal = sys.argv[1] if len(sys.argv) > 1 else None
    if not journal:
        import os

        journal = os.path.expanduser("~/.mtplx/logs/oc-tap.jsonl")
    rows = []
    with open(journal, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("request_body", {}).get("messages"):
                rows.append(r)
    prev = None
    for idx, r in enumerate(rows):
        msgs = [summarize(m) for m in r["request_body"]["messages"]]
        total_chars = sum(
            m["chars"] + m["args_chars"] for m in msgs
        )
        line = (
            f"req{idx} ts={r['ts_ms']} n={len(msgs)} chars={total_chars} "
            f"stream={r['request_body'].get('stream')} "
            f"resp_ms={r.get('response_ms')}"
        )
        mutations = []
        if prev is not None:
            shared = min(len(prev), len(msgs))
            for slot in range(shared):
                a, b = prev[slot], msgs[slot]
                if a["role"] != b["role"]:
                    mutations.append(
                        f"  slot{slot}: ROLE {a['role']}->{b['role']}"
                    )
                elif a["sha8"] != b["sha8"]:
                    kind = (
                        "SHRANK"
                        if b["chars"] < a["chars"]
                        else "GREW" if b["chars"] > a["chars"] else "CHANGED"
                    )
                    mutations.append(
                        f"  slot{slot} ({b['role']} tool_call_id={b['tool_call_id']}): "
                        f"{kind} {a['chars']}->{b['chars']} chars "
                        f"({a['sha8']}->{b['sha8']})"
                    )
            if len(msgs) < len(prev):
                mutations.append(
                    f"  TRUNCATED: {len(prev)} -> {len(msgs)} messages"
                )
        print(line)
        for m in mutations:
            print("  MUTATION" + m)
        prev = msgs
    if not rows:
        print("no chat requests captured yet")


if __name__ == "__main__":
    main()
