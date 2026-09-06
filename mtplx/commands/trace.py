"""MTPLX trace: first-class diagnosis tooling for agent/coding sessions.

Joins three local data sources into one timeline so a slow or misbehaving run
can be diagnosed in seconds instead of ad-hoc scripts:

  1. Serve request receipts   ~/.mtplx/logs/request-log-<port>.jsonl
  2. Flight-recorder events   ~/.mtplx/metrics/flight-<port>-<day>.jsonl
     (per-second samples: ev "s"; lifecycle: "begin"/"prefill"/"end"/"pc")
  3. OpenCode's database      ~/.local/share/opencode/opencode.db

Receipts carry the OpenCode session id (ses_...) via the session-headers
plugin, so modern joins are exact; older data falls back to time + token
cross-foot matching. Every view takes --json for machine consumption.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

LOGS_DIR = Path(os.path.expanduser("~/.mtplx/logs"))
METRICS_DIR = Path(os.path.expanduser("~/.mtplx/metrics"))
OPENCODE_DB = Path(os.path.expanduser("~/.local/share/opencode/opencode.db"))
AUTOPSY_DIR = METRICS_DIR / "autopsy"

_SPARK = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# formatting helpers


def _fmt_clock(ts: float | None) -> str:
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts, tz=_dt.UTC).astimezone().strftime("%m-%d %H:%M:%S")


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _fmt_tok(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _sparkline(values: list[float], width: int = 60) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    if len(vals) > width:
        # average into `width` buckets so long runs stay readable
        bucket = len(vals) / width
        vals = [
            sum(vals[int(i * bucket) : max(int(i * bucket) + 1, int((i + 1) * bucket))])
            / max(1, len(vals[int(i * bucket) : max(int(i * bucket) + 1, int((i + 1) * bucket))]))
            for i in range(width)
        ]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(_SPARK[min(7, int((v - lo) / span * 7.999))] for v in vals)


def _print_kv_block(title: str, pairs: list[tuple[str, Any]]) -> None:
    rows = [(k, v) for k, v in pairs if v is not None]
    if not rows:
        return
    print(f"  {title}")
    for key, val in rows:
        print(f"    {key:<34} {val}")


# ---------------------------------------------------------------------------
# data loading


def _detect_port(explicit: int | None) -> int | None:
    if explicit:
        return explicit
    candidates = sorted(
        LOGS_DIR.glob("request-log-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for path in candidates:
        match = re.match(r"request-log-(\d+)\.jsonl", path.name)
        if match:
            return int(match.group(1))
    return None


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def _load_receipts(port: int, since_s: float | None = None, *, path: str | None = None) -> list[dict]:
    records = _load_rotated(Path(path).expanduser() if path else LOGS_DIR / f"request-log-{port}.jsonl")
    if since_s is not None:
        records = [r for r in records if float(r.get("logged_at_s") or 0) >= since_s]
    return records


def _load_rotated(path: Path) -> list[dict]:
    generations = []
    for candidate in path.parent.glob(path.name + ".*"):
        suffix = candidate.name.rsplit(".", 1)[-1]
        if suffix.isdigit():
            generations.append((int(suffix), candidate))
    return [row for _, part in sorted(generations, reverse=True)
            for row in _load_jsonl(part)] + _load_jsonl(path)


def _source_port(args: argparse.Namespace) -> int | None:
    return (args.port or 0) if getattr(args, "request_log", None) else _detect_port(args.port)


def _load_flight(port: int, since_s: float | None = None, *, path: str | None = None) -> list[dict]:
    """Flight events oldest-first across the rotation cascade
    (flight-<port>.jsonl.N .. .1, then the live file)."""
    events = _load_rotated(Path(path).expanduser() if path else METRICS_DIR / f"flight-{port}.jsonl")
    if since_s is not None:
        events = [e for e in events if float(e.get("ts") or 0) >= since_s]
    return events


def _flight_by_rid(events: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for event in events:
        rid = event.get("rid")
        if rid:
            grouped.setdefault(rid, []).append(event)
    return grouped


def _opencode_connect(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    return conn


def _opencode_sessions(conn: sqlite3.Connection, limit: int = 15) -> list[dict]:
    rows = conn.execute(
        "SELECT id, title, directory, time_created, time_updated,"
        " tokens_input, tokens_output, tokens_reasoning, tokens_cache_read"
        " FROM session ORDER BY time_updated DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def _opencode_session_row(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def _latest_opencode_session(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT id FROM session ORDER BY time_updated DESC LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def _opencode_messages(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, time_created, data FROM message WHERE session_id = ?"
        " ORDER BY time_created ASC",
        (session_id,),
    ).fetchall()
    messages = []
    for row in rows:
        try:
            data = json.loads(row["data"])
        except (TypeError, json.JSONDecodeError):
            continue
        data["_id"] = row["id"]
        data["_time_created_s"] = (row["time_created"] or 0) / 1000.0
        messages.append(data)
    return messages


def _opencode_parts(conn: sqlite3.Connection, message_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, time_created, data FROM part WHERE message_id = ?"
        " ORDER BY time_created ASC",
        (message_id,),
    ).fetchall()
    parts = []
    for row in rows:
        try:
            data = json.loads(row["data"])
        except (TypeError, json.JSONDecodeError):
            continue
        data["_id"] = row["id"]
        parts.append(data)
    return parts


# ---------------------------------------------------------------------------
# joining

def _msg_tokens(message: dict) -> dict:
    tokens = message.get("tokens") or {}
    cache = tokens.get("cache") or {}
    return {
        "input": tokens.get("input") or 0,
        "output": tokens.get("output") or 0,
        "reasoning": tokens.get("reasoning") or 0,
        "cache_read": cache.get("read") or 0,
    }


def _match_receipt(message: dict, receipts: list[dict], used: set[int]) -> dict | None:
    """Best receipt for an assistant message: exact session ids narrow the pool,
    then nearest logged_at_s to the message completion, with a token cross-foot
    tiebreak (completion_tokens ~ output+reasoning) for historical fuzzy joins."""
    if message.get("_client") == "hermes":
        counts = message.get("_hermes_api_counts")
        if not counts:
            return None
        completed = message["time"]["completed"] / 1000
        created = message["time"].get("created", message["time"]["completed"]) / 1000
        matches = [(i, r) for i, r in enumerate(receipts) if i not in used
                   and [r.get("prompt_tokens"), r.get("completion_tokens")] == list(counts)
                   and created - 1 <= float(r.get("logged_at_s") or 0) <= completed + 3]
        if len(matches) != 1:
            return None
        used.add(matches[0][0])
        return matches[0][1]
    parent_entry = message.get("_parent_entry_id")
    if parent_entry:
        exact = [(i, r) for i, r in enumerate(receipts) if i not in used
                 and r.get("request_client_entry_id") == parent_entry]
        # Retries may share a leaf. Do not claim an exact join when ambiguous.
        if len(exact) == 1:
            used.add(exact[0][0])
            return exact[0][1]
    msg_time = message.get("time") or {}
    completed_s = (msg_time.get("completed") or 0) / 1000.0
    created_s = (msg_time.get("created") or 0) / 1000.0
    if not created_s:
        return None
    tokens = _msg_tokens(message)
    expected = tokens["output"] + tokens["reasoning"]
    best, best_score = None, None
    for idx, receipt in enumerate(receipts):
        if idx in used:
            continue
        turn = receipt.get("request_client_turn_id")
        if turn and message.get("parentID") and turn != message["parentID"]:
            continue
        logged = float(receipt.get("logged_at_s") or 0)
        anchor = completed_s or (created_s + float(receipt.get("request_elapsed_s") or 0))
        gap = abs(logged - anchor)
        if gap > 900:
            continue
        score = gap
        if expected:
            comp = int(receipt.get("completion_tokens") or 0)
            drift = abs(comp - expected)
            if drift <= 8:
                score -= 120  # strong cross-foot agreement dominates clock drift
            else:
                score += min(drift / 50.0, 120)
        if best_score is None or score < best_score:
            best, best_score = idx, score
    if best is None:
        return None
    used.add(best)
    return receipts[best]


def _join_session(
    conn: sqlite3.Connection | None, session_id: str, receipts: list[dict], flight: list[dict],
    *, session: dict | None = None, messages: list[dict] | None = None,
) -> dict:
    if session is None:
        session = _opencode_session_row(conn, session_id) or {"id": session_id}
    if messages is None:
        messages = _opencode_messages(conn, session_id)
    scoped = [r for r in receipts if r.get("session_id") == session_id]
    pool = scoped or receipts
    flight_rids = _flight_by_rid(flight)
    used: set[int] = set()
    turns = []
    turn_no = 0
    for message in messages:
        role = message.get("role")
        if role == "user":
            turns.append({"kind": "user", "message": message})
            continue
        if role != "assistant":
            continue
        turn_no += 1
        receipt = _match_receipt(message, pool, used)
        rid = (receipt or {}).get("request_id")
        turns.append(
            {
                "kind": "assistant",
                "turn": turn_no,
                "message": message,
                "receipt": receipt,
                "join": ("Hermes input/output tokens and completion clock" if receipt and message.get("_client") == "hermes"
                         else "exact Pi parent entry" if receipt and message.get("_parent_entry_id")
                         and receipt.get("request_client_entry_id") == message["_parent_entry_id"]
                         and sum(r.get("request_client_entry_id") == message["_parent_entry_id"] for r in pool) == 1
                         else "exact client turn; request by time/tokens" if receipt and receipt.get("request_client_turn_id")
                         else "session/time/tokens"),
                "flight": flight_rids.get(rid, []) if rid else [],
            }
        )
    modes = {t["join"] for t in turns if t["kind"] == "assistant" and t.get("receipt")}
    return {"session": session, "turns": turns, "receipt_pool_scoped": bool(scoped),
            "join_mode": next(iter(modes)) if len(modes) == 1 else "mixed joins; see per-request evidence",
            "unmatched_receipts": [r for i, r in enumerate(pool) if i not in used]}


def _load_joined(args: argparse.Namespace, receipts: list[dict], flight: list[dict]):
    hermes_db = getattr(args, "hermes_db", None)
    if hermes_db:
        from .trace_clients import load_hermes_session

        log = getattr(args, "hermes_log", None)
        session, messages = load_hermes_session(Path(hermes_db).expanduser(), args.session,
                                              Path(log).expanduser() if log else None)
        if messages:
            start = min(m["time"]["created"] for m in messages) / 1000 - 3
            end = max(m["time"]["completed"] for m in messages) / 1000 + 90
            receipts = [r for r in receipts if start <= float(r.get("logged_at_s") or 0) <= end]
        return None, _join_session(None, session["id"], receipts, flight,
                                   session=session, messages=messages)
    pi_path = getattr(args, "pi_session", None)
    if pi_path:
        from .trace_clients import load_pi_session

        session, messages = load_pi_session(Path(pi_path).expanduser())
        if not session.get("id"):
            raise ValueError("Pi transcript has no session identity")
        return None, _join_session(None, session["id"], receipts, flight,
                                   session=session, messages=messages)
    conn = _opencode_connect(Path(args.db))
    if conn is None:
        raise ValueError(f"OpenCode database not found: {args.db}")
    session_id = _resolve_session_arg(conn, args.session)
    if not session_id:
        raise ValueError("no OpenCode sessions found")
    return conn, _join_session(conn, session_id, receipts, flight)


# ---------------------------------------------------------------------------
# spiral / pathology detectors (encode the hard-won forensic heuristics)


def _detect_pathologies(turns: list[dict]) -> list[str]:
    flags: list[str] = []
    assistant = [t for t in turns if t["kind"] == "assistant" and t.get("receipt")]
    receipts = [t["receipt"] for t in assistant]
    if not receipts:
        return flags

    # committed-stream starvation: committed_len growth lagging far behind the
    # generated stream (the true postcommit-starvation mechanism — committed froze
    # at 15,389 while 66k+ was generated on the receipted 08-21 spiral)
    canon_seq = [
        (
            (r.get("committed_reasoning_canonicalization") or {}).get("committed_len"),
            int(r.get("completion_tokens") or 0),
        )
        for r in receipts
    ]
    canon_seq = [(c, g) for c, g in canon_seq if c is not None]
    if len(canon_seq) >= 3:
        committed_growth = canon_seq[-1][0] - canon_seq[-3][0]
        generated = sum(g for _, g in canon_seq[-3:-1])  # last turn's output can't be committed yet
        if generated > 4_000 and committed_growth < generated * 0.2:
            flags.append(
                f"COMMITTED STARVATION: committed_len grew {_fmt_tok(committed_growth)} while "
                f"{_fmt_tok(generated)} tokens were generated over the prior turns "
                f"(saves not landing -> history renders thin -> re-derivation risk)"
            )

    # postcommit waits: starvation (timeouts / not-stored) vs a mere latency tax
    pcw_seq = [(r.get("postcommit_wait") or {}) for r in receipts]
    waits = [float(p.get("elapsed_s") or 0) for p in pcw_seq]
    rising = [w for w in waits if w > 0]
    starved = any(
        p.get("outcome") == "timeout" or p.get("job_stored") is False for p in pcw_seq
    )
    if rising and starved and rising[-1] > 10:
        flags.append(
            "POSTCOMMIT STARVATION WAITS (timeout/not-stored present): "
            + " -> ".join(f"{w:.1f}s" for w in rising[-6:])
        )
    elif rising and max(rising) > 10:
        flags.append(
            f"postcommit wait tax: max {max(rising):.1f}s (all stored — TTFT tax, not starvation)"
        )

    # substitution shortfall: canon covered fewer turns than history carries.
    # Dangerous only when the server can't fall back to client-echoed reasoning
    # (pre-2009005 builds dropped the echo; echo presence shows as rhc > 0).
    for turn in assistant[-3:]:
        receipt = turn["receipt"]
        canon = receipt.get("committed_reasoning_canonicalization") or {}
        substituted = canon.get("turns_substituted")
        history_msgs = receipt.get("transcript_assistant_reasoning_history_messages")
        echo_chars = receipt.get("transcript_assistant_reasoning_history_chars") or 0
        if substituted is not None and history_msgs and substituted < history_msgs - 1:
            tail = (
                "client echo present — safe on >=2009005 (echo-carry), EMPTY renders on older builds"
                if echo_chars
                else "no client echo — uncovered turns render EMPTY (re-derivation risk)"
            )
            flags.append(
                f"reasoning coverage gap turn {turn['turn']}: canon substituted {substituted} of "
                f"{history_msgs} history turns ({tail})"
            )
            break

    # cap hits
    caps = sum(1 for r in receipts if r.get("server_cap_applied") or r.get("context_cap_applied"))
    if caps:
        flags.append(f"CAP APPLIED on {caps} turn(s)")

    # think explosion after a small-think turn (re-derivation signature)
    thinks = [(t["message"].get("tokens") or {}).get("reasoning") for t in assistant]
    for i in range(1, len(thinks)):
        if thinks[i] is None or thinks[i - 1] is None:
            continue
        if thinks[i] > 20_000 and thinks[i] > 5 * max(thinks[i - 1], 1):
            flags.append(
                f"THINK EXPLOSION turn {assistant[i]['turn']}: {_fmt_tok(thinks[i])} reasoning tokens "
                f"(prev {_fmt_tok(thinks[i - 1])}) — check reasoning coverage above"
            )
            break

    # New file/tool content legitimately requires prefill. Flag a reduction
    # in reused prefix separately; a large suffix alone is not a cache fault.
    walls = [
        (t["turn"], int(t["receipt"].get("new_prefill_tokens") or 0))
        for previous, t in zip(assistant, assistant[1:])
        if int(t["receipt"].get("new_prefill_tokens") or 0) > 1_000
        and t["receipt"].get("cached_tokens") is not None
        and int(t["receipt"]["cached_tokens"]) < int(previous["receipt"].get("prompt_tokens") or 0)
    ]
    if walls:
        flags.append(
            "REDUCED PREFIX REUSE (inspect history edits, compaction and cache receipts): "
            + ", ".join(f"t{n}={_fmt_tok(w)}" for n, w in walls[:8])
        )
    return flags


# ---------------------------------------------------------------------------
# subcommand: sessions


def _cmd_sessions(args: argparse.Namespace) -> int:
    conn = _opencode_connect(Path(args.db))
    if conn is None:
        print(f"opencode db not found: {args.db}", file=sys.stderr)
        return 1
    port = _source_port(args)
    receipts = _load_receipts(port, path=getattr(args, "request_log", None)) if port is not None else []
    sessions = _opencode_sessions(conn, limit=args.limit)
    by_session: dict[str, int] = {}
    for receipt in receipts:
        sid = receipt.get("session_id")
        if sid:
            by_session[sid] = by_session.get(sid, 0) + 1
    out = []
    for row in sessions:
        out.append(
            {
                "id": row["id"],
                "title": (row.get("title") or "")[:48],
                "directory": row.get("directory"),
                "updated": (row.get("time_updated") or 0) / 1000.0,
                "tokens_output": row.get("tokens_output"),
                "tokens_reasoning": row.get("tokens_reasoning"),
                "server_requests_matched": by_session.get(row["id"], 0),
            }
        )
    if args.json:
        print(json.dumps({"port": port, "sessions": out}, indent=2))
        return 0
    print(f"OpenCode sessions (newest first; server receipts matched on port {port})")
    print(f"{'session':<30} {'updated':<15} {'reqs':>5} {'out tok':>9} {'think tok':>10}  title")
    for row in out:
        print(
            f"{row['id']:<30} {_fmt_clock(row['updated']):<15} {row['server_requests_matched']:>5}"
            f" {_fmt_tok(row['tokens_output']):>9} {_fmt_tok(row['tokens_reasoning']):>10}  {row['title']}"
        )
    return 0


# ---------------------------------------------------------------------------
# subcommand: session


def _resolve_session_arg(conn: sqlite3.Connection, ident: str | None) -> str | None:
    if not ident or ident == "latest":
        return _latest_opencode_session(conn)
    exact = conn.execute("SELECT id FROM session WHERE id = ?", (ident,)).fetchone()
    if exact:
        return exact["id"]
    row = conn.execute(
        "SELECT id FROM session WHERE id LIKE ? ORDER BY time_updated DESC LIMIT 1",
        (f"%{ident}%",),
    ).fetchone()
    return row["id"] if row else ident


def _turn_row(turn: dict) -> dict:
    message = turn["message"]
    receipt = turn.get("receipt") or {}
    tokens = _msg_tokens(message)
    msg_time = message.get("time") or {}
    created_s = (msg_time.get("created") or 0) / 1000.0
    completed_s = (msg_time.get("completed") or 0) / 1000.0
    pcw = receipt.get("postcommit_wait") or {}
    canon = receipt.get("committed_reasoning_canonicalization") or {}
    samples = [e for e in turn.get("flight", []) if e.get("ev") == "s"]
    status = "ok"
    if message.get("error") or message.get("stopReason") in {"error", "aborted"}:
        status = "CANCEL/ERR"
    elif not receipt:
        status = "missing receipt" if completed_s else "in progress"
    return {
        "turn": turn["turn"],
        "start": created_s,
        "wall_s": (completed_s - created_s) if completed_s and created_s else None,
        "status": status,
        "join": turn.get("join"),
        "tool_errors": sum(p.get("state", {}).get("status") == "error"
                           for p in message.get("_parts", []) if p.get("type") == "tool"),
        "prompt_tokens": receipt.get("prompt_tokens"),
        "cached_tokens": receipt.get("cached_tokens"),
        "new_prefill_tokens": receipt.get("new_prefill_tokens"),
        "cache_source": receipt.get("cache_source"),
        "cache_miss_reason": receipt.get("cache_miss_reason"),
        "completion_tokens": receipt.get("completion_tokens"),
        "client_reasoning_tokens": None if message.get("_client") in {"pi", "hermes"} else tokens["reasoning"],
        "client_output_tokens": tokens["output"],
        "client_cache_read": tokens["cache_read"],
        "decode_tok_s": receipt.get("decode_tok_s"),
        "ttft_s": receipt.get("ttft_s"),
        "effort": receipt.get("reasoning_effort") or receipt.get("effective_reasoning_effort"),
        "postcommit_wait": {k: pcw.get(k) for k in ("outcome", "elapsed_s", "job_stored", "job_mode", "job_reason") if k in pcw} or None,
        "canon": {k: canon.get(k) for k in ("applied", "cp_raw", "cp_canon", "committed_len", "turns_substituted") if k in canon} or None,
        "request_id": receipt.get("request_id"),
        "tps_sparkline": _sparkline([float(s.get("tps") or 0) for s in samples], width=24) or None,
        "receipt_missing": not receipt,
    }


def _cmd_session(args: argparse.Namespace) -> int:
    port = _source_port(args)
    if port is None:
        print("no request logs found under ~/.mtplx/logs", file=sys.stderr)
        return 1
    receipts = _load_receipts(port, path=getattr(args, "request_log", None))
    flight = _load_flight(port, path=getattr(args, "flight_log", None))
    try:
        _conn, joined = _load_joined(args, receipts, flight)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    session_id = joined["session"]["id"]
    rows = [_turn_row(t) for t in joined["turns"] if t["kind"] == "assistant"]
    flags = _detect_pathologies(joined["turns"])

    warm = [r for r in rows[1:] if r["prompt_tokens"] and r["cached_tokens"] is not None]
    reuse = (
        sum(r["cached_tokens"] or 0 for r in warm) / max(1, sum(r["prompt_tokens"] or 0 for r in warm))
        if warm
        else None
    )
    summary = {
        "session_id": session_id,
        "title": joined["session"].get("title"),
        "directory": joined["session"].get("directory"),
        "port": port,
        "turns": len(rows),
        "join_mode": joined["join_mode"],
        "warm_cache_reuse": round(reuse, 4) if reuse is not None else None,
        "total_completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
        "total_client_reasoning_tokens": (None if any(r["client_reasoning_tokens"] is None for r in rows)
                                          else sum(r["client_reasoning_tokens"] for r in rows)),
        "pathologies": flags,
    }
    if args.json:
        print(json.dumps({"summary": summary, "turns": rows}, indent=2))
        return 0

    print(f"session {session_id}  ({summary['title'] or 'untitled'})")
    print(f"  dir={summary['directory']}  port={port}  join={summary['join_mode']}")
    reuse_str = f"{reuse * 100:.1f}%" if reuse is not None else "-"
    print(
        f"  turns={summary['turns']}  warm-reuse={reuse_str}"
        f"  completion={_fmt_tok(summary['total_completion_tokens'])}"
        f"  client-think={_fmt_tok(summary['total_client_reasoning_tokens'])}"
    )
    print()
    header = (
        f"{'t':>3} {'start':<15} {'wall':>8} {'st':<10} {'prompt':>8} {'cached':>8} {'+pre':>7}"
        f" {'comp':>7} {'think':>7} {'tok/s':>6} {'ttft':>6} {'pcw':<22} {'canon':<20} tps"
    )
    print(header)
    for row in rows:
        pcw = row["postcommit_wait"] or {}
        pcw_str = "-"
        if pcw:
            stored = pcw.get("job_stored")
            pcw_str = f"{pcw.get('outcome','?')}/{pcw.get('elapsed_s',0):.1f}s"
            if stored is not None:
                pcw_str += f"/{'stored' if stored else 'NOT-stored'}"
        canon = row["canon"] or {}
        canon_str = "-"
        if canon:
            canon_str = (
                f"{'A' if canon.get('applied') else '.'}"
                f" cl={_fmt_tok(canon.get('committed_len'))} sub={canon.get('turns_substituted', '-')}"
            )
        print(
            f"{row['turn']:>3} {_fmt_clock(row['start']):<15} {_fmt_dur(row['wall_s']):>8} {row['status']:<10}"
            f" {_fmt_tok(row['prompt_tokens']):>8} {_fmt_tok(row['cached_tokens']):>8} {_fmt_tok(row['new_prefill_tokens']):>7}"
            f" {_fmt_tok(row['completion_tokens']):>7} {_fmt_tok(row['client_reasoning_tokens']):>7}"
            f" {row['decode_tok_s'] or 0:>6.1f} {row['ttft_s'] or 0:>6.2f} {pcw_str:<22} {canon_str:<20} {row['tps_sparkline'] or ''}"
        )
    if flags:
        print("\n  PATHOLOGY FLAGS")
        for flag in flags:
            print(f"    !! {flag}")
    missing = [r["turn"] for r in rows if r["receipt_missing"]]
    if missing:
        print(f"\n  turns with no matched server receipt: {missing}")
    return 0


# ---------------------------------------------------------------------------
# subcommand: request


_RECEIPT_GROUPS: list[tuple[str, list[str]]] = [
    ("identity", ["request_id", "session_id", "logged_at_s", "generation_mode", "warmup"]),
    ("tokens", ["prompt_tokens", "cached_tokens", "new_prefill_tokens", "completion_tokens",
                "context_len", "remaining_context_tokens", "bonus_tokens", "correction_tokens"]),
    ("speed", ["ttft_s", "prefill_tok_s", "prefill_wall_tok_s", "prefill_compute_tok_s",
               "decode_tok_s", "display_decode_tok_s", "request_tok_s", "request_elapsed_s",
               "decode_elapsed_s", "sliding_decode_tok_s_first_64", "sliding_decode_tok_s_last_64",
               "sliding_decode_tok_s_last_256", "producer_gap_ms_p95", "producer_gap_ms_max"]),
    ("mtp", ["mtp_depth", "requested_mtp_depth", "accepted_by_depth", "drafted_by_depth",
             "mean_accept_probability_by_depth", "verify_calls", "draft_time_s", "verify_time_s",
             "accept_time_s", "target_forward_time_s", "verify_forward_time_s"]),
    ("sampling", ["effective_temperature", "effective_top_p", "effective_top_k",
                  "draft_sampler_policy", "draft_sampler_policy_source",
                  "draft_sampler_resolved_temperature",
                  # Adaptive-dtemp trajectory (MTPLX_ADAPTIVE_DTEMP): absent
                  # unless the gate is on; carries current temp, EMA,
                  # transitions, and the transition log.
                  "draft_sampler_adaptive_dtemp"]),
    ("caps", ["request_max_tokens", "effective_max_tokens", "server_max_response_tokens",
              "server_cap_applied", "context_cap_applied", "uncapped_response_requested",
              "uncapped_response_lease_applied", "uncapped_repetition_stop_enabled"]),
    ("cache/session", ["cache_source", "cache_miss_reason", "session_cache_hit",
                       "session_restore_mode", "session_restore_served", "session_prefill_store",
                       "session_prompt_prefix_bank_commit", "stable_prefix_len",
                       "cache_restore_time_s", "ssd_cache_hit", "ssd_cached_tokens", "ssd_restore_s"]),
    ("reasoning", ["committed_reasoning_canonicalization", "postcommit_wait",
                   "transcript_assistant_reasoning_history_chars",
                   "transcript_assistant_reasoning_history_messages",
                   "live_frontier_extended", "live_frontier_hit"]),
    ("guards", ["repetition_stop_triggered", "repetition_stop_reason", "loop_guard", "thinking_guard"]),
    ("memory", ["peak_memory_bytes", "active_memory_bytes", "cache_memory_bytes"]),
]


def _cmd_request(args: argparse.Namespace) -> int:
    port = _source_port(args)
    if port is None:
        print("no request logs found under ~/.mtplx/logs", file=sys.stderr)
        return 1
    receipts = _load_receipts(port, path=getattr(args, "request_log", None))
    if not receipts:
        print(f"no receipts for port {port}", file=sys.stderr)
        return 1
    receipt = None
    if not args.request or args.request == "latest":
        receipt = receipts[-1]
    else:
        for candidate in reversed(receipts):
            if candidate.get("request_id") and args.request in str(candidate["request_id"]):
                receipt = candidate
                break
        if receipt is None and args.request.isdigit():
            idx = int(args.request)
            if 0 <= idx < len(receipts):
                receipt = receipts[idx]
    if receipt is None:
        print(f"request {args.request!r} not found in receipts for port {port}", file=sys.stderr)
        return 1

    rid = receipt.get("request_id")
    flight = _flight_by_rid(_load_flight(port, path=getattr(args, "flight_log", None))).get(rid, []) if rid else []
    samples = [e for e in flight if e.get("ev") == "s"]
    if args.json:
        from .trace_metrics import mtp_economics, sample_intervals
        print(json.dumps({"receipt": receipt, "flight": flight,
                          "economics": mtp_economics(receipt, getattr(args, "ar_tok_s", None)),
                          "intervals": sample_intervals(flight)}, indent=2))
        return 0

    index = receipts.index(receipt)
    print(
        f"request #{index} on port {port}  rid={rid or '-'}  at {_fmt_clock(receipt.get('logged_at_s'))}"
    )
    shown: set[str] = set()
    for title, keys in _RECEIPT_GROUPS:
        pairs = []
        for key in keys:
            if key in receipt:
                shown.add(key)
                val = receipt[key]
                if isinstance(val, dict):
                    val = json.dumps(val, separators=(",", ":"))
                pairs.append((key, val))
        _print_kv_block(title, pairs)
    rest = sorted(k for k in receipt if k not in shown)
    if rest and args.all:
        _print_kv_block("other", [(k, receipt[k]) for k in rest])
    elif rest:
        print(f"  ({len(rest)} more fields; --all to show)")
    if samples:
        tps = [float(s.get("tps") or 0) for s in samples]
        print("\n  per-second decode (flight recorder)")
        print(f"    tps  min={min(tps):.1f} mean={sum(tps)/len(tps):.1f} max={max(tps):.1f}  n={len(tps)}s")
        print(f"    {_sparkline(tps, width=80)}")
        rc = [int(s.get("rc") or 0) for s in samples]
        if rc and rc[-1]:
            print(f"    reasoning chars {_fmt_tok(rc[-1])} / content chars {_fmt_tok(int(samples[-1].get('cc') or 0))}")
    else:
        print("\n  (no flight samples for this request — recorder not active or pre-recorder data)")
    return 0


# ---------------------------------------------------------------------------
# subcommand: autopsy


def _ngram_repeat_mass(words: list[str], n: int = 8) -> float:
    if len(words) < n * 2:
        return 0.0
    shingles = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(shingles)) / len(shingles))


def _cmd_autopsy(args: argparse.Namespace) -> int:
    conn = _opencode_connect(Path(args.db))
    if conn is None:
        print(f"opencode db not found: {args.db}", file=sys.stderr)
        return 1
    session_id = _resolve_session_arg(conn, args.session)
    if not session_id:
        print("no opencode sessions found", file=sys.stderr)
        return 1
    messages = [m for m in _opencode_messages(conn, session_id) if m.get("role") == "assistant"]
    if not messages:
        print(f"no assistant messages in {session_id}", file=sys.stderr)
        return 1
    if args.turn is not None:
        if not 1 <= args.turn <= len(messages):
            print(f"turn out of range 1..{len(messages)}", file=sys.stderr)
            return 1
        targets = [(args.turn, messages[args.turn - 1])]
    else:
        # Default: the biggest think by ACTUAL part text length. Client token
        # accounting is zeroed on cancelled turns — exactly the marathons this
        # command exists for — so ranking by message.tokens would skip them.
        sized = []
        for i, message in enumerate(messages):
            parts = _opencode_parts(conn, message["_id"])
            chars = sum(
                len(p.get("text") or "") for p in parts if p.get("type") == "reasoning"
            )
            sized.append((i + 1, message, chars))
        turn_no, message, _ = max(sized, key=lambda x: x[2])
        targets = [(turn_no, message)]

    results = []
    for turn_no, message in targets:
        parts = _opencode_parts(conn, message["_id"])
        reasoning = "\n\n".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning")
        text = "\n\n".join(p.get("text") or "" for p in parts if p.get("type") == "text")
        words = reasoning.split()
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in reasoning.split("\n\n")]
        paragraphs = [p for p in paragraphs if len(p) > 60]
        counts: dict[str, int] = {}
        for para in paragraphs:
            counts[para] = counts.get(para, 0) + 1
        dupes = sorted(
            ((c, p) for p, c in counts.items() if c > 1), key=lambda x: -x[0]
        )[:5]
        mass = _ngram_repeat_mass(words)
        msg_time = message.get("time") or {}
        wall_s = ((msg_time.get("completed") or 0) - (msg_time.get("created") or 0)) / 1000.0
        verdict = (
            "LOOP" if mass >= 0.40 else "mixed/suspicious" if mass >= 0.15 else "legitimate derivation"
        )
        AUTOPSY_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = AUTOPSY_DIR / f"{session_id}-t{turn_no}.txt"
        dump_path.write_text(reasoning + "\n\n===== VISIBLE OUTPUT =====\n\n" + text, encoding="utf-8")
        results.append(
            {
                "turn": turn_no,
                "reasoning_chars": len(reasoning),
                "reasoning_words": len(words),
                "visible_chars": len(text),
                "wall_s": wall_s if wall_s > 0 else None,
                "eight_gram_repeat_mass": round(mass, 4),
                "duplicated_paragraphs": [{"count": c, "head": p[:110]} for c, p in dupes],
                "verdict": verdict,
                "dump_path": str(dump_path),
                "error": bool(message.get("error")),
            }
        )
    if args.json:
        print(json.dumps({"session_id": session_id, "results": results}, indent=2))
        return 0
    for res in results:
        print(f"autopsy {session_id} turn {res['turn']}  ({'CANCELLED/ERROR' if res['error'] else 'completed'})")
        print(
            f"  reasoning {_fmt_tok(res['reasoning_chars'])} chars / {_fmt_tok(res['reasoning_words'])} words"
            f"  visible {_fmt_tok(res['visible_chars'])} chars  wall {_fmt_dur(res['wall_s'])}"
        )
        print(f"  8-gram repeat mass: {res['eight_gram_repeat_mass'] * 100:.1f}%   verdict: {res['verdict']}")
        if res["duplicated_paragraphs"]:
            print("  repeated paragraphs:")
            for dup in res["duplicated_paragraphs"]:
                print(f"    x{dup['count']}  {dup['head']}")
        print(f"  full text -> {res['dump_path']}")
    return 0


# ---------------------------------------------------------------------------
# subcommand: live


def _cmd_live(args: argparse.Namespace) -> int:
    port = _detect_port(args.port) or 8002
    url = f"http://127.0.0.1:{port}/v1/mtplx/flight"

    def fetch() -> dict | None:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — any failure means "not reachable"
            print(f"({url} not reachable: {exc})", file=sys.stderr)
            return None

    while True:
        snapshot = fetch()
        if snapshot is None:
            return 1
        if args.json:
            print(json.dumps(snapshot, indent=2))
        else:
            active = snapshot.get("active") or []
            if not active:
                print(f"{_fmt_clock(time.time())}  idle — no request in flight")
            for req in active:
                tail = (req.get("tail") or "").replace("\n", " ")[-160:]
                print(
                    f"{_fmt_clock(time.time())}  rid={req.get('rid')}  session={req.get('session_id')}"
                    f"  phase={req.get('phase')}  {_fmt_dur(req.get('elapsed_s'))}"
                    f"  gen={_fmt_tok(req.get('gen_tokens'))}  tps={req.get('tps_now', 0):.1f}"
                    f"  think={_fmt_tok(req.get('reasoning_chars'))}c"
                )
                if tail:
                    print(f"    ...{tail}")
        if not args.watch:
            return 0
        time.sleep(args.interval)


# ---------------------------------------------------------------------------
# dispatcher


def cmd_trace(args: argparse.Namespace) -> int:
    action = getattr(args, "trace_action", None)
    handlers = {
        "sessions": _cmd_sessions,
        "session": _cmd_session,
        "request": _cmd_request,
        "autopsy": _cmd_autopsy,
        "live": _cmd_live,
    }
    if action == "report":
        from .trace_report import cmd_trace_report

        return cmd_trace_report(args)
    handler = handlers.get(action)
    if handler is None:
        print(f"unknown trace action: {action}", file=sys.stderr)
        return 2
    return handler(args)
