"""OpenCode 2.3 SQLite schema (`session_v2`, `session_message`, `session_part`) + legacy tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mtplx.commands.trace import (
    _latest_opencode_session,
    _opencode_connect,
    _opencode_messages,
    _opencode_parts,
    _opencode_session_row,
    _opencode_sessions,
    _resolve_session_arg,
)


def _make_legacy_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE session ("
        "id TEXT PRIMARY KEY, title TEXT, directory TEXT, time_created INTEGER, time_updated INTEGER,"
        " tokens_input INTEGER, tokens_output INTEGER, tokens_reasoning INTEGER, tokens_cache_read INTEGER)"
    )
    conn.execute(
        "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT)"
    )
    conn.execute(
        "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, time_created INTEGER, data TEXT)"
    )
    conn.execute(
        "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ses_legacy_1", "Legacy work", "/tmp/work", 1000, 2000, 100, 40, 20, 80),
    )
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?)",
        ("msg_u1", "ses_legacy_1", 1100, json.dumps({"role": "user"})),
    )
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?)",
        ("msg_a1", "ses_legacy_1", 1200, json.dumps({"role": "assistant", "time": {"created": 1200, "completed": 1500}})),
    )
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?)",
        ("prt_1", "msg_a1", 1250, json.dumps({"type": "reasoning", "text": "step 1"})),
    )
    conn.commit()
    conn.close()
    conn = _opencode_connect(path)
    assert conn is not None
    return conn


def _make_v2_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE session_v2 ("
        "id TEXT PRIMARY KEY, project_id TEXT, title TEXT, directory TEXT, time_created INTEGER,"
        " time_updated INTEGER, tokens_input INTEGER, tokens_output INTEGER,"
        " tokens_reasoning INTEGER, tokens_cache_read INTEGER)"
    )
    conn.execute(
        "CREATE TABLE session_message ("
        "id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER,"
        " time_created INTEGER, time_updated INTEGER, data TEXT)"
    )
    conn.execute(
        "CREATE TABLE session_part ("
        "message_id TEXT, session_id TEXT, ordinal INTEGER, type TEXT,"
        " call_id TEXT, time_created INTEGER, time_updated INTEGER, data TEXT,"
        " PRIMARY KEY (message_id, ordinal))"
    )
    conn.execute(
        "INSERT INTO session_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ses_v2_older", "proj_1", "Older task", "/tmp/a", 1000, 2000, 50, 20, 10, 0),
    )
    conn.execute(
        "INSERT INTO session_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ses_v2_latest_abc", "proj_1", "Latest OpenCode 2.3 task", "/tmp/b", 3000, 5000, 200, 90, 45, 150),
    )
    # Note: OpenCode 2.3 stores role in `type` column, NOT in `data["role"]`.
    conn.execute(
        "INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "msg_v2_u1",
            "ses_v2_latest_abc",
            "user",
            1,
            3100,
            3100,
            json.dumps({"text": "Please inspect trace.py", "time": {"created": 3100}}),
        ),
    )
    conn.execute(
        "INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "msg_v2_idle",
            "ses_v2_latest_abc",
            "idle",
            2,
            3150,
            3150,
            json.dumps({"outcome": "interrupted"}),
        ),
    )
    # Assistant message with inline parts (when session_part has 0 rows)
    conn.execute(
        "INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "msg_v2_a1_inline",
            "ses_v2_latest_abc",
            "assistant",
            3,
            3200,
            3600,
            json.dumps({
                "time": {"created": 3200, "completed": 3600},
                "content": [
                    {"type": "reasoning", "text": "checking tables"},
                    {"type": "text", "text": "Here is the schema."},
                ],
            }),
        ),
    )
    # Assistant message backed by session_part (no id column, ordered by ordinal)
    conn.execute(
        "INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "msg_v2_a2_parts",
            "ses_v2_latest_abc",
            "assistant",
            4,
            3700,
            4000,
            json.dumps({"time": {"created": 3700, "completed": 4000}}),
        ),
    )
    conn.execute(
        "INSERT INTO session_part VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("msg_v2_a2_parts", "ses_v2_latest_abc", 1, "text", None, 3800, 3800, json.dumps({"text": "Done."})),
    )
    conn.execute(
        "INSERT INTO session_part VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("msg_v2_a2_parts", "ses_v2_latest_abc", 0, "tool", "call_9", 3750, 3750, json.dumps({"tool": "read", "state": {"status": "ok"}})),
    )
    conn.commit()
    conn.close()
    conn = _opencode_connect(path)
    assert conn is not None
    return conn


def test_legacy_opencode_schema_continues_to_work(tmp_path):
    conn = _make_legacy_db(tmp_path / "legacy.db")
    try:
        sessions = _opencode_sessions(conn)
        assert [s["id"] for s in sessions] == ["ses_legacy_1"]
        assert _latest_opencode_session(conn) == "ses_legacy_1"
        assert _resolve_session_arg(conn, "legacy_1") == "ses_legacy_1"
        assert _opencode_session_row(conn, "ses_legacy_1")["title"] == "Legacy work"
        msgs = _opencode_messages(conn, "ses_legacy_1")
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        parts = _opencode_parts(conn, "msg_a1")
        assert len(parts) == 1 and parts[0]["_id"] == "prt_1" and parts[0]["text"] == "step 1"
    finally:
        conn.close()


def test_opencode_v2_schema_reads_sessions_messages_and_inline_or_table_parts(tmp_path):
    conn = _make_v2_db(tmp_path / "v2.db")
    try:
        sessions = _opencode_sessions(conn, limit=10)
        assert [s["id"] for s in sessions] == ["ses_v2_latest_abc", "ses_v2_older"]
        assert _latest_opencode_session(conn) == "ses_v2_latest_abc"
        assert _resolve_session_arg(conn, "latest") == "ses_v2_latest_abc"
        assert _resolve_session_arg(conn, "ses_v2_older") == "ses_v2_older"
        assert _resolve_session_arg(conn, "latest_abc") == "ses_v2_latest_abc"
        assert _opencode_session_row(conn, "ses_v2_latest_abc")["title"] == "Latest OpenCode 2.3 task"

        msgs = _opencode_messages(conn, "ses_v2_latest_abc")
        assert [m["role"] for m in msgs] == ["user", "idle", "assistant", "assistant"]
        assert msgs[0]["_parts"][0]["text"] == "Please inspect trace.py"

        # Inline content parts populate both _parts on message and _opencode_parts(conn, id)
        assert [p["type"] for p in msgs[2]["_parts"]] == ["reasoning", "text"]
        assert [p["text"] for p in _opencode_parts(conn, "msg_v2_a1_inline")] == [
            "checking tables",
            "Here is the schema.",
        ]

        # session_part table rows use ordinal order and composite _id (no id column)
        assert "_parts" not in msgs[3]
        parts2 = _opencode_parts(conn, "msg_v2_a2_parts")
        assert [p["type"] for p in parts2] == ["tool", "text"]
        assert parts2[0]["_id"] == "msg_v2_a2_parts:0"
        assert parts2[0]["callID"] == "call_9"
        assert parts2[1]["_id"] == "msg_v2_a2_parts:1"
    finally:
        conn.close()
