"""CPU tests for OpenAI server snapshot & request telemetry truthfulness.

Covers:
1. ``/v1/mtplx/snapshot`` ``profile`` reflecting the actually loaded model,
   family, live default sampler (including ``presence_penalty``), and
   ``resolved_reasoning_effort`` instead of stale profile defaults.
2. Disambiguated ``first_visible_s`` / ``first_visible_kind`` /
   ``first_answer_s`` and per-request ``resolved_reasoning_effort`` in both
   JSONL request logs and ``/v1/mtplx/snapshot`` (``latest`` & ``recent``).
3. Verify/draft timing pass-through (``verify_time_s``,
   ``verify_forward_time_s``, ``draft_time_s``, ``verify_calls``) from engine
   generation stats into JSONL logs, ``/v1/mtplx/snapshot``, and public
   ``mtplx_stats``.
4. Non-zero session bytes and hit counts in ``/v1/mtplx/snapshot``,
   ``/admin/sessions``, and ``_store_retokenized_history_snapshot`` for live
   leases and oversized-skip commits.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.backends.descriptors import INKLING_DESCRIPTOR
from mtplx.server import openai, request_spans
from mtplx.server.openai import create_app

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_server_openai import _fake_state, _fake_streaming_session_state  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_registry():
    request_spans.reset_for_tests()
    yield
    request_spans.reset_for_tests()


def _inkling_state():
    state = _fake_state()
    state.model_id = "Inkling-2-Mini-MTPLX-Optimized-Quality"
    state.args.model = "/Users/example/Models/Inkling-2-Mini-MTPLX-Optimized-Quality"
    state.args.temperature = 1.0
    state.args.top_p = 1.0
    state.args.top_k = 0
    state.args.default_presence_penalty = 0.25
    state.args.enable_thinking = True
    state.args.reasoning_effort = "auto"
    state.runtime.backend_descriptor = INKLING_DESCRIPTOR
    # Simulate stale built-in profile defaults that previously leaked into /v1/mtplx/snapshot.
    state.profile = SimpleNamespace(
        name="balanced",
        to_dict=lambda: {
            "name": "balanced",
            "model_id": "Qwen3.8-27B-MTPLX-Balanced",
            "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        },
    )
    return state


def test_snapshot_profile_and_resolved_reasoning_effort_reflect_live_model():
    state = _inkling_state()
    client = TestClient(create_app(state))

    snap = client.get("/v1/mtplx/snapshot").json()
    profile = snap["profile"]
    assert profile["model_id"] == "Inkling-2-Mini-MTPLX-Optimized-Quality"
    assert profile["model_name"] == "Inkling-2-Mini-MTPLX-Optimized-Quality"
    assert profile["model_family"] == "inkling2"
    assert profile["profile_default_model_id"] == "Qwen3.8-27B-MTPLX-Balanced"
    assert profile["sampler"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.25,
    }
    assert profile["profile_default_sampler"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    # Inkling's descriptor declares default_effort="high" when server effort is "auto".
    assert profile["resolved_reasoning_effort"] == "high"
    assert snap["resolved_reasoning_effort"] == "high"
    assert snap["settings"]["resolved_reasoning_effort"] == "high"

    # Explicit effort override resolves cleanly.
    state.args.reasoning_effort = "xhigh"
    snap_xhigh = client.get("/v1/mtplx/snapshot").json()
    assert snap_xhigh["profile"]["resolved_reasoning_effort"] == "xhigh"
    assert snap_xhigh["resolved_reasoning_effort"] == "xhigh"
    assert snap_xhigh["settings"]["resolved_reasoning_effort"] == "xhigh"

    # Thinking off resolves to None.
    state.args.enable_thinking = False
    snap_off = client.get("/v1/mtplx/snapshot").json()
    assert snap_off["profile"]["resolved_reasoning_effort"] is None
    assert snap_off["resolved_reasoning_effort"] is None
    assert snap_off["settings"]["resolved_reasoning_effort"] is None


def _run_stream_turn(
    tmp_path: Path,
    monkeypatch,
    text: str,
    body: dict,
    *,
    extra_stats: dict | None = None,
    configure_state=None,
    finish_reason: str = "stop",
) -> tuple[dict, dict, dict]:
    state = _fake_streaming_session_state()
    state.model_id = "Inkling-2-Mini-MTPLX-Optimized-Quality"
    state.args.model = "/Users/example/Models/Inkling-2-Mini-MTPLX-Optimized-Quality"
    state.draft_sampler = None
    state.requests_completed = 0
    state.args.stream_interval = 1
    state.args.enable_thinking = True
    state.args.reasoning_effort = "auto"
    state.runtime.backend_descriptor = INKLING_DESCRIPTOR
    if configure_state is not None:
        configure_state(state)
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "requests.jsonl"
    state.args.request_log_jsonl = str(log)

    def fake_generate_mtpk(_runtime, _prompt_ids, **kwargs):
        tokens = [ord(ch) for ch in text]
        callback = kwargs.get("token_callback")
        for tok in tokens:
            time.sleep(0.002)
            if callback is not None:
                callback([tok])
        stats_payload = {
            "mode": "mtp",
            "prompt_eval_time_s": 0.001,
            "generated_tokens": len(tokens),
            "elapsed_s": 0.002 * max(1, len(tokens)),
            "tok_s": 500.0,
            **(extra_stats or {}),
        }
        return SimpleNamespace(
            tokens=tokens,
            text=text,
            stats=SimpleNamespace(to_dict=lambda: dict(stats_payload)),
            final_state=None,
            finish_reason=finish_reason,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"max_tokens": 256, **body, "stream": True},
        ) as response:
            assert response.status_code == 200
            _ = response.read()
        snap = client.get("/v1/mtplx/snapshot").json()
        rows = [json.loads(line) for line in log.read_text().splitlines()]

    assert len(rows) == 1
    row = rows[0]
    recent = snap["recent"]
    assert recent and recent[-1]["request_id"] == row["request_id"]
    return row, recent[-1], snap


def test_first_visible_kind_and_answer_s_distinguish_text_tool_call_and_reasoning_only(
    tmp_path, monkeypatch
):
    # 1) Thinking followed by user-visible text answer.
    row_text, snap_text, _ = _run_stream_turn(
        tmp_path / "text",
        monkeypatch,
        "<think>checking math</think>\nThe answer is 42.",
        {"messages": [{"role": "user", "content": "answer?"}], "enable_thinking": True},
    )
    for rec in (row_text, snap_text):
        assert rec["first_visible_kind"] == "text"
        assert rec["first_visible_s"] is not None and rec["first_visible_s"] > 0
        assert rec["first_answer_s"] == rec["first_visible_s"]
        assert rec["resolved_reasoning_effort"] == "high"

    # 2) Thinking followed by a tool call (no user-visible text answer).
    call_text = (
        "<think>need to inspect files</think>"
        "<tool_call>\n<function=list_files>\n<parameter=path>\n.\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    tool_body = {
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List directory entries.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    row_tool, snap_tool, _ = _run_stream_turn(
        tmp_path / "tool", monkeypatch, call_text, tool_body
    )
    for rec in (row_tool, snap_tool):
        assert rec["first_visible_kind"] == "tool_call"
        assert rec["first_visible_s"] is not None and rec["first_visible_s"] > 0
        assert rec["first_answer_s"] is None
        assert rec["resolved_reasoning_effort"] == "xhigh"

    # 3) Reasoning-only turn (cut by max_tokens while still thinking: no answer or tool call).
    truncated = "<think>still thinking when budget ran out"
    row_think_only, snap_think_only, _ = _run_stream_turn(
        tmp_path / "think_only",
        monkeypatch,
        truncated,
        {
            "messages": [{"role": "user", "content": "hard problem"}],
            "enable_thinking": True,
            "max_tokens": len(truncated),
        },
        finish_reason="length",
    )
    for rec in (row_think_only, snap_think_only):
        assert rec["reasoning_s"] is not None and rec["reasoning_s"] > 0
        assert rec["first_visible_s"] is None
        assert rec["first_visible_kind"] is None
        assert rec["first_answer_s"] is None

    # 3b) Direct _StreamVisibility check: tool_call first, then text later.
    vis = openai._StreamVisibility(100.0)
    vis.saw("reasoning", 100.2)
    vis.saw("tool_call", 101.5)
    vis.saw("answer", 103.0)
    fields = vis.fields(12)
    assert fields["reasoning_s"] == pytest.approx(0.2)
    assert fields["first_visible_s"] == pytest.approx(1.5)
    assert fields["first_visible_kind"] == "tool_call"
    assert fields["first_answer_s"] == pytest.approx(3.0)

    # 4) Thinking disabled -> resolved_reasoning_effort is absent/None.
    row_no_think, snap_no_think, _ = _run_stream_turn(
        tmp_path / "no_think",
        monkeypatch,
        "Direct reply.",
        {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False},
    )
    for rec in (row_no_think, snap_no_think):
        assert rec["first_visible_kind"] == "text"
        assert rec.get("resolved_reasoning_effort") is None


def test_verify_and_draft_timing_fields_pass_through_to_logs_snapshot_and_stats(
    tmp_path, monkeypatch
):
    timing = {
        "verify_time_s": 0.412,
        "verify_forward_time_s": 0.318,
        "draft_time_s": 0.074,
        "verify_calls": 19,
        "drafted_tokens": 57,
        "accepted_drafts": 41,
        "rejected_drafts": 16,
    }
    # Streaming path -> JSONL + /v1/mtplx/snapshot (`latest` & `recent`).
    row, recent_row, snap = _run_stream_turn(
        tmp_path / "stream_timing",
        monkeypatch,
        "Verified output.",
        {"messages": [{"role": "user", "content": "run"}], "enable_thinking": False},
        extra_stats=timing,
    )
    for rec in (row, recent_row, snap["latest"]):
        assert rec["verify_time_s"] == pytest.approx(0.412)
        assert rec["verify_forward_time_s"] == pytest.approx(0.318)
        assert rec["draft_time_s"] == pytest.approx(0.074)
        assert rec["verify_calls"] == 19

    # Non-streaming path -> JSONL + /v1/mtplx/snapshot + public `mtplx_stats`.
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    state.runtime.backend_descriptor = INKLING_DESCRIPTOR
    log = tmp_path / "nonstream.jsonl"
    state.args.request_log_jsonl = str(log)

    def fake_generate_mtpk(_runtime, _prompt_ids, **_kwargs):
        text = "Non-stream reply."
        return SimpleNamespace(
            tokens=[ord(c) for c in text],
            text=text,
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "mode": "mtp",
                    "prompt_eval_time_s": 0.002,
                    "generated_tokens": len(text),
                    "elapsed_s": 0.50,
                    "tok_s": 34.0,
                    **timing,
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    with TestClient(create_app(state)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "enable_thinking": False,
                "include_mtplx_stats": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        snap_ns = client.get("/v1/mtplx/snapshot").json()
        ns_rows = [json.loads(line) for line in log.read_text().splitlines()]

    assert len(ns_rows) == 1
    for rec in (ns_rows[0], snap_ns["latest"], snap_ns["recent"][-1], body["mtplx_stats"]):
        assert rec["verify_time_s"] == pytest.approx(0.412)
        assert rec["verify_forward_time_s"] == pytest.approx(0.318)
        assert rec["draft_time_s"] == pytest.approx(0.074)
        assert rec["verify_calls"] == 19
    assert body["mtplx_stats"]["first_visible_kind"] == "text"
    assert snap_ns["latest"]["first_visible_kind"] == "text"


def test_snapshot_and_admin_sessions_backfill_live_lease_bytes_and_hits():
    state = _inkling_state()
    lease_bytes = 512 * (1 << 20)
    state.sessions = SimpleNamespace(
        bank=SimpleNamespace(
            to_dict=lambda: {
                "entries": 1,
                "total_nbytes": lease_bytes,
                "held_nbytes": lease_bytes,
                "hits": 7,
                "cumulative_hits": 7,
                "ram_hits": 7,
                "ssd_hits": 0,
                "lease_handoffs": 6,
                "prefixes": [
                    {
                        "session_id": "sess-live-1",
                        "prefix_len": 115186,
                        "hits": 7,
                        "cumulative_hits": 7,
                        "nbytes": lease_bytes,
                        "held_nbytes": lease_bytes,
                        "snapshot_nbytes": 0,
                        "created_at_s": 100.0,
                        "last_access_s": 200.0,
                        "has_live_ref": True,
                    }
                ],
                "eviction_log": [],
                "recent_evictions": [],
            }
        ),
        list_sessions=lambda: {
            "count": 1,
            "sessions": [
                {
                    "session_id": "sess-live-1",
                    "prefix_len": 115186,
                    "bytes": 0,  # Previously stayed 0 on one-copy live leases.
                    "in_flight": False,
                    "last_access_s": 200.0,
                }
            ],
            "session_bank": {
                "entries": 1,
                "total_nbytes": lease_bytes,
                "held_nbytes": lease_bytes,
                "hits": 7,
                "cumulative_hits": 7,
                "ram_hits": 7,
                "ssd_hits": 0,
                "lease_handoffs": 6,
                "prefixes": [
                    {
                        "session_id": "sess-live-1",
                        "prefix_len": 115186,
                        "hits": 7,
                        "cumulative_hits": 7,
                        "nbytes": lease_bytes,
                        "held_nbytes": lease_bytes,
                        "snapshot_nbytes": 0,
                        "created_at_s": 100.0,
                        "last_access_s": 200.0,
                        "has_live_ref": True,
                    }
                ],
                "eviction_log": [],
                "recent_evictions": [],
            },
        },
    )

    client = TestClient(create_app(state))
    snap = client.get("/v1/mtplx/snapshot").json()
    admin = client.get("/admin/sessions").json()

    for payload in (snap["sessions"], admin):
        row = payload["sessions"][0]
        assert row["bytes"] == lease_bytes
        assert row["held_nbytes"] == lease_bytes
        assert row["hits"] == 7
        assert row["cumulative_hits"] == 7

    bank = snap["session_bank"]
    assert bank["total_nbytes"] == lease_bytes
    assert bank["held_nbytes"] == lease_bytes
    assert bank["hits"] == 7
    assert bank["cumulative_hits"] == 7
    assert bank["prefixes"][0]["nbytes"] == lease_bytes
    assert bank["prefixes"][0]["hits"] == 7
    assert bank["eviction_log"] == []


def test_retokenized_oversized_skip_preserves_non_zero_session_nbytes(monkeypatch):
    recorded_nbytes: list[int] = []

    class FakeSession:
        session_id = "sess-oversized"
        prefix_len = 100
        bytes_estimate = 400_000_000

        def commit_retokenized_prefix(self, *, token_ids, expected_revision, nbytes):
            recorded_nbytes.append(int(nbytes))
            self.prefix_len = len(token_ids)
            self.bytes_estimate = int(nbytes)
            return SimpleNamespace(committed=True, reason="ok", prefix_len=len(token_ids))

    class FakeBank:
        max_bytes = 2_000_000_000
        per_session_max_bytes = 500_000_000

        def longest_prefix(self, _ids):
            return SimpleNamespace(prefix_len=100, nbytes=0, oversized_nbytes=480_000_000)

        def warn_oversized_snapshot_skip(self, _sid, *, needed_nbytes):
            pass

    state = _inkling_state()
    state.sessions = SimpleNamespace(bank=FakeBank())
    monkeypatch.setattr(
        openai,
        "_history_ids_for_postcommit",
        lambda *_a, **_kw: (list(range(1, 131)), None),
    )
    session = FakeSession()

    outcome = openai._store_retokenized_history_snapshot(
        state,
        session_id="sess-oversized",
        messages=[openai.ChatMessage(role="user", content="hello")],
        assistant_content="world",
        thinking_enabled=False,
        policy_fingerprint="fp-test",
        session=session,
        expected_session_revision=1,
        keep_live_ref=False,
    )

    assert outcome["stored"] is False
    assert outcome["reason"] == "estimated_oversized_snapshot"
    assert outcome["session_commit"]["committed"] is True
    assert recorded_nbytes and recorded_nbytes[0] > 500_000_000
    assert session.bytes_estimate == recorded_nbytes[0]
