"""When a streamed turn's reader saw the model think and answer (12 Nov).

The founder asked why a Pi turn took 30 seconds and whether the cache missed.
In that session the logged first token came at a median 0.74 s, but the first
answer text or tool call came after a median 5.5 s of reasoning at xhigh
(34.6 s at 88K), and nothing in the request log said so. Each streamed chat
turn's request-log row and the app's request record now carry
``reasoning_tokens``, ``reasoning_s`` (request arrival to the last reasoning
delta sent) and ``first_visible_s`` (to the first answer-text or tool-call
delta sent), from the deltas the stream itself sends.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai, request_spans
from mtplx.server.openai import create_app

from test_server_openai import _fake_streaming_session_state

DELAY_S = 0.004  # per token: the turn's phases are far apart on the clock


@pytest.fixture(autouse=True)
def _fresh_registry():
    request_spans.reset_for_tests()
    yield
    request_spans.reset_for_tests()


def _paced_engine(monkeypatch, text: str) -> None:
    def fake_generate_mtpk(_runtime, prompt_ids, **kwargs):
        tokens = [ord(ch) for ch in text]
        callback = kwargs.get("token_callback")
        for token in tokens:
            time.sleep(DELAY_S)
            if callback is not None:
                callback([token])
        return SimpleNamespace(
            tokens=tokens,
            text=text,
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.001,
                    "generated_tokens": len(tokens),
                    "elapsed_s": DELAY_S * len(tokens),
                    "tok_s": 1.0 / DELAY_S,
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)


def _turn(tmp_path: Path, monkeypatch, text: str, body: dict) -> tuple[dict, dict, str]:
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    state.args.stream_interval = 1
    log = tmp_path / "requests.jsonl"
    state.args.request_log_jsonl = str(log)
    _paced_engine(monkeypatch, text)
    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={**body, "stream": True, "max_tokens": 256},
        ) as response:
            assert response.status_code == 200
            sse = response.read().decode()
        # Read right after [DONE]: the row is already written.
        rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1, rows
    request_id = rows[0]["request_id"]
    # The app's request record: the RAM ring /v1/mtplx/snapshot serves as
    # ``latest`` and ``recent``.
    record = next(row for row in state.last_metrics if row.get("request_id") == request_id)
    return rows[0], record, sse


def _deltas(sse: str, field: str) -> str:
    out = []
    for line in sse.splitlines():
        if line.startswith("data: {"):
            for choice in json.loads(line[6:]).get("choices", []):
                value = choice.get("delta", {}).get(field)
                if isinstance(value, str):
                    out.append(value)
    return "".join(out)


FIELDS = ("reasoning_tokens", "reasoning_s", "first_visible_s")


def test_a_thinking_turn_says_how_long_its_reader_watched_it_think(tmp_path, monkeypatch):
    reasoning = "plan the answer step by step, then check it twice"
    row, record, sse = _turn(
        tmp_path,
        monkeypatch,
        f"<think>{reasoning}</think>\nThe answer is 4.",
        {"messages": [{"role": "user", "content": "2+2?"}], "enable_thinking": True},
    )
    assert _deltas(sse, "reasoning_content") == reasoning
    for got in (row, record):
        assert {key: got[key] for key in FIELDS} == {key: row[key] for key in FIELDS}
    assert row["reasoning_tokens"] == len(reasoning)
    # First token, then the reasoning streams, then the answer starts.
    assert row["ttft_s"] - 1e-3 <= row["reasoning_s"] < row["first_visible_s"]
    assert row["reasoning_s"] - row["ttft_s"] >= 0.5 * DELAY_S * len(reasoning)


def test_a_tool_call_turn_is_visible_when_its_call_goes_out(tmp_path, monkeypatch):
    call = (
        "<tool_call>\n<function=count_lines>\n<parameter=part>\n2\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    row, record, sse = _turn(
        tmp_path,
        monkeypatch,
        call,
        {
            "messages": [
                {"role": "user", "content": "Count the lines of each part."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "count_lines", "arguments": '{"part": 1}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "93 lines"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "count_lines",
                        "description": "Count the lines in one part.",
                        "parameters": {
                            "type": "object",
                            "properties": {"part": {"type": "integer"}},
                            "required": ["part"],
                        },
                    },
                }
            ],
            "enable_thinking": False,
        },
    )
    assert '"tool_calls": [' in sse
    assert {key: record[key] for key in FIELDS} == {key: row[key] for key in FIELDS}
    assert row["reasoning_tokens"] == 0 and row["reasoning_s"] is None
    # The call is held until its markup is whole: the reader sees it at the
    # end of the turn, long after the engine's first token.
    assert row["first_visible_s"] - row["ttft_s"] >= 0.5 * DELAY_S * len(call)


def test_a_turn_without_thinking_is_visible_at_its_first_token(tmp_path, monkeypatch):
    row, record, _sse = _turn(
        tmp_path,
        monkeypatch,
        "Hello there, how can I help?",
        {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False},
    )
    assert {key: record[key] for key in FIELDS} == {key: row[key] for key in FIELDS}
    assert row["reasoning_tokens"] == 0 and row["reasoning_s"] is None
    assert row["ttft_s"] - 1e-3 <= row["first_visible_s"] < row["ttft_s"] + 5 * DELAY_S


def test_a_held_row_is_written_once_whatever_order_things_end_in(tmp_path):
    rows = openai._StreamRows()
    log = tmp_path / "log.jsonl"
    # Not a registered stream: written by its recorder as before.
    assert rows.hold({"request_id": "plain"}, str(log)) == (False, None)
    # A stream's row waits; a second attempt's row supersedes the first,
    # which its recorder writes at once.
    rows.open("s1")
    first, second = {"request_id": "s1", "attempt": 1}, {"request_id": "s1", "attempt": 2}
    assert rows.hold(first, str(log)) == (True, None)
    held, superseded = rows.hold(second, str(log))
    assert held and superseded == (first, str(log))
    assert rows.release("s1") == (second, str(log))
    assert rows.release("s1") is None
    # A stream that ended before its generation did: the late row is not held.
    rows.open("s2")
    assert rows.release("s2") is None
    assert rows.hold({"request_id": "s2"}, str(log)) == (False, None)
