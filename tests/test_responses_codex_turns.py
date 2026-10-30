"""Codex on the Responses API: one response is one assistant turn (#603).

Codex CLI sends the model's short preamble ("Reading the importer first:") as
an assistant message item followed by the function_call item of the same
response. Translated as two assistant messages, the Qwen template rendered
every tool call in the history as "announcement, end of turn" plus a second
turn holding the call, and after 10 to 25 calls the model copied the pattern
and stopped. Item shapes come from the #603 report (Codex CLI 0.154).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server import responses as responses_api
from mtplx.server.openai import create_app
from tests.test_server_openai import (
    CaptureTokenizer,
    _fake_state,
    _fake_streaming_generation,
)

EXEC_COMMAND = {
    "type": "function",
    "name": "exec_command",
    "description": "run",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}


def _user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _assistant(text):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _call(call_id, cmd):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "exec_command",
        "arguments": json.dumps({"cmd": cmd}),
    }


def _output(call_id, text):
    return {"type": "function_call_output", "call_id": call_id, "output": text}


def _messages(items):
    request = responses_api.ResponsesRequest(model="m", input=items, tools=[EXEC_COMMAND])
    return responses_api.translate_request(request).chat["messages"]


def _shape(messages):
    return [
        (
            message["role"],
            message.get("content"),
            [call["id"] for call in message.get("tool_calls") or []],
        )
        for message in messages
    ]


def test_the_reporters_preamble_and_call_are_one_assistant_turn():
    messages = _messages(
        [
            _user("Fix the bug."),
            _assistant("Reading the importer first:"),
            _call("c1", "ls"),
            _output("c1", "a.py"),
        ]
    )
    assert _shape(messages) == [
        ("user", "Fix the bug.", []),
        ("assistant", "Reading the importer first:", ["c1"]),
        ("tool", "a.py", []),
    ]


def test_a_codex_tool_loop_keeps_one_assistant_turn_per_response():
    messages = _messages(
        [
            _user("Write the tests."),
            _assistant("Reading the importer first:"),
            _call("c1", "cat importer.py"),
            _output("c1", "def load(): ..."),
            _assistant("Now the two test files:"),
            _call("c2", "cat tests/a.py"),
            _call("c3", "cat tests/b.py"),
            _output("c2", "a"),
            _output("c3", "b"),
            _call("c4", "pytest -q"),
            _output("c4", "2 passed"),
            _assistant("Both tests pass."),
        ]
    )
    assert _shape(messages) == [
        ("user", "Write the tests.", []),
        ("assistant", "Reading the importer first:", ["c1"]),
        ("tool", "def load(): ...", []),
        ("assistant", "Now the two test files:", ["c2", "c3"]),
        ("tool", "a", []),
        ("tool", "b", []),
        ("assistant", "", ["c4"]),
        ("tool", "2 passed", []),
        ("assistant", "Both tests pass.", []),
    ]


def test_mtplxs_own_empty_message_item_does_not_become_an_empty_turn():
    # output_from_chat writes a message item whenever content is not None, so
    # a calls-only answer with content "" comes back as an empty message item.
    messages = _messages([_user("go"), _assistant(""), _call("c1", "ls"), _output("c1", "x")])
    assert _shape(messages) == [
        ("user", "go", []),
        ("assistant", "", ["c1"]),
        ("tool", "x", []),
    ]


def test_a_call_after_a_user_message_still_opens_its_own_turn():
    messages = _messages(
        [_assistant("Done."), _user("Run it again."), _call("c1", "ls"), _output("c1", "x")]
    )
    assert _shape(messages) == [
        ("assistant", "Done.", []),
        ("user", "Run it again.", []),
        ("assistant", "", ["c1"]),
        ("tool", "x", []),
    ]


def test_the_merged_turn_reaches_the_chat_template_through_the_server(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    state.args.enable_thinking = False
    state.runtime.tokenizer = CaptureTokenizer()
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation("OK"))

    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "stream": True,
            "tools": [EXEC_COMMAND],
            "input": [
                _user("Fix the bug."),
                _assistant("Reading the importer first:"),
                _call("c1", "ls"),
                _output("c1", "a.py"),
            ],
        },
    )

    assert response.status_code == 200, response.text
    rendered, _kwargs = state.runtime.tokenizer.calls[0]
    assistant_turns = [message for message in rendered if message["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["content"] == "Reading the importer first:"
    assert [call["function"]["name"] for call in assistant_turns[0]["tool_calls"]] == [
        "exec_command"
    ]
