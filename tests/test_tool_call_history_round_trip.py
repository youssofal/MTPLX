"""#691 end to end: a tool call the client sends back renders as the tokens the model wrote.

The session bank serves a coding agent's next turn only when that turn's prompt
extends the committed stream (the previous turn's prompt ids plus its generated
ids). Pi sends each tool call back with the arguments exactly as it received
them. Before #691 a string argument lost its edge newlines and the arguments
were reordered path first, so the call rendered differently on the way back:
the end-of-answer save was refused (retokenized_history_mismatch) and the next
turn prefilled the call again, while the file on disk lost its last newline.

These tests drive the server's own /v1/chat/completions stream path in Pi's
request shape (its read and write tool schemas, streamed with usage) on a
Qwen 3.6 pack's real tokenizer and chat template. There is no model: a scripted
generation hands the server one real token per callback, so the "\\n\\n" token
before "</parameter>" reaches the stream path as a model emits it. Thinking is
off so the comparison isolates the call. They skip where the pack is not
installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.server import openai as oa

PACK = Path.home() / ".mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"

pytestmark = pytest.mark.skipif(
    not (PACK / "tokenizer.json").exists() or not (PACK / "chat_template.jinja").exists(),
    reason="needs the Qwen 3.6 27B Optimized Speed pack's tokenizer and chat template",
)

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read the contents of a file.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
         "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
         "limit": {"type": "number", "description": "Maximum number of lines to read"}},
         "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to a file.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
         "content": {"type": "string", "description": "Content to write to the file"}},
         "required": ["path", "content"]}}},
]

TOTAL = "def total(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"

CASES = [
    pytest.param("write", [("path", "total.py"), ("content", TOTAL)],
                 {"path": "total.py", "content": TOTAL}, id="file-ending-in-newline"),
    pytest.param("write", [("path", "x.py"), ("content", "x = 1\n\n")],
                 {"path": "x.py", "content": "x = 1\n\n"}, id="file-ending-in-blank-line"),
    pytest.param("write", [("path", "x.py"), ("content", "  x = 1  ")],
                 {"path": "x.py", "content": "  x = 1  "}, id="file-with-edge-spaces"),
    pytest.param("read", [("limit", "50"), ("offset", "147"), ("path", "a.py")],
                 {"limit": 50, "offset": 147, "path": "a.py"}, id="read-limit-before-path"),
]


def _model_call(tool: str, params: list[tuple[str, str]]) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={tool}>\n{body}</function>\n</tool_call>"


@pytest.fixture(scope="module")
def tokenizer():
    from mtplx.runtime import _load_tokenizer_resilient

    return _load_tokenizer_resilient(PACK, json.loads((PACK / "config.json").read_text()))


@pytest.fixture
def pi_server(monkeypatch, tokenizer):
    from fastapi.testclient import TestClient

    from test_server_openai import _fake_state

    state = _fake_state()
    state.args = oa.parse_args(["--warmup-tokens", "0", "--model", str(PACK)])
    state.args.stream_interval = 1
    state.args.stats_footer = False
    state.runtime.model_path = PACK
    state.runtime.tokenizer = tokenizer
    state.context_window = 262144
    if hasattr(oa, "_DATE_LINE_PIN"):
        monkeypatch.setitem(oa._DATE_LINE_PIN, "day", "November 11, 2026")
    prompts: list[list[int]] = []
    script: dict[str, list[int]] = {}

    def scripted_generation(_state, prompt_ids, **kwargs):
        prompts.append([int(t) for t in prompt_ids])
        ids = script["ids"]
        callback = kwargs.get("token_callback")
        if callback is not None:
            for token in ids:
                callback([token])
        return {
            "text": tokenizer.decode(ids),
            "tokens": list(ids),
            "stats": {"generation_mode": kwargs.get("generation_mode"), "mtp_depth": kwargs.get("depth"),
                      "completion_tokens": len(ids)},
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(ids),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(oa, "_run_generation", scripted_generation)
    client = TestClient(oa.create_app(state))

    def turn(messages: list[dict], model_text: str) -> tuple[list[int], list[dict]]:
        script["ids"] = [int(t) for t in tokenizer.encode(model_text, add_special_tokens=False)]
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-cache-mode": "bypass"},
            json={"messages": messages, "tools": TOOLS, "stream": True,
                  "stream_options": {"include_usage": True}, "enable_thinking": False,
                  "max_tokens": 4096},
        )
        assert response.status_code == 200, response.text
        payloads = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
                    if line.startswith("data: {")]
        return prompts[-1] + script["ids"], payloads

    return turn, prompts


def _received_call(payloads: list[dict]) -> dict:
    call = {"id": None, "name": "", "arguments": ""}
    for payload in payloads:
        for delta in (payload.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or []:
            call["id"] = delta.get("id") or call["id"]
            function = delta.get("function") or {}
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""
    return call


@pytest.mark.parametrize("tool, params, expected", CASES)
def test_the_call_pi_sends_back_extends_the_committed_stream(pi_server, tool, params, expected):
    turn, prompts = pi_server
    first = [{"role": "system", "content": "You are a coding assistant."},
             {"role": "user", "content": "Do the next step."}]
    committed, payloads = turn(first, _model_call(tool, params))
    call = _received_call(payloads)
    received = json.loads(call["arguments"])
    # The client gets the file exactly, in the model's argument order.
    assert received == expected
    assert list(received) == list(expected)
    # Pi sends the call back unchanged with the tool's result.
    second = first + [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": call["id"], "type": "function",
                         "function": {"name": call["name"], "arguments": call["arguments"]}}]},
        {"role": "tool", "tool_call_id": call["id"], "content": "ok"},
    ]
    turn(second, "Done.")
    next_prompt = prompts[-1]
    assert next_prompt[: len(committed)] == committed, (
        "the next prompt no longer extends the committed stream at token "
        f"{next(i for i, (a, b) in enumerate(zip(committed, next_prompt)) if a != b)}"
    )
