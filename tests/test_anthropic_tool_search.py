"""Claude Code's tool search through /v1/messages.

Against a host that is not Anthropic's, Claude Code sends every MCP tool
definition in full unless ENABLE_TOOL_SEARCH is set; on 12 Nov that was 820
tools and a 254,636-token first prompt. With tool search on it declares
deferred tools with ``defer_loading: true`` and loads one by answering its own
ToolSearch call with ``tool_reference`` blocks (Claude Code 2.1.201).

The server must keep deferred tools out of the prompt's tool section, write a
loaded tool's definition where it was loaded (never into the system prompt, so
the cached prefix survives the load), and still parse the model's call to it.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import (
    AnthropicMessagesRequest,
    _anthropic_to_chat_request,
    create_app,
    resolve_request_policy,
)

from test_server_openai import (  # noqa: E402 - shared fixtures
    CaptureTokenizer,
    JSONToolPrefixTokenizer,
    _fake_generation,
    _fake_state,
)

BASH = {
    "name": "Bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}
TOOL_SEARCH = {
    "name": "ToolSearch",
    "description": "Fetches full schema definitions for deferred tools.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
STATUS = {
    "name": "mcp__history__status",
    "description": "Report whether the recorder is running.",
    "input_schema": {
        "type": "object",
        "properties": {"verbose": {"type": "boolean"}},
    },
    "defer_loading": True,
}
STATUS_SPEC = {
    "type": "function",
    "function": {
        "name": STATUS["name"],
        "description": STATUS["description"],
        "parameters": STATUS["input_schema"],
    },
}

ASK = {"role": "user", "content": "Is the recorder running?"}
SEARCH_CALL = {
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_search",
            "name": "ToolSearch",
            "input": {"query": "select:mcp__history__status"},
        }
    ],
}
SEARCH_RESULT = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_search",
            "content": [{"type": "tool_reference", "tool_name": STATUS["name"]}],
        }
    ],
}


def _request(messages, tools) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="mtplx-test-model", max_tokens=64, messages=messages, tools=tools
    )


def _names(specs) -> list[str]:
    return [spec["function"]["name"] for spec in specs or ()]


class QwenToolsTokenizer(CaptureTokenizer):
    """Lists tools as the Qwen templates do: one JSON object per line inside
    <tools> ... </tools>, ahead of the conversation (the encoding a loaded
    tool's definition is written in)."""

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        tools = kwargs.get("tools") or []
        text = "<tools>" + "".join(
            "\n" + json.dumps(tool, ensure_ascii=False) for tool in tools
        ) + "\n</tools>\n"
        text += "\n".join(
            f"{message['role']}:{message.get('content') or ''}" for message in messages
        )
        if kwargs.get("add_generation_prompt"):
            text += "\nassistant:"
        return self.encode(text) if kwargs.get("tokenize", True) else text


def test_a_deferred_tool_is_declared_but_not_rendered():
    chat = _anthropic_to_chat_request(_request([ASK], [BASH, TOOL_SEARCH, STATUS]))

    # The parser knows every tool; the tool section leaves the deferred one out.
    assert _names(chat.tools) == ["Bash", "ToolSearch", STATUS["name"]]
    assert chat.deferred_tool_names == frozenset({STATUS["name"]})
    assert chat.loaded_tool_names == ()
    policy = resolve_request_policy(_fake_state(), chat, headers={}, metadata={})
    assert _names(policy.prompt_tool_specs) == ["Bash", "ToolSearch"]
    assert _names(policy.tool_specs) == ["Bash", "ToolSearch", STATUS["name"]]
    assert policy.postcommit_tool_specs == policy.prompt_tool_specs
    assert policy.observability["request_deferred_tool_count"] == 1
    assert policy.observability["request_loaded_tool_count"] == 0


def test_a_tool_reference_writes_the_definition_where_it_was_loaded():
    chat = _anthropic_to_chat_request(
        _request([ASK, SEARCH_CALL, SEARCH_RESULT], [BASH, TOOL_SEARCH, STATUS])
    )

    tool_message = chat.messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "toolu_search"
    # The prompt's own tool encoding: one JSON function per line in <tools>,
    # keys sorted as the template receives them.
    line = json.dumps(STATUS_SPEC, sort_keys=True, ensure_ascii=False)
    assert tool_message.content == (
        "You now have access to the following functions:\n\n<tools>\n"
        f"{line}\n</tools>"
    )
    assert chat.loaded_tool_names == (STATUS["name"],)
    policy = resolve_request_policy(_fake_state(), chat, headers={}, metadata={})
    assert _names(policy.prompt_tool_specs) == ["Bash", "ToolSearch"]
    assert policy.observability["request_loaded_tool_count"] == 1


def test_consecutive_references_load_as_one_block_and_unknown_names_say_so():
    other = {**STATUS, "name": "mcp__history__search"}
    result = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_search",
                "content": [
                    {"type": "tool_reference", "tool_name": STATUS["name"]},
                    {"type": "tool_reference", "tool_name": other["name"]},
                    {"type": "tool_reference", "tool_name": "mcp__gone__tool"},
                ],
            }
        ],
    }
    chat = _anthropic_to_chat_request(
        _request([ASK, SEARCH_CALL, result], [BASH, TOOL_SEARCH, STATUS, other])
    )

    content = chat.messages[-1].content
    assert content.count("<tools>") == 1
    assert f'"name": "{STATUS["name"]}"' in content
    assert f'"name": "{other["name"]}"' in content
    assert content.endswith("These tools are not available: mcp__gone__tool")
    assert chat.loaded_tool_names == (STATUS["name"], other["name"])


def test_a_request_without_deferred_tools_converts_as_before():
    chat = _anthropic_to_chat_request(_request([ASK, SEARCH_CALL], [BASH, TOOL_SEARCH]))

    assert chat.deferred_tool_names == frozenset()
    assert chat.tools == openai._anthropic_tools_to_openai([BASH, TOOL_SEARCH])
    policy = resolve_request_policy(_fake_state(), chat, headers={}, metadata={})
    assert policy.prompt_tool_specs == policy.tool_specs
    # Content without a reference is returned as the same object: its bytes
    # cannot move.
    content = SEARCH_CALL["content"]
    assert openai._anthropic_expand_tool_references(content, {}, []) is content


def test_anthropic_server_search_tool_keeps_every_tool_in_the_tool_section():
    server_search = {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }
    chat = _anthropic_to_chat_request(
        _request([ASK], [server_search, BASH, STATUS])
    )

    # The server cannot run Anthropic's own search, so nothing is deferred
    # out of the model's reach, and the search tool is not offered as a call.
    assert chat.deferred_tool_names == frozenset()
    assert _names(chat.tools) == ["Bash", STATUS["name"]]


def test_every_tool_deferred_is_refused_as_the_anthropic_api_refuses_it():
    with pytest.raises(openai.HTTPException) as refused:
        _anthropic_to_chat_request(_request([ASK], [STATUS]))
    assert refused.value.status_code == 400


def test_loading_a_tool_leaves_the_tool_section_and_prefix_unchanged(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    state = _fake_state()
    state.runtime.tokenizer = QwenToolsTokenizer()
    state.args.stats_footer = False
    prompts: list[str] = []

    def fake_run_generation(_state, prompt_ids, **_kwargs):
        prompts.append("".join(map(chr, prompt_ids)))
        return _fake_generation("Checked.")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    client = TestClient(create_app(state))
    before = {
        "model": "mtplx-test-model",
        "max_tokens": 16,
        "messages": [ASK],
        "tools": [BASH, TOOL_SEARCH],
    }
    after = {
        **before,
        "messages": [ASK, SEARCH_CALL, SEARCH_RESULT],
        # Claude Code adds a tool to `tools` once it has loaded it.
        "tools": [BASH, TOOL_SEARCH, STATUS],
    }
    for body in (before, after):
        response = client.post(
            "/v1/messages", headers={"x-mtplx-cache-mode": "bypass"}, json=body
        )
        assert response.status_code == 200, response.text

    tool_section = [prompt.split("</tools>", 1)[0] for prompt in prompts]
    assert tool_section[0] == tool_section[1]
    assert STATUS["name"] not in tool_section[1]
    # The definition rides the tool result, after everything the first turn read.
    loaded_at = prompts[1].index('"name": "mcp__history__status"')
    assert loaded_at > prompts[1].index(ASK["content"])


def test_the_model_can_call_a_tool_it_loaded(monkeypatch):
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            "<tool_call>\n"
            f"<function={STATUS['name']}>\n"
            "<parameter=verbose>\n"
            "true\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )
    client = TestClient(create_app(_fake_state()))

    response = client.post(
        "/v1/messages",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "max_tokens": 64,
            "messages": [ASK, SEARCH_CALL, SEARCH_RESULT],
            "tools": [BASH, TOOL_SEARCH, STATUS],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["stop_reason"] == "tool_use"
    call = next(block for block in payload["content"] if block["type"] == "tool_use")
    assert call["name"] == STATUS["name"]
    assert call["input"] == {"verbose": True}


def test_count_tokens_counts_the_prompt_the_model_reads():
    state = _fake_state()
    state.runtime.tokenizer = QwenToolsTokenizer()
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "mtplx-test-model",
            "messages": [ASK, SEARCH_CALL, SEARCH_RESULT],
            "tools": [BASH, TOOL_SEARCH, STATUS],
        },
    )

    assert response.status_code == 200, response.text
    messages, kwargs = state.runtime.tokenizer.calls[-1]
    assert _names(kwargs["tools"]) == ["Bash", "ToolSearch"]
    assert any(
        '"name": "mcp__history__status"' in str(message.get("content") or "")
        for message in messages
    )



def test_the_tool_encoding_probe_reads_how_the_template_lists_tools():
    probe = openai._TOOL_ENCODING_PROBE
    qwen = "<tools>\n" + json.dumps(probe) + "\n</tools>\nuser:Hi."
    as_array = json.dumps([probe]) + "\nuser:Hi."
    gemma_like = (
        "<start_of_turn>system\ndeclaration:mtplx_tool_encoding_probe"
        "{description:<escape>Probe.<escape>}<end_of_turn>"
    )
    assert openai._template_lists_tools_as_json_lines(qwen)
    assert not openai._template_lists_tools_as_json_lines(as_array)
    assert not openai._template_lists_tools_as_json_lines(gemma_like)
    assert not openai._template_lists_tools_as_json_lines("<tools>\nnot json\n</tools>")


def test_the_served_template_is_probed_once_per_tokenizer():
    state = _fake_state()
    state.runtime.tokenizer = QwenToolsTokenizer()
    assert openai._served_template_lists_tools_as_json_lines(state) is True
    assert openai._served_template_lists_tools_as_json_lines(state) is True
    assert len(state.runtime.tokenizer.calls) == 1
    # A reloaded model is probed again.
    state.runtime.tokenizer = JSONToolPrefixTokenizer()
    assert openai._served_template_lists_tools_as_json_lines(state) is False


def test_without_that_encoding_every_tool_is_declared_up_front():
    chat = _anthropic_to_chat_request(
        _request([ASK, SEARCH_CALL, SEARCH_RESULT], [BASH, TOOL_SEARCH, STATUS]),
        defer_tools=False,
    )

    assert chat.deferred_tool_names == frozenset()
    assert chat.loaded_tool_names == ()
    assert chat.defer_loading_declined_names == frozenset({STATUS["name"]})
    policy = resolve_request_policy(_fake_state(), chat, headers={}, metadata={})
    assert _names(policy.prompt_tool_specs) == ["Bash", "ToolSearch", STATUS["name"]]
    assert policy.observability["request_deferred_tool_count"] == 0
    assert policy.observability["request_defer_loading_declined_count"] == 1
    # The reference names the tool and writes no definition in another
    # template's encoding.
    texts = [str(message.content) for message in chat.messages]
    assert any("Loaded: mcp__history__status" in text for text in texts)
    assert not any("<tools>" in text for text in texts)


def test_a_server_whose_template_lists_tools_otherwise_keeps_them_up_front(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    state = _fake_state()
    # Tools as one JSON array, not one object per line inside <tools>.
    state.runtime.tokenizer = JSONToolPrefixTokenizer()
    state.args.stats_footer = False
    prompts: list[str] = []

    def fake_run_generation(_state, prompt_ids, **_kwargs):
        prompts.append("".join(map(chr, prompt_ids)))
        return _fake_generation("Checked.")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    client = TestClient(create_app(state))
    response = client.post(
        "/v1/messages",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "max_tokens": 16,
            "messages": [ASK, SEARCH_CALL, SEARCH_RESULT],
            "tools": [BASH, TOOL_SEARCH, STATUS],
        },
    )

    assert response.status_code == 200, response.text
    assert STATUS["name"] in prompts[-1].split("\n", 1)[0]
    assert "<tools>" not in prompts[-1]
    assert "Loaded: mcp__history__status" in prompts[-1]
    assert client.get("/health").json()["anthropic_tool_search"] is False
