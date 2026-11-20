"""What the Codex desktop app (26.1106) sends to /v1/responses, accepted.

Shapes recorded from a real copy of the app talking to a stand-in server on
12 Nov 2026 (receipts/codex-copy-live-turn): every turn asks for detailed
reasoning summaries, its title request carries reasoning.context and an
additional_tools item whose namespace holds a freeform tool. Before this,
every one of those turns was answered with a 400.
"""

from __future__ import annotations

import pytest

from mtplx.server import openai
from mtplx.server import responses as R


def _translate(body):
    return R.translate_request(R.ResponsesRequest.model_validate(body))


def _user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


@pytest.mark.parametrize("summary", ["auto", "concise", "detailed", "none"])
def test_every_reasoning_summary_style_is_accepted(summary):
    chat = _translate(
        {"model": "m", "input": [_user("hi")], "reasoning": {"effort": "low", "summary": summary}}
    ).chat
    assert chat["reasoning_effort"] == "low"
    assert chat["metadata"]["responses_reasoning_summary"] == summary


def test_reasoning_context_is_accepted_and_recorded():
    chat = _translate(
        {"model": "m", "input": [_user("hi")], "reasoning": {"effort": "low", "context": "all_turns"}}
    ).chat
    assert chat["metadata"]["responses_reasoning_context"] == "all_turns"
    with pytest.raises(R.ResponsesProtocolError, match="reasoning.context must be a string"):
        _translate({"model": "m", "input": [_user("hi")], "reasoning": {"context": 3}})
    with pytest.raises(R.ResponsesProtocolError, match="unsupported key"):
        _translate({"model": "m", "input": [_user("hi")], "reasoning": {"budget": 3}})


def _additional_tools():
    return {
        "type": "additional_tools",
        "id": "at_1",
        "role": "developer",
        "tools": [
            {
                "type": "namespace",
                "name": "functions",
                "description": "Tools",
                "tools": [
                    {"type": "custom", "name": "apply", "description": "Freeform input"},
                    {
                        "type": "function",
                        "name": "wait",
                        "description": "Wait",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            }
        ],
    }


def test_additional_tools_join_the_declared_tools_and_render_no_message():
    body = {
        "model": "m",
        "tools": [
            {"type": "function", "name": "exec_command", "description": "Run",
             "parameters": {"type": "object", "properties": {}}}
        ],
        "input": [_additional_tools(), _user("hi")],
    }
    conversion = _translate(body)
    names = [tool["function"]["name"] for tool in conversion.chat["tools"]]
    assert names[0] == "exec_command" and len(names) == 3
    assert [message["role"] for message in conversion.chat["messages"]] == ["user"]
    # The freeform tool in the namespace comes back as a custom call with its
    # namespace, the function as a function call with its namespace.
    flat = {identity.name: chat for chat, identity in conversion.namespace_functions.items()}
    assert flat["apply"] in conversion.custom_tool_names
    assert flat["wait"] not in conversion.custom_tool_names
    output = R.output_from_chat(
        {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": flat["apply"], "arguments": '{"input": "x"}'}},
            {"id": "c2", "type": "function", "function": {"name": flat["wait"], "arguments": "{}"}},
        ]}}]},
        custom_tool_names=conversion.custom_tool_names,
        namespace_functions=conversion.namespace_functions,
    )
    calls = [item for item in output if item["type"] in {"custom_tool_call", "function_call"}]
    assert [(c["type"], c["name"], c.get("namespace")) for c in calls] == [
        ("custom_tool_call", "apply", "functions"),
        ("function_call", "wait", "functions"),
    ]
    assert calls[0]["input"] == "x"
    # The same item replayed on the next request declares the same tools in
    # the same order, so the thread's prefix stays reusable.
    again = _translate({**body, "input": [_additional_tools(), _user("hi"), _user("more")]})
    assert [t["function"]["name"] for t in again.chat["tools"]] == names


def test_additional_tools_must_come_from_the_developer_with_a_tool_list():
    item = _additional_tools()
    with pytest.raises(R.ResponsesProtocolError, match="developer role"):
        _translate({"model": "m", "input": [{**item, "role": "user"}, _user("hi")]})
    with pytest.raises(R.ResponsesProtocolError, match="list of tool objects"):
        _translate({"model": "m", "input": [{**item, "tools": "x"}, _user("hi")]})


def test_a_grammar_tool_is_still_refused_because_it_cannot_be_enforced():
    item = _additional_tools()
    item["tools"][0]["tools"][0]["format"] = {"type": "grammar", "syntax": "lark", "definition": "start: /x/"}
    with pytest.raises(R.ResponsesProtocolError, match="only freeform text"):
        _translate({"model": "m", "input": [item, _user("hi")]})


def test_an_original_detail_image_is_accepted():
    body = {
        "model": "m",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "original"},
        ]}],
    }
    content = _translate(body).chat["messages"][0]["content"]
    assert content[0]["image_url"]["detail"] == "original"


# Through the server: the same request shape the app sends on every turn
# (receipts/codex-copy-effort-switch), with our own text and a few tools.

PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+aL1sAAAAASUVORK5CYII="
)


def _function(name):
    return {
        "type": "function",
        "name": name,
        "description": f"{name} tool",
        "strict": False,
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
    }


def _app_turn(effort, inputs):
    return {
        "model": "mtplx-test-model",
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "prompt_cache_key": "thread-1",
        "client_metadata": {"thread_id": "thread-1", "turn_id": "turn-1"},
        "reasoning": {"effort": effort, "summary": "detailed"},
        "tools": [
            _function("exec_command"),
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Agents",
                "tools": [_function("spawn_agent"), _function("wait_agent")],
            },
        ],
        "input": inputs,
    }


def _developer(text):
    return {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]}


def _assistant(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _post(monkeypatch, body):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tests.test_server_openai import CaptureTokenizer, _fake_state, _fake_streaming_generation

    state = _fake_state()
    state.args.stats_footer = False
    state.context_window = 262144
    state.runtime.tokenizer = CaptureTokenizer()
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation("ok"))
    response = TestClient(openai.create_app(state)).post(
        "/v1/responses", headers={"x-mtplx-cache-mode": "bypass"}, json=body
    )
    return response, state.runtime.tokenizer.calls


def test_an_app_turn_streams_through_the_server(monkeypatch):
    response, calls = _post(monkeypatch, _app_turn("low", [_developer("Be brief."), _user("hi")]))
    assert response.status_code == 200, response.text
    assert "event: response.completed" in response.text
    _messages, kwargs = calls[-1]
    assert [tool["function"]["name"] for tool in kwargs["tools"]][0] == "exec_command"
    assert len(kwargs["tools"]) == 3
    assert kwargs["reasoning_effort"] == "low"


def test_an_effort_change_mid_thread_arrives_on_the_next_turn(monkeypatch):
    """The app's effort slider sends no item of its own: the next turn simply
    carries the new effort, with the thread's messages unchanged before it.
    The Qwen 3.8 family has no "high", so the app's High is that family's
    next level up, "xhigh"."""
    first = [_developer("Be brief."), _user("First.")]
    response, calls = _post(monkeypatch, _app_turn("medium", first))
    assert response.status_code == 200, response.text
    first_messages, first_kwargs = calls[-1]
    second = [*first, _assistant("ok"), _user("Second.")]
    response, calls = _post(monkeypatch, _app_turn("high", second))
    assert response.status_code == 200, response.text
    second_messages, second_kwargs = calls[-1]
    assert (first_kwargs["reasoning_effort"], second_kwargs["reasoning_effort"]) == ("medium", "xhigh")
    assert second_messages[: len(first_messages)] == first_messages
    assert second_kwargs["tools"] == first_kwargs["tools"]


def test_an_app_image_turn_reaches_the_vision_path(monkeypatch):
    """The app sends a sketch or screenshot as input_image (detail "high")
    after the text. The test runtime has no vision tower, so the request is
    answered by the server's vision check, not by the translator: the image
    was accepted and extracted. On a pack with its tower it is read."""
    image_turn = {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "What is in this sketch? One word."},
        {"type": "input_image", "image_url": PNG, "detail": "high"},
    ]}
    response, _calls = _post(monkeypatch, _app_turn("medium", [_developer("Be brief."), image_turn]))
    assert response.status_code == 400
    assert "no vision tower" in response.json()["error"]["message"]
