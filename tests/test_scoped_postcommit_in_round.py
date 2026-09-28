"""Scoped reasoning history: the postcommit prefix must extend in-round.

Under scoped mode (the Qwen3.6 default) the template's rolling checkpoint
keeps think blocks for every assistant turn after the last real user query,
so an agent's tool round carries its reasoning forward: the next request
(the client echoing reasoning_content) renders this turn's think interior.
The postcommit prediction appended the generated turn WITHOUT that interior
and skipped the committed-think walk under scoped, so it rendered an empty
think scaffold where the next prompt has the interior. Every in-round
generation-final commit was refused with reasoning_history_scoping_mismatch
and the turn re-prefilled from the last block boundary.

These tests run the REAL encode path (Qwen3.6 tokenizer + chat template, CPU
only, no model load): the predicted next-turn prefix must be a byte prefix
of the next request's encode, both in-round (tool continuation) and when a
new user query closes the round.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.server import openai as oa

MODEL_DIR = Path.home() / ".mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "chat_template.jinja").exists(),
    reason="Qwen3.6 model pack not cached locally",
)

SYSTEM = {"role": "system", "content": "You are a careful assistant for a small archive."}
U1 = {"role": "user", "content": "Read part 1 of the notes, then answer."}
THINK = "The user wants part 1 first. I will call read_notes with part 1."
TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_notes",
        "description": "Read one part of the archive notes.",
        "parameters": {
            "type": "object",
            "properties": {"part": {"type": "integer"}},
            "required": ["part"],
        },
    },
}]
TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "read_notes", "arguments": json.dumps({"part": 1})},
}
TOOL_CALL_TEXT = (
    "<tool_call>\n<function=read_notes>\n<parameter=part>\n1\n</parameter>\n"
    "</function>\n</tool_call>"
)


@pytest.fixture(scope="module")
def tok():
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    return _load_tokenizer_resilient(MODEL_DIR, config)


@pytest.fixture
def scoped(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: True)
    monkeypatch.setattr(oa, "_reasoning_history_preserve_echo_active", lambda state: False)
    monkeypatch.setattr(oa, "_reasoning_effort_for_state", lambda *a, **kw: None)
    monkeypatch.setattr(oa, "_reasoning_parser_for_state", lambda state: "qwen3")


def _encode(tok, messages, *, add_generation_prompt=True):
    request = oa.ChatCompletionRequest(model="m", messages=messages, tools=TOOLS)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=True,
        scoped_reasoning_history=True,
        add_generation_prompt=add_generation_prompt,
        tools=TOOLS,
        template_observability={},
    )


def _state(tok):
    return SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False, tool_prompt_mode="hybrid"),
        runtime=SimpleNamespace(tokenizer=tok),
    )


def _predicted_prefix(tok, committed_stream_ids):
    ids, _splice = oa._history_ids_for_postcommit(
        _state(tok),
        messages=oa.ChatCompletionRequest(model="m", messages=[SYSTEM, U1], tools=TOOLS).messages,
        assistant_content="",
        assistant_tool_calls=[TOOL_CALL],
        thinking_enabled=True,
        tool_specs=TOOLS,
        tool_prompt_mode="hybrid",
        committed_stream_ids=committed_stream_ids,
    )
    return [int(t) for t in ids]


def _turn1(tok):
    prompt = _encode(tok, [SYSTEM, U1])
    generated = oa._encode_rendered_chat_text(
        tok, f"{THINK}\n</think>\n\n{TOOL_CALL_TEXT}<|im_end|>\n"
    )
    return prompt, generated


ECHOED_ASSISTANT = {
    "role": "assistant", "content": "", "reasoning_content": THINK, "tool_calls": [TOOL_CALL],
}


def test_in_round_prefix_extends_to_the_next_tool_continuation(tok, scoped):
    prompt, generated = _turn1(tok)
    predicted = _predicted_prefix(tok, prompt + generated)
    assert predicted
    assert THINK in tok.decode(predicted), "this turn's think must survive in-round"

    next_prompt = _encode(tok, [
        SYSTEM, U1, ECHOED_ASSISTANT,
        {"role": "tool", "tool_call_id": "call_1", "content": "Part 1: the harbour."},
    ])
    assert next_prompt[: len(predicted)] == predicted, (
        "the banked prefix must be a byte prefix of the next in-round prompt"
    )
    assert predicted[: len(prompt) + len(generated)] == prompt + generated, (
        "and it must byte-extend the generation it was committed from"
    )


def test_a_new_user_query_still_scopes_the_round_out(tok, scoped):
    """The next real user query closes the round: its prompt drops the think
    interior, so the predicted in-round prefix is not reused there - the
    template, not the prediction, decides what is kept."""
    prompt, generated = _turn1(tok)
    predicted = _predicted_prefix(tok, prompt + generated)
    next_prompt = _encode(tok, [
        SYSTEM, U1, ECHOED_ASSISTANT,
        {"role": "tool", "tool_call_id": "call_1", "content": "Part 1: the harbour."},
        {"role": "assistant", "content": "The harbour."},
        {"role": "user", "content": "Thanks. Now part 2."},
    ])
    assert THINK not in tok.decode(next_prompt)
    assert next_prompt[: len(predicted)] != predicted


def test_without_committed_stream_the_legacy_render_is_kept(tok, scoped):
    prompt, generated = _turn1(tok)
    assert THINK not in tok.decode(_predicted_prefix(tok, None))
    assert THINK in tok.decode(_predicted_prefix(tok, prompt + generated))


# --- Anthropic bridge: client bookkeeping after tool results ---------------------------


def _bridge(content):
    return oa._anthropic_message_to_chat_messages(oa.AnthropicMessage(role="user", content=content))


def test_trailing_client_metadata_folds_into_the_tool_response():
    out = _bridge([
        {"type": "tool_result", "tool_use_id": "call_1", "content": "Part 1: the harbour."},
        {"type": "text", "text": "<total_tokens>198629 tokens left</total_tokens>"},
    ])
    assert [m.role for m in out] == ["tool"]
    assert out[0].content == "Part 1: the harbour.\n\n<total_tokens>198629 tokens left</total_tokens>"


def test_a_real_user_sentence_after_tool_results_stays_a_user_query():
    out = _bridge([
        {"type": "tool_result", "tool_use_id": "call_1", "content": "Part 1: the harbour."},
        {"type": "text", "text": "Stop reading, just answer now."},
    ])
    assert [m.role for m in out] == ["tool", "user"]


def test_metadata_without_a_tool_result_is_left_alone():
    out = _bridge([{"type": "text", "text": "<system-reminder>be brief</system-reminder>"}])
    assert [m.role for m in out] == ["user"]


def test_claude_code_shaped_round_keeps_its_reasoning_in_the_prompt(tok, scoped):
    """End to end over the bridge: a Claude Code tool step (tool_result plus a
    trailing <total_tokens> block) must not close the round, so the prompt
    still carries the assistant's think interior."""
    anthropic = [
        oa.AnthropicMessage(role="user", content=U1["content"]),
        oa.AnthropicMessage(role="assistant", content=[
            {"type": "thinking", "thinking": THINK, "signature": ""},
            {"type": "tool_use", "id": "call_1", "name": "read_notes", "input": {"part": 1}},
        ]),
        oa.AnthropicMessage(role="user", content=[
            {"type": "tool_result", "tool_use_id": "call_1", "content": "Part 1: the harbour."},
            {"type": "text", "text": "<total_tokens>198629 tokens left</total_tokens>"},
        ]),
    ]
    messages = [oa.ChatMessage(role="system", content=SYSTEM["content"])]
    for m in anthropic:
        messages.extend(oa._anthropic_message_to_chat_messages(m))
    ids = oa._encode_messages(
        tok, messages, enable_thinking=True, scoped_reasoning_history=True,
        tools=TOOLS, template_observability={},
    )
    assert THINK in tok.decode(ids)


def test_thinking_off_tool_round_prefix_extends_with_client_metadata(tok, scoped):
    """Thinking off: the carry is inert, but the metadata fold still matters.
    Without it the trailing <total_tokens> message closed the round and the
    next prompt re-rendered the previous tool turn differently, so the banked
    prefix stopped matching inside that turn."""
    def conv(msgs):
        out = [oa.ChatMessage(role="system", content=SYSTEM["content"])]
        for m in msgs:
            out.extend(oa._anthropic_message_to_chat_messages(oa.AnthropicMessage(**m)))
        return out

    def step(n):
        call = {"type": "tool_use", "id": f"call_{n}", "name": "read_notes", "input": {"part": n}}
        result = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"call_{n}", "content": f"Part {n}."},
            {"type": "text", "text": f"<total_tokens>{1000 - n} tokens left</total_tokens>"},
        ]}
        return {"role": "assistant", "content": [call]}, result

    def enc(messages):
        return oa._encode_messages(
            tok, messages, enable_thinking=False, scoped_reasoning_history=True,
            tools=TOOLS, template_observability={},
        )

    a1, r1 = step(1)
    a2, r2 = step(2)
    before = conv([U1, a1, r1])
    prompt = enc(before)
    generated = oa._encode_rendered_chat_text(tok, TOOL_CALL_TEXT.replace(">\n1\n<", ">\n2\n<") + "<|im_end|>\n")
    call2 = {"id": "call_2", "type": "function",
             "function": {"name": "read_notes", "arguments": json.dumps({"part": 2})}}
    predicted, _ = oa._history_ids_for_postcommit(
        _state(tok), messages=before, assistant_content="", assistant_tool_calls=[call2],
        thinking_enabled=False, tool_specs=TOOLS, tool_prompt_mode="hybrid",
        committed_stream_ids=prompt + generated,
    )
    predicted = [int(t) for t in predicted]
    next_prompt = enc(conv([U1, a1, r1, a2, r2]))
    assert predicted[: len(prompt) + len(generated)] == prompt + generated
    assert next_prompt[: len(predicted)] == predicted
