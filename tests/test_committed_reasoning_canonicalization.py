"""Session-owned committed-think canonicalization (defect B, 2.8 headline).

The committed stream holds the model's real <think> bytes; clients resend
history with reasoning elided or summarized, so the re-encoded prompt
diverges at the FIRST assistant turn's think block and restores freeze at
the turn-1 boundary while context grows (founder's OpenCode Desktop x MTPLX
Desktop session, LOG 2026-08-16 09:15 BST). The canonicalizer substitutes
the committed think bytes before encode — served only when the substituted
encode provably matches the committed stream further than the raw one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.server import openai as oa

MODEL_DIR = Path.home() / ".mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed"


def _committed_text(turns):
    parts = ["<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"]
    for think, content in turns:
        parts.append(
            "<|im_start|>assistant\n<think>\n"
            + (think + "\n" if think else "")
            + "</think>\n\n"
            + content
            + "<|im_end|>\n"
        )
        parts.append("<|im_start|>user\nnext<|im_end|>\n")
    return "".join(parts)


def test_committed_assistant_turns_parses_interiors_and_gates():
    text = _committed_text(
        [
            ("I should call the tool.", '<tool_call>{"name": "glob"}</tool_call>'),
            ("Now answer plainly.", "The answer is 42."),
        ]
    )
    turns = oa._committed_assistant_turns(text)
    assert len(turns) == 2
    assert turns[0][0] == "I should call the tool."
    assert turns[0][1] == ""  # tool-call markup excluded from the gate
    assert turns[0][2] == '<tool_call>{"name": "glob"}</tool_call>'
    assert turns[1][0] == "Now answer plainly."
    assert turns[1][1] == "The answer is 42."
    assert turns[1][2] == ""  # no tool markup on the plain turn


def test_committed_assistant_turns_handles_missing_think():
    text = (
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nplain, no think.<|im_end|>\n"
    )
    turns = oa._committed_assistant_turns(text)
    assert turns == [(None, "plain, no think.", "")]


def _fake_state(committed_ids, committed_text, session_id="s1"):
    session = SimpleNamespace(committed_token_ids=tuple(committed_ids))
    sessions = SimpleNamespace(
        resolve_session_id=lambda **kw: (session_id, "header.x-mtplx-session-id"),
        peek=lambda sid: session if sid == session_id else None,
    )
    committed_list = [int(i) for i in committed_ids]

    def _decode(ids):
        # The whole committed stream decodes to the transcript; any other run
        # decodes to one distinct marker per id, so two runs of different ids
        # never read as the same text (the token splice compares windows).
        ids = [int(i) for i in ids]
        if ids == committed_list:
            return committed_text
        return "".join(f"<{i}>" for i in ids)

    tokenizer = SimpleNamespace(decode=_decode)
    return SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=sessions,
        runtime=SimpleNamespace(tokenizer=tokenizer),
    )


def _canonicalize(state, messages, prompt_ids, monkeypatch, canon_ids):
    captured: dict[str, object] = {}

    def _fake_encode(tokenizer, msgs, **kwargs):
        captured["messages"] = msgs
        captured["allow_committed_reasoning"] = kwargs.get(
            "allow_committed_reasoning"
        )
        return list(canon_ids)

    monkeypatch.setattr(oa, "_encode_messages", _fake_encode)
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    observability: dict[str, object] = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state,
        messages=request.messages,
        prompt_ids=list(prompt_ids),
        headers={},
        metadata={},
        request=request,
        thinking_enabled=True,
        reasoning_effort="xhigh",
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        request_observability=observability,
    )
    return result, captured, observability


def test_canonicalizer_substitutes_and_serves_on_cp_improvement(monkeypatch):
    committed = list(range(100, 200))
    text = _committed_text([("Real think bytes.", "The answer is 42.")])
    state = _fake_state(committed, text)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "The answer is 42."},
        {"role": "user", "content": "next"},
    ]
    raw_ids = committed[:10] + [1, 2, 3]  # diverges inside the committed stream
    canon_ids = committed[:80] + [7, 8, 9]  # extends much further
    result, captured, observability = _canonicalize(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is not None
    canon_messages, served_ids = result
    assert served_ids == canon_ids
    assert captured["allow_committed_reasoning"] is True
    substituted = [
        oa._message_extra(m, oa._COMMITTED_REASONING_FIELD)
        for m in canon_messages
        if m.role == "assistant"
    ]
    assert substituted == ["Real think bytes."]
    outcome = observability["committed_reasoning_canonicalization"]
    assert outcome["applied"] is True
    assert outcome["turns_substituted"] == 1
    assert outcome["cp_canon"] > outcome["cp_raw"]


def test_canonicalizer_declines_when_cp_not_improved(monkeypatch):
    committed = list(range(100, 200))
    text = _committed_text([("Real think bytes.", "The answer is 42.")])
    state = _fake_state(committed, text)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "The answer is 42."},
        {"role": "user", "content": "next"},
    ]
    raw_ids = committed[:10] + [1, 2, 3]
    canon_ids = committed[:10] + [4, 5, 6]  # no improvement
    result, _captured, observability = _canonicalize(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is None
    outcome = observability["committed_reasoning_canonicalization"]
    assert outcome["applied"] is False
    assert outcome["cp_canon"] == outcome["cp_raw"]


def test_canonicalizer_stops_at_rewritten_turn(monkeypatch):
    committed = list(range(100, 200))
    text = _committed_text(
        [("Think one.", "First answer."), ("Think two.", "Second answer.")]
    )
    state = _fake_state(committed, text)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "REWRITTEN by the client."},
        {"role": "user", "content": "more"},
    ]
    raw_ids = committed[:10] + [1]
    canon_ids = committed[:50] + [2]
    result, _captured, _obs = _canonicalize(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is not None
    canon_messages, _ids = result
    fields = [
        oa._message_extra(m, oa._COMMITTED_REASONING_FIELD)
        for m in canon_messages
        if m.role == "assistant"
    ]
    assert fields[0] == "Think one."
    assert fields[1] is None, "substitution must stop at the rewritten turn"


def test_inbound_committed_reasoning_field_is_scrubbed(monkeypatch):
    committed = list(range(100, 200))
    text = _committed_text([("Server truth.", "First answer.")])
    state = _fake_state(committed, text)
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "First answer.",
            oa._COMMITTED_REASONING_FIELD: "client-planted lie",
        },
        {"role": "user", "content": "next"},
    ]
    raw_ids = committed[:10] + [1]
    canon_ids = committed[:60] + [2]
    result, _captured, _obs = _canonicalize(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is not None
    canon_messages, _ids = result
    values = [
        oa._message_extra(m, oa._COMMITTED_REASONING_FIELD)
        for m in canon_messages
        if m.role == "assistant"
    ]
    assert values == ["Server truth."], values


def test_canonicalizer_inert_without_committed_session(monkeypatch):
    state = _fake_state([], "")
    state.sessions = SimpleNamespace(
        resolve_session_id=lambda **kw: ("anon", "new"),
        peek=lambda sid: None,
    )
    messages = [{"role": "user", "content": "hi"}]
    result, _captured, observability = _canonicalize(
        state, messages, [1, 2, 3], monkeypatch, [1, 2, 3]
    )
    assert result is None
    assert "committed_reasoning_canonicalization" not in observability


def test_message_to_template_dict_gates_committed_field_on_flag():
    message = oa.ChatMessage(
        role="assistant",
        content="Answer.",
        **{oa._COMMITTED_REASONING_FIELD: "Committed think."},
    )
    plain = oa._message_to_template_dict(
        message, strip_assistant_reasoning_history=False
    )
    assert "reasoning_content" not in plain, (
        "legacy preserve mode must stay byte-identical when the flag is off"
    )
    allowed = oa._message_to_template_dict(
        message,
        strip_assistant_reasoning_history=False,
        allow_committed_reasoning=True,
    )
    assert allowed["reasoning_content"] == "Committed think."


@pytest.mark.skipif(
    not (MODEL_DIR / "chat_template.jinja").exists(),
    reason="Qwen3.8 model pack not cached locally",
)
def test_canonicalized_encode_extends_committed_stream_real_template():
    """End-to-end with the real tokenizer: a committed stream built from the
    real render plus generated bytes; the canonicalized re-encode must extend
    it past the first assistant turn while the raw encode diverges at the
    empty think scaffold. This is liveqa/encode_divergence_repro.py promoted
    to a regression test."""
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    tok = _load_tokenizer_resilient(MODEL_DIR, config)

    system = {"role": "system", "content": "You are a terse coding assistant."}
    u1 = {"role": "user", "content": "Read calc.py and summarize it."}
    think = "The user wants a summary of calc.py. I will answer from memory."
    answer = "calc.py defines add, sub and mul - three arithmetic helpers."
    u2 = {"role": "user", "content": "Now add a divide function."}

    def encode(messages, allow=False):
        obs: dict[str, object] = {}
        request = oa.ChatCompletionRequest(model="m", messages=messages)
        return oa._encode_messages(
            tok,
            request.messages,
            enable_thinking=True,
            reasoning_effort="xhigh",
            strip_assistant_reasoning_history=False,
            scoped_reasoning_history=False,
            tools=None,
            tool_choice=None,
            template_observability=obs,
            allow_committed_reasoning=allow,
        )

    # Committed stream = turn-1 prompt (ends with the open think scaffold)
    # + the generated bytes, exactly as EngineSession.commit stores them.
    r1_ids = encode([system, u1])
    generated = oa._encode_rendered_chat_text(
        tok, f"{think}\n</think>\n\n{answer}<|im_end|>\n"
    )
    committed = list(r1_ids) + list(generated)

    history = [system, u1, {"role": "assistant", "content": answer}, u2]
    raw_ids = encode(history)
    cp_raw = oa.common_prefix_len(raw_ids, committed)

    canon_history = [
        system,
        u1,
        {
            "role": "assistant",
            "content": answer,
            oa._COMMITTED_REASONING_FIELD: think,
        },
        u2,
    ]
    canon_ids = encode(canon_history, allow=True)
    cp_canon = oa.common_prefix_len(canon_ids, committed)

    assert cp_raw < len(committed) - len(generated) + 8, (
        f"raw encode unexpectedly matched deep into the committed stream "
        f"(cp_raw={cp_raw}, committed={len(committed)})"
    )
    assert cp_canon > cp_raw, (
        f"canonicalized encode must extend the committed stream further: "
        f"cp_canon={cp_canon} cp_raw={cp_raw}"
    )
    assert cp_canon >= len(committed) - 4, (
        f"canonicalized encode should track the committed frontier: "
        f"cp_canon={cp_canon} committed={len(committed)}"
    )


# ---------------------------------------------------------------------------
# Committed tool-call body substitution (2026-09-03)
# ---------------------------------------------------------------------------

_WRITE_MARKUP = (
    "<tool_call>\n<function=write>\n<parameter=filePath>\nindex.html\n</parameter>\n"
    "<parameter=content>\n<!DOCTYPE html>\n<html></html>\n\n</parameter>\n</function>\n</tool_call>"
)


def _write_turn_committed_text(visible: str = "I'll write the file."):
    return (
        "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nmake it<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nPlan the file.\n</think>\n\n"
        + (visible + "\n\n" if visible else "")
        + _WRITE_MARKUP
        + "<|im_end|>\n"
    )


def _write_turn_message(visible: str = "I'll write the file."):
    # What the client echoes back: the parser stripped the content's own
    # trailing newline, so a re-render from these arguments is one token
    # short of the generated stream (the measured 271 "\n\n" -> 198 "\n").
    return oa.ChatMessage(
        role="assistant",
        content=visible,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps(
                        {"filePath": "index.html", "content": "<!DOCTYPE html>\n<html></html>"}
                    ),
                },
            }
        ],
    )


def test_committed_turns_carry_the_exact_post_think_body():
    turns = oa._committed_assistant_turns(_write_turn_committed_text())
    assert len(turns) == 1
    interior, gate, markup = turns[0]  # 3-tuple contract intact
    assert interior == "Plan the file."
    assert gate == "I'll write the file."
    assert markup == _WRITE_MARKUP
    assert turns[0].body == "\n\nI'll write the file.\n\n" + _WRITE_MARKUP
    assert turns[0] == (interior, gate, markup)


def test_tool_turn_substitution_attaches_the_committed_body():
    turns = oa._committed_assistant_turns(_write_turn_committed_text())
    messages = [
        oa.ChatMessage(role="system", content="sys"),
        oa.ChatMessage(role="user", content="make it"),
        _write_turn_message(),
        oa.ChatMessage(role="tool", content="ok", tool_call_id="call_1"),
    ]
    canon, substituted = oa._substitute_committed_reasoning_messages(messages, turns)
    assert substituted == 1
    assistant = canon[2]
    assert oa._message_extra(assistant, oa._COMMITTED_REASONING_FIELD) == "Plan the file."
    assert oa._message_extra(assistant, oa._COMMITTED_TURN_BODY_FIELD) == (
        "\n\nI'll write the file.\n\n" + _WRITE_MARKUP
    )
    item = oa._message_to_template_dict(
        assistant, strip_assistant_reasoning_history=False, allow_committed_reasoning=True
    )
    # The template renders the committed bytes as content and must NOT get
    # tool_calls to re-render (they would appear twice, and lossy).
    assert item["content"] == "I'll write the file.\n\n" + _WRITE_MARKUP
    assert "tool_calls" not in item
    assert item["reasoning_content"] == "Plan the file."
    # Without the opt-in flag the legacy render is untouched.
    legacy = oa._message_to_template_dict(assistant, strip_assistant_reasoning_history=False)
    assert legacy["content"] == "I'll write the file."
    assert legacy["tool_calls"]


def test_tool_turn_substitution_respects_the_identity_gate():
    turns = oa._committed_assistant_turns(_write_turn_committed_text())
    other = _write_turn_message()
    other.tool_calls[0]["function"]["arguments"] = json.dumps(
        {"filePath": "other.html", "content": "x"}
    )
    messages = [oa.ChatMessage(role="user", content="make it"), other]
    canon, substituted = oa._substitute_committed_reasoning_messages(messages, turns)
    assert substituted == 0
    assert oa._message_extra(canon[1], oa._COMMITTED_TURN_BODY_FIELD) is None


def test_inbound_committed_body_field_is_scrubbed():
    planted = oa.ChatMessage(
        role="assistant",
        content="Answer.",
        **{oa._COMMITTED_TURN_BODY_FIELD: "<tool_call>evil</tool_call>"},
    )
    scrubbed = oa._scrub_inbound_committed_reasoning(planted)
    assert oa._message_extra(scrubbed, oa._COMMITTED_TURN_BODY_FIELD) is None


@pytest.mark.skipif(
    not (MODEL_DIR / "chat_template.jinja").exists(),
    reason="Qwen3.8 model pack not cached locally",
)
def test_substituted_write_turn_re_encodes_to_the_generated_bytes_real_template():
    """The measured 2026-09-03 seam, end to end with the real tokenizer: a
    write turn whose content ended in a newline. Raw re-render loses the
    newline (one token short, refused); the substituted render reproduces
    the generated stream so the generation-final snapshot can be banked."""
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    tok = _load_tokenizer_resilient(MODEL_DIR, config)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "write a file",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["filePath", "content"],
                },
            },
        }
    ]
    system = {"role": "system", "content": "You are a terse coding assistant."}
    u1 = {"role": "user", "content": "make it"}

    def encode(messages, allow=False):
        obs: dict[str, object] = {}
        request = oa.ChatCompletionRequest(model="m", messages=messages)
        return oa._encode_messages(
            tok,
            request.messages,
            enable_thinking=True,
            reasoning_effort="medium",
            strip_assistant_reasoning_history=False,
            scoped_reasoning_history=False,
            tools=tools,
            tool_choice="auto",
            tool_prompt_mode="compact",
            template_observability=obs,
            allow_committed_reasoning=allow,
        )

    prompt_ids = encode([system, u1])
    generated = oa._encode_rendered_chat_text(
        tok, "Plan the file.\n</think>\n\nI'll write the file.\n\n" + _WRITE_MARKUP + "<|im_end|>\n"
    )
    committed = list(prompt_ids) + list(generated)
    committed_text = tok.decode(committed)
    turns = oa._committed_assistant_turns(committed_text)
    assert turns and turns[-1][2] == _WRITE_MARKUP

    history = [
        oa.ChatMessage(role="system", content=system["content"]),
        oa.ChatMessage(role="user", content=u1["content"]),
        _write_turn_message(),
        oa.ChatMessage(role="tool", content="ok", tool_call_id="call_1"),
    ]
    raw_ids = encode(history)
    canon, substituted = oa._substitute_committed_reasoning_messages(history, turns)
    assert substituted == 1
    canon_ids = encode(canon, allow=True)
    cp_raw = oa.common_prefix_len(raw_ids, committed)
    cp_canon = oa.common_prefix_len(canon_ids, committed)
    assert cp_raw < len(committed), "raw re-render is expected to lose the trailing newline"
    assert cp_canon >= len(committed) - 1, (cp_canon, len(committed))


# ---------------------------------------------------------------------------
# 2026-09-08: the gate compares the COMPLETE tool-call identity, not the loop
# key (command / path only). Codex's audit reproduced a write to the same path
# with new content passing the old gate, after which the committed OLD body
# was served in place of the client's NEW content.
# ---------------------------------------------------------------------------


def test_same_path_different_content_refuses_the_committed_body():
    turns = oa._committed_assistant_turns(_write_turn_committed_text())
    changed = _write_turn_message()
    changed.tool_calls[0]["function"]["arguments"] = json.dumps(
        {"filePath": "index.html", "content": "NEW_FIXED_IMPLEMENTATION"}
    )
    messages = [oa.ChatMessage(role="user", content="make it"), changed]
    canon, substituted = oa._substitute_committed_reasoning_messages(messages, turns)
    assert substituted == 0
    assert oa._message_extra(canon[1], oa._COMMITTED_TURN_BODY_FIELD) is None
    assert oa._message_extra(canon[1], oa._COMMITTED_REASONING_FIELD) is None


def test_identity_ignores_edge_whitespace_and_scalar_typing():
    # committed <parameter> values are strings with one newline stripped;
    # the client echoes typed JSON. Same call -> same identity.
    committed = "<tool_call>\n<function=read><parameter=filePath>\na.py\n</parameter><parameter=offset>10</parameter><parameter=limit>20</parameter></function>\n</tool_call>"
    incoming = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"filePath": "a.py", "offset": 10, "limit": 20})},
        }
    ]
    assert oa._committed_turn_tool_identities(committed) == oa._incoming_tool_identities(incoming)


def test_identity_sees_nested_json_parameters_the_same_way():
    edits = [{"oldString": "a", "newString": "b"}, {"oldString": "c", "newString": "d"}]
    committed = (
        "<tool_call>\n<function=multiedit><parameter=filePath>x.py</parameter>"
        f"<parameter=edits>{json.dumps(edits)}</parameter></function>\n</tool_call>"
    )
    incoming = [
        {"id": "c1", "type": "function", "function": {"name": "multiedit", "arguments": json.dumps({"filePath": "x.py", "edits": edits})}}
    ]
    assert oa._committed_turn_tool_identities(committed) == oa._incoming_tool_identities(incoming)
    other = [
        {"id": "c1", "type": "function", "function": {"name": "multiedit", "arguments": json.dumps({"filePath": "x.py", "edits": edits[:1]})}}
    ]
    assert oa._committed_turn_tool_identities(committed) != oa._incoming_tool_identities(other)


def test_identity_covers_the_json_dialect_too():
    committed = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la", "timeout": 30}}\n</tool_call>'
    same = [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"timeout": 30, "command": "ls -la"})}}]
    different_timeout = [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"timeout": 60, "command": "ls -la"})}}]
    assert oa._committed_turn_tool_identities(committed) == oa._incoming_tool_identities(same)
    assert oa._committed_turn_tool_identities(committed) != oa._incoming_tool_identities(different_timeout)
    # the loop key would have called these the same call
    assert oa._committed_turn_tool_keys(committed) == oa._incoming_tool_loop_keys(different_timeout)
