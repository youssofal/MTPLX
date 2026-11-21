"""Inkling's committed turns against the next prompt.

The bank holds an answer as the ids the model wrote; the client sends the turn back as parsed
fields and Inkling's template renders them again. In a real Pi session (RC3, 21 Nov) the turn
after two big file writes read 28,522 and 35,427 tokens again (65.5 s and 128.9 s to the first
token): the model had written a 60-character `=` banner as ` =` + `========`... where the
encoder writes ` ` + `========`..., a split no 8-token window bridges, so the prompt left the
bank inside the answer. The splice now compares whole assistant turns, framed by the dialect's
control tokens, and serves the bank's ids for a turn that says the same thing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.models.inkling import protocol
from mtplx.server import openai as oa

NAMES = (
    "message_user", "message_model", "message_system", "message_tool", "content_text",
    "content_image", "content_thinking", "content_tool_error", "content_xml",
    "content_invoke_tool_json", "content_invoke_tool_text", "end_message",
    "content_model_end_sampling", "begin_of_text", "endoftext",
)
CONTROL = {name: 1000 + index for index, name in enumerate(NAMES)}
MERGED = {300: b" =", 301: b"=" * 8, 302: b"=" * 4, 303: b"\n\n"}


class InklingTokenizer:
    """Byte-level ids 0-255, a few merged ids and Inkling's control tokens."""

    inkling_protocol = True

    def convert_tokens_to_ids(self, text):
        return CONTROL[text[2:-2]]

    @property
    def all_special_ids(self):
        return list(CONTROL.values())

    def decode(self, ids):
        names = {value: key for key, value in CONTROL.items()}
        out = b""
        for token in ids:
            token = int(token)
            if token in names:
                out += f"<|{names[token]}|>".encode()
            elif token in MERGED:
                out += MERGED[token]
            else:
                out += bytes([token])
        return out.decode("utf-8", errors="replace")


def enc(text):
    """The encoder's split: `========` and `====` runs, `\\n\\n`, every other byte alone."""
    data, ids, i = text.encode(), [], 0
    while i < len(data):
        if data[i : i + 8] == b"=" * 8:
            ids.append(301)
            i += 8
        elif data[i : i + 4] == b"=" * 4:
            ids.append(302)
            i += 4
        elif data[i : i + 2] == b"\n\n":
            ids.append(303)
            i += 2
        else:
            ids.append(data[i])
            i += 1
    return ids


def model_enc(text):
    """The model's split of the same text: ` =` opens every banner."""
    out, rest = [], text
    while " =" in rest:
        head, rest = rest.split(" =", 1)
        out += enc(head) + [300]
    return out + enc(rest)


C = CONTROL
BANNER = "/* " + "=" * 60 + " */"


def segment(opener, payload, header=()):
    return [C["message_model"], *header, C[opener], *payload, C["end_message"]]


def rendered(name, args):
    """The template's bytes for a call: keys sorted, each value through compact `tojson`."""
    body = ",".join(json.dumps(key) + ":" + json.dumps(args[key], ensure_ascii=False, separators=(",", ":"))
                    for key in sorted(args))
    return '{"name":' + json.dumps(name) + ',"args":{' + body + "}}"


def tool_json(name, args, *, model=False, split=None):
    """A tool-call segment: the template's render, or the model's own bytes (keys unsorted,
    spaces, the em dash escaped) in its own split. `split` alone keeps the template's bytes."""
    if model:
        text = json.dumps({"name": name, "args": args}, ensure_ascii=True)
        return segment("content_invoke_tool_json", model_enc(text), enc(name))
    return segment("content_invoke_tool_json", (split or enc)(rendered(name, args)), enc(name))


def message(role, text, name=""):
    return [C["message_" + role], *enc(name), C["content_text"], *enc(text), C["end_message"]]


PREFIX = message("system", "You are a coding agent.") + message("user", "Make the game.")
FILE = {"path": "index.html", "content": "<script>\n" + BANNER + "\n  AURORA FLAP — the game\n</script>\n"}


def answer(*, model, reasoning="Write the file.\n", args=FILE, text=None):
    """One assistant turn: reasoning, optional text, a write call, the closing marker."""
    split = model_enc if model else enc
    parts = segment("content_thinking", split(reasoning)) if reasoning is not None else []
    if text is not None:
        parts += segment("content_text", split(text))
    return parts + tool_json("write", args, model=model) + [C["content_model_end_sampling"]]


RESULT = message("tool", "Successfully wrote the file.", name="write")


def splice(prompt, committed, **kwargs):
    return oa._splice_committed_token_ids(prompt, committed, InklingTokenizer(), **kwargs)


def test_a_long_banner_and_a_re_rendered_call_restore_the_whole_answer():
    committed = PREFIX + answer(model=True)
    prompt = PREFIX + answer(model=False) + RESULT
    tok = InklingTokenizer()
    assert tok.decode(prompt[: len(committed)]) != tok.decode(committed)  # keys, spacing, escapes
    # The window splice of every other family stops inside the answer.
    window_tok = SimpleNamespace(decode=tok.decode)
    window_out, window_receipt = oa._splice_committed_token_ids(prompt, committed, window_tok)
    assert oa.common_prefix_len(window_out, committed) < len(committed)
    out, receipt = splice(prompt, committed)
    assert out == committed + RESULT
    assert receipt["spans"] == 1 and receipt["cp_after"] == len(committed)
    assert "stopped" not in receipt and "put_back_turns" not in receipt


def test_the_banner_seam_alone_restores_and_the_text_is_unchanged():
    """The RC3 case: the model wrote the template's very bytes, split its own way."""
    think = segment("content_thinking", enc("Write the file.\n"))
    committed = PREFIX + think + tool_json("write", FILE, split=model_enc) + [C["content_model_end_sampling"]]
    prompt = PREFIX + think + tool_json("write", FILE) + [C["content_model_end_sampling"]] + RESULT
    tok = InklingTokenizer()
    sent_turns = prompt[: len(prompt) - len(RESULT)]
    assert tok.decode(committed) == tok.decode(sent_turns) and committed != sent_turns
    window_out, _ = oa._splice_committed_token_ids(prompt, committed, SimpleNamespace(decode=tok.decode))
    assert oa.common_prefix_len(window_out, committed) < len(committed)
    out, receipt = splice(prompt, committed)
    assert out == committed + RESULT and receipt["cp_after"] == len(committed)
    assert tok.decode(out) == tok.decode(prompt)


def test_a_changed_argument_ends_the_walk_and_the_prompt_is_served_as_sent():
    committed = PREFIX + answer(model=True)
    changed = {**FILE, "content": FILE["content"].replace("game", "gam3")}
    prompt = PREFIX + answer(model=False, args=changed) + RESULT
    out, receipt = splice(prompt, committed)
    assert out == prompt
    assert receipt["spans"] == 0 and receipt["stopped"] == "tool_call_changed"


@pytest.mark.parametrize(
    ("banked", "sent", "same"),
    [
        ({"n": 1}, {"n": 1.0}, True),
        ({"n": 1}, {"n": "1"}, False),
        ({"n": True}, {"n": 1}, False),
        ({"n": None}, {"n": False}, False),
        ({"a": [1, {"b": "x"}]}, {"a": [1, {"b": "x"}]}, True),
        ({"a": [1, 2]}, {"a": [2, 1]}, False),
        ({"a": "x"}, {"a": "x", "b": "y"}, False),
    ],
)
def test_arguments_compare_as_json_values_with_their_types(banked, sent, same):
    committed = PREFIX + [*segment("content_thinking", enc("t")), *tool_json("bash", banked, model=True),
                          C["content_model_end_sampling"]]
    prompt = PREFIX + [*segment("content_thinking", enc("t")), *tool_json("bash", sent),
                       C["content_model_end_sampling"]] + RESULT
    out, receipt = splice(prompt, committed)
    if same:
        assert out == committed + RESULT
    else:
        assert receipt["spans"] == 0 and receipt["stopped"] == "tool_call_changed"


def test_a_repeated_key_in_the_models_json_is_never_taken_as_the_sent_call():
    text = '{"name":"bash","args":{"command":"ls","command":"rm -rf x"}}'
    committed = PREFIX + [*segment("content_invoke_tool_json", enc(text), enc("bash")), C["content_model_end_sampling"]]
    prompt = PREFIX + [*tool_json("bash", {"command": "rm -rf x"}), C["content_model_end_sampling"]] + RESULT
    out, receipt = splice(prompt, committed)
    assert out == prompt and receipt["stopped"] == "turn_unreadable"


def test_a_different_tool_name_refuses():
    committed = PREFIX + [*tool_json("bash", {"command": "ls"}, model=True), C["content_model_end_sampling"]]
    prompt = PREFIX + [*tool_json("read", {"command": "ls"}), C["content_model_end_sampling"]] + RESULT
    out, receipt = splice(prompt, committed)
    assert out == prompt and receipt["spans"] == 0


def test_reasoning_the_client_dropped_comes_back_only_in_preserve_mode():
    committed = PREFIX + answer(model=True)
    prompt = PREFIX + answer(model=False, reasoning=None) + RESULT
    out, receipt = splice(prompt, committed)
    assert out == prompt and receipt["stopped"] == "reasoning_changed"
    out, receipt = splice(prompt, committed, substitute_reasoning=True)
    assert out == committed + RESULT
    assert receipt["put_back_turns"] == 1 and receipt["cp_after"] == len(committed)


def test_changed_visible_text_refuses_unless_the_gate_normalizes_both_sides():
    committed = PREFIX + answer(model=True, text="I'll write it now.")
    prompt = PREFIX + answer(model=False, text="") + RESULT
    out, receipt = splice(prompt, committed, substitute_reasoning=True)
    assert out == prompt and receipt["stopped"] == "text_changed"

    def drops_preamble(text, has_tool_calls):
        return "" if has_tool_calls else text.strip()

    out, receipt = splice(prompt, committed, substitute_reasoning=True, visible_gate=drops_preamble)
    assert out == committed + RESULT and receipt["put_back_turns"] == 1
    # The gate's normalizer is the substitution contract: a plain splice never uses it.
    out, receipt = splice(prompt, committed, visible_gate=drops_preamble)
    assert out == prompt and receipt["stopped"] == "text_changed"


def test_whitespace_the_response_stripped_is_not_a_change():
    committed = PREFIX + answer(model=True, reasoning="Plan.\n\n", text="Done.\n")
    prompt = PREFIX + answer(model=False, reasoning="Plan.", text="Done.") + RESULT
    out, receipt = splice(prompt, committed)
    assert out == committed + RESULT and "put_back_turns" not in receipt


def test_a_changed_tool_result_is_read_again_from_where_it_changed():
    committed = PREFIX + answer(model=True) + RESULT + answer(model=True)
    edited = message("tool", "Wrote the file (edited by the client).", name="write")
    prompt = PREFIX + answer(model=False) + edited + answer(model=False) + RESULT
    out, receipt = splice(prompt, committed)
    first_answer_end = len(PREFIX) + len(answer(model=True))
    sent_answer_end = len(PREFIX) + len(answer(model=False))
    assert out == committed[:first_answer_end] + prompt[sent_answer_end:]
    assert receipt["spans"] == 1 and receipt["stopped"] == "outside_assistant_turn"
    assert receipt["cp_after"] == oa.common_prefix_len(out, committed)


def test_every_matching_turn_is_served_and_the_first_changed_one_ends_the_walk():
    second = {"path": "b.js", "content": BANNER + "\nlet x = 1;\n"}
    third = {"path": "c.js", "content": "x"}
    committed = (PREFIX + answer(model=True) + RESULT + answer(model=True, args=second) + RESULT
                 + answer(model=True, args=third))
    prompt = (PREFIX + answer(model=False) + RESULT + answer(model=False, args=second) + RESULT
              + answer(model=False, args={**third, "content": "y"}) + RESULT)
    out, receipt = splice(prompt, committed)
    served = len(PREFIX) + len(answer(model=True)) + len(RESULT) + len(answer(model=True, args=second)) + len(RESULT)
    sent = len(PREFIX) + len(answer(model=False)) + len(RESULT) + len(answer(model=False, args=second)) + len(RESULT)
    assert out == committed[:served] + prompt[sent:]
    assert receipt["spans"] == 2 and receipt["stopped"] == "tool_call_changed"


def test_an_answer_cut_at_its_length_restores_up_to_where_the_model_stopped():
    cut = segment("content_thinking", model_enc("Plan the banner " + "=" * 60))[:-1]  # no end marker
    committed = PREFIX + cut
    prompt = PREFIX + segment("content_thinking", enc("Plan the banner " + "=" * 60)) + [C["content_model_end_sampling"]] + RESULT
    out, receipt = splice(prompt, committed)
    assert out == committed + [C["end_message"], C["content_model_end_sampling"]] + RESULT
    assert receipt["cp_after"] == len(committed)


def test_a_stream_already_extended_is_left_alone():
    committed = PREFIX + answer(model=False)
    prompt = committed + RESULT
    assert splice(prompt, committed) == (prompt, {"spans": 0, "tokens_in": 0, "tokens_out": 0})


def _gate(committed, *, strip=False):
    session = SimpleNamespace(committed_token_ids=tuple(committed))
    return SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=strip),
        sessions=SimpleNamespace(
            resolve_session_id=lambda **kw: ("s1", "header.x-mtplx-session-id"),
            peek=lambda sid: session if sid == "s1" else None,
        ),
        runtime=SimpleNamespace(tokenizer=InklingTokenizer()),
    )


def _run_gate(state, prompt):
    request = oa.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "Make the game."}])
    observed: dict = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state, messages=request.messages, prompt_ids=list(prompt), headers={}, metadata={},
        request=request, thinking_enabled=True, reasoning_effort="xhigh", tools=None, tool_choice=None,
        tool_prompt_mode="native", template_observability={}, request_observability=observed,
    )
    return result, observed.get("committed_reasoning_canonicalization")


def test_the_gate_serves_the_banked_turn_and_puts_back_dropped_reasoning(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    committed = PREFIX + answer(model=True)
    prompt = PREFIX + answer(model=False, reasoning=None) + RESULT
    result, outcome = _run_gate(_gate(committed), prompt)
    assert result is not None and result[1] == committed + RESULT
    assert outcome["applied"] is True and outcome["cp_spliced"] == len(committed)
    assert outcome["token_splice"]["put_back_turns"] == 1


def test_the_gate_never_puts_reasoning_back_when_history_is_stripped(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    committed = PREFIX + answer(model=True)
    prompt = PREFIX + answer(model=False, reasoning=None) + RESULT
    result, outcome = _run_gate(_gate(committed, strip=True), prompt)
    assert result is None
    assert outcome["declined"] == "reasoning_history_stripped"
    assert outcome["token_splice"]["stopped"] == "reasoning_changed"
    # A turn that kept its reasoning still restores in that mode.
    prompt = PREFIX + answer(model=False) + RESULT
    result, outcome = _run_gate(_gate(committed, strip=True), prompt)
    assert result is not None and result[1] == committed + RESULT


def test_the_gate_names_why_a_changed_turn_was_read_again(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    committed = PREFIX + answer(model=True)
    prompt = PREFIX + answer(model=False, args={**FILE, "path": "other.html"}) + RESULT
    result, outcome = _run_gate(_gate(committed), prompt)
    assert result is None
    assert outcome["applied"] is False and outcome["token_splice"]["stopped"] == "tool_call_changed"


def test_the_generation_final_snapshot_accepts_the_re_rendered_turn(monkeypatch):
    from tests.test_openai_bridge import _final_state, _postcommit_state

    generated = answer(model=True)
    history = PREFIX + answer(model=False)
    monkeypatch.setattr(oa, "_history_ids_for_postcommit", lambda *a, **k: (list(history), None))
    state = _postcommit_state(tokenizer=InklingTokenizer())
    result = oa._store_generation_final_history_snapshot(
        state,
        session_id="session-1",
        prompt_ids=list(PREFIX),
        generated={"tokens": generated, "_final_state": _final_state(generated)},
        messages=[oa.ChatMessage(role="user", content="Make the game.")],
        assistant_content="",
        thinking_enabled=True,
        policy_fingerprint="policy",
    )
    assert result["stored"] is True
    assert result["reason"] == "token_identical_after_splice"
    assert state.sessions.bank.puts[0]["token_ids"] == PREFIX + generated


def test_same_value_is_json_equality_with_types():
    assert protocol.same_value({"a": [1, 2.0, "x"]}, {"a": [1.0, 2, "x"]})
    assert not protocol.same_value({"a": 0}, {"a": False})
    assert not protocol.same_value({"a": float("nan")}, {"a": float("nan")})
    assert not protocol.same_value([], {})


PACK = Path.home() / "Models/mtplx/MTPLX-Inkling-2-Mini-Bare-Speed"


@pytest.mark.skipif(not (PACK / "tokenizer.json").is_file(), reason="needs the Inkling pack's tokenizer")
def test_the_real_template_and_tokenizer_restore_a_banner_turn():
    """The RC3 mechanism with the pack's own tokenizer and chat template: the model's ` =` split
    of a 60-character banner defeats the window splice and the turn splice restores the turn."""
    from mtplx.runtime import _load_tokenizer_resilient

    tok = _load_tokenizer_resilient(PACK, json.loads((PACK / "config.json").read_text()))
    tok.inkling_protocol = True
    tools = [{"type": "function", "function": {"name": "write", "description": "Write a file",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}}]
    call = {"id": "c1", "type": "function", "function": {"name": "write", "arguments": json.dumps(FILE)}}
    turn = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Make the game."}]
    before = protocol.encode(tok, turn, tools=tools, effort="xhigh", thinking=True, strip_reasoning=False,
                             add_generation_prompt=True)
    history = turn + [{"role": "assistant", "content": None, "reasoning_content": "Write it.", "tool_calls": [call]},
                      {"role": "tool", "content": "Wrote it.", "tool_call_id": "c1"}]
    after = protocol.encode(tok, history, tools=tools, effort="xhigh", thinking=True, strip_reasoning=False,
                            add_generation_prompt=True)
    end = tok.convert_tokens_to_ids("<|content_model_end_sampling|>")
    canonical = after[len(before) : after.index(end, len(before)) + 1]
    space, space_eq = tok.encode(" ", add_special_tokens=False), tok.encode(" =", add_special_tokens=False)
    run = tok.encode("=" * 60, add_special_tokens=False)
    at = next(i for i in range(len(canonical)) if canonical[i : i + 1] == space and canonical[i + 1 : i + 1 + len(run)] == run)
    model = canonical[:at] + space_eq + tok.encode("=" * 59, add_special_tokens=False) + canonical[at + 1 + len(run):]
    assert tok.decode(model) == tok.decode(canonical) and model != canonical
    committed = before + model
    window_out, _ = oa._splice_committed_token_ids(after, committed, SimpleNamespace(decode=tok.decode))
    assert oa.common_prefix_len(window_out, committed) < len(committed)
    out, receipt = oa._splice_committed_token_ids(after, committed, tok)
    assert out[: len(committed)] == committed and receipt["cp_after"] == len(committed)
    assert tok.decode(out) == tok.decode(after)
