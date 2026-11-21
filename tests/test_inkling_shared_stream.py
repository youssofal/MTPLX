"""A resent history whose conversation has no stream under its own id.

Hermes sends no session id: in a running server the registry finds its conversation by the prompt,
but after a restart (or an idle eviction) its next request is a new session, so the gate had no
stream to compare with and the history it resent left the bank at the first answer whose ids the
splice once served. Everything after that answer was read again (T1, 21 Nov). A client that forks a
conversation under a new id is the same case. The gate now takes the bank's streams that share the
prompt into its first answer, whatever conversation stored them, and serves their turns only where
they say exactly what the client sent: reasoning included, nothing put back.
"""

from __future__ import annotations

from types import SimpleNamespace

from mtplx.cache_bank import SessionBankColdTier
from mtplx.models.inkling import protocol
from mtplx.server import openai as oa

from tests.test_inkling_committed_turns import (
    FILE, PREFIX, RESULT, InklingTokenizer, answer, message,
)
from tests.test_session_stream_after_restart import MODEL, _bank, _put

BANKED = PREFIX + answer(model=True)  # the conversation as the bank holds it: the model's own ids
RESENT = PREFIX + answer(model=False) + RESULT  # as Hermes sends it after the restart


def _state(shared, *, tokenizer=None, calls=None, registry=None):
    """A gate state that resolves a new anonymous session (the registry holds nothing for it,
    unless `registry`), whose bank holds nothing under that id and `shared` by shared prefix."""

    def shared_streams(prompt, *, model_path, min_shared, limit=4):
        if calls is not None:
            calls.append(min_shared)
        return [tuple(s) for s in shared if oa.common_prefix_len(prompt, s) >= min_shared][:limit]

    bank = SimpleNamespace(session_streams=lambda session_id, **kw: [],
                           session_stream=lambda session_id, **kw: (),
                           shared_streams=shared_streams)
    session = SimpleNamespace(committed_token_ids=tuple(registry)) if registry is not None else None
    return SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=SimpleNamespace(resolve_session_id=lambda **kw: ("anon-new", "new"),
                                 peek=lambda sid: session, bank=bank),
        runtime=SimpleNamespace(tokenizer=tokenizer or InklingTokenizer(), model_path=MODEL),
    )


def _gate(state, prompt, *, thinking=True, compared=None):
    request = oa.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "Make the game."}])
    observed: dict = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state, messages=request.messages, prompt_ids=list(prompt), headers={}, metadata={}, request=request,
        thinking_enabled=thinking, reasoning_effort="xhigh", tools=None, tool_choice=None,
        tool_prompt_mode="native", template_observability={}, request_observability=observed,
        compared_stream=compared,
    )
    return result, observed.get("committed_reasoning_canonicalization")


def test_an_anonymous_conversation_after_a_restart_is_served_from_the_stream_it_shares(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    compared: list = []
    result, outcome = _gate(_state([BANKED]), RESENT, compared=compared)
    assert result is not None and result[1] == BANKED + RESULT
    assert outcome["applied"] is True and outcome["committed_source"] == "bank_prefix"
    assert outcome["cp_spliced"] == len(BANKED)
    assert compared == [tuple(BANKED)]  # the image gate judges a repair against the same stream


def test_another_conversations_reasoning_is_never_put_back(monkeypatch):
    """The same task and the same call, written after other reasoning: in the conversation's own
    stream the gate puts the model's reasoning back; a stream found by its prefix alone may be
    another conversation's, so the turn is served only when it says exactly what was sent."""
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    other = PREFIX + answer(model=True, reasoning="Write the file, then test it.\n")
    result, outcome = _gate(_state([other]), RESENT)
    assert result is None
    assert outcome["applied"] is False and outcome["token_splice"]["stopped"] == "reasoning_changed"
    # The conversation's own stream (found under its id, or held by the registry) still puts it back.
    result, outcome = _gate(_state([], registry=other), RESENT)
    assert result is not None and outcome["token_splice"]["put_back_turns"] == 1


def test_the_stream_the_walk_carries_furthest_wins_over_the_longest_shared_prefix(monkeypatch):
    """Another conversation wrote the same file with a different last line, in the encoder's own
    split: it shares the resent prompt further than the true one (whose own split differs at the
    call's first key), but its call is not the one sent."""
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    changed = dict(FILE, content=FILE["content"] + "<!-- v2 -->\n")
    decoy = PREFIX + answer(model=False, args=changed)
    assert oa.common_prefix_len(RESENT, decoy) > oa.common_prefix_len(RESENT, BANKED)
    result, outcome = _gate(_state([decoy, BANKED]), RESENT)
    assert result is not None and result[1] == BANKED + RESULT
    assert outcome["committed_source"] == "bank_prefix" and outcome["committed_len"] == len(BANKED)


def test_a_first_request_never_probes_the_bank(monkeypatch):
    """No finished answer in the history: nothing a stored stream could stand for."""
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    calls: list = []
    first = PREFIX + [protocol.dialect(InklingTokenizer()).model]  # the template opens the answer
    result, outcome = _gate(_state([BANKED], calls=calls), first)
    assert result is None and outcome is None and calls == []
    assert protocol.answer_floor(first, InklingTokenizer()) is None
    assert protocol.answer_floor(RESENT, InklingTokenizer()) == len(PREFIX) + 1


def test_a_history_no_stored_stream_shares_stays_inert(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    calls: list = []
    elsewhere = message("system", "You are a coding agent.") + message("user", "Write a poem.") + answer(model=True)
    result, outcome = _gate(_state([elsewhere], calls=calls), RESENT)
    assert result is None and outcome is None  # no stream: inert, no receipt, as before
    assert calls == [len(PREFIX) + 1]


def test_a_conversation_the_registry_holds_never_probes_by_prefix(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    calls: list = []
    result, outcome = _gate(_state([BANKED], calls=calls, registry=BANKED), RESENT)
    assert result is not None and "committed_source" not in outcome and calls == []


def test_thinking_off_splices_strictly_too(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    result, outcome = _gate(_state([BANKED]), RESENT, thinking=False)
    assert result is not None and result[1] == BANKED + RESULT
    assert outcome["committed_source"] == "bank_prefix" and outcome["declined"] == "thinking_disabled"


def test_other_families_never_probe_by_prefix(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    calls: list = []
    plain = SimpleNamespace(decode=InklingTokenizer().decode)  # no inkling_protocol
    result, outcome = _gate(_state([BANKED], tokenizer=plain, calls=calls), RESENT)
    assert result is None and outcome is None and calls == []


def test_the_bank_ranks_every_conversations_streams_by_shared_prefix(tmp_path):
    cold = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=2)
    try:
        bank = _bank(cold)
        base = list(range(1, 30))
        _put(bank, base + [50], "s1")                  # shares 30
        _put(bank, base + [50, 51, 52], "anon-1")      # shares 32
        _put(bank, base[:10], "s3")                    # shorter than the floor
        _put(bank, base + [60, 61], "s4")              # shares 29
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        _put(bank, base + [50, 51, 52, 53], "anon-2")  # RAM only, shares 33
        prompt = base + [50, 51, 52, 53, 99]
        streams = bank.shared_streams(prompt, model_path=str(MODEL), min_shared=20)
        assert streams == [tuple(base + tail) for tail in ([50, 51, 52, 53], [50, 51, 52], [50], [60, 61])]
        assert bank.shared_streams(prompt, model_path=str(MODEL), min_shared=31) == streams[:2]
        assert bank.shared_streams(prompt, model_path=str(MODEL), min_shared=20, limit=1) == streams[:1]
        assert cold.shared_streams(prompt, model_path=str(MODEL), min_shared=20) == streams[1:]
        assert cold.shared_streams(prompt, model_path="models/other", min_shared=20) == []
        assert bank.shared_streams(prompt[:15], model_path=str(MODEL), min_shared=20) == []
    finally:
        cold.close()


def test_the_real_template_marks_where_the_first_answer_starts():
    """With the pack's own tokenizer and template: a first request (its only model header opens
    the new answer) has no floor; the next request's floor is just inside its first answer."""
    import json

    import pytest

    from tests.test_inkling_committed_turns import PACK

    if not (PACK / "tokenizer.json").is_file():
        pytest.skip("needs the Inkling pack's tokenizer")
    from mtplx.runtime import _load_tokenizer_resilient

    tok = _load_tokenizer_resilient(PACK, json.loads((PACK / "config.json").read_text()))
    tok.inkling_protocol = True
    call = {"id": "c1", "type": "function", "function": {"name": "write", "arguments": json.dumps(FILE)}}
    turn = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Make the game."}]
    before = protocol.encode(tok, turn, effort="xhigh", thinking=True, strip_reasoning=False,
                             add_generation_prompt=True)
    history = turn + [{"role": "assistant", "content": None, "reasoning_content": "Write it.", "tool_calls": [call]},
                      {"role": "tool", "content": "Wrote it.", "tool_call_id": "c1"}]
    after = protocol.encode(tok, history, effort="xhigh", thinking=True, strip_reasoning=False,
                            add_generation_prompt=True)
    model = tok.convert_tokens_to_ids("<|message_model|>")
    assert protocol.answer_floor(before, tok) is None
    floor = protocol.answer_floor(after, tok)
    assert floor is not None and after[floor - 1] == model and floor <= len(before)
    assert model not in after[: floor - 1]
