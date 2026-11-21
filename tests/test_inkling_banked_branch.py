"""A resent history that continues a branch the session registry no longer points at.

The registry keeps one stream per conversation: the last turn's. When the client goes back in its
history and continues another branch (Pi's tree, a system prompt edited and then restored, a request
made on the side under the same id), that stream leaves the prompt at a message the client wrote,
and every turn whose ids the splice once served is read again from the first one (the T1 audit,
21 Nov: 9,634 tokens and 11.0 s on the turn back to the main thread at 11K). The bank still holds
the branch the prompt continues; the gate compares with it, proving each turn as before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.cache_bank import SessionBankColdTier
from mtplx.server import openai as oa

from tests.test_inkling_committed_turns import PREFIX, RESULT, InklingTokenizer, answer, message
from tests.test_session_stream_after_restart import MODEL, _bank, _put

# The side branch: the same system prompt, then a client-written summary instead of the task.
SIDE = message("system", "You are a coding agent.") + message("user", "Summary: the game is made.") + answer(model=True)
MAIN = PREFIX + answer(model=True)  # the main thread as the bank holds it: the model's own ids


def _state(registry, banked, *, tokenizer=None, calls=None):
    """A gate state whose registry holds `registry` for the conversation and whose bank holds
    `banked` (ranked as the bank ranks them); `calls` records every bank lookup."""

    def streams(session_id, *, model_path, prompt, limit=4):
        if calls is not None:
            calls.append(limit)
        return [tuple(s) for s in banked][:limit] if session_id == "s1" else []

    bank = SimpleNamespace(session_streams=streams,
                           session_stream=lambda session_id, *, model_path, prompt: (streams(session_id, model_path=model_path, prompt=prompt) or [()])[0])
    session = SimpleNamespace(committed_token_ids=tuple(registry))
    return SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=SimpleNamespace(resolve_session_id=lambda **kw: ("s1", "header.x-mtplx-session-id"),
                                 peek=lambda sid: session if sid == "s1" else None, bank=bank),
        runtime=SimpleNamespace(tokenizer=tokenizer or InklingTokenizer(), model_path=MODEL),
    )


def _gate(state, prompt, compared=None):
    request = oa.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "Make the game."}])
    observed: dict = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state, messages=request.messages, prompt_ids=list(prompt), headers={}, metadata={}, request=request,
        thinking_enabled=True, reasoning_effort="xhigh", tools=None, tool_choice=None, tool_prompt_mode="native",
        template_observability={}, request_observability=observed, compared_stream=compared,
    )
    return result, observed.get("committed_reasoning_canonicalization")


def test_the_turn_back_to_the_main_thread_is_served_from_its_banked_branch(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    prompt = PREFIX + answer(model=False) + RESULT
    compared: list = []
    result, outcome = _gate(_state(SIDE, [SIDE, MAIN]), prompt, compared)
    assert result is not None and result[1] == MAIN + RESULT
    assert outcome["applied"] is True and outcome["committed_source"] == "bank_branch"
    assert outcome["cp_spliced"] == len(MAIN) and outcome["committed_len"] == len(MAIN)
    assert outcome["left_stream"] == {"source": "session", "committed_len": len(SIDE),
                                      "cp_raw": oa.common_prefix_len(prompt, SIDE), "stopped": "outside_assistant_turn"}
    assert compared == [tuple(MAIN)]  # the image gate judges against the same stream


def test_a_registry_stream_the_walk_carries_never_reads_the_bank(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    calls: list = []
    result, outcome = _gate(_state(MAIN, [SIDE], calls=calls), PREFIX + answer(model=False) + RESULT)
    assert result is not None and result[1] == MAIN + RESULT
    assert "committed_source" not in outcome and calls == []


def test_a_history_no_banked_branch_continues_is_served_as_sent(monkeypatch):
    """An edited system prompt: every branch leaves the prompt inside it, so nothing is served
    and the receipt keeps the registry stream's reason."""
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    edited = message("system", "You are a careful coding agent.") + message("user", "Make the game.")
    calls: list = []
    result, outcome = _gate(_state(MAIN, [MAIN, SIDE], calls=calls), edited + answer(model=False) + RESULT)
    assert result is None
    assert outcome["applied"] is False and outcome["token_splice"]["stopped"] == "outside_assistant_turn"
    assert "committed_source" not in outcome and calls == [4]


def test_the_branch_the_walk_carries_furthest_wins_over_the_longest_shared_prefix(monkeypatch):
    """Two banked branches share the prompt up to the same seam; the walk decides, not the
    raw prefix: the older entry of the main thread ends before the prompt's last turn."""
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    older = PREFIX + answer(model=True)
    newer = older + RESULT + answer(model=True, reasoning="Fix the pipes.\n")
    prompt = PREFIX + answer(model=False) + RESULT + answer(model=False, reasoning="Fix the pipes.\n") + RESULT
    result, outcome = _gate(_state(SIDE, [older, newer]), prompt)
    assert result is not None and result[1] == newer + RESULT
    assert outcome["cp_spliced"] == len(newer)


def test_other_families_never_read_a_banked_branch(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    calls: list = []
    plain = SimpleNamespace(decode=InklingTokenizer().decode)  # no inkling_protocol
    compared: list = []
    _gate(_state(SIDE, [MAIN], tokenizer=plain, calls=calls), PREFIX + answer(model=False) + RESULT, compared)
    assert calls == [] and compared == []


def test_the_image_gate_judges_a_repair_against_the_stream_the_gate_compared_with():
    """The registry's stream is a branch that never saw the image; the banked branch the gate
    compared with holds it, so the repair stands. Without that stream it is refused."""
    from tests.test_vision_session_restore import (
        DIGEST_A, PAD, VISION_END, VISION_START, _keyed, _one_image_splice,
    )

    raw = [1, 2, 9, VISION_START, PAD, VISION_END, 5]
    raw_keyed = _keyed(raw, (DIGEST_A, 1))
    repaired = [1, 2, 3, 4, *raw_keyed[3:6], 6]
    side = [1, 2, 8, 8, 8]  # the other branch: no image

    def gate(compared):
        state = SimpleNamespace(
            _vision_spec_cache=SimpleNamespace(image_token_id=PAD),
            sessions=SimpleNamespace(peek=lambda _sid: SimpleNamespace(committed_token_ids=tuple(side))),
            runtime=SimpleNamespace(tokenizer=InklingTokenizer(), model_path=MODEL),
        )
        receipt = {"enabled": True, "canonicalized": False, "refused": None}
        served = oa._vision_gate_canonicalization(
            state, raw_keyed=raw_keyed, canonicalized=(["canon"], repaired), splice=_one_image_splice(),
            canon_observability={"committed_reasoning_canonicalization": {"applied": True}},
            template_observability={}, receipt=receipt, session_id="s1", compared_stream=compared,
        )
        return served, receipt

    served, receipt = gate([tuple(repaired)])
    assert served is not None and receipt["session_committed_tokens"] == len(repaired)
    served, receipt = gate(None)
    assert served is None and receipt["refused"] == "canonicalization_crossed_an_image_not_in_the_session"


def test_the_bank_ranks_a_conversations_streams_by_shared_prefix_then_length(tmp_path):
    cold = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=2)
    try:
        bank = _bank(cold)
        base = list(range(1, 30))
        _put(bank, base + [50], "s1")              # shares 30
        _put(bank, base + [50, 51, 52], "s1")      # shares 32
        _put(bank, base + [60, 61, 62, 63], "s1")  # shares 29
        _put(bank, base + [50, 51, 70], "s2")      # another conversation
        assert cold.flush(timeout_s=5.0) is True
        _put(bank, base + [50, 51, 52, 53], "s1")  # RAM and SSD: the same stream once
        prompt = base + [50, 51, 52, 53, 99]
        streams = bank.session_streams("s1", model_path=str(MODEL), prompt=prompt)
        assert streams == [tuple(base + tail) for tail in ([50, 51, 52, 53], [50, 51, 52], [50], [60, 61, 62, 63])]
        assert bank.session_streams("s1", model_path=str(MODEL), prompt=prompt, limit=2) == streams[:2]
        assert cold.session_streams("s2", model_path=str(MODEL), prompt=prompt) == [tuple(base + [50, 51, 70])]
        assert bank.session_stream("s1", model_path=str(MODEL), prompt=prompt) == streams[0]
    finally:
        cold.close()
