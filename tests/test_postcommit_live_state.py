"""The state that wrote an answer is the cache for the next turn.

At the end of a streamed turn the server compares the tokens the model wrote with
its own guess of how the client will send that turn back. When only the guess
disagreed, the live state used to be refused and the turn re-read in the
background, and the next request waited for that re-read, cancelled it and read the
whole answer again (32.9 s to first text on 2.12.3, 10.0 s on the candidate). The
state is now banked under the tokens it holds, whatever the guess; the next request
decides by its real prompt. These tests pin that contract, its exceptions (the
server's own reasoning reshapes, refusals about the engine's state, a caller that
does not say how its prompt's images were spliced) and its rollback switch, plus the
two neighbours: a session-less client on a one-copy runtime keeps its answer as a
lease, and a refusal for size never starts a background re-read of the same size.
"""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import pytest

from mtplx.server import openai


PROMPT = [11, 12, 13, 14]
WRITTEN = [21, 22, 23]


def _final_state(tokens, *, safe=True):
    return SimpleNamespace(
        final_trunk_cache=["cache"],
        final_logits="logits",
        final_hidden="hidden",
        final_committed_mtp_cache=None,
        generated_token_ids=tuple(tokens),
        safe_to_commit=safe,
        finish_reason="stop",
    )


def _generated(tokens=WRITTEN, *, safe=True):
    return {"tokens": list(tokens), "_final_state": _final_state(tokens, safe=safe)}


def _mismatch(reason="retokenized_history_mismatch"):
    return {
        "safe": False,
        "mode": "unsafe",
        "reason": reason,
        "history_tokens": 5,
        "generation_boundary_tokens": 7,
        "divergence_token": 4,
        "divergence_offset_in_turn": 0,
    }


@pytest.fixture(autouse=True)
def _switch_default(monkeypatch):
    monkeypatch.delenv("MTPLX_POSTCOMMIT_KEEP_LIVE_STATE", raising=False)


@pytest.mark.parametrize(
    "reason", ["retokenized_history_mismatch", "stop_token_boundary_mismatch"]
)
def test_a_wrong_guess_keeps_the_written_tokens(reason):
    kept = openai._live_state_over_prediction(
        _mismatch(reason), prompt_ids=PROMPT, generated=_generated(), prompt_vision_splice=None
    )
    assert kept is not None
    assert kept["safe"] is True
    assert kept["mode"] == "generation_final_live"
    assert kept["token_ids"] == PROMPT + WRITTEN
    assert kept["history_suffix_tokens"] == 0
    assert kept["prediction"]["reason"] == reason
    assert kept["prediction"]["history_tokens"] == 5
    assert kept["prediction"]["generation_boundary_tokens"] == 7


@pytest.mark.parametrize(
    "reason",
    [
        # The server strips or scopes reasoning out of the next prompt itself.
        "reasoning_history_stripping_mismatch",
        "reasoning_history_scoping_mismatch",
        # Refusals about the engine's own state, never about the guess.
        "missing_generation_final_state",
        "generation_final_state_unsafe",
        "generated_token_mismatch",
        "stats_footer_in_assistant_history",
    ],
)
def test_other_refusals_stand(reason):
    assert (
        openai._live_state_over_prediction(
            _mismatch(reason), prompt_ids=PROMPT, generated=_generated(), prompt_vision_splice=None
        )
        is None
    )


def test_an_engine_state_that_is_not_whole_is_never_kept():
    assert (
        openai._live_state_over_prediction(
            _mismatch(), prompt_ids=PROMPT, generated=_generated(safe=False), prompt_vision_splice=None
        )
        is None
    )
    generated = _generated()
    generated["_final_state"] = _final_state([21, 22, 99])
    assert (
        openai._live_state_over_prediction(
            _mismatch(), prompt_ids=PROMPT, generated=generated, prompt_vision_splice=None
        )
        is None
    )


def test_a_caller_that_does_not_name_its_splice_keeps_the_old_refusal():
    assert (
        openai._live_state_over_prediction(
            _mismatch(),
            prompt_ids=PROMPT,
            generated=_generated(),
            prompt_vision_splice=openai._PROMPT_SPLICE_UNKNOWN,
        )
        is None
    )


def test_the_rollback_switch_restores_the_refusal(monkeypatch):
    monkeypatch.setenv("MTPLX_POSTCOMMIT_KEEP_LIVE_STATE", "0")
    assert (
        openai._live_state_over_prediction(
            _mismatch(), prompt_ids=PROMPT, generated=_generated(), prompt_vision_splice=None
        )
        is None
    )


def test_an_image_prompt_is_banked_in_its_content_keyed_view(monkeypatch):
    calls = []

    def frontier(ids, splice):
        calls.append((list(ids), splice))
        return [900 + int(token) for token in ids]

    monkeypatch.setattr(openai, "_session_frontier_ids", frontier)
    splice = object()
    kept = openai._live_state_over_prediction(
        _mismatch(), prompt_ids=PROMPT, generated=_generated(), prompt_vision_splice=splice
    )
    assert calls == [(PROMPT, splice)]
    # The prompt in the bank's view, the written tokens as they are.
    assert kept["token_ids"] == [911, 912, 913, 914] + WRITTEN


def test_an_image_prompt_that_cannot_be_keyed_keeps_the_refusal(monkeypatch):
    monkeypatch.setattr(openai, "_session_frontier_ids", lambda ids, splice: None)
    assert (
        openai._live_state_over_prediction(
            _mismatch(), prompt_ids=PROMPT, generated=_generated(), prompt_vision_splice=object()
        )
        is None
    )


class _Bank:
    def __init__(self, entry=True):
        self.puts = []
        self._entry = entry

    def put(self, **kwargs):
        self.puts.append(kwargs)
        if not self._entry:
            return None
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]), nbytes=0, token_hash="h"
        )


class _Flight:
    def __init__(self):
        self.events = []

    def pc(self, session_id, outcome):
        self.events.append((session_id, dict(outcome)))


def _state(bank):
    flight = _Flight()
    state = SimpleNamespace(
        sessions=SimpleNamespace(bank=bank, peek=lambda _sid: None),
        lock=Lock(),
        runtime=SimpleNamespace(model_path="m", mtp_enabled=True),
        template_hash="t",
        draft_head_identity="d",
        backend_descriptor=SimpleNamespace(backend_id="qwen4_exp"),
        flight=flight,
    )
    return state, flight


def _store(state, monkeypatch, compatibility, **kwargs):
    monkeypatch.setattr(
        openai, "_generation_final_postcommit_compatibility", lambda *_a, **_k: dict(compatibility)
    )
    monkeypatch.setattr(openai, "_flight", lambda s: s.flight)
    monkeypatch.setattr(
        openai, "_generation_final_bank_metadata", lambda *_a, **_k: {"hidden_variant": "post_norm"}
    )
    return openai._store_generation_final_history_snapshot(
        state,
        session_id="s1",
        prompt_ids=list(PROMPT),
        generated=_generated(),
        messages=[],
        assistant_content="answer",
        thinking_enabled=True,
        policy_fingerprint="p",
        **kwargs,
    )


def test_the_streamed_commit_banks_the_written_tokens_over_a_wrong_guess(monkeypatch):
    bank = _Bank()
    state, flight = _state(bank)
    outcome = _store(state, monkeypatch, _mismatch(), prompt_vision_splice=None)
    assert outcome["stored"] is True
    assert outcome["mode"] == "generation_final_live"
    assert outcome["reason"] == "live_state_kept_over_prediction"
    assert outcome["prefix_len"] == len(PROMPT) + len(WRITTEN)
    assert outcome["prediction"]["reason"] == "retokenized_history_mismatch"
    assert [put["token_ids"] for put in bank.puts] == [PROMPT + WRITTEN]
    # The guess that lost is in the flight record of the turn.
    assert flight.events[-1][1]["prediction"]["generation_boundary_tokens"] == 7


def test_a_streamed_commit_without_the_splice_still_refuses(monkeypatch):
    bank = _Bank()
    state, _flight = _state(bank)
    outcome = _store(state, monkeypatch, _mismatch())
    assert outcome["stored"] is False
    assert outcome["reason"] == "retokenized_history_mismatch"
    assert bank.puts == []


def test_a_matching_guess_is_unchanged(monkeypatch):
    bank = _Bank()
    state, _flight = _state(bank)
    compatibility = {
        "safe": True,
        "mode": "generation_final_prefix",
        "reason": "generation_boundary_prefix_of_history",
        "token_ids": PROMPT + WRITTEN,
        "history_suffix_tokens": 1,
    }
    outcome = _store(state, monkeypatch, compatibility, prompt_vision_splice=None)
    assert outcome["mode"] == "generation_final_prefix"
    assert outcome["history_suffix_tokens"] == 1
    assert "prediction" not in outcome


def test_one_copy_runtime_leases_a_sessionless_answer(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS", raising=False)
    for source in ("new", "longest_prefix", "implicit_hash"):
        assert (
            openai._session_keep_live_refs_for_request(
                session_source=source, session_id="anon-bench", one_copy=True
            )
            is True
        )
        # A copying runtime keeps the snapshot policy.
        assert (
            openai._session_keep_live_refs_for_request(
                session_source=source, session_id="anon-bench", one_copy=False
            )
            is False
        )
    # No session at all: nothing to lease under.
    assert (
        openai._session_keep_live_refs_for_request(
            session_source=None, session_id=None, one_copy=True
        )
        is False
    )


def test_a_refusal_for_size_never_rereads_the_same_size():
    skipped = openai._skipped_idle_postcommit_snapshot(
        state=SimpleNamespace(backend_descriptor=SimpleNamespace(backend_id="qwen4_exp")),
        unsafe_reason="sessionbank_snapshot_skipped",
        prompt_prefix_len=193533,
    )
    assert skipped is not None
    assert skipped["stored"] is False
    assert skipped["reason"] == "bank_refused_this_size"
    assert skipped["prompt_prefix_len"] == 193533
    # A guess that disagreed still has its own path.
    assert (
        openai._skipped_idle_postcommit_snapshot(
            state=SimpleNamespace(backend_descriptor=SimpleNamespace(backend_id="qwen4_exp")),
            unsafe_reason="reasoning_history_stripping_mismatch",
        )
        is None
    )


# -- the end of a turn on a one-copy runtime needs no guess -------------------


def _one_copy_state(*, strip=False):
    return SimpleNamespace(
        runtime=SimpleNamespace(qwen4_fixed_m4_compiled_verify=True),
        args=SimpleNamespace(strip_assistant_reasoning_history=strip),
    )


def _compatibility(state, monkeypatch, *, guessed=None, **kwargs):
    """The real check, with the guess replaced by ``guessed`` (or a failure
    when the guess must not run at all)."""

    def guess(*_a, **_k):
        if guessed is None:
            raise AssertionError("the end of the turn re-rendered the conversation")
        return list(guessed), None

    monkeypatch.setattr(openai, "_history_ids_for_postcommit", guess)
    return openai._generation_final_postcommit_compatibility(
        state,
        prompt_ids=list(PROMPT),
        generated=_generated(),
        messages=[],
        assistant_content="answer",
        thinking_enabled=True,
        **kwargs,
    )


def test_a_one_copy_turn_is_banked_as_written_without_guessing(monkeypatch):
    monkeypatch.setattr(openai, "_reasoning_history_scoped_active", lambda _s: False)
    out = _compatibility(_one_copy_state(), monkeypatch, prompt_vision_splice=None)
    assert out == {
        "safe": True,
        "mode": "generation_final_live",
        "reason": "live_state_without_guess",
        "token_ids": PROMPT + WRITTEN,
        "history_suffix_tokens": 0,
    }


def test_an_image_turn_is_banked_in_its_keyed_view_without_guessing(monkeypatch):
    monkeypatch.setattr(openai, "_reasoning_history_scoped_active", lambda _s: False)
    monkeypatch.setattr(
        openai, "_session_frontier_ids", lambda ids, splice: [900 + int(t) for t in ids]
    )
    out = _compatibility(_one_copy_state(), monkeypatch, prompt_vision_splice=object())
    assert out["token_ids"] == [911, 912, 913, 914] + WRITTEN


@pytest.mark.parametrize("case", ["copying_runtime", "stripped", "scoped", "unknown_splice",
                                  "switch_off", "unkeyable_image"])
def test_the_guess_still_runs_where_it_decides_something(monkeypatch, case):
    state = _one_copy_state(strip=case == "stripped")
    if case == "copying_runtime":
        state.runtime = SimpleNamespace(qwen4_fixed_m4_compiled_verify=False)
    monkeypatch.setattr(openai, "_reasoning_history_scoped_active", lambda _s: case == "scoped")
    if case == "switch_off":
        monkeypatch.setenv("MTPLX_POSTCOMMIT_KEEP_LIVE_STATE", "0")
    splice = None
    if case == "unkeyable_image":
        splice = object()
        monkeypatch.setattr(openai, "_session_frontier_ids", lambda ids, s: None)
    kwargs = {} if case == "unknown_splice" else {"prompt_vision_splice": splice}
    out = _compatibility(state, monkeypatch, guessed=PROMPT + WRITTEN, **kwargs)
    # The guess ran and matched: the old exact outcome.
    assert out["mode"] == "generation_final_exact" and out["reason"] == "token_identical"


def test_refusals_about_the_engine_state_come_before_any_shortcut(monkeypatch):
    monkeypatch.setattr(openai, "_reasoning_history_scoped_active", lambda _s: False)
    state = _one_copy_state()
    monkeypatch.setattr(openai, "_history_ids_for_postcommit", lambda *_a, **_k: ([], None))
    unsafe = openai._generation_final_postcommit_compatibility(
        state, prompt_ids=list(PROMPT), generated=_generated(safe=False), messages=[],
        assistant_content="answer", thinking_enabled=True, prompt_vision_splice=None,
    )
    assert unsafe["reason"] == "generation_final_state_unsafe"
    generated = _generated()
    generated["_final_state"] = _final_state([21, 22, 99])
    mismatch = openai._generation_final_postcommit_compatibility(
        state, prompt_ids=list(PROMPT), generated=generated, messages=[],
        assistant_content="answer", thinking_enabled=True, prompt_vision_splice=None,
    )
    assert mismatch["reason"] == "generated_token_mismatch"


def test_the_streamed_commit_passes_its_splice_to_the_check(monkeypatch):
    seen = {}

    def check(*_a, **kwargs):
        seen.update(kwargs)
        return {"safe": True, "mode": "generation_final_live", "reason": "live_state_without_guess",
                "token_ids": PROMPT + WRITTEN, "history_suffix_tokens": 0}

    bank = _Bank()
    state, _flight = _state(bank)
    monkeypatch.setattr(openai, "_flight", lambda s: s.flight)
    monkeypatch.setattr(
        openai, "_generation_final_bank_metadata", lambda *_a, **_k: {"hidden_variant": "post_norm"}
    )
    monkeypatch.setattr(openai, "_generation_final_postcommit_compatibility", check)
    splice = object()
    outcome = openai._store_generation_final_history_snapshot(
        state, session_id="s1", prompt_ids=list(PROMPT), generated=_generated(), messages=[],
        assistant_content="answer", thinking_enabled=True, policy_fingerprint="p",
        prompt_vision_splice=splice,
    )
    assert seen["prompt_vision_splice"] is splice
    assert outcome["stored"] is True and outcome["reason"] == "live_state_without_guess"
