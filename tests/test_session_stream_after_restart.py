"""A conversation the session registry no longer holds (the server restarted, or it was evicted
idle) still has its entries in the bank. Every turn whose ids the splice once served differs from
the client's re-render, so without a stream to compare with, the first turn after a restart leaves
every stored entry at the first such turn and reads the rest again. The gate now takes that
conversation's stored stream from the bank (RAM, then the SSD manifest) for Inkling.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.server import openai as oa
from mtplx.session_bank import SessionBank

from tests.test_inkling_committed_turns import PREFIX, RESULT, InklingTokenizer, answer

MODEL = Path("models/example")


def _runtime():
    return SimpleNamespace(model_path=MODEL, mtp_enabled=True)


def _bank(cold=None):
    return SessionBank(max_entries=16, max_bytes=1 << 20, per_session_max_bytes=1 << 20, cold_tier=cold)


def _put(bank, token_ids, session_id):
    return bank.put_snapshot(
        runtime=_runtime(),
        token_ids=list(token_ids),
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        session_id=session_id,
        snapshot_epoch=len(token_ids),
        nbytes_override=128,
    )


def test_the_ssd_manifest_gives_the_conversations_entry_that_shares_the_most(tmp_path):
    cold = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=2)
    try:
        bank = _bank(cold)
        _put(bank, range(1, 30), "s1")
        _put(bank, list(range(1, 30)) + [50, 51, 52], "s1")
        _put(bank, list(range(1, 30)) + [60, 61, 62, 63, 64], "s2")
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        prompt = list(range(1, 30)) + [50, 51, 70]
        assert cold.session_stream("s1", model_path=str(MODEL), prompt=prompt) == tuple(range(1, 30)) + (50, 51, 52)
        assert bank.session_stream("s1", model_path=str(MODEL), prompt=prompt) == tuple(range(1, 30)) + (50, 51, 52)
        # Never another conversation's entry, another model's, or a session it does not know.
        assert cold.session_stream("s3", model_path=str(MODEL), prompt=prompt) == ()
        assert cold.session_stream("s1", model_path="models/other", prompt=prompt) == ()
        assert bank.session_stream(None, model_path=str(MODEL), prompt=prompt) == ()
    finally:
        cold.close()


def test_ram_entries_answer_first_and_the_longer_shared_prefix_wins(tmp_path):
    cold = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=2)
    try:
        bank = _bank(cold)
        _put(bank, list(range(1, 30)) + [50], "s1")
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        _put(bank, list(range(1, 30)) + [50, 51, 52, 53], "s1")  # RAM only
        prompt = list(range(1, 30)) + [50, 51, 52, 53, 54]
        assert bank.session_stream("s1", model_path=str(MODEL), prompt=prompt) == tuple(range(1, 30)) + (50, 51, 52, 53)
    finally:
        cold.close()


def _restarted(stream, *, tokenizer=None):
    """A gate state whose registry holds no conversation and whose bank holds `stream`."""
    bank = SimpleNamespace(session_stream=lambda session_id, *, model_path, prompt: tuple(stream) if session_id == "s1" else ())
    return SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=SimpleNamespace(resolve_session_id=lambda **kw: ("s1", "header.x-mtplx-session-id"),
                                 peek=lambda sid: None, bank=bank),
        runtime=SimpleNamespace(tokenizer=tokenizer or InklingTokenizer(), model_path=MODEL),
    )


def _gate(state, prompt):
    request = oa.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "Make the game."}])
    observed: dict = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state, messages=request.messages, prompt_ids=list(prompt), headers={}, metadata={}, request=request,
        thinking_enabled=True, reasoning_effort="xhigh", tools=None, tool_choice=None, tool_prompt_mode="native",
        template_observability={}, request_observability=observed,
    )
    return result, observed.get("committed_reasoning_canonicalization")


def test_after_a_restart_the_gate_compares_the_resent_history_with_the_banked_stream(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    banked = PREFIX + answer(model=True)
    prompt = PREFIX + answer(model=False) + RESULT
    result, outcome = _gate(_restarted(banked), prompt)
    assert result is not None and result[1] == banked + RESULT
    assert outcome["committed_source"] == "bank" and outcome["cp_spliced"] == len(banked)


def test_a_new_conversation_after_a_restart_is_left_alone(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    result, outcome = _gate(_restarted(()), PREFIX + answer(model=False) + RESULT)
    assert result is None and outcome is None


def test_other_families_never_read_the_bank_stream(monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    plain = SimpleNamespace(decode=InklingTokenizer().decode)  # no inkling_protocol
    banked = PREFIX + answer(model=True)
    result, outcome = _gate(_restarted(banked, tokenizer=plain), PREFIX + answer(model=False) + RESULT)
    assert result is None and outcome is None


def test_an_image_conversation_restored_after_a_restart_keeps_its_repair():
    """The image gate checks a repaired prompt against the stream the gate compared with: the
    bank's, when the conversation is not in the registry (otherwise every repair that crosses an
    image the restarted conversation holds would be refused)."""
    from tests.test_vision_session_restore import (
        DIGEST_A, PAD, VISION_END, VISION_START, _keyed, _one_image_splice,
    )

    raw = [1, 2, 9, VISION_START, PAD, VISION_END, 5]
    raw_keyed = _keyed(raw, (DIGEST_A, 1))
    repaired = [1, 2, 3, 4, *raw_keyed[3:6], 6]  # the stretch after the image is the conversation's own
    held = list(repaired)

    def state(tokenizer):
        bank = SimpleNamespace(session_stream=lambda session_id, *, model_path, prompt: tuple(held))
        return SimpleNamespace(
            _vision_spec_cache=SimpleNamespace(image_token_id=PAD),
            sessions=SimpleNamespace(peek=lambda _sid: None, bank=bank),
            runtime=SimpleNamespace(tokenizer=tokenizer, model_path=MODEL),
        )

    def gate(tokenizer):
        receipt = {"enabled": True, "canonicalized": False, "refused": None}
        served = oa._vision_gate_canonicalization(
            state(tokenizer), raw_keyed=raw_keyed, canonicalized=(["canon"], repaired), splice=_one_image_splice(),
            canon_observability={"committed_reasoning_canonicalization": {"applied": True}},
            template_observability={}, receipt=receipt, session_id="s1",
        )
        return served, receipt

    served, receipt = gate(InklingTokenizer())
    assert served is not None and receipt["canonicalized"] is True
    assert receipt["session_committed_tokens"] == len(held)
    # Another family keeps today's rule: no registry stream, no crossing of an image.
    served, receipt = gate(SimpleNamespace(decode=InklingTokenizer().decode))
    assert served is None and receipt["refused"] == "canonicalization_crossed_an_image_not_in_the_session"
