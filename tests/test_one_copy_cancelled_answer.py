"""A cancelled answer leaves its conversation exactly where the prompt left it.

In the one-copy store a prompt's lease is its conversation's only copy, and
the answer decodes into it. Two faults broke a retry after a cancel
(2026-09-30):

- The client's cancel raises out of the token callback while the fixed-M4
  verifier's bank still holds the cache as tensor-offset adapters. The
  session bank could not read their offset, took the answer's rows for the
  prompt's own and resumed a recurrent state the answer had already
  advanced: the retry decoded other tokens than an uninterrupted turn. The
  bank is now handed back before the raise leaves
  (``one_copy.hand_back_on_raise``), and a lease whose offset cannot be read
  is never served.
- An identical prompt taken as a lease landed one slot short of its end
  while decoding from the entry's stored logits, so its first answer token
  overwrote its last prompt token (on the copying store too). A lease now
  lands where a clone lands, and an identical prompt banks its lease again
  before the answer decodes into it, so a cancel cannot drop it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.generation as generation
from mtplx.session_bank import SessionBank, _lease_advance
from test_lease_survives_aborted_prefill import _prompt, _runtime
from test_one_copy_conversation import _one_turn
from test_qwen4_fixed_m4_capacity_bucket import NATIVE, SEED, lane, pack  # noqa: F401


class _Cancelled(Exception):
    pass


def _cancel_after(tokens: int):
    seen = {"tokens": 0}

    def callback(new_tokens):
        seen["tokens"] += len(new_tokens)
        if seen["tokens"] >= tokens:
            raise _Cancelled()

    return callback


def _answer(rt, bank, prompt, *, turn, restore_mode="reference", token_callback=None):
    return generation.generate_mtpk(
        rt, list(prompt), max_tokens=24, sampler=NATIVE, draft_sampler=NATIVE,
        speculative_depth=3, seed=SEED + turn, mtp_cache_policy="persistent",
        mtp_history_policy="committed", verify_strategy="batched",
        stop_token_ids=set(), capture_final_state=True, session_bank=bank,
        session_id="s", session_restore_mode=restore_mode,
        commit_prompt_state_to_bank=True, token_callback=token_callback,
    )


def _cancelled(rt, bank, prompt):
    with pytest.raises(_Cancelled):
        _answer(rt, bank, prompt, turn=1, token_callback=_cancel_after(6))


def _bank():
    return SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)


def _conversation(rt, hidden_variant, *, one_copy=True):
    """A bank holding one finished turn, and the next turn's prompt."""

    bank = _bank()
    _tokens, entry, _result = _one_turn(rt, bank, _prompt(48), 0, one_copy, hidden_variant)
    return bank, list(entry.token_ids) + _prompt(40, salt=5)


@pytest.mark.parametrize("retry", ["extends", "identical"])
def test_a_retry_after_a_cancelled_answer_writes_what_an_uninterrupted_turn_writes(
    pack, lane, retry
):
    rt, hidden_variant = _runtime(pack, lane)
    bank, prompt = _conversation(rt, hidden_variant)
    _cancelled(rt, bank, prompt)

    [lease] = [e for k, e in bank._entries.items() if len(k) == len(prompt)]
    # Stock containers, whose offset says how far the answer ran.
    assert not any("TensorOffset" in type(layer).__name__ for layer in lease.cache_ref)
    assert _lease_advance(lease) > 0

    retried_prompt = prompt + [5] if retry == "extends" else prompt
    retried = _answer(rt, bank, retried_prompt, turn=1)
    assert retried.stats.cached_tokens == len(prompt)
    assert retried.stats.session_restore_mode == "reference_lease"

    reference_bank, _ = _conversation(rt, hidden_variant)
    expected = _answer(rt, reference_bank, retried_prompt, turn=1)
    assert list(retried.tokens) == list(expected.tokens)


@pytest.mark.parametrize("one_copy", [True, False], ids=["one_copy", "copying"])
def test_an_identical_prompt_resumes_a_lease_as_a_clone_would(pack, lane, one_copy):
    rt, hidden_variant = _runtime(pack, lane)
    lane.setenv("MTPLX_ONE_COPY", "1" if one_copy else "0")
    answers = {}
    for mode in ("reference", "clone"):
        bank = _bank()
        _tokens, entry, _result = _one_turn(
            rt, bank, _prompt(48), 0, one_copy, hidden_variant
        )
        answer = _answer(rt, bank, list(entry.token_ids), turn=1, restore_mode=mode)
        assert answer.stats.cached_tokens == len(entry.token_ids)
        answers[answer.stats.session_restore_mode] = list(answer.tokens)
    assert answers["reference_lease"] == answers["clone"]


def test_a_cancelled_identical_retry_keeps_the_conversation(pack, lane):
    rt, hidden_variant = _runtime(pack, lane)
    bank, prompt = _conversation(rt, hidden_variant)
    _cancelled(rt, bank, prompt)
    _cancelled(rt, bank, prompt)  # the identical retry, cancelled as well
    assert bank.restore_plan(prompt + [5])["reuse_tokens"] == len(prompt)

    retried = _answer(rt, bank, prompt, turn=1)
    assert retried.stats.cached_tokens == len(prompt)
    reference_bank, _ = _conversation(rt, hidden_variant)
    assert list(retried.tokens) == list(_answer(rt, reference_bank, prompt, turn=1).tokens)


def test_another_message_after_a_cancelled_answer_resumes_where_that_turn_began(
    pack, lane
):
    """The client cancels an answer and sends another message in its place:
    the new prompt leaves the cancelled one where the finished turn ended.
    The cancelled prompt's lease keeps a checkpoint there
    (``one_copy.prompt_lease_fields``), so only the new message is read, and
    the answer is the one an uninterrupted conversation gets. The nearest
    checkpoint used to be the previous prompt's end, a whole answer back
    (tests/test_cancelled_turn_resumes_in_ram.py has the 12 November case)."""
    lane.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "8")
    lane.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "8")
    rt, hidden_variant = _runtime(pack, lane)
    bank, prompt = _conversation(rt, hidden_variant)
    finished = prompt[:-40]
    _cancelled(rt, bank, prompt)

    replaced = finished + _prompt(40, salt=9)
    answer = _answer(rt, bank, replaced, turn=2)
    assert answer.stats.cached_tokens == len(finished)
    assert answer.stats.session_restore_mode.endswith("reference_lease")

    reference_bank, _ = _conversation(rt, hidden_variant)
    expected = _answer(rt, reference_bank, replaced, turn=2)
    assert list(answer.tokens) == list(expected.tokens)


def test_a_lease_whose_offset_cannot_be_read_is_not_served(monkeypatch):
    import mtplx.session_bank as session_bank

    entry = SimpleNamespace(lease_kv_offset=112, cache_ref=[object()])
    monkeypatch.setattr(session_bank, "_cache_kv_offset", lambda _cache: None)
    assert session_bank._lease_advance(entry) is None


def test_the_hand_back_runs_before_the_cancel_leaves():
    from mtplx.one_copy import hand_back_on_raise

    calls: list[dict] = []
    live = ["cache"]

    class Bank:
        def demote(self, cache, **kwargs):
            calls.append({"cache": cache, **kwargs})

    def cancelled(_tokens):
        raise _Cancelled()

    emit = hand_back_on_raise(cancelled, Bank(), lambda: live)
    with pytest.raises(_Cancelled):
        emit([1])
    assert calls == [{"cache": live, "compact": False, "keep_capacity": True}]

    # A failed hand-back never hides the cancel.
    class Broken:
        def demote(self, cache, **kwargs):
            raise RuntimeError("demote failed")

    with pytest.raises(_Cancelled):
        hand_back_on_raise(cancelled, Broken(), lambda: live)([1])

    # Tokens pass through untouched when nothing raises.
    seen: list[list[int]] = []
    hand_back_on_raise(seen.append, Bank(), lambda: live)([7, 8])
    assert seen == [[7, 8]] and len(calls) == 1


def test_the_batched_lane_leaves_a_full_prefix_lease_alone():
    pytest.importorskip("fastapi")
    from mtplx.server.openai import _BatchedARGenerationService

    prompt = [1, 2, 3, 4]

    class Bank:
        def longest_prefix(self, _ids):
            return SimpleNamespace(prefix_len=len(prompt), live_ref_only=True)

        def restore(self, *_args, **_kwargs):
            raise AssertionError("the lease must not be taken")

    job = SimpleNamespace(session_bank=Bank(), prompt_ids=prompt, cache_miss_reason=None)
    service = SimpleNamespace(state=SimpleNamespace(runtime=None))
    assert not _BatchedARGenerationService._prepare_session_bank_restore(service, job)
    assert job.cache_miss_reason == "ar_batch_full_prefix_not_insertable"
