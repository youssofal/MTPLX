"""The Gemma 4 near-prefix lane lets go of the candidates it does not hold.

Same loop shape as ``generation._restore_near_prefix_prompt_state`` before
it took candidates off its list (tests/test_near_lane_releases_refused.py):
iterating over the bank's candidate list kept every refused and untried
candidate alive through the served candidate's tail forward, and the last
one tried stayed bound through the cold prefill when none was served. A
candidate the SSD tier decoded is a whole private copy of that entry.

CPU-only: a fake bank and a fake prefill that checks, on every call, which
candidates are still alive.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import mtplx.backends.gemma4_assistant as gemma4

PROMPT = list(range(1, 41))


class _TokenCache:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.offset = len(self.token_ids)


class _Runtime:
    model_path = Path("models/gemma4")
    mtp_enabled = True

    def make_cache(self):
        return [_TokenCache([])]


class _Candidate:
    """A bank entry as far as the Gemma lane reads it."""

    model_path = "models/gemma4"
    mtp_enabled = True
    hidden_variant = "gemma4_pre_norm"
    template_hash = None
    mtp_history_policy = gemma4.GEMMA4_SESSION_STATE_POLICY
    draft_head_identity = None
    policy_fingerprint = None

    def __init__(self, token_ids, *, cache_source="ram"):
        self.token_ids = tuple(token_ids)
        self.prefix_len = len(self.token_ids)
        self.cache_source = cache_source
        self.hits = 0
        self.last_access_s = 0.0


class _ListBank:
    """Hands out a fixed candidate list; restores only ``serves`` (by its
    tokens: the bank does not hold an SSD candidate)."""

    last_miss_reason = "prefix_divergence_at_token"

    def __init__(self, candidates, serves):
        self._candidates = candidates
        self._serves = None if serves is None else serves.token_ids
        self.restore_asked: list[int] = []

    def restore(self, *_args, **_kwargs):
        return None

    def near_prefix_candidates(self, _ids, **_kwargs):
        candidates, self._candidates = self._candidates, None
        return candidates

    def restore_entry_prefix_cache(self, _rt, entry, matched, **_kwargs):
        self.restore_asked.append(int(matched))
        if entry.token_ids != self._serves:
            return None
        return [_TokenCache(PROMPT[: int(matched) - 1])], None, "clone", int(matched)


def _watching_prefill(monkeypatch, watched):
    alive_per_call: list[list[bool]] = []

    def fake_prefill(_runtime, prompt_ids, *, cache, phase, abort_check=None):
        gc.collect()
        alive_per_call.append([ref() is not None for ref in watched])
        cache[0].token_ids.extend(int(token) for token in prompt_ids)
        cache[0].offset = len(cache[0].token_ids)
        length = len(prompt_ids)
        return (
            SimpleNamespace(
                logits=np.zeros((1, length, 4), dtype=np.float32),
                hidden=np.zeros((1, length, 2), dtype=np.float32),
                shared_kv_states={},
                cache_offset=cache[0].offset,
            ),
            0.0,
        )

    monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", fake_prefill)
    return alive_per_call


def _restore(bank):
    return gemma4._restore_or_prefill_gemma4_prompt(
        _Runtime(), PROMPT, session_bank=bank, require_shared_kv=False
    )


def test_only_the_served_ram_entry_is_held_through_the_tail_forward(monkeypatch):
    whole = _Candidate(PROMPT[:30])  # a whole prefix: out of range here
    failing = _Candidate(PROMPT[:28] + [99, 98])  # its restore fails
    served = _Candidate(PROMPT[:20] + [97])
    untried = _Candidate(PROMPT[:12] + [96])
    watched = [weakref.ref(c) for c in (whole, failing, untried)]
    alive_per_call = _watching_prefill(monkeypatch, watched)
    bank = _ListBank([(whole, 30), (failing, 28), (served, 20), (untried, 12)], served)
    served_ref = weakref.ref(served)
    del whole, failing, untried

    state = _restore(bank)

    assert state.cache_hit is True and state.cached_tokens == 19
    assert bank.restore_asked == [28, 20]
    assert alive_per_call == [[False, False, False]]
    # A RAM entry stays with the bank, which counts the hit.
    assert served_ref() is not None and served_ref().hits == 1


def test_a_served_ssd_candidate_is_released_before_the_tail_forward(monkeypatch):
    served = _Candidate(PROMPT[:20] + [97], cache_source="ssd")
    watched = [weakref.ref(served)]
    alive_per_call = _watching_prefill(monkeypatch, watched)
    bank = _ListBank([(served, 20)], served)
    del served

    state = _restore(bank)

    assert state.cache_hit is True and state.cached_tokens == 19
    assert alive_per_call == [[False]]


def test_the_last_refused_candidate_is_released_before_the_cold_prefill(
    monkeypatch,
):
    whole = _Candidate(PROMPT[:30])
    failing = _Candidate(PROMPT[:28] + [99, 98], cache_source="ssd")
    watched = [weakref.ref(c) for c in (whole, failing)]
    alive_per_call = _watching_prefill(monkeypatch, watched)
    bank = _ListBank([(whole, 30), (failing, 28)], serves=None)
    del whole, failing

    state = _restore(bank)

    assert state.cache_hit is False and state.restore_mode == "cold"
    assert bank.restore_asked == [28]
    assert alive_per_call == [[False, False]]
