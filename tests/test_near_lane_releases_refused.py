"""The near-prefix lane lets go of the candidates it does not serve.

Measured on 2026-10-09 (Qwen3.8-27B, 7 GB bank): the SSD tier decoded a
spilled conversation's entry (5.4 GB) into a near-prefix candidate, the lane
refused it (a whole prefix of the prompt), served a neighbour's checkpoint at
the shared system prompt and re-read 54K tokens. The lane iterated over the
candidate list while that prefill ran, so the list kept the refused, decoded
entry alive through it; the memory guard then stopped the prefill (507).

The lane now takes each candidate off its list as it tries it and drops the
untried rest before the served one's suffix prefill. The tests check, from
inside the prefill's forwards, that nothing else holds a refused or untried
candidate.

CPU-sized: toy runtime; the first test uses a real SessionBank and cold tier
on tmp_path.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mtplx import generation
from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.session_bank import SessionBank

SPILLED = list(range(1, 42))  # conversation A, on SSD only
PROMPT = SPILLED + [70, 71, 72]  # A's next turn
NEIGHBOUR = SPILLED[:16] + [90 + i for i in range(30)]  # B shares 16 tokens


class _WatchingModel:
    """Toy model that records, on every forward, whether the watched objects
    are still alive."""

    def __init__(self):
        self.watched: list[weakref.ref] = []
        self.alive_per_forward: list[list[bool]] = []

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def __call__(self, input_ids, *, cache=None, return_hidden=False,
                 hidden_variant=None, emit_logits=True, logits_keep=None):
        gc.collect()
        self.alive_per_forward.append([ref() is not None for ref in self.watched])
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        return (logits, hidden) if return_hidden else logits


def _runtime(model) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=SimpleNamespace(decode=lambda tokens, **_: ""),
        model_path=Path("models/example"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _near(rt, prompt, bank, **kwargs):
    return generation._restore_near_prefix_prompt_state(
        rt,
        list(prompt),
        base_hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
        mtp_hidden_variant=None,
        mtp_history_policy="cycle",
        session_bank=bank,
        template_hash=None,
        draft_head_identity=None,
        policy_fingerprint=None,
        allow_block_prefix=True,
        **kwargs,
    )


def _put(bank: SessionBank, rt, token_ids, session_id: str):
    return bank.put_snapshot(
        runtime=rt,
        token_ids=token_ids,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
        session_id=session_id,
        snapshot_epoch=len(token_ids),
        mtp_history_policy="cycle",
        nbytes_override=128,
    )


def test_a_refused_ssd_candidate_is_released_before_the_served_prefill(
    tmp_path, monkeypatch
):
    """The 2026-10-09 shape on a real bank and cold tier: A's spilled whole
    prefix is decoded and refused, B's entry at the shared opening serves."""
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "8")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "8")
    monkeypatch.setenv("MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS", "4")
    model = _WatchingModel()
    rt = _runtime(model)
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=2
    )
    try:
        bank = SessionBank(
            max_entries=8, max_bytes=4096, per_session_max_bytes=4096, cold_tier=cold
        )
        assert _put(bank, rt, SPILLED, "a") is not None
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        assert _put(bank, rt, NEIGHBOUR, "b") is not None

        decoded: list[weakref.ref] = []
        original = bank._cold_near_prefix_candidate

        def watch(*args, **kwargs):
            found = original(*args, **kwargs)
            if found is not None:
                decoded.append(weakref.ref(found[0]))
                model.watched.append(decoded[-1])
            return found

        monkeypatch.setattr(bank, "_cold_near_prefix_candidate", watch)
        state = _near(rt, PROMPT, bank, session_id="a")

        assert len(decoded) == 1, "the cold tier decoded A's entry"
        assert state is not None and state.cached_tokens == 16
        assert state.restore_served["candidate_index"] == 2
        assert state.suffix_tokens == len(PROMPT) - 16
        assert model.alive_per_forward, "the suffix prefill ran"
        # The seed forward and every suffix forward ran without it.
        assert all(alive == [False] for alive in model.alive_per_forward)
    finally:
        cold.close()


class _Candidate:
    """A bank entry as far as the lane's gates read it."""

    has_recurrent = False
    mtp_history_snapshot = None
    mtp_history_cache_ref = None
    cache_ref = None
    live_ref_only = False
    gdn_boundaries = ()
    cold_encode_completed_at = None

    def __init__(self, rt, token_ids):
        self.token_ids = tuple(token_ids)
        self.prefix_len = len(self.token_ids)
        self.token_hash = f"h{self.prefix_len}"
        self.model_path = str(rt.model_path)
        self.mtp_enabled = True
        self.hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)
        self.template_hash = None
        self.mtp_history_policy = "cycle"
        self.draft_head_identity = None
        self.policy_fingerprint = None
        self.hits = 0
        self.last_access_s = 0.0


class _ListBank:
    """Hands out a fixed candidate list; restores only ``serves``."""

    last_miss_reason = None

    def __init__(self, candidates, serves):
        self._candidates = candidates
        self._serves = serves
        self.restore_asked: list[int] = []

    def near_prefix_candidates(self, _ids, **_kwargs):
        candidates, self._candidates = self._candidates, None
        return candidates

    def restore_entry_prefix_cache(self, _rt, entry, matched, **_kwargs):
        self.restore_asked.append(int(matched))
        if entry is not self._serves:
            return None
        return [], None, "clone", int(matched)


def test_refused_and_untried_candidates_go_before_the_served_prefill():
    model = _WatchingModel()
    rt = _runtime(model)
    whole = _Candidate(rt, PROMPT[:30])  # a whole prefix: out of range here
    failing = _Candidate(rt, PROMPT[:28] + [99, 98])  # its restore fails
    served = _Candidate(rt, PROMPT[:20] + [97])
    untried = _Candidate(rt, PROMPT[:12] + [96])
    model.watched = [weakref.ref(c) for c in (whole, failing, untried)]
    bank = _ListBank([(whole, 30), (failing, 28), (served, 20), (untried, 12)], served)
    served_ref = weakref.ref(served)
    del whole, failing, untried

    state = _near(rt, PROMPT, bank)

    assert state is not None and state.cached_tokens == 20
    assert state.restore_served["candidate_index"] == 3
    assert bank.restore_asked == [28, 20]
    assert model.alive_per_forward
    assert model.alive_per_forward[-1] == [False, False, False]
    assert served_ref() is not None


def test_with_no_candidate_served_the_lane_still_reports_the_first_refusal():
    model = _WatchingModel()
    rt = _runtime(model)
    whole = _Candidate(rt, PROMPT[:30])
    failing = _Candidate(rt, PROMPT[:28] + [99, 98])
    bank = _ListBank([(whole, 30), (failing, 28)], serves=None)
    bank.last_prefix_diagnostic = {}

    assert _near(rt, PROMPT, bank) is None
    assert bank.restore_asked == [28]
    assert bank.last_prefix_diagnostic["ram_miss_reason"] == "matched_out_of_range"
    assert model.alive_per_forward == []
