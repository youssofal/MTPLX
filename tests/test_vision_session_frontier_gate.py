"""Vision session-frontier gate (2.8.0 pillar alias leg).

The engine-session frontier is raw-token-id keyed and has no image
identity: every image pad shares one vocab id, so a committed vision
frontier lets a later request with DIFFERENT pixels but identical pad
ids restore another image's KV. The pillar gate's correctness sentinel
(``different_image_alias_blocked``) caught exactly that once the F39
frontier fix made these histories commit.

Gated behavior, proven here on the F39 harness pattern
(``test_final_committed_frontier_byte_skip``):

  1. a vision history NEVER advances the session frontier — both the
     stored arm and the oversized-projection arm stamp
     ``vision_session_frontier_skip``;
  2. the bank store still proceeds, keyed by content surrogates
     (``vision_bank_key_ids``): the raw id sequence cannot find the
     entry, the surrogate sequence finds it at full length;
  3. the same ids keyed for DIFFERENT pixels can never match the entry
     (the alias-blocking property, stated as data);
  4. text-only histories keep the F39 contract: the frontier commits.

The two foreground ``session.commit`` sites in ``chat_completions``
share the same ``vision_splice is None`` predicate; their integration
coverage is the pillar gate itself (``scripts/pillar_gate_qa.py``,
vision_cache leg) which the release pipeline runs against a live
daemon.

CPU-only: tiny deterministic model, real ``SessionBank``; no model
packs, no GPU.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.engine_session import EngineSession
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.server import openai as oa
from mtplx.session_bank import SessionBank
from mtplx.vision.splice import vision_bank_key_ids

VOCAB = 32
PAD = 31
# Text tokens stay in [0, 30) so PAD occurrences are exactly the ones we
# place: 40 text tokens, one 6-pad image, 12 trailing text tokens.
TEXT_PREFIX = [(i * 5 + 3) % 30 for i in range(40)]
TEXT_TAIL = [(i * 7 + 1) % 30 for i in range(12)]
PAD_COUNT = 6
VISION_HISTORY = TEXT_PREFIX + [PAD] * PAD_COUNT + TEXT_TAIL
TEXT_HISTORY = TEXT_PREFIX + TEXT_TAIL
BLUE_DIGEST = 0x1122334455667788
RED_DIGEST = 0x99AABBCCDDEEFF00
POLICY = "vision-frontier-gate-policy"

_MIX = mx.array(
    [[((i * 7 + j * 13) % 31) - 15 for j in range(VOCAB)] for i in range(VOCAB)],
    dtype=mx.float32,
)


def _splice(digest: int) -> SimpleNamespace:
    """Minimal stand-in carrying exactly the content-identity surface
    ``vision_bank_key_ids`` and the postcommit store read."""

    return SimpleNamespace(
        image_digests=[digest],
        pad_counts=[PAD_COUNT],
        image_pad_token_id=PAD,
    )


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class HistoryCountModel:
    """Causal toy model (F39 harness): logits are exact integer sums of
    the history through a fixed mixing matrix."""

    def __init__(self):
        self.calls: list[int] = []

    def make_cache(self):
        return [KVCache()]

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        del hidden_variant
        batch, length = int(input_ids.shape[0]), int(input_ids.shape[1])
        self.calls.append(length)
        onehot = mx.eye(VOCAB, dtype=mx.float32)[input_ids]
        entry = cache[0]
        keys, _values = entry.update_and_fetch(
            onehot[:, None, :, :], onehot[:, None, :, :]
        )
        hidden = mx.zeros((batch, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        counts = mx.cumsum(keys[:, 0, :, :], axis=1)
        counts = counts[:, -length:, :]
        logits = counts @ _MIX
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = logits[:, -keep:, :]
        if return_hidden:
            return logits, hidden[:, -keep:, :]
        return logits


def _runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=HistoryCountModel(),
        tokenizer=_Tokenizer(),
        model_path=Path("models/vision-frontier-gate"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


class _ForegroundState:
    """Minimal ServerState stand-in for _store_retokenized_history_snapshot."""

    def __init__(self, runtime: MTPLXRuntime, bank: SessionBank) -> None:
        self.runtime = runtime
        self.sessions = SimpleNamespace(bank=bank)
        self.session_cache_route = oa._GENERIC_SESSION_CACHE_ROUTE
        self.lock = threading.Lock()
        self.template_hash = None
        self.draft_head_identity = None

    def begin_foreground(self) -> None:
        pass

    def end_foreground(self) -> None:
        pass


def _patch_history(
    monkeypatch: pytest.MonkeyPatch, history_ids: list[int], splice
) -> None:
    monkeypatch.setattr(
        oa,
        "_history_ids_for_postcommit",
        lambda *_args, **_kwargs: (list(history_ids), splice),
    )
    if splice is not None:
        # The store's committed-history prefill feeds the splice's
        # embedding rows into the model; the toy model is id-driven, and
        # nothing in this file asserts KV content — only keying and
        # frontier behavior. Strip the vision kwarg so the real prefill
        # runs on ids while the store's keying still sees the splice.
        real_prefill = oa.restore_or_prefill_prompt_state

        def _prefill_without_vision(*args, **kwargs):
            kwargs.pop("vision_splice", None)
            return real_prefill(*args, **kwargs)

        monkeypatch.setattr(
            oa, "restore_or_prefill_prompt_state", _prefill_without_vision
        )


def _run_postcommit(state: _ForegroundState, session: EngineSession | None) -> dict:
    return oa._store_retokenized_history_snapshot(
        state,
        session_id="vision-gate",
        messages=[],
        assistant_content="turn answer",
        thinking_enabled=False,
        policy_fingerprint=POLICY,
        session=session,
        expected_session_revision=(
            session.revision if session is not None else None
        ),
        keep_live_ref=True,
    )


def _bank() -> SessionBank:
    return SessionBank(max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30)


def test_vision_history_skips_frontier_but_banks_surrogate_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    _patch_history(monkeypatch, VISION_HISTORY, _splice(BLUE_DIGEST))

    outcome = _run_postcommit(state, session)

    # Bank store proceeded; the session frontier deliberately stayed put.
    assert outcome["stored"] is True, outcome
    assert outcome["session_commit"] == {
        "committed": False,
        "reason": "vision_session_frontier_skip",
        "prefix_len": 0,
    }
    assert list(session.committed_token_ids) == []
    assert session.prefix_len == 0

    # The entry is content-keyed: raw pad ids cannot find it, the
    # surrogate view finds it at full length.
    assert bank.longest_prefix(VISION_HISTORY) is None
    blue_keys = vision_bank_key_ids(VISION_HISTORY, _splice(BLUE_DIGEST))
    entry = bank.longest_prefix(blue_keys)
    assert entry is not None
    assert entry.prefix_len == len(VISION_HISTORY)


def test_different_pixels_cannot_match_the_banked_vision_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    _patch_history(monkeypatch, VISION_HISTORY, _splice(BLUE_DIGEST))
    assert _run_postcommit(state, session)["stored"] is True

    # Same token ids, different pixels: the surrogate sequences diverge
    # at the first pad, so the full-length entry can never serve them.
    red_keys = vision_bank_key_ids(VISION_HISTORY, _splice(RED_DIGEST))
    blue_keys = vision_bank_key_ids(VISION_HISTORY, _splice(BLUE_DIGEST))
    first_pad = len(TEXT_PREFIX)
    assert red_keys[:first_pad] == blue_keys[:first_pad]
    assert red_keys[first_pad] != blue_keys[first_pad]
    assert bank.longest_prefix(red_keys) is None


def test_oversized_vision_projection_also_skips_the_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate site 1: the oversized-projection arm (which F39 taught to
    commit the frontier without storing) must skip for vision too."""

    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    _patch_history(monkeypatch, VISION_HISTORY, _splice(BLUE_DIGEST))
    monkeypatch.setattr(
        oa,
        "_estimate_retokenized_snapshot_nbytes",
        lambda *_args, **_kwargs: (1 << 40, 0),
        raising=False,
    )
    # Whatever arm the projection helper takes, the invariant under test
    # is frontier stasis; tolerate either stored outcome.
    outcome = _run_postcommit(state, session)
    commit = outcome.get("session_commit")
    assert commit is not None, outcome
    assert commit["committed"] is False
    assert commit["reason"] == "vision_session_frontier_skip"
    assert list(session.committed_token_ids) == []


def test_text_only_history_still_commits_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the F39 contract for text sessions is untouched."""

    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    _patch_history(monkeypatch, TEXT_HISTORY, None)

    outcome = _run_postcommit(state, session)

    assert outcome["stored"] is True, outcome
    assert outcome["session_commit"] == {
        "committed": True,
        "reason": "committed_retokenized_prefix",
        "prefix_len": len(TEXT_HISTORY),
    }
    assert list(session.committed_token_ids) == [int(t) for t in TEXT_HISTORY]


def test_vision_keying_failure_stays_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A splice without content identity must neither store nor commit
    (the legacy bypass shape, never a raw-id entry)."""

    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    broken = SimpleNamespace(
        image_digests=[], pad_counts=[], image_pad_token_id=PAD
    )
    _patch_history(monkeypatch, VISION_HISTORY, broken)

    outcome = _run_postcommit(state, session)

    assert outcome["stored"] is False
    assert outcome["reason"] == "vision_keying_failed"
    assert len(bank) == 0
    assert list(session.committed_token_ids) == []
