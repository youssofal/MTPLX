"""F39: the ~38k committed-frontier freeze (issue #255 receipts).

``_store_retokenized_history_snapshot`` early-returned on its oversized-byte
projection BEFORE ``bank.put`` AND before ``session.commit_retokenized_prefix``
with no live-ref fallback — unlike ``bank.put``'s own oversized branch, which
installs a live-reference lease (#150/#229 policy). The committed frontier
therefore froze at the last under-budget boundary (~38.3k tokens on the 8-GiB
tier at ~220 KB/token) and every later turn restored that frozen prefix and
re-prefilled a growing suffix forever (kmike: cache_read plateaued at
38335/42969; a 111k prompt hit 34% and paid 277s TTFT).

Fixed behavior, proven here on the integer-exact harness pattern of
``test_tail_ar_warm_restore_identity`` plus the SimpleNamespace state pattern
of ``test_postcommit_wait_integration``:

  1. an oversized projection with a session and live refs allowed PROCEEDS,
     routes the store through put()'s existing oversized branch
     (``nbytes_override``: no snapshot materialization) and lands a live-ref
     lease at the full canonical frontier;
  2. the session frontier (``committed_token_ids``) advances past the
     byte-skip on every arm, including the no-lease arm;
  3. the NEXT turn restores the lease and prefills only the true remainder —
     tokens-to-prefill SHRINKS versus the frozen-frontier behavior — and the
     restored state is byte-exact (greedy decode identity vs a cold run);
  4. the projection stays honest in the lease regime (leases carry
     ``oversized_nbytes``), so later postcommits keep taking the
     no-materialize branch instead of building multi-GiB snapshots that the
     cap will reject;
  5. the byte ceiling is never silent: the once-per-session #229 warning
     fires and the outcome stamps projected bytes + budget.

CPU-only: tiny deterministic model, real ``SessionBank``, real
``restore_or_prefill_prompt_state``; no model packs, no GPU.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.cache_state import snapshot_cache
from mtplx.engine_session import EngineSession
from mtplx.generation import (
    _resolve_runtime_base_hidden_variant,
    generate_ar,
    restore_or_prefill_prompt_state,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.server import openai as oa
from mtplx.session_bank import SessionBank

VOCAB = 32
# Turn 1 boundary (the last under-budget snapshot), turn 2 canonical history,
# and the next turn's new user tokens.
PREFIX = [(i * 5 + 3) % VOCAB for i in range(48)]
HISTORY = PREFIX + [(i * 7 + 1) % VOCAB for i in range(48)]
NEXT_TURN = [(i * 11 + 2) % VOCAB for i in range(8)]
POLICY = "final-f39-policy"
GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
MAX_TOKENS = 6

_MIX = mx.array(
    [[((i * 7 + j * 13) % 31) - 15 for j in range(VOCAB)] for i in range(VOCAB)],
    dtype=mx.float32,
)


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class HistoryCountModel:
    """Causal toy model: position t's logits depend on tokens[0..t].

    One-hot key history in a real KVCache; logits are integer-valued f32
    sums of the whole history through a fixed mixing matrix, so restored
    state that drops, duplicates, or corrupts any prefix token moves every
    following argmax — equality checks are exact, never approximate.
    """

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
        model_path=Path("models/final-frontier-byte-skip"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def _seed_nbytes() -> int:
    """Real snapshot size of the PREFIX boundary on this harness."""
    producer = _runtime()
    state = restore_or_prefill_prompt_state(
        producer,
        list(PREFIX),
        base_hidden_variant=None,
        mtp_history_policy="cycle",
    )
    probe = SessionBank(max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    entry = probe.put_snapshot(
        runtime=producer,
        token_ids=tuple(PREFIX),
        cache_snapshot=snapshot_cache(state.trunk_cache),
        logits=state.logits,
        hidden_variant=_resolve_runtime_base_hidden_variant(producer, None),
        session_id="probe",
        mtp_history_policy="cycle",
        policy_fingerprint=POLICY,
        snapshot_epoch=len(PREFIX),
    )
    assert entry is not None
    return int(entry.nbytes)


def _oversized_bank_with_prefix(runtime: MTPLXRuntime) -> SessionBank:
    """Bank whose budget admits the PREFIX snapshot but rejects the scaled
    HISTORY projection — the exact #255 shape (last under-budget boundary
    banked, next boundary projected oversized)."""
    seed = _seed_nbytes()
    # HISTORY is 2x PREFIX, so the projection (~2.06x seed) beats seed+4096
    # while the seed itself stays admitted.
    bank = SessionBank(
        max_entries=8,
        max_bytes=1 << 30,
        per_session_max_bytes=seed + 4096,
    )
    state = restore_or_prefill_prompt_state(
        runtime,
        list(PREFIX),
        base_hidden_variant=None,
        mtp_history_policy="cycle",
    )
    entry = bank.put_snapshot(
        runtime=runtime,
        token_ids=tuple(PREFIX),
        cache_snapshot=snapshot_cache(state.trunk_cache),
        logits=state.logits,
        hidden_variant=_resolve_runtime_base_hidden_variant(runtime, None),
        session_id="final-f39",
        mtp_history_policy="cycle",
        policy_fingerprint=POLICY,
        snapshot_epoch=len(PREFIX),
    )
    assert entry is not None, "the under-budget PREFIX snapshot must be admitted"
    assert entry.nbytes <= bank.per_session_max_bytes
    return bank


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


def _patch_history(monkeypatch: pytest.MonkeyPatch, history_ids: list[int]) -> None:
    monkeypatch.setattr(
        oa,
        "_history_ids_for_postcommit",
        lambda *_args, **_kwargs: (list(history_ids), None),
    )


def _run_postcommit(
    state: _ForegroundState,
    session: EngineSession | None,
    *,
    keep_live_ref: bool = True,
) -> dict:
    return oa._store_retokenized_history_snapshot(
        state,
        session_id="final-f39",
        messages=[],
        assistant_content="turn answer",
        thinking_enabled=False,
        policy_fingerprint=POLICY,
        session=session,
        expected_session_revision=(
            session.revision if session is not None else None
        ),
        keep_live_ref=keep_live_ref,
    )


def test_oversized_projection_lands_live_ref_lease_and_advances_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    bank = _oversized_bank_with_prefix(runtime)
    state = _ForegroundState(runtime, bank)
    session = EngineSession("final-f39")
    frozen_boundary = len(PREFIX)
    _patch_history(monkeypatch, HISTORY)

    outcome = _run_postcommit(state, session)

    # The store proceeded through put()'s oversized branch: a live-ref lease
    # at the FULL canonical frontier, no snapshot bytes, projection stamped.
    assert outcome["stored"] is True, outcome
    assert outcome["prefix_len"] == len(HISTORY)
    assert outcome["nbytes"] == 0
    assert outcome["live_ref_lease"] is True
    assert outcome["estimated_nbytes"] > outcome["budget"]
    assert bank.last_put_skipped_oversized_snapshot is True
    assert bank.eviction_log[-1]["fallback"] == "live_reference_lease"
    lease = bank.longest_prefix(HISTORY)
    assert lease is not None and lease.live_ref_only is True
    assert lease.prefix_len == len(HISTORY) > frozen_boundary
    # Lease-regime projection stays honest: the rejected size is recorded.
    assert lease.oversized_nbytes == outcome["estimated_nbytes"]

    # The committed frontier advanced past the byte-skip (the freeze bug was
    # exactly this staying at frozen_boundary forever).
    assert outcome["session_commit"] == {
        "committed": True,
        "reason": "committed_retokenized_prefix",
        "prefix_len": len(HISTORY),
    }
    assert list(session.committed_token_ids) == [int(t) for t in HISTORY]
    assert session.prefix_len > frozen_boundary


def test_next_turn_prefill_shrinks_and_restore_is_byte_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The receipts' acceptance shape: cache_read grows past the frozen
    boundary and the next turn prefills only the true remainder — with a
    greedy-decode identity proof that the lease restored the right bytes."""
    runtime = _runtime()
    bank = _oversized_bank_with_prefix(runtime)
    state = _ForegroundState(runtime, bank)
    session = EngineSession("final-f39")
    frozen_boundary = len(PREFIX)
    _patch_history(monkeypatch, HISTORY)

    outcome = _run_postcommit(state, session)
    assert outcome["stored"] is True, outcome

    next_prompt = list(HISTORY) + list(NEXT_TURN)

    # Cold baseline on a fresh runtime: full prefill, greedy decode.
    cold = generate_ar(
        _runtime(),
        list(next_prompt),
        max_tokens=MAX_TOKENS,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
    )
    assert len(cold.tokens) == MAX_TOKENS

    # Warm turn through the lease (the default foreground restore mode is
    # reference_lease). The runtime is fresh: only the banked lease can
    # supply the prefix state.
    warm_runtime = _runtime()
    warm = generate_ar(
        warm_runtime,
        list(next_prompt),
        max_tokens=MAX_TOKENS,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
        session_bank=bank,
        session_id="final-f39",
        session_restore_mode="reference",
        session_policy_fingerprint=POLICY,
    )

    # Prefill work SHRANK versus the frozen-frontier behavior: the frozen
    # bank could serve at most `frozen_boundary` cached tokens, leaving
    # len(next_prompt) - frozen_boundary to re-prefill. The lease serves the
    # full canonical frontier, leaving only the new turn's tokens.
    assert warm.stats.session_cache_hit is True
    assert warm.stats.cached_tokens == len(HISTORY) > frozen_boundary
    assert warm.stats.new_prefill_tokens == len(NEXT_TURN)
    assert warm.stats.new_prefill_tokens < len(next_prompt) - frozen_boundary
    assert len(next_prompt) not in warm_runtime.model.calls

    # Nothing restored wrong bytes: byte-identical greedy decode.
    assert list(warm.tokens) == list(cold.tokens)


def test_frozen_frontier_control_shows_the_shrink_is_real() -> None:
    """Teeth: WITHOUT the postcommit (the pre-fix skip), the same bank serves
    only the frozen boundary and the next turn re-prefills the growing
    suffix — the behavior the receipts show plateauing at 38335."""
    runtime = _runtime()
    bank = _oversized_bank_with_prefix(runtime)
    next_prompt = list(HISTORY) + list(NEXT_TURN)

    frozen_runtime = _runtime()
    frozen = generate_ar(
        frozen_runtime,
        list(next_prompt),
        max_tokens=MAX_TOKENS,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
        session_bank=bank,
        session_id="final-f39",
        session_restore_mode="reference",
        session_policy_fingerprint=POLICY,
    )
    assert frozen.stats.cached_tokens == len(PREFIX)
    assert frozen.stats.new_prefill_tokens == len(next_prompt) - len(PREFIX)


def test_second_oversized_postcommit_projects_from_the_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lease regime: once the longest banked prefix is a zero-byte lease, the
    projection must keep reading oversized (via oversized_nbytes) and keep
    routing put() through the no-materialize override branch."""
    runtime = _runtime()
    bank = _oversized_bank_with_prefix(runtime)
    state = _ForegroundState(runtime, bank)
    session = EngineSession("final-f39")
    _patch_history(monkeypatch, HISTORY)
    first = _run_postcommit(state, session)
    assert first["stored"] is True and first["live_ref_lease"] is True

    history2 = list(HISTORY) + list(NEXT_TURN)
    _patch_history(monkeypatch, history2)
    second = _run_postcommit(state, session)

    assert second["stored"] is True, second
    assert second["live_ref_lease"] is True
    assert second["estimated_nbytes"] > second["budget"]
    # put() took the override branch: last_put_nbytes is the projection, not
    # a computed snapshot size (nothing was materialized to compute one).
    assert bank.last_put_nbytes == second["estimated_nbytes"]
    assert bank.last_put_skipped_oversized_snapshot is True
    lease2 = bank.longest_prefix(history2)
    assert lease2 is not None and lease2.prefix_len == len(history2)
    assert list(session.committed_token_ids) == [int(t) for t in history2]


def test_no_lease_arm_still_commits_frontier_without_model_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """keep_live_ref=False (forked busy session): the prefill+put is provably
    doomed and stays skipped — but the frontier still advances and the skip
    is stamped with the projection, never silent."""
    runtime = _runtime()
    bank = _oversized_bank_with_prefix(runtime)
    state = _ForegroundState(runtime, bank)
    session = EngineSession("final-f39")
    _patch_history(monkeypatch, HISTORY)

    def _no_model_work(*_args, **_kwargs):
        raise AssertionError("no-lease oversized skip must not prefill")

    monkeypatch.setattr(oa, "restore_or_prefill_prompt_state", _no_model_work)

    outcome = _run_postcommit(state, session, keep_live_ref=False)

    assert outcome["stored"] is False
    assert outcome["reason"] == "estimated_oversized_snapshot"
    assert outcome["estimated_nbytes"] > outcome["budget"]
    assert outcome["session_commit"] == {
        "committed": True,
        "reason": "committed_retokenized_prefix",
        "prefix_len": len(HISTORY),
    }
    assert list(session.committed_token_ids) == [int(t) for t in HISTORY]
    assert len(bank) == 1, "no bank entry may appear on the no-lease arm"


def test_sessionless_skip_shape_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """session=None keeps the pre-fix early-skip contract exactly (the shape
    test_postcommit_wait_integration pins): no model work, no commit."""
    runtime = _runtime()
    bank = _oversized_bank_with_prefix(runtime)
    state = _ForegroundState(runtime, bank)
    _patch_history(monkeypatch, HISTORY)

    def _no_model_work(*_args, **_kwargs):
        raise AssertionError("sessionless oversized skip must not prefill")

    monkeypatch.setattr(oa, "restore_or_prefill_prompt_state", _no_model_work)

    outcome = _run_postcommit(state, None)

    assert outcome["stored"] is False
    assert outcome["reason"] == "estimated_oversized_snapshot"
    assert "session_commit" not in outcome
    assert len(bank) == 1


def test_byte_ceiling_warning_fires_once_per_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Visibility (#229 idiom): the ceiling warns loudly exactly once per
    session, naming the projected need, the cap, and the knob."""
    runtime = _runtime()
    bank = _oversized_bank_with_prefix(runtime)
    state = _ForegroundState(runtime, bank)
    session = EngineSession("final-f39")
    _patch_history(monkeypatch, HISTORY)

    first = _run_postcommit(state, session)
    assert first["stored"] is True
    out = capsys.readouterr().out
    assert "session-bank snapshot skipped" in out
    assert "MTPLX_SESSION_BANK_PER_SESSION_BYTES" in out
    assert "final-f39" in out

    history2 = list(HISTORY) + list(NEXT_TURN)
    _patch_history(monkeypatch, history2)
    second = _run_postcommit(state, session)
    assert second["stored"] is True
    assert "session-bank snapshot skipped" not in capsys.readouterr().out
