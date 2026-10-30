"""A snapshot entry carries the draft history it was handed by reference (#591).

On 2.12.2 the one-copy store handed the draft head's committed history to the
session bank by reference only, expecting the put to keep a live lease. Puts
that keep no live reference (OpenCode's snapshot-only tool turns, anonymous
API sessions, a postcommit without a lease) dropped that reference and stored
no history at all, so the next turn's restore matched the prompt, found no
history the committed policy can resume from, and prefilled the whole
conversation again: 0% cache hits for OpenCode (97% on 2.12.0) and 45 s
instead of 0.5 s to the first token at 71K tokens. The miss was reported as
``new_session``. These tests pin the contract: an entry without a live
reference owns a copy of its history, and the restore resumes from it.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import mtplx.generation as generation
from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
from mtplx.server import openai as server
from mtplx.session_bank import SessionBank, _tree_nbytes
from test_one_copy_conversation import RUNTIME, _advance, _bits, _conversation
from test_qwen4_fixed_m4_capacity_bucket import NATIVE, SEED, lane, pack  # noqa: F401


def _bank() -> SessionBank:
    return SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)


def test_a_snapshot_put_owns_the_history_it_was_handed_by_reference(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    trunk, mtp = _conversation(40, 64)
    history_bits = _bits(mtp[0].kv.keys[..., :39, :])
    prompt = list(range(40))
    entry = bank.put(
        runtime=RUNTIME, token_ids=prompt, cache=trunk, logits=mx.zeros((1, 16)),
        hidden=mx.ones((1, 1, 8)), keep_live_ref=False, session_id="anon-1",
        mtp_history_policy="committed", mtp_history_cache_ref=mtp,
        snapshot_epoch=40, mtp_snapshot_epoch=40,
    )
    assert entry is not None and not entry.live_ref_only
    assert entry.cache_ref is None and entry.mtp_history_cache_ref is None
    assert entry.mtp_history_snapshot is not None
    assert entry.mtp_snapshot_epoch == entry.snapshot_epoch == 40
    # The copy is billed to the entry like the trunk snapshot.
    assert entry.nbytes >= _tree_nbytes(entry.mtp_history_snapshot) > 0
    # The runtime keeps decoding into its live caches; the entry's copy stays.
    _advance(trunk, mtp, 6)
    restored = bank.restore(
        RUNTIME, prompt + [7], mode="clone", session_id="anon-1",
        mtp_history_policy="committed",
    )
    assert restored is not None and restored.restore_mode == "clone"
    assert restored.mtp_history_cache is not None
    assert restored.mtp_history_cache is not mtp
    assert restored.mtp_history_cache[0].offset == 39
    assert np.array_equal(
        _bits(restored.mtp_history_cache[0].kv.keys[..., :39, :]), history_bits
    )


def test_a_lease_put_still_holds_the_history_by_reference(monkeypatch):
    """The fix copies nothing on the one-copy lease path."""

    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    trunk, mtp = _conversation(40, 64)
    entry = bank.put(
        runtime=RUNTIME, token_ids=list(range(40)), cache=trunk,
        logits=mx.zeros((1, 16)), hidden=mx.ones((1, 1, 8)), keep_live_ref=True,
        session_id="s", mtp_history_policy="committed", mtp_history_cache_ref=mtp,
        snapshot_epoch=40, mtp_snapshot_epoch=40,
    )
    assert entry.live_ref_only and entry.mtp_history_cache_ref is mtp
    assert entry.mtp_history_snapshot is None


def test_the_final_commit_of_a_snapshot_only_turn_banks_a_resumable_entry(monkeypatch):
    """The server's generation-final put on a one-copy runtime, keep_live_ref off.

    OpenCode tool turns (``opencode_snapshot_only``) and anonymous sessions
    without tools take this path; it is the exact pairing #591 traced.
    """

    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    trunk, mtp = _conversation(40, 64)
    state = SimpleNamespace(
        runtime=SimpleNamespace(qwen4_fixed_m4_compiled_verify=True, backend_id="native_mtp"),
        backend_descriptor=SimpleNamespace(backend_id="native_mtp"),
    )
    final_state = SimpleNamespace(final_committed_mtp_cache=mtp)
    metadata = server._generation_final_bank_metadata(state, final_state, token_count=40)
    assert metadata["mtp_history_cache_ref"] is mtp
    bank = _bank()
    entry = bank.put(
        runtime=RUNTIME, token_ids=list(range(40)), cache=trunk,
        logits=mx.zeros((1, 16)), hidden=mx.ones((1, 1, 8)), keep_live_ref=False,
        session_id="anon-2", snapshot_epoch=40, **metadata,
    )
    assert entry is not None and entry.mtp_history_snapshot is not None
    restored = bank.restore(
        RUNTIME, list(range(40)) + [3], mode="clone", session_id="anon-2",
        mtp_history_policy=metadata["mtp_history_policy"],
    )
    assert restored is not None and restored.mtp_history_cache is not None


def _snapshot_only_turns(pack, monkeypatch, *, one_copy: bool):
    """Two agent turns on the tiny fixed-M4 lane, banked the way the server
    banks a turn that keeps no live reference (keep_live_ref off)."""

    smoke, model = pack
    monkeypatch.setenv("MTPLX_ONE_COPY", "1" if one_copy else "0")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)
    state = SimpleNamespace(
        runtime=rt, backend_descriptor=SimpleNamespace(backend_id="native_mtp")
    )
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    prompt = [(7 * i + 3) % 128 for i in range(48)]
    results = []
    for turn in range(2):
        result = generation.generate_mtpk(
            rt, list(prompt), max_tokens=24, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=SEED + turn, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched",
            stop_token_ids=set(), capture_final_state=True, session_bank=bank,
            session_id="anon-t", session_restore_mode="reference",
            commit_prompt_state_to_bank=True,
        )
        final = result.final_state
        assert final is not None and final.safe_to_commit
        tokens = list(prompt) + list(result.tokens)
        metadata = server._generation_final_bank_metadata(
            state, final, token_count=len(tokens)
        )
        metadata["mtp_history_policy"] = final.mtp_history_policy
        metadata["hidden_variant"] = hidden_variant
        entry = bank.put(
            runtime=rt, token_ids=tokens, cache=final.final_trunk_cache,
            logits=final.final_logits, hidden=final.final_hidden, keep_live_ref=False,
            session_id="anon-t", snapshot_epoch=len(tokens), **metadata,
        )
        assert entry is not None
        results.append((list(result.tokens), entry, result))
        prompt = tokens + [5, 6, 7, 8]
    return results


def test_a_snapshot_only_session_resumes_its_turn_and_matches_the_copying_store(pack, lane):
    one = _snapshot_only_turns(pack, lane, one_copy=True)
    two = _snapshot_only_turns(pack, lane, one_copy=False)
    second = one[1][2]
    assert second.stats.graphbank["compiled_verify"]["fixed_m4"]["installed"]
    # 2.12.2: 0 cached tokens and a cold restore on every follow-up turn.
    assert second.stats.cached_tokens == len(one[0][1].token_ids)
    assert second.stats.session_restore_mode == "clone"
    assert two[1][2].stats.cached_tokens == len(two[0][1].token_ids)
    # Resuming from the entry's own copy of the history decodes exactly the
    # tokens the copying store decodes.
    assert [tokens for tokens, _e, _r in one] == [tokens for tokens, _e, _r in two]


def test_a_restore_without_history_reports_why_it_went_cold(pack, lane):
    """An entry 2.12.2 wrote (no history at all) matches and is refused: the
    request log must say so instead of the server's default new_session."""

    smoke, model = pack
    lane.setenv("MTPLX_ONE_COPY", "1")
    lane.setenv("MTPLX_COMPILED_VERIFY", "1")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    prompt = [(5 * i + 1) % 128 for i in range(48)]
    first = generation.generate_mtpk(
        rt, list(prompt), max_tokens=8, sampler=NATIVE, draft_sampler=NATIVE,
        speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
        mtp_history_policy="committed", verify_strategy="batched",
        stop_token_ids=set(), capture_final_state=True,
    )
    final = first.final_state
    tokens = list(prompt) + list(first.tokens)
    bank.put(
        runtime=rt, token_ids=tokens, cache=final.final_trunk_cache,
        logits=final.final_logits, hidden=final.final_hidden, keep_live_ref=False,
        session_id="anon-old", hidden_variant=hidden_variant,
        mtp_history_policy=final.mtp_history_policy, snapshot_epoch=len(tokens),
    )
    second = generation.generate_mtpk(
        rt, tokens + [9, 9], max_tokens=4, sampler=NATIVE, draft_sampler=NATIVE,
        speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
        mtp_history_policy="committed", verify_strategy="batched",
        stop_token_ids=set(), session_bank=bank, session_id="anon-old",
        session_restore_mode="reference",
    )
    assert second.stats.cached_tokens == 0
    assert second.stats.cache_miss_reason == "no_snapshot_coverage"


def test_an_identical_retry_resumes_from_the_answer_lease_and_decodes_the_same(pack, lane):
    """A regenerate sends the same prompt again; the answer's lease still holds
    the prompt-end state and its logits, so the retry restores instead of
    re-reading the prompt, and with the same seed it decodes the same tokens."""

    smoke, model = pack
    lane.setenv("MTPLX_ONE_COPY", "1")
    lane.setenv("MTPLX_COMPILED_VERIFY", "1")
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    # The prompt is banked as a lease with its prompt-end anchor, as a real
    # prompt of 1,024 tokens or more is.
    lane.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "16")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)
    state = SimpleNamespace(
        runtime=rt, backend_descriptor=SimpleNamespace(backend_id="native_mtp")
    )
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    # Long enough for the near-prefix lookup's block floor and minimum match.
    prompt = [(11 * i + 2) % 128 for i in range(600)]
    runs = []
    for _attempt in range(2):
        result = generation.generate_mtpk(
            rt, list(prompt), max_tokens=16, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched",
            stop_token_ids=set(), capture_final_state=True, session_bank=bank,
            session_id="chat", session_restore_mode="reference",
            commit_prompt_state_to_bank=True,
        )
        final = result.final_state
        tokens = list(prompt) + list(result.tokens)
        metadata = server._generation_final_bank_metadata(
            state, final, token_count=len(tokens)
        )
        metadata["mtp_history_policy"] = final.mtp_history_policy
        metadata["hidden_variant"] = hidden_variant
        bank.put(
            runtime=rt, token_ids=tokens, cache=final.final_trunk_cache,
            logits=final.final_logits, hidden=final.final_hidden, keep_live_ref=True,
            session_id="chat", snapshot_epoch=len(tokens), **metadata,
        )
        runs.append(result)
    assert runs[0].stats.cached_tokens == 0
    assert runs[1].stats.cached_tokens == len(prompt)
    assert list(runs[1].tokens) == list(runs[0].tokens)
