"""Through ``generate_mtpk``: a turn restored from the SSD tier below a longer conversation's end.

The request lane of #592 and #635 in miniature, on the tiny Flash-Next pack
with two QSA layers (GPU): an older, longer conversation is saved to the SSD
tier; a new conversation shares its first 1,100 tokens (a system prompt and
tools, or a compacted history) and goes on with its own. The bank restores the
older entry's prefix at its recurrent boundary, the turn prefills its own
tokens, and the one-copy store resizes the conversation's QSA buffers to the
compiled verifier's rows.

On 2.12.2 that turn failed before its first token ("QSA pooled hold ... would
cut them"): the restore handed the cache the older entry's pooled block count,
the prefill pooled none of the new tokens' blocks, and the resize counted the
older conversation's blocks as rows to keep. Now the turn is served on the
compiled verifier and decodes exactly what the same prompt decodes with no
saved conversation at all.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

import mtplx.generation as generation
from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_state import snapshot_cache
from mtplx.session_bank import SessionBank
from test_one_copy_reserve_resize import FLASH, SEED, gpu, lane, tiny  # noqa: F401

CHUNK = 512  # the app's prefill chunk: recurrent boundaries every 512 tokens
OLDER = [(7 * i + 3) % 128 for i in range(2600)]
SHARED = 1100
NEWER = OLDER[:SHARED] + [(11 * i + 5) % 128 for i in range(200)]


def _tier(tmp_path) -> SessionBankColdTier:
    tier = SessionBankColdTier(base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=1)
    # The configured cap alone sizes the store, whatever this machine's free disk is.
    tier._free_disk_bytes = lambda: None
    return tier


def _generate(rt, lane, prompt, *, bank, session_id):
    with lane.context() as patch:
        # The served prefill: chunked, with a recurrent checkpoint at every
        # chunk edge (the shipped profiles arm it).
        patch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
        patch.setenv("MTPLX_PREFILL_CHUNK_SIZE", str(CHUNK))
        return generation.generate_mtpk(
            rt, list(prompt), max_tokens=24, sampler=FLASH, draft_sampler=FLASH,
            speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched",
            stop_token_ids=set(), capture_final_state=True,
            session_bank=bank, session_id=session_id, session_restore_mode="reference",
            commit_prompt_state_to_bank=True,
        )


def _save_to_ssd(rt, bank, tier, prompt, result, *, session_id):
    """The older turn's final commit, written to the SSD tier as a server commit is."""

    final = result.final_state
    assert final is not None and final.safe_to_commit
    tokens = list(prompt) + list(result.tokens)
    entry = bank.put(
        runtime=rt, token_ids=tokens, cache=final.final_trunk_cache,
        logits=final.final_logits, hidden=final.final_hidden,
        hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
        session_id=session_id, mtp_history_policy=final.mtp_history_policy,
        mtp_history_snapshot=snapshot_cache(final.final_committed_mtp_cache),
        snapshot_epoch=len(tokens), mtp_snapshot_epoch=len(tokens),
    )
    assert entry is not None and entry.gdn_boundaries
    assert tier.flush(timeout_s=30.0) is True
    bank.clear()
    return tokens


@gpu
def test_a_turn_restored_below_a_longer_conversation_is_served_and_decodes_as_cold(
    tiny, lane, tmp_path,
):
    tier = _tier(tmp_path)
    try:
        bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32, cold_tier=tier)
        older = _generate(tiny, lane, OLDER, bank=bank, session_id="older")
        saved = _save_to_ssd(tiny, bank, tier, OLDER, older, session_id="older")
        assert tier.stats()["entries"] >= 1

        restored = _generate(tiny, lane, NEWER, bank=bank, session_id="newer")
        stats = restored.stats
        # It was the SSD tier's prefix of the older conversation that served.
        assert stats.cache_source == "ssd"
        assert str(stats.session_restore_mode).startswith("ssd_block_prefix")
        assert 0 < stats.cached_tokens <= SHARED
        # The resize ran and the compiled verifier took the turn.
        admission = dict(stats.fixed_m4_admission or {})
        assert admission.get("engaged") is True, admission
        assert admission.get("promotion") == "resized", admission
        # The saved conversation's block count was past the bank's blocks.
        assert len(saved) // 4 > admission["promotion_rows"] // 4
        verifier = dict((stats.graphbank or {}).get("compiled_verify") or {})
        assert verifier["fixed_m4"]["installed"] is True
        assert verifier["compiled_calls"] > 0 and verifier["fallback_calls"] == 0

        cold = _generate(
            tiny, lane, NEWER,
            bank=SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32),
            session_id="cold",
        )
        assert cold.stats.cached_tokens == 0
        assert list(restored.tokens) == list(cold.tokens)
        assert mx.array_equal(
            restored.final_state.final_logits, cold.final_state.final_logits
        ).item()
    finally:
        tier.close()
