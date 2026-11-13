"""Restore landings of a plain attention model (Llama/Mistral style: KV
only, no recurrent state) through the regular restore path, from RAM and
from the SSD tier.

Without recurrent layers the attention KV trims to any token, so a restore
lands on the matched length itself: restore point equals the match, the
checkpoint gap is 0 and the landing says hybrid false. The SSD case also
fills the restore split. Uses the tiny history-count model of
test_tail_ar_warm_restore_identity; stores live in tmp_path only.
"""

from __future__ import annotations

import time

import mlx.core as mx
from test_tail_ar_warm_restore_identity import VOCAB, _runtime

from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_bank.telemetry import ORIGIN_OTHER, ORIGIN_SAME
from mtplx.cache_state import snapshot_cache
from mtplx.generation import (
    _resolve_runtime_base_hidden_variant,
    restore_or_prefill_prompt_state,
)
from mtplx.session_bank import SessionBank

STORED = [(i * 7 + 3) % VOCAB for i in range(160)]
# Shares the first 156 stored tokens, then diverges: a near-prefix restore.
DIVERGED = STORED[:156] + [(STORED[156] + 1) % VOCAB, 2, 4, 6]
# Extends the stored tokens: an exact-prefix restore.
EXTENDED = STORED + [1, 2, 3, 4, 5]


def _bank(cold_tier=None) -> SessionBank:
    return SessionBank(
        max_entries=8,
        max_bytes=1 << 24,
        per_session_max_bytes=1 << 24,
        cold_tier=cold_tier,
    )


def _put(bank: SessionBank, *, session_id: str):
    producer = _runtime()
    state = restore_or_prefill_prompt_state(
        producer, list(STORED), base_hidden_variant=None, mtp_history_policy="cycle"
    )
    entry = bank.put_snapshot(
        runtime=producer,
        token_ids=tuple(STORED),
        cache_snapshot=snapshot_cache(state.trunk_cache),
        logits=state.logits,
        # The exact SSD restore asks for the final hidden as well.
        hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
        hidden_variant=_resolve_runtime_base_hidden_variant(producer, None),
        session_id=session_id,
        mtp_history_policy="cycle",
        snapshot_epoch=len(STORED),
    )
    assert entry is not None and not entry.has_recurrent
    return entry


def _restore(bank: SessionBank, prompt, *, session_id: str):
    return restore_or_prefill_prompt_state(
        _runtime(),
        list(prompt),
        base_hidden_variant=None,
        mtp_history_policy="cycle",
        session_bank=bank,
        session_id=session_id,
    )


def _ssd_tier(tmp_path, *, session_id: str) -> SessionBankColdTier:
    """A tier holding STORED, written from a bank that is then dropped, so
    the restoring bank has nothing in RAM."""
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    entry = _put(_bank(), session_id=session_id)
    assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
    deadline = time.time() + 10.0
    while tier.stats()["writes_completed"] < 1:
        assert time.time() < deadline, "writer did not complete"
        time.sleep(0.02)
    return tier


def _assert_attention_landing(landing, *, matched, partial, origin, source):
    assert landing["source"] == source
    assert landing["hybrid"] is False
    assert landing["partial"] is partial
    assert landing["matched_tokens"] == matched
    assert landing["restore_point_tokens"] == matched
    assert landing["checkpoint_gap_tokens"] == 0
    assert landing["origin"] == origin


def test_ram_prefix_restore_lands_on_the_match():
    bank = _bank()
    _put(bank, session_id="other")
    state = _restore(bank, DIVERGED, session_id="s1")
    assert state.cache_hit and state.cache_source == "ram"
    landing = state.restore_served["landing"]
    _assert_attention_landing(
        landing, matched=156, partial=True, origin=ORIGIN_OTHER, source="ram"
    )
    assert landing["suffix_tokens"] == len(DIVERGED) - 156
    assert "ssd_restore" not in state.restore_served
    counters = bank.to_dict()["restore_landings"]["ram"]
    assert counters["partial"] == 1 and counters["hybrid"] == 0
    assert counters["checkpoint_gap_tokens"] == 0


def test_ram_exact_restore_lands_on_the_entry():
    bank = _bank()
    _put(bank, session_id="s1")
    state = _restore(bank, EXTENDED, session_id="s1")
    _assert_attention_landing(
        state.restore_served["landing"],
        matched=len(STORED),
        partial=False,
        origin=ORIGIN_SAME,
        source="ram",
    )


def test_ssd_exact_restore_lands_on_the_entry_and_carries_the_split(tmp_path):
    tier = _ssd_tier(tmp_path, session_id="s1")
    try:
        bank = _bank(tier)
        state = _restore(bank, EXTENDED, session_id="s1")
        assert state.ssd_cache_hit
        _assert_attention_landing(
            state.restore_served["landing"],
            matched=len(STORED),
            partial=False,
            origin=ORIGIN_SAME,
            source="ssd",
        )
        split = state.restore_served["ssd_restore"]
        assert split["bytes"] > 0 and split["read_s"] > 0.0
        assert bank.to_dict()["restore_landings"]["ssd"]["exact"] == 1
    finally:
        tier.close()


def test_ssd_prefix_restore_lands_on_the_match_and_carries_the_split(tmp_path):
    tier = _ssd_tier(tmp_path, session_id="other")
    try:
        bank = _bank(tier)
        state = _restore(bank, DIVERGED, session_id="s1")
        assert state.ssd_cache_hit
        _assert_attention_landing(
            state.restore_served["landing"],
            matched=156,
            partial=True,
            origin=ORIGIN_OTHER,
            source="ssd",
        )
        split = state.restore_served["ssd_restore"]
        assert split["bytes"] > 0 and split["read_s"] > 0.0
        assert bank.to_dict()["restore_landings"]["ssd"]["partial"] == 1
    finally:
        tier.close()
