"""A hybrid SSD row whose restore point lies below the caller's floor is
refused before any tensor is read, and the next ranked row is tried.

A hybrid entry restores only at its newest recurrent boundary at or below
the match (SessionBank.restore_entry_prefix_cache). The near-prefix lane
refuses a candidate whose boundary does not beat its floor (its exact
prefix in RAM, or the spilled SSD entry the exact lane serves) as
``boundary_not_better``, but only after the cold tier decoded it. The row's
payload.json already says where that boundary is.
"""

from __future__ import annotations

import time

import mlx.core as mx

from mtplx.cache_bank import cold_tier
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank

HYBRID = list(range(2000))
QUERY = tuple(HYBRID[:1500] + [9999] * 64)  # matched 1488 on a 16-token grid


def _entry(tokens, *, boundaries=()):
    class Entry:
        token_ids = tuple(tokens)
        nbytes = 2048
        cache_snapshot = CacheSnapshot(
            states=[
                mx.zeros((1, 2, len(tokens), 4), dtype=mx.float16),
                mx.zeros((1, 4), dtype=mx.float16) if boundaries else None,
            ],
            meta_states=[{"offset": len(tokens)}, None],
        )
        logits = mx.zeros((1, 8), dtype=mx.float16)
        hidden = mx.zeros((1, 8), dtype=mx.float16)
        mtp_history_snapshot = None
        gdn_boundaries = tuple(
            (
                position,
                CacheSnapshot(
                    states=(None, mx.full((1, 4), 7.0, dtype=mx.float16)),
                    meta_states=(None, None),
                ),
                None,
            )
            for position in boundaries
        )
        has_recurrent = bool(boundaries)
        session_id = "s1"
        token_hash = f"hash-{len(tokens):04d}" * 2
        prefix_len = len(tokens)
        model_path = "model"
        mtp_enabled = False
        hidden_variant = None
        template_hash = None
        mtp_history_policy = None
        draft_head_identity = None
        policy_fingerprint = None

    return Entry()


def _tier(tmp_path):
    return SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)


def _store(tier, *entries):
    for entry in entries:
        assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if tier.stats()["writes_completed"] >= len(entries):
            return
        time.sleep(0.05)
    raise AssertionError("writer did not complete")


def _lookup(tier, tokens, **kwargs):
    return tier.lookup_prefix_boundary(
        tokens,
        model_path="model",
        mtp_enabled=False,
        max_token_gap=8,
        min_matched_tokens=8,
        block_size=16,
        block_min_matched_tokens=16,
        **kwargs,
    )


def _no_decode(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a refused row must not be decoded")

    monkeypatch.setattr(cold_tier, "decode_payload", refuse)
    monkeypatch.setattr(cold_tier, "decode_payload_prefix", refuse)


def test_a_boundary_below_the_floor_is_refused_before_decoding(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    _store(tier, _entry(HYBRID, boundaries=(512,)))
    _no_decode(monkeypatch)

    assert _lookup(tier, QUERY, min_restore_point=600) is None
    stats = tier.stats()
    assert stats["last_miss_reason"] == "ssd_prefix_boundary_below_floor"
    assert stats["prefix_restores_below_caller_floor"] == 1
    assert stats.get("restore_hits", 0) == 0


def test_a_boundary_at_or_above_the_floor_still_decodes(tmp_path):
    tier = _tier(tmp_path)
    _store(tier, _entry(HYBRID, boundaries=(512,)))

    for floor in (0, 400, 512):
        hit = _lookup(tier, QUERY, min_restore_point=floor)
        assert hit is not None, floor
        assert hit.record.cache_snapshot_prefix_len == 512
    assert tier.stats().get("prefix_restores_below_caller_floor", 0) == 0


def test_the_next_ranked_row_serves_when_the_longest_lands_below_the_floor(tmp_path):
    tier = _tier(tmp_path)
    attention = list(range(1200)) + [5555] * 100
    _store(tier, _entry(HYBRID, boundaries=(512,)), _entry(attention))

    hit = _lookup(tier, QUERY, min_restore_point=600)
    assert hit is not None
    assert hit.record.token_ids == tuple(attention)
    assert hit.matched_tokens == 1200
    assert tier.stats()["prefix_restores_below_caller_floor"] == 1


def test_the_bank_passes_the_near_lane_floor_to_the_tier(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    tier = _tier(tmp_path)
    try:
        _store(tier, _entry(HYBRID, boundaries=(512,)))
        bank = SessionBank(
            max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20, cold_tier=tier
        )
        _no_decode(monkeypatch)

        found = bank.near_prefix_candidates(
            QUERY,
            block_size=16,
            block_min_matched_tokens=16,
            model_path="model",
            mtp_enabled=False,
            min_restore_tokens=600,
        )
        assert found == []
        assert tier.stats()["prefix_restores_below_caller_floor"] == 1
    finally:
        tier.close()
