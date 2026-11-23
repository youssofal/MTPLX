"""Resident-duplicate shadowing of cold prefix lookups.

The 2026-08-06 causal probe pair measured 0.66-1.17s of unattributed
prompt-state wall per warm turn: near_prefix_candidates() called the cold
tier's lookup_prefix_boundary() unconditionally, which fully hydrated
(_restore_row) a candidate that then LOST the combined sort to its own
RAM-resident twin. The fix shadows ONLY a serve-equivalent resident
duplicate of the cold metadata candidate — same token hash and stored
length, identity-compatible with the request, snapshot-capable, boundary-
covered, and matching the row's committed-MTP coverage. Raw RAM matches
are never a floor: generation may later reject them, and an ineligible
RAM match must never suppress a valid cold candidate or cold-only
recovery.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank
from mtplx.cache_bank import SessionBankColdTier


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


# 41-token stored prefix; the prompt shares its first 40 tokens then
# diverges -> matched=40, gap=1: the tiny-gap near-prefix case the live
# receipts showed (matched == prefix_len routes to the EXACT path instead
# and the near-prefix loop rejects it, so a gap-0 fixture would test
# nothing).
PREFIX = list(range(1, 42))
PROMPT = tuple(PREFIX[:40] + [99, 98])
MODEL = str(Path("models/example"))


def _tier(tmp_path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )


def _bank(cold) -> SessionBank:
    return SessionBank(
        max_entries=8, max_bytes=4096, per_session_max_bytes=4096, cold_tier=cold
    )


def _put(bank: SessionBank, token_ids, *, template_hash=None, mtp_snapshot=None):
    return bank.put_snapshot(
        runtime=_runtime(),
        token_ids=token_ids,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        template_hash=template_hash,
        mtp_history_snapshot=mtp_snapshot,
        snapshot_epoch=len(token_ids),
        nbytes_override=128,
    )


def _spy_restore_row(cold: SessionBankColdTier, monkeypatch) -> list:
    calls: list = []
    original = cold._restore_row

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cold, "_restore_row", spy)
    return calls


def _candidates(bank: SessionBank, **overrides):
    kwargs = dict(
        min_matched_tokens=4,
        block_size=8,
        block_min_matched_tokens=8,
        model_path=MODEL,
        mtp_enabled=True,
    )
    kwargs.update(overrides)
    return bank.near_prefix_candidates(PROMPT, **kwargs)


def test_resident_duplicate_shadows_cold_without_hydration(tmp_path, monkeypatch):
    """The measured causal case: the same put resident in RAM and on SSD."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX)
        assert cold.flush(timeout_s=5.0) is True
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert matches and matches[0][1] == 40
        assert str(getattr(matches[0][0], "cache_source", "ram") or "ram") == "ram"
        assert calls == [], "resident-duplicate lookup must not hydrate from SSD"
        assert cold.stats().get("prefix_lookups_shadowed_by_ram") == 1
    finally:
        cold.close()


def test_identity_incompatible_higher_ram_match_does_not_shadow_valid_cold(
    tmp_path, monkeypatch
):
    """Regression guard: a raw-higher but request-incompatible RAM entry
    must never floor out the valid SSD candidate."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX, template_hash="template-a")
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        # Higher raw match (48 tokens of the prompt region... longer stored
        # prefix), but stored under a DIFFERENT template identity.
        longer = list(range(1, 41)) + [99, 98, 97, 96, 95, 94, 93, 92]
        _put(bank, longer, template_hash="template-other")
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank, template_hash="template-a")
        # The valid SSD candidate must still hydrate and be present.
        assert calls == [1], "valid cold candidate must hydrate despite raw RAM match"
        ssd = [m for m in matches if getattr(m[0], "cache_source", None) == "ssd"]
        assert ssd and ssd[0][1] == 40
        assert not cold.stats().get("prefix_lookups_shadowed_by_ram")
    finally:
        cold.close()


def test_coverage_missing_ram_twin_does_not_shadow_mtp_bearing_cold_row(
    tmp_path, monkeypatch
):
    """Same tokens resident in RAM but WITHOUT committed-MTP history must not
    shadow a cold row that carries it."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(
            bank,
            PREFIX,
            mtp_snapshot=CacheSnapshot(states=(), meta_states=()),
        )
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        # Re-put the SAME tokens RAM-only (tier detached) without MTP history.
        bank.cold_tier = None
        _put(bank, PREFIX)
        bank.cold_tier = cold
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert calls == [1], "mtp-bearing cold row must hydrate past an uncovered twin"
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
    finally:
        cold.close()


def test_epoch_desynced_ram_twin_does_not_shadow_valid_cold(tmp_path, monkeypatch):
    """Regression guard: restore() rejects entries whose MTP snapshot epoch
    desynced from the trunk epoch; such a twin must not shadow the valid
    SSD copy. (put_snapshot forbids creating desync, so mutate the resident
    entry the way a production defect would present.)"""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX, mtp_snapshot=CacheSnapshot(states=(), meta_states=()))
        assert cold.flush(timeout_s=5.0) is True
        entry = next(iter(bank._entries.values()))
        entry.mtp_snapshot_epoch = int(entry.snapshot_epoch) + 1
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert calls == [1], "desynced twin must not suppress the valid cold row"
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
        assert not cold.stats().get("prefix_lookups_shadowed_by_ram")
    finally:
        cold.close()


def test_recurrent_boundary_below_min_restore_does_not_shadow_valid_cold(
    tmp_path, monkeypatch
):
    """Regression guard: generation rejects a recurrent candidate whose
    achievable boundary is <= min_restore_tokens; such a RAM match must not
    shadow a valid cold candidate."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX)  # valid 40-token SSD row
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        # RAM-only recurrent entry: 50 stored tokens sharing 40 with the
        # prompt (gap 10 > tiny limit 8 -> boundary path), only boundary at 8.
        bank.cold_tier = None
        _put(bank, list(range(1, 51)))
        bank.cold_tier = cold
        entry = next(iter(bank._entries.values()))
        entry.has_recurrent = True
        entry.gdn_boundaries = [(8, CacheSnapshot(states=(), meta_states=()), None)]
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank, min_restore_tokens=10)
        assert calls == [1], (
            "achievable-boundary-below-floor RAM match must not suppress cold"
        )
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
        assert not cold.stats().get("prefix_lookups_shadowed_by_ram")
    finally:
        cold.close()


def test_cold_only_recovery_still_hydrates(tmp_path, monkeypatch):
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX)
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert matches and matches[0][1] == 40
        assert getattr(matches[0][0], "cache_source", None) == "ssd"
        assert calls == [1]
    finally:
        cold.close()


def test_duck_tier_without_capability_marker_gets_pre_shadow_call(tmp_path):
    seen: list = []

    def legacy_lookup(
        tokens,
        *,
        model_path,
        mtp_enabled,
        hidden_variant=None,
        template_hash=None,
        mtp_history_policy=None,
        draft_head_identity=None,
        policy_fingerprint=None,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    ):
        seen.append(tokens)
        return None

    bank = SessionBank(
        max_entries=8,
        max_bytes=4096,
        per_session_max_bytes=4096,
        cold_tier=SimpleNamespace(lookup_prefix_boundary=legacy_lookup),
    )
    _put(bank, PREFIX)
    matches = _candidates(bank)
    # No capability marker -> the shadow kwarg is never passed (no TypeError,
    # no retry) and RAM still serves.
    assert len(seen) == 1
    assert matches and matches[0][1] == 40


def test_cold_row_at_or_below_the_exact_prefix_floor_is_not_hydrated(
    tmp_path, monkeypatch
):
    """2026-09-03: every warm agent turn hydrated its own SSD twin.

    The caller passes min_restore_tokens = the exact-prefix RAM entry's
    length and rejects every near/block candidate with matched <= that, so
    a cold row matching no more than the floor can never be served through
    this lane. It was still hydrated (0.59-0.66 s of unattributed wall per
    warm turn on the candidate daemon) because the `_cand <= floor` guard
    kept the exact entry out of the RAM bar, which then read 0.
    """
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX)  # exact prefix of the prompt below, resident AND on SSD
        assert cold.flush(timeout_s=5.0) is True
        calls = _spy_restore_row(cold, monkeypatch)
        prompt = tuple(PREFIX + [99, 98])  # PREFIX is an exact prefix: the warm-turn shape
        matches = bank.near_prefix_candidates(
            prompt,
            min_matched_tokens=4,
            block_size=8,
            block_min_matched_tokens=8,
            model_path=MODEL,
            mtp_enabled=True,
            min_restore_tokens=len(PREFIX),
        )
        assert calls == [], "a cold row the caller cannot serve must not hydrate"
        assert cold.stats().get("prefix_lookups_not_better_than_ram") == 1
        assert not any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
    finally:
        cold.close()


def test_a_cold_row_strictly_past_the_floor_still_hydrates(tmp_path, monkeypatch):
    """Cold-only recovery past the exact entry is untouched by the floor."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        longer = PREFIX + [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
        _put(bank, longer)
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        bank.cold_tier = None
        _put(bank, PREFIX[:20])  # the exact-prefix RAM entry the caller holds
        bank.cold_tier = cold
        calls = _spy_restore_row(cold, monkeypatch)
        prompt = tuple(PREFIX + [99, 98, 97, 96, 95, 94, 93, 92, 1, 2])
        matches = bank.near_prefix_candidates(
            prompt,
            min_matched_tokens=4,
            block_size=8,
            block_min_matched_tokens=8,
            model_path=MODEL,
            mtp_enabled=True,
            min_restore_tokens=20,
        )
        assert calls == [1], "a strictly-better cold row must still hydrate"
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
    finally:
        cold.close()


class _DummyTrimmableCache:
    def __init__(self, offset: int) -> None:
        self.offset = int(offset)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(n), self.offset)
        self.offset -= n
        return n


def test_live_recurrent_lease_shadows_cold_on_exact_length_repeat(
    tmp_path, monkeypatch
):
    """A healthy one-copy recurrent lease with boundary_logits at the exact
    repeat length serves the repeat in RAM and must not hydrate its SSD twin."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        stored = list(range(1, 51))
        _put(bank, stored)
        assert cold.flush(timeout_s=5.0) is True
        entry = next(iter(bank._entries.values()))
        entry.live_ref_only = True
        entry.has_recurrent = True
        entry.cache_ref = [_DummyTrimmableCache(len(stored))]
        entry.lease_kv_offset = len(stored)
        entry.gdn_boundaries = [
            (40, CacheSnapshot(states=(), meta_states=()), None)
        ]
        entry.boundary_logits = (40, "logits")
        calls = _spy_restore_row(cold, monkeypatch)
        matches = bank.near_prefix_candidates(
            tuple(range(1, 41)),
            min_matched_tokens=4,
            block_size=8,
            block_min_matched_tokens=8,
            model_path=MODEL,
            mtp_enabled=True,
        )
        assert calls == [], "live recurrent lease must shadow its SSD twin"
        assert matches and matches[0][1] == 40
        assert str(getattr(matches[0][0], "cache_source", "ram") or "ram") == "ram"
        assert cold.stats().get("prefix_lookups_shadowed_by_ram") == 1
    finally:
        cold.close()


def test_live_recurrent_lease_shadows_cold_on_followup_with_nonempty_suffix(
    tmp_path, monkeypatch
):
    """A follow-up prompt leaves a non-empty suffix after the recurrent boundary
    and needs no boundary_logits, so the live lease shadows its SSD twin."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        stored = list(range(1, 51))
        _put(bank, stored)
        assert cold.flush(timeout_s=5.0) is True
        entry = next(iter(bank._entries.values()))
        entry.live_ref_only = True
        entry.has_recurrent = True
        entry.cache_ref = [_DummyTrimmableCache(len(stored))]
        entry.lease_kv_offset = len(stored)
        entry.gdn_boundaries = [
            (40, CacheSnapshot(states=(), meta_states=()), None)
        ]
        entry.boundary_logits = None
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert calls == []
        assert matches and matches[0][1] == 40
        assert str(getattr(matches[0][0], "cache_source", "ram") or "ram") == "ram"
        assert cold.stats().get("prefix_lookups_shadowed_by_ram") == 1
    finally:
        cold.close()


def test_live_recurrent_lease_missing_boundary_logits_on_exact_repeat_does_not_shadow_cold(
    tmp_path, monkeypatch
):
    """A live recurrent lease without boundary_logits at an exact-length repeat
    is declined by generation and must not suppress cold rows — while a sibling
    resident lease that DOES carry boundary_logits still blocks shorter SSD rows."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        stored_50 = list(range(1, 51))
        _put(bank, stored_50)
        assert cold.flush(timeout_s=5.0) is True
        entry_50 = next(iter(bank._entries.values()))
        entry_50.live_ref_only = True
        entry_50.has_recurrent = True
        entry_50.cache_ref = [_DummyTrimmableCache(len(stored_50))]
        entry_50.lease_kv_offset = len(stored_50)
        entry_50.gdn_boundaries = [
            (40, CacheSnapshot(states=(), meta_states=()), None)
        ]
        entry_50.boundary_logits = None
        calls = _spy_restore_row(cold, monkeypatch)
        matches = bank.near_prefix_candidates(
            tuple(range(1, 41)),
            min_matched_tokens=4,
            block_size=8,
            block_min_matched_tokens=8,
            model_path=MODEL,
            mtp_enabled=True,
        )
        assert calls == [1], "lease missing boundary_logits must not shadow SSD"
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)

        # Now add a second live recurrent lease (RAM-only) that covers 41 tokens
        # WITH boundary_logits at 41: it outranks the 40-token block-aligned SSD
        # row, so SSD must stay unhydrated even while entry_50 lacks logits at 41.
        bank.cold_tier = None
        stored_45 = list(range(1, 46))
        entry_45 = _put(bank, stored_45)
        assert entry_45 is not None
        entry_45.live_ref_only = True
        entry_45.has_recurrent = True
        entry_45.cache_ref = [_DummyTrimmableCache(len(stored_45))]
        entry_45.lease_kv_offset = len(stored_45)
        entry_45.gdn_boundaries = [
            (41, CacheSnapshot(states=(), meta_states=()), None)
        ]
        entry_45.boundary_logits = (41, "logits")
        entry_50.gdn_boundaries = [
            (41, CacheSnapshot(states=(), meta_states=()), None)
        ]
        bank.cold_tier = cold
        calls.clear()
        matches_with_sibling = bank.near_prefix_candidates(
            tuple(range(1, 42)),
            min_matched_tokens=4,
            block_size=8,
            block_min_matched_tokens=8,
            model_path=MODEL,
            mtp_enabled=True,
        )
        assert calls == [], "valid sibling lease matching 41 > SSD block 40 suppresses SSD"
        assert [m[0] for m in matches_with_sibling] == [entry_50, entry_45]
        assert not any(
            getattr(m[0], "cache_source", None) == "ssd"
            for m in matches_with_sibling
        )
        assert cold.stats().get("prefix_lookups_not_better_than_ram") == 1
    finally:
        cold.close()


def test_consumed_live_lease_does_not_shadow_cold(tmp_path, monkeypatch):
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        stored = list(range(1, 51))
        _put(bank, stored)
        assert cold.flush(timeout_s=5.0) is True
        entry = next(iter(bank._entries.values()))
        entry.live_ref_only = True
        entry.has_recurrent = True
        entry.cache_ref = None
        entry.gdn_boundaries = [
            (40, CacheSnapshot(states=(), meta_states=()), None)
        ]
        entry.boundary_logits = (40, "logits")
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert calls == [1]
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
    finally:
        cold.close()


def test_advanced_live_lease_without_rewind_anchor_does_not_shadow_cold(
    tmp_path, monkeypatch
):
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        stored = list(range(1, 51))
        _put(bank, stored)
        assert cold.flush(timeout_s=5.0) is True
        entry = next(iter(bank._entries.values()))
        entry.live_ref_only = True
        entry.has_recurrent = True
        entry.cache_ref = [_DummyTrimmableCache(len(stored) + 5)]
        entry.lease_kv_offset = len(stored)
        # Anchor only at 40, missing at entry.prefix_len (50) -> unrewindable.
        entry.gdn_boundaries = [
            (40, CacheSnapshot(states=(), meta_states=()), None)
        ]
        entry.boundary_logits = (40, "logits")
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert calls == [1]
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
    finally:
        cold.close()


def test_non_recurrent_live_lease_stays_fail_open_to_cold(tmp_path, monkeypatch):
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX)
        assert cold.flush(timeout_s=5.0) is True
        entry = next(iter(bank._entries.values()))
        entry.live_ref_only = True
        entry.has_recurrent = False
        entry.cache_ref = [_DummyTrimmableCache(len(PREFIX))]
        entry.lease_kv_offset = len(PREFIX)
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert calls == [1]
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
    finally:
        cold.close()
