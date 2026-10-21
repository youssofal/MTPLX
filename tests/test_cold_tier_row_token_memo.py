"""Prefix-boundary lookup: SQL length floor and parsed-row memo.

The lookup must pick exactly what the unfiltered, unmemoized scan picks, with
the same miss reasons and counters. The reference below is that scan, written
out against the same manifest rows.
"""

from __future__ import annotations

import json
import random

import pytest

from mtplx.cache_bank import cold_tier as cold_tier_module
from mtplx.cache_bank.cold_tier import (
    ColdPrefixRestoreRecord,
    SessionBankColdTier,
    block_aligned_prefix_len,
    common_prefix_len,
)

BLOCK = 16
GAP = 8
MIN_MATCH = 8


def _meta(entry_id, tokens, *, last_access=0.0):
    tokens = list(tokens)
    return {
        "entry_id": entry_id,
        "token_hash": f"hash-{entry_id}-{len(tokens)}",
        "prefix_len": len(tokens),
        "token_ids": tokens,
        "model_path": "model",
        "mtp_enabled": False,
        "snapshot_epoch": len(tokens),
        "capabilities": [],
        "block_size": BLOCK,
        "block_hashes": [],
        "entry_dir": f"entries/{entry_id}",
        "nbytes": 1,
        "created_at_s": 0.0,
    }


class _Stub:
    def __init__(self, entry_id):
        self.entry_id = entry_id


@pytest.fixture()
def tier(tmp_path, monkeypatch):
    t = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    calls = []

    def fake_restore(row, tokens, **kw):
        calls.append((str(row["entry_id"]), kw.get("prefix_restore_tokens")))
        return _Stub(str(row["entry_id"]))

    monkeypatch.setattr(t, "_restore_row", fake_restore)
    t.calls = calls
    yield t
    t.close()


def _reference(tier, tokens, *, floor):
    """The pre-change scan: every row parsed and ranked, floor applied last."""
    with tier._connect() as conn:
        rows = list(
            conn.execute(
                "SELECT * FROM entries WHERE model_path = ? AND mtp_enabled = ? "
                "ORDER BY prefix_len DESC, last_access_s DESC",
                ("model", 0),
            ).fetchall()
        )
    block_min = max(BLOCK, BLOCK)
    ranked = []
    for row in rows:
        prefix = tuple(int(t) for t in json.loads(row["token_ids_json"]))
        if not prefix:
            continue
        matched = common_prefix_len(tokens, prefix)
        gap = len(prefix) - matched
        safe = min(
            block_aligned_prefix_len(matched, block_size=BLOCK), len(prefix), len(tokens)
        )
        required = min(MIN_MATCH, max(1, len(prefix) - GAP))
        near = 0 <= gap <= GAP and matched >= required
        block = safe >= block_min and safe >= 2 and safe <= matched
        if near:
            ranked.append(((matched, matched, len(prefix)), row["entry_id"], matched, "near_prefix"))
        elif block:
            ranked.append(((safe, matched, len(prefix)), row["entry_id"], safe, "block_prefix"))
    if not ranked:
        return None, "ssd_prefix_miss"
    ranked.sort(key=lambda item: item[0], reverse=True)
    if floor > 0 and ranked[0][2] < floor:
        return None, "ssd_prefix_not_better_than_ram"
    best = ranked[0]
    return (best[1], best[2], best[3]), None


def _lookup(tier, tokens, floor):
    tier.calls.clear()
    return tier.lookup_prefix_boundary(
        tokens,
        model_path="model",
        mtp_enabled=False,
        max_token_gap=GAP,
        min_matched_tokens=MIN_MATCH,
        block_size=BLOCK,
        block_min_matched_tokens=BLOCK,
        min_useful_matched_tokens=floor,
    )


def _populate(tier, rng):
    base = [rng.randrange(1000, 60000) for _ in range(400)]
    lengths = [20, 64, 100, 128, 200, 256, 330, 400]
    for i, n in enumerate(lengths):
        toks = base[:n]
        if i % 3 == 1:
            toks = toks[: n - 5] + [7000 + i] * 5  # diverges near the end
        if i % 4 == 3:
            toks = toks[:40] + [8000 + j for j in range(n - 40)]  # diverges early
        tier._insert_manifest(_meta(f"e{i}", toks, last_access=i))
    return base


def test_filter_and_memo_match_reference(tier):
    rng = random.Random(7)
    base = _populate(tier, rng)
    queries = [
        tuple(base[:n]) + tuple(rng.randrange(70000, 80000) for _ in range(k))
        for n in (30, 64, 129, 250, 331, 400)
        for k in (0, 3, 40)
    ]
    for _round in range(2):  # second round runs on a warm memo
        for query in queries:
            for floor in (0, 1, 64, 129, 200, 331, 400, 500):
                expected, reason = _reference(tier, query, floor=floor)
                got = _lookup(tier, query, floor)
                if expected is None:
                    assert got is None
                    assert tier.stats()["last_miss_reason"] == reason
                else:
                    assert isinstance(got, ColdPrefixRestoreRecord)
                    assert (got.record.entry_id, got.matched_tokens, got.restore_kind) == expected
                    assert tier.calls[0] == (expected[0], expected[1])


def test_sql_floor_keeps_rows_at_the_boundary(tier):
    toks = list(range(100))
    tier._insert_manifest(_meta("a", toks))
    query = tuple(toks[:100]) + (9999,)
    # candidate_matched == prefix_len == floor: not below the floor, so it is kept.
    got = _lookup(tier, query, 100)
    assert got is not None and got.matched_tokens == 100
    assert _lookup(tier, query, 101) is None
    assert tier.stats()["last_miss_reason"] == "ssd_prefix_not_better_than_ram"


def test_short_rows_still_decide_the_miss_reason(tier):
    tier._insert_manifest(_meta("a", list(range(64))))
    query = tuple(range(64)) + (5000, 5001)
    assert _lookup(tier, query, 500) is None
    assert tier.stats()["last_miss_reason"] == "ssd_prefix_not_better_than_ram"
    unrelated = tuple(range(9000, 9100))
    assert _lookup(tier, unrelated, 500) is None
    assert tier.stats()["last_miss_reason"] == "ssd_prefix_miss"


def test_memo_parses_each_row_once(tier, monkeypatch):
    toks = list(range(300))
    tier._insert_manifest(_meta("a", toks))
    parses = []
    real = json.loads

    def counting(text, *a, **k):
        parses.append(1)
        return real(text, *a, **k)

    monkeypatch.setattr(cold_tier_module.json, "loads", counting)
    query = tuple(toks[:290]) + (1, 2, 3)
    for _ in range(3):
        assert _lookup(tier, query, 0) is not None
    token_parses = len(parses)
    assert token_parses == 1


def test_memo_follows_delete_and_rewrite(tier):
    a = list(range(200))
    tier._insert_manifest(_meta("a", a))
    query = tuple(a[:195]) + (9, 9, 9, 9, 9)
    assert _lookup(tier, query, 0).matched_tokens == 195
    # Rewrite the same entry_id with different tokens (and token_hash).
    b = [x + 100000 for x in a]
    tier._insert_manifest(_meta("a", b))
    assert _lookup(tier, query, 0) is None
    assert _lookup(tier, tuple(b[:195]) + (1, 2, 3, 4, 5), 0).matched_tokens == 195
    # Delete through the tier's own retire path.
    with tier._connect() as conn:
        row = conn.execute("SELECT * FROM entries WHERE entry_id = 'a'").fetchone()
    tier._delete_entry_row(row)
    assert "a" not in tier._row_tokens_memo
    assert _lookup(tier, tuple(b[:195]) + (1, 2, 3, 4, 5), 0) is None


def test_memo_rewrite_with_same_id_by_other_process(tier):
    """A row rewritten behind our back (other process) is caught by token_hash."""
    a = list(range(200))
    tier._insert_manifest(_meta("a", a))
    query = tuple(a[:195]) + (9, 9, 9, 9, 9)
    assert _lookup(tier, query, 0) is not None
    b = [x + 100000 for x in a]
    with tier._connect() as conn:  # bypasses _insert_manifest
        conn.execute(
            "UPDATE entries SET token_ids_json = ?, token_hash = ? WHERE entry_id = 'a'",
            (json.dumps(b), "other-hash"),
        )
    assert _lookup(tier, query, 0) is None


def test_memo_is_bounded(tier, monkeypatch):
    monkeypatch.setattr(cold_tier_module, "_ROW_TOKENS_MEMO_MAX_TOKENS", 500)
    for i in range(6):
        tier._insert_manifest(_meta(f"e{i}", list(range(i, i + 200))))
    _lookup(tier, tuple(range(300)), 0)
    assert tier._row_tokens_memo_tokens <= 500
    assert tier._row_tokens_memo_tokens == sum(
        len(t) for _, t in tier._row_tokens_memo.values()
    )


def test_full_memo_keeps_residents_so_repeated_scans_still_hit(tier, monkeypatch):
    """A scan over more tokens than the cap must not evict every row before its reuse."""
    monkeypatch.setattr(cold_tier_module, "_ROW_TOKENS_MEMO_MAX_TOKENS", 500)
    for i in range(6):
        tier._insert_manifest(_meta(f"e{i}", list(range(i, i + 200))))
    parses = []
    real = json.loads

    def counting(text, *a, **k):
        parses.append(1)
        return real(text, *a, **k)

    monkeypatch.setattr(cold_tier_module.json, "loads", counting)
    query = tuple(range(300))
    _lookup(tier, query, 0)
    first = len(parses)
    _lookup(tier, query, 0)
    second = len(parses) - first
    assert len(tier._row_tokens_memo) == 2
    assert second == first - 2


def test_memo_holds_compact_arrays(tier):
    tier._insert_manifest(_meta("a", list(range(200))))
    _lookup(tier, tuple(range(195)) + (9, 9, 9, 9, 9), 0)
    (_, tokens), = tier._row_tokens_memo.values()
    assert isinstance(tokens, cold_tier_module.array)
    assert tokens.typecode == "I"


def test_token_array_falls_back_to_a_tuple_for_ids_that_do_not_fit():
    assert cold_tier_module._token_array([1, 2**40]) == (1, 2**40)
    assert cold_tier_module._token_array([-1]) == (-1,)


@pytest.mark.parametrize(
    "tokens, prefix, expected",
    [
        ((1, 2, 3, 4), (1, 2, 3), True),
        ((1, 2, 3, 4), (1, 2, 4), False),
        ((1, 2), (1, 2, 3), False),
        ((1, 2, 3), (), True),
        ((-1, 2), (-1,), True),
    ],
)
def test_starts_with_matches_tuple_slicing_for_arrays_and_tuples(tokens, prefix, expected):
    as_stored = cold_tier_module._token_array(list(prefix))
    assert cold_tier_module._starts_with(tokens, as_stored) is expected
    assert (tokens[: len(prefix)] == prefix) is expected
