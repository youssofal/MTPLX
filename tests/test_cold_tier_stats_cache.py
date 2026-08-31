"""stats() must not open the manifest on every poll (issue #280).

/health and the dashboard call stats() continuously. The manifest aggregate
is exact while the store is unchanged, so repeated polls must reuse the
cached row and only a store mutation (or the staleness TTL) may trigger a
fresh sqlite connection.
"""

from mtplx.cache_bank import SessionBankColdTier


def _tier(tmp_path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )


def _count_connects(tier, monkeypatch) -> list:
    calls = []
    original = tier._connect

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tier, "_connect", spy)
    return calls


def test_repeated_stats_polls_reuse_cached_aggregate(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    try:
        tier._ensure_disk_usage_snapshot()
        calls = _count_connects(tier, monkeypatch)
        first = tier.stats()
        connects_after_first = len(calls)
        assert connects_after_first >= 1
        for _ in range(10):
            polled = tier.stats()
        assert polled["entries"] == first["entries"]
        # Ten further polls within the TTL and with no store mutation must
        # not open the manifest again.
        assert len(calls) == connects_after_first
    finally:
        tier.close()


def test_store_mutation_invalidates_stats_snapshot(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    try:
        tier._ensure_disk_usage_snapshot()
        calls = _count_connects(tier, monkeypatch)
        tier.stats()
        baseline = len(calls)
        # Every store mutation funnels through _invalidate_disk_usage_cache.
        tier._invalidate_disk_usage_cache()
        tier.stats()
        assert len(calls) > baseline
    finally:
        tier.close()
