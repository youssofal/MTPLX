"""stats() reports what the SSD store holds, for the dashboard cache overview.

The figures come from the same cached manifest aggregate as the byte totals:
how many conversations the snapshots belong to, the oldest and newest
snapshot, and how many snapshots were never restored.
"""

from mtplx.cache_bank import SessionBankColdTier


def _tier(tmp_path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )


def _insert(tier, entry_id, *, session_id, created_at_s, hits, nbytes=1024):
    with tier._connect() as conn:
        conn.execute(
            "INSERT INTO entries (entry_id, token_hash, prefix_len, token_ids_json, "
            "model_path, mtp_enabled, session_id, snapshot_epoch, capabilities_json, "
            "block_size, block_hashes_json, entry_dir, nbytes, physical_nbytes, "
            "created_at_s, last_access_s, hits, format_version) "
            "VALUES (?, ?, 512, '[]', 'model', 1, ?, 0, '[]', 256, '[]', ?, ?, ?, ?, ?, ?, 3)",
            (
                entry_id,
                entry_id,
                session_id,
                f"entries/{entry_id}",
                nbytes,
                nbytes,
                created_at_s,
                created_at_s,
                hits,
            ),
        )
    tier._invalidate_disk_usage_cache()


def test_empty_manifest_reports_no_sessions_and_no_ages(tmp_path):
    tier = _tier(tmp_path)
    try:
        stats = tier.stats()
        assert stats["entries"] == 0
        assert stats["sessions"] == 0
        assert stats["entries_without_session"] == 0
        assert stats["oldest_created_at_s"] is None
        assert stats["newest_created_at_s"] is None
        assert stats["never_restored_entries"] == 0
    finally:
        tier.close()


def test_mixed_ages_and_hits_are_aggregated(tmp_path):
    tier = _tier(tmp_path)
    try:
        _insert(tier, "a1", session_id="s-a", created_at_s=1000.0, hits=0)
        _insert(tier, "a2", session_id="s-a", created_at_s=3000.0, hits=2)
        _insert(tier, "b1", session_id="s-b", created_at_s=2000.0, hits=0)
        _insert(tier, "c1", session_id="s-c", created_at_s=1500.0, hits=1)
        stats = tier.stats()
        assert stats["entries"] == 4
        assert stats["sessions"] == 3
        assert stats["entries_without_session"] == 0
        assert stats["oldest_created_at_s"] == 1000.0
        assert stats["newest_created_at_s"] == 3000.0
        assert stats["never_restored_entries"] == 2
    finally:
        tier.close()


def test_snapshots_without_session_are_counted_apart(tmp_path):
    tier = _tier(tmp_path)
    try:
        _insert(tier, "n1", session_id=None, created_at_s=1000.0, hits=0)
        _insert(tier, "n2", session_id=None, created_at_s=1100.0, hits=0)
        _insert(tier, "s1", session_id="s-a", created_at_s=1200.0, hits=1)
        stats = tier.stats()
        assert stats["entries"] == 3
        # NULL session ids are not one shared conversation.
        assert stats["sessions"] == 1
        assert stats["entries_without_session"] == 2
        assert stats["never_restored_entries"] == 2
    finally:
        tier.close()


def test_overview_fields_refresh_after_a_store_mutation(tmp_path):
    tier = _tier(tmp_path)
    try:
        _insert(tier, "a1", session_id="s-a", created_at_s=1000.0, hits=0)
        assert tier.stats()["sessions"] == 1
        _insert(tier, "b1", session_id="s-b", created_at_s=2000.0, hits=0)
        stats = tier.stats()
        assert stats["sessions"] == 2
        assert stats["newest_created_at_s"] == 2000.0
        assert stats["never_restored_entries"] == 2
    finally:
        tier.close()
