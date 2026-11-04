"""SSD cold tier: the reconciliation walk is maintenance, never hot-path work.

Receipt that motivated this contract (2026-08-15, M5 Max, 816,220-file bank):
one ``os.walk`` + ``stat`` of the store took 41.7 s. The previous code walked
synchronously on every write's cap gate and re-walked in the background
whenever ``stats()`` (``/health``, and the lookup-miss path) found the 30 s
TTL expired, so an idle daemon burned most of a core forever once the bank
was large. The contract pinned here:

- the cap gate prices orphans from the last snapshot's delta; it never walks;
- ``stats()`` schedules a rescan only for a *changed* store, and only after
  ``max(TTL, DUTY_DIVISOR x last walk duration)``;
- the walk runs without ``_base_lock`` and installs a torn view as stale;
- a lookup miss reads ``last_miss_reason`` without touching ``stats()``.
"""

from __future__ import annotations

import time
from pathlib import Path

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_bank import cold_tier as cold_tier_module
from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank


class FakeRuntime:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []


def _put(bank: SessionBank, tokens: list[int], *, epoch: int) -> None:
    bank.put_snapshot(
        runtime=FakeRuntime(),
        token_ids=tokens,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        template_hash="template-a",
        policy_fingerprint="policy-a",
        snapshot_epoch=epoch,
        nbytes_override=128,
    )


def _count_scans(cold: SessionBankColdTier, monkeypatch) -> list[int]:
    calls = [0]
    original = cold._scan_managed_disk_usage

    def counted():
        calls[0] += 1
        return original()

    monkeypatch.setattr(cold, "_scan_managed_disk_usage", counted)
    return calls


def test_cap_gate_walks_the_store_once_at_cold_start_then_never_per_write(tmp_path, monkeypatch):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=8 * 1024 * 1024,
        min_prefix_tokens=2,
    )
    scans = _count_scans(cold, monkeypatch)
    try:
        bank = SessionBank(cold_tier=cold)
        for i in range(5):
            _put(bank, [1, 2, 3, 4 + i], epoch=i)
        assert cold.flush(timeout_s=5.0) is True

        stats = cold.stats()
        assert stats["writes_completed"] == 5
        # One cold-start snapshot for the first write; the other four writes
        # priced their cap on manifest bytes plus the snapshot's orphan delta.
        assert scans[0] == 1
    finally:
        cold.close()


def test_stats_never_rescans_an_unchanged_store(tmp_path, monkeypatch):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=8 * 1024 * 1024,
        min_prefix_tokens=2,
    )
    scans = _count_scans(cold, monkeypatch)
    try:
        cold._managed_disk_usage(force=True)
        assert scans[0] == 1

        # Pretend the TTL expired long ago: an unchanged store is still exact.
        with cold._disk_usage_lock:
            cold._disk_usage_cache["disk_usage_last_scan_s"] = time.time() - 3600.0
        for _ in range(10):
            view = cold.stats()
            assert view["disk_usage_scan_pending"] is False
            assert view["disk_usage_stale"] is False
        time.sleep(0.05)
        assert scans[0] == 1
    finally:
        cold.close()


def test_rescan_interval_scales_with_the_last_walk_duration(tmp_path, monkeypatch):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=8 * 1024 * 1024,
        min_prefix_tokens=2,
    )
    scans = _count_scans(cold, monkeypatch)
    try:
        cold._managed_disk_usage(force=True)
        assert scans[0] == 1
        # A 40 s walk (measured) means: changed store, but no rescan for
        # DUTY_DIVISOR x 40 s. Bump the generation as a mutation would.
        with cold._disk_usage_lock:
            cold._disk_usage_cache["disk_usage_scan_s"] = 40.0
            cold._disk_usage_cache["disk_usage_last_scan_s"] = time.time() - 60.0
        cold._invalidate_disk_usage_cache()

        view = cold.stats()
        assert view["disk_usage_stale"] is True
        assert view["disk_usage_scan_pending"] is False
        time.sleep(0.05)
        assert scans[0] == 1

        # Once the adaptive interval has elapsed, the dirty store rescans.
        with cold._disk_usage_lock:
            cold._disk_usage_cache["disk_usage_last_scan_s"] = (
                time.time() - cold_tier_module.DISK_USAGE_SCAN_DUTY_DIVISOR * 40.0 - 1.0
            )
        view = cold.stats()
        assert view["disk_usage_scan_pending"] is True
        deadline = time.time() + 5.0
        while scans[0] < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert scans[0] == 2
    finally:
        cold.close()


def test_torn_walk_installs_as_stale_and_never_inflates_orphans(tmp_path, monkeypatch):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=8 * 1024 * 1024,
        min_prefix_tokens=2,
    )
    try:
        bank = SessionBank(cold_tier=cold)
        _put(bank, [1, 2, 3], epoch=1)
        assert cold.flush(timeout_s=5.0) is True

        original = cold._scan_managed_disk_usage

        def scan_across_a_mutation():
            usage = original()
            # A write commits mid-walk: files it added were (say) counted,
            # and the manifest grew after the pre-walk total was captured.
            cold._invalidate_disk_usage_cache()
            with cold._connect() as conn:
                conn.execute(
                    "UPDATE entries SET physical_nbytes = physical_nbytes + 65536, "
                    "nbytes = nbytes + 65536"
                )
            usage["managed_file_bytes"] = int(usage["managed_file_bytes"]) + 65536
            return usage

        monkeypatch.setattr(cold, "_scan_managed_disk_usage", scan_across_a_mutation)
        usage = cold._managed_disk_usage(force=True)
        assert usage["disk_usage_stale"] is True
        # The larger (post-walk) manifest total is paired with the physical
        # bytes, so the mid-walk commit is not reported as orphan bytes.
        assert cold._untracked_bytes_estimate() == 0
        # And stats() will not start an orphan cleanup from a torn view.
        stats = cold.stats()
        assert stats["disk_usage_stale"] is True
        assert stats["orphan_cleanup_running"] is False
    finally:
        cold.close()


def test_lookup_miss_reads_last_miss_reason_without_stats(tmp_path, monkeypatch):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    stats_calls = [0]
    original_stats = cold.stats

    def counted_stats():
        stats_calls[0] += 1
        return original_stats()

    monkeypatch.setattr(cold, "stats", counted_stats)
    try:
        bank = SessionBank(cold_tier=cold)
        restored = bank.restore(
            FakeRuntime(),
            [9, 9, 9, 9],
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )
        assert restored is None
        assert bank.last_miss_reason == "new_session"
        assert cold.last_miss_reason == "ssd_prefix_miss"
        assert stats_calls[0] == 0
    finally:
        cold.close()
