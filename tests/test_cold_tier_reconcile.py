"""SSD cold tier: the store reconciles itself against its manifest (#493).

Three reporters measured the same shape: 394,155 orphan blobs (44.1 GB)
against 17 manifest entries after three weeks; 471,541 blobs (67 GB) on a
60 GB cap after three days of uptime. The reconciliation existed but only
ran when a write found the store near its cap, so a generous cap never
reached it, and the cap gate zeroed the untracked delta after each cleanup
even when those bytes were live blobs booked under evicted entries, so the
on-disk total could sit above the cap for good.

Contract pinned here:

- the tier reconciles once when it opens a store that has content, in the
  background, and reports one summary to its listener;
- a fresh store skips the pass (nothing to reconcile, no thread);
- a write in flight keeps its blobs and entry directory through a pass,
  and a manifest row committed while the pass runs keeps the blobs it names;
- the cap gate prices the whole managed directory (manifest bytes plus
  live-but-unattributed bytes plus reclaimable garbage), reclaims the
  garbage on the writer thread before admitting a write, and never walks
  on the owner thread (spill path) — that thread schedules the pass;
- a delete-mode pass installs the census it took, so no second walk;
- temp files older than an hour are garbage, younger ones are not;
- ``close()`` stops a pass at its next yield and the partial view is stale.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_bank import cold_tier as cold_tier_module
from mtplx.cache_bank.reconcile import (
    TEMP_GRACE_S,
    collect_garbage,
    manifest_blob_hashes,
    reconcile_store,
)
from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank


class FakeRuntime:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []


def _put(bank: SessionBank, tokens: list[int], *, epoch: int, nbytes: int = 128) -> None:
    bank.put_snapshot(
        runtime=FakeRuntime(),
        token_ids=tokens,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        template_hash="template-a",
        policy_fingerprint="policy-a",
        snapshot_epoch=epoch,
        nbytes_override=nbytes,
    )


def _open(tmp_path: Path, **overrides) -> SessionBankColdTier:
    kwargs = {
        "base_dir": tmp_path / "session-bank",
        "mode": "on",
        "max_bytes": 8 * 1024 * 1024,
        "min_prefix_tokens": 2,
    }
    kwargs.update(overrides)
    return SessionBankColdTier(**kwargs)


def _seed_store(tmp_path: Path, *, entries: int = 2) -> Path:
    """A closed store with ``entries`` real manifest rows."""
    cold = _open(tmp_path)
    try:
        bank = SessionBank(cold_tier=cold)
        for i in range(entries):
            _put(bank, [1, 2, 3, 4 + i], epoch=i)
        assert cold.flush(timeout_s=5.0) is True
    finally:
        cold.close()
    return cold.base_dir


def _plant_orphans(base_dir: Path, *, blobs: int = 3, blob_bytes: int = 4096) -> list[Path]:
    planted: list[Path] = []
    for i in range(blobs):
        digest = f"{i:02d}" + "f" * 62
        path = base_dir / "blobs" / digest[:2] / f"{digest}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"o" * blob_bytes)
        planted.append(path)
    orphan_entry = base_dir / "entries" / "zz" / ("z" * 32)
    orphan_entry.mkdir(parents=True, exist_ok=True)
    (orphan_entry / "payload.json").write_text("{}", encoding="utf-8")
    planted.append(orphan_entry)
    evicted = base_dir / "evicted_entries" / "aa" / "old"
    evicted.mkdir(parents=True, exist_ok=True)
    (evicted / "payload.json").write_text("x" * 512, encoding="utf-8")
    planted.append(evicted)
    return planted


def _wait_for_pass(cold: SessionBankColdTier, *, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not cold._orphan_cleanup_is_running():
            with cold._stats_lock:
                if "startup_reconcile" in cold._stats:
                    return
        time.sleep(0.01)
    raise AssertionError("startup reconciliation did not finish")


def _rows(cold: SessionBankColdTier) -> list[sqlite3.Row]:
    with cold._connect() as conn:
        return list(conn.execute("SELECT * FROM entries ORDER BY created_at_s").fetchall())


# --- opening a store ---------------------------------------------------------


def test_opening_a_store_with_orphans_reclaims_them_in_the_background(tmp_path):
    base_dir = _seed_store(tmp_path, entries=2)
    planted = _plant_orphans(base_dir, blobs=3)
    live_hashes_before = manifest_blob_hashes(base_dir)
    summaries: list[dict] = []

    cold = _open(tmp_path, reconcile_listener=summaries.append)
    try:
        _wait_for_pass(cold)
        for path in planted:
            assert not path.exists(), path
        assert not (base_dir / "evicted_entries").exists()
        # Live entries and their blobs are untouched.
        assert len(_rows(cold)) == 2
        assert manifest_blob_hashes(base_dir) == live_hashes_before
        # One summary, with the numbers a user needs.
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary["files_deleted"] == 5  # 3 blobs + 2 payload.json
        assert summary["disk_bytes_deleted"] >= 3 * 4096 + 512
        assert summary["entries"] == 2
        assert summary["max_bytes"] == cold.max_bytes
        assert summary["dir"] == str(base_dir)
        stats = cold.stats()
        assert stats["orphan_cleanup_runs"] == 1
        assert stats["orphan_cleanup_files_deleted"] == 5
        assert stats["startup_reconcile"]["files_deleted"] == 5
        # The pass installed the census: no scan pending, nothing untracked.
        assert stats["disk_usage_scan_pending"] is False
        assert stats["orphan_file_bytes"] == 0
        assert stats["untracked_file_bytes"] == 0
    finally:
        cold.close()


def test_opening_a_clean_store_reports_nothing_to_reclaim_without_deleting(tmp_path):
    base_dir = _seed_store(tmp_path, entries=1)
    files_before = sorted(p for p in base_dir.rglob("*") if p.is_file())
    summaries: list[dict] = []

    cold = _open(tmp_path, reconcile_listener=summaries.append)
    try:
        _wait_for_pass(cold)
        assert summaries == [summaries[0]]
        assert summaries[0]["files_deleted"] == 0
        assert summaries[0]["entries"] == 1
        assert sorted(p for p in base_dir.rglob("*") if p.is_file()) == files_before
    finally:
        cold.close()


def test_opening_a_fresh_store_skips_the_pass(tmp_path, monkeypatch):
    walks = [0]
    original = cold_tier_module.reconcile_store

    def counted(*args, **kwargs):
        walks[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cold_tier_module, "reconcile_store", counted)
    summaries: list[dict] = []
    cold = _open(tmp_path, reconcile_listener=summaries.append)
    try:
        time.sleep(0.05)
        assert walks[0] == 0
        assert cold._orphan_cleanup_is_running() is False
        assert summaries == []
    finally:
        cold.close()


def test_pass_does_not_run_when_the_tier_is_off_or_write_only_off(tmp_path):
    base_dir = _seed_store(tmp_path, entries=1)
    planted = _plant_orphans(base_dir, blobs=1)
    cold = _open(tmp_path, mode="off")
    try:
        time.sleep(0.05)
        for path in planted:
            assert path.exists(), "an off tier must not touch the store"
    finally:
        cold.close()


# --- writes racing the pass --------------------------------------------------


def test_blobs_claimed_by_a_write_in_flight_survive_a_pass(tmp_path):
    base_dir = _seed_store(tmp_path, entries=1)
    cold = _open(tmp_path)
    try:
        _wait_for_pass(cold)
        digest = "ab" + "c" * 62
        blob = base_dir / "blobs" / "ab" / f"{digest}.bin"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"phase-two-blob")
        entry_rel = "entries/ab/" + "a" * 32
        entry_dir = base_dir / entry_rel
        entry_dir.mkdir(parents=True)
        (entry_dir / "payload.json").write_text("{}", encoding="utf-8")

        # Phase 1 of a write claimed them; phase 3 has not landed the row.
        cold._claim_inflight(entry_dirs=(entry_rel,), digests={digest})
        result = cold._cleanup_untracked_cache_once()
        assert result["files_deleted"] == 0
        assert blob.exists() and entry_dir.exists()

        # Released without a manifest row (the write was refused): garbage.
        cold._release_inflight(entry_dirs=(entry_rel,), digests={digest})
        result = cold._cleanup_untracked_cache_once()
        assert result["files_deleted"] == 2
        assert not blob.exists() and not entry_dir.exists()
    finally:
        cold.close()


def test_two_writes_sharing_a_digest_release_independently(tmp_path):
    cold = _open(tmp_path)
    try:
        cold._claim_inflight(digests={"d"})
        cold._claim_inflight(digests={"d"})
        cold._release_inflight(digests={"d"})
        assert "d" in cold._inflight_snapshot()[1]
        cold._release_inflight(digests={"d"})
        assert cold._inflight_snapshot() == (set(), set())
    finally:
        cold.close()


def test_a_row_committed_during_the_pass_keeps_the_blobs_it_names(tmp_path):
    """The classic race: a blob is unreferenced when the walk lists it, then
    a new entry dedupes against it before the batch is deleted."""
    base_dir = _seed_store(tmp_path, entries=1)
    digest = "ee" + "1" * 62
    blob = base_dir / "blobs" / "ee" / f"{digest}.bin"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"shared-later")
    generation = [0]

    class _CommitOnEnter:
        """A lock whose acquisition lands a manifest row naming the blob."""

        entered = 0

        def __enter__(self):
            type(self).entered += 1
            if type(self).entered == 1:
                entry_rel = "entries/ee/" + "e" * 32
                entry_dir = base_dir / entry_rel
                entry_dir.mkdir(parents=True)
                (entry_dir / "payload.json").write_text(
                    json.dumps({"tensor_blobs": {"t": {"sha256": digest, "nbytes": 12}}}),
                    encoding="utf-8",
                )
                with sqlite3.connect(str(base_dir / "manifest.sqlite")) as conn:
                    columns = [
                        row[1] for row in conn.execute("PRAGMA table_info(entries)")
                    ]
                    template = conn.execute("SELECT * FROM entries LIMIT 1").fetchone()
                    values = dict(zip(columns, template))
                    values.update(
                        entry_id="e" * 32,
                        token_hash="t" * 64,
                        entry_dir=entry_rel,
                        token_ids_json="[9,9,9]",
                    )
                    conn.execute(
                        f"INSERT INTO entries ({', '.join(columns)}) VALUES "
                        f"({', '.join('?' for _ in columns)})",
                        [values[c] for c in columns],
                    )
                generation[0] += 1
            return self

        def __exit__(self, *exc):
            return False

    report = reconcile_store(
        base_dir,
        dry_run=False,
        lock=_CommitOnEnter(),
        manifest_generation=lambda: generation[0],
    )
    assert blob.exists(), "a blob a fresh row names must survive"
    assert report.protected_skipped == 1
    assert report.files_deleted == 0


# --- the cap prices the whole directory -----------------------------------


def test_cap_gate_prices_live_bytes_booked_under_an_evicted_entry(tmp_path):
    """Two entries share one blob; the second wrote 16 bytes of its own and
    deduped the rest. Evicting the first (which paid for the blob) drops the
    manifest SUM to 16 but not the file. The gate must still price the file,
    and a cleanup (which deletes nothing) must not zero it, or the store
    sits above the cap for good (67 GB on a 60 GB cap, #493)."""
    cold = _open(tmp_path, max_bytes=8 * 1024 * 1024)
    try:
        base_dir = cold.base_dir
        digest = "aa" + "b" * 62
        blob = base_dir / "blobs" / "aa" / f"{digest}.bin"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"s" * 100_000)
        for name, physical in (("first", 100_000), ("second", 16)):
            entry_rel = f"entries/{name[:2]}/{name * 4}"
            entry_dir = base_dir / entry_rel
            entry_dir.mkdir(parents=True)
            (entry_dir / "payload.json").write_text(
                json.dumps({"tensor_blobs": {"t": {"sha256": digest, "nbytes": 100_000}}}),
                encoding="utf-8",
            )
            cold._insert_manifest(
                {
                    "entry_id": name * 4,
                    "format_version": cold_tier_module.COLD_TIER_FORMAT_VERSION,
                    "token_hash": name * 16,
                    "prefix_len": 3,
                    "token_ids": [1, 2, 3],
                    "model_path": "m",
                    "mtp_enabled": True,
                    "hidden_variant": None,
                    "template_hash": None,
                    "mtp_history_policy": None,
                    "draft_head_identity": None,
                    "policy_fingerprint": None,
                    "session_id": None,
                    "snapshot_epoch": 0,
                    "mtp_snapshot_epoch": None,
                    "capabilities": [],
                    "block_size": 256,
                    "block_hashes": [],
                    "nbytes": 100_000,
                    "logical_nbytes": 100_000,
                    "physical_nbytes": physical,
                    "deduped_nbytes": 100_000 - physical,
                    "entry_dir": entry_rel,
                    "created_at_s": time.time() - (10 if name == "first" else 0),
                    "last_access_s": time.time() - (10 if name == "first" else 0),
                }
            )
        cold._invalidate_disk_usage_cache()
        cold._managed_disk_usage(force=True)
        assert cold._current_bytes() == 100_016
        assert cold._current_bytes_for_cap() >= 100_000

        # Evict "first": the blob stays (second names it), the SUM drops to 16.
        cold._delete_entry_row(_rows(cold)[0])
        assert blob.exists()
        assert cold._current_bytes() == 16
        cold._managed_disk_usage(force=True)
        stats = cold.stats()
        assert stats["orphan_file_bytes"] == 0, "a referenced blob is not garbage"
        assert stats["untracked_file_bytes"] >= 100_000, "but it is on disk"
        priced = cold._current_bytes_for_cap()
        assert priced >= 100_000

        # A cleanup deletes nothing and must leave the pricing alone.
        result = cold._cleanup_untracked_cache_once()
        assert result["files_deleted"] == 0
        assert blob.exists()
        assert cold._current_bytes_for_cap() == priced
    finally:
        cold.close()


def test_writer_reclaims_garbage_before_the_gate_would_evict_live_entries(tmp_path):
    cold = _open(tmp_path, max_bytes=64 * 1024)
    try:
        bank = SessionBank(cold_tier=cold)
        _put(bank, [1, 2, 3], epoch=1)
        assert cold.flush(timeout_s=5.0) is True
        live = _rows(cold)
        assert len(live) == 1
        # Garbage fills the cap; the next write must reclaim it, not evict
        # the live entry to make room beside it.
        orphan = cold.base_dir / "blobs" / "cc" / f'{"c" * 64}.bin'
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"z" * (64 * 1024))
        cold._invalidate_disk_usage_cache()
        cold._managed_disk_usage(force=True)
        assert cold._reclaimable_bytes_estimate() == 64 * 1024

        _put(bank, [1, 2, 3, 4], epoch=2)
        assert cold.flush(timeout_s=5.0) is True
        stats = cold.stats()
        assert stats["writes_completed"] == 2
        assert stats["entries_evicted"] == 0
        assert not orphan.exists()
        assert stats["orphan_cleanup_runs"] == 1
        assert {row["entry_id"] for row in _rows(cold)} >= {live[0]["entry_id"]}
    finally:
        cold.close()


def test_owner_thread_path_never_walks_it_schedules_the_pass(tmp_path, monkeypatch):
    cold = _open(tmp_path, max_bytes=64 * 1024)
    try:
        orphan = cold.base_dir / "blobs" / "cc" / f'{"c" * 64}.bin'
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"z" * (64 * 1024))
        cold._invalidate_disk_usage_cache()
        cold._managed_disk_usage(force=True)

        started: list[threading.Thread] = []
        original_start = cold._start_orphan_cleanup

        def record_start(**kwargs):
            started.append(threading.current_thread())
            original_start(**kwargs)

        monkeypatch.setattr(cold, "_start_orphan_cleanup", record_start)
        inline_walks = [0]
        original_run = cold._run_orphan_cleanup

        def counted_run():
            inline_walks[0] += 1
            return original_run()

        monkeypatch.setattr(cold, "_run_orphan_cleanup", counted_run)

        cold._reclaim_orphans_if_over_cap(1024, inline=False)
        assert inline_walks[0] == 0, "the owner thread must not walk"
        assert started, "the pass is scheduled in the background"
        deadline = time.time() + 5.0
        while cold._orphan_cleanup_is_running() and time.time() < deadline:
            time.sleep(0.01)
        assert not orphan.exists()
    finally:
        cold.close()


def test_bytes_a_running_pass_will_free_are_not_charged_to_a_write(tmp_path):
    cold = _open(tmp_path, max_bytes=64 * 1024)
    try:
        orphan = cold.base_dir / "blobs" / "cc" / f'{"c" * 64}.bin'
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"z" * (64 * 1024))
        cold._invalidate_disk_usage_cache()
        cold._managed_disk_usage(force=True)
        assert cold._current_bytes_for_cap() == 64 * 1024
        assert cold._claim_orphan_cleanup() is True
        try:
            assert cold._current_bytes_for_cap() == 0
        finally:
            cold._release_orphan_cleanup()
        assert cold._current_bytes_for_cap() == 64 * 1024
    finally:
        cold.close()


def test_delete_pass_installs_its_census_without_a_second_walk(tmp_path, monkeypatch):
    base_dir = _seed_store(tmp_path, entries=1)
    _plant_orphans(base_dir, blobs=2)
    cold = _open(tmp_path)
    try:
        _wait_for_pass(cold)
        scans = [0]
        original = cold._scan_managed_disk_usage

        def counted():
            scans[0] += 1
            return original()

        monkeypatch.setattr(cold, "_scan_managed_disk_usage", counted)
        for _ in range(5):
            view = cold.stats()
            assert view["disk_usage_scan_pending"] is False
            assert view["disk_usage_stale"] is False
        time.sleep(0.05)
        assert scans[0] == 0
    finally:
        cold.close()


# --- temp files, interruption ------------------------------------------------


def test_stale_temp_files_are_garbage_and_fresh_ones_are_not(tmp_path):
    base_dir = _seed_store(tmp_path, entries=1)
    fresh = base_dir / "blobs" / "ab" / ".abc.bin.tmp-1"
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_bytes(b"writing")
    stale = base_dir / "blobs" / "ab" / ".abd.bin.tmp-2"
    stale.write_bytes(b"crashed")
    old = time.time() - TEMP_GRACE_S - 60
    import os

    os.utime(stale, (old, old))

    report = reconcile_store(base_dir, dry_run=True)
    assert report.stale_temp_files == 1
    assert report.orphan_file_bytes == len(b"crashed")
    report = reconcile_store(base_dir, dry_run=False)
    assert report.files_deleted == 1
    assert fresh.exists() and not stale.exists()


def test_writer_survives_a_pruned_prefix_directory(tmp_path):
    cold = _open(tmp_path)
    try:
        prefix = cold.base_dir / "blobs" / "ab"
        prefix.mkdir(parents=True, exist_ok=True)
        original_write_file = cold_tier_module.write_file
        pruned = [False]

        def prune_then_write(path, data, *args, **kwargs):
            path = Path(path)
            if not pruned[0] and path.name.startswith(".") and path.parent == prefix:
                pruned[0] = True
                prefix.rmdir()
                raise FileNotFoundError(str(path))
            return original_write_file(path, data, *args, **kwargs)

        from unittest import mock

        with mock.patch.object(cold_tier_module, "write_file", prune_then_write):
            assert cold._write_blob("ab" + "0" * 62, b"payload") is True
        assert (prefix / ("ab" + "0" * 62 + ".bin")).read_bytes() == b"payload"
        assert pruned[0] is True
    finally:
        cold.close()


def test_close_stops_a_pass_and_the_partial_view_is_stale(tmp_path):
    base_dir = _seed_store(tmp_path, entries=1)
    stopped = threading.Event()
    report = reconcile_store(
        base_dir,
        dry_run=True,
        should_stop=lambda: True,
        on_yield=stopped.set,
        yield_every=1,
    )
    assert report.interrupted is True
    assert stopped.is_set()

    cold = _open(tmp_path)
    try:
        _wait_for_pass(cold)
        cold._stop.set()
        usage = cold._walk_and_install(cold._scan_managed_disk_usage)
        assert usage["disk_usage_stale"] is True
    finally:
        cold.close()


# --- the CLI view of the same walk -------------------------------------------


def test_collect_garbage_reports_entries_and_totals(tmp_path):
    base_dir = _seed_store(tmp_path, entries=2)
    _plant_orphans(base_dir, blobs=1)
    report = collect_garbage(base_dir, dry_run=True)
    assert report["entries"] == 2
    assert report["manifest_found"] is True
    assert report["orphan_blob_files"] == 1
    assert report["orphan_entry_dirs"] == ["entries/zz/" + "z" * 32]
    assert report["evicted_entries_bytes"] == 512
    assert report["orphan_file_bytes"] == 4096 + 2 + 512
    assert report["disk_bytes"] >= report["database_disk_bytes"] > 0
    assert report["deleted"] is False


@pytest.mark.parametrize("mode", ["on", "write-only"])
def test_enabled_modes_reconcile_at_open(tmp_path, mode):
    base_dir = _seed_store(tmp_path, entries=1)
    planted = _plant_orphans(base_dir, blobs=1)
    cold = _open(tmp_path, mode=mode)
    try:
        _wait_for_pass(cold)
        for path in planted:
            assert not path.exists()
    finally:
        cold.close()
