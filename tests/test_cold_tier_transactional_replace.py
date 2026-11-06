"""An SSD replacement is a transaction, and the SSD cap holds two copies of a session.

2026-09-29, Pi session d89c3e, 128 GB Mac, 25 GiB of free disk. The tier's
cap was min(100 GiB, free / 4) = 6.25 GiB: room for one ~4 GB entry. Every
new write evicted the only entry before writing its own blobs (the
131,735-token write retired the 128,009 entry at 20:54:24), and the new entry
stayed invisible until its manifest row landed at 20:56:32. At 20:56:30 a
138K request found no finished entry and had nothing to restore.

Both writers (the staged writer thread and the owner-thread spill) now
reserve the new copy's disk, keep the old copy, write, install the new
manifest row, and only then retire older entries for the cap; the cap holds
two copies of the largest session whenever the disk above a free-disk floor
can. These tests drive the real tier on a temp directory with the free disk
faked. The tier-level ones run at MiB scale (every GiB of the incident is a
MiB here, the floor included; the policy is scale-free once its floor is
scaled with it):

* the 09-29 shape (25 GiB free, a 4 GB session) keeps its predecessor;
* the predecessor restores while its replacement is being written;
* ENOSPC at every write boundary leaves the predecessor whole and the
  replacement invisible, releases the reservation and records why;
* a process killed before or after the manifest install leaves a valid
  predecessor or replacement on disk;
* blobs shared with the predecessor are not rewritten and survive its
  retirement;
* a disk too full for the replacement beside its predecessor skips the write
  with a recorded reason, and the stats say the disk is low;
* two writers in flight cannot both spend the same free space.

MLX work stays on the test thread: the staged path encodes in put_entry and
its writer thread only moves bytes, as in production.
"""

from __future__ import annotations

import errno
import shutil
import sqlite3
import threading
from collections import namedtuple
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.cache_bank import cold_tier
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_bank.disk_budget import DISK_FULL, DISK_LOW, DISK_OK
from mtplx.session_bank import SessionBank

# Byte scale of the tier-level tests: one MiB here stands for one GiB then.
UNIT = 1024**2
_Usage = namedtuple("_Usage", "total used free")
IDENTITY = {
    "model_path": "models/example",
    "mtp_enabled": True,
    "template_hash": "template-a",
    "policy_fingerprint": "policy-a",
}
WRITERS = ("spill", "staged")
PRED_TOKENS = tuple(range(1, 41))
REPL_TOKENS = tuple(range(1, 57))
# 2,048 rows of 512 int32 are 4 MiB: the "4 GB" session. The codec slices
# the rows into 256-row blocks, one content-addressed blob each.
PRED_ROWS = 2048
REPL_ROWS = 2304


class FakeRuntime:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []


class KV:
    """A trimmable cache leaf whose rows are a pure function of position,
    so a longer copy of a conversation repeats the shorter copy's rows and
    deduplicates against its blocks. ``salt`` gives a rewritten history:
    the same shape with none of the blocks in common."""

    def __init__(self, rows: int, *, salt: int = 0, width: int = 512) -> None:
        values = mx.arange(rows * width, dtype=mx.int32) + int(salt)
        self.state = values.reshape(1, 1, rows, width)
        self.meta_state = ("kv", str(rows))

    def is_trimmable(self) -> bool:
        return True


def _entry(tokens, rows, *, salt: int = 0, session: str = "pi"):
    bank = SessionBank(max_bytes=256 * UNIT, per_session_max_bytes=256 * UNIT)
    entry = bank.put(
        runtime=FakeRuntime(),
        token_ids=list(tokens),
        cache=[KV(rows, salt=salt)],
        logits=mx.array([[float(len(tokens)), 1.0]], dtype=mx.float32),
        hidden=mx.array([[[float(len(tokens)), 2.0]]], dtype=mx.float32),
        session_id=session,
        template_hash="template-a",
        policy_fingerprint="policy-a",
        snapshot_epoch=len(tokens),
    )
    assert entry is not None and not entry.live_ref_only
    return entry


def _replacement(content: str):
    """The next copy of the conversation: extending the predecessor (its
    blocks deduplicate) or after a rewritten history (none do)."""

    return _entry(REPL_TOKENS, REPL_ROWS, salt=0 if content == "extends" else 7)


def _disk(monkeypatch, free_units: float) -> None:
    """Fake the free disk, and scale the floor with the byte unit."""

    monkeypatch.setattr(cold_tier, "LOW_DISK_FLOOR_BYTES", 10 * UNIT)
    free = int(free_units * UNIT)
    monkeypatch.setattr(
        cold_tier.shutil,
        "disk_usage",
        lambda _path: _Usage(4000 * UNIT, 4000 * UNIT - free, free),
    )


def _tier(base: Path, *, max_units: float = 100) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=base, mode="on", min_prefix_tokens=2, max_bytes=int(max_units * UNIT)
    )


def _write(tier: SessionBankColdTier, entry, writer: str) -> bool:
    if writer == "spill":
        return bool(tier.spill_entry(entry, capabilities=["ar_insert"]))
    assert tier.put_entry(entry, capabilities=["ar_insert"])
    assert tier.flush(timeout_s=30.0)
    return tier.is_published(entry)


def _restores(tier: SessionBankColdTier, tokens) -> tuple[int, ...] | None:
    """The tokens of the entry a lookup for ``tokens`` fully restores (every
    blob read and decoded), or None."""

    record = tier.lookup(list(tokens), **IDENTITY)
    return None if record is None else tuple(record.token_ids)


def _row(tier: SessionBankColdTier, entry) -> sqlite3.Row | None:
    with tier._connect() as conn:
        return conn.execute(
            "SELECT * FROM entries WHERE entry_id = ?", (tier.entry_id_for(entry),)
        ).fetchone()


def _assert_released(tier: SessionBankColdTier) -> None:
    assert tier._reserved_bytes == 0
    assert not tier._inflight_entry_dirs
    assert not tier._inflight_blob_hashes


class MidWrite:
    """Run ``check`` once, in the middle of a replacement's blob writes:
    inline for the owner-thread spill, and from the test thread while the
    writer thread is held for the staged path."""

    def __init__(self, tier: SessionBankColdTier, check) -> None:
        self.tier = tier
        self.check = check
        self.results: list = []
        self._original = tier._write_blob
        self._main = threading.get_ident()
        self._paused = threading.Event()
        self._resume = threading.Event()
        self._fired = False
        tier._write_blob = self._hook

    def _hook(self, digest: str, raw: bytes) -> bool:
        if not self._fired and not self.tier._blob_path(digest).exists():
            self._fired = True
            if threading.get_ident() == self._main:
                self.results.append(self.check())
            else:
                self._paused.set()
                assert self._resume.wait(30.0)
        return self._original(digest, raw)

    def run(self, writer: str, entry) -> bool:
        try:
            if writer == "spill":
                return _write(self.tier, entry, writer)
            assert self.tier.put_entry(entry, capabilities=["ar_insert"])
            assert self._paused.wait(30.0), "the writer never reached a new blob"
            self.results.append(self.check())
            self._resume.set()
            assert self.tier.flush(timeout_s=30.0)
            return self.tier.is_published(entry)
        finally:
            self._resume.set()
            self.tier._write_blob = self._original


@pytest.mark.parametrize("content", ("extends", "rewritten"))
@pytest.mark.parametrize("writer", WRITERS)
def test_a_4_gb_session_at_25_gib_free_is_replaced_beside_its_predecessor(
    tmp_path, monkeypatch, writer, content
):
    """The 09-29 shape. On the old code the spill (the path those 4 GB
    entries took) evicted the predecessor before writing, in both content
    cases; the staged path did so when the history was rewritten."""

    _disk(monkeypatch, free_units=25)
    tier = _tier(tmp_path / "bank")
    try:
        pred = _entry(PRED_TOKENS, PRED_ROWS)
        assert _write(tier, pred, writer)
        repl = _replacement(content)
        mid = MidWrite(tier, lambda: _restores(tier, PRED_TOKENS))
        assert mid.run(writer, repl)
        assert mid.results == [PRED_TOKENS], "the only copy was gone mid-write"
        assert _restores(tier, PRED_TOKENS) == PRED_TOKENS
        assert _restores(tier, REPL_TOKENS) == REPL_TOKENS
        stats = tier.stats()
        assert stats["entries_evicted"] == 0
        assert stats["disk_state"] == DISK_OK and stats["low_disk"] is False
        assert stats["effective_max_bytes"] >= 2 * int(_row(tier, repl)["nbytes"])
        _assert_released(tier)
    finally:
        tier.close()


@pytest.mark.parametrize("content", ("extends", "rewritten"))
@pytest.mark.parametrize("writer", WRITERS)
def test_the_predecessor_restores_while_its_replacement_is_written(
    tmp_path, monkeypatch, writer, content
):
    """A configured cap that holds one copy (6 of 100 units free-disk
    share): the replacement is written beside the predecessor and retires it
    only once installed, if the cap still needs the room."""

    _disk(monkeypatch, free_units=1000)
    tier = _tier(tmp_path / "bank", max_units=6)
    try:
        pred = _entry(PRED_TOKENS, PRED_ROWS)
        assert _write(tier, pred, writer)
        repl = _replacement(content)
        mid = MidWrite(tier, lambda: _restores(tier, PRED_TOKENS))
        assert mid.run(writer, repl)
        assert mid.results == [PRED_TOKENS]
        assert _restores(tier, REPL_TOKENS) == REPL_TOKENS
        if content == "rewritten":
            # Four plus 4.5 units do not fit six: the predecessor went, after.
            assert _row(tier, pred) is None
            assert tier.stats()["entries_evicted"] == 1
        else:
            # Shared blocks: both copies fit, nothing was retired.
            assert _restores(tier, PRED_TOKENS) == PRED_TOKENS
        _assert_released(tier)
    finally:
        tier.close()


def _fail_blob_write(monkeypatch) -> None:
    # Blobs are written through blob_io.write_file (F_NOCACHE on macOS), not
    # Path.write_bytes, so the fault goes where cold_tier calls it.
    original = cold_tier.write_file
    fired = []

    def write_file(path, data, *args, **kwargs):
        path = Path(path)
        if ".bin.tmp-" in path.name and not fired:
            fired.append(path)
            with open(path, "wb") as handle:  # a partial write, then the disk is full
                handle.write(bytes(data)[: len(data) // 2])
            raise OSError(errno.ENOSPC, "No space left on device", str(path))
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(cold_tier, "write_file", write_file)


def _fail_payload_write(monkeypatch) -> None:
    original = Path.write_text

    def write_text(self, data, *args, **kwargs):
        if self.name == "payload.json":
            raise OSError(errno.ENOSPC, "No space left on device", str(self))
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)


def _fail_entry_rename(monkeypatch) -> None:
    original = Path.rename

    def rename(self, target):
        if self.name.startswith(".") and ".tmp-" in self.name and self.is_dir():
            raise OSError(errno.ENOSPC, "No space left on device", str(self))
        return original(self, target)

    monkeypatch.setattr(Path, "rename", rename)


def _fail_manifest_insert(monkeypatch, tier) -> None:
    def insert(_metadata):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(tier, "_insert_manifest", insert)


BOUNDARIES = ("blob", "payload", "rename", "manifest")


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("writer", WRITERS)
def test_enospc_at_every_write_boundary_keeps_the_predecessor(
    tmp_path, monkeypatch, writer, boundary
):
    """A cap that holds one copy, a rewritten history (no shared blocks), and
    the disk refusing one step of the replacement. The old code had already
    evicted the predecessor by then."""

    _disk(monkeypatch, free_units=1000)
    base = tmp_path / "bank"
    tier = _tier(base, max_units=6)
    try:
        pred = _entry(PRED_TOKENS, PRED_ROWS)
        assert _write(tier, pred, writer)
        repl = _replacement("rewritten")
        with monkeypatch.context() as fault:
            if boundary == "blob":
                _fail_blob_write(fault)
            elif boundary == "payload":
                _fail_payload_write(fault)
            elif boundary == "rename":
                _fail_entry_rename(fault)
            else:
                _fail_manifest_insert(fault, tier)
            assert not _write(tier, repl, writer)
        assert not tier.is_published(repl)
        assert _restores(tier, PRED_TOKENS) == PRED_TOKENS
        assert _restores(tier, REPL_TOKENS) == PRED_TOKENS
        stats = tier.stats()
        assert stats["entries_evicted"] == 0
        assert stats["write_failures"] == 1
        skip = stats["last_write_skip"]
        assert skip["reason"] == "write_failed" and "was kept" in skip["message"]
        # Nothing of the failed write is left where a restore or the next
        # write of this entry would look: no temp blob, no temp or final
        # entry directory.
        assert not [p for p in (base / "blobs").rglob("*") if ".tmp-" in p.name]
        assert not [p for p in (base / "entries").rglob("*") if ".tmp-" in p.name]
        repl_dir = base / "entries" / tier.entry_id_for(repl)[:2] / tier.entry_id_for(repl)
        assert not repl_dir.exists()
        _assert_released(tier)
        # The disk has room again: the same write lands and retires the
        # predecessor only after it is installed.
        assert _write(tier, repl, writer)
        assert _restores(tier, REPL_TOKENS) == REPL_TOKENS
        _assert_released(tier)
    finally:
        tier.close()


CRASH_POINTS = ("blobs_written", "directory_renamed", "row_installed", "retiring_older")


def _arm_crash_image(monkeypatch, tier, base: Path, image: Path, point: str):
    """Copy the store at ``point``: what a process killed there leaves on
    disk (copytree reads files, nothing is flushed that the write had not)."""

    taken: list[Path] = []

    def take() -> None:
        if not taken:
            shutil.copytree(base, image)
            taken.append(image)

    if point == "blobs_written":
        original = cold_tier.tempfile.mkdtemp

        def mkdtemp(*args, **kwargs):
            if str(kwargs.get("dir", "")).startswith(str(base)):
                take()
            return original(*args, **kwargs)

        monkeypatch.setattr(cold_tier.tempfile, "mkdtemp", mkdtemp)
    elif point in ("directory_renamed", "row_installed"):
        original_insert = tier._insert_manifest

        def insert(metadata):
            if point == "directory_renamed":
                take()
            original_insert(metadata)
            if point == "row_installed":
                take()

        monkeypatch.setattr(tier, "_insert_manifest", insert)
    else:
        original_rmtree = cold_tier.shutil.rmtree

        def rmtree(path, *args, **kwargs):
            if Path(path).is_relative_to(base / "entries"):
                take()
            return original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(cold_tier.shutil, "rmtree", rmtree)
    return taken


@pytest.mark.parametrize("point", CRASH_POINTS)
@pytest.mark.parametrize("writer", WRITERS)
def test_a_process_killed_around_the_install_leaves_one_valid_copy(
    tmp_path, monkeypatch, writer, point
):
    """One restorable copy survives a kill before the replacement's row, after
    it, and while older entries are being retired. The old code retired the
    predecessor before the replacement's first blob, so a kill anywhere in
    the write left nothing."""

    _disk(monkeypatch, free_units=1000)
    base = tmp_path / "bank"
    image = tmp_path / "crashed"
    tier = _tier(base, max_units=6)
    try:
        pred = _entry(PRED_TOKENS, PRED_ROWS)
        assert _write(tier, pred, writer)
        repl = _replacement("rewritten")
        with monkeypatch.context() as crash:
            taken = _arm_crash_image(crash, tier, base, image, point)
            assert _write(tier, repl, writer)
        assert taken, f"the write never reached {point}"
    finally:
        tier.close()
    restarted = _tier(image, max_units=6)
    try:
        survivor = _restores(restarted, REPL_TOKENS)
        assert survivor in (PRED_TOKENS, REPL_TOKENS), f"nothing restorable at {point}"
        if point in ("blobs_written", "directory_renamed"):
            assert survivor == PRED_TOKENS
            assert _restores(restarted, PRED_TOKENS) == PRED_TOKENS
        else:
            assert survivor == REPL_TOKENS
        if point == "row_installed":
            # Both rows are in the manifest; the older one is still whole.
            assert _restores(restarted, PRED_TOKENS) == PRED_TOKENS
    finally:
        restarted.close()


@pytest.mark.parametrize("writer", WRITERS)
def test_blobs_shared_with_the_predecessor_are_kept_and_not_rewritten(
    tmp_path, monkeypatch, writer
):
    """The replacement extends the predecessor: its first eight blocks are
    the predecessor's blobs. Written beside the predecessor it writes only
    its new block; when the predecessor is later retired for the cap, the
    shared blobs stay and only the predecessor's own blobs go. The old spill
    evicted the predecessor (and its blobs) first and rewrote every block."""

    _disk(monkeypatch, free_units=1000)
    base = tmp_path / "bank"
    tier = _tier(base, max_units=6)
    try:
        pred = _entry(PRED_TOKENS, PRED_ROWS)
        assert _write(tier, pred, writer)
        pred_blobs = cold_tier._entry_blob_hashes_of(base / _row(tier, pred)["entry_dir"])
        repl = _entry(REPL_TOKENS, REPL_ROWS)
        assert _write(tier, repl, writer)
        repl_row = _row(tier, repl)
        repl_blobs = cold_tier._entry_blob_hashes_of(base / repl_row["entry_dir"])
        shared = pred_blobs & repl_blobs
        assert len(shared) >= PRED_ROWS // 256
        assert int(repl_row["physical_nbytes"]) < int(repl_row["logical_nbytes"]) // 4
        # Another conversation's write pushes the store over its cap: the
        # least recently used entry, the predecessor, is retired.
        # 45 tokens: its logits and hidden blobs differ from the 40-token
        # predecessor's, which are then that entry's own.
        other = _entry(tuple(range(900, 945)), 1024, salt=99, session="other")
        assert _write(tier, other, writer)
        assert _row(tier, pred) is None and _row(tier, repl) is not None
        for digest in shared:
            assert tier._blob_path(digest).exists()
        for digest in pred_blobs - repl_blobs:
            assert not tier._blob_path(digest).exists()
        record = tier.lookup(list(REPL_TOKENS), **IDENTITY)
        assert record is not None and tuple(record.token_ids) == REPL_TOKENS
        # Every row of the restored KV, the shared blocks included.
        restored = np.array(record.cache_snapshot.states[0])
        assert np.array_equal(restored, np.array(KV(REPL_ROWS).state))
        _assert_released(tier)
    finally:
        tier.close()


@pytest.mark.parametrize("writer", WRITERS)
def test_a_disk_too_full_for_two_copies_keeps_the_saved_copy_and_says_so(
    tmp_path, monkeypatch, writer
):
    _disk(monkeypatch, free_units=25)
    tier = _tier(tmp_path / "bank")
    try:
        pred = _entry(PRED_TOKENS, PRED_ROWS)
        assert _write(tier, pred, writer)
        # Twelve free: two above the floor, and the rewritten replacement
        # needs 4.5.
        _disk(monkeypatch, free_units=12)
        stats = tier.stats()
        assert stats["disk_state"] == DISK_LOW and stats["low_disk"] is True
        assert "two copies" in stats["disk_message"]
        assert stats["disk_floor_bytes"] == 10 * UNIT
        assert not _write(tier, _replacement("rewritten"), writer)
        stats = tier.stats()
        assert stats["skipped_no_disk_room"] == 1
        skip = stats["last_write_skip"]
        assert skip["reason"] == "no_disk_room"
        assert "The copy saved before it was kept." in skip["message"]
        assert _restores(tier, PRED_TOKENS) == PRED_TOKENS
        _assert_released(tier)
        # Below the floor the tier writes nothing, and says so.
        _disk(monkeypatch, free_units=9)
        stats = tier.stats()
        assert stats["disk_state"] == DISK_FULL
        assert "stopped saving" in stats["disk_message"]
        assert not _write(tier, _replacement("extends"), writer)
        stats = tier.stats()
        assert stats["skipped_low_disk"] == 1
        assert stats["low_disk_writes_disabled"] is True
        assert stats["last_write_skip"]["reason"] == "low_disk"
        assert _restores(tier, PRED_TOKENS) == PRED_TOKENS
    finally:
        tier.close()


def test_two_writes_in_flight_cannot_spend_the_same_free_space(tmp_path, monkeypatch):
    """Sixteen free: six above the floor. A spill of the 4.5-unit
    replacement reserves its size; a staged write of another 4.5 units that
    arrives mid-spill finds 1.5 left and is skipped, instead of both writes
    pushing the disk into the floor."""

    _disk(monkeypatch, free_units=16)
    tier = _tier(tmp_path / "bank")
    try:
        other = _entry(tuple(range(500, 556)), REPL_ROWS, salt=11, session="other")
        seen: list = []

        def during_spill():
            assert tier.put_entry(other, capabilities=["ar_insert"])
            assert tier.flush(timeout_s=30.0)
            seen.append((tier.is_published(other), tier._reserved_bytes))

        mid = MidWrite(tier, during_spill)
        assert mid.run("spill", _replacement("rewritten"))
        assert len(seen) == 1
        published, reserved_during = seen[0]
        assert published is False and reserved_during > 0
        stats = tier.stats()
        assert stats["skipped_no_disk_room"] == 1
        assert stats["last_write_skip"]["reason"] == "no_disk_room"
        _assert_released(tier)
    finally:
        tier.close()
