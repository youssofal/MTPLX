from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import mlx.core as mx
import pytest

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_bank.codec import (
    decode_payload,
    decode_payload_prefix,
    encode_payload,
    payload_supports_prefix_decode,
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


def _cold_rows(cold: SessionBankColdTier) -> list[sqlite3.Row]:
    with cold._connect() as conn:
        return list(conn.execute("SELECT * FROM entries ORDER BY created_at_s ASC").fetchall())


def test_session_bank_cold_tier_flush_waits_for_in_flight_write(tmp_path, monkeypatch):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    release_write = threading.Event()
    write_started = threading.Event()
    original_write = cold._write_pending

    def blocked_write(pending):
        write_started.set()
        if not release_write.wait(timeout=5.0):
            raise TimeoutError("test did not release the cold-tier writer")
        return original_write(pending)

    monkeypatch.setattr(cold, "_write_pending", blocked_write)
    try:
        bank = SessionBank(cold_tier=cold)
        bank.put_snapshot(
            runtime=FakeRuntime(),
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )

        assert write_started.wait(timeout=5.0)
        assert cold.flush(timeout_s=0.05) is False
        release_write.set()
        assert cold.flush(timeout_s=5.0) is True
        assert cold.stats()["writes_completed"] == 1
    finally:
        release_write.set()
        cold.close()


def test_cache_bank_codec_round_trips_nested_snapshot():
    snapshot = CacheSnapshot(
        states=((mx.array([1, 2, 3], dtype=mx.int32), None),),
        meta_states=({"offset": 3, "mode": "test"},),
    )
    logits = mx.array([[1.5, 2.5]], dtype=mx.float16)
    hidden = mx.array([[[3.0, 4.0]]], dtype=mx.bfloat16)

    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=logits,
        hidden=hidden,
        mtp_history_snapshot=None,
    )
    decoded = decode_payload(encoded.spec, encoded.tensors.__getitem__)

    assert decoded.cache_snapshot.meta_states == snapshot.meta_states
    assert decoded.cache_snapshot.states[0][1] is None
    assert decoded.cache_snapshot.states[0][0].tolist() == [1, 2, 3]
    assert decoded.logits.tolist() == logits.tolist()
    assert decoded.hidden.dtype == mx.bfloat16
    assert decoded.hidden.shape == hidden.shape


def test_cache_bank_codec_chunks_large_sequence_tensors():
    snapshot = CacheSnapshot(
        states=((mx.zeros((1, 1, 512, 2), dtype=mx.float16),),),
        meta_states=(),
    )

    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        mtp_history_snapshot=None,
        block_size=256,
    )
    state_spec = encoded.spec["cache_snapshot"]["states"]["items"][0]["items"][0]

    assert state_spec["kind"] == "tensor_blocks"
    assert len(state_spec["blocks"]) == 2
    decoded = decode_payload(encoded.spec, encoded.tensors.__getitem__)
    assert decoded.cache_snapshot.states[0][0].shape == (1, 1, 512, 2)


def test_cache_bank_codec_prefix_decode_reads_only_needed_kv_blocks():
    """A boundary restore must not hydrate the long snapshot tail.

    This is the regression behind a short foreground request waiting on a
    multi-GB SSD entry: only KV through the chosen boundary and that boundary's
    recurrent state may be read.
    """

    snapshot = CacheSnapshot(
        states=(
            mx.zeros((1, 1, 1024, 2), dtype=mx.float16),
            mx.full((1, 4), 9.0, dtype=mx.float16),
        ),
        meta_states=(None, None),
    )
    boundary = CacheSnapshot(
        states=(None, mx.full((1, 4), 7.0, dtype=mx.float16)),
        meta_states=(None, None),
    )
    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        mtp_history_snapshot=None,
        gdn_boundaries=[(256, boundary, None)],
        has_recurrent=True,
        block_size=256,
    )
    reads: list[str] = []

    def read_tensor(name: str) -> bytes:
        reads.append(name)
        return encoded.tensors[name]

    decoded = decode_payload_prefix(
        encoded.spec,
        read_tensor,
        cache_prefix_len=256,
        boundary_prefix_len=256,
    )
    state_spec = encoded.spec["cache_snapshot"]["states"]["items"][0]
    later_blocks = {block["name"] for block in state_spec["blocks"][1:]}

    assert decoded.cache_snapshot_prefix_len == 256
    assert decoded.cache_snapshot.states[0].shape == (1, 1, 256, 2)
    assert decoded.cache_snapshot.states[1] is None
    assert decoded.gdn_boundaries[0][0] == 256
    assert decoded.gdn_boundaries[0][1].states[1].tolist() == [[7.0] * 4]
    assert not later_blocks.intersection(reads)


def test_cache_bank_codec_prefix_decode_reads_only_needed_mtp_history():
    """Committed MTP history is shifted by one target token."""

    snapshot = CacheSnapshot(
        states=(mx.zeros((1, 1, 1024, 2), dtype=mx.float16),),
        meta_states=(None,),
    )
    mtp_snapshot = CacheSnapshot(
        states=(mx.full((1, 1, 1024, 2), 3.0, dtype=mx.float16),),
        meta_states=(None,),
    )
    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        mtp_history_snapshot=mtp_snapshot,
        block_size=256,
    )

    decoded = decode_payload_prefix(
        encoded.spec,
        encoded.tensors.__getitem__,
        cache_prefix_len=256,
        mtp_history_prefix_len=255,
    )

    assert decoded.cache_snapshot.states[0].shape == (1, 1, 256, 2)
    assert decoded.mtp_history_snapshot_prefix_len == 255
    assert decoded.mtp_history_snapshot is not None
    assert decoded.mtp_history_snapshot.states[0].shape == (1, 1, 255, 2)


def test_cache_bank_codec_rejects_prefix_decode_for_coupled_cache_metadata():
    """Ring/compressor cache metadata cannot be reconstructed by KV slicing."""

    snapshot = CacheSnapshot(
        states=(mx.zeros((1, 1, 1024, 2), dtype=mx.float16),),
        meta_states=(("mtplx-deepseek-v4-cache-v1", "1024", "0", "8", "2"),),
    )
    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        mtp_history_snapshot=None,
        block_size=256,
    )

    assert payload_supports_prefix_decode(encoded.spec) is False


@pytest.mark.parametrize("boundary_true", ["1", "0"])
def test_cold_tier_offers_a_tiny_recurrent_prefix_only_with_the_legacy_switch(
    tmp_path, monkeypatch, boundary_true
):
    """A hybrid entry with no recurrent boundary cannot serve a sub-prefix,
    not even a one-token tokenizer-drift seam: its recurrent state has read
    the token the prompt does not have (boundary-true restore, 2026-09-08).
    The SSD tier refuses it before reading its tensors (2026-10-01: a 131K
    entry decoded in full for a turn that could never use it). The legacy
    off-switch keeps the old offer."""

    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", boundary_true)
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )

    class RecurrentStub:
        def __init__(self):
            self.state = [mx.ones((2, 2)), None]
            self.meta_state = ("owned_recurrent_state", "persistent_eval")

        def is_trimmable(self):
            return False

    try:
        runtime = FakeRuntime()
        bank = SessionBank(cold_tier=cold)
        entry = bank.put(
            runtime=runtime,
            token_ids=list(range(41)),
            cache=[RecurrentStub()],
            logits=mx.zeros((1, 4)),
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=41,
            gdn_boundaries=[],
        )
        assert entry is not None
        assert entry.has_recurrent is True
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()

        candidates = bank.near_prefix_candidates(
            list(range(40)) + [99_001],
            max_token_gap=8,
            min_matched_tokens=4,
            block_size=8,
            block_min_matched_tokens=8,
            model_path=str(runtime.model_path),
            mtp_enabled=runtime.mtp_enabled,
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )

        if boundary_true == "1":
            assert candidates == []
            assert cold.stats()["last_miss_reason"] == "ssd_prefix_no_recurrent_boundary"
        else:
            assert candidates
            assert candidates[0][1] == 40
            assert getattr(candidates[0][0], "cache_source") == "ssd"
    finally:
        cold.close()


def test_session_bank_cold_tier_write_only_writes_but_does_not_restore(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="write-only",
        min_prefix_tokens=2,
    )
    try:
        bank = SessionBank(
            max_entries=1,
            max_bytes=1024,
            per_session_max_bytes=1024,
            cold_tier=cold,
        )
        runtime = FakeRuntime()

        entry = bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )

        assert entry is not None
        assert cold.flush(timeout_s=5.0) is True
        assert cold.stats()["entries"] == 1
        bank.clear()
        restored = bank.restore(
            runtime,
            [1, 2, 3, 4],
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )
        assert restored is None
        assert bank.last_miss_reason == "ssd_cache_write_only"
    finally:
        cold.close()


def test_session_bank_restores_exact_prefix_from_ssd_after_ram_clear(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=1,
            max_bytes=1024,
            per_session_max_bytes=1024,
            cold_tier=cold,
        )
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()

        restored = bank.restore(
            runtime,
            [1, 2, 3, 4, 5],
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )

        assert restored is not None
        assert restored.cache_source == "ssd"
        assert restored.restore_mode == "ssd_clone"
        assert restored.ssd_cache_hit is True
        assert restored.ssd_cached_tokens == 3
        assert restored.entry.prefix_len == 3
        assert len(bank) == 1
    finally:
        cold.close()


def test_session_bank_ssd_restore_relinks_same_prefix_to_new_session(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=1,
            max_bytes=1024,
            per_session_max_bytes=1024,
            cold_tier=cold,
        )
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            session_id="opencode-session-a",
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()

        restored = bank.restore(
            runtime,
            [1, 2, 3, 4],
            session_id="opencode-session-b",
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )

        assert restored is not None
        assert restored.cache_source == "ssd"
        assert restored.entry.session_id == "opencode-session-b"
        assert bank.clear(session_id="opencode-session-b") == 1
    finally:
        cold.close()


def test_session_bank_ssd_different_session_different_prompt_misses(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(cold_tier=cold)
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            session_id="opencode-session-a",
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()

        restored = bank.restore(
            runtime,
            [9, 2, 3, 4],
            session_id="opencode-session-b",
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )

        assert restored is None
        assert bank.last_miss_reason == "new_session"
    finally:
        cold.close()


def test_session_bank_cold_tier_finds_block_prefix_boundary_after_ram_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=1,
            max_bytes=1024,
            per_session_max_bytes=1024,
            cold_tier=cold,
        )
        bank.put_snapshot(
            runtime=runtime,
            token_ids=list(range(1200)),
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            session_id="opencode-session-a",
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=1200,
            nbytes_override=128,
        )
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()

        candidates = bank.near_prefix_candidates(
            list(range(1050)) + [99_001, 99_002, 99_003],
            block_size=256,
            block_min_matched_tokens=512,
            allow_block_prefix=True,
            model_path=str(runtime.model_path),
            mtp_enabled=runtime.mtp_enabled,
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )

        assert candidates
        entry, matched = candidates[0]
        assert matched == 1024
        assert entry.prefix_len == 1200
        assert getattr(entry, "cache_source") == "ssd"
        assert getattr(entry, "ssd_cache_hit") is True
        assert bank.last_prefix_diagnostic is not None
        assert bank.last_prefix_diagnostic["cache_source"] == "ssd"
        assert bank.last_prefix_diagnostic["restore_kind"] == "block_prefix"
    finally:
        cold.close()


def test_session_bank_cold_tier_finds_match_after_many_nonmatching_longer_entries(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=256,
            max_bytes=1024 * 1024,
            per_session_max_bytes=1024,
            cold_tier=cold,
        )
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=64,
        )
        for index in range(140):
            bank.put_snapshot(
                runtime=runtime,
                token_ids=[1000 + index, 2, 3, 4],
                cache_snapshot=CacheSnapshot(states=(), meta_states=()),
                logits=None,
                hidden=None,
                template_hash="template-a",
                policy_fingerprint="policy-a",
                snapshot_epoch=4,
                nbytes_override=64,
            )
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()

        restored = bank.restore(
            runtime,
            [1, 2, 3, 4, 5],
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )

        assert restored is not None
        assert restored.entry.token_ids == (1, 2, 3)
        assert restored.cache_source == "ssd"
    finally:
        cold.close()


def test_session_bank_cold_tier_honors_size_cap_by_skipping_new_writes(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=32,
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(cold_tier=cold)
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )
        assert cold.flush(timeout_s=5.0) is True

        stats = cold.stats()
        assert stats["entries"] == 0
        assert stats["writes_completed"] == 0
        assert stats["skipped_size_cap"] == 1
    finally:
        cold.close()


def test_session_bank_cold_tier_dedupes_shared_tensor_blocks(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=8 * 1024 * 1024,
        min_prefix_tokens=2,
        block_size=256,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            per_session_max_bytes=8 * 1024 * 1024,
            cold_tier=cold,
        )
        snapshot = CacheSnapshot(
            states=((mx.zeros((1, 1, 1024, 64), dtype=mx.float16),),),
            meta_states=(),
        )
        for offset in (0, 10_000):
            bank.put_snapshot(
                runtime=runtime,
                token_ids=list(range(offset, offset + 1024)),
                cache_snapshot=snapshot,
                logits=None,
                hidden=None,
                template_hash="template-a",
                policy_fingerprint="policy-a",
                snapshot_epoch=1024,
                nbytes_override=512 * 1024,
            )
        assert cold.flush(timeout_s=5.0) is True

        stats = cold.stats()
        assert stats["entries"] == 2
        assert stats["logical_bytes"] > stats["physical_bytes"]
        assert stats["deduped_bytes"] > 0
        assert stats["dedupe_ratio"] > 0
    finally:
        cold.close()


def test_session_bank_cold_tier_eviction_deletes_entry_and_unreferenced_blob(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=2600,
        min_prefix_tokens=2,
        block_size=256,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            per_session_max_bytes=8 * 1024 * 1024,
            cold_tier=cold,
        )
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(
                states=((mx.arange(256, dtype=mx.int32),),),
                meta_states=(),
            ),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=2048,
        )
        assert cold.flush(timeout_s=5.0) is True

        rows = _cold_rows(cold)
        assert len(rows) == 1
        first_entry_dir = cold.base_dir / str(rows[0]["entry_dir"])
        first_blob_paths = set((cold.base_dir / "blobs").rglob("*.bin"))
        assert first_entry_dir.exists()
        assert first_blob_paths

        bank.put_snapshot(
            runtime=runtime,
            token_ids=[4, 5, 6],
            cache_snapshot=CacheSnapshot(
                states=((mx.arange(256, dtype=mx.int32) + 10_000,),),
                meta_states=(),
            ),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=2048,
        )
        assert cold.flush(timeout_s=5.0) is True

        stats = cold.stats()
        assert stats["entries"] == 1
        assert stats["entries_evicted"] == 1
        assert stats["physical_bytes"] <= cold.max_bytes
        assert not first_entry_dir.exists()
        assert not (cold.base_dir / "evicted_entries").exists()
        assert all(not path.exists() for path in first_blob_paths)
        assert stats["untracked_file_bytes"] <= 1
    finally:
        cold.close()


def test_session_bank_cold_tier_eviction_keeps_shared_blob(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=8 * 1024 * 1024,
        min_prefix_tokens=2,
        block_size=256,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            per_session_max_bytes=8 * 1024 * 1024,
            cold_tier=cold,
        )
        shared_snapshot = CacheSnapshot(
            states=((mx.arange(256, dtype=mx.int32),),),
            meta_states=(),
        )
        for offset in (0, 10_000):
            bank.put_snapshot(
                runtime=runtime,
                token_ids=list(range(offset, offset + 3)),
                cache_snapshot=shared_snapshot,
                logits=None,
                hidden=None,
                template_hash="template-a",
                policy_fingerprint="policy-a",
                snapshot_epoch=3,
                nbytes_override=2048,
            )
        assert cold.flush(timeout_s=5.0) is True

        rows = _cold_rows(cold)
        assert len(rows) == 2
        blob_paths = list((cold.base_dir / "blobs").rglob("*.bin"))
        assert len(blob_paths) == 1

        cold._delete_entry_row(rows[0])

        assert _cold_rows(cold)[0]["entry_id"] == rows[1]["entry_id"]
        assert blob_paths[0].exists()
    finally:
        cold.close()


def test_session_bank_cold_tier_stats_report_untracked_cache_disk_usage(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=8 * 1024 * 1024,
        min_prefix_tokens=2,
    )
    try:
        orphan_dir = cold.base_dir / "evicted_entries" / "aa" / "orphan"
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "payload.bin").write_bytes(b"x" * 4096)

        cold._managed_disk_usage(force=True)
        stats = cold.stats()

        assert stats["entries"] == 0
        assert stats["physical_bytes"] == 0
        assert stats["managed_file_bytes"] >= 4096
        assert stats["managed_disk_bytes"] >= 4096
        assert stats["untracked_file_bytes"] >= 4096
        assert stats["untracked_disk_bytes"] >= 4096
    finally:
        cold.close()


def test_session_bank_cold_tier_cleanup_prunes_untracked_cache(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=1024,
        min_prefix_tokens=2,
    )
    try:
        evicted_dir = cold.base_dir / "evicted_entries" / "aa" / "old-entry"
        evicted_dir.mkdir(parents=True)
        (evicted_dir / "payload.bin").write_bytes(b"x" * 4096)

        orphan_entry = cold.base_dir / "entries" / "bb" / ("b" * 64)
        orphan_entry.mkdir(parents=True)
        (orphan_entry / "payload.json").write_text("{}", encoding="utf-8")

        orphan_blob = cold.base_dir / "blobs" / "cc" / f'{"c" * 64}.bin'
        orphan_blob.parent.mkdir(parents=True)
        orphan_blob.write_bytes(b"z" * 4096)

        result = cold._cleanup_untracked_cache_once()
        cold._managed_disk_usage(force=True)
        stats = cold.stats()

        assert result["files_deleted"] == 3
        assert result["disk_bytes_deleted"] >= 8192
        assert not (cold.base_dir / "evicted_entries").exists()
        assert not orphan_entry.exists()
        assert not orphan_blob.exists()
        assert stats["untracked_file_bytes"] == 0
        assert stats["untracked_disk_bytes"] == 0
    finally:
        cold.close()


def test_session_bank_cold_tier_writer_cleans_untracked_cache_before_cap_check(
    tmp_path,
):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=64 * 1024,
        min_prefix_tokens=2,
    )
    try:
        orphan_blob = cold.base_dir / "blobs" / "cc" / f'{"c" * 64}.bin'
        orphan_blob.parent.mkdir(parents=True)
        orphan_blob.write_bytes(b"z" * (64 * 1024))

        runtime = FakeRuntime()
        bank = SessionBank(cold_tier=cold)
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )
        assert cold.flush(timeout_s=5.0) is True

        cold._managed_disk_usage(force=True)
        stats = cold.stats()

        assert stats["entries"] == 1
        assert stats["writes_completed"] == 1
        assert stats["orphan_cleanup_runs"] >= 1
        assert stats["orphan_cleanup_disk_bytes_deleted"] >= 64 * 1024
        assert not orphan_blob.exists()
        assert stats["untracked_file_bytes"] == 0
        assert stats["untracked_disk_bytes"] < 4096
    finally:
        cold.close()


def test_session_bank_cold_tier_stats_schedules_disk_scan_without_blocking(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=8 * 1024 * 1024,
        min_prefix_tokens=2,
    )
    try:
        stats = cold.stats()

        assert stats["disk_usage_scan_pending"] is True
        assert stats["disk_usage_stale"] is True
        assert stats["managed_file_bytes"] == 0
    finally:
        cold.close()


def test_session_bank_ram_hit_wins_over_ssd(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=2,
            max_bytes=1024,
            per_session_max_bytes=1024,
            cold_tier=cold,
        )
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )
        assert cold.flush(timeout_s=5.0) is True

        restored = bank.restore(
            runtime,
            [1, 2, 3, 4],
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )

        assert restored is not None
        assert restored.cache_source == "ram"
        assert restored.ssd_cache_hit is False
    finally:
        cold.close()


def test_session_bank_cold_tier_rejects_identity_mismatch(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(cold_tier=cold)
        bank.put_snapshot(
            runtime=runtime,
            token_ids=[1, 2, 3],
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=3,
            nbytes_override=128,
        )
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()

        restored = bank.restore(
            runtime,
            [1, 2, 3, 4],
            template_hash="template-b",
            policy_fingerprint="policy-a",
        )

        assert restored is None
        assert bank.last_miss_reason == "new_session"
    finally:
        cold.close()


def test_session_bank_cold_tier_archive_renames_cache_dir(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        bank = SessionBank(cold_tier=cold)
        archive = bank.archive_cold_tier()

        assert archive["archived"] is True
        assert Path(archive["archive_path"]).exists()
        assert (tmp_path / "session-bank").exists()
        assert cold.stats()["entries"] == 0
    finally:
        cold.close()


def test_session_bank_cold_tier_archives_legacy_manifest_on_start(tmp_path):
    base_dir = tmp_path / "session-bank"
    base_dir.mkdir()
    with sqlite3.connect(str(base_dir / "manifest.sqlite")) as conn:
        conn.execute(
            "CREATE TABLE entries (entry_id TEXT PRIMARY KEY, format_version INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO entries (entry_id, format_version) VALUES ('legacy', 1)"
        )

    cold = SessionBankColdTier(
        base_dir=base_dir,
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        stats = cold.stats()
        assert stats["entries"] == 0
        assert stats["last_miss_reason"] == "legacy_ssd_cache_archived"
        assert stats["last_archive_path"] is not None
        assert Path(str(stats["last_archive_path"])).exists()
        assert base_dir.exists()
    finally:
        cold.close()


def test_cold_tier_v3_roundtrips_gdn_boundaries_and_boundary_restores(tmp_path, monkeypatch):
    """kvcache-v2 format v3: interior recurrent boundaries survive the SSD
    roundtrip, and a partial restore from an SSD-restored hybrid entry lands on
    the persisted boundary instead of failing closed."""
    import mlx.core as mx

    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=1,
            max_bytes=1 << 30,
            per_session_max_bytes=1 << 30,
            cold_tier=cold,
        )

        class RecurrentStub:
            def __init__(self):
                self.state = [mx.ones((2, 2)), None]
                self.meta_state = ("owned_recurrent_state", "persistent_eval")

            def is_trimmable(self):
                return False

            def replace_state(self, value):
                self.state = list(value)

        class AttentionStub:
            """Tiny trimmable KV stand-in with an on-disk block tail."""

            def __init__(self, tokens: int):
                self.state = (
                    mx.full((1, 1, tokens, 2), 5.0, dtype=mx.float16),
                    mx.full((1, 1, tokens, 2), 6.0, dtype=mx.float16),
                )
                self.meta_state = None

            def is_trimmable(self):
                return True

        boundary_state = CacheSnapshot(
            states=(None, mx.full((2, 2), 7.0)),
            meta_states=(None, ("boundary_recurrent_state",)),
        )
        hidden_last = mx.full((1, 1, 4), 3.0)
        entry = bank.put(
            runtime=runtime,
            token_ids=list(range(1200)),
            cache=[AttentionStub(1200), RecurrentStub()],
            logits=mx.zeros((1, 4)),
            hidden=None,
            session_id="hybrid-session",
            template_hash="template-a",
            mtp_history_policy="committed",
            policy_fingerprint="policy-a",
            mtp_history_snapshot=CacheSnapshot(
                states=(AttentionStub(1199).state,),
                meta_states=(None,),
            ),
            snapshot_epoch=1200,
            gdn_boundaries=[(1024, boundary_state, hidden_last)],
        )
        assert entry is not None
        assert entry.has_recurrent is True
        assert cold.flush(timeout_s=10.0) is True
        bank.clear()

        candidates = bank.near_prefix_candidates(
            list(range(1050)) + [99_001, 99_002, 99_003],
            block_size=256,
            block_min_matched_tokens=512,
            allow_block_prefix=True,
            model_path=str(runtime.model_path),
            mtp_enabled=runtime.mtp_enabled,
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )
        assert candidates
        ssd_entry, matched = candidates[0]
        assert getattr(ssd_entry, "cache_source") == "ssd"
        assert ssd_entry.has_recurrent is True
        assert [b for b, _, _ in ssd_entry.gdn_boundaries] == [1024]
        assert ssd_entry.cache_snapshot_prefix_len == 1024
        restored_k, restored_v = ssd_entry.cache_snapshot.states[0]
        assert restored_k.shape == (1, 1, 1024, 2)
        assert restored_v.shape == (1, 1, 1024, 2)
        assert ssd_entry.cache_snapshot.states[1] is None
        assert ssd_entry.mtp_history_snapshot_prefix_len == 1023
        assert ssd_entry.mtp_history_snapshot is not None
        assert ssd_entry.mtp_history_snapshot.states[0][0].shape == (1, 1, 1023, 2)
        assert ssd_entry.mtp_history_snapshot.states[0][1].shape == (1, 1, 1023, 2)
        restored_hidden = ssd_entry.gdn_boundaries[0][2]
        assert restored_hidden is not None
        assert float(mx.max(mx.abs(restored_hidden - hidden_last)).item()) == 0.0

        restored = bank.restore_entry_prefix_cache(
            runtime,
            ssd_entry,
            int(matched),
            mode="clone",
            # The foreground near-prefix path supplies a target-specific
            # factory.  Boundary metadata must still replace this cache's
            # initial recurrent metadata.
            cache_factory=lambda: [AttentionStub(0), RecurrentStub()],
            mtp_cache_factory=lambda: [AttentionStub(0)],
        )
        assert restored is not None
        cache, mtp_cache, mode, restore_point, boundary_hidden = restored
        assert restore_point == 1024
        assert boundary_hidden is not None
        attention, recurrent = cache
        assert attention.state[0].shape == (1, 1, 1024, 2)
        assert attention.state[1].shape == (1, 1, 1024, 2)
        assert mtp_cache is not None
        assert mtp_cache[0].state[0].shape == (1, 1, 1023, 2)
        assert mtp_cache[0].state[1].shape == (1, 1, 1023, 2)
        assert float(mx.max(mx.abs(recurrent.state[0] - mx.full((2, 2), 7.0))).item()) == 0.0
        assert recurrent.meta_state == ("boundary_recurrent_state",)
    finally:
        cold.close()


def test_parse_size_bytes_auto_falls_back_to_default():
    from mtplx.cache_bank import parse_size_bytes

    # The app's SSD "Auto" picker sends the literal string; it must resolve
    # to the RAM-tiered default instead of crashing daemon startup.
    assert parse_size_bytes("auto", 16 * 1024**3) == 16 * 1024**3
    assert parse_size_bytes("garbage", 7) == 7
