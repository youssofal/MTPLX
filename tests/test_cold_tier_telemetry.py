"""SSD cold-tier telemetry: where restore time goes, where restores land, and
how much the writer puts on disk.

Every store lives in pytest's tmp_path; nothing reads ~/.mtplx or a server
config.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import mlx.core as mx

from mtplx.cache_bank import codec
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_bank.telemetry import (
    ORIGIN_OTHER,
    ORIGIN_SAME,
    ORIGIN_UNKNOWN,
    RestoreLandings,
    WriteVolume,
    active_restore_timings,
    entry_origin,
    gdn_boundary_tensor_names,
    restore_timing,
)
from mtplx.cache_state import CacheSnapshot
from mtplx.generation import PromptState, _record_restore_landing
from mtplx.server.openai import (
    _metrics_envelope,
    _restore_telemetry_fields,
    _ssd_restore_split_fields,
)

BLOCK = 16
TOKENS = 128


def _random(shape, dtype, seed):
    # Distinct content per tensor and block: equal bytes would deduplicate
    # into one blob and blur the byte counts under test.
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(dtype)


def _entry(tokens, *, session_id="s1", boundaries=(), seed=0):
    kv = _random((1, 2, len(tokens), 4), mx.float16, seed + 1)
    recurrent = _random((1, 4), mx.float32, seed + 2)
    return SimpleNamespace(
        token_ids=tuple(tokens),
        nbytes=int(kv.nbytes + recurrent.nbytes),
        cache_snapshot=CacheSnapshot(
            states=[kv, recurrent], meta_states=[{"offset": len(tokens)}, None]
        ),
        logits=_random((1, 8), mx.float16, seed + 3),
        hidden=_random((1, 8), mx.float16, seed + 4),
        mtp_history_snapshot=None,
        gdn_boundaries=tuple(boundaries),
        has_recurrent=bool(boundaries),
        session_id=session_id,
        token_hash=f"hash-{session_id}-{len(tokens)}",
        prefix_len=len(tokens),
        model_path="model",
        mtp_enabled=False,
        hidden_variant=None,
        template_hash=None,
        mtp_history_policy=None,
        draft_head_identity=None,
        policy_fingerprint=None,
    )


def _boundary(tokens):
    return (
        int(tokens),
        CacheSnapshot(
            states=[None, _random((1, 4), mx.float32, 100 + tokens)],
            meta_states=[None, None],
        ),
        _random((1, 1, 8), mx.float16, 200 + tokens),
    )


def _tier(tmp_path):
    return SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1, block_size=BLOCK
    )


def _store(tier, *entries):
    before = tier.stats()["writes_completed"]
    for entry in entries:
        assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if tier.stats()["writes_completed"] >= before + len(entries):
            return
        time.sleep(0.02)
    raise AssertionError("writer did not complete")


def _lookup(tier, tokens):
    return tier.lookup(tokens, model_path="model", mtp_enabled=False)


def _lookup_prefix(tier, tokens):
    return tier.lookup_prefix_boundary(
        tokens,
        model_path="model",
        mtp_enabled=False,
        max_token_gap=8,
        min_matched_tokens=8,
        block_size=BLOCK,
        block_min_matched_tokens=BLOCK,
    )


def _blob_bytes(tier):
    return sum(path.stat().st_size for path in (tier.base_dir / "blobs").rglob("*.bin"))


class TestRestoreSplit:
    def test_exact_restore_reports_read_copy_eval_and_bytes(self, tmp_path, caplog):
        tier = _tier(tmp_path)
        try:
            tokens = list(range(TOKENS))
            _store(tier, _entry(tokens))
            with caplog.at_level(logging.INFO, logger="mtplx.cache_bank.cold_tier"):
                record = _lookup(tier, tuple(tokens) + (999,))
            assert record is not None
            split = record.restore_split
            assert split["bytes"] == _blob_bytes(tier)
            assert split["read_s"] > 0.0
            assert split["copy_s"] > 0.0
            assert split["eval_s"] > 0.0
            parts = split["read_s"] + split["copy_s"] + split["eval_s"] + split["other_s"]
            assert abs(parts - record.restore_s) < 1e-4
            stats = tier.stats()
            assert stats["last_restore_split"] == split
            assert stats["restore_bytes_total"] == split["bytes"]
            assert stats["restore_read_s_total"] == split["read_s"]
            line = next(
                message
                for message in caplog.messages
                if message.startswith("SessionBank SSD restored")
            )
            assert "restore_s=" in line and "read_s=" in line
            assert f"bytes={split['bytes']}" in line
        finally:
            tier.close()

    def test_partial_restore_counts_only_the_blobs_it_reads(self, tmp_path):
        tier = _tier(tmp_path)
        try:
            tokens = list(range(TOKENS))
            _store(tier, _entry(tokens, boundaries=[_boundary(48), _boundary(96)]))
            hit = _lookup_prefix(tier, tuple(tokens[:80]) + (999,) * 8)
            assert hit is not None
            split = hit.record.restore_split
            assert 0 < split["bytes"] < _blob_bytes(tier)
            assert split["read_s"] > 0.0
            assert hit.record.cache_snapshot_prefix_len == 48
            assert tier.stats()["restore_bytes_total"] == split["bytes"]
        finally:
            tier.close()

    def test_decodes_outside_a_restore_are_not_timed(self):
        assert active_restore_timings() is None
        spec = {"kind": "tensor", "name": "t", "dtype": "float32", "shape": [2]}
        raw = bytes(memoryview(mx.array([1.0, 2.0], dtype=mx.float32)))
        codec.decode_tree(spec, lambda name: raw)
        with restore_timing() as timings:
            assert active_restore_timings() is timings
            codec.decode_tree(spec, lambda name: raw)
        assert active_restore_timings() is None
        assert timings.bytes_read == len(raw)


class TestWriteVolume:
    def test_writer_splits_checkpoint_and_kv_bytes_and_counts_dedupe(self, tmp_path):
        tier = _tier(tmp_path)
        try:
            tokens = list(range(TOKENS))
            _store(tier, _entry(tokens, boundaries=[_boundary(64)]))
            volume = tier.stats()["write_volume"]
            assert volume["written_bytes_last_hour"] == _blob_bytes(tier)
            assert volume["gdn_boundary_written_bytes_last_hour"] > 0
            assert volume["kv_written_bytes_last_hour"] > 0
            assert volume["deduped_bytes_last_hour"] == 0
            assert volume["write_budget_per_hour_bytes"] == 128 * 1024**3
            # Same tensors under another session: every blob is on disk.
            _store(tier, _entry(tokens, session_id="s2", boundaries=[_boundary(64)]))
            again = tier.stats()["write_volume"]
            assert again["written_bytes_last_hour"] == volume["written_bytes_last_hour"]
            assert again["deduped_bytes_last_hour"] == volume["written_bytes_last_hour"]
            assert (
                again["gdn_boundary_deduped_bytes_last_hour"]
                == volume["gdn_boundary_written_bytes_last_hour"]
            )
        finally:
            tier.close()

    def test_spill_is_counted_too(self, tmp_path):
        tier = _tier(tmp_path)
        try:
            entry = _entry(list(range(TOKENS)), boundaries=[_boundary(64)])
            assert tier.spill_entry(entry, capabilities=["ar_insert"]) is True
            volume = tier.stats()["write_volume"]
            assert volume["written_bytes_last_hour"] == _blob_bytes(tier)
            assert volume["gdn_boundary_written_bytes_last_hour"] > 0
        finally:
            tier.close()

    def test_rolling_hour_and_budget_fraction(self):
        now = [10_000.0]
        volume = WriteVolume(clock=lambda: now[0])
        blobs = {"a": {"nbytes": 100}, "b": {"nbytes": 50}, "c": {"nbytes": 25}}
        volume.record(blobs, {"c"}, {"b"})
        snap = volume.snapshot(budget_per_hour_bytes=1000)
        assert snap["kv_written_bytes_last_hour"] == 100
        assert snap["gdn_boundary_written_bytes_last_hour"] == 50
        assert snap["kv_deduped_bytes_last_hour"] == 25
        assert snap["budget_used_fraction"] == 0.15
        now[0] += 3601.0
        later = volume.snapshot(budget_per_hour_bytes=1000)
        assert later["written_bytes_last_hour"] == 0
        assert later["written_bytes_total"] == 150

    def test_gdn_boundary_tensor_names(self):
        spec = {
            "cache_snapshot": {"states": {"kind": "tensor", "name": "t0"}},
            "gdn_boundaries": [
                {
                    "tokens": 64,
                    "states": {
                        "kind": "list",
                        "items": [
                            {"kind": "none"},
                            {"kind": "tensor_blocks", "blocks": [{"name": "t1"}, {"name": "t2"}]},
                        ],
                    },
                    "meta_states": {"kind": "list", "items": []},
                    "hidden_last": {"kind": "tensor", "name": "t3"},
                }
            ],
        }
        assert gdn_boundary_tensor_names(spec) == {"t1", "t2", "t3"}
        assert gdn_boundary_tensor_names(None) == set()


def _blob_count(tier):
    return sum(1 for _ in (tier.base_dir / "blobs").rglob("*.bin"))


class TestWriteSplit:
    """Deduplicated blobs by how they were found: referenced by a memo (no
    capture, no hash) or captured and hashed before the disk said so."""

    def _spill(self, tier, entry):
        assert tier.spill_entry(entry, capabilities=["ar_insert"]) is True

    def _queued(self, tier, entry):
        assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
        # flush, not _store: the writer updates the memo after the write.
        assert tier.flush(timeout_s=10.0) is True

    def _check_write_twice(self, tier, write):
        tokens = list(range(TOKENS))
        write(tier, _entry(tokens, boundaries=[_boundary(64)]))
        first = tier.stats()
        blobs, nbytes = _blob_count(tier), _blob_bytes(tier)
        assert first["write_hashed_new_blobs"] == blobs
        assert first["write_hashed_new_bytes"] == nbytes
        assert first["write_hashed_duplicate_blobs"] == 0
        assert first["write_referenced_blobs"] == 0
        assert first["write_hashed_new_hash_s"] > 0.0
        assert first["write_hashed_new_capture_s"] > 0.0
        # The same tensors under another session (no memo applies): every
        # blob is captured and hashed, then found on disk.
        write(tier, _entry(tokens, session_id="s2", boundaries=[_boundary(64)]))
        second = tier.stats()
        last = second["last_write_split"]
        assert last["hashed_duplicate"]["blobs"] == blobs
        assert last["hashed_duplicate"]["bytes"] == nbytes
        assert last["hashed_duplicate"]["hash_s"] > 0.0
        assert last["hashed_duplicate"]["capture_s"] > 0.0
        assert last["hashed_new"]["blobs"] == 0
        assert last["referenced"] == {"blobs": 0, "bytes": 0}
        assert second["write_hashed_duplicate_bytes"] == nbytes
        assert second["deduped_blob_hits"] == blobs
        volume = second["write_volume"]
        assert volume["deduped_hashed_bytes_total"] == nbytes
        assert volume["deduped_referenced_bytes_total"] == 0
        assert volume["deduped_bytes_total"] == nbytes
        return last

    def test_queued_write(self, tmp_path):
        tier = _tier(tmp_path)
        try:
            assert self._check_write_twice(tier, self._queued)
            assert tier.stats()["last_write_split"]["path"] == "queued"
        finally:
            tier.close()

    def test_spill_write(self, tmp_path):
        tier = _tier(tmp_path)
        try:
            assert self._check_write_twice(tier, self._spill)
            assert tier.stats()["last_write_split"]["path"] == "spill"
        finally:
            tier.close()

    def test_memo_hits_are_referenced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
        tier = _tier(tmp_path)
        try:
            tokens = list(range(TOKENS))
            self._queued(tier, _entry(tokens))
            # Other tokens, the same tensors: a new entry of the same session.
            self._queued(tier, _entry(tokens[:-1] + [7777]))
            stats = tier.stats()
            last = stats["last_write_split"]
            # Every full KV block was referenced; the small tensors below the
            # whole-tensor fingerprint floor were captured and deduplicated.
            assert last["referenced"]["blobs"] == TOKENS // BLOCK
            assert last["referenced"]["blobs"] == stats["incremental_reused_blobs"]
            assert last["referenced"]["bytes"] == stats["incremental_reused_bytes"]
            assert last["hashed_duplicate"]["blobs"] == 3
            assert last["hashed_new"]["blobs"] == 0
            volume = stats["write_volume"]
            assert volume["deduped_referenced_bytes_total"] == last["referenced"]["bytes"]
            assert volume["deduped_hashed_bytes_total"] == last["hashed_duplicate"]["bytes"]
        finally:
            tier.close()

    def test_write_volume_splits_referenced_from_hashed(self):
        volume = WriteVolume(clock=lambda: 10_000.0)
        blobs = {"a": {"nbytes": 100}, "b": {"nbytes": 50}, "c": {"nbytes": 25}}
        volume.record(blobs, {"b", "c"}, {"c"}, {"b"})
        snap = volume.snapshot(budget_per_hour_bytes=1000)
        assert snap["deduped_bytes_last_hour"] == 75
        assert snap["deduped_referenced_bytes_last_hour"] == 50
        assert snap["deduped_hashed_bytes_last_hour"] == 25
        assert snap["deduped_referenced_bytes_total"] == 50
        assert snap["deduped_hashed_bytes_total"] == 25


class TestRestoreLandings:
    def test_origin(self):
        assert entry_origin("a", "a") == ORIGIN_SAME
        assert entry_origin("a", "b") == ORIGIN_OTHER
        assert entry_origin(None, "b") == ORIGIN_UNKNOWN

    def test_counters_per_source(self):
        landings = RestoreLandings()
        landings.record(
            {
                "source": "ssd",
                "partial": True,
                "hybrid": True,
                "matched_tokens": 80,
                "restore_point_tokens": 48,
                "checkpoint_gap_tokens": 32,
                "suffix_tokens": 40,
                "origin": ORIGIN_OTHER,
            }
        )
        landings.record({"source": "ram", "matched_tokens": 10, "origin": None})
        snap = landings.snapshot()
        assert snap["ssd"]["restores"] == 1
        assert snap["ssd"]["partial"] == 1
        assert snap["ssd"][ORIGIN_OTHER] == 1
        assert snap["ssd"]["checkpoint_gap_tokens"] == 32
        assert snap["ram"]["exact"] == 1
        assert snap["ram"][ORIGIN_UNKNOWN] == 1
        assert snap["last"]["source"] == "ram"

    def test_generation_records_the_landing(self):
        recorded = []
        bank = SimpleNamespace(record_restore_landing=recorded.append)
        served = {
            "entry_prefix_len": 128,
            "requested_matched": 80,
            "actual_restore_point": 48,
            "has_recurrent": True,
            "entry_origin": ORIGIN_SAME,
        }
        state = PromptState(
            trunk_cache=None,
            logits=None,
            hidden=None,
            committed_mtp_cache=None,
            token_prefix=(),
            prompt_eval_time_s=0.0,
            suffix_tokens=40,
            cache_source="ssd",
            ssd_cache_hit=True,
            restore_served=served,
        )
        _record_restore_landing(bank, state)
        assert recorded == [served["landing"]]
        assert served["landing"] == {
            "source": "ssd",
            "partial": True,
            "hybrid": True,
            "matched_tokens": 80,
            "restore_point_tokens": 48,
            "checkpoint_gap_tokens": 32,
            "suffix_tokens": 40,
            "origin": ORIGIN_SAME,
        }

    def test_cold_prompt_state_records_nothing(self):
        recorded = []
        bank = SimpleNamespace(record_restore_landing=recorded.append)
        state = PromptState(
            trunk_cache=None,
            logits=None,
            hidden=None,
            committed_mtp_cache=None,
            token_prefix=(),
            prompt_eval_time_s=0.0,
        )
        _record_restore_landing(bank, state)
        assert recorded == []


class TestRequestLogFields:
    def test_split_fields_default_to_zero(self):
        assert _ssd_restore_split_fields(None) == {
            "ssd_restore_read_s": 0.0,
            "ssd_restore_copy_s": 0.0,
            "ssd_restore_eval_s": 0.0,
            "ssd_restore_other_s": 0.0,
            "ssd_restore_bytes": 0,
        }

    def test_envelope_carries_split_and_landing(self):
        stats = {
            "ssd_cache_hit": True,
            "ssd_restore_s": 0.5,
            "session_restore_served": {
                "ssd_restore": {
                    "read_s": 0.3,
                    "copy_s": 0.1,
                    "eval_s": 0.05,
                    "other_s": 0.05,
                    "bytes": 4096,
                },
                "landing": {
                    "partial": True,
                    "matched_tokens": 80,
                    "restore_point_tokens": 48,
                    "checkpoint_gap_tokens": 32,
                    "origin": ORIGIN_OTHER,
                },
            },
        }
        envelope = _metrics_envelope(
            stats=stats,
            prompt_tokens=88,
            completion_tokens=1,
            request_elapsed_s=1.0,
            token_times=[],
            request_started_s=0.0,
            lock_wait_time_s=0.0,
            session_id="s1",
            session_cache_hit=True,
            cache_miss_reason=None,
            session_restore_mode="ssd_block_prefix_boundary_clone",
            mtp_depth=0,
            generation_limits={},
        )
        assert envelope["ssd_restore_read_s"] == 0.3
        assert envelope["ssd_restore_bytes"] == 4096
        assert envelope["prefix_restore_point_tokens"] == 48
        assert envelope["prefix_restore_checkpoint_gap_tokens"] == 32
        assert envelope["prefix_restore_origin"] == ORIGIN_OTHER
        assert envelope["prefix_restore_partial"] is True

    def test_a_ram_hit_reports_no_ssd_split(self):
        stats = {
            "ssd_cache_hit": False,
            "session_restore_served": {"ssd_restore": {"read_s": 1.0, "bytes": 9}},
        }
        fields = _restore_telemetry_fields(stats)
        assert fields["ssd_restore_read_s"] == 0.0
        assert fields["ssd_restore_bytes"] == 0
        assert fields["prefix_restore_origin"] is None

    def test_ar_batch_flat_fields_win(self):
        stats = {"ssd_cache_hit": True, **_ssd_restore_split_fields({"read_s": 0.2, "bytes": 7})}
        fields = _restore_telemetry_fields(stats)
        assert fields["ssd_restore_read_s"] == 0.2
        assert fields["ssd_restore_bytes"] == 7


def test_session_bank_carries_the_split_and_counts_landings(tmp_path):
    from mtplx.session_bank import SessionBank

    tier = _tier(tmp_path)
    try:
        tokens = list(range(TOKENS))
        _store(tier, _entry(tokens, boundaries=[_boundary(48)]))
        bank = SessionBank(
            max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30, cold_tier=tier
        )
        matches = bank.near_prefix_candidates(
            tuple(tokens[:80]) + (999,) * 8,
            max_token_gap=8,
            min_matched_tokens=8,
            block_size=BLOCK,
            block_min_matched_tokens=BLOCK,
            model_path="model",
            mtp_enabled=False,
        )
        cold = [entry for entry, _ in matches if getattr(entry, "cache_source", "") == "ssd"]
        assert cold, matches
        split = cold[0].ssd_restore_split
        assert split["bytes"] > 0 and split["read_s"] > 0.0
        bank.record_restore_landing({"source": "ssd", "partial": True, "matched_tokens": 80})
        landings = bank.to_dict()["restore_landings"]
        assert landings["ssd"]["restores"] == 1
        assert landings["ssd"]["matched_tokens"] == 80
    finally:
        tier.close()
