"""Restart-warm boundary persistence (#121/#159/#144 residual, 2.0.4).

The kvcache-v2 SSD format persists interior recurrent boundaries and the
exact-restore lane rehydrates them lazily (``gdn_boundary_loader``). Two
wiring gaps silently stripped the records from every SECOND generation of
persistence — the exact restart-warm agentic shape:

1. ``SessionBank.put``'s prefix-donor inheritance only accepted donors with
   MATERIALIZED boundaries, so a loader-backed donor (any exact SSD restore
   after a restart) contributed nothing to the extended turn's entry.
2. ``SessionBankColdTier.put_entry`` read only materialized
   ``entry.gdn_boundaries``, so a loader-backed entry re-persisted to SSD
   wrote a package with zero boundary records.

Composition: restart -> exact restore (loader-backed) -> next tool turn puts
the extended prefix (boundary-less pre-fix) -> that entry hits SSD without
records -> every later near-prefix restore on the lineage fails closed to
clean-prefix. These tests pin the composed flow end to end.
"""

from pathlib import Path

import mlx.core as mx
import pytest

from mtplx.cache_bank import SessionBankColdTier
from mtplx.session_bank import CacheSnapshot, SessionBank


class FakeRuntime:
    model_path = Path("/tmp/fake-model")
    mtp_enabled = True

    def make_cache(self):
        return [RecurrentStub()]

    def make_mtp_cache(self):
        return []


class RecurrentStub:
    def __init__(self):
        self.state = [mx.ones((2, 2)), None]
        self.meta_state = ("owned_recurrent_state", "persistent_eval")

    def is_trimmable(self):
        return False

    def replace_state(self, value):
        self.state = list(value)


def _boundary_state() -> CacheSnapshot:
    return CacheSnapshot(states=(mx.full((2, 2), 7.0),), meta_states=(None,))


def test_cold_put_hydrates_loader_backed_boundaries(tmp_path, monkeypatch):
    """A loader-backed entry re-persisted to SSD must keep its records.

    Identity re-puts are additionally protected by the writer's entry-id
    idempotency (same tokens -> same entry_id -> touch, not overwrite), so
    the original boundary-carrying package survives either way; this test
    pins the whole path so neither layer regresses.
    """
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=4,
            max_bytes=1 << 30,
            per_session_max_bytes=1 << 30,
            cold_tier=cold,
        )
        hidden_last = mx.full((1, 1, 4), 3.0)
        entry = bank.put(
            runtime=runtime,
            token_ids=list(range(1200)),
            cache=[RecurrentStub()],
            logits=mx.zeros((1, 4)),
            hidden=None,
            session_id="restart-session",
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=1200,
            gdn_boundaries=[(1024, _boundary_state(), hidden_last)],
        )
        assert entry is not None
        assert cold.flush(timeout_s=10.0) is True
        bank.clear()

        # Exact restore after "restart": entry comes back loader-backed
        # (boundaries deferred — the premise this regression guards).
        restored = bank.restore(
            runtime,
            list(range(1200)),
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )
        assert restored is not None
        assert restored.cache_source == "ssd"
        loader_backed = restored.entry
        assert loader_backed.gdn_boundaries == []
        assert loader_backed.gdn_boundary_loader is not None

        # Re-persist the loader-backed entry (identity re-put is the
        # postcommit shape). The SSD package must carry the records.
        assert cold.stats()["entries"] == 1
        bank._enqueue_cold_entry(loader_backed)
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
        assert candidates, "SSD near-prefix candidate expected after re-persist"
        ssd_entry, _matched = candidates[0]
        assert [b for b, _, _ in ssd_entry.gdn_boundaries] == [1024], (
            "re-persisted package lost its boundary records"
        )
    finally:
        cold.close()


def test_prefix_donor_inheritance_accepts_loader_backed_donor(tmp_path, monkeypatch):
    """An extended-turn put must inherit boundaries from a loader-backed donor."""
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )
    try:
        runtime = FakeRuntime()
        bank = SessionBank(
            max_entries=4,
            max_bytes=1 << 30,
            per_session_max_bytes=1 << 30,
            cold_tier=cold,
        )
        hidden_last = mx.full((1, 1, 4), 3.0)
        assert (
            bank.put(
                runtime=runtime,
                token_ids=list(range(1200)),
                cache=[RecurrentStub()],
                logits=mx.zeros((1, 4)),
                hidden=None,
                session_id="restart-session",
                template_hash="template-a",
                policy_fingerprint="policy-a",
                snapshot_epoch=1200,
                gdn_boundaries=[(1024, _boundary_state(), hidden_last)],
            )
            is not None
        )
        assert cold.flush(timeout_s=10.0) is True
        bank.clear()
        restored = bank.restore(
            runtime,
            list(range(1200)),
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )
        assert restored is not None
        assert restored.entry.gdn_boundary_loader is not None

        # Next tool turn: extended prefix, no PromptState in scope
        # (generation-final commit shape — gdn_boundaries=None).
        extended = bank.put(
            runtime=runtime,
            token_ids=list(range(1200)) + [50_001, 50_002, 50_003],
            cache=[RecurrentStub()],
            logits=mx.zeros((1, 4)),
            hidden=None,
            session_id="restart-session",
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=1203,
            gdn_boundaries=None,
        )
        assert extended is not None
        assert extended.gdn_boundaries or extended.gdn_boundary_loader is not None, (
            "extended turn lost the donor's boundary records "
            "(loader-backed donor was skipped)"
        )

        # And the descendant's SSD package must carry them too — the full
        # restart-warm composition (#159's cross-session shape).
        assert cold.flush(timeout_s=10.0) is True
        bank.clear()
        candidates = bank.near_prefix_candidates(
            list(range(1050)) + [77_001, 77_002, 77_003],
            block_size=256,
            block_min_matched_tokens=512,
            allow_block_prefix=True,
            model_path=str(runtime.model_path),
            mtp_enabled=runtime.mtp_enabled,
            template_hash="template-a",
            policy_fingerprint="policy-a",
        )
        assert candidates, "SSD near-prefix candidate expected after restart-warm turn"
        boundary_lists = [
            [b for b, _, _ in entry.gdn_boundaries] for entry, _ in candidates
        ]
        assert any(bl == [1024] for bl in boundary_lists), (
            f"no candidate carried the inherited boundary: {boundary_lists}"
        )
    finally:
        cold.close()
