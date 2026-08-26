from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.session_bank import SessionBank


class DenseMaterializingCache:
    @property
    def state(self):
        raise RuntimeError("Paged KV cache attempted to materialize active K/V arrays")


class TrimmableLiveCache:
    def __init__(self, offset: int):
        self.offset = offset
        self.trimmed: list[int] = []

    @property
    def state(self):
        raise RuntimeError("Paged KV cache attempted to materialize active K/V arrays")

    def is_trimmable(self) -> bool:
        # Models a real attention KV container; without this the bank's
        # conservative recurrent detection treats it as untrimmable state and
        # boundary-true restores fail closed.
        return True

    def trim(self, n: int) -> int:
        self.trimmed.append(int(n))
        self.offset -= int(n)
        return int(n)


class RuntimeWithCaches:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return [TrimmableLiveCache(0)]

    def make_mtp_cache(self):
        return [TrimmableLiveCache(0)]


def test_session_bank_skips_single_oversized_snapshot_before_insert():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=2048,
    )

    assert entry is None
    assert len(bank) == 0
    assert bank.last_put_nbytes == 2048
    assert bank.last_put_skipped_oversized_snapshot is True
    assert bank.eviction_log[-1]["reason"] == "skipped_oversized_snapshot"


def test_session_bank_skips_dense_materializing_snapshot():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[DenseMaterializingCache()],
        logits=None,
        hidden=None,
        session_id="session-1",
    )

    assert entry is None
    assert len(bank) == 0
    assert bank.last_put_skipped_oversized_snapshot is True
    assert bank.eviction_log[-1]["reason"] == "skipped_dense_materializing_snapshot"


def test_session_bank_oversized_prompt_prefix_can_use_live_reference_lease():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=11)]
    mtp_cache = [TrimmableLiveCache(offset=11)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(10)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="committed",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=10,
        mtp_snapshot_epoch=10,
        nbytes_override=2048,
    )

    assert entry is not None
    assert entry.live_ref_only is True
    assert entry.cache_ref is cache
    assert entry.mtp_history_cache_ref is mtp_cache
    assert bank.eviction_log[-1]["fallback"] == "live_reference_lease"

    restored = bank.restore(
        runtime,
        list(range(10)),
        mode="reference",
        session_id="session-1",
        mtp_history_policy="committed",
    )

    assert restored is not None
    assert restored.restore_mode == "reference_lease"
    assert restored.cache is cache
    assert restored.mtp_history_cache is mtp_cache
    assert cache[0].offset == 9
    assert mtp_cache[0].offset == 9
    assert entry.cache_ref is None
    assert entry.mtp_history_cache_ref is None


def test_session_bank_clone_restore_can_use_custom_cache_factory():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    custom_cache = []

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits="logits",
        hidden="hidden",
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    restored = bank.restore(
        runtime,
        [1, 2, 3, 4],
        mode="clone",
        cache_factory=lambda: custom_cache,
    )

    assert restored is not None
    assert restored.cache is custom_cache
    assert restored.restore_mode == "clone"


def test_session_bank_live_reference_can_restore_block_prefix_boundary():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=1199)]
    mtp_cache = [TrimmableLiveCache(offset=1199)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="committed",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=1200,
        mtp_snapshot_epoch=1200,
        nbytes_override=2048,
    )

    assert entry is not None
    assert entry.live_ref_only is True

    restored = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        1024,
        mode="reference",
    )

    assert restored is not None
    restored_cache, restored_mtp_cache, restore_mode, restore_point, boundary_hidden = (
        restored
    )
    assert restored_cache is cache
    assert restored_mtp_cache is mtp_cache
    assert restore_mode == "reference_lease"
    assert restore_point == 1024
    assert boundary_hidden is None
    assert cache[0].offset == 1023
    assert mtp_cache[0].offset == 1023
    assert entry.cache_ref is None
    assert entry.mtp_history_cache_ref is None


def test_session_bank_near_prefix_trims_mtp_history_by_gap_not_absolute_offset():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=1199)]
    mtp_cache = [TrimmableLiveCache(offset=127)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="last_window",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=1200,
        mtp_snapshot_epoch=1200,
        nbytes_override=2048,
    )

    assert entry is not None

    restored = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        1199,
        mode="reference",
    )

    assert restored is not None
    assert cache[0].trimmed == [1]
    assert mtp_cache[0].trimmed == [1]
    assert cache[0].offset == 1198
    assert mtp_cache[0].offset == 126


def test_session_bank_near_prefix_candidates_only_accept_boundary_drift():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    near = list(range(197)) + [10_001, 10_002, 10_003, 10_004]
    far = list(range(120)) + [20_001, 20_002]

    candidates = bank.near_prefix_candidates(
        near,
        max_token_gap=8,
        min_matched_tokens=64,
    )

    assert candidates == [(entry, 197)]
    assert (
        bank.near_prefix_candidates(
            far,
            max_token_gap=8,
            min_matched_tokens=64,
            allow_block_prefix=False,
        )
        == []
    )


def test_session_bank_near_prefix_rejects_prompt_inside_longer_completion():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(70)) + [90_001, 90_002],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    prompt_only = list(range(70))

    assert (
        bank.near_prefix_candidates(
            prompt_only,
            max_token_gap=8,
            min_matched_tokens=64,
            allow_block_prefix=True,
        )
        == []
    )


def test_session_bank_contained_long_prompt_uses_block_prefix_not_answer_tail():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    prompt_inside_completion = list(range(1197))
    candidates = bank.near_prefix_candidates(
        prompt_inside_completion,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    )

    # kvcache-v2: matches are token-exact (no block quantization) for entries
    # that can restore at any offset. A contained prompt restores at its own
    # full length; the trim + seed-forward make that state cold-identical, so
    # the pre-v2 "back off to the last block edge" conservatism is obsolete.
    assert candidates == [(entry, 1197)]
    assert bank.last_prefix_diagnostic is not None
    assert bank.last_prefix_diagnostic["restore_kind"] == "near_boundary"


def test_session_bank_block_prefix_candidates_restore_large_agent_overlap():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    followup = list(range(1050)) + [99_001, 99_002, 99_003]
    candidates = bank.near_prefix_candidates(
        followup,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    )

    # kvcache-v2 token-granularity: the agent follow-up diverges at 1050, so
    # the candidate matches exactly there instead of backing off to 1024.
    assert candidates == [(entry, 1050)]
    assert bank.last_prefix_diagnostic is not None
    assert bank.last_prefix_diagnostic["restore_kind"] == "block_prefix"
    assert bank.last_prefix_diagnostic["new_prefill_tokens"] == len(followup) - 1050


# --- prefix-supersede (2026-07-04 multitask capacity fix) --------------------
# One busy OpenCode conversation banked 13/16 RAM entries (20.6 of 24 GB),
# a third of them strict prefixes of a newer entry; multitasking across
# projects then churned every other project out of RAM. A newer entry that
# extends an older one dominates it for every restore shape, so the bank
# drops the contained entry at put() time.


def test_session_bank_put_supersedes_contained_prefixes():
    bank = SessionBank(max_entries=8, max_bytes=4096, per_session_max_bytes=2048)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    short = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4],
        cache=[],
        logits=None,
        hidden=None,
        session_id="round-1",
        nbytes_override=64,
    )
    assert short is not None

    longer = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4, 5, 6],
        cache=[],
        logits=None,
        hidden=None,
        session_id="round-2",
        nbytes_override=64,
    )
    assert longer is not None

    assert len(bank) == 1
    assert bank.longest_prefix([1, 2, 3, 4, 5, 6, 7]) is longer
    assert bank.eviction_log[-1]["reason"] == "superseded_by_longer_prefix"
    assert bank.eviction_log[-1]["prefix_len"] == 4


def test_session_bank_put_keeps_divergent_and_policy_mismatched_entries():
    bank = SessionBank(max_entries=8, max_bytes=4096, per_session_max_bytes=2048)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    divergent = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 9, 9],
        cache=[],
        logits=None,
        hidden=None,
        session_id="other-project",
        nbytes_override=64,
    )
    policy_mismatch = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        session_id="old-policy",
        policy_fingerprint="policy-A",
        nbytes_override=64,
    )
    container = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4, 5],
        cache=[],
        logits=None,
        hidden=None,
        session_id="round-2",
        policy_fingerprint="policy-B",
        nbytes_override=64,
    )

    assert divergent is not None
    assert policy_mismatch is not None
    assert container is not None
    # The divergent prefix is not contained; the contained entry carries a
    # different policy fingerprint and can serve requests the container
    # cannot. Both must survive.
    assert len(bank) == 3


def test_session_bank_recurrent_container_without_boundaries_does_not_supersede():
    bank = SessionBank(max_entries=8, max_bytes=4096, per_session_max_bytes=2048)
    runtime = RuntimeWithCaches()

    class RecurrentCache:
        state = None

        def is_trimmable(self) -> bool:
            return False

    short = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4],
        cache=[RecurrentCache()],
        logits=None,
        hidden=None,
        session_id="round-1",
        nbytes_override=64,
    )
    longer = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4, 5, 6],
        cache=[RecurrentCache()],
        logits=None,
        hidden=None,
        session_id="round-2",
        nbytes_override=64,
    )

    assert short is not None
    assert longer is not None
    # A recurrent container with no interior boundaries fails closed on
    # sub-prefix restores, so the shorter exact frontier still adds coverage.
    assert len(bank) == 2


def test_eviction_log_is_bounded_for_daemon_lifetime():
    # The log is appended on every eviction/skip forever while health
    # snapshots read only the newest entries: an unbounded list is pure
    # retention on a long-running agent daemon (external review F5).
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    assert bank.eviction_log.maxlen == 256
    for index in range(300):
        bank.put(
            runtime=runtime,
            token_ids=[1, 2, index],
            cache=[],
            logits=None,
            hidden=None,
            session_id=f"session-{index}",
            nbytes_override=2048,
        )
    assert len(bank.eviction_log) == 256
    # Newest entry survives at the tail; the oldest 44 fell off the front.
    assert bank.eviction_log[-1]["reason"] == "skipped_oversized_snapshot"


def test_cross_session_eviction_prefers_idle_sessions_over_active_ones():
    # 2026-07-31 live incident: cross-session LRU pressure evicted a
    # mid-run coding session's warm entry, forcing an 85.6k-token full
    # re-prefill on its next turn. Sessions that touched the bank within
    # the active-pin TTL are eviction-last under cross-session pressure.
    bank = SessionBank(max_entries=8, max_bytes=1000, per_session_max_bytes=1000)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    bank.put(
        runtime=runtime, token_ids=[1, 2, 3], cache=[], logits=None,
        hidden=None, session_id="idle", nbytes_override=400,
    )
    bank.put(
        runtime=runtime, token_ids=[9, 9, 9], cache=[], logits=None,
        hidden=None, session_id="active", nbytes_override=400,
    )
    # The idle session went stale past the TTL; rig last_access so pure LRU
    # would pick the ACTIVE session's entry — the preference must override.
    bank._session_last_active["idle"] -= bank.active_pin_ttl_s + 1.0
    for entry in bank._entries.values():
        entry.last_access_s = 0.0 if entry.session_id == "active" else 1e12
    bank.put(
        runtime=runtime, token_ids=[5, 5, 5], cache=[], logits=None,
        hidden=None, session_id="trigger", nbytes_override=400,
    )
    survivors = {entry.session_id for entry in bank._entries.values()}
    assert survivors == {"active", "trigger"}
    assert bank.eviction_log[-1]["session_id"] == "idle"
    assert bank.eviction_log[-1]["session_active"] is False


def test_active_session_over_its_own_budget_still_self_evicts():
    # Per-session budget enforcement is self-inflicted pressure: an active
    # session exceeding its own cap sheds its oldest entries even while
    # pinned, keeping the newest (protected) snapshot.
    bank = SessionBank(max_entries=8, max_bytes=10_000, per_session_max_bytes=500)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    bank.put(
        runtime=runtime, token_ids=[1, 2], cache=[], logits=None,
        hidden=None, session_id="live", nbytes_override=300,
    )
    bank.put(
        runtime=runtime, token_ids=[7, 7, 7], cache=[], logits=None,
        hidden=None, session_id="live", nbytes_override=300,
    )
    lens = sorted(entry.prefix_len for entry in bank._entries.values())
    assert lens == [3]
    assert bank.eviction_log[-1]["session_id"] == "live"


def test_per_session_entry_retention_bounds_divergent_siblings():
    # 2026-08-01 live leak: divergent same-session tails are not strict
    # prefixes, so supersede never fires and one agent session accumulated
    # 5 near-duplicate multi-GB snapshots. Newest-K retention bounds it.
    bank = SessionBank(max_entries=16, max_bytes=10_000, per_session_max_bytes=10_000)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    for i in range(5):
        bank.put(
            runtime=runtime, token_ids=[7, 7, 100 + i], cache=[], logits=None,
            hidden=None, session_id="agent", nbytes_override=100,
        )
    survivors = sorted(e.token_ids[-1] for e in bank._entries.values())
    assert len(survivors) == bank.per_session_max_entries == 3
    assert 104 in survivors  # newest always kept
    assert bank.eviction_log[-1]["reason"] == "session_entry_retention"
    # Other sessions unaffected by one session's churn.
    bank.put(
        runtime=runtime, token_ids=[9, 9, 9], cache=[], logits=None,
        hidden=None, session_id="other", nbytes_override=100,
    )
    assert sum(1 for e in bank._entries.values() if e.session_id == "other") == 1
    assert sum(1 for e in bank._entries.values() if e.session_id == "agent") == 3
