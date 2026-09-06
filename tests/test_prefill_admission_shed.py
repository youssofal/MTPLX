"""#415: pre-prefill admission shed.

The shipped failure: a Pi agent auto-compacted at 248k, the compaction
rewrote the whole prefix, and the 38k replacement prefill — a guaranteed
cache miss — started while the superseded 6.09 GiB SessionBank snapshot
of the pre-compaction transcript stayed resident on a memory plan that
admits 262K with zero headroom. The footprint crossed the Metal cap
mid-prefill and the sustained-pressure guard killed the request with a
structured 507 ~30 s later.

These tests pin the admission guard that sheds BEFORE the prefill:
unused allocator storage first, then superseded same-session entries and
LRU idle entries with active sessions protected — and stays perfectly inert
when memory is healthy.
"""

from __future__ import annotations

from types import SimpleNamespace

import mtplx.server.openai as srv

GIB = 1024**3
LIMIT = 96 * GIB
KV_PER_TOKEN = 24576
AUX_PER_TOKEN = 7872


class _Entry:
    def __init__(self, token_ids, session_id, nbytes):
        self.token_ids = tuple(int(t) for t in token_ids)
        self.session_id = session_id
        self.nbytes = int(nbytes)


class _Bank:
    def __init__(self, entries):
        self.entries = list(entries)
        self.cleared_sessions: list[str | None] = []
        self.shrink_calls: list[tuple[int, str, bool]] = []
        self.touched: list[str] = []
        self.probe_calls = 0

    @property
    def total_nbytes(self):
        return sum(entry.nbytes for entry in self.entries)

    def longest_prefix(self, token_ids):
        self.probe_calls += 1
        tokens = tuple(int(t) for t in token_ids)
        best = None
        for entry in self.entries:
            prefix = entry.token_ids
            if len(prefix) > len(tokens) or tokens[: len(prefix)] != prefix:
                continue
            if best is None or len(prefix) > len(best.token_ids):
                best = entry
        return best

    def longest_shared_prefix_tokens(self, token_ids, *, session_id=None):
        tokens = tuple(int(t) for t in token_ids)
        best = 0
        for entry in self.entries:
            if session_id is not None and entry.session_id != session_id:
                continue
            matched = 0
            for left, right in zip(tokens, entry.token_ids):
                if left != right:
                    break
                matched += 1
            best = max(best, matched)
        return best

    def clear(self, *, session_id=None):
        victims = [e for e in self.entries if e.session_id == session_id]
        self.entries = [e for e in self.entries if e.session_id != session_id]
        self.cleared_sessions.append(session_id)
        return len(victims)

    def touch_sessions(self, session_ids):
        self.touched.extend(session_ids)

    def shrink_to_bytes(self, target_bytes, *, reason="", protect_active=False):
        self.shrink_calls.append((int(target_bytes), reason, protect_active))
        evicted = 0
        while self.entries and self.total_nbytes > int(target_bytes):
            self.entries.pop(0)
            evicted += 1
        return evicted


def _state():
    return SimpleNamespace(
        metal_memory_caps={"memory_limit_bytes": LIMIT},
        memory_plan=SimpleNamespace(
            kv_bytes_per_token_effective=KV_PER_TOKEN,
            aux_bytes_per_token=AUX_PER_TOKEN,
            prefill_transient_bytes_per_token=0,
        ),
        dashboard=SimpleNamespace(),
    )


def _pin_live_stats(monkeypatch, *, active, cache):
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": int(active),
            "cache_memory_bytes": int(cache),
        },
    )


def _shed(state, prompt_ids, bank, session_id):
    return srv._prefill_admission_shed(
        state,
        prompt_ids=prompt_ids,
        session_bank=bank,
        session_id=session_id,
    )


def _vision_prompt(digest=123):
    prompt = [1] * 53_000 + [99] * 300 + [2] * 10_000
    splice = SimpleNamespace(
        image_pad_token_id=99, image_digests=[digest], pad_counts=[300]
    )
    return prompt, splice


def test_warm_vision_admission_does_not_evict_its_keyed_snapshot(monkeypatch):
    from mtplx.vision.splice import vision_bank_key_ids

    prompt, splice = _vision_prompt()
    entry = _Entry(vision_bank_key_ids(prompt, splice), "images", 3 * GIB)
    bank = _Bank([entry])
    _pin_live_stats(monkeypatch, active=93 * GIB, cache=2 * GIB)
    assert srv._prefill_admission_shed(
        _state(), prompt_ids=prompt + [3] * 100,
        session_bank=bank, session_id="images", vision_splice=splice,
    ) is None
    assert bank.entries == [entry]
    assert bank.shrink_calls == []


def test_changed_pixels_cannot_borrow_raw_live_frontier_for_admission(monkeypatch):
    from mtplx.vision.splice import vision_bank_key_ids

    prompt, old_splice = _vision_prompt()
    _, new_splice = _vision_prompt(456)
    bank = _Bank([_Entry(vision_bank_key_ids(prompt, old_splice), "old", 3 * GIB)])
    state = _state_with_live(prompt)
    _pin_live_stats(monkeypatch, active=93 * GIB, cache=2 * GIB)
    receipt = srv._prefill_admission_shed(
        state, prompt_ids=prompt, session_bank=bank,
        session_id="new", vision_splice=new_splice,
    )
    assert receipt is not None
    assert receipt["reusable_prefix_tokens"] <= 53_000
    assert receipt["miss_tokens"] >= 10_300
    assert receipt["reusable_prefix_mode"] != "live_session"


def test_vision_lineage_comes_from_matching_pixels_not_shared_text():
    from mtplx.vision.splice import vision_bank_key_ids

    prompt, splice = _vision_prompt()
    bank = _Bank([_Entry(vision_bank_key_ids(prompt, splice), "images", 3 * GIB)])
    assert srv._vision_bank_session_id(bank, prompt + [3], splice) == "images"
    _, changed = _vision_prompt(456)
    assert srv._vision_bank_session_id(bank, prompt, changed) is None
    bank.entries.append(_Entry(prompt[:53_000], "shared-text", GIB))
    assert srv._vision_bank_session_id(bank, prompt, changed) is None


class TestInertWhenHealthy:
    def test_healthy_memory_is_a_no_op_without_probing(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=60 * GIB, cache=2 * GIB)
        bank = _Bank([_Entry(range(100), "pi", 6 * GIB)])
        assert _shed(_state(), list(range(40_000)), bank, "pi") is None
        assert bank.probe_calls == 0
        assert bank.cleared_sessions == []
        assert bank.shrink_calls == []

    def test_short_prompt_never_triggers(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        bank = _Bank([_Entry(range(100), "pi", 6 * GIB)])
        assert _shed(_state(), list(range(1024)), bank, "pi") is None

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("MTPLX_PREFILL_ADMISSION_SHED", "0")
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        bank = _Bank([_Entry(range(100), "pi", 6 * GIB)])
        assert _shed(_state(), list(range(40_000)), bank, "pi") is None

    def test_no_metal_caps_is_inert(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        state = _state()
        state.metal_memory_caps = None
        assert _shed(state, list(range(40_000)), _Bank([]), "pi") is None


class TestIncidentShape:
    """The #415 timeline: cache-miss prefill + superseded resident snapshot."""

    def test_allocator_cache_reclaims_without_evicting_useful_sessions(self, monkeypatch):
        import mlx.core as mx

        live = {"ok": True, "active_memory_bytes": 85 * GIB,
                "cache_memory_bytes": 8 * GIB}
        monkeypatch.setattr(srv, "_mlx_memory_stats_live", lambda: dict(live))
        monkeypatch.setattr(mx, "clear_cache", lambda: live.update(cache_memory_bytes=0))
        bank = _Bank([_Entry(range(900, 1000), "pi", 6 * GIB)])
        receipt = _shed(_state(), list(range(40_000)), bank, "pi")
        assert receipt["cache_cleared"] is True
        assert bank.cleared_sessions == []
        assert bank.shrink_calls == []
        assert bank.total_nbytes == 6 * GIB

    def test_superseded_session_snapshot_released_first(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=93 * GIB, cache=2 * GIB)
        # The compaction rewrote the prefix: the session's banked snapshot
        # (prefix 900..999) can never match the new prompt (0..39999).
        superseded = _Entry(range(900, 1000), "pi", 6 * GIB)
        stranger = _Entry(range(500, 600), "other", 2 * GIB)
        bank = _Bank([stranger, superseded])
        receipt = _shed(_state(), list(range(40_000)), bank, "pi")
        assert receipt is not None
        assert receipt["action"] == "prefill_admission_shed"
        assert receipt["reusable_prefix_tokens"] == 0
        assert receipt["miss_tokens"] == 40_000
        assert receipt["superseded_session_entries_evicted"] == 1
        assert bank.cleared_sessions == ["pi"]
        assert receipt["cache_cleared"] is True

    def test_guard_event_recorded(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        state = _state()
        bank = _Bank([_Entry(range(900, 1000), "pi", 6 * GIB)])
        receipt = _shed(state, list(range(40_000)), bank, "pi")
        assert receipt is not None
        events = list(getattr(state.dashboard, "memory_guard_events", []))
        assert any(
            event.get("action") == "prefill_admission_shed" for event in events
        )

    def test_lru_shrink_protects_active_and_runs_after_superseded(
        self, monkeypatch
    ):
        # Deficit larger than the superseded snapshot alone: the LRU pass
        # must run with protect_active=True.
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=3 * GIB)
        superseded = _Entry(range(900, 1000), "pi", 1 * GIB)
        strangers = [
            _Entry(range(50_000 + i * 100, 50_000 + i * 100 + 50), f"idle-{i}", GIB)
            for i in range(4)
        ]
        bank = _Bank([*strangers, superseded])
        receipt = _shed(_state(), list(range(40_000)), bank, "pi")
        assert receipt is not None
        assert receipt["superseded_session_entries_evicted"] == 1
        assert len(bank.shrink_calls) == 1
        _target, reason, protect_active = bank.shrink_calls[0]
        assert reason == "prefill_admission"
        assert protect_active is True

    def test_reusable_prefix_is_pinned_not_cleared(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        prompt = list(range(40_000))
        # The session HAS a restorable prefix: entry matches prompt[:8192].
        restorable = _Entry(prompt[:8192], "pi", 3 * GIB)
        bank = _Bank([restorable])
        receipt = _shed(_state(), prompt, bank, "pi")
        assert receipt is not None
        assert receipt["reusable_prefix_tokens"] == 8192
        assert bank.cleared_sessions == []
        assert bank.touched == ["pi"]
        assert "superseded_session_entries_evicted" not in receipt

    def test_no_bank_still_sheds_allocator_cache(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        receipt = _shed(_state(), list(range(40_000)), None, None)
        assert receipt is not None
        assert receipt["cache_cleared"] is True
        assert "bank_bytes_before" not in receipt

    def test_never_raises(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)

        class _ExplodingBank(_Bank):
            def longest_prefix(self, token_ids):  # noqa: ARG002
                raise RuntimeError("bank exploded")

            @property
            def total_nbytes(self):
                raise RuntimeError("bank exploded")

        receipt = _shed(
            _state(), list(range(40_000)), _ExplodingBank([]), "pi"
        )
        # Probe failure degrades to miss==prompt; bank steps report the
        # error but the shed still completes with the cache clear.
        assert receipt is not None
        assert receipt["cache_cleared"] is True
        assert "bank_error" in receipt


class TestBlockPrefixRestorableEntries:
    """The 2.11 release-gate failure (tool_result_forced, 128 GB Flash-Next).

    The forced tool round banks prompt + transient sentinel + completion; the
    next prompt shares 41k tokens with that entry but the entry is not an
    exact prefix of it. The restore path serves it by block prefix. The shed
    must estimate reuse the same way, or it evicts the session's only entry
    as "superseded" and the turn re-prefills cold (54 s measured).
    """

    def test_block_restorable_follow_up_leaves_the_guard_inert(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        shared = list(range(41_000))
        banked = shared + [999_001, 999_002] + list(range(70_000, 70_020))
        prompt = shared + list(range(80_000, 80_900))
        entry = _Entry(banked, "gate", 2 * GIB)
        bank = _Bank([entry])
        # The gate's exact shape: 41k shared, a 900-token tail. Under the
        # old exact-only estimate this read as a 41,900-token miss and the
        # entry was cleared; the block-aware miss is 940 tokens, under the
        # guard's own floor, so it does not act at all.
        assert _shed(_state(), prompt, bank, "gate") is None
        assert bank.cleared_sessions == []
        assert bank.shrink_calls == []
        assert entry in bank.entries

    def test_block_restorable_entry_is_pinned_not_cleared(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        shared = list(range(41_000))
        banked = shared + [999_001, 999_002] + list(range(70_000, 70_020))
        # A 5,000-token tail (a large tool result) is a real miss: the guard
        # acts, but on the block estimate, and it pins the entry.
        prompt = shared + list(range(80_000, 85_000))
        bank = _Bank([_Entry(banked, "gate", 2 * GIB)])
        receipt = _shed(_state(), prompt, bank, "gate")
        assert receipt is not None
        # 41,000 shared tokens rewound to the 256-token block edge.
        assert receipt["reusable_prefix_tokens"] == 40_960
        assert receipt["reusable_prefix_mode"] == "block_prefix"
        assert receipt["miss_tokens"] == len(prompt) - 40_960
        assert bank.cleared_sessions == []
        assert bank.touched == ["gate"]
        assert "superseded_session_entries_evicted" not in receipt
        assert all(protect for _t, _r, protect in bank.shrink_calls)

    def test_exact_prefix_still_wins_over_block_estimate(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        prompt = list(range(40_000))
        exact = _Entry(prompt[:8_200], "pi", 3 * GIB)
        bank = _Bank([exact])
        receipt = _shed(_state(), prompt, bank, "pi")
        assert receipt is not None
        # 8,200 exact beats the 8,192 block-aligned figure.
        assert receipt["reusable_prefix_tokens"] == 8_200
        assert receipt["reusable_prefix_mode"] == "exact"

    def test_overlap_under_the_restore_floor_is_still_superseded(
        self, monkeypatch
    ):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        prompt = list(range(40_000))
        # A compaction that kept only the first 300 tokens (system prompt)
        # shares less than a restorable block: the entry is superseded.
        stale = _Entry(prompt[:300] + list(range(90_000, 91_000)), "pi", 6 * GIB)
        bank = _Bank([stale])
        receipt = _shed(_state(), prompt, bank, "pi")
        assert receipt is not None
        assert receipt["reusable_prefix_tokens"] == 0
        assert receipt["reusable_prefix_mode"] == "none"
        assert bank.cleared_sessions == ["pi"]

    def test_bank_without_the_shared_prefix_probe_keeps_the_exact_estimate(
        self, monkeypatch
    ):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)

        class _ExactOnlyBank(_Bank):
            longest_shared_prefix_tokens = None

        shared = list(range(41_000))
        banked = shared + [999_001]
        prompt = shared + list(range(80_000, 80_900))
        bank = _ExactOnlyBank([_Entry(banked, "gate", 2 * GIB)])
        receipt = _shed(_state(), prompt, bank, "gate")
        assert receipt is not None
        assert receipt["reusable_prefix_tokens"] == 0
        assert bank.cleared_sessions == ["gate"]


class _LiveSession:
    def __init__(self, committed):
        self.committed_token_ids = tuple(int(t) for t in committed)


def _state_with_live(committed, *, near=None, common=None):
    """Fake EngineSessionManager exposing the resolution ladder: exact
    containment, then (near, matched) / (common, matched) tuples."""
    state = _state()
    live = _LiveSession(committed)

    def longest_prefix_session(token_ids):
        tokens = tuple(int(t) for t in token_ids)
        prefix = live.committed_token_ids
        if prefix and len(prefix) <= len(tokens) and tokens[: len(prefix)] == prefix:
            return live
        return None

    state.sessions = SimpleNamespace(
        longest_prefix_session=longest_prefix_session,
        pending_near_prefix_session=lambda token_ids: near or (None, 0),
        best_common_prefix_session=lambda token_ids: common or (None, 0),
    )
    return state


class TestLiveSessionPrefix:
    """#447, receipt 2: a warm 212k session whose newest turns were never
    banked (snapshot commits refused on retokenized-history mismatch) read
    as a full miss, was cleared as "superseded", and every client retry was
    a 211,807-token cold miss. The engine's live sessions are the state the
    request actually reuses; the shed must ask them too."""

    def test_live_prefix_under_the_miss_floor_leaves_the_guard_inert(
        self, monkeypatch
    ):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        state = _state_with_live(prompt[:97_000])
        entry = _Entry(list(range(500_000, 500_040)), "warm", 4 * GIB)
        bank = _Bank([entry])
        assert _shed(state, prompt, bank, "warm") is None
        assert bank.cleared_sessions == []
        assert entry in bank.entries

    def test_live_prefix_pins_instead_of_clearing(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        state = _state_with_live(prompt[:60_000])
        entry = _Entry(list(range(500_000, 500_040)), "warm", 4 * GIB)
        bank = _Bank([entry])
        receipt = _shed(state, prompt, bank, "warm")
        assert receipt is not None
        assert receipt["reusable_prefix_mode"] == "live_session"
        assert receipt["reusable_prefix_tokens"] == 60_000
        assert receipt["miss_tokens"] == 40_000
        assert bank.cleared_sessions == []
        assert "warm" in bank.touched

    def test_live_prefix_yields_to_the_export(self, monkeypatch):
        monkeypatch.setenv("MTPLX_PREFILL_ADMISSION_LIVE_PREFIX", "0")
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        state = _state_with_live(prompt[:60_000])
        entry = _Entry(list(range(500_000, 500_040)), "warm", 4 * GIB)
        bank = _Bank([entry])
        receipt = _shed(state, prompt, bank, "warm")
        assert receipt is not None
        assert receipt["reusable_prefix_mode"] == "none"
        assert bank.cleared_sessions == ["warm"]

    def test_state_without_sessions_keeps_the_bank_estimate(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        entry = _Entry(list(range(500_000, 500_040)), "warm", 4 * GIB)
        bank = _Bank([entry])
        receipt = _shed(_state(), prompt, bank, "warm")
        assert receipt is not None
        assert receipt["reusable_prefix_mode"] == "none"
        assert bank.cleared_sessions == ["warm"]


class _ChainBank(_Bank):
    """Active-protecting bank: shrink_to_bytes honors protect_active the way
    the real bank does (touched sessions are the active set), and the chain
    walk keeps each session's longest entry plus the protect_tokens match."""

    def __init__(self, entries):
        super().__init__(entries)
        self.chain_calls: list[tuple[int, str]] = []

    def shrink_to_bytes(self, target_bytes, *, reason="", protect_active=False):
        self.shrink_calls.append((int(target_bytes), reason, protect_active))
        evicted = 0
        active = set(self.touched)
        while self.entries and self.total_nbytes > int(target_bytes):
            candidates = [
                e
                for e in self.entries
                if not (protect_active and e.session_id in active)
            ]
            if not candidates:
                break
            self.entries.remove(candidates[0])
            evicted += 1
        return evicted

    def shrink_for_admission(self, target_bytes, *, protect_tokens=None, reason=""):
        self.chain_calls.append((int(target_bytes), reason))
        protected = set()
        if protect_tokens:
            tokens = tuple(int(t) for t in protect_tokens)
            best, best_common = None, 0
            for e in self.entries:
                common = 0
                for left, right in zip(tokens, e.token_ids):
                    if left != right:
                        break
                    common += 1
                if common > best_common:
                    best, best_common = e, common
            if best is not None:
                protected.add(id(best))
        non_terminal = 0
        while self.entries and self.total_nbytes > int(target_bytes):
            terminal = {}
            for e in self.entries:
                key = e.session_id or ""
                terminal[key] = max(terminal.get(key, -1), len(e.token_ids))
            candidates = [
                e
                for e in self.entries
                if id(e) not in protected
                and len(e.token_ids) < terminal.get(e.session_id or "", -1)
            ]
            if not candidates:
                break
            self.entries.remove(candidates[0])
            non_terminal += 1
        terminals = 0
        while self.entries and self.total_nbytes > int(target_bytes):
            candidates = [e for e in self.entries if id(e) not in protected]
            if not candidates:
                break
            self.entries.remove(candidates[0])
            terminals += 1
        return non_terminal, terminals


class TestChainPrefixEscalation:
    """#447, receipt 1: 12.6 GiB of a deep session's own sibling snapshots
    were active-protected, the LRU pass evicted nothing, and a warm turn
    with a 23k miss died on the sustained-pressure 507."""

    def _fixture(self):
        prompt = list(range(100_000))
        exact = _Entry(prompt[:50_000], "deep", 2 * GIB)
        terminal = _Entry(prompt[:49_000] + list(range(700_000, 711_000)), "deep", 3 * GIB)
        sibling = _Entry(prompt[:30_000] + list(range(800_000, 800_400)), "deep", 6 * GIB)
        return prompt, exact, terminal, sibling

    def test_sibling_snapshots_are_walked_when_lru_is_protected(
        self, monkeypatch
    ):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt, exact, terminal, sibling = self._fixture()
        bank = _ChainBank([exact, terminal, sibling])
        receipt = _shed(_state(), prompt, bank, "deep")
        assert receipt is not None
        assert receipt["reusable_prefix_tokens"] == 50_000
        assert receipt["lru_entries_evicted"] == 0
        assert receipt["chain_entries_evicted"] == 1
        assert receipt["terminal_entries_evicted"] == 0
        assert sibling not in bank.entries
        assert exact in bank.entries
        assert terminal in bank.entries
        assert receipt["bank_bytes_after"] == 5 * GIB

    def test_chain_walk_yields_to_the_export(self, monkeypatch):
        monkeypatch.setenv("MTPLX_PREFILL_ADMISSION_CHAIN_SHED", "0")
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt, exact, terminal, sibling = self._fixture()
        bank = _ChainBank([exact, terminal, sibling])
        receipt = _shed(_state(), prompt, bank, "deep")
        assert receipt is not None
        assert "chain_entries_evicted" not in receipt
        assert bank.chain_calls == []
        assert sibling in bank.entries

    def test_bank_without_the_method_is_tolerated(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt, exact, terminal, sibling = self._fixture()
        bank = _Bank([exact, terminal, sibling])
        bank.shrink_to_bytes = lambda *a, **k: 0
        receipt = _shed(_state(), prompt, bank, "deep")
        assert receipt is not None
        assert "chain_entries_evicted" not in receipt


class TestLiveSessionLadder:
    """The resolution ladder: a mutated history (retokenized turns, the
    #446 fork shape) misses exact containment but is still served by the
    pending-near-prefix and best-common lookups; the shed's estimate must
    follow the same ladder or it reads a warm engine as a full miss."""

    def test_pending_near_prefix_serves_the_estimate(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        state = _state_with_live((), near=(_LiveSession(prompt[:60_000]), 60_000))
        bank = _Bank([])
        receipt = _shed(state, prompt, bank, "warm")
        assert receipt is not None
        assert receipt["reusable_prefix_mode"] == "live_session"
        expected = srv._block_restorable_prefix_tokens(60_000)
        assert receipt["reusable_prefix_tokens"] == expected
        assert bank.cleared_sessions == []

    def test_best_common_serves_the_estimate(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        state = _state_with_live((), common=(_LiveSession(prompt[:60_000]), 60_000))
        bank = _Bank([])
        receipt = _shed(state, prompt, bank, "warm")
        assert receipt is not None
        assert receipt["reusable_prefix_mode"] == "live_session"
        assert receipt["reusable_prefix_tokens"] == srv._block_restorable_prefix_tokens(60_000)

    def test_exact_wins_over_the_tolerant_rungs(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        state = _state_with_live(
            prompt[:60_000], common=(_LiveSession(prompt[:10_000]), 10_000)
        )
        bank = _Bank([])
        receipt = _shed(state, prompt, bank, "warm")
        assert receipt is not None
        assert receipt["reusable_prefix_tokens"] == 60_000

    def test_ladder_exceptions_never_cost_the_request(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        state = _state()

        def boom(token_ids):
            raise RuntimeError("probe blew up")

        state.sessions = SimpleNamespace(
            longest_prefix_session=boom,
            pending_near_prefix_session=boom,
            best_common_prefix_session=boom,
        )
        bank = _Bank([])
        receipt = _shed(state, prompt, bank, "warm")
        assert receipt is not None
        assert receipt["reusable_prefix_mode"] == "none"


class TestTerminalPhase:
    """Phase 2 (#447): when non-terminal walking cannot cover the deficit
    (every fork is its own session's terminal — the post-#446 shape), the
    walk takes remaining entries rather than letting the request die, but
    never the entry the imminent prompt restores from."""

    def test_other_terminals_fall_when_siblings_are_not_enough(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=92 * GIB, cache=GIB // 2)
        prompt = list(range(100_000))
        exact = _Entry(prompt[:50_000], "deep", 2 * GIB)
        fork = _Entry(prompt[:20_000] + list(range(900_000, 950_000)), "fork-session", 6 * GIB)
        bank = _ChainBank([exact, fork])
        bank.touched.append("fork-session")  # recently active: LRU pass skips it
        receipt = _shed(_state(), prompt, bank, "deep")
        assert receipt is not None
        assert receipt["chain_entries_evicted"] == 0
        assert receipt["terminal_entries_evicted"] == 1
        assert fork not in bank.entries
        assert exact in bank.entries


class TestRefusalPastTheLimit:
    """#450: a projection still over the hard limit after every reclamation
    step is refused before prefill (structured 507), never admitted after a
    cache clear. On the reporter's 128 GB Mac the admitted request pushed
    wired memory past the kernel's limit and panicked the machine."""

    @staticmethod
    def _per_token():
        return KV_PER_TOKEN + AUX_PER_TOKEN

    @staticmethod
    def _transients():
        from mtplx.memory_plan import RUNTIME_TRANSIENTS_BYTES

        return int(RUNTIME_TRANSIENTS_BYTES)

    def test_refuses_when_reclamation_cannot_bring_the_projection_under_the_limit(
        self, monkeypatch
    ):
        miss = 60_000
        # Lands 1 GiB over the limit with nothing left to reclaim.
        active = LIMIT + GIB - miss * self._per_token() - self._transients()
        _pin_live_stats(monkeypatch, active=active, cache=0)
        receipt = _shed(_state(), list(range(miss)), _Bank([]), "pi")
        assert receipt is not None
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "projected_over_limit_after_reclamation"
        assert receipt["projected_bytes_after"] > LIMIT

    def test_admits_between_the_warning_line_and_the_limit(self, monkeypatch):
        miss = 40_000
        # Crosses the 0.97 WARNING line by 0.5 GiB but stays 2.4 GiB under the limit.
        active = int(LIMIT * 0.97) + GIB // 2 - miss * self._per_token() - self._transients()
        _pin_live_stats(monkeypatch, active=active, cache=0)
        receipt = _shed(_state(), list(range(miss)), _Bank([]), "pi")
        assert receipt is not None
        assert receipt["projected_bytes_after"] <= LIMIT
        assert "refused" not in receipt

    def test_allow_swap_keeps_the_operator_choice(self, monkeypatch):
        miss = 60_000
        active = LIMIT + GIB - miss * self._per_token() - self._transients()
        _pin_live_stats(monkeypatch, active=active, cache=0)
        state = _state()
        state.allow_swap = True
        receipt = _shed(state, list(range(miss)), _Bank([]), "pi")
        assert receipt is not None
        assert "refused" not in receipt

    def test_refusal_is_a_structured_507_with_the_numbers(self):
        exc = srv._prefill_admission_refusal(
            _state(),
            {
                "limit_bytes": LIMIT,
                "projected_bytes_after": LIMIT + 2 * GIB,
                "prompt_tokens": 136_549,
                "miss_tokens": 136_549,
            },
        )
        assert exc.status_code == 507
        assert "98.0 GiB" in str(exc.detail)
        assert "96.0 GiB" in str(exc.detail)
        assert "refused before prefill" in str(exc.detail)
        assert "136549" in str(exc.detail)
