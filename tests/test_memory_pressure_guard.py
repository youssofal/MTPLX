"""Issue #144: shed cache weight under system memory pressure.

The bank held its full budget while a 64 GB Mac swapped 60 GB. The guard
loop shrinks the bank to half budget on WARNING and empties it on
CRITICAL; `shrink_to_bytes` is the bank-side primitive.
"""

from __future__ import annotations

import asyncio
from threading import Lock
from types import SimpleNamespace

import mtplx.server.openai as srv


class FakeBank:
    def __init__(self, total, max_bytes):
        self.total_nbytes = total
        self.max_bytes = max_bytes
        self.calls = []
        self.protect_active_calls = []

    def shrink_to_bytes(self, target, *, reason, protect_active=False):
        self.calls.append((target, reason))
        self.protect_active_calls.append(protect_active)
        evicted = 1 if self.total_nbytes > target else 0
        self.total_nbytes = min(self.total_nbytes, target)
        return evicted


def make_state(bank):
    return SimpleNamespace(
        lock=Lock(),
        foreground_count=lambda: 0,
        model_scheduler=SimpleNamespace(run_idle_maintenance=lambda fn: (fn(), True)[1]),
        sessions=SimpleNamespace(bank=bank),
        dashboard=SimpleNamespace(
            last_memory_pressure_level=0,
            in_flight=SimpleNamespace(count=lambda: 0, session_ids=lambda: []),
        ),
    )


def run_one_tick(state, level, monkeypatch):
    monkeypatch.setattr(srv, "_memory_pressure_level", lambda: level)

    async def one_tick():
        task = asyncio.ensure_future(
            srv._memory_pressure_loop(state, interval_s=3600)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(one_tick())


def test_warning_shrinks_bank_to_half_budget(monkeypatch):
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    run_one_tick(state, level=2, monkeypatch=monkeypatch)
    assert bank.calls == [(4 << 30, "memory_pressure_warning")]
    assert state.dashboard.last_memory_pressure_level == 2


def test_critical_empties_bank(monkeypatch):
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    run_one_tick(state, level=4, monkeypatch=monkeypatch)
    assert bank.calls == [(0, "memory_pressure_critical")]


def test_normal_level_touches_nothing(monkeypatch):
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    run_one_tick(state, level=1, monkeypatch=monkeypatch)
    assert bank.calls == []
    assert state.dashboard.last_memory_pressure_level == 1


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_MEMORY_PRESSURE_GUARD", "0")
    assert not srv._memory_pressure_guard_enabled()
    monkeypatch.delenv("MTPLX_MEMORY_PRESSURE_GUARD")
    assert srv._memory_pressure_guard_enabled()


def test_guard_core_sustained_warning_rearms_after_interval():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=1000.0, busy=False) is True  # rising edge acts
    assert g.decide(2, now=1010.0, busy=False) is False  # no per-tick re-trim
    assert g.decide(2, now=1100.0, busy=False) is False  # inside 120s window
    assert g.decide(2, now=1125.0, busy=False) is True  # re-armed


def test_guard_core_warning_defers_while_busy_then_acts():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=True) is False
    assert g.decide(2, now=30.0, busy=True) is False
    assert g.decide(2, now=61.0, busy=True) is True  # defer window expired


def test_guard_core_warning_acts_when_engine_goes_idle():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=True) is False
    assert g.decide(2, now=10.0, busy=False) is True


def test_guard_core_critical_never_defers():
    g = srv._MemoryPressureGuard()
    assert g.decide(4, now=0.0, busy=True) is True


def test_guard_core_flapping_cannot_retrigger_fast():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=False) is True
    assert g.decide(1, now=10.0, busy=False) is False
    assert g.decide(2, now=20.0, busy=False) is False  # edge inside 30s spacing
    assert g.decide(1, now=30.0, busy=False) is False
    assert g.decide(2, now=40.0, busy=False) is True  # edge after spacing


def test_guard_core_escalation_to_critical_acts_immediately():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=False) is True
    assert g.decide(4, now=10.0, busy=True) is True  # 2->4 edge, no spacing gate


def test_guard_core_recovery_clears_pending_action():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=True) is False  # owed, deferred
    assert g.decide(1, now=10.0, busy=False) is False  # recovered: owed cleared
    assert g.decide(1, now=200.0, busy=False) is False


class FakeDynamicBank(FakeBank):
    """FakeBank + the dynamic-ceiling interface (memory governor, #305)."""

    def __init__(self, total, max_bytes, ceiling):
        super().__init__(total, max_bytes)
        self.dynamic_ceiling_fn = lambda: ceiling
        self._ceiling = ceiling

    def effective_max_bytes(self):
        return min(self.max_bytes, self._ceiling)


def test_dynamic_ceiling_enforced_every_tick_even_at_level_normal(monkeypatch):
    # At a safe point, enforce the existing ceiling without waiting for a
    # macOS pressure edge. Recently active chains retain their protection.
    bank = FakeDynamicBank(total=8 << 30, max_bytes=8 << 30, ceiling=2 << 30)
    state = make_state(bank)
    run_one_tick(state, level=1, monkeypatch=monkeypatch)
    assert bank.calls == [(2 << 30, "dynamic_ceiling")]
    assert bank.protect_active_calls == [True]
    events = list(state.dashboard.memory_guard_events)
    assert events and events[-1]["action"] == "dynamic_ceiling"


def test_dynamic_ceiling_tick_restamps_sessions_inside_safe_point(monkeypatch):
    # No bank mutation, including activity-pin writes, precedes the gate.
    bank = FakeDynamicBank(total=8 << 30, max_bytes=8 << 30, ceiling=2 << 30)
    bank.touched = []
    bank.touch_sessions = lambda ids: bank.touched.extend(ids)
    state = make_state(bank)
    state.dashboard.in_flight = SimpleNamespace(
        count=lambda: 0, session_ids=lambda: ["ses_live"]
    )
    run_one_tick(state, level=1, monkeypatch=monkeypatch)
    assert bank.touched == ["ses_live"]
    assert bank.calls == [(2 << 30, "dynamic_ceiling")]


def test_dynamic_ceiling_within_slack_is_left_alone(monkeypatch):
    # 128 MiB over the ceiling is inside the 256 MiB slack: no churn.
    bank = FakeDynamicBank(
        total=(2 << 30) + (128 << 20), max_bytes=8 << 30, ceiling=2 << 30
    )
    state = make_state(bank)
    run_one_tick(state, level=1, monkeypatch=monkeypatch)
    assert bank.calls == []


def test_allocator_pressure_escalates_before_macos(monkeypatch):
    # macOS says normal, but the allocator sits at 99% of the Metal limit:
    # the guard must act as WARNING now, not minutes into the swap.
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    state.metal_memory_caps = {"memory_limit_bytes": 100 << 30}
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": 95 << 30,
            "cache_memory_bytes": 4 << 30,
        },
    )
    run_one_tick(state, level=1, monkeypatch=monkeypatch)
    assert bank.calls == [(4 << 30, "memory_pressure_warning")]
    # Real pressure keeps take-anything semantics — no active protection.
    assert bank.protect_active_calls == [False]
    assert state.dashboard.last_memory_pressure_level == 2
    events = list(state.dashboard.memory_guard_events)
    assert events and events[-1]["level_source"] == "allocator"
    # The dashboard carries the attribution every tick (not only on guard
    # actions) so the app banner can name the culprit: engine footprint vs
    # an external process's allocation storm (2026-08-28 A/B receipt).
    assert state.dashboard.last_memory_pressure_source == "allocator"
    assert abs(state.dashboard.last_allocator_fraction - 0.99) < 1e-6


def test_macos_sourced_warning_stamps_external_attribution(monkeypatch):
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    state.metal_memory_caps = {"memory_limit_bytes": 100 << 30}
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": 10 << 30,
            "cache_memory_bytes": 1 << 30,
        },
    )
    run_one_tick(state, level=2, monkeypatch=monkeypatch)
    assert state.dashboard.last_memory_pressure_level == 2
    # The allocator probe read a genuinely healthy fraction, so the
    # external attribution ("another process") is attested, not assumed.
    assert state.dashboard.last_memory_pressure_source == "macos"


def test_macos_warning_without_allocator_reading_stamps_unknown(monkeypatch):
    # No Metal caps: the allocator probe bails (fraction 0.0). The old
    # code still blamed "macos", and the app copy then asserted the
    # engine footprint was steady with nothing to back it. Now the
    # source is "unknown" and the banner stays neutral.
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    run_one_tick(state, level=2, monkeypatch=monkeypatch)
    assert state.dashboard.last_memory_pressure_level == 2
    assert state.dashboard.last_memory_pressure_source == "unknown"


def test_tied_warning_blames_allocator(monkeypatch):
    # macOS WARNING and the allocator at 99% in the same tick: the old
    # strict > kept the "macos" attribution (banner: "another process...
    # the engine's own footprint is steady"), inverting the blame while
    # the engine itself sat at the wall. Ties at WARNING+ now name the
    # allocator.
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    state.metal_memory_caps = {"memory_limit_bytes": 100 << 30}
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": 95 << 30,
            "cache_memory_bytes": 4 << 30,
        },
    )
    run_one_tick(state, level=2, monkeypatch=monkeypatch)
    assert state.dashboard.last_memory_pressure_level == 2
    assert state.dashboard.last_memory_pressure_source == "allocator"


def test_allocator_pressure_below_threshold_stays_normal():
    state = make_state(FakeBank(total=0, max_bytes=1 << 30))
    state.metal_memory_caps = {"memory_limit_bytes": 100 << 30}
    level, fraction = srv._allocator_pressure_level(state)
    # _mlx_memory_stats_live runs for real here; whatever it reports on a
    # quiet test process is far below 97% of a 100 GiB limit.
    assert level == 1
    assert fraction < 0.97


def test_is_allocation_failure_classifier():
    assert srv._is_allocation_failure(MemoryError("x"))
    assert srv._is_allocation_failure(
        RuntimeError(
            "[metal::malloc] Attempting to allocate 21474836480 bytes which is "
            "greater than the maximum allowed buffer size of 17179869184 bytes."
        )
    )
    assert srv._is_allocation_failure(
        RuntimeError(
            "[METAL] Command buffer execution failed: Insufficient Memory "
            "(kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        )
    )
    assert not srv._is_allocation_failure(RuntimeError("shape mismatch"))
    assert not srv._is_allocation_failure(ValueError("failed to allocate"))


def test_allocation_failure_becomes_shed_plus_507():
    bank = FakeDynamicBank(total=8 << 30, max_bytes=8 << 30, ceiling=8 << 30)
    state = make_state(bank)
    state.memory_plan = SimpleNamespace(
        context_overcommitted=True,
        context_window_resolved=262_144,
        context_window_fit=196_608,
    )
    exc = srv._allocation_failure_http_exception(
        state, RuntimeError("Insufficient Memory")
    )
    assert exc.status_code == 507
    assert "insufficient memory" in str(exc.detail)
    # The advice names the user's own overcommit, not a generic tip.
    assert "196608" in str(exc.detail).replace(",", "")
    # The shed ran: bank asked to drop to half its effective budget.
    assert bank.calls and bank.calls[0] == (4 << 30, "allocation_failure")
    events = list(state.dashboard.memory_guard_events)
    assert events and events[-1]["action"] == "allocation_failure_shed"


def test_bank_shrink_to_bytes_evicts_lru_first():
    from mtplx.session_bank import SessionBank

    bank = SessionBank(max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    # Fabricate three entries with staggered ages via the internal table:
    # shrink must drop the oldest-accessed first.
    from mtplx.session_bank import SessionBankEntry, CacheSnapshot

    def entry(name, last_access, nbytes):
        return SessionBankEntry(
            token_ids=(hash(name) % 1000, 2, 3),
            token_hash=name,
            model_path="/m",
            mtp_enabled=False,
            hidden_variant=None,
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            cache_ref=None,
            nbytes=nbytes,
            session_id=name,
            last_access_s=last_access,
        )

    # The bank's invariant: entries are keyed by their token_ids tuple.
    # (This test originally used string keys, which made _evict_entry's
    # pop-by-token_ids miss forever — shrink_to_bytes spun allocating
    # eviction-log records until the machine ran out of RAM.)
    fabricated = [
        entry("old", last_access=1.0, nbytes=400),
        entry("mid", last_access=2.0, nbytes=400),
        entry("new", last_access=3.0, nbytes=400),
    ]
    bank._entries = {e.token_ids: e for e in fabricated}
    evicted = bank.shrink_to_bytes(500)
    assert evicted == 2
    assert [e.session_id for e in bank._entries.values()] == ["new"]


def test_bank_shrink_protect_active_never_evicts_the_live_session():
    """Dynamic-ceiling semantics: idle sessions yield, the in-flight
    session's prefix chain survives even when the bank stays above target
    (2026-08-28 receipt: a 93k OpenCode session's own entries were walked
    to 0 bytes mid-request; the next turns re-prefilled for 54-57 s)."""
    import time as _time

    from mtplx.session_bank import CacheSnapshot, SessionBank, SessionBankEntry

    bank = SessionBank(max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30)

    def entry(name, session, last_access, nbytes):
        return SessionBankEntry(
            token_ids=(hash(name) % 1000, 2, 3),
            token_hash=name,
            model_path="/m",
            mtp_enabled=False,
            hidden_variant=None,
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            cache_ref=None,
            nbytes=nbytes,
            session_id=session,
            last_access_s=last_access,
        )

    fabricated = [
        entry("live-a", "ses_live", last_access=1.0, nbytes=400),
        entry("live-b", "ses_live", last_access=2.0, nbytes=400),
        entry("idle-a", "ses_idle", last_access=3.0, nbytes=400),
    ]
    bank._entries = {e.token_ids: e for e in fabricated}
    bank._session_last_active = {"ses_live": _time.monotonic()}

    evicted = bank.shrink_to_bytes(0, reason="dynamic_ceiling", protect_active=True)

    # Only the idle session's entry went; the live chain survives above
    # target, so the bank ends over the ceiling by design.
    assert evicted == 1
    remaining = {e.session_id for e in bank._entries.values()}
    assert remaining == {"ses_live"}
    assert bank.total_nbytes == 800

    # Real pressure (protect_active omitted) still takes everything.
    evicted = bank.shrink_to_bytes(0, reason="memory_pressure_critical")
    assert evicted == 2
    assert bank.total_nbytes == 0
