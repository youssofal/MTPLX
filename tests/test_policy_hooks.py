from __future__ import annotations

import time

import pytest

from mtplx.policy_hooks import (
    FailureMode,
    HookPhase,
    HookResult,
    PolicyBus,
    PolicyContext,
    PolicyHookConfig,
    PolicyValidationError,
)


def test_hooks_run_in_priority_order_and_rewrite_isolated_values():
    bus = PolicyBus()
    seen = []

    def second(value, _context):
        seen.append(("second", value["n"]))
        return HookResult.annotate({"second": True})

    def first(value, _context):
        seen.append(("first", value["n"]))
        value["n"] = 999  # Mutating the isolated input cannot escape unless rewritten.
        return HookResult.rewrite({"n": 2}, annotations={"first": True})

    bus.register("second", second, phases=[HookPhase.REQUEST], priority=20)
    bus.register("first", first, phases=[HookPhase.REQUEST], priority=10)
    original = {"n": 1}
    outcome = bus.execute(HookPhase.REQUEST, original)

    assert original == {"n": 1}
    assert seen == [("first", 1), ("second", 2)]
    assert outcome.allowed is True
    assert outcome.rewritten is True
    assert outcome.value == {"n": 2}
    assert outcome.annotations == {"first": True, "second": True}


def test_rejection_stops_later_hooks():
    bus = PolicyBus()
    called = []
    bus.register(
        "reject",
        lambda _value, _ctx: HookResult.reject("blocked", status_code=429),
        phases=[HookPhase.REQUEST],
        priority=1,
    )
    bus.register(
        "later",
        lambda _value, _ctx: called.append(True),
        phases=[HookPhase.REQUEST],
        priority=2,
    )
    outcome = bus.execute(HookPhase.REQUEST, {"model": "x"})
    assert outcome.allowed is False
    assert outcome.reason == "blocked"
    assert outcome.status_code == 429
    assert called == []


def test_timeout_returns_quickly_and_respects_failure_mode():
    def slow(_value, _context):
        time.sleep(0.30)
        return HookResult.allow()

    open_bus = PolicyBus(PolicyHookConfig(default_timeout_s=0.02))
    open_bus.register("slow", slow, phases=[HookPhase.REQUEST])
    started = time.monotonic()
    open_outcome = open_bus.execute(HookPhase.REQUEST, {})
    assert time.monotonic() - started < 0.20
    assert open_outcome.allowed is True
    assert open_outcome.timed_out_hooks == ("slow",)

    closed_bus = PolicyBus(PolicyHookConfig(default_timeout_s=0.02))
    closed_bus.register(
        "slow",
        slow,
        phases=[HookPhase.REQUEST],
        failure_mode=FailureMode.CLOSED,
    )
    closed_outcome = closed_bus.execute(HookPhase.REQUEST, {})
    assert closed_outcome.allowed is False
    assert closed_outcome.status_code == 503
    assert closed_outcome.reason == "policy_timeout:slow"


def test_callback_failure_is_sanitized_and_fail_open_by_default():
    def broken(_value, _context):
        raise RuntimeError("contains a private value")

    bus = PolicyBus()
    bus.register("broken", broken, phases=[HookPhase.ERROR])
    outcome = bus.execute(
        HookPhase.ERROR,
        {"error_type": "RuntimeError"},
        context=PolicyContext(phase=HookPhase.ERROR),
    )
    assert outcome.allowed is True
    assert outcome.failed_hooks == ("broken",)
    assert "private value" not in outcome.reason
    assert bus.snapshot()["failures"] == 1


def test_rewrite_size_and_input_size_are_bounded():
    bus = PolicyBus(PolicyHookConfig(maximum_value_bytes=32))
    with pytest.raises(PolicyValidationError):
        bus.execute(HookPhase.REQUEST, {"value": "x" * 100})

    bus.register(
        "large",
        lambda _value, _ctx: HookResult.rewrite({"value": "x" * 100}),
        phases=[HookPhase.REQUEST],
        failure_mode=FailureMode.CLOSED,
    )
    outcome = bus.execute(HookPhase.REQUEST, {})
    assert outcome.allowed is False
    assert outcome.reason == "rewrite_too_large:large"


def test_registry_is_bounded_and_phase_counts_are_truthful():
    bus = PolicyBus(PolicyHookConfig(maximum_hooks=1))
    bus.register("one", lambda _v, _c: None, phases=[HookPhase.RESPONSE])
    with pytest.raises(PolicyValidationError):
        bus.register("two", lambda _v, _c: None)
    snapshot = bus.snapshot()
    assert snapshot["registered_hooks"] == 1
    assert snapshot["hooks_by_phase"]["response"] == 1
    assert snapshot["hooks_by_phase"]["request"] == 0
    assert snapshot["external_policy_dependency"] is False


def test_policy_executor_is_lazy_and_worker_count_is_bounded():
    import threading

    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def slow(_value, _context):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.12)
        finally:
            with state_lock:
                active -= 1
        return HookResult.allow()

    bus = PolicyBus(
        PolicyHookConfig(
            default_timeout_s=0.005,
            maximum_workers=2,
            maximum_pending_tasks=2,
        )
    )
    assert bus.snapshot()["executor_started"] is False
    bus.register("slow", slow, phases=[HookPhase.REQUEST])
    assert bus.snapshot()["executor_started"] is False
    started = time.monotonic()
    for _ in range(12):
        bus.execute(HookPhase.REQUEST, {})
    elapsed = time.monotonic() - started
    assert bus.snapshot()["executor_started"] is True
    bus.shutdown()
    assert elapsed < 0.30
    assert maximum_active <= 2
