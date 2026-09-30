"""CPU-safe tests for the hyper scheduler H0 singleton chassis.

Covers mode parsing/validation, the admission-cap FIFO semantics (mocked
generation path — no model, no GPU), the width seam contract, and the stats
shape. The GPU parity gate (>=0.99x serial tok/s + byte-identical trajectory
sha on nprompt-16k/71k) is controller-run and deliberately not here.
"""

from threading import Event, Lock, Thread
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from mtplx.batching import SchedulerMode
from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.server import openai
from mtplx.server.hyper import (
    HYPER_ADMISSION_CAP,
    HyperAdmissionGate,
    HyperCohortPlan,
    HyperRequestMeta,
    SINGLETON_PLAN,
)
from mtplx.server.openai import parse_args


def _wait_until(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


# ---------------------------------------------------------------------------
# Mode parsing + registration
# ---------------------------------------------------------------------------


def test_hyper_is_a_first_class_scheduler_mode():
    assert SchedulerMode.HYPER.value == "hyper"

    args = parse_args(["--warmup-tokens", "0", "--scheduler-mode", "hyper"])
    config = openai._scheduler_config_from_args(args)

    assert config.mode == SchedulerMode.HYPER
    assert openai._scheduler_policy_label(config) == "hyper_singleton_admission_1"


def test_hyper_mode_choices_derive_from_the_enum_everywhere():
    from mtplx.cli import SCHEDULER_MODE_CHOICES as cli_choices
    from mtplx.server.openai import SCHEDULER_MODE_CHOICES as server_choices

    expected = tuple(mode.value for mode in SchedulerMode)
    assert cli_choices == expected
    assert tuple(server_choices) == expected
    assert "hyper" in cli_choices


def test_hyper_mode_env_default(monkeypatch):
    monkeypatch.setenv("MTPLX_SCHEDULER_MODE", "hyper")

    args = parse_args(["--warmup-tokens", "0"])

    assert args.scheduler_mode == "hyper"


def test_config_set_accepts_hyper(tmp_path):
    from mtplx.commands.public import cmd_config_public

    config_path = tmp_path / "config.json"
    args = SimpleNamespace(
        config=str(config_path),
        config_action="set",
        key="scheduler_mode",
        value="hyper",
        json=False,
    )

    assert cmd_config_public(args) == 0

    with pytest.raises(SystemExit, match="hyper"):
        cmd_config_public(
            SimpleNamespace(
                config=str(config_path),
                config_action="set",
                key="scheduler_mode",
                value="warp",
                json=False,
            )
        )


# ---------------------------------------------------------------------------
# Launch validation: flags that contradict admission=1 fail loudly
# ---------------------------------------------------------------------------


def _hyper_args(*extra: str):
    return parse_args(["--warmup-tokens", "0", "--scheduler-mode", "hyper", *extra])


def test_hyper_validation_accepts_the_plain_launch():
    openai._validate_hyper_settings(_hyper_args())
    openai._validate_hyper_settings(_hyper_args("--max-active-requests", "1"))
    openai._validate_hyper_settings(_hyper_args("--batching-preset", "solo"))
    openai._validate_hyper_settings(_hyper_args("--batch-wait-ms", "0"))


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (("--max-active-requests", "8"), "external admission"),
        (("--decode-batch-max", "2"), "decodes one external request"),
        (("--batch-wait-ms", "50"), "never holds a gather window"),
        (("--batching-preset", "agent"), "single-admission"),
        (("--batching-preset", "throughput"), "single-admission"),
    ],
)
def test_hyper_validation_rejects_multi_admission_flags(extra, match):
    with pytest.raises(ValueError, match=match):
        openai._validate_hyper_settings(_hyper_args(*extra))


def test_hyper_validation_never_touches_other_modes():
    args = parse_args(
        [
            "--warmup-tokens",
            "0",
            "--scheduler-mode",
            "ar_batch",
            "--batching-preset",
            "throughput",
            "--max-active-requests",
            "8",
            "--decode-batch-max",
            "8",
        ]
    )
    openai._validate_hyper_settings(args)


# ---------------------------------------------------------------------------
# Routing: hyper never rides a batched driver
# ---------------------------------------------------------------------------


def test_hyper_never_routes_through_batched_drivers():
    args = _hyper_args()
    state = SimpleNamespace(args=args)

    # Report-6 receipt: a width-1 ride through a batched driver loses the
    # paged KV cache + custom kernels + banked verify. Hyper must always be
    # invisible to both batched lanes.
    assert openai._use_live_mtp_batch(state, effective_mode="mtp") is False
    assert openai._use_live_ar_batch(state, effective_mode="mtp") == (False, None)
    assert openai._use_live_ar_batch(state, effective_mode="ar") == (False, None)


# ---------------------------------------------------------------------------
# Admission-cap semantics through the real dispatch (mocked generation)
# ---------------------------------------------------------------------------


def _hyper_dispatch_state():
    """Server-state stand-in with the REAL ModelWorkScheduler + hyper gate.

    The generation path is mocked; the FIFO band and the gate are the units
    under test, wired exactly as ServerState wires them.
    """

    args = _hyper_args()
    return SimpleNamespace(
        args=args,
        model_scheduler=ModelWorkScheduler(name="test-hyper"),
        hyper_gate=HyperAdmissionGate(),
        runtime=SimpleNamespace(mtp_enabled=True),
    )


def test_hyper_dispatch_caps_concurrency_at_one_and_preserves_fifo(monkeypatch):
    state = _hyper_dispatch_state()
    gate = state.hyper_gate
    release = Event()
    tracking = {"in_flight": 0, "max_in_flight": 0, "order": [], "lanes": []}
    tracking_lock = Lock()

    def fake_generation(_state, _prompt_ids, **kwargs):
        with tracking_lock:
            tracking["in_flight"] += 1
            tracking["max_in_flight"] = max(
                tracking["max_in_flight"], tracking["in_flight"]
            )
            observability = kwargs.get("request_observability") or {}
            tracking["order"].append(observability.get("request_id"))
            tracking["lanes"].append(observability.get("scheduler_lane"))
        assert release.wait(timeout=10.0)
        with tracking_lock:
            tracking["in_flight"] -= 1
        return {"text": "ok", "tokens": [1], "stats": {}}

    monkeypatch.setattr(openai, "_run_generation", fake_generation)

    results = {}

    def dispatch(index: int) -> None:
        results[index] = openai._run_generation_dispatched(
            state,
            [1, 2, 3],
            batch_key=f"test.hyper.{index}",
            response_id=f"req-{index}",
            generation_mode="mtp",
            request_observability={},
        )

    threads = []
    try:
        # Deterministic arrival order: each request must be visibly admitted
        # (running or queued) before the next arrives.
        for index in range(3):
            thread = Thread(target=dispatch, args=(index,), daemon=True)
            thread.start()
            threads.append(thread)
            assert _wait_until(
                lambda index=index: (
                    gate.snapshot()["active"] + gate.snapshot()["queue_depth"]
                )
                == index + 1
            )

        snapshot = gate.snapshot()
        assert snapshot["active"] == 1
        assert snapshot["queue_depth"] == 2
        assert snapshot["admitted_total"] == 3
        assert snapshot["admission_cap"] == HYPER_ADMISSION_CAP == 1

        release.set()
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive()
    finally:
        release.set()
        state.model_scheduler.shutdown(wait=False, cancel_futures=True)

    # One at a time, in arrival order, all on the hyper singleton lane.
    assert tracking["max_in_flight"] == 1
    assert tracking["order"] == ["req-0", "req-1", "req-2"]
    assert tracking["lanes"] == ["hyper_singleton"] * 3

    snapshot = gate.snapshot()
    assert snapshot["active"] == 0
    assert snapshot["queue_depth"] == 0
    assert snapshot["completed_total"] == 3
    assert snapshot["failed_total"] == 0
    assert snapshot["width_stats"]["requests_by_width"] == {"1": 3}
    assert all(results[index]["text"] == "ok" for index in range(3))


def test_hyper_dispatch_settles_the_ticket_when_generation_fails(monkeypatch):
    state = _hyper_dispatch_state()

    def failing_generation(_state, _prompt_ids, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(openai, "_run_generation", failing_generation)

    try:
        with pytest.raises(RuntimeError, match="boom"):
            openai._run_generation_dispatched(
                state,
                [1],
                batch_key="test.hyper.fail",
                response_id="req-fail",
                generation_mode="mtp",
                request_observability={},
            )
    finally:
        state.model_scheduler.shutdown(wait=False, cancel_futures=True)

    snapshot = state.hyper_gate.snapshot()
    assert snapshot["active"] == 0
    assert snapshot["queue_depth"] == 0
    assert snapshot["failed_total"] == 1
    assert snapshot["completed_total"] == 0


def test_serial_dispatch_has_no_hyper_gate_and_still_serves(monkeypatch):
    args = parse_args(["--warmup-tokens", "0"])
    state = SimpleNamespace(
        args=args,
        model_scheduler=ModelWorkScheduler(name="test-serial"),
        runtime=SimpleNamespace(mtp_enabled=True),
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda _state, _prompt_ids, **kwargs: {
            "text": "serial-ok",
            "lane": (kwargs.get("request_observability") or {}).get(
                "scheduler_lane"
            ),
        },
    )

    try:
        generated = openai._run_generation_dispatched(
            state,
            [1, 2],
            batch_key="test.serial",
            response_id="req-serial",
            generation_mode="mtp",
            request_observability={},
        )
    finally:
        state.model_scheduler.shutdown(wait=False, cancel_futures=True)

    assert generated["text"] == "serial-ok"
    # Serial requests never get the hyper lane stamp.
    assert generated["lane"] is None


def test_serial_dispatch_passes_scheduler_queue_wait(monkeypatch):
    args = parse_args(["--warmup-tokens", "0"])
    state = SimpleNamespace(
        args=args,
        model_scheduler=ModelWorkScheduler(name="test-serial-queue"),
        runtime=SimpleNamespace(mtp_enabled=True),
    )
    first_running = Event()
    release_first = Event()
    queue_waits: dict[str, float | None] = {}

    def fake_generation(_state, _prompt_ids, **kwargs):
        request_id = (kwargs.get("request_observability") or {}).get("request_id")
        queue_waits[request_id] = kwargs.get("queue_wait_s")
        if request_id == "req-first":
            first_running.set()
            assert release_first.wait(timeout=10.0)
        return {"text": "ok"}

    monkeypatch.setattr(openai, "_run_generation", fake_generation)

    def dispatch(request_id: str) -> None:
        openai._run_generation_dispatched(
            state,
            [1, 2],
            batch_key=f"test.serial.{request_id}",
            response_id=request_id,
            generation_mode="mtp",
            request_observability={},
        )

    first = Thread(target=dispatch, args=("req-first",), daemon=True)
    second = Thread(target=dispatch, args=("req-second",), daemon=True)
    try:
        first.start()
        assert first_running.wait(timeout=5.0)
        second.start()
        assert _wait_until(
            lambda: state.model_scheduler.stats()["foreground_pending"] == 1
        )
        time.sleep(0.3)
        release_first.set()
        first.join(timeout=10.0)
        second.join(timeout=10.0)
        assert not first.is_alive() and not second.is_alive()
    finally:
        release_first.set()
        state.model_scheduler.shutdown(wait=False, cancel_futures=True)

    # The first request found the owner idle; the second queued behind it.
    assert queue_waits["req-first"] is not None
    assert queue_waits["req-first"] < 0.25
    assert queue_waits["req-second"] is not None
    assert queue_waits["req-second"] >= 0.3


# ---------------------------------------------------------------------------
# Admission gate unit semantics
# ---------------------------------------------------------------------------


def test_gate_release_settles_only_never_started_tickets():
    gate = HyperAdmissionGate()
    ran = gate.admit(request_id="ran", prompt_tokens=4)
    gate.bind(ran, lambda: {"ok": True})()
    gate.release(ran)  # idempotent no-op after completion

    abandoned = gate.admit(request_id="abandoned", prompt_tokens=4)
    gate.release(abandoned)
    gate.release(abandoned)  # idempotent

    snapshot = gate.snapshot()
    assert snapshot["completed_total"] == 1
    assert snapshot["abandoned_before_start_total"] == 1
    assert snapshot["queue_depth"] == 0
    assert snapshot["active"] == 0


# ---------------------------------------------------------------------------
# Width seam contract
# ---------------------------------------------------------------------------


class _Executor:
    def __init__(self, width: int):
        self.width = width
        self.plans = []
        self.cohorts = []

    def plan(self, request: HyperRequestMeta) -> HyperCohortPlan:
        self.plans.append(request)
        return HyperCohortPlan(width=self.width, reason=f"test_w{self.width}")

    def run_cohort(self, plan, singleton):
        self.cohorts.append(plan)
        return singleton()


def test_seam_default_is_singleton_passthrough():
    gate = HyperAdmissionGate()
    meta = HyperRequestMeta(request_id="r", prompt_tokens=8, generation_mode="mtp")

    assert gate.width_executor_installed is False
    assert gate.plan_cohort(meta) is SINGLETON_PLAN


def test_seam_width_one_bypasses_the_executor_entirely():
    """The oMLX lesson, as an executable contract: width 1 never builds a
    cohort — the singleton closure is called directly."""

    gate = HyperAdmissionGate()
    executor = _Executor(width=1)
    gate.install_width_executor(executor)
    calls = []

    ticket = gate.admit(request_id="w1", prompt_tokens=8)
    result = gate.bind(ticket, lambda: calls.append("singleton") or {"ok": 1})()

    assert result == {"ok": 1}
    assert calls == ["singleton"]
    assert executor.plans and executor.plans[0].request_id == "w1"
    assert executor.cohorts == []  # run_cohort must not fire at width 1
    assert gate.snapshot()["width_stats"]["requests_by_width"] == {"1": 1}


def test_seam_width_two_routes_through_the_executor():
    gate = HyperAdmissionGate()
    executor = _Executor(width=2)
    gate.install_width_executor(executor)

    ticket = gate.admit(request_id="w2", prompt_tokens=8)
    result = gate.bind(ticket, lambda: {"ok": 2})()

    assert result == {"ok": 2}
    assert len(executor.cohorts) == 1
    assert executor.cohorts[0].width == 2
    assert gate.snapshot()["width_stats"]["requests_by_width"] == {"2": 1}


def test_seam_wide_plan_without_executor_fails_loudly():
    """A wide plan that outlives its executor (uninstall race) must error,
    never silently pad a cohort or pretend the request was singleton."""

    gate = HyperAdmissionGate()
    executor = _Executor(width=2)
    gate.install_width_executor(executor)
    ticket = gate.admit(request_id="race", prompt_tokens=8)
    bound = gate.bind(ticket, lambda: {"ok": True})

    original_plan_cohort = gate.plan_cohort

    def plan_then_lose_executor(meta):
        plan = original_plan_cohort(meta)  # width 2 from the executor
        gate.clear_width_executor()
        return plan

    gate.plan_cohort = plan_then_lose_executor
    try:
        with pytest.raises(RuntimeError, match="no width executor"):
            bound()
    finally:
        gate.plan_cohort = original_plan_cohort

    snapshot = gate.snapshot()
    assert snapshot["failed_total"] == 1
    assert snapshot["active"] == 0


def test_seam_rejects_malformed_executors_and_plans():
    gate = HyperAdmissionGate()

    with pytest.raises(TypeError, match="plan"):
        gate.install_width_executor(SimpleNamespace())

    class _BadPlan:
        def plan(self, request):
            return {"width": 2}

        def run_cohort(self, plan, singleton):  # pragma: no cover - unused
            return singleton()

    gate.install_width_executor(_BadPlan())
    meta = HyperRequestMeta(request_id="r", prompt_tokens=1, generation_mode="mtp")
    with pytest.raises(TypeError, match="HyperCohortPlan"):
        gate.plan_cohort(meta)

    with pytest.raises(ValueError, match="width"):
        HyperCohortPlan(width=0, reason="nope")


# ---------------------------------------------------------------------------
# Stats surface shape
# ---------------------------------------------------------------------------


def _hyper_stats_state(generation_mode: str = "mtp"):
    from mtplx.server.dashboard_state import DashboardState

    args = _hyper_args()
    args.generation_mode = generation_mode
    return SimpleNamespace(
        args=args,
        dashboard=DashboardState(),
        runtime=SimpleNamespace(mtp_enabled=True),
        hyper_gate=HyperAdmissionGate(),
        foreground_count=lambda: 0,
    )


def test_hyper_scheduler_state_shape():
    state = _hyper_stats_state()
    ticket = state.hyper_gate.admit(request_id="done", prompt_tokens=3)
    state.hyper_gate.bind(ticket, lambda: {"ok": True})()
    queued = state.hyper_gate.admit(request_id="queued", prompt_tokens=3)

    payload = openai._mtplx_scheduler_state(state)

    assert payload["mode"] == "hyper"
    assert payload["scheduler_policy"] == "hyper_singleton_admission_1"
    assert payload["active_lane"] == "hyper_singleton"
    assert payload["mtp_disabled_reason"] is None

    hyper = payload["hyper"]
    assert hyper["stage"] == "h0"
    assert hyper["admission_cap"] == 1
    assert hyper["width"] == 1
    assert hyper["passthrough"] == "serial_b1"
    assert hyper["width_executor_installed"] is False
    assert hyper["queue_depth"] == 1
    assert hyper["completed_total"] == 1
    assert hyper["queue_wait_s"]["count"] == 1
    # The placeholder home for H1/H2 width receipts must already exist.
    assert hyper["width_stats"] == {
        "requests_by_width": {"1": 1},
        "self_spec_rows_launched": 0,
        "self_spec_rows_committed": 0,
        "fork_attempts": 0,
        "fork_wins": 0,
    }
    # Dashboard telemetry carries the same block in hyper mode.
    assert payload["telemetry"]["admission_cap"] == 1

    state.hyper_gate.release(queued)


def test_hyper_scheduler_state_queued_requests_never_flip_to_ar():
    state = _hyper_stats_state()
    state.foreground_count = lambda: 5

    payload = openai._mtplx_scheduler_state(state)

    assert payload["active_lane"] == "hyper_singleton"
    assert payload["mtp_disabled_reason"] is None


def test_hyper_scheduler_state_labels_ar_generation_mode():
    state = _hyper_stats_state(generation_mode="ar")

    payload = openai._mtplx_scheduler_state(state)

    assert payload["active_lane"] == "hyper_singleton_ar"
    assert payload["mtp_disabled_reason"] == "generation_mode_ar"


def test_serial_scheduler_state_reports_empty_hyper_block():
    from mtplx.server.dashboard_state import DashboardState

    args = parse_args(["--warmup-tokens", "0"])
    state = SimpleNamespace(
        args=args,
        dashboard=DashboardState(),
        runtime=SimpleNamespace(mtp_enabled=True),
        foreground_count=lambda: 0,
    )

    payload = openai._mtplx_scheduler_state(state)

    assert payload["mode"] == "serial"
    assert payload["hyper"] == {}
