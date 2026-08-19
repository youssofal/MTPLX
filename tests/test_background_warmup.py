"""Silent background warming (Lane E2) unit tests.

No model: the model scheduler, generation, and kernel prewarm are faked.
What's under test is the plan mechanics — idle-lane submission ordering,
foreground-yield resubmission, abandonment under sustained load, status
publication (JSON-safe, replace-not-mutate), and the startup-warmup mode
split (background default vs legacy blocking via env).
"""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import mtplx.server.openai as server


class FakeScheduler:
    """Collects submissions; the test drains the idle queue explicitly."""

    def __init__(self) -> None:
        self.idle: deque = deque()
        self.foreground_busy = False
        self.idle_batch_keys: list[str] = []

    def submit_foreground(self, fn, *args, batch_key=None, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - test aid
            future.set_exception(exc)
        return future

    def submit_idle_postcommit(self, fn, *args, batch_key=None, **kwargs):
        future: Future = Future()
        self.idle.append((fn, args, kwargs, future))
        self.idle_batch_keys.append(str(batch_key))
        return future

    def foreground_pending_or_active(self) -> bool:
        return self.foreground_busy

    def drain(self, limit: int = 64) -> int:
        ran = 0
        while self.idle and ran < limit:
            fn, args, kwargs, future = self.idle.popleft()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # pragma: no cover - test aid
                future.set_exception(exc)
            ran += 1
        return ran


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [7] * max(1, len(text) // 4)


def make_state(scheduler: FakeScheduler | None = None, **args_overrides):
    args = SimpleNamespace(
        warmup_tokens=16,
        strict_warmup=False,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )
    for key, value in args_overrides.items():
        setattr(args, key, value)
    state = SimpleNamespace(
        args=args,
        model_scheduler=scheduler or FakeScheduler(),
        runtime=SimpleNamespace(tokenizer=FakeTokenizer()),
        context_window=262144,
        foreground_active=0,
    )
    # Mirrors ServerState.has_foreground: non-zero while a request is being
    # served, independent of the completion stamp in last_request_at.
    state.has_foreground = lambda: state.foreground_active > 0
    return state


def test_background_warmup_enabled_env(monkeypatch):
    monkeypatch.delenv("MTPLX_WARMUP_BACKGROUND", raising=False)
    assert server._background_warmup_enabled() is True
    for off in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv("MTPLX_WARMUP_BACKGROUND", off)
        assert server._background_warmup_enabled() is False
    monkeypatch.setenv("MTPLX_WARMUP_BACKGROUND", "1")
    assert server._background_warmup_enabled() is True


def test_warmup_ladder_contexts_default_and_overrides(monkeypatch):
    state = make_state()
    monkeypatch.delenv("MTPLX_WARMUP_LADDER", raising=False)
    assert server._warmup_ladder_contexts(state) == [512, 2560]
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "2048, 512, junk, 512, -3, 0")
    assert server._warmup_ladder_contexts(state) == [512, 2048]
    # Classes that would not fit the model context window are dropped.
    state.context_window = 1024
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "512,2560")
    assert server._warmup_ladder_contexts(state) == [512]
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "")
    assert server._warmup_ladder_contexts(state) == []


def test_foreground_yield_shim_reads_scheduler_queues():
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    shim = server._ForegroundYield(state)
    assert shim.is_set() is False
    scheduler.foreground_busy = True
    assert shim.is_set() is True
    # A broken scheduler must never break warming.
    state.model_scheduler = SimpleNamespace()
    assert server._ForegroundYield(state).is_set() is False


def _deferral_probe(monkeypatch, state, scheduler, grace: str = "90"):
    """Run one warmup plan with timers faked; report whether any rung
    actually generated and what the first step's published state is."""
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "16")
    monkeypatch.setenv("MTPLX_WARMUP_IDLE_GRACE_S", grace)
    monkeypatch.setattr(server, "_prewarm_gqa_packed_pipelines", lambda: True)
    generations: list[int] = []
    monkeypatch.setattr(
        server,
        "_run_generation",
        lambda _state, prompt_ids, **kwargs: generations.append(len(prompt_ids))
        or {"tok_s": 1.0},
    )
    timers: list[tuple[float, object, tuple]] = []

    class FakeTimer:
        def __init__(self, interval, fn, args=()):
            timers.append((interval, fn, tuple(args)))
            self.daemon = False

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Timer", FakeTimer)
    status_host: dict = {}
    warming = server._BackgroundWarmup(state, status_host, [1, 2, 3])
    warming.submit(0)
    scheduler.drain()
    return generations, status_host, timers


def test_background_warmup_runs_all_steps_and_publishes_done(monkeypatch):
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "16,32")
    monkeypatch.setattr(server, "_prewarm_gqa_packed_pipelines", lambda: True)
    seen_lengths: list[int] = []

    def fake_run_generation(_state, prompt_ids, **kwargs):
        assert kwargs["request_observability"]["warmup"] is True
        assert isinstance(kwargs["cancel_event"], server._ForegroundYield)
        seen_lengths.append(len(prompt_ids))
        return {"tok_s": 42.0}

    monkeypatch.setattr(server, "_run_generation", fake_run_generation)
    status_host: dict = {}
    warming = server._BackgroundWarmup(state, status_host, [1, 2, 3])
    assert status_host["background"]["state"] == "pending"
    json.dumps(status_host["background"])  # publish must be JSON-safe
    warming.submit(0)
    scheduler.drain()
    snapshot = status_host["background"]
    json.dumps(snapshot)
    assert snapshot["state"] == "done"
    assert seen_lengths == [16, 32]
    assert [step["state"] for step in snapshot["steps"]] == ["ok", "ok", "ok"]
    assert snapshot["steps"][1]["tok_s"] == 42.0
    assert snapshot["resubmits"] == 0


def test_background_warmup_defers_while_foreground_recent(monkeypatch):
    """2.8.3 idle grace: a daemon that served traffic recently must not
    burn GPU on warm rungs between chat turns — steps defer on a timer
    (no model work, no resubmit budget) until the grace elapses."""
    import time as _time

    scheduler = FakeScheduler()
    state = make_state(scheduler)
    state.last_request_at = _time.time()  # a response just finished
    generations, status_host, timers = _deferral_probe(monkeypatch, state, scheduler)
    # No model work ran; the plan is waiting for idle, budget untouched.
    assert generations == []
    assert status_host["background"]["steps"][0]["state"] == "waiting_idle"
    assert status_host["background"]["resubmits"] == 0
    assert len(timers) == 1
    wait_s, fn, args = timers[0]
    assert 0.0 < wait_s <= 90.0
    # Grace elapses: the deferred submit now runs the plan to completion.
    state.last_request_at = _time.time() - 3600.0
    fn(*args)
    scheduler.drain()
    assert generations == [16]
    assert status_host["background"]["state"] == "done"


def test_background_warmup_yield_resubmits_then_completes(monkeypatch):
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "16")
    monkeypatch.setattr(server, "_prewarm_gqa_packed_pipelines", lambda: True)
    cancels = {"remaining": 2}

    def flaky_run_generation(_state, prompt_ids, **kwargs):
        if cancels["remaining"] > 0:
            cancels["remaining"] -= 1
            raise server._StreamCancelled("foreground preempted warming")
        return {"tok_s": 55.0}

    monkeypatch.setattr(server, "_run_generation", flaky_run_generation)
    status_host: dict = {}
    warming = server._BackgroundWarmup(state, status_host, [1])
    warming.submit(0)
    scheduler.drain()
    snapshot = status_host["background"]
    assert snapshot["state"] == "done"
    ladder_step = snapshot["steps"][1]
    assert ladder_step["state"] == "ok"
    assert ladder_step["yields"] == 2
    assert snapshot["resubmits"] == 2


def test_background_warmup_abandons_under_sustained_load(monkeypatch):
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "16,32")
    monkeypatch.setattr(server, "_prewarm_gqa_packed_pipelines", lambda: True)

    def always_cancelled(_state, prompt_ids, **kwargs):
        raise server._StreamCancelled("foreground preempted warming")

    monkeypatch.setattr(server, "_run_generation", always_cancelled)
    status_host: dict = {}
    warming = server._BackgroundWarmup(state, status_host, [1])
    warming.submit(0)
    scheduler.drain(limit=128)
    snapshot = status_host["background"]
    json.dumps(snapshot)
    assert snapshot["state"] == "abandoned_busy"
    assert snapshot["resubmits"] == server._BackgroundWarmup.MAX_RESUBMITS + 1
    assert snapshot["steps"][1]["state"] == "abandoned"
    assert snapshot["steps"][2]["state"] == "abandoned"
    # The plan stopped: nothing left in the idle queue.
    assert not scheduler.idle


def test_background_warmup_step_failure_does_not_stop_the_plan(monkeypatch):
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "16,32")
    monkeypatch.setattr(server, "_prewarm_gqa_packed_pipelines", lambda: True)
    calls = {"n": 0}

    def first_ladder_fails(_state, prompt_ids, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("metal exploded")
        return {"tok_s": 33.0}

    monkeypatch.setattr(server, "_run_generation", first_ladder_fails)
    status_host: dict = {}
    warming = server._BackgroundWarmup(state, status_host, [1])
    warming.submit(0)
    scheduler.drain()
    snapshot = status_host["background"]
    assert snapshot["state"] == "done"
    assert snapshot["steps"][1]["state"] == "failed"
    assert "RuntimeError" in snapshot["steps"][1]["error"]
    assert snapshot["steps"][2]["state"] == "ok"


def test_run_startup_warmup_background_mode_returns_without_extended_block(
    monkeypatch,
):
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    monkeypatch.delenv("MTPLX_WARMUP_EXTENDED", raising=False)
    monkeypatch.delenv("MTPLX_WARMUP_BACKGROUND", raising=False)
    monkeypatch.setenv("MTPLX_WARMUP_LADDER", "16")
    monkeypatch.setattr(server, "_prewarm_gqa_packed_pipelines", lambda: True)
    monkeypatch.setattr(
        server,
        "_run_generation",
        lambda _state, prompt_ids, **kwargs: {
            "tok_s": 50.0,
            "completion_tokens": kwargs.get("max_tokens"),
        },
    )
    status = server._run_startup_warmup(state)
    # Startup returned with the extended pass still queued, not executed.
    assert status["ran"] is True
    assert status["extended"]["mode"] == "background"
    assert status["background"]["state"] == "pending"
    assert len(scheduler.idle) == 1
    scheduler.drain()
    assert status["background"]["state"] == "done"
    json.dumps(status)


def test_run_startup_warmup_legacy_blocking_mode(monkeypatch):
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    monkeypatch.delenv("MTPLX_WARMUP_EXTENDED", raising=False)
    monkeypatch.setenv("MTPLX_WARMUP_BACKGROUND", "0")
    monkeypatch.setattr(server, "_prewarm_gqa_packed_pipelines", lambda: True)
    monkeypatch.setattr(
        server,
        "_run_generation",
        lambda _state, prompt_ids, **kwargs: {
            "tok_s": 50.0,
            "completion_tokens": kwargs.get("max_tokens"),
        },
    )
    status = server._run_startup_warmup(state)
    assert status["extended"]["ran"] is True
    assert status["extended"]["gqa_packed_pipelines"] is True
    assert "background" not in status
    assert not scheduler.idle


def test_run_startup_warmup_disabled_has_no_background_key(monkeypatch):
    state = make_state(warmup_tokens=0)
    monkeypatch.delenv("MTPLX_WARMUP_BACKGROUND", raising=False)
    status = server._run_startup_warmup(state)
    assert status["enabled"] is False
    assert "background" not in status
    assert "extended" not in status


def test_dashboard_record_completion_skips_warmup_rows():
    calls: list[str] = []
    dashboard = SimpleNamespace(
        in_flight=SimpleNamespace(
            deregister=lambda *_: calls.append("deregister"),
            count=lambda: 0,
        ),
        progress_events=SimpleNamespace(forget=lambda *_: calls.append("forget")),
        lifetime=SimpleNamespace(
            record_completion=lambda **_: calls.append("lifetime")
        ),
        rolling=SimpleNamespace(append=lambda *_a, **_k: calls.append("rolling")),
        prefill_history=SimpleNamespace(
            append=lambda *_: calls.append("prefill_history")
        ),
        bus=SimpleNamespace(publish=lambda *_: calls.append("bus")),
    )
    state = SimpleNamespace(dashboard=dashboard, model_id="m")
    server._dashboard_record_completion(
        state,
        envelope={"warmup": True, "decode_tok_s": 99.0},
        stats={},
    )
    assert calls == []
    server._dashboard_record_completion(
        state,
        envelope={"decode_tok_s": 99.0, "prompt_tokens": 4, "completion_tokens": 2},
        stats={},
    )
    assert "lifetime" in calls and "rolling" in calls


def test_background_warmup_defers_while_a_request_is_in_flight(monkeypatch):
    """A request that has ARRIVED but not finished must hold the ladder.

    last_request_at is stamped at completion, so a daemon that has not
    completed a request yet still reads 0.0 while one is generating. That
    read as "infinitely quiet" and let a warm rung run against live
    traffic -- and because a UI that restarts the engine on a config change
    is typed into immediately afterwards, the very first request of a serve
    was the one most likely to be warmed over.
    """
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    state.last_request_at = 0.0  # nothing has COMPLETED yet
    state.foreground_active = 1  # ...but a real request is generating now
    generations, status_host, timers = _deferral_probe(monkeypatch, state, scheduler)
    assert generations == [], "warming generated while a request was in flight"
    assert status_host["background"]["steps"][0]["state"] == "waiting_idle"
    assert status_host["background"]["resubmits"] == 0
    assert len(timers) == 1


def test_background_warmup_holds_a_live_request_even_at_zero_grace(monkeypatch):
    """MTPLX_WARMUP_IDLE_GRACE_S=0 drops the between-turns wait, not the
    live-traffic guard.

    Expressing "busy" as zero quiet made the hold collapse at grace 0
    (``0.0 < 0.0`` is False), so the one knob an operator reaches for when
    warm rungs are not running also handed them a request that was still
    generating -- the exact starvation the guard exists to prevent.
    """
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    state.last_request_at = 0.0
    state.foreground_active = 1  # a real request is generating right now
    generations, status_host, timers = _deferral_probe(
        monkeypatch, state, scheduler, grace="0"
    )
    assert generations == [], "warming generated against a live request at grace 0"
    assert status_host["background"]["steps"][0]["state"] == "waiting_idle"
    assert status_host["background"]["resubmits"] == 0
    assert len(timers) == 1
    wait_s, fn, args = timers[0]
    assert wait_s >= 1.0  # _defer_step floors the re-check, never a hot loop
    # The request finishes and nothing else is queued: zero grace means the
    # plan runs on the very next admission, with no wait to serve out.
    state.foreground_active = 0
    fn(*args)
    scheduler.drain()
    assert generations == [16]
    assert status_host["background"]["state"] == "done"


def test_background_warmup_defers_while_foreground_is_queued(monkeypatch):
    """Foreground work queued on the scheduler counts as busy at admission
    time, not only once it starts executing."""
    scheduler = FakeScheduler()
    scheduler.foreground_busy = True
    state = make_state(scheduler)
    state.last_request_at = 0.0
    generations, status_host, _ = _deferral_probe(monkeypatch, state, scheduler)
    assert generations == [], "warming generated while foreground work was queued"
    assert status_host["background"]["steps"][0]["state"] == "waiting_idle"


def test_background_warmup_still_warms_a_genuinely_idle_fresh_daemon(monkeypatch):
    """The guard must not cost the case it exists for: nothing running and
    nothing ever completed still warms immediately."""
    scheduler = FakeScheduler()
    state = make_state(scheduler)
    state.last_request_at = 0.0
    generations, status_host, _ = _deferral_probe(monkeypatch, state, scheduler)
    assert generations == [16]
    assert status_host["background"]["state"] == "done"
