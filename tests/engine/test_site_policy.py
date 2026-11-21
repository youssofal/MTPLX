"""The policy of a replay site, against a stand-in executor.

Which call of a geometry key is served by MLX, recorded, checked and replayed;
what is counted; what refuses a program; what the switch does. The compiled
module is not needed: ``site_fakes.FakeEngine`` replays by calling the function.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest
from site_fakes import FakeEngine

from mtplx.engine import site as site_module
from mtplx.engine.site import ReplaySite, compare_outputs, private_copy


def _fn(inputs):
    """A fixed-shape function with state: (x, state) -> (y, new_state)."""

    x, state = inputs
    y = (x * 2.0 + state).astype(mx.float32)
    return [y, state + 1.0]


def _inputs(step: int, width: int = 4):
    x = (mx.arange(width, dtype=mx.float32) + step).reshape(1, width)
    state = mx.full((1, width), float(step), dtype=mx.float32)
    mx.eval(x, state)
    return [x, state]


class _Driver:
    """Calls a site the way the product path does and checks every answer."""

    def __init__(self, site, fn=_fn):
        self.site = site
        self.fn = fn
        self.mlx_calls = 0
        self.kinds: list[str] = []

    def call(self, key, inputs, admit=None, quiet=False):
        expected = list(self.fn(list(inputs)))
        mx.eval(*expected)
        ticket = self.site.begin(key, list(inputs), admit=admit, quiet=quiet)
        if ticket.native:
            outputs = ticket.outputs
        else:
            self.mlx_calls += 1
            outputs = list(self.fn(list(inputs)))
        ticket.finish(outputs)
        mx.eval(*outputs)
        assert compare_outputs(list(outputs), expected) is None
        self.kinds.append(ticket.kind)
        return ticket


@pytest.fixture(autouse=True)
def _switch(monkeypatch):
    for name in (
        "MTPLX_NATIVE_ENGINE", "MTPLX_NATIVE_ENGINE_DUMP", "MTPLX_NATIVE_ENGINE_REPLAY",
        "MTPLX_NATIVE_ENGINE_PROFILE", "MTPLX_NATIVE_ENGINE_MIXED",
        "MTPLX_NATIVE_ENGINE_PROGRAMS", "MTPLX_NATIVE_ENGINE_DUMP_DATA",
        "MTPLX_NATIVE_ENGINE_WARM_CALLS", "MTPLX_NATIVE_ENGINE_PROGRAM_MB",
    ):
        monkeypatch.delenv(name, raising=False)
    # These tests walk a key through its states call by call: record on the
    # second call (the product default waits longer; see the test below).
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_WARM_CALLS", "1")


# A key's life with these settings, call by call: the MLX path alone; recorded;
# recorded again (the two agree, and the program is sealed at the end of that
# call); checked; then native.
WALK = ["fallback", "record", "record", "verify"]
WALK_REASONS = {"warming": 1, "recording": 2, "verifying": 1}


def _site(engine, **kwargs):
    lines: list[str] = []
    site = ReplaySite("test_site", engine=engine, log=lines.append, **kwargs)
    site.lines = lines
    return site


def test_a_key_is_warmed_recorded_checked_and_then_replayed():
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(10):
        driver.call("k", _inputs(step))
    assert driver.kinds == WALK + ["native"] * 6
    # The MLX function ran for the first four calls only.
    assert driver.mlx_calls == 4
    # Two recordings that agree; the check records nothing.
    assert engine.recordings == 2 and engine.builds == 2
    # The check replayed once, then six native rounds.
    assert engine.runs == 7
    report = driver.site.report()
    assert report["calls"] == 10 and report["native"] == 6
    assert report["fallback"] == WALK_REASONS
    assert report["programs_built"] == 1 and report["programs_live"] == 1
    assert report["native_share"] == 0.6
    # The seal is no call of its own: it is timed on the call that settled the program.
    assert report["seal_s"] > 0 and "sealing" not in report["fallback"]
    assert report["launches_per_run"] == 3 and report["command_buffers_per_run"] == 1
    assert report["gpu_ms_per_run"] == 0.005


def test_the_first_call_of_a_key_is_never_recorded():
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    driver.call("k", _inputs(0))
    assert engine.recordings == 0 and engine.recording is None


def test_a_new_geometry_records_its_own_program_and_the_old_one_keeps_serving():
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("a", _inputs(step))
    for step in range(6):
        driver.call("b", _inputs(step, width=6))
    driver.call("a", _inputs(9))
    assert driver.kinds == (WALK + ["native"] * 2) * 2 + ["native"]
    report = driver.site.report()
    assert report["fallback"] == {"warming": 1, "key_new": 1, "recording": 4, "verifying": 2}
    assert report["programs_live"] == 2 and report["keys"] == 2


def test_a_replay_that_differs_in_one_bit_is_refused_and_mlx_keeps_serving():
    engine = FakeEngine(_fn)
    engine.wrong_on_run = 1  # the check's replay
    driver = _Driver(_site(engine))
    for step in range(8):
        driver.call("k", _inputs(step))
    assert driver.kinds == WALK + ["fallback"] * 4
    assert driver.mlx_calls == 8
    report = driver.site.report()
    assert report["native"] == 0
    assert report["programs_refused"] == {"bit_mismatch": 1}
    assert report["fallback"]["refused:bit_mismatch"] == 4
    refusal = json.loads(report["last_refusal"]["detail"])
    assert refusal["output"] == 0 and refusal["elements_differing"] == 1
    assert any("REFUSED (bit_mismatch)" in line for line in driver.site.lines)


def test_two_recordings_with_different_signatures_refuse_the_program():
    engine = FakeEngine(_fn)
    engine.unstable_inline = True  # the same launches, other argument bytes
    driver = _Driver(_site(engine))
    for step in range(5):
        driver.call("k", _inputs(step))
    assert driver.site.report()["programs_refused"] == {"signature_unstable": 1}
    assert driver.kinds[-1] == "fallback" and driver.mlx_calls == 5


def test_a_first_recording_with_extra_launches_is_replaced_by_the_shorter_one():
    """MLX copied a state bank whole in the recorded call because something
    still held it. That recording must not become the steady program."""

    engine = FakeEngine(_fn)
    engine.launches_by_build = [5, 3, 3]
    driver = _Driver(_site(engine))
    for step in range(8):
        driver.call("k", _inputs(step))
    # first call; recorded (5 launches); recorded again: shorter, it replaces
    # the first; a third recording agrees with it and is sealed; checked, native.
    assert driver.kinds == ["fallback", "record", "record", "record", "verify", "native", "native", "native"]
    assert engine.runs == 4  # one check, whatever the recordings did, and three native rounds
    report = driver.site.report()
    assert report["programs_built"] == 1 and report["programs_rerecorded"] == 1
    assert report["programs_live"] == 1 and report["programs_refused"] == {}
    assert len(driver.site._entries[next(iter(driver.site._entries))].ir.steps) == 3


def test_a_longer_second_recording_keeps_the_first_and_records_again():
    engine = FakeEngine(_fn)
    engine.launches_by_build = [3, 5, 3]
    driver = _Driver(_site(engine))
    for step in range(8):
        driver.call("k", _inputs(step))
    assert driver.kinds == ["fallback", "record", "record", "record", "verify", "native", "native", "native"]
    report = driver.site.report()
    assert report["rechecks"] == 1 and report.get("programs_rerecorded") is None
    assert report["programs_live"] == 1
    assert len(driver.site._entries[next(iter(driver.site._entries))].ir.steps) == 3


def test_recordings_that_never_agree_in_length_refuse_the_program():
    engine = FakeEngine(_fn)
    engine.unstable_signature = True  # every recording is one launch longer
    driver = _Driver(_site(engine))
    # Nine recordings that never agree, a rest of 64 calls, and again, three
    # rests in all; the fourth spent budget refuses the key.
    for step in range(1 + 10 + 3 * (64 + 9) + 3):
        driver.call("k", _inputs(step))
    round_of_recordings = ["record"] * 9
    assert driver.kinds == (
        ["fallback"] + ["record"] * 10
        + (["fallback"] * 64 + round_of_recordings) * 3
        + ["fallback"] * 3
    )
    report = driver.site.report()
    assert report["programs_refused"] == {"signature_unstable": 1}
    assert report["rechecks"] == 32 and report["rests"] == 3
    assert report["fallback"]["resting"] == 3 * 64
    assert report["native"] == 0 and engine.runs == 0  # nothing was ever sealed or replayed
    # The refusal names the kernels that make the difference, so the server
    # log alone says whether it was a bank copy or something else.
    detail = driver.site.last_refusal["detail"]
    assert "launches against" in detail and "fake_kernel_" in detail and "more" in detail


def test_the_launch_difference_names_what_the_longer_recording_adds():
    from mtplx.engine.site import launch_difference

    engine = FakeEngine(_fn)
    engine.launches_by_build = [4, 7, 4]
    recordings = []
    real_build = engine.build

    def build(*args, **kwargs):
        ir = real_build(*args, **kwargs)
        recordings.append(ir)
        return ir

    engine.build = build
    driver = _Driver(_site(engine))
    for step in range(5):
        driver.call("k", _inputs(step))
    short, long, again = recordings[:3]
    # Four launches alternate two kernels; seven add two of one and one of the other.
    assert launch_difference(short, long) == "fake_kernel_0 2 more, fake_kernel_1 1 more"
    assert launch_difference(long, short) == "fake_kernel_0 2 fewer, fake_kernel_1 1 fewer"
    assert launch_difference(short, again) == ""
    assert launch_difference(short, long, limit=1) == "fake_kernel_0 2 more, and 1 other kernels"


def test_a_refused_build_is_counted_under_its_code():
    engine = FakeEngine(_fn)
    engine.build_refuse = "undeclared_external"
    driver = _Driver(_site(engine))
    for step in range(4):
        driver.call("k", _inputs(step))
    report = driver.site.report()
    assert report["programs_refused"] == {"undeclared_external": 1}
    assert report["fallback"]["refused:undeclared_external"] == 2
    assert "some_kernel" in report["last_refusal"]["detail"]


def test_a_call_the_executor_declines_goes_to_mlx_and_is_counted_with_its_reason():
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("k", _inputs(step))
    engine.can_run_reason = "state_shared"
    driver.call("k", _inputs(6))
    engine.can_run_reason = ""
    driver.call("k", _inputs(7))
    assert driver.kinds[-3:] == ["native", "fallback", "native"]
    assert driver.site.report()["fallback"]["state_shared"] == 1


def test_a_failed_run_refuses_the_key_and_mlx_serves_the_call():
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("k", _inputs(step))
    assert driver.kinds[-1] == "native"
    engine.run_raises = True
    driver.call("k", _inputs(6))
    engine.run_raises = False
    driver.call("k", _inputs(7))
    assert driver.kinds[-2:] == ["fallback", "fallback"]
    report = driver.site.report()
    assert report["programs_refused"] == {"run_error": 1} and report["programs_live"] == 0


def test_a_lost_replay_that_wrote_its_inputs_ends_the_call_and_refuses_the_key():
    """Committed, no outputs, and the program writes its inputs in place: the
    live state is of unknown content, so nothing may serve the call."""

    from mtplx.engine.runtime import ReplayLost

    engine = FakeEngine(_fn, clobbered=(1,))
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("k", _inputs(step))
    assert driver.kinds[-1] == "native"
    real = engine.try_run

    def lost(program, inputs, **kwargs):
        raise ReplayLost("failed:timeout: no completion within 30000 ms", wrote_inputs=True)

    engine.try_run = lost
    with pytest.raises(ReplayLost):
        driver.site.begin("k", _inputs(6))
    engine.try_run = real
    report = driver.site.report()
    assert report["programs_refused"] == {"replay_lost_timeout": 1} and report["programs_live"] == 0
    assert report["fallback"]["replay_lost"] == 1
    # The key stays refused for the life of the process.
    driver.call("k", _inputs(7))
    assert driver.kinds[-1] == "fallback"
    assert driver.site.report()["fallback"]["refused:replay_lost_timeout"] == 1


def test_a_lost_replay_that_wrote_nothing_is_a_refused_key_and_mlx_serves():
    from mtplx.engine.runtime import ReplayLost

    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("k", _inputs(step))

    def lost(program, inputs, **kwargs):
        raise ReplayLost("failed:command_buffer: Metal said no", wrote_inputs=False)

    engine.try_run = lost
    driver.call("k", _inputs(6))
    assert driver.kinds[-1] == "fallback"
    report = driver.site.report()
    assert report["programs_refused"] == {"replay_lost_command_buffer": 1}
    assert report["fallback"]["refused:replay_lost_command_buffer"] == 1
    assert "replay_lost" not in report["fallback"]


def test_the_watched_wait_is_used_when_the_executor_has_one():
    engine = FakeEngine(_fn)
    asked = []

    def run_patient(program, inputs, *, extra_refs=1, slice_ms=0, limit_ms=0, should_stop=None):
        asked.append((slice_ms, limit_ms, should_stop))
        return engine.try_run(program, inputs, extra_refs=extra_refs)

    engine.run_patient = run_patient
    site = _site(engine)
    driver = _Driver(site)
    for step in range(5):
        driver.call("k", _inputs(step))
    assert driver.kinds[-1] == "native" and asked == [(2000, 30_000, None)]
    stop = lambda: False  # noqa: E731 - the request's cancel flag
    ticket = site.begin("k", _inputs(6), should_stop=stop)
    ticket.finish(ticket.outputs)
    assert ticket.native and asked[-1] == (2000, 30_000, stop)


def test_switch_off_never_touches_the_executor(monkeypatch):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE", "0")
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(5):
        driver.call("k", _inputs(step))
    assert driver.kinds == ["fallback"] * 5 and driver.mlx_calls == 5
    assert engine.recordings == 0 and engine.runs == 0 and engine.pinned == []
    report = driver.site.report()
    assert report["mode"] == "0" and report["fallback"] == {"disabled": 5}


def test_a_missing_module_keeps_every_call_on_mlx(monkeypatch):
    from mtplx.engine import runtime

    monkeypatch.setattr(runtime, "available", lambda: False)
    monkeypatch.setattr(runtime, "unavailable", lambda: "module_missing")
    driver = _Driver(_site(None))
    for step in range(3):
        driver.call("k", _inputs(step))
    report = driver.site.report()
    assert report["fallback"] == {"unavailable": 3}
    assert report["unavailable"] == "module_missing" and report["available"] is False


def test_shadow_mode_runs_both_every_round_and_reports_a_mismatch(monkeypatch):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE", "shadow")
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(9):
        if step == 7:
            engine.wrong_on_run = engine.runs + 1
        driver.call("k", _inputs(step))
    assert driver.kinds == WALK + ["shadow"] * 5
    assert driver.mlx_calls == 9  # MLX serves every round
    report = driver.site.report()
    assert report["native"] == 0
    assert report["shadow_checks"] == 5 and report["shadow_mismatches"] == 1
    assert report["fallback"]["shadow"] == 5
    assert report["last_refusal"]["code"] == "shadow_mismatch"
    assert any("SHADOW MISMATCH" in line for line in driver.site.lines)


def test_the_check_replays_on_private_copies_of_inputs_the_program_writes_into():
    engine = FakeEngine(_fn, clobbered=(1,))
    driver = _Driver(_site(engine))
    for step in range(4):
        inputs = _inputs(step)
        driver.call("k", inputs)
    assert driver.kinds[-1] == "verify"
    check_inputs = engine.run_inputs[0]
    # Input 0 is read only: the same array. Input 1 is written in place by the
    # program: the check got a copy with the same bits.
    assert check_inputs[0] is inputs[0]
    assert check_inputs[1] is not inputs[1]
    assert compare_outputs([check_inputs[1]], [inputs[1]]) is None
    # A native round binds the live arrays themselves.
    live = _inputs(5)
    driver.call("k", live)
    assert engine.run_inputs[-1][0] is live[0] and engine.run_inputs[-1][1] is live[1]


def test_a_check_that_does_not_fit_in_memory_waits(monkeypatch):
    engine = FakeEngine(_fn, clobbered=(1,))
    driver = _Driver(_site(engine))
    asked: list[int] = []

    def refuse(nbytes):
        asked.append(nbytes)
        return False

    for step in range(3):
        driver.call("k", _inputs(step))
    driver.call("k", _inputs(4), admit=refuse)
    driver.call("k", _inputs(5), admit=refuse)
    assert asked == [16, 16]  # the bytes of the one input it would copy
    assert driver.kinds[-2:] == ["fallback", "fallback"] and engine.runs == 0
    driver.call("k", _inputs(6), admit=lambda nbytes: True)
    driver.call("k", _inputs(7))
    assert driver.kinds[-2:] == ["verify", "native"]
    assert driver.site.report()["fallback"]["verify_memory"] == 2


def test_dump_writes_the_program_and_a_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_DUMP", str(tmp_path))
    engine = FakeEngine(_fn, clobbered=(1,))
    driver = _Driver(_site(engine))
    for step in range(4):
        driver.call(("verify", 4), _inputs(step))
    (ir_path,) = tmp_path.glob("*.ir.json")
    (summary_path,) = tmp_path.glob("*.summary.json")
    obj = json.loads(ir_path.read_text())
    assert len(obj["steps"]) == 3 and obj["steps"][0]["kernel_name"] == "fake_kernel_0"
    const = next(buf for buf in obj["buffers"] if buf["role"] == "const")
    assert "data" not in const and const["data_len"] == 1024  # bytes are not dumped
    summary = json.loads(summary_path.read_text())
    assert summary["launches"] == 3
    assert summary["kernel_counts"] == {"fake_kernel_0": 2, "fake_kernel_1": 1}
    assert summary["inputs_written_in_place"][0]["input"] == 1
    # The key is the caller's geometry and the layout of the inputs.
    assert summary["key"].startswith("(('verify', 4), ") and summary["site"] == "test_site"
    assert not list(tmp_path.glob("*.profile.json"))


def test_record_only_dumps_and_profiles_once_and_never_replays(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_DUMP", str(tmp_path))
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_REPLAY", "0")
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_PROFILE", "1")
    engine = FakeEngine(_fn, clobbered=(1,))
    driver = _Driver(_site(engine))
    for step in range(6):
        inputs = _inputs(step)
        driver.call("k", inputs)
    assert driver.kinds == ["fallback", "record"] + ["fallback"] * 4
    assert driver.mlx_calls == 6 and engine.runs == 0 and engine.profiles == 1
    # The profile ran on a private copy of the input the program writes into.
    assert engine.run_inputs[0][1] is not inputs[1]
    report = driver.site.report()
    assert report["replay"] is False and report["native"] == 0
    assert report["fallback"] == {"warming": 1, "recording": 1, "record_only": 4}
    profile = json.loads(next(tmp_path.glob("*.profile.json")).read_text())
    assert profile["launches"] == 3 and profile["gpu_ns"] == 6000
    assert list(profile["by_kernel"]) == ["fake_kernel_0", "fake_kernel_1"]


def test_profile_with_replay_on_is_written_at_the_check(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_DUMP", str(tmp_path))
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_PROFILE", "1")
    engine = FakeEngine(_fn, clobbered=(1,))
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("k", _inputs(step))
    assert driver.kinds == WALK + ["native", "native"]
    assert engine.profiles == 1 and len(list(tmp_path.glob("*.profile.json"))) == 1


def test_a_call_that_never_finishes_does_not_leave_a_recording_open():
    engine = FakeEngine(_fn)
    site = _site(engine)
    driver = _Driver(site)
    driver.call("k", _inputs(0))
    ticket = site.begin("k", _inputs(1))
    assert ticket.kind == "record" and engine.recording is not None
    # The MLX path raised: finish is never called. The next call recovers.
    driver.call("k", _inputs(2))
    assert engine.recording is None or driver.kinds[-1] == "record"
    for step in range(3, 7):
        driver.call("k", _inputs(step))
    assert driver.kinds[-1] == "native"


def test_a_recorder_that_fails_refuses_the_key():
    engine = FakeEngine(_fn)
    engine.record_raises = True
    driver = _Driver(_site(engine))
    for step in range(3):
        driver.call("k", _inputs(step))
    assert driver.site.report()["programs_refused"] == {"record_error": 1}


def test_only_the_most_recent_programs_are_kept(monkeypatch):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_PROGRAMS", "1")
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("a", _inputs(step))
    for step in range(6):
        driver.call("b", _inputs(step, width=6))
    report = driver.site.report()
    assert report["programs_evicted"] == 1 and report["programs_held"] == 1
    assert report["programs_live"] == 1
    # The evicted key starts over; nothing wrong is ever served. Its first
    # recording launches what its program launched while it was live, so it
    # is the steady program at once: one recording, the check, native.
    for step in range(6):
        driver.call("a", _inputs(step))
    assert driver.kinds[-6:] == ["fallback", "record", "verify"] + ["native"] * 3
    assert driver.site.report()["steady_by_kin"] == 1


def test_mixed_mode_serves_live_rounds_from_either_path(monkeypatch):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_MIXED", "7")
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(40):
        driver.call("k", _inputs(step))
    live = driver.kinds[len(WALK):]
    assert 5 < live.count("native") < 30 and 5 < live.count("fallback") < 30
    assert driver.site.report()["fallback"]["mixed"] == live.count("fallback")


def test_report_since_a_snapshot_is_one_requests_share():
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(5):
        driver.call("k", _inputs(step))
    before = driver.site.snapshot()
    for step in range(5, 9):
        driver.call("k", _inputs(step))
    request = driver.site.report(since=before)
    assert request["calls"] == 4 and request["native"] == 4 and request["fallback"] == {}
    assert request["native_share"] == 1.0 and request["programs_built"] == 0
    assert driver.site.report()["calls"] == 9


def test_pins_reach_the_executor_before_the_first_recording():
    engine = FakeEngine(_fn)
    site = _site(engine)
    weights = [mx.ones((2, 2))]
    site.pin(weights)
    driver = _Driver(site)
    driver.call("k", _inputs(0))
    assert engine.pinned == []
    driver.call("k", _inputs(1))
    assert engine.pinned == weights


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_a_private_copy_keeps_every_bit(dtype):
    values = mx.array([0.0, -0.0, 1.5, float("nan"), float("inf"), -3.25]).astype(dtype)
    copied = private_copy(values)
    mx.eval(values, copied)
    assert copied.dtype == values.dtype and copied.shape == values.shape
    assert compare_outputs([copied], [values]) is None


def test_private_copies_of_integers_booleans_and_scalars():
    for values in (
        mx.array([[1, -2, 3]], dtype=mx.int32),
        mx.array([True, False]),
        mx.array(7, dtype=mx.int32),
        mx.array(2.5, dtype=mx.float32),
        mx.array([1, 2], dtype=mx.uint8),
    ):
        copied = private_copy(values)
        mx.eval(copied)
        assert copied.dtype == values.dtype and copied.shape == values.shape
        assert compare_outputs([copied], [values]) is None


def test_the_comparison_sees_a_signed_zero_and_a_different_nan():
    plus = mx.array([0.0, 1.0], dtype=mx.float32)
    minus = mx.array([-0.0, 1.0], dtype=mx.float32)
    assert bool(mx.array_equal(plus, minus).item())  # value equality cannot tell
    difference = compare_outputs([minus], [plus])
    assert difference["output"] == 0 and difference["elements_differing"] == 1
    nan_a = mx.array([0x7FC00000], dtype=mx.uint32).view(mx.float32)
    nan_b = mx.array([0x7FC00001], dtype=mx.uint32).view(mx.float32)
    assert compare_outputs([nan_a], [nan_a]) is None
    assert compare_outputs([nan_a], [nan_b])["elements_differing"] == 1
    shaped = compare_outputs([mx.zeros((2, 2))], [mx.zeros((4,))])
    assert shaped["why"].startswith("shape or dtype")


def test_the_same_shapes_at_another_layout_are_another_program():
    """A state leaf arriving as a view of a longer buffer launches other
    kernels, so the layout is part of the key and each form has its program."""

    engine = FakeEngine(_fn)
    layouts = {"form": "contiguous"}
    engine.layout_key = lambda arrays: hash((layouts["form"], len(arrays)))
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("k", _inputs(step))
    assert driver.kinds == WALK + ["native"] * 2
    layouts["form"] = "view"
    for step in range(6, 12):
        driver.call("k", _inputs(step))
    # The other layout is its own program, checked bit for bit like the first.
    # Its first recording launches what the live one launches, so it is the
    # steady program without a second recording.
    assert driver.kinds[6:] == ["fallback", "record", "verify"] + ["native"] * 3
    layouts["form"] = "contiguous"
    driver.call("k", _inputs(12))
    assert driver.kinds[-1] == "native"
    report = driver.site.report()
    assert report["programs_built"] == 2 and report["programs_live"] == 2
    assert report["programs_refused"] == {} and report["steady_by_kin"] == 1
    assert report["fallback"] == {"warming": 1, "key_new": 1, "recording": 3, "verifying": 2}


def test_a_check_the_executor_cannot_run_now_is_tried_again_then_refused():
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(3):
        driver.call("k", _inputs(step))
    engine.can_run_reason = "state_shared"
    driver.call("k", _inputs(3))
    engine.can_run_reason = ""
    for step in range(4, 6):
        driver.call("k", _inputs(step))
    assert driver.kinds == WALK[:3] + ["fallback", "verify", "native"]
    assert driver.site.report()["fallback"]["check_blocked:state_shared"] == 1

    stuck = FakeEngine(_fn)
    driver = _Driver(_site(stuck))
    for step in range(3):
        driver.call("k", _inputs(step))
    stuck.can_run_reason = "input_alias"
    for step in range(3, 7):
        driver.call("k", _inputs(step))
    report = driver.site.report()
    assert report["programs_refused"] == {"check_illegal": 1}
    assert report["fallback"]["check_blocked:input_alias"] == 2


def test_two_recordings_at_different_input_layouts_are_not_held_to_one_signature():
    """MLX hands some kernels the stride of an extent-1 dimension, and it
    follows the array's history. The bit check is then the proof."""

    engine = FakeEngine(_fn)
    engine.unstable_inline = True  # the second recording has other argument bytes
    built = engine.build

    def build(trace, **kwargs):
        ir = built(trace, **kwargs)
        # The second recording's first input carries another extent-1 stride.
        ir.inputs[0].strides = (4 * engine.builds, 1)
        return ir

    engine.build = build
    driver = _Driver(_site(engine))
    for step in range(5):
        driver.call("k", _inputs(step))
    assert driver.kinds == WALK + ["native"]
    report = driver.site.report()
    assert report["stability_not_comparable"] == 1 and report["programs_refused"] == {}


def test_the_report_times_native_calls():
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(4):
        driver.call("k", _inputs(step))
    base = driver.site.snapshot()
    for step in range(4, 24):
        driver.call("k", _inputs(step))
    timing = driver.site.report(since=base)["timing_ms"]
    assert timing["samples"] == 20
    # The stand-in reports 9000 ns of wall, 5000 of GPU and 2000 of encode.
    assert timing["engine_wall"] == {"mean": 0.009, "p50": 0.009, "p95": 0.009}
    assert timing["gpu"]["p95"] == 0.005 and timing["encode"]["mean"] == 0.002
    assert timing["wait"]["p50"] == 0.002  # wall less GPU less encode
    for phase in ("pre", "inputs", "site", "post", "whole"):
        assert timing[phase]["p95"] >= timing[phase]["p50"] >= 0
    assert timing["whole"]["p50"] >= timing["engine_wall"]["p50"]
    assert driver.site.report()["timing_ms"]["samples"] == 20
    assert driver.site.report(since=driver.site.snapshot()).get("timing_ms") is None
    # The replay of the first-use check is timed too, apart.
    check = driver.site.report()["check_timing_ms"]
    assert check["samples"] == 1 and check["gpu"]["mean"] == 0.005 and "site" not in check


def test_shadow_rounds_report_the_replays_times(monkeypatch):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE", "shadow")
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(9):
        driver.call("k", _inputs(step))
    report = driver.site.report()
    assert report["native"] == 0 and "timing_ms" not in report
    assert report["check_timing_ms"]["samples"] == report["shadow_checks"] + 1 == 6


def test_by_default_a_key_is_recorded_from_its_third_call(monkeypatch):
    """The call that traces the graph and compiles its kernels, one settled
    call, then the recordings: a new step is not left on MLX for dozens of
    rounds to show that it is long-lived."""

    monkeypatch.delenv("MTPLX_NATIVE_ENGINE_WARM_CALLS")
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    warm = site_module.WARM_CALLS
    assert warm == 2
    for step in range(warm + 5):
        driver.call("k", _inputs(step))
    assert driver.kinds == ["fallback"] * warm + WALK[1:] + ["native"] * 2
    assert engine.recordings == 2 and driver.site.report()["fallback"]["warming"] == warm


def test_more_warm_calls_put_off_the_recording(monkeypatch):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_WARM_CALLS", "3")
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(8):
        driver.call("k", _inputs(step))
    assert driver.kinds == ["fallback", "fallback"] + WALK + ["native"] * 2


def test_a_requests_first_rounds_are_never_recorded_or_checked():
    """A request's first text must not wait for a recording or a first-use check."""

    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    # A new geometry, already past its warm calls, at the start of a request.
    for step in range(5):
        driver.call("k", _inputs(step), quiet=True)
    assert driver.kinds == ["fallback"] * 5 and engine.recordings == 0 and engine.runs == 0
    assert driver.site.report()["fallback"] == {"warming": 1, "request_start": 4}
    # Later in the request it is recorded...
    driver.call("k", _inputs(5))
    assert driver.kinds[-1] == "record" and engine.recordings == 1
    # ...and no later step of its life falls in a request's first rounds either:
    # not the confirming recording (with its seal), not the check.
    for step, kind in ((6, "record"), (8, "verify")):
        driver.call("k", _inputs(step), quiet=True)
        assert driver.kinds[-1] == "fallback" and engine.runs == 0
        driver.call("k", _inputs(step + 1))
        assert driver.kinds[-1] == kind
    assert engine.recordings == 2
    driver.call("k", _inputs(12))
    assert driver.kinds[-1] == "native"
    # A live program serves a request's first rounds: that is the point of keeping it.
    mlx_before = driver.mlx_calls
    for step in range(13, 16):
        driver.call("k", _inputs(step), quiet=True)
    assert driver.kinds[-3:] == ["native"] * 3 and driver.mlx_calls == mlx_before
    assert driver.site.report()["fallback"]["request_start"] == 6


def test_programs_are_kept_by_the_memory_they_own(monkeypatch):
    engine = FakeEngine(_fn)
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("a", _inputs(step))
    report = driver.site.report()
    # The stand-in program owns a 4,096-byte constant, the 1,024 bytes the
    # recording holds of it, and three launches. Not its outputs (allocated
    # per run) and no temporaries (the executor's shared pool).
    owned = 4096 + 1024 + 3 * site_module._LAUNCH_HOST_BYTES
    assert report["programs_held"] == 1 and report["programs_budget_mb"] == 256
    assert driver.site._entries[next(iter(driver.site._entries))].owned_bytes == owned
    # Room for two such programs, not three: the least recently used goes.
    monkeypatch.setattr(site_module, "_DEFAULT_PROGRAM_BYTES", 2 * owned)
    for step in range(6):
        driver.call("b", _inputs(step, width=6))
    assert driver.site.report()["programs_held"] == 2
    driver.call("a", _inputs(9))  # "a" is the most recently used now
    for step in range(6):
        driver.call("c", _inputs(step, width=8))
    report = driver.site.report()
    assert report["programs_held"] == 2 and report["programs_evicted"] == 1
    assert driver.call("a", _inputs(10)).kind == "native"
    assert driver.call("b", _inputs(10, width=6)).kind == "fallback"  # starts again
    # One program larger than the whole budget is still kept: the newest always stays.
    monkeypatch.setattr(site_module, "_DEFAULT_PROGRAM_BYTES", 16)
    for step in range(6):
        driver.call("d", _inputs(step, width=10))
    assert driver.site.report()["programs_held"] == 1
    assert driver.call("d", _inputs(7, width=10)).kind == "native"


def test_the_program_budget_setting(monkeypatch):
    site = _site(FakeEngine(_fn))
    assert site._program_budget() == 256 << 20
    for raw, expected in (("512", 512 << 20), ("0", 1 << 20), ("junk", 256 << 20)):
        monkeypatch.setenv("MTPLX_NATIVE_ENGINE_PROGRAM_MB", raw)
        assert site._program_budget() == expected, raw


class _Lane(_Driver):
    """Calls a site the way a pipelining caller does: ``plan`` before anything
    is dispatched, ``begin`` only for a call the site takes. A call it leaves
    is the MLX path's alone ("mlx")."""

    def call(self, key, inputs, admit=None, quiet=False, family=None, kin=None):
        if not self.site.plan(key, quiet=quiet, family=family):
            self.mlx_calls += 1
            outputs = list(self.fn(list(inputs)))
            mx.eval(*outputs)
            self.kinds.append("mlx")
            return None
        expected = list(self.fn(list(inputs)))
        mx.eval(*expected)
        ticket = self.site.begin(key, list(inputs), admit=admit, quiet=quiet, planned=True, kin=kin)
        if ticket.native:
            outputs = ticket.outputs
        else:
            self.mlx_calls += 1
            outputs = list(self.fn(list(inputs)))
        ticket.finish(outputs)
        mx.eval(*outputs)
        assert compare_outputs(list(outputs), expected) is None
        self.kinds.append(ticket.kind)
        return ticket


def test_the_site_says_before_the_dispatch_which_calls_it_takes(monkeypatch):
    """A call the site will not serve, record, seal or check never reaches it:
    the caller's own sequence runs untouched, and the call is still counted."""

    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_WARM_CALLS", "3")
    engine = FakeEngine(_fn)
    site = _site(engine)
    lane = _Lane(site)
    begun = []
    real_begin = site.begin
    site.begin = lambda *args, **kwargs: (begun.append(1), real_begin(*args, **kwargs))[1]
    for step in range(12):
        lane.call("g", _inputs(step))
    assert lane.kinds == ["mlx"] * 3 + ["record", "record", "verify"] + ["native"] * 6
    assert len(begun) == 9 and engine.settles == 2  # nothing of the executor in the warm calls
    report = site.report()
    assert report["calls"] == 12 and report["native"] == 6
    assert report["fallback"] == {"warming": 3, "recording": 2, "verifying": 1}
    # A second geometry's first call is counted as a new key.
    lane.call("h", _inputs(0, width=6))
    assert lane.kinds[-1] == "mlx" and site.report()["fallback"]["key_new"] == 1


def test_a_new_capacity_of_a_step_that_served_live_waits_one_call_not_its_warm_calls(monkeypatch):
    """A growing session crosses a bank capacity edge: the same step at a new
    capacity is a new geometry. Once a geometry of its family has served a live
    program, the new one waits only the call that grew the bank; recording
    and the first-use check are unchanged (this caller names no kin), and so
    is any geometry of a family that never was live, or that names no family."""

    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_WARM_CALLS", "6")
    engine = FakeEngine(_fn)
    lane = _Lane(_site(engine))
    life = ["record", "record", "verify"]
    for step in range(12):
        lane.call(("step", 512), _inputs(step), family="step")
    assert lane.kinds == ["mlx"] * 6 + life + ["native"] * 3
    # The session grows into the next capacity.
    for step in range(12, 20):
        lane.call(("step", 1024), _inputs(step), family="step")
    assert lane.kinds[12:] == ["mlx"] + life + ["native"] * 4
    report = lane.site.report()
    assert report["fallback"] == {"warming": 6, "key_new": 1, "recording": 4, "verifying": 2}
    assert report["native"] == 7 and report["programs_live"] == 2
    # Another family, and a call that names none, still wait their warm calls.
    for family, key in (("copy", ("copy", 1024)), (None, ("plain", 1024))):
        before = len(lane.kinds)
        for step in range(8):
            lane.call(key, _inputs(step, width=6), family=family)
        assert lane.kinds[before:] == ["mlx"] * 6 + ["record", "record"]


def test_a_new_capacity_of_a_live_step_still_leaves_an_answers_first_tokens_alone(monkeypatch):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_WARM_CALLS", "6")
    engine = FakeEngine(_fn)
    lane = _Lane(_site(engine))
    for step in range(12):
        lane.call(("step", 512), _inputs(step), family="step")
    assert lane.kinds[-1] == "native"
    for step in range(12, 16):
        lane.call(("step", 1024), _inputs(step), quiet=True, family="step")
    assert lane.kinds[12:] == ["mlx"] * 4 and engine.recordings == 2
    assert lane.site.report()["fallback"]["request_start"] == 3
    lane.call(("step", 1024), _inputs(16), family="step")
    assert lane.kinds[-1] == "record"


def test_how_long_an_answers_start_is_depends_on_whether_the_step_has_run_long(monkeypatch):
    """The caller's ``quiet`` comes from the site. A step never seen live keeps
    a short answer from paying for it (96 tokens); once a geometry of its
    family has served live, the next capacity is recorded as soon as the
    request's first rounds are out. A conversation of short tool turns
    otherwise never got a program for the capacity it was in."""

    monkeypatch.delenv("MTPLX_NATIVE_ENGINE_QUIET_TOKENS", raising=False)
    lane = _Lane(_site(FakeEngine(_fn)))
    site = lane.site
    quiet = site_module._FAMILY_QUIET_TOKENS
    assert 0 < quiet < site_module.QUIET_TOKENS
    for tokens, early in ((0, True), (quiet, True), (95, True), (96, False)):
        assert site.quiet_start(tokens, "step") is early
        assert site.quiet_start(tokens) is early
    for step in range(6):
        lane.call(("step", 512), _inputs(step), family="step")
    assert lane.kinds[-1] == "native"
    for tokens, early in ((0, True), (quiet - 1, True), (quiet, False), (95, False)):
        assert site.quiet_start(tokens, "step") is early
    # Another family, and a call that names none, keep the long wait.
    assert site.quiet_start(quiet, "copy") and site.quiet_start(quiet)
    # The setting is one number for both, as the tests and a rollback use it.
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_QUIET_TOKENS", "96")
    assert site.quiet_start(95, "step") and not site.quiet_start(96, "step")
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_QUIET_TOKENS", "0")
    assert not site.quiet_start(0, "step") and not site.quiet_start(0, "copy")


def test_a_short_turn_at_a_new_capacity_of_a_live_step_gets_its_program(monkeypatch):
    """A 64-token answer, about twenty rounds, in a capacity the session just
    grew into: the request's first round (the one that grew the bank) is
    MLX's, then the program is recorded, sealed, checked and serves the rest:
    four rounds on MLX, sixteen replayed."""

    monkeypatch.delenv("MTPLX_NATIVE_ENGINE_QUIET_TOKENS", raising=False)
    lane = _Lane(_site(FakeEngine(_fn)))
    site = lane.site
    for step in range(6):
        lane.call(("step", 512), _inputs(step), family="step")
    written = 0
    for step in range(6, 26):
        quiet = site.quiet_start(written, "step")
        lane.call(("step", 1024), _inputs(step), quiet=quiet, family="step")
        written += 3  # tokens a round commits
    assert lane.kinds[6:] == ["mlx"] + ["record", "record", "verify"] + ["native"] * 16
    # The same turn in a family that never was live: once its geometry has had
    # its warm calls (one here) it has run in this process, so it waits a live
    # family's one round and gets its program in the same short turn. An
    # agent's stream of 50 to 200-token tool turns otherwise never records a
    # geometry it first meets there (V3 night 2 baseline: request_start rounds).
    written = 0
    before = len(lane.kinds)
    for step in range(20):
        quiet = site.quiet_start(written, "copy")
        lane.call(("copy", 1024), _inputs(step, width=6), quiet=quiet, family="copy")
        written += 3
    assert lane.kinds[before:] == ["mlx"] + ["record", "record", "verify"] + ["native"] * 16


def test_a_short_turn_before_the_warm_calls_stays_on_mlx(monkeypatch):
    """A family neither live nor past its warm calls keeps the long wait: a
    geometry seen for the first time in a short answer is not recorded in it."""

    monkeypatch.delenv("MTPLX_NATIVE_ENGINE_QUIET_TOKENS", raising=False)
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_WARM_CALLS", "24")
    lane = _Lane(_site(FakeEngine(_fn)))
    site = lane.site
    written = 0
    for step in range(20):
        quiet = site.quiet_start(written, "copy")
        lane.call(("copy", 1024), _inputs(step, width=6), quiet=quiet, family="copy")
        written += 3
    assert lane.kinds == ["mlx"] * 20
    assert set(lane.site.report()["fallback"]) <= {"warming", "key_new"}


def test_the_plan_leaves_an_answers_first_tokens_alone_unless_a_program_is_live():
    engine = FakeEngine(_fn)
    lane = _Lane(_site(engine))
    for step in range(4):
        lane.call("g", _inputs(step), quiet=True)
    assert lane.kinds == ["mlx"] * 4 and engine.recordings == 0
    assert lane.site.report()["fallback"] == {"warming": 1, "request_start": 3}
    # Every later step of the geometry's life waits out a quiet call too.
    for step, kind in ((4, "record"), (6, "record"), (8, "verify")):
        lane.call("g", _inputs(step))
        assert lane.kinds[-1] == kind
        lane.call("g", _inputs(step + 1), quiet=True)
        assert lane.kinds[-1] == ("mlx" if kind != "verify" else "native")
    # Live: the site takes the first tokens of the next answers.
    for step in range(12, 15):
        lane.call("g", _inputs(step), quiet=True)
    assert lane.kinds[-3:] == ["native"] * 3
    assert lane.site.report()["fallback"]["request_start"] == 5


def test_a_refused_geometry_is_left_to_the_callers_own_sequence():
    engine = FakeEngine(_fn)
    engine.wrong_on_run = 1  # the check's replay
    lane = _Lane(_site(engine))
    for step in range(9):
        lane.call("g", _inputs(step))
    assert lane.kinds == ["mlx", "record", "record", "verify"] + ["mlx"] * 5
    report = lane.site.report()
    assert report["programs_refused"] == {"bit_mismatch": 1} and report["native"] == 0
    assert report["fallback"]["refused:bit_mismatch"] == 5 and report["calls"] == 9


def test_the_plan_with_the_switch_off_or_no_module(monkeypatch):
    engine = FakeEngine(_fn)
    lane = _Lane(_site(engine))
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE", "0")
    for step in range(3):
        lane.call("g", _inputs(step))
    assert lane.kinds == ["mlx"] * 3 and engine.recordings == 0 and engine.settles == 0
    assert lane.site.report()["fallback"] == {"disabled": 3}

    from mtplx.engine import runtime

    monkeypatch.delenv("MTPLX_NATIVE_ENGINE")
    monkeypatch.setattr(runtime, "available", lambda: False)
    monkeypatch.setattr(runtime, "unavailable", lambda: "module_missing")
    gone = _Lane(_site(None))
    for step in range(2):
        gone.call("g", _inputs(step))
    assert gone.kinds == ["mlx"] * 2 and gone.site.report()["fallback"] == {"unavailable": 2}


def test_an_evicted_program_makes_its_geometry_start_recording_again(monkeypatch):
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_PROGRAMS", "1")
    engine = FakeEngine(_fn)
    lane = _Lane(_site(engine))
    for step in range(6):
        lane.call("a", _inputs(step))
    for step in range(6):
        lane.call("b", _inputs(step, width=6))
    assert lane.kinds[-1] == "native" and lane.site.report()["programs_evicted"] == 1
    # "a" lost its program when "b" was sealed. Its warm calls are long past:
    # it is recorded at once (once: the recording launches what its program
    # launched while it was live), and nothing wrong is served meanwhile.
    for step in range(6, 11):
        lane.call("a", _inputs(step))
    assert lane.kinds[-5:] == ["record", "verify", "native", "native", "native"]


def test_a_recording_left_open_is_closed_before_a_call_the_site_leaves():
    engine = FakeEngine(_fn)
    site = _site(engine)
    lane = _Lane(site)
    lane.call("g", _inputs(0))
    assert site.plan("g") is True
    ticket = site.begin("g", _inputs(1), planned=True)
    assert ticket.kind == "record" and engine.recording is not None
    # The MLX path raised and ``finish`` never came. The next answer's first
    # call is one the site leaves alone: the recorder must not run over it.
    lane.call("g", _inputs(2), quiet=True)
    assert lane.kinds[-1] == "mlx" and engine.recording is None
    for step in range(3, 8):
        lane.call("g", _inputs(step))
    assert lane.kinds[-1] == "native"


def test_declined_counts_the_call_and_asks_nothing_of_the_caller(monkeypatch):
    engine = FakeEngine(_fn)
    site = _site(engine)
    assert site.declined() is False
    assert site.report()["calls"] == 0
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE", "0")
    assert site.declined() is True
    assert site.report()["calls"] == 1 and site.report()["fallback"] == {"disabled": 1}


def test_a_call_with_strided_inputs_is_left_to_mlx_and_counted():
    """The fence of 10 Nov: no program is recorded on inputs that are views."""

    from types import SimpleNamespace

    engine = FakeEngine(_fn)
    views = {"on": True}
    engine.layout_key = lambda arrays: hash(("view" if views["on"] else "plain", len(arrays)))
    engine.describe = lambda arrays: [
        SimpleNamespace(flags=0 if views["on"] and index == 1 else 3, nbytes=16, shape=tuple(a.shape), dtype=str(a.dtype))
        for index, a in enumerate(arrays)
    ]
    built = engine.build
    engine.build = lambda trace, *, inputs, outputs, name="": built(
        trace, inputs=[(i, r.shape, r.dtype) for i, r in enumerate(inputs)], outputs=outputs, name=name)
    driver = _Driver(_site(engine))
    for step in range(5):
        driver.call("k", _inputs(step))
    assert driver.kinds == ["fallback"] * 5 and engine.recordings == 0
    assert driver.site.report()["fallback"] == {"strided_inputs": 5}
    views["on"] = False
    for step in range(5, 11):
        driver.call("k", _inputs(step))
    assert driver.kinds[5:] == WALK + ["native"] * 2
    assert driver.site.report()["programs_refused"] == {}


def _strided_engine(views):
    from types import SimpleNamespace

    engine = FakeEngine(_fn)
    engine.layout_key = lambda arrays: hash(("view" if views["on"] else "plain", len(arrays)))
    engine.describe = lambda arrays: [
        SimpleNamespace(flags=0 if views["on"] and index == 1 else 3, nbytes=16, shape=tuple(a.shape), dtype=str(a.dtype))
        for index, a in enumerate(arrays)
    ]
    built = engine.build
    engine.build = lambda trace, *, inputs, outputs, name="": built(
        trace, inputs=[(i, r.shape, r.dtype) for i, r in enumerate(inputs)], outputs=outputs, name=name)
    return engine


def test_a_fenced_first_round_does_not_close_its_geometry_to_the_plan():
    """A request's first round arrives as views (fenced); the rounds after it
    are the steady layout, and the plan must still take them."""

    views = {"on": False}
    lane = _Lane(_site(_strided_engine(views)))
    lane.call("g", _inputs(0))  # the warm call
    views["on"] = True
    lane.call("g", _inputs(1))  # taken, and found fenced
    assert lane.kinds == ["mlx", "fallback"]
    views["on"] = False
    for step in range(2, 7):
        lane.call("g", _inputs(step))
    assert lane.kinds[2:] == ["record", "record", "verify", "native", "native"]
    # A later request's first round: the live program's geometry is taken,
    # the fenced layout is MLX's, and the round after is native again.
    views["on"] = True
    lane.call("g", _inputs(7), quiet=True)
    views["on"] = False
    lane.call("g", _inputs(8), quiet=True)
    assert lane.kinds[-2:] == ["fallback", "native"]
    assert lane.site.report()["fallback"]["strided_inputs"] == 2


def test_a_geometry_that_only_ever_arrives_fenced_is_left_to_the_callers_sequence():
    views = {"on": True}
    lane = _Lane(_site(_strided_engine(views)))
    for step in range(12):
        lane.call("g", _inputs(step))
    run = site_module._FENCED_RUN
    assert lane.kinds == ["mlx"] + ["fallback"] * run + ["mlx"] * (11 - run)
    assert lane.site.report()["fallback"] == {"warming": 1, "strided_inputs": 11}
    # Looked at again now and then: when the layout turns steady it is recorded.
    views["on"] = False
    for step in range(12, 12 + site_module._FENCED_RETRY + 6):
        lane.call("g", _inputs(step))
    assert lane.kinds[-1] == "native"


def test_site_module_exports():
    assert site_module.WARM_CALLS >= 1


def test_unset_struct_padding_is_not_a_difference_between_recordings():
    """MLX sends its matrix-product kernels a 72-byte struct with 68 bytes of
    fields and leaves the last four as the stack had them. Two recordings of
    one program that differ only there are the same program, in the
    comparison and in the hash; a difference in a field still is one."""

    import dataclasses

    from mtplx.engine.ir import inline_data
    from mtplx.engine.site import compare_structure

    engine = FakeEngine(_fn)
    recordings = []
    real_build = engine.build

    def build(*args, **kwargs):
        ir = real_build(*args, **kwargs)
        recordings.append(ir)
        return ir

    engine.build = build
    driver = _Driver(_site(engine))
    for step in range(3):
        driver.call("k", _inputs(step))
    base = recordings[0]

    def with_argument(ir, name, data):
        steps = [dataclasses.replace(ir.steps[0], kernel_name=name, inline=[(4, data)]), *ir.steps[1:]]
        return dataclasses.replace(ir, steps=steps)

    fields = bytes(range(1, 69))
    name = "steel_gemm_fused_nax_nn_float32_float32_bm64_bn128_bk256"
    one = with_argument(base, name, fields + b"\x7b\x00\x00\x00")
    two = with_argument(base, name, fields + b"\x00\x00\x00\x00")
    assert compare_structure(one, two) is None
    assert one.signature() == two.signature()
    assert inline_data(name, fields + b"\x7b\x00\x00\x00") == fields + bytes(4)
    # A field: still a difference.
    other = with_argument(base, name, bytes([9]) + fields[1:] + bytes(4))
    assert compare_structure(one, other) is not None
    assert one.signature() != other.signature()
    # Another kernel's 72 bytes, or this kernel with another length, are read whole.
    assert compare_structure(
        with_argument(base, "some_other_kernel", fields + b"\x7b\x00\x00\x00"),
        with_argument(base, "some_other_kernel", fields + bytes(4)),
    ) is not None
    assert inline_data(name, fields) == fields


def test_a_steady_program_seen_again_after_several_longer_recordings_is_sealed():
    """The shape of the real-pack refusal of 11 November: the first recording
    was the steady program and the next three carried bank copies. The fourth
    agreeing recording seals it."""

    engine = FakeEngine(_fn)
    engine.launches_by_build = [3, 5, 5, 4, 3]
    driver = _Driver(_site(engine))
    for step in range(9):
        driver.call("k", _inputs(step))
    assert driver.kinds == ["fallback"] + ["record"] * 5 + ["verify", "native", "native"]
    report = driver.site.report()
    assert report["rechecks"] == 3 and report["programs_refused"] == {}
    assert report["programs_live"] == 1
    assert len(driver.site._entries[next(iter(driver.site._entries))].ir.steps) == 3


def test_a_key_that_spent_its_rechecks_rests_and_is_sealed_when_the_steady_program_returns():
    """Eight longer recordings in a row put the key to rest, not to death:
    MLX serves 64 calls unrecorded, the next recording is the steady program
    again, and the key goes live."""

    engine = FakeEngine(_fn)
    engine.launches_by_build = [3] + [5] * 9 + [3]
    driver = _Driver(_site(engine))
    for step in range(1 + 10 + 64 + 4):
        driver.call("k", _inputs(step))
    assert driver.kinds == (
        ["fallback"] + ["record"] * 10 + ["fallback"] * 64 + ["record", "verify", "native", "native"]
    )
    report = driver.site.report()
    assert report["rests"] == 1 and report["rechecks"] == 8
    assert report["programs_refused"] == {} and report["programs_live"] == 1
    assert report["fallback"]["resting"] == 64


_COPY = "g1_copybfloat16bfloat16"
_GDN = "custom_kernel_mtplx_gdn_rows_fused_t512__bfloat16_t_float_4"


def test_recordings_of_one_length_with_a_copy_in_another_place_are_recorded_again():
    """The shape of the real-pack refusal of 12 November ("launch 546: kernel
    g1_copybfloat16bfloat16 against" a GDN kernel, then 2,377 rounds of one
    answer on MLX): two recordings of one length, each with one transient copy
    of another array. Neither is the steady program; the next clean recording
    is, and it is sealed instead of the key being refused. (A default site
    does not settle a pair on kernel names alone: ``settle_copies``.)"""

    steady = ["fake_kernel_0", _GDN, "fake_kernel_1"]
    engine = FakeEngine(_fn)
    engine.kernels_by_build = [
        ["fake_kernel_0", _COPY, _GDN, "fake_kernel_1"],  # copies the array the GDN kernel reads
        ["fake_kernel_0", _GDN, _COPY, "fake_kernel_1"],  # one length, the copy elsewhere
        steady,
        steady,
    ]
    driver = _Driver(_site(engine))
    for step in range(9):
        driver.call("k", _inputs(step))
    assert driver.kinds == ["fallback"] + ["record"] * 4 + ["verify", "native", "native", "native"]
    report = driver.site.report()
    assert report["programs_refused"] == {} and report["programs_live"] == 1
    # One recheck for the moved copy, one for the shorter steady recording,
    # which replaced the first.
    assert report["rechecks"] == 2 and report["programs_rerecorded"] == 1
    assert "copies_settled" not in report
    entry = driver.site._entries[next(iter(driver.site._entries))]
    assert [step.kernel_name for step in entry.ir.steps] == steady


def test_a_new_capacity_of_a_live_step_is_recorded_once_when_it_launches_the_same_kernels():
    """A caller whose key carries the bank's capacity names the step without
    it as the key's kin. The second recording of a key only asks whether the
    first carries a copy the steady path does not make; a live program of the
    same step answers that, so the new capacity costs the round that grew the
    bank, one recording and the bit check."""

    engine = FakeEngine(_fn)
    lane = _Lane(_site(engine))
    for step in range(6):
        lane.call(("step", 512), _inputs(step), family="step", kin="step")
    assert lane.kinds == ["mlx", "record", "record", "verify", "native", "native"]
    for step in range(6, 12):
        lane.call(("step", 1024), _inputs(step), family="step", kin="step")
    assert lane.kinds[6:] == ["mlx", "record", "verify", "native", "native", "native"]
    report = lane.site.report()
    assert report["steady_by_kin"] == 1 and report["programs_live"] == 2
    assert report["fallback"] == {"warming": 1, "key_new": 1, "recording": 3, "verifying": 2}
    # Another step is not vouched for by this one.
    for step in range(5):
        lane.call(("other", 512), _inputs(step, width=6), kin="other")
    assert lane.kinds[-5:] == ["mlx", "record", "record", "verify", "native"]


def test_a_recording_with_a_copy_the_live_step_does_not_make_is_recorded_again():
    """The round after a bank grew can copy it whole. That recording is one
    launch longer than the live program of the step: it is held, not sealed,
    and the next recording, which launches what the live one launches, is."""

    steady = ["fake_kernel_0", _GDN, "fake_kernel_1"]
    engine = FakeEngine(_fn)
    engine.kernels_by_build = [steady, steady, ["fake_kernel_0", _COPY, _GDN, "fake_kernel_1"], steady]
    lane = _Lane(_site(engine))
    for step in range(5):
        lane.call(("step", 512), _inputs(step), family="step", kin="step")
    assert lane.kinds == ["mlx", "record", "record", "verify", "native"]
    for step in range(5, 11):
        lane.call(("step", 1024), _inputs(step), family="step", kin="step")
    assert lane.kinds[5:] == ["mlx", "record", "record", "verify", "native", "native"]
    report = lane.site.report()
    assert report["steady_by_kin"] == 1 and report["programs_rerecorded"] == 1
    assert report["programs_refused"] == {} and report["programs_live"] == 2
    newest = list(lane.site._entries.values())[-1]
    assert [step.kernel_name for step in newest.ir.steps] == steady


def test_a_program_its_kin_vouches_for_is_still_checked_bit_for_bit():
    """What a live program of the step settles is that a recording is the
    steady one. Whether it computes the right bits is the checked call's, and
    a replay that differs refuses the key as it always did."""

    engine = FakeEngine(_fn)
    lane = _Lane(_site(engine))
    for step in range(5):
        lane.call(("step", 512), _inputs(step), family="step", kin="step")
    engine.wrong_on_run = engine.runs + 1  # the next replay: the new capacity's check
    for step in range(5, 10):
        lane.call(("step", 1024), _inputs(step), family="step", kin="step")
    assert lane.kinds[5:] == ["mlx", "record", "verify", "mlx", "mlx"]
    report = lane.site.report()
    assert report["programs_refused"] == {"bit_mismatch": 1} and report["programs_live"] == 1
    # The capacity that was live still serves.
    lane.call(("step", 512), _inputs(10), family="step", kin="step")
    assert lane.kinds[-1] == "native"


def test_recordings_of_one_length_that_keep_moving_a_copy_rest_before_any_refusal():
    """Every recording copies another array (the copy at another launch each
    time, never twice in one place): the budget is spent, the key rests and is
    not refused."""

    engine = FakeEngine(_fn)
    engine.kernels_by_build = [
        [_GDN] * position + [_COPY] + [_GDN] * (12 - position) for position in range(12)
    ]
    driver = _Driver(_site(engine))
    for step in range(1 + 10 + 64):
        driver.call("k", _inputs(step))
    report = driver.site.report()
    assert report["programs_refused"] == {}
    assert report["rests"] == 1 and report["fallback"]["resting"] == 64


def test_recordings_of_one_length_that_differ_in_a_kernel_that_is_no_copy_refuse_the_key():
    engine = FakeEngine(_fn)
    engine.kernels_by_build = [
        ["fake_kernel_0", _GDN, "fake_kernel_1"],
        ["fake_kernel_0", "fake_kernel_2", "fake_kernel_1"],
    ]
    driver = _Driver(_site(engine))
    for step in range(5):
        driver.call("k", _inputs(step))
    report = driver.site.report()
    assert report["programs_refused"] == {"signature_unstable": 1}
    assert "fake_kernel_2" in driver.site.last_refusal["detail"]


def test_copy_moved_reads_only_the_first_differing_kernel():
    from mtplx.engine.ir import ProgramIR, StepIR
    from mtplx.engine.site import copy_moved

    def program(*names):
        steps = [StepIR(kernel=1, kernel_name=name, bindings=[], inline=[],
                        dispatch=("threads", (1, 1, 1), (1, 1, 1))) for name in names]
        return ProgramIR(name="p", buffers=[], steps=steps, inputs=[], outputs=[])

    a = program("k0", "v_copybfloat16float32", "k1")
    assert copy_moved(a, program("k0", "k1", "gg2_dynamic_copybfloat16bfloat16")) == (
        "launch 1: v_copybfloat16float32 against k1"
    )
    assert copy_moved(program("k0", "k1"), program("k0", "gn4_copyfloat32float32")) is not None
    assert copy_moved(program("k0", "k1"), program("k0", "k2")) is None
    assert copy_moved(program("k0", "custom_kernel_mtplx_kv_copy_t"), program("k0", "k2")) is None
    assert copy_moved(a, a) is None


def test_recordings_that_differ_only_by_copies_settle_on_the_shorter_without_rests():
    """MLX copies a bank it cannot write in place because a command buffer it
    committed earlier in the same evaluation still holds it: whether it does
    depends on how far the GPU has got. On a site that opts in (the round
    program's), recordings that alternate between the two are one program;
    each pair settles on the shorter (in place) at once, whichever of the two
    came first, and no key ever rests. Only kernel names were compared, so the
    program is bit-checked on two calls before it goes live."""

    steady = ["fake_kernel_0", _GDN, "fake_kernel_1"]
    copying = ["fake_kernel_0", _COPY, _GDN, _COPY, "fake_kernel_1"]
    for builds in ([copying, steady], [steady, copying]):
        engine = FakeEngine(_fn)
        engine.kernels_by_build = list(builds) * 4
        driver = _Driver(_site(engine, settle_copies=True))
        for step in range(7):
            driver.call("k", _inputs(step))
        assert driver.kinds == ["fallback", "record", "record", "verify", "verify", "native", "native"], builds
        report = driver.site.report()
        assert report["copies_settled"] == 1 and report["copies_settled_rechecked"] == 1, report
        assert "rests" not in report and "rechecks" not in report, report
        entry = driver.site._entries[next(iter(driver.site._entries))]
        assert [step.kernel_name for step in entry.ir.steps] == steady


def test_a_length_difference_that_is_not_only_copies_is_still_checked_again():
    """Only MLX's copies settle a pair: a recording that launches another
    kernel as well is the program's own difference and takes the old patience."""

    steady = ["fake_kernel_0", _GDN, "fake_kernel_1"]
    other = ["fake_kernel_0", _COPY, _GDN, "fake_kernel_2", "fake_kernel_1"]
    engine = FakeEngine(_fn)
    engine.kernels_by_build = [other, steady, steady]
    driver = _Driver(_site(engine, settle_copies=True))
    for step in range(6):
        driver.call("k", _inputs(step))
    report = driver.site.report()
    assert "copies_settled" not in report and report["rechecks"] == 1, report


def test_a_site_that_does_not_opt_in_never_settles_on_kernel_names():
    """Every site but the round program's (Flash-Next's and the 35B verify
    among them) keeps the full structural rule: a copies-only pair is checked
    again, as before ``copies_only`` existed."""

    steady = ["fake_kernel_0", _GDN, "fake_kernel_1"]
    copying = ["fake_kernel_0", _COPY, _GDN, _COPY, "fake_kernel_1"]
    engine = FakeEngine(_fn)
    engine.kernels_by_build = [copying, steady, steady]
    driver = _Driver(_site(engine))
    for step in range(6):
        driver.call("k", _inputs(step))
    report = driver.site.report()
    assert "copies_settled" not in report and report["rechecks"] == 1, report


def test_a_constant_and_a_seed_with_the_same_bytes_are_one_operand_only_where_a_site_opts_in():
    """The 35B round on the real pack (r5-round, boots 4 and 6): one operand of
    the verify norm-rotary kernel was a constant in one recording and a seed
    (the allocator handed its buffer to a later result) in the other, with the
    same bytes, and the key was refused as unstable. A site built with
    ``settle_copies`` compares such an operand by its bytes; every other site
    keeps telling the two roles apart."""

    from mtplx.engine.ir import BufferIR, ProgramIR, StepIR
    from mtplx.engine.site import compare_structure

    def recording(role, data, recycled):
        buffers = [BufferIR(id=1, role=role, nbytes=8, data=data), BufferIR(id=2, role="temp", nbytes=8)]
        grid = ("threads", (1, 1, 1), (1, 1, 1))
        steps = [
            StepIR(kernel=1, kernel_name="k0", bindings=[(0, 1, 0, "r"), (1, 2, 0, "w")], inline=[], dispatch=grid),
            # Later: a result lands in the constant's buffer (a seed) or in fresh memory.
            StepIR(kernel=2, kernel_name="k1", bindings=[(0, 1 if recycled else 2, 0, "w")], inline=[], dispatch=grid),
        ]
        return ProgramIR(name="p", buffers=buffers, steps=steps)

    const = recording("const", b"12345678", recycled=False)
    seed = recording("seed", b"12345678", recycled=True)
    other = recording("seed", b"87654321", recycled=True)
    assert compare_structure(const, seed) is not None
    assert compare_structure(const, seed, same_bytes=True) is None
    assert compare_structure(const, other, same_bytes=True) is not None


def _settle_pair(shorter_steps, longer_steps, consts=()):
    from mtplx.engine.ir import ArrayIR, BufferIR, ProgramIR

    def program(steps):
        buffers = [BufferIR(id=0, role="input", nbytes=64, index=0), BufferIR(id=1, role="temp", nbytes=64),
                   *[BufferIR(id=10 + i, role="const", nbytes=8, data=data) for i, data in enumerate(consts)]]
        return ProgramIR(name="p", buffers=buffers, steps=steps,
                         inputs=[ArrayIR(buffer=0, offset=0, shape=(32,), dtype="bfloat16")], outputs=[])

    return program(shorter_steps), program(longer_steps)


def _step(name, bindings, inline=()):
    from mtplx.engine.ir import StepIR

    return StepIR(kernel=1, kernel_name=name, bindings=list(bindings), inline=list(inline),
                  dispatch=("threads", (4, 1, 1), (4, 1, 1)))


_VN_COPY = "vn_copybfloat16bfloat16"


def test_a_copy_settles_only_when_the_rest_is_the_same_program():
    """Review 2, finding 14: the settle compared launch names only, so a one-launch recording
    that updates an input at offset 4 settled with a copy plus the same update at offset 5.
    Now the copy is removed, its destination read as its source, and the rest compared."""
    from mtplx.engine.site import copies_only

    copy = _step(_VN_COPY, [(0, 0, 0, "r"), (1, 1, 0, "w")])
    shorter, longer = _settle_pair([_step("k_update", [(0, 0, 4, "w")])],
                                   [copy, _step("k_update", [(0, 1, 5, "w")])])
    assert not copies_only(shorter, longer)
    shorter, longer = _settle_pair([_step("k_update", [(0, 0, 4, "w")])],
                                   [copy, _step("k_update", [(0, 1, 4, "w")])])
    assert copies_only(shorter, longer)  # the same update, written in place in one of them


def test_a_copy_does_not_settle_a_pair_whose_scalar_differs():
    """An operand frozen into a recording (an inline scalar, or a constant's bytes) that differs
    between the two recordings is not one program, copy or no copy."""
    from mtplx.engine.site import copies_only

    copy = _step(_VN_COPY, [(0, 0, 0, "r"), (1, 1, 0, "w")])
    shorter, longer = _settle_pair([_step("k_add", [(0, 0, 0, "r")], inline=[(3, b"\x01\x00\x00\x00")])],
                                   [copy, _step("k_add", [(0, 0, 0, "r")], inline=[(3, b"\x02\x00\x00\x00")])])
    assert not copies_only(shorter, longer)
    shorter, _ = _settle_pair([_step("k_add", [(0, 0, 0, "r"), (1, 10, 0, "r")])], [], consts=(b"\x01" * 8,))
    _, longer = _settle_pair([], [copy, _step("k_add", [(0, 0, 0, "r"), (1, 10, 0, "r")])], consts=(b"\x02" * 8,))
    assert not copies_only(shorter, longer)
    _, same = _settle_pair([], [copy, _step("k_add", [(0, 0, 0, "r"), (1, 10, 0, "r")])], consts=(b"\x01" * 8,))
    assert copies_only(shorter, same)


def test_a_launch_that_reads_the_copy_reads_its_source():
    """The longer recording reads the copy where the shorter reads the input itself: one program."""
    from mtplx.engine.site import copies_only

    copy = _step(_VN_COPY, [(0, 0, 0, "r"), (1, 1, 0, "w")])
    shorter, longer = _settle_pair([_step("k_read", [(0, 0, 8, "r")])],
                                   [copy, _step("k_read", [(0, 1, 8, "r")])])
    assert copies_only(shorter, longer)
    shorter, longer = _settle_pair([_step("k_read", [(0, 0, 8, "r")])],
                                   [copy, _step("k_read", [(0, 1, 16, "r")])])
    assert not copies_only(shorter, longer)
