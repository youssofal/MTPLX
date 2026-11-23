from mtplx import thermal
import subprocess
import time as _time

import pytest


@pytest.fixture(autouse=True)
def _default_no_daemon_socket(monkeypatch):
    """Default every test to "no ThermalForge daemon socket" so the suite never
    touches a real daemon on the dev machine. Socket-path tests opt back in by
    re-patching ``_daemon_socket_send``."""
    monkeypatch.setattr(thermal, "_daemon_socket_send", lambda *a, **k: None)


def test_detect_thermal_control_reports_none_without_tools(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    # Must mock both the PATH lookup AND the MTPLX-private bin lookup —
    # detect_thermal_control checks ``~/.mtplx/bin/thermalforge`` first via
    # ``_find_thermalforge`` so a real install on the dev machine would
    # otherwise leak into the test.
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: None)
    monkeypatch.setattr(thermal.shutil, "which", lambda _name: None)

    detected = thermal.detect_thermal_control()

    assert detected["available"] is False
    assert detected["selected"] is None
    assert "mtplx max --install" in detected["instructions"]
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_without_tool_is_actionable(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: None)
    monkeypatch.setattr(thermal.shutil, "which", lambda _name: None)

    result = thermal.set_thermal_profile("performance")

    assert result["ok"] is False
    assert result["profile"] == "performance"
    assert "mtplx max --install" in result["message"]
    thermal.detect_thermal_control.cache_clear()


_FAKE_THERMALFORGE_DETECTION = {
    "available": True,
    "selected": {"kind": "thermalforge", "path": "/usr/local/bin/thermalforge"},
    "instructions": "",
}


_AUTO_SUMMARY = {
    "ok": True,
    "fans": [
        {
            "mode": "auto",
            "target_rpm": 2317,
            "actual_rpm": 2320,
            "max_capacity_rpm": 7826,
        }
    ],
}


def test_set_thermal_profile_silent_prefers_daemon_socket(monkeypatch):
    """The fan reset goes through the daemon socket (no sudo, no app-kill) and
    does not fall back to the `auto` CLI when the socket accepts it AND the
    fan rows verify back on the auto curve (#201)."""
    monkeypatch.setattr(thermal, "detect_thermal_control", lambda: _FAKE_THERMALFORGE_DETECTION)

    sent: list[str] = []

    def fake_socket(command, *, timeout_s=3.0):
        sent.append(command)
        return {"ok": True, "response": "ok", "command": ["<thermalforge-daemon-socket>", command]}

    monkeypatch.setattr(thermal, "_daemon_socket_send", fake_socket)
    monkeypatch.setattr(thermal, "fan_summary", lambda: _AUTO_SUMMARY)

    def no_cli(command, *, timeout_s=None, cwd=None):
        raise AssertionError(f"CLI should not run when the socket handles it: {command}")

    monkeypatch.setattr(thermal, "_run_probe", no_cli)

    result = thermal.set_thermal_profile("silent")

    assert result["ok"] is True
    assert sent == ["auto"]
    assert result["command"] == ["<thermalforge-daemon-socket>", "auto"]
    assert result["attempts"][0]["verified"] is True


def test_set_thermal_profile_silent_socket_ack_without_effect_falls_back_to_cli(monkeypatch):
    """#201 regression guard: a daemon that replies ok but leaves the fans
    pinned must not be trusted — the same call falls through to the CLI
    candidates instead of reporting a restore that never happened."""
    monkeypatch.setattr(thermal, "detect_thermal_control", lambda: _FAKE_THERMALFORGE_DETECTION)
    monkeypatch.setattr(
        thermal,
        "_daemon_socket_send",
        lambda command, *, timeout_s=3.0: {
            "ok": True,
            "response": "ok",
            "command": ["<thermalforge-daemon-socket>", command],
        },
    )
    # Fans stay ramped no matter what the daemon claims.
    monkeypatch.setattr(thermal, "fan_summary", lambda: _RAMPED_SUMMARY)
    monkeypatch.setattr(thermal, "time", _FastTime())

    ran: list[list[str]] = []

    def fake_run(command, *, timeout_s=None, cwd=None):
        ran.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.set_thermal_profile("silent")

    assert result["ok"] is True
    assert result["attempts"][0]["verified"] is False
    assert ran and ran[0][-1] == "auto"


class _FastTime:
    """time shim: monotonic advances 1s per call so bounded verify loops
    exhaust instantly and sleep is a no-op."""

    def __init__(self) -> None:
        self._now = 0.0

    def monotonic(self) -> float:
        self._now += 1.0
        return self._now

    def time(self) -> float:
        return self.monotonic()

    def sleep(self, _s: float) -> None:
        return None


def test_wait_for_auto_fans_polls_until_rows_report_auto(monkeypatch):
    """The shared #201 verification (used by set_thermal_profile and the fan
    sidecar) keeps polling through ramped and unreadable summaries and returns
    True the moment every fan row is back on auto."""
    summaries = iter([_RAMPED_SUMMARY, {"ok": False, "fans": []}, _AUTO_SUMMARY])
    polled = {"n": 0}

    def fake_summary():
        polled["n"] += 1
        return next(summaries)

    monkeypatch.setattr(thermal, "fan_summary", fake_summary)
    monkeypatch.setattr(thermal, "time", _FastTime())

    assert thermal.wait_for_auto_fans(timeout_s=10.0) is True
    assert polled["n"] == 3


def test_wait_for_auto_fans_gives_up_at_the_deadline(monkeypatch):
    monkeypatch.setattr(thermal, "fan_summary", lambda: _RAMPED_SUMMARY)
    monkeypatch.setattr(thermal, "time", _FastTime())

    assert thermal.wait_for_auto_fans(timeout_s=3.0) is False


def test_set_thermal_profile_silent_falls_back_to_cli_without_daemon(monkeypatch):
    """With no daemon socket reachable (the autouse default), the reset uses the
    `auto` CLI candidates."""
    monkeypatch.setattr(thermal, "detect_thermal_control", lambda: _FAKE_THERMALFORGE_DETECTION)

    ran: list[list[str]] = []

    def fake_run(command, *, timeout_s=None, cwd=None):
        ran.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.set_thermal_profile("silent")

    assert result["ok"] is True
    assert ran and ran[0][-1] == "auto"


_RAMPED_SUMMARY = {
    "ok": True,
    "fans": [
        {
            "mode": "manual",
            "target_rpm": 7826,
            "actual_rpm": 7800,
            "max_capacity_rpm": 7826,
        }
    ],
}


def _patch_smart_fan_hardware(monkeypatch, calls, *, set_results=None):
    """Stub every hardware touchpoint the SmartFanController worker uses."""

    results = list(set_results or [])

    def fake_set(profile):
        calls.append(profile)
        if results:
            return results.pop(0)
        return {"ok": True, "profile": profile}

    monkeypatch.setattr(thermal, "check_and_recover_stale_max", lambda: None)
    monkeypatch.setattr(
        thermal,
        "install_max_lifecycle_hooks",
        lambda: (lambda: calls.append("auto") or {"ok": True, "profile": "silent"}),
    )
    monkeypatch.setattr(thermal, "set_thermal_profile", fake_set)
    monkeypatch.setattr(thermal, "fan_summary", lambda: _RAMPED_SUMMARY)
    # #227's heat-soak hold made the worker's restore path probe the real
    # SoC die temperature (soc_temperature_c -> `thermalforge status`).
    # Stub it like every other hardware touchpoint: on a machine whose die
    # is above MTPLX_SMART_FAN_SOAK_RELEASE_C (75C) -- routine late in a
    # full pytest battery -- the hold defers the restore, and any test that
    # waits for "auto" without end_request's wait_for_restore soak bypass
    # times out. "No usable reading" selects the legacy instant restore;
    # soak-specific tests override this via _patch_soak_probe.
    monkeypatch.setattr(
        thermal,
        "soc_temperature_c",
        lambda: {"ok": False, "celsius": None, "sensor": None},
    )


def test_smart_fan_controller_keeps_max_until_final_request(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)

    controller = thermal.SmartFanController(restore_delay_s=0, sensorless_delay_s=0)
    controller.begin_request("first")
    # begin_request is non-blocking; synchronize with the worker before
    # asserting on the hardware call log.
    assert controller.wait_for_ramp(5.0) is True
    controller.begin_request("second")

    assert calls == ["performance"]
    controller.end_request("first", wait_for_restore=True)
    assert calls == ["performance"]

    status = controller.end_request("second", wait_for_restore=True)

    assert calls == ["performance", "auto"]
    assert status["active"] is False
    assert status["active_count"] == 0
    assert status["commanded_max"] is False


def test_smart_fan_controller_begin_does_not_block_on_hardware(monkeypatch):
    """The whole point of the worker-thread overhaul: a slow fan daemon
    must not delay the request path. begin_request returns immediately
    even while the (stubbed) hardware command is still running."""
    import threading as _threading

    release = _threading.Event()
    calls: list[str] = []

    def slow_set(profile):
        release.wait(timeout=10.0)
        calls.append(profile)
        return {"ok": True, "profile": profile}

    monkeypatch.setattr(thermal, "check_and_recover_stale_max", lambda: None)
    monkeypatch.setattr(
        thermal,
        "install_max_lifecycle_hooks",
        lambda: (lambda: {"ok": True, "profile": "silent"}),
    )
    monkeypatch.setattr(thermal, "set_thermal_profile", slow_set)
    monkeypatch.setattr(thermal, "fan_summary", lambda: _RAMPED_SUMMARY)
    monkeypatch.setattr(thermal, "soc_temperature_c", lambda: {"ok": False, "celsius": None, "sensor": None})

    controller = thermal.SmartFanController(restore_delay_s=0, sensorless_delay_s=0)
    import time as _time

    started = _time.monotonic()
    status = controller.begin_request("req")
    elapsed = _time.monotonic() - started

    assert elapsed < 0.5, f"begin_request blocked for {elapsed:.2f}s"
    assert status["active"] is True
    release.set()
    assert controller.wait_for_ramp(5.0) is True
    controller.end_request("req", wait_for_restore=True)


def test_smart_fan_controller_retries_failed_ramp_once(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(
        monkeypatch,
        calls,
        set_results=[
            {"ok": False, "message": "daemon busy"},
            {"ok": True, "profile": "performance"},
        ],
    )

    controller = thermal.SmartFanController(restore_delay_s=0, sensorless_delay_s=0)
    controller.begin_request("req")

    assert controller.wait_for_ramp(5.0) is True
    status = controller.status()
    assert status["ramp_attempts"] == 2
    assert status["target_verified"] is True
    assert status["ramp_latency_s"] is not None
    controller.end_request("req", wait_for_restore=True)


def test_smart_fan_controller_reports_failure_without_hammering(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(
        monkeypatch,
        calls,
        set_results=[
            {"ok": False, "message": "no sudo"},
            {"ok": False, "message": "no sudo"},
        ],
    )

    controller = thermal.SmartFanController(restore_delay_s=0, sensorless_delay_s=0)
    controller.begin_request("req")

    assert controller.wait_for_ramp(5.0) is False
    status = controller.status()
    assert status["commanded_max"] is False
    assert "no sudo" in (status["last_error"] or "")
    # Initial attempt + exactly one retry — no further hammering while the
    # same lease generation stays active.
    assert calls == ["performance", "performance"]
    controller.end_request("req", wait_for_restore=True)


def test_smart_fan_controller_detach_never_touches_hardware(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)

    controller = thermal.SmartFanController(restore_delay_s=0, sensorless_delay_s=0)
    controller.begin_request("req")
    assert controller.wait_for_ramp(5.0) is True

    status = controller.detach()

    assert status["active"] is False
    assert status["commanded_max"] is False
    # detach() must NOT issue `thermalforge auto` — an external owner (Max
    # mode) is taking over and a delayed smart restore would silently drop
    # the new Max pin back to auto.
    import time as _time

    _time.sleep(0.2)
    assert calls == ["performance"]


def _wait_until(predicate, timeout_s=5.0, interval_s=0.02):
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(interval_s)
    return predicate()


def test_smart_fan_restore_retries_with_backoff_until_verified(monkeypatch):
    """#201: a restore that fails verification must be retried until the
    fans actually come back to auto — not marked restored and forgotten
    while the hardware stays pinned at max."""
    calls: list[str] = []
    monkeypatch.setattr(thermal, "check_and_recover_stale_max", lambda: None)
    monkeypatch.setattr(thermal, "set_thermal_profile", lambda profile: (
        calls.append(profile) or {"ok": True, "profile": profile}
    ))
    monkeypatch.setattr(thermal, "fan_summary", lambda: _RAMPED_SUMMARY)
    monkeypatch.setattr(thermal, "_clear_max_marker", lambda: None)
    monkeypatch.setattr(thermal, "soc_temperature_c", lambda: {"ok": False, "celsius": None, "sensor": None})

    restore_results = [
        {"ok": False, "message": "socket ack without effect"},
        {"ok": False, "message": "socket ack without effect"},
        {"ok": True, "profile": "silent"},
    ]
    restore_calls: list[int] = []

    def fake_cleanup():
        restore_calls.append(1)
        return restore_results.pop(0)

    monkeypatch.setattr(thermal, "install_max_lifecycle_hooks", lambda: fake_cleanup)
    monkeypatch.setattr(
        thermal, "restore_thermal_profile_verified", lambda **_kw: fake_cleanup()
    )
    monkeypatch.setattr(
        thermal.SmartFanController, "_RESTORE_RETRY_BACKOFF_S", (0.05, 0.05, 0.05, 0.05)
    )

    controller = thermal.SmartFanController(restore_delay_s=0, sensorless_delay_s=0)
    controller.begin_request("req")
    assert controller.wait_for_ramp(5.0) is True
    controller.end_request("req", wait_for_restore=True)

    # First restore failed; the worker must keep retrying on its own with
    # backoff until the third attempt verifies.
    assert _wait_until(lambda: len(restore_calls) >= 3)
    assert _wait_until(lambda: controller.status()["restore_verified"] is True)
    status = controller.status()
    assert status["restore_failures"] == 0
    assert status["commanded_max"] is False
    controller._shutdown = True


def test_smart_fan_stale_lease_reconciler_drops_leaked_leases(monkeypatch):
    """#201: a lease held while the engine is continuously idle is a leak
    from a wedged request path — it must be dropped and fans restored
    instead of pinning the fans forever."""
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    monkeypatch.setattr(thermal.SmartFanController, "_ACTIVITY_POLL_INTERVAL_S", 0.05)
    monkeypatch.setenv("MTPLX_SMART_FAN_STALE_LEASE_S", "0.2")

    controller = thermal.SmartFanController(
        restore_delay_s=0, sensorless_delay_s=0, activity_probe=lambda: False
    )
    controller.begin_request("leaked-lease")
    assert controller.wait_for_ramp(5.0) is True

    # Never end_request: the reconciler must clear it once the probe has
    # reported the engine idle past the stale window.
    assert _wait_until(lambda: controller.status()["active_count"] == 0)
    assert _wait_until(lambda: "auto" in calls)
    status = controller.status()
    assert status["stale_leases_reconciled"] == 1
    assert status["commanded_max"] is False
    controller._shutdown = True


def test_smart_fan_stale_lease_reconciler_never_fires_while_engine_busy(monkeypatch):
    """The reconciler must not drop leases while the activity probe reports
    model work — a long legitimate generation keeps its fan boost."""
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    monkeypatch.setattr(thermal.SmartFanController, "_ACTIVITY_POLL_INTERVAL_S", 0.02)
    monkeypatch.setenv("MTPLX_SMART_FAN_STALE_LEASE_S", "0.1")

    controller = thermal.SmartFanController(
        restore_delay_s=0, sensorless_delay_s=0, activity_probe=lambda: True
    )
    controller.begin_request("long-generation")
    assert controller.wait_for_ramp(5.0) is True

    import time as _time

    _time.sleep(0.5)
    status = controller.status()
    assert status["active_count"] == 1
    assert status["stale_leases_reconciled"] == 0
    assert status["commanded_max"] is True
    controller.end_request("long-generation", wait_for_restore=True)
    controller._shutdown = True


def test_thermalforge_profile_candidates_match_real_cli():
    """ThermalForge's actual CLI is `thermalforge max` and `thermalforge auto`.
    Verified live (May 2026) that even with the privileged daemon running,
    fan-set commands require sudo (the daemon doesn't proxy SMC writes), so
    we try ``sudo -n`` first so cleanup can complete inside Terminal's short
    close window, while keeping the unprefixed form as a fallback for users
    who somehow set up their own daemon path."""

    max_cmds = thermal._profile_command_candidates(
        {"kind": "thermalforge", "path": "/usr/local/bin/thermalforge"},
        "max",
    )
    assert max_cmds == [
        ["sudo", "-n", "/usr/local/bin/thermalforge", "max"],
        ["/usr/local/bin/thermalforge", "max"],
    ]

    perf_cmds = thermal._profile_command_candidates(
        {"kind": "thermalforge", "path": "/usr/local/bin/thermalforge"},
        "performance",
    )
    assert perf_cmds == [
        ["sudo", "-n", "/usr/local/bin/thermalforge", "max"],
        ["/usr/local/bin/thermalforge", "max"],
    ]

    silent_cmds = thermal._profile_command_candidates(
        {"kind": "thermalforge", "path": "/usr/local/bin/thermalforge"},
        "silent",
    )
    assert silent_cmds == [
        ["sudo", "-n", "/usr/local/bin/thermalforge", "auto"],
        ["/usr/local/bin/thermalforge", "auto"],
    ]


def test_thermalforge_status_uses_native_status_command():
    cmds = thermal._status_command_candidates(
        {"kind": "thermalforge", "path": "/usr/local/bin/thermalforge"}
    )
    assert cmds == [["/usr/local/bin/thermalforge", "status"]]


def test_install_thermal_control_homebrew_without_brew(monkeypatch):
    monkeypatch.setattr(thermal.shutil, "which", lambda name: None)
    result = thermal.install_thermal_control(method="homebrew")
    assert result["ok"] is False
    assert result.get("needs_prereq") == "brew"
    assert "Homebrew" in result["message"]


def test_install_thermal_control_homebrew_runs_tap(monkeypatch, tmp_path):
    """The Homebrew path must shell out to the real upstream tap."""
    thermal.detect_thermal_control.cache_clear()

    def fake_which(name):
        if name == "brew":
            return "/opt/homebrew/bin/brew"
        if name == "thermalforge":
            return "/opt/homebrew/bin/thermalforge"
        return None

    monkeypatch.setattr(thermal.shutil, "which", fake_which)

    invocations: list[list[str]] = []

    def fake_run(command, *, timeout_s=None, cwd=None):
        invocations.append(command)
        return {"command": command, "returncode": 0, "ok": True}

    monkeypatch.setattr(thermal, "_run_streaming", fake_run)
    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.install_thermal_control(method="homebrew")

    assert result["ok"] is True
    assert any("ProducerGuy/tap/thermalforge" in tok for cmd in invocations for tok in cmd)
    assert any("install" in cmd for cmd in invocations)
    thermal.detect_thermal_control.cache_clear()


def test_install_thermal_control_source_installs_to_mtplx_bin(monkeypatch, tmp_path):
    """The source path must build with `swift build -c release` and copy the
    binary to ``~/.mtplx/bin/thermalforge`` so the install is owned end-to-end
    by MTPLX. It must NEVER touch ``/usr/local/bin/`` or run upstream's
    `setup.sh` / `thermalforge install` (which has a destructive cwd bug)."""
    thermal.detect_thermal_control.cache_clear()

    def fake_which(name):
        if name == "git":
            return "/usr/bin/git"
        if name == "swift":
            return "/usr/bin/swift"
        return None

    monkeypatch.setattr(thermal.shutil, "which", fake_which)
    build_dir = tmp_path / "ThermalForge"
    bin_dir = tmp_path / "mtplx-bin"
    monkeypatch.setattr(thermal, "THERMALFORGE_BUILD_DIR", str(build_dir))
    monkeypatch.setattr(thermal, "MTPLX_THERMALFORGE_DIR", str(bin_dir))
    monkeypatch.setattr(thermal, "MTPLX_THERMALFORGE_PATH", str(bin_dir / "thermalforge"))

    invocations: list[list[str]] = []

    def fake_run(command, *, timeout_s=None, cwd=None):
        invocations.append(command)
        if command[0] == "/usr/bin/git" and "clone" in command:
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / ".git").mkdir(exist_ok=True)
        if command[0] == "/usr/bin/swift" and "build" in command:
            release_dir = build_dir / ".build" / "release"
            release_dir.mkdir(parents=True, exist_ok=True)
            (release_dir / "thermalforge").write_text("#!/bin/sh\necho fake\n")
            (release_dir / "thermalforge").chmod(0o755)
        # Stub out passwordless-sudo install + visudo.
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_streaming", fake_run)
    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    # Stub the sudoers-rule install (writes via subprocess.run with stdin) and
    # the live verification (would otherwise call the fake CLI).
    monkeypatch.setattr(
        thermal,
        "install_passwordless_sudoers_rule",
        lambda **kw: {"ok": True, "step": "passwordless_sudoers", "message": "ok"},
    )
    monkeypatch.setattr(
        thermal,
        "set_thermal_profile_verified",
        lambda profile, **kw: {"ok": True, "message": "fans pinned"},
    )
    monkeypatch.setattr(
        thermal,
        "set_thermal_profile",
        lambda profile, **kw: {"ok": True, "profile": profile},
    )

    result = thermal.install_thermal_control(method="source")

    assert result["ok"] is True, result
    assert result["method"] == "source"
    assert result["binary"] == str(bin_dir / "thermalforge")
    # We must never have invoked the upstream installer (`thermalforge install`)
    # or `setup.sh` — that's what the destructive cwd bug lives in.
    assert not any("install" in cmd and cmd[0].endswith("thermalforge") for cmd in invocations)
    assert not any(cmd[0] == "bash" and cmd[1].endswith("setup.sh") for cmd in invocations)
    # We DID build with swift.
    assert any(cmd[0] == "/usr/bin/swift" and "build" in cmd for cmd in invocations)
    # Binary actually landed at the MTPLX-private path.
    assert (bin_dir / "thermalforge").exists()
    thermal.detect_thermal_control.cache_clear()


def test_install_thermal_control_auto_uses_bundled_helper_before_source(monkeypatch, tmp_path):
    """The macOS app ships ThermalForge in its bundle. Fresh DMG users should
    copy that binary into MTPLX's private bin dir instead of needing git,
    Homebrew, or Xcode command-line tools during onboarding."""
    thermal.detect_thermal_control.cache_clear()
    bundled_dir = tmp_path / "AppResources" / "ThermalForge"
    bundled_dir.mkdir(parents=True)
    bundled = bundled_dir / "thermalforge"
    bundled.write_text("#!/bin/sh\necho bundled\n")
    bundled.chmod(0o755)

    bin_dir = tmp_path / "mtplx-bin"
    monkeypatch.setattr(thermal, "MTPLX_THERMALFORGE_DIR", str(bin_dir))
    monkeypatch.setattr(thermal, "MTPLX_THERMALFORGE_PATH", str(bin_dir / "thermalforge"))
    monkeypatch.setenv(thermal.BUNDLED_THERMALFORGE_ENV, str(bundled))

    invocations: list[list[str]] = []

    def fake_run(command, *, timeout_s=None, cwd=None):
        invocations.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_streaming", fake_run)
    monkeypatch.setattr(thermal, "_run_probe", fake_run)
    monkeypatch.setattr(
        thermal,
        "install_passwordless_sudoers_rule",
        lambda **kw: {"ok": True, "step": "passwordless_sudoers", "message": "ok"},
    )
    monkeypatch.setattr(
        thermal,
        "set_thermal_profile_verified",
        lambda profile, **kw: {"ok": True, "message": "fans pinned"},
    )
    monkeypatch.setattr(
        thermal,
        "set_thermal_profile",
        lambda profile, **kw: {"ok": True, "profile": profile},
    )

    result = thermal.install_thermal_control(method="auto")

    assert result["ok"] is True, result
    assert result["method"] == "bundled"
    assert result["binary"] == str(bin_dir / "thermalforge")
    assert (bin_dir / "thermalforge").read_text() == bundled.read_text()
    assert not any("git" in cmd[0] for cmd in invocations)
    assert not any("swift" in cmd[0] for cmd in invocations)
    assert result["steps"][0]["step"] == "copy_bundled_to_mtplx_bin"
    thermal.detect_thermal_control.cache_clear()


def test_install_thermal_control_auto_does_not_fall_back_to_homebrew(monkeypatch, tmp_path):
    """`auto` must NOT silently fall back to Homebrew. The upstream brew
    formula has been observed to fail mid-build (missing
    ``Scripts/generate-icon.swift`` in the tarball, May 2026), and when it
    does succeed it installs into ``/usr/local/bin`` which we can't keep
    stable. We pin source-only behaviour."""
    thermal.detect_thermal_control.cache_clear()
    invocations: list[list[str]] = []

    def fake_which(name):
        # No swift -> source install bails at the prereq check.
        if name == "git":
            return "/usr/bin/git"
        if name == "brew":
            return "/opt/homebrew/bin/brew"
        return None

    monkeypatch.setattr(thermal.shutil, "which", fake_which)

    def fake_run(command, *, timeout_s=None, cwd=None):
        invocations.append(command)
        return {"command": command, "returncode": 0, "ok": True}

    monkeypatch.setattr(thermal, "_run_streaming", fake_run)
    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.install_thermal_control(method="auto")

    assert result["ok"] is False
    assert result.get("needs_prereq") == "swift"
    # Crucially: no `brew install` ever ran.
    assert not any(
        cmd and cmd[0] == "/opt/homebrew/bin/brew" and "install" in cmd for cmd in invocations
    ), invocations
    thermal.detect_thermal_control.cache_clear()


def test_install_thermal_control_homebrew_alias_still_works(monkeypatch):
    """Backwards-compat: the old ``install_thermal_control_homebrew`` name
    still resolves (covers any external callers from earlier versions)."""
    assert thermal.install_thermal_control_homebrew is thermal.install_thermal_control


def _is_sudo_install(command) -> bool:
    return (
        command
        and command[0] == "sudo"
        and "/bin/sh" in command
        and thermal._PRIVILEGED_SUDOERS_SCRIPT in command
    )


def _is_security_install(command) -> bool:
    return command[:3] == ["/usr/bin/security", "execute-with-privileges", "/bin/sh"]


def _fake_probe_ok(command, *, timeout_s=None, cwd=None):
    """visudo pre-check and the final `sudo -n ... status` verification both pass."""
    return {"command": command, "returncode": 0, "ok": True, "stdout": "{}", "stderr": ""}


def test_passwordless_sudoers_rule_uses_security_prompt_when_gui_sudo_fails(monkeypatch):
    invocations: list[list[str]] = []

    def fake_which(name):
        if name == "security":
            return "/usr/bin/security"
        if name == "visudo":
            return "/usr/sbin/visudo"
        return None

    def fake_subprocess_run(command, **kwargs):
        invocations.append(command)
        if _is_sudo_install(command):
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="sudo: a terminal is required to read the password",
            )
        if _is_security_install(command):
            # The privileged script receives the finished rule line, never
            # the raw user name, so the escaping decided in Python is what
            # lands on disk.
            assert command[-3:] == [
                thermal.SUDOERS_FILE,
                "testuser ALL=(root) NOPASSWD: /tmp/mtplx-bin/thermalforge",
                "/tmp/mtplx-bin/thermalforge",
            ]
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess call: {command}")

    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(thermal, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr(thermal.shutil, "which", fake_which)
    monkeypatch.setattr(thermal.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(thermal, "_run_probe", _fake_probe_ok)

    result = thermal.install_passwordless_sudoers_rule(
        binary_path="/tmp/mtplx-bin/thermalforge"
    )

    assert result["ok"] is True, result
    assert result["method"] == "security_execute_with_privileges"
    assert any(_is_security_install(command) for command in invocations)
    # Without a terminal the sudo attempt must be non-interactive so it can
    # never sit on a password prompt nobody can see.
    sudo_calls = [command for command in invocations if _is_sudo_install(command)]
    assert sudo_calls and sudo_calls[0][1] == "-n", sudo_calls


def test_passwordless_sudoers_rule_reports_security_prompt_failure(monkeypatch):
    def fake_which(name):
        if name == "security":
            return "/usr/bin/security"
        return None

    def fake_subprocess_run(command, **kwargs):
        if _is_sudo_install(command):
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="sudo: a terminal is required to read the password",
            )
        if _is_security_install(command):
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="The user canceled authorization.",
            )
        raise AssertionError(f"unexpected subprocess call: {command}")

    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(thermal, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr(thermal.shutil, "which", fake_which)
    monkeypatch.setattr(thermal.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(thermal, "_run_probe", _fake_probe_ok)

    result = thermal.install_passwordless_sudoers_rule(
        binary_path="/tmp/mtplx-bin/thermalforge"
    )

    assert result["ok"] is False
    assert result["step"] == "sudo_install"
    assert "macOS admin authorization failed" in result["message"]
    # No terminal: the user is told where to type the password instead of
    # being left with a command that appears to hang.
    assert "Run `mtplx max --grant-sudo` in a terminal" in result["message"]


# -- sudoers rule construction (C-15) -----------------------------------------


def test_sudoers_rule_escapes_spaces_in_binary_path():
    """A home on a volume with a space used to produce a rule whose command was
    split at the space; visudo accepts that rule, so the install reported ok
    while `sudo -n thermalforge max` was refused and fans never ramped."""
    rule = thermal.sudoers_rule_for("testuser", "/Volumes/My Drive/.mtplx/bin/thermalforge")
    assert rule == "testuser ALL=(root) NOPASSWD: /Volumes/My\\ Drive/.mtplx/bin/thermalforge"
    # The other characters sudo's lexer accepts only when escaped.
    rule = thermal.sudoers_rule_for("testuser", "/tmp/a=b,c:d#e/thermalforge")
    assert rule == "testuser ALL=(root) NOPASSWD: /tmp/a\\=b\\,c\\:d\\#e/thermalforge"
    # A plain path is written verbatim.
    assert (
        thermal.sudoers_rule_for("test.user-2", "/Users/test/.mtplx/bin/thermalforge")
        == "test.user-2 ALL=(root) NOPASSWD: /Users/test/.mtplx/bin/thermalforge"
    )


@pytest.mark.parametrize(
    "binary_path",
    [
        "/tmp/mtplx-bin/thermalforge\nroot ALL=(ALL) NOPASSWD: ALL",
        "/tmp/mtplx-bin/thermal\rforge",
        '/tmp/my "quoted" dir/thermalforge',
        "/tmp/it's/thermalforge",
        "/tmp/mtplx-bin/*",
        "/tmp/mtplx-bin/thermalforge?",
        "/tmp/[bin]/thermalforge",
        "/tmp/back\\slash/thermalforge",
        "/tmp/tab\tdir/thermalforge",
        "relative/thermalforge",
    ],
)
def test_sudoers_rule_rejects_unsafe_binary_paths(binary_path):
    with pytest.raises(ValueError) as excinfo:
        thermal.sudoers_rule_for("testuser", binary_path)
    assert "sudoers rule" in str(excinfo.value) or "absolute" in str(excinfo.value)


@pytest.mark.parametrize("user", ["", "_unknown user", "-root", "%admin", "#0", "a,b", "bob\n"])
def test_sudoers_rule_rejects_unsafe_user_names(user):
    with pytest.raises(ValueError):
        thermal.sudoers_rule_for(user, "/tmp/mtplx-bin/thermalforge")


def test_install_passwordless_sudoers_rule_installs_nothing_when_visudo_rejects(monkeypatch):
    """The rule is parsed on a temporary file we own before any privileged
    call; a rejected rule ends the install with nothing written."""
    privileged: list[list[str]] = []

    def fake_probe(command, *, timeout_s=None, cwd=None):
        if command[1:3] == ["-c", "-f"]:
            assert command[-1] != thermal.SUDOERS_FILE
            return {
                "command": command,
                "returncode": 1,
                "ok": False,
                "stdout": "",
                "stderr": f"{command[-1]}:1:10: syntax error",
            }
        raise AssertionError(f"unexpected probe: {command}")

    def fake_subprocess_run(command, **kwargs):
        privileged.append(command)
        raise AssertionError(f"privileged call must not run: {command}")

    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(thermal, "_run_probe", fake_probe)
    monkeypatch.setattr(thermal.subprocess, "run", fake_subprocess_run)

    result = thermal.install_passwordless_sudoers_rule(binary_path="/tmp/mtplx-bin/thermalforge")

    assert result["ok"] is False
    assert result["step"] == "visudo_check"
    assert "nothing was installed" in result["message"]
    assert "syntax error" in result["message"]
    assert privileged == []


def test_install_passwordless_sudoers_rule_refuses_unsafe_path_before_any_call(monkeypatch):
    def fake_probe(command, *, timeout_s=None, cwd=None):
        raise AssertionError(f"no command may run: {command}")

    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(thermal, "_run_probe", fake_probe)
    monkeypatch.setattr(thermal.subprocess, "run", fake_probe)

    result = thermal.install_passwordless_sudoers_rule(
        binary_path="/tmp/mtplx-bin/thermalforge\nroot ALL=(ALL) NOPASSWD: ALL"
    )

    assert result["ok"] is False
    assert result["step"] == "build_rule"
    assert "cannot be written to a sudoers rule" in result["message"]


def test_install_passwordless_sudoers_rule_defaults_to_the_binary_fan_commands_use(
    monkeypatch, tmp_path
):
    """`mtplx max --grant-sudo` used to grant the PATH copy of thermalforge
    (or /usr/local/bin/thermalforge) while fan control ran MTPLX's own copy in
    ~/.mtplx/bin, so the grant never applied to the binary that needed it."""
    bin_dir = tmp_path / "mtplx bin"
    bin_dir.mkdir()
    own = bin_dir / "thermalforge"
    own.write_text("#!/bin/sh\nexit 0\n")
    own.chmod(0o755)
    monkeypatch.setattr(thermal, "MTPLX_THERMALFORGE_DIR", str(bin_dir))
    monkeypatch.setattr(thermal, "MTPLX_THERMALFORGE_PATH", str(own))
    monkeypatch.setattr(
        thermal.shutil,
        "which",
        lambda name: "/usr/local/bin/thermalforge" if name == "thermalforge" else None,
    )

    sudo_calls: list[list[str]] = []

    def fake_subprocess_run(command, **kwargs):
        assert _is_sudo_install(command), command
        assert "timeout" in kwargs and kwargs["timeout"], "sudo must be bounded"
        sudo_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(thermal, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(thermal.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(thermal, "_run_probe", _fake_probe_ok)

    result = thermal.install_passwordless_sudoers_rule()

    assert result["ok"] is True, result
    assert result["method"] == "sudo"
    assert result["binary_path"] == str(own)
    assert result["rule"] == f"testuser ALL=(root) NOPASSWD: {tmp_path}/mtplx\\ bin/thermalforge"
    # Interactive terminal: plain `sudo` (may prompt), the script gets the
    # escaped rule and the raw path it must find executable.
    assert len(sudo_calls) == 1
    assert sudo_calls[0][1] == "/bin/sh"
    assert sudo_calls[0][-3:] == [thermal.SUDOERS_FILE, result["rule"], str(own)]


def test_install_passwordless_sudoers_rule_reports_missing_thermalforge(monkeypatch):
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: None)
    monkeypatch.setattr(
        thermal.subprocess, "run", lambda *a, **k: pytest.fail("no command may run")
    )

    result = thermal.install_passwordless_sudoers_rule()

    assert result["ok"] is False
    assert result["step"] == "locate_thermalforge"
    assert "mtplx max --install" in result["message"]


def test_install_passwordless_sudoers_rule_sudo_timeout_is_reported(monkeypatch):
    def fake_which(name):
        if name == "security":
            return "/usr/bin/security"
        return None

    def fake_subprocess_run(command, **kwargs):
        if _is_sudo_install(command):
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))
        if _is_security_install(command):
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr="The user canceled authorization."
            )
        raise AssertionError(f"unexpected subprocess call: {command}")

    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(thermal, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(thermal.shutil, "which", fake_which)
    monkeypatch.setattr(thermal.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(thermal, "_run_probe", _fake_probe_ok)

    result = thermal.install_passwordless_sudoers_rule(binary_path="/tmp/mtplx-bin/thermalforge")

    assert result["ok"] is False
    assert result["step"] == "sudo_install"
    assert "did not finish within" in result["message"]


def test_fan_summary_parses_thermalforge_status_json(monkeypatch):
    """Verify-after-set relies on parsing `thermalforge status` JSON. Pin the
    parser against the *real* upstream shape (captured live from
    `thermalforge status` on macOS, May 2026) so a future schema drift is
    caught immediately. The original parser only knew about a flat ``rpm``
    field that doesn't exist in the actual output and silently returned
    ``ok=False`` — which is exactly the bug that made --max appear to do
    nothing."""
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(
        thermal.shutil,
        "which",
        lambda name: "/usr/local/bin/thermalforge" if name == "thermalforge" else None,
    )

    fake_status_json = (
        '{"fans": ['
        '{"actual_rpm": 5800, "target_rpm": 5800, "min_rpm": 2317, '
        '"max_rpm": 7826, "mode": "max", "index": 0},'
        '{"actual_rpm": 6100, "target_rpm": 6100, "min_rpm": 2317, '
        '"max_rpm": 7826, "mode": "max", "index": 1}'
        '], "temperatures": {"TCMb": 78.5}}'
    )

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            return {"command": command, "returncode": 0, "ok": True, "stdout": fake_status_json, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)
    monkeypatch.setattr(thermal, "_run_streaming", fake_run)

    summary = thermal.fan_summary()
    assert summary["ok"] is True
    assert summary["min_rpm"] == 5800
    assert summary["max_rpm"] == 6100
    # Verify we kept the rich fields so callers can reason about target vs
    # actual when, for example, fans are still ramping toward the target.
    assert summary["fans"][0]["target_rpm"] == 5800
    assert summary["fans"][0]["actual_rpm"] == 5800
    assert summary["fans"][0]["max_capacity_rpm"] == 7826
    assert summary["fans"][0]["mode"] == "max"
    thermal.detect_thermal_control.cache_clear()


def test_fan_summary_handles_no_tool(monkeypatch):
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: None)
    monkeypatch.setattr(thermal.shutil, "which", lambda name: None)
    thermal.detect_thermal_control.cache_clear()
    summary = thermal.fan_summary()
    assert summary["ok"] is False
    assert summary["min_rpm"] is None
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_verified_succeeds_when_daemon_commands_max(monkeypatch):
    """Happy path: ``thermalforge max`` is accepted and the daemon flips
    target_rpm + mode immediately. We must succeed even though actual_rpm
    is still climbing — Apple Silicon fans need ~15s to physically reach
    the target, and waiting that long during install is unacceptable.
    Pinning ``mode`` and ``target_rpm`` is what catches the regression
    that pinned the user's fans during install."""
    thermal.detect_thermal_control.cache_clear()
    thermal._find_thermalforge.__defaults__ if hasattr(thermal._find_thermalforge, "__defaults__") else None
    monkeypatch.setattr(
        thermal,
        "_find_thermalforge",
        lambda: "/usr/local/bin/thermalforge",
    )
    call_log: list[str] = []
    pre_status = (
        '{"fans": ['
        '{"actual_rpm": 1850, "target_rpm": 2317, "max_rpm": 7826, "mode": "auto", "index": 0},'
        '{"actual_rpm": 1900, "target_rpm": 2317, "max_rpm": 7826, "mode": "auto", "index": 1}'
        ']}'
    )
    # Note actual_rpm is still way below the old threshold (1900-2400),
    # but mode flipped to "manual" and target_rpm jumped to 7826. That's
    # the only reliable signal that the command was accepted.
    post_status = (
        '{"fans": ['
        '{"actual_rpm": 2380, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0},'
        '{"actual_rpm": 2410, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 1}'
        ']}'
    )
    state = {"phase": "pre"}

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            stdout = pre_status if state["phase"] == "pre" else post_status
            return {"command": command, "returncode": 0, "ok": True, "stdout": stdout, "stderr": ""}
        if command and command[-1] == "max":
            state["phase"] = "post"
            return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)
    monkeypatch.setattr(thermal, "_run_streaming", fake_run)

    result = thermal.set_thermal_profile_verified(
        "performance",
        settle_seconds=0,
        log=call_log.append,
    )

    assert result["ok"] is True, result
    assert any("fans commanded" in line.lower() for line in call_log)
    assert "7826 RPM target" in result["message"]
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_verified_does_not_report_zero_target_when_actual_is_zero(monkeypatch):
    """Regression: ThermalForge can command max immediately while actual_rpm is
    still 0. The message must report the target_rpm, not the actual fallback,
    or the CLI prints the nonsense "target 0 RPM" line."""

    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: "/usr/local/bin/thermalforge")
    call_log: list[str] = []
    statuses = [
        (
            '{"fans": ['
            '{"actual_rpm": 0, "target_rpm": 2317, "max_rpm": 7826, "mode": "auto", "index": 0},'
            '{"actual_rpm": 0, "target_rpm": 2502, "max_rpm": 7826, "mode": "auto", "index": 1}'
            ']}'
        ),
        (
            '{"fans": ['
            '{"actual_rpm": 0, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0},'
            '{"actual_rpm": 0, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 1}'
            ']}'
        ),
    ]

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            stdout = statuses.pop(0) if statuses else (
                '{"fans": ['
                '{"actual_rpm": 0, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0},'
                '{"actual_rpm": 0, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 1}'
                ']}'
            )
            return {"command": command, "returncode": 0, "ok": True, "stdout": stdout, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.set_thermal_profile_verified(
        "performance",
        settle_seconds=0,
        log=call_log.append,
    )

    assert result["ok"] is True, result
    assert result["after"]["actual_max_rpm"] == 0
    assert result["after"]["target_max_rpm"] == 7826
    assert "7826 RPM target" in result["message"]
    assert "target 0 RPM" not in "\n".join(call_log)
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_verified_waits_for_actual_ramp_when_required(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: "/usr/local/bin/thermalforge")
    monkeypatch.setattr(thermal.time, "sleep", lambda _seconds: None)
    call_log: list[str] = []
    statuses = [
        (
            '{"fans": ['
            '{"actual_rpm": 2315, "target_rpm": 2317, "max_rpm": 7826, "mode": "auto", "index": 0},'
            '{"actual_rpm": 2501, "target_rpm": 2502, "max_rpm": 7826, "mode": "auto", "index": 1}'
            ']}'
        ),
        (
            '{"fans": ['
            '{"actual_rpm": 0, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0},'
            '{"actual_rpm": 0, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 1}'
            ']}'
        ),
        (
            '{"fans": ['
            '{"actual_rpm": 4400, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0},'
            '{"actual_rpm": 4500, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 1}'
            ']}'
        ),
        (
            '{"fans": ['
            '{"actual_rpm": 6900, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0},'
            '{"actual_rpm": 7000, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 1}'
            ']}'
        ),
    ]

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            stdout = statuses.pop(0) if statuses else (
                '{"fans": ['
                '{"actual_rpm": 6900, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0},'
                '{"actual_rpm": 7000, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 1}'
                ']}'
            )
            return {"command": command, "returncode": 0, "ok": True, "stdout": stdout, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.set_thermal_profile_verified(
        "performance",
        settle_seconds=0,
        require_actual_ramp=True,
        actual_ramp_timeout_s=5,
        actual_ramp_poll_interval_s=0.1,
        log=call_log.append,
    )

    assert result["ok"] is True, result
    assert result["after"]["actual_min_rpm"] == 6900
    assert "fans ramped to max" in result["message"]
    assert any("waiting for actual fans to ramp" in line for line in call_log)
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_verified_fails_when_actual_ramp_is_required_but_stuck(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: "/usr/local/bin/thermalforge")
    stuck_max = (
        '{"fans": ['
        '{"actual_rpm": 0, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0},'
        '{"actual_rpm": 0, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 1}'
        ']}'
    )
    statuses = [
        (
            '{"fans": ['
            '{"actual_rpm": 2315, "target_rpm": 2317, "max_rpm": 7826, "mode": "auto", "index": 0},'
            '{"actual_rpm": 2501, "target_rpm": 2502, "max_rpm": 7826, "mode": "auto", "index": 1}'
            ']}'
        ),
        stuck_max,
    ]

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            stdout = statuses.pop(0) if statuses else stuck_max
            return {"command": command, "returncode": 0, "ok": True, "stdout": stdout, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.set_thermal_profile_verified(
        "performance",
        settle_seconds=0,
        require_actual_ramp=True,
        actual_ramp_timeout_s=0,
    )

    assert result["ok"] is False
    assert "actual fan RPM did not ramp" in result["message"]
    assert "Actual: 0 RPM" in result["message"]
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_verified_fails_when_daemon_ignores_command(monkeypatch):
    """``thermalforge max`` returned 0 but the daemon's mode is still ``auto``
    (we got fooled into believing it worked). Verifier must catch this so
    we can roll back and surface a clear actionable error."""
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(
        thermal,
        "_find_thermalforge",
        lambda: "/usr/local/bin/thermalforge",
    )
    stuck_status = (
        '{"fans": ['
        '{"actual_rpm": 1850, "target_rpm": 2317, "max_rpm": 7826, "mode": "auto", "index": 0},'
        '{"actual_rpm": 1900, "target_rpm": 2317, "max_rpm": 7826, "mode": "auto", "index": 1}'
        ']}'
    )

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            return {"command": command, "returncode": 0, "ok": True, "stdout": stuck_status, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)
    monkeypatch.setattr(thermal, "_run_streaming", fake_run)

    result = thermal.set_thermal_profile_verified(
        "performance",
        settle_seconds=0,
    )

    assert result["ok"] is False
    assert "not commanding max" in result["message"]
    assert "ThermalForge.app" in (result.get("actionable") or "")
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_verified_warns_not_to_run_tune_with_sudo(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: "/usr/local/bin/thermalforge")
    status = (
        '{"fans": ['
        '{"actual_rpm": 1850, "target_rpm": 2317, "max_rpm": 7826, "mode": "auto", "index": 0}'
        ']}'
    )

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            return {"command": command, "returncode": 0, "ok": True, "stdout": status, "stderr": ""}
        return {
            "command": command,
            "returncode": 1,
            "ok": False,
            "stdout": "",
            "stderr": "Error: Fan unlock failed: Failed to write Ftst=1. Run with sudo.",
        }

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.set_thermal_profile_verified("performance", settle_seconds=0)

    actionable = result.get("actionable") or ""
    assert result["ok"] is False
    assert "Do not run `sudo mtplx tune`" in actionable
    assert "mtplx max --grant-sudo" in actionable
    assert "Then re-run `mtplx tune` as your normal user" in actionable
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_verified_handles_no_tool(monkeypatch):
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: None)
    monkeypatch.setattr(thermal.shutil, "which", lambda name: None)
    thermal.detect_thermal_control.cache_clear()
    result = thermal.set_thermal_profile_verified("performance", settle_seconds=0)
    assert result["ok"] is False
    assert "mtplx max --install" in (result.get("actionable") or "")
    thermal.detect_thermal_control.cache_clear()


def test_restore_thermal_profile_verified_requires_auto_status(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: "/usr/local/bin/thermalforge")
    auto_status = (
        '{"fans": ['
        '{"actual_rpm": 2400, "target_rpm": 2318, "max_rpm": 7826, "mode": "auto", "index": 0},'
        '{"actual_rpm": 2450, "target_rpm": 2503, "max_rpm": 7826, "mode": "auto", "index": 1}'
        ']}'
    )

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            return {"command": command, "returncode": 0, "ok": True, "stdout": auto_status, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.restore_thermal_profile_verified()

    assert result["ok"] is True
    assert result["message"] == "fan profile restored"
    thermal.detect_thermal_control.cache_clear()


def test_restore_thermal_profile_verified_fails_when_still_manual(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: "/usr/local/bin/thermalforge")
    manual_status = (
        '{"fans": ['
        '{"actual_rpm": 2400, "target_rpm": 7826, "max_rpm": 7826, "mode": "manual", "index": 0}'
        ']}'
    )

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            return {"command": command, "returncode": 0, "ok": True, "stdout": manual_status, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.restore_thermal_profile_verified()

    assert result["ok"] is False
    assert "not verified" in result["message"]
    thermal.detect_thermal_control.cache_clear()


def test_restore_thermal_profile_verified_accepts_auto_with_transient_max_target(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: "/usr/local/bin/thermalforge")
    suspicious_status = (
        '{"fans": ['
        '{"actual_rpm": 2400, "target_rpm": 7826, "max_rpm": 7826, "mode": "auto", "index": 0}'
        ']}'
    )

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            return {"command": command, "returncode": 0, "ok": True, "stdout": suspicious_status, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)

    result = thermal.restore_thermal_profile_verified(settle_timeout_s=0)

    assert result["ok"] is True
    assert result["message"] == "fan profile restored"
    thermal.detect_thermal_control.cache_clear()


def test_restore_thermal_profile_verified_waits_for_auto_target_to_settle(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: "/usr/local/bin/thermalforge")
    snapshots = [
        (
            '{"fans": ['
            '{"actual_rpm": 7400, "target_rpm": 7826, "max_rpm": 7826, "mode": "auto", "index": 0}'
            ']}'
        ),
        (
            '{"fans": ['
            '{"actual_rpm": 2400, "target_rpm": 2318, "max_rpm": 7826, "mode": "auto", "index": 0}'
            ']}'
        ),
    ]

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command and command[-1] == "status":
            stdout = snapshots.pop(0) if snapshots else (
                '{"fans": ['
                '{"actual_rpm": 2400, "target_rpm": 2318, "max_rpm": 7826, "mode": "auto", "index": 0}'
                ']}'
            )
            return {"command": command, "returncode": 0, "ok": True, "stdout": stdout, "stderr": ""}
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_run)
    monkeypatch.setattr(thermal.time, "sleep", lambda _seconds: None)

    result = thermal.restore_thermal_profile_verified(settle_timeout_s=2, poll_interval_s=0.1)

    assert result["ok"] is True
    assert result["message"] == "fan profile restored"
    thermal.detect_thermal_control.cache_clear()


def test_install_always_restores_fans_even_when_verification_fails(monkeypatch, tmp_path):
    """Regression: a previous bug pinned the user's fans because verification
    failed (settle window too short) and we skipped the restore call. The
    install path must ALWAYS run `thermalforge auto` on the way out."""
    thermal.detect_thermal_control.cache_clear()
    bin_dir = tmp_path / "mtplx-bin"
    monkeypatch.setattr(thermal, "MTPLX_THERMALFORGE_DIR", str(bin_dir))
    monkeypatch.setattr(thermal, "MTPLX_THERMALFORGE_PATH", str(bin_dir / "thermalforge"))
    monkeypatch.setattr(
        thermal.shutil,
        "which",
        lambda name: {"git": "/usr/bin/git", "swift": "/usr/bin/swift"}.get(name),
    )

    build_dir = tmp_path / "ThermalForge"
    monkeypatch.setattr(thermal, "THERMALFORGE_BUILD_DIR", str(build_dir))

    def fake_run(command, *, timeout_s=None, cwd=None):
        if command[0] == "/usr/bin/git" and "clone" in command:
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / ".git").mkdir(exist_ok=True)
        if command[0] == "/usr/bin/swift" and "build" in command:
            release_dir = build_dir / ".build" / "release"
            release_dir.mkdir(parents=True, exist_ok=True)
            (release_dir / "thermalforge").write_text("#!/bin/sh\nexit 0\n")
            (release_dir / "thermalforge").chmod(0o755)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_streaming", fake_run)
    monkeypatch.setattr(thermal, "_run_probe", fake_run)
    monkeypatch.setattr(
        thermal,
        "install_passwordless_sudoers_rule",
        lambda **kw: {"ok": True, "step": "passwordless_sudoers", "message": "ok"},
    )
    # Force verification to FAIL — this is the bug-trigger scenario.
    monkeypatch.setattr(
        thermal,
        "set_thermal_profile_verified",
        lambda profile, **kw: {"ok": False, "message": "fans not commanding max"},
    )

    restored: list[str] = []

    def fake_set(profile, **kw):
        restored.append(profile)
        return {"ok": True, "profile": profile}

    monkeypatch.setattr(thermal, "set_thermal_profile", fake_set)

    result = thermal.install_thermal_control(method="source")

    assert result["ok"] is False
    # Critical: even though verification failed, we restored fans.
    assert "silent" in restored, restored
    thermal.detect_thermal_control.cache_clear()


# -- heat-soak release hold (#227) ------------------------------------------


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(0.01)
    return False


def _patch_soak_probe(monkeypatch, temps):
    """soc_temperature_c stub returning readings from ``temps`` (last repeats)."""
    sequence = list(temps)

    def fake_probe():
        value = sequence.pop(0) if len(sequence) > 1 else sequence[0]
        if value is None:
            return {"ok": False, "celsius": None, "sensor": None}
        return {"ok": True, "celsius": float(value), "sensor": "TCMb"}

    monkeypatch.setattr(thermal, "soc_temperature_c", fake_probe)
    monkeypatch.setattr(thermal.SmartFanController, "_SOAK_PROBE_INTERVAL_S", 0.01)


def test_smart_fan_soak_hold_keeps_max_until_cooled(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    _patch_soak_probe(monkeypatch, [92.0, 91.0, 60.0])

    controller = thermal.SmartFanController(restore_delay_s=0)
    controller.begin_request("burst")
    assert controller.wait_for_ramp(5.0) is True

    controller.end_request("burst")

    assert _wait_for(lambda: "auto" in calls), controller.status()
    status = controller.status()
    assert status["soak_holds"] >= 1
    assert status["soak_release_reason"] == "cooled"
    assert status["soak_last_temp_c"] == 60.0


def test_smart_fan_soak_hold_cap_bounds_the_pin(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    _patch_soak_probe(monkeypatch, [95.0])
    monkeypatch.setenv("MTPLX_SMART_FAN_SOAK_HOLD_CAP_S", "0.2")

    controller = thermal.SmartFanController(restore_delay_s=0)
    controller.begin_request("burst")
    assert controller.wait_for_ramp(5.0) is True

    controller.end_request("burst")

    # The die never cools, but the cap guarantees the fans come back.
    assert _wait_for(lambda: "auto" in calls), controller.status()
    assert controller.status()["soak_release_reason"] == "hold_cap"


def test_smart_fan_soak_probe_failure_falls_back_to_legacy_restore(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    _patch_soak_probe(monkeypatch, [None])

    controller = thermal.SmartFanController(restore_delay_s=0, sensorless_delay_s=0)
    controller.begin_request("burst")
    assert controller.wait_for_ramp(5.0) is True

    controller.end_request("burst")

    assert _wait_for(lambda: "auto" in calls), controller.status()
    status = controller.status()
    assert status["soak_holds"] == 0
    assert status["soak_release_reason"] is None


def test_smart_fan_soak_disabled_by_env_restores_immediately(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    probes: list[str] = []

    def fake_probe():
        probes.append("probe")
        return {"ok": True, "celsius": 99.0, "sensor": "TCMb"}

    monkeypatch.setattr(thermal, "soc_temperature_c", fake_probe)
    monkeypatch.setenv("MTPLX_SMART_FAN_SOAK_RELEASE_C", "0")

    controller = thermal.SmartFanController(restore_delay_s=0)
    controller.begin_request("burst")
    assert controller.wait_for_ramp(5.0) is True
    probes.clear()

    controller.end_request("burst", wait_for_restore=True)

    assert calls == ["performance", "auto"]
    assert probes == []


def test_smart_fan_wait_for_restore_bypasses_soak_hold(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    _patch_soak_probe(monkeypatch, [95.0])

    controller = thermal.SmartFanController(restore_delay_s=0)
    controller.begin_request("bench")
    assert controller.wait_for_ramp(5.0) is True

    # Bench lanes and shutdown paths must not sit behind a hot-die hold.
    controller.end_request("bench", wait_for_restore=True)

    assert calls == ["performance", "auto"]


def _patch_status_temperatures(monkeypatch, temperatures):
    import json as _json

    monkeypatch.setattr(
        thermal,
        "thermal_status",
        lambda: {
            "ok": True,
            "status": {"stdout": _json.dumps({"fans": [], "temperatures": temperatures})},
        },
    )


def test_soc_temperature_prefers_hottest_cpu_die_sensor(monkeypatch):
    _patch_status_temperatures(
        monkeypatch,
        {"TCMb": 91.8, "TCDX": 60.9, "TB0T": 35.8, "Tp0C": 99.0},
    )

    reading = thermal.soc_temperature_c()

    assert reading["ok"] is True
    assert reading["sensor"] == "TCMb"
    assert reading["celsius"] == 91.8


def test_soc_temperature_filters_sentinel_values_and_falls_back(monkeypatch):
    _patch_status_temperatures(monkeypatch, {"TCMb": 0.0, "Tp0C": 61.7})

    reading = thermal.soc_temperature_c()

    assert reading["ok"] is True
    assert reading["sensor"] == "Tp0C"


def test_soc_temperature_pinned_sensor_env(monkeypatch):
    _patch_status_temperatures(monkeypatch, {"TCMb": 91.8, "TCDX": 60.9})
    monkeypatch.setenv("MTPLX_SMART_FAN_SOAK_SENSOR", "TCDX")

    reading = thermal.soc_temperature_c()

    assert reading == {"ok": True, "celsius": 60.9, "sensor": "TCDX"}


def test_soc_temperature_without_temperature_data_is_not_ok(monkeypatch):
    _patch_status_temperatures(monkeypatch, {})

    assert thermal.soc_temperature_c()["ok"] is False


def test_is_desktop_mac_detects_studio_and_respects_override(monkeypatch):
    monkeypatch.delenv("MTPLX_DESKTOP_MAC", raising=False)
    for model_id in ("Mac13,1", "Mac14,14", "Mac15,14", "Mac16,9", "MacStudio1,1", "Macmini9,1"):
        monkeypatch.setattr(thermal, "_sysctl_hw_model", lambda m=model_id: m)
        assert thermal.is_desktop_mac() is True
    for model_id in ("Mac17,7", "Mac16,1", "Mac16,12", "MacBookPro18,2"):
        monkeypatch.setattr(thermal, "_sysctl_hw_model", lambda m=model_id: m)
        assert thermal.is_desktop_mac() is False
    # Unlisted future Mac IDs fall back to battery presence detection.
    monkeypatch.setattr(thermal, "_sysctl_hw_model", lambda: "Mac18,9")
    monkeypatch.setattr(thermal, "_has_internal_battery", lambda: False)
    assert thermal.is_desktop_mac() is True
    monkeypatch.setattr(thermal, "_has_internal_battery", lambda: True)
    assert thermal.is_desktop_mac() is False
    monkeypatch.setattr(thermal, "_has_internal_battery", lambda: None)
    assert thermal.is_desktop_mac() is False
    monkeypatch.setenv("MTPLX_DESKTOP_MAC", "1")
    assert thermal.is_desktop_mac() is True
    monkeypatch.setenv("MTPLX_DESKTOP_MAC", "0")
    assert thermal.is_desktop_mac() is False


def test_set_fan_rpm_prefers_daemon_socket_and_falls_back_to_cli(monkeypatch):
    monkeypatch.setattr(thermal, "detect_thermal_control", lambda: _FAKE_THERMALFORGE_DETECTION)
    sent: list[str] = []
    current_target = [2600]

    def fake_sock(cmd, *, timeout_s=3.0):
        sent.append(cmd)
        return {"ok": True, "response": "ok", "command": ["<socket>", cmd]}

    monkeypatch.setattr(thermal, "_daemon_socket_send", fake_sock)
    monkeypatch.setattr(
        thermal,
        "fan_summary",
        lambda: {"ok": True, "fans": [{"mode": "manual", "target_rpm": current_target[0], "max_capacity_rpm": 5000}]},
    )

    res = thermal.set_fan_rpm(2600)
    assert res["ok"] is True
    assert res["rpm"] == 2600
    assert sent == ["set 2600"]

    # Socket absent -> CLI `set <rpm>` fallback must verify actual target_rpm
    monkeypatch.setattr(thermal, "_daemon_socket_send", lambda *a, **k: None)
    cli_cmds: list[list[str]] = []

    def fake_probe(cmd, *, timeout_s=3.0):
        cli_cmds.append(cmd)
        current_target[0] = int(cmd[-1])
        return {"ok": True, "command": cmd, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(thermal, "_run_probe", fake_probe)
    res2 = thermal.set_fan_rpm(2200)
    assert res2["ok"] is True
    assert cli_cmds[0][-2:] == ["set", "2200"]

    # CLI exit 0 without verified target RPM must fail closed.
    monkeypatch.setattr(
        thermal,
        "_run_probe",
        lambda cmd, *, timeout_s=3.0: {"ok": True, "command": cmd, "returncode": 0, "stdout": "", "stderr": ""},
    )
    monkeypatch.setattr(
        thermal,
        "fan_summary",
        lambda: {"ok": True, "fans": [{"mode": "auto", "target_rpm": 1200, "max_capacity_rpm": 5000}]},
    )
    res3 = thermal.set_fan_rpm(3000)
    assert res3["ok"] is False


def test_smart_fan_cool_turn_stays_on_auto_with_zero_hardware_calls(monkeypatch):
    calls: list[object] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    monkeypatch.setattr(thermal, "set_fan_rpm", lambda rpm: calls.append(("rpm", rpm)) or {"ok": True, "rpm": rpm})
    monkeypatch.setattr(thermal, "soc_temperature_c", lambda: {"ok": True, "celsius": 60.0, "sensor": "TCMb"})

    controller = thermal.SmartFanController(restore_delay_s=0, desktop=False)
    controller.begin_request("cool")
    assert controller.wait_for_step(5.0) == 0
    assert controller.status()["current_level"] == 0
    assert calls == []

    status = controller.end_request("cool", wait_for_restore=True)
    assert status["current_level"] == 0
    assert calls == []
    controller._shutdown = True


def test_smart_fan_stepped_ramp_and_hysteresis_laptop_and_desktop(monkeypatch):
    calls: list[object] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    fan_mode = ["auto"]

    def track_profile(profile):
        calls.append(profile)
        fan_mode[0] = "max" if profile in ("performance", "max") else "auto"
        return {"ok": True, "profile": profile}

    monkeypatch.setattr(thermal, "set_thermal_profile", track_profile)
    monkeypatch.setattr(
        thermal,
        "fan_summary",
        lambda: {
            "ok": True,
            "fans": [
                {"mode": fan_mode[0], "target_rpm": 5000 if fan_mode[0] == "max" else 1200, "actual_rpm": 1200, "max_capacity_rpm": 5000, "raw": {"min_rpm": 1000}},
            ],
        },
    )
    monkeypatch.setattr(
        thermal,
        "set_fan_rpm",
        lambda rpm: calls.append(("rpm", rpm)) or {"ok": True, "rpm": rpm},
    )
    current_temp = [65.0]
    monkeypatch.setattr(
        thermal,
        "soc_temperature_c",
        lambda: {"ok": True, "celsius": current_temp[0], "sensor": "TCMb"},
    )
    monkeypatch.setattr(thermal.SmartFanController, "_ACTIVE_TEMP_POLL_S", 0.02)

    controller = thermal.SmartFanController(restore_delay_s=0, desktop=False)
    assert controller.status()["profile_kind"] == "laptop"
    controller.begin_request("r1")
    assert controller.wait_for_step(5.0) == 0

    # Warm to 72C -> level 1 (55% of 1000..5000 = 3200 RPM)
    current_temp[0] = 72.0
    assert _wait_until(lambda: controller.status()["current_level"] == 1)
    assert calls[-1] == ("rpm", 3200)

    # Drop slightly to 69C: hysteresis (70 - 5 = 65) keeps level 1
    current_temp[0] = 69.0
    _time.sleep(0.08)
    assert controller.status()["current_level"] == 1

    # Climb to 84C -> level 2 (80% of 1000..5000 = 4200 RPM)
    current_temp[0] = 84.0
    assert _wait_until(lambda: controller.status()["current_level"] == 2)
    assert calls[-1] == ("rpm", 4200)

    # Peak at 92C -> level 3 (max/performance)
    current_temp[0] = 92.0
    assert _wait_until(lambda: controller.status()["current_level"] == 3 and controller.status()["commanded_max"] is True)

    # Cool down to 64C while still active -> steps down cleanly and eventually returns to auto
    current_temp[0] = 64.0
    assert _wait_until(lambda: controller.status()["current_level"] == 0)
    assert "silent" in calls

    controller.end_request("r1", wait_for_restore=True)
    controller._shutdown = True

    # Desktop profile (Mac Studio): higher thresholds and gentler RPM fractions
    desktop_ctrl = thermal.SmartFanController(restore_delay_s=0, desktop=True)
    assert desktop_ctrl.status()["profile_kind"] == "desktop"
    assert desktop_ctrl._desired_level_for_temp(72.0, 0) == 0
    assert desktop_ctrl._desired_level_for_temp(77.0, 0) == 1
    assert desktop_ctrl._target_rpm_for_level(1) == 2800  # 1000 + 0.45 * 4000
    assert desktop_ctrl._desired_level_for_temp(87.0, 1) == 2
    assert desktop_ctrl._desired_level_for_temp(93.0, 2) == 3
    desktop_ctrl._shutdown = True


def test_smart_fan_sensorless_delay_avoids_ramp_on_short_turn(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    controller = thermal.SmartFanController(restore_delay_s=0, sensorless_delay_s=0.25)
    controller.begin_request("short")
    assert controller.wait_for_step(5.0) == 0
    assert calls == []
    controller.end_request("short", wait_for_restore=True)
    assert calls == []

    # Longer sensorless turn ramps once delay elapses
    controller.begin_request("long")
    assert controller.wait_for_ramp(5.0) is True
    assert calls == ["performance"]
    controller.end_request("long", wait_for_restore=True)
    assert calls == ["performance", "auto"]
    controller._shutdown = True


def test_smart_fan_step_active_detach_race_never_touches_hardware_or_restores(monkeypatch):
    calls: list[object] = []
    _patch_smart_fan_hardware(monkeypatch, calls)

    # Case 1: detach fires while soc_temperature_c() is in flight
    c1 = thermal.SmartFanController(restore_delay_s=0, desktop=False)
    monkeypatch.setattr(c1, "_ensure_lifecycle_hooks", lambda: calls.append("hooks"))
    monkeypatch.setattr(c1, "_ensure_worker_locked", lambda: None)
    c1.begin_request("r1")
    monkeypatch.setattr(
        thermal,
        "soc_temperature_c",
        lambda: (c1.detach(), {"ok": True, "celsius": 76.0, "sensor": "TCMb"})[1],
    )
    monkeypatch.setattr(thermal, "set_fan_rpm", lambda rpm: calls.append(("rpm", rpm)) or {"ok": True, "rpm": rpm})
    c1._do_step_active()
    assert calls == []
    status1 = c1.status()
    assert status1["current_level"] == 0
    assert status1["commanded_max"] is False
    c1._shutdown = True

    # Case 2: detach fires while set_fan_rpm() is in flight
    calls.clear()
    c2 = thermal.SmartFanController(restore_delay_s=0, desktop=False)
    monkeypatch.setattr(c2, "_ensure_lifecycle_hooks", lambda: calls.append("hooks"))
    monkeypatch.setattr(c2, "_ensure_worker_locked", lambda: None)
    c2.begin_request("r2")
    monkeypatch.setattr(thermal, "soc_temperature_c", lambda: {"ok": True, "celsius": 76.0, "sensor": "TCMb"})
    monkeypatch.setattr(
        thermal,
        "set_fan_rpm",
        lambda rpm: (c2.detach(), calls.append(("rpm", rpm)), {"ok": True, "rpm": rpm})[2],
    )
    c2._do_step_active()
    assert len(calls) == 1 and calls[0][0] == "rpm"
    assert "hooks" not in calls
    assert "auto" not in calls
    status2 = c2.status()
    assert status2["current_level"] == 0
    assert status2["commanded_max"] is False
    with c2._lock:
        assert c2._cleanup is None
    c2._shutdown = True


def test_smart_fan_unsupported_and_failed_step_avoid_hook_leak_and_retry_storm(monkeypatch):
    calls: list[object] = []
    _patch_smart_fan_hardware(monkeypatch, calls)
    hook_calls: list[str] = []
    monkeypatch.setattr(thermal, "install_max_lifecycle_hooks", lambda: hook_calls.append("installed") or (lambda: {"ok": True}))
    monkeypatch.setattr(thermal, "soc_temperature_c", lambda: {"ok": True, "celsius": 76.0, "sensor": "TCMb"})
    monkeypatch.setattr(thermal.SmartFanController, "_ACTIVE_TEMP_POLL_S", 0.02)

    # Unsupported intermediate controller (e.g. TG Pro): falls back once to _do_ramp()
    rpm_attempts: list[int] = []
    monkeypatch.setattr(
        thermal,
        "set_fan_rpm",
        lambda rpm: (rpm_attempts.append(rpm), {"ok": False, "unsupported_controller": True, "rpm": rpm, "message": "unsupported"})[1],
    )
    ctrl = thermal.SmartFanController(restore_delay_s=0, desktop=False)
    ctrl.begin_request("u1")
    assert ctrl.wait_for_ramp(5.0) is True
    assert rpm_attempts == [rpm_attempts[0]]
    assert calls == ["performance"]
    assert hook_calls == ["installed"]
    ctrl.end_request("u1", wait_for_restore=True)
    ctrl._shutdown = True

    # ThermalForge intermediate RPM failure: no hook leak, sets _ramp_failed_generation (no 2s spin)
    hook_calls.clear()
    calls.clear()
    rpm_attempts.clear()
    monkeypatch.setattr(
        thermal,
        "set_fan_rpm",
        lambda rpm: (rpm_attempts.append(rpm), {"ok": False, "rpm": rpm, "message": "sudo denied"})[1],
    )
    ctrl2 = thermal.SmartFanController(restore_delay_s=0, desktop=False)
    ctrl2.begin_request("f1")
    assert ctrl2.wait_for_ramp(5.0) is False
    _time.sleep(0.06)
    assert len(rpm_attempts) == 2
    assert hook_calls == []
    ctrl2._shutdown = True
