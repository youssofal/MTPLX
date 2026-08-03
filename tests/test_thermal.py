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


def test_smart_fan_controller_keeps_max_until_final_request(monkeypatch):
    calls: list[str] = []
    _patch_smart_fan_hardware(monkeypatch, calls)

    controller = thermal.SmartFanController(restore_delay_s=0)
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

    controller = thermal.SmartFanController(restore_delay_s=0)
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

    controller = thermal.SmartFanController(restore_delay_s=0)
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

    controller = thermal.SmartFanController(restore_delay_s=0)
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

    controller = thermal.SmartFanController(restore_delay_s=0)
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

    controller = thermal.SmartFanController(restore_delay_s=0)
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
        restore_delay_s=0, activity_probe=lambda: False
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
        restore_delay_s=0, activity_probe=lambda: True
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


def test_passwordless_sudoers_rule_uses_security_prompt_when_gui_sudo_fails(monkeypatch):
    invocations: list[list[str]] = []

    def fake_which(name):
        if name == "security":
            return "/usr/bin/security"
        return None

    def fake_subprocess_run(command, **kwargs):
        invocations.append(command)
        if command == ["sudo", "tee", thermal.SUDOERS_FILE]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="sudo: a terminal is required to read the password",
            )
        if command[:3] == ["/usr/bin/security", "execute-with-privileges", "/bin/sh"]:
            assert command[-3:] == [
                thermal.SUDOERS_FILE,
                "testuser",
                "/tmp/mtplx-bin/thermalforge",
            ]
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess call: {command}")

    def fake_probe(command, *, timeout_s=None):
        assert command == ["sudo", "-n", "/tmp/mtplx-bin/thermalforge", "status"]
        return {"command": command, "returncode": 0, "ok": True, "stdout": "{}", "stderr": ""}

    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(thermal.shutil, "which", fake_which)
    monkeypatch.setattr(thermal.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(thermal, "_run_probe", fake_probe)

    result = thermal.install_passwordless_sudoers_rule(
        binary_path="/tmp/mtplx-bin/thermalforge"
    )

    assert result["ok"] is True, result
    assert result["method"] == "security_execute_with_privileges"
    assert any(
        command[:3] == ["/usr/bin/security", "execute-with-privileges", "/bin/sh"]
        for command in invocations
    )


def test_passwordless_sudoers_rule_reports_security_prompt_failure(monkeypatch):
    def fake_which(name):
        if name == "security":
            return "/usr/bin/security"
        return None

    def fake_subprocess_run(command, **kwargs):
        if command == ["sudo", "tee", thermal.SUDOERS_FILE]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="sudo: a terminal is required to read the password",
            )
        if command[:3] == ["/usr/bin/security", "execute-with-privileges", "/bin/sh"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="The user canceled authorization.",
            )
        raise AssertionError(f"unexpected subprocess call: {command}")

    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(thermal.shutil, "which", fake_which)
    monkeypatch.setattr(thermal.subprocess, "run", fake_subprocess_run)

    result = thermal.install_passwordless_sudoers_rule(
        binary_path="/tmp/mtplx-bin/thermalforge"
    )

    assert result["ok"] is False
    assert result["step"] == "sudo_tee"
    assert "macOS admin authorization failed" in result["message"]


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

    controller = thermal.SmartFanController(restore_delay_s=0)
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
