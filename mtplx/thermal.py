"""Opt-in thermal-control helpers for MTPLX.

The public contract is deliberately conservative: detect known fan-control
tools, run their documented/profile-style CLI commands when available, and
otherwise report clear installation instructions. This module never falls back
to spin loops or clock-anchor hacks.
"""

from __future__ import annotations

import os
import pwd
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator


PROFILE_LABELS = {
    "performance": "Performance",
    "max": "Max",
    "silent": "Silent",
}

# Real ThermalForge CLI — verified against
# https://github.com/ProducerGuy/ThermalForge (Apr 2026):
#   thermalforge max     → ramp fans to max
#   thermalforge auto    → restore Apple defaults
#   thermalforge status  → JSON
THERMALFORGE_TAP = "ProducerGuy/tap/thermalforge"

INSTALL_INSTRUCTIONS = {
    "thermalforge": "Install ThermalForge and ensure the thermalforge CLI is on PATH.",
    "tgpro": "Install TG Pro and ensure tgpro or tgpro-cli is on PATH.",
    "none": (
        "Run `mtplx max --install` to install ThermalForge automatically, or "
        "install TG Pro manually if you prefer. MTPLX will continue without "
        "fan control when --max is requested and no supported tool is present."
    ),
}


def _run_probe(command: list[str], *, timeout_s: float = 3.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


# Unix socket the ThermalForge privileged daemon listens on (matches
# ThermalForgeDaemon.socketPath). Reaching it resets fans as root without sudo
# and, unlike the `thermalforge auto` CLI, without quitting the menu bar app.
THERMALFORGE_DAEMON_SOCKET = "/tmp/thermalforge.sock"


def _daemon_socket_send(command: str, *, timeout_s: float = 3.0) -> dict[str, Any] | None:
    """Send one newline-terminated command to the ThermalForge daemon socket.

    Returns None when no daemon is reachable, so callers fall back to the CLI;
    otherwise ``{"ok": bool, "response": str, "command": [...]}``. The daemon
    speaks "auto" / "max" / "set <rpm>" / "status" and replies "ok", JSON, or
    "error: ...".
    """
    if not os.path.exists(THERMALFORGE_DAEMON_SOCKET):
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(THERMALFORGE_DAEMON_SOCKET)
            sock.sendall((command + "\n").encode())
            response = sock.recv(8192).decode(errors="replace").strip()
    except OSError:
        return None
    return {
        "ok": bool(response) and not response.lower().startswith("error"),
        "response": response,
        "command": ["<thermalforge-daemon-socket>", command],
    }


def _version(path: str) -> dict[str, Any]:
    for args in (["--version"], ["version"]):
        result = _run_probe([path, *args])
        if result["ok"]:
            return result
    return _run_probe([path, "--help"])


MTPLX_THERMALFORGE_DIR = os.path.expanduser("~/.mtplx/bin")
MTPLX_THERMALFORGE_PATH = os.path.join(MTPLX_THERMALFORGE_DIR, "thermalforge")
BUNDLED_THERMALFORGE_ENV = "MTPLX_BUNDLED_THERMALFORGE"


def _find_thermalforge() -> str | None:
    """Locate a ``thermalforge`` binary in MTPLX's preferred order.

    1. ``~/.mtplx/bin/thermalforge`` — the copy MTPLX owns and installs. We
       prefer this so that anything external (an upstream installer with
       broken cwd handling, an OS update, ``brew uninstall``, etc.)
       cannot break MTPLX's fan control.
    2. ``shutil.which("thermalforge")`` — system PATH (typically
       ``/usr/local/bin/thermalforge`` from a manual install).
    """

    if os.path.isfile(MTPLX_THERMALFORGE_PATH) and os.access(MTPLX_THERMALFORGE_PATH, os.X_OK):
        return MTPLX_THERMALFORGE_PATH
    return shutil.which("thermalforge")


def _bundled_thermalforge_path() -> str | None:
    path = os.environ.get(BUNDLED_THERMALFORGE_ENV, "").strip()
    if not path:
        return None
    expanded = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
        return expanded
    return None


@lru_cache(maxsize=1)
def detect_thermal_control() -> dict[str, Any]:
    thermalforge = _find_thermalforge()
    tgpro = shutil.which("tgpro") or shutil.which("tgpro-cli")
    tools: list[dict[str, Any]] = []
    if thermalforge:
        tools.append(
            {
                "kind": "thermalforge",
                "path": thermalforge,
                "version": _version(thermalforge),
            }
        )
    if tgpro:
        tools.append(
            {
                "kind": "tgpro",
                "path": tgpro,
                "version": _version(tgpro),
            }
        )
    selected = tools[0] if tools else None
    selected_kind = selected["kind"] if selected else "none"
    return {
        "available": selected is not None,
        "selected": selected,
        "tools": tools,
        "instructions": INSTALL_INSTRUCTIONS[selected_kind],
        "clock_anchor_enabled": os.environ.get("MTPLX_GPU_CLOCK_ANCHOR") == "1",
        "clock_anchor_policy": "explicit experimental only; never used for product claims",
    }


def _profile_command_candidates(tool: dict[str, Any], profile: str) -> list[list[str]]:
    path = str(tool["path"])
    kind = str(tool["kind"])
    if kind == "thermalforge":
        # Verified on macOS 14+ (May 2026): even with the privileged daemon
        # running, ``thermalforge max`` and ``thermalforge auto`` still fail
        # without sudo ("Fan unlock failed: Run with sudo."). Status is fine
        # without sudo. So we always wrap fan-set commands in ``sudo -n``,
        # which the install path enables via a passwordless sudoers rule.
        # Keep the unprefixed form as a fallback for hand-configured daemon
        # setups, but try sudo first so cleanup can finish inside macOS
        # Terminal's short SIGHUP grace window.
        if profile in ("performance", "max"):
            return [
                ["sudo", "-n", path, "max"],
                [path, "max"],
            ]
        if profile == "silent":
            return [
                ["sudo", "-n", path, "auto"],
                [path, "auto"],
            ]
        return [[path, "auto"]]
    if kind == "tgpro":
        # TG Pro CLI syntax is product-documented as best-effort across
        # versions; keep the original try-list.
        label = PROFILE_LABELS[profile]
        return [
            [path, "profile", label],
            [path, "set-profile", label],
            [path, "--profile", label],
        ]
    return []


def _status_command_candidates(tool: dict[str, Any]) -> list[list[str]]:
    path = str(tool["path"])
    kind = str(tool["kind"])
    if kind == "thermalforge":
        # ThermalForge `status` already emits JSON.
        return [[path, "status"]]
    return [
        [path, "status", "--json"],
        [path, "status"],
        [path, "sensors", "--json"],
        [path, "sensors"],
    ]


def thermal_status() -> dict[str, Any]:
    detection = detect_thermal_control()
    selected = detection.get("selected")
    if not selected:
        return {"ok": False, "detection": detection, "status": None}
    attempts = [_run_probe(command) for command in _status_command_candidates(selected)]
    first_ok = next((attempt for attempt in attempts if attempt["ok"]), None)
    return {
        "ok": first_ok is not None,
        "detection": detection,
        "status": first_ok,
        "attempts": attempts,
    }


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _min_or_none(values: list[int]) -> int | None:
    return min(values) if values else None


def _max_or_none(values: list[int]) -> int | None:
    return max(values) if values else None


def fan_summary() -> dict[str, Any]:
    """Best-effort fan-RPM summary, used to verify ``thermalforge max`` actually
    ramped the fans rather than silently no-op'ing.

    Returns ``{"ok": bool, "min_rpm": int|None, "max_rpm": int|None,
    "fans": [...]}``. ``ok`` is True when at least one fan reading was parsed.
    """

    status = thermal_status()
    if not status.get("ok"):
        return {"ok": False, "min_rpm": None, "max_rpm": None, "fans": [], "raw": status}
    raw_stdout = status.get("status", {}).get("stdout") or ""
    rpms: list[int] = []
    actual_rpms: list[int] = []
    target_rpms: list[int] = []
    capacity_rpms: list[int] = []
    fans: list[dict[str, Any]] = []
    try:
        import json as _json

        parsed = _json.loads(raw_stdout)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        # Real ThermalForge JSON shape (verified live against
        # `thermalforge status` on macOS, May 2026):
        #
        #   {"fans": [{"actual_rpm": 2315, "target_rpm": 2317,
        #              "min_rpm": 2317, "max_rpm": 7826, "mode": "auto",
        #              "index": 0}, ...], "temperatures": {...}}
        #
        # `actual_rpm` is what the fan is currently spinning at;
        # `target_rpm` is the daemon's setpoint; `max_rpm` / `min_rpm` are the
        # fan's hardware capacity envelope. We read `actual_rpm` first
        # (verifies real ramp), fall back to `target_rpm` (verifies
        # commanded ramp), and finally any flat `rpm` field for non-
        # ThermalForge tools.
        candidates = parsed.get("fans") or parsed.get("Fans") or []
        if isinstance(candidates, dict):
            candidates = list(candidates.values())
        for entry in candidates if isinstance(candidates, list) else []:
            if not isinstance(entry, dict):
                continue
            actual_int = _int_or_none(entry.get("actual_rpm"))
            target_int = _int_or_none(entry.get("target_rpm"))
            capacity_int = _int_or_none(entry.get("max_rpm"))
            fallback_int = (
                _int_or_none(entry.get("rpm"))
                if entry.get("rpm") is not None
                else _int_or_none(entry.get("RPM"))
                if entry.get("RPM") is not None
                else _int_or_none(entry.get("speed"))
            )
            rpm_int = actual_int if actual_int is not None else target_int
            if rpm_int is None:
                rpm_int = fallback_int
            if rpm_int is None:
                continue
            rpms.append(rpm_int)
            if actual_int is not None:
                actual_rpms.append(actual_int)
            if target_int is not None:
                target_rpms.append(target_int)
            if capacity_int is not None:
                capacity_rpms.append(capacity_int)
            fans.append(
                {
                    "rpm": rpm_int,
                    "target_rpm": target_int,
                    "actual_rpm": actual_int,
                    "max_capacity_rpm": capacity_int,
                    "mode": entry.get("mode"),
                    "raw": entry,
                }
            )
    return {
        "ok": bool(rpms),
        "min_rpm": _min_or_none(rpms),
        "max_rpm": _max_or_none(rpms),
        "actual_min_rpm": _min_or_none(actual_rpms),
        "actual_max_rpm": _max_or_none(actual_rpms),
        "target_min_rpm": _min_or_none(target_rpms),
        "target_max_rpm": _max_or_none(target_rpms),
        "capacity_min_rpm": _min_or_none(capacity_rpms),
        "capacity_max_rpm": _max_or_none(capacity_rpms),
        "fans": fans,
        "raw": status,
    }


def soc_temperature_c() -> dict[str, Any]:
    """Best-effort SoC die temperature from the thermal tool's status JSON.

    ThermalForge's status payload carries a ``temperatures`` map of SMC
    sensors. The heat-soak bug class (#227) is a CPU-die soak, so prefer the
    hottest ``TC*`` (CPU cluster/die) sensor; fall back to the hottest
    plausible sensor of any name. Sensor names differ per Apple Silicon
    generation, so ``MTPLX_SMART_FAN_SOAK_SENSOR`` can pin an exact key.

    Returns ``{"ok": bool, "celsius": float|None, "sensor": str|None}``.
    ``ok`` is False whenever no usable die reading exists — callers must
    treat that as "no temperature data", never as "cool".
    """
    status = thermal_status()
    if not status.get("ok"):
        return {"ok": False, "celsius": None, "sensor": None}
    raw_stdout = status.get("status", {}).get("stdout") or ""
    try:
        import json as _json

        parsed = _json.loads(raw_stdout)
    except Exception:
        parsed = None
    temps = parsed.get("temperatures") if isinstance(parsed, dict) else None
    if not isinstance(temps, dict) or not temps:
        return {"ok": False, "celsius": None, "sensor": None}
    readings: dict[str, float] = {}
    for key, value in temps.items():
        try:
            celsius = float(value)
        except (TypeError, ValueError):
            continue
        # Discard sentinel / absent-sensor values (0, negatives, SMC junk).
        if 1.0 <= celsius <= 130.0:
            readings[str(key)] = celsius
    if not readings:
        return {"ok": False, "celsius": None, "sensor": None}
    pinned = os.environ.get("MTPLX_SMART_FAN_SOAK_SENSOR", "").strip()
    if pinned:
        if pinned in readings:
            return {"ok": True, "celsius": readings[pinned], "sensor": pinned}
        return {"ok": False, "celsius": None, "sensor": None}
    cpu_sensors = {key: val for key, val in readings.items() if key.upper().startswith("TC")}
    pool = cpu_sensors or readings
    sensor = max(pool, key=pool.get)
    return {"ok": True, "celsius": pool[sensor], "sensor": sensor}


# Fraction of a fan's reported capacity that proves a max/performance command
# has reached the controller. Some Macs report very different RPM envelopes, so
# use hardware capacity when available and keep the old absolute threshold only
# as a sensorless fallback.
FAN_RAMP_TARGET_FRACTION = 0.85
FAN_RAMP_FALLBACK_THRESHOLD_RPM = 4000


def _fan_max_capacity(fan: dict[str, Any]) -> int | None:
    """The fan's hardware max RPM, from the summary field or the raw status."""
    raw = fan.get("max_capacity_rpm") or (fan.get("raw") or {}).get("max_rpm")
    return _int_or_none(raw)


def _fan_target_is_ramped(fan: dict[str, Any]) -> bool:
    target_int = _int_or_none(fan.get("target_rpm"))
    if target_int is None:
        return False
    max_int = _fan_max_capacity(fan)
    if max_int and max_int > 0:
        return target_int >= int(max_int * FAN_RAMP_TARGET_FRACTION)
    return target_int >= FAN_RAMP_FALLBACK_THRESHOLD_RPM


def _fan_actual_is_ramped(
    fan: dict[str, Any],
    *,
    fraction: float = FAN_RAMP_TARGET_FRACTION,
) -> bool:
    actual_int = _int_or_none(fan.get("actual_rpm"))
    if actual_int is None:
        return False
    max_int = _fan_max_capacity(fan)
    if max_int and max_int > 0:
        return actual_int >= int(max_int * fraction)
    target_int = _int_or_none(fan.get("target_rpm"))
    if target_int and target_int > 0:
        return actual_int >= int(target_int * fraction)
    return actual_int >= FAN_RAMP_FALLBACK_THRESHOLD_RPM


def _ok_fans(summary: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Parsed fan rows when the summary is usable, else None."""
    if not summary.get("ok"):
        return None
    return summary.get("fans") or []


def _summary_indicates_max(summary: dict[str, Any]) -> bool:
    """Return True iff the parsed status snapshot says fans are commanded
    to ramp (mode is manual/max OR target RPM is clearly above idle).

    We check ``target_rpm`` rather than ``actual_rpm`` because the daemon
    sets the target instantly when ``thermalforge max`` is accepted, while
    actual RPM only catches up over ~15 seconds (Apple Silicon fans ramp at
    roughly 400 RPM/sec). Reading actual_rpm too early is what made an
    earlier verification path falsely report failure.
    """

    for fan in _ok_fans(summary) or []:
        mode = (fan.get("mode") or "").lower()
        if mode in {"manual", "max"}:
            return True
        if _fan_target_is_ramped(fan):
            return True
    return False


def _summary_indicates_actual_ramp(
    summary: dict[str, Any],
    *,
    fraction: float = FAN_RAMP_TARGET_FRACTION,
) -> bool:
    fans = _ok_fans(summary)
    return bool(fans) and all(_fan_actual_is_ramped(fan, fraction=fraction) for fan in fans)


def _rpm_range(min_value: Any, max_value: Any) -> str:
    low = _int_or_none(min_value)
    high = _int_or_none(max_value)
    if low is None and high is None:
        return "unavailable"
    if low is None:
        return f"{high} RPM"
    if high is None or high == low:
        return f"{low} RPM"
    return f"{low}-{high} RPM"


def _summary_indicates_auto(summary: dict[str, Any]) -> bool:
    """Return True iff all parsed fan rows are back on the automatic curve."""

    fans = _ok_fans(summary)
    if not fans:
        return False
    for fan in fans:
        mode = str(fan.get("mode") or "").lower()
        target_int = _int_or_none(fan.get("target_rpm"))
        # ThermalForge may briefly keep reporting the previous max target while
        # mode has already flipped to auto. The mode is the authoritative
        # restore signal; target only helps when a tool omits mode.
        if mode in {"auto", "automatic", "default"}:
            continue
        if target_int is not None and target_int < FAN_RAMP_FALLBACK_THRESHOLD_RPM:
            continue
        return False
    return True


# How long a restore may take to show up in the fan rows before the reply
# that promised it is disbelieved (#201).
FAN_RESTORE_VERIFY_TIMEOUT_S = 3.0


def wait_for_auto_fans(
    *,
    timeout_s: float = FAN_RESTORE_VERIFY_TIMEOUT_S,
    poll_interval_s: float = 0.5,
) -> bool:
    """Poll the fan rows until every one reports the automatic curve.

    A daemon "ok" reply or a zero exit from ``thermalforge auto`` is a
    promise, not proof: a wedged or stale daemon can acknowledge without
    acting (#201). Only the fan rows say whether the restore happened. Returns
    False once ``timeout_s`` passes without a verified reading. Shared by the
    in-process restore path and the detached fan-restore sidecar.
    """

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        try:
            if _summary_indicates_auto(fan_summary()):
                return True
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.05, float(poll_interval_s)))


def set_thermal_profile_verified(
    profile: str,
    *,
    settle_seconds: float = 1.0,
    require_actual_ramp: bool = False,
    actual_ramp_timeout_s: float = 20.0,
    actual_ramp_poll_interval_s: float = 1.0,
    actual_ramp_fraction: float = FAN_RAMP_TARGET_FRACTION,
    log: Any = None,
) -> dict[str, Any]:
    """Set a fan profile and *prove* it took effect.

    ``log`` is an optional callable that receives one human-readable line per
    diagnostic step (e.g. ``print`` or a logger.info).

    Returns a dict with:
      ``ok``         True iff fans are confirmed ramped (or the profile was
                     ``silent``, where we don't verify upward motion).
      ``baseline``   ``fan_summary`` before the profile change.
      ``after``      ``fan_summary`` after waiting ``settle_seconds``.
      ``profile``    The profile we set.
      ``set_result`` Underlying set_thermal_profile() output.
      ``message``    Human-readable summary suitable for printing.
      ``actionable`` (when ``ok`` is False) Suggested next step for the user.
    """

    def _emit(line: str) -> None:
        if log is not None:
            try:
                log(line)
            except Exception:
                pass

    detection = detect_thermal_control()
    if not detection.get("available"):
        return {
            "ok": False,
            "baseline": None,
            "after": None,
            "profile": profile,
            "set_result": {"ok": False, "detection": detection},
            "message": detection.get("instructions") or "no fan controller installed",
            "actionable": "run `mtplx max --install` to auto-install ThermalForge",
        }

    selected = detection["selected"]
    tool_kind = selected.get("kind")
    tool_path = selected.get("path")
    _emit(f"[max] fan tool: {tool_kind} ({tool_path})")

    baseline = fan_summary()
    if baseline.get("ok"):
        _emit(
            f"[max] baseline fans: min {baseline['min_rpm']} RPM, "
            f"max {baseline['max_rpm']} RPM"
        )
    else:
        _emit("[max] baseline fans: no reading (daemon may be inactive)")

    _emit(f"[max] running `{tool_kind} {'max' if profile != 'silent' else 'auto'}`...")
    set_result = set_thermal_profile(profile)
    if not set_result.get("ok"):
        return {
            "ok": False,
            "baseline": baseline,
            "after": None,
            "profile": profile,
            "set_result": set_result,
            "message": (
                f"`{tool_kind}` did not accept the command. "
                f"stderr: {(set_result.get('attempts') or [{}])[-1].get('stderr', '')}"
            ),
            "actionable": (
                "Do not run `sudo mtplx tune`; it changes the MTPLX user/cache "
                "context. Run `mtplx max --grant-sudo` once, or open "
                "/Applications/ThermalForge.app once and enable 'Launch at Login', "
                "or run `sudo thermalforge install` to (re)install the daemon. "
                "Then re-run `mtplx tune` as your normal user."
            ),
        }

    if profile == "silent":
        # Don't try to verify silent — fans may legitimately stay where they
        # were (the daemon eases off rather than slamming low).
        return {
            "ok": True,
            "baseline": baseline,
            "after": fan_summary(),
            "profile": profile,
            "set_result": set_result,
            "message": "fan profile restored",
        }

    if settle_seconds > 0:
        _emit(f"[max] waiting {settle_seconds:.1f}s for the daemon to update target RPM...")
        import time as _time

        _time.sleep(settle_seconds)

    after = fan_summary()
    if not _summary_indicates_max(after):
        # Helpful failure message: include what we did read so the user can
        # see whether fans were genuinely ignored vs. just slow to ramp.
        target = ", ".join(
            f"fan{f.get('raw', {}).get('index', '?')}: target={f.get('target_rpm')} mode={f.get('mode')}"
            for f in (after.get("fans") or [])
        ) or "no fan readings"
        return {
            "ok": False,
            "baseline": baseline,
            "after": after,
            "profile": profile,
            "set_result": set_result,
            "message": (
                f"`{tool_kind} {profile}` was accepted but the fan daemon is "
                f"not commanding max. State: {target}"
            ),
            "actionable": (
                "open /Applications/ThermalForge.app once (this wakes the "
                "menu-bar app and confirms the daemon is talking), then re-run."
            ),
        }

    if require_actual_ramp:
        target_label = _rpm_range(after.get("target_min_rpm"), after.get("target_max_rpm"))
        _emit(f"[max] target accepted: {target_label}; waiting for actual fans to ramp...")
        deadline = time.monotonic() + max(0.0, float(actual_ramp_timeout_s))
        while not _summary_indicates_actual_ramp(after, fraction=float(actual_ramp_fraction)):
            if time.monotonic() >= deadline:
                break
            time.sleep(max(0.1, float(actual_ramp_poll_interval_s)))
            after = fan_summary()
        if not _summary_indicates_actual_ramp(after, fraction=float(actual_ramp_fraction)):
            target_label = _rpm_range(after.get("target_min_rpm"), after.get("target_max_rpm"))
            actual_label = _rpm_range(after.get("actual_min_rpm"), after.get("actual_max_rpm"))
            return {
                "ok": False,
                "baseline": baseline,
                "after": after,
                "profile": profile,
                "set_result": set_result,
                "message": (
                    f"`{tool_kind} {profile}` was accepted and target is {target_label}, "
                    f"but actual fan RPM did not ramp before timeout. Actual: {actual_label}"
                ),
                "actionable": (
                    "run `mtplx max --status --json` and check ThermalForge fan "
                    "actual_rpm readings; MTPLX will not claim Max until the "
                    "hardware ramp is visible."
                ),
            }

        actual_label = _rpm_range(after.get("actual_min_rpm"), after.get("actual_max_rpm"))
        target_label = _rpm_range(after.get("target_min_rpm"), after.get("target_max_rpm"))
        _emit(f"[max] fans ramped: actual {actual_label}; target {target_label}")
        return {
            "ok": True,
            "baseline": baseline,
            "after": after,
            "profile": profile,
            "set_result": set_result,
            "message": f"fans ramped to max (actual {actual_label}; target {target_label})",
        }

    target_label = _rpm_range(after.get("target_min_rpm"), after.get("target_max_rpm"))
    actual_label = _rpm_range(after.get("actual_min_rpm"), after.get("actual_max_rpm"))
    _emit(
        f"[max] fans commanded: target {target_label} (actual will catch "
        f"up over ~15s); current actual {actual_label}"
    )
    return {
        "ok": True,
        "baseline": baseline,
        "after": after,
        "profile": profile,
        "set_result": set_result,
        "message": f"fans commanded to max ({target_label} target)",
    }


def restore_thermal_profile_verified(
    *,
    log: Any = None,
    settle_timeout_s: float = 12.0,
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Restore Apple-default fan control and prove the daemon accepted it.

    This is intentionally stricter than the old ``set_thermal_profile("silent")``
    cleanup path. Marker files are only safe to delete once we know the fan
    controller reports the automatic fan curve again.
    """

    def _emit(line: str) -> None:
        if log is not None:
            try:
                log(line)
            except Exception:
                pass

    _emit("[max] restoring fans to Apple auto curve...")
    set_result = set_thermal_profile("silent")
    deadline = time.monotonic() + max(0.0, float(settle_timeout_s))
    after = fan_summary()
    while bool(set_result.get("ok")) and not _summary_indicates_auto(after):
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.1, float(poll_interval_s)))
        after = fan_summary()
    ok = bool(set_result.get("ok")) and _summary_indicates_auto(after)
    message = "fan profile restored" if ok else "fan restore was attempted but not verified"
    if ok:
        _emit("[max] fans restored: mode=auto")
    else:
        _emit(f"[max] WARNING: {message}")
    return {
        "ok": ok,
        "profile": "silent",
        "set_result": set_result,
        "after": after,
        "message": message,
    }


def open_thermalforge_app() -> dict[str, Any]:
    """Best-effort `open /Applications/ThermalForge.app` to wake the menu-bar
    app, which in turn ensures the daemon is talking. Returns ``{"ok": ...}``.
    """

    app_path = "/Applications/ThermalForge.app"
    if not os.path.isdir(app_path):
        return {"ok": False, "message": f"{app_path} not found"}
    try:
        subprocess.run(["open", "-g", app_path], check=False, timeout=5.0)
    except Exception as exc:
        return {"ok": False, "message": f"failed to open: {exc}"}
    return {"ok": True, "message": f"opened {app_path}"}


SUDOERS_FILE = "/etc/sudoers.d/mtplx-thermalforge"

# The one privileged step, shared by the terminal (``sudo``) and GUI
# (``security execute-with-privileges``) paths. It receives the finished,
# already-validated rule line and writes it to a temporary name that sudo
# ignores (sudoers.d skips file names containing a '.'), sets owner and
# mode, re-checks the fragment with visudo as root, and only then renames it
# into place. A rejected fragment is removed by the trap and never becomes
# live, so a bad rule can never take every sudo on the machine down with it.
_PRIVILEGED_SUDOERS_SCRIPT = r"""
set -eu
sudoers_file="$1"
rule_line="$2"
binary_path="$3"

case "$sudoers_file" in
  /etc/sudoers.d/*) ;;
  *) echo "refusing to write outside /etc/sudoers.d" >&2; exit 64 ;;
esac

case "$binary_path" in
  /*) ;;
  *) echo "thermalforge path must be absolute" >&2; exit 64 ;;
esac

if [ ! -x "$binary_path" ]; then
  echo "thermalforge binary is not executable: $binary_path" >&2
  exit 65
fi

tmp_file="${sudoers_file}.tmp.$$"
trap 'rm -f "$tmp_file"' EXIT

umask 077
printf '%s\n' "$rule_line" > "$tmp_file"
chown root:wheel "$tmp_file"
chmod 440 "$tmp_file"
/usr/sbin/visudo -c -f "$tmp_file" >/dev/null
mv "$tmp_file" "$sudoers_file"
trap - EXIT
"""

# Inside a sudoers command path these characters are accepted only when
# backslash-escaped, and the parser strips the backslash again, so escaping
# them yields exactly the on-disk path (sudo: plugins/sudoers/toke.l PATH
# token and toke_util.c SPECIAL()). An unescaped space is not a syntax
# error: it ends the command and turns the rest of the path into an
# argument, which visudo happily accepts.
_SUDOERS_PATH_ESCAPE = frozenset(", :=#")
# The escape character itself cannot appear in a path token; '*', '?', '[',
# ']' are wildcards when sudo matches a command, so a rule containing them
# would cover more than the one binary; quotes have no place in a binary
# path. These are refused rather than escaped.
_SUDOERS_PATH_REJECT = frozenset("\\*?[]\"'")
_SUDOERS_USER_RE = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*")

# Interactive sudo waits for a password; a non-interactive ``sudo -n`` either
# succeeds on cached credentials or fails at once. Both are bounded so a
# prompt nobody can see never wedges the caller.
SUDO_PROMPT_TIMEOUT_S = 180.0
SUDO_NONINTERACTIVE_TIMEOUT_S = 20.0


def sudoers_rule_for(user: str, binary_path: str) -> str:
    """Return the one-line sudoers rule granting ``user`` passwordless root for
    exactly ``binary_path`` (no trailing newline).

    Raises ``ValueError`` with a plain message when either value cannot be
    written safely.
    """

    if not _SUDOERS_USER_RE.fullmatch(user or ""):
        raise ValueError(f"the user name {user!r} cannot be written to a sudoers rule")
    if not binary_path.startswith("/"):
        raise ValueError("the thermalforge path must be absolute")
    for ch in binary_path:
        if ch in _SUDOERS_PATH_REJECT or not ch.isprintable() or (ch.isspace() and ch != " "):
            raise ValueError(
                f"the thermalforge path {binary_path!r} contains a character "
                "that cannot be written to a sudoers rule"
            )
    escaped = "".join(f"\\{ch}" if ch in _SUDOERS_PATH_ESCAPE else ch for ch in binary_path)
    return f"{user} ALL=(root) NOPASSWD: {escaped}"


def _current_user() -> str:
    for key in ("USER", "LOGNAME"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return ""


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (AttributeError, ValueError):
        return False


def _visudo_check_rule(rule_line: str) -> dict[str, Any]:
    """Parse ``rule_line`` with ``visudo -c -f`` on a temporary file we own.

    ``-f`` skips the owner and mode checks, so this needs no privilege and
    runs before any password prompt. It catches syntax errors only; the
    character rules in :func:`sudoers_rule_for` are what keep the command
    path intact.
    """

    visudo = shutil.which("visudo") or "/usr/sbin/visudo"
    fd, tmp_path = tempfile.mkstemp(prefix="mtplx-sudoers-", suffix=".check")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(rule_line + "\n")
        result = _run_probe([visudo, "-c", "-f", tmp_path], timeout_s=10.0)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    result["step"] = "visudo_check"
    return result


def _install_sudoers_rule_with_sudo(
    *,
    rule_line: str,
    binary_path: str,
    interactive: bool,
) -> dict[str, Any]:
    """Run the privileged install script through ``sudo``.

    With ``interactive`` false the call uses ``sudo -n`` and cannot prompt;
    it either rides cached credentials or fails immediately.
    """

    command = ["sudo"]
    if not interactive:
        command.append("-n")
    command += ["/bin/sh", "-c", _PRIVILEGED_SUDOERS_SCRIPT, "sh", SUDOERS_FILE, rule_line, binary_path]
    timeout_s = SUDO_PROMPT_TIMEOUT_S if interactive else SUDO_NONINTERACTIVE_TIMEOUT_S
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "step": "sudo_install",
            "command": ["sudo", *([] if interactive else ["-n"]), "/bin/sh", "-c", "..."],
            "returncode": None,
            "stdout": "",
            "stderr": f"sudo did not finish within {int(timeout_s)} seconds",
        }
    except Exception as exc:
        return {
            "ok": False,
            "step": "sudo_install",
            "command": ["sudo", *([] if interactive else ["-n"]), "/bin/sh", "-c", "..."],
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": proc.returncode == 0,
        "step": "sudo_install",
        "command": ["sudo", *([] if interactive else ["-n"]), "/bin/sh", "-c", "..."],
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def _install_sudoers_rule_with_security(
    *,
    rule_line: str,
    binary_path: str,
) -> dict[str, Any]:
    security = shutil.which("security")
    if security is None and os.path.exists("/usr/bin/security"):
        security = "/usr/bin/security"
    if security is None:
        return {
            "ok": False,
            "step": "security_execute_with_privileges",
            "message": "macOS security tool was not found for GUI admin authorization.",
        }

    command = [
        security,
        "execute-with-privileges",
        "/bin/sh",
        "-c",
        _PRIVILEGED_SUDOERS_SCRIPT,
        "sh",
        SUDOERS_FILE,
        rule_line,
        binary_path,
    ]
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300.0,
        )
    except Exception as exc:
        return {
            "ok": False,
            "step": "security_execute_with_privileges",
            "command": [security, "execute-with-privileges", "/bin/sh", "-c", "..."],
            "message": f"Admin authorization failed: {type(exc).__name__}: {exc}",
        }

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    detail = stderr or stdout
    return {
        "ok": proc.returncode == 0,
        "step": "security_execute_with_privileges",
        "command": [security, "execute-with-privileges", "/bin/sh", "-c", "..."],
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "message": (
            "Passwordless sudoers rule installed with macOS admin authorization."
            if proc.returncode == 0
            else f"macOS admin authorization failed: {detail or f'exit {proc.returncode}'}"
        ),
    }


def install_passwordless_sudoers_rule(
    *,
    binary_path: str | None = None,
    streaming: bool = True,
) -> dict[str, Any]:
    """Install a sudoers rule allowing the current user to run ``thermalforge``
    without a password, so MTPLX's fan control doesn't prompt every server
    start (and the idle watchdog can ramp fans back up unattended).

    Why this is necessary: ThermalForge's privileged daemon does not fully
    proxy SMC fan-unlock writes; ``thermalforge max`` returns
    "Run with sudo." even with the daemon running. The sudoers rule scopes
    NOPASSWD to exactly the ``thermalforge`` binary, which is the minimum
    elevation needed for fan control.

    The rule is built and character-checked in Python, parsed by visudo on a
    temporary file we own, and only then handed to one privileged script
    that installs it atomically (see ``_PRIVILEGED_SUDOERS_SCRIPT``). Nothing
    unvalidated ever reaches ``/etc/sudoers.d``. ``streaming`` is accepted
    for existing callers; sudo prompts on the terminal itself in both modes.
    """

    # Grant exactly the binary the fan commands run. `mtplx max --grant-sudo`
    # used to default to the PATH copy while fan control ran MTPLX's own
    # ~/.mtplx/bin/thermalforge, so the grant never applied.
    if binary_path is None:
        binary_path = _find_thermalforge()
        if binary_path is None:
            return {
                "ok": False,
                "step": "locate_thermalforge",
                "message": "ThermalForge is not installed. Run `mtplx max --install` first.",
            }

    user = _current_user()
    try:
        rule_line = sudoers_rule_for(user, binary_path)
    except ValueError as exc:
        return {
            "ok": False,
            "step": "build_rule",
            "binary_path": binary_path,
            "message": f"Passwordless sudo was not configured: {exc}.",
        }

    # Check the rule before anything privileged runs, on a file we own.
    check = _visudo_check_rule(rule_line)
    if not check.get("ok"):
        return {
            "ok": False,
            "step": "visudo_check",
            "binary_path": binary_path,
            "rule": rule_line,
            "message": (
                "Passwordless sudo was not configured: visudo rejected the rule, "
                f"so nothing was installed. {(check.get('stderr') or '').strip()}"
            ).strip(),
            "steps": [check],
        }

    steps: list[dict[str, Any]] = [check]
    interactive = _stdin_is_tty()
    install = _install_sudoers_rule_with_sudo(
        rule_line=rule_line, binary_path=binary_path, interactive=interactive
    )
    steps.append(install)
    method = "sudo"
    if not install.get("ok"):
        # No terminal to type a password into (the GUI app, a piped or
        # background `mtplx start`), or sudo refused: ask through macOS's
        # admin authorization dialog instead of hanging on a prompt.
        security_result = _install_sudoers_rule_with_security(
            rule_line=rule_line, binary_path=binary_path
        )
        steps.append(security_result)
        method = "security_execute_with_privileges"
        if not security_result.get("ok"):
            parts = [f"Could not write {SUDOERS_FILE}."]
            sudo_said = (install.get("stderr") or "").strip()
            if sudo_said:
                parts.append(f"sudo said: {sudo_said}.")
            if security_result.get("message"):
                parts.append(str(security_result["message"]))
            if not interactive:
                parts.append("Run `mtplx max --grant-sudo` in a terminal to enter your password.")
            return {
                "ok": False,
                "step": "sudo_install",
                "binary_path": binary_path,
                "message": " ".join(parts),
                "steps": steps,
            }

    # Final sanity: passwordless sudo should now work for this exact binary.
    probe = _run_probe(["sudo", "-n", binary_path, "status"], timeout_s=5.0)
    steps.append({**probe, "step": "verify"})
    return {
        "ok": probe.get("ok", False),
        "step": "verify_passwordless",
        "method": method,
        "binary_path": binary_path,
        "rule": rule_line,
        "message": (
            "Passwordless sudo for thermalforge is configured."
            if probe.get("ok")
            else (
                "The sudoers rule was installed, but `sudo -n thermalforge status` "
                "still failed: " + (probe.get("stderr") or "").strip()
            )
        ),
        "steps": steps,
    }


def remove_passwordless_sudoers_rule(*, streaming: bool = True) -> dict[str, Any]:
    """Counterpart to ``install_passwordless_sudoers_rule`` for clean uninstall."""

    runner = _run_streaming if streaming else _run_probe
    if not os.path.exists(SUDOERS_FILE):
        return {"ok": True, "message": f"{SUDOERS_FILE} already absent"}
    result = runner(["sudo", "rm", "-f", SUDOERS_FILE], timeout_s=SUDO_PROMPT_TIMEOUT_S)
    return {
        "ok": result.get("ok", False),
        "message": (
            f"removed {SUDOERS_FILE}"
            if result.get("ok")
            else f"failed to remove {SUDOERS_FILE}"
        ),
        "result": result,
    }


# ---------- max-mode crash safety -------------------------------------------
#
# When `mtplx start --max` pins the fans, we also write a marker file with
# the pid that did the pinning. If that process dies *cleanly* (Ctrl-C,
# SIGTERM, normal exit) we run `thermalforge auto` and clear the marker. If
# it dies hard (kill -9, terminal slammed shut, OOM), the marker stays
# behind. The next MTPLX invocation detects the stale marker, sees the pid
# isn't running anymore, and restores fans automatically — so the user
# doesn't end up with a screaming Mac because of a previous crash.

import atexit  # noqa: E402  (deferred until after main module body)
import fcntl  # noqa: E402  (macOS process-wide marker coordination)
import json as _json  # noqa: E402  (avoid clashing with local json imports)
import secrets  # noqa: E402
import signal  # noqa: E402
from pathlib import Path  # noqa: E402

MAX_MARKER_FILE = Path("~/.mtplx/max-active.json").expanduser()


@contextmanager
def _max_marker_lock(marker_path: str | Path | None = None) -> Iterator[None]:
    """Serialize fan-owner handoffs across overlapping MTPLX processes."""

    handle = None
    try:
        target = Path(marker_path) if marker_path is not None else MAX_MARKER_FILE
        lock_path = Path(f"{target}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except Exception:
        # Marker ownership is crash-safety, not permission to make `serve`
        # unusable on an unusual filesystem. The marker write below remains
        # best-effort, matching the historical behavior.
        if handle is not None:
            handle.close()
        handle = None
    try:
        yield
    finally:
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            handle.close()


def _max_marker_owned_by(
    marker_path: str | Path | None,
    owner_token: str | None,
) -> bool:
    """True for legacy sidecars, or when the current lease is still theirs."""

    if not owner_token:
        return True
    if marker_path is None:
        return False
    try:
        loaded = _json.loads(Path(marker_path).read_text())
        return isinstance(loaded, dict) and loaded.get("owner_token") == owner_token
    except Exception:
        return False


def _write_max_marker(pid: int | None = None) -> str | None:
    if pid is None:
        pid = os.getpid()
    owner_token = secrets.token_hex(16)
    binary = None
    try:
        selected = detect_thermal_control().get("selected")
        if selected:
            binary = selected.get("path")
    except Exception:
        binary = None
    with _max_marker_lock():
        try:
            MAX_MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
            MAX_MARKER_FILE.write_text(
                _json.dumps(
                    {
                        "pid": int(pid),
                        "owner_token": owner_token,
                        "started_at": time.time(),
                        "binary": binary or _find_thermalforge(),
                    }
                )
            )
            return owner_token
        except Exception:
            # Marker is best-effort; don't crash --max because we can't write it.
            return None


def _clear_max_marker_unlocked() -> None:
    if MAX_MARKER_FILE.exists():
        MAX_MARKER_FILE.unlink()


def _clear_max_marker() -> None:
    with _max_marker_lock():
        try:
            _clear_max_marker_unlocked()
        except Exception:
            pass


def _read_max_marker_unlocked() -> dict[str, Any] | None:
    if not MAX_MARKER_FILE.exists():
        return None
    loaded = _json.loads(MAX_MARKER_FILE.read_text())
    return loaded if isinstance(loaded, dict) else None


def _read_max_marker() -> dict[str, Any] | None:
    with _max_marker_lock():
        try:
            return _read_max_marker_unlocked()
        except Exception:
            return None


def check_and_recover_stale_max() -> dict[str, Any]:
    """Detect a marker left by a crashed --max session and restore fans.

    Returns ``{"recovered": bool, "stale_pid": int|None,
    "still_running": bool}``. Safe to call from any startup path; no-op when
    no marker exists.
    """

    # Hold the ownership lock through restore. A newer daemon waits here,
    # writes its own lease after Auto is confirmed, then commands Max; an old
    # cleanup can no longer land between those steps and undo the new pin.
    with _max_marker_lock():
        try:
            marker = _read_max_marker_unlocked()
        except Exception:
            marker = None
        if not marker:
            return {"recovered": False, "stale_pid": None, "still_running": False}
        stale_pid = marker.get("pid")
        if isinstance(stale_pid, int):
            try:
                os.kill(stale_pid, 0)
                return {
                    "recovered": False,
                    "stale_pid": stale_pid,
                    "still_running": True,
                }
            except OSError:
                pass  # process is gone, marker is stale
        restore = restore_thermal_profile_verified()
        if restore.get("ok"):
            try:
                _clear_max_marker_unlocked()
            except Exception:
                pass
        return {
            "recovered": bool(restore.get("ok")),
            "stale_pid": stale_pid,
            "still_running": False,
            "restore": restore,
            "marker_cleared": bool(restore.get("ok")),
        }


def _spawn_thermal_sidecar(owner_token: str | None = None) -> subprocess.Popen | None:
    """Launch a detached fan-restore watchdog.

    Required because closing a macOS Terminal window sends SIGHUP and
    waits ~5 s before SIGKILL. The signal handler we install in this
    process can take longer than that (the un-sudo'd
    ``thermalforge auto`` candidate can block on the daemon for up to
    15 s), so SIGKILL ends up landing while fans are still pinned. The
    sidecar runs in its own session, survives the parent's death, and
    issues ``sudo -n thermalforge auto`` the moment the parent
    disappears.
    """

    detection = detect_thermal_control()
    selected = detection.get("selected") if isinstance(detection, dict) else None
    if not selected:
        return None
    binary = str(selected.get("path") or "")
    if not binary:
        return None
    sidecar_module = "mtplx.thermal_sidecar"
    cmd = [
        sys.executable,
        "-m",
        sidecar_module,
        "--parent-pid",
        str(os.getpid()),
        "--binary",
        binary,
        "--marker",
        str(MAX_MARKER_FILE),
    ]
    if owner_token:
        cmd.extend(["--owner-token", owner_token])
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        return None


def install_max_lifecycle_hooks() -> Any:
    """Wire signal handlers + atexit + a detached sidecar so fans always
    get restored when the parent process exits — including SIGKILL,
    terminal-close, OOM, and other uncatchable cases.

    Returns a callable that runs the in-process cleanup explicitly;
    callers should also invoke it from their own ``finally`` block as
    belt-and-suspenders alongside the sidecar.
    """

    owner_token = _write_max_marker()
    _sidecar = _spawn_thermal_sidecar(owner_token)
    cleaned_up = [False]

    def cleanup() -> dict[str, Any]:
        if cleaned_up[0]:
            return {"ok": True, "already_cleaned": True}
        cleaned_up[0] = True
        with _max_marker_lock():
            try:
                marker = _read_max_marker_unlocked()
            except Exception:
                marker = None
            if owner_token and (not marker or marker.get("owner_token") != owner_token):
                # A newer Max session owns the machine (or the user already
                # cleared our lease). Old cleanup must never switch it to Auto.
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "max_owner_changed",
                }
            try:
                restore = restore_thermal_profile_verified()
            except Exception as exc:
                restore = {
                    "ok": False,
                    "error": str(exc),
                    "message": "fan restore raised",
                }
            if restore.get("ok"):
                try:
                    _clear_max_marker_unlocked()
                except Exception:
                    pass
        # The sidecar will notice the parent is gone and re-issue auto
        # too when this lease is still current. If another process has already
        # taken ownership, both cleanup paths leave its Max pin untouched.
        return restore

    def _signal_handler(signum: int, _frame: Any) -> None:
        cleanup()
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except Exception:
            os._exit(128 + int(signum))

    atexit.register(cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # signal.signal can only be called from the main thread of the
            # main interpreter; ignore in non-standard contexts (tests).
            pass
    return cleanup


class MaxSession:
    """Verified, crash-safe lifecycle for every user-facing ``--max`` path."""

    def __init__(
        self,
        *,
        log: Any = None,
        retry_open_app: bool = True,
        require_actual_ramp: bool = True,
        actual_ramp_timeout_s: float = 25.0,
    ) -> None:
        self.log = log
        self.retry_open_app = bool(retry_open_app)
        self.require_actual_ramp = bool(require_actual_ramp)
        self.actual_ramp_timeout_s = float(actual_ramp_timeout_s)
        self.cleanup: Any = None
        self.active = False
        self.thermal: dict[str, Any] = {
            "enabled": False,
            "start": None,
            "verified": None,
            "restore": None,
        }

    def _emit(self, line: str) -> None:
        if self.log is not None:
            try:
                self.log(line)
            except Exception:
                pass

    def start(self) -> bool:
        recovery = check_and_recover_stale_max()
        self.thermal["recovery"] = recovery
        if recovery.get("recovered"):
            self._emit(
                "[max] recovered fans from a previous --max session that did "
                f"not exit cleanly (stale pid {recovery.get('stale_pid')})"
            )
        elif recovery.get("restore") and not recovery.get("marker_cleared"):
            self._emit(
                "[max] WARNING: previous --max marker is stale, but fan restore "
                "could not be verified; leaving the marker for the next status check"
            )

        detection = detect_thermal_control()
        if not detection.get("available"):
            verified = set_thermal_profile_verified("performance", log=self._emit)
            self.thermal["verified"] = verified
            self.thermal["start"] = verified.get("set_result")
            return False

        # Install marker, signal hooks, and detached sidecar before commanding
        # max. That closes the small but important crash window during live
        # verification itself.
        self.cleanup = install_max_lifecycle_hooks()
        verified = set_thermal_profile_verified(
            "performance",
            log=self._emit,
            require_actual_ramp=self.require_actual_ramp,
            actual_ramp_timeout_s=self.actual_ramp_timeout_s,
        )
        if not verified.get("ok") and self.retry_open_app:
            self._emit(f"[max] FIRST ATTEMPT FAILED: {verified.get('message')}")
            actionable = verified.get("actionable")
            if actionable:
                self._emit(f"[max] trying recovery: {str(actionable).split('.')[0]}...")
            open_result = open_thermalforge_app()
            self.thermal["open_thermalforge_app"] = open_result
            if open_result.get("ok"):
                self._emit("[max] opened ThermalForge.app; retrying in 4s...")
                time.sleep(4.0)
                verified = set_thermal_profile_verified(
                    "performance",
                    log=self._emit,
                    require_actual_ramp=self.require_actual_ramp,
                    actual_ramp_timeout_s=self.actual_ramp_timeout_s,
                )

        self.thermal["verified"] = verified
        self.thermal["start"] = verified.get("set_result", {"ok": False})
        if not verified.get("ok"):
            self.stop()
            return False

        self.active = True
        self.thermal["enabled"] = True
        return True

    def stop(self) -> dict[str, Any]:
        restore: dict[str, Any]
        if self.cleanup is not None:
            restore = self.cleanup()
        else:
            restore = restore_thermal_profile_verified(log=self._emit)
            if restore.get("ok"):
                _clear_max_marker()
        self.active = False
        self.thermal["restore"] = restore
        return restore

    def __enter__(self) -> "MaxSession":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self.active or self.cleanup is not None:
            self.stop()


_DESKTOP_MAC_MODEL_PREFIXES = ("MacStudio", "MacPro", "Macmini", "iMac")
_LAPTOP_MAC_MODEL_PREFIXES = ("MacBook",)
_DESKTOP_MAC_MODELS = frozenset(
    {
        "Mac13,1",
        "Mac13,2",
        "Mac14,3",
        "Mac14,8",
        "Mac14,12",
        "Mac14,13",
        "Mac14,14",
        "Mac15,4",
        "Mac15,5",
        "Mac15,14",
        "Mac16,2",
        "Mac16,3",
        "Mac16,9",
        "Mac16,10",
        "Mac16,11",
    }
)
_LAPTOP_MAC_MODELS = frozenset(
    {
        "Mac14,2",
        "Mac14,5",
        "Mac14,6",
        "Mac14,7",
        "Mac14,9",
        "Mac14,10",
        "Mac14,15",
        "Mac15,3",
        "Mac15,6",
        "Mac15,7",
        "Mac15,8",
        "Mac15,9",
        "Mac15,10",
        "Mac15,11",
        "Mac15,12",
        "Mac15,13",
        "Mac16,1",
        "Mac16,5",
        "Mac16,6",
        "Mac16,7",
        "Mac16,8",
        "Mac16,12",
        "Mac16,13",
        "Mac17,2",
        "Mac17,6",
        "Mac17,7",
    }
)


def _sysctl_hw_model() -> str:
    if sys.platform != "darwin":
        return ""
    probe = _run_probe(["/usr/sbin/sysctl", "-n", "hw.model"], timeout_s=1.0)
    return str(probe.get("stdout") or "").strip() if probe.get("ok") else ""


def _has_internal_battery() -> bool | None:
    if sys.platform != "darwin":
        return None
    probe = _run_probe(["/usr/bin/pmset", "-g", "batt"], timeout_s=1.0)
    stdout = str(probe.get("stdout") or "")
    if not probe.get("ok") or not stdout.strip():
        return None
    return "InternalBattery" in stdout


def is_desktop_mac() -> bool:
    """Return True on desktop Macs (Mac Studio / Mac Pro / Mac mini / iMac)."""
    override = (os.environ.get("MTPLX_DESKTOP_MAC") or "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    model = _sysctl_hw_model()
    if not model:
        return False
    if model in _DESKTOP_MAC_MODELS or any(
        model.startswith(prefix) for prefix in _DESKTOP_MAC_MODEL_PREFIXES
    ):
        return True
    if model in _LAPTOP_MAC_MODELS or any(
        model.startswith(prefix) for prefix in _LAPTOP_MAC_MODEL_PREFIXES
    ):
        return False
    has_batt = _has_internal_battery()
    return has_batt is False


def _verify_rpm_target(summary: dict[str, Any], target_rpm: int) -> bool:
    fans = _ok_fans(summary)
    if not fans:
        return False
    tol = max(250, int(0.12 * target_rpm))
    for fan in fans:
        mode = str(fan.get("mode") or "").lower()
        target = _int_or_none(fan.get("target_rpm"))
        if mode != "auto" and target is not None and abs(target - target_rpm) <= tol:
            return True
    return False


def set_fan_rpm(rpm: int, *, dry_run: bool = False) -> dict[str, Any]:
    """Set manual target fan RPM via ThermalForge daemon socket or CLI."""
    target_rpm = max(1000, int(rpm))
    detection = detect_thermal_control()
    selected = detection.get("selected")
    if not selected or str(selected.get("kind")) != "thermalforge":
        return {
            "ok": False,
            "unsupported_controller": True,
            "rpm": target_rpm,
            "dry_run": dry_run,
            "detection": detection,
            "attempts": [],
            "message": "ThermalForge is required to set intermediate fan RPM.",
        }
    path = str(selected["path"])
    commands = [
        ["sudo", "-n", path, "set", str(target_rpm)],
        [path, "set", str(target_rpm)],
    ]
    if dry_run:
        return {
            "ok": True,
            "rpm": target_rpm,
            "dry_run": True,
            "detection": detection,
            "command": commands[0],
            "attempts": [],
        }
    attempts: list[dict[str, Any]] = []
    sock_res = _daemon_socket_send(f"set {target_rpm}")
    if sock_res is not None:
        attempts.append(sock_res)
        if sock_res.get("ok"):
            verified = _verify_rpm_target(fan_summary(), target_rpm)
            sock_res["verified"] = verified
            if verified:
                return {
                    "ok": True,
                    "rpm": target_rpm,
                    "dry_run": False,
                    "detection": detection,
                    "command": sock_res["command"],
                    "attempts": attempts,
                }
    for command in commands:
        result = _run_probe(command, timeout_s=15.0)
        attempts.append(result)
        if result.get("ok"):
            verified = _verify_rpm_target(fan_summary(), target_rpm)
            result["verified"] = verified
            if verified:
                return {
                    "ok": True,
                    "rpm": target_rpm,
                    "dry_run": False,
                    "detection": detection,
                    "command": command,
                    "attempts": attempts,
                }
    return {
        "ok": False,
        "rpm": target_rpm,
        "dry_run": False,
        "detection": detection,
        "attempts": attempts,
        "message": f"ThermalForge could not set fan speed to {target_rpm} RPM.",
    }


class SmartFanController:
    """Request-scoped stepped temperature-gated fan controller for Smart mode.

    Stays on Apple's quiet ``auto`` curve while a turn stays cool or short,
    ramps in smooth RPM steps as SoC temperature rises (with desktop vs laptop
    thresholds and 5C downward hysteresis), and restores ``auto`` cleanly
    after ``restore_delay_s`` (plus heat-soak drain when ramped).
    """

    _ACTUAL_RAMP_TIMEOUT_S = 30.0
    _ACTUAL_RAMP_POLL_INTERVAL_S = 1.0
    _WAIT_FOR_RESTORE_TIMEOUT_S = 30.0
    _ACTIVITY_POLL_INTERVAL_S = 5.0
    _ACTIVE_TEMP_POLL_S = 2.0
    _STEP_HYSTERESIS_C = 5.0
    _LAPTOP_STEPS = ((70.0, 0.55), (82.0, 0.80), (90.0, 1.0))
    _DESKTOP_STEPS = ((75.0, 0.45), (85.0, 0.70), (92.0, 1.0))
    _RESTORE_RETRY_BACKOFF_S = (5.0, 15.0, 30.0, 60.0)
    _SOAK_PROBE_INTERVAL_S = 5.0

    def __init__(
        self,
        *,
        log: Any = None,
        restore_delay_s: float = 2.0,
        activity_probe: Any = None,
        sensorless_delay_s: float | None = None,
        desktop: bool | None = None,
    ) -> None:
        self.log = log
        self.restore_delay_s = max(0.0, float(restore_delay_s))
        self._activity_probe = activity_probe
        if sensorless_delay_s is None:
            try:
                sensorless_delay_s = float(
                    os.environ.get("MTPLX_SMART_FAN_SENSORLESS_DELAY_S", "8")
                )
            except ValueError:
                sensorless_delay_s = 8.0
        self._sensorless_delay_s = max(0.0, float(sensorless_delay_s))
        self._desktop_mac = bool(is_desktop_mac() if desktop is None else desktop)
        try:
            self._stale_lease_idle_s = max(
                0.0, float(os.environ.get("MTPLX_SMART_FAN_STALE_LEASE_S", "120"))
            )
        except ValueError:
            self._stale_lease_idle_s = 120.0
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._active_requests: set[str] = set()
        self._cleanup: Any | None = None
        self._generation = 0
        self._detach_epoch = 0
        self._hw_op_in_flight = False
        self._current_level = 0
        self._current_target_rpm: int | None = None
        self._active_started_at: float | None = None
        self._next_temp_probe_at: float | None = None
        self._last_active_temp_c: float | None = None
        self._last_active_sensor: str | None = None
        self._step_evaluated_generation = 0
        self._commanded_max = False
        self._target_verified = False
        self._actual_ramp_verified = False
        self._ramp_requested_at: float | None = None
        self._ramp_latency_s: float | None = None
        self._actual_ramp_latency_s: float | None = None
        self._ramp_attempts = 0
        self._ramp_failed_generation: int | None = None
        self._actual_poll_deadline: float | None = None
        self._next_actual_probe_at: float | None = None
        self._idle_since: float | None = None
        self._last_transition_at: float | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._restore_unverified = False
        self._restore_retry_at: float | None = None
        self._restore_failures = 0
        self._engine_idle_since: float | None = None
        self._next_activity_probe_at: float | None = None
        self._stale_leases_reconciled = 0
        try:
            self._soak_release_temp_c = float(
                os.environ.get("MTPLX_SMART_FAN_SOAK_RELEASE_C", "75")
            )
        except ValueError:
            self._soak_release_temp_c = 75.0
        try:
            self._soak_hold_cap_s = max(
                0.0, float(os.environ.get("MTPLX_SMART_FAN_SOAK_HOLD_CAP_S", "180"))
            )
        except ValueError:
            self._soak_hold_cap_s = 180.0
        self._soak_bypass = False
        self._soak_probe_failed = False
        self._soak_holding = False
        self._soak_holds = 0
        self._soak_last_temp_c: float | None = None
        self._soak_last_sensor: str | None = None
        self._soak_next_probe_at: float | None = None
        self._soak_release_reason: str | None = None
        self._worker: threading.Thread | None = None
        self._shutdown = False

    def _emit(self, line: str) -> None:
        if self.log is not None:
            try:
                self.log(line)
            except Exception:
                pass

    def _steps(self) -> tuple[tuple[float, float], ...]:
        return self._DESKTOP_STEPS if self._desktop_mac else self._LAPTOP_STEPS

    def _desired_level_for_temp(self, temp_c: float, current_level: int) -> int:
        steps = self._steps()
        t1, t2, t3 = steps[0][0], steps[1][0], steps[2][0]
        hyst = self._STEP_HYSTERESIS_C
        if temp_c >= t3 or (current_level >= 3 and temp_c > t3 - hyst):
            return 3
        if temp_c >= t2 or (current_level >= 2 and temp_c > t2 - hyst):
            return 2
        if temp_c >= t1 or (current_level >= 1 and temp_c > t1 - hyst):
            return 1
        return 0

    def _target_rpm_for_level(self, level: int) -> int:
        fraction = self._steps()[max(0, min(2, level - 1))][1]
        summary = fan_summary()
        fans = _ok_fans(summary) or []
        lows: list[int] = []
        highs: list[int] = []
        for fan in fans:
            raw = fan.get("raw") if isinstance(fan.get("raw"), dict) else {}
            low = _int_or_none(raw.get("min_rpm"))
            high = _fan_max_capacity(fan)
            lows.append(low if low and low > 0 else 1100)
            highs.append(high if high and high > 0 else 4000)
        low_rpm = sum(lows) / len(lows) if lows else 1100.0
        high_rpm = sum(highs) / len(highs) if highs else 4000.0
        return max(1000, int(round(low_rpm + fraction * max(500.0, high_rpm - low_rpm))))

    def _ensure_lifecycle_hooks(self) -> None:
        try:
            check_and_recover_stale_max()
        except Exception as exc:
            with self._lock:
                self._last_error = f"stale_recovery:{type(exc).__name__}: {exc}"
        with self._lock:
            needs_hooks = self._cleanup is None
        if needs_hooks:
            hooks = install_max_lifecycle_hooks()
            with self._lock:
                if self._cleanup is None:
                    self._cleanup = hooks

    # -- public lease API (non-blocking) ---------------------------------

    def begin_request(self, request_id: str) -> dict[str, Any]:
        request_key = str(request_id or "request")
        with self._cond:
            if request_key in self._active_requests:
                return self._status_locked()
            was_idle = not self._active_requests
            self._active_requests.add(request_key)
            self._generation += 1
            self._idle_since = None
            self._engine_idle_since = None
            now = time.monotonic()
            if was_idle:
                self._active_started_at = now
            self._next_temp_probe_at = 0.0
            self._soak_bypass = False
            self._soak_probe_failed = False
            self._soak_holding = False
            self._soak_last_temp_c = None
            self._soak_last_sensor = None
            self._soak_next_probe_at = None
            self._soak_release_reason = None
            if not self._commanded_max and self._ramp_requested_at is None:
                self._ramp_requested_at = now
            self._ensure_worker_locked()
            self._cond.notify_all()
            return self._status_locked()

    def end_request(self, request_id: str, *, wait_for_restore: bool = False) -> dict[str, Any]:
        request_key = str(request_id or "request")
        with self._cond:
            self._active_requests.discard(request_key)
            self._generation += 1
            became_idle = not self._active_requests
            if became_idle:
                self._idle_since = time.monotonic()
                if wait_for_restore:
                    self._idle_since -= self.restore_delay_s
                    self._soak_bypass = True
            self._ensure_worker_locked()
            self._cond.notify_all()
            if not became_idle:
                return self._status_locked()
            if (
                not self._commanded_max
                and self._current_level == 0
                and self._cleanup is None
                and not self._hw_op_in_flight
            ):
                return self._status_locked()
        if wait_for_restore:
            self._wait_until_restored()
        return self.status()

    def restore_now(self, *, wait: bool = True) -> dict[str, Any]:
        with self._cond:
            self._active_requests.clear()
            self._generation += 1
            self._idle_since = time.monotonic() - self.restore_delay_s
            self._soak_bypass = True
            self._ensure_worker_locked()
            self._cond.notify_all()
        if wait:
            self._wait_until_restored()
        return self.status()

    def detach(self) -> dict[str, Any]:
        """Drop all leases and pending fan work WITHOUT touching hardware."""
        with self._cond:
            self._active_requests.clear()
            self._generation += 1
            self._detach_epoch += 1
            self._current_level = 0
            self._current_target_rpm = None
            self._active_started_at = None
            self._next_temp_probe_at = None
            self._commanded_max = False
            self._target_verified = False
            self._actual_ramp_verified = False
            self._ramp_requested_at = None
            self._actual_poll_deadline = None
            self._next_actual_probe_at = None
            self._idle_since = None
            self._restore_unverified = False
            self._restore_retry_at = None
            self._restore_failures = 0
            self._engine_idle_since = None
            self._soak_bypass = False
            self._soak_probe_failed = False
            self._soak_holding = False
            self._soak_last_temp_c = None
            self._soak_last_sensor = None
            self._soak_next_probe_at = None
            self._soak_release_reason = None
            self._cleanup = None
            self._cond.notify_all()
            return self._status_locked()

    def wait_for_ramp(self, timeout_s: float = 10.0) -> bool:
        """Block until the worker has commanded a fan boost (True) or failed (False)."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cond:
            while True:
                if self._commanded_max or self._current_level > 0:
                    return True
                if self._ramp_failed_generation == self._generation:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return bool(self._commanded_max or self._current_level > 0)
                self._cond.wait(timeout=remaining)

    def wait_for_step(self, timeout_s: float = 5.0) -> int:
        """Wait until the current lease generation has evaluated its temperature step."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cond:
            target_gen = self._generation
            while self._step_evaluated_generation < target_gen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._current_level
                self._cond.wait(timeout=remaining)
            return self._current_level

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        return {
            "active": bool(self._active_requests),
            "active_count": len(self._active_requests),
            "active_requests": sorted(self._active_requests),
            "current_level": self._current_level,
            "current_target_rpm": self._current_target_rpm,
            "profile_kind": "desktop" if self._desktop_mac else "laptop",
            "active_temp_c": self._last_active_temp_c,
            "commanded_max": bool(self._commanded_max),
            "target_verified": bool(self._target_verified),
            "actual_ramp_verified": bool(self._actual_ramp_verified),
            "ramp_latency_s": self._ramp_latency_s,
            "actual_ramp_latency_s": self._actual_ramp_latency_s,
            "ramp_attempts": self._ramp_attempts,
            "last_transition_at": self._last_transition_at,
            "last_error": self._last_error,
            "last_result": self._last_result,
            "restore_verified": not self._restore_unverified,
            "restore_failures": self._restore_failures,
            "stale_leases_reconciled": self._stale_leases_reconciled,
            "soak_release_temp_c": self._soak_release_temp_c,
            "soak_hold_cap_s": self._soak_hold_cap_s,
            "soak_holding": bool(self._soak_holding),
            "soak_holds": self._soak_holds,
            "soak_last_temp_c": self._soak_last_temp_c,
            "soak_last_sensor": self._soak_last_sensor,
            "soak_release_reason": self._soak_release_reason,
        }

    # -- worker machinery -------------------------------------------------

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="mtplx-smart-fan-worker",
            daemon=True,
        )
        self._worker.start()

    def _wait_until_restored(self) -> None:
        deadline = time.monotonic() + self._WAIT_FOR_RESTORE_TIMEOUT_S
        with self._cond:
            while (
                self._commanded_max
                or self._current_level > 0
                or self._cleanup is not None
                or self._hw_op_in_flight
            ) and not self._active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(timeout=remaining)

    def _reconciler_enabled_locked(self) -> bool:
        return self._activity_probe is not None and self._stale_lease_idle_s > 0

    def _wait_capped_by_activity_probe_locked(self, timeout: float | None, now: float) -> None:
        if self._active_requests and self._reconciler_enabled_locked():
            probe_in = max(0.05, (self._next_activity_probe_at or now) - now)
            timeout = probe_in if timeout is None else min(timeout, probe_in)
        self._cond.wait(timeout=timeout)

    def _worker_loop(self) -> None:
        while True:
            action: str | None = None
            with self._cond:
                while action is None:
                    if self._shutdown:
                        return
                    now = time.monotonic()
                    desired_active = bool(self._active_requests)
                    if (
                        desired_active
                        and self._reconciler_enabled_locked()
                        and now >= (self._next_activity_probe_at or 0.0)
                    ):
                        action = "probe_activity"
                    elif desired_active:
                        if self._ramp_failed_generation == self._generation:
                            self._wait_capped_by_activity_probe_locked(None, now)
                            continue
                        next_temp = self._next_temp_probe_at or 0.0
                        if now >= next_temp:
                            action = "step_active"
                        elif (
                            self._commanded_max
                            and not self._actual_ramp_verified
                            and self._actual_poll_deadline is not None
                            and now >= (self._next_actual_probe_at or 0.0)
                        ):
                            action = "probe_actual"
                        else:
                            waits = [max(0.01, next_temp - now)]
                            if (
                                self._commanded_max
                                and not self._actual_ramp_verified
                                and self._actual_poll_deadline is not None
                            ):
                                waits.append(
                                    max(0.05, (self._next_actual_probe_at or now) - now)
                                )
                            self._wait_capped_by_activity_probe_locked(min(waits), now)
                    elif (
                        not desired_active
                        and (self._commanded_max or self._current_level > 0 or self._cleanup is not None)
                    ):
                        if self._idle_since is None:
                            self._idle_since = now
                        remaining = self._idle_since + self.restore_delay_s - now
                        if remaining > 0:
                            self._cond.wait(timeout=remaining)
                        else:
                            soak = self._soak_decision_locked(now)
                            if soak == "probe":
                                action = "probe_soak"
                            elif soak == "hold":
                                hold_until = min(
                                    self._soak_next_probe_at or now,
                                    self._idle_since + self._soak_hold_cap_s,
                                )
                                self._cond.wait(timeout=max(0.05, hold_until - now))
                            else:
                                action = "restore"
                    elif not desired_active and self._restore_unverified:
                        retry_in = (self._restore_retry_at or now) - now
                        if retry_in <= 0:
                            action = "restore"
                        else:
                            self._cond.wait(timeout=max(0.05, retry_in))
                    else:
                        self._wait_capped_by_activity_probe_locked(None, now)
            if action == "step_active":
                self._do_step_active()
            elif action == "ramp":
                self._do_ramp()
            elif action == "restore":
                self._do_restore()
            elif action == "probe_actual":
                self._do_probe_actual()
            elif action == "probe_activity":
                self._do_probe_activity()
            elif action == "probe_soak":
                self._do_probe_soak()

    def _soak_decision_locked(self, now: float) -> str:
        if not self._commanded_max and self._current_level == 0:
            return "restore"  # stayed on auto: nothing ramped, nothing soaked
        if (
            self._soak_bypass
            or self._soak_probe_failed
            or self._soak_release_temp_c <= 0
            or self._soak_hold_cap_s <= 0
        ):
            return "restore"
        if self._idle_since is not None and now >= self._idle_since + self._soak_hold_cap_s:
            self._soak_release_reason = "hold_cap"
            return "restore"
        if (
            self._soak_last_temp_c is not None
            and self._soak_last_temp_c <= self._soak_release_temp_c
        ):
            self._soak_release_reason = "cooled"
            return "restore"
        if now >= (self._soak_next_probe_at or 0.0):
            return "probe"
        return "hold"

    def _is_stale_generation_locked(self, generation: int) -> bool:
        return (
            self._generation != generation
            or self._shutdown
            or not self._active_requests
        )

    def _is_detached_locked(self, detach_epoch: int) -> bool:
        return self._detach_epoch != detach_epoch or self._shutdown

    def _do_step_active(self) -> None:
        with self._lock:
            generation = self._generation
            detach_epoch = self._detach_epoch
            current_level = self._current_level
            active_started_at = self._active_started_at or time.monotonic()
            self._active_started_at = active_started_at
        try:
            reading = soc_temperature_c()
        except Exception:
            reading = {"ok": False, "celsius": None, "sensor": None}
        now = time.monotonic()
        with self._lock:
            if self._is_stale_generation_locked(generation):
                return
        if reading.get("ok") and reading.get("celsius") is not None:
            temp_c = float(reading["celsius"])
            sensor = reading.get("sensor")
            target_level = self._desired_level_for_temp(temp_c, current_level)
            if target_level == current_level:
                with self._cond:
                    if self._is_stale_generation_locked(generation):
                        return
                    self._last_active_temp_c = temp_c
                    self._last_active_sensor = sensor
                    self._step_evaluated_generation = max(
                        self._step_evaluated_generation, generation
                    )
                    self._next_temp_probe_at = now + self._ACTIVE_TEMP_POLL_S
                    self._cond.notify_all()
                return
            if target_level == 3:
                with self._lock:
                    if self._is_stale_generation_locked(generation):
                        return
                    self._last_active_temp_c = temp_c
                    self._last_active_sensor = sensor
                self._do_ramp()
                with self._cond:
                    if self._is_stale_generation_locked(generation):
                        return
                    self._step_evaluated_generation = max(
                        self._step_evaluated_generation, generation
                    )
                    self._next_temp_probe_at = time.monotonic() + self._ACTIVE_TEMP_POLL_S
                    self._cond.notify_all()
                return
            if target_level in (1, 2):
                rpm = self._target_rpm_for_level(target_level)
                res: dict[str, Any] = {"ok": False}
                err: str | None = None
                with self._lock:
                    if self._is_stale_generation_locked(generation):
                        return
                    self._hw_op_in_flight = True
                try:
                    for _attempt in range(2):
                        with self._lock:
                            if self._is_stale_generation_locked(generation):
                                break
                        try:
                            res = set_fan_rpm(rpm)
                        except Exception as exc:
                            res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        if res.get("ok"):
                            err = None
                            break
                        err = str(res.get("message") or res.get("error") or "set_fan_rpm failed")
                        if res.get("unsupported_controller"):
                            break
                    if res.get("unsupported_controller"):
                        with self._lock:
                            if self._is_stale_generation_locked(generation):
                                return
                            self._last_active_temp_c = temp_c
                            self._last_active_sensor = sensor
                        self._do_ramp()
                        with self._cond:
                            if self._is_stale_generation_locked(generation):
                                return
                            self._step_evaluated_generation = max(
                                self._step_evaluated_generation, generation
                            )
                            self._next_temp_probe_at = (
                                time.monotonic() + self._ACTIVE_TEMP_POLL_S
                            )
                            self._cond.notify_all()
                        return
                    if res.get("ok"):
                        with self._lock:
                            if self._is_detached_locked(detach_epoch):
                                return
                        self._ensure_lifecycle_hooks()
                    now = time.monotonic()
                    with self._cond:
                        if res.get("ok"):
                            if self._is_detached_locked(detach_epoch):
                                return
                        elif self._is_stale_generation_locked(generation):
                            return
                        self._last_active_temp_c = temp_c
                        self._last_active_sensor = sensor
                        self._last_result = res
                        self._last_transition_at = time.time()
                        if res.get("ok"):
                            self._current_level = target_level
                            self._current_target_rpm = rpm
                            self._commanded_max = False
                            self._target_verified = True
                            self._last_error = None
                            self._emit(
                                f"[smart-fan] stepped to level {target_level} "
                                f"({rpm} RPM at {temp_c:.1f}C)"
                            )
                        else:
                            self._last_error = err
                            self._ramp_failed_generation = generation
                        self._step_evaluated_generation = max(
                            self._step_evaluated_generation, generation
                        )
                        self._next_temp_probe_at = now + self._ACTIVE_TEMP_POLL_S
                        self._cond.notify_all()
                    return
                finally:
                    with self._cond:
                        self._hw_op_in_flight = False
                        self._cond.notify_all()
            # Step down to quiet auto (target_level == 0 and current_level > 0)
            with self._lock:
                if self._is_stale_generation_locked(generation):
                    return
                self._hw_op_in_flight = True
            try:
                try:
                    res = set_thermal_profile("silent")
                except Exception as exc:
                    res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                now = time.monotonic()
                with self._cond:
                    if self._is_detached_locked(detach_epoch):
                        return
                    self._last_active_temp_c = temp_c
                    self._last_active_sensor = sensor
                    self._last_result = res
                    self._last_transition_at = time.time()
                    if res.get("ok"):
                        self._current_level = 0
                        self._current_target_rpm = None
                        self._commanded_max = False
                        self._target_verified = False
                        self._actual_ramp_verified = False
                        self._last_error = None
                    else:
                        self._last_error = str(
                            res.get("message") or res.get("error") or "step-down auto failed"
                        )
                    self._step_evaluated_generation = max(
                        self._step_evaluated_generation, generation
                    )
                    self._next_temp_probe_at = now + self._ACTIVE_TEMP_POLL_S
                    self._cond.notify_all()
                return
            finally:
                with self._cond:
                    self._hw_op_in_flight = False
                    self._cond.notify_all()

        # Sensorless fallback: stay quiet on short turns, ramp after sensorless_delay_s.
        if now - active_started_at >= self._sensorless_delay_s:
            if current_level < 3:
                self._do_ramp()
            with self._cond:
                if self._is_stale_generation_locked(generation):
                    return
                self._step_evaluated_generation = max(
                    self._step_evaluated_generation, generation
                )
                self._next_temp_probe_at = time.monotonic() + self._ACTIVE_TEMP_POLL_S
                self._cond.notify_all()
            return
        with self._cond:
            if self._is_stale_generation_locked(generation):
                return
            self._step_evaluated_generation = max(
                self._step_evaluated_generation, generation
            )
            self._next_temp_probe_at = min(
                now + self._ACTIVE_TEMP_POLL_S,
                active_started_at + self._sensorless_delay_s,
            )
            self._cond.notify_all()

    def _do_probe_soak(self) -> None:
        try:
            reading = soc_temperature_c()
        except Exception:
            reading = {"ok": False, "celsius": None, "sensor": None}
        now = time.monotonic()
        with self._cond:
            if reading.get("ok") and reading.get("celsius") is not None:
                self._soak_last_temp_c = float(reading["celsius"])
                self._soak_last_sensor = reading.get("sensor")
                if (
                    self._soak_last_temp_c > self._soak_release_temp_c
                    and not self._soak_holding
                ):
                    self._soak_holding = True
                    self._soak_holds += 1
                    self._emit(
                        "[smart-fan] heat-soak hold: "
                        f"{self._soak_last_sensor} {self._soak_last_temp_c:.1f}C is above "
                        f"the {self._soak_release_temp_c:.1f}C release threshold — "
                        "holding max fans until the die cools (#227)"
                    )
                elif self._soak_last_temp_c <= self._soak_release_temp_c and self._soak_holding:
                    self._soak_holding = False
                    self._emit(
                        "[smart-fan] heat-soak drained: "
                        f"{self._soak_last_sensor} {self._soak_last_temp_c:.1f}C — releasing"
                    )
            else:
                self._soak_probe_failed = True
            self._soak_next_probe_at = now + self._SOAK_PROBE_INTERVAL_S
            self._cond.notify_all()

    def _do_ramp(self) -> None:
        with self._lock:
            generation = self._generation
            detach_epoch = self._detach_epoch
            if self._is_stale_generation_locked(generation):
                return
            requested_at = self._ramp_requested_at or time.monotonic()
            self._hw_op_in_flight = True
        try:
            self._ensure_lifecycle_hooks()
            result: dict[str, Any] | None = None
            error: str | None = None
            ok = False
            attempts = 0
            for _attempt in range(2):
                with self._lock:
                    if self._is_stale_generation_locked(generation):
                        break
                attempts += 1
                try:
                    result = set_thermal_profile("performance")
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                if not result.get("ok"):
                    error = str(result.get("message") or result.get("error") or "max command failed")
                    continue
                try:
                    summary = fan_summary()
                except Exception as exc:
                    summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                if _summary_indicates_max(summary):
                    ok = True
                    error = None
                    break
                error = "daemon accepted the command but is not commanding max target RPM"
            now = time.monotonic()
            with self._cond:
                if ok:
                    if self._is_detached_locked(detach_epoch):
                        return
                elif self._is_stale_generation_locked(generation):
                    return
                if not attempts:
                    return
                self._ramp_attempts += attempts
                self._last_result = result
                self._last_transition_at = time.time()
                self._commanded_max = ok
                self._target_verified = ok
                if ok:
                    self._current_level = 3
                    self._current_target_rpm = None
                    self._last_error = None
                    self._ramp_latency_s = now - requested_at
                    self._actual_ramp_verified = False
                    self._actual_poll_deadline = now + self._ACTUAL_RAMP_TIMEOUT_S
                    self._next_actual_probe_at = now
                    self._emit(
                        "[smart-fan] max commanded and target verified in "
                        f"{self._ramp_latency_s:.2f}s (attempt {attempts})"
                    )
                else:
                    self._last_error = error
                    self._ramp_failed_generation = generation
                    self._emit(
                        f"[smart-fan] ramp FAILED after {attempts} attempt(s): {error} "
                        "— run `mtplx max --status` to check ThermalForge"
                    )
                self._cond.notify_all()
        finally:
            with self._cond:
                self._hw_op_in_flight = False
                self._cond.notify_all()

    def _do_probe_actual(self) -> None:
        try:
            summary = fan_summary()
        except Exception:
            summary = {"ok": False}
        ramped = _summary_indicates_actual_ramp(summary)
        now = time.monotonic()
        with self._cond:
            self._next_actual_probe_at = now + self._ACTUAL_RAMP_POLL_INTERVAL_S
            if not self._commanded_max or self._actual_poll_deadline is None:
                return
            if ramped:
                self._actual_ramp_verified = True
                self._actual_poll_deadline = None
                if self._ramp_requested_at is not None:
                    self._actual_ramp_latency_s = now - self._ramp_requested_at
                    self._emit(
                        "[smart-fan] actual fan RPM ramp verified in "
                        f"{self._actual_ramp_latency_s:.1f}s"
                    )
            elif now >= self._actual_poll_deadline:
                self._actual_poll_deadline = None
                self._last_error = (
                    "target accepted but actual fan RPM did not ramp within "
                    f"{self._ACTUAL_RAMP_TIMEOUT_S:.0f}s"
                )
                self._emit(
                    f"[smart-fan] WARNING: {self._last_error} "
                    "— check `mtplx max --status`"
                )
            self._cond.notify_all()

    def _do_probe_activity(self) -> None:
        busy = True
        probe = self._activity_probe
        if probe is not None:
            try:
                busy = bool(probe())
            except Exception:
                busy = True
        now = time.monotonic()
        with self._cond:
            self._next_activity_probe_at = now + self._ACTIVITY_POLL_INTERVAL_S
            if not self._active_requests:
                self._engine_idle_since = None
                return
            if busy:
                self._engine_idle_since = None
                return
            if self._engine_idle_since is None:
                self._engine_idle_since = now
                return
            if now - self._engine_idle_since < self._stale_lease_idle_s:
                return
            stale = sorted(self._active_requests)
            self._active_requests.clear()
            self._generation += 1
            self._stale_leases_reconciled += len(stale)
            self._engine_idle_since = None
            self._idle_since = now - self.restore_delay_s
            self._emit(
                f"[smart-fan] WARNING: dropped {len(stale)} stale fan lease(s) held "
                f"while the engine was idle for {self._stale_lease_idle_s:.0f}s "
                f"({', '.join(stale)}) — restoring fans. A request path leaked its "
                "lease; please report this log line on GitHub issue #201."
            )
            self._cond.notify_all()

    def _do_restore(self) -> None:
        with self._lock:
            if self._active_requests:
                return
            cleanup = self._cleanup
            self._cleanup = None
        try:
            if cleanup is not None:
                result = cleanup()
            else:
                result = restore_thermal_profile_verified(log=self._emit)
                if result.get("ok"):
                    _clear_max_marker()
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        with self._cond:
            self._last_result = result
            self._last_transition_at = time.time()
            if result.get("ok"):
                self._current_level = 0
                self._current_target_rpm = None
                self._active_started_at = None
                self._next_temp_probe_at = None
                self._commanded_max = False
                self._target_verified = False
                self._actual_ramp_verified = False
                self._ramp_requested_at = None
                self._actual_poll_deadline = None
                self._last_error = None
                if self._restore_unverified:
                    self._emit(
                        "[smart-fan] fans verified back on the Apple auto curve "
                        f"after {self._restore_failures} failed restore attempt(s)"
                    )
                self._restore_unverified = False
                self._restore_retry_at = None
                self._restore_failures = 0
            else:
                self._last_error = str(
                    result.get("message") or result.get("error") or "restore failed"
                )
                self._current_level = 0
                self._current_target_rpm = None
                self._active_started_at = None
                self._next_temp_probe_at = None
                self._commanded_max = False
                self._target_verified = False
                self._actual_ramp_verified = False
                self._ramp_requested_at = None
                self._actual_poll_deadline = None
                self._restore_unverified = True
                self._restore_failures += 1
                backoff = self._RESTORE_RETRY_BACKOFF_S[
                    min(self._restore_failures - 1, len(self._RESTORE_RETRY_BACKOFF_S) - 1)
                ]
                self._restore_retry_at = time.monotonic() + backoff
                if self._restore_failures <= 3 or self._restore_failures % 5 == 0:
                    self._emit(
                        f"[smart-fan] restore FAILED (attempt {self._restore_failures}): "
                        f"{self._last_error} — retrying in {backoff:.0f}s; fans may still "
                        "be ramped, check `mtplx max --status`"
                    )
            self._cond.notify_all()


def set_thermal_profile(profile: str, *, dry_run: bool = False) -> dict[str, Any]:
    if profile not in PROFILE_LABELS:
        raise ValueError(f"unknown thermal profile: {profile}")
    detection = detect_thermal_control()
    selected = detection.get("selected")
    if not selected:
        return {
            "ok": False,
            "profile": profile,
            "dry_run": dry_run,
            "detection": detection,
            "attempts": [],
            "message": detection["instructions"],
        }
    commands = _profile_command_candidates(selected, profile)
    if dry_run:
        return {
            "ok": True,
            "profile": profile,
            "dry_run": True,
            "detection": detection,
            "command": commands[0] if commands else None,
            "attempts": [],
        }
    attempts: list[dict[str, Any]] = []

    # ThermalForge exposes a privileged daemon socket that resets fans as root
    # (no sudo) and, unlike the `auto` CLI, never quits the menu bar app. Prefer
    # it for the fan reset so restoring fans can't take down a running app; the
    # CLI candidates below stay the fallback when no daemon is reachable.
    #
    # #201: the daemon's "ok" reply is not proof the fans actually dropped —
    # a wedged/stale daemon can acknowledge without acting, and trusting it
    # here left fans pinned at max after the workload ended. Verify the fan
    # rows are back on the auto curve before accepting the socket path; on
    # verification failure fall through to the CLI candidates in this same
    # call instead of reporting a restore that never happened.
    if profile == "silent" and str(selected.get("kind")) == "thermalforge":
        reset = _daemon_socket_send("auto")
        if reset is not None:
            attempts.append(reset)
            if reset["ok"]:
                verified = wait_for_auto_fans()
                reset["verified"] = verified
                if verified:
                    return {
                        "ok": True,
                        "profile": profile,
                        "dry_run": False,
                        "detection": detection,
                        "command": reset["command"],
                        "attempts": attempts,
                    }

    for command in commands:
        result = _run_probe(command, timeout_s=15.0)
        attempts.append(result)
        if result["ok"]:
            return {
                "ok": True,
                "profile": profile,
                "dry_run": False,
                "detection": detection,
                "command": command,
                "attempts": attempts,
            }
    return {
        "ok": False,
        "profile": profile,
        "dry_run": False,
        "detection": detection,
        "attempts": attempts,
        "message": (
            "Thermal tool was detected, but MTPLX could not switch profiles "
            "through its CLI. Check the tool's CLI syntax or run mtplx max --status."
        ),
    }


@contextmanager
def thermal_profile(profile: str, *, enabled: bool) -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {"enabled": bool(enabled), "start": None, "restore": None}
    if enabled:
        state["start"] = set_thermal_profile(profile)
    try:
        yield state
    finally:
        if enabled and state.get("start", {}).get("detection", {}).get("available"):
            state["restore"] = set_thermal_profile("silent")


def run_command_with_profile(
    command: list[str],
    *,
    profile: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    with thermal_profile(profile, enabled=True) as thermal:
        proc = subprocess.run(command, env=env, cwd=cwd, check=False)
    return {"returncode": proc.returncode, "thermal": thermal, "command": command}


# ---------- auto-install -----------------------------------------------------
#
# Bootstrap path for users who pick a fan-backed mode in `mtplx start` and don't
# already have a fan controller.
#
# As of 2026-04 the upstream Homebrew tap (ProducerGuy/tap) ships a formula
# whose tarball is missing ``Scripts/generate-icon.swift``, so ``brew install
# ProducerGuy/tap/thermalforge`` aborts with a Swift compile error. Source
# install (``git clone … && ./setup.sh``) ships the missing file and is the
# reliable path. The default auto path now uses only the MTPLX-owned source
# build; Homebrew remains an explicit diagnostic/manual method.

THERMALFORGE_GIT_URL = "https://github.com/ProducerGuy/ThermalForge.git"
THERMALFORGE_BUILD_DIR = "~/.mtplx/build/ThermalForge"

INSTALL_PREREQ_HINTS = {
    "swift": (
        "Xcode command-line tools are required to build ThermalForge from "
        "source. Install them with:\n"
        "  xcode-select --install\n"
        "Then re-run `mtplx max --install`."
    ),
    "git": (
        "git is required to clone ThermalForge. Install it with:\n"
        "  xcode-select --install\n"
        "Then re-run `mtplx max --install`."
    ),
    "brew": (
        'Homebrew not found. Install it with:\n'
        '  /bin/bash -c "$(curl -fsSL '
        'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n'
        "Then re-run `mtplx max --install`."
    ),
}


def _run_streaming(
    command: list[str],
    *,
    timeout_s: float | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run ``command`` letting stdin/stdout/stderr pass through to the user.

    Required so brew/git/sudo can show download progress and prompt for a
    password naturally inside the user's terminal.
    """

    try:
        proc = subprocess.run(command, check=False, timeout=timeout_s, cwd=cwd)
    except Exception as exc:
        return {"command": command, "returncode": None, "ok": False, "error": str(exc)}
    return {
        "command": command,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
    }


def install_thermal_control(
    *,
    method: str = "auto",
    install_daemon: bool = True,
    streaming: bool = True,
) -> dict[str, Any]:
    """Auto-install ThermalForge.

    ``method`` selects the install path:
      ``"auto"``    — use the app-bundled helper when present, otherwise
                      build from source into ``~/.mtplx/bin`` (default).
      ``"source"``  — git clone + ./setup.sh (uses Xcode CLI tools).
      ``"homebrew"`` — brew install ProducerGuy/tap/thermalforge + sudo daemon.

    Returns a dict with ``ok`` (whether a working ``thermalforge`` CLI is on
    PATH afterwards), ``method`` (which path actually succeeded), ``steps``
    (per-step results), ``message`` (a single human-readable summary), and
    ``detection`` (post-install thermal-control detection).
    """

    if method == "homebrew":
        return _install_from_homebrew(install_daemon=install_daemon, streaming=streaming)
    if method == "source":
        return _install_from_source(install_daemon=install_daemon, streaming=streaming)
    if method != "auto":
        raise ValueError(f"unknown install method: {method}")

    # ``auto`` no longer falls through to Homebrew. The macOS app passes an
    # already-built helper through MTPLX_BUNDLED_THERMALFORGE; CLI installs
    # still build from source if that resource is absent. The upstream
    # ``ProducerGuy/tap/thermalforge`` formula has been observed to fail
    # mid-build (missing ``Scripts/generate-icon.swift`` in the formula's
    # source archive, May 2026) AND, when it does succeed, installs into
    # ``/usr/local/bin/`` which we can't keep stable. Source build to
    # ``~/.mtplx/bin`` is the only path we trust.
    return _install_from_source(install_daemon=install_daemon, streaming=streaming)


# Backwards-compatible alias for the old ``mtplx max --install`` wiring and
# any external imports.
install_thermal_control_homebrew = install_thermal_control


def _finish_private_thermalforge_install(
    *,
    method: str,
    steps: list[dict[str, Any]],
    install_label: str,
    failure_label: str,
    streaming: bool,
) -> dict[str, Any]:
    detect_thermal_control.cache_clear()
    detection = detect_thermal_control()

    # Sudoers rule scoped exactly to OUR binary path. Any external user of
    # ``thermalforge`` (the upstream installer, a manual /usr/local/bin copy)
    # gets no extra privilege from this.
    sudoers_result = install_passwordless_sudoers_rule(
        binary_path=MTPLX_THERMALFORGE_PATH, streaming=streaming
    )
    sudoers_result["step"] = "passwordless_sudoers"
    steps.append(sudoers_result)
    if not sudoers_result.get("ok"):
        return {
            "ok": False,
            "method": method,
            "message": (
                f"{install_label} at {MTPLX_THERMALFORGE_PATH}, but "
                "passwordless sudo could not be configured. Re-run "
                "`mtplx max --grant-sudo` after fixing the cause: "
                f"{sudoers_result.get('message')}"
            ),
            "steps": steps,
            "detection": detection,
        }

    # Final live verification: ramp fans, confirm the daemon accepted the
    # command, then ALWAYS restore — even if verification reports failure —
    # so we never leave the user's machine with fans blasting because of a
    # bad install path.
    verify = set_thermal_profile_verified("performance", settle_seconds=1.0)
    try:
        set_thermal_profile("silent")
    except Exception:
        pass  # restore is best-effort; never let it mask the real result
    steps.append(
        {"step": "live_fan_test", **{k: v for k, v in verify.items() if k != "raw"}}
    )

    if not verify.get("ok"):
        return {
            "ok": False,
            "method": method,
            "message": (
                f"{failure_label}, but the live fan-ramp test did not "
                f"register at the daemon. Reason: {verify.get('message')}. "
                f"Action: {verify.get('actionable', '')}"
            ),
            "steps": steps,
            "detection": detection,
        }

    return {
        "ok": True,
        "method": method,
        "binary": MTPLX_THERMALFORGE_PATH,
        "message": (
            f"ThermalForge installed at {MTPLX_THERMALFORGE_PATH} and "
            "verified live (fans ramped and restored). MAX mode is ready."
        ),
        "steps": steps,
        "detection": detection,
    }


def _copy_thermalforge_to_private_bin(
    source_path: str,
    *,
    steps: list[dict[str, Any]],
    step: str,
) -> dict[str, Any] | None:
    os.makedirs(MTPLX_THERMALFORGE_DIR, exist_ok=True)
    try:
        shutil.copy2(source_path, MTPLX_THERMALFORGE_PATH)
        os.chmod(MTPLX_THERMALFORGE_PATH, 0o755)
    except OSError as exc:
        return {
            "ok": False,
            "step": step,
            "message": f"Could not copy thermalforge to {MTPLX_THERMALFORGE_PATH}: {exc}",
            "steps": steps,
            "detection": detect_thermal_control(),
        }
    steps.append(
        {
            "step": step,
            "ok": True,
            "src": source_path,
            "dst": MTPLX_THERMALFORGE_PATH,
        }
    )
    return None


def _install_from_bundled(
    bundled_binary: str,
    *,
    streaming: bool,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    failure = _copy_thermalforge_to_private_bin(
        bundled_binary,
        steps=steps,
        step="copy_bundled_to_mtplx_bin",
    )
    if failure is not None:
        failure["method"] = "bundled"
        return failure
    return _finish_private_thermalforge_install(
        method="bundled",
        steps=steps,
        install_label="ThermalForge copied from the MTPLX app bundle",
        failure_label="ThermalForge is copied from the MTPLX app bundle and sudoers is configured",
        streaming=streaming,
    )


def _install_from_source(
    *,
    install_daemon: bool,  # kept for API parity; ignored (we don't install daemon)
    streaming: bool,
) -> dict[str, Any]:
    """Build ThermalForge from source and install ONLY into MTPLX's private
    ``~/.mtplx/bin`` directory.

    Why we don't run upstream's ``setup.sh`` or ``thermalforge install``:
    the upstream installer copies a binary to ``/usr/local/bin/`` and writes
    a privileged launchd plist whose binary path it then locks in. If
    *anything* (a stray ``thermalforge install`` invocation from the wrong
    cwd, an OS update, a ``brew uninstall``, etc.) removes
    ``/usr/local/bin/thermalforge`` after the fact, the daemon dies and the
    CLI is gone — leaving the user with no fan control and no obvious
    recovery path. Witnessed in the wild May 2026.

    By installing under ``~/.mtplx/bin`` and using passwordless sudo
    targeting that exact path, MTPLX's fan control is owned end-to-end by
    MTPLX. Nothing outside ``mtplx max --install`` can break it.
    """

    runner = _run_streaming if streaming else _run_probe
    steps: list[dict[str, Any]] = []

    bundled = _bundled_thermalforge_path()
    if bundled is not None:
        return _install_from_bundled(bundled, streaming=streaming)

    git = shutil.which("git")
    if git is None:
        return {
            "ok": False,
            "step": "prereq_git",
            "needs_prereq": "git",
            "message": INSTALL_PREREQ_HINTS["git"],
            "steps": steps,
            "detection": detect_thermal_control(),
        }
    swift = shutil.which("swift")
    if swift is None:
        return {
            "ok": False,
            "step": "prereq_swift",
            "needs_prereq": "swift",
            "message": INSTALL_PREREQ_HINTS["swift"],
            "steps": steps,
            "detection": detect_thermal_control(),
        }

    build_dir = os.path.expanduser(THERMALFORGE_BUILD_DIR)
    parent = os.path.dirname(build_dir)
    os.makedirs(parent, exist_ok=True)

    if os.path.isdir(os.path.join(build_dir, ".git")):
        pull_result = runner([git, "-C", build_dir, "pull", "--ff-only"], timeout_s=120.0)
        pull_result["step"] = "git_pull"
        steps.append(pull_result)
        if not pull_result.get("ok"):
            # Wipe and re-clone if pull fails — the working tree may be dirty.
            shutil.rmtree(build_dir, ignore_errors=True)

    if not os.path.isdir(os.path.join(build_dir, ".git")):
        clone_result = runner(
            [git, "clone", "--depth", "1", THERMALFORGE_GIT_URL, build_dir],
            timeout_s=300.0,
        )
        clone_result["step"] = "git_clone"
        steps.append(clone_result)
        if not clone_result.get("ok"):
            return {
                "ok": False,
                "step": "git_clone",
                "message": (
                    f"git clone of {THERMALFORGE_GIT_URL} failed. Check your "
                    "network connection and re-run `mtplx max --install`."
                ),
                "steps": steps,
                "detection": detect_thermal_control(),
            }

    # `swift build -c release` — incremental and fast on subsequent runs.
    build_result = runner(
        [swift, "build", "-c", "release"],
        cwd=build_dir,
        timeout_s=900.0,
    )
    build_result["step"] = "swift_build"
    steps.append(build_result)
    if not build_result.get("ok"):
        return {
            "ok": False,
            "step": "swift_build",
            "message": (
                "swift build failed. The most common cause is incomplete "
                "Xcode command-line tools. Run `xcode-select --install` and "
                "re-run `mtplx max --install`."
            ),
            "steps": steps,
            "detection": detect_thermal_control(),
        }

    # Find the freshly-built binary and copy it to MTPLX's private bin dir.
    candidates = [
        os.path.join(build_dir, ".build", "release", "thermalforge"),
        os.path.join(build_dir, ".build", "arm64-apple-macosx", "release", "thermalforge"),
        os.path.join(build_dir, ".build", "x86_64-apple-macosx", "release", "thermalforge"),
    ]
    built_binary = next((p for p in candidates if os.path.isfile(p)), None)
    if built_binary is None:
        return {
            "ok": False,
            "step": "post_build_locate",
            "message": (
                "swift build reported success but no thermalforge binary was "
                "produced. Re-run `mtplx max --install` after deleting "
                f"{build_dir}."
            ),
            "steps": steps,
            "detection": detect_thermal_control(),
        }

    failure = _copy_thermalforge_to_private_bin(
        built_binary,
        steps=steps,
        step="copy_to_mtplx_bin",
    )
    if failure is not None:
        failure["method"] = "source"
        return failure

    return _finish_private_thermalforge_install(
        method="source",
        steps=steps,
        install_label="ThermalForge built and copied",
        failure_label="ThermalForge is built and sudoers is configured",
        streaming=streaming,
    )


def _install_from_homebrew(
    *,
    install_daemon: bool,
    streaming: bool,
) -> dict[str, Any]:
    runner = _run_streaming if streaming else _run_probe
    steps: list[dict[str, Any]] = []

    brew = shutil.which("brew")
    if brew is None:
        return {
            "ok": False,
            "step": "prereq_brew",
            "needs_prereq": "brew",
            "message": INSTALL_PREREQ_HINTS["brew"],
            "steps": steps,
            "detection": detect_thermal_control(),
        }

    install_result = runner([brew, "install", THERMALFORGE_TAP], timeout_s=600.0)
    install_result["step"] = "brew_install"
    steps.append(install_result)
    if not install_result.get("ok"):
        return {
            "ok": False,
            "step": "brew_install",
            "message": (
                f"`brew install {THERMALFORGE_TAP}` failed. The upstream "
                "formula has been flaky; try the source path: "
                "`mtplx max --install` (auto) usually falls back to source."
            ),
            "steps": steps,
            "detection": detect_thermal_control(),
        }

    detect_thermal_control.cache_clear()
    detection = detect_thermal_control()
    if not detection.get("available"):
        return {
            "ok": False,
            "step": "post_install_detect",
            "message": (
                "thermalforge installed via brew but the CLI is not on PATH. "
                "Open a new terminal (or `hash -r`) and re-run."
            ),
            "steps": steps,
            "detection": detection,
        }

    if not install_daemon:
        return {
            "ok": True,
            "daemon_ok": None,
            "message": (
                "ThermalForge CLI installed. To enable fan control without "
                "a password every time, run: sudo thermalforge install"
            ),
            "steps": steps,
            "detection": detection,
        }

    selected = detection["selected"]
    daemon_path = str(selected["path"]) if selected else "thermalforge"
    daemon_result = runner(["sudo", daemon_path, "install"], timeout_s=120.0)
    daemon_result["step"] = "sudo_thermalforge_install"
    steps.append(daemon_result)
    if not daemon_result.get("ok"):
        return {
            "ok": True,
            "daemon_ok": False,
            "message": (
                "ThermalForge CLI installed, but the daemon setup "
                "(`sudo thermalforge install`) did not complete. MAX mode "
                "still works but will ask for your password each run."
            ),
            "steps": steps,
            "detection": detection,
        }

    return {
        "ok": True,
        "daemon_ok": True,
        "method": "homebrew",
        "message": "ThermalForge installed and ready. MAX mode will ramp the fans.",
        "steps": steps,
        "detection": detection,
    }
