"""Opt-in thermal-control helpers for MTPLX.

The public contract is deliberately conservative: detect known fan-control
tools, run their documented/profile-style CLI commands when available, and
otherwise report clear installation instructions. This module never falls back
to spin loops or clock-anchor hacks.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
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

_PRIVILEGED_SUDOERS_SCRIPT = r"""
set -eu
sudoers_file="$1"
user_name="$2"
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
printf '%s ALL=(root) NOPASSWD: %s\n' "$user_name" "$binary_path" > "$tmp_file"
chown root:wheel "$tmp_file"
chmod 440 "$tmp_file"
/usr/sbin/visudo -c -f "$tmp_file" >/dev/null
mv "$tmp_file" "$sudoers_file"
trap - EXIT
"""


def _install_sudoers_rule_with_security(
    *,
    user: str,
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
        user,
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
    """

    runner = _run_streaming if streaming else _run_probe

    if binary_path is None:
        binary_path = shutil.which("thermalforge") or "/usr/local/bin/thermalforge"

    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "_unknown"
    rule = f"{user} ALL=(root) NOPASSWD: {binary_path}\n"

    # `sudo tee` keeps terminal installs simple. In the GUI app there is no
    # terminal for sudo to read from, so we fall back to macOS's admin
    # authorization prompt below.
    write_proc = subprocess.run(
        ["sudo", "tee", SUDOERS_FILE],
        input=rule,
        text=True,
        check=False,
        capture_output=True,
    )
    if write_proc.returncode != 0:
        security_result = _install_sudoers_rule_with_security(
            user=user,
            binary_path=binary_path,
        )
        if security_result.get("ok"):
            probe = _run_probe(["sudo", "-n", binary_path, "status"], timeout_s=5.0)
            return {
                "ok": probe.get("ok", False),
                "step": "verify_passwordless",
                "method": "security_execute_with_privileges",
                "binary_path": binary_path,
                "message": (
                    "Passwordless sudo for thermalforge is configured."
                    if probe.get("ok")
                    else (
                        "Admin authorization installed the sudoers rule, but "
                        "`sudo -n thermalforge status` still failed: "
                        + (probe.get("stderr") or "").strip()
                    )
                ),
                "steps": [security_result, {**probe, "step": "verify"}],
            }
        return {
            "ok": False,
            "step": "sudo_tee",
            "message": (
                f"Could not write {SUDOERS_FILE}. sudo said: "
                f"{(write_proc.stderr or '').strip()}. "
                f"{security_result.get('message', '')}"
            ),
            "steps": [security_result],
        }

    chmod_result = runner(["sudo", "chmod", "440", SUDOERS_FILE])
    chmod_result["step"] = "chmod"
    validate_result = runner(["sudo", "visudo", "-c", "-f", SUDOERS_FILE])
    validate_result["step"] = "visudo_check"
    if not validate_result.get("ok"):
        # If syntax is bad, remove the file so we don't leave the system
        # with an unparseable sudoers fragment.
        runner(["sudo", "rm", "-f", SUDOERS_FILE])
        return {
            "ok": False,
            "step": "visudo_check",
            "message": (
                "Sudoers rule was rejected by visudo and rolled back. "
                f"Validation error: {validate_result.get('stderr', '').strip()}"
            ),
            "steps": [chmod_result, validate_result],
        }

    # Final sanity: passwordless sudo should now work.
    probe = _run_probe(["sudo", "-n", binary_path, "status"], timeout_s=5.0)
    return {
        "ok": probe.get("ok", False),
        "step": "verify_passwordless",
        "binary_path": binary_path,
        "message": (
            "Passwordless sudo for thermalforge is configured."
            if probe.get("ok")
            else (
                "Sudoers rule installed but `sudo -n thermalforge status` "
                "still failed: " + (probe.get("stderr") or "").strip()
            )
        ),
        "steps": [chmod_result, validate_result, {**probe, "step": "verify"}],
    }


def remove_passwordless_sudoers_rule(*, streaming: bool = True) -> dict[str, Any]:
    """Counterpart to ``install_passwordless_sudoers_rule`` for clean uninstall."""

    runner = _run_streaming if streaming else _run_probe
    if not os.path.exists(SUDOERS_FILE):
        return {"ok": True, "message": f"{SUDOERS_FILE} already absent"}
    result = runner(["sudo", "rm", "-f", SUDOERS_FILE])
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
import json as _json  # noqa: E402  (avoid clashing with local json imports)
import signal  # noqa: E402
from pathlib import Path  # noqa: E402

MAX_MARKER_FILE = Path("~/.mtplx/max-active.json").expanduser()


def _write_max_marker(pid: int | None = None) -> None:
    if pid is None:
        pid = os.getpid()
    binary = None
    try:
        selected = detect_thermal_control().get("selected")
        if selected:
            binary = selected.get("path")
    except Exception:
        binary = None
    try:
        MAX_MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        MAX_MARKER_FILE.write_text(
            _json.dumps(
                {
                    "pid": int(pid),
                    "started_at": time.time(),
                    "binary": binary or _find_thermalforge(),
                }
            )
        )
    except Exception:
        pass  # marker is best-effort; don't crash --max because we can't write it


def _clear_max_marker() -> None:
    try:
        if MAX_MARKER_FILE.exists():
            MAX_MARKER_FILE.unlink()
    except Exception:
        pass


def _read_max_marker() -> dict[str, Any] | None:
    try:
        if not MAX_MARKER_FILE.exists():
            return None
        return _json.loads(MAX_MARKER_FILE.read_text())
    except Exception:
        return None


def check_and_recover_stale_max() -> dict[str, Any]:
    """Detect a marker left by a crashed --max session and restore fans.

    Returns ``{"recovered": bool, "stale_pid": int|None,
    "still_running": bool}``. Safe to call from any startup path; no-op when
    no marker exists.
    """

    marker = _read_max_marker()
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
        _clear_max_marker()
    return {
        "recovered": bool(restore.get("ok")),
        "stale_pid": stale_pid,
        "still_running": False,
        "restore": restore,
        "marker_cleared": bool(restore.get("ok")),
    }


def _spawn_thermal_sidecar() -> subprocess.Popen | None:
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

    _write_max_marker()
    _sidecar = _spawn_thermal_sidecar()
    cleaned_up = [False]

    def cleanup() -> dict[str, Any]:
        if cleaned_up[0]:
            return {"ok": True, "already_cleaned": True}
        cleaned_up[0] = True
        try:
            restore = restore_thermal_profile_verified()
        except Exception as exc:
            restore = {"ok": False, "error": str(exc), "message": "fan restore raised"}
        if restore.get("ok"):
            _clear_max_marker()
        # The sidecar will notice the parent is gone and re-issue auto
        # too — that's intentional belt-and-suspenders. If the in-process
        # cleanup succeeded, the sidecar's call is a harmless no-op.
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


class SmartFanController:
    """Request-scoped max-fan lease for Smart fan mode.

    Smart is intentionally lighter than ``MaxSession``: it commands max as
    soon as a request arrives, without making the request wait for the fan
    hardware. It reference-counts overlapping leases and restores Apple auto
    only after the last lease has been idle for ``restore_delay_s``.

    Contract (July 2026 overhaul, issue #127):
    - ``begin_request``/``end_request`` never block and never run a
      subprocess on the caller thread. All hardware commands execute on one
      dedicated worker thread; the controller lock only guards state.
      This is what allows the ramp to be issued at HTTP request arrival
      (before prompt encoding and queueing) at zero request-path cost.
    - After commanding max the worker verifies the daemon accepted the
      target RPM (``fan_summary``) and retries the command once if not.
      It then keeps polling until the physical ramp is visible. Ramp
      latencies and verification state are surfaced via ``status()``
      (and therefore ``/health``).
    - The restore is debounced: back-to-back requests (agent tool loops,
      generation -> idle-postcommit handoffs) reuse the ramp instead of
      flapping the fans down and up again.
    - ``detach()`` lets an external owner (Max mode) take over fan
      hardware without a scheduled smart restore racing it back to auto.
    """

    _ACTUAL_RAMP_TIMEOUT_S = 30.0
    _ACTUAL_RAMP_POLL_INTERVAL_S = 1.0
    _WAIT_FOR_RESTORE_TIMEOUT_S = 30.0
    _ACTIVITY_POLL_INTERVAL_S = 5.0
    _RESTORE_RETRY_BACKOFF_S = (5.0, 15.0, 30.0, 60.0)

    def __init__(
        self,
        *,
        log: Any = None,
        restore_delay_s: float = 2.0,
        activity_probe: Any = None,
    ) -> None:
        self.log = log
        self.restore_delay_s = max(0.0, float(restore_delay_s))
        # Optional callable returning True while the engine is executing or
        # queueing model work. Powers the stale-lease reconciler (#201): a
        # lease held while the engine has been continuously idle for
        # MTPLX_SMART_FAN_STALE_LEASE_S seconds is a leak upstream (hung
        # HTTP response, abandoned future), not a running request, and must
        # not pin the fans forever. 0 disables the reconciler.
        self._activity_probe = activity_probe
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
        # #201 restore-retry state: a restore that ran but could not verify
        # the fans back on the auto curve re-arms with backoff instead of
        # being forgotten while the hardware stays pinned.
        self._restore_unverified = False
        self._restore_retry_at: float | None = None
        self._restore_failures = 0
        self._engine_idle_since: float | None = None
        self._next_activity_probe_at: float | None = None
        self._stale_leases_reconciled = 0
        self._worker: threading.Thread | None = None
        self._shutdown = False

    def _emit(self, line: str) -> None:
        if self.log is not None:
            try:
                self.log(line)
            except Exception:
                pass

    # -- public lease API (non-blocking) ---------------------------------

    def begin_request(self, request_id: str) -> dict[str, Any]:
        request_key = str(request_id or "request")
        with self._cond:
            if request_key in self._active_requests:
                return self._status_locked()
            self._active_requests.add(request_key)
            self._generation += 1
            self._idle_since = None
            self._engine_idle_since = None
            if not self._commanded_max and self._ramp_requested_at is None:
                self._ramp_requested_at = time.monotonic()
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
                    # Skip the debounce for explicit synchronous restores
                    # (bench lanes, shutdown paths).
                    self._idle_since -= self.restore_delay_s
            self._ensure_worker_locked()
            self._cond.notify_all()
            if not became_idle:
                return self._status_locked()
            if not self._commanded_max and self._cleanup is None:
                return self._status_locked()
        if wait_for_restore:
            self._wait_until_restored()
        return self.status()

    def restore_now(self, *, wait: bool = True) -> dict[str, Any]:
        with self._cond:
            self._active_requests.clear()
            self._generation += 1
            self._idle_since = time.monotonic() - self.restore_delay_s
            self._ensure_worker_locked()
            self._cond.notify_all()
        if wait:
            self._wait_until_restored()
        return self.status()

    def detach(self) -> dict[str, Any]:
        """Drop all leases and pending fan work WITHOUT touching hardware.

        Used when an external owner takes over fan control (switching the
        server to Max mode): the old behavior scheduled a delayed smart
        restore that could fire *after* the Max pin landed and silently
        drop fans back to auto while the UI showed Max (issue #127).
        """
        with self._cond:
            self._active_requests.clear()
            self._generation += 1
            self._commanded_max = False
            self._target_verified = False
            self._actual_ramp_verified = False
            self._ramp_requested_at = None
            self._actual_poll_deadline = None
            self._next_actual_probe_at = None
            self._idle_since = None
            # The external owner (Max mode) owns the hardware now: pending
            # restore retries and idle bookkeeping belong to the old regime.
            self._restore_unverified = False
            self._restore_retry_at = None
            self._restore_failures = 0
            self._engine_idle_since = None
            # Drop our reference so the worker does not schedule a restore;
            # the atexit hook installed by install_max_lifecycle_hooks stays
            # registered and still restores fans on process exit.
            self._cleanup = None
            self._cond.notify_all()
            return self._status_locked()

    def wait_for_ramp(self, timeout_s: float = 10.0) -> bool:
        """Block until the worker has commanded max (True) or the ramp
        attempt for the current lease generation failed / timed out (False).

        The request path never calls this — it exists for bench lanes,
        tests, and QA probes that need a synchronization point with the
        async worker.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cond:
            while True:
                if self._commanded_max:
                    return True
                if self._ramp_failed_generation == self._generation:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return bool(self._commanded_max)
                self._cond.wait(timeout=remaining)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        return {
            "active": bool(self._active_requests),
            "active_count": len(self._active_requests),
            "active_requests": sorted(self._active_requests),
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
            while self._commanded_max and not self._active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(timeout=remaining)

    def _reconciler_enabled_locked(self) -> bool:
        return self._activity_probe is not None and self._stale_lease_idle_s > 0

    def _wait_capped_by_activity_probe_locked(self, timeout: float | None, now: float) -> None:
        """cond.wait, but never sleep past the next activity-probe slot while
        leases are active and the reconciler is enabled — a wedged lease must
        still get its periodic engine-idle check (#201)."""
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
                    desired_max = bool(self._active_requests)
                    if (
                        desired_max
                        and self._reconciler_enabled_locked()
                        and now >= (self._next_activity_probe_at or 0.0)
                    ):
                        action = "probe_activity"
                    elif desired_max and not self._commanded_max:
                        if self._ramp_failed_generation == self._generation:
                            # The last ramp (with its retry) failed for this
                            # lease generation; don't hammer the daemon.
                            # A new begin_request bumps the generation and
                            # re-arms the attempt.
                            self._wait_capped_by_activity_probe_locked(None, now)
                            continue
                        action = "ramp"
                    elif not desired_max and (self._commanded_max or self._cleanup is not None):
                        if self._idle_since is None:
                            self._idle_since = now
                        remaining = self._idle_since + self.restore_delay_s - now
                        if remaining <= 0:
                            action = "restore"
                        else:
                            self._cond.wait(timeout=remaining)
                    elif not desired_max and self._restore_unverified:
                        # A previous restore ran but the fans never verified
                        # back on the auto curve (#201). Keep retrying with
                        # backoff until they do or a new lease re-ramps.
                        retry_in = (self._restore_retry_at or now) - now
                        if retry_in <= 0:
                            action = "restore"
                        else:
                            self._cond.wait(timeout=max(0.05, retry_in))
                    elif (
                        desired_max
                        and self._commanded_max
                        and not self._actual_ramp_verified
                        and self._actual_poll_deadline is not None
                    ):
                        if now >= (self._next_actual_probe_at or 0.0):
                            action = "probe_actual"
                        else:
                            self._wait_capped_by_activity_probe_locked(
                                max(0.05, (self._next_actual_probe_at or now) - now), now
                            )
                    else:
                        self._wait_capped_by_activity_probe_locked(None, now)
            if action == "ramp":
                self._do_ramp()
            elif action == "restore":
                self._do_restore()
            elif action == "probe_actual":
                self._do_probe_actual()
            elif action == "probe_activity":
                self._do_probe_activity()

    def _do_ramp(self) -> None:
        with self._lock:
            generation = self._generation
            requested_at = self._ramp_requested_at or time.monotonic()
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
        result: dict[str, Any] | None = None
        error: str | None = None
        ok = False
        attempts = 0
        for _attempt in range(2):  # initial command + one bounded retry
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
            self._ramp_attempts += attempts
            self._last_result = result
            self._last_transition_at = time.time()
            self._commanded_max = ok
            self._target_verified = ok
            if ok:
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
        """Stale-lease reconciler (#201). A smart-fan lease is only legitimate
        while its request is somewhere in the engine (queued, executing, or
        streaming tokens). If leases are held while the activity probe reports
        the engine continuously idle for ``_stale_lease_idle_s`` seconds, the
        leases are leaked bookkeeping from a wedged request path: drop them,
        log loudly, and let the normal restore flow bring the fans back to
        auto. The probe fails open (busy) so a broken probe can never restore
        fans under a live workload."""
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
                # #201: a failed restore used to be marked "restored" and
                # forgotten, leaving the physical fans pinned at max until
                # the next request cycle happened to fix them. Keep the
                # clean-slate contract for the NEXT lease (commanded_max
                # drops so a new request re-commands max), but re-arm the
                # restore with backoff so the hardware is actually brought
                # back to auto even when no further request ever arrives.
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
                verify_deadline = time.monotonic() + 3.0
                verified = False
                while True:
                    try:
                        if _summary_indicates_auto(fan_summary()):
                            verified = True
                            break
                    except Exception:
                        pass
                    if time.monotonic() >= verify_deadline:
                        break
                    time.sleep(0.5)
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
