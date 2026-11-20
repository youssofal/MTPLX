"""Local hardware inspection helpers for MTPLX QA."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from importlib import metadata
from typing import Any


# Absolute tool paths: the app launches its subprocesses with a PATH that is
# not guaranteed to carry /usr/sbin (see mistakes ledger, 2026-07 sanitized
# PATH incident), and both tools live there on every macOS.
_SYSCTL = "/usr/sbin/sysctl"
_SYSTEM_PROFILER = "/usr/sbin/system_profiler"


def _run_text(*cmd: str, timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _run_json(*cmd: str, timeout: float = 5.0) -> dict[str, Any]:
    text = _run_text(*cmd, timeout=timeout)
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in str(value or "").split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            break
    return tuple(parts)


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _sysctl_int(name: str) -> int | None:
    raw = _run_text(_SYSCTL, "-n", name) if platform.system() == "Darwin" else ""
    try:
        return int(raw)
    except ValueError:
        return None


def _hardware_json() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {}
    return _run_json(_SYSTEM_PROFILER, "SPHardwareDataType", "-json", timeout=8.0)


def _display_json() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {}
    return _run_json(_SYSTEM_PROFILER, "SPDisplaysDataType", "-json", timeout=8.0)


def _first_item(payload: dict[str, Any], key: str) -> dict[str, Any]:
    values = payload.get(key)
    if isinstance(values, list) and values and isinstance(values[0], dict):
        return values[0]
    return {}


#: The newest M-series generation this build's packs and kernels were measured on.
NEWEST_KNOWN_GENERATION = 5

_APPLE_SILICON_CHIP_RE = re.compile(r"\bM(\d+)\b", re.IGNORECASE)


def chip_generation_number(chip: str | None) -> int | None:
    """The M-series number in a chip's name ("Apple M6 Max" is 6), or None."""

    match = _APPLE_SILICON_CHIP_RE.search(str(chip or ""))
    return int(match.group(1)) if match else None


def classify_apple_silicon_generation(
    chip: str | None,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Normalize local Mac hardware into the generation used by model policy.

    A chip newer than this build knows is classified as the newest one it
    does know.  Everything chosen by generation (the default pack, the
    tensor-unit report) then takes the newest path, which is the best guess
    for a newer Mac; before, such a chip was "unknown" and took the oldest.
    ``newer_chip_note`` is the sentence that says so.
    """

    system_value = system or platform.system()
    machine_value = machine or platform.machine()
    chip_value = str(chip or "").strip()
    lowered = chip_value.lower()
    if system_value == "Darwin" and machine_value in {"x86_64", "i386"}:
        return "intel"
    if "intel" in lowered:
        return "intel"
    number = chip_generation_number(chip_value)
    if number is not None and number >= 1:
        return f"m{min(number, NEWEST_KNOWN_GENERATION)}"
    return "unknown"


def newer_chip_note(chip: str | None) -> str | None:
    """What to tell the owner of a chip newer than this build knows; None for any other."""

    number = chip_generation_number(chip)
    if number is None or number <= NEWEST_KNOWN_GENERATION:
        return None
    return (
        f"{str(chip).strip()} is newer than this MTPLX build knows. It runs the paths "
        f"measured on the M{NEWEST_KNOWN_GENERATION} until a build measured on it ships."
    )


def total_memory_gib() -> float:
    """Total unified memory in GiB; 0.0 when it cannot be determined."""

    if platform.system() != "Darwin":
        return 0.0
    raw = _run_text(_SYSCTL, "-n", "hw.memsize")
    try:
        return int(raw) / 1_073_741_824.0
    except ValueError:
        return 0.0


def detect_apple_silicon() -> dict[str, Any]:
    """Return the cheap local hardware facts needed for default-model routing."""

    system = platform.system()
    machine = platform.machine()
    chip = ""
    if system == "Darwin":
        chip = _run_text(_SYSCTL, "-n", "machdep.cpu.brand_string")
        generation = classify_apple_silicon_generation(chip, system=system, machine=machine)
        if generation == "unknown":
            hardware = _first_item(_hardware_json(), "SPHardwareDataType")
            profiler_chip = str(hardware.get("chip_type") or "").strip()
            if profiler_chip:
                chip = profiler_chip
                generation = classify_apple_silicon_generation(chip, system=system, machine=machine)
    else:
        generation = classify_apple_silicon_generation(chip, system=system, machine=machine)
    return {
        "system": system,
        "machine": machine,
        "chip": chip,
        "apple_silicon_generation": generation,
        "is_apple_silicon": generation not in {"intel", "unknown"},
        "memory_gib": total_memory_gib(),
        "newer_chip_note": newer_chip_note(chip),
    }


def inspect_hardware() -> dict[str, Any]:
    """Return the hardware facts used by prefill QA and release claims."""

    system = platform.system()
    machine = platform.machine()
    macos_version = _run_text("sw_vers", "-productVersion") if system == "Darwin" else ""
    hardware = _first_item(_hardware_json(), "SPHardwareDataType")
    display = _first_item(_display_json(), "SPDisplaysDataType")
    chip = (
        str(hardware.get("chip_type") or "")
        or _run_text(_SYSCTL, "-n", "machdep.cpu.brand_string")
    )
    generation = classify_apple_silicon_generation(chip, system=system, machine=machine)
    ram_bytes = 0
    raw_mem = _run_text(_SYSCTL, "-n", "hw.memsize") if system == "Darwin" else ""
    try:
        ram_bytes = int(raw_mem)
    except ValueError:
        ram_bytes = 0
    physical_cpu = _sysctl_int("hw.physicalcpu")
    logical_cpu = _sysctl_int("hw.logicalcpu")
    perflevel0_cpu = _sysctl_int("hw.perflevel0.physicalcpu")
    perflevel1_cpu = _sysctl_int("hw.perflevel1.physicalcpu")
    try:
        gpu_cores = int(display.get("sppci_cores") or display.get("spdisplays_cores") or 0)
    except (TypeError, ValueError):
        gpu_cores = None
    mlx_version = _dist_version("mlx")
    mlx_lm_version = _dist_version("mlx-lm")
    # By generation, so a chip newer than this build knows reports as the
    # newest class it does know (the GPU's own tensor-unit check is
    # mtplx.nax_detect, which reads the Metal architecture, never the name).
    is_m5 = generation == "m5"
    is_m3_ultra = "m3 ultra" in chip.lower()
    macos_eligible = _version_tuple(macos_version) >= (26, 2)
    mlx_eligible = _version_tuple(mlx_version or "") >= (0, 31)
    tensorops_eligible = bool(
        system == "Darwin"
        and machine == "arm64"
        and is_m5
        and macos_eligible
        and mlx_eligible
    )
    warnings: list[str] = []
    newer = newer_chip_note(chip)
    if newer:
        warnings.append(newer)
    if is_m5 and not macos_eligible:
        warnings.append("M5 TensorOps eligibility requires macOS 26.2 or newer.")
    if is_m5 and not mlx_eligible:
        warnings.append("M5 TensorOps eligibility requires an MLX stack new enough for the M5 path.")
    if is_m5:
        warnings.append("Eligibility is not proof; public Neural Accelerator claims require xctrace evidence.")
    if is_m3_ultra:
        warnings.append("M3 Ultra should be treated as a high-bandwidth GPU path, not an M5 TensorOps path.")

    return {
        "system": system,
        "machine": machine,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_is_arm64": bool(machine == "arm64"),
        "macos_version": macos_version,
        "darwin_kernel": platform.release(),
        "chip": chip,
        "apple_silicon_generation": generation,
        "model_identifier": hardware.get("machine_model") or hardware.get("machine_name") or "",
        "cpu_cores": logical_cpu or physical_cpu,
        "physical_cpu_cores": physical_cpu,
        "logical_cpu_cores": logical_cpu,
        "cpu_perflevel0_cores": perflevel0_cpu,
        "cpu_perflevel1_cores": perflevel1_cpu,
        "gpu": display.get("sppci_model") or display.get("_name") or "",
        "gpu_cores": gpu_cores,
        "unified_memory_bytes": ram_bytes,
        "unified_memory_gb": round(ram_bytes / (1024**3), 2) if ram_bytes else None,
        "memory_bandwidth_class_gb_s": 614 if "m5 max" in chip.lower() else None,
        "mlx_version": mlx_version,
        "mlx_lm_version": mlx_lm_version,
        "metal_device": display.get("sppci_model") or display.get("_name") or chip,
        "m5_neural_accelerator_eligible": tensorops_eligible,
        "hardware_acceleration_eligible": tensorops_eligible,
        "hardware_acceleration_confirmed": False,
        "hardware_acceleration_confirmation": "not_profiled",
        "warnings": warnings,
    }
