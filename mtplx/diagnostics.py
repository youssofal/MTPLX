"""Production diagnostics for MTPLX.

This module intentionally avoids MLX imports.  `mtplx doctor` must be useful on
fresh machines before the runtime stack is installed.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.hf_loader import cached_model_path, validate_mtplx_model_files
from mtplx.profiles import DEFAULT_HF_MODEL_ID, DEFAULT_PROFILE_NAME


GIB = 1024**3
DEFAULT_SPEED_MODEL_SIZE_BYTES = 16_430_000_000
MIN_RECOMMENDED_MEMORY_BYTES = 48 * GIB
SUPPORT_MACOS_MAJOR = 14
SUPPORT_PYTHON = (3, 11)
SUPPORT_MATRIX = {
    "supported": {
        "platform": "Apple Silicon arm64 Mac",
        "macos": ">= 14.0",
        "python": "native arm64 Python >= 3.11",
        "docker": "Docker Desktop current plus previous two macOS major releases",
        "default_model": DEFAULT_HF_MODEL_ID,
        "default_profile": DEFAULT_PROFILE_NAME,
    },
    "preview_test_targets": [
        "M3 Max",
        "M4 Max",
        "M3 Ultra / Mac Studio",
        "M5 Max developer machine",
    ],
}
DOCS = {
    "mlx": "https://ml-explore.github.io/mlx/build/html/install.html",
    "docker": "https://docs.docker.com/desktop/setup/install/mac-install/",
    "openwebui": "https://docs.openwebui.com/getting-started/quick-start/",
    "openwebui_env": "https://docs.openwebui.com/reference/env-configuration/",
    "docker_openwebui": "https://docs.docker.com/ai/model-runner/openwebui-integration/",
}


@dataclass(frozen=True)
class DiagnosticCheck:
    id: str
    status: str
    severity: str
    observed: Any
    expected: Any
    fix: str | None = None
    docs_url: str | None = None
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "severity": self.severity,
            "observed": self.observed,
            "expected": self.expected,
            "fix": self.fix,
            "docs_url": self.docs_url,
            "command": self.command,
        }


def _run(args: list[str], *, timeout: float = 2.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - host dependent
        return {"ok": False, "error": repr(exc), "command": args}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "command": args,
    }


def _parse_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw in value.split("."):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def _sysctl(name: str) -> str | None:
    result = _run(["/usr/sbin/sysctl", "-n", name])
    if result.get("ok"):
        return str(result.get("stdout") or "").strip() or None
    return None


def _pmset_report() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"available": False, "reason": "not macOS"}
    power = _run(["pmset", "-g"], timeout=2.0)
    therm = _run(["pmset", "-g", "therm"], timeout=2.0)
    low_power: str | None = None
    power_mode: str | None = None
    if power.get("ok"):
        for line in str(power.get("stdout") or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "lowpowermode":
                low_power = parts[1]
            if len(parts) >= 2 and parts[0] == "powermode":
                power_mode = parts[1]
    therm_text = str(therm.get("stdout") or therm.get("stderr") or "").strip()
    thermal_ok = bool(therm.get("ok")) and (
        "No thermal warning level has been recorded" in therm_text
        and "No performance warning level has been recorded" in therm_text
    )
    return {
        "available": bool(power.get("ok") or therm.get("ok")),
        "lowpowermode": low_power,
        "powermode": power_mode,
        "thermal": therm_text or None,
        "thermal_ok": thermal_ok,
    }


def _http_probe(url: str, *, timeout: float = 1.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {
                "ok": 200 <= int(response.status) < 300,
                "status": int(response.status),
                "body_prefix": body[:240],
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def host_report(*, model_cache: str | Path | None = None) -> dict[str, Any]:
    cache_root = Path(model_cache or os.environ.get("MTPLX_MODEL_DIR") or "~/.mtplx/models").expanduser()
    macos = _run(["sw_vers", "-productVersion"]) if platform.system() == "Darwin" else {}
    mem_raw = _sysctl("hw.memsize") if platform.system() == "Darwin" else None
    memory_bytes = int(mem_raw) if mem_raw and mem_raw.isdigit() else None
    machine_model = _sysctl("hw.model") if platform.system() == "Darwin" else None
    chip = _sysctl("machdep.cpu.brand_string") if platform.system() == "Darwin" else None
    disk_parent = cache_root if cache_root.exists() else cache_root.parent
    try:
        disk = shutil.disk_usage(disk_parent)
    except OSError:
        disk = None
    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "macos_version": macos.get("stdout") if macos.get("ok") else None,
        "mac_model": machine_model,
        "chip": chip,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_executable": sys.executable,
        "memory_bytes": memory_bytes,
        "memory_gib": round(memory_bytes / GIB, 2) if memory_bytes else None,
        "cache_dir": str(cache_root),
        "disk_free_bytes": disk.free if disk else None,
        "disk_free_gib": round(disk.free / GIB, 2) if disk else None,
    }


def estimate_runtime_memory_bytes(
    *,
    model_size_bytes: int = DEFAULT_SPEED_MODEL_SIZE_BYTES,
    profile: str = DEFAULT_PROFILE_NAME,
) -> int:
    overhead = 24 * GIB if profile == "performance-cold" else 20 * GIB
    return int(model_size_bytes + overhead)


def required_download_free_bytes(model_size_bytes: int = DEFAULT_SPEED_MODEL_SIZE_BYTES) -> int:
    return max(int(model_size_bytes * 2.5), int(model_size_bytes + 20 * GIB))


def _startup_default_model_observed(
    *,
    model_cache: str | Path | None,
    include_startup_default_model: bool,
) -> dict[str, Any] | None:
    if not include_startup_default_model:
        return None
    try:
        from mtplx.default_models import optimized_speed_model_ref
    except Exception:
        return None
    try:
        model_ref = str(optimized_speed_model_ref())
    except Exception:
        return None
    if not model_ref or model_ref == DEFAULT_HF_MODEL_ID:
        return None
    path = Path(model_ref).expanduser()
    if not path.is_dir():
        return None
    validation = validate_mtplx_model_files(path)
    return {
        "path": str(path),
        "validation": validation,
        "ok": bool(validation.get("ok")),
    }


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def build_diagnostic_checks(
    *,
    model_cache: str | Path | None = None,
    include_startup_default_model: bool = True,
    deep: bool = False,
    server_port: int = 8000,
    openwebui_port: int = 3000,
    mlx_info: dict[str, Any] | None = None,
    thermal_control: dict[str, Any] | None = None,
    server_dependencies: dict[str, bool] | None = None,
) -> tuple[dict[str, Any], list[DiagnosticCheck]]:
    host = host_report(model_cache=model_cache)
    checks: list[DiagnosticCheck] = []
    cache_root = Path(model_cache or os.environ.get("MTPLX_MODEL_DIR") or "~/.mtplx/models").expanduser()
    macos_version = host.get("macos_version")
    macos_ok = platform.system() == "Darwin" and bool(macos_version) and _parse_version(str(macos_version)) >= (SUPPORT_MACOS_MAJOR, 0)
    checks.append(
        DiagnosticCheck(
            "os.macos_version",
            "pass" if macos_ok else "fail",
            "error",
            macos_version or platform.system(),
            "macOS >= 14.0 on Apple Silicon",
            "Upgrade to macOS 14+; MLX does not support older macOS.",
            DOCS["mlx"],
        )
    )
    native_arm = platform.machine() == "arm64" and platform.processor() != "i386"
    checks.append(
        DiagnosticCheck(
            "python.native_arm64",
            "pass" if native_arm else "fail",
            "error",
            {"machine": platform.machine(), "processor": platform.processor()},
            "native arm64 Python, not Rosetta",
            "Install/use a native arm64 Python. If needed, reinstall via Homebrew arm64 or uv.",
            DOCS["mlx"],
            "python3 -c \"import platform; print(platform.machine(), platform.processor())\"",
        )
    )
    python_ok = sys.version_info >= SUPPORT_PYTHON
    checks.append(
        DiagnosticCheck(
            "python.version",
            "pass" if python_ok else "fail",
            "error",
            host["python_version"],
            "Python >= 3.11",
            "Install Python 3.11 or newer.",
            DOCS["mlx"],
        )
    )
    mlx = mlx_info or {}
    mlx_ok = "mlx_error" not in mlx
    checks.append(
        DiagnosticCheck(
            "mlx.import",
            "pass" if mlx_ok else "fail",
            "error",
            mlx if mlx else "not probed",
            "mlx importable",
            "Install MLX into this same native Python environment.",
            DOCS["mlx"],
            "python3 -m pip install mlx",
        )
    )
    memory_bytes = host.get("memory_bytes")
    estimated = estimate_runtime_memory_bytes(profile=DEFAULT_PROFILE_NAME)
    memory_status = "warn"
    memory_fix = "Close other heavy apps or use a smaller model/profile."
    if isinstance(memory_bytes, int):
        if estimated > int(memory_bytes * 0.80):
            memory_status = "fail"
        elif memory_bytes < MIN_RECOMMENDED_MEMORY_BYTES:
            memory_status = "warn"
        else:
            memory_status = "pass"
            memory_fix = None
    checks.append(
        DiagnosticCheck(
            "resource.memory",
            memory_status,
            "error" if memory_status == "fail" else "warning",
            {
                "unified_memory_gib": host.get("memory_gib"),
                "estimated_peak_gib": round(estimated / GIB, 2),
            },
            "estimated peak <= 80% of unified memory; 48 GiB+ recommended",
            memory_fix,
        )
    )
    required_free = required_download_free_bytes()
    disk_free = host.get("disk_free_bytes")
    disk_status = "warn"
    if isinstance(disk_free, int):
        disk_status = "pass" if disk_free >= required_free else "fail"
    checks.append(
        DiagnosticCheck(
            "resource.model_cache_disk",
            disk_status,
            "error" if disk_status == "fail" else "warning",
            {
                "cache_dir": host.get("cache_dir"),
                "free_gib": host.get("disk_free_gib"),
                "required_gib": round(required_free / GIB, 2),
            },
            "free space for model + temp download + safety headroom",
            "Free disk space or set MTPLX_MODEL_DIR to a larger volume.",
        )
    )
    default_cached = cached_model_path(DEFAULT_HF_MODEL_ID, cache_dir=cache_root)
    default_validation = validate_mtplx_model_files(default_cached) if default_cached.exists() else None
    startup_default = _startup_default_model_observed(
        model_cache=model_cache,
        include_startup_default_model=include_startup_default_model,
    )
    cache_ok = bool(default_validation and default_validation.get("ok"))
    startup_default_ok = bool(startup_default and startup_default.get("ok"))
    if cache_ok:
        model_cache_status = "pass"
        model_cache_fix = "No action needed."
    elif startup_default_ok:
        model_cache_status = "pass"
        model_cache_fix = (
            "No first-run action needed; the normal startup path has a complete "
            "local model. Run the pull command only if you want to repair the "
            "partial Hugging Face cache copy."
        )
    else:
        model_cache_status = "warn"
        model_cache_fix = "Download the default model before first run."
    checks.append(
        DiagnosticCheck(
            "model.cache",
            model_cache_status,
            "warning",
            {
                "hf_cache_path": str(default_cached),
                "hf_cache_exists": default_cached.exists(),
                "hf_cache_validation": default_validation,
                "startup_default_model": startup_default,
            },
            "default model available in the HF cache or as the verified local startup model",
            model_cache_fix,
            "https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            f"mtplx pull {DEFAULT_HF_MODEL_ID}",
        )
    )
    stale = DEFAULT_HF_MODEL_ID.startswith("mtplx/") or DEFAULT_HF_MODEL_ID.startswith("models/")
    checks.append(
        DiagnosticCheck(
            "model.default_repo",
            "fail" if stale else "pass",
            "error",
            DEFAULT_HF_MODEL_ID,
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "Pull the default model, or pass --model to serve a different one.",
            "https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            f"mtplx pull {DEFAULT_HF_MODEL_ID}",
        )
    )
    if deep:
        try:
            from huggingface_hub import model_info

            info = model_info(DEFAULT_HF_MODEL_ID)
            hf_observed: Any = {
                "repo_id": DEFAULT_HF_MODEL_ID,
                "sha": getattr(info, "sha", None),
                "private": getattr(info, "private", None),
                "gated": getattr(info, "gated", None),
            }
            hf_ok = True
        except Exception as exc:  # pragma: no cover - network/token dependent
            hf_observed = {"repo_id": DEFAULT_HF_MODEL_ID, "error": repr(exc)}
            hf_ok = False
        checks.append(
            DiagnosticCheck(
                "hf.default_repo_access",
                "pass" if hf_ok else "fail",
                "error",
                hf_observed,
                "default Hugging Face repo is reachable from this machine",
                "Check network/HF auth or verify the model repo is public.",
                "https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
                f"mtplx pull {DEFAULT_HF_MODEL_ID}",
            )
        )
    docker_path = shutil.which("docker")
    checks.append(
        DiagnosticCheck(
            "docker.binary",
            "pass" if docker_path else "warn",
            "warning",
            docker_path,
            "Docker Desktop installed for Open WebUI Docker path",
            "Install Docker Desktop if you want the Open WebUI Docker integration.",
            DOCS["docker"],
        )
    )
    if deep and docker_path:
        docker_info = _run(["docker", "info", "--format", "{{json .ServerVersion}}"], timeout=3.0)
        docker_ok = bool(docker_info.get("ok"))
        checks.append(
            DiagnosticCheck(
                "docker.daemon",
                "pass" if docker_ok else "warn",
                "warning",
                docker_info,
                "Docker daemon running",
                "Start Docker Desktop.",
                DOCS["docker"],
                "docker info",
            )
        )
        if docker_ok:
            host_gateway = _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--add-host=host.docker.internal:host-gateway",
                    "busybox",
                    "sh",
                    "-lc",
                    "getent hosts host.docker.internal || nslookup host.docker.internal",
                ],
                timeout=8.0,
            )
            checks.append(
                DiagnosticCheck(
                    "docker.host_gateway",
                    "pass" if host_gateway.get("ok") else "warn",
                    "warning",
                    host_gateway,
                    "container can resolve host.docker.internal",
                    "Use --add-host=host.docker.internal:host-gateway in the Open WebUI Docker command.",
                    DOCS["docker_openwebui"],
                    "mtplx openwebui docker-command",
                )
            )
            openwebui_ps = _run(
                ["docker", "ps", "--filter", "name=open-webui", "--format", "{{.Names}} {{.Ports}}"],
                timeout=3.0,
            )
            checks.append(
                DiagnosticCheck(
                    "docker.openwebui_container",
                    "pass" if openwebui_ps.get("ok") and openwebui_ps.get("stdout") else "warn",
                    "warning",
                    openwebui_ps,
                    "Open WebUI container running if using Docker integration",
                    "Start Open WebUI with the MTPLX Docker command.",
                    DOCS["openwebui"],
                    "mtplx openwebui docker-command",
                )
            )
    checks.append(
        DiagnosticCheck(
            "port.mtplx_server",
            "warn" if _port_open("127.0.0.1", server_port) else "pass",
            "warning",
            {"host": "127.0.0.1", "port": server_port, "open": _port_open("127.0.0.1", server_port)},
            "port free before starting mtplx serve, or already a healthy MTPLX server",
            "A healthy MTPLX server already on this port is fine to keep using; "
            f"if something else holds it, stop that process or use --port {server_port + 1}.",
        )
    )
    if deep:
        models_probe = _http_probe(f"http://127.0.0.1:{server_port}/v1/models", timeout=1.0)
        checks.append(
            DiagnosticCheck(
                "server.openai_models",
                "pass" if models_probe.get("ok") else "warn",
                "warning",
                models_probe,
                "running MTPLX server exposes /v1/models",
                "Start the server or choose the correct port.",
                "https://github.com/youssofal/MTPLX/blob/main/docs/server.md",
                f"curl http://127.0.0.1:{server_port}/v1/models",
            )
        )
    checks.append(
        DiagnosticCheck(
            "port.openwebui",
            "warn" if _port_open("127.0.0.1", openwebui_port) else "pass",
            "warning",
            {"host": "127.0.0.1", "port": openwebui_port, "open": _port_open("127.0.0.1", openwebui_port)},
            "port free before starting Open WebUI, or already an Open WebUI container",
            f"Use a different Open WebUI host port or stop the process on {openwebui_port}.",
        )
    )
    if server_dependencies is not None:
        for name, ok in sorted(server_dependencies.items()):
            checks.append(
                DiagnosticCheck(
                    f"python.dep.{name}",
                    "pass" if ok else "fail",
                    "error",
                    ok,
                    f"{name} installed",
                    "Reinstall mtplx (fastapi and uvicorn ship as base dependencies).",
                    command="python3 -m pip install --force-reinstall mtplx",
                )
            )
    thermal = thermal_control or {}
    checks.append(
        DiagnosticCheck(
            "thermal.control",
            "pass" if thermal.get("available") else "warn",
            "warning",
            thermal.get("selected") or "none",
            "ThermalForge or TG Pro available for explicit --max only",
            "Install ThermalForge only if you want opt-in fan boost.",
        )
    )
    pmset = _pmset_report()
    low_power_enabled = pmset.get("lowpowermode") == "1"
    checks.append(
        DiagnosticCheck(
            "power.low_power_mode",
            "warn" if low_power_enabled else "pass",
            "warning",
            pmset,
            "Low Power Mode off for best sustained decode",
            "Turn off Low Power Mode before benchmarking or serving long responses.",
        )
    )
    thermal_pressure_ok = pmset.get("thermal_ok") is True
    checks.append(
        DiagnosticCheck(
            "power.thermal_pressure",
            "pass" if thermal_pressure_ok else "warn",
            "warning",
            pmset.get("thermal"),
            "no recorded thermal or performance warning",
            "Let the Mac cool down or improve airflow before sustained benchmarks.",
        )
    )
    return host, checks


def summarize_checks(checks: list[DiagnosticCheck]) -> str:
    if any(check.status == "fail" and check.severity == "error" for check in checks):
        return "fail"
    if any(check.status in {"fail", "warn"} for check in checks):
        return "warn"
    return "pass"


def build_diagnostics_payload(
    *,
    model_cache: str | Path | None = None,
    include_startup_default_model: bool = True,
    deep: bool = False,
    server_port: int = 8000,
    mlx_info: dict[str, Any] | None = None,
    thermal_control: dict[str, Any] | None = None,
    server_dependencies: dict[str, bool] | None = None,
) -> dict[str, Any]:
    host, checks = build_diagnostic_checks(
        model_cache=model_cache,
        include_startup_default_model=include_startup_default_model,
        deep=deep,
        server_port=server_port,
        mlx_info=mlx_info,
        thermal_control=thermal_control,
        server_dependencies=server_dependencies,
    )
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "overall": summarize_checks(checks),
        "support_matrix": SUPPORT_MATRIX,
        "host": host,
        "resources": {
            "default_model_size_bytes": DEFAULT_SPEED_MODEL_SIZE_BYTES,
            "estimated_runtime_memory_bytes": estimate_runtime_memory_bytes(),
            "required_download_free_bytes": required_download_free_bytes(),
        },
        "checks": [check.to_dict() for check in checks],
    }


def write_doctor_bundle(
    *,
    report: dict[str, Any],
    output_dir: str | Path | None = None,
    include_paths: bool = False,
) -> dict[str, Any]:
    base = Path(output_dir or "~/.mtplx/reports").expanduser()
    bundle_id = f"doctor-{time.strftime('%Y%m%d-%H%M%S')}"
    root = base / bundle_id
    root.mkdir(parents=True, exist_ok=True)
    redacted = json.loads(json.dumps(report))
    if not include_paths:
        for section in ("environment", "host"):
            item = redacted.get(section)
            if isinstance(item, dict):
                for key in ("project_root", "python_executable", "cache_dir"):
                    if key in item:
                        item[key] = "[redacted-path]"
    (root / "doctor.json").write_text(json.dumps(redacted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "bundle_id": bundle_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "redacted": not include_paths,
                "files": ["doctor.json", "summary.json"],
                "zip": f"{bundle_id}.zip",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    zip_path = base / f"{bundle_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(root.iterdir()):
            if file_path.is_file():
                archive.write(file_path, arcname=f"{bundle_id}/{file_path.name}")
    return {
        "bundle_id": bundle_id,
        "bundle_dir": str(root),
        "bundle_zip": str(zip_path),
        "redacted": not include_paths,
        "files": ["doctor.json", "summary.json"],
    }
