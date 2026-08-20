"""Backend implementation for the MTPLX Forge app surface."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from mtplx import artifacts
from mtplx.artifacts import inspect_model, text_config
from mtplx.benchmarks.validators.basic import summarize_benchmark_quality
from mtplx.hf_loader import (
    directory_size_bytes,
    hf_token_for_download,
    pull_model,
    repo_id_from_model_ref,
)
from mtplx.gemma4_pair import (
    GEMMA4_PAIR_FILE,
    is_gemma4_pair_repo_id,
    resolve_gemma4_pair_paths,
)
from mtplx.mtp_patch import MTPContract
from mtplx.version import __version__


SOURCE_BF16_NATIVE = "bf16_native"
SOURCE_MLX_AFFINE = "mlx_affine"
SOURCE_MLX_AFFINE_WITH_MTP = "mlx_affine_with_mtp"
SOURCE_AUTOAWQ = "autoawq"
SOURCE_COMPRESSED_TENSORS_AWQ = "compressed_tensors_awq"
SOURCE_COMPRESSED_TENSORS_NVFP4 = "compressed_tensors_nvfp4"
SOURCE_GEMMA4_ASSISTANT_PAIR = "gemma4_assistant_pair"
SOURCE_HF_VLLM = "hf_vllm"
SOURCE_UNKNOWN = "unknown"

SUPPORTED_SOURCE_FORMATS = {
    SOURCE_BF16_NATIVE,
    SOURCE_MLX_AFFINE,
    SOURCE_MLX_AFFINE_WITH_MTP,
    SOURCE_AUTOAWQ,
    SOURCE_COMPRESSED_TENSORS_AWQ,
    SOURCE_COMPRESSED_TENSORS_NVFP4,
    SOURCE_GEMMA4_ASSISTANT_PAIR,
    SOURCE_HF_VLLM,
    SOURCE_UNKNOWN,
}
UNSUPPORTED_MTP_SOURCE_FORMATS = {
}

GGUF_UNSUPPORTED_MESSAGE = (
    "GGUF sources are detected but Forge V1 builds MLX/safetensors artifacts only."
)
REQUANTIZE_REFUSAL = (
    "MTP policy 'requantize' degrades acceptance; "
    "pass --allow-degraded-mtp to confirm"
)
ACCEPTANCE_COLLAPSE_THRESHOLD = 0.05
FORGE_VERIFY_DEFAULT_MAX_TOKENS = 2048
FORGE_VERIFY_DEFAULT_PROMPT_SUITE = "long-code-uncapped"
MTP_PAYLOAD_AUDIT_ZERO_SAMPLE_LIMIT = 12
RUNTIME_BLOCKER_STATUS_PREFIXES = (
    "candidate",
    "fail",
    "pending",
    "blocked",
    "blocker",
    "needs",
    "unverified",
)


class ForgeError(RuntimeError):
    def __init__(self, message: str, *, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def cmd_forge_public(args: Any) -> int:
    action = getattr(args, "forge_action", None)
    try:
        if action == "probe":
            return _cmd_probe(args)
        if action == "build":
            return _cmd_build(args)
        if action == "discover":
            return _cmd_discover(args)
        if action == "publish":
            return _cmd_publish(args)
        if action == "inspect":
            return _cmd_inspect(args)
        if action == "verify":
            return _cmd_verify(args)
        if action == "cancel":
            return _cmd_cancel(args)
        raise ForgeError("missing forge subcommand", code=2)
    except ForgeError as exc:
        _err(str(exc))
        return int(exc.code)
    except KeyboardInterrupt:
        _err("forge interrupted")
        return 130
    except Exception as exc:
        _err(f"forge failed: {exc}")
        return 1


def atomic_write_json(path: Path | str, payload: dict[str, Any] | list[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)


def _json_out(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()


def _err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(out: str | Path, run_id: str) -> Path:
    if not str(run_id).strip():
        raise ForgeError("--run-id is required", code=2)
    root = Path(out).expanduser()
    run = root / str(run_id)
    run.mkdir(parents=True, exist_ok=True)
    return run


def _write_download(
    run: Path,
    *,
    bytes_on_disk: int,
    total_bytes: int | None = None,
    mb_per_s: float = 0.0,
    eta_s: float | None = None,
    label: str | None = None,
    finished: bool = False,
    stalled_s: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "bytes_on_disk": int(max(0, bytes_on_disk)),
        "mb_per_s": float(max(0.0, mb_per_s)),
        "finished": bool(finished),
    }
    if total_bytes is not None:
        payload["total_bytes"] = int(max(0, total_bytes))
    if eta_s is not None:
        payload["eta_s"] = float(max(0.0, eta_s))
    if label:
        payload["label"] = label
    if stalled_s is not None:
        payload["stalled_s"] = float(max(0.0, stalled_s))
    atomic_write_json(run / "download.json", payload)


def _write_progress(
    run: Path,
    name: str,
    *,
    progress: float,
    label: str | None = None,
    finished: bool = False,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "progress": max(0.0, min(1.0, float(progress))),
        "finished": bool(finished),
    }
    if label:
        payload["label"] = label
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    atomic_write_json(run / f"{name}.json", payload)


def _write_verify(run: Path, rows: list[dict[str, Any]]) -> None:
    annotated = _annotate_verify_rows(rows) if rows else []
    payload: dict[str, Any] = {"rows": annotated}
    if annotated:
        evidence = _speed_evidence(annotated)
        payload["verdict"] = evidence.get("verdict")
        payload["failure_reasons"] = evidence.get("failure_reasons", [])
    atomic_write_json(run / "verify.json", payload)


def _write_build_outcome(run: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(run / "build_outcome.json", payload)


def _write_brand(run: Path, branded_name: str, runtime_metadata: dict[str, Any]) -> None:
    atomic_write_json(
        run / "brand.json",
        {"branded_name": branded_name, "runtime_metadata": runtime_metadata},
    )


def _write_forge(run: Path, local_path: Path, runtime_metadata: dict[str, Any]) -> None:
    atomic_write_json(
        run / "forge.json",
        {"local_path": str(local_path), "runtime_metadata": runtime_metadata},
    )


def _write_publish(
    run: Path,
    *,
    bytes_uploaded: int,
    total_bytes: int | None = None,
    mb_per_s: float = 0.0,
    repo_created: str | None = None,
    revision: str | None = None,
    repo: str | None = None,
    finished: bool = False,
) -> None:
    payload: dict[str, Any] = {
        "bytes_uploaded": int(max(0, bytes_uploaded)),
        "mb_per_s": float(max(0.0, mb_per_s)),
        "finished": bool(finished),
    }
    if total_bytes is not None:
        payload["total_bytes"] = int(max(0, total_bytes))
    if repo_created:
        payload["repo_created"] = repo_created
    if revision:
        payload["revision"] = revision
    if repo:
        payload["repo"] = repo
    atomic_write_json(run / "publish.json", payload)


def _read_recipe(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ForgeError(f"--recipe must be JSON: {exc}", code=2) from exc
    if not isinstance(value, dict):
        raise ForgeError("--recipe must decode to an object", code=2)
    return value


def _guard_degraded_mtp(recipe: dict[str, Any], *, allow: bool) -> None:
    if str(recipe.get("mtp_policy") or "").strip() == "requantize" and not allow:
        raise ForgeError(REQUANTIZE_REFUSAL, code=2)


# FP16 forge lane (#166): M1/M2 GPUs have no native BF16, so FP16-typed
# non-quantized parameters prompt-process measurably faster there. "bf16"
# keeps today's behaviour (mlx-lm follows the source config's torch_dtype);
# "auto" resolves by host chip so the app and CLI share one detection.
_BODY_DTYPE_ALIASES = {
    "auto": "auto",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "f16": "fp16",
}


def _host_chip_brand() -> str:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _host_prefers_fp16() -> bool:
    """M1/M2-family GPUs have no native BF16 (emulated, slower prompt
    processing); M3 and newer run BF16 natively. Unknown chips keep bf16."""
    match = re.search(r"\bM(\d+)\b", _host_chip_brand())
    if match is None:
        return False
    return int(match.group(1)) <= 2


def _body_dtype(recipe: dict[str, Any]) -> str:
    raw = str(recipe.get("body_dtype") or "bf16").strip().lower()
    normalized = _BODY_DTYPE_ALIASES.get(raw)
    if normalized is None:
        raise ForgeError(
            f"recipe body_dtype must be auto, bf16, or fp16, got {raw!r}", code=2
        )
    if normalized == "auto":
        return "fp16" if _host_prefers_fp16() else "bf16"
    return normalized


def _normalize_source(source: str) -> tuple[Path | None, str | None]:
    local = Path(source).expanduser()
    if local.exists():
        return local, None
    return None, repo_id_from_model_ref(source)


def _quantization(config: dict[str, Any]) -> dict[str, Any]:
    tcfg = text_config(config)
    quant = config.get("quantization_config") or config.get("quantization") or {}
    if not quant:
        quant = tcfg.get("quantization_config") or tcfg.get("quantization") or {}
    return quant if isinstance(quant, dict) else {}


def _quant_text(quant: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("quant_method", "format", "bits", "group_size", "desc_act", "scheme"):
        if key in quant:
            parts.append(str(quant[key]))
    parts.extend(str(item) for item in quant.values() if isinstance(item, str))
    return " ".join(parts).lower()


def _declares_mtp(config: dict[str, Any]) -> bool:
    tcfg = text_config(config)
    return bool(
        tcfg.get("mtp_num_hidden_layers")
        or tcfg.get("num_nextn_predict_layers")
        or tcfg.get("num_mtp_modules")
        or config.get("num_nextn_predict_layers")
        or config.get("num_mtp_modules")
        or "mtp" in " ".join(str(item) for item in config.get("architectures") or []).lower()
        or "nextn" in " ".join(str(item) for item in config.get("architectures") or []).lower()
    )


def _inspection_has_mtp_weight_evidence(inspection: Any) -> bool:
    mtp = getattr(inspection, "mtp", None)
    if mtp is not None and bool(getattr(mtp, "exists", False)):
        if int(getattr(mtp, "tensor_count", 0) or 0) > 0:
            return True
        if bool(getattr(mtp, "passes_tensor_gate", False)):
            return True

    keys = tuple(str(key) for key in (getattr(inspection, "weight_keys", ()) or ()))
    if not keys:
        return False
    mtp_layers = int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0)
    start = int(getattr(inspection, "num_hidden_layers", 0) or 0)
    if mtp_layers <= 0:
        return False
    for local_idx in range(mtp_layers):
        layer_idx = start + local_idx if start > 0 else local_idx
        prefixes = (
            f"model.layers.{layer_idx}.",
            f"backbone.layers.{layer_idx}.",
            f"mtp.layers.{local_idx}.",
            f"layers.{local_idx}.",
            f"model.mtp_layers.{local_idx}.",
            f"model.mtp_layers.{layer_idx}.",
        )
        if any(key.startswith(prefixes) for key in keys):
            return True
    return False


def _compatibility_probe_fields(compatibility: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(compatibility, dict) or not compatibility:
        return {}
    return {
        "architecture_id": compatibility.get("arch_id"),
        "recommended_backend": compatibility.get("recommended_backend"),
        "runtime_compatibility": compatibility.get("runtime_compatibility"),
        "support_level": compatibility.get("support_level"),
        "architecture_recognized": compatibility.get("recognized", False),
        "backend_status": compatibility.get("runtime_compatibility"),
    }


def _backend_pending_probe_message(compatibility: dict[str, Any] | None) -> str | None:
    if not isinstance(compatibility, dict):
        return None
    runtime_compatibility = str(compatibility.get("runtime_compatibility") or "")
    support_level = str(compatibility.get("support_level") or "")
    if runtime_compatibility == "recognized-backend-pending" or support_level == "recognized-backend-pending":
        return "MTP detected, backend not ready."
    return None


def _gemma_pair_required_probe_message(compatibility: dict[str, Any] | None) -> str | None:
    if not isinstance(compatibility, dict):
        return None
    if compatibility.get("arch_id") == "gemma4-assistant-mtp":
        return "Gemma MTP is supported through assistant-pair bundles only."
    return None


def _probe_runtime_mtp_evidence(
    source_ref: str,
    *,
    runtime_metadata: dict[str, Any] | None,
) -> tuple[bool, str | None, dict[str, Any]]:
    if runtime_metadata is not None:
        return True, None, {
            "arch_id": runtime_metadata.get("arch_id"),
            "runtime_compatibility": "runtime-contract",
            "support_level": "already-mtplx",
            "recognized": True,
            "can_run": True,
            "recommended_backend": runtime_metadata.get("recommended_backend"),
        }
    try:
        inspection = inspect_model(source_ref)
    except Exception as exc:
        return False, str(exc), {}
    compatibility = getattr(inspection, "compatibility", {}) or {}
    if bool(compatibility.get("can_run")):
        return True, None, compatibility
    mtp = getattr(inspection, "mtp", None)
    tensor_count = int(getattr(mtp, "tensor_count", 0) or 0) if mtp is not None else 0
    if tensor_count > 0 or _inspection_has_mtp_weight_evidence(inspection):
        return True, str(compatibility.get("runtime_compatibility") or "unverified"), compatibility
    return False, str(compatibility.get("runtime_compatibility") or "missing-mtp-weights"), compatibility


def _no_mtp_probe_message(diagnostic: str | None) -> str:
    if diagnostic == "incomplete-assistant-pair":
        return (
            "Gemma 4 assistant/target subfolder detected, but MTPLX Gemma "
            "requires the assistant-pair bundle root containing mtplx_pair.json, "
            "target/, and assistant/. Point Forge at the bundle root instead."
        )
    return (
        "Source does not contain runnable MTP weights; Forge refuses "
        "to brand AR-only or config-only artifacts."
    )


def _source_format_from_config(
    config: dict[str, Any],
    *,
    files: Iterable[str] = (),
    runtime_metadata: dict[str, Any] | None = None,
) -> str:
    file_set = set(files)
    quant = _quantization(config)
    quant_text = _quant_text(quant)
    extra = config.get("mlx_lm_extra_tensors") or {}
    mtp_file = str(extra.get("mtp_file") or "") if isinstance(extra, dict) else ""
    has_mlx_quant = bool(config.get("quantization") or text_config(config).get("quantization"))
    if runtime_metadata is not None:
        return SOURCE_MLX_AFFINE_WITH_MTP
    if mtp_file or any(Path(name).name == "mtp.safetensors" for name in file_set):
        if quant or has_mlx_quant:
            return SOURCE_MLX_AFFINE_WITH_MTP
        return SOURCE_BF16_NATIVE
    if _is_autoawq_quantization(quant, quant_text):
        return SOURCE_AUTOAWQ
    if _is_compressed_tensors_nvfp4(quant, quant_text):
        return SOURCE_COMPRESSED_TENSORS_NVFP4
    if _is_compressed_tensors_awq(quant, quant_text):
        return SOURCE_COMPRESSED_TENSORS_AWQ
    if quant or has_mlx_quant:
        return SOURCE_MLX_AFFINE
    if _declares_mtp(config):
        return SOURCE_BF16_NATIVE
    return SOURCE_HF_VLLM if file_set else SOURCE_UNKNOWN


def _is_autoawq_quantization(quant: dict[str, Any], quant_text: str) -> bool:
    method = str(quant.get("quant_method") or "").lower()
    return method == "awq" or "autoawq" in quant_text


def _is_compressed_tensors_nvfp4(quant: dict[str, Any], quant_text: str) -> bool:
    if "nvfp4" in quant_text:
        return True
    if str(quant.get("quant_method") or "").lower() != "compressed-tensors":
        return False
    groups = quant.get("config_groups")
    if not isinstance(groups, dict):
        return False
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if isinstance(weights, dict) and str(weights.get("type") or "").lower() == "float":
            return True
    return False


def _is_compressed_tensors_awq(quant: dict[str, Any], quant_text: str) -> bool:
    if "compressed" not in quant_text and "awq" not in quant_text:
        return False
    if "awq" in quant_text:
        return True
    groups = quant.get("config_groups")
    if not isinstance(groups, dict):
        return "compressed" in quant_text
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if isinstance(weights, dict) and str(weights.get("type") or "").lower() == "int":
            return True
    return False


def _unsupported_mtp_source_message(source_format: str) -> str | None:
    if source_format not in UNSUPPORTED_MTP_SOURCE_FORMATS:
        return None
    if source_format == SOURCE_COMPRESSED_TENSORS_NVFP4:
        return (
            "Source has real MTP weights, but its compressed-tensors NVFP4 format "
            "needs a dedicated float4 converter; Forge will not treat it as AWQ."
        )
    return None


def _has_gguf_marker(source: Path | None, files: Iterable[str]) -> bool:
    if source is not None:
        if source.is_file() and source.suffix.lower() == ".gguf":
            return True
        if source.is_dir():
            return any(path.suffix.lower() == ".gguf" for path in source.rglob("*"))
    return any(Path(name).suffix.lower() == ".gguf" for name in files)


def _is_gemma4_assistant_pair_source(
    *,
    local: Path | None,
    repo_id: str | None,
    files: Iterable[str],
) -> bool:
    if local is not None and local.is_dir():
        return resolve_gemma4_pair_paths(local) is not None
    file_set = {str(name).strip("/") for name in files}
    return bool(
        is_gemma4_pair_repo_id(repo_id)
        or (
            GEMMA4_PAIR_FILE in file_set
            and "target/config.json" in file_set
            and "assistant/config.json" in file_set
        )
    )


def probe_source(source: str) -> dict[str, Any]:
    local, repo_id = _normalize_source(source)
    if local is None and repo_id is None:
        return {
            "verdict": "probe_failed",
            "forgeable": False,
            "supported": False,
            "source": source,
            "hf_repo": source,
            "source_format": SOURCE_UNKNOWN,
            "has_mtp_weights": False,
            "estimated_size_bytes": None,
            "estimated_peak_gib": None,
            "message": "Source must be a local path, Hugging Face repo id, or Hugging Face URL.",
            "diagnostic": "unrecognized_source_ref",
        }

    files: set[str] = set()
    estimated_size = None
    source_sha = None
    if repo_id is not None:
        try:
            api = _make_hf_api()
            info = api.model_info(
                repo_id=repo_id,
                files_metadata=True,
                token=hf_token_for_download(),
            )
            source_sha = getattr(info, "sha", None)
            siblings = getattr(info, "siblings", None) or []
            files = {
                str(getattr(sibling, "rfilename", "") or getattr(sibling, "path", ""))
                for sibling in siblings
            }
            estimated_size = sum(
                int(getattr(sibling, "size", 0) or 0) for sibling in siblings
            ) or None
            if not files:
                files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        except Exception as exc:
            return {
                "verdict": "probe_failed",
                "forgeable": False,
                "supported": False,
                "source": source,
                "hf_repo": repo_id,
                "source_format": SOURCE_UNKNOWN,
                "has_mtp_weights": False,
                "estimated_size_bytes": None,
                "estimated_peak_gib": None,
                "message": "Could not inspect Hugging Face source.",
                "diagnostic": str(exc),
            }
    elif local is not None and local.is_dir():
        files = {
            str(path.relative_to(local))
            for path in local.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        estimated_size = directory_size_bytes(local)
    elif local is not None and local.is_file():
        files = {local.name}
        estimated_size = local.stat().st_size

    if _has_gguf_marker(local, files):
        return {
            "verdict": "unsupported_source",
            "forgeable": False,
            "supported": False,
            "source": source,
            "hf_repo": repo_id or source,
            "source_format": SOURCE_UNKNOWN,
            "has_mtp_weights": False,
            "estimated_size_bytes": estimated_size,
            "estimated_peak_gib": _estimated_peak_gib(estimated_size),
            "message": GGUF_UNSUPPORTED_MESSAGE,
            "diagnostic": "gguf_unsupported_v1",
            "source_sha": source_sha,
        }

    runtime_metadata = None
    config: dict[str, Any] = {}
    try:
        if local is not None:
            runtime_path = local / "mtplx_runtime.json" if local.is_dir() else None
            if runtime_path is not None and runtime_path.exists():
                runtime_metadata = _load_json(runtime_path)
            config_path = local / "config.json" if local.is_dir() else None
            if config_path is not None and config_path.exists():
                config = _load_json(config_path)
        elif repo_id is not None:
            if "mtplx_runtime.json" in files:
                runtime_metadata = _hf_json(repo_id, "mtplx_runtime.json")
            if "config.json" in files:
                config = _hf_json(repo_id, "config.json")
    except Exception as exc:
        return {
            "verdict": "probe_failed",
            "forgeable": False,
            "supported": False,
            "source": source,
            "hf_repo": repo_id or source,
            "source_format": SOURCE_UNKNOWN,
            "has_mtp_weights": False,
            "estimated_size_bytes": estimated_size,
            "estimated_peak_gib": _estimated_peak_gib(estimated_size),
            "message": "Source metadata is not readable.",
            "diagnostic": str(exc),
            "source_sha": source_sha,
        }

    if _is_gemma4_assistant_pair_source(local=local, repo_id=repo_id, files=files):
        return {
            "verdict": "already_mtplx",
            "forgeable": False,
            "supported": True,
            "already_mtplx": True,
            "source": source,
            "hf_repo": repo_id or source,
            "source_format": SOURCE_GEMMA4_ASSISTANT_PAIR,
            "has_mtp_weights": True,
            "estimated_size_bytes": estimated_size,
            "estimated_peak_gib": _estimated_peak_gib(estimated_size),
            "message": (
                "Already a runnable MTPLX Gemma 4 assistant-pair bundle; "
                "Forge build conversion is not needed for this artifact."
            ),
            "diagnostic": None,
            "source_sha": source_sha,
        }

    source_format = _source_format_from_config(
        config,
        files=files,
        runtime_metadata=runtime_metadata,
    )
    inspection_ref = str(local) if local is not None and local.is_dir() else (repo_id or source)
    has_mtp, mtp_diagnostic, compatibility = _probe_runtime_mtp_evidence(
        inspection_ref,
        runtime_metadata=runtime_metadata,
    )
    compatibility_fields = _compatibility_probe_fields(compatibility)

    if runtime_metadata is not None:
        message = "Already MTPLX-branded; Forge can verify and restamp provenance."
        return {
            "verdict": "already_mtplx",
            "forgeable": True,
            "supported": True,
            "already_mtplx": True,
            "source": source,
            "hf_repo": repo_id or source,
            "source_format": SOURCE_MLX_AFFINE_WITH_MTP,
            "has_mtp_weights": True,
            "estimated_size_bytes": estimated_size,
            "estimated_peak_gib": _estimated_peak_gib(estimated_size),
            "message": message,
            "diagnostic": None,
            "source_sha": source_sha,
            **compatibility_fields,
        }
    if not has_mtp:
        return {
            "verdict": "no_mtp_heads",
            "forgeable": False,
            "supported": False,
            "source": source,
            "hf_repo": repo_id or source,
            "source_format": source_format,
            "has_mtp_weights": False,
            "estimated_size_bytes": estimated_size,
            "estimated_peak_gib": _estimated_peak_gib(estimated_size),
            "message": _no_mtp_probe_message(mtp_diagnostic),
            "diagnostic": mtp_diagnostic or "no_mtp_heads",
            "source_sha": source_sha,
            **compatibility_fields,
        }

    backend_pending_message = _backend_pending_probe_message(compatibility)
    if backend_pending_message is not None:
        architecture_id = compatibility.get("arch_id") if isinstance(compatibility, dict) else None
        return {
            "verdict": "unsupported_source",
            "forgeable": False,
            "supported": False,
            "already_mtplx": False,
            "source": source,
            "hf_repo": repo_id or source,
            "source_format": source_format,
            "has_mtp_weights": True,
            "estimated_size_bytes": estimated_size,
            "estimated_peak_gib": _estimated_peak_gib(estimated_size),
            "message": backend_pending_message,
            "diagnostic": f"backend_pending_mtp:{architecture_id or 'unknown'}",
            "source_sha": source_sha,
            **compatibility_fields,
        }

    gemma_pair_message = _gemma_pair_required_probe_message(compatibility)
    if gemma_pair_message is not None:
        return {
            "verdict": "unsupported_source",
            "forgeable": False,
            "supported": False,
            "already_mtplx": False,
            "source": source,
            "hf_repo": repo_id or source,
            "source_format": source_format,
            "has_mtp_weights": True,
            "estimated_size_bytes": estimated_size,
            "estimated_peak_gib": _estimated_peak_gib(estimated_size),
            "message": gemma_pair_message,
            "diagnostic": "gemma_assistant_pair_required",
            "source_sha": source_sha,
            **compatibility_fields,
        }

    unsupported_message = _unsupported_mtp_source_message(source_format)
    if unsupported_message is not None:
        return {
            "verdict": "unsupported_source",
            "forgeable": False,
            "supported": False,
            "already_mtplx": False,
            "source": source,
            "hf_repo": repo_id or source,
            "source_format": source_format,
            "has_mtp_weights": True,
            "estimated_size_bytes": estimated_size,
            "estimated_peak_gib": _estimated_peak_gib(estimated_size),
            "message": unsupported_message,
            "diagnostic": f"unsupported_quantization:{source_format}",
            "source_sha": source_sha,
            **compatibility_fields,
        }

    message_by_format = {
        SOURCE_BF16_NATIVE: "BF16/native MTP source detected; Forge will convert the body and preserve MTP.",
        SOURCE_MLX_AFFINE: "MLX affine source detected; Forge will package and verify MTP metadata.",
        SOURCE_MLX_AFFINE_WITH_MTP: "MLX affine source with MTP sidecar detected; Forge can verify and brand it.",
        SOURCE_AUTOAWQ: "AutoAWQ MTP source detected; Forge will convert packed INT4 weights to MLX affine.",
        SOURCE_COMPRESSED_TENSORS_AWQ: "Compressed-tensors/AWQ source detected; Forge will attempt MLX conversion.",
        SOURCE_COMPRESSED_TENSORS_NVFP4: "Compressed-tensors NVFP4 MTP source detected; Forge will convert FP4 weights to MLX affine and verify.",
        SOURCE_HF_VLLM: "HF/vLLM-compatible MTP source detected; Forge will attempt MLX conversion.",
        SOURCE_UNKNOWN: "MTP source detected but format is unfamiliar; Forge will attempt the conservative path.",
    }
    return {
        "verdict": "forgeable",
        "forgeable": True,
        "supported": True,
        "already_mtplx": False,
        "source": source,
        "hf_repo": repo_id or source,
        "source_format": source_format,
        "has_mtp_weights": True,
        "estimated_size_bytes": estimated_size,
        "estimated_peak_gib": _estimated_peak_gib(estimated_size),
        "message": message_by_format.get(source_format, message_by_format[SOURCE_UNKNOWN]),
        "diagnostic": None,
        "source_sha": source_sha,
        **compatibility_fields,
    }


def _cmd_probe(args: Any) -> int:
    payload = probe_source(str(args.source))
    if bool(getattr(args, "json", False)):
        _json_out(payload)
    else:
        print(payload["message"])
    return 1 if payload.get("verdict") == "probe_failed" else 0


def _cmd_discover(args: Any) -> int:
    limit = max(1, min(100, int(getattr(args, "limit", 20) or 20)))
    offset = max(0, int(getattr(args, "offset", 0) or 0))
    query = str(getattr(args, "query", None) or "MTPLX")
    try:
        rows = discover_models(query=query, limit=limit, offset=offset)
    except Exception as exc:
        _err(f"hf_unreachable: {exc}")
        return 1
    _json_out(rows)
    return 0


# Downloads-sorted rows scanned per discover call. Bounded so a broad
# --query cannot walk the whole hub; ~10x the full MTPLX result set
# (109 repos for search=MTPLX, measured 2026-08-16).
_DISCOVER_SCAN_LIMIT = 1000


def is_discoverable_repo(repo: str) -> bool:
    """Discover keeps every repo whose *name* carries the MTPLX brand —
    any case, any position ("…-MTPLX", "MTPLX-…", "…-mtplx-4bit").

    The old ``"-MTPLX-" in repo`` check was a fossil of the retired
    ``<base>-MTPLX-<role>`` branding: Forge now brands artifacts
    ``<base>-MTPLX`` (suffix), so the dash-bounded case-sensitive match
    dropped the app's own output plus every lowercase/prefix community
    variant (36 of the 109 live MTPLX repos on 2026-08-16, including
    the #2-by-downloads one).
    """
    return "mtplx" in repo.rsplit("/", 1)[-1].lower()


def discover_models(*, query: str, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    api = _make_hf_api()
    # Sorted by downloads descending; the name filter runs while we
    # iterate, so rows keep coming until `limit` cards survive it. The
    # old shape sliced at HF first (`limit + offset` rows) and filtered
    # second, so every filtered row shrank the page and any repo ranked
    # below the slice — exactly the fresh, low-download models — could
    # never surface. huggingface_hub versions differ on offset support,
    # so offset still pages locally over the matching rows.
    list_kwargs = {
        "search": query or "MTPLX",
        "sort": "downloads",
        "direction": -1,
        "limit": _DISCOVER_SCAN_LIMIT,
    }
    try:
        models = api.list_models(**list_kwargs)
    except TypeError as exc:
        if "direction" not in str(exc):
            raise
        list_kwargs.pop("direction", None)
        models = api.list_models(**list_kwargs)
    cards: list[dict[str, Any]] = []
    matched = 0
    for model in models:
        repo = str(
            getattr(model, "modelId", None)
            or getattr(model, "id", None)
            or getattr(model, "model_id", None)
            or ""
        )
        if not is_discoverable_repo(repo):
            continue
        matched += 1
        if matched <= offset:
            continue
        cards.append(_discover_card(model, repo))
        if len(cards) >= limit:
            break
    return cards


def _discover_card(model: Any, repo: str) -> dict[str, Any]:
    owner = repo.split("/", 1)[0] if "/" in repo else ""
    branded_name = repo.split("/", 1)[1] if "/" in repo else repo
    siblings = getattr(model, "siblings", None) or []
    size = sum(int(getattr(item, "size", 0) or 0) for item in siblings) or None
    tags = [str(tag) for tag in (getattr(model, "tags", None) or [])]
    license_tag = next(
        (tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")),
        None,
    )
    updated = (
        getattr(model, "lastModified", None)
        or getattr(model, "last_modified", None)
        or getattr(model, "createdAt", None)
    )
    card: dict[str, Any] = {
        "repo": repo,
        "owner": owner,
        "branded_name": branded_name,
        "downloads": int(getattr(model, "downloads", 0) or 0),
    }
    if size is not None:
        card["size_bytes"] = size
    if license_tag:
        card["license"] = license_tag
    if updated is not None:
        card["last_updated"] = updated.isoformat() if hasattr(updated, "isoformat") else str(updated)
    if os.environ.get("MTPLX_FORGE_DISCOVER_RUNTIME") == "1":
        runtime = _try_hf_runtime(repo)
        if runtime:
            depth = runtime.get("mtp_depth_max")
            if isinstance(depth, int):
                card["depth"] = depth
            multiplier = _runtime_multiplier(runtime)
            if multiplier is not None:
                card["multiplier_vs_ar"] = multiplier
    return card


def _cmd_inspect(args: Any) -> int:
    model_path = Path(args.path).expanduser()
    runtime_path = model_path / "mtplx_runtime.json"
    if runtime_path.exists():
        payload = _load_json(runtime_path)
    else:
        payload = inspect_model(model_path).to_dict()
    if bool(getattr(args, "json", False)):
        _json_out(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_verify(args: Any) -> int:
    model_path = Path(args.path).expanduser()
    if getattr(args, "out", None) and getattr(args, "run_id", None):
        run = _run_dir(args.out, args.run_id)
    else:
        run = _run_dir(Path("outputs/forge-verify"), f"verify-{int(time.time())}")
    rows = _run_verify(
        model_path,
        run,
        max_fans=bool(getattr(args, "max", False)),
        mtp_contract=_runtime_or_default_mtp_contract(_read_runtime(model_path), model_path),
        max_tokens=_forge_verify_max_tokens(args),
        prompt_suite=_forge_verify_prompt_suite(args),
    )
    payload = {"rows": rows}
    if bool(getattr(args, "json", False)):
        _json_out(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if rows else 1


def _cmd_build(args: Any) -> int:
    recipe = _read_recipe(args.recipe)
    if getattr(args, "dtype", None):
        recipe["body_dtype"] = str(args.dtype)
    _body_dtype(recipe)  # validate early, before any download starts
    _guard_degraded_mtp(recipe, allow=bool(getattr(args, "allow_degraded_mtp", False)))
    run = _run_dir(args.out, args.run_id)
    branded_name = _sanitize_branded_name(args.branded_name)
    if not branded_name:
        raise ForgeError("--branded-name cannot be empty", code=2)

    _write_download(run, bytes_on_disk=0, label="starting", finished=False)

    probe = probe_source(args.repo)
    if not probe.get("forgeable"):
        message = str(probe.get("message") or "source is not forgeable")
        diagnostic = str(probe.get("diagnostic") or "").strip()
        if diagnostic and diagnostic not in message:
            message = f"{message} ({diagnostic})"
        raise ForgeError(message)
    if _cancel_requested(args.run_id):
        raise ForgeError("forge cancelled", code=130)

    source_path, source_repo, source_sha = _prepare_source(args.repo, run, probe)
    if _cancel_requested(args.run_id):
        raise ForgeError("forge cancelled", code=130)

    destination = _unique_model_dir(branded_name)
    source_format = str(probe.get("source_format") or SOURCE_UNKNOWN)
    _err(f"[forge] source format: {source_format}")
    if source_format in {SOURCE_MLX_AFFINE, SOURCE_MLX_AFFINE_WITH_MTP} or probe.get("already_mtplx"):
        if _body_dtype(recipe) == "fp16":
            _err(
                "[forge] source is already MLX; body dtype follows the source "
                "weights (fp16 request applies only to fresh conversions)"
            )
        _mirror_model_tree(source_path, destination)
        _write_progress(run, "convert", progress=1.0, label="to_mlx", finished=True)
    elif source_format in {SOURCE_AUTOAWQ, SOURCE_COMPRESSED_TENSORS_AWQ}:
        _convert_compressed_tensors_awq(
            source_path,
            destination,
            run=run,
            source_repo=source_repo or args.repo,
            source_sha=source_sha or str(probe.get("source_sha") or ""),
        )
    elif source_format == SOURCE_COMPRESSED_TENSORS_NVFP4:
        _convert_compressed_tensors_nvfp4(
            source_path,
            destination,
            run=run,
            source_repo=source_repo or args.repo,
            source_sha=source_sha or str(probe.get("source_sha") or ""),
            recipe=recipe,
        )
    else:
        _convert_with_mlx_lm(
            source_path,
            destination,
            recipe=recipe,
            source_format=source_format,
            run=run,
        )
    if _cancel_requested(args.run_id):
        raise ForgeError("forge cancelled", code=130)

    _ensure_vision_tower(source_path, destination)
    _validate_vision_payload(source_path, destination)

    _calibrate_sidecar(
        source_path,
        destination,
        recipe=recipe,
        run=run,
    )

    existing_runtime = _read_runtime(destination) or _read_runtime(source_path)
    require_all_depths = True
    rows = _verify_rows_from_runtime(existing_runtime)
    calibration_diagnostic: str | None = None
    has_saved_contract = _runtime_has_mtp_contract(existing_runtime, destination)
    has_legacy_speed_grid = _runtime_has_legacy_speed_grid(existing_runtime)
    reuse_blocker = (
        _saved_verify_rows_reuse_blocker(
            rows,
            existing_runtime,
            model_path=destination,
            source_path=source_path,
            require_all_depths=require_all_depths,
        )
        if has_saved_contract or has_legacy_speed_grid
        else "missing saved MTP contract"
    )
    if (has_saved_contract or has_legacy_speed_grid) and reuse_blocker is None:
        mtp_contract = _runtime_or_default_mtp_contract(existing_runtime, destination)
        _patch_config_for_mtp_contract(destination, mtp_contract)
        _write_verify(run, rows)
    else:
        if reuse_blocker:
            _err(f"[forge] saved verification not reusable: {reuse_blocker}; re-verifying")
        mtp_contract = _calibrate_mtp_contract(
            destination,
            run,
            recipe=recipe,
            existing=existing_runtime,
            max_fans=bool(getattr(args, "max", False)),
        )
        calibration = mtp_contract.get("calibration") if isinstance(mtp_contract, dict) else None
        if isinstance(calibration, dict):
            calibration_diagnostic = str(calibration.get("diagnostic") or "").strip() or None
        _patch_config_for_mtp_contract(destination, mtp_contract)
        rows = _run_verify(
            destination,
            run,
            max_fans=bool(getattr(args, "max", False)),
            mtp_contract=mtp_contract,
            max_tokens=_forge_verify_max_tokens(args),
            prompt_suite=_forge_verify_prompt_suite(args),
        )
    if not rows:
        raise ForgeError("verification produced no usable AR/MTP rows")
    _require_verify_rows(rows, require_all_depths=require_all_depths)
    _require_speed_win_or_write_outcome(
        destination,
        run,
        rows,
        diagnostic=calibration_diagnostic,
    )

    runtime_metadata = _stamp_runtime_metadata(
        destination,
        branded_name=branded_name,
        source_repo=source_repo or args.repo,
        source_sha=source_sha or str(probe.get("source_sha") or ""),
        source_format=source_format,
        recipe=recipe,
        forge_inputs={
            "trunk_path": str(destination),
            "mtp_source_path": str(source_path),
        },
        rows=rows,
        mtp_contract=mtp_contract,
        existing=existing_runtime,
    )
    atomic_write_json(destination / "mtplx_runtime.json", runtime_metadata)
    _write_brand(run, branded_name, runtime_metadata)
    _write_forge(run, destination, runtime_metadata)
    if bool(getattr(args, "json", False)):
        _json_out({"local_path": str(destination), "runtime_metadata": runtime_metadata})
    return 0


def _prepare_source(source: str, run: Path, probe: dict[str, Any]) -> tuple[Path, str | None, str | None]:
    local, repo_id = _normalize_source(source)
    if local is not None:
        size = directory_size_bytes(local) if local.is_dir() else local.stat().st_size
        _write_download(
            run,
            bytes_on_disk=size,
            total_bytes=size,
            mb_per_s=0.0,
            label="local source",
            finished=True,
        )
        return local, None, None
    if repo_id is None:
        raise ForgeError("build requires a local path or Hugging Face repo id", code=2)

    total = probe.get("estimated_size_bytes")
    started_at = time.monotonic()

    def progress(payload: dict[str, Any]) -> None:
        size = int(payload.get("size_bytes") or 0)
        elapsed = max(0.001, time.monotonic() - started_at)
        interval = _as_float(payload.get("interval_s"))
        delta = _as_float(payload.get("delta_bytes"))
        if interval is not None and interval > 0 and delta is not None:
            bytes_per_s = max(0.0, delta) / interval
        else:
            bytes_per_s = size / elapsed
        mb_per_s = bytes_per_s / (1024 * 1024)
        total_bytes = payload.get("total_bytes") or total
        eta = None
        if total_bytes and bytes_per_s > 0:
            eta = max(0.0, (int(total_bytes) - size) / bytes_per_s)
        event = str(payload.get("event") or "")
        if event == "complete":
            label = "downloaded"
        elif delta is not None and delta <= 0:
            label = "waiting for shard data"
        else:
            label = "downloading"
        _write_download(
            run,
            bytes_on_disk=size,
            total_bytes=int(total_bytes) if total_bytes else None,
            mb_per_s=mb_per_s,
            eta_s=eta,
            label=label,
            finished=payload.get("event") == "complete",
            stalled_s=_as_float(payload.get("stalled_s")),
        )

    _err(f"[forge] downloading {repo_id}")
    result = pull_model(
        repo_id,
        progress_callback=progress,
        progress_interval_s=2.0,
    )
    resolved = Path(result["path"])
    final_size = int(result.get("size_bytes") or directory_size_bytes(resolved))
    _write_download(
        run,
        bytes_on_disk=final_size,
        total_bytes=int(total or final_size),
        mb_per_s=0.0,
        label="downloaded",
        finished=True,
    )
    return resolved, repo_id, str(probe.get("source_sha") or "")


def _convert_with_mlx_lm(
    source: Path,
    destination: Path,
    *,
    recipe: dict[str, Any],
    source_format: str,
    run: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_progress(run, "convert", progress=0.05, label="to_mlx", finished=False)
    if recipe.get("module_overrides"):
        command = _mixed_convert_command(
            source,
            destination,
            recipe=recipe,
            source_format=source_format,
        )
        _err("[forge] converting with mlx-lm (mixed-precision module overrides)")
    else:
        command = _mlx_lm_convert_command(
            source,
            destination,
            recipe=recipe,
            source_format=source_format,
        )
        _err("[forge] converting with mlx-lm")
    stdout_path = run / "convert.stdout.log"
    stderr_path = run / "convert.stderr.log"
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=str(run),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        tick = 0
        while proc.poll() is None:
            tick += 1
            _write_progress(
                run,
                "convert",
                progress=min(0.95, 0.05 + tick * 0.01),
                label="quantize_body",
                finished=False,
            )
            time.sleep(1.0)
        returncode = proc.returncode
    if returncode != 0:
        raise ForgeError(
            "mlx-lm conversion failed: " + _tail(stderr_path, fallback=stdout_path)
        )
    _write_progress(
        run,
        "convert",
        progress=1.0,
        label="quantize_body",
        finished=True,
        elapsed_s=round(time.monotonic() - started, 3),
    )


def _mixed_convert_command(
    source: Path,
    destination: Path,
    *,
    recipe: dict[str, Any],
    source_format: str,
) -> list[str]:
    # module_overrides quantize per-module via a custom predicate, which only
    # the in-process mlx-lm API supports; BF16-native sources only, because
    # packed sources dequantize through dedicated lanes with their own params.
    if source_format != SOURCE_BF16_NATIVE:
        raise ForgeError(
            "recipe module_overrides require a BF16/native source, got "
            f"{source_format}",
            code=2,
        )
    raw_body_bits = recipe.get("body_bits")
    if int(4 if raw_body_bits is None else raw_body_bits) <= 0:
        raise ForgeError("recipe module_overrides require body_bits > 0", code=2)
    command = [
        sys.executable,
        "-P",
        "-m",
        "mtplx.commands.forge_mixed_convert",
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--recipe-json",
        json.dumps(recipe, sort_keys=True),
    ]
    if _body_dtype(recipe) == "fp16":
        command.extend(["--dtype", "float16"])
    return command


def _mlx_lm_convert_command(
    source: Path,
    destination: Path,
    *,
    recipe: dict[str, Any],
    source_format: str,
) -> list[str]:
    command = [
        sys.executable,
        "-P",
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        str(source),
        "--mlx-path",
        str(destination),
    ]
    if source_format == SOURCE_COMPRESSED_TENSORS_AWQ:
        command.append("--dequantize")
    bits = int(recipe.get("body_bits") or 4)
    group_size = int(recipe.get("body_group_size") or 64)
    mode = str(recipe.get("body_mode") or "affine")
    if bits > 0:
        command.extend(
            [
                "--quantize",
                "--q-bits",
                str(bits),
                "--q-group-size",
                str(group_size),
                "--q-mode",
                mode,
            ]
        )
    # bf16 passes no --dtype so mlx-lm keeps following the source config's
    # torch_dtype (today's behaviour); fp16 covers every non-quantized
    # parameter, and with body_bits 0 the whole trunk.
    if _body_dtype(recipe) == "fp16":
        command.extend(["--dtype", "float16"])
    return command


def _convert_compressed_tensors_awq(
    source: Path,
    destination: Path,
    *,
    run: Path,
    source_repo: str,
    source_sha: str,
) -> None:
    _write_progress(run, "convert", progress=0.05, label="convert_packed_awq", finished=False)

    def progress(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        if total <= 0:
            return
        fraction = max(0.0, min(1.0, completed / total))
        _write_progress(
            run,
            "convert",
            progress=min(0.98, 0.05 + fraction * 0.9),
            label=str(payload.get("filename") or "convert_packed_awq"),
            finished=False,
        )

    _err("[forge] converting compressed-tensors AWQ to MLX affine")
    started = time.monotonic()
    from mtplx.compressed_tensors import convert_compressed_tensors_awq_to_mlx

    report = convert_compressed_tensors_awq_to_mlx(
        source,
        destination,
        source_repo=source_repo,
        source_sha=source_sha or None,
        progress_callback=progress,
    )
    audit = report.get("audit") if isinstance(report, dict) else None
    if isinstance(audit, dict) and not audit.get("passed", False):
        problems = ", ".join(str(item) for item in audit.get("problems") or [])
        raise ForgeError(f"compressed-tensors conversion audit failed: {problems}")
    _write_progress(
        run,
        "convert",
        progress=1.0,
        label="convert_packed_awq",
        finished=True,
        elapsed_s=round(time.monotonic() - started, 3),
    )


def _convert_compressed_tensors_nvfp4(
    source: Path,
    destination: Path,
    *,
    run: Path,
    source_repo: str,
    source_sha: str,
    recipe: dict[str, Any],
) -> None:
    _write_progress(run, "convert", progress=0.05, label="convert_nvfp4", finished=False)

    def progress(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        if total <= 0:
            return
        fraction = max(0.0, min(1.0, completed / total))
        _write_progress(
            run,
            "convert",
            progress=min(0.98, 0.05 + fraction * 0.9),
            label=str(payload.get("filename") or "convert_nvfp4"),
            finished=False,
        )

    bits = int(recipe.get("body_bits") or 4)
    group_size = int(recipe.get("body_group_size") or 64)
    mode = str(recipe.get("body_mode") or "affine")
    _err(
        "[forge] converting compressed-tensors NVFP4 to MLX affine "
        f"(q{bits}, group {group_size}, {mode})"
    )
    started = time.monotonic()
    from mtplx.compressed_tensors import convert_compressed_tensors_nvfp4_to_mlx

    report = convert_compressed_tensors_nvfp4_to_mlx(
        source,
        destination,
        source_repo=source_repo,
        source_sha=source_sha or None,
        progress_callback=progress,
        target_bits=bits,
        target_group_size=group_size,
        target_mode=mode,
    )
    audit = report.get("audit") if isinstance(report, dict) else None
    if isinstance(audit, dict) and not audit.get("passed", False):
        problems = ", ".join(str(item) for item in audit.get("problems") or [])
        raise ForgeError(f"compressed-tensors NVFP4 conversion audit failed: {problems}")
    _write_progress(
        run,
        "convert",
        progress=1.0,
        label="convert_nvfp4",
        finished=True,
        elapsed_s=round(time.monotonic() - started, 3),
    )


def _calibrate_sidecar(source: Path, destination: Path, *, recipe: dict[str, Any], run: Path) -> None:
    policy = str(recipe.get("mtp_policy") or "keep_bf16")
    label = "extract_mtp" if policy != "requantize" else "requantize_mtp"
    _write_progress(run, "calibrate", progress=0.2, label=label, finished=False)
    if policy == "requantize":
        _write_progress(
            run,
            "calibrate",
            progress=1.0,
            label="requantize_mtp",
            finished=True,
        )
        return
    if not _ensure_mtp_sidecar(source, destination):
        _err("[forge] no standalone MTP sidecar extracted; relying on embedded MTP keys if present")
    elif _body_dtype(recipe) == "fp16":
        _cast_sidecar_float_tensors_fp16(destination / "mtp.safetensors")
    _write_progress(run, "calibrate", progress=0.75, label="pack_sidecar", finished=False)
    _patch_config_for_mtp(destination)
    _validate_mtp_sidecar_payload(destination)
    _write_progress(run, "calibrate", progress=1.0, label="pack_sidecar", finished=True)


def _cast_sidecar_float_tensors_fp16(sidecar_path: Path) -> None:
    """Cast the sidecar's floating tensors to float16 for fp16 builds.

    Matches the released FP16 siblings (their mtp.safetensors carries F16
    floats): a BF16 sidecar on an FP16 trunk would draft through the emulated
    BF16 path on M1/M2, giving up part of the speed the build exists for.
    Integer tensors (quantization indices) are untouched.
    """
    if not sidecar_path.exists():
        return
    import mlx.core as mx

    weights = mx.load(str(sidecar_path))
    cast: dict[str, Any] = {}
    changed = False
    for key, value in weights.items():
        if value.dtype in (mx.bfloat16, mx.float32):
            cast[key] = value.astype(mx.float16)
            changed = True
        else:
            cast[key] = value
    if not changed:
        return
    # mx.load is lazy (file-backed): every value in ``cast`` must be
    # materialized before the file it references is rewritten, and the write
    # must not target the path being read. Saving in place over the lazy
    # handles corrupted the 4B sidecar (#176): tensors materialized mid-write
    # read clobbered regions, silently swapping norm payloads between keys.
    mx.eval(list(cast.values()))
    tmp_path = sidecar_path.with_name(f"{sidecar_path.stem}.fp16-tmp.safetensors")
    mx.save_safetensors(str(tmp_path), cast, metadata={"format": "mlx"})
    written = mx.load(str(tmp_path))
    for key, value in cast.items():
        reloaded = written.get(key)
        if reloaded is None or not bool(mx.array_equal(reloaded, value)):
            raise ForgeError(
                f"fp16 sidecar cast readback mismatch for tensor {key}; "
                "refusing to replace the sidecar"
            )
    del written
    os.replace(tmp_path, sidecar_path)
    _err("[forge] MTP sidecar floats cast to fp16 to match the trunk dtype")


_SIDECAR_RMSNORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "norm.weight",
)


def _sidecar_norm_degeneracy(sidecar_path: Path) -> str | None:
    """Return a description if any sidecar RMSNorm tensor is degenerate.

    Catches corrupted writes (all-zero norms) and similar clobbered payloads
    so a stale broken sidecar can never be silently reused by a re-forge.
    Healthy final-convention norms sit well above the 0.05 mean floor.
    """
    try:
        import mlx.core as mx

        weights = mx.load(str(sidecar_path))
    except Exception as exc:
        return f"unreadable: {exc}"
    for key, value in weights.items():
        if value.ndim != 1 or not any(key.endswith(sfx) for sfx in _SIDECAR_RMSNORM_SUFFIXES):
            continue
        if value.dtype not in (mx.bfloat16, mx.float16, mx.float32):
            continue
        mean = float(value.astype(mx.float32).mean().item())
        std = float(value.astype(mx.float32).std().item())
        if not math.isfinite(mean) or not math.isfinite(std):
            return f"{key} has non-finite values"
        if std < 1e-4 and abs(mean) < 1e-4:
            return f"{key} is all-zero"
    return None


def _ensure_mtp_sidecar(source: Path, destination: Path) -> bool:
    target = destination / "mtp.safetensors"
    if target.exists():
        problem = _sidecar_norm_degeneracy(target)
        if problem is None:
            return True
        quarantined = target.with_name(f"mtp.safetensors.corrupt-{int(time.time())}")
        os.replace(target, quarantined)
        _err(
            f"[forge] existing MTP sidecar failed the norm sanity check ({problem}); "
            f"moved to {quarantined.name} and re-extracting"
        )
    try:
        config = _load_json(source / "config.json")
    except Exception:
        config = {}
    mtp_source = artifacts.expected_mtp_file(source, config) if source.is_dir() else source
    if mtp_source.exists() and mtp_source.is_file():
        _copy_file(mtp_source, target)
        return True
    if source.is_dir():
        return _extract_embedded_mtp(
            source,
            target,
            sanitize_values=_should_sanitize_extracted_mtp_weights(config),
        )
    return False


def _should_sanitize_extracted_mtp_weights(config: dict[str, Any]) -> bool:
    markers: list[str] = []
    for candidate in (config, text_config(config)):
        if not isinstance(candidate, dict):
            continue
        for key in ("model_type", "architectures"):
            value = candidate.get(key)
            if isinstance(value, str):
                markers.append(value)
            elif isinstance(value, list):
                markers.extend(str(item) for item in value)
    normalized = " ".join(markers).lower()
    return "qwen3_5" in normalized or "qwen3next" in normalized or "qwen3_next" in normalized


def _extract_embedded_mtp(
    source: Path,
    target: Path,
    *,
    sanitize_values: bool = False,
) -> bool:
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        return False
    index = _load_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        return False
    mtp_keys = [key for key in weight_map if artifacts.is_mtp_key(str(key))]
    if not mtp_keys:
        return False

    by_file: dict[str, list[str]] = {}
    for key in mtp_keys:
        filename = str(weight_map[key])
        by_file.setdefault(filename, []).append(key)
    _copy_safetensors_subset(
        source,
        by_file,
        target,
        key_transform=artifacts.normalize_mtp_key,
        sanitize_values=sanitize_values,
    )
    return True


def _copy_safetensors_subset(
    source_dir: Path,
    by_file: dict[str, list[str]],
    target: Path,
    *,
    key_transform: Callable[[str], str] | None = None,
    sanitize_values: bool = False,
) -> None:
    if sanitize_values:
        _copy_safetensors_subset_sanitized(
            source_dir,
            by_file,
            target,
            key_transform=key_transform,
        )
        return
    tensors: list[tuple[str, dict[str, Any], bytes]] = []
    for filename, keys in by_file.items():
        shard = source_dir / filename
        header_size, header = _read_safetensors_header(shard)
        for key in keys:
            raw_info = header.get(key)
            if not isinstance(raw_info, dict):
                raise ForgeError(f"MTP tensor {key} missing from {filename}")
            offsets = raw_info.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
            ):
                raise ForgeError(f"MTP tensor {key} has invalid safetensors offsets in {filename}")
            start, end = offsets
            if start < 0 or end < start:
                raise ForgeError(f"MTP tensor {key} has invalid safetensors range in {filename}")
            with shard.open("rb") as handle:
                handle.seek(8 + header_size + start)
                payload = handle.read(end - start)
            if len(payload) != end - start:
                raise ForgeError(f"MTP tensor {key} payload is truncated in {filename}")
            output_key = key_transform(key) if key_transform is not None else key
            info = {
                "dtype": raw_info.get("dtype"),
                "shape": raw_info.get("shape"),
            }
            if not isinstance(info["dtype"], str) or not isinstance(info["shape"], list):
                raise ForgeError(f"MTP tensor {key} has invalid safetensors metadata in {filename}")
            tensors.append((str(output_key), info, payload))
    if not tensors:
        raise ForgeError("embedded MTP extraction found no tensors to write")
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_safetensors_raw(target, tensors)


def _copy_safetensors_subset_sanitized(
    source_dir: Path,
    by_file: dict[str, list[str]],
    target: Path,
    *,
    key_transform: Callable[[str], str] | None = None,
) -> None:
    import mlx.core as mx

    from mtplx.compressed_tensors import sanitize_plain_weight

    tensors: dict[str, Any] = {}
    for filename, keys in by_file.items():
        shard = source_dir / filename
        try:
            loaded = mx.load(str(shard))
        except Exception as exc:
            raise ForgeError(f"could not load safetensors shard {filename}: {exc}") from exc
        for key in keys:
            if key not in loaded:
                raise ForgeError(f"MTP tensor {key} missing from {filename}")
            output_key = str(key_transform(key) if key_transform is not None else key)
            tensors[output_key] = sanitize_plain_weight(output_key, loaded[key])
    if not tensors:
        raise ForgeError("embedded MTP extraction found no tensors to write")
    mx.eval(list(tensors.values()))
    target.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(target), tensors, metadata={"format": "mlx"})


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                raise ForgeError(f"{path.name} is not a valid safetensors file")
            header_size = int.from_bytes(header_size_raw, "little")
            header = json.loads(handle.read(header_size).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ForgeError(f"{path.name} has invalid safetensors JSON metadata: {exc}") from exc
    except OSError as exc:
        raise ForgeError(f"could not read safetensors shard {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ForgeError(f"{path.name} has invalid safetensors metadata")
    return header_size, header


def _write_safetensors_raw(
    path: Path,
    tensors: list[tuple[str, dict[str, Any], bytes]],
) -> None:
    header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for key, info, payload in tensors:
        end = offset + len(payload)
        header[key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, end],
        }
        chunks.append(payload)
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for chunk in chunks:
            handle.write(chunk)


def _patch_config_for_mtp(destination: Path) -> None:
    config_path = destination / "config.json"
    if not config_path.exists() or not (destination / "mtp.safetensors").exists():
        return
    config = _load_json(config_path)
    extra = config.get("mlx_lm_extra_tensors")
    if not isinstance(extra, dict):
        extra = {}
    extra["mtp_file"] = "mtp.safetensors"
    config["mlx_lm_extra_tensors"] = extra
    atomic_write_json(config_path, config)


def _validate_mtp_sidecar_payload(destination: Path) -> dict[str, Any] | None:
    config_path = destination / "config.json"
    if not config_path.exists():
        return None
    try:
        config = _load_json(config_path)
    except Exception:
        config = {}
    mtp_path = artifacts.expected_mtp_file(destination, config)
    if not mtp_path.exists():
        return None
    audit = _audit_mtp_sidecar_payload(mtp_path)
    if not audit.get("passed", False):
        problems = ", ".join(str(item) for item in audit.get("problems") or [])
        raise ForgeError(f"MTP sidecar payload audit failed for {mtp_path.name}: {problems}")
    config = _load_json(config_path)
    config["mtplx_mtp_payload_audit"] = audit
    atomic_write_json(config_path, config)
    return audit


def _ensure_vision_tower(source: Path, destination: Path) -> None:
    """Restore the vision tower that text-only convert lanes drop.

    ``mlx_lm.convert`` serializes only the text model, so multimodal sources
    lose their vision tensors, ``vision_config``, and preprocessor sidecars
    (issue #263). Graft them back from the source directory. A no-op for
    text-only sources and for lanes (mirror, compressed-tensors) that already
    preserved the tower.
    """
    if not source.is_dir():
        return
    from mtplx.vision_graft import VisionGraftError, graft_vision_tower

    try:
        report = graft_vision_tower(source, destination, verify_load=True)
    except VisionGraftError as exc:
        raise ForgeError(f"vision tower graft failed: {exc}") from exc
    if report.get("status") == "grafted":
        _err(
            f"[forge] restored vision tower: {report.get('tensors')} tensors "
            f"({report.get('bytes')} bytes) in {report.get('vision_file')}"
        )


def _validate_vision_payload(source: Path, destination: Path) -> None:
    """Fail closed when a multimodal source produced a blind artifact."""
    if not source.is_dir():
        return
    try:
        source_config = _load_json(source / "config.json")
    except Exception:
        return
    if not isinstance(source_config, dict) or not isinstance(
        source_config.get("vision_config"), dict
    ):
        return
    from mtplx.vision import vision_spec_for_model_dir

    if vision_spec_for_model_dir(destination) is None:
        raise ForgeError(
            "source declares vision_config but the forged artifact resolves no "
            "vision tower; the convert lane dropped it (issue #263)"
        )


def _audit_mtp_sidecar_payload(mtp_path: Path) -> dict[str, Any]:
    problems: list[str] = []
    try:
        import mlx.core as mx

        tensors = mx.load(str(mtp_path))
    except Exception as exc:
        return {
            "passed": False,
            "problems": [f"could not read MTP sidecar tensors: {exc}"],
            "mtp_file": str(mtp_path),
        }
    if not tensors:
        problems.append("MTP sidecar is empty")

    payload_keys: list[str] = []
    nonzero_payload_keys: list[str] = []
    zero_payload_keys: list[str] = []
    scale_keys: list[str] = []
    zero_scale_keys: list[str] = []
    for key, value in sorted(tensors.items()):
        if str(key).endswith(".scales"):
            scale_keys.append(str(key))
            if not _tensor_has_nonzero(value):
                zero_scale_keys.append(str(key))
        if not _is_mtp_payload_tensor(str(key), value):
            continue
        payload_keys.append(str(key))
        if _tensor_has_nonzero(value):
            nonzero_payload_keys.append(str(key))
        else:
            zero_payload_keys.append(str(key))

    if tensors and not payload_keys:
        problems.append("MTP sidecar has no projection or MLP payload tensors")
    if payload_keys and not nonzero_payload_keys:
        problems.append("MTP sidecar has no nonzero projection or MLP payload tensors")
    if scale_keys and len(scale_keys) == len(zero_scale_keys):
        problems.append("MTP sidecar quantization scales are all zero")

    return {
        "passed": not problems,
        "problems": problems,
        "mtp_file": str(mtp_path),
        "tensor_count": len(tensors),
        "payload_tensor_count": len(payload_keys),
        "nonzero_payload_tensor_count": len(nonzero_payload_keys),
        "scale_tensor_count": len(scale_keys),
        "zero_payload_sample": zero_payload_keys[:MTP_PAYLOAD_AUDIT_ZERO_SAMPLE_LIMIT],
        "zero_scale_sample": zero_scale_keys[:MTP_PAYLOAD_AUDIT_ZERO_SAMPLE_LIMIT],
    }


def _is_mtp_payload_tensor(key: str, value: Any) -> bool:
    shape = tuple(int(dim) for dim in getattr(value, "shape", ()) or ())
    if len(shape) < 2:
        return False
    lowered = key.lower()
    if lowered.endswith(".scales") or lowered.endswith(".biases"):
        return False
    if "norm" in lowered or "layernorm" in lowered:
        return False
    return lowered.endswith(".weight")


def _tensor_has_nonzero(value: Any) -> bool:
    try:
        import mlx.core as mx

        shape = tuple(int(dim) for dim in getattr(value, "shape", ()) or ())
        if not shape:
            return bool(value.item() != 0)
        if math.prod(shape) == 0:
            return False
        return bool(mx.any(value != 0).item())
    except Exception:
        return True


def _patch_config_for_mtp_contract(destination: Path, contract: dict[str, Any]) -> None:
    config_path = destination / "config.json"
    if not config_path.exists():
        return
    config = _load_json(config_path)
    config["mtplx_mtp_contract"] = _normalize_mtp_contract_for_config(contract, config)
    atomic_write_json(config_path, config)


def _calibrate_mtp_contract(
    model_path: Path,
    run: Path,
    *,
    recipe: dict[str, Any],
    existing: dict[str, Any] | None,
    max_fans: bool = False,
) -> dict[str, Any]:
    fallback = _runtime_or_default_mtp_contract(existing)
    allow_uncalibrated = bool(
        recipe.get("allow_uncalibrated_mtp_contract")
        or recipe.get("allow_uncalibrated_contract")
    )
    if recipe.get("calibrate_contract") is False:
        _write_progress(
            run,
            "calibrate",
            progress=1.0,
            label="contract_calibration_skipped",
            finished=True,
            mtp_contract=fallback,
        )
        return fallback

    prompts_path = _contract_calibration_prompts_path(
        recipe.get("contract_calibration_prompts")
    )
    _write_progress(
        run,
        "calibrate",
        progress=0.82,
        label="contract_calibration",
        finished=False,
    )
    output_path = run / "contract_probe.json"
    command = [
        sys.executable,
        "-P",
        "-m",
        "mtplx.cli",
        "mtp-chain-probe",
        "--model",
        str(model_path),
        "--prompts",
        str(prompts_path),
        "--depth",
        str(int(recipe.get("contract_calibration_depth") or 3)),
        "--limit",
        str(int(recipe.get("contract_calibration_limit") or 1)),
        "--max-prompt-tokens",
        str(int(recipe.get("contract_calibration_max_prompt_tokens") or 192)),
        "--windows",
        str(int(recipe.get("contract_calibration_windows") or 1)),
        "--stride",
        str(int(recipe.get("contract_calibration_stride") or 1)),
        "--top-ranks",
        str(recipe.get("contract_calibration_top_ranks") or "1,2,4,8"),
        "--base-hidden-variants",
        str(recipe.get("base_hidden_variants") or "post_norm,pre_norm"),
        "--mtp-hidden-variants",
        str(recipe.get("mtp_hidden_variants") or "post_norm,pre_norm,fc,prev"),
        "--cache-policies",
        str(recipe.get("mtp_cache_policies") or "persistent"),
        "--concat-orders",
        str(recipe.get("concat_orders") or "embedding_hidden,hidden_embedding"),
        "--mtp-position-modes",
        str(recipe.get("mtp_position_modes") or "local,absolute"),
        "--history-modes",
        str(recipe.get("mtp_history_modes") or "recursive"),
        "--anchors",
        str(recipe.get("contract_calibration_anchors") or "prompt_boundary,after_one_target"),
        "--disable-thinking",
        "--output",
        str(output_path),
    ]
    timeout_s = float(recipe.get("contract_calibration_timeout_s") or 900.0)
    stdout_path = run / "contract_probe.stdout.log"
    stderr_path = run / "contract_probe.stderr.log"
    _err("[forge] calibrating MTP contract")
    max_session = _start_max_session_if_requested(max_fans)
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(
                command,
                cwd=str(run),
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired:
        summary = _contract_probe_summary(
            None,
            model_path=model_path,
            diagnostic="contract calibration timed out",
        )
        _write_progress(
            run,
            "calibrate",
            progress=1.0,
            label="contract_calibration_timeout",
            finished=True,
            mtp_contract=fallback,
            **summary,
        )
        _write_build_outcome(
            run,
            _build_outcome_payload(
                model_path,
                rows=[],
                verdict="contract_calibration_failed",
                failure_reasons=["contract_calibration_timeout"],
                diagnostic="contract calibration timed out",
                phase="calibrate",
                architecture_id=summary.get("architecture_id"),
            ),
        )
        if allow_uncalibrated:
            return fallback
        raise ForgeError("MTP contract calibration timed out")
    finally:
        if max_session is not None:
            max_session.stop()
    if proc.returncode != 0 or not output_path.exists():
        detail = _tail(stderr_path, fallback=stdout_path)
        summary = _contract_probe_summary(None, model_path=model_path, diagnostic=detail)
        _write_progress(
            run,
            "calibrate",
            progress=1.0,
            label="contract_calibration_failed",
            finished=True,
            mtp_contract=fallback,
            **summary,
        )
        _write_build_outcome(
            run,
            _build_outcome_payload(
                model_path,
                rows=[],
                verdict="contract_calibration_failed",
                failure_reasons=["contract_calibration_failed"],
                diagnostic=detail,
                phase="calibrate",
                architecture_id=summary.get("architecture_id"),
            ),
        )
        if allow_uncalibrated:
            return fallback
        raise ForgeError(
            "MTP contract calibration failed: " + (detail or "probe failed")
        )
    try:
        payload = _load_json(output_path)
        contract = _contract_from_chain_probe(payload, fallback=fallback)
    except ForgeError as exc:
        payload_for_summary = payload if "payload" in locals() else None
        summary = _contract_probe_summary(
            payload_for_summary,
            model_path=model_path,
            diagnostic=str(exc),
        )
        _write_progress(
            run,
            "calibrate",
            progress=1.0,
            label="contract_calibration_failed",
            finished=True,
            mtp_contract=fallback,
            **summary,
        )
        _write_build_outcome(
            run,
            _build_outcome_payload(
                model_path,
                rows=[],
                verdict="contract_calibration_failed",
                failure_reasons=["contract_calibration_failed"],
                diagnostic=str(exc),
                phase="calibrate",
                architecture_id=summary.get("architecture_id"),
            ),
        )
        if allow_uncalibrated:
            return fallback
        raise
    except Exception as exc:
        payload_for_summary = payload if "payload" in locals() else None
        summary = _contract_probe_summary(
            payload_for_summary,
            model_path=model_path,
            diagnostic=str(exc),
        )
        _write_progress(
            run,
            "calibrate",
            progress=1.0,
            label="contract_calibration_unreadable",
            finished=True,
            mtp_contract=fallback,
            **summary,
        )
        _write_build_outcome(
            run,
            _build_outcome_payload(
                model_path,
                rows=[],
                verdict="contract_calibration_failed",
                failure_reasons=["contract_calibration_unreadable"],
                diagnostic=str(exc),
                phase="calibrate",
                architecture_id=summary.get("architecture_id"),
            ),
        )
        if allow_uncalibrated:
            return fallback
        raise ForgeError(f"MTP contract calibration produced unreadable output: {exc}")
    calibration = contract.get("calibration") if isinstance(contract, dict) else None
    calibration_status = (
        str(calibration.get("status") or "").strip()
        if isinstance(calibration, dict)
        else ""
    )
    calibration_diagnostic = (
        str(calibration.get("diagnostic") or "").strip()
        if isinstance(calibration, dict)
        else ""
    )
    summary = _contract_probe_summary(
        payload,
        model_path=model_path,
        diagnostic=calibration_diagnostic or None,
    )
    label = (
        "contract_calibration"
        if calibration_status in {"", "exact_agreement"}
        else "contract_calibration_diagnostic"
    )
    _write_progress(
        run,
        "calibrate",
        progress=1.0,
        label=label,
        finished=True,
        mtp_contract=contract,
        calibration_status=calibration_status or None,
        **summary,
    )
    return contract


def _contract_calibration_prompts_path(raw: Any) -> Path:
    if raw is not None and str(raw).strip():
        candidate = Path(str(raw)).expanduser()
        if candidate.is_absolute() or candidate.exists():
            return candidate
        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            return cwd_candidate
    return (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "prompts"
        / "calibration_coding.jsonl"
    )


def _normalize_mtp_contract_for_config(
    contract: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    base = MTPContract().with_metadata(contract, preserve_explicit=False)
    return base.with_config_defaults(config).to_dict()


def _load_model_config_for_contract(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {}
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        return _load_json(config_path)
    except Exception:
        return {}


def _runtime_or_default_mtp_contract(
    runtime: dict[str, Any] | None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    config = _load_model_config_for_contract(model_path)
    if isinstance(runtime, dict) and isinstance(runtime.get("mtp_contract"), dict):
        try:
            return _normalize_mtp_contract_for_config(runtime["mtp_contract"], config)
        except Exception:
            pass
    return _normalize_mtp_contract_for_config(None, config)


def _runtime_has_mtp_contract(
    runtime: dict[str, Any] | None,
    model_path: Path | None = None,
) -> bool:
    if not isinstance(runtime, dict) or not isinstance(runtime.get("mtp_contract"), dict):
        return False
    try:
        stored = MTPContract().with_metadata(
            runtime["mtp_contract"],
            preserve_explicit=False,
        ).to_dict()
        normalized = _runtime_or_default_mtp_contract(runtime, model_path)
    except Exception:
        return False
    quant_keys = {
        "mtp_quant_bits",
        "mtp_quant_group_size",
        "mtp_quant_mode",
        "mtp_quant_policy",
        "mtp_prequantized",
    }
    for key in quant_keys:
        if normalized.get(key) != stored.get(key):
            return False
    return True


def _start_max_session_if_requested(enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        from mtplx.thermal import MaxSession
    except Exception as exc:
        raise ForgeError(
            f"verified max-fan mode is unavailable; refusing Forge model load: {exc}"
        ) from exc
    session = MaxSession(log=lambda line: _err(f"[forge] {line}"))
    if not session.start():
        verified = session.thermal.get("verified") or {}
        detail = str(verified.get("message") or "fan ramp did not verify").strip()
        raise ForgeError(
            "verified max-fan mode did not start; refusing Forge model load: "
            + detail
        )
    return session


def _contract_from_chain_probe(
    payload: dict[str, Any],
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    variants = payload.get("variants") if isinstance(payload, dict) else None
    if not isinstance(variants, list) or not variants:
        return fallback
    winner = max(
        (variant for variant in variants if isinstance(variant, dict)),
        key=_chain_probe_contract_score,
        default=None,
    )
    if not winner:
        return fallback
    score = _chain_probe_contract_score(winner)
    agreement = _numeric_list(winner.get("agreement_by_depth"))
    if score <= 0.0:
        return _contract_from_probe_variant(
            winner,
            payload=payload,
            fallback=fallback,
            score=score,
            status="no_agreement_signal",
            diagnostic=(
                "MTP contract calibration found no agreement signal; "
                "the MTP sidecar is probably mismatched to the trunk"
            ),
        )
    if not agreement or sum(agreement) <= 0.0:
        return _contract_from_probe_variant(
            winner,
            payload=payload,
            fallback=fallback,
            score=score,
            status="topk_only_no_exact_agreement",
            diagnostic=(
                "MTP contract calibration found only top-k hints and no exact "
                "agreement signal; the MTP sidecar is probably mismatched to the trunk"
            ),
        )
    return _contract_from_probe_variant(
        winner,
        payload=payload,
        fallback=fallback,
        score=score,
        status="exact_agreement",
        diagnostic=None,
    )


def _contract_from_probe_variant(
    winner: dict[str, Any],
    *,
    payload: dict[str, Any],
    fallback: dict[str, Any],
    score: float,
    status: str,
    diagnostic: str | None,
) -> dict[str, Any]:
    contract = dict(fallback)
    contract.update(
        {
            "base_hidden_variant": str(
                winner.get("base_hidden_variant")
                or fallback.get("base_hidden_variant")
                or "post_norm"
            ),
            "hidden_variant": str(
                winner.get("mtp_hidden_variant")
                or winner.get("hidden_variant")
                or fallback.get("hidden_variant")
                or "post_norm"
            ),
            "concat_order": str(
                winner.get("concat_order")
                or fallback.get("concat_order")
                or "embedding_hidden"
            ),
            "mtp_position_mode": str(
                winner.get("mtp_position_mode")
                or fallback.get("mtp_position_mode")
                or "cache"
            ),
            "calibration": {
                "probe": "mtp-chain-probe",
                "status": status,
                "score": score,
                "diagnostic": diagnostic,
                "agreement_by_depth": winner.get("agreement_by_depth"),
                "topk_rates_by_depth": winner.get("topk_rates_by_depth"),
                "cache_policy": winner.get("cache_policy"),
                "history_mode": winner.get("history_mode"),
                "anchor": winner.get("anchor"),
                "mean_prefix": winner.get("mean_prefix"),
            },
        }
    )
    for key in ("mtp_quant_bits", "mtp_quant_group_size", "mtp_quant_mode", "mtp_quant_policy"):
        if payload.get(key) is not None:
            contract[key] = payload[key]
    return MTPContract().with_metadata(contract, preserve_explicit=False).to_dict() | {
        "calibration": contract["calibration"]
    }


def _contract_probe_summary(
    payload: dict[str, Any] | None,
    *,
    model_path: Path,
    diagnostic: str | None = None,
) -> dict[str, Any]:
    variants = payload.get("variants") if isinstance(payload, dict) else None
    clean = [variant for variant in variants or [] if isinstance(variant, dict)]
    best_agreement = 0.0
    best_topk = 0.0
    for variant in clean:
        agreement = _numeric_list(variant.get("agreement_by_depth"))
        if agreement:
            best_agreement = max(best_agreement, max(agreement))
        topk = variant.get("topk_rates_by_depth")
        if isinstance(topk, dict):
            for value in topk.values():
                rates = _numeric_list(value)
                if rates:
                    best_topk = max(best_topk, max(rates))
    return {
        "architecture_id": _architecture_id_for_model_path(model_path),
        "contract_candidates_tested": len(clean),
        "best_agreement": float(best_agreement),
        "best_topk_hint": float(best_topk),
        "diagnostic": diagnostic,
    }


def _chain_probe_contract_score(variant: dict[str, Any]) -> float:
    agreement = _numeric_list(variant.get("agreement_by_depth"))
    topk = variant.get("topk_rates_by_depth")
    top4 = _numeric_list(topk.get("4") if isinstance(topk, dict) else None)
    top8 = _numeric_list(topk.get("8") if isinstance(topk, dict) else None)
    d1 = agreement[0] if agreement else 0.0
    return (4.0 * sum(agreement)) + (1.5 * sum(top4)) + sum(top8) + (2.0 * d1)


def _numeric_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, (int, float))]


def _forge_verify_max_tokens(args: Any) -> int:
    value = getattr(args, "max_tokens", None)
    if value is None:
        return FORGE_VERIFY_DEFAULT_MAX_TOKENS
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return FORGE_VERIFY_DEFAULT_MAX_TOKENS
    return max(1, parsed)


def _forge_verify_prompt_suite(args: Any) -> str:
    value = getattr(args, "suite", None)
    text = str(value or "").strip()
    return text or FORGE_VERIFY_DEFAULT_PROMPT_SUITE


def _run_verify(
    model_path: Path,
    run: Path,
    *,
    max_fans: bool,
    mtp_contract: dict[str, Any] | None = None,
    max_tokens: int = FORGE_VERIFY_DEFAULT_MAX_TOKENS,
    prompt_suite: str = FORGE_VERIFY_DEFAULT_PROMPT_SUITE,
) -> list[dict[str, Any]]:
    tune_root = run / "tune"
    tune_run_id = "forge-verify"
    output_root = tune_root / tune_run_id
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-P",
        "-m",
        "mtplx.cli",
        "tune",
        "--model",
        str(model_path),
        "--json",
        "--retune",
        "--no-save",
        "--output-dir",
        str(tune_root),
        "--run-id",
        tune_run_id,
        "--depths",
        "1,2,3",
        "--max-tokens",
        str(max(1, int(max_tokens))),
        "--prompt-suite",
        str(prompt_suite),
        "--yes",
    ]
    if max_fans:
        command.append("--require-max-fans")
    if isinstance(mtp_contract, dict):
        base_hidden_variant = mtp_contract.get("base_hidden_variant")
        hidden_variant = mtp_contract.get("hidden_variant")
        concat_order = mtp_contract.get("concat_order")
        if base_hidden_variant:
            command.extend(["--base-hidden-variant", str(base_hidden_variant)])
        if hidden_variant:
            command.extend(["--mtp-hidden-variant", str(hidden_variant)])
        if concat_order:
            command.extend(["--concat-order", str(concat_order)])
    _err(
        "[forge] verifying AR and MTP depths with mtplx tune "
        f"(suite={prompt_suite}, max_tokens={max(1, int(max_tokens))})"
    )
    stdout_path = run / "verify.stdout.log"
    stderr_path = run / "verify.stderr.log"
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=str(run),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        while proc.poll() is None:
            _merge_candidate_rows(output_root, rows, seen)
            if rows:
                _write_verify(run, _annotate_verify_rows(rows))
            time.sleep(1.0)
        returncode = proc.returncode
    _merge_candidate_rows(output_root, rows, seen)
    payload = _load_json_from_text_file(stdout_path)
    if isinstance(payload, dict):
        payload_rows = _verify_rows_from_tune_payload(payload)
        if payload_rows:
            rows = payload_rows
    rows = _annotate_verify_rows(rows)
    _write_verify(run, rows)
    if returncode != 0 and not rows:
        raise ForgeError(
            "mtplx tune failed: "
            + _tune_failure_detail(
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_root=output_root,
            )
        )
    return rows


def _merge_candidate_rows(output_root: Path, rows: list[dict[str, Any]], seen: set[int]) -> None:
    for candidate, depth in (("ar", 0), ("d1", 1), ("d2", 2), ("d3", 3)):
        if depth in seen:
            continue
        path = output_root / f"{candidate}.json"
        if not path.exists():
            continue
        row = _row_from_tune_file(path, depth=depth)
        if row is None:
            continue
        rows.append(row)
        seen.add(depth)


def _row_from_tune_file(path: Path, *, depth: int) -> dict[str, Any] | None:
    try:
        data = _load_json(path)
    except Exception:
        return None
    if depth == 0:
        ar_rows = data.get("ar_rows") if isinstance(data, dict) else None
        if not isinstance(ar_rows, list):
            return None
        tok_s = _mean(
            _as_float(row.get("decode_tok_s") or row.get("tok_s"))
            for row in ar_rows
            if isinstance(row, dict)
        )
        verify_time = _sum(
            _as_float(row.get("elapsed_s")) for row in ar_rows if isinstance(row, dict)
        )
        hit_token_budget_count = sum(
            1
            for row in ar_rows
            if isinstance(row, dict) and row.get("hit_token_budget")
        )
        quality = summarize_benchmark_quality(
            [row for row in ar_rows if isinstance(row, dict)]
        )
        if tok_s is None:
            return None
        return {
            "depth": 0,
            "tok_s": tok_s,
            "multiplier_vs_ar": 1.0,
            "acceptance_by_position": [],
            "verify_time_s": verify_time or 0.0,
            "hit_token_budget": hit_token_budget_count > 0,
            "hit_token_budget_count": hit_token_budget_count,
            **quality,
        }
    depth_rows = data.get("depths") or data.get("depth_results") or []
    if not depth_rows:
        return None
    summary = depth_rows[0].get("summary") or {}
    tok_s = _as_float(summary.get("mean_decode_tok_s") or summary.get("mean_tok_s"))
    if tok_s is None:
        return None
    generation_rows = [
        result_row
        for depth_row in depth_rows
        for result_row in depth_row.get("rows", [])
        if isinstance(result_row, dict)
    ]
    quality = summarize_benchmark_quality(generation_rows)
    acceptance = summary.get("acceptance_by_depth")
    if acceptance is None:
        acceptance = _rate_lists(
            summary.get("accepted_by_depth") or [],
            summary.get("drafted_by_depth") or [],
        )
    return {
        "depth": depth,
        "tok_s": tok_s,
        "multiplier_vs_ar": None,
        "acceptance_by_position": [
            float(value) for value in acceptance or [] if isinstance(value, (int, float))
        ],
        "verify_time_s": float(summary.get("verify_time_s") or 0.0),
        **quality,
        "hit_token_budget": bool(summary.get("hit_token_budget_count")),
        "hit_token_budget_count": int(summary.get("hit_token_budget_count") or 0),
        "finish_reasons": summary.get("finish_reasons") or {},
    }


def _verify_rows_from_tune_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("results") or payload.get("rows") or []
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        depth = row.get("depth")
        depth_value = 0 if depth is None and row.get("mode") == "AR" else depth
        if not isinstance(depth_value, int):
            continue
        tok_s = _as_float(row.get("tok_s") or row.get("decode_tok_s"))
        if tok_s is None:
            continue
        acceptance = row.get("acceptance_by_depth") or row.get("acceptance_by_position") or []
        rows.append(
            {
                "depth": depth_value,
                "tok_s": tok_s,
                "multiplier_vs_ar": _as_float(row.get("multiplier_vs_ar")),
                "acceptance_by_position": [
                    float(value)
                    for value in acceptance
                    if isinstance(value, (int, float))
                ],
                "verify_time_s": float(row.get("verify_time_s") or row.get("elapsed_s") or 0.0),
                "quality_passed": row.get("quality_passed"),
                "hit_token_budget": row.get("hit_token_budget"),
                "hit_token_budget_count": row.get("hit_token_budget_count"),
                "finish_reasons": row.get("finish_reasons") or {},
            }
        )
    return _annotate_verify_rows(rows)


def _verify_rows_from_runtime(runtime: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(runtime, dict):
        return []
    evidence = runtime.get("speed_evidence")
    if not isinstance(evidence, dict):
        return []
    saved_rows = evidence.get("forge_verify_rows")
    if isinstance(saved_rows, list):
        rows = [
            _verify_row_from_mapping(row)
            for row in saved_rows
            if isinstance(row, dict)
        ]
        rows = [row for row in rows if row is not None]
        if rows:
            return _annotate_verify_rows(rows)
    legacy_rows = _legacy_speed_grid_rows(evidence)
    if legacy_rows:
        return _annotate_verify_rows(legacy_rows)
    ar_tok = None
    greedy = evidence.get("greedy_diagnostic")
    if isinstance(greedy, dict):
        ar_tok = _as_float(greedy.get("tok_s"))
    if ar_tok is None:
        ar_tok = _as_float(evidence.get("ar_tok_s"))
    tok_values = evidence.get("tok_s")
    tok_s = None
    if isinstance(tok_values, list):
        numeric = [_as_float(value) for value in tok_values]
        numeric = [value for value in numeric if value is not None]
        tok_s = max(numeric) if numeric else None
    else:
        tok_s = _as_float(tok_values)
    if tok_s is None:
        tok_s = _as_float(evidence.get("mtp_tok_s"))
    depth = evidence.get("depth")
    if not isinstance(depth, int):
        depth = evidence.get("mtp_depth")
    if not isinstance(depth, int):
        depth = runtime.get("mtp_depth_max") if isinstance(runtime.get("mtp_depth_max"), int) else 0
    rows: list[dict[str, Any]] = []
    if ar_tok is not None:
        rows.append(
            {
                "depth": 0,
                "tok_s": ar_tok,
                "multiplier_vs_ar": 1.0,
                "acceptance_by_position": [],
                "verify_time_s": 0.0,
            }
        )
    if tok_s is not None and depth and depth > 0:
        acceptance = evidence.get("acceptance_by_depth") or []
        rows.append(
            {
                "depth": int(depth),
                "tok_s": tok_s,
                "multiplier_vs_ar": None,
                "acceptance_by_position": [
                    float(value) for value in acceptance if isinstance(value, (int, float))
                ],
                "verify_time_s": 0.0,
            }
        )
    return _annotate_verify_rows(rows)


def _runtime_has_legacy_speed_grid(runtime: dict[str, Any] | None) -> bool:
    if not isinstance(runtime, dict):
        return False
    evidence = runtime.get("speed_evidence")
    if not isinstance(evidence, dict):
        return False
    artifact = evidence.get("artifact")
    if not artifact:
        return False
    return bool(_legacy_speed_grid_rows(evidence))


def _legacy_speed_grid_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    artifact = evidence.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        return []
    path = Path(artifact).expanduser()
    if not path.exists():
        return []
    try:
        payload = _load_json(path)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    rows: list[dict[str, Any]] = []
    ar_rows = payload.get("ar_rows")
    if isinstance(ar_rows, list):
        ar_candidates = [
            row for row in (_verify_row_from_legacy_ar_row(item) for item in ar_rows if isinstance(item, dict))
            if row is not None
        ]
        if ar_candidates:
            rows.append(max(ar_candidates, key=lambda row: float(row.get("tok_s") or 0.0)))

    depth_entries = payload.get("depths")
    if isinstance(depth_entries, list):
        for entry in depth_entries:
            if not isinstance(entry, dict):
                continue
            depth = entry.get("depth")
            if not isinstance(depth, int) or depth <= 0:
                continue
            depth_rows = entry.get("rows")
            if not isinstance(depth_rows, list):
                continue
            candidates = [
                row
                for row in (
                    _verify_row_from_legacy_depth_row(item, depth=depth)
                    for item in depth_rows
                    if isinstance(item, dict)
                )
                if row is not None
            ]
            if candidates:
                rows.append(max(candidates, key=lambda row: float(row.get("tok_s") or 0.0)))
    return rows


def _verify_row_from_legacy_ar_row(row: dict[str, Any]) -> dict[str, Any] | None:
    tok_s = _as_float(row.get("tok_s"))
    if tok_s is None:
        return None
    return {
        "depth": 0,
        "tok_s": tok_s,
        "multiplier_vs_ar": 1.0,
        "acceptance_by_position": [],
        "verify_time_s": float(row.get("elapsed_s") or row.get("verify_time_s") or 0.0),
    }


def _verify_row_from_legacy_depth_row(row: dict[str, Any], *, depth: int) -> dict[str, Any] | None:
    tok_s = _as_float(row.get("tok_s") or row.get("mtp_tok_s"))
    if tok_s is None:
        return None
    acceptance = row.get("acceptance_by_depth") or row.get("acceptance_by_position") or []
    return {
        "depth": int(depth),
        "tok_s": tok_s,
        "multiplier_vs_ar": _as_float(row.get("multiplier_vs_ar")),
        "acceptance_by_position": [
            float(value)
            for value in acceptance
            if isinstance(value, (int, float))
        ],
        "verify_time_s": float(row.get("elapsed_s") or row.get("verify_time_s") or 0.0),
        "quality_passed": row.get("quality_passed"),
        "hit_token_budget": row.get("hit_token_budget"),
        "hit_token_budget_count": row.get("hit_token_budget_count"),
        "finish_reasons": row.get("finish_reasons") or {},
    }


def _verify_row_from_mapping(row: dict[str, Any]) -> dict[str, Any] | None:
    depth = row.get("depth")
    if not isinstance(depth, int):
        return None
    tok_s = _as_float(row.get("tok_s") or row.get("decode_tok_s"))
    if tok_s is None:
        return None
    acceptance = row.get("acceptance_by_position")
    if acceptance is None:
        acceptance = row.get("acceptance_by_depth")
    return {
        "depth": int(depth),
        "tok_s": float(tok_s),
        "multiplier_vs_ar": _as_float(row.get("multiplier_vs_ar")),
        "acceptance_by_position": [
            float(value)
            for value in acceptance or []
            if isinstance(value, (int, float))
        ],
        "verify_time_s": float(row.get("verify_time_s") or row.get("elapsed_s") or 0.0),
        "quality_passed": row.get("quality_passed"),
        "hit_token_budget": row.get("hit_token_budget"),
        "hit_token_budget_count": row.get("hit_token_budget_count"),
        "finish_reasons": row.get("finish_reasons") or {},
    }


def _annotate_verify_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = sorted(rows, key=lambda row: int(row.get("depth") or 0))
    ar = next((row for row in clean if int(row.get("depth") or 0) == 0), None)
    ar_tok = _as_float(ar.get("tok_s")) if ar else None
    annotated: list[dict[str, Any]] = []
    for row in clean:
        item = dict(row)
        tok_s = _as_float(item.get("tok_s"))
        if int(item.get("depth") or 0) == 0:
            item["multiplier_vs_ar"] = 1.0 if tok_s is not None else 0.0
        elif item.get("multiplier_vs_ar") is None:
            item["multiplier_vs_ar"] = _safe_ratio(tok_s, ar_tok) or 0.0
        item["tok_s"] = float(tok_s or 0.0)
        item["depth"] = int(item.get("depth") or 0)
        item.setdefault("acceptance_by_position", [])
        item["verify_time_s"] = float(item.get("verify_time_s") or 0.0)
        annotated.append(item)
    return annotated


def _verify_rows_have_all_depths(rows: list[dict[str, Any]]) -> bool:
    depths = {int(row.get("depth") or 0) for row in rows if isinstance(row, dict)}
    return all(depth in depths for depth in (0, 1, 2, 3))


def _require_verify_rows(
    rows: list[dict[str, Any]],
    *,
    require_all_depths: bool = False,
) -> None:
    depths = {int(row.get("depth") or 0) for row in rows if isinstance(row, dict)}
    required = (0, 1, 2, 3) if require_all_depths else (0,)
    missing = [depth for depth in required if depth not in depths]
    if missing:
        raise ForgeError(
            "verification did not complete required depths: "
            + ", ".join(f"D{depth}" if depth else "AR" for depth in missing)
        )
    if not any(depth > 0 for depth in depths):
        raise ForgeError("verification did not produce any MTP depth rows")


def _require_speed_win_or_write_outcome(
    model_path: Path,
    run: Path,
    rows: list[dict[str, Any]],
    *,
    diagnostic: str | None = None,
) -> dict[str, Any]:
    annotated = _annotate_verify_rows(rows)
    evidence = _speed_evidence(annotated)
    if evidence.get("verdict") == "mtp_depth_wins":
        return evidence
    outcome = _build_outcome_payload(
        model_path,
        rows=annotated,
        speed_evidence=evidence,
        diagnostic=diagnostic,
        phase="verify",
    )
    _write_build_outcome(run, outcome)
    raise ForgeError(_speed_gate_failure_message(outcome))


def _saved_verify_rows_reuse_blocker(
    rows: list[dict[str, Any]],
    runtime: dict[str, Any] | None,
    *,
    model_path: Path,
    source_path: Path | None = None,
    require_all_depths: bool,
) -> str | None:
    if not rows:
        return "missing saved verification rows"
    if require_all_depths and not _verify_rows_have_all_depths(rows):
        return "saved verification is missing AR/D1/D2/D3 rows"
    if not any(int(row.get("depth") or 0) > 0 for row in rows):
        return "saved verification has no MTP depth rows"
    if any(row.get("hit_token_budget") for row in rows):
        return "saved verification hit the token budget"
    if _runtime_evidence_has_launch_blocker(
        runtime.get("exactness_baseline") if isinstance(runtime, dict) else None
    ):
        return "saved exactness baseline is marked as a launch blocker"

    raw_evidence = runtime.get("speed_evidence") if isinstance(runtime, dict) else None
    if isinstance(raw_evidence, dict):
        current_fingerprint = _verification_artifact_fingerprint(model_path)
        saved_fingerprint = raw_evidence.get("artifact_fingerprint")
        if saved_fingerprint:
            if current_fingerprint is None:
                return "current artifact cannot be fingerprinted"
            if saved_fingerprint != current_fingerprint:
                return "saved verification belongs to different artifact bytes"
        elif source_path is not None:
            source_fingerprint = _verification_artifact_fingerprint(source_path)
            if (
                current_fingerprint is not None
                and source_fingerprint is not None
                and current_fingerprint != source_fingerprint
            ):
                return "artifact changed after unbound saved verification"

        raw_verdict = str(raw_evidence.get("verdict") or "").strip()
        if raw_verdict and raw_verdict != "mtp_depth_wins":
            return f"saved speed evidence verdict is {raw_verdict}"
        raw_failure_reasons = raw_evidence.get("failure_reasons")
        if isinstance(raw_failure_reasons, list) and raw_failure_reasons:
            return "saved speed evidence has failure reasons"

    if _verification_predates_forge(runtime):
        return "saved verification predates the forged artifact"

    evidence = _speed_evidence(_annotate_verify_rows(rows))
    if evidence.get("verdict") != "mtp_depth_wins":
        return f"saved verification verdict is {evidence.get('verdict') or 'unknown'}"
    if evidence.get("failure_reasons"):
        return "saved verification has failure reasons"
    return None


def _verification_artifact_fingerprint(model_path: Path) -> str | None:
    """Bind reusable speed rows to the config and MTP payload they measured."""

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = _load_json(config_path)
        mtp_path = artifacts.expected_mtp_file(model_path, config)
    except Exception:
        return None
    if not mtp_path.is_file():
        return None

    digest = hashlib.sha256()
    for label, path in (("config.json", config_path), ("mtp", mtp_path)):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _verification_predates_forge(runtime: dict[str, Any] | None) -> bool:
    if not isinstance(runtime, dict):
        return False
    verified = runtime.get("verified_on")
    provenance = runtime.get("forge_provenance")
    if not isinstance(verified, dict) or not isinstance(provenance, dict):
        return False
    verified_at = str(verified.get("timestamp") or "").strip()
    forged_at = str(provenance.get("forged_at") or "").strip()
    if not verified_at or not forged_at:
        return False
    try:
        verified_time = datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
        forged_time = datetime.fromisoformat(forged_at.replace("Z", "+00:00"))
        return forged_time > verified_time
    except (TypeError, ValueError):
        return False


def _recommended_profile_stamp(model_path: Path, *, best_depth: int) -> str:
    """Profile per-model launch resolution will actually pick for this artifact.

    A hard-coded "sustained" stamp lied for flagship-identified rebuilds
    (drop-day forge-local builds carry first-party branded names): serve
    resolves them to turbo, and a mismatched stamp then hid the artifact's
    profile-scoped runtime contract on the profile it actually runs under
    (``_profile_scoped_model_runtime_contract``). Identity at forge time is
    exactly what serve-time identity will see: the artifact directory —
    its existing mtplx_runtime.json id claim, else its branded dir name —
    mapped through the same ``public_model_id_for_ref`` resolver. Artifacts
    with no MTP depth win keep the "stable" recommendation.
    """

    if best_depth <= 0:
        return "stable"
    from mtplx.commands.public import resolved_default_profile_name_for_ref

    return resolved_default_profile_name_for_ref(model_path)


def _stamp_runtime_metadata(
    model_path: Path,
    *,
    branded_name: str,
    source_repo: str,
    source_sha: str,
    source_format: str,
    recipe: dict[str, Any],
    forge_inputs: dict[str, Any],
    rows: list[dict[str, Any]],
    mtp_contract: dict[str, Any],
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(existing or {})
    if _runtime_evidence_has_launch_blocker(metadata.get("exactness_baseline")):
        metadata["exactness_baseline"] = {}
    config = _load_json(model_path / "config.json") if (model_path / "config.json").exists() else {}
    mtp_contract = _normalize_mtp_contract_for_config(mtp_contract, config)
    inspection = None
    try:
        inspection = inspect_model(model_path)
    except Exception:
        inspection = None

    winner = _winning_row(rows)
    best_depth = int(winner.get("depth") or 0) if winner else 0
    verified_depth_max = max(
        (int(row.get("depth") or 0) for row in rows if int(row.get("depth") or 0) > 0),
        default=0,
    )
    metadata["mtplx_version"] = __version__
    metadata.setdefault("arch_id", _arch_id_from_config(config, inspection))
    metadata["mtp_depth_max"] = max(verified_depth_max, int(metadata.get("mtp_depth_max") or 0))
    metadata.setdefault(
        "recommended_profile",
        _recommended_profile_stamp(model_path, best_depth=best_depth),
    )
    metadata.setdefault("sampler", {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    metadata["verified_on"] = {
        "timestamp": _now_iso(),
        "hardware": platform.platform(),
        "machine_arch": platform.machine(),
        "macos": platform.mac_ver()[0],
        "model": branded_name,
    }
    metadata.setdefault("exactness_baseline", {})
    speed_evidence = _speed_evidence(rows)
    artifact_fingerprint = _verification_artifact_fingerprint(model_path)
    if artifact_fingerprint is not None:
        speed_evidence["artifact_fingerprint"] = artifact_fingerprint
    metadata["speed_evidence"] = speed_evidence
    metadata["mtp_contract"] = dict(mtp_contract)
    if inspection and inspection.mtp is not None:
        metadata["mtp_sidecar"] = inspection.mtp.sidecar_format
    else:
        metadata.setdefault("mtp_sidecar", "mtp.safetensors")
    vision_stamp = _vision_metadata_stamp(model_path)
    if vision_stamp is not None:
        metadata["vision"] = vision_stamp
    metadata["base_trunk"] = source_repo
    metadata.setdefault("artifact_role", "forge-local")
    metadata["forge_provenance"] = {
        "source_repo": source_repo,
        "source_sha": source_sha or None,
        "source_format": source_format if source_format in SUPPORTED_SOURCE_FORMATS else SOURCE_UNKNOWN,
        "forge_recipe": recipe,
        "mtp_contract": dict(mtp_contract),
        "forge_inputs": forge_inputs,
        "forged_at": _now_iso(),
        "mtplx_version": __version__,
        "forged_locally": True,
        "published_to_hf": None,
    }
    return metadata


def _vision_metadata_stamp(model_path: Path) -> dict[str, Any] | None:
    """Provenance for artifacts that carry a vision tower, or None."""
    from mtplx.vision import resolve_vision_prefix

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    try:
        index = _load_json(index_path)
    except Exception:
        return None
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        return None
    prefix = resolve_vision_prefix(weight_map)
    if prefix is None:
        return None
    vision_keys = [key for key in weight_map if str(key).startswith(prefix)]
    shards = sorted({str(weight_map[key]) for key in vision_keys})
    return {
        "tensor_count": len(vision_keys),
        "prefix": prefix,
        "shards": shards,
    }


def _speed_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    winner = _winning_row(rows)
    ar_row = next((row for row in rows if row.get("depth") == 0), None)
    quality_rejected = _quality_rejected_winning_rows(rows)
    acceptance_collapsed = _acceptance_collapsed_rows(rows)
    if winner is None:
        winner = ar_row
    depth = int(winner.get("depth") or 0) if winner else 0
    if depth > 0:
        verdict = "mtp_depth_wins"
    elif quality_rejected:
        verdict = "no_quality_passed_mtp_depth_beat_ar"
    elif acceptance_collapsed:
        verdict = "mtp_acceptance_collapsed"
    else:
        verdict = "no_mtp_depth_beat_ar"
    return {
        "depth": depth,
        "tok_s": [float(winner.get("tok_s") or 0.0)] if winner else [],
        "acceptance_by_depth": list(winner.get("acceptance_by_position") or []) if winner else [],
        "greedy_diagnostic": {
            "tok_s": float(ar_row.get("tok_s") or 0.0) if ar_row else None,
        },
        "forge_verify_rows": rows,
        "quality_rejected": quality_rejected,
        "acceptance_collapsed": acceptance_collapsed,
        "failure_reasons": _speed_failure_reasons(
            rows,
            quality_rejected=quality_rejected,
            acceptance_collapsed=acceptance_collapsed,
            winner_depth=depth,
        ),
        "verdict": verdict,
    }


def _runtime_evidence_has_launch_blocker(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("public_release_blocker") is True:
        return True
    status = str(value.get("status") or "").strip().lower().replace("-", "_").replace(" ", "_")
    return bool(
        status
        and any(
            status == prefix or status.startswith(f"{prefix}_")
            for prefix in RUNTIME_BLOCKER_STATUS_PREFIXES
        )
    )


def _build_outcome_payload(
    model_path: Path,
    *,
    rows: list[dict[str, Any]],
    speed_evidence: dict[str, Any] | None = None,
    verdict: str | None = None,
    failure_reasons: list[str] | None = None,
    diagnostic: str | None = None,
    phase: str = "verify",
    architecture_id: str | None = None,
) -> dict[str, Any]:
    annotated = _annotate_verify_rows(rows) if rows else []
    evidence = speed_evidence or (_speed_evidence(annotated) if annotated else {})
    resolved_verdict = str(verdict or evidence.get("verdict") or "failed")
    reasons = list(failure_reasons or evidence.get("failure_reasons") or [resolved_verdict])
    ar_row = next((row for row in annotated if int(row.get("depth") or 0) == 0), None)
    mtp_rows = [row for row in annotated if int(row.get("depth") or 0) > 0]
    best_mtp = max(mtp_rows, key=lambda row: float(row.get("tok_s") or 0.0), default=None)
    payload: dict[str, Any] = {
        "converted_path": str(model_path),
        "phase": phase,
        "verdict": resolved_verdict,
        "failure_reasons": reasons,
        "message": _build_outcome_message(resolved_verdict, reasons),
        "diagnostic": diagnostic,
        "verify_rows": annotated,
        "speed_evidence": evidence,
        "ar_tok_s": float(ar_row.get("tok_s") or 0.0) if ar_row else None,
        "best_mtp_depth": int(best_mtp.get("depth") or 0) if best_mtp else None,
        "best_mtp_tok_s": float(best_mtp.get("tok_s") or 0.0) if best_mtp else None,
        "best_mtp_multiplier_vs_ar": (
            float(best_mtp.get("multiplier_vs_ar") or 0.0) if best_mtp else None
        ),
    }
    arch = architecture_id or _architecture_id_for_model_path(model_path)
    if arch:
        payload["architecture_id"] = arch
    return payload


def _build_outcome_message(verdict: str, failure_reasons: list[str]) -> str:
    if verdict == "mtp_acceptance_collapsed" or "mtp_acceptance_collapsed" in failure_reasons:
        return "MTP did not accelerate this model; draft acceptance collapsed."
    if verdict == "no_quality_passed_mtp_depth_beat_ar":
        return "MTP measured faster, but the quality gate rejected the fast depth."
    if verdict == "no_mtp_depth_beat_ar":
        return "MTP did not accelerate this model; AR was faster."
    if verdict == "contract_calibration_failed":
        return "MTP contract calibration failed; the sidecar did not agree with the trunk."
    return "Forge could not produce an accelerated MTPLX model."


def _speed_gate_failure_message(outcome: dict[str, Any]) -> str:
    message = str(outcome.get("message") or "MTP did not accelerate this model.")
    ar = outcome.get("ar_tok_s")
    best_depth = outcome.get("best_mtp_depth")
    best_tok = outcome.get("best_mtp_tok_s")
    if isinstance(ar, (int, float)) and isinstance(best_tok, (int, float)) and best_depth:
        return f"{message} AR {float(ar):.2f} tok/s; best MTP D{int(best_depth)} {float(best_tok):.2f} tok/s."
    return message


def _architecture_id_for_model_path(model_path: Path) -> str | None:
    try:
        config = _load_json(model_path / "config.json")
    except Exception:
        config = {}
    inspection = None
    try:
        inspection = inspect_model(model_path)
    except Exception:
        inspection = None
    try:
        return _arch_id_from_config(config, inspection)
    except Exception:
        return None


def _winning_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if int(row.get("depth") or 0) > 0
        and _as_float(row.get("multiplier_vs_ar")) is not None
        and float(row.get("multiplier_vs_ar") or 0.0) > 1.0
        and row.get("quality_passed") is not False
    ]
    if candidates:
        return max(candidates, key=lambda row: float(row.get("tok_s") or 0.0))
    return next((row for row in rows if int(row.get("depth") or 0) == 0), None)


def _quality_rejected_winning_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rejected = [
        row
        for row in rows
        if int(row.get("depth") or 0) > 0
        and _as_float(row.get("multiplier_vs_ar")) is not None
        and float(row.get("multiplier_vs_ar") or 0.0) > 1.0
        and row.get("quality_passed") is False
    ]
    return [
        {
            "depth": row.get("depth"),
            "tok_s": row.get("tok_s"),
            "multiplier_vs_ar": row.get("multiplier_vs_ar"),
            "hit_token_budget": row.get("hit_token_budget"),
            "hit_token_budget_count": row.get("hit_token_budget_count"),
            "finish_reasons": row.get("finish_reasons"),
        }
        for row in rejected
    ]


def _acceptance_collapsed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("depth") or 0) <= 0:
            continue
        acceptance = [
            float(value)
            for value in row.get("acceptance_by_position") or []
            if isinstance(value, (int, float))
        ]
        if not acceptance:
            continue
        if max(acceptance) > ACCEPTANCE_COLLAPSE_THRESHOLD:
            continue
        collapsed.append(
            {
                "depth": row.get("depth"),
                "tok_s": row.get("tok_s"),
                "multiplier_vs_ar": row.get("multiplier_vs_ar"),
                "acceptance_by_position": acceptance,
                "verify_time_s": row.get("verify_time_s"),
            }
        )
    return collapsed


def _speed_failure_reasons(
    rows: list[dict[str, Any]],
    *,
    quality_rejected: list[dict[str, Any]],
    acceptance_collapsed: list[dict[str, Any]],
    winner_depth: int,
) -> list[str]:
    if winner_depth > 0:
        return []
    reasons: list[str] = []
    if acceptance_collapsed:
        reasons.append("mtp_acceptance_collapsed")
    if quality_rejected:
        reasons.append("quality_failed_fast_mtp")
    if any(int(row.get("depth") or 0) > 0 for row in rows):
        reasons.append("no_mtp_depth_beat_ar")
    return reasons


def _cmd_publish(args: Any) -> int:
    if str(args.token) != "stdin":
        raise ForgeError("--token stdin is required; tokens are never accepted on argv", code=2)
    token = sys.stdin.readline().strip()
    if not token:
        raise ForgeError("missing Hugging Face token on stdin", code=2)
    run = _run_dir(args.out, args.run_id)
    local = Path(args.path).expanduser()
    if not local.is_dir():
        raise ForgeError(f"local model path does not exist: {local}")
    total = directory_size_bytes(local)
    readme_path = Path(args.readme_path).expanduser() if getattr(args, "readme_path", None) else None
    if readme_path and readme_path.exists():
        total += readme_path.stat().st_size
    _write_publish(run, bytes_uploaded=0, total_bytes=total, repo=args.repo, finished=False)
    api = _make_hf_api()
    private = str(args.visibility) == "private"
    _err(f"[forge] creating/updating Hugging Face repo {args.repo}")
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )
    _write_publish(
        run,
        bytes_uploaded=0,
        total_bytes=total,
        repo_created=args.repo,
        repo=args.repo,
        finished=False,
    )
    started = time.monotonic()
    _err(f"[forge] uploading {local}")
    upload_result = api.upload_folder(
        folder_path=str(local),
        repo_id=args.repo,
        repo_type="model",
        token=token,
        commit_message="Publish MTPLX forged model",
    )
    revision = _revision_from_upload_result(upload_result)
    if readme_path and readme_path.exists():
        readme_result = api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            token=token,
            commit_message="Update MTPLX Forge README",
        )
        revision = _revision_from_upload_result(readme_result) or revision
    try:
        info = api.model_info(repo_id=args.repo, token=token)
        revision = str(getattr(info, "sha", None) or revision or "")
    except Exception:
        revision = revision or ""
    _update_published_runtime(
        local,
        repo=args.repo,
        revision=revision,
        visibility=args.visibility,
        license_spdx=args.license,
    )
    elapsed = max(0.001, time.monotonic() - started)
    _write_publish(
        run,
        bytes_uploaded=total,
        total_bytes=total,
        mb_per_s=total / elapsed / 1_000_000,
        repo_created=args.repo,
        revision=revision or None,
        repo=args.repo,
        finished=True,
    )
    return 0


def _cmd_cancel(args: Any) -> int:
    marker = _cancel_marker(args.run_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, {"run_id": args.run_id, "cancel_requested_at": _now_iso()})
    _json_out({"run_id": args.run_id, "cancel_requested": True, "marker": str(marker)})
    return 0


def _cancel_marker(run_id: str) -> Path:
    root = Path(os.environ.get("MTPLX_FORGE_CANCEL_DIR", "~/.mtplx/forge-cancel")).expanduser()
    return root / f"{run_id}.json"


def _cancel_requested(run_id: str) -> bool:
    return _cancel_marker(run_id).exists()


def _make_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise ForgeError(f"huggingface_hub is required for Forge: {exc}") from exc
    return HfApi()


def _hf_json(repo_id: str, filename: str) -> dict[str, Any]:
    data, _path, error = artifacts._hf_download_json(repo_id, filename)
    if data is None:
        raise RuntimeError(error or f"{filename} not found")
    return data


def _try_hf_runtime(repo_id: str) -> dict[str, Any] | None:
    try:
        return _hf_json(repo_id, "mtplx_runtime.json")
    except Exception:
        return None


def _default_model_root() -> Path:
    env = os.environ.get("MTPLX_FORGE_MODEL_ROOT") or os.environ.get("MTPLX_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    research_models = Path("~/Documents/MTPLX/models").expanduser()
    if research_models.exists():
        return research_models
    return Path("~/.mtplx/models").expanduser()


def _unique_model_dir(branded_name: str) -> Path:
    """Return an unused artifact path without creating the final directory.

    `mlx_lm.convert` refuses an output path that already exists, while local
    mirror paths are happy to create their destination on first copy.
    """

    root = _default_model_root()
    root.mkdir(parents=True, exist_ok=True)
    base = root / branded_name
    if not base.exists():
        return base
    for index in range(1, 10_000):
        candidate = root / f"{branded_name}-{index}"
        if not candidate.exists():
            return candidate
    raise ForgeError(f"could not allocate model directory for {branded_name}")


def _sanitize_branded_name(value: str) -> str:
    clean = "".join(
        ch if ch.isalnum() or ch in {".", "_", "-"} else "-"
        for ch in str(value).strip()
    )
    return clean.strip(".-_")


def _mirror_model_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ForgeError(f"source must be a model directory: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    for child in source.iterdir():
        target = destination / child.name
        if child.name == "mtplx_runtime.json":
            continue
        if child.name == "config.json" and child.is_file():
            _copy_file(child, target)
            continue
        _mirror_entry(child, target)


def _mirror_entry(source: Path, target: Path) -> None:
    if source.is_dir() and not source.is_symlink():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _mirror_entry(child, target / child.name)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(source.resolve(strict=False), target)
    except OSError:
        _copy_file(source, target)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    shutil.copy2(source, target)


def _read_runtime(path: Path) -> dict[str, Any] | None:
    runtime_path = path / "mtplx_runtime.json"
    if not runtime_path.exists():
        return None
    try:
        return _load_json(runtime_path)
    except Exception:
        return None


def _update_published_runtime(
    local: Path,
    *,
    repo: str,
    revision: str,
    visibility: str,
    license_spdx: str,
) -> None:
    runtime_path = local / "mtplx_runtime.json"
    if not runtime_path.exists():
        return
    runtime = _load_json(runtime_path)
    provenance = runtime.get("forge_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["published_to_hf"] = {
        "repo": repo,
        "revision": revision or None,
        "visibility": visibility,
        "license_spdx": license_spdx,
        "uploaded_at": _now_iso(),
    }
    runtime["forge_provenance"] = provenance
    atomic_write_json(runtime_path, runtime)


def _revision_from_upload_result(result: Any) -> str | None:
    for attr in ("oid", "commit_id", "sha", "commit_hash"):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    if isinstance(result, str):
        return result.rsplit("/", 1)[-1] if "/" in result else result
    return None


def _arch_id_from_config(config: dict[str, Any], inspection: Any) -> str:
    if inspection is not None:
        compat = getattr(inspection, "compatibility", {}) or {}
        arch = compat.get("arch_id")
        if arch:
            return str(arch)
    tcfg = text_config(config)
    return str(tcfg.get("model_type") or config.get("model_type") or "unknown-mtp")


def _runtime_multiplier(runtime: dict[str, Any]) -> float | None:
    evidence = runtime.get("speed_evidence")
    if not isinstance(evidence, dict):
        return None
    rows = evidence.get("forge_verify_rows")
    if isinstance(rows, list):
        best = _winning_row([row for row in rows if isinstance(row, dict)])
        if best:
            return _as_float(best.get("multiplier_vs_ar"))
    greedy = evidence.get("greedy_diagnostic")
    ar = _as_float(greedy.get("tok_s")) if isinstance(greedy, dict) else None
    tok = evidence.get("tok_s")
    if isinstance(tok, list):
        numeric = [_as_float(value) for value in tok]
        numeric = [value for value in numeric if value is not None]
        if numeric:
            return _safe_ratio(max(numeric), ar)
    return _safe_ratio(_as_float(tok), ar)


def _load_json_from_text_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _tune_failure_detail(*, stdout_path: Path, stderr_path: Path, output_root: Path) -> str:
    payload = _load_json_from_text_file(stdout_path)
    if not isinstance(payload, dict):
        return _tail(stderr_path, fallback=stdout_path)
    results = payload.get("results")
    if not isinstance(results, list):
        return _tail(stderr_path, fallback=stdout_path)

    details: list[str] = []
    seen: set[str] = set()
    for row in results:
        if not isinstance(row, dict):
            continue
        candidate = str(row.get("candidate") or row.get("mode") or "candidate")
        error = str(row.get("error") or "failed").strip()
        log_path = row.get("stdout")
        line = ""
        if isinstance(log_path, str):
            path = Path(log_path)
            if not path.is_absolute():
                path = output_root / path
            line = _last_error_line(path)
        detail = f"{candidate}: {error}"
        if line:
            detail += f" ({line})"
        if detail in seen:
            continue
        seen.add(detail)
        details.append(detail)
        if len(details) >= 4:
            break
    if details:
        return "\n".join(details)
    return _tail(stderr_path, fallback=stdout_path)


def _last_error_line(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("ValueError:", "RuntimeError:", "TypeError:", "ImportError:")):
            return stripped
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith(("File ", "Traceback ")):
            return stripped
    return ""


def _tail(path: Path, *, fallback: Path | None = None, limit: int = 4000) -> str:
    for candidate in (path, fallback):
        if candidate is None or not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text[-limit:]
    return "no log output"


def _estimated_peak_gib(size: int | None) -> float | None:
    if not isinstance(size, int) or size <= 0:
        return None
    return round((size * 1.35) / (1024**3), 2)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _sum(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    if not clean:
        return None
    return sum(clean)


def _rate_lists(accepted: Iterable[Any], drafted: Iterable[Any]) -> list[float | None]:
    rates: list[float | None] = []
    for acc, draft in zip(accepted, drafted, strict=False):
        acc_value = _as_float(acc)
        draft_value = _as_float(draft)
        if acc_value is None or draft_value is None or draft_value <= 0:
            rates.append(None)
        else:
            rates.append(acc_value / draft_value)
    return rates
