"""Hardware-aware verified default model selection for product CLI paths."""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.hardware import classify_apple_silicon_generation, detect_apple_silicon
from mtplx.profiles import (
    DEFAULT_FP16_PUBLIC_MODEL_ID,
    DEFAULT_FP16_HF_MODEL_ID,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_PUBLIC_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
    QUALITY_HF_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID,
)


DEFAULT_MODEL_VARIANT_ENV = "MTPLX_DEFAULT_MODEL_VARIANT"
SPEED_MODEL_ENV = "MTPLX_OPTIMIZED_SPEED_MODEL"
QUALITY_MODEL_ENV = "MTPLX_OPTIMIZED_QUALITY_MODEL"
DEFAULT_MODEL_VARIANTS = frozenset({"auto", "speed", "q4", "bf16", "fp16"})
_LEGACY_APPLE_FP16_GENERATIONS = frozenset({"m1", "m2"})
_NEWER_APPLE_SPEED_GENERATIONS = frozenset({"m3", "m4", "m5"})
OPTIMIZED_SPEED_LABEL = "Qwen3.6 27B MTPLX Optimized Speed"
OPTIMIZED_SPEED_DESCRIPTION = "Q4 target with Q4 MTP sidecar"
OPTIMIZED_QUALITY_LABEL = "Qwen3.6 27B MTPLX Optimized Quality"
OPTIMIZED_QUALITY_DESCRIPTION = "Flat8 target with INT8 MTP sidecar"
_OPTIMIZED_SPEED_LOCAL_CANDIDATES = (
    "~/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
    "~/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized-Speed",
    "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
    "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Speed",
)
_OPTIMIZED_QUALITY_LOCAL_CANDIDATES = (
    "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Quality",
    "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality",
)
_VERIFIED_DEFAULT_LOCAL_NAMES = frozenset(
    {
        "Qwen3.6-27B-MTPLX-Optimized-Speed",
        "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
        "Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
        "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
    }
)


@dataclass(frozen=True)
class DefaultModelSelection:
    model: str
    hf_model: str
    variant: str
    precision: str
    chip_generation: str
    chip: str
    reason: str
    auto_selected: bool
    env_override: str | None = None

    @property
    def display_name(self) -> str:
        if self.variant == "fp16":
            return "Qwen3.6 27B Optimized Speed FP16"
        return OPTIMIZED_SPEED_LABEL

    @property
    def label(self) -> str:
        return f"{self.hf_model}  ·  {self.precision}  ·  {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hf_model": self.hf_model,
            "variant": self.variant,
            "precision": self.precision,
            "chip_generation": self.chip_generation,
            "chip": self.chip,
            "reason": self.reason,
            "auto_selected": self.auto_selected,
            "env_override": self.env_override,
            "display_name": self.display_name,
            "label": self.label,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_complete_local_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (
        (path / "mtplx_pair.json").is_file()
        and (path / "target").is_dir()
        and (path / "assistant").is_dir()
    ):
        return True
    if not (path / "config.json").is_file():
        return False
    has_weights = any(path.glob("model-*.safetensors")) or (path / "model.safetensors").is_file()
    has_mtp = (path / "mtp.safetensors").is_file()
    return has_weights and has_mtp


def _env_ref_disabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "none", "off", "disabled"}


def _complete_local_model_ref(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if not candidate or _env_ref_disabled(candidate):
            continue
        path = Path(candidate).expanduser()
        if _is_complete_local_model(path):
            return str(path)
    return None


def optimized_speed_model_ref() -> str:
    env_ref = str(os.environ.get(SPEED_MODEL_ENV) or "").strip()
    candidates: tuple[str, ...]
    if env_ref:
        if _env_ref_disabled(env_ref):
            return DEFAULT_HF_MODEL_ID
        else:
            candidates = (env_ref, *_OPTIMIZED_SPEED_LOCAL_CANDIDATES)
    else:
        candidates = _OPTIMIZED_SPEED_LOCAL_CANDIDATES
    repo_local = str((_repo_root() / DEFAULT_RUNTIME_MODEL_DIR).resolve())
    local = _complete_local_model_ref((*candidates, repo_local))
    return local or DEFAULT_HF_MODEL_ID


def optimized_quality_model_ref() -> str:
    env_ref = str(os.environ.get(QUALITY_MODEL_ENV) or "").strip()
    if env_ref:
        if _env_ref_disabled(env_ref):
            candidates = ()
        else:
            candidates = (env_ref, *_OPTIMIZED_QUALITY_LOCAL_CANDIDATES)
    else:
        candidates = _OPTIMIZED_QUALITY_LOCAL_CANDIDATES
    local = _complete_local_model_ref(candidates)
    return local or QUALITY_HF_MODEL_ID


def is_optimized_quality_model_ref(model: str | Path | None) -> bool:
    if model is None:
        return False
    text = str(model).strip()
    if not text:
        return False
    if text == QUALITY_HF_MODEL_ID:
        return True
    return "qwen3.6-27b-mtplx-optimized-quality" in text.lower()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _artifact_role_model_id(role: str) -> str | None:
    normalized = role.strip().lower().replace("_", "-")
    if not normalized:
        return None
    if "quality" in normalized:
        return QUALITY_PUBLIC_MODEL_ID
    if "fp16" in normalized or "float16" in normalized:
        return DEFAULT_FP16_PUBLIC_MODEL_ID
    if "gdn8" in normalized or normalized in {"optimized", "legacy-optimized"}:
        return LEGACY_OPTIMIZED_PUBLIC_MODEL_ID
    if "speed" in normalized or "flat4" in normalized or "maximum-speed" in normalized:
        return DEFAULT_PUBLIC_MODEL_ID
    return None


def _public_model_id_from_metadata(path: Path) -> str | None:
    runtime = _read_json(path / "mtplx_runtime.json")
    for key in ("public_model_id", "served_model_id", "model_id"):
        value = runtime.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_public_model_id(value)
    precision = runtime.get("precision_variant")
    if isinstance(precision, str) and precision.strip().lower() in {"fp16", "float16"}:
        return DEFAULT_FP16_PUBLIC_MODEL_ID
    role = runtime.get("artifact_role")
    if isinstance(role, str):
        inferred = _artifact_role_model_id(role)
        if inferred:
            return inferred
    verified_on = runtime.get("verified_on")
    if isinstance(verified_on, dict):
        verified_model = verified_on.get("model")
        if isinstance(verified_model, str):
            inferred = _artifact_role_model_id(verified_model)
            if inferred:
                return inferred

    # The config.json quantization layout (Q4 weights with a Q8 head, flat Q8,
    # etc.) is shared by many MLX builds, so it can only distinguish the
    # Qwen3.6-27B MTPLX Speed/Quality split — not the model family. Skip this
    # refinement for folders that don't already identify as a Qwen3.6-27B MTPLX
    # artifact; otherwise a third-party model like
    # `Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed` would silently be served as
    # `mtplx-qwen36-27b-optimized-quality`.
    if _public_model_id_from_name(str(path)) is None:
        return None

    config = _read_json(path / "config.json")
    quantization = config.get("quantization") or config.get("quantization_config")
    if isinstance(quantization, dict):
        bits = quantization.get("bits")
        if bits == 4:
            child_bits = [
                value.get("bits")
                for value in quantization.values()
                if isinstance(value, dict) and "bits" in value
            ]
            if child_bits and all(bit == 8 for bit in child_bits):
                return QUALITY_PUBLIC_MODEL_ID
            return DEFAULT_PUBLIC_MODEL_ID
        if bits == 8:
            return QUALITY_PUBLIC_MODEL_ID
    return None


def _sanitize_public_model_id(value: str) -> str:
    lowered = str(value).strip().lower()
    lowered = lowered.replace("_", "-")
    lowered = re.sub(r"[^a-z0-9.-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-.")
    return lowered or DEFAULT_PUBLIC_MODEL_ID


def _public_model_id_from_name(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    lowered = text.replace("\\", "/").lower()
    if "qwen3.6-27b-mtplx-optimized-quality" in lowered:
        return QUALITY_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-speed-fp16" in lowered:
        return DEFAULT_FP16_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-speed" in lowered:
        return DEFAULT_PUBLIC_MODEL_ID
    legacy_names = {
        "qwen3.6-27b-mtplx-optimized",
        "youssofal--qwen3.6-27b-mtplx-optimized",
    }
    basename = Path(text).name.lower()
    if basename in legacy_names or lowered.endswith("/qwen3.6-27b-mtplx-optimized"):
        return LEGACY_OPTIMIZED_PUBLIC_MODEL_ID
    return None


def public_model_id_for_ref(
    model: str | Path | None,
    *,
    default_model_id: str = DEFAULT_PUBLIC_MODEL_ID,
) -> str:
    """Return the served OpenAI model id for the selected artifact.

    The default public id is only used when no model was provided. Once a
    concrete repo/path exists, MTPLX should report that artifact instead of
    silently claiming the speed default.
    """

    if model is None:
        return default_model_id
    text = str(model).strip()
    if not text:
        return default_model_id
    path = Path(text).expanduser()
    if path.is_dir():
        inferred = _public_model_id_from_metadata(path)
        if inferred:
            return inferred
    inferred = _public_model_id_from_name(text)
    if inferred:
        return inferred
    basename = Path(text).name or text.split("/")[-1]
    return _sanitize_public_model_id(basename)


def _normalize_variant(value: str | None) -> tuple[str, str | None]:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto"}:
        return "auto", None
    aliases = {
        "speed": "speed",
        "optimized-speed": "speed",
        "q4": "speed",
        "int4": "speed",
        "bf16": "speed",
        "bfloat16": "speed",
        "bfloat": "speed",
        "fp16": "fp16",
        "float16": "fp16",
        "f16": "fp16",
    }
    normalized = aliases.get(raw)
    if normalized is None:
        return "auto", raw
    return normalized, raw


def _hardware_generation(hardware: Mapping[str, Any]) -> str:
    generation = str(hardware.get("apple_silicon_generation") or "").strip().lower()
    if generation:
        return generation
    return classify_apple_silicon_generation(
        str(hardware.get("chip") or ""),
        system=str(hardware.get("system") or ""),
        machine=str(hardware.get("machine") or ""),
    )


def select_default_model(
    *,
    variant_override: str | None = None,
    hardware: Mapping[str, Any] | None = None,
) -> DefaultModelSelection:
    """Select the verified default model for this machine.

    Auto policy is intentionally simple and visible:
    M1/M2 -> FP16, M3/M4/M5/unknown -> quantized Optimized Speed.
    """

    env_value = variant_override if variant_override is not None else os.environ.get(DEFAULT_MODEL_VARIANT_ENV)
    requested_variant, invalid_override = _normalize_variant(env_value)
    hardware_info = dict(detect_apple_silicon() if hardware is None else hardware)
    generation = _hardware_generation(hardware_info)
    chip = str(hardware_info.get("chip") or "").strip()

    if requested_variant == "fp16":
        variant = "fp16"
        reason = f"forced by {DEFAULT_MODEL_VARIANT_ENV}=fp16"
        auto_selected = False
    elif requested_variant == "speed":
        variant = "speed"
        if str(env_value or "").strip().lower() in {"bf16", "bfloat16", "bfloat"}:
            reason = f"forced by {DEFAULT_MODEL_VARIANT_ENV}={env_value} (legacy alias for optimized speed)"
        else:
            reason = f"forced by {DEFAULT_MODEL_VARIANT_ENV}=speed"
        auto_selected = False
    elif generation in _LEGACY_APPLE_FP16_GENERATIONS:
        variant = "fp16"
        reason = "selected for M1/M2 Apple Silicon"
        auto_selected = True
    else:
        variant = "speed"
        auto_selected = True
        if generation in _NEWER_APPLE_SPEED_GENERATIONS:
            reason = "selected for newer Apple Silicon"
        elif generation == "intel":
            reason = "selected because this is not Apple Silicon"
        else:
            reason = "selected because hardware is unknown"

    if invalid_override is not None and requested_variant == "auto":
        reason = f"{reason}; ignored invalid {DEFAULT_MODEL_VARIANT_ENV}={invalid_override}"

    if variant == "fp16":
        model = DEFAULT_FP16_HF_MODEL_ID
        hf_model = DEFAULT_FP16_HF_MODEL_ID
        precision = "FP16"
    else:
        model = optimized_speed_model_ref()
        hf_model = DEFAULT_HF_MODEL_ID
        precision = OPTIMIZED_SPEED_DESCRIPTION
        if model != hf_model:
            reason = f"{reason}; installed locally"
    return DefaultModelSelection(
        model=model,
        hf_model=hf_model,
        variant=variant,
        precision=precision,
        chip_generation=generation,
        chip=chip,
        reason=reason,
        auto_selected=auto_selected,
        env_override=env_value if env_value else None,
    )


def verified_default_refs() -> set[str]:
    root = _repo_root()
    local_speed = optimized_speed_model_ref()
    refs = {
        DEFAULT_HF_MODEL_ID,
        DEFAULT_FP16_HF_MODEL_ID,
        DEFAULT_MODEL_ID,
        local_speed,
        str(DEFAULT_RUNTIME_MODEL_DIR),
        str((root / DEFAULT_RUNTIME_MODEL_DIR).resolve()),
    }
    return {ref for ref in refs if ref}


def is_verified_default_model_ref(model: str | Path | None) -> bool:
    if model is None:
        return True
    text = str(model).strip()
    if not text:
        return True
    refs = verified_default_refs()
    if text in refs:
        return True
    if text.startswith(("~", "/", "./", "../")):
        path = Path(text).expanduser()
        if path.name in _VERIFIED_DEFAULT_LOCAL_NAMES:
            return True
        try:
            expanded = str(path.resolve())
        except OSError:
            expanded = str(path)
        return expanded in refs
    return False
