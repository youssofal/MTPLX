"""Model artifact inspection for Qwen3.6 MTP gates."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    EXPECTED_ALL_PREQUANTIZED_MTP_KEYS,
    EXPECTED_MTP_KEYS,
    EXPECTED_MTP_TENSOR_COUNT,
    EXPECTED_PREQUANTIZED_MTP_KEYS,
    EXPECTED_QWEN_MOE_MTP_KEYS,
    EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS,
    EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS,
    MULTIMODAL_SIDECARS,
    expand_mtp_layer_keys,
)
from .models.laguna_config import (
    LAGUNA_S_2_1_REPO_ID,
    LAGUNA_S_2_1_REQUIRED_FILES,
    LAGUNA_S_2_1_REVISION,
    LAGUNA_S_2_1_WEIGHT_SHARDS,
    is_laguna_s_2_1_mlx_4bit_config,
    laguna_s_2_1_artifact_integrity_errors,
)
from .profiles import (
    DEFAULT_FP16_HF_MODEL_ID,
    DEFAULT_FP16_PUBLIC_MODEL_ID,
    LEGACY_OPTIMIZED_HF_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
    OPTIMIZED_SPEED_V1_HF_MODEL_ID,
    OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID,
    OPTIMIZED_SPEED_V2_HF_MODEL_ID,
    OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID,
    QUALITY_FP16_HF_MODEL_ID,
    QUALITY_FP16_PUBLIC_MODEL_ID,
    QUALITY_HF_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
)

MTP_KEY_PREFIXES = ("mtp.", "language_model.mtp.")
_KNOWN_PUBLIC_MODEL_ALIASES = {
    # Served public ids (the exact strings /v1/models advertises) resolve to
    # their first-party repos. Explicit ids only — consistent with the July
    # 2026 contract-match-only identity stance (#57): pasting the id the
    # server displayed into `mtplx serve/run/pull --model` must work.
    OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID: OPTIMIZED_SPEED_V2_HF_MODEL_ID,
    OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID: OPTIMIZED_SPEED_V1_HF_MODEL_ID,
    DEFAULT_FP16_PUBLIC_MODEL_ID: DEFAULT_FP16_HF_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID: QUALITY_HF_MODEL_ID,
    QUALITY_FP16_PUBLIC_MODEL_ID: QUALITY_FP16_HF_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID: LEGACY_OPTIMIZED_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID: QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID: QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID: QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID: QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID: QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID: QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
    # Artifact-basename aliases (folder-name style).
    "qwen3.5-9b-mtplx-optimized-speed": QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
    "qwen3.5-9b-mtplx-optimized-speed-fp16": QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized-speed-v2": OPTIMIZED_SPEED_V2_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized-speed": OPTIMIZED_SPEED_V1_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized": LEGACY_OPTIMIZED_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized-speed-fp16": DEFAULT_FP16_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized-quality": QUALITY_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized-quality-fp16": QUALITY_FP16_HF_MODEL_ID,
    "qwen3.6-35b-a3b-mtplx-optimized-speed": QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
    "qwen3.6-35b-a3b-mtplx-optimized-speed-fp16": QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    "qwen3.6-35b-a3b-mtplx-optimized-balance": QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
    "qwen3.6-35b-a3b-mtplx-optimized-balance-fp16": QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
}


def normalize_mtp_key(key: str) -> str:
    text = str(key)
    for prefix in MTP_KEY_PREFIXES:
        if text.startswith(prefix):
            return "mtp." + text[len(prefix) :]
    return text


def is_mtp_key(key: str) -> bool:
    text = str(key)
    return any(text.startswith(prefix) for prefix in MTP_KEY_PREFIXES)


def _num_mtp_layers(config: dict[str, Any]) -> int:
    tcfg = text_config(config)
    return int(
        tcfg.get("mtp_num_hidden_layers")
        or tcfg.get("num_nextn_predict_layers")
        or config.get("num_nextn_predict_layers")
        or 0
    )


def _qwen_moe_numbered_expert_keys(
    config: dict[str, Any],
    *,
    prequantized: bool,
) -> tuple[set[str], int, str]:
    tcfg = text_config(config)
    num_experts = int(tcfg.get("num_experts") or 0)
    n_layers = max(_num_mtp_layers(config), 1)
    keys: set[str] = {
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    for layer_index in range(n_layers):
        base = f"mtp.layers.{layer_index}"
        keys.update(
            {
                f"{base}.input_layernorm.weight",
                f"{base}.post_attention_layernorm.weight",
                f"{base}.self_attn.q_proj.weight",
                f"{base}.self_attn.k_proj.weight",
                f"{base}.self_attn.v_proj.weight",
                f"{base}.self_attn.o_proj.weight",
                f"{base}.self_attn.q_norm.weight",
                f"{base}.self_attn.k_norm.weight",
                f"{base}.mlp.gate.weight",
                f"{base}.mlp.shared_expert.gate_proj.weight",
                f"{base}.mlp.shared_expert.up_proj.weight",
                f"{base}.mlp.shared_expert.down_proj.weight",
                f"{base}.mlp.shared_expert_gate.weight",
            }
        )
        for expert_index in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"{base}.mlp.experts.{expert_index}.{proj}"
                keys.add(f"{prefix}.weight")
                if prequantized:
                    keys.add(f"{prefix}.scales")
                    keys.add(f"{prefix}.biases")
    sidecar_format = "prequantized-mlx-affine-qwen-moe-experts" if prequantized else "bf16-qwen-moe-experts"
    return keys, len(keys), sidecar_format


def _has_numbered_moe_experts(keys: set[str]) -> bool:
    marker = ".mlp.experts."
    return any(
        marker in key
        and len(parts := key.split(marker, 1)[1].split(".", 1)) == 2
        and parts[0].isdigit()
        for key in keys
    )


def _expected_prequantized_keys_for_present_aux(
    base_keys: set[str],
    normalized_keys: set[str],
) -> set[str]:
    """Require complete quant triples only for modules that carry aux leaves."""
    aux_prefixes = {
        key.rsplit(".", 1)[0]
        for key in normalized_keys
        if key.endswith(".scales") or key.endswith(".biases")
    }
    expected = set(base_keys)
    for prefix in aux_prefixes:
        if f"{prefix}.weight" in base_keys:
            expected.add(f"{prefix}.scales")
            expected.add(f"{prefix}.biases")
    return expected


def _mtp_expected_key_set(
    config: dict[str, Any],
    *,
    keys: tuple[str, ...] = (),
) -> tuple[set[str], int, str]:
    mtp_quant = config.get("mtplx_mtp_quantization", {})
    prequantized = isinstance(mtp_quant, dict) and bool(mtp_quant.get("prequantized"))
    quant_policy = str(mtp_quant.get("policy") or "") if isinstance(mtp_quant, dict) else ""
    normalized = {normalize_mtp_key(key) for key in keys}
    # Every named key set below is the canonical depth-1 template; checkpoints
    # declaring mtp_num_hidden_layers > 1 replicate the layer keys per index.
    n_layers = max(_num_mtp_layers(config), 1)

    def _expanded(base: tuple[str, ...]) -> set[str]:
        return expand_mtp_layer_keys(base, n_layers)

    if _is_qwen_moe_mtp_layout(config, normalized):
        if any(".mlp.switch_mlp." in key for key in normalized):
            has_prequantized_aux = any(
                key.endswith(".scales") or key.endswith(".biases")
                for key in normalized
            )
            if prequantized or has_prequantized_aux:
                expected = _expected_prequantized_keys_for_present_aux(
                    _expanded(EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS),
                    normalized,
                )
                return (
                    expected,
                    len(expected),
                    "prequantized-mlx-affine-qwen-moe-switch-mlx",
                )
            expected = _expanded(EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS)
            return (
                expected,
                len(expected),
                "bf16-qwen-moe-switch-mlx",
            )
        if _has_numbered_moe_experts(normalized):
            has_prequantized_aux = any(
                key.endswith(".scales") or key.endswith(".biases")
                for key in normalized
            )
            return _qwen_moe_numbered_expert_keys(
                config,
                prequantized=prequantized or has_prequantized_aux,
            )
        if prequantized or normalized == _expanded(EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS):
            expected = _expanded(EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS)
            return (
                expected,
                len(expected),
                "prequantized-mlx-affine-qwen-moe",
            )
        expected = _expanded(EXPECTED_QWEN_MOE_MTP_KEYS)
        return (
            expected,
            len(expected),
            "bf16-qwen-moe",
        )
    if prequantized and quant_policy == "all":
        expected = _expanded(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS)
        return (
            expected,
            len(expected),
            "prequantized-mlx-affine",
        )
    if prequantized:
        expected = _expanded(EXPECTED_PREQUANTIZED_MTP_KEYS)
        return (
            expected,
            len(expected),
            "prequantized-mlx-affine",
        )
    if normalized == _expanded(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS):
        return (
            set(normalized),
            len(normalized),
            "prequantized-mlx-affine",
        )
    if normalized == _expanded(EXPECTED_PREQUANTIZED_MTP_KEYS):
        return (
            set(normalized),
            len(normalized),
            "prequantized-mlx-affine",
        )
    expected = _expanded(EXPECTED_MTP_KEYS)
    return expected, len(expected), "bf16"


def _observed_sidecar_format(sidecar_format: str, tensors: tuple[TensorInfo, ...]) -> str:
    if sidecar_format != "bf16" or not tensors:
        return sidecar_format
    dtypes = {tensor.dtype.upper() for tensor in tensors}
    if dtypes == {"F16"}:
        return "fp16"
    return sidecar_format


def _is_qwen_moe_mtp_layout(config: dict[str, Any], normalized_keys: set[str]) -> bool:
    tcfg = text_config(config)
    markers = (
        str(config.get("model_type") or ""),
        str(tcfg.get("model_type") or ""),
        " ".join(str(item) for item in (config.get("architectures") or [])),
        " ".join(str(item) for item in (tcfg.get("architectures") or [])),
    )
    if any("qwen3_5_moe" in marker.lower() or "qwen3_5moe" in marker.lower() for marker in markers):
        return True
    return any(
        key.startswith("mtp.layers.0.mlp.experts.")
        or key.startswith("mtp.layers.0.mlp.switch_mlp.")
        or key.startswith("mtp.layers.0.mlp.shared_expert")
        for key in normalized_keys
    )


def load_config(model_dir: Path | str) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    return json.loads(path.read_text())


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config", config)


def expected_mtp_file(model_dir: Path | str, config: dict[str, Any] | None = None) -> Path:
    model_path = Path(model_dir)
    config = config if config is not None else load_config(model_path)
    extra = config.get("mlx_lm_extra_tensors", {})
    if isinstance(extra, dict) and extra.get("mtp_file"):
        return model_path / str(extra["mtp_file"])
    for rel in ("mtp.safetensors", "mtp/weights.safetensors", "model-mtp.safetensors"):
        candidate = model_path / rel
        if candidate.exists():
            return candidate
    return model_path / "mtp.safetensors"


def mtp_weights_present_on_disk(
    model_dir: Path | str, config: dict[str, Any] | None = None
) -> bool:
    """Whether a model that declares MTP layers actually ships MTP weights.

    A conversion can declare ``num_nextn_predict_layers`` in the config while
    dropping the MTP weights themselves (e.g. the DeepSeek-V4-Flash 2bit-DQ
    build). The runtime uses this probe to tell that benign case (config field
    only -> degrade to autoregressive) apart from a genuine injection failure
    (weights present but unusable -> raise).

    Conservative by design: it only returns ``False`` when it can *positively*
    confirm absence via a shard index that carries no MTP-shaped keys under any
    known naming convention. A sidecar file, a missing/unreadable index, or any
    ambiguity returns ``True`` so the existing injection + validation path runs
    unchanged and a real detection bug on an MTP-bearing model still surfaces.
    """
    model_path = Path(model_dir)
    config = config if config is not None else load_config(model_path)

    # 1. Explicit MTP sidecar file (Qwen/GLM/hy3 external draft head).
    if expected_mtp_file(model_path, config).exists():
        return True

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        # No index to inspect: cannot prove absence, preserve legacy behavior.
        return True
    try:
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get(
            "weight_map", {}
        )
    except Exception:
        return True
    keys = [str(k) for k in weight_map]

    # 2. Namespaced embedded MTP weights ("mtp.*" / "language_model.mtp.*").
    if any(is_mtp_key(k) for k in keys):
        return True

    # 3. DeepSeek-style trailing MTP decoder layer(s) appended after the trunk:
    #    model.layers.{num_hidden_layers + i}.*
    start = int(
        text_config(config).get("num_hidden_layers")
        or config.get("num_hidden_layers")
        or 0
    )
    count = _num_mtp_layers(config)
    if start and count:
        wanted = tuple(f"model.layers.{start + i}." for i in range(count))
        if any(k.startswith(wanted) for k in keys):
            return True

    return False


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "elements": self.elements,
        }


@dataclass(frozen=True)
class MTPInspection:
    mtp_file: str
    exists: bool
    tensor_count: int = 0
    sidecar_format: str = "bf16"
    expected_tensor_count: int = EXPECTED_MTP_TENSOR_COUNT
    metadata_only: bool = False
    tensors: tuple[TensorInfo, ...] = ()
    missing_expected_keys: tuple[str, ...] = ()
    extra_keys: tuple[str, ...] = ()

    @property
    def passes_tensor_gate(self) -> bool:
        return (
            self.exists
            and self.tensor_count == self.expected_tensor_count
            and not self.missing_expected_keys
            and not self.extra_keys
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mtp_file": self.mtp_file,
            "exists": self.exists,
            "tensor_count": self.tensor_count,
            "sidecar_format": self.sidecar_format,
            "expected_tensor_count": self.expected_tensor_count,
            "metadata_only": self.metadata_only,
            "passes_tensor_gate": self.passes_tensor_gate,
            "missing_expected_keys": list(self.missing_expected_keys),
            "extra_keys": list(self.extra_keys),
            "tensors": [t.to_dict() for t in self.tensors],
        }


@dataclass(frozen=True)
class ModelInspection:
    model_dir: str
    config_exists: bool
    architecture: str | None
    model_type: str | None
    mtp_num_hidden_layers: int
    hidden_size: int | None
    num_hidden_layers: int | None
    vocab_size: int | None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    laguna_s_2_1_mlx_4bit_match: bool = False
    laguna_s_2_1_artifacts_complete: bool = False
    mtp_pattern: str | None = None
    source: str = "local"
    quantization: dict[str, Any] = field(default_factory=dict)
    sidecars: dict[str, bool] = field(default_factory=dict)
    model_files: tuple[str, ...] = ()
    weight_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    mtp: MTPInspection | None = None
    runtime_contract_data: dict[str, Any] | None = None
    runtime_contract_error: str | None = None
    runtime_contract_path: str | None = None
    compatibility: dict[str, Any] = field(default_factory=dict)
    runtime_model: str | None = None
    assistant_model: str | None = None
    recommended_sampler: dict[str, Any] | None = None
    backend_status: str | None = None
    backend_artifact: dict[str, Any] | None = None
    gemma4_pair: dict[str, Any] | None = None
    dflash_pair: dict[str, Any] | None = None

    @property
    def passes_primary_gate(self) -> bool:
        if self.compatibility:
            return bool(self.compatibility.get("can_run"))
        return (
            self.config_exists
            and (self.model_type or "").startswith("qwen3_5")
            and self.mtp_num_hidden_layers == 1
            and self.mtp is not None
            and self.mtp.passes_tensor_gate
        )

    def to_dict(self) -> dict[str, Any]:
        runtime_contract_path = (
            self.runtime_contract_path
            or self.compatibility.get("runtime_contract_path")
        )
        return {
            "model_dir": self.model_dir,
            "source": self.source,
            "config_exists": self.config_exists,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
            "mtp_pattern": self.mtp_pattern,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "laguna_s_2_1_mlx_4bit_match": self.laguna_s_2_1_mlx_4bit_match,
            "laguna_s_2_1_artifacts_complete": self.laguna_s_2_1_artifacts_complete,
            "quantization": self.quantization,
            "sidecars": self.sidecars,
            "model_files": list(self.model_files),
            "passes_primary_gate": self.passes_primary_gate,
            "mtp": self.mtp.to_dict() if self.mtp else None,
            "runtime_contract_path": runtime_contract_path,
            "compatibility": self.compatibility,
            "runtime_model": self.runtime_model,
            "assistant_model": self.assistant_model,
            "recommended_sampler": self.recommended_sampler,
            "backend_status": self.backend_status,
            "backend_artifact": self.backend_artifact,
            "gemma4_pair": self.gemma4_pair,
            "dflash_pair": self.dflash_pair,
            "mtp_supported": self.compatibility.get("mtp_supported"),
            "mtp_arch": self.compatibility.get("arch_id"),
            "recommended_backend": self.compatibility.get("recommended_backend"),
            "recommended_profile": self.compatibility.get("recommended_profile"),
            "runtime_compatibility": self.compatibility.get("runtime_compatibility"),
            "architecture_recognized": self.compatibility.get("recognized", False),
            "support_level": self.compatibility.get("support_level"),
            "support_notes": self.compatibility.get("support_notes"),
            "unverified_model": self.compatibility.get("unverified_model", False),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def inspect_mtp_tensors(model_dir: Path | str, config: dict[str, Any] | None = None) -> MTPInspection:
    mtp_path = expected_mtp_file(model_dir, config)
    expected_keys, expected_count, sidecar_format = _mtp_expected_key_set(config or {})
    if not mtp_path.exists():
        return MTPInspection(
            mtp_file=str(mtp_path),
            exists=False,
            sidecar_format=sidecar_format,
            expected_tensor_count=expected_count,
        )

    tensors, tensor_error = _safetensors_header_tensor_infos(mtp_path)
    if tensor_error:
        return MTPInspection(
            mtp_file=str(mtp_path),
            exists=True,
            sidecar_format=sidecar_format,
            expected_tensor_count=expected_count,
            metadata_only=True,
            extra_keys=(tensor_error,),
        )

    key_set = {normalize_mtp_key(t.key) for t in tensors}
    expected_keys, expected_count, sidecar_format = _mtp_expected_key_set(
        config or {},
        keys=tuple(key_set),
    )
    sidecar_format = _observed_sidecar_format(sidecar_format, tuple(tensors))
    return MTPInspection(
        mtp_file=str(mtp_path),
        exists=True,
        tensor_count=len(tensors),
        sidecar_format=sidecar_format,
        expected_tensor_count=expected_count,
        metadata_only=True,
        tensors=tuple(tensors),
        missing_expected_keys=tuple(sorted(expected_keys - key_set)),
        extra_keys=tuple(sorted(key_set - expected_keys)),
    )


def _is_local_like_model_ref(value: str) -> bool:
    expanded = os.path.expanduser(value)
    if Path(expanded).exists():
        return True
    if value.startswith(("/", "./", "../", "~")):
        return True
    first = value.split("/", 1)[0]
    return first in {"models", "outputs", "RESEARCH:PLANS", "REFERENCES:TOOLS"}


def _hf_repo_id_from_ref(value: Path | str) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    known_alias = _KNOWN_PUBLIC_MODEL_ALIASES.get(text.lower())
    if known_alias:
        return known_alias
    if _is_local_like_model_ref(text):
        return None
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc.lower()
        if host not in {"huggingface.co", "www.huggingface.co"}:
            return None
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            return None
        return "/".join(parts[:2])
    if parsed.scheme:
        return None
    parts = [part for part in text.split("/") if part]
    if len(parts) == 2 and all(part not in {".", ".."} for part in parts):
        return "/".join(parts)
    return None


def _hf_download_json(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        return None, None, f"huggingface_hub is required for HF inspection: {exc}"
    try:
        cache_dir = _hf_download_cache_dir()
        kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            **kwargs,
        )
    except Exception as exc:
        return None, None, str(exc)
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), path, None
    except Exception as exc:
        return None, path, str(exc)


def _hf_download_cache_dir() -> Path | None:
    if any(os.environ.get(name) for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")):
        return None
    default = Path("~/.cache/huggingface").expanduser()
    if default.is_symlink() and not default.exists():
        fallback = Path("~/.mtplx/hf-cache").expanduser()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    try:
        default.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        fallback = Path("~/.mtplx/hf-cache").expanduser()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return None


def _looks_like_missing_hf_file(error: str | None) -> bool:
    text = str(error or "").lower()
    return any(
        marker in text
        for marker in (
            "404",
            "entry not found",
            "not found",
            "does not exist",
        )
    )


def _hf_list_repo_files(
    repo_id: str,
    *,
    revision: str | None = None,
) -> tuple[set[str], str | None]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return set(), f"huggingface_hub is required for HF inspection: {exc}"
    try:
        return (
            set(
                HfApi().list_repo_files(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=revision,
                )
            ),
            None,
        )
    except Exception as exc:
        return set(), str(exc)


def _hf_url(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id=repo_id, filename=filename, repo_type="model")


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _hf_fetch_prefix(repo_id: str, filename: str, *, end: int) -> bytes:
    headers = {"Range": f"bytes=0-{end}", "User-Agent": "mtplx-inspect/0.1"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(_hf_url(repo_id, filename), headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _remote_safetensors_keys(repo_id: str, filename: str) -> tuple[tuple[str, ...], str | None]:
    try:
        prefix = _hf_fetch_prefix(repo_id, filename, end=16_383)
        if len(prefix) < 8:
            return (), "safetensors header is shorter than 8 bytes"
        header_len = int.from_bytes(prefix[:8], "little")
        needed = 8 + header_len
        if needed > 1_000_000:
            return (), f"safetensors header too large for metadata-only inspect: {needed} bytes"
        if len(prefix) < needed:
            prefix = _hf_fetch_prefix(repo_id, filename, end=needed - 1)
        header = json.loads(prefix[8:needed].decode("utf-8"))
        return tuple(sorted(key for key in header if key != "__metadata__")), None
    except Exception as exc:
        return (), str(exc)


def _safetensors_header_keys(path: Path) -> tuple[tuple[str, ...], str | None]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                return (), "safetensors header is shorter than 8 bytes"
            header_len = int.from_bytes(prefix, "little")
            if header_len > 1_000_000:
                return (), f"safetensors header too large for metadata-only inspect: {header_len + 8} bytes"
            header = json.loads(handle.read(header_len).decode("utf-8"))
        return tuple(sorted(key for key in header if key != "__metadata__")), None
    except Exception as exc:
        return (), str(exc)


def _safetensors_header_tensor_infos(
    path: Path,
) -> tuple[tuple[TensorInfo, ...], str | None]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                return (), "safetensors header is shorter than 8 bytes"
            header_len = int.from_bytes(prefix, "little")
            if header_len > 1_000_000:
                return (), (
                    "safetensors header too large for metadata-only inspect: "
                    f"{header_len + 8} bytes"
                )
            header = json.loads(handle.read(header_len).decode("utf-8"))
    except Exception as exc:
        return (), str(exc)

    tensors: list[TensorInfo] = []
    for key, payload in sorted(header.items()):
        if key == "__metadata__":
            continue
        if not isinstance(payload, dict):
            return (), f"safetensors tensor metadata for {key!r} is not an object"
        shape = payload.get("shape")
        if not isinstance(shape, list):
            return (), f"safetensors tensor metadata for {key!r} has no shape"
        try:
            tensors.append(
                TensorInfo(
                    key=str(key),
                    dtype=str(payload.get("dtype") or ""),
                    shape=tuple(int(dim) for dim in shape),
                )
            )
        except Exception as exc:
            return (), f"safetensors tensor metadata for {key!r} is invalid: {exc}"
    return tuple(tensors), None


def _weight_keys_from_index_payload(payload: dict[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return ()
    return tuple(sorted(str(key) for key in weight_map))


def _local_model_weight_keys(model_path: Path) -> tuple[tuple[str, ...], str | None]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            return _weight_keys_from_index_payload(
                json.loads(index_path.read_text(encoding="utf-8"))
            ), None
        except Exception as exc:
            return (), str(exc)

    keys: set[str] = set()
    errors: list[str] = []
    for shard in sorted(model_path.glob("model*.safetensors")):
        shard_keys, error = _safetensors_header_keys(shard)
        keys.update(shard_keys)
        if error:
            errors.append(f"{shard.name}: {error}")
    return tuple(sorted(keys)), "; ".join(errors) if errors else None


def _hf_model_weight_keys(repo_id: str, files: set[str]) -> tuple[tuple[str, ...], str | None]:
    if "model.safetensors.index.json" in files:
        index, _path, error = _hf_download_json(repo_id, "model.safetensors.index.json")
        if index is not None:
            return _weight_keys_from_index_payload(index), None
        if error:
            return (), error

    keys: set[str] = set()
    errors: list[str] = []
    for filename in sorted(
        name
        for name in files
        if Path(name).name.startswith("model") and name.endswith(".safetensors")
    ):
        shard_keys, error = _remote_safetensors_keys(repo_id, filename)
        keys.update(shard_keys)
        if error:
            errors.append(f"{filename}: {error}")
    return tuple(sorted(keys)), "; ".join(errors) if errors else None


def _embedded_mtp_keys(weight_keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(normalize_mtp_key(key) for key in weight_keys if is_mtp_key(key)))


def _inspect_mtp_tensors_from_keys(
    mtp_file: str,
    *,
    config: dict[str, Any],
    exists: bool,
    keys: tuple[str, ...] = (),
) -> MTPInspection:
    normalized_keys = tuple(sorted(normalize_mtp_key(key) for key in keys))
    expected_keys, expected_count, sidecar_format = _mtp_expected_key_set(
        config,
        keys=normalized_keys,
    )
    key_set = set(normalized_keys)
    return MTPInspection(
        mtp_file=mtp_file,
        exists=exists,
        tensor_count=len(normalized_keys),
        sidecar_format=sidecar_format,
        expected_tensor_count=expected_count,
        metadata_only=True,
        missing_expected_keys=tuple(sorted(expected_keys - key_set)) if keys else (),
        extra_keys=tuple(sorted(key_set - expected_keys)) if keys else (),
    )


def _mtp_pattern_from_config(config: dict[str, Any]) -> str | None:
    tcfg = text_config(config)
    raw = (
        tcfg.get("mtp_hybrid_override_pattern")
        or config.get("mtp_hybrid_override_pattern")
        or tcfg.get("hybrid_override_pattern")
        or config.get("hybrid_override_pattern")
        or tcfg.get("layers_block_type")
        or config.get("layers_block_type")
    )
    if raw is None:
        return None
    mapping = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}
    if isinstance(raw, str):
        if "," in raw:
            parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
            return "".join(mapping.get(part, part) for part in parts)
        return raw
    if isinstance(raw, list):
        return "".join(mapping.get(str(part).lower(), str(part)) for part in raw)
    return str(raw)


def _remote_laguna_artifacts_complete(repo_id: str, files: set[str]) -> bool:
    return bool(
        repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold()
        and LAGUNA_S_2_1_REQUIRED_FILES.issubset(files)
    )


def _local_laguna_artifacts_complete(model_path: Path) -> bool:
    try:
        source = json.loads(
            (model_path / ".mtplx-source.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if source != {
        "repo_id": LAGUNA_S_2_1_REPO_ID,
        "revision": LAGUNA_S_2_1_REVISION,
    }:
        return False
    if laguna_s_2_1_artifact_integrity_errors(model_path):
        return False
    try:
        index = json.loads(
            (model_path / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    return set(weight_map.values()) == set(LAGUNA_S_2_1_WEIGHT_SHARDS)


def _inspect_hf_model(repo_id: str) -> ModelInspection:
    revision = (
        LAGUNA_S_2_1_REVISION
        if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold()
        else None
    )
    revision_kwargs = {"revision": revision} if revision is not None else {}
    files, files_error = _hf_list_repo_files(repo_id, **revision_kwargs)
    config, config_path, config_error = _hf_download_json(
        repo_id,
        "config.json",
        **revision_kwargs,
    )
    if config is None and "mtplx_pair.json" in files:
        # Assistant-pair bundles (Gemma 4) have no root config.json by
        # design: weights and configs live under target/ and assistant/
        # with mtplx_pair.json at the bundle root. The local loader and
        # the app already understand this layout; classify the remote
        # repo from the pair manifest plus the target config so the HF
        # preflight reaches the same verdict instead of refusing what
        # the engine can run.
        pair_manifest, _pair_path, _pair_error = _hf_download_json(
            repo_id,
            "mtplx_pair.json",
            **revision_kwargs,
        )
        target_config, target_path, _target_error = _hf_download_json(
            repo_id,
            "target/config.json",
            **revision_kwargs,
        )
        if pair_manifest is not None and target_config is not None:
            config = dict(target_config)
            config["assistant_pair_bundle"] = pair_manifest
            config["mtplx_pair.json"] = True
            config_error = None
    if config is None:
        raise RuntimeError(
            f"Could not inspect HF model {repo_id}: "
            f"config.json unavailable: {config_error or files_error}"
        )
    runtime_contract_data, runtime_contract_path, runtime_contract_error = _hf_download_json(
        repo_id,
        "mtplx_runtime.json",
        **revision_kwargs,
    )
    if runtime_contract_data is None and _looks_like_missing_hf_file(runtime_contract_error):
        runtime_contract_error = None
        runtime_contract_path = None
    elif runtime_contract_data is None and runtime_contract_error:
        runtime_contract_path = None
    tcfg = text_config(config)
    archs = config.get("architectures") or tcfg.get("architectures") or []
    architecture = archs[0] if archs else None
    quant = config.get("quantization_config") or config.get("quantization") or {}
    if not quant:
        quant = tcfg.get("quantization_config") or tcfg.get("quantization") or {}
    model_files = tuple(
        sorted(
            name
            for name in files
            if Path(name).name.startswith("model") and name.endswith(".safetensors")
        )
    )
    mtp_file = str(expected_mtp_file(Path("."), config))
    if mtp_file.startswith("./"):
        mtp_file = mtp_file[2:]
    mtp_file = mtp_file.lstrip("/")
    mtp_exists = mtp_file in files if files else False
    weight_keys: tuple[str, ...] = ()
    config_declares_mtp = bool(
        tcfg.get("mtp_num_hidden_layers")
        or tcfg.get("num_nextn_predict_layers")
        or tcfg.get("num_mtp_modules")
        or config.get("num_nextn_predict_layers")
        or config.get("num_mtp_modules")
        or "mtp" in str(architecture or "").lower()
        or "nextn" in str(architecture or "").lower()
    )
    if not mtp_exists and config_declares_mtp:
        weight_keys, _weight_keys_error = _hf_model_weight_keys(repo_id, files)
    mtp_keys: tuple[str, ...] = ()
    mtp_error = None
    if mtp_exists and mtp_file.endswith(".safetensors"):
        mtp_keys, mtp_error = _remote_safetensors_keys(repo_id, mtp_file)
    combined_weight_keys = tuple(sorted(set(weight_keys).union(mtp_keys)))
    if mtp_exists:
        mtp = _inspect_mtp_tensors_from_keys(
            mtp_file,
            config=config,
            exists=True,
            keys=mtp_keys,
        )
    else:
        embedded = _embedded_mtp_keys(weight_keys)
        mtp = _inspect_mtp_tensors_from_keys(
            "model.safetensors.index.json::embedded",
            config=config,
            exists=bool(embedded),
            keys=embedded,
        )
    if mtp_error and not mtp_keys:
        mtp = MTPInspection(
            mtp_file=mtp_file,
            exists=mtp_exists,
            sidecar_format=mtp.sidecar_format,
            expected_tensor_count=mtp.expected_tensor_count,
            metadata_only=True,
        )
    inspection = ModelInspection(
        model_dir=repo_id,
        source="hf",
        config_exists=True,
        architecture=architecture,
        model_type=tcfg.get("model_type") or config.get("model_type"),
        mtp_num_hidden_layers=int(
            tcfg.get("mtp_num_hidden_layers")
            or tcfg.get("num_nextn_predict_layers")
            or tcfg.get("num_mtp_modules")
            or config.get("num_nextn_predict_layers")
            or config.get("num_mtp_modules")
            or 0
        ),
        hidden_size=tcfg.get("hidden_size"),
        num_hidden_layers=tcfg.get("num_hidden_layers"),
        vocab_size=tcfg.get("vocab_size"),
        num_experts=tcfg.get("num_experts"),
        num_experts_per_tok=tcfg.get("num_experts_per_tok"),
        laguna_s_2_1_mlx_4bit_match=is_laguna_s_2_1_mlx_4bit_config(config),
        laguna_s_2_1_artifacts_complete=_remote_laguna_artifacts_complete(
            repo_id,
            files,
        ),
        mtp_pattern=_mtp_pattern_from_config(config),
        quantization=quant,
        sidecars={name: name in files for name in MULTIMODAL_SIDECARS},
        model_files=model_files,
        weight_keys=combined_weight_keys,
        mtp=mtp,
        runtime_contract_data=runtime_contract_data,
        runtime_contract_error=runtime_contract_error,
        runtime_contract_path=(
            runtime_contract_path if runtime_contract_data is not None else None
        ),
    )
    from mtplx.backends.registry import compatibility_for_inspection

    compatibility = compatibility_for_inspection(inspection).to_dict()
    return ModelInspection(
        model_dir=inspection.model_dir,
        source=inspection.source,
        config_exists=inspection.config_exists,
        architecture=inspection.architecture,
        model_type=inspection.model_type,
        mtp_num_hidden_layers=inspection.mtp_num_hidden_layers,
        hidden_size=inspection.hidden_size,
        num_hidden_layers=inspection.num_hidden_layers,
        vocab_size=inspection.vocab_size,
        num_experts=inspection.num_experts,
        num_experts_per_tok=inspection.num_experts_per_tok,
        laguna_s_2_1_mlx_4bit_match=inspection.laguna_s_2_1_mlx_4bit_match,
        laguna_s_2_1_artifacts_complete=inspection.laguna_s_2_1_artifacts_complete,
        mtp_pattern=inspection.mtp_pattern,
        quantization=inspection.quantization,
        sidecars=inspection.sidecars,
        model_files=inspection.model_files,
        weight_keys=inspection.weight_keys,
        mtp=inspection.mtp,
        runtime_contract_data=inspection.runtime_contract_data,
        runtime_contract_error=inspection.runtime_contract_error,
        runtime_contract_path=inspection.runtime_contract_path,
        compatibility=compatibility,
    )


def inspect_model(model_dir: Path | str) -> ModelInspection:
    repo_id = _hf_repo_id_from_ref(model_dir)
    if repo_id is not None:
        return _inspect_hf_model(repo_id)
    model_path = Path(model_dir)
    try:
        from .dflash_pair import dflash_pair_inspection, resolve_dflash_pair_paths
    except Exception:
        dflash_pair = None
    else:
        dflash_pair = resolve_dflash_pair_paths(model_path)
    if dflash_pair is not None:
        payload = dflash_pair_inspection(
            model_ref=str(model_path),
            bundle_root=dflash_pair["bundle_root"],
            target_model=dflash_pair["target_model"],
            drafter_model=dflash_pair["drafter_model"],
            metadata=dflash_pair["metadata"],
        )
        target_config = load_config(dflash_pair["target_model"])
        tcfg = text_config(target_config)
        target_quant = (
            target_config.get("quantization")
            or target_config.get("quantization_config")
            or tcfg.get("quantization")
            or tcfg.get("quantization_config")
            or {}
        )
        return ModelInspection(
            model_dir=str(model_path),
            source="local",
            config_exists=True,
            architecture=str(payload.get("architecture") or "DFlashDrafterPair"),
            model_type=str(payload.get("model_type") or "dflash_pair"),
            mtp_num_hidden_layers=1,
            hidden_size=tcfg.get("hidden_size"),
            num_hidden_layers=tcfg.get("num_hidden_layers"),
            vocab_size=tcfg.get("vocab_size"),
            num_experts=tcfg.get("n_routed_experts") or tcfg.get("num_experts"),
            num_experts_per_tok=tcfg.get("num_experts_per_tok"),
            mtp_pattern="dflash-pair",
            quantization=target_quant,
            sidecars={name: False for name in MULTIMODAL_SIDECARS},
            model_files=tuple(
                sorted(
                    path.name
                    for path in Path(dflash_pair["target_model"]).glob(
                        "model*.safetensors"
                    )
                )
            ),
            runtime_model=payload.get("runtime_model"),
            dflash_pair=payload.get("dflash_pair")
            if isinstance(payload.get("dflash_pair"), dict)
            else None,
            compatibility=payload.get("compatibility")
            if isinstance(payload.get("compatibility"), dict)
            else {},
        )
    try:
        from .gemma4_pair import gemma4_pair_inspection, resolve_gemma4_pair_paths
    except Exception:
        pair = None
    else:
        pair = resolve_gemma4_pair_paths(model_path)
    if pair is not None:
        payload = gemma4_pair_inspection(
            model_ref=str(model_path),
            bundle_root=pair["bundle_root"],
            target_model=pair["target_model"],
            assistant_model=pair["assistant_model"],
            metadata=pair["metadata"],
        )
        target_config = load_config(pair["target_model"])
        tcfg = text_config(target_config)
        archs = target_config.get("architectures") or tcfg.get("architectures") or []
        target_quant = (
            target_config.get("quantization_config")
            or target_config.get("quantization")
            or tcfg.get("quantization_config")
            or tcfg.get("quantization")
            or {}
        )
        return ModelInspection(
            model_dir=str(model_path),
            source="local",
            config_exists=True,
            architecture=str(payload.get("architecture") or (archs[0] if archs else "Gemma4AssistantPair")),
            model_type=str(payload.get("model_type") or "gemma4_pair"),
            mtp_num_hidden_layers=1,
            hidden_size=tcfg.get("hidden_size"),
            num_hidden_layers=tcfg.get("num_hidden_layers"),
            vocab_size=tcfg.get("vocab_size"),
            num_experts=tcfg.get("num_experts"),
            num_experts_per_tok=tcfg.get("num_experts_per_tok"),
            laguna_s_2_1_mlx_4bit_match=is_laguna_s_2_1_mlx_4bit_config(
                target_config
            ),
            laguna_s_2_1_artifacts_complete=_local_laguna_artifacts_complete(
                Path(pair["target_model"])
            ),
            mtp_pattern="assistant-pair",
            quantization=target_quant,
            sidecars={name: False for name in MULTIMODAL_SIDECARS},
            model_files=tuple(sorted(p.name for p in Path(pair["target_model"]).glob("model*.safetensors"))),
            runtime_model=payload.get("runtime_model"),
            assistant_model=payload.get("assistant_model"),
            recommended_sampler=payload.get("recommended_sampler")
            if isinstance(payload.get("recommended_sampler"), dict)
            else None,
            backend_status=payload.get("backend_status"),
            backend_artifact=payload.get("backend_artifact")
            if isinstance(payload.get("backend_artifact"), dict)
            else None,
            gemma4_pair=payload.get("gemma4_pair")
            if isinstance(payload.get("gemma4_pair"), dict)
            else None,
            compatibility=payload.get("compatibility")
            if isinstance(payload.get("compatibility"), dict)
            else {},
        )
    config_path = model_path / "config.json"
    config_exists = config_path.exists()
    config: dict[str, Any] = json.loads(config_path.read_text()) if config_exists else {}
    tcfg = text_config(config)
    archs = config.get("architectures") or tcfg.get("architectures") or []
    architecture = archs[0] if archs else None
    quant = config.get("quantization_config") or config.get("quantization") or {}
    if not quant:
        quant = tcfg.get("quantization_config") or tcfg.get("quantization") or {}

    weight_keys, _weight_keys_error = (
        _local_model_weight_keys(model_path) if config_exists else ((), None)
    )
    if config_exists:
        sidecar_mtp = inspect_mtp_tensors(model_path, config)
        if sidecar_mtp.exists:
            mtp = sidecar_mtp
            weight_keys = tuple(
                sorted(set(weight_keys).union(tensor.key for tensor in sidecar_mtp.tensors))
            )
        else:
            embedded = _embedded_mtp_keys(weight_keys)
            mtp = _inspect_mtp_tensors_from_keys(
                "model.safetensors.index.json::embedded",
                config=config,
                exists=bool(embedded),
                keys=embedded,
            )
    else:
        mtp = None
    inspection = ModelInspection(
        model_dir=str(model_path),
        source="local",
        config_exists=config_exists,
        architecture=architecture,
        model_type=tcfg.get("model_type") or config.get("model_type"),
        mtp_num_hidden_layers=int(
            tcfg.get("mtp_num_hidden_layers")
            or tcfg.get("num_nextn_predict_layers")
            or tcfg.get("num_mtp_modules")
            or config.get("num_nextn_predict_layers")
            or config.get("num_mtp_modules")
            or 0
        ),
        hidden_size=tcfg.get("hidden_size"),
        num_hidden_layers=tcfg.get("num_hidden_layers"),
        vocab_size=tcfg.get("vocab_size"),
        num_experts=tcfg.get("num_experts"),
        num_experts_per_tok=tcfg.get("num_experts_per_tok"),
        laguna_s_2_1_mlx_4bit_match=is_laguna_s_2_1_mlx_4bit_config(config),
        laguna_s_2_1_artifacts_complete=_local_laguna_artifacts_complete(
            model_path
        ),
        mtp_pattern=_mtp_pattern_from_config(config),
        quantization=quant,
        sidecars={name: (model_path / name).exists() for name in MULTIMODAL_SIDECARS},
        model_files=tuple(sorted(p.name for p in model_path.glob("model*.safetensors"))),
        weight_keys=weight_keys,
        mtp=mtp,
    )
    from mtplx.backends.registry import compatibility_for_inspection

    compatibility = compatibility_for_inspection(inspection).to_dict()
    return ModelInspection(
        model_dir=inspection.model_dir,
        source=inspection.source,
        config_exists=inspection.config_exists,
        architecture=inspection.architecture,
        model_type=inspection.model_type,
        mtp_num_hidden_layers=inspection.mtp_num_hidden_layers,
        hidden_size=inspection.hidden_size,
        num_hidden_layers=inspection.num_hidden_layers,
        vocab_size=inspection.vocab_size,
        num_experts=inspection.num_experts,
        num_experts_per_tok=inspection.num_experts_per_tok,
        laguna_s_2_1_mlx_4bit_match=inspection.laguna_s_2_1_mlx_4bit_match,
        laguna_s_2_1_artifacts_complete=inspection.laguna_s_2_1_artifacts_complete,
        mtp_pattern=inspection.mtp_pattern,
        quantization=inspection.quantization,
        sidecars=inspection.sidecars,
        model_files=inspection.model_files,
        weight_keys=inspection.weight_keys,
        mtp=inspection.mtp,
        runtime_contract_data=inspection.runtime_contract_data,
        runtime_contract_error=inspection.runtime_contract_error,
        runtime_contract_path=inspection.runtime_contract_path,
        compatibility=compatibility,
    )


def require_primary_mtp_artifact(model_dir: Path | str) -> ModelInspection:
    inspection = inspect_model(model_dir)
    if not inspection.passes_primary_gate:
        raise RuntimeError(inspection.to_json())
    return inspection
