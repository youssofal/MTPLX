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
    EXPECTED_ALL_PREQUANTIZED_MTP_TENSOR_COUNT,
    EXPECTED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_TENSOR_COUNT,
    EXPECTED_MTP_TENSOR_COUNT,
    MULTIMODAL_SIDECARS,
)
from .profiles import (
    DEFAULT_FP16_HF_MODEL_ID,
    DEFAULT_HF_MODEL_ID,
    LEGACY_OPTIMIZED_HF_MODEL_ID,
    QUALITY_HF_MODEL_ID,
)

MTP_KEY_PREFIXES = ("mtp.", "language_model.mtp.")
_KNOWN_PUBLIC_MODEL_ALIASES = {
    "qwen3.6-27b-mtplx-optimized-speed": DEFAULT_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized": LEGACY_OPTIMIZED_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized-speed-fp16": DEFAULT_FP16_HF_MODEL_ID,
    "qwen3.6-27b-mtplx-optimized-quality": QUALITY_HF_MODEL_ID,
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


def _moe_mtp_expected_key_set(config: dict[str, Any]) -> tuple[set[str], int, str]:
    """Expected MTP keys for a Qwen3.5-MoE (bf16) MTP head.

    The MoE MTP layer mirrors a Qwen3.5-MoE decoder layer: full self-attention
    plus a SparseMoeBlock (router ``gate`` + ``num_experts`` per-expert MLPs and
    a ``shared_expert`` / ``shared_expert_gate``). Experts are stored one tensor
    per expert in the sidecar, so the expected count scales with ``num_experts``
    instead of the dense head's fixed 15.
    """
    tcfg = text_config(config)
    num_experts = int(tcfg.get("num_experts") or 0)
    n_layers = max(_num_mtp_layers(config), 1)
    keys: set[str] = {
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    for li in range(n_layers):
        base = f"mtp.layers.{li}"
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
        for e in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys.add(f"{base}.mlp.experts.{e}.{proj}.weight")
    return keys, len(keys), "bf16-moe"


def _mtp_expected_key_set(
    config: dict[str, Any],
    *,
    keys: tuple[str, ...] = (),
) -> tuple[set[str], int, str]:
    mtp_quant = config.get("mtplx_mtp_quantization", {})
    prequantized = isinstance(mtp_quant, dict) and bool(mtp_quant.get("prequantized"))
    if not prequantized and int(text_config(config).get("num_experts") or 0) > 0:
        return _moe_mtp_expected_key_set(config)
    quant_policy = str(mtp_quant.get("policy") or "") if isinstance(mtp_quant, dict) else ""
    normalized = {normalize_mtp_key(key) for key in keys}
    if prequantized and quant_policy == "all":
        return (
            set(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS),
            EXPECTED_ALL_PREQUANTIZED_MTP_TENSOR_COUNT,
            "prequantized-mlx-affine",
        )
    if prequantized:
        return (
            set(EXPECTED_PREQUANTIZED_MTP_KEYS),
            EXPECTED_PREQUANTIZED_MTP_TENSOR_COUNT,
            "prequantized-mlx-affine",
        )
    if normalized == set(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS):
        return (
            set(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS),
            EXPECTED_ALL_PREQUANTIZED_MTP_TENSOR_COUNT,
            "prequantized-mlx-affine",
        )
    if normalized == set(EXPECTED_PREQUANTIZED_MTP_KEYS):
        return (
            set(EXPECTED_PREQUANTIZED_MTP_KEYS),
            EXPECTED_PREQUANTIZED_MTP_TENSOR_COUNT,
            "prequantized-mlx-affine",
        )
    return set(EXPECTED_MTP_KEYS), EXPECTED_MTP_TENSOR_COUNT, "bf16"


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
            "quantization": self.quantization,
            "sidecars": self.sidecars,
            "model_files": list(self.model_files),
            "passes_primary_gate": self.passes_primary_gate,
            "mtp": self.mtp.to_dict() if self.mtp else None,
            "runtime_contract_path": runtime_contract_path,
            "compatibility": self.compatibility,
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

    from safetensors import safe_open

    tensors: list[TensorInfo] = []
    with safe_open(str(mtp_path), framework="np") as handle:
        keys = sorted(handle.keys())
        for key in keys:
            sl = handle.get_slice(key)
            tensors.append(
                TensorInfo(
                    key=key,
                    dtype=str(sl.get_dtype()),
                    shape=tuple(int(x) for x in sl.get_shape()),
                )
            )

    key_set = {normalize_mtp_key(t.key) for t in tensors}
    expected_keys, expected_count, sidecar_format = _mtp_expected_key_set(
        config or {},
        keys=tuple(key_set),
    )
    return MTPInspection(
        mtp_file=str(mtp_path),
        exists=True,
        tensor_count=len(tensors),
        sidecar_format=sidecar_format,
        expected_tensor_count=expected_count,
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


def _hf_download_json(repo_id: str, filename: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
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


def _hf_list_repo_files(repo_id: str) -> tuple[set[str], str | None]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return set(), f"huggingface_hub is required for HF inspection: {exc}"
    try:
        return set(HfApi().list_repo_files(repo_id=repo_id, repo_type="model")), None
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


def _inspect_hf_model(repo_id: str) -> ModelInspection:
    files, files_error = _hf_list_repo_files(repo_id)
    config, config_path, config_error = _hf_download_json(repo_id, "config.json")
    if config is None:
        raise RuntimeError(
            f"Could not inspect HF model {repo_id}: "
            f"config.json unavailable: {config_error or files_error}"
        )
    runtime_contract_data, runtime_contract_path, runtime_contract_error = _hf_download_json(
        repo_id,
        "mtplx_runtime.json",
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
