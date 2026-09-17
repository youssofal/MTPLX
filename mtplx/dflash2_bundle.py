"""Canonical artifact contract for Qwen3.8 DFlash2 bundles.

A bundle keeps the target and DFlash2 drafter in separate model directories.
This module owns manifest names, provenance, path resolution, and inspection;
loaders and backends consume these helpers rather than interpreting manifests
independently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DFLASH2_MANIFEST = "mtplx_dflash2.json"
DFLASH2_MANIFEST_SCHEMA_VERSION = 1
DFLASH2_BACKEND = "dflash2"
DFLASH2_ARCH_ID = "dflash2-qwen38"
DFLASH2_TARGET_REPO = "Qwen/Qwen3.8-27B"
DFLASH2_DRAFT_REPO = "z-lab/Qwen3.8-27B-DFlash2"
DFLASH2_ALGORITHM_REPO = "z-lab/dflash"
DFLASH2_DEFAULT_SAMPLER = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
DFLASH2_DRAFT_LAYERS = 5
DFLASH2_TARGET_LAYERS = 64
DFLASH2_DRAFT_PRECISIONS = frozenset({"unquantized", "8bit", "4bit"})
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_dflash2_metadata(bundle_root: str | Path) -> dict[str, Any] | None:
    """Load a manifest, returning ``None`` when it is absent or malformed."""

    return _read_json(Path(bundle_root).expanduser() / DFLASH2_MANIFEST)


# Compatibility aliases for callers that used the backend-local contract.
load_dflash2_manifest = load_dflash2_metadata


def _manifest_backend(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("backend")
        or metadata.get("backend_id")
        or metadata.get("selected_backend")
        or ""
    ).strip()


def _layout_value(metadata: dict[str, Any], *names: str, default: str) -> str:
    layout = metadata.get("layout")
    layout = layout if isinstance(layout, dict) else {}
    for name in names:
        value = layout.get(name)
        if value is None:
            value = metadata.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _safe_child(root: Path, relative: str) -> Path | None:
    candidate = Path(str(relative))
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        return None
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        return None
    return root / candidate


def _manifest_is_resolvable(metadata: dict[str, Any]) -> bool:
    raw_version = metadata.get("schemaVersion", metadata.get("schema_version", DFLASH2_MANIFEST_SCHEMA_VERSION))
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        return False
    return version == DFLASH2_MANIFEST_SCHEMA_VERSION and _manifest_backend(metadata) == DFLASH2_BACKEND


def resolve_dflash2_bundle_paths(bundle_root: str | Path) -> dict[str, Any] | None:
    """Resolve the immutable target and drafter directories.

    This is a layout resolver, not a weight loader.  ``None`` means the path
    is not a resolvable DFlash2 bundle; :func:`dflash2_bundle_inspection`
    provides the detailed fail-closed errors.
    """

    root = Path(bundle_root).expanduser()
    manifest_path = root / DFLASH2_MANIFEST
    if not manifest_path.is_file():
        return None
    metadata = load_dflash2_metadata(root)
    if metadata is None:
        raise ValueError(f"invalid {DFLASH2_MANIFEST}: {manifest_path}")
    if _manifest_backend(metadata) != DFLASH2_BACKEND:
        raise ValueError(f"{DFLASH2_MANIFEST} backend must be 'dflash2'")
    target_name = _layout_value(metadata, "target", default="target")
    draft_name = _layout_value(
        metadata, "draft", "dflash2", "draft_model", "dflash2_model", default="dflash2"
    )
    target = _safe_child(root, target_name)
    draft = _safe_child(root, draft_name)
    if target is None or draft is None or target == root or draft == root:
        raise ValueError(f"{DFLASH2_MANIFEST} layout contains an unsafe path")
    if not target.is_dir() or not draft.is_dir():
        raise ValueError(f"{DFLASH2_MANIFEST} must provide target/ and dflash2/ directories")
    raw_precision = _metadata_precision(metadata) or "unquantized"
    try:
        draft_precision = normalize_dflash2_precision(raw_precision)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    draft_section = metadata.get("draft")
    draft_section = draft_section if isinstance(draft_section, dict) else {}
    raw_block_size = metadata.get("block_size", draft_section.get("block_size", DFLASH2_DRAFT_LAYERS))
    try:
        draft_block_size = int(raw_block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{DFLASH2_BACKEND} block_size must be an integer") from exc
    if draft_block_size < 1:
        raise ValueError(f"{DFLASH2_BACKEND} block_size must be positive")
    return {
        "bundle_root": str(root),
        "target_model": str(target),
        "draft_model": str(draft),
        "dflash2_model": str(draft),
        "target_config": str(target / "config.json"),
        "draft_config": str(draft / "config.json"),
        "metadata": metadata,
        "draft_quantization": draft_precision,
        "draft_precision": draft_precision,
        "draft_block_size": draft_block_size,
    }


resolve_dflash2_paths = resolve_dflash2_bundle_paths


def normalize_dflash2_precision(value: Any) -> str:
    """Normalize the manifest's selected drafter precision."""

    text = str(value or "").strip().lower().replace("-", "").replace(" ", "")
    aliases = {
        "4": "4bit", "q4": "4bit", "4bit": "4bit",
        "8": "8bit", "q8": "8bit", "8bit": "8bit",
        "bf16": "unquantized", "bfloat16": "unquantized",
        "fp16": "unquantized", "float16": "unquantized",
        "fp32": "unquantized", "float32": "unquantized", "none": "unquantized",
        "unquantized": "unquantized",
    }
    result = aliases.get(text)
    if result not in DFLASH2_DRAFT_PRECISIONS:
        raise ValueError(
            "DFlash2 draft precision must be one of 'unquantized', '8bit', or '4bit'"
        )
    return result


def _section(metadata: dict[str, Any], name: str, *aliases: str) -> dict[str, Any]:
    containers: list[dict[str, Any]] = [metadata]
    provenance = metadata.get("provenance")
    if isinstance(provenance, dict):
        containers.append(provenance)
    for container in containers:
        for key in (name, *aliases):
            value = container.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _value(metadata: dict[str, Any], section: str, *keys: str) -> Any:
    aliases = ("dflash",) if section == "algorithm" else ()
    section_data = _section(metadata, section, *aliases)
    for key in keys:
        if section_data.get(key) is not None:
            return section_data[key]
    for container_name in (metadata, metadata.get("provenance")):
        if not isinstance(container_name, dict):
            continue
        for key in keys:
            candidate = container_name.get(f"{section}_{key}")
            if candidate is not None:
                return candidate
    return None


def _metadata_repo(metadata: dict[str, Any], section: str) -> str | None:
    value = _value(metadata, section, "repo", "repository")
    return str(value).strip() if value is not None and str(value).strip() else None


def _metadata_base_model(metadata: dict[str, Any]) -> str | None:
    value = _value(metadata, "target", "base_model", "baseModel")
    return str(value).strip() if value is not None and str(value).strip() else None


def _metadata_revision(metadata: dict[str, Any], section: str) -> str | None:
    value = _value(metadata, section, "revision", "commit", "ref")
    return str(value).strip() if value is not None and str(value).strip() else None


def _metadata_precision(metadata: dict[str, Any]) -> str | None:
    value = _value(metadata, "draft", "precision", "draft_precision", "quantization")
    if value is None:
        value = metadata.get(
            "selected_draft_precision",
            metadata.get("selected_precision", metadata.get("draft_precision", metadata.get("draft_quantization"))),
        )
    if value is None:
        return None
    try:
        return normalize_dflash2_precision(value)
    except ValueError:
        return str(value).strip().lower()


def _config_text(config: dict[str, Any]) -> str:
    text = config.get("text_config")
    text = text if isinstance(text, dict) else {}
    values = [
        config.get("model_type"), text.get("model_type"),
        *(config.get("architectures") or ()), *(text.get("architectures") or ()),
    ]
    return " ".join(str(value or "") for value in values).lower()


def _config_value(config: dict[str, Any], *keys: str) -> Any:
    nested = config.get("text_config")
    nested = nested if isinstance(nested, dict) else {}
    for key in keys:
        if config.get(key) is not None:
            return config[key]
        if nested.get(key) is not None:
            return nested[key]
    return None


def _positive_int(config: dict[str, Any], *keys: str) -> int | None:
    value = _config_value(config, *keys)
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _weight_files(model_dir: Path | None) -> list[Path]:
    if model_dir is None or not model_dir.is_dir():
        return []
    return sorted(path for path in model_dir.rglob("*.safetensors") if path.is_file())


def _target_layer_ids(config: dict[str, Any]) -> Any:
    dflash = config.get("dflash_config")
    dflash = dflash if isinstance(dflash, dict) else {}
    return dflash.get("target_layer_ids", config.get("target_layer_ids"))


def _checksum_entries(metadata: dict[str, Any], name: str) -> list[tuple[str | None, str | None]]:
    checksums = metadata.get("checksums")
    if not isinstance(checksums, dict):
        checksums = metadata.get("sha256")
    checksums = checksums if isinstance(checksums, dict) else {}
    aliases = {
        "target_config": ("target_config", "target/config.json"),
        "draft_config": ("draft_config", "dflash2_config", "draft/config.json"),
        "draft_weights": ("draft_weights", "dflash2_weights", "draft/weights"),
    }[name]
    raw: Any = None
    for key in aliases:
        if checksums.get(key) is not None:
            raw = checksums[key]
            break
    if raw is None:
        raw = metadata.get(f"{name}_sha256")
    if raw is None:
        section_name = "target" if name == "target_config" else "draft"
        section = _section(metadata, section_name)
        section_key = "config_sha256" if name != "draft_weights" else "weights_sha256"
        raw = section.get(section_key)
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    entries: list[tuple[str | None, str | None]] = []
    for value in values:
        if isinstance(value, str):
            entries.append((None, value))
        elif isinstance(value, dict):
            if "sha256" in value or "checksum" in value:
                entries.append((value.get("path") or value.get("file"), value.get("sha256") or value.get("checksum")))
            elif "path" in value and "digest" in value:
                entries.append((value.get("path"), value.get("digest")))
            elif name == "draft_weights" and "files" in value and isinstance(value["files"], list):
                for item in value["files"]:
                    if isinstance(item, dict):
                        entries.append((item.get("path") or item.get("file"), item.get("sha256") or item.get("checksum")))
            else:
                entries.extend((str(path), checksum) for path, checksum in value.items())
        else:
            entries.append((None, None))
    return entries


def _validate_checksums(metadata: dict[str, Any], root: Path, target_dir: Path | None, draft_dir: Path | None) -> list[str]:
    errors: list[str] = []
    defaults = {
        "target_config": "target/config.json",
        "draft_config": "dflash2/config.json",
        "draft_weights": None,
    }
    if target_dir is not None:
        defaults["target_config"] = str(target_dir.relative_to(root) / "config.json")
    if draft_dir is not None:
        defaults["draft_config"] = str(draft_dir.relative_to(root) / "config.json")
    for name, default in defaults.items():
        entries = _checksum_entries(metadata, name)
        if not entries:
            errors.append(f"manifest must record SHA-256 checksum for {name.replace('_', ' ')}")
            continue
        weight_files = _weight_files(draft_dir) if name == "draft_weights" else []
        if name == "draft_weights" and len(entries) == 1 and entries[0][0] is None:
            if len(weight_files) == 1:
                entries = [(str(weight_files[0].relative_to(root)), entries[0][1])]
            else:
                errors.append("draft weights checksum must reference each weight file")
                continue
        for relative, digest in entries:
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                errors.append(f"{name} checksum must be a 64-character SHA-256 hex digest")
                continue
            relative = relative or default
            candidate = _safe_child(root, str(relative))
            if candidate is None or not candidate.is_file():
                errors.append(f"{name} checksum references missing or unsafe file: {relative}")
    return errors


def validate_dflash2_manifest(metadata: dict[str, Any], bundle_root: str | Path | None = None) -> list[str]:
    """Return actionable manifest errors without hashing model weights."""

    errors: list[str] = []
    if not _manifest_is_resolvable(metadata):
        errors.append(f"{DFLASH2_MANIFEST} must declare backend 'dflash2' and a supported schema version")
    for section, label in (("target", "target"), ("draft", "DFlash2 draft"), ("algorithm", "DFlash algorithm")):
        if not _metadata_revision(metadata, section):
            errors.append(f"manifest must pin {label} revision")
    if not _metadata_repo(metadata, "target"):
        errors.append("manifest must record target.repo")
    if not _metadata_repo(metadata, "draft"):
        errors.append("manifest must record draft.repo")
    if not _metadata_repo(metadata, "algorithm"):
        errors.append("manifest must record algorithm.repo")
    precision = _metadata_precision(metadata)
    if precision not in DFLASH2_DRAFT_PRECISIONS:
        errors.append("manifest must record selected draft precision (unquantized, 8bit, or 4bit)")
    if bundle_root is not None:
        root = Path(bundle_root).expanduser()
        for label, name in (("target", "target"), ("draft", "draft")):
            relative = _layout_value(
                metadata,
                name,
                "dflash2" if name == "draft" else name,
                default="dflash2" if name == "draft" else "target",
            )
            if _safe_child(root, relative) is None:
                errors.append(f"{label} layout path is unsafe: {relative}")
        try:
            resolved = resolve_dflash2_bundle_paths(root)
        except (OSError, TypeError, ValueError):
            resolved = None
        target_dir = Path(resolved["target_model"]) if resolved else _safe_child(root, _layout_value(metadata, "target", default="target"))
        draft_dir = Path(resolved["draft_model"]) if resolved else _safe_child(root, _layout_value(metadata, "draft", "dflash2", default="dflash2"))
        errors.extend(_validate_checksums(metadata, root, target_dir, draft_dir))
    return errors


def _validate_bundle(paths: dict[str, Any] | None, bundle_root: Path | None = None) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    metadata = paths.get("metadata") if isinstance(paths, dict) else None
    root = Path(paths["bundle_root"]) if paths and paths.get("bundle_root") else bundle_root
    if not isinstance(metadata, dict) and root is not None:
        metadata = load_dflash2_metadata(root)
    metadata = metadata if isinstance(metadata, dict) else {}
    target_dir = Path(paths["target_model"]) if paths and paths.get("target_model") else None
    draft_dir = Path(paths["draft_model"]) if paths and paths.get("draft_model") else None
    if paths is None:
        errors.append(f"missing or invalid {DFLASH2_MANIFEST} (backend must be 'dflash2')")
    errors.extend(validate_dflash2_manifest(metadata, root) if root is not None else validate_dflash2_manifest(metadata))
    target_config = _read_json(target_dir / "config.json") if target_dir else None
    draft_config = _read_json(draft_dir / "config.json") if draft_dir else None
    if target_config is None:
        errors.append("target/config.json is missing or invalid")
    if draft_config is None:
        errors.append("dflash2/config.json is missing or invalid")
    if not _weight_files(target_dir):
        errors.append("target/ has no safetensors weights")
    if not _weight_files(draft_dir):
        errors.append("dflash2/ has no safetensors weights")
    target_config = target_config or {}
    draft_config = draft_config or {}
    target_text = _config_text(target_config)
    draft_text = _config_text(draft_config)
    target_repo = _metadata_repo(metadata, "target")
    target_base_model = _metadata_base_model(metadata)
    draft_repo = _metadata_repo(metadata, "draft")
    target_identity = f"{target_text} {target_repo or ''} {target_base_model or ''}".lower()
    qwen_target = "qwen3" in target_text and (
        "qwen3.8" in target_identity or "qwen3_8" in target_identity
        or "qwen38" in target_identity
        or (target_repo or "").casefold() == DFLASH2_TARGET_REPO.casefold()
        or (target_base_model or "").casefold() == DFLASH2_TARGET_REPO.casefold()
    )
    if not qwen_target:
        errors.append("target config is not the Qwen3.8 target (use target.base_model for converted targets)")
    if not ("dflash2draftmodel" in draft_text or "dflash2" in draft_text or "dflash_2" in draft_text):
        errors.append("draft config is not a DFlash2 draft")

    target_hidden = _positive_int(target_config, "hidden_size")
    target_vocab = _positive_int(target_config, "vocab_size")
    draft_hidden = _positive_int(draft_config, "hidden_size")
    draft_vocab = _positive_int(draft_config, "vocab_size", "output_vocab_size")
    if target_hidden is None:
        errors.append("target config must declare a positive hidden_size")
    if target_vocab is None:
        errors.append("target config must declare a positive vocab_size")
    if draft_hidden is None:
        errors.append("DFlash2 config must declare a positive hidden_size")
    if draft_vocab is None:
        errors.append("DFlash2 config must declare a positive vocab_size")
    if target_hidden is not None and draft_hidden is not None and target_hidden != draft_hidden:
        errors.append(f"target/draft hidden_size mismatch ({target_hidden} != {draft_hidden})")
    if target_vocab is not None and draft_vocab is not None and target_vocab != draft_vocab:
        errors.append(f"target/draft vocab_size mismatch ({target_vocab} != {draft_vocab})")

    target_layers = _positive_int(target_config, "num_hidden_layers", "num_layers")
    draft_target_layers = _positive_int(draft_config, "num_target_layers")
    draft_layers = _positive_int(draft_config, "num_hidden_layers", "num_layers")
    if target_layers is None:
        errors.append("target config must declare a positive num_hidden_layers")
    elif target_layers != DFLASH2_TARGET_LAYERS:
        errors.append(f"Qwen3.8 target must have {DFLASH2_TARGET_LAYERS} layers")
    if draft_target_layers is None:
        errors.append("DFlash2 config must declare num_target_layers")
    elif target_layers is not None and draft_target_layers != target_layers:
        errors.append(f"num_target_layers mismatch ({draft_target_layers} != target layer count {target_layers})")
    if draft_layers != DFLASH2_DRAFT_LAYERS:
        errors.append(
            f"DFlash2 draft must have {DFLASH2_DRAFT_LAYERS} layers"
            if draft_layers is not None else "DFlash2 config must declare num_hidden_layers"
        )
    layer_ids = _target_layer_ids(draft_config)
    if not isinstance(layer_ids, list) or not layer_ids:
        errors.append("DFlash2 config must declare dflash_config.target_layer_ids")
    else:
        if draft_layers is not None and len(layer_ids) != draft_layers:
            errors.append("target_layer_ids length must equal draft num_hidden_layers")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in layer_ids):
            errors.append("target_layer_ids must contain integers")
        elif target_layers is not None and any(value < 0 or value >= target_layers for value in layer_ids):
            errors.append("target_layer_ids must be within target layer range")
        elif len(set(layer_ids)) != len(layer_ids):
            errors.append("target_layer_ids must be unique")

    details = {
        "target_revision": _metadata_revision(metadata, "target"),
        "draft_revision": _metadata_revision(metadata, "draft"),
        "algorithm_revision": _metadata_revision(metadata, "algorithm"),
        "target_repo": target_repo,
        "target_base_model": target_base_model,
        "draft_repo": draft_repo,
        "algorithm_repo": _metadata_repo(metadata, "algorithm"),
        "draft_precision": _metadata_precision(metadata),
        "target_hidden_size": target_hidden,
        "draft_hidden_size": draft_hidden,
        "target_vocab_size": target_vocab,
        "draft_vocab_size": draft_vocab,
        "target_layer_count": target_layers,
        "num_target_layers": draft_target_layers,
        "draft_layer_count": draft_layers,
        "target_layer_ids": list(layer_ids) if isinstance(layer_ids, list) else None,
        "target_model": str(target_dir) if target_dir else None,
        "draft_model": str(draft_dir) if draft_dir else None,
    }
    return errors, metadata, details


def _sampler_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("sampler") or metadata.get("generation_config")
    if not isinstance(value, dict):
        return dict(DFLASH2_DEFAULT_SAMPLER)
    try:
        sampler = {"temperature": float(value["temperature"]), "top_p": float(value["top_p"]), "top_k": int(value["top_k"])}
    except (KeyError, TypeError, ValueError):
        return dict(DFLASH2_DEFAULT_SAMPLER)
    if sampler["temperature"] < 0 or not 0 < sampler["top_p"] <= 1 or sampler["top_k"] < 0:
        return dict(DFLASH2_DEFAULT_SAMPLER)
    return sampler


def dflash2_bundle_inspection(*, model_ref: str, bundle_root: str | Path, paths: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return inspection data and a fail-closed compatibility verdict."""

    root = Path(bundle_root).expanduser()
    if paths is not None:
        resolved = paths
    else:
        try:
            resolved = resolve_dflash2_bundle_paths(root)
        except ValueError:
            resolved = None
    errors, metadata, details = _validate_bundle(resolved, root)
    can_run = not errors
    sampler = _sampler_from_metadata(metadata)
    verdict = {
        "tier": "verified" if can_run else "incompatible-architecture",
        "can_run": can_run,
        "supported": can_run,
        "recognized": True,
        "exit_code": 0 if can_run else 3,
        "message": "Qwen3.8 DFlash2 bundle is complete and selected explicitly." if can_run else "DFlash2 bundle rejected: " + "; ".join(errors),
        "arch_id": DFLASH2_ARCH_ID,
        "recommended_backend": DFLASH2_BACKEND,
        "backend": DFLASH2_BACKEND,
        "selected_backend": DFLASH2_BACKEND,
        "target_revision": details["target_revision"],
        "draft_revision": details["draft_revision"],
        "algorithm_revision": details["algorithm_revision"],
        "draft_precision": details["draft_precision"],
        "sampler": sampler,
        "runtime_compatibility": "dflash2-bundle-native" if can_run else "invalid-dflash2-bundle",
        "support_level": "dflash2-bundle" if can_run else "dflash2-bundle-invalid",
        "unverified_model": not can_run,
        "errors": list(errors),
    }
    return {
        "source": model_ref,
        "model_dir": str(root),
        "runtime_model": details["target_model"],
        "draft_model": details["draft_model"],
        "architecture": "DFlash2Qwen38Bundle",
        "model_type": "dflash2_bundle",
        "mtp_arch": DFLASH2_ARCH_ID,
        "backend": DFLASH2_BACKEND,
        "backend_id": DFLASH2_BACKEND,
        "selected_backend": DFLASH2_BACKEND,
        "target_revision": details["target_revision"],
        "draft_revision": details["draft_revision"],
        "algorithm_revision": details["algorithm_revision"],
        "draft_precision": details["draft_precision"],
        "recommended_sampler": sampler,
        "sampler": sampler,
        "dflash2_bundle": {
            "bundle_root": str(root),
            "target_model": details["target_model"],
            "draft_model": details["draft_model"],
            "backend": DFLASH2_BACKEND,
            "metadata": metadata,
            **details,
        },
        "verdict": verdict,
        "compatibility": verdict,
    }


inspect_dflash2_bundle = dflash2_bundle_inspection


def is_dflash2_bundle_candidate(bundle_root: str | Path) -> bool:
    return (Path(bundle_root).expanduser() / DFLASH2_MANIFEST).is_file()
