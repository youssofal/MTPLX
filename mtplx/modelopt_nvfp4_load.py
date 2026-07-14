"""Load native ModelOpt NVFP4 MLX artifacts on stock mlx_lm."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mtplx.artifacts import load_config

NVFP4_AUX_SUFFIXES = (".global_scale_w", ".input_scale")


def is_modelopt_nvfp4_native_config(config: dict[str, Any]) -> bool:
    policy = config.get("mtplx_policy")
    if not isinstance(policy, dict):
        return False
    source_format = str(policy.get("source_format") or "")
    return source_format == "modelopt-nvfp4-native-w4a16"


def partition_modelopt_nvfp4_weights(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    aux: dict[str, mx.array] = {}
    for key in list(weights):
        if any(key.endswith(suffix) for suffix in NVFP4_AUX_SUFFIXES):
            aux[key] = weights.pop(key)
    return weights, aux


def resolve_module_path(model: nn.Module, module_path: str) -> nn.Module:
    current: Any = model
    for part in module_path.split("."):
        if isinstance(current, nn.Module):
            if part in current:
                current = current[part]
                continue
            if part.isdigit():
                try:
                    current = current[int(part)]
                    continue
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
            current = getattr(current, part)
            continue
        if part.isdigit():
            current = current[int(part)]
            continue
        raise ValueError(f"cannot resolve {module_path!r} at {part!r}")
    if not isinstance(current, nn.Module):
        raise ValueError(f"{module_path!r} did not resolve to an nn.Module")
    return current


def attach_modelopt_nvfp4_aux(
    model: nn.Module, aux_weights: dict[str, mx.array]
) -> int:
    attached = 0
    for key, value in aux_weights.items():
        for suffix in NVFP4_AUX_SUFFIXES:
            if not key.endswith(suffix):
                continue
            module_path = key[: -len(suffix)]
            param_name = suffix[1:]
            module = resolve_module_path(model, module_path)
            module[param_name] = value
            attached += 1
            break
    return attached


def load_modelopt_nvfp4_artifact(
    model_path: Path | str,
    *,
    lazy: bool = False,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load a native ModelOpt NVFP4 artifact with auxiliary NVFP4 scale tensors."""
    from mlx_lm.utils import load_model, load_tokenizer

    path = Path(model_path).expanduser()
    config = load_config(path)
    weights: dict[str, mx.array] = {}
    for weight_file in sorted(glob.glob(str(path / "model*.safetensors"))):
        weights.update(mx.load(weight_file))
    _, aux = partition_modelopt_nvfp4_weights(weights)
    model, loaded_config = load_model(path, lazy=lazy, strict=False)
    attach_modelopt_nvfp4_aux(model, aux)
    tokenizer = load_tokenizer(
        path, eos_token_ids=loaded_config.get("eos_token_id", None)
    )
    return model, tokenizer, loaded_config
