"""NVIDIA ModelOpt NVFP4/FP8 to standard MLX affine conversion."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx

from mtplx.compressed_tensors import (
    SIDECAR_FILES,
    _read_f8_e4m3_tensor,
    _TensorReader,
)
from mtplx.expert_layout import NumberedExpertAccumulator, num_experts_from_config

_E2M1 = mx.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=mx.float32,
)


def dequantize_modelopt_nvfp4(
    packed: mx.array,
    block_scale: mx.array,
    weight_scale_2: mx.array,
) -> mx.array:
    """Apply ModelOpt's exported ``nibble * block_scale * scale_2`` contract."""
    packed = packed.astype(mx.uint8)
    low = mx.take(_E2M1, packed & 0xF)
    high = mx.take(_E2M1, (packed >> 4) & 0xF)
    values = mx.stack([low, high], axis=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    scales = mx.repeat(block_scale.astype(mx.float32), repeats=16, axis=-1)
    if tuple(scales.shape) != tuple(values.shape):
        raise ValueError(
            f"ModelOpt NVFP4 scales {scales.shape} do not match weights {values.shape}"
        )
    return values * scales * weight_scale_2.astype(mx.float32)


def dequantize_modelopt_fp8(weight: mx.array, weight_scale: mx.array) -> mx.array:
    """Dequantize ModelOpt FP8 weight tensors before MLX affine requantization."""
    return weight.astype(mx.float32) * weight_scale.astype(mx.float32)


def requantize_affine4(weight: mx.array, *, group_size: int = 64) -> dict[str, mx.array]:
    qweight, scales, biases = mx.quantize(
        weight.astype(mx.float16), group_size=group_size, bits=4, mode="affine"
    )
    return {"weight": qweight, "scales": scales, "biases": biases}


def _normalized_expert_key(key: str) -> str:
    return key.replace(".switch_mlp.up_proj.", ".switch_mlp.fc1.").replace(
        ".switch_mlp.down_proj.", ".switch_mlp.fc2."
    )


def _quantized_module(prefix: str) -> str:
    if ".experts." not in prefix:
        return prefix
    before, after = prefix.split(".experts.", 1)
    _expert, projection = after.split(".", 1)
    projection = {"up_proj": "fc1", "down_proj": "fc2"}.get(
        projection, projection
    )
    return f"{before}.switch_mlp.{projection}"


def convert_modelopt_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    *,
    group_size: int = 64,
    source_repo: str | None = None,
    source_sha: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Stream a ModelOpt mixed NVFP4/FP8 checkpoint into MLX affine INT4."""
    source = Path(source_path).expanduser()
    output = Path(output_path).expanduser()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        source_index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = {str(k): str(v) for k, v in source_index["weight_map"].items()}
    else:
        only_file = "model.safetensors"
        weights = mx.load(str(source / only_file))
        weight_map = {str(key): only_file for key in weights}
        del weights

    source_files = sorted(set(weight_map.values()))
    keys_by_file = {filename: [] for filename in source_files}
    for key, filename in weight_map.items():
        keys_by_file[filename].append(key)

    num_experts = num_experts_from_config(config)
    experts = NumberedExpertAccumulator(num_experts=num_experts or None)
    quantized_modules: set[str] = set()
    output_map: dict[str, str] = {}
    total_size = 0
    counts = {"nvfp4": 0, "fp8": 0, "plain": 0}

    with _TensorReader(source, weight_map) as reader:
        for file_index, filename in enumerate(source_files, start=1):
            if progress_callback:
                progress_callback(
                    {
                        "event": "shard_start",
                        "filename": filename,
                        "completed": file_index - 1,
                        "total": len(source_files),
                    }
                )
            out: dict[str, mx.array] = {}
            for key in sorted(keys_by_file[filename]):
                if key.endswith((".k_scale", ".v_scale")):
                    # Exported KV-cache quantization metadata is not a model
                    # parameter in MLX's Nemotron-H implementation.
                    continue
                if key.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                    continue
                if not key.endswith(".weight"):
                    out[key] = reader.tensor(key)
                    counts["plain"] += 1
                    continue

                prefix = key[: -len(".weight")]
                scale_key = f"{prefix}.weight_scale"
                scale2_key = f"{prefix}.weight_scale_2"
                if scale2_key in weight_map:
                    packed = reader.tensor(key).astype(mx.uint8)
                    scale = _read_f8_e4m3_tensor(reader, scale_key)
                    scale2 = reader.tensor(scale2_key)
                    weight = dequantize_modelopt_nvfp4(packed, scale, scale2)
                    converted = requantize_affine4(weight, group_size=group_size)
                    counts["nvfp4"] += 1
                elif scale_key in weight_map:
                    weight = _read_f8_e4m3_tensor(reader, key)
                    scale = reader.tensor(scale_key)
                    converted = requantize_affine4(
                        dequantize_modelopt_fp8(weight, scale), group_size=group_size
                    )
                    counts["fp8"] += 1
                else:
                    out[key] = reader.tensor(key)
                    counts["plain"] += 1
                    continue

                quantized_modules.add(_quantized_module(prefix))
                for leaf, value in converted.items():
                    out_key = f"{prefix}.{leaf}"
                    if not experts.add(out_key, value):
                        out[out_key] = value

            out.update(
                {
                    _normalized_expert_key(key): value
                    for key, value in experts.flush_complete().items()
                }
            )
            if out:
                output_file = output / filename
                mx.save_safetensors(str(output_file), out, metadata={"format": "mlx"})
                for key, value in out.items():
                    output_map[key] = filename
                    total_size += int(value.nbytes)
            if progress_callback:
                progress_callback(
                    {
                        "event": "shard_complete",
                        "filename": filename,
                        "completed": file_index,
                        "total": len(source_files),
                    }
                )

    remaining = {
        _normalized_expert_key(key): value
        for key, value in experts.flush_remaining(strict=True).items()
    }
    if remaining:
        filename = "model-experts.safetensors"
        mx.save_safetensors(str(output / filename), remaining, metadata={"format": "mlx"})
        for key, value in remaining.items():
            output_map[key] = filename
            total_size += int(value.nbytes)

    qparams = {"group_size": group_size, "bits": 4, "mode": "affine"}
    quantization = dict(qparams)
    quantization.update(
        {key: dict(qparams) for key in sorted(quantized_modules)}
    )
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    config["mtplx_source_quantization"] = {
        "format": "modelopt-w4a16-nvfp4-mixed-fp8",
        "source": source_repo,
        "revision": source_sha,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    for name in SIDECAR_FILES:
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, output / name)
    index = {
        "metadata": {"total_size": total_size, "source_sha": source_sha},
        "weight_map": {key: output_map[key] for key in sorted(output_map)},
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "source": str(source),
        "output": str(output),
        "counts": counts,
        "quantized_modules": len(quantized_modules),
        "total_size": total_size,
    }
