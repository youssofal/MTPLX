"""NVIDIA ModelOpt NVFP4 to native MLX conversion helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx

from mtplx.artifacts import text_config
from mtplx.compressed_tensors import (
    ProgressCallback,
    _TensorReader,
    _add_main_tensor,
    _audit_conversion,
    _copy_sidecars,
    _decode_e4m3fn,
    _emit,
    _key_mapper_for_config,
    _load_json,
    _quant_group,
    _quantized_module_prefixes,
    _record_stacked_quantized_modules,
    _sanitize_plain_weight,
    _write_json,
    _write_readme,
    convert_fp8_weight_to_q8_0,
    global_scale_w_from_inverse_tensor,
    pack_nvfp4_weight_bytes_to_mlx,
    repack_nvfp4_block_scales_to_mlx,
)
from mtplx.expert_layout import (
    NumberedExpertAccumulator,
    num_experts_from_config,
    stack_numbered_experts,
)

MODEL_QUANT_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
)
MTP_SIDECAR_RELATIVE = Path("mtp") / "weights.safetensors"
NVFP4_QUANT_PARAMS = {"bits": 4, "group_size": 16, "mode": "nvfp4"}


def convert_modelopt_nvfp4_to_mlx(
    source_path: Path,
    output_path: Path,
    *,
    source_repo: str,
    source_sha: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Convert NVIDIA ModelOpt NVFP4 weights to native MLX nvfp4 artifacts."""

    source_path = Path(source_path).expanduser()
    output_path = Path(output_path).expanduser()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True)

    index = _load_json(source_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("modelopt source is missing model.safetensors weight_map")
    weight_map = {str(key): str(value) for key, value in weight_map.items()}
    source_files = sorted(set(weight_map.values()))
    source_config = _load_json(source_path / "config.json")
    key_mapper = _key_mapper_for_config(source_config)
    num_experts = num_experts_from_config(source_config)

    main_index: dict[str, Any] = {"metadata": {}, "weight_map": {}}
    mtp_weights: dict[str, mx.array] = {}
    total_size = 0
    quantized_modules: set[str] = set()
    mtp_quantized_modules: set[str] = set()
    main_experts = NumberedExpertAccumulator(num_experts=num_experts or None)
    source_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()

    keys_by_file: dict[str, list[str]] = {name: [] for name in source_files}
    for key, filename in weight_map.items():
        keys_by_file.setdefault(filename, []).append(key)

    for index_i, filename in enumerate(source_files, start=1):
        _emit(
            progress_callback,
            {
                "event": "shard_start",
                "filename": filename,
                "completed": index_i - 1,
                "total": len(source_files),
            },
        )
        out: dict[str, mx.array] = {}
        consumed: set[str] = set()
        with _TensorReader(source_path, weight_map) as reader:
            for key in sorted(keys_by_file.get(filename, [])):
                if key in consumed:
                    continue
                if key.endswith(".weight") and _is_modelopt_nvfp4_prefix(
                    key[: -len(".weight")], weight_map
                ):
                    prefix = key[: -len(".weight")]
                    packed = _convert_modelopt_nvfp4(prefix, reader)
                    consumed.update(_modelopt_nvfp4_keys(prefix))
                    if prefix.startswith("mtp."):
                        mtp_weights.update(packed)
                        output_counts[f"mtp_{_quant_group(prefix)}_nvfp4"] += 1
                    else:
                        module_path = key_mapper(prefix)
                        for packed_key, packed_value in packed.items():
                            _add_main_tensor(
                                out,
                                key_mapper(packed_key),
                                packed_value,
                                main_experts,
                            )
                        if ".mlp.experts." not in module_path:
                            quantized_modules.add(module_path)
                            output_counts[
                                f"main_{_quant_group(module_path)}_nvfp4"
                            ] += 1
                    source_counts[f"nvfp4_{_quant_group(prefix)}"] += 1
                    continue
                if _is_modelopt_aux_key(key, weight_map):
                    continue
                if key.startswith("mtp."):
                    try:
                        value = reader.tensor(key)
                    except Exception as exc:
                        raise RuntimeError(
                            f"failed to convert {filename}:{key}: {exc}"
                        ) from exc
                    mtp_weights[key] = _sanitize_plain_weight(key, value)
                    source_counts["mtp_bf16"] += 1
                    continue
                if _is_fp8_linear_weight(key, weight_map):
                    prefix = key[: -len(".weight")]
                    packed = _convert_modelopt_fp8_linear(prefix, reader)
                    consumed.update(_modelopt_fp8_keys(prefix))
                    module_path = key_mapper(prefix)
                    for packed_key, packed_value in packed.items():
                        _add_main_tensor(
                            out,
                            key_mapper(packed_key),
                            packed_value,
                            main_experts,
                        )
                    if ".mlp.experts." not in module_path:
                        quantized_modules.add(module_path)
                        output_counts[f"main_{_quant_group(module_path)}_q8_0"] += 1
                    source_counts[f"fp8_{_quant_group(module_path)}"] += 1
                    continue
                try:
                    value = reader.tensor(key)
                    out_key = key_mapper(key)
                    _add_main_tensor(
                        out,
                        out_key,
                        _sanitize_plain_weight(out_key, value),
                        main_experts,
                    )
                    source_counts[f"plain_{_quant_group(out_key)}"] += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"failed to convert {filename}:{key}: {exc}"
                    ) from exc

        stacked_main = main_experts.flush_complete()
        out.update(stacked_main)
        _record_stacked_quantized_modules(
            stacked_main,
            quantized_modules=quantized_modules,
            output_counts=output_counts,
        )
        if out:
            out_path = output_path / filename
            mx.save_safetensors(str(out_path), out, metadata={"format": "mlx"})
            for out_key, value in out.items():
                main_index["weight_map"][out_key] = filename
                total_size += int(value.nbytes)
        del out
        _emit(
            progress_callback,
            {
                "event": "shard_complete",
                "filename": filename,
                "completed": index_i,
                "total": len(source_files),
            },
        )

    remaining_main = main_experts.flush_remaining(strict=True)
    if remaining_main:
        _record_stacked_quantized_modules(
            remaining_main,
            quantized_modules=quantized_modules,
            output_counts=output_counts,
        )
        expert_filename = "model-experts.safetensors"
        mx.save_safetensors(
            str(output_path / expert_filename),
            remaining_main,
            metadata={"format": "mlx"},
        )
        for out_key, value in remaining_main.items():
            main_index["weight_map"][out_key] = expert_filename
            total_size += int(value.nbytes)
        del remaining_main

    mtp_size = 0
    if mtp_weights:
        if num_experts > 0:
            mtp_weights = stack_numbered_experts(
                mtp_weights,
                num_experts=num_experts,
                strict=True,
            )
        mtp_quantized_modules = _quantized_module_prefixes(mtp_weights)
        mtp_path = output_path / MTP_SIDECAR_RELATIVE
        mtp_path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(mtp_path), mtp_weights, metadata={"format": "mlx"})
        mtp_size = sum(int(value.nbytes) for value in mtp_weights.values())

    main_index["metadata"] = {
        "total_size": total_size,
        "source_repo": source_repo,
        "source_sha": source_sha,
        "format": "mlx",
        "mtp_sidecar_size": mtp_size,
    }
    main_index["weight_map"] = {
        key: main_index["weight_map"][key] for key in sorted(main_index["weight_map"])
    }
    _write_json(output_path / "model.safetensors.index.json", main_index)
    _copy_sidecars(source_path, output_path)

    stats = {
        "source_files": source_files,
        "source_counts": dict(sorted(source_counts.items())),
        "output_counts": dict(sorted(output_counts.items())),
        "main_quantized_modules": len(quantized_modules),
        "mtp_quantized_modules": len(mtp_quantized_modules),
        "main_tensor_count": len(main_index["weight_map"]),
        "mtp_tensor_count": len(mtp_weights),
        "main_total_size": total_size,
        "mtp_sidecar_size": mtp_size,
    }
    audit = _audit_conversion(
        quantized_modules=quantized_modules,
        mtp_quantized_modules=mtp_quantized_modules,
        main_index=main_index,
        mtp_weights=mtp_weights,
    )
    _write_modelopt_config(
        source_config,
        output_path,
        source_repo=source_repo,
        source_sha=source_sha,
        quantized_modules=quantized_modules,
        mtp_quantized_modules=mtp_quantized_modules,
        stats=stats,
        audit=audit,
    )
    _write_readme(
        output_path,
        source_repo=source_repo,
        source_sha=source_sha,
        stats=stats,
        audit=audit,
        source_format_label="modelopt-nvfp4-native-w4a16",
    )
    return {
        "source": source_repo,
        "source_path": str(source_path),
        "source_sha": source_sha,
        "output_path": str(output_path),
        "stats": stats,
        "audit": audit,
    }


def _convert_modelopt_nvfp4(prefix: str, reader: _TensorReader) -> dict[str, mx.array]:
    packed = reader.tensor(f"{prefix}.weight").astype(mx.uint8)
    scale = _read_modelopt_scale_bytes(reader, f"{prefix}.weight_scale")
    out = {
        f"{prefix}.weight": pack_nvfp4_weight_bytes_to_mlx(packed),
        f"{prefix}.scales": repack_nvfp4_block_scales_to_mlx(scale),
    }
    try:
        inverse = reader.tensor(f"{prefix}.weight_scale_2").astype(mx.float32)
        out[f"{prefix}.global_scale_w"] = global_scale_w_from_inverse_tensor(inverse)
    except KeyError:
        pass
    try:
        out[f"{prefix}.input_scale"] = (
            reader.tensor(f"{prefix}.input_scale").astype(mx.float32).reshape(1)
        )
    except KeyError:
        pass
    return out


def _convert_modelopt_fp8_linear(
    prefix: str, reader: _TensorReader
) -> dict[str, mx.array]:
    raw, metadata = reader.raw_tensor(f"{prefix}.weight")
    if metadata.get("dtype") == "F8_E4M3":
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError(
                f"numpy is required to read F8_E4M3 safetensors: {exc}"
            ) from exc
        shape = tuple(int(dim) for dim in metadata.get("shape") or ())
        decoded = _decode_e4m3fn(np.frombuffer(raw, dtype=np.uint8)).reshape(shape)
        dense = mx.array(decoded, dtype=mx.float16)
    else:
        dense = reader.tensor(f"{prefix}.weight").astype(mx.float16)
    q = convert_fp8_weight_to_q8_0(dense)
    return {
        f"{prefix}.weight": q["weight"],
        f"{prefix}.scales": q["scales"],
        f"{prefix}.biases": q["biases"],
    }


def _read_modelopt_scale_bytes(reader: _TensorReader, key: str) -> mx.array:
    raw, metadata = reader.raw_tensor(key)
    shape = tuple(int(dim) for dim in metadata.get("shape") or ())
    if metadata.get("dtype") == "F8_E4M3":
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError(
                f"numpy is required to read F8_E4M3 safetensors: {exc}"
            ) from exc
        return mx.array(
            np.frombuffer(raw, dtype=np.uint8).reshape(shape), dtype=mx.uint8
        )
    tensor = reader.tensor(key)
    if tensor.dtype == mx.uint8:
        return tensor
    raise ValueError(f"{key} must be raw F8_E4M3 or uint8 for native NVFP4 repack")


def _is_modelopt_nvfp4_prefix(prefix: str, weight_map: dict[str, str]) -> bool:
    return (
        f"{prefix}.weight_scale" in weight_map
        and f"{prefix}.weight_scale_2" in weight_map
        and not prefix.endswith(".weight")
    )


def _modelopt_nvfp4_keys(prefix: str) -> set[str]:
    return {f"{prefix}{suffix}" for suffix in MODEL_QUANT_SUFFIXES}


def _modelopt_fp8_keys(prefix: str) -> set[str]:
    return {f"{prefix}.weight", f"{prefix}.weight_scale", f"{prefix}.input_scale"}


def _is_modelopt_aux_key(key: str, weight_map: dict[str, str]) -> bool:
    for suffix in MODEL_QUANT_SUFFIXES:
        if suffix == ".weight":
            continue
        if key.endswith(suffix):
            prefix = key[: -len(suffix)]
            return _is_modelopt_nvfp4_prefix(prefix, weight_map)
    return False


def _is_fp8_linear_weight(key: str, weight_map: dict[str, str]) -> bool:
    if not key.endswith(".weight"):
        return False
    prefix = key[: -len(".weight")]
    if _is_modelopt_nvfp4_prefix(prefix, weight_map):
        return False
    scale_key = f"{prefix}.weight_scale"
    if scale_key not in weight_map:
        return False
    return ".linear_attn." in prefix or ".self_attn." in prefix


def _write_modelopt_config(
    source_config: dict[str, Any],
    output_path: Path,
    *,
    source_repo: str,
    source_sha: str | None,
    quantized_modules: set[str],
    mtp_quantized_modules: set[str],
    stats: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    final_config = dict(source_config)
    tcfg = dict(text_config(source_config))
    tcfg.pop("quantization_config", None)
    tcfg.pop("quantization", None)
    if tcfg:
        final_config["text_config"] = tcfg
    final_config["library_name"] = "mlx"
    final_config.setdefault("language_model_only", False)

    quantization: dict[str, Any] = dict(NVFP4_QUANT_PARAMS)
    for module_path in sorted(quantized_modules):
        if ".linear_attn." in module_path:
            quantization[module_path] = {"bits": 8, "group_size": 32, "mode": "affine"}
        else:
            quantization[module_path] = dict(NVFP4_QUANT_PARAMS)
    final_config["quantization"] = quantization
    final_config["quantization_config"] = quantization
    if mtp_quantized_modules:
        final_config["mtplx_mtp_quantization"] = {
            "policy": "cyankiwi",
            "source_format": "modelopt-nvfp4-native-w4a16",
            "prequantized": True,
            **NVFP4_QUANT_PARAMS,
            "quantized_modules": len(mtp_quantized_modules),
        }
    if stats.get("mtp_tensor_count"):
        final_config["mlx_lm_extra_tensors"] = {
            "mtp_file": str(MTP_SIDECAR_RELATIVE).replace("\\", "/"),
            "mtp_tensor_count": stats["mtp_tensor_count"],
        }
    final_config["mtplx_policy"] = {
        "name": "forge-modelopt-nvfp4-native-w4a16",
        "source": source_repo,
        "source_sha": source_sha,
        "quantization_family": "modelopt-nvfp4-native",
        "source_format": "modelopt-nvfp4-native-w4a16",
        "awq_calibrated": False,
        "main_quantized_modules": len(quantized_modules),
        "mtp_quantized_modules": len(mtp_quantized_modules),
        "stats": stats,
        "audit": audit,
    }
    _write_json(output_path / "config.json", dict(sorted(final_config.items())))
