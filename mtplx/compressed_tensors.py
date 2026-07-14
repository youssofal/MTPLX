"""Compressed-tensors to MLX conversion helpers for Forge."""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import struct
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx

from mtplx.artifacts import text_config
from mtplx.expert_layout import (
    NumberedExpertAccumulator,
    num_experts_from_config,
    stack_numbered_experts,
)


ProgressCallback = Callable[[dict[str, Any]], None]

SIDECAR_FILES = (
    "LICENSE",
    "chat_template.jinja",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
MAIN_RMSNORM_SHIFT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "model.norm.weight",
)
MTP_RMSNORM_SHIFT_IF_LOW_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
)
MTP_RMSNORM_ALWAYS_SHIFT_SUFFIXES = (
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mtp.norm.weight",
)


def convert_compressed_tensors_awq_to_mlx(
    source_path: Path,
    output_path: Path,
    *,
    source_repo: str,
    source_sha: str | None = None,
    progress_callback: ProgressCallback | None = None,
    packed_format: str = "awq",
    target_bits: int | None = None,
    target_group_size: int | None = None,
    target_mode: str | None = None,
) -> dict[str, Any]:
    """Convert compressed-tensors pack-quantized AWQ weights to MLX affine.

    The Hugging Face compressed-tensors layout stores each quantized linear as
    ``weight_packed`` plus scale/zero-point tensors. MLX's QuantizedLinear
    expects the same calibrated payload under ``weight``, ``scales``, and
    ``biases``. This conversion preserves the calibrated INT4 body instead of
    dequantizing and requantizing it.
    """

    source_path = Path(source_path).expanduser()
    output_path = Path(output_path).expanduser()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True)

    index = _load_json(source_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(
            "compressed-tensors source is missing model.safetensors weight_map"
        )
    weight_map = {str(key): str(value) for key, value in weight_map.items()}
    source_files = sorted(set(weight_map.values()))
    source_config = _load_json(source_path / "config.json")
    quant_params = _quant_params_from_config(source_config)
    if packed_format in {"nvfp4", "nvfp4_native"}:
        quant_params = {
            "bits": int(target_bits or 4),
            "group_size": int(
                target_group_size or (16 if packed_format == "nvfp4_native" else 64)
            ),
            "mode": str(
                target_mode
                or ("nvfp4" if packed_format == "nvfp4_native" else "affine")
            ),
        }
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
                if key in consumed or key.endswith(".weight_shape"):
                    continue
                if key.endswith(".weight_packed"):
                    prefix = key[: -len(".weight_packed")]
                    if packed_format == "nvfp4_native":
                        packed = _convert_nvfp4_native(prefix, reader)
                        source_label = f"nvfp4_native_{_quant_group(prefix)}"
                    elif packed_format == "nvfp4":
                        packed = _convert_nvfp4(
                            prefix,
                            reader,
                            target_bits=int(quant_params["bits"]),
                            target_group_size=int(quant_params["group_size"]),
                            target_mode=str(quant_params["mode"]),
                        )
                        source_label = f"nvfp4_{_quant_group(prefix)}"
                    else:
                        packed = _convert_packed(prefix, reader)
                        source_label = f"packed_{_quant_group(prefix)}"
                    consumed.update(
                        {
                            f"{prefix}.weight_packed",
                            f"{prefix}.weight_scale",
                            f"{prefix}.weight_zero_point",
                            f"{prefix}.weight_shape",
                            f"{prefix}.weight_global_scale",
                            f"{prefix}.input_global_scale",
                        }
                    )
                    if prefix.startswith("mtp."):
                        mtp_weights.update(packed)
                        output_counts[f"mtp_{_quant_group(prefix)}_int4"] += 1
                    else:
                        module_path = key_mapper(prefix)
                        for packed_key, packed_value in packed.items():
                            _add_main_tensor(
                                out,
                                key_mapper(packed_key),
                                packed_value,
                                main_experts,
                            )
                        if not _is_numbered_expert_module(module_path):
                            quantized_modules.add(module_path)
                            output_counts[f"main_{_quant_group(module_path)}_int4"] += 1
                    source_counts[source_label] += 1
                    continue
                if key.endswith(".qweight"):
                    prefix = key[: -len(".qweight")]
                    packed = _convert_autoawq(prefix, reader)
                    consumed.update(
                        {
                            f"{prefix}.qweight",
                            f"{prefix}.qzeros",
                            f"{prefix}.scales",
                        }
                    )
                    if prefix.startswith("mtp."):
                        mtp_weights.update(packed)
                        output_counts[f"mtp_{_quant_group(prefix)}_int4"] += 1
                        source_counts[f"autoawq_{_quant_group(prefix)}"] += 1
                        continue
                    module_path = key_mapper(prefix)
                    for packed_key, packed_value in packed.items():
                        out_key = key_mapper(packed_key)
                        _add_main_tensor(
                            out,
                            out_key,
                            packed_value,
                            main_experts,
                        )
                    if not _is_numbered_expert_module(module_path):
                        quantized_modules.add(module_path)
                        output_counts[f"main_{_quant_group(module_path)}_int4"] += 1
                    source_counts[f"autoawq_{_quant_group(module_path)}"] += 1
                    continue
                if (
                    key.endswith(".weight_scale")
                    or key.endswith(".weight_zero_point")
                    or key.endswith(".weight_global_scale")
                    or key.endswith(".input_global_scale")
                    or _is_autoawq_aux_key(key, weight_map)
                ):
                    continue
                try:
                    value = reader.tensor(key)
                    if key.startswith("mtp."):
                        mtp_weights[key] = _sanitize_plain_weight(key, value)
                        source_counts["mtp_bf16"] += 1
                        continue
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
        mtp_path = output_path / "mtp.safetensors"
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
    _write_config(
        source_config,
        output_path,
        source_repo=source_repo,
        source_sha=source_sha,
        quant_params=quant_params,
        quantized_modules=quantized_modules,
        mtp_quantized_modules=mtp_quantized_modules,
        stats=stats,
        audit=audit,
        source_format_label=(
            "compressed-tensors-nvfp4-native-w4a16"
            if packed_format == "nvfp4_native"
            else "compressed-tensors-nvfp4-w4a16"
            if packed_format == "nvfp4"
            else "compressed-tensors-awq"
        ),
        policy_name=(
            "forge-compressed-tensors-nvfp4-native-w4a16"
            if packed_format == "nvfp4_native"
            else "forge-compressed-tensors-nvfp4-w4a16"
            if packed_format == "nvfp4"
            else "forge-compressed-tensors-awq"
        ),
        awq_calibrated=packed_format not in {"nvfp4", "nvfp4_native"},
    )
    _write_readme(
        output_path,
        source_repo=source_repo,
        source_sha=source_sha,
        stats=stats,
        audit=audit,
        source_format_label=(
            "compressed-tensors NVFP4 native W4A16"
            if packed_format == "nvfp4_native"
            else "compressed-tensors NVFP4 W4A16"
            if packed_format == "nvfp4"
            else "compressed-tensors AWQ"
        ),
    )
    return {
        "source": source_repo,
        "source_path": str(source_path),
        "source_sha": source_sha,
        "output_path": str(output_path),
        "stats": stats,
        "audit": audit,
    }


def convert_compressed_tensors_nvfp4_requant_affine_to_mlx(
    source_path: Path,
    output_path: Path,
    *,
    source_repo: str,
    source_sha: str | None = None,
    progress_callback: ProgressCallback | None = None,
    target_bits: int = 4,
    target_group_size: int = 64,
    target_mode: str = "affine",
) -> dict[str, Any]:
    """Convert compressed-tensors NVFP4 W4A16 weights to MLX affine.

    NVFP4 stores FP4 payloads plus FP8 block scales and an inverse global
    scale. This fallback path dequantizes each packed module and requantizes
    it into MLX's affine INT4 representation.
    """

    return convert_compressed_tensors_awq_to_mlx(
        source_path,
        output_path,
        source_repo=source_repo,
        source_sha=source_sha,
        progress_callback=progress_callback,
        packed_format="nvfp4",
        target_bits=target_bits,
        target_group_size=target_group_size,
        target_mode=target_mode,
    )


def convert_compressed_tensors_nvfp4_to_mlx(
    source_path: Path,
    output_path: Path,
    *,
    source_repo: str,
    source_sha: str | None = None,
    progress_callback: ProgressCallback | None = None,
    target_bits: int = 4,
    target_group_size: int = 64,
    target_mode: str = "affine",
) -> dict[str, Any]:
    """Backward-compatible alias for the affine requantization fallback."""

    return convert_compressed_tensors_nvfp4_requant_affine_to_mlx(
        source_path,
        output_path,
        source_repo=source_repo,
        source_sha=source_sha,
        progress_callback=progress_callback,
        target_bits=target_bits,
        target_group_size=target_group_size,
        target_mode=target_mode,
    )


def convert_compressed_tensors_nvfp4_native_to_mlx(
    source_path: Path,
    output_path: Path,
    *,
    source_repo: str,
    source_sha: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Convert compressed-tensors NVFP4 W4A16 weights to native MLX nvfp4."""

    return convert_compressed_tensors_awq_to_mlx(
        source_path,
        output_path,
        source_repo=source_repo,
        source_sha=source_sha,
        progress_callback=progress_callback,
        packed_format="nvfp4_native",
        target_bits=4,
        target_group_size=16,
        target_mode="nvfp4",
    )


class _TensorReader:
    def __init__(self, source_path: Path, weight_map: dict[str, str]) -> None:
        self.source_path = source_path
        self.weight_map = weight_map
        self.stack = contextlib.ExitStack()
        self.handles: dict[str, Any] = {}
        self.headers: dict[str, tuple[int, dict[str, Any]]] = {}

    def __enter__(self) -> "_TensorReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()

    def tensor(self, key: str) -> mx.array:
        filename = self.weight_map.get(key)
        if not filename:
            raise KeyError(key)
        metadata = self._tensor_metadata(filename, key)
        if metadata.get("dtype") == "BF16":
            return self._read_bf16_tensor(filename, metadata)
        handle = self.handles.get(filename)
        if handle is None:
            try:
                from safetensors import safe_open
            except Exception as exc:
                raise RuntimeError(
                    f"safetensors is required for compressed-tensors Forge: {exc}"
                ) from exc
            handle = self.stack.enter_context(
                safe_open(str(self.source_path / filename), framework="mlx")
            )
            self.handles[filename] = handle
        return handle.get_tensor(key)

    def raw_tensor(self, key: str) -> tuple[bytes, dict[str, Any]]:
        filename = self.weight_map.get(key)
        if not filename:
            raise KeyError(key)
        metadata = self._tensor_metadata(filename, key)
        header_len, _ = self._header(filename)
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list | tuple)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
        ):
            raise ValueError(f"{filename} has invalid data offsets")
        start, end = offsets
        with (self.source_path / filename).open("rb") as handle:
            handle.seek(8 + header_len + start)
            raw = handle.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"{filename} ended while reading {key}")
        return raw, metadata

    def _tensor_metadata(self, filename: str, key: str) -> dict[str, Any]:
        _, header = self._header(filename)
        metadata = header.get(key)
        if not isinstance(metadata, dict):
            raise KeyError(key)
        return metadata

    def _header(self, filename: str) -> tuple[int, dict[str, Any]]:
        cached = self.headers.get(filename)
        if cached is not None:
            return cached
        path = self.source_path / filename
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        if not isinstance(header, dict):
            raise ValueError(f"{filename} has invalid safetensors header")
        cached = (header_len, header)
        self.headers[filename] = cached
        return cached

    def _read_bf16_tensor(self, filename: str, metadata: dict[str, Any]) -> mx.array:
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError(
                f"numpy is required to read BF16 safetensors: {exc}"
            ) from exc

        header_len, _ = self._header(filename)
        shape = tuple(int(dim) for dim in metadata.get("shape") or ())
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list | tuple)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
        ):
            raise ValueError(f"{filename} has invalid BF16 data offsets")
        start, end = offsets
        expected_bytes = math.prod(shape) * 2
        if end - start != expected_bytes:
            raise ValueError(
                f"{filename} BF16 tensor byte size {end - start} does not match shape {shape}"
            )
        with (self.source_path / filename).open("rb") as handle:
            handle.seek(8 + header_len + start)
            raw = handle.read(end - start)
        if len(raw) != expected_bytes:
            raise ValueError(f"{filename} ended while reading BF16 tensor")
        tensor = mx.array(np.frombuffer(raw, dtype="<u2")).view(mx.bfloat16)
        return tensor.reshape(shape)


def _convert_packed(prefix: str, reader: _TensorReader) -> dict[str, mx.array]:
    packed = reader.tensor(f"{prefix}.weight_packed")
    scale = reader.tensor(f"{prefix}.weight_scale")
    zero_key = f"{prefix}.weight_zero_point"
    out = {
        f"{prefix}.weight": packed.view(mx.uint32),
        f"{prefix}.scales": scale,
    }
    try:
        zero_point = reader.tensor(zero_key)
    except KeyError:
        zero_point = None
    if zero_point is not None:
        zp = _unpack_zero_points(zero_point, scale).astype(scale.dtype)
        out[f"{prefix}.biases"] = -(zp * scale)
    else:
        out[f"{prefix}.biases"] = -8 * scale
    return out


def pack_nvfp4_weight_bytes_to_mlx(packed: mx.array) -> mx.array:
    if packed.ndim < 1:
        raise ValueError("NVFP4 packed weights must have at least one dimension")
    tail = int(packed.shape[-1])
    if tail % 4 != 0:
        raise ValueError(
            f"NVFP4 packed tail dimension must be divisible by 4, got {tail}"
        )
    grouped = packed.astype(mx.uint32).reshape(*packed.shape[:-1], tail // 4, 4)
    shifts = mx.arange(4, dtype=mx.uint32) * 8
    return mx.sum(grouped << shifts, axis=-1).astype(mx.uint32)


def repack_nvfp4_block_scales_to_mlx(scale: mx.array) -> mx.array:
    return scale.astype(mx.uint8)


def global_scale_w_from_inverse_tensor(inverse_scale: mx.array) -> mx.array:
    inv = inverse_scale.astype(mx.float32).reshape(-1)
    return (1.0 / mx.max(inv)).reshape(1)


def convert_fp8_weight_to_q8_0(
    weight: mx.array, *, group_size: int = 32
) -> dict[str, mx.array]:
    dense = weight.astype(mx.float16)
    qweight, qscales, qbiases = mx.quantize(
        dense,
        group_size=int(group_size),
        bits=8,
        mode="affine",
    )
    return {
        "weight": qweight,
        "scales": qscales,
        "biases": qbiases,
    }


def _convert_nvfp4_native(prefix: str, reader: _TensorReader) -> dict[str, mx.array]:
    packed = reader.tensor(f"{prefix}.weight_packed").astype(mx.uint8)
    scale = _read_f8_e4m3_tensor_as_u8(reader, f"{prefix}.weight_scale")
    out = {
        f"{prefix}.weight": pack_nvfp4_weight_bytes_to_mlx(packed),
        f"{prefix}.scales": repack_nvfp4_block_scales_to_mlx(scale),
    }
    for global_key in (f"{prefix}.weight_global_scale", f"{prefix}.weight_scale_2"):
        try:
            inverse = reader.tensor(global_key).astype(mx.float32)
        except KeyError:
            continue
        out[f"{prefix}.global_scale_w"] = global_scale_w_from_inverse_tensor(inverse)
        break
    for input_key in (f"{prefix}.input_global_scale", f"{prefix}.input_scale"):
        try:
            out[f"{prefix}.input_scale"] = (
                reader.tensor(input_key).astype(mx.float32).reshape(1)
            )
        except KeyError:
            continue
        break
    return out


def _read_f8_e4m3_tensor_as_u8(reader: _TensorReader, key: str) -> mx.array:
    raw, metadata = reader.raw_tensor(key)
    shape = tuple(int(dim) for dim in metadata.get("shape") or ())
    if metadata.get("dtype") == "F8_E4M3":
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError(
                f"numpy is required to read F8_E4M3 safetensors: {exc}"
            ) from exc
        byte_values = np.frombuffer(raw, dtype=np.uint8)
        expected = math.prod(shape)
        if byte_values.size != expected:
            raise ValueError(
                f"{key} byte size {byte_values.size} does not match shape {shape}"
            )
        return mx.array(byte_values.reshape(shape), dtype=mx.uint8)
    tensor = reader.tensor(key)
    if tensor.dtype == mx.uint8:
        return tensor
    raise ValueError(
        f"{key} must be raw F8_E4M3 or uint8 for native NVFP4 repack, got {tensor.dtype}"
    )


def _convert_nvfp4(
    prefix: str,
    reader: _TensorReader,
    *,
    target_bits: int,
    target_group_size: int,
    target_mode: str,
) -> dict[str, mx.array]:
    packed = reader.tensor(f"{prefix}.weight_packed").astype(mx.uint8)
    scale = _read_f8_e4m3_tensor(reader, f"{prefix}.weight_scale")
    global_inv = reader.tensor(f"{prefix}.weight_global_scale").astype(mx.float32)
    values = _unpack_nvfp4_e2m1(packed)
    expanded_scale = mx.repeat(scale, repeats=16, axis=-1)
    if tuple(expanded_scale.shape) != tuple(values.shape):
        raise ValueError(
            f"NVFP4 scale shape {expanded_scale.shape} does not match unpacked weights {values.shape}"
        )
    global_scale = 1.0 / mx.max(global_inv)
    weight = (values * expanded_scale * global_scale).astype(mx.float16)
    qweight, qscales, qbiases = mx.quantize(
        weight,
        group_size=int(target_group_size),
        bits=int(target_bits),
        mode=str(target_mode),
    )
    return {
        f"{prefix}.weight": qweight,
        f"{prefix}.scales": qscales,
        f"{prefix}.biases": qbiases,
    }


def _unpack_nvfp4_e2m1(packed: mx.array) -> mx.array:
    table = mx.array(
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
    low = mx.take(table, packed & 0xF)
    high = mx.take(table, (packed >> 4) & 0xF)
    return mx.stack([low, high], axis=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def _read_f8_e4m3_tensor(reader: _TensorReader, key: str) -> mx.array:
    raw, metadata = reader.raw_tensor(key)
    if metadata.get("dtype") != "F8_E4M3":
        return reader.tensor(key).astype(mx.float32)
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            f"numpy is required to read F8_E4M3 safetensors: {exc}"
        ) from exc

    shape = tuple(int(dim) for dim in metadata.get("shape") or ())
    byte_values = np.frombuffer(raw, dtype=np.uint8)
    expected = math.prod(shape)
    if byte_values.size != expected:
        raise ValueError(
            f"{key} byte size {byte_values.size} does not match shape {shape}"
        )
    values = _decode_e4m3fn(byte_values).reshape(shape)
    return mx.array(values, dtype=mx.float32)


def _decode_e4m3fn(byte_values: Any) -> Any:
    import numpy as np

    raw = byte_values.astype(np.uint16)
    sign = np.where(raw & 0x80, -1.0, 1.0).astype(np.float32)
    exponent = ((raw >> 3) & 0xF).astype(np.int16)
    mantissa = (raw & 0x7).astype(np.float32)
    normal = sign * (1.0 + mantissa / 8.0) * np.exp2(exponent.astype(np.float32) - 7.0)
    subnormal = sign * mantissa * (2.0**-9)
    decoded = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    decoded[(exponent == 15) & (mantissa == 7)] = np.nan
    return decoded


_AUTOAWQ_ORDER_MAP = (0, 2, 4, 6, 1, 3, 5, 7)


def _convert_autoawq(prefix: str, reader: _TensorReader) -> dict[str, mx.array]:
    qweight = reader.tensor(f"{prefix}.qweight")
    scales = reader.tensor(f"{prefix}.scales")
    zero_key = f"{prefix}.qzeros"
    unpacked = _unpack_autoawq_int4(qweight)
    out = {
        f"{prefix}.weight": _pack_mlx_int4_rows(unpacked.T),
        f"{prefix}.scales": scales.T,
    }
    try:
        qzeros = reader.tensor(zero_key)
    except KeyError:
        qzeros = None
    if qzeros is not None:
        zeros = _unpack_autoawq_int4(qzeros).astype(scales.dtype).T
        out[f"{prefix}.biases"] = -(zeros * out[f"{prefix}.scales"])
    else:
        out[f"{prefix}.biases"] = -8 * out[f"{prefix}.scales"]
    return out


def _unpack_autoawq_int4(packed: mx.array) -> mx.array:
    if packed.ndim < 1:
        raise ValueError("AutoAWQ packed tensors must have at least one dimension")
    packed_u = packed.astype(mx.uint32)
    columns: list[mx.array | None] = [None] * len(_AUTOAWQ_ORDER_MAP)
    for packed_idx, original_idx in enumerate(_AUTOAWQ_ORDER_MAP):
        columns[original_idx] = (packed_u >> (packed_idx * 4)) & 0xF
    ordered = [column for column in columns if column is not None]
    return mx.stack(ordered, axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def _pack_mlx_int4_rows(values: mx.array) -> mx.array:
    if values.shape[-1] % 8 != 0:
        raise ValueError(f"MLX INT4 packing requires groups of 8, got {values.shape}")
    values_u = values.astype(mx.uint32)
    grouped = values_u.reshape(*values_u.shape[:-1], values_u.shape[-1] // 8, 8)
    shifts = mx.arange(8, dtype=mx.uint32) * 4
    return mx.sum(grouped << shifts, axis=-1).astype(mx.uint32)


def _unpack_zero_points(zero_point: mx.array, scale: mx.array) -> mx.array:
    if zero_point.shape[-1] != scale.shape[-1]:
        raise ValueError(
            f"zero-point groups {zero_point.shape} do not match scales {scale.shape}"
        )
    pack_factor = 8
    if zero_point.shape[0] * pack_factor != scale.shape[0]:
        raise ValueError(
            f"zero-point rows {zero_point.shape} do not unpack to scales {scale.shape}"
        )
    shifts = mx.arange(pack_factor, dtype=mx.uint32) * 4
    unpacked = (zero_point.astype(mx.uint32)[..., None] >> shifts) & 0xF
    return unpacked.transpose(0, 2, 1).reshape(scale.shape)


def _is_autoawq_aux_key(key: str, weight_map: dict[str, str]) -> bool:
    if key.endswith(".qzeros"):
        return f"{key[: -len('.qzeros')]}.qweight" in weight_map
    if key.endswith(".scales"):
        return f"{key[: -len('.scales')]}.qweight" in weight_map
    return False


def _key_mapper_for_config(config: dict[str, Any]) -> Callable[[str], str]:
    model_type = str(
        text_config(config).get("model_type") or config.get("model_type") or ""
    ).lower()
    if model_type in {"glm4_moe", "glm4_moe_lite"}:
        return _mlx_key_identity
    return _mlx_key


def _mlx_key_identity(key: str) -> str:
    return key


def _mlx_key(key: str) -> str:
    if key.startswith("model.language_model"):
        return key.replace("model.language_model", "language_model.model", 1)
    if key.startswith("language_model."):
        return key
    if key.startswith("model.visual."):
        return "vision_tower." + key[len("model.visual.") :]
    if key.startswith("vision_tower."):
        return key
    return "language_model." + key


def _is_numbered_expert_module(module_path: str) -> bool:
    return ".mlp.experts." in module_path


def _add_main_tensor(
    out: dict[str, mx.array],
    key: str,
    value: mx.array,
    expert_accumulator: NumberedExpertAccumulator,
) -> None:
    if expert_accumulator.add(key, value):
        return
    out[key] = value


def _record_stacked_quantized_modules(
    tensors: dict[str, mx.array],
    *,
    quantized_modules: set[str],
    output_counts: Counter[str],
) -> None:
    for key in sorted(tensors):
        if not key.endswith(".weight"):
            continue
        module_path = key[: -len(".weight")]
        if ".switch_mlp." not in module_path:
            continue
        quantized_modules.add(module_path)
        output_counts[f"main_{_quant_group(module_path)}_int4"] += 1


def _quantized_module_prefixes(weights: dict[str, mx.array]) -> set[str]:
    modules: set[str] = set()
    for key in weights:
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len(".weight")]
        if f"{prefix}.scales" in weights:
            modules.add(prefix)
    return modules


def sanitize_plain_weight(key: str, value: mx.array) -> mx.array:
    if key.endswith("conv1d.weight") and value.ndim >= 3 and value.shape[-1] != 1:
        value = value.moveaxis(2, 1)
    if value.ndim == 1:
        if key.startswith("mtp."):
            if any(
                key.endswith(suffix) for suffix in MTP_RMSNORM_ALWAYS_SHIFT_SUFFIXES
            ):
                value = value + 1.0
            elif any(
                key.endswith(suffix) for suffix in MTP_RMSNORM_SHIFT_IF_LOW_SUFFIXES
            ):
                if float(value.mean().item()) < 0.5:
                    value = value + 1.0
        elif any(key.endswith(suffix) for suffix in MAIN_RMSNORM_SHIFT_SUFFIXES):
            value = value + 1.0
    return value


def _sanitize_plain_weight(key: str, value: mx.array) -> mx.array:
    return sanitize_plain_weight(key, value)


def _quant_params_from_config(config: dict[str, Any]) -> dict[str, Any]:
    quant = (
        config.get("quantization_config")
        or text_config(config).get("quantization_config")
        or {}
    )
    if isinstance(quant, dict):
        bits = quant.get("bits")
        group_size = quant.get("group_size")
        if bits is not None and group_size is not None:
            return {
                "bits": int(bits),
                "group_size": int(group_size),
                "mode": "affine",
            }
    groups = quant.get("config_groups") if isinstance(quant, dict) else None
    if isinstance(groups, dict):
        for group in groups.values():
            weights = group.get("weights") if isinstance(group, dict) else None
            if isinstance(weights, dict):
                return {
                    "bits": int(weights.get("num_bits") or 4),
                    "group_size": int(weights.get("group_size") or 32),
                    "mode": "affine",
                }
    return {"bits": 4, "group_size": 32, "mode": "affine"}


def _write_config(
    source_config: dict[str, Any],
    output_path: Path,
    *,
    source_repo: str,
    source_sha: str | None,
    quant_params: dict[str, Any],
    quantized_modules: set[str],
    mtp_quantized_modules: set[str],
    stats: dict[str, Any],
    audit: dict[str, Any],
    source_format_label: str,
    policy_name: str,
    awq_calibrated: bool,
) -> None:
    final_config = dict(source_config)
    tcfg = dict(text_config(source_config))
    tcfg.pop("quantization_config", None)
    tcfg.pop("quantization", None)
    if tcfg:
        final_config["text_config"] = tcfg
    final_config["library_name"] = "mlx"
    final_config.setdefault("language_model_only", False)
    if "vision_config" in final_config and isinstance(
        final_config["vision_config"], dict
    ):
        vision_config = dict(final_config["vision_config"])
        if vision_config.get("model_type") == "qwen3_5_vision":
            vision_config["model_type"] = "qwen3_5"
        vision_config.pop("dtype", None)
        final_config["vision_config"] = vision_config

    quantization: dict[str, Any] = dict(quant_params)
    for module_path in sorted(quantized_modules):
        quantization[module_path] = dict(quant_params)
    final_config["quantization"] = quantization
    final_config["quantization_config"] = quantization
    if mtp_quantized_modules:
        final_config["mtplx_mtp_quantization"] = {
            "policy": "cyankiwi",
            "source_format": source_format_label,
            "prequantized": True,
            **quant_params,
            "quantized_modules": len(mtp_quantized_modules),
        }
    if stats.get("mtp_tensor_count"):
        final_config["mlx_lm_extra_tensors"] = {
            "mtp_file": "mtp.safetensors",
            "mtp_tensor_count": stats["mtp_tensor_count"],
        }
    final_config["mtplx_policy"] = {
        "name": policy_name,
        "source": source_repo,
        "source_sha": source_sha,
        "quantization_family": "converted-compressed-tensors",
        "source_format": source_format_label,
        "awq_calibrated": bool(awq_calibrated),
        "main_quantized_modules": len(quantized_modules),
        "mtp_quantized_modules": len(mtp_quantized_modules),
        "stats": stats,
        "audit": audit,
    }
    _write_json(output_path / "config.json", dict(sorted(final_config.items())))


def _copy_sidecars(source_path: Path, output_path: Path) -> None:
    for name in SIDECAR_FILES:
        src = source_path / name
        if src.exists():
            shutil.copy2(src, output_path / name)
    _processor_config_fallback(output_path)


def _processor_config_fallback(output_path: Path) -> None:
    processor_path = output_path / "processor_config.json"
    if processor_path.exists():
        return
    image_path = output_path / "preprocessor_config.json"
    video_path = output_path / "video_preprocessor_config.json"
    if not image_path.exists() or not video_path.exists():
        return
    image = _load_json(image_path)
    video = _load_json(video_path)
    _write_json(
        processor_path,
        {
            "processor_class": image.get("processor_class", "Qwen3VLProcessor"),
            "image_processor": {
                "image_processor_type": "Qwen2VLImageProcessor",
                "size": image.get("size", {}),
            },
            "video_processor": {
                "video_processor_type": "Qwen3VLVideoProcessor",
                "size": video.get("size", {}),
            },
        },
    )


def _write_readme(
    output_path: Path,
    *,
    source_repo: str,
    source_sha: str | None,
    stats: dict[str, Any],
    audit: dict[str, Any],
    source_format_label: str,
) -> None:
    tags = ["mlx", "mlx-lm", "mtplx", "compressed-tensors"]
    if "awq" in source_format_label.lower():
        tags.append("awq")
    if "nvfp4" in source_format_label.lower():
        tags.append("nvfp4")
    tag_lines = "\n".join(f"- {tag}" for tag in tags)
    readme = f"""---
library_name: mlx
tags:
{tag_lines}
---

# MTPLX Forge {source_format_label} Artifact

Direct MLX conversion of `{source_repo}`.

Forge converts the source repo's `{source_format_label}` packed weights into
MLX affine tensors. The normal Forge verifier must still pass before this
artifact can be treated as launch-ready.

Source revision: `{source_sha or "unknown"}`

```json
{json.dumps({"stats": stats, "audit": audit}, indent=2, sort_keys=True)}
```
"""
    (output_path / "README.md").write_text(readme, encoding="utf-8")


def _audit_conversion(
    *,
    quantized_modules: set[str],
    mtp_quantized_modules: set[str],
    main_index: dict[str, Any],
    mtp_weights: dict[str, mx.array],
) -> dict[str, Any]:
    problems: list[str] = []
    if not quantized_modules:
        problems.append("no compressed-tensors packed modules were converted")
    if (
        "language_model.lm_head.weight" not in main_index["weight_map"]
        and "lm_head.weight" not in main_index["weight_map"]
    ):
        problems.append("missing lm_head.weight")
    if mtp_quantized_modules and not mtp_weights:
        problems.append(
            "MTP quantized modules were detected but mtp.safetensors is empty"
        )
    return {
        "passed": not problems,
        "problems": problems,
        "main_quantized_modules": len(quantized_modules),
        "mtp_quantized_modules": len(mtp_quantized_modules),
        "gdn_quantized_modules": len(
            [path for path in quantized_modules if ".linear_attn." in path]
        ),
        "module_groups": dict(
            sorted(Counter(_quant_group(path) for path in quantized_modules).items())
        ),
        "mtp_module_groups": dict(
            sorted(
                Counter(_quant_group(path) for path in mtp_quantized_modules).items()
            )
        ),
    }


def _quant_group(path: str) -> str:
    if ".linear_attn." in path:
        return "linear_attn"
    if ".self_attn." in path:
        return "self_attn"
    if ".mlp." in path:
        return "mlp"
    if path.startswith("mtp."):
        return "mtp"
    if path.endswith("lm_head"):
        return "lm_head"
    return "other"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _emit(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(payload)
