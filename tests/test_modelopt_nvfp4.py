from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from mtplx.compressed_tensors import (
    convert_compressed_tensors_nvfp4_native_to_mlx,
    global_scale_w_from_inverse_tensor,
    pack_nvfp4_weight_bytes_to_mlx,
    repack_nvfp4_block_scales_to_mlx,
)
from mtplx.modelopt_nvfp4 import (
    _convert_modelopt_nvfp4,
    _read_modelopt_scale_bytes,
    convert_modelopt_nvfp4_to_mlx,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_raw_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    offset = 0
    header: dict[str, dict] = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = offset
        payload.extend(raw)
        offset += len(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, offset],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


class _FakeReader:
    def __init__(
        self,
        tensors: dict[str, mx.array],
        raw: dict[str, tuple[dict, bytes]] | None = None,
    ) -> None:
        self.tensors = tensors
        self.raw = raw or {}

    def tensor(self, key: str) -> mx.array:
        return self.tensors[key]

    def raw_tensor(self, key: str) -> tuple[bytes, dict]:
        metadata, raw = self.raw[key]
        return raw, metadata


def test_pack_nvfp4_weight_bytes_to_mlx_packs_eight_nibbles_per_uint32():
    packed = mx.array([0x21, 0x43, 0x65, 0x87], dtype=mx.uint8)
    out = pack_nvfp4_weight_bytes_to_mlx(packed)
    assert out.dtype == mx.uint32
    assert tuple(out.shape) == (1,)
    assert int(out.item()) == int(0x87654321)


def test_repack_nvfp4_block_scales_preserves_raw_bytes():
    scales = mx.array([[18, 19, 16, 15]], dtype=mx.uint8)
    out = repack_nvfp4_block_scales_to_mlx(scales)
    assert out.dtype == mx.uint8
    assert mx.array_equal(out, scales)


def test_global_scale_w_from_inverse_tensor_matches_three_tier_recipe():
    inverse = mx.array([2.0], dtype=mx.float32)
    out = global_scale_w_from_inverse_tensor(inverse)
    assert tuple(out.shape) == (1,)
    assert abs(float(out.item()) - 0.5) < 1e-6


def test_convert_modelopt_nvfp4_repacks_without_affine_fields():
    prefix = "model.language_model.layers.0.self_attn.q_proj"
    packed = mx.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=mx.uint8)
    inverse = mx.array([4.0], dtype=mx.float32)
    reader = _FakeReader(
        {
            f"{prefix}.weight": packed,
            f"{prefix}.weight_scale_2": inverse,
            f"{prefix}.input_scale": mx.array([1.0], dtype=mx.float32),
        },
        raw={
            f"{prefix}.weight_scale": (
                {"dtype": "F8_E4M3", "shape": [1, 2]},
                np.array([[18, 19]], dtype=np.uint8).tobytes(),
            ),
        },
    )
    out = _convert_modelopt_nvfp4(prefix, reader)
    assert f"{prefix}.weight" in out
    assert f"{prefix}.scales" in out
    assert f"{prefix}.global_scale_w" in out
    assert f"{prefix}.input_scale" in out
    assert "biases" not in "".join(out)
    assert out[f"{prefix}.weight"].dtype == mx.uint32
    assert out[f"{prefix}.scales"].dtype == mx.uint8
    assert abs(float(out[f"{prefix}.global_scale_w"].item()) - 0.25) < 1e-6


def test_read_modelopt_scale_bytes_reads_f8_e4m3_payload():
    scales = mx.array([[18, 19]], dtype=mx.uint8)
    reader = _FakeReader(
        {},
        raw={
            "module.weight_scale": (
                {"dtype": "F8_E4M3", "shape": [1, 2]},
                np.array([[18, 19]], dtype=np.uint8).tobytes(),
            )
        },
    )
    out = _read_modelopt_scale_bytes(reader, "module.weight_scale")
    assert out.dtype == mx.uint8
    assert mx.array_equal(out, scales)


def test_modelopt_converter_writes_native_nvfp4_and_mtp_sidecar(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    prefix = "model.language_model.layers.0.self_attn.q_proj"
    mtp_key = "mtp.layers.0.enorm.weight"
    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1},
        },
    )
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                f"{prefix}.weight": "model.safetensors",
                f"{prefix}.weight_scale": "model.safetensors",
                f"{prefix}.weight_scale_2": "model.safetensors",
                f"{prefix}.input_scale": "model.safetensors",
                mtp_key: "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
        },
    )
    _write_raw_safetensors(
        source / "model.safetensors",
        {
            f"{prefix}.weight": (
                "U8",
                (1, 8),
                np.array(
                    [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=np.uint8
                ).tobytes(),
            ),
            f"{prefix}.weight_scale": (
                "F8_E4M3",
                (1, 2),
                np.array([18, 19], dtype=np.uint8).tobytes(),
            ),
            f"{prefix}.weight_scale_2": (
                "F32",
                (1,),
                np.array([2.0], dtype="<f4").tobytes(),
            ),
            f"{prefix}.input_scale": (
                "F32",
                (1,),
                np.array([1.0], dtype="<f4").tobytes(),
            ),
            mtp_key: (
                "BF16",
                (4,),
                np.array([1, 2, 3, 4], dtype="<u2").tobytes(),
            ),
            "lm_head.weight": (
                "F32",
                (2, 2),
                np.ones((2, 2), dtype="<f4").tobytes(),
            ),
        },
    )

    report = convert_modelopt_nvfp4_to_mlx(
        source,
        output,
        source_repo="nvidia/Qwen3.6-27B-NVFP4",
        source_sha="rev",
    )
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    mtp = mx.load(str(output / "mtp" / "weights.safetensors"))

    assert report["audit"]["passed"] is True
    assert (
        "language_model.model.layers.0.self_attn.q_proj.weight" in index["weight_map"]
    )
    assert mtp_key in mtp
    assert config["quantization"]["mode"] == "nvfp4"
    assert config["quantization"]["group_size"] == 16
    assert "language_model.model.layers.0.self_attn.q_proj" in config["quantization"]
    assert config["mlx_lm_extra_tensors"]["mtp_file"] == "mtp/weights.safetensors"
    assert config["mtplx_policy"]["source_format"] == "modelopt-nvfp4-native-w4a16"


def test_compressed_tensors_native_converter_emits_nvfp4_without_biases(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    prefix = "model.language_model.layers.0.self_attn.q_proj"
    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
            },
        },
    )
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                f"{prefix}.weight_packed": "model.safetensors",
                f"{prefix}.weight_scale": "model.safetensors",
                f"{prefix}.weight_global_scale": "model.safetensors",
                f"{prefix}.input_global_scale": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
        },
    )
    _write_raw_safetensors(
        source / "model.safetensors",
        {
            f"{prefix}.weight_packed": (
                "U8",
                (1, 8),
                np.array(
                    [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=np.uint8
                ).tobytes(),
            ),
            f"{prefix}.weight_scale": (
                "F8_E4M3",
                (1, 2),
                np.array([18, 19], dtype=np.uint8).tobytes(),
            ),
            f"{prefix}.weight_global_scale": (
                "F32",
                (1,),
                np.array([2.0], dtype="<f4").tobytes(),
            ),
            f"{prefix}.input_global_scale": (
                "F32",
                (1,),
                np.array([1.0], dtype="<f4").tobytes(),
            ),
            "lm_head.weight": (
                "F32",
                (2, 2),
                np.ones((2, 2), dtype="<f4").tobytes(),
            ),
        },
    )

    report = convert_compressed_tensors_nvfp4_native_to_mlx(
        source,
        output,
        source_repo="owner/nvfp4-native",
        source_sha="rev",
    )
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_name = index["weight_map"][
        "language_model.model.layers.0.self_attn.q_proj.weight"
    ]
    converted = mx.load(str(output / shard_name))

    assert report["audit"]["passed"] is True
    assert config["quantization"]["mode"] == "nvfp4"
    assert config["quantization"]["group_size"] == 16
    assert "language_model.model.layers.0.self_attn.q_proj.biases" not in converted
    assert (
        converted["language_model.model.layers.0.self_attn.q_proj.scales"].dtype
        == mx.uint8
    )
