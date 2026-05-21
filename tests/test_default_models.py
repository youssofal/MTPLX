from __future__ import annotations

import json

import pytest

from mtplx.default_models import (
    DEFAULT_MODEL_VARIANT_ENV,
    OPTIMIZED_SPEED_DESCRIPTION,
    QUALITY_MODEL_ENV,
    SPEED_MODEL_ENV,
    is_verified_default_model_ref,
    optimized_quality_model_ref,
    optimized_speed_model_ref,
    public_model_id_for_ref,
    select_default_model,
)
from mtplx import hardware as hardware_module
from mtplx.hardware import classify_apple_silicon_generation, detect_apple_silicon
from mtplx.profiles import (
    DEFAULT_FP16_HF_MODEL_ID,
    DEFAULT_FP16_PUBLIC_MODEL_ID,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_PUBLIC_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID,
)


def _make_complete_model(path):
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "mtp.safetensors").write_bytes(b"mtp")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"model")
    return path


@pytest.mark.parametrize(
    ("chip", "system", "machine", "expected"),
    [
        ("Apple M1", "Darwin", "arm64", "m1"),
        ("Apple M1 Pro", "Darwin", "arm64", "m1"),
        ("Apple M1 Max", "Darwin", "arm64", "m1"),
        ("Apple M1 Ultra", "Darwin", "arm64", "m1"),
        ("Apple M2", "Darwin", "arm64", "m2"),
        ("Apple M2 Pro", "Darwin", "arm64", "m2"),
        ("Apple M2 Max", "Darwin", "arm64", "m2"),
        ("Apple M2 Ultra", "Darwin", "arm64", "m2"),
        ("Apple M3", "Darwin", "arm64", "m3"),
        ("Apple M3 Max", "Darwin", "arm64", "m3"),
        ("Apple M4", "Darwin", "arm64", "m4"),
        ("Apple M5 Max", "Darwin", "arm64", "m5"),
        ("Intel Core i9", "Darwin", "x86_64", "intel"),
        ("", "Darwin", "arm64", "unknown"),
        ("", "Linux", "x86_64", "unknown"),
    ],
)
def test_classify_apple_silicon_generation(chip, system, machine, expected):
    assert classify_apple_silicon_generation(chip, system=system, machine=machine) == expected


def test_detect_apple_silicon_uses_system_profiler_when_sysctl_is_unparseable(monkeypatch):
    monkeypatch.setattr(hardware_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware_module, "_run_text", lambda *args, **kwargs: "Apple processor")
    monkeypatch.setattr(
        hardware_module,
        "_hardware_json",
        lambda: {"SPHardwareDataType": [{"chip_type": "Apple M2 Max"}]},
    )

    detected = detect_apple_silicon()

    assert detected["chip"] == "Apple M2 Max"
    assert detected["apple_silicon_generation"] == "m2"
    assert detected["is_apple_silicon"] is True


@pytest.mark.parametrize("generation", ["m1", "m2"])
def test_auto_default_uses_fp16_for_m1_m2(monkeypatch, generation):
    monkeypatch.delenv(DEFAULT_MODEL_VARIANT_ENV, raising=False)

    selection = select_default_model(
        hardware={
            "chip": f"Apple {generation.upper()} Max",
            "apple_silicon_generation": generation,
        }
    )

    assert selection.variant == "fp16"
    assert selection.precision == "FP16"
    assert selection.model == DEFAULT_FP16_HF_MODEL_ID
    assert "M1/M2" in selection.reason
    assert selection.auto_selected is True


@pytest.mark.parametrize("generation", ["m3", "m4", "m5", "unknown", "intel"])
def test_auto_default_uses_q4_speed_for_newer_unknown_and_intel(monkeypatch, generation):
    monkeypatch.delenv(DEFAULT_MODEL_VARIANT_ENV, raising=False)
    monkeypatch.setenv(SPEED_MODEL_ENV, "off")

    selection = select_default_model(
        hardware={
            "chip": "Apple M5 Max" if generation == "m5" else "",
            "apple_silicon_generation": generation,
        }
    )

    assert selection.variant == "speed"
    assert selection.precision == OPTIMIZED_SPEED_DESCRIPTION
    assert selection.model == DEFAULT_HF_MODEL_ID
    assert "BF16" not in selection.label
    assert selection.auto_selected is True


def test_default_model_variant_env_override_forces_fp16(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_VARIANT_ENV, "fp16")

    selection = select_default_model(
        hardware={
            "chip": "Apple M5 Max",
            "apple_silicon_generation": "m5",
        }
    )

    assert selection.variant == "fp16"
    assert selection.model == DEFAULT_FP16_HF_MODEL_ID
    assert selection.auto_selected is False


def test_default_model_variant_env_override_legacy_bf16_alias_forces_speed(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_VARIANT_ENV, "bf16")
    monkeypatch.setenv(SPEED_MODEL_ENV, "off")

    selection = select_default_model(
        hardware={
            "chip": "Apple M1 Max",
            "apple_silicon_generation": "m1",
        }
    )

    assert selection.variant == "speed"
    assert selection.precision == OPTIMIZED_SPEED_DESCRIPTION
    assert selection.model == DEFAULT_HF_MODEL_ID
    assert "legacy alias" in selection.reason
    assert selection.auto_selected is False


def test_invalid_default_model_variant_env_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_VARIANT_ENV, "wat")

    selection = select_default_model(
        hardware={
            "chip": "Apple M2 Max",
            "apple_silicon_generation": "m2",
        }
    )

    assert selection.variant == "fp16"
    assert selection.model == DEFAULT_FP16_HF_MODEL_ID
    assert "ignored invalid" in selection.reason


def test_verified_default_refs_include_speed_and_fp16():
    assert is_verified_default_model_ref(DEFAULT_HF_MODEL_ID)
    assert is_verified_default_model_ref(DEFAULT_FP16_HF_MODEL_ID)
    assert not is_verified_default_model_ref(
        "/Users/example/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized"
    )
    assert is_verified_default_model_ref(
        "/Users/example/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed"
    )
    assert is_verified_default_model_ref(
        "/Users/example/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed-FP16"
    )
    assert not is_verified_default_model_ref("someone/custom-model")
    assert not is_verified_default_model_ref("/Users/example/models/custom-model")


def test_optimized_speed_prefers_complete_local_env_model(tmp_path, monkeypatch):
    local_speed = _make_complete_model(tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Speed")
    monkeypatch.setenv(SPEED_MODEL_ENV, str(local_speed))

    selection = select_default_model(
        hardware={
            "chip": "Apple M5 Max",
            "apple_silicon_generation": "m5",
        }
    )

    assert optimized_speed_model_ref() == str(local_speed)
    assert selection.model == str(local_speed)
    assert selection.hf_model == DEFAULT_HF_MODEL_ID
    assert selection.variant == "speed"
    assert selection.precision == OPTIMIZED_SPEED_DESCRIPTION
    assert "installed locally" in selection.reason
    assert "BF16" not in selection.label


def test_optimized_quality_prefers_complete_local_env_model(tmp_path, monkeypatch):
    local_quality = _make_complete_model(tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Quality")
    monkeypatch.setenv(QUALITY_MODEL_ENV, str(local_quality))

    assert optimized_quality_model_ref() == str(local_quality)


@pytest.mark.parametrize(
    ("model_ref", "expected"),
    [
        (
            "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
            DEFAULT_PUBLIC_MODEL_ID,
        ),
        (
            "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
            DEFAULT_FP16_PUBLIC_MODEL_ID,
        ),
        (
            "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Quality",
            QUALITY_PUBLIC_MODEL_ID,
        ),
        (
            "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized",
            LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
        ),
    ],
)
def test_public_model_id_for_ref_maps_known_local_names(model_ref, expected):
    assert public_model_id_for_ref(model_ref) == expected


def test_public_model_id_for_ref_uses_runtime_metadata_before_folder_name(tmp_path):
    model = tmp_path / "whatever-local-folder"
    model.mkdir()
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"artifact_role": "optimized-quality"}),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == QUALITY_PUBLIC_MODEL_ID


def test_public_model_id_for_ref_maps_gdn8_metadata_to_legacy_optimized(tmp_path):
    model = tmp_path / "whatever-local-folder"
    model.mkdir()
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"artifact_role": "gdn8-speed4"}),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == LEGACY_OPTIMIZED_PUBLIC_MODEL_ID


def test_public_model_id_for_ref_maps_mixed_q4_speed_metadata_to_speed(tmp_path):
    model = tmp_path / "Qwen3.6-27B-MTPLX-Optimized"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "bits": 4,
                    "language_model.model.layers.0.mlp.down_proj": {"bits": 4},
                    "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 8},
                }
            }
        ),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == DEFAULT_PUBLIC_MODEL_ID


def test_public_model_id_for_ref_maps_flat8_metadata_to_quality(tmp_path):
    model = tmp_path / "Qwen3.6-27B-MTPLX-Optimized"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "bits": 4,
                    "language_model.model.layers.0.mlp.down_proj": {"bits": 8},
                    "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 8},
                }
            }
        ),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == QUALITY_PUBLIC_MODEL_ID


def test_public_model_id_for_ref_maps_unknown_local_name_to_sanitized_id():
    assert (
        public_model_id_for_ref("/tmp/My Custom Local Model!")
        == "my-custom-local-model"
    )


def test_public_model_id_for_ref_third_party_q4_q8_keeps_folder_name(tmp_path):
    # Regression: a third-party Qwen3.6-35B-A3B build with the same Q4-weights/
    # Q8-head config layout as the Qwen3.6-27B Quality artifact must not be
    # served as `mtplx-qwen36-27b-optimized-quality`.
    model = tmp_path / "Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "bits": 4,
                    "language_model.model.layers.0.mlp.down_proj": {"bits": 8},
                    "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 8},
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        public_model_id_for_ref(model)
        == "qwen3.6-35b-a3b-4bit-mtplx-optimized-speed"
    )
