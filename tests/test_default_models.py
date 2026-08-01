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
    QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
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


def test_optimized_quality_routes_fp16_sibling_on_legacy_silicon(monkeypatch):
    """A quality pick on M1/M2 resolves the Quality-FP16 sibling (2.0.1),
    mirroring the speed lane's precision routing."""
    from mtplx.profiles import QUALITY_FP16_HF_MODEL_ID, QUALITY_HF_MODEL_ID

    monkeypatch.delenv(QUALITY_MODEL_ENV, raising=False)
    legacy = {"chip": "Apple M1 Pro", "apple_silicon_generation": "m1"}
    modern = {"chip": "Apple M5 Max", "apple_silicon_generation": "m5"}

    legacy_ref = optimized_quality_model_ref(hardware=legacy)
    assert "Quality-FP16" in legacy_ref or legacy_ref == QUALITY_FP16_HF_MODEL_ID

    modern_ref = optimized_quality_model_ref(hardware=modern)
    assert "FP16" not in modern_ref or modern_ref == QUALITY_HF_MODEL_ID


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
            "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
            "mtplx-qwen36-27b-optimized-quality-fp16",
        ),
        (
            "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized",
            LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
        ),
        # HF repo id given directly, and the HF cache-dir path shape the
        # resolver historically needed substring matching for — both must
        # keep mapping to the first-party id under component equality.
        (
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            DEFAULT_PUBLIC_MODEL_ID,
        ),
        (
            "/Users/example/.cache/huggingface/hub/"
            "models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed/"
            "snapshots/abc1234def",
            DEFAULT_PUBLIC_MODEL_ID,
        ),
    ],
)
def test_public_model_id_for_ref_maps_known_local_names(model_ref, expected):
    assert public_model_id_for_ref(model_ref) == expected


@pytest.mark.parametrize(
    ("model_ref", "expected"),
    [
        # A derivative artifact whose folder name EXTENDS a first-party
        # name must serve under its own identity, not the flagship's:
        # substring matching made a V3-RC build report itself as
        # mtplx-qwen36-27b-optimized-speed (health payload + app model
        # chip both lied, reported live 2026-07-31).
        (
            "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Speed-V3-RC",
            "qwen3.6-27b-mtplx-optimized-speed-v3-rc",
        ),
        (
            "/Users/example/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality-V2",
            "youssofal-qwen3.6-27b-mtplx-optimized-quality-v2",
        ),
        (
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V3-RC",
            "qwen3.6-27b-mtplx-optimized-speed-v3-rc",
        ),
    ],
)
def test_public_model_id_for_ref_refuses_derivative_name_collisions(
    model_ref, expected
):
    assert public_model_id_for_ref(model_ref) == expected


def test_public_model_id_for_ref_uses_explicit_runtime_id_before_folder_name(tmp_path):
    model = tmp_path / "whatever-local-folder"
    model.mkdir()
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"public_model_id": QUALITY_PUBLIC_MODEL_ID}),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == QUALITY_PUBLIC_MODEL_ID


def test_public_model_id_for_ref_ignores_artifact_role_substrings(tmp_path):
    """artifact_role is written by MTPLX tooling for third-party builds too,
    so a "quality"/"speed"/"gdn8" substring is NOT proof of first-party
    identity (the July 2026 contract-match-only fix, issue #57 class)."""
    for role in ("optimized-quality", "gdn8-speed4", "maximum-speed"):
        model = tmp_path / f"whatever-{role}-folder"
        model.mkdir()
        (model / "mtplx_runtime.json").write_text(
            json.dumps({"artifact_role": role}),
            encoding="utf-8",
        )
        assert public_model_id_for_ref(model) == f"whatever-{role}-folder"


def test_public_model_id_for_ref_ignores_precision_variant_coercion(tmp_path):
    model = tmp_path / "SomeFinetune-FP16"
    model.mkdir()
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"precision_variant": "fp16"}),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == "somefinetune-fp16"


def test_public_model_id_for_ref_ignores_verified_on_inference(tmp_path):
    """The nom666 Qwopus repro: a forge-built third-party artifact whose
    verified_on.model contains "Speed" must keep its own identity."""
    model = tmp_path / "Qwopus3.6-27B-Coder-MTPLX-4bit-Speed"
    model.mkdir()
    (model / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "artifact_role": "forge-local",
                "verified_on": {"model": "Qwopus3.6-27B-Coder-MTPLX-4bit-Speed"},
            }
        ),
        encoding="utf-8",
    )
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "quantization": {"bits": 4},
            }
        ),
        encoding="utf-8",
    )

    assert (
        public_model_id_for_ref(model) == "qwopus3.6-27b-coder-mtplx-4bit-speed"
    )


def test_public_model_id_for_ref_does_not_map_small_speed_role_to_27b(tmp_path):
    model = tmp_path / "Qwen3.5-4B-MTPLX-Optimized-Speed"
    model.mkdir()
    (model / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "artifact_role": "small-q4-speed-test",
                "verified_on": {"model": "Qwen3.5-4B-MTPLX-Optimized-Speed"},
            }
        ),
        encoding="utf-8",
    )
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3_5",
                "quantization": {"bits": 4},
            }
        ),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == "qwen3.5-4b-mtplx-optimized-speed"


def test_public_model_id_for_ref_no_quantization_upgrade_of_legacy_name(tmp_path):
    """Issue #57: the user loaded Qwen3.6-27B-MTPLX-Optimized (the legacy
    artifact) and the CLI reported mtplx-qwen36-27b-optimized-speed because
    the quantization layout was "upgrading" the identity. The served id
    must match the artifact the user actually selected, regardless of its
    quantization layout."""
    for layout in (
        {  # Q4 layout — used to coerce to the speed id
            "bits": 4,
            "language_model.model.layers.0.mlp.down_proj": {"bits": 4},
            "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 8},
        },
        {  # Flat8-style layout — used to coerce to the quality id
            "bits": 4,
            "language_model.model.layers.0.mlp.down_proj": {"bits": 8},
            "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 8},
        },
    ):
        model = tmp_path / f"case-{layout['language_model.model.layers.0.mlp.down_proj']['bits']}" / "Qwen3.6-27B-MTPLX-Optimized"
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps({"quantization": layout}),
            encoding="utf-8",
        )
        assert public_model_id_for_ref(model) == LEGACY_OPTIMIZED_PUBLIC_MODEL_ID


def test_public_model_id_for_ref_keeps_qwen36_35b_identity(tmp_path):
    model = tmp_path / "Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "quantization": {
                    "bits": 4,
                    "language_model.model.layers.0.mlp.down_proj": {"bits": 4},
                },
            }
        ),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID


@pytest.mark.parametrize(
    ("name", "expected_id"),
    [
        (
            "Qwen3.5-9B-MTPLX-Optimized-Speed",
            QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        ),
        (
            "Qwen-Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI",
            QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        ),
        (
            "Qwen3.5-9B-MTPLX-Optimized-Speed-FP16",
            QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
        ),
        (
            "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16",
            QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
        ),
        (
            "Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
            QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID,
        ),
        (
            "Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16",
            QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID,
        ),
    ],
)
def test_public_model_id_for_ref_maps_release_catalog_names(name, expected_id, tmp_path):
    model = tmp_path / name
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "quantization": {"bits": 6},
            }
        ),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == expected_id


def test_public_model_id_for_ref_does_not_map_custom_qwen_to_27b(tmp_path):
    model = tmp_path / "Acme-Qwen3.6-Custom-MTP-Speed"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "quantization": {"bits": 4},
            }
        ),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == "acme-qwen3.6-custom-mtp-speed"


def test_public_model_id_for_ref_does_not_map_step_quantization_to_qwen(tmp_path):
    model = tmp_path / "Step-3.7-Flash-MTPLX-step3p5"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Step3p5ForCausalLM"],
                "model_type": "step3p5",
                "quantization": {
                    "bits": 4,
                    "language_model.model.layers.0.mlp.down_proj": {"bits": 8},
                    "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 8},
                },
            }
        ),
        encoding="utf-8",
    )

    assert public_model_id_for_ref(model) == "step-3.7-flash-mtplx-step3p5"


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # nom666 Qwopus builds (real third-party HF repos observed
        # mislabeled in the June 2026 triage).
        (
            "nom666/Qwopus3.6-27B-Coder-MTPLX-4bit-Speed",
            "qwopus3.6-27b-coder-mtplx-4bit-speed",
        ),
        (
            "nom666/Qwopus3.6-27B-Coder-MTPLX-8bit-Quality",
            "qwopus3.6-27b-coder-mtplx-8bit-quality",
        ),
        # samuelfaj 35B case from PR #77.
        (
            "samuelfaj/Qwopus3.6-35B-A3B-v1-8bit-MTPLX-Optimized-Speed",
            "qwopus3.6-35b-a3b-v1-8bit-mtplx-optimized-speed",
        ),
        # A third-party remix that merely mentions the 35B family + MTPLX
        # must NOT be claimed as the first-party 35B artifact (the removed
        # family-name coercion).
        (
            "someguy/Qwen3.6-35B-A3B-MTPLX-Remix",
            "qwen3.6-35b-a3b-mtplx-remix",
        ),
    ],
)
def test_public_model_id_for_ref_keeps_third_party_identity(ref, expected):
    assert public_model_id_for_ref(ref) == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed", DEFAULT_PUBLIC_MODEL_ID),
        (
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
            DEFAULT_FP16_PUBLIC_MODEL_ID,
        ),
        ("Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality", QUALITY_PUBLIC_MODEL_ID),
        ("Youssofal/Qwen3.6-27B-MTPLX-Optimized", LEGACY_OPTIMIZED_PUBLIC_MODEL_ID),
        (
            "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
            QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        ),
        (
            "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed",
            QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        ),
        (
            "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
            DEFAULT_PUBLIC_MODEL_ID,
        ),
    ],
)
def test_public_model_id_for_ref_first_party_matrix(ref, expected):
    assert public_model_id_for_ref(ref) == expected


def test_public_model_id_for_ref_maps_unknown_local_name_to_sanitized_id():
    assert (
        public_model_id_for_ref("/tmp/My Custom Local Model!")
        == "my-custom-local-model"
    )
