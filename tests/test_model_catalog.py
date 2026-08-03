from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mtplx.app_settings import (
    APPLE_EPOCH_OFFSET_S,
    read_app_settings,
)
from mtplx.default_models import select_default_model
from mtplx.model_catalog import (
    DISK_MULTIPLIER,
    INTEL_TIER,
    LEGACY_TIER,
    MEMORY_SAFETY_FACTOR,
    MODERN_TIER,
    OFFICIAL_CATALOG,
    UNKNOWN_TIER,
    catalog_model_matching,
    catalog_model_with_id,
    chip_tier_for_generation,
    default_catalog_model,
    evaluate_feasibility,
    recommended_catalog_ids,
    recommended_models,
    scan_installed_models,
)
from mtplx.profiles import (
    DEFAULT_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
)


def test_catalog_has_fifteen_unique_entries():
    ids = [model.id for model in OFFICIAL_CATALOG]
    assert len(ids) == 15
    assert len(set(ids)) == 15
    hf_ids = [model.hf_model_id for model in OFFICIAL_CATALOG]
    assert len(set(hf_ids)) == 15


def test_catalog_matches_swift_official_catalog():
    """Guard the SYNC PAIR: entries must mirror MTPLXModelOption.swift."""

    swift_path = (
        Path(__file__).resolve().parents[1]
        / "apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXModelOption.swift"
    )
    if not swift_path.is_file():
        pytest.skip("app sources not present in this checkout")
    source = swift_path.read_text(encoding="utf-8")
    catalog_block = source.split("officialCatalog: [MTPLXModelOption] = [", 1)[1]
    catalog_block = catalog_block.split("\n    ]", 1)[0]
    swift_entries = list(
        zip(
            re.findall(r'id: "([^"]+)"', catalog_block),
            re.findall(r'hfModelID: "([^"]+)"', catalog_block),
            re.findall(r"sizeBytes: ([0-9_]+)", catalog_block),
            re.findall(r"peakMemoryGiB: ([0-9.]+)", catalog_block),
            re.findall(r"recommendedFor: \[([^\]]*)\]", catalog_block),
        )
    )
    swift_tier_names = {".modernApple": "modern", ".legacyApple": "legacy"}
    assert len(swift_entries) == len(OFFICIAL_CATALOG)
    for python_model, (swift_id, swift_hf, swift_size, swift_peak, swift_tiers) in zip(
        OFFICIAL_CATALOG, swift_entries
    ):
        assert python_model.id == swift_id
        assert python_model.hf_model_id == swift_hf
        assert python_model.size_bytes == int(swift_size.replace("_", ""))
        assert python_model.peak_memory_gib == pytest.approx(float(swift_peak))
        # The tier marker is load-bearing: the app's picker hides any
        # installed official entry whose recommendedFor is empty (the
        # orphan-protection that hid the broken 4B), so a Python-side
        # tier without its Swift mirror makes the model invisible in
        # the app selector on big-RAM Macs (2026-07-19 Quality 4B bug).
        parsed_tiers = frozenset(
            swift_tier_names[token.strip()]
            for token in swift_tiers.split(",")
            if token.strip()
        )
        assert python_model.recommended_tiers == parsed_tiers, swift_id


def test_chip_tier_for_generation():
    assert chip_tier_for_generation("m1") == LEGACY_TIER
    assert chip_tier_for_generation("m2") == LEGACY_TIER
    assert chip_tier_for_generation("m3") == MODERN_TIER
    assert chip_tier_for_generation("m5") == MODERN_TIER
    assert chip_tier_for_generation("intel") == INTEL_TIER
    assert chip_tier_for_generation("") == UNKNOWN_TIER
    assert chip_tier_for_generation(None) == UNKNOWN_TIER


def test_recommended_ids_mirror_app_ram_tiers():
    # Low-RAM tiers carry the rebuilt 4B pair (2026-07-19).
    assert recommended_catalog_ids(memory_gib=8, chip_tier=MODERN_TIER) == [
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    assert recommended_catalog_ids(memory_gib=8, chip_tier=LEGACY_TIER) == [
        "qwen35-9b-optimized-speed-fp16"
    ]
    assert recommended_catalog_ids(memory_gib=24, chip_tier=MODERN_TIER) == [
        "qwen35-9b-optimized-speed",
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    assert recommended_catalog_ids(memory_gib=36, chip_tier=MODERN_TIER) == [
        "optimized-speed-v2",
        "optimized-speed",
        "qwen35-9b-optimized-speed",
        "gemma4-optimized-speed",
        "qwen36-35b-a3b-optimized-speed",
        "optimized-quality",
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    assert recommended_catalog_ids(memory_gib=32, chip_tier=MODERN_TIER)[:2] == [
        "optimized-speed-v2",
        "optimized-speed",
    ]
    assert recommended_catalog_ids(memory_gib=64, chip_tier=MODERN_TIER) == [
        "optimized-speed-v2",
        "optimized-speed",
        "optimized-quality",
        "qwen36-35b-a3b-optimized-speed",
        "qwen36-35b-a3b-optimized-balance",
        "gemma4-optimized-speed",
        "qwen35-9b-optimized-speed",
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    assert recommended_catalog_ids(memory_gib=64, chip_tier=LEGACY_TIER) == [
        "optimized-speed-fp16",
        # Quality on legacy silicon resolves the FP16 sibling (2.0.1,
        # 2026-07-07) so an M1/M2 quality pick gets the fp16-activation
        # artifact the turbo vk q8 kernels were measured on.
        "optimized-quality-fp16",
        "qwen36-35b-a3b-optimized-speed-fp16",
        "qwen36-35b-a3b-optimized-balance-fp16",
        "gemma4-optimized-speed",
        "qwen35-9b-optimized-speed-fp16",
    ]
    assert recommended_catalog_ids(memory_gib=36, chip_tier=LEGACY_TIER) == [
        "qwen35-9b-optimized-speed-fp16",
        "optimized-speed-fp16",
        "gemma4-optimized-speed",
        "qwen36-35b-a3b-optimized-speed-fp16",
        "optimized-quality-fp16",
    ]
    assert recommended_catalog_ids(memory_gib=64, chip_tier=INTEL_TIER) == []
    assert recommended_catalog_ids(
        memory_gib=None, chip_tier=MODERN_TIER
    ) == [
        "optimized-speed-v2",
        "optimized-speed",
        "optimized-quality",
        "qwen36-35b-a3b-optimized-speed",
        "qwen36-35b-a3b-optimized-balance",
        "gemma4-optimized-speed",
        "qwen35-9b-optimized-speed",
    ]


def test_recommended_models_filter_by_peak_memory():
    # An 8 GiB Mac cannot hold the 9B (10 GiB peak) but holds the 4B pair.
    assert [model.id for model in recommended_models(memory_gib=8, chip_tier=MODERN_TIER)] == [
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    models = recommended_models(memory_gib=24, chip_tier=MODERN_TIER)
    assert [model.id for model in models] == [
        "qwen35-9b-optimized-speed",
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    default = default_catalog_model(memory_gib=64, chip_tier=MODERN_TIER)
    assert default is not None and default.id == "optimized-speed-v2"


def test_feasibility_verdicts_mirror_app_rules():
    speed = catalog_model_with_id("optimized-speed")
    assert speed is not None
    assert speed.peak_memory_gib == 17.0

    recommended = evaluate_feasibility(
        speed, chip_tier=MODERN_TIER, ram_gib=64, disk_free_gib=500
    )
    assert recommended.verdict == "recommended"

    tight = evaluate_feasibility(
        speed, chip_tier=MODERN_TIER, ram_gib=24, disk_free_gib=500
    )
    assert tight.verdict == "tight_fit"
    assert 24 < speed.peak_memory_gib * MEMORY_SAFETY_FACTOR

    no_memory = evaluate_feasibility(
        speed, chip_tier=MODERN_TIER, ram_gib=16, disk_free_gib=500
    )
    assert no_memory.verdict == "insufficient_memory"
    assert no_memory.needs_gib == pytest.approx(17.0 * MEMORY_SAFETY_FACTOR)

    no_disk = evaluate_feasibility(
        speed, chip_tier=MODERN_TIER, ram_gib=64, disk_free_gib=10
    )
    assert no_disk.verdict == "insufficient_disk"
    assert no_disk.needs_gib == pytest.approx(
        speed.download_gib * DISK_MULTIPLIER
    )

    intel = evaluate_feasibility(
        speed, chip_tier=INTEL_TIER, ram_gib=128, disk_free_gib=500
    )
    assert intel.verdict == "insufficient_memory"


def test_catalog_model_matching_accepts_ids_repos_cache_dirs_and_aliases():
    speed = catalog_model_with_id("optimized-speed")
    speed_v2 = catalog_model_with_id("optimized-speed-v2")
    assert catalog_model_matching("optimized-speed") == speed
    assert catalog_model_matching(DEFAULT_HF_MODEL_ID) == speed_v2
    assert catalog_model_matching("optimized-speed-v2") == speed_v2
    assert (
        catalog_model_matching("Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed")
        == speed
    )
    assert (
        catalog_model_matching(
            "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
        )
        == speed
    )
    assert catalog_model_matching("mtplx-qwen36-27b-optimized-speed") == speed
    assert (
        catalog_model_matching("mtplx-qwen36-27b-optimized-speed-v2")
        == speed_v2
    )
    assert catalog_model_matching("someone/custom-model") is None
    assert catalog_model_matching("") is None
    assert catalog_model_matching(None) is None


def _write_complete_model(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_scan_installed_models_orders_catalog_first_and_skips_partials(tmp_path):
    cache = tmp_path / "models"
    _write_complete_model(cache / "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed")
    _write_complete_model(cache / "acme--custom-model")
    _write_complete_model(
        cache / "Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed"
    )
    partial = cache / "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality"
    partial.mkdir()
    (partial / "config.json").write_text("{}", encoding="utf-8")
    pair = cache / "paired-bundle"
    pair.mkdir()
    (pair / "mtplx_pair.json").write_text("{}", encoding="utf-8")
    (pair / "target").mkdir()
    (pair / "assistant").mkdir()

    installed = scan_installed_models(cache)

    names = [model.name for model in installed]
    assert names == [
        "Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed",
        "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
        "acme--custom-model",
        "paired-bundle",
    ]
    assert installed[0].catalog is not None
    assert installed[0].catalog.id == "qwen35-9b-optimized-speed"
    assert installed[1].catalog is not None
    assert installed[1].catalog.id == "optimized-speed"
    assert installed[2].catalog is None
    assert installed[2].display_name == "acme/custom-model"
    assert all(model.size_bytes >= 0 for model in installed)


def test_scan_installed_models_handles_missing_cache(tmp_path):
    assert scan_installed_models(tmp_path / "does-not-exist") == []


def test_read_app_settings_parses_snake_case_fields(tmp_path):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "model": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
                "host": "127.0.0.1",
                "port": 8000,
                "api_key": "mtplx-local",
                "last_launch_target": "opencode",
                "onboarding_completed_at": 771_000_000.0,
            }
        ),
        encoding="utf-8",
    )

    settings = read_app_settings(settings_file)

    assert settings is not None
    assert settings.model == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.api_key == "mtplx-local"
    assert settings.last_launch_target == "opencode"
    assert settings.onboarding_completed_at == pytest.approx(
        771_000_000.0 + APPLE_EPOCH_OFFSET_S
    )
    assert settings.onboarding_completed is True


def test_read_app_settings_degrades_to_none(tmp_path):
    assert read_app_settings(tmp_path / "missing.json") is None
    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json", encoding="utf-8")
    assert read_app_settings(garbage) is None
    wrong_shape = tmp_path / "list.json"
    wrong_shape.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_app_settings(wrong_shape) is None
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"model": "  ", "port": "8000"}), encoding="utf-8")
    settings = read_app_settings(partial)
    assert settings is not None
    assert settings.model is None
    assert settings.port is None
    assert settings.onboarding_completed is False


def test_select_default_model_routes_small_macs_to_9b(monkeypatch):
    monkeypatch.delenv("MTPLX_DEFAULT_MODEL_VARIANT", raising=False)

    small_modern = select_default_model(
        hardware={
            "chip": "Apple M4",
            "apple_silicon_generation": "m4",
            "memory_gib": 16.0,
        }
    )
    assert small_modern.model == QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID
    assert small_modern.hf_model == QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID
    assert small_modern.variant == "speed"
    assert "9B" in small_modern.reason
    assert small_modern.display_name == "Qwen3.5 9B Optimized Speed"
    assert small_modern.memory_gib == 16.0

    small_legacy = select_default_model(
        hardware={
            "chip": "Apple M1 Max",
            "apple_silicon_generation": "m1",
            "memory_gib": 16.0,
        }
    )
    assert small_legacy.model == QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID
    assert small_legacy.variant == "fp16"
    assert small_legacy.display_name == "Qwen3.5 9B Optimized Speed FP16"


def test_select_default_model_keeps_27b_with_enough_memory(monkeypatch):
    monkeypatch.delenv("MTPLX_DEFAULT_MODEL_VARIANT", raising=False)
    monkeypatch.setenv("MTPLX_OPTIMIZED_SPEED_MODEL", "off")

    selection = select_default_model(
        hardware={
            "chip": "Apple M4",
            "apple_silicon_generation": "m4",
            "memory_gib": 64.0,
        }
    )
    assert selection.model == DEFAULT_HF_MODEL_ID
    assert "9B" not in selection.reason


def test_select_default_model_uses_v2_at_32_gib_and_above(monkeypatch):
    monkeypatch.delenv("MTPLX_DEFAULT_MODEL_VARIANT", raising=False)
    monkeypatch.setenv("MTPLX_OPTIMIZED_SPEED_MODEL", "off")

    base_hardware = {
        "chip": "Apple M4 Max",
        "apple_silicon_generation": "m4",
    }
    at_32 = select_default_model(hardware={**base_hardware, "memory_gib": 32.0})
    at_36 = select_default_model(hardware={**base_hardware, "memory_gib": 36.0})

    assert at_32.model == DEFAULT_HF_MODEL_ID
    assert at_36.model == DEFAULT_HF_MODEL_ID


def test_select_default_model_without_memory_keeps_generation_policy(monkeypatch):
    monkeypatch.delenv("MTPLX_DEFAULT_MODEL_VARIANT", raising=False)
    monkeypatch.setenv("MTPLX_OPTIMIZED_SPEED_MODEL", "off")

    selection = select_default_model(
        hardware={
            "chip": "Apple M4",
            "apple_silicon_generation": "m4",
        }
    )
    assert selection.model == DEFAULT_HF_MODEL_ID
    assert selection.memory_gib is None
