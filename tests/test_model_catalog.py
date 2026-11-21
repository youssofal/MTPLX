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
    DOWNLOAD_HEADROOM_GIB,
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
from mtplx import default_models as default_models_module
from mtplx.default_models import QWEN38_OPTIMIZED_SPEED_MODEL_ENV
from mtplx.profiles import (
    DEFAULT_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
)


@pytest.fixture(autouse=True)
def _no_installed_qwen38(monkeypatch):
    # Pin the public policy: a complete local Qwen 3.8 build on this Mac is
    # legitimately preferred ("installed locally"), so switch it off here.
    monkeypatch.setenv(QWEN38_OPTIMIZED_SPEED_MODEL_ENV, "off")
    monkeypatch.setattr(default_models_module, "_QWEN38_OPTIMIZED_SPEED_FP16_LOCAL_CANDIDATES", ())


def test_catalog_has_thirty_two_unique_entries():
    ids = [model.id for model in OFFICIAL_CATALOG]
    assert len(ids) == len(set(ids)) == 32
    assert len({model.hf_model_id for model in OFFICIAL_CATALOG}) == 32


def test_inkling_trio_follows_each_tiers_first_pick_where_the_plan_admits_it():
    inkling = ["inkling2-mini-bare-speed", "inkling2-mini-optimized-speed", "inkling2-mini-optimized-quality"]
    for ram in (32, 36, 40):
        ids = recommended_catalog_ids(memory_gib=ram, chip_tier=MODERN_TIER)
        assert ids[0] == "qwen38-27b-optimized-speed"
        assert ids[1:4] == inkling
    for ram in (16, 18):
        assert not set(inkling) & set(recommended_catalog_ids(memory_gib=ram, chip_tier=MODERN_TIER))
    assert set(inkling) & set(recommended_catalog_ids(memory_gib=24, chip_tier=MODERN_TIER)) == {inkling[0]}
    for ram in (64, 128):
        ids = recommended_catalog_ids(memory_gib=ram, chip_tier=MODERN_TIER)
        assert ids[ids.index("bonsai-2-27b-optimized-speed") + 1 :][:3] == inkling
    for ram in (32, 64):
        assert not set(inkling) & set(recommended_catalog_ids(memory_gib=ram, chip_tier=LEGACY_TIER))
    for model_id in inkling:
        model = catalog_model_with_id(model_id)
        assert model.recommended_tiers == frozenset({MODERN_TIER})
        # The house form of every MTPLX pack repository: <Model>-MTPLX-<Tier>.
        tier = model.display_name.removeprefix("Inkling 2 Mini ").replace(" ", "-")
        assert model.hf_model_id == f"Youssofal/Inkling-2-Mini-MTPLX-{tier}"



def test_the_24_gib_inkling_offer_is_on(monkeypatch):
    import mtplx.model_catalog as catalog

    assert catalog.INKLING_TIGHT_MAC_OFFER is True
    for ram in (24, 28):
        ids = recommended_catalog_ids(memory_gib=ram, chip_tier=MODERN_TIER)
        assert ids[1] == "inkling2-mini-bare-speed"  # directly after the tier's first pick
        assert ids[0] == "bonsai-2-27b-optimized-speed"  # first pick unchanged
        assert "inkling2-mini-optimized-speed" not in ids and "inkling2-mini-optimized-quality" not in ids
    assert "inkling2-mini-bare-speed" not in recommended_catalog_ids(memory_gib=18, chip_tier=MODERN_TIER)
    assert "inkling2-mini-bare-speed" not in recommended_catalog_ids(memory_gib=24, chip_tier=LEGACY_TIER)
    monkeypatch.setattr(catalog, "INKLING_TIGHT_MAC_OFFER", False)
    for ram in (24, 28):
        assert "inkling2-mini-bare-speed" not in recommended_catalog_ids(memory_gib=ram, chip_tier=MODERN_TIER)

def test_qwen38_fp16_siblings_mirror_their_parents():
    # Same packs, same peak; only the tier, id suffix and HF repo differ.
    for base in ("qwen38-27b-bare-speed", "qwen38-27b-optimized-speed", "qwen38-27b-optimized-quality"):
        parent = catalog_model_with_id(base)
        sibling = catalog_model_with_id(f"{base}-fp16")
        assert parent is not None and sibling is not None
        assert sibling.hf_model_id == f"{parent.hf_model_id}-FP16"
        assert sibling.peak_memory_gib == parent.peak_memory_gib
        assert sibling.recommended_tiers == frozenset({LEGACY_TIER})
        assert parent.recommended_tiers == frozenset({MODERN_TIER})
        assert "FP16 build for M1 and M2 Macs" in sibling.detail
        assert f"mtplx-{base}-fp16" in sibling.aliases


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
    swift_details = re.findall(r'localizedDetailKey: "([^"\n]+)"', catalog_block)
    assert swift_details == [model.detail for model in OFFICIAL_CATALOG]
    assert all("recommended" not in model.detail.lower() for model in OFFICIAL_CATALOG)
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
        "bonsai-2-27b-optimized-speed",
        "inkling2-mini-bare-speed",
        "mimo-v26-qwen-9b-optimized-speed",
        "qwen35-9b-optimized-speed",
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    # Qwen 3.8 trio (2026-08-15): Optimized Speed leads every tier with at
    # least 32 GiB, then Bare Speed, then Optimized Quality.
    trio38 = [
        "qwen38-27b-optimized-speed",
        "qwen38-27b-bare-speed",
        "qwen38-27b-optimized-quality",
    ]
    trio38_fp16 = [f"{model_id}-fp16" for model_id in trio38]
    inkling_fp16 = [
        "inkling2-mini-bare-speed-fp16",
        "inkling2-mini-optimized-speed-fp16",
        "inkling2-mini-optimized-quality-fp16",
    ]
    assert recommended_catalog_ids(memory_gib=36, chip_tier=MODERN_TIER) == [
        trio38[0],
        "inkling2-mini-bare-speed",
        "inkling2-mini-optimized-speed",
        "inkling2-mini-optimized-quality",
        *trio38[1:],
        "optimized-speed-v2",
        "optimized-speed",
        "mimo-v26-qwen-9b-optimized-speed",
        "qwen35-9b-optimized-speed",
        "gemma4-optimized-speed",
        "qwen36-35b-a3b-optimized-speed",
        "optimized-quality",
        "bonsai-2-27b-optimized-speed",
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    assert recommended_catalog_ids(memory_gib=32, chip_tier=MODERN_TIER)[0] == trio38[0]
    assert recommended_catalog_ids(memory_gib=64, chip_tier=MODERN_TIER) == [
        *trio38,
        "optimized-speed-v2",
        "optimized-speed",
        "optimized-quality",
        "qwen36-35b-a3b-optimized-speed",
        "qwen36-35b-a3b-optimized-balance",
        "gemma4-optimized-speed",
        "mimo-v26-qwen-9b-optimized-speed",
        "qwen35-9b-optimized-speed",
        "bonsai-2-27b-optimized-speed",
        "inkling2-mini-bare-speed",
        "inkling2-mini-optimized-speed",
        "inkling2-mini-optimized-quality",
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    # Legacy (M1/M2) silicon sees the same trio as its FP16 precision
    # siblings, same order, ahead of the 3.6 fp16 lane.
    assert recommended_catalog_ids(memory_gib=64, chip_tier=LEGACY_TIER) == [
        *trio38_fp16,
        "optimized-speed-fp16",
        # Quality on legacy silicon resolves the FP16 sibling (2.0.1,
        # 2026-07-07) so an M1/M2 quality pick gets the fp16-activation
        # artifact the turbo vk q8 kernels were measured on.
        "optimized-quality-fp16",
        "qwen36-35b-a3b-optimized-speed-fp16",
        "qwen36-35b-a3b-optimized-balance-fp16",
        "gemma4-optimized-speed",
        "qwen35-9b-optimized-speed-fp16",
        # The Inkling 2 Mini FP16 siblings keep the family's slot near the end (INKLING_LEGACY_OFFER).
        *inkling_fp16,
    ]
    # From 32 to 47 GiB they ride directly after the tier's first pick, as the modern trio does.
    assert recommended_catalog_ids(memory_gib=36, chip_tier=LEGACY_TIER) == [
        trio38_fp16[0],
        *inkling_fp16,
        *trio38_fp16[1:],
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
        *trio38,
        "flash-next-bare-speed",
        "flash-next-optimized-speed",
        "flash-next-optimized-quality",
        "optimized-speed-v2",
        "optimized-speed",
        "optimized-quality",
        "qwen36-35b-a3b-optimized-speed",
        "qwen36-35b-a3b-optimized-balance",
        "gemma4-optimized-speed",
        "mimo-v26-qwen-9b-optimized-speed",
        "qwen35-9b-optimized-speed",
        "bonsai-2-27b-optimized-speed",
    ]
    assert recommended_catalog_ids(
        memory_gib=None, chip_tier=LEGACY_TIER
    )[:3] == trio38_fp16


def test_recommended_models_filter_by_peak_memory():
    # An 8 GiB Mac cannot hold the 9B (10 GiB peak) but holds the 4B pair.
    assert [model.id for model in recommended_models(memory_gib=8, chip_tier=MODERN_TIER)] == [
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    models = recommended_models(memory_gib=24, chip_tier=MODERN_TIER)
    assert [model.id for model in models] == [
        "bonsai-2-27b-optimized-speed",
        "inkling2-mini-bare-speed",
        "mimo-v26-qwen-9b-optimized-speed",
        "qwen35-9b-optimized-speed",
        "qwen35-4b-optimized-speed",
        "qwen35-4b-optimized-quality",
    ]
    default = default_catalog_model(memory_gib=64, chip_tier=MODERN_TIER)
    assert default is not None and default.id == "qwen38-27b-optimized-speed"
    # Legacy (M1/M2) silicon defaults to the FP16 precision sibling; a
    # 32 GiB M1/M2 keeps it (25 GiB peak fits) while Optimized Quality
    # (33 GiB peak) drops out until 48 GiB.
    legacy_default = default_catalog_model(memory_gib=32, chip_tier=LEGACY_TIER)
    assert legacy_default is not None and legacy_default.id == "qwen38-27b-optimized-speed-fp16"
    legacy_32 = [model.id for model in recommended_models(memory_gib=32, chip_tier=LEGACY_TIER)]
    assert legacy_32[:3] == [
        "qwen38-27b-optimized-speed-fp16",
        "inkling2-mini-bare-speed-fp16",
        "inkling2-mini-optimized-speed-fp16",
    ]
    assert "qwen38-27b-optimized-quality-fp16" not in legacy_32
    assert "qwen38-27b-optimized-quality-fp16" in [
        model.id for model in recommended_models(memory_gib=48, chip_tier=LEGACY_TIER)
    ]


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
        speed.download_gib + DOWNLOAD_HEADROOM_GIB
    )

    intel = evaluate_feasibility(
        speed, chip_tier=INTEL_TIER, ram_gib=128, disk_free_gib=500
    )
    assert intel.verdict == "insufficient_memory"


def test_catalog_model_matching_accepts_ids_repos_cache_dirs_and_aliases():
    speed = catalog_model_with_id("optimized-speed")
    speed_v2 = catalog_model_with_id("optimized-speed-v2")
    bare38 = catalog_model_with_id("qwen38-27b-bare-speed")
    os38 = catalog_model_with_id("qwen38-27b-optimized-speed")
    assert catalog_model_matching("optimized-speed") == speed
    # The public quickstart default is Qwen 3.8 Optimized Speed (2026-08-15);
    # every 3.8 build resolves to its own entry in every spelling, and the
    # FP16 siblings resolve to theirs (never to the parent).
    assert catalog_model_matching(DEFAULT_HF_MODEL_ID) == os38
    assert catalog_model_matching("Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2") == speed_v2
    for base in ("qwen38-27b-bare-speed", "qwen38-27b-optimized-speed", "qwen38-27b-optimized-quality"):
        sibling = catalog_model_with_id(f"{base}-fp16")
        assert catalog_model_matching(f"{base}-fp16") == sibling
        assert catalog_model_matching(f"mtplx-{base}-fp16") == sibling
        assert catalog_model_matching(sibling.hf_model_id) == sibling
        assert catalog_model_matching(f"~/.mtplx/models/{sibling.hf_model_id.replace('/', '--')}") == sibling
    assert catalog_model_matching("qwen38-27b-bare-speed") == bare38
    assert catalog_model_matching("mtplx-qwen38-27b-bare-speed") == bare38
    assert (
        catalog_model_matching(
            "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed"
        )
        == bare38
    )
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


def test_scan_installed_models_retains_duplicate_root_identity(tmp_path):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    name = "acme--custom-model"
    first = _write_complete_model(primary / name)
    second = _write_complete_model(secondary / name)

    installed = scan_installed_models(primary, search_dirs=[secondary])

    assert [model.path for model in installed] == [first, second]
    assert [model.root for model in installed] == [
        primary.resolve(),
        secondary.resolve(),
    ]
    assert [model.root_index for model in installed] == [0, 1]
    assert [model.is_primary for model in installed] == [True, False]


def test_scan_installed_models_dedupes_physical_aliases(tmp_path):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    model = _write_complete_model(primary / "acme--custom-model")
    secondary.mkdir()
    (secondary / "acme--custom-model").symlink_to(model, target_is_directory=True)

    installed = scan_installed_models(primary, search_dirs=[secondary])

    assert [row.path for row in installed] == [model]


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


def test_select_default_model_routes_small_macs_to_packs_that_fit(monkeypatch):
    monkeypatch.delenv("MTPLX_DEFAULT_MODEL_VARIANT", raising=False)

    # 24 GiB: Bonsai leads the smaller-Mac tier.
    small_modern = select_default_model(
        hardware={
            "chip": "Apple M4",
            "apple_silicon_generation": "m4",
            "memory_gib": 24.0,
        }
    )
    assert small_modern.model == "Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
    assert small_modern.hf_model == "Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
    assert small_modern.variant == "speed"
    assert "Bonsai 2 27B" in small_modern.reason
    assert small_modern.display_name == "Bonsai 2 27B Optimized Speed"
    assert small_modern.memory_gib == 24.0

    # 16 GiB: the picker's smaller-Mac tier leads with Bonsai, so the CLI
    # default matches it (the 4B pair leads only below 16 GiB).
    sixteen_modern = select_default_model(
        hardware={
            "chip": "Apple M4",
            "apple_silicon_generation": "m4",
            "memory_gib": 16.0,
        }
    )
    assert sixteen_modern.model == "Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
    assert "Bonsai" in sixteen_modern.reason
    assert recommended_models(memory_gib=16.0, chip_tier=MODERN_TIER)[0].hf_model_id == (
        sixteen_modern.model
    )

    # 8 GiB: the 9B's 10 GiB peak does not fit, so the 4B leads, exactly as
    # in the picker (the 9B used to be handed to every Mac under 32 GiB).
    tiny_modern = select_default_model(
        hardware={
            "chip": "Apple M4",
            "apple_silicon_generation": "m4",
            "memory_gib": 8.0,
        }
    )
    assert tiny_modern.model == "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed"
    assert "4B" in tiny_modern.reason
    assert tiny_modern.display_name == "Qwen3.5 4B Optimized Speed"
    assert recommended_models(memory_gib=8.0, chip_tier=MODERN_TIER)[0].hf_model_id == (
        tiny_modern.model
    )

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


def test_select_default_model_uses_public_v2_without_local_qwen38(monkeypatch):
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
    """Unreadable memory keeps the generation's precision lane but can no
    longer pick the 27B: with nothing to say what fits, the smallest pack is
    chosen and the reason says so."""
    monkeypatch.delenv("MTPLX_DEFAULT_MODEL_VARIANT", raising=False)
    monkeypatch.setenv("MTPLX_OPTIMIZED_SPEED_MODEL", "off")

    selection = select_default_model(
        hardware={
            "chip": "Apple M4",
            "apple_silicon_generation": "m4",
        }
    )
    assert selection.variant == "speed"
    assert selection.model == "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed"
    assert selection.memory_gib is None
    assert "memory could not be read" in selection.reason


# ---- README model table stays true to the catalog (issues #238, #408) -----

_README_TABLE_ROW = re.compile(
    r"^\|\s*`(?P<repo>Qwen[\w.\-]+|Gemma[\w.\-]+|Ternary-Bonsai[\w.\-]+)`\s*\|"
    r"(?P<fits>[^|]*)\|(?P<purpose>[^|]*)\|(?P<preset>[^|]*)\|\s*$"
)
_README_PEAK = re.compile(r"peaks at (?P<peak>[\d.]+) GiB")


def _readme_model_rows() -> list[re.Match[str]]:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    return [
        match
        for line in readme.splitlines()
        if (match := _README_TABLE_ROW.match(line)) is not None
    ]


def test_readme_model_table_names_real_catalog_packs():
    """Every repo the README recommends has to exist in the shipped catalog.

    The table is a promise about what a user can download; a renamed or
    retired pack must not survive in it silently.
    """
    rows = _readme_model_rows()
    assert len(rows) >= 11, "the README model table lost rows"
    catalog_repos = {model.hf_model_id for model in OFFICIAL_CATALOG}
    for row in rows:
        repo = f"Youssofal/{row.group('repo')}"
        assert repo in catalog_repos, f"README names a pack the catalog does not ship: {repo}"


def test_readme_model_table_quotes_the_catalog_peak_memory():
    """The "fits" column is the number the app checks a Mac against."""
    peaks = {model.hf_model_id: model.peak_memory_gib for model in OFFICIAL_CATALOG}
    for row in _readme_model_rows():
        repo = f"Youssofal/{row.group('repo')}"
        stated = _README_PEAK.search(row.group("fits"))
        assert stated is not None, f"README row for {repo} states no peak"
        assert abs(float(stated.group("peak")) - peaks[repo]) <= 0.05, (
            f"README peak for {repo} drifted from the catalog"
        )


def test_every_inkling_folder_serves_one_id():
    """A download (Youssofal--<repo>), a folder named like the repository, the folder the packs were built under
    and the catalog id all serve one id, and the app's twin names the same one; a derivative folder does not."""
    from mtplx.default_models import public_model_id_for_ref

    for model_id in ("inkling2-mini-bare-speed", "inkling2-mini-optimized-speed", "inkling2-mini-optimized-quality",
                     "inkling2-mini-bare-speed-fp16", "inkling2-mini-optimized-speed-fp16",
                     "inkling2-mini-optimized-quality-fp16"):
        model = catalog_model_with_id(model_id)
        served = model.aliases[0]
        repo = model.hf_model_id.split("/")[1]
        built = "MTPLX-Inkling-2-Mini-" + repo.removeprefix("Inkling-2-Mini-MTPLX-")
        assert served == "mtplx-" + model_id.replace("inkling2-mini", "inkling-2-mini")
        for ref in (model.hf_model_id, model_id, f"/Users/u/.mtplx/models/Youssofal--{repo}",
                    f"/Users/u/.mtplx/models/{repo}", f"/Users/u/Models/mtplx/{built}"):
            assert public_model_id_for_ref(ref) == served, ref
            assert catalog_model_matching(ref) is model, ref
    for derivative in ("/x/Inkling-2-Mini-MTPLX-Native-Experts-Quality", "/x/MTPLX-Inkling-2-Mini-Bare-Speed-V2"):
        assert public_model_id_for_ref(derivative) not in {"mtplx-inkling-2-mini-bare-speed",
                                                            "mtplx-inkling-2-mini-optimized-quality"}, derivative
        assert catalog_model_matching(derivative) is None

