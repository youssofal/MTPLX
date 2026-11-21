"""MiMo V2.6 Qwen 9B joins the catalog on the Qwen 3.5 contract (CPU only).

The pack is Xiaomi's Qwen3.5-9B fine-tune (config model_type qwen3_5,
architecture Qwen3_5ForConditionalGeneration). "MiMo" in its names is the
publisher: every identity must resolve to the qwen3_5 family, its served id,
and a place right ahead of the Qwen 3.5 9B in every modern recommendation
list, without changing any default model.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.artifacts import _hf_repo_id_from_ref
from mtplx.backends.descriptors import (
    BONSAI2_REASONING_CODEC,
    QWEN3_8_REASONING_CODEC,
    QWEN3_NEXT_DESCRIPTOR,
    context_window_policy_for_model,
    draft_semantics_for_model,
    kv_quant_policy_for_model,
    model_family_from_inspection,
    reasoning_policy_for_model,
    sampler_defaults_for_model,
    tune_policy_for_model,
)
from mtplx.commands.public import _apply_model_default_profile, _model_ref_from_public_model_id
from mtplx.default_models import public_model_id_for_ref
from mtplx.model_catalog import (
    LEGACY_TIER,
    MODERN_TIER,
    catalog_model_matching,
    catalog_model_with_id,
    recommended_catalog_ids,
    recommended_models,
)
from mtplx.profiles import (
    MIMO_V26_QWEN_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
    MIMO_V26_QWEN_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
)

MIMO = "mimo-v26-qwen-9b-optimized-speed"
NINE = "qwen35-9b-optimized-speed"
BONSAI = "bonsai-2-27b-optimized-speed"
TINY = ["qwen35-4b-optimized-speed", "qwen35-4b-optimized-quality"]
HF_ID = "Youssofal/MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed"
PUBLIC_ID = "mtplx-mimo-v26-qwen-9b-optimized-speed"


def _refs() -> list[str]:
    pack = catalog_model_with_id(MIMO)
    return [MIMO, pack.hf_model_id, *[alias for alias in pack.aliases if " " not in alias]]


def test_catalog_entry_mirrors_the_6bit_9b():
    pack = catalog_model_with_id(MIMO)
    nine = catalog_model_with_id(NINE)
    assert pack.display_name == "MiMo V2.6 Qwen 9B Optimized Speed"
    assert pack.detail == "6-bit quantization. Xiaomi's agentic coding distill of Qwen 3.5 9B."
    assert pack.hf_model_id == HF_ID == MIMO_V26_QWEN_9B_OPTIMIZED_SPEED_HF_MODEL_ID
    assert PUBLIC_ID == MIMO_V26_QWEN_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    assert pack.aliases == (PUBLIC_ID, "MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed")
    assert pack.size_bytes == 8_695_116_595
    assert pack.peak_memory_gib == nine.peak_memory_gib == 10.0
    # No FP16 sibling: M1 and M2 are never offered it.
    assert pack.recommended_tiers == frozenset({MODERN_TIER})
    assert not pack.ar_only


@pytest.mark.parametrize("ref", _refs() + [PUBLIC_ID])
def test_every_identity_resolves_to_the_pack_its_served_id_and_qwen35(ref):
    pack = catalog_model_with_id(MIMO)
    assert catalog_model_matching(ref) == pack
    assert public_model_id_for_ref(ref) == PUBLIC_ID
    assert _model_ref_from_public_model_id(ref) == HF_ID
    assert _hf_repo_id_from_ref(ref) == HF_ID
    assert model_family_from_inspection(model_ref=ref) == "qwen3_5"
    # The lane default descriptor used to turn the unknown name into qwen3_6.
    assert model_family_from_inspection(model_ref=ref, descriptor=QWEN3_NEXT_DESCRIPTOR) == "qwen3_5"


@pytest.mark.parametrize("folder", [
    "MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
    "Youssofal--MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
])
def test_local_folders_resolve_like_the_repo(folder, tmp_path):
    local = tmp_path / folder
    local.mkdir()
    assert catalog_model_matching(local) == catalog_model_with_id(MIMO)
    assert public_model_id_for_ref(local) == PUBLIC_ID
    assert model_family_from_inspection(model_ref=str(local)) == "qwen3_5"


def test_a_derivative_gets_no_first_party_identity():
    derivative = Path(HF_ID).name + "-third-party"
    assert catalog_model_matching(derivative) is None
    assert public_model_id_for_ref(derivative) != PUBLIC_ID


@pytest.mark.parametrize("ref", _refs() + [PUBLIC_ID])
def test_the_qwen35_contract_applies_by_name(ref):
    codec = reasoning_policy_for_model(ref)
    assert codec == QWEN3_NEXT_DESCRIPTOR.reasoning_codec
    assert codec.supported and codec.parser == "qwen3"
    assert tuple(codec.effort_levels) == ()  # no reasoning-effort dial
    assert codec not in (QWEN3_8_REASONING_CODEC, BONSAI2_REASONING_CODEC)
    assert sampler_defaults_for_model(ref).to_dict() == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert tune_policy_for_model(ref).candidates == ("AR", "D1", "D2", "D3")
    assert kv_quant_policy_for_model(ref) == QWEN3_NEXT_DESCRIPTOR.kv_quant_policy
    assert context_window_policy_for_model(ref) == QWEN3_NEXT_DESCRIPTOR.context_window_policy
    assert draft_semantics_for_model(ref) == QWEN3_NEXT_DESCRIPTOR.draft_semantics


def test_the_published_runtime_contract_resolves_in_a_renamed_folder(tmp_path):
    pack = tmp_path / "renamed-pack"
    pack.mkdir()
    (pack / "mtplx_runtime.json").write_text(json.dumps({
        "arch_id": "qwen3-next-mtp",
        "public_model_id": PUBLIC_ID,
        "served_model_id": PUBLIC_ID,
        "model_family": "qwen3_5",
        "base_trunk": "Qwen/Qwen3.5-9B",
        "mtp_depth_default": 2,
        "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    }))
    assert public_model_id_for_ref(pack) == PUBLIC_ID
    assert model_family_from_inspection(model_ref=str(pack)) == "qwen3_5"
    assert reasoning_policy_for_model(model_ref=str(pack)) == QWEN3_NEXT_DESCRIPTOR.reasoning_codec


def test_the_checkpoint_inspection_never_yields_the_mimo_family():
    inspection = {
        "model_dir": "/Users/example/.mtplx/models/Youssofal--MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
        "runtime_model": HF_ID,
        "model_type": "qwen3_5",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "mtp_num_hidden_layers": 1,
        "recommended_backend": "qwen3_next",
    }
    assert model_family_from_inspection(inspection) == "qwen3_5"
    assert model_family_from_inspection(inspection, model_ref=HF_ID) == "qwen3_5"
    assert tune_policy_for_model(HF_ID, inspection).candidates == ("AR", "D1", "D2", "D3")


@pytest.mark.parametrize("memory_gib,expected", [
    (15.9, TINY),
    (16, [BONSAI, MIMO, NINE, *TINY]),
    # From 24 GiB Inkling 2 Mini Bare Speed rides directly after the tier's first pick
    # (INKLING_TIGHT_MAC_OFFER, on since 1282d177).
    (24, [BONSAI, "inkling2-mini-bare-speed", MIMO, NINE, *TINY]),
])
def test_small_modern_tiers(memory_gib, expected):
    assert recommended_catalog_ids(memory_gib=memory_gib, chip_tier=MODERN_TIER) == expected
    assert [m.id for m in recommended_models(memory_gib=memory_gib, chip_tier=MODERN_TIER)] == expected


@pytest.mark.parametrize("memory_gib", [16, 24, 32, 48, 64, 96, 128, 256, None])
def test_mimo_rides_immediately_ahead_of_the_9b_on_every_modern_tier(memory_gib):
    raw = recommended_catalog_ids(memory_gib=memory_gib, chip_tier=MODERN_TIER)
    assert raw.count(MIMO) == 1
    assert raw.index(MIMO) + 1 == raw.index(NINE)
    visible = [m.id for m in recommended_models(memory_gib=memory_gib, chip_tier=MODERN_TIER)]
    assert visible.index(MIMO) + 1 == visible.index(NINE)
    # Never the first pick: Bonsai, the Qwen 3.8 trio or Flash-Next leads.
    assert raw[0] != MIMO


@pytest.mark.parametrize("memory_gib", [8, 15.9, 16, 24, 32, 48, 64, 96, 128, 256, None])
def test_the_legacy_tier_never_lists_mimo(memory_gib):
    assert MIMO not in recommended_catalog_ids(memory_gib=memory_gib, chip_tier=LEGACY_TIER)


@pytest.mark.parametrize("memory_gib,expected", [
    (12.0, "qwen35-4b-optimized-speed"),
    (16.0, BONSAI),
    (24.0, BONSAI),
    (32.0, "qwen38-27b-optimized-speed"),
    (256.0, "flash-next-optimized-speed"),
    (None, "qwen35-4b-optimized-speed"),
])
def test_no_default_model_changes(memory_gib, expected, monkeypatch):
    from mtplx import default_models as defaults

    monkeypatch.delenv(defaults.DEFAULT_MODEL_VARIANT_ENV, raising=False)
    monkeypatch.setenv(defaults.SPEED_MODEL_ENV, "off")
    monkeypatch.setenv(defaults.QWEN38_OPTIMIZED_SPEED_MODEL_ENV, "off")
    hardware = {"chip": "Apple M5", "apple_silicon_generation": "m5"}
    if memory_gib is not None:
        hardware["memory_gib"] = memory_gib
    selection = defaults.select_default_model(hardware=hardware)
    assert catalog_model_matching(selection.hf_model).id == expected


def test_a_mimo_selection_is_labelled_with_its_own_name():
    from mtplx.default_models import DefaultModelSelection

    selection = DefaultModelSelection(
        model=HF_ID, hf_model=HF_ID, variant="speed", precision="Compact 6-bit model for smaller Macs",
        chip_generation="m5", chip="Apple M5", reason="test", auto_selected=True,
    )
    assert selection.display_name == "MiMo V2.6 Qwen 9B Optimized Speed"


@pytest.mark.parametrize("ref", _refs())
def test_no_unmeasured_turbo_promotion(ref):
    args = SimpleNamespace(model=ref, profile="sustained", _cli_flags=set())
    assert not _apply_model_default_profile(args, PUBLIC_ID)
    assert args.profile == "sustained"


def test_the_models_manifest_blesses_the_pack_from_2120():
    from scripts import gen_models_manifest as manifest

    assert manifest.BLESSED[HF_ID] == "2.12.0"
