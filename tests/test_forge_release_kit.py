"""The three texts published about a pack come from one description and agree."""

from __future__ import annotations

import json

import pytest

from mtplx import model_catalog
from mtplx.commands import forge_release_kit as kit

ROWS = (
    kit.Row("Coding task, long answer", "71.2 tok/s", "rows/dzr-optimized-speed/01-cand/boot.json#flappy-cold"),
    kit.Row("First text on a 16K prompt", "3.1 s", "rows/dzr-optimized-speed/01-cand/boot.json#L16384"),
)


def _facts(**changes) -> kit.PackFacts:
    described = dict(
        model="Qwen 3.8 27B",
        kind="Optimized Speed",
        detail="4-bit dynamic quant. Great coding speeds and good quality.",
        hf_owner="Youssofal",
        base_model="Qwen/Qwen3.8-27B",
        license="apache-2.0",
        size_bytes=20_703_486_605,
        peak_memory_gib=25.0,
        vision=True,
        rows=ROWS,
        measured_on="16 November 2026",
    )
    described.update(changes)
    return kit.PackFacts(**described)


def test_names_follow_the_catalogs_own_pattern():
    """Rendered for a pack we already publish, the names are the ones in the catalog."""

    published = next(model for model in model_catalog.OFFICIAL_CATALOG if model.id == "qwen38-27b-optimized-speed")
    facts = _facts()

    assert facts.id == published.id
    assert facts.display_name == published.display_name
    assert facts.hf_model_id == published.hf_model_id
    assert set(facts.aliases) <= set(published.aliases)
    assert _facts(model="Qwen 4 27B").id == "qwen4-27b-optimized-speed"
    assert _facts(model="Qwen 4 27B").repo_name == "Qwen4-27B-MTPLX-Optimized-Speed"


def test_the_python_entry_is_a_catalog_model():
    facts = _facts()
    entry = eval(  # noqa: S307 - our own rendered text
        kit.python_entry(facts).strip().rstrip(","),
        {"CatalogModel": model_catalog.CatalogModel, "MODERN_TIER": model_catalog.MODERN_TIER, "frozenset": frozenset},
    )

    assert entry.id == facts.id and entry.size_bytes == facts.size_bytes
    assert entry.peak_memory_gib == 25.0 and entry.ar_only is False
    assert entry.aliases == facts.aliases


@pytest.mark.parametrize("changes", [{}, {"model": "Qwen 4 27B", "kind": "Optimized Quality", "ar_only": True}])
def test_the_python_and_swift_entries_say_the_same(changes):
    facts = _facts(**changes)
    python = kit.entry_fields(kit.python_entry(facts))
    swift = kit.entry_fields(kit.swift_entry(facts))

    assert python == swift
    assert python["id"] == facts.id and python["size_bytes"] == facts.size_bytes
    assert python["aliases"] == facts.aliases
    assert python["ar_only"] is facts.ar_only


def test_a_swift_entry_lists_both_local_folders_a_download_can_land_in():
    swift = kit.swift_entry(_facts())

    assert '"~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"' in swift
    assert '"~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed"' in swift
    assert "recommendedFor: [.modernApple]" in swift


def test_a_vision_pack_is_labelled_as_one():
    card = kit.card(_facts(), mtplx_version="3.0.0")

    assert "pipeline_tag: image-text-to-text" in card and "\n- vision\n" in card
    assert "library_name: mtplx" in card and "base_model: Qwen/Qwen3.8-27B" in card
    assert "# Qwen 3.8 27B Optimized Speed" in card
    assert "## Measured on MTPLX 3.0.0 (16 November 2026)" in card
    assert "| Coding task, long answer | 71.2 tok/s |" in card
    assert "—" not in card  # public text carries no em dash

    blind = kit.card(_facts(vision=False), mtplx_version="3.0.0")
    assert "pipeline_tag: text-generation" in blind and "- vision" not in blind


def test_a_pack_without_a_draft_head_says_so():
    card = kit.card(_facts(ar_only=True), mtplx_version="3.0.0")

    assert "decodes one token at a time" in card
    assert "multi-token-prediction\n" not in card
    assert "ar_only=True" in kit.python_entry(_facts(ar_only=True))


def test_no_card_without_measured_rows():
    with pytest.raises(ValueError, match="no rows were given"):
        kit.card(_facts(rows=()), mtplx_version="3.0.0")


def _pack(folder, *, tower: bool):
    folder.mkdir()
    names = ["language_model.model.norm.weight"] + (["vision_tower.merger.norm.weight"] if tower else [])
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors" for name in names}}))
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "vision_config": {"depth": 27}}))
    (folder / "model.safetensors").write_bytes(b"\0" * 1000)
    return folder


def test_facts_come_from_the_pack_on_disk(tmp_path):
    described = dict(
        model="Qwen 4 27B", kind="Optimized Speed", detail="d", hf_owner="Youssofal",
        base_model="Qwen/Qwen4-27B", license="apache-2.0", peak_memory_gib=24.0,
    )  # fmt: skip
    pack = _pack(tmp_path / "pack", tower=True)

    facts = kit.facts_from_pack(pack, **described)

    assert facts.vision is True
    assert facts.size_bytes == sum(path.stat().st_size for path in pack.iterdir())


def test_a_blind_pack_of_a_vision_model_cannot_be_described(tmp_path):
    """Law 1 at the last door: nothing is written about a pack that lost its tower."""

    with pytest.raises(ValueError, match="carries no tower tensor"):
        kit.facts_from_pack(
            _pack(tmp_path / "pack", tower=False),
            model="Qwen 4 27B", kind="Optimized Speed", detail="d", hf_owner="Youssofal",
            base_model="Qwen/Qwen4-27B", license="apache-2.0", peak_memory_gib=24.0,
        )  # fmt: skip
