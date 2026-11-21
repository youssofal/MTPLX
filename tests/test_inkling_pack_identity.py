"""A folder that carries an Inkling pack's name is that pack only when its own mtplx-pack.json says so.

The Mac the packs were built on kept two research builds (20 November) under the published repository names, and the
catalog served them as Optimized Quality and Optimized Speed from the name alone. The identity is the source, the recipe
and the FP16 siblings' precision variant (PACK_IDENTITIES). Mirrors the app's
InklingLaunchContractTests.testAFolderUnderAPackNameIsThatPackOnlyWhenItSaysSo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mtplx.hf_loader as hf_loader
from mtplx.commands.forge_inkling import PACK_NAMES
from mtplx.default_models import catalog_model_ref, public_model_id_for_ref
from mtplx.hf_loader import resolve_model_path
from mtplx.model_catalog import (
    PACK_IDENTITIES,
    catalog_model_matching,
    catalog_model_with_id,
    scan_installed_models,
)

SOURCE = "thinkingmachines/Inkling-2-Mini"
QUALITY = {"display_name": "Inkling 2 Mini Optimized Quality", "recipe": "quality", "source": SOURCE}
RESEARCH = {"display_name": "Inkling 2 Mini Research Q8 Experts", "recipe": "research-all-q8", "source": SOURCE}
REPO = "Youssofal/Inkling-2-Mini-MTPLX-Optimized-Quality"


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(root))
    monkeypatch.delenv("MTPLX_MODEL_DIRS", raising=False)
    monkeypatch.setattr(hf_loader, "huggingface_cache_path", lambda repo_id: None)
    return root


def _pack(path: Path, stamp: dict | None) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    if stamp is not None:
        (path / "mtplx-pack.json").write_text(json.dumps(stamp), encoding="utf-8")
    return path


def test_a_research_build_under_the_published_name_is_passed_over(library):
    pack = catalog_model_with_id("inkling2-mini-optimized-quality")
    research = _pack(library / "Inkling-2-Mini-MTPLX-Optimized-Quality", RESEARCH)
    with pytest.raises(FileNotFoundError) as refused:
        resolve_model_path(REPO)
    message = str(refused.value)
    assert f"{research} is not served" in message
    assert "Inkling 2 Mini Research Q8 Experts (recipe research-all-q8), not Inkling 2 Mini Optimized Quality" in message
    assert message.endswith(f"Run: mtplx pull {REPO}")
    assert catalog_model_ref(pack) == REPO
    assert catalog_model_matching(research) is None
    # It keeps its own folder's name, the one the app writes into a client's config for it.
    assert public_model_id_for_ref(research) == "inkling-2-mini-mtplx-optimized-quality"
    assert [m.catalog for m in scan_installed_models() if m.path == research] == [None]
    # A pull does not take it for the pack it would download.
    assert not hf_loader._cached_model_ready_for_repo(research, REPO)


def test_the_published_pack_is_found_beside_a_research_build(library):
    pack = catalog_model_with_id("inkling2-mini-optimized-quality")
    _pack(library / "Inkling-2-Mini-MTPLX-Optimized-Quality", RESEARCH)
    built = _pack(library / "MTPLX-Inkling-2-Mini-Optimized-Quality", QUALITY)
    assert catalog_model_ref(pack) == str(built)
    downloaded = _pack(library / "Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality", QUALITY)
    assert resolve_model_path(REPO) == downloaded
    for folder in (built, downloaded):
        assert catalog_model_matching(folder) is pack
        assert public_model_id_for_ref(folder) == "mtplx-inkling-2-mini-optimized-quality"
    assert {m.name: m.catalog and m.catalog.id for m in scan_installed_models()} == {
        "Inkling-2-Mini-MTPLX-Optimized-Quality": None,
        "MTPLX-Inkling-2-Mini-Optimized-Quality": "inkling2-mini-optimized-quality",
        "Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality": "inkling2-mini-optimized-quality",
    }


def test_a_folder_without_a_pack_description_proves_nothing(library):
    pack = catalog_model_with_id("inkling2-mini-optimized-quality")
    bare = _pack(library / "Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality", None)
    assert catalog_model_ref(pack) == REPO
    assert catalog_model_matching(bare) is None
    with pytest.raises(FileNotFoundError, match="no readable mtplx-pack.json"):
        resolve_model_path(REPO)


def test_the_fp16_name_takes_only_the_fp16_sibling(library):
    fp16 = catalog_model_with_id("inkling2-mini-optimized-quality-fp16")
    bf16 = _pack(library / "Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality-FP16", QUALITY)
    assert catalog_model_ref(fp16) == fp16.hf_model_id
    assert catalog_model_matching(bf16) is None
    sibling = _pack(library / "Inkling-2-Mini-MTPLX-Optimized-Quality-FP16",
                    {**QUALITY, "display_name": "Inkling 2 Mini Optimized Quality FP16", "precision_variant": "fp16"})
    assert catalog_model_ref(fp16) == str(sibling)
    assert catalog_model_matching(sibling) is fp16
    # The sibling under the BF16 pack's name is not the BF16 pack either.
    under_bf16_name = library / "Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality"
    sibling.rename(under_bf16_name)
    assert catalog_model_matching(under_bf16_name) is None


def test_names_that_are_not_folders_match_as_before(library):
    pack = catalog_model_with_id("inkling2-mini-optimized-quality")
    assert catalog_model_matching("Inkling 2 Mini Optimized Quality") is pack
    for ref in (REPO, "mtplx-inkling-2-mini-optimized-quality", "/Users/u/.mtplx/models/Inkling-2-Mini-MTPLX-Optimized-Quality"):
        assert catalog_model_matching(ref) is pack, ref
        assert public_model_id_for_ref(ref) == "mtplx-inkling-2-mini-optimized-quality", ref


@pytest.mark.parametrize("pack_id", sorted(PACK_IDENTITIES))
def test_each_identity_is_what_the_forge_stamps_for_that_pack(pack_id):
    """The catalog's identity is the stamp the forge writes (write_pack_metadata, stamp_fp16_variant): the display
    name it derives from the recipe is the catalog's own, so a recipe renamed on one side fails here."""

    pack = catalog_model_with_id(pack_id)
    identity = PACK_IDENTITIES[pack_id]
    assert identity.source == SOURCE
    suffix = " FP16" if identity.precision_variant == "fp16" else ""
    assert f"Inkling 2 Mini {PACK_NAMES[identity.recipe]}{suffix}" == pack.display_name
    assert identity.precision_variant == ("fp16" if pack_id.endswith("-fp16") else None)
