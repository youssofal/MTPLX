"""Shared official-model catalog with RAM-aware recommendation tiers.

SYNC PAIR: this module is the Python port of the macOS app's catalog and
feasibility policy. When editing either side, update the other in the same
change:

- apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXModelOption.swift
  (``officialCatalog``, ``recommendedCatalogIDs``/``recommendationIDs``)
- apps/MTPLXApp/Sources/MTPLXAppCore/Onboarding/ModelFeasibility.swift
  (``memorySafetyFactor``, ``diskMultiplier``, verdict rules)

The CLI previously picked default models from chip generation alone, which
offered a 27B download to 8 GB Macs. This catalog gives every surface the
same answer: which models exist, how much memory they peak at, and which
ones this machine should be offered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mtplx.hf_loader import cached_model_is_complete, model_cache_dir

MEMORY_SAFETY_FACTOR = 1.5
DISK_MULTIPLIER = 2.5

MODERN_TIER = "modern"
LEGACY_TIER = "legacy"
INTEL_TIER = "intel"
UNKNOWN_TIER = "unknown"

_LEGACY_GENERATIONS = frozenset({"m1", "m2"})
_MODERN_GENERATIONS = frozenset({"m3", "m4", "m5"})


@dataclass(frozen=True)
class CatalogModel:
    """One official artifact, mirroring ``MTPLXModelOption`` in the app."""

    id: str
    display_name: str
    detail: str
    hf_model_id: str
    size_bytes: int
    peak_memory_gib: float
    recommended_tiers: frozenset[str]
    aliases: tuple[str, ...] = ()
    # Target-only AR model (no native MTP head); serve must use --no-mtp.
    ar_only: bool = False

    @property
    def download_gib(self) -> float:
        return self.size_bytes / 1_073_741_824.0


OFFICIAL_CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel(
        id="qwen35-4b-optimized-speed",
        display_name="Qwen 3.5 4B Optimized Speed",
        detail="4-bit quantization. Fastest fit for smaller Macs.",
        hf_model_id="Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed",
        size_bytes=2_474_027_992,
        peak_memory_gib=2.86,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen35-4b-optimized-speed",
            "qwen3.5-4b-mtplx-optimized-speed",
            "Qwen3.5 4B Optimized Speed",
            "Qwen 3.5 4B",
            "Small Qwen",
        ),
    ),
    CatalogModel(
        id="qwen35-4b-optimized-quality",
        display_name="Qwen 3.5 4B Optimized Quality",
        detail="8-bit quantization. Highest-fidelity 4B; 2x MTP multiplier.",
        hf_model_id="Youssofal/Qwen3.5-4B-MTPLX-Optimized-Quality",
        size_bytes=4_576_423_401,
        peak_memory_gib=4.75,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen35-4b-optimized-quality",
            "qwen3.5-4b-mtplx-optimized-quality",
            "Qwen3.5 4B Optimized Quality",
            "Qwen 3.5 4B Quality",
        ),
    ),
    CatalogModel(
        id="qwen35-9b-optimized-speed",
        display_name="Qwen 3.5 9B Optimized Speed",
        detail="6-bit quantization. Strong small-Mac speed pick.",
        hf_model_id="Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed",
        size_bytes=7_783_037_915,
        peak_memory_gib=10.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen35-9b-optimized-speed",
            "mtplx-qwen35-9b-speed-6bit",
            "Qwen-Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI",
            "Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI",
            "Qwen 3.5 9B Speed 6-bit",
            "Qwen 3.5 9B Speed",
        ),
    ),
    CatalogModel(
        id="qwen35-9b-optimized-speed-fp16",
        display_name="Qwen 3.5 9B Optimized Speed FP16",
        detail="FP16-friendly 9B speed artifact for M1 and M2 Macs.",
        hf_model_id="Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16",
        size_bytes=7_783_301_179,
        peak_memory_gib=10.5,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen35-9b-optimized-speed-fp16",
            "Qwen3.5 9B Optimized Speed FP16",
            "Qwen 3.5 9B Speed FP16",
        ),
    ),
    CatalogModel(
        id="optimized-speed-v2",
        display_name="Qwen 3.6 27B Optimized Speed V2",
        detail=(
            "Much higher quality for coding. Dynamic 4-bit hybrid quantization "
            "keeps hand-tuned sensitive parts at up to 16-bit. Faster on long "
            "agent tasks, slightly larger, and a little slower for short chats."
        ),
        hf_model_id="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
        size_bytes=19_887_448_095,
        peak_memory_gib=21.5,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen36-27b-optimized-speed-v2",
            "Qwen3.6 27B Optimized Speed V2",
            "Optimized Speed V2",
        ),
    ),
    CatalogModel(
        id="optimized-speed",
        display_name="Qwen 3.6 27B Optimized Speed",
        detail="Smaller 4-bit model. A little faster for short chats.",
        hf_model_id="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        size_bytes=16_106_127_360,
        peak_memory_gib=17.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen36-27b-optimized-speed",
            "Qwen3.6 27B Optimized Speed",
            "Optimized Speed",
        ),
    ),
    CatalogModel(
        id="optimized-speed-fp16",
        display_name="Qwen 3.6 27B Optimized Speed FP16",
        detail="FP16 speed artifact recommended for M1 and M2 Macs.",
        hf_model_id="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
        # Exact sum of the published HF repo files (2026-07-03 audit); the
        # previous 16-GiB figure was a pre-publish estimate ~0.7 GiB high.
        size_bytes=16_419_644_370,
        peak_memory_gib=17.5,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen36-27b-optimized-speed-fp16",
            "Qwen3.6 27B Optimized Speed FP16",
            "Optimized Speed FP16",
        ),
    ),
    CatalogModel(
        id="qwen36-35b-a3b-optimized-speed",
        display_name="Qwen 3.6 35B-A3B Optimized Speed",
        detail="4-bit quantization. Blazingly fast and quite smart.",
        hf_model_id="Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
        size_bytes=21_016_117_499,
        peak_memory_gib=28.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen36-35b-a3b-optimized-speed",
            "Qwen3.6 35B-A3B Optimized Speed",
            "Qwen3.6 35B Speed",
            "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
            "Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe",
            "Qwen3.6-35B-A3B-MTPLX-Flat4-CyanKiwiMTP-ForgeRepairClean",
        ),
    ),
    CatalogModel(
        id="qwen36-35b-a3b-optimized-speed-fp16",
        display_name="Qwen 3.6 35B-A3B Optimized Speed FP16",
        detail="FP16-friendly 35B speed artifact for M1 and M2 Macs.",
        hf_model_id="Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16",
        size_bytes=21_016_116_512,
        peak_memory_gib=28.5,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen36-35b-a3b-optimized-speed-fp16",
            "Qwen3.6 35B-A3B Optimized Speed FP16",
            "Qwen3.6 35B Speed FP16",
        ),
    ),
    CatalogModel(
        id="qwen36-35b-a3b-optimized-balance",
        display_name="Qwen 3.6 35B-A3B Optimized Balance",
        detail="6-bit quantization. Stronger balance of speed and quality.",
        hf_model_id="Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
        size_bytes=29_672_250_227,
        peak_memory_gib=32.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen36-35b-a3b-optimized-balance",
            "Qwen3.6 35B-A3B Optimized Balance",
            "Qwen3.6 35B Balance",
        ),
    ),
    CatalogModel(
        id="qwen36-35b-a3b-optimized-balance-fp16",
        display_name="Qwen 3.6 35B-A3B Optimized Balance FP16",
        detail="FP16-friendly 35B balance artifact for M1 and M2 Macs.",
        hf_model_id="Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16",
        size_bytes=29_672_249_552,
        peak_memory_gib=32.5,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen36-35b-a3b-optimized-balance-fp16",
            "Qwen3.6 35B-A3B Optimized Balance FP16",
            "Qwen3.6 35B Balance FP16",
        ),
    ),
    CatalogModel(
        id="gemma4-optimized-speed",
        display_name="Gemma 4 31B Optimized Speed",
        detail="High quality. Moderate speeds.",
        hf_model_id="Youssofal/Gemma4-MTPLX-Optimized-Speed",
        size_bytes=17_715_675_136,
        peak_memory_gib=18.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "Gemma4-MTPLX-Optimized-Speed",
            "Gemma4 Optimized Speed",
            "Gemma4 Speed",
            "gemma4-mtplx-optimized-speed",
            "mtplx/gemma4-mtplx-optimized-speed",
            "mtplx-gemma4-optimized-speed",
        ),
    ),
    CatalogModel(
        id="optimized-quality",
        display_name="Qwen 3.6 27B Optimized Quality",
        detail="Maximum quality. Moderate speeds.",
        hf_model_id="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality",
        size_bytes=30_064_771_072,
        peak_memory_gib=27.62,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen36-27b-optimized-quality",
            "Qwen3.6 27B Optimized Quality",
            "Optimized Quality",
        ),
    ),
    CatalogModel(
        id="optimized-quality-fp16",
        display_name="Qwen 3.6 27B Optimized Quality FP16",
        detail="FP16 quality artifact recommended for M1 and M2 Macs.",
        hf_model_id="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
        # Exact byte sum of the published HF repo files (2026-07-07 upload,
        # verified via the tree API).
        size_bytes=30_017_528_922,
        peak_memory_gib=28.12,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen36-27b-optimized-quality-fp16",
            "Qwen3.6 27B Optimized Quality FP16",
            "Optimized Quality FP16",
        ),
    ),
    CatalogModel(
        id="laguna-s21-oq4e",
        display_name="Laguna S-2.1 (community oQ4e)",
        detail="Poolside coding model, mixed-precision 4-bit. AR-only (no MTP head yet).",
        hf_model_id="mlx-community/Laguna-S-2.1-oQ4e",
        size_bytes=64_129_728_868,
        peak_memory_gib=74.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-laguna-s21",
            "Laguna S-2.1",
            "Laguna-S-2.1-oQ4e",
        ),
        ar_only=True,
    ),
)

# Mirrors `modernTopRecommendationIDs` in MTPLXModelOption.swift: the
# fallback matrix when hardware is unknown.
_MODERN_TOP_RECOMMENDATION_IDS = (
    "optimized-speed-v2",
    "optimized-speed",
    "optimized-quality",
    "qwen36-35b-a3b-optimized-speed",
    "qwen36-35b-a3b-optimized-balance",
    "gemma4-optimized-speed",
    "qwen35-9b-optimized-speed",
)


def catalog_model_with_id(model_id: str) -> CatalogModel | None:
    for model in OFFICIAL_CATALOG:
        if model.id == model_id:
            return model
    return None


def _normalized(value: str) -> str:
    # Mirrors MTPLXModelOption.normalized: HF cache directory names use
    # `owner--repo`, which must compare equal to the `owner/repo` id.
    return value.strip().lower().replace("\\", "/").replace("--", "/")


def catalog_model_matching(ref: str | Path | None) -> CatalogModel | None:
    """Match a model reference (id, HF repo, path, alias) to the catalog."""

    if ref is None:
        return None
    text = str(ref).strip()
    if not text:
        return None
    normalized = _normalized(text)
    basename = _normalized(Path(text).name)
    for model in OFFICIAL_CATALOG:
        candidates = {
            _normalized(model.id),
            _normalized(model.display_name),
            _normalized(model.hf_model_id),
            *(_normalized(alias) for alias in model.aliases),
        }
        if normalized in candidates or basename in candidates:
            return model
    return None


def chip_tier_for_generation(generation: str | None) -> str:
    normalized = str(generation or "").strip().lower()
    if normalized in _LEGACY_GENERATIONS:
        return LEGACY_TIER
    if normalized in _MODERN_GENERATIONS:
        return MODERN_TIER
    if normalized == "intel":
        return INTEL_TIER
    return UNKNOWN_TIER


def recommended_catalog_ids(
    *,
    memory_gib: float | None,
    chip_tier: str,
) -> list[str]:
    """RAM-tiered recommendation order, ported from recommendedCatalogIDs."""

    if chip_tier == INTEL_TIER:
        return []
    if chip_tier == LEGACY_TIER:
        small = "qwen35-9b-optimized-speed-fp16"
        speed27 = "optimized-speed-fp16"
        speed27_v2 = None
        speed35 = "qwen36-35b-a3b-optimized-speed-fp16"
        balance35 = "qwen36-35b-a3b-optimized-balance-fp16"
        quality27 = "optimized-quality-fp16"
    else:
        small = "qwen35-9b-optimized-speed"
        speed27 = "optimized-speed"
        speed27_v2 = "optimized-speed-v2"
        speed35 = "qwen36-35b-a3b-optimized-speed"
        balance35 = "qwen36-35b-a3b-optimized-balance"
        quality27 = "optimized-quality"
    if memory_gib is None or memory_gib <= 0:
        if chip_tier == LEGACY_TIER:
            return [speed27, quality27, speed35, balance35, "gemma4-optimized-speed", small]
        return list(_MODERN_TOP_RECOMMENDATION_IDS)
    # The rebuilt 4B pair leads the sub-16GB tiers and trails every larger
    # modern tier so it stays discoverable as the fast-small pick. No fp16
    # siblings yet, so the legacy (M1/M2) matrix keeps its fp16-only entries.
    tiny_ids = (
        ["qwen35-4b-optimized-speed", "qwen35-4b-optimized-quality"]
        if chip_tier != LEGACY_TIER
        else []
    )
    if memory_gib < 16:
        return tiny_ids or [small]
    if memory_gib < 32:
        return [small, *tiny_ids]
    if memory_gib < 48:
        if speed27_v2 is None:
            return [small, speed27, "gemma4-optimized-speed", speed35, quality27]
        return [
            speed27_v2,
            speed27,
            small,
            "gemma4-optimized-speed",
            speed35,
            quality27,
            *tiny_ids,
        ]
    return [
        *([speed27_v2] if speed27_v2 else []),
        speed27,
        quality27,
        speed35,
        balance35,
        "gemma4-optimized-speed",
        small,
        *tiny_ids,
    ]


def recommended_models(
    *,
    memory_gib: float | None,
    chip_tier: str,
) -> list[CatalogModel]:
    models = [
        model
        for model_id in recommended_catalog_ids(
            memory_gib=memory_gib, chip_tier=chip_tier
        )
        if (model := catalog_model_with_id(model_id)) is not None
    ]
    if memory_gib is not None and memory_gib > 0:
        # Mirrors shouldShowOfficialOption: never offer a model whose peak
        # memory exceeds the machine.
        models = [
            model for model in models if memory_gib >= model.peak_memory_gib
        ]
    return models


def default_catalog_model(
    *,
    memory_gib: float | None,
    chip_tier: str,
) -> CatalogModel | None:
    models = recommended_models(memory_gib=memory_gib, chip_tier=chip_tier)
    return models[0] if models else None


@dataclass(frozen=True)
class FeasibilityVerdict:
    """Single exhaustive verdict, mirroring ModelFeasibilityVerdict."""

    verdict: str  # recommended | tight_fit | insufficient_memory | insufficient_disk
    needs_gib: float | None = None

    @property
    def ok(self) -> bool:
        return self.verdict in {"recommended", "tight_fit"}


def evaluate_feasibility(
    model: CatalogModel,
    *,
    chip_tier: str,
    ram_gib: float,
    disk_free_gib: float | None = None,
) -> FeasibilityVerdict:
    safe_memory_floor = model.peak_memory_gib * MEMORY_SAFETY_FACTOR
    if disk_free_gib is not None:
        disk_required = model.download_gib * DISK_MULTIPLIER
        if disk_free_gib < disk_required:
            return FeasibilityVerdict("insufficient_disk", needs_gib=disk_required)
    if chip_tier == INTEL_TIER:
        return FeasibilityVerdict("insufficient_memory", needs_gib=safe_memory_floor)
    if ram_gib < model.peak_memory_gib:
        return FeasibilityVerdict("insufficient_memory", needs_gib=safe_memory_floor)
    if ram_gib < safe_memory_floor:
        return FeasibilityVerdict("tight_fit")
    return FeasibilityVerdict("recommended")


@dataclass(frozen=True)
class InstalledModel:
    """A complete model directory found in the local cache."""

    path: Path
    name: str
    size_bytes: int
    catalog: CatalogModel | None

    @property
    def display_name(self) -> str:
        if self.catalog is not None:
            return self.catalog.display_name
        return self.name.replace("--", "/")


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _is_complete_install(path: Path) -> bool:
    if cached_model_is_complete(path):
        return True
    # Paired Gemma-style bundles keep weights in target/assistant subtrees.
    return (
        (path / "mtplx_pair.json").is_file()
        and (path / "target").is_dir()
        and (path / "assistant").is_dir()
    )


def scan_installed_models(
    cache_dir: str | Path | None = None,
) -> list[InstalledModel]:
    """Complete installs under the model cache, official catalog first.

    Official models are ordered by their catalog position; custom installs
    follow alphabetically. Partial downloads are excluded outright so the
    picker never offers a model that cannot load.
    """

    root = model_cache_dir(cache_dir)
    if not root.is_dir():
        return []
    installed: list[InstalledModel] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not _is_complete_install(child):
            continue
        installed.append(
            InstalledModel(
                path=child,
                name=child.name,
                size_bytes=_directory_size_bytes(child),
                catalog=catalog_model_matching(child.name),
            )
        )

    def sort_key(model: InstalledModel) -> tuple[int, int, str]:
        if model.catalog is not None:
            catalog_index = next(
                index
                for index, entry in enumerate(OFFICIAL_CATALOG)
                if entry.id == model.catalog.id
            )
            return (0, catalog_index, model.name.lower())
        return (1, 0, model.name.lower())

    return sorted(installed, key=sort_key)


def installed_catalog_ids(
    installed: Iterable[InstalledModel],
) -> set[str]:
    return {
        model.catalog.id for model in installed if model.catalog is not None
    }
