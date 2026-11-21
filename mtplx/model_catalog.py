"""Shared official-model catalog with RAM-aware recommendation tiers.

SYNC PAIR: this module is the Python port of the macOS app's catalog and
feasibility policy. When editing either side, update the other in the same
change:

- apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXModelOption.swift
  (``officialCatalog``, ``recommendedCatalogIDs``/``recommendationIDs``)
- apps/MTPLXApp/Sources/MTPLXAppCore/Onboarding/ModelFeasibility.swift
  (``memorySafetyFactor``, ``downloadHeadroomGiB``, verdict rules)

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

from mtplx.hf_loader import cached_model_is_complete, model_library_roots

MEMORY_SAFETY_FACTOR = 1.5
# `mtplx pull` needs the bytes still to download plus this headroom. Files land
# as `.incomplete` beside their final names and are renamed in place.
DOWNLOAD_HEADROOM_GIB = 5.0
# Bonsai 2 leads from this much RAM; its measured 8K-context peak is 11.80 GiB.
BONSAI_RECOMMENDATION_MIN_GIB = 16.0
# Inkling 2 Mini Bare Speed on 24 to 31 GiB Macs, directly after the tier's first
# pick. The memory plan admits it there (12,288 tokens on 24 GiB with the family's
# measured reserve), but in a simulated 24 GiB envelope an image turn and a 4K
# prompt peaked at 19.44 GiB against its 18 GiB engine budget, and 12K tokens
# leave an agent client little room. Off until the founder decides; True is the
# whole switch (mirrors MTPLXModelOption.inklingTightMacOffer).
INKLING_TIGHT_MAC_OFFER = False
# The Inkling 2 Mini FP16 siblings on M1 and M2 Macs (32 GiB and up), in the places the
# modern tier gives the trio. Their FP16 numbers are checked on an M5; no M1 or M2 Mac has
# loaded them yet. Off until one has; True is the whole switch (mirrors
# MTPLXModelOption.inklingLegacyOffer).
INKLING_LEGACY_OFFER = False

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
        size_bytes=2_567_456_768,
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
        size_bytes=4_576_426_393,
        peak_memory_gib=4.75,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen35-4b-optimized-quality",
            "qwen3.5-4b-mtplx-optimized-quality",
            "Qwen3.5 4B Optimized Quality",
            "Qwen 3.5 4B Quality",
        ),
    ),
    # MiMo V2.6 Qwen 9B (2026-09-23): Xiaomi's supervised fine-tune of Qwen3.5-9B
    # for agentic coding (XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B, MIT), packed like
    # the 6-bit 9B: 6-bit group-64 body, the Qwen3.5-9B MTP head and the vision
    # tower in BF16. The architecture is plain Qwen 3.5, so it takes the Qwen 3.5
    # contract, never the MiMo one. No FP16 sibling exists, so M1 and M2 are not
    # offered it. Mirrors MTPLXModelOption.officialCatalog.
    CatalogModel(
        id="mimo-v26-qwen-9b-optimized-speed",
        display_name="MiMo V2.6 Qwen 9B Optimized Speed",
        detail="6-bit quantization. Xiaomi's agentic coding distill of Qwen 3.5 9B.",
        hf_model_id="Youssofal/MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
        # Published Hub bytes (2026-09-23). Same geometry and bytes as the 6-bit
        # 9B, so its peak carries over; measured 8.70 GiB at 15K tokens.
        size_bytes=8_695_116_595,
        peak_memory_gib=10.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-mimo-v26-qwen-9b-optimized-speed",
            "MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
        ),
    ),
    CatalogModel(
        id="qwen35-9b-optimized-speed",
        display_name="Qwen 3.5 9B Optimized Speed",
        detail="6-bit quantization. Strong small-Mac speed pick.",
        hf_model_id="Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed",
        size_bytes=8_695_118_659,
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
        size_bytes=7_783_301_181,
        peak_memory_gib=10.5,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen35-9b-optimized-speed-fp16",
            "Qwen3.5 9B Optimized Speed FP16",
            "Qwen 3.5 9B Speed FP16",
        ),
    ),
    CatalogModel(
        id="qwen38-27b-bare-speed",
        display_name="Qwen 3.8 27B Bare Speed",
        detail=(
            "Quickest burst chat speeds. Lower quality and slower on long "
            "coding tasks."
        ),
        hf_model_id="Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
        # Exact byte sum of the published HF repo files (2026-08-15 tree API;
        # three trunk shards + bf16 MTP sidecar + restored bf16 vision tower
        # (#263) + tokenizer + card).
        size_bytes=16_313_700_871,
        # Measured 2026-08-14: request-log MLX high-water 19.6 GiB during
        # quiet-window 2.4k-context serving (boot + Flappy arms + rung).
        peak_memory_gib=20.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen38-27b-bare-speed",
            "Qwen3.8 27B Bare Speed",
            "Qwen 3.8 Bare Speed",
            "Bare Speed",
        ),
    ),
    CatalogModel(
        id="qwen38-27b-optimized-speed",
        display_name="Qwen 3.8 27B Optimized Speed",
        detail=(
            "4-bit dynamic quant. Great coding speeds and good quality."
        ),
        hf_model_id="Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
        # Exact byte sum of the published HF repo files (2026-08-15 tree API;
        # module_overrides recipe, 5.807 bits/weight, bf16 MTP sidecar,
        # restored bf16 vision tower (#263)).
        size_bytes=20_703_486_605,
        # Measured 2026-08-14: request-log MLX high-water 24.6 GiB during
        # quiet-window 2.4k-context serving (boot + Flappy arms + rung).
        peak_memory_gib=25.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen38-27b-optimized-speed",
            "Qwen3.8 27B Optimized Speed",
            "Qwen 3.8 Optimized Speed",
        ),
    ),
    CatalogModel(
        id="qwen38-27b-optimized-quality",
        display_name="Qwen 3.8 27B Optimized Quality",
        detail=(
            "8-bit dynamic quant. Good coding speeds and perfect quality."
        ),
        hf_model_id="Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
        # Exact byte sum of the published HF repo files (2026-08-15 tree API;
        # includes the restored bf16 vision tower, #263).
        size_bytes=29_972_714_046,
        # Measured 2026-08-14: request-log MLX high-water 32.9 GiB during
        # quiet-window 2.4k-context serving (boot + Flappy arms + rung).
        peak_memory_gib=33.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-qwen38-27b-optimized-quality",
            "Qwen3.8 27B Optimized Quality",
            "Qwen 3.8 Optimized Quality",
        ),
    ),
    # Qwen 3.8 FP16 precision siblings (built 2026-08-15): the same quantized
    # packs byte for byte, every 16-bit tensor cast bf16 -> fp16, so M1 and M2
    # Macs (no native bf16) run the identical model at full speed. They are the
    # legacy (M1/M2) tier's face of the trio; the modern tier never sees them.
    CatalogModel(
        id="qwen38-27b-bare-speed-fp16",
        display_name="Qwen 3.8 27B Bare Speed FP16",
        detail=(
            "Quickest burst chat speeds. Lower quality and slower on long "
            "coding tasks. FP16 build for M1 and M2 Macs."
        ),
        hf_model_id="Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
        # Exact byte sum of the published HF repo files (2026-08-15 tree API;
        # includes the restored bf16 vision tower, #263).
        size_bytes=16_314_184_539,
        # Same packs and tensor bytes as the parent; peak carried over.
        peak_memory_gib=20.0,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen38-27b-bare-speed-fp16",
            "Qwen3.8 27B Bare Speed FP16",
            "Qwen 3.8 Bare Speed FP16",
            "Bare Speed FP16",
        ),
    ),
    CatalogModel(
        id="qwen38-27b-optimized-speed-fp16",
        display_name="Qwen 3.8 27B Optimized Speed FP16",
        detail=(
            "4-bit dynamic quant. Great coding speeds and good quality. "
            "FP16 build for M1 and M2 Macs."
        ),
        hf_model_id="Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
        # Exact byte sum of the published HF repo files (2026-08-15 tree API;
        # includes the restored bf16 vision tower, #263).
        size_bytes=20_703_971_173,
        # Same packs and tensor bytes as the parent; peak carried over.
        peak_memory_gib=25.0,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen38-27b-optimized-speed-fp16",
            "Qwen3.8 27B Optimized Speed FP16",
            "Qwen 3.8 Optimized Speed FP16",
        ),
    ),
    CatalogModel(
        id="qwen38-27b-optimized-quality-fp16",
        display_name="Qwen 3.8 27B Optimized Quality FP16",
        detail=(
            "8-bit dynamic quant. Good coding speeds and perfect quality. "
            "FP16 build for M1 and M2 Macs."
        ),
        hf_model_id="Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
        # Exact byte sum of the published HF repo files (2026-08-15 tree API;
        # includes the restored bf16 vision tower, #263).
        size_bytes=29_973_199_603,
        # Same packs and tensor bytes as the parent; peak carried over.
        peak_memory_gib=33.0,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=(
            "mtplx-qwen38-27b-optimized-quality-fp16",
            "Qwen3.8 27B Optimized Quality FP16",
            "Qwen 3.8 Optimized Quality FP16",
        ),
    ),
    # Qwen 3.8 Flash-Next (day-0 native, 2026-08-26): the 125B-A6B
    # Qwen4-generation preview (GDN hybrid MoE + Qwen Sparse Attention +
    # n-gram memory sidecar). The 32 GB n-gram table streams from SSD by
    # default, so resident peak is weights + MTP + working set, not the full
    # download size. Big-Mac exclusive: the peak-memory filter hides both
    # entries below ~96 GB unified memory. Mirrors MTPLXModelOption.
    CatalogModel(
        id="flash-next-bare-speed",
        display_name="Qwen 3.8 Flash-Next Bare Speed",
        detail="Flat 4-bit quantization. Quickest Flash-Next speeds for chat and coding.",
        hf_model_id="Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed",
        # Exact byte sum of the published HF repo (2026-08-28 audit); includes
        # the 32 GB SSD-streamed n-gram table and the vision tower.
        size_bytes=106_336_814_165,
        # Weights 72.6 GB + MTP 1.7 GB resident (n-gram on SSD) plus
        # KV/working headroom at the default profile.
        peak_memory_gib=78.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-flash-next-bare-speed",
            "Qwen3.8 Flash-Next Bare Speed",
            "Flash-Next Bare Speed",
            "Qwen3.8-Flash-Next-MTPLX-Bare-Speed",
        ),
    ),
    CatalogModel(
        id="flash-next-optimized-speed",
        display_name="Qwen 3.8 Flash-Next Optimized Speed",
        detail=(
            "Dynamic 4-bit quant with 8-bit attention. Higher quality and "
            "slightly slower."
        ),
        hf_model_id="Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
        # Exact byte sum of the published HF repo (2026-08-28 audit); includes
        # the 32 GB SSD-streamed n-gram table and the vision tower.
        size_bytes=115_061_255_377,
        # Weights 81.4 GB + MTP 1.7 GB resident (n-gram on SSD) plus
        # KV/working headroom at the default profile.
        peak_memory_gib=87.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-flash-next-optimized-speed",
            "Qwen3.8 Flash-Next Optimized Speed",
            "Flash-Next Optimized Speed",
            "Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
        ),
    ),
    CatalogModel(
        id="flash-next-optimized-quality",
        display_name="Qwen 3.8 Flash-Next Optimized Quality",
        detail="8-bit body and MTP head, BF16 structural tensors, and a 4-bit n-gram table. Higher-fidelity Flash-Next build.",
        hf_model_id="Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality",
        # Recipe: body/MTP 8-bit group 64, BF16 structural tensors, n-gram 4-bit group 32.
        # Planner need at 128K with the n-gram table streamed from SSD (calculated,
        # not yet measured on a 256 GB Mac).
        size_bytes=169_958_537_520, peak_memory_gib=136.4,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-flash-next-optimized-quality",
            "Qwen3.8-Flash-Next-MTPLX-Optimized-Quality",
            "Flash-Next Optimized Quality",
        ),
    ),
    CatalogModel(
        id="bonsai-2-27b-optimized-speed",
        display_name="Bonsai 2 27B Optimized Speed",
        detail="Prism ML ternary 27B model with vision and MTP. Compact weights for smaller Macs.",
        hf_model_id="Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
        # Published Hub bytes (2026-09-22) and the measured 8K context peak.
        size_bytes=8_847_919_618, peak_memory_gib=11.80,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-bonsai-2-27b-optimized-speed",
            "Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
            "Bonsai-3.8-27B-MTPLX-Optimized-Speed",
            "mtplx-bonsai-38-27b-optimized-speed",
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
        size_bytes=19_887_455_621,
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
        size_bytes=16_419_081_848,
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
        detail="FP16 speed artifact for M1 and M2 Macs.",
        hf_model_id="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
        # Exact sum of the published HF repo files (2026-07-03 audit); the
        # previous 16-GiB figure was a pre-publish estimate ~0.7 GiB high.
        size_bytes=16_419_644_386,
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
        size_bytes=21_014_908_552,
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
        size_bytes=21_016_116_514,
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
        size_bytes=29_671_037_163,
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
        size_bytes=29_672_249_554,
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
        size_bytes=17_715_574_397,
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
        size_bytes=30_016_961_495,
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
        detail="FP16 quality artifact for M1 and M2 Macs.",
        hf_model_id="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
        # Exact byte sum of the published HF repo files (2026-07-07 upload,
        # verified via the tree API).
        size_bytes=30_017_528_942,
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
        size_bytes=64_129_781_104,
        peak_memory_gib=74.0,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=(
            "mtplx-laguna-s21",
            "Laguna S-2.1",
            "Laguna-S-2.1-oQ4e",
        ),
        ar_only=True,
    ),
    CatalogModel(
        id="inkling2-mini-bare-speed",
        display_name="Inkling 2 Mini Bare Speed",
        detail="Fastest Inkling 2 Mini: 4-bit attention and output, with vision and speculative decoding.",
        hf_model_id="Youssofal/MTPLX-Inkling-2-Mini-Bare-Speed",
        size_bytes=16_023_243_308,
        peak_memory_gib=18.21,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=("mtplx-inkling-2-mini-bare-speed", "Inkling 2 Mini Bare Speed"),
    ),
    CatalogModel(
        id="inkling2-mini-optimized-speed",
        display_name="Inkling 2 Mini Optimized Speed",
        detail="Closer to the original weights: 8-bit attention, with vision and speculative decoding.",
        hf_model_id="Youssofal/MTPLX-Inkling-2-Mini-Optimized-Speed",
        size_bytes=17_242_658_297,
        peak_memory_gib=19.92,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=("mtplx-inkling-2-mini-optimized-speed", "Inkling 2 Mini Optimized Speed"),
    ),
    CatalogModel(
        id="inkling2-mini-optimized-quality",
        display_name="Inkling 2 Mini Optimized Quality",
        detail="Original expert and attention weights, with vision and speculative decoding.",
        hf_model_id="Youssofal/MTPLX-Inkling-2-Mini-Optimized-Quality",
        size_bytes=20_191_646_421,
        peak_memory_gib=23.52,
        recommended_tiers=frozenset({MODERN_TIER}),
        aliases=("mtplx-inkling-2-mini-optimized-quality", "Inkling 2 Mini"),
    ),
    # FP16 siblings for M1 and M2 Macs, which have no native bfloat16: the same packs with every
    # BF16 tensor cast to FP16 (vision tower and drafter included) and every NVFP4 and 4-bit code
    # byte for byte; peak carried over from the parent (same tensor bytes). Recommended on the
    # legacy tier only behind INKLING_LEGACY_OFFER (off): no M1 or M2 Mac has run them yet.
    CatalogModel(
        id="inkling2-mini-bare-speed-fp16",
        display_name="Inkling 2 Mini Bare Speed FP16",
        detail="Fastest Inkling 2 Mini: 4-bit attention and output, with vision and speculative decoding. FP16 build for M1 and M2 Macs.",
        hf_model_id="Youssofal/MTPLX-Inkling-2-Mini-Bare-Speed-FP16",
        size_bytes=16_023_246_321,
        peak_memory_gib=18.21,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=("mtplx-inkling-2-mini-bare-speed-fp16", "Inkling 2 Mini Bare Speed FP16"),
    ),
    CatalogModel(
        id="inkling2-mini-optimized-speed-fp16",
        display_name="Inkling 2 Mini Optimized Speed FP16",
        detail="Closer to the original weights: 8-bit attention, with vision and speculative decoding. FP16 build for M1 and M2 Macs.",
        hf_model_id="Youssofal/MTPLX-Inkling-2-Mini-Optimized-Speed-FP16",
        size_bytes=17_242_661_312,
        peak_memory_gib=19.92,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=("mtplx-inkling-2-mini-optimized-speed-fp16", "Inkling 2 Mini Optimized Speed FP16"),
    ),
    CatalogModel(
        id="inkling2-mini-optimized-quality-fp16",
        display_name="Inkling 2 Mini Optimized Quality FP16",
        detail="Original expert and attention weights, with vision and speculative decoding. FP16 build for M1 and M2 Macs.",
        hf_model_id="Youssofal/MTPLX-Inkling-2-Mini-Optimized-Quality-FP16",
        size_bytes=20_191_648_443,
        peak_memory_gib=23.52,
        recommended_tiers=frozenset({LEGACY_TIER}),
        aliases=("mtplx-inkling-2-mini-optimized-quality-fp16", "Inkling 2 Mini Optimized Quality FP16"),
    ),
)

# Mirrors `modernTopRecommendationIDs` in MTPLXModelOption.swift: the
# fallback matrix when hardware is unknown.
_MODERN_TOP_RECOMMENDATION_IDS = (
    "qwen38-27b-optimized-speed",
    "qwen38-27b-bare-speed",
    "qwen38-27b-optimized-quality",
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
)

# The 9B-class speed picks on modern chips. MiMo V2.6 Qwen 9B rides
# immediately ahead of the Qwen 3.5 9B wherever the 9B is offered, so it is a
# top onboarding suggestion (founder, 2026-09-23) without replacing any tier's
# first pick. Mirrors MTPLXModelOption.modernSmallIDs.
_MODERN_SMALL_IDS = (
    "mimo-v26-qwen-9b-optimized-speed",
    "qwen35-9b-optimized-speed",
)

# Inkling 2 Mini trio on modern chips (2.13): Bare Speed is the recommended
# pack. On 32 to 47 GiB Macs the trio rides directly after the tier's first
# pick (founder, 2026-11-21: recommend Bare Speed to 36 GB Macs), Bare Speed,
# then Optimized Speed, then Optimized Quality; the peak-memory filter drops a
# pack where it does not fit. From 48 GiB it keeps the family's slot near the
# end. Below 32 GiB only Bare Speed, behind INKLING_TIGHT_MAC_OFFER (off).
# Mirrors MTPLXModelOption.inklingIDs.
_INKLING_IDS = (
    "inkling2-mini-bare-speed",
    "inkling2-mini-optimized-speed",
    "inkling2-mini-optimized-quality",
)

# Flash-Next options on modern chips; Optimized Speed and Quality lead from
# 256 GiB. Mirrors MTPLXModelOption.flashNextIDs.
_FLASH_NEXT_IDS = (
    "flash-next-bare-speed",
    "flash-next-optimized-speed",
    "flash-next-optimized-quality",
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
        small = ["qwen35-9b-optimized-speed-fp16"]
        speed27 = "optimized-speed-fp16"
        speed27_v2 = None
        speed35 = "qwen36-35b-a3b-optimized-speed-fp16"
        balance35 = "qwen36-35b-a3b-optimized-balance-fp16"
        quality27 = "optimized-quality-fp16"
    else:
        small = list(_MODERN_SMALL_IDS)
        speed27 = "optimized-speed"
        speed27_v2 = "optimized-speed-v2"
        speed35 = "qwen36-35b-a3b-optimized-speed"
        balance35 = "qwen36-35b-a3b-optimized-balance"
        quality27 = "optimized-quality"
    # Qwen 3.8 trio (2026-08-15 release): Optimized Speed is the recommended
    # pick and leads every tier with at least 32 GiB, then Bare Speed and
    # Optimized Quality (the latter drops out of the 32-47 GiB tier via the
    # peak-memory filter in recommended_models). The legacy (M1/M2) tier gets
    # the same three picks as their FP16 precision siblings, same order.
    trio38 = [
        "qwen38-27b-optimized-speed",
        "qwen38-27b-bare-speed",
        "qwen38-27b-optimized-quality",
    ]
    if chip_tier == LEGACY_TIER:
        trio38 = [f"{model_id}-fp16" for model_id in trio38]
    if memory_gib is None or memory_gib <= 0:
        if chip_tier == LEGACY_TIER:
            return [*trio38, speed27, quality27, speed35, balance35, "gemma4-optimized-speed", *small]
        return list(_MODERN_TOP_RECOMMENDATION_IDS)
    # The rebuilt 4B pair leads the sub-16GB tiers and trails every larger
    # modern tier so it stays discoverable as the fast-small pick. No fp16 4B
    # siblings exist, so the legacy (M1/M2) matrix keeps its fp16-only entries.
    tiny_ids = (
        ["qwen35-4b-optimized-speed", "qwen35-4b-optimized-quality"]
        if chip_tier != LEGACY_TIER
        else []
    )
    if memory_gib < 16:
        return tiny_ids or small
    if memory_gib < 32:
        if chip_tier == LEGACY_TIER:
            return small
        bonsai = "bonsai-2-27b-optimized-speed"
        leading = [bonsai, *small] if memory_gib >= BONSAI_RECOMMENDATION_MIN_GIB else [*small, bonsai]
        # Inkling Bare Speed only behind INKLING_TIGHT_MAC_OFFER (off): a 24 GB Mac's fit is proven
        # in a simulated envelope only.
        inkling = ["inkling2-mini-bare-speed"] if INKLING_TIGHT_MAC_OFFER and memory_gib >= 24 else []
        return [*leading[:1], *inkling, *leading[1:], *tiny_ids]
    if memory_gib < 48:
        if speed27_v2 is None:
            inkling16 = [f"{model_id}-fp16" for model_id in _INKLING_IDS] if INKLING_LEGACY_OFFER else []
            return [trio38[0], *inkling16, *trio38[1:], *small, speed27, "gemma4-optimized-speed", speed35, quality27]
        return [
            trio38[0],
            *_INKLING_IDS,
            *trio38[1:],
            speed27_v2,
            speed27,
            *small,
            "gemma4-optimized-speed",
            speed35,
            quality27,
            "bonsai-2-27b-optimized-speed",
            *tiny_ids,
        ]
    # Offer Flash-Next from 96 GiB, Optimized Speed from 128 GiB: its wired
    # floor (80.3 GiB) is above a 96 GB Mac's stock GPU limit (about 72 GiB).
    # From 256 GiB Optimized Speed leads and Optimized Quality follows, as
    # the 27B trio does. The normal peak-memory filter still applies.
    flash_next = list(_FLASH_NEXT_IDS) if chip_tier != LEGACY_TIER and memory_gib >= 96 else []
    if memory_gib < 128:
        flash_next = [model_id for model_id in flash_next if model_id != "flash-next-optimized-speed"]
    leading = []
    if chip_tier != LEGACY_TIER and memory_gib >= 256:
        leading = ["flash-next-optimized-speed", "flash-next-optimized-quality"]
        flash_next = [model_id for model_id in flash_next if model_id not in leading]
    return [
        *leading,
        *trio38,
        *flash_next,
        *([speed27_v2] if speed27_v2 else []),
        speed27,
        quality27,
        speed35,
        balance35,
        "gemma4-optimized-speed",
        *small,
        *(["bonsai-2-27b-optimized-speed"] if chip_tier != LEGACY_TIER else []),
        *(
            _INKLING_IDS
            if chip_tier != LEGACY_TIER
            else tuple(f"{model_id}-fp16" for model_id in _INKLING_IDS) if INKLING_LEGACY_OFFER else ()
        ),
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
    downloaded_bytes: int = 0,
) -> FeasibilityVerdict:
    safe_memory_floor = model.peak_memory_gib * MEMORY_SAFETY_FACTOR
    if disk_free_gib is not None and model.size_bytes > 0:
        remaining_gib = max(0, model.size_bytes - downloaded_bytes) / 1_073_741_824.0
        disk_required = remaining_gib + DOWNLOAD_HEADROOM_GIB
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
    root: Path | None = None
    root_index: int = 0
    is_primary: bool = True

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
    *,
    search_dirs: Iterable[str | Path] | None = None,
) -> list[InstalledModel]:
    """Complete installs under the model cache, official catalog first.

    Official models are ordered by their catalog position; custom installs
    follow alphabetically. Partial downloads are excluded outright so the
    picker never offers a model that cannot load.
    """

    installed: list[InstalledModel] = []
    seen: set[str] = set()
    for root_index, root in enumerate(
        model_library_roots(cache_dir, search_dirs=search_dirs)
    ):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            identity = os.path.normcase(str(child.resolve(strict=False)))
            if identity in seen:
                continue
            seen.add(identity)
            if not _is_complete_install(child):
                continue
            installed.append(
                InstalledModel(
                    path=child,
                    name=child.name,
                    size_bytes=_directory_size_bytes(child),
                    catalog=catalog_model_matching(child.name),
                    root=root,
                    root_index=root_index,
                    is_primary=root_index == 0,
                )
            )

    def sort_key(model: InstalledModel) -> tuple[int, int, int, str]:
        if model.catalog is not None:
            catalog_index = next(
                index
                for index, entry in enumerate(OFFICIAL_CATALOG)
                if entry.id == model.catalog.id
            )
            return (0, catalog_index, model.root_index, model.name.lower())
        return (1, 0, model.root_index, model.name.lower())

    return sorted(installed, key=sort_key)


def installed_catalog_ids(
    installed: Iterable[InstalledModel],
) -> set[str]:
    return {
        model.catalog.id for model in installed if model.catalog is not None
    }
