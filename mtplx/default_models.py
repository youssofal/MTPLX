"""Hardware-aware verified default model selection for product CLI paths."""

from __future__ import annotations

import math
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mtplx.hardware import classify_apple_silicon_generation, detect_apple_silicon
from mtplx.model_catalog import (
    LEGACY_TIER,
    MODERN_TIER,
    CatalogModel,
    catalog_model_with_id,
    recommended_models,
)
from mtplx.profiles import (
    DEFAULT_FP16_PUBLIC_MODEL_ID,
    DEFAULT_FP16_HF_MODEL_ID,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_PUBLIC_MODEL_ID,
    FLASH_NEXT_OPTIMIZED_QUALITY_HF_MODEL_ID,
    FLASH_NEXT_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID,
    BONSAI_OPTIMIZED_SPEED_HF_MODEL_ID,
    BONSAI_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    BONSAI_LEGACY_PUBLIC_MODEL_ID,
    BONSAI_LEGACY_LOCAL_NAME,
    FLASH_NEXT_BARE_SPEED_HF_MODEL_ID,
    FLASH_NEXT_BARE_SPEED_PUBLIC_MODEL_ID,
    FLASH_NEXT_OPTIMIZED_SPEED_HF_MODEL_ID,
    FLASH_NEXT_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
    MIMO_V26_QWEN_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
    MIMO_V26_QWEN_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    OPTIMIZED_SPEED_V1_HF_MODEL_ID,
    OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID,
    OPTIMIZED_SPEED_V2_HF_MODEL_ID,
    OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID,
    QUALITY_FP16_HF_MODEL_ID,
    QUALITY_FP16_PUBLIC_MODEL_ID,
    QUALITY_HF_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN38_BARE_SPEED_HF_MODEL_ID,
    QWEN38_BARE_SPEED_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN38_BARE_SPEED_FP16_HF_MODEL_ID,
    QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
)


DEFAULT_MODEL_VARIANT_ENV = "MTPLX_DEFAULT_MODEL_VARIANT"
SPEED_MODEL_ENV = "MTPLX_OPTIMIZED_SPEED_MODEL"
QWEN38_BARE_SPEED_MODEL_ENV = "MTPLX_QWEN38_BARE_SPEED_MODEL"
QWEN38_OPTIMIZED_SPEED_MODEL_ENV = "MTPLX_QWEN38_OPTIMIZED_SPEED_MODEL"
QUALITY_MODEL_ENV = "MTPLX_OPTIMIZED_QUALITY_MODEL"
DEFAULT_MODEL_VARIANTS = frozenset({"auto", "speed", "q4", "bf16", "fp16"})
_LEGACY_APPLE_FP16_GENERATIONS = frozenset({"m1", "m2"})
_NEWER_APPLE_SPEED_GENERATIONS = frozenset({"m3", "m4", "m5"})
# Below this much unified memory the 27B default cannot load safely, so the
# default routes to the smaller pack the app's picker lists first (the 9B
# and Bonsai from its named tier bound, the 4B below 16 GiB; model_catalog.recommended_catalog_ids).
SMALL_DEFAULT_MEMORY_FLOOR_GIB = 32.0
# The smaller speed packs, in the order the app's tiers list them. Under the
# 27B floor the default is the first of these the app's tiers offer this
# machine; with unreadable memory it is the last one. MiMo V2.6 Qwen 9B sits
# where the picker puts it, ahead of the Qwen 3.5 9B: Bonsai leads every tier
# that lists MiMo, so no default changes, and the CLI still names the app's
# first pick if Bonsai's bound ever moves. There is no FP16 4B build, so
# M1/M2 Macs stop at the 9B.
_SMALL_SPEED_CATALOG_IDS = (
    "bonsai-2-27b-optimized-speed",
    "mimo-v26-qwen-9b-optimized-speed",
    "qwen35-9b-optimized-speed",
    "qwen35-4b-optimized-speed",
)
_SMALL_FP16_CATALOG_IDS = ("qwen35-9b-optimized-speed-fp16",)
INTEL_REFUSAL_MESSAGE = (
    "MTPLX runs on Apple Silicon Macs (M1 and later); this Mac has an Intel "
    "processor, so there is no model to download."
)
# V2 peaks at about 21.5 GiB, leaving practical headroom on a 32 GiB Mac.
OPTIMIZED_SPEED_V2_MEMORY_FLOOR_GIB = 32.0
# Qwen 3.8 Bare Speed measured peak 17.0 GiB (installed app, 2026-08-14); the
# same 32 GiB floor as V2 is therefore conservative.
QWEN38_BARE_SPEED_MEMORY_FLOOR_GIB = 32.0
# Qwen 3.8 Optimized Speed measured peak 23.6 GiB (installed app, 2026-08-14),
# two GiB above V2's 21.5 on the same instrument; it keeps V2's 32 GiB floor
# (the app additionally hides any pick whose peak exceeds unified memory).
QWEN38_OPTIMIZED_SPEED_MEMORY_FLOOR_GIB = 32.0
QWEN35_9B_SPEED_DESCRIPTION = "Compact 6-bit model for smaller Macs"
QWEN35_4B_SPEED_DESCRIPTION = "Compact 4-bit model for the smallest Macs"
OPTIMIZED_SPEED_V1_LABEL = "Qwen 3.6 27B Optimized Speed"
OPTIMIZED_SPEED_V1_DESCRIPTION = "Smaller 4-bit model that is a little faster for short chats"
OPTIMIZED_SPEED_V2_LABEL = "Qwen 3.6 27B Optimized Speed V2"
OPTIMIZED_SPEED_V2_DESCRIPTION = (
    "Much higher quality for coding, with dynamic 4-bit hybrid quantization "
    "and hand-tuned sensitive parts kept at up to 16-bit. Faster on long "
    "agent tasks, slightly larger, and a little slower for short chats"
)
# Qwen 3.8 trio wording (founder, 2026-08-15): plain human descriptions.
QWEN38_BARE_SPEED_LABEL = "Qwen 3.8 27B Bare Speed"
QWEN38_BARE_SPEED_DESCRIPTION = (
    "Quickest burst chat speeds. Lower quality and slower on long coding tasks"
)
QWEN38_OPTIMIZED_SPEED_LABEL = "Qwen 3.8 27B Optimized Speed"
QWEN38_OPTIMIZED_SPEED_DESCRIPTION = (
    "4-bit dynamic quant. Great coding speeds and good quality. Recommended"
)
QWEN38_OPTIMIZED_QUALITY_LABEL = "Qwen 3.8 27B Optimized Quality"
QWEN38_OPTIMIZED_QUALITY_DESCRIPTION = (
    "8-bit dynamic quant. Good coding speeds and perfect quality"
)
QWEN38_BARE_SPEED_FP16_LABEL = "Qwen 3.8 27B Bare Speed FP16"
QWEN38_OPTIMIZED_SPEED_FP16_LABEL = "Qwen 3.8 27B Optimized Speed FP16"
QWEN38_OPTIMIZED_QUALITY_FP16_LABEL = "Qwen 3.8 27B Optimized Quality FP16"
QWEN38_FP16_SUFFIX = "FP16 build for M1 and M2 Macs"
# Backward-compatible names used by integrations that mean the public default.
OPTIMIZED_SPEED_LABEL = QWEN38_OPTIMIZED_SPEED_LABEL
OPTIMIZED_SPEED_DESCRIPTION = QWEN38_OPTIMIZED_SPEED_DESCRIPTION
OPTIMIZED_QUALITY_LABEL = "Qwen3.6 27B MTPLX Optimized Quality"
OPTIMIZED_QUALITY_DESCRIPTION = "Flat8 target with INT8 MTP sidecar"
_QWEN38_BARE_SPEED_LOCAL_CANDIDATES = (
    "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed",
    # Forge-local drop-day build (forge writes the branded name directly).
    "~/.mtplx/models/Qwen3.8-27B-MTPLX-Bare-Speed",
)
_QWEN38_OPTIMIZED_SPEED_LOCAL_CANDIDATES = (
    "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
    "~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed",
)
_QWEN38_OPTIMIZED_QUALITY_LOCAL_CANDIDATES = (
    "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality",
    "~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality",
)
_QWEN38_BARE_SPEED_FP16_LOCAL_CANDIDATES = (
    "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
    "~/.mtplx/models/Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
)
_QWEN38_OPTIMIZED_SPEED_FP16_LOCAL_CANDIDATES = (
    "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
    "~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
)
_QWEN38_OPTIMIZED_QUALITY_FP16_LOCAL_CANDIDATES = (
    "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
    "~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
)
_OPTIMIZED_SPEED_V2_LOCAL_CANDIDATES = (
    "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
    "~/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
    "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
)
_OPTIMIZED_SPEED_V1_LOCAL_CANDIDATES = (
    "~/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
    "~/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized-Speed",
    "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
    "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Speed",
)
_OPTIMIZED_QUALITY_LOCAL_CANDIDATES = (
    "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Quality",
    "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality",
)
_OPTIMIZED_QUALITY_FP16_LOCAL_CANDIDATES = (
    "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
    "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
)
_OPTIMIZED_35B_SPEED_LOCAL_CANDIDATES = (
    "~/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
    "~/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe",
    "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
)
class DefaultModelUnavailable(RuntimeError):
    """No verified default model can run on this machine.

    ``message`` is the one plain sentence to show the user. Raised instead of
    returning a selection so that no caller can download or load a model
    that was never chosen; first-run callers turn it into a clean non-zero
    exit.
    """

    def __init__(
        self,
        message: str,
        *,
        chip_generation: str = "",
        chip: str = "",
        memory_gib: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.chip_generation = chip_generation
        self.chip = chip
        self.memory_gib = memory_gib


@dataclass(frozen=True)
class DefaultModelSelection:
    model: str
    hf_model: str
    variant: str
    precision: str
    chip_generation: str
    chip: str
    reason: str
    auto_selected: bool
    env_override: str | None = None
    memory_gib: float | None = None

    @property
    def display_name(self) -> str:
        # Catalog names first for packs the substring checks below would
        # mislabel (the MiMo repo name carries "9B").
        for model_id in (
            "bonsai-2-27b-optimized-speed",
            "flash-next-optimized-speed",
            "mimo-v26-qwen-9b-optimized-speed",
        ):
            pack = catalog_model_with_id(model_id)
            if pack and self.hf_model == pack.hf_model_id:
                return pack.display_name
        if "4B" in self.hf_model:
            return "Qwen3.5 4B Optimized Speed"
        if "9B" in self.hf_model:
            if self.variant == "fp16":
                return "Qwen3.5 9B Optimized Speed FP16"
            return "Qwen3.5 9B Optimized Speed"
        if self.hf_model == QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID:
            return QWEN38_OPTIMIZED_SPEED_FP16_LABEL
        if self.variant == "fp16":
            return "Qwen3.6 27B Optimized Speed FP16"
        if self.hf_model == OPTIMIZED_SPEED_V1_HF_MODEL_ID:
            return OPTIMIZED_SPEED_V1_LABEL
        if self.hf_model == OPTIMIZED_SPEED_V2_HF_MODEL_ID:
            return OPTIMIZED_SPEED_V2_LABEL
        if self.hf_model == QWEN38_BARE_SPEED_HF_MODEL_ID:
            return QWEN38_BARE_SPEED_LABEL
        return QWEN38_OPTIMIZED_SPEED_LABEL

    @property
    def label(self) -> str:
        return f"{self.display_name}. {self.precision}. {self.reason}."

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hf_model": self.hf_model,
            "variant": self.variant,
            "precision": self.precision,
            "chip_generation": self.chip_generation,
            "chip": self.chip,
            "reason": self.reason,
            "auto_selected": self.auto_selected,
            "env_override": self.env_override,
            "memory_gib": self.memory_gib,
            "display_name": self.display_name,
            "label": self.label,
        }


def _is_complete_local_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (
        (path / "mtplx_pair.json").is_file()
        and (path / "target").is_dir()
        and (path / "assistant").is_dir()
    ):
        return True
    if not (path / "config.json").is_file():
        return False
    has_weights = any(path.glob("model-*.safetensors")) or (path / "model.safetensors").is_file()
    has_mtp = any(
        (path / rel).is_file()
        for rel in ("mtp.safetensors", "mtp/weights.safetensors", "model-mtp.safetensors")
    )
    return has_weights and has_mtp


def _env_ref_disabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "none", "off", "disabled"}


def _complete_local_model_ref(candidates: tuple[str, ...]) -> str | None:
    # Configured libraries own discovery precedence. The static candidates
    # remain as compatibility fallbacks for older Documents and staging
    # layouts, but their directory names are also probed under every ordered
    # model root so users can keep the whole collection on one volume.
    library_candidates: list[str] = []
    try:
        from mtplx.hf_loader import model_library_roots

        for root in model_library_roots():
            for candidate in candidates:
                name = Path(candidate).expanduser().name
                if name:
                    library_candidates.append(str(root / name))
    except Exception:
        # Default selection must remain usable on fresh or partially-installed
        # environments where the cache helper itself cannot be imported.
        pass

    seen: set[str] = set()
    for candidate in (*library_candidates, *candidates):
        if not candidate or _env_ref_disabled(candidate):
            continue
        path = Path(candidate).expanduser()
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if _is_complete_local_model(path):
            return str(path)
    return None


def catalog_model_ref(pack: CatalogModel) -> str:
    """Prefer a complete copy in configured libraries, including released aliases."""
    from mtplx.hf_loader import cached_model_is_complete, model_library_roots

    basename = Path(pack.hf_model_id).name
    names = (pack.hf_model_id.replace("/", "--"), basename,
             *(alias for alias in pack.aliases if "MTPLX" in alias and " " not in alias))
    for root in model_library_roots():
        for name in names:
            candidate = root / name
            if cached_model_is_complete(candidate):
                return str(candidate)
    return pack.hf_model_id


def _optimized_speed_model_ref(
    *,
    hf_model_id: str,
    local_candidates: tuple[str, ...],
    env_name: str = SPEED_MODEL_ENV,
) -> str:
    env_ref = str(os.environ.get(env_name) or "").strip()
    candidates: tuple[str, ...]
    if env_ref:
        if _env_ref_disabled(env_ref):
            return hf_model_id
        else:
            candidates = (env_ref, *local_candidates)
    else:
        candidates = local_candidates
    local = _complete_local_model_ref(candidates)
    return local or hf_model_id


def qwen38_bare_speed_model_ref() -> str:
    """Resolve the complete local release-day Qwen 3.8 Bare Speed build."""

    # The long-standing speed override's explicit value owns default model
    # selection. A V2 path must never be mistaken for this Qwen3.8 artifact,
    # hence the dedicated override below.
    if QWEN38_BARE_SPEED_MODEL_ENV not in os.environ:
        legacy_speed_override = str(os.environ.get(SPEED_MODEL_ENV) or "").strip()
        if legacy_speed_override:
            return QWEN38_BARE_SPEED_HF_MODEL_ID
    return _optimized_speed_model_ref(
        hf_model_id=QWEN38_BARE_SPEED_HF_MODEL_ID,
        local_candidates=_QWEN38_BARE_SPEED_LOCAL_CANDIDATES,
        env_name=QWEN38_BARE_SPEED_MODEL_ENV,
    )


def _legacy_speed_override_active() -> bool:
    """True when MTPLX_OPTIMIZED_SPEED_MODEL names a real 3.6-era artifact.

    Disabled spellings ("off", "0", ...) only switch local resolution off;
    they do not pin the 3.6 lane.
    """

    value = str(os.environ.get(SPEED_MODEL_ENV) or "").strip()
    return bool(value) and not _env_ref_disabled(value)


def _qwen38_speed_env_pins_artifact() -> bool:
    """True when MTPLX_QWEN38_OPTIMIZED_SPEED_MODEL names an artifact.

    A disabled spelling only switches the local 3.8 lookup off (Hub repo);
    it does not silence an explicit legacy 3.6 override.
    """

    value = str(os.environ.get(QWEN38_OPTIMIZED_SPEED_MODEL_ENV) or "").strip()
    return bool(value) and not _env_ref_disabled(value)


def _legacy_speed_override_owns_default() -> bool:
    """An explicit 3.6-era speed override keeps its lane unless a Qwen 3.8
    artifact is pinned explicitly (the more specific pin wins)."""

    return _legacy_speed_override_active() and not _qwen38_speed_env_pins_artifact()


def qwen38_optimized_speed_model_ref() -> str:
    """Resolve the Qwen 3.8 Optimized Speed default (complete local build or Hub)."""

    # Same override discipline as the Bare resolver: an explicit legacy
    # MTPLX_OPTIMIZED_SPEED_MODEL points at a 3.6-era artifact and must never
    # be relabeled as this Qwen3.8 artifact.
    if _legacy_speed_override_owns_default():
        return QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID
    return _optimized_speed_model_ref(
        hf_model_id=QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
        local_candidates=_QWEN38_OPTIMIZED_SPEED_LOCAL_CANDIDATES,
        env_name=QWEN38_OPTIMIZED_SPEED_MODEL_ENV,
    )


def qwen38_optimized_quality_model_ref() -> str:
    """Resolve the Qwen 3.8 Optimized Quality pick (complete local build or Hub)."""

    local = _complete_local_model_ref(_QWEN38_OPTIMIZED_QUALITY_LOCAL_CANDIDATES)
    return local or QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID


def qwen38_optimized_speed_fp16_model_ref() -> str:
    """Resolve the Qwen 3.8 Optimized Speed FP16 sibling (M1/M2 default)."""

    local = _complete_local_model_ref(_QWEN38_OPTIMIZED_SPEED_FP16_LOCAL_CANDIDATES)
    return local or QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID


def qwen38_bare_speed_fp16_model_ref() -> str:
    """Resolve the Qwen 3.8 Bare Speed FP16 sibling (M1/M2)."""

    local = _complete_local_model_ref(_QWEN38_BARE_SPEED_FP16_LOCAL_CANDIDATES)
    return local or QWEN38_BARE_SPEED_FP16_HF_MODEL_ID


def qwen38_optimized_quality_fp16_model_ref() -> str:
    """Resolve the Qwen 3.8 Optimized Quality FP16 sibling (M1/M2)."""

    local = _complete_local_model_ref(_QWEN38_OPTIMIZED_QUALITY_FP16_LOCAL_CANDIDATES)
    return local or QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID


def optimized_speed_model_ref() -> str:
    """Resolve the 3.6 V2 coding artifact without relabeling a V1 folder."""

    return _optimized_speed_model_ref(
        hf_model_id=OPTIMIZED_SPEED_V2_HF_MODEL_ID,
        local_candidates=_OPTIMIZED_SPEED_V2_LOCAL_CANDIDATES,
    )


def optimized_speed_v1_model_ref() -> str:
    """Resolve the original smaller speed model for lower-memory Macs."""

    return _optimized_speed_model_ref(
        hf_model_id=OPTIMIZED_SPEED_V1_HF_MODEL_ID,
        local_candidates=_OPTIMIZED_SPEED_V1_LOCAL_CANDIDATES,
    )


def optimized_quality_model_ref(
    *,
    hardware: Mapping[str, Any] | None = None,
) -> str:
    env_ref = str(os.environ.get(QUALITY_MODEL_ENV) or "").strip()
    env_disabled = _env_ref_disabled(env_ref) if env_ref else False
    if env_ref and not env_disabled:
        local = _complete_local_model_ref((env_ref,))
        if local:
            return local
    # A quality pick on legacy (M1/M2) silicon resolves the FP16 sibling,
    # mirroring the speed lane's precision routing (2.0.1, 2026-07-07).
    hardware_info = dict(detect_apple_silicon() if hardware is None else hardware)
    generation = _hardware_generation(hardware_info)
    if generation in _LEGACY_APPLE_FP16_GENERATIONS:
        if not env_disabled:
            local = _complete_local_model_ref(_OPTIMIZED_QUALITY_FP16_LOCAL_CANDIDATES)
            if local:
                return local
        return QUALITY_FP16_HF_MODEL_ID
    if env_disabled:
        return QUALITY_HF_MODEL_ID
    local = _complete_local_model_ref(_OPTIMIZED_QUALITY_LOCAL_CANDIDATES)
    return local or QUALITY_HF_MODEL_ID


def is_optimized_quality_model_ref(model: str | Path | None) -> bool:
    if model is None:
        return False
    text = str(model).strip()
    if not text:
        return False
    if text == QUALITY_HF_MODEL_ID:
        return True
    return "qwen3.6-27b-mtplx-optimized-quality" in text.lower()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _public_model_id_from_metadata(path: Path) -> str | None:
    """Resolve identity from the artifact's explicit runtime contract only.

    Canonical ``mtplx-*`` ids are a first-party claim, so they require a
    true first-party match: an explicit ``public_model_id`` /
    ``served_model_id`` / ``model_id`` written into ``mtplx_runtime.json``,
    or an exact first-party name (handled by ``_public_model_id_from_name``
    on the path). The old fuzzy lanes — ``artifact_role`` substring
    matching, ``verified_on`` model-string inference, ``precision_variant``
    coercion, family-name coercion, and quantization-layout inference —
    all mislabeled third-party builds (nom666/samuelfaj Qwopus artifacts,
    issue #57, PR #77) and were removed in July 2026. Third-party builds
    made with MTPLX tooling carry the same metadata shapes, so nothing
    short of an explicit id is proof of identity.
    """

    runtime = _read_json(path / "mtplx_runtime.json")
    for key in ("public_model_id", "served_model_id", "model_id"):
        value = runtime.get(key)
        if isinstance(value, str) and value.strip():
            # Existing Bonsai packs keep working while advertising the current id.
            if value.strip().lower() == BONSAI_LEGACY_PUBLIC_MODEL_ID:
                return BONSAI_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
            return _sanitize_public_model_id(value)
    inferred = _public_model_id_from_name(str(path))
    if inferred:
        return inferred
    # Symlinks are identity-preserving, not inference: a link into the
    # canonical store serves the canonical artifact (issue #268 — turbo and
    # the served id were lost behind /tmp symlinks). Copies under neutral
    # names still need an explicit id claim per the July 2026 fence above.
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if resolved != path:
        return _public_model_id_from_name(str(resolved))
    return None


def _sanitize_public_model_id(value: str) -> str:
    lowered = str(value).strip().lower()
    lowered = lowered.replace("_", "-")
    lowered = re.sub(r"[^a-z0-9.-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-.")
    return lowered or DEFAULT_PUBLIC_MODEL_ID


def _ref_name_components(text: str) -> set[str]:
    """Complete artifact-name components of a model ref, lowered.

    A ref names its artifact as a whole path component (local dirs), an
    HF repo id ("org/name" — also derived from adjacent components and
    from HF cache dirs "models--org--name"), or the bare ref string.
    First-party names are matched against these with EQUALITY: substring
    matching against the whole ref claimed derivative artifacts whose
    folder name merely CONTAINS a first-party name (…-V3-RC, …-V2) as
    the first-party id — the served id, health payload, and app model
    chip then all lied about which model was loaded (reported live
    2026-07-31; same class as issue #57 / PR #77).
    """

    lowered = text.strip().replace("\\", "/").lower()
    parts = [part for part in lowered.split("/") if part]
    components = set(parts)
    components.add(lowered)
    for left, right in zip(parts, parts[1:]):
        components.add(f"{left}/{right}")
    for part in parts:
        if "--" not in part:
            continue
        # "--" joins org/name segments in both HF cache dirs
        # (models--org--name) and first-party local dir names
        # (Youssofal--Name): expose the terminal name and the org/name
        # pair as components of their own.
        segments = [seg for seg in part.split("--") if seg]
        if not segments:
            continue
        components.add(segments[-1])
        if len(segments) >= 2:
            components.add(f"{segments[-2]}/{segments[-1]}")
    return components


def _public_model_id_from_name(value: str) -> str | None:
    """Map exact first-party names (public ids, HF repo ids, released
    folder names) to their canonical public ids.

    Every pattern here is a complete first-party artifact name, compared
    for EQUALITY against the ref's name components (never substrings of
    the whole ref). Loose family matches were removed in July 2026
    (issue #57 / PR #77 class), and substring matches were removed
    2026-07-31 after a derivative folder (…-Optimized-Speed-V3-RC) was
    served under the flagship id: any name that merely resembles or
    extends a first-party artifact falls through to the sanitized
    artifact name.
    """

    text = value.strip()
    if not text:
        return None
    lowered = text.replace("\\", "/").lower()
    components = _ref_name_components(text)
    for public_id, repo_id, aliases in (
        (FLASH_NEXT_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID, FLASH_NEXT_OPTIMIZED_QUALITY_HF_MODEL_ID,
         ("flash-next-optimized-quality",)),
        (BONSAI_OPTIMIZED_SPEED_PUBLIC_MODEL_ID, BONSAI_OPTIMIZED_SPEED_HF_MODEL_ID,
         ("bonsai-2-27b-optimized-speed", BONSAI_LEGACY_PUBLIC_MODEL_ID, BONSAI_LEGACY_LOCAL_NAME)),
        (MIMO_V26_QWEN_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID, MIMO_V26_QWEN_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
         ("mimo-v26-qwen-9b-optimized-speed",)),
    ):
        names = {public_id, repo_id.lower(), Path(repo_id).name.lower(), *(a.lower() for a in aliases)}
        if components & names:
            return public_id
    if FLASH_NEXT_BARE_SPEED_PUBLIC_MODEL_ID in components:
        return FLASH_NEXT_BARE_SPEED_PUBLIC_MODEL_ID
    if FLASH_NEXT_OPTIMIZED_SPEED_PUBLIC_MODEL_ID in components:
        return FLASH_NEXT_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if FLASH_NEXT_BARE_SPEED_HF_MODEL_ID.lower() in components:
        return FLASH_NEXT_BARE_SPEED_PUBLIC_MODEL_ID
    if FLASH_NEXT_OPTIMIZED_SPEED_HF_MODEL_ID.lower() in components:
        return FLASH_NEXT_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if "qwen3.8-flash-next-mtplx-bare-speed" in components:
        return FLASH_NEXT_BARE_SPEED_PUBLIC_MODEL_ID
    if "qwen3.8-flash-next-mtplx-optimized-speed" in components:
        return FLASH_NEXT_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID in components:
        return QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID in components:
        return QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID.lower() in components:
        return QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID.lower() in components:
        return QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if "qwen3.5-9b-mtplx-optimized-speed-fp16" in components:
        return QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if "qwen3.5-9b-mtplx-optimized-speed" in components:
        return QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if any("qwen3.5-9b-mtplx-speed-6bit" in part for part in components):
        # Deliberate wildcard family (released artifacts are named
        # Qwen-Qwen3.5-9B-MTPLX-Speed-6bit-<suffix>): containment is
        # per-component, so it can no longer match across path pieces.
        return QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID in components:
        return QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID
    if QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID in components:
        return QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID
    if QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID.lower() in components:
        return QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID
    if QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID.lower() in components:
        return QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID
    if "qwen3.6-35b-a3b-mtplx-optimized-balance-fp16" in components:
        return QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID
    if "qwen3.6-35b-a3b-mtplx-optimized-balance" in components:
        return QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID
    if QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID in components:
        return QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID in components:
        return QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID.lower() in components:
        return QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID.lower() in components:
        return QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if "qwen3.6-35b-a3b-mtplx-optimized-speed-fp16" in components:
        return QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if "qwen3.6-35b-a3b-mtplx-optimized-speed" in components:
        return QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if "qwen3.6-35b-a3b-mtplx-official4-cyankiwimtp-cleanrecipe" in components:
        # First-party local research build of the released 35B speed
        # artifact (listed in _OPTIMIZED_35B_SPEED_LOCAL_CANDIDATES).
        return QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID in components:
        return QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID
    if QWEN38_BARE_SPEED_FP16_HF_MODEL_ID.lower() in components:
        return QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID
    if "qwen3.8-27b-mtplx-bare-speed-fp16" in components:
        return QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID
    if QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID in components:
        return QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID
    if QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID.lower() in components:
        return QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID
    if "qwen3.8-27b-mtplx-optimized-quality-fp16" in components:
        return QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID
    if QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID in components:
        return QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID.lower() in components:
        return QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if "qwen3.8-27b-mtplx-optimized-speed-fp16" in components:
        return QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID
    if QWEN38_BARE_SPEED_PUBLIC_MODEL_ID in components:
        return QWEN38_BARE_SPEED_PUBLIC_MODEL_ID
    if QWEN38_BARE_SPEED_HF_MODEL_ID.lower() in components:
        return QWEN38_BARE_SPEED_PUBLIC_MODEL_ID
    if "qwen3.8-27b-mtplx-bare-speed" in components:
        return QWEN38_BARE_SPEED_PUBLIC_MODEL_ID
    if QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID in components:
        return QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID
    if QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID.lower() in components:
        return QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID
    if "qwen3.8-27b-mtplx-optimized-quality" in components:
        return QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID
    if QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID in components:
        return QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID.lower() in components:
        return QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if "qwen3.8-27b-mtplx-optimized-speed" in components:
        return QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-quality-fp16" in components:
        return QUALITY_FP16_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-quality" in components:
        return QUALITY_PUBLIC_MODEL_ID
    if OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID in components:
        return OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID
    if OPTIMIZED_SPEED_V2_HF_MODEL_ID.lower() in components:
        return OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-speed-v2" in components:
        return OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-speed-fp16" in components:
        return DEFAULT_FP16_PUBLIC_MODEL_ID
    if OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID in components:
        return OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID
    if OPTIMIZED_SPEED_V1_HF_MODEL_ID.lower() in components:
        return OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-speed" in components:
        return OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID
    legacy_names = {
        "qwen3.6-27b-mtplx-optimized",
        "youssofal--qwen3.6-27b-mtplx-optimized",
    }
    basename = Path(text).name.lower()
    if basename in legacy_names or lowered.endswith("/qwen3.6-27b-mtplx-optimized"):
        return LEGACY_OPTIMIZED_PUBLIC_MODEL_ID
    return None


def public_model_id_for_ref(
    model: str | Path | None,
    *,
    default_model_id: str = DEFAULT_PUBLIC_MODEL_ID,
) -> str:
    """Return the served OpenAI model id for the selected artifact.

    The default public id is only used when no model was provided. Once a
    concrete repo/path exists, MTPLX should report that artifact instead of
    silently claiming the speed default.
    """

    if model is None:
        return default_model_id
    text = str(model).strip()
    if not text:
        return default_model_id
    path = Path(text).expanduser()
    if path.is_dir():
        inferred = _public_model_id_from_metadata(path)
        if inferred:
            return inferred
    inferred = _public_model_id_from_name(text)
    if inferred:
        return inferred
    basename = Path(text).name or text.split("/")[-1]
    return _sanitize_public_model_id(basename)


def forge_local_public_model_id(branded_name: str, model_path: str | Path) -> str | None:
    """Served id Forge stamps on a pack it built, or None to leave it unset.

    First-party names already resolve to their canonical id by name at serve
    time, so they get no stamp. Any other pack gets the sanitized branded
    name: the id ``public_model_id_for_ref`` would infer from a directory of
    that name, but pinned in ``mtplx_runtime.json`` so it survives a rename,
    a ``-1`` suffix from ``_unique_model_dir`` or a copy of the pack.
    """

    if not re.search(r"[A-Za-z0-9]", str(branded_name or "")):
        return None
    path = Path(model_path)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for name in (str(branded_name), str(path), str(resolved)):
        if _public_model_id_from_name(name):
            return None
    return _sanitize_public_model_id(branded_name)


def _normalize_variant(value: str | None) -> tuple[str, str | None]:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto"}:
        return "auto", None
    aliases = {
        "speed": "speed",
        "optimized-speed": "speed",
        "q4": "speed",
        "int4": "speed",
        "bf16": "speed",
        "bfloat16": "speed",
        "bfloat": "speed",
        "fp16": "fp16",
        "float16": "fp16",
        "f16": "fp16",
    }
    normalized = aliases.get(raw)
    if normalized is None:
        return "auto", raw
    return normalized, raw


def _hardware_generation(hardware: Mapping[str, Any]) -> str:
    generation = str(hardware.get("apple_silicon_generation") or "").strip().lower()
    if generation:
        return generation
    return classify_apple_silicon_generation(
        str(hardware.get("chip") or ""),
        system=str(hardware.get("system") or ""),
        machine=str(hardware.get("machine") or ""),
    )


def _hardware_memory_gib(hardware: Mapping[str, Any]) -> float | None:
    value = hardware.get("memory_gib")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def pack_fits_memory(pack: CatalogModel, memory_gib: float) -> bool:
    """Whether the app offers this pack on a Mac with ``memory_gib`` of
    unified memory: its measured peak fits. This is the picker's own rule
    (``shouldShowOfficialOption`` / ``recommended_models``), so the CLI
    default and the app's recommendation can never name different packs
    for the same machine."""

    return memory_gib >= float(pack.peak_memory_gib)


def minimum_memory_gib_for_pack(pack: CatalogModel) -> int:
    """Smallest whole number of GiB at which ``pack_fits_memory`` holds."""

    return int(math.ceil(float(pack.peak_memory_gib)))


def _small_pack_ladder(variant: str) -> tuple[CatalogModel, ...]:
    ids = _SMALL_FP16_CATALOG_IDS if variant == "fp16" else _SMALL_SPEED_CATALOG_IDS
    ladder = tuple(catalog_model_with_id(model_id) for model_id in ids)
    if any(pack is None for pack in ladder):
        raise RuntimeError("the small-model ladder names a pack missing from the catalog")
    return ladder  # type: ignore[return-value]


def _small_pack_offered_first(variant: str, memory_gib: float) -> CatalogModel | None:
    """The first small speed pack the app's picker would list for this Mac.

    ``recommended_models`` is the catalog's RAM-tiered order with the
    peak-memory filter applied (the 4B pair leads below 16 GiB, Bonsai
    leads 16-31 GiB; M1/M2 have only the FP16 9B), restricted here to the
    speed ladder so the default is never a quality build.
    """

    ladder_ids = {pack.id for pack in _small_pack_ladder(variant)}
    tier = LEGACY_TIER if variant == "fp16" else MODERN_TIER
    for pack in recommended_models(memory_gib=memory_gib, chip_tier=tier):
        if pack.id in ladder_ids:
            return pack
    return None


def _small_pack_precision(pack: CatalogModel, variant: str) -> str:
    if variant == "fp16":
        return "FP16"
    if pack.id == "bonsai-2-27b-optimized-speed":
        return pack.detail
    return QWEN35_4B_SPEED_DESCRIPTION if "4B" in pack.hf_model_id else QWEN35_9B_SPEED_DESCRIPTION


def select_default_model(
    *,
    variant_override: str | None = None,
    hardware: Mapping[str, Any] | None = None,
) -> DefaultModelSelection:
    """Select the verified default model for this machine.

    Auto policy is intentionally simple and visible: M1/M2 -> FP16, modern
    Macs from 256 GiB -> Flash-Next Optimized Speed; 32-255 GiB ->
    Qwen 3.8 Optimized Speed (the complete local
    build when installed, otherwise the published Hub repo), under 32 GiB ->
    the smaller pack the app's picker lists first for that much memory (the
    Bonsai from its named tier bound, the 4B below 16 GiB), and memory that could not be read -> the
    smallest pack. An explicit legacy
    MTPLX_OPTIMIZED_SPEED_MODEL override keeps the 3.6 Optimized Speed V2
    lane it was written for.

    Raises ``DefaultModelUnavailable`` (one plain sentence) on an Intel Mac,
    which cannot run any MTPLX model, and on a Mac with less memory than the
    smallest pack needs.
    """

    env_value = variant_override if variant_override is not None else os.environ.get(DEFAULT_MODEL_VARIANT_ENV)
    requested_variant, invalid_override = _normalize_variant(env_value)
    hardware_info = dict(detect_apple_silicon() if hardware is None else hardware)
    generation = _hardware_generation(hardware_info)
    chip = str(hardware_info.get("chip") or "").strip()
    memory_gib = _hardware_memory_gib(hardware_info)

    if generation == "intel":
        raise DefaultModelUnavailable(
            INTEL_REFUSAL_MESSAGE,
            chip_generation=generation,
            chip=chip,
            memory_gib=memory_gib,
        )

    if requested_variant == "fp16":
        variant = "fp16"
        reason = f"forced by {DEFAULT_MODEL_VARIANT_ENV}=fp16"
        auto_selected = False
    elif requested_variant == "speed":
        variant = "speed"
        if str(env_value or "").strip().lower() in {"bf16", "bfloat16", "bfloat"}:
            reason = f"forced by {DEFAULT_MODEL_VARIANT_ENV}={env_value} (legacy alias for optimized speed)"
        else:
            reason = f"forced by {DEFAULT_MODEL_VARIANT_ENV}=speed"
        auto_selected = False
    elif generation in _LEGACY_APPLE_FP16_GENERATIONS:
        variant = "fp16"
        reason = "selected for M1/M2 Apple Silicon"
        auto_selected = True
    else:
        variant = "speed"
        auto_selected = True
        if generation in _NEWER_APPLE_SPEED_GENERATIONS:
            reason = "selected for newer Apple Silicon"
        else:
            reason = "selected because hardware is unknown"

    if invalid_override is not None and requested_variant == "auto":
        reason = f"{reason}; ignored invalid {DEFAULT_MODEL_VARIANT_ENV}={invalid_override}"

    # Memory routing. The variant override still controls precision; memory
    # only changes the model size, mirroring the app's recommendation tiers.
    small_pack: CatalogModel | None = None
    if memory_gib is None:
        # Unreadable memory used to route to the 27B. Nothing here can say
        # what fits, so take the smallest pack and say why; --model overrides.
        small_pack = _small_pack_ladder(variant)[-1]
        reason = (
            f"{reason}; selected the smallest model because this Mac's memory "
            "could not be read (pass --model to choose another)"
        )
    elif memory_gib < SMALL_DEFAULT_MEMORY_FLOOR_GIB:
        # The same tiers and peak-memory filter the app's picker uses, so
        # `mtplx start` and first-run onboarding in the app name one pack.
        small_pack = _small_pack_offered_first(variant, memory_gib)
        if small_pack is None:
            smallest = _small_pack_ladder(variant)[-1]
            raise DefaultModelUnavailable(
                f"MTPLX needs at least {minimum_memory_gib_for_pack(smallest)} GB of "
                f"memory to run its smallest model ({smallest.display_name}) on this "
                f"Mac, which has {memory_gib:.0f} GB.",
                chip_generation=generation,
                chip=chip,
                memory_gib=memory_gib,
            )
        size_label = small_pack.display_name
        reason = f"{reason}; routed to {size_label} for {memory_gib:.0f} GiB unified memory"

    qwen38_model = qwen38_optimized_speed_model_ref()
    legacy_speed_override = _legacy_speed_override_owns_default()
    use_qwen38 = (
        variant == "speed"
        and generation not in _LEGACY_APPLE_FP16_GENERATIONS
        and not legacy_speed_override
        and (
            memory_gib is None
            or memory_gib >= QWEN38_OPTIMIZED_SPEED_MEMORY_FLOOR_GIB
        )
    )
    use_v2 = (
        variant == "speed"
        and generation not in _LEGACY_APPLE_FP16_GENERATIONS
        and (
            memory_gib is None
            or memory_gib >= OPTIMIZED_SPEED_V2_MEMORY_FLOOR_GIB
        )
    )
    large_pack = None
    if variant == "speed" and not legacy_speed_override and memory_gib is not None:
        candidate = recommended_models(memory_gib=memory_gib, chip_tier=MODERN_TIER)[0]
        explicit_speed = str(os.environ.get(QWEN38_OPTIMIZED_SPEED_MODEL_ENV) or "").strip()
        if candidate.id == "flash-next-optimized-speed" and (not explicit_speed or _env_ref_disabled(explicit_speed)):
            large_pack = candidate
    if large_pack is not None:
        model = catalog_model_ref(large_pack)
        hf_model = large_pack.hf_model_id
        precision = large_pack.detail
    elif small_pack is not None:
        model = catalog_model_ref(small_pack)
        hf_model = small_pack.hf_model_id
        precision = _small_pack_precision(small_pack, variant)
    elif variant == "fp16":
        if legacy_speed_override:
            # An explicit 3.6-era speed override keeps its FP16 sibling.
            model = DEFAULT_FP16_HF_MODEL_ID
            hf_model = DEFAULT_FP16_HF_MODEL_ID
            precision = "FP16"
        else:
            model = qwen38_optimized_speed_fp16_model_ref()
            hf_model = QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID
            precision = f"{QWEN38_FP16_SUFFIX}. {QWEN38_OPTIMIZED_SPEED_DESCRIPTION}"
            if model != hf_model:
                reason = f"{reason}; installed locally"
    elif use_qwen38:
        model = qwen38_model
        hf_model = QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID
        precision = QWEN38_OPTIMIZED_SPEED_DESCRIPTION
        if model != hf_model:
            reason = f"{reason}; installed locally"
    elif use_v2:
        model = optimized_speed_model_ref()
        hf_model = OPTIMIZED_SPEED_V2_HF_MODEL_ID
        precision = OPTIMIZED_SPEED_V2_DESCRIPTION
        if model != hf_model:
            reason = f"{reason}; installed locally"
    else:
        model = optimized_speed_v1_model_ref()
        hf_model = OPTIMIZED_SPEED_V1_HF_MODEL_ID
        precision = OPTIMIZED_SPEED_V1_DESCRIPTION
        if memory_gib is not None:
            reason = (
                f"{reason}; selected the smaller model for "
                f"{memory_gib:.0f} GiB unified memory"
            )
        if model != hf_model:
            reason = f"{reason}; installed locally"
    return DefaultModelSelection(
        model=model,
        hf_model=hf_model,
        variant=variant,
        precision=precision,
        chip_generation=generation,
        chip=chip,
        reason=reason,
        memory_gib=memory_gib,
        auto_selected=auto_selected,
        env_override=env_value if env_value else None,
    )


def verified_default_refs() -> set[str]:
    """Model ids that name this Mac's verified default by themselves.

    `mtplx setup` and `mtplx init` write the default repo id into
    config.toml, and a start setup saved without a selection record may hold
    one too. Such a ref follows this Mac's default when it moves (an M1 or M2
    Mac gets the FP16 build). Flash-Next ids stay off: smaller Macs run those
    packs by choice. No path is listed: the relative `models/...` folder the
    CLI used as its own --model default until 2026-05-15 reaches here today
    only when a user types it.
    """

    return {
        BONSAI_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        BONSAI_LEGACY_PUBLIC_MODEL_ID,
        BONSAI_OPTIMIZED_SPEED_HF_MODEL_ID,
        DEFAULT_HF_MODEL_ID,
        DEFAULT_FP16_HF_MODEL_ID,
        DEFAULT_MODEL_ID,
        OPTIMIZED_SPEED_V2_HF_MODEL_ID,
        QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    }


def is_verified_default_model_ref(model: str | Path | None) -> bool:
    """Whether a model ref with no record of how it was picked follows this
    Mac's verified default.

    Only the ids in ``verified_default_refs`` do. Any other folder path
    never does, even one named like a default pack or holding the default's
    own copy: MTPLX writes the default into config.toml only as a repo id,
    and start setup records when a folder pick was the default, so an
    unrecorded folder is one the user pointed at (#573).
    """

    if model is None:
        return True
    text = str(model).strip()
    return not text or text in verified_default_refs()
