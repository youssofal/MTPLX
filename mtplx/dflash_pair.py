"""dflash drafter-pair bundle helpers.

A dflash artifact is a bundle root with the *target* verifier under ``target/``
and the *dflash drafter* under ``drafter/``, described by a ``dflash_pair.json``
manifest. This mirrors :mod:`mtplx.gemma4_pair` so the native-MTP path learns no
dflash-specific assumptions, and so adding a future dflash drafter is purely
"drop a bundle" — no code.

Manifest shape (``dflash_pair.json``)::

    {
      "layout": {"target": "target", "drafter": "drafter"},
      "backend": "dflash",
      "diffusion": {"num_steps": 8},           # optional; drafter denoise budget
      "benchmark": {"best_block_size": 16, "mean_accept": 2.1}
    }

The drafter's own ``config.json`` (a :class:`~mtplx.models.dflash.DFlashConfig`)
carries ``target_layers``, ``block_size``, ``mask_token_id`` etc., so the
manifest only needs the layout + serving knobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DFLASH_PAIR_FILE = "dflash_pair.json"
DFLASH_BACKEND = "dflash"
DFLASH_ARCH_ID = "dflash-drafter-pair"


def load_dflash_pair_metadata(bundle_root: str | Path) -> dict[str, Any] | None:
    path = Path(bundle_root).expanduser() / DFLASH_PAIR_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_dflash_pair_paths(bundle_root: str | Path) -> dict[str, Any] | None:
    """Return the resolved target/drafter paths for a dflash bundle, or ``None``
    if ``bundle_root`` is not a dflash pair (so ``load()`` falls through)."""
    root = Path(bundle_root).expanduser()
    metadata = load_dflash_pair_metadata(root)
    if metadata is None:
        return None
    layout = metadata.get("layout") if isinstance(metadata.get("layout"), dict) else {}
    target = root / str(layout.get("target") or "target")
    drafter = root / str(layout.get("drafter") or "drafter")
    if not (target / "config.json").is_file() or not (drafter / "config.json").is_file():
        return None
    return {
        "bundle_root": str(root),
        "target_model": str(target),
        "drafter_model": str(drafter),
        "metadata": metadata,
    }


def dflash_pair_block_size(metadata: dict[str, Any] | None, fallback: int) -> int:
    if isinstance(metadata, dict):
        bench = metadata.get("benchmark")
        if isinstance(bench, dict):
            try:
                return int(bench["best_block_size"])
            except (KeyError, TypeError, ValueError):
                pass
    return int(fallback)


def dflash_pair_num_steps(metadata: dict[str, Any] | None, fallback: int) -> int:
    if isinstance(metadata, dict):
        diff = metadata.get("diffusion")
        if isinstance(diff, dict):
            try:
                return int(diff["num_steps"])
            except (KeyError, TypeError, ValueError):
                pass
    return int(fallback)


def dflash_pair_inspection(
    *,
    model_ref: str,
    bundle_root: str | Path,
    target_model: str | Path,
    drafter_model: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    benchmark = metadata.get("benchmark") if isinstance(metadata.get("benchmark"), dict) else {}
    return {
        "source": model_ref,
        "model_dir": str(bundle_root),
        "runtime_model": str(target_model),
        "drafter_model": str(drafter_model),
        "architecture": "DFlashDrafterPair",
        "model_type": "dflash_pair",
        "mtp_arch": DFLASH_ARCH_ID,
        "mtp_supported": True,
        "recommended_backend": DFLASH_BACKEND,
        "recommended_profile": "sustained",
        "runtime_compatibility": "drafter-pair-native",
        "dflash_pair": {
            "bundle_root": str(bundle_root),
            "target_model": str(target_model),
            "drafter_model": str(drafter_model),
            "benchmark": benchmark,
        },
        "compatibility": {
            "tier": "family-compatible-unverified",
            "can_run": True,
            "recognized": True,
            "exit_code": 0,
            "arch_id": DFLASH_ARCH_ID,
            "recommended_backend": DFLASH_BACKEND,
            "mtp_supported": "yes",
            "runtime_compatibility": "drafter-pair-native",
            "support_notes": (
                "External dflash block-diffusion drafter; target and drafter live "
                "in bundle subdirectories and load together. Verified via the "
                "acceptance@K bench."
            ),
            "unverified_model": True,
        },
    }
