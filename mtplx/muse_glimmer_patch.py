"""Register the vendored ``muse_glimmer_text`` model class for MTPLX.

No released mlx-lm ships a muse_glimmer model class, so ``mlx_lm.utils.load``
(and therefore ``mtplx serve`` / inspect) cannot build the Muse-Glimmer text
tower without this shim. The vendored class lives in
``mtplx.vendored_muse_glimmer_text``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def is_muse_glimmer_config(config: dict[str, Any]) -> bool:
    """True for a Muse-Glimmer text checkpoint (converted text tower or the
    multimodal wrapper's text_config)."""
    model_type = str(config.get("model_type", "")).lower()
    if model_type in ("muse_glimmer_text", "muse_glimmer"):
        return True
    architectures = [str(a) for a in config.get("architectures") or []]
    return any("museglimmer" in a.lower() for a in architectures)


def install_muse_glimmer_model_shim() -> None:
    """Register the vendored Muse-Glimmer classes under ``mlx_lm.models`` so
    ``mlx_lm.utils.load`` can resolve them. Idempotent.

    * ``muse_glimmer_text`` -> the text backbone
    * ``muse_glimmer``      -> the qwen3_vl-style multimodal wrapper
    """
    from . import vendored_muse_glimmer, vendored_muse_glimmer_text

    for name, mod in (
        ("mlx_lm.models.muse_glimmer_text", vendored_muse_glimmer_text),
        ("mlx_lm.models.muse_glimmer", vendored_muse_glimmer),
    ):
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "Model", None) is not None:
            continue
        sys.modules[name] = mod
        logger.info("[muse-glimmer] vendored model registered as %s", name)
