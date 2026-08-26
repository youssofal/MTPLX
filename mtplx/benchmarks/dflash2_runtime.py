"""Construction-only bridge from one MTPLX runtime to stock DFlash2 APIs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


_DRAFT_QUANT = "w4:gs64"
_CHECKPOINT_BLOCK_SIZE = 8
_TARGET_LAYER_IDS = (5, 19, 33, 47, 61)


@dataclass(frozen=True, slots=True)
class MTPLXDFlash2Bundle:
    """Validated objects needed by the unchanged DFlash2 runtime."""

    runtime: Any
    target_model: Any
    tokenizer: Any
    target_ops: Any
    draft_model: Any
    draft_backend: Any
    draft_meta: dict[str, Any]
    checkpoint_block_size: int
    target_layer_ids: tuple[int, ...]


def load_mtplx_runtime(model_path: str):
    """Load the sole target model through MTPLX."""

    from mtplx.runtime import load

    return load(model_path, mtp=True)


def load_draft(draft_ref: str, *, draft_quant: str):
    """Load the stock DFlash2 draft without importing the extra at module import."""

    from dflash_mlx.runtime.loading import load_draft_bundle

    return load_draft_bundle(draft_ref, lazy=True, draft_quant=draft_quant)


def make_target_ops():
    """Construct the stock Qwen GDN target operations object."""

    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps

    return QwenGdnTargetOps()


def bind_draft(draft_model, target_model, *, target_ops) -> None:
    """Bind the loaded draft to the already-loaded target exactly once."""

    from dflash_mlx.engine.target_ops import bind_draft_to_target

    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)


def make_draft_backend():
    """Construct the unchanged eager draft backend."""

    from dflash_mlx.draft_backend import EagerDraftBackend

    return EagerDraftBackend()


def _checkpoint_geometry(draft_model) -> tuple[int, tuple[int, ...]]:
    try:
        block_size = draft_model.block_size
    except AttributeError as error:
        raise ValueError("Qwen3.8 DFlash2 checkpoint must declare block size 8") from error
    if type(block_size) is not int:
        raise ValueError("Qwen3.8 DFlash2 checkpoint must declare block size 8")

    try:
        raw_layer_ids = draft_model.target_layer_ids
    except AttributeError as error:
        raise ValueError(
            f"Qwen3.8 DFlash2 checkpoint must declare target layer IDs "
            f"{_TARGET_LAYER_IDS}"
        ) from error
    try:
        layer_ids = tuple(raw_layer_ids)
    except TypeError as error:
        raise ValueError(
            f"Qwen3.8 DFlash2 checkpoint must declare target layer IDs "
            f"{_TARGET_LAYER_IDS}"
        ) from error
    if any(type(value) is not int for value in layer_ids):
        raise ValueError(
            f"Qwen3.8 DFlash2 checkpoint must declare target layer IDs "
            f"{_TARGET_LAYER_IDS}"
        )
    return block_size, layer_ids


def _install_checkpoint_capabilities(draft_model) -> None:
    try:
        capabilities = draft_model.capabilities
        draft_model.capabilities = replace(
            capabilities,
            default_block_tokens=_CHECKPOINT_BLOCK_SIZE,
            max_block_tokens=_CHECKPOINT_BLOCK_SIZE,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Qwen3.8 DFlash2 draft must expose replaceable runtime capabilities"
        ) from error


def load_mtplx_dflash2_bundle(
    model_path: str,
    draft_ref: str,
) -> MTPLXDFlash2Bundle:
    """Validate and bind stock DFlash2 to one MTPLX-loaded target model."""

    runtime = load_mtplx_runtime(model_path)
    target_model = runtime.model
    target_ops = make_target_ops()

    if not target_ops.supports_model(target_model):
        raise ValueError("QwenGdnTargetOps does not support the MTPLX-loaded target")
    family = target_ops.family(target_model)
    if family != "hybrid_gdn":
        raise ValueError(
            "DFlash2 requires target family hybrid_gdn, "
            f"got {family!r}"
        )

    draft_model, draft_meta = load_draft(draft_ref, draft_quant=_DRAFT_QUANT)
    block_size, layer_ids = _checkpoint_geometry(draft_model)
    if block_size != _CHECKPOINT_BLOCK_SIZE:
        raise ValueError(
            "Qwen3.8 DFlash2 checkpoint must have block size 8, "
            f"got {block_size}"
        )
    if layer_ids != _TARGET_LAYER_IDS:
        raise ValueError(
            "Qwen3.8 DFlash2 checkpoint target layer IDs must be "
            f"{_TARGET_LAYER_IDS}, got {layer_ids}"
        )

    _install_checkpoint_capabilities(draft_model)
    bind_draft(draft_model, target_model, target_ops=target_ops)
    return MTPLXDFlash2Bundle(
        runtime=runtime,
        target_model=target_model,
        tokenizer=runtime.tokenizer,
        target_ops=target_ops,
        draft_model=draft_model,
        draft_backend=make_draft_backend(),
        draft_meta=dict(draft_meta),
        checkpoint_block_size=block_size,
        target_layer_ids=layer_ids,
    )
