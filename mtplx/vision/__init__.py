"""Self-contained Qwen3-VL vision support for qwen3_5 / qwen3_6 checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mtplx.vision.qwen3_vl_tower import (
    Qwen3VLVisionConfig,
    Qwen3VLVisionTower,
    checkpoint_weight_map,
    resolve_vision_prefix,
)

__all__ = [
    "Qwen3VLVisionConfig",
    "Qwen3VLVisionTower",
    "VisionSpec",
    "checkpoint_weight_map",
    "load_vision_tower",
    "resolve_vision_prefix",
    "vision_spec_for_model_dir",
]

# Family defaults from mlx-vlm's qwen3_5 ModelConfig; real checkpoints carry
# explicit values in config.json which always win.
_DEFAULT_IMAGE_TOKEN_ID = 248056
_DEFAULT_VIDEO_TOKEN_ID = 248057
_DEFAULT_VISION_START_TOKEN_ID = 248045
_DEFAULT_VISION_END_TOKEN_ID = 248046


@dataclass(frozen=True)
class VisionSpec:
    model_dir: str
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    spatial_merge_size: int
    patch_size: int
    temporal_patch_size: int
    out_hidden_size: int
    # M-RoPE contract from text_config.rope_parameters, for families whose
    # attention ropes image tokens at (t, h, w) grid positions: qwen4_exp in
    # its own attention, the dense qwen3_5 packs through mtplx.dense_mrope
    # (which reads the section layout from the config at load). None when the
    # family doesn't declare sections; serving then keeps plain sequential
    # rope, which is what pre-mrope families expect.
    model_type: str = ""
    mrope_section: tuple[int, ...] | None = None
    mrope_interleaved: bool = False


def _read_json(path: Path) -> dict | None:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def vision_spec_for_model_dir(path: str | Path) -> VisionSpec | None:
    model_dir = Path(path)

    config = _read_json(model_dir / "config.json")
    if config is None:
        return None
    vision_config = config.get("vision_config")
    if not isinstance(vision_config, dict):
        return None

    weight_map = checkpoint_weight_map(model_dir)
    if weight_map is None or resolve_vision_prefix(weight_map) is None:
        return None

    text_config = config.get("text_config")
    if config.get("model_type") == "inkling2_loop":
        return VisionSpec(model_dir=str(model_dir), image_token_id=200054, video_token_id=-1,
                          vision_start_token_id=200005, vision_end_token_id=200010,
                          spatial_merge_size=vision_config["adapter_merge_size"], patch_size=vision_config["native_patch_size"],
                          temporal_patch_size=vision_config["temporal_patch_size"],
                          out_hidden_size=text_config["hidden_size"], model_type="inkling2_loop")
    rope_parameters = (
        text_config.get("rope_parameters") if isinstance(text_config, dict) else None
    )
    mrope_section: tuple[int, ...] | None = None
    mrope_interleaved = False
    if isinstance(rope_parameters, dict):
        raw_section = rope_parameters.get("mrope_section")
        if isinstance(raw_section, (list, tuple)) and raw_section:
            mrope_section = tuple(int(x) for x in raw_section)
        mrope_interleaved = bool(rope_parameters.get("mrope_interleaved", False))

    return VisionSpec(
        model_dir=str(model_dir),
        image_token_id=int(config.get("image_token_id", _DEFAULT_IMAGE_TOKEN_ID)),
        video_token_id=int(config.get("video_token_id", _DEFAULT_VIDEO_TOKEN_ID)),
        vision_start_token_id=int(
            config.get("vision_start_token_id", _DEFAULT_VISION_START_TOKEN_ID)
        ),
        vision_end_token_id=int(
            config.get("vision_end_token_id", _DEFAULT_VISION_END_TOKEN_ID)
        ),
        spatial_merge_size=int(vision_config.get("spatial_merge_size", 2)),
        patch_size=int(vision_config.get("patch_size", 16)),
        temporal_patch_size=int(vision_config.get("temporal_patch_size", 2)),
        out_hidden_size=int(vision_config.get("out_hidden_size", 5120)),
        model_type=str(config.get("model_type", "")),
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )


_TOWER_CACHE: dict[str, Qwen3VLVisionTower] = {}


def load_vision_tower(path: str | Path) -> Qwen3VLVisionTower:
    key = str(Path(path).resolve())
    tower = _TOWER_CACHE.get(key)
    if tower is None:
        tower = Qwen3VLVisionTower.from_model_dir(key)
        _TOWER_CACHE[key] = tower
    return tower
