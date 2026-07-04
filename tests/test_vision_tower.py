"""Vision tower, preprocessing, and spec resolution tests (no network, no real model)."""

from __future__ import annotations

import io
import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from PIL import Image

from mtplx.vision import load_vision_tower, vision_spec_for_model_dir
from mtplx.vision.processing import (
    MAX_IMAGE_BYTES,
    decode_image,
    image_pad_token_count,
    preprocess_images,
    smart_resize,
)
from mtplx.vision.qwen3_vl_tower import Qwen3VLVisionConfig, Qwen3VLVisionTower

TINY_CONFIG = Qwen3VLVisionConfig(
    depth=2,
    hidden_size=32,
    intermediate_size=64,
    out_hidden_size=64,
    num_heads=2,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    in_channels=3,
    num_position_embeddings=16,
    deepstack_visual_indexes=[0, 1],
)

TINY_PREPROCESSOR_CONFIG = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "min_pixels": 32 * 32,
    "max_pixels": 16777216,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


def _random_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(
        rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    )


def test_tower_forward_shapes():
    pixel_values, grid_thw = preprocess_images(
        [_random_image(96, 64)], TINY_PREPROCESSOR_CONFIG
    )
    assert grid_thw == [(1, 4, 6)]
    assert pixel_values.shape == (24, 3 * 2 * 16 * 16)

    tower = Qwen3VLVisionTower(TINY_CONFIG)
    embeddings, deepstack = tower(pixel_values, grid_thw)
    mx.eval(embeddings)

    # (64 / 32) * (96 / 32) = 6 tokens after the 2x2 spatial merge.
    assert embeddings.shape == (6, 64)
    assert np.isfinite(np.array(embeddings, copy=False)).all()

    assert [layer for layer, _ in deepstack] == [0, 1]
    for _, features in deepstack:
        mx.eval(features)
        assert features.shape == (6, 64)


def test_tower_forward_multiple_images():
    pixel_values, grid_thw = preprocess_images(
        [_random_image(64, 64), _random_image(96, 64)], TINY_PREPROCESSOR_CONFIG
    )
    assert grid_thw == [(1, 4, 4), (1, 4, 6)]

    tower = Qwen3VLVisionTower(TINY_CONFIG)
    embeddings, deepstack = tower(pixel_values, grid_thw)
    mx.eval(embeddings)

    assert embeddings.shape == (4 + 6, 64)
    assert len(deepstack) == 2


def test_smart_resize_factor_rounding():
    assert smart_resize(100, 200, factor=32, min_pixels=1024, max_pixels=16777216) == (
        96,
        192,
    )


def test_smart_resize_min_pixel_clamp():
    # 32x32 = 1024 < 4096, beta = 2, both sides scale to 64.
    assert smart_resize(32, 32, factor=32, min_pixels=4096, max_pixels=16777216) == (
        64,
        64,
    )


def test_smart_resize_min_clamp_recovers_zero_rounding():
    # round(8 / 32) * 32 = 0; the min-pixel branch rescales to (32, 192).
    assert smart_resize(8, 64, factor=32, min_pixels=4096, max_pixels=16777216) == (
        32,
        192,
    )


def test_smart_resize_max_pixel_clamp():
    # beta = 1000 / 256, floor(1000 / beta / 32) * 32 = 256 on both sides.
    assert smart_resize(1000, 1000, factor=32, min_pixels=1024, max_pixels=65536) == (
        256,
        256,
    )


def test_smart_resize_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError):
        smart_resize(8050, 40, factor=32, min_pixels=1024, max_pixels=16777216)


def test_image_pad_token_count():
    assert image_pad_token_count((1, 4, 6)) == 6
    assert image_pad_token_count((2, 8, 10), merge_size=2) == 40


def test_decode_image_rejects_oversized_payload():
    with pytest.raises(ValueError):
        decode_image(b"\0" * (MAX_IMAGE_BYTES + 1))


def test_decode_image_rejects_oversized_dimensions():
    buffer = io.BytesIO()
    Image.new("RGB", (8001, 8)).save(buffer, format="PNG")
    with pytest.raises(ValueError):
        decode_image(buffer.getvalue())


def test_decode_image_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        decode_image(b"definitely not an image")


def test_decode_image_roundtrip():
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color=(200, 10, 10)).save(buffer, format="PNG")
    image = decode_image(buffer.getvalue())
    assert image.mode == "RGB"
    assert image.size == (40, 30)


def test_vision_spec_none_without_vision_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert vision_spec_for_model_dir(tmp_path) is None


def test_vision_spec_none_without_vision_tower_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {"patch_size": 16}})
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}})
    )
    assert vision_spec_for_model_dir(tmp_path) is None


def test_vision_spec_none_without_index(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {"patch_size": 16}})
    )
    assert vision_spec_for_model_dir(tmp_path) is None


def test_vision_spec_populated(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "image_token_id": 248056,
                "video_token_id": 248057,
                "vision_start_token_id": 248053,
                "vision_end_token_id": 248054,
                "vision_config": {
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "out_hidden_size": 5120,
                },
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"vision_tower.pos_embed.weight": "model.safetensors"}}
        )
    )
    spec = vision_spec_for_model_dir(tmp_path)
    assert spec is not None
    assert spec.image_token_id == 248056
    assert spec.video_token_id == 248057
    assert spec.vision_start_token_id == 248053
    assert spec.vision_end_token_id == 248054
    assert spec.spatial_merge_size == 2
    assert spec.patch_size == 16
    assert spec.temporal_patch_size == 2
    assert spec.out_hidden_size == 5120


# --- qwen3_5_moe checkpoints store the vision tower under model.visual.*
#     rather than vision_tower.* ----------------------------------------------

_QWEN3_5_MOE_VISION_CONFIG = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "vision_config": {
        "model_type": "qwen3_5_moe_vision",
        "deepstack_visual_indexes": [],
        "depth": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
        "out_hidden_size": 32,
        "num_heads": 2,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "num_position_embeddings": 16,
    },
}


def _write_qwen3_5_moe_fixture(model_dir, *, weights: bool) -> None:
    """Write a qwen3_5_moe-style model dir: config.json + index (+ shards).

    Vision weights are written under ``model.visual.*`` as HF's
    Qwen3_5MoeForConditionalGeneration layout stores them, so the loader
    must recognise that prefix rather than only ``vision_tower.*``.
    """
    (model_dir / "config.json").write_text(json.dumps(_QWEN3_5_MOE_VISION_CONFIG))
    if not weights:
        # Index-only fixture: one vision tensor + one text tensor is enough
        # to exercise prefix detection.
        weight_map = {
            "model.visual.pos_embed.weight": "model.safetensors",
            "model.embed_tokens.weight": "model.safetensors",
        }
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        return

    tower = Qwen3VLVisionTower(
        Qwen3VLVisionConfig.from_dict(_QWEN3_5_MOE_VISION_CONFIG["vision_config"])
    )
    # Re-prefix every tower parameter to the HF model.visual.* layout.
    raw = {f"model.visual.{path}": v for path, v in tree_flatten(tower.parameters())}
    # A non-vision tensor so the shard isn't vision-only.
    raw["model.embed_tokens.weight"] = mx.zeros((10, 32), dtype=mx.float16)
    mx.save_safetensors(str(model_dir / "model.safetensors"), raw)
    weight_map = {key: "model.safetensors" for key in raw}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_vision_spec_detects_model_visual_prefix(tmp_path):
    """A qwen3_5_moe index with model.visual.* keys must resolve to a spec."""
    _write_qwen3_5_moe_fixture(tmp_path, weights=False)
    spec = vision_spec_for_model_dir(tmp_path)
    assert spec is not None
    assert spec.image_token_id == 248056
    assert spec.out_hidden_size == 32


def test_load_vision_tower_loads_model_visual_prefix(tmp_path):
    """load_vision_tower must load a model.visual.* checkpoint end-to-end."""
    _write_qwen3_5_moe_fixture(tmp_path, weights=True)
    tower = load_vision_tower(tmp_path)
    pixel_values, grid_thw = preprocess_images(
        [_random_image(96, 64)], TINY_PREPROCESSOR_CONFIG
    )
    embeddings, _ = tower(pixel_values, grid_thw)
    mx.eval(embeddings)
    assert embeddings.shape == (6, 32)
    assert np.isfinite(np.array(embeddings, copy=False)).all()


# --- splice window helpers (MTP history alignment, issue #103) --------------


def _make_splice(pad_id: int, rows: int, hidden: int = 8):
    from mtplx.vision.splice import VisionSplice

    return VisionSplice(
        image_pad_token_id=pad_id,
        embeddings=mx.arange(rows * hidden, dtype=mx.float32).reshape(rows, hidden)
        + 1000.0,
    )


class _IdentityEmbed:
    """Fake embed_tokens: token id t -> row of value t (easy to assert on)."""

    def __call__(self, ids):
        base = ids.astype(mx.float32)
        return mx.broadcast_to(base[..., None], (*ids.shape, 8)).astype(mx.float32)


def test_spliced_embeddings_for_window_replaces_correct_rows():
    from mtplx.vision.splice import spliced_embeddings_for_window

    pad = 99
    splice = _make_splice(pad, rows=4)
    # Prompt: [t, PAD, PAD, t, PAD, t, PAD] — window covers tokens 3..7
    # (one PAD before the window, so rows_before=2: pads at idx 1, 2).
    window = mx.array([[5, pad, 7, pad]])
    out = spliced_embeddings_for_window(
        _IdentityEmbed(), window, splice, rows_before=2
    )
    assert out is not None
    got = np.array(out)
    # Non-pad positions keep the token embedding.
    assert np.allclose(got[0, 0], 5.0)
    assert np.allclose(got[0, 2], 7.0)
    # Pad positions get vision rows 2 and 3 (rows_before=2 offset).
    expected_row2 = np.array(splice.embeddings[2])
    expected_row3 = np.array(splice.embeddings[3])
    assert np.allclose(got[0, 1], expected_row2)
    assert np.allclose(got[0, 3], expected_row3)
    # Cursor untouched — the trunk owns the sequential cursor.
    assert splice.cursor == 0


def test_spliced_embeddings_for_window_none_without_pads():
    from mtplx.vision.splice import spliced_embeddings_for_window

    splice = _make_splice(99, rows=2)
    out = spliced_embeddings_for_window(
        _IdentityEmbed(), mx.array([[1, 2, 3]]), splice, rows_before=0
    )
    assert out is None


def test_spliced_embeddings_for_window_overflow_raises():
    from mtplx.vision.splice import spliced_embeddings_for_window

    splice = _make_splice(99, rows=1)
    with pytest.raises(ValueError, match="window overflow"):
        spliced_embeddings_for_window(
            _IdentityEmbed(), mx.array([[99, 99]]), splice, rows_before=0
        )


def test_trunk_and_history_windows_share_rows():
    """The history window is the trunk chunk shifted one token right; the
    vision rows each pad receives must agree between the two lanes."""
    from mtplx.vision.splice import (
        spliced_chunk_embeddings,
        spliced_embeddings_for_window,
    )

    pad = 99
    prompt = [1, pad, pad, 2, pad, 3]
    splice = _make_splice(pad, rows=3)
    embed = _IdentityEmbed()
    prompt_arr = mx.array([prompt])

    # Trunk consumes the body [1, pad, pad, 2, pad] sequentially.
    trunk = spliced_chunk_embeddings(embed, prompt_arr[:, :5], splice)
    assert splice.cursor == 3
    # History window = prompt[1:6] = [pad, pad, 2, pad, 3], rows_before=0.
    hist = spliced_embeddings_for_window(
        embed, prompt_arr[:, 1:6], splice, rows_before=0
    )
    trunk_np, hist_np = np.array(trunk), np.array(hist)
    # Same prompt position => same vision row in both lanes:
    # prompt idx 1 is trunk col 1 and history col 0, etc.
    for prompt_idx in (1, 2, 4):
        assert np.allclose(trunk_np[0, prompt_idx], hist_np[0, prompt_idx - 1])
