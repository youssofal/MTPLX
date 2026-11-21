# Ported from mlx-vlm (Apache-2.0), Blaizzy/mlx-vlm, adapted to MTPLX checkpoint naming.
"""Image decoding and Qwen3-VL patch preprocessing (Pillow + numpy only)."""

from __future__ import annotations

import io
import math

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_DIM = 8000

_DEFAULT_MEAN = (0.5, 0.5, 0.5)
_DEFAULT_STD = (0.5, 0.5, 0.5)


def decode_image(data: bytes, *, preserve_alpha: bool = False) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"image payload is {len(data)} bytes, limit is {MAX_IMAGE_BYTES}"
        )
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
            raise ValueError(
                f"image is {width}x{height}, limit is {MAX_IMAGE_DIM} per side"
            )
        image = ImageOps.exif_transpose(image)
        transparent = "A" in image.getbands() or "transparency" in image.info
        return image.convert("RGBA" if preserve_alpha and transparent else "RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"cannot decode image: {exc}") from exc


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _resolve_pixel_bounds(preprocessor_config: dict) -> tuple[int, int]:
    size = preprocessor_config.get("size") or {}
    min_pixels = size.get("shortest_edge", 56 * 56)
    max_pixels = size.get("longest_edge", 14 * 14 * 4 * 1280)
    if preprocessor_config.get("min_pixels") is not None:
        min_pixels = preprocessor_config["min_pixels"]
    if preprocessor_config.get("max_pixels") is not None:
        max_pixels = preprocessor_config["max_pixels"]
    return int(min_pixels), int(max_pixels)


def preprocess_images(
    images: list[Image.Image], preprocessor_config: dict
) -> tuple[mx.array, list[tuple[int, int, int]]]:
    patch_size = int(preprocessor_config.get("patch_size", 16))
    merge_size = int(preprocessor_config.get("merge_size", 2))
    temporal_patch_size = int(preprocessor_config.get("temporal_patch_size", 2))
    min_pixels, max_pixels = _resolve_pixel_bounds(preprocessor_config)
    do_rescale = bool(preprocessor_config.get("do_rescale", True))
    rescale_factor = float(preprocessor_config.get("rescale_factor", 1 / 255.0))
    do_normalize = bool(preprocessor_config.get("do_normalize", True))
    image_mean = preprocessor_config.get("image_mean", _DEFAULT_MEAN)
    image_std = preprocessor_config.get("image_std", _DEFAULT_STD)

    factor = patch_size * merge_size
    ps = patch_size
    tps = temporal_patch_size
    ms = merge_size

    all_patches: list[np.ndarray] = []
    grid_thw: list[tuple[int, int, int]] = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        resized_h, resized_w = smart_resize(
            height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        if (resized_w, resized_h) != (width, height):
            image = image.resize((resized_w, resized_h), resample=Image.BICUBIC)

        img = np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))
        c = img.shape[0]
        img = img.astype(np.float32)
        if do_rescale:
            img = img * rescale_factor
        if do_normalize:
            mean = np.array(image_mean, dtype=np.float32)[:, None, None]
            std = np.array(image_std, dtype=np.float32)[:, None, None]
            img = (img - mean) / std

        # Still images are duplicated along T to fill the temporal patch.
        patches = np.repeat(img[None, None, ...], tps, axis=1)

        grid_t = 1
        grid_h = resized_h // ps
        grid_w = resized_w // ps

        patches = patches.reshape(
            1,
            grid_t,
            tps,
            c,
            grid_h // ms,
            ms,
            ps,
            grid_w // ms,
            ms,
            ps,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        all_patches.append(
            patches.reshape(grid_t * grid_h * grid_w, c * tps * ps * ps)
        )
        grid_thw.append((grid_t, grid_h, grid_w))

    pixel_values = mx.array(np.concatenate(all_patches, axis=0))
    return pixel_values, grid_thw


def image_pad_token_count(grid: tuple[int, int, int], merge_size: int = 2) -> int:
    t, h, w = grid
    return (int(t) * int(h) * int(w)) // (merge_size**2)
