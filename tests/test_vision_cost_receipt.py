"""Each image request's row says what its images cost (12 Nov).

The founder's rule: a screenshot already seen should cost nothing on later
turns, a new one should cost the image model once. The row now carries
``vision_cache_hits`` (images whose embedding was cached), ``vision_tower_misses``
and ``vision_tower_s`` (images the image model ran on, and its time) and
``vision_materialize_s`` (the whole step), so every turn shows which it was.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import mlx.core as mx

import mtplx.vision as vision
import mtplx.vision.processing as processing
from mtplx.server import openai
from mtplx.vision import VisionSpec

PAD = 99
GRID = (1, 4, 6)  # merge 2: six image tokens


def _state(tmp_path, tower_calls: list) -> SimpleNamespace:
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    spec = VisionSpec(
        model_dir=str(tmp_path),
        image_token_id=PAD,
        video_token_id=98,
        vision_start_token_id=1,
        vision_end_token_id=2,
        spatial_merge_size=2,
        patch_size=16,
        temporal_patch_size=2,
        out_hidden_size=64,
        model_type="qwen4_exp",
        mrope_section=(3, 3, 2),
        mrope_interleaved=True,
    )
    return SimpleNamespace(
        _vision_spec_cache=spec,
        args=SimpleNamespace(model=str(tmp_path)),
        runtime=SimpleNamespace(model=None),
    )


def _fake_tower(monkeypatch, tower_calls: list) -> None:
    def tower(pixel_values, grids):
        tower_calls.append(tuple(grids[0]))
        return mx.ones((6, 64)), None

    monkeypatch.setattr(vision, "load_vision_tower", lambda _path: tower)
    monkeypatch.setattr(processing, "decode_image", lambda raw: raw)
    monkeypatch.setattr(
        processing, "preprocess_images", lambda images, _config: (mx.zeros((4, 8)), [GRID])
    )


def _prompt(images: int) -> list[int]:
    ids = [5, 6, 7]
    for index in range(images):
        ids += [PAD, 20 + index]
    return ids


def test_a_screenshot_already_seen_costs_a_hit_and_no_tower_time(tmp_path, monkeypatch):
    calls: list = []
    _fake_tower(monkeypatch, calls)
    state = _state(tmp_path, calls)

    def turn(images: list[bytes]) -> dict:
        receipt: dict = {}
        asyncio.run(
            openai._materialize_vision_splice_observed(
                state, images, _prompt(len(images)), receipt
            )
        )
        return receipt

    first = turn([b"screenshot one"])
    assert (first["vision_cache_hits"], first["vision_tower_misses"]) == (0, 1)
    assert first["vision_tower_s"] > 0 and first["vision_materialize_s"] >= first["vision_tower_s"]
    # The next turn adds a new screenshot: the first is a hit, the new one runs.
    second = turn([b"screenshot one", b"screenshot two"])
    assert (second["vision_cache_hits"], second["vision_tower_misses"]) == (1, 1)
    # A turn that adds no image: both hits, the image model does not run.
    third = turn([b"screenshot one", b"screenshot two"])
    assert (third["vision_cache_hits"], third["vision_tower_misses"]) == (2, 0)
    assert third["vision_tower_s"] == 0.0
    assert len(calls) == 2


def test_a_failed_image_still_leaves_its_receipt(tmp_path, monkeypatch):
    calls: list = []
    _fake_tower(monkeypatch, calls)
    state = _state(tmp_path, calls)
    receipt: dict = {}

    def broken(*_args, **_kwargs):
        raise ValueError("image could not be decoded")

    monkeypatch.setattr(openai, "_materialize_vision_splice", broken)
    try:
        asyncio.run(
            openai._materialize_vision_splice_observed(state, [b"x"], _prompt(1), receipt)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("the error must reach the caller (it answers 400)")
    assert receipt["vision_tower_misses"] == 0 and "vision_materialize_s" in receipt
