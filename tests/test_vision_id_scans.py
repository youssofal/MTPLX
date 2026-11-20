"""The image id scans give exactly what they gave before they were made fast.

An image request at 150K tokens ran its ids through these helpers several times
(the splice, every keyed view, the repair gate, the position tables), each one a
Python step per id: 112 ms of host work per request with every image cached, at
the founder's shape on 12 Nov (155,912 tokens, 18 screenshots). They now use
the list's own scans and numpy. Each is checked here against its previous
implementation, kept below as the reference, on random layouts: images as one
run, split runs, images that touch, images with no rows, too many and too few
placeholders, keyed and raw sequences, a video pad.
"""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np
import pytest

from mtplx.server import openai as server
from mtplx.vision import splice as vs
from mtplx.vision.mrope import build_mrope_positions
from mtplx.vision.splice import VisionSplice

PAD = 248056
VIDEO = 248057


# -- the previous implementations, verbatim --------------------------------


def ref_vision_bank_key_ids(prompt_ids, splice):
    digests = splice.image_digests
    pad_counts = splice.pad_counts
    if not digests or not pad_counts or len(digests) != len(pad_counts):
        return None
    pad_id = splice.image_pad_token_id
    total_pads = sum(1 for token in prompt_ids if token == pad_id)
    if total_pads != sum(int(count) for count in pad_counts):
        return None
    keyed = list(prompt_ids)
    salts = vs._position_scheme_salts(splice, len(digests))
    image_idx = 0
    row_in_image = 0
    for pos, token in enumerate(keyed):
        if token != pad_id:
            continue
        while row_in_image >= int(pad_counts[image_idx]):
            image_idx += 1
            row_in_image = 0
        mixed = (
            (int(digests[image_idx]) ^ (row_in_image * vs._BANK_KEY_MIX) ^ salts[image_idx])
            & vs._BANK_KEY_MASK
        )
        keyed[pos] = vs._BANK_KEY_FLAG | mixed
        row_in_image += 1
    return keyed


def ref_vision_image_spans(prompt_ids, splice):
    pad_counts = splice.pad_counts
    if not pad_counts:
        return None
    pad_id = splice.image_pad_token_id
    positions = [
        pos for pos, token in enumerate(prompt_ids)
        if token == pad_id or (int(token) & vs._BANK_KEY_FLAG)
    ]
    if len(positions) != sum(int(c) for c in pad_counts):
        return None
    spans = []
    cursor = 0
    for count in pad_counts:
        count = int(count)
        if count <= 0:
            continue
        run = positions[cursor: cursor + count]
        spans.append((run[0], run[-1] + 1))
        cursor += count
    return spans


def ref_image_key_runs(keyed_ids):
    runs = []
    start = None
    for position, token in enumerate(keyed_ids):
        if vs.is_image_key(token):
            if start is None:
                start = position
        elif start is not None:
            runs.append((start, position))
            start = None
    if start is not None:
        runs.append((start, len(keyed_ids)))
    return runs


def ref_unkeyed_ids(keyed_ids, image_pad_token_id):
    pad = int(image_pad_token_id)
    return [pad if vs.is_image_key(token) else int(token) for token in keyed_ids]


def ref_expand_image_pads(prompt_ids, *, image_pad_id, pad_counts):
    expanded = []
    image_index = 0
    for token in prompt_ids:
        if token == image_pad_id:
            if image_index >= len(pad_counts):
                raise ValueError("prompt contains more image placeholders than images")
            expanded.extend([token] * pad_counts[image_index])
            image_index += 1
        else:
            expanded.append(token)
    if image_index != len(pad_counts):
        raise ValueError(
            "chat template dropped image placeholders: "
            f"expected {len(pad_counts)}, found {image_index}"
        )
    return expanded


def ref_build_mrope_positions(input_ids, *, image_token_id, image_grids, spatial_merge_size,
                              video_token_id=None):
    ids = list(int(t) for t in input_ids)
    n = len(ids)
    if video_token_id is not None and any(t == int(video_token_id) for t in ids):
        return None
    merge = max(1, int(spatial_merge_size))
    pad_id = int(image_token_id)
    chunks = []
    st = 0
    next_pos = 0
    for t, h, w in image_grids:
        try:
            ed = ids.index(pad_id, st)
        except ValueError:
            return None
        llm_t = int(t)
        llm_h = int(h) // merge
        llm_w = int(w) // merge
        block = llm_t * llm_h * llm_w
        if ed + block > n or any(x != pad_id for x in ids[ed: ed + block]):
            return None
        text_len = ed - st
        if text_len:
            text = np.arange(next_pos, next_pos + text_len, dtype=np.int32)
            chunks.append(np.broadcast_to(text, (3, text_len)).copy())
            next_pos += text_len
        t_idx = np.repeat(np.arange(llm_t, dtype=np.int32), llm_h * llm_w)
        h_idx = np.tile(np.repeat(np.arange(llm_h, dtype=np.int32), llm_w), llm_t)
        w_idx = np.tile(np.arange(llm_w, dtype=np.int32), llm_t * llm_h)
        chunks.append(np.stack([t_idx, h_idx, w_idx]) + next_pos)
        next_pos += max(llm_t, llm_h, llm_w)
        st = ed + block
    if any(x == pad_id for x in ids[st:]):
        return None
    tail = n - st
    if tail:
        text = np.arange(next_pos, next_pos + tail, dtype=np.int32)
        chunks.append(np.broadcast_to(text, (3, tail)).copy())
        next_pos += tail
    if not chunks:
        return None
    table = np.concatenate(chunks, axis=1).astype(np.int32)
    if table.shape != (3, n):
        return None
    return table, int(next_pos - n)


# -- random layouts -----------------------------------------------------------


def _layout(rng: random.Random):
    """Text ids with images: (raw ids with one placeholder each, pad counts,
    grids, expanded ids), occasionally malformed."""

    images = rng.randrange(0, 6)
    grids, counts = [], []
    for _ in range(images):
        shape = rng.choice([(1, 4, 4), (1, 8, 6), (1, 2, 2), (2, 4, 4), (1, 6, 10)])
        grids.append(shape)
        counts.append(shape[0] * (shape[1] // 2) * (shape[2] // 2))
    text = [rng.randrange(0, 240000) for _ in range(rng.randrange(0, 400))]
    raw = list(text)
    for _ in range(images):
        raw.insert(rng.randrange(0, len(raw) + 1), PAD)
    if images and rng.random() < 0.15:
        counts[rng.randrange(images)] = 0
    if rng.random() < 0.1:
        raw.insert(rng.randrange(0, len(raw) + 1), PAD)  # one placeholder too many
    if images and rng.random() < 0.1:
        raw.remove(PAD)  # one too few
    if rng.random() < 0.05:
        raw.insert(rng.randrange(0, len(raw) + 1), VIDEO)
    return raw, counts, grids


def _splice(rng: random.Random, counts, grids, scheme: str) -> VisionSplice:
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.zeros((max(1, sum(counts)), 4)),
        image_digests=tuple(rng.getrandbits(64) for _ in counts) or None,
        pad_counts=tuple(counts) or None,
        image_grids=tuple(grids) or None,
        mrope_table=mx.zeros((3, 1)) if scheme == "qwen4" else None,
        mrope_delta=3 if scheme == "qwen4" else 0,
    )


def _expanded_or_error(fn, raw, counts):
    try:
        return fn(raw, image_pad_id=PAD, pad_counts=counts)
    except ValueError as exc:
        return ("error", str(exc))


@pytest.mark.parametrize("seed", range(6))
def test_expansion_matches_the_previous_scan(seed):
    rng = random.Random(seed)
    for _ in range(300):
        raw, counts, _grids = _layout(rng)
        assert _expanded_or_error(server._expand_image_pads, raw, counts) == _expanded_or_error(
            ref_expand_image_pads, raw, counts)
        assert _expanded_or_error(server._expand_image_pads, tuple(raw), counts) == _expanded_or_error(
            ref_expand_image_pads, tuple(raw), counts)


@pytest.mark.parametrize("seed", range(6))
def test_keys_spans_runs_and_unkeying_match_the_previous_scans(seed):
    rng = random.Random(100 + seed)
    for _ in range(300):
        raw, counts, grids = _layout(rng)
        splice = _splice(rng, counts, grids, rng.choice(["sequential", "qwen4"]))
        expanded = _expanded_or_error(ref_expand_image_pads, raw, counts)
        ids = raw if isinstance(expanded, tuple) else expanded
        if rng.random() < 0.2 and PAD in ids:
            # A split run: text inside an image's rows.
            at = ids.index(PAD) + 1
            ids = ids[:at] + [rng.randrange(0, 240000)] + ids[at:]
        new = vs.vision_bank_key_ids(ids, splice)
        assert new == ref_vision_bank_key_ids(ids, splice)
        assert vs.vision_bank_key_ids(tuple(ids), splice) == new
        assert vs.vision_image_spans(ids, splice) == ref_vision_image_spans(ids, splice)
        for seq in (ids, new or []):
            assert vs.image_key_runs(seq) == ref_image_key_runs(seq)
            assert vs.unkeyed_ids(seq, PAD) == ref_unkeyed_ids(seq, PAD)
        if new is not None:
            assert vs.vision_image_spans(new, splice) == ref_vision_image_spans(new, splice)
            assert all(type(token) is int for token in vs.unkeyed_ids(new, PAD))


def test_touching_images_and_an_image_at_each_end():
    splice = VisionSplice(image_pad_token_id=PAD, embeddings=mx.zeros((12, 4)),
                          image_digests=(7, 9, 11), pad_counts=(4, 4, 4),
                          image_grids=((1, 4, 4),) * 3)
    for ids in ([PAD] * 8 + [5, 6] + [PAD] * 4, [PAD] * 12, [1] + [PAD] * 12 + [2]):
        assert vs.vision_bank_key_ids(ids, splice) == ref_vision_bank_key_ids(ids, splice)
        keyed = vs.vision_bank_key_ids(ids, splice)
        assert vs.image_key_runs(keyed) == ref_image_key_runs(keyed)
    assert vs.image_key_runs([]) == [] and vs.unkeyed_ids([], PAD) == []


@pytest.mark.parametrize("seed", range(6))
def test_position_tables_match_the_previous_builder(seed):
    rng = random.Random(200 + seed)
    for _ in range(300):
        raw, counts, grids = _layout(rng)
        expanded = _expanded_or_error(ref_expand_image_pads, raw, counts)
        ids = raw if isinstance(expanded, tuple) else expanded
        if grids and rng.random() < 0.1:
            grids = grids[:-1]  # more pads than grids
        for seq in (ids, tuple(ids)):
            got = build_mrope_positions(seq, image_token_id=PAD, image_grids=grids,
                                        spatial_merge_size=2, video_token_id=VIDEO)
            want = ref_build_mrope_positions(seq, image_token_id=PAD, image_grids=grids,
                                             spatial_merge_size=2, video_token_id=VIDEO)
            if want is None:
                assert got is None
            else:
                assert got is not None and got[1] == want[1]
                assert got[0].dtype == want[0].dtype and np.array_equal(got[0], want[0])


def test_row_keys_memo_stays_bounded():
    vs._ROW_KEYS.clear()
    for digest in range(40):
        vs._image_row_keys(digest, 0, 2068)
    assert sum(key[2] for key in vs._ROW_KEYS) <= vs._ROW_KEYS_MAX_ROWS
    # The newest images are the ones kept.
    assert (39, 0, 2068) in vs._ROW_KEYS and (0, 0, 2068) not in vs._ROW_KEYS
    big = vs._image_row_keys(999, 0, vs._ROW_KEYS_MAX_ROWS + 5)
    assert len(big) == vs._ROW_KEYS_MAX_ROWS + 5 and len(vs._ROW_KEYS) == 1
