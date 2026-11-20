"""Multi-axis (M-RoPE) position tables for vision requests.

Qwen-VL-family models rope image tokens at (t, h, w) grid positions while
text tokens advance all three axes together; text after an image resumes at
max+1, so the image occupies max(frames, rows, cols) positions rather than
one per token ("position contraction"). Serving image tokens with plain
sequential positions diverges from training.

The table is a pure function of the expanded prompt (ids + per-image grids):
it is recomputed per request and never persisted in cache state, so warm
restores stay format-stable. Decode continues with equal-axes positions at
sequence_index + delta, which is exactly plain rope shifted by delta — the
scalar decode path stays untouched.

Algorithm adapted from mlx-vlm's get_rope_index (MIT, Prince Canuma and
contributors), simplified to batch-1 text+image serving (no attention-mask
padding, no video).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# The tag that keys banked image prefixes roped through this table and delta
# (see mtplx.vision.splice.vision_bank_key_ids). It names how the ROWS of a
# banked entry were positioned, so it changes whenever that changes.
#
# v1 was implicit: builds 2.10.1 to 2.11.3 wrote these entries under the same
# unsalted key as sequentially roped ones. Those builds opened the position
# scope around the main verify forward only, so the rows their copy rounds,
# repairs, lazy bonus commits and final commits wrote sit |delta| positions
# off, inside entries nothing else tells apart. v2 entries are written with
# every trunk forward inside the scope. A v1 entry can no longer match at or
# past an image; nothing is deleted, it ages out through normal eviction.
BANK_KEY_SCHEME = "qwen4_mrope_v2"


def build_mrope_positions(
    input_ids: Sequence[int],
    *,
    image_token_id: int,
    image_grids: Sequence[tuple[int, int, int]],
    spatial_merge_size: int,
    video_token_id: int | None = None,
) -> tuple[np.ndarray, int] | None:
    """Return (positions [3, len] int32, rope_delta) or None when unbuildable.

    ``image_grids`` are raw (t, h, w) patch grids in prompt order; the LLM
    sees h//merge x w//merge tokens per frame. Returns None (caller falls
    back to plain sequential rope) when a video pad is present or the pad
    layout disagrees with the grids — never a wrong table.
    """

    # The list's own scans (count, index, slices) do the per-id work: a
    # Python step per id cost 20 ms a call on a 150K-token prompt, and every
    # image request builds this table more than once.
    ids = input_ids if type(input_ids) is list else list(map(int, input_ids))
    n = len(ids)
    if video_token_id is not None and ids.count(int(video_token_id)):
        return None

    merge = max(1, int(spatial_merge_size))
    pad_id = int(image_token_id)
    chunks: list[np.ndarray] = []
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
        if ed + block > n or ids[ed : ed + block] != [pad_id] * block:
            return None

        text_len = ed - st
        if text_len:
            text = np.arange(next_pos, next_pos + text_len, dtype=np.int32)
            chunks.append(np.broadcast_to(text, (3, text_len)).copy())
            next_pos += text_len

        t_idx = np.repeat(np.arange(llm_t, dtype=np.int32), llm_h * llm_w)
        h_idx = np.tile(
            np.repeat(np.arange(llm_h, dtype=np.int32), llm_w), llm_t
        )
        w_idx = np.tile(np.arange(llm_w, dtype=np.int32), llm_t * llm_h)
        chunks.append(np.stack([t_idx, h_idx, w_idx]) + next_pos)
        next_pos += max(llm_t, llm_h, llm_w)
        st = ed + block

    if pad_id in ids[st:]:
        # More pads than grids: layout mismatch, refuse rather than mis-rope.
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
    delta = int(next_pos - n)
    return table, delta
