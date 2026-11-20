"""Vision embedding splice for chunked prefill.

A vision request carries one embedding row per expanded image pad token,
in prompt order. Prefill consumes chunks strictly left to right on the
solo lane, so the splice is a sequential queue: each chunk replaces its
pad-token rows with the next rows from the queue. Deepstack features, if
any, ride alongside with the same ordering and are applied by the layer
injection when enabled.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import mlx.core as mx
import numpy as np


@dataclass
class VisionSplice:
    """Per-request vision state consumed during prefill."""

    image_pad_token_id: int
    embeddings: Any  # mx.array [total_pad_tokens, text_hidden_size]
    deepstack: dict[int, Any] = field(default_factory=dict)
    cursor: int = 0
    # Content identity for token-keyed caches: one 64-bit digest of the raw
    # image bytes per image, plus that image's expanded pad-token count, in
    # prompt order. None on legacy constructions; cache keying then stays
    # disabled for the request (bypass semantics).
    image_digests: tuple[int, ...] | None = None
    pad_counts: tuple[int, ...] | None = None
    # Raw (t, h, w) patch grids per image, prompt order — the M-RoPE inputs.
    image_grids: tuple[tuple[int, int, int], ...] | None = None
    # M-RoPE position table [3, expanded_prompt_len] (mx.array) and the
    # decode-time position delta, derived per request from ids + grids by
    # mtplx.vision.mrope.build_mrope_positions. None/0 for families without
    # an mrope contract; attention then keeps plain sequential rope.
    mrope_table: Any | None = None
    mrope_delta: int = 0
    # Dense Qwen3.5 / Qwen3.8 path: the request's host-side position state
    # (mtplx.dense_mrope.DenseMRopeState) when its image tokens are roped at
    # grid positions. None keeps the sequential positions that path used
    # before. It also picks the bank key scheme below, so it is decided once,
    # where the splice is built, and never changed afterwards.
    dense_mrope: Any | None = None

    @property
    def total_rows(self) -> int:
        return int(self.embeddings.shape[0])

    def remaining(self) -> int:
        return self.total_rows - self.cursor

    def reset(self) -> None:
        self.cursor = 0


# Surrogate ids live far above any real vocabulary id (vocab ~248k << 2^40)
# so a keyed sequence can never collide with a plain text prompt.
_BANK_KEY_FLAG = 1 << 62
_BANK_KEY_MIX = 0x9E3779B97F4A7C15  # golden-ratio odd constant, stable mix
_BANK_KEY_MASK = (1 << 62) - 1
# Position-scheme salts for image rows roped at grid positions: blake2b-8 of
# the scheme tag under the 62-bit mask, written out as constants so keys stay
# stable across processes and on disk. Sequentially roped requests keep the
# unsalted surrogates they always had.
#   dense path   b"dense_mrope_v1"  (mtplx.dense_mrope.SCHEME)
#   Flash-Next   b"qwen4_mrope_v2"  (mtplx.vision.mrope.BANK_KEY_SCHEME; v1
#                was the unsalted key, whose entries hold misplaced rows)
_DENSE_MROPE_KEY_SALT = 0x1B8EFCA3B1C212C2
_QWEN4_MROPE_KEY_SALT = 0x17ABB1A5A8303B77
_GRID_MIX_T = 0x100000001B3
_GRID_MIX_H = 0xC2B2AE3D27D4EB4F
_GRID_MIX_W = 0x165667B19E3779F9


def mrope_rope_state(splice: Any | None) -> tuple[Any, int] | None:
    """(table, delta) of a request roped through the Flash-Next M-RoPE scope.

    None for a text request, for a family without an M-RoPE contract, and for
    an image request whose table could not be built: those rope sequentially
    on every forward. The single predicate for "this request's rows are
    positioned by its table and delta": generation arms the attention scope
    on it and the bank key is salted on it, so the two cannot drift apart.
    """

    if splice is None:
        return None
    table = getattr(splice, "mrope_table", None)
    delta = int(getattr(splice, "mrope_delta", 0) or 0)
    if table is None and delta == 0:
        return None
    return table, delta


def _position_scheme_salt(splice: VisionSplice) -> int:
    """The salt of the scheme that positions this request's image rows."""

    if getattr(splice, "dense_mrope", None) is not None:
        return _DENSE_MROPE_KEY_SALT
    if mrope_rope_state(splice) is not None:
        return _QWEN4_MROPE_KEY_SALT
    return 0


def _position_scheme_salts(splice: VisionSplice, images: int) -> list[int]:
    """Per-image key salt: 0 for sequential positions.

    KV rows of an image, and of every token after it, depend on how the image
    was roped. A request roped at grid positions therefore must never restore
    a prefix that was roped sequentially (kill switch, fallback, an older
    build) or the other way round, and never one built for another grid. The
    salt folds the scheme and the image's (t, h, w) grid into the surrogate of
    every pad row: a prefix match can then reach past an image only when the
    text, the pixels, the scheme and the grid all agree, which is exactly when
    the position table of that prefix is the same. Rows before the first image
    are identical under both schemes and stay shareable.

    The scheme tag also carries a version: Flash-Next entries written before
    every trunk forward ran inside the position scope hold misplaced rows, and
    the tag that keys today's entries keeps those from matching past an image.
    """

    scheme_salt = _position_scheme_salt(splice)
    if not scheme_salt:
        return [0] * images
    grids = splice.image_grids
    salts: list[int] = []
    for index in range(images):
        salt = scheme_salt
        if grids is not None and len(grids) == images:
            t, h, w = (int(x) for x in grids[index])
            salt ^= (t * _GRID_MIX_T) ^ (h * _GRID_MIX_H) ^ (w * _GRID_MIX_W)
        salts.append(salt & _BANK_KEY_MASK)
    return salts


def vision_bank_key_ids(
    prompt_ids: list[int], splice: VisionSplice
) -> list[int] | None:
    """Content-true cache-key view of a vision prompt.

    Every image pad token shares one vocab id, so a token-keyed cache cannot
    tell two different images apart — the reason vision requests historically
    bypassed the session bank outright. For cache keying only, each pad
    position is remapped to a surrogate derived from its image's content
    digest and row index: the key sequence becomes a pure function of
    (text tokens, pixel content, positions). Same pixels restore exactly;
    different pixels can never match. The model input is untouched. A request
    roped at grid positions (the dense path, and Flash-Next through its
    M-RoPE table) also folds that scheme and each image's grid into its
    surrogates (see _position_scheme_salts). Text requests never come here:
    their key is their token ids.

    Returns None when the splice carries no content identity (legacy
    construction) or the pad layout does not match the supplied images;
    callers must then keep the conservative bypass behavior.
    """

    digests = splice.image_digests
    pad_counts = splice.pad_counts
    if not digests or not pad_counts or len(digests) != len(pad_counts):
        return None
    pad_id = splice.image_pad_token_id
    keyed = list(prompt_ids)
    counts = [int(count) for count in pad_counts]
    if keyed.count(pad_id) != sum(counts):
        return None
    salts = _position_scheme_salts(splice, len(digests))
    # Only the pad positions change, so only they are touched: each image's
    # rows are found with the list's own scans and written as one slice when
    # they are one run, as the expanded ids always lay them out (a request is
    # served 150K ids and 18 images several times over; a Python step per id
    # cost 20 ms a pass). Rows go to the pads in order either way.
    position = 0
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        keys = _image_row_keys(int(digests[index]), salts[index], count)
        position = keyed.index(pad_id, position)
        end = position + count
        if keyed[position:end] == [pad_id] * count:
            keyed[position:end] = keys
            position = end
            continue
        for key in keys:
            position = keyed.index(pad_id, position)
            keyed[position] = key
            position += 1
    return keyed


# Row keys of recently keyed images, by (digest, salt, rows): a conversation
# keys the same screenshots on every request. Bounded by rows, as the
# embedding cache is (about 1.4 MB at this bound).
_ROW_KEYS: OrderedDict[tuple[int, int, int], tuple[int, ...]] = OrderedDict()
_ROW_KEYS_MAX_ROWS = 32768
_ROW_KEYS_LOCK = threading.Lock()


def _image_row_keys(digest: int, salt: int, count: int) -> tuple[int, ...]:
    """The surrogate of every row of one image: what ``vision_bank_key_ids``
    writes over that image's pads."""

    memo_key = (int(digest), int(salt), int(count))
    with _ROW_KEYS_LOCK:
        keys = _ROW_KEYS.get(memo_key)
        if keys is not None:
            _ROW_KEYS.move_to_end(memo_key)
            return keys
    keys = tuple(
        _BANK_KEY_FLAG | ((memo_key[0] ^ (row * _BANK_KEY_MIX) ^ memo_key[1]) & _BANK_KEY_MASK)
        for row in range(memo_key[2])
    )
    with _ROW_KEYS_LOCK:
        _ROW_KEYS[memo_key] = keys
        held = sum(key[2] for key in _ROW_KEYS)
        while held > _ROW_KEYS_MAX_ROWS and len(_ROW_KEYS) > 1:
            dropped, _ = _ROW_KEYS.popitem(last=False)
            held -= dropped[2]
    return keys


def _as_int64(token_ids: Sequence[int]) -> np.ndarray:
    """Token ids as one int64 array: every model id and every surrogate
    (below 2**63) fits, so a comparison on it is the comparison on the ids."""

    return np.asarray(token_ids, dtype=np.int64)


def vision_image_spans(
    prompt_ids: list[int], splice: VisionSplice
) -> list[tuple[int, int]] | None:
    """[start, end) prompt positions of each image's expanded pad run.

    Computed on the RAW prompt ids (pads not yet surrogate-remapped) or on
    keyed ids (surrogates carry the flag bit, never equal to the pad id) —
    callers pass whichever sequence they hold alongside the pad layout. A
    restore that lands strictly inside one of these spans would resurrect
    KV whose embeddings came from other pixels even when token ids match
    (the 2026-08-07 pillar alias-leg regression): image content rides
    out-of-band of the ids, so id-equality inside a span is not
    input-equality unless the WHOLE span matched.
    """

    pad_counts = splice.pad_counts
    if not pad_counts:
        return None
    pad_id = splice.image_pad_token_id
    ids = _as_int64(prompt_ids)
    positions = np.flatnonzero((ids == int(pad_id)) | ((ids & _BANK_KEY_FLAG) != 0))
    if len(positions) != sum(int(c) for c in pad_counts):
        return None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for count in pad_counts:
        count = int(count)
        if count <= 0:
            continue
        spans.append((int(positions[cursor]), int(positions[cursor + count - 1]) + 1))
        cursor += count
    return spans


def clamp_matched_outside_image_spans(
    matched: int, spans: list[tuple[int, int]] | None
) -> int:
    """Snap a prefix-match that ends inside an image span back to its start."""

    if not spans:
        return int(matched)
    m = int(matched)
    for start, end in spans:
        if start < m < end:
            return int(start)
    return m


# -- reading a keyed sequence -------------------------------------------------
#
# The session bank, the session's committed stream and the admission all hold
# an image conversation in the view vision_bank_key_ids builds. These read that
# view back, so the key format stays known to this module alone.


def is_image_key(token: int) -> bool:
    """Whether ``token`` stands for an image row in a keyed sequence.

    Every image row of a keyed sequence carries the flag bit; no model id does.
    """

    return bool(int(token) & _BANK_KEY_FLAG)


def image_key_runs(keyed_ids: Sequence[int]) -> list[tuple[int, int]]:
    """[start, end) of each run of consecutive image rows, in order.

    Each image expands to one contiguous run of rows, and the chat templates
    put markup between images, so a run is one image. Two images that touch
    read as one run, which only makes every rule built on runs stricter.
    """

    if len(keyed_ids) == 0:
        return []
    rows = ((_as_int64(keyed_ids) & _BANK_KEY_FLAG) != 0).astype(np.int8)
    edges = np.flatnonzero(np.diff(rows, prepend=0, append=0))
    return [(int(start), int(end)) for start, end in zip(edges[0::2], edges[1::2])]


def unkeyed_ids(keyed_ids: Sequence[int], image_pad_token_id: int) -> list[int]:
    """The model ids behind a keyed sequence: every image row back to the pad.

    What the model is served (the rows themselves ride the splice) and what a
    tokenizer can decode.
    """

    if len(keyed_ids) == 0:
        return []
    ids = _as_int64(keyed_ids)
    return np.where((ids & _BANK_KEY_FLAG) != 0, int(image_pad_token_id), ids).tolist()


def inside_image(keyed_ids: Sequence[int], position: int) -> bool:
    """Whether a cut at ``position`` would split an image: image rows on both sides."""

    p = int(position)
    return (
        0 < p < len(keyed_ids)
        and is_image_key(keyed_ids[p - 1])
        and is_image_key(keyed_ids[p])
    )


def image_safe_restore_len(
    keyed_ids: Sequence[int], matched: int, *, reforwards_last_token: bool
) -> int:
    """The longest restore within ``matched`` tokens that keeps every image whole.

    A matched keyed prefix is content-true: the same text and, row for row,
    the same pixels at the same positions. A restore point inside an image is
    still refused: no restore has ever started there (they stopped before the
    first image or at an entry's end), so that case is unproven. A restore
    that re-forwards its last token (an entry without recurrent state) must
    not end on an image row either, since that forward embeds the pad id
    instead of the image row. Such a restore steps back to the first row of
    the image, where the prefix is text. Text sequences carry no image rows
    and come back unchanged.
    """

    m = max(0, min(int(matched), len(keyed_ids)))
    if inside_image(keyed_ids, m) or (
        reforwards_last_token and m > 0 and is_image_key(keyed_ids[m - 1])
    ):
        while m > 0 and is_image_key(keyed_ids[m - 1]):
            m -= 1
    return m


def _splice_rows_into_embedded(
    embedded: Any,
    mask: Any,
    rows: Any,
) -> Any:
    flat_mask = mask.reshape(-1)
    positions = mx.array(
        [i for i, hit in enumerate(flat_mask.tolist()) if hit], dtype=mx.int32
    )
    batch, seq, hidden = embedded.shape
    flat = embedded.reshape(batch * seq, hidden)
    flat[positions] = rows.astype(embedded.dtype)
    return flat.reshape(batch, seq, hidden)


def spliced_chunk_embeddings(
    embed_tokens: Any,
    chunk_array: Any,
    splice: VisionSplice,
) -> Any | None:
    """Embed one prefill chunk, replacing pad rows with vision rows.

    Returns None when the chunk holds no image pad tokens, so callers can
    keep the plain token-id fast path. Advances the splice cursor by the
    number of pads consumed; raises if the prompt contains more pads than
    the request supplied vision rows for, which would silently misalign
    every later image.
    """

    ids = chunk_array
    mask = ids == splice.image_pad_token_id
    pad_count = int(mask.sum().item())
    if pad_count == 0:
        return None
    if splice.remaining() < pad_count:
        raise ValueError(
            "vision splice underflow: prompt has more image pad tokens "
            f"({splice.cursor + pad_count}) than vision rows ({splice.total_rows})"
        )
    embedded = embed_tokens(ids)
    rows = splice.embeddings[splice.cursor : splice.cursor + pad_count]
    splice.cursor += pad_count
    return _splice_rows_into_embedded(embedded, mask, rows)


def spliced_embeddings_for_window(
    embed_tokens: Any,
    window_array: Any,
    splice: VisionSplice,
    *,
    rows_before: int,
) -> Any | None:
    """Cursor-free splice for an arbitrary prompt window.

    The MTP committed-history stream pairs hidden state t with token t+1,
    so its embedding window is shifted one token right of the trunk prefill
    chunk that produced the hidden states. This variant reads vision rows
    at an explicit offset (``rows_before`` = pad tokens before the window
    start) without touching the sequential cursor the trunk consumes.

    Returns None when the window holds no image pad tokens.
    """

    mask = window_array == splice.image_pad_token_id
    pad_count = int(mask.sum().item())
    if pad_count == 0:
        return None
    if rows_before + pad_count > splice.total_rows:
        raise ValueError(
            "vision splice window overflow: window needs rows "
            f"[{rows_before}, {rows_before + pad_count}) but only "
            f"{splice.total_rows} vision rows exist"
        )
    embedded = embed_tokens(window_array)
    rows = splice.embeddings[rows_before : rows_before + pad_count]
    return _splice_rows_into_embedded(embedded, mask, rows)
