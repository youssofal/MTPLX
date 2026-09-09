"""The QSA prefill "middle path": a wide chunk with narrow attention.

The 36 GDN layers, the MoE grouped GEMM and every projection want a WIDE
prefill chunk (better grouped-GEMM tile occupancy, fewer per-chunk syncs);
the 12 dense QSA layers want a NARROW one -- their score tensor is
``[H, rows, T]`` and their work term ``rows x T`` grows with the chunk.
Splitting only the attention into query tiles gives both: tile A never reads
tile B's keys, so a 4,096-row chunk tiled at 2,048 has exactly the peak AND
exactly the ``sum(rows x context)`` of an 8 x 2,048 cut, while everything
outside attention still sees 4,096 rows.

``MTPLX_QSA_PREFILL_QUERY_TILE`` (old name ``MTPLX_FABLE_PREFILL_QSA_QUERY_TILE``
still works as an alias) is the rows-per-tile knob; 0 / unset means the whole
chunk, i.e. today's behaviour, so flag-off is byte-identical.
"""

from __future__ import annotations

import os
from typing import Mapping

#: Rows per QSA attention query tile. The new key wins when set to a
#: non-empty value; the old name is honoured only when the new one is unset.
QUERY_TILE_ENV = "MTPLX_QSA_PREFILL_QUERY_TILE"
QUERY_TILE_ALIAS_ENV = "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE"


def _env(name: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return str(source.get(name) or "").strip()


def resolve_query_tile_rows(environ: Mapping[str, str] | None = None) -> int:
    """Rows per QSA attention query tile; 0 (default) = whole chunk."""

    raw = _env(QUERY_TILE_ENV, environ) or _env(QUERY_TILE_ALIAS_ENV, environ)
    if not raw:
        return 0
    try:
        rows = int(raw)
    except ValueError:
        return 0
    return rows if rows > 0 else 0


def query_tile_spans(
    rows: int, *, context_before: int, tile: int
) -> list[tuple[int, int, int]]:
    """``(row_start, row_end, keys_visible)`` for one chunk's query tiles.

    Attention rows are independent -- each row's softmax runs over its own
    causal/selected key set -- so grouping rows differently cannot change
    which keys a row sees.  ``keys_visible`` is the exclusive key bound for
    the tile's LAST row, and every earlier row in the tile is masked down to
    its own bound exactly as before.  Dropping the keys past that bound is
    mathematically a no-op: the dense path fills them with the score dtype's
    ``finfo.min``, whose ``exp`` underflows to a hard zero.

    The reduction *order* changes (shorter softmax rows, shorter P@V K), so
    the result is exact-visible-set, not bit-identical -- the same class as
    the portable gather tier.

    Empty list when tiling does not apply, so callers keep one code path.
    """

    span_rows = int(rows)
    tile_rows = int(tile)
    if span_rows <= 0 or tile_rows <= 0 or tile_rows >= span_rows:
        return []
    base = max(0, int(context_before))
    spans: list[tuple[int, int, int]] = []
    row = 0
    while row < span_rows:
        row_end = min(span_rows, row + tile_rows)
        spans.append((row, row_end, base + row_end))
        row = row_end
    return spans
