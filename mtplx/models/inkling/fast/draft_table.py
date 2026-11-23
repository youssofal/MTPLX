"""The drafter's vocabulary table: drafts scored on a table of the most frequent tokens instead of the
whole vocabulary (class B, behind ``MTPLX_INKLING_DRAFT_TABLE``, off by default: the founder decides).

Every round the drafter scores its proposals against the target's whole vocabulary head (200,058
rows: 260 MB of 4-bit codes read for the proposal rows, about 1.3 ms of a 3-guess round on the M5
Max) and adds its folded Markov bias (a 103 MB table, once a proposal). On code and English prose a
table of the 32,768 most frequent tokens holds 99.3 to 100% of the tokens a text uses (d1/table:
coverage.json), so the drafter reads a sixth of those bytes when it scores only the table's rows.

What changes and what does not: a draft is now proposed from the table (the drafter's law is its
softmax over the table's scores; tokens outside it keep a score far below every table score), so the
drafts and the per-seed tokens change: class B. The target's output distribution does not: the
round's decision accepts a draft with the law the drafter actually used and resamples a rejection
from the target's residual, so a token the table cannot propose is still emitted with its target
probability, after a rejected draft. The scores of the table's tokens are bit for bit the ones the
whole head gives them (the same rows through the same row-stable launch).

``MTPLX_INKLING_DRAFT_TABLE=<ranking.json>[:size]``: a JSON file holding ``"ranking"`` (token ids,
most frequent first); the table is its first ``size`` ids (default 32,768). Memory: the table's rows
of the head and of the folded bias are gathered once per drafter (about 58 MB on Bare Speed).
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple, Optional

import mlx.core as mx

#: Score of a token outside the table: below every table score, finite (the soft cap and the law's
#: arithmetic keep it finite and its weight vanishes next to the table's).
_OUTSIDE = -1.0e30


class Table(NamedTuple):
    ids: mx.array          # (N,) int32, ascending token ids
    head: object           # the target head's rows at ids (AffineWeight or dense)
    folded: mx.array       # the drafter's folded bias rows at ids, (N, rank + 1)
    outside: mx.array      # (1, vocab) float32 filled with _OUTSIDE


def _setting() -> Optional[tuple[str, int]]:
    raw = os.environ.get("MTPLX_INKLING_DRAFT_TABLE", "").strip()
    if not raw:
        return None
    path, _, size = raw.partition(":")
    return path, int(size or 32768)


def bind(drafter, target) -> Optional[Table]:
    """The table for this drafter and target, gathered once, or ``None`` when the switch is off."""
    setting = _setting()
    if setting is None:
        return None
    cached = getattr(drafter, "_draft_table", None)
    if cached is not None and cached[0] == setting:
        return cached[1]
    from ..precision import AffineWeight

    path, size = setting
    with open(path) as handle:
        ranking = json.load(handle)["ranking"]
    keep = [int(t) for t in ranking if 0 <= int(t) < drafter.vocab][:size]
    ids = mx.array(sorted(set(keep)), dtype=mx.int32)
    head = drafter.draft_head(target) if hasattr(drafter, "draft_head") else target.unembed
    if isinstance(head, AffineWeight):
        rows = AffineWeight(mx.take(head.codes, ids, axis=0), mx.take(head.scales, ids, axis=0),
                            mx.take(head.biases, ids, axis=0), head.bits, head.group_size)
        mx.eval(rows.codes, rows.scales, rows.biases)
    else:
        rows = mx.take(head, ids, axis=0)
        mx.eval(rows)
    folded = mx.take(drafter.fold(target.unembed), ids, axis=0)
    outside = mx.full((1, drafter.vocab), _OUTSIDE, dtype=mx.float32)
    mx.eval(folded, outside)
    table = Table(ids, rows, folded, outside)
    drafter._draft_table = (setting, table)
    return table


def expand(table: Table, scores: mx.array) -> mx.array:
    """``scores`` ``(1, N)`` of the table's tokens as a ``(1, vocab)`` row (outside tokens far below)."""
    return mx.put_along_axis(table.outside, table.ids[None], scores.astype(mx.float32), axis=1)
