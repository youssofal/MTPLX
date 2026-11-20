"""Which recurrent checkpoints a conversation keeps, and why.

A checkpoint (a "boundary record" ``(position, recurrent snapshot, hidden)``)
is the recurrent state (GDN matrix state, conv and PLE tails) after the first
``position`` tokens of a prompt, captured during prefill. A hybrid model can
only resume at a checkpoint at or below the matched prefix: the KV is trimmed
to it and the rest of the prompt is prefilled again. Each one costs about
115.6 MB on Flash-Next, so a conversation keeps a few.

Prefill captures one at every chunk edge (and at the tail ladder's rungs).
Until 2026-09-30 the list was thinned back to eight after every capture,
keeping one record per power-of-two distance from the NEWEST record. Thinning
again each time the newest record moved eroded the middle: on 2026-09-29 a
16,371-token match restored at 4,096 (B-cache 2.6 reproduced it with the real
function). The kept set is now chosen from anchors that do not move as the
frontier advances, in this order:

* the prompt-end anchor: the newest checkpoint at least ``tail_backoff``
  tokens before the prompt end, where the next agent turn diverges (the
  reply after it is re-rendered);
* for an image prompt, the pre-image anchor: the newest checkpoint at or
  before its first image, where a turn that changes that image resumes;
* the turn-start anchor: the newest checkpoint at or before the first token
  of the answer to the last real user message (``user_turn_start``), where
  the next user turn's history changes when the chat template drops this
  turn's thinking (Qwen 3.5 and 3.6), kept through the turn's whole tool loop;
* the restore point this prefill resumed from (the previous prompt's end);
* the stable prompt-prefix edge (the turn the tool-continuation hint rides);
* the end of the prompt's system segment (``system_segment_end``), where a
  new session of the same client resumes;
* the newest checkpoint;
* a fixed absolute grid: the first checkpoint in every ``grid_tokens`` cell,
  thinned by halving its resolution when the budget cannot hold every cell
  (every 16,384 tokens, then 32,768, ...), so a kept grid anchor keeps its
  position and a restore point once covered stays covered at the resolution
  the budget affords;
* then the newest remaining checkpoints while the budget lasts.

A checkpoint's grid rank depends only on its cell and the cell of the
checkpoint before it, and dropping a checkpoint that shares its cell with an
earlier one never changes a later one's rank. So within one prefill,
trimming after every capture keeps exactly what trimming once at the end
would, and the order captures arrive in cannot erode the set (tested on
random prefills). Across turns a dropped checkpoint stays dropped: a session
that grew long and then rewound keeps the grid its longest prompt could
afford, never less.

Retention is bounded by bytes, passed in by the caller (``budget_bytes``, the
session bank's ``checkpoint_budget_bytes``). Without one it is what today's
memory plan prices: ``MTPLX_GDN_BOUNDARY_MAX`` (8) checkpoints of the model at
hand. A restore never starts past the proven matched prefix: selection
(``SessionBankEntry.recurrent_boundary_at_or_below``) only ever returns a
checkpoint at or below it, and ``image_spans`` keeps every checkpoint of an
image prompt out of its images (a restore never resumes with part of an
image's rows, ``vision.splice.inside_image``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

#: Cell width of the absolute grid, in tokens.
GRID_TOKENS = 8192
#: A system segment shorter than this gets no checkpoint of its own.
SYSTEM_SEGMENT_MIN_TOKENS = 2048
#: Checkpoints kept when the caller gives no byte budget (the count the
#: prefill memory plan has always priced: eight boundary snapshots).
DEFAULT_CHECKPOINT_COUNT = 8
#: Where the prompt-end anchor sits, in tokens before the prompt end
#: (MTPLX_GDN_BOUNDARY_TAIL_BACKOFF's default, where the tail ladder puts it).
DEFAULT_TAIL_BACKOFF = 64

_FIRST_LEVEL = 1 << 30  # the first checkpoint represents every grid cell


def default_checkpoint_count() -> int:
    """``MTPLX_GDN_BOUNDARY_MAX``: the no-budget retention, in checkpoints."""

    raw = os.environ.get("MTPLX_GDN_BOUNDARY_MAX", str(DEFAULT_CHECKPOINT_COUNT))
    try:
        return max(2, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_CHECKPOINT_COUNT


def default_tail_backoff() -> int:
    """``MTPLX_GDN_BOUNDARY_TAIL_BACKOFF``: how far before the prompt end the
    prefill's tail ladder puts its top rung, the prompt-end anchor. The one
    parse of it, so the prefill and the session bank agree on the anchor."""

    raw = os.environ.get("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", str(DEFAULT_TAIL_BACKOFF))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_TAIL_BACKOFF


def system_prompt_checkpoint_enabled() -> bool:
    """``MTPLX_SYSTEM_PROMPT_CHECKPOINT=0`` turns off the checkpoint at the
    end of a system prompt and the SSD lane's exact point that serves it."""

    raw = str(os.environ.get("MTPLX_SYSTEM_PROMPT_CHECKPOINT", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def system_segment_end(tokens: Sequence[int]) -> int | None:
    """Where a chat prompt's first turn ends, in tokens: the next occurrence
    of the token the prompt opens with (ChatML's ``<|im_start|>``), when that
    turn is at least ``SYSTEM_SEGMENT_MIN_TOKENS`` long.

    For an agent client the first turn is its system prompt and tools, which
    every new session of the client in the same project, a subagent and a
    compacted conversation send again token for token before a different
    first message (the founder's bank, 12 Nov: a Pi pair and an OpenCode pair
    matched exactly to this point plus the next turn's 3-token header; a Pi
    pair from two project folders differed 373 tokens before it, at a path).
    Prefill records a checkpoint here for such prompts and every retention
    keeps it. It is read from the tokens alone, so the session bank needs no
    tokenizer; a template that opens with a BOS token has none.
    ``MTPLX_SYSTEM_PROMPT_CHECKPOINT=0`` turns the checkpoint off.
    """

    if not system_prompt_checkpoint_enabled() or len(tokens) < 2:
        return None
    if not hasattr(tokens, "index"):
        tokens = tuple(tokens)
    try:
        # A C-level scan that stops at the segment's end, a few thousand
        # tokens in for an agent prompt.
        end = int(tokens.index(tokens[0], 1))
    except ValueError:
        return None
    return end if end >= SYSTEM_SEGMENT_MIN_TOKENS else None


#: ``(<|im_start|> id, ids of "user\n", ids of "assistant\n", <tool_response> id)``
#: of the served model's chat template, registered once by the runtime that
#: owns the tokenizer (``configure_turn_markers``); None leaves the turn-start
#: anchor off (the session bank itself has no tokenizer).
_TURN_MARKERS: tuple[int, tuple[int, ...], tuple[int, ...], int] | None = None


def configure_turn_markers(
    markers: tuple[int, tuple[int, ...], tuple[int, ...], int] | None,
) -> None:
    global _TURN_MARKERS
    _TURN_MARKERS = markers


def turn_markers() -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
    return _TURN_MARKERS


def user_turn_start(tokens: Sequence[int] | None, *, limit: int | None = None) -> int | None:
    """The position of the first token of the answer to the last real user
    message in ``tokens[:limit]`` (a ChatML user turn that is not a tool
    result), or None.

    With thinking on, the Qwen 3.5 and 3.6 templates drop the thinking of
    every assistant message before the last user message, so when the next
    user message arrives the history changes right here: everything the turn
    wrote after it (its tool calls, the tool results, its answer) is read
    again. A checkpoint at or before this position is what that read resumes
    from (17 Nov 2026, founder's Pi task on the 35B: the turn's checkpoint at
    22,164 was evicted by tool-result checkpoints and the next turn resumed at
    18,176, 4,051 tokens earlier).
    """

    markers = _TURN_MARKERS
    if markers is None or not tokens:
        return None
    start_id, user_ids, assistant_ids, tool_id = markers
    import numpy as np

    end = len(tokens) if limit is None else max(0, min(len(tokens), int(limit)))
    if end <= 0:
        return None
    try:
        ids = np.asarray(tokens[:end], dtype=np.int64)
    except (TypeError, ValueError):
        ids = np.fromiter((int(token) for token in tokens[:end]), dtype=np.int64, count=end)
    if ids.ndim != 1:
        return None
    starts = np.flatnonzero(ids == start_id)
    user = np.asarray(user_ids, dtype=np.int64)
    assistant = np.asarray(assistant_ids, dtype=np.int64)
    for index in range(len(starts) - 1, -1, -1):
        start = int(starts[index])
        role_end = start + 1 + len(user)
        if role_end >= end or not np.array_equal(ids[start + 1 : role_end], user):
            continue
        if int(ids[role_end]) == tool_id:
            continue
        if index + 1 >= len(starts):
            return None
        answer = int(starts[index + 1])
        answer_end = answer + 1 + len(assistant)
        if answer_end > end or not np.array_equal(ids[answer + 1 : answer_end], assistant):
            return None
        return answer_end
    return None


@dataclass(frozen=True)
class AnchorPlan:
    """What one prefill knows about its checkpoints before capturing any.

    ``prompt_end``: the last position this prefill can checkpoint (the prompt
    length minus one; None when unknown, and the newest checkpoint stands in).
    ``restore_point``: where the prefill resumed from a stored entry.
    ``stable_prefix``: the stable prompt-prefix edge, when the prompt has one.
    ``ceiling``: no checkpoint past this position.
    ``image_spans``: ``[start, end)`` of each image's rows in an image prompt;
    no checkpoint strictly inside one. Image prompts kept none past their
    first image until the restores could resume past a whole image (the
    content-keyed near-prefix lane, 2026-09-30); a checkpoint between or
    after images is a state a cold prefill of the same keyed prefix reaches
    exactly.
    ``system_end``: the end of the prompt's system segment
    (``system_segment_end``), protected: the checkpoint there shares the
    grid's first cell with the chunk-grid checkpoint below it, which the
    grid keeps first, so it was the first one dropped.
    ``turn_start``: ``user_turn_start`` of the prompt, protected: the newest
    checkpoint at or before it is where the next user turn resumes.
    """

    budget_bytes: int | None = None
    record_count: int | None = None
    prompt_end: int | None = None
    tail_backoff: int = DEFAULT_TAIL_BACKOFF
    restore_point: int | None = None
    stable_prefix: int | None = None
    ceiling: int | None = None
    image_spans: tuple[tuple[int, int], ...] = ()
    grid_tokens: int = GRID_TOKENS
    system_end: int | None = None
    turn_start: int | None = None

    def admits(self, position: int) -> bool:
        position = int(position)
        if self.ceiling is not None and position > int(self.ceiling):
            return False
        return not any(
            int(start) < position < int(end) for start, end in self.image_spans
        )

    @property
    def first_image_start(self) -> int | None:
        return int(self.image_spans[0][0]) if self.image_spans else None


def _tree_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and not isinstance(value, (bytes, bytearray)):
        return int(nbytes)
    states = getattr(value, "states", None)
    if states is not None and hasattr(value, "meta_states"):
        return _tree_nbytes(states) + _tree_nbytes(value.meta_states)
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_nbytes(item) for item in value)
    return 0


def record_nbytes(record: Any) -> int:
    """Bytes one checkpoint holds (its snapshot and hidden row); at least 1,
    so records without arrays still count as one each."""

    snapshot = record[1] if len(record) > 1 else None
    hidden = record[2] if len(record) > 2 else None
    return max(1, _tree_nbytes(snapshot) + _tree_nbytes(hidden))


def budget_bytes_for(records: Iterable[Any], plan: AnchorPlan) -> int:
    """The plan's byte budget, or ``record_count`` checkpoints of the
    largest record present when the caller gave none."""

    if plan.budget_bytes is not None:
        return max(0, int(plan.budget_bytes))
    count = plan.record_count if plan.record_count is not None else default_checkpoint_count()
    largest = max((record_nbytes(record) for record in records), default=1)
    return max(0, int(count)) * largest


def _grid_levels(positions: list[int], grid_tokens: int) -> list[int]:
    """Per ascending position: the coarsest grid level whose cell it is the
    first checkpoint of, or -1 when an earlier checkpoint shares its finest
    cell. Cells of level L are ``grid_tokens * 2**L`` wide and nest, so the
    level is the highest bit where its cell index differs from the previous
    checkpoint's."""

    grid = max(1, int(grid_tokens))
    levels: list[int] = []
    previous: int | None = None
    for position in positions:
        cell = int(position) // grid
        if previous is None:
            levels.append(_FIRST_LEVEL)
        elif cell == previous:
            levels.append(-1)
        else:
            levels.append((cell ^ previous).bit_length() - 1)
        previous = cell
    return levels


def retain_checkpoints(records: Iterable[Any], plan: AnchorPlan) -> list:
    """The checkpoints to keep, ascending by position, within the budget."""

    by_position: dict[int, Any] = {}
    for record in records:
        position = int(record[0])
        if position <= 0 or not plan.admits(position):
            continue
        # A later capture of the same position replaces the earlier one.
        by_position[position] = record
    if not by_position:
        return []
    positions = sorted(by_position)
    newest = positions[-1]
    end = int(plan.prompt_end) if plan.prompt_end is not None else newest
    limit = end - max(0, int(plan.tail_backoff))
    if plan.ceiling is not None:
        limit = min(limit, int(plan.ceiling))
    below_limit = [position for position in positions if position <= limit]
    first_image = plan.first_image_start
    pre_image = (
        [position for position in positions if position <= first_image]
        if first_image is not None
        else []
    )
    turn = (
        [position for position in positions if position <= int(plan.turn_start)]
        if plan.turn_start is not None
        else []
    )
    protected: list[int] = []
    for position in (
        below_limit[-1] if below_limit else None,
        pre_image[-1] if pre_image else None,
        turn[-1] if turn else None,
        plan.restore_point,
        plan.stable_prefix,
        plan.system_end,
        newest,
    ):
        if position is None:
            continue
        position = int(position)
        if position in by_position and position not in protected:
            protected.append(position)
    levels = dict(zip(positions, _grid_levels(positions, plan.grid_tokens)))
    rest = sorted(
        (position for position in positions if position not in protected),
        key=lambda position: (levels[position] < 0, -levels[position], -position),
    )
    budget = budget_bytes_for(by_position.values(), plan)
    kept: list[int] = []
    spent = 0
    for position in (*protected, *rest):
        size = record_nbytes(by_position[position])
        if spent + size > budget:
            continue
        kept.append(position)
        spent += size
    return [by_position[position] for position in sorted(kept)]


class CheckpointSink(list):
    """A prefill's checkpoint list that knows its plan and keeps its budget.

    It is a list, so the prefill loops, ``PromptState`` and the session bank
    take it as one. ``cuts_forwards=False`` marks a sink that may only record
    where the forwards already end: the prefill must not add a chunk edge, a
    tail ladder or an in-forward capture for it. Image prompts get one, so
    their prefill stays exactly what it was before they kept checkpoints.
    """

    def __init__(
        self,
        records: Iterable[Any] = (),
        *,
        plan: AnchorPlan,
        cuts_forwards: bool = True,
    ) -> None:
        super().__init__(records)
        self.plan = plan
        self.cuts_forwards = bool(cuts_forwards)

    def admits(self, position: int) -> bool:
        return self.plan.admits(position)

    def retain(self) -> None:
        self[:] = retain_checkpoints(self, self.plan)


def checkpoint_coverage(
    records: Iterable[Any],
    *,
    budget_bytes: int | None = None,
    first_image_start: int | None = None,
) -> dict[str, Any]:
    """What a prompt's checkpoints cover, for the request receipt.

    With an image prompt, ``pre_image`` says how close to the first image the
    newest checkpoint before it sits: the most a later turn that changes
    something after the image can restore without re-reading the text.
    """

    records = list(records or ())
    positions = sorted(int(record[0]) for record in records)
    total = sum(record_nbytes(record) for record in records)
    report: dict[str, Any] = {
        "anchors": positions,
        "count": len(positions),
        "bytes": int(total),
        "budget_bytes": (
            int(budget_bytes)
            if budget_bytes is not None
            else budget_bytes_for(records, AnchorPlan())
        ),
    }
    if first_image_start is not None:
        start = int(first_image_start)
        before = [position for position in positions if position <= start]
        anchor = before[-1] if before else None
        report["pre_image"] = {
            "first_image_start": start,
            "anchor": anchor,
            "tokens_to_reread": start - (anchor or 0),
        }
    return report
