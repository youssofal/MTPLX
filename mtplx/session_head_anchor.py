"""Where the fixed head of a chat prompt ends: the session-head anchor.

Agent harnesses start every new chat with the same head: the tool schemas and
the system text, rendered into the system turn. The first user message comes
next, and that is where two new sessions of the same agent diverge. A hybrid
model restores a shared prefix only from a recurrent (GDN) boundary at or
below it, so a boundary exactly at the head's end lets every new session
reuse the whole head.

The position is read from the prompt tokens themselves: the first chat turn
that is not a system turn opens with an atomic ``<|im_start|>`` token, so its
index is the head length in tokens, token-exact for every template, harness
and endpoint that renders ChatML, and recomputable for any banked entry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

TURN_OPEN = "<|im_start|>"
SYSTEM_ROLE = "system"


@dataclass(frozen=True)
class TurnMarkers:
    """Token ids that open a turn and name the system role after it."""

    turn_open: int
    system_role: tuple[int, ...]


_MARKERS: dict[int, tuple[Any, TurnMarkers | None]] = {}


def turn_markers(tokenizer: Any) -> TurnMarkers | None:
    """The tokenizer's ChatML turn markers, or None when it has none.

    ``<|im_start|>`` must be an added token the tokenizer never normalizes or
    strips around, so it always encodes as one id and nothing merges across
    it. Resolved once per tokenizer.
    """

    if tokenizer is None:
        return None
    cached = _MARKERS.get(id(tokenizer))
    if cached is not None and cached[0] is tokenizer:
        return cached[1]
    markers = _resolve_turn_markers(tokenizer)
    _MARKERS[id(tokenizer)] = (tokenizer, markers)
    return markers


def _resolve_turn_markers(tokenizer: Any) -> TurnMarkers | None:
    try:
        added = dict(tokenizer.added_tokens_decoder)
        turn_open = next(
            (
                int(token_id)
                for token_id, token in added.items()
                if getattr(token, "content", None) == TURN_OPEN
                and getattr(token, "normalized", True) is False
                and getattr(token, "lstrip", True) is False
                and getattr(token, "rstrip", True) is False
            ),
            None,
        )
        system_role = tuple(
            int(token) for token in tokenizer.encode(SYSTEM_ROLE, add_special_tokens=False)
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if turn_open is None or not system_role:
        return None
    return TurnMarkers(turn_open=turn_open, system_role=system_role)


def session_head_length(
    token_ids: Sequence[int], markers: TurnMarkers
) -> int | None:
    """Tokens before the first turn that is not a system turn.

    None when the prompt opens with a non-system turn (no head) or has no
    such turn at all (nothing after the head yet).
    """

    ids = token_ids if isinstance(token_ids, (list, tuple)) else list(token_ids)
    role_width = len(markers.system_role)
    position = -1
    while True:
        try:
            position = ids.index(markers.turn_open, position + 1)
        except ValueError:
            return None
        role = tuple(ids[position + 1 : position + 1 + role_width])
        if role != markers.system_role:
            return position if position > 0 else None
