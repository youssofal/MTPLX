"""The checkpoint at the start of the current user turn survives its tool loop.

With thinking on, the Qwen 3.5 and 3.6 templates drop the thinking of every
assistant message before the last user message, so when the next user message
arrives the history changes at the first token of the previous turn's answer.
In the founder's Pi task on the 35B (rows barsx-1117-0500/01-cand) that
turn's own checkpoint (22,164) was evicted by the checkpoints of a 11.5K-token
tool result, and the next turn resumed at 18,176: 4,051 tokens read again.
``user_turn_start`` finds that position from the tokens and every retention
(prefill, inherited set, bank put) keeps the newest checkpoint at or before it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx import checkpoint_anchors as anchors
from mtplx.checkpoint_anchors import AnchorPlan, retain_checkpoints, user_turn_start
from mtplx.generation import _inherited_gdn_boundaries

START, END, NL, TOOL, TOOL_END, THINK = 900, 901, 10, 902, 903, 904
USER, ASSISTANT, SYSTEM = 20, 21, 22
MARKERS = (START, (USER, NL), (ASSISTANT, NL), TOOL)


@pytest.fixture(autouse=True)
def _markers():
    before = anchors.turn_markers()
    anchors.configure_turn_markers(MARKERS)
    yield
    anchors.configure_turn_markers(before)


def _message(role: int, body: list[int]) -> list[int]:
    return [START, role, NL, *body, END, NL]


def _conversation():
    """system, user, assistant tool call, tool result, assistant answer."""

    tokens = _message(SYSTEM, [1] * 30)
    tokens += _message(USER, [2] * 10)
    answer_start = len(tokens) + 3  # after <|im_start|>assistant\n
    tokens += _message(ASSISTANT, [THINK, 3, 3, 3])
    tokens += [START, USER, NL, TOOL, NL, 4, 4, 4, TOOL_END, END, NL]
    tokens += _message(ASSISTANT, [THINK, 5, 5])
    return tokens, answer_start


def test_the_turn_starts_at_the_answer_to_the_last_real_user_message() -> None:
    tokens, answer_start = _conversation()
    assert tokens[answer_start] == THINK
    assert user_turn_start(tokens) == answer_start
    # The tool loop of the same turn does not move it.
    more = tokens + [START, USER, NL, TOOL, NL, 6, 6, TOOL_END, END, NL]
    assert user_turn_start(more) == answer_start
    # A new user message with its generation prompt starts a new turn.
    nxt = more + _message(USER, [7, 7]) + [START, ASSISTANT, NL, THINK]
    assert user_turn_start(nxt) == len(nxt) - 1
    # Read only up to a limit (a restore point inside the second turn).
    assert user_turn_start(nxt, limit=len(more)) == answer_start


def test_no_markers_no_user_turn_or_no_answer_yet_gives_none() -> None:
    tokens, _ = _conversation()
    anchors.configure_turn_markers(None)
    assert user_turn_start(tokens) is None
    anchors.configure_turn_markers(MARKERS)
    assert user_turn_start(_message(SYSTEM, [1, 1])) is None
    assert user_turn_start(_message(SYSTEM, [1]) + _message(USER, [2])) is None
    assert user_turn_start([]) is None


def _founder_records():
    # The anchors of the founder's turn-2 tool loop (request 15 and 16 rows).
    return [(p, None, None) for p in (2048, 8192, 9971, 18176, 20224, 22164, 22228, 26491, 32635, 33847, 33911)]


def test_the_turn_start_checkpoint_outlives_the_tool_result_checkpoints() -> None:
    plan = AnchorPlan(record_count=8, prompt_end=33911, restore_point=22395, system_end=9971)
    kept = [record[0] for record in retain_checkpoints(_founder_records(), plan)]
    assert 22164 not in kept  # what happened: the next turn resumed at 18,176
    plan = AnchorPlan(
        record_count=8, prompt_end=33911, restore_point=22395, system_end=9971, turn_start=22227
    )
    kept = [record[0] for record in retain_checkpoints(_founder_records(), plan)]
    assert 22164 in kept
    assert len(kept) == 8
    assert 33847 in kept and 33911 in kept  # the prompt-end anchor and the newest stay


def test_the_inherited_set_keeps_the_turn_start_checkpoint() -> None:
    tokens, answer_start = _conversation()
    records = [(p, None, None) for p in range(4, len(tokens) + 1, 4)]
    entry = SimpleNamespace(gdn_boundaries=records, token_ids=tokens)
    kept = [r[0] for r in _inherited_gdn_boundaries(entry, len(tokens), budget_bytes=3)]
    newest_at_turn = max(p for p in range(4, len(tokens) + 1, 4) if p <= answer_start)
    assert newest_at_turn in kept
    assert len(kept) == 3
