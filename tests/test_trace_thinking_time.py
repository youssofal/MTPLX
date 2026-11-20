"""``mtplx trace`` shows when the reader saw each turn think and answer.

Beside TTFT per turn: the turn's reasoning tokens, ``reasoning_s`` and
``first_visible_s`` from its request-log row; in the summary: thinking before
the answer (request arrival to the last reasoning delta, 0 for a turn without
reasoning), median, turns over 10 s and the longest.
"""

from __future__ import annotations

from mtplx.commands.trace import (
    _fmt_thinking,
    _think_tokens,
    _thinking_s,
    _thinking_summary,
    _turn_row,
)
from mtplx.commands.trace_report import _sec_digest


def _turn(receipt: dict, *, client: str = "pi") -> dict:
    return {
        "turn": 3,
        "kind": "assistant",
        "message": {"time": {"created": 1_000, "completed": 41_000}, "_client": client},
        "receipt": receipt,
    }


def test_each_turn_carries_the_three_beside_its_ttft():
    row = _turn_row(
        _turn(
            {"ttft_s": 0.74, "reasoning_tokens": 1852, "reasoning_s": 34.1,
             "first_visible_s": 34.6, "completion_tokens": 2100}
        )
    )
    assert (row["ttft_s"], row["reasoning_tokens"], row["reasoning_s"], row["first_visible_s"]) == (
        0.74, 1852, 34.1, 34.6,
    )
    # Pi reports no reasoning count of its own: the server's stands in.
    assert row["client_reasoning_tokens"] is None and _think_tokens(row) == 1852
    assert _thinking_s(row) == 34.1
    # A row from an older server: nothing measured, nothing made up.
    old = _turn_row(_turn({"ttft_s": 0.5}))
    assert old["first_visible_s"] is None and _thinking_s(old) is None


def test_the_summary_gives_the_median_and_the_turns_over_ten_seconds():
    rows = [
        {"reasoning_tokens": 0, "reasoning_s": None, "first_visible_s": 0.8},  # no thinking
        {"reasoning_tokens": 300, "reasoning_s": 6.1, "first_visible_s": 6.2},
        {"reasoning_tokens": 700, "reasoning_s": 12.8, "first_visible_s": 12.9},
        # Cut by the length limit inside the reasoning: never visible.
        {"reasoning_tokens": 2000, "reasoning_s": 35.5, "first_visible_s": None},
        {"reasoning_tokens": None, "reasoning_s": None, "first_visible_s": None},  # old row
    ]
    summary = _thinking_summary(rows)
    assert summary == {"turns": 4, "median_s": 9.45, "over_10s": 2, "longest_s": 35.5}
    assert _fmt_thinking(summary) == "median 9.4s, 2 of 4 turns over 10s (longest 35.5s)"
    assert _thinking_summary([{"reasoning_tokens": None}]) is None
    assert _fmt_thinking(None) == "-"


def test_the_report_digest_shows_them_per_turn():
    row = _turn_row(
        _turn({"ttft_s": 0.74, "reasoning_tokens": 1852, "reasoning_s": 34.1,
               "first_visible_s": 34.6, "completion_tokens": 2100, "decode_tok_s": 56.0})
    )
    html = _sec_digest([{"row": row, "prompt": None, "reasoning_chars": None, "output_chars": None}])
    assert "<th class=\"n\">think s</th><th class=\"n\">first visible s</th>" in html
    assert "<td class=\"n\">0.74</td><td class=\"n\">34.10</td><td class=\"n\">34.60</td>" in html
