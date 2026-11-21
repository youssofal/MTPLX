"""The stream splitter holds back only reasoning that could still become a marker.

It used to hold the last 32 characters of reasoning whatever they were, so every thinking answer's
first visible text waited for its first 32 characters: seven to nine tokens, the fourth round of a
turn (T1, 21 Nov: 0.13 s of a warm Inkling turn's 0.15 s to the first visible text, on every
family's thinking stream). Every reasoning tag, chat-template sentinel and tool marker begins with
"<", so the reasoning window now waits only while it holds one (the content channel keeps its
hold). The bytes each channel carries are unchanged, proven against the old rule over random token
splits, except for one case the old rule already decided by token boundaries (below).
"""

from __future__ import annotations

import random
import re

import pytest

from mtplx.reasoning_codecs import QWEN_STYLE_REASONING_TAG_NAMES
from mtplx.server import openai as oa

PROSE = "The user wants a flappy bird clone in one HTML file. I will write the canvas loop first, then the pipes. "
CORPUS = {
    "prose_then_answer": PROSE * 3 + "</think>\n\nHere is the plan.\n\n1. Canvas\n2. Pipes\n",
    "code_with_angles": PROSE + "if (a < b && c > d) { x <<= 1; }\n<div class=\"bird\"></div>\n" * 3 + "</think>Done <b>now</b>.",
    "suffixed_close": PROSE * 2 + "</think:opensource>The answer is 4.",
    "spaced_close": PROSE * 2 + "</ think>The answer is 4.",
    "tool_call_from_thinking": PROSE + "\n<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.js\"}}\n</tool_call>",
    "function_call": PROSE + "</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\na.js\n</parameter>\n</function>\n</tool_call>",
    "quoted_markup": PROSE + "The protocol uses `<tool_call>` tags and </parameter> closers. " * 2 + "</think>Ok.",
    "sentinels": PROSE + "</think>Answer text<|im_end|>",
    "sentinel_turn": PROSE + "</think>Answer<|im_start|>assistant\nmore",
    "cjk_emoji": "模型正在思考这个问题并且不包含任何空格 🚀✅ " * 4 + "</think>回答 🎉",
    "whitespace_edges": "\n\n" + PROSE + "\n\n</think>\n\n  Answer with spaces.  \n\n",
    "no_close": PROSE * 4,
}
MODES = {
    "inside_thinking_no_tools": dict(thinking_enabled=True, start_inside_thinking=True,
                                     suppress_orphan_tool_markup=True, no_tools_declared=True),
    "inside_thinking_tools": dict(thinking_enabled=True, start_inside_thinking=True,
                                  tool_names=("write", "read", "bash")),
    "outside_thinking_tools": dict(thinking_enabled=True, start_inside_thinking=False,
                                   tool_names=("write", "read", "bash")),
    "thinking_off": dict(thinking_enabled=False, start_inside_thinking=False,
                         suppress_orphan_tool_markup=True, no_tools_declared=True),
}


def splitter(old: bool, mode: dict):
    sp = oa._ThinkingContentStreamSplitter(recover_unclosed_reasoning_as_content=False,
                                           trim_visible_content_edges=True, **mode)
    if old:
        sp._stream_holdback = lambda keep: keep  # the rule before 21 Nov
    return sp


def runs(sp, feeds):
    """Every (channel, text) the splitter emits, merged into runs of one channel, and the
    index of the feed that produced each channel's first character."""
    merged: list[list[str]] = []
    first: dict[str, int] = {}
    for index, feed in enumerate([*feeds, None]):
        out = sp.feed(feed) if feed is not None else sp.finish()
        for channel, text in out:
            if not text:
                continue
            first.setdefault(channel, index)
            if merged and merged[-1][0] == channel:
                merged[-1][1] += text
            else:
                merged.append([channel, text])
    return merged, first


def feeds_of(text: str, rng: random.Random) -> list[str]:
    out, i = [], 0
    while i < len(text):
        step = rng.randint(1, 6)
        out.append(text[i : i + step])
        i += step
    return out


REOPENED = PROSE + "<think>more thought " * 3 + "</think>Answer."
OPEN_TAG = re.compile(r"<\s*(?:" + "|".join(QWEN_STYLE_REASONING_TAG_NAMES) + r")\b[^>\n]*>", re.IGNORECASE)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_every_channel_carries_the_same_bytes_as_under_the_old_rule(mode):
    rng = random.Random(20261121)
    for name, text in CORPUS.items():
        for _ in range(60):
            feeds = feeds_of(text, rng)
            old, old_first = runs(splitter(True, MODES[mode]), feeds)
            new, new_first = runs(splitter(False, MODES[mode]), feeds)
            assert new == old, (mode, name, feeds)
            for channel, index in new_first.items():
                assert index <= old_first[channel], (mode, name, channel)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_an_opening_tag_written_inside_open_thinking_is_the_one_difference(mode):
    """The splitter drops an opening reasoning tag only when it heads the held text, so a
    literal `<think>` the model writes inside its own thinking was kept or dropped by token
    boundaries before, and still is (at other boundaries). The answer, tool calls and every
    other byte of the reasoning are unchanged."""
    rng = random.Random(7)
    for _ in range(200):
        feeds = feeds_of(REOPENED, rng)
        old, _ = runs(splitter(True, MODES[mode]), feeds)
        new, _ = runs(splitter(False, MODES[mode]), feeds)
        strip = lambda merged: [[c, OPEN_TAG.sub("", t) if c == "reasoning_content" else t] for c, t in merged]  # noqa: E731
        assert strip(new) == strip(old), feeds


def test_the_first_reasoning_words_leave_with_the_token_that_carries_them():
    words = [w + " " for w in PROSE.split()]
    _, first = runs(splitter(False, MODES["inside_thinking_no_tools"]), words)
    _, old_first = runs(splitter(True, MODES["inside_thinking_no_tools"]), words)
    assert first["reasoning_content"] == 0
    assert old_first["reasoning_content"] >= 5  # the old rule waited for 32 characters


def test_a_window_that_could_still_open_a_marker_still_waits():
    for tail in ("<", "</thi", "</think:opens", "<tool_ca", "<|im_e"):
        sp = splitter(False, MODES["inside_thinking_tools"])
        out = [t for _, t in sp.feed(PROSE + tail) if t]
        assert tail not in "".join(out), tail


def test_inkling_turns_show_their_reasoning_from_the_first_text_token():
    """Inkling's own channels put `<think>` before the first reasoning text; the first visible
    delta now leaves with that text, the third token of the turn."""
    from tests.test_inkling_committed_turns import C, InklingTokenizer, enc

    decoder = oa._IncrementalTokenDecoder(InklingTokenizer())
    sp = splitter(False, MODES["inside_thinking_no_tools"])
    ids = [C["message_model"], C["content_thinking"], *enc("The user wants a game.")]
    seen = None
    for index, token in enumerate(ids):
        if any(text for _, text in sp.feed(decoder.feed([token]))) and seen is None:
            seen = index
    assert seen == 2
