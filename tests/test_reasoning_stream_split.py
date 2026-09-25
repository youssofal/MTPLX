"""Pure (MLX-free) regression tests for the streaming reasoning splitter.

The existing splitter tests live in ``tests/test_openai_bridge.py``, which
transitively imports ``mlx`` and therefore only runs on macOS/arm64. These
tests import ``mtplx.reasoning_codecs`` directly (no MLX) so they run
everywhere, including CI on non-Apple platforms.
"""

from __future__ import annotations

import pytest

from mtplx.reasoning_codecs import (
    GEMMA4_THINK_CLOSE,
    GEMMA4_THINK_OPEN,
    QwenThinkingContentStreamSplitter,
    split_reasoning_text,
    stream_splitter_for_parser,
)


def _split(chunks: list[str]) -> tuple[str, str]:
    sp = QwenThinkingContentStreamSplitter(thinking_enabled=True)
    out: list[tuple[str, str]] = []
    for c in chunks:
        out += sp.feed(c)
    out += sp.finish()
    content = "".join(t for f, t in out if f == "content")
    reasoning = "".join(t for f, t in out if f == "reasoning_content")
    return content, reasoning


def test_no_reasoning_leak_when_long_alias_tag_splits_across_chunks() -> None:
    # A reasoning tag longer than "</think>" (e.g. "<reasoning>") split across
    # an SSE chunk boundary must not leak the reasoning block -- or its raw
    # markup -- into the user-visible content.
    content, reasoning = _split(["R1</think>V1 <reasoni", "ng>SECRET</reasoning> V2"])
    assert "SECRET" not in content, f"reasoning leaked into visible content: {content!r}"
    assert "<reasoning" not in content, f"raw markup leaked into visible content: {content!r}"
    assert "SECRET" in reasoning


def test_visible_content_preserved_around_split_reasoning() -> None:
    content, _ = _split(["R1</think>V1 <reasoni", "ng>SECRET</reasoning> V2"])
    assert content == "V1 V2"


def test_poolside_v1_uses_think_tag_reasoning_codec() -> None:
    parts = split_reasoning_text(
        "<think>inspect inputs</think>Final answer",
        parser="poolside_v1",
        thinking_enabled=True,
    )
    assert parts.reasoning == "inspect inputs"
    assert parts.content == "Final answer"

    splitter = stream_splitter_for_parser(
        "poolside_v1",
        thinking_enabled=True,
    )
    chunks = splitter.feed("inspect inputs</think>Final") + splitter.finish()
    assert "".join(text for field, text in chunks if field == "reasoning_content") == "inspect inputs"
    assert "".join(text for field, text in chunks if field == "content") == "Final"


def _split_lfm2(chunks: list[str], *, thinking_enabled: bool = True) -> tuple[str, str]:
    sp = stream_splitter_for_parser("lfm2", thinking_enabled=thinking_enabled)
    out: list[tuple[str, str]] = []
    for c in chunks:
        out += sp.feed(c)
    out += sp.finish()
    content = "".join(t for f, t in out if f == "content")
    reasoning = "".join(t for f, t in out if f == "reasoning_content")
    return content, reasoning


def test_lfm2_stream_without_think_block_is_all_visible_content() -> None:
    # LFM templates never prefill an open think tag, so a response that skips
    # thinking must stream as content. The prefilled-start splitter would file
    # every token as reasoning until a close tag that never arrives.
    content, reasoning = _split_lfm2(["The answer", " is 4."])
    assert content == "The answer is 4."
    assert reasoning == ""


def test_lfm2_stream_splits_model_emitted_think_block() -> None:
    content, reasoning = _split_lfm2(["<think>count the legs</think>", "Four."])
    assert content == "Four."
    assert "count the legs" in reasoning
    assert "<think>" not in content


def test_lfm2_stream_think_marker_split_across_chunks() -> None:
    content, reasoning = _split_lfm2(["<thi", "nk>SECRET</th", "ink>Visible"])
    assert "SECRET" not in content
    assert "SECRET" in reasoning
    assert "Visible" in content


def test_lfm2_split_reasoning_text_handles_both_shapes() -> None:
    parts = split_reasoning_text(
        "<think>R</think>C",
        parser="lfm2",
        thinking_enabled=True,
    )
    assert (parts.reasoning, parts.content) == ("R", "C")
    bare = split_reasoning_text(
        "plain answer",
        parser="lfm2",
        thinking_enabled=True,
    )
    assert bare.reasoning == ""
    assert bare.content == "plain answer"


def test_lfm2_normalize_keeps_plain_history_as_content() -> None:
    from mtplx.reasoning_codecs import normalize_reasoning_tags

    # A historical assistant message without think markup must stay visible
    # content; the prefilled-thinking normalizer would wrap it in think tags.
    normalized = normalize_reasoning_tags(
        "Deployed the fix.",
        parser="lfm2",
        thinking_enabled=True,
    )
    assert normalized == "Deployed the fix."
    with_think = normalize_reasoning_tags(
        "<think>weigh options</think>Ship it.",
        parser="lfm2",
        thinking_enabled=True,
    )
    assert "Ship it." in with_think
    assert with_think.index("weigh options") < with_think.index("Ship it.")


# The model separates the close tag from its answer with "\n\n". Non-stream
# responses strip it; streamed content must concatenate to the same text
# whether the separator arrives with the close tag or in a later chunk.
_ANSWER_SEPARATOR_CASES = (
    ("qwen3", True, "plan</think>\n\nThe answer is 4."),
    ("qwen3", True, "plan</think>\n \t\nThe answer is 4."),
    ("qwen3", False, "\n\nThe answer is 4."),
    ("lfm2", True, "<think>plan</think>\n\nThe answer is 4."),
    (
        "gemma4",
        True,
        f"{GEMMA4_THINK_OPEN}plan{GEMMA4_THINK_CLOSE}\n\nThe answer is 4.",
    ),
    ("gemma4", False, "\n\nThe answer is 4."),
)


def _stream_content(parser: str, thinking_enabled: bool, chunks: list[str]) -> str:
    sp = stream_splitter_for_parser(parser, thinking_enabled=thinking_enabled)
    out = list(sp.start())
    for c in chunks:
        out += sp.feed(c)
    out += sp.finish()
    return "".join(t for f, t in out if f == "content")


@pytest.mark.parametrize(
    ("parser", "thinking_enabled", "text"), _ANSWER_SEPARATOR_CASES
)
def test_answer_separator_never_streams_as_content(
    parser, thinking_enabled, text
) -> None:
    expected = split_reasoning_text(
        text, parser=parser, thinking_enabled=thinking_enabled
    ).content
    assert expected == "The answer is 4."
    splits = [[text], list(text)]
    splits += [[text[:i], text[i:]] for i in range(1, len(text))]
    for chunks in splits:
        assert _stream_content(parser, thinking_enabled, chunks) == expected, chunks


def test_answer_whitespace_after_the_first_visible_character_is_kept() -> None:
    content = _stream_content(
        "qwen3", True, ["plan</think>", "\n\n", "Line one.", "\n\n", "Line two."]
    )
    assert content == "Line one.\n\nLine two."
