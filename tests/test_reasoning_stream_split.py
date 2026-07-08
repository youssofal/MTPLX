"""Pure (MLX-free) regression tests for the streaming reasoning splitter.

The existing splitter tests live in ``tests/test_openai_bridge.py``, which
transitively imports ``mlx`` and therefore only runs on macOS/arm64. These
tests import ``mtplx.reasoning_codecs`` directly (no MLX) so they run
everywhere, including CI on non-Apple platforms.
"""

from __future__ import annotations

from mtplx.reasoning_codecs import QwenThinkingContentStreamSplitter


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
