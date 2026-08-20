"""Visible-stream cadence gates (streamwar 2026-08-19).

The freeze-then-vomit stutter shipped six times because nothing asserted the
POST-DECODER release cadence: the producer census read clean while
_IncrementalTokenDecoder held whitespace-free text hostage (150-880 ms per
line measured live — outputs/streamscope-20260819/baseline/). These tests
gate the release policy itself, model-free, so the bug class cannot return:

  1. Every feed that decodes at least one complete codepoint must emit —
     on every content class the replay ranked (prose, dense code, table
     rows, minified JSON, URLs, CJK, emoji).
  2. Bursts stay round-sized: an emit never exceeds what its feed carried.
  3. Reassembly is byte-exact across thousands of random token splits.
  4. Multi-byte codepoints split across feeds are held only until their
     continuation bytes arrive (U+FFFD never leaks, text never drops).
  5. The chunk-split </think> and tool-marker protocol still splits
     correctly downstream when the decoder releases per token boundary.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from mtplx.server.openai import (
    _IncrementalTokenDecoder,
    _ThinkingContentStreamSplitter,
    _coalesce_stream_fields,
)


class ByteTokenizer:
    """Byte-level tokenizer stub with real byte-BPE decode semantics.

    Token id == byte value; decode() is bytes -> UTF-8 with replacement,
    which reproduces exactly how a byte-level BPE tokenizer surfaces an
    incomplete multi-byte codepoint (U+FFFD until the tail bytes arrive).
    """

    def decode(self, tokens, **_kwargs):
        return bytes(int(t) & 0xFF for t in tokens).decode("utf-8", errors="replace")


def byte_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def random_token_feeds(
    data: list[int], rng: random.Random, lo: int = 1, hi: int = 4
) -> list[list[int]]:
    feeds: list[list[int]] = []
    index = 0
    while index < len(data):
        step = rng.randint(lo, hi)
        feeds.append(data[index : index + step])
        index += step
    return feeds


CONTENT_CLASSES = {
    "prose": "The quick brown fox jumps over the lazy dog and keeps going. " * 12,
    "dense_python": (
        "def pack(x):\n"
        "    return {'k':x*2,'j':x**2,'m':[i for i in range(x)],'s':str(x)}\n"
    ) * 10,
    "markdown_table": ("|---------|--------|---------|-------|\n"
                       "| quick | merge | bubble | heap |\n") * 12,
    "minified_json": (
        '{"id":1,"name":"item","tags":["a","b","c"],"active":true,"score":9.75},'
    ) * 14,
    "url_run": "https://example.com/deep/path/segment?alpha=1&beta=two&gamma=3.14&delta=x" * 8,
    "cjk": "模型正在流式生成中文文本并且不包含任何空格所以旧的空格闸门会把它整段扣住" * 6,
    "emoji_mixed": "Build 🚀 status ✅ heat 🔥 loop ➰ done 🎉 " * 10,
}

MULTIBYTE_CLASSES = {"cjk", "emoji_mixed"}


@pytest.mark.parametrize("class_name", sorted(CONTENT_CLASSES))
def test_every_token_boundary_releases_text(class_name: str) -> None:
    # The cadence gate: simulate verify rounds of 4 tokens each on a
    # virtual clock. A feed may come up empty ONLY while a multi-byte
    # codepoint is split across feeds — never because of whitespace
    # gating, hold thresholds, or any other cadence-destroying policy.
    corpus = CONTENT_CLASSES[class_name]
    rng = random.Random(20260819)
    decoder = _IncrementalTokenDecoder(ByteTokenizer())
    feeds = random_token_feeds(byte_ids(corpus), rng, lo=4, hi=4)

    emitted: list[str] = []
    empty_streak = 0
    worst_empty_streak = 0
    for feed in feeds:
        text = decoder.feed(feed)
        emitted.append(text)
        if text:
            empty_streak = 0
        else:
            empty_streak += 1
            worst_empty_streak = max(worst_empty_streak, empty_streak)

    allowed = 1 if class_name in MULTIBYTE_CLASSES else 0
    assert worst_empty_streak <= allowed, (
        f"{class_name}: decoder went silent for {worst_empty_streak} consecutive "
        f"4-token rounds — a visible-stream freeze (allowed: {allowed})"
    )
    # Byte-exact reassembly.
    assert "".join(emitted) + decoder.finish() == corpus
    # Burst gate: one emit never exceeds one feed's text plus a completed
    # carry-over codepoint (4 bytes). No multi-round vomit pastes.
    max_feed_chars = 4 + 4
    oversized = [len(piece) for piece in emitted if len(piece) > max_feed_chars]
    assert not oversized, f"{class_name}: burst(s) of {oversized} chars from 4-byte feeds"


def test_byte_exact_reassembly_across_10k_random_splits() -> None:
    corpus = (
        "prose then `code_with_underscores(1,2)` then 中文 then 🚀 then\n"
        '{"minified":true,"n":[1,2,3]},"url":"https://x.y/z?a=1&b=2"\n'
    )
    data = byte_ids(corpus)
    rng = random.Random(7)
    for _ in range(10_000):
        decoder = _IncrementalTokenDecoder(ByteTokenizer())
        parts = [decoder.feed(feed) for feed in random_token_feeds(data, rng)]
        assert "".join(parts) + decoder.finish() == corpus


@pytest.mark.parametrize("codepoint", ["é", "中", "🚀", "𝕏"])
def test_split_codepoint_is_held_then_released(codepoint: str) -> None:
    prefix, suffix = "a", "b"
    data = byte_ids(prefix + codepoint + suffix)
    # Split at EVERY byte boundary, including through the codepoint.
    for split in range(1, len(data)):
        decoder = _IncrementalTokenDecoder(ByteTokenizer())
        first = decoder.feed(data[:split])
        second = decoder.feed(data[split:])
        joined = first + second + decoder.finish()
        assert joined == prefix + codepoint + suffix, (
            f"split at byte {split}: got {joined!r}"
        )
        assert "\ufffd" not in joined


def test_legitimate_replacement_char_still_flows() -> None:
    # A real U+FFFD in the content (its own valid UTF-8 encoding) must not
    # deadlock the tail hold: the next feed releases it.
    decoder = _IncrementalTokenDecoder(ByteTokenizer())
    first = decoder.feed(byte_ids("x\ufffd"))
    second = decoder.feed(byte_ids("y"))
    assert first + second + decoder.finish() == "x\ufffdy"


def test_split_think_close_tag_splits_channels_correctly() -> None:
    # The old decoder held text specifically so a chunk-split </think>
    # completed inside its cache. That duty belongs to the splitter's
    # partial-prefix hold; prove the chain end-to-end with the tag split
    # at every byte boundary.
    body = "deep thought</think>The answer is 42."
    data = byte_ids(body)
    tag_start = body.index("</think>")
    for split in range(tag_start, tag_start + len("</think>") + 1):
        decoder = _IncrementalTokenDecoder(ByteTokenizer())
        splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)
        chunks = list(splitter.start())
        for feed in (data[:split], data[split:]):
            text = decoder.feed(feed)
            if text:
                chunks.extend(splitter.feed(text))
        tail = decoder.finish()
        if tail:
            chunks.extend(splitter.feed(tail))
        chunks.extend(splitter.finish())
        reasoning = "".join(t for f, t in chunks if f == "reasoning_content")
        content = "".join(t for f, t in chunks if f == "content")
        assert reasoning == "deep thought", f"split {split}: {reasoning!r}"
        assert content == "The answer is 42.", f"split {split}: {content!r}"


def test_split_tool_call_marker_survives_streaming() -> None:
    # Tool-call protocol: the marker split across decoder feeds must
    # reassemble byte-exactly on the content channel (the translator
    # downstream needs the exact span).
    body = '</think>before <tool_call>{"name":"x"}</tool_call> after'
    data = byte_ids(body)
    marker_start = body.index("<tool_call>")
    for split in (marker_start + 3, marker_start + 7, marker_start + 10):
        decoder = _IncrementalTokenDecoder(ByteTokenizer())
        splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)
        chunks = list(splitter.start())
        for feed in (data[:split], data[split:]):
            text = decoder.feed(feed)
            if text:
                chunks.extend(splitter.feed(text))
        tail = decoder.finish()
        if tail:
            chunks.extend(splitter.feed(tail))
        chunks.extend(splitter.finish())
        content = "".join(t for f, t in chunks if f == "content")
        assert '<tool_call>{"name":"x"}</tool_call>' in content, (
            f"split {split}: {content!r}"
        )


_QWEN_PACK = Path.home() / ".mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed"


@pytest.mark.skipif(
    not (_QWEN_PACK / "tokenizer.json").exists(),
    reason="local Qwen pack not present",
)
def test_real_qwen_tokenizer_minified_json_never_stalls() -> None:
    # The worst measured class (p50 386 ms gaps on the shipped build)
    # replayed through the REAL tokenizer: token-per-feed, every feed
    # with a complete codepoint must emit.
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(_QWEN_PACK), trust_remote_code=False
    )
    line = (
        '{"id":7,"name":"payload","tags":["alpha","beta","gamma"],'
        '"active":true,"nested":{"k":"v","n":[1,2,3]}},'
    ) * 12
    ids = tokenizer.encode(line, add_special_tokens=False)
    decoder = _IncrementalTokenDecoder(tokenizer)
    emitted = []
    empty_streak = 0
    worst = 0
    for token in ids:
        text = decoder.feed([token])
        emitted.append(text)
        if text:
            empty_streak = 0
        else:
            empty_streak += 1
            worst = max(worst, empty_streak)
    assert worst <= 1, f"real-tokenizer stall: {worst} consecutive silent feeds"
    assert "".join(emitted) + decoder.finish() == line


# --- Same-field coalescing (2026-08-19 round 2: one write per channel run,
# --- not one per token — the 20-token starved-drain burst arrived as 20
# --- main-queue hops in the app at ~80 envelope bytes per visible char).


def test_coalesce_stream_fields_is_interleaving_identity() -> None:
    # Strongest property: expanding every (field, text) tuple to per-char
    # (field, char) pairs must give the identical sequence before and
    # after coalescing — bytes, order, and channel of every character are
    # untouched; only write granularity changes. And no two adjacent
    # output tuples may share a field.
    rng = random.Random(20260819)
    fields = ["content", "reasoning_content"]
    for _ in range(200):
        chunks = [
            (rng.choice(fields), "".join(rng.choice("ab{}:,\n ") for _ in range(rng.randint(1, 5))))
            for _ in range(rng.randint(0, 40))
        ]
        merged = _coalesce_stream_fields(list(chunks))
        expand = lambda pairs: [(f, ch) for f, t in pairs for ch in t]
        assert expand(merged) == expand(chunks)
        assert all(a[0] != b[0] for a, b in zip(merged, merged[1:])), merged


def test_coalesce_merges_burst_to_one_write_per_channel_run() -> None:
    burst = [("content", "x")] * 20
    assert _coalesce_stream_fields(burst) == [("content", "x" * 20)]
    runs = (
        [("reasoning_content", "a")] * 3
        + [("content", "b")] * 5
        + [("reasoning_content", "c")] * 2
    )
    assert _coalesce_stream_fields(runs) == [
        ("reasoning_content", "aaa"),
        ("content", "bbbbb"),
        ("reasoning_content", "cc"),
    ]
    assert _coalesce_stream_fields([]) == []
    assert _coalesce_stream_fields([("content", "solo")]) == [("content", "solo")]


def test_coalesce_never_merges_across_think_close_boundary() -> None:
    # Chain proof: per-token feeds through decoder + splitter produce many
    # tiny same-channel tuples with one reasoning->content flip at the
    # </think> tag. Coalescing must fold each side to single runs and
    # never join across the flip.
    body = "chain of thought here</think>final answer text"
    decoder = _IncrementalTokenDecoder(ByteTokenizer())
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)
    pairs = list(splitter.start())
    for token in byte_ids(body):
        text = decoder.feed([token])
        if text:
            pairs.extend((f, t) for f, t in splitter.feed(text) if t)
    tail = decoder.finish()
    if tail:
        pairs.extend((f, t) for f, t in splitter.feed(tail) if t)
    pairs.extend((f, t) for f, t in splitter.finish() if t)

    merged = _coalesce_stream_fields(pairs)
    assert all(a[0] != b[0] for a, b in zip(merged, merged[1:]))
    reasoning = "".join(t for f, t in merged if f == "reasoning_content")
    content = "".join(t for f, t in merged if f == "content")
    assert reasoning == "chain of thought here"
    assert content == "final answer text"
