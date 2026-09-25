"""Per-segment encode memo: bounded, isolated, and token-identical.

The segmented chat encode is the concatenation of independent per-segment
encodes, so memoizing a segment's ids by (tokenizer, exact text) must never
change a single token. These tests pin that parity (stub tokenizers always,
the real Qwen3.6 tokenizer when it is cached locally) plus the memo's bounds,
off switch and observability.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import mtplx.server.openai as oa
from mtplx.chat_encode_cache import ChatSegmentEncodeMemo


class GreedyTokenizer:
    """Longest-match tokenizer: a split point changes the ids, like BPE."""

    chat_template = "{{ messages }}"

    def __init__(self, vocab: list[str]):
        self.vocab = sorted(vocab, key=len, reverse=True)
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        ids: list[int] = []
        at = 0
        while at < len(text):
            for piece in self.vocab:
                if piece and text.startswith(piece, at):
                    ids.append(1000 + self.vocab.index(piece))
                    at += len(piece)
                    break
            else:
                ids.append(ord(text[at]))
                at += 1
        return ids

    @property
    def vocab_size(self):
        return 1000

    @property
    def added_tokens_decoder(self):
        return dict(enumerate(self.vocab))

    def add_tokens(self, pieces):
        self.vocab = sorted([*self.vocab, *pieces], key=len, reverse=True)


class WrappedTokenizer:
    """Like mlx-lm's TokenizerWrapper: forwards attributes to the HF one."""

    def __init__(self, inner):
        self._tokenizer = inner

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


@pytest.fixture
def memo(monkeypatch):
    fresh = ChatSegmentEncodeMemo(max_tokens=200_000)
    monkeypatch.setattr(oa, "GLOBAL_CHAT_SEGMENT_MEMO", fresh)
    monkeypatch.delenv("MTPLX_CHAT_SEGMENT_MEMO", raising=False)
    return fresh


def _segmented(tok, text, boundaries, obs=None):
    return oa._encode_rendered_chat_text_segmented(
        tok, text, boundaries, template_observability=obs
    )


def _without_memo(monkeypatch, tok, text, boundaries):
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO", "off")
    try:
        return _segmented(tok, text, boundaries)
    finally:
        monkeypatch.delenv("MTPLX_CHAT_SEGMENT_MEMO")


# --- the memo itself -------------------------------------------------------


def test_hit_returns_a_fresh_copy(memo):
    memo.put("k", [1, 2, 3])
    first = memo.get("k")
    first.append(99)
    assert memo.get("k") == [1, 2, 3]
    assert memo.stats() == {"entries": 1, "tokens": 3, "hits": 2, "misses": 0}


def test_token_budget_evicts_least_recently_used():
    memo = ChatSegmentEncodeMemo(max_tokens=10)
    memo.put("a", [1] * 4)
    memo.put("b", [2] * 4)
    assert memo.get("a") is not None  # a is now most recent
    memo.put("c", [3] * 4)
    assert memo.get("b") is None
    assert memo.get("a") == [1] * 4
    assert memo.stats()["tokens"] == 8


def test_entry_cap_bounds_tiny_segments(monkeypatch):
    monkeypatch.setattr(ChatSegmentEncodeMemo, "MAX_ENTRIES", 3)
    memo = ChatSegmentEncodeMemo(max_tokens=1_000)
    for i in range(5):
        memo.put(f"k{i}", [i])
    assert memo.stats()["entries"] == 3
    assert memo.get("k0") is None


def test_oversized_segment_is_not_stored():
    memo = ChatSegmentEncodeMemo(max_tokens=4)
    memo.put("small", [1, 2])
    memo.put("big", [1, 2, 3, 4, 5])
    assert memo.get("big") is None
    assert memo.get("small") == [1, 2]


def test_storing_a_key_twice_counts_its_tokens_once():
    memo = ChatSegmentEncodeMemo(max_tokens=100)
    memo.put("k", [1, 2, 3])
    memo.put("k", [1, 2, 3])
    assert memo.stats()["tokens"] == 3


def test_token_budget_env_override(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO_TOKENS", "123")
    assert ChatSegmentEncodeMemo().max_tokens == 123
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO_TOKENS", "not-a-number")
    assert ChatSegmentEncodeMemo().max_tokens == ChatSegmentEncodeMemo.DEFAULT_MAX_TOKENS


def test_concurrent_use_stays_consistent():
    memo = ChatSegmentEncodeMemo(max_tokens=50)

    def worker(offset):
        for i in range(500):
            key = f"k{(i + offset) % 40}"
            if memo.get(key) is None:
                memo.put(key, [i % 7] * 3)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stats = memo.stats()
    assert stats["tokens"] == 3 * stats["entries"] <= 50


# --- parity through the segmented encoder (stub tokenizer) ----------------

VOCAB = ["ab", "abc", "bc", "cd", "<s>", "\n\n", "é", "世界"]
TEXT = "<s>abcd\n\nabc</s>" + "bcabé世界" * 5 + "<s>ab\n\ncd" + "abc" * 50
BOUNDARIES = [5, 9, 20, 40, 70]


def test_memo_is_token_identical_to_plain_segmented_encode(memo, monkeypatch):
    tok = GreedyTokenizer(VOCAB)
    expected = _without_memo(monkeypatch, tok, TEXT, BOUNDARIES)
    cold = _segmented(tok, TEXT, BOUNDARIES)
    warm = _segmented(tok, TEXT, BOUNDARIES)
    assert cold == expected
    assert warm == expected


def test_growing_transcript_only_encodes_new_segments(memo, monkeypatch):
    tok = GreedyTokenizer(VOCAB)
    turn1, turn2 = TEXT[:40], TEXT
    _segmented(tok, turn1, [5, 9, 20])
    tok.encode_calls = 0
    obs: dict = {}
    ids = _segmented(tok, turn2, BOUNDARIES, obs)
    assert tok.encode_calls == 2
    assert ids == _without_memo(monkeypatch, tok, turn2, BOUNDARIES)
    # every turn-1 segment (its tail 20:40 included) is reused; 40:70 and
    # the new tail are the only segments tokenized
    assert obs["chat_segment_memo"]["hits"] == 4
    assert obs["chat_segment_memo"]["misses"] == 2
    assert obs["chat_segment_memo"]["reused_tokens"] == len(
        _without_memo(monkeypatch, tok, turn1, [5, 9, 20])
    )


def test_edit_in_earlier_history_is_not_served_stale(memo, monkeypatch):
    tok = GreedyTokenizer(VOCAB)
    _segmented(tok, TEXT, BOUNDARIES)
    edited = TEXT[:10] + "X" + TEXT[11:]
    obs: dict = {}
    ids = _segmented(tok, edited, BOUNDARIES, obs)
    assert ids == _without_memo(monkeypatch, tok, edited, BOUNDARIES)
    assert ids != _without_memo(monkeypatch, tok, TEXT, BOUNDARIES)
    assert obs["chat_segment_memo"]["misses"] == 1


def test_tokenizers_never_share_entries(memo, monkeypatch):
    plain = GreedyTokenizer([])
    merging = GreedyTokenizer(VOCAB)
    _segmented(plain, TEXT, BOUNDARIES)
    ids = _segmented(merging, TEXT, BOUNDARIES)
    assert ids == _without_memo(monkeypatch, merging, TEXT, BOUNDARIES)
    assert ids != _without_memo(monkeypatch, plain, TEXT, BOUNDARIES)


def test_tokens_added_in_place_are_not_served_stale(memo, monkeypatch):
    tok = GreedyTokenizer(VOCAB)
    _segmented(tok, TEXT, BOUNDARIES)
    tok.add_tokens(["abcab"])
    obs: dict = {}
    ids = _segmented(tok, TEXT, BOUNDARIES, obs)
    assert ids == _without_memo(monkeypatch, tok, TEXT, BOUNDARIES)
    assert obs["chat_segment_memo"]["hits"] == 0


def test_vocab_key_reads_through_a_wrapper():
    inner = GreedyTokenizer(VOCAB)
    assert oa._chat_segment_vocab_size(WrappedTokenizer(inner)) == "1000+8"
    inner.add_tokens(["abcab"])
    assert oa._chat_segment_vocab_size(WrappedTokenizer(inner)) == "1000+9"
    assert oa._chat_segment_vocab_size(object()) is None


def test_wrapped_tokenizer_sees_tokens_added_to_the_inner_one(memo, monkeypatch):
    inner = GreedyTokenizer(VOCAB)
    wrapped = WrappedTokenizer(inner)
    _segmented(wrapped, TEXT, BOUNDARIES)
    inner.add_tokens(["abcab"])
    obs: dict = {}
    ids = _segmented(wrapped, TEXT, BOUNDARIES, obs)
    assert ids == _without_memo(monkeypatch, wrapped, TEXT, BOUNDARIES)
    assert obs["chat_segment_memo"]["hits"] == 0


def test_off_switch_bypasses_the_memo(memo, monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO", "off")
    obs: dict = {}
    _segmented(GreedyTokenizer(VOCAB), TEXT, BOUNDARIES, obs)
    assert memo.stats() == {"entries": 0, "tokens": 0, "hits": 0, "misses": 0}
    assert "chat_segment_memo" not in obs


def test_unsegmented_encode_does_not_touch_the_memo(memo):
    obs: dict = {}
    _segmented(GreedyTokenizer(VOCAB), TEXT, [], obs)
    assert memo.stats()["entries"] == 0
    assert "chat_segment_memo" not in obs


# --- parity with the real tokenizer and chat template ---------------------

MODEL_DIR = (
    Path.home() / ".mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance"
)
needs_real_tokenizer = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").exists(),
    reason="Qwen3.6 model pack not cached locally",
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
    for name in ("read", "bash")
]
TINY_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlE"
    "QVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture(scope="module")
def real_tok():
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    return _load_tokenizer_resilient(MODEL_DIR, config)


def _tool_turn(i: int, result: str) -> list[dict]:
    call = {
        "id": f"call_{i}",
        "type": "function",
        "function": {"name": "read", "arguments": json.dumps({"path": f"f{i}.py"})},
    }
    return [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"Step {i}: read the next file. Ünïcödé 世界 🚀",
            "tool_calls": [call],
        },
        {"role": "tool", "tool_call_id": f"call_{i}", "content": result},
    ]


def _agent_transcript(turns: int) -> list[dict]:
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Fix the bug shown here."},
                {"type": "image_url", "image_url": {"url": TINY_PNG}},
            ],
        },
    ]
    for i in range(turns):
        if i in (2, 3):
            result = "same file content\n" * 20  # repeated segments
        elif i == 4:
            result = "def huge():\n    return 'x' * 80\n" * 4000  # very long segment
        else:
            result = f"def f{i}():\n    return {i}  # naïve café 日本語 \u200b\n" * 10
        messages.extend(_tool_turn(i, result))
    return messages


def _encode_messages(tok, messages, *, thinking, mode, obs=None):
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=thinking,
        reasoning_effort="medium",
        tools=TOOLS,
        tool_prompt_mode=mode,
        template_observability=obs if obs is not None else {},
    )


@needs_real_tokenizer
@pytest.mark.parametrize("mode", ["hybrid", "compact"])
@pytest.mark.parametrize("thinking", [True, False])
def test_real_growing_agent_transcript_is_token_identical(
    real_tok, memo, monkeypatch, mode, thinking
):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    for turns in range(1, 7):
        messages = _agent_transcript(turns)
        with_memo = _encode_messages(real_tok, messages, thinking=thinking, mode=mode)
        monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO", "off")
        without = _encode_messages(real_tok, messages, thinking=thinking, mode=mode)
        monkeypatch.delenv("MTPLX_CHAT_SEGMENT_MEMO")
        assert with_memo == without, f"diverged at {turns} turns"


@needs_real_tokenizer
def test_real_warm_turn_reuses_history_and_survives_an_edit(
    real_tok, memo, monkeypatch
):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    _encode_messages(real_tok, _agent_transcript(5), thinking=True, mode="hybrid")

    obs: dict = {}
    grown = _agent_transcript(6)
    ids = _encode_messages(real_tok, grown, thinking=True, mode="hybrid", obs=obs)
    assert obs["chat_segment_memo"]["hits"] >= 5
    assert obs["chat_segment_memo"]["misses"] == 1

    edited = _agent_transcript(6)
    edited[3]["content"] = "an earlier tool result, edited"
    obs = {}
    edited_ids = _encode_messages(real_tok, edited, thinking=True, mode="hybrid", obs=obs)
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO", "off")
    assert ids == _encode_messages(real_tok, grown, thinking=True, mode="hybrid")
    assert edited_ids == _encode_messages(
        real_tok, edited, thinking=True, mode="hybrid"
    )
    assert edited_ids != ids
    assert obs["chat_segment_memo"]["misses"] >= 1
