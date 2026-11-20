"""The committed-token splice gives exactly what it gave before its equal runs
were compared in slices (12 Nov: the walk stepped one token at a time, 10 ms per call
at 150K tokens, and a request runs it more than once).

The previous implementation is kept below, verbatim, as the reference. A small
vocabulary whose tokens spell overlapping strings makes every seam case: one token
against two that spell it, whitespace the response stripped, a chat-control token at
the end, a real edit that never re-synchronises, image rows, too many spans.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import pytest

from mtplx.server import openai
from mtplx.token_prefix import common_prefix_len, common_run_len
from mtplx.vision.splice import _BANK_KEY_FLAG


def ref_splice_committed_token_ids(
    prompt_ids: Sequence[int],
    committed: Sequence[int],
    tokenizer: Any,
    *,
    window: int = openai._COMMITTED_SPLICE_WINDOW,
    max_spans: int = openai._COMMITTED_SPLICE_MAX_SPANS,
) -> tuple[list[int], dict[str, Any]]:
    """Re-express ``prompt_ids`` with the session's committed ids wherever the
    two decode to the same text.

    A model's sampled token sequence is not always the canonical BPE encoding
    of its own text: the 2026-09-08 Hermes receipt had ``"Nothing`` emitted
    as one token 8,498 tokens into a write_file call where the tokenizer
    encodes ``"`` + ``Nothing``. The client's re-tokenized history then
    diverges from the committed stream at that spot, the generation-final
    snapshot is refused, and the next turn re-prefills the whole assistant
    turn (30,038 tokens, 41 s TTFT). The bytes are identical; only the token
    boundaries differ, and the KV state the model built belongs to its own
    ids. At each divergence this walks a bounded window on both sides for
    the shortest pair of token runs with the same decoded text and takes the
    committed run; a window that does not re-synchronise (a real edit) ends
    the splice and the rest of the prompt is kept as sent. Windows whose
    text carries a replacement character (a split multi-byte character) are
    never matched. Returns the ids and a receipt.
    """

    from mtplx.vision.splice import is_image_key

    prompt = [int(token) for token in prompt_ids]
    stream = [int(token) for token in committed]
    receipt: dict[str, Any] = {"spans": 0, "tokens_in": 0, "tokens_out": 0}
    if not prompt or not stream:
        return prompt, receipt
    common = common_prefix_len(prompt, stream)
    if common >= len(prompt) or common >= len(stream):
        return prompt, receipt

    def _text(ids: Sequence[int]) -> str | None:
        # An image row of a keyed stream (vision_bank_key_ids) is never text:
        # the splice copies it only where both sides hold the same row.
        if any(is_image_key(token) for token in ids):
            return None
        try:
            text = tokenizer.decode(list(ids))
        except Exception:
            return None
        if not isinstance(text, str) or not text or "\ufffd" in text:
            return None
        return text

    out: list[int] = prompt[:common]
    p = common
    c = common
    spans = 0
    while p < len(prompt) and c < len(stream):
        if prompt[p] == stream[c]:
            out.append(prompt[p])
            p += 1
            c += 1
            continue
        if spans >= max_spans:
            break
        # Whitespace the response stripped. The visible content is served
        # with its leading and trailing whitespace removed, so the model's
        # own whitespace tokens are missing from the re-rendered history
        # (a length-cut turn ending in a newline: committed ``:``, ``\n``,
        # history ``:``, ``<|im_end|>``). Put them back where the streams
        # re-align right after them, or where the committed stream ends
        # with them and the prompt goes on with a chat-control token.
        ws = 0
        while ws < window and c + ws < len(stream):
            piece = _text(stream[c + ws : c + ws + 1])
            if piece is None or piece.strip():
                break
            ws += 1
        if ws:
            end = c + ws
            realigned = end < len(stream) and stream[end] == prompt[p]
            tail = end == len(stream) and openai._is_chat_control_token(tokenizer, prompt[p])
            if realigned or tail:
                out.extend(stream[c:end])
                receipt["whitespace_tokens"] = int(receipt.get("whitespace_tokens", 0)) + ws
                spans += 1
                c = end
                continue
        found: tuple[int, int] | None = None
        for total in range(2, 2 * window + 1):
            for dp in range(1, min(window, total - 1) + 1):
                dc = total - dp
                if dc < 1 or dc > window:
                    continue
                if p + dp > len(prompt) or c + dc > len(stream):
                    continue
                left = _text(prompt[p : p + dp])
                if left is None:
                    continue
                if left == _text(stream[c : c + dc]):
                    found = (dp, dc)
                    break
            if found is not None:
                break
        if found is None:
            break
        dp, dc = found
        out.extend(stream[c : c + dc])
        receipt["tokens_in"] += dp
        receipt["tokens_out"] += dc
        spans += 1
        p += dp
        c += dc
    out.extend(prompt[p:])
    receipt["spans"] = spans
    if spans:
        receipt["first_divergence"] = int(common)
        receipt["cp_after"] = int(common_prefix_len(out, stream))
    return out, receipt


VOCAB = {1: "a", 2: "b", 3: "ab", 4: "c", 5: "bc", 6: "abc", 7: " ", 8: "\n", 9: "  ",
         10: "x", 11: "<|im_end|>", 12: "d", 13: "cd", 14: "\ufffd"}


class _Tok:
    all_special_ids = [11]

    def decode(self, ids):
        return "".join(VOCAB[int(i)] for i in ids)


def _respell(rng, ids):
    """The same text with some tokens split or merged."""
    out = []
    i = 0
    while i < len(ids):
        t = ids[i]
        if t == 3 and rng.random() < 0.5:
            out += [1, 2]
        elif t == 6 and rng.random() < 0.5:
            out += rng.choice([[1, 5], [3, 4], [1, 2, 4]])
        elif t == 1 and i + 1 < len(ids) and ids[i + 1] == 2 and rng.random() < 0.5:
            out.append(3)
            i += 1
        elif t == 13 and rng.random() < 0.5:
            out += [4, 12]
        else:
            out.append(t)
        i += 1
    return out


def _case(rng):
    base = [rng.choice([1, 2, 3, 4, 5, 6, 7, 10, 12, 13]) for _ in range(rng.randrange(0, 300))]
    committed = list(base)
    prompt = _respell(rng, base)
    if rng.random() < 0.2 and prompt:
        prompt.insert(rng.randrange(len(prompt)), 10)  # a real edit
    if rng.random() < 0.2:
        committed += [rng.choice([7, 8, 9])] * rng.randrange(1, 4)  # whitespace the client dropped
        prompt += [11]
    if rng.random() < 0.1 and committed:
        committed[rng.randrange(len(committed))] = _BANK_KEY_FLAG | rng.getrandbits(20)
    if rng.random() < 0.05 and prompt:
        prompt[rng.randrange(len(prompt))] = 14
    return prompt, committed


@pytest.mark.parametrize("seed", range(8))
def test_the_splice_matches_its_previous_walk(seed):
    rng = random.Random(seed)
    tok = _Tok()
    for _ in range(400):
        prompt, committed = _case(rng)
        for kwargs in ({}, {"max_spans": 2}, {"window": 3}):
            assert openai._splice_committed_token_ids(prompt, committed, tok, **kwargs) == \
                ref_splice_committed_token_ids(prompt, committed, tok, **kwargs)
        assert openai._splice_committed_token_ids(tuple(prompt), tuple(committed), tok) == \
            ref_splice_committed_token_ids(tuple(prompt), tuple(committed), tok)


def test_common_run_len_is_the_prefix_length_from_the_offsets():
    rng = random.Random(3)
    for _ in range(2000):
        left = [rng.randrange(3) for _ in range(rng.randrange(0, 300))]
        right = list(left) if rng.random() < 0.5 else [rng.randrange(3) for _ in range(rng.randrange(0, 300))]
        i = rng.randrange(0, len(left) + 2)
        j = rng.randrange(0, len(right) + 2)
        assert common_run_len(left, i, right, j) == common_prefix_len(left[i:], right[j:])
        assert common_run_len(tuple(left), i, right, j) == common_prefix_len(left[i:], right[j:])
