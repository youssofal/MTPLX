"""Exact-match memoization for chat prompt encoding.

Template render + BPE tokenization of the full message list runs on every
request and grows linearly with prompt size (Python-side wall time ahead of
any GPU work, i.e. pure TTFT tax). Agent clients (OpenCode/Pi/Claude Code)
resend byte-identical prefixes every turn, and warm-prefix session-bank hits
still paid a full re-encode of the entire transcript before the bank lookup
could even run.

This cache memoizes the FINAL token ids keyed on every input that affects
encoding. It is content-keyed (full messages/tools payload hashed), so two
requests that differ anywhere — including inside image payloads — never
share an entry. Hits return a copy; entries are immutable tuples.

Env: MTPLX_CHAT_ENCODE_CACHE=off disables; MTPLX_CHAT_ENCODE_CACHE_ENTRIES
overrides capacity (default 128).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from array import array
from collections import OrderedDict
from typing import Any


def _env_flag_on(name: str, default: str = "on") -> bool:
    return str(os.environ.get(name, default)).strip().lower() not in (
        "off",
        "0",
        "false",
        "no",
    )


class ChatEncodeCache:
    def __init__(self, max_entries: int | None = None) -> None:
        if max_entries is None:
            try:
                max_entries = int(os.environ.get("MTPLX_CHAT_ENCODE_CACHE_ENTRIES", "128"))
            except ValueError:
                max_entries = 128
        self.max_entries = max(1, max_entries)
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[tuple[int, ...], dict[str, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def enabled() -> bool:
        return _env_flag_on("MTPLX_CHAT_ENCODE_CACHE")

    @staticmethod
    def make_key(
        *,
        tokenizer_key: str,
        payload: dict[str, Any],
    ) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return (
            tokenizer_key
            + ":"
            + hashlib.sha256(blob.encode("utf-8", errors="surrogatepass")).hexdigest()
        )

    def get(self, key: str) -> tuple[list[int], dict[str, Any]] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            ids, observability = entry
        return list(ids), dict(observability)

    def put(self, key: str, ids: list[int], observability: dict[str, Any]) -> None:
        entry = (tuple(ids), dict(observability))
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
            }


GLOBAL_CHAT_ENCODE_CACHE = ChatEncodeCache()


class ChatSegmentEncodeMemo:
    """Token ids per segment of a segmented chat encode.

    The whole-payload cache above misses on every new agent turn, because
    the transcript grew. The segmented encoder
    (``_encode_rendered_chat_text_segmented``) already tokenizes each
    segment on its own and concatenates the results, so a segment's ids
    depend only on the tokenizer and the segment's exact text. Memoizing
    them lets a new turn tokenize only the segments it has not seen yet.

    Keys are the tokenizer identity plus a SHA-256 of the segment text, so
    the text itself is not retained. Ids are stored as int32 arrays (4 bytes
    per token). The memo is bounded by total stored tokens and by entry
    count; least recently used entries go first.

    Env: MTPLX_CHAT_SEGMENT_MEMO=off disables; MTPLX_CHAT_SEGMENT_MEMO_TOKENS
    overrides the token budget (default 1,048,576, about 4 MB of ids).
    """

    DEFAULT_MAX_TOKENS = 1 << 20
    MAX_ENTRIES = 4096

    def __init__(self, max_tokens: int | None = None) -> None:
        if max_tokens is None:
            max_tokens = _env_int(
                "MTPLX_CHAT_SEGMENT_MEMO_TOKENS", self.DEFAULT_MAX_TOKENS
            )
        self.max_tokens = max(1, max_tokens)
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, array] = OrderedDict()
        self._stored_tokens = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def enabled() -> bool:
        return _env_flag_on("MTPLX_CHAT_SEGMENT_MEMO")

    @staticmethod
    def make_key(*, tokenizer_key: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass"))
        return tokenizer_key + ":" + digest.hexdigest()

    def get(self, key: str) -> list[int] | None:
        with self._lock:
            ids = self._entries.get(key)
            if ids is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
        return ids.tolist()

    def put(self, key: str, ids: list[int]) -> None:
        if len(ids) > self.max_tokens:
            return
        try:
            stored = array("i", ids)
        except OverflowError:
            return
        with self._lock:
            self._forget(key)
            self._entries[key] = stored
            self._stored_tokens += len(stored)
            self._evict_to_bounds()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "tokens": self._stored_tokens,
                "hits": self.hits,
                "misses": self.misses,
            }

    def _forget(self, key: str) -> None:
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._stored_tokens -= len(previous)

    def _evict_to_bounds(self) -> None:
        while (
            self._stored_tokens > self.max_tokens
            or len(self._entries) > self.MAX_ENTRIES
        ):
            _, evicted = self._entries.popitem(last=False)
            self._stored_tokens -= len(evicted)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


GLOBAL_CHAT_SEGMENT_MEMO = ChatSegmentEncodeMemo()
