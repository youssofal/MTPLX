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
