"""Chat-encode memoization: exact-match hits, key sensitivity, off-switch."""

from __future__ import annotations

import mtplx.server.openai as server
from mtplx.chat_encode_cache import ChatEncodeCache
from mtplx.server.openai import ChatMessage, _encode_messages


class CountingTokenizer:
    """Deterministic template+encode stub that counts render calls."""

    chat_template = "{{ messages }}"

    def __init__(self):
        self.render_calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.render_calls += 1
        text = repr(messages) + repr(sorted(kwargs.items()))
        return [len(text) % 251, kwargs.get("enable_thinking") and 1 or 0, len(messages)]

    def encode(self, text):
        return [ord(c) % 251 for c in str(text)]

    def decode(self, tokens, **_kwargs):
        return " ".join(str(t) for t in tokens)


def _messages():
    return [
        ChatMessage(role="system", content="be terse"),
        ChatMessage(role="user", content="write a haiku about caches"),
    ]


def _fresh_cache(monkeypatch, entries: int = 8) -> ChatEncodeCache:
    cache = ChatEncodeCache(max_entries=entries)
    monkeypatch.setattr(server, "GLOBAL_CHAT_ENCODE_CACHE", cache)
    return cache


def test_hit_returns_identical_ids_and_skips_render(monkeypatch):
    cache = _fresh_cache(monkeypatch)
    tok = CountingTokenizer()
    obs1: dict = {}
    ids1 = _encode_messages(
        tok, _messages(), enable_thinking=True, template_observability=obs1
    )
    assert obs1["chat_encode_cache"] == "miss"
    renders_after_first = tok.render_calls
    assert renders_after_first >= 1

    obs2: dict = {}
    ids2 = _encode_messages(
        tok, _messages(), enable_thinking=True, template_observability=obs2
    )
    assert ids2 == ids1
    assert obs2["chat_encode_cache"] == "hit"
    assert tok.render_calls == renders_after_first  # no re-render on hit
    assert cache.stats()["hits"] == 1

    # hit result must be a fresh list — caller mutation cannot poison the cache
    ids2.append(999)
    ids3 = _encode_messages(tok, _messages(), enable_thinking=True)
    assert ids3 == ids1


def test_key_sensitivity(monkeypatch):
    _fresh_cache(monkeypatch)
    tok = CountingTokenizer()
    base = _encode_messages(tok, _messages(), enable_thinking=True)
    flipped_thinking = _encode_messages(tok, _messages(), enable_thinking=False)
    changed_text = _encode_messages(
        tok,
        [ChatMessage(role="user", content="write a haiku about caches!")],
        enable_thinking=True,
    )
    with_tools = _encode_messages(
        tok,
        _messages(),
        enable_thinking=True,
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )
    # four distinct keys -> four misses, zero hits
    assert server.GLOBAL_CHAT_ENCODE_CACHE.stats()["misses"] == 4
    assert server.GLOBAL_CHAT_ENCODE_CACHE.stats()["hits"] == 0
    assert base != with_tools or base != flipped_thinking or base != changed_text


def test_template_change_invalidates(monkeypatch):
    _fresh_cache(monkeypatch)
    tok = CountingTokenizer()
    _encode_messages(tok, _messages(), enable_thinking=True)
    # simulate a template swap (startup profile application does this);
    # the per-tokenizer memoized key must not leak across templates
    tok.chat_template = "{{ messages }}v2"
    _encode_messages(tok, _messages(), enable_thinking=True)
    assert server.GLOBAL_CHAT_ENCODE_CACHE.stats()["misses"] == 2


def test_env_off_switch(monkeypatch):
    cache = _fresh_cache(monkeypatch)
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tok = CountingTokenizer()
    _encode_messages(tok, _messages(), enable_thinking=True)
    _encode_messages(tok, _messages(), enable_thinking=True)
    assert tok.render_calls == 2
    assert cache.stats() == {"entries": 0, "hits": 0, "misses": 0}


def test_lru_bound(monkeypatch):
    cache = _fresh_cache(monkeypatch, entries=2)
    tok = CountingTokenizer()
    for i in range(4):
        _encode_messages(
            tok,
            [ChatMessage(role="user", content=f"m{i}")],
            enable_thinking=True,
        )
    assert cache.stats()["entries"] == 2


def test_render_day_is_part_of_the_key(monkeypatch):
    """The rendered prompt embeds the current date (tool contract's date
    line, strftime_now templates). An exact repeat across local midnight
    must MISS and re-render — regression for the 2.5.3 pre-ship review F2
    (day-1 token ids were served on day 2 until eviction)."""
    cache = _fresh_cache(monkeypatch)
    tok = CountingTokenizer()
    _encode_messages(tok, _messages(), enable_thinking=True)
    _encode_messages(tok, _messages(), enable_thinking=True)
    assert cache.stats() == {"entries": 1, "hits": 1, "misses": 1}

    real_strftime = server.time.strftime

    def next_day(fmt, *args):
        if fmt == "%Y-%m-%d" and not args:
            return "2099-01-02"
        return real_strftime(fmt, *args)

    monkeypatch.setattr(server.time, "strftime", next_day)
    _encode_messages(tok, _messages(), enable_thinking=True)
    assert cache.stats()["misses"] == 2  # midnight rollover re-rendered
