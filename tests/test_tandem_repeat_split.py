"""The retry-pollution collapse finds the same split as its full scan, in linear time.

`_collapse_repeated_user_text` collapses a user message that is one text said
twice (a cancelled retry OpenCode persisted as "Hi, how are you?Hi, how are
you?"). It used to build two retry keys for every split of the message: on
Claude Code's requests, whose user messages carry 1K to 4K characters of
context, the request policy spent 75 to 105 ms there on every request.
`_first_tandem_split` tries only the splits where the two keys can be equal
and must return exactly what the scan returns, for every text.
"""

from __future__ import annotations

import random
import time

from mtplx.server.openai import (
    _collapse_repeated_user_text,
    _first_tandem_split,
    _first_tandem_split_by_scan,
    _retry_user_key,
)

_WORDS = ["hi", "Hello", "how", "are", "you?", "the", "file", "notes.js", "\u03a3\u039f\u03a6\u0399\u0391\u03a3", "stra\u00dfe", "x", "a1", "--", "\u00e9"]
_SPACES = [" ", "  ", "\t", "\n", " \n ", "\r\n", "\f", "\v"]
_EXOTIC = ["\xa0", "\u2003", "\x1c", "\u0130", "\u2009", "\u3000"]


def _text(rng: random.Random, words: int) -> str:
    out = []
    for _ in range(words):
        word = rng.choice(_WORDS)
        if rng.random() < 0.2:
            word = word.upper() if rng.random() < 0.5 else word.lower()
        out.append(word)
        out.append(rng.choice(_SPACES) if rng.random() < 0.85 else "")
    return "".join(out)


def _case(rng: random.Random, text: str) -> str:
    return "".join(ch.upper() if rng.random() < 0.3 else ch.lower() if rng.random() < 0.3 else ch for ch in text)


def _respace(rng: random.Random, text: str) -> str:
    return "".join(rng.choice(_SPACES) if ch.isspace() else ch for ch in text)


def _cases(seed: int, count: int):
    rng = random.Random(seed)
    for _ in range(count):
        kind = rng.random()
        base = _text(rng, rng.randrange(1, 40))
        if kind < 0.35:  # one text said twice, its second copy re-cased and re-spaced
            text = base + rng.choice(["", " ", "\n\n", "\t"]) + _respace(rng, _case(rng, base))
        elif kind < 0.5:  # said twice, the second copy one character off
            other = list(base + base)
            if other:
                other[rng.randrange(len(other))] = rng.choice("abcXYZ!? ")
            text = "".join(other)
        elif kind < 0.6:  # short texts and whitespace-heavy ones: keys under 8 characters
            text = rng.choice(["ok", "hi", "yes", "no!!", "thanks"]) + " " * rng.randrange(1, 12) + rng.choice(["ok", "thanks!!", "hey you", "x"])
        elif kind < 0.7:  # a character the plain path does not take
            pos = rng.randrange(len(base) + 1)
            text = base[:pos] + rng.choice(_EXOTIC) + base[pos:]
            if rng.random() < 0.5:
                text = text + " " + text
        else:
            text = base
        yield text


def test_the_split_is_the_scans_on_every_kind_of_text() -> None:
    checked = matched = 0
    for text in _cases(1, 40_000):
        stripped = text.strip()
        if _retry_user_key(stripped) is None or len(stripped) < 16:
            continue  # the caller's own precondition
        expected = _first_tandem_split_by_scan(stripped)
        assert _first_tandem_split(stripped) == expected, repr(stripped)
        checked += 1
        matched += expected is not None
    assert checked > 20_000
    assert matched > 5_000  # the repeats are found, not only the misses


def test_the_collapse_is_unchanged_for_whole_messages() -> None:
    for text in _cases(2, 20_000):
        stripped = text.strip()
        key = _retry_user_key(stripped)
        old = None
        if key is not None and len(stripped) >= 16:
            old = _first_tandem_split_by_scan(stripped)
        new = _collapse_repeated_user_text(text)
        if old is not None:
            assert new == old, repr(text)


def test_a_long_context_message_is_linear() -> None:
    rng = random.Random(3)
    message = _text(rng, 600)[:3_900].strip()  # a Claude Code user message: context blocks under 4K
    assert _retry_user_key(message) is not None
    started = time.perf_counter()
    for _ in range(20):
        _first_tandem_split(message)
    per_call = (time.perf_counter() - started) / 20
    started = time.perf_counter()
    scan = _first_tandem_split_by_scan(message)
    scan_s = time.perf_counter() - started
    assert _first_tandem_split(message) == scan
    assert per_call < scan_s / 10, (per_call, scan_s)
