"""A session commit records its 512-token boundaries in one pass, and every
token hash is built in C (12 Nov).

Every commit adds a boundary each 512 tokens of the committed stream with the
hash of the prefix it ends. Each prefix used to be hashed from its first token
by a Python loop: after a 42,626-token answer at 22K context the commit hashed
83 prefixes of 22K to 64K tokens (0.32 s), on the worker that holds the
response tail the next Pi tool round waits on (window n4a7: 0.69 s). The
hashes and the boundaries are unchanged; they are now computed in C, the
prefixes extending one running hash. The bank's and the SSD tier's token and
block hashes (persisted in the SSD manifest) took the same per-token loop on
every put; they are the same values, built from the same bytes in C.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from mtplx.cache_bank.cold_tier import chain_block_hashes, token_hash
from mtplx.engine_session import EngineSession, token_hash_short
from mtplx.session_bank import token_prefix_hash
from mtplx.token_ids import as_token_tuple


def _reference_hash(token_ids) -> str:
    """The hash as it was computed until 12 Nov."""
    h = hashlib.sha256()
    for token in token_ids:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()[:16]


def _reference_full(token_ids) -> str:
    h = hashlib.sha256()
    for token in token_ids:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def _reference_blocks(token_ids, *, block_size: int, identity: dict) -> list[str]:
    parent, out = "", []
    for start in range(0, len(token_ids), block_size):
        h = hashlib.sha256()
        h.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(parent.encode("utf-8"))
        for token in token_ids[start : start + block_size]:
            h.update(int(token).to_bytes(8, byteorder="little", signed=True))
        parent = h.hexdigest()
        out.append(parent)
    return out


def test_the_bank_and_ssd_hashes_are_the_ones_on_disk():
    rng = random.Random(11)
    identity = {"model": "m", "mtp": True, "template": "t"}
    for length in (0, 1, 255, 256, 257, 5_000):
        ids = [rng.randrange(0, 248_100) for _ in range(length)]
        for container in (ids, tuple(ids), as_token_tuple(ids)):
            assert token_prefix_hash(container) == _reference_full(ids)
            assert token_hash(container) == _reference_full(ids)
            assert chain_block_hashes(container, block_size=256, identity=identity) == (
                _reference_blocks(ids, block_size=256, identity=identity)
            )
    assert chain_block_hashes([], block_size=256) == []


def _reference_intervals(boundaries: list, token_ids, every: int = 512) -> list:
    existing = {token_len for _kind, token_len, _hash in boundaries}
    out = list(boundaries)
    for token_len in range(every, len(token_ids), every):
        if token_len not in existing:
            out.append(("interval_512", token_len, _reference_hash(token_ids[:token_len])))
    return out


def test_the_hash_is_the_one_it_always_was():
    rng = random.Random(20261112)
    for length in (0, 1, 7, 512, 513, 4096, 20_000):
        ids = [rng.randrange(0, 248_100) for _ in range(length)]
        for container in (ids, tuple(ids)):
            assert token_hash_short(container) == _reference_hash(ids)
    # Negative and int64-edge ids, as int.to_bytes(8, signed=True) took them.
    edge = [-1, 0, 2**63 - 1, -(2**63)]
    assert token_hash_short(edge) == _reference_hash(edge)
    with pytest.raises(OverflowError):
        token_hash_short([2**63])


def _session() -> EngineSession:
    return EngineSession("s1")


def _view(session: EngineSession) -> list:
    return [(b.kind, b.token_len, b.token_hash) for b in session.boundaries]


def test_a_commit_records_the_same_boundaries_in_the_same_order():
    rng = random.Random(7)
    prompt = [rng.randrange(0, 248_100) for _ in range(22_057)]
    answer = [rng.randrange(0, 248_100) for _ in range(42_626)]
    session = _session()
    expected: list = []
    # The request's prompt prefix, then the answer, then a tool round's
    # prompt that extends both: each adds only the boundaries it lacks.
    session.commit_prompt_prefix(prompt_ids=prompt, finish_reason="stop")
    expected = _reference_intervals(expected, prompt)
    expected.append(("prompt_prefix", len(prompt), _reference_hash(prompt)))
    assert _view(session) == expected
    session.commit(prompt_ids=prompt, generated_ids=answer, finish_reason="tool_calls")
    full = prompt + answer
    expected = _reference_intervals(expected, full)
    expected.append(("assistant_end", len(full), _reference_hash(full)))
    assert _view(session) == expected
    follow = full + [rng.randrange(0, 248_100) for _ in range(1_300)]
    session.commit_prompt_prefix(prompt_ids=follow, finish_reason="stop")
    expected = _reference_intervals(expected, follow)
    expected.append(("prompt_prefix", len(follow), _reference_hash(follow)))
    assert _view(session) == expected
    # Epochs still count the boundaries in the order they were added.
    assert [b.snapshot_epoch for b in session.boundaries] == list(range(len(expected)))


def test_a_rewritten_history_keeps_the_boundaries_it_had():
    """A commit that does not extend the last one (a retokenized turn) adds
    the lengths it lacks, hashed over its own tokens, as before."""

    session = _session()
    first = list(range(3_000))
    session.commit(prompt_ids=first[:1_000], generated_ids=first[1_000:], finish_reason="stop")
    rewritten = list(range(100_000, 104_000))
    session.commit(prompt_ids=rewritten[:2_000], generated_ids=rewritten[2_000:], finish_reason="stop")
    expected = _reference_intervals([], first)
    expected.append(("assistant_end", len(first), _reference_hash(first)))
    expected = _reference_intervals(expected, rewritten)
    expected.append(("assistant_end", len(rewritten), _reference_hash(rewritten)))
    assert _view(session) == expected
