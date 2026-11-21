"""Inkling 2 Mini's conversation is kept once (mtplx/one_copy.py, ``one_copy_store``).

Its decode writes the global banks in place, and a banked snapshot held views of the live
buffers: MLX cannot write a buffer in place while a view shares it, so the first write after
the prompt's publish (and every follow-up turn's first write after the answer's commit)
copied every global visit's whole history, 32 KB a token held twice (Bare Speed through
`mtplx serve`, 21 November: 2.28 GB of a 64K prompt's decode start). These tests pin the
one-copy store on Inkling's caches: a window ring, its convolution tails and a global visit,
the same cache classes the model makes.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.cache_state import snapshot_untrimmable_cache
from mtplx.models.inkling.cache import WindowCache
from mtplx.models.inkling.runtime import InklingRuntime
from mtplx.one_copy import (
    one_copy_runtime,
    one_copy_store,
    one_copy_store_cache,
    prompt_lease_fields,
)
from mtplx.session_bank import SessionBank

WINDOW, HEADS, DIM = 8, 2, 16

RUNTIME = SimpleNamespace(
    model_path="tiny-inkling",
    mtp_enabled=True,
    make_cache=lambda: [WindowCache(max_size=WINDOW), ArraysCache(4), KVCache(), ArraysCache(4)],
    make_mtp_cache=lambda: [],
)


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def _rows(count: int, seed: int) -> mx.array:
    return mx.random.normal((1, HEADS, count, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)


def _tails(seed: int) -> ArraysCache:
    tails = ArraysCache(4)
    for slot in range(4):
        tails[slot] = mx.random.normal((1, 3, 12), key=mx.random.key(seed + slot))
    return tails


def _conversation(tokens: int, *, seed: int = 0) -> list:
    """One window visit and one global visit after a ``tokens``-token prompt read."""

    window, global_ = WindowCache(max_size=WINDOW), KVCache()
    window.update_and_fetch(_rows(tokens, seed), _rows(tokens, seed + 1))
    global_.update_and_fetch(_rows(tokens, seed + 2), _rows(tokens, seed + 3))
    cache = [window, _tails(seed + 10), global_, _tails(seed + 20)]
    mx.eval(*_leaves(cache))
    return cache


def _leaves(cache) -> list[mx.array]:
    out = []
    for entry in cache:
        if isinstance(entry, ArraysCache):
            out += [leaf for leaf in entry.cache if leaf is not None]
        else:
            out += [entry.keys, entry.values]
    return out


def _decode(cache, tokens: int, *, seed: int = 50) -> None:
    """What an answer decoding into the live cache does: one row at a time into the ring and
    the global visit, new convolution tails."""

    for step in range(tokens):
        cache[0].update_and_fetch(_rows(1, seed + 4 * step), _rows(1, seed + 4 * step + 1))
        cache[2].update_and_fetch(_rows(1, seed + 4 * step + 2), _rows(1, seed + 4 * step + 3))
        for tails in (cache[1], cache[3]):
            for slot in range(4):
                tails[slot] = tails[slot] + 1.0
    mx.eval(*_leaves(cache))


def _address(value: mx.array) -> int:
    mx.eval(value)
    view = np.asarray(value.view(mx.uint16), copy=False)
    address = int(view.__array_interface__["data"][0])
    del view
    return address


def _bits(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value.view(mx.uint16) if value.dtype == mx.bfloat16 else value).copy()


def _bank() -> SessionBank:
    return SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)


def _put_answer(bank, cache, tokens, **extra):
    return bank.put(
        runtime=RUNTIME, token_ids=tokens, cache=cache, logits=mx.zeros((1, 16)),
        hidden=mx.ones((1, 1, 8)), keep_live_ref=True, session_id="s",
        snapshot_epoch=len(tokens), **extra,
    )


def test_inkling_caches_and_runtime_are_kept_by_the_one_copy_store(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    assert one_copy_store_cache(RUNTIME.make_cache())
    assert not one_copy_store_cache([KVCache(), ArraysCache(4)])
    assert not one_copy_store_cache([])
    assert one_copy_store(InklingRuntime) and not one_copy_runtime(InklingRuntime)
    assert not one_copy_store(SimpleNamespace())
    monkeypatch.setenv("MTPLX_ONE_COPY", "0")
    assert not one_copy_store_cache(RUNTIME.make_cache())
    assert not one_copy_store(InklingRuntime)


def test_the_answer_is_banked_as_a_lease_and_the_next_write_is_in_place(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    cache = _conversation(20)
    entry = _put_answer(bank, cache, list(range(20)))
    assert entry.live_ref_only and entry.cache_ref is cache
    assert all(state is None for state in entry.cache_snapshot.states)
    assert entry.lease_kv_offset == 20
    before = _address(cache[2].keys)
    _decode(cache, 1)
    # No view shares the global visit's buffer: the row is written where the history is.
    assert _address(cache[2].keys) == before


def test_the_copying_store_behind_the_switch_holds_views_and_copies_at_the_next_write(monkeypatch):
    """What the one-copy store removes, measured on the same caches with it switched off."""

    monkeypatch.setenv("MTPLX_ONE_COPY", "0")
    bank = _bank()
    cache = _conversation(20)
    entry = _put_answer(bank, cache, list(range(20)))
    assert not entry.live_ref_only and entry.cache_snapshot.states[2] is not None
    before = _address(cache[2].keys)
    _decode(cache, 1)
    assert _address(cache[2].keys) != before


@pytest.mark.parametrize("extends", [False, True])
def test_a_prompt_lease_the_answer_ran_past_restores_through_its_anchor(monkeypatch, extends):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    cache = _conversation(20)
    anchor = snapshot_untrimmable_cache(cache)
    want = {i: [_bits(leaf) for leaf in state] for i, state in enumerate(anchor.states) if state is not None}
    prompt = list(range(20))
    fields = prompt_lease_fields(
        cache, committed_mtp_cache=None, hidden=mx.ones((1, 1, 8)), prompt_len=len(prompt), boundaries=[],
    )
    entry = bank.put(
        runtime=RUNTIME, token_ids=prompt, cache=cache, logits=mx.zeros((1, 16)),
        hidden=mx.ones((1, 1, 8)), session_id="s", snapshot_epoch=len(prompt), **fields,
    )
    assert entry.live_ref_only and entry.lease_kv_offset == 20
    _decode(cache, 6)  # the answer decodes into the leased cache and is never committed
    assert cache[2].offset == 26 and cache[0].offset == 26
    lookup = prompt + [900, 901] if extends else prompt
    restored = bank.restore(RUNTIME, lookup, mode="reference", session_id="s")
    assert restored is not None and restored.restore_mode == "reference_lease"
    assert restored.cache is cache and entry.cache_ref is None
    assert cache[2].offset == 20 and cache[0].offset == 20
    for index, bits in want.items():
        state = cache[index].state
        leaves = list(state) if isinstance(state, (list, tuple)) else [state]
        assert all(np.array_equal(_bits(leaf), expect) for leaf, expect in zip(leaves, bits)), index


def test_another_session_gets_a_copy_and_the_owner_keeps_its_lease(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    cache = _conversation(20)
    prompt = list(range(20))
    entry = _put_answer(bank, cache, prompt)
    other = bank.restore(RUNTIME, prompt + [5], mode="reference", session_id="other")
    assert other is not None and other.cache is not cache
    assert entry.cache_ref is cache
    owner = bank.restore(RUNTIME, prompt + [6], mode="reference", session_id="s")
    assert owner is not None and owner.cache is cache and owner.restore_mode == "reference_lease"


def test_the_prompt_anchor_is_the_reads_own_checkpoint_at_the_prompts_end(monkeypatch):
    """The read keeps a checkpoint at the prompt's end (copied and evaluated at capture); the lease's
    anchor is that record rather than a second 180 MB copy beside it for the whole answer."""

    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    cache = _conversation(20)
    hidden = mx.ones((1, 1, 8))
    early = (12, snapshot_untrimmable_cache(cache), None)
    at_end = (20, snapshot_untrimmable_cache(cache), None)
    inkling = SimpleNamespace(one_copy_store=True)  # opted in; not the fixed-M4 verifier's runtime
    fields = prompt_lease_fields(cache, committed_mtp_cache=None, hidden=hidden, prompt_len=20,
                                 boundaries=[early, at_end], runtime=inkling)
    records = fields["gdn_boundaries"]
    assert [int(r[0]) for r in records] == [12, 20]
    assert records[-1][1] is at_end[1] and records[-1][2] is hidden
    # Without one at the end, the anchor is copied from the cache as before.
    fresh = prompt_lease_fields(cache, committed_mtp_cache=None, hidden=hidden, prompt_len=20,
                                boundaries=[early], runtime=inkling)
    anchor = fresh["gdn_boundaries"][-1][1]
    assert anchor is not at_end[1] and anchor.states[2] is None and anchor.states[0] is not None
    # The fixed-M4 verifier's runtime keeps its own copy whatever the read kept: its path is the base's.
    verifier = SimpleNamespace(qwen4_fixed_m4_compiled_verify=True)
    assert one_copy_runtime(verifier)
    own = prompt_lease_fields(cache, committed_mtp_cache=None, hidden=hidden, prompt_len=20,
                              boundaries=[early, at_end], runtime=verifier)
    assert [int(r[0]) for r in own["gdn_boundaries"]] == [12, 20]
    assert own["gdn_boundaries"][-1][1] is not at_end[1]
