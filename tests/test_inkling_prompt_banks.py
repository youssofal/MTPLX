"""Inkling 2 Mini's prompt read allocates each global visit's bank once (cache.size_global_banks).

The stock cache grew a global visit by copying its whole history into a buffer one chunk longer
at every 512-row chunk: about 132 GB of copies over a 64K read, the old buffer held beside the
new each time, and a buffer the fast path then copied again into whole 4,096-row banks at the
decode start. Sized to the prompt's rows in whole bank steps, the read writes every chunk in
place and the decode start finds its banks ready. The rows the attention reads do not change.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.models.inkling.cache import WindowCache, prompt_rows, settle_bank_steps, size_global_banks
from mtplx.models.inkling.fast.state import BANK_STEP

HEADS, DIM, CHUNK = 2, 16, 512


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def _chunk(start: int, rows: int):
    keys = mx.random.normal((1, HEADS, rows, DIM), key=mx.random.key(start)).astype(mx.bfloat16)
    values = mx.random.normal((1, HEADS, rows, DIM), key=mx.random.key(start + 1)).astype(mx.bfloat16)
    return keys, values


def _address(value: mx.array) -> int:
    mx.eval(value)
    view = np.asarray(value.view(mx.uint16), copy=False)
    address = int(view.__array_interface__["data"][0])
    del view
    return address


def _read(cache: KVCache, tokens: int, *, start: int = 0, size: bool = True):
    """A chunked prompt read of ``tokens`` rows into ``cache``; the addresses of its buffer."""

    addresses, done = [], start
    while done < start + tokens:
        rows = min(CHUNK, start + tokens - done)
        if size:
            size_global_banks([cache], rows)
        k, v = _chunk(done, rows)
        out_k, out_v = cache.update_and_fetch(k, v)
        mx.eval(out_k, out_v)
        if size:
            settle_bank_steps([cache])
        addresses.append(_address(cache.keys))
        done += rows
    return addresses


def _bits(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value.view(mx.uint16)).copy()


def test_a_cold_read_allocates_the_prompts_bank_once_and_writes_every_chunk_in_place():
    sized, stock = KVCache(), KVCache()
    with prompt_rows(10_000):
        addresses = _read(sized, 10_000)
    _read(stock, 10_000, size=False)
    assert sized.keys.shape[2] == 12_288 == -(-10_000 // BANK_STEP) * BANK_STEP
    assert len(set(addresses)) == 1  # one allocation, every chunk written into it
    assert sized.step == BANK_STEP
    # The rows the attention reads are the stock cache's rows, bit for bit.
    assert sized.offset == stock.offset == 10_000
    assert np.array_equal(_bits(sized.keys[:, :, :10_000]), _bits(stock.keys[:, :, :10_000]))
    assert np.array_equal(_bits(sized.values[:, :, :10_000]), _bits(stock.values[:, :, :10_000]))


def test_a_warm_read_copies_the_restored_history_once_into_the_prompts_bank():
    cache = KVCache()
    _read(cache, 3_000, size=False)  # a restored prefix, at the stock layout
    before = _bits(cache.keys[:, :, :3_000])
    with prompt_rows(9_000):
        addresses = _read(cache, 6_000, start=3_000)
    assert cache.keys.shape[2] == 12_288 and len(set(addresses)) == 1
    assert np.array_equal(_bits(cache.keys[:, :, :3_000]), before)


def test_without_a_prompt_the_bank_grows_in_whole_steps_and_other_caches_are_left_alone():
    cache, window, tails = KVCache(), WindowCache(max_size=8), ArraysCache(4)
    size_global_banks([cache, window, tails], 100)
    k, v = _chunk(0, 100)
    cache.update_and_fetch(k, v)
    settle_bank_steps([cache])
    assert cache.keys.shape[2] == BANK_STEP and cache.step == BANK_STEP
    assert window.keys is None and tails[0] is None
    assert size_global_banks([cache], BANK_STEP - 100) == 0  # room left: nothing grows
    assert size_global_banks([cache], BANK_STEP - 99) == 1 and cache.keys.shape[2] == 2 * BANK_STEP
