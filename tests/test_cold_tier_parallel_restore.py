"""Parallel SSD restore: the read_into hook decodes exactly what read_bytes did.

The cold tier hands the codec a ``read_tensor`` with ``read_into`` and
``allocate`` attached; the codec then reads all blocks of a tensor into one
buffer (blob_io.read_into_many) and makes one host copy. The decoded arrays
must be bit-identical to the blob-by-blob path, for full and prefix decodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.cache_bank import blob_io
from mtplx.cache_bank.codec import (
    decode_payload,
    decode_payload_prefix,
    encode_payload,
)
from mtplx.cache_state import CacheSnapshot


def _rand(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)


def _stored(tmp_path: Path):
    k = _rand((1, 2, 1100, 64), mx.bfloat16, 1)
    v = _rand((1, 2, 1100, 64), mx.float16, 2)
    gdn = _rand((1, 4, 64, 128), mx.float32, 3)
    snapshot = CacheSnapshot(
        states=((k, v), (gdn,)),
        meta_states=({"offset": 1100}, {"kind": "gdn"}),
    )
    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=mx.arange(32, dtype=mx.float32).reshape(1, 32),
        hidden=_rand((1, 16), mx.bfloat16, 4),
        mtp_history_snapshot=None,
        block_size=256,
    )
    for name, raw in encoded.tensors.items():
        (tmp_path / f"{name}.bin").write_bytes(raw)

    def plain(name: str) -> bytes:
        return (tmp_path / f"{name}.bin").read_bytes()

    def hooked(name: str) -> bytes:
        raise AssertionError(f"{name} read blob by blob despite read_into")

    hooked.read_into = lambda items: blob_io.read_into_many(
        (tmp_path / f"{name}.bin", view) for name, view in items
    )
    hooked.allocate = blob_io.allocate
    return encoded.spec, plain, hooked


def _assert_same_tree(left, right):
    if isinstance(left, mx.array):
        assert isinstance(right, mx.array)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert mx.array_equal(left, right).item()
    elif isinstance(left, CacheSnapshot):
        _assert_same_tree(left.states, right.states)
        _assert_same_tree(left.meta_states, right.meta_states)
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same_tree(a, b)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same_tree(left[key], right[key])
    else:
        assert left == right


def test_full_decode_matches_blob_by_blob(tmp_path):
    spec, plain, hooked = _stored(tmp_path)
    a = decode_payload(spec, plain)
    b = decode_payload(spec, hooked)
    _assert_same_tree(a.cache_snapshot, b.cache_snapshot)
    _assert_same_tree(a.logits, b.logits)
    _assert_same_tree(a.hidden, b.hidden)


def test_prefix_decode_matches_blob_by_blob(tmp_path):
    spec, plain, hooked = _stored(tmp_path)
    a = decode_payload_prefix(spec, plain, cache_prefix_len=600)
    b = decode_payload_prefix(spec, hooked, cache_prefix_len=600)
    _assert_same_tree(a.cache_snapshot, b.cache_snapshot)
    assert b.cache_snapshot.states[0][0].shape[2] == 600
