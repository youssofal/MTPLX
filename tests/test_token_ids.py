"""TokenTuple: convert a prompt's ids once, recognise them afterwards."""

from __future__ import annotations

import pytest

from mtplx.token_ids import TokenTuple, as_token_tuple


def test_token_tuple_fast_path():
    plain = tuple(range(1000))
    first = as_token_tuple(plain)
    assert type(first) is TokenTuple and first == plain and hash(first) == hash(plain)
    assert as_token_tuple(first) is first
    assert as_token_tuple([1, 2, 3]) == (1, 2, 3)
    assert type(as_token_tuple(iter([1, 2]))) is TokenTuple
    np = pytest.importorskip("numpy")
    converted = as_token_tuple(np.arange(5))
    assert all(type(t) is int for t in converted)
