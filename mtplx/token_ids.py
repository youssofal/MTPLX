"""Token id tuples that are converted once and recognised afterwards."""

from __future__ import annotations

import sys
from array import array
from typing import Any, Iterable


class TokenTuple(tuple):
    """A tuple whose elements are all plain ``int``.

    Equal to, hashing as, and iterating like the plain tuple. It exists so
    that code receiving a request's prompt ids can tell "already converted"
    in O(1) instead of re-running ``tuple(int(t) for t in ids)`` over
    80K+ tokens at every layer.
    """

    __slots__ = ()


def as_token_tuple(token_ids: Iterable[int]) -> TokenTuple:
    """Return ``token_ids`` as a tuple of ints, converting only when needed."""
    if type(token_ids) is TokenTuple:
        return token_ids
    return TokenTuple([int(token) for token in token_ids])


def token_id_bytes(token_ids: Any) -> bytes:
    """Token ids as the bytes MTPLX hashes them in (the session's boundary
    hashes, the bank's and the SSD tier's token and block hashes): each id
    as 8 little-endian signed bytes, ``int(id).to_bytes(8, "little",
    signed=True)``, built in C. A per-token Python loop of ``to_bytes`` and
    ``update`` cost 16 ms at 156K tokens and ran several times per turn. An
    id outside int64 raises OverflowError, as ``to_bytes`` did."""

    try:
        data = array("q", token_ids)
    except TypeError:
        data = array("q", [int(token) for token in token_ids])
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()
