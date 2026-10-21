"""Token id tuples that are converted once and recognised afterwards."""

from __future__ import annotations

from typing import Iterable


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
