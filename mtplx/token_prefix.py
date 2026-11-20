"""Token-prefix comparison shared by the session bank, the engine session, the cold tier and the server.

Standard library only: ``mtplx.cache_bank`` must stay importable without
mlx, and it compares prefixes too.
"""

from __future__ import annotations

from collections.abc import Sequence

# Tokens compared per slice. Slice equality runs in C, so an equal chunk
# costs far less than comparing its tokens one by one in Python; the chunk
# holding the first mismatch is then scanned token by token, so a smaller
# chunk keeps that scan short. 64 measured best for 300-8000 token prompts
# whose shared prefix ends anywhere from token 0 to the full length.
_PREFIX_COMPARE_CHUNK = 64


def common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    """Number of leading tokens ``left`` and ``right`` have in common.

    Accepts lists and tuples (also mixed). Compares chunk by chunk so long
    equal prefixes are checked at C speed, then locates the first mismatch
    inside the differing chunk.
    """
    limit = min(len(left), len(right))
    for start in range(0, limit, _PREFIX_COMPARE_CHUNK):
        stop = min(start + _PREFIX_COMPARE_CHUNK, limit)
        if tuple(left[start:stop]) == tuple(right[start:stop]):
            continue
        for index in range(start, stop):
            if left[index] != right[index]:
                return index
    return limit


def common_run_len(
    left: Sequence[int], left_start: int, right: Sequence[int], right_start: int
) -> int:
    """Number of equal tokens ``left`` and ``right`` hold from those offsets
    on: ``common_prefix_len(left[left_start:], right[right_start:])`` without
    copying either remainder."""
    limit = max(0, min(len(left) - left_start, len(right) - right_start))
    for offset in range(0, limit, _PREFIX_COMPARE_CHUNK):
        stop = min(offset + _PREFIX_COMPARE_CHUNK, limit)
        if tuple(left[left_start + offset : left_start + stop]) == tuple(
            right[right_start + offset : right_start + stop]
        ):
            continue
        for index in range(offset, stop):
            if left[left_start + index] != right[right_start + index]:
                return index
    return limit
