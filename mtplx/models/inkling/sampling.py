"""The model's native full-vocabulary categorical draw (the reference's FP64 arithmetic)."""

import os
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from mtplx.sampling import client_controls, non_finite_logits_error


def is_native_sampler(config):
    return (
        config.temperature == 1.0
        and config.top_p == 1.0
        and not config.top_k
        and not config.presence_penalty
        and not config.frequency_penalty
        and not client_controls(config)
    )


def _accumulated(row):
    """One float32 row's categorical masses in FP64, accumulated; None when the row has no
    finite positive total (the caller raises the product's error before drawing)."""
    mass = row.astype(np.float64)
    mass -= mass.max()
    np.exp(mass, out=mass)
    total = mass.sum()
    if not np.isfinite(total) or total <= 0:
        return None

    # Keep the reference's softmax, unfiltered support and categorical
    # normalizations in the same order, including their FP64 rounding.
    mass /= total
    mass /= mass.sum()
    mass /= mass.sum()
    # Into a second buffer: the same sums as in place, and numpy runs an in-place
    # accumulation with the interpreter lock held, so rows on threads would wait on it.
    return np.cumsum(mass)


def _pick(accumulated, rng):
    total = accumulated[-1]
    return bisect_right(accumulated, rng.random(), key=lambda value: value / total)


def draw_native(logits, rng):
    row = np.asarray(logits.astype(mx.float32)).reshape(-1)
    accumulated = _accumulated(row)
    if accumulated is None:
        raise non_finite_logits_error(row, "softmax")
    return _pick(accumulated, rng)


# Worker threads for a panel's rows: half the cores, at most 8 (numpy releases the interpreter
# lock inside each row's passes, so rows run side by side).
_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))
_pool = None


def _workers():
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="inkling-draw")
    return _pool


def _accumulated_as(errors, row):
    # numpy's floating-point error settings are per thread: the caller's apply to its rows.
    with np.errstate(**errors):
        return _accumulated(row)


class NativeDrawsAhead:
    """``draw_native`` over the rows of one verified panel, walked in order, each row's
    accumulated masses computed ahead on worker threads while the walk draws.

    The same FP64 operations on the same float32 values, so the same masses bit for bit; the
    generator still draws once per row walked, in order, on the caller's thread. A row the walk
    never reaches is never drawn from (``close`` drops the rows not started), and a non-finite
    row raises the product's error when the walk reaches it, before its draw. A walk of 16 rows
    is about 3 ms of host time instead of about 16 (one row is about 1 ms; M5 Max, CPU bench).
    """

    def __init__(self, rows):
        # One read of the panel (the rows are already evaluated); the walk's own row reads
        # gave these values one row at a time.
        self._host = np.asarray(rows.astype(mx.float32))
        pool, errors = _workers(), np.geterr()
        # The first row on the caller's thread: a walk that stops there costs what it did.
        self._rows = [None] + [pool.submit(_accumulated_as, errors, row) for row in self._host[1:]]

    def draw(self, pos, rng):
        accumulated = _accumulated(self._host[0]) if pos == 0 else self._rows[pos].result()
        if accumulated is None:
            raise non_finite_logits_error(self._host[pos], "softmax")
        return _pick(accumulated, rng)

    def close(self):
        for row in self._rows[1:]:
            row.cancel()
