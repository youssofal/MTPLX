"""Window state saved at a conversation boundary, including its physical write cursor; and the
rows a prompt read will reach, so its global banks are allocated once."""

import contextlib
import contextvars

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

_PROMPT_ROWS: contextvars.ContextVar = contextvars.ContextVar("inkling_prompt_rows", default=None)


@contextlib.contextmanager
def prompt_rows(rows):
    """The prompt read inside this block reaches ``rows`` tokens (``size_global_banks``)."""
    token = _PROMPT_ROWS.set(int(rows) if rows else None)
    try:
        yield
    finally:
        _PROMPT_ROWS.reset(token)


def _round_up(value, step):
    return -(-int(value) // step) * step


def size_global_banks(caches, rows):
    """Room for a forward of ``rows`` rows in every global visit's cache, allocated once.

    The stock cache grows a global visit by copying its whole history into a buffer a chunk
    longer, at every chunk of a prompt read: a 512-row chunk at history H copied 8 visits x H x
    4 KB, about 132 GB over a 64K read and 0.5 TB over a 128K one, and held the old buffer beside
    the new one while it did. A cache that would overflow is grown here instead, straight to the
    rows the read reaches (``prompt_rows``; else this forward's end) in whole ``BANK_STEP``
    steps: an empty one by allocating that capacity on its first write, one that holds rows by
    one copy into it, evaluated a visit at a time so one old buffer at most lives beside the new.
    The rows the attention reads are the same rows (class A); the fast path's ``prepare`` then
    finds its banks already in whole capacity steps and copies nothing at the decode start.
    Returns how many caches it grew."""
    from .fast.state import BANK_STEP

    target = _PROMPT_ROWS.get()
    grown = 0
    for kv in caches:
        if type(kv) is not KVCache:
            continue
        need = int(kv.offset) + int(rows)
        held = 0 if kv.keys is None else int(kv.keys.shape[2])
        if need <= held:
            continue
        capacity = _round_up(max(need, target or 0), BANK_STEP)
        if kv.keys is None:
            # The stock cache allocates ``step`` rows (a whole number of them) on its first write.
            kv.step = capacity
        else:
            k_shape = (*kv.keys.shape[:2], capacity, kv.keys.shape[3])
            v_shape = (*kv.values.shape[:2], capacity, kv.values.shape[3])
            keys, values = mx.zeros(k_shape, kv.keys.dtype), mx.zeros(v_shape, kv.values.dtype)
            keys[:, :, : kv.offset, :] = kv.keys[:, :, : kv.offset, :]
            values[:, :, : kv.offset, :] = kv.values[:, :, : kv.offset, :]
            mx.eval(keys, values)
            kv.keys, kv.values = keys, values
            kv.step = BANK_STEP
        grown += 1
    return grown


def settle_bank_steps(caches):
    """After the forward that allocated: later growth goes in ``BANK_STEP`` rows again."""
    from .fast.state import BANK_STEP

    for kv in caches:
        if type(kv) is KVCache and kv.step != BANK_STEP and kv.keys is not None:
            kv.step = BANK_STEP


class WindowCache(RotatingKVCache):
    # The same number of rows can describe different absolute positions and ring orders.
    requires_restore_meta = True
    # The session bank keeps a conversation of these caches once (``mtplx/one_copy.py``): a
    # lease on the live cache with its window rings and convolution tails as the anchor, never
    # views of the live buffers. Decode writes the global banks in place, and a view held by a
    # banked snapshot made that write copy every global visit's whole history: 32 KB a token
    # held twice at the decode start of a prompt and through every follow-up turn.
    one_copy_store = True

    def is_trimmable(self):
        # A short prefix is still present now, but may be overwritten by the time it is restored.
        return False

    @RotatingKVCache.state.getter
    def state(self):
        if self.keys is not None and self.keys.shape[2] > self.max_size:
            # Multi-token attention used these extra rows for its earliest queries. Only
            # the last window can be read by the first query after this checkpoint.
            return self.keys[:, :, -self.max_size :], self.values[
                :, :, -self.max_size :
            ]
        return super().state

    @RotatingKVCache.meta_state.getter
    def meta_state(self):
        return tuple(
            map(
                str,
                (self.keep, self.max_size, self.offset, min(self._idx, self.max_size)),
            )
        )
