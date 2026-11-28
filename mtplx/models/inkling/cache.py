"""Window state saved at a conversation boundary, including its physical write cursor; and the
rows a prompt read will reach, so its global banks are allocated once."""

import contextlib
import contextvars

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

_PROMPT_ROWS: contextvars.ContextVar = contextvars.ContextVar("inkling_prompt_rows", default=None)

#: 8-bit global KV (``--kv-quant q8``): each row of a global visit's bank is a packed row of
#: ``mx.quantize(group_size=64, bits=8)`` values (``head_dim`` bytes) followed by each group's bf16
#: scale and bias (4 bytes a group): 136 bytes for 128 dimensions, 53% of a bf16 row.
Q8_GROUP = 64


def q8_enabled():
    """Whether this process stores the global visits' KV in packed 8-bit rows."""
    from mtplx.kv_quant import paged_kv_quant_mode_from_env

    return paged_kv_quant_mode_from_env() == "q8"


def settle_kv_quant_for_pack(dtype):
    """Turn ``--kv-quant q8`` off for a pack whose activations are not bf16: packed rows hold bf16
    scales and rebuild bf16 values (``pack_rows``, ``attend_q8``), so an FP16 pack keeps its 16-bit
    banks. Returns the mode this process runs with (``q8`` or ``off``)."""
    if not q8_enabled():
        return "off"
    if dtype == mx.bfloat16:
        return "q8"
    from mtplx.runtime_options import apply_paged_kv_quantization_env

    print(f"[mtplx] inkling: --kv-quant q8 needs a bf16 pack, this one is {dtype}; KV quantization is off",
          flush=True)
    apply_paged_kv_quantization_env("off")
    return "off"


def packed_width(head_dim):
    """Bytes of one packed row of ``head_dim`` dimensions."""
    return int(head_dim) + 4 * (int(head_dim) // Q8_GROUP)


def pack_rows(x):
    """``(..., head_dim)`` bf16 rows as ``(..., packed_width(head_dim))`` uint8 packed rows."""
    if x.dtype != mx.bfloat16:
        raise ValueError(f"packed rows hold bf16 KV, got {x.dtype}")
    w, s, b = mx.quantize(x, group_size=Q8_GROUP, bits=8)
    sb = mx.stack([s, b], axis=-1).reshape(*s.shape[:-1], 2 * s.shape[-1])
    return mx.concatenate([w.view(mx.uint8), sb.view(mx.uint8)], axis=-1)


def unpack_rows(p, dtype=mx.bfloat16):
    """Packed rows back to ``(..., head_dim)`` rows, exactly ``mx.dequantize``'s values."""
    dim = p.shape[-1] * Q8_GROUP // (Q8_GROUP + 4)
    groups = dim // Q8_GROUP
    w = mx.contiguous(p[..., :dim]).view(mx.uint32)
    sb = mx.contiguous(p[..., dim:]).view(mx.bfloat16).reshape(*p.shape[:-1], groups, 2)
    out = mx.dequantize(w, sb[..., 0], sb[..., 1], group_size=Q8_GROUP, bits=8)
    return out if out.dtype == dtype else out.astype(dtype)


def is_global_bank(kv):
    """A global visit's cache: the stock bf16 ``KVCache`` or the 8-bit ``Q8BankCache``."""
    return type(kv) is KVCache or type(kv) is Q8BankCache


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
        if not is_global_bank(kv):
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
        if is_global_bank(kv) and kv.step != BANK_STEP and kv.keys is not None:
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
    # During a chunked prompt read the ring's state is the last window of a window + chunk - 1 row
    # buffer: a checkpoint left lazy pins that whole buffer (``generation._capture_gdn_boundary``
    # materializes a kept one).
    materialize_checkpoints = True

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


class Q8BankCache(KVCache):
    """A global visit's KV in packed 8-bit rows (``pack_rows``): the same bank of whole steps,
    offset and trim as ``KVCache``, ``uint8 (1, heads, capacity, packed_width)`` arrays where the
    stock cache holds bf16 ``(1, heads, capacity, head_dim)``. The reference attention reads every
    row unpacked (``update_and_fetch``); the fast path reads the packed bank in place."""

    def update_and_fetch(self, keys, values):
        prev, rows = self.offset, keys.shape[2]
        width = packed_width(keys.shape[3])
        if self.keys is None or prev + rows > self.keys.shape[2]:
            n_steps = (self.step + rows - 1) // self.step
            shape = (keys.shape[0], keys.shape[1], n_steps * self.step, width)
            new_k, new_v = mx.zeros(shape, mx.uint8), mx.zeros(shape, mx.uint8)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = self.keys[..., :prev, :], self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
        self.offset += rows
        self.keys[..., prev : self.offset, :] = pack_rows(keys)
        self.values[..., prev : self.offset, :] = pack_rows(values)
        return (unpack_rows(self.keys[..., : self.offset, :], keys.dtype),
                unpack_rows(self.values[..., : self.offset, :], values.dtype))

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError("Q8BankCache is already quantized")
