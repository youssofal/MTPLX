"""Decode state of Inkling 2 Mini as rings and banks addressed by one absolute position.

Every per-visit state is append-only and addressed by the absolute position of the token
it belongs to, so a verify panel writes ahead of the committed position and accepting n
rows is ``pos += n``. Nothing is copied, reordered or rolled back.

Per layer visit (48 a token):

* window visit (20 layers, window 1,024): ``keys``/``values`` of shape
  ``(1, kv_heads, RING, head_dim)`` in the activation dtype, ``RING = window + MAX_PANEL``.
  Position ``p`` lives in slot ``p % RING``.
* global visit (layers 5, 11, 17, 23): ``keys``/``values`` of shape
  ``(1, kv_heads, cap, head_dim)``; position ``p`` lives in slot ``p``. ``cap`` is shared by
  every global visit and grows in large steps (``grow``), never per token.
* four convolution tap rings ``(1, TAPS, width)`` float32 (key, value, attention output,
  expert output), ``TAPS = taps - 1 + MAX_PANEL``; the input of position ``p`` lives in
  slot ``p % TAPS``. Positions before the first token read as zeros.

``pos`` is a uint32 scalar array, the committed token count. ``length`` mirrors it on the
host for growth and conversion; only ``commit`` changes either.

The conversions to and from the reference cache objects (``WindowCache``, ``KVCache``,
``ArraysCache``) are exact, so prefill, the session bank, SSD restore and compaction keep
working on the reference layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from ..cache import WindowCache

#: Rows one panel may write ahead of the committed position.
MAX_PANEL = 32
#: Global banks grow in steps of this many slots.
BANK_STEP = 2048


def round_up(value, step):
    return -(-value // step) * step


def _int(pos):
    return pos.astype(mx.int32)


def slots(pos, count, size, offset=0):
    """Ring slots of positions ``pos + offset .. pos + offset + count - 1``, int32 ``(count,)``.

    ``pos`` is the uint32 position array (or an int). Negative positions wrap like the
    ring does (``-1`` is the last slot), which is where the zero history before the first
    token lives.
    """
    base = _int(pos) if isinstance(pos, mx.array) else mx.array(pos, mx.int32)
    return (base + (offset + size) + mx.arange(count, dtype=mx.int32)) % size


@dataclass
class FastState:
    """The decode state of one stream. Lists are indexed by layer visit."""

    pos: mx.array
    length: int
    keys: list
    values: list
    taps: list
    local: tuple
    window: int
    ring: int
    tap_slots: int
    cap: int

    # ------------------------------------------------------------------ allocation
    @classmethod
    def allocate(cls, model, length=0, cap=None, dtype=None):
        cfg = model.config
        dtype = dtype or model.embed_tokens.weight.dtype
        heads, width = cfg["num_key_value_heads"], cfg["head_dim"]
        window = cfg["sliding_window_size"]
        size = cfg["sconv_kernel_size"]
        local = tuple(model.layers[i].sliding for i in model.visit_order)
        ring = window + MAX_PANEL
        tap_slots = size - 1 + MAX_PANEL
        cap = round_up(max(cap or 0, length + MAX_PANEL, 1), BANK_STEP)
        keys, values, taps = [], [], []
        for visit, index in enumerate(model.visit_order):
            slots_ = ring if local[visit] else cap
            keys.append(mx.zeros((1, heads, slots_, width), dtype))
            values.append(mx.zeros((1, heads, slots_, width), dtype))
            taps.append([mx.zeros((1, tap_slots, conv.weight.shape[1]), mx.float32)
                         for conv in model.layers[index].convs])
        return cls(mx.array(length, mx.uint32), length, keys, values, taps, local,
                   window, ring, tap_slots, cap)

    def grow(self, rows):
        """Make room for a panel of ``rows`` (one allocation per BANK_STEP tokens)."""
        if rows > MAX_PANEL:
            raise ValueError(f"A panel holds at most {MAX_PANEL} rows")
        need = self.length + MAX_PANEL
        if need <= self.cap:
            return False
        cap = round_up(need, BANK_STEP)
        for visit, local in enumerate(self.local):
            if local:
                continue
            for bank in (self.keys, self.values):
                old = bank[visit]
                new = mx.zeros((*old.shape[:2], cap, old.shape[3]), old.dtype)
                new[:, :, : self.length, :] = old[:, :, : self.length, :]
                bank[visit] = new
        self.cap = cap
        return True

    def commit(self, rows):
        """Accept the first ``rows`` rows of the last panel: the only state change."""
        if not 0 <= rows <= MAX_PANEL:
            raise ValueError("commit takes 0 to MAX_PANEL rows")
        self.length += rows
        self.pos = mx.array(self.length, mx.uint32)

    def arrays(self):
        """Every array of the state, for mx.eval."""
        return [self.pos, *self.keys, *self.values, *(t for visit in self.taps for t in visit)]

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays())

    # ------------------------------------------------------------------ reads
    def window_positions(self, rows):
        """First position and count of the keys a reference panel of ``rows`` attends to.

        This is the chronological key array the reference window cache hands to attention:
        the last ``window - 1`` committed positions (fewer near the start) and the panel.
        """
        first = max(0, self.length - (self.window - 1))
        return first, self.length - first + rows

    def history_keys(self, visit, rows):
        """Keys and values in chronological order, as the reference cache returns them.

        Window visits gather from the ring (a copy, as the reference reorder is); global
        visits return a view of the bank. Host-side shape: the reference attention only.
        """
        if self.local[visit]:
            first, count = self.window_positions(rows)
            index = slots(first, count, self.ring)
            return (mx.take(self.keys[visit], index, axis=2),
                    mx.take(self.values[visit], index, axis=2))
        end = self.length + rows
        return self.keys[visit][:, :, :end, :], self.values[visit][:, :, :end, :]

    def tap_history(self, visit, slot, pos=None):
        """The float32 inputs of the three positions before ``pos``: ``(1, taps - 1, width)``."""
        ring = self.taps[visit][slot]
        keep = self.tap_slots - MAX_PANEL
        return mx.take(ring, slots(self.pos if pos is None else pos, keep, self.tap_slots, -keep), axis=1)

    # ------------------------------------------------------------------ writes
    def write_kv(self, visit, keys, values, pos=None):
        """Write a panel's key/value rows ``(1, kv_heads, rows, head_dim)`` at ``pos ..``."""
        pos = self.pos if pos is None else pos
        rows = keys.shape[2]
        if self.local[visit]:
            index = slots(pos, rows, self.ring)
            self.keys[visit][:, :, index, :] = keys
            self.values[visit][:, :, index, :] = values
        else:
            start = _int(pos).reshape(1)
            self.keys[visit] = mx.slice_update(self.keys[visit], keys, start, (2,))
            self.values[visit] = mx.slice_update(self.values[visit], values, start, (2,))

    def write_taps(self, visit, slot, rows, pos=None):
        """Write a panel's convolution inputs ``(1, rows, width)`` as float32 at ``pos ..``."""
        pos = self.pos if pos is None else pos
        index = slots(pos, rows.shape[1], self.tap_slots)
        self.taps[visit][slot][:, index, :] = rows.astype(mx.float32)

    # ------------------------------------------------------------------ conversion
    @classmethod
    def from_reference(cls, model, cache, cap=None):
        """Exact copy of the reference trunk cache (2 entries a visit) into the fast layout."""
        visits = len(model.visit_order)
        if len(cache) < 2 * visits:
            raise ValueError("Inkling needs attention and convolution state for every visit")
        lengths = {int(cache[2 * v].offset) for v in range(visits)}
        if len(lengths) != 1:
            raise ValueError(f"Visits disagree on the committed length: {sorted(lengths)}")
        length = lengths.pop()
        state = cls.allocate(model, length=length, cap=cap)
        for visit in range(visits):
            kv, conv = cache[2 * visit], cache[2 * visit + 1]
            if kv.keys is not None and length:
                if state.local[visit]:
                    keys, values = kv._temporal_order(kv.keys), kv._temporal_order(kv.values)
                    count = min(keys.shape[2], state.window)
                    keys, values = keys[:, :, -count:, :], values[:, :, -count:, :]
                    index = slots(length - count, count, state.ring)
                    state.keys[visit][:, :, index, :] = keys
                    state.values[visit][:, :, index, :] = values
                else:
                    state.keys[visit][:, :, :length, :] = kv.keys[:, :, :length, :]
                    state.values[visit][:, :, :length, :] = kv.values[:, :, :length, :]
            for slot in range(len(state.taps[visit])):
                tail = conv[slot]
                if tail is None:
                    continue
                keep = tail.shape[1]
                index = slots(length - keep, keep, state.tap_slots)
                state.taps[visit][slot][:, index, :] = tail.astype(mx.float32)
        return state

    def to_reference(self, model):
        """The reference trunk cache this state stands for (fresh objects, exact contents)."""
        cache = []
        length = self.length
        for visit in range(len(self.local)):
            if self.local[visit]:
                kv = WindowCache(max_size=self.window)
                count = min(length, self.window)
                if count:
                    index = slots(length - count, count, self.ring)
                    kv.keys = mx.take(self.keys[visit], index, axis=2)
                    kv.values = mx.take(self.values[visit], index, axis=2)
                    kv._idx = count
                kv.offset = length
            else:
                kv = KVCache()
                if length:
                    kv.keys = mx.contiguous(self.keys[visit][:, :, :length, :])
                    kv.values = mx.contiguous(self.values[visit][:, :, :length, :])
                kv.offset = length
            conv = ArraysCache(len(self.taps[visit]))
            if length:
                for slot in range(len(self.taps[visit])):
                    conv[slot] = self.tap_history(visit, slot, mx.array(length, mx.uint32))
            cache.extend((kv, conv))
        return cache
