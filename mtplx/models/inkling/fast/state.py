"""Decode state of Inkling 2 Mini, read in place from the reference cache objects.

The fast step reads RC6's own trunk cache (per layer visit a ``WindowCache`` or
``KVCache`` and an ``ArraysCache`` of four convolution tails) as read-only inputs and
returns the panel's own rows (``VisitDelta``). Nothing writes state during a round. After
the accept decision ``commit`` writes the accepted rows into those same objects with MLX
slice updates, which are in place while the decode path owns the arrays exclusively
(``summary/l0/receipts/bank-append-1.json``). Rejected rows never touch state.

Layout the kernels see (``SIGNATURES.md`` section 2), established by ``prepare``:

* window visit: keys/values ``(1, kv_heads, window, head_dim)`` in the ring's physical
  order; the row of age ``a`` (0 = newest) sits at slot ``(newest - a) % window``;
  ``count = min(pos, window)`` rows are valid. A reference ``WindowCache`` stays valid
  throughout (``_idx`` is the slot after the newest row, ``offset`` the position).
* global visit: keys/values ``(1, kv_heads, cap, head_dim)``, rows ``0 .. pos-1`` valid;
  ``cap`` grows in ``BANK_STEP`` buckets at commit, never per token.
* convolution tails ``(1, taps - 1, width)`` float32, oldest first, zeros before the
  first token, exactly as the reference keeps them.

So prefill, the session bank, SSD restore and compaction keep their existing layout and
code: there is no conversion at the edges, only ``prepare`` after anything that
rewrote a window cache by concatenation (a prefill chunk).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

#: Rows one panel may hold.
MAX_PANEL = 32
#: Global banks grow in steps of this many rows (128 MB of slack at most, 8 global visits).
BANK_STEP = 4096


def round_up(value, step):
    return -(-value // step) * step


@dataclass
class VisitDelta:
    """The panel's own rows of one visit, committed (first ``kept``) after the accept."""

    keys: mx.array  # (1, kv_heads, W, head_dim), the rows the reference would append
    values: mx.array
    taps: tuple  # four (1, W, width) float32: key, value, attention and MLP conv inputs

    def arrays(self):
        return [self.keys, self.values, *self.taps]


class DecodeState:
    """The fast path's view of one stream's trunk cache (2 entries a visit)."""

    def __init__(self, model, cache):
        self.model = model
        self.cache = cache
        cfg = model.config
        self.window = cfg["sliding_window_size"]
        self.keep = cfg["sconv_kernel_size"] - 1
        self.local = tuple(model.layers[i].sliding for i in model.visit_order)
        self.widths = tuple(
            tuple(conv.weight.shape[1] for conv in model.layers[i].convs) for i in model.visit_order
        )
        self.heads, self.head_dim = cfg["num_key_value_heads"], cfg["head_dim"]
        self.dtype = model.embed_tokens.weight.dtype
        if len(cache) < 2 * len(self.local):
            raise ValueError("Inkling needs attention and convolution state for every visit")
        self.length = self._length()
        self.pos = mx.array(self.length, mx.uint32)

    def _length(self):
        lengths = {int(self.cache[2 * v].offset) for v in range(len(self.local))}
        if len(lengths) != 1:
            raise ValueError(f"Visits disagree on the committed length: {sorted(lengths)}")
        return lengths.pop()

    # ------------------------------------------------------------------ prepare
    def prepare(self):
        """Put every visit in the fixed layout. Cheap when it already is (every step)."""
        self.length = self._length()
        self.pos = mx.array(self.length, mx.uint32)
        changed = []
        for visit, local in enumerate(self.local):
            kv, conv = self.cache[2 * visit], self.cache[2 * visit + 1]
            if local and not self._is_ring(kv):
                self._make_ring(kv)
                changed.append(visit)
            if not local and kv.keys is None:
                shape = (1, self.heads, BANK_STEP, self.head_dim)
                kv.keys, kv.values = mx.zeros(shape, self.dtype), mx.zeros(shape, self.dtype)
                changed.append(visit)
            for slot, width in enumerate(self.widths[visit]):
                if conv[slot] is None:
                    conv[slot] = mx.zeros((1, self.keep, width), mx.float32)
        return changed

    def _is_ring(self, kv):
        if kv.keys is None or kv.keys.shape[2] != self.window or kv.keep:
            return False
        if kv.offset >= self.window:
            return 1 <= kv._idx <= self.window
        return kv._idx == kv.offset

    def _make_ring(self, kv):
        """The same chronological rows as a ``window``-row ring (one copy, after prefill)."""
        shape = (1, self.heads, self.window, self.head_dim)
        if kv.keys is None:
            kv.keys, kv.values = mx.zeros(shape, self.dtype), mx.zeros(shape, self.dtype)
            kv._idx = 0
            return
        keys, values = kv._temporal_order(kv.keys), kv._temporal_order(kv.values)
        count = min(kv.offset, self.window)
        if keys.shape[2] < count:
            raise ValueError("window cache holds fewer rows than its position needs")
        keys, values = keys[:, :, keys.shape[2] - count :], values[:, :, values.shape[2] - count :]
        if count == self.window:
            kv.keys, kv.values = mx.contiguous(keys), mx.contiguous(values)
        else:
            ring_k, ring_v = mx.zeros(shape, self.dtype), mx.zeros(shape, self.dtype)
            ring_k[:, :, :count, :] = keys
            ring_v[:, :, :count, :] = values
            kv.keys, kv.values = ring_k, ring_v
        kv._idx = count

    # ------------------------------------------------------------------ reads
    def window_view(self, visit):
        """``(keys, values, newest, count)`` of a window visit; integers as uint32 arrays."""
        kv = self.cache[2 * visit]
        count = min(self.length, self.window)
        newest = (kv._idx - 1) % self.window
        return kv.keys, kv.values, mx.array(newest, mx.uint32), mx.array(count, mx.uint32)

    def global_view(self, visit):
        """``(keys, values, count)`` of a global visit; the count as a uint32 array."""
        kv = self.cache[2 * visit]
        return kv.keys, kv.values, mx.array(self.length, mx.uint32)

    def tail(self, visit, slot):
        """The three previous float32 inputs of one convolution ``(1, 3, width)``."""
        return self.cache[2 * visit + 1][slot]

    def history(self, visit, k_rows, v_rows):
        """Keys and values in chronological order with the panel appended, exactly the
        arrays the reference cache hands to attention (reference stage B only: host shapes)."""
        kv = self.cache[2 * visit]
        if self.local[visit]:
            count = min(self.length, self.window - 1)
            index = (mx.arange(count, dtype=mx.int32) + (kv._idx - count)) % self.window
            keys, values = mx.take(kv.keys, index, axis=2), mx.take(kv.values, index, axis=2)
        else:
            keys, values = kv.keys[:, :, : self.length], kv.values[:, :, : self.length]
        return mx.concatenate([keys, k_rows], axis=2), mx.concatenate([values, v_rows], axis=2)

    # ------------------------------------------------------------------ commit
    def commit(self, deltas, kept):
        """Write the first ``kept`` rows of every visit's delta into the cache objects."""
        if not 0 <= kept <= MAX_PANEL:
            raise ValueError("commit takes 0 to MAX_PANEL rows")
        if kept == 0:
            return
        end = self.length + kept
        for visit, delta in enumerate(deltas):
            kv, conv = self.cache[2 * visit], self.cache[2 * visit + 1]
            keys, values = delta.keys[:, :, :kept], delta.values[:, :, :kept]
            if self.local[visit]:
                index = (mx.arange(kept, dtype=mx.int32) + kv._idx) % self.window
                kv.keys[:, :, index, :] = keys
                kv.values[:, :, index, :] = values
                kv._idx = (kv._idx + kept - 1) % self.window + 1
            else:
                if end > kv.keys.shape[2]:
                    self._grow(kv, end)
                kv.keys[:, :, self.length : end, :] = keys
                kv.values[:, :, self.length : end, :] = values
            kv.offset = end
            for slot, rows in enumerate(delta.taps):
                conv[slot] = mx.concatenate([conv[slot], rows[:, :kept]], axis=1)[:, -self.keep :]
        self.length = end
        self.pos = mx.array(end, mx.uint32)

    def _grow(self, kv, end):
        cap = round_up(end, BANK_STEP)
        shape = (1, self.heads, cap, self.head_dim)
        keys, values = mx.zeros(shape, self.dtype), mx.zeros(shape, self.dtype)
        keys[:, :, : self.length, :] = kv.keys[:, :, : self.length, :]
        values[:, :, : self.length, :] = kv.values[:, :, : self.length, :]
        kv.keys, kv.values = keys, values

    def arrays(self):
        out = []
        for visit in range(len(self.local)):
            kv, conv = self.cache[2 * visit], self.cache[2 * visit + 1]
            out += [kv.keys, kv.values, *conv.cache]
        return [a for a in out if a is not None]


def is_window(kv):
    return isinstance(kv, RotatingKVCache)
