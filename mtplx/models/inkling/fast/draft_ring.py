"""A drafter window block's observe of several rows written straight into its ring (lane F1).

After a round keeps two or more rows, the drafter observes them: each window block appends their
keys and values through the generic rotating cache, which puts the ring in time order, trims it and
joins the new rows (copies of the whole window), leaving an array longer than the window. The next
round's ``round_program.draft_inputs`` then finds no ring there and rebuilds one
(``round_program._make_ring``: the last ``window`` rows in time order at slots 0.., ``_idx`` the row
count) and evaluates it on the spot: a host wait between every multi-row round and the next, behind
the GPU's copies.

Here the observe writes that rebuilt ring directly, in one launch for the keys and values of a
block: slot ``s`` of the new ring holds the row of time index ``total - rows + s`` (``rows`` = the
ring's valid rows after the observe), read from the old ring's slot of that row or from the new rows,
zeros past ``rows`` (a ring not yet full), exactly the arrays and counters ``_make_ring`` leaves after
``update_and_fetch``. Bytes are moved, nothing is computed, so the drafter's state, its later
attention and its proposals are unchanged (class A); the next round finds a ring and evaluates
nothing. A single row (the cache's in-place write, already a ring), an entry that is not a full-size
ring, another dtype or a CPU run keeps the generic path; the first call of each shape is checked
against the generic path on copies, bit for bit (counted ``fallback.draft_ring.<why>``).
``MTPLX_INKLING_DRAFT_RING=0`` turns it off.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache

import mlx.core as mx
from .. import launch_limits

from .routes import count

_VERDICT: dict = {}

_SOURCE = """
    const uint gid = thread_position_in_grid.x;
    if (gid >= H * W * D) return;
    const uint d = gid % D, s = (gid / D) % W, h = gid / (D * W);
    const uint offset = meta[0], newest = meta[1], total = meta[2];
    const uint rows = metal::min(total, uint(W));
    TK kv = TK(0);
    TV vv = TV(0);
    if (s < rows) {
        const uint c = total - rows + s;                   // the row's time index
        if (c < offset) {
            const uint back = (offset - 1 - c) % W;           // rows between it and the newest
            const size_t at = (size_t(h) * W + (newest + W - back) % W) * D + d;
            kv = old_k[at];
            vv = old_v[at];
        } else {
            const size_t at = (size_t(h) * K + (c - offset)) * D + d;
            kv = new_k[at];
            vv = new_v[at];
        }
    }
    out_k[gid] = kv;
    out_v[gid] = vv;
"""


def enabled() -> bool:
    return os.environ.get("MTPLX_INKLING_DRAFT_RING", "1").strip().lower() not in ("0", "off", "false", "no", "")


@lru_cache(maxsize=1)
def _kernel():
    return launch_limits.kernel(name="mtplx_inkling_f1_draft_ring", input_names=["old_k", "old_v", "new_k", "new_v",
                                                                                 "meta"],
                                output_names=["out_k", "out_v"], source=_SOURCE)


def _same(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


def _ring(entry, keys, values):
    """(new keys, new values, offset, _idx) of the rebuilt ring, lazily."""
    window = entry.max_size
    heads, kept, dim = keys.shape[1], keys.shape[2], keys.shape[3]
    offset = int(entry.offset)
    total = offset + kept
    newest = (int(entry._idx) - 1) % window
    meta = mx.array([offset, newest, total], mx.uint32)
    out_k, out_v = _kernel()(
        inputs=[entry.keys, entry.values, keys, values, meta],
        template=[("TK", entry.keys.dtype), ("TV", entry.values.dtype), ("H", heads), ("W", window), ("D", dim),
                  ("K", kept)],
        grid=(heads * window * dim, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[entry.keys.shape, entry.values.shape],
        output_dtypes=[entry.keys.dtype, entry.values.dtype],
    )
    return out_k, out_v, total, min(total, window)


def _generic(entry, keys, values):
    """The generic path on a copy of ``entry``: update_and_fetch, then draft_inputs' ring rebuild."""
    from .round_program import _make_ring

    twin = copy.copy(entry)
    twin.update_and_fetch(keys, values)
    _make_ring(twin, entry.max_size)
    return twin.keys, twin.values, int(twin.offset), int(twin._idx)


def observe_rows(entry, keys, values) -> bool:
    """Write ``keys``, ``values`` ``(1, H, K, D)`` (K >= 2, the observed rows, rotated) into a window
    block's ring as ``update_and_fetch`` and the next round's ring rebuild would leave it. True when
    done here; False when the caller must take the generic path (nothing changed)."""
    from .round_program import _is_ring

    window = getattr(entry, "max_size", None)
    why = None
    if not enabled():
        return False
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        why = "device"
    elif (window is None or entry.keys is None or keys.ndim != 4 or keys.shape[2] < 2 or keys.shape[2] > window
          or values.shape[:3] != keys.shape[:3]):
        why = "shape"
    elif getattr(entry, "keep", 0) or entry.keys.shape[2] != window or entry.keys.shape[:2] != keys.shape[:2]:
        why = "not_a_ring"
    elif entry.keys.dtype != keys.dtype or entry.values.dtype != values.dtype or entry.keys.shape[3] != keys.shape[3]:
        why = "dtype"
    elif not _is_ring(entry, window):
        why = "not_a_ring"
    if why is not None:
        count(f"fallback.draft_ring.{why}")
        return False
    key = (keys.shape[1], window, keys.shape[3], keys.shape[2], str(keys.dtype), int(entry.offset) >= window)
    verdict = _VERDICT.get(key)
    if verdict is None:
        try:
            ours = _ring(entry, keys, values)
            stock = _generic(entry, keys, values)
            mx.eval(ours[0], ours[1], stock[0], stock[1])
            verdict = _same(ours[0], stock[0]) and _same(ours[1], stock[1]) and ours[2:] == stock[2:]
        except (RuntimeError, ValueError):
            verdict = False
        _VERDICT[key] = verdict
        from .attn_input import count_route

        count_route(("proved:" if verdict else "refused:") + "draft_ring_" + "_".join(map(str, key)))
    if not verdict:
        count("fallback.draft_ring.differs")
        return False
    entry.keys, entry.values, entry.offset, entry._idx = _ring(entry, keys, values)
    return True


__all__ = ["enabled", "observe_rows"]
