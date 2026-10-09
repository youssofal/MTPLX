"""Intra-forward async dispatch rungs for chunked prefill (Speed War 2 row 5-C4).

Mechanism (mlxfast arena receipt, prefill 939->972 on the M5-class ranked box):
MLX builds a whole prefill-chunk forward lazily and dispatches only at the
end-of-chunk eval, so the GPU idles while the host walks 64 layers of graph
construction. Dispatching ``mx.async_eval`` on the hidden stream at layer 0
and every Nth layer afterwards lets the GPU execute layer k while the host
builds layer k+1. ``async_eval`` changes scheduling, never values.

Off by default. ``MTPLX_PREFILL_ASYNC_RUNGS=<stride>`` (>=1) enables the
install; rungs fire only on forwards whose sequence length is at least
``MTPLX_PREFILL_ASYNC_RUNGS_MIN_SEQ`` (default 512), so decode/verify widths
never take the hook. Class-level wraps with engagement counters (mistakes/:
verify a monkeypatch engaged with a counter before reading any A/B).

``MTPLX_PREFILL_ASYNC_RUNGS_STATES=1`` (default off, needs a stride) makes a
rung also carry the recurrent states the layers since the previous rung left
on their cache entries: each GDN layer's conv tail (a 3-row copy of its
``[rows + 3, conv_dim]`` pre-conv stream), its delta state and any in-forward
boundary captures. With the hidden alone those stay lazy until the chunk's
end-of-forward eval, so every GDN layer's pre-conv stream stays alive to the
end of the forward (A3B: 30 x 2,048 x 8,192 x 2 B, ~1 GB per 2,048-row
forward). Naming them lets each stream die with its layer (Flash-Next's
mid-loop eval, 262592b6). With it, draft-head history appends
(``draft_history_scope``) never fire a rung: their output is not needed
(``MTPLX_MTP_HISTORY_CACHE_ONLY``) and they are no trunk layer. Same graph,
same kernels: only when work reaches the GPU changes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import mlx.core as mx

# Attribute an in-forward GDN boundary capture lives under on a cache entry
# (set by gdn_inforward_boundaries, where that lane is installed). Named here
# so this module does not depend on the lane.
BOUNDARY_CAPTURE_ATTR = "_mtplx_boundary_captures"

logger = logging.getLogger(__name__)

COUNTERS: dict[str, int] = {
    "installed": 0,
    "wide_layer_calls": 0,
    "rungs_dispatched": 0,
    "state_arrays_dispatched": 0,
}
_DRAFT_HISTORY: ContextVar[bool] = ContextVar("mtplx_prefill_rungs_draft_history", default=False)

# Per-forward layer counter. Layer calls within one forward are strictly
# sequential (single-threaded graph build), so a module counter that resets on
# every narrow (decode/verify) width tracks position inside a prefill forward
# to within one stride across chunk boundaries — good enough for rung pacing,
# and immune to whichever TextModel wrapper class runs the layer loop
# (mtp_patch shadows the stock forward with its own loop over stock layers).
_STATE: dict[str, Any] = {"idx": 0, "pending": []}


def rungs_stride() -> int:
    raw = os.environ.get("MTPLX_PREFILL_ASYNC_RUNGS", "0")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _min_seq() -> int:
    raw = os.environ.get("MTPLX_PREFILL_ASYNC_RUNGS_MIN_SEQ", "512")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 512


def rungs_carry_states() -> bool:
    """``MTPLX_PREFILL_ASYNC_RUNGS_STATES=1``: rungs carry the recurrent states."""

    raw = os.environ.get("MTPLX_PREFILL_ASYNC_RUNGS_STATES", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def draft_history_scope() -> Iterator[None]:
    """Mark the enclosed calls as a draft-head history append (no trunk forward)."""

    token = _DRAFT_HISTORY.set(True)
    try:
        yield
    finally:
        _DRAFT_HISTORY.reset(token)


def _collect_arrays(node: Any, found: list) -> None:
    if isinstance(node, mx.array):
        found.append(node)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_arrays(item, found)
    elif isinstance(node, dict):
        for item in node.values():
            _collect_arrays(item, found)


def recurrent_state_arrays(entries: list) -> list:
    """The arrays finished GDN layers left on their cache entries: the conv
    tail and delta state, plus any in-forward boundary captures. KV entries
    are consumed by their own layer's attention and need no naming."""

    from mlx_lm.models.cache import ArraysCache

    found: list = []
    for entry in entries:
        if isinstance(entry, ArraysCache):
            _collect_arrays(getattr(entry, "cache", None), found)
            _collect_arrays(getattr(entry, BOUNDARY_CAPTURE_ATTR, None), found)
    return found


def install_qwen3_5_prefill_rungs() -> bool:
    """Install the rung wrappers on the qwen3_5 model classes (idempotent)."""

    stride = rungs_stride()
    if stride < 1:
        return False
    import mlx_lm.models.qwen3_5 as qwen3_5_module

    if getattr(qwen3_5_module, "_mtplx_prefill_rungs_installed", False):
        return True

    layer_call = qwen3_5_module.DecoderLayer.__call__
    min_seq = _min_seq()
    carry_states = rungs_carry_states()

    def layer_call_with_rungs(self, x, mask=None, cache=None):
        out = layer_call(self, x, mask=mask, cache=cache)
        if carry_states and _DRAFT_HISTORY.get():
            return out
        wide = x.ndim > 1 and int(x.shape[1]) >= min_seq
        pending = _STATE["pending"]
        if wide:
            COUNTERS["wide_layer_calls"] += 1
            idx = _STATE["idx"]
            _STATE["idx"] = idx + 1
            if carry_states and cache is not None:
                pending.append(cache)
            if idx % stride == 0:
                states = recurrent_state_arrays(pending) if carry_states else []
                pending.clear()
                mx.async_eval(out, *states)
                COUNTERS["rungs_dispatched"] += 1
                COUNTERS["state_arrays_dispatched"] += len(states)
        else:
            _STATE["idx"] = 0
            pending.clear()
        return out

    qwen3_5_module.DecoderLayer.__call__ = layer_call_with_rungs
    qwen3_5_module._mtplx_prefill_rungs_installed = True
    COUNTERS["installed"] += 1
    logger.info(
        "[prefill-rungs] installed: stride=%d min_seq=%d carry_states=%s",
        stride,
        min_seq,
        carry_states,
    )
    # Engagement receipt at exit (mistakes/: a monkeypatch A/B is void until a
    # counter proves the patch engaged). Only registered when enabled.
    import atexit
    import sys

    def _dump_counters() -> None:
        sys.stderr.write(
            "[prefill-rungs] exit receipt: "
            f"installed={COUNTERS['installed']} "
            f"wide_layer_calls={COUNTERS['wide_layer_calls']} "
            f"rungs_dispatched={COUNTERS['rungs_dispatched']} "
            f"state_arrays_dispatched={COUNTERS['state_arrays_dispatched']}\n"
        )

    atexit.register(_dump_counters)
    return True
