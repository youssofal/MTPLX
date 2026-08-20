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
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

COUNTERS: dict[str, int] = {
    "installed": 0,
    "wide_layer_calls": 0,
    "rungs_dispatched": 0,
}

# Per-forward layer counter. Layer calls within one forward are strictly
# sequential (single-threaded graph build), so a module counter that resets on
# every narrow (decode/verify) width tracks position inside a prefill forward
# to within one stride across chunk boundaries — good enough for rung pacing,
# and immune to whichever TextModel wrapper class runs the layer loop
# (mtp_patch shadows the stock forward with its own loop over stock layers).
_STATE: dict[str, Any] = {"idx": 0}


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

    def layer_call_with_rungs(self, x, mask=None, cache=None):
        out = layer_call(self, x, mask=mask, cache=cache)
        wide = x.ndim > 1 and int(x.shape[1]) >= min_seq
        if wide:
            COUNTERS["wide_layer_calls"] += 1
            idx = _STATE["idx"]
            _STATE["idx"] = idx + 1
            if idx % stride == 0:
                mx.async_eval(out)
                COUNTERS["rungs_dispatched"] += 1
        else:
            _STATE["idx"] = 0
        return out

    qwen3_5_module.DecoderLayer.__call__ = layer_call_with_rungs
    qwen3_5_module._mtplx_prefill_rungs_installed = True
    COUNTERS["installed"] += 1
    logger.info(
        "[prefill-rungs] installed: stride=%d min_seq=%d", stride, min_seq
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
            f"rungs_dispatched={COUNTERS['rungs_dispatched']}\n"
        )

    atexit.register(_dump_counters)
    return True
