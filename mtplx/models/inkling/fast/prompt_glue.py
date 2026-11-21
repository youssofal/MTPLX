"""The prompt read's glue between the expert blocks and attention, as our launches.

A prompt chunk's layer visit reads its input from the closed residual blocks twice (once with the
attention query, once with the MLP query) and norms what it reads, and the chunk ends with one more
read and the final norm. The prompt path writes each read as MLX operations: a stack of the blocks, a
cast, a norm of every block row, the scores of the query against them, a scale, a softmax, the mix,
and then the input norm and a cast: about twelve MLX launches a read. ``input_norm`` runs it as one
launch of the depth-read kernel the decode step already uses (``fast/read.py``): one threadgroup a row,
every step of it the same operation in the same order as MLX's kernel for that step, so its output
has the prompt path's bits at every chunk width (checked against the MLX chain at 33 to 4,096 rows,
one to four closed blocks, a bfloat16 or float32 partial).

Served: chunks of more than 32 rows (decode and verify panels keep their own routes), one query and
no cross-pass message (the first pass of the loop, and the visits outside it), up to four closed
blocks, a bfloat16 stream (the FP16 sibling packs keep MLX's chain: the kernel rounds to bfloat16),
and ``MTPLX_INKLING_PROMPT_KERNELS`` not ``mlx``. Everything else keeps MLX's chain, counted.
"""

from __future__ import annotations

from collections import Counter
import os
from threading import Lock

import mlx.core as mx

PROMPT_MIN_ROWS = 33
_lock = Lock()
_counts: Counter = Counter()


def counts():
    with _lock:
        return dict(_counts)


def _count(key):
    with _lock:
        _counts[key] += 1


def enabled():
    return os.environ.get("MTPLX_INKLING_PROMPT_KERNELS", "ours").strip().lower() not in ("mlx", "0", "off", "stock")


def _unserved(partial, closed, gain, where):
    if partial.ndim != 3 or partial.shape[1] < PROMPT_MIN_ROWS:
        return "rows"
    if not enabled():
        return "disabled"
    if not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        return "cpu"
    if len(closed) > 4:
        return "closed_blocks"
    width = partial.shape[-1]
    if width % 128 or width > 4096:
        return "width"
    if gain.dtype != mx.bfloat16 or partial.dtype not in (mx.bfloat16, mx.float32):
        return "stream_dtype"
    if any(c.dtype != mx.bfloat16 for c in closed):
        return "stream_dtype"
    return None


def input_norm(partial, closed, query, gain, eps, where, stand_in=None):
    """``norm(read_blocks(closed, partial, query, eps, True)[..., 0, :], gain, eps)`` in bfloat16 as one
    launch, or None when this call is not served (counted under ``where``). ``query`` None: no depth
    read (the first layer), the partial itself is normed; ``stand_in`` is then any float32 array."""
    reason = _unserved(partial, closed, gain, where)
    if reason is None and query is None and stand_in is None:
        reason = "stand_in"
    if reason is not None:
        _count(f"{where}.mlx.{reason}")
        return None
    from .read import read

    (out,) = read(partial, closed, query, eps, mode=0, gains=[gain], f32_stand_in=stand_in)
    _count(f"{where}.ours")
    return out.reshape(partial.shape)
