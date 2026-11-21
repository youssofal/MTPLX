"""Fewer launches for the stock commit (lane L7, for lane L0 to wire): the same bytes, one launch
where MLX takes several.

With the committing round off, the commit and the drafter's observe run on MLX between two rounds,
on the dependent chain (the next round reads what they write), and every MLX launch costs the host
its encode: about 380 launches and 2 to 3 ms a round (21 November, the load's own timing). Two of
their pieces are pure copies, or a float32 sum in a fixed order, spread over many small launches:

- ``visit_tails``: a visit's four convolution tails (four in-place slice copies, or a
  concatenation each when fewer rows than the tail are kept) in one launch writing the four new
  tails; 48 visits make 144 fewer launches.
- ``mix``: a drafter mix (each of the four taps converted to float32, multiplied by its weight and
  summed left to right from 0, then MLX's RMS norm) in one launch for the weighted sum: about 15
  fewer launches a mix, four mixes a commit (the three blocks and the Markov head's feature).

Both write the stock operations' bytes: the tails are copies; the sum is taken in the same order
with every product and every sum rounded once, as MLX's separate multiply and add kernels round
them (explicit ``fma(a, b, -0)`` for the product and ``fma(s, 1, p)`` for the sum, so the compiler
cannot contract a product into the sum). ``tests/test_inkling_fast_commit_launches.py`` checks
both bit for bit against the stock operations.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from . import round as fast_round
from .round_program import _tails_kernel

_MIX = """
    uint gid = thread_position_in_grid.x;
    if (gid >= N) return;
    float s = 0.0f;
    s = metal::fma(s, 1.0f, metal::fma(float(h0[gid]), float(w[0]), -0.0f));
    s = metal::fma(s, 1.0f, metal::fma(float(h1[gid]), float(w[1]), -0.0f));
    s = metal::fma(s, 1.0f, metal::fma(float(h2[gid]), float(w[2]), -0.0f));
    s = metal::fma(s, 1.0f, metal::fma(float(h3[gid]), float(w[3]), -0.0f));
    out[gid] = s;
"""


@lru_cache(maxsize=None)
def _mix_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_l7_mix4",
        input_names=["h0", "h1", "h2", "h3", "w"],
        output_names=["out"],
        source=_MIX,
    )


def visit_tails(tails, rows, kept: int):
    """The four new convolution tails of one visit after ``kept`` committed rows: the stock
    commit's ``tail[:, :, :] = rows[:, kept - keep : kept]`` (``kept >= keep``) or
    ``concatenate([tail[:, kept:], rows[:, :kept]])``, for the four slots in one launch.
    ``tails``: the visit's four tails ``(1, keep, W_s)``; ``rows``: the verified rows of each slot
    ``(1, R, W_s)``. Returns four new arrays (the tails are small: no in-place write needed)."""
    if len(tails) != 4 or len(rows) != 4:
        raise ValueError("a visit has four convolution tails")
    keep = int(tails[0].shape[1])
    widths = [int(t.shape[-1]) for t in tails]
    return _tails_kernel()(
        inputs=[*tails, *rows, fast_round._const([int(kept)], mx.uint32)],
        template=[("KEEP", keep), ("W0", widths[0]), ("W1", widths[1]), ("W2", widths[2]), ("W3", widths[3])],
        grid=(keep * sum(widths), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[t.shape for t in tails],
        output_dtypes=[t.dtype for t in tails],
    )


def mix(features, weights, norm_weight, eps: float):
    """``Drafter.mix``: ``rms_norm(sum(h.astype(float32) * w.astype(float32)), norm_weight, eps)``
    in the taps' dtype, with the weighted sum of the four taps in one launch. ``features``: four
    taps of one shape; ``weights``: their four mixing weights (a row of the drafter's blend)."""
    if len(features) != 4:
        raise ValueError("the drafter mixes four taps")
    dtype = features[0].dtype
    shape = features[0].shape
    n = 1
    for d in shape:
        n *= int(d)
    mixed = _mix_kernel()(
        inputs=[*features, weights.reshape(-1)],
        template=[("N", n)],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
    )[0]
    return mx.fast.rms_norm(mixed, norm_weight, eps).astype(dtype)


__all__ = ["mix", "visit_tails"]
