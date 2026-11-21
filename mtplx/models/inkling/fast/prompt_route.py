"""The prompt read's fast route for Inkling: wider chunks and the expert products on bfloat16 operands.

``MTPLX_INKLING_PROMPT_FAST`` (off by default) is ``on`` (every part) or a comma list of:

* ``chunk``: the prompt read takes 4,096 rows a forward instead of 512;
* ``experts``: the MXFP8 blocks' gate-up takes the sorted product on the accelerator (MLX's grouped product,
  as the NVFP4 blocks' does) instead of the one-row panel product, both MXFP8 products on bfloat16 operands
  (the rounded expert inputs, which bfloat16 holds exactly). NVFP4 blocks are unchanged.

Neither part is bit-identical to the default prompt read: the MXFP8 gate-up sums in the sorted product's order
instead of the one-row product's, and a chunk boundary moves rounding in the attention calls. So the route
stays off until its exactness class is accepted; its rows are in the lane's records.
"""

from __future__ import annotations

import os

PROMPT_FAST_ENV = "MTPLX_INKLING_PROMPT_FAST"
PROMPT_FAST_PARTS = ("chunk", "experts")


def prompt_fast(part):
    """Whether the prompt read's fast route serves ``part`` (read per call)."""
    raw = os.environ.get(PROMPT_FAST_ENV, "").strip().lower()
    if raw in ("", "0", "off", "no", "false"):
        return False
    if raw in ("1", "on", "yes", "true", "all"):
        return True
    return part in {word.strip() for word in raw.split(",")}
