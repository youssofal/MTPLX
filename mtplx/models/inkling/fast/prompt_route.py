"""The prompt read's fast route for Inkling: wider chunks and the expert products on bfloat16 operands.

``MTPLX_INKLING_PROMPT_FAST`` (off by default) is ``on`` (every part) or a comma list of:

* ``chunk``: the prompt read takes ``WIDE_ROWS`` (2,048) rows a forward instead of 512, on a prompt the
  attention's planes allow (``WIDE_PROMPT_TOKENS``) and a Mac with the memory for it (``WIDE_HEADROOM_BYTES``);
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


#: Rows a forward the ``chunk`` part reads. Not 4,096 while the prompt attention computes every key of its call:
#: a window visit at 4,096 rows computes 5,119 keys a row for at most 1,024 visible, and the read is slower than
#: at 2,048 (Optimized Quality, 4K in process: 4.77 s against 4.12 s; Bare Speed, 16K served: 805 against 965
#: tok/s; `lanes/p1/sheet-oq-4k-2.json`, `lanes/p1/mem-bare-16k-b-20261121T162621Z.json`).
WIDE_ROWS = 2048
#: The longest prompt the wide width serves. The prompt attention writes bias and mask planes as tall as the
#: chunk and as long as the context: a cold 16K read on Bare Speed grew MLX's memory by 5.5 GiB at 2,048 rows
#: against 3.6 at 512 (`lanes/p1/mem-pair-bare-cand-20261121T174711Z.json`), and nothing past 16K is measured,
#: so a longer prompt keeps 512 rows until the prompt attention stops writing them.
WIDE_PROMPT_TOKENS = 18_432
#: GPU memory above the loaded model a Mac needs before the wide width: the 2,048-row read's growth at the longest
#: prompt it serves (5.5 GiB at 16K, 1.9 more than at 512 rows), with room beside it for the session's own state.
#: Every 24, 32 and 36 GB Mac keeps 512 rows; the admission may still narrow any request.
WIDE_HEADROOM_BYTES = 12 * 1024**3

#: With the one-launch prompt attention on (``fast/prompt_flash.py``, ``MTPLX_INKLING_PROMPT_FLASH``, lane P2) a
#: call computes only the keys its rows may see and writes no bias or mask plane: the width is 4,096 rows and the
#: prompt fence does not apply (the forward's own memory, 2.8 to 2.9 GB in process at 4,096 rows, is priced by
#: the admission like any width).
WIDE_ROWS_FLASH = 4096

_headroom = {}


def flash_attention_on():
    """Whether the prompt chunk's attention is the one-launch route (present in this tree and switched on)."""
    raw = os.environ.get("MTPLX_INKLING_PROMPT_FLASH", "").strip().lower()
    if raw in ("", "0", "off", "no", "false"):
        return False
    import importlib.util

    return importlib.util.find_spec("mtplx.models.inkling.fast.prompt_flash") is not None


def gpu_headroom_bytes():
    """The GPU memory limit less what MLX holds, read once per process (at the first prompt read, when it holds
    the model), so every later call (the admission's pricing and the read itself) gets the same answer."""
    if "bytes" not in _headroom:
        try:
            import mlx.core as mx

            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            limit = int(info.get("max_recommended_working_set_size", 0))
            _headroom["bytes"] = limit - int(mx.get_active_memory()) if limit else None
        except Exception:  # noqa: BLE001 - no Metal device: no wide width
            _headroom["bytes"] = None
    return _headroom["bytes"]


def wide_rows(prompt_tokens=None, headroom_bytes=None):
    """The ``chunk`` part's width for a prompt of ``prompt_tokens``, or None for the default 512 rows."""
    if not prompt_fast("chunk"):
        return None
    flash = flash_attention_on()
    if not flash and prompt_tokens is not None and int(prompt_tokens) > WIDE_PROMPT_TOKENS:
        return None
    room = gpu_headroom_bytes() if headroom_bytes is None else headroom_bytes
    if room is None or room < WIDE_HEADROOM_BYTES:
        return None
    return WIDE_ROWS_FLASH if flash else WIDE_ROWS
