"""The prompt read's fast route for Inkling: wider chunks and the expert products on bfloat16 operands.

``MTPLX_INKLING_PROMPT_FAST`` (off by default) is ``on`` (every part) or a comma list of:

* ``chunk``: the prompt read takes ``WIDE_ROWS`` (2,048) rows a forward instead of 512, on a prompt the
  attention's planes allow (``WIDE_PROMPT_TOKENS``) and a Mac with the memory for it (``WIDE_HEADROOM_BYTES``);
* ``experts``: the MXFP8 blocks' gate-up takes the sorted product on the accelerator (MLX's grouped product,
  as the NVFP4 blocks' does) instead of the one-row panel product, both MXFP8 products on bfloat16 operands
  (the rounded expert inputs, which bfloat16 holds exactly). NVFP4 blocks are unchanged.
* ``invariant``: the prompt read's sums do not depend on the chunk width. Every product of a prompt chunk runs
  on our launches at every row count (the projections, the router, the controllers, both expert products in
  both formats, the drafter's context heads), each output element summed in one fixed order whatever the
  rows around it; the glue kernels serve every row count; with ``chunk`` the width is then the widest that
  the Mac's memory allows (``wide_rows``), and every width gives the same cache. Implies ``experts``.

Neither part is bit-identical to the default prompt read: the MXFP8 gate-up sums in the sorted product's order
instead of the one-row product's, and a chunk boundary moves rounding in the attention calls. So the route
stays off until its exactness class is accepted; its rows are in the lane's records.
"""

from __future__ import annotations

import os

PROMPT_FAST_ENV = "MTPLX_INKLING_PROMPT_FAST"
PROMPT_FAST_PARTS = ("chunk", "experts", "invariant")


def prompt_fast(part):
    """Whether the prompt read's fast route serves ``part`` (read per call). Unset, the read width (``chunk``)
    is on (its class accepted on 21 November, with its accuracy table) and the other parts are off."""
    raw = os.environ.get(PROMPT_FAST_ENV, "").strip().lower()
    if not raw:
        return part == "chunk"
    if raw in ("0", "off", "no", "false"):
        return False
    if raw in ("1", "on", "yes", "true", "all"):
        return True
    words = {word.strip() for word in raw.split(",")}
    return part in words or (part == "experts" and "invariant" in words)


def invariant_sums():
    """Whether the prompt read's sums are width-independent (the ``invariant`` part, read per call)."""
    return prompt_fast("invariant")


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
#: call computes only the keys its rows may see and writes no bias or mask plane: the prompt fence does not apply
#: (the forward's own memory is priced by the admission like any width). 2,048 rows until 4,096 is measured faster
#: with that attention (with today's it was slower: `lanes/p1/sheet-oq-4k-2.json`).
WIDE_ROWS_FLASH = 2048

#: The prompt read hands its work to the GPU without waiting inside a forward (``MTPLX_INKLING_PROMPT_SYNC=1`` restores
#: the waits): a prompt call's query tiles go over as they are built with at most ``TILES_IN_FLIGHT`` unfinished, and
#: the forward hands over its residual stream every ``HAND_OVER_VISITS`` visits, so the GPU starts a chunk while
#: Python builds the rest of it. The waits were the attention's: one per query tile (about 110 a 512-row chunk, 500
#: a 2,048-row chunk), each draining the GPU and leaving it idle while the host built the next tile. Same graph,
#: same kernels, same inputs to every kernel: only when the host waits changes, so every bit is unchanged.
SYNC_ENV = "MTPLX_INKLING_PROMPT_SYNC"
TILES_IN_FLIGHT = 2
HAND_OVER_VISITS = 4


def asynchronous():
    """Whether the prompt read hands work over without waiting (read per call; on unless the switch says sync)."""
    return os.environ.get(SYNC_ENV, "").strip().lower() not in ("1", "on", "yes", "true", "sync")


def hand_over(parts, part):
    """Evaluate ``part``, a prompt call's next query tile (``parts``: its earlier tiles, kept alive by the caller).
    Synchronous: wait for it, as the attention did. Otherwise start it and wait only for the tile
    ``TILES_IN_FLIGHT`` places back: the same few tiles alive, the GPU never drained between them."""
    import mlx.core as mx

    if not asynchronous():
        mx.eval(part)
        return
    mx.async_eval(part)
    if len(parts) >= TILES_IN_FLIGHT:
        mx.eval(parts[-TILES_IN_FLIGHT])


_headroom = {}


def flash_attention_on(prompt_tokens=None):
    """Whether the prompt chunk's attention is the one-launch route: switched on (the default) and, for a
    prompt of ``prompt_tokens``, one the route reads (not shorter than its crossover)."""
    from ..attention import flash_switch_on
    from . import prompt_flash

    if not flash_switch_on():
        return False
    return prompt_tokens is None or not prompt_flash.short_prompt(int(prompt_tokens))


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


#: Widths the invariant read may take, widest first: every one gives the same cache, so the widest that fits wins.
INVARIANT_WIDTHS = (4096, 2048, 1024, 512, 256)
#: One prompt forward's working memory beyond its KV, in process (OQ and Bare Speed, 4K): 1.03 to 1.13 GB at 512
#: rows, 2.03 at 2,048, 2.8 to 2.9 at 4,096 (`lanes/p1/sheet-*-4k-*.json`): about 0.6 GB plus 0.55 MB a row.
FORWARD_FIXED_BYTES = 640 * 1024**2
FORWARD_ROW_BYTES = 576 * 1024
#: The attention's bias and mask planes at today's route: 24 heads x rows x keys of 16-bit values, about three
#: alive at once (a cold 16K read grew 1.9 GiB more at 2,048 rows than at 512: `lanes/p1/mem-pair-bare-cand-*`).
PLANE_COPIES = 3
#: What the width may use of the GPU headroom: the rest stays for the session's own state and the admission.
HEADROOM_SHARE = 0.5


def forward_bytes(rows, keys, flash=None):
    """A prompt forward's working memory at ``rows`` rows against ``keys`` keys (arithmetic from measured rows)."""
    flash = flash_attention_on() if flash is None else flash
    planes = 0 if flash else PLANE_COPIES * 24 * int(rows) * max(int(keys), int(rows)) * 2
    return FORWARD_FIXED_BYTES + FORWARD_ROW_BYTES * int(rows) + planes


def wide_rows(prompt_tokens=None, headroom_bytes=None):
    """The ``chunk`` part's width for a prompt of ``prompt_tokens``, or None for the default 512 rows."""
    if not prompt_fast("chunk"):
        return None
    flash = flash_attention_on(prompt_tokens)
    room = gpu_headroom_bytes() if headroom_bytes is None else headroom_bytes
    if invariant_sums():
        # every width gives the same cache: the widest whose forward fits in its share of the headroom
        if room is None:
            return None
        # (no wider than the attention pays for: 2,048 rows while it computes every key of its call)
        keys = int(prompt_tokens) if prompt_tokens is not None else 0
        widest = WIDE_ROWS_FLASH if flash else WIDE_ROWS
        for rows in (w for w in INVARIANT_WIDTHS if w <= widest):
            if forward_bytes(rows, keys, flash) <= HEADROOM_SHARE * room:
                return rows
        return min(INVARIANT_WIDTHS)
    if not flash and prompt_tokens is not None and int(prompt_tokens) > WIDE_PROMPT_TOKENS:
        return None
    if room is None or room < WIDE_HEADROOM_BYTES:
        return None
    return WIDE_ROWS_FLASH if flash else WIDE_ROWS
