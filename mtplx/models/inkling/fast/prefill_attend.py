"""Inkling prompt attention for full-history visits at long context, exact against RC6.

RC6 reads a prompt chunk of R rows (512 by default) with ``attention.stock_attention``: it
appends a zero key/value row carrying the sink logit, writes an additive bf16 mask, and calls
MLX's attention on query tiles of ``step = 4M // (24 * N)`` rows, N = keys + 1, evaluating
each tile before the next. Past about 19,400 keys a tile holds 8 rows or fewer, so every call
takes MLX 0.32.2's *vector* kernel in its two-pass form: for each query row and head, key i is
folded into block ``i mod BLOCKS`` by MLX's per-key statements (keys the mask sets to -inf are
skipped), each block's output is rounded to bf16, and a second pass merges the blocks. BLOCKS
follows MLX's rule for the call's shape: n_simds = 3 x (rows in the call).

This module computes exactly that arithmetic for every row of the chunk with the decode
kernels of ``attend.py`` (proven equal to MLX's vector kernel bit for bit, decode and verify),
generalised in three ways and nothing else: rows in groups of 8 per threadgroup (16 rows a
launch), the sink folded as the call's last key (index N - 1 for every row, where decode
folds it after the row's own position), and no gate (RC6 gates after attention). The
history, the chunk's own rows (already in the cache) and the bias are read in place: no
concatenation, no mask tensor, one read of the bank per 16 rows instead of per tile.

Rows of a chunk whose tile RC6 sends to MLX's *full* kernel (more than 8 rows a call) are
not served here: ``decline`` names the reason and the caller runs RC6's own call.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from . import attend as A

#: Rows a launch serves (a threadgroup holds 8 of them; partials for 16 rows at 1,024 blocks
#: are 100 MB in bf16, so a 104K chunk stays well inside any Mac's working set).
LAUNCH_ROWS = 16
GROUP_ROWS = 8
#: Launches evaluated together. A visit's launches stay lazy until something evaluates them,
#: and MLX then keeps about ten command buffers in flight, so a 512-row visit held about 20
#: launches' partials at once (1.04 GB at 32K keys, 2.08 GB at 64K against RC6's 0.19 and
#: 0.31; lanes/l6/probe-prompt-memory-1.json). Evaluating each group of EVAL_LAUNCHES as it is
#: made holds a few launches' partials; the evaluation boundary changes no arithmetic.
EVAL_LAUNCHES = 4

_SRC_ROW = "    const int r = int(simdgroup_index_in_threadgroup);     // panel row\n"
_DST_ROW = ("    const int sg = int(simdgroup_index_in_threadgroup);\n"
            "    const int r = int(threadgroup_position_in_grid.z) * ROWS + sg;   // row of this launch\n"
            "    if (r >= int(q_shape[1])) {\n"
            "        return;\n"
            "    }\n")


def _derive_partials():
    src = A._PARTIALS
    for old, new in (
        (_SRC_ROW, _DST_ROW),
        ("qsh + r * 3 * 128", "qsh + sg * 3 * 128"),
        ("esh + r * 32 * 4", "esh + sg * 32 * 4"),
        ("    const bool two = (n + 1) >= THRESH;        // + 1: RC6's appended zero row\n",
         "    const bool two = true;                     // prompt calls here are past the threshold\n"),
        ("    if (((P + 1 - rho) % span + span) % span == 0) {\n",
         "    if (((int(sinkpos[0]) - rho) % span + span) % span == 0) {\n"),
        # partials kept in the query type, as MLX keeps them (the values are rounded to it anyway)
        ("            po[c * 128 + 4 * lane + j] = two ? static_cast<float>(static_cast<T>(o[h3][j])) : o[h3][j];\n",
         "            po[c * 128 + 4 * lane + j] = static_cast<T>(o[h3][j]);\n"),
    ):
        count = src.count(old)
        if count == 0:
            raise RuntimeError(f"prefill kernel derivation: pattern not found: {old.strip()[:60]}")
        src = src.replace(old, new)
    return src


def _derive_merge():
    src = A._MERGE
    old_gate = src[src.index("            // The attention result in the query type, then the gate as RC6 rounds it:"):
                   src.index("            hg[at] = static_cast<T>(hv * gv);\n") + len("            hg[at] = static_cast<T>(hv * gv);\n")]
    for old, new in (
        ("    const bool two = (n + 1) >= THRESH;\n", "    const bool two = true;\n"),
        (old_gate, "            hg[at] = static_cast<T>(o[i]);   // MLX's second pass writes the query type\n"),
    ):
        if src.count(old) != 1:
            raise RuntimeError("prefill merge derivation: pattern not found")
        src = src.replace(old, new)
    return src


_PARTIALS_P = _derive_partials()
_MERGE_P = _derive_merge()
_PARTIALS_NAME = "mtplx_inkling_prompt_attend_partials"
_MERGE_NAME = "mtplx_inkling_prompt_attend_merge"
A._INPUTS[_PARTIALS_NAME] = ["q", "kc", "vc", "kr", "vr", "prof", "sink", "pos", "count", "newest", "sinkpos"]
A._OUTPUTS[_PARTIALS_NAME] = ["po", "pm", "ps"]
A._INPUTS[_MERGE_NAME] = ["po", "pm", "ps", "pos", "count"]
A._OUTPUTS[_MERGE_NAME] = ["hg"]


def vector_blocks(rows_in_call, n_keys):
    """MLX 0.32.2's block count for its two-pass vector kernel at this call shape, on this GPU
    family (``sdpa_vector_2pass``), including ``MLX_SDPA_BLOCKS``."""
    n_simds = 3 * rows_in_call
    tail = A._family()
    if tail == "s":
        blocks = 64
        if n_keys > 1024 and n_simds > 4:
            blocks = 128 if n_keys <= 8192 else 256 if n_keys <= 32768 else 512 if n_keys <= 65536 else 1024
    elif tail == "d":
        blocks = 128
        if n_simds <= 2 and n_keys > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_keys < 65536:
                blocks = 512
            elif n_keys >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    override = A._atoi(os.environ.get("MLX_SDPA_BLOCKS"))
    if override > 0:
        blocks = ((override + 31) // 32) * 32
    return blocks


def tiles(rows, n_keys, heads=24):
    """RC6's query tiles for a chunk: [(first row, end row, rows a call)]."""
    step = max(1, (4 * 1024 * 1024) // (heads * n_keys))
    out = []
    for start in range(0, rows, step):
        end = min(rows, start + step)
        out.append((start, end, end - start))
    return out


def decline(query, key, value, profiles, sink, window):
    """None when every row of this prompt call can be served exactly here, else why not."""
    if window:
        return "window"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if query.dtype != mx.bfloat16 or any(x.dtype != mx.bfloat16 for x in (key, value, profiles, sink)):
        return "dtype"
    batch, heads, rows, width = query.shape
    if (batch, heads, width, key.shape[1], key.shape[3]) != (1, 24, 128, 8, 128) or value.shape != key.shape \
            or profiles.shape != (1, rows, 24, 1024) or sink.shape != (1, rows, 24) or rows > key.shape[2]:
        return "geometry"
    n_keys = key.shape[2] + 1
    threshold, _ = A.reduction_plan()
    if n_keys < threshold:
        return "single_pass"
    if any(q_len > 8 for _, _, q_len in tiles(rows, n_keys)):
        return "full_kernel_tiles"
    for _, _, q_len in tiles(rows, n_keys):
        if vector_blocks(q_len, n_keys) > 1024:
            return "blocks"
    reason = A.decline_reason(mx.bfloat16)
    if reason is not None:
        return "decode_route_" + reason
    if not A.lane_tree_ok():
        return "lane_tree"
    return _qualify(A.reduction_plan(), os.environ.get("MLX_SDPA_BLOCKS"))


@lru_cache(maxsize=8)
def _qualify(plan, override):
    """Bits against RC6's own call on a synthetic chunk, once per (plan, override)."""
    try:
        if not _same_as_rc6(n_hist=20000, rows=40):
            return "bits"
    except (RuntimeError, ValueError) as exc:
        A.DECLINED[("prompt", plan)] = ("kernel_refused", str(exc)[:300])
        return "kernel_refused"
    return None


def _same_as_rc6(n_hist, rows):
    from ..attention import stock_attention

    keys = n_hist + rows
    ks = mx.random.split(mx.random.key(20261121 + keys), 5)

    def clipped(shape, limit, key):
        v = mx.random.normal(shape, key=key) * 2.2
        n = mx.linalg.norm(v, axis=-1, keepdims=True)
        return (v * mx.minimum(1.0, limit / (n + 1e-6))).astype(mx.bfloat16)

    q = clipped((1, rows, 24, 128), 22.6, ks[0]).transpose(0, 2, 1, 3)
    k = clipped((1, 8, keys, 128), 22.6, ks[1])
    v = mx.fast.rms_norm(mx.random.normal((1, 8, keys, 128), key=ks[2]), None, 1e-6).astype(mx.bfloat16)
    prof = clipped((1, rows, 24, 1024), 8.0, ks[3])
    sink = (mx.random.normal((1, rows, 24), key=ks[4]) * 3.0).astype(mx.bfloat16)
    ref = stock_attention(q, k, v, prof, sink, 0)
    got = _prompt_attention(q, k, v, prof, sink)
    mx.eval(ref, got)
    return bool(mx.array_equal(ref.view(mx.uint16), got.view(mx.uint16)).item())


def prompt_attention(query, key, value, profiles, sink, window):
    """RC6's ``stock_attention`` result for this prompt call, ``(1, 24, rows, 128)``, computed
    in place. Call only where ``decline`` returned None."""
    return _prompt_attention(query, key, value, profiles, sink)


def _prompt_attention(query, key, value, profiles, sink):
    rows = query.shape[2]
    n_keys = key.shape[2] + 1
    hist = key.shape[2] - rows
    threshold, _ = A.reduction_plan()
    q = query.transpose(0, 2, 1, 3)                       # (1, rows, 24, 128), a view
    sinkpos = mx.array([key.shape[2]], mx.uint32)       # the appended row's index: N - 1
    newest = mx.array([0], mx.uint32)
    long = 2 if A.lane_tree_ok() else 0
    parts = []
    # RC6's tiles of one shape share a block count; slices never straddle two tile shapes.
    groups = []
    for start, end, q_len in tiles(rows, n_keys):
        blocks = vector_blocks(q_len, n_keys)
        if groups and groups[-1][2] == blocks and groups[-1][1] == start:
            groups[-1] = (groups[-1][0], end, blocks)
        else:
            groups.append((start, end, blocks))
    for g0, g1, blocks in groups:
        for s0 in range(g0, g1, LAUNCH_ROWS):
            s1 = min(g1, s0 + LAUNCH_ROWS)
            n = s1 - s0
            pos0 = mx.array([hist + s0], mx.uint32)
            chains = n * A.HEADS * blocks
            po, pm, ps = A._kernel(_PARTIALS_NAME, _PARTIALS_P, A._HEADER)(
                inputs=[q[:, s0:s1], key, value, key[:, :, hist + s0:hist + s1], value[:, :, hist + s0:hist + s1],
                        profiles[:, s0:s1], sink[:, s0:s1], pos0, pos0, newest, sinkpos],
                template=[("T", mx.bfloat16), ("WINDOW", False), ("BLOCKS", blocks), ("THRESH", threshold),
                          ("SCALE", A.SCALE_BITS), ("ROWS", GROUP_ROWS), ("LONG", long)],
                grid=(32 * GROUP_ROWS * blocks, A.KV_HEADS, -(-n // GROUP_ROWS)),
                threadgroup=(32 * GROUP_ROWS, 1, 1),
                output_shapes=[(chains * A.HEAD_DIM,), (chains,), (chains,)],
                output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
            )
            (h,) = A._kernel(_MERGE_NAME, _MERGE_P)(
                inputs=[po, pm, ps, pos0, pos0],
                template=[("T", mx.bfloat16), ("WINDOW", False), ("BLOCKS", blocks), ("THRESH", threshold)],
                grid=(1024 * A.HEADS, n, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(1, n, A.HEADS * A.HEAD_DIM)],
                output_dtypes=[mx.bfloat16],
            )
            parts.append(h)
            if len(parts) % EVAL_LAUNCHES == 0:
                mx.eval(parts[-EVAL_LAUNCHES:])
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    return out.reshape(1, rows, A.HEADS, A.HEAD_DIM).transpose(0, 2, 1, 3)


# ----------------------------------------------------------------------------------------
# Prompt calls on MLX's full attention kernel (window visits, and full-history visits below the
# vector regime): the same calls as RC6's, merged into fewer and larger ones. MLX's full kernel
# gives a row the same bits whatever call it sits in, aligned to its query block or not
# (lanes/l6/probe-full-tiles*.json: 2,048 to 16,384 keys and window visits, calls of 32 to 512
# rows, every row equal), so only the call boundaries change: fewer calls each read the bank
# once and pad less to the kernel's 32-row query blocks. Rows RC6 sends to the vector kernel
# (a last tile of 8 rows or fewer) keep RC6's own call.
# ----------------------------------------------------------------------------------------

#: Largest relative-bias tensor one merged call writes (bf16: 32 MB; RC6 writes 8 MB a call).
MERGED_ELEMENTS = 16 * 1024 * 1024


def merged_tiles(rows, n_keys, heads=24):
    """RC6's tiles regrouped, [(first row, end row)], or None when merging saves no call."""
    rc6 = tiles(rows, n_keys, heads)
    step = rc6[0][2]
    if step <= 8:
        return None
    full_end = rows if rc6[-1][2] > 8 else rc6[-1][0]
    big = max(step, MERGED_ELEMENTS // (heads * n_keys))
    calls = [(s, min(full_end, s + big)) for s in range(0, full_end, big)]
    if len(calls) > 1 and calls[-1][1] - calls[-1][0] <= 8:
        last = calls.pop()
        calls[-1] = (calls[-1][0], last[1])
    if len(calls) >= sum(1 for t in rc6 if t[2] > 8):
        return None
    return calls + ([] if full_end == rows else [(full_end, rows)])


def merged_decline(query, key, value, profiles, sink, window):
    """None when this prompt call can run as merged full-kernel calls, else why not."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if query.dtype not in (mx.bfloat16, mx.float16) or any(x.dtype != query.dtype for x in (key, value, profiles, sink)):
        return "dtype"
    if merged_tiles(query.shape[2], key.shape[2] + 1, query.shape[1]) is None:
        return "no_merge"
    if not _merged_qualified(str(query.dtype), bool(window)):
        return "bits"
    return None


def merged_attention(query, key, value, profiles, sink, window):
    """RC6's `stock_attention` result for a prompt call, from fewer full-kernel calls."""
    from mtplx.models.inkling.attention import relative_bias

    batch, heads, rows, width = query.shape
    keys = key.shape[2]
    plan = merged_tiles(rows, keys + 1, heads)
    zero = mx.zeros((batch, key.shape[1], 1, width), key.dtype)
    key = mx.concatenate([key, zero], axis=2)
    value = mx.concatenate([value, zero], axis=2)
    parts = []
    for start, end in plan:
        bias = relative_bias(profiles[:, start:end], sink[:, start:end], keys, keys - rows + start, window)
        part = mx.fast.scaled_dot_product_attention(query[:, :, start:end], key, value, scale=width ** -0.5,
                                                    mask=bias)
        if len(plan) > 1:
            mx.eval(part)
        parts.append(part)
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)


@lru_cache(maxsize=8)
def _merged_qualified(dtype_name, window):
    """First use, per dtype and visit kind: merged calls against RC6's own tiles on a synthetic
    chunk of 200 rows (RC6: 77-row calls; merged: one call), every row bit for bit."""
    from mtplx.models.inkling.attention import stock_attention

    dtype = getattr(mx, dtype_name.split(".")[-1])
    hist, rows = (1023, 200) if window else (2048, 200)
    keys = hist + rows
    ks = mx.random.split(mx.random.key(1729), 5)
    q = (mx.random.normal((1, 24, rows, 128), key=ks[0]) * 0.6).astype(dtype)
    k = (mx.random.normal((1, 8, keys, 128), key=ks[1]) * 0.6).astype(dtype)
    v = mx.random.normal((1, 8, keys, 128), key=ks[2]).astype(dtype)
    prof = (mx.random.normal((1, rows, 24, 1024), key=ks[3]) * 0.5).astype(dtype)
    sink = mx.random.normal((1, rows, 24), key=ks[4]).astype(dtype)
    w = 1024 if window else 0
    try:
        ref = stock_attention(q, k, v, prof, sink, w)
        got = merged_attention(q, k, v, prof, sink, w)
        mx.eval(ref, got)
    except (RuntimeError, ValueError):
        return False
    view = mx.uint16
    return bool(mx.array_equal(ref.view(view), got.view(view)).item())
