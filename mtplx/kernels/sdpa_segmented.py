"""Decode/verify attention over a list of KV segments (segmented KV cache).

A conversation's full-attention KV is a list of sealed segments plus one growing tail
(``mtplx/segmented_kv.py``). Attention over the list is split-softmax-merge, the way
``sdpa_nax_flash_dsplit`` already splits one buffer into key blocks: every block yields an
unnormalised partial output, a row sum and a row max, and one reduce merges them exactly
(global max, exp-sum, weighted sum). A segment is just more blocks.

Two kernels:

* the partial kernel is the head-dim-split TensorOps kernel of ``sdpa_nax_flash_dsplit``,
  derived from its source at import (``_fused_source``), launched ONCE for up to
  ``MAX_FUSED_SEGMENTS`` segments: grid.z walks the blocks of all segments, a ``meta`` array
  (block starts, row counts, capacities) selects the segment and its K/V buffers inside the
  kernel. Segments before the last one run with every key visible (no tail-causal mask);
  only the last segment, which holds the rows being verified, keeps the mask;
* the reduce kernel is ``_paged_reduce_kernel`` generalised to a list of partial buffers
  with arbitrary block counts (blocks are addressed through a ``starts`` prefix table).

Receipts (phase 1, 2026-10-07, Qwen3.8-27B shapes: 4 KV heads, 24 query heads, head_dim 256,
bf16; docs/onderzoek/2026-10-07-segmented-kv-prototype.md in the research branch):
segmented output equals the contiguous kernel's error against an fp32 reference (rel L2
2.8e-3, 1 segment bit-identical, more segments within one bf16 ulp of the contiguous
kernel); cost per extra segment at q_len 4 is 0 to 0.005 ms/layer fused up to 12 segments
(0.012 to 0.016 with one launch per segment); a verify forward over 6 or 20 segments at 50K
and 80K costs 0 to 1.1 % more than the contiguous one.

The kernel follows the contiguous dispatcher's configuration: halves, or quarters with
``MTPLX_NAX_FLASH_DSPLIT4=1``, for a window of ``q_len <= MAX_Q_LEN`` rows with ``gqa_factor *
q_len <= MAX_M_ROWS``, and the wide route's split and row groups (``wide_shape``) for a wider
window of up to 32 rows with ``MTPLX_NAX_FLASH_WIDE=1``, from 6 rows at GQA 6 (the contiguous
dispatcher gives 6 to 8 rows to another kernel; here they would run as two sub-windows, 5 rows
plus the rest, each a full walk over the keys). With the wide route off
``segmented_kv.decode_segments_attention`` runs a longer verify window as sub-windows within
both limits. A call no segment route takes (a mask, a kernel bail) runs the stock route on the
gathered rows (``segmented_kv.gathered_view``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

import mlx.core as mx

from ..nax_verify import nax_available
from . import sdpa_nax_flash_dsplit as _dsplit
from .sdpa_2pass import unnormalized_partials_dtype
from .sdpa_nax_flash import _HEADER

#: Head dimensions the segment kernel serves (the head-dim-split TensorOps kernel; ``D`` is a
#: template parameter, 128 gives 64 dims per simdgroup at the halves split). The contiguous
#: dispatcher in sdpa_nax_flash_dsplit stays 256 only: a model with head_dim 128 that does not
#: use segments keeps exactly the routes it has today.
SUPPORTED_HEAD_DIMS = (128, 256)

#: Metal allows 31 buffer bindings: queries + 2 per segment + meta + scale + 3 outputs.
MAX_FUSED_SEGMENTS = 12
#: Query rows of a window the shipping split takes whole: ``gqa_factor * q_len <= MAX_M_ROWS``
#: and ``q_len <= MAX_Q_LEN``; a wider window takes the wide route (``_route_shape``) or runs as
#: sub-windows within both limits (``segmented_kv.decode_segments_attention``).
MAX_M_ROWS = 32
MAX_Q_LEN = 10
#: The reduce kernel binds one partial buffer per launch group plus sums, maxs, starts, out.
MAX_GROUPS = 20

segmented_bail_counts: dict[str, int] = {}
segmented_dispatch_counts: dict[str, int] = {}


def _bail(reason: str):
    segmented_bail_counts[reason] = segmented_bail_counts.get(reason, 0) + 1
    if os.environ.get("MTPLX_NAX_FLASH_DEBUG"):
        print(f"[nax-flash-segments bail] {reason}")


_OLD_HEAD = """    const int block_idx = int(threadgroup_position_in_grid.z);
    const int n_blocks = int(blocks[0]);

    const int n_kv = static_cast<int>(offset[0]);"""


def _fused_source(nseg: int) -> str:
    """The partial kernel over ``nseg`` segment buffers in one launch.

    Same body as ``sdpa_nax_flash_dsplit._SOURCE``; only the prologue (segment lookup,
    buffer selection, per-segment mask flag) and the partial/stat indexing change, and the
    tail-causal mask is skipped for every segment before the last one (``nomask_seg``: a
    sealed segment lies entirely before the rows being verified; the stock kernel derives the
    mask from the same ``offset`` that bounds the keys, so it cannot be switched off through
    the inputs).
    ``meta`` = bstart[nseg+1] | n[nseg] | cap[nseg] (int32).
    """
    src = _dsplit._SOURCE
    if _OLD_HEAD not in src:
        raise RuntimeError("sdpa_nax_flash_dsplit source changed: update sdpa_segmented")
    sel = "".join(f"    if (seg == {i}) {{ keys = K{i}; values = V{i}; }}\n" for i in range(1, nseg))
    new_head = f"""    const int gblock = int(threadgroup_position_in_grid.z);
    const int B_total = meta[{nseg}];
    int seg = 0;
    while (seg + 1 < {nseg} && gblock >= meta[seg + 1]) seg++;
    const int block_idx = gblock - meta[seg];
    const int n_blocks = meta[seg + 1] - meta[seg];
    const int n_kv = meta[{nseg + 1} + seg];
    const int kcap = meta[{2 * nseg + 1} + seg];
    const bool nomask_seg = (seg < NLAST);
    const device InT* keys = K0;
    const device InT* values = V0;
{sel}"""
    src = src.replace(_OLD_HEAD, new_head)
    for old, new in (
        ("gp < kv_end && gp <= row_limit", "gp < kv_end && (nomask_seg || gp <= row_limit)"),
        ("((size_t)hq_row * n_blocks + block_idx) * D + db", "((size_t)hq_row * B_total + gblock) * D + db"),
        ("sums[hq_row * n_blocks + block_idx]", "sums[hq_row * B_total + gblock]"),
        ("maxs[hq_row * n_blocks + block_idx]", "maxs[hq_row * B_total + gblock]"),
    ):
        if old not in src:
            raise RuntimeError(f"sdpa_nax_flash_dsplit source changed ({old!r}): update sdpa_segmented")
        src = src.replace(old, new)
    return src


@lru_cache(maxsize=16)
def _fused_kernel(nseg: int):
    try:
        return mx.fast.metal_kernel(
            name=f"mtplx_nax_flash_dsplit_segments_{nseg}",
            input_names=["queries"] + [f"K{i}" for i in range(nseg)] + [f"V{i}" for i in range(nseg)] + ["meta", "scale"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER,
            source=_fused_source(nseg),
        )
    except Exception:  # noqa: BLE001 — toolchain without Metal4/MPP support
        return None


def _reduce_source(ngroups: int) -> str:
    """Merge kernel over ``ngroups`` partial buffers with per-buffer block counts.

    Same arithmetic as ``mtplx_sdpa_2pass_paged_reduce`` (global max over all blocks, global
    exp-sum, then the sum of ``exp(m_b - m) * partial_b`` divided by the exp-sum); only the
    block loop is generalised: ``starts`` (int32[ngroups+1]) holds the prefix sums of the
    per-buffer block counts, so counts need not be multiples of 32.
    """
    acc = []
    for s in range(ngroups):
        acc.append(f"""
        {{
            const int b0 = starts[{s}];
            const int nb = starts[{s + 1}] - b0;
            const auto pbase = partials{s} + (size_t)q_offset * nb * V + lane * ept;
            for (int lb = sg; lb < nb; lb += BN) {{
                U factor = fast::exp(maxs[b0 + lb] - max_score);
                for (int i = 0; i < ept; ++i) {{
                    o[i] += factor * static_cast<U>(pbase[(size_t)lb * V + i]);
                }}
            }}
        }}""")
    head = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int ept = V / BD;
        typedef float U;
        thread U o[ept] = {0};
        threadgroup U outputs[BN * BD];
        const int head_idx = threadgroup_position_in_grid.x;
        const int q_seq_idx = threadgroup_position_in_grid.y;
        const int q_offset = head_idx * threadgroups_per_grid.y + q_seq_idx;
        const int sg = simdgroup_index_in_threadgroup;
        const int lane = thread_index_in_simdgroup;
        const int btot = starts[NGROUPS];
        sums += (size_t)q_offset * btot;
        maxs += (size_t)q_offset * btot;
        out += q_offset * V + sg * ept;
        U sum_exp_score = 0.0f;
        U max_score = Limits<U>::finite_min;
        for (int b = lane; b < btot; b += BN) max_score = metal::max(max_score, maxs[b]);
        max_score = simd_max(max_score);
        for (int b = lane; b < btot; b += BN)
            sum_exp_score += fast::exp(maxs[b] - max_score) * sums[b];
        sum_exp_score = simd_sum(sum_exp_score);
    """.replace("NGROUPS", str(ngroups))
    tail = """
        for (int i = 0; i < ept; ++i) {
            outputs[lane * BD + sg] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[sg * BD + lane]);
            o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            for (int i = 0; i < ept; ++i) out[i] = static_cast<InT>(o[i]);
        }
    """
    return head + "".join(acc) + tail


@lru_cache(maxsize=32)
def _reduce_kernel(ngroups: int):
    try:
        return mx.fast.metal_kernel(
            name=f"mtplx_nax_flash_segments_reduce_{ngroups}",
            input_names=[f"partials{s}" for s in range(ngroups)] + ["sums", "maxs", "starts"],
            output_names=["out"],
            source=_reduce_source(ngroups),
        )
    except Exception:  # noqa: BLE001
        return None


def segment_block_counts(rows: Sequence[int], *, ndh: int = 2, wide_rows: int = 0) -> list[int]:
    """Key blocks per segment: the chunk the contiguous dispatcher uses for the same total.

    ``ndh`` 4 reads the quarter-split table (``MTPLX_NAX_FLASH_DSPLIT4``), ``wide_rows`` > 0 the
    wide route's table for that many live query rows per KV head.
    """
    total = sum(int(n) for n in rows)
    override = int(os.environ.get("MTPLX_NAX_FLASH_DSPLIT_BLOCKS", "0") or 0)
    if wide_rows:
        default = _dsplit._default_wide_blocks(total, wide_rows)
    elif ndh == 4:
        default = _dsplit._default_dsplit4_blocks(total)
    else:
        default = _dsplit._default_dsplit_blocks(total)
    base = override or default
    base = max(32, (base // 32) * 32)
    chunk = (-(-total // base) + 31) // 32 * 32
    return [max(1, -(-int(n) // chunk)) for n in rows]


def _route_shape(q_len: int, gqa_factor: int, head_dim: int = 256) -> tuple[bool, int, int] | None:
    """(wide, ndh, tg_m) the contiguous dispatcher would pick for this window, or None.

    Mirrors ``sdpa_nax_flash_dsplit._dsplit_dispatch`` and the call order in ``attention_split``:
    windows of up to 32 live rows per KV head take the shipping contract (halves, or quarters
    with ``MTPLX_NAX_FLASH_DSPLIT4=1`` unless the quarter self-check lane is off); a wider
    window of up to 32 query rows takes the wide contract (``wide_shape``: the same split choice,
    row groups) when ``MTPLX_NAX_FLASH_WIDE=1``. One segment then calls the contiguous wide
    entry with the same shape, so it stays bit-identical to the contiguous kernel.
    """
    live = gqa_factor * q_len
    if live <= 32 and 1 <= q_len <= 10:
        # head_dim 128 keeps the halves split (quarters would leave 32 dims per simdgroup, untuned).
        ndh = 4 if head_dim == 256 and os.environ.get("MTPLX_NAX_FLASH_DSPLIT4", "0") == "1" else 2
        if ndh == 4:
            from ..kernel_selfcheck import lane_disabled

            if lane_disabled("nax_flash_dsplit4_sdpa"):
                ndh = 2
        return False, ndh, (live + 15) // 16
    if (
        # From 2 rows: a decode row past the shipping contract (GQA above 32) stays unsupported.
        2 <= q_len <= _dsplit.WIDE_MAX_Q
        and live <= _dsplit.WIDE_MAX_ROWS
        and os.environ.get("MTPLX_NAX_FLASH_WIDE", "0").strip().lower() in {"1", "true", "yes", "on"}
    ):
        from ..kernel_selfcheck import lane_disabled

        if lane_disabled("nax_flash_dsplit_wide_sdpa"):
            return None
        # head_dim 128 keeps the halves split here too.
        ndh, tg_m = _dsplit.wide_shape(live, quarters=head_dim == 256)
        return True, ndh, tg_m
    return None


def sdpa_nax_flash_dsplit_segments(
    *,
    queries: mx.array,
    segments: Sequence[tuple[mx.array, mx.array, int]],
    scale: float,
) -> mx.array | None:
    """Exact attention of ``queries`` over ``segments`` = [(keys, values, n), ...], oldest first.

    ``keys``/``values`` are (1, Hk, capacity, D) buffers, D in SUPPORTED_HEAD_DIMS, of which the first ``n`` rows are
    live. Every segment but the last is fully visible; the last holds the rows being verified
    as its final ``q_len`` rows and keeps the tail-causal mask. None when the contract is not
    met (the caller then runs the stock route, see ``segmented_kv.decode_segments_attention``).
    """
    if os.environ.get("MTPLX_NAX_FLASH_DSPLIT", "1") == "0":
        return _bail("env_disabled")
    if not mx.metal.is_available():
        return _bail("metal_unavailable")
    if not nax_available():
        return _bail("gpu_family_or_os")
    if queries.ndim != 4:
        return _bail("ndim")
    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1 or d not in SUPPORTED_HEAD_DIMS:
        return _bail("shape_gate")
    if not segments:
        return _bail("no_segments")
    hk = int(segments[0][0].shape[1])
    if hk <= 0 or hq % hk:
        return _bail("gqa_heads")
    gqa_factor = hq // hk
    shape = _route_shape(q_len, gqa_factor, d)
    if shape is None:
        return _bail("route_shape")
    wide, ndh, tg_m = shape
    live = gqa_factor * q_len
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return _bail("query_dtype")
    for keys, values, n in segments:
        if keys.dtype != queries.dtype or values.dtype != queries.dtype:
            return _bail("kv_dtype_mismatch")
        if (
            keys.ndim != 4
            or int(keys.shape[1]) != hk
            or tuple(values.shape) != tuple(keys.shape)
            or int(keys.shape[3]) != d
        ):
            return _bail("kv_layout_mismatch")
        if int(n) <= 0 or int(n) > int(keys.shape[2]):
            return _bail("segment_rows")
    if int(segments[-1][2]) < q_len:
        return _bail("tail_shorter_than_q")

    if len(segments) == 1 and d == 256:
        # One segment is the contiguous kernel's input (bit-identical). The contiguous
        # dispatcher is 256 only; a head_dim-128 segment goes through the fused path below.
        keys, values, n = segments[0]
        entry = _dsplit.sdpa_nax_flash_dsplit_wide if wide else _dsplit.sdpa_nax_flash_dsplit
        return entry(queries=queries, keys=keys, values=values, offset=int(n), scale=scale)

    groups = [segments[i:i + MAX_FUSED_SEGMENTS] for i in range(0, len(segments), MAX_FUSED_SEGMENTS)]
    if len(groups) > MAX_GROUPS:
        return _bail("too_many_segments")
    rows = [int(n) for _, _, n in segments]
    blocks = segment_block_counts(rows, ndh=ndh, wide_rows=live if wide else 0)
    reduce_kernel = _reduce_kernel(len(groups))
    if reduce_kernel is None:
        return _bail("kernel_unavailable")

    nthreads = 32 * tg_m * ndh
    row_groups = (live + 16 * tg_m - 1) // (16 * tg_m)
    pdt = unnormalized_partials_dtype(queries.dtype)
    parts, sums, maxs, counts = [], [], [], []
    first = 0
    for gi, group in enumerate(groups):
        nseg = len(group)
        kernel = _fused_kernel(nseg)
        if kernel is None:
            return _bail("kernel_unavailable")
        nbs = blocks[first:first + nseg]
        first += nseg
        bstart = [0]
        for nb in nbs:
            bstart.append(bstart[-1] + nb)
        total_blocks = bstart[-1]
        meta = mx.array(
            bstart + [int(n) for _, _, n in group] + [int(k.shape[2]) for k, _, _ in group],
            dtype=mx.int32,
        )
        last_group = gi == len(groups) - 1
        try:
            p, s_, m_ = kernel(
                inputs=[queries] + [k for k, _, _ in group] + [v for _, v, _ in group] + [meta, float(scale)],
                template=[
                    ("InT", queries.dtype),
                    ("PartT", pdt),
                    ("D", d),
                    ("QL", q_len),
                    ("GQA_F", gqa_factor),
                    ("TG_M", tg_m),
                    ("NDH", ndh),
                    ("NLAST", nseg - 1 if last_group else nseg),
                ],
                grid=(hk * nthreads, row_groups, total_blocks),
                threadgroup=(nthreads, 1, 1),
                output_shapes=[
                    (bsz, hq, q_len, total_blocks, d),
                    (bsz, hq, q_len, total_blocks),
                    (bsz, hq, q_len, total_blocks),
                ],
                output_dtypes=[pdt, mx.float32, mx.float32],
            )
        except Exception as exc:  # noqa: BLE001 — dispatch/compile failure => gather fallback
            return _bail(f"dispatch_failed: {type(exc).__name__}: {str(exc)[:2000]}")
        parts.append(p)
        sums.append(s_)
        maxs.append(m_)
        counts.append(total_blocks)

    sums_all = sums[0] if len(sums) == 1 else mx.concatenate(sums, axis=3)
    maxs_all = maxs[0] if len(maxs) == 1 else mx.concatenate(maxs, axis=3)
    starts = [0]
    for c in counts:
        starts.append(starts[-1] + c)
    (out,) = reduce_kernel(
        inputs=parts + [sums_all, maxs_all, mx.array(starts, dtype=mx.int32)],
        template=[("InT", queries.dtype), ("V", d)],
        grid=(bsz * hq * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(bsz, hq, q_len, d)],
        output_dtypes=[queries.dtype],
    )
    segmented_dispatch_counts["dispatched"] = segmented_dispatch_counts.get("dispatched", 0) + 1
    return out


def segments_supported(q_len: int, gqa_factor: int, head_dim: int) -> bool:
    """Static part of the contract (no device work), for the route gate in attention_split."""
    return head_dim in SUPPORTED_HEAD_DIMS and _route_shape(q_len, gqa_factor, head_dim) is not None
