"""Stable counting sort for the MoE route table, ported to Laguna S-2.1.

Port of the mlx.fast **Laguna XS2.1** challenge kernel
``DARKBLOOM_ROUTE_COUNTING_SORT`` (the ``routeTileHistKernel`` /
``routeScanKernel`` / ``routeScatterKernel`` chain in
``Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift``), re-expressed
as three Python ``mx.fast.metal_kernel`` dispatches.

## What it replaces

The MoE MLP sorts the flattened top-k route table before the gathered expert
GEMM so ``gather_qmm(sorted_indices=True)`` sees contiguous per-expert runs.
Upstream that sort is ``mx.argsort(indices.flatten())``.  The route table is
pure integer data — uint32 expert-ids in ``[0, 256)`` — so it is a natural
counting-sort target that is entirely **quant/layout-agnostic**: it ports to
affine oQ4e S-2.1 exactly as it ran on NVFP4 XS2.1, no numeric surface at all.

## The three dispatches

1. ``hist``   — per-tile histogram.  ``TILE=128`` keys per threadgroup, counted
   into ``threadgroup atomic_uint counts[256]``; emits ``tile_hist[tiles, 256]``.
2. ``scan``   — one 256-thread group: total per key over all tiles, then an
   exclusive scan over the 256 keys (serial on lane 0) -> ``base[256]``.
3. ``scatter``— one thread per (tile, key): rank base = global ``base[k]`` plus
   counts of key ``k`` in earlier tiles, then walk this tile's 128 keys **in
   input order**, appending each matching index.  Stability is by construction
   (one writer per output slot, write order == input order), so the emitted
   permutation reproduces a *stable* argsort for every input, not just tested
   ones.

## Stability / argsort-identity

The scatter is stable by construction.  Whether the emitted ``order`` is
*bit-identical* to ``mx.argsort(keys)`` depends on whether ``mx.argsort`` is
itself stable on this key domain — that is an empirical question answered by the
check harness (``scratchpad_route_csort_check.py``), not an assumption baked in
here.  Either way ``keys[order]`` is the fully-sorted key multiset and ``order``
is a valid permutation, which is all ``gather_qmm(sorted_indices=True)`` needs.

Callers gate on :func:`is_route_csort_eligible` first; :func:`route_counting_sort`
falls back to ``mx.argsort`` on any shape/dtype it does not cover.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


# Fixed by the port: 128 keys per histogram tile, 256 expert bins.
_TILE = 128
_EXPERTS = 256


def is_route_csort_eligible(keys: mx.array) -> bool:
    """Whether the counting-sort chain covers this route table.

    Deliberately narrow, matching the kernel's hard assumptions: a 1-D uint32
    key vector whose length is a whole number of 128-key tiles, with keys in
    ``[0, 256)`` (the histogram indexes ``counts[key]`` directly).  Decode
    (M=10) is not a whole tile and falls back — that path is trivial anyway.
    """

    if not mx.metal.is_available():
        return False
    if keys.dtype != mx.uint32:
        return False
    if keys.ndim != 1:
        return False
    n = int(keys.shape[0])
    if n <= 0 or n % _TILE != 0:
        return False
    return True


@lru_cache(maxsize=None)
def _hist_kernel(tile: int, experts: int):
    header = f"""
        using namespace metal;
        constant constexpr uint TILE = {tile};
        constant constexpr uint EXPERTS = {experts};
    """
    source = """
        uint t = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup atomic_uint counts[EXPERTS];
        atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
        atomic_store_explicit(&counts[lid + TILE], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint key = keys[t * TILE + lid];
        atomic_fetch_add_explicit(&counts[key], 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_hist[t * EXPERTS + lid] =
            atomic_load_explicit(&counts[lid], memory_order_relaxed);
        tile_hist[t * EXPERTS + lid + TILE] =
            atomic_load_explicit(&counts[lid + TILE], memory_order_relaxed);
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_route_csort_hist_t{tile}_e{experts}",
        input_names=["keys"],
        output_names=["tile_hist"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _scan_kernel(experts: int):
    header = f"""
        using namespace metal;
        constant constexpr uint EXPERTS = {experts};
    """
    source = """
        uint k = thread_position_in_threadgroup.x;
        uint nt = uint(tiles);
        uint total = 0;
        for (uint t = 0; t < nt; ++t) {
            total += tile_hist[t * EXPERTS + k];
        }
        threadgroup uint totals[EXPERTS];
        totals[k] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k == 0) {
            uint acc = 0;
            for (uint i = 0; i < EXPERTS; ++i) {
                uint c = totals[i];
                totals[i] = acc;
                acc += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        base[k] = totals[k];
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_route_csort_scan_e{experts}",
        input_names=["tile_hist", "tiles"],
        output_names=["base"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _scatter_kernel(tile: int, experts: int):
    header = f"""
        using namespace metal;
        constant constexpr uint TILE = {tile};
        constant constexpr uint EXPERTS = {experts};
    """
    source = """
        uint t = threadgroup_position_in_grid.x;
        uint k = thread_position_in_threadgroup.x;
        // Rank base for key k in tile t: global base + counts in earlier tiles.
        uint off = base[k];
        for (uint tp = 0; tp < t; ++tp) {
            off += tile_hist[tp * EXPERTS + k];
        }
        // Walk this tile's slice in input order: stability by construction.
        for (uint i = 0; i < TILE; ++i) {
            uint idx = t * TILE + i;
            if (keys[idx] == k) {
                order[off++] = idx;
            }
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_route_csort_scatter_t{tile}_e{experts}",
        input_names=["keys", "tile_hist", "base"],
        output_names=["order"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def route_counting_sort(keys: mx.array) -> mx.array:
    """Return the sort ``order`` (uint32 permutation) of ``keys``.

    ``keys[route_counting_sort(keys)]`` is the sorted key multiset; the
    permutation is stable (ties keep input order).  Falls back to
    ``mx.argsort(keys)`` on any shape/dtype outside
    :func:`is_route_csort_eligible`, so callers can switch it on without owning
    a correctness branch.
    """

    if not is_route_csort_eligible(keys):
        return mx.argsort(keys)

    keys = mx.contiguous(keys)
    n = int(keys.shape[0])
    tiles = n // _TILE

    hist = _hist_kernel(_TILE, _EXPERTS)(
        inputs=[keys],
        grid=(tiles * _TILE, 1, 1),
        threadgroup=(_TILE, 1, 1),
        output_shapes=[(tiles * _EXPERTS,)],
        output_dtypes=[mx.uint32],
    )[0]

    base = _scan_kernel(_EXPERTS)(
        inputs=[hist, int(tiles)],
        grid=(_EXPERTS, 1, 1),
        threadgroup=(_EXPERTS, 1, 1),
        output_shapes=[(_EXPERTS,)],
        output_dtypes=[mx.uint32],
    )[0]

    order = _scatter_kernel(_TILE, _EXPERTS)(
        inputs=[keys, hist, base],
        grid=(tiles * _EXPERTS, 1, 1),
        threadgroup=(_EXPERTS, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.uint32],
    )[0]

    return order
