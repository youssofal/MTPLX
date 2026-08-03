"""Fused residual-add + RMSNorm + MoE router GEMV for the Laguna S-2.1 decode step.

Ported from the mlx.fast **Laguna XS2.1** challenge kernel
``lagunaResidualRMSNormRouter`` (Sources/MLXFastModel/LagunaRuntimeModel.swift),
re-expressed as a Python ``mx.fast.metal_kernel`` and *adapted* to Laguna S-2.1,
which is a different model from the challenge's XS2.1:

    axis (hidden)   XS2.1 2048   ->  S2.1 3072
    router experts  256          ->  256          (unchanged)
    quant           NVFP4        ->  affine oQ4e  (router gate is BF16 in BOTH,
                                                   so this kernel is quant-agnostic)

The single dispatch it replaces on a sparse decoder layer is the pair

    hidden, normed = fused_add_rmsnorm(attn_out, hidden, post_ln_w, eps)   # kernels/fused_norm.py
    logits         = router_gemv_logits(normed, gate_weight)               # kernels/laguna_decode.py

i.e. the post-attention residual add + RMSNorm, immediately followed by the
router's ``[256, hidden]`` BF16 GEMV that consumes its output.  Nothing else can
overlap that GEMV — it is the very next link in the dependency chain — so fusing
it into the norm removes one kernel launch (and the round-trip re-read of the
normalised row) from the critical path of every one of the 47 sparse layers.

## Why the XS2.1 topology could NOT be copied verbatim

The donor kernel is a 512-thread / 16-simdgroup / ``n_reads == 4`` threadgroup,
which is exactly ``512 * 4 == 2048`` — its hidden size.  Its own source comments
warn that the ``(512 threads, n_reads 4)`` shape is load-bearing for the
bit-exactness of the FP32 RMS reduction and must not be moved.  At S2.1's 3072
that identity fails (``512 * 4 == 2048 != 3072``), so a verbatim port would
either read out of bounds or silently regroup the reduction.

The correct adaptation keeps ``n_reads == 4`` and widens the threadgroup to
``3072 / 4 == 768`` threads = **24 simdgroups**.  That is precisely the topology
of this repo's own single-pass ``_add_rmsnorm_exact_kernel`` for the 3072 axis,
so the residual/norm half of this kernel is **bit-identical** to the stock
``fused_add_rmsnorm`` it replaces (same rounding, same reduction order, same
``local_sums[32]`` gather with 8 of the 32 slots left zero).

``router_blocks = 3072 / 128 == 24`` (still divisible by 4, so the donor's 4x
weight-load unroll stays tail-free — it was 16 at 2048).

## Router numerics

Each active simdgroup owns one expert and walks the 3072-wide row in strict
``(block, i)`` order into a single FP32 accumulator (no tree regrouping), then a
``simd_shuffle_down`` butterfly and a round to BF16 -> FP32.  That reproduces the
*value* the stock ``self.gate(x)`` GEMV produces (``float(bf16(fp32_dot))``), but
NOT bit-for-bit against ``router_gemv_logits`` (which splits each dot across 8
simdgroups and sums the partials in a different order).  The consumer is an
argpartition top-k, so the bar here is top-k **selection parity**, verified in
the benchmark harness, not bitwise equality of the logits.

Callers check :func:`is_residual_router_eligible` first; the public helper falls
back to the stock two-op chain on any shape it does not cover.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


# The router half is only wired for the exact Laguna S-2.1 routing shape.
_EXPERTS = 256
_BLOCK_WIDTH = 128  # 32 lanes * n_reads(4); one simdgroup covers one block
_N_READS = 4


def is_residual_router_eligible(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    router_weight: mx.array,
    *,
    rows_per_group: int,
) -> bool:
    """Whether the fused residual+norm+router kernel covers this exact shape.

    Deliberately narrow, matching the stock pieces it fuses:
    ``fused_add_rmsnorm`` is bf16/fp16 only and needs an axis a 768-thread
    group covers in one N_READS==4 pass; ``router_gemv_logits`` is bf16 only.
    """

    if not mx.metal.is_available():
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    if router_weight.dtype != x.dtype:
        return False
    if tuple(x.shape) != tuple(residual.shape):
        return False
    if x.ndim < 2 or weight.ndim != 1 or router_weight.ndim != 2:
        return False
    axis = int(x.shape[-1])
    if axis != int(weight.shape[0]):
        return False
    # Exact-fit reduction: threads == axis / n_reads must be a whole number of
    # 32-lane simdgroups and fit one threadgroup, so the norm half stays
    # bit-identical to _add_rmsnorm_exact_kernel.
    if axis <= 0 or axis % (32 * _N_READS) != 0:
        return False
    threads = axis // _N_READS
    if threads > 1024:
        return False
    if axis % _BLOCK_WIDTH != 0 or (axis // _BLOCK_WIDTH) % 4 != 0:
        return False
    experts = int(router_weight.shape[0])
    if experts != _EXPERTS or int(router_weight.shape[1]) != axis:
        return False
    # rows_per_group picks the router tiling: one expert per active simdgroup,
    # so it must divide the experts evenly and not exceed the simdgroup count.
    if rows_per_group <= 0 or experts % rows_per_group != 0:
        return False
    if rows_per_group > threads // 32:
        return False
    return True


@lru_cache(maxsize=None)
def _residual_router_kernel(axis: int, experts: int, rows_per_group: int):
    tiles = experts // rows_per_group
    router_blocks = axis // _BLOCK_WIDTH
    header = f"""
        using namespace metal;
        constant constexpr uint AXIS = {axis};
        constant constexpr uint N_READS = {_N_READS};
        constant constexpr uint SIMD_SIZE = 32;
        constant constexpr uint EXPERTS = {experts};
        constant constexpr uint TILES = {tiles};
        constant constexpr uint ROWS_PER_GROUP = {rows_per_group};
        constant constexpr uint BLOCK_WIDTH = {_BLOCK_WIDTH};
        constant constexpr uint ROUTER_BLOCKS = {router_blocks};
    """

    # `normalized_row` stages the norm output in threadgroup memory so the
    # router GEMV never re-reads it from device. summed/normalized are written
    # once (tile 0); every tile recomputes the (cheap) norm and writes its own
    # disjoint slice of router_logits.
    source = """
        uint tg = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;

        uint row = tg / TILES;
        uint tile = tg - row * TILES;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];
        threadgroup T normalized_row[AXIS];

        size_t row_offset = size_t(row) * size_t(AXIS);
        uint base = lid * N_READS;

        // --- residual add + FP32 RMS statistic (bit-exact vs _add_rmsnorm_exact) ---
        T values[N_READS];
        float acc = 0.0f;
        for (uint i = 0; i < N_READS; ++i) {
            T value = x[row_offset + base + i] + residual[row_offset + base + i];
            values[i] = value;
            float fv = float(value);
            acc += fv * fv;
        }

        acc = simd_sum(acc);
        if (simd_group == 0) {
            local_sums[simd_lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            acc = simd_sum(local_sums[simd_lane]);
            if (simd_lane == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(acc / float(AXIS) + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = local_inv_mean[0];

        // --- normalize + weight; stage in threadgroup, publish once (tile 0) ---
        for (uint i = 0; i < N_READS; ++i) {
            T normed_v = weight[base + i] *
                static_cast<T>(float(values[i]) * inv);
            normalized_row[base + i] = normed_v;
            if (tile == 0) {
                summed[row_offset + base + i] = values[i];
                normalized[row_offset + base + i] = normed_v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- router GEMV: one expert per active simdgroup (rows_per_thread==1) ---
        if (simd_group < ROWS_PER_GROUP) {
            uint expert = tile * ROWS_PER_GROUP + simd_group;
            const device T* w_row = router_weight + size_t(expert) * size_t(AXIS);
            float router_acc = 0.0f;
            uint column = simd_lane * N_READS;
            // 4x unrolled block loads; ROUTER_BLOCKS % 4 == 0 so no tail.
            for (uint block = 0; block < ROUTER_BLOCKS; block += 4) {
                vec<T, 4> rw[4];
                for (uint u = 0; u < 4; ++u) {
                    const device vec<T, 4>* wv =
                        (const device vec<T, 4>*)(w_row + column + u * BLOCK_WIDTH);
                    rw[u] = wv[0];
                }
                for (uint u = 0; u < 4; ++u) {
                    uint col_u = column + u * BLOCK_WIDTH;
                    for (uint i = 0; i < 4; ++i) {
                        router_acc += float(rw[u][i]) *
                            float(normalized_row[col_u + i]);
                    }
                }
                column += 4 * BLOCK_WIDTH;
            }
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                router_acc += metal::simd_shuffle_down(router_acc, delta);
            }
            if (simd_lane == 0) {
                // Round to T then widen, exactly where the stock gemv rounds.
                float logit = float(static_cast<T>(router_acc));
                router_logits[size_t(row) * size_t(EXPERTS) + size_t(expert)] =
                    logit;
            }
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_residual_router_a{axis}_e{experts}_r{rows_per_group}",
        input_names=["x", "residual", "weight", "router_weight", "eps"],
        output_names=["summed", "normalized", "router_logits"],
        header=header,
        source=source,
    )


def fused_residual_norm_router(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    router_weight: mx.array,
    eps: float,
    *,
    rows_per_group: int = 8,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return ``(x + residual, rms_norm(x+residual)*weight, router_logits)``.

    ``router_logits`` is float32 ``[rows, experts]`` — the same dtype and value
    class as ``router_gemv_logits``.  Falls back to the stock two-op chain
    (``fused_add_rmsnorm`` then a bf16 gate matmul) on any unsupported shape, so
    callers can switch it on without owning a correctness branch.
    """

    if not is_residual_router_eligible(
        x, residual, weight, router_weight, rows_per_group=rows_per_group
    ):
        from .fused_norm import fused_add_rmsnorm

        h, normed = fused_add_rmsnorm(x, residual, weight, eps)
        logits = (normed.reshape(-1, normed.shape[-1]) @ router_weight.swapaxes(-1, -2))
        return h, normed, logits.astype(mx.float32)

    leading = x.shape[:-1]
    axis = int(x.shape[-1])
    experts = int(router_weight.shape[0])
    rows = 1
    for dim in leading:
        rows *= int(dim)
    x2 = x.reshape(rows, axis)
    residual2 = residual.reshape(rows, axis)

    threads = axis // _N_READS
    tiles = experts // rows_per_group
    kernel = _residual_router_kernel(axis, experts, rows_per_group)
    summed, normalized, logits = kernel(
        inputs=[x2, residual2, weight, router_weight, float(eps)],
        template=[("T", x.dtype)],
        grid=(threads * tiles * rows, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, axis), (rows, axis), (rows, experts)],
        output_dtypes=[x.dtype, x.dtype, mx.float32],
    )
    return (
        summed.reshape(*leading, axis),
        normalized.reshape(*leading, axis),
        logits,
    )
