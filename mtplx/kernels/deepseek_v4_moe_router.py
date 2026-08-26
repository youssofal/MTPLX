"""Fixed DeepSeek V4 sqrt-softplus MoE routing kernels.

The Mia/vLLM route computes the gate projection separately, then performs
sqrt-softplus, bias-only selection, top-6, unbiased weight gathering,
normalization, and route scaling in one bounded kernel.  Hash layers use the
same score arithmetic but take their six owners directly from ``tid2eid``.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_ROUTER_HEADER = r"""
    using namespace metal;

    inline float mtplx_sqrt_softplus(float value) {
        // Match vLLM's topkGatingSoftplusSqrt arithmetic.  The threshold is
        // part of the installed DeepSeek V4 router contract; using the usual
        // algebraically-equivalent stable form changes rounding before top-k.
        float softplus = value > 20.0f ? value : log1p(exp(value));
        return sqrt(softplus);
    }
"""


@lru_cache(maxsize=4)
def _router_gemm_kernel(experts: int, block_m: int):
    if int(experts) not in (64, 216) or int(block_m) not in (8, 64):
        raise ValueError("Mia router GEMM supports K64/K216 with BM8/BM64")
    simdgroups = int(block_m) // 8 * 2
    threads = simdgroups * 32
    header = f"""
        using namespace metal;
        constant constexpr uint EXPERTS = {int(experts)}u;
        constant constexpr uint BM = {int(block_m)}u;
        constant constexpr uint BN = 32u;
        constant constexpr uint BK = 32u;
        constant constexpr uint HIDDEN = 4096u;
        constant constexpr uint THREADS = {threads}u;
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint m_block = threadgroup_position_in_grid.z;
        uint n0 = threadgroup_position_in_grid.y * BN;
        uint row0 = m_block * BM;
        uint sg_m = sg / 2u;
        uint sg_n = (sg & 1u) * 16u;

        threadgroup T A_tile[BM * BK];
        threadgroup T B_tile[BK * BN];
        threadgroup float C_tile[BM * BN];
        simdgroup_matrix<T, 8, 8> a, b_left, b_right;
        simdgroup_matrix<float, 8, 8> c_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_right =
            simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0u; k0 < HIDDEN; k0 += BK) {
            for (uint index = tid; index < BM * BK; index += THREADS) {
                uint local_row = index / BK;
                uint local_k = index - local_row * BK;
                uint row = row0 + local_row;
                A_tile[index] = row < uint(rows)
                    ? x[size_t(row) * HIDDEN + k0 + local_k]
                    : T(0.0f);
            }
            for (uint index = tid; index < BK * BN; index += THREADS) {
                uint local_k = index / BN;
                uint local_n = index - local_k * BN;
                uint expert = n0 + local_n;
                B_tile[index] = expert < EXPERTS
                    ? weight[size_t(expert) * HIDDEN + k0 + local_k]
                    : T(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0u; ks < BK; ks += 8u) {
                simdgroup_load(a, A_tile + sg_m * 8u * BK + ks, BK);
                simdgroup_load(b_left, B_tile + ks * BN + sg_n, BN);
                simdgroup_load(b_right, B_tile + ks * BN + sg_n + 8u, BN);
                simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
                simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(c_left, C_tile + sg_m * 8u * BN + sg_n, BN);
        simdgroup_store(c_right, C_tile + sg_m * 8u * BN + sg_n + 8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < BM * BN; index += THREADS) {
            uint local_row = index / BN;
            uint local_n = index - local_row * BN;
            uint row = row0 + local_row;
            uint expert = n0 + local_n;
            if (row < uint(rows) && expert < EXPERTS) {
                logits[size_t(row) * EXPERTS + expert] = C_tile[index];
            }
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_mia_router_bf16_fp32_k4096_n{int(experts)}"
            f"_bm{int(block_m)}"
        ),
        input_names=["x", "weight", "rows"],
        output_names=["logits"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def install_router_projection(*, experts: int):
    """Bind the BF16 x BF16 to FP32 router projection for K64/K216."""
    experts = int(experts)
    if experts not in (64, 216):
        raise ValueError(f"Mia router projection requires K64 or K216, got {experts}")
    kernels = {
        block_m: _router_gemm_kernel(experts, block_m)
        for block_m in (8, 64)
    }

    def project(x: mx.array, weight: mx.array) -> mx.array:
        rows = int(x.shape[0])
        block_m = 8 if rows <= 32 else 64
        threads = block_m * 8
        return kernels[block_m](
            inputs=[mx.contiguous(x), mx.contiguous(weight), rows],
            template=[("T", x.dtype)],
            grid=(threads, (experts + 31) // 32, (rows + block_m - 1) // block_m),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, experts)],
            output_dtypes=[mx.float32],
        )[0]

    return project


@lru_cache(maxsize=2)
def _score_router_kernel(experts: int):
    source = r"""
        constexpr uint EXPERTS = MTPLX_ROUTER_EXPERTS;
        constexpr uint TOPK = 6u;

        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.z;
        threadgroup float candidate_scores[32u * TOPK];
        threadgroup int candidate_ids[32u * TOPK];

        float local_scores[TOPK];
        int local_ids[TOPK];
        for (uint slot = 0u; slot < TOPK; ++slot) {
            local_scores[slot] = -INFINITY;
            local_ids[slot] = -1;
        }

        for (uint expert = lane; expert < EXPERTS; expert += 32u) {
            float unbiased = mtplx_sqrt_softplus(
                float(logits[size_t(row) * EXPERTS + expert])
            );
            float choice = unbiased + float(correction[expert]);
            uint insert = TOPK;
            for (uint slot = 0u; slot < TOPK; ++slot) {
                if (choice > local_scores[slot]
                    || (choice == local_scores[slot]
                        && int(expert) < local_ids[slot])) {
                    insert = slot;
                    break;
                }
            }
            if (insert < TOPK) {
                for (uint slot = TOPK - 1u; slot > insert; --slot) {
                    local_scores[slot] = local_scores[slot - 1u];
                    local_ids[slot] = local_ids[slot - 1u];
                }
                local_scores[insert] = choice;
                local_ids[insert] = int(expert);
            }
        }

        for (uint slot = 0u; slot < TOPK; ++slot) {
            candidate_scores[lane * TOPK + slot] = local_scores[slot];
            candidate_ids[lane * TOPK + slot] = local_ids[slot];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            float best_scores[TOPK];
            int best_ids[TOPK];
            for (uint slot = 0u; slot < TOPK; ++slot) {
                best_scores[slot] = -INFINITY;
                best_ids[slot] = -1;
            }
            for (uint candidate = 0u; candidate < 32u * TOPK; ++candidate) {
                float choice = candidate_scores[candidate];
                int expert = candidate_ids[candidate];
                if (expert < 0) {
                    continue;
                }
                uint insert = TOPK;
                for (uint slot = 0u; slot < TOPK; ++slot) {
                    if (choice > best_scores[slot]
                        || (choice == best_scores[slot]
                            && expert < best_ids[slot])) {
                        insert = slot;
                        break;
                    }
                }
                if (insert < TOPK) {
                    for (uint slot = TOPK - 1u; slot > insert; --slot) {
                        best_scores[slot] = best_scores[slot - 1u];
                        best_ids[slot] = best_ids[slot - 1u];
                    }
                    best_scores[insert] = choice;
                    best_ids[insert] = expert;
                }
            }

            float selected[TOPK];
            float denominator = 0.0f;
            for (uint slot = 0u; slot < TOPK; ++slot) {
                int expert = best_ids[slot];
                selected[slot] = mtplx_sqrt_softplus(
                    float(logits[size_t(row) * EXPERTS + uint(expert)])
                );
                denominator += selected[slot];
            }
            float factor = float(route_scale) / max(denominator, 1.0e-20f);
            for (uint slot = 0u; slot < TOPK; ++slot) {
                size_t offset = size_t(row) * TOPK + slot;
                expert_ids[offset] = best_ids[slot];
                route_weights[offset] = selected[slot] * factor;
            }
        }
    """.replace("MTPLX_ROUTER_EXPERTS", f"{int(experts)}u")
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_mia_sqrtsoftplus_top6_k{int(experts)}",
        input_names=["logits", "correction", "route_scale"],
        output_names=["expert_ids", "route_weights"],
        header=_ROUTER_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _hash_router_kernel():
    source = r"""
        constexpr uint EXPERTS = 216u;
        constexpr uint TOPK = 6u;

        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.z;
        threadgroup float selected[TOPK];

        if (lane < TOPK) {
            int token = int(input_ids[row]);
            int expert = int(tid2eid[size_t(token) * TOPK + lane]);
            expert_ids[size_t(row) * TOPK + lane] = expert;
            selected[lane] = mtplx_sqrt_softplus(
                float(logits[size_t(row) * EXPERTS + uint(expert)])
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            float denominator = 0.0f;
            for (uint slot = 0u; slot < TOPK; ++slot) {
                denominator += selected[slot];
            }
            float factor = float(route_scale) / max(denominator, 1.0e-20f);
            for (uint slot = 0u; slot < TOPK; ++slot) {
                route_weights[size_t(row) * TOPK + slot] = selected[slot] * factor;
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_hash_sqrtsoftplus_top6_k216",
        input_names=["logits", "input_ids", "tid2eid", "route_scale"],
        output_names=["expert_ids", "route_weights"],
        header=_ROUTER_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def install_score_router(*, experts: int, topk: int, route_scale: float):
    """Bind the exact K216/top-6 score route at model construction."""
    if int(experts) not in (64, 216) or int(topk) != 6:
        raise ValueError(
            "Mia score router requires K64 or K216 and top-k=6; "
            f"got experts={experts}, topk={topk}"
        )
    kernel = _score_router_kernel(int(experts))
    scale = float(route_scale)

    def route(logits: mx.array, correction: mx.array, input_ids=None):
        del input_ids
        rows = int(logits.shape[0])
        return kernel(
            inputs=[mx.contiguous(logits), mx.contiguous(correction), scale],
            grid=(32, 1, rows),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, 6), (rows, 6)],
            output_dtypes=[mx.int32, mx.float32],
        )

    return route


def install_hash_router(*, experts: int, topk: int, route_scale: float):
    """Bind the exact K216/top-6 hash route at model construction."""
    if (int(experts), int(topk)) != (216, 6):
        raise ValueError(
            "Mia hash router requires 216 experts and top-k=6; "
            f"got experts={experts}, topk={topk}"
        )
    kernel = _hash_router_kernel()
    scale = float(route_scale)

    def route(logits: mx.array, tid2eid: mx.array, input_ids: mx.array):
        rows = int(logits.shape[0])
        return kernel(
            inputs=[
                mx.contiguous(logits),
                mx.contiguous(input_ids),
                mx.contiguous(tid2eid),
                scale,
            ],
            grid=(32, 1, rows),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, 6), (rows, 6)],
            output_dtypes=[mx.int32, mx.float32],
        )

    return route
