"""Two-pass Metal SDPA probe for tiny Qwen3Next verify windows.

This is an exact-attention diagnostic kernel adapted for MTPLX's verify shapes:
``B=1``, GQA full-attention layers, ``q_len`` usually depth+1, and long cached
KV.  It keeps the product path opt-in because reduction order differs from MLX's
stock fused SDPA and must pass acceptance parity before any serving claim.
"""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import mlx.core as mx


def _compute_blocks(gqa_factor: int, n_kv: int) -> int:
    override = (os.environ.get("MTPLX_SDPA_2PASS_BLOCKS") or "").strip()
    if override:
        try:
            blocks = int(override)
        except ValueError:
            blocks = 0
        if blocks > 0:
            return int(blocks)
    arch = str(mx.device_info().get("architecture", ""))
    devc = arch[-1] if arch else ""
    n_simds = int(gqa_factor)
    n = int(n_kv)
    if devc == "d":
        blocks = 128
        if n_simds <= 2 and n > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n < 65536:
                blocks = 512
            elif n >= 65536:
                blocks = 1024
    elif devc == "s":
        blocks = 64
        if n > 1024 and n_simds > 4:
            if n <= 8192:
                blocks = 128
            elif n <= 32768:
                blocks = 256
            elif n <= 65536:
                blocks = 512
            else:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    return int(blocks)


@lru_cache(maxsize=None)
def _partials_kernel(*, has_mask: bool):
    if not mx.metal.is_available():
        return None

    mask_setup = ""
    mask_use_key = ""
    mask_score = ""
    mask_advance = ""
    inputs = [
        "queries",
        "keys",
        "values",
        "gqa_factor",
        "N",
        "k_head_stride",
        "k_seq_stride",
        "v_head_stride",
        "v_seq_stride",
        "scale",
        "blocks",
    ]
    if has_mask:
        inputs.append("mask")
        mask_setup = """
        auto mask_ = mask + (((b_idx * Hq + q_head_idx) * M_FIXED + q_seq_idx) * N + block_idx);
        """
        mask_use_key = """
            auto mask_value = static_cast<float>(mask_[0]);
            use_key = use_key && (mask_value >= Limits<InT>::finite_min);
        """
        mask_score = """
                score += static_cast<float>(mask_[0]);
        """
        mask_advance = """
            mask_ += blocks;
        """

    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        auto q_head_idx = threadgroup_position_in_grid.x;
        auto b_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto q_seq_idx = thread_position_in_threadgroup.z;
        auto simd_lid = thread_index_in_simdgroup;

        auto Hq = threadgroups_per_grid.x;
        auto hk_idx = q_head_idx / gqa_factor;
        auto q_batch_head_idx = b_idx * Hq + q_head_idx;
        auto o_offset = q_batch_head_idx * M_FIXED + q_seq_idx;

        auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
        auto k_ = keys + ((b_idx * Hk + hk_idx) * k_head_stride) + block_idx * k_seq_stride + simd_lid * qk_per_thread;
        auto v_ = values + ((b_idx * Hk + hk_idx) * v_head_stride) + block_idx * v_seq_stride + simd_lid * v_per_thread;

        partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;
        {mask_setup}

        thread float q[qk_per_thread];
        thread float o[v_per_thread];
        threadgroup InT tg_k[BD * qk_per_thread];
        threadgroup InT tg_v[BD * v_per_thread];

        for (int i = 0; i < qk_per_thread; ++i) {{
            q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
        }}
        for (int i = 0; i < v_per_thread; ++i) {{
            o[i] = 0.0f;
        }}

        float max_score = Limits<float>::finite_min;
        float sum_exp_score = 0.0f;

        for (int n = block_idx; n < N; n += blocks) {{
            if (q_seq_idx == 0) {{
                for (int i = 0; i < qk_per_thread; ++i) {{
                    tg_k[simd_lid * qk_per_thread + i] = k_[i];
                }}
                for (int i = 0; i < v_per_thread; ++i) {{
                    tg_v[simd_lid * v_per_thread + i] = v_[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool use_key = (n <= (N - M_FIXED + q_seq_idx));
            {mask_use_key}

            if (use_key) {{
                float score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {{
                    score += q[i] * static_cast<float>(tg_k[simd_lid * qk_per_thread + i]);
                }}
                score = simd_sum(score);
                {mask_score}

                float new_max = metal::max(max_score, score);
                float factor = fast::exp(max_score - new_max);
                float exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int i = 0; i < v_per_thread; ++i) {{
                    o[i] = o[i] * factor + exp_score * static_cast<float>(tg_v[simd_lid * v_per_thread + i]);
                }}
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            k_ += blocks * int(k_seq_stride);
            v_ += blocks * int(v_seq_stride);
            {mask_advance}
        }}

        if (simd_lid == 0) {{
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }}
        for (int i = 0; i < v_per_thread; ++i) {{
            partials[i] = static_cast<InT>(o[i]);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_sdpa_2pass_partials{'_mask' if has_mask else ''}",
        input_names=inputs,
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


@lru_cache(maxsize=1)
def _reduce_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = V / BD;

        auto head_idx = threadgroup_position_in_grid.x;
        auto q_seq_idx = threadgroup_position_in_grid.y;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto q_offset = head_idx * M_FIXED + q_seq_idx;
        partials += (q_offset * blocks + simd_gid) * V + simd_lid * elem_per_thread;
        sums += q_offset * blocks;
        maxs += q_offset * blocks;
        out += q_offset * V + simd_gid * elem_per_thread;

        thread float o[elem_per_thread];
        threadgroup float outputs[BN * BD];

        for (int i = 0; i < elem_per_thread; ++i) {
            o[i] = 0.0f;
        }

        float sum_exp_score = 0.0f;
        float max_score = Limits<float>::finite_min;

        for (int b = 0; b < blocks / BN; ++b) {
            max_score = metal::max(max_score, maxs[simd_lid + BN * b]);
        }
        max_score = simd_max(max_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
            sum_exp_score += factor * sums[simd_lid + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_gid] - max_score);
            for (int i = 0; i < elem_per_thread; ++i) {
                o[i] += factor * static_cast<float>(partials[i]);
            }
            maxs += BN;
            partials += BN * V;
        }

        for (int i = 0; i < elem_per_thread; ++i) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; ++i) {
                out[i] = static_cast<InT>(o[i]);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_2pass_reduce",
        input_names=["partials", "sums", "maxs", "blocks"],
        output_names=["out"],
        source=source,
    )


def _causal_mask_for_tail(mask: Any, *, bsz: int, hq: int, q_len: int, n_kv: int, dtype: mx.Dtype):
    if mask is None or (isinstance(mask, str) and mask == "causal"):
        return None
    if mask.dtype == mx.bool_:
        converted = mx.where(
            mask,
            mx.zeros(mask.shape, dtype=dtype),
            mx.full(mask.shape, mx.finfo(dtype).min, dtype=dtype),
        )
    else:
        converted = mask.astype(dtype) if mask.dtype != dtype else mask
    return mx.contiguous(mx.broadcast_to(converted, (bsz, hq, q_len, n_kv)))


def sdpa_2pass_tail(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Any = None,
    max_q_len: int = 16,
) -> mx.array | None:
    """Return causal SDPA for the newest ``q_len`` rows, or ``None`` if unsupported."""

    if not mx.metal.is_available():
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    bsz, hq, q_len, d = queries.shape
    if q_len <= 0 or q_len > int(max_q_len):
        return None
    _, hk, n_kv, kd = keys.shape
    if int(kd) != int(d):
        return None
    vdim = int(values.shape[-1])
    input_type = queries.dtype
    if input_type not in (mx.bfloat16, mx.float16):
        return None
    if int(d) not in {128, 256} or vdim not in {128, 256} or int(d) != vdim:
        return None
    if int(hk) <= 0 or int(hq) % int(hk):
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)

    gqa_factor = int(hq) // int(hk)
    blocks = _compute_blocks(gqa_factor, int(n_kv))
    if blocks <= 0 or blocks % 32:
        return None

    mask_tensor = _causal_mask_for_tail(
        mask,
        bsz=int(bsz),
        hq=int(hq),
        q_len=int(q_len),
        n_kv=int(n_kv),
        dtype=input_type,
    )
    kernel = _partials_kernel(has_mask=mask_tensor is not None)
    reduce_kernel = _reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return None

    inputs = [
        queries,
        keys,
        values,
        int(gqa_factor),
        int(n_kv),
        int(keys.shape[2] * keys.shape[3]),
        int(keys.shape[3]),
        int(values.shape[2] * values.shape[3]),
        int(values.shape[3]),
        float(scale),
        int(blocks),
    ]
    if mask_tensor is not None:
        inputs.append(mask_tensor)

    partial_shape = (int(bsz) * int(hq), int(q_len), int(blocks), int(vdim))
    stats_shape = (int(bsz) * int(hq), int(q_len), int(blocks))
    partials, sums, maxs = kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("D", int(d)),
            ("V", int(vdim)),
            ("Hk", int(hk)),
            ("M_FIXED", int(q_len)),
        ],
        grid=(int(hq) * 32, int(bsz), int(blocks) * int(q_len)),
        threadgroup=(32, 1, int(q_len)),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[input_type, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", input_type),
            ("V", int(vdim)),
            ("M_FIXED", int(q_len)),
        ],
        grid=((int(bsz) * int(hq)) * 1024, int(q_len), 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[input_type],
    )
    return out


@lru_cache(maxsize=None)
def _partials_kernel_rowoffs():
    """Partials kernel with PER-ROW causal bounds from an int32[B] offsets
    vector (the ragged-batch lane): valid keys for row b, query j are
    ``n <= offsets[b] + j``, the row's pre-write logical length plus the
    causal diagonal into the newly written q-block. No mask tensor at all,
    so the [B, Hq, q, N] mask materialization of the masked variant never
    happens."""
    if not mx.metal.is_available():
        return None
    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        auto q_head_idx = threadgroup_position_in_grid.x;
        auto b_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto q_seq_idx = thread_position_in_threadgroup.z;
        auto simd_lid = thread_index_in_simdgroup;

        auto Hq = threadgroups_per_grid.x;
        auto hk_idx = q_head_idx / gqa_factor;
        auto q_batch_head_idx = b_idx * Hq + q_head_idx;
        auto o_offset = q_batch_head_idx * M_FIXED + q_seq_idx;

        auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
        auto k_ = keys + ((b_idx * Hk + hk_idx) * k_head_stride) + block_idx * k_seq_stride + simd_lid * qk_per_thread;
        auto v_ = values + ((b_idx * Hk + hk_idx) * v_head_stride) + block_idx * v_seq_stride + simd_lid * v_per_thread;

        partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;

        const int row_bound = offsets[b_idx] + int(q_seq_idx);

        thread float q[qk_per_thread];
        thread float o[v_per_thread];
        threadgroup InT tg_k[BD * qk_per_thread];
        threadgroup InT tg_v[BD * v_per_thread];

        for (int i = 0; i < qk_per_thread; ++i) {
            q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
        }
        for (int i = 0; i < v_per_thread; ++i) {
            o[i] = 0.0f;
        }

        float max_score = Limits<float>::finite_min;
        float sum_exp_score = 0.0f;

        for (int n = block_idx; n < N; n += blocks) {
            if (q_seq_idx == 0) {
                for (int i = 0; i < qk_per_thread; ++i) {
                    tg_k[simd_lid * qk_per_thread + i] = k_[i];
                }
                for (int i = 0; i < v_per_thread; ++i) {
                    tg_v[simd_lid * v_per_thread + i] = v_[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool use_key = (n <= row_bound);

            if (use_key) {
                float score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score += q[i] * static_cast<float>(tg_k[simd_lid * qk_per_thread + i]);
                }
                score = simd_sum(score);

                float new_max = metal::max(max_score, score);
                float factor = fast::exp(max_score - new_max);
                float exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int i = 0; i < v_per_thread; ++i) {
                    o[i] = o[i] * factor + exp_score * static_cast<float>(tg_v[simd_lid * v_per_thread + i]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            k_ += blocks * int(k_seq_stride);
            v_ += blocks * int(v_seq_stride);
        }

        if (simd_lid == 0) {
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }
        for (int i = 0; i < v_per_thread; ++i) {
            partials[i] = static_cast<InT>(o[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_2pass_partials_rowoffs",
        input_names=[
            "queries",
            "keys",
            "values",
            "offsets",
            "gqa_factor",
            "N",
            "k_head_stride",
            "k_seq_stride",
            "v_head_stride",
            "v_seq_stride",
            "scale",
            "blocks",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def sdpa_2pass_tail_rowoffs(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    offsets: mx.array,
    max_q_len: int = 32,
) -> mx.array | None:
    """Ragged-batch exact SDPA: per-row causal bounds, streaming K/V, no mask
    tensor. ``keys``/``values`` are the FULL padded ragged buffers
    ``[B, Hk, cap, D]``; ``offsets`` is the int32[B] PRE-write logical length
    per row (the ragged cache's offsets before ``update_and_fetch``)."""
    if not mx.metal.is_available():
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    bsz, hq, q_len, d = queries.shape
    if q_len <= 0 or q_len > int(max_q_len) or 32 * int(q_len) > 1024:
        return None
    _, hk, n_kv, kd = keys.shape
    if int(kd) != int(d):
        return None
    vdim = int(values.shape[-1])
    input_type = queries.dtype
    if input_type not in (mx.bfloat16, mx.float16):
        return None
    if int(d) not in {128, 256} or vdim not in {128, 256} or int(d) != vdim:
        return None
    if int(hk) <= 0 or int(hq) % int(hk):
        return None
    if offsets.ndim != 1 or int(offsets.shape[0]) != int(bsz):
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)
    offsets = mx.contiguous(offsets.astype(mx.int32))

    gqa_factor = int(hq) // int(hk)
    blocks = _compute_blocks(gqa_factor, int(n_kv))
    if blocks <= 0 or blocks % 32:
        return None

    kernel = _partials_kernel_rowoffs()
    reduce_kernel = _reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return None

    partial_shape = (int(bsz) * int(hq), int(q_len), int(blocks), int(vdim))
    stats_shape = (int(bsz) * int(hq), int(q_len), int(blocks))
    partials, sums, maxs = kernel(
        inputs=[
            queries,
            keys,
            values,
            offsets,
            int(gqa_factor),
            int(n_kv),
            int(keys.shape[2] * keys.shape[3]),
            int(keys.shape[3]),
            int(values.shape[2] * values.shape[3]),
            int(values.shape[3]),
            float(scale),
            int(blocks),
        ],
        template=[
            ("InT", input_type),
            ("D", int(d)),
            ("V", int(vdim)),
            ("Hk", int(hk)),
            ("M_FIXED", int(q_len)),
        ],
        grid=(int(hq) * 32, int(bsz), int(blocks) * int(q_len)),
        threadgroup=(32, 1, int(q_len)),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[input_type, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", input_type),
            ("V", int(vdim)),
            ("M_FIXED", int(q_len)),
        ],
        grid=((int(bsz) * int(hq)) * 1024, int(q_len), 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[input_type],
    )
    return out


@lru_cache(maxsize=None)
def _partials_kernel_rowoffs_kvshared():
    """Ragged-batch partials kernel with KV tiles staged ONCE PER KV-HEAD.

    The stock 2pass (and MLX's fused vector path) grid per Q-head, so the same
    KV tiles are loaded once per GQA head (6x duplicate traffic at 24/4 GQA).
    This variant grids per KV-head: one threadgroup stages each K/V tile into
    threadgroup memory and ALL (gqa x q_len) simdgroups consume it. Per-row
    causal bounds via the offsets vector (ragged lane). Threadgroup is
    (32, GQA, M_FIXED) = 32*GQA*M_FIXED threads (<= 1024 for GQA=6, q<=5).
    Tile is TILE tokens of K and V (TILE*D*2*2 bytes threadgroup memory)."""
    if not mx.metal.is_available():
        return None
    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;
        constexpr int TILE = 32;

        auto hk_idx = threadgroup_position_in_grid.x;
        auto b_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto gqa_idx = thread_position_in_threadgroup.y;
        auto q_seq_idx = thread_position_in_threadgroup.z;
        auto simd_lid = thread_index_in_simdgroup;

        auto Hk = threadgroups_per_grid.x;
        auto Hq = Hk * GQA;
        auto q_head_idx = hk_idx * GQA + gqa_idx;
        auto q_batch_head_idx = b_idx * Hq + q_head_idx;
        auto o_offset = q_batch_head_idx * M_FIXED + q_seq_idx;

        auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
        auto k_base = keys + ((b_idx * Hk + hk_idx) * k_head_stride);
        auto v_base = values + ((b_idx * Hk + hk_idx) * v_head_stride);

        partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;

        const int row_bound = offsets[b_idx] + int(q_seq_idx);

        thread float q[qk_per_thread];
        thread float o[v_per_thread];
        threadgroup InT tg_k[TILE * D];
        threadgroup InT tg_v[TILE * V];

        for (int i = 0; i < qk_per_thread; ++i) {
            q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
        }
        for (int i = 0; i < v_per_thread; ++i) {
            o[i] = 0.0f;
        }

        float max_score = Limits<float>::finite_min;
        float sum_exp_score = 0.0f;

        // Each block owns a strided sequence of TILE-token tiles.
        for (int t0 = block_idx * TILE; t0 < N; t0 += blocks * TILE) {
            // Stage TILE tokens of K and V cooperatively: the first q-row's
            // simdgroups (z==0) load, one token per gqa lane group.
            {
                const int sg_id = int(gqa_idx) + int(q_seq_idx) * GQA;
                for (int tok = sg_id; tok < TILE; tok += GQA * M_FIXED) {
                    auto ksrc = k_base + (t0 + tok) * k_seq_stride + simd_lid * qk_per_thread;
                    auto vsrc = v_base + (t0 + tok) * v_seq_stride + simd_lid * v_per_thread;
                    bool in_range = (t0 + tok) < N;
                    for (int i = 0; i < qk_per_thread; ++i) {
                        tg_k[tok * D + simd_lid * qk_per_thread + i] =
                            in_range ? ksrc[i] : InT(0);
                    }
                    for (int i = 0; i < v_per_thread; ++i) {
                        tg_v[tok * V + simd_lid * v_per_thread + i] =
                            in_range ? vsrc[i] : InT(0);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // v3a: two tokens per iteration, independent partial dots and
            // paired simd_sums overlap the reduction latency that serialized
            // the v2 loop; the online-softmax update handles both results.
            for (int tok = 0; tok < TILE; tok += 2) {
                int n0 = t0 + tok;
                int n1 = t0 + tok + 1;
                bool use0 = (n0 <= row_bound) && (n0 < N);
                bool use1 = (n1 <= row_bound) && (n1 < N);
                if (!use0 && !use1) { continue; }
                float s0 = 0.0f;
                float s1 = 0.0f;
                auto k0 = tg_k + tok * D + simd_lid * qk_per_thread;
                auto k1 = k0 + D;
                for (int i = 0; i < qk_per_thread; ++i) {
                    float qi = q[i];
                    s0 += qi * static_cast<float>(k0[i]);
                    s1 += qi * static_cast<float>(k1[i]);
                }
                s0 = simd_sum(s0);
                s1 = simd_sum(s1);

                float m01 = metal::max(
                    use0 ? s0 : Limits<float>::finite_min,
                    use1 ? s1 : Limits<float>::finite_min);
                float new_max = metal::max(max_score, m01);
                float factor = fast::exp(max_score - new_max);
                float e0 = use0 ? fast::exp(s0 - new_max) : 0.0f;
                float e1 = use1 ? fast::exp(s1 - new_max) : 0.0f;

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + e0 + e1;
                auto v0 = tg_v + tok * V + simd_lid * v_per_thread;
                auto v1 = v0 + V;
                for (int i = 0; i < v_per_thread; ++i) {
                    o[i] = o[i] * factor + e0 * static_cast<float>(v0[i])
                                         + e1 * static_cast<float>(v1[i]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }
        for (int i = 0; i < v_per_thread; ++i) {
            partials[i] = static_cast<InT>(o[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_2pass_partials_rowoffs_kvshared",
        input_names=[
            "queries",
            "keys",
            "values",
            "offsets",
            "N",
            "k_head_stride",
            "k_seq_stride",
            "v_head_stride",
            "v_seq_stride",
            "scale",
            "blocks",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def sdpa_2pass_tail_rowoffs_kvshared(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    offsets: mx.array,
    max_q_len: int = 5,
) -> mx.array | None:
    """KV-shared ragged SDPA: tiles staged once per KV-head, consumed by all
    gqa x q_len rows. Threadgroup (32, GQA, q_len) caps q_len at
    1024/(32*GQA), 5 for GQA=6."""
    if not mx.metal.is_available():
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    bsz, hq, q_len, d = queries.shape
    _, hk, n_kv, kd = keys.shape
    if int(hk) <= 0 or int(hq) % int(hk):
        return None
    gqa = int(hq) // int(hk)
    if q_len <= 0 or q_len > int(max_q_len) or 32 * gqa * int(q_len) > 1024:
        return None
    if int(kd) != int(d):
        return None
    vdim = int(values.shape[-1])
    input_type = queries.dtype
    if input_type not in (mx.bfloat16, mx.float16):
        return None
    if int(d) not in {128, 256} or vdim not in {128, 256} or int(d) != vdim:
        return None
    if offsets.ndim != 1 or int(offsets.shape[0]) != int(bsz):
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)
    offsets = mx.contiguous(offsets.astype(mx.int32))

    _b_override = (os.environ.get("MTPLX_KVSHARED_BLOCKS") or "").strip()
    blocks = int(_b_override) if _b_override else min(64, _compute_blocks(gqa, int(n_kv)))
    if blocks <= 0 or blocks % 32:
        return None

    kernel = _partials_kernel_rowoffs_kvshared()
    reduce_kernel = _reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return None

    partial_shape = (int(bsz) * int(hq), int(q_len), int(blocks), int(vdim))
    stats_shape = (int(bsz) * int(hq), int(q_len), int(blocks))
    partials, sums, maxs = kernel(
        inputs=[
            queries,
            keys,
            values,
            offsets,
            int(n_kv),
            int(keys.shape[2] * keys.shape[3]),
            int(keys.shape[3]),
            int(values.shape[2] * values.shape[3]),
            int(values.shape[3]),
            float(scale),
            int(blocks),
        ],
        template=[
            ("InT", input_type),
            ("D", int(d)),
            ("V", int(vdim)),
            ("GQA", gqa),
            ("M_FIXED", int(q_len)),
        ],
        grid=(int(hk) * 32, int(bsz) * gqa, int(blocks) * int(q_len)),
        threadgroup=(32, gqa, int(q_len)),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[input_type, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", input_type),
            ("V", int(vdim)),
            ("M_FIXED", int(q_len)),
        ],
        grid=((int(bsz) * int(hq)) * 1024, int(q_len), 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[input_type],
    )
    return out
