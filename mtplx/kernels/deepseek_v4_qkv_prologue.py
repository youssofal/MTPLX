"""Construction-bound Mia DeepSeek-V4 Q/KV prologue kernels.

The pinned source has two relevant dispatch boundaries.  The first applies the
learned RMS weights to the 1024-wide query rank and 512-wide latent projection.
The second keeps per-head Q normalization and Q/KV RoPE in FP32, stores Q once
as BF16, and finalizes KV directly into the exact 432-byte NVFP4 record.

This module owns only arithmetic and bounded record outputs.  Persistent cache
frontiers and slot writes remain owned by :mod:`mtplx.deepseek_v4_nvfp4_kv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable

import mlx.core as mx

from mtplx.deepseek_v4_nvfp4_kv import (
    MIA_DSPARK_CONTEXT_ROWS,
    MIA_NVFP4_HEAD_DIM,
    MIA_NVFP4_RECORD_BYTES,
    MIA_NVFP4_ROPE_DIM,
    _NVFP4_HEADER,
)


MIA_Q_RANK = 1024
MIA_Q_HEADS = 64
MIA_QKV_PROJECTION_WIDTH = 1536
MIA_DSPARK_PROPOSAL_ROWS = 5
MIA_TARGET_DECODE_ROWS = (1, 6)
MIA_TARGET_PREFILL_TILE_ROWS = 1024
MIA_LEARNED_NORM_THREADS = 256
MIA_QKV_FINALIZE_THREADS = 256
MIA_QKV_SIMDGROUPS = 8
MIA_QKV_SLOTS = MIA_Q_HEADS + 1


_PROLOGUE_HEADER = _NVFP4_HEADER + r"""
    inline float mtplx_bf16_roundtrip(float value) {
        return float(bfloat(value));
    }
"""


@lru_cache(maxsize=1)
def _learned_qkv_norm_kernel():
    source = r"""
        uint task = threadgroup_position_in_grid.x;
        uint row = task >> 1u;
        bool is_q_rank = (task & 1u) == 0u;
        if (row >= uint(rows)) return;

        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd = simdgroup_index_in_threadgroup;
        uint width = is_q_rank ? 1024u : 512u;
        const device T* input = projection + size_t(row) * 1536u
            + (is_q_rank ? 0u : 1024u);
        const device T* weight = is_q_rank ? q_weight : kv_weight;
        device T* output = is_q_rank
            ? q_rank_norm + size_t(row) * 1024u
            : kv_norm + size_t(row) * 512u;

        threadgroup float simd_sums[8];
        threadgroup float rrms_shared;
        float local_sq = 0.0f;
        for (uint dim = tid; dim < width; dim += 256u) {
            float value = float(input[dim]);
            local_sq += value * value;
        }
        float partial_sum = simd_sum(local_sq);
        if (lane == 0u) {
            simd_sums[simd] = partial_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total = 0.0f;
            for (uint group = 0u; group < 8u; ++group) {
                total += simd_sums[group];
            }
            rrms_shared = rsqrt(total / float(width) + float(rms_eps));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rrms = rrms_shared;
        for (uint dim = tid; dim < width; dim += 256u) {
            float normalized = float(input[dim]) * rrms
                * float(weight[dim]);
            output[dim] = T(normalized);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_learned_qkv_rms",
        input_names=[
            "projection",
            "q_weight",
            "kv_weight",
            "rows",
            "rms_eps",
        ],
        output_names=["q_rank_norm", "kv_norm"],
        source=source,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=1)
def _learned_kv_norm_kernel():
    source = r"""
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(rows)) return;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd = simdgroup_index_in_threadgroup;
        const device T* input = projection + size_t(row) * 1536u + 1024u;
        device T* output = kv_norm + size_t(row) * 512u;

        threadgroup float simd_sums[8];
        threadgroup float rrms_shared;
        float local_sq = 0.0f;
        for (uint dim = tid; dim < 512u; dim += 256u) {
            float value = float(input[dim]);
            local_sq += value * value;
        }
        float partial_sum = simd_sum(local_sq);
        if (lane == 0u) simd_sums[simd] = partial_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total = 0.0f;
            for (uint group = 0u; group < 8u; ++group) total += simd_sums[group];
            rrms_shared = rsqrt(total / 512.0f + float(rms_eps));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint dim = tid; dim < 512u; dim += 256u) {
            output[dim] = T(float(input[dim]) * rrms_shared * float(kv_weight[dim]));
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_learned_kv_rms",
        input_names=["projection", "kv_weight", "rows", "rms_eps"],
        output_names=["kv_norm"],
        source=source,
        ensure_row_contiguous=False,
    )


_QKV_RECORD_BODY = r"""
    uint dim_base = lane * 16u;
    bool is_kv = slot == 64u;

    float elements[16];
    if (is_kv) {
        const device T* input = kv_norm + size_t(row) * 512u + dim_base;
        for (uint i = 0u; i < 16u; ++i) {
            elements[i] = float(input[i]);
        }
    } else {
        size_t q_offset = (size_t(row) * 64u + slot) * 512u;
        const device T* input = q_pre + q_offset + dim_base;
        float local_sq = 0.0f;
        for (uint i = 0u; i < 16u; ++i) {
            elements[i] = float(input[i]);
            local_sq += elements[i] * elements[i];
        }
        float rms_rcp = rsqrt(simd_sum(local_sq) / 512.0f + float(rms_eps));
        for (uint i = 0u; i < 16u; ++i) {
            float normalized = elements[i] * rms_rcp;
            float rotated = normalized;
            uint dim = dim_base + i;
            if (dim >= 448u) {
                uint local = dim - 448u;
                uint pair = local >> 1u;
                uint pair_base = i & ~1u;
                float even = elements[pair_base] * rms_rcp;
                float odd = elements[pair_base + 1u] * rms_rcp;
                float c = float(rope_cos[size_t(row) * 32u + pair]);
                float s = float(rope_sin[size_t(row) * 32u + pair]);
                rotated = (local & 1u) == 0u
                    ? even * c - odd * s
                    : even * s + odd * c;
            }
            q_out[q_offset + dim] = T(rotated);
        }
    }

    if (is_kv) {
        float rope_elements[16];
        for (uint i = 0u; i < 16u; ++i) {
            elements[i] = mtplx_bf16_roundtrip(elements[i]);
        }
        if (dim_base >= 448u) {
            for (uint i = 0u; i < 16u; i += 2u) {
                uint local = dim_base + i - 448u;
                uint pair = local >> 1u;
                float even = elements[i];
                float odd = elements[i + 1u];
                float c = float(rope_cos[size_t(row) * 32u + pair]);
                float s = float(rope_sin[size_t(row) * 32u + pair]);
                elements[i] = mtplx_bf16_roundtrip(even * c - odd * s);
                elements[i + 1u] = mtplx_bf16_roundtrip(even * s + odd * c);
                rope_elements[i] = elements[i];
                rope_elements[i + 1u] = elements[i + 1u];
            }
        }

        float group_max = 0.0f;
        for (uint i = 0u; i < 16u; ++i) {
            group_max = max(group_max, abs(elements[i]));
        }
        float bounded_max = max(group_max, 6.0f * 0x1p-126f);
        uchar scale_byte = mtplx_e4m3_encode_positive(bounded_max / 6.0f);
        float scale = mtplx_e4m3_decode(scale_byte);
        float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        device uchar* record = records + size_t(row) * 432u;
        for (uint packed = 0u; packed < 8u; ++packed) {
            uchar low = mtplx_e2m1_encode(elements[packed * 2u] * inverse);
            uchar high = mtplx_e2m1_encode(
                elements[packed * 2u + 1u] * inverse
            );
            record[lane * 8u + packed] = uchar(low | uchar(high << 4));
        }
        record[256u + lane] = scale_byte;
        if (lane < 16u) {
            record[288u + lane] = uchar(0);
        }
        if (lane >= 28u) {
            for (uint i = 0u; i < 16u; ++i) {
                uint rope_dim = dim_base + i - 448u;
                uint rope_byte = rope_dim * 2u;
                ushort bits = as_type<ushort>(bfloat(rope_elements[i]));
                record[304u + rope_byte] = uchar(bits & 0xffu);
                record[304u + rope_byte + 1u] = uchar(bits >> 8u);
            }
        }
    }
"""


def _qkv_record_source(*, prefill: bool) -> str:
    if prefill:
        return (
            r"""
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    for (uint slot = simdgroup_index_in_threadgroup;
         slot < 65u; slot += 8u) {
"""
            + _QKV_RECORD_BODY
            + r"""
    }
"""
        )
    return (
        r"""
    uint task = threadgroup_position_in_grid.x * 8u
        + simdgroup_index_in_threadgroup;
    if (task >= uint(rows) * 65u) return;
    uint row = task / 65u;
    uint slot = task - row * 65u;
    uint lane = thread_index_in_simdgroup;
"""
        + _QKV_RECORD_BODY
    )


@lru_cache(maxsize=2)
def _qkv_record_kernel(*, prefill: bool):
    return mx.fast.metal_kernel(
        name=(
            "mtplx_dsv4_mia_qnorm_rope_kv_stock432_prefill"
            if prefill
            else "mtplx_dsv4_mia_qnorm_rope_kv_stock432_decode"
        ),
        input_names=[
            "q_pre",
            "kv_norm",
            "rope_cos",
            "rope_sin",
            "rows",
            "rms_eps",
        ],
        output_names=["q_out", "records"],
        header=_PROLOGUE_HEADER,
        source=_qkv_record_source(prefill=prefill),
        ensure_row_contiguous=False,
    )


_KV_RECORD_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x * 8u
        + simdgroup_index_in_threadgroup;
    if (row >= uint(rows)) return;
    uint lane = thread_index_in_simdgroup;
    uint dim_base = lane * 16u;
    const device T* input = kv_norm + size_t(row) * 512u + dim_base;
    float elements[16];
    for (uint i = 0u; i < 16u; ++i) {
        elements[i] = float(input[i]);
    }
    float rope_elements[16];
    for (uint i = 0u; i < 16u; ++i) {
        elements[i] = mtplx_bf16_roundtrip(elements[i]);
    }
    if (dim_base >= 448u) {
        for (uint i = 0u; i < 16u; i += 2u) {
            uint local = dim_base + i - 448u;
            uint pair = local >> 1u;
            float even = elements[i];
            float odd = elements[i + 1u];
            float c = float(rope_cos[size_t(row) * 32u + pair]);
            float s = float(rope_sin[size_t(row) * 32u + pair]);
            elements[i] = mtplx_bf16_roundtrip(even * c - odd * s);
            elements[i + 1u] = mtplx_bf16_roundtrip(even * s + odd * c);
            rope_elements[i] = elements[i];
            rope_elements[i + 1u] = elements[i + 1u];
        }
    }

    float group_max = 0.0f;
    for (uint i = 0u; i < 16u; ++i) {
        group_max = max(group_max, abs(elements[i]));
    }
    float bounded_max = max(group_max, 6.0f * 0x1p-126f);
    uchar scale_byte = mtplx_e4m3_encode_positive(bounded_max / 6.0f);
    float scale = mtplx_e4m3_decode(scale_byte);
    float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    device uchar* record = records + size_t(row) * 432u;
    for (uint packed = 0u; packed < 8u; ++packed) {
        uchar low = mtplx_e2m1_encode(elements[packed * 2u] * inverse);
        uchar high = mtplx_e2m1_encode(elements[packed * 2u + 1u] * inverse);
        record[lane * 8u + packed] = uchar(low | uchar(high << 4));
    }
    record[256u + lane] = scale_byte;
    if (lane < 16u) {
        record[288u + lane] = uchar(0);
    }
    if (lane >= 28u) {
        for (uint i = 0u; i < 16u; ++i) {
            uint rope_dim = dim_base + i - 448u;
            uint rope_byte = rope_dim * 2u;
            ushort bits = as_type<ushort>(bfloat(rope_elements[i]));
            record[304u + rope_byte] = uchar(bits & 0xffu);
            record[304u + rope_byte + 1u] = uchar(bits >> 8u);
        }
    }
"""


@lru_cache(maxsize=1)
def _kv_record_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_kv_rope_stock432",
        input_names=["kv_norm", "rope_cos", "rope_sin", "rows"],
        output_names=["records"],
        header=_PROLOGUE_HEADER,
        source=_KV_RECORD_SOURCE,
        ensure_row_contiguous=False,
    )


def _row_count(prefix: tuple[int, ...]) -> int:
    rows = 1
    for size in prefix:
        rows *= int(size)
    return rows


def _run_learned_qkv_norm(
    projection: mx.array,
    q_weight: mx.array,
    kv_weight: mx.array,
    *,
    kernel,
    rms_eps: float,
) -> tuple[mx.array, mx.array]:
    prefix = tuple(int(size) for size in projection.shape[:-1])
    rows = _row_count(prefix)
    return tuple(
        kernel(
            inputs=[
                projection,
                q_weight,
                kv_weight,
                rows,
                float(rms_eps),
            ],
            template=[("T", mx.bfloat16)],
            grid=(rows * 2 * MIA_LEARNED_NORM_THREADS, 1, 1),
            threadgroup=(MIA_LEARNED_NORM_THREADS, 1, 1),
            output_shapes=[
                (*prefix, MIA_Q_RANK),
                (*prefix, MIA_NVFP4_HEAD_DIM),
            ],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
    )


def _run_learned_kv_norm(
    projection: mx.array,
    kv_weight: mx.array,
    *,
    kernel,
    rms_eps: float,
) -> mx.array:
    prefix = tuple(int(size) for size in projection.shape[:-1])
    rows = _row_count(prefix)
    return kernel(
        inputs=[projection, kv_weight, rows, float(rms_eps)],
        template=[("T", mx.bfloat16)],
        grid=(rows * MIA_LEARNED_NORM_THREADS, 1, 1),
        threadgroup=(MIA_LEARNED_NORM_THREADS, 1, 1),
        output_shapes=[(*prefix, MIA_NVFP4_HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _run_qkv_records(
    q_pre: mx.array,
    kv_norm: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    *,
    kernel,
    rows: int,
    prefill: bool,
    rms_eps: float,
) -> tuple[mx.array, mx.array]:
    prefix = tuple(int(size) for size in q_pre.shape[:-2])
    threadgroups = (
        int(rows)
        if prefill
        else (int(rows) * MIA_QKV_SLOTS + MIA_QKV_SIMDGROUPS - 1)
        // MIA_QKV_SIMDGROUPS
    )
    return tuple(
        kernel(
            inputs=[
                q_pre,
                kv_norm,
                rope_cos,
                rope_sin,
                int(rows),
                float(rms_eps),
            ],
            template=[("T", mx.bfloat16)],
            grid=(threadgroups * MIA_QKV_FINALIZE_THREADS, 1, 1),
            threadgroup=(MIA_QKV_FINALIZE_THREADS, 1, 1),
            output_shapes=[q_pre.shape, (*prefix, MIA_NVFP4_RECORD_BYTES)],
            output_dtypes=[mx.bfloat16, mx.uint8],
        )
    )


def _run_target_qkv_records(
    q_pre: mx.array,
    kv_norm: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    *,
    decode_kernel,
    rms_eps: float,
) -> tuple[mx.array, mx.array]:
    rows = _row_count(tuple(int(size) for size in q_pre.shape[:-2]))
    return _run_qkv_records(
        q_pre,
        kv_norm,
        rope_cos,
        rope_sin,
        kernel=decode_kernel,
        rows=rows,
        prefill=False,
        rms_eps=rms_eps,
    )


def _run_prefill_qkv_records(
    q_pre: mx.array,
    kv_norm: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    *,
    full_kernel,
    reduced_kernel,
    rms_eps: float,
) -> tuple[mx.array, mx.array]:
    rows = _row_count(tuple(int(size) for size in q_pre.shape[:-2]))
    reduced = rows == MIA_TARGET_PREFILL_TILE_ROWS
    return _run_qkv_records(
        q_pre,
        kv_norm,
        rope_cos,
        rope_sin,
        kernel=reduced_kernel if reduced else full_kernel,
        rows=rows,
        prefill=reduced,
        rms_eps=rms_eps,
    )


def _run_k5_proposal_records(
    q_pre: mx.array,
    kv_norm: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    *,
    kernel,
    rms_eps: float,
) -> tuple[mx.array, mx.array]:
    return _run_qkv_records(
        q_pre,
        kv_norm,
        rope_cos,
        rope_sin,
        kernel=kernel,
        rows=MIA_DSPARK_PROPOSAL_ROWS,
        prefill=False,
        rms_eps=rms_eps,
    )


def _run_context_kv_records(
    kv_norm: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    *,
    kernel,
) -> mx.array:
    prefix = tuple(int(size) for size in kv_norm.shape[:-1])
    rows = _row_count(prefix)
    return kernel(
        inputs=[
            kv_norm,
            rope_cos,
            rope_sin,
            rows,
        ],
        template=[("T", mx.bfloat16)],
        grid=(
            (
                (rows + MIA_QKV_SIMDGROUPS - 1)
                // MIA_QKV_SIMDGROUPS
            )
            * MIA_QKV_FINALIZE_THREADS,
            1,
            1,
        ),
        threadgroup=(MIA_QKV_FINALIZE_THREADS, 1, 1),
        output_shapes=[(*prefix, MIA_NVFP4_RECORD_BYTES)],
        output_dtypes=[mx.uint8],
    )[0]


@dataclass(frozen=True)
class MiaQKVPrologue:
    """Prebound exact target/draft prologue entrypoints."""

    learned_norm: Callable
    kv_norm: Callable
    target_records: Callable
    prefill_records: Callable
    proposal_records: Callable
    context_records: Callable
    q_rank: int
    heads: int
    head_dim: int
    rope_dim: int
    proposal_rows: int
    context_rows: int
    prefill_tile_rows: int

    @property
    def geometry(self) -> dict[str, int | tuple[int, int]]:
        return {
            "q_rank": self.q_rank,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "rope_dim": self.rope_dim,
            "target_decode_rows": MIA_TARGET_DECODE_ROWS,
            "proposal_rows": self.proposal_rows,
            "context_rows": self.context_rows,
            "prefill_tile_rows": self.prefill_tile_rows,
        }


def _project_learned_norms(values, *, projection_owner, learned_norm):
    return learned_norm(projection_owner.project_fused(values))


def _project_kv_norm(values, *, projection_owner, kv_norm):
    return kv_norm(projection_owner.project_fused(values))


@dataclass(frozen=True)
class MiaBoundQKVPrologue:
    """One attention's weight- and projection-bound exact Q/KV plan."""

    raw: MiaQKVPrologue
    projection_owner: object
    q_weight: mx.array
    kv_weight: mx.array
    rms_eps: float
    project_learned: Callable
    project_kv: Callable
    target_records: Callable
    prefill_records: Callable
    proposal_records: Callable
    context_records: Callable


def bind_mia_qkv_prologue(
    raw: MiaQKVPrologue,
    *,
    projection_owner,
    q_weight: mx.array,
    kv_weight: mx.array,
    rms_eps: float,
) -> MiaBoundQKVPrologue:
    """Prove weights/owner once and return hot callables taking arrays only."""
    if int(getattr(projection_owner, "split", -1)) != MIA_Q_RANK or not callable(
        getattr(projection_owner, "project_fused", None)
    ):
        raise ValueError("Mia Q/KV prologue requires the fused 1024/512 owner")
    for label, weight, shape in (
        ("query", q_weight, (MIA_Q_RANK,)),
        ("KV", kv_weight, (MIA_NVFP4_HEAD_DIM,)),
    ):
        if tuple(int(size) for size in getattr(weight, "shape", ())) != shape:
            raise ValueError(f"Mia {label} learned norm weight shape changed")
        if getattr(weight, "dtype", None) != mx.bfloat16:
            raise ValueError(f"Mia {label} learned norm weight must be BF16")
    eps = float(rms_eps)
    learned_norm = partial(
        raw.learned_norm,
        q_weight=q_weight,
        kv_weight=kv_weight,
        rms_eps=eps,
    )
    kv_norm = partial(raw.kv_norm, kv_weight=kv_weight, rms_eps=eps)
    return MiaBoundQKVPrologue(
        raw=raw,
        projection_owner=projection_owner,
        q_weight=q_weight,
        kv_weight=kv_weight,
        rms_eps=eps,
        project_learned=partial(
            _project_learned_norms,
            projection_owner=projection_owner,
            learned_norm=learned_norm,
        ),
        project_kv=partial(
            _project_kv_norm,
            projection_owner=projection_owner,
            kv_norm=kv_norm,
        ),
        target_records=partial(raw.target_records, rms_eps=eps),
        prefill_records=partial(raw.prefill_records, rms_eps=eps),
        proposal_records=partial(raw.proposal_records, rms_eps=eps),
        context_records=raw.context_records,
    )


def install_mia_qkv_prologue(
    *,
    q_rank: int,
    heads: int,
    head_dim: int,
    rope_dim: int,
    proposal_rows: int,
    context_rows: int,
    prefill_tile_rows: int,
) -> MiaQKVPrologue:
    """Validate the pinned TP1 geometry once and bind every kernel variant."""
    observed = (
        int(q_rank),
        int(heads),
        int(head_dim),
        int(rope_dim),
        int(proposal_rows),
        int(context_rows),
        int(prefill_tile_rows),
    )
    expected = (
        MIA_Q_RANK,
        MIA_Q_HEADS,
        MIA_NVFP4_HEAD_DIM,
        MIA_NVFP4_ROPE_DIM,
        MIA_DSPARK_PROPOSAL_ROWS,
        MIA_DSPARK_CONTEXT_ROWS,
        MIA_TARGET_PREFILL_TILE_ROWS,
    )
    if observed != expected:
        raise ValueError(f"unsupported Mia Q/KV prologue geometry: {observed}")
    if not mx.metal.is_available():
        raise RuntimeError("Mia Q/KV prologue installation requires Metal")

    learned_kernel = _learned_qkv_norm_kernel()
    kv_learned_kernel = _learned_kv_norm_kernel()
    qkv_decode_kernel = _qkv_record_kernel(prefill=False)
    qkv_prefill_kernel = _qkv_record_kernel(prefill=True)
    context_kernel = _kv_record_kernel()
    return MiaQKVPrologue(
        learned_norm=partial(_run_learned_qkv_norm, kernel=learned_kernel),
        kv_norm=partial(_run_learned_kv_norm, kernel=kv_learned_kernel),
        target_records=partial(
            _run_target_qkv_records,
            decode_kernel=qkv_decode_kernel,
        ),
        prefill_records=partial(
            _run_prefill_qkv_records,
            full_kernel=qkv_decode_kernel,
            reduced_kernel=qkv_prefill_kernel,
        ),
        proposal_records=partial(
            _run_k5_proposal_records,
            kernel=qkv_decode_kernel,
        ),
        context_records=partial(_run_context_kv_records, kernel=context_kernel),
        q_rank=MIA_Q_RANK,
        heads=MIA_Q_HEADS,
        head_dim=MIA_NVFP4_HEAD_DIM,
        rope_dim=MIA_NVFP4_ROPE_DIM,
        proposal_rows=MIA_DSPARK_PROPOSAL_ROWS,
        context_rows=MIA_DSPARK_CONTEXT_ROWS,
        prefill_tile_rows=MIA_TARGET_PREFILL_TILE_ROWS,
    )


__all__ = [
    "MIA_DSPARK_PROPOSAL_ROWS",
    "MIA_LEARNED_NORM_THREADS",
    "MIA_QKV_FINALIZE_THREADS",
    "MIA_QKV_PROJECTION_WIDTH",
    "MIA_Q_HEADS",
    "MIA_Q_RANK",
    "MIA_TARGET_DECODE_ROWS",
    "MIA_TARGET_PREFILL_TILE_ROWS",
    "MiaQKVPrologue",
    "MiaBoundQKVPrologue",
    "bind_mia_qkv_prologue",
    "install_mia_qkv_prologue",
]
