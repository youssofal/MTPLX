"""Mia-compatible paged FP8 indexer cache for DeepSeek V4 ratio-4 layers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import math

import mlx.core as mx

from mtplx.attention_context import current_attention_phase
from mtplx.paged_cache import PagedCachePlan, PagedCachePool


INDEXER_HEADS = 64
INDEXER_HEAD_DIM = 128
INDEXER_TOPK = 512
INDEXER_RECORD_BYTES = 132
INDEXER_ROPE_POSITIONS = 384_005
# Pinned vLLM bounds the sparse-indexer FP32 logits slab to 512 MiB and
# query-subchunks only when one request exceeds that construction policy.
INDEXER_PREFILL_MAX_LOGITS_BYTES = 512 * 1024 * 1024
INDEXER_PREFILL_Q_TILE = 32
INDEXER_PREFILL_K_TILE = 256
INDEXER_PREFILL_K_SIMD_SPAN = 128
INDEXER_PREFILL_DIM_PANEL = 8
INDEXER_PREFILL_SCORE_THREADS = 256
# SparkInfer's production paged prefill route owns one 32K K supertile and
# folds it into a fixed top-512 carry.  Its CUDA scorer owns Q32xK512 with 256
# threads and keeps the completed head reduction in registers.  Metal retains
# Q32 and 256-thread ownership, but uses K256: eight SIMD groups each own
# Q8xK128, keeping the score plus current-head dot fragments to 64 FP32 values
# per thread instead of the K512 port's 128 before other live state.
INDEXER_PREFILL_SCORE_CHUNK_ROWS = 32768
INDEXER_DECODE_SLICE_ROWS = 4096


_FP8_HEADER = r"""
    using namespace metal;

    inline uchar mtplx_indexer_e4m3_encode(float value) {
        uint sign = as_type<uint>(value) >> 31;
        float magnitude = min(abs(value), 448.0f);
        uchar code;
        constexpr float MIN_NORMAL = 0.015625f;
        constexpr float SUB_STEP = 0.001953125f;
        if (!(magnitude > 0.0f)) {
            code = uchar(0);
        } else if (magnitude < MIN_NORMAL) {
            uint mantissa = uint(rint(magnitude / SUB_STEP));
            code = mantissa >= 8u ? uchar(0x08) : uchar(mantissa);
        } else {
            int exponent = int(floor(log2(magnitude)));
            float step = exp2(float(exponent - 3));
            uint significand = uint(rint(magnitude / step));
            if (significand >= 16u) {
                exponent += 1;
                significand = 8u;
            }
            uint stored_exponent = uint(exponent + 7);
            if (stored_exponent >= 15u) {
                stored_exponent = 15u;
                significand = min(significand, 14u);
            }
            code = uchar((stored_exponent << 3) | (significand - 8u));
        }
        return uchar(code | uchar(sign << 7));
    }

    inline float mtplx_indexer_e4m3_decode(uchar raw) {
        uint exponent = (uint(raw) >> 3) & 0x0fu;
        uint mantissa = uint(raw) & 0x07u;
        float magnitude = exponent == 0u
            ? float(mantissa) * 0.001953125f
            : (1.0f + float(mantissa) * 0.125f)
                * exp2(float(int(exponent) - 7));
        return (uint(raw) & 0x80u) != 0u ? -magnitude : magnitude;
    }

    inline float mtplx_indexer_record_scale(const device uchar* record) {
        uint scale_bits = uint(record[128u])
            | (uint(record[129u]) << 8u)
            | (uint(record[130u]) << 16u)
            | (uint(record[131u]) << 24u);
        return as_type<float>(scale_bits);
    }
"""


@lru_cache(maxsize=1)
def _pack_kernel():
    source = r"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        const device T* source_row = rows + size_t(row) * 128u;
        device uchar* record = records + size_t(row) * 132u;

        uint dim0 = lane * 4u;
        float local_amax = 0.0f;
        for (uint element = 0u; element < 4u; ++element) {
            local_amax = max(
                local_amax,
                abs(float(source_row[dim0 + element]))
            );
        }
        float amax = simd_max(local_amax);
        float scale = exp2(ceil(log2(max(amax, 1.0e-4f) / 448.0f)));
        for (uint element = 0u; element < 4u; ++element) {
            uint dim = dim0 + element;
            record[dim] = mtplx_indexer_e4m3_encode(
                float(source_row[dim]) / scale
            );
        }
        if (lane < 4u) {
            uint scale_bits = as_type<uint>(scale);
            record[128u + lane] = uchar((scale_bits >> (8u * lane)) & 0xffu);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_indexer_fp8_pack",
        input_names=["rows"],
        output_names=["records"],
        header=_FP8_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def _pack_indexer132(rows: mx.array) -> mx.array:
    if rows.ndim < 2 or int(rows.shape[-1]) != INDEXER_HEAD_DIM:
        raise ValueError("Mia indexer rows must end in width 128")
    row_count = math.prod(int(dim) for dim in rows.shape[:-1])
    return _pack_kernel()(
        inputs=[mx.contiguous(rows)],
        template=[("T", rows.dtype)],
        grid=(row_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*rows.shape[:-1], INDEXER_RECORD_BYTES)],
        output_dtypes=[mx.uint8],
    )[0]


@dataclass(frozen=True)
class MiaIndexerRoPETable:
    """Shared read-only vLLM-layout cos/sin cache for every ratio-4 indexer.

    ``values[position]`` stores 32 FP32 cosines followed by 32 FP32 sines.
    One engine-owned table is shared by all ratio-4 layers; it must not be
    replicated per layer.
    """

    values: mx.array

    @property
    def max_positions(self) -> int:
        return int(self.values.shape[0])

    @property
    def nbytes(self) -> int:
        return math.prod(int(dim) for dim in self.values.shape) * 4


def precompute_indexer_rope_table(
    inv_freq: mx.array,
    *,
    max_positions: int,
) -> MiaIndexerRoPETable:
    """Build one engine-owned FP32 ``[positions, cos32 | sin32]`` table."""
    max_positions = int(max_positions)
    if max_positions <= 0:
        raise ValueError("Mia indexer RoPE table capacity must be positive")
    if inv_freq.ndim != 1 or int(inv_freq.shape[0]) != 32:
        raise ValueError("Mia indexer RoPE frequencies must have shape [32]")
    positions = mx.arange(max_positions, dtype=mx.float32)
    angles = positions[:, None] * inv_freq.astype(mx.float32)[None, :]
    values = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)
    values = mx.contiguous(values.astype(mx.float32))
    mx.eval(values)
    return MiaIndexerRoPETable(values=values)


@lru_cache(maxsize=1)
def _query_rope_quant_kernel():
    """Pack Q using the construction-bound vLLM cos/sin cache."""
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint head = group % 64u;
        uint query = group / 64u;
        const device T* q_row = queries
            + (size_t(query) * 64u + size_t(head)) * 128u;
        device uchar* record = records
            + (size_t(query) * 64u + size_t(head)) * 132u;
        threadgroup float rotated[128];

        uint dim0 = lane * 4u;
        float local_max = 0.0f;
        for (uint element = 0u; element < 4u; ++element) {
            uint dim = dim0 + element;
            float value = float(q_row[dim]);
            if (dim >= 64u) {
                uint rope_dim = dim - 64u;
                uint pair = rope_dim / 2u;
                uint position = uint(positions[query]);
                const device float* rope_row = cos_sin_cache
                    + size_t(position) * 64u;
                float c = rope_row[pair];
                float s = rope_row[32u + pair];
                uint even_dim = 64u + pair * 2u;
                float even = float(q_row[even_dim]);
                float odd = float(q_row[even_dim + 1u]);
                value = (rope_dim & 1u) == 0u
                    ? even * c - odd * s
                    : odd * c + even * s;
                value = float(bfloat(value));
            }
            rotated[dim] = value;
            local_max = max(local_max, abs(value));
        }
        float amax = simd_max(local_max);
        float scale = exp2(ceil(log2(max(amax, 1.0e-4f) / 448.0f)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint element = 0u; element < 4u; ++element) {
            uint dim = dim0 + element;
            record[dim] = mtplx_indexer_e4m3_encode(rotated[dim] / scale);
        }
        if (lane < 4u) {
            uint one_bits = as_type<uint>(1.0f);
            record[128u + lane] = uchar((one_bits >> (8u * lane)) & 0xffu);
        }
        if (lane == 0u) {
            float folded_weight =
                float(weights[size_t(query) * 64u + head])
                * float(weight_scale);
            scaled_weights[size_t(query) * 64u + head] =
                folded_weight * scale;
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_indexer_q_rope_fp8",
        input_names=[
            "queries",
            "weights",
            "positions",
            "cos_sin_cache",
            "weight_scale",
        ],
        output_names=["records", "scaled_weights"],
        header=_FP8_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _query_rope_quant_from_inv_freq_kernel():
    """Validated reference entry point; the installed lane never binds this."""
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint head = group % 64u;
        uint query = group / 64u;
        const device T* q_row = queries
            + (size_t(query) * 64u + size_t(head)) * 128u;
        device uchar* record = records
            + (size_t(query) * 64u + size_t(head)) * 132u;
        threadgroup float rotated[128];

        uint dim0 = lane * 4u;
        float local_max = 0.0f;
        for (uint element = 0u; element < 4u; ++element) {
            uint dim = dim0 + element;
            float value = float(q_row[dim]);
            if (dim >= 64u) {
                uint rope_dim = dim - 64u;
                uint pair = rope_dim / 2u;
                float angle = float(positions[query]) * float(inv_freq[pair]);
                float c = cos(angle);
                float s = sin(angle);
                uint even_dim = 64u + pair * 2u;
                float even = float(q_row[even_dim]);
                float odd = float(q_row[even_dim + 1u]);
                value = (rope_dim & 1u) == 0u
                    ? even * c - odd * s
                    : odd * c + even * s;
                value = float(bfloat(value));
            }
            rotated[dim] = value;
            local_max = max(local_max, abs(value));
        }
        float amax = simd_max(local_max);
        float scale = exp2(ceil(log2(max(amax, 1.0e-4f) / 448.0f)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint element = 0u; element < 4u; ++element) {
            uint dim = dim0 + element;
            record[dim] = mtplx_indexer_e4m3_encode(rotated[dim] / scale);
        }
        if (lane < 4u) {
            uint one_bits = as_type<uint>(1.0f);
            record[128u + lane] = uchar((one_bits >> (8u * lane)) & 0xffu);
        }
        if (lane == 0u) {
            float folded_weight =
                float(weights[size_t(query) * 64u + head])
                * float(weight_scale);
            scaled_weights[size_t(query) * 64u + head] =
                folded_weight * scale;
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_indexer_q_rope_fp8_reference",
        input_names=[
            "queries",
            "weights",
            "positions",
            "inv_freq",
            "weight_scale",
        ],
        output_names=["records", "scaled_weights"],
        header=_FP8_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def fused_indexer_query_records(
    queries: mx.array,
    weights: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    *,
    weight_scale: float,
) -> tuple[MiaIndexerQueryRecords, mx.array]:
    """Fuse source-order Q RoPE/FP8 quantization and Q-scale weight folding."""
    batch, query_count, heads, head_dim = (int(dim) for dim in queries.shape)
    if (batch, heads, head_dim) != (1, INDEXER_HEADS, INDEXER_HEAD_DIM):
        raise ValueError("Mia indexer Q requires [1, rows, 64, 128]")
    return _run_fused_indexer_query_records(
        queries,
        weights,
        positions,
        inv_freq,
        weight_scale=weight_scale,
    )


def _run_fused_indexer_query_records(
    queries: mx.array,
    weights: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    *,
    weight_scale: float,
) -> tuple[MiaIndexerQueryRecords, mx.array]:
    """Validated scalar wrapper for the generic/reference entry point."""
    batch, query_count = (int(dim) for dim in queries.shape[:2])
    records, scaled_weights = _query_rope_quant_from_inv_freq_kernel()(
        inputs=[
            mx.contiguous(queries),
            mx.contiguous(weights),
            mx.contiguous(positions.astype(mx.int32)),
            mx.contiguous(inv_freq),
            mx.array(float(weight_scale), dtype=mx.float32),
        ],
        template=[("T", queries.dtype)],
        grid=(query_count * INDEXER_HEADS * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch, query_count, INDEXER_HEADS, INDEXER_RECORD_BYTES),
            (batch, query_count, INDEXER_HEADS),
        ],
        output_dtypes=[mx.uint8, mx.float32],
    )
    return MiaIndexerQueryRecords(records), scaled_weights


def _run_installed_indexer_query_records(
    queries: mx.array,
    weights: mx.array,
    positions: mx.array,
    *,
    cos_sin_cache: mx.array,
    weight_scale: mx.array,
    kernel,
) -> tuple[MiaIndexerQueryRecords, mx.array]:
    """Direct finalizer with construction-bound table and scalar operands."""
    batch, query_count = (int(dim) for dim in queries.shape[:2])
    records, scaled_weights = kernel(
        inputs=[
            mx.contiguous(queries),
            mx.contiguous(weights),
            mx.contiguous(positions),
            cos_sin_cache,
            weight_scale,
        ],
        template=[("T", queries.dtype)],
        grid=(query_count * INDEXER_HEADS * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch, query_count, INDEXER_HEADS, INDEXER_RECORD_BYTES),
            (batch, query_count, INDEXER_HEADS),
        ],
        output_dtypes=[mx.uint8, mx.float32],
    )
    return MiaIndexerQueryRecords(records), scaled_weights


def install_indexer_query_records(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    weight_scale: float,
    rope_table: MiaIndexerRoPETable | None = None,
):
    observed = (int(heads), int(head_dim), int(rope_dim))
    expected = (INDEXER_HEADS, INDEXER_HEAD_DIM, 64)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia indexer Q geometry: {observed} != {expected}"
        )
    if rope_table is None:
        raise ValueError(
            "exact Mia indexer installation requires a shared RoPE table"
        )
    if (
        rope_table.values.ndim != 2
        or tuple(int(dim) for dim in rope_table.values.shape)
        != (INDEXER_ROPE_POSITIONS, 64)
        or rope_table.values.dtype != mx.float32
    ):
        raise ValueError(
            "Mia indexer RoPE table must be FP32 [384005, 64]"
        )
    installed_scale = mx.array(float(weight_scale), dtype=mx.float32)
    mx.eval(installed_scale)
    kernel = _query_rope_quant_kernel()
    return partial(
        _run_installed_indexer_query_records,
        cos_sin_cache=rope_table.values,
        weight_scale=installed_scale,
        kernel=kernel,
    )


def install_reference_indexer_query_records(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    weight_scale: float,
):
    """Install the explicit inv-freq reference path, never the exact lane."""
    observed = (int(heads), int(head_dim), int(rope_dim))
    expected = (INDEXER_HEADS, INDEXER_HEAD_DIM, 64)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia indexer Q geometry: {observed} != {expected}"
        )
    _query_rope_quant_from_inv_freq_kernel()
    return partial(
        _run_fused_indexer_query_records,
        weight_scale=float(weight_scale),
    )


def _decode_e4m3(raw_bytes: mx.array) -> mx.array:
    raw = raw_bytes.astype(mx.uint32)
    negative = (raw & 0x80) != 0
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    subnormal = mantissa.astype(mx.float32) * (2.0**-9)
    normal = (1.0 + mantissa.astype(mx.float32) / 8.0) * mx.power(
        mx.array(2.0, dtype=mx.float32), exponent.astype(mx.float32) - 7.0
    )
    magnitude = mx.where(exponent == 0, subnormal, normal)
    return mx.where(negative, -magnitude, magnitude)


def decode_indexer132(records: mx.array) -> mx.array:
    if records.dtype != mx.uint8 or int(records.shape[-1]) != INDEXER_RECORD_BYTES:
        raise ValueError("Mia indexer records must end in 132 uint8 bytes")
    scales = mx.contiguous(records[..., 128:132]).view(mx.float32)
    return _decode_e4m3(records[..., :128]) * scales


@dataclass(frozen=True)
class PagedMiaIndexerRecords:
    records: mx.array
    block_table: mx.array
    length: int
    block_size: int
    record_bytes: int = INDEXER_RECORD_BYTES

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, int(self.length), INDEXER_HEAD_DIM)


@dataclass(frozen=True)
class MiaTopKSelection:
    """Compact Mia sparse-indexer interchange consumed by sparse MLA."""

    indices: mx.array
    lengths: mx.array


@dataclass(frozen=True)
class MiaIndexerQueryRecords:
    """Post-RoPE FP8 query records produced by the fused Mia Q boundary."""

    records: mx.array

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            int(self.records.shape[0]),
            int(self.records.shape[1]),
            INDEXER_HEADS,
            INDEXER_HEAD_DIM,
        )


@dataclass(frozen=True)
class MiaIndexerWorkspace:
    """Construction-owned seed storage shared by every ratio-4 layer.

    MLX Metal kernels return functional output arrays, so candidate buffers are
    allocator-owned results.  The large, repeatedly identical carry seeds are
    different: they are allocated once for the installed launcher envelope and
    sliced by the phase route.  ``sentinel`` is the fixed ratio-4 capacity and
    is therefore outside every live logical row range.
    """

    max_query_rows: int
    topk: int
    sentinel: int
    empty_scores: mx.array
    empty_indices: mx.array

    @classmethod
    def allocate(
        cls,
        *,
        max_query_rows: int,
        topk: int,
        sentinel: int,
    ) -> "MiaIndexerWorkspace":
        max_query_rows = int(max_query_rows)
        topk = int(topk)
        sentinel = int(sentinel)
        if max_query_rows <= 0 or topk <= 0 or sentinel <= 0:
            raise ValueError("Mia indexer workspace geometry must be positive")
        shape = (1, max_query_rows, topk)
        return cls(
            max_query_rows=max_query_rows,
            topk=topk,
            sentinel=sentinel,
            empty_scores=mx.full(shape, -float("inf"), dtype=mx.float32),
            empty_indices=mx.full(shape, sentinel, dtype=mx.int32),
        )

    def seeds(self, query_count: int) -> tuple[mx.array, mx.array]:
        stop = int(query_count)
        return self.empty_scores[:, :stop], self.empty_indices[:, :stop]


class PagedMiaIndexerRows:
    """Fixed pages for the 132-byte FP8+scale indexer records Mia uses."""

    mode = "fp8_e4m3_ue8m0_scale132_paged"
    record_bytes = INDEXER_RECORD_BYTES

    def __init__(self, *, capacity_rows: int, block_size: int = 64) -> None:
        capacity_rows = int(capacity_rows)
        if capacity_rows <= 0:
            raise ValueError("paged Mia indexer capacity_rows must be positive")
        self._capacity_rows = capacity_rows
        plan = PagedCachePlan.contiguous(
            block_size=int(block_size),
            num_blocks=math.ceil(capacity_rows / int(block_size)),
            array_names=("records",),
        )
        self._pool = PagedCachePool(plan)
        self._pages = self._pool.bind(
            "records", row_shape=(self.record_bytes,), dtype=mx.uint8
        )

    def __len__(self) -> int:
        return int(self._pool.offset)

    @property
    def capacity(self) -> int:
        return self._capacity_rows

    @property
    def pages(self) -> mx.array:
        return self._pages

    @property
    def block_table(self) -> mx.array:
        return self._pool.block_table

    @property
    def block_size(self) -> int:
        return self._pool.block_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, len(self), INDEXER_HEAD_DIM)

    @property
    def paged_records(self) -> PagedMiaIndexerRecords:
        return PagedMiaIndexerRecords(
            records=self.pages,
            block_table=self.block_table,
            length=len(self),
            block_size=self.block_size,
        )

    @property
    def records(self) -> mx.array:
        return self._pool.active("records")[None]

    @property
    def state(self):
        return self.pages, self.block_table, len(self)

    def append(self, rows: mx.array) -> None:
        if rows.ndim != 3 or tuple(int(dim) for dim in rows.shape[::2]) != (1, 128):
            raise ValueError("paged Mia indexer rows must be [1, rows, 128]")
        self.append_records(_pack_indexer132(rows))

    def append_records(self, records: mx.array) -> None:
        """Insert records already finalized by the fused Mia compressor."""
        if (
            records.dtype != mx.uint8
            or records.ndim != 3
            or tuple(int(dim) for dim in records.shape[:1]) != (1,)
            or int(records.shape[-1]) != self.record_bytes
        ):
            raise ValueError(
                "paged Mia indexer records must be [1, rows, 132] uint8"
            )
        count = int(records.shape[1])
        if len(self) + count > self.capacity:
            raise ValueError(
                f"paged Mia indexer capacity exceeded: {len(self) + count} "
                f"> {self.capacity}"
            )
        self._append_installed_records(records)

    def _append_installed_records(self, records: mx.array) -> None:
        """Insert records emitted by the installed Mia132 finalizer."""
        self._pool._write_installed_tail(
            {"records": records[0]},
            count=int(records.shape[1]),
        )

    def _append_m6_records(self, records: mx.array, schedule) -> None:
        """Insert physical-M6 records through the shared ratio-4 mapping."""
        self._pool._write_installed_mapping(
            {"records": records[0]},
            physical_blocks=schedule.compressed_blocks,
            block_offsets=schedule.compressed_offsets,
            new_offset=schedule.first_window + schedule.emitted_rows,
        )

    def decode(self) -> mx.array:
        return decode_indexer132(self.records)

    def truncate(self, length: int) -> None:
        length = max(0, int(length))
        if length < len(self):
            self._pool.truncate(length)

    def clear(self) -> None:
        self._pool.clear()

    def replace_state(self, state) -> None:
        if state is None:
            self.clear()
            return
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError("invalid paged Mia indexer state")
        pages, block_table, length = state
        if tuple(int(value) for value in block_table.shape) != (
            self._pool.num_blocks,
        ):
            raise ValueError("paged Mia indexer block table shape changed")
        self._pool.block_table = block_table
        self._pool.replace_state({"records": pages}, int(length))
        self._pages = self._pool.buffer("records")


@lru_cache(maxsize=1)
def _oracle_score_kernel():
    """Score ordinary packed Q records whose per-head scale remains stored."""
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint row = group % uint(n_rows);
        uint query = group / uint(n_rows);
        uint logical_row = uint(row_start) + row;
        uint logical_block = logical_row / uint(block_size);
        uint row_in_block = logical_row - logical_block * uint(block_size);
        uint physical_block = uint(block_table[logical_block]);
        uint physical_row = physical_block * uint(block_size) + row_in_block;
        const device uchar* k_record = k_records + size_t(physical_row) * 132u;
        uint k_scale_bits = uint(k_record[128u])
            | (uint(k_record[129u]) << 8u)
            | (uint(k_record[130u]) << 16u)
            | (uint(k_record[131u]) << 24u);
        float k_scale = as_type<float>(k_scale_bits);

        float score = 0.0f;
        for (uint head = 0u; head < 64u; ++head) {
            const device uchar* q_record = q_records
                + (size_t(query) * 64u + size_t(head)) * 132u;
            float q_scale = mtplx_indexer_record_scale(q_record);
            float partial = 0.0f;
            uint dim0 = lane * 4u;
            for (uint element = 0u; element < 4u; ++element) {
                uint dim = dim0 + element;
                partial += mtplx_indexer_e4m3_decode(q_record[dim])
                    * mtplx_indexer_e4m3_decode(k_record[dim]);
            }
            float dot = simd_sum(partial) * q_scale * k_scale;
            if (lane == 0u) {
                score += max(dot, 0.0f)
                    * weights[size_t(query) * 64u + size_t(head)];
            }
        }
        if (lane == 0u) {
            scores[size_t(query) * size_t(n_rows) + size_t(row)] = score;
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_paged_indexer_oracle_scores",
        input_names=[
            "q_records",
            "weights",
            "k_records",
            "block_table",
            "row_start",
            "n_rows",
            "block_size",
        ],
        output_names=["scores"],
        header=_FP8_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def _tiled_score_source(*, apply_query_scale: bool) -> tuple[str, str]:
    """Build distinct exact and ordinary-record Metal scorer sources."""
    fragment_count = INDEXER_PREFILL_K_SIMD_SPAN // 8
    header = _FP8_HEADER + rf"""
        // SparkInfer owns Q32xK512 with 256 CUDA threads and retains the
        // completed 64-head reduction in registers.  Metal keeps Q32 and 256
        // threads, but splits K256 over two SIMD groups per Q8 tile.  K512
        // would require 128 live FP32 score/dot values per thread before the
        // input fragments; K256 requires 64.  Eight-dimension staging panels
        // keep threadgroup scratch at __MTPLX_INDEXER_SCRATCH_DESCRIPTION__
        constant constexpr int MTPLX_INDEX_Q_TILE = {INDEXER_PREFILL_Q_TILE};
        constant constexpr int MTPLX_INDEX_K_TILE = {INDEXER_PREFILL_K_TILE};
        constant constexpr int MTPLX_INDEX_K_SIMD_SPAN = {INDEXER_PREFILL_K_SIMD_SPAN};
        constant constexpr int MTPLX_INDEX_DIM_PANEL = {INDEXER_PREFILL_DIM_PANEL};
        constant constexpr int MTPLX_INDEX_DIM = 128;
        constant constexpr int MTPLX_INDEX_HEADS = 64;
        constant constexpr int MTPLX_INDEX_THREADS = {INDEXER_PREFILL_SCORE_THREADS};

        inline uint mtplx_indexer_mma_row(uint lane) {{
            uint quad = lane / 4u;
            return (quad & 4u) + ((lane / 2u) % 4u);
        }}

        inline uint mtplx_indexer_mma_col(uint lane) {{
            uint quad = lane / 4u;
            return (quad & 2u) * 2u + (lane % 2u) * 2u;
        }}
    """

    score_declarations = "\n".join(
        f"""        simdgroup_matrix<float, 8, 8> score{index} =
            simdgroup_matrix<float, 8, 8>(0.0f);"""
        for index in range(fragment_count)
    )
    dot_declarations = "\n".join(
        f"""            simdgroup_matrix<float, 8, 8> dot{index} =
                simdgroup_matrix<float, 8, 8>(0.0f);"""
        for index in range(fragment_count)
    )
    mma_steps = "\n".join(
        f"""                simdgroup_load(
                    k_matrix,
                    k_values + k_simd_base + {index * 8}u,
                    MTPLX_INDEX_K_TILE
                );
                simdgroup_multiply_accumulate(
                    dot{index}, q_matrix, k_matrix, dot{index}
                );"""
        for index in range(fragment_count)
    )
    q_scale_factor = " * q_scales[local_q]" if apply_query_scale else ""
    score_updates = "\n".join(
        f"""            score{index}.thread_elements()[0] += max(
                dot{index}.thread_elements()[0]{q_scale_factor}, 0.0f
            ) * q_weight;
            score{index}.thread_elements()[1] += max(
                dot{index}.thread_elements()[1]{q_scale_factor}, 0.0f
            ) * q_weight;"""
        for index in range(fragment_count)
    )
    score_stores = "\n".join(
        f"""        uint local_k{index} = k_simd_base + {index * 8}u + mma_col;
        if (query < uint(n_queries)
            && k0 + local_k{index} < uint(n_rows)) {{
            output[
                size_t(query) * size_t(n_rows) + size_t(k0 + local_k{index})
            ] = score{index}.thread_elements()[0] * k_scales[local_k{index}];
        }}
        if (query < uint(n_queries)
            && k0 + local_k{index} + 1u < uint(n_rows)) {{
            output[
                size_t(query) * size_t(n_rows)
                    + size_t(k0 + local_k{index} + 1u)
            ] = score{index}.thread_elements()[1]
                * k_scales[local_k{index} + 1u];
        }}"""
        for index in range(fragment_count)
    )

    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tile = threadgroup_position_in_grid.x;
        uint k_tiles = (uint(n_rows) + MTPLX_INDEX_K_TILE - 1u)
            / MTPLX_INDEX_K_TILE;
        uint q_tile = tile / k_tiles;
        uint k_tile = tile - q_tile * k_tiles;
        uint q0 = q_tile * MTPLX_INDEX_Q_TILE;
        uint k0 = k_tile * MTPLX_INDEX_K_TILE;

        threadgroup half q_values[
            MTPLX_INDEX_Q_TILE * MTPLX_INDEX_DIM_PANEL
        ];
        threadgroup half k_values[
            MTPLX_INDEX_DIM_PANEL * MTPLX_INDEX_K_TILE
        ];
        threadgroup uint physical_rows[MTPLX_INDEX_K_TILE];
        threadgroup float q_weights[MTPLX_INDEX_Q_TILE];
        __MTPLX_INDEXER_Q_SCALE_STORAGE__
        threadgroup float k_scales[MTPLX_INDEX_K_TILE];

        if (tid < MTPLX_INDEX_K_TILE) {
            uint local_k = tid;
            uint logical_row = uint(row_start) + k0 + local_k;
            if (k0 + local_k < uint(n_rows)) {
                uint logical_block = logical_row / uint(block_size);
                uint row_in_block = logical_row
                    - logical_block * uint(block_size);
                uint physical_block = uint(block_table[logical_block]);
                uint physical_row = physical_block * uint(block_size)
                    + row_in_block;
                physical_rows[local_k] = physical_row;
                const device uchar* record = k_records
                    + size_t(physical_row) * 132u;
                k_scales[local_k] = mtplx_indexer_record_scale(record);
            } else {
                physical_rows[local_k] = 0u;
                k_scales[local_k] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

__MTPLX_INDEXER_SCORE_DECLARATIONS__

        for (uint head = 0u; head < MTPLX_INDEX_HEADS; ++head) {
            if (tid < MTPLX_INDEX_Q_TILE) {
                uint local_q = tid;
                uint query = q0 + local_q;
                if (query < uint(n_queries)) {
                    q_weights[local_q] = weights[
                        size_t(query) * MTPLX_INDEX_HEADS + size_t(head)
                    ];
                    __MTPLX_INDEXER_Q_SCALE_VALID__
                } else {
                    q_weights[local_q] = 0.0f;
                    __MTPLX_INDEXER_Q_SCALE_INVALID__
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<half, 8, 8> q_matrix;
            simdgroup_matrix<half, 8, 8> k_matrix;
__MTPLX_INDEXER_DOT_DECLARATIONS__
            uint q_simd_base = (simd_group / 2u)
                * 8u * MTPLX_INDEX_DIM_PANEL;
            uint k_simd_base = (simd_group % 2u)
                * MTPLX_INDEX_K_SIMD_SPAN;
            for (uint dim0 = 0u;
                 dim0 < MTPLX_INDEX_DIM;
                 dim0 += MTPLX_INDEX_DIM_PANEL) {
                uint local_q = tid / MTPLX_INDEX_DIM_PANEL;
                uint local_dim = tid - local_q * MTPLX_INDEX_DIM_PANEL;
                uint query = q0 + local_q;
                half q_value = half(0.0f);
                if (query < uint(n_queries)) {
                    const device uchar* record = q_records
                        + (size_t(query) * MTPLX_INDEX_HEADS + size_t(head))
                            * 132u;
                    q_value = half(mtplx_indexer_e4m3_decode(
                        record[dim0 + local_dim]
                    ));
                }
                q_values[tid] = q_value;

                for (uint index = tid;
                     index < MTPLX_INDEX_DIM_PANEL * MTPLX_INDEX_K_TILE;
                     index += MTPLX_INDEX_THREADS) {
                    uint panel_dim = index / MTPLX_INDEX_K_TILE;
                    uint local_k = index - panel_dim * MTPLX_INDEX_K_TILE;
                    half k_value = half(0.0f);
                    if (k0 + local_k < uint(n_rows)) {
                        const device uchar* record = k_records
                            + size_t(physical_rows[local_k]) * 132u;
                        // E4M3 values are exactly representable in FP16.  The
                        // positive K scale is applied only after the complete
                        // source-order head reduction below.
                        k_value = half(mtplx_indexer_e4m3_decode(
                            record[dim0 + panel_dim]
                        ));
                    }
                    k_values[index] = k_value;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_load(
                    q_matrix,
                    q_values + q_simd_base,
                    MTPLX_INDEX_DIM_PANEL
                );
__MTPLX_INDEXER_MMA_STEPS__
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            uint local_q = (simd_group / 2u) * 8u
                + mtplx_indexer_mma_row(lane);
            float q_weight = q_weights[local_q];
__MTPLX_INDEXER_SCORE_UPDATES__
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint local_q = (simd_group / 2u) * 8u
            + mtplx_indexer_mma_row(lane);
        uint query = q0 + local_q;
        uint k_simd_base = (simd_group % 2u) * MTPLX_INDEX_K_SIMD_SPAN;
        uint mma_col = mtplx_indexer_mma_col(lane);
__MTPLX_INDEXER_SCORE_STORES__
    """

    if apply_query_scale:
        scratch_description = "6,912 bytes for the ordinary-record oracle."
        q_scale_storage = "threadgroup float q_scales[MTPLX_INDEX_Q_TILE];"
        q_scale_valid = (
            "const device uchar* q_scale_record = q_records + "
            "(size_t(query) * MTPLX_INDEX_HEADS + size_t(head)) * 132u; "
            "q_scales[local_q] = "
            "mtplx_indexer_record_scale(q_scale_record);"
        )
        q_scale_invalid = "q_scales[local_q] = 0.0f;"
    else:
        scratch_description = "6,784 bytes for installed unit-scale Q records."
        q_scale_storage = ""
        q_scale_valid = ""
        q_scale_invalid = ""

    header = header.replace(
        "__MTPLX_INDEXER_SCRATCH_DESCRIPTION__", scratch_description
    )
    source = source.replace(
        "__MTPLX_INDEXER_Q_SCALE_STORAGE__", q_scale_storage
    )
    source = source.replace(
        "__MTPLX_INDEXER_Q_SCALE_VALID__", q_scale_valid
    )
    source = source.replace(
        "__MTPLX_INDEXER_Q_SCALE_INVALID__", q_scale_invalid
    )
    source = source.replace(
        "__MTPLX_INDEXER_SCORE_DECLARATIONS__", score_declarations
    )
    source = source.replace(
        "__MTPLX_INDEXER_DOT_DECLARATIONS__", dot_declarations
    )
    source = source.replace("__MTPLX_INDEXER_MMA_STEPS__", mma_steps)
    source = source.replace(
        "__MTPLX_INDEXER_SCORE_UPDATES__", score_updates
    )
    source = source.replace("__MTPLX_INDEXER_SCORE_STORES__", score_stores)
    return header, source


def _make_tiled_score_kernel(*, name: str, apply_query_scale: bool):
    header, source = _tiled_score_source(apply_query_scale=apply_query_scale)
    return mx.fast.metal_kernel(
        name=name,
        input_names=[
            "q_records",
            "weights",
            "k_records",
            "block_table",
            "row_start",
            "n_rows",
            "block_size",
            "n_queries",
        ],
        output_names=["output"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _tiled_score_kernel():
    """Installed Mia scorer for unit-scale Q records and scale-folded weights."""
    return _make_tiled_score_kernel(
        name="mtplx_dsv4_mia_paged_indexer_tiled_scores",
        apply_query_scale=False,
    )


@lru_cache(maxsize=1)
def _tiled_score_oracle_kernel():
    """Ordinary-record oracle scorer that consumes each stored Q scale."""
    return _make_tiled_score_kernel(
        name="mtplx_dsv4_mia_paged_indexer_tiled_oracle_scores",
        apply_query_scale=True,
    )


@lru_cache(maxsize=1)
def _radix_fold_kernel():
    """Exact SparkInfer-style four-pass MSD radix fold into top-512."""
    header = r"""
        using namespace metal;

        inline uint mtplx_indexer_ordered_key(float value) {
            if (value == 0.0f) {
                return 0x80000000u;
            }
            uint bits = as_type<uint>(value);
            return (bits & 0x80000000u) != 0u
                ? ~bits
                : bits | 0x80000000u;
        }
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint query = threadgroup_position_in_grid.x;
        constexpr uint TOPK = 512u;
        constexpr uint THREADS = 256u;

        threadgroup atomic_uint histogram[256];
        threadgroup uint simd_counts[8];
        threadgroup uint simd_offsets[8];
        threadgroup uint output_counter;
        threadgroup uint batch_start;
        threadgroup uint prefix;
        threadgroup uint remaining;
        threadgroup uint pivot;
        threadgroup uint index_prefix;
        threadgroup uint index_remaining;
        threadgroup uint index_pivot;

        int causal = causal_lengths[query];
        int available = causal - int(row_start);
        uint local_count = uint(max(0, min(available, int(n_local))));
        uint previous_count = uint(has_carry)
            * uint(min(min(causal, int(row_start)), int(TOPK)));
        uint total = local_count + previous_count;
        device float* output_value_row = output_values
            + size_t(query) * TOPK;
        device int* output_index_row = output_indices
            + size_t(query) * TOPK;
        const device float* score_row = scores
            + size_t(query) * size_t(n_local);
        const device int* score_index_row = score_indices
            + size_t(query) * size_t(n_local);
        const device float* carry_value_row = carry_values
            + size_t(query) * TOPK;
        const device int* carry_index_row = carry_indices
            + size_t(query) * TOPK;

        for (uint i = tid; i < TOPK; i += THREADS) {
            output_value_row[i] = -INFINITY;
            output_index_row[i] = int(sentinel);
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        if (total <= TOPK) {
            for (uint i = tid; i < total; i += THREADS) {
                bool local = i < local_count;
                output_value_row[i] = local
                    ? score_row[i]
                    : carry_value_row[i - local_count];
                output_index_row[i] = local
                    ? (use_score_indices
                        ? score_index_row[i]
                        : int(row_start) + int(i))
                    : carry_index_row[i - local_count];
            }
            return;
        }

        if (tid == 0u) {
            prefix = 0u;
            remaining = TOPK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint round = 0u; round < 4u; ++round) {
            if (tid < 256u) {
                atomic_store_explicit(
                    &histogram[tid], 0u, memory_order_relaxed
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint shift = 24u - round * 8u;
            uint locked = prefix;
            uint mask = round == 0u
                ? 0u
                : 0xffffffffu << (32u - round * 8u);
            for (uint i = tid; i < total; i += THREADS) {
                float value = i < local_count
                    ? score_row[i]
                    : carry_value_row[i - local_count];
                uint key = mtplx_indexer_ordered_key(value);
                if (round == 0u || (key & mask) == locked) {
                    uint bucket = (key >> shift) & 0xffu;
                    atomic_fetch_add_explicit(
                        &histogram[bucket], 1u, memory_order_relaxed
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint need = remaining;
                uint chosen = 0u;
                for (int bucket = 255; bucket >= 0; --bucket) {
                    uint count = atomic_load_explicit(
                        &histogram[uint(bucket)], memory_order_relaxed
                    );
                    if (need <= count) {
                        chosen = uint(bucket);
                        break;
                    }
                    need -= count;
                }
                prefix = locked | (chosen << shift);
                remaining = need;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            pivot = prefix;
            index_prefix = 0u;
            index_remaining = remaining;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A value-only radix leaves the final pivot bucket tied.  Resolve its
        // cutoff with a second MSD radix over the already-global logical row
        // indices, ascending.  This is the strict-`>` reference rule: among
        // equal values, lower logical rows win even when they came from carry.
        for (uint round = 0u; round < 4u; ++round) {
            if (tid < 256u) {
                atomic_store_explicit(
                    &histogram[tid], 0u, memory_order_relaxed
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint shift = 24u - round * 8u;
            uint locked = index_prefix;
            uint mask = round == 0u
                ? 0u
                : 0xffffffffu << (32u - round * 8u);
            for (uint i = tid; i < total; i += THREADS) {
                bool local = i < local_count;
                uint carry_slot = i - local_count;
                float value = local
                    ? score_row[i]
                    : carry_value_row[carry_slot];
                uint key = mtplx_indexer_ordered_key(value);
                int logical_index = local
                    ? (use_score_indices
                        ? score_index_row[i]
                        : int(row_start) + int(i))
                    : carry_index_row[carry_slot];
                uint index_key = uint(logical_index);
                if (key == pivot
                    && (round == 0u || (index_key & mask) == locked)) {
                    atomic_fetch_add_explicit(
                        &histogram[(index_key >> shift) & 0xffu],
                        1u,
                        memory_order_relaxed
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint need = index_remaining;
                uint chosen = 0u;
                for (uint bucket = 0u; bucket < 256u; ++bucket) {
                    uint count = atomic_load_explicit(
                        &histogram[bucket], memory_order_relaxed
                    );
                    if (need <= count) {
                        chosen = bucket;
                        break;
                    }
                    need -= count;
                }
                index_prefix = locked | (chosen << shift);
                index_remaining = need;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            index_pivot = index_prefix;
            output_counter = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Deterministic thread-order compaction.  Atomics are fine for the
        // histograms (only counts matter), but not for emitting candidates:
        // their winning order otherwise varies across SIMD-group schedules.
        for (uint selection_pass = 0u; selection_pass < 2u; ++selection_pass) {
            for (uint base = 0u; base < total; base += THREADS) {
                uint i = base + tid;
                bool selected = false;
                bool local = i < local_count;
                uint carry_slot = i - local_count;
                float value = -INFINITY;
                int logical_index = int(sentinel);
                if (i < total) {
                    value = local
                        ? score_row[i]
                        : carry_value_row[carry_slot];
                    logical_index = local
                        ? (use_score_indices
                            ? score_index_row[i]
                            : int(row_start) + int(i))
                        : carry_index_row[carry_slot];
                    uint key = mtplx_indexer_ordered_key(value);
                    selected = selection_pass == 0u
                        ? key > pivot
                        : key == pivot
                            && uint(logical_index) <= index_pivot;
                }
                uint lane_offset = simd_prefix_exclusive_sum(uint(selected));
                uint simd_count = simd_sum(uint(selected));
                if (lane == 0u) {
                    simd_counts[sg] = simd_count;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    uint running = 0u;
                    batch_start = output_counter;
                    for (uint group_id = 0u; group_id < 8u; ++group_id) {
                        simd_offsets[group_id] = running;
                        running += simd_counts[group_id];
                    }
                    output_counter += running;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (selected) {
                    uint position = batch_start + simd_offsets[sg] + lane_offset;
                    if (position < TOPK) {
                        output_value_row[position] = value;
                        output_index_row[position] = logical_index;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_indexer_radix_top512_fold",
        input_names=[
            "scores",
            "score_indices",
            "carry_values",
            "carry_indices",
            "causal_lengths",
            "row_start",
            "n_local",
            "use_score_indices",
            "has_carry",
            "sentinel",
        ],
        output_names=["output_values", "output_indices"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def _run_radix_fold(
    scores: mx.array,
    carry_values: mx.array,
    carry_indices: mx.array,
    causal_lengths: mx.array,
    *,
    row_start: int,
    score_indices: mx.array | None = None,
    has_carry: bool,
    sentinel: int,
    kernel,
) -> tuple[mx.array, mx.array]:
    query_count = int(scores.shape[1])
    n_local = int(scores.shape[2])
    explicit_indices = score_indices is not None
    if score_indices is None:
        score_indices = carry_indices
    return tuple(
        kernel(
            inputs=[
                mx.contiguous(scores),
                mx.contiguous(score_indices),
                mx.contiguous(carry_values),
                mx.contiguous(carry_indices),
                mx.contiguous(causal_lengths),
                int(row_start),
                n_local,
                bool(explicit_indices),
                bool(has_carry),
                int(sentinel),
            ],
            template=[],
            grid=(query_count * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, query_count, INDEXER_TOPK)] * 2,
            output_dtypes=[mx.float32, mx.int32],
        )
    )


@lru_cache(maxsize=1)
def _fused_decode_candidates_kernel():
    """Fused paged FP8 score plus exact local radix top-512."""
    header = _FP8_HEADER + r"""
        inline uint mtplx_indexer_ordered_key(float value) {
            if (value == 0.0f) {
                return 0x80000000u;
            }
            uint bits = as_type<uint>(value);
            return (bits & 0x80000000u) != 0u
                ? ~bits
                : bits | 0x80000000u;
        }
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint query = group / uint(n_slices);
        uint slice = group - query * uint(n_slices);
        constexpr uint TOPK = 512u;
        constexpr uint SLICE_ROWS = 4096u;
        constexpr uint THREADS = 256u;
        constexpr uint SIMD_GROUPS = 8u;

        threadgroup float local_scores[SLICE_ROWS];
        threadgroup atomic_uint histogram[256];
        threadgroup uint simd_counts[SIMD_GROUPS];
        threadgroup uint simd_offsets[SIMD_GROUPS];
        threadgroup uint output_counter;
        threadgroup uint batch_start;
        threadgroup uint prefix;
        threadgroup uint remaining;
        threadgroup uint pivot;
        threadgroup uint index_prefix;
        threadgroup uint index_remaining;
        threadgroup uint index_pivot;

        int causal = causal_lengths[query];
        uint row_start = slice * SLICE_ROWS;
        uint local_count = uint(max(
            0,
            min(causal - int(row_start), int(SLICE_ROWS))
        ));
        device float* output_value_row = candidate_values
            + size_t(group) * TOPK;
        device int* output_index_row = candidate_indices
            + size_t(group) * TOPK;

        for (uint i = tid; i < TOPK; i += THREADS) {
            output_value_row[i] = -INFINITY;
            output_index_row[i] = int(sentinel);
        }

        for (uint local_row = sg;
             local_row < local_count;
             local_row += SIMD_GROUPS) {
            uint logical_row = row_start + local_row;
            uint logical_block = logical_row / uint(block_size);
            uint row_in_block = logical_row
                - logical_block * uint(block_size);
            uint physical_block = uint(block_table[logical_block]);
            uint physical_row = physical_block * uint(block_size) + row_in_block;
            const device uchar* k_record = k_records
                + size_t(physical_row) * 132u;
            float k_scale = mtplx_indexer_record_scale(k_record);
            float score = 0.0f;
            for (uint head = 0u; head < 64u; ++head) {
                const device uchar* q_record = q_records
                    + (size_t(query) * 64u + size_t(head)) * 132u;
                float partial = 0.0f;
                uint dim0 = lane * 4u;
                for (uint element = 0u; element < 4u; ++element) {
                    uint dim = dim0 + element;
                    partial += mtplx_indexer_e4m3_decode(q_record[dim])
                        * mtplx_indexer_e4m3_decode(k_record[dim]);
                }
                float dot = simd_sum(partial);
                if (lane == 0u) {
                    score += max(dot, 0.0f)
                        * weights[size_t(query) * 64u + size_t(head)];
                }
            }
            if (lane == 0u) {
                local_scores[local_row] = score * k_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        if (local_count <= TOPK) {
            for (uint i = tid; i < local_count; i += THREADS) {
                output_value_row[i] = local_scores[i];
                output_index_row[i] = int(row_start + i);
            }
            return;
        }

        if (tid == 0u) {
            prefix = 0u;
            remaining = TOPK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint round = 0u; round < 4u; ++round) {
            atomic_store_explicit(
                &histogram[tid], 0u, memory_order_relaxed
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint shift = 24u - round * 8u;
            uint locked = prefix;
            uint mask = round == 0u
                ? 0u
                : 0xffffffffu << (32u - round * 8u);
            for (uint i = tid; i < local_count; i += THREADS) {
                uint key = mtplx_indexer_ordered_key(local_scores[i]);
                if (round == 0u || (key & mask) == locked) {
                    atomic_fetch_add_explicit(
                        &histogram[(key >> shift) & 0xffu],
                        1u,
                        memory_order_relaxed
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint need = remaining;
                uint chosen = 0u;
                for (int bucket = 255; bucket >= 0; --bucket) {
                    uint count = atomic_load_explicit(
                        &histogram[uint(bucket)], memory_order_relaxed
                    );
                    if (need <= count) {
                        chosen = uint(bucket);
                        break;
                    }
                    need -= count;
                }
                prefix = locked | (chosen << shift);
                remaining = need;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            pivot = prefix;
            index_prefix = 0u;
            index_remaining = remaining;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Resolve the final equal-score bucket by ascending logical row.
        // Atomic histogram counts are deterministic; candidate emission is
        // deliberately handled by the ordered compaction below.
        for (uint round = 0u; round < 4u; ++round) {
            atomic_store_explicit(
                &histogram[tid], 0u, memory_order_relaxed
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint shift = 24u - round * 8u;
            uint locked = index_prefix;
            uint mask = round == 0u
                ? 0u
                : 0xffffffffu << (32u - round * 8u);
            for (uint i = tid; i < local_count; i += THREADS) {
                float value = local_scores[i];
                uint key = mtplx_indexer_ordered_key(value);
                uint logical_index = row_start + i;
                if (key == pivot
                    && (round == 0u || (logical_index & mask) == locked)) {
                    atomic_fetch_add_explicit(
                        &histogram[(logical_index >> shift) & 0xffu],
                        1u,
                        memory_order_relaxed
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint need = index_remaining;
                uint chosen = 0u;
                for (uint bucket = 0u; bucket < 256u; ++bucket) {
                    uint count = atomic_load_explicit(
                        &histogram[bucket], memory_order_relaxed
                    );
                    if (need <= count) {
                        chosen = bucket;
                        break;
                    }
                    need -= count;
                }
                index_prefix = locked | (chosen << shift);
                index_remaining = need;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            index_pivot = index_prefix;
            output_counter = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint selection_pass = 0u; selection_pass < 2u; ++selection_pass) {
            for (uint base = 0u; base < local_count; base += THREADS) {
                uint i = base + tid;
                bool selected = false;
                float value = -INFINITY;
                uint logical_index = uint(sentinel);
                if (i < local_count) {
                    value = local_scores[i];
                    logical_index = row_start + i;
                    uint key = mtplx_indexer_ordered_key(value);
                    selected = selection_pass == 0u
                        ? key > pivot
                        : key == pivot && logical_index <= index_pivot;
                }
                uint lane_offset = simd_prefix_exclusive_sum(uint(selected));
                uint simd_count = simd_sum(uint(selected));
                if (lane == 0u) {
                    simd_counts[sg] = simd_count;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    uint running = 0u;
                    batch_start = output_counter;
                    for (uint group_id = 0u; group_id < SIMD_GROUPS; ++group_id) {
                        simd_offsets[group_id] = running;
                        running += simd_counts[group_id];
                    }
                    output_counter += running;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (selected) {
                    uint position = batch_start + simd_offsets[sg] + lane_offset;
                    if (position < TOPK) {
                        output_value_row[position] = value;
                        output_index_row[position] = int(logical_index);
                    }
                }
                threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_decode_indexer_top512",
        input_names=[
            "q_records",
            "weights",
            "k_records",
            "block_table",
            "causal_lengths",
            "n_slices",
            "block_size",
            "sentinel",
        ],
        output_names=["candidate_values", "candidate_indices"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def _run_fused_decode_candidates(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    causal_lengths: mx.array,
    *,
    kernel,
) -> tuple[mx.array, mx.array]:
    query_count = int(q_records.shape[1])
    n_slices = max(
        1,
        (int(rows.length) + INDEXER_DECODE_SLICE_ROWS - 1)
        // INDEXER_DECODE_SLICE_ROWS,
    )
    return tuple(
        kernel(
            inputs=[
                mx.contiguous(q_records),
                mx.contiguous(weights),
                rows.records,
                rows.block_table,
                mx.contiguous(causal_lengths),
                n_slices,
                int(rows.block_size),
                int(rows.length),
            ],
            template=[],
            grid=(query_count * n_slices * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, query_count, n_slices, INDEXER_TOPK)] * 2,
            output_dtypes=[mx.float32, mx.int32],
        )
    )


def _run_paged_indexer_score_slice_oracle(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    row_start: int,
    row_count: int,
) -> mx.array:
    batch, query_count = (int(dim) for dim in q_records.shape[:2])
    (scores,) = _oracle_score_kernel()(
        inputs=[
            q_records,
            weights,
            rows.records,
            rows.block_table,
            int(row_start),
            int(row_count),
            int(rows.block_size),
        ],
        template=[],
        grid=(batch * query_count * int(row_count) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, query_count, int(row_count))],
        output_dtypes=[mx.float32],
    )
    return scores


def _run_paged_indexer_tiled_score_slice_oracle(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    row_start: int,
    row_count: int,
) -> mx.array:
    """Tiled oracle for ordinary records that retain per-head Q scales."""
    batch, query_count = (int(dim) for dim in q_records.shape[:2])
    q_tiles = (
        query_count + INDEXER_PREFILL_Q_TILE - 1
    ) // INDEXER_PREFILL_Q_TILE
    k_tiles = (
        int(row_count) + INDEXER_PREFILL_K_TILE - 1
    ) // INDEXER_PREFILL_K_TILE
    (scores,) = _tiled_score_oracle_kernel()(
        inputs=[
            q_records,
            weights,
            rows.records,
            rows.block_table,
            int(row_start),
            int(row_count),
            int(rows.block_size),
            int(query_count),
        ],
        template=[],
        grid=(q_tiles * k_tiles * INDEXER_PREFILL_SCORE_THREADS, 1, 1),
        threadgroup=(INDEXER_PREFILL_SCORE_THREADS, 1, 1),
        output_shapes=[(batch, query_count, int(row_count))],
        output_dtypes=[mx.float32],
    )
    return scores


def _run_paged_indexer_score_slice(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    row_start: int,
    row_count: int,
    *,
    kernel,
) -> mx.array:
    batch, query_count = (int(dim) for dim in q_records.shape[:2])
    q_tiles = (
        query_count + INDEXER_PREFILL_Q_TILE - 1
    ) // INDEXER_PREFILL_Q_TILE
    k_tiles = (
        int(row_count) + INDEXER_PREFILL_K_TILE - 1
    ) // INDEXER_PREFILL_K_TILE
    (scores,) = kernel(
        inputs=[
            q_records,
            weights,
            rows.records,
            rows.block_table,
            int(row_start),
            int(row_count),
            int(rows.block_size),
            int(query_count),
        ],
        template=[],
        grid=(q_tiles * k_tiles * INDEXER_PREFILL_SCORE_THREADS, 1, 1),
        threadgroup=(INDEXER_PREFILL_SCORE_THREADS, 1, 1),
        output_shapes=[(batch, query_count, int(row_count))],
        output_dtypes=[mx.float32],
    )
    return scores


def _run_paged_indexer_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    return _run_paged_indexer_score_slice_oracle(
        _pack_indexer132(queries),
        weights,
        rows,
        0,
        int(rows.length),
    )


def _run_paged_indexer_tiled_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    return _run_paged_indexer_tiled_score_slice_oracle(
        _pack_indexer132(queries),
        weights,
        rows,
        0,
        int(rows.length),
    )


def _iter_prefill_k_tiles(row_count: int, score_chunk_rows: int):
    for row_start in range(0, int(row_count), int(score_chunk_rows)):
        yield row_start, min(int(score_chunk_rows), int(row_count) - row_start)


def _iter_prefill_query_tiles(
    *,
    query_count: int,
    row_count: int,
    score_chunk_rows: int,
    max_logits_bytes: int = INDEXER_PREFILL_MAX_LOGITS_BYTES,
):
    """Apply vLLM's M*N*4 logits cap only to oversized single requests."""
    query_count = int(query_count)
    max_score_rows = min(int(row_count), int(score_chunk_rows))
    if max_score_rows <= 0:
        yield 0, query_count
        return
    max_queries = max(1, int(max_logits_bytes) // (max_score_rows * 4))
    for query_start in range(0, query_count, max_queries):
        yield query_start, min(max_queries, query_count - query_start)


def _run_paged_indexer_records_topk(
    q_records: mx.array,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
    score_chunk_rows: int,
    query_count: int,
    score_slice,
    radix_fold,
) -> MiaTopKSelection:
    """Exact tiled-MMA scorer plus unordered fixed top-k winner set."""
    n_rows = int(rows.length)
    topk = int(topk)
    score_chunk_rows = int(score_chunk_rows)
    causal_lengths = mx.minimum(
        (positions + 1) // int(compress_ratio),
        n_rows,
    )[None]
    output_lengths = mx.minimum(causal_lengths, topk).astype(mx.int32)
    query_tiles = tuple(
        _iter_prefill_query_tiles(
            query_count=query_count,
            row_count=n_rows,
            score_chunk_rows=score_chunk_rows,
        )
    )

    # The ordinary exact route owns the complete query axis and performs only
    # score+fold per K supertile.  Only a single request above vLLM's pinned
    # 512-MiB logits cap enters the source-style query-subchunk route below.
    if len(query_tiles) == 1:
        carry_indices = _run_paged_indexer_records_topk_query_tile(
            q_records,
            weights,
            rows,
            causal_lengths,
            workspace=workspace,
            score_chunk_rows=score_chunk_rows,
            query_count=query_count,
            sentinel=n_rows,
            score_slice=score_slice,
            radix_fold=radix_fold,
        )
        return MiaTopKSelection(indices=carry_indices, lengths=output_lengths)

    index_tiles = []
    for query_start, tile_queries in query_tiles:
        query_stop = query_start + tile_queries
        index_tiles.append(
            _run_paged_indexer_records_topk_query_tile(
                q_records[:, query_start:query_stop],
                weights[:, query_start:query_stop],
                rows,
                causal_lengths[:, query_start:query_stop],
                workspace=workspace,
                score_chunk_rows=score_chunk_rows,
                query_count=tile_queries,
                sentinel=n_rows,
                score_slice=score_slice,
                radix_fold=radix_fold,
            )
        )
    return MiaTopKSelection(
        indices=mx.concatenate(index_tiles, axis=1),
        lengths=output_lengths,
    )


def _run_paged_indexer_records_topk_query_tile(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    causal_lengths: mx.array,
    *,
    workspace: MiaIndexerWorkspace,
    score_chunk_rows: int,
    query_count: int,
    sentinel: int,
    score_slice,
    radix_fold,
) -> mx.array:
    """Fold tiled-MMA score supertiles; output order is not a sort contract."""
    carry_scores, carry_indices = workspace.seeds(query_count)
    has_carry = False
    for row_start, row_count in _iter_prefill_k_tiles(
        int(rows.length), score_chunk_rows
    ):
        scores = score_slice(
            q_records,
            weights,
            rows,
            row_start,
            row_count,
        )
        carry_scores, carry_indices = radix_fold(
            scores,
            carry_scores,
            carry_indices,
            causal_lengths,
            row_start=row_start,
            score_indices=None,
            has_carry=has_carry,
            sentinel=sentinel,
        )
        has_carry = True
    return carry_indices


def _run_paged_indexer_records_decode_topk(
    q_records: mx.array,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
    query_count: int,
    decode_candidates,
    radix_fold,
) -> MiaTopKSelection:
    """Direct installed decode selector over already-qualified Q records."""
    n_rows = int(rows.length)
    causal_lengths = mx.minimum(
        (positions + 1) // int(compress_ratio),
        n_rows,
    )[None]
    output_lengths = mx.minimum(causal_lengths, int(topk)).astype(mx.int32)
    candidate_values, candidate_indices = decode_candidates(
        q_records,
        weights,
        rows,
        causal_lengths,
    )
    candidate_width = int(candidate_values.shape[2]) * int(topk)
    values = candidate_values.reshape(1, query_count, candidate_width)
    indices = candidate_indices.reshape(1, query_count, candidate_width)
    if candidate_width > int(topk):
        empty_values, empty_indices = workspace.seeds(query_count)
        merge_lengths = mx.full(
            (1, query_count), candidate_width, dtype=mx.int32
        )
        _, indices = radix_fold(
            values,
            empty_values,
            empty_indices,
            merge_lengths,
            row_start=0,
            score_indices=indices,
            has_carry=False,
            sentinel=n_rows,
        )
    return MiaTopKSelection(indices=indices, lengths=output_lengths)


def _run_paged_indexer_records_m6_topk(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    causal_lengths: mx.array,
    *,
    topk: int,
    workspace: MiaIndexerWorkspace,
    query_count: int,
    decode_candidates,
    radix_fold,
) -> MiaTopKSelection:
    """Direct physical-M6 selector with request-owned causal lengths."""
    n_rows = int(rows.length)
    causal_lengths = causal_lengths[None]
    output_lengths = mx.minimum(causal_lengths, int(topk)).astype(mx.int32)
    candidate_values, candidate_indices = decode_candidates(
        q_records,
        weights,
        rows,
        causal_lengths,
    )
    candidate_width = int(candidate_values.shape[2]) * int(topk)
    values = candidate_values.reshape(1, query_count, candidate_width)
    indices = candidate_indices.reshape(1, query_count, candidate_width)
    if candidate_width > int(topk):
        empty_values, empty_indices = workspace.seeds(query_count)
        merge_lengths = mx.full(
            (1, query_count), candidate_width, dtype=mx.int32
        )
        _, indices = radix_fold(
            values,
            empty_values,
            empty_indices,
            merge_lengths,
            row_start=0,
            score_indices=indices,
            has_carry=False,
            sentinel=n_rows,
        )
    return MiaTopKSelection(indices=indices, lengths=output_lengths)


def _run_installed_paged_indexer_phase_topk(
    queries: MiaIndexerQueryRecords,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
    score_chunk_rows: int,
    score_slice,
    radix_fold,
    decode_candidates,
) -> MiaTopKSelection:
    """Phase-only route for the installed Mia record type."""
    q_records = queries.records
    query_count = int(q_records.shape[1])
    if current_attention_phase() == "prefill":
        return _run_paged_indexer_records_topk(
            q_records,
            weights,
            positions,
            rows,
            topk=topk,
            compress_ratio=compress_ratio,
            workspace=workspace,
            score_chunk_rows=score_chunk_rows,
            query_count=query_count,
            score_slice=score_slice,
            radix_fold=radix_fold,
        )
    return _run_paged_indexer_records_decode_topk(
        q_records,
        weights,
        positions,
        rows,
        topk=topk,
        compress_ratio=compress_ratio,
        workspace=workspace,
        query_count=query_count,
        decode_candidates=decode_candidates,
        radix_fold=radix_fold,
    )


def paged_indexer_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    """Validated construction/oracle boundary for the direct paged indexer."""
    if not mx.metal.is_available():
        raise RuntimeError("Mia paged indexer requires Metal")
    if tuple(int(dim) for dim in queries.shape[:1]) != (1,) or tuple(
        int(dim) for dim in queries.shape[2:]
    ) != (INDEXER_HEADS, INDEXER_HEAD_DIM):
        raise ValueError("Mia indexer queries must be [1, rows, 64, 128]")
    if tuple(int(dim) for dim in weights.shape) != tuple(
        int(dim) for dim in queries.shape[:3]
    ):
        raise ValueError("Mia indexer weights must be [1, rows, 64]")
    if rows.record_bytes != INDEXER_RECORD_BYTES or rows.records.dtype != mx.uint8:
        raise ValueError("invalid Mia paged indexer records")
    return _run_paged_indexer_scores(queries, weights, rows)


def paged_indexer_tiled_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    """Validated oracle boundary for Mia's bounded tiled prefill scorer."""
    if not mx.metal.is_available():
        raise RuntimeError("Mia tiled paged indexer requires Metal")
    if tuple(int(dim) for dim in queries.shape[:1]) != (1,) or tuple(
        int(dim) for dim in queries.shape[2:]
    ) != (INDEXER_HEADS, INDEXER_HEAD_DIM):
        raise ValueError("Mia indexer queries must be [1, rows, 64, 128]")
    if tuple(int(dim) for dim in weights.shape) != tuple(
        int(dim) for dim in queries.shape[:3]
    ):
        raise ValueError("Mia indexer weights must be [1, rows, 64]")
    if rows.record_bytes != INDEXER_RECORD_BYTES or rows.records.dtype != mx.uint8:
        raise ValueError("invalid Mia paged indexer records")
    return _run_paged_indexer_tiled_scores(queries, weights, rows)


def install_paged_indexer_scores(*, heads: int, head_dim: int):
    observed = (int(heads), int(head_dim))
    expected = (INDEXER_HEADS, INDEXER_HEAD_DIM)
    if observed != expected:
        raise ValueError(f"unsupported Mia paged indexer geometry: {observed} != {expected}")
    if not mx.metal.is_available():
        raise RuntimeError("Mia paged indexer installation requires Metal")
    _pack_kernel()
    _oracle_score_kernel()
    return _run_paged_indexer_scores


def install_paged_indexer_topk(
    *,
    heads: int,
    head_dim: int,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
):
    observed = (int(heads), int(head_dim), int(topk), int(compress_ratio))
    expected = (INDEXER_HEADS, INDEXER_HEAD_DIM, INDEXER_TOPK, 4)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia paged indexer top-k geometry: {observed} != {expected}"
        )
    if (
        int(workspace.topk) != INDEXER_TOPK
        or int(workspace.max_query_rows) <= 0
        or int(workspace.sentinel) <= 0
    ):
        raise ValueError("the Mia paged indexer workspace geometry is invalid")
    if not mx.metal.is_available():
        raise RuntimeError("Mia paged indexer top-k installation requires Metal")
    score_slice = partial(
        _run_paged_indexer_score_slice,
        kernel=_tiled_score_kernel(),
    )
    radix_fold = partial(
        _run_radix_fold,
        kernel=_radix_fold_kernel(),
    )
    decode_candidates = partial(
        _run_fused_decode_candidates,
        kernel=_fused_decode_candidates_kernel(),
    )

    return partial(
        _run_installed_paged_indexer_phase_topk,
        topk=int(topk),
        compress_ratio=int(compress_ratio),
        workspace=workspace,
        score_chunk_rows=INDEXER_PREFILL_SCORE_CHUNK_ROWS,
        score_slice=score_slice,
        radix_fold=radix_fold,
        decode_candidates=decode_candidates,
    )


def install_paged_indexer_m6_topk(
    *,
    heads: int,
    head_dim: int,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
):
    """Install the decode-only physical-M6 selector without phase routing."""
    observed = (int(heads), int(head_dim), int(topk), int(compress_ratio))
    expected = (INDEXER_HEADS, INDEXER_HEAD_DIM, INDEXER_TOPK, 4)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia paged indexer M6 geometry: {observed} != {expected}"
        )
    if (
        int(workspace.topk) != INDEXER_TOPK
        or int(workspace.max_query_rows) < 6
        or int(workspace.sentinel) <= 0
    ):
        raise ValueError("the Mia paged indexer M6 workspace geometry is invalid")
    if not mx.metal.is_available():
        raise RuntimeError("Mia paged indexer M6 installation requires Metal")
    radix_fold = partial(
        _run_radix_fold,
        kernel=_radix_fold_kernel(),
    )
    decode_candidates = partial(
        _run_fused_decode_candidates,
        kernel=_fused_decode_candidates_kernel(),
    )
    return partial(
        _run_paged_indexer_records_m6_topk,
        topk=int(topk),
        workspace=workspace,
        query_count=6,
        decode_candidates=decode_candidates,
        radix_fold=radix_fold,
    )
