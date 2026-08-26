"""Direct sparse MLA over Mia DeepSeek-V4 ``stock432`` NVFP4 records."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from functools import partial

import mlx.core as mx

from mtplx.deepseek_v4_nvfp4_kv import (
    FixedMiaNVFP4WindowRecords,
    PagedMiaNVFP4Records,
)


_HEADS = 64
_HEAD_DIM = 512
_WINDOW = 128
_RECORD_BYTES = 432
_LANES = 32
_VALUES_PER_LANE = 16
_DECODE_HEADS_PER_GROUP = 16
_DECODE_CANDIDATE_TILE = 64
_DECODE_METAL_PANEL = 32
_DECODE_QK_GROUPS = 4
_DECODE_PV_GROUPS = 8
_DECODE_MATH_THREADS = _DECODE_PV_GROUPS * _LANES
_DECODE_NAX_GROUPS = 9
_DECODE_NAX_THREADS = _DECODE_NAX_GROUPS * _LANES
_DECODE_NAX_SCRATCH_BYTES = 30 * 1024
_DECODE_SECOND_SCORE_RANGE = (0, 2_048)
_DECODE_FIRST_SCORE_RANGE = (16_384, 18_432)
_DECODE_PROBABILITY_RANGE = (18_432, 20_480)
_PREFILL_HEADS_PER_GROUP = 16
_PREFILL_CANDIDATE_TILE = 32
_PREFILL_QK_GROUPS = 4
_PREFILL_NAX_GROUPS = 8
_PREFILL_NAX_THREADS = _PREFILL_NAX_GROUPS * _LANES
_PREFILL_NAX_SCRATCH_BYTES = 28 * 1024
_MAX_QUERY_ROWS = 8_224
_DSPARK_ROWS = 5

_ROUTE_WINDOW = "window"
_ROUTE_INDEXED_PAGED = "indexed_paged"
_ROUTE_SEQUENTIAL_PAGED = "sequential_paged"
_ROUTE_INDEXED_CONTIGUOUS = "indexed_contiguous"
_ROUTE_SEQUENTIAL_CONTIGUOUS = "sequential_contiguous"
_ROUTE_DSPARK = "dspark"
_ROUTE_IDS = {
    _ROUTE_WINDOW: 0,
    _ROUTE_INDEXED_PAGED: 1,
    _ROUTE_SEQUENTIAL_PAGED: 2,
    _ROUTE_INDEXED_CONTIGUOUS: 3,
    _ROUTE_SEQUENTIAL_CONTIGUOUS: 4,
    _ROUTE_DSPARK: 5,
}


@dataclass(frozen=True)
class MiaMLAWorkspace:
    """Shared fixed inputs for invariant empty MLA operands."""

    max_query_rows: int
    dummy_record: mx.array
    dummy_block_table: mx.array
    dummy_indices: mx.array
    identity_index_row: mx.array
    empty_lengths: mx.array

    def indices(self, query_count: int) -> mx.array:
        return self.dummy_indices[:, : int(query_count)]

    def lengths(self, query_count: int) -> mx.array:
        return self.empty_lengths[:, : int(query_count)]

    def identity_indices(self, query_count: int) -> mx.array:
        return mx.broadcast_to(
            self.identity_index_row,
            (1, int(query_count), int(self.identity_index_row.shape[2])),
        )


@lru_cache(maxsize=1)
def mia_mla_workspace() -> MiaMLAWorkspace:
    return MiaMLAWorkspace(
        max_query_rows=_MAX_QUERY_ROWS,
        dummy_record=mx.zeros((1, 1, _RECORD_BYTES), dtype=mx.uint8),
        dummy_block_table=mx.zeros((1,), dtype=mx.int32),
        dummy_indices=mx.zeros((1, _MAX_QUERY_ROWS, 1), dtype=mx.int32),
        identity_index_row=mx.arange(512, dtype=mx.int32)[None, None],
        empty_lengths=mx.zeros((1, _MAX_QUERY_ROWS), dtype=mx.int32),
    )


_HEADER = f"""
    using namespace metal;

    constant constexpr int MTPLX_H = {_HEADS};
    constant constexpr int MTPLX_D = {_HEAD_DIM};
    constant constexpr int MTPLX_WINDOW = {_WINDOW};
    constant constexpr int MTPLX_RECORD = {_RECORD_BYTES};
    constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};

    inline float mtplx_dsv4_e4m3(uchar raw) {{
        uint exponent = (uint(raw) >> 3) & 0x0fu;
        uint mantissa = uint(raw) & 0x07u;
        float magnitude = exponent == 0u
            ? float(mantissa) * 0.001953125f
            : (1.0f + float(mantissa) * 0.125f)
                * exp2(float(int(exponent) - 7));
        return (uint(raw) & 0x80u) != 0u ? -magnitude : magnitude;
    }}

    inline float mtplx_dsv4_e2m1(uchar raw) {{
        constexpr float values[8] = {{0.0f, 0.5f, 1.0f, 1.5f,
                                      2.0f, 3.0f, 4.0f, 6.0f}};
        float magnitude = values[uint(raw) & 0x07u];
        return (uint(raw) & 0x08u) != 0u ? -magnitude : magnitude;
    }}

    inline float mtplx_dsv4_latent(
        const device uchar* record,
        uint dim,
        float scale
    ) {{
        uchar packed = record[dim >> 1];
        uchar code = (dim & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
        return mtplx_dsv4_e2m1(code) * scale;
    }}

    inline bfloat mtplx_dsv4_device_value_bf16(
        const device uchar* record,
        uint dim,
        float latent_scale
    ) {{
        return bfloat(mtplx_dsv4_latent(record, dim, latent_scale));
    }}

"""


_PREFILL_HEADER = _HEADER.replace(
    f"constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};",
    f"constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};\n"
    f"    constant constexpr int MTPLX_LANES = {_LANES};\n"
    f"    constant constexpr int MTPLX_PREFILL_HEADS = "
    f"{_PREFILL_HEADS_PER_GROUP};\n"
    f"    constant constexpr int MTPLX_PREFILL_TILE = {_PREFILL_CANDIDATE_TILE};\n"
    f"    constant constexpr int MTPLX_PREFILL_QK_GROUPS = "
    f"{_PREFILL_QK_GROUPS};\n"
    f"    constant constexpr int MTPLX_PREFILL_NAX_GROUPS = "
    f"{_PREFILL_NAX_GROUPS};\n"
    f"    constant constexpr int MTPLX_PREFILL_NAX_THREADS = "
    f"{_PREFILL_NAX_THREADS};\n"
    f"    constant constexpr int MTPLX_PREFILL_NAX_SCRATCH = "
    f"{_PREFILL_NAX_SCRATCH_BYTES};",
)

_DECODE_HEADER = _HEADER.replace(
    f"constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};",
    f"constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};\n"
    f"    constant constexpr int MTPLX_LANES = {_LANES};\n"
    f"    constant constexpr int MTPLX_DECODE_HEADS = {_DECODE_HEADS_PER_GROUP};\n"
    f"    constant constexpr int MTPLX_DECODE_TILE = {_DECODE_CANDIDATE_TILE};\n"
    f"    constant constexpr int MTPLX_DECODE_PANEL = {_DECODE_METAL_PANEL};\n"
    f"    constant constexpr int MTPLX_DECODE_QK_GROUPS = {_DECODE_QK_GROUPS};\n"
    f"    constant constexpr int MTPLX_DECODE_PV_GROUPS = {_DECODE_PV_GROUPS};\n"
    f"    constant constexpr int MTPLX_DECODE_GROUPS = {_DECODE_NAX_GROUPS};\n"
    f"    constant constexpr int MTPLX_DECODE_THREADS = {_DECODE_NAX_THREADS};\n"
    f"    constant constexpr int MTPLX_DECODE_SCRATCH = "
    f"{_DECODE_NAX_SCRATCH_BYTES};",
)


_DECODE_NAX_SOURCE = r"""
        using namespace mpp::tensor_ops;

        uint lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint thread_index = simd_group * MTPLX_LANES + lane;
        uint group = threadgroup_position_in_grid.x;
        uint query_count = uint(n_queries);
        uint head_groups = uint(MTPLX_H / MTPLX_DECODE_HEADS);
        uint groups_per_batch = head_groups * query_count;
        uint batch = group / groups_per_batch;
        uint within_batch = group - batch * groups_per_batch;
        uint head_group = within_batch / query_count;
        uint query_row = within_batch - head_group * query_count;
        uint head_base = head_group * uint(MTPLX_DECODE_HEADS);

        // The image-selected generic decode CTA owns sixteen query heads and
        // shares each resolved KV record across them. Nine Metal SIMD groups
        // preserve its 288-thread owner: eight math plus one coordination group.
        threadgroup uchar scratch[MTPLX_DECODE_SCRATCH];
        threadgroup bfloat* q_shared =
            reinterpret_cast<threadgroup bfloat*>(scratch);

        threadgroup float running_max[MTPLX_DECODE_HEADS];
        threadgroup float running_sum[MTPLX_DECODE_HEADS];
        threadgroup float row_correction[MTPLX_DECODE_HEADS];
        threadgroup uint tile_rows[MTPLX_DECODE_TILE];
        threadgroup uchar tile_kinds[MTPLX_DECODE_TILE];
        if (thread_index < uint(MTPLX_DECODE_HEADS)) {
            running_max[thread_index] =
                sinks[head_base + thread_index] * MTPLX_LOG2E;
            running_sum[thread_index] = 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr auto qk_desc = matmul2d_descriptor(
            16, 32, 16, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate
        );
        constexpr auto pv_desc = matmul2d_descriptor(
            16, 32, 16, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate
        );
        matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
        matmul2d<pv_desc, metal::execution_simdgroup> pv_op;

        auto pv_acc_lo = pv_op.template get_destination_cooperative_tensor<
            tensor<threadgroup bfloat, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup bfloat, extents<int, 32, 16>, tensor_inline>,
            float
        >();
        auto pv_acc_hi = pv_op.template get_destination_cooperative_tensor<
            tensor<threadgroup bfloat, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup bfloat, extents<int, 32, 16>, tensor_inline>,
            float
        >();
        for (uint16_t index = 0; index < pv_acc_lo.get_capacity(); ++index) {
            pv_acc_lo[index] = 0.0f;
            pv_acc_hi[index] = 0.0f;
        }

        uint visible_window;
        uint compressed_length;
        int first_window;
        size_t window_batch;
        size_t selected_base;
        size_t compressed_batch;
        if (MTPLX_ROUTE == 5) {
            visible_window = min(uint(n_compressed_records), uint(MTPLX_WINDOW));
            compressed_length = uint(MTPLX_DSPARK_ROWS);
            first_window = max(0, int(window_start) - int(visible_window));
            window_batch = size_t(batch) * size_t(MTPLX_DSPARK_ROWS);
            selected_base = 0;
            compressed_batch = size_t(batch) * size_t(MTPLX_WINDOW);
        } else {
            int query_position = query_positions[query_row];
            first_window = max(
                0, query_position - int(window_start) - MTPLX_WINDOW + 1
            );
            int window_end = min(
                int(n_window_records),
                query_position - int(window_start) + 1
            );
            visible_window = uint(max(0, window_end - first_window));
            compressed_length = MTPLX_ROUTE == 0
                ? 0u
                : uint(compressed_lengths[batch * query_count + query_row]);
            window_batch = MTPLX_WINDOW_PAGED
                ? 0u
                : size_t(batch) * size_t(n_window_records);
            selected_base = (
                size_t(batch * query_count + query_row)
                * size_t(MTPLX_SELECTED_WIDTH)
            );
            compressed_batch = size_t(batch) * size_t(n_compressed_records);
        }
        uint total_candidates = visible_window + compressed_length;

        for (uint tile_start = 0u; tile_start < total_candidates;
             tile_start += uint(MTPLX_DECODE_TILE)) {
            uint tile_count = min(
                uint(MTPLX_DECODE_TILE), total_candidates - tile_start
            );

            // Panel 1 aliases its FP32 scores over scratch[0:2048], which is
            // part of q_shared. Reload the immutable query at every candidate
            // tile so later tiles never consume the preceding tile's scores.
            for (uint index = thread_index;
                 index < 16u * uint(MTPLX_D);
                 index += uint(MTPLX_DECODE_THREADS)) {
                uint local_head = index / uint(MTPLX_D);
                uint dim = index - local_head * uint(MTPLX_D);
                if (local_head < uint(MTPLX_DECODE_HEADS)) {
                    size_t query_index = MTPLX_QUERY_TOKEN_MAJOR
                        ? (
                            (size_t(batch * query_count + query_row)
                             * size_t(MTPLX_H)
                             + size_t(head_base + local_head))
                            * size_t(MTPLX_D) + size_t(dim)
                        )
                        : (
                            (size_t(batch * uint(MTPLX_H)
                             + head_base + local_head)
                             * size_t(query_count) + size_t(query_row))
                            * size_t(MTPLX_D) + size_t(dim)
                        );
                    q_shared[index] = bfloat(queries[query_index]);
                } else {
                    q_shared[index] = bfloat(0.0f);
                }
            }

            if (thread_index < uint(MTPLX_DECODE_TILE)) {
                uint candidate = thread_index;
                uint physical_row = 0u;
                uchar kind = 0u;
                if (candidate < tile_count) {
                    uint global_candidate = tile_start + candidate;
                    if (MTPLX_ROUTE == 5) {
                        if (global_candidate < visible_window) {
                            kind = 1u;
                            uint absolute_position = uint(first_window)
                                + global_candidate;
                            physical_row = uint(compressed_batch)
                                + absolute_position % uint(MTPLX_WINDOW);
                        } else {
                            physical_row = uint(window_batch)
                                + global_candidate - visible_window;
                        }
                    } else if (global_candidate < visible_window) {
                        if (MTPLX_WINDOW_PAGED) {
                            size_t absolute_position = size_t(window_start)
                                + size_t(first_window)
                                + size_t(global_candidate);
                            uint slot = uint(
                                absolute_position % size_t(window_capacity)
                            );
                            uint logical_block = slot
                                / uint(window_block_size);
                            uint row_in_block = slot
                                - logical_block * uint(window_block_size);
                            uint physical_block = uint(
                                window_block_table[logical_block]
                            );
                            physical_row = physical_block
                                * uint(window_block_size) + row_in_block;
                        } else {
                            physical_row = uint(window_batch)
                                + uint(first_window) + global_candidate;
                        }
                    } else {
                        kind = 1u;
                        uint slot = global_candidate - visible_window;
                        uint row = (MTPLX_ROUTE == 1 || MTPLX_ROUTE == 3)
                            ? uint(compressed_indices[
                                selected_base + size_t(slot)
                            ])
                            : slot;
                        physical_row = row;
                        if (MTPLX_ROUTE == 1 || MTPLX_ROUTE == 2) {
                            uint logical_block =
                                row / uint(MTPLX_BLOCK_SIZE);
                            uint row_in_block = row
                                - logical_block * uint(MTPLX_BLOCK_SIZE);
                            uint physical_block = uint(
                                compressed_block_table[logical_block]
                            );
                            physical_row = physical_block
                                * uint(MTPLX_BLOCK_SIZE) + row_in_block;
                        } else {
                            physical_row += uint(compressed_batch);
                        }
                    }
                }
                tile_rows[candidate] = physical_row;
                tile_kinds[candidate] = kind;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // The immutable image selects the generic H16/288 owner. Metal's
            // NAX N dimension is 32, so its logical tile64 is evaluated as two
            // bounded N32 panels. Query ownership and FP32 softmax state remain
            // shared by all sixteen heads across both panels.
            threadgroup float* first_panel_scores =
                reinterpret_cast<threadgroup float*>(scratch + 16384);
            threadgroup bfloat* qk_values =
                reinterpret_cast<threadgroup bfloat*>(scratch + 18432);
            threadgroup float* qk_partials =
                reinterpret_cast<threadgroup float*>(scratch + 22528);
            for (uint candidate_half = 0u; candidate_half < 2u;
                 ++candidate_half) {
                uint panel_start = candidate_half * uint(MTPLX_DECODE_PANEL);
                uint panel_count = tile_count > panel_start
                    ? min(uint(MTPLX_DECODE_PANEL), tile_count - panel_start)
                    : 0u;
                if (panel_count == 0u) {
                    continue;
                }
                threadgroup float* panel_scores = candidate_half == 0u
                    ? first_panel_scores
                    : reinterpret_cast<threadgroup float*>(scratch);
                if (simd_group < uint(MTPLX_DECODE_QK_GROUPS)) {
                    threadgroup bfloat* group_values = qk_values
                        + simd_group * 32u * 16u;
                    threadgroup float* group_partial = qk_partials
                        + simd_group * 16u * 32u;
                    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> Q(
                        q_shared,
                        dextents<int, 2>{MTPLX_D, 16},
                        array<int, 2>{1, MTPLX_D}
                    );
                    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> K(
                        group_values,
                        dextents<int, 2>{32, 16},
                        array<int, 2>{1, 32}
                    );
                    tensor<threadgroup float, dextents<int, 2>, tensor_inline> S(
                        group_partial,
                        dextents<int, 2>{32, 16},
                        array<int, 2>{1, 32}
                    );
                    auto qk_acc = qk_op.template get_destination_cooperative_tensor<
                        tensor<threadgroup bfloat,
                               extents<int, 16, 16>, tensor_inline>,
                        tensor<threadgroup bfloat,
                               extents<int, 32, 16>, tensor_inline>,
                        float
                    >();
                    for (uint16_t index = 0;
                         index < qk_acc.get_capacity(); ++index) {
                        qk_acc[index] = 0.0f;
                    }
                    uint k_begin = simd_group * 128u;
                    for (uint k0 = k_begin; k0 < k_begin + 128u; k0 += 16u) {
                        uint candidate = lane;
                        uint tile_candidate = panel_start + candidate;
                        const device uchar* record = tile_kinds[tile_candidate] == 0u
                            ? window_records
                                + size_t(tile_rows[tile_candidate])
                                    * size_t(MTPLX_RECORD)
                            : compressed_records
                                + size_t(tile_rows[tile_candidate])
                                    * size_t(MTPLX_RECORD);
                        if (tile_candidate < tile_count) {
                            if (k0 < 448u) {
                                float latent_scale = mtplx_dsv4_e4m3(
                                    record[256u + (k0 >> 4)]
                                );
                                for (uint element = 0u; element < 16u; ++element) {
                                    group_values[element * 32u + candidate] =
                                        mtplx_dsv4_device_value_bf16(
                                            record, k0 + element, latent_scale
                                        );
                                }
                            } else {
                                const device bfloat* rope = reinterpret_cast<
                                    const device bfloat*>(record + 304u);
                                for (uint element = 0u; element < 16u; ++element) {
                                    group_values[element * 32u + candidate] =
                                        rope[k0 + element - 448u];
                                }
                            }
                        } else {
                            for (uint element = 0u; element < 16u; ++element) {
                                group_values[element * 32u + candidate] =
                                    bfloat(0.0f);
                            }
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        auto q_tile = Q.template slice<16, 16>(k0, 0);
                        auto k_tile = K.template slice<32, 16>(0, 0);
                        qk_op.run(q_tile, k_tile, qk_acc);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    auto partial_tile = S.template slice<32, 16>(0, 0);
                    qk_acc.store(partial_tile);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint offset = thread_index;
                     offset < uint(MTPLX_DECODE_HEADS) * 32u;
                     offset += uint(MTPLX_DECODE_THREADS)) {
                    uint head = offset / 32u;
                    uint candidate = offset - head * 32u;
                    float sum01 = qk_partials[head * 32u + candidate]
                        + qk_partials[16u * 32u + head * 32u + candidate];
                    float sum23 = qk_partials[
                            2u * 16u * 32u + head * 32u + candidate
                        ] + qk_partials[
                            3u * 16u * 32u + head * 32u + candidate
                        ];
                    panel_scores[head * uint(MTPLX_DECODE_PANEL) + candidate] =
                        (sum01 + sum23) * float(scale) * MTPLX_LOG2E;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Both FP32 score panels now exist: panel 0 at 16 KiB and panel 1
            // in the query arena, which became dead after its QK completed.
            // Compute one tile64 maximum before the sole BF16 P boundary.
            threadgroup float* second_panel_scores =
                reinterpret_cast<threadgroup float*>(scratch);
            threadgroup bfloat* probabilities =
                reinterpret_cast<threadgroup bfloat*>(scratch + 18432);
            if (thread_index < uint(MTPLX_DECODE_HEADS)) {
                uint row = thread_index;
                uint score_base = row * uint(MTPLX_DECODE_PANEL);
                float old_max = running_max[row];
                float next_max = old_max;
                for (uint candidate = 0u; candidate < tile_count; ++candidate) {
                    float score = candidate < uint(MTPLX_DECODE_PANEL)
                        ? first_panel_scores[score_base + candidate]
                        : second_panel_scores[
                            score_base + candidate - uint(MTPLX_DECODE_PANEL)
                        ];
                    next_max = max(next_max, score);
                }
                float correction = fast::exp2(old_max - next_max);
                float next_sum = running_sum[row] * correction;
                for (uint candidate = 0u; candidate < tile_count; ++candidate) {
                    float score = candidate < uint(MTPLX_DECODE_PANEL)
                        ? first_panel_scores[score_base + candidate]
                        : second_panel_scores[
                            score_base + candidate - uint(MTPLX_DECODE_PANEL)
                        ];
                    next_sum += fast::exp2(score - next_max);
                }
                // This order is source-significant: correction is represented
                // in the final FP32 score frame before BF16 conversion.
                uint probability_base = row * uint(MTPLX_DECODE_TILE);
                for (uint candidate = 0u;
                     candidate < uint(MTPLX_DECODE_TILE); ++candidate) {
                    float score = candidate < uint(MTPLX_DECODE_PANEL)
                        ? first_panel_scores[score_base + candidate]
                        : second_panel_scores[
                            score_base + candidate - uint(MTPLX_DECODE_PANEL)
                        ];
                    float corrected_probability = candidate < tile_count
                        ? fast::exp2(score - next_max)
                        : 0.0f;
                    bfloat probability_bf16 = bfloat(corrected_probability);
                    probabilities[probability_base + candidate] =
                        probability_bf16;
                }
                row_correction[row] = correction;
                running_max[row] = next_max;
                running_sum[row] = next_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint matrix_row_base = ((lane & 7u) >> 1)
                + ((lane >> 4) * 4u);
            for (uint16_t index = 0;
                 index < pv_acc_lo.get_capacity(); ++index) {
                uint matrix_row = matrix_row_base
                    + (((uint(index) >> 2) & 1u) << 3);
                float correction = row_correction[matrix_row];
                pv_acc_lo[index] *= correction;
                pv_acc_hi[index] *= correction;
            }

            if (simd_group < uint(MTPLX_DECODE_PV_GROUPS)) {
                threadgroup bfloat* value_tiles =
                    reinterpret_cast<threadgroup bfloat*>(scratch + 2048);
                threadgroup bfloat* value_base = value_tiles
                    + simd_group * 32u * 16u;
                tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> P(
                    probabilities,
                    dextents<int, 2>{MTPLX_DECODE_TILE, 16},
                    array<int, 2>{1, MTPLX_DECODE_TILE}
                );
                tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> V(
                    value_base,
                    dextents<int, 2>{32, 16},
                    array<int, 2>{1, 32}
                );
                for (uint panel_start = 0u;
                     panel_start < uint(MTPLX_DECODE_TILE);
                     panel_start += uint(MTPLX_DECODE_PANEL)) {
                    uint panel_count = tile_count > panel_start
                        ? min(uint(MTPLX_DECODE_PANEL), tile_count - panel_start)
                        : 0u;
                    for (uint candidate_base = 0u;
                         candidate_base < uint(MTPLX_DECODE_PANEL);
                         candidate_base += 16u) {
                        auto probability_tile = P.template slice<16, 16>(
                            panel_start + candidate_base, 0
                        );
                        for (uint output_half = 0u; output_half < 2u;
                             ++output_half) {
                            uint candidate = lane >> 1;
                            uint half_group = lane & 1u;
                            uint dim_base = simd_group * 64u
                                + output_half * 32u + half_group * 16u;
                            uint panel_candidate = candidate_base + candidate;
                            uint tile_candidate = panel_start + panel_candidate;
                            const device uchar* record =
                                tile_kinds[tile_candidate] == 0u
                                ? window_records
                                    + size_t(tile_rows[tile_candidate])
                                        * size_t(MTPLX_RECORD)
                                : compressed_records
                                    + size_t(tile_rows[tile_candidate])
                                        * size_t(MTPLX_RECORD);
                            if (panel_candidate < panel_count) {
                                float latent_scale = mtplx_dsv4_e4m3(
                                    record[256u + (dim_base >> 4)]
                                );
                                for (uint element = 0u; element < 16u; ++element) {
                                    value_base[candidate * 32u
                                               + half_group * 16u + element] =
                                        mtplx_dsv4_device_value_bf16(
                                            record, dim_base + element, latent_scale
                                        );
                                }
                            } else {
                                for (uint element = 0u; element < 16u; ++element) {
                                    value_base[candidate * 32u
                                               + half_group * 16u + element] =
                                        bfloat(0.0f);
                                }
                            }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            auto value_tile = V.template slice<32, 16>(0, 0);
                            if (output_half == 0u) {
                                pv_op.run(probability_tile, value_tile, pv_acc_lo);
                            } else {
                                pv_op.run(probability_tile, value_tile, pv_acc_hi);
                            }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_group < uint(MTPLX_DECODE_PV_GROUPS)) {
            uint matrix_row_base = ((lane & 7u) >> 1)
                + ((lane >> 4) * 4u);
            uint matrix_col_base = ((lane & 1u) << 2)
                + (((lane >> 3) & 1u) << 3);
            for (uint16_t index = 0;
                 index < pv_acc_lo.get_capacity(); ++index) {
                uint matrix_row = matrix_row_base
                    + (((uint(index) >> 2) & 1u) << 3);
                uint matrix_col = matrix_col_base + (uint(index) & 3u)
                    + (uint(index) >> 3) * 16u;
                uint output_head = head_base + matrix_row;
                uint output_dim = simd_group * 64u + matrix_col;
                size_t output_base = MTPLX_QUERY_TOKEN_MAJOR
                    ? (
                        (size_t(batch * query_count + query_row)
                         * size_t(MTPLX_H) + size_t(output_head))
                        * size_t(MTPLX_D)
                    )
                    : (
                        (size_t(batch * uint(MTPLX_H) + output_head)
                         * size_t(query_count) + size_t(query_row))
                        * size_t(MTPLX_D)
                    );
                float inverse_sum = 1.0f / running_sum[matrix_row];
                out[output_base + size_t(output_dim)] =
                    T(pv_acc_lo[index] * inverse_sum);
                out[output_base + size_t(output_dim + 32u)] =
                    T(pv_acc_hi[index] * inverse_sum);
            }
        }
    """


@lru_cache(maxsize=None)
def _kernel(
    route: str,
    selected_width: int,
    block_size: int,
    window_paged: bool = False,
    query_token_major: bool = False,
):
    if route not in _ROUTE_IDS:
        raise ValueError(f"unsupported Mia decode route: {route!r}")
    header = (
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        + _DECODE_HEADER
        + f"\nconstant constexpr int MTPLX_ROUTE = {_ROUTE_IDS[route]};"
        + f"\nconstant constexpr int MTPLX_SELECTED_WIDTH = {int(selected_width)};"
        + f"\nconstant constexpr int MTPLX_BLOCK_SIZE = {int(block_size)};"
        + "\nconstant constexpr bool MTPLX_WINDOW_PAGED = "
        + ("true;" if window_paged else "false;")
        + "\nconstant constexpr bool MTPLX_QUERY_TOKEN_MAJOR = "
        + ("true;" if query_token_major else "false;")
        + "\nconstant constexpr float MTPLX_LOG2E = 1.4426950408889634f;"
        + f"\nconstant constexpr int MTPLX_DSPARK_ROWS = {_DSPARK_ROWS};\n"
    )
    window_layout = "paged_window" if window_paged else "contiguous_window"
    query_layout = "token_major" if query_token_major else "bhmd"
    return mx.fast.metal_kernel(
        name=(
            "mtplx_dsv4_mia_stock432_decode_h16_"
            f"{route}_sw{int(selected_width)}_bs{int(block_size)}_"
            f"{window_layout}_{query_layout}_v3"
        ),
        input_names=[
            "queries",
            "window_records",
            "window_block_table",
            "window_start",
            "window_capacity",
            "window_block_size",
            "query_positions",
            "compressed_records",
            "compressed_block_table",
            "compressed_indices",
            "compressed_lengths",
            "sinks",
            "scale",
            "n_queries",
            "n_window_records",
            "n_compressed_records",
        ],
        output_names=["out"],
        header=header,
        source=_DECODE_NAX_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _dspark_placeholder_inputs():
    return (
        mx.zeros((_DSPARK_ROWS,), dtype=mx.int32),
        mx.zeros((1,), dtype=mx.int32),
        mx.zeros((1, _DSPARK_ROWS, 1), dtype=mx.int32),
        mx.zeros((1, _DSPARK_ROWS), dtype=mx.int32),
    )


def _run_dspark_k5_nvfp4_mla(
    queries: mx.array,
    context_records: mx.array,
    draft_records: mx.array,
    prefix_length: int,
    sinks: mx.array,
    scale: float,
    *,
    kernel,
    query_positions: mx.array,
    block_table: mx.array,
    indices: mx.array,
    lengths: mx.array,
) -> mx.array:
    batch = int(queries.shape[0])
    context_count = min(int(prefix_length), _WINDOW)
    (output,) = kernel(
        inputs=[
            queries,
            draft_records,
            block_table,
            int(prefix_length),
            1,
            1,
            query_positions,
            context_records,
            block_table,
            indices,
            lengths,
            sinks,
            float(scale),
            _DSPARK_ROWS,
            _DSPARK_ROWS,
            context_count,
        ],
        template=[("T", mx.bfloat16)],
        grid=(
            batch
            * (_HEADS // _DECODE_HEADS_PER_GROUP)
            * _DSPARK_ROWS
            * _DECODE_NAX_THREADS,
            1,
            1,
        ),
        threadgroup=(_DECODE_NAX_THREADS, 1, 1),
        output_shapes=[(batch, _DSPARK_ROWS, _HEADS, _HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def _run_dspark_k5_nvfp4_mla_graph(
    queries: mx.array,
    context_records: mx.array,
    draft_records: mx.array,
    start_position: mx.array,
    sinks: mx.array,
    scale: float,
    *,
    kernel,
    query_positions: mx.array,
    block_table: mx.array,
    indices: mx.array,
    lengths: mx.array,
) -> mx.array:
    """Graph-safe K5 launch with the live position supplied as an array input."""

    batch = int(queries.shape[0])
    prefix_length = start_position[0]
    context_count = mx.minimum(
        prefix_length,
        mx.array(_WINDOW, dtype=start_position.dtype),
    )
    (output,) = kernel(
        inputs=[
            queries,
            draft_records,
            block_table,
            prefix_length,
            1,
            1,
            query_positions,
            context_records,
            block_table,
            indices,
            lengths,
            sinks,
            float(scale),
            _DSPARK_ROWS,
            _DSPARK_ROWS,
            context_count,
        ],
        template=[("T", mx.bfloat16)],
        grid=(
            batch
            * (_HEADS // _DECODE_HEADS_PER_GROUP)
            * _DSPARK_ROWS
            * _DECODE_NAX_THREADS,
            1,
            1,
        ),
        threadgroup=(_DECODE_NAX_THREADS, 1, 1),
        output_shapes=[(batch, _DSPARK_ROWS, _HEADS, _HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )
    return output


@lru_cache(maxsize=None)
def _prefill_nax_mg16_kernel(
    route: str,
    selected_width: int,
    block_size: int,
    window_paged: bool = False,
    query_token_major: bool = False,
):
    """Mia/SparkInfer NVFP4 prefill mapped to the M5 NAX tile geometry.

    SparkInfer owns sixteen query heads together, dequantizes each selected KV
    record once per tensor operand, runs BF16 QK and P.V tensor operations, and
    carries one FP32 online softmax over the SWA/indexed-cache union.  Metal's
    native NAX primitive is M16xN32xK16, so this implementation keeps the same
    ownership with a 32-candidate tile, four 128-wide QK splits, and eight PV
    SIMD groups that each own two 32-wide output fragments.
    """
    if route == _ROUTE_DSPARK or route not in _ROUTE_IDS:
        raise ValueError(f"unsupported Mia prefill route: {route!r}")
    source = r"""
        using namespace mpp::tensor_ops;

        uint lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint thread_index = simd_group * MTPLX_LANES + lane;
        uint group = threadgroup_position_in_grid.x;
        uint query_count = uint(n_queries);
        uint head_groups = uint(MTPLX_H / MTPLX_PREFILL_HEADS);
        uint groups_per_batch = head_groups * query_count;
        uint batch = group / groups_per_batch;
        uint within_batch = group - batch * groups_per_batch;
        uint head_group = within_batch / query_count;
        uint query_row = within_batch - head_group * query_count;
        uint head_base = head_group * uint(MTPLX_PREFILL_HEADS);
        int query_position = query_positions[query_row];

        // The query is invariant across all candidate tiles.  Load its sixteen
        // BF16 head rows once, matching SparkInfer's S0 ownership.
        threadgroup uchar scratch[MTPLX_PREFILL_NAX_SCRATCH];
        threadgroup bfloat* q_shared =
            reinterpret_cast<threadgroup bfloat*>(scratch);
        for (uint index = thread_index;
             index < uint(MTPLX_PREFILL_HEADS * MTPLX_D);
             index += uint(MTPLX_PREFILL_NAX_THREADS)) {
            uint local_head = index / uint(MTPLX_D);
            uint dim = index - local_head * uint(MTPLX_D);
            size_t query_index = MTPLX_QUERY_TOKEN_MAJOR
                ? (
                    (size_t(batch * query_count + query_row)
                     * size_t(MTPLX_H) + size_t(head_base + local_head))
                    * size_t(MTPLX_D) + size_t(dim)
                )
                : (
                    (size_t(batch * uint(MTPLX_H) + head_base + local_head)
                     * size_t(query_count) + size_t(query_row))
                    * size_t(MTPLX_D) + size_t(dim)
                );
            q_shared[index] = bfloat(queries[query_index]);
        }

        threadgroup float running_max[MTPLX_PREFILL_HEADS];
        threadgroup float running_sum[MTPLX_PREFILL_HEADS];
        threadgroup float row_correction[MTPLX_PREFILL_HEADS];
        threadgroup uint tile_rows[MTPLX_PREFILL_TILE];
        threadgroup uchar tile_kinds[MTPLX_PREFILL_TILE];
        if (thread_index < uint(MTPLX_PREFILL_HEADS)) {
            running_max[thread_index] =
                sinks[head_base + thread_index] * MTPLX_LOG2E;
            running_sum[thread_index] = 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr auto qk_desc = matmul2d_descriptor(
            16,
            32,
            16,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate
        );
        constexpr auto pv_desc = matmul2d_descriptor(
            16,
            32,
            16,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate
        );
        matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
        matmul2d<pv_desc, metal::execution_simdgroup> pv_op;

        auto pv_acc_lo = pv_op.template get_destination_cooperative_tensor<
            tensor<threadgroup bfloat, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup bfloat, extents<int, 32, 16>, tensor_inline>,
            float
        >();
        auto pv_acc_hi = pv_op.template get_destination_cooperative_tensor<
            tensor<threadgroup bfloat, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup bfloat, extents<int, 32, 16>, tensor_inline>,
            float
        >();
        for (uint16_t index = 0;
             index < pv_acc_lo.get_capacity(); ++index) {
            pv_acc_lo[index] = 0.0f;
            pv_acc_hi[index] = 0.0f;
        }

        int first_window = max(
            0,
            query_position - int(window_start) - MTPLX_WINDOW + 1
        );
        int window_end = min(
            int(n_window_records),
            query_position - int(window_start) + 1
        );
        uint visible_window = uint(window_end - first_window);
        uint compressed_length = MTPLX_ROUTE == 0
            ? 0u
            : uint(compressed_lengths[batch * query_count + query_row]);
        uint total_candidates = visible_window + compressed_length;
        size_t window_batch = MTPLX_WINDOW_PAGED
            ? 0u
            : size_t(batch) * size_t(n_window_records);
        size_t selected_base = (
            size_t(batch * query_count + query_row)
            * size_t(MTPLX_SELECTED_WIDTH)
        );
        size_t compressed_batch = size_t(batch) * size_t(n_compressed_records);

        for (uint tile_start = 0u; tile_start < total_candidates;
             tile_start += uint(MTPLX_PREFILL_TILE)) {
            uint tile_count = min(
                uint(MTPLX_PREFILL_TILE), total_candidates - tile_start
            );

            // Resolve the dual-cache union once for this candidate tile.  Every
            // QK/PV operand thereafter consumes the same physical-row table.
            if (thread_index < uint(MTPLX_PREFILL_TILE)) {
                uint candidate = thread_index;
                uint physical_row = 0u;
                uchar kind = 0u;
                if (candidate < tile_count) {
                    uint global_candidate = tile_start + candidate;
                    if (global_candidate < visible_window) {
                        if (MTPLX_WINDOW_PAGED) {
                            size_t absolute_position = size_t(window_start)
                                + size_t(first_window)
                                + size_t(global_candidate);
                            uint slot = uint(
                                absolute_position % size_t(window_capacity)
                            );
                            uint logical_block = slot
                                / uint(window_block_size);
                            uint row_in_block = slot
                                - logical_block * uint(window_block_size);
                            uint physical_block = uint(
                                window_block_table[logical_block]
                            );
                            physical_row = physical_block
                                * uint(window_block_size) + row_in_block;
                        } else {
                            physical_row = uint(
                                window_batch + size_t(first_window)
                                + size_t(global_candidate)
                            );
                        }
                    } else {
                        kind = 1u;
                        uint slot = global_candidate - visible_window;
                        uint row = (MTPLX_ROUTE == 1 || MTPLX_ROUTE == 3)
                            ? uint(compressed_indices[
                                selected_base + size_t(slot)
                            ])
                            : slot;
                        physical_row = row;
                        if (MTPLX_ROUTE == 1 || MTPLX_ROUTE == 2) {
                            uint logical_block =
                                row / uint(MTPLX_BLOCK_SIZE);
                            uint row_in_block = row
                                - logical_block * uint(MTPLX_BLOCK_SIZE);
                            uint physical_block = uint(
                                compressed_block_table[logical_block]
                            );
                            physical_row = physical_block
                                * uint(MTPLX_BLOCK_SIZE) + row_in_block;
                        } else {
                            physical_row += uint(compressed_batch);
                        }
                    }
                }
                tile_rows[candidate] = physical_row;
                tile_kinds[candidate] = kind;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // QK: four NAX SIMD groups split K=512 into four K=128 ranges.
            // Each group builds the native BF16 K operand directly from the
            // stock432 record, then stores one FP32 M16xN32 partial matrix.
            threadgroup bfloat* qk_tiles =
                reinterpret_cast<threadgroup bfloat*>(scratch + 16384);
            threadgroup float* qk_partials =
                reinterpret_cast<threadgroup float*>(scratch + 20480);
            if (simd_group < uint(MTPLX_PREFILL_QK_GROUPS)) {
                threadgroup bfloat* qk_values = qk_tiles
                    + simd_group * uint(MTPLX_PREFILL_TILE * 16);
                threadgroup float* qk_partial = qk_partials
                    + simd_group * uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE);

                tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> Q(
                    q_shared,
                    dextents<int, 2>{MTPLX_D, MTPLX_PREFILL_HEADS},
                    array<int, 2>{1, MTPLX_D}
                );
                tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> K(
                    qk_values,
                    dextents<int, 2>{MTPLX_PREFILL_TILE, 16},
                    array<int, 2>{1, MTPLX_PREFILL_TILE}
                );
                tensor<threadgroup float, dextents<int, 2>, tensor_inline> S(
                    qk_partial,
                    dextents<int, 2>{MTPLX_PREFILL_TILE, MTPLX_PREFILL_HEADS},
                    array<int, 2>{1, MTPLX_PREFILL_TILE}
                );
                auto qk_acc = qk_op.template get_destination_cooperative_tensor<
                    tensor<threadgroup bfloat,
                           extents<int, 16, 16>, tensor_inline>,
                    tensor<threadgroup bfloat,
                           extents<int, 32, 16>, tensor_inline>,
                    float
                >();
                for (uint16_t index = 0;
                     index < qk_acc.get_capacity(); ++index) {
                    qk_acc[index] = 0.0f;
                }

                uint k_begin = simd_group * 128u;
                for (uint k0 = k_begin; k0 < k_begin + 128u; k0 += 16u) {
                    uint candidate = lane;
                    const device uchar* record = tile_kinds[candidate] == 0u
                        ? window_records
                            + size_t(tile_rows[candidate]) * size_t(MTPLX_RECORD)
                        : compressed_records
                            + size_t(tile_rows[candidate]) * size_t(MTPLX_RECORD);
                    if (candidate < tile_count) {
                        if (k0 < 448u) {
                            float latent_scale = mtplx_dsv4_e4m3(
                                record[256u + (k0 >> 4)]
                            );
                            for (uint element = 0u; element < 16u; ++element) {
                                uint dim = k0 + element;
                                qk_values[element * uint(MTPLX_PREFILL_TILE)
                                          + candidate] = bfloat(
                                    mtplx_dsv4_latent(
                                        record, dim, latent_scale
                                    )
                                );
                            }
                        } else {
                            const device bfloat* rope = reinterpret_cast<
                                const device bfloat*>(record + 304u);
                            for (uint element = 0u; element < 16u; ++element) {
                                uint dim = k0 + element;
                                qk_values[element * uint(MTPLX_PREFILL_TILE)
                                          + candidate] = rope[dim - 448u];
                            }
                        }
                    } else {
                        for (uint element = 0u; element < 16u; ++element) {
                            qk_values[element * uint(MTPLX_PREFILL_TILE)
                                      + candidate] = bfloat(0.0f);
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    auto q_tile = Q.template slice<16, 16>(k0, 0);
                    auto k_tile = K.template slice<32, 16>(0, 0);
                    qk_op.run(q_tile, k_tile, qk_acc);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
                auto score_tile = S.template slice<32, 16>(0, 0);
                qk_acc.store(score_tile);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Sum the four K partials in FP32 and apply the attention scale
            // after QK, matching SparkInfer's S3 ordering.
            threadgroup float* tile_scores =
                reinterpret_cast<threadgroup float*>(scratch + 16384);
            for (uint offset = thread_index;
                 offset < uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE);
                 offset += uint(MTPLX_PREFILL_NAX_THREADS)) {
                float sum01 = qk_partials[offset]
                    + qk_partials[
                        uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE) + offset
                    ];
                float sum23 = qk_partials[
                        2u * uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE)
                        + offset
                    ]
                    + qk_partials[
                        3u * uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE)
                        + offset
                    ];
                tile_scores[offset] =
                    (sum01 + sum23) * float(scale) * MTPLX_LOG2E;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // S4/S5: one online-softmax state per head; probabilities are BF16
            // because the native stock432 P.V path is BF16 tensor math.
            threadgroup bfloat* probabilities =
                reinterpret_cast<threadgroup bfloat*>(scratch + 18432);
            if (thread_index < uint(MTPLX_PREFILL_HEADS)) {
                uint row = thread_index;
                uint row_base = row * uint(MTPLX_PREFILL_TILE);
                float old_max = running_max[row];
                float next_max = old_max;
                for (uint candidate = 0u; candidate < tile_count; ++candidate) {
                    next_max = max(next_max, tile_scores[row_base + candidate]);
                }
                float correction = fast::exp2(old_max - next_max);
                float next_sum = running_sum[row] * correction;
                for (uint candidate = 0u;
                     candidate < uint(MTPLX_PREFILL_TILE); ++candidate) {
                    float probability = candidate < tile_count
                        ? fast::exp2(
                            tile_scores[row_base + candidate] - next_max
                        )
                        : 0.0f;
                    probabilities[row_base + candidate] = bfloat(probability);
                    next_sum += probability;
                }
                row_correction[row] = correction;
                running_max[row] = next_max;
                running_sum[row] = next_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint matrix_row_base = ((lane & 7u) >> 1)
                + ((lane >> 4) * 4u);
            for (uint16_t index = 0;
                 index < pv_acc_lo.get_capacity(); ++index) {
                uint matrix_row = matrix_row_base
                    + (((uint(index) >> 2) & 1u) << 3);
                float correction = row_correction[matrix_row];
                pv_acc_lo[index] *= correction;
                pv_acc_hi[index] *= correction;
            }

            // S6: eight groups cover 64 V dimensions each.  Each group reuses
            // one M16xN32xK16 B tile for its low/high 32-dimension fragments.
            threadgroup bfloat* value_tiles =
                reinterpret_cast<threadgroup bfloat*>(scratch + 19456);
            threadgroup bfloat* value_base = value_tiles
                + simd_group * 16u * 32u;
            tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> P(
                probabilities,
                dextents<int, 2>{MTPLX_PREFILL_TILE, MTPLX_PREFILL_HEADS},
                array<int, 2>{1, MTPLX_PREFILL_TILE}
            );
            tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> V(
                value_base,
                dextents<int, 2>{32, 16},
                array<int, 2>{1, 32}
            );

            for (uint candidate_base = 0u;
                 candidate_base < uint(MTPLX_PREFILL_TILE);
                 candidate_base += 16u) {
                auto probability_tile = P.template slice<16, 16>(
                    candidate_base, 0
                );
                for (uint output_half = 0u; output_half < 2u; ++output_half) {
                    uint candidate = lane >> 1;
                    uint half_group = lane & 1u;
                    uint dim_base = simd_group * 64u + output_half * 32u
                        + half_group * 16u;
                    uint tile_candidate = candidate_base + candidate;
                    const device uchar* record = tile_kinds[tile_candidate] == 0u
                        ? window_records
                            + size_t(tile_rows[tile_candidate])
                                * size_t(MTPLX_RECORD)
                        : compressed_records
                            + size_t(tile_rows[tile_candidate])
                                * size_t(MTPLX_RECORD);
                    if (tile_candidate < tile_count) {
                        float latent_scale = mtplx_dsv4_e4m3(
                            record[256u + (dim_base >> 4)]
                        );
                        for (uint element = 0u; element < 16u; ++element) {
                            value_base[candidate * 32u
                                       + half_group * 16u + element] =
                                mtplx_dsv4_device_value_bf16(
                                    record, dim_base + element, latent_scale
                                );
                        }
                    } else {
                        for (uint element = 0u; element < 16u; ++element) {
                            value_base[candidate * 32u
                                       + half_group * 16u + element] =
                                bfloat(0.0f);
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    auto value_tile = V.template slice<32, 16>(0, 0);
                    if (output_half == 0u) {
                        pv_op.run(
                            probability_tile, value_tile, pv_acc_lo
                        );
                    } else {
                        pv_op.run(
                            probability_tile, value_tile, pv_acc_hi
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint matrix_row_base = ((lane & 7u) >> 1)
            + ((lane >> 4) * 4u);
        uint matrix_col_base = ((lane & 1u) << 2)
            + (((lane >> 3) & 1u) << 3);
        for (uint16_t index = 0;
             index < pv_acc_lo.get_capacity(); ++index) {
            uint matrix_row = matrix_row_base
                + (((uint(index) >> 2) & 1u) << 3);
            uint matrix_col = matrix_col_base + (uint(index) & 3u)
                + (uint(index) >> 3) * 16u;
            uint output_head = head_base + matrix_row;
            uint output_dim_lo = simd_group * 64u + matrix_col;
            uint output_dim_hi = output_dim_lo + 32u;
            size_t output_base = MTPLX_QUERY_TOKEN_MAJOR
                ? (
                    (size_t(batch * query_count + query_row)
                     * size_t(MTPLX_H) + size_t(output_head))
                    * size_t(MTPLX_D)
                )
                : (
                    (size_t(batch * uint(MTPLX_H) + output_head)
                     * size_t(query_count) + size_t(query_row))
                    * size_t(MTPLX_D)
                );
            float inverse_sum = 1.0f / running_sum[matrix_row];
            out[output_base + size_t(output_dim_lo)] =
                T(pv_acc_lo[index] * inverse_sum);
            out[output_base + size_t(output_dim_hi)] =
                T(pv_acc_hi[index] * inverse_sum);
        }
    """
    window_layout = "paged_window" if window_paged else "contiguous_window"
    query_layout = "token_major" if query_token_major else "bhmd"
    return mx.fast.metal_kernel(
        name=(
            "mtplx_dsv4_mia_stock432_prefill_nax_mg16_"
            f"{route}_sw{int(selected_width)}_bs{int(block_size)}_"
            f"{window_layout}_{query_layout}_v3"
        ),
        input_names=[
            "queries",
            "window_records",
            "window_block_table",
            "window_start",
            "window_capacity",
            "window_block_size",
            "query_positions",
            "compressed_records",
            "compressed_block_table",
            "compressed_indices",
            "compressed_lengths",
            "sinks",
            "scale",
            "n_queries",
            "n_window_records",
            "n_compressed_records",
        ],
        output_names=["out"],
        header=(
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            + _PREFILL_HEADER
            + f"\nconstant constexpr int MTPLX_ROUTE = {_ROUTE_IDS[route]};\n"
            + f"constant constexpr int MTPLX_SELECTED_WIDTH = {int(selected_width)};\n"
            + f"constant constexpr int MTPLX_BLOCK_SIZE = {int(block_size)};\n"
            + "constant constexpr bool MTPLX_WINDOW_PAGED = "
            + ("true;\n" if window_paged else "false;\n")
            + "constant constexpr bool MTPLX_QUERY_TOKEN_MAJOR = "
            + ("true;\n" if query_token_major else "false;\n")
            + "constant constexpr float MTPLX_LOG2E = 1.4426950408889634f;\n"
        ),
        source=source,
        ensure_row_contiguous=True,
    )


def _launch_nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_block_table: mx.array,
    window_start: int,
    window_capacity: int,
    window_block_size: int,
    window_count: int,
    query_positions: mx.array,
    compressed_records: mx.array,
    compressed_block_table: mx.array,
    compressed_count: int,
    compressed_indices: mx.array,
    compressed_lengths: mx.array,
    sinks: mx.array,
    scale: float,
    *,
    query_count: int,
    output_shape: tuple[int, int, int, int],
    kernel,
) -> mx.array:
    batch = int(queries.shape[0])
    (output,) = kernel(
        inputs=[
            queries,
            window_records,
            window_block_table,
            int(window_start),
            int(window_capacity),
            int(window_block_size),
            query_positions,
            compressed_records,
            compressed_block_table,
            compressed_indices,
            compressed_lengths,
            sinks,
            float(scale),
            int(query_count),
            int(window_count),
            int(compressed_count),
        ],
        template=[("T", mx.bfloat16)],
        grid=(
            batch
            * (_HEADS // _DECODE_HEADS_PER_GROUP)
            * query_count
            * _DECODE_NAX_THREADS,
            1,
            1,
        ),
        threadgroup=(_DECODE_NAX_THREADS, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.bfloat16],
    )
    return output


def _run_nvfp4_sparse_mla_storage(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_block_table: mx.array,
    compressed_count: int,
    compressed_block_size: int,
    use_paged_compressed: int,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
    route: str | None = None,
) -> mx.array:
    batch, _heads, query_count, _width = (int(value) for value in queries.shape)
    window_count = int(window_records.shape[1])
    # MLX assigns zero-sized array inputs to Metal's constant address space.
    # Keep the logical counts at zero, but pass one real device-backed record so
    # the fixed kernel signature is identical for every layer and phase.
    if window_count == 0:
        window_records = workspace.dummy_record
    if compressed_records is None:
        compressed_records = window_records[:, :1]
    if compressed_indices is None:
        compressed_indices = workspace.indices(query_count)
        selected_width = 1
        use_indices = 0
    else:
        selected_width = int(compressed_indices.shape[2])
        use_indices = 1
    if compressed_lengths is None:
        compressed_lengths = workspace.lengths(query_count)

    if route is None:
        if compressed_count == 0:
            route = _ROUTE_WINDOW
        elif use_indices:
            route = (
                _ROUTE_INDEXED_PAGED
                if use_paged_compressed
                else _ROUTE_INDEXED_CONTIGUOUS
            )
        else:
            route = (
                _ROUTE_SEQUENTIAL_PAGED
                if use_paged_compressed
                else _ROUTE_SEQUENTIAL_CONTIGUOUS
            )

    kernel = _kernel(route, selected_width, compressed_block_size)
    return _launch_nvfp4_sparse_mla(
        queries,
        window_records,
        workspace.dummy_block_table,
        window_start,
        max(1, window_count),
        1,
        window_count,
        query_positions,
        compressed_records,
        compressed_block_table,
        compressed_count,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        query_count=query_count,
        output_shape=(batch, _HEADS, query_count, _HEAD_DIM),
        kernel=kernel,
    )


def _launch_nvfp4_prefill_mla(
    queries: mx.array,
    window_records: mx.array,
    window_block_table: mx.array,
    window_start: int,
    window_capacity: int,
    window_block_size: int,
    window_count: int,
    query_positions: mx.array,
    compressed_records: mx.array,
    compressed_block_table: mx.array,
    compressed_count: int,
    compressed_indices: mx.array,
    compressed_lengths: mx.array,
    sinks: mx.array,
    scale: float,
    *,
    query_count: int,
    output_shape: tuple[int, int, int, int],
    kernel,
) -> mx.array:
    batch = int(queries.shape[0])
    (output,) = kernel(
        inputs=[
            queries,
            window_records,
            window_block_table,
            int(window_start),
            int(window_capacity),
            int(window_block_size),
            query_positions,
            compressed_records,
            compressed_block_table,
            compressed_indices,
            compressed_lengths,
            sinks,
            float(scale),
            int(query_count),
            int(window_count),
            int(compressed_count),
        ],
        template=[("T", mx.bfloat16)],
        grid=(
            batch
            * (_HEADS // _PREFILL_HEADS_PER_GROUP)
            * query_count
            * _PREFILL_NAX_THREADS,
            1,
            1,
        ),
        threadgroup=(_PREFILL_NAX_THREADS, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.bfloat16],
    )
    return output


def _run_nvfp4_prefill_mla_storage(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_block_table: mx.array,
    compressed_count: int,
    compressed_block_size: int,
    use_paged_compressed: int,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
    route: str | None = None,
) -> mx.array:
    batch, _heads, query_count, _width = (int(value) for value in queries.shape)
    window_count = int(window_records.shape[1])
    if window_count == 0:
        window_records = workspace.dummy_record
    if compressed_records is None:
        compressed_records = window_records[:, :1]
    if compressed_indices is None:
        compressed_indices = workspace.indices(query_count)
        selected_width = 1
        use_indices = 0
    else:
        selected_width = int(compressed_indices.shape[2])
        use_indices = 1
    if compressed_lengths is None:
        compressed_lengths = workspace.lengths(query_count)

    if route is None:
        if compressed_count == 0:
            route = _ROUTE_WINDOW
        elif use_indices:
            route = (
                _ROUTE_INDEXED_PAGED
                if use_paged_compressed
                else _ROUTE_INDEXED_CONTIGUOUS
            )
        else:
            route = (
                _ROUTE_SEQUENTIAL_PAGED
                if use_paged_compressed
                else _ROUTE_SEQUENTIAL_CONTIGUOUS
            )

    kernel = _prefill_nax_mg16_kernel(
        route,
        selected_width,
        compressed_block_size,
    )
    return _launch_nvfp4_prefill_mla(
        queries,
        window_records,
        workspace.dummy_block_table,
        window_start,
        max(1, window_count),
        1,
        window_count,
        query_positions,
        compressed_records,
        compressed_block_table,
        compressed_count,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        query_count=query_count,
        output_shape=(batch, _HEADS, query_count, _HEAD_DIM),
        kernel=kernel,
    )


def _run_nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    compressed_count = (
        0 if compressed_records is None else int(compressed_records.shape[1])
    )
    return _run_nvfp4_sparse_mla_storage(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        workspace.dummy_block_table,
        compressed_count,
        1,
        0,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=workspace,
    )


def _run_nvfp4_prefill_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    compressed_count = (
        0 if compressed_records is None else int(compressed_records.shape[1])
    )
    return _run_nvfp4_prefill_mla_storage(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        workspace.dummy_block_table,
        compressed_count,
        1,
        0,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=workspace,
    )


def _run_paged_nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: PagedMiaNVFP4Records | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    if compressed_records is None:
        pages = None
        block_table = workspace.dummy_block_table
        compressed_count = 0
        block_size = 1
    else:
        pages = compressed_records.records
        block_table = compressed_records.block_table
        compressed_count = int(compressed_records.length)
        block_size = int(compressed_records.block_size)
    return _run_nvfp4_sparse_mla_storage(
        queries,
        window_records,
        window_start,
        query_positions,
        pages,
        block_table,
        compressed_count,
        block_size,
        1,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=workspace,
    )


def _run_installed_window_nvfp4_sparse_mla(
    queries: mx.array,
    window_records: FixedMiaNVFP4WindowRecords,
    window_start: int,
    query_positions: mx.array,
    compressed_records,
    compressed_indices,
    compressed_lengths,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
    kernel,
) -> mx.array:
    del compressed_records, compressed_indices, compressed_lengths
    batch = int(queries.shape[0])
    query_count = int(queries.shape[1])
    return _launch_nvfp4_sparse_mla(
        queries,
        window_records.pages,
        window_records.block_table,
        window_start,
        window_records.capacity,
        window_records.block_size,
        window_records.length,
        query_positions,
        workspace.dummy_record,
        workspace.dummy_block_table,
        0,
        workspace.indices(query_count),
        workspace.lengths(query_count),
        sinks,
        scale,
        query_count=query_count,
        output_shape=(batch, query_count, _HEADS, _HEAD_DIM),
        kernel=kernel,
    )


def _run_installed_indexed_paged_nvfp4_sparse_mla(
    queries: mx.array,
    window_records: FixedMiaNVFP4WindowRecords,
    window_start: int,
    query_positions: mx.array,
    compressed_records: PagedMiaNVFP4Records,
    compressed_indices: mx.array,
    compressed_lengths: mx.array,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
    kernel,
) -> mx.array:
    del workspace
    batch = int(queries.shape[0])
    query_count = int(queries.shape[1])
    return _launch_nvfp4_sparse_mla(
        queries,
        window_records.pages,
        window_records.block_table,
        window_start,
        window_records.capacity,
        window_records.block_size,
        window_records.length,
        query_positions,
        compressed_records.records,
        compressed_records.block_table,
        int(compressed_records.length),
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        query_count=query_count,
        output_shape=(batch, query_count, _HEADS, _HEAD_DIM),
        kernel=kernel,
    )


def _run_installed_sequential_paged_nvfp4_sparse_mla(
    queries: mx.array,
    window_records: FixedMiaNVFP4WindowRecords,
    window_start: int,
    query_positions: mx.array,
    compressed_records: PagedMiaNVFP4Records,
    compressed_indices,
    compressed_lengths: mx.array,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
    kernel,
) -> mx.array:
    del compressed_indices
    batch = int(queries.shape[0])
    query_count = int(queries.shape[1])
    return _launch_nvfp4_sparse_mla(
        queries,
        window_records.pages,
        window_records.block_table,
        window_start,
        window_records.capacity,
        window_records.block_size,
        window_records.length,
        query_positions,
        compressed_records.records,
        compressed_records.block_table,
        int(compressed_records.length),
        workspace.indices(query_count),
        compressed_lengths,
        sinks,
        scale,
        query_count=query_count,
        output_shape=(batch, query_count, _HEADS, _HEAD_DIM),
        kernel=kernel,
    )


def _run_installed_window_nvfp4_prefill_mla(
    queries,
    window_records: FixedMiaNVFP4WindowRecords,
    window_start,
    query_positions,
    compressed_records,
    compressed_indices,
    compressed_lengths,
    sinks,
    scale,
    *,
    workspace,
    kernel,
):
    del compressed_records, compressed_indices, compressed_lengths
    batch = int(queries.shape[0])
    query_count = int(queries.shape[1])
    return _launch_nvfp4_prefill_mla(
        queries,
        window_records.pages,
        window_records.block_table,
        window_start,
        window_records.capacity,
        window_records.block_size,
        window_records.length,
        query_positions,
        workspace.dummy_record,
        workspace.dummy_block_table,
        0,
        workspace.indices(query_count),
        workspace.lengths(query_count),
        sinks,
        scale,
        query_count=query_count,
        output_shape=(batch, query_count, _HEADS, _HEAD_DIM),
        kernel=kernel,
    )


def _run_installed_indexed_paged_nvfp4_prefill_mla(
    queries,
    window_records: FixedMiaNVFP4WindowRecords,
    window_start,
    query_positions,
    compressed_records,
    compressed_indices,
    compressed_lengths,
    sinks,
    scale,
    *,
    workspace,
    kernel,
):
    del workspace
    batch = int(queries.shape[0])
    query_count = int(queries.shape[1])
    return _launch_nvfp4_prefill_mla(
        queries,
        window_records.pages,
        window_records.block_table,
        window_start,
        window_records.capacity,
        window_records.block_size,
        window_records.length,
        query_positions,
        compressed_records.records,
        compressed_records.block_table,
        int(compressed_records.length),
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        query_count=query_count,
        output_shape=(batch, query_count, _HEADS, _HEAD_DIM),
        kernel=kernel,
    )


def _run_installed_sequential_paged_nvfp4_prefill_mla(
    queries,
    window_records: FixedMiaNVFP4WindowRecords,
    window_start,
    query_positions,
    compressed_records,
    compressed_indices,
    compressed_lengths,
    sinks,
    scale,
    *,
    workspace,
    kernel,
):
    del compressed_indices
    batch = int(queries.shape[0])
    query_count = int(queries.shape[1])
    return _launch_nvfp4_prefill_mla(
        queries,
        window_records.pages,
        window_records.block_table,
        window_start,
        window_records.capacity,
        window_records.block_size,
        window_records.length,
        query_positions,
        compressed_records.records,
        compressed_records.block_table,
        int(compressed_records.length),
        workspace.indices(query_count),
        compressed_lengths,
        sinks,
        scale,
        query_count=query_count,
        output_shape=(batch, query_count, _HEADS, _HEAD_DIM),
        kernel=kernel,
    )


def _run_paged_nvfp4_prefill_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: PagedMiaNVFP4Records | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    if compressed_records is None:
        pages = None
        block_table = workspace.dummy_block_table
        compressed_count = 0
        block_size = 1
    else:
        pages = compressed_records.records
        block_table = compressed_records.block_table
        compressed_count = int(compressed_records.length)
        block_size = int(compressed_records.block_size)
    return _run_nvfp4_prefill_mla_storage(
        queries,
        window_records,
        window_start,
        query_positions,
        pages,
        block_table,
        compressed_count,
        block_size,
        1,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=workspace,
    )


def nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    """Validate and run the fixed Mia sparse-MLA contract.

    Exact-model execution uses :func:`install_nvfp4_sparse_mla` once and calls
    its returned function directly; this checked entry point is the codec oracle
    and construction boundary.
    """
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 sparse MLA requires Metal")
    if tuple(queries.shape[:2]) != (1, _HEADS) or int(queries.shape[-1]) != _HEAD_DIM:
        raise ValueError("Mia sparse MLA queries must have shape [1, 64, rows, 512]")
    if queries.dtype != mx.bfloat16:
        raise ValueError("Mia sparse MLA queries must be BF16")
    if (
        window_records.dtype != mx.uint8
        or tuple(window_records.shape[:1]) != (1,)
        or int(window_records.shape[-1]) != _RECORD_BYTES
    ):
        raise ValueError("Mia sparse MLA window records must be [1, rows, 432] uint8")
    query_count = int(queries.shape[2])
    if tuple(query_positions.shape) != (query_count,):
        raise ValueError("Mia sparse MLA query positions must match query rows")
    if tuple(sinks.shape) != (_HEADS,):
        raise ValueError("Mia sparse MLA sinks must have shape [64]")
    paged_compressed = isinstance(compressed_records, PagedMiaNVFP4Records)
    if paged_compressed:
        if (
            compressed_records.records.dtype != mx.uint8
            or int(compressed_records.records.shape[-1]) != _RECORD_BYTES
            or int(compressed_records.block_size) <= 0
        ):
            raise ValueError("invalid paged Mia sparse MLA compressed records")
    elif compressed_records is not None and (
        compressed_records.dtype != mx.uint8
        or tuple(compressed_records.shape[:1]) != (1,)
        or int(compressed_records.shape[-1]) != _RECORD_BYTES
    ):
        raise ValueError(
            "Mia sparse MLA compressed records must be [1, rows, 432] uint8"
        )
    if compressed_indices is not None and tuple(compressed_indices.shape[:2]) != (
        1,
        query_count,
    ):
        raise ValueError("Mia sparse MLA selected indices must be [1, queries, K]")
    if compressed_lengths is not None and tuple(compressed_lengths.shape) != (
        1,
        query_count,
    ):
        raise ValueError("Mia sparse MLA compressed lengths must be [1, queries]")
    runner = _run_paged_nvfp4_sparse_mla if paged_compressed else _run_nvfp4_sparse_mla
    return runner(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=mia_mla_workspace(),
    )


def nvfp4_prefill_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    """Validated oracle boundary for the measured large-M head-group route."""
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 sparse prefill MLA requires Metal")
    if tuple(queries.shape[:2]) != (1, _HEADS) or int(queries.shape[-1]) != _HEAD_DIM:
        raise ValueError("Mia sparse prefill queries must have shape [1, 64, rows, 512]")
    if queries.dtype != mx.bfloat16:
        raise ValueError("Mia sparse prefill queries must be BF16")
    if (
        window_records.dtype != mx.uint8
        or tuple(window_records.shape[:1]) != (1,)
        or int(window_records.shape[-1]) != _RECORD_BYTES
    ):
        raise ValueError(
            "Mia sparse prefill window records must be [1, rows, 432] uint8"
        )
    query_count = int(queries.shape[2])
    if tuple(query_positions.shape) != (query_count,):
        raise ValueError("Mia sparse prefill positions must match query rows")
    if tuple(sinks.shape) != (_HEADS,):
        raise ValueError("Mia sparse prefill sinks must have shape [64]")
    paged_compressed = isinstance(compressed_records, PagedMiaNVFP4Records)
    if paged_compressed:
        if (
            compressed_records.records.dtype != mx.uint8
            or int(compressed_records.records.shape[-1]) != _RECORD_BYTES
            or int(compressed_records.block_size) <= 0
        ):
            raise ValueError("invalid paged Mia sparse prefill records")
    elif compressed_records is not None and (
        compressed_records.dtype != mx.uint8
        or tuple(compressed_records.shape[:1]) != (1,)
        or int(compressed_records.shape[-1]) != _RECORD_BYTES
    ):
        raise ValueError(
            "Mia sparse prefill records must be [1, rows, 432] uint8"
        )
    if compressed_indices is not None and tuple(compressed_indices.shape[:2]) != (
        1,
        query_count,
    ):
        raise ValueError("Mia sparse prefill indices must be [1, queries, K]")
    if compressed_lengths is not None and tuple(compressed_lengths.shape) != (
        1,
        query_count,
    ):
        raise ValueError("Mia sparse prefill lengths must be [1, queries]")
    runner = (
        _run_paged_nvfp4_prefill_mla
        if paged_compressed
        else _run_nvfp4_prefill_mla
    )
    return runner(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=mia_mla_workspace(),
    )


def install_nvfp4_sparse_mla(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    window_size: int,
    compress_ratio: int,
    workspace: MiaMLAWorkspace | None = None,
):
    """Validate Mia's fixed geometry once and return the direct hot callable."""
    observed = (
        int(heads),
        int(head_dim),
        int(rope_dim),
        int(window_size),
        int(compress_ratio),
    )
    if observed[:4] != (_HEADS, _HEAD_DIM, 64, _WINDOW) or observed[4] not in (
        0,
        4,
        128,
    ):
        raise ValueError(
            f"unsupported Mia stock432 sparse MLA geometry: {observed!r}"
        )
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 sparse MLA installation requires Metal")
    route, runner = {
        0: (_ROUTE_WINDOW, _run_installed_window_nvfp4_sparse_mla),
        4: (
            _ROUTE_INDEXED_PAGED,
            _run_installed_indexed_paged_nvfp4_sparse_mla,
        ),
        128: (
            _ROUTE_SEQUENTIAL_PAGED,
            _run_installed_sequential_paged_nvfp4_sparse_mla,
        ),
    }[int(compress_ratio)]
    selected_width, block_size = {
        0: (1, 1),
        4: (512, 64),
        128: (1, 2),
    }[int(compress_ratio)]
    kernel = _kernel(
        route,
        selected_width,
        block_size,
        window_paged=True,
        query_token_major=True,
    )
    return partial(
        runner,
        workspace=workspace or mia_mla_workspace(),
        kernel=kernel,
    )


def install_dspark_k5_nvfp4_mla(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    window_size: int,
    block_size: int,
):
    """Install the packaged fixed-K5, 128-row stock432 DSpark attention."""

    observed = (
        int(heads),
        int(head_dim),
        int(rope_dim),
        int(window_size),
        int(block_size),
    )
    expected = (_HEADS, _HEAD_DIM, 64, _WINDOW, _DSPARK_ROWS)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia stock432 DSpark geometry: {observed!r} != {expected!r}"
        )
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 DSpark K5 installation requires Metal")
    kernel = _kernel(
        _ROUTE_DSPARK,
        1,
        1,
        query_token_major=True,
    )
    query_positions, block_table, indices, lengths = _dspark_placeholder_inputs()
    return partial(
        _run_dspark_k5_nvfp4_mla,
        kernel=kernel,
        query_positions=query_positions,
        block_table=block_table,
        indices=indices,
        lengths=lengths,
    )


def install_dspark_k5_nvfp4_mla_graph(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    window_size: int,
    block_size: int,
):
    """Install the graph-safe form of the fixed-K5 DSpark attention launch."""

    observed = (
        int(heads),
        int(head_dim),
        int(rope_dim),
        int(window_size),
        int(block_size),
    )
    expected = (_HEADS, _HEAD_DIM, 64, _WINDOW, _DSPARK_ROWS)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia stock432 DSpark geometry: {observed!r} != {expected!r}"
        )
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 DSpark K5 installation requires Metal")
    kernel = _kernel(
        _ROUTE_DSPARK,
        1,
        1,
        query_token_major=True,
    )
    query_positions, block_table, indices, lengths = _dspark_placeholder_inputs()
    return partial(
        _run_dspark_k5_nvfp4_mla_graph,
        kernel=kernel,
        query_positions=query_positions,
        block_table=block_table,
        indices=indices,
        lengths=lengths,
    )


def install_nvfp4_prefill_mla(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    window_size: int,
    compress_ratio: int,
    workspace: MiaMLAWorkspace | None = None,
):
    """Install Mia's M5 NAX prefill engine for the fixed stock432 geometry."""
    observed = (
        int(heads), int(head_dim), int(rope_dim), int(window_size), int(compress_ratio)
    )
    if observed[:4] != (_HEADS, _HEAD_DIM, 64, _WINDOW) or observed[4] not in (
        0,
        4,
        128,
    ):
        raise ValueError(
            f"unsupported Mia stock432 prefill geometry: {observed!r}"
        )
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 prefill installation requires Metal")
    from mtplx.nax_verify import nax_available

    if not nax_available():
        raise RuntimeError(
            "Mia stock432 prefill requires Apple G17 NAX on macOS 26.2 or newer"
        )
    route, runner = {
        0: (_ROUTE_WINDOW, _run_installed_window_nvfp4_prefill_mla),
        4: (
            _ROUTE_INDEXED_PAGED,
            _run_installed_indexed_paged_nvfp4_prefill_mla,
        ),
        128: (
            _ROUTE_SEQUENTIAL_PAGED,
            _run_installed_sequential_paged_nvfp4_prefill_mla,
        ),
    }[int(compress_ratio)]
    selected_width, block_size = {
        0: (1, 1),
        4: (512, 64),
        128: (1, 2),
    }[int(compress_ratio)]
    kernel = _prefill_nax_mg16_kernel(
        route,
        selected_width,
        block_size,
        window_paged=True,
        query_token_major=True,
    )
    return partial(
        runner,
        workspace=workspace or mia_mla_workspace(),
        kernel=kernel,
    )
