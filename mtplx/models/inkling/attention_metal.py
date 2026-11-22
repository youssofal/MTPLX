"""Inkling's chronological attention over grouped KV, with one sink per query."""

from functools import lru_cache

import mlx.core as mx
from . import launch_limits


@lru_cache(maxsize=1)
def _prompt_kernel():
    return launch_limits.kernel(
        name="mtplx_inkling_streamed_attention",
        input_names=["q", "k", "v", "profiles", "sinks"],
        output_names=["out"],
        ensure_row_contiguous=False,
        header="#include <metal_simdgroup_matrix>\n",
        source="""
            // Eight SIMD groups share a 16-query by 32-key score tile. The
            // 128-wide queries and weighted sums live in matrix registers.
            using Block = simdgroup_matrix<float, 8, 8>;
            constexpr int queries = 16, columns = 32, width = 128, threads = 256;
            const int tid = thread_position_in_threadgroup.x;
            const int group = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
            const int row0 = threadgroup_position_in_grid.x * queries;
            const int head = threadgroup_position_in_grid.y, batch = threadgroup_position_in_grid.z;
            const int heads = q_shape[1], rows = q_shape[2], keys = k_shape[2];
            const int kvhead = head / (heads / k_shape[1]);
            const int first = keys - rows + row0;
            const int live = min(queries, rows - row0);
            const int stop = first + live;
            const int begin = WINDOW ? max(0, first - WINDOW + 1) : 0;
            const int extent = profiles_shape[3];
            const int row_panel = (group / 4) * 8, col_panel = (group % 4) * 8;
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1];
            const device T* kp = k + batch * k_strides[0] + kvhead * k_strides[1];
            const device T* vp = v + batch * v_strides[0] + kvhead * v_strides[1];
            const device T* rp = profiles + batch * profiles_strides[0] + head * profiles_strides[2];
            const device T* sp = sinks + batch * sinks_strides[0] + head * sinks_strides[2];

            threadgroup float staging[columns * width];
            threadgroup float scores[queries * columns];
            threadgroup float row_factor[queries * 8];
            Block query_blocks[width / 8], weighted[4];

            for (int i = tid; i < queries * width; i += threads) {
                int r = i / width, d = i % width;
                staging[i] = r < live ? float(qp[(row0 + r) * q_strides[2] + d * q_strides[3]]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int d = 0; d < width / 8; ++d)
                simdgroup_load(query_blocks[d], staging + row_panel * width + 8 * d, width);
            for (int panel = 0; panel < 4; ++panel)
                weighted[panel] = Block(0.0f);

            // Each SIMD group also owns the softmax of two entire rows. The
            // sink contributes mass but no value, including in the first tile.
            float peak[2], mass[2];
            for (int half_row = 0; half_row < 2; ++half_row) {
                int r = group + 8 * half_row;
                peak[half_row] = r < live ? float(sp[(row0 + r) * sinks_strides[1]]) : 0.0f;
                mass[half_row] = peak[half_row] == -INFINITY ? 0.0f : 1.0f;
            }

            for (int col0 = begin; col0 < stop; col0 += columns) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = tid; i < columns * width; i += threads) {
                    int col = col0 + i / width, d = i % width;
                    staging[i] = col < stop ? float(kp[col * k_strides[2] + d * k_strides[3]]) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                Block dot = Block(0.0f);
                for (int d = 0; d < width / 8; ++d) {
                    Block keys_t;
                    simdgroup_load(keys_t, staging + col_panel * width + 8 * d, width, ulong2(0), true);
                    simdgroup_multiply_accumulate(dot, query_blocks[d], keys_t, dot);
                }
                simdgroup_store(dot, scores + row_panel * columns + col_panel, columns);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int half_row = 0; half_row < 2; ++half_row) {
                    int r = group + 8 * half_row, col = col0 + lane;
                    int distance = first + r - col;
                    float score = -INFINITY;
                    if (r < live && distance >= 0 && (!WINDOW || distance < WINDOW)) {
                        float bias = distance < extent
                            ? float(rp[(row0 + r) * profiles_strides[1] + distance * profiles_strides[3]])
                            : 0.0f;
                        score = scores[r * columns + lane] * rsqrt(float(width)) + bias;
                    }
                    float next_peak = max(peak[half_row], simd_max(score));
                    float carry = next_peak == peak[half_row] ? 1.0f : exp(peak[half_row] - next_peak);
                    float weight = score == -INFINITY ? 0.0f : exp(score - next_peak);
                    mass[half_row] = mass[half_row] * carry + simd_sum(weight);
                    peak[half_row] = next_peak;
                    scores[r * columns + lane] = weight;
                    if (lane < 8) row_factor[r * 8 + lane] = carry;
                }
                for (int i = tid; i < columns * width; i += threads) {
                    int col = col0 + i / width, d = i % width;
                    staging[i] = col < stop ? float(vp[col * v_strides[2] + d * v_strides[3]]) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Loading the row factor as a matrix gives it exactly the
                // accumulator's lane layout, without assuming a fragment layout.
                Block factor;
                simdgroup_load(factor, row_factor + row_panel * 8, 8);
                for (int panel = 0; panel < 4; ++panel)
                    for (int e = 0; e < 2; ++e)
                        weighted[panel].thread_elements()[e] *= factor.thread_elements()[e];
                for (int offset = 0; offset < columns; offset += 8) {
                    Block prob;
                    simdgroup_load(prob, scores + row_panel * columns + offset, columns);
                    for (int panel = 0; panel < 4; ++panel) {
                        Block values;
                        simdgroup_load(values, staging + offset * width + col_panel + panel * columns, width);
                        simdgroup_multiply_accumulate(weighted[panel], prob, values, weighted[panel]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 8) {
                row_factor[group * 8 + lane] = 1.0f / mass[0];
                row_factor[(group + 8) * 8 + lane] = 1.0f / mass[1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            Block inv_mass;
            simdgroup_load(inv_mass, row_factor + row_panel * 8, 8);
            for (int panel = 0; panel < 4; ++panel) {
                for (int e = 0; e < 2; ++e)
                    weighted[panel].thread_elements()[e] *= inv_mass.thread_elements()[e];
                simdgroup_store(weighted[panel], staging + row_panel * width + col_panel + panel * columns, width);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = tid; i < live * width; i += threads) {
                size_t row = size_t(batch) * rows + row0 + i / width;
                out[(row * heads + head) * width + i % width] = T(staging[i]);
            }
        """,
    )


def streamed_attention(query, key, value, profiles, sink, window):
    batch, heads, rows, width = query.shape
    result = _prompt_kernel()(
        inputs=[query, key, value, profiles, sink],
        template=[("T", query.dtype), ("WINDOW", window)],
        grid=(((rows + 15) // 16) * 256, heads, batch),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, rows, heads, width)],
        output_dtypes=[query.dtype],
    )[0]
    # The output projection consumes rows then heads. This is a view in both directions.
    return result.transpose(0, 2, 1, 3)


@lru_cache(maxsize=1)
def _grouped_mask_kernel():
    return launch_limits.kernel(
        name="mtplx_inkling_grouped_sink_mask",
        input_names=["profiles", "keys"],
        output_names=["mask"],
        ensure_row_contiguous=False,
        source="""
            const int col = thread_position_in_grid.x, row = thread_position_in_grid.y;
            const int head = thread_position_in_grid.z, rows = profiles_shape[1];
            const int count = keys_shape[2], extent = profiles_shape[3];
            if (col >= count) return;
            const int distance = count - rows + row - col;
            const bool visible = distance >= 0 && (!WINDOW || distance < WINDOW);
            T bias = T(0);
            if (!visible) bias = T(-INFINITY);
            else if (distance < extent)
                bias = profiles[row * profiles_strides[1] + head * profiles_strides[2] + distance * profiles_strides[3]];
            mask[(head * rows + row) * size_t(count) + col] = bias;
        """,
    )


def grouped_sink_attention(query, key, value, profiles, sink, window):
    """One native SDPA sink per (head,row), sharing the original KV heads."""
    _, heads, rows, width = query.shape
    if window and key.shape[2] > window + rows - 1:
        start = key.shape[2] - (window + rows - 1)
        key, value = key[:, :, start:], value[:, :, start:]
    count = key.shape[2]
    shape = (1, heads * rows, 1, count)
    mask = _grouped_mask_kernel()(
        inputs=[profiles, key],
        template=[("T", query.dtype), ("WINDOW", window)],
        grid=(count, rows, heads),
        threadgroup=(min(128, count), 1, 1),
        output_shapes=[shape],
        output_dtypes=[query.dtype],
    )[0]
    # Flatten head first so integer division by the expanded GQA group still
    # maps every query row to the original KV head. Each sink stays independent.
    result = mx.fast.scaled_dot_product_attention(
        query.reshape(1, heads * rows, 1, width),
        key, value, scale=width**-0.5, mask=mask, sinks=sink[0].T.reshape(-1),
    )
    return result.reshape(query.shape)
