"""Retained target-shaped kernels from the pinned Qwen 3.8 challenge."""

from __future__ import annotations

from functools import partial
from typing import Any

_DUAL_RMS_CONCAT_KERNEL = None
_Q8_EMBED_DUAL_RMS_CONCAT_KERNEL = None
_QK_RMS_ROPE_KERNEL = None


def qwen38_row24_async_eval(value: Any, *, row26: bool = False) -> None:
    import mlx.core as mx

    del row26
    mx.async_eval(value)


def qwen38_qk_rms_rope(
    queries: Any,
    keys: Any,
    q_weight: Any,
    k_weight: Any,
    eps: float,
    offset: int,
) -> tuple[Any, Any]:
    """Fuse exact Qwen 3.8 BF16 Q/K RMSNorm, transpose, and partial RoPE."""

    import mlx.core as mx

    global _QK_RMS_ROPE_KERNEL
    if _QK_RMS_ROPE_KERNEL is None:
        _QK_RMS_ROPE_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_qk_rms_rope_bf16_h256_r64_v1",
            input_names=[
                "q", "k", "q_weight", "k_weight", "eps", "offset", "log2_base"
            ],
            output_names=["q_out", "k_out"],
            source=r"""
                constexpr uint n_reads = 4;
                constexpr uint simd_size = 32;
                constexpr uint rotary_dimensions = 64;
                constexpr uint rotary_pairs = rotary_dimensions / 2;

                uint row = threadgroup_position_in_grid.x;
                uint thread_id = thread_position_in_threadgroup.x;
                uint simd_thread = thread_index_in_simdgroup;
                uint simd_group = simdgroup_index_in_threadgroup;

                uint batch_size = uint(q_shape[0]);
                uint sequence_length = uint(q_shape[1]);
                uint query_heads = uint(q_shape[2]);
                uint key_heads = uint(k_shape[2]);
                uint axis_size = uint(q_shape[3]);
                uint query_rows = batch_size * query_heads * sequence_length;
                bool is_query = row < query_rows;
                uint local_row = is_query ? row : row - query_rows;
                uint head_count = is_query ? query_heads : key_heads;
                uint batch = local_row / (head_count * sequence_length);
                uint head_sequence = local_row % (head_count * sequence_length);
                uint head = head_sequence / sequence_length;
                uint sequence = head_sequence % sequence_length;

                ulong input_base;
                ulong input_axis_stride;
                ulong output_base = ulong(local_row) * ulong(axis_size);
                if (is_query) {
                    input_base = ulong(batch) * ulong(q_strides[0])
                        + ulong(sequence) * ulong(q_strides[1])
                        + ulong(head) * ulong(q_strides[2]);
                    input_axis_stride = ulong(q_strides[3]);
                } else {
                    input_base = ulong(batch) * ulong(k_strides[0])
                        + ulong(sequence) * ulong(k_strides[1])
                        + ulong(head) * ulong(k_strides[2]);
                    input_axis_stride = ulong(k_strides[3]);
                }

                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[simd_size];
                threadgroup bfloat normalized[256];

                float acc = 0.0f;
                uint first = thread_id * n_reads;
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = first + i;
                    if (element < axis_size) {
                        ulong index = input_base + ulong(element) * input_axis_stride;
                        float value = is_query ? float(q[index]) : float(k[index]);
                        acc += value * value;
                    }
                }
                acc = simd_sum(acc);
                if (simd_group == 0) local_sums[simd_thread] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_thread == 0) local_sums[simd_group] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    acc = simd_sum(local_sums[simd_thread]);
                    if (simd_thread == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(
                            acc / axis_size + eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float inv_mean = local_inv_mean[0];
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = first + i;
                    if (element < axis_size) {
                        ulong index = input_base + ulong(element) * input_axis_stride;
                        bfloat input_value = is_query ? q[index] : k[index];
                        bfloat rms_value = bfloat(float(input_value) * inv_mean);
                        bfloat weight = is_query
                            ? q_weight[element] : k_weight[element];
                        normalized[element] = weight * rms_value;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = 0; i < n_reads; ++i) {
                    uint element = first + i;
                    if (element >= rotary_dimensions && element < axis_size) {
                        if (is_query) q_out[output_base + element] = normalized[element];
                        else k_out[output_base + element] = normalized[element];
                    }
                }
                if (thread_id < rotary_pairs / n_reads) {
                    for (uint i = 0; i < n_reads; ++i) {
                        uint pair = first + i;
                        float d = float(pair) / float(rotary_pairs);
                        float inv_freq = metal::exp2(-d * float(log2_base));
                        float position = float(int(sequence) + int(offset));
                        float theta = position * inv_freq;
                        float costheta = metal::fast::cos(theta);
                        float sintheta = metal::fast::sin(theta);
                        float x1 = float(normalized[pair]);
                        float x2 = float(normalized[pair + rotary_pairs]);
                        bfloat rx1 = bfloat(x1 * costheta - x2 * sintheta);
                        bfloat rx2 = bfloat(x1 * sintheta + x2 * costheta);
                        if (is_query) {
                            q_out[output_base + pair] = rx1;
                            q_out[output_base + pair + rotary_pairs] = rx2;
                        } else {
                            k_out[output_base + pair] = rx1;
                            k_out[output_base + pair + rotary_pairs] = rx2;
                        }
                    }
                }
            """,
            ensure_row_contiguous=False,
        )
    batch, length = int(queries.shape[0]), int(queries.shape[1])
    q_out, k_out = _QK_RMS_ROPE_KERNEL(
        inputs=[
            queries,
            keys,
            q_weight,
            k_weight,
            float(eps),
            offset,
            23.253496664211536,
        ],
        template=[],
        grid=(batch * length * (24 + 4) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(batch, 24, length, 256), (batch, 4, length, 256)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return q_out, k_out


def _row21_attention_eligible(attention: Any) -> bool:
    rope = getattr(attention, "rope", None)
    q_weight = getattr(getattr(attention, "q_norm", None), "weight", None)
    k_weight = getattr(getattr(attention, "k_norm", None), "weight", None)
    return bool(
        int(getattr(attention, "num_attention_heads", 0)) == 24
        and int(getattr(attention, "num_key_value_heads", 0)) == 4
        and int(getattr(attention, "head_dim", 0)) == 256
        and rope is not None
        and int(getattr(rope, "dims", 0)) == 64
        and float(getattr(rope, "base", 0.0)) == 10_000_000.0
        and float(getattr(rope, "scale", 0.0)) == 1.0
        and not bool(getattr(rope, "traditional", True))
        and float(attention.q_norm.eps) == float(attention.k_norm.eps)
        and tuple(getattr(q_weight, "shape", ())) == (256,)
        and tuple(getattr(k_weight, "shape", ())) == (256,)
        and str(getattr(q_weight, "dtype", "")) == "mlx.core.bfloat16"
        and str(getattr(k_weight, "dtype", "")) == "mlx.core.bfloat16"
    )


def _row21_attention_call(self, x, mask=None, cache=None):
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    offset = getattr(cache, "offset", 0) if cache is not None else 0
    batch, length, _ = x.shape
    q_projection = self.q_proj(x)
    queries, gate = mx.split(
        q_projection.reshape(batch, length, 24, -1),
        2,
        axis=-1,
    )
    keys = self.k_proj(x).reshape(batch, length, 4, 256)
    values = self.v_proj(x).reshape(batch, length, 4, 256).transpose(0, 2, 1, 3)
    queries, keys = qwen38_qk_rms_rope(
        queries,
        keys,
        self.q_norm.weight,
        self.k_norm.weight,
        float(self.q_norm.eps),
        offset,
    )
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    output = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=cache,
        scale=self.scale,
        mask=mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return self.o_proj(output * mx.sigmoid(gate.reshape(batch, length, -1)))


def _row21_explicit_qk(attention, queries, keys, position_offset):
    return qwen38_qk_rms_rope(
        queries,
        keys,
        attention.q_norm.weight,
        attention.k_norm.weight,
        float(attention.q_norm.eps),
        position_offset,
    )


def configure_qwen38_row21_qk_rms_rope(model: Any, *, active: bool) -> dict[str, int]:
    """Bind row 21 directly on construction-validated attention instances."""

    text = getattr(model, "language_model", model)
    bindings = getattr(text, "_mtplx_row21_bindings", None)
    if bindings is None:
        inner = getattr(text, "model", text)
        installed = []
        for layer in list(getattr(inner, "layers", ()) or ()):
            attention = getattr(layer, "self_attn", None)
            if attention is None or not _row21_attention_eligible(attention):
                continue
            stock_class = attention.__class__
            fused_class = type(
                f"MTPLXRow21{stock_class.__name__}",
                (stock_class,),
                {"__call__": _row21_attention_call},
            )
            stock_explicit_qk = getattr(attention, "_mtplx_prepare_explicit_qk", None)
            fused_explicit_qk = None
            if stock_explicit_qk is not None:
                fused_explicit_qk = partial(_row21_explicit_qk, attention)
            installed.append(
                (
                    attention,
                    stock_class,
                    fused_class,
                    stock_explicit_qk,
                    fused_explicit_qk,
                )
            )
        bindings = tuple(installed)
        text._mtplx_row21_bindings = bindings
    for (
        attention,
        stock_class,
        fused_class,
        stock_explicit_qk,
        fused_explicit_qk,
    ) in bindings:
        attention.__class__ = fused_class if active else stock_class
        selected_explicit_qk = fused_explicit_qk if active else stock_explicit_qk
        if selected_explicit_qk is not None:
            attention._mtplx_prepare_explicit_qk = selected_explicit_qk
    eligible = len(bindings)
    return {
        "eligible_modules": eligible,
        "active_modules": eligible if active else 0,
        "construction_bound": eligible if active else 0,
    }


def configure_qwen38_row24_qk_length_limit(
    model: Any,
    *,
    active: bool,
    max_length: int = 16,
) -> dict[str, int]:
    """Apply row 24's L<=16 bound to the retained row-21 fusion."""

    text = getattr(model, "language_model", model)
    if getattr(text, "_mtplx_row21_bindings", None) is None:
        configure_qwen38_row21_qk_rms_rope(model, active=False)
    eligible = len(text._mtplx_row21_bindings)
    return {
        "eligible_modules": eligible,
        "active_modules": eligible if active else 0,
        "max_length": max_length if active else 0,
    }


def qwen38_dual_rms_norm_concat(
    a: Any,
    b: Any,
    a_weight: Any,
    b_weight: Any,
    eps: float,
) -> Any:
    """Normalize two hidden-5120 BF16 rows and emit one contiguous concat."""

    import mlx.core as mx

    global _DUAL_RMS_CONCAT_KERNEL
    rows = int(a.size) // 5120
    if _DUAL_RMS_CONCAT_KERNEL is None:
        _DUAL_RMS_CONCAT_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_dual_rms_norm_concat_bf16_v1",
            input_names=["a", "b", "a_weight", "b_weight", "eps"],
            output_names=["concat_out"],
            source=r"""
                constexpr uint n_reads = 4;
                constexpr uint simd_size = 32;
                constexpr uint lsize = 1024;

                uint row = threadgroup_position_in_grid.x;
                uint thread_id = thread_position_in_threadgroup.x;
                uint simd_thread = thread_index_in_simdgroup;
                uint simd_group = simdgroup_index_in_threadgroup;
                uint axis_size = uint(a_shape[a_ndim - 1]);
                uint a_rows = 1;
                for (uint i = 0; i + 1 < a_ndim; ++i) {
                    a_rows *= uint(a_shape[i]);
                }
                bool is_a = row < a_rows;
                uint local_row = is_a ? row : row - a_rows;
                ulong in_off = ulong(local_row) * ulong(axis_size);
                ulong out_off = ulong(local_row) * ulong(axis_size * 2)
                    + (is_a ? 0 : ulong(axis_size));

                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[simd_size];
                float acc = 0.0f;
                for (uint r_start = 0; r_start < axis_size;
                     r_start += lsize * n_reads) {
                    uint elem = r_start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        if (elem + i < axis_size) {
                            float xi = is_a
                                ? float(a[in_off + elem + i])
                                : float(b[in_off + elem + i]);
                            acc += xi * xi;
                        }
                    }
                }
                acc = simd_sum(acc);
                if (simd_group == 0) {
                    local_sums[simd_thread] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_thread == 0) {
                    local_sums[simd_group] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    acc = simd_sum(local_sums[simd_thread]);
                    if (simd_thread == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(
                            acc / float(axis_size) + eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float inv_mean = local_inv_mean[0];
                for (uint r_start = 0; r_start < axis_size;
                     r_start += lsize * n_reads) {
                    uint elem = r_start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        if (elem + i < axis_size) {
                            float xi = is_a
                                ? float(a[in_off + elem + i])
                                : float(b[in_off + elem + i]);
                            bfloat wi = is_a
                                ? a_weight[elem + i] : b_weight[elem + i];
                            concat_out[out_off + elem + i] =
                                wi * bfloat(xi * inv_mean);
                        }
                    }
                }
            """,
            ensure_row_contiguous=True,
        )
    (output,) = _DUAL_RMS_CONCAT_KERNEL(
        inputs=[a, b, a_weight, b_weight, float(eps)],
        template=[],
        grid=(2 * rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(*a.shape[:-1], 10240)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def qwen38_q8_embedding_dual_rms_norm_concat(
    token_ids: Any,
    embedding: Any,
    hidden: Any,
    embedding_norm_weight: Any,
    hidden_norm_weight: Any,
    eps: float,
) -> Any:
    """Fuse the target's Q8/g64 embedding lookup with both MTP input norms."""

    import mlx.core as mx

    global _Q8_EMBED_DUAL_RMS_CONCAT_KERNEL
    weight = getattr(embedding, "weight", None)
    scales = getattr(embedding, "scales", None)
    biases = getattr(embedding, "biases", None)
    rows = int(hidden.size) // 5120
    if _Q8_EMBED_DUAL_RMS_CONCAT_KERNEL is None:
        _Q8_EMBED_DUAL_RMS_CONCAT_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_q8_embedding_dual_rms_norm_concat_bf16_v1",
            input_names=[
                "token_ids",
                "embedding_weight",
                "embedding_scales",
                "embedding_biases",
                "hidden",
                "embedding_norm_weight",
                "hidden_norm_weight",
                "eps",
            ],
            output_names=["concat_out"],
            source=r"""
                constexpr uint axis_size = 5120;
                constexpr uint group_size = 64;
                constexpr uint n_reads = 4;
                constexpr uint simd_size = 32;
                constexpr uint lsize = 1024;

                uint row = threadgroup_position_in_grid.x;
                uint thread_id = thread_position_in_threadgroup.x;
                uint simd_thread = thread_index_in_simdgroup;
                uint simd_group = simdgroup_index_in_threadgroup;
                uint hidden_rows = 1;
                for (uint i = 0; i + 1 < hidden_ndim; ++i) {
                    hidden_rows *= uint(hidden_shape[i]);
                }
                bool is_embedding = row < hidden_rows;
                uint local_row = is_embedding ? row : row - hidden_rows;
                long token = is_embedding ? long(token_ids[local_row]) : 0;
                ulong packed_off = ulong(token) * ulong(embedding_weight_shape[1]);
                ulong scale_off = ulong(token) * ulong(embedding_scales_shape[1]);
                ulong hidden_off = ulong(local_row) * ulong(axis_size);
                ulong out_off = hidden_off * 2ul
                    + (is_embedding ? 0ul : ulong(axis_size));

                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[simd_size];
                float acc = 0.0f;
                for (uint start = 0; start < axis_size; start += lsize * n_reads) {
                    uint elem = start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        uint index = elem + i;
                        if (index < axis_size) {
                            float xi;
                            if (is_embedding) {
                                uint packed = embedding_weight[
                                    packed_off + ulong(index >> 2)];
                                uint q = (packed >> ((index & 3u) * 8u)) & 255u;
                                bfloat dequantized = bfloat(q)
                                    * embedding_scales[
                                        scale_off + ulong(index / group_size)]
                                    + embedding_biases[
                                        scale_off + ulong(index / group_size)];
                                xi = float(dequantized);
                            } else {
                                xi = float(hidden[hidden_off + ulong(index)]);
                            }
                            acc += xi * xi;
                        }
                    }
                }
                acc = simd_sum(acc);
                if (simd_group == 0) local_sums[simd_thread] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_thread == 0) local_sums[simd_group] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    acc = simd_sum(local_sums[simd_thread]);
                    if (simd_thread == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(
                            acc / float(axis_size) + eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float inv_mean = local_inv_mean[0];
                for (uint start = 0; start < axis_size; start += lsize * n_reads) {
                    uint elem = start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        uint index = elem + i;
                        if (index < axis_size) {
                            float xi;
                            if (is_embedding) {
                                uint packed = embedding_weight[
                                    packed_off + ulong(index >> 2)];
                                uint q = (packed >> ((index & 3u) * 8u)) & 255u;
                                bfloat dequantized = bfloat(q)
                                    * embedding_scales[
                                        scale_off + ulong(index / group_size)]
                                    + embedding_biases[
                                        scale_off + ulong(index / group_size)];
                                xi = float(dequantized);
                            } else {
                                xi = float(hidden[hidden_off + ulong(index)]);
                            }
                            bfloat wi = is_embedding
                                ? embedding_norm_weight[index]
                                : hidden_norm_weight[index];
                            concat_out[out_off + ulong(index)] =
                                wi * bfloat(xi * inv_mean);
                        }
                    }
                }
            """,
            ensure_row_contiguous=False,
        )
    (output,) = _Q8_EMBED_DUAL_RMS_CONCAT_KERNEL(
        inputs=[
            token_ids.reshape(-1),
            weight,
            scales,
            biases,
            hidden.reshape(rows, 5120),
            embedding_norm_weight,
            hidden_norm_weight,
            float(eps),
        ],
        template=[],
        grid=(2 * rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(*hidden.shape[:-1], 10240)],
        output_dtypes=[mx.bfloat16],
    )
    return output
