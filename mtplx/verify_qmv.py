"""Small-M quantized vector-matrix probes for VerifyCore.

This module intentionally starts narrow: affine int4, M=3, row-contiguous
inputs, and the Qwen3.6 MLP/GDN dimensions that MLX routes through qmv.
The goal is to test whether reusing dequantized weight loads across the three
verify rows can beat stock ``mx.quantized_matmul`` before committing to an MLX
fork.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def is_nvfp4_eligible(module: Any) -> bool:
    """Return whether a QuantizedLinear uses native MLX nvfp4 (group size 16)."""
    if not isinstance(module, nn.QuantizedLinear):
        return False
    if int(getattr(module, "bits", 0) or 0) != 4:
        return False
    if int(getattr(module, "group_size", 0) or 0) != 16:
        return False
    if str(getattr(module, "mode", "")) != "nvfp4":
        return False
    if getattr(module, "scales", None) is None:
        return False
    return True


def nvfp4_stock_matmul(
    x: mx.array,
    module: nn.QuantizedLinear,
) -> mx.array:
    """Route nvfp4 modules through stock ``mx.quantized_matmul``."""
    if not is_nvfp4_eligible(module):
        return module(x)
    if len(x.shape) < 2 or x.dtype not in (mx.bfloat16, mx.float16):
        return module(x)

    kwargs: dict[str, Any] = {
        "transpose": True,
        "group_size": int(module.group_size),
        "bits": int(module.bits),
        "mode": str(module.mode),
    }
    global_scale_w = module.get("global_scale_w") if hasattr(module, "get") else None
    if global_scale_w is not None:
        kwargs["global_scale_w"] = global_scale_w
    out = mx.quantized_matmul(x, module.weight, module.scales, **kwargs)
    if "bias" in module:
        out = out + module["bias"]
    return out


def is_multi3_qmv4_eligible(module: Any) -> bool:
    """Return whether a QuantizedLinear can use the experimental M=3 qmv path."""
    if not isinstance(module, nn.QuantizedLinear):
        return False
    if int(getattr(module, "bits", 0) or 0) != 4:
        return False
    if int(getattr(module, "group_size", 0) or 0) not in {32, 64, 128}:
        return False
    if str(getattr(module, "mode", "affine")) != "affine":
        return False
    if getattr(module, "biases", None) is None:
        return False
    k = int(module.weight.shape[1]) * 8
    n = int(module.weight.shape[0])
    return k % 512 == 0 and n % 8 == 0


def is_smalln_pair_qmv4_eligible(left: Any, right: Any) -> bool:
    """Return whether two tiny-N affine int4 linears can use the pair kernel."""
    if not (
        isinstance(left, nn.QuantizedLinear) and isinstance(right, nn.QuantizedLinear)
    ):
        return False
    if (
        int(getattr(left, "bits", 0) or 0) != 4
        or int(getattr(right, "bits", 0) or 0) != 4
    ):
        return False
    if int(getattr(left, "group_size", 0) or 0) != int(
        getattr(right, "group_size", 0) or 0
    ):
        return False
    if int(getattr(left, "group_size", 0) or 0) not in {32, 64, 128}:
        return False
    if (
        str(getattr(left, "mode", "affine")) != "affine"
        or str(getattr(right, "mode", "affine")) != "affine"
    ):
        return False
    if getattr(left, "biases", None) is None or getattr(right, "biases", None) is None:
        return False
    if (
        left.weight.shape != right.weight.shape
        or left.scales.shape != right.scales.shape
    ):
        return False
    k = int(left.weight.shape[1]) * 8
    n = int(left.weight.shape[0])
    return k % 512 == 0 and n == 48


@lru_cache(maxsize=None)
def _multi3_qmv4_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = 2;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
    """

    source = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* ws_base =
          (const device uint8_t*)w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* scales_base =
          scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* biases_base =
          biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

        const device T* x0_base = x + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x1_base = x + K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x2_base = x + 2 * K + int(simd_lid) * VALUES_PER_THREAD;

        float result0[RESULTS_PER_SIMDGROUP] = {0.0f};
        float result1[RESULTS_PER_SIMDGROUP] = {0.0f};
        float result2[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x0_thread[VALUES_PER_THREAD];
        float x1_thread[VALUES_PER_THREAD];
        float x2_thread[VALUES_PER_THREAD];

        const device uint8_t* ws = ws_base;
        const device T* sc = scales_base;
        const device T* bs = biases_base;
        const device T* x0 = x0_base;
        const device T* x1 = x1_base;
        const device T* x2 = x2_base;

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float sum0 = load_vector4_exact<T>(x0, x0_thread);
          float sum1 = load_vector4_exact<T>(x1, x1_thread);
          float sum2 = load_vector4_exact<T>(x2, x2_thread);

          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              const device T* sl = sc + row * in_vec_size_g;
              const device T* bl = bs + row * in_vec_size_g;
              float s = float(sl[0]);
              float b = float(bl[0]);
              result0[row] += qdot4_exact(wl, x0_thread, s, b, sum0);
              result1[row] += qdot4_exact(wl, x1_thread, s, b, sum1);
              result2[row] += qdot4_exact(wl, x2_thread, s, b, sum2);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          x0 += BLOCK_SIZE;
          x1 += BLOCK_SIZE;
          x2 += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            float r0 = simd_sum(result0[row]);
            float r1 = simd_sum(result1[row]);
            float r2 = simd_sum(result2[row]);
            if (simd_lid == 0) {
              y[n] = T(r0);
              y[N + n] = T(r1);
              y[2 * N + n] = T(r2);
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_multi3_qmv4_v2_gs{group_size}_{dtype_tag}",
        input_names=["x", "w", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
        header=header,
    )


def multi3_qmv4_matmul(
    x: mx.array,
    module: nn.QuantizedLinear,
) -> mx.array:
    """Compute ``x @ module.weight.T`` with an experimental M=3 qmv kernel.

    Falls back to the module for unsupported shapes so callers can wire it into
    probes without risking invalid runtime behavior.
    """
    if is_nvfp4_eligible(module):
        return nvfp4_stock_matmul(x, module)
    if not is_multi3_qmv4_eligible(module):
        return module(x)
    if len(x.shape) < 2 or int(x.shape[-2]) != 3:
        return module(x)
    if x.dtype not in (mx.bfloat16, mx.float16):
        return module(x)
    if module.scales.dtype != x.dtype or module.biases.dtype != x.dtype:
        return module(x)

    k = int(x.shape[-1])
    n = int(module.weight.shape[0])
    if k != int(module.weight.shape[1]) * 8 or k % 512 != 0 or n % 8 != 0:
        return module(x)

    leading = x.shape[:-2]
    batch_count = 1
    for dim in leading:
        batch_count *= int(dim)
    if batch_count != 1:
        return module(x)

    x2 = mx.contiguous(x.reshape(3, k))
    kernel = _multi3_qmv4_kernel(int(module.group_size), x.dtype)
    (y,) = kernel(
        inputs=[x2, module.weight, module.scales, module.biases, k, n],
        template=[("T", x.dtype), ("GS", int(module.group_size))],
        grid=(32, 2 * (n // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(3, n)],
        output_dtypes=[x.dtype],
    )
    if "bias" in module:
        y = y + module["bias"]
    return y.reshape(*leading, 3, n)


@lru_cache(maxsize=None)
def _smalln_pair_qmv4_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
    """
    source = """
        uint out_index = threadgroup_position_in_grid.y;
        uint simd_lid = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        bool use_right = int(out_index) >= N;
        int n = int(out_index) - (use_right ? N : 0);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* w_base =
          (const device uint8_t*)(use_right ? w_right : w_left)
          + n * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* scales_base =
          (use_right ? scales_right : scales_left)
          + n * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* biases_base =
          (use_right ? biases_right : biases_left)
          + n * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

        const device T* x0 = x + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x1 = x + K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x2 = x + 2 * K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x3 = x + 3 * K + int(simd_lid) * VALUES_PER_THREAD;

        float result0 = 0.0f;
        float result1 = 0.0f;
        float result2 = 0.0f;
        float result3 = 0.0f;
        float x0_thread[VALUES_PER_THREAD];
        float x1_thread[VALUES_PER_THREAD];
        float x2_thread[VALUES_PER_THREAD];
        float x3_thread[VALUES_PER_THREAD];

        const device uint8_t* w = w_base;
        const device T* sc = scales_base;
        const device T* bs = biases_base;

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float sum0 = load_vector4_exact<T>(x0, x0_thread);
          float sum1 = load_vector4_exact<T>(x1, x1_thread);
          float sum2 = load_vector4_exact<T>(x2, x2_thread);
          float sum3 = load_vector4_exact<T>(x3, x3_thread);
          float scale = float(sc[0]);
          float bias = float(bs[0]);
          result0 += qdot4_exact(w, x0_thread, scale, bias, sum0);
          result1 += qdot4_exact(w, x1_thread, scale, bias, sum1);
          result2 += qdot4_exact(w, x2_thread, scale, bias, sum2);
          result3 += qdot4_exact(w, x3_thread, scale, bias, sum3);

          w += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          x0 += BLOCK_SIZE;
          x1 += BLOCK_SIZE;
          x2 += BLOCK_SIZE;
          x3 += BLOCK_SIZE;
        }

        device T* y = use_right ? y_right : y_left;
        float reduced0 = simd_sum(result0);
        float reduced1 = simd_sum(result1);
        float reduced2 = simd_sum(result2);
        float reduced3 = simd_sum(result3);
        if (simd_lid == 0) {
          y[n] = T(reduced0);
          y[N + n] = T(reduced1);
          y[2 * N + n] = T(reduced2);
          y[3 * N + n] = T(reduced3);
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_smalln_pair_qmv4_gs{group_size}_{dtype_tag}",
        input_names=[
            "x",
            "w_left",
            "scales_left",
            "biases_left",
            "w_right",
            "scales_right",
            "biases_right",
            "K_size",
            "N_size",
        ],
        output_names=["y_left", "y_right"],
        source=source,
        header=header,
    )


def smalln_pair_qmv4_matmul(
    x: mx.array,
    left_module: nn.QuantizedLinear,
    right_module: nn.QuantizedLinear,
) -> tuple[mx.array, mx.array]:
    """Compute a pair of tiny-N int4 projections for M=4 verify tensors."""
    if not is_smalln_pair_qmv4_eligible(left_module, right_module):
        return left_module(x), right_module(x)
    if len(x.shape) < 2 or int(x.shape[-2]) != 4:
        return left_module(x), right_module(x)
    if x.dtype not in (mx.bfloat16, mx.float16):
        return left_module(x), right_module(x)
    if (
        left_module.scales.dtype != x.dtype
        or left_module.biases.dtype != x.dtype
        or right_module.scales.dtype != x.dtype
        or right_module.biases.dtype != x.dtype
    ):
        return left_module(x), right_module(x)

    k = int(x.shape[-1])
    n = int(left_module.weight.shape[0])
    if k != int(left_module.weight.shape[1]) * 8 or k % 512 != 0 or n != 48:
        return left_module(x), right_module(x)

    leading = x.shape[:-2]
    batch_count = 1
    for dim in leading:
        batch_count *= int(dim)
    if batch_count != 1:
        return left_module(x), right_module(x)

    x2 = mx.contiguous(x.reshape(4, k))
    kernel = _smalln_pair_qmv4_kernel(int(left_module.group_size), x.dtype)
    y_left, y_right = kernel(
        inputs=[
            x2,
            left_module.weight,
            left_module.scales,
            left_module.biases,
            right_module.weight,
            right_module.scales,
            right_module.biases,
            k,
            n,
        ],
        template=[("T", x.dtype), ("GS", int(left_module.group_size))],
        grid=(32, 2 * n, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(4, n), (4, n)],
        output_dtypes=[x.dtype, x.dtype],
    )
    if "bias" in left_module:
        y_left = y_left + left_module["bias"]
    if "bias" in right_module:
        y_right = y_right + right_module["bias"]
    return y_left.reshape(*leading, 4, n), y_right.reshape(*leading, 4, n)


@lru_cache(maxsize=None)
def _multi3_dual_qmv4_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = 2;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
    """
    source = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* wg_base =
          (const device uint8_t*)w_gate + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sg_base =
          scales_gate + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bg_base =
          biases_gate + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

        const device uint8_t* wu_base =
          (const device uint8_t*)w_up + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* su_base =
          scales_up + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bu_base =
          biases_up + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

        const device T* x0_base = x + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x1_base = x + K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x2_base = x + 2 * K + int(simd_lid) * VALUES_PER_THREAD;

        float gate0[RESULTS_PER_SIMDGROUP] = {0.0f};
        float gate1[RESULTS_PER_SIMDGROUP] = {0.0f};
        float gate2[RESULTS_PER_SIMDGROUP] = {0.0f};
        float up0[RESULTS_PER_SIMDGROUP] = {0.0f};
        float up1[RESULTS_PER_SIMDGROUP] = {0.0f};
        float up2[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x0_thread[VALUES_PER_THREAD];
        float x1_thread[VALUES_PER_THREAD];
        float x2_thread[VALUES_PER_THREAD];

        const device uint8_t* wg = wg_base;
        const device T* sg = sg_base;
        const device T* bg = bg_base;
        const device uint8_t* wu = wu_base;
        const device T* su = su_base;
        const device T* bu = bu_base;
        const device T* x0 = x0_base;
        const device T* x1 = x1_base;
        const device T* x2 = x2_base;

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float sum0 = load_vector4_exact<T>(x0, x0_thread);
          float sum1 = load_vector4_exact<T>(x1, x1_thread);
          float sum2 = load_vector4_exact<T>(x2, x2_thread);

          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* wgl = wg + row * in_vec_size_w;
              const device T* sgl = sg + row * in_vec_size_g;
              const device T* bgl = bg + row * in_vec_size_g;
              float gs = float(sgl[0]);
              float gb = float(bgl[0]);
              gate0[row] += qdot4_exact(wgl, x0_thread, gs, gb, sum0);
              gate1[row] += qdot4_exact(wgl, x1_thread, gs, gb, sum1);
              gate2[row] += qdot4_exact(wgl, x2_thread, gs, gb, sum2);

              const device uint8_t* wul = wu + row * in_vec_size_w;
              const device T* sul = su + row * in_vec_size_g;
              const device T* bul = bu + row * in_vec_size_g;
              float us = float(sul[0]);
              float ub = float(bul[0]);
              up0[row] += qdot4_exact(wul, x0_thread, us, ub, sum0);
              up1[row] += qdot4_exact(wul, x1_thread, us, ub, sum1);
              up2[row] += qdot4_exact(wul, x2_thread, us, ub, sum2);
            }
          }

          wg += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sg += BLOCK_SIZE / GS;
          bg += BLOCK_SIZE / GS;
          wu += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          su += BLOCK_SIZE / GS;
          bu += BLOCK_SIZE / GS;
          x0 += BLOCK_SIZE;
          x1 += BLOCK_SIZE;
          x2 += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            float g0 = simd_sum(gate0[row]);
            float g1 = simd_sum(gate1[row]);
            float g2 = simd_sum(gate2[row]);
            float u0 = simd_sum(up0[row]);
            float u1 = simd_sum(up1[row]);
            float u2 = simd_sum(up2[row]);
            if (simd_lid == 0) {
              y_gate[n] = T(g0);
              y_gate[N + n] = T(g1);
              y_gate[2 * N + n] = T(g2);
              y_up[n] = T(u0);
              y_up[N + n] = T(u1);
              y_up[2 * N + n] = T(u2);
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_multi3_dual_qmv4_gs{group_size}_{dtype_tag}",
        input_names=[
            "x",
            "w_gate",
            "scales_gate",
            "biases_gate",
            "w_up",
            "scales_up",
            "biases_up",
            "K_size",
            "N_size",
        ],
        output_names=["y_gate", "y_up"],
        source=source,
        header=header,
    )


def multi3_dual_qmv4_matmul(
    x: mx.array,
    gate_module: nn.QuantizedLinear,
    up_module: nn.QuantizedLinear,
) -> tuple[mx.array, mx.array]:
    """Compute MLP gate/up projections together for M=3 VerifyCore probes."""
    if not (
        is_multi3_qmv4_eligible(gate_module) and is_multi3_qmv4_eligible(up_module)
    ):
        return gate_module(x), up_module(x)
    if gate_module.weight.shape != up_module.weight.shape:
        return gate_module(x), up_module(x)
    if gate_module.scales.shape != up_module.scales.shape:
        return gate_module(x), up_module(x)
    if int(gate_module.group_size) != int(up_module.group_size):
        return gate_module(x), up_module(x)
    if len(x.shape) < 2 or int(x.shape[-2]) != 3:
        return gate_module(x), up_module(x)
    if x.dtype not in (mx.bfloat16, mx.float16):
        return gate_module(x), up_module(x)
    if (
        gate_module.scales.dtype != x.dtype
        or gate_module.biases.dtype != x.dtype
        or up_module.scales.dtype != x.dtype
        or up_module.biases.dtype != x.dtype
    ):
        return gate_module(x), up_module(x)

    k = int(x.shape[-1])
    n = int(gate_module.weight.shape[0])
    if k != int(gate_module.weight.shape[1]) * 8 or k % 512 != 0 or n % 8 != 0:
        return gate_module(x), up_module(x)

    leading = x.shape[:-2]
    batch_count = 1
    for dim in leading:
        batch_count *= int(dim)
    if batch_count != 1:
        return gate_module(x), up_module(x)

    x2 = mx.contiguous(x.reshape(3, k))
    kernel = _multi3_dual_qmv4_kernel(int(gate_module.group_size), x.dtype)
    y_gate, y_up = kernel(
        inputs=[
            x2,
            gate_module.weight,
            gate_module.scales,
            gate_module.biases,
            up_module.weight,
            up_module.scales,
            up_module.biases,
            k,
            n,
        ],
        template=[("T", x.dtype), ("GS", int(gate_module.group_size))],
        grid=(32, 2 * (n // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(3, n), (3, n)],
        output_dtypes=[x.dtype, x.dtype],
    )
    if "bias" in gate_module:
        y_gate = y_gate + gate_module["bias"]
    if "bias" in up_module:
        y_up = y_up + up_module["bias"]
    return y_gate.reshape(*leading, 3, n), y_up.reshape(*leading, 3, n)


@lru_cache(maxsize=None)
def _multi3_parallel_dual_qmv4_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int SIMDGROUPS_PER_PROJECTION = 2;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * SIMDGROUPS_PER_PROJECTION;

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
    """

    source = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        bool is_up = simd_gid >= SIMDGROUPS_PER_PROJECTION;
        uint local_gid = simd_gid - (is_up ? SIMDGROUPS_PER_PROJECTION : 0);

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        int out_row = int(n_tile) * BN + int(local_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* w_base =
          (const device uint8_t*)(is_up ? w_up : w_gate)
          + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* scales_base =
          (is_up ? scales_up : scales_gate)
          + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* biases_base =
          (is_up ? biases_up : biases_gate)
          + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

        const device T* x0_base = x + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x1_base = x + K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x2_base = x + 2 * K + int(simd_lid) * VALUES_PER_THREAD;

        float result0[RESULTS_PER_SIMDGROUP] = {0.0f};
        float result1[RESULTS_PER_SIMDGROUP] = {0.0f};
        float result2[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x0_thread[VALUES_PER_THREAD];
        float x1_thread[VALUES_PER_THREAD];
        float x2_thread[VALUES_PER_THREAD];

        const device uint8_t* w = w_base;
        const device T* sc = scales_base;
        const device T* bs = biases_base;
        const device T* x0 = x0_base;
        const device T* x1 = x1_base;
        const device T* x2 = x2_base;

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float sum0 = load_vector4_exact<T>(x0, x0_thread);
          float sum1 = load_vector4_exact<T>(x1, x1_thread);
          float sum2 = load_vector4_exact<T>(x2, x2_thread);

          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* wl = w + row * in_vec_size_w;
              const device T* sl = sc + row * in_vec_size_g;
              const device T* bl = bs + row * in_vec_size_g;
              float s = float(sl[0]);
              float b = float(bl[0]);
              result0[row] += qdot4_exact(wl, x0_thread, s, b, sum0);
              result1[row] += qdot4_exact(wl, x1_thread, s, b, sum1);
              result2[row] += qdot4_exact(wl, x2_thread, s, b, sum2);
            }
          }

          w += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          x0 += BLOCK_SIZE;
          x1 += BLOCK_SIZE;
          x2 += BLOCK_SIZE;
        }

        device T* y = is_up ? y_up : y_gate;
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            float r0 = simd_sum(result0[row]);
            float r1 = simd_sum(result1[row]);
            float r2 = simd_sum(result2[row]);
            if (simd_lid == 0) {
              y[n] = T(r0);
              y[N + n] = T(r1);
              y[2 * N + n] = T(r2);
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_multi3_parallel_dual_qmv4_gs{group_size}_{dtype_tag}",
        input_names=[
            "x",
            "w_gate",
            "scales_gate",
            "biases_gate",
            "w_up",
            "scales_up",
            "biases_up",
            "K_size",
            "N_size",
        ],
        output_names=["y_gate", "y_up"],
        source=source,
        header=header,
    )


def multi3_parallel_dual_qmv4_matmul(
    x: mx.array,
    gate_module: nn.QuantizedLinear,
    up_module: nn.QuantizedLinear,
) -> tuple[mx.array, mx.array]:
    """Compute MLP gate/up with one dispatch and parallel simdgroup halves."""
    if not (
        is_multi3_qmv4_eligible(gate_module) and is_multi3_qmv4_eligible(up_module)
    ):
        return gate_module(x), up_module(x)
    if gate_module.weight.shape != up_module.weight.shape:
        return gate_module(x), up_module(x)
    if gate_module.scales.shape != up_module.scales.shape:
        return gate_module(x), up_module(x)
    if int(gate_module.group_size) != int(up_module.group_size):
        return gate_module(x), up_module(x)
    if len(x.shape) < 2 or int(x.shape[-2]) != 3:
        return gate_module(x), up_module(x)
    if x.dtype not in (mx.bfloat16, mx.float16):
        return gate_module(x), up_module(x)
    if (
        gate_module.scales.dtype != x.dtype
        or gate_module.biases.dtype != x.dtype
        or up_module.scales.dtype != x.dtype
        or up_module.biases.dtype != x.dtype
    ):
        return gate_module(x), up_module(x)

    k = int(x.shape[-1])
    n = int(gate_module.weight.shape[0])
    if k != int(gate_module.weight.shape[1]) * 8 or k % 512 != 0 or n % 8 != 0:
        return gate_module(x), up_module(x)

    leading = x.shape[:-2]
    batch_count = 1
    for dim in leading:
        batch_count *= int(dim)
    if batch_count != 1:
        return gate_module(x), up_module(x)

    x2 = mx.contiguous(x.reshape(3, k))
    kernel = _multi3_parallel_dual_qmv4_kernel(int(gate_module.group_size), x.dtype)
    y_gate, y_up = kernel(
        inputs=[
            x2,
            gate_module.weight,
            gate_module.scales,
            gate_module.biases,
            up_module.weight,
            up_module.scales,
            up_module.biases,
            k,
            n,
        ],
        template=[("T", x.dtype), ("GS", int(gate_module.group_size))],
        grid=(32, 4 * (n // 8), 1),
        threadgroup=(32, 4, 1),
        output_shapes=[(3, n), (3, n)],
        output_dtypes=[x.dtype, x.dtype],
    )
    if "bias" in gate_module:
        y_gate = y_gate + gate_module["bias"]
    if "bias" in up_module:
        y_up = y_up + up_module["bias"]
    return y_gate.reshape(*leading, 3, n), y_up.reshape(*leading, 3, n)


@lru_cache(maxsize=None)
def _multi3_swiglu_down_qmv4_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = 2;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

        template <typename T>
        inline T sigmoid_mlx_exact(T x) {
          auto y = 1 / (1 + metal::exp(metal::abs(x)));
          return (x < T(0)) ? y : 1 - y;
        }

        template <typename T>
        inline T swiglu_mlx_exact(T gate, T up) {
          T silu = gate * sigmoid_mlx_exact<T>(gate);
          return T(silu * up);
        }

        template <typename T>
        inline float load_swiglu_vector4_exact(
            const device T* gate,
            const device T* up,
            thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            T x0 = swiglu_mlx_exact<T>(gate[i], up[i]);
            T x1 = swiglu_mlx_exact<T>(gate[i + 1], up[i + 1]);
            T x2 = swiglu_mlx_exact<T>(gate[i + 2], up[i + 2]);
            T x3 = swiglu_mlx_exact<T>(gate[i + 3], up[i + 3]);
            sum += x0 + x1 + x2 + x3;
            x_thread[i] = x0;
            x_thread[i + 1] = x1 / 16.0f;
            x_thread[i + 2] = x2 / 256.0f;
            x_thread[i + 3] = x3 / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
    """

    source = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* ws_base =
          (const device uint8_t*)w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* scales_base =
          scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* biases_base =
          biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

        const device T* gate0_base = gate + int(simd_lid) * VALUES_PER_THREAD;
        const device T* gate1_base = gate + K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* gate2_base = gate + 2 * K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* up0_base = up + int(simd_lid) * VALUES_PER_THREAD;
        const device T* up1_base = up + K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* up2_base = up + 2 * K + int(simd_lid) * VALUES_PER_THREAD;

        float result0[RESULTS_PER_SIMDGROUP] = {0.0f};
        float result1[RESULTS_PER_SIMDGROUP] = {0.0f};
        float result2[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x0_thread[VALUES_PER_THREAD];
        float x1_thread[VALUES_PER_THREAD];
        float x2_thread[VALUES_PER_THREAD];

        const device uint8_t* ws = ws_base;
        const device T* sc = scales_base;
        const device T* bs = biases_base;
        const device T* gate0 = gate0_base;
        const device T* gate1 = gate1_base;
        const device T* gate2 = gate2_base;
        const device T* up0 = up0_base;
        const device T* up1 = up1_base;
        const device T* up2 = up2_base;

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float sum0 = load_swiglu_vector4_exact<T>(gate0, up0, x0_thread);
          float sum1 = load_swiglu_vector4_exact<T>(gate1, up1, x1_thread);
          float sum2 = load_swiglu_vector4_exact<T>(gate2, up2, x2_thread);

          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              const device T* sl = sc + row * in_vec_size_g;
              const device T* bl = bs + row * in_vec_size_g;
              float s = float(sl[0]);
              float b = float(bl[0]);
              result0[row] += qdot4_exact(wl, x0_thread, s, b, sum0);
              result1[row] += qdot4_exact(wl, x1_thread, s, b, sum1);
              result2[row] += qdot4_exact(wl, x2_thread, s, b, sum2);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          gate0 += BLOCK_SIZE;
          gate1 += BLOCK_SIZE;
          gate2 += BLOCK_SIZE;
          up0 += BLOCK_SIZE;
          up1 += BLOCK_SIZE;
          up2 += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            float r0 = simd_sum(result0[row]);
            float r1 = simd_sum(result1[row]);
            float r2 = simd_sum(result2[row]);
            if (simd_lid == 0) {
              y[n] = T(r0);
              y[N + n] = T(r1);
              y[2 * N + n] = T(r2);
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_multi3_swiglu_down_qmv4_gs{group_size}_{dtype_tag}",
        input_names=["gate", "up", "w", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
        header=header,
    )


def multi3_swiglu_down_qmv4_matmul(
    gate: mx.array,
    up: mx.array,
    module: nn.QuantizedLinear,
) -> mx.array:
    """Compute ``module(nn.silu(gate) * up)`` with fused activation input."""

    def fallback() -> mx.array:
        return module(nn.silu(gate) * up)

    if gate.shape != up.shape:
        return fallback()
    if not is_multi3_qmv4_eligible(module):
        return fallback()
    if len(gate.shape) < 2 or int(gate.shape[-2]) != 3:
        return fallback()
    if gate.dtype not in (mx.bfloat16, mx.float16) or up.dtype != gate.dtype:
        return fallback()
    if module.scales.dtype != gate.dtype or module.biases.dtype != gate.dtype:
        return fallback()

    k = int(gate.shape[-1])
    n = int(module.weight.shape[0])
    if k != int(module.weight.shape[1]) * 8 or k % 512 != 0 or n % 8 != 0:
        return fallback()

    leading = gate.shape[:-2]
    batch_count = 1
    for dim in leading:
        batch_count *= int(dim)
    if batch_count != 1:
        return fallback()

    gate2 = mx.contiguous(gate.reshape(3, k))
    up2 = mx.contiguous(up.reshape(3, k))
    kernel = _multi3_swiglu_down_qmv4_kernel(int(module.group_size), gate.dtype)
    (y,) = kernel(
        inputs=[gate2, up2, module.weight, module.scales, module.biases, k, n],
        template=[("T", gate.dtype), ("GS", int(module.group_size))],
        grid=(32, 2 * (n // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(3, n)],
        output_dtypes=[gate.dtype],
    )
    if "bias" in module:
        y = y + module["bias"]
    return y.reshape(*leading, 3, n)


@lru_cache(maxsize=None)
def _stocklike_qmv4_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = 2;

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
            accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
               x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
               x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
               x_thread[4 * i + 3] * (ws[i] & 0xf000));
          }
          return scale * accum + sum * bias;
        }
    """

    source = """
        uint m_idx = threadgroup_position_in_grid.x;
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;

        int out_row = int(n_tile) * (NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP)
          + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* ws =
          (const device uint8_t*)w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
          scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
          biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* x_row = x + int(m_idx) * K + int(simd_lid) * VALUES_PER_THREAD;

        float result[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x_thread[VALUES_PER_THREAD];

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float sum = load_vector4_exact<T>(x_row, x_thread);
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device T* sl = sc + row * in_vec_size_g;
            const device T* bl = bs + row * in_vec_size_g;
            result[row] += qdot4_exact(wl, x_thread, float(sl[0]), float(bl[0]), sum);
          }
          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          x_row += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          int n = out_row + row;
          if (n < N) {
            float reduced = simd_sum(result[row]);
            if (simd_lid == 0) {
              y[int(m_idx) * N + n] = T(reduced);
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_stocklike_qmv4_gs{group_size}_{dtype_tag}",
        input_names=["x", "w", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
        header=header,
    )


def stocklike_qmv4_matmul(
    x: mx.array,
    module: nn.QuantizedLinear,
) -> mx.array:
    """Probe kernel matching MLX qmv-fast tiling for exactness diagnosis."""
    if is_nvfp4_eligible(module):
        return nvfp4_stock_matmul(x, module)
    if not is_multi3_qmv4_eligible(module):
        return module(x)
    if len(x.shape) < 2 or x.dtype not in (mx.bfloat16, mx.float16):
        return module(x)
    if module.scales.dtype != x.dtype or module.biases.dtype != x.dtype:
        return module(x)

    m = int(x.shape[-2])
    k = int(x.shape[-1])
    n = int(module.weight.shape[0])
    if k != int(module.weight.shape[1]) * 8 or k % 512 != 0 or n % 8 != 0:
        return module(x)
    leading = x.shape[:-2]
    batch_count = 1
    for dim in leading:
        batch_count *= int(dim)
    if batch_count != 1:
        return module(x)

    x2 = mx.contiguous(x.reshape(m, k))
    kernel = _stocklike_qmv4_kernel(int(module.group_size), x.dtype)
    (y,) = kernel(
        inputs=[x2, module.weight, module.scales, module.biases, k, n],
        template=[("T", x.dtype), ("GS", int(module.group_size))],
        grid=(32 * m, 2 * (n // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    if "bias" in module:
        y = y + module["bias"]
    return y.reshape(*leading, m, n)


def is_stocklike_qmv8_eligible(module: Any) -> bool:
    """Return whether a QuantizedLinear can use the stocklike int8 qmv probe."""
    if not isinstance(module, nn.QuantizedLinear):
        return False
    if int(getattr(module, "bits", 0) or 0) != 8:
        return False
    if int(getattr(module, "group_size", 0) or 0) not in {32, 64, 128}:
        return False
    if str(getattr(module, "mode", "affine")) != "affine":
        return False
    if getattr(module, "biases", None) is None:
        return False
    k = int(module.weight.shape[1]) * 4
    n = int(module.weight.shape[0])
    return k % 256 == 0 and n % 8 == 0


@lru_cache(maxsize=None)
def _stocklike_qmv8_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = 8;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = 2;

        template <typename T>
        inline float load_vector8_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; ++i) {
            float xi = float(x[i]);
            sum += xi;
            x_thread[i] = xi;
          }
          return sum;
        }

        inline float qdot8_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          float accum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; ++i) {
            accum += x_thread[i] * float(w[i]);
          }
          return scale * accum + sum * bias;
        }
    """

    source = """
        uint m_idx = threadgroup_position_in_grid.x;
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;

        int out_row = int(n_tile) * (NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP)
          + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / 4;
        int in_vec_size_g = K / GS;

        const device uint8_t* ws =
          (const device uint8_t*)w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
          scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
          biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* x_row = x + int(m_idx) * K + int(simd_lid) * VALUES_PER_THREAD;

        float result[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x_thread[VALUES_PER_THREAD];

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float sum = load_vector8_exact<T>(x_row, x_thread);
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              const device T* sl = sc + row * in_vec_size_g;
              const device T* bl = bs + row * in_vec_size_g;
              result[row] += qdot8_exact(
                wl, x_thread, float(sl[0]), float(bl[0]), sum
              );
            }
          }
          ws += BLOCK_SIZE * BYTES_PER_PACK / 4;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          x_row += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            float reduced = simd_sum(result[row]);
            if (simd_lid == 0) {
              y[int(m_idx) * N + n] = T(reduced);
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_stocklike_qmv8_gs{group_size}_{dtype_tag}",
        input_names=["x", "w", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
        header=header,
    )


def stocklike_qmv8_matmul(
    x: mx.array,
    module: nn.QuantizedLinear,
) -> mx.array:
    """Probe kernel matching MLX qmv-fast tiling for affine int8 linears."""
    if not is_stocklike_qmv8_eligible(module):
        return module(x)
    if len(x.shape) < 2 or x.dtype not in (mx.bfloat16, mx.float16):
        return module(x)
    if module.scales.dtype != x.dtype or module.biases.dtype != x.dtype:
        return module(x)

    m = int(x.shape[-2])
    k = int(x.shape[-1])
    n = int(module.weight.shape[0])
    if k != int(module.weight.shape[1]) * 4 or k % 256 != 0 or n % 8 != 0:
        return module(x)
    leading = x.shape[:-2]
    batch_count = 1
    for dim in leading:
        batch_count *= int(dim)
    if batch_count != 1:
        return module(x)

    x2 = mx.contiguous(x.reshape(m, k))
    kernel = _stocklike_qmv8_kernel(int(module.group_size), x.dtype)
    (y,) = kernel(
        inputs=[x2, module.weight, module.scales, module.biases, k, n],
        template=[("T", x.dtype), ("GS", int(module.group_size))],
        grid=(32 * m, 2 * (n // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    if "bias" in module:
        y = y + module["bias"]
    return y.reshape(*leading, m, n)
