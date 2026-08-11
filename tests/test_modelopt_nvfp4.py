from __future__ import annotations

import mlx.core as mx

from mtplx.modelopt_nvfp4 import (
    dequantize_modelopt_fp8,
    dequantize_modelopt_nvfp4,
)


def test_modelopt_nvfp4_dequantizes_nibbles_with_both_scales():
    # Low nibble first: 0x21 -> [0.5, 1.0], 0xCB -> [-1.5, -2.0].
    packed = mx.array([[0x21, 0xCB] + [0x00] * 6], dtype=mx.uint8)
    block_scale = mx.array([[2.0]], dtype=mx.float32)
    global_scale = mx.array(0.25, dtype=mx.float32)

    weight = dequantize_modelopt_nvfp4(packed, block_scale, global_scale)

    assert mx.allclose(
        weight[:, :4], mx.array([[0.25, 0.5, -0.75, -1.0]])
    ).item()


def test_modelopt_fp8_dequantizes_with_exported_weight_scale():
    weight = mx.array([[1.0, -2.0]], dtype=mx.float32)
    scale = mx.array(0.125, dtype=mx.float32)

    actual = dequantize_modelopt_fp8(weight, scale)

    assert mx.allclose(actual, mx.array([[0.125, -0.25]])).item()
