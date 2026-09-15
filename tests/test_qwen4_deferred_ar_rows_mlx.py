"""Real-MLX checks for the mutable packed leaves used by streamed AR."""

from __future__ import annotations

import numpy as np

import mlx.core as mx
import mtplx_native_ple_cpu_rows as native


def _payload(offset: int = 0):
    return (
        (np.arange(16 * 20, dtype=np.uint32) + offset).reshape(16, 20),
        (np.arange(16 * 5, dtype=np.uint16) + offset).reshape(16, 5),
        (np.arange(16 * 5, dtype=np.uint16) + offset + 500).reshape(16, 5),
    )


def test_gpu_consumer_built_before_fill_reads_the_filled_bytes():
    handle = native.make_deferred_ar_rows()
    weight, scales, biases = handle.planes()
    assert weight.shape == (16, 20) and weight.dtype == mx.uint32
    assert scales.shape == (16, 5) and scales.dtype == mx.bfloat16
    assert biases.shape == (16, 5) and biases.dtype == mx.bfloat16

    # Build all consumers while the leaves still contain zeros. Evaluation is
    # deliberately delayed until after fill, matching the AR scheduler.
    weight_seen = weight + mx.array(1, dtype=mx.uint32)
    scales_seen = scales.view(mx.uint16) + mx.array(2, dtype=mx.uint16)
    biases_seen = biases.view(mx.uint16) + mx.array(3, dtype=mx.uint16)
    payload = _payload()
    handle.fill(*payload)
    mx.eval(weight_seen, scales_seen, biases_seen)

    np.testing.assert_array_equal(np.asarray(weight_seen), payload[0] + 1)
    np.testing.assert_array_equal(np.asarray(scales_seen), payload[1] + 2)
    np.testing.assert_array_equal(np.asarray(biases_seen), payload[2] + 3)


def test_row_leaves_are_independent_across_steps():
    first = native.make_deferred_ar_rows()
    second = native.make_deferred_ar_rows()
    first.fill(*_payload(7))
    second.fill(*_payload(19))
    first_weight, _, _ = first.planes()
    second_weight, _, _ = second.planes()
    mx.eval(first_weight, second_weight)
    np.testing.assert_array_equal(np.asarray(first_weight), _payload(7)[0])
    np.testing.assert_array_equal(np.asarray(second_weight), _payload(19)[0])


def test_token_consumer_built_before_fill_reads_exact_int64_value():
    handle = native.make_deferred_ar_token()
    token = handle.array()
    assert token.shape == (1,) and token.dtype == mx.int64

    consumer = token * mx.array(3, dtype=mx.int64)
    handle.fill(712_345)
    mx.eval(consumer)

    np.testing.assert_array_equal(np.asarray(consumer), [2_137_035])


def test_token_leaves_are_independent_across_steps():
    first = native.make_deferred_ar_token()
    second = native.make_deferred_ar_token()
    first.fill(17)
    second.fill(23)
    first_value = first.array()
    second_value = second.array()
    mx.eval(first_value, second_value)
    np.testing.assert_array_equal(np.asarray(first_value), [17])
    np.testing.assert_array_equal(np.asarray(second_value), [23])
