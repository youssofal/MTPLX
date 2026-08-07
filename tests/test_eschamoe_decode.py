"""Bit-exact regression for the eschamoe MLX decode vs vendor CUDA goldens.

Fixture `tests/fixtures/eschamoe_mini.npz` holds real Escha-W2 codes + the weights the vendor
`escham_reconstruct` kernel produced for them (2x2 tiles of layer-0 expert 0, both projections).
The MLX decode must reproduce them exactly.
"""
import os
import numpy as np
import mlx.core as mx
import pytest

from mtplx.eschamoe import decode_expert_weights, decode_expert_weights_fast, t128

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "eschamoe_mini.npz")


@pytest.mark.parametrize("proj", ["gate_up_proj", "down_proj"])
def test_decode_bit_exact(proj):
    fx = np.load(FIX)
    K = int(fx[f"{proj}_K"])
    code = fx[f"{proj}_code"].astype(np.int16)     # [2,2,16K]
    gold = fx[f"{proj}_W"].astype(np.float16)       # [32,32]
    W = np.array(decode_expert_weights(mx.array(code), K)).astype(np.float16)
    assert W.shape == gold.shape
    assert np.array_equal(W, gold), f"{proj}: {(W != gold).sum()} / {W.size} mismatches"


def test_dec_table_value_count():
    # the codebook must have exactly 10746 distinct fp16 values (matches the vendor kernel)
    from mtplx.eschamoe import _build_dec_table
    assert np.unique(np.array(_build_dec_table())).size == 10746


def test_t128_involution_scale():
    # T128 is orthonormal: applying it twice (no scales) returns the input.
    # Tolerance is loose because MLX GPU fp32 matmul is lower precision than numpy;
    # the model runs fp16, so this is well within budget.
    x = mx.array(np.random.RandomState(0).randn(3, 256).astype(np.float32))
    y = t128(t128(x))
    assert float(mx.max(mx.abs(y - x))) < 1e-2
