"""A 4-bit embedding (Inkling Bare Speed) gathers the same rows as its dequantized matrix."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mtplx.models.inkling.model import Embedding
from mtplx.models.inkling.precision import AffineWeight


def _packed(rows=96, columns=256, bits=4, group=64):
    dense = (mx.random.normal((rows, columns), key=mx.random.key(7)) * 0.05).astype(mx.bfloat16)
    codes, scales, biases = mx.quantize(dense, group_size=group, bits=bits)
    return AffineWeight(codes, scales, biases, bits, group)


def test_packed_embedding_matches_its_dequantized_matrix_row_for_row():
    with mx.stream(mx.cpu):
        weight = _packed()
        gain = mx.ones((256,), dtype=mx.bfloat16) * 1.5
        dense = mx.dequantize(weight.codes, weight.scales, weight.biases, bits=4, group_size=64)
        ids = mx.array([[3, 0, 95, 3, 41]])
        packed_rows = Embedding(weight, gain, 1e-6)(ids)
        dense_rows = Embedding(dense, gain, 1e-6)(ids)
        assert packed_rows.dtype == dense_rows.dtype == mx.bfloat16
        assert mx.array_equal(packed_rows, dense_rows).item()


def test_packed_embedding_answers_shape_and_dtype_but_holds_no_data():
    embedding = Embedding(_packed(), mx.ones((256,), dtype=mx.bfloat16), 1e-6)
    assert embedding.weight.shape == (96, 256)
    assert embedding.weight.shape[0] == 96
    assert embedding.weight.dtype == mx.bfloat16 and embedding.weight.dtype.size == 2
    with pytest.raises(TypeError):
        embedding.weight[mx.array([1])]
    # The codes are the module's parameters; the stand-in is not.
    names = {name for name, _ in tree_flatten(embedding.parameters())}
    assert {"packed.codes", "packed.scales", "packed.biases", "gain"} <= names
    assert "weight" not in names


def test_bf16_embedding_is_unchanged():
    dense = mx.arange(12, dtype=mx.float32).reshape(4, 3).astype(mx.bfloat16)
    embedding = Embedding(dense, mx.ones((3,), dtype=mx.bfloat16), 1e-6)
    assert embedding.packed is None and embedding.weight is dense
