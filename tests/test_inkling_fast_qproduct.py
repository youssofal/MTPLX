"""The 4-bit family's row-stable product (fast/qproduct.py) against MLX's one-row product.

Device tests (Metal GPU only), synthetic matrices with the shapes of the packs' output projection
(2,304 x 3,072: MLX's one-row kernel takes 16 values a lane at 4 bits) and of a head-like matrix
with K = 2,304 (8 values a lane): every row of a 2- to 8-row panel must carry the bits of
``mx.quantized_matmul`` on that row alone, for bf16 and float32 inputs, through the launch and
through the device function another lane's kernel calls.
"""

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")


class _Affine:
    def __init__(self, rows, cols, bits, group, seed, dtype="bfloat16"):
        w = mx.random.normal((rows, cols), key=mx.random.key(seed)).astype(getattr(mx, dtype)) * 0.02
        self.codes, self.scales, self.biases = mx.quantize(w, group_size=group, bits=bits)
        self.bits, self.group_size = bits, group

    def one_row(self, x):
        return mx.quantized_matmul(x, self.codes, self.scales, self.biases, transpose=True,
                                   group_size=self.group_size, bits=self.bits)


def _bits(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


SHAPES = [(2304, 3072, 4, 64), (2304, 3072, 8, 64), (8200, 2304, 4, 64), (8192, 2304, 8, 32)]


@needs
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_every_panel_row_equals_the_one_row_product(shape, dtype):
    from mtplx.models.inkling.fast.qproduct import affine_product

    rows, cols, bits, group = shape
    m = _Affine(rows, cols, bits, group, seed=rows + bits)
    for width in (2, 3, 4, 7, 8):
        x = mx.random.normal((1, width, cols), key=mx.random.key(width)).astype(mx.bfloat16)
        x = x.astype(getattr(mx, dtype))
        got = affine_product(x, m)
        assert got.dtype == x.dtype and got.shape == (1, width, rows)
        for r in range(width):
            assert _bits(got[:, r:r + 1], m.one_row(x[:, r:r + 1])), (width, r)


@needs
@pytest.mark.parametrize("tm", [4, 2])
def test_device_function_gives_the_same_bits(tm):
    from mtplx.models.inkling.fast.qproduct import qmv_constants, qmv_rows_probe

    m = _Affine(2304, 3072, 4, 64, seed=11)
    assert qmv_constants(m) == {"K": 3072, "BITS": 4, "GROUP": 64, "VPT": 16}
    x = mx.random.normal((5, 3072), key=mx.random.key(3)).astype(mx.bfloat16)
    got = qmv_rows_probe(x, m, rows_each=tm)
    for r in range(5):
        assert _bits(got[r:r + 1], m.one_row(x[r:r + 1])), r


@needs
def test_a_form_it_does_not_twin_is_refused_and_counted():
    from mtplx.models.inkling.fast.attn_input import reset_route_counts, route_counts
    from mtplx.models.inkling.fast.qproduct import affine_product

    m = _Affine(256, 3072, 6, 64, seed=5)
    reset_route_counts()
    with pytest.raises(RuntimeError):
        affine_product(mx.zeros((1, 3, 3072), mx.bfloat16), m)
    assert any(k.startswith("refused:qproduct_") for k in route_counts())


@needs
def test_one_row_is_mlx_own_product():
    from mtplx.models.inkling.fast.qproduct import affine_product

    m = _Affine(2304, 3072, 4, 64, seed=2)
    x = mx.random.normal((1, 1, 3072), key=mx.random.key(9)).astype(mx.bfloat16)
    assert _bits(affine_product(x, m), m.one_row(x))


@needs
@pytest.mark.parametrize("width", [1, 3, 4, 8])
@pytest.mark.parametrize("cap", [0.0, 30.0])
def test_head_logits_in_one_launch_equal_the_sliced_capped_product(width, cap):
    """affine_logits: the product, the vocabulary slice and the soft cap in one launch, each value as
    MLX's one-row product sliced and capped by MLX's own divide, tanh and multiply."""
    from mtplx.models.inkling.fast.qproduct import affine_logits

    m = _Affine(8200, 2304, 4, 64, seed=5)
    vocab = 8137
    x = mx.random.normal((1, width, 2304), key=mx.random.key(40 + width)).astype(mx.bfloat16).astype(mx.float32)
    got = affine_logits(x, m, width=vocab, cap=cap)
    assert got.shape == (1, width, vocab) and got.dtype == mx.float32
    for r in range(width):
        want = m.one_row(x[:, r:r + 1])[..., :vocab]
        want = mx.tanh(want / cap) * cap if cap else want
        assert _bits(got[:, r:r + 1], want), (width, r)


@needs
@pytest.mark.parametrize("shape", [(2304, 3072, 4, 64), (8200, 2304, 4, 64), (8192, 2304, 8, 32)])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_fp16_packs_every_panel_row_equals_the_one_row_product(shape, dtype):
    """The FP16 sibling packs: float16 scales and biases, float16 stream (the head takes float32)."""
    from mtplx.models.inkling.fast.qproduct import affine_product

    rows, cols, bits, group = shape
    m = _Affine(rows, cols, bits, group, seed=rows + bits + 1, dtype="float16")
    assert m.scales.dtype == mx.float16
    for width in (1, 2, 4, 8):
        x = mx.random.normal((1, width, cols), key=mx.random.key(width + 9)).astype(mx.float16)
        x = x.astype(getattr(mx, dtype))
        got = affine_product(x, m)
        assert got.dtype == x.dtype and got.shape == (1, width, rows)
        for r in range(width):
            assert _bits(got[:, r:r + 1], m.one_row(x[:, r:r + 1])), (width, r)
