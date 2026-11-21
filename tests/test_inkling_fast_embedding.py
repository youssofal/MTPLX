"""The one-launch token embedding (fast/embedding.py) against the model's own embedding.

Device tests: tables with the packs' row width, stored as 4- and 8-bit affine codes or kept 16-bit, on
a bf16 and a float16 stream, with a gain that is not all ones. Every value must equal
``model.Embedding`` (gather, decode, norm), and the launch must refuse what it does not serve.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.models.inkling.precision import AffineWeight  # noqa: E402

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")
COLS, VOCAB = 2304, 4096


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


def _table(kind, dtype):
    w = (mx.random.normal((VOCAB, COLS), key=mx.random.key(3)) * 0.05).astype(dtype)
    if kind == "16-bit":
        return w
    bits = 4 if kind == "4-bit" else 8
    return AffineWeight(*mx.quantize(w, group_size=64, bits=bits), bits, 64)


@needs_gpu
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
@pytest.mark.parametrize("kind", ["4-bit", "8-bit", "16-bit"])
def test_every_value_equals_the_models_embedding(kind, dtype):
    from mtplx.models.inkling.fast.embedding import embed_rows
    from mtplx.models.inkling.model import Embedding

    table = _table(kind, dtype)
    gain = (1 + 0.3 * mx.random.normal((COLS,), key=mx.random.key(4))).astype(dtype)
    model = Embedding(table, gain, 1e-6)
    for ids in ([[0]], [[VOCAB - 1]], [[5, 4095, 77, 1024, 3, 3, 9, 2048]]):
        ids = mx.array(ids, mx.int32)
        got, want = embed_rows(ids, table, gain, 1e-6), model(ids)
        mx.eval(got, want)
        assert _same(got, want), (kind, dtype, ids.tolist())


def test_tables_the_launch_does_not_serve_are_named():
    from mtplx.models.inkling.fast.embedding import reason

    half = mx.zeros((8, COLS), mx.float16)
    assert reason(half, mx.zeros((COLS,), mx.float16)) is None
    assert reason(half, mx.zeros((COLS,), mx.bfloat16)) == "gain"           # the gain's type must be the table's
    assert reason(mx.zeros((8, COLS), mx.float32), mx.zeros((COLS,), mx.float32)) == "table_dtype"
    six = SimpleNamespace(bits=6, group_size=64, codes=mx.zeros((8, COLS * 6 // 32), mx.uint32),
                          scales=mx.zeros((8, COLS // 64), mx.float16), biases=mx.zeros((8, COLS // 64), mx.float16))
    stored = AffineWeight(six.codes, six.scales, six.biases, 6, 64)
    assert reason(stored, mx.zeros((COLS,), mx.float16)) == "bits_6"
