"""The prompt chunk's output gate as one launch (fast.prompt_glue.output_gate) against model.Layer.attention's own
statements, bit for bit: attend's output read heads first in place, the gate's sigmoid and product in float32,
one rounding to the stream's type; contiguous and strided attention outputs, both stream types."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the gate kernel needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling.fast import prompt_glue as G  # noqa: E402


def statements(heads_first, gate_lin):
    rows = heads_first.shape[2]
    h = heads_first.transpose(0, 2, 1, 3).reshape(1, rows, -1)
    gate = mx.sigmoid(gate_lin.astype(mx.float32))
    return (h.astype(mx.float32) * gate).astype(h.dtype)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows", [33, 700, 2048])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_the_gate_launch_gives_the_layers_bits(dtype, rows, layout):
    k1, k2 = mx.random.split(mx.random.key(rows))
    gate_lin = (mx.random.normal((1, rows, 3072), key=k1) * 4.0).astype(dtype)
    if layout == "contiguous":
        heads_first = (mx.random.normal((1, 24, rows, 128), key=k2) * 2.0).astype(dtype)
    else:  # heads first as a transposed view of a rows-first array, as merged attention calls can leave it
        heads_first = (mx.random.normal((1, rows, 24, 128), key=k2) * 2.0).astype(dtype).transpose(0, 2, 1, 3)
    # extreme gates in the first columns of every row: saturated, zero, negative zero, tiny
    extremes = mx.array([88.0, -88.0, 0.0, -0.0, 1e-6, -1e-6, 30.0, -30.0], dtype=mx.float32).astype(dtype)
    gate_lin = mx.concatenate([mx.broadcast_to(extremes.reshape(1, 1, 8), (1, rows, 8)), gate_lin[:, :, 8:]], axis=2)
    want = statements(heads_first, gate_lin)
    got = G.output_gate(heads_first, gate_lin)
    mx.eval(want, got)
    assert got is not None and same(got, want), (str(dtype), rows, layout)


def test_short_and_switched_off_calls_keep_the_statements(monkeypatch):
    h = mx.zeros((1, 24, 16, 128), mx.bfloat16)
    assert G.output_gate(h, mx.zeros((1, 16, 3072), mx.bfloat16)) is None
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_KERNELS", "mlx")
    h = mx.zeros((1, 24, 64, 128), mx.bfloat16)
    assert G.output_gate(h, mx.zeros((1, 64, 3072), mx.bfloat16)) is None
    assert G.counts().get("gate.mlx.disabled", 0) >= 1
