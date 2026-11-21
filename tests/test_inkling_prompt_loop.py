"""The loop passes' controllers and input norms at a prompt chunk as our launches (fast.prompt_loop) against
model.Controller and model.norm, bit for bit, at chunk widths, for a shared and a per-source input."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the loop kernels need a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import model as M  # noqa: E402
from mtplx.models.inkling.fast import prompt_loop as L  # noqa: E402


def same(a, b):
    word = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(word), b.view(word)).item())


def controller(seed):
    ks = mx.random.split(mx.random.key(seed), 3)
    weights = {"c.down_weight": (mx.random.normal((64, 4608), key=ks[0]) * 0.02).astype(mx.bfloat16),
               "c.forget_weight": (mx.random.normal((2304, 64), key=ks[1]) * 0.2).astype(mx.bfloat16),
               "c.gain_weight": (mx.random.normal((2304, 64), key=ks[2]) * 0.2).astype(mx.bfloat16)}
    return M.Controller(weights, "c")


@pytest.mark.parametrize("rows", [40, 700, 2048])
@pytest.mark.parametrize("sources", [1, 3])
def test_a_controller_and_the_norms_have_the_layers_bits(monkeypatch, rows, sources):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    ks = mx.random.split(mx.random.key(rows * 3 + sources), 4)
    x = mx.random.normal((rows, sources, 2304), key=ks[0]) * 2
    message = mx.random.normal((rows, 4, 2304), key=ks[1]) * 0.7
    gains = [(1 + 0.1 * mx.random.normal((2304,), key=k)).astype(mx.bfloat16) for k in mx.random.split(ks[2], 3)]
    ctrls = [controller(rows + i) for i in range(3)]
    got, want = [], []
    for i in range(3):
        src = 0 if sources == 1 else i
        want.append(ctrls[i](x[:, src], message[:, i], 1e-6))
        got.append(L.controller(ctrls[i], x, src, message, i, 1e-6))
    normed = L.input_norms(got, gains, 1e-6, mx.bfloat16)
    wanted = [M.norm(h, g, 1e-6).astype(mx.bfloat16) for h, g in zip(want, gains)]
    mx.eval(got, want, normed, wanted)
    for i in range(3):
        assert same(got[i], want[i]), ("controller", i)
        assert same(normed[i], wanted[i]), ("norm", i)


def layer_like(sliding, dtype, seed):
    """The attributes of model.Layer that the loop glue reads, with real shapes."""
    ks = iter(mx.random.split(mx.random.key(seed), 24))
    gain = lambda: (1 + 0.1 * mx.random.normal((2304,), key=next(ks))).astype(dtype)

    def ctrl(i):
        w = {"c.down_weight": (mx.random.normal((64, 4608), key=next(ks)) * 0.02).astype(dtype),
             "c.forget_weight": (mx.random.normal((2304, 64), key=next(ks)) * 0.2).astype(dtype),
             "c.gain_weight": (mx.random.normal((2304, 64), key=next(ks)) * 0.2).astype(dtype)}
        return M.Controller(w, "c")

    return SimpleNamespace(
        sliding=sliding, eps=1e-6, cfg={"attn_res_normalize_values": True},
        attn_query=mx.random.normal((3 if not sliding else 1, 2304), key=next(ks)).reshape(-1) * 0.05,
        mlp_query=mx.random.normal((2304,), key=next(ks)) * 0.05,
        qkv_norm=[gain()] if sliding else [gain(), gain(), gain()], ffn_norm=gain(),
        controllers=[ctrl(i) for i in range(4)])


def stream(rows, dtype, seed, blocks=2):
    ks = mx.random.split(mx.random.key(seed), blocks + 2)
    x = (mx.random.normal((1, rows, 2304), key=ks[0]) * 2).astype(dtype)
    closed = [(mx.random.normal((1, rows, 2304), key=k) * 2).astype(dtype) for k in ks[1:blocks + 1]]
    message = mx.random.normal((1, rows, 4, 2304), key=ks[-1]) * 0.7
    return x, closed, message


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("sliding", [True, False])
@pytest.mark.parametrize("rows", [40, 1100])
def test_a_loop_visits_inputs_have_the_layers_bits(monkeypatch, dtype, sliding, rows):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    layer = layer_like(sliding, dtype, rows + sliding)
    x, closed, message = stream(rows, dtype, rows * 2 + sliding)
    # model.Layer.__call__'s statements on passes two and three
    read = M.read_blocks(closed, x, layer.attn_query, layer.eps, True)
    sources = [read[..., 0 if sliding else i, :] for i in range(3)]
    sources = [layer.controllers[i](h, message[..., i, :], layer.eps) for i, h in enumerate(sources)]
    want = [M.norm(h, layer.qkv_norm[0 if sliding else i], layer.eps).astype(dtype) for i, h in enumerate(sources)]
    got = L.visit_inputs(layer, x, closed, message)
    assert got is not None
    mx.eval(got, want)
    for i in range(3):
        assert same(got[i], want[i]), ("source", i)
    # the expert input: the MLP read, the MLP controller, the norm
    residual = (mx.random.normal((1, rows, 2304), key=mx.random.key(rows + 77)) * 2).astype(mx.float32)
    ffn = M.read_blocks(closed, residual, layer.mlp_query, layer.eps, True)[..., 0, :]
    ffn = layer.controllers[3](ffn, message[..., 3, :], layer.eps)
    want_in = M.norm(ffn, layer.ffn_norm, layer.eps).astype(dtype)
    got_in = L.expert_input(layer, residual, closed, message, dtype)
    assert got_in is not None
    mx.eval(got_in, want_in)
    assert same(got_in, want_in)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_a_global_visits_first_pass_norms_have_the_layers_bits(monkeypatch, dtype):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    layer = layer_like(False, dtype, 5)
    x, closed, _ = stream(600, dtype, 6)
    read = M.read_blocks(closed, x, layer.attn_query, layer.eps, True)
    want = [M.norm(read[..., i, :], layer.qkv_norm[i], layer.eps).astype(dtype) for i in range(3)]
    got = L.global_norms(layer, read, dtype)
    assert got is not None
    mx.eval(got, want)
    for i in range(3):
        assert same(got[i], want[i]), i


def test_short_panels_and_the_switch_keep_the_layers_statements(monkeypatch):
    layer = layer_like(True, mx.bfloat16, 9)
    x, closed, message = stream(8, mx.bfloat16, 10)
    assert L.visit_inputs(layer, x, closed, message) is None
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_KERNELS", "mlx")
    x, closed, message = stream(64, mx.bfloat16, 11)
    assert L.visit_inputs(layer, x, closed, message) is None
    counts = L.counts()
    assert counts.get("attention_sources.mlx.rows", 0) >= 1 and counts.get("attention_sources.mlx.disabled", 0) >= 1
