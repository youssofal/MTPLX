"""The prompt chunk's depth reads and input norms as one launch (fast.prompt_glue.input_norm) against the
prompt path's MLX chain (model.read_blocks, then the norm and the cast): bit for bit at chunk widths, one to
four closed blocks, a bfloat16 or float32 partial, and the first layer's read-free norm; decode and verify
widths, a fifth closed block, a float16 stream and the switch keep MLX's chain, counted."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the depth-read kernel needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import model as M  # noqa: E402
from mtplx.models.inkling.fast import prompt_glue as G  # noqa: E402


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def case(rows, blocks, dtype, seed):
    ks = mx.random.split(mx.random.key(seed), 8)
    closed = [(mx.random.normal((1, rows, 2304), key=ks[i]) * (1 + i)).astype(mx.bfloat16) for i in range(blocks)]
    partial = (mx.random.normal((1, rows, 2304), key=ks[5]) * 2).astype(dtype)
    query = (mx.random.normal((1, 2304), key=ks[6]) * 0.05).astype(mx.float32)
    gain = (1 + 0.1 * mx.random.normal((2304,), key=ks[7])).astype(mx.bfloat16)
    return closed, partial, query, gain


@pytest.mark.parametrize("rows", [33, 700, 4096])
@pytest.mark.parametrize("blocks", [1, 4])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_one_launch_read_and_norm_give_the_mlx_chains_bits(monkeypatch, rows, blocks, dtype):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    closed, partial, query, gain = case(rows, blocks, dtype, seed=rows + blocks)
    want = M.norm(M.read_blocks(closed, partial, query, 1e-6, True)[..., 0, :], gain, 1e-6).astype(mx.bfloat16)
    got = G.input_norm(partial, closed, query, gain, 1e-6, "test")
    mx.eval(want, got)
    assert same(got, want)


def test_the_first_layer_norms_its_input_without_a_read(monkeypatch):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    _, partial, _, gain = case(512, 1, mx.bfloat16, seed=3)
    want = M.norm(partial, gain, 1e-6).astype(mx.bfloat16)
    got = G.input_norm(partial, [], None, gain, 1e-6, "test", stand_in=mx.zeros((2304,), mx.float32))
    mx.eval(want, got)
    assert same(got, want)


def test_what_it_does_not_serve_keeps_the_mlx_chain(monkeypatch):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    closed, partial, query, gain = case(512, 4, mx.bfloat16, seed=5)
    before = G.counts()
    assert G.input_norm(partial[:, :32], [c[:, :32] for c in closed], query, gain, 1e-6, "t") is None
    assert G.input_norm(partial, closed + closed[:1], query, gain, 1e-6, "t") is None
    assert G.input_norm(partial, closed, query, gain.astype(mx.float16), 1e-6, "t") is None
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_KERNELS", "mlx")
    assert G.input_norm(partial, closed, query, gain, 1e-6, "t") is None
    after = G.counts()
    for reason in ("rows", "closed_blocks", "stream_dtype", "disabled"):
        assert after.get(f"t.mlx.{reason}", 0) == before.get(f"t.mlx.{reason}", 0) + 1
