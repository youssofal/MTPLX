"""The prompt chunk's projections from one input as one launch (fast.prompt_gemm.project): whatever the shape,
its outputs have the bits of MLX's own products (admitted per shape by a first-use comparison, else MLX's
products serve); at the wide chunks every Inkling projection shape is served by our launch."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the projection launch needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling.fast import prompt_gemm as PG  # noqa: E402

if not PG.available()[0]:  # pragma: no cover - GPUs without tensor units
    pytest.skip("the projection launch needs tensor units", allow_module_level=True)

SHAPES = {"q": (3072, 2304), "r": (384, 2304), "sink": (24, 2304), "gate": (3072, 2304), "k": (1024, 2304),
          "v": (1024, 2304)}


def same(a, b):
    word = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(word), b.view(word)).item())


def matrices(seed, dtype=mx.bfloat16):
    keys = mx.random.split(mx.random.key(seed), len(SHAPES))
    return [(mx.random.normal(shape, key=k) * 0.02).astype(dtype) for shape, k in zip(SHAPES.values(), keys)]


@pytest.mark.parametrize("rows", [40, 512, 2048, 4096])
def test_projections_have_mlx_bits_and_wide_chunks_take_our_launch(monkeypatch, rows):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    ws = matrices(rows)
    x = mx.random.normal((1, rows, 2304), key=mx.random.key(rows + 1)).astype(mx.bfloat16)
    before = PG.counts().get("t.ours", 0)
    got = PG.project(x, ws, "t")
    want = [x @ w.T for w in ws]
    mx.eval(got, want)
    for name, a, b in zip(SHAPES, got, want):
        assert same(a, b), (rows, name)
    if rows >= 2048:
        assert PG.counts().get("t.ours", 0) == before + 1


def test_the_router_shape_in_float32_at_a_wide_chunk(monkeypatch):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    x = mx.random.normal((4096, 2304), key=mx.random.key(5))
    w = mx.random.normal((128, 2304), key=mx.random.key(6)) * 0.02
    before = PG.counts().get("router_t.ours", 0)
    (got,) = PG.project(x, [w], "router_t")
    want = x @ w.T
    mx.eval(got, want)
    assert same(got, want)
    assert PG.counts().get("router_t.ours", 0) == before + 1


def test_decode_widths_and_the_switch_keep_mlx_products(monkeypatch):
    ws = matrices(9)[:1]
    x = mx.random.normal((8, 2304), key=mx.random.key(10)).astype(mx.bfloat16)
    before = PG.counts()
    PG.project(x, ws, "u")
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_KERNELS", "mlx")
    PG.project(mx.zeros((512, 2304), mx.bfloat16), ws, "u")
    after = PG.counts()
    assert after.get("u.mlx.rows", 0) == before.get("u.mlx.rows", 0) + 1
    assert after.get("u.mlx.disabled", 0) == before.get("u.mlx.disabled", 0) + 1
