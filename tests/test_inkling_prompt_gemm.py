"""The prompt chunk's projections from one input as one launch (fast.prompt_gemm.project): whatever the shape,
its outputs have the bits of MLX's own products on every call, not only the first (admitted per shape by a
comparison on order-revealing operands and on the first call's own, else MLX's products serve); at the wide
chunks every Inkling projection shape is served by our launch."""

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


def test_the_ordered_operands_reveal_a_split_sum():
    # A sum over the depth in two halves (MLX's split products) against the whole: the order-revealing
    # operands make them differ on most outputs, where random operands rounded to bfloat16 mostly agree.
    x, (w,) = PG._ordered_operands(64, 2304, [512], mx.bfloat16)
    whole = (x.astype(mx.float32) @ w.astype(mx.float32).T)
    halves = (x[:, :1152].astype(mx.float32) @ w[:, :1152].astype(mx.float32).T
              + x[:, 1152:].astype(mx.float32) @ w[:, 1152:].astype(mx.float32).T)
    differ = mx.mean((whole.astype(mx.bfloat16) != halves.astype(mx.bfloat16)).astype(mx.float32)).item()
    assert differ > 0.2, differ
    # and every value is a small integer, exact in bfloat16
    assert bool(mx.all(mx.abs(whole) < 256).item())


@pytest.mark.parametrize("rows,shape", [(45, (1024, 2304)), (45, (3072, 2304)), (512, (2304, 3072)),
                                        (2048, (1024, 2304)), (512, (64, 4608))])
def test_an_admitted_shape_keeps_mlx_bits_on_later_calls(monkeypatch, rows, shape):
    # The defect this guards: a 45-row key projection matched MLX on its first call by chance, was admitted,
    # and differed on most later calls.
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    key = mx.random.key(rows * 7 + shape[0])
    w = (mx.random.normal(shape, key=key) * 0.02).astype(mx.bfloat16)
    for draw in range(12):
        x = mx.random.normal((1, rows, shape[1]), key=mx.random.key(1000 + draw)).astype(mx.bfloat16) * 3
        (got,) = PG.project(x, [w], f"later_{rows}_{shape[0]}")
        want = x @ w.T
        mx.eval(got, want)
        assert same(got, want), (rows, shape, draw)
