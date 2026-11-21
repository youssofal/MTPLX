"""Our prompt-width pair sort and expert products (fast.prompt_experts) against MLX's argsort and sorted
products: the sort must be a grouping of the pairs by expert with consistent rows, experts, inverse and
run starts; every product row must have MLX's sorted product's bits, in both expert formats, whatever
the operand type, including subnormal and extreme codes and scales."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the prompt expert kernels need a Metal GPU", allow_module_level=True)

from mtplx.models.inkling.fast import prompt_experts as PE  # noqa: E402

if not PE.available()[0]:  # pragma: no cover - GPUs without tensor units
    pytest.skip("the prompt expert products need tensor units", allow_module_level=True)

EXPERTS = 128


class Packed:
    def __init__(self, codes, scales, mode):
        self.codes, self.scales, self.mode = codes.view(mx.uint32), scales, mode


def weights(mode, n_out, depth, seed):
    """Every code value appears, and the scales reach their subnormal (E4M3) and smallest (E8M0) codes."""
    k1, k2 = mx.random.split(mx.random.key(seed))
    if mode == "nvfp4":
        codes = mx.random.randint(0, 256, (EXPERTS, n_out, depth // 2), key=k1).astype(mx.uint8)
        scales = mx.random.randint(1, 0x50, (EXPERTS, n_out, depth // 16), key=k2).astype(mx.uint8)
    else:
        codes = mx.random.randint(0, 256, (EXPERTS, n_out, depth), key=k1).astype(mx.uint8)
        scales = mx.random.randint(100, 135, (EXPERTS, n_out, depth // 32), key=k2).astype(mx.uint8)
    return Packed(codes, scales, mode)


def routes(rows, seed):
    k1, k2 = mx.random.split(mx.random.key(seed))
    logits = mx.random.normal((rows, EXPERTS), key=k1) + 2.0 * mx.random.normal((1, EXPERTS), key=k2)
    return mx.argpartition(-logits, kth=7, axis=-1)[:, :8].astype(mx.uint32).reshape(-1)


def inputs(rows, depth, mode, seed):
    from mtplx.models.inkling.activations import expert_inputs

    x = (mx.random.normal((rows, depth), key=mx.random.key(seed)) * 0.7).astype(mx.bfloat16)
    return expert_inputs(x, mode)[0].reshape(rows, depth)


def same(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)).item())


@pytest.mark.parametrize("rows", [40, 512, 4096])
def test_the_pair_sort_groups_every_pair_by_expert(rows):
    experts = routes(rows, seed=rows)
    order, srows, sexp, inverse, tiles, starts = PE.sort_pairs(experts, 8, EXPERTS)
    pairs = experts.size
    assert bool(mx.array_equal(mx.sort(order), mx.arange(pairs, dtype=mx.uint32)).item())
    assert bool(mx.all(sexp[1:] >= sexp[:-1]).item())
    assert bool(mx.array_equal(sexp, experts[order]).item())
    assert bool(mx.array_equal(srows, order // 8).item())
    assert bool(mx.array_equal(inverse[order], mx.arange(pairs, dtype=mx.uint32)).item())
    sizes = mx.zeros((EXPERTS,), mx.int32).at[experts.astype(mx.int32)].add(1)
    assert bool(mx.array_equal(starts[1:] - starts[:-1], sizes).item()) and starts[0].item() == 0
    assert bool(mx.array_equal(tiles[1:] - tiles[:-1], (sizes + PE.BM - 1) // PE.BM).item())


@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("rows", [64, 520])
def test_the_products_give_mlx_sorted_product_bits(mode, rows):
    bits, group = (4, 16) if mode == "nvfp4" else (8, 32)
    experts = routes(rows, seed=rows + 1)
    order, srows, sexp, inverse, tiles, starts = PE.sort_pairs(experts, 8, EXPERTS)
    pairs = experts.size
    gate_up = weights(mode, 2304, 2304, seed=1)
    down = weights(mode, 2304, 1152, seed=2)
    values = inputs(rows, 2304, mode, seed=3)
    mid = inputs(pairs, 1152, mode, seed=4)
    want13 = mx.gather_qmm(values[srows].reshape(-1, 1, 2304), gate_up.codes, scales=gate_up.scales,
                           rhs_indices=sexp, group_size=group, bits=bits, mode=mode, sorted_indices=True)
    want2 = mx.gather_qmm(mid.reshape(-1, 1, 1152), down.codes, scales=down.scales, rhs_indices=sexp,
                          group_size=group, bits=bits, mode=mode, sorted_indices=True)
    for operand in (mx.bfloat16, mx.float32):
        got13 = PE.product(values.astype(operand), gate_up, tiles, starts, pairs, rowmap=srows)
        got2 = PE.product(mid.astype(operand), down, tiles, starts, pairs)
        mx.eval(want13, want2, got13, got2)
        assert same(got13, want13.reshape(pairs, 2304)), (mode, rows, operand)
        assert same(got2, want2.reshape(pairs, 2304)), (mode, rows, operand)


@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("rows", [64, 520, 2100])
@pytest.mark.parametrize("wide", [(2, 2), (4, 2), (2, 4), (4, 4), (2, 2, 64), (1, 4), (1, 2), (1, 2, 64), (4, 2, 64)])
def test_the_wide_tilings_give_the_same_bits(mode, rows, wide):
    """More rows and columns a threadgroup, a double-buffered decode 32 or 64 deep a step, and 32-row tiles
    change no sum: every product row has MLX's sorted product's float32 bits, the sort's tiles cut at 32 SGM rows."""
    bits, group = (4, 16) if mode == "nvfp4" else (8, 32)
    experts = routes(rows, seed=rows + 7)
    order, srows, sexp, inverse, tiles, starts = PE.sort_pairs(experts, 8, EXPERTS, 32 * wide[0])
    pairs = experts.size
    sizes = mx.zeros((EXPERTS,), mx.int32).at[experts.astype(mx.int32)].add(1)
    assert bool(mx.array_equal(tiles[1:] - tiles[:-1], (sizes + 32 * wide[0] - 1) // (32 * wide[0])).item())
    gate_up = weights(mode, 2304, 2304, seed=5)
    down = weights(mode, 2304, 1152, seed=6)
    values = inputs(rows, 2304, mode, seed=7)
    mid = inputs(pairs, 1152, mode, seed=8)
    want13 = mx.gather_qmm(values[srows].reshape(-1, 1, 2304), gate_up.codes, scales=gate_up.scales,
                           rhs_indices=sexp, group_size=group, bits=bits, mode=mode, sorted_indices=True)
    want2 = mx.gather_qmm(mid.reshape(-1, 1, 1152), down.codes, scales=down.scales, rhs_indices=sexp,
                          group_size=group, bits=bits, mode=mode, sorted_indices=True)
    for operand in (mx.bfloat16, mx.float32):
        got13 = PE.product(values.astype(operand), gate_up, tiles, starts, pairs, rowmap=srows, wide=wide)
        got2 = PE.product(mid.astype(operand), down, tiles, starts, pairs, wide=wide)
        mx.eval(want13, want2, got13, got2)
        assert same(got13, want13.reshape(pairs, 2304).astype(mx.float32)), (mode, rows, wide, operand)
        assert same(got2, want2.reshape(pairs, 2304).astype(mx.float32)), (mode, rows, wide, operand)


def test_the_tiling_follows_rows_an_expert(monkeypatch):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_EXPERT_TILES", raising=False)
    for floor, shape in PE.WIDE_PLAN:
        want = (PE.BM, None) if shape is None else (32 * shape[0], shape)
        assert PE.tiling(max(1, int(floor * EXPERTS)), EXPERTS) == want
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_EXPERT_TILES", "narrow")
    assert PE.tiling(4096 * 8, EXPERTS) == (PE.BM, None)
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_EXPERT_TILES", "2x4")
    assert PE.tiling(40 * 8, EXPERTS) == (64, (2, 4))
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_EXPERT_TILES", "1x2d64")
    assert PE.tiling(40 * 8, EXPERTS) == (32, (1, 2, 64))
