"""Fused prompt attention for full-history visits (lane L6): bits against RC6's own call.

RC6's prompt read calls ``attention.stock_attention`` on query tiles of ``4M // (24 * N)``
rows; past the vector-kernel threshold every tile holds 8 rows or fewer and runs MLX's
two-pass vector kernel. ``prefill_attend.prompt_attention`` must give that call's result bit
for bit on every row, for every tile shape (including a short last tile with its own block
count), and ``decline`` must refuse every call it cannot serve exactly.
"""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the fused prompt attention needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling.attention import stock_attention  # noqa: E402
from mtplx.models.inkling.fast import prefill_attend as P  # noqa: E402


def _clipped(shape, limit, key):
    v = mx.random.normal(shape, key=key) * 2.2
    n = mx.linalg.norm(v, axis=-1, keepdims=True)
    return (v * mx.minimum(1.0, limit / (n + 1e-6))).astype(mx.bfloat16)


def _chunk(hist, rows, seed):
    keys = hist + rows
    ks = mx.random.split(mx.random.key(seed), 5)
    q = _clipped((1, rows, 24, 128), 22.6, ks[0]).transpose(0, 2, 1, 3)
    k = _clipped((1, 8, keys, 128), 22.6, ks[1])
    v = mx.fast.rms_norm(mx.random.normal((1, 8, keys, 128), key=ks[2]), None, 1e-6).astype(mx.bfloat16)
    prof = _clipped((1, rows, 24, 1024), 8.0, ks[3])
    sink = (mx.random.normal((1, rows, 24), key=ks[4]) * 3.0).astype(mx.bfloat16)
    mx.eval(q, k, v, prof, sink)
    return q, k, v, prof, sink


def test_tiles_follow_rc6s_query_tiling():
    for keys, rows in ((4097, 512), (16385, 512), (32769, 512), (65537, 512), (20041, 40)):
        step = max(1, (4 * 1024 * 1024) // (24 * keys))
        got = P.tiles(rows, keys)
        assert got[0] == (0, min(step, rows), min(step, rows))
        assert sum(n for _, _, n in got) == rows
        assert all(n == step for _, _, n in got[:-1])


def test_vector_blocks_follow_mlx_for_this_family(monkeypatch):
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    tail = str(mx.device_info()["architecture"])[-1:]
    if tail == "s":
        assert [P.vector_blocks(q, n) for q, n in ((1, 70000), (2, 8000), (2, 20000), (5, 40000), (2, 70000))] == \
            [64, 128, 256, 512, 1024]
    elif tail == "d":
        assert [P.vector_blocks(q, n) for q, n in ((1, 70000), (2, 10000), (2, 20000), (2, 70000))] == \
            [128, 128, 512, 1024]
    else:
        assert [P.vector_blocks(q, n) for q, n in ((1, 70000), (2, 70000))] == [32, 64]
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "100")
    assert P.vector_blocks(2, 70000) == 128


def test_declines_what_it_cannot_serve():
    q, k, v, prof, sink = _chunk(1500, 40, seed=1)
    assert P.decline(q, k, v, prof, sink, 1024) == "window"
    assert P.decline(q, k, v, prof, sink, 0) in ("full_kernel_tiles", "single_pass")


@pytest.mark.parametrize("hist,rows", [(20000, 40), (20001, 37), (33000, 21), (20000, 100)])
def test_prompt_attention_equals_rc6_call_bit_for_bit(hist, rows):
    """Tiles of 8 rows (256 blocks); a short last tile; 5-row tiles at 512 blocks with a 1-row
    last tile at 64 blocks (its own reduction); 100 rows: 7 launches, the first 4 evaluated as
    a group (EVAL_LAUNCHES) before the rest are made."""
    q, k, v, prof, sink = _chunk(hist, rows, seed=hist + rows)
    assert P.decline(q, k, v, prof, sink, 0) is None
    ref = stock_attention(q, k, v, prof, sink, 0)
    got = P.prompt_attention(q, k, v, prof, sink, 0)
    mx.eval(ref, got)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert bool(mx.array_equal(ref.view(mx.uint16), got.view(mx.uint16)).item())


def test_prompt_read_takes_the_fused_route_and_keeps_rc6s_bits(monkeypatch):
    """Through the prefill path of `attention.attend` (more than 32 rows, full history)."""
    from mtplx.models.inkling import attention

    q, k, v, prof, sink = _chunk(20000, 40, seed=5)
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_ATTENTION", raising=False)
    before = attention.attention_route_counts().get("prompt_fused.full.selected", 0)
    got = attention.attend(q, k, v, prof, sink, 0)
    assert attention.attention_route_counts().get("prompt_fused.full.selected", 0) == before + 1
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_ATTENTION", "stock")
    ref = attention.attend(q, k, v, prof, sink, 0)
    mx.eval(got, ref)
    assert bool(mx.array_equal(ref.view(mx.uint16), got.view(mx.uint16)).item())
    # panels of 32 rows or fewer never reach the prompt route
    small = attention.attention_route_counts().get("prompt_fused.full.selected", 0)
    attention.attend(q[:, :, :8], k[:, :, :-32], v[:, :, :-32], prof[:, :8], sink[:, :8], 0, verify=True)
    assert attention.attention_route_counts().get("prompt_fused.full.selected", 0) == small


def test_merged_tiles_cover_rc6s_rows_and_keep_its_vector_tail():
    for keys, rows in ((4097, 512), (16385, 512), (1536, 512), (12289, 512), (8193, 37)):
        plan = P.merged_tiles(rows, keys)
        rc6 = P.tiles(rows, keys)
        assert plan is not None and len(plan) < len(rc6)
        assert plan[0][0] == 0 and plan[-1][1] == rows
        assert all(a[1] == b[0] for a, b in zip(plan, plan[1:]))
        if rc6[-1][2] <= 8:  # RC6's vector-kernel tail stays its own call
            assert plan[-1] == (rc6[-1][0], rc6[-1][1])
            assert all(e - s > 8 for s, e in plan[:-1])
        else:
            assert all(e - s > 8 for s, e in plan)
    assert P.merged_tiles(512, 20001) is None      # vector regime: the fused kernel's
    assert P.merged_tiles(100, 101) is None        # one call already


@pytest.mark.parametrize("hist,rows,window", [(3584, 512, 0), (15872, 512, 0), (1023, 512, 1024), (7000, 45, 0)])
def test_merged_calls_equal_rc6s_tiles_bit_for_bit(hist, rows, window):
    q, k, v, prof, sink = _chunk(hist, rows, seed=hist + rows + window)
    assert P.merged_decline(q, k, v, prof, sink, window) is None
    ref = stock_attention(q, k, v, prof, sink, window)
    got = P.merged_attention(q, k, v, prof, sink, window)
    mx.eval(ref, got)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert bool(mx.array_equal(ref.view(mx.uint16), got.view(mx.uint16)).item())


def test_window_prompt_call_takes_the_merged_route(monkeypatch):
    from mtplx.models.inkling import attention

    q, k, v, prof, sink = _chunk(1023, 512, seed=77)
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_ATTENTION", raising=False)
    before = attention.attention_route_counts().get("prompt_merged.window.selected", 0)
    got = attention.attend(q, k, v, prof, sink, 1024)
    assert attention.attention_route_counts().get("prompt_merged.window.selected", 0) == before + 1
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_ATTENTION", "stock")
    ref = attention.attend(q, k, v, prof, sink, 1024)
    mx.eval(got, ref)
    assert bool(mx.array_equal(ref.view(mx.uint16), got.view(mx.uint16)).item())
    # the merged calls alone can be switched off; the call then runs RC6's own tiles
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_ATTENTION", raising=False)
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_MERGED", "off")
    still = attention.attention_route_counts().get("prompt_merged.window.selected", 0)
    off = attention.attend(q, k, v, prof, sink, 1024)
    mx.eval(off)
    assert attention.attention_route_counts().get("prompt_merged.window.selected", 0) == still
    assert bool(mx.array_equal(ref.view(mx.uint16), off.view(mx.uint16)).item())
