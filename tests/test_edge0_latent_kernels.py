"""The Edge0 latent MoE decode kernels (mtplx.kernels.edge0_latent) and the load-time layout changes.

Every kernel replaces an MLX op chain and must reproduce it bit for bit; the
block built on them must match the NumPy reference at the decode tolerance and
give the same bits before and after ``merge_gate_up`` / ``compact_scales``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.kernels.edge0_latent import (
    FEW_ROWS_MAX,
    latent_combine,
    latent_combine_reference,
    latent_swiglu,
    latent_swiglu_reference,
    masked_qmv,
    masked_qmv_eligible,
    masked_qmv_rows,
    router_logits_rows,
    router_rows_eligible,
)
from mtplx.models import edge0_latent_moe as moe
from tests.test_edge0_latent_moe import TF32, TIGHT, Synth, _rel

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("T", [1, 3, 70])
def test_combine_is_bit_identical(dtype, T):
    mx.random.seed(T)
    K, G, L, E = 10, 8, 256, 64
    down = (mx.random.normal((T, K, L)) * 3).astype(dtype)
    ids = mx.random.randint(0, E, (T, K)).astype(mx.int32)
    scalars = mx.random.uniform(0, 1e-3, (E, 3))
    w = mx.random.uniform(shape=(T, K))
    groups = mx.random.randint(0, G, (T, K)).astype(mx.int32)
    a = latent_combine(down, ids, scalars, w, groups, G)
    b = latent_combine_reference(down, ids, scalars, w, groups, G)
    assert a.dtype == mx.float32 and mx.array_equal(a, b).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("lead", [(1, 10, 1), (300, 1)])
def test_swiglu_is_bit_identical(dtype, lead):
    mx.random.seed(len(lead))
    E, I = 64, 96
    rows = int(np.prod(lead[:-1]))
    ids = mx.random.randint(0, E, (rows,)).astype(mx.int32)
    scalars = mx.random.uniform(0.5, 3.0, (E, 3))
    gu = (mx.random.normal((*lead, 2 * I)) * 8).astype(dtype)
    want = latent_swiglu_reference(gu, ids, scalars, I, dtype)
    assert mx.array_equal(latent_swiglu(gu, ids, scalars, I, dtype), want).item()
    split = (mx.contiguous(gu[..., :I]), mx.contiguous(gu[..., I:]))
    assert mx.array_equal(latent_swiglu(split, ids, scalars, I, dtype), want).item()


def _q4(n, k, seed):
    rng = np.random.default_rng(seed)
    w = mx.array(rng.integers(0, 2**32, (n, k // 8), dtype=np.uint64).astype(np.uint32))
    s = mx.array(rng.uniform(0.002, 0.02, (n, k // 32)).astype(np.float16).astype(np.float32))
    return w, s, -8 * s


@pytest.mark.parametrize("xdt", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("sdt", [mx.float32, mx.float16])
def test_masked_qmv_matches_quantized_matmul(xdt, sdt):
    G, L, H = 4, 256, 512
    groups = mx.array([[2, 2, 0, 2, 0, 2, 2, 0, 0, 2]], dtype=mx.int32)
    used = {0, 2}
    w, s, b = _q4(G * L, H, 1)
    x = mx.random.normal((1, H)).astype(xdt)
    assert masked_qmv_eligible(x, w, s, 4, 32, L, "rows")
    want = mx.quantized_matmul(x.astype(mx.float32), w, s, b, transpose=True, group_size=32, bits=4).reshape(G, L)
    got = masked_qmv(x, w, s.astype(sdt), b.astype(sdt), groups, by="rows", latent=L, group_size=32, out_dtype=mx.float32).reshape(G, L)
    for g in range(G):
        if g in used:
            assert mx.array_equal(got[g], want[g]).item()
        else:
            assert (got[g] == 0).all().item()
    w, s, b = _q4(H, G * L, 2)
    mask = mx.array([1.0 if g in used else 0.0 for g in range(G)])[:, None]
    z = (mx.random.normal((G, L)) * mask).reshape(1, G * L)
    want = mx.quantized_matmul(z, w, s, b, transpose=True, group_size=32, bits=4)
    got = masked_qmv(z, w, s.astype(sdt), b.astype(sdt), groups, by="cols", latent=L, group_size=32, out_dtype=mx.float32)
    assert mx.array_equal(got, want).item()


def test_masked_qmv_eligibility():
    w, s, _ = _q4(1024, 512, 3)
    assert not masked_qmv_eligible(mx.zeros((2, 512)), w, s, 4, 32, 256, "rows")  # two rows
    assert not masked_qmv_eligible(mx.zeros((1, 512)), w, s, 8, 32, 256, "rows")  # 8-bit
    assert not masked_qmv_eligible(mx.zeros((1, 512)), w, s, 4, 32, 100, "rows")  # rows not group-aligned
    assert not masked_qmv_eligible(mx.zeros((1, 512)), w, s.astype(mx.bfloat16), 4, 32, 256, "rows")


# --------------------------------------------------------------------------- the block on kernel-eligible shapes


@pytest.fixture(scope="module")
def wide():
    # H and G*L multiples of 512: the one-token D/U kernels engage
    return Synth(seed=5, H=512, L=128, F=64, G=4, E=32, K=6, IS=64, labels=np.arange(32, dtype=np.int32) % 4)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_wide_block_matches_reference(wide, seed):
    blk = wide.block()
    X = np.random.default_rng(20 + seed).standard_normal((1, wide.H)).astype(np.float32)
    y, ids, _ = blk(mx.array(X)[None], return_routing=True)
    res = wide.reference(X)
    np.testing.assert_array_equal(np.array(ids)[0], np.stack([r.ids for r in res]))
    assert _rel(np.array(y)[0], np.stack([r.out for r in res])) < TIGHT  # one row: no TF32 GEMM anywhere


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("S", [1, 4, 40])
def test_layout_changes_keep_the_bits(wide, dtype, S, monkeypatch):
    X = mx.array(np.random.default_rng(30 + S).standard_normal((1, S, wide.H)).astype(np.float32)).astype(dtype)
    plain = wide.block(latent_dtype=dtype)
    packed = wide.block(latent_dtype=dtype)
    packed.experts.merge_gate_up()
    assert packed.experts.merged and "gate_proj" not in packed.experts
    assert packed.dense_d.compact_scales() and packed.dense_u.compact_scales()
    assert packed.dense_d.scales.dtype == mx.float16
    want = plain(X)
    assert mx.array_equal(packed(X), want).item()
    monkeypatch.setattr(moe, "_use_kernels", lambda: False)
    if S == 1 or S > FEW_ROWS_MAX:
        # the MLX-op fallback (no kernels) agrees with the kernels bit for bit
        assert mx.array_equal(plain(X), want).item()
    else:
        # verify widths: the kernels compute each row as one decode row, which MLX's
        # multi-row matmuls do not (they round differently); same routes, float32-close
        got = plain(X)
        tol = TIGHT if dtype == mx.float32 else 1e-2
        assert _rel(np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32), np.float64)) < tol


@pytest.mark.parametrize("by", ["rows", "cols"])
@pytest.mark.parametrize("xdt", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("M", [1, 2, 3, 4, 8])
def test_masked_qmv_rows_is_one_row_per_row(by, xdt, M):
    """Each row of the few-rows kernel is bit-identical to the one-row kernel on that row."""
    G, L, H = 4, 256, 512
    rng = np.random.default_rng(100 + M)
    groups = mx.array(rng.integers(0, G, (M, 10)).astype(np.int32))
    groups = mx.concatenate([groups[:, :1] * 0 + 1, groups[:, 1:] % 2 * 3], axis=1) if M == 3 else groups  # rows using few groups
    if by == "rows":
        w, s, b = _q4(G * L, H, 4)
        x = mx.array(rng.standard_normal((M, H)).astype(np.float32)).astype(xdt)
    else:
        w, s, b = _q4(H, G * L, 5)
        used = np.zeros((M, G, 1), np.float32)
        for m, row in enumerate(np.array(groups)):
            used[m, sorted(set(row.tolist()))] = 1.0
        x = (mx.array(rng.standard_normal((M, G, L)).astype(np.float32)) * mx.array(used)).reshape(M, G * L).astype(xdt)
    for sdt in (mx.float32, mx.float16):
        assert masked_qmv_eligible(x, w, s.astype(sdt), 4, 32, L, by, max_rows=FEW_ROWS_MAX)
        got = masked_qmv_rows(x, w, s.astype(sdt), b.astype(sdt), groups, by=by, latent=L, group_size=32, out_dtype=xdt)
        assert got.shape == (M, w.shape[0]) and got.dtype == xdt
        for m in range(M):
            one = masked_qmv(x[m : m + 1], w, s.astype(sdt), b.astype(sdt), groups[m : m + 1], by=by, latent=L, group_size=32, out_dtype=xdt)
            assert mx.array_equal(got[m : m + 1], one).item(), (by, M, m)


def test_masked_qmv_rows_eligibility():
    w, s, _ = _q4(1024, 512, 3)
    assert masked_qmv_eligible(mx.zeros((4, 512)), w, s, 4, 32, 256, "rows", max_rows=FEW_ROWS_MAX)
    assert not masked_qmv_eligible(mx.zeros((FEW_ROWS_MAX + 1, 512)), w, s, 4, 32, 256, "rows", max_rows=FEW_ROWS_MAX)
    assert not masked_qmv_eligible(mx.zeros((4, 512)), w, s, 4, 32, 256, "rows")  # max_rows defaults to one
    w2, s2, _ = _q4(64 * 8, 512, 3)
    assert not masked_qmv_eligible(mx.zeros((2, 512)), w2, s2, 4, 32, 8, "rows", max_rows=FEW_ROWS_MAX)  # 64 groups > 32-bit mask


@pytest.mark.parametrize("wdt", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("M", [2, 3, 4, 8])
def test_router_logits_rows_are_float32_accurate(wdt, M):
    rng = np.random.default_rng(200 + M)
    E, H = 64, 1024
    W = mx.array((rng.standard_normal((E, H)) * 0.05).astype(np.float32)).astype(wdt)
    X = mx.array(rng.standard_normal((M, H)).astype(np.float32)).astype(mx.bfloat16)
    assert router_rows_eligible(X, W) and not router_rows_eligible(X[:1], W)
    got = np.array(router_logits_rows(X, W))
    want = np.array(X.astype(mx.float32)).astype(np.float64) @ np.array(W.astype(mx.float32)).astype(np.float64).T
    assert got.dtype == np.float32 and _rel(got, want) < 1e-6
    one = np.concatenate([np.array(moe.float32_logits(X[m : m + 1], W)) for m in range(M)])
    assert _rel(got, one.astype(np.float64)) < 1e-6


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("S", [2, 3, 4, 8])
def test_verify_rows_match_decode_rows(wide, dtype, S):
    """A verify-width call computes what one-token decode computes for each row (same routes, float32-close)."""
    blk = wide.block(latent_dtype=dtype)
    blk.experts.merge_gate_up()
    blk.dense_d.compact_scales()
    blk.dense_u.compact_scales()
    X = mx.array(np.random.default_rng(40 + S).standard_normal((1, S, wide.H)).astype(np.float32)).astype(dtype)
    y, ids, w = blk(X, return_routing=True)
    for t in range(S):
        y1, ids1, w1 = blk(X[:, t : t + 1], return_routing=True)
        np.testing.assert_array_equal(np.array(ids)[0, t], np.array(ids1)[0, 0])
        np.testing.assert_allclose(np.array(w)[0, t], np.array(w1)[0, 0], rtol=1e-5, atol=1e-7)  # float32 logits, two summation orders
        tol = TF32 if dtype == mx.float32 else 1e-2  # float32: the shared expert's multi-row GEMM runs as TF32
        assert _rel(np.array(y[0, t].astype(mx.float32)), np.array(y1[0, 0].astype(mx.float32), np.float64)) < tol
    res = wide.reference(np.array(X.astype(mx.float32))[0])
    np.testing.assert_array_equal(np.array(ids)[0], np.stack([r.ids for r in res]))
