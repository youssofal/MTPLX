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
    latent_down_combine,
    latent_experts_eligible,
    latent_gate_up_swiglu,
    latent_swiglu,
    latent_swiglu_reference,
    masked_qmv,
    masked_qmv_eligible,
    masked_qmv_rows,
    route_topk,
    route_topk_eligible,
    router_logits,
    router_logits_eligible,
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


def _q4(n, k, seed, bits=4):
    """A random affine ``bits``-bit ``[n, k]`` matrix (group 32), ggml-style float16 scales and ``-2**(bits-1)`` biases."""
    rng = np.random.default_rng(seed)
    w = mx.array(rng.integers(0, 2**32, (n, k * bits // 32), dtype=np.uint64).astype(np.uint32))
    s = mx.array(rng.uniform(0.002, 0.02, (n, k // 32)).astype(np.float16).astype(np.float32))
    return w, s, -(2 ** (bits - 1)) * s


# (G, L, H, groups used): 120B-like and 145B-like latent widths (a K block straddles two groups at 4 bits),
# and L below the K block, so one block spans four groups, only an inner one of them used (block 1: groups 4..7)
_GEOMS = [(4, 256, 512, (0, 2)), (2, 1280, 512, (1,)), (2, 1792, 512, (0,)), (8, 128, 512, (0, 5))]


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("geom", _GEOMS)
@pytest.mark.parametrize("xdt", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("sdt", [mx.float32, mx.float16])
def test_masked_qmv_matches_quantized_matmul(xdt, sdt, geom, bits):
    G, L, H, used = geom
    groups = mx.array([[used[j % len(used)] for j in range(10)]], dtype=mx.int32)
    used = set(used)
    w, s, b = _q4(G * L, H, 1, bits)
    x = mx.random.normal((1, H)).astype(xdt)
    assert masked_qmv_eligible(x, w, s, bits, 32, L, "rows")
    want = mx.quantized_matmul(x.astype(mx.float32), w, s, b, transpose=True, group_size=32, bits=bits).reshape(G, L)
    got = masked_qmv(x, w, s.astype(sdt), b.astype(sdt), groups, by="rows", latent=L, group_size=32, out_dtype=mx.float32).reshape(G, L)
    for g in range(G):
        if g in used:
            assert mx.array_equal(got[g], want[g]).item()
        else:
            assert (got[g] == 0).all().item()
    w, s, b = _q4(H, G * L, 2, bits)
    mask = mx.array([1.0 if g in used else 0.0 for g in range(G)])[:, None]
    z = (mx.random.normal((G, L)) * mask).reshape(1, G * L)
    assert masked_qmv_eligible(z, w, s, bits, 32, L, "cols")
    want = mx.quantized_matmul(z, w, s, b, transpose=True, group_size=32, bits=bits)
    got = masked_qmv(z, w, s.astype(sdt), b.astype(sdt), groups, by="cols", latent=L, group_size=32, out_dtype=mx.float32)
    assert mx.array_equal(got, want).item()


def test_masked_qmv_eligibility():
    w, s, _ = _q4(1024, 512, 3)
    assert not masked_qmv_eligible(mx.zeros((2, 512)), w, s, 4, 32, 256, "rows")  # two rows
    assert not masked_qmv_eligible(mx.zeros((1, 512)), w, s, 8, 32, 256, "rows")  # 4-bit words named 8-bit
    w8, s8, _ = _q4(1024, 512, 3, bits=8)
    assert masked_qmv_eligible(mx.zeros((1, 512)), w8, s8, 8, 32, 256, "rows")
    assert not masked_qmv_eligible(mx.zeros((1, 512)), w8, s8, 6, 32, 256, "rows")  # other widths: MLX's path
    assert not masked_qmv_eligible(mx.zeros((1, 512)), w, s, 4, 32, 100, "rows")  # rows not group-aligned
    assert not masked_qmv_eligible(mx.zeros((1, 512)), w, s.astype(mx.bfloat16), 4, 32, 256, "rows")


# --------------------------------------------------------------------------- the block on kernel-eligible shapes


@pytest.fixture(scope="module", params=["q4", "q8"])
def wide(request):
    # H and G*L multiples of 512: the one-token D/U kernels engage; D/U 4-bit (120B) or 8-bit (145B)
    return Synth(seed=5, H=512, L=128, F=64, G=4, E=32, K=6, IS=64, labels=np.arange(32, dtype=np.int32) % 4, dense=request.param)


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


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("by", ["rows", "cols"])
@pytest.mark.parametrize("xdt", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("M", [1, 2, 3, 4, 8])
def test_masked_qmv_rows_is_one_row_per_row(by, xdt, M, bits):
    """Each row of the few-rows kernel is bit-identical to the one-row kernel on that row."""
    G, L, H = 4, 256, 512
    rng = np.random.default_rng(100 + M)
    groups = mx.array(rng.integers(0, G, (M, 10)).astype(np.int32))
    groups = mx.concatenate([groups[:, :1] * 0 + 1, groups[:, 1:] % 2 * 3], axis=1) if M == 3 else groups  # rows using few groups
    if by == "rows":
        w, s, b = _q4(G * L, H, 4, bits)
        x = mx.array(rng.standard_normal((M, H)).astype(np.float32)).astype(xdt)
    else:
        w, s, b = _q4(H, G * L, 5, bits)
        used = np.zeros((M, G, 1), np.float32)
        for m, row in enumerate(np.array(groups)):
            used[m, sorted(set(row.tolist()))] = 1.0
        x = (mx.array(rng.standard_normal((M, G, L)).astype(np.float32)) * mx.array(used)).reshape(M, G * L).astype(xdt)
    for sdt in (mx.float32, mx.float16):
        assert masked_qmv_eligible(x, w, s.astype(sdt), bits, 32, L, by, max_rows=FEW_ROWS_MAX)
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


# --------------------------------------------------------------------------- fused routing and routed experts


@pytest.mark.parametrize("wdt", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("xdt", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("M", [1, 2, 4, 8])
def test_router_logits_are_the_one_row_gemv(wdt, xdt, M):
    """Every row is MLX's one-row float32 GEMV of that row, bit for bit (a bfloat16 weight is read as its float32 values)."""
    rng = np.random.default_rng(300 + M)
    E, H = 128, 1024  # H < 16 E: the GEMV layout the kernel reproduces
    W = mx.array((rng.standard_normal((E, H)) * 0.05).astype(np.float32)).astype(wdt)
    X = mx.array((rng.standard_normal((M, H)) * 3).astype(np.float32)).astype(xdt)
    assert router_logits_eligible(X, W)
    got = router_logits(X, W)
    assert got.dtype == mx.float32 and got.shape == (M, E)
    for m in range(M):
        assert mx.array_equal(got[m : m + 1], moe.float32_logits(X[m : m + 1], W.astype(mx.float32))).item(), m


def test_router_logits_eligibility():
    assert router_logits_eligible(mx.zeros((1, 512)), mx.zeros((64, 512)))
    assert router_logits_eligible(mx.zeros((FEW_ROWS_MAX, 512)), mx.zeros((64, 512), dtype=mx.bfloat16))
    assert not router_logits_eligible(mx.zeros((1, 512)), mx.zeros((32, 512)))  # H >= 16 E: MLX splits H, another order
    assert not router_logits_eligible(mx.zeros((FEW_ROWS_MAX + 1, 512)), mx.zeros((64, 512)))  # prefill: MLX's GEMM
    assert not router_logits_eligible(mx.zeros((1, 448)), mx.zeros((64, 448)))  # H not a multiple of 128
    assert not router_logits_eligible(mx.zeros((1, 512)), mx.zeros((60, 512)))  # E not a multiple of 8


def _route_chain(logits, labels, k):
    p = mx.softmax(logits, axis=-1, precise=True)
    ids, p_sel = moe.top_k_lowest_id(p, k)
    return ids, p_sel / mx.maximum(p_sel.sum(axis=-1, keepdims=True), moe.ROUTER_WEIGHT_FLOOR), labels[ids]


@pytest.mark.parametrize("E", [64, 512])
@pytest.mark.parametrize("M", [1, 3, 8])
def test_route_topk_matches_the_op_chain(E, M):
    """softmax + top-k (ties to the lower id) + renormalisation + labels, row by row as the one-row chain."""
    rng = np.random.default_rng(400 + E + M)
    K = 10
    labels = mx.array(rng.integers(0, 8, E).astype(np.int32))
    assert route_topk_eligible((M, E), labels, K)
    cases = [rng.standard_normal((M, E)) * s for s in (0.3, 3.0, 30.0)]
    cases.append(np.round(rng.standard_normal((M, E)) * 2) / 2)  # many exact ties
    cases.append(np.zeros((M, E)))  # every probability equal
    cases.append(np.repeat(rng.standard_normal((1, E)) * 4, M, axis=0))  # identical rows
    for lg in cases:
        lg = mx.array(lg.astype(np.float32))
        ids, w, g = route_topk(lg, labels, K)
        for m in range(M):
            ri, rw, rg = _route_chain(lg[m : m + 1], labels, K)
            assert mx.array_equal(ids[m : m + 1], ri).item() and mx.array_equal(w[m : m + 1], rw).item()
            assert mx.array_equal(g[m : m + 1], rg).item()
    lg = np.full((1, E), -5.0, np.float32)
    lg[0, [40, 7, 33, 2]] = 3.0
    ids, _, _ = route_topk(mx.array(lg), labels, K)
    assert np.array(ids)[0, :4].tolist() == [2, 7, 33, 40]  # equal logits: lowest id first
    assert not route_topk_eligible((FEW_ROWS_MAX + 1, E), labels, K)
    assert not route_topk_eligible((1, E), labels.astype(mx.uint32), K)


def _nvfp4(rng, n_exp, rows, cols):
    w = mx.array(rng.integers(0, 2**32, (n_exp, rows, cols // 8), dtype=np.uint64).astype(np.uint32))
    s = mx.array(rng.integers(0x28, 0x48, (n_exp, rows, cols // 16), dtype=np.uint8))
    return w, s


# (L, I): a full last K block, a half-lane tail (L % 256 == 128) with an I tail, MLX's fp_qmv_fast (L % 512 == 0)
_EXPERT_GEOMS = [(256, 64), (384, 80), (512, 48)]


@pytest.mark.parametrize("geom", _EXPERT_GEOMS)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32, mx.float16])
@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_fused_experts_match_gather_qmm(geom, dtype, T):
    """gate_up + SwiGLU and down + combine against gather_qmm + latent_swiglu / latent_combine, bit for bit;
    with several rows the rows share experts (each distinct expert's weights stream once)."""
    L, F = geom
    rng = np.random.default_rng(500 + L + T)
    E, G, K = 40, 4, 6
    gu_w, gu_s = _nvfp4(rng, E, 2 * F, L)
    dn_w, dn_s = _nvfp4(rng, E, L, F)
    scal = mx.array(np.stack([rng.uniform(0.02, 0.05, E), rng.uniform(0.02, 0.05, E), rng.uniform(5e-5, 3e-4, E)], 1).astype(np.float32))
    lab = mx.array(rng.integers(0, G, E).astype(np.int32))
    ids = mx.array(np.stack([rng.choice(9 if t % 2 else E, K, replace=False) for t in range(T)]).astype(np.int32))
    groups = lab[ids]
    w = mx.array(rng.uniform(0.01, 0.3, (T, K)).astype(np.float32))
    xl = mx.array((rng.standard_normal((T, G, L)) * 2).astype(np.float32)).astype(dtype)
    assert latent_experts_eligible(xl, gu_w, gu_s, dn_w, dn_s, K)
    lhs = (mx.arange(T, dtype=mx.int32)[:, None] * G + groups).astype(mx.uint32)
    gu = mx.gather_qmm(xl.reshape(T * G, 1, L), gu_w, gu_s, lhs_indices=lhs, rhs_indices=ids, transpose=True, group_size=16, bits=4, mode="nvfp4")
    a_ref = latent_swiglu(gu, ids, scal, F, dtype)
    a, info = latent_gate_up_swiglu(xl, gu_w, gu_s, ids, groups, scal)
    assert (info is None) == (T == 1)
    assert a.dtype == dtype and mx.array_equal(a.reshape(a_ref.shape), a_ref).item()
    dn = mx.gather_qmm(a_ref, dn_w, dn_s, rhs_indices=ids, transpose=True, group_size=16, bits=4, mode="nvfp4")
    z_ref = latent_combine(dn.reshape(T, K, L), ids, scal, w, groups, G)
    z = latent_down_combine(a, dn_w, dn_s, ids, groups, scal, w, info, G)
    assert z.dtype == mx.float32 and mx.array_equal(z, z_ref).item()


def test_fused_experts_eligibility():
    rng = np.random.default_rng(7)
    gu_w, gu_s = _nvfp4(rng, 8, 128, 256)
    dn_w, dn_s = _nvfp4(rng, 8, 256, 64)
    xl = mx.zeros((1, 4, 256), dtype=mx.bfloat16)
    assert latent_experts_eligible(xl, gu_w, gu_s, dn_w, dn_s, 6)
    assert not latent_experts_eligible(mx.zeros((FEW_ROWS_MAX + 1, 4, 256)), gu_w, gu_s, dn_w, dn_s, 6)  # prefill
    assert not latent_experts_eligible(xl.astype(mx.int32), gu_w, gu_s, dn_w, dn_s, 6)
    assert not latent_experts_eligible(xl, gu_w, gu_s.astype(mx.uint32), dn_w, dn_s, 6)
    assert not latent_experts_eligible(mx.zeros((1, 4, 128)), gu_w, gu_s, dn_w, dn_s, 6)  # L disagrees with the weights


@pytest.fixture(scope="module", params=["q4", "q8"])
def fusable(request):
    # H < 16 E (the fused router GEMV), H and G*L multiples of 512 (the masked D/U kernels)
    return Synth(seed=9, H=512, L=256, F=64, G=4, E=64, K=6, IS=64, labels=np.arange(64, dtype=np.int32) % 4, dense=request.param)


def _packed(synth, dtype, bf16_router=False):
    blk = synth.block(latent_dtype=dtype)
    blk.experts.merge_gate_up()
    blk.dense_d.compact_scales()
    blk.dense_u.compact_scales()
    if bf16_router:  # as a loaded pack: a bfloat16 router, held in float32 with the 16-bit copy kept
        blk.gate.weight = blk.gate.weight.astype(mx.bfloat16)
        blk.cast_router_float32()
    return blk


@pytest.mark.parametrize("bf16_router", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("S", [1, 2, 3, 4, 8, 9])
def test_fused_block_is_the_op_chain(fusable, dtype, S, bf16_router, monkeypatch):
    """Decode (one row), verify widths (2..8 rows, rows sharing experts) and prefill widths (which fall back):
    the fused kernels give the op chain's bits, routes and routing weights included."""
    blk = _packed(fusable, dtype, bf16_router)
    X = mx.array(np.random.default_rng(60 + S).standard_normal((1, S, fusable.H)).astype(np.float32)).astype(dtype)
    if S > 2:
        X = mx.concatenate([X[:, : S - 1], X[:, :1]], axis=1)  # the last row repeats the first: all its experts are shared
    monkeypatch.setattr(moe, "_FUSED_DEFAULT", True)
    got = blk(X, return_routing=True)
    monkeypatch.setattr(moe, "_FUSED_DEFAULT", False)
    want = blk(X, return_routing=True)
    assert all(mx.array_equal(a, b).item() for a, b in zip(got, want))
    if not bf16_router:  # (the reference holds the unrounded router)
        res = fusable.reference(np.array(X.astype(mx.float32))[0])
        np.testing.assert_array_equal(np.array(got[1])[0], np.stack([r.ids for r in res]))


def test_fused_block_paths_and_router_shadow(fusable, monkeypatch):
    """Unmerged experts take the op chain; a bfloat16 router kept by cast_router_float32 gives the float32 bits;
    replacing gate.weight drops the stale copy."""
    monkeypatch.setattr(moe, "_FUSED_DEFAULT", True)
    X = mx.array(np.random.default_rng(80).standard_normal((1, 1, fusable.H)).astype(np.float32)).astype(mx.bfloat16)
    plain = fusable.block(latent_dtype=mx.bfloat16)  # not merged: the op-chain experts
    packed = _packed(fusable, mx.bfloat16)
    assert mx.array_equal(plain(X), packed(X)).item()
    blk = _packed(fusable, mx.bfloat16)
    want = blk(X)
    blk.gate.weight = blk.gate.weight.astype(mx.bfloat16)
    blk.cast_router_float32()
    assert blk.gate.weight.dtype == mx.float32 and blk._router_kernel_weight().dtype == mx.bfloat16
    assert "_router_lowp" not in dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(blk.parameters()))
    bf_exact = blk(X)
    blk.gate.weight = blk.gate.weight * 1.0  # a new array: the 16-bit copy no longer shadows it
    assert blk._router_kernel_weight() is blk.gate.weight
    assert mx.array_equal(blk(X), bf_exact).item()
    monkeypatch.setattr(moe, "_FUSED_DEFAULT", False)
    assert mx.array_equal(blk(X), bf_exact).item()
    del want
