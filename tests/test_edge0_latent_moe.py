"""The MLX latent MoE block (mtplx.models.edge0_latent_moe) against the NumPy reference.

Tolerances. The reference runs in float64. With float32 activations the block
is float32 throughout except where MLX on a tensor-unit GPU (M5) runs a
float32 GEMM with TF32 operands: the quantized matmuls and ``nn.Linear`` once
there are more than a handful of rows. So:

- one row (decode) and the routed part of small batches: ~1e-6 relative to the
  output's largest entry (float32 accumulation over 1280-2560 terms, with
  4-bit weights and per-expert scalars spanning 1e-4..1e-2);
- the shared expert (plain ``nn.Linear`` here) at more than one row, and every
  matmul at prefill sizes: TF32, ~1e-3 relative;
- bfloat16 activations: the output itself is rounded to bfloat16 (2**-9).
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.edge0 import latent_convert as lc
from mtplx.edge0 import reference as ref
from mtplx.edge0.ggml_quants import KVALUES_FP4, decode_q4_0, decode_q8_0, ue4m3_to_float
from mtplx.models.edge0_latent_moe import LatentGeometry, LatentMoeBlock, float32_logits, top_k_lowest_id
from tests import edge0_synth as syn

REAL = os.environ.get("MTPLX_EDGE0_MODEL")
real_only = pytest.mark.skipif(not REAL, reason="MTPLX_EDGE0_MODEL not set")

TIGHT = 2e-5  # float32, no TF32 GEMM involved
TF32 = 6e-3  # float32 with TF32 GEMMs (shared expert at S > 1, everything at prefill sizes)
BF16 = 1.5e-2


def _rel(got, want) -> float:
    return float(np.abs(np.asarray(got, np.float64) - want).max() / np.abs(want).max())


# --------------------------------------------------------------------------- a synthetic layer


class Synth:
    """Small layer: ggml-coded weights, converted for the block and decoded for the reference."""

    def __init__(self, seed=0, *, H=128, L=128, F=64, G=3, E=24, K=4, IS=64, labels=None, router=None, dense="q4"):
        rng = np.random.default_rng(seed)
        self.H, self.L, self.F, self.G, self.E, self.K = H, L, F, G, E, K
        arrays, floats = {}, {}
        for name, (r, c) in (("gate_proj", (F, L)), ("up_proj", (F, L)), ("down_proj", (L, F))):
            codes = rng.integers(0, 16, (E, r * c), dtype=np.uint8)
            scales = rng.integers(0x30, 0x40, (E, r * c // 16), dtype=np.uint8)
            w, s = lc.nvfp4_codes_to_mlx(codes, scales, (E, r, c))
            arrays[f"experts.{name}.weight"], arrays[f"experts.{name}.scales"] = w, s
            vals = KVALUES_FP4[codes].reshape(E, -1, 16) * 0.5 * ue4m3_to_float(scales)[..., None]
            floats[name] = vals.reshape(E, r, c)
        scal = np.stack([rng.uniform(0.02, 0.05, E), rng.uniform(0.02, 0.05, E), rng.uniform(5e-5, 3e-4, E)], 1).astype(np.float32)
        arrays["experts.scalars"] = scal

        def qd(r, c):  # ggml Q4_0 (Edge0-120B's D/U) or Q8_0 (Edge0-145B's)
            if dense == "q4":
                raw = syn.pack_q4_0(rng.integers(0, 16, r * c, dtype=np.uint8), rng.uniform(0.002, 0.02, r * c // 32).astype(np.float16))
                return raw, decode_q4_0(raw)
            raw = syn.pack_q8_0(rng.integers(-128, 128, r * c), rng.uniform(0.0002, 0.002, r * c // 32).astype(np.float16))
            return raw, decode_q8_0(raw)

        d_raw, d = qd(G * L, H)
        u_raw, u = qd(G * H, L)
        arrays.update({f"dense_d.{k}": v for k, v in lc.dense_to_affine(d_raw, dense, G * L, H).items()})
        arrays.update({f"dense_u.{k}": v for k, v in lc.regroup_u(lc.dense_to_affine(u_raw, dense, G * H, L), G).items()})
        if labels is None:  # unbalanced: one expert in group 0, three in group 2, the rest in group 1
            labels = np.ones(E, np.int32)
            labels[rng.permutation(E)[:4]] = [0, 2, 2, 2]
        arrays["labels"] = np.asarray(labels, np.int32)
        self.arrays = arrays
        self.dense_d = d.reshape(G, L, H)
        self.dense_u = u.reshape(G, H, L)
        self.labels = arrays["labels"]
        self.experts = [(floats["gate_proj"][e] * scal[e, 0], floats["up_proj"][e] * scal[e, 1], floats["down_proj"][e] * scal[e, 2]) for e in range(E)]
        self.router = (rng.standard_normal((E, H)) * 0.3).astype(np.float32) if router is None else router
        self.shared = ref.SharedExpertWeights(
            gate=(rng.standard_normal((IS, H)) * 0.1).astype(np.float32),
            up=(rng.standard_normal((IS, H)) * 0.1).astype(np.float32),
            down=(rng.standard_normal((H, IS)) * 0.1).astype(np.float32),
            gate_inp=(rng.standard_normal(H) * 0.1).astype(np.float32),
        )
        self.IS = IS

    def block(self, **kw) -> LatentMoeBlock:
        return make_block(self.arrays, self.router, self.shared, top_k=self.K, **kw)

    def reference(self, X):
        kw = dict(dense_d=self.dense_d, dense_u=self.dense_u, labels=self.labels, experts=self.experts, router=self.router, shared=self.shared, top_k=self.K, combine="per_group")
        return [ref.latent_moe(x, **kw) for x in np.asarray(X, np.float64)]


def make_block(arrays, router, shared, *, top_k, **kw) -> LatentMoeBlock:
    geo = LatentGeometry.from_arrays(arrays)
    blk = LatentMoeBlock(geo, top_k=top_k, shared_expert_ff=shared.gate.shape[0], **kw)
    blk.load_latent_arrays(arrays)
    blk.load_weights(
        [
            ("gate.weight", mx.array(router)),
            ("shared_expert.gate_proj.weight", mx.array(shared.gate)),
            ("shared_expert.up_proj.weight", mx.array(shared.up)),
            ("shared_expert.down_proj.weight", mx.array(shared.down)),
            ("shared_expert_gate.weight", mx.array(np.asarray(shared.gate_inp).reshape(1, -1))),
        ],
        strict=False,
    )
    mx.eval(blk.parameters())
    return blk


def _shared_mlx(blk, X):
    from mtplx.attention_math import attention_gate

    x = mx.array(np.asarray(X, np.float32))
    return np.array(attention_gate(blk.shared_expert(x), blk.shared_expert_gate(x)))


@pytest.fixture(scope="module")
def synth():
    return Synth()


# --------------------------------------------------------------------------- structure


def test_parameter_names_and_shapes(synth):
    blk = synth.block()
    H, L, F, G, E, IS = synth.H, synth.L, synth.F, synth.G, synth.E, synth.IS
    from mlx.utils import tree_flatten

    got = {k: (tuple(v.shape), v.dtype) for k, v in tree_flatten(blk.parameters())}
    assert got == {
        "gate.weight": ((E, H), mx.float32),
        "shared_expert.gate_proj.weight": ((IS, H), mx.float32),
        "shared_expert.up_proj.weight": ((IS, H), mx.float32),
        "shared_expert.down_proj.weight": ((H, IS), mx.float32),
        "shared_expert_gate.weight": ((1, H), mx.float32),
        "dense_d.weight": ((G * L, H // 8), mx.uint32),
        "dense_d.scales": ((G * L, H // 32), mx.float32),
        "dense_d.biases": ((G * L, H // 32), mx.float32),
        "dense_u.weight": ((H, G * L // 8), mx.uint32),
        "dense_u.scales": ((H, G * L // 32), mx.float32),
        "dense_u.biases": ((H, G * L // 32), mx.float32),
        "experts.gate_proj.weight": ((E, F, L // 8), mx.uint32),
        "experts.gate_proj.scales": ((E, F, L // 16), mx.uint8),
        "experts.up_proj.weight": ((E, F, L // 8), mx.uint32),
        "experts.up_proj.scales": ((E, F, L // 16), mx.uint8),
        "experts.down_proj.weight": ((E, L, F // 8), mx.uint32),
        "experts.down_proj.scales": ((E, L, F // 16), mx.uint8),
        "experts.scalars": ((E, 3), mx.float32),
        "labels": ((E,), mx.int32),
    }
    assert LatentGeometry.from_arrays(synth.arrays) == LatentGeometry(hidden=H, latent_dim=L, expert_ff=F, n_groups=G, n_experts=E, dense_bits=4)


def test_not_an_moe_block_subclass(synth):
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    blk = synth.block()
    assert not isinstance(blk, Qwen3NextSparseMoeBlock)
    assert not any("SparseMoe" in c.__name__ or "SwitchGLU" in c.__name__ for c in type(blk).__mro__)
    assert not any("Switch" in type(m).__name__ for _, m in blk.named_modules())


def test_load_latent_arrays_rejects_wrong_shapes(synth):
    blk = synth.block()
    bad = dict(synth.arrays)
    bad["experts.scalars"] = np.zeros((synth.E, 2), np.float32)
    with pytest.raises(ValueError, match="experts.scalars"):
        blk.load_latent_arrays(bad)
    with pytest.raises(KeyError):
        blk.load_latent_arrays({"experts.nope": np.zeros(1)})


# --------------------------------------------------------------------------- against the reference


@pytest.mark.parametrize("S", [1, 2, 4, 8])
def test_decode_and_verify_rows_match_reference(synth, S):
    blk = synth.block()
    rng = np.random.default_rng(10 + S)
    X = rng.standard_normal((S, synth.H)).astype(np.float32)
    y, ids, w = blk(mx.array(X)[None], return_routing=True)
    assert y.shape == (1, S, synth.H) and y.dtype == mx.float32
    assert ids.shape == (1, S, synth.K) and ids.dtype == mx.int32 and w.dtype == mx.float32
    res = synth.reference(X)
    np.testing.assert_array_equal(np.array(ids)[0], np.stack([r.ids for r in res]))
    np.testing.assert_allclose(np.array(w)[0], np.stack([r.weights for r in res]), rtol=2e-6, atol=1e-7)
    out = np.stack([r.out for r in res])
    routed = np.stack([r.routed for r in res])
    y = np.array(y)[0]
    assert _rel(y - _shared_mlx(blk, X), routed) < TIGHT  # the latent path: no TF32 GEMM below the sort threshold
    assert _rel(y, out) < (TIGHT if S == 1 else TF32)


def test_prefill_rows_match_reference(synth):
    blk = synth.block()
    S = 300  # 1200 routes: the expert-sorted path
    X = np.random.default_rng(3).standard_normal((S, synth.H)).astype(np.float32)
    y, ids, _ = blk(mx.array(X).reshape(2, S // 2, synth.H), return_routing=True)
    res = synth.reference(X)
    np.testing.assert_array_equal(np.array(ids).reshape(S, -1), np.stack([r.ids for r in res]))
    assert _rel(np.array(y).reshape(S, -1), np.stack([r.out for r in res])) < TF32


def test_sorted_and_unsorted_paths_agree(synth):
    X = mx.array(np.random.default_rng(4).standard_normal((1, 6, synth.H)).astype(np.float32))
    a = synth.block()(X)
    b = synth.block(sort_threshold=1)(X)
    assert _rel(np.array(b), np.array(a, np.float64)) < TF32


@pytest.mark.parametrize("S", [1, 4, 64])
def test_compiled_matches_eager(synth, S):
    blk = synth.block()
    fn = mx.compile(lambda x: blk(x, return_routing=True))
    X = mx.array(np.random.default_rng(5).standard_normal((1, S, synth.H)).astype(np.float32))
    ye, ie, we = blk(X, return_routing=True)
    for _ in range(2):  # second call reuses the trace
        yc, ic, wc = fn(X)
        np.testing.assert_array_equal(np.array(ic), np.array(ie))
        np.testing.assert_allclose(np.array(wc), np.array(we), rtol=1e-6)
        assert _rel(np.array(yc), np.array(ye, np.float64)) < 1e-5


def test_bfloat16_activations(synth):
    blk = synth.block()
    X = np.random.default_rng(6).standard_normal((1, 5, synth.H)).astype(np.float32)
    xb = mx.array(X).astype(mx.bfloat16)
    y = blk(xb)
    assert y.dtype == mx.bfloat16 and y.shape == xb.shape
    res = synth.reference(np.array(xb.astype(mx.float32))[0])
    assert _rel(np.array(y.astype(mx.float32))[0], np.stack([r.out for r in res])) < BF16
    lowp = synth.block(latent_dtype=mx.bfloat16)(xb)
    assert _rel(np.array(lowp.astype(mx.float32))[0], np.stack([r.out for r in res])) < BF16


def test_router_ties_go_to_the_lower_id():
    rng = np.random.default_rng(7)
    E, H = 24, 128
    router = (rng.standard_normal((E, H)) * 0.3).astype(np.float32)
    router[[3, 11, 17, 20]] = router[5]  # five experts with identical logits for every x
    router[[9, 14]] = router[2]
    s = Synth(router=router)
    blk = s.block()
    # x along the shared row makes the tied experts the top ones
    X = np.stack([router[5] * 4 + rng.standard_normal(H) * 0.05, router[2] * 4 + rng.standard_normal(H) * 0.05, rng.standard_normal(H)]).astype(np.float32)
    for S in (1, 3):
        _, ids, _ = blk(mx.array(X[:S])[None], return_routing=True)
        want = np.stack([r.ids for r in s.reference(X[:S])])
        np.testing.assert_array_equal(np.array(ids)[0], want)
    _, ids, _ = blk(mx.array(X[:1])[None], return_routing=True)
    assert np.array(ids)[0, 0, :4].tolist() == [3, 5, 11, 17]


def test_top_k_lowest_id_on_wide_rows():
    rng = np.random.default_rng(8)
    p = rng.integers(0, 6, (300, 512)).astype(np.float32) / 8  # lots of ties
    ids, vals = top_k_lowest_id(mx.array(p), 10)
    want = np.argsort(-p, axis=-1, kind="stable")[:, :10]
    np.testing.assert_array_equal(np.array(ids), want)
    np.testing.assert_array_equal(np.array(vals), np.take_along_axis(p, want, -1))


def test_float32_logits_are_float32_accurate():
    rng = np.random.default_rng(9)
    W = rng.standard_normal((64, 512)).astype(np.float32)
    X = rng.standard_normal((40, 512)).astype(np.float32)
    want = X.astype(np.float64) @ W.T.astype(np.float64)
    for w in (mx.array(W), mx.array(W).astype(mx.bfloat16)):
        wd = np.array(w.astype(mx.float32)).astype(np.float64)
        want = X.astype(np.float64) @ wd.T
        for rows in (1, 40):
            got = np.array(float32_logits(mx.array(X[:rows]), w))
            assert _rel(got, want[:rows]) < 1e-6


def test_unbalanced_groups_and_a_singleton_group():
    """Every expert of group 1 but one each in groups 0 and 2; route onto the singletons."""
    E = 24
    labels = np.ones(E, np.int32)
    labels[7], labels[19] = 0, 2
    s = Synth(seed=1, labels=labels)
    blk = s.block()
    rng = np.random.default_rng(11)
    X = np.stack([s.router[7] * 5, s.router[19] * 5, s.router[7] * 3 + s.router[19] * 3, rng.standard_normal(s.H)]).astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32) * 0.01
    y, ids, _ = blk(mx.array(X[:, None, :]), return_routing=True)  # B = 4, S = 1
    ids = np.array(ids)[:, 0]
    assert ids[0, 0] == 7 and ids[1, 0] == 19 and {7, 19} <= set(ids[2, :2].tolist())
    res = s.reference(X)
    np.testing.assert_array_equal(ids, np.stack([r.ids for r in res]))
    y = np.array(y)[:, 0]
    assert _rel(y - _shared_mlx(blk, X), np.stack([r.routed for r in res])) < TIGHT
    assert _rel(y, np.stack([r.out for r in res])) < TF32  # 4 rows: the f32 shared expert runs as TF32


def test_all_experts_in_one_group():
    s = Synth(seed=2, labels=np.full(24, 1, np.int32))
    X = np.random.default_rng(12).standard_normal((1, 1, s.H)).astype(np.float32)
    y = s.block()(mx.array(X))
    assert _rel(np.array(y)[0], np.stack([r.out for r in s.reference(X[0])])) < TIGHT


# --------------------------------------------------------------------------- real layers


@pytest.fixture(scope="module")
def real_pack():
    from mtplx.edge0.pack import Edge0Pack

    return Edge0Pack(Path(REAL))


def _real_block(pack, layer):
    lw = ref.layer_weights_from_pack(pack, layer)
    arrays = lc.convert_layer(pack, layer)
    blk = make_block(arrays, lw.router, lw.shared, top_k=ref.DEFAULT_TOP_K)
    cache: dict[int, tuple] = {}

    def experts(e):
        if e not in cache:
            cache[e] = lw.experts(e)
        return cache[e]

    kw = dict(dense_d=lw.dense_d, dense_u=lw.dense_u, labels=lw.labels, experts=experts, router=lw.router, shared=lw.shared, combine="per_group")
    return blk, lw, kw


def _plausible_inputs(lw, n, rng):
    """Block inputs as the layer produces them: gr_read of random residual stacks."""
    hc = lw.hc.inject.shape[0]
    hidden = lw.router.shape[1]
    xs = [ref.gr_read(rng.standard_normal((hc, hidden)), lw.hc.norm, lw.hc.down, lw.hc.up, lw.hc.inject)[0] for _ in range(n)]
    return np.stack(xs).astype(np.float32)


@real_only
@pytest.mark.parametrize("layer", [0, 23, 47])
def test_real_layer_against_reference(real_pack, layer):
    blk, lw, kw = _real_block(real_pack, layer)
    rng = np.random.default_rng(layer)
    X = _plausible_inputs(lw, 8, rng)
    norm = float(np.sqrt((X.astype(np.float64) ** 2).mean()))

    # (b) inputs aimed at chosen experts: the smallest and the largest group's first members, and one more
    counts = np.bincount(lw.labels, minlength=real_pack.n_groups)
    targets = [int(np.flatnonzero(lw.labels == np.argmin(counts))[0]), int(np.flatnonzero(lw.labels == np.argmax(counts))[0]), 300]
    aimed = []
    for e in targets:
        v = lw.router[e] / np.linalg.norm(lw.router[e])
        aimed.append(v * norm * np.sqrt(lw.router.shape[1]) * 0.5 + X[0] * 0.5)
    A = np.stack(aimed).astype(np.float32)

    for name, Z, S_list in (("random", X, (1, 4, 8)), ("aimed", A, (1, 3))):
        for S in S_list:
            y, ids, w = blk(mx.array(Z[:S])[None], return_routing=True)
            res = [ref.latent_moe(z, **kw) for z in Z[:S].astype(np.float64)]
            np.testing.assert_array_equal(np.array(ids)[0], np.stack([r.ids for r in res]))
            np.testing.assert_allclose(np.array(w)[0], np.stack([r.weights for r in res]), rtol=1e-5, atol=1e-7)
            y = np.array(y)[0]
            routed_err = _rel(y - _shared_mlx(blk, Z[:S]), np.stack([r.routed for r in res]))
            out_err = _rel(y, np.stack([r.out for r in res]))
            print(f"layer {layer} {name} S={S}: routed rel {routed_err:.2e}, out rel {out_err:.2e}")
            assert routed_err < TIGHT
            assert out_err < (TIGHT if S == 1 else TF32)
            if name == "aimed":
                assert all(t in r.ids for t, r in zip(targets, res))

    # prefill size, bfloat16 activations, under mx.compile
    fn = mx.compile(blk)
    xb = mx.array(X).astype(mx.bfloat16)[None]
    big = mx.concatenate([xb] * 16, axis=1)  # 128 rows: the sorted, TF32/bf16 GEMM path
    y = np.array(fn(big).astype(mx.float32))[0]
    res = [ref.latent_moe(z, **kw) for z in np.array(xb.astype(mx.float32))[0].astype(np.float64)]
    want = np.concatenate([np.stack([r.out for r in res])] * 16)
    err = _rel(y, want)
    print(f"layer {layer} bf16 compiled S=128: out rel {err:.2e}")
    assert err < BF16


@real_only
@pytest.mark.parametrize("layer", [0, 47])
def test_real_layer_fused_kernels(real_pack, layer, monkeypatch):
    """The fused routing and routed-expert kernels on a real layer (merged gate_up, compacted D/U scales, the
    router read through its kept bfloat16 copy when exact): bit-identical to the op chain at one row and at
    verify widths (routes, routing weights and outputs)."""
    from mtplx.models import edge0_latent_moe as moe

    blk, lw, _ = _real_block(real_pack, layer)
    blk.experts.merge_gate_up()
    blk.dense_d.compact_scales()
    blk.dense_u.compact_scales()
    w16 = mx.array(lw.router).astype(mx.bfloat16)
    if mx.array_equal(w16.astype(mx.float32), blk.gate.weight).item():
        blk.gate.weight = w16
        blk.cast_router_float32()
        assert blk._router_kernel_weight().dtype == mx.bfloat16
    X = mx.array(_plausible_inputs(lw, 8, np.random.default_rng(100 + layer)))
    for dtype in (mx.float32, mx.bfloat16):
        blk.latent_dtype = dtype
        xs = X.astype(dtype)[None]
        calls = [xs[:, t : t + 1] for t in range(xs.shape[1])] + [xs[:, :2], xs[:, :4], xs, mx.concatenate([xs[:, :3], xs[:, :1]], axis=1)]
        for xc in calls:
            monkeypatch.setattr(moe, "_FUSED_DEFAULT", True)
            got = blk(xc, return_routing=True)
            monkeypatch.setattr(moe, "_FUSED_DEFAULT", False)
            want = blk(xc, return_routing=True)
            assert all(mx.array_equal(a, b).item() for a, b in zip(got, want)), (dtype, xc.shape)
