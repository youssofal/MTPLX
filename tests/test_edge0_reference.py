"""Tests for the NumPy reference of the Edge0 latent MoE half (mtplx.edge0.reference)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from mtplx.edge0 import reference as ref


def _silu(v: float) -> float:
    return v / (1.0 + math.exp(-v))


def _synthetic(seed=0, *, H=24, L=16, F=8, G=3, E=12, HC=4, r=5):  # F: expert width
    rng = np.random.default_rng(seed)
    experts = [
        (rng.standard_normal((F, L)) * 0.3, rng.standard_normal((F, L)) * 0.3, rng.standard_normal((L, F)) * 0.3)
        for _ in range(E)
    ]
    shared = ref.SharedExpertWeights(
        gate=rng.standard_normal((F, H)) * 0.2,
        up=rng.standard_normal((F, H)) * 0.2,
        down=rng.standard_normal((H, F)) * 0.2,
        gate_inp=rng.standard_normal(H) * 0.2,
    )
    weights = ref.LayerWeights(
        layer=0,
        hc=ref.HyperConnectionWeights(
            norm=1.0 + 0.1 * rng.standard_normal(HC * H),
            down=rng.standard_normal((r, HC * H)) * 0.1,
            up=rng.standard_normal((HC * H, r)) * 0.1,
            inject=rng.standard_normal((HC, HC * H)) * 0.1,
        ),
        router=rng.standard_normal((E, H)),
        shared=shared,
        dense_d=rng.standard_normal((G, L, H)) * 0.2,
        dense_u=rng.standard_normal((G, H, L)) * 0.2,
        labels=rng.integers(0, G, size=E).astype(np.int32),
        experts=lambda e: experts[e],
    )
    return weights, rng


def _moe_kwargs(w: ref.LayerWeights, top_k: int):
    return dict(
        dense_d=w.dense_d,
        dense_u=w.dense_u,
        labels=w.labels,
        experts=w.experts,
        router=w.router,
        shared=w.shared,
        top_k=top_k,
    )


def test_shapes_and_batch():
    w, rng = _synthetic()
    X = rng.standard_normal((5, 24))
    out, ids, wts = ref.latent_moe_batch(X, **_moe_kwargs(w, 4))
    assert out.shape == (5, 24) and ids.shape == (5, 4) and wts.shape == (5, 4)
    assert ids.dtype == np.int32
    np.testing.assert_allclose(wts.sum(axis=1), 1.0, rtol=1e-12)
    one = ref.latent_moe(X[2], **_moe_kwargs(w, 4))
    np.testing.assert_array_equal(one.ids, ids[2])
    np.testing.assert_allclose(one.out, out[2], rtol=0, atol=0)
    np.testing.assert_array_equal(one.groups, w.labels[one.ids])
    assert len(one.latent) == 4 and one.latent[0].shape == (16,)

    R = rng.standard_normal((4, 24))
    R_new, res, x, inject = ref.moe_half(R, w, top_k=4)
    assert R_new.shape == (4, 24) and x.shape == (24,) and inject.shape == (4,)
    assert res.out.shape == (24,)


def test_float32_option_close_to_float64():
    w, rng = _synthetic(1)
    x = rng.standard_normal(24)
    a = ref.latent_moe(x, **_moe_kwargs(w, 4))
    b = ref.latent_moe(x, **_moe_kwargs(w, 4), dtype=np.float32)
    assert b.out.dtype == np.float32
    np.testing.assert_array_equal(a.ids, b.ids)
    np.testing.assert_allclose(b.out, a.out, rtol=1e-4, atol=1e-5)


def test_route_tie_rule_and_rank_order():
    # experts 1, 3 and 6 share the top logit; 2 and 5 share the next one
    W = np.zeros((8, 1))
    W[[1, 3, 6], 0] = 2.0
    W[[2, 5], 0] = 1.0
    ids, w = ref.route(np.array([1.0]), W, top_k=4)
    assert ids.tolist() == [1, 3, 6, 2]
    # renormalised over the selected four
    p = np.exp(np.array([2.0, 2.0, 2.0, 1.0]))
    np.testing.assert_allclose(w, p / p.sum(), rtol=1e-12)
    # all equal: the lowest ids, in order
    ids, _ = ref.route(np.array([1.0]), np.ones((8, 1)), top_k=3)
    assert ids.tolist() == [0, 1, 2]
    # float32 path keeps the same rule
    ids, _ = ref.route(np.array([1.0], np.float32), W.astype(np.float32), top_k=4, dtype=np.float32)
    assert ids.tolist() == [1, 3, 6, 2]


def test_route_weight_floor():
    # 40000 equal logits: each p = 2.5e-5, below the 2**-14 floor for k = 1
    E = 40000
    ids, w = ref.route(np.array([1.0]), np.ones((E, 1)), top_k=1)
    assert ids.tolist() == [0]
    assert w[0] == pytest.approx((1.0 / E) / ref.ROUTER_WEIGHT_FLOOR, rel=1e-12)
    assert w[0] < 1.0
    # above the floor the weights sum to one
    _, w = ref.route(np.array([1.0]), np.ones((E, 1)), top_k=10)
    assert w.sum() == pytest.approx(1.0, rel=1e-12)


def test_per_group_combine_matches_per_route():
    for seed in range(4):
        w, rng = _synthetic(seed, G=3, E=12)
        for _ in range(5):
            x = rng.standard_normal(24)
            a = ref.latent_moe(x, **_moe_kwargs(w, 6), combine="per_route")
            b = ref.latent_moe(x, **_moe_kwargs(w, 6), combine="per_group")
            assert len(set(a.groups.tolist())) < 6  # some group carries several routes
            np.testing.assert_array_equal(a.ids, b.ids)
            np.testing.assert_allclose(b.routed, a.routed, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(b.out, a.out, rtol=1e-12, atol=1e-12)


def test_hand_computed_case():
    # H = 2, L = 1, I = 1, G = 2, E = 3, K = 2
    x = np.array([1.0, 2.0])
    D = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # xl[0] = 1, xl[1] = 2
    U = np.array([[[1.0], [0.0]], [[1.0], [1.0]]])
    router = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # logits 0, 1, 2
    labels = np.array([0, 0, 1], np.int32)
    experts = {
        0: (np.array([[9.0]]), np.array([[9.0]]), np.array([[9.0]])),  # never routed
        1: (np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])),
        2: (np.array([[1.0]]), np.array([[3.0]]), np.array([[2.0]])),
    }
    shared = {
        "gate": np.array([[1.0, 0.0]]),
        "up": np.array([[0.0, 1.0]]),
        "down": np.array([[1.0], [0.0]]),
        "gate_inp": np.array([0.0, 0.0]),
    }
    res = ref.latent_moe(
        x, dense_d=D, dense_u=U, labels=labels, experts=experts, router=router, shared=shared, top_k=2
    )
    assert res.ids.tolist() == [2, 1]
    e = math.e
    w2, w1 = e / (e + 1.0), 1.0 / (e + 1.0)  # e^2 / (e^2 + e), e / (e^2 + e)
    np.testing.assert_allclose(res.weights, [w2, w1], rtol=1e-14)
    h2 = 2.0 * _silu(2.0) * 6.0  # down * silu(gate * 2) * (up * 2)
    h1 = _silu(1.0) * 1.0
    routed = w2 * np.array([h2, h2]) + w1 * np.array([h1, 0.0])
    sh = 0.5 * np.array([_silu(1.0) * 2.0, 0.0])  # sigmoid(0) * down @ (silu(1) * 2)
    np.testing.assert_allclose(res.routed, routed, rtol=1e-14)
    np.testing.assert_allclose(res.shared, sh, rtol=1e-14)
    np.testing.assert_allclose(res.out, routed + sh, rtol=1e-14)


def test_latent_expert_scales_are_output_scalars():
    rng = np.random.default_rng(3)
    g, u, d = rng.standard_normal((4, 6)), rng.standard_normal((4, 6)), rng.standard_normal((6, 4))
    xl = rng.standard_normal(6)
    a = ref.latent_expert(xl, g, u, d, s_gate=0.5, s_up=0.25, s_down=3.0)
    b = ref.latent_expert(xl, 0.5 * g, 0.25 * u, 3.0 * d)
    np.testing.assert_allclose(a, b, rtol=1e-13)


def test_gr_read_hand_case_and_write_identity():
    HC, H = 2, 3
    R = np.array([[3.0, 0.0, 4.0], [1.0, 1.0, 1.0]])
    norm = np.ones(HC * H)
    # zero projections: lo = silu(0) = 0, the gate is sigmoid(0) = 1/2
    mixed, inject = ref.gr_read(R, norm, np.zeros((2, HC * H)), np.zeros((HC * H, 2)), np.zeros((HC, HC * H)), eps=0.0)
    xn0 = R[0] / math.sqrt(25.0 / 3.0)
    xn1 = R[1] / 1.0
    np.testing.assert_allclose(mixed, 0.5 * (0.5 * xn0 + 0.5 * xn1), rtol=1e-14)
    np.testing.assert_allclose(inject, [0.0, 0.0])
    # 1/HC inside the silu
    w_down = np.zeros((1, HC * H))
    w_down[0, 0] = 1.0
    w_up = np.zeros((HC * H, 1))
    w_up[:, 0] = 1.0
    mixed, _ = ref.gr_read(R, norm, w_down, w_up, None, eps=0.0)
    gate = 1.0 / (1.0 + math.exp(-_silu(xn0[0] / HC)))
    np.testing.assert_allclose(mixed, gate * 0.5 * (xn0 + xn1), rtol=1e-14)
    # a zero injection is a plain residual add; others scale per stream by 2*sigmoid(inject/HC)
    out = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(ref.residual_write(R, out, np.zeros(HC)), R + out)
    got = ref.residual_write(R, out, np.array([2.0, -4.0]))
    np.testing.assert_allclose(got[0], R[0] + out * 2.0 / (1.0 + math.exp(-1.0)))
    np.testing.assert_allclose(got[1], R[1] + out * 2.0 / (1.0 + math.exp(2.0)))


MODEL = os.environ.get("MTPLX_EDGE0_MODEL")


@pytest.mark.skipif(not MODEL, reason="set MTPLX_EDGE0_MODEL to an Edge0 model folder")
def test_real_folder_layer_runs():
    from mtplx.edge0.pack import Edge0Pack

    pack = Edge0Pack(Path(MODEL))
    w = ref.layer_weights_from_pack(pack, 0)
    E, H = w.router.shape
    G, L, H2 = w.dense_d.shape
    assert H2 == H and w.dense_u.shape == (G, H, L) and w.labels.shape == (E,)
    assert w.hc.norm.shape[0] % H == 0
    gate, up, down = w.experts(int(np.argmax(np.bincount(w.labels))))
    assert gate.shape[1] == L and down.shape[0] == L
    rng = np.random.default_rng(0)
    HC = w.hc.inject.shape[0]
    R = rng.standard_normal((HC, H)).astype(np.float32)
    R_new, res, x, inject = ref.moe_half(R, w)
    assert np.isfinite(R_new).all() and np.isfinite(res.out).all()
    assert res.ids.shape == (ref.DEFAULT_TOP_K,) and len(set(res.ids.tolist())) == ref.DEFAULT_TOP_K
    assert res.weights.sum() == pytest.approx(1.0, rel=1e-9)
    grouped = ref.latent_moe(
        x,
        dense_d=w.dense_d,
        dense_u=w.dense_u,
        labels=w.labels,
        experts=w.experts,
        router=w.router,
        shared=w.shared,
        combine="per_group",
    )
    np.testing.assert_allclose(grouped.out, res.out, rtol=1e-9, atol=1e-9)
