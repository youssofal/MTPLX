from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.kernels import deepseek_v4_moe_router as router


def test_mia_router_row_axis_matches_installed_launch(monkeypatch):
    captured = {}

    def capture_kernel(**spec):
        name = spec["name"]
        captured[name] = {"source": spec["source"]}

        def launch(**kwargs):
            captured[name]["launch"] = kwargs
            return tuple(
                mx.zeros(shape, dtype=dtype)
                for shape, dtype in zip(
                    kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
                )
            )

        return launch

    router._score_router_kernel.cache_clear()
    router._hash_router_kernel.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", capture_kernel)
    rows = 127
    logits = mx.zeros((rows, 216), dtype=mx.float32)

    score = router.install_score_router(experts=216, topk=6, route_scale=1.5)
    score(logits, mx.zeros((216,), dtype=mx.float32))
    hashed = router.install_hash_router(experts=216, topk=6, route_scale=1.5)
    hashed(
        logits,
        mx.zeros((129280, 6), dtype=mx.int32),
        mx.zeros((rows,), dtype=mx.int32),
    )

    for name in (
        "mtplx_dsv4_mia_sqrtsoftplus_top6_k216",
        "mtplx_dsv4_mia_hash_sqrtsoftplus_top6_k216",
    ):
        assert "uint row = threadgroup_position_in_grid.z;" in captured[name]["source"]
        assert captured[name]["launch"]["grid"] == (32, 1, rows)
        assert captured[name]["launch"]["threadgroup"] == (32, 1, 1)

    router._score_router_kernel.cache_clear()
    router._hash_router_kernel.cache_clear()


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is unavailable")
def test_mia_router_writes_every_prefill_row_with_source_arithmetic():
    rows = 127
    experts = 216
    row = np.arange(rows, dtype=np.float32)[:, None]
    expert = np.arange(experts, dtype=np.float32)[None, :]
    logits = (expert - np.float32(107.5)) / np.float32(31.0)
    logits = logits + (row - np.float32(63.0)) / np.float32(127.0)
    correction = (
        (np.arange(experts, dtype=np.float32) % np.float32(11.0))
        / np.float32(997.0)
    )
    softplus = np.log1p(np.exp(logits, dtype=np.float32), dtype=np.float32)
    unbiased = np.sqrt(softplus, dtype=np.float32)
    selected = np.argsort(
        -(unbiased + correction[None, :]), axis=1, kind="stable"
    )[:, :6]
    selected_scores = np.take_along_axis(unbiased, selected, axis=1)
    expected_score_weights = (
        selected_scores
        / selected_scores.sum(axis=1, keepdims=True, dtype=np.float32)
        * np.float32(1.5)
    )

    score = router.install_score_router(experts=216, topk=6, route_scale=1.5)
    score_ids, score_weights = score(
        mx.array(logits), mx.array(correction, dtype=mx.float32)
    )

    input_ids = np.arange(rows, dtype=np.int32)
    hash_owners = (
        input_ids[:, None] * np.int32(17)
        + np.arange(6, dtype=np.int32)[None, :] * np.int32(29)
    ) % np.int32(experts)
    tid2eid = np.zeros((129280, 6), dtype=np.int32)
    tid2eid[input_ids] = hash_owners
    hash_scores = np.take_along_axis(unbiased, hash_owners, axis=1)
    expected_hash_weights = (
        hash_scores
        / hash_scores.sum(axis=1, keepdims=True, dtype=np.float32)
        * np.float32(1.5)
    )
    hashed = router.install_hash_router(experts=216, topk=6, route_scale=1.5)
    hash_ids, hash_weights = hashed(
        mx.array(logits),
        mx.array(tid2eid),
        mx.array(input_ids),
    )
    mx.eval(score_ids, score_weights, hash_ids, hash_weights)

    np.testing.assert_array_equal(np.array(score_ids), selected.astype(np.int32))
    np.testing.assert_allclose(
        np.array(score_weights), expected_score_weights, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_array_equal(np.array(hash_ids), hash_owners)
    np.testing.assert_allclose(
        np.array(hash_weights), expected_hash_weights, rtol=2e-6, atol=2e-6
    )
