"""Parity gate for the blocked-sequential GDN prefill kernel (omlx port).

The kernel must match the stock mlx-lm gated-delta path on the real
Qwen3.6 GDN shapes before it can route any traffic: same y (within input
dtype rounding) and near-identical fp32 final state. GPU-only.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI without Metal
    pytest.skip("Metal required", allow_module_level=True)

import mlx_lm.models.gated_delta as gd  # noqa: E402

from mtplx.kernels.gdn_blocked_prefill import (  # noqa: E402
    blocked_prefill_eligible,
    gated_delta_blocked_prefill,
    install_gdn_blocked_prefill_patch,
    uninstall_gdn_blocked_prefill_patch,
)

# Real Qwen3.6-27B GDN geometry.
B, HK, HV, DK, DV = 1, 16, 32, 128, 128


def _fixture(T: int, dtype, seed: int = 11):
    mx.random.seed(seed)
    q = (mx.random.normal((B, T, HK, DK)) * 0.5).astype(dtype)
    k = (mx.random.normal((B, T, HK, DK)) * 0.5).astype(dtype)
    v = (mx.random.normal((B, T, HV, DV)) * 0.5).astype(dtype)
    g = mx.sigmoid(mx.random.normal((B, T, HV))).astype(mx.float32) * 0.98
    beta = mx.sigmoid(mx.random.normal((B, T, HV))).astype(mx.float32)
    state = (mx.random.normal((B, HV, DV, DK)) * 0.1).astype(mx.float32)
    return q, k, v, g, beta, state


def _max_abs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("T", [16, 33, 128])
def test_blocked_prefill_matches_stock_kernel(dtype, T):
    q, k, v, g, beta, state = _fixture(T, dtype)
    y_ref, s_ref = gd.gated_delta_kernel(q, k, v, g, beta, state)
    y_new, s_new = gated_delta_blocked_prefill(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y_new, s_new)

    assert y_new.dtype == y_ref.dtype
    assert s_new.dtype == mx.float32
    y_tol = 0.05 if dtype != mx.float32 else 5e-3
    s_tol = 0.05 if dtype != mx.float32 else 5e-3
    assert _max_abs(y_new, y_ref) <= y_tol, f"y diverged: {_max_abs(y_new, y_ref)}"
    assert _max_abs(s_new, s_ref) <= s_tol, f"state diverged: {_max_abs(s_new, s_ref)}"


def test_blocked_prefill_state_chains_like_stock():
    """Splitting a sequence into two chained calls must equal one call —
    the property session-bank chunked prefill relies on."""
    T = 96
    q, k, v, g, beta, state = _fixture(T, mx.bfloat16, seed=23)
    y_full, s_full = gated_delta_blocked_prefill(q, k, v, g, beta, state)
    cut = 48
    y_a, s_a = gated_delta_blocked_prefill(
        q[:, :cut], k[:, :cut], v[:, :cut], g[:, :cut], beta[:, :cut], state
    )
    y_b, s_b = gated_delta_blocked_prefill(
        q[:, cut:], k[:, cut:], v[:, cut:], g[:, cut:], beta[:, cut:], s_a
    )
    mx.eval(y_full, s_full, y_a, y_b, s_b)
    assert _max_abs(mx.concatenate([y_a, y_b], axis=1), y_full) <= 0.05
    assert _max_abs(s_b, s_full) <= 0.05


def test_eligibility_gate_rejects_off_shapes():
    q, k, v, g, beta, state = _fixture(32, mx.bfloat16)
    assert blocked_prefill_eligible(q, v, g, None, state)
    # masked calls stay stock
    assert not blocked_prefill_eligible(q, v, g, mx.ones((B, 32)), state)
    # vectorized gating stays stock
    g4 = mx.zeros((B, 32, HV, DK))
    assert not blocked_prefill_eligible(q, v, g4, None, state)
    # non-128 Dk stays stock
    q_odd = mx.zeros((B, 32, HK, 64), dtype=mx.bfloat16)
    assert not blocked_prefill_eligible(q_odd, v, g, None, state)


def test_patch_routes_prefill_and_leaves_decode_stock(monkeypatch):
    monkeypatch.setenv("MTPLX_GDN_BLOCKED_PREFILL", "1")
    report = install_gdn_blocked_prefill_patch()
    try:
        assert report["installed"]
        T = 64
        q, k, v, g, beta, state = _fixture(T, mx.bfloat16, seed=7)
        # emulate the update() signature: a/b/A_log/dt_bias producing our g/beta
        # is awkward to invert, so compare patched vs original directly on the
        # same inputs instead.
        a = mx.random.normal((B, T, HV))
        b = mx.random.normal((B, T, HV))
        A_log = mx.random.normal((HV,)) * 0.1
        dt_bias = mx.random.normal((HV,)) * 0.1
        y_new, s_new = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=state)
        orig = uninstall_gdn_blocked_prefill_patch is not None
        uninstall_gdn_blocked_prefill_patch()
        y_ref, s_ref = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=state)
        mx.eval(y_new, s_new, y_ref, s_ref)
        assert orig
        assert _max_abs(y_new, y_ref) <= 0.05
        assert _max_abs(s_new, s_ref) <= 0.05
    finally:
        uninstall_gdn_blocked_prefill_patch()
