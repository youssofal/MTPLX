"""GDN prefill through MLX's ``mx.fast.gated_delta_update`` (MTPLX_GDN_FAST_PREFILL).

The route must match the float32 recurrence on Qwen3.8 GDN shapes with a
decay close to 1 (where a bf16 decay loses most of its precision), keep every
call it declines on the previous path, install nothing on MLX builds without
the op, and name itself in the session policy fingerprint only while it is on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI without Metal
    pytest.skip("Metal required", allow_module_level=True)

import mlx_lm.models.gated_delta as gd  # noqa: E402

from mtplx.kernels import gdn_fast_prefill as module  # noqa: E402
from mtplx.kernels.gdn_fast_prefill import (  # noqa: E402
    fast_prefill_available,
    gated_delta_fast_prefill,
    gdn_prefill_numerics_identity,
    install_gdn_fast_prefill_patch,
    uninstall_gdn_fast_prefill_patch,
)

# Qwen3.8-27B GDN geometry.
B, HK, HV, DK, DV = 1, 16, 48, 128, 128

needs_fast_op = pytest.mark.skipif(
    not fast_prefill_available(),
    reason="mx.fast.gated_delta_update needs MLX 0.32.4",
)


def _inputs(T: int, seed: int = 5):
    """Normalized q/k as mlx-lm feeds them; decay mostly above 0.99, as on Qwen3.8."""
    mx.random.seed(seed)
    inv = DK**-0.5
    q = (inv**2) * mx.fast.rms_norm(mx.random.normal((B, T, HK, DK)), None, 1e-6)
    k = inv * mx.fast.rms_norm(mx.random.normal((B, T, HK, DK)), None, 1e-6)
    v = mx.random.normal((B, T, HV, DV)) * 0.5
    a = mx.random.normal((B, T, HV))
    b = mx.random.normal((B, T, HV))
    A_log = mx.log(mx.random.uniform(low=0.01, high=0.5, shape=(HV,)))
    dt_bias = mx.full((HV,), -5.0)
    state = mx.random.normal((B, HV, DV, DK)) * 0.05
    bf16 = mx.bfloat16
    return (
        q.astype(bf16),
        k.astype(bf16),
        v.astype(bf16),
        a.astype(bf16),
        b.astype(bf16),
        A_log,
        dt_bias,
        state,
    )


def _rel(a, ref):
    a = a.astype(mx.float32)
    ref = ref.astype(mx.float32)
    return float(mx.sqrt(mx.sum((a - ref) ** 2)) / mx.sqrt(mx.sum(ref**2)))


@needs_fast_op
@pytest.mark.parametrize("T", [64, 100, 257])
def test_fast_prefill_matches_float32_recurrence(T):
    q, k, v, a, b, A_log, dt_bias, state = _inputs(T)
    g = gd.compute_g(A_log, a, dt_bias)
    beta = mx.sigmoid(b)
    assert float(mx.mean(g > 0.99)) > 0.5
    f32 = mx.float32
    y_ref, s_ref = gd.gated_delta_ops(
        q.astype(f32),
        k.astype(f32),
        v.astype(f32),
        g.astype(f32),
        beta.astype(f32),
        state,
    )
    y, s = gated_delta_fast_prefill(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y, s)
    assert y.dtype == mx.bfloat16
    assert s.dtype == mx.float32
    # bf16 output rounding is ~2e-3 relative; the chunked state sums in another
    # order than the per-token loop (~1e-3 measured on real activations).
    assert _rel(y, y_ref) < 5e-3
    assert _rel(s, s_ref) < 3e-3


@needs_fast_op
def test_patch_routes_prefill_calls_and_keeps_the_rest(monkeypatch):
    from mtplx import demotions

    monkeypatch.setenv("MTPLX_GDN_FAST_PREFILL", "1")
    monkeypatch.delenv("MTPLX_GDN_FAST_PREFILL_MIN_T", raising=False)
    demotions.reset()
    stock = gd.gated_delta_update
    report = install_gdn_fast_prefill_patch()
    try:
        assert report["installed"]
        assert report["min_t"] == 64
        assert gd.gated_delta_update is not stock
        before = dict(module.ROUTE_COUNTS)
        q, k, v, a, b, A_log, dt_bias, state = _inputs(64)
        y, s = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=state)
        mx.eval(y, s)
        assert module.ROUTE_COUNTS["routed"] == before["routed"] + 1
        y_ref, s_ref = stock(q, k, v, a, b, A_log, dt_bias, state=state)
        assert _rel(s, s_ref) < 3e-3

        # Below the floor: the previous path, byte for byte, not counted.
        q, k, v, a, b, A_log, dt_bias, state = _inputs(16)
        y, s = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=state)
        y_ref, s_ref = stock(q, k, v, a, b, A_log, dt_bias, state=state)
        assert mx.array_equal(y, y_ref) and mx.array_equal(s, s_ref)
        assert module.ROUTE_COUNTS["routed"] == before["routed"] + 1

        # A masked prefill-sized call falls back and is reported.
        q, k, v, a, b, A_log, dt_bias, state = _inputs(64)
        mask = mx.ones((B, 64), dtype=mx.bool_)
        y, s = gd.gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, state=state, mask=mask
        )
        y_ref, s_ref = stock(q, k, v, a, b, A_log, dt_bias, state=state, mask=mask)
        assert mx.array_equal(y, y_ref) and mx.array_equal(s, s_ref)
        assert (
            module.ROUTE_COUNTS["fallback_prefill_sized"]
            == before["fallback_prefill_sized"] + 1
        )
        assert demotions.counts()["gdn_fast_prefill_not_engaged"] == 1
    finally:
        uninstall_gdn_fast_prefill_patch()
        demotions.reset()
    assert gd.gated_delta_update is stock


@needs_fast_op
def test_float32_gates_probe_matches_the_build():
    q, k, v, a, b, A_log, dt_bias, state = _inputs(64)
    g = gd.compute_g(A_log, a, dt_bias)
    beta = mx.sigmoid(b).astype(mx.float32)
    f32 = mx.float32
    _, s16 = mx.fast.gated_delta_update(q, k, v, g, beta, state)
    _, s32 = mx.fast.gated_delta_update(
        q.astype(f32), k.astype(f32), v.astype(f32), g, beta, state
    )
    keeps = module.mlx_keeps_float32_gates()
    # A build that rounds the gates to bf16 is far off (~5e-2 on this decay).
    assert (_rel(s16, s32) < 1e-3) == keeps


@needs_fast_op
@pytest.mark.parametrize("keeps", [True, False])
def test_route_passes_16_bit_activations_only_when_the_gates_stay_float32(
    monkeypatch, keeps
):
    seen = []
    real = mx.fast.gated_delta_update

    def spy(q, k, v, g, beta, state=None):
        seen.append((q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype))
        return real(q, k, v, g, beta, state)

    monkeypatch.setitem(module._FLOAT32_GATES, "probed", True)
    monkeypatch.setitem(module._FLOAT32_GATES, "value", keeps)
    monkeypatch.setattr(module.mx.fast, "gated_delta_update", spy)
    q, k, v, a, b, A_log, dt_bias, state = _inputs(64)
    y, s = gated_delta_fast_prefill(
        q, k, v, gd.compute_g(A_log, a, dt_bias), mx.sigmoid(b), state
    )
    mx.eval(y, s)
    act = mx.bfloat16 if keeps else mx.float32
    assert seen == [(act, act, act, mx.float32, mx.float32)]
    assert y.dtype == mx.bfloat16


def test_without_the_mlx_op_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(module, "fast_prefill_available", lambda: False)
    stock = gd.gated_delta_update
    report = install_gdn_fast_prefill_patch()
    assert not report["installed"]
    assert "0.32.4" in report["error"]
    assert gd.gated_delta_update is stock
    assert gdn_prefill_numerics_identity() is None


def _fingerprint_state() -> SimpleNamespace:
    args = SimpleNamespace(
        strip_assistant_reasoning_history=False,
        adaptive_policy="none",
        online_correction_cache=False,
        online_correction_cache_min_depth=1,
        online_correction_cache_key="local_prefix",
        prompt_correction_cache=False,
        prompt_correction_cache_min_depth=2,
        online_hidden_corrector_alpha=0.25,
        online_hidden_corrector_decay=0.7,
        online_hidden_corrector_warmup=2,
        online_hidden_corrector_max_feed_depth=2,
        online_hidden_corrector_key="token",
    )
    return SimpleNamespace(
        args=args, template_hash="template", draft_head_identity="draft"
    )


@needs_fast_op
def test_policy_fingerprint_names_the_route_only_while_it_is_on():
    from mtplx.server.openai import _policy_fingerprint

    state = _fingerprint_state()
    off = _policy_fingerprint(state, thinking_enabled=False)
    assert "gdn_prefill=" not in off
    install_gdn_fast_prefill_patch()
    try:
        on = _policy_fingerprint(state, thinking_enabled=False)
        identity = gdn_prefill_numerics_identity()
        assert identity is not None and identity.startswith("mlx_fast_f32_v1:mlx")
        assert on == f"{off};gdn_prefill={identity}"
    finally:
        uninstall_gdn_fast_prefill_patch()
    assert _policy_fingerprint(state, thinking_enabled=False) == off
