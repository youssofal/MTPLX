"""Load-time turbo kernel self-validation: pass, fallback, and force-fallback."""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import gdn_capture, kernel_selfcheck, nax_verify
from mtplx.kernel_selfcheck import (
    lane_disabled,
    report_for_health,
    run_kernel_selfcheck,
    selfcheck_enabled,
)


@pytest.fixture(autouse=True)
def _clean_selfcheck_state():
    kernel_selfcheck._reset_for_tests()
    yield
    kernel_selfcheck._reset_for_tests()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bits", [4, 8])
def test_selfcheck_passes_on_this_machine(monkeypatch, dtype, bits) -> None:
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    monkeypatch.setenv("MTPLX_NAX_FLASH_ROUTE", "1")
    monkeypatch.setenv("MTPLX_NAX_TILE_ROUTE", "1")
    report = run_kernel_selfcheck(dtype, bits, 64)
    lanes = report["lanes"]
    checked = {lane: s for lane, s in lanes.items() if s != "skipped"}
    assert checked, "no lanes engaged — the selfcheck validated nothing"
    bad = {lane: s for lane, s in checked.items() if s != "ok"}
    assert not bad, f"selfcheck lanes failed on this machine: {bad} dmax={report['dmax']}"
    assert not any(lane_disabled(lane) for lane in lanes)
    # The qmm lanes for the model's bits and the packed-GQA lane must be
    # among the validated set.
    assert lanes["qmm_m4"] == "ok"
    assert lanes["qmm_m6"] == "ok"
    assert lanes["gqa_packed_sdpa"] == "ok"
    for lane in ("nax_flash_sdpa", "nax_flash_dsplit_sdpa", "nax_tile_sdpa"):
        assert lanes[lane] == ("ok" if nax_verify.nax_available() else "skipped")


def test_nax_attention_selfcheck_failure_is_local_to_its_lane(monkeypatch) -> None:
    if not nax_verify.nax_available():
        pytest.skip("NAX hardware/OS unavailable")
    from mtplx.kernels import sdpa_nax_flash_dsplit as module

    monkeypatch.setenv("MTPLX_NAX_VERIFY", "0")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    monkeypatch.setenv("MTPLX_NAX_FLASH_ROUTE", "1")
    monkeypatch.setattr(module, "sdpa_nax_flash_dsplit", lambda **kwargs: None)
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["nax_flash_dsplit_sdpa"] == "fallback"
    assert lane_disabled("nax_flash_dsplit_sdpa")
    assert report["lanes"]["nax_flash_sdpa"] == "ok"
    assert report["lanes"]["gqa_packed_sdpa"] == "ok"


def test_nax_dsplit4_selfcheck_failure_falls_back_to_two_way(monkeypatch) -> None:
    if not nax_verify.nax_available():
        pytest.skip("NAX hardware/OS unavailable")
    from mtplx.kernels import sdpa_nax_flash_dsplit as module

    monkeypatch.setenv("MTPLX_NAX_VERIFY", "0")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    monkeypatch.setenv("MTPLX_NAX_FLASH_ROUTE", "1")
    monkeypatch.setenv("MTPLX_NAX_FLASH_DSPLIT4", "1")
    real = module.sdpa_nax_flash_dsplit

    def wrong_quarter(**kwargs):
        if kwargs.get("ndh") == 4:
            return None
        return real(**kwargs)

    monkeypatch.setattr(module, "sdpa_nax_flash_dsplit", wrong_quarter)
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["nax_flash_dsplit4_sdpa"] == "fallback"
    assert lane_disabled("nax_flash_dsplit4_sdpa")
    assert report["lanes"]["nax_flash_dsplit_sdpa"] == "ok"


def test_selfcheck_mismatch_disables_lane_and_surfaces_in_health(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")

    original = nax_verify.nax_qmm_m4

    def corrupted(x2, w_q, scales, biases, *, group_size=64):
        return original(x2, w_q, scales, biases, group_size=group_size) + 1000.0

    monkeypatch.setattr(nax_verify, "nax_qmm_m4", corrupted)
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["qmm_m4"] == "fallback"
    assert lane_disabled("qmm_m4")
    # The sibling lanes stay engaged: fallback is per-lane, not global.
    assert report["lanes"]["qmm_m6"] == "ok"
    assert not lane_disabled("qmm_m6")

    health = report_for_health()
    assert health["ran"] is True
    assert health["qmm_m4"] == "fallback"
    assert health["qmm_m6"] == "ok"
    json.dumps(health)  # JSON primitives only — the watchdog Codable lesson


def test_disabled_lane_routes_stock_through_the_qlinear_patch(monkeypatch) -> None:
    from mtplx.attention_context import attention_phase

    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")

    original = nax_verify.nax_qmm_m4

    def corrupted(x2, w_q, scales, biases, *, group_size=64):
        return original(x2, w_q, scales, biases, group_size=group_size) + 1000.0

    monkeypatch.setattr(nax_verify, "nax_qmm_m4", corrupted)
    run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert lane_disabled("qmm_m4")

    calls = {"m4": 0}

    def counting(x2, w_q, scales, biases, *, group_size=64):
        calls["m4"] += 1
        return original(x2, w_q, scales, biases, group_size=group_size)

    monkeypatch.setattr(nax_verify, "nax_qmm_m4", counting)
    report = nax_verify.install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            y = layer(x)
            mx.eval(y)
        assert y.shape == (4, 256)
        assert calls["m4"] == 0, "disabled qmm_m4 lane still routed the custom kernel"
    finally:
        nax_verify.uninstall_nax_qlinear_patch()


def test_selfcheck_kernel_exception_falls_back_instead_of_raising(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic kernel failure")

    monkeypatch.setattr(nax_verify, "nax_qmm_m6", broken)
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["qmm_m6"] == "fallback"
    assert lane_disabled("qmm_m6")
    assert report["lanes"]["qmm_m4"] == "ok"


def test_force_gpu_family_fallback_disables_nax_lane(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    nax_verify.nax_available.cache_clear()
    try:
        assert nax_verify.nax_available() is False
        assert not nax_verify.m16_nax_eligible(8, 5120, 17408, 4, 64, mx.bfloat16)
        report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
        assert report["lanes"]["qmm_m16_nax"] == "skipped"
        # The plain-SIMD lanes carry the win and must still validate.
        assert report["lanes"]["qmm_m4"] == "ok"
        assert report["lanes"]["qmm_m6"] == "ok"
    finally:
        nax_verify.nax_available.cache_clear()


def test_selfcheck_enabled_gating(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_KERNEL_SELFCHECK", raising=False)
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    monkeypatch.delenv("MTPLX_FUSE_GDN_POST_CONV", raising=False)
    assert selfcheck_enabled() is False
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    assert selfcheck_enabled() is True
    monkeypatch.delenv("MTPLX_FUSE_GDN_POST_CONV", raising=False)
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    assert selfcheck_enabled() is True
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "0")
    assert selfcheck_enabled() is False
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "1")
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    assert selfcheck_enabled() is True


def test_postconv_fusion_has_a_fail_closed_selfcheck_lane(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    monkeypatch.setattr(
        kernel_selfcheck,
        "_check_gdn_postconv_inline_g",
        lambda mx_module, dtype: 0.0,
        raising=False,
    )
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["gdn_postconv_inline_g"] == "ok"


def test_gdn_postconv_selfcheck_invokes_m1_and_m2(monkeypatch) -> None:
    real_m1 = gdn_capture._a3b_compiled_target_gdn_postconv_m1_tgy4
    real_m2 = gdn_capture._a3b_compiled_target_gdn_postconv_m2_tgy4
    calls: list[int] = []

    def m1(*args, **kwargs):
        calls.append(1)
        return real_m1(*args, **kwargs)

    def m2(*args, **kwargs):
        calls.append(2)
        return real_m2(*args, **kwargs)

    monkeypatch.setattr(gdn_capture, "_a3b_compiled_target_gdn_postconv_m1_tgy4", m1)
    monkeypatch.setattr(gdn_capture, "_a3b_compiled_target_gdn_postconv_m2_tgy4", m2)

    # Bit-exact (0.0) on mlx 0.31.2; mlx 0.32.0 shifted accumulation order
    # somewhere in the stock capture vs route pair by ~1.1e-8. The production
    # gate for this lane tolerates 0.03125; keep the test far tighter so a
    # genuinely broken kernel still fails, without pinning MLX's internal
    # reduction order.
    assert kernel_selfcheck._check_gdn_postconv_inline_g(mx, mx.bfloat16) <= 1e-6
    assert calls == [1, 2]


@pytest.mark.parametrize(
    "corruption",
    ("m1_output", "m1_state", "m2_output", "m2_state"),
)
def test_postconv_selfcheck_rejects_output_or_captured_state_corruption(
    monkeypatch,
    corruption,
) -> None:
    observed_states = []
    mode = {"value": "exact"}

    def stock(q, k, v, a, b, state, mask, gdn):
        mx.eval(state)
        assert bool(mx.all(mx.isfinite(state)).item())
        assert float(mx.abs(state).max()) > 0.0
        observed_states.append(state)
        logical_m = int(q.shape[1])
        return (
            mx.zeros((1, logical_m, 32, 128), dtype=mx.bfloat16),
            mx.zeros((1, logical_m, 32, 128, 128), dtype=mx.float32),
        )

    def candidate(logical_m, conv_out, a, b, state, *, A_log, dt_bias):
        out = mx.zeros((1, logical_m, 32, 128), dtype=mx.bfloat16)
        states = mx.zeros((1, logical_m, 32, 128, 128), dtype=mx.float32)
        if mode["value"] == f"m{logical_m}_output":
            out = out + 0.125
        if mode["value"] == f"m{logical_m}_state":
            states = states + 0.125
        return out, states

    def m1(*args, **kwargs):
        return candidate(1, *args, **kwargs)

    def m2(*args, **kwargs):
        return candidate(2, *args, **kwargs)

    monkeypatch.setattr(gdn_capture, "_stock_gated_delta_capture", stock)
    monkeypatch.setattr(
        gdn_capture,
        "_a3b_compiled_target_gdn_postconv_m1_tgy4",
        m1,
    )
    monkeypatch.setattr(
        gdn_capture,
        "_a3b_compiled_target_gdn_postconv_m2_tgy4",
        m2,
    )

    assert kernel_selfcheck._check_gdn_postconv_inline_g(mx, mx.bfloat16) == 0.0
    mode["value"] = corruption
    assert kernel_selfcheck._check_gdn_postconv_inline_g(mx, mx.bfloat16) > 0.03125
    mx.eval(*observed_states)
    assert len(observed_states) == 4
    assert all(
        bool(mx.array_equal(observed_states[0], state).item())
        for state in observed_states[1:]
    )


def test_gdn_postconv_m2_primary_state_continues_exactly_through_m1() -> None:
    conv_values = mx.arange(3 * 8192, dtype=mx.float32).reshape(1, 3, 8192)
    conv_rows = (mx.sin(conv_values * 0.013) * 0.5).astype(mx.bfloat16)
    gate_values = mx.arange(3 * 32, dtype=mx.float32).reshape(1, 3, 32)
    a_rows = (mx.sin(gate_values * 0.11) * 0.5).astype(mx.bfloat16)
    b_rows = (mx.cos(gate_values * 0.07) * 0.5).astype(mx.bfloat16)
    state_values = mx.arange(32 * 128 * 128, dtype=mx.float32).reshape(
        1, 32, 128, 128
    )
    state = mx.sin(state_values * 0.001) * 0.1
    A_log = mx.linspace(0.0, 2.0, 32).astype(mx.bfloat16)
    dt_bias = mx.linspace(-5.0, -3.0, 32).astype(mx.bfloat16)

    conv_ad = conv_rows[:, :2]
    a_ad = a_rows[:, :2]
    b_ad = b_rows[:, :2]
    conv_ac = mx.stack([conv_rows[:, 0], conv_rows[:, 2]], axis=1)
    a_ac = mx.stack([a_rows[:, 0], a_rows[:, 2]], axis=1)
    b_ac = mx.stack([b_rows[:, 0], b_rows[:, 2]], axis=1)
    conv_c = conv_rows[:, 2:3]
    a_c = a_rows[:, 2:3]
    b_c = b_rows[:, 2:3]

    out_ad, states_ad = gdn_capture._a3b_compiled_target_gdn_postconv_m2_tgy4(
        conv_ad,
        a_ad,
        b_ad,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    out_ac, states_ac = gdn_capture._a3b_compiled_target_gdn_postconv_m2_tgy4(
        conv_ac,
        a_ac,
        b_ac,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    out_c, states_c = gdn_capture._a3b_compiled_target_gdn_postconv_m1_tgy4(
        conv_c,
        a_c,
        b_c,
        states_ad[:, 0, :, :, :],
        A_log=A_log,
        dt_bias=dt_bias,
    )
    mx.eval(out_ad, states_ad, out_ac, states_ac, out_c, states_c)

    assert kernel_selfcheck._max_abs_diff(mx, out_c[:, 0], out_ac[:, 1]) == 0.0
    assert kernel_selfcheck._max_abs_diff(mx, states_c[:, 0], states_ac[:, 1]) == 0.0


def test_health_payload_before_any_run_is_safe() -> None:
    payload = report_for_health()
    assert payload == {"ran": False}
    json.dumps(payload)
