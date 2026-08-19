"""Tests for the m4/NAX verify kernel module."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.nax_verify import (
    install_nax_qlinear_patch,
    m4_ksplit_eligible,
    m16_nax_eligible,
    nax_available,
    nax_qmm_m4,
    nax_qmm_m16,
    uninstall_nax_qlinear_patch,
)


def test_eligibility_shape_policy() -> None:
    dt = mx.bfloat16
    # m4: exact 4 rows only, no NAX hardware requirement
    assert m4_ksplit_eligible(4, 5120, 17408, 4, 64, dt)
    assert not m4_ksplit_eligible(5, 5120, 17408, 4, 64, dt)
    assert not m4_ksplit_eligible(4, 5120, 17408, 8, 64, dt)
    # m16: K % 256, N % 32, 4-bit, M in 1..16 (and NAX hardware)
    expect = nax_available()
    assert m16_nax_eligible(5, 5120, 17408, 4, 64, dt) == expect
    assert m16_nax_eligible(16, 17408, 5120, 4, 64, dt) == expect
    assert not m16_nax_eligible(17, 5120, 17408, 4, 64, dt)
    assert not m16_nax_eligible(5, 5120 + 64, 17408, 4, 64, dt)
    assert not m16_nax_eligible(5, 5120, 17408 + 8, 4, 64, dt)


def _quantized_fixture(K: int, N: int):
    mx.random.seed(3)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(w_q, scales, biases)
    return w_q, scales, biases


def _stock(x, w_q, scales, biases):
    return mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases, transpose=True, group_size=64, bits=4
    )


def test_m4_kernel_matches_stock_within_tolerance() -> None:
    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    x = (mx.random.normal((4, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    y = nax_qmm_m4(x, w_q, scales, biases, group_size=64)
    ref = _stock(x, w_q, scales, biases)
    diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert y.shape == (4, N)
    assert diff < 0.25, f"m4 kernel drift too large: {diff}"


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
def test_m16_nax_kernel_pads_and_matches_stock_within_tolerance() -> None:
    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    for m in (5, 16):
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m16(x, w_q, scales, biases, group_size=64)
        ref = _stock(x, w_q, scales, biases)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert y.shape == (m, N)
        assert diff < 0.25, f"nax16 kernel drift too large at M={m}: {diff}"


def test_qlinear_patch_routes_only_verify_shapes() -> None:
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        for m in (1, 3, 4, 8, 17, 64):
            x = (mx.random.normal((m, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
            y = layer(x)
            mx.eval(y)
            assert y.shape == (m, 256)
    finally:
        uninstall_nax_qlinear_patch()


def test_turbo_profile_carries_nax_env() -> None:
    from mtplx.profiles import PROFILES, PROFILE_CHOICES, apply_profile_env, restore_profile_env
    import os

    assert "turbo" in PROFILE_CHOICES
    profile = PROFILES["turbo"]
    assert profile.env_dict().get("MTPLX_NAX_VERIFY") == "1"
    assert profile.product_claim_eligible is False
    # Sustained env must be a subset (turbo = sustained + kernels).
    sustained = PROFILES["sustained"].env_dict()
    turbo = profile.env_dict()
    missing = {k: v for k, v in sustained.items() if turbo.get(k) != v}
    assert not missing, f"turbo drops sustained env keys: {missing}"
    previous = apply_profile_env("turbo")
    try:
        assert os.environ.get("MTPLX_NAX_VERIFY") == "1"
    finally:
        restore_profile_env(previous)
        assert os.environ.get("MTPLX_NAX_VERIFY") != "1"


def test_qlinear_patch_never_routes_in_prefill_phase() -> None:
    """Regression guard: prefill must stay on stock kernels byte-for-byte."""
    import mlx.core as mx
    from mtplx.attention_context import attention_phase
    from mtplx import nax_verify

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0, "m16": 0}
    orig_m4, orig_m16 = nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16

    def count_m4(*a, **k):
        calls["m4"] += 1
        return orig_m4(*a, **k)

    def count_m16(*a, **k):
        calls["m16"] += 1
        return orig_m16(*a, **k)

    nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16 = count_m4, count_m16
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("prefill"):
            mx.eval(layer(x))
        assert calls == {"m4": 0, "m16": 0}, f"kernels routed during prefill: {calls}"
        with attention_phase("decode_verify"):
            mx.eval(layer(x))
        assert calls["m4"] == 1, f"m4 kernel did not engage outside prefill: {calls}"
    finally:
        nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16 = orig_m4, orig_m16
        uninstall_nax_qlinear_patch()


def test_m6_kernel_matches_stock_within_tolerance() -> None:
    from mtplx.nax_verify import m6_ksplit_eligible, nax_qmm_m6

    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    for m in (5, 6):
        assert m6_ksplit_eligible(m, K, N, 4, 64, mx.bfloat16)
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m6(x, w_q, scales, biases, group_size=64)
        ref = _stock(x, w_q, scales, biases)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert y.shape == (m, N)
        assert diff < 0.25, f"m6 kernel drift too large at M={m}: {diff}"
    assert not m6_ksplit_eligible(4, K, N, 4, 64, mx.bfloat16)
    assert not m6_ksplit_eligible(7, K, N, 4, 64, mx.bfloat16)


def test_vk_6bit_hexpack_ksplit_matches_stock() -> None:
    """The 9B-tier 6-bit lane (2026-07-07): MLX packs 6-bit values
    bit-contiguously little-endian; the hexpack kernels must agree with
    stock quantized_matmul within the accumulation-order ULP band."""
    from mtplx.verify_kernels import (
        vk_eligible_ksplit,
        vk_qmm_m4_ksplit,
        vk_qmm_m6_ksplit,
    )

    K, N = 4096, 1024
    for dtype in (mx.bfloat16, mx.float16):
        for gs in (32, 64, 128):
            mx.random.seed(5)
            w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(dtype)
            w_q, scales, biases = mx.quantize(w, group_size=gs, bits=6)
            mx.eval(w_q, scales, biases)
            for m, fn in ((4, vk_qmm_m4_ksplit), (5, vk_qmm_m6_ksplit), (6, vk_qmm_m6_ksplit)):
                assert vk_eligible_ksplit(m, K, N, 6, gs, dtype)
                x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(dtype)
                y = fn(x, w_q, scales, biases, bits=6, group_size=gs)
                ref = mx.quantized_matmul(
                    x, w_q, scales=scales, biases=biases,
                    transpose=True, group_size=gs, bits=6,
                )
                diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
                assert y.shape == (m, N)
                assert diff < 0.05, f"6-bit drift {dtype} gs={gs} M={m}: {diff}"


def test_qlinear_patch_routes_6bit_verify_shapes(monkeypatch) -> None:
    """6-bit verify routing: the m5/m6 hexpack routes are on by default, the
    m4 route is opt-in, and the N >= 2048 floor plus the prefill exclusion
    hold for both.

    m4 is opt-in because on applegpu_g15s it is a measured loss on a
    27B-class trunk with the whole model resident (+20 to +26% against
    stock), while m5/m6 win 22-35% in the same sweep. See
    vk_qmm6_m4_enabled and benchmarks/repro_vk_qmm6_m4_route.py.
    """
    from mtplx import verify_kernels

    monkeypatch.delenv("MTPLX_VK_QMM6_M4", raising=False)
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0, "m6": 0}
    orig4 = verify_kernels.vk_qmm_m4_ksplit
    orig6 = verify_kernels.vk_qmm_m6_ksplit

    def counting4(*a, **k):
        calls["m4"] += 1
        return orig4(*a, **k)

    def counting6(*a, **k):
        calls["m6"] += 1
        return orig6(*a, **k)

    from mtplx.attention_context import attention_phase

    import mtplx.nax_verify  # noqa: F401  (patch reads through the module)

    verify_kernels.vk_qmm_m4_ksplit = counting4
    verify_kernels.vk_qmm_m6_ksplit = counting6
    try:
        big = nn.QuantizedLinear(512, 2048, bias=False, group_size=64, bits=6)
        small = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=6)
        x4 = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        x5 = (mx.random.normal((5, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            mx.eval(big(x4))
            assert calls["m4"] == 0, "6-bit m4 route must be opt-in"
            mx.eval(big(x5))
            assert calls["m6"] == 1, "6-bit m5 verify shape did not route the hexpack kernel"
            mx.eval(small(x5))
            assert calls["m6"] == 1, "small-N 6-bit projection must stay stock"
        with attention_phase("prefill"):
            mx.eval(big(x5))
            assert calls["m6"] == 1, "prefill must stay stock"

        # Opt in: the m4 route comes back for hardware where it wins.
        monkeypatch.setenv("MTPLX_VK_QMM6_M4", "1")
        with attention_phase("decode_verify"):
            mx.eval(big(x4))
            assert calls["m4"] == 1, "MTPLX_VK_QMM6_M4=1 did not restore the m4 route"
            mx.eval(small(x4))
            assert calls["m4"] == 1, "small-N must stay stock even when opted in"
    finally:
        verify_kernels.vk_qmm_m4_ksplit = orig4
        verify_kernels.vk_qmm_m6_ksplit = orig6
        uninstall_nax_qlinear_patch()


def test_vk_qmm6_m4_env_flag(monkeypatch) -> None:
    from mtplx.nax_verify import vk_qmm6_m4_enabled

    monkeypatch.delenv("MTPLX_VK_QMM6_M4", raising=False)
    assert vk_qmm6_m4_enabled() is False
    for on in ("1", "true", "on", "yes", " ON "):
        monkeypatch.setenv("MTPLX_VK_QMM6_M4", on)
        assert vk_qmm6_m4_enabled() is True
    for off in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("MTPLX_VK_QMM6_M4", off)
        assert vk_qmm6_m4_enabled() is False


def test_4bit_m4_route_is_unaffected_by_the_6bit_gate(monkeypatch) -> None:
    """The regression is 6-bit specific: at 4 bits the m4 split-K route is a
    measured win (-17% on the same sweep) and must still fire by default."""
    monkeypatch.delenv("MTPLX_VK_QMM6_M4", raising=False)
    install_nax_qlinear_patch()
    import mtplx.nax_verify as nv

    calls = {"n": 0}
    orig = nv.nax_qmm_m4

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    from mtplx.attention_context import attention_phase

    nv.nax_qmm_m4 = counting
    try:
        big = nn.QuantizedLinear(5120, 2048, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((4, 5120), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            mx.eval(big(x))
        assert calls["n"] == 1, "4-bit m4 route must be unaffected"
    finally:
        nv.nax_qmm_m4 = orig
        uninstall_nax_qlinear_patch()
