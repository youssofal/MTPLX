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


def test_qlinear_patch_routes_6bit_verify_shapes() -> None:
    """The patch routes 6-bit verify shapes (N >= 2048 floor) through the
    hexpack kernels and leaves small-N projections on stock."""
    from mtplx import verify_kernels

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0}
    orig = verify_kernels.vk_qmm_m4_ksplit

    def counting(*a, **k):
        calls["m4"] += 1
        return orig(*a, **k)

    from mtplx.attention_context import attention_phase

    import mtplx.nax_verify  # noqa: F401  (patch reads through the module)

    verify_kernels.vk_qmm_m4_ksplit = counting
    try:
        big = nn.QuantizedLinear(512, 2048, bias=False, group_size=64, bits=6)
        small = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=6)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            mx.eval(big(x))
            assert calls["m4"] == 1, "6-bit verify shape did not route the hexpack kernel"
            mx.eval(small(x))
            assert calls["m4"] == 1, "small-N 6-bit projection must stay stock"
        with attention_phase("prefill"):
            mx.eval(big(x))
            assert calls["m4"] == 1, "prefill must stay stock"
    finally:
        verify_kernels.vk_qmm_m4_ksplit = orig
        uninstall_nax_qlinear_patch()


def test_m5_falls_through_to_stock_by_default(monkeypatch) -> None:
    """2026-08-22 routing fix: M=5 4-bit skips the padded m6/m16 lanes
    (three-session micro: padded-m6 +3..13% vs stock, m16 worst) unless
    MTPLX_M5_PADDED_LANE opts back in. Receipt = the b4_m5 fallback counter
    (entered the verify window, took no custom lane)."""
    from mtplx.nax_verify import _m5_padded_lane, nax_qlinear_fallback_counts

    monkeypatch.delenv("MTPLX_M5_PADDED_LANE", raising=False)
    assert _m5_padded_lane() is False
    monkeypatch.setenv("MTPLX_M5_PADDED_LANE", "1")
    assert _m5_padded_lane() is True
    monkeypatch.delenv("MTPLX_M5_PADDED_LANE", raising=False)

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        # K=512 %256==0, N=256 %32==0: m6-ksplit and m16 would both accept
        # this shape at M=5 — only the new guard keeps it on stock.
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((5, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        before = nax_qlinear_fallback_counts.get("b4_m5", 0)
        y = layer(x)
        mx.eval(y)
        assert y.shape == (5, 256)
        assert nax_qlinear_fallback_counts.get("b4_m5", 0) == before + 1
        ref = _stock(x, layer["weight"], layer["scales"], layer["biases"])
        assert mx.array_equal(y, ref), "M=5 default must be the stock result"
    finally:
        uninstall_nax_qlinear_patch()


# ---------------------------------------------------------------------------
# Multi-row lanes (2026-10-04): exact 5-row split-K lane and the pipelined m16
# tile, both behind default-off switches.
# ---------------------------------------------------------------------------


def test_multirow_switches_default_off(monkeypatch) -> None:
    from mtplx.nax_verify import m5_rows_lane_enabled, m16_pipelined_enabled

    monkeypatch.delenv("MTPLX_M5_ROWS_LANE", raising=False)
    monkeypatch.delenv("MTPLX_NAX_M16_PIPELINED", raising=False)
    assert m5_rows_lane_enabled() is False
    assert m16_pipelined_enabled() is False
    monkeypatch.setenv("MTPLX_M5_ROWS_LANE", "1")
    monkeypatch.setenv("MTPLX_NAX_M16_PIPELINED", "on")
    assert m5_rows_lane_enabled() is True
    assert m16_pipelined_enabled() is True


def test_multirow_eligibility_shape_policy() -> None:
    from mtplx.nax_verify import m5_rows_eligible, m16_pipelined_eligible

    dt = mx.bfloat16
    assert m5_rows_eligible(5, 5120, 17408, 4, 32, dt)
    assert m5_rows_eligible(5, 17408, 5120, 4, 64, dt)
    assert not m5_rows_eligible(4, 5120, 17408, 4, 32, dt)
    assert not m5_rows_eligible(6, 5120, 17408, 4, 32, dt)
    assert not m5_rows_eligible(5, 5120, 17408, 8, 32, dt)
    assert not m5_rows_eligible(5, 5120 + 32, 17408, 4, 32, dt)
    assert not m5_rows_eligible(5, 5120, 17408 + 2, 4, 32, dt)
    # pipelined tile: m16 contract plus K % 512 (whole two-step units per simdgroup)
    expect = nax_available()
    assert m16_pipelined_eligible(9, 5120, 17408, 4, 32, dt) == expect
    assert m16_pipelined_eligible(16, 17408, 5120, 4, 32, dt) == expect
    assert not m16_pipelined_eligible(9, 5120 + 256, 17408, 4, 32, dt)
    assert not m16_pipelined_eligible(17, 5120, 17408, 4, 32, dt)


@pytest.mark.parametrize("group_size", [32, 64])
def test_m5_rows_kernel_matches_stock_within_tolerance(group_size: int) -> None:
    from mtplx.nax_verify import nax_qmm_m5

    K, N = 1024, 768
    mx.random.seed(11)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=4)
    x = (mx.random.normal((5, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    y = nax_qmm_m5(x, w_q, scales, biases, group_size=group_size)
    ref = mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases, transpose=True,
        group_size=group_size, bits=4,
    )
    diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert y.shape == (5, N)
    assert diff < 0.05, f"m5 rows kernel drift too large: {diff}"


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
@pytest.mark.parametrize("group_size", [32, 64])
def test_m16_pipelined_is_bit_identical_to_default_tile(monkeypatch, group_size: int) -> None:
    """Same arithmetic and summation order: switching the schedule must not
    change a single output bit, at any row count 1..16."""
    K, N = 2048, 1024
    mx.random.seed(5)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=4)
    for m in (5, 9, 13, 16):
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        monkeypatch.delenv("MTPLX_NAX_M16_PIPELINED", raising=False)
        base = nax_qmm_m16(x, w_q, scales, biases, group_size=group_size)
        monkeypatch.setenv("MTPLX_NAX_M16_PIPELINED", "1")
        piped = nax_qmm_m16(x, w_q, scales, biases, group_size=group_size)
        mx.eval(base, piped)
        assert mx.array_equal(base, piped), f"pipelined tile differs at M={m}, gs={group_size}"


def test_m16_pipelined_falls_back_when_k_is_not_a_multiple_of_512(monkeypatch) -> None:
    """K=768 is a valid m16 shape (K % 256) but not a whole number of two-step
    units per simdgroup: the pipelined switch must keep the default kernel."""
    if not nax_available():
        pytest.skip("requires Apple G17 + macOS >= 26.2")
    K, N = 768, 256
    mx.random.seed(2)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    x = (mx.random.normal((8, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    monkeypatch.delenv("MTPLX_NAX_M16_PIPELINED", raising=False)
    base = nax_qmm_m16(x, w_q, scales, biases, group_size=64)
    monkeypatch.setenv("MTPLX_NAX_M16_PIPELINED", "1")
    piped = nax_qmm_m16(x, w_q, scales, biases, group_size=64)
    mx.eval(base, piped)
    assert mx.array_equal(base, piped)


def test_m5_rows_lane_routes_only_when_switched_on(monkeypatch) -> None:
    from mtplx.nax_verify import nax_qlinear_fallback_counts, nax_qmm_m5

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((5, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        w_q, scales, biases = layer["weight"], layer["scales"], layer["biases"]

        monkeypatch.delenv("MTPLX_M5_ROWS_LANE", raising=False)
        before = nax_qlinear_fallback_counts.get("b4_m5", 0)
        y_off = layer(x)
        mx.eval(y_off)
        assert nax_qlinear_fallback_counts.get("b4_m5", 0) == before + 1
        assert mx.array_equal(y_off, _stock(x, w_q, scales, biases))

        monkeypatch.setenv("MTPLX_M5_ROWS_LANE", "1")
        before = nax_qlinear_fallback_counts.get("b4_m5", 0)
        y_on = layer(x)
        mx.eval(y_on)
        assert nax_qlinear_fallback_counts.get("b4_m5", 0) == before, (
            "M=5 with the switch on must take the rows lane, not the stock fallback"
        )
        direct = nax_qmm_m5(x, w_q, scales, biases, group_size=64)
        mx.eval(direct)
        assert mx.array_equal(y_on, direct)
        diff = float(mx.abs(y_on.astype(mx.float32) - y_off.astype(mx.float32)).max())
        assert diff < 0.05
    finally:
        uninstall_nax_qlinear_patch()


def test_umbrella_switch_turns_on_every_multirow_lane(monkeypatch) -> None:
    from mtplx.nax_verify import (
        m5_rows_lane_enabled,
        m16_pipelined_enabled,
        m32_tile_enabled,
    )

    for name in ("MTPLX_M5_ROWS_LANE", "MTPLX_NAX_M16_PIPELINED", "MTPLX_NAX_M32_TILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MTPLX_MULTIROW_QMM", raising=False)
    assert not (m5_rows_lane_enabled() or m16_pipelined_enabled() or m32_tile_enabled())
    monkeypatch.setenv("MTPLX_NAX_M32_TILE", "1")
    assert m32_tile_enabled() and not m5_rows_lane_enabled() and not m16_pipelined_enabled()
    monkeypatch.delenv("MTPLX_NAX_M32_TILE", raising=False)
    monkeypatch.setenv("MTPLX_MULTIROW_QMM", "1")
    assert m5_rows_lane_enabled() and m16_pipelined_enabled() and m32_tile_enabled()


def test_m32_eligibility_shape_policy() -> None:
    from mtplx.nax_verify import m32_nax_eligible

    dt = mx.bfloat16
    expect = nax_available()
    assert m32_nax_eligible(17, 5120, 17408, 4, 32, dt) == expect
    assert m32_nax_eligible(32, 17408, 5120, 4, 64, dt) == expect
    assert not m32_nax_eligible(16, 5120, 17408, 4, 32, dt)
    assert not m32_nax_eligible(33, 5120, 17408, 4, 32, dt)
    assert not m32_nax_eligible(24, 5120 + 256, 17408, 4, 32, dt)
    assert not m32_nax_eligible(24, 5120, 17408, 8, 32, dt)


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
@pytest.mark.parametrize("group_size", [32, 64])
def test_m32_tile_matches_stock_and_the_m16_tile(group_size: int) -> None:
    """Rows 0..15 of the two-row-tile kernel are bit-identical to the one-tile
    kernel (same arithmetic per row); against stock the drift is bf16 rounding."""
    from mtplx.nax_verify import nax_qmm_m32

    K, N = 2048, 1024
    mx.random.seed(9)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=4)
    for m in (17, 24, 32):
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m32(x, w_q, scales, biases, group_size=group_size)
        ref = mx.quantized_matmul(
            x, w_q, scales=scales, biases=biases, transpose=True,
            group_size=group_size, bits=4,
        )
        mx.eval(y, ref)
        assert y.shape == (m, N)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 0.05, f"m32 tile drift too large at M={m}: {diff}"
        one_tile = nax_qmm_m16(x[:16], w_q, scales, biases, group_size=group_size)
        mx.eval(one_tile)
        assert mx.array_equal(y[:16], one_tile)


def test_m32_tile_routes_only_when_switched_on(monkeypatch) -> None:
    if not nax_available():
        pytest.skip("requires Apple G17 + macOS >= 26.2")
    from mtplx.nax_verify import nax_qlinear_fallback_counts, nax_qmm_m32

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        # the NAX kernels read scales and biases as the activation dtype (real
        # packs store bf16; a fresh layer initialises them as fp32)
        layer.scales = layer.scales.astype(mx.bfloat16)
        layer.biases = layer.biases.astype(mx.bfloat16)
        x = (mx.random.normal((24, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        w_q, scales, biases = layer["weight"], layer["scales"], layer["biases"]

        monkeypatch.delenv("MTPLX_NAX_M32_TILE", raising=False)
        monkeypatch.delenv("MTPLX_MULTIROW_QMM", raising=False)
        before = dict(nax_qlinear_fallback_counts)
        y_off = layer(x)
        mx.eval(y_off)
        assert nax_qlinear_fallback_counts == before, "M>16 is outside the window by default"
        assert mx.array_equal(y_off, _stock(x, w_q, scales, biases))

        monkeypatch.setenv("MTPLX_NAX_M32_TILE", "1")
        y_on = layer(x)
        mx.eval(y_on)
        direct = nax_qmm_m32(x, w_q, scales, biases, group_size=64)
        mx.eval(direct)
        assert mx.array_equal(y_on, direct)
    finally:
        uninstall_nax_qlinear_patch()
