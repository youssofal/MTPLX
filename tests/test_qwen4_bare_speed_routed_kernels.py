"""Bare-Speed Q4/g64 routed kernels: instantiation, contracts, decline policy.

CPU-only. Source-string checks, routed-contract membership, and the install
wrapper's default-vs-strict failure policy (via a monkeypatched impl). Nothing
here dispatches Metal -- the g64 parity self-check runs under the GPU flock.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.kernels import qwen4_m4_routed_down as down
from mtplx.kernels import qwen4_m4_routed_glu as glu
from mtplx import qwen4_m4_stage3 as stage3


def _only_group_size_line_differs(mod) -> None:
    s32 = mod.source(32)
    s64 = mod.source(64)
    assert "constant constexpr uint GROUP_SIZE = 32;" in s32
    assert "constant constexpr uint GROUP_SIZE = 64;" in s64
    assert "GROUP_SIZE = 32;" not in s64
    # The g64 variant is the g32 kernel with ONLY the GROUP_SIZE constexpr
    # changed; every stride derives from it and the 4-bit packing is identical.
    assert s32.replace("GROUP_SIZE = 32;", "GROUP_SIZE = @;") == s64.replace(
        "GROUP_SIZE = 64;", "GROUP_SIZE = @;"
    )


def test_routed_glu_g64_source_is_g32_with_only_group_size_changed() -> None:
    _only_group_size_line_differs(glu)


def test_routed_down_g64_source_is_g32_with_only_group_size_changed() -> None:
    _only_group_size_line_differs(down)


def test_routed_kernels_default_to_g32() -> None:
    # The default preserves the Optimized-Speed geometry exactly.
    assert "constant constexpr uint GROUP_SIZE = 32;" in glu.source()
    assert "constant constexpr uint GROUP_SIZE = 32;" in down.source()


def test_routed_contracts_accept_g32_and_g64() -> None:
    assert {c[1] for c in stage3._ROUTED_GU_CONTRACTS} == {32, 64}
    assert {c[1] for c in stage3._ROUTED_DOWN_CONTRACTS} == {32, 64}
    # Q4 in both, so the packed weight columns are identical; g64 halves the
    # scale/bias group count.
    gu32 = next(c for c in stage3._ROUTED_GU_CONTRACTS if c[1] == 32)
    gu64 = next(c for c in stage3._ROUTED_GU_CONTRACTS if c[1] == 64)
    assert gu32[0] == gu64[0] == 4
    assert gu32[6] == gu64[6] == (512, 1280, 320)
    assert gu32[7] == (512, 1280, 80)
    assert gu64[7] == (512, 1280, 40)
    dn32 = next(c for c in stage3._ROUTED_DOWN_CONTRACTS if c[1] == 32)
    dn64 = next(c for c in stage3._ROUTED_DOWN_CONTRACTS if c[1] == 64)
    assert dn32[6] == dn64[6] == (512, 2560, 80)
    assert dn32[7] == (512, 2560, 20)
    assert dn64[7] == (512, 2560, 10)


def _runtime(monkeypatch, group_size: int):
    monkeypatch.setattr(stage3, "_routed_group_size", lambda text: group_size)
    return SimpleNamespace(model=SimpleNamespace())


def test_stage3_g64_declines_to_stock_under_default_arming(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("layer 7 self-check failed: dmax=0.01")

    monkeypatch.setattr(stage3, "_install_qwen4_m4_stage3_impl", _boom)
    monkeypatch.setattr(stage3, "strict_claims", lambda: False)
    rt = _runtime(monkeypatch, 64)

    report = stage3.install_qwen4_m4_stage3(
        rt,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=False,
        routed_glu_enabled=True,
    )
    assert report["installed"] is False
    assert report["reason"] == "q4_g64_declined_to_stock"
    assert report["group_size"] == 64
    assert "dmax" in report["detail"]
    assert rt.qwen4_m4_stage3_report is report


def test_stage3_g64_fails_closed_when_strict(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("layer 7 self-check failed")

    monkeypatch.setattr(stage3, "_install_qwen4_m4_stage3_impl", _boom)
    monkeypatch.setattr(stage3, "strict_claims", lambda: True)
    rt = _runtime(monkeypatch, 64)

    with pytest.raises(RuntimeError):
        stage3.install_qwen4_m4_stage3(
            rt,
            routed_down_reduce_enabled=True,
            routed_down_residual_tail_enabled=False,
            routed_glu_enabled=True,
        )


def test_stage3_g32_never_declines(monkeypatch) -> None:
    # Optimized-Speed (g32) keeps its raise-always contract regardless of strict.
    def _boom(*args, **kwargs):
        raise RuntimeError("layer 7 self-check failed")

    monkeypatch.setattr(stage3, "_install_qwen4_m4_stage3_impl", _boom)
    monkeypatch.setattr(stage3, "strict_claims", lambda: False)
    rt = _runtime(monkeypatch, 32)

    with pytest.raises(RuntimeError):
        stage3.install_qwen4_m4_stage3(
            rt,
            routed_down_reduce_enabled=True,
            routed_down_residual_tail_enabled=False,
            routed_glu_enabled=True,
        )


def test_stage3_g64_engages_when_impl_succeeds(monkeypatch) -> None:
    sentinel = {"installed": True, "group_size": 64, "paired_routed_glu": True}
    monkeypatch.setattr(
        stage3, "_install_qwen4_m4_stage3_impl", lambda *a, **k: sentinel
    )
    monkeypatch.setattr(stage3, "strict_claims", lambda: False)
    rt = _runtime(monkeypatch, 64)

    report = stage3.install_qwen4_m4_stage3(
        rt,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=False,
        routed_glu_enabled=True,
    )
    assert report is sentinel
