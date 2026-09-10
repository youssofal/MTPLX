"""The m4_stage3 /health boundary label derives q<bits>g<group> from the pack.

Before this fix the boundary + routed labels were hardcoded "q4g32", so a
Bare-Speed (Q4/g64) window mislabelled its routed geometry at /health. The
label is now derived from the installed routed projection's bits/group_size.
CPU-only: _installation_report is pure string/dict logic (no GPU).
"""

from __future__ import annotations

from mtplx.qwen4_m4_stage3 import _installation_report


def _report(*, routed_group_size, routed_bits, routed_glu_enabled,
            routed_down_residual_tail_enabled):
    return _installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=False,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        routed_group_size=routed_group_size,
        routed_bits=routed_bits,
    )


def test_bare_speed_q4_g64_paired_glu_label():
    r = _report(routed_group_size=64, routed_bits=4, routed_glu_enabled=True,
                routed_down_residual_tail_enabled=True)
    assert r["boundary"] == "paired_routed_q4g64_glu_reduce_shared_add_mlp_residual"
    assert r["routed"] == "stock_q4/g64"


def test_optimized_speed_q4_g32_paired_glu_label_unchanged():
    r = _report(routed_group_size=32, routed_bits=4, routed_glu_enabled=True,
                routed_down_residual_tail_enabled=True)
    assert r["boundary"] == "paired_routed_q4g32_glu_reduce_shared_add_mlp_residual"
    assert r["routed"] == "stock_q4/g32"


def test_bare_speed_q4_g64_routed_down_residual_tail_label():
    r = _report(routed_group_size=64, routed_bits=4, routed_glu_enabled=False,
                routed_down_residual_tail_enabled=True)
    assert r["boundary"] == "routed_q4g64_reduce_shared_add_mlp_residual"
    assert r["routed"] == "stock_q4/g64"
