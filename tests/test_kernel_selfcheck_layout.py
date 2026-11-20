"""#677: an mxfp8 pack's start-up log names its weight layout instead of four "raised" lines.

The fast matmul lanes read affine weights (scales and biases). On an mxfp8
pack (E8M0 scales stored as uint8, no biases) the self-check built its
fixtures in the scales' dtype, uint8, so every matmul probe failed inside
mx.quantize and the log said "raised" four times, which reads like a kernel
bug. The lanes now stay off without a probe under one line naming the
layout, and the fixtures of the other lanes are built in the compute dtype.

Host-only: no probe below reaches a kernel.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import kernel_selfcheck
from mtplx.kernel_selfcheck import lane_disabled, run_kernel_selfcheck

QMM_LANES = ("qmm_m4", "qmm_m4_wide", "qmm_m6", "qmm_m6_wide", "qmm_m16_nax")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in ("MTPLX_GQA_PACKED_SDPA", "MTPLX_QWEN_ROW_OWNED_ROUTER", "MTPLX_QWEN_COMBINE_TAIL",
                "MTPLX_FUSE_GDN_POST_CONV", "MTPLX_A3B_WHOLE_MOE_FUSION"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    kernel_selfcheck._reset_for_tests()
    yield
    kernel_selfcheck._reset_for_tests()


class _Layer(SimpleNamespace):
    def __getitem__(self, name):
        return getattr(self, name)


def _model(**projection):
    q_proj = _Layer(**projection)
    layer = SimpleNamespace(self_attn=SimpleNamespace(q_proj=q_proj))
    return SimpleNamespace(model=SimpleNamespace(layers=[layer]))


MXFP8 = _model(bits=8, group_size=32, mode="mxfp8", scales=mx.zeros((4,), mx.uint8))
AFFINE = _model(bits=4, group_size=64, mode="affine", scales=mx.zeros((4,), mx.bfloat16),
                biases=mx.zeros((4,), mx.bfloat16))
NO_MODE = _model(bits=4, group_size=64, scales=mx.zeros((4,), mx.float16), biases=mx.zeros((4,), mx.float16))


def test_the_fixture_dtype_is_the_compute_dtype_not_the_scale_bytes():
    assert kernel_selfcheck._model_quant_signature(MXFP8) == (mx.bfloat16, 8, 32)
    assert kernel_selfcheck._model_quant_signature(AFFINE) == (mx.bfloat16, 4, 64)
    assert kernel_selfcheck._model_quant_signature(NO_MODE) == (mx.float16, 4, 64)


def test_the_layout_is_read_from_the_layer():
    assert kernel_selfcheck._model_matmul_layout(MXFP8) == "mxfp8"
    assert kernel_selfcheck._model_matmul_layout(AFFINE) == "affine"
    assert kernel_selfcheck._model_matmul_layout(NO_MODE) == "affine"  # no mode: as before


def test_an_mxfp8_pack_gets_one_line_naming_its_layout_and_its_matmul_lanes_stay_off(monkeypatch, caplog):
    def no_probe(*_args, **_kwargs):
        raise AssertionError("a matmul lane was probed on a layout it cannot read")

    monkeypatch.setattr(kernel_selfcheck, "_check_qmm_lane", no_probe)
    with caplog.at_level(logging.WARNING):
        report = run_kernel_selfcheck(mx.bfloat16, 8, 32, matmul_layout="mxfp8")

    for lane in QMM_LANES:
        assert report["lanes"][lane] == "fallback"
        assert lane_disabled(lane)  # the call sites keep the stock matmul
    lines = [r.getMessage() for r in caplog.records if "kernel selfcheck" in r.getMessage()]
    assert lines == [
        "[mtplx] kernel selfcheck: the fast matmul kernels do not support this pack's "
        "mxfp8 weight layout; matmuls use the stock kernels"
    ]
    assert not any("raised" in line for line in lines)
