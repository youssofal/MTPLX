"""The fixed-M4 stage-3 lane for q2/g64 routed experts.

A Flash-Next pack whose routed experts are MLX affine 2-bit with 64-weight
groups (a ggml Q2_0 expert pack repacked without loss: scale = d, bias = -d)
runs the same stage-3 lane as the q4/g32 forge packs, through q2/g64 twins of
the routed GLU and routed-down kernels. These tests pin the twins to MLX
0.32.2's bits=2 ``gather_qmv_fast`` / ``gather_qmv`` arithmetic bit for bit,
and the contract, binder and launcher changes that admit the geometry.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from tests.test_env_flag_parsing import (
    _FLASH_NEXT_LANE_KEYS,
    _flash_next_fixed_m4_config,
    _flash_next_quantization,
)
from tests.test_qwen4_m4_stage3 import _ArraySpec, _projection, _valid_block

ROUTED_Q2G64 = {"bits": 2, "group_size": 64, "mode": "affine"}


def _u32_pack(shape):
    size = 1
    for extent in shape:
        size *= extent
    values = mx.arange(size, dtype=mx.uint32)
    return (values * mx.array(2654435761, dtype=mx.uint32)).reshape(shape)


def _metadata(shape, *, offset: int):
    size = 1
    for extent in shape:
        size *= extent
    values = mx.arange(size, dtype=mx.float32)
    return (
        mx.sin((values + float(offset)) * 0.03125) * 0.00390625
    ).reshape(shape).astype(mx.bfloat16)


def _q2g64_block(monkeypatch):
    block = _valid_block(monkeypatch)
    routed = block.switch_mlp
    routed.bits = 2
    routed.group_size = 64
    routed.gu_weight = _ArraySpec((512, 1280, 160), mx.uint32)
    routed.gu_scales = _ArraySpec((512, 1280, 40), mx.bfloat16)
    routed.gu_biases = _ArraySpec((512, 1280, 40), mx.bfloat16)
    routed.down_proj = _projection(2, 64, (512, 2560, 40), (512, 2560, 10))
    return block


# --- kernel sources --------------------------------------------------------


def test_q2g64_paired_routed_gu_decodes_mlx_bits2_gather_qmv_fast() -> None:
    from mtplx.kernels import qwen4_m4_routed_glu as q4
    from mtplx.kernels.qwen4_m4_routed_glu_q2 import launch_geometry, source

    kernel_source = source()
    assert "constexpr uint GROUP_SIZE = 64" in kernel_source
    assert "constexpr uint VALUES_PER_THREAD = 16" in kernel_source
    assert "constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32" in kernel_source
    assert "constexpr uint WEIGHT_BYTES_PER_ROW = HIDDEN / 4" in kernel_source
    # MLX quantized.h load_vector / qdot, bits == 2.
    assert "x_thread[i + 1] = x[i + 1] / 4.0f;" in kernel_source
    assert "x_thread[i + 2] = x[i + 2] / 16.0f;" in kernel_source
    assert "x_thread[i + 3] = x[i + 3] / 64.0f;" in kernel_source
    for mask in ("0x03", "0x0c", "0x30", "0xc0"):
        assert f"(ws[i] & {mask})" in kernel_source
    assert "return scale * accum + sum * bias;" in kernel_source
    # 16 values = 4 bytes per lane, 512-value blocks = 128 bytes per row.
    assert "+ lane * (VALUES_PER_THREAD / 4);" in kernel_source
    assert "gate_weights += BLOCK_SIZE / 4;" in kernel_source
    assert "up_weights += BLOCK_SIZE / 4;" in kernel_source
    # The BF16 GLU epilogue is the q4 producer's, line for line.
    epilogue = q4._SOURCE[q4._SOURCE.index("for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {\n        gate_result[out] = simd_sum"):]
    assert epilogue in kernel_source
    assert launch_geometry() == q4.launch_geometry()


def test_q2g64_routed_down_decodes_mlx_bits2_gather_qmv_with_safe_tail() -> None:
    from mtplx.kernels import qwen4_m4_routed_down as q4
    from mtplx.kernels import qwen4_m4_routed_down_q2 as q2

    for kernel_source in (q2.source(), q2.parallel_residual_source()):
        assert "constant constexpr uint K = 640;" in kernel_source
        assert "constexpr uint GROUP_SIZE = 64;" in kernel_source
        assert "constexpr uint VALUES_PER_THREAD = 16;" in kernel_source
        assert "constexpr uint WEIGHT_BYTES_PER_ROW = K / 4;" in kernel_source
        assert "x_thread[i + 3] = x[i + 3] / 64.0f;" in kernel_source
        assert "const device uchar* ws = w;" in kernel_source
        assert "qdot_q2_safe(" in kernel_source
        assert "w += BLOCK_SIZE / 4;" in kernel_source
        assert "constexpr ushort SLOT_ORDER[TOP_K] = {0, 8, 1, 9, 2, 3, 4, 5, 6, 7};" in kernel_source
    # The weight-free tails are the q4 module's own.
    assert q2.tail_source() is q4.tail_source()
    assert q2.residual_tail_source() is q4.residual_tail_source()
    assert q2.launch_geometry() == q4.launch_geometry()
    assert q2.parallel_launch_geometry() == q4.parallel_launch_geometry()


# --- Metal parity ----------------------------------------------------------


def test_q2g64_routed_glu_and_residual_tail_are_bit_exact_with_the_stock_chain() -> None:
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal parity requires the guarded GPU lane")
    from mtplx.kernels.qwen4_m4_routed_down_q2 import (
        bind as bind_routed_down_reduce,
    )
    from mtplx.kernels.qwen4_m4_routed_down_q2 import (
        bind_residual_tail,
    )
    from mtplx.kernels.qwen4_m4_routed_glu_q2 import bind as bind_routed_glu
    from mtplx.kernels.qwen4_m4_stage3 import bind as bind_combine
    from mtplx.qwen4_m4_stage3 import _same_bits

    experts = 3
    value = mx.sin(mx.arange(4 * 2560, dtype=mx.float32) * 0.001953125)
    value = value.reshape(1, 4, 2560).astype(mx.bfloat16)
    weights = _u32_pack((experts, 1280, 160))
    scales = _metadata((experts, 1280, 40), offset=3)
    biases = _metadata((experts, 1280, 40), offset=11)
    expert_ids = (mx.arange(4 * 10, dtype=mx.uint32) % experts).reshape(1, 4, 10)

    # Stock chain, as the installer's reference forward runs it.
    gu = mx.gather_qmm(
        mx.expand_dims(value, (-2, -3)),
        weights,
        scales,
        biases,
        rhs_indices=expert_ids,
        transpose=True,
        group_size=64,
        bits=2,
        mode="affine",
        sorted_indices=False,
    )
    gate, up = mx.split(gu, 2, axis=-1)
    expected_h = nn.silu(gate) * up
    actual_h = bind_routed_glu()(
        value.reshape(4, 2560), weights, scales, biases, expert_ids.reshape(4, 10)
    )
    mx.eval(expected_h, actual_h)
    assert bool(_same_bits(expected_h.reshape(4, 10, 640), actual_h).item())

    down_weights = _u32_pack((experts, 2560, 40))
    down_scales = _metadata((experts, 2560, 10), offset=17)
    down_biases = _metadata((experts, 2560, 10), offset=23)
    route_logits = mx.cos(mx.arange(40, dtype=mx.float32) * 0.125).reshape(4, 10)
    route_scores = mx.softmax(route_logits, axis=-1, precise=True).astype(mx.bfloat16)
    shared_down = mx.sin(mx.arange(4 * 2560, dtype=mx.float32) * 0.00390625)
    shared_down = shared_down.reshape(4, 2560).astype(mx.bfloat16)
    shared_factor = mx.sigmoid(mx.arange(4, dtype=mx.float32)).astype(mx.bfloat16)
    hyper = mx.cos(mx.arange(4 * 4 * 2560, dtype=mx.float32) * 0.0009765625)
    hyper = hyper.reshape(1, 4, 4 * 2560).astype(mx.bfloat16)
    inject = mx.sigmoid(mx.arange(16, dtype=mx.float32) * 0.0625)
    inject = inject.reshape(1, 4, 4).astype(mx.bfloat16)

    routed_down = mx.gather_qmm(
        expected_h,
        down_weights,
        down_scales,
        down_biases,
        rhs_indices=expert_ids,
        transpose=True,
        group_size=64,
        bits=2,
        mode="affine",
        sorted_indices=False,
    ).squeeze(-2).reshape(4, 10, 2560)
    combined = bind_combine()(routed_down, shared_down, route_scores, shared_factor)
    expected = hyper + (
        combined.reshape(1, 4, 2560)[..., None, :] * inject[..., :, None]
    ).reshape(*hyper.shape)

    down_args = (down_weights, down_scales, down_biases, expert_ids.reshape(4, 10))
    reduced = bind_routed_down_reduce()(
        actual_h, *down_args, route_scores, shared_down, shared_factor
    )
    actual = bind_residual_tail()(
        actual_h, *down_args, route_scores, shared_down, shared_factor, hyper, inject
    )
    mx.eval(combined, expected, reduced, actual)
    assert bool(_same_bits(combined, reduced).item())
    assert bool(_same_bits(expected, actual).item())


# --- construction contracts and binders ------------------------------------


def test_stage3_accepts_the_q2g64_routed_contract(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    block = _q2g64_block(monkeypatch)
    stage3_module._validate_block_contract(block, index=7)
    assert stage3_module._block_routed_geometry(block) == stage3_module.ROUTED_Q2G64


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda block: setattr(block.switch_mlp, "group_size", 32), "routed fused GU"),
        (
            lambda block: setattr(
                block.switch_mlp,
                "down_proj",
                _projection(4, 32, (512, 2560, 80), (512, 2560, 20)),
            ),
            "routed down",
        ),
        (
            lambda block: setattr(
                block.switch_mlp, "gu_scales", _ArraySpec((512, 1280, 80), mx.bfloat16)
            ),
            "routed fused GU",
        ),
    ],
)
def test_stage3_rejects_partial_q2g64_contracts(monkeypatch, mutate, match) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    block = _q2g64_block(monkeypatch)
    mutate(block)
    with pytest.raises(ValueError, match=match):
        stage3_module._validate_block_contract(block, index=7)


def test_stage3_binds_the_kernels_of_the_routed_geometry(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    for name in (
        "bind_routed_glu",
        "bind_routed_down_reduce",
        "bind_residual_tail",
        "bind_routed_glu_q2g64",
        "bind_routed_down_reduce_q2g64",
        "bind_residual_tail_q2g64",
    ):
        monkeypatch.setattr(stage3_module, name, name)
    assert stage3_module._routed_binders(stage3_module.ROUTED_Q4G32) == (
        "bind_routed_glu",
        "bind_routed_down_reduce",
        "bind_residual_tail",
    )
    assert stage3_module._routed_binders(stage3_module.ROUTED_Q2G64) == (
        "bind_routed_glu_q2g64",
        "bind_routed_down_reduce_q2g64",
        "bind_residual_tail_q2g64",
    )


def test_stage3_install_refuses_a_mixed_routed_geometry(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    q4 = _valid_block(monkeypatch)
    q2 = _q2g64_block(monkeypatch)
    layers = tuple(SimpleNamespace(mlp=q2 if i == 5 else q4) for i in range(48))
    runtime = SimpleNamespace()
    monkeypatch.setattr(
        stage3_module,
        "_text_model",
        lambda _: SimpleNamespace(model=SimpleNamespace(layers=layers)),
    )
    monkeypatch.setattr(stage3_module, "qwen4_route_kernel_enabled", lambda: False)
    monkeypatch.setattr(
        stage3_module,
        "_build_install_plans",
        lambda layers, **_: tuple((layer, layer.mlp, None, None) for layer in layers),
    )
    with pytest.raises(ValueError, match="one routed expert geometry"):
        stage3_module.install_qwen4_m4_stage3(
            runtime,
            routed_down_reduce_enabled=True,
            routed_down_residual_tail_enabled=True,
            routed_glu_enabled=True,
        )
    assert not hasattr(runtime, "qwen4_m4_stage3_report")


def test_stage3_report_names_the_q2g64_boundary() -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    report = stage3_module._installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
        routed_geometry=stage3_module.ROUTED_Q2G64,
    )
    assert report["boundary"] == "paired_routed_q2g64_glu_reduce_shared_add_mlp_residual"
    assert report["routed"] == "stock_q2/g64"
    assert report["shared"] == "stock_q8/g64"


# --- launcher: the stage-3 lane is stamped for a q2/g64 pack ----------------


def _q2g64_quantization(layers=range(48), *, routed=ROUTED_Q2G64) -> dict:
    quantization = _flash_next_quantization(lm_head_bits=8, stage3=True)
    for index in layers:
        prefix = f"language_model.model.layers.{index}.mlp.switch_mlp."
        for name in ("gate_proj", "up_proj", "down_proj"):
            quantization[prefix + name] = dict(routed)
    return quantization


@pytest.mark.parametrize(
    ("quantization", "stamped"),
    [
        (_flash_next_quantization(lm_head_bits=8, stage3=True), True),
        (_q2g64_quantization(), True),
        # One layer left at q4/g32: the kernels bind once per stack.
        (_q2g64_quantization(range(47)), False),
        # A geometry without kernels.
        (_q2g64_quantization(routed={"bits": 2, "group_size": 32, "mode": "affine"}), False),
    ],
)
def test_stage3_lane_is_stamped_for_every_routed_geometry_with_kernels(
    tmp_path, monkeypatch, quantization, stamped
) -> None:
    from mtplx.server.openai import _server_runtime_env_overrides

    for key in (*_FLASH_NEXT_LANE_KEYS, "MTPLX_QSA_GATHER", "MTPLX_FUSED_GATE_UP",
                "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL",
                "MTPLX_QWEN4_M4_ROUTED_GLU", "MTPLX_QWEN4_ROUTE_KERNEL"):
        monkeypatch.delenv(key, raising=False)
    model = tmp_path / "flash-next"
    model.mkdir()
    config = _flash_next_fixed_m4_config()
    config["quantization"] = quantization
    (model / "config.json").write_text(json.dumps(config))
    args = argparse.Namespace(
        model=str(model),
        verify_strategy="batched",
        generation_mode="mtp",
        scheduler_mode="serial",
    )
    overrides = _server_runtime_env_overrides(args, {})
    for key in (
        "MTPLX_QWEN4_M4_STAGE3",
        "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE",
        "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL",
        "MTPLX_QWEN4_M4_ROUTED_GLU",
        "MTPLX_QWEN4_ROUTE_KERNEL",
    ):
        assert (overrides.get(key) == "1") is stamped, key
