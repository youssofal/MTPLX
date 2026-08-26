from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.attention_context import attention_phase
from mtplx.kernels import deepseek_v4_wo_mxfp8 as wo
import mtplx.deepseek_v4_mia_engine as mia_engine
from mtplx.models import deepseek_v4 as deepseek_v4_model


_MIA_EXACT_MODEL = Path(
    "/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1"
)
_LAYER0_CARRIED = _MIA_EXACT_MODEL / "carried-001.safetensors"
_WO_B_WEIGHT = "layers.0.attn.wo_b.weight"
_WO_B_SCALE = "layers.0.attn.wo_b.scale"


class _StaticArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    def reshape(self, *shape):
        return _StaticArray(shape, self.dtype)


def _weights():
    return {
        "wo_a_weight": _StaticArray((8 * 1024, 4096 // 4), mx.uint32),
        "wo_a_scales": _StaticArray((8 * 1024, 4096 // 32), mx.uint8),
        "wo_b_weight": _StaticArray((4096, (8 * 1024) // 4), mx.uint32),
        "wo_b_scales": _StaticArray((4096, (8 * 1024) // 32), mx.uint8),
    }


class _MXFP8Linear:
    def __init__(self, weight_shape, scale_shape):
        self.weight = _StaticArray(weight_shape, mx.uint32)
        self.scales = _StaticArray(scale_shape, mx.uint8)
        self.biases = None
        self.group_size = 32
        self.bits = 8
        self.mode = "mxfp8"


def _attention():
    return SimpleNamespace(
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
        n_groups=8,
        o_lora_rank=1024,
        dim=4096,
        wo_a=_MXFP8Linear((8192, 1024), (8192, 128)),
        wo_b=_MXFP8Linear((4096, 2048), (4096, 256)),
        _output_projection_impl=lambda *_args: None,
    )


def test_mia_e4m3_and_ceil_ue8m0_byte_oracles_pin_source_boundaries():
    values = (
        0.0,
        2.0**-9,
        2.0**-6,
        0.5,
        1.0,
        1.5,
        448.0,
        -0.5,
        -448.0,
    )
    expected = (0x00, 0x01, 0x08, 0x30, 0x38, 0x3C, 0x7E, 0xB0, 0xFE)

    encoded = tuple(wo.mia_e4m3_encode_byte(value) for value in values)

    assert encoded == expected
    assert tuple(wo.mia_e4m3_decode_byte(raw) for raw in expected) == values
    assert wo.mia_ceil_ue8m0_scale_byte(0.0) == 127
    assert wo.mia_ceil_ue8m0_scale_byte(224.0) == 126
    assert wo.mia_ceil_ue8m0_scale_byte(448.0) == 127
    assert wo.mia_ceil_ue8m0_scale_byte(449.0) == 128


def test_installed_tp1_wo_core_owns_weights_and_never_reenters_factories(monkeypatch):
    calls = []
    decode_block_ns = []

    def kernel(name):
        def run(**kwargs):
            calls.append((name, kwargs))
            return tuple(
                _StaticArray(shape, dtype)
                for shape, dtype in zip(
                    kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
                )
            )

        return run

    monkeypatch.setattr(wo, "_inverse_rope_quant_kernel", lambda: kernel("inv_quant"))
    monkeypatch.setattr(wo, "_group_major_quant_kernel", lambda: kernel("tmp_quant"))
    monkeypatch.setattr(
        wo,
        "_mxfp8_mma_kernel",
        lambda stage, block_m: kernel(f"{stage}_bm{block_m}"),
    )
    def decode_kernel(block_n):
        decode_block_ns.append(block_n)
        return kernel(f"wo_b_decode_bn{block_n}")

    monkeypatch.setattr(wo, "_wo_b_decode_fused_quant_kernel", decode_kernel)
    monkeypatch.setattr(
        wo,
        "_wo_b_decode_quantized_kernel",
        lambda block_n: kernel(f"wo_b_decode_quantized_bn{block_n}"),
    )
    def exact_decode(weight, scales):
        assert weight is weights["wo_b_weight"]
        assert scales is weights["wo_b_scales"]

        def run(values, activation_scales):
            calls.append(
                (
                    "wo_b_exact_quantized_mxfp8",
                    {"values": values, "activation_scales": activation_scales},
                )
            )
            return _StaticArray((values.shape[0], 4096), mx.bfloat16)

        return run

    def native_decode(weight, scales):
        assert weight is weights["wo_b_weight"]
        assert scales is weights["wo_b_scales"]

        def run(values, activation_scales):
            calls.append(
                (
                    "wo_b_native_quantized_mxfp8",
                    {"values": values, "activation_scales": activation_scales},
                )
            )
            return _StaticArray((values.shape[0], 4096), mx.bfloat16)

        return run

    monkeypatch.setattr(
        wo,
        "_exact_quantized_mxfp8_wo_b",
        exact_decode,
        raising=False,
    )
    monkeypatch.setattr(
        wo,
        "_native_quantized_mxfp8_wo_b",
        native_decode,
        raising=False,
    )
    monkeypatch.setattr(
        wo,
        "_wo_a_m16_quantized_kernel",
        lambda: kernel("wo_a_m16_quantized"),
        raising=False,
    )
    weights = _weights()
    plan = wo.install_mia_tp1_wo_mxfp8(
        owner_role="target",
        max_prefill_rows=8224,
        **weights,
    )
    draft_plan = wo.install_mia_tp1_wo_mxfp8(
        owner_role="draft",
        max_prefill_rows=8224,
        **weights,
    )

    assert plan.wo_a_weight is weights["wo_a_weight"]
    assert plan.wo_a_scales is weights["wo_a_scales"]
    assert plan.wo_b_weight is weights["wo_b_weight"]
    assert plan.wo_b_scales is weights["wo_b_scales"]
    assert plan.max_prefill_rows == 8224
    assert plan.owner_role == "target"
    assert draft_plan.owner_role == "draft"
    assert decode_block_ns == []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("installed WO execution re-entered a kernel factory")

    monkeypatch.setattr(wo, "_inverse_rope_quant_kernel", forbidden)
    monkeypatch.setattr(wo, "_group_major_quant_kernel", forbidden)
    monkeypatch.setattr(wo, "_mxfp8_mma_kernel", forbidden)
    monkeypatch.setattr(wo, "_wo_b_decode_fused_quant_kernel", forbidden)
    monkeypatch.setattr(wo, "_wo_b_decode_quantized_kernel", forbidden)
    monkeypatch.setattr(
        wo, "_exact_quantized_mxfp8_wo_b", forbidden, raising=False
    )
    monkeypatch.setattr(
        wo, "_native_quantized_mxfp8_wo_b", forbidden, raising=False
    )
    monkeypatch.setattr(
        wo, "_wo_a_m16_quantized_kernel", forbidden, raising=False
    )

    for rows in (6,):
        calls.clear()
        cos = _StaticArray((1, rows, 32), mx.float32)
        sin = _StaticArray((1, rows, 32), mx.float32)
        with attention_phase("decode_verify"):
            output = plan(
                _StaticArray((1, rows, 64, 512), mx.bfloat16),
                cos,
                sin,
            )
        assert output.shape == (1, rows, 4096)
        assert [name for name, _kwargs in calls] == [
            "inv_quant",
            "wo_a_bm8",
            "tmp_quant",
            "wo_b_native_quantized_mxfp8",
        ]
        inv_call = calls[0][1]
        assert inv_call["inputs"][1] is cos
        assert inv_call["inputs"][2] is sin
        assert inv_call["output_shapes"] == [(8, rows, 4096), (8, rows, 128)]
        assert inv_call["output_dtypes"] == [mx.uint8, mx.uint8]
        assert (1, rows, 64 * 512) not in inv_call["output_shapes"]
        assert calls[1][1]["output_shapes"] == [(rows, 8, 1024)]
        assert calls[1][1]["output_dtypes"] == [mx.bfloat16]
        assert calls[2][1]["output_shapes"] == [(rows, 8192), (rows, 256)]
        assert calls[2][1]["output_dtypes"] == [mx.uint8, mx.uint8]
        assert calls[3][1]["values"].shape == (rows, 8192)
        assert calls[3][1]["activation_scales"].shape == (rows, 256)

    for owner, rows, phase, decode_owner in (
        (plan, 1, "prefill", "wo_b_native_quantized_mxfp8"),
        (draft_plan, 5, "decode_verify", "wo_b_exact_quantized_mxfp8"),
    ):
        calls.clear()
        with attention_phase(phase):
            output = owner(
                _StaticArray((1, rows, 64, 512), mx.bfloat16),
                _StaticArray((1, rows, 32), mx.float32),
                _StaticArray((1, rows, 32), mx.float32),
            )
        assert output.shape == (1, rows, 4096)
        assert [name for name, _kwargs in calls] == [
            "inv_quant",
            "wo_a_bm8",
            "tmp_quant",
            decode_owner,
        ]

    calls.clear()
    output = plan(
        _StaticArray((9, 64, 512), mx.bfloat16),
        _StaticArray((9, 32), mx.float32),
        _StaticArray((9, 32), mx.float32),
    )
    assert output.shape == (9, 4096)
    assert [name for name, _kwargs in calls] == [
        "inv_quant",
        "wo_a_bm64",
        "tmp_quant",
        "wo_b_bm64",
    ]
    assert calls[2][1]["output_shapes"] == [(9, 8192), (9, 256)]
    assert calls[2][1]["output_dtypes"] == [mx.uint8, mx.uint8]
    assert calls[3][1]["output_shapes"] == [(9, 4096)]

    calls.clear()
    output = plan(
        _StaticArray((16, 64, 512), mx.bfloat16),
        _StaticArray((16, 32), mx.float32),
        _StaticArray((16, 32), mx.float32),
    )
    assert output.shape == (16, 4096)
    assert [name for name, _kwargs in calls] == [
        "inv_quant",
        "wo_a_m16_quantized",
        "wo_b_bm64",
    ]
    assert calls[1][1]["output_shapes"] == [(16, 8192), (16, 256)]
    assert calls[1][1]["output_dtypes"] == [mx.uint8, mx.uint8]
    assert mx.bfloat16 not in calls[1][1]["output_dtypes"]


@pytest.mark.parametrize(
    ("field", "shape", "dtype"),
    (
        ("wo_a_weight", (8191, 1024), mx.uint32),
        ("wo_a_scales", (8192, 127), mx.uint8),
        ("wo_b_weight", (4096, 2047), mx.uint32),
        ("wo_b_scales", (4096, 255), mx.uint8),
        ("wo_a_weight", (8192, 1024), mx.uint8),
    ),
)
def test_tp1_wo_install_rejects_shape_or_storage_poison(
    monkeypatch, field, shape, dtype
):
    monkeypatch.setattr(wo, "_inverse_rope_quant_kernel", lambda: object())
    monkeypatch.setattr(wo, "_group_major_quant_kernel", lambda: object())
    monkeypatch.setattr(wo, "_mxfp8_mma_kernel", lambda *_args: object())
    monkeypatch.setattr(
        wo, "_wo_b_decode_fused_quant_kernel", lambda _block_n: object()
    )
    monkeypatch.setattr(
        wo, "_wo_b_decode_quantized_kernel", lambda _block_n: object()
    )
    monkeypatch.setattr(
        wo, "_wo_a_m16_quantized_kernel", lambda: object(), raising=False
    )
    weights = _weights()
    weights[field] = _StaticArray(shape, dtype)

    with pytest.raises(ValueError, match="Mia TP1 WO"):
        wo.install_mia_tp1_wo_mxfp8(
            owner_role="target",
            max_prefill_rows=8224,
            **weights,
        )


def test_wo_sources_pin_quantization_and_decode_route_contracts():
    exact_source = "\n".join(
        (
            wo._MXFP8_MMA_SOURCE,
            wo._WO_A_M16_QUANTIZED_SOURCE,
            wo._WO_B_DECODE_FUSED_QUANT_SOURCE,
            wo._WO_B_DECODE_QUANTIZED_SOURCE,
            inspect.getsource(wo.MiaTP1WOMXFP8Plan.__call__),
        )
    )
    native_source = inspect.getsource(wo._native_quantized_mxfp8_wo_b)

    assert "simdgroup_matrix<float, 8, 8>" in exact_source
    assert "row < uint(rows)" in exact_source
    assert "mia_ceil_ue8m0" in exact_source
    assert "rows <= 8" in exact_source
    assert "32768" not in exact_source
    assert "mx.quantized_matmul" not in exact_source
    assert "mx.dequantize" in native_source
    assert "mx.quantized_matmul" in native_source
    assert wo._MXFP8_MMA_SOURCE.count(
        "simdgroup_multiply_accumulate(c_left, a, b_left, c_left);"
    ) == wo._MXFP8_MMA_SOURCE.count(
        "simdgroup_multiply_accumulate(c_right, a, b_right, c_right);"
    )


def test_authentic_mia_m6_wo_b_routes_bound_native_drift():
    """The target native owner stays within the measured BF16 drift bound."""

    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    if not _LAYER0_CARRIED.is_file():
        pytest.skip("exact MiaAI TP1 artifact is not installed")

    source = mx.load(str(_LAYER0_CARRIED))
    weight = mx.contiguous(source[_WO_B_WEIGHT]).view(mx.uint32)
    block_scales = source[_WO_B_SCALE]
    scales = mx.contiguous(
        mx.repeat(mx.repeat(block_scales, 128, axis=0), 4, axis=1)[
            :4096, : 8192 // 32
        ]
    )

    indices = mx.arange(6 * 8192, dtype=mx.uint32).reshape(6, 8192)
    values = (
        ((indices * 37) % 1021).astype(mx.float32) - 510.0
    ) * 0.125
    values = mx.where(indices % 4096 == 0, -0.0, values)
    values = mx.where(indices % 4096 == 1, 448.0, values)
    values = mx.where(indices % 4096 == 2, -448.0, values)
    tmp = values.astype(mx.bfloat16)

    def project(block_n):
        threads = (block_n // 16) * 32
        (output,) = wo._wo_b_decode_fused_quant_kernel(block_n)(
            inputs=[tmp, weight, scales, 6],
            template=[("T", mx.bfloat16)],
            grid=(threads, 4096 // block_n, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(6, 4096)],
            output_dtypes=[mx.bfloat16],
        )
        return output

    baseline = project(32)
    widened = project(64)
    tmp_quantized, tmp_scales = wo._group_major_quant_kernel()(
        inputs=[tmp, 6],
        template=[("T", mx.bfloat16)],
        grid=(32, 8192 // 32, 6),
        threadgroup=(32, 1, 1),
        output_shapes=[(6, 8192), (6, 8192 // 32)],
        output_dtypes=[mx.uint8, mx.uint8],
    )
    (quantize_once,) = wo._wo_b_decode_quantized_kernel(64)(
        inputs=[tmp_quantized, tmp_scales, weight, scales, 6],
        template=[("T", mx.bfloat16)],
        grid=(128, 4096 // 64, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(6, 4096)],
        output_dtypes=[mx.bfloat16],
    )
    native = wo._native_quantized_mxfp8_wo_b(weight, scales)(
        tmp_quantized,
        tmp_scales,
    )
    mx.eval(
        baseline,
        widened,
        quantize_once,
        native,
    )

    np.testing.assert_array_equal(
        np.array(widened.view(mx.uint16)),
        np.array(baseline.view(mx.uint16)),
    )
    np.testing.assert_array_equal(
        np.array(quantize_once.view(mx.uint16)),
        np.array(baseline.view(mx.uint16)),
    )
    native_delta = np.abs(
        np.array(native.astype(mx.float32))
        - np.array(baseline.astype(mx.float32))
    )
    assert float(native_delta.max()) <= 0.03125


def test_m16_wo_a_quantizes_fp32_accumulators_directly_in_group_major_order():
    source = wo._WO_A_M16_QUANTIZED_SOURCE

    assert "threadgroup float c_tile[16u * 32u]" in source
    assert "float max_abs" in source
    assert "mia_ceil_ue8m0_exponent(max_abs)" in source
    assert "mia_e4m3_encode(c_tile[index] / scale_shared[local_row])" in source
    assert "size_t(local_row) * 8192u + group * 1024u + n" in source
    assert "size_t(tid) * 256u + group * 32u + n0 / 32u" in source
    assert "bfloat" not in source


def test_engine_receipt_describes_four_physical_wo_scratch_arrays():
    source = inspect.getsource(mia_engine.build_mia_engine_plan)

    assert '"wo_a_mxfp8_activation_values"' in source
    assert '"wo_a_mxfp8_activation_scales"' in source
    assert '"wo_b_prefill_mxfp8_values"' in source
    assert '"wo_b_prefill_mxfp8_scales"' in source
    assert '"wo_a_mxfp8_activation_scratch"' not in source
    assert '"wo_b_prefill_mxfp8_scratch"' not in source
    assert "4096 + 128" not in source
    assert "8192 + 256" not in source


def test_exact_model_install_owns_46_distinct_plans_and_native_parameters(
    monkeypatch,
):
    monkeypatch.setattr(wo, "_inverse_rope_quant_kernel", lambda: object())
    monkeypatch.setattr(wo, "_group_major_quant_kernel", lambda: object())
    monkeypatch.setattr(wo, "_mxfp8_mma_kernel", lambda *_args: object())
    monkeypatch.setattr(
        wo, "_wo_b_decode_fused_quant_kernel", lambda _block_n: object()
    )
    monkeypatch.setattr(
        wo, "_wo_b_decode_quantized_kernel", lambda _block_n: object()
    )
    monkeypatch.setattr(
        wo, "_wo_a_m16_quantized_kernel", lambda: object(), raising=False
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exact Mia install reached the retired gather route")

    monkeypatch.setattr(
        deepseek_v4_model,
        "_MiaInverseRopeGatherOLora",
        forbidden,
    )
    target = tuple(SimpleNamespace(attn=_attention()) for _ in range(43))
    draft = tuple(SimpleNamespace(attn=_attention()) for _ in range(3))
    model = SimpleNamespace(layers=target, mtp_blocks=draft)

    receipt = deepseek_v4_model.install_mia_tp1_wo_projection_routes(
        model,
        max_prefill_rows=8224,
    )

    attentions = tuple(layer.attn for layer in target + draft)
    plans = tuple(attention._output_projection_impl for attention in attentions)
    assert len(plans) == 46
    assert len({id(plan) for plan in plans}) == 46
    assert all(isinstance(plan, wo.MiaTP1WOMXFP8Plan) for plan in plans)
    assert tuple(plan.owner_role for plan in plans) == ("target",) * 43 + (
        "draft",
    ) * 3
    assert all(
        plan.wo_a_weight is attention.wo_a.weight
        and plan.wo_a_scales is attention.wo_a.scales
        and plan.wo_b_weight is attention.wo_b.weight
        and plan.wo_b_scales is attention.wo_b.scales
        for attention, plan in zip(attentions, plans, strict=True)
    )
    assert receipt["route"] == "mia_tp1_b12x_wo_mxfp8"
    assert receipt["target_attention"] == 43
    assert receipt["draft_attention"] == 3
    assert receipt["plan_count"] == receipt["unique_plan_count"] == 46
    assert receipt["plan_type"] == "MiaTP1WOMXFP8Plan"
    assert deepseek_v4_model.mia_tp1_wo_projection_receipt(model) == receipt
    with pytest.raises(ValueError, match="already installed"):
        deepseek_v4_model.install_mia_tp1_wo_projection_routes(
            model,
            max_prefill_rows=8224,
        )

    attentions[-1]._output_projection_impl = lambda *_args: None
    with pytest.raises(ValueError, match="Mia TP1 WO"):
        deepseek_v4_model.mia_tp1_wo_projection_receipt(model)
