from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest


class _ArraySpec:
    def __init__(self, shape, dtype) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype

    def reshape(self, *shape):
        return _ArraySpec(shape, self.dtype)


def _bf16(values):
    """Round float32 values to BF16 with round-to-nearest-even."""

    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return ((words + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _cpu_residual_tail(block_out, hyper, inject):
    block_out = _bf16(block_out)
    hyper = _bf16(hyper)
    inject = _bf16(inject)
    products = _bf16(block_out[:, None, :] * inject[:, :, None])
    return _bf16(hyper + products).reshape(hyper.shape[0], -1)


def _cpu_combined_tail(reduced_routed, shared_down, shared_factor, hyper, inject):
    reduced_routed = _bf16(reduced_routed)
    shared_down = _bf16(shared_down)
    shared_factor = _bf16(shared_factor)
    gated_shared = _bf16(shared_factor[:, None] * shared_down)
    block_out = _bf16(reduced_routed + gated_shared)
    return _cpu_residual_tail(block_out, hyper, inject)


def _valid_layer():
    return SimpleNamespace(
        mlp=object(),
        mlp_hyper_connection=SimpleNamespace(
            hc_count=4,
            hidden_size=2560,
            hc_norm=SimpleNamespace(
                weight=_ArraySpec((4 * 2560,), mx.bfloat16),
            ),
            block_inject_weight=SimpleNamespace(
                weight=_ArraySpec((4, 4 * 2560), mx.bfloat16),
            ),
        ),
    )


def test_residual_source_preserves_combine_and_bf16_decoder_order() -> None:
    from mtplx.kernels.qwen4_m4_stage3 import residual_source, source

    stock = source()
    fused = residual_source()

    reduction = "bfloat routed_products[TOP_K]"
    assert reduction in stock
    assert reduction in fused
    assert "hyper" not in stock
    assert "inject" not in stock
    assert "bfloat block_out = bfloat(" in fused
    assert (
        "bfloat product = bfloat(\n"
        "                float(block_out) * float(inject_value));"
    ) in fused
    assert (
        "output[hidden_index] = bfloat(\n"
        "                float(hyper[hidden_index]) + float(product));"
    ) in fused
    assert fused.index("bfloat product") < fused.index("output[hidden_index]")


def test_cpu_reference_detects_reassociated_product_and_maps_streams() -> None:
    block_out = np.array([[-3.0, 1.0]], dtype=np.float32)
    inject = np.array([[-2.915, 0.5, -1.0, 2.0]], dtype=np.float32)
    hyper = np.array(
        [
            [
                [-2.845, 10.0],
                [1.0, 20.0],
                [2.0, 30.0],
                [3.0, 40.0],
            ]
        ],
        dtype=np.float32,
    )

    actual = _cpu_residual_tail(block_out, hyper, inject)
    staged_first_value = np.float32(5.90625)
    assert actual.shape == (1, 8)
    assert actual[0, 0] == staged_first_value
    assert actual.reshape(1, 4, 2)[0, :, 1].tolist() == [
        7.0625,
        20.5,
        29.0,
        42.0,
    ]

    no_product_round = _bf16(
        _bf16(hyper[0, 0, 0])
        + _bf16(block_out[0, 0]) * _bf16(inject[0, 0])
    )
    assert no_product_round == np.float32(5.9375)
    assert actual[0, 0] != no_product_round


def test_residual_tail_geometry_reuses_one_thread_per_m4_hidden_value() -> None:
    from mtplx.kernels.qwen4_m4_stage3 import (
        launch_geometry,
        residual_launch_geometry,
    )

    assert launch_geometry() == ((4 * 2560, 1, 1), (256, 1, 1))
    assert residual_launch_geometry() == launch_geometry()


def test_residual_binding_is_six_input_bf16_construction_callable(
    monkeypatch,
) -> None:
    from mtplx.kernels import qwen4_m4_stage3 as kernel_module

    definition = {}
    launch = {}

    def fake_metal_kernel(**kwargs):
        definition.update(kwargs)

        def run(**kwargs):
            launch.update(kwargs)
            return (
                _ArraySpec(
                    kwargs["output_shapes"][0],
                    kwargs["output_dtypes"][0],
                ),
            )

        return run

    monkeypatch.setattr(kernel_module, "_RESIDUAL_KERNEL", None)
    monkeypatch.setattr(kernel_module.mx.fast, "metal_kernel", fake_metal_kernel)
    residual_tail = kernel_module.bind_residual_tail()
    inputs = tuple(object() for _ in range(4)) + (
        _ArraySpec((1, 4, 4 * 2560), mx.bfloat16),
        _ArraySpec((1, 4, 4), mx.bfloat16),
    )

    output = residual_tail(*inputs)
    assert output.shape == inputs[4].shape
    assert output.dtype == mx.bfloat16
    assert definition["input_names"] == [
        "routed_down",
        "shared_down",
        "route_scores",
        "shared_factor",
        "hyper",
        "inject",
    ]
    assert definition["ensure_row_contiguous"] is True
    assert launch["inputs"] == list(inputs)
    assert launch["grid"] == (4 * 2560, 1, 1)
    assert launch["threadgroup"] == (256, 1, 1)
    assert launch["output_shapes"] == [(4, 4 * 2560)]
    assert launch["output_dtypes"] == [mx.bfloat16]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda layer: setattr(layer.mlp_hyper_connection, "hc_count", 2),
            "four hyper streams",
        ),
        (
            lambda layer: setattr(
                layer.mlp_hyper_connection.block_inject_weight.weight,
                "dtype",
                mx.float16,
            ),
            "BF16 inject ownership",
        ),
        (
            lambda layer: setattr(
                layer.mlp_hyper_connection.block_inject_weight.weight,
                "shape",
                (4, 2560),
            ),
            "BF16 inject ownership",
        ),
    ],
)
def test_residual_tail_construction_admission_rejects_wrong_hyper_contract(
    monkeypatch, mutate, match
) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    layer = _valid_layer()
    mutate(layer)
    monkeypatch.setattr(stage3_module, "_validate_block_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(stage3_module, "bind_residual_tail", lambda: object())

    with pytest.raises(ValueError, match=match):
        stage3_module.bind_qwen4_m4_residual_tail(layer, index=7)


def test_residual_tail_construction_admission_returns_prebound_callable(
    monkeypatch,
) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    layer = _valid_layer()
    candidate = object()
    seen = []
    monkeypatch.setattr(
        stage3_module,
        "_validate_block_contract",
        lambda block, *, index: seen.append((block, index)),
    )
    monkeypatch.setattr(stage3_module, "bind_residual_tail", lambda: candidate)

    assert stage3_module.bind_qwen4_m4_residual_tail(layer, index=7) is candidate
    assert seen == [(layer.mlp, 7)]


def test_routed_residual_tail_source_preserves_every_bf16_boundary() -> None:
    from mtplx.kernels.qwen4_m4_routed_down import residual_tail_source, source

    routed = source()
    tail = residual_tail_source()

    assert "shared_down" not in routed
    assert "shared_factor" not in routed
    assert "hyper" not in routed
    assert "inject" not in routed
    assert "bfloat gated_shared = bfloat(" in tail
    assert "float(shared_factor[row]) * float(shared_down[index])" in tail
    assert "bfloat block_out = bfloat(" in tail
    assert "float(routed_down[index]) + float(gated_shared)" in tail
    assert "bfloat product = bfloat(" in tail
    assert "float(block_out) * float(inject_value)" in tail
    assert "output[hidden_index] = bfloat(" in tail
    assert "float(hyper[hidden_index]) + float(product)" in tail
    assert tail.index("bfloat gated_shared") < tail.index("bfloat block_out")
    assert tail.index("bfloat block_out") < tail.index("bfloat product")
    assert tail.index("bfloat product") < tail.index("output[hidden_index]")


def test_cpu_combined_tail_oracle_detects_omitted_bf16_narrowing() -> None:
    routed = np.array([[3.2917359]], dtype=np.float32)
    shared = np.array([[9.159982]], dtype=np.float32)
    factor = np.array([-29.321726], dtype=np.float32)
    inject = np.array([[-3.3585732, 0.5, -1.0, 2.0]], dtype=np.float32)
    hyper = np.array(
        [[[14.509214], [0.0], [0.0], [0.0]]],
        dtype=np.float32,
    )

    exact = _cpu_combined_tail(routed, shared, factor, hyper, inject)
    gated_shared = _bf16(_bf16(factor)[:, None] * _bf16(shared))
    without_block_narrowing = _bf16(
        _bf16(hyper)
        + _bf16(
            (_bf16(routed) + gated_shared)[:, None, :]
            * _bf16(inject)[:, :, None]
        )
    ).reshape(1, -1)
    reassociated = _bf16(
        hyper
        + (
            routed[:, None, :]
            + shared[:, None, :] * factor[:, None, None]
        )
        * inject[:, :, None]
    ).reshape(1, -1)

    assert exact[0, 0] == np.float32(908.0)
    assert without_block_narrowing[0, 0] == np.float32(912.0)
    assert not np.array_equal(exact, without_block_narrowing)
    assert not np.array_equal(exact, reassociated)


def test_routed_residual_binding_uses_exactly_two_dispatches(monkeypatch) -> None:
    from mtplx.kernels import qwen4_m4_routed_down as kernel_module

    definitions = []
    launches = []
    outputs = []

    def fake_metal_kernel(**kwargs):
        definitions.append(kwargs)

        def run(**launch):
            launches.append(launch)
            output = _ArraySpec(
                launch["output_shapes"][0],
                launch["output_dtypes"][0],
            )
            outputs.append(output)
            return (output,)

        return run

    monkeypatch.setattr(kernel_module, "_ROUTED_KERNEL", {})
    monkeypatch.setattr(kernel_module, "_RESIDUAL_TAIL_KERNEL", None, raising=False)
    monkeypatch.setattr(kernel_module.mx.fast, "metal_kernel", fake_metal_kernel)
    combined = kernel_module.bind_residual_tail()
    routed_inputs = tuple(object() for _ in range(6))
    shared_down = object()
    shared_factor = object()
    hyper = _ArraySpec((1, 4, 4 * 2560), mx.bfloat16)
    inject = _ArraySpec((1, 4, 4), mx.bfloat16)

    output = combined(
        *routed_inputs,
        shared_down,
        shared_factor,
        hyper,
        inject,
    )

    assert len(definitions) == 2
    assert definitions[0]["input_names"] == [
        "routed_h",
        "weights",
        "scales",
        "biases",
        "expert_ids",
        "route_scores",
    ]
    assert definitions[1]["input_names"] == [
        "routed_down",
        "shared_down",
        "shared_factor",
        "hyper",
        "inject",
    ]
    assert len(launches) == 2
    assert launches[1]["inputs"] == [
        outputs[0],
        shared_down,
        shared_factor,
        hyper,
        inject,
    ]
    assert output.shape == hyper.shape
    assert output.dtype == mx.bfloat16


def test_routed_residual_route_is_physical_m4_only(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    candidate = object()
    stock = object()
    layer = stage3_module._M4RoutedDownResidualTailDecoderLayer.__new__(
        stage3_module._M4RoutedDownResidualTailDecoderLayer
    )
    monkeypatch.setattr(
        stage3_module,
        "_m4_routed_down_residual_tail_layer_forward",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        stage3_module.DecoderLayer,
        "__call__",
        lambda *_args, **_kwargs: stock,
    )

    assert layer(
        SimpleNamespace(size=4 * 4 * 2560, shape=(1, 4, 4 * 2560)),
        input_ids=object(),
        ssm_mask=object(),
        cache=object(),
    ) is candidate
    for rows in (1, 2, 3):
        assert layer(
            SimpleNamespace(
                size=rows * 4 * 2560,
                shape=(1, rows, 4 * 2560),
            ),
            input_ids=object(),
            ssm_mask=object(),
            cache=object(),
        ) is stock


@pytest.mark.parametrize("is_linear", [True, False])
def test_routed_residual_layer_preserves_decoder_call_order_and_arguments(
    monkeypatch, is_linear
) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    calls = []
    input_ids = object()
    ssm_mask = object()
    cache = object()
    hidden = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
    ple_out = np.full_like(hidden, 0.5)
    attn_mixed = np.full_like(hidden, 2.0)
    attn_hyper = np.arange(32, dtype=np.float32).reshape(1, 4, 8)
    attn_inject = np.arange(16, dtype=np.float32).reshape(1, 4, 4) * 0.125
    block_out = np.full_like(hidden, 3.0)
    mlp_mixed = np.full_like(hidden, 4.0)
    mlp_hyper = np.full((1, 4, 8), 5.0, dtype=np.float32)
    mlp_inject = np.full((1, 4, 4), 0.25, dtype=np.float32)
    result = object()

    class Layer:
        def __init__(self):
            self.is_linear = is_linear
            self.mlp = object()
            self._mtplx_m4_routed_down_residual_tail = object()

        def __contains__(self, name):
            return name == "ple"

        def ple(self, value, observed_ids, observed_cache):
            calls.append(("ple", value.copy(), observed_ids, observed_cache))
            return ple_out

        def attn_hyper_connection(self, value):
            calls.append(("attn_hyper", value.copy()))
            return attn_mixed, attn_hyper, attn_inject

        def linear_attn(self, mixed, mask, observed_cache):
            calls.append(("linear", mixed, mask, observed_cache))
            return block_out

        def self_attn(self, mixed, observed_cache):
            calls.append(("self", mixed, observed_cache))
            return block_out

        def mlp_hyper_connection(self, value):
            calls.append(("mlp_hyper", value.copy()))
            return mlp_mixed, mlp_hyper, mlp_inject

    layer = Layer()

    def fake_mlp_forward(block, mixed, tail, hyper, inject):
        calls.append(("mlp", block, mixed, tail, hyper, inject))
        return result

    monkeypatch.setattr(
        stage3_module,
        "_m4_routed_down_residual_tail_forward",
        fake_mlp_forward,
    )

    actual = stage3_module._m4_routed_down_residual_tail_layer_forward(
        layer,
        hidden,
        input_ids=input_ids,
        ssm_mask=ssm_mask,
        cache=cache,
    )

    assert actual is result
    assert [call[0] for call in calls] == [
        "ple",
        "attn_hyper",
        "linear" if is_linear else "self",
        "mlp_hyper",
        "mlp",
    ]
    np.testing.assert_array_equal(calls[1][1], hidden + ple_out)
    expected_hidden = attn_hyper + (
        block_out[..., None, :] * attn_inject[..., :, None]
    ).reshape(*attn_hyper.shape)
    np.testing.assert_array_equal(calls[3][1], expected_hidden)
    if is_linear:
        assert calls[2][1] is attn_mixed
        assert calls[2][2] is ssm_mask
        assert calls[2][3] is cache
    else:
        assert calls[2][1] is attn_mixed
        assert calls[2][2] is cache
    assert calls[4][1] is layer.mlp
    assert calls[4][2] is mlp_mixed
    assert calls[4][3] is layer._mtplx_m4_routed_down_residual_tail
    assert calls[4][4] is mlp_hyper
    assert calls[4][5] is mlp_inject


def test_successful_combined_install_assigns_all_48_exact_owners(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    class Layer:
        pass

    class CombinedLayer(Layer):
        pass

    class Block:
        pass

    monkeypatch.setattr(
        stage3_module, "_M4RoutedDownResidualTailDecoderLayer", CombinedLayer
    )
    stage3 = object()
    combined = object()
    layers = tuple(Layer() for _ in range(48))
    blocks = tuple(Block() for _ in range(48))
    plans = tuple(
        (layer, block, stage3, combined)
        for layer, block in zip(layers, blocks, strict=True)
    )

    stage3_module._install_validated_plans(
        plans,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
    )

    assert all(type(layer) is CombinedLayer for layer in layers)
    assert all(
        layer._mtplx_m4_routed_down_residual_tail is combined for layer in layers
    )
    assert all(block._mtplx_m4_stage3 is stage3 for block in blocks)
