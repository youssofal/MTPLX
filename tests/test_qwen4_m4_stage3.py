from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest


class _ArraySpec:
    def __init__(self, shape, dtype) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


class _RoutedOwner:
    pass


class _SharedOwner:
    pass


def _projection(bits, group_size, weight_shape, metadata_shape):
    return SimpleNamespace(
        bits=bits,
        group_size=group_size,
        mode="affine",
        weight=_ArraySpec(weight_shape, mx.uint32),
        scales=_ArraySpec(metadata_shape, mx.bfloat16),
        biases=_ArraySpec(metadata_shape, mx.bfloat16),
    )


def _valid_block(monkeypatch):
    from mtplx import qwen4_m4_stage3 as stage3_module

    monkeypatch.setattr(stage3_module, "SparseMoeBlock", SimpleNamespace)
    monkeypatch.setattr(stage3_module, "_FusedGateUpSwitchGLU", _RoutedOwner)
    monkeypatch.setattr(stage3_module, "_FusedGateUpMLP", _SharedOwner)

    routed = _RoutedOwner()
    routed.bits = 4
    routed.group_size = 32
    routed.mode = "affine"
    routed.gu_weight = _ArraySpec((512, 1280, 320), mx.uint32)
    routed.gu_scales = _ArraySpec((512, 1280, 80), mx.bfloat16)
    routed.gu_biases = _ArraySpec((512, 1280, 80), mx.bfloat16)
    routed.down_proj = _projection(
        4,
        32,
        (512, 2560, 80),
        (512, 2560, 20),
    )

    shared = _SharedOwner()
    shared.bits = 8
    shared.group_size = 64
    shared.mode = "affine"
    shared.gu_weight = _ArraySpec((1280, 640), mx.uint32)
    shared.gu_scales = _ArraySpec((1280, 40), mx.bfloat16)
    shared.gu_biases = _ArraySpec((1280, 40), mx.bfloat16)
    shared.down_proj = _projection(8, 64, (2560, 160), (2560, 10))

    return SimpleNamespace(
        num_experts=512,
        top_k=10,
        norm_topk_prob=True,
        sharding_group=None,
        gate=_projection(8, 64, (512, 640), (512, 40)),
        shared_expert_gate=_projection(8, 64, (1, 640), (1, 40)),
        switch_mlp=routed,
        shared_expert=shared,
    )


def test_m4_stage3_keeps_quantized_matmuls_on_the_stock_path() -> None:
    from mtplx.kernels.qwen4_m4_stage3 import launch_geometry, source

    kernel_source = source()
    assert "constexpr uint ROWS = 4" in kernel_source
    assert "routed_down_weight" not in kernel_source
    assert "shared_down_weight" not in kernel_source
    assert launch_geometry() == ((10240, 1, 1), (256, 1, 1))


def test_m4_stage3_source_matches_mlx_bf16_column_reduction_order() -> None:
    from mtplx.kernels.qwen4_m4_stage3 import source

    kernel_source = source()
    assert "bfloat routed_products[TOP_K]" in kernel_source
    assert "routed_products[0]" in kernel_source
    assert "routed_products[8]" in kernel_source
    assert "routed_products[1]" in kernel_source
    assert "routed_products[9]" in kernel_source
    assert "for (uint slot = 2; slot < 8; ++slot)" in kernel_source


def test_m4_stage3_installer_keeps_both_down_projections_stock() -> None:
    import inspect

    from mtplx import qwen4_m4_stage3

    forward_source = inspect.getsource(qwen4_m4_stage3._m4_forward)
    install_source = inspect.getsource(qwen4_m4_stage3._install_qwen4_m4_stage3_impl)
    assert "routed.down_proj(" in forward_source
    assert "shared.down_proj(" in forward_source
    assert "bind()" in install_source


def test_m4_routed_down_reduce_copies_exact_q4_g32_qmv_and_bf16_epilogue() -> None:
    from mtplx.kernels.qwen4_m4_routed_down import source

    kernel_source = source()
    assert "constant constexpr uint K = 640;" in kernel_source
    assert "constant constexpr ushort SLOT_ORDER[TOP_K]" in kernel_source
    assert "constexpr uint K = 640;" in kernel_source
    assert "constexpr uint GROUP_SIZE = 32;" in kernel_source
    assert "constexpr uint VALUES_PER_THREAD = 8;" in kernel_source
    assert "x_thread[i + 1] = x[i + 1] / 16.0f;" in kernel_source
    assert "x_thread[i + 2] = x[i + 2] / 256.0f;" in kernel_source
    assert "x_thread[i + 3] = x[i + 3] / 4096.0f;" in kernel_source
    assert "const device ushort* ws = (const device ushort*)w;" in kernel_source
    assert "return scale * accum + sum * bias;" in kernel_source
    assert "result[out] = simd_sum(result[out]);" in kernel_source
    assert "bfloat down_value = bfloat(result[out]);" in kernel_source
    assert "float(down_value) * float(route_scores" in kernel_source
    assert "constexpr ushort SLOT_ORDER[TOP_K] = {0, 8, 1, 9, 2, 3, 4, 5, 6, 7};" in kernel_source
    assert "routed_value[out] = bfloat(float(pending[out]) + float(product));" in kernel_source
    assert "bfloat second = bfloat(float(pending[out]) + float(product));" in kernel_source
    assert "routed_value[out] = bfloat(float(second) + float(routed_value[out]));" in kernel_source
    assert "routed_value[out] = bfloat(float(product) + float(routed_value[out]));" in kernel_source


def test_m4_routed_down_reduce_keeps_shared_branch_in_a_separate_exact_tail() -> None:
    from mtplx.kernels.qwen4_m4_routed_down import tail_source

    kernel_source = tail_source()
    assert "float(shared_factor[row])" in kernel_source
    assert "float(shared_down[index])" in kernel_source
    assert "bfloat gated_shared = bfloat(" in kernel_source
    assert "bfloat(float(routed_down[index]) + float(gated_shared))" in kernel_source


def test_m4_routed_down_reduce_geometry_retains_1280_live_threadgroups() -> None:
    from mtplx.kernels.qwen4_m4_routed_down import launch_geometry

    assert launch_geometry() == ((320 * 64, 4, 1), (64, 1, 1))


def test_m4_routed_down_reduce_bypasses_only_stock_routed_down() -> None:
    import inspect

    from mtplx import qwen4_m4_stage3

    forward_source = inspect.getsource(qwen4_m4_stage3._m4_routed_down_reduce_forward)
    assert "routed.down_proj(" not in forward_source
    assert "routed.down_proj.weight" in forward_source
    assert "shared.down_proj(" in forward_source
    assert "mx.sigmoid(block.shared_expert_gate(x))" in forward_source
    assert "route_scores.reshape(4, 10).astype(mx.bfloat16)" in forward_source


def test_m4_routed_down_reduce_routes_only_physical_m4(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    candidate = object()
    stock = object()
    block = stage3_module._M4RoutedDownReduceSparseMoeBlock.__new__(
        stage3_module._M4RoutedDownReduceSparseMoeBlock
    )
    object.__setattr__(block, "_mtplx_m4_routed_down_reduce", object())
    monkeypatch.setattr(
        stage3_module,
        "_m4_routed_down_reduce_forward",
        lambda *_args: candidate,
    )
    monkeypatch.setattr(
        stage3_module.SparseMoeBlock,
        "__call__",
        lambda *_args: stock,
    )

    assert block(SimpleNamespace(size=4 * 2560, shape=(1, 4, 2560))) is candidate
    for rows in (1, 2, 3):
        assert (
            block(SimpleNamespace(size=rows * 2560, shape=(1, rows, 2560)))
            is stock
        )


def test_m4_stage3_flag_is_construction_bound(monkeypatch) -> None:
    from mtplx.qwen4_m4_stage3 import (
        qwen4_m4_routed_down_reduce_enabled,
        qwen4_m4_routed_down_residual_tail_enabled,
        qwen4_m4_routed_glu_enabled,
        qwen4_m4_stage3_enabled,
    )

    monkeypatch.delenv("MTPLX_QWEN4_M4_STAGE3", raising=False)
    assert not qwen4_m4_stage3_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "enabled")
    assert qwen4_m4_stage3_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "disabled")
    assert not qwen4_m4_stage3_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "maybe")
    with pytest.raises(ValueError, match="is not a boolean"):
        qwen4_m4_stage3_enabled()

    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", "enabled")
    assert qwen4_m4_routed_down_reduce_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", "disabled")
    assert not qwen4_m4_routed_down_reduce_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", "maybe")
    with pytest.raises(ValueError, match="is not a boolean"):
        qwen4_m4_routed_down_reduce_enabled()

    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL", "enabled")
    assert qwen4_m4_routed_down_residual_tail_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL", "disabled")
    assert not qwen4_m4_routed_down_residual_tail_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL", "maybe")
    with pytest.raises(ValueError, match="is not a boolean"):
        qwen4_m4_routed_down_residual_tail_enabled()

    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_GLU", "enabled")
    assert qwen4_m4_routed_glu_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_GLU", "disabled")
    assert not qwen4_m4_routed_glu_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_GLU", "maybe")
    with pytest.raises(ValueError, match="is not a boolean"):
        qwen4_m4_routed_glu_enabled()


def test_m4_routed_residual_tail_requires_routed_reduction_at_construction() -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    stage3_module._validate_feature_combination(
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
    )
    with pytest.raises(ValueError, match="requires routed-down reduction"):
        stage3_module._validate_feature_combination(
            routed_down_reduce_enabled=False,
            routed_down_residual_tail_enabled=True,
        )

    stage3_module._validate_feature_combination(
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
    )
    with pytest.raises(ValueError, match="requires routed residual tail"):
        stage3_module._validate_feature_combination(
            routed_down_reduce_enabled=True,
            routed_down_residual_tail_enabled=False,
            routed_glu_enabled=True,
        )


def test_m4_child_flags_require_stage3_at_runtime_construction(monkeypatch) -> None:
    import inspect

    from mtplx import qwen4_m4_stage3 as stage3_module
    from mtplx import runtime as runtime_module

    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "0")
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", "1")
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL", "0")
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_GLU", "0")
    with pytest.raises(ValueError, match="require M4 stage3"):
        stage3_module.qwen4_m4_stage3_flags()

    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL", "1")
    with pytest.raises(ValueError, match="require M4 stage3"):
        stage3_module.qwen4_m4_stage3_flags()

    runtime_source = "".join(inspect.getsource(runtime_module.load).split())
    assert "qwen4_m4_stage3_flags()" in runtime_source
    assert "routed_down_reduce_enabled=routed_reduce_enabled" in runtime_source
    assert (
        "routed_down_residual_tail_enabled="
        "residual_tail_enabled" in runtime_source
    )
    assert "routed_glu_enabled=routed_glu_enabled" in runtime_source


def test_m4_routed_residual_tail_keeps_shared_projection_stock() -> None:
    import inspect

    from mtplx import qwen4_m4_stage3

    forward_source = inspect.getsource(
        qwen4_m4_stage3._m4_routed_down_residual_tail_forward
    )
    assert "routed.down_proj(" not in forward_source
    assert "shared.down_proj(" in forward_source
    assert "mx.sigmoid(block.shared_expert_gate(x))" in forward_source
    assert "hyper" in forward_source
    assert "inject" in forward_source


def test_m4_paired_routed_gu_clones_q4_g32_qmv_and_bf16_glu_boundary() -> None:
    from mtplx.kernels.qwen4_m4_routed_glu import launch_geometry, source

    kernel_source = source()
    assert "constexpr uint ROWS = 4" in kernel_source
    assert "constexpr uint TOP_K = 10" in kernel_source
    assert "constexpr uint HIDDEN = 2560" in kernel_source
    assert "constexpr uint INTERMEDIATE = 640" in kernel_source
    assert "constexpr uint GROUP_SIZE = 32" in kernel_source
    assert "constexpr uint VALUES_PER_THREAD = 16" in kernel_source
    assert "constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32" in kernel_source
    assert "float gate_result[OUTPUTS_PER_SIMD] = {0.0f}" in kernel_source
    assert "float up_result[OUTPUTS_PER_SIMD] = {0.0f}" in kernel_source
    assert "gate_result[out] += qdot_q4(" in kernel_source
    assert "up_result[out] += qdot_q4(" in kernel_source
    assert "gate_result[out] = simd_sum(gate_result[out])" in kernel_source
    assert "up_result[out] = simd_sum(up_result[out])" in kernel_source
    assert "bfloat gate_value = bfloat(gate_result[out])" in kernel_source
    assert "bfloat up_value = bfloat(up_result[out])" in kernel_source
    assert "bfloat sigmoid_value" in kernel_source
    assert "bfloat silu = bfloat(gate_value * sigmoid_value)" in kernel_source
    assert "bfloat(silu * up_value)" in kernel_source
    assert launch_geometry() == ((5120, 40, 1), (64, 1, 1))


def test_m4_residual_route_uses_paired_gu_producer_without_hot_fallback() -> None:
    import inspect

    from mtplx import qwen4_m4_stage3

    forward_source = inspect.getsource(
        qwen4_m4_stage3._m4_paired_routed_glu_residual_tail_forward
    )
    assert "routed._gu(" not in forward_source
    assert "routed_gu_activation(" in forward_source
    assert "routed.gu_weight" in forward_source
    assert "routed.gu_scales" in forward_source
    assert "routed.gu_biases" in forward_source
    assert "try:" not in forward_source
    assert "except" not in forward_source

    install_source = inspect.getsource(qwen4_m4_stage3._install_qwen4_m4_stage3_impl)
    plan_source = inspect.getsource(qwen4_m4_stage3._build_install_plans)
    mutation_source = inspect.getsource(qwen4_m4_stage3._install_validated_plans)
    assert "_validate_residual_tail_contract" in plan_source
    assert "_M4RoutedDownResidualTailDecoderLayer" in mutation_source
    assert "_install_validated_plans" in install_source
    assert install_source.index("_build_install_plans(") < install_source.index(
        "stage3 = bind()"
    )


def test_m4_residual_install_requires_exact_decoder_owner(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    class ExactLayer:
        pass

    monkeypatch.setattr(stage3_module, "DecoderLayer", ExactLayer)
    stage3_module._validate_layer_owner(ExactLayer(), index=7)
    with pytest.raises(ValueError, match="wrong decoder owner"):
        stage3_module._validate_layer_owner(SimpleNamespace(), index=7)


def test_m4_exact_decoder_owner_is_required_only_for_residual_route(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    layer = SimpleNamespace(mlp=object())
    monkeypatch.setattr(
        stage3_module, "_validate_input_contract", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        stage3_module, "_validate_block_contract", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        stage3_module, "_validate_residual_tail_contract", lambda *_a, **_k: None
    )

    plans = stage3_module._build_install_plans(
        (layer,),
        stage3=object(),
        routed_down_reduce=object(),
        routed_down_residual_tail_enabled=False,
    )
    assert plans[0][0] is layer
    with pytest.raises(ValueError, match="wrong decoder owner"):
        stage3_module._build_install_plans(
            (layer,),
            stage3=object(),
            routed_down_reduce=object(),
            routed_down_residual_tail_enabled=True,
        )


def test_m4_residual_install_validation_cannot_partially_mutate(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    class ExactLayer:
        def __init__(self, index):
            self.index = index
            self.mlp = object()

    layers = tuple(ExactLayer(index) for index in range(48))
    original_types = tuple(type(layer) for layer in layers)
    monkeypatch.setattr(stage3_module, "DecoderLayer", ExactLayer)
    monkeypatch.setattr(
        stage3_module, "_validate_input_contract", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        stage3_module,
        "_validate_block_contract",
        lambda _block, *, index: (_ for _ in ()).throw(ValueError("late layer"))
        if index == 47
        else None,
    )
    monkeypatch.setattr(
        stage3_module, "_validate_residual_tail_contract", lambda *_a, **_k: None
    )

    with pytest.raises(ValueError, match="late layer"):
        stage3_module._build_install_plans(
            layers,
            stage3=object(),
            routed_down_reduce=object(),
            routed_down_residual_tail_enabled=True,
        )

    assert tuple(type(layer) for layer in layers) == original_types


def test_m4_residual_report_records_all_48_exact_layers() -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    report = stage3_module._installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
    )

    assert report["boundary"] == "routed_q4g32_reduce_shared_add_mlp_residual"
    assert report["reference_boundary"] == "retained_m4_routed_down_then_stock_residual"
    assert report["exact_layers"] == 48
    assert report["combined_residual_tail_layers"] == 48


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda block: setattr(block, "top_k", 9), "top-k"),
        (lambda block: setattr(block, "sharding_group", object()), "sharding"),
        (
            lambda block: setattr(block.switch_mlp, "group_size", 64),
            "routed fused GU",
        ),
    ],
)
def test_m4_stage3_rejects_wrong_construction_contract(
    monkeypatch, mutate, match
) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    block = _valid_block(monkeypatch)
    mutate(block)
    with pytest.raises(ValueError, match=match):
        stage3_module._validate_block_contract(block, index=7)


def test_m4_stage3_accepts_exact_construction_contract(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    stage3_module._validate_block_contract(_valid_block(monkeypatch), index=7)


def test_m4_stage3_requires_bf16_hidden_input_ownership() -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    layer = SimpleNamespace(
        mlp_hyper_connection=SimpleNamespace(
            hc_norm=SimpleNamespace(
                weight=_ArraySpec((4 * 2560,), mx.bfloat16),
            )
        )
    )
    stage3_module._validate_input_contract(layer, index=7)
    layer.mlp_hyper_connection.hc_norm.weight.dtype = mx.float16
    with pytest.raises(ValueError, match="BF16 hidden input"):
        stage3_module._validate_input_contract(layer, index=7)


def test_m4_stage3_routes_only_physical_m4_directly(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    candidate = object()
    stock = object()
    block = stage3_module._M4Stage3SparseMoeBlock.__new__(
        stage3_module._M4Stage3SparseMoeBlock
    )
    object.__setattr__(block, "_mtplx_m4_stage3", object())
    monkeypatch.setattr(stage3_module, "_m4_forward", lambda *_args: candidate)
    monkeypatch.setattr(
        stage3_module.SparseMoeBlock,
        "__call__",
        lambda *_args: stock,
    )

    assert block(SimpleNamespace(size=4 * 2560, shape=(1, 4, 2560))) is candidate
    for rows in (1, 2, 3):
        assert (
            block(SimpleNamespace(size=rows * 2560, shape=(1, rows, 2560)))
            is stock
        )
