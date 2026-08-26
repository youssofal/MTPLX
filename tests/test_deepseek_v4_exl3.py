from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from safetensors import safe_open

import mlx.core as mx

import mtplx.deepseek_v4_exl3 as exl3
import mtplx.models.deepseek_v4 as target_module
from mtplx.deepseek_v4_exl3 import (
    EXL3SwitchGLU,
    decode_mcg_trellis_tile,
    exl3_mcg_grouped_mma,
    exl3_mcg_grouped_qmv,
    exl3_mcg_qmv,
    load_indexed_safetensors,
    load_mia_exl3_dspark_model,
    sanitize_mia_dspark_weights,
)
from mtplx.models.deepseek_v4 import DeepseekV4MoE, ModelArgs
from mtplx.kernels.deepseek_v4_mhc import MiaMHCPlan


def test_mia_loader_installs_stacked_projections_after_weights_before_plan() -> None:
    source = inspect.getsource(load_mia_exl3_dspark_model)

    draft_load = source.index(
        "model.load_weights(list(draft_weights.items()), strict=False)"
    )
    wo_install = source.index("install_mia_tp1_wo_projection_routes(")
    stacked_install = source.index("install_mia_stacked_projections(model)")
    qkv_install = source.index("install_mia_qkv_prologue_routes(model)")
    plan_build = source.index("engine_plan = build_mia_engine_plan(")

    assert draft_load < wo_install < stacked_install < qkv_install < plan_build


_MIA_EXACT_MODEL = Path(
    "/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1"
)
_LAYER0 = _MIA_EXACT_MODEL / "exl3-layer-000-tp1-rank0.safetensors"
_MIA_K64_DRAFT = Path(
    "/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-dspark-k64"
)
_W1_TRELLIS = "layers.0.ffn.experts.0.w1.rank0.trellis"
_W1_SUH = "layers.0.ffn.experts.0.w1.rank0.suh"
_W1_SVH = "layers.0.ffn.experts.0.w1.rank0.svh"


class _StaticArray:
    def __init__(self, shape, dtype=mx.float16):
        self.shape = tuple(shape)
        self.size = int(np.prod(self.shape))
        self.dtype = dtype

    def reshape(self, *_shape):
        return self

    def astype(self, _dtype):
        return self

    def __mul__(self, _other):
        return self


_FAKE_METAL_KERNEL_CACHES = (
    exl3._mcg_qmv_kernel,
    exl3._route_hadamard_kernel,
    exl3._mma_route_pack_kernel,
    exl3._mcg_grouped_mma_kernel,
    exl3._route_output_hadamard_kernel,
    exl3._m6_quad_qmv_kernel,
    exl3._m6_dual_fc1_input_kernel,
    exl3._m6_dual_fc1_inner_kernel,
    exl3._m6_clamp10_activation_down_kernel,
    exl3._m6_down_inner_kernel,
    exl3._m6_direct_final_tail_kernel,
)


def _clear_fake_metal_kernel_caches() -> None:
    for factory in _FAKE_METAL_KERNEL_CACHES:
        factory.cache_clear()


_MIA_MHC_ROUTE_CONTRACT = (
    "broadcast_fn_fp32",
    "attention_post_pre_fn_fp32",
    "ffn_tiny_post_pre_fn_bf16_split32_fp32",
    "ffn_prefill_post_pre_fn_bf16_mma_bm64_fp32",
    "compact_gram_finalize",
    "head_bf16_then_rmsnorm",
)


def test_mia_mhc_post_pre_precision_is_selected_by_connection_role() -> None:
    plan = MiaMHCPlan.__new__(MiaMHCPlan)
    calls = []

    def record_split(
        _self,
        x,
        residual,
        post,
        comb,
        hc,
        norm,
        *,
        projection_weight,
    ):
        calls.append(("split", projection_weight))
        return x, residual, post, comb

    def record_prefill(_self, x, residual, post, comb, hc, norm):
        calls.append(("prefill", hc._mia_mhc_weight.fn_bf16))
        return x, residual, post, comb

    plan._post_pre_connection = MethodType(record_split, plan)
    plan._post_pre_ffn_prefill = MethodType(record_prefill, plan)
    plan.prefill_min_rows = 384
    fp32_weight = object()
    bf16_weight = object()
    hc = SimpleNamespace(
        fn=fp32_weight,
        _mia_mhc_weight=SimpleNamespace(fn_bf16=bf16_weight),
    )
    args = tuple(object() for _ in range(4))
    norm = object()

    plan._rows = lambda _value: 6
    plan.post_pre_attn(*args, hc, norm)
    plan.post_pre_ffn(*args, hc, norm)
    plan._rows = lambda _value: 1024
    plan.post_pre_attn(*args, hc, norm)
    plan.post_pre_ffn(*args, hc, norm)

    assert calls == [
        ("split", fp32_weight),
        ("split", bf16_weight),
        ("split", fp32_weight),
        ("prefill", bf16_weight),
    ]


def _callable_name(value):
    return str(getattr(value, "__name__", type(value).__name__))


def _assert_mia_carried_mhc_contract(model) -> None:
    """Assert the installed carried route, not the retired layer-local route."""

    target_connections = tuple(
        connection
        for layer in model.model.layers
        for connection in (layer.attn_hc, layer.ffn_hc)
    )
    draft_connections = tuple(
        connection
        for stage in model.dspark.stages
        for connection in (stage.attn_hc, stage.ffn_hc)
    )
    assert len(target_connections) == 43 * 2
    assert len(draft_connections) == 3 * 2

    target_mhc = model.model._mia_mhc
    draft_mhc = model.dspark._mia_mhc
    assert target_mhc is not draft_mhc
    assert target_mhc.bound_hyper_connections == len(target_connections)
    assert draft_mhc.bound_hyper_connections == len(draft_connections)
    assert target_mhc.route_contract == _MIA_MHC_ROUTE_CONTRACT
    assert draft_mhc.route_contract == _MIA_MHC_ROUTE_CONTRACT
    assert all(
        actual is expected
        for actual, expected in zip(
            target_mhc._hyper_connections,
            target_connections,
            strict=True,
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            draft_mhc._hyper_connections,
            draft_connections,
            strict=True,
        )
    )

    for connections in (target_connections, draft_connections):
        bindings = tuple(connection._mia_mhc_weight for connection in connections)
        assert len({id(binding) for binding in bindings}) == len(bindings)
        assert all(binding.fn_bf16.dtype == mx.bfloat16 for binding in bindings)
        assert bindings[0].fn_broadcast.shape == (24, 4096)
        assert bindings[0].fn_broadcast.dtype == mx.float32
        assert all(binding.fn_broadcast is None for binding in bindings[1:])

    installed_hot_routes = (
        model.model._hc_hidden_impl,
        model.model._collapse_impl,
        model._target_forward_route,
        model.dspark._propose_impl,
    )
    assert tuple(map(_callable_name, installed_hot_routes)) == (
        "_mia_hc_hidden",
        "_mia_collapse",
        "_mia_target_forward",
        "_mia_fullgraph_propose_k5",
    )
    generic_sinkhorn_callables = {
        id(connection._sinkhorn_normalise)
        for connection in target_connections + draft_connections
    }
    assert all(id(route) not in generic_sinkhorn_callables for route in installed_hot_routes)


def _named_route(name):
    def route(*_args, **_kwargs):
        raise AssertionError("route execution is outside this construction test")

    route.__name__ = name
    return route


def test_m6_quad_descriptors_exhaustively_match_scalar_trellis_states():
    plan = exl3._mcg_quad_descriptor_plan()
    words = np.array(
        [(index * 0x9E3779B9 + 0x7F4A7C15) & 0xFFFFFFFF for index in range(24)],
        dtype=np.uint32,
    )

    assert len(plan.descriptors) == 64
    assert plan.sha256 == exl3.EXL3_M6_QUAD_DESCRIPTOR_SHA256

    def merge(low, high, shift):
        if shift == 0:
            return high
        return ((high >> shift) | (low << (32 - shift))) & 0xFFFFFFFF

    observed_shifts = []
    for quad_row in range(4):
        for local_n in range(16):
            descriptor = plan.descriptors[quad_row * 16 + local_n]
            index0 = descriptor & 0x1F
            index1 = (descriptor >> 5) & 0x1F
            index2 = (descriptor >> 10) & 0x1F
            shift0 = (descriptor >> 15) & 0x1F
            shift1 = (descriptor >> 20) & 0x1F
            observed_shifts.extend((shift0, shift1))
            word0 = int(words[index0])
            word1 = int(words[index1])
            if quad_row in (0, 2):
                word2 = int(words[index2])
                window0 = merge(word0, word1, shift0)
                window1 = merge(word1, word2, shift1)
            else:
                high0 = word1 if descriptor & (1 << 25) else word0
                low1 = word1 if descriptor & (1 << 26) else word0
                window0 = merge(word0, high0, shift0)
                window1 = merge(low1, word1, shift1)
            decoded = (
                (window0 >> 3) & 0xFFFF,
                window0 & 0xFFFF,
                (window1 >> 3) & 0xFFFF,
                window1 & 0xFFFF,
            )

            row0 = quad_row * 4
            tensor_cores = tuple(
                exl3.EXL3_TENSOR_CORE_INVERSE[(row0 + offset) * 16 + local_n]
                for offset in range(4)
            )
            assert tensor_cores == (
                tensor_cores[0],
                tensor_cores[0] + 1,
                tensor_cores[0] + 8,
                tensor_cores[0] + 9,
            )

            expected = []
            for tensor_core in tensor_cores:
                bit0 = tensor_core * 3 + 755
                bit1 = bit0 + 16
                scalar_index0 = bit0 // 32
                scalar_index1 = (bit1 - 1) // 32
                scalar_shift = (scalar_index1 + 1) * 32 - bit1
                low = int(words[scalar_index0 % 24])
                high = int(words[scalar_index1 % 24])
                expected.append(
                    (((low << 32) | high) >> scalar_shift) & 0xFFFF
                )
            assert decoded == tuple(expected)

    assert 0 in observed_shifts


def test_carried_mhc_contract_owns_43_target_and_3_draft_layers():
    def connection():
        return SimpleNamespace(
            _sinkhorn_normalise=_named_route("stock"),
            _mia_mhc_weight=SimpleNamespace(
                fn_bf16=SimpleNamespace(dtype=mx.bfloat16),
                fn_broadcast=None,
            ),
        )

    target_layers = tuple(
        SimpleNamespace(attn_hc=connection(), ffn_hc=connection())
        for _ in range(43)
    )
    draft_stages = tuple(
        SimpleNamespace(attn_hc=connection(), ffn_hc=connection())
        for _ in range(3)
    )
    target_connections = tuple(
        owner
        for layer in target_layers
        for owner in (layer.attn_hc, layer.ffn_hc)
    )
    draft_connections = tuple(
        owner
        for stage in draft_stages
        for owner in (stage.attn_hc, stage.ffn_hc)
    )
    target_connections[0]._mia_mhc_weight.fn_broadcast = SimpleNamespace(
        shape=(24, 4096), dtype=mx.float32
    )
    draft_connections[0]._mia_mhc_weight.fn_broadcast = SimpleNamespace(
        shape=(24, 4096), dtype=mx.float32
    )
    target_mhc = SimpleNamespace(
        bound_hyper_connections=86,
        route_contract=_MIA_MHC_ROUTE_CONTRACT,
        _hyper_connections=target_connections,
    )
    draft_mhc = SimpleNamespace(
        bound_hyper_connections=6,
        route_contract=_MIA_MHC_ROUTE_CONTRACT,
        _hyper_connections=draft_connections,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=target_layers,
            _mia_mhc=target_mhc,
            _hc_hidden_impl=_named_route("_mia_hc_hidden"),
            _collapse_impl=_named_route("_mia_collapse"),
        ),
        dspark=SimpleNamespace(
            stages=draft_stages,
            _mia_mhc=draft_mhc,
            _propose_impl=_named_route("_mia_fullgraph_propose_k5"),
        ),
        _target_forward_route=_named_route("_mia_target_forward"),
    )

    _assert_mia_carried_mhc_contract(model)


def test_trellis_bm64_descriptors_and_launch_use_populated_block_bound(
    monkeypatch,
):
    """M1024/top-k6/K216 can populate at most 308 BM64 route blocks."""

    calls = {}

    def route_kernel(**kwargs):
        calls["route"] = kwargs
        return tuple(_StaticArray(shape) for shape in kwargs["output_shapes"])

    def mma_kernel(**kwargs):
        calls["mma"] = kwargs
        return (object(),)

    monkeypatch.setattr(exl3.mx, "contiguous", lambda value: value)
    monkeypatch.setattr(
        exl3,
        "_trellis_route_pack_kernel",
        lambda _experts, _topk, _block_m: route_kernel,
    )
    monkeypatch.setattr(
        exl3,
        "_mcg_trellis_mma_kernel",
        lambda _size_k, _size_n, _experts, _block_m: mma_kernel,
    )

    tasks = 1024 * 6
    routes = exl3._pack_trellis_routes(
        _StaticArray((1024, 6)),
        experts=216,
        topk=6,
        block_m=64,
        kernel=route_kernel,
    )
    owner = SimpleNamespace(experts=216)
    bank = SimpleNamespace(
        input_dims=4096,
        output_dims=2048,
        trellis=object(),
    )
    exl3.EXL3SwitchGLU._trellis_mma(
        owner,
        bank,
        _StaticArray((tasks, 4096)),
        routes[3:],
        block_m=64,
        kernel=mma_kernel,
    )

    assert calls["route"]["output_shapes"] == [
        (tasks,),
        (tasks,),
        (tasks,),
        (308,),
        (308,),
        (308,),
        (1,),
    ]
    assert calls["mma"]["grid"] == (512, 64, 308)


def test_trellis_uses_measured_bm8_route_through_m127() -> None:
    source = inspect.getsource(EXL3SwitchGLU.fused)

    assert "self._trellis_plans[0 if rows <= 127 else 1]" in source


def test_trellis_swiglu_limit_is_a_valid_metal_float_literal(monkeypatch):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(exl3.mx.fast, "metal_kernel", capture)
    exl3._trellis_activation_down_hadamard_kernel.__wrapped__(2048, 216, 10.0)

    assert "constant constexpr float LIMIT = 10.0f;" in captured["header"]


def test_mia_exl3_install_binds_unconditional_direct_qmv_and_exact_tail(
    monkeypatch,
):
    direct_source = inspect.getsource(EXL3SwitchGLU.direct_qmv)
    assert "rows" not in direct_source
    assert "mma" not in direct_source
    assert "self.gate_proj(x_half, expert_ids)" in direct_source
    assert "self.up_proj(x_half, expert_ids)" in direct_source
    assert "self.down_proj(activated, expert_ids)" in direct_source

    switch = EXL3SwitchGLU(256, 256, 2, 1, limit=0.0)
    installs = []
    monkeypatch.setattr(
        switch,
        "install_trellis_runtime",
        lambda *, max_tokens: installs.append(max_tokens),
    )
    owner = SimpleNamespace(
        switch_mlp=switch,
        gate=SimpleNamespace(install_mia_router=lambda: installs.append("router")),
    )
    owner._mia_exl3_forward = MethodType(DeepseekV4MoE._mia_exl3_forward, owner)
    owner._required_input_rows = DeepseekV4MoE._required_input_rows

    DeepseekV4MoE.install_mia_exl3_runtime(owner, max_tokens=64)

    assert owner._mia_exl3_direct_qmv.__self__ is switch
    assert owner._mia_exl3_direct_qmv.__func__ is EXL3SwitchGLU.direct_qmv
    assert owner._mia_exl3_trellis_fused.__self__ is switch
    assert owner._mia_exl3_trellis_fused.__func__ is EXL3SwitchGLU.fused
    assert owner._mia_exl3_tail_combine is target_module._stock_moe_tail_combine
    assert installs == ["router", 64]


def test_mia_loader_rebinds_quad_qmv_only_after_generic_install():
    source = inspect.getsource(load_mia_exl3_dspark_model)
    generic_install = source.index(
        "layer.ffn.install_mia_exl3_runtime(max_tokens=8224)"
    )
    quad_install = source.index("install_mia_m6_quad_qmv_routes(model)")
    assert generic_install < quad_install

    events = []

    class Switch:
        def direct_qmv(self, _x, _expert_ids):
            return "oracle"

        def direct_qmv_m6_quad(self, _x, _expert_ids):
            return "quad"

        def direct_m6_clamp10(
            self,
            _x,
            _expert_ids,
            _route_weights,
            _shared,
        ):
            return "fused"

        def install_m6_quad_qmv_runtime(self):
            events.append("plan")

    switch = Switch()
    ffn = SimpleNamespace(
        switch_mlp=switch,
        _mia_exl3_direct_qmv=switch.direct_qmv,
    )
    exl3.install_mia_m6_quad_qmv_routes(
        SimpleNamespace(layers=(SimpleNamespace(ffn=ffn),))
    )

    assert events == ["plan"]
    assert ffn._mia_exl3_m6_fused.__self__ is switch
    assert ffn._mia_exl3_m6_fused.__func__ is Switch.direct_m6_clamp10


def test_direct_qmv_banks_bind_production_bn256_geometry(monkeypatch, request):
    """Mia's three direct banks must reuse each input H128 across two N panels."""

    qmv_sources = []
    launches = []

    def kernel(**kwargs):
        launches.append(kwargs)
        return tuple(
            _StaticArray(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
            )
        )

    def capture_metal_kernel(**kwargs):
        if kwargs["name"].startswith("mtplx_dsv4_exl3_mcg_qmv_"):
            qmv_sources.append(kwargs)
        return kernel

    monkeypatch.setattr(
        exl3.mx, "zeros", lambda shape, dtype: _StaticArray(shape, dtype)
    )
    monkeypatch.setattr(exl3.mx, "contiguous", lambda value: value)
    monkeypatch.setattr(exl3.mx.fast, "metal_kernel", capture_metal_kernel)
    _clear_fake_metal_kernel_caches()
    request.addfinalizer(_clear_fake_metal_kernel_caches)

    switch = EXL3SwitchGLU(4096, 2048, 216, 6, limit=0.0)
    banks = (switch.gate_proj, switch.up_proj, switch.down_proj)

    assert [bank._qmv_output_tile for bank in banks] == [256, 256, 256]
    assert [(bank.input_dims, bank.output_dims) for bank in banks] == [
        (4096, 2048),
        (4096, 2048),
        (2048, 4096),
    ]
    assert len(qmv_sources) == 2
    for captured in qmv_sources:
        assert "constant constexpr uint BLOCK_TILES_N = 16;" in captured["header"]
        assert "simd_shuffle_xor(value, ushort(stride))" in captured["header"]
        assert "for (uint stride = 1u; stride < 32u; stride <<= 1u)" in captured[
            "header"
        ]
        assert "for (uint stride = 32u; stride < HAD; stride <<= 1u)" in captured[
            "header"
        ]
        assert "float accumulator0 = 0.0f;" in captured["source"]
        assert "float accumulator1 = 0.0f;" in captured["source"]
        assert "BLOCK_TILES * BLOCK_TILES_N * TILE_WORDS" in captured["source"]
        assert captured["source"].count("x[x_row * SIZE_K + k]") == 1

    expert_ids = _StaticArray((6, 6), mx.int32)
    switch.gate_proj(_StaticArray((6, 4096)), expert_ids)
    switch.up_proj(_StaticArray((6, 4096)), expert_ids)
    switch.down_proj(_StaticArray((6, 6, 2048)), expert_ids)

    assert [call["grid"] for call in launches[-3:]] == [
        (128, 2048 // 256, 36),
        (128, 2048 // 256, 36),
        (128, 4096 // 256, 36),
    ]


def test_m6_quad_qmv_is_construction_bound_and_never_reenters_factories(
    monkeypatch,
    request,
):
    captured = {}
    launches = []

    def capture_metal_kernel(**kwargs):
        captured[kwargs["name"]] = kwargs

        def kernel(**launch):
            launches.append(launch)
            return tuple(
                _StaticArray(shape, dtype)
                for shape, dtype in zip(
                    launch["output_shapes"],
                    launch["output_dtypes"],
                    strict=True,
                )
            )

        return kernel

    monkeypatch.setattr(
        exl3.mx, "zeros", lambda shape, dtype: _StaticArray(shape, dtype)
    )
    monkeypatch.setattr(exl3.mx, "contiguous", lambda value: value)
    monkeypatch.setattr(exl3.mx, "minimum", lambda value, _limit: value)
    monkeypatch.setattr(exl3.mx, "clip", lambda value, _low, _high: value)
    monkeypatch.setattr(exl3.nn, "silu", lambda value: value)
    monkeypatch.setattr(exl3.mx.fast, "metal_kernel", capture_metal_kernel)
    _clear_fake_metal_kernel_caches()
    request.addfinalizer(_clear_fake_metal_kernel_caches)

    switch = EXL3SwitchGLU(4096, 2048, 216, 6, limit=10.0)
    switch.install_m6_quad_qmv_runtime()
    plan = switch._m6_quad_qmv_plan

    assert plan.geometry == (4096, 2048, 216, 6, 10.0, 256, 36)
    assert plan.descriptor_sha256 == exl3.EXL3_M6_QUAD_DESCRIPTOR_SHA256
    assert plan.stage_vector_bytes == 16
    assert plan.stage_vectors_per_k_tile == 96
    assert plan.hidden_to_intermediate is exl3._m6_quad_qmv_kernel(
        4096, 2048, False
    )
    assert plan.intermediate_to_hidden is exl3._m6_quad_qmv_kernel(
        2048, 4096, True
    )
    assert plan.dual_fc1_input is exl3._m6_dual_fc1_input_kernel()
    assert plan.dual_fc1_inner is exl3._m6_dual_fc1_inner_kernel()
    assert plan.activation_down is exl3._m6_clamp10_activation_down_kernel()
    assert plan.down_inner is exl3._m6_down_inner_kernel()
    assert plan.direct_final_tail is exl3._m6_direct_final_tail_kernel()
    project_source = inspect.getsource(EXL3SwitchGLU._m6_quad_project)
    assert "routed_input" not in project_source

    quad = [
        (name, value)
        for name, value in captured.items()
        if "m6_quad_mcg_qmv" in name
    ]
    assert len(quad) == 2
    for name, kernel in quad:
        assert name.endswith("_bn256_u4stage_v2")
        assert "constant uint QUAD_DESCRIPTORS[64]" in kernel["header"]
        assert "constant constexpr uint TILE_VECTORS = 6;" in kernel["header"]
        assert (
            "constant constexpr uint STAGE_VECTORS_PER_K_TILE = 96;"
            in kernel["header"]
        )
        assert "if (shift == 0u)" in kernel["header"]
        quad3 = kernel["header"].split("inline half4 decode_mcg_quad3", 1)[1]
        quad3, quad2 = quad3.split("inline half4 decode_mcg_quad2", 1)
        assert quad3.count("words[index") == 3
        assert quad2.count("words[index") == 2
        assert "uint state0 = (window0 >> 3u) & 0xffffu;" in kernel["header"]
        assert "uint state3 = window1 & 0xffffu;" in kernel["header"]
        assert "for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k)" in kernel[
            "source"
        ]
        assert kernel["source"].count("decode_mcg_quad3(") == 4
        assert kernel["source"].count("decode_mcg_quad2(") == 4
        source = kernel["source"]
        assert "threadgroup uint4 packed_tile_vectors[" in source
        assert "BLOCK_TILES * STAGE_VECTORS_PER_K_TILE" in source
        assert "reinterpret_cast<const device uint4*>(trellis)" in source
        assert "if (lane < STAGE_VECTORS_PER_K_TILE)" in source
        assert source.count(
            "for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k)"
        ) == 2
        assert "packed_index" not in source
        assert "reinterpret_cast<const device ushort*>(trellis)" not in source
        copy_start = source.index("if (lane < STAGE_VECTORS_PER_K_TILE)")
        assert source.index("size_t expert_base") < copy_start
        assert source.index("size_t k_base") < copy_start
        assert source.index("size_t n_base") < copy_start
        assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 2
        k0_accumulator0 = kernel["source"].index(
            "accumulator0 += value0_0 * float(weights0_0.x);"
        )
        k1_accumulator0 = kernel["source"].index(
            "accumulator0 += value0_1 * float(weights0_0.y);"
        )
        k2_accumulator0 = kernel["source"].index(
            "accumulator0 += value0_2 * float(weights0_0.z);"
        )
        k3_accumulator0 = kernel["source"].index(
            "accumulator0 += value0_3 * float(weights0_0.w);"
        )
        assert k0_accumulator0 < k1_accumulator0 < k2_accumulator0 < k3_accumulator0

    stages = {
        name: kernel
        for name, kernel in captured.items()
        if name.startswith("mtplx_dsv4_exl3_m6_")
        and "quad_mcg_qmv" not in name
    }
    assert set(stages) == {
        "mtplx_dsv4_exl3_m6_dual_fc1_input_h4096_v2",
        "mtplx_dsv4_exl3_m6_dual_fc1_inner_h4096_i2048_v3",
        "mtplx_dsv4_exl3_m6_clamp10_activation_down_i2048_v2",
        "mtplx_dsv4_exl3_m6_down_inner_i2048_h4096_v2",
        "mtplx_dsv4_exl3_m6_direct_final_tail_h4096_t6_v2",
    }

    h128_stages = [
        stages["mtplx_dsv4_exl3_m6_dual_fc1_input_h4096_v2"],
        stages["mtplx_dsv4_exl3_m6_clamp10_activation_down_i2048_v2"],
        stages["mtplx_dsv4_exl3_m6_direct_final_tail_h4096_t6_v2"],
    ]
    for kernel in h128_stages:
        header = kernel["header"]
        assert "inline float4 hadamard_h128_quad" in header
        assert "float s0 = value.x + value.y;" in header
        assert "float d0 = value.x - value.y;" in header
        assert "float s1 = value.z + value.w;" in header
        assert "float d1 = value.z - value.w;" in header
        assert "float h0 = s0 + s1;" in header
        assert "float h1 = d0 + d1;" in header
        assert "float h2 = s0 - s1;" in header
        assert "float h3 = d0 - d1;" in header
        assert "for (uint step = 0u; step < 5u; ++step)" in header
        assert "simd_shuffle_xor" in header
        for component in range(4):
            assert f"h{component} = p{component} - h{component};" in header
            assert f"h{component} = h{component} + p{component};" in header
            assert f"h{component} = p{component} + h{component};" not in header
        assert "threadgroup float exchange" not in header
        assert "threadgroup_barrier" not in header
        assert "threadgroup float exchange" not in kernel["source"]
        assert "threadgroup_barrier" not in kernel["source"]

    staged = stages["mtplx_dsv4_exl3_m6_dual_fc1_input_h4096_v2"]
    assert staged["output_names"] == ["gate_h", "up_h"]
    staged_source = staged["source"]
    assert "uint k0 = k_block * HAD + lane * 4u;" in staged_source
    assert "half x0 = half(x[" in staged_source
    assert "half gate_scaled0 = half(" in staged_source
    assert "float4 gate_transformed = hadamard_h128_quad(" in staged_source
    assert "gate_h[(size_t)task * HIDDEN + k0 + 3u] = half(" in staged_source
    assert "float4 up_transformed = hadamard_h128_quad(" in staged_source
    assert "up_h[(size_t)task * HIDDEN + k0 + 3u] = half(" in staged_source

    dual = stages["mtplx_dsv4_exl3_m6_dual_fc1_inner_h4096_i2048_v3"]
    assert dual["output_names"] == ["gate_inner", "up_inner"]
    assert dual["input_names"] == [
        "gate_h",
        "up_h",
        "gate_trellis",
        "up_trellis",
        "expert_ids",
    ]
    assert "gate_suh" not in dual["source"]
    assert "hadamard_h128" not in dual["source"]
    assert "half(gate_accumulator0)" in dual["source"]
    assert "struct QuadWeightPair" in dual["header"]
    assert "decode_mcg_column_words" in dual["header"]
    assert "decode_mcg_column_pair" in dual["header"]
    assert "constant uint QUAD_DESCRIPTORS" not in dual["header"]
    assert dual["source"].count("decode_mcg_column_pair(") == 2

    activation = stages[
        "mtplx_dsv4_exl3_m6_clamp10_activation_down_i2048_v2"
    ]
    activation_source = activation["source"]
    assert "uint column0 = block * HAD + lane * 4u;" in activation_source
    assert "float4 gate_had = hadamard_h128_quad(" in activation_source
    assert "half gate_rotated0 = half(gate_had.x * HAD_SCALE);" in activation_source
    assert "half gate0 = half(" in activation_source
    assert (
        "half silu0 = half(gate0 * sigmoid_mlx_exact(gate0));"
        in activation_source
    )
    assert "half activated0 = half(silu0 * up0);" in activation_source
    assert "half down_scaled0 = half(" in activation_source
    assert "float4 down_had = hadamard_h128_quad(" in activation_source
    assert "down_h[(size_t)task * INTERMEDIATE + column0 + 3u] = half(" in (
        activation_source
    )

    down = stages["mtplx_dsv4_exl3_m6_down_inner_i2048_h4096_v2"]
    assert down["input_names"] == ["down_h", "down_trellis", "expert_ids"]
    assert "down_inner[(size_t)task * SIZE_N + n0] = half(accumulator0);" in down[
        "source"
    ]

    final = stages["mtplx_dsv4_exl3_m6_direct_final_tail_h4096_t6_v2"]
    final_source = final["source"]
    assert "uint column0 = block * HAD + lane * 4u;" in final_source
    assert "T mixed0 = T(0.0f);" in final_source
    assert "T mixed3 = T(0.0f);" in final_source
    assert "float4 output_had = hadamard_h128_quad(" in final_source
    assert "half rotated0 = half(output_had.x * HAD_SCALE);" in final_source
    assert "T projected0 = T(projected_half0);" in final_source
    assert "T weight = T(route_weights[task]);" in final_source
    assert "T product0 = T(projected0 * weight);" in final_source
    assert "mixed0 = T(product0 + mixed0);" in final_source
    assert "T product3 = T(projected3 * weight);" in final_source
    assert "mixed3 = T(product3 + mixed3);" in final_source

    def forbidden_factory(*_args, **_kwargs):
        raise AssertionError("installed quad QMV re-entered its kernel factory")

    monkeypatch.setattr(exl3, "_m6_quad_qmv_kernel", forbidden_factory)
    monkeypatch.setattr(exl3, "_m6_dual_fc1_input_kernel", forbidden_factory)
    monkeypatch.setattr(exl3, "_m6_dual_fc1_inner_kernel", forbidden_factory)
    monkeypatch.setattr(
        exl3, "_m6_clamp10_activation_down_kernel", forbidden_factory
    )
    monkeypatch.setattr(exl3, "_m6_down_inner_kernel", forbidden_factory)
    monkeypatch.setattr(exl3, "_m6_direct_final_tail_kernel", forbidden_factory)
    result = switch.direct_m6_clamp10(
        _StaticArray((6, 4096), mx.bfloat16),
        _StaticArray((6, 6), mx.int32),
        _StaticArray((6, 6), mx.float32),
        _StaticArray((6, 4096), mx.bfloat16),
    )

    assert result is not None
    assert [launch["grid"] for launch in launches[-5:]] == [
        (32, 32, 36),
        (128, 8, 36),
        (32, 16, 36),
        (128, 16, 36),
        (32, 32, 6),
    ]
    assert [launch["threadgroup"] for launch in launches[-5:]] == [
        (32, 1, 1),
        (128, 1, 1),
        (32, 1, 1),
        (128, 1, 1),
        (32, 1, 1),
    ]


def test_mia_exl3_forward_selects_direct_only_for_physical_m6_verify(monkeypatch):
    phase = "decode_verify"
    calls = []
    indices = object()
    weights = object()
    shared = object()
    direct_output = object()
    trellis_output = object()

    def direct_fused(
        xf,
        observed_indices,
        observed_weights,
        observed_shared,
    ):
        calls.append(
            (
                "direct_fused",
                xf,
                observed_indices,
                observed_weights,
                observed_shared,
            )
        )
        return direct_output

    def trellis(xf, observed_indices, observed_weights, observed_shared):
        calls.append(
            ("trellis", xf, observed_indices, observed_weights, observed_shared)
        )
        return trellis_output

    owner = SimpleNamespace(
        gate=lambda _xf, _ids: (indices, weights),
        shared_experts=lambda _xf: shared,
        _mia_exl3_m6_fused=direct_fused,
        _mia_exl3_trellis_fused=trellis,
    )
    monkeypatch.setattr(target_module, "current_attention_phase", lambda: phase)

    verify_x = _StaticArray((6, 4096))
    assert DeepseekV4MoE._mia_exl3_forward(owner, verify_x, None) is direct_output
    assert [call[0] for call in calls] == ["direct_fused"]

    calls.clear()
    phase = "prefill"
    assert DeepseekV4MoE._mia_exl3_forward(owner, verify_x, None) is trellis_output
    assert [call[0] for call in calls] == ["trellis"]

    calls.clear()
    phase = "decode_verify"
    decode_x = _StaticArray((5, 4096))
    assert DeepseekV4MoE._mia_exl3_forward(owner, decode_x, None) is trellis_output
    assert [call[0] for call in calls] == ["trellis"]


def test_installed_trellis_runtime_never_reenters_kernel_factories(monkeypatch):
    """BM8/BM64 execution must use only construction-bound Metal kernels."""

    def kernel(**kwargs):
        return tuple(
            _StaticArray(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
            )
        )

    monkeypatch.setattr(exl3, "_mma_route_pack_kernel", lambda _experts: kernel)
    monkeypatch.setattr(exl3.mx, "contiguous", lambda value: value)
    monkeypatch.setattr(exl3, "_mcg_qmv_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_route_hadamard_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_mcg_grouped_mma_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_route_output_hadamard_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_trellis_route_pack_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_packed_route_hadamard_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_mcg_trellis_mma_kernel", lambda *_args: kernel)
    monkeypatch.setattr(
        exl3, "_trellis_activation_down_hadamard_kernel", lambda *_args: kernel
    )
    monkeypatch.setattr(exl3, "_trellis_final_reduce_kernel", lambda *_args: kernel)

    owner = EXL3SwitchGLU(256, 256, 2, 1, limit=0.0)
    owner.install_trellis_runtime(max_tokens=64)

    def forbidden_factory(*_args, **_kwargs):
        raise AssertionError("installed Trellis execution re-entered a kernel factory")

    monkeypatch.setattr(exl3, "_trellis_route_pack_kernel", forbidden_factory)
    monkeypatch.setattr(exl3, "_packed_route_hadamard_kernel", forbidden_factory)
    monkeypatch.setattr(exl3, "_mcg_trellis_mma_kernel", forbidden_factory)
    monkeypatch.setattr(
        exl3, "_trellis_activation_down_hadamard_kernel", forbidden_factory
    )
    monkeypatch.setattr(exl3, "_trellis_final_reduce_kernel", forbidden_factory)

    for rows in (1, 33):
        owner.fused(
            mx.zeros((rows, 256), dtype=mx.float16),
            mx.zeros((rows, 1), dtype=mx.int32),
            mx.ones((rows, 1), dtype=mx.float32),
            mx.zeros((rows, 256), dtype=mx.float16),
        )


def _authentic_layer0_w1_tile() -> np.ndarray:
    if not _LAYER0.is_file():
        pytest.skip("exact MiaAI TP1 artifact is not installed")
    with safe_open(_LAYER0, framework="np") as handle:
        return handle.get_tensor(_W1_TRELLIS)[0, 0]


def test_authentic_mia_mcg_tile_decodes_in_source_layout():
    """Gate the exact MCG decode, bit windows, and tensor-core permutation."""

    tile = decode_mcg_trellis_tile(_authentic_layer0_w1_tile())

    assert tile.shape == (16, 16)
    assert tile.dtype == np.float16
    assert sha256(tile.tobytes()).hexdigest() == (
        "9c5d060bb4bb9caca2d16886d0c2c1192755571d651668e9332225a0b808e954"
    )
    np.testing.assert_array_equal(
        tile[0],
        np.array(
            [
                1.169921875,
                2.23046875,
                -0.97265625,
                -0.3203125,
                0.533203125,
                0.54052734375,
                -0.8876953125,
                1.32421875,
                0.100341796875,
                0.420166015625,
                1.388671875,
                1.45703125,
                0.64013671875,
                -0.78564453125,
                -1.0830078125,
                -0.9833984375,
            ],
            dtype=np.float16,
        ),
    )


def _hadamard128(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    stride = 1
    while stride < 128:
        for start in range(0, 128, stride * 2):
            left = out[start : start + stride].copy()
            right = out[start + stride : start + 2 * stride].copy()
            out[start : start + stride] = left + right
            out[start + stride : start + 2 * stride] = left - right
        stride *= 2
    return out * np.float32(1.0 / np.sqrt(128.0))


def _authentic_w1_block():
    if not _LAYER0.is_file():
        pytest.skip("exact MiaAI TP1 artifact is not installed")
    with safe_open(_LAYER0, framework="np") as handle:
        return (
            handle.get_tensor(_W1_TRELLIS)[:8, :8].copy(),
            handle.get_tensor(_W1_SUH)[:128].copy(),
            handle.get_tensor(_W1_SVH)[:128].copy(),
        )


def test_authentic_mia_projection_fuses_h128_signs_and_mcg_qmv():
    """The Metal operator must reproduce the pinned source projection order."""

    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    trellis, suh, svh = _authentic_w1_block()
    inner = np.empty((128, 128), dtype=np.float16)
    for tile_k in range(8):
        for tile_n in range(8):
            inner[
                tile_k * 16 : (tile_k + 1) * 16,
                tile_n * 16 : (tile_n + 1) * 16,
            ] = decode_mcg_trellis_tile(trellis[tile_k, tile_n])

    x = np.linspace(-1.0, 1.0, 128, dtype=np.float16)
    x_had = _hadamard128((x * suh).astype(np.float16)).astype(np.float16)
    projected = (x_had.astype(np.float32) @ inner.astype(np.float32)).astype(
        np.float16
    )
    expected = (
        _hadamard128(projected).astype(np.float16) * svh.astype(np.float16)
    ).astype(np.float16)

    actual = exl3_mcg_qmv(
        mx.array(x)[None],
        mx.array(trellis),
        mx.array(suh),
        mx.array(svh),
    )
    mx.eval(actual)

    assert tuple(actual.shape) == (1, 128)
    assert actual.dtype == mx.float16
    np.testing.assert_allclose(np.array(actual)[0], expected, rtol=2e-2, atol=2e-2)

    grouped = exl3_mcg_grouped_qmv(
        mx.array(x)[None],
        mx.array(trellis)[None],
        mx.array(suh)[None],
        mx.array(svh)[None],
        mx.array([[0]], dtype=mx.int32),
    )
    mx.eval(grouped)
    assert tuple(grouped.shape) == (1, 1, 128)
    np.testing.assert_array_equal(np.array(grouped)[0, 0], np.array(actual)[0])


def test_authentic_mia_bn256_matches_two_bn128_output_panels():
    """The production-wide QMV must preserve each independent H128 panel."""

    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    trellis, suh, svh = _authentic_w1_block()
    right_trellis = np.flip(trellis, axis=1).copy()
    right_svh = np.flip(svh).copy()
    wide_trellis = np.concatenate((trellis, right_trellis), axis=1)
    wide_svh = np.concatenate((svh, right_svh))
    x = mx.array(np.linspace(-1.0, 1.0, 128, dtype=np.float16))[None]

    left = exl3_mcg_qmv(
        x,
        mx.array(trellis),
        mx.array(suh),
        mx.array(svh),
    )
    right = exl3_mcg_qmv(
        x,
        mx.array(right_trellis),
        mx.array(suh),
        mx.array(right_svh),
    )
    wide_kernel = exl3._mcg_qmv_kernel(128, 256, block_n=256)
    (wide,) = wide_kernel(
        inputs=[
            mx.contiguous(x),
            mx.array(wide_trellis),
            mx.array(suh),
            mx.array(wide_svh),
        ],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 256)],
        output_dtypes=[mx.float16],
    )
    expected = mx.concatenate((left, right), axis=-1)
    mx.eval(wide, expected)

    np.testing.assert_array_equal(np.array(wide), np.array(expected))


def test_authentic_mia_m6_quad_qmv_matches_three_banks_and_final_bits():
    """Gate full production K/N/M arithmetic on authentic layer-0 storage."""

    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    if not _LAYER0.is_file():
        pytest.skip("exact MiaAI TP1 artifact is not installed")

    switch = EXL3SwitchGLU(4096, 2048, 216, 6, limit=10.0)
    with safe_open(_LAYER0, framework="np") as handle:
        for bank, weight in (
            (switch.gate_proj, "w1"),
            (switch.up_proj, "w3"),
            (switch.down_proj, "w2"),
        ):
            prefix = f"layers.0.ffn.experts.0.{weight}.rank0"
            bank.trellis = mx.array(handle.get_tensor(f"{prefix}.trellis"))[None]
            bank.suh = mx.array(handle.get_tensor(f"{prefix}.suh"))[None]
            bank.svh = mx.array(handle.get_tensor(f"{prefix}.svh"))[None]

    descriptor_plan = exl3._mcg_quad_descriptor_plan()
    switch._m6_quad_qmv_plan = exl3._InstalledM6QuadQMVPlan(
        geometry=(4096, 2048, 216, 6, 10.0, 256, 36),
        descriptor_sha256=descriptor_plan.sha256,
        stage_vector_bytes=16,
        stage_vectors_per_k_tile=96,
        hidden_to_intermediate=exl3._m6_quad_qmv_kernel(4096, 2048, False),
        intermediate_to_hidden=exl3._m6_quad_qmv_kernel(2048, 4096, True),
        dual_fc1_input=exl3._m6_dual_fc1_input_kernel(),
        dual_fc1_inner=exl3._m6_dual_fc1_inner_kernel(),
        activation_down=exl3._m6_clamp10_activation_down_kernel(),
        down_inner=exl3._m6_down_inner_kernel(),
        direct_final_tail=exl3._m6_direct_final_tail_kernel(),
    )
    x = mx.array(
        np.linspace(-1.0, 1.0, 6 * 4096, dtype=np.float32).reshape(6, 4096)
    ).astype(mx.bfloat16)
    expert_ids = mx.zeros((6, 6), dtype=mx.int32)
    flat_ids = mx.contiguous(expert_ids.reshape(36).astype(mx.uint32))
    x_half = x.astype(mx.float16)

    oracle_gate_h = switch.gate_proj.transform_routes(x_half, flat_ids)
    oracle_up_h = switch.up_proj.transform_routes(x_half, flat_ids)
    staged_gate_h, staged_up_h = switch._m6_quad_qmv_plan.dual_fc1_input(
        inputs=[
            mx.contiguous(x),
            switch.gate_proj.suh,
            switch.up_proj.suh,
            mx.contiguous(expert_ids),
        ],
        grid=(32, 32, 36),
        threadgroup=(32, 1, 1),
        output_shapes=[(36, 4096), (36, 4096)],
        output_dtypes=[mx.float16, mx.float16],
    )

    oracle_gate = switch.gate_proj(x_half, expert_ids)
    quad_gate = switch._m6_quad_project(
        switch.gate_proj,
        x_half,
        flat_ids,
        switch._m6_quad_qmv_plan.hidden_to_intermediate,
    )
    oracle_up = switch.up_proj(x_half, expert_ids)
    quad_up = switch._m6_quad_project(
        switch.up_proj,
        x_half,
        flat_ids,
        switch._m6_quad_qmv_plan.hidden_to_intermediate,
    )
    activated = (
        exl3.nn.silu(mx.minimum(oracle_gate, 10.0))
        * mx.clip(oracle_up, -10.0, 10.0)
    ).astype(mx.float16)
    oracle_down = switch.down_proj(activated, expert_ids)
    quad_down = switch._m6_quad_project(
        switch.down_proj,
        activated.reshape(36, 2048),
        flat_ids,
        switch._m6_quad_qmv_plan.intermediate_to_hidden,
    )
    oracle_final = switch.direct_qmv(x, expert_ids)
    quad_final = switch.direct_qmv_m6_quad(x, expert_ids)
    route_weights = mx.array(
        np.linspace(0.03125, 0.96875, 36, dtype=np.float32).reshape(6, 6)
    )
    shared = mx.array(
        np.cos(np.arange(6 * 4096, dtype=np.float32) * 0.017).reshape(6, 4096)
    ).astype(mx.bfloat16)
    oracle_fused = target_module._stock_moe_tail_combine(
        quad_final,
        route_weights,
        shared,
    )
    quad_fused = switch.direct_m6_clamp10(
        x,
        expert_ids,
        route_weights,
        shared,
    )
    mx.eval(
        oracle_gate_h,
        staged_gate_h,
        oracle_up_h,
        staged_up_h,
        oracle_gate,
        quad_gate,
        oracle_up,
        quad_up,
        oracle_down,
        quad_down,
        oracle_final,
        quad_final,
        oracle_fused,
        quad_fused,
    )

    for oracle, quad in (
        (oracle_gate_h, staged_gate_h),
        (oracle_up_h, staged_up_h),
        (oracle_gate, quad_gate),
        (oracle_up, quad_up),
        (oracle_down, quad_down),
        (oracle_final, quad_final),
        (oracle_fused, quad_fused),
    ):
        np.testing.assert_array_equal(
            np.array(oracle.view(mx.uint16)),
            np.array(quad.view(mx.uint16)),
        )


def test_authentic_mia_grouped_mma_matches_exl3_projection():
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    trellis, suh, svh = _authentic_w1_block()
    rows = np.stack(
        [np.linspace(-1.0 + i / 32, 1.0 + i / 32, 128, dtype=np.float16)
         for i in range(9)]
    )
    ids = mx.zeros((9, 1), dtype=mx.int32)
    expected = exl3_mcg_grouped_qmv(
        mx.array(rows),
        mx.array(trellis)[None],
        mx.array(suh)[None],
        mx.array(svh)[None],
        ids,
    )
    actual = exl3_mcg_grouped_mma(
        mx.array(rows),
        mx.array(trellis)[None],
        mx.array(suh)[None],
        mx.array(svh)[None],
        ids,
    )
    mx.eval(expected, actual)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-2, atol=2e-2)


def test_exact_mia_config_installs_exl3_only_on_target_layers():
    if not (_MIA_EXACT_MODEL / "config.json").is_file():
        pytest.skip("exact MiaAI TP1 artifact is not installed")
    import json

    args = ModelArgs.from_dict(
        json.loads((_MIA_EXACT_MODEL / "config.json").read_text())
    )
    target = DeepseekV4MoE(args, 0)
    draft = DeepseekV4MoE(args, args.num_hidden_layers)

    assert isinstance(target.switch_mlp, EXL3SwitchGLU)
    assert tuple(target.switch_mlp.gate_proj.trellis.shape) == (216, 256, 128, 48)
    assert not isinstance(draft.switch_mlp, EXL3SwitchGLU)


def test_exact_mia_k64_draft_maps_native_fp4_and_fp8_storage():
    if not (_MIA_K64_DRAFT / "model.safetensors.index.json").is_file():
        pytest.skip("exact MiaAI K64 draft artifact is not installed")

    source = load_indexed_safetensors(_MIA_K64_DRAFT)
    mapped = sanitize_mia_dspark_weights(source, stages=3, experts=64)

    assert len(source) == 1249
    assert mapped["mtp.0.ffn.switch_mlp.gate_proj.weight"].shape == (
        64,
        2048,
        512,
    )
    assert mapped["mtp.0.ffn.switch_mlp.gate_proj.weight"].dtype == mx.uint32
    assert mapped["mtp.0.ffn.switch_mlp.gate_proj.scales"].shape == (
        64,
        2048,
        128,
    )
    assert mapped["mtp.0.main_proj.weight"].shape == (4096, 3072)
    assert mapped["mtp.0.main_proj.weight"].dtype == mx.uint32
    assert mapped["mtp.0.main_proj.scales"].shape == (4096, 384)
    assert "mtp.2.markov_head.markov_w1.weight" in mapped
    assert not any(".experts." in name for name in mapped)


def test_streaming_carried_split_fp8_pairs_keep_quantized_parameter_names():
    geometries = {
        "model.layers.20.ffn.shared_experts.down_proj": (128, 128),
        "model.layers.8.attn.wo_a": (128, 128),
    }
    scale_shard = {
        "layers.20.ffn.shared_experts.w2.scale": mx.zeros(
            (1, 1), dtype=mx.uint8
        ),
        "layers.8.attn.wo_a.scale": mx.zeros((1, 1), dtype=mx.uint8),
    }
    weight_shard = {
        "layers.20.ffn.shared_experts.w2.weight": mx.zeros(
            (128, 128), dtype=mx.uint8
        ),
        "layers.8.attn.wo_a.weight": mx.zeros(
            (128, 128), dtype=mx.uint8
        ),
    }

    mapped = {
        **exl3._map_mia_target_carried_shard(
            scale_shard,
            fp8_geometries=geometries,
        ),
        **exl3._map_mia_target_carried_shard(
            weight_shard,
            fp8_geometries=geometries,
        ),
    }

    assert set(mapped) == {
        "model.layers.20.ffn.shared_experts.down_proj.weight",
        "model.layers.20.ffn.shared_experts.down_proj.scales",
        "model.layers.8.attn.wo_a.weight",
        "model.layers.8.attn.wo_a.scales",
    }


def test_exact_mia_split_artifact_constructs_k216_target_and_k64_owner():
    if not (_MIA_K64_DRAFT / "model.safetensors.index.json").is_file():
        pytest.skip("exact MiaAI K64 draft artifact is not installed")

    model = load_mia_exl3_dspark_model(
        _MIA_EXACT_MODEL,
        draft_root=_MIA_K64_DRAFT,
        lazy=True,
    )

    assert model.args.n_routed_experts == 216
    assert model.args.dspark_block_size == 5
    assert model.args.dspark_target_layer_ids == [40, 41, 42]
    assert model._target_cache_type.__name__ == "DeepseekV4NVFP4Cache"
    assert model.dspark.args.n_routed_experts == 64
    assert len(model.dspark.stages) == 3
    _assert_mia_carried_mhc_contract(model)
    attention_owners = tuple(model.layers) + tuple(model.dspark.stages)
    assert len(attention_owners) == 46
    wo_plans = tuple(
        layer.attn._output_projection_impl for layer in attention_owners
    )
    assert len({id(plan) for plan in wo_plans}) == 46
    assert all(type(plan).__name__ == "MiaTP1WOMXFP8Plan" for plan in wo_plans)
    assert model._mia_wo_projection_receipt["plan_ids"] == tuple(
        id(plan) for plan in wo_plans
    )
    assert model.mtp[0].ffn.switch_mlp.gate_proj.mode == "mxfp4"
    assert model.mtp[0].main_proj.mode == "mxfp8"
