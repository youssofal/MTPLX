"""CPU gates for the fixed-shape post-attention DeepSeek-V4 island."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mtplx import deepseek_v4_attention_island as AI  # noqa: E402
from mtplx.models import deepseek_v4 as D  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    # CPU-pinned by design, but the pin must stay test-scoped: a module-level
    # set_default_device leaks into every later-collected module (pytest
    # imports all test modules before running any) and flips the engine's
    # Metal bit-exactness suites onto CPU fallbacks process-wide.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _args(*, hash_layers: int = 0):
    return D.ModelArgs(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        num_hash_layers=hash_layers,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=16,
        compress_ratios=[0, 0],
        sliding_window=16,
        swiglu_limit=1.25,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
    )


def _quantized_layer(
    seed: int, *, hash_layer: bool = False, routed_gate_group: int = 32
):
    mx.random.seed(seed)
    args = _args(hash_layers=1 if hash_layer else 0)
    layer = D.DeepseekV4DecoderLayer(args, layer_id=0 if hash_layer else 1)
    filled = []
    for name, value in tree_flatten(layer.parameters()):
        if name.endswith("tid2eid"):
            new = mx.random.randint(0, args.n_routed_experts, value.shape)
        elif value.ndim == 1:
            new = mx.random.normal(value.shape) * 0.05
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    layer.update(tree_unflatten(filled))
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(layer.ffn.switch_mlp, name)
        group_size = routed_gate_group if name == "gate_proj" else 64
        setattr(
            layer.ffn.switch_mlp,
            name,
            projection.to_quantized(group_size=group_size, bits=2),
        )
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(layer.ffn.shared_experts, name)
        setattr(
            layer.ffn.shared_experts,
            name,
            projection.to_quantized(group_size=64, bits=4),
        )
    mx.eval(layer.parameters())
    return args, layer


def _post_attention_inputs(args, width: int):
    mx.random.seed(90 + width)
    x = mx.random.normal((1, width, args.hidden_size)).astype(mx.bfloat16)
    residual = mx.random.normal(
        (1, width, args.hc_mult, args.hidden_size)
    ).astype(mx.bfloat16)
    post = mx.random.normal((1, width, args.hc_mult)).astype(mx.float32)
    comb = mx.softmax(
        mx.random.normal((1, width, args.hc_mult, args.hc_mult)), axis=-1
    ).astype(mx.float32)
    ids = mx.random.randint(0, args.vocab_size, (1, width))
    mx.eval(x, residual, post, comb, ids)
    return x, residual, post, comb, ids


def _stock_post_attention(layer, x, residual, post, comb, ids):
    h = layer.attn_hc.post(x, residual, post, comb)
    ffn_residual = h
    y, ffn_post, ffn_comb = layer.ffn_hc.pre(h)
    y = layer.ffn_norm(y)
    y = layer.ffn(y, input_ids=ids)
    return layer.ffn_hc.post(y, ffn_residual, ffn_post, ffn_comb)


def _shape_model(seed: int = 71):
    """Three tiny layers spanning the production router/Q2 layout classes."""

    mx.random.seed(seed)
    args = D.ModelArgs(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=3,
        num_hash_layers=1,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=64,
        compress_ratios=[0, 4, 128],
        sliding_window=16,
        swiglu_limit=1.25,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
    )
    model = D.Model(args)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        if name.endswith("tid2eid"):
            new = mx.random.randint(0, args.n_routed_experts, value.shape)
        elif value.ndim == 1:
            centre = 1.0 if name.endswith("norm.weight") else 0.0
            new = mx.random.normal(value.shape) * 0.05 + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    for layer_index, layer in enumerate(model.layers):
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(layer.ffn.switch_mlp, name)
            group_size = 64 if layer_index == 2 and name == "gate_proj" else (
                32 if name == "gate_proj" else 64
            )
            setattr(
                layer.ffn.switch_mlp,
                name,
                projection.to_quantized(group_size=group_size, bits=2),
            )
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(layer.ffn.shared_experts, name)
            setattr(
                layer.ffn.shared_experts,
                name,
                projection.to_quantized(group_size=64, bits=4),
            )
    mx.eval(model.parameters())
    return args, model


def _assert_cache_equal(control, candidate):
    assert len(control) == len(candidate)
    for left, right in zip(control, candidate, strict=True):
        assert left.offset == right.offset
        assert left.n_compressed == right.n_compressed
        assert left.n_index_compressed == right.n_index_compressed
        for name in ("window", "compressed", "index_compressed"):
            lhs = getattr(left, name)
            rhs = getattr(right, name)
            if lhs is None or rhs is None:
                assert lhs is rhs
            else:
                mx.eval(lhs, rhs)
                assert mx.array_equal(lhs, rhs)
        for lane_name in ("comp", "index_comp"):
            lhs_lane = getattr(left, lane_name)
            rhs_lane = getattr(right, lane_name)
            if lhs_lane is None or rhs_lane is None:
                assert lhs_lane is rhs_lane
                continue
            for name in ("cur_kv", "cur_score", "prev_kv", "prev_score"):
                lhs = getattr(lhs_lane, name)
                rhs = getattr(rhs_lane, name)
                if lhs is None or rhs is None:
                    assert lhs is rhs
                else:
                    mx.eval(lhs, rhs)
                    assert mx.array_equal(lhs, rhs)


@pytest.mark.parametrize(
    ("width", "hash_layer", "gate_group"),
    [(2, True, 32), (3, False, 32), (4, False, 64)],
)
def test_attention_island_matches_eager_post_attention_chain(
    width, hash_layer, gate_group
):
    args, layer = _quantized_layer(
        11 + width, hash_layer=hash_layer, routed_gate_group=gate_group
    )
    inputs = _post_attention_inputs(args, width)
    want = _stock_post_attention(layer, *inputs)
    bound = AI._bind_attention_island_layer(layer, width=width)
    got = bound(*inputs)
    mx.eval(want, got)
    assert mx.array_equal(want, got)


def test_attention_island_reuses_tapes_by_width_router_and_q2_layout():
    _, score_a = _quantized_layer(31, routed_gate_group=32)
    _, score_b = _quantized_layer(32, routed_gate_group=32)
    _, score_late = _quantized_layer(33, routed_gate_group=64)
    _, hashed = _quantized_layer(34, hash_layer=True, routed_gate_group=32)

    a = AI._bind_attention_island_layer(score_a, width=3)
    b = AI._bind_attention_island_layer(score_b, width=3)
    late = AI._bind_attention_island_layer(score_late, width=3)
    hashed_bound = AI._bind_attention_island_layer(hashed, width=3)
    other_width = AI._bind_attention_island_layer(score_a, width=2)

    assert a._tape is b._tape
    assert late._tape is not a._tape
    assert hashed_bound._tape is not a._tape
    assert other_width._tape is not a._tape


@pytest.mark.parametrize("width", [2, 3, 4])
def test_bound_width_body_matches_full_stock_body_cache_logits_and_argmax(width):
    args, model = _shape_model(80 + width)
    control_cache = model.make_cache()
    candidate_cache = model.make_cache()
    prompt = mx.random.randint(0, args.vocab_size, (1, 17))
    verify = mx.random.randint(0, args.vocab_size, (1, width))
    mx.eval(
        model.model.hc_hidden(prompt, control_cache),
        model.model.hc_hidden(prompt, candidate_cache),
    )
    _assert_cache_equal(control_cache, candidate_cache)

    want_hidden = model.model.hc_hidden(verify, control_cache)
    bound_layers = tuple(
        (layer, AI._bind_attention_island_layer(layer, width=width))
        for layer in model.layers
    )
    candidate_body = AI._BoundWidthBody(model.model, bound_layers, width)
    got_hidden = candidate_body(verify, candidate_cache)
    want_logits = model.logits_from_hc_hidden(want_hidden)
    got_logits = model.logits_from_hc_hidden(got_hidden)
    mx.eval(want_hidden, got_hidden, want_logits, got_logits)

    assert mx.array_equal(want_hidden, got_hidden)
    assert mx.array_equal(want_logits, got_logits)
    assert mx.array_equal(mx.argmax(want_logits, axis=-1), mx.argmax(got_logits, axis=-1))
    _assert_cache_equal(control_cache, candidate_cache)
    # The ratio-4 cache exposes only its logical rows, never physical padding.
    assert control_cache[1].compressed.shape[1] == control_cache[1].n_compressed


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("layer_index", [0, 1, 2])
def test_compiled_router_indices_and_weights_match_stock(width, layer_index):
    args, model = _shape_model(101 + width)
    layer = model.layers[layer_index]
    x = mx.random.normal((width, args.hidden_size)).astype(mx.bfloat16)
    ids = mx.random.randint(0, args.vocab_size, (1, width))
    stock_indices, stock_weights = layer.ffn.gate(x, ids.reshape(-1))
    gate = layer.ffn.gate
    auxiliary = gate.tid2eid if gate.hash else gate.e_score_correction_bias
    got_indices, got_weights = AI._route(
        x,
        ids,
        gate.weight,
        auxiliary,
        hash_router=gate.hash,
        topk=gate.topk,
        score_func=gate.score_func,
        route_scale=gate.route_scale,
    )
    mx.eval(stock_indices, stock_weights, got_indices, got_weights)
    assert mx.array_equal(stock_indices, got_indices)
    assert mx.array_equal(stock_weights, got_weights)


def test_production_43_by_three_bindings_create_and_reuse_exactly_nine_tapes():
    AI._TAPES.clear()
    _, model = _shape_model(121)
    hash32, score32, score64 = model.layers
    production_layers = (hash32,) * 3 + (score32,) * 39 + (score64,)
    bound = [
        AI._bind_attention_island_layer(layer, width=width)
        for width in (2, 3, 4)
        for layer in production_layers
    ]
    assert len(bound) == 129
    assert len({id(route._tape) for route in bound}) == 9
    assert len(AI._TAPES) == 9

    for route in bound:
        inputs = _post_attention_inputs(model.args, route.width)
        mx.eval(route(*inputs))
    assert len(AI._TAPES) == 9


def test_bound_hot_call_does_not_rediscover_projection_metadata(monkeypatch):
    args, layer = _quantized_layer(41)
    bound = AI._bind_attention_island_layer(layer, width=3)
    inputs = _post_attention_inputs(args, 3)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot path rediscovered invariant projection metadata")

    monkeypatch.setattr(AI, "_projection_contract", forbidden)
    got = bound(*inputs)
    mx.eval(got)
    assert got.shape == (1, 3, args.hc_mult, args.hidden_size)


def test_target_router_uses_only_decode_verify_m2_m3_m4(monkeypatch):
    calls = []

    def stock(ids, cache):
        calls.append(("stock", tuple(ids.shape), cache))
        return "stock"

    widths = {
        width: (lambda ids, cache, width=width: ("candidate", width, cache))
        for width in (2, 3, 4)
    }
    route = AI._AttentionIslandTargetRoute(stock=stock, widths=widths)
    monkeypatch.setattr(AI, "current_attention_phase", lambda: "decode_verify")
    monkeypatch.setattr(AI, "current_model_forward_kind", lambda: "target_verify")
    for width in (2, 3, 4):
        assert route(SimpleNamespace(shape=(1, width)), "cache") == (
            "candidate",
            width,
            "cache",
        )
    for phase in ("prefill", "decode_ar", "decode_repair", "mtp_draft"):
        monkeypatch.setattr(AI, "current_attention_phase", lambda phase=phase: phase)
        assert route(SimpleNamespace(shape=(1, 3)), "cache") == "stock"
    monkeypatch.setattr(AI, "current_attention_phase", lambda: "decode_verify")
    for kind in ("repair", "other"):
        monkeypatch.setattr(
            AI, "current_model_forward_kind", lambda kind=kind: kind
        )
        assert route(SimpleNamespace(shape=(1, 3)), "cache") == "stock"
    monkeypatch.setattr(AI, "current_model_forward_kind", lambda: "target_verify")
    assert route(SimpleNamespace(shape=(1, 1)), "cache") == "stock"
    assert route(SimpleNamespace(shape=(2, 2)), "cache") == "stock"
    assert calls == [
        *(("stock", (1, 3), "cache") for _ in range(6)),
        ("stock", (1, 1), "cache"),
        ("stock", (2, 2), "cache"),
    ]


def test_invalid_quantization_fails_at_binding_without_fallback():
    _, layer = _quantized_layer(51)
    layer.ffn.switch_mlp.gate_proj.bits = 4
    with pytest.raises(AI.AttentionIslandError, match="2-bit affine"):
        AI._bind_attention_island_layer(layer, width=3)


def test_enable_flag_is_read_only_at_construction(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV4_ATTENTION_ISLAND", raising=False)
    assert AI.deepseek_v4_attention_island_enabled() is False
    monkeypatch.setenv("MTPLX_DSV4_ATTENTION_ISLAND", "1")
    assert AI.deepseek_v4_attention_island_enabled() is True


def test_runtime_installs_island_after_loaded_o_lora_routes():
    runtime_source = Path("mtplx/runtime.py").read_text()
    o_lora = runtime_source.index("install_deepseek_v4_o_lora_routes(")
    island = runtime_source.index("install_deepseek_v4_attention_island(")
    runtime = runtime_source.index("runtime = runtime_class(")
    assert o_lora < island < runtime
    assert "deepseek_v4_attention_island_report" in runtime_source


def test_arm_selector_switches_prebound_model_route_without_hot_checks():
    model = SimpleNamespace(_target_hc_hidden_route=None)
    stock = object()
    candidate = object()
    selector = AI._AttentionIslandArmSelector(model, stock, candidate)
    model._mtplx_dsv4_attention_island_selector = selector

    assert model._target_hc_hidden_route is candidate
    AI.select_deepseek_v4_attention_island_arm(model, False)
    assert model._target_hc_hidden_route is stock
    assert selector.candidate_selected is False
    AI.select_deepseek_v4_attention_island_arm(model, True)
    assert model._target_hc_hidden_route is candidate
    assert selector.candidate_selected is True
