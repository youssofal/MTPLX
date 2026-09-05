from __future__ import annotations

import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import a3b_compiled_target_prefix as a3b_target
from mtplx.gdn_capture import A3BGDNPostconvFactory
from mtplx.graphbank import TensorOffsetKVCache
from mtplx.sampling import SamplerConfig


LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


def _config() -> dict:
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "dtype": "bfloat16",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "layer_types": list(LAYER_TYPES),
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_value_head_dim": 128,
            "linear_key_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "mtp_num_hidden_layers": 1,
        },
    }


def _model() -> SimpleNamespace:
    layers = []
    for index, layer_type in enumerate(LAYER_TYPES):
        is_linear = layer_type == "linear_attention"
        layer = SimpleNamespace(is_linear=is_linear)
        if is_linear:
            layer.linear_attn = SimpleNamespace(
                sharding_group=None,
                num_v_heads=32,
                num_k_heads=16,
                head_v_dim=128,
                head_k_dim=128,
                conv_kernel_size=4,
                conv_dim=8192,
            )
        else:
            layer.self_attn = SimpleNamespace(
                sharding_group=None,
                num_attention_heads=16,
                num_key_value_heads=2,
                head_dim=256,
            )
        layers.append(layer)
    return SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        mtp=SimpleNamespace(layers=[SimpleNamespace()]),
    )


def _postconv_factory() -> A3BGDNPostconvFactory:
    return A3BGDNPostconvFactory(
        m1_implementations=tuple(lambda *args: args for _ in range(30)),
        m2_implementations=tuple(lambda *args: args for _ in range(30)),
    )


def test_flag_off_installs_no_model_factory(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_COMPILED_TARGET_PREFIX", raising=False)
    model = SimpleNamespace()

    assert a3b_target.prepare_a3b_compiled_target_prefix(model, config={}) is None
    assert vars(model) == {}


def test_exact_model_contract_installs_one_immutable_factory(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    model = _model()
    postconv_factory = _postconv_factory()

    factory = a3b_target.prepare_a3b_compiled_target_prefix(
        model,
        config=_config(),
        gdn_postconv_factory=postconv_factory,
    )

    assert factory is not None
    assert not vars(model).get("_mtplx_a3b_compiled_target_prefix_factory")
    assert factory.layer_types == LAYER_TYPES
    assert factory.gdn_layers == 30
    assert factory.full_attention_layers == 10
    assert factory.gdn_postconv is postconv_factory
    assert factory.gdn_postconv.m1_implementations is postconv_factory.m1_implementations
    assert factory.gdn_postconv.m2_implementations is postconv_factory.m2_implementations

    source = inspect.getsource(a3b_target.prepare_a3b_compiled_target_prefix)
    assert "setattr(" not in source
    assert "_FACTORY_ATTRIBUTE" not in source


def test_full_attention_fields_match_upstream_qwen3_next_contract() -> None:
    from mlx_lm.models.qwen3_next import Qwen3NextAttention

    upstream_source = inspect.getsource(Qwen3NextAttention.__init__)
    validator_source = inspect.getsource(
        a3b_target.prepare_a3b_compiled_target_prefix
    )
    for field in ("num_attention_heads", "num_key_value_heads", "head_dim"):
        assert f"self.{field}" in upstream_source
        assert f'getattr(attention, "{field}"' in validator_source
    assert 'getattr(attention, "num_heads"' not in validator_source
    assert 'getattr(attention, "num_kv_heads"' not in validator_source


def test_invented_attention_field_names_do_not_install(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    model = _model()
    for index in a3b_target._FULL_ATTENTION_INDICES:
        attention = model.language_model.model.layers[index].self_attn
        del attention.num_attention_heads
        del attention.num_key_value_heads
        attention.num_heads = 16
        attention.num_kv_heads = 2

    with pytest.raises(
        a3b_target.A3BCompiledTargetPrefixConfigError,
        match="attention ownership",
    ):
        a3b_target.prepare_a3b_compiled_target_prefix(
            model,
            config=_config(),
            gdn_postconv_factory=_postconv_factory(),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("quantization", "bits"), 8),
        (("quantization", "group_size"), 32),
        (("text_config", "num_key_value_heads"), 4),
        (("text_config", "head_dim"), 128),
        (("text_config", "mtp_num_hidden_layers"), 2),
    ),
)
def test_invalid_model_contract_fails_during_load(monkeypatch, path, value) -> None:
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    config = _config()
    config[path[0]][path[1]] = value

    with pytest.raises(a3b_target.A3BCompiledTargetPrefixConfigError):
        a3b_target.prepare_a3b_compiled_target_prefix(
            _model(),
            config=config,
            gdn_postconv_factory=_postconv_factory(),
        )


def test_factory_requires_constructed_postconv_factory(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    model = _model()

    with pytest.raises(
        a3b_target.A3BCompiledTargetPrefixConfigError,
        match="GDN postconv factory",
    ):
        a3b_target.prepare_a3b_compiled_target_prefix(
            model,
            config=_config(),
            gdn_postconv_factory=None,
        )

    source = inspect.getsource(a3b_target.prepare_a3b_compiled_target_prefix)
    assert "_mtplx_a3b_gdn_postconv_m1_impl" not in source
    assert "_mtplx_a3b_gdn_postconv_m2_impl" not in source
    assert "callable(" not in source
    for duplicated_postconv_fact in (
        'config.get("model_type"',
        'config.get("architectures"',
        'text.get("model_type"',
        'text.get("dtype"',
        'text.get("hidden_size"',
        'text.get("num_hidden_layers"',
        'text.get("layer_types"',
        'text.get("linear_num_value_heads"',
        'text.get("linear_num_key_heads"',
        'text.get("linear_value_head_dim"',
        'text.get("linear_key_head_dim"',
        'text.get("linear_conv_kernel_dim"',
        '"linear_attn"',
        'getattr(gdn, "num_v_heads"',
        'getattr(gdn, "num_k_heads"',
        'getattr(gdn, "head_v_dim"',
        'getattr(gdn, "head_k_dim"',
        'getattr(gdn, "conv_kernel_size"',
        'getattr(gdn, "conv_dim"',
    ):
        assert duplicated_postconv_fact not in source
    assert "for index in _FULL_ATTENTION_INDICES" in source


def test_request_construction_trusts_finalized_model_factory() -> None:
    source = inspect.getsource(a3b_target.install_a3b_k1_target_prefix_route)

    cache_construction = source.index("_construct_a3b_target_cache")
    m2_install = source.index("_shared_m2_step")
    m1_install = source.index("_shared_m1_step")
    assert "factory: A3BCompiledTargetPrefixFactory" in source
    assert "getattr(" not in source
    assert "isinstance(" not in source
    assert "runtime.model" not in source
    for forbidden in (
        "_validate_request_cache",
        "build_verify_state_spec",
        "cache_has_python_offsets",
        "ArraysCache",
        "TensorOffsetKVCache",
        ".shape",
        ".dtype",
        "required_capacity",
        "permanent_eager",
        "_resolve_bucket",
        "CompiledVerifyBank",
        "_ensure_shadow",
        "_clear_shadow_leaf_refs",
        "promote_kv_cache_offsets",
        "failures",
    ):
        assert forbidden not in source
    assert not hasattr(a3b_target, "_validate_request_cache")
    assert cache_construction < m2_install < m1_install

    construction_source = inspect.getsource(a3b_target._construct_a3b_target_cache)
    assert "for index in _FULL_ATTENTION_INDICES" in construction_source
    assert "TensorOffsetKVCache.from_kv_cache" in construction_source
    for forbidden in (
        "promote_kv_cache_offsets",
        "failures",
        "isinstance(",
        "getattr(",
        ".shape",
        ".dtype",
        "eligible",
        "fallback",
    ):
        assert forbidden not in construction_source


def test_exact_request_rejects_unsupported_sampler_before_prompt_construction() -> None:
    with pytest.raises(
        a3b_target.A3BCompiledTargetPrefixConfigError,
        match="stochastic top-k sampler",
    ):
        a3b_target.validate_a3b_k1_target_prefix_sampler(
            SamplerConfig(temperature=0.6, top_p=0.95, top_k=0)
        )
    with pytest.raises(
        a3b_target.A3BCompiledTargetPrefixConfigError,
        match="requires top_k <= 32",
    ):
        a3b_target.validate_a3b_k1_target_prefix_sampler(
            SamplerConfig(temperature=0.6, top_p=0.95, top_k=33)
        )


@pytest.mark.parametrize(
    "sampler",
    (
        SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        SamplerConfig(temperature=-1.0, top_p=1.0, top_k=0),
    ),
)
def test_exact_request_accepts_greedy_as_deterministic_argmax_contract(
    sampler,
) -> None:
    # Greedy is the AR-exactness gate lane: every route sample site
    # degenerates to argmax, so the request is deterministic end-to-end.
    a3b_target.validate_a3b_k1_target_prefix_sampler(sampler)


def test_exact_request_rejects_oversized_device_draft_before_prompt() -> None:
    with pytest.raises(
        a3b_target.A3BCompiledTargetPrefixConfigError,
        match="requires top_k <= 32",
    ):
        a3b_target.validate_a3b_k1_device_draft_request(
            SamplerConfig(temperature=0.6, top_p=0.95, top_k=33),
            draft_margin_threshold=None,
            adaptive_policy=None,
            draft_core="stock",
            online_correction_cache=False,
            prompt_correction_cache=False,
            adapter_ensemble_q=False,
            mtp_topk_reranker=None,
            loop_guard=False,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )


def test_takeover_lane_uses_draft_source_not_block_rounds() -> None:
    # The block-round machinery is not AR-exact on the target_prefix lane
    # (M>2 forwards leave ulp-perturbed retained rows).  The takeover lane
    # must feed the copy match as the depth-1 draft (2-row proven geometry)
    # and leave block rounds to the capture_commit lane.
    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    round_gate = source.index(
        "if ccopy_active and _ccopy_capture_lane and cycle_depth >= 1"
    )
    assert round_gate > 0
    substitution = source.index("_cc_draft_source_token: int | None = None")
    assert "a3b_target_prefix_route is None" in source[substitution:substitution + 400]
    # The streak proposes from the prompt and never runs its own forward.
    streak = source.index('"mode": "draft_source"')
    assert streak > 0


def test_trim_lane_defers_correction_repairs() -> None:
    # The 2.3.0 deferred-correction fix, ported to the trim (target_prefix)
    # lane: a rejection must emit the correction as the pending primary and
    # never pay a dedicated one-row correction forward.
    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    assert "trimmed_prefix_pending_correction" in source
    assert "trimmed_prefix_correction_forward" not in source
    trim_branch = source[
        source.index("elif committed_from_trim:"):
        source.index('event["capture_repair"] = "trimmed_prefix_pending_correction"')
    ]
    assert "forward_ar" not in trim_branch
    assert "deferred_correction_repairs += 1" in source[
        source.index("elif committed_from_trim:"):
    ]


def test_route_records_rejection_correction_under_greedy() -> None:
    # The compiled route commits + repair-forwards the rejection correction
    # in-cycle (fixed 2-token geometry); the greedy lane's defer-to-next-cycle
    # convention would hand it a None and crash mx.array([[None]]).  The
    # acceptance loop must record the correction for the route at ANY
    # temperature -- under greedy it is the pre-sampled argmax target id.
    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    guard = source.index("or a3b_target_prefix_route is not None")
    record = source.index("rejection_correction = int(correction)")
    assert guard < record
    repair = source.index("committed.append(rejection_correction)")
    assert record < repair


def test_generic_target_prefix_sampler_contract_is_proven_without_sampling() -> None:
    from mtplx import generation

    with pytest.raises(
        RuntimeError,
        match="target_prefix verification requires top-k sampling or top_p=1",
    ):
        generation._validate_target_prefix_sampler_request(
            SamplerConfig(temperature=0.6, top_p=0.95, top_k=0)
        )

    generation._validate_target_prefix_sampler_request(
        SamplerConfig(temperature=0.0, top_p=0.95, top_k=0)
    )
    generation._validate_target_prefix_sampler_request(
        SamplerConfig(temperature=0.6, top_p=1.0, top_k=0)
    )
    generation._validate_target_prefix_sampler_request(
        SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    )
    with pytest.raises(RuntimeError, match="requires top_k <= 32"):
        generation._validate_target_prefix_sampler_request(
            SamplerConfig(temperature=0.6, top_p=0.95, top_k=33)
        )


def test_generation_routes_on_direct_runtime_factory_ownership() -> None:
    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    assert "rt.a3b_compiled_target_prefix_factory" in source
    assert "factory=exact_a3b_target_prefix_factory" in source
    request_sampler_proof = source.index("validate_a3b_k1_target_prefix_sampler(")
    prompt_construction = source.index("restore_or_prefill_prompt_state(")
    assert request_sampler_proof < prompt_construction
    assert "generic_compiled_target_prefix" in source
    exact_factory_assignment = source[
        source.index("target_prefix_verify =") :
        source.index("exact_a3b_target_prefix =")
    ]
    assert 'verify_strategy == "target_prefix"' in exact_factory_assignment
    assert "if target_prefix_verify and constraint is None" in exact_factory_assignment
    assert "generic_compiled_target_prefix" not in exact_factory_assignment
    assert "_env_truthy" not in exact_factory_assignment
    assert "a3b_compiled_target_prefix_factory(" not in source
    assert "getattr(rt, \"model\"" not in source


def test_fixed_m1_m2_trace_bodies_own_distinct_callable_tuples() -> None:
    m1_source = inspect.getsource(a3b_target._make_a3b_k1_target_prefix_m1_step)
    m2_source = inspect.getsource(a3b_target._make_a3b_k1_target_prefix_m2_step)

    assert "postconv_implementations=host[\"postconv_implementations\"]" in m1_source
    assert "postconv_implementations=host[\"postconv_implementations\"]" in m2_source
    for source in (m1_source, m2_source):
        assert "_forward_ar_capture_a3b_postconv" in source
        assert "forward_ar_capture(" not in source
        assert "gdn_forward_with_capture" not in source
        assert "_forward_with_gdn_capture" not in source


def test_installed_m1_m2_dispatch_contains_no_runtime_validation_or_fallback() -> None:
    sources = (
        inspect.getsource(a3b_target.A3BK1TargetPrefixRoute._forward_m2),
        inspect.getsource(a3b_target.A3BK1TargetPrefixRoute._forward_m1),
    )

    for source in sources:
        for forbidden in (
            "os.environ",
            "getenv",
            "shape",
            "dtype",
            "validate",
            "eligible",
            "promote",
            "build_verify_state_spec",
            "fallback",
            "stats",
            "try:",
            "except",
            "forward_ar",
            "_decode_length",
            "_unpack_outputs",
            "_rebuild_captures",
            "repair_m2",
        ):
            assert forbidden not in source


def test_fixed_compiled_bodies_contain_no_generic_dispatch_or_validation() -> None:
    sources = (
        inspect.getsource(a3b_target._make_a3b_k1_target_prefix_m2_step),
        inspect.getsource(a3b_target._make_a3b_k1_target_prefix_m1_step),
    )

    for source in sources:
        for forbidden in (
            "_decode_length",
            "len(outputs)",
            "expected",
            "fallback",
            ".forward_ar_capture(",
            "build_verify_state_spec",
            "promote_kv_cache_offsets",
            "try:",
            "except",
        ):
            assert forbidden not in source


def test_report_derives_m2_verify_and_m1_repair_without_counters() -> None:
    route = object.__new__(a3b_target.A3BK1TargetPrefixRoute)
    route.request_max_tokens = 10_000
    route.growth_reserve_tokens = 10_002
    route.prompt_tokens = 181

    report = route.final_report(verify_calls=1_604, repair_calls=318)

    assert report["calls"] == report["compiled_calls"] == 1_922
    assert report["m2_verify_calls"] == report["m2_calls"] == 1_604
    assert report["m1_repair_calls"] == report["m1_calls"] == 318
    assert report["buckets"] == {"m2_verify:0": 1_604, "m1_repair:0": 318}
    assert report["compiled_keys"] == ["m2:verify:b0", "m1:repair:b0"]
    assert report["fallback_calls"] == report["growth_demotions"] == 0
    assert report["fallback_reasons"] == {}
    assert report["device_draft_input"] is True


def test_fixed_state_layout_has_primary_then_final_m2_outputs() -> None:
    assert a3b_target._STATE_LEAVES == 90
    assert a3b_target._PRIMARY_STATE_START == 2
    assert a3b_target._FINAL_STATE_START == 92
    assert a3b_target._M1_FINAL_STATE_START == 2


def test_exact_route_constructs_and_demotes_all_40_positions_in_fixed_order(
    monkeypatch,
) -> None:
    class FakeArraysCache:
        def __init__(self, size):
            self.cache = [None] * size

    class FakeTensorOffsetKVCache:
        promoted_indices: list[int] = []

        def __init__(self, keys, values, offset, *, step=256):
            self.cache = [keys, values, offset]
            self.rollback_state = [None, None, None]
            self.step = step

        @classmethod
        def from_kv_cache(cls, entry, *, reserve_tokens):
            cls.promoted_indices.append(entry.layer_index)
            promoted = cls(entry.keys, entry.values, entry.offset, step=entry.step)
            promoted.reserve_tokens = reserve_tokens
            promoted.layer_index = entry.layer_index
            return promoted

        def demote(self):
            return ("demoted", self.layer_index)

    cache = []
    original_gdns = {}
    for index, kind, _leaves in a3b_target._STATE_SPEC:
        if kind == a3b_target.VERIFY_SPEC_KIND_GDN:
            entry = FakeArraysCache(2)
            entry.cache[:] = [(index, "conv"), (index, "state")]
            original_gdns[index] = entry
        else:
            entry = SimpleNamespace(
                layer_index=index,
                keys=(index, "keys"),
                values=(index, "values"),
                offset=(index, "offset"),
                step=256,
            )
        cache.append(entry)

    shared = {}
    monkeypatch.setattr(a3b_target, "ArraysCache", FakeArraysCache)
    monkeypatch.setattr(a3b_target, "TensorOffsetKVCache", FakeTensorOffsetKVCache)
    monkeypatch.setattr(a3b_target, "_owned_state_env_active", lambda _name: False)
    monkeypatch.setattr(a3b_target, "_compiled_verify_boundary", lambda: "both")
    monkeypatch.setattr(a3b_target, "_compiled_verify_donation_enabled", lambda: True)
    monkeypatch.setattr(
        a3b_target,
        "_shared_m2_step",
        lambda runtime, shadow, hidden_variant, implementations: shared.setdefault(
            "m2", (runtime, shadow, hidden_variant, implementations)
        ),
    )
    monkeypatch.setattr(
        a3b_target,
        "_shared_m1_step",
        lambda runtime, shadow, hidden_variant, implementations: shared.setdefault(
            "m1", (runtime, shadow, hidden_variant, implementations)
        ),
    )
    runtime = object()
    m1_implementations = tuple(("m1", index) for index in range(30))
    m2_implementations = tuple(("m2", index) for index in range(30))
    factory = SimpleNamespace(
        gdn_postconv=SimpleNamespace(
            m1_implementations=m1_implementations,
            m2_implementations=m2_implementations,
        )
    )

    route = a3b_target.install_a3b_k1_target_prefix_route(
        runtime,
        cache,
        factory=factory,
        max_tokens=1_024,
        prompt_tokens=181,
        verify_strategy="target_prefix",
        speculative_depth=1,
        requested_speculative_depth=1,
        verify_core="stock",
        hidden_variant="post_norm",
        state_rebase_every=0,
    )

    assert tuple(FakeTensorOffsetKVCache.promoted_indices) == (
        a3b_target._FULL_ATTENTION_INDICES
    )
    assert len(route.state_slots) == 90
    position = 0
    for index, kind, leaves in a3b_target._STATE_SPEC:
        entry = cache[index]
        if kind == a3b_target.VERIFY_SPEC_KIND_GDN:
            assert entry is original_gdns[index]
        else:
            assert isinstance(entry, FakeTensorOffsetKVCache)
            assert entry.reserve_tokens == 1_026
        assert route.state_slots[position : position + leaves] == tuple(
            (entry.cache, slot) for slot in range(leaves)
        )
        position += leaves

    m2_shadow = shared["m2"][1]
    m1_shadow = shared["m1"][1]
    assert m1_shadow is m2_shadow
    assert shared["m1"][3] is m1_implementations
    assert shared["m2"][3] is m2_implementations
    for index, kind, _leaves in a3b_target._STATE_SPEC:
        shadow_entry = m2_shadow[index]
        if kind == a3b_target.VERIFY_SPEC_KIND_GDN:
            assert isinstance(shadow_entry, FakeArraysCache)
            assert shadow_entry.cache == [None, None]
        else:
            assert isinstance(shadow_entry, FakeTensorOffsetKVCache)
            assert shadow_entry.cache == [None, None, None]

    assert route.demote() == 10
    for index, kind, _leaves in a3b_target._STATE_SPEC:
        if kind == a3b_target.VERIFY_SPEC_KIND_GDN:
            assert cache[index] is original_gdns[index]
        else:
            assert cache[index] == ("demoted", index)


def test_m2_writes_final_and_returns_primary(monkeypatch) -> None:
    slots = tuple(([None], 0) for _ in range(a3b_target._STATE_LEAVES))
    primary = tuple(object() for _ in range(a3b_target._STATE_LEAVES))
    final = tuple(object() for _ in range(a3b_target._STATE_LEAVES))
    route = object.__new__(a3b_target.A3BK1TargetPrefixRoute)
    route.state_slots, route.rollback_slots = slots, ()
    route.compiled_m2 = lambda *_args: ("logits", "hidden", *primary, *final)
    monkeypatch.setattr(a3b_target.mx, "async_eval", lambda *_args: None)

    logits, hidden, got_primary = route.verify_m2(object())

    assert (logits, hidden, got_primary) == ("logits", "hidden", primary)
    assert tuple(container[0] for container, _slot in slots) == final


def test_m1_consumes_primary_and_installs_correction_final(monkeypatch) -> None:
    slots = tuple(([None], 0) for _ in range(a3b_target._STATE_LEAVES))
    primary = tuple(object() for _ in range(a3b_target._STATE_LEAVES))
    final = tuple(object() for _ in range(a3b_target._STATE_LEAVES))
    seen = []
    route = object.__new__(a3b_target.A3BK1TargetPrefixRoute)
    route.state_slots, route.rollback_slots = slots, ()
    route.compiled_m1 = lambda token, *state: (
        seen.append(state) or ("logits", "hidden", *final)
    )
    monkeypatch.setattr(a3b_target.mx, "async_eval", lambda *_args: None)

    result = route.repair_m1(object(), primary)

    assert result == ("logits", "hidden", None)
    assert seen == [primary]
    assert tuple(container[0] for container, _slot in slots) == final


def test_tensor_offset_primary_then_m1_matches_reference_prefix() -> None:
    zeros = mx.zeros((1, 2, 8, 256), dtype=mx.bfloat16)
    candidate = TensorOffsetKVCache(zeros, zeros, 0)
    reference = TensorOffsetKVCache(zeros, zeros, 0)
    key_a = mx.full((1, 2, 1, 256), 1, dtype=mx.bfloat16)
    key_d = mx.full((1, 2, 1, 256), 2, dtype=mx.bfloat16)
    key_c = mx.full((1, 2, 1, 256), 3, dtype=mx.bfloat16)
    value_a, value_d, value_c = key_a * 4, key_d * 4, key_c * 4

    candidate.update_and_fetch(
        mx.concatenate((key_a, key_d), axis=2),
        mx.concatenate((value_a, value_d), axis=2),
    )
    candidate.cache[2] = candidate.cache[2] - 1
    candidate.update_and_fetch(key_c, value_c)
    reference.update_and_fetch(
        mx.concatenate((key_a, key_c), axis=2),
        mx.concatenate((value_a, value_c), axis=2),
    )
    mx.eval(*candidate.cache, *reference.cache)

    assert int(candidate.cache[2].item()) == int(reference.cache[2].item()) == 2
    assert mx.array_equal(candidate.cache[0][:, :, :2], reference.cache[0][:, :, :2])
    assert mx.array_equal(candidate.cache[1][:, :, :2], reference.cache[1][:, :, :2])


def test_generation_exact_route_has_fixed_m2_m1_schedule_without_generic_repair() -> None:
    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    event_block = source[source.index("if graphbank is not None:") : source.index(
        "accepted_count = 0"
    )]
    snapshot_block = source[
        source.index("before_verify = None") : source.index(
            "lazy_bonus_verify_min_depth"
        )
    ]
    rejection_start = source.index("committed = [primary] + draft_tokens[:accepted_count]")
    exact_verify_start = source.index("elif a3b_target_prefix_route is not None:")
    draft_sample_start = source.index("sample_token_ids_from_mlx_logits(")
    target_sample_start = source.index(
        "sample_token_ids_from_mlx_logits(", draft_sample_start + 1
    )
    acceptance_start = source.index("accepted_count = 0")
    exact_repair_start = source.index(
        "if a3b_target_prefix_route is not None:", rejection_start
    )
    generic_optional_commit = source.index(
        "if rejection_correction is not None:", rejection_start
    )
    exact_repair_block = source[exact_repair_start:generic_optional_commit]

    assert "compiled_verify_bank.to_dict()" not in event_block
    assert "a3b_target_prefix_route.final_report" in source
    assert "if a3b_target_prefix_route is None:" in snapshot_block
    # The env gate lives behind PR #208's _skip_verify_snapshot() helper now
    # (same env, plus the recurrent-cache loud-failure guard); the invariant —
    # snapshot handling only on the non-compiled route — is unchanged.
    assert snapshot_block.index("if a3b_target_prefix_route is None:") < (
        snapshot_block.index("if _skip_verify_snapshot()")
    )
    assert "verify_logits, verify_hidden, a3b_primary_state = (" in source
    assert draft_sample_start < exact_verify_start < target_sample_start
    assert target_sample_start < acceptance_start
    assert source.count("sample_token_ids_from_mlx_logits(") == 2
    assert "verify_input_array = mx.concatenate(" in source
    assert "_eval(sampled_target_ids, device_draft_token)" in source
    assert "if sampled_target_ids is None" not in source
    assert "target_prefix_sampler =" not in source
    assert not hasattr(generation, "_sample_target_prefix_ids_checked")
    generic_proof = inspect.getsource(
        generation._validate_target_prefix_sampler_request
    )
    assert "sample_token_ids_from_mlx_logits" not in generic_proof
    assert "a3b_primary_state = None" not in source
    assert "if a3b_primary_state" not in source
    # Deferred-correction fold: repair_m1 is never dispatched; a rejection
    # stashes the post-primary state and the next verify is the rebased M2.
    assert "a3b_target_prefix_route.repair_m1(" not in source
    assert "a3b_target_prefix_route.verify_m2_rebased(" in source
    assert "a3b_rebase_state = a3b_primary_state" in exact_repair_block
    assert "deferred_correction_repairs += 1" in exact_repair_block
    assert exact_repair_start < generic_optional_commit
    assert "committed.append(rejection_correction)" in exact_repair_block
    assert "pending_primary = int(rejection_correction)" in exact_repair_block
    assert "verify_hidden[:, 0:1, :]" in exact_repair_block
    assert (
        "cache_committed_token_count = max(0, len(tokens) - 1)"
        in exact_repair_block
    )
    for forbidden in (
        "snapshot_untrimmable_cache",
        "rollback_after_verify",
        "trim_verified_window_to_prefix",
        "commit_captured_prefix",
        "repair_m2",
        "rt.forward_ar",
    ):
        assert forbidden not in exact_repair_block

    generic_repair = source[source.index("committed_prefix_len =") :]
    for preserved in (
        "trim_verified_window_to_prefix",
        "commit_captured_prefix",
        "rollback_after_verify",
        "rt.forward_ar",
        "pending_primary",
    ):
        assert preserved in generic_repair


def test_generation_exact_route_never_engages_under_grammar_constraint() -> None:
    """The exact route pre-commits its rejection correction (no None-guard on
    the append), while the #186 phase-3 grammar clamp drops grammar-illegal
    corrections so the next masked primary resamples them. A constrained
    request must therefore fall back to the stock target_prefix lane."""
    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    factory_block = source[
        source.index("exact_a3b_target_prefix_factory = (") : source.index(
            "exact_a3b_target_prefix = "
        )
    ]
    assert "if target_prefix_verify and constraint is None" in factory_block
    # The gate exists because the exact commit path appends the correction
    # unconditionally; if that ever changes, revisit whether the gate can lift.
    rejection_start = source.index(
        "committed = [primary] + draft_tokens[:accepted_count]"
    )
    exact_repair_block = source[
        source.index("if a3b_target_prefix_route is not None:", rejection_start) :
        source.index("if rejection_correction is not None:", rejection_start)
    ]
    assert "committed.append(rejection_correction)" in exact_repair_block


def test_generation_exact_route_never_engages_with_penalties() -> None:
    """Penalties are host-side sampler state (running token counts). The
    compiled device-draft validator hard-fails on them, so a penalty-bearing
    request must steer to the eager host lane instead of the compiled route.
    Regression: a solo presence_penalty request on the composite daemon
    answered HTTP 500 (A3BCompiledTargetPrefixConfigError) because losing the
    ccopy takeover dropped it onto the compiled factory — while a penalty
    cohort worked fine via the batch scheduler's dense host fallback."""
    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    assert "_penalty_bearing_request = bool(sampler.presence_penalty) or bool(" in source
    factory_block = source[
        source.index("exact_a3b_target_prefix_factory = (") : source.index(
            "exact_a3b_target_prefix = "
        )
    ]
    assert "and not _penalty_bearing_request" in factory_block
    # The ccopy takeover keeps its own penalty gate; both lanes decline and the
    # request lands on the host target-prefix path, which samples penalties.
    ccopy_block = source[
        source.index("_ccopy_takes_over_lane = (") : source.index(
            "exact_a3b_target_prefix_factory = ("
        )
    ]
    assert "not _penalty_bearing_request" in ccopy_block
