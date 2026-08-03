"""Contracts for the preregistered DeepSeek-V4 adaptive max-K3 policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx import generation
from mtplx.deepseek_v4_adaptive_width import (  # noqa: E402
    D1_MARGIN_THRESHOLD,
    D2_MARGIN_THRESHOLD,
    MAX_SPECULATIVE_DEPTH,
    install_deepseek_v4_adaptive_width_policy,
)
from mtplx.sampling import SamplerConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _canonical_route_report() -> dict:
    return {
        "mode": "gather_qmm",
        "module_count": 44,
        "trunk_module_count": 43,
        "mtp_module_count": 1,
        "body_direct": 43,
        "mtp_stock": 1,
        "body_all_mode_matches": True,
        "route_plan_matches": True,
        "callable_census": {
            "body_route_objects": 43,
            "body_route_kind": "gather_qmm_m4_wide_direct",
            "body_callable_class": "_DirectGatherOLoraWideM4",
            "mtp_route_objects": 1,
            "mtp_route_kind": "dense_bf16_stock_direct",
            "mtp_callable_class": "_DirectDenseMTPOLora",
            "total_route_objects": 44,
            "unique_route_objects": 44,
            "mtp_distinct_type": True,
        },
    }


def _tiny_runtime_and_prompt():
    from test_deepseek_v4_spec import _prompt, _runtime

    rt = _runtime(vocab=8)
    rt.deepseek_v4_o_lora_report = _canonical_route_report()
    return rt, _prompt(17, vocab=8)


def _install(rt, *, sampler=None, draft_sampler=None, depth=3):
    return install_deepseek_v4_adaptive_width_policy(
        rt,
        sampler=sampler or SamplerConfig(temperature=0.0),
        draft_sampler=draft_sampler,
        speculative_depth=depth,
        verify_strategy="capture_commit",
        verify_core="stock",
        mtp_history_policy="committed",
    )


def _adaptive(rt, prompt, *, policy, max_tokens=16, stop_token_ids=None):
    return generation.generate_mtpk(
        rt,
        prompt,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=3,
        verify_strategy="capture_commit",
        verify_core="stock",
        mtp_history_policy="committed",
        stop_token_ids=set() if stop_token_ids is None else stop_token_ids,
        adaptive_width_policy=policy,
    )


def test_policy_is_frozen_preregistered_and_ties_continue_deeper():
    rt, _ = _tiny_runtime_and_prompt()
    policy = _install(rt)

    assert D1_MARGIN_THRESHOLD == 0.25
    assert D2_MARGIN_THRESHOLD == 10.0
    assert MAX_SPECULATIVE_DEPTH == 3
    assert policy.d1_margin_threshold == 0.25
    assert policy.d2_margin_threshold == 10.0
    assert policy.max_speculative_depth == 3
    assert policy.stop_after_d1(0.249999) is True
    assert policy.stop_after_d1(0.25) is False
    assert policy.stop_after_d2(9.999999) is True
    assert policy.stop_after_d2(10.0) is False
    with pytest.raises(FrozenInstanceError):
        policy.d1_margin_threshold = 0.5
    parameters = inspect.signature(type(policy)).parameters
    assert "d1_margin_threshold" not in parameters
    assert "d2_margin_threshold" not in parameters
    assert "max_speculative_depth" not in parameters


def test_policy_type_is_private_and_factory_only():
    rt, _ = _tiny_runtime_and_prompt()
    policy = _install(rt)

    assert type(policy).__name__.startswith("_")
    with pytest.raises(TypeError):
        type(policy)(
            runtime_object_id=id(rt),
            target_routes=policy.target_routes,
        )


def test_hand_forged_policy_object_fails_before_prefill(monkeypatch):
    rt, prompt = _tiny_runtime_and_prompt()

    class ForgedPolicy:
        d1_margin_threshold = 0.25
        d2_margin_threshold = 10.0
        max_speculative_depth = 3
        target_routes = (lambda *_a, **_k: None,) * 3

        def validate_request(self, *_args, **_kwargs):
            return None

        def stop_after_d1(self, margin):
            return margin < 0.25

        def stop_after_d2(self, margin):
            return margin < 10.0

    monkeypatch.setattr(
        generation,
        "restore_or_prefill_prompt_state",
        lambda *_a, **_k: pytest.fail("forged object reached prefill"),
    )

    with pytest.raises(ValueError, match="factory-installed"):
        generation.generate_mtpk(
            rt,
            prompt,
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0),
            speculative_depth=3,
            verify_strategy="capture_commit",
            verify_core="stock",
            mtp_history_policy="committed",
            stop_token_ids=set(),
            adaptive_width_policy=ForgedPolicy(),
        )


@pytest.mark.parametrize(
    "forgery",
    ("runtime_id", "capture_callable", "physical_rows", "gather_report"),
)
def test_forged_policy_authority_fails_before_prefill(monkeypatch, forgery):
    rt, prompt = _tiny_runtime_and_prompt()
    policy = _install(rt)
    selected_rt = rt
    if forgery == "runtime_id":
        selected_rt, _ = _tiny_runtime_and_prompt()
        selected_rt.deepseek_v4_o_lora_report = _canonical_route_report()
        object.__setattr__(policy, "runtime_object_id", id(selected_rt))
    elif forgery == "capture_callable":
        object.__setattr__(policy.target_routes[0], "forward", lambda *_a, **_k: None)
    elif forgery == "physical_rows":
        object.__setattr__(policy.target_routes[1], "expected_physical_rows", 4)
    else:
        rt.deepseek_v4_o_lora_report["callable_census"]["mtp_route_kind"] = "stock"

    monkeypatch.setattr(
        generation,
        "restore_or_prefill_prompt_state",
        lambda *_a, **_k: pytest.fail("forgery reached prefill"),
    )
    with pytest.raises(ValueError, match="adaptive width policy"):
        generation.generate_mtpk(
            selected_rt,
            prompt,
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0),
            speculative_depth=3,
            verify_strategy="capture_commit",
            verify_core="stock",
            mtp_history_policy="committed",
            stop_token_ids=set(),
            adaptive_width_policy=policy,
        )


@pytest.mark.parametrize(
    ("target_temp", "draft_temp", "depth", "match"),
    [
        (0.2, 0.0, 3, "greedy target"),
        (0.0, 0.2, 3, "greedy draft"),
        (0.0, 0.0, 2, "max-K3"),
        (0.0, 0.0, 4, "max-K3"),
    ],
)
def test_install_rejects_temperature_and_non_k3(target_temp, draft_temp, depth, match):
    rt, _ = _tiny_runtime_and_prompt()
    with pytest.raises(ValueError, match=match):
        _install(
            rt,
            sampler=SamplerConfig(temperature=target_temp),
            draft_sampler=SamplerConfig(temperature=draft_temp),
            depth=depth,
        )


def test_install_fails_closed_on_runtime_identity_and_route_plan():
    rt, _ = _tiny_runtime_and_prompt()
    rt.model.model_type = "not_deepseek_v4"
    with pytest.raises(ValueError, match="DeepSeek-V4"):
        _install(rt)

    rt, _ = _tiny_runtime_and_prompt()
    rt.deepseek_v4_o_lora_report["mtp_stock"] = 0
    with pytest.raises(ValueError, match="o-LoRA route"):
        _install(rt)


def test_greedy_token_and_fp32_top2_share_one_eval_without_hidden(monkeypatch):
    original_eval = generation._eval
    calls = []

    def audited_eval(*values, **kwargs):
        calls.append(values)
        return original_eval(*values, **kwargs)

    monkeypatch.setattr(generation, "_eval", audited_eval)
    logits = mx.array([[[1.0, 4.0, 3.25, -2.0]]], dtype=mx.float16)
    hidden = mx.zeros((1, 1, 32), dtype=mx.float32)

    token, top1, top2 = generation._greedy_draft_token_and_top2(logits)

    assert (token, top1, top2) == pytest.approx((1, 4.0, 3.25))
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert all(value is not hidden for value in calls[0])
    assert calls[0][0].ndim == 0
    assert tuple(calls[0][1].shape) == (2,)
    assert calls[0][1].dtype == mx.float32


def test_selected_width_mix_uses_one_target_verify_per_cycle(monkeypatch):
    rt, prompt = _tiny_runtime_and_prompt()
    policy = _install(rt)
    original = generation._greedy_draft_token_and_top2
    margins = iter((0.10, 0.50, 0.50, 0.50, 10.50) * 20)

    def scripted_margin(logits):
        token, top1, _top2 = original(logits)
        margin = next(margins)
        return token, top1, top1 - margin

    monkeypatch.setattr(generation, "_greedy_draft_token_and_top2", scripted_margin)
    out = _adaptive(rt, prompt, policy=policy, max_tokens=24)
    policy_events = [
        event["adaptive_width_policy"]
        for event in out.stats.events
        if "adaptive_width_policy" in event
    ]
    widths = [event["selected_draft_depth"] for event in policy_events]

    assert {1, 2, 3} <= set(widths)
    assert len(widths) == out.stats.verify_calls
    assert sum(width == 1 for width in widths) == out.stats.drafted_by_depth[0] - out.stats.drafted_by_depth[1]
    assert sum(width == 2 for width in widths) == out.stats.drafted_by_depth[1] - out.stats.drafted_by_depth[2]
    assert sum(width == 3 for width in widths) == out.stats.drafted_by_depth[2]
    assert all(event["target_rows"] == event["selected_draft_depth"] + 1 for event in policy_events)


def test_target_corrections_preserve_authoritative_ar_sequence(monkeypatch):
    rt, prompt = _tiny_runtime_and_prompt()
    from mtplx.generation import generate_ar

    baseline = generate_ar(
        rt,
        prompt,
        max_tokens=32,
        sampler=SamplerConfig(temperature=0.0),
        stop_token_ids=set(),
    )
    policy = _install(rt)
    original = generation._greedy_draft_token_and_top2
    monkeypatch.setattr(
        generation,
        "_greedy_draft_token_and_top2",
        lambda logits: (lambda row: (row[0], row[1], row[1] - 0.10))(original(logits)),
    )
    out = _adaptive(rt, prompt, policy=policy, max_tokens=32)

    assert out.stats.rejected_drafts > 0
    assert out.tokens == baseline.tokens


def test_terminal_primary_never_enters_adaptive_draft_path(monkeypatch):
    rt, prompt = _tiny_runtime_and_prompt()
    from mtplx.generation import generate_ar

    baseline = generate_ar(
        rt,
        prompt,
        max_tokens=1,
        sampler=SamplerConfig(temperature=0.0),
        stop_token_ids=set(),
    )
    policy = _install(rt)
    monkeypatch.setattr(
        generation,
        "_greedy_draft_token_and_top2",
        lambda *_args, **_kwargs: pytest.fail("terminal cycle must not draft"),
    )
    out = _adaptive(
        rt,
        prompt,
        policy=policy,
        max_tokens=8,
        stop_token_ids={baseline.tokens[0]},
    )

    assert out.tokens == [baseline.tokens[0]]
    assert out.finish_reason == "stop"
    assert out.stats.verify_calls == 0


@pytest.mark.parametrize(
    ("max_tokens", "draft_depth", "physical_rows"),
    ((2, 1, [2]), (3, 2, [3, 2])),
)
def test_terminal_tail_readers_are_explicit_and_target_correct(
    monkeypatch, max_tokens, draft_depth, physical_rows
):
    rt, prompt = _tiny_runtime_and_prompt()
    from mtplx.generation import generate_ar

    baseline = generate_ar(
        rt,
        prompt,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0),
        stop_token_ids=set(),
    )
    captured_rows = []
    runtime_type = type(rt)
    original_capture = runtime_type.forward_ar_capture

    def audited_capture(self, input_ids, **kwargs):
        captured_rows.append(int(input_ids.shape[1]))
        return original_capture(self, input_ids, **kwargs)

    monkeypatch.setattr(runtime_type, "forward_ar_capture", audited_capture)
    policy = _install(rt)
    original_sample = generation._sample_draft_from_logits

    def force_wrong_draft(logits, config, rng, *, need_distribution):
        _token, _distribution = original_sample(
            logits,
            config,
            rng,
            need_distribution=need_distribution,
        )
        wrong_token = (int(baseline.tokens[1]) + 1) % int(logits.shape[-1])
        return wrong_token, None

    monkeypatch.setattr(generation, "_sample_draft_from_logits", force_wrong_draft)
    monkeypatch.setattr(
        generation,
        "_fixed_width_draft_reader",
        lambda *_a, **_k: pytest.fail("adaptive tail used fixed-reader fallback"),
        raising=False,
    )

    out = _adaptive(rt, prompt, policy=policy, max_tokens=max_tokens)
    policy_events = [
        event["adaptive_width_policy"]
        for event in out.stats.events
        if "adaptive_width_policy" in event
    ]

    assert out.tokens == baseline.tokens
    assert out.stats.rejected_drafts >= 1
    assert captured_rows == physical_rows
    assert policy_events
    assert all(event["eligible_full_k3"] is False for event in policy_events)
    assert policy_events[0]["selected_draft_depth"] == draft_depth
    assert all(event["decision_margins"] == [] for event in policy_events)
    correction = next(
        event["drafts"][0]["correction"]
        for event in out.stats.events
        if event.get("rejected_at_depth") == 1
    )
    assert correction == baseline.tokens[1]


def test_default_fixed_k3_remains_argmax_only(monkeypatch):
    def fail_if_policy_helper(*_args, **_kwargs):
        raise AssertionError("ordinary fixed K3 must remain argmax-only")

    monkeypatch.setattr(
        generation, "_greedy_draft_token_and_top2", fail_if_policy_helper
    )
    rt, prompt = _tiny_runtime_and_prompt()
    out = generation.generate_mtpk(
        rt,
        prompt,
        max_tokens=8,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=3,
        verify_strategy="capture_commit",
        verify_core="stock",
        mtp_history_policy="committed",
        stop_token_ids=set(),
    )

    assert out.tokens
    assert not any("adaptive_width_policy" in event for event in out.stats.events)


def test_policy_source_has_no_environment_reads_fallback_or_mutable_counters():
    import mtplx.deepseek_v4_adaptive_width as policy_module

    source = inspect.getsource(policy_module)
    forbidden = (
        "os.environ",
        "getenv(",
        "fallback",
        "diagnostic_counters",
        "try:",
    )
    assert all(fragment not in source for fragment in forbidden)


def test_decode_loop_uses_prebound_policy_surfaces():
    source = inspect.getsource(generation.generate_mtpk)
    decode_loop = source.split("while len(tokens) < max_tokens:", 1)[1]

    assert "adaptive_width_policy" not in decode_loop


def test_enabled_readers_have_no_fixed_reader_fallback():
    source = inspect.getsource(generation.generate_mtpk)
    enabled_setup = source.split("else:\n        adaptive_width_margin_stops", 1)[1]
    enabled_setup = enabled_setup.split("if mtp_corrector is not None:", 1)[0]

    assert "_fixed_width_draft_reader" not in enabled_setup
