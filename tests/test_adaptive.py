from types import SimpleNamespace

from mtplx.adaptive import (
    AdaptiveDepthPolicy,
    ExpectedValueDepthPolicy,
    PositionEMADepthPolicy,
)
from mtplx.benchmarks.runners import mtp_adaptive


def test_adaptive_policy_increases_after_full_accept_streak():
    policy = AdaptiveDepthPolicy(max_depth=4, start_depth=2, increase_after=2)

    first = policy.observe(attempted_depth=2, accepted_depths=2)
    second = policy.observe(attempted_depth=2, accepted_depths=2)

    assert first["action"] == "hold"
    assert second["action"] == "increase"
    assert second["next_depth"] == 3


def test_adaptive_policy_decreases_on_early_reject():
    policy = AdaptiveDepthPolicy(max_depth=5, start_depth=4, decrease_after=1)

    decision = policy.observe(attempted_depth=4, accepted_depths=0)

    assert decision["action"] == "decrease"
    assert decision["next_depth"] == 3


def test_adaptive_policy_clamps_start_depth():
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=2, start_depth=9)

    assert policy.current_depth == 3


def test_adaptive_policy_holds_on_late_reject():
    policy = AdaptiveDepthPolicy(max_depth=5, start_depth=4, decrease_after=1)

    decision = policy.observe(attempted_depth=4, accepted_depths=3)

    assert decision["action"] == "hold"
    assert decision["next_depth"] == 4


def test_position_ema_policy_matches_row11_initial_schedule():
    policy = PositionEMADepthPolicy(max_depth=8, depth_cap=4)

    # Source priors 0.85 * 0.98**i and h=0.20 clear every marginal gate
    # through the row-11 hard cap, independent of the larger offered width.
    assert policy.current_depth == 4
    assert policy.position_accept_ema == [
        0.85 * (0.98**index) for index in range(8)
    ]


def test_position_ema_policy_updates_only_observed_rejection_prefix():
    policy = PositionEMADepthPolicy(max_depth=8, depth_cap=4)
    before = list(policy.position_accept_ema)

    result = policy.observe(attempted_depth=4, accepted_depths=2)

    assert policy.position_accept_ema[:2] == [
        value + 0.15 * (1.0 - value) for value in before[:2]
    ]
    assert policy.position_accept_ema[2] == before[2] * 0.85
    assert policy.position_accept_ema[3:] == before[3:]
    assert result["next_depth"] == policy.current_depth


def test_position_ema_policy_transfers_full_accept_optimism():
    policy = PositionEMADepthPolicy(max_depth=8, depth_cap=4)
    before = list(policy.position_accept_ema)

    policy.observe(attempted_depth=2, accepted_depths=2)

    assert policy.position_accept_ema[0] > before[0]
    assert policy.position_accept_ema[1] > before[1]
    assert policy.position_accept_ema[2] > before[2]
    assert policy.position_accept_ema[3:] == before[3:]


def test_position_ema_policy_can_choose_true_serial_depth_zero():
    policy = PositionEMADepthPolicy(max_depth=4, depth_cap=4, min_depth=0)
    policy.position_accept_ema[0] = 0.19

    policy.recompute_depth()

    assert policy.current_depth == 0


def test_position_ema_policy_honors_minimum_depth():
    policy = PositionEMADepthPolicy(max_depth=3, depth_cap=3, min_depth=2)
    policy.position_accept_ema[0] = 0.0

    policy.recompute_depth()

    assert policy.current_depth == 2
    assert policy.allows_depth_zero is False


def test_position_ema_policy_probes_after_bounded_serial_cycles():
    policy = PositionEMADepthPolicy(
        max_depth=3,
        depth_cap=3,
        min_depth=0,
        serial_probe_interval=3,
    )
    policy.position_accept_ema[0] = 0.0
    policy.recompute_depth()

    first = policy.observe_serial_skip()
    second = policy.observe_serial_skip()
    probe = policy.observe_serial_skip()

    assert first["next_depth"] == 0
    assert second["next_depth"] == 0
    assert probe["next_depth"] == 1
    assert probe["action"] == "probe"


def test_expected_value_policy_stops_d3_when_ev_fails():
    policy = ExpectedValueDepthPolicy(
        max_depth=3,
        base_depth=2,
        accept_priors=(0.92, 0.64, 0.32),
        draft_cost_s=0.0048,
        extra_verify_cost_s=0.006,
        baseline_tok_s=40.0,
        safety_margin=0.1,
        warmup_full_depth_cycles=0,
        exploration_interval=0,
    )
    policy.observe(attempted_depth=3, accepted_depths=0)

    decision = policy.should_continue_after_draft(
        drafted_depth=2,
        max_depth=3,
        draft_metrics={"top2_margin": 0.5, "top1_prob_topk": 0.35},
    )

    assert decision["action"] == "stop"
    assert decision["reason"] == "ev_fail"
    assert decision["expected_extra_accept"] < decision["required_extra_accept"]


def test_expected_value_policy_allows_d3_when_ev_clears_cost():
    policy = ExpectedValueDepthPolicy(
        max_depth=3,
        base_depth=2,
        accept_priors=(0.99, 0.96, 0.9),
        draft_cost_s=0.002,
        extra_verify_cost_s=0.002,
        baseline_tok_s=40.0,
        safety_margin=0.05,
        warmup_full_depth_cycles=0,
        exploration_interval=0,
    )
    policy.observe(attempted_depth=3, accepted_depths=3)

    decision = policy.should_continue_after_draft(
        drafted_depth=2,
        max_depth=3,
        draft_metrics={"top2_margin": 6.0, "top1_prob_topk": 0.9},
    )

    assert decision["action"] == "continue"
    assert decision["reason"] == "ev_pass"


def test_expected_value_policy_can_gate_d2_when_base_depth_is_one():
    policy = ExpectedValueDepthPolicy(
        max_depth=3,
        base_depth=1,
        accept_priors=(0.92, 0.64, 0.32),
        draft_cost_s=0.0048,
        extra_verify_cost_s=0.006,
        baseline_tok_s=40.0,
        safety_margin=0.1,
        warmup_full_depth_cycles=0,
        exploration_interval=0,
    )
    for _ in range(4):
        policy.observe(attempted_depth=2, accepted_depths=1)

    decision = policy.should_continue_after_draft(
        drafted_depth=1,
        max_depth=3,
        draft_metrics={"top2_margin": 0.5, "top1_prob_topk": 0.35},
    )

    assert decision["action"] == "stop"
    assert decision["reason"] == "ev_fail"
    assert decision["next_depth"] == 2


def test_expected_value_policy_keeps_d2_when_base_depth_one_ev_clears():
    policy = ExpectedValueDepthPolicy(
        max_depth=3,
        base_depth=1,
        accept_priors=(0.99, 0.96, 0.9),
        draft_cost_s=0.0048,
        extra_verify_cost_s=0.006,
        baseline_tok_s=40.0,
        safety_margin=0.1,
        warmup_full_depth_cycles=0,
        exploration_interval=0,
    )
    policy.observe(attempted_depth=2, accepted_depths=2)

    decision = policy.should_continue_after_draft(
        drafted_depth=1,
        max_depth=3,
        draft_metrics={"top2_margin": 4.0, "top1_prob_topk": 0.8},
    )

    assert decision["action"] == "continue"
    assert decision["reason"] == "ev_pass"
    assert decision["next_depth"] == 2


def test_expected_value_policy_updates_acceptance_ewma():
    policy = ExpectedValueDepthPolicy(max_depth=3, accept_priors=(0.5, 0.5, 0.5), ewma_alpha=0.5)

    decision = policy.observe(attempted_depth=2, accepted_depths=1)

    assert decision["kind"] == "expected_value"
    assert decision["accept_ewma"][:2] == [0.75, 0.25]
    assert decision["cycles_observed"] == 1
    assert decision["attempt_counts"][:2] == [1, 1]


def test_adaptive_runner_passes_adapter_and_step_contract(monkeypatch, tmp_path):
    calls = []
    fake_runtime = SimpleNamespace(
        tokenizer=object(),
        contract=SimpleNamespace(
            base_hidden_variant="pre_norm",
            hidden_variant="pre_norm",
            concat_order="embedding_hidden",
        ),
        mtp_adapter_metadata={"kind": "c4_mtp_lora_adapter"},
        mtp_adapter_merge_report={"merged": 1},
    )

    def fake_load(*_args, **kwargs):
        calls.append(kwargs)
        return fake_runtime

    monkeypatch.setattr(mtp_adaptive, "load", fake_load)
    monkeypatch.setattr(mtp_adaptive, "load_prompt_suite", lambda *_args, **_kwargs: [])

    result = mtp_adaptive.run_mtp_adaptive(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        max_depth=2,
        base_hidden_variant="pre_norm",
        mtp_hidden_variant="pre_norm",
        concat_order="embedding_hidden",
        mtp_adapter_path=tmp_path / "adapter.npz",
        merge_mtp_adapter=True,
    )

    assert calls[0]["mtp_adapter"] == tmp_path / "adapter.npz"
    assert calls[0]["merge_mtp_adapter"] is True
    assert calls[0]["contract"].base_hidden_variant == "pre_norm"
    assert calls[0]["contract"].hidden_variant == "pre_norm"
    assert result["mtp_adapter_kind"] == "c4_mtp_lora_adapter"
    assert result["mtp_adapter_merged"] is True


def test_expected_value_policy_warms_up_full_depth_before_ev_gate():
    policy = ExpectedValueDepthPolicy(
        max_depth=3,
        base_depth=2,
        accept_priors=(0.92, 0.64, 0.32),
        draft_cost_s=0.0048,
        extra_verify_cost_s=0.006,
        baseline_tok_s=40.0,
        safety_margin=0.1,
        warmup_full_depth_cycles=2,
        exploration_interval=0,
    )

    first = policy.should_continue_after_draft(
        drafted_depth=2,
        max_depth=3,
        draft_metrics={"top2_margin": 0.5, "top1_prob_topk": 0.35},
    )
    assert first["action"] == "continue"
    assert first["reason"] == "warmup_full_depth"
    policy.observe(attempted_depth=3, accepted_depths=3)

    second = policy.should_continue_after_draft(
        drafted_depth=2,
        max_depth=3,
        draft_metrics={"top2_margin": 0.5, "top1_prob_topk": 0.35},
    )
    assert second["action"] == "continue"
    assert second["reason"] == "warmup_full_depth"
    policy.observe(attempted_depth=3, accepted_depths=0)

    third = policy.should_continue_after_draft(
        drafted_depth=2,
        max_depth=3,
        draft_metrics={"top2_margin": 0.5, "top1_prob_topk": 0.35},
    )
    assert third["action"] == "stop"
    assert third["reason"] == "ev_fail"
