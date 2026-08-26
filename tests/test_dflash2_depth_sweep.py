import json
from types import SimpleNamespace

import pytest

from mtplx.benchmarks.runners import dflash2_depth_sweep as runner


def _summary(**overrides):
    from dflash_mlx.engine.events import SummaryEvent

    values = {
        "elapsed_us": 3_000_000.0,
        "prompt_token_count": 3,
        "generated_token_ids": (11, 12),
        "generation_tokens": 2,
        "accepted_from_draft": 1,
        "acceptance_ratio": 0.5,
        "cycles_completed": 1,
        "phase_timings_us": {"prefill": 1_000_000.0},
        "block_tokens": 8,
        "peak_memory_gb": 12.5,
        "acceptance_history": (1,),
        "fallback_ar": False,
        "fallback_reason": None,
    }
    values.update(overrides)
    return SummaryEvent(**values)


def test_target_oracle_uses_exact_greedy_contract(monkeypatch):
    calls = []
    output = SimpleNamespace(tokens=[7, 8])

    def fake_generate(runtime, prompt_ids, **kwargs):
        calls.append((runtime, prompt_ids, kwargs))
        return output

    monkeypatch.setattr(runner, "_generate_ar", fake_generate)
    bundle = SimpleNamespace(runtime=object())

    assert runner.run_target_oracle(bundle, (1, 2, 3), max_tokens=2) == (7, 8)
    assert calls == [
        (
            bundle.runtime,
            [1, 2, 3],
            {
                "max_tokens": 2,
                "sampler": runner.GREEDY,
                "seed": 0,
                "stop_token_ids": set(),
            },
        )
    ]


def test_greedy_contract_standardizes_temperature_at_one():
    assert runner.GREEDY.temperature == 1.0
    assert runner.GREEDY.top_p == 1.0
    assert runner.GREEDY.top_k == 1


def test_target_oracle_rejects_short_output(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_generate_ar",
        lambda *_args, **_kwargs: SimpleNamespace(tokens=[7]),
    )
    with pytest.raises(RuntimeError, match="forced token count"):
        runner.run_target_oracle(SimpleNamespace(runtime=object()), (1,), max_tokens=2)


def test_mtp_control_uses_promoted_depth_three_contract(monkeypatch):
    calls = []
    stats = SimpleNamespace(
        generated_tokens=2,
        decode_tok_s=41.5,
        elapsed_s=0.08,
        decode_elapsed_s=0.05,
        peak_memory_bytes=3 * 1024**3,
        verify_calls=1,
        accepted_by_depth=[1, 0, 0],
    )

    def fake_generate(runtime, prompt_ids, **kwargs):
        calls.append((runtime, prompt_ids, kwargs))
        return SimpleNamespace(tokens=[7, 8], stats=stats)

    monkeypatch.setattr(runner, "_generate_mtpk", fake_generate)
    bundle = SimpleNamespace(runtime=object())
    receipt = runner.run_mtp_control(bundle, (1, 2), max_tokens=2)

    metrics = {
        name: receipt.pop(name)
        for name in (
            "prefill_s",
            "prefill_tps",
            "spec_decode_hit_rate",
        )
    }
    assert receipt == {
        "tokens": (7, 8),
        "generated_tokens": 2,
        "decode_tps": 41.5,
        "elapsed_s": 0.08,
        "decode_elapsed_s": 0.05,
        "peak_memory_gb": 3.0,
        "verify_calls": 1,
        "accepted_by_depth": [1, 0, 0],
        "accepted_from_draft": 1,
        "engine": "mtplx_mtp",
    }
    assert metrics == pytest.approx(
        {
            "prefill_s": 0.03,
            "prefill_tps": 2 / 0.03,
            "spec_decode_hit_rate": 0.5,
        }
    )
    assert calls[0][0] is bundle.runtime
    assert calls[0][1] == [1, 2]
    assert calls[0][2] == {
        "max_tokens": 2,
        "sampler": runner.GREEDY,
        "speculative_depth": 3,
        "seed": 0,
        "stop_token_ids": set(),
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "cycle",
    }


def test_fixed_dflash_context_uses_closed_settings(monkeypatch):
    calls = []
    expected = object()

    def fake_build(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(runner, "_build_offline_runtime_context", fake_build)
    assert runner.build_fixed_dflash_runtime_context() is expected
    assert calls == [
        {
            "quantize_kv_cache": False,
            "verify_mode": "dflash",
            "copyspec_mode": "off",
        }
    ]


def test_dflash_candidate_passes_exact_bundle_and_width(monkeypatch):
    calls = []
    summary = _summary()

    def fake_stream(**kwargs):
        calls.append(kwargs)
        return iter((summary,))

    monkeypatch.setattr(runner, "_stream_dflash_generate", fake_stream)
    bundle = SimpleNamespace(
        target_model=object(),
        target_ops=object(),
        tokenizer=object(),
        draft_model=object(),
        draft_backend=object(),
    )
    context = object()
    receipt = runner.run_dflash2_candidate(
        bundle,
        (1, 2, 3),
        8,
        context,
        max_tokens=2,
    )

    assert receipt["tokens"] == (11, 12)
    assert receipt["decode_tps"] == 1.0
    assert receipt["requested_width"] == receipt["effective_width"] == 8
    assert calls == [
        {
            "target_model": bundle.target_model,
            "target_ops": bundle.target_ops,
            "tokenizer": bundle.tokenizer,
            "draft_model": bundle.draft_model,
            "draft_backend": bundle.draft_backend,
            "prompt_tokens_override": [1, 2, 3],
            "prompt": "",
            "use_chat_template": False,
            "max_new_tokens": 2,
            "block_tokens": 8,
            "stop_token_ids": [],
            "runtime_context": context,
        }
    ]


def test_dflash_summary_adapter_returns_measured_metrics():
    receipt = runner.arm_receipt_from_dflash_events(
        (_summary(),),
        requested_width=8,
        expected_tokens=2,
    )
    assert receipt == {
        "tokens": (11, 12),
        "generated_tokens": 2,
        "decode_tps": 1.0,
        "elapsed_s": 3.0,
        "prefill_s": 1.0,
        "prefill_tps": 3.0,
        "decode_elapsed_s": 2.0,
        "peak_memory_gb": 12.5,
        "cycles_completed": 1,
        "accepted_from_draft": 1,
        "acceptance_ratio": 0.5,
        "spec_decode_hit_rate": 0.5,
        "acceptance_history": [1],
        "requested_width": 8,
        "effective_width": 8,
        "fallback_ar": False,
        "fallback_reason": None,
        "engine": "dflash_mlx_0_1_10",
    }


def test_dflash_summary_adapter_requires_one_summary():
    with pytest.raises(RuntimeError, match="without exactly one summary"):
        runner.arm_receipt_from_dflash_events((), requested_width=8, expected_tokens=2)
    with pytest.raises(RuntimeError, match="without exactly one summary"):
        runner.arm_receipt_from_dflash_events(
            (_summary(), _summary()),
            requested_width=8,
            expected_tokens=2,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"elapsed_us": 1_000_000.0}, "positive decode duration"),
        ({"elapsed_us": float("nan")}, "positive decode duration"),
        ({"block_tokens": 7}, "requested width 8 became 7"),
        ({"fallback_ar": True, "fallback_reason": "fallback"}, "fallback"),
        ({"generation_tokens": 1, "generated_token_ids": (11,)}, "forced token count"),
        ({"generation_tokens": 2, "generated_token_ids": (11,)}, "token ID count"),
    ],
)
def test_dflash_summary_adapter_rejects_invalid_result(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        runner.arm_receipt_from_dflash_events(
            (_summary(**overrides),),
            requested_width=8,
            expected_tokens=2,
        )


def test_existing_receipt_can_be_enriched_with_prefill_and_hit_rate_metrics():
    receipt = {
        "workload": {"prompt_tokens": 1024},
        "brackets": [
            {
                "control_before": {
                    "engine": "mtplx_mtp",
                    "elapsed_s": 30.0,
                    "decode_elapsed_s": 25.0,
                    "generated_tokens": 1024,
                    "accepted_by_depth": [200, 100, 50],
                },
                "candidate": {
                    "engine": "dflash_mlx_0_1_10",
                    "prefill_s": 4.0,
                    "generated_tokens": 1024,
                    "accepted_from_draft": 640,
                },
                "control_after": {
                    "engine": "mtplx_mtp",
                    "elapsed_s": 31.0,
                    "decode_elapsed_s": 26.0,
                    "generated_tokens": 1024,
                    "accepted_by_depth": [200, 100, 50],
                },
            }
        ],
    }

    enriched = runner.enrich_depth_sweep_metrics(receipt)

    assert enriched is receipt
    before = receipt["brackets"][0]["control_before"]
    candidate = receipt["brackets"][0]["candidate"]
    assert before["prefill_s"] == 5.0
    assert before["prefill_tps"] == 1024 / 5.0
    assert before["accepted_from_draft"] == 350
    assert before["spec_decode_hit_rate"] == 350 / 1024
    assert candidate["prefill_tps"] == 256.0
    assert candidate["spec_decode_hit_rate"] == 0.625


def _fake_arm(
    oracle,
    calls,
    *,
    short_width=None,
    divergent_width=None,
    fallback_width=None,
):
    def run(kind, width):
        calls.append((kind, width))
        tokens = oracle
        if kind == "dflash2" and width == short_width:
            tokens = oracle[:-1]
        if kind == "dflash2" and width == divergent_width:
            tokens = (*oracle[:-1], oracle[-1] + 1)
        receipt = {
            "tokens": tokens,
            "generated_tokens": len(tokens),
            "decode_tps": 60.0 if kind == "mtp" else 61.0 + width,
        }
        if kind == "dflash2":
            receipt.update(
                requested_width=width,
                effective_width=width,
                fallback_ar=width == fallback_width,
            )
        return receipt

    return run


def test_sweep_rotates_widths_and_brackets_every_candidate():
    oracle = tuple(range(4))
    calls = []
    receipt = runner.run_dflash2_depth_sweep(
        bundle=object(),
        prompt_ids=(9, 8),
        widths=(1, 2, 3),
        repetitions=2,
        max_tokens=4,
        oracle_tokens=oracle,
        arm_runner=_fake_arm(oracle, calls),
    )

    assert calls == [
        ("mtp", 3), ("dflash2", 1), ("mtp", 3),
        ("mtp", 3), ("dflash2", 2), ("mtp", 3),
        ("mtp", 3), ("dflash2", 3), ("mtp", 3),
        ("mtp", 3), ("dflash2", 2), ("mtp", 3),
        ("mtp", 3), ("dflash2", 3), ("mtp", 3),
        ("mtp", 3), ("dflash2", 1), ("mtp", 3),
    ]
    assert receipt["workload"] == {
        "prompt_tokens": 2,
        "generated_tokens": 4,
        "greedy": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 1,
    }
    assert len(receipt["brackets"]) == 6
    assert all(row["validation_passed"] for row in receipt["brackets"])
    assert all(row["token_parity_passed"] for row in receipt["brackets"])
    assert receipt["determinism"] == {
        "control_stable": True,
        "candidate_repeats_checked": True,
        "candidate_stable_by_width": {"1": True, "2": True, "3": True},
        "passed": True,
    }
    assert receipt["selection"] is not None
    assert "tokens" not in receipt["brackets"][0]["candidate"]
    assert len(receipt["brackets"][0]["candidate"]["token_sha256"]) == 64
    assert receipt["brackets"][0]["candidate"]["oracle_comparison"] == {
        "exact_match": True,
        "matching_prefix_tokens": 4,
        "first_mismatch": None,
    }
    json.dumps(receipt)


@pytest.mark.parametrize("failure", ["short", "fallback", "width"])
def test_sweep_rejects_invalid_candidate_before_selection(failure):
    oracle = tuple(range(4))
    calls = []
    arm = _fake_arm(
        oracle,
        calls,
        short_width=8 if failure == "short" else None,
        fallback_width=8 if failure == "fallback" else None,
    )

    if failure == "width":
        base = arm

        def arm(kind, width):
            receipt = base(kind, width)
            if kind == "dflash2":
                receipt["effective_width"] = 7
            return receipt

    receipt = runner.run_dflash2_depth_sweep(
        bundle=object(),
        prompt_ids=(1,),
        widths=(8,),
        repetitions=1,
        max_tokens=4,
        oracle_tokens=oracle,
        arm_runner=arm,
    )
    assert receipt["selection"] is None
    assert receipt["brackets"][0]["validation_passed"] is False


def test_sweep_records_token_divergence_without_blocking_selection():
    oracle = tuple(range(4))
    receipt = runner.run_dflash2_depth_sweep(
        bundle=object(),
        prompt_ids=(1,),
        widths=(8,),
        repetitions=1,
        max_tokens=4,
        oracle_tokens=oracle,
        arm_runner=_fake_arm(oracle, [], divergent_width=8),
    )

    assert receipt["selection"] is not None
    assert receipt["brackets"][0]["validation_passed"] is True
    assert receipt["brackets"][0]["token_parity_passed"] is False
    assert receipt["brackets"][0]["candidate"]["oracle_comparison"] == {
        "exact_match": False,
        "matching_prefix_tokens": 3,
        "first_mismatch": {
            "index": 3,
            "oracle_token": 3,
            "arm_token": 4,
        },
    }


def test_sweep_rejects_nondeterministic_candidate_repetitions():
    oracle = tuple(range(4))
    candidate_calls = 0

    def arm(kind, width):
        nonlocal candidate_calls
        tokens = oracle
        if kind == "dflash2":
            candidate_calls += 1
            if candidate_calls == 2:
                tokens = (*oracle[:-1], 99)
        receipt = {
            "tokens": tokens,
            "generated_tokens": len(tokens),
            "decode_tps": 60.0,
        }
        if kind == "dflash2":
            receipt.update(
                requested_width=width,
                effective_width=width,
                fallback_ar=False,
            )
        return receipt

    receipt = runner.run_dflash2_depth_sweep(
        bundle=object(),
        prompt_ids=(1,),
        widths=(8,),
        repetitions=2,
        max_tokens=4,
        oracle_tokens=oracle,
        arm_runner=arm,
    )

    assert receipt["selection"] is None
    assert receipt["determinism"] == {
        "control_stable": True,
        "candidate_repeats_checked": True,
        "candidate_stable_by_width": {"8": False},
        "passed": False,
    }


def test_sweep_rejects_nondeterministic_mtp_controls():
    oracle = tuple(range(4))
    control_calls = 0

    def arm(kind, width):
        nonlocal control_calls
        tokens = oracle
        if kind == "mtp":
            control_calls += 1
            if control_calls == 2:
                tokens = (*oracle[:-1], 99)
        receipt = {
            "tokens": tokens,
            "generated_tokens": len(tokens),
            "decode_tps": 60.0,
        }
        if kind == "dflash2":
            receipt.update(
                requested_width=width,
                effective_width=width,
                fallback_ar=False,
            )
        return receipt

    receipt = runner.run_dflash2_depth_sweep(
        bundle=object(),
        prompt_ids=(1,),
        widths=(8,),
        repetitions=1,
        max_tokens=4,
        oracle_tokens=oracle,
        arm_runner=arm,
    )

    assert receipt["selection"] is None
    assert receipt["determinism"]["control_stable"] is False
    assert receipt["determinism"]["passed"] is False


def test_sweep_production_path_warms_each_width_and_propagates_smoke_budget(monkeypatch):
    oracle = tuple(range(4))
    calls = {"oracle": [], "mtp": [], "dflash": [], "context": 0}
    context = object()

    def fake_oracle(bundle, prompt_ids, *, max_tokens):
        calls["oracle"].append((bundle, tuple(prompt_ids), max_tokens))
        return oracle

    def fake_mtp(bundle, prompt_ids, *, max_tokens):
        calls["mtp"].append((bundle, tuple(prompt_ids), max_tokens))
        return {"tokens": oracle, "generated_tokens": max_tokens, "decode_tps": 60.0}

    def fake_dflash(bundle, prompt_ids, width, runtime_context, *, max_tokens):
        calls["dflash"].append(
            (bundle, tuple(prompt_ids), width, runtime_context, max_tokens)
        )
        return {
            "tokens": oracle,
            "generated_tokens": max_tokens,
            "decode_tps": 61.0 + width,
            "requested_width": width,
            "effective_width": width,
            "fallback_ar": False,
        }

    def fake_context():
        calls["context"] += 1
        return context

    monkeypatch.setattr(runner, "run_target_oracle", fake_oracle)
    monkeypatch.setattr(runner, "run_mtp_control", fake_mtp)
    monkeypatch.setattr(runner, "run_dflash2_candidate", fake_dflash)
    monkeypatch.setattr(runner, "build_fixed_dflash_runtime_context", fake_context)
    bundle = object()

    receipt = runner.run_dflash2_depth_sweep(
        bundle=bundle,
        prompt_ids=(4, 5),
        widths=(1, 8),
        repetitions=1,
        max_tokens=4,
    )

    assert calls["oracle"] == [(bundle, (4, 5), 4)]
    assert calls["context"] == 1
    assert len(calls["mtp"]) == 4
    assert all(call[-1] == 4 for call in calls["mtp"])
    assert [(call[2], call[-1]) for call in calls["dflash"]] == [
        (1, 32), (1, 4), (8, 32), (8, 4)
    ]
    assert receipt["selection"] is not None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"widths": (), "repetitions": 1, "max_tokens": 4}, "width"),
        ({"widths": (1,), "repetitions": 0, "max_tokens": 4}, "repetitions"),
        ({"widths": (1,), "repetitions": 1, "max_tokens": 0}, "max_tokens"),
    ],
)
def test_sweep_rejects_invalid_workload_contract(kwargs, message):
    with pytest.raises(ValueError, match=message):
        runner.run_dflash2_depth_sweep(
            bundle=object(),
            prompt_ids=(1,),
            oracle_tokens=(),
            arm_runner=lambda *_args: None,
            **kwargs,
        )
