from __future__ import annotations

import asyncio
import time

import pytest

from mtplx.deterministic_replay import (
    CounterfactualReplay,
    Evaluation,
    RegressionPolicy,
    ReplayCase,
    ReplayValidationError,
    public_request_fingerprint,
)


def score_eval(case, output, baseline):
    score = float(output["score"])
    baseline_score = None if baseline is None else float(baseline["score"])
    return Evaluation(
        name="score",
        score=score,
        passed=score >= 0.8,
        baseline_score=baseline_score,
        details={"case": case.case_id, "authorization": "secret"},
    )


def test_public_fingerprint_redacts_credentials_but_private_dedupe_does_not():
    left = {"prompt": "hello", "api_key": "one"}
    right = {"prompt": "hello", "api_key": "two"}
    assert public_request_fingerprint(left) == public_request_fingerprint(right)

    calls = []

    def candidate(request):
        calls.append(request["api_key"])
        return {"score": 1.0}

    report = CounterfactualReplay(max_concurrency=2).run(
        [ReplayCase("a", left), ReplayCase("b", right)],
        candidate=candidate,
        evaluators={"score": score_eval},
    )
    assert sorted(calls) == ["one", "two"]
    assert [result.case_id for result in report.results] == ["a", "b"]


def test_candidate_execution_is_deduplicated_but_evaluation_is_case_specific():
    calls = 0

    def candidate(request):
        nonlocal calls
        calls += 1
        return {"score": request["score"]}

    report = CounterfactualReplay(max_concurrency=2).run(
        [
            ReplayCase("a", {"score": 1.0}, baseline_output={"score": 0.7}),
            ReplayCase("b", {"score": 1.0}, baseline_output={"score": 0.9}),
        ],
        candidate=candidate,
        evaluators={"score": score_eval},
    )
    assert calls == 1
    assert report.results[0].candidate_reused is False
    assert report.results[1].candidate_reused is True
    assert [row.evaluations[0].baseline_score for row in report.results] == [0.7, 0.9]


def test_candidate_and_evaluator_inputs_are_mutation_isolated():
    request = {"nested": {"value": 1}}
    baseline = {"score": 0.5}

    def candidate(candidate_request):
        candidate_request["nested"]["value"] = 99
        return {"score": 1.0}

    def evaluator(case, output, baseline_output):
        case.metadata["seen"] = True
        output["score"] = 0.0
        baseline_output["score"] = 0.0
        return Evaluation("score", score=1.0, passed=True, baseline_score=0.5)

    case = ReplayCase("a", request, baseline_output=baseline, metadata={})
    report = CounterfactualReplay().run(
        [case], candidate=candidate, evaluators={"score": evaluator}
    )
    assert request == {"nested": {"value": 1}}
    assert baseline == {"score": 0.5}
    assert case.metadata == {}
    assert report.results[0].output == {"score": 1.0}


def test_async_candidate_and_evaluator_are_supported():
    async def candidate(request):
        await asyncio.sleep(0)
        return {"score": request["score"]}

    async def evaluator(case, output, baseline):
        await asyncio.sleep(0)
        return Evaluation("score", score=output["score"], passed=True)

    report = CounterfactualReplay().run(
        [ReplayCase("a", {"score": 1.0})],
        candidate=candidate,
        evaluators={"score": evaluator},
    )
    assert report.results[0].ok


def test_candidate_errors_are_isolated_and_report_serialization_omits_content():
    def candidate(request):
        if request["fail"]:
            raise RuntimeError("authorization=secret")
        return {"score": 1.0, "secret": "output-secret"}

    report = CounterfactualReplay().run(
        [ReplayCase("bad", {"fail": True}), ReplayCase("ok", {"fail": False})],
        candidate=candidate,
        evaluators={"score": score_eval},
    )
    payload = report.to_dict()
    assert report.results[0].errors[0].error_type == "RuntimeError"
    assert report.results[1].ok
    rendered = str(payload)
    assert "authorization=secret" not in rendered
    assert "output-secret" not in rendered
    assert "details" not in rendered


def test_explicit_report_content_is_redacted():
    report = CounterfactualReplay().run(
        [ReplayCase("a", {"api_key": "request-secret"}, metadata={"password": "p"})],
        candidate=lambda _request: {"score": 1.0, "access_token": "out"},
        evaluators={"score": score_eval},
    )
    payload = report.to_dict(
        include_outputs=True,
        include_metadata=True,
        include_evaluation_details=True,
    )
    rendered = str(payload)
    assert "request-secret" not in rendered
    assert "'out'" not in rendered
    assert "authorization" in rendered
    assert "<redacted>" in rendered


def test_regression_policy_promotes_clean_report():
    report = CounterfactualReplay().run(
        [ReplayCase("a", {"score": 1.0}, baseline_output={"score": 0.9})],
        candidate=lambda request: request,
        evaluators={"score": score_eval},
        candidate_name="candidate-a",
    )
    decision = RegressionPolicy().evaluate(report)
    assert decision.promote is True
    assert decision.violations == ()
    assert len(decision.decision_id) == 64


def test_regression_policy_reports_all_failed_gates():
    report = CounterfactualReplay().run(
        [ReplayCase("a", {"score": 0.2}, baseline_output={"score": 0.9})],
        candidate=lambda request: request,
        evaluators={"score": score_eval},
    )
    decision = RegressionPolicy(
        minimum_cases=2,
        minimum_evaluations=2,
        minimum_pass_rate=1.0,
        maximum_regression_rate=0.0,
        maximum_mean_score_drop=0.0,
    ).evaluate(report)
    assert decision.promote is False
    assert set(decision.violations) == {
        "minimum_case_count",
        "minimum_evaluation_count",
        "minimum_pass_rate",
        "maximum_regression_rate",
        "maximum_mean_score_drop",
    }


def test_score_tolerance_prevents_false_regression():
    report = CounterfactualReplay().run(
        [ReplayCase("a", {"score": 0.8999}, baseline_output={"score": 0.9})],
        candidate=lambda request: request,
        evaluators={"score": score_eval},
    )
    decision = RegressionPolicy(
        minimum_pass_rate=0.0,
        score_tolerance=0.001,
        maximum_mean_score_drop=1.0,
    ).evaluate(report)
    assert decision.promote is True


def test_empty_and_duplicate_suites_fail_closed():
    replay = CounterfactualReplay()
    with pytest.raises(ReplayValidationError, match="at least one case"):
        replay.run([], candidate=lambda value: value, evaluators={"x": score_eval})
    with pytest.raises(ReplayValidationError, match="unique"):
        replay.run(
            [ReplayCase("a", {}), ReplayCase("a", {})],
            candidate=lambda value: value,
            evaluators={"x": score_eval},
        )


def test_unsupported_request_values_fail_closed():
    with pytest.raises(ReplayValidationError, match="unsupported replay value"):
        CounterfactualReplay().run(
            [ReplayCase("a", {"bad": object()})],
            candidate=lambda value: value,
            evaluators={"score": score_eval},
        )


def test_invalid_evaluator_return_is_isolated():
    report = CounterfactualReplay().run(
        [ReplayCase("a", {})],
        candidate=lambda _request: {"score": 1.0},
        evaluators={"broken": lambda *_args: 123},
    )
    assert report.results[0].errors[0].phase == "evaluator"
    assert report.results[0].errors[0].error_type == "ReplayValidationError"


def test_candidate_timeout_bounds_caller_latency():
    def candidate(_request):
        time.sleep(0.30)
        return {"score": 1.0}

    started = time.monotonic()
    report = CounterfactualReplay(candidate_timeout_s=0.03).run(
        [ReplayCase("slow", {})],
        candidate=candidate,
        evaluators={"score": score_eval},
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.20
    assert report.results[0].errors[0].phase == "candidate"
    assert report.results[0].errors[0].error_type == "TimeoutError"


def test_evaluator_timeout_bounds_caller_latency():
    def evaluator(_case, _output, _baseline):
        time.sleep(0.30)
        return Evaluation("slow", score=1.0, passed=True)

    started = time.monotonic()
    report = CounterfactualReplay(evaluator_timeout_s=0.03).run(
        [ReplayCase("slow", {})],
        candidate=lambda _request: {"score": 1.0},
        evaluators={"slow": evaluator},
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.20
    assert report.results[0].errors[0].phase == "evaluator"
    assert report.results[0].errors[0].error_type == "TimeoutError"
