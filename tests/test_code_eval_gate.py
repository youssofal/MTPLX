"""Tests for the HumanEval / MBPP completion driver.

Nothing here touches a server. The HTTP seam (``_post_json``) is replaced, so
every test asserts on what the driver *would have sent* and how it behaves when
the transport misbehaves.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from mtplx.benchmarks import code_eval as ce


def _load_gate_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "code_eval_gate.py"
    spec = importlib.util.spec_from_file_location("code_eval_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _he_task() -> ce.CodeTask:
    return ce.CodeTask(
        task_id="HumanEval/0",
        suite="humaneval",
        prompt="def add(a, b):\n",
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        entry_point="add",
    )


def _mbpp_task() -> ce.CodeTask:
    return ce.CodeTask(
        task_id="MBPP/3",
        suite="mbpp",
        prompt="Write a function to add two numbers.\nassert add(1, 2) == 3\n",
        test="assert add(1, 2) == 3",
    )


def _args(**overrides) -> argparse.Namespace:
    values = {
        "model": "mtplx-test",
        "endpoint": "chat",
        "temperature": 0.0,
        "top_p": None,
        "max_tokens": 512,
        "seed": 42,
        # build_payload reads args.extra_body (the --extra-body passthrough); the
        # real parser always sets it, so the minimal namespace must too.
        "extra_body": {},
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _humaneval_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "HumanEval.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": f"HumanEval/{index}",
                    "prompt": "def add(a, b):\n",
                    "canonical_solution": "    return a + b\n",
                    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                    "entry_point": "add",
                }
            )
            for index in range(3)
        )
        + "\n"
    )
    return path


def _chat_response(text: str) -> dict:
    return {
        "id": "chatcmpl-x",
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"completion_tokens": 7},
    }


_GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
_WRONG = "```python\ndef add(a, b):\n    return a * b\n```"


def _cli(dataset: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--dataset-path",
        str(dataset),
        "--output-json",
        str(output),
        "--base-url",
        "http://127.0.0.1:9",
        "--model",
        "mtplx-test",
        "--workers",
        "2",
        "--score-workers",
        "2",
        *extra,
    ]


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------


def test_humaneval_chat_prompt_carries_the_signature_and_asks_for_a_fence() -> None:
    messages = gate.build_messages(_he_task())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "def add(a, b):" in messages[1]["content"]
    assert "fenced" in messages[0]["content"].lower()


def test_mbpp_chat_prompt_keeps_the_asserts() -> None:
    """The asserts pin the function name; dropping them makes MBPP unscoreable."""

    messages = gate.build_messages(_mbpp_task())
    assert "assert add(1, 2) == 3" in messages[1]["content"]


def test_chat_payload_is_deterministic_by_default() -> None:
    payload = gate.build_payload(
        _args(), _he_task(), sample=0, client_request_id="rid"
    )
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 512
    assert payload["seed"] == 42
    assert payload["stream"] is False
    assert "messages" in payload and "prompt" not in payload


def test_completions_endpoint_sends_the_raw_prompt_verbatim() -> None:
    """Base-model continuation: any wrapper text changes what is being measured."""

    payload = gate.build_payload(
        _args(endpoint="completions"), _he_task(), sample=0, client_request_id="rid"
    )
    assert payload["prompt"] == "def add(a, b):\n"
    assert "messages" not in payload
    assert "\ndef " in payload["stop"]


def test_completions_endpoint_omits_humaneval_stops_for_mbpp() -> None:
    payload = gate.build_payload(
        _args(endpoint="completions"), _mbpp_task(), sample=0, client_request_id="rid"
    )
    assert "stop" not in payload


def test_each_sample_gets_its_own_seed() -> None:
    """Without this, n>1 draws the same completion n times and pass@k is a lie."""

    seeds = {
        gate.build_payload(
            _args(temperature=0.8), _he_task(), sample=i, client_request_id="rid"
        )["seed"]
        for i in range(4)
    }
    assert len(seeds) == 4


def test_top_p_is_omitted_unless_asked_for() -> None:
    assert "top_p" not in gate.build_payload(
        _args(), _he_task(), sample=0, client_request_id="rid"
    )
    assert (
        gate.build_payload(
            _args(top_p=0.9), _he_task(), sample=0, client_request_id="rid"
        )["top_p"]
        == 0.9
    )


def test_endpoint_url() -> None:
    assert (
        gate.endpoint_url("http://h:1/", "chat") == "http://h:1/v1/chat/completions"
    )
    assert gate.endpoint_url("http://h:1", "completions") == "http://h:1/v1/completions"


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------


def test_extract_completion_text_handles_both_shapes() -> None:
    assert gate.extract_completion_text(_chat_response("hi"), "chat") == ("hi", "stop")
    raw = {"choices": [{"text": "  body", "finish_reason": "length"}]}
    assert gate.extract_completion_text(raw, "completions") == ("  body", "length")


def test_missing_choices_is_empty_not_an_exception() -> None:
    assert gate.extract_completion_text({}, "chat") == ("", None)
    assert gate.extract_completion_text({"choices": [{}]}, "chat") == ("", None)


# --------------------------------------------------------------------------
# retry behavior
# --------------------------------------------------------------------------


def _retry_call(monkeypatch, responses, *, retries=2):
    calls = {"n": 0}

    def fake_post(url, payload, **kwargs):
        index = calls["n"]
        calls["n"] += 1
        outcome = responses[min(index, len(responses) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(gate, "_post_json", fake_post)
    monkeypatch.setattr(gate.time, "sleep", lambda _s: None)
    return calls, fake_post


def test_transient_failure_is_retried_then_succeeds(monkeypatch) -> None:
    calls, _ = _retry_call(
        monkeypatch,
        [gate.GateHTTPError("busy", status=503), _chat_response("ok")],
    )
    response, attempts = gate.request_with_retries(
        "u", {}, timeout_s=1, api_key=None, client_request_id="r", retries=2,
        sleep=lambda _s: None,
    )
    assert attempts == 2 and calls["n"] == 2
    assert response["choices"][0]["message"]["content"] == "ok"


def test_retries_are_bounded(monkeypatch) -> None:
    calls, _ = _retry_call(monkeypatch, [gate.GateHTTPError("busy", status=503)])
    with pytest.raises(gate.GateHTTPError):
        gate.request_with_retries(
            "u", {}, timeout_s=1, api_key=None, client_request_id="r", retries=2,
            sleep=lambda _s: None,
        )
    assert calls["n"] == 3  # 1 attempt + 2 retries


def test_a_client_error_is_not_retried(monkeypatch) -> None:
    """A 400 is a misconfigured run; retrying it three times just wastes time."""

    calls, _ = _retry_call(monkeypatch, [gate.GateHTTPError("bad", status=400)])
    with pytest.raises(gate.GateHTTPError):
        gate.request_with_retries(
            "u", {}, timeout_s=1, api_key=None, client_request_id="r", retries=5,
            sleep=lambda _s: None,
        )
    assert calls["n"] == 1


def test_a_connection_failure_with_no_status_is_transient(monkeypatch) -> None:
    calls, _ = _retry_call(monkeypatch, [gate.GateHTTPError("socket died")])
    with pytest.raises(gate.GateHTTPError):
        gate.request_with_retries(
            "u", {}, timeout_s=1, api_key=None, client_request_id="r", retries=1,
            sleep=lambda _s: None,
        )
    assert calls["n"] == 2


def test_rate_limit_is_treated_as_transient() -> None:
    assert gate.GateHTTPError("x", status=429).transient
    assert not gate.GateHTTPError("x", status=404).transient


# --------------------------------------------------------------------------
# per-task failure must not kill the run
# --------------------------------------------------------------------------


def test_generate_one_records_a_failure_instead_of_raising(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "_post_json",
        lambda *a, **k: (_ for _ in ()).throw(gate.GateHTTPError("down", status=500)),
    )
    monkeypatch.setattr(gate.time, "sleep", lambda _s: None)
    config = {
        "base_url": "http://x",
        "api_key": None,
        "timeout_s": 1,
        "retries": 0,
        "request_args": vars(_args()),
    }
    row = gate.generate_one(config, _he_task(), 0)
    assert row["completion"] is None
    assert "down" in row["error"]


def test_a_failed_request_scores_as_its_own_status() -> None:
    """It must not be silently dropped, and it must not look like a wrong answer."""

    result = gate.score_one(
        _he_task(),
        {"error": "boom", "completion": None},
        allow_execution=True,
        timeout_s=5,
    )
    assert not result.passed and result.status == "request_error"


def test_one_dead_task_does_not_kill_the_run(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    output = tmp_path / "report.json"
    seen = {"n": 0}

    def fake_post(url, payload, **kwargs):
        seen["n"] += 1
        if payload["metadata"]["task_id"] == "HumanEval/1":
            raise gate.GateHTTPError("gateway gone", status=502)
        return _chat_response(_GOOD)

    monkeypatch.setattr(gate, "_post_json", fake_post)
    monkeypatch.setattr(gate.time, "sleep", lambda _s: None)

    code = gate.main(_cli(dataset, output, "--allow-code-execution", "--retries", "1"))

    report = json.loads(output.read_text())
    assert report["summary"]["tasks"] == 3
    assert report["summary"]["passed"] == 2
    assert report["request_errors"] == 1
    statuses = {row["task_id"]: row["status"] for row in report["rows"]}
    assert statuses["HumanEval/1"] == "request_error"
    assert statuses["HumanEval/0"] == "passed"
    assert code == 1  # a run that could not reach the model is not a green run


# --------------------------------------------------------------------------
# the execution opt-in
# --------------------------------------------------------------------------


def test_refuses_without_the_execution_flag(tmp_path, monkeypatch, capsys) -> None:
    dataset = _humaneval_dataset(tmp_path)
    output = tmp_path / "report.json"

    def explode(*a, **k):
        raise AssertionError("no request may be sent before the opt-in")

    monkeypatch.setattr(gate, "_post_json", explode)

    code = gate.main(_cli(dataset, output))

    assert code == 2
    assert "--allow-code-execution" in capsys.readouterr().err
    assert not output.exists()


def test_the_flag_is_what_reaches_run_candidate(tmp_path, monkeypatch) -> None:
    """The opt-in must actually be threaded through, not just checked at the door."""

    seen = {}

    def fake_run_candidate(task, completion, *, allow_execution=False, timeout_s=None):
        seen["allow_execution"] = allow_execution
        seen["timeout_s"] = timeout_s
        return ce.TaskResult(task.task_id, True, "passed", "", 0.1)

    monkeypatch.setattr(gate, "run_candidate", fake_run_candidate)
    gate.score_one(
        _he_task(),
        {"error": None, "completion": _GOOD},
        allow_execution=True,
        timeout_s=9.0,
    )
    assert seen == {"allow_execution": True, "timeout_s": 9.0}


# --------------------------------------------------------------------------
# report shape
# --------------------------------------------------------------------------


def test_report_shape_and_provenance(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(gate, "_post_json", lambda *a, **k: _chat_response(_GOOD))

    code = gate.main(_cli(dataset, output, "--allow-code-execution"))
    assert code == 0

    report = json.loads(output.read_text())
    assert report["schema"] == gate.SCHEMA

    summary = report["summary"]
    assert summary["tasks"] == 3 and summary["passed"] == 3
    assert summary["pass@1"] == pytest.approx(1.0)
    assert summary["by_status"] == {"passed": 3}

    rows = report["rows"]
    assert len(rows) == 3
    assert {"task_id", "status", "seconds"} <= set(rows[0])
    assert all(row["suite"] == "humaneval" for row in rows)

    prov = report["provenance"]
    assert prov["base_url"] == "http://127.0.0.1:9"
    assert prov["model"] == "mtplx-test"
    assert prov["suite"] == "humaneval"
    assert prov["tasks"] == 3
    assert len(prov["dataset_sha256"]) == 64
    assert prov["dataset_sha256"] == gate._dataset_sha256(dataset)
    assert prov["timestamp_utc"].endswith("+00:00")
    assert "mtplx_version" in prov

    params = report["params"]
    assert params["temperature"] == 0.0 and params["n"] == 1
    assert params["allow_code_execution"] is True
    assert params["endpoint"] == "chat"

    # n == 1 is not a pass@k run.
    assert "pass_at_k" not in report


def test_wrong_answers_are_scored_wrong(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(gate, "_post_json", lambda *a, **k: _chat_response(_WRONG))

    gate.main(_cli(dataset, output, "--allow-code-execution"))
    report = json.loads(output.read_text())
    assert report["summary"]["passed"] == 0
    assert report["summary"]["by_status"] == {"failed": 3}


def test_limit_truncates_the_task_list(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(gate, "_post_json", lambda *a, **k: _chat_response(_GOOD))

    gate.main(_cli(dataset, output, "--allow-code-execution", "--limit", "2"))
    report = json.loads(output.read_text())
    assert report["provenance"]["tasks"] == 2 and len(report["rows"]) == 2


def test_output_dir_is_accepted_like_the_sibling_gates(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    monkeypatch.setattr(gate, "_post_json", lambda *a, **k: _chat_response(_GOOD))

    gate.main(
        [
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(tmp_path / "out"),
            "--allow-code-execution",
        ]
    )
    assert (tmp_path / "out" / "code-eval-humaneval-chat.json").exists()


def test_min_pass_rate_gates_the_exit_code(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(gate, "_post_json", lambda *a, **k: _chat_response(_WRONG))

    code = gate.main(
        _cli(dataset, output, "--allow-code-execution", "--min-pass-rate", "0.5")
    )
    assert code == 1


# --------------------------------------------------------------------------
# pass@k
# --------------------------------------------------------------------------


def test_pass_at_k_is_averaged_per_task_not_pooled() -> None:
    """Two tasks, 2 samples each: one always right, one always wrong -> 0.5."""

    results = [
        ce.TaskResult("a", True, "passed"),
        ce.TaskResult("a", True, "passed"),
        ce.TaskResult("b", False, "failed"),
        ce.TaskResult("b", False, "failed"),
    ]
    report = gate.pass_at_k_report(results, n=2, ks=[1, 2])
    assert report["pass@1"] == pytest.approx(0.5)
    assert report["pass@2"] == pytest.approx(0.5)


def test_pass_at_k_rewards_one_lucky_sample() -> None:
    results = [
        ce.TaskResult("a", False, "failed"),
        ce.TaskResult("a", True, "passed"),
    ]
    report = gate.pass_at_k_report(results, n=2, ks=[1, 2])
    assert report["pass@1"] == pytest.approx(0.5)
    assert report["pass@2"] == pytest.approx(1.0)


def test_pass_at_k_skips_k_larger_than_n() -> None:
    results = [ce.TaskResult("a", True, "passed")]
    assert gate.pass_at_k_report(results, n=1, ks=[1, 5]) == {"pass@1": 1.0}


def test_n_greater_than_one_appears_in_the_report(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(gate, "_post_json", lambda *a, **k: _chat_response(_GOOD))

    gate.main(
        _cli(
            dataset,
            output,
            "--allow-code-execution",
            "--n",
            "2",
            "--temperature",
            "0.6",
        )
    )
    report = json.loads(output.read_text())
    assert len(report["rows"]) == 6  # 3 tasks x 2 samples
    assert report["pass_at_k"]["pass@1"] == pytest.approx(1.0)
    assert report["pass_at_k"]["pass@2"] == pytest.approx(1.0)
    assert report["params"]["n"] == 2


def test_sampling_n_at_temperature_zero_is_rejected(tmp_path) -> None:
    """n>1 at temp 0 returns the same completion n times; pass@k would be fake."""

    dataset = _humaneval_dataset(tmp_path)
    with pytest.raises(SystemExit):
        gate.main(
            _cli(dataset, tmp_path / "r.json", "--allow-code-execution", "--n", "3")
        )


# --------------------------------------------------------------------------
# --save-completions sidecar (enables the offline HumanEval+ rescore)
# --------------------------------------------------------------------------


def test_save_completions_sidecar_persists_task_id_and_solution(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    output = tmp_path / "report.json"
    sidecar = tmp_path / "samples.jsonl"

    monkeypatch.setattr(gate, "_post_json", lambda *a, **k: _chat_response(_GOOD))
    monkeypatch.setattr(gate.time, "sleep", lambda _s: None)

    code = gate.main(
        _cli(dataset, output, "--allow-code-execution", "--save-completions", str(sidecar))
    )
    assert code == 0
    assert sidecar.exists()
    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert {r["task_id"] for r in rows} == {"HumanEval/0", "HumanEval/1", "HumanEval/2"}
    for r in rows:
        assert r["completion"] == _GOOD          # the raw model text is preserved
        assert "return a + b" in r["solution"]   # and the extracted, runnable code
        assert r["sample"] == 0


def test_save_completions_is_opt_in_and_leaves_the_report_unchanged(tmp_path, monkeypatch) -> None:
    dataset = _humaneval_dataset(tmp_path)
    monkeypatch.setattr(gate, "_post_json", lambda *a, **k: _chat_response(_GOOD))
    monkeypatch.setattr(gate.time, "sleep", lambda _s: None)

    out_plain = tmp_path / "plain.json"
    assert gate.main(_cli(dataset, out_plain, "--allow-code-execution")) == 0
    # No --save-completions -> no sidecar is written.
    assert not (tmp_path / "samples.jsonl").exists()

    out_side = tmp_path / "withside.json"
    sidecar = tmp_path / "samples.jsonl"
    assert gate.main(
        _cli(dataset, out_side, "--allow-code-execution", "--save-completions", str(sidecar))
    ) == 0
    assert sidecar.exists()

    a = json.loads(out_plain.read_text())
    b = json.loads(out_side.read_text())
    # The report is identical run-for-run once wall-clock fields are dropped:
    # the sidecar is a side output, not a change to the scored report.
    for rep in (a, b):
        rep["provenance"].pop("timestamp_utc", None)
        rep["provenance"].pop("wall_s", None)
        for row in rep["rows"]:
            row.pop("seconds", None)
            row.pop("request_seconds", None)
    assert a["summary"] == b["summary"]
    assert a["rows"] == b["rows"]
