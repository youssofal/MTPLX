from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mtplx.kpi.runtime_kpis import exact_paged_attention_env, run_exactness_smoke, summarize_decode_trace


def test_summarize_decode_trace_computes_window_ratios(tmp_path):
    trace = tmp_path / "decode.jsonl"
    rows = [
        {
            "event": "decode_trace_bucket",
            "generated_tokens_delta": 80,
            "generated_tokens_total": 80,
            "elapsed_s": 1.0,
            "verify_ms_per_call_delta": 40.0,
        },
        {
            "event": "decode_trace_bucket",
            "generated_tokens_delta": 80,
            "generated_tokens_total": 160,
            "elapsed_s": 2.0,
            "verify_ms_per_call_delta": 60.0,
            "mlx_memory": {"cache_memory_bytes": 1073741824},
        },
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarize_decode_trace(trace)

    assert summary["available"] is True
    assert summary["first64_tok_s"] == 80.0
    assert summary["last64_tok_s"] == 40.0
    assert summary["last64_over_first64"] == 0.5
    assert summary["late_verify_ms"] == 60.0
    assert summary["cache_gib_last"] == 1.0


def test_exact_paged_attention_env_defaults_to_vector_impl():
    env = exact_paged_attention_env()

    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] == "mlx_vector_paged"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"] == "2048"


def test_run_exactness_smoke_uses_vector_paged_profile(monkeypatch, tmp_path):
    from mtplx.kpi import runtime_kpis

    monkeypatch.setattr(runtime_kpis, "repo_root", lambda: Path(__file__).resolve().parents[1])
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "smoke.json"

    result = run_exactness_smoke("models/example", output=out)

    assert result["passed"] is True
    assert "--attention-impl" in seen["cmd"]
    assert "mlx_vector_paged" in seen["cmd"]
    assert "--partition-threshold" in seen["cmd"]
    assert "2048" in seen["cmd"]


def test_decode_trace_tolerates_lane_specific_counter_sets(tmp_path, monkeypatch):
    """AR totals omit MTP-only counters; emitting twice (the second emit
    diffs against the lane's own totals) must not KeyError and must report
    zero deltas for absent counters."""

    import json as _json

    monkeypatch.setenv("MTPLX_DECODE_TRACE_JSONL", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("MTPLX_DECODE_TRACE_INTERVAL_S", "0.1")
    from mtplx.generation import SamplerConfig, _DecodeTrace

    trace = _DecodeTrace(
        prompt_tokens=8,
        max_tokens=4,
        speculative_depth=0,
        sampler=SamplerConfig(),
        verify_strategy="ar",
        verify_core="stock",
        mtp_history_policy="none",
        mtp_cache_policy="none",
        trace_label=None,
        trace_metadata={"generation_mode": "ar"},
    )
    # The scalar/list key set the AR lane's trace_totals() actually
    # provides — everything else (MTP-only counters like
    # target_distribution_materialized_rows) is intentionally absent.
    ar_scalar_keys = (
        "accepted_drafts rejected_drafts drafted_tokens evaluated_drafts "
        "fully_accepted_verify_calls verify_calls correction_tokens "
        "bonus_tokens verify_time_s verify_forward_time_s verify_eval_time_s "
        "verify_logits_eval_time_s verify_hidden_eval_time_s "
        "verify_joint_eval_time_s verify_target_distribution_time_s "
        "verify_eval_unattributed_time_s draft_time_s accept_time_s "
        "repair_time_s commit_time_s capture_commit_time_s snapshot_time_s "
        "bonus_time_s verify_output_nbytes draft_output_nbytes "
        "mtp_history_append_nbytes clear_cache_events clear_cache_time_s "
        "trunk_cache_materialize_events trunk_cache_materialize_time_s "
        "dirty_detach_events dirty_detach_time_s dirty_detach_arrays "
        "dirty_detach_bytes live_output_detach_events "
        "live_output_detach_time_s live_output_detach_arrays "
        "live_output_detach_bytes state_rebase_events state_rebase_time_s "
        "state_root_eval_events state_root_eval_time_s "
        "state_root_eval_arrays trace_accounting_time_s"
    ).split()
    ar_totals = {key: 0 for key in ar_scalar_keys}
    ar_totals.update(
        generated_tokens=2,
        accepted_by_depth=[],
        drafted_by_depth=[],
        evaluated_by_depth=[],
        accept_probability_sum_by_depth=[],
    )
    for generated in (2, 4):
        ar_totals = dict(ar_totals, generated_tokens=generated)
        trace.maybe_emit(
            force=True,
            final=generated == 4,
            totals=ar_totals,
            cache=None,
            mtp_cache=None,
            mtp_history_materialize_every=0,
            mtp_history_materialize_events=0,
        )
    rows = [
        _json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[1]["generated_tokens_delta"] == 2
    assert rows[1]["target_distribution_materialized_rows_delta"] == 0


def test_prompt_suite_path_is_absolute_and_exists_from_any_directory(monkeypatch, tmp_path):
    # The suites used to be cwd-relative strings ("mtplx/benchmarks/prompts/
    # default.jsonl"), so `mtplx bench` crashed with FileNotFoundError from
    # every directory except a checkout root. They ship inside the package
    # and must resolve against it.
    from pathlib import Path

    from mtplx.kpi.runtime_kpis import PROMPT_SUITES, prompt_suite_path

    monkeypatch.chdir(tmp_path)

    for suite in (None, *PROMPT_SUITES):
        resolved = Path(prompt_suite_path(suite))
        assert resolved.is_absolute(), suite
        assert resolved.is_file(), suite
        assert resolved.parent.name == "prompts"
        assert resolved.parent.parent.name == "benchmarks"


def test_prompt_suite_path_accepts_a_file_and_names_the_suites_on_a_bad_name(tmp_path):
    from pathlib import Path

    import pytest

    from mtplx.kpi.runtime_kpis import prompt_suite_path

    custom = tmp_path / "mine.jsonl"
    custom.write_text('{"prompt": "hi"}\n', encoding="utf-8")

    assert Path(prompt_suite_path(str(custom))) == custom

    with pytest.raises(SystemExit) as excinfo:
        prompt_suite_path("no-such-suite")
    message = str(excinfo.value.code)
    assert "unknown prompt suite 'no-such-suite'" in message
    assert "flappy" in message and ".jsonl" in message
