from __future__ import annotations

from types import SimpleNamespace

from mtplx.benchmarks.runners import mtp_adaptive, mtp_depth_grid
from mtplx.benchmarks.schema import PromptCase


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=object(),
        contract=SimpleNamespace(
            base_hidden_variant="pre_norm",
            hidden_variant="post_norm",
            concat_order="base_then_mtp",
        ),
        mtp_adapter_metadata=None,
        mtp_adapter_merge_report=None,
    )


def _output(sync_calls: int, token_reuses: int) -> SimpleNamespace:
    stats = SimpleNamespace(
        generated_tokens=2,
        elapsed_s=1.0,
        tok_s=2.0,
        accepted_drafts=1,
        rejected_drafts=0,
        drafted_tokens=1,
        accepted_by_depth=[1, 0],
        drafted_by_depth=[1, 0],
        verify_time_s=0.2,
        draft_time_s=0.1,
        target_forward_time_s=0.2,
        snapshot_time_s=0.0,
        accept_time_s=0.0,
        rollback_time_s=0.0,
        repair_time_s=0.0,
        commit_time_s=0.0,
        capture_commit_time_s=0.0,
        bonus_time_s=0.0,
        draft_core={
            "requested": "stock",
            "greedy_confidence_sync_calls": sync_calls,
            "greedy_confidence_token_reuses": token_reuses,
        },
        bonus_tokens=0,
        correction_tokens=0,
        verify_calls=1,
        peak_memory_bytes=1024,
        graphbank={},
        events=[{}],
    )
    return SimpleNamespace(tokens=[1, 2], text="safe output", stats=stats)


def _prompt_cases() -> list[PromptCase]:
    return [
        PromptCase(id="one", category="general", prompt="one", max_tokens=2),
        PromptCase(id="two", category="general", prompt="two", max_tokens=2),
    ]


def test_adaptive_runner_exposes_greedy_confidence_counters(
    monkeypatch,
    tmp_path,
) -> None:
    outputs = iter([_output(3, 2), _output(4, 4)])
    monkeypatch.setattr(mtp_adaptive, "load", lambda *_args, **_kwargs: _runtime())
    monkeypatch.setattr(
        mtp_adaptive,
        "load_prompt_suite",
        lambda *_args, **_kwargs: _prompt_cases(),
    )
    monkeypatch.setattr(
        mtp_adaptive,
        "encode_prompt_case",
        lambda *_args, **_kwargs: [1],
    )
    monkeypatch.setattr(
        mtp_adaptive,
        "generate_mtpk",
        lambda *_args, **_kwargs: next(outputs),
    )

    result = mtp_adaptive.run_mtp_adaptive(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        max_depth=2,
        policy_kind="expected_value",
        temperature=0.0,
        draft_temperature=0.0,
    )

    assert result["rows"][0]["draft_core"]["greedy_confidence_sync_calls"] == 3
    assert result["summary"]["draft_core"]["greedy_confidence_sync_calls"] == 7
    assert result["summary"]["draft_core"]["greedy_confidence_token_reuses"] == 6


def test_margin_grid_exposes_greedy_confidence_counters(
    monkeypatch,
    tmp_path,
) -> None:
    outputs = iter([_output(3, 2), _output(4, 4)])
    monkeypatch.setattr(mtp_depth_grid, "load", lambda *_args, **_kwargs: _runtime())
    monkeypatch.setattr(
        mtp_depth_grid,
        "load_prompt_suite",
        lambda *_args, **_kwargs: _prompt_cases(),
    )
    monkeypatch.setattr(
        mtp_depth_grid,
        "encode_prompt_case",
        lambda *_args, **_kwargs: [1],
    )
    monkeypatch.setattr(
        mtp_depth_grid,
        "generate_mtpk",
        lambda *_args, **_kwargs: next(outputs),
    )

    result = mtp_depth_grid.run_mtp_depth_policy_grid(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depth=2,
        thresholds=[0.5],
        min_depths=[0],
        temperature=0.0,
        draft_temperature=0.0,
    )

    cell = result["grid"][0]
    assert cell["rows"][0]["draft_core"]["greedy_confidence_sync_calls"] == 3
    assert cell["summary"]["draft_core"]["greedy_confidence_sync_calls"] == 7
    assert cell["summary"]["draft_core"]["greedy_confidence_token_reuses"] == 6
