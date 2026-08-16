from __future__ import annotations

from types import SimpleNamespace

from mtplx.benchmarks.runners import mtp_depth_grid
from mtplx.benchmarks.schema import PromptCase
from mtplx.generation import GenerationOutput, GenerationStats


def _generation_output(*, events: list[dict], verify_calls: int) -> GenerationOutput:
    return GenerationOutput(
        tokens=[1, 2],
        text="ok",
        stats=GenerationStats(
            mode="mtp",
            generated_tokens=2,
            elapsed_s=1.0,
            tok_s=2.0,
            accepted_drafts=6,
            drafted_tokens=8,
            accepted_by_depth=[3, 3],
            drafted_by_depth=[4, 4],
            verify_calls=verify_calls,
            events=events,
        ),
    )


def test_depth_grid_uses_verify_calls_only_when_events_are_empty(
    monkeypatch, tmp_path
) -> None:
    fake_runtime = SimpleNamespace(tokenizer=object())
    cases = [
        PromptCase(id="native", category="general", prompt="native"),
        PromptCase(id="generic", category="general", prompt="generic"),
    ]
    outputs = iter(
        [
            _generation_output(events=[], verify_calls=4),
            _generation_output(events=[{}, {}], verify_calls=9),
        ]
    )

    monkeypatch.setattr(mtp_depth_grid, "load", lambda *_args, **_kwargs: fake_runtime)
    monkeypatch.setattr(mtp_depth_grid, "load_prompt_suite", lambda *_args: cases)
    monkeypatch.setattr(
        mtp_depth_grid, "encode_prompt_case", lambda *_args, **_kwargs: [1, 2]
    )
    monkeypatch.setattr(
        mtp_depth_grid, "generate_mtpk", lambda *_args, **_kwargs: next(outputs)
    )

    result = mtp_depth_grid.run_mtp_depth_policy_grid(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depth=2,
        thresholds=[None],
        min_depths=[0],
    )

    grid = result["grid"][0]
    native, generic = grid["rows"]
    assert native["cycles"] == 4
    assert native["accepted_drafts_per_cycle"] == 1.5
    assert generic["cycles"] == 2
    assert generic["accepted_drafts_per_cycle"] == 3.0
    assert grid["summary"]["cycles"] == 6
    assert grid["summary"]["accepted_drafts_per_cycle"] == 2.0
