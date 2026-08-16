from __future__ import annotations

from types import SimpleNamespace

from mtplx.benchmarks.runners import mtp_depth_sweep
from mtplx.benchmarks.schema import PromptCase
from mtplx.generation import GenerationOutput, GenerationStats


def _generation_output(
    *,
    events: list[dict],
    verify_calls: int,
    mode: str = "mtp",
) -> GenerationOutput:
    return GenerationOutput(
        tokens=[1, 2],
        text="ok",
        finish_reason="length",
        stats=GenerationStats(
            mode=mode,
            generated_tokens=2,
            elapsed_s=1.0,
            tok_s=2.0,
            decode_elapsed_s=1.0,
            decode_tok_s=2.0,
            end_to_end_tok_s=2.0,
            prompt_tps=50.0,
            prompt_target_prefill_tok_s=48.0,
            peak_memory_bytes=1_500,
            accepted_drafts=6,
            drafted_tokens=8,
            accepted_by_depth=[3, 3],
            drafted_by_depth=[4, 4],
            accept_probability_sum_by_depth=[3.0, 3.0],
            mean_accept_probability_by_depth=[0.75, 0.75],
            verify_calls=verify_calls,
            events=events,
        ),
    )


def test_depth_sweep_reports_prefill_and_memory_growth_for_ar_and_each_depth(
    monkeypatch, tmp_path
) -> None:
    fake_runtime = SimpleNamespace(
        tokenizer=object(),
        contract=SimpleNamespace(
            base_hidden_variant="pre_norm",
            hidden_variant="pre_norm",
            concat_order="base_then_mtp",
            mtp_quant_bits=None,
            mtp_quant_group_size=64,
            mtp_quant_mode="affine",
            mtp_quant_policy=None,
        ),
        mtp_adapter_metadata=None,
        mtp_adapter_merge_report=None,
    )
    active_memory = iter([1_000, 1_125, 1_250])

    monkeypatch.setattr(mtp_depth_sweep, "load", lambda *_args, **_kwargs: fake_runtime)
    monkeypatch.setattr(
        mtp_depth_sweep,
        "load_prompt_suite",
        lambda *_args: [PromptCase(id="one", category="general", prompt="one")],
    )
    monkeypatch.setattr(
        mtp_depth_sweep, "encode_prompt_case", lambda *_args, **_kwargs: [1, 2]
    )
    monkeypatch.setattr(
        mtp_depth_sweep,
        "generate_ar",
        lambda *_args, **_kwargs: _generation_output(
            events=[], verify_calls=0, mode="ar"
        ),
    )
    monkeypatch.setattr(
        mtp_depth_sweep,
        "generate_mtpk",
        lambda *_args, **_kwargs: _generation_output(events=[{}], verify_calls=1),
    )
    monkeypatch.setattr(
        mtp_depth_sweep, "_active_memory_bytes", lambda: next(active_memory)
    )
    monkeypatch.setattr(
        mtp_depth_sweep, "validate_benchmark_output", lambda *_args, **_kwargs: []
    )

    result = mtp_depth_sweep.run_mtp_depth_sweep(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depths=[1],
        compare_ar=True,
        temperature=0.0,
    )

    assert result["load_active_memory_bytes"] == 1_000
    ar_row = result["ar_rows"][0]
    assert ar_row["prompt_tokens"] == 2
    assert ar_row["prompt_target_prefill_tok_s"] == 48.0
    assert ar_row["prompt_tps"] == 50.0
    assert ar_row["active_memory_bytes"] == 1_125
    assert ar_row["active_memory_growth_bytes"] == 125
    assert ar_row["peak_memory_bytes"] == 1_500
    depth = result["depths"][0]
    depth_row = depth["rows"][0]
    assert depth_row["prompt_tokens"] == 2
    assert depth_row["prompt_target_prefill_tok_s"] == 48.0
    assert depth_row["prompt_tps"] == 50.0
    assert depth_row["active_memory_bytes"] == 1_250
    assert depth_row["active_memory_growth_bytes"] == 250
    assert depth_row["peak_memory_bytes"] == 1_500
    assert depth["summary"]["active_memory_bytes"] == 1_250
    assert depth["summary"]["active_memory_growth_bytes"] == 250


def test_depth_sweep_uses_verify_calls_only_when_events_are_empty(
    monkeypatch, tmp_path
) -> None:
    fake_runtime = SimpleNamespace(
        tokenizer=object(),
        contract=SimpleNamespace(
            base_hidden_variant="pre_norm",
            hidden_variant="pre_norm",
            concat_order="base_then_mtp",
            mtp_quant_bits=None,
            mtp_quant_group_size=64,
            mtp_quant_mode="affine",
            mtp_quant_policy=None,
        ),
        mtp_adapter_metadata=None,
        mtp_adapter_merge_report=None,
    )
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

    monkeypatch.setattr(mtp_depth_sweep, "load", lambda *_args, **_kwargs: fake_runtime)
    monkeypatch.setattr(mtp_depth_sweep, "load_prompt_suite", lambda *_args: cases)
    monkeypatch.setattr(
        mtp_depth_sweep, "encode_prompt_case", lambda *_args, **_kwargs: [1, 2]
    )
    monkeypatch.setattr(
        mtp_depth_sweep, "generate_mtpk", lambda *_args, **_kwargs: next(outputs)
    )
    monkeypatch.setattr(
        mtp_depth_sweep, "validate_benchmark_output", lambda *_args, **_kwargs: []
    )

    result = mtp_depth_sweep.run_mtp_depth_sweep(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depths=[2],
    )

    native, generic = result["depths"][0]["rows"]
    assert native["mean_accepted_drafts_per_cycle"] == 1.5
    assert generic["mean_accepted_drafts_per_cycle"] == 3.0


def test_depth_sweep_uses_packaged_draft_lm_head_helper(monkeypatch, tmp_path) -> None:
    calls = []
    fake_runtime = SimpleNamespace(
        model=object(),
        tokenizer=object(),
        contract=SimpleNamespace(
            base_hidden_variant="pre_norm",
            hidden_variant="pre_norm",
            concat_order="base_then_mtp",
            mtp_quant_bits=None,
            mtp_quant_group_size=64,
            mtp_quant_mode="affine",
            mtp_quant_policy=None,
        ),
        mtp_adapter_metadata=None,
        mtp_adapter_merge_report=None,
    )

    monkeypatch.setattr(mtp_depth_sweep, "load", lambda *_args, **_kwargs: fake_runtime)
    monkeypatch.setattr(
        mtp_depth_sweep, "load_prompt_suite", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "mtplx.draft_lm_head._install_draft_lm_head",
        lambda runtime, **kwargs: (
            calls.append((runtime, kwargs)) or {"installed": True}
        ),
    )

    result = mtp_depth_sweep.run_mtp_depth_sweep(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depths=[1],
        draft_lm_head_bits=4,
        draft_lm_head_group_size=64,
        draft_lm_head_mode="affine",
    )

    assert result["draft_lm_head"] == {"installed": True}
    assert result["mtp_adapter_merged"] is False
    assert result["mtp_adapter_merge_report"] is None
    assert calls == [
        (
            fake_runtime,
            {"bits": 4, "group_size": 64, "mode": "affine"},
        )
    ]


def test_depth_sweep_passes_merge_mtp_adapter_to_runtime(monkeypatch, tmp_path) -> None:
    load_kwargs = []
    fake_runtime = SimpleNamespace(
        model=object(),
        tokenizer=object(),
        contract=SimpleNamespace(
            base_hidden_variant="pre_norm",
            hidden_variant="pre_norm",
            concat_order="base_then_mtp",
            mtp_quant_bits=None,
            mtp_quant_group_size=64,
            mtp_quant_mode="affine",
            mtp_quant_policy=None,
        ),
        mtp_adapter_metadata={"kind": "c4_mtp_lora_adapter"},
        mtp_adapter_merge_report={"merged": 1, "targets": [{"target": "fc"}]},
    )

    def fake_load(*_args, **kwargs):
        load_kwargs.append(kwargs)
        return fake_runtime

    monkeypatch.setattr(mtp_depth_sweep, "load", fake_load)
    monkeypatch.setattr(
        mtp_depth_sweep, "load_prompt_suite", lambda *_args, **_kwargs: []
    )

    result = mtp_depth_sweep.run_mtp_depth_sweep(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depths=[1],
        mtp_adapter_path=tmp_path / "adapter.npz",
        merge_mtp_adapter=True,
    )

    assert load_kwargs[0]["mtp_adapter"] == tmp_path / "adapter.npz"
    assert load_kwargs[0]["merge_mtp_adapter"] is True
    assert result["mtp_adapter_kind"] == "c4_mtp_lora_adapter"
    assert result["mtp_adapter_merged"] is True
    assert result["mtp_adapter_merge_report"] == {
        "merged": 1,
        "targets": [{"target": "fc"}],
    }


def test_depth_sweep_selects_construction_bound_0731_stack(
    monkeypatch, tmp_path
) -> None:
    load_kwargs = []
    fake_runtime = SimpleNamespace(
        tokenizer=object(),
        contract=SimpleNamespace(
            base_hidden_variant="pre_norm",
            hidden_variant="pre_norm",
            concat_order="base_then_mtp",
            mtp_quant_bits=None,
            mtp_quant_group_size=64,
            mtp_quant_mode="affine",
            mtp_quant_policy=None,
        ),
        mtp_adapter_metadata=None,
        mtp_adapter_merge_report=None,
    )

    def fake_load(*_args, **kwargs):
        load_kwargs.append(kwargs)
        return fake_runtime

    monkeypatch.setattr(mtp_depth_sweep, "load", fake_load)
    monkeypatch.setattr(mtp_depth_sweep, "_active_memory_bytes", lambda: 0)
    monkeypatch.setattr(
        mtp_depth_sweep, "load_prompt_suite", lambda *_args, **_kwargs: []
    )

    mtp_depth_sweep.run_mtp_depth_sweep(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depths=[1],
        deepseek_v4_0731_k2=True,
    )

    assert load_kwargs[0]["deepseek_v4_0731_k2"] is True


def test_depth_sweep_omits_hidden_variants_for_native_block_backend(
    monkeypatch, tmp_path
) -> None:
    generate_kwargs = []
    fake_runtime = SimpleNamespace(
        tokenizer=object(),
        block_speculative_backend=object(),
        contract=SimpleNamespace(
            base_hidden_variant="pre_norm",
            hidden_variant="pre_norm",
            concat_order="base_then_mtp",
            mtp_quant_bits=None,
            mtp_quant_group_size=64,
            mtp_quant_mode="affine",
            mtp_quant_policy=None,
        ),
        mtp_adapter_metadata=None,
        mtp_adapter_merge_report=None,
    )

    monkeypatch.setattr(mtp_depth_sweep, "load", lambda *_args, **_kwargs: fake_runtime)
    monkeypatch.setattr(mtp_depth_sweep, "_active_memory_bytes", lambda: 0)
    monkeypatch.setattr(
        mtp_depth_sweep,
        "load_prompt_suite",
        lambda *_args: [PromptCase(id="one", category="general", prompt="one")],
    )
    monkeypatch.setattr(
        mtp_depth_sweep, "encode_prompt_case", lambda *_args, **_kwargs: [1, 2]
    )
    monkeypatch.setattr(
        mtp_depth_sweep,
        "generate_mtpk",
        lambda *_args, **kwargs: (
            generate_kwargs.append(kwargs)
            or _generation_output(events=[], verify_calls=1)
        ),
    )
    monkeypatch.setattr(
        mtp_depth_sweep, "validate_benchmark_output", lambda *_args, **_kwargs: []
    )

    mtp_depth_sweep.run_mtp_depth_sweep(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depths=[1],
        deepseek_v4_0731_k2=True,
    )

    assert generate_kwargs[0]["base_hidden_variant"] is None
    assert generate_kwargs[0]["mtp_hidden_variant"] is None
    assert generate_kwargs[0]["mtp_history_policy"] == "cycle"
    assert generate_kwargs[0]["verify_strategy"] == "batched"
