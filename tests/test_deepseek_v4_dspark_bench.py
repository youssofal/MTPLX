from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import pytest

from dflash_mlx.engine.events import SummaryEvent

import scripts.deepseek_v4_dspark_k5_bench as bench


def _generation_output(*, events, new_prefill_tokens: int, tokens=(7,)):
    stats = {
        "mode": "ar",
        "events": events,
        "new_prefill_tokens": new_prefill_tokens,
        "prompt_eval_time_s": 0.5,
        "decode_elapsed_s": 0.25,
        "elapsed_s": 0.75,
        "decode_tok_s": 4.0,
        "peak_memory_bytes": 123,
        "accepted_drafts": 0,
        "drafted_tokens": 0,
    }
    return SimpleNamespace(
        tokens=list(tokens),
        stats=SimpleNamespace(to_dict=lambda: stats),
    )


def test_arm_payload_uses_stable_prefill_count_with_token_or_empty_events() -> None:
    for events in ([{"type": "token", "token_id": 7}], []):
        receipt = bench._arm_payload(
            _generation_output(events=events, new_prefill_tokens=3, tokens=()),
            total_prompt_tokens=5,
        )

        assert receipt["prompt_tokens"] == 5
        assert receipt["new_prefill_tokens"] == 3
        assert receipt["prefill_tok_s"] == 6.0


def test_dspark_payload_counts_its_summary_prompt_as_new_prefill(monkeypatch) -> None:
    output = _generation_output(
        events=[{"type": "summary", "prompt_token_count": 3}],
        new_prefill_tokens=0,
    )
    def fake_generate(*_args, **kwargs):
        kwargs["token_callback"]([7])
        return output

    monkeypatch.setattr(
        "mtplx.deepseek_v4_dflash2.generate_deepseek_v4_dflash2",
        fake_generate,
    )
    clock = iter((10.0, 12.5))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))

    receipt = bench._dspark(object(), [10, 11, 12], 1, object())

    assert receipt["prompt_tokens"] == 3
    assert receipt["new_prefill_tokens"] == 3
    assert receipt["prefill_tok_s"] == 6.0
    assert receipt["ttft_s"] == 2.5


def test_python_vocabulary_prompt_uses_unique_normal_ids_before_exact_tail(
    monkeypatch,
) -> None:
    class Tokenizer:
        vocab_size = 20
        all_special_ids = (0, 19)

        @staticmethod
        def encode(_text):
            return list(range(10, 18))

    monkeypatch.setattr(
        "mtplx.benchmarks.programming_prompts.build_unique_programming_context",
        lambda **_kwargs: "coherent unique Python repository task",
    )

    token_ids, metadata = bench._python_vocabulary_prompt_ids(
        Tokenizer(),
        context_tokens=20,
        python_prompt_tokens=8,
    )

    filler = token_ids[:-8]
    assert len(token_ids) == 20
    assert token_ids[-8:] == list(range(10, 18))
    assert len(filler) == len(set(filler))
    assert not set(filler) & {0, 19}
    assert metadata["prompt_policy"] == "python_vocab_tail_v1"
    assert metadata["python_prompt_tokens"] == 8
    assert metadata["vocabulary_filler_tokens"] == 12
    assert metadata["vocabulary_unique_ids"] == 12
    assert metadata["vocabulary_duplicate_ids"] == 0


def test_mia_committed_token_divergence_is_always_enforced(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "deepseek_v4_mtpk_bench",
        SimpleNamespace(
            _divergence=lambda candidate, control: {
                "pass": candidate == control,
                "first_divergence_index": 1,
            },
            _exactness_is_enforced=lambda _required: False,
        ),
    )
    gate = bench._deepseek_quality_gate([10, 12], [10, 11])

    assert gate["pass"] is False
    assert gate["first_divergence_index"] == 1
    assert gate["enforced"] is True


def test_first_epoch_uses_sealed_fixed_linear_summary_without_profiling(
    monkeypatch,
) -> None:
    context = SimpleNamespace(diagnostics=SimpleNamespace(mode="off"))
    bundle = SimpleNamespace(
        target_model=object(),
        target_ops=object(),
        tokenizer=object(),
        draft_model=object(),
        draft_backend=object(),
    )
    summary = SummaryEvent(
        elapsed_us=10.0,
        prompt_token_count=3,
        generated_token_ids=(41, 42, 43, 44, 45, 46),
        generation_tokens=6,
        accepted_from_draft=3,
        acceptance_ratio=0.5,
        cycles_completed=3,
        phase_timings_us={"prefill": 4.0},
        block_tokens=6,
        verify_len_cap=6,
        acceptance_history=(3, 0, 0),
    )
    observed = {}

    def fake_stream(**kwargs):
        observed.update(kwargs)
        yield summary

    monkeypatch.setattr("dflash_mlx.runtime.stream_dflash_generate", fake_stream)

    receipt = bench._first_epoch(bundle, [10, 11, 12], context)

    assert observed["runtime_context"] is context
    assert observed["max_new_tokens"] == 6
    assert observed["block_tokens"] == 6
    assert receipt == {
        "cycle": 1,
        "block_len": 6,
        "proposed_token_count": 6,
        "future_draft_count": 5,
        "physical_verify_width": 6,
        "acceptance_len": 3,
        "commit_count": 4,
        "committed_output_ids": [41, 42, 43, 44],
        "committed_output_relation": "summary_generated_prefix",
        "summary": summary.to_payload(),
    }


def test_target_control_uses_explicit_sealed_ar_entrypoint(monkeypatch) -> None:
    observed = {}

    class Stats:
        @staticmethod
        def to_dict():
            return {
                "events": [{"prompt_token_count": 1}],
                "new_prefill_tokens": 1,
                "prompt_eval_time_s": 0.5,
                "decode_elapsed_s": 0.25,
                "elapsed_s": 0.75,
                "decode_tok_s": 4.0,
                "peak_memory_bytes": 123,
                "accepted_drafts": 0,
                "drafted_tokens": 0,
            }

    def fake_sealed_ar(runtime, prompt_ids, **kwargs):
        observed.update(runtime=runtime, prompt_ids=prompt_ids, kwargs=kwargs)
        return SimpleNamespace(tokens=[7], stats=Stats())

    monkeypatch.setattr(
        "mtplx.generation.generate_sealed_target_ar",
        fake_sealed_ar,
        raising=False,
    )
    runtime = object()

    receipt = bench._target_ar(
        SimpleNamespace(runtime=runtime),
        [5],
        max_tokens=1,
    )

    assert observed["runtime"] is runtime
    assert observed["prompt_ids"] == [5]
    assert observed["kwargs"]["max_tokens"] == 1
    assert receipt["tokens"] == [7]


def test_source_preflight_rejects_dirty_and_records_clean_commit(monkeypatch) -> None:
    monkeypatch.setattr(
        bench.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "?? bench/deepseek-v4-mia/receipt.json\n",
    )
    with pytest.raises(RuntimeError, match="worktree is dirty"):
        bench._require_clean_source(bench.Path("/repo"))

    commit = "a" * 40
    output = iter(("", f"{commit}\n"))
    monkeypatch.setattr(
        bench.subprocess,
        "check_output",
        lambda *_args, **_kwargs: next(output),
    )
    assert bench._require_clean_source(bench.Path("/repo")) == commit

    main_source = inspect.getsource(bench.main)
    assert main_source.index("source_commit = _require_clean_source(repo)") < (
        main_source.index("import mlx.core as mx")
    )
    assert '"source_head": source_commit' in main_source


def test_cache_contract_reports_fixed_arena_separately_from_request_span() -> None:
    requested_span = 16_384 + 1_024
    plan = SimpleNamespace(
        context_capacity_tokens=384_000,
        target_physical_capacity_tokens=384_005,
        max_batch_tokens=8_224,
        page_geometry=tuple(
            SimpleNamespace(
                layer_id=layer_id,
                compress_ratio=ratio,
                compressed_capacity=(
                    0 if ratio == 0 else (384_005 + ratio - 1) // ratio
                ),
            )
            for layer_id, ratio in enumerate((0, 4, 128))
        ),
    )

    def target_cache(ratio: int):
        values = {
            "compress_ratio": ratio,
            "window": SimpleNamespace(
                mode="nvfp4_stock432_fixed_window",
                record_bytes=432,
                capacity=8_416,
            ),
        }
        if ratio:
            values["compressed"] = SimpleNamespace(
                mode="nvfp4_stock432_paged",
                record_bytes=432,
                capacity=(384_005 + ratio - 1) // ratio,
            )
        if ratio == 4:
            values["index_compressed"] = SimpleNamespace(
                mode="fp8_e4m3_ue8m0_scale132_paged",
                record_bytes=132,
                capacity=96_002,
            )
        return SimpleNamespace(**values)

    target = [target_cache(ratio) for ratio in (0, 4, 128)]
    draft = [
        SimpleNamespace(
            ring=SimpleNamespace(
                mode="nvfp4_stock432_fixed_ring",
                record_bytes=432,
                nbytes=128 * 432,
            )
        )
        for _ in range(3)
    ]
    calls = []

    class TargetOps:
        @staticmethod
        def make_cache(_model, **kwargs):
            calls.append(("target", kwargs))
            return target

        @staticmethod
        def cleanup_generation_caches(target_arg, draft_arg):
            assert target_arg is target
            assert draft_arg is draft
            calls.append(("cleanup", None))

    class DraftBackend:
        @staticmethod
        def make_cache(**kwargs):
            calls.append(("draft", kwargs))
            return draft

    bundle = SimpleNamespace(
        target_model=SimpleNamespace(_mia_engine_plan=plan),
        target_ops=TargetOps(),
        draft_model=SimpleNamespace(args=SimpleNamespace(sliding_window=128)),
        draft_backend=DraftBackend(),
    )

    contract = bench._cache_contract(
        bundle,
        requested_span_tokens=requested_span,
    )

    assert contract["request"]["span_tokens"] == requested_span
    assert contract["installed_cache_plan"] == {
        "context_capacity_tokens": 384_000,
        "target_physical_capacity_tokens": 384_005,
        "max_batch_tokens": 8_224,
    }
    assert contract["target_kv"]["window_capacity_records"] == 8_416
    assert "capacity_tokens" not in contract["target_kv"]
    assert contract["target_kv"]["compressed_pages"] == [
        {"compress_ratio": 4, "capacity_records": 96_002, "layers": 1},
        {"compress_ratio": 128, "capacity_records": 3_001, "layers": 1},
    ]
    assert contract["target_kv"]["indexer_capacity_records"] == 96_002
    assert contract["dspark_kv"]["ring_capacity_records"] == 128
    assert calls[0] == (
        "target",
        {
            "enable_speculative_linear_cache": True,
            "quantize_kv_cache": False,
            "cache_capacity_tokens": requested_span,
        },
    )
    assert calls[-1] == ("cleanup", None)


def test_memory_receipt_labels_mlx_arm_peak_and_process_lifetime_rss(
    monkeypatch,
) -> None:
    observed_mlx = iter((101, 202))
    monkeypatch.setattr(bench, "_memory", lambda _getter: next(observed_mlx))
    monkeypatch.setattr(bench, "_process_peak_rss_bytes", lambda: 303)

    receipt = bench._finish_memory_receipt(
        mlx_active_after_load_bytes=11,
        mlx_peak_after_load_bytes=22,
        process_peak_rss_after_load_bytes=33,
        mlx_peak_reset_before_arm=True,
        mlx_active_before_arm_bytes=44,
        mlx_peak_before_arm_bytes=0,
    )

    assert receipt == {
        "mlx": {
            "after_load": {
                "active_bytes": 11,
                "peak_bytes": 22,
                "peak_scope": "process_lifetime_through_load",
            },
            "arm": {
                "active_bytes_before": 44,
                "peak_bytes_before": 0,
                "active_bytes_after": 101,
                "peak_bytes": 202,
                "peak_reset_before_arm": True,
                "peak_scope": "since_explicit_arm_reset",
            },
        },
        "process": {
            "peak_rss_bytes_after_load": 33,
            "peak_rss_bytes_after_arm": 303,
            "peak_rss_scope": "process_lifetime_since_exec",
        },
    }
