from __future__ import annotations

from types import SimpleNamespace

from mtplx.benchmarks.runners import mtp_depth_sweep


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
    monkeypatch.setattr(mtp_depth_sweep, "load_prompt_suite", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "mtplx.draft_lm_head._install_draft_lm_head",
        lambda runtime, **kwargs: calls.append((runtime, kwargs)) or {"installed": True},
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
    monkeypatch.setattr(mtp_depth_sweep, "load_prompt_suite", lambda *_args, **_kwargs: [])

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
    assert result["mtp_adapter_merge_report"] == {"merged": 1, "targets": [{"target": "fc"}]}


def test_sum_draft_core_preserves_greedy_confidence_counters() -> None:
    summary = mtp_depth_sweep._sum_draft_core(
        [
            {
                "requested": "stock",
                "greedy_confidence_sync_calls": 3,
                "greedy_confidence_token_reuses": 2,
            },
            {
                "requested": "stock",
                "greedy_confidence_sync_calls": 4,
                "greedy_confidence_token_reuses": 4,
            },
            {"requested": "stock"},
        ]
    )

    assert summary["greedy_confidence_sync_calls"] == 7
    assert summary["greedy_confidence_token_reuses"] == 6
