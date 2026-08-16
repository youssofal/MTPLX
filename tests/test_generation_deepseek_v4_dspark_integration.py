"""Upstream request-option boundary for the DSpark native backend."""

from __future__ import annotations

import pytest

from mtplx.generation import generate_mtpk
from mtplx.sampling import SamplerConfig
from mtplx.thinking_guard import ThinkingGuardConfig
from tests.test_deepseek_v4_dspark_generation import _DSparkRuntime


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("session_bank", object(), "session bank"),
        ("capture_final_state", True, "final state"),
        ("trace_label", "diagnostic", "decode trace"),
        ("repetition_stop", True, "repetition stop"),
        ("loop_guard", True, "loop guard"),
        (
            "thinking_guard",
            ThinkingGuardConfig(enabled=True),
            "thinking guard",
        ),
    ],
)
def test_dspark_rejects_unsupported_upstream_features_before_prefill(
    option, value, message
):
    rt = _DSparkRuntime()

    with pytest.raises(ValueError, match=message):
        generate_mtpk(
            rt,
            [10],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0),
            speculative_depth=2,
            stop_token_ids=set(),
            **{option: value},
        )

    assert rt.target_cache.offset == 0
    assert rt.prefill_hidden is None


def test_dspark_rejects_construction_selected_decode_trace_before_prefill(tmp_path):
    rt = _DSparkRuntime()
    rt.block_speculative_decode_trace_requested = True

    with pytest.raises(ValueError, match="decode trace"):
        generate_mtpk(
            rt,
            [10],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0),
            speculative_depth=2,
            stop_token_ids=set(),
        )

    assert rt.target_cache.offset == 0
    assert not (tmp_path / "trace.jsonl").exists()


def test_dspark_rejects_generic_mtp_policy_options_before_prefill():
    rt = _DSparkRuntime()

    with pytest.raises(ValueError, match="mtp_cache_policy"):
        generate_mtpk(
            rt,
            [10],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0),
            speculative_depth=2,
            stop_token_ids=set(),
            mtp_cache_policy="fresh",
        )

    assert rt.target_cache.offset == 0


def test_dspark_allows_inert_session_metadata_when_cache_is_bypassed(monkeypatch):
    rt = _DSparkRuntime()
    sentinel = object()

    def fake_generate(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(
        "mtplx.native_block_speculation.generate_native_block_speculative",
        fake_generate,
    )

    result = generate_mtpk(
        rt,
        [10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
        session_id="stateless-request-label",
        session_template_hash="unused-without-bank",
        session_draft_head_identity="unused-without-bank",
        session_policy_fingerprint="unused-without-bank",
    )

    assert result is sentinel
