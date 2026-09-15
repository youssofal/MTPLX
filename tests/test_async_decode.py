"""Double-buffered AR decode (PR #396 by maceip, ported dark).

Upstream shipped the double-buffer default-on with MTPLX_SYNC_AR as the
opt-out. House default-flip discipline lands it dark instead: the shipping
default keeps the historical blocking eval and MTPLX_ASYNC_AR=1 arms the
double-buffer. These tests pin the dark-port contract.
"""

import os
from pathlib import Path

import mlx.core as mx
import pytest

from mtplx.generation import generate_ar
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class _DeferredToken:
    def __init__(self):
        self._array = mx.array([0], dtype=mx.int64)
        self._filled = False

    def array(self):
        return self._array

    def fill(self, _token):
        if self._filled:
            raise RuntimeError("deferred AR token already filled")
        self._filled = True


class TinyModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def sanitize(self, weights):
        return weights

    def make_ar_pipeline_token(self):
        return _DeferredToken()

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            if return_hidden:
                return None, hidden
            return None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


def _make_runtime(model: TinyModel) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def _run(monkeypatch, **env) -> list[int]:
    for key in ("MTPLX_ASYNC_AR", "MTPLX_EVAL_AUDIT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    rt = _make_runtime(TinyModel())
    out = generate_ar(
        rt,
        [0],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )
    return out.tokens


def test_dark_default_stays_synchronous_and_generates(monkeypatch):
    """Shipping default (no env): blocking eval path, identical output."""
    tokens = _run(monkeypatch)
    assert tokens == [1, 1, 1, 1]


def test_async_armed_generates_identically(monkeypatch):
    """MTPLX_ASYNC_AR=1 arms the double-buffer; graph identical, so tokens too."""
    tokens = _run(monkeypatch, MTPLX_ASYNC_AR="1")
    assert tokens == [1, 1, 1, 1]


def test_eval_audit_forces_synchronous(monkeypatch, tmp_path):
    """Audit runs never go async even when armed.

    MTPLX_EVAL_AUDIT is a file path, so the test points it at tmp_path; a bare
    "1" would leave a file literally named 1 in the working directory."""
    audit = tmp_path / "eval-audit.jsonl"
    tokens = _run(monkeypatch, MTPLX_ASYNC_AR="1", MTPLX_EVAL_AUDIT=str(audit))
    assert tokens == [1, 1, 1, 1]
    assert audit.exists() and audit.read_text().count("\n") >= 1


def test_pipeline_lane_gating_independent_of_async_flag(monkeypatch):
    """The MTPLX_AR_PIPELINE lane keeps its own gate; the async flag neither
    arms nor blocks it (dark-port divergence from upstream #396, which coupled
    the two)."""
    monkeypatch.delenv("MTPLX_ASYNC_AR", raising=False)
    monkeypatch.setenv("MTPLX_AR_PIPELINE", "1")

    class PipelineModel(TinyModel):
        def __init__(self):
            super().__init__()
            self.pipeline_mode_calls: list[bool] = []

        def set_ar_pipeline_mode(self, val):
            self.pipeline_mode_calls.append(bool(val))
            # Refuse engagement so the classic loop still runs on TinyModel.
            return False

    model = PipelineModel()
    rt = _make_runtime(model)
    out = generate_ar(
        rt,
        [0],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.7, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )
    assert len(out.tokens) == 4
    # The lane was OFFERED engagement (gate independent of MTPLX_ASYNC_AR).
    assert model.pipeline_mode_calls[:1] == [True]


def test_pipeline_flushes_each_forward_before_async_submission(monkeypatch):
    monkeypatch.setenv("MTPLX_AR_PIPELINE", "1")
    monkeypatch.delenv("MTPLX_ASYNC_AR", raising=False)
    trace = []

    class PipelineModel(TinyModel):
        def __init__(self):
            super().__init__()
            self.active = False

        def set_ar_pipeline_mode(self, value):
            self.active = bool(value)
            trace.append(("mode", self.active))
            return True

        def flush_ar_pipeline_ple(self):
            trace.append(("flush",))

        def discard_ar_pipeline_ple(self):
            trace.append(("discard",))

        def __call__(self, *args, **kwargs):
            if self.active:
                trace.append(("forward",))
            return super().__call__(*args, **kwargs)

    original_async_eval = mx.async_eval

    def tracked_async_eval(*arrays):
        trace.append(("async_eval",))
        return original_async_eval(*arrays)

    monkeypatch.setattr(mx, "async_eval", tracked_async_eval)
    out = generate_ar(
        _make_runtime(PipelineModel()),
        [0],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.7, top_p=1.0, top_k=4),
        stop_token_ids=set(),
        seed=7,
    )

    assert len(out.tokens) == 4
    assert out.stats.ar_pipeline_active is True
    assert out.stats.ar_pipeline_variant == "unknown"
    forward_positions = [i for i, event in enumerate(trace) if event == ("forward",)]
    assert forward_positions
    for position in forward_positions:
        assert trace[position + 1 : position + 3] == [("flush",), ("async_eval",)]
    assert trace[-1] == ("mode", False)


def test_pipeline_flush_failure_discards_before_disabling(monkeypatch):
    monkeypatch.setenv("MTPLX_AR_PIPELINE", "1")
    trace = []

    class PipelineModel(TinyModel):
        def set_ar_pipeline_mode(self, value):
            trace.append(("mode", bool(value)))
            return True

        def flush_ar_pipeline_ple(self):
            trace.append(("flush",))
            raise RuntimeError("fill failed")

        def discard_ar_pipeline_ple(self):
            trace.append(("discard",))

    with pytest.raises(RuntimeError, match="fill failed"):
        generate_ar(
            _make_runtime(PipelineModel()),
            [0],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.7, top_p=1.0, top_k=4),
            stop_token_ids=set(),
            seed=7,
        )
    assert trace[-3:] == [("flush",), ("discard",), ("mode", False)]


def test_pipeline_replays_classic_temperature_one_tokens(monkeypatch):
    """Pipelining may overlap graph construction, but it must not replace the
    request's sampler or RNG stream."""

    class PipelineModel(TinyModel):
        def set_ar_pipeline_mode(self, value):
            self.active = bool(value)
            return True

        def flush_ar_pipeline_ple(self):
            pass

        def discard_ar_pipeline_ple(self):
            pass

    sampler = SamplerConfig(temperature=1.0, top_p=0.95, top_k=4)

    monkeypatch.setenv("MTPLX_AR_PIPELINE", "0")
    control = generate_ar(
        _make_runtime(TinyModel()),
        [0],
        max_tokens=32,
        sampler=sampler,
        stop_token_ids=set(),
        seed=20260829,
    )

    monkeypatch.setenv("MTPLX_AR_PIPELINE", "1")
    candidate = generate_ar(
        _make_runtime(PipelineModel()),
        [0],
        max_tokens=32,
        sampler=sampler,
        stop_token_ids=set(),
        seed=20260829,
    )

    assert candidate.stats.ar_pipeline_active is True
    assert candidate.tokens == control.tokens


def test_pipeline_engagement_fields_reach_the_public_stats_envelope():
    from mtplx.server.openai import _public_mtplx_stats

    public = _public_mtplx_stats(
        {
            "stats": {
                "mode": "ar",
                "ar_pipeline_active": True,
                "ar_pipeline_variant": "streamed",
            }
        }
    )

    assert public["ar_pipeline_active"] is True
    assert public["ar_pipeline_variant"] == "streamed"
