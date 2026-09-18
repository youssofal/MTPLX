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


class TinyModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def sanitize(self, weights):
        return weights

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


class _SpreadModel(TinyModel):
    """Deterministic but high-entropy logits that shift with the last token,
    so a divergent RNG stream picks a different continuation.

    ``set_ar_pipeline_mode`` returns True, so the (now removed) device-sampled
    pipeline lane would engage on this stub — the regression is that
    MTPLX_AR_PIPELINE must still emit the classic loop's tokens, not an
    mx.random continuation.
    """

    VOCAB = 64

    def __init__(self):
        super().__init__()
        self.pipeline_mode_calls: list[bool] = []

    def set_ar_pipeline_mode(self, enabled):
        self.pipeline_mode_calls.append(bool(enabled))
        return True

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
            return (None, hidden) if return_hidden else None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        j = mx.arange(self.VOCAB, dtype=mx.float32)[None, :]
        last = input_ids[:, -1:].astype(mx.float32)  # (1, 1): drives per-step shift
        row = 0.9 * mx.cos(0.4 * j + 0.7 * last) + 0.6 * mx.sin(0.2 * j - 0.5 * last)
        logits = mx.broadcast_to(row[:, None, :], (1, keep, self.VOCAB))
        if return_hidden:
            return logits, hidden
        return logits


def _spread_tokens(
    monkeypatch, *, pipeline: bool, seed: int, max_tokens: int, prompt
) -> list[int]:
    for key in ("MTPLX_ASYNC_AR", "MTPLX_EVAL_AUDIT", "MTPLX_AR_PIPELINE"):
        monkeypatch.delenv(key, raising=False)
    if pipeline:
        monkeypatch.setenv("MTPLX_AR_PIPELINE", "1")
    rt = _make_runtime(_SpreadModel())
    # CPU only: this box gates every Metal exec behind the GPU flock, and the
    # comparison is device-agnostic (same device on both runs).
    with mx.stream(mx.cpu):
        out = generate_ar(
            rt,
            prompt,
            max_tokens=max_tokens,
            sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=20),
            stop_token_ids=set(),
            seed=seed,
        )
    return out.tokens


def test_ar_pipeline_matches_classic_token_for_token(monkeypatch):
    """MTPLX_AR_PIPELINE=1 emits the SAME tokens as the classic AR loop for one
    request and seed.

    The old lane drew output token 0 with the request's numpy rng but every
    later token from an mx.random stream keyed off the seed; the streams
    differ, so the flag diverged from classic AR (PR #501's own test asserted
    divergence by token 3). The fix routes the flag through the classic
    per-token ``_sample_from_logits(row, sampler, rng)`` draw, so the two token
    sequences are byte-identical."""
    seed = 20250917
    prompt = [5]
    max_tokens = 12
    classic = _spread_tokens(
        monkeypatch, pipeline=False, seed=seed, max_tokens=max_tokens, prompt=prompt
    )
    pipelined = _spread_tokens(
        monkeypatch, pipeline=True, seed=seed, max_tokens=max_tokens, prompt=prompt
    )
    assert len(classic) == max_tokens
    # A degenerate (near one-hot) distribution would let the streams agree by
    # luck and rob the test of its teeth; require real sampling variety.
    assert len(set(classic)) > 1
    assert pipelined == classic
