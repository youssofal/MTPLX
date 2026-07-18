"""Context-copy GenerationStats counters: probes, rounds, accepted blocks/tokens,
suspend/backoff state, public-envelope exposure (#151 follow-up).

Deterministic next-token stub models drive generate_mtpk on CPU:
- a mod-VOCAB cycle whose prompt continuation always agrees with the model
  (full-accept copy rounds),
- a non-repeating ramp whose tail never matches a prompt gram (probe misses),
- a "trap" cycle whose prompt continuation always disagrees (zero-acceptance
  rounds driving the EMA into suspension + exponential backoff).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.generation import GenerationStats, generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class _ScriptedModel:
    """Deterministic automaton: after token t the model wants next_map(t)."""

    def __init__(self, vocab: int, next_map):
        self.vocab = vocab
        self.next_map = next_map
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _logits_for(self, last_tokens: list[int]) -> mx.array:
        rows = []
        for token in last_tokens:
            row = [0.0] * self.vocab
            row[self.next_map(int(token)) % self.vocab] = 10.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

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
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        keep = len(tokens) if logits_keep is None else min(len(tokens), max(1, int(logits_keep)))
        logits = self._logits_for(tokens[-keep:]) if emit_logits else None
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        if return_hidden:
            return logits, hidden
        return logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        tokens = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        logits = self._logits_for(tokens)
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


def _runtime(model: _ScriptedModel) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path("tiny-copy"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _mtpk(model: _ScriptedModel, prompt: list[int], max_tokens: int):
    return generate_mtpk(
        _runtime(model),
        prompt,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=1,
        seed=0,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
    )


def _clean_env(monkeypatch) -> None:
    for name in (
        "MTPLX_CONTEXT_COPY",
        "MTPLX_CONTEXT_COPY_K",
        "MTPLX_CONTEXT_COPY_NGMIN",
        "MTPLX_CONTEXT_COPY_NGMAX",
        "MTPLX_CONTEXT_COPY_MINEXT",
        "MTPLX_SKIP_VERIFY_SNAPSHOT",
    ):
        monkeypatch.delenv(name, raising=False)


# --- full-accept copy rounds (mod-8 cycle, prompt continuation agrees) ---


def test_full_accept_rounds_count_probes_rounds_blocks_tokens(monkeypatch):
    _clean_env(monkeypatch)
    out = _mtpk(_ScriptedModel(8, lambda t: t + 1), [0, 1, 2, 3, 4, 5, 6, 7, 0], max_tokens=80)
    stats = out.stats
    assert stats.context_copy_active is True
    assert stats.context_copy_disabled_reason is None
    assert stats.context_copy_rounds >= 2
    # Early cycles probe before the generated tail re-enters a prompt gram.
    assert stats.context_copy_probes > stats.context_copy_rounds
    # The prompt continuation is exactly what the model wants: every proposed
    # block verifies in full.
    assert stats.context_copy_accepted_blocks == stats.context_copy_rounds
    assert stats.context_copy_drafted_tokens == stats.context_copy_accepted_tokens
    assert stats.context_copy_accepted_tokens > 0
    assert stats.context_copy_suspensions == 0
    assert stats.context_copy_suspended is False
    assert stats.context_copy_backoff_tokens == 64


def test_kill_switch_disables_and_output_is_byte_identical(monkeypatch):
    _clean_env(monkeypatch)
    on = _mtpk(_ScriptedModel(8, lambda t: t + 1), [0, 1, 2, 3, 4, 5, 6, 7, 0], max_tokens=80)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    off = _mtpk(_ScriptedModel(8, lambda t: t + 1), [0, 1, 2, 3, 4, 5, 6, 7, 0], max_tokens=80)
    assert off.stats.context_copy_active is False
    assert off.stats.context_copy_probes == 0
    assert off.stats.context_copy_rounds == 0
    assert off.stats.context_copy_backoff_tokens == 0
    assert list(on.tokens) == list(off.tokens)


# --- probe misses (non-repeating ramp, tail never matches a prompt gram) ---


def test_probe_misses_count_probes_but_no_rounds(monkeypatch):
    _clean_env(monkeypatch)
    out = _mtpk(_ScriptedModel(64, lambda t: t + 1), [0, 1, 2, 3, 4, 5], max_tokens=40)
    stats = out.stats
    assert stats.context_copy_active is True
    assert stats.context_copy_probes > 0
    assert stats.context_copy_rounds == 0
    assert stats.context_copy_drafted_tokens == 0
    assert stats.context_copy_accepted_blocks == 0
    assert stats.context_copy_accepted_tokens == 0
    assert stats.context_copy_suspensions == 0


# --- zero-acceptance rounds -> suspension with exponential backoff ---


def _trap_next(t: int) -> int:
    # Cycle 10..15; every token outside the cycle re-enters it at 10. The
    # prompt continuation after the (10..15) gram is 99, which the model
    # never produces, so every copy round verifies 0 tokens.
    if 10 <= t < 15:
        return t + 1
    return 10


def test_zero_acceptance_rounds_drive_suspension_and_backoff(monkeypatch):
    _clean_env(monkeypatch)
    # Every rotation of the 10..15 cycle appears in the prompt, each followed
    # by a token the model never produces: whatever the generated tail's
    # alignment, the probe finds a match and the copy block verifies 0 tokens.
    cycle = [10, 11, 12, 13, 14, 15]
    prompt = []
    for i in range(6):
        prompt += cycle[i:] + cycle[:i] + [99 - i]
    out = _mtpk(_ScriptedModel(128, _trap_next), prompt, max_tokens=200)
    stats = out.stats
    assert stats.context_copy_active is True
    assert stats.context_copy_rounds >= 4
    assert stats.context_copy_drafted_tokens > 0
    assert stats.context_copy_accepted_blocks == 0
    assert stats.context_copy_accepted_tokens == 0
    assert stats.context_copy_suspensions >= 1
    # Each suspension doubles the retry backoff from its 64-token floor.
    assert stats.context_copy_backoff_tokens >= 128


# --- defaults + public envelope exposure ---


def test_generation_stats_defaults_serialize_context_copy_fields():
    stats = GenerationStats(mode="ar", generated_tokens=0, elapsed_s=0.0, tok_s=0.0)
    payload = stats.to_dict()
    assert payload["context_copy_active"] is False
    assert payload["context_copy_probes"] == 0
    assert payload["context_copy_rounds"] == 0
    assert payload["context_copy_drafted_tokens"] == 0
    assert payload["context_copy_accepted_blocks"] == 0
    assert payload["context_copy_accepted_tokens"] == 0
    assert payload["context_copy_suspensions"] == 0
    assert payload["context_copy_suspended"] is False
    assert payload["context_copy_backoff_tokens"] == 0
    assert payload["context_copy_disabled_reason"] is None


def test_public_mtplx_stats_expose_context_copy_counters():
    from mtplx.server.openai import PUBLIC_MTPLX_STATS_KEYS, _public_mtplx_stats

    keys = {
        "context_copy_active",
        "context_copy_probes",
        "context_copy_rounds",
        "context_copy_drafted_tokens",
        "context_copy_accepted_blocks",
        "context_copy_accepted_tokens",
        "context_copy_suspensions",
        "context_copy_suspended",
        "context_copy_backoff_tokens",
        "context_copy_disabled_reason",
    }
    assert keys <= set(PUBLIC_MTPLX_STATS_KEYS)
    generated = {"stats": {key: 1 for key in keys}}
    public = _public_mtplx_stats(generated)
    assert keys <= set(public)
