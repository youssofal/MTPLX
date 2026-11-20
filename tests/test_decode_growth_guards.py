"""Long-generation growth guards + batched-lane context copy (2026-08-28).

Covers the two decode-decay fixes:

1. The committed MTP-history cache is bounded DURING decode: once a
   generation appends MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD tokens, the
   cache resets and regrows (draft conditioning only — output law unchanged).
   Keyed on live appends so a bank-restored seed is never dropped at round 1.

2. The clear-cache cadence arms mid-generation when the LIVE total crosses
   the auto threshold (it used to read only the prefill-time context, so a
   66-token prompt with a 34k-token answer never cleared once).

3. Context-copy block rounds run on the batched verify lane (qwen4_exp's
   lane): a prompt n-gram match proposes the prompt continuation as one
   block through the lane's normal verify forward, with the same point-mass
   probability-ratio acceptance — the emitted stream is unchanged.
"""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.generation import SamplerConfig, generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime

VOCAB = 16


class _ParrotTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class _CountingMTPCacheEntry:
    """Offset-bearing stand-in for the draft head's history cache entry."""

    def __init__(self):
        self.offset = 0


class _ParrotModel:
    """Deterministic cycle machine over VOCAB tokens: after t comes
    (t+1) % VOCAB. With the full cycle as the prompt, the generation
    re-emits prompt content forever — the grounded-re-emission regime the
    copy lane targets."""

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return [_CountingMTPCacheEntry()]

    def mtp_update_cache(self, hidden_states, next_token_ids, mtp_cache=None, **_kwargs):
        if mtp_cache:
            mtp_cache[0].offset += int(np.asarray(next_token_ids).size)
        return hidden_states

    def _logits_for(self, last_tokens: list[int]) -> mx.array:
        rows = []
        for token in last_tokens:
            row = [0.0] * VOCAB
            row[(int(token) + 1) % VOCAB] = 32.0
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
        keep = (
            len(tokens)
            if logits_keep is None
            else min(len(tokens), max(1, int(logits_keep)))
        )
        logits = self._logits_for(tokens[-keep:]) if emit_logits else None
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        if return_hidden:
            return logits, hidden
        return logits


class _ParrotMTPModel(_ParrotModel):
    def __init__(self):
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

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


def _runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=_ParrotMTPModel(),
        tokenizer=_ParrotTokenizer(),
        model_path=Path("tiny-parrot"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


_FULL_CYCLE_PROMPT = list(range(VOCAB))


def _expected_cycle(start: int, length: int) -> list[int]:
    return [(start + 1 + index) % VOCAB for index in range(length)]


def _run(monkeypatch, *, max_tokens: int = 220, **env: str):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return generate_mtpk(
        _runtime(),
        list(_FULL_CYCLE_PROMPT),
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=1),
        speculative_depth=2,
        seed=11,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )


def test_batched_lane_copy_rounds_engage_and_output_is_exact(monkeypatch):
    out = _run(monkeypatch, MTPLX_CONTEXT_COPY_BATCHED="1")
    assert out.stats.context_copy_active
    assert out.stats.context_copy_rounds > 0
    assert out.stats.context_copy_accepted_tokens > 0
    assert list(out.tokens) == _expected_cycle(
        _FULL_CYCLE_PROMPT[-1], len(out.tokens)
    )


def test_batched_copy_kill_switch_keeps_stream_and_disables_lane(monkeypatch):
    on = _run(monkeypatch, MTPLX_CONTEXT_COPY_BATCHED="1")
    off = _run(monkeypatch, MTPLX_CONTEXT_COPY_BATCHED="0")
    assert list(on.tokens) == list(off.tokens)
    assert off.stats.context_copy_rounds == 0
    # Copy blocks commit far more per verify call on re-emission.
    assert on.stats.verify_calls < off.stats.verify_calls


def test_committed_history_live_reset_fires_and_stream_is_unchanged(monkeypatch):
    bounded = _run(
        monkeypatch,
        MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD="64",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    unbounded = _run(
        monkeypatch,
        MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD="0",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert bounded.stats.mtp_history_live_resets >= 1
    assert bounded.stats.mtp_history_live_reset_threshold == 64
    assert unbounded.stats.mtp_history_live_resets == 0
    # Draft-history conditioning affects acceptance only, never the stream.
    assert list(bounded.tokens) == list(unbounded.tokens)


def test_committed_history_reset_keys_on_live_appends_not_seed(monkeypatch):
    # A 220-token generation with a threshold ABOVE it must never reset even
    # though the prompt seed already sits in the cache — restored long
    # sessions must not be dropped at round 1.
    out = _run(
        monkeypatch,
        MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD="100000",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert out.stats.mtp_history_live_resets == 0


def test_clear_cache_live_latch_arms_past_threshold(monkeypatch):
    out = _run(
        monkeypatch,
        max_tokens=260,
        MTPLX_CLEAR_CACHE_EVERY="auto",
        MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD="128",
        MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT="32",
        MTPLX_SUSTAINED_PREFILL_LAYOUT="contiguous_dense_decode",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert out.stats.clear_cache_events >= 1


def test_clear_cache_stays_off_below_threshold(monkeypatch):
    out = _run(
        monkeypatch,
        max_tokens=64,
        MTPLX_CLEAR_CACHE_EVERY="auto",
        MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD="100000",
        MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT="32",
        MTPLX_SUSTAINED_PREFILL_LAYOUT="contiguous_dense_decode",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert out.stats.clear_cache_events == 0


def test_a_long_prefill_is_cleared_out_of_the_cache_at_the_first_round(monkeypatch):
    """A long prefill's freed buffers are not the answer's sizes: the first
    round after it clears MLX's cache, though the answer never reaches the
    cadence's first clear (window n4a4, 12 Nov: 97.7 GB held for the first
    120 tokens of a 131K answer against 92.1 GB of live arrays)."""

    calls: list[str] = []
    monkeypatch.setattr("mtplx.generation.mx.clear_cache", lambda: calls.append("clear"))
    out = _run(
        monkeypatch,
        max_tokens=64,
        MTPLX_CLEAR_CACHE_EVERY="auto",
        MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD="8",  # the 16-token prompt is "long"
        MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT="100000",  # the cadence alone never fires
        MTPLX_SUSTAINED_PREFILL_LAYOUT="contiguous_dense_decode",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert out.stats.clear_cache_events == 1
    assert calls.count("clear") >= 1
    assert list(out.tokens) == _expected_cycle(_FULL_CYCLE_PROMPT[-1], len(out.tokens))


def test_a_short_prefill_keeps_the_cache_for_the_answer(monkeypatch):
    out = _run(
        monkeypatch,
        max_tokens=64,
        MTPLX_CLEAR_CACHE_EVERY="auto",
        MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD="1000",
        MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT="100000",
        MTPLX_SUSTAINED_PREFILL_LAYOUT="contiguous_dense_decode",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert out.stats.clear_cache_events == 0


def test_an_operator_who_turned_the_clears_off_gets_none(monkeypatch):
    out = _run(
        monkeypatch,
        max_tokens=64,
        MTPLX_CLEAR_CACHE_EVERY="0",
        MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD="8",
        MTPLX_SUSTAINED_PREFILL_LAYOUT="contiguous_dense_decode",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert out.stats.clear_cache_events == 0



def _restored(monkeypatch, *, source: str) -> None:
    """The prompt comes back from ``source`` with one new token, as a
    follow-up's does: the real prefill runs, and its state then reads as a
    restore of all but the last token."""

    import dataclasses

    from mtplx import generation

    real = generation.restore_or_prefill_prompt_state

    def restored(*args, **kwargs):
        state = real(*args, **kwargs)
        cached = len(state.token_prefix) - 1
        return dataclasses.replace(
            state,
            cached_tokens=cached,
            suffix_tokens=1,
            cache_hit=True,
            cache_source=source,
            ssd_cache_hit=source == "ssd",
            ssd_cached_tokens=cached if source == "ssd" else 0,
        )

    monkeypatch.setattr(generation, "restore_or_prefill_prompt_state", restored)


def test_a_long_conversation_restored_from_ssd_is_cleared_out_of_the_cache_at_the_first_round(
    monkeypatch,
):
    """An SSD restore frees buffers behind a short suffix (the decode's read
    buffers, each layer's buffers from before it grew): in window n4k1 (12
    Nov, the built candidate) a 118,096-token restore left 6.25 to 6.73 GB in
    MLX's cache for the answer's first 1,041 tokens against 88.38 GB of live
    arrays. The first round clears them, as after a long prefill."""

    _restored(monkeypatch, source="ssd")
    calls: list[str] = []
    monkeypatch.setattr("mtplx.generation.mx.clear_cache", lambda: calls.append("clear"))
    out = _run(
        monkeypatch,
        max_tokens=64,
        MTPLX_CLEAR_CACHE_EVERY="auto",
        MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD="8",  # the 15 restored tokens are "long"
        MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT="100000",  # the cadence alone never fires
        MTPLX_SUSTAINED_PREFILL_LAYOUT="contiguous_dense_decode",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert out.stats.clear_cache_events == 1
    assert calls.count("clear") >= 1
    assert list(out.tokens) == _expected_cycle(_FULL_CYCLE_PROMPT[-1], len(out.tokens))


def test_a_conversation_served_from_ram_keeps_the_cache_for_the_answer(monkeypatch):
    """A lease or RAM restore allocated nothing to free: no clear at its first round."""

    _restored(monkeypatch, source="ram")
    out = _run(
        monkeypatch,
        max_tokens=64,
        MTPLX_CLEAR_CACHE_EVERY="auto",
        MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD="8",
        MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT="100000",
        MTPLX_SUSTAINED_PREFILL_LAYOUT="contiguous_dense_decode",
        MTPLX_CONTEXT_COPY_BATCHED="0",
    )
    assert out.stats.clear_cache_events == 0
