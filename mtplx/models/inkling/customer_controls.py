"""Explicit proposal sampling and the public free-form repetition stop."""

import numpy as np

from mtplx.context_copy import JoinedTokens
from mtplx.generation import (
    _detect_repeated_token_suffix, _long_cycle_stop, _RepetitionStreamGate,
    _repetition_stop_config, _repetition_stream_holdback_tokens, _sample_from_logits,
)


class ProposalSampler:
    def __init__(self, config, seed):
        self.config = config
        self.sampled = config is not None and config.temperature > 0
        self.penalties = config is not None and (config.presence_penalty or config.frequency_penalty)
        # A separate child stream leaves the target RNG untouched on accept and reject.
        self.rng = (np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
                    if self.sampled or self.penalties else None)

    def options(self, committed_counts):
        if self.rng is None:
            return {}
        counts = committed_counts.copy() if self.penalties else None

        def select_token(row):
            token, _ = _sample_from_logits(row, self.config, self.rng, token_counts=counts)
            if counts is not None:
                counts[token] += 1
            return token

        return {"select_token": select_token}

    def stats(self, dspark_tokens):
        return {
            "explicit": self.config is not None,
            "method": "sampled" if self.sampled else "argmax",
            "temperature": self.config.temperature if self.config is not None else 0.0,
            "top_p": self.config.top_p if self.config is not None else 1.0,
            "top_k": self.config.top_k if self.config is not None else 1,
            "dspark_tokens": dspark_tokens,
            "sampled_tokens": dspark_tokens if self.sampled else 0,
        }


class RepetitionOutput:
    """Observe committed target IDs once across anchor, verified and single-target rounds."""

    def __init__(self, callback, stop_ids):
        self.config = _repetition_stop_config(True)
        self.cycle = _long_cycle_stop(self.config)
        self.gate = (_RepetitionStreamGate(self.config, _repetition_stream_holdback_tokens(self.config))
                     if callback is not None else None)
        self.callback, self.stop_ids = callback, stop_ids
        self.observed = self.streamed = 0
        self.result = None

    def observe(self, tokens, pending=None):
        # stream() also observes an anchor or parked AR draw. A verified draw
        # already inspected before the next row must not be fed to the detector twice.
        history = JoinedTokens(tokens, pending) if pending else tokens
        if self.result is None and len(history) > self.observed:
            self.observed = len(history)
            self.result = _detect_repeated_token_suffix(history, self.config)
            if self.result is None and self.cycle is not None:
                self.result = self.cycle.observe(history)
        return self.result

    def stream(self, tokens, *, flush=False):
        self.observe(tokens)
        if self.result is not None:
            del tokens[self.result.trim_start:]
        if self.callback is None:
            return
        limit = len(tokens) if flush else self.gate.emit_limit(tokens)
        if limit > self.streamed:
            visible = [token for token in tokens[self.streamed:limit] if token not in self.stop_ids]
            self.streamed = limit
            if visible:
                self.callback(visible)

    def stats(self):
        if self.result is None:
            return {}
        return {
            "finish_stop_origin": "repetition_stop",
            "repetition_stop_triggered": True,
            "repetition_stop_reason": self.result.reason,
            "repetition_stop_block_tokens": self.result.block_tokens,
            "repetition_stop_repeats": self.result.repeats,
            "repetition_stop_trimmed_tokens": self.result.repeated_tokens,
            "repetition_stop_raw_tokens": self.observed,
        }
