"""End-to-end ar_batch penalty parity test (PR #157, mmmugh review).

Drives the REAL ``mlx_lm.generate.BatchGenerator`` with the real pump
append-after-``next()`` ordering and compares per-position output to the serial
``generate_ar`` path. Covers the mmmugh repro: token 7 leads token 3 by
exactly 1.0, so ``presence_penalty=1.5`` must flip the winner the instant 7
has been emitted once.

Why this test exists
--------------------
``tests/test_penalties.py`` (``_ar_batch_sample_once``) fabricates
``job.tokens`` directly and calls the sampler closure in isolation. That
verifies the penalty *math* but is blind to the *staleness* bug: the pump's
``emit_token`` runs AFTER ``generator.next()`` returns, but the sampler runs
one step ahead INSIDE ``next()`` (mlx-lm's one-step-lookahead pipeline). So
``Counter(job.tokens)`` is missing the token generated immediately before —
precisely the token a loop-breaking penalty most needs to see.

This test closes that gap by driving the real BatchGenerator round trip:

- FAILS when ``_make_sampler`` reads ``Counter(job.tokens)`` (one step stale:
  position 1 outputs 7 — an immediate repeat that escapes the penalty —
  instead of 3).
- PASSES when ``_make_sampler`` maintains a closure-local ``counts: Counter``
  updated the instant a token is sampled (no lag, matching serial
  ``generate_ar``).

Evaluate with ``git stash`` (reverts the openai.py fix, keeps this test) to
verify the FAIL/PASS flip.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from mtplx.generation import generate_ar
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.server import openai


VOCAB = 8


class _FixedLogitsModel:
    """Scripted model: constant logits regardless of input.

    Token 7 = 5.0, token 3 = 4.0, all others = 0.0. The 1.0 margin means a
    ``presence_penalty`` of 1.5 (one-off flat subtraction) flips the argmax
    the moment 7 has appeared in the completion: 5.0 - 1.5 = 3.5 < 4.0.
    """

    def make_cache(self):
        return []

    def __call__(self, input_ids, *, cache=None, **_kwargs):
        shape = input_ids.shape
        if len(shape) == 1:
            batch, seq = 1, int(shape[0])
        else:
            batch, seq = int(shape[0]), int(shape[1])
        logits = np.zeros((batch, seq, VOCAB), dtype=np.float32)
        logits[:, :, 7] = 5.0
        logits[:, :, 3] = 4.0
        return mx.array(logits)


class _NoopTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(t)) for t in tokens)


def _runtime(model) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=_NoopTokenizer(),
        model_path=Path("fixed-logits"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _make_job(sampler: SamplerConfig, *, max_tokens: int = 6) -> "openai._BatchedARJob":
    return openai._BatchedARJob(
        request_id="penalty-parity",
        prompt_ids=[0],
        max_tokens=max_tokens,
        sampler=sampler,
        seed=0,
        stop_token_ids=set(),
        token_callback=None,
        prefill_callback=None,
        request_observability=None,
        mtp_disabled_reason=None,
        generation_limits={},
        seed_is_explicit=True,
    )


def _run_ar_batch(job, model, sampler_fn, *, max_tokens: int = 6) -> list[int]:
    """Drive the real BatchGenerator with the pump's append-after-next() order.

    Mirrors ``_BatchedARGenerationService._pump``: insert with the per-job
    sampler, then loop calling ``generator.next()`` and emit each returned
    token AFTER next() returns (never before).
    """
    from mlx_lm.generate import BatchGenerator

    generator = BatchGenerator(
        model,
        max_tokens=max_tokens,
        stop_tokens=[],
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=1,
    )
    try:
        uids = generator.insert(
            [job.prompt_ids],
            max_tokens=[max_tokens],
            samplers=[sampler_fn],
        )
        job.uid = int(uids[0])
        while len(job.tokens) < max_tokens:
            _prompt_resps, gen_resps = generator.next()
            for resp in gen_resps:
                if int(resp.uid) == job.uid:
                    job.emit_token(int(resp.token))
    finally:
        generator.close()
    return list(job.tokens)


def test_ar_batch_penalty_parity_matches_serial():
    # mmmugh repro: token 7 leads token 3 by exactly 1.0; presence_penalty=1.5
    # must demote 7 the moment it has been emitted once, flipping to 3. The
    # expected sequence is [7, 3, 7, 7, 7, 7]: 7 wins the first step (no
    # penalty yet), 3 wins the second (7 just seen), then 7 wins again (both
    # seen, 3.5 > 2.5), and so on.
    cfg = SamplerConfig(
        temperature=0.0, top_p=1.0, top_k=0, presence_penalty=1.5
    )
    model = _FixedLogitsModel()

    # Serial reference (mtplx/generation.py:3945) — the correctness oracle.
    # generate_ar uses Counter(tokens) where tokens already includes the
    # immediately-preceding sample, so there is no staleness.
    ar = generate_ar(
        _runtime(model),
        [0],
        max_tokens=6,
        sampler=cfg,
        seed=0,
        stop_token_ids=set(),
    )
    assert list(ar.tokens) == [7, 3, 7, 7, 7, 7]

    # Real ar_batch round trip through BatchGenerator + append-after-next().
    job = _make_job(cfg)
    service = object.__new__(openai._BatchedARGenerationService)
    sampler_fn = service._make_sampler(job)
    batch_tokens = _run_ar_batch(job, model, sampler_fn, max_tokens=6)

    assert batch_tokens == [7, 3, 7, 7, 7, 7]
    # Per-position equality with serial generate_ar — the core parity claim.
    assert batch_tokens == list(ar.tokens)
