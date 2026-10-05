"""Context-copy for AR-only runtimes: drafter, acceptance, KV trim, end to end.

CPU only: a tiny stub model with a real (list-backed) KV cache drives
``generate_ar``. The stub's next token depends on the position and the last
token, so a wrong cache trim changes the stream.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.context_copy import CopyGovernor, context_copy_ar_enabled
from mtplx.context_copy_ar import (
    ArContextCopy,
    accepted_prefix_greedy,
    copy_ar_eligible,
    verify_copy_block,
)
from mtplx.generation import generate_ar
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig

VOCAB = 8
GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


@pytest.fixture(autouse=True)
def _cpu_and_env(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MTPLX_CONFIG", "/nonexistent")
    for name in (
        "MTPLX_CONTEXT_COPY",
        "MTPLX_CONTEXT_COPY_AR",
        "MTPLX_CONTEXT_COPY_K",
        "MTPLX_CONTEXT_COPY_PROBATION_K",
        "MTPLX_CONTEXT_COPY_NGMIN",
        "MTPLX_CONTEXT_COPY_NGMAX",
        "MTPLX_CONTEXT_COPY_MINEXT",
        "MTPLX_ASYNC_AR",
        "MTPLX_EVAL_AUDIT",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    mx.set_default_device(mx.gpu)


class ListCache:
    """Minimal trimmable KV cache: it stores the token ids it has seen."""

    def __init__(self):
        self.tokens: list[int] = []

    @property
    def offset(self) -> int:
        return len(self.tokens)

    def trim(self, n: int) -> int:
        n = min(int(n), len(self.tokens))
        del self.tokens[len(self.tokens) - n :]
        return n


class Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(t)) for t in tokens)


def stub_next(position: int, last: int) -> int:
    """Next token given the position of ``last`` and ``last`` itself."""
    return (last + 1 + (1 if position % 5 == 0 else 0)) % VOCAB


class DeterministicModel:
    """Greedy-peaked logits from stub_next; reads positions from the cache."""

    def __init__(self):
        self.row_calls: list[int] = []

    def make_cache(self):
        return [ListCache()]

    def __call__(self, input_ids, *, cache=None, **_kwargs):
        ids = [int(t) for t in np.asarray(input_ids).reshape(-1)]
        entry = cache[0]
        base = entry.offset
        entry.tokens.extend(ids)
        self.row_calls.append(len(ids))
        rows = []
        for j, tok in enumerate(ids):
            row = [0.0] * VOCAB
            row[stub_next(base + j, tok)] = 10.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)


class IidModel(DeterministicModel):
    """Context-free next-token distribution, to test sampled exactness."""

    LOGITS = (1.2, 0.3, 0.0, -0.5)

    def __call__(self, input_ids, *, cache=None, **_kwargs):
        ids = np.asarray(input_ids).reshape(-1)
        cache[0].tokens.extend(int(t) for t in ids)
        return mx.array([[list(self.LOGITS)] * len(ids)], dtype=mx.float32)


def runtime(model) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=Tokenizer(),
        model_path=Path("tiny-ar"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def trajectory(first: int, start_pos: int, n: int) -> list[int]:
    out = [first]
    for i in range(n - 1):
        out.append(stub_next(start_pos + i, out[-1]))
    return out


def run(prompt, max_tokens, *, stop=None, sampler=GREEDY, seed=0, model=None):
    model = model or DeterministicModel()
    out = generate_ar(
        runtime(model),
        list(prompt),
        max_tokens=max_tokens,
        sampler=sampler,
        seed=seed,
        stop_token_ids=stop or set(),
    )
    return out, model


def copy_summary(out):
    for event in out.stats.events:
        if "context_copy_ar" in event:
            return event["context_copy_ar"]
    return None


def enable(monkeypatch, ngmin="3", ngmax="5"):
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_AR", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_NGMIN", ngmin)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_NGMAX", ngmax)


# --- switches ------------------------------------------------------------


def test_ar_switch_is_off_by_default_and_respects_global_off(monkeypatch):
    assert context_copy_ar_enabled() is False
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_AR", "1")
    assert context_copy_ar_enabled() is True
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    assert context_copy_ar_enabled() is False


def test_eligibility_excludes_mtp_penalties_guards_async_and_untrimmable(monkeypatch):
    enable(monkeypatch)
    cache = [ListCache()]
    base = {
        "mtp_enabled": False,
        "penalized": False,
        "guarded": False,
        "sync_eval": True,
        "cache": cache,
    }
    assert copy_ar_eligible(**base)
    for key, value in (
        ("mtp_enabled", True),
        ("penalized", True),
        ("guarded", True),
        ("sync_eval", False),
        ("cache", []),
        ("cache", [object()]),
    ):
        assert not copy_ar_eligible(**{**base, key: value})


# --- governor (shared policy) ---------------------------------------------


def test_governor_probation_then_suspend_with_doubling_backoff():
    gov = CopyGovernor()
    assert gov.block_cap(24, 8) == 8
    gov.record(8, 8, 10)
    gov.record(8, 8, 20)
    assert gov.block_cap(24, 8) == 24
    for n in range(3):
        gov.record(0, 8, 100 + n)
    assert gov.suspensions == 1 and gov.suspended(150) and not gov.suspended(100 + 2 + 64)
    assert gov.backoff == 128 and gov.seen == 0 and gov.block_cap(24, 8) == 8


# --- drafter ---------------------------------------------------------------


def test_drafter_copies_prompt_continuation_and_never_generated_text(monkeypatch):
    enable(monkeypatch, "3", "4")
    prompt = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 9, 9]
    drafter = ArContextCopy(prompt)
    assert drafter.propose([8, 5, 6, 7], 10)[:4] == [0, 1, 2, 3]
    assert drafter.propose([8, 5, 6, 7], 2) == [0, 1]               # budget cap
    assert drafter.propose([8, 5, 6, 7], 0) == []
    assert drafter.propose([4, 5, 7, 7, 7], 10) == []               # no match
    # a gram that only recurs inside generated text is not a prompt match
    assert drafter.propose([9, 8, 7, 9, 8, 7, 9, 8, 7], 10) == []


def test_drafter_block_length_follows_ladder_and_probation(monkeypatch):
    enable(monkeypatch, "2", "2")
    prompt = list(range(40))
    drafter = ArContextCopy(prompt)
    assert len(drafter.propose([100, 10, 11], 50)) == 8             # ladder rung 0, probation
    drafter.governor.seen, drafter.governor.ema = 5, 0.9
    assert len(drafter.propose([100, 10, 11], 50)) == 8             # ext 0 -> still rung 0


# --- greedy acceptance + KV trim ------------------------------------------


def test_accepted_prefix_greedy():
    assert accepted_prefix_greedy([1, 2, 3], [1, 2, 9, 4]) == 2
    assert accepted_prefix_greedy([1, 2, 3], [1, 2, 3, 4]) == 3
    assert accepted_prefix_greedy([1], [2, 5]) == 0


def _verify(block, cache_tokens, *, stop=None):
    model = DeterministicModel()
    cache = [ListCache()]
    cache[0].tokens.extend(cache_tokens)
    token = stub_next(len(cache_tokens) - 1, cache_tokens[-1])
    result = verify_copy_block(
        lambda ids: model(ids, cache=cache),
        lambda keep: cache[0].trim(cache[0].offset - keep) >= 0,
        cache[0].offset,
        token,
        block,
        stop,
        greedy=True,
    )
    return result, cache[0], token


def test_verify_trims_cache_to_accepted_prefix_on_rejection():
    base = [3, 4, 5]
    token = stub_next(2, 5)
    good = trajectory(stub_next(3, token), 4, 2)          # two tokens the model agrees with
    block = [*good, (good[-1] + 4) % VOCAB, 1]            # then a wrong one
    result, cache, tok = _verify(block, base)
    assert tok == token
    assert result.tokens == good and result.correction is None and not result.stopped
    assert cache.tokens == [*base, token, *good]
    assert cache.offset == len(base) + 1 + len(good)


def test_verify_full_accept_keeps_whole_block_and_returns_next_logits():
    base = [3, 4, 5]
    token = stub_next(2, 5)
    block = trajectory(stub_next(3, token), 4, 4)
    result, cache, _ = _verify(block, base)
    assert result.tokens == block
    assert cache.offset == len(base) + 1 + len(block)
    nxt = int(mx.argmax(result.logits[0]).item())
    assert nxt == stub_next(len(base) + len(block), block[-1])


def test_verify_stop_inside_block_cuts_and_keeps_stop_out_of_cache():
    base = [3, 4, 5]
    token = stub_next(2, 5)
    block = trajectory(stub_next(3, token), 4, 5)
    result, cache, _ = _verify(block, base, stop={block[2]})
    assert result.stopped and result.tokens == block[:3]
    assert cache.offset == len(base) + 1 + 2              # token + 2 drafts, stop excluded


# --- end to end vs plain AR ------------------------------------------------


def _prompt_with_repeat():
    # the model's own trajectory twice, so the tail keeps matching the prompt
    first = trajectory(1, 0, 24)
    return first + first[:6]


def test_greedy_stream_matches_plain_ar_with_fewer_forwards(monkeypatch):
    prompt = _prompt_with_repeat()
    plain, plain_model = run(prompt, 40)
    enable(monkeypatch)
    copied, copy_model = run(prompt, 40)
    assert copied.tokens == plain.tokens
    summary = copy_summary(copied)
    assert summary["rounds"] > 0 and summary["accepted_tokens"] > 0
    assert len(copy_model.row_calls) < len(plain_model.row_calls)
    assert any(n > 1 for n in copy_model.row_calls[1:])


def test_greedy_with_diverging_prompt_still_matches_plain_ar(monkeypatch):
    base = trajectory(1, 0, 30)
    prompt = base[:12] + [(t + 3) % VOCAB for t in base[12:18]] + base[18:] + base[:5]
    plain, _ = run(prompt, 60)
    enable(monkeypatch, "2", "4")
    copied, _ = run(prompt, 60)
    assert copied.tokens == plain.tokens
    assert copy_summary(copied)["rounds"] > 0


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 7, 16])
def test_max_tokens_is_exact(monkeypatch, max_tokens):
    prompt = _prompt_with_repeat()
    plain, _ = run(prompt, max_tokens)
    enable(monkeypatch)
    copied, _ = run(prompt, max_tokens)
    assert len(copied.tokens) == max_tokens
    assert copied.tokens == plain.tokens
    assert copied.finish_reason == plain.finish_reason


def test_stop_token_inside_accepted_block_ends_stream_like_plain_ar(monkeypatch):
    prompt = _prompt_with_repeat()
    plain_full, _ = run(prompt, 40)
    stop = {plain_full.tokens[9]}
    plain, _ = run(prompt, 40, stop=stop)
    enable(monkeypatch)
    copied, _ = run(prompt, 40, stop=stop)
    assert copied.tokens == plain.tokens
    assert copied.tokens[-1] in stop
    assert copied.finish_reason == plain.finish_reason


def test_disabled_by_default_runs_plain_loop():
    prompt = _prompt_with_repeat()
    out, model = run(prompt, 20)
    assert copy_summary(out) is None
    assert all(n == 1 for n in model.row_calls[1:])


# --- sampled exactness ------------------------------------------------------


def _softmax(logits):
    z = np.exp(np.array(logits) - np.max(logits))
    return z / z.sum()


def _chi2(counts, probs):
    counts = np.asarray(counts, dtype=float)
    expected = probs * counts.sum()
    return float(((counts - expected) ** 2 / expected).sum())


def test_sampled_copy_blocks_follow_the_target_distribution(monkeypatch):
    """Copy rounds fire constantly (prompt over the 4 live tokens, 2-gram keys),
    yet the emitted stream must stay i.i.d. from the target distribution."""
    sampler = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0)
    probs = _softmax(IidModel.LOGITS)
    rng = np.random.default_rng(7)
    prompt = [int(t) for t in rng.integers(0, 4, size=64)]
    enable(monkeypatch, "2", "3")
    singles = np.zeros(4)
    pairs = np.zeros((4, 4))
    rounds = accepted = drafted = 0
    for seed in range(400):
        out, _ = run(prompt, 24, sampler=sampler, seed=seed, model=IidModel())
        toks = out.tokens
        assert len(toks) == 24
        for t in toks:
            singles[t] += 1
        for a in range(0, 24, 2):
            pairs[toks[a], toks[a + 1]] += 1
        summary = copy_summary(out)
        rounds += summary["rounds"]
        accepted += summary["accepted_tokens"]
        drafted += summary["drafted_tokens"]
    assert rounds > 400 and 0 < accepted < drafted        # blocks fired, some accepted, some rejected
    # 3 dof / 15 dof; thresholds at p = 0.001 (fixed seeds, so deterministic)
    assert _chi2(singles, probs) < 16.3
    assert _chi2(pairs.ravel(), np.outer(probs, probs).ravel()) < 37.7


def test_sampled_run_is_reproducible_for_a_seed(monkeypatch):
    sampler = SamplerConfig(temperature=1.0, top_p=0.9, top_k=3)
    prompt = [int(t) for t in np.random.default_rng(3).integers(0, 4, size=48)]
    enable(monkeypatch, "2", "3")
    a, _ = run(prompt, 20, sampler=sampler, seed=5, model=IidModel())
    b, _ = run(prompt, 20, sampler=sampler, seed=5, model=IidModel())
    assert a.tokens == b.tokens
