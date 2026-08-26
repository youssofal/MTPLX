"""CPU tests for the Phase-1 multi-stream batched greedy decoder.

These exercise the DRIVER's batching bookkeeping (prefill batching, per-stream
argmax, [B,2] verify, uniform full-B repair + rollback, per-stream termination)
against a tiny FAKE runtime — deterministic per-row logits over a 64-token vocab,
tiny MLX tensors, no model.  The fake is ROW-ISOLATED by construction, so any
per-stream sha divergence between a batched run and the same stream run alone is
a driver bug (cross-stream contamination), which is exactly the Phase-1
correctness contract.  The real-model per-stream sha gate is fable-main's GPU
window (``a3b_174_batched_decode_bench.py``); it validates batch-numerical
invariance, which a CPU fake cannot.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import astuple

import mlx.core as mx
import numpy as np
import pytest

import mtplx.batched_decode as bd
from mtplx.batched_decode import (
    BATCHED_DECODE_ENV,
    BATCHED_DECODE_REJECT_ENV,
    BATCHED_DECODE_SERIAL_ENV,
    FOLDIN_CLONE_SNAPSHOT_ENV,
    batched_decode_enabled,
    batched_decode_reject_mode,
    batched_decode_serial,
    diff_streams,
    foldin_clone_snapshot,
    generate_greedy_batched,
    left_pad_prompts,
    streams_all_match,
    token_sha,
)
from mtplx.sampling import (
    SamplerConfig,
    distribution_from_logits,
    sample_from_distribution,
    verify_one_token,
)

VOCAB = 64
HID = 4
STOP_ID = 63


def _reference_sampled_k1_cycle(
    primary_logits,
    draft_logits,
    verify_logits,
    bonus_logits,
    *,
    sampler,
    draft_sampler,
    rng,
    history_tokens,
    omit_speculative_bonus,
    pending_primary=None,
):
    counts = Counter(history_tokens)
    if pending_primary is None:
        primary_p = distribution_from_logits(
            primary_logits, sampler, token_counts=counts
        )
        primary = (
            int(np.argmax(primary_logits))
            if sampler.temperature <= 0
            else sample_from_distribution(primary_p, rng)
        )
        counts[primary] += 1
    else:
        primary = int(pending_primary)
    draft_q = distribution_from_logits(draft_logits, draft_sampler)
    draft = (
        int(np.argmax(draft_logits))
        if draft_sampler.temperature <= 0
        else sample_from_distribution(draft_q, rng)
    )
    target_p = distribution_from_logits(verify_logits, sampler, token_counts=counts)
    if sampler.temperature <= 0:
        accepted = draft == int(np.argmax(target_p))
        second = draft if accepted else int(np.argmax(target_p))
        accept_probability = 1.0 if accepted else 0.0
    else:
        decision = verify_one_token(target_p, draft_q, draft, rng)
        accepted = decision.accepted
        second = decision.token_id
        accept_probability = decision.accept_probability
    counts[second] += 1
    bonus = None
    if accepted and not omit_speculative_bonus and bonus_logits is not None:
        bonus_p = distribution_from_logits(bonus_logits, sampler, token_counts=counts)
        bonus = (
            int(np.argmax(bonus_p))
            if sampler.temperature <= 0
            else sample_from_distribution(bonus_p, rng)
        )
    next_primary = bonus if accepted else second
    return (
        primary,
        draft,
        accepted,
        second,
        bonus,
        accept_probability,
        next_primary,
    )


class _FakeTrunkEntry:
    """One non-trimmable cache entry holding per-row token histories.

    Non-trimmable so ``snapshot_untrimmable_cache`` clones ``state`` and
    ``rollback_after_verify`` restores it — faithfully modelling the GDN
    recurrent snapshot/restore the driver relies on for its full-B repair.
    """

    def __init__(self) -> None:
        self.histories: list[list[int]] | None = None
        self.prompt_len: list[int] | None = None

    def is_trimmable(self) -> bool:
        return False

    @property
    def state(self) -> list[list[int]] | None:
        return self.histories

    @state.setter
    def state(self, value: list[list[int]] | None) -> None:
        self.histories = value


class _FakeRuntime:
    """Deterministic, row-isolated stand-in for MTPLXRuntime.

    ``forward_ar`` returns one-hot logits whose per-row/per-position argmax is a
    deterministic ``next_token`` of that row's cumulative history — so greedy
    decode is a fixed pseudo-sequence per row.  ``draft_mtp`` returns the exact
    correct next token for rows whose identity is NOT in ``broken_rids`` (forcing
    accept) and a wrong token otherwise (forcing the repair path).  Because the
    output is greedy, it is draft-independent: batched and single-stream agree
    regardless of which rows draft badly — that invariance is under test.
    """

    def __init__(
        self,
        *,
        broken_rids: set[int] | None = None,
        stop_at: dict[int, int] | None = None,
        seed: int = 7,
    ) -> None:
        self.mtp_enabled = True
        self.broken_rids = set(broken_rids or set())
        self.stop_at = dict(stop_at or {})
        self.seed = int(seed)
        self._trunk_entry: _FakeTrunkEntry | None = None

    # -- cache factories ---------------------------------------------------
    def make_cache(self) -> list[_FakeTrunkEntry]:
        entry = _FakeTrunkEntry()
        self._trunk_entry = entry
        return [entry]

    def make_mtp_cache(self) -> object:
        return object()

    # -- deterministic token model ----------------------------------------
    def _next_token(self, hist: list[int], prompt_len: int) -> int:
        rid = hist[0]
        generated = len(hist) - int(prompt_len)
        limit = self.stop_at.get(rid)
        if limit is not None and generated >= int(limit):
            return STOP_ID
        pseudo = (
            rid * 1000003 + sum(hist) * 7 + len(hist) * 13 + self.seed
        ) % (VOCAB - 2)
        return pseudo + 1  # in [1, VOCAB-2]; never STOP_ID or 0

    @staticmethod
    def _onehot(token: int) -> list[float]:
        row = [0.0] * VOCAB
        row[int(token)] = 10.0
        return row

    # -- forwards ----------------------------------------------------------
    def forward_ar(self, input_ids, cache, return_hidden: bool = False, **_kw):
        rows = input_ids.tolist()
        batch = len(rows)
        length = len(rows[0])
        entry = cache[0]
        if entry.histories is None:
            entry.histories = [[] for _ in range(batch)]
            entry.prompt_len = [length for _ in range(batch)]  # prefill = prompt
        logits: list[list[list[float]]] = []
        hidden: list[list[list[float]]] = []
        for b in range(batch):
            hist = list(entry.histories[b])
            row_logits: list[list[float]] = []
            row_hidden: list[list[float]] = []
            for i in range(length):
                hist.append(int(rows[b][i]))
                nxt = self._next_token(hist, entry.prompt_len[b])
                row_logits.append(self._onehot(nxt))
                row_hidden.append([float(len(hist))] * HID)
            entry.histories[b] = hist
            logits.append(row_logits)
            hidden.append(row_hidden)
        log = mx.array(logits)
        if return_hidden:
            return log, mx.array(hidden)
        return log

    def draft_mtp(self, hidden, next_token_ids, mtp_cache=None, **_kw):
        assert self._trunk_entry is not None and self._trunk_entry.histories is not None
        x0_rows = next_token_ids.tolist()
        batch = len(x0_rows)
        out: list[list[list[float]]] = []
        for b in range(batch):
            hist = list(self._trunk_entry.histories[b])
            x0 = int(x0_rows[b][-1])
            correct = self._next_token(hist + [x0], self._trunk_entry.prompt_len[b])
            rid = hist[0]
            if rid in self.broken_rids:
                wrong = correct + 1 if correct + 1 != STOP_ID else correct + 2
                token = wrong % VOCAB
            else:
                token = correct
            out.append([self._onehot(token)])
        return mx.array(out)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _distinct_prompts(batch: int, length: int = 3) -> list[list[int]]:
    # prompt[0] is a UNIQUE row identity (rid); rest fill the shared length.
    return [[10 + b] + [1 + ((b + j) % 5) for j in range(length - 1)] for b in range(batch)]


def _reference_single_stream(
    prompts: list[list[int]], *, max_new_tokens: int, stop_token_ids=None, **rt_kwargs
) -> list[list[int]]:
    """Run each prompt ALONE (B=1) through the same driver — the sha reference."""
    out: list[list[int]] = []
    for prompt in prompts:
        rt = _FakeRuntime(**rt_kwargs)
        res = generate_greedy_batched(
            rt, [prompt], max_new_tokens=max_new_tokens, stop_token_ids=stop_token_ids
        )
        out.append(res.streams[0].tokens)
    return out


# --------------------------------------------------------------------------- #
# Correctness: batched per-stream sha == single-stream
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [2, 4, 8])
def test_all_accept_matches_single_stream(batch: int) -> None:
    prompts = _distinct_prompts(batch)
    rt = _FakeRuntime()  # no broken rows -> every cycle all-accepts
    res = generate_greedy_batched(rt, prompts, max_new_tokens=16)
    ref = _reference_single_stream(prompts, max_new_tokens=16)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.repair_cycles == 0  # perfect drafts -> speculative fast path only
    assert res.all_accept_cycles > 0
    assert res.batch_size == batch
    # 2 tokens per cycle, all-accept => 1 forward per cycle.
    assert res.forwards == res.cycles


@pytest.mark.parametrize("batch", [2, 4])
def test_all_broken_matches_single_stream(batch: int) -> None:
    prompts = _distinct_prompts(batch)
    broken = {p[0] for p in prompts}
    rt = _FakeRuntime(broken_rids=broken)  # every row drafts wrong -> repair every cycle
    res = generate_greedy_batched(rt, prompts, max_new_tokens=16)
    ref = _reference_single_stream(prompts, max_new_tokens=16, broken_rids=broken)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.all_accept_cycles == 0
    assert res.repair_cycles == res.cycles
    assert res.forwards == 2 * res.cycles  # verify + repair each cycle


def test_mixed_accept_reject_isolation() -> None:
    """The load-bearing test: an accepting stream batched next to a rejecting
    one (which forces the full-B repair) must stay byte-identical to running it
    alone (where it would take the all-accept fast path).  Proves the repair
    re-forward never perturbs an accepting neighbour."""
    prompts = _distinct_prompts(6)
    broken = {prompts[b][0] for b in (1, 3, 5)}  # half draft badly
    rt = _FakeRuntime(broken_rids=broken)
    res = generate_greedy_batched(rt, prompts, max_new_tokens=24)
    # Reference: each row alone keeps its OWN broken-ness (keyed by rid), so a
    # "perfect" row runs all-accept alone yet must equal its batched (repaired)
    # self.  Output is greedy => draft-independent => must match either way.
    ref = _reference_single_stream(prompts, max_new_tokens=24, broken_rids=broken)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.repair_cycles > 0  # at least one stream rejected each cycle


def test_per_stream_stop_termination() -> None:
    prompts = _distinct_prompts(4)
    # Each row stops after a DIFFERENT number of generated tokens.
    stop_at = {prompts[b][0]: 3 + 4 * b for b in range(4)}  # 3, 7, 11, 15
    rt = _FakeRuntime(stop_at=stop_at)
    res = generate_greedy_batched(
        rt, prompts, max_new_tokens=64, stop_token_ids={STOP_ID}
    )
    ref = _reference_single_stream(
        prompts, max_new_tokens=64, stop_token_ids={STOP_ID}, stop_at=stop_at
    )
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    # Every stream stopped (not length/cap), at its own scheduled point.
    for b in range(4):
        stream = res.streams[b]
        assert stream.finish_reason == "stop", stream
        assert stream.tokens[-1] == STOP_ID
        # length == scheduled generated count (+1 for the committed stop token).
        assert len(stream.tokens) == stop_at[prompts[b][0]] + 1
    # Streams have genuinely different lengths (ragged termination).
    assert len({len(s.tokens) for s in res.streams}) == 4


def test_odd_max_new_tokens_final_single_commit() -> None:
    prompts = _distinct_prompts(3)
    rt = _FakeRuntime()
    res = generate_greedy_batched(rt, prompts, max_new_tokens=7)  # odd
    ref = _reference_single_stream(prompts, max_new_tokens=7)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    for stream in res.streams:
        assert len(stream.tokens) == 7
        assert stream.finish_reason == "length"


def test_output_is_draft_independent() -> None:
    """use_mtp_draft False (self-draft, mostly rejects) yields the SAME greedy
    sequence as the real MTP draft — confirms the sequence is a pure function of
    the prompt, not of speculation."""
    prompts = _distinct_prompts(4)
    with_draft = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=16, use_mtp_draft=True
    )
    without_draft = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=16, use_mtp_draft=False
    )
    records = diff_streams(
        [s.tokens for s in with_draft.streams],
        [s.tokens for s in without_draft.streams],
    )
    assert streams_all_match(records), records


# --------------------------------------------------------------------------- #
# Build-1: fixed-shape cohort padding
# --------------------------------------------------------------------------- #
def test_cohort_padding_matches_explicit_dummies() -> None:
    """cohort_slots=N with M real prompts is EQUIVALENT to appending N-M explicit
    dummy prompts and slicing the first M — the internal padding is exactly a
    fixed pad-token prompt of the shared length, and dummy slots commit nothing.
    (Pure driver logic on the row-isolated fake runtime.)"""
    prompts = _distinct_prompts(3)
    prompt_len = len(prompts[0])
    pad_id = 0
    res_cohort = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=16, cohort_slots=8, pad_id=pad_id
    )
    dummy = [pad_id] * prompt_len
    explicit = prompts + [list(dummy) for _ in range(5)]
    res_explicit = generate_greedy_batched(_FakeRuntime(), explicit, max_new_tokens=16)

    # Results report REAL streams only.
    assert len(res_cohort.streams) == 3
    assert res_cohort.batch_size == 3
    assert [s.index for s in res_cohort.streams] == [0, 1, 2]
    records = diff_streams(
        [s.tokens for s in res_cohort.streams],
        [s.tokens for s in res_explicit.streams[:3]],
    )
    assert streams_all_match(records), records
    # Dummy slots contributed nothing to the reported / generated totals.
    assert res_cohort.generated_tokens == sum(
        len(s.tokens) for s in res_cohort.streams
    )
    assert res_cohort.meta["cohort_slots"] == 8
    assert res_cohort.meta["real_streams"] == 3


def test_cohort_slots_below_real_count_rejected() -> None:
    with pytest.raises(ValueError, match="cohort_slots"):
        generate_greedy_batched(
            _FakeRuntime(), _distinct_prompts(4), max_new_tokens=4, cohort_slots=2
        )


def test_cohort_fixed_shape_gate_reference() -> None:
    """The harness's fixed-shape gate, on the fake: real stream b in an N-slot
    cohort of real prompts equals prompt b ALONE in an N-slot cohort (1 real +
    N-1 dummies).  Only the OTHER rows' content differs — the row-isolated fake
    proves the driver never leaks it across slots (the real per-row-independence
    claim is the Metal test in ``test_moe_force_unsorted``)."""
    prompts = _distinct_prompts(8)
    slots = 8
    res = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=20, cohort_slots=slots
    )
    references: list[list[int]] = []
    for prompt in prompts:
        ref = generate_greedy_batched(
            _FakeRuntime(), [prompt], max_new_tokens=20, cohort_slots=slots
        )
        assert len(ref.streams) == 1  # only the one real stream is reported
        references.append(ref.streams[0].tokens)
    records = diff_streams([s.tokens for s in res.streams], references)
    assert streams_all_match(records), records


# --------------------------------------------------------------------------- #
# Build-1: pipelined single-sync loop == serial Phase-1 loop
# --------------------------------------------------------------------------- #
def _pattern_case(pattern: str):
    """(prompts, rt_kwargs, decode_kwargs) for each accept/reject/stop pattern."""
    if pattern == "all_accept":
        return _distinct_prompts(6), {}, {"max_new_tokens": 20}
    if pattern == "all_reject":
        prompts = _distinct_prompts(4)
        return prompts, {"broken_rids": {p[0] for p in prompts}}, {"max_new_tokens": 20}
    if pattern == "mixed":
        prompts = _distinct_prompts(6)
        return (
            prompts,
            {"broken_rids": {prompts[b][0] for b in (1, 3, 5)}},
            {"max_new_tokens": 24},
        )
    if pattern == "stop":
        prompts = _distinct_prompts(4)
        return (
            prompts,
            {"stop_at": {prompts[b][0]: 3 + 4 * b for b in range(4)}},
            {"max_new_tokens": 64, "stop_token_ids": {STOP_ID}},
        )
    if pattern == "length_odd":
        return _distinct_prompts(3), {}, {"max_new_tokens": 7}
    raise AssertionError(pattern)


@pytest.mark.parametrize(
    "pattern", ["all_accept", "all_reject", "mixed", "stop", "length_odd"]
)
def test_pipelined_loop_matches_serial(pattern: str) -> None:
    """The load-bearing regression: the pipelined (single-sync) loop and the
    serial Phase-1 loop commit the IDENTICAL tokens and finish reasons — the
    parallelization is a pure SCHEDULING change.  Repair/accept classification is
    identical too; only the cycle count may grow by the bounded deferred-commit
    lag."""
    prompts, rt_kwargs, decode_kwargs = _pattern_case(pattern)
    res_par = generate_greedy_batched(
        _FakeRuntime(**rt_kwargs), prompts, serial=False, **decode_kwargs
    )
    res_ser = generate_greedy_batched(
        _FakeRuntime(**rt_kwargs), prompts, serial=True, **decode_kwargs
    )
    assert [s.tokens for s in res_par.streams] == [s.tokens for s in res_ser.streams]
    assert [s.finish_reason for s in res_par.streams] == [
        s.finish_reason for s in res_ser.streams
    ]
    assert res_par.generated_tokens == res_ser.generated_tokens
    # Scheduling-only: the pipeline never commits FEWER cycles than serial and
    # over-runs by at most the bounded deferred-commit lag (<=2 garbage cycles).
    assert res_ser.cycles <= res_par.cycles <= res_ser.cycles + 2
    assert res_par.meta["loop"] == "pipelined_single_sync"
    assert res_ser.meta["loop"] == "serial"


@pytest.mark.parametrize("cohort_stop", [False, True])
def test_pipelined_matches_serial_in_cohort_mode(cohort_stop: bool) -> None:
    """Loops agree with dummy slots present (dummies masked from repair), for
    both a rejecting-stream pattern and RAGGED per-stream STOP — the latter
    exercises real streams finishing at different cycles WHILE dummies occupy the
    rest of the fixed-shape cohort."""
    prompts = _distinct_prompts(3)
    kw: dict = {"cohort_slots": 8}
    if cohort_stop:
        rt_kwargs = {"stop_at": {prompts[b][0]: 2 + 3 * b for b in range(3)}}  # 2,5,8
        kw.update(max_new_tokens=40, stop_token_ids={STOP_ID})
    else:
        rt_kwargs = {"broken_rids": {prompts[1][0]}}  # one rejecting real stream
        kw.update(max_new_tokens=18)
    res_par = generate_greedy_batched(_FakeRuntime(**rt_kwargs), prompts, serial=False, **kw)
    res_ser = generate_greedy_batched(_FakeRuntime(**rt_kwargs), prompts, serial=True, **kw)
    assert [s.tokens for s in res_par.streams] == [s.tokens for s in res_ser.streams]
    assert [s.finish_reason for s in res_par.streams] == [
        s.finish_reason for s in res_ser.streams
    ]
    if cohort_stop:
        # Every real stream stopped at its own scheduled point (ragged), inside a
        # fixed-shape cohort padded to 8 slots.
        assert all(s.finish_reason == "stop" for s in res_par.streams)
        assert len({len(s.tokens) for s in res_par.streams}) == 3


# --------------------------------------------------------------------------- #
# Build-1: single-sync accounting
# --------------------------------------------------------------------------- #
def test_pipelined_one_blocking_sync_per_cycle(monkeypatch) -> None:
    """The pipelined loop reads the GPU exactly ONCE per cycle: it calls the
    ``_eval_bundle`` seam once per cycle and nothing else blocks on the critical
    path.  The serial loop never touches that seam."""
    calls = {"n": 0}
    real_eval_bundle = bd._eval_bundle

    def _counting(bundle):
        calls["n"] += 1
        return real_eval_bundle(bundle)

    monkeypatch.setattr(bd, "_eval_bundle", _counting)

    res = generate_greedy_batched(
        _FakeRuntime(), _distinct_prompts(4), max_new_tokens=16, serial=False
    )
    assert calls["n"] == res.cycles  # exactly one bundle readback per cycle
    assert res.meta["syncs_per_cycle"] == 1

    calls["n"] = 0
    res_ser = generate_greedy_batched(
        _FakeRuntime(), _distinct_prompts(4), max_new_tokens=16, serial=True
    )
    assert calls["n"] == 0  # serial loop uses _argmax_ids, never the bundle seam
    assert res_ser.meta["syncs_per_cycle"] is None


def test_serial_env_selects_serial_loop(monkeypatch) -> None:
    assert batched_decode_serial({}) is False
    assert batched_decode_serial({BATCHED_DECODE_SERIAL_ENV: "1"}) is True
    monkeypatch.setenv(BATCHED_DECODE_SERIAL_ENV, "1")
    res = generate_greedy_batched(
        _FakeRuntime(), _distinct_prompts(3), max_new_tokens=8
    )
    assert res.meta["loop"] == "serial"  # env fallback, no explicit serial= arg


# --------------------------------------------------------------------------- #
# Guards / validation
# --------------------------------------------------------------------------- #
def test_ragged_prompts_rejected() -> None:
    rt = _FakeRuntime()
    with pytest.raises(ValueError, match="share a length"):
        generate_greedy_batched(rt, [[1, 2, 3], [4, 5]], max_new_tokens=4)


def test_requires_mtp_runtime() -> None:
    rt = _FakeRuntime()
    rt.mtp_enabled = False
    with pytest.raises(RuntimeError, match="MTP-enabled"):
        generate_greedy_batched(rt, [[1, 2]], max_new_tokens=4)


def test_bad_max_tokens_rejected() -> None:
    rt = _FakeRuntime()
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_greedy_batched(rt, [[1, 2]], max_new_tokens=0)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_token_sha_stable_and_sensitive() -> None:
    assert token_sha([1, 2, 3]) == token_sha([1, 2, 3])
    assert token_sha([1, 2, 3]) != token_sha([1, 2, 4])
    assert len(token_sha([1, 2, 3])) == 16


def test_sampled_k1_row_cycle_matches_reference_acceptance_and_rng_order():
    sampler = SamplerConfig(temperature=0.8, top_p=1.0, top_k=0)
    draft_sampler = SamplerConfig(temperature=0.7, top_p=1.0, top_k=0)
    primary_logits = np.array([1.5, 0.5, -0.25, 0.0])
    draft_logits = np.array([-0.2, 0.1, 1.2, 0.3])
    verify_logits = np.array([0.2, 1.1, -0.4, 0.7])
    bonus_logits = np.array([-0.1, 0.4, 0.2, 1.4])

    for seed in range(32):
        expected_rng = np.random.default_rng(seed)
        actual_rng = np.random.default_rng(seed)
        expected = _reference_sampled_k1_cycle(
            primary_logits,
            draft_logits,
            verify_logits,
            bonus_logits,
            sampler=sampler,
            draft_sampler=draft_sampler,
            rng=expected_rng,
            history_tokens=[1, 1, 3],
            omit_speculative_bonus=False,
        )
        actual = bd._sample_mtp_k1_row_cycle(
            primary_logits,
            draft_logits,
            verify_logits,
            bonus_logits,
            sampler=sampler,
            draft_sampler=draft_sampler,
            rng=actual_rng,
            history_tokens=[1, 1, 3],
            omit_speculative_bonus=False,
        )

        assert astuple(actual) == expected
        assert actual_rng.random() == expected_rng.random()


def test_sampled_k1_rows_keep_independent_rng_streams():
    sampler = SamplerConfig(temperature=0.9, top_p=0.95, top_k=3)
    draft_sampler = SamplerConfig(temperature=0.6, top_p=1.0, top_k=3)
    rows = [
        (
            np.array([0.1 * (i + 1), 0.7, -0.3, 0.2]),
            np.array([0.2, -0.1 * i, 0.9, 0.4]),
            np.array([0.8, 0.3, 0.1 * i, -0.2]),
            np.array([-0.4, 0.2, 0.6, 0.9 + 0.1 * i]),
        )
        for i in range(8)
    ]

    def run(order):
        rngs = [np.random.default_rng(100 + i) for i in range(8)]
        out = {}
        for i in order:
            out[i] = bd._sample_mtp_k1_row_cycle(
                *rows[i],
                sampler=sampler,
                draft_sampler=draft_sampler,
                rng=rngs[i],
                history_tokens=[i, i],
                omit_speculative_bonus=False,
            )
        return out, [rng.random() for rng in rngs]

    forward, forward_next = run(range(8))
    reverse, reverse_next = run(reversed(range(8)))

    assert forward == reverse
    assert forward_next == reverse_next


def test_sampled_k1_split_propose_and_finish_matches_combined_rng_order():
    sampler = SamplerConfig(temperature=0.85, top_p=0.9, top_k=4)
    draft_sampler = SamplerConfig(temperature=0.65, top_p=1.0, top_k=4)
    rows = (
        np.array([0.4, -0.1, 0.8, 0.2]),
        np.array([0.3, 0.7, -0.2, 0.1]),
        np.array([-0.1, 0.5, 0.2, 0.9]),
        np.array([0.6, 0.1, 0.4, -0.2]),
    )
    combined_rng = np.random.default_rng(44)
    split_rng = np.random.default_rng(44)

    combined = bd._sample_mtp_k1_row_cycle(
        *rows,
        sampler=sampler,
        draft_sampler=draft_sampler,
        rng=combined_rng,
        history_tokens=[3, 1],
        omit_speculative_bonus=False,
    )
    proposal = bd._sample_mtp_k1_row_proposal(
        rows[0],
        rows[1],
        sampler=sampler,
        draft_sampler=draft_sampler,
        rng=split_rng,
        history_tokens=[3, 1],
    )
    split = bd._finish_mtp_k1_row_cycle(
        proposal,
        rows[2],
        rows[3],
        sampler=sampler,
        rng=split_rng,
        history_tokens=[3, 1, proposal.primary_token],
        omit_speculative_bonus=False,
    )

    assert astuple(split) == astuple(combined)
    assert split_rng.random() == combined_rng.random()


def test_sampled_k1_bonus_policy_does_not_consume_bonus_rng_when_omitted():
    sampler = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0)
    draft_sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    logits = np.array([0.0, 0.0, 4.0])
    accepted_target = np.array([-2.0, -1.0, 4.0])
    bonus_logits = np.array([0.3, 0.2, 0.1])
    expected_rng = np.random.default_rng(123)
    actual_rng = np.random.default_rng(123)

    result = bd._sample_mtp_k1_row_cycle(
        logits,
        logits,
        accepted_target,
        bonus_logits,
        sampler=sampler,
        draft_sampler=draft_sampler,
        rng=actual_rng,
        history_tokens=[],
        omit_speculative_bonus=True,
    )
    _reference_sampled_k1_cycle(
        logits,
        logits,
        accepted_target,
        bonus_logits,
        sampler=sampler,
        draft_sampler=draft_sampler,
        rng=expected_rng,
        history_tokens=[],
        omit_speculative_bonus=True,
    )

    assert result.bonus_token is None
    assert actual_rng.random() == expected_rng.random()


def test_sampled_k1_pending_primary_skips_target_sample_rng_draw():
    sampler = SamplerConfig(temperature=0.8, top_p=1.0, top_k=0)
    draft_sampler = SamplerConfig(temperature=0.7, top_p=1.0, top_k=0)
    primary_logits = np.array([0.9, 0.2, -0.1])
    draft_logits = np.array([0.1, 0.8, 0.3])
    verify_logits = np.array([0.7, -0.2, 0.5])
    bonus_logits = np.array([0.2, 0.4, 0.6])
    expected_rng = np.random.default_rng(91)
    actual_rng = np.random.default_rng(91)

    expected = _reference_sampled_k1_cycle(
        primary_logits,
        draft_logits,
        verify_logits,
        bonus_logits,
        sampler=sampler,
        draft_sampler=draft_sampler,
        rng=expected_rng,
        history_tokens=[2, 1],
        omit_speculative_bonus=False,
        pending_primary=2,
    )
    actual = bd._sample_mtp_k1_row_cycle(
        primary_logits,
        draft_logits,
        verify_logits,
        bonus_logits,
        sampler=sampler,
        draft_sampler=draft_sampler,
        rng=actual_rng,
        history_tokens=[2, 1],
        omit_speculative_bonus=False,
        pending_primary=2,
    )

    assert astuple(actual) == expected
    assert actual_rng.random() == expected_rng.random()


def test_sampled_k1_pending_primary_is_not_double_counted_for_penalties(monkeypatch):
    sampler = SamplerConfig(
        temperature=0.8,
        top_p=1.0,
        top_k=0,
        presence_penalty=0.4,
        frequency_penalty=0.2,
    )
    observed = []
    original = bd.distribution_from_logits

    def recording_distribution(logits, config, *, token_counts=None):
        observed.append(None if token_counts is None else Counter(token_counts))
        return original(logits, config, token_counts=token_counts)

    monkeypatch.setattr(bd, "distribution_from_logits", recording_distribution)
    bd._sample_mtp_k1_row_cycle(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.3, 0.2, 0.1]),
        np.array([0.2, 0.4, 0.1]),
        None,
        sampler=sampler,
        draft_sampler=SamplerConfig(temperature=0.0),
        rng=np.random.default_rng(7),
        history_tokens=[2, 1],
        omit_speculative_bonus=True,
        pending_primary=2,
    )

    assert observed[-1] == Counter({2: 1, 1: 1})


def test_greedy_k1_sampling_skips_full_distribution_construction(monkeypatch):
    sampler = SamplerConfig(
        temperature=0.0,
        top_p=0.95,
        top_k=2,
        presence_penalty=0.4,
        frequency_penalty=0.2,
    )
    draft_sampler = SamplerConfig(temperature=0.0, top_p=0.95, top_k=2)
    rng = np.random.default_rng(123)
    expected_next_random = np.random.default_rng(123).random()

    def fail_distribution(*_args, **_kwargs):
        raise AssertionError("greedy sampling must not build a full distribution")

    monkeypatch.setattr(bd, "distribution_from_logits", fail_distribution)
    primary = bd._sample_mtp_k1_primary(
        np.array([0.0, 1.0, 0.9]),
        sampler=sampler,
        rng=rng,
        history_tokens=[1, 1],
    )
    proposal = bd._sample_mtp_k1_draft(
        primary,
        np.array([0.0, 0.2, 2.0]),
        draft_sampler=draft_sampler,
        rng=rng,
    )
    result = bd._finish_mtp_k1_row_cycle(
        proposal,
        np.array([0.0, 0.2, 2.0]),
        np.array([0.0, 3.0, 1.0]),
        sampler=sampler,
        rng=rng,
        history_tokens=[1, 1, primary],
        omit_speculative_bonus=False,
    )

    assert primary == 2
    assert proposal.draft_token == 2
    assert result.accepted is True
    assert result.second_token == 2
    assert result.bonus_token == 1
    assert rng.random() == expected_next_random


def test_left_pad_prompts() -> None:
    padded, lengths = left_pad_prompts([[5, 6, 7], [8], [9, 10]], pad_id=0)
    assert lengths == [3, 1, 2]
    assert padded == [[5, 6, 7], [0, 0, 8], [0, 9, 10]]
    assert len({len(p) for p in padded}) == 1


def test_diff_streams_localizes_divergence() -> None:
    records = diff_streams([[1, 2, 3, 4]], [[1, 2, 9, 4]])
    assert records[0]["match"] is False
    assert records[0]["first_divergence"] == 2
    assert records[0]["batched_window"] == [1, 2, 3, 4]
    assert not streams_all_match(records)
    good = diff_streams([[1, 2]], [[1, 2]])
    assert streams_all_match(good)


def test_diff_streams_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="stream count"):
        diff_streams([[1]], [[1], [2]])


def test_batched_decode_enabled_failclosed() -> None:
    assert batched_decode_enabled({}) is False
    assert batched_decode_enabled({BATCHED_DECODE_ENV: "0"}) is False
    assert batched_decode_enabled({BATCHED_DECODE_ENV: "nope"}) is False
    for truthy in ("1", "true", "YES", "On"):
        assert batched_decode_enabled({BATCHED_DECODE_ENV: truthy}) is True


# --------------------------------------------------------------------------- #
# R3: FOLD-IN REPLAY loop
# --------------------------------------------------------------------------- #
class _AltFakeRuntime(_FakeRuntime):
    """Fake whose ``alt_rids`` rows draft wrong on EVEN generated positions and
    right on odd ones -- an accept/miss ALTERNATION per row (so a row goes
    miss -> replay -> accept -> miss ...), exercising a mode change every cycle
    that the static ``broken_rids`` fake cannot.  Output stays greedy => the
    committed sequence is unchanged, only the accept pattern differs."""

    def __init__(self, *, alt_rids=None, **kw) -> None:
        super().__init__(**kw)
        self.alt_rids = set(alt_rids or set())

    def draft_mtp(self, hidden, next_token_ids, mtp_cache=None, **_kw):
        assert self._trunk_entry is not None and self._trunk_entry.histories is not None
        x0_rows = next_token_ids.tolist()
        out: list[list[list[float]]] = []
        for b in range(len(x0_rows)):
            hist = list(self._trunk_entry.histories[b])
            x0 = int(x0_rows[b][-1])
            correct = self._next_token(hist + [x0], self._trunk_entry.prompt_len[b])
            rid = hist[0]
            wrong = correct + 1 if correct + 1 != STOP_ID else correct + 2
            generated = len(hist) - int(self._trunk_entry.prompt_len[b])
            if rid in self.broken_rids or (rid in self.alt_rids and generated % 2 == 0):
                token = wrong % VOCAB
            else:
                token = correct
            out.append([self._onehot(token)])
        return mx.array(out)


def _foldin_vs_repair(rt_factory, prompts, **decode_kwargs) -> None:
    """THE gate (a): fold-in committed tokens + finish reasons are IDENTICAL to
    the Build-1 repair loop on the same fake runtime.  Greedy => the sequence is a
    pure function of the prompt, so the reject mechanism cannot change it."""
    res_fold = generate_greedy_batched(
        rt_factory(), prompts, reject_mode="foldin", **decode_kwargs
    )
    res_repair = generate_greedy_batched(
        rt_factory(), prompts, reject_mode="repair", **decode_kwargs
    )
    assert [s.tokens for s in res_fold.streams] == [s.tokens for s in res_repair.streams]
    assert [s.finish_reason for s in res_fold.streams] == [
        s.finish_reason for s in res_repair.streams
    ]
    assert res_fold.generated_tokens == res_repair.generated_tokens
    assert res_fold.meta["reject_mode"] == "foldin"
    assert res_fold.meta["loop"] == "foldin_replay_single_sync"
    return res_fold


def test_foldin_equiv_all_accept() -> None:
    prompts = _distinct_prompts(6)
    res = _foldin_vs_repair(lambda: _FakeRuntime(), prompts, max_new_tokens=20)
    assert res.replay_rows == 0  # perfect drafts -> never any miss/replay


def test_foldin_equiv_all_miss() -> None:
    # every row drafts wrong EVERY spec cycle => miss -> replay -> miss ... on the
    # same row (the consecutive-miss case).
    prompts = _distinct_prompts(4)
    broken = {p[0] for p in prompts}
    res = _foldin_vs_repair(
        lambda: _FakeRuntime(broken_rids=broken), prompts, max_new_tokens=20
    )
    assert res.replay_rows > 0
    assert res.repair_cycles == 0  # fold-in never repairs


def test_foldin_equiv_alternating() -> None:
    # per-row accept/miss ALTERNATION (miss -> replay -> accept -> miss ...).
    prompts = _distinct_prompts(6)
    alt = {prompts[b][0] for b in (0, 2, 4)}
    _foldin_vs_repair(
        lambda: _AltFakeRuntime(alt_rids=alt), prompts, max_new_tokens=24
    )


def test_foldin_equiv_heterogeneous() -> None:
    # per-row heterogeneous: some rows always miss, some alternate, some accept.
    prompts = _distinct_prompts(6)
    broken = {prompts[1][0], prompts[4][0]}
    alt = {prompts[2][0], prompts[5][0]}
    _foldin_vs_repair(
        lambda: _AltFakeRuntime(broken_rids=broken, alt_rids=alt),
        prompts,
        max_new_tokens=24,
    )


def test_foldin_equiv_miss_at_stop() -> None:
    # a chronically-missing row that also STOPS: the stop can fall on a miss's x0
    # (replay x1 dropped) or a replay's x1 -- both must match the repair loop.
    prompts = _distinct_prompts(4)
    broken = {p[0] for p in prompts}
    stop_at = {prompts[b][0]: 3 + 2 * b for b in range(4)}  # 3,5,7,9
    _foldin_vs_repair(
        lambda: _FakeRuntime(broken_rids=broken, stop_at=stop_at),
        prompts,
        max_new_tokens=40,
        stop_token_ids={STOP_ID},
    )


def test_foldin_equiv_length_cap_odd() -> None:
    # odd length cap on a missing row: the final committed token may be the x0 of a
    # miss (its deferred x1 must be dropped by the cap, exactly as in repair).
    prompts = _distinct_prompts(3)
    broken = {prompts[0][0]}
    for cap in (7, 8, 9):
        _foldin_vs_repair(
            lambda: _FakeRuntime(broken_rids=broken), prompts, max_new_tokens=cap
        )


def test_foldin_equiv_in_cohort_mode() -> None:
    # fold-in inside a fixed-shape cohort with dummy slots + ragged per-stream stop.
    prompts = _distinct_prompts(3)
    stop_at = {prompts[b][0]: 2 + 3 * b for b in range(3)}  # 2,5,8
    broken = {prompts[1][0]}  # one chronically-missing real stream
    res_fold = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken, stop_at=stop_at), prompts,
        reject_mode="foldin", cohort_slots=8, max_new_tokens=40,
        stop_token_ids={STOP_ID},
    )
    res_repair = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken, stop_at=stop_at), prompts,
        reject_mode="repair", cohort_slots=8, max_new_tokens=40,
        stop_token_ids={STOP_ID},
    )
    assert [s.tokens for s in res_fold.streams] == [s.tokens for s in res_repair.streams]
    assert len(res_fold.streams) == 3  # dummies not reported
    assert all(s.finish_reason == "stop" for s in res_fold.streams)


def test_foldin_matches_single_stream_reference() -> None:
    # fold-in batched stream b == prompt b decoded ALONE (the sha-gate contract).
    prompts = _distinct_prompts(6)
    broken = {prompts[b][0] for b in (1, 3, 5)}
    res = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, reject_mode="foldin",
        max_new_tokens=24,
    )
    ref = []
    for prompt in prompts:
        r = generate_greedy_batched(
            _FakeRuntime(broken_rids=broken), [prompt], reject_mode="foldin",
            max_new_tokens=24,
        )
        ref.append(r.streams[0].tokens)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records


# --------------------------------------------------------------------------- #
# Gate b: ragged per-row bookkeeping under divergence
# --------------------------------------------------------------------------- #
def test_foldin_ragged_bookkeeping_divergence() -> None:
    """A chronically-MISSING row (1 tok/cycle) beside an all-ACCEPT row (2/cycle):
    lengths diverge cycle-to-cycle yet both land exactly on their own greedy
    continuation with the right finish reason."""
    prompts = _distinct_prompts(2)
    broken = {prompts[0][0]}  # row 0 misses forever, row 1 always accepts
    res = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, reject_mode="foldin",
        max_new_tokens=16,
    )
    ref = _reference_single_stream(prompts, max_new_tokens=16, broken_rids=broken)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    for s in res.streams:
        assert len(s.tokens) == 16
        assert s.finish_reason == "length"
    # the missing row genuinely deferred corrections (replay_rows counts misses).
    assert res.replay_rows > 0


def test_foldin_ragged_bookkeeping_ragged_stop() -> None:
    prompts = _distinct_prompts(4)
    # different stop points AND some rows missing -> maximally ragged commit cadence.
    stop_at = {prompts[b][0]: 3 + 4 * b for b in range(4)}  # 3,7,11,15
    broken = {prompts[1][0], prompts[3][0]}
    res = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken, stop_at=stop_at), prompts,
        reject_mode="foldin", max_new_tokens=64, stop_token_ids={STOP_ID},
    )
    ref = _reference_single_stream(
        prompts, max_new_tokens=64, stop_token_ids={STOP_ID},
        broken_rids=broken, stop_at=stop_at,
    )
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    for b in range(4):
        s = res.streams[b]
        assert s.finish_reason == "stop"
        assert s.tokens[-1] == STOP_ID
        assert len(s.tokens) == stop_at[prompts[b][0]] + 1
    assert len({len(s.tokens) for s in res.streams}) == 4  # genuinely ragged


# --------------------------------------------------------------------------- #
# Gate c: single-sync accounting
# --------------------------------------------------------------------------- #
def test_foldin_one_blocking_sync_per_cycle(monkeypatch) -> None:
    """The fold-in loop reads the GPU exactly ONCE per cycle -- the ``_eval_bundle``
    seam -- regardless of the miss pattern (no repair sync, no cache-induced sync;
    the ragged offset ops are pure device where + the host capacity bound)."""
    calls = {"n": 0}
    real = bd._eval_bundle

    def _counting(bundle):
        calls["n"] += 1
        return real(bundle)

    monkeypatch.setattr(bd, "_eval_bundle", _counting)
    prompts = _distinct_prompts(4)
    broken = {prompts[1][0], prompts[3][0]}  # misses -> still one sync/cycle
    res = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, reject_mode="foldin",
        max_new_tokens=16,
    )
    assert calls["n"] == res.cycles  # exactly one bundle readback per cycle
    assert res.meta["syncs_per_cycle"] == 1


# --------------------------------------------------------------------------- #
# reject_mode selection + default fail-closed
# --------------------------------------------------------------------------- #
def test_foldin_reject_mode_env_and_arg() -> None:
    assert batched_decode_reject_mode({}) == "repair"
    assert batched_decode_reject_mode({BATCHED_DECODE_REJECT_ENV: "foldin"}) == "foldin"
    assert batched_decode_reject_mode({BATCHED_DECODE_REJECT_ENV: "FoldIn"}) == "foldin"
    assert batched_decode_reject_mode({BATCHED_DECODE_REJECT_ENV: "nonsense"}) == "repair"
    assert batched_decode_reject_mode({BATCHED_DECODE_REJECT_ENV: "repair"}) == "repair"


def test_foldin_default_is_repair_byte_identical() -> None:
    # DEFAULT (no arg, no env) is repair: byte-identical to Build-1, meta reflects it.
    prompts = _distinct_prompts(4)
    broken = {prompts[1][0]}
    res_default = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, max_new_tokens=16
    )
    res_repair = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, reject_mode="repair",
        max_new_tokens=16,
    )
    assert [s.tokens for s in res_default.streams] == [
        s.tokens for s in res_repair.streams
    ]
    assert res_default.meta["reject_mode"] == "repair"
    assert res_default.replay_rows == 0


def test_foldin_env_selects_foldin(monkeypatch) -> None:
    monkeypatch.setenv(BATCHED_DECODE_REJECT_ENV, "foldin")
    res = generate_greedy_batched(
        _FakeRuntime(), _distinct_prompts(3), max_new_tokens=8
    )
    assert res.meta["reject_mode"] == "foldin"
    assert res.meta["loop"] == "foldin_replay_single_sync"


def test_foldin_bad_reject_mode_rejected() -> None:
    with pytest.raises(ValueError, match="reject_mode"):
        generate_greedy_batched(
            _FakeRuntime(), _distinct_prompts(2), max_new_tokens=4,
            reject_mode="bogus",
        )


# --------------------------------------------------------------------------- #
# Ragged-BACKED integration: the loop actually drives a RaggedBatchKVCache (tiny
# Metal).  Validates item 1 (KV -> ragged conversion), the write-offset /
# offset-fix cadence (committed tokens land at the right KV positions, a REPLAY
# overwrites the stale draft slot), and the reserve()/make_mask key_len coordination
# -- none of which the pure recurrent fake exercises.
# --------------------------------------------------------------------------- #
class _RaggedFakeRuntime(_FakeRuntime):
    """Fake with a REAL trimmable KV entry (converted to RaggedBatchKVCache by the
    fold-in path) alongside the recurrent trunk.  The KV is written with token-
    valued K/V so committed tokens are recoverable from the buffer; the greedy
    logits come from the recurrent trunk exactly as the base fake, so tokens are
    identical.  Also fires the real ``create_attention_mask`` -> ``make_mask`` seam."""

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        trunk = _FakeTrunkEntry()
        self._trunk_entry = trunk
        cache = [KVCache(), trunk]
        self._last_cache = cache
        self.last_mask_shape = None
        self.last_kv_cap = None
        return cache

    def forward_ar(self, input_ids, cache, return_hidden: bool = False, **kw):
        from mlx_lm.models.base import create_attention_mask

        kv = cache[0]
        rows = input_ids.tolist()
        B, L = len(rows), len(rows[0])
        # the real seam: create_attention_mask delegates to the ragged make_mask
        # once the KV is on the ragged lane (during prefill kv is stock -> "causal").
        mask = create_attention_mask(mx.zeros((B, L, 1)), kv)
        self.last_mask_shape = (
            None if (mask is None or isinstance(mask, str)) else tuple(mask.shape)
        )
        self.last_kv_cap = None if getattr(kv, "keys", None) is None else int(kv.keys.shape[2])
        kvals = mx.array(rows, dtype=mx.float32).reshape(B, 1, L, 1)
        kv.update_and_fetch(kvals, kvals)  # advances at the loop-preset write_start
        # logits/hidden come from the recurrent trunk (cache[1]) -- base fake logic.
        return super().forward_ar(input_ids, [cache[1]], return_hidden=return_hidden, **kw)


def test_foldin_ragged_backed_kv_content_and_offsets() -> None:
    from mtplx.ragged_kv_cache import RaggedBatchKVCache

    prompts = _distinct_prompts(2)  # prompt_len 3
    prompt_len = len(prompts[0])
    broken = {prompts[1][0]}  # row 0 always accepts (2/cyc), row 1 always misses
    rt = _RaggedFakeRuntime(broken_rids=broken)
    res = generate_greedy_batched(
        rt, prompts, reject_mode="foldin", max_new_tokens=12
    )
    # 1. equivalence: identical committed tokens to the pure recurrent fold-in fake.
    ref = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, reject_mode="foldin",
        max_new_tokens=12,
    )
    assert [s.tokens for s in res.streams] == [s.tokens for s in ref.streams]

    # 2. item 1: the trimmable KV entry became the ragged lane.
    rg = rt._last_cache[0]
    assert isinstance(rg, RaggedBatchKVCache)

    # 3. cadence: committed tokens sit at their committed KV positions (a REPLAY
    #    overwrote the stale draft slot), and per-row offsets track committed length.
    for b, s in enumerate(res.streams):
        toks = s.tokens
        committed = rg.keys[b, 0, prompt_len : prompt_len + len(toks), 0].tolist()
        assert [int(round(x)) for x in committed] == toks, (b, committed, toks)
        assert int(rg.offsets[b].item()) >= prompt_len + len(toks)

    # 4. reserve()/make_mask coordination: the last decode mask's key_len equals the
    #    KV buffer capacity (no grow slipped between make_mask and update_and_fetch).
    assert rt.last_mask_shape is not None
    assert rt.last_mask_shape[:3] == (2, 1, 2)  # [B, 1, q=2, cap]
    assert rt.last_mask_shape[-1] == rt.last_kv_cap


def test_foldin_ragged_backed_matches_repair() -> None:
    # the ragged-backed fold-in still equals the repair loop (which runs on the
    # stock KV lane) -- the reject mechanism + the cache lane are both transparent.
    prompts = _distinct_prompts(4)
    broken = {prompts[1][0], prompts[2][0]}
    res_fold = generate_greedy_batched(
        _RaggedFakeRuntime(broken_rids=broken), prompts, reject_mode="foldin",
        max_new_tokens=18,
    )
    res_repair = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, reject_mode="repair",
        max_new_tokens=18,
    )
    assert [s.tokens for s in res_fold.streams] == [s.tokens for s in res_repair.streams]


# --------------------------------------------------------------------------- #
# FIX 1: conditional REPLAY restore (skip the whole-batch masked restore when no
# real row replays -- an all-False mask restore is a byte-identical no-op).
# --------------------------------------------------------------------------- #
def test_foldin_conditional_restore_skipped_when_no_replay(monkeypatch) -> None:
    import mtplx.cache_state as cs

    calls = {"n": 0}
    real = cs.restore_untrimmable_cache_masked

    def _counting(cache, snap, mask):
        calls["n"] += 1
        return real(cache, snap, mask)

    monkeypatch.setattr(cs, "restore_untrimmable_cache_masked", _counting)
    # all-accept: no row ever replays -> the restore is elided EVERY cycle.
    res = generate_greedy_batched(
        _FakeRuntime(), _distinct_prompts(4), reject_mode="foldin", max_new_tokens=16
    )
    assert res.replay_rows == 0
    assert calls["n"] == 0


def test_foldin_conditional_restore_runs_only_with_a_live_mask(monkeypatch) -> None:
    import mtplx.cache_state as cs

    calls = {"n": 0}
    real = cs.restore_untrimmable_cache_masked

    def _counting(cache, snap, mask):
        calls["n"] += 1
        # FIX 1 invariant: the restore is NEVER invoked with an all-False mask.
        assert any(mask), "restore called with an all-False (no-op) mask"
        return real(cache, snap, mask)

    monkeypatch.setattr(cs, "restore_untrimmable_cache_masked", _counting)
    prompts = _distinct_prompts(4)
    broken = {p[0] for p in prompts}  # every row misses -> genuine replays
    res_fold = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, reject_mode="foldin",
        max_new_tokens=16,
    )
    assert res_fold.replay_rows > 0
    assert calls["n"] > 0  # restore genuinely ran, only for live-mask cycles
    # still byte-identical to the repair loop despite the gated restore.
    res_repair = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), prompts, reject_mode="repair",
        max_new_tokens=16,
    )
    assert [s.tokens for s in res_fold.streams] == [s.tokens for s in res_repair.streams]


# --------------------------------------------------------------------------- #
# FIX 2: lazy zero-copy view recurrent snapshot (default) vs the eager clone
# fallback -- the fold-in loop commits the identical sequence either way.
# --------------------------------------------------------------------------- #
def test_foldin_clone_snapshot_env() -> None:
    assert foldin_clone_snapshot({}) is False  # default OFF -> lazy view
    assert foldin_clone_snapshot({FOLDIN_CLONE_SNAPSHOT_ENV: "1"}) is True
    assert foldin_clone_snapshot({FOLDIN_CLONE_SNAPSHOT_ENV: "true"}) is True
    assert foldin_clone_snapshot({FOLDIN_CLONE_SNAPSHOT_ENV: "off"}) is False
    assert foldin_clone_snapshot({FOLDIN_CLONE_SNAPSHOT_ENV: ""}) is False


def test_foldin_lazy_and_clone_snapshot_agree(monkeypatch) -> None:
    # The default lazy-view snapshot and the eager-clone fallback commit the
    # SAME streams (ragged-backed fake -> real mx.array KV + create_attention_mask).
    prompts = _distinct_prompts(4)
    broken = {prompts[1][0], prompts[3][0]}

    def _run():
        return generate_greedy_batched(
            _RaggedFakeRuntime(broken_rids=broken), prompts, reject_mode="foldin",
            max_new_tokens=18,
        )

    monkeypatch.delenv(FOLDIN_CLONE_SNAPSHOT_ENV, raising=False)
    res_lazy = _run()  # default: lazy zero-copy view
    monkeypatch.setenv(FOLDIN_CLONE_SNAPSHOT_ENV, "1")
    res_clone = _run()  # fallback: eager clone
    assert [s.tokens for s in res_lazy.streams] == [s.tokens for s in res_clone.streams]
    assert [s.finish_reason for s in res_lazy.streams] == [
        s.finish_reason for s in res_clone.streams
    ]


# --------------------------------------------------------------------------- #
# REFILL / continuous batching (fold-in + cohort mode only)
# --------------------------------------------------------------------------- #
def _refill_reference(
    prompts, *, cohort, max_new_tokens, stop_token_ids=None, **rt_kwargs
):
    """Each prompt alone through the REFILL machinery (admission-path entry)."""
    out = []
    for prompt in prompts:
        res = generate_greedy_batched(
            _FakeRuntime(**rt_kwargs),
            [prompt],
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            cohort_slots=cohort,
            reject_mode="foldin",
            refill_queue=[],
        )
        out.append(res.streams[0].tokens)
    return out


def test_refill_requires_foldin_and_cohort() -> None:
    prompts = _distinct_prompts(2)
    with pytest.raises(ValueError, match="foldin"):
        generate_greedy_batched(
            _FakeRuntime(), prompts, max_new_tokens=4, cohort_slots=4,
            reject_mode="repair", refill_queue=[],
        )
    with pytest.raises(ValueError, match="cohort"):
        generate_greedy_batched(
            _FakeRuntime(), prompts, max_new_tokens=4,
            reject_mode="foldin", refill_queue=[],
        )
    with pytest.raises(ValueError, match="at least one token"):
        generate_greedy_batched(
            _FakeRuntime(), prompts, max_new_tokens=4, cohort_slots=4,
            reject_mode="foldin", refill_queue=[[]],
        )


def test_a_refill_prompt_of_a_different_length_is_served_not_refused() -> None:
    """The `_admit_rows` padding trap, closed.

    A ragged refill queue used to be refused, with a message telling the caller
    to "left_pad the whole request set together" -- which is the ONE workaround
    that must not be used on this trunk, because `left_pad_prompts` folds its
    pads into the GDN recurrent state with no mask and fails silently. The
    refusal was loud; the remedy it named was not. Admission now groups joiners
    by length and prefills each group at its own length, so no pad token ever
    enters a joining row and there is nothing to left-pad.

    The joiners are three different lengths, none of them the cohort's, so no
    single shared width could have served them.
    """

    prompts = _distinct_prompts(2)
    cohort_len = len(prompts[0])
    short = [40, 7]
    long_one = [41, 6, 5, 4, 3]
    longer = [42, 2, 1, 5, 4, 3, 2]
    assert len({len(short), len(long_one), len(longer), cohort_len}) == 4

    res = generate_greedy_batched(
        _FakeRuntime(),
        prompts,
        max_new_tokens=4,
        cohort_slots=4,
        reject_mode="foldin",
        refill_queue=[short, long_one, longer],
    )
    assert len(res.streams) == 5, "one stream per REQUEST"
    solo = _reference_single_stream([short, long_one, longer], max_new_tokens=4)
    for offset, (joiner, expected) in enumerate(
        zip([short, long_one, longer], solo)
    ):
        stream = res.streams[2 + offset]
        assert stream.prompt_len == len(joiner), stream.index
        assert stream.tokens == expected, (
            f"joiner of length {len(joiner)} did not produce its own "
            "continuation; a pad prefix reached its recurrent state"
        )


def test_refill_reference_matches_plain_foldin_reference() -> None:
    # The admission-path entry (dummy prefill + zero-state admission) must
    # reproduce the plain fold-in cohort decode bitwise on the row-isolated fake.
    prompts = _distinct_prompts(4)
    plain = [
        generate_greedy_batched(
            _FakeRuntime(), [p], max_new_tokens=16, cohort_slots=4,
            reject_mode="foldin",
        ).streams[0].tokens
        for p in prompts
    ]
    via_admission = _refill_reference(prompts, cohort=4, max_new_tokens=16)
    assert plain == via_admission


@pytest.mark.parametrize("batch", [2, 4])
def test_refill_serves_queue_and_matches_references(batch: int) -> None:
    total = 3 * batch
    all_prompts = _distinct_prompts(total)
    cohort = max(4, batch)
    res = generate_greedy_batched(
        _FakeRuntime(), all_prompts[:batch], max_new_tokens=12,
        cohort_slots=cohort, reject_mode="foldin",
        refill_queue=all_prompts[batch:],
    )
    assert len(res.streams) == total
    ref = _refill_reference(all_prompts, cohort=cohort, max_new_tokens=12)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.meta["refill"] is True
    assert res.meta["requests"] == total
    assert res.meta["admission_passes"] >= 2  # initial + at least one refill
    assert res.repair_cycles == 0
    assert res.forwards == res.cycles  # admission passes are counted separately


def test_refill_with_replays_matches_references() -> None:
    total = 8
    all_prompts = _distinct_prompts(total)
    broken = {p[0] for p in all_prompts[::2]}  # alternating requests draft wrong
    res = generate_greedy_batched(
        _FakeRuntime(broken_rids=broken), all_prompts[:4], max_new_tokens=12,
        cohort_slots=4, reject_mode="foldin", refill_queue=all_prompts[4:],
    )
    ref = _refill_reference(
        all_prompts, cohort=4, max_new_tokens=12, broken_rids=broken
    )
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.replay_rows > 0  # staggered finishes exercise partial admission


def test_refill_with_stops_staggered_admission() -> None:
    total = 6
    all_prompts = _distinct_prompts(total)
    stop_at = {all_prompts[i][0]: 3 + i for i in range(total)}
    res = generate_greedy_batched(
        _FakeRuntime(stop_at=stop_at), all_prompts[:2], max_new_tokens=20,
        cohort_slots=2, reject_mode="foldin", refill_queue=all_prompts[2:],
        stop_token_ids={STOP_ID},
    )
    assert len(res.streams) == total
    assert all(s.finish_reason == "stop" for s in res.streams)
    ref = _refill_reference(
        all_prompts, cohort=2, max_new_tokens=20,
        stop_token_ids={STOP_ID}, stop_at=stop_at,
    )
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records


def test_refill_instance_invariance() -> None:
    # The same prompt admitted multiple times commits the same sequence.
    prompts = _distinct_prompts(2)
    queue = [list(prompts[0]), list(prompts[1]), list(prompts[0])]
    res = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=10, cohort_slots=2,
        reject_mode="foldin", refill_queue=queue,
    )
    seq = [s.tokens for s in res.streams]
    assert seq[0] == seq[2] == seq[4]
    assert seq[1] == seq[3]


# --------------------------------------------------------------------------- #
# AR row-packing mode ([B,1] plain batched decode, decode_mode="ar")
# --------------------------------------------------------------------------- #
def test_ar_mode_validation() -> None:
    with pytest.raises(ValueError, match="decode_mode"):
        generate_greedy_batched(
            _FakeRuntime(), _distinct_prompts(2), max_new_tokens=4,
            decode_mode="bogus",
        )


def test_ar_matches_single_stream_references() -> None:
    prompts = _distinct_prompts(4)
    res = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=16, cohort_slots=8,
        decode_mode="ar",
    )
    ref = [
        generate_greedy_batched(
            _FakeRuntime(), [p], max_new_tokens=16, cohort_slots=8,
            decode_mode="ar",
        ).streams[0].tokens
        for p in prompts
    ]
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    # exactly 1 token/stream/cycle: forwards == cycles, and cycles is
    # max_new + the pipeline's bounded lag.
    assert res.forwards == res.cycles
    assert 16 <= res.cycles <= 18
    assert res.meta["decode_mode"] == "ar"


def test_ar_matches_spec_committed_sequences() -> None:
    # Greedy is decode-strategy-invariant: AR and spec commit the same tokens.
    prompts = _distinct_prompts(4)
    ar = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=12, cohort_slots=4,
        decode_mode="ar",
    )
    spec = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=12, cohort_slots=4,
        reject_mode="foldin",
    )
    assert [s.tokens for s in ar.streams] == [s.tokens for s in spec.streams]


def test_ar_refill_serves_queue_and_matches_references() -> None:
    total = 9
    all_prompts = _distinct_prompts(total)
    res = generate_greedy_batched(
        _FakeRuntime(), all_prompts[:3], max_new_tokens=10, cohort_slots=4,
        decode_mode="ar", refill_queue=all_prompts[3:],
    )
    assert len(res.streams) == total
    ref = [
        generate_greedy_batched(
            _FakeRuntime(), [p], max_new_tokens=10, cohort_slots=4,
            decode_mode="ar", refill_queue=[],
        ).streams[0].tokens
        for p in all_prompts
    ]
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.meta["admission_passes"] >= 2
    assert res.meta["refill"] is True


def test_ar_refill_with_stops() -> None:
    total = 6
    all_prompts = _distinct_prompts(total)
    stop_at = {all_prompts[i][0]: 2 + i for i in range(total)}
    res = generate_greedy_batched(
        _FakeRuntime(stop_at=stop_at), all_prompts[:2], max_new_tokens=20,
        cohort_slots=2, decode_mode="ar", refill_queue=all_prompts[2:],
        stop_token_ids={STOP_ID},
    )
    assert len(res.streams) == total
    assert all(s.finish_reason == "stop" for s in res.streams)
