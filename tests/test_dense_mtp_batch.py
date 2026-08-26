"""CPU tests for the dense batched-MTP cohort driver.

These exercise ``generate_dense_mtp_batch``'s batching bookkeeping, chunked
prefill, depth-K chained drafting, the ``[B, K+1]`` verify, per-row
accept-length math, per-row ragged commit, the committed head-history cache's
rewind/re-append arithmetic, and per-stream termination, against a tiny
deterministic FAKE runtime, while routing the commit through the REAL
machinery: ``commit_captured_rows`` + ``RaggedBatchKVCache`` +
``OwnedRecurrentStateCache`` (via ``to_foldin_cache``). The fake encodes each
row's token history directly in its capture arrays and hidden states, so the
real per-row ``take_along_axis`` state selection is what decides every row's
post-commit history, a wrong per-row commit corrupts the row's future tokens
and fails the sha gate.

The fake is row-isolated by construction: any per-stream divergence between a
batched run and the same stream run alone is a driver bug (cross-stream
contamination). Real-model batch-numerical invariance is the GPU bench's job.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import numpy as np
import pytest

from mtplx.dense_mtp_batch import generate_dense_mtp_batch
from mtplx.ragged_kv_cache import RaggedBatchKVCache

VOCAB = 64
STOP_ID = 63
PAD = -1


def _next_token(hist: list[int], *, stop_after: int | None, prompt_len: int, seed: int) -> int:
    """Deterministic pseudo-sequence: next token from a row's full history."""
    rid = hist[0]
    if stop_after is not None and len(hist) - prompt_len >= stop_after:
        return STOP_ID
    pseudo = (rid * 1000003 + sum(hist) * 7 + len(hist) * 13 + seed) % (VOCAB - 2)
    return pseudo + 1  # in [1, VOCAB-2]; never STOP_ID or 0


def _pad_hist(hist: list[int], width: int) -> list[int]:
    return hist + [PAD] * (width - len(hist))


def _unpad(row: list[int]) -> list[int]:
    """Decode a padded history row back to the tokens it holds.

    Terminates on PAD *or* 0, and the 0 matters for continuous batching. A
    real admission pass resets a joining row by ZEROING its recurrent state --
    a zero state is a fresh row, which is exactly the semantics
    ``_zero_untrimmable_rows`` provides. This fake encodes history as token ids
    in that same state, so it has to agree that all-zeros means empty. Without
    this it read a zeroed row as thousands of real tokens and the admission
    forward blew past the history width.

    Safe because no real token id here is 0: prompts use ids >= 1 and 0 is the
    pad id, so a 0 can only ever mean absence.
    """

    out = []
    for tok in row:
        if tok == PAD or tok == 0:
            break
        out.append(int(tok))
    return out


def _onehot(token: int) -> list[float]:
    row = [0.0] * VOCAB
    row[int(token)] = 10.0
    return row


class _FakeScalarKV:
    """Minimal trimmable scalar KV entry; becomes a RaggedBatchKVCache."""

    def __init__(self) -> None:
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        self.offset = max(0, self.offset - int(n))
        return int(n)

    def append_rows(self, batch: int, q: int) -> None:
        pad = mx.zeros((batch, 1, q, 1), dtype=mx.float32)
        if self.keys is None:
            self.keys, self.values = pad, pad
        else:
            self.keys = mx.concatenate([self.keys, pad], axis=2)
            self.values = mx.concatenate([self.values, pad], axis=2)
        self.offset += int(q)


class _FakeRecurrentEntry:
    """Array-state recurrent entry; becomes an OwnedRecurrentStateCache."""

    def __init__(self) -> None:
        self._state: list = [None, None]

    @property
    def state(self) -> list:
        return self._state

    @state.setter
    def state(self, value) -> None:
        self._state = list(value)


class _FakeRuntime:
    """Deterministic, row-isolated stand-in for the dense MTPLX runtime.

    The authoritative per-row committed history lives in the recurrent cache's
    slot 1 as a PAD-padded ``int32[B, W]`` array, exactly the leaf
    ``commit_captured_rows`` rebinds from the capture's per-step ``states``, so
    the real per-row selection is what carries each row's history forward.
    ``hidden`` tensors are the same history encoding (float), which lets
    ``draft_mtp`` chain drafts from the driver-supplied hidden alone.

    ``break_depth_for(rid, hist)`` returns the draft depth (0-based) at which
    this row's draft goes wrong given its current history (None = all correct).
    It must depend only on the row's own identity/history so batched and alone
    runs see identical drafts (row isolation under test).

    The head-history cache is modelled shape-only: ``make_mtp_cache`` returns a
    convertible scalar KV entry and ``update_mtp_cache`` appends the right
    number of positions, the driver's rewind/keeps offset arithmetic then runs
    on the REAL RaggedBatchKVCache and is audited by the tests. The fake's
    drafts do not read the head cache (accept patterns are schedule-driven).
    """

    def __init__(
        self,
        *,
        width: int,
        break_depth_for: Callable[[int, list[int]], int | None] | None = None,
        stop_after: dict[int, int] | None = None,
        seed: int = 7,
        alt_gap_verify: float | None = None,
        alt_gap_draft: float | None = None,
    ) -> None:
        self.mtp_enabled = True
        self.width = int(width)
        self.break_depth_for = break_depth_for or (lambda rid, hist: None)
        self.stop_after = dict(stop_after or {})
        self.seed = int(seed)
        self.alt_gap_verify = alt_gap_verify
        self.alt_gap_draft = alt_gap_draft
        self.prompt_len = 0
        self._draft_calls = 0
        self._cache_list: list | None = None
        self._mtp_cache_list: list | None = None


    @staticmethod
    def alt_of(token: int) -> int:
        """The soft-logit alternative for a target token (never STOP/OOB)."""
        alt = token + 1
        return alt if alt not in (STOP_ID, VOCAB) else 1

    def _next(self, hist: list[int]) -> int:
        return _next_token(
            hist,
            stop_after=self.stop_after.get(hist[0]),
            prompt_len=int(self.prompt_len),
            seed=self.seed,
        )

    # -- cache factories ----------------------------------------------------
    def make_cache(self) -> list:
        self._cache_list = [_FakeRecurrentEntry(), _FakeScalarKV()]
        self.prompt_len = 0
        return self._cache_list

    def make_mtp_cache(self) -> list:
        self._mtp_cache_list = [_FakeScalarKV()]
        return self._mtp_cache_list

    # -- prefill (chunked: called once per chunk) ---------------------------
    def forward_ar(self, input_ids, cache, return_hidden=False, logits_keep=None, **_kw):
        rows = input_ids.tolist()
        batch = len(rows)
        length = len(rows[0])
        entry = cache[0]
        kv = cache[1]
        if entry.state[1] is None:
            hists: list[list[int]] = [[] for _ in range(batch)]
        else:
            hists = [_unpad([int(v) for v in r]) for r in np.array(entry.state[1]).tolist()]
        self._draft_calls = 0
        logits = np.zeros((batch, length, VOCAB), dtype=np.float32)
        hidden = np.full((batch, length, self.width), float(PAD), dtype=np.float32)
        for b in range(batch):
            hist = hists[b]
            for i in range(length):
                hist.append(int(rows[b][i]))
                hidden[b, i, : len(hist)] = hist
                tgt = self._next(hist)
                logits[b, i, tgt] = 10.0
                if self.alt_gap_verify is not None:
                    logits[b, i, self.alt_of(tgt)] = float(self.alt_gap_verify)
            hists[b] = hist
        self.prompt_len += length
        entry.state = [
            mx.zeros((batch, 1, 1), dtype=mx.float32),
            mx.array([_pad_hist(h, self.width) for h in hists], dtype=mx.int32),
        ]
        # The real runtime's forward_ar works on either cache lane. Before
        # continuous batching nothing called it AFTER the ragged conversion, so
        # this fake only modelled the pre-conversion scalar entry. An admission
        # pass prefills a joining row into the already-converted cache, so the
        # fake has to advance whichever lane it is handed.
        if isinstance(kv, RaggedBatchKVCache):
            kv.reserve(length)
            kv.update_and_fetch(
                mx.zeros((batch, 1, length, 1), dtype=mx.float32),
                mx.zeros((batch, 1, length, 1), dtype=mx.float32),
            )
        else:
            kv.append_rows(batch, length)
        log = mx.array(logits)
        hid = mx.array(hidden)
        if return_hidden:
            return log, hid
        return log

    # -- head-history append (seed + per-cycle committed re-append) ---------
    def update_mtp_cache(self, hidden_states, next_token_ids, mtp_cache=None, **_kw):
        q = int(next_token_ids.shape[1])
        batch = int(next_token_ids.shape[0])
        for entry in mtp_cache or []:
            if isinstance(entry, RaggedBatchKVCache):
                pad = mx.zeros((batch, 1, q, 1), dtype=mx.float32)
                entry.update_and_fetch(pad, pad)
            elif isinstance(entry, _FakeScalarKV):
                entry.append_rows(batch, q)
        return hidden_states

    # -- drafting -----------------------------------------------------------
    def draft_mtp(self, hidden_states, next_token_ids, mtp_cache=None, return_hidden=False, **_kw):
        hist_rows = [
            _unpad([int(round(v)) for v in row])
            for row in hidden_states[:, -1, :].tolist()
        ]
        toks = [int(row[-1]) for row in next_token_ids.tolist()]
        depth_idx = self._draft_calls
        self._draft_calls += 1
        logits_out = []
        hidden_out = []
        for hist, tok in zip(hist_rows, toks):
            new_hist = hist + [tok]
            correct = self._next(new_hist)
            rid = new_hist[0]
            brk = self.break_depth_for(rid, hist)
            if brk is not None and depth_idx >= int(brk):
                token = (correct + 1) if (correct + 1) not in (STOP_ID, VOCAB) else 1
            else:
                token = correct
            row_logits = _onehot(token)
            if self.alt_gap_draft is not None:
                row_logits = list(row_logits)
                row_logits[self.alt_of(token)] = float(self.alt_gap_draft)
            logits_out.append([row_logits])
            hidden_out.append([_pad_hist(new_hist, self.width)])
        logits = mx.array(logits_out)
        hidden = mx.array(hidden_out, dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits

    # -- verify with capture -------------------------------------------------
    def forward_ar_capture(
        self, input_ids, cache, return_hidden=False, hidden_variant=None, capture_backend=None, **_kw
    ):
        rows = input_ids.tolist()
        batch = len(rows)
        steps = len(rows[0])
        entry = cache[0]  # OwnedRecurrentStateCache after to_foldin_cache
        rc = cache[1]
        assert isinstance(rc, RaggedBatchKVCache), "driver must convert to ragged"
        hists = [_unpad([int(v) for v in row]) for row in np.array(entry[1]).tolist()]
        states = np.full((batch, steps, self.width), PAD, dtype=np.int32)
        v_logits = np.zeros((batch, steps, VOCAB), dtype=np.float32)
        for b in range(batch):
            hist = list(hists[b])
            for s in range(steps):
                hist.append(int(rows[b][s]))
                states[b, s, : len(hist)] = hist
                tgt = self._next(hist)
                v_logits[b, s, tgt] = 10.0
                if self.alt_gap_verify is not None:
                    v_logits[b, s, self.alt_of(tgt)] = float(self.alt_gap_verify)
        states_mx = mx.array(states)
        captures = {
            0: {
                "conv_states": mx.zeros((batch, steps, 1, 1), dtype=mx.float32),
                "states": states_mx,
            }
        }
        # Speculative cache advance, as the real capture forward does.
        entry[0] = mx.zeros((batch, 1, 1), dtype=mx.float32)
        entry[1] = states_mx[:, -1, :]
        rc.update_and_fetch(
            mx.zeros((batch, 1, steps, 1), dtype=mx.float32),
            mx.zeros((batch, 1, steps, 1), dtype=mx.float32),
        )
        self._draft_calls = 0
        v_logits_mx = mx.array(v_logits)
        v_hidden_mx = states_mx.astype(mx.float32)
        if return_hidden:
            return v_logits_mx, v_hidden_mx, captures
        return v_logits_mx, captures


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _distinct_prompts(batch: int, length: int = 3) -> list[list[int]]:
    return [
        [10 + b] + [1 + ((b + j) % 5) for j in range(length - 1)] for b in range(batch)
    ]


def _expected_chain(prompt: list[int], n: int, *, stop_after=None, seed: int = 7) -> list[int]:
    """The pure greedy continuation the deterministic fake defines."""
    hist = list(prompt)
    out: list[int] = []
    for _ in range(n):
        tok = _next_token(
            hist, stop_after=stop_after, prompt_len=len(prompt), seed=seed
        )
        out.append(tok)
        hist.append(tok)
        if tok == STOP_ID:
            break
    return out


def _run(prompts, *, depth, max_new, width=256, cohort_slots=None,
         head_history="committed", loop_mode="pipelined", **rt_kwargs):
    rt = _FakeRuntime(width=width, **rt_kwargs)
    res = generate_dense_mtp_batch(
        rt,
        prompts,
        max_new_tokens=max_new,
        depth=depth,
        stop_token_ids={STOP_ID},
        cohort_slots=cohort_slots,
        head_history=head_history,
        loop_mode=loop_mode,
    )
    return res, rt


# --------------------------------------------------------------------------- #
# Correctness: committed tokens match the deterministic chain, batched == alone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("loop_mode", ["pipelined", "serial"])
@pytest.mark.parametrize("head_history", ["cycle", "committed"])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("batch", [2, 4])
def test_all_accept_matches_expected_chain(batch: int, depth: int, head_history: str, loop_mode: str) -> None:
    prompts = _distinct_prompts(batch)
    res, _ = _run(prompts, depth=depth, max_new=16, head_history=head_history, loop_mode=loop_mode)
    for b, stream in enumerate(res.streams):
        assert stream.tokens == _expected_chain(prompts[b], 16), (b, stream.tokens)
    assert res.accepted_by_depth == res.drafted_by_depth
    assert res.cycles <= (16 + depth) // (depth + 1) + 1


@pytest.mark.parametrize("head_history", ["cycle", "committed"])
@pytest.mark.parametrize("depth", [2, 3])
def test_mixed_accept_rows_match_chain_and_counters(depth: int, head_history: str) -> None:
    # Row rids are 10+b: row 0 rejects at depth 0, row 1 at depth 1, row 2
    # fully accepts, row 3 at depth 0.
    break_map = {10: 0, 11: 1, 12: None, 13: 0}
    prompts = _distinct_prompts(4)
    res, _ = _run(
        prompts,
        depth=depth,
        max_new=15,
        head_history=head_history,
        break_depth_for=lambda rid, hist: break_map[rid],
    )
    for b, stream in enumerate(res.streams):
        assert stream.tokens == _expected_chain(prompts[b], 15), (b, stream.tokens)
    assert res.accepted_by_depth[0] > 0
    if depth >= 2:
        assert res.accepted_by_depth[1] <= res.accepted_by_depth[0]


@pytest.mark.parametrize("loop_mode", ["pipelined", "serial"])
@pytest.mark.parametrize("head_history", ["cycle", "committed"])
def test_history_dependent_accept_pattern_matches_alone(head_history: str, loop_mode: str) -> None:
    depth = 3

    def break_fn(rid: int, hist: list[int]) -> int | None:
        h = (rid * 31 + sum(hist[-3:]) * 5 + len(hist)) % 5
        return None if h >= depth else h

    prompts = _distinct_prompts(4)
    batched, _ = _run(
        prompts, depth=depth, max_new=24, head_history=head_history,
        loop_mode=loop_mode, break_depth_for=break_fn,
    )
    for b, prompt in enumerate(prompts):
        alone, _ = _run(
            [prompt], depth=depth, max_new=24, head_history=head_history,
            loop_mode=loop_mode, break_depth_for=break_fn,
        )
        assert batched.streams[b].tokens == alone.streams[0].tokens, b
        assert batched.streams[b].sha == alone.streams[0].sha


def test_stop_token_isolates_rows() -> None:
    prompts = _distinct_prompts(3)
    res, _ = _run(prompts, depth=3, max_new=20, stop_after={11: 5})
    assert res.streams[1].tokens[-1] == STOP_ID
    assert res.streams[1].finish_reason == "stop"
    assert len(res.streams[1].tokens) == 6  # 5 pseudo + STOP
    for b in (0, 2):
        assert res.streams[b].tokens == _expected_chain(prompts[b], 20)
        assert res.streams[b].finish_reason == "length"


def test_cohort_dummy_slots_do_not_perturb_real_rows() -> None:
    prompts = _distinct_prompts(2)
    res_padded, _ = _run(prompts, depth=3, max_new=16, cohort_slots=6)
    res_bare, _ = _run(prompts, depth=3, max_new=16)
    assert res_padded.shas == res_bare.shas
    assert res_padded.batch_size == 2


# --------------------------------------------------------------------------- #
# Cache-state audits
# --------------------------------------------------------------------------- #
def test_head_history_offsets_track_committed_lengths() -> None:
    # After a committed-mode run, every row's head-cache logical length must be
    # seed + its committed cache length delta: head_offset - seed ==
    # trunk_hist_len - prompt_len (both advance by exactly `keeps` per cycle).
    prompts = _distinct_prompts(3, length=5)
    break_map = {10: 0, 11: 1, 12: None}
    res, rt = _run(
        prompts,
        depth=3,
        max_new=12,
        head_history="committed",
        break_depth_for=lambda rid, hist: break_map[rid],
    )
    for b, stream in enumerate(res.streams):
        assert stream.tokens == _expected_chain(prompts[b], 12)
    head = rt._mtp_cache_list[0]
    assert isinstance(head, RaggedBatchKVCache)
    head_offsets = [int(v) for v in np.array(head.offsets).tolist()]
    seed = min(len(prompts[0]) - 1, 8192)
    gdn_hist = [
        len(_unpad([int(v) for v in row]))
        for row in np.array(rt._cache_list[0][1]).tolist()
    ]
    trunk_ragged = rt._cache_list[1]
    trunk_offsets = [int(v) for v in np.array(trunk_ragged.offsets).tolist()]
    for b in range(len(prompts)):
        committed = gdn_hist[b] - len(prompts[0])
        assert head_offsets[b] - seed == committed, (b, head_offsets, gdn_hist)
        assert trunk_offsets[b] == gdn_hist[b], (b, trunk_offsets, gdn_hist)


@pytest.mark.parametrize("chunk", [3, 4, 9])
def test_chunked_prefill_matches_expected_chain(chunk: int) -> None:
    # A prompt split across prefill chunks must produce the same streams as
    # the deterministic chain (the fake accumulates history across calls),
    # and the committed head seed must survive the chunked hidden tail.
    prompts = _distinct_prompts(2, length=9)
    rt = _FakeRuntime(width=256)
    res = generate_dense_mtp_batch(
        rt,
        prompts,
        max_new_tokens=10,
        depth=2,
        stop_token_ids={STOP_ID},
        prefill_chunk=chunk,
    )
    for b, stream in enumerate(res.streams):
        assert stream.tokens == _expected_chain(prompts[b], 10)


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #
def test_tape_backend_refused() -> None:
    rt = _FakeRuntime(width=64)
    with pytest.raises(ValueError, match="per-step"):
        generate_dense_mtp_batch(
            rt,
            _distinct_prompts(2),
            max_new_tokens=4,
            capture_backend="linear-gdn-from-conv-tape",
        )


def test_unequal_prompts_refused() -> None:
    rt = _FakeRuntime(width=64)
    with pytest.raises(ValueError, match="share a length"):
        generate_dense_mtp_batch(
            rt, [[1, 2, 3], [1, 2]], max_new_tokens=4
        )


def test_bad_head_history_refused() -> None:
    rt = _FakeRuntime(width=64)
    with pytest.raises(ValueError, match="head_history"):
        generate_dense_mtp_batch(
            rt, _distinct_prompts(2), max_new_tokens=4, head_history="bogus"
        )


def test_serial_and_pipelined_commit_identical_streams() -> None:
    # The pipelined loop is a pure scheduling change: same committed sequence.
    def break_fn(rid: int, hist: list[int]) -> int | None:
        h = (rid * 17 + sum(hist[-2:]) * 3 + len(hist)) % 4
        return None if h >= 3 else h

    prompts = _distinct_prompts(4)
    a, _ = _run(prompts, depth=3, max_new=20, loop_mode="serial", break_depth_for=break_fn)
    b, _ = _run(prompts, depth=3, max_new=20, loop_mode="pipelined", break_depth_for=break_fn)
    assert a.shas == b.shas
    assert [s.tokens for s in a.streams] == [s.tokens for s in b.streams]


# --------------------------------------------------------------------------- #
# Exact speculative sampling (temperature > 0)
# --------------------------------------------------------------------------- #
def _sampled_result(seed: int, *, max_new: int = 200, gap_v=8.7, gap_d=7.0):
    rt = _FakeRuntime(width=1024, alt_gap_verify=gap_v, alt_gap_draft=gap_d)
    return rt, generate_dense_mtp_batch(
        rt,
        _distinct_prompts(2),
        max_new_tokens=max_new,
        depth=3,
        stop_token_ids={STOP_ID},
        capture_backend="stock",
        head_history="committed",
        history_window=64,
        loop_mode="pipelined",
        temperature=1.0,
        sampling_seed=seed,
    )


def test_sampling_deterministic_under_seed() -> None:
    _, r1 = _sampled_result(11, max_new=60)
    _, r2 = _sampled_result(11, max_new=60)
    _, r3 = _sampled_result(12, max_new=60)
    assert [st.tokens for st in r1.streams] == [st.tokens for st in r2.streams]
    assert [st.tokens for st in r1.streams] != [st.tokens for st in r3.streams]  # collision astronomically unlikely


def test_sampling_rejects_non_eager_core() -> None:
    rt = _FakeRuntime(width=1024)
    with pytest.raises(ValueError):
        generate_dense_mtp_batch(
            rt,
            _distinct_prompts(1),
            max_new_tokens=4,
            depth=2,
            temperature=0.7,
            draft_core="compiled",
        )


def test_sampling_marginal_matches_verify_distribution() -> None:
    """The merge-gate exactness test: committed tokens must follow the VERIFY
    distribution p, not the draft distribution q. The fake gives the target
    logit 10 everywhere, the alternative logit 8.7 under p but only 7.0 under
    q, so p(alt | {target, alt}) = 1/(1+e^1.3) = 0.214 while
    q(alt | {target, alt}) = 1/(1+e^3) = 0.047. A lenient or draft-biased
    acceptance implementation lands near 0.047; exact p/q acceptance with
    residual resampling lands near 0.214."""
    import math

    alt_count = 0
    considered = 0
    for seed in (3, 5):
        rt, res = _sampled_result(seed, max_new=300)
        for stream, prompt in zip(res.streams, _distinct_prompts(2)):
            hist = list(prompt)
            for tok in stream.tokens:
                tgt = rt._next(hist)
                alt = rt.alt_of(tgt)
                if tok == alt:
                    alt_count += 1
                    considered += 1
                elif tok == tgt:
                    considered += 1
                hist.append(tok)
    assert considered > 800, f"too few committed tokens ({considered})"
    p_alt = 1.0 / (1.0 + math.exp(10.0 - 8.7))
    freq = alt_count / considered
    sigma = math.sqrt(p_alt * (1 - p_alt) / considered)
    assert abs(freq - p_alt) < 5 * sigma, (
        f"committed alt-fraction {freq:.4f} deviates from p {p_alt:.4f} "
        f"(5-sigma {5*sigma:.4f}); draft-side q would give 0.047"
    )


# --------------------------------------------------------------------------- #
# Serving additions (T-204 item 1): per-row token caps and the commit callback
# --------------------------------------------------------------------------- #
def test_per_row_caps_stop_each_row_at_its_own_limit() -> None:
    """A cohort mixing a short request with a long one must not over-generate.

    Concurrent callers ask for different ``max_tokens``. Without per-row caps a
    serving lane would have to either run every row to the cohort maximum and
    throw the surplus away, or refuse to batch requests whose limits differ.
    """

    prompts = _distinct_prompts(4)
    caps = [2, 5, 9, 3]
    rt = _FakeRuntime(width=256)
    res = generate_dense_mtp_batch(
        rt,
        prompts,
        max_new_tokens=max(caps),
        max_new_tokens_per_row=caps,
        depth=3,
        stop_token_ids={STOP_ID},
    )
    for stream, cap, prompt in zip(res.streams, caps, prompts):
        assert len(stream.tokens) == cap, (stream.index, len(stream.tokens), cap)
        assert stream.finish_reason == "length"
        # The cap truncates; it must not change WHICH tokens are produced.
        assert stream.tokens == _expected_chain(prompt, cap)


def test_per_row_caps_default_to_the_cohort_bound() -> None:
    """Omitting the caps must be byte-identical to the shipped behaviour."""

    prompts = _distinct_prompts(3)
    plain, _ = _run(prompts, depth=3, max_new=6)
    rt = _FakeRuntime(width=256)
    capped = generate_dense_mtp_batch(
        rt,
        prompts,
        max_new_tokens=6,
        max_new_tokens_per_row=[6, 6, 6],
        depth=3,
        stop_token_ids={STOP_ID},
    )
    assert [s.sha for s in capped.streams] == [s.sha for s in plain.streams]


@pytest.mark.parametrize(
    "caps, message",
    [
        ([1, 2], "one entry per prompt"),
        ([1, 0, 2], "must be >= 1"),
        ([1, 2, 99], "must not exceed max_new_tokens"),
    ],
)
def test_per_row_caps_reject_malformed_input(caps, message) -> None:
    rt = _FakeRuntime(width=256)
    with pytest.raises(ValueError, match=message):
        generate_dense_mtp_batch(
            rt,
            _distinct_prompts(3),
            max_new_tokens=8,
            max_new_tokens_per_row=caps,
            depth=3,
        )


def test_on_commit_sees_exactly_the_committed_tokens_in_order() -> None:
    """The callback is the serving lane's only token path, so it must be exact.

    Two properties matter and both are checked: every committed token reaches
    the callback in commit order, and NOTHING else does. The pipelined loop can
    overshoot a finished row by a bounded amount, and if that overshoot reached
    a client the caller would receive tokens the model never committed.
    """

    prompts = _distinct_prompts(4)
    seen: list[tuple[int, int]] = []
    rt = _FakeRuntime(width=256)
    res = generate_dense_mtp_batch(
        rt,
        prompts,
        max_new_tokens=7,
        max_new_tokens_per_row=[7, 3, 5, 7],
        depth=3,
        stop_token_ids={STOP_ID},
        on_commit=lambda row, token: seen.append((row, token)),
    )
    for stream in res.streams:
        emitted = [tok for row, tok in seen if row == stream.index]
        assert emitted == stream.tokens, stream.index


def test_on_commit_default_none_leaves_results_unchanged() -> None:
    prompts = _distinct_prompts(3)
    with_cb, _ = _run(prompts, depth=2, max_new=5)
    rt = _FakeRuntime(width=256)
    without = generate_dense_mtp_batch(
        rt,
        prompts,
        max_new_tokens=5,
        depth=2,
        stop_token_ids={STOP_ID},
        on_commit=lambda row, token: None,
    )
    assert [s.sha for s in without.streams] == [s.sha for s in with_cb.streams]


# --------------------------------------------------------------------------- #
# Ragged prompt lengths (T-204 item 2): per-length-group prefill, pinned offsets
# --------------------------------------------------------------------------- #
def _ragged_prompts() -> list[list[int]]:
    """Four prompts with four different true lengths."""
    return [
        [10, 1, 2],
        [11, 2, 3, 4, 5],
        [12, 3],
        [13, 4, 5, 6],
    ]


def test_ragged_cohort_matches_each_row_decoded_alone() -> None:
    """Grouping preserves per-row semantics; padding does not.

    SCOPE, because this test is easy to over-read and once was. The fake's
    arithmetic is exact and deterministic, so equality here means the ASSEMBLY
    LOGIC is right: rows are prefilled at their true lengths, offsets are pinned
    per row, and the permutation back to caller order is correct. A mismatch
    would be a real bug.

    It does NOT establish that a row's output in a real cohort equals that row
    decoded alone. On real weights it does not, at any geometry, because float
    reduction order changes with matmul shape — measured directly, see the
    correctness contract in ``mtplx/dense_mtp_batch.py``. This test cannot see
    that and must not be cited for it.
    """

    prompts = _ragged_prompts()
    rt = _FakeRuntime(width=256)
    together = generate_dense_mtp_batch(
        rt,
        prompts,
        max_new_tokens=6,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
    )
    for row, prompt in enumerate(prompts):
        alone = generate_dense_mtp_batch(
            _FakeRuntime(width=256),
            [prompt],
            max_new_tokens=6,
            depth=3,
            stop_token_ids={STOP_ID},
            ragged_prompts=True,
        )
        assert together.streams[row].tokens == alone.streams[0].tokens, row
        assert together.streams[row].sha == alone.streams[0].sha, row


def test_ragged_cohort_matches_the_expected_greedy_chain() -> None:
    """Independent of the driver: the deterministic fake's own continuation."""

    prompts = _ragged_prompts()
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
    )
    for stream, prompt in zip(res.streams, prompts):
        assert stream.tokens == _expected_chain(prompt, 5)


def test_ragged_streams_report_their_own_true_prompt_length() -> None:
    prompts = _ragged_prompts()
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=3,
        depth=2,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
    )
    assert [s.prompt_len for s in res.streams] == [len(p) for p in prompts]


def test_row_order_survives_grouping() -> None:
    """Groups are prefilled in length order, so rows come back out of order.

    The permutation that puts them back is easy to get subtly wrong, and the
    failure mode is one caller receiving another caller's completion, so it is
    tested against a prompt order deliberately unsorted by length.
    """

    prompts = [
        [20, 1, 2, 3, 4, 5, 6],  # longest FIRST
        [21, 2],                 # shortest second
        [22, 3, 4, 5],
        [23, 4, 5, 6, 7, 8],
    ]
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
    )
    for stream, prompt in zip(res.streams, prompts):
        assert stream.tokens == _expected_chain(prompt, 4)
        assert stream.prompt_len == len(prompt)


@pytest.mark.parametrize("head_history", ["cycle", "committed"])
@pytest.mark.parametrize("loop_mode", ["pipelined", "serial"])
def test_ragged_holds_across_head_history_and_loop_mode(
    head_history: str, loop_mode: str
) -> None:
    prompts = _ragged_prompts()
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        head_history=head_history,
        loop_mode=loop_mode,
    )
    for stream, prompt in zip(res.streams, prompts):
        assert stream.tokens == _expected_chain(prompt, 5)


def test_uniform_cohort_is_unchanged_when_ragged_is_enabled() -> None:
    """One length means one group, so the flag must be a no-op there."""

    prompts = _distinct_prompts(4)
    plain, _ = _run(prompts, depth=3, max_new=6)
    ragged = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=6,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
    )
    assert [s.sha for s in ragged.streams] == [s.sha for s in plain.streams]


def test_mixed_lengths_are_still_refused_without_the_flag() -> None:
    with pytest.raises(ValueError, match="unless ragged_prompts=True"):
        generate_dense_mtp_batch(
            _FakeRuntime(width=256),
            _ragged_prompts(),
            max_new_tokens=4,
            depth=3,
        )


def test_ragged_composes_with_per_row_caps_and_the_commit_callback() -> None:
    prompts = _ragged_prompts()
    caps = [2, 5, 3, 4]
    seen: list[tuple[int, int]] = []
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=max(caps),
        max_new_tokens_per_row=caps,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        on_commit=lambda row, token: seen.append((row, token)),
    )
    for stream, cap, prompt in zip(res.streams, caps, prompts):
        assert stream.tokens == _expected_chain(prompt, cap)
        assert [tok for row, tok in seen if row == stream.index] == stream.tokens


def test_ragged_with_cohort_slots_pads_with_cheap_dummy_rows() -> None:
    """Dummy rows exist for fixed shapes; under ragged they should cost ~nothing."""

    prompts = _ragged_prompts()
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        cohort_slots=8,
    )
    assert len(res.streams) == len(prompts)
    for stream, prompt in zip(res.streams, prompts):
        assert stream.tokens == _expected_chain(prompt, 4)


# --------------------------------------------------------------------------- #
# Per-request sampling parameters (T-204 item 3)
# --------------------------------------------------------------------------- #
def test_per_row_zero_temperature_matches_the_scalar_greedy_path() -> None:
    """A list of zeros must be the same run as the scalar zero, byte for byte."""

    prompts = _distinct_prompts(4)
    scalar, _ = _run(prompts, depth=3, max_new=6)
    listed = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=6,
        depth=3,
        stop_token_ids={STOP_ID},
        temperature=[0.0, 0.0, 0.0, 0.0],
    )
    assert [s.sha for s in listed.streams] == [s.sha for s in scalar.streams]
    assert listed.meta["sampling"] is False


def test_a_greedy_row_is_exact_inside_a_sampling_cohort() -> None:
    """The property that lets one cohort mix greedy and sampling callers.

    Row 0 asks for temperature 0. It shares a cohort with rows that sample, so
    it goes down the sampling path, where greedy is encoded as top_k=1 at
    temperature 1 — a point mass on the argmax. If that encoding is wrong, this
    caller silently receives sampled tokens while having asked for greedy,
    which is the kind of failure no throughput number would ever reveal.
    """

    prompts = _distinct_prompts(3)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=6,
        depth=3,
        stop_token_ids={STOP_ID},
        temperature=[0.0, 1.5, 2.0],
        sampling_seed=11,
    )
    assert res.meta["sampling"] is True
    assert res.streams[0].tokens == _expected_chain(prompts[0], 6)


def test_top_k_1_is_argmax_at_any_temperature() -> None:
    """Exercises the per-row top_k index arithmetic directly.

    Per-row k is applied as an index into the ascending sort (position
    ``vocab - k``). Off by one there would silently widen or empty a row's
    candidate set, so a row pinned to k=1 at a high temperature must still
    reproduce the greedy chain exactly.
    """

    prompts = _distinct_prompts(2)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        temperature=[5.0, 5.0],
        top_k=[1, 1],
        sampling_seed=3,
    )
    for stream, prompt in zip(res.streams, prompts):
        assert stream.tokens == _expected_chain(prompt, 5)


def test_mixed_top_k_leaves_the_pinned_row_exact() -> None:
    """One row pinned to k=1 next to rows with k disabled and k wide."""

    prompts = _distinct_prompts(3)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        temperature=[1.0, 1.0, 1.0],
        top_k=[1, 0, 40],
        top_p=[1.0, 0.9, 1.0],
        sampling_seed=5,
    )
    assert res.streams[0].tokens == _expected_chain(prompts[0], 5)
    for stream in res.streams:
        assert len(stream.tokens) > 0


def test_per_row_sampling_is_recorded_in_the_receipt() -> None:
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=3,
        depth=2,
        stop_token_ids={STOP_ID},
        temperature=[0.0, 0.7],
        top_k=[1, 20],
        top_p=[1.0, 0.95],
    )
    assert res.meta["row_temperature"] == [0.0, 0.7]
    assert res.meta["row_top_k"] == [1, 20]
    assert res.meta["row_top_p"] == [1.0, 0.95]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"temperature": [0.0, 0.0]}, "one value per prompt"),
        ({"top_k": [1]}, "one value per prompt"),
        ({"top_p": [1.0, 1.0, 1.0, 1.0]}, "one value per prompt"),
        ({"temperature": [0.0, -1.0, 0.0]}, "must be >= 0"),
    ],
)
def test_per_row_sampling_rejects_malformed_input(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        generate_dense_mtp_batch(
            _FakeRuntime(width=256),
            _distinct_prompts(3),
            max_new_tokens=4,
            depth=2,
            **kwargs,
        )


def test_per_row_sampling_composes_with_ragged_prompts() -> None:
    """Items 2 and 3 have to hold at the same time, since item 4 needs both."""

    prompts = _ragged_prompts()
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        temperature=[0.0, 1.2, 0.0, 0.9],
        top_k=[0, 30, 0, 0],
        sampling_seed=7,
    )
    # The two greedy rows keep their exact chains despite ragged neighbours
    # that are sampling.
    assert res.streams[0].tokens == _expected_chain(prompts[0], 5)
    assert res.streams[2].tokens == _expected_chain(prompts[2], 5)
    assert [s.prompt_len for s in res.streams] == [len(p) for p in prompts]


def test_greedy_row_is_exact_on_both_top_k_paths() -> None:
    """The reduction fast path must agree with the sort path exactly.

    A greedy row is encoded as top_k=1. When some OTHER row asks for a genuine
    top_k the cohort takes the sort path; when none does, it takes the cheaper
    max/min reduction path. A greedy caller must get identical tokens either
    way, or its output would depend on what its cohort-mates asked for.
    """

    prompts = _distinct_prompts(3)
    common = dict(
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        temperature=[0.0, 1.0, 1.0],
        sampling_seed=13,
    )
    reduction_path = generate_dense_mtp_batch(
        _FakeRuntime(width=256), prompts, top_k=[0, 0, 0], **common
    )
    sort_path = generate_dense_mtp_batch(
        _FakeRuntime(width=256), prompts, top_k=[0, 25, 0], **common
    )
    expected = _expected_chain(prompts[0], 5)
    assert reduction_path.streams[0].tokens == expected
    assert sort_path.streams[0].tokens == expected


def test_all_greedy_rows_take_the_reduction_path_and_stay_exact() -> None:
    """Several greedy rows beside one sampling row, no caller-requested top_k."""

    prompts = _distinct_prompts(4)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        temperature=[0.0, 0.0, 1.1, 0.0],
        sampling_seed=17,
    )
    for row in (0, 1, 3):
        assert res.streams[row].tokens == _expected_chain(prompts[row], 5), row


# --------------------------------------------------------------------------- #
# Continuous batching (T-204 item 4): a slot serves several requests
# --------------------------------------------------------------------------- #
def test_refill_admits_a_queued_request_into_a_freed_slot() -> None:
    """The core of item 4: a finished row's slot is reused, and the newcomer
    produces its own correct continuation rather than the previous occupant's.
    """

    prompts = _distinct_prompts(2)
    joiner = [40, 3, 1]
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        max_new_tokens_per_row=[2, 4],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        refill_queue=[{"prompt": joiner, "max_new_tokens": 3}],
    )
    assert len(res.streams) == 3, "one result per REQUEST, not per slot"
    assert res.meta["requests_total"] == 3
    assert res.meta["requests_queued"] == 1
    assert res.meta["requests_admitted"] == 3

    # The two initial requests keep their own chains.
    assert res.streams[0].tokens == _expected_chain(prompts[0], 2)
    assert res.streams[1].tokens == _expected_chain(prompts[1], 4)
    # The joiner produces ITS chain, from ITS prompt, not a continuation of
    # whoever held the slot before it.
    assert res.streams[2].tokens == _expected_chain(joiner, 3)
    assert res.streams[2].slot == 0, "should have taken the first slot to free"


def test_refilled_request_matches_initial_cohort_run() -> None:
    """A request must not care whether it was admitted at the start or later.

    This is the property that makes continuous batching safe to enable, and it
    is stated as same-geometry-vs-same-geometry deliberately: both runs are
    two-row cohorts on the fake, whose arithmetic is exact, so a mismatch here
    is an admission bug rather than the float-reduction effect measured on real
    weights.
    """

    joiner = [41, 2, 4]
    base = _distinct_prompts(2)

    refilled = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        base,
        max_new_tokens=5,
        max_new_tokens_per_row=[1, 5],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        refill_queue=[{"prompt": joiner, "max_new_tokens": 5}],
    )
    assert refilled.streams[2].tokens == _expected_chain(joiner, 5)


def test_refill_leaves_the_other_rows_untouched() -> None:
    """An admission forward runs over every row. The non-admitted ones must
    come out of it exactly as they went in, or admitting a request corrupts a
    caller who was mid-generation.
    """

    prompts = _distinct_prompts(3)
    caps = [1, 8, 8]
    with_refill = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=8,
        max_new_tokens_per_row=caps,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        refill_queue=[{"prompt": [42, 1, 2], "max_new_tokens": 4}],
    )
    # Rows 1 and 2 are mid-generation when row 0 finishes and is refilled.
    assert with_refill.streams[1].tokens == _expected_chain(prompts[1], 8)
    assert with_refill.streams[2].tokens == _expected_chain(prompts[2], 8)


def test_no_refill_queue_is_the_identity() -> None:
    prompts = _distinct_prompts(3)
    plain, _ = _run(prompts, depth=3, max_new=5, loop_mode="serial")
    empty = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        loop_mode="serial",
        refill_queue=[],
    )
    assert [s.sha for s in empty.streams] == [s.sha for s in plain.streams]


@pytest.mark.parametrize("loop_mode", ["serial", "pipelined"])
def test_refill_gives_the_same_result_in_both_loop_modes(loop_mode: str) -> None:
    """The pipelined loop pauses for admission; the answer must not change.

    Increment 2 refused pipelined mode outright rather than splice a joiner
    into an in-flight graph. Increment 3 pauses the pipeline for exactly the
    cycles an admission happens. If the pause is wrong, the two modes disagree
    here, which is the whole point of running both.
    """

    prompts = _distinct_prompts(2)
    joiner = [46, 2, 5]
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        max_new_tokens_per_row=[2, 5],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode=loop_mode,
        refill_queue=[{"prompt": joiner, "max_new_tokens": 4}],
    )
    assert res.meta["requests_admitted"] == 3
    assert res.streams[0].tokens == _expected_chain(prompts[0], 2)
    assert res.streams[1].tokens == _expected_chain(prompts[1], 5)
    assert res.streams[2].tokens == _expected_chain(joiner, 4)


def test_refill_drains_a_longer_queue_under_the_pipelined_loop() -> None:
    """Several admissions in one run, each pausing the pipeline once."""

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=4,
        max_new_tokens_per_row=[1, 2],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="pipelined",
        refill_queue=[
            {"prompt": [47, 1], "max_new_tokens": 2},
            {"prompt": [48, 2, 1], "max_new_tokens": 2},
            {"prompt": [49, 3], "max_new_tokens": 3},
        ],
    )
    assert res.meta["requests_total"] == 5
    assert res.meta["requests_admitted"] == 5, "the whole queue should drain"
    assert res.streams[2].tokens == _expected_chain([47, 1], 2)
    assert res.streams[3].tokens == _expected_chain([48, 2, 1], 2)
    assert res.streams[4].tokens == _expected_chain([49, 3], 3)


def test_every_queued_request_is_served_when_slots_free() -> None:
    """A queue drains through the slots rather than starving.

    This started as a "never admitted" test on the assumption that two rows at
    the same cap would leave no slot free. That was wrong: they finish, which
    is precisely when a slot frees, so the queued request IS admitted. The
    correct assertion is that the queue drains. ``not_admitted`` remains
    reachable only when the cycle guard runs out with requests still queued,
    which the guard is deliberately generous enough to make rare.
    """

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=3,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        refill_queue=[
            {"prompt": [44, 1], "max_new_tokens": 3},
            {"prompt": [45, 2, 3], "max_new_tokens": 2},
        ],
    )
    assert res.meta["requests_total"] == 4
    assert res.meta["requests_admitted"] == 4, "the queue should drain"
    assert [s.finish_reason for s in res.streams] == ["length"] * 4
    # Each queued request produced ITS OWN continuation.
    assert res.streams[2].tokens == _expected_chain([44, 1], 3)
    assert res.streams[3].tokens == _expected_chain([45, 2, 3], 2)
    # Two requests shared a slot, which is the whole point.
    assert len({s.slot for s in res.streams}) < len(res.streams)


def test_multiple_joiners_of_different_lengths_admitted_together() -> None:
    """Several slots free at once, joiners of differing lengths.

    Grouped by length, so each group's forward has no padding and every row's
    logits land on its own last token. A joiner must get ITS chain regardless
    of which group it landed in or who it shared a forward with.
    """

    prompts = _distinct_prompts(3)
    joiners = [
        {"prompt": [50, 1, 2], "max_new_tokens": 3},      # len 3
        {"prompt": [51, 4], "max_new_tokens": 3},          # len 2
        {"prompt": [52, 5, 6], "max_new_tokens": 2},       # len 3, shares a group
    ]
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        max_new_tokens_per_row=[1, 1, 1],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        refill_queue=joiners,
    )
    assert res.meta["requests_admitted"] == 6
    for offset, joiner in enumerate(joiners):
        stream = res.streams[3 + offset]
        assert stream.tokens == _expected_chain(
            joiner["prompt"], joiner["max_new_tokens"]
        ), f"joiner {offset} got the wrong continuation"


def test_a_joiner_is_unaffected_by_who_shares_its_admission_forward() -> None:
    """Admission must not leak between joiners sharing one forward.

    The same joiner is admitted twice: once alongside a same-length partner
    (sharing its forward) and once alongside a different-length one (its own
    forward). Its tokens must be identical either way, or the admission pass is
    contaminating rows through the batch it was prefilled in.
    """

    target = {"prompt": [53, 7, 8], "max_new_tokens": 3}
    partner_same = {"prompt": [54, 9, 1], "max_new_tokens": 3}   # same length
    partner_diff = {"prompt": [54, 9], "max_new_tokens": 3}      # different

    def run(partner):
        return generate_dense_mtp_batch(
            _FakeRuntime(width=256),
            _distinct_prompts(2),
            max_new_tokens=3,
            max_new_tokens_per_row=[1, 1],
            depth=3,
            stop_token_ids={STOP_ID},
            ragged_prompts=True,
            loop_mode="serial",
            refill_queue=[dict(target), dict(partner)],
        )

    shared = run(partner_same)
    separate = run(partner_diff)
    assert shared.streams[2].tokens == _expected_chain(target["prompt"], 3)
    assert shared.streams[2].tokens == separate.streams[2].tokens


def test_on_commit_reports_the_request_not_the_slot() -> None:
    """A joiner's tokens must be attributable to the joiner.

    ``on_commit`` used to report the SLOT, which under refill serves several
    requests in turn, so a serving lane had no way to tell whose token it was
    holding. Reporting the request index fixes that, and because request index
    equals slot index without a queue, it changes nothing for callers that
    never refill.
    """

    prompts = _distinct_prompts(2)
    joiner = [55, 1, 3]
    seen: list[tuple[int, int]] = []
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        max_new_tokens_per_row=[1, 4],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        refill_queue=[{"prompt": joiner, "max_new_tokens": 3}],
        on_commit=lambda request, token: seen.append((request, token)),
    )
    for stream in res.streams:
        emitted = [tok for req, tok in seen if req == stream.index]
        assert emitted == stream.tokens, f"request {stream.index}"
    # The joiner is request 2 even though it ran in slot 0.
    assert res.streams[2].slot == 0
    assert any(req == 2 for req, _ in seen), "joiner tokens were never attributed"


def test_joiner_keeps_its_own_sampling_not_the_slots_previous_occupant() -> None:
    """A greedy joiner in a sampling cohort must stay exact.

    Sampling vectors are built once at setup, so before this a joiner inherited
    whatever its slot's previous occupant used — a caller silently receiving
    another caller's temperature. The joiner here asks for temperature 0 while
    landing in a slot vacated by a temperature-1.5 row.
    """

    prompts = _distinct_prompts(2)
    joiner = [56, 2, 4]
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        max_new_tokens_per_row=[1, 4],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        temperature=[1.5, 1.5],
        sampling_seed=5,
        refill_queue=[
            {"prompt": joiner, "max_new_tokens": 3, "temperature": 0.0}
        ],
    )
    # Greedy means exact: the joiner must reproduce the deterministic chain.
    assert res.streams[2].tokens == _expected_chain(joiner, 3)


def test_an_all_greedy_cohort_acquires_sampling_for_a_joiner_that_needs_it() -> None:
    """It used to be refused. Refusing it was measured costing 27-second waits.

    An all-greedy cohort takes a dedicated path that consumes no randomness, so
    a sampling joiner had nowhere to go and the driver failed up front. Under an
    ordinary mixed load -- 65% of requests greedy -- most cohorts sealed
    all-greedy, and every sampling request then waited out a cohort that can
    serve eight times its own width before winding down. Found by hammering.

    The cohort now ACQUIRES the sampling path when a joiner needs it. That is
    safe in this direction and not the other: a greedy row is encoded as
    top_k=1 at temperature 1, a point mass on the argmax, so it decodes
    correctly on the sampling path and merely pays for draws it did not need.
    Going back would change tie-breaking for rows already mid-generation.
    """

    prompts = _distinct_prompts(2)
    joiner = [57, 1]
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        depth=3,
        loop_mode="serial",
        stop_token_ids={STOP_ID},
        refill_queue=[
            {"prompt": joiner, "max_new_tokens": 2, "temperature": 0.9}
        ],
    )
    assert res.meta["sampling"] is True, (
        "the cohort stayed on the greedy path, so the joiner was served with "
        "no randomness at all -- silently, which is the outcome the old "
        "refusal existed to prevent"
    )
    assert len(res.streams) == 3
    assert res.streams[2].finish_reason != "not_admitted"
    assert res.streams[2].tokens, "the joiner produced nothing"


def test_a_live_queue_joiner_also_upgrades_the_cohort() -> None:
    """The frozen list is decided up front; a live pull is not.

    The frozen case can simply build the cohort on the sampling path before
    anything runs. A pulled joiner arrives mid-decode, so the upgrade has to
    happen at the admission boundary -- and BEFORE the joiner's first token is
    drawn, or the one token nobody looks at is drawn greedily.
    """

    pull, _ = _live_queue(
        [{"prompt": [58, 2, 1], "max_new_tokens": 3, "temperature": 0.8, "seed": 5}]
    )
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=4,
        depth=3,
        loop_mode="serial",
        stop_token_ids={STOP_ID},
        pull_queued=pull,
        max_cohort_rows=3,
    )
    assert res.meta["sampling"] is True
    assert res.meta["rows_peak"] == 3
    assert res.streams[2].tokens


# --------------------------------------------------------------------------- #
# P0-2: eviction — an abandoned row must give its slot back
# --------------------------------------------------------------------------- #
def test_cancelled_row_frees_its_slot_for_a_queued_request() -> None:
    """The capacity leak, as a test.

    Before this, `done[b]` was set only by a stop token or the cap, so a client
    that disconnected left its row decoding to its full max_tokens holding a
    slot no queued request could take. Under load with real disconnects the
    lane silently loses capacity to rows nobody is listening to.

    Here row 0 is abandoned after its first token while holding a large budget.
    Its slot must be reclaimed and the queued request served.
    """

    prompts = _distinct_prompts(2)
    joiner = [60, 1, 2]
    committed: list[int] = []
    cancelled_requests = set()

    def on_commit(request, token):
        committed.append(request)
        # Row 0 goes away right after its first token.
        if request == 0 and committed.count(0) >= 1:
            cancelled_requests.add(0)

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=40,
        max_new_tokens_per_row=[40, 3],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        refill_queue=[{"prompt": joiner, "max_new_tokens": 3}],
        on_commit=on_commit,
        is_cancelled=lambda request: request in cancelled_requests,
    )

    assert res.streams[0].finish_reason == "cancelled"
    assert len(res.streams[0].tokens) < 40, "an evicted row must stop early"
    assert res.meta["requests_evicted"] >= 1
    # The whole point: the queued request got the freed slot.
    assert res.streams[2].finish_reason == "length"
    assert res.streams[2].tokens == _expected_chain(joiner, 3)
    assert res.streams[2].slot == 0, "it should have taken the evicted slot"


def test_eviction_does_not_disturb_the_other_rows() -> None:
    """Evicting one row must not perturb a healthy neighbour."""

    prompts = _distinct_prompts(2)
    cancelled = set()

    def on_commit(request, token):
        if request == 0:
            cancelled.add(0)

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=6,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        on_commit=on_commit,
        is_cancelled=lambda request: request in cancelled,
    )
    assert res.streams[0].finish_reason == "cancelled"
    assert res.streams[1].tokens == _expected_chain(prompts[1], 6)


def test_a_throwing_cancellation_probe_does_not_kill_the_cohort() -> None:
    """A broken probe must degrade to 'still wanted', not take the run down."""

    def _boom(request):
        raise RuntimeError("probe exploded")

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        is_cancelled=_boom,
    )
    assert [s.finish_reason for s in res.streams] == ["length", "length"]
    assert res.meta["requests_evicted"] == 0


def test_no_cancellation_probe_is_the_identity() -> None:
    prompts = _distinct_prompts(3)
    plain, _ = _run(prompts, depth=3, max_new=5, loop_mode="serial")
    same = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        loop_mode="serial",
        is_cancelled=None,
    )
    assert [s.sha for s in same.streams] == [s.sha for s in plain.streams]


# --------------------------------------------------------------------------- #
# Per-request seeds
# --------------------------------------------------------------------------- #
def _sampled(seeds, run_seed=7, n=3):
    prompts = _distinct_prompts(n)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=8,
        depth=3,
        stop_token_ids=set(),
        temperature=[1.4] * n,
        sampling_seed=run_seed,
        row_sampling_seeds=seeds,
    )
    return res, [tuple(stream.tokens) for stream in res.streams]


def test_one_callers_seed_does_not_steer_another_callers_tokens() -> None:
    """The defect this exists to fix.

    The cohort used jobs[0].seed for every row, so request B's output moved when
    request A changed its seed. Nobody would see it until two callers compared
    notes, and by then the run is unreproducible.

    Row 1's seed is swept rather than swapped once. On a peaked distribution two
    different seeds can legitimately draw the same tokens, so a single swap
    proves nothing either way. Isolation is the invariant and is asserted on
    every trial; self-steering is asserted across the sweep.
    """

    _, base = _sampled([100, 200, 300])

    moved_own = 0
    for seed in (201, 250, 333, 444, 555, 666, 777, 888, 999, 1234, 4242):
        _, cur = _sampled([100, seed, 300])
        assert cur[0] == base[0], "row 0's seed did not change; it must not move"
        assert cur[2] == base[2], "row 2's seed did not change; it must not move"
        if cur[1] != base[1]:
            moved_own += 1

    assert moved_own > 0, "row 1's own seed never steered its own tokens"


def test_identical_row_seeds_keep_the_shared_key_path() -> None:
    """Cost control: the per-row path must not engage when it buys nothing.

    Greedy rows draw no randomness, so their seeds are irrelevant and must not
    drag a cohort onto per-row draws.
    """

    res, _ = _sampled([42, 42, 42])
    assert res.meta["per_row_random"] is False

    res, _ = _sampled([1, 2, 3])
    assert res.meta["per_row_random"] is True

    # Distinct seeds but every row greedy: nothing to decorrelate.
    prompts = _distinct_prompts(3)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=6,
        depth=3,
        stop_token_ids=set(),
        temperature=[1.0, 0.0, 0.0],
        sampling_seed=5,
        row_sampling_seeds=[1, 2, 3],
    )
    assert res.meta["per_row_random"] is False, (
        "only one row samples, so there is one seed that matters"
    )


def test_same_seed_same_slot_reproduces_within_a_fixed_geometry() -> None:
    """What per-row seeds DO promise: repeatability at fixed geometry.

    Not repeatability against a solo run -- a row's logits depend on batch
    width, so that is out of reach and is documented as such.
    """

    _, first = _sampled([11, 22, 33])
    _, again = _sampled([11, 22, 33])
    assert first == again


def test_unseeded_rows_fall_back_to_the_run_seed() -> None:
    """A short seed list must not throw or silently zero the tail."""

    res, tokens = _sampled([5], n=3)
    assert len(tokens) == 3
    assert all(len(t) > 0 for t in tokens)


# --------------------------------------------------------------------------- #
# Wall-clock deadline (P1-5)
# --------------------------------------------------------------------------- #
def test_deadline_cuts_the_run_and_says_so_distinctly() -> None:
    """A timeout must not be reported as the caller's own token limit.

    "length" means the caller got what it asked for; "deadline" means the
    server gave up. An operator debugging a timeout that reads as "length"
    looks in entirely the wrong place.
    """

    prompts = _distinct_prompts(2)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=512),
        prompts,
        max_new_tokens=400,
        depth=3,
        stop_token_ids=set(),
        deadline_s=0.05,
    )
    assert res.meta["deadline_hit"] is True
    assert {s.finish_reason for s in res.streams} == {"deadline"}
    # Cut short rather than run to the cap. Not asserting a non-empty result:
    # a deadline that expires before the first cycle legitimately yields
    # nothing, and a test should not demand more than the design promises.
    assert all(len(s.tokens) < 400 for s in res.streams)


def test_a_row_that_finished_on_its_own_keeps_its_own_reason() -> None:
    """The deadline must not relabel rows that had already completed."""

    prompts = _distinct_prompts(2)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        depth=3,
        stop_token_ids=set(),
        deadline_s=30.0,
    )
    assert res.meta["deadline_hit"] is False
    assert {s.finish_reason for s in res.streams} == {"length"}


def test_no_deadline_is_the_default_and_changes_nothing() -> None:
    """Off by default: a lane must not start truncating unasked."""

    prompts = _distinct_prompts(2)
    plain = generate_dense_mtp_batch(
        _FakeRuntime(width=256), prompts, max_new_tokens=6, depth=3,
        stop_token_ids={STOP_ID},
    )
    generous = generate_dense_mtp_batch(
        _FakeRuntime(width=256), prompts, max_new_tokens=6, depth=3,
        stop_token_ids={STOP_ID}, deadline_s=600.0,
    )
    assert [s.tokens for s in plain.streams] == [s.tokens for s in generous.streams]
    assert plain.meta["deadline_hit"] is False


# --------------------------------------------------------------------------- #
# P1-4: the monotone per-slot KV reservation is reported, not just true
# --------------------------------------------------------------------------- #
def test_run_reports_what_its_kv_actually_reserves() -> None:
    """Per-slot capacity never shrinks, so the reservation must be visible.

    Measured on the real trunk at 64 KiB per token per slot: 8 slots at 128k
    context reserve 64 GiB. A property that large should not be folklore.
    """

    prompts = _distinct_prompts(3)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=512), prompts, max_new_tokens=8, depth=3,
        stop_token_ids=set(),
    )
    meta = res.meta
    assert meta["kv_reserved_bytes"] > 0
    assert meta["kv_capacity_tokens_per_slot"] > 0
    assert meta["kv_bytes_per_token_per_slot"] > 0
    # Reserved must bound used; the gap is exactly what the monotone bound costs.
    assert meta["kv_used_bytes_estimate"] <= meta["kv_reserved_bytes"]


def test_a_longer_prompt_reserves_strictly_more() -> None:
    """The reservation must track the longest prompt, not the request count."""

    short = generate_dense_mtp_batch(
        _FakeRuntime(width=1024), _distinct_prompts(2, length=8),
        max_new_tokens=4, depth=3, stop_token_ids=set(),
    )
    long = generate_dense_mtp_batch(
        _FakeRuntime(width=1024), _distinct_prompts(2, length=600),
        max_new_tokens=4, depth=3, stop_token_ids=set(),
    )
    assert (
        long.meta["kv_capacity_tokens_per_slot"]
        > short.meta["kv_capacity_tokens_per_slot"]
    )
    assert long.meta["kv_reserved_bytes"] > short.meta["kv_reserved_bytes"]


# --------------------------------------------------------------------------- #
# Presence / frequency penalties (were silently dropped)
#
# These use a fake with a REACHABLE runner-up (alt_gap 9.0 against a target of
# 10.0). The default fake gives the winner a 10-point lead over an otherwise
# empty vocabulary, so no penalty in the canonical [-2, 2] range could ever flip
# it -- a test written against that fake would pass whether or not penalties
# were implemented at all, which is the failure mode this whole task keeps
# meeting.
# --------------------------------------------------------------------------- #
def _pen_fixture() -> "_FakeRuntime":
    return _FakeRuntime(width=256, alt_gap_verify=9.0, alt_gap_draft=9.0)


def _pen_run(pen_kw, *, n=2, steps=16, temperature=0.0, runtime=None):
    return generate_dense_mtp_batch(
        runtime or _pen_fixture(),
        _distinct_prompts(n),
        max_new_tokens=steps,
        depth=3,
        stop_token_ids=set(),
        temperature=temperature,
        **pen_kw,
    )


def test_a_frequency_penalty_breaks_a_repetition_loop() -> None:
    """The bug: the server accepted penalties and the driver ignored them.

    Silently -- no error, output identical to penalty 0. This is what would have
    caught it, and it also shows the penalty doing its actual job: the
    unpenalised chain settles into a five-token loop, and the penalty leaves it
    at exactly the point the repeat begins.
    """

    plain = _pen_run({}).streams[0].tokens
    penalised = _pen_run({"frequency_penalty": 2.0}).streams[0].tokens

    assert len(set(plain[6:])) < len(plain[6:]), "fixture must actually loop"
    assert penalised != plain, "frequency_penalty=2.0 changed nothing"
    assert penalised[:6] == plain[:6], (
        "the penalty must only bite once a token repeats, not from the start"
    )
    assert len(set(penalised[6:])) == len(penalised[6:]), (
        "the penalised continuation should stop repeating"
    )


def test_penalties_apply_on_the_sampling_path_too() -> None:
    """The greedy paths bypass the filtering pipeline, the sampling paths do
    not. Both need the penalty, and covering only one would be the same bug in
    a narrower form."""

    plain = _pen_run({}, temperature=1.0).streams[0].tokens
    penalised = _pen_run(
        {"frequency_penalty": 2.0}, temperature=1.0
    ).streams[0].tokens
    assert penalised != plain


def test_penalties_are_per_row() -> None:
    """One penalised row must not perturb an unpenalised cohort-mate."""

    plain = _pen_run({}, n=3)
    mixed = _pen_run({"frequency_penalty": [0.0, 2.0, 0.0]}, n=3)
    assert mixed.streams[1].tokens != plain.streams[1].tokens, "row 1 must move"
    assert mixed.streams[0].tokens == plain.streams[0].tokens, "row 0 must not"
    assert mixed.streams[2].tokens == plain.streams[2].tokens, "row 2 must not"


def test_zero_penalty_changes_nothing() -> None:
    """The machinery must be entirely absent when no penalty is set."""

    a = _pen_run({}).streams
    b = _pen_run({"presence_penalty": 0.0, "frequency_penalty": 0.0}).streams
    assert [s.tokens for s in a] == [s.tokens for s in b]


def test_penalty_coefficients_clamp_to_the_canonical_range() -> None:
    """fast_sampling.apply_penalties_mlx clamps to [-2, 2]; this lane must not
    quietly diverge from the lane it is meant to match."""

    at_limit = _pen_run({"frequency_penalty": 2.0}).streams
    way_over = _pen_run({"frequency_penalty": 500.0}).streams
    assert [s.tokens for s in at_limit] == [s.tokens for s in way_over]


def test_presence_and_frequency_are_different_knobs() -> None:
    """presence is a FLAT charge for having appeared; frequency SCALES.

    The first version of this test asserted only that each differed from no
    penalty at all -- which would have passed with both wired to the same term,
    the very thing it claims to check. Same defect class as the count test: an
    assertion about direction cannot separate two things that both move.

    The discriminator is a coefficient BELOW the fixture's 1.0 gap. A flat
    presence charge of 0.6 can never overcome it, however many times a token
    repeats. A frequency charge of 0.6 reaches 1.2 on the second occurrence and
    does. So one must leave the chain untouched and the other must not.
    """

    plain = _pen_run({}, steps=24).streams[0].tokens
    presence = _pen_run({"presence_penalty": 0.6}, steps=24).streams[0].tokens
    frequency = _pen_run({"frequency_penalty": 0.6}, steps=24).streams[0].tokens

    assert presence == plain, (
        "a flat 0.6 charge cannot beat a 1.0 gap at any repeat count; "
        "presence is scaling with the count, so it is wired as frequency"
    )
    assert frequency != plain, (
        "0.6 x 2 occurrences = 1.2 must beat the 1.0 gap; frequency is not "
        "scaling with the count, so it is wired as presence"
    )


def test_an_unpenalised_cohort_acquires_penalty_machinery_for_a_joiner() -> None:
    """Also used to be refused, and for a better reason than it needed to be.

    Serving a penalised request unpenalised is a silent wrong answer, so the
    old refusal was right to be loud. But the machinery is a lazily allocated
    [B, V] counts buffer, so the cohort can simply acquire it -- and every
    existing row's counts start at zero, which is correct, because penalties
    count the COMPLETION only and those rows have not been penalised.

    The fail-loud branch is kept as a backstop rather than deleted. It is how a
    future drift becomes a crash instead of a wrong answer.
    """

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=512), _distinct_prompts(2),
        max_new_tokens=6, depth=3,
        stop_token_ids={STOP_ID}, cohort_slots=2,
        refill_queue=[
            {"prompt": [11, 2, 3], "max_new_tokens": 6,
             "temperature": 0.0, "frequency_penalty": 1.5},
        ],
    )
    assert res.meta["penalties"] is True, (
        "the cohort ran without penalty machinery, so the joiner was served "
        "unpenalised -- the silent wrong answer this check exists to prevent"
    )
    assert len(res.streams) == 3
    assert res.streams[2].finish_reason != "not_admitted"


def test_penalty_counts_advance_exactly_once_per_committed_token() -> None:
    """Pins the count arithmetic instead of trusting the argument for it.

    A double-advance would be invisible in every other penalty test: output
    would still change, just more than it should. So this measures WHEN the
    penalty bites rather than whether it does.

    Method: the fixture's runner-up trails by exactly 1.0, so with
    frequency_penalty f the winner flips once f * count > 1.0. The unpenalised
    chain loops with period 5 from index 1, so a token at index 1 + 5n has n
    prior occurrences. That makes the first divergence index a direct readout
    of the count.
    """

    def run(f, depth=3):
        return generate_dense_mtp_batch(
            _pen_fixture(), _distinct_prompts(1), max_new_tokens=20,
            depth=depth, stop_token_ids=set(), temperature=0.0,
            frequency_penalty=f,
        ).streams[0].tokens

    base = run(0.0)
    assert base[1:6] == base[6:11], "fixture must loop with period 5"

    def first_divergence(tokens):
        return next(
            (i for i in range(len(tokens)) if tokens[i] != base[i]), None
        )

    # f * count > 1.0 -> flips at the (ceil(1/f))-th occurrence.
    for penalty, prior_occurrences in ((1.01, 1), (0.51, 2), (0.34, 3)):
        assert first_divergence(run(penalty)) == 1 + 5 * prior_occurrences, (
            f"frequency_penalty={penalty} bit at the wrong occurrence; the "
            "counts are not advancing once per committed token"
        )


def test_penalty_counts_do_not_depend_on_draft_depth() -> None:
    """Counts must track COMMITTED TOKENS, not decode cycles.

    Each cycle commits a variable number of tokens (x0 plus however many drafts
    were accepted), so a per-cycle counter would drift with depth. Same run at
    four depths must bite at the same place.
    """

    def run(depth):
        return generate_dense_mtp_batch(
            _pen_fixture(), _distinct_prompts(1), max_new_tokens=20,
            depth=depth, stop_token_ids=set(), temperature=0.0,
            frequency_penalty=0.34,
        ).streams[0].tokens

    base = generate_dense_mtp_batch(
        _pen_fixture(), _distinct_prompts(1), max_new_tokens=20, depth=3,
        stop_token_ids=set(), temperature=0.0,
    ).streams[0].tokens

    divergences = {
        depth: next(
            (i for i in range(20) if run(depth)[i] != base[i]), None
        )
        for depth in (1, 2, 3, 4)
    }
    assert len(set(divergences.values())) == 1, (
        f"depth changed when the penalty bit: {divergences}"
    )


# --------------------------------------------------------------------------- #
# Gaps found by mutation audit: three fixes that nothing was checking
# --------------------------------------------------------------------------- #
def test_a_joiner_uses_its_own_temperature_not_its_predecessors() -> None:
    """Found by mutation: deleting the joiner sampling rebuild kept the suite
    green.

    The fix's own comment says "without this a caller silently receives another
    caller's temperature" -- and no test checked it. A greedy joiner taking a
    slot whose previous occupant sampled at 1.5 must produce the deterministic
    greedy chain; if it inherits 1.5 it will not.
    """

    prompts = _distinct_prompts(2)
    joiner = [77, 3, 4]
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=6,
        max_new_tokens_per_row=[2, 6],
        depth=3,
        stop_token_ids={STOP_ID},
        # Temperature 50, not 1.5, and that matters. At 1.5 the fake's target
        # token still wins about 93% of the time, so "sampled output differs
        # from greedy" would hold only by luck and the test would pass most
        # runs even with the fix removed -- which is exactly what the mutation
        # audit caught it doing. At 50 the distribution is near-uniform over the
        # vocabulary, so an inherited temperature diverges from the greedy chain
        # with overwhelming probability.
        temperature=[50.0, 50.0],
        sampling_seed=5,
        loop_mode="serial",
        refill_queue=[
            {"prompt": joiner, "max_new_tokens": 4, "temperature": 0.0},
        ],
    )
    assert res.streams[2].tokens == _expected_chain(joiner, 4), (
        "the joiner asked for temperature 0 and must get the greedy chain, "
        "not its predecessor's 50"
    )


def test_a_joiner_uses_its_own_seed_not_the_slots_stream_position() -> None:
    """Found by mutation: deleting the joiner stream reset kept the suite green.

    A joiner must draw from a stream keyed on ITS seed. If it instead continues
    wherever the slot's previous occupant had got to, its output depends on that
    occupant -- the same cross-request coupling per-row seeds exist to remove,
    only harder to see because the influence comes from a request that has
    already finished.

    Two runs differing ONLY in a predecessor's seed must leave the joiner
    byte-identical.
    """

    prompts = _distinct_prompts(2)
    joiner = [88, 2, 3]

    def run(predecessor_seed):
        return generate_dense_mtp_batch(
            _FakeRuntime(width=256),
            prompts,
            max_new_tokens=8,
            max_new_tokens_per_row=[2, 8],
            depth=3,
            stop_token_ids=set(),
            # High temperature for the same reason as the sibling test: at
            # 1.4 the fake's target token dominates, so two different streams
            # would usually produce the SAME tokens and the assertion would
            # hold whether or not the reset happened. Near-uniform sampling
            # makes the streams observable.
            temperature=[50.0, 50.0],
            row_sampling_seeds=[predecessor_seed, 20],
            loop_mode="serial",
            refill_queue=[
                {"prompt": joiner, "max_new_tokens": 6,
                 "temperature": 50.0, "seed": 99},
            ],
        ).streams[2].tokens

    assert run(10) == run(11), (
        "the joiner's tokens moved when only a PREDECESSOR's seed changed"
    )


# --------------------------------------------------------------------------- #
# Continuous batching (T-204 item 4): a LIVE queue, and width that follows it
# --------------------------------------------------------------------------- #
def _live_queue(items: list[dict]) -> "tuple[callable, list[int]]":
    """A pull callable over a fixed list, plus a record of the capacities asked.

    The capacity record is the point: it is how a test proves the driver asked
    for room it actually had, rather than pulling work it then had to queue.
    """

    state = {"next": 0}
    asked: list[int] = []

    def pull(capacity: int) -> list[dict]:
        asked.append(int(capacity))
        take = max(0, min(int(capacity), len(items) - state["next"]))
        out = items[state["next"] : state["next"] + take]
        state["next"] += take
        return list(out)

    return pull, asked


def test_width_follows_demand_in_steps_of_one() -> None:
    """The headline property. Three requests run as a batch of three.

    A cohort that seals at one row and then meets three waiting requests must
    GROW to four, not pad to a fixed shape and not queue them behind the row
    already running. `rows_peak` is the receipt.
    """

    base = _distinct_prompts(1)
    joiners = _distinct_prompts(4)[1:]
    pull, asked = _live_queue(
        [{"prompt": p, "max_new_tokens": 6} for p in joiners]
    )
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        base,
        max_new_tokens=6,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=pull,
        max_cohort_rows=4,
    )
    assert res.meta["continuous"] is True
    assert res.meta["rows_peak"] == 4, (
        f"cohort never grew past {res.meta['rows_peak']} rows; a live queue "
        "that cannot widen the batch is the frozen refill list again"
    )
    assert res.meta["max_cohort_rows"] == 4
    assert res.meta["cohort_resizes"] >= 1
    assert len(res.streams) == 4, "one stream per REQUEST"
    assert all(s.finish_reason in {"length", "stop"} for s in res.streams)
    assert max(asked) <= 4, "never ask for more rows than the cohort may hold"


def test_the_cohort_never_pads_to_the_row_ceiling() -> None:
    """Width is the number of live rows, not the ceiling it is allowed to reach.

    The distinction is the whole of "in steps of one": a ceiling of eight with
    two callers must decode two rows, not eight with six inert ones.
    """

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=lambda capacity: [],
        max_cohort_rows=8,
    )
    assert res.meta["rows_peak"] == 2, "an empty queue must not inflate the batch"
    assert res.meta["max_cohort_rows"] == 8


def test_a_finished_row_leaves_the_batch_instead_of_idling() -> None:
    """Eviction on the row axis, which is the other half of following demand.

    Before this, a row that finished stayed in the rectangle and was computed
    every cycle until the whole cohort drained. Per-stream throughput flattens
    from about sixteen rows on, so an unfilled row is not cheap -- it is a row
    of work spent on nobody.
    """

    prompts = _distinct_prompts(3)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=8,
        max_new_tokens_per_row=[1, 2, 8],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=lambda capacity: [],
        max_cohort_rows=3,
    )
    assert res.meta["rows_peak"] == 3
    assert res.meta["rows_final"] < 3, (
        "the two short rows were still in the batch at the end"
    )
    assert res.streams[0].tokens == _expected_chain(prompts[0], 1)
    assert res.streams[1].tokens == _expected_chain(prompts[1], 2)
    assert res.streams[2].tokens == _expected_chain(prompts[2], 8)


def test_a_joiner_of_a_different_length_is_prefilled_at_its_own_length() -> None:
    """THE padding trap, as a test. It is the only silent failure on the list.

    The admission path this replaces prefilled every joiner of one boundary at
    a single shared ``prompt_len``. Pad tokens entering a GDN layer are folded
    into a recurrent state that no offset rewinds, so a padded joiner conditions
    on tokens its caller never sent -- and it fails quietly: the model loads,
    runs, and returns fluent text. The check is that each joiner emits exactly
    the continuation of ITS OWN prompt, and the prompts are deliberately three
    different lengths so no single shared length could have been used.
    """

    base = _distinct_prompts(1, length=4)
    short = [21, 5]
    medium = [22, 4, 3, 2, 1, 6]
    tall = [23] + [1 + (j % 5) for j in range(11)]
    assert len({len(short), len(medium), len(tall), len(base[0])}) == 4
    pull, _ = _live_queue(
        [
            {"prompt": short, "max_new_tokens": 5},
            {"prompt": medium, "max_new_tokens": 5},
            {"prompt": tall, "max_new_tokens": 5},
        ]
    )
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        base,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=pull,
        max_cohort_rows=4,
    )
    assert res.streams[1].tokens == _expected_chain(short, 5), "short joiner"
    assert res.streams[2].tokens == _expected_chain(medium, 5), "medium joiner"
    assert res.streams[3].tokens == _expected_chain(tall, 5), "long joiner"
    assert res.streams[1].prompt_len == len(short)
    assert res.streams[3].prompt_len == len(tall)


def test_joiners_admitted_at_one_boundary_do_not_contaminate_each_other() -> None:
    """Two different lengths joining on the SAME cycle is the trap's live case.

    One joiner is the easy case: there is nothing to pad it against. The
    dangerous shape is several arriving together, which is precisely when the
    old path picked one length and padded the rest to it.
    """

    base = _distinct_prompts(1, length=3)
    a = [31, 9]
    b = [32, 8, 7, 6, 5, 4, 3]
    pulled = {"done": False}

    def pull(capacity: int) -> list[dict]:
        if pulled["done"] or capacity < 2:
            return []
        pulled["done"] = True
        return [
            {"prompt": a, "max_new_tokens": 4},
            {"prompt": b, "max_new_tokens": 4},
        ]

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        base,
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=pull,
        max_cohort_rows=3,
    )
    assert res.streams[1].tokens == _expected_chain(a, 4)
    assert res.streams[2].tokens == _expected_chain(b, 4)


def test_a_pulled_joiner_carries_its_own_sampling_not_a_neighbours() -> None:
    """A joiner must never inherit the cohort's sampling or its randomness."""

    base = _distinct_prompts(2)
    pull, _ = _live_queue(
        [{"prompt": [44, 3, 2], "max_new_tokens": 4, "temperature": 0.0, "seed": 99}]
    )
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        base,
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        temperature=[0.0, 0.0],
        pull_queued=pull,
        max_cohort_rows=3,
    )
    # A greedy joiner in a greedy cohort must produce its own greedy chain.
    assert res.streams[2].tokens == _expected_chain([44, 3, 2], 4)


def test_the_pipelined_loop_admits_too() -> None:
    """The serial loop is the A/B baseline; the pipelined one is what ships."""

    base = _distinct_prompts(1)
    pull, _ = _live_queue(
        [{"prompt": p, "max_new_tokens": 5} for p in _distinct_prompts(3)[1:]]
    )
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        base,
        max_new_tokens=5,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="pipelined",
        pull_queued=pull,
        max_cohort_rows=3,
    )
    assert res.meta["rows_peak"] == 3
    assert len(res.streams) == 3
    assert all(s.tokens for s in res.streams)


def test_an_empty_queue_ends_the_cohort_rather_than_spinning() -> None:
    """A continuous cohort must still terminate, and on the queue's word."""

    calls = {"n": 0}

    def pull(capacity: int) -> list[dict]:
        calls["n"] += 1
        return []

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=3,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=pull,
        max_cohort_rows=4,
    )
    assert calls["n"] > 0, "the queue was never asked"
    assert all(s.finish_reason == "length" for s in res.streams)


def test_a_queue_that_raises_does_not_take_the_cohort_down() -> None:
    """The rows already decoding belong to callers who are still waiting."""

    prompts = _distinct_prompts(2)

    def pull(capacity: int) -> list[dict]:
        raise RuntimeError("scheduler exploded")

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=pull,
        max_cohort_rows=4,
    )
    assert res.streams[0].tokens == _expected_chain(prompts[0], 4)
    assert res.streams[1].tokens == _expected_chain(prompts[1], 4)


def test_fixed_shape_padding_and_demand_following_are_refused_together() -> None:
    """They mean opposite things, so accepting both would serve one silently."""

    with pytest.raises(ValueError, match="cannot both be set"):
        generate_dense_mtp_batch(
            _FakeRuntime(width=256),
            _distinct_prompts(2),
            max_new_tokens=2,
            depth=3,
            cohort_slots=8,
            pull_queued=lambda capacity: [],
        )


def test_growth_stops_at_the_memory_budget_instead_of_running_out() -> None:
    """Found by hammering: a burst of long prompts OOM'd the GPU at width 8.

    The failure was contained -- every caller got the exception and the server
    kept serving -- but a caller still lost its request, and a raw Metal
    "Insufficient Memory" names nothing an operator controls. Growth is what
    this session added, so growth is what gets a budget: a row that will not
    fit waits for capacity instead of taking the cohort down.

    An impossible headroom stands in for a full device, so the test asserts the
    MECHANISM rather than a machine-specific number.
    """

    pull, _ = _live_queue(
        [{"prompt": p, "max_new_tokens": 4} for p in _distinct_prompts(4)[1:]]
    )
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(1),
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=pull,
        max_cohort_rows=4,
        memory_headroom=1e-9,
    )
    assert res.meta["rows_blocked_by_memory"] > 0, (
        "the budget admitted rows it had no room for"
    )
    assert res.meta["rows_peak"] < 4, (
        "the cohort grew to its full ceiling against an exhausted budget"
    )
    # NOBODY IS STRANDED. A budget that throttles growth is a feature; one that
    # strands callers is the same outage wearing a politer message. The lane
    # must still serve every request, just more slowly and in narrower groups.
    assert len(res.streams) == 4
    for stream in res.streams:
        assert stream.finish_reason != "not_admitted", stream.index
        assert stream.tokens, stream.index


def test_the_memory_budget_is_off_when_asked_and_admits_everything() -> None:
    """A guard that cannot be turned off is a guard that will one day be wrong.

    It is an estimate over an allocator it does not control, so an operator who
    finds it too conservative must be able to disable it rather than patch it.
    """

    pull, _ = _live_queue(
        [{"prompt": p, "max_new_tokens": 4} for p in _distinct_prompts(3)[1:]]
    )
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(1),
        max_new_tokens=4,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
        pull_queued=pull,
        max_cohort_rows=3,
        memory_headroom=0.0,
    )
    assert res.meta["rows_blocked_by_memory"] == 0
    assert res.meta["rows_peak"] == 3


def test_slow_cycles_are_recorded_when_a_threshold_is_set(monkeypatch) -> None:
    """Instrumentation for the unexplained 110-144 second pauses.

    Two hypotheses for those pauses were tested in an earlier session and both
    died, which is the situation where you stop guessing and start recording.
    The existing phase timing cannot be used: it inserts an `mx.eval` per phase,
    so leaving it on changes the thing being measured -- which is very likely
    why the pauses were never caught in the act.

    This costs one `perf_counter()` and a comparison per cycle, adds no device
    sync, and can therefore be left on for a soak. A threshold of zero seconds
    makes every cycle an outlier, which is how the recording is tested without
    needing a slow one.
    """

    monkeypatch.setenv("MTPLX_DENSE_BATCH_CYCLE_WARN_S", "0.000001")
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=3,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
    )
    slow = res.meta["slow_cycles"]
    assert slow, "no cycle was recorded against a threshold of one microsecond"
    first = slow[0]
    for field in (
        "cycle",
        "elapsed_s",
        "rows",
        "live_rows",
        "admission_boundary",
        "queued_unadmitted",
        "rows_blocked_by_memory",
    ):
        assert field in first, field
    assert first["rows"] == 2
    # The whole point: a pause becomes a fact with a width and a queue depth
    # attached, rather than "the cohort went quiet".
    assert isinstance(first["admission_boundary"], bool)


def test_slow_cycle_recording_is_off_by_default() -> None:
    """It must cost nothing unless asked for, or nobody will leave it on."""

    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        _distinct_prompts(2),
        max_new_tokens=3,
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
    )
    assert res.meta["slow_cycles"] == []


# --------------------------------------------------------------------------- #
# T-210: prefix reuse through SessionBank.
#
# The tests that used to live here exercised a `PrefixStore` that no longer
# exists, and they are not simply ported, because they were the wrong tests.
# Every one passed against token sequences built to nest inside each other,
# which is what the mechanism needed and NOT what a real chat template
# produces. They then passed again while the bank path stored 352 MB and served
# zero restores, and again while a restore was 88% faster and gave a DIFFERENT
# answer six tokens in.
#
# What is checked here is what a deterministic fake can honestly check: that
# the lane fails CLOSED. Whether reuse is correct and worth having is settled
# on real weights, against a real tokenizer applying a real chat template --
# see the T-210 evidence folder. That is where all four defects were found.
# --------------------------------------------------------------------------- #
def test_no_bank_means_no_reuse_and_no_bookkeeping() -> None:
    """The default path must be untouched by any of this."""

    rt = _FakeRuntime(width=256)
    res = generate_dense_mtp_batch(
        rt, [[1, 2, 3, 4, 5, 6, 7, 8]], max_new_tokens=4, depth=3,
        ragged_prompts=True, stop_token_ids=set(),
    )
    assert res.meta["prefix_restores"] == 0
    assert res.meta["prefix_restore_failures"] == 0
    assert res.meta["prefix_entries_stored"] == 0


def test_an_empty_bank_costs_a_lookup_and_nothing_else() -> None:
    """A miss is a miss, not a failure -- the two must not be conflated.

    Counting a cold-start miss as a failure would make a healthy first request
    look like a broken one, and a real failure invisible among them.
    """

    from mtplx.session_bank import SessionBank

    rt = _FakeRuntime(width=256)
    bank = SessionBank()
    res = generate_dense_mtp_batch(
        rt, [[1, 2, 3, 4, 5, 6, 7, 8]], max_new_tokens=4, depth=3,
        ragged_prompts=True, stop_token_ids=set(), session_bank=bank,
    )
    assert res.meta["prefix_restores"] == 0
    assert res.meta["prefix_restore_failures"] == 0


def test_a_bank_that_raises_does_not_take_the_request_down() -> None:
    """Reuse is an optimisation. It must never be why a request fails."""

    from mtplx.session_bank import SessionBank

    class _Exploding(SessionBank):
        def near_prefix_candidates(self, *a, **k):  # noqa: D102
            raise RuntimeError("bank is having a bad day")

        def restore(self, *a, **k):  # noqa: D102
            raise RuntimeError("bank is having a bad day")

    rt = _FakeRuntime(width=256)
    res = generate_dense_mtp_batch(
        rt, [[1, 2, 3, 4, 5, 6, 7, 8]], max_new_tokens=4, depth=3,
        ragged_prompts=True, stop_token_ids=set(), session_bank=_Exploding(),
    )
    assert res.meta["prefix_restores"] == 0
    assert len(res.streams) == 1
    assert len(res.streams[0].tokens) == 4


def test_boundaries_are_capped_so_a_long_prompt_cannot_hoard_memory(monkeypatch) -> None:
    """One recurrent boundary is 49 MB on the 4B and more on the 27B.

    Left uncapped, a ladder over a long prompt is hundreds of megabytes for a
    single entry -- which is how a cache becomes the memory leak it was meant
    to avoid.
    """

    from mtplx.session_bank import SessionBank

    # Stride down so a prompt the fake can actually hold still produces more
    # boundary positions than the cap allows.
    monkeypatch.setenv("MTPLX_DENSE_PREFIX_LADDER_STRIDE", "16")
    # The fake writes the whole running history into each hidden vector, so its
    # width must exceed everything this request will ever have seen.
    rt = _FakeRuntime(width=512)
    bank = SessionBank()
    generate_dense_mtp_batch(
        rt, [list(range(1, 201))], max_new_tokens=2, depth=3,
        ragged_prompts=True, stop_token_ids=set(), session_bank=bank,
        prefill_chunk=16,
    )
    # The invariant is on boundaries CAPTURED for this prefill, not on how many
    # an entry lists: an entry inherits its donor's boundaries as shared
    # references, which cost no additional memory. This test passed while
    # asserting the wrong thing because a single prompt has no donor to inherit
    # from -- a stress run with a donor present is what caught it.
    for entry in bank._entries.values():
        owned = [b for b in (entry.gdn_boundaries or []) if int(b[0]) <= entry.prefix_len]
        assert len(owned) <= 4, (
            f"{len(owned)} boundaries captured for one prefill; the cap is 4"
        )


def test_a_plain_call_with_no_queue_never_resizes_its_row_axis() -> None:
    """The compatibility guarantee: every pre-item-4 caller keeps its path.

    A resize changes batch geometry, and geometry changes a row's tokens. A
    library caller who asked for none of this must not silently get different
    output because rows now leave the batch when they finish.
    """

    prompts = _distinct_prompts(3)
    res = generate_dense_mtp_batch(
        _FakeRuntime(width=256),
        prompts,
        max_new_tokens=6,
        max_new_tokens_per_row=[1, 3, 6],
        depth=3,
        stop_token_ids={STOP_ID},
        ragged_prompts=True,
        loop_mode="serial",
    )
    assert res.meta["continuous"] is False
    assert res.meta["cohort_resizes"] == 0
    assert res.meta["rows_peak"] == 3
    assert res.meta["rows_final"] == 3, "rows must stay put without a queue"
