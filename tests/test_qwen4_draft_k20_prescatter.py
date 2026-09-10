"""MTPLX_QWEN4_DRAFT_K20_PRESCATTER -- equivalence and construction proofs.

The lane claims the draft K20 support built from the FR-Spec head's 65,536-row
output is the SAME support the shipped builder produces from the 248,320-wide
scattered row.  These tests hold the two next to each other on random rows and
on rows with constructed exact ties, and compare RAW BITS (through an integer
view, so a one-ulp drift or a sign-flipped zero fails).

They run entirely on the CPU stream: no Metal, no model, no kernels.  The one
claim that is stream-dependent -- whether the two DIFFERENT-WIDTH reductions
behind ``mx.logsumexp`` associate their float32 partial sums identically -- is
pinned here on the CPU stream and re-measured on the Metal stream by
``the PR #391 harness micro_draft_k20.py``, which prints the differing-row counters
instead of assuming.  See ``mtplx/qwen4_draft_k20_prescatter`` for why a
residual ULP there could not change the SUPPORT.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import re
from types import SimpleNamespace

import mtplx.qwen4_claim_contract as contract
import mtplx.qwen4_draft_k20_prescatter as prescatter
import mtplx.fast_sampling as fast_sampling
import mtplx.frspec_draft as frspec_draft
from mtplx.qwen4_draft_k20_prescatter import (
    DraftK20PrescatterIneligible,
    DraftK20PrescatterPlan,
)
from mtplx.sampling import SamplerConfig, sample_from_distribution


TOP_K = 20
TEMPERATURE = 0.6
TOP_P = 0.95


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Confine every op in this module to the CPU stream."""

    with mx.stream(mx.cpu):
        yield


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ranked_ids(rows: int, vocab_rows: int, seed: int) -> np.ndarray:
    """A strictly ascending ranked table -- the shipped artifact's shape."""

    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(vocab_rows, size=rows, replace=False)
    ).astype(np.int64)


def _plan(
    rows: int,
    vocab_rows: int,
    seed: int,
    head=None,
    *,
    greedy: bool = False,
) -> DraftK20PrescatterPlan:
    ids_np = _ranked_ids(rows, vocab_rows, seed)
    return DraftK20PrescatterPlan(
        head=head,
        ids_np=ids_np,
        ids_mx=mx.array(ids_np, dtype=mx.int32),
        rows=rows,
        vocab_rows=vocab_rows,
        top_k=0 if greedy else TOP_K,
        temperature=0.0 if greedy else TEMPERATURE,
        top_p=TOP_P,
        route="native_mtp_head",
        greedy=greedy,
    )


def _compact_row(rows: int, seed: int, *, dtype=mx.float32) -> mx.array:
    """A logit row shaped like a real one: a heavy head, a long tail."""

    rng = np.random.default_rng(seed)
    base = (rng.standard_normal(rows) * 2.0).astype(np.float32)
    peaks = rng.choice(rows, size=min(64, rows), replace=False)
    base[peaks] += rng.uniform(6.0, 14.0, size=peaks.shape[0]).astype(np.float32)
    return mx.array(base).astype(dtype)


def _scatter(compact: mx.array, plan: DraftK20PrescatterPlan) -> mx.array:
    """Exactly ``frspec_draft._FullVocabDraftHead.__call__``'s scatter."""

    subset = compact.reshape(1, -1)
    output = mx.full((1, plan.vocab_rows), -1.0e30, dtype=subset.dtype)
    index = mx.broadcast_to(
        mx.array(plan.ids_np, dtype=mx.int32).reshape(1, -1), subset.shape
    )
    return mx.put_along_axis(output, index, subset, axis=-1).reshape(-1)


def _config(*, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K) -> SamplerConfig:
    return SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k)


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    assert a.dtype == b.dtype
    width = {4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    return bool(np.array_equal(a.view(width), b.view(width)))


# ---------------------------------------------------------------------------
# 1. The sentinel contributes exactly zero
# ---------------------------------------------------------------------------


def test_sentinel_underflows_to_exact_zero_in_float32():
    """``exp(sentinel/T - max)`` is exactly +0.0, so the sum is unchanged."""

    sentinel = mx.array([-1.0e30], dtype=mx.float32) * (1.0 / TEMPERATURE)
    assert bool(mx.isfinite(sentinel).item()), "the pad must not be an inf"
    for peak in (-50.0, 0.0, 40.0, 1e4):
        term = mx.exp(sentinel - mx.array([peak], dtype=mx.float32))
        value = np.asarray(term, dtype=np.float32)[0]
        assert value == np.float32(0.0)
        # +0.0, not -0.0: adding it cannot even flip a signed zero.
        assert value.view(np.uint32) == np.uint32(0)
    # x + 0.0 == x exactly, for the whole float32 range the sum can hold.
    for magnitude in (1e-38, 1.0, 3.14159, 1e30):
        x = np.float32(magnitude)
        assert (x + np.float32(0.0)).view(np.uint32) == x.view(np.uint32)


def test_sentinel_survives_a_bfloat16_head_output():
    """A bf16 head row keeps the pad finite and far below every real logit."""

    pad = mx.array([-1.0e30], dtype=mx.bfloat16)
    assert bool(mx.isfinite(pad).item())
    widened = np.asarray(pad.astype(mx.float32), dtype=np.float32)[0]
    assert widened < np.float32(-9.9e29)
    term = mx.exp(pad.astype(mx.float32) * (1.0 / TEMPERATURE) - mx.array([0.0]))
    assert np.asarray(term, dtype=np.float32)[0].view(np.uint32) == np.uint32(0)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_full_vocab_logsumexp_equals_compact_logsumexp(seed):
    """The normaliser the pre-scatter row computes is the full row's."""

    plan = _plan(65_536, 248_320, seed)
    compact = _compact_row(plan.rows, seed + 900)
    dense = _scatter(compact, plan)
    scale = 1.0 / TEMPERATURE
    compact_lse = mx.logsumexp(compact.astype(mx.float32) * scale, axis=-1)
    dense_lse = mx.logsumexp(dense.astype(mx.float32) * scale, axis=-1)
    mx.eval(compact_lse, dense_lse)
    assert _same_bits(
        np.asarray(compact_lse, dtype=np.float32),
        np.asarray(dense_lse, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# 2. Support equivalence -- stock builder on the dense row vs pre-scatter
# ---------------------------------------------------------------------------


def _assert_support_equivalent(plan, compact, config):
    dense = _scatter(compact, plan)
    want_ids, want_probs, want_vocab = fast_sampling._device_serial_support_arrays(
        dense.astype(mx.float32), config
    )
    got_ids, got_probs, got_vocab = prescatter.prescatter_serial_support_arrays(
        plan, compact.astype(mx.float32), config
    )
    assert got_vocab == want_vocab == plan.vocab_rows
    assert np.array_equal(got_ids, want_ids), (want_ids[0][:6], got_ids[0][:6])
    assert _same_bits(got_probs, want_probs)
    return want_ids, want_probs


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_support_matches_stock_at_production_shape(seed):
    plan = _plan(65_536, 248_320, seed)
    compact = _compact_row(plan.rows, seed + 7)
    _assert_support_equivalent(plan, compact, _config())


@pytest.mark.parametrize("top_p", [0.95, 1.0, 0.0])
def test_support_matches_stock_across_top_p_branches(top_p):
    """``top_p`` in (0,1) takes the logsumexp branch; 1.0/0.0 take softmax."""

    plan = _plan(8_192, 32_768, 55)
    compact = _compact_row(plan.rows, 56)
    _assert_support_equivalent(plan, compact, _config(top_p=top_p))


def test_support_matches_stock_with_exact_ties_inside_the_support():
    """Exact float ties inside the top-20 must break the same way."""

    plan = _plan(8_192, 32_768, 71)
    compact = np.asarray(_compact_row(plan.rows, 72), dtype=np.float32)
    order = np.argsort(-compact)
    # Flatten ranks 3..8 onto one value: six exact ties well inside the top-20.
    compact[order[3:9]] = compact[order[3]]
    # And a second tie group straddling ranks 18..22, i.e. the k-th cutoff.
    compact[order[18:23]] = compact[order[18]]
    _assert_support_equivalent(plan, mx.array(compact), _config())


def test_support_matches_stock_with_a_cutoff_tie_beyond_the_superset():
    """The stock spill fallback and the pre-scatter one agree.

    ``_device_serial_support_arrays`` re-derives with the deterministic device
    selector when the cutoff tie group may reach past the 4k candidate
    superset.  A flat row makes every lane tie, which forces that branch on
    both rows (the condition reads only the sorted candidate VALUES, which are
    identical).
    """

    plan = _plan(4_096, 16_384, 83)
    compact = mx.full((plan.rows,), 1.25, dtype=mx.float32)
    config = _config()
    dense = _scatter(compact, plan)
    scaled_dense = dense.astype(mx.float32) * (1.0 / TEMPERATURE)
    cand = mx.argpartition(-scaled_dense.reshape(1, -1), kth=79, axis=-1)[:, :80]
    vals = mx.take_along_axis(scaled_dense.reshape(1, -1), cand, axis=-1)
    mx.eval(vals)
    values = np.asarray(vals, dtype=np.float32)
    assert np.nanmin(values, axis=1)[0] >= values[0, TOP_K - 1], "spill not armed"
    _assert_support_equivalent(plan, compact, config)


def test_support_matches_stock_from_a_bfloat16_head_row():
    plan = _plan(8_192, 32_768, 91)
    compact = _compact_row(plan.rows, 92, dtype=mx.bfloat16)
    _assert_support_equivalent(plan, compact, _config())


# ---------------------------------------------------------------------------
# 3. Distribution + sampled token equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [401, 402])
def test_distribution_and_draw_match_the_stock_draft_read(seed):
    plan = _plan(65_536, 248_320, seed)
    compact = _compact_row(plan.rows, seed + 3)
    dense = _scatter(compact, plan)
    config = _config()

    want = fast_sampling.sparse_distribution_from_mlx_logits(
        dense.astype(mx.float32), config
    )
    got = prescatter.prescatter_sparse_distribution(
        plan, compact.astype(mx.float32), config
    )
    assert want is not None
    assert got.vocab_size == want.vocab_size == plan.vocab_rows
    assert np.array_equal(got.token_ids, want.token_ids)
    assert _same_bits(
        np.asarray(got.probs, dtype=np.float64),
        np.asarray(want.probs, dtype=np.float64),
    )
    # And the same one rng draw, from the same stream position.
    assert sample_from_distribution(
        got, np.random.default_rng(5150)
    ) == sample_from_distribution(want, np.random.default_rng(5150))


def test_greedy_read_matches_the_stock_argmax():
    plan = _plan(8_192, 32_768, 611)
    compact = _compact_row(plan.rows, 612)
    dense = _scatter(compact, plan)
    config = _config(temperature=0.0)
    head = _FakeHead(dense, compact)
    plan = DraftK20PrescatterPlan(**{**plan.__dict__, "head": head})

    token, distribution = prescatter.read_draft(
        plan, dense, config, np.random.default_rng(1), need_distribution=True
    )
    assert token == int(mx.argmax(dense.astype(mx.float32), axis=-1).item())
    assert distribution.vocab_size == plan.vocab_rows
    assert list(distribution.token_ids) == [token]


def test_greedy_argmax_tie_picks_the_lowest_real_id():
    """``argmax``'s lowest-LOCAL-index tie-break maps to the lowest real id."""

    plan = _plan(1_024, 4_096, 613)
    compact = np.full(plan.rows, -3.0, dtype=np.float32)
    compact[[17, 900]] = 9.0  # an exact tie between two local rows
    compact = mx.array(compact)
    dense = _scatter(compact, plan)
    head = _FakeHead(dense, compact)
    plan = DraftK20PrescatterPlan(**{**plan.__dict__, "head": head})
    token, _ = prescatter.read_draft(
        plan, dense, _config(temperature=0.0), None, need_distribution=False
    )
    assert token == int(plan.ids_np[17])
    assert token == int(mx.argmax(dense.astype(mx.float32), axis=-1).item())


# ---------------------------------------------------------------------------
# 3b. The greedy DEVICE chain read (`generate_mtpk`'s one-sync greedy path)
# ---------------------------------------------------------------------------


def _greedy_plan(rows, vocab_rows, seed, row=None, *, dtype=mx.float32):
    """A greedy plan plus the (dense, compact) pair its head would stash."""

    plan = _plan(rows, vocab_rows, seed, greedy=True)
    compact = _compact_row(rows, seed + 1, dtype=dtype) if row is None else row
    dense = _scatter(compact, plan)
    plan = DraftK20PrescatterPlan(
        **{**plan.__dict__, "head": _FakeHead(dense, compact)}
    )
    return plan, compact, dense


@pytest.mark.parametrize("seed", [701, 702, 703, 704])
def test_greedy_chain_step_token_is_the_stock_dense_argmax(seed):
    """The whole point: same token, from a row 3.8x narrower."""

    plan, _, dense = _greedy_plan(prescatter.FRSPEC_ROWS, 248_320, seed)
    token, confidence = prescatter.greedy_chain_step(
        plan, dense, want_confidence=False
    )
    assert confidence is None
    assert int(token.item()) == int(mx.argmax(dense, axis=-1).item())


def test_greedy_chain_step_returns_unevaluated_device_arrays():
    """It must add NO sync: the chain's own single `mx.eval` covers it."""

    plan, _, dense = _greedy_plan(4_096, 16_384, 711)
    token, confidence = prescatter.greedy_chain_step(
        plan, dense, want_confidence=True
    )
    assert isinstance(token, mx.array)
    assert isinstance(confidence, mx.array)
    # The id map is a device gather, so the token carries the table's dtype
    # and reshapes into the chain's next input token without a host round trip.
    assert token.dtype == mx.int32
    assert tuple(token.reshape(1, 1).shape) == (1, 1)


def test_greedy_chain_step_tie_picks_the_lowest_real_id():
    """``argmax``'s lowest-LOCAL-index tie-break maps to the lowest real id.

    The map is strictly increasing, so it preserves the tie-break itself --
    not merely produces an equally valid winner.
    """

    rows, vocab_rows = 1_024, 4_096
    row = np.full(rows, -3.0, dtype=np.float32)
    row[[17, 900, 1_000]] = 9.0  # three-way exact tie at the maximum
    plan, _, dense = _greedy_plan(rows, vocab_rows, 713, row=mx.array(row))
    token, _ = prescatter.greedy_chain_step(plan, dense, want_confidence=False)
    assert int(token.item()) == int(plan.ids_np[17])
    assert int(token.item()) == int(mx.argmax(dense, axis=-1).item())


def test_greedy_chain_step_from_a_bfloat16_head_row():
    """The sentinel is finite in bfloat16 (-9.9964e29), not an inf or a NaN."""

    plan, _, dense = _greedy_plan(8_192, 32_768, 715, dtype=mx.bfloat16)
    assert dense.dtype == mx.bfloat16
    token, _ = prescatter.greedy_chain_step(plan, dense, want_confidence=False)
    assert int(token.item()) == int(mx.argmax(dense, axis=-1).item())


@pytest.mark.parametrize("seed", [721, 722])
def test_greedy_chain_step_confidence_agrees_but_not_to_the_bit(seed):
    """The sentinel contributes nothing; the REDUCTION SHAPE still costs ULP.

    ``max`` is bit-identical -- which is what settles the token.  The
    ``logsumexp`` behind the confidence is the same real number computed from
    the same terms, but 65,536 partial sums associate differently from
    248,320, so the pair lands ~2e-6 apart.  That is why
    ``claim_draft_route`` routes confidence-tracing requests to the stock
    reader instead of shaving a step and moving a printed number.
    """

    plan, _, dense = _greedy_plan(prescatter.FRSPEC_ROWS, 248_320, seed)
    _, confidence = prescatter.greedy_chain_step(
        plan, dense, want_confidence=True
    )
    stock = mx.exp(mx.max(dense) - mx.logsumexp(dense))
    mx.eval(confidence, stock)
    got = float(np.asarray(confidence, dtype=np.float32))
    want = float(np.asarray(stock, dtype=np.float32))
    assert got == pytest.approx(want, rel=1e-5)
    # Close, but NOT the same bits -- that is the whole point of the routing.
    assert 0.0 < abs(got - want) < 1e-5 * want or got == want


def test_the_max_behind_the_greedy_confidence_is_bit_identical(seed=723):
    """``max`` -- unlike the sum -- is exact on both rows, so the token is."""

    plan, compact, dense = _greedy_plan(prescatter.FRSPEC_ROWS, 248_320, seed)
    a = mx.max(compact.reshape(-1))
    b = mx.max(dense)
    mx.eval(a, b)
    assert _same_bits(
        np.asarray(a, dtype=np.float32).reshape(1),
        np.asarray(b, dtype=np.float32).reshape(1),
    )


def test_a_confidence_tracing_request_is_routed_to_the_stock_reader(armed):
    """The routing that keeps the ULP above out of any receipt."""

    rt, head, _ = _runtime()
    contract.reset_for_test()
    receipt: dict[str, object] = {}
    plan = prescatter.claim_draft_route(
        rt,
        draft_sampler=_config(temperature=0.0, top_k=0),
        receipt=receipt,
        greedy_chain_enabled=True,
        **{**_ELIGIBLE, "draft_confidence_needed": True},
    )
    assert plan is None
    assert receipt["declined"] == "draft_confidence"
    assert head._prescatter_capture is False


def test_greedy_chain_step_refuses_a_stale_stash():
    """A head that changed mid-request must not silently score another step."""

    plan, _, _ = _greedy_plan(4_096, 16_384, 731)
    other = mx.zeros((1, 1, plan.vocab_rows))
    with pytest.raises(DraftK20PrescatterIneligible, match="did not capture"):
        prescatter.greedy_chain_step(plan, other, want_confidence=False)


def test_a_whole_greedy_chain_matches_the_stock_chain_token_for_token():
    """Three depths, one `mx.eval`, exactly as `generate_mtpk` runs it.

    Mirrors the chain's shape: each depth feeds the previous depth's token
    back in, everything stays lazy, and the single eval at the end is the only
    sync.  The stock arm runs the same rows through `mx.argmax(dense)`.
    """

    rows, vocab_rows = prescatter.FRSPEC_ROWS, 248_320
    plan = _plan(rows, vocab_rows, 741, greedy=True)
    compacts = [_compact_row(rows, 742 + d) for d in range(3)]
    denses = [_scatter(c, plan) for c in compacts]

    pending, feed = [], []
    for compact, dense in zip(compacts, denses):
        plan = DraftK20PrescatterPlan(
            **{**plan.__dict__, "head": _FakeHead(dense, compact)}
        )
        token, _ = prescatter.greedy_chain_step(
            plan, dense, want_confidence=False
        )
        pending.append(token)
        # what the chain feeds the next depth
        feed.append(token.reshape(1, 1).astype(mx.int32))
    mx.eval(*pending, *feed)  # the chain's ONE sync

    stock = [int(mx.argmax(dense, axis=-1).item()) for dense in denses]
    assert [int(t.item()) for t in pending] == stock
    assert [int(f.reshape(-1)[0].item()) for f in feed] == stock


def test_the_greedy_chain_call_site_uses_the_plan():
    """`generate_mtpk`'s greedy chain reads the compact row when armed."""

    import inspect

    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    assert "_qwen4_draft_k20_prescatter_greedy_step(" in source
    # ...and still has the stock chain for a flag-off / declined request.
    assert "_chain_arg = mx.argmax(_chain_row, axis=-1)" in source


def test_sampled_read_returns_the_row_it_sampled_from():
    """The sampled branch always returns q, as ``_sample_draft_from_logits``
    does (it tail-calls ``_sample_from_logits``, ignoring need_distribution)."""

    plan = _plan(4_096, 16_384, 621)
    compact = _compact_row(plan.rows, 622)
    dense = _scatter(compact, plan)
    head = _FakeHead(dense, compact)
    plan = DraftK20PrescatterPlan(**{**plan.__dict__, "head": head})
    _, distribution = prescatter.read_draft(
        plan, dense, _config(), np.random.default_rng(3), need_distribution=False
    )
    assert distribution is not None
    assert distribution.vocab_size == plan.vocab_rows


class _FakeHead:
    """The head surface ``take_compact_row`` needs, without a model."""

    def __init__(self, dense, compact):
        self._dense = dense
        self._compact = compact
        self.armed = False

    def arm_prescatter_capture(self, enabled):
        self.armed = bool(enabled)

    def take_prescatter_row(self, dense):
        if dense is not self._dense:
            return None
        return self._compact


def test_take_compact_row_refuses_a_stale_stash():
    plan = _plan(4_096, 16_384, 631)
    compact = _compact_row(plan.rows, 632)
    dense = _scatter(compact, plan)
    plan = DraftK20PrescatterPlan(
        **{**plan.__dict__, "head": _FakeHead(dense, compact)}
    )
    other = _scatter(compact, plan)
    with pytest.raises(DraftK20PrescatterIneligible, match="did not capture"):
        prescatter.take_compact_row(plan, other)


# ---------------------------------------------------------------------------
# 4. The head's capture surface
# ---------------------------------------------------------------------------


def _full_vocab_head(rows: int, vocab_rows: int, seed: int):
    import mlx.nn as nn

    ids = mx.array(_ranked_ids(rows, vocab_rows, seed), dtype=mx.int32)
    inner = nn.Linear(8, rows, bias=False)
    return frspec_draft._full_vocab_head(inner, ids, vocab_rows), ids


def test_head_capture_is_disarmed_by_default():
    head, _ = _full_vocab_head(256, 1_024, 7)
    x = mx.zeros((1, 1, 8))
    dense = head(x)
    assert int(dense.shape[-1]) == 1_024
    assert head.take_prescatter_row(dense) is None


def test_head_capture_stashes_the_compact_row_and_clears_on_take():
    head, _ = _full_vocab_head(256, 1_024, 8)
    head.arm_prescatter_capture(True)
    x = mx.random.normal((1, 1, 8))
    dense = head(x)
    row = head.take_prescatter_row(dense)
    assert row is not None
    assert int(row.reshape(-1).shape[0]) == 256
    # Consumed: a second take cannot silently succeed.
    assert head.take_prescatter_row(dense) is None
    # And the compact row really is the pre-scatter one.
    assert _same_bits(
        np.asarray(row.reshape(-1), dtype=np.float32),
        np.asarray(mx.take(dense.reshape(-1), head._ids), dtype=np.float32),
    )


def test_head_capture_identity_rejects_another_call_dense_row():
    head, _ = _full_vocab_head(256, 1_024, 9)
    head.arm_prescatter_capture(True)
    first = head(mx.random.normal((1, 1, 8)))
    head(mx.random.normal((1, 1, 8)))
    assert head.take_prescatter_row(first) is None


def test_head_disarm_drops_the_stash():
    head, _ = _full_vocab_head(256, 1_024, 10)
    head.arm_prescatter_capture(True)
    dense = head(mx.random.normal((1, 1, 8)))
    head.arm_prescatter_capture(False)
    assert head.take_prescatter_row(dense) is None


# ---------------------------------------------------------------------------
# 5. Construction-bound install / refusal
# ---------------------------------------------------------------------------


_ELIGIBLE = dict(
    draft_core="stock",
    target_prefix_verify=False,
    a3b_target_prefix_route=None,
    pr391_route=None,
    device_k20_route=None,
    frspec_legacy_ids=None,
    adaptive_width_policy=None,
    combine_greedy_draft_read=False,
    draft_confidence_needed=False,
    draft_margin_threshold=None,
    wants_policy_metrics=False,
    correction_cache_enabled=False,
    adapter_ensemble_q=False,
    mtp_topk_reranker=None,
    relaxed_draft_ties=False,
    penalties_active=False,
    steer_active=False,
)


class _TextModelStandIn:
    """``models/qwen4_exp.TextModel``'s draft surface, on a plain object.

    Only the three members the FR-Spec install and this module's liveness
    probe touch: the draft-head hook ``mtp_forward`` projects through, the
    default binding ``TextModel.__init__`` puts there, and the bind hook the
    install calls.  Nothing here evaluates.
    """

    def __init__(self) -> None:
        # qwen4_exp.TextModel.__init__
        self._mtp_draft_head_logits = self._head_logits

    def _head_logits(self, h):  # pragma: no cover - never called in these tests
        raise AssertionError("the unpruned full-vocab projection was called")

    def _mtplx_bind_draft_lm_head(self, head) -> None:
        # qwen4_exp.TextModel._mtplx_bind_draft_lm_head
        self._mtp_draft_head_logits = head.__call__


def _configured_head():
    """The unpruned quantized head the install leaves at the legacy stamp."""

    import mlx.nn as nn

    return nn.QuantizedLinear(64, 32, bias=False, group_size=64, bits=8)


def _installed_text_model(head, *, route="native_mtp_head"):
    """A text model shaped exactly as ``install_frspec_draft_head`` leaves it.

    ``native_mtp_head`` is the shipped ``--full-frspec`` lane (install report
    ``source: native_mtp_head``, ``legacy_swap: False``): the wrapper is
    published at ``_mtplx_frspec_draft_head`` and bound onto the native MTP
    draft-head hook, while ``_mtplx_draft_lm_head`` KEEPS the unpruned
    ``QuantizedLinear`` for legacy consumers.  ``legacy_swap`` is
    ``MTPLX_FRSPEC_LEGACY=1`` on a generic ``mtp_patch`` model: no bind hook,
    the wrapper swapped in globally.  ``none`` is an install whose wrapper
    never reached either route.
    """

    text = _TextModelStandIn()
    text._mtplx_frspec_draft_head = head
    text._mtplx_frspec_full_vocab = int(head._vocab_rows)
    text._mtplx_frspec_ids = head._ids
    text._mtplx_draft_lm_head = _configured_head()
    if route == "native_mtp_head":
        text._mtplx_bind_draft_lm_head(head)
    elif route == "legacy_swap":
        text._mtplx_frspec_saved_head = text._mtplx_draft_lm_head
        text._mtplx_draft_lm_head = head
    elif route != "none":  # pragma: no cover - test typo guard
        raise AssertionError(f"unknown route {route!r}")
    return text


def _runtime(
    rows=prescatter.FRSPEC_ROWS,
    vocab_rows=248_320,
    *,
    ascending=True,
    route="native_mtp_head",
):
    head, ids = _full_vocab_head(rows, vocab_rows, 4242)
    if not ascending:
        shuffled = np.asarray(ids, dtype=np.int64)[::-1].copy()
        object.__setattr__(head, "_ids", mx.array(shuffled, dtype=mx.int32))
    text = _installed_text_model(head, route=route)
    return SimpleNamespace(model=SimpleNamespace(language_model=text)), head, text


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(prescatter, "_ENABLED", True)
    return True


def test_claim_returns_none_when_the_flag_is_off():
    rt, head, _ = _runtime(rows=256, vocab_rows=1_024)
    assert prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE) is None
    assert head._prescatter_capture is False


@pytest.mark.parametrize("route", ["native_mtp_head", "legacy_swap"])
def test_claim_installs_and_arms_at_the_production_width(armed, route):
    rt, head, text = _runtime(route=route)
    plan = prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)
    assert plan is not None
    assert plan.rows == prescatter.FRSPEC_ROWS
    assert plan.vocab_rows == 248_320
    assert plan.head is head
    assert plan.route == route
    assert head._prescatter_capture is True
    receipt = plan.to_dict()
    assert receipt["installed"] is True
    assert receipt["rows"] == prescatter.FRSPEC_ROWS
    assert receipt["route"] == route
    prescatter.release_draft_route(plan)
    assert head._prescatter_capture is False


def test_claim_succeeds_while_the_legacy_stamp_holds_a_quantized_head(armed):
    """The shipped lane's shape: live via the bind hook, legacy stamp unpruned.

    This is the exact structure the 2026-09-02 ABBA candidate arm refused on
    (``live=QuantizedLinear``) while its own load log said the FR-Spec head was
    installed in full-vocabulary output mode.
    """

    rt, head, text = _runtime(route="native_mtp_head")
    assert type(text._mtplx_draft_lm_head).__name__ == "QuantizedLinear"
    assert text._mtplx_draft_lm_head is not head
    assert text._mtp_draft_head_logits.__self__ is head
    plan = prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)
    assert plan is not None and plan.route == "native_mtp_head"
    prescatter.release_draft_route(plan)


def test_claim_refuses_a_non_frspec_width(armed):
    rt, _, _ = _runtime(rows=4_096, vocab_rows=16_384)
    with pytest.raises(DraftK20PrescatterIneligible, match="proven only at"):
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)


def test_claim_refuses_a_non_ascending_ranked_table(armed):
    rt, _, _ = _runtime(ascending=False)
    with pytest.raises(DraftK20PrescatterIneligible, match="strictly ascending"):
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)


def test_claim_refuses_without_an_frspec_head(armed):
    rt = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace()))
    with pytest.raises(DraftK20PrescatterIneligible, match="no FR-Spec draft head"):
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)


def test_claim_refuses_when_the_frspec_head_is_not_live(armed):
    """Published but on neither route: no bind hook fired, no legacy swap."""

    rt, head, text = _runtime(route="none")
    with pytest.raises(
        DraftK20PrescatterIneligible, match="not on the live draft route"
    ) as excinfo:
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)
    message = str(excinfo.value)
    # The refusal names what each probe actually saw.
    assert "_mtp_draft_head_logits=_TextModelStandIn._head_logits" in message
    assert "_mtplx_draft_lm_head=QuantizedLinear" in message
    assert head._prescatter_capture is False


def test_claim_refuses_when_another_head_owns_the_native_hook(armed):
    """The hook is live, but bound to somebody else's head."""

    rt, head, text = _runtime(route="none")
    other, _ = _full_vocab_head(256, 1_024, 99)
    text._mtplx_bind_draft_lm_head(other)
    with pytest.raises(
        DraftK20PrescatterIneligible, match="not on the live draft route"
    ) as excinfo:
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)
    assert "_FullVocabDraftHead.__call__" in str(excinfo.value)
    assert head._prescatter_capture is False
    assert other._prescatter_capture is False


def test_the_real_install_produces_the_route_the_claim_probes():
    """Pin the probe to ``install_frspec_draft_head``'s ACTUAL output.

    Runs the shipped install at a toy width against the qwen4 draft surface,
    so a future change to how the wrapper is made live fails here rather than
    at the next benchmark.  No model, no Metal: a 1,024-row toy head on the
    CPU stream.
    """

    import mlx.nn as nn

    ids = sorted(int(i) for i in _ranked_ids(256, 1_024, 77))
    text = _TextModelStandIn()
    native = nn.QuantizedLinear(
        64, 1_024, bias=False, group_size=64, bits=8, mode="affine"
    )
    text._mtplx_native_mtp_draft_head = lambda: native
    # draft_lm_head._install_draft_lm_head stamps the configured draft head
    # here before it calls the FR-Spec install.
    text._mtplx_draft_lm_head = _configured_head()

    saved = frspec_draft.load_frspec_ids
    frspec_draft.load_frspec_ids = lambda: list(ids)
    try:
        report = frspec_draft.install_frspec_draft_head(text)
    finally:
        frspec_draft.load_frspec_ids = saved

    assert report["installed"] is True
    assert report["source"] == "native_mtp_head"
    assert report["output_mode"] == "full"
    assert report["legacy_swap"] is False

    head = text._mtplx_frspec_draft_head
    # The install leaves the unpruned head at the legacy stamp -- probing THAT
    # for liveness is the bug this test guards against.
    assert text._mtplx_draft_lm_head is not head
    assert isinstance(text._mtplx_draft_lm_head, nn.QuantizedLinear)
    assert prescatter._live_draft_route(text, head) == "native_mtp_head"
    # And the array the wrapper returns is the one the forward hands the
    # reader, so the stash identity check holds on this route.
    head.arm_prescatter_capture(True)
    dense = text._mtp_draft_head_logits(mx.zeros((1, 1, 64)))
    row = head.take_prescatter_row(dense)
    assert row is not None and int(row.reshape(-1).shape[0]) == 256
    head.arm_prescatter_capture(False)


def test_the_real_install_legacy_swap_is_also_recognised(monkeypatch):
    """``MTPLX_FRSPEC_LEGACY=1`` swaps the wrapper in globally instead."""

    import mlx.nn as nn

    ids = sorted(int(i) for i in _ranked_ids(256, 1_024, 78))
    text = SimpleNamespace()  # no bind hook: the generic mtp_patch surface
    text._mtplx_draft_lm_head = nn.QuantizedLinear(
        64, 1_024, bias=False, group_size=64, bits=8, mode="affine"
    )
    monkeypatch.setattr(frspec_draft, "load_frspec_ids", lambda: list(ids))
    monkeypatch.setattr(frspec_draft, "frspec_legacy_enabled", lambda: True)

    report = frspec_draft.install_frspec_draft_head(text)
    assert report["installed"] is True
    assert report["legacy_swap"] is True
    head = text._mtplx_frspec_draft_head
    assert prescatter._live_draft_route(text, head) == "legacy_swap"


@pytest.mark.parametrize(
    "override, match",
    [
        ({"draft_core": "device"}, "stock draft selector"),
        ({"relaxed_draft_ties": True}, "RELAXED_DRAFT_TIES"),
        ({"frspec_legacy_ids": np.arange(4)}, "FRSPEC_LEGACY"),
        ({"device_k20_route": object()}, "DEVICE_K20"),
        ({"pr391_route": object()}, "PR391"),
        ({"target_prefix_verify": True}, "target-prefix"),
        ({"a3b_target_prefix_route": object()}, "target-prefix"),
        ({"adaptive_width_policy": object()}, "adaptive-width"),
        ({"combine_greedy_draft_read": True}, "dense row"),
        ({"draft_confidence_needed": True}, "dense row"),
        ({"draft_margin_threshold": 0.5}, "dense row"),
        ({"wants_policy_metrics": True}, "dense row"),
        ({"correction_cache_enabled": True}, "correction cache"),
        ({"adapter_ensemble_q": True}, "adapter ensemble"),
        ({"mtp_topk_reranker": object()}, "reranker"),
        ({"penalties_active": True}, "real token id"),
        ({"steer_active": True}, "real token id"),
    ],
)
def test_claim_declines_every_unsupported_request_term(armed, override, match):
    """Request-shaped ineligibility stands aside; it does not raise.

    Every override here is a property of ONE REQUEST, and the stock draft
    reader serves all of them.  Raising made each one an HTTP 500 in serving
    -- the greedy case took down the composed-stack HumanEval gate on its very
    first request, 2026-09-02.
    """

    rt, head, _ = _runtime()
    contract.reset_for_test()
    kwargs = {**_ELIGIBLE, **override}
    receipt: dict[str, object] = {}
    plan = prescatter.claim_draft_route(
        rt, draft_sampler=_config(), receipt=receipt, **kwargs
    )
    assert plan is None
    assert receipt["installed"] is False
    assert re.search(match, str(receipt["declined_detail"]))
    assert contract.decline_counts(prescatter._ENV_VAR)[receipt["declined"]] == 1
    # The stash stays disarmed: the stock reader owns the dense row.
    assert head._prescatter_capture is False


@pytest.mark.parametrize("top_k", [0, 20])
def test_a_greedy_request_claims_the_greedy_route(armed, top_k):
    """The 2026-09-02 production failure, fixed: greedy WORKS here.

    HumanEval -- and every ``temperature: 0`` API call -- is greedy, and the
    greedy chain owns the draft read on those requests.  The lane serves it
    (``greedy_chain_step``) instead of standing aside, so the pre-scatter is
    exactly where the sentinel lanes are most wasteful.  ``top_k`` is not part
    of a greedy contract: the read is an argmax, so both spellings of a greedy
    sampler claim.
    """

    rt, head, _ = _runtime()
    contract.reset_for_test()
    receipt: dict[str, object] = {}
    plan = prescatter.claim_draft_route(
        rt,
        draft_sampler=_config(temperature=0.0, top_k=top_k),
        receipt=receipt,
        greedy_chain_enabled=True,
        **_ELIGIBLE,
    )
    assert plan is not None
    assert plan.greedy is True
    assert plan.to_dict()["read"] == "greedy_argmax"
    assert head._prescatter_capture is True
    assert receipt == {}
    assert contract.decline_counts(prescatter._ENV_VAR) == {}
    prescatter.release_draft_route(plan)


def test_a_sampled_request_still_claims_the_k20_route(armed):
    """The temperature-1 shape the ABBA windows measure is unchanged."""

    rt, head, _ = _runtime()
    contract.reset_for_test()
    plan = prescatter.claim_draft_route(
        rt,
        draft_sampler=_config(temperature=1.0, top_k=TOP_K),
        greedy_chain_enabled=False,
        **_ELIGIBLE,
    )
    assert plan is not None
    assert plan.greedy is False
    assert plan.top_k == TOP_K
    assert plan.to_dict()["read"] == "sampled_k20"
    assert head._prescatter_capture is True
    prescatter.release_draft_route(plan)


def test_a_greedy_request_claims_without_the_chain_too(armed):
    """A greedy cycle that falls out of the chain reads the compact row too.

    ``ccopy``, a mid-generation steering arm and a depth-0 cycle all drop the
    greedy chain for that cycle; those land in ``read_draft``'s own
    ``temperature <= 0`` branch, which is the same argmax on the host.
    """

    rt, _, _ = _runtime()
    plan = prescatter.claim_draft_route(
        rt,
        draft_sampler=_config(temperature=0.0, top_k=0),
        greedy_chain_enabled=False,
        **_ELIGIBLE,
    )
    assert plan is not None and plan.greedy is True
    prescatter.release_draft_route(plan)


def test_strict_claims_turns_a_decline_back_into_a_failure(armed, monkeypatch):
    """A measured arm still fails closed under MTPLX_STRICT_CLAIMS."""

    monkeypatch.setattr(contract, "_STRICT", True)
    rt, head, _ = _runtime()
    with pytest.raises(DraftK20PrescatterIneligible, match="reranker"):
        prescatter.claim_draft_route(
            rt,
            draft_sampler=_config(),
            **{**_ELIGIBLE, "mtp_topk_reranker": object()},
        )
    assert head._prescatter_capture is False


def test_claim_declines_draft_sampler_penalties(armed):
    rt, _, _ = _runtime()
    contract.reset_for_test()
    config = SamplerConfig(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        presence_penalty=0.2,
    )
    receipt: dict[str, object] = {}
    assert (
        prescatter.claim_draft_route(
            rt, draft_sampler=config, receipt=receipt, **_ELIGIBLE
        )
        is None
    )
    assert receipt["declined"] == "draft_sampler_penalties"


def test_claim_declines_without_top_k(armed):
    rt, _, _ = _runtime()
    contract.reset_for_test()
    receipt: dict[str, object] = {}
    assert (
        prescatter.claim_draft_route(
            rt, draft_sampler=_config(top_k=0), receipt=receipt, **_ELIGIBLE
        )
        is None
    )
    assert receipt["declined"] == "no_top_k"


# ---------------------------------------------------------------------------
# 6. Wiring: the gate is off by default and the receipt exists
# ---------------------------------------------------------------------------


def test_gate_is_off_by_default_in_generation(monkeypatch):
    from mtplx import generation
    from mtplx import qwen4_draft_k20_prescatter as prescatter

    # Read at USE (not frozen at import): unforced global + unset env => off.
    monkeypatch.setattr(prescatter, "_ENABLED", None)
    monkeypatch.delenv("MTPLX_QWEN4_DRAFT_K20_PRESCATTER", raising=False)
    assert generation._qwen4_draft_k20_prescatter_enabled() is False
    assert "draft_k20_prescatter" in generation.GenerationStats.__dataclass_fields__


def test_the_draft_loop_reads_the_plan_before_the_stock_reader():
    """The one hot-loop site is guarded by ``is not None``."""

    import inspect

    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    assert "elif _draft_k20_prescatter_plan is not None:" in source
    assert "_qwen4_draft_k20_prescatter_read(" in source
    assert "draft_k20_prescatter=_draft_k20_prescatter_receipt," in source


def test_the_call_site_declines_instead_of_raising_on_a_greedy_request():
    """The 2026-09-02 outage, pinned at the call site.

    The greedy-chain term is decided INSIDE the claim (so it declines like
    every other request term) and the receipt dict is handed in, so a decline
    is recorded rather than raised.  A `raise DraftK20PrescatterIneligible`
    in `generate_mtpk` would put the 500 back.
    """

    import inspect

    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    assert "greedy_chain_enabled=_greedy_chain_eligible," in source
    assert "receipt=_draft_k20_prescatter_receipt," in source
    assert "raise DraftK20PrescatterIneligible(" not in source
    # The device-K20 sibling's CLAIM-site raise went too; the only
    # DeviceK20Ineligible left in the loop is the mid-decode guard, which is
    # a different class of failure (see the module's report).
    assert "device K20 requires the stock draft route selector" not in source


def test_the_claim_reads_the_effective_relaxed_reader_condition():
    """Greedy requests under MTPLX_QWEN4_RELAXED_DRAFT_TIES=1 take the compact-row read.

    The relaxed-tie cycle reader is installed only for sampled drafts
    (temperature > 0 with a top-k); a greedy request runs the stock reader,
    so the claim must decline on the INSTALLED condition, not on the runtime
    flag alone (2026-09-03 W1: the flag alone declined every request).
    """

    import pathlib

    source = pathlib.Path("mtplx/generation.py").read_text()
    start = source.index("_draft_k20_prescatter_plan = _qwen4_draft_k20_prescatter_claim(")
    block = source[start : source.index("if _draft_k20_prescatter_plan is not None:", start)]
    assert 'getattr(rt, "qwen4_relaxed_draft_ties", False)' in block
    assert "and draft_sampler.temperature > 0" in block
    assert "and int(draft_sampler.top_k) > 0" in block
    reader = source[source.index("installed_cycle_draft_reader = (") :][:400]
    assert "_relaxed_tie_cycle_draft_reader" in reader
    assert "draft_sampler.temperature > 0" in reader
    assert "int(draft_sampler.top_k) > 0" in reader
