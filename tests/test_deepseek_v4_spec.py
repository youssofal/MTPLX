"""Speculative-decode gates for the DeepSeek-V4 MLX backend: cache rollback + spec==AR.

Speculation only ever costs quality in one place: the cache.  Draft and verify are
both just forwards, and the accept/reject rule is exact by construction; what can
silently corrupt a run is a *rejected* tail that does not fully un-decode, because
every later token then conditions on state the committed prefix never produced.  So
the gates here are two, in order of strength:

  1. **Rollback exactness** (unit, no engine).  Decode ``k`` extra tokens, ``trim(k)``,
     and require the cache to be the one the shorter context would have held, on
     every lane the backend has: the sliding-window KV, the attention compressor's
     frontier and its emitted rows, and the indexer's own second compressor lane.
     The claim is *bit* equality, not tolerance, and the headline form of it is that
     the next tokens' logits are bit-identical to the never-decoded path.

     One field is deliberately excluded from the "every field" claim and asserted
     more weakly: the *retained* depth of the window buffer and the frontier
     journals.  Those are bounded ring-style buffers, so overshooting by ``k`` and
     rewinding evicts up to ``k`` rows off their old end that a shorter run would
     still hold.  Those rows are unattendable and unemittable by construction --
     they are rollback headroom, not model state -- so they are gated as
     newest-suffix equality plus a coverage floor.  Both regimes are covered: cases
     where the buffers have not reached their cap (then EVERY field, meta_state
     included, is exactly equal) and cases where they have.

  2. **spec == AR** (through the real engine).  ``generate_mtpk`` at depth 1/2/3 must
     emit the identical greedy token sequence as ``generate_ar``.  This is the shop's
     standard speculative gate and it is run here against the actual
     ``mtplx.generation`` machine -- prefill, draft chain, batched verify, accept,
     reject, rollback, repair -- not a hand-rolled loop, so it also gates the
     registry/runtime wiring, not just the model.  The prompts and lengths are chosen
     so verify batches straddle a ratio-4 emission boundary, the ratio-128 boundary,
     the ``window_size`` eviction edge and the indexer's dense->sparse threshold.

  3. **Mutation gate** on (1): four plausible rollback bugs -- a stale compressor
     frontier, an off-by-one on the emitted-row drop, an indexer lane left un-rewound,
     and a window rewind past what retention can serve -- must each be caught.

Self-contained: shrunk seeded config, no downloads, no checkpoint, no torch.  CPU
device so MLX fp32 is bit-exact (its GPU fast path carries ~7.5e-4 relative) -- the
same convention as the parity, decode and indexer gates.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

@pytest.fixture(autouse=True)
def _cpu_default_device():
    # CPU-pinned by design, but the pin must stay test-scoped: a module-level
    # set_default_device leaks into every later-collected module (pytest
    # imports all test modules before running any) and flips the engine's
    # Metal bit-exactness suites onto CPU fallbacks process-wide.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "..", "mtplx", "models", "deepseek_v4.py")
_spec = importlib.util.spec_from_file_location("dsv4_spec_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_spec_undertest"] = D
_spec.loader.exec_module(D)

# Same layer menu as the decode/indexer gates -- ratio-0 sliding window (also the
# hash-routed layer), ratio-4 overlap compressor + indexer, ratio-128 non-overlap
# compressor -- with index_topk shrunk so the sparse regime is reachable in a unit
# test (a ratio-4 layer emits one row per 4 tokens, so n_comp > 6 from token 28).
DIM = 32
N_HEADS = 4
HEAD_DIM = 16
INDEX_HEAD_DIM = 16       # power of two: the indexer Hadamard-rotates it
ROPE_DIM = 8
N_EXPERTS = 8
RATIOS = [0, 4, 128, 4]
WINDOW = 16
INDEX_TOPK = 6
SPARSE_FROM = (INDEX_TOPK + 1) * 4    # 28: first position with n_comp > index_topk


def _args(vocab: int = 64, **over):
    kwargs = dict(
        vocab_size=vocab,
        hidden_size=DIM,
        num_hidden_layers=len(RATIOS),
        num_hash_layers=1,
        num_attention_heads=N_HEADS,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=16,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2,
        index_n_heads=N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        compress_ratios=list(RATIOS),
        compress_rope_theta=160000.0,
        sliding_window=WINDOW,
        rope_scaling={
            "original_max_position_embeddings": 65536,
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "type": "yarn",
        },
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=0.0,
        num_nextn_predict_layers=1,
    )
    kwargs.update(over)
    return D.ModelArgs(**kwargs)


def _seeded_model(seed=0, vocab=64, **over):
    """Model with every parameter filled from the module tree's own shapes."""
    mx.random.seed(seed)
    args = _args(vocab=vocab, **over)
    model = D.Model(args)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if leaf == "tid2eid":
            new = mx.random.randint(0, args.n_routed_experts, value.shape).astype(
                mx.int32
            )
        elif value.ndim == 1:
            noise = mx.random.normal(value.shape) * 0.1
            centre = 1.0 if leaf == "scale" or name.endswith("norm.weight") else 0.0
            new = noise + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return args, model


def _tokens(seq_len, vocab=64, batch=1, seed=1234):
    mx.random.seed(seed)
    return mx.random.randint(0, vocab, (batch, seq_len))


# --------------------------------------------------------------------------- #
# 1. rollback exactness
# --------------------------------------------------------------------------- #
# Which entries of DeepseekV4Cache.state carry model state (must be bit-equal after
# a rollback) and which are bounded rollback buffers (newest-suffix equality); see
# the module docstring for why the split exists.
_SEMANTIC_FIELDS = {
    1: "compressed",
    2: "comp.cur_kv",
    3: "comp.cur_score",
    4: "comp.prev_kv",
    5: "comp.prev_score",
    8: "index_compressed",
    9: "index_comp.cur_kv",
    10: "index_comp.cur_score",
    11: "index_comp.prev_kv",
    12: "index_comp.prev_score",
}
_RETAINED_FIELDS = {
    0: "window",
    6: "comp.tail_kv",
    7: "comp.tail_score",
    13: "index_comp.tail_kv",
    14: "index_comp.tail_score",
}


def _field(value):
    return None if value is None else np.array(value)


def _cache_fields(cache):
    return [[_field(v) for v in c.state] for c in cache]


def _exactly_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and np.array_equal(a, b)


def _suffix_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    n = min(a.shape[1], b.shape[1])
    return np.array_equal(a[:, -n:], b[:, -n:])


def _assert_rolled_back_exactly(ref_cache, got_cache, label):
    """Every semantic field bit-equal; buffers equal on their newest rows."""
    assert len(ref_cache) == len(got_cache)
    ref, got = _cache_fields(ref_cache), _cache_fields(got_cache)
    fully_exact = True
    for layer, (rc, gc, rf, gf) in enumerate(
        zip(ref_cache, got_cache, ref, got)
    ):
        assert rc.offset == gc.offset, f"{label}: layer {layer} offset"
        assert rc.comp.n_emitted == gc.comp.n_emitted, f"{label}: layer {layer} n_emitted"
        assert rc.index_comp.n_emitted == gc.index_comp.n_emitted, (
            f"{label}: layer {layer} index n_emitted"
        )
        for idx, name in _SEMANTIC_FIELDS.items():
            assert _exactly_equal(rf[idx], gf[idx]), (
                f"{label}: layer {layer} field {name} is not bit-equal after rollback"
            )
        for idx, name in _RETAINED_FIELDS.items():
            assert _suffix_equal(rf[idx], gf[idx]), (
                f"{label}: layer {layer} buffer {name} diverges on its newest rows"
            )
            if not _exactly_equal(rf[idx], gf[idx]):
                fully_exact = False
        held = 0 if gc.window is None else int(gc.window.shape[1])
        assert held >= min(int(gc.offset), gc.window_size), (
            f"{label}: layer {layer} window kept {held} rows, too few to attend"
        )
    return fully_exact


def _decode_to(model, cache, ids, start, end, step=1):
    for t in range(start, end, step):
        model(ids[:, t: min(t + step, end)], cache=cache)


def _rollback_case(prompt, decoded, k, *, batch=1, seed=0, tail=6, vocab=64):
    """Two arms to the same offset; the second overshoots by ``k`` and trims it back.

    Returns ``(fully_exact, ref_logits, got_logits)``.
    """
    args, model = _seeded_model(seed=seed, vocab=vocab)
    total = prompt + decoded
    ids = _tokens(total + k + tail, vocab=vocab, batch=batch)

    def primed():
        cache = model.make_cache()
        model(ids[:, :prompt], cache=cache)
        _decode_to(model, cache, ids, prompt, total)
        return cache

    ref_cache = primed()
    got_cache = primed()
    # the verify shape: K+1 tokens in one forward, then the rejected tail comes off
    model(ids[:, total: total + k], cache=got_cache)
    for c in got_cache:
        assert c.trim(k) == k, "trim must report the number of positions it removed"
    fully_exact = _assert_rolled_back_exactly(
        ref_cache, got_cache, f"P={prompt} D={decoded} k={k} b={batch}"
    )

    ref_logits = [
        np.array(model(ids[:, t: t + 1], cache=ref_cache))
        for t in range(total, total + tail)
    ]
    got_logits = [
        np.array(model(ids[:, t: t + 1], cache=got_cache))
        for t in range(total, total + tail)
    ]
    return fully_exact, ref_logits, got_logits


def _assert_logits_bit_equal(ref_logits, got_logits, label):
    assert len(set(int(x.argmax()) for x in ref_logits)) > 1, (
        f"{label}: oracle logits are degenerate (constant argmax)"
    )
    for step, (a, b) in enumerate(zip(ref_logits, got_logits)):
        assert np.array_equal(a, b), (
            f"{label}: step {step} logits differ after rollback "
            f"(max_abs={float(np.max(np.abs(a - b))):.3e})"
        )


def test_trim_advertises_the_engine_cache_contract():
    """``mtplx.cache_state`` decides rollback strategy off these three answers."""
    _, model = _seeded_model()
    cache = model.make_cache()
    assert all(c.is_trimmable() for c in cache), (
        "the engine routes non-trimmable caches to snapshot/restore instead"
    )
    model(_tokens(30), cache=cache)
    for c in cache:
        assert c.max_rollback == min(c.rollback_capacity, 30)
        assert c.trim(0) == 0 and c.offset == 30
        assert c.trim(3) == 3
        assert c.offset == 27 and c.size() == 27


def test_rollback_across_a_ratio4_emission_boundary():
    """The rejected tail completes a ratio-4 window, so the rewind has to drop an
    emitted compressed row *and* rebuild a frontier the emit had already reset."""
    prompt, decoded, k = 13, 10, 4      # offset 23 -> 27: window 5 (20..23) completes
    assert (prompt + decoded) % 4 != 0
    assert (prompt + decoded + k) // 4 > (prompt + decoded) // 4
    exact, ref, got = _rollback_case(prompt, decoded, k)
    _assert_logits_bit_equal(ref, got, "ratio-4 boundary")
    assert exact, "at this length no buffer has reached its cap; expect full equality"


def test_rollback_across_a_ratio128_emission_boundary():
    """The ratio-128 lane emits its first row at position 127; a verify batch that
    straddles it must un-emit that row and restore a 127-row frontier."""
    prompt, decoded, k = 100, 26, 5     # offset 126 -> 131 crosses 127
    total = prompt + decoded
    assert total < 128 <= total + k
    exact, ref, got = _rollback_case(prompt, decoded, k)
    _assert_logits_bit_equal(ref, got, "ratio-128 boundary")
    assert not exact, (
        "premise: past window_size + rollback_capacity the retained buffers have "
        "reached their cap, which is the regime the suffix rule exists for"
    )


def test_rollback_across_the_indexer_dense_to_sparse_threshold():
    """Crossing n_comp > index_topk switches the ratio-4 layers onto the scoring
    path, so the indexer's own compressor lane becomes load-bearing exactly here."""
    prompt, decoded, k = 13, 13, 3      # offset 26 -> 29, and SPARSE_FROM == 28
    total = prompt + decoded
    assert total < SPARSE_FROM <= total + k
    exact, ref, got = _rollback_case(prompt, decoded, k)
    _assert_logits_bit_equal(ref, got, "sparse threshold")
    assert exact


def test_rollback_deep_in_the_sparse_regime():
    """Well past the threshold, where the top-k filter is selecting every step and a
    stale indexer row would change which compressed rows attention can see."""
    exact, ref, got = _rollback_case(13, 40, 3)
    _assert_logits_bit_equal(ref, got, "sparse regime")
    # offset 53 is still under window_size + rollback_capacity, so nothing has been
    # pruned yet and the stricter claim holds here too.
    assert exact


def test_rollback_from_a_partial_window_and_before_any_emission():
    """Both frontier edges: a prompt that ends mid-window, and a rewind that lands
    before the lane has emitted anything at all (n_emitted back to 0)."""
    exact, ref, got = _rollback_case(3, 0, 2)     # offset 3 -> 5 -> 3, no ratio-4 row
    _assert_logits_bit_equal(ref, got, "pre-emission")
    assert exact
    exact, ref, got = _rollback_case(6, 0, 3)     # 6 -> 9 -> 6, crosses the row at 8
    _assert_logits_bit_equal(ref, got, "partial window")
    assert exact


def test_rollback_is_exact_with_batch_gt_1():
    """The cache carries a batch axis on every lane; a rewind must not smear rows
    between rows of the batch."""
    exact, ref, got = _rollback_case(13, 25, 4, batch=3)
    for step, (a, b) in enumerate(zip(ref, got)):
        assert np.array_equal(a, b), f"batched: step {step} logits differ"
    assert ref[0].shape[0] == 3


def test_rollback_survives_repeated_verify_reject_cycles():
    """One rollback being exact is not enough: the speculative loop rewinds on every
    rejection, so excursions must leave no residue over many cycles.

    Both arms commit through the same one-token forwards; the speculative arm takes
    an extra 3-token excursion each cycle and rolls it back.  (Committing through a
    *wide* forward instead would compare a different computation -- a row projected
    in a 4-row batch is not bit-identical to the same row projected alone -- which is
    a batching question, not a rollback one.  Speculative decode is committed-
    sequence exact, which is what the spec==AR gates below assert.)
    """
    args, model = _seeded_model()
    ids = _tokens(120)
    clean, dirty = model.make_cache(), model.make_cache()
    model(ids[:, :13], cache=clean)
    model(ids[:, :13], cache=dirty)
    pos = 13
    for _cycle in range(10):
        model(ids[:, pos: pos + 1], cache=clean)
        model(ids[:, pos: pos + 1], cache=dirty)
        # draft 3 more, reject all 3
        model(ids[:, pos + 1: pos + 4], cache=dirty)
        for c in dirty:
            assert c.trim(3) == 3
        pos += 1
    _assert_rolled_back_exactly(clean, dirty, "repeated cycles")
    ref = [np.array(model(ids[:, t: t + 1], cache=clean)) for t in range(pos, pos + 6)]
    got = [np.array(model(ids[:, t: t + 1], cache=dirty)) for t in range(pos, pos + 6)]
    _assert_logits_bit_equal(ref, got, "repeated cycles")


def test_rollback_depth_is_bounded_and_refuses_rather_than_half_rewinding():
    """Eviction is irreversible, so the window can only serve a bounded rewind.  Past
    it the cache must raise: ``rollback_after_verify`` ignores trim's return value, so
    a clamped rewind would leave a silently desynced cache decoding on."""
    _, model = _seeded_model()
    cache = model.make_cache()
    model(_tokens(100), cache=cache)
    entry = cache[0]
    assert entry.rollback_capacity == D._DEFAULT_ROLLBACK_CAPACITY
    assert entry.offset > entry.rollback_capacity, (
        "premise: the refusal must be the capacity bound, not 'deeper than context'"
    )
    with pytest.raises(ValueError, match="rollback_capacity"):
        entry.trim(entry.rollback_capacity + 1)
    assert entry.offset == 100, "a refused trim must not mutate the cache"
    with pytest.raises(ValueError, match="cannot trim"):
        entry.trim(1000)

    tight = D.DeepseekV4Cache(WINDOW, 4, HEAD_DIM, rollback_capacity=2)
    assert tight.rollback_capacity == 2
    tight.offset = 10
    with pytest.raises(ValueError, match="rollback_capacity"):
        tight.trim(3)


def test_engine_snapshot_free_repair_accepts_this_cache():
    """The smallest faithful integration: because ``trim`` is exact, the engine's
    generic all-trimmable repair serves this backend and no bespoke restore path is
    needed.  This is the very helper ``generate_mtpk`` calls when the verify snapshot
    is skipped (MTPLX_SKIP_VERIFY_SNAPSHOT=1, the product-profile default)."""
    from mtplx.cache_state import (
        trim_verified_window_without_snapshot,
        snapshot_untrimmable_cache,
    )

    _, model = _seeded_model()
    cache = model.make_cache()
    model(_tokens(40), cache=cache)
    # verified a 4-wide window, committing 1 token of it
    assert trim_verified_window_without_snapshot(
        cache, verified_tokens=4, keep_tokens=1
    )
    assert all(c.offset == 37 for c in cache)
    # and the snapshot lane agrees there is no recurrent state to restore
    snap = snapshot_untrimmable_cache(cache)
    assert all(state is None for state in snap.states)


# --------------------------------------------------------------------------- #
# 2. spec == AR through the real engine
# --------------------------------------------------------------------------- #
class _FixedTokenizer:
    eos_token_id = None
    eos_token_ids: set = set()

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


def _runtime(seed=0, vocab=64):
    """A real MTPLXRuntime over the shrunk model, wired the way ``mtplx.runtime``
    wires it: the config declares the draft head and the injection publishes it."""
    from mtplx.models.deepseek_v4 import (
        inject_deepseek_v4_mtp_support,
        is_deepseek_v4_mtp_config,
    )
    from mtplx.mtp_patch import MTPContract, validate_mtp_support
    from mtplx.runtime import MTPLXRuntime

    config = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_nextn_predict_layers": 1,
    }
    assert is_deepseek_v4_mtp_config(config)
    _args_, model = _seeded_model(seed=seed, vocab=vocab)
    assert inject_deepseek_v4_mtp_support(model, Path("."), config, MTPContract())
    assert validate_mtp_support(model), (
        "runtime.load raises 'MTP injection failed' unless this passes"
    )
    return MTPLXRuntime(
        model=model,
        tokenizer=_FixedTokenizer(),
        model_path=Path("."),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _prompt(n, vocab=64, seed=7):
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, vocab, size=n)]


def _ar(rt, prompt, max_tokens):
    from mtplx.generation import generate_ar
    from mtplx.sampling import SamplerConfig

    return generate_ar(
        rt,
        prompt,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0),
        stop_token_ids=set(),
    )


def _spec(rt, prompt, max_tokens, depth, verify_strategy="batched"):
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    return generate_mtpk(
        rt,
        prompt,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=depth,
        mtp_history_policy="committed",
        stop_token_ids=set(),
        verify_strategy=verify_strategy,
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_spec_decode_reproduces_ar_token_for_token(depth):
    """The standard gate.  Greedy speculative decode is a pure latency optimisation:
    if the committed sequence ever differs from AR, the rollback is lossy.

    Lengths are picked so the verify batches cross everything that can break a
    rewind: window_size eviction (16), ratio-4 emissions (every 4 tokens) and the
    indexer's dense->sparse threshold (28).
    """
    prompt = _prompt(17)
    baseline = _ar(_runtime(), prompt, 40)
    assert len(baseline.tokens) == 40
    assert len(set(baseline.tokens)) > 1, "premise: AR output must not be degenerate"
    assert 17 + 40 > SPARSE_FROM > WINDOW

    out = _spec(_runtime(), prompt, 40, depth)
    assert out.tokens == baseline.tokens, (
        f"depth {depth}: speculative decode diverged from AR\n"
        f"  AR  : {baseline.tokens}\n"
        f"  spec: {out.tokens}"
    )


def test_spec_decode_reproduces_ar_across_the_ratio128_boundary():
    """The ratio-128 lane emits its first compressed row at position 127, i.e. inside
    a verify batch here rather than inside a prefill."""
    prompt = _prompt(100)
    baseline = _ar(_runtime(), prompt, 40)
    assert 100 < 127 < 140
    out = _spec(_runtime(), prompt, 40, 3)
    assert out.tokens == baseline.tokens


def test_spec_decode_exercises_both_accept_and_reject():
    """A gate that only ever rejects never tests the accept path, and vice versa.
    A small vocabulary makes an untrained draft head agree often enough for both to
    happen in one short run."""
    prompt = _prompt(17, vocab=8)
    baseline = _ar(_runtime(vocab=8), prompt, 32)
    out = _spec(_runtime(vocab=8), prompt, 32, 2)
    stats = out.stats.to_dict()
    assert out.tokens == baseline.tokens
    assert stats["accepted_drafts"] > 0, "premise: no draft was ever accepted"
    assert stats["rejected_drafts"] > 0, "premise: no draft was ever rejected"


@pytest.mark.parametrize("skip_snapshot", [False, True])
@pytest.mark.parametrize(
    "verify_strategy", ["batched", "trim_commit", "target_prefix", "capture_commit"]
)
def test_spec_matches_ar_on_every_commit_lane(verify_strategy, skip_snapshot,
                                              monkeypatch):
    """Each verify strategy repairs a rejection through a different path, and the
    exactness of ``trim`` is what all of them lean on here.

    ``trim_commit``/``target_prefix`` commit the accepted prefix by trimming the
    verify tail; ``capture_commit`` falls through to the same trim because this is a
    pure-attention model with nothing recurrent to capture; ``batched`` rolls the
    whole verify back and re-forwards.  With MTPLX_SKIP_VERIFY_SNAPSHOT=1 -- the
    product-profile default -- there is no snapshot to restore from either, so the
    only repair left is the engine's snapshot-free all-trimmable path.  All eight
    combinations must land on the AR sequence.
    """
    if skip_snapshot:
        monkeypatch.setenv("MTPLX_SKIP_VERIFY_SNAPSHOT", "1")
    else:
        monkeypatch.delenv("MTPLX_SKIP_VERIFY_SNAPSHOT", raising=False)
    prompt = _prompt(17, vocab=8)
    baseline = _ar(_runtime(vocab=8), prompt, 32)
    out = _spec(_runtime(vocab=8), prompt, 32, 2, verify_strategy=verify_strategy)
    stats = out.stats.to_dict()
    assert stats["rejected_drafts"] > 0, "premise: the repair path must be exercised"
    assert out.tokens == baseline.tokens, (
        f"{verify_strategy} (skip_snapshot={skip_snapshot}) diverged from AR"
    )


def test_acceptance_counters_are_populated_per_depth():
    """What the bench window reads.  The counters are the engine's, shared with every
    other MTP backend; this gates that driving V4 through it actually fills them --
    a backend whose draft never runs would report zeros and look like 100% rejection.
    """
    depth = 3
    out = _spec(_runtime(vocab=8), _prompt(17, vocab=8), 32, depth)
    stats = out.stats.to_dict()
    assert stats["runtime_mtp_enabled"] is True
    assert stats["mode"] in {"mtpk", f"mtp{depth}", "mtp"} or stats["mode"]
    assert len(stats["drafted_by_depth"]) == depth
    assert len(stats["accepted_by_depth"]) == depth
    assert sum(stats["drafted_by_depth"]) == stats["drafted_tokens"] > 0
    assert sum(stats["accepted_by_depth"]) > 0
    assert stats["drafted_by_depth"][0] >= stats["drafted_by_depth"][depth - 1] > 0, (
        "a depth-i draft only runs when depth i-1 was proposed, so the histogram "
        "must be non-increasing"
    )
    assert stats["mtp_forward_calls"] > 0 and stats["make_mtp_cache_calls"] > 0


def test_runtime_reports_no_mtp_when_the_checkpoint_dropped_the_draft_head():
    """The published mlx-community conversions declare num_nextn_predict_layers and
    ship no mtp.* tensor.  Injection must report False there so ``runtime.load``
    takes its degrade-to-autoregressive branch instead of raising."""
    from mtplx.models.deepseek_v4 import inject_deepseek_v4_mtp_support
    from mtplx.mtp_patch import MTPContract

    config = {"model_type": "deepseek_v4", "num_nextn_predict_layers": 1}
    _, model = _seeded_model()
    weights = {k: mx.zeros(v.shape, v.dtype)
               for k, v in tree_flatten(D.Model(_args(num_nextn_predict_layers=0))
                                        .parameters())}
    model.sanitize(weights)
    assert not model.has_mtp
    assert inject_deepseek_v4_mtp_support(
        model, Path("."), config, MTPContract()
    ) is False
    # and a model that is not V4 at all is never captured by this injection
    assert inject_deepseek_v4_mtp_support(
        model, Path("."), {"model_type": "deepseek_v3", "num_nextn_predict_layers": 1},
        MTPContract(),
    ) is False


def test_draft_surface_rejects_the_contract_violations_it_cannot_honour():
    """Two knobs the uniform runtime signature carries that this architecture has no
    faithful answer for.  Silently ignoring either would corrupt drafting rather than
    fail, so both raise."""
    _, model = _seeded_model()
    h = model.hc_hidden(mx.array([[1, 2, 3]]))
    ids = mx.array([[4, 5, 6]])
    with pytest.raises(ValueError, match="position_offset"):
        model.mtp_forward(h, ids, position_offset=7)
    with pytest.raises(ValueError, match="input_embeddings"):
        model(mx.array([[1, 2, 3]]), input_embeddings=h)
    with pytest.raises(TypeError, match="not the list"):
        model.mtp_forward(h, ids, cache=model.make_mtp_cache())
    with pytest.raises(TypeError, match="not both"):
        model.mtp_forward(h, ids, cache=model.make_mtp_cache()[0],
                          mtp_cache=model.make_mtp_cache())


def test_forward_surface_matches_the_runtime_contract():
    """``MTPLXRuntime.forward_ar`` probes the signature for emit_logits/logits_keep
    and calls with return_hidden; the hidden it gets back must be the hc-form state
    ``mtp_forward`` consumes, and the logits must not change because of any of it."""
    args, model = _seeded_model()
    ids = _tokens(9)
    plain = np.array(model(ids))
    logits, hidden = model(ids, return_hidden=True)
    assert hidden.shape == (1, 9, args.hc_mult, args.hidden_size)
    assert np.array_equal(np.array(logits), plain)
    kept = model(ids, logits_keep=1)
    assert kept.shape == (1, 1, args.vocab_size)
    # logits_keep changes the SHAPE of the lm_head matmul (one row instead of nine),
    # so it is argmax-exact rather than bit-exact -- a one-row GEMM does not
    # accumulate identically to the same row inside a nine-row one.  Everything
    # upstream of the head is untouched, which is the part the draft consumes.
    one_row = np.array(kept)[:, 0]
    assert int(one_row.argmax()) == int(plain[:, -1].argmax())
    scale = float(np.max(np.abs(plain[:, -1]))) + 1e-12
    assert float(np.max(np.abs(one_row - plain[:, -1]))) / scale < 1e-6
    none_logits, hidden2 = model(ids, return_hidden=True, emit_logits=False)
    assert none_logits is None and np.array_equal(np.array(hidden2), np.array(hidden))
    # the draft's own hc-form output is what a depth>1 chain feeds back in
    draft_logits, draft_hidden = model.mtp_forward(hidden, ids, return_hidden=True)
    assert draft_hidden.shape == hidden.shape
    assert draft_logits.shape == (1, 9, args.vocab_size)


# --------------------------------------------------------------------------- #
# 3. mutation gate on the rollback
# --------------------------------------------------------------------------- #
def _defective_trim(cache_entry, n, defect):
    """Re-implementation of ``DeepseekV4Cache.trim`` carrying one named bug.

    Mirrors the real method step for step so the mutation is the *only* difference;
    each of these is a rollback error that would leave the model conditioning on
    state the committed prefix never produced.
    """
    entry = cache_entry
    new_offset = int(entry.offset) - int(n)
    if entry.window is not None:
        kept = int(entry.window.shape[1]) - int(n)
        entry.window = None if kept <= 0 else entry.window[:, :kept]
    if entry.compress_ratio:
        n_rows = new_offset // entry.compress_ratio
        if defect == "off_by_one_rows":
            n_rows += 1                                  # keep one un-emitted row
        if entry.compressed is not None:
            entry.compressed = (
                None if n_rows == 0 else entry.compressed[:, :n_rows]
            )
        if defect != "stale_frontier":
            entry.comp.rollback(n, new_offset)
        if entry.compress_ratio == 4 and defect != "unrewound_indexer":
            if entry.index_compressed is not None:
                entry.index_compressed = (
                    None if n_rows == 0 else entry.index_compressed[:, :n_rows]
                )
            entry.index_comp.rollback(n, new_offset)
    entry.offset = new_offset
    return n


@pytest.mark.parametrize(
    "defect",
    ["stale_frontier", "off_by_one_rows", "unrewound_indexer"],
)
def test_rollback_mutations_are_detected(defect):
    """Sensitivity check on the exactness gate itself.

    Each defect is run through the *whole* gate the real cases use -- the cache-state
    comparison and then 24 further decode steps -- and must be caught.  Both halves
    are reported, because they are different kinds of evidence: the state comparison
    proves the rewind is wrong, the logits prove the wrongness actually reaches the
    model.  ``stale_frontier`` and ``unrewound_indexer`` additionally trip the
    backend's own lane-desync assertion on a later forward, which is a third,
    earlier detector.
    """
    prompt, decoded, k = 13, 40, 3       # deep in the sparse regime, past eviction
    horizon = 24
    args, model = _seeded_model()
    total = prompt + decoded
    ids = _tokens(total + k + horizon)

    def primed():
        cache = model.make_cache()
        model(ids[:, :prompt], cache=cache)
        _decode_to(model, cache, ids, prompt, total)
        return cache

    ref_cache = primed()
    bad_cache = primed()
    model(ids[:, total: total + k], cache=bad_cache)
    for c in bad_cache:
        _defective_trim(c, k, defect)

    state_caught = False
    try:
        _assert_rolled_back_exactly(ref_cache, bad_cache, defect)
    except AssertionError:
        state_caught = True

    forward_caught = False
    try:
        for t in range(total, total + horizon):
            a = np.array(model(ids[:, t: t + 1], cache=ref_cache))
            b = np.array(model(ids[:, t: t + 1], cache=bad_cache))
            if not np.array_equal(a, b):
                forward_caught = True
                break
    except AssertionError:
        # the backend's own "indexer compressor lane desynced" guard
        forward_caught = True

    assert state_caught, f"{defect!r}: the cache-state comparison missed it"
    assert forward_caught, f"{defect!r}: the defect never reached the model output"


def test_window_rewind_past_retention_is_detected():
    """The fourth mutation: pretend the window kept no rollback margin at all.  The
    rows a deeper rewind needs are physically gone, so the only correct behaviours
    are 'refuse' or 'wrong'; silently succeeding is the bug this guards."""
    _, model = _seeded_model()
    cache = [
        D.DeepseekV4Cache(
            window_size=layer.attn.window_size,
            compress_ratio=layer.attn.compress_ratio,
            head_dim=layer.attn.head_dim,
            rollback_capacity=0,
        )
        for layer in model.layers
    ]
    model(_tokens(40), cache=cache)
    for c in cache:
        assert c.max_rollback == 0
        with pytest.raises(ValueError, match="rollback_capacity"):
            c.trim(1)
        assert c.offset == 40
    # and the engine's helper reports the refusal rather than half-trimming
    from mtplx.cache_state import trim_verified_window_without_snapshot

    with pytest.raises(ValueError):
        trim_verified_window_without_snapshot(cache, verified_tokens=4, keep_tokens=1)
    assert all(c.offset == 40 for c in cache)


# ---------------------------------------------------------------------------
# 4. the bench harness's spec-vs-AR policy
# ---------------------------------------------------------------------------
# The gates above run an all-fp32 shrunk model -- ``_seeded_model`` never calls
# ``set_dtype``, so every cast the activation-storage fix introduces is a no-op
# here and byte identity is the right bar.  On the real checkpoint at bf16 storage
# it is not: draft and verify are batch-shaped forwards, so the committed row's KV
# is projected inside a K+1-wide GEMM rather than alone, and at bf16 that reaches
# the argmax on near-tied tokens.  scripts/deepseek_v4_mtpk_bench.py therefore
# gates byte identity on the fp32 lane and reports it as data on the bf16 one.
#
# That decision is one boolean and one comparison, and it decides whether a GPU
# window's exit status means anything -- so it is gated here rather than left to
# the script.
def _bench_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "deepseek_v4_mtpk_bench.py"
    spec = importlib.util.spec_from_file_location("_dsv4_mtpk_bench_undertest", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dsv4_mtpk_bench_undertest"] = module
    spec.loader.exec_module(module)
    return module


def test_spec_gates_are_captured_at_fp32_so_the_fix_is_a_no_op_here():
    """The premise the whole section rests on, asserted rather than assumed."""
    _, model = _seeded_model()
    floats = [(k, v) for k, v in tree_flatten(model.parameters())
              if v.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64)]
    assert floats, "no float parameters found — the check below would be vacuous"
    assert all(v.dtype == mx.float32 for _, v in floats), (
        "a spec-gate parameter is not fp32: " + ", ".join(
            f"{k}={v.dtype}" for k, v in floats if v.dtype != mx.float32))
    from mtplx.models import deepseek_v4 as backend

    assert backend._store_dtype(mx.float32) == mx.float32
    assert not backend._FP32_ACTIVATIONS, "the escape hatch leaked into the gates"


def test_bench_env_flag_parsing_matches_the_backends(monkeypatch):
    """The harness re-derives the flag so it can decide before a 90 GiB load; if the
    two parsers drift, a window silently gates the wrong lane."""
    bench = _bench_module()
    from mtplx.models import deepseek_v4 as backend

    for raw in ("1", "true", "TRUE", "yes", "on", "0", "false", "no", "off", "", "  ", "maybe"):
        monkeypatch.setenv("MTPLX_DSV4_FP32_ACTIVATIONS", raw)
        assert bench._fp32_activations_env() == backend._env_flag(
            "MTPLX_DSV4_FP32_ACTIVATIONS", False
        ), f"parsers disagree on {raw!r}"
    monkeypatch.delenv("MTPLX_DSV4_FP32_ACTIVATIONS", raising=False)
    assert bench._fp32_activations_env() is False


def test_byte_identity_is_gated_on_the_fp32_lane_and_on_demand(monkeypatch):
    bench = _bench_module()
    monkeypatch.setenv("MTPLX_DSV4_FP32_ACTIVATIONS", "1")
    assert bench._exactness_is_enforced(False) is True, "fp32 lane must stay a hard gate"
    assert bench._exactness_is_enforced(True) is True

    monkeypatch.setenv("MTPLX_DSV4_FP32_ACTIVATIONS", "0")
    assert bench._exactness_is_enforced(False) is False, (
        "the bf16 default must report divergence rather than fail on it")
    assert bench._exactness_is_enforced(True) is True, "--require-exact must restore it"


def test_divergence_reports_the_whole_shape_not_just_the_first_index():
    bench = _bench_module()

    same = bench._divergence([1, 2, 3], [1, 2, 3])
    assert same["pass"] and same["divergent_tokens"] == 0
    assert same["first_divergence_index"] is None

    one = bench._divergence([1, 9, 3], [1, 2, 3])
    assert not one["pass"]
    assert one["divergent_tokens"] == 1 and one["first_divergence_index"] == 1
    assert one["baseline_at_divergence"] == 2 and one["arm_at_divergence"] == 9

    # a near-tie both arms recover from vs a rollback that desyncs: same first
    # index, and only the count tells them apart.
    recovered = bench._divergence([1, 9, 3, 4, 5], [1, 2, 3, 4, 5])
    desynced = bench._divergence([1, 9, 8, 7, 6], [1, 2, 3, 4, 5])
    assert recovered["first_divergence_index"] == desynced["first_divergence_index"] == 1
    assert recovered["divergent_tokens"] == 1 and desynced["divergent_tokens"] == 4


def test_a_truncated_arm_cannot_look_identical_by_ending_early():
    bench = _bench_module()
    short = bench._divergence([1, 2], [1, 2, 3, 4])
    assert not short["pass"]
    assert short["compared_tokens"] == 2 and short["divergent_tokens"] == 2
    assert short["first_divergence_index"] == 2
    assert short["baseline_at_divergence"] == 3 and short["arm_at_divergence"] is None


def test_the_summary_cell_says_which_it_is():
    """A receipt read months later must not mistake an ungated run for a passing one."""
    bench = _bench_module()
    assert bench._summary_cell(None) == "-"
    assert bench._summary_cell({"pass": True, "enforced": False}) == "PASS"
    gated = {"pass": False, "enforced": True, "divergent_tokens": 3}
    assert bench._summary_cell(gated) == "FAIL"
    reported = {"pass": False, "enforced": False, "divergent_tokens": 3}
    assert bench._summary_cell(reported) == "3 div"
    line = bench._divergence_line(
        dict(reported, compared_tokens=256, first_divergence_index=41,
             baseline_at_divergence=7, arm_at_divergence=9)
    )
    assert "reported not gated" in line and "3 divergent of 256" in line
