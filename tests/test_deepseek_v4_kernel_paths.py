"""Gates for the two dispatch-structure levers on the DeepSeek-V4 decode path.

Both levers are pure restructurings — they must not move the model's numbers —
so every test here compares an arm against the path it replaced rather than
against a golden.  The goldens themselves are still gated, unchanged, by
tests/test_deepseek_v4_parity.py and tests/test_deepseek_v4_decode.py.

**Attention (``MTPLX_DSV4_ATTN``).**  ``dense`` is the pre-change path and the
oracle: a materialised score block, upcast to fp32, with ``attn_sink`` folded
into ``max`` and into the denominator by hand.  ``fused`` writes the same
softmax as ordinary attention by appending one all-zero KV row — whose raw score
is therefore exactly 0, and which contributes exactly nothing to the numerator
because the same row is the V row — and carrying the per-head sink as an additive
column on it.  ``sdpa`` hands the identical statement to
``mx.fast.scaled_dot_product_attention``'s native ``sinks=``.  All three are the
same real-number computation; they differ only in where the rounding lands, so
the fp32 lane is gated tight (1e-5 relative, argmax exact over 200+ decode steps)
and the bf16 lane is gated on argmax stability with the observed spread recorded.

**Hyper-Connections (``MTPLX_DSV4_HC_COMPILE``).**  The Sinkhorn chain is
deterministic arithmetic, so there is no tolerance to spend: :func:`_hc_pre_impl`
must be *bit-identical* to the reference transcription it replaces
(``_mixes`` + :func:`hc_split_sinkhorn` + the weighted sum, which is what
tests/test_deepseek_v4_new_math.py pins against the reference), and the compiled
tape must be bit-identical to the eager one.  Anything less would mean
``mx.compile`` had reassociated the arithmetic, which is exactly the thing that
would have to be caught here rather than in a quality run.

Self-contained: shrunk seeded config, no downloads, no torch, CPU device.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
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
_spec = importlib.util.spec_from_file_location("dsv4_kernels_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_kernels_undertest"] = D
_spec.loader.exec_module(D)

VOCAB = 64
DIM = 32
N_HEADS = 4
HEAD_DIM = 16
ROPE_DIM = 8
N_EXPERTS = 8
RATIOS = [0, 4, 128, 4]      # every layer type: window, ratio-4, ratio-128, ratio-4
WINDOW = 16

MODES = ("fused", "sdpa")


def _args(**over):
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=DIM, num_hidden_layers=len(RATIOS),
        num_hash_layers=1, num_attention_heads=N_HEADS, head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=32, n_routed_experts=N_EXPERTS, num_experts_per_tok=2,
        index_n_heads=N_HEADS, index_head_dim=HEAD_DIM, index_topk=512,
        compress_ratios=list(RATIOS), compress_rope_theta=160000.0,
        sliding_window=WINDOW,
        rope_scaling={"original_max_position_embeddings": 65536, "factor": 16,
                      "beta_fast": 32, "beta_slow": 1, "type": "yarn"},
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
    )
    kwargs.update(over)
    return D.ModelArgs(**kwargs)


def _quantisable(path, module):
    """What the shipped checkpoint quantises — and what MLX's dense ``GatherMM``
    forces here anyway, since it is fp32-only on CPU (so the bf16 lane cannot run
    with dense routed experts)."""
    return path.endswith("attn.wo_a") or any(
        path.endswith(f"switch_mlp.{p}") for p in ("gate_proj", "up_proj", "down_proj")
    )


def _seeded_model(seed=0, dtype=None, **over):
    mx.random.seed(seed)
    args = _args(**over)
    model = D.Model(args)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if leaf == "tid2eid":
            new = mx.random.randint(0, args.n_routed_experts, value.shape).astype(
                mx.int32
            )
        elif value.ndim == 1:
            centre = 1.0 if leaf == "scale" or name.endswith("norm.weight") else 0.0
            new = mx.random.normal(value.shape) * 0.1 + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    if dtype is not None:
        nn.quantize(model, group_size=32, bits=4, class_predicate=_quantisable)
        model.set_dtype(dtype)
    mx.eval(model.parameters())
    return args, model


def _tokens(seq_len, batch=1, seed=1234):
    mx.random.seed(seed)
    return mx.random.randint(0, VOCAB, (batch, seq_len))


def _set_mode(model, mode):
    for layer in model.model.layers:
        layer.attn.attn_mode = mode
    for block in getattr(model, "mtp", []):
        block.attn.attn_mode = mode


def _one_shot(mode, *, dtype=None, seed=0, seq=48, **over):
    _, model = _seeded_model(seed=seed, dtype=dtype, **over)
    _set_mode(model, mode)
    out = model(_tokens(seq))
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def _streaming(mode, *, dtype=None, seed=0, prompt=21, total=225, **over):
    _, model = _seeded_model(seed=seed, dtype=dtype, **over)
    _set_mode(model, mode)
    ids = _tokens(total)
    cache = model.make_cache()
    pieces = [model(ids[:, :prompt], cache=cache)]
    for t in range(prompt, total):
        pieces.append(model(ids[:, t : t + 1], cache=cache))
    out = mx.concatenate(pieces, axis=1)
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def _compare(ref, got, label, rel_tol):
    """Worst per-row relative error; argmax must agree everywhere."""
    worst = 0.0
    for row in range(ref.shape[0]):
        for t in range(ref.shape[1]):
            scale = float(np.max(np.abs(ref[row, t]))) + 1e-12
            worst = max(worst, float(np.max(np.abs(got[row, t] - ref[row, t]))) / scale)
    bad = int((got.argmax(-1) != ref.argmax(-1)).sum())
    assert bad == 0, f"{label}: argmax disagrees at {bad}/{ref[..., 0].size} rows"
    assert worst <= rel_tol, f"{label}: max_rel={worst:.3e} > {rel_tol:.0e}"
    return worst


@pytest.fixture(autouse=True)
def _restore_knobs():
    """Every test states the arm it wants; none may leak into the next."""
    saved = (D._FP32_ACTIVATIONS, D._HC_COMPILE, D._HC_COMPILE_MAX_ROWS)
    yield
    D._FP32_ACTIVATIONS, D._HC_COMPILE, D._HC_COMPILE_MAX_ROWS = saved


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------
def test_attn_mode_env(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV4_ATTN", raising=False)
    assert D._attn_mode_from_env() == "fused"
    for mode in D._ATTN_MODES:
        monkeypatch.setenv("MTPLX_DSV4_ATTN", mode.upper())
        assert D._attn_mode_from_env() == mode
    monkeypatch.setenv("MTPLX_DSV4_ATTN", "flash")
    with pytest.raises(ValueError):
        D._attn_mode_from_env()


def test_hc_compile_env(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV4_HC_COMPILE", raising=False)
    assert D._env_flag("MTPLX_DSV4_HC_COMPILE", True) is True
    monkeypatch.setenv("MTPLX_DSV4_HC_COMPILE", "0")
    assert D._env_flag("MTPLX_DSV4_HC_COMPILE", True) is False


# ---------------------------------------------------------------------------
# Lever A: attention arms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_attn_arm_matches_dense_one_shot(mode):
    """Whole-sequence forward, every layer type, fp32 lane."""
    ref = _one_shot("dense")
    assert len(set(ref[0].argmax(-1).tolist())) > 1, "oracle logits are degenerate"
    _compare(ref, _one_shot(mode), f"{mode} one-shot", 1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_attn_arm_matches_dense_sparse_regime(mode):
    """``index_topk`` crossed, so the ratio-4 indexer's top-k mask is live.

    That is the regime where the additive mask is not just causality: unselected
    compressed columns carry ``finfo.min``.  The appended sink column has to
    survive next to them — a padding bug there is invisible in the dense regime,
    where the mask is ``None`` on the decode step.
    """
    over = dict(index_topk=8)
    ref = _one_shot("dense", **over)
    got = _one_shot(mode, **over)
    # Guard that the arm is actually exercised rather than trivially skipped.
    _, probe = _seeded_model(**over)
    assert probe.model.layers[1].attn._indexer_active(12)
    _compare(ref, got, f"{mode} one-shot sparse", 1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_attn_arm_matches_dense_streaming_decode(mode):
    """204 single-token decode steps off a live cache, argmax exact throughout."""
    ref = _streaming("dense")
    got = _streaming(mode)
    assert ref.shape[1] - 21 >= 200
    _compare(ref, got, f"{mode} streaming", 1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_attn_arm_matches_dense_streaming_sparse(mode):
    over = dict(index_topk=8)
    ref = _streaming("dense", total=120, **over)
    got = _streaming(mode, total=120, **over)
    _compare(ref, got, f"{mode} streaming sparse", 1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_attn_arm_bf16_argmax_stable(mode):
    """bf16 lane: tolerance-gated, and the spread is recorded in the message.

    ``dense`` upcasts the whole block to fp32 before the softmax; ``fused`` keeps
    it at bf16 with fp32 accumulators inside ``mx.softmax(precise=True)``, and
    rounds ``attn_sink`` to bf16 before adding it as a column.  Both are bf16
    rounding of the same real number, so they cannot agree bit-for-bit; the bar
    is that the token the model emits does not move.
    """
    ref = _one_shot("dense", dtype=mx.bfloat16)
    got = _one_shot(mode, dtype=mx.bfloat16)
    worst = 0.0
    for t in range(ref.shape[1]):
        scale = float(np.max(np.abs(ref[0, t]))) + 1e-12
        worst = max(worst, float(np.max(np.abs(got[0, t] - ref[0, t]))) / scale)
    bad = int((got.argmax(-1) != ref.argmax(-1)).sum())
    assert bad == 0, f"{mode} bf16: argmax moved at {bad} rows (max_rel={worst:.3e})"
    assert worst <= 5e-2, f"{mode} bf16 max_rel={worst:.3e}"


def test_attn_arm_matches_dense_under_fp32_escape_hatch():
    """``MTPLX_DSV4_FP32_ACTIVATIONS=1`` promotes the KV block; the sink column
    and the zero KV row have to follow it rather than pin themselves to bf16."""
    D._FP32_ACTIVATIONS = True
    ref = _one_shot("dense", dtype=mx.bfloat16)
    for mode in MODES:
        _compare(ref, _one_shot(mode, dtype=mx.bfloat16), f"{mode} fp32-hatch", 1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_attn_sink_is_load_bearing(mode):
    """Mutation guard: the sink must reach the denominator on every arm.

    Without it, ``fused``'s extra column would be a plain zero-logit KV row with
    a zero value — invisible in the output — and the test above would pass while
    silently dropping the parameter.
    """
    _, model = _seeded_model()
    _set_mode(model, mode)
    ids = _tokens(24)
    base = np.array(model(ids).astype(mx.float32))
    for layer in model.model.layers:
        layer.attn.attn_sink = layer.attn.attn_sink + 4.0
    mx.eval(model.parameters())
    moved = np.array(model(ids).astype(mx.float32))
    assert not np.allclose(base, moved, atol=1e-4), (
        f"{mode}: shifting attn_sink by +4 left the logits unchanged"
    )


# ---------------------------------------------------------------------------
# Lever B: Hyper-Connection chain
# ---------------------------------------------------------------------------
def _hc_module(seed=3, hc=4, dim=DIM, eps=1e-6, iters=20):
    mx.random.seed(seed)
    h = D.HyperConnection(dim, hc, eps)
    h._iters = iters
    h.fn = mx.random.normal(h.fn.shape) * 0.1
    h.base = mx.random.normal(h.base.shape) * 0.1
    h.scale = mx.random.normal((3,)) * 0.1 + 1.0
    mx.eval(h.parameters())
    return h


def _reference_pre(h, x):
    """``pre`` exactly as it was written before the collapse: ``_mixes`` +
    :func:`hc_split_sinkhorn` (the reference transcription) + the weighted sum."""
    xf = x.astype(mx.float32)
    mixes = h._mixes(xf)
    pre, post, comb = D.hc_split_sinkhorn(
        mixes,
        h.scale.astype(mx.float32),
        h.base.astype(mx.float32),
        h.hc,
        h._iters,
        h.eps,
    )
    y = mx.sum(pre[..., None] * xf, axis=-2)
    return y.astype(x.dtype), post, comb


@pytest.mark.parametrize("rows", [(1, 1), (2, 5)])
def test_hc_pre_impl_bit_identical_to_reference_transcription(rows):
    """The fused-affine rewrite is the same arithmetic, not an approximation."""
    h = _hc_module()
    x = mx.random.normal((*rows, h.hc, h.dim))
    mx.eval(x)
    want = _reference_pre(h, x)
    fn_t, base, scale_vec = h._static()
    got = D._hc_pre_impl(x, fn_t, base, scale_vec, h.hc, h._iters, h.eps)
    for a, b, name in zip(got, want, ("y", "post", "comb")):
        mx.eval(a, b)
        assert mx.array_equal(a, b), (
            f"{name} not bit-identical: max_abs="
            f"{float(mx.max(mx.abs(a - b))):.3e}"
        )


def test_hc_compiled_bit_identical_to_eager():
    """``mx.compile`` must fuse without reassociating, at the decode shape.

    Bit-identity holds up to 4 rows, which is what B=1 decode and a short
    speculative verify batch are.  From 8 rows up the fused kernel vectorises
    differently and the two drift by a couple of ULP — measured below and pinned
    at 1e-6, three orders inside the 5e-5 the streaming-decode gate already
    spends.  This is the reason the row cap is a cap and not just a shape filter.
    """
    h = _hc_module()
    x = mx.random.normal((1, 1, h.hc, h.dim))
    residual = mx.random.normal((1, 1, h.hc, h.dim))
    mx.eval(x, residual)
    fn_t, base, scale_vec = h._static()

    compiled_pre = D._hc_compiled("pre", h.hc, h._iters, h.eps)
    eager = D._hc_pre_impl(x, fn_t, base, scale_vec, h.hc, h._iters, h.eps)
    comp = compiled_pre(x, fn_t, base, scale_vec)
    for a, b, name in zip(comp, eager, ("y", "post", "comb")):
        mx.eval(a, b)
        assert mx.array_equal(a, b), f"pre.{name} moved under compile"

    for rows, tol in ((2, 0.0), (4, 0.0), (8, 1e-6), (32, 1e-6)):
        xr = mx.random.normal((1, rows, h.hc, h.dim))
        mx.eval(xr)
        e = D._hc_pre_impl(xr, fn_t, base, scale_vec, h.hc, h._iters, h.eps)
        c = compiled_pre(xr, fn_t, base, scale_vec)
        for a, b, name in zip(c, e, ("y", "post", "comb")):
            mx.eval(a, b)
            rel = float(mx.max(mx.abs(a - b))) / (float(mx.max(mx.abs(b))) + 1e-12)
            assert rel <= tol, f"rows={rows} pre.{name} rel={rel:.3e} > {tol:.0e}"

    y, post, comb = eager
    pe = D._hc_post_impl(y, residual, post, comb)
    pc = D._hc_compiled("post")(y, residual, post, comb)
    mx.eval(pe, pc)
    assert mx.array_equal(pe, pc), "post moved under compile"

    mx.random.seed(9)
    head = D.HeadHC(h.dim, h.hc, h.eps)
    head.fn = mx.random.normal(head.fn.shape) * 0.1
    head.base = mx.random.normal(head.base.shape) * 0.1
    head.scale = mx.random.normal((1,)) * 0.1 + 1.0
    mx.eval(head.parameters())
    ft, bb, sc = head._static()
    he = D._hc_head_impl(x, ft, bb, sc, head.eps)
    hcp = D._hc_compiled("head", head.eps)(x, ft, bb, sc)
    mx.eval(he, hcp)
    assert mx.array_equal(he, hcp), "hc_head moved under compile"


def test_hc_compile_row_cap_switches_paths_without_moving_numbers():
    """The row cap changes which path runs; it must not change the answer.

    The cap exists because MLX keeps one tape per input shape in an unbounded,
    linearly scanned list (``CompilerCache::find``) and prefill shapes are
    effectively arbitrary.  Above it the eager path runs, and the two agree to
    the ULP bound recorded in
    :func:`test_hc_compiled_bit_identical_to_eager` — not bit-for-bit, because a
    multi-row fused kernel vectorises differently from the op chain it replaced.
    """
    h = _hc_module()
    x = mx.random.normal((1, 40, h.hc, h.dim))
    mx.eval(x)

    D._HC_COMPILE, D._HC_COMPILE_MAX_ROWS = True, 32
    assert D._hc_use_compile(x) is False          # 40 rows > cap
    capped = h.pre(x)

    D._HC_COMPILE_MAX_ROWS = 64
    assert D._hc_use_compile(x) is True
    compiled = h.pre(x)

    D._HC_COMPILE = False
    assert D._hc_use_compile(x) is False
    off = h.pre(x)

    for a, b, name in zip(off, capped, ("y", "post", "comb")):
        mx.eval(a, b)
        assert mx.array_equal(a, b), f"{name} moved with compile disabled"
    for a, b, name in zip(compiled, capped, ("y", "post", "comb")):
        mx.eval(a, b)
        rel = float(mx.max(mx.abs(a - b))) / (float(mx.max(mx.abs(b))) + 1e-12)
        assert rel <= 1e-6, f"{name} moved across the row cap: rel={rel:.3e}"


def test_hc_static_cache_invalidates_on_weight_rebind():
    """The derived fp32 weights are keyed on the parameter arrays, so an
    ``update``/``load_weights``/``set_dtype`` cannot be served a stale copy."""
    h = _hc_module()
    x = mx.random.normal((1, 1, h.hc, h.dim))
    mx.eval(x)
    before = h.pre(x)[0]
    mx.eval(before)

    h.update({"fn": h.fn * -1.0})
    mx.eval(h.parameters())
    after = h.pre(x)[0]
    mx.eval(after)
    assert not mx.array_equal(before, after), "stale _static cache served"
    # and the fresh value is the one the reference formulation gives
    assert mx.array_equal(after, _reference_pre(h, x)[0])


@pytest.mark.parametrize("dtype", [None, mx.bfloat16])
def test_decode_logits_bit_identical_with_and_without_hc_compile(dtype):
    """The lever's own regime: every forward one row, end to end, both lanes.

    This is the shape B=1 decode actually runs, it is inside the bit-identity
    band measured in :func:`test_hc_compiled_bit_identical_to_eager`, and it is
    where the lever is claimed to pay — so here the bar is equality, not a
    tolerance.
    """
    D._HC_COMPILE, D._HC_COMPILE_MAX_ROWS = True, 32
    assert D._hc_use_compile(mx.zeros((1, 1, 4, DIM))) is True, "nothing compiled"
    on = _streaming("fused", dtype=dtype, prompt=1, total=40)
    D._HC_COMPILE = False
    off = _streaming("fused", dtype=dtype, prompt=1, total=40)
    assert np.array_equal(on, off), (
        f"hc compile moved the decode logits: max_abs={np.max(np.abs(on - off)):.3e}"
    )


def test_multi_row_logits_track_across_hc_compile():
    """Multi-row prefill inside the cap: the ULP drift compounds over the layer
    stack, so state the bound end to end rather than per call.

    ~3e-6 relative at four layers, an order inside the 5e-5 the streaming-decode
    gate spends and three inside the 1e-3 the reference-golden gate spends.
    """
    D._HC_COMPILE, D._HC_COMPILE_MAX_ROWS = True, 32
    on = _one_shot("fused", seq=24)
    D._HC_COMPILE = False
    off = _one_shot("fused", seq=24)
    _compare(off, on, "hc compile multi-row", 1e-5)


def test_hc_compile_shared_across_modules_and_shapes():
    """One tape per (kind, constants) — not one per Hyper-Connection module.

    43 layers x 2 blocks share every shape and differ only in weight values,
    which are inputs; rebuilding the wrapper per module would retrace 86 times
    and leak a cache entry per call.
    """
    D._HC_COMPILE = True
    a = D._hc_compiled("pre", 4, 20, 1e-6)
    b = D._hc_compiled("pre", 4, 20, 1e-6)
    assert a is b
    assert D._hc_compiled("pre", 4, 19, 1e-6) is not a
    assert D._hc_compiled("post") is D._hc_compiled("post")
