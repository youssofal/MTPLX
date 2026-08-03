"""Activation-dtype gates for the DeepSeek-V4 MLX backend.

The reference runs the whole attention lane at the model dtype and uses fp32 only
as *math*, never as storage:

  * ``apply_rotary_emb`` rotates ``x.float()`` and copies the result back into the
    caller's own tensor (``y.copy_(x)``, model.py L234/L243), so a roped q / KV row
    comes back bf16.
  * the compressor pools in fp32 ("compression need fp32", L321-322) but casts the
    pooled row back before the norm (``kv = self.norm(kv.to(dtype))``, L362), and
    ``rotate_activation`` then *asserts* the row is bf16 (L249).
  * ``sparse_attn`` is declared ``q: BF16, kv: BF16, o: BF16`` with fp32 accumulator
    fragments, and casts the probability block to BF16 before the PV gemm
    (kernel.py L295-297, L305, L340).

This backend stored all three in fp32.  Because ``mx.concatenate`` and ``mx.matmul``
promote, one fp32 tensor was enough to pull the KV cache, both attention matmuls,
the o-LoRA einsum (which then had to upcast ``wo_a`` as well) and finally the whole
residual stream up to fp32 — on every layer, not only the compressed ones.

These tests pin the corrected flow, and pin that it is a **no-op at fp32**, which is
what keeps the parity/decode goldens (captured fp32) exactly where they were.
``MTPLX_DSV4_FP32_ACTIVATIONS=1`` restores the old promoting path as the A/B arm.

Self-contained: shrunk seeded config, no downloads, no torch, CPU device.  The
routed experts are quantised because MLX's dense ``gather_mm`` is fp32-only on CPU,
which is also the shape the real checkpoint has.
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
_spec = importlib.util.spec_from_file_location("dsv4_dtypes_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_dtypes_undertest"] = D
_spec.loader.exec_module(D)

VOCAB = 64
DIM = 32
N_HEADS = 4
HEAD_DIM = 16
ROPE_DIM = 8
N_EXPERTS = 8
RATIOS = [0, 4, 128, 4]      # every layer type: window, ratio-4, ratio-128, ratio-4
WINDOW = 16
GROUP_SIZE = 32
BITS = 4


def _quantisable(path, module):
    return path.endswith("attn.wo_a") or any(
        path.endswith(f"switch_mlp.{p}") for p in ("gate_proj", "up_proj", "down_proj")
    )


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


def _seeded_model(seed=0, dtype=None, quantise=True, **over):
    mx.random.seed(seed)
    args = _args(**over)
    model = D.Model(args)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if leaf == "tid2eid":
            new = mx.random.randint(0, args.n_routed_experts, value.shape).astype(mx.int32)
        elif value.ndim == 1:
            noise = mx.random.normal(value.shape) * 0.1
            centre = 1.0 if leaf in ("scale",) or name.endswith("norm.weight") else 0.0
            new = noise + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    if quantise:
        nn.quantize(model, group_size=GROUP_SIZE, bits=BITS, class_predicate=_quantisable)
    if dtype is not None:
        model.set_dtype(dtype)
    mx.eval(model.parameters())
    return args, model


def _tokens(seq_len, batch=1, seed=1234):
    mx.random.seed(seed)
    return mx.random.randint(0, VOCAB, (batch, seq_len))


def _run(model, ids, prompt_len=None):
    """One-shot logits plus the live cache from a prefill+decode run."""
    cache = model.make_cache()
    if prompt_len is None:
        prompt_len = ids.shape[1]
    pieces = [model(ids[:, :prompt_len], cache=cache)]
    for t in range(prompt_len, ids.shape[1]):
        pieces.append(model(ids[:, t:t + 1], cache=cache))
    out = mx.concatenate(pieces, axis=1)
    mx.eval(out)
    return out, cache


@pytest.fixture(autouse=True)
def _restore_flag():
    """Every test states the arm it wants; none may leak into the next."""
    saved = D._FP32_ACTIVATIONS
    yield
    D._FP32_ACTIVATIONS = saved


# ---------------------------------------------------------------------------
# env / knob
# ---------------------------------------------------------------------------
def test_fp32_escape_hatch_defaults_off(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV4_FP32_ACTIVATIONS", raising=False)
    assert D._env_flag("MTPLX_DSV4_FP32_ACTIVATIONS") is False
    for on in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MTPLX_DSV4_FP32_ACTIVATIONS", on)
        assert D._env_flag("MTPLX_DSV4_FP32_ACTIVATIONS") is True
    monkeypatch.setenv("MTPLX_DSV4_FP32_ACTIVATIONS", "0")
    assert D._env_flag("MTPLX_DSV4_FP32_ACTIVATIONS") is False


def test_store_dtype_follows_the_flag():
    D._FP32_ACTIVATIONS = False
    assert D._store_dtype(mx.bfloat16) == mx.bfloat16
    assert D._store_dtype(mx.float32) == mx.float32
    D._FP32_ACTIVATIONS = True
    assert D._store_dtype(mx.bfloat16) == mx.float32
    assert D._store_dtype(mx.float32) == mx.float32


# ---------------------------------------------------------------------------
# the flow itself
# ---------------------------------------------------------------------------
def test_rope_stores_at_the_input_dtype():
    """Reference ``apply_rotary_emb``: fp32 math, ``y.copy_(x)`` back into the
    caller's tensor.  fp32 cos/sin must not drag a bf16 activation up with them."""
    x = mx.random.normal((2, 4, 8)).astype(mx.bfloat16)
    cos = mx.random.normal((4, 4))
    sin = mx.random.normal((4, 4))
    D._FP32_ACTIVATIONS = False
    assert D._apply_interleaved_rope(x, cos, sin).dtype == mx.bfloat16
    assert D._apply_interleaved_rope(x.astype(mx.float32), cos, sin).dtype == mx.float32
    D._FP32_ACTIVATIONS = True
    assert D._apply_interleaved_rope(x, cos, sin).dtype == mx.float32


def test_bf16_model_keeps_every_activation_at_bf16():
    """No fp32 anywhere the reference does not have it: KV window, both compressed
    lanes, the pre-head hyper-connection state and the logits.

    140 tokens so the ratio-128 lane completes a window too — at 40 it emits
    nothing and the compressed-row assertions would be vacuous on that layer.
    """
    D._FP32_ACTIVATIONS = False
    _, model = _seeded_model(dtype=mx.bfloat16)
    ids = _tokens(140)

    logits, cache = _run(model, ids, prompt_len=13)
    assert logits.dtype == mx.bfloat16
    for i, c in enumerate(cache):
        assert c.window.dtype == mx.bfloat16, f"layer {i} window KV promoted"
        if c.compressed is not None:
            assert c.compressed.dtype == mx.bfloat16, f"layer {i} compressed rows promoted"
        if c.index_compressed is not None:
            assert c.index_compressed.dtype == mx.bfloat16, f"layer {i} index rows promoted"
    # every ratio!=0 layer really did emit rows, or the assertions above are vacuous
    assert [c.compressed is not None for c in cache] == [r != 0 for r in RATIOS]
    assert [c.index_compressed is not None for c in cache] == [r == 4 for r in RATIOS]

    h = model.model.hc_hidden(ids)
    mx.eval(h)
    assert h.dtype == mx.bfloat16, "residual stream promoted"
    one_shot = model(ids)
    mx.eval(one_shot)
    assert one_shot.dtype == mx.bfloat16


def test_fp32_escape_hatch_restores_the_promotion():
    """The A/B control: the pre-fix behaviour, still reachable, still fp32."""
    D._FP32_ACTIVATIONS = True
    _, model = _seeded_model(dtype=mx.bfloat16)
    logits, cache = _run(model, _tokens(140), prompt_len=13)
    assert logits.dtype == mx.float32
    for c in cache:
        assert c.window.dtype == mx.float32
        if c.compressed is not None:
            assert c.compressed.dtype == mx.float32


def test_fp32_model_is_bit_identical_between_arms():
    """The goldens do not move.

    Both parity goldens and the streaming-decode oracle were captured with an
    all-fp32 model, where every cast this change introduces is a no-op — so the two
    arms have to agree *exactly*, not approximately.  This is the whole reason no
    golden tolerance was touched.
    """
    _, model = _seeded_model(dtype=None)
    ids = _tokens(40)

    D._FP32_ACTIVATIONS = False
    fixed, cache_fixed = _run(model, ids, prompt_len=13)
    D._FP32_ACTIVATIONS = True
    legacy, cache_legacy = _run(model, ids, prompt_len=13)

    assert fixed.dtype == legacy.dtype == mx.float32
    assert mx.array_equal(fixed, legacy), "fp32 path is not bit-identical across arms"
    for a, b in zip(cache_fixed, cache_legacy):
        if a.compressed is not None:
            assert mx.array_equal(a.compressed, b.compressed)


def _arm_logits(arm, ids, **over):
    D._FP32_ACTIVATIONS = arm
    _, model = _seeded_model(dtype=mx.bfloat16, **over)
    out = model(ids)
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def test_bf16_arithmetic_gap_is_one_bf16_ulp_on_a_compressorless_layer():
    """The tight arithmetic gate.

    Layer 0 has ``compress_ratio == 0`` and hash routing, so the only differences
    between the arms there are the two that always apply: the roped q/KV stored at
    bf16 and the probability block cast to bf16 before the PV matmul.  bf16 carries
    8 mantissa bits (~3.9e-3 relative), so the attention output must land within a
    small multiple of one ulp — anything larger would mean a cast landed somewhere
    it changes the *math*, not just the storage.
    """
    ids = _tokens(140)

    def attn0(arm):
        D._FP32_ACTIVATIONS = arm
        _, model = _seeded_model(dtype=mx.bfloat16)
        layer = model.layers[0]
        assert layer.attn.compress_ratio == 0
        h = model.model.embed_tokens(ids)
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], model.args.hc_mult, h.shape[-1]))
        x, _, _ = layer.attn_hc.pre(h)
        out = layer.attn(layer.attn_norm(x), mask=None, cache=None)
        mx.eval(out)
        return np.array(out.astype(mx.float32))

    legacy, fixed = attn0(True), attn0(False)
    rel = float(np.max(np.abs(fixed - legacy)) / np.max(np.abs(legacy)))
    assert rel < 2e-2, f"compressorless attention gap {rel:.3e} is not bf16 rounding"


def test_compressed_rows_are_only_a_storage_cast_from_the_fp32_arm():
    """The compressor in isolation: same module, same input, both arms.

    The pooling stays fp32 in both — the reference says so outright — so the only
    admissible difference in the emitted row is the cast back to the model dtype.
    Feeding the module directly (rather than reading the rows out of a full forward)
    is the point: inside a forward the compressor's *input* has already drifted, and
    the comparison would measure that instead.
    """
    D._FP32_ACTIVATIONS = False
    _, model = _seeded_model(dtype=mx.bfloat16)
    mx.random.seed(7)
    x = (mx.random.normal((1, 140, DIM)) * 0.5).astype(mx.bfloat16)

    for i, ratio in enumerate(RATIOS):
        if not ratio:
            continue
        comp = model.layers[i].attn.compressor
        D._FP32_ACTIVATIONS = False
        fixed = comp(x)
        D._FP32_ACTIVATIONS = True
        legacy = comp(x)
        mx.eval(fixed, legacy)
        assert fixed.dtype == mx.bfloat16 and legacy.dtype == mx.float32
        assert fixed.shape == legacy.shape and fixed.shape[1] > 0
        a = np.array(legacy)
        b = np.array(fixed.astype(mx.float32))
        rel = float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
        # one bf16 ulp is ~3.9e-3; the cast lands twice (before the norm and on the
        # roped tail), so allow a small multiple of it and nothing more.
        assert rel < 1.5e-2, f"layer {i} (ratio {ratio}) rows moved {rel:.3e}, not a cast"


def test_end_to_end_bf16_gap_is_moe_routing_amplification_not_arithmetic():
    """Why there is no end-to-end bf16 argmax gate here.

    The per-layer arithmetic gap between the arms is one bf16 ulp (above).  End to
    end it is ~50%, and the reason is discrete, not arithmetic: a bf16-sized nudge
    flips which expert the MoE gate picks for a handful of near-tied tokens, and one
    flipped expert rewrites that token's residual completely.  In this shrunk fixture
    (8 random experts, top-2) the gate is near-tied constantly; the shipped model has
    256 trained experts, so the fixture cannot stand in for it either way.

    So: bound the arithmetic with the routing decision removed (every token routed to
    every expert -> no discrete branch left), and *exhibit* the flips rather than
    pretend the end-to-end number is a quality signal.  The real quality gate is a
    task eval on the real checkpoint, which is a GPU-window job.
    """
    ids = _tokens(140)

    # (1) no discrete decision left: pure arithmetic, through all four layers.
    legacy = _arm_logits(True, ids, num_experts_per_tok=N_EXPERTS)
    fixed = _arm_logits(False, ids, num_experts_per_tok=N_EXPERTS)
    rel = float(np.max(np.abs(fixed - legacy)) / np.max(np.abs(legacy)))
    assert rel < 1e-1, f"arithmetic-only bf16 gap {rel:.3e} is larger than accumulation"

    # (2) with top-k routing back on, the gap explodes — and the flips are there.
    def routes(arm):
        D._FP32_ACTIVATIONS = arm
        _, model = _seeded_model(dtype=mx.bfloat16)
        h = model.model.embed_tokens(ids)
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], model.args.hc_mult, h.shape[-1]))
        picked = []
        for layer in model.layers:
            residual = h
            x, post, comb = layer.attn_hc.pre(h)
            h = layer.attn_hc.post(
                layer.attn(layer.attn_norm(x), mask=None, cache=None), residual, post, comb
            )
            residual = h
            x, post, comb = layer.ffn_hc.pre(h)
            x = layer.ffn_norm(x)
            idx, _ = layer.ffn.gate(x.reshape(-1, DIM), ids.reshape(-1))
            mx.eval(idx)
            picked.append(np.sort(np.array(idx), axis=-1))
            h = layer.ffn_hc.post(layer.ffn(x, input_ids=ids), residual, post, comb)
        return picked

    legacy_routes, fixed_routes = routes(True), routes(False)
    flips = sum(
        int((a != b).any(-1).sum()) for a, b in zip(legacy_routes, fixed_routes)
    )
    topk_legacy = _arm_logits(True, ids)
    topk_fixed = _arm_logits(False, ids)
    end_to_end = float(
        np.max(np.abs(topk_fixed - topk_legacy)) / np.max(np.abs(topk_legacy))
    )
    assert flips > 0, (
        "no routing flip in the fixture — then the end-to-end gap needs another "
        "explanation and this test is telling the wrong story")
    assert end_to_end > rel, (
        f"top-k end-to-end gap {end_to_end:.3e} is not larger than the "
        f"arithmetic-only gap {rel:.3e}; the amplification story does not hold")


def test_bf16_streaming_decode_still_tracks_the_one_shot_forward():
    """The decode state machine keeps holding at the corrected dtypes.

    Prefill and decode reduce over different column counts, so at bf16 the attention
    scores round differently between the two lanes — that is inherent to storing the
    score block at bf16 and is why the fp32 oracle in
    tests/test_deepseek_v4_decode.py stays the gate for the *state machine*.  What is
    checked here is that the bf16 lane still tracks: with the discrete router taken
    out, decode reproduces prefill to a few bf16 ulps rather than drifting.
    """
    D._FP32_ACTIVATIONS = False
    _, model = _seeded_model(dtype=mx.bfloat16, num_experts_per_tok=N_EXPERTS)
    ids = _tokens(140)
    one_shot = np.array(model(ids).astype(mx.float32))
    streamed = np.array(_run(model, ids, prompt_len=13)[0].astype(mx.float32))
    rel = float(np.max(np.abs(streamed - one_shot)) / np.max(np.abs(one_shot)))
    assert rel < 5e-2, f"bf16 decode drifts from the one-shot oracle: rel={rel:.3e}"


# ---------------------------------------------------------------------------
# the MTP draft block
# ---------------------------------------------------------------------------
# ``DeepseekV4MTP`` subclasses ``DeepseekV4DecoderLayer``, so the draft head runs
# the same attention and inherits all three storage points by construction.  It is
# gated separately because it is reached through a different entry point
# (``Model.mtp_forward``) with a different cache (``make_mtp_cache``), and because
# it is the module whose output the accept/reject comparison consumes: an fp32
# draft against a bf16 target would silently make every verify a cross-dtype
# comparison.
def _mtp_model(dtype, seed=0):
    args, model = _seeded_model(seed=seed, dtype=dtype, num_nextn_predict_layers=1)
    assert model.mtp_blocks, "fixture built no draft block"
    return args, model


def _mtp_inputs(args, dtype, seq=40, seed=7):
    mx.random.seed(seed)
    h = (mx.random.normal((1, seq, args.hc_mult, args.hidden_size)) * 0.5).astype(dtype)
    return h, mx.random.randint(0, VOCAB, (1, seq))


def test_draft_block_keeps_its_activations_at_bf16():
    """Draft logits and the draft block's own KV window stay at the model dtype."""
    D._FP32_ACTIVATIONS = False
    args, model = _mtp_model(mx.bfloat16)
    h, ids = _mtp_inputs(args, mx.bfloat16)
    cache = model.make_mtp_cache()

    out = model.mtp_forward(h, ids, cache=cache[0])
    mx.eval(out)
    assert out.dtype == mx.bfloat16, "draft logits promoted"
    assert cache[0].window.dtype == mx.bfloat16, "draft window KV promoted"
    # compress_ratio is 0 on the draft layer, so there are no compressed rows to
    # check — asserting that rather than silently skipping it.
    assert cache[0].compressed is None


def test_draft_block_honours_the_fp32_escape_hatch():
    """The A/B control reaches the draft block too, or an arm would be half-applied."""
    D._FP32_ACTIVATIONS = True
    args, model = _mtp_model(mx.bfloat16)
    h, ids = _mtp_inputs(args, mx.bfloat16)
    cache = model.make_mtp_cache()
    out = model.mtp_forward(h, ids, cache=cache[0])
    mx.eval(out)
    assert out.dtype == mx.float32
    assert cache[0].window.dtype == mx.float32


def test_fp32_draft_block_is_bit_identical_between_arms():
    """At fp32 every cast is a no-op on the draft path as well as the trunk."""
    args, model = _mtp_model(dtype=None)
    h, ids = _mtp_inputs(args, mx.float32)

    D._FP32_ACTIVATIONS = False
    fixed = model.mtp_forward(h, ids, cache=model.make_mtp_cache()[0])
    D._FP32_ACTIVATIONS = True
    legacy = model.mtp_forward(h, ids, cache=model.make_mtp_cache()[0])
    mx.eval(fixed, legacy)

    assert fixed.dtype == legacy.dtype == mx.float32
    assert mx.array_equal(fixed, legacy), "draft fp32 path is not bit-identical across arms"
    assert len(set(np.array(mx.argmax(fixed[0], axis=-1)).tolist())) > 1
