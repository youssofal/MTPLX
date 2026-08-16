"""Gates for the DeepSeek-V4 output-LoRA weight path (``_o_lora``).

``wo_a`` is a *static* ``[o_groups*o_lora_rank, n_heads*head_dim/o_groups]``
matrix — ``[8192, 4096]`` on DeepSeek-V4-Flash — and the pre-cache backend ran
``mx.dequantize`` on it inside every call, i.e. per token, per layer, 43 layers
deep.  Three ways to consume it now exist (``MTPLX_DSV4_O_LORA``):

  * ``cached`` (default) — dequantise once, keep the dense result.  The cache holds
    exactly what ``mx.dequantize`` returned, so this must be **bit-identical** to
    the old path; that is what ``test_cached_dequant_is_bit_identical`` proves,
    with ``mx.array_equal`` on the logits, over a config carrying every layer type.
  * ``dequant`` — the old per-call behaviour, kept as the A/B control and as the
    oracle the bit-identity gate compares against.
  * ``gather_qmm`` — the quantised block-diagonal matmul, no dense tensor at all.
    Not bit-identical (the kernel dequantises inside the accumulation), so it is
    gated on tolerance + argmax stability and stays off by default.

Self-contained: shrunk seeded config, no downloads, no torch.  Runs on the CPU
device, same convention as the parity and decode tests.
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
_spec = importlib.util.spec_from_file_location("dsv4_olora_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_olora_undertest"] = D
_spec.loader.exec_module(D)

# Same shrunk config as tests/test_deepseek_v4_decode.py: every layer type the
# backend has (ratio-0 sliding window + hash routing, ratio-4 overlap compressor
# + indexer, ratio-128 non-overlap compressor, ratio-4 on the score-routed side).
# ``wo_a`` lands at [g*r, per] = [16, 32], which quantises at group_size 32.
VOCAB = 64
DIM = 32
N_HEADS = 4
HEAD_DIM = 16
ROPE_DIM = 8
N_EXPERTS = 8
RATIOS = [0, 4, 128, 4]
WINDOW = 16
O_GROUPS = 2
O_RANK = 8
GROUP_SIZE = 32
BITS = 4


def _args(**over):
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=DIM, num_hidden_layers=len(RATIOS),
        num_hash_layers=1, num_attention_heads=N_HEADS, head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM, q_lora_rank=16, o_lora_rank=O_RANK,
        o_groups=O_GROUPS, moe_intermediate_size=16, n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2, index_n_heads=N_HEADS, index_head_dim=HEAD_DIM,
        index_topk=512, compress_ratios=list(RATIOS), compress_rope_theta=160000.0,
        sliding_window=WINDOW,
        rope_scaling={"original_max_position_embeddings": 65536, "factor": 16,
                      "beta_fast": 32, "beta_slow": 1, "type": "yarn"},
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
    )
    kwargs.update(over)
    return D.ModelArgs(**kwargs)


def _seeded_model(seed=0, **over):
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
    mx.eval(model.parameters())
    return args, model


def _quantized_model(seed=0, **over):
    """Seeded model with **only** ``wo_a`` quantised.

    Quantising just the module under test keeps every other tensor bit-identical
    between arms, so a logits difference can only have come from ``_o_lora``.
    """
    args, model = _seeded_model(seed=seed, **over)
    nn.quantize(
        model, group_size=GROUP_SIZE, bits=BITS,
        class_predicate=lambda path, m: path.endswith("attn.wo_a"),
    )
    mx.eval(model.parameters())
    assert isinstance(model.layers[0].attn.wo_a, nn.QuantizedLinear)
    return args, model


def _tokens(seq_len, batch=1, seed=1234):
    mx.random.seed(seed)
    return mx.random.randint(0, VOCAB, (batch, seq_len))


def _set_mode(model, mode):
    for layer in model.layers:
        layer.attn.install_o_lora_route(mode)


def _decode(model, ids, prompt_len):
    cache = model.make_cache()
    pieces = [np.array(model(ids[:, :prompt_len], cache=cache).astype(mx.float32))]
    for t in range(prompt_len, ids.shape[1]):
        pieces.append(np.array(model(ids[:, t:t + 1], cache=cache).astype(mx.float32)))
    return np.concatenate(pieces, axis=1)


# ---------------------------------------------------------------------------
# env parsing
# ---------------------------------------------------------------------------
def test_default_mode_is_cached(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV4_O_LORA", raising=False)
    assert D._o_lora_mode_from_env() == "cached"
    monkeypatch.setenv("MTPLX_DSV4_O_LORA", "  ")
    assert D._o_lora_mode_from_env() == "cached"


def test_env_selects_each_mode(monkeypatch):
    for mode in D._O_LORA_MODES:
        monkeypatch.setenv("MTPLX_DSV4_O_LORA", mode.upper())
        assert D._o_lora_mode_from_env() == mode


def test_unknown_mode_is_rejected_loudly(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV4_O_LORA", "fast")
    with pytest.raises(ValueError, match="MTPLX_DSV4_O_LORA"):
        D._o_lora_mode_from_env()


def test_attention_picks_the_mode_up_at_construction(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV4_O_LORA", "gather_qmm")
    _, model = _seeded_model()
    assert [layer.attn.o_lora_mode for layer in model.layers] == [
        "gather_qmm"
    ] * len(RATIOS)


# ---------------------------------------------------------------------------
# arm (a): cached dequant is bit-identical
# ---------------------------------------------------------------------------
def test_cached_dequant_is_bit_identical():
    """The whole point of the cache: same weights + same input -> same logits.

    Not "close" — ``mx.array_equal``.  The cache stores exactly the array
    ``mx.dequantize`` produced, so the consuming einsum is handed the identical
    values it was handed before, on every layer type at once.
    """
    _, model = _quantized_model()
    ids = _tokens(40)

    _set_mode(model, "dequant")
    ref = model(ids)
    mx.eval(ref)
    assert all(layer.attn._wo_a_cache.value is None for layer in model.layers), (
        "the dequant arm must not populate the cache, or it is not a control")

    _set_mode(model, "cached")
    first = model(ids)          # populates
    second = model(ids)         # serves from cache
    mx.eval(first, second)

    assert all(layer.attn._wo_a_cache.value is not None for layer in model.layers), (
        "cache never populated — the gate would be vacuous")
    assert mx.array_equal(ref, first), "cached first call is not bit-identical"
    assert mx.array_equal(ref, second), "cached cache-hit call is not bit-identical"
    # A degenerate model would make the comparison meaningless.
    assert len(set(np.array(mx.argmax(ref[0], axis=-1)).tolist())) > 1


def test_cached_dequant_is_bit_identical_through_streaming_decode():
    """Same proof on the incremental path, where ``_o_lora`` runs once per token."""
    _, model = _quantized_model()
    ids = _tokens(40)
    _set_mode(model, "dequant")
    ref = _decode(model, ids, 13)
    _set_mode(model, "cached")
    got = _decode(model, ids, 13)
    assert np.array_equal(ref, got), (
        f"streaming decode diverged: max_abs={np.max(np.abs(got - ref)):.3e}")


def test_cache_is_reused_not_rebuilt():
    """Identity, not just equality: a rebuilt cache would be the bug the lever exists
    to remove, and it would still pass a value comparison."""
    _, model = _quantized_model()
    ids = _tokens(20)
    _set_mode(model, "cached")
    model(ids)
    held = [layer.attn._wo_a_cache.value for layer in model.layers]
    model(ids)
    again = [layer.attn._wo_a_cache.value for layer in model.layers]
    assert all(a is b for a, b in zip(held, again))


def test_cache_is_invalidated_when_the_weights_are_rebound():
    """``load_weights``/``update``/``set_dtype`` rebind the quantised tensors; the
    cache is keyed on their identity so it cannot serve a stale dense copy."""
    _, model = _quantized_model()
    ids = _tokens(20)
    _set_mode(model, "cached")
    model(ids)
    stale = [layer.attn._wo_a_cache.value for layer in model.layers]

    # Rebind scales only — the packed weight object stays the same, which is
    # exactly what a dtype cast or a partial reload does.
    for layer in model.layers:
        layer.attn.wo_a.scales = layer.attn.wo_a.scales * 1.5
    mx.eval(model.parameters())

    got = model(ids)
    _set_mode(model, "dequant")
    ref = model(ids)
    mx.eval(got, ref)
    assert mx.array_equal(ref, got), "stale dense cache served after a weight rebind"
    fresh = [layer.attn._wo_a_cache.value for layer in model.layers]
    assert all(a is not b for a, b in zip(stale, fresh))


def test_cache_never_reaches_the_weight_tree():
    """The derived tensor must be invisible to ``parameters()``/``load_weights``."""
    _, model = _quantized_model()
    before = {k for k, _ in tree_flatten(model.parameters())}
    _set_mode(model, "cached")
    model(_tokens(20))
    after = {k for k, _ in tree_flatten(model.parameters())}
    assert before == after
    attn = model.layers[0].attn
    assert attn._wo_a_cache.value is not None
    # nn.Module is a dict; the cache must not be in it under any key.
    assert not any(k.startswith("_wo_a") for k in dict(attn))
    # strict load still matches exactly
    model.load_weights(tree_flatten(model.parameters()), strict=True)


def test_unquantised_wo_a_rejects_gather_at_installation():
    """An enabled gather route cannot silently execute the dense implementation."""
    _, model = _seeded_model()
    ids = _tokens(20)
    outs = []
    for mode in ("cached", "dequant"):
        _set_mode(model, mode)
        out = model(ids)
        mx.eval(out)
        outs.append(out)
    with pytest.raises(ValueError, match="quantized"):
        _set_mode(model, "gather_qmm")
    assert all(layer.attn._wo_a_cache.value is None for layer in model.layers)
    assert mx.array_equal(outs[0], outs[1])


# ---------------------------------------------------------------------------
# arm (b): grouped quantised matmul
# ---------------------------------------------------------------------------
def test_gather_qmm_calling_convention_shapes():
    """``x`` carries the group axis in its batch dims: ``[g, rows, per]`` in,
    ``[g, rows, r]`` out — including the decode shape ``[g, 1, per]``."""
    _, model = _quantized_model()
    attn = model.layers[0].attn
    per = N_HEADS * HEAD_DIM // O_GROUPS
    for rows in (1, 7):
        o = mx.random.normal((1, rows, N_HEADS * HEAD_DIM))
        x = o.reshape(rows, O_GROUPS, per).swapaxes(0, 1)
        assert tuple(x.shape) == (O_GROUPS, rows, per)
        w, sc, bi, gs, bits, mode = attn._wo_a_quant()
        out = mx.gather_qmm(
            x, w.reshape(O_GROUPS, O_RANK, -1), sc.reshape(O_GROUPS, O_RANK, -1),
            bi.reshape(O_GROUPS, O_RANK, -1), transpose=True,
            group_size=gs, bits=bits, mode=mode,
        )
        mx.eval(out)
        assert tuple(out.shape) == (O_GROUPS, rows, O_RANK)
        # ...and it computes the same block-diagonal product as the dense einsum.
        dense = attn._wo_a_grouped()
        ref = mx.einsum("bsgp,grp->bsgr", o.reshape(1, rows, O_GROUPS, per), dense)
        got = out.swapaxes(0, 1).reshape(1, rows, O_GROUPS, O_RANK)
        rel = float(mx.max(mx.abs(got - ref))) / (float(mx.max(mx.abs(ref))) + 1e-12)
        assert rel < 1e-3, f"rows={rows} grouped qmm rel={rel:.3e}"


def test_gather_qmm_accepts_mlx_bitstream_packed_q6_rows():
    """Q6 packs 16 values across three uint32 words, not 32//bits per word."""
    _, model = _seeded_model(o_lora_rank=16)
    nn.quantize(
        model,
        group_size=32,
        bits=6,
        class_predicate=lambda path, _module: path.endswith(
            ("attn.wo_a", "attn.wo_b")
        ),
    )
    mx.eval(model.parameters())
    attn = model.layers[0].attn
    assert tuple(attn.wo_a.weight.shape) == (32, 6)
    assert D._o_lora_linear_logical_weight_shape(attn.wo_a) == (32, 32)
    assert D._o_lora_linear_logical_weight_shape(attn.wo_b) == (32, 32)
    o = mx.sin(mx.arange(N_HEADS * HEAD_DIM, dtype=mx.float32) * 0.017)
    o = o.reshape(1, 1, -1).astype(mx.bfloat16)
    want = attn._o_lora_dense(o)

    receipt = attn.install_o_lora_route("gather_qmm")
    got = attn._o_lora(o)
    mx.eval(want, got)

    assert receipt["direct"] is True
    assert tuple(got.shape) == tuple(want.shape)
    assert float(mx.max(mx.abs(got - want))) <= 0.125


def test_gather_qmm_is_prebound_and_never_rechecks_or_falls_back(monkeypatch):
    _, model = _quantized_model()
    attn = model.layers[0].attn
    o = mx.random.normal((1, 3, N_HEADS * HEAD_DIM))
    receipt = attn.install_o_lora_route("gather_qmm")
    assert receipt["direct"] is True
    installed = attn._o_lora_impl
    monkeypatch.setattr(
        attn,
        "_wo_a_quant",
        lambda: (_ for _ in ()).throw(AssertionError("hot eligibility lookup")),
    )
    got = attn._o_lora(o)
    mx.eval(got)
    assert attn._o_lora_impl is installed
    assert tuple(got.shape) == (1, 3, DIM)


def test_gather_route_preserves_and_calls_a_real_quantized_wo_b():
    _, model = _seeded_model(o_lora_rank=16)
    nn.quantize(
        model,
        group_size=GROUP_SIZE,
        bits=BITS,
        class_predicate=lambda path, _module: path.endswith(
            ("attn.wo_a", "attn.wo_b")
        ),
    )
    mx.eval(model.parameters())
    attn = model.layers[0].attn
    assert isinstance(attn.wo_a, nn.QuantizedLinear)
    assert isinstance(attn.wo_b, nn.QuantizedLinear)
    stock_wo_b = attn.wo_b
    assert getattr(stock_wo_b, "bias", None) is None
    stock_storage = (
        stock_wo_b.weight,
        stock_wo_b.scales,
        stock_wo_b.biases,
        stock_wo_b.bits,
        stock_wo_b.group_size,
        stock_wo_b.mode,
    )
    assert D._o_lora_linear_logical_weight_shape(stock_wo_b) == (DIM, 32)

    receipt = attn.install_o_lora_route("gather_qmm")
    assert receipt["direct"] is True
    assert attn._o_lora_impl.wo_b is stock_wo_b
    result = attn._o_lora(mx.random.normal((1, 3, N_HEADS * HEAD_DIM)))
    mx.eval(result)
    assert tuple(result.shape) == (1, 3, DIM)
    assert stock_wo_b.weight is stock_storage[0]
    assert stock_wo_b.scales is stock_storage[1]
    assert stock_wo_b.biases is stock_storage[2]
    assert (stock_wo_b.bits, stock_wo_b.group_size, stock_wo_b.mode) == stock_storage[3:]


def test_gather_qmm_matches_the_cached_arm_within_tolerance():
    """Arm (b) is *not* bit-identical — the quantised kernel dequantises inside the
    accumulation.  What it must hold is the numeric envelope and the decision:
    every argmax the default arm makes, arm (b) makes too."""
    _, model = _quantized_model()
    ids = _tokens(40)
    _set_mode(model, "cached")
    ref = np.array(model(ids).astype(mx.float32))
    _set_mode(model, "gather_qmm")
    got = np.array(model(ids).astype(mx.float32))

    assert not np.array_equal(ref, got), (
        "arm (b) came out bit-identical — either it is not running or the tolerance "
        "gate is testing the wrong thing")
    scale = float(np.max(np.abs(ref)))
    rel = float(np.max(np.abs(got - ref))) / (scale + 1e-12)
    assert rel < 2e-3, f"grouped qmm logits rel={rel:.3e}"
    assert np.array_equal(ref[0].argmax(-1), got[0].argmax(-1)), (
        f"argmax moved under gather_qmm (rel={rel:.3e})")


def test_gather_qmm_precision_drops_at_bf16_and_that_is_arm_bs_open_risk():
    """The number the GPU window has to re-measure before arm (b) can be defaulted on.

    Against an fp32 activation the quantised kernel tracks the dense einsum to ~1e-6
    (above).  Hand it bf16 activations and the *CPU* kernel loses two orders of
    magnitude — it accumulates the dequantised products at lower precision than the
    dense matmul does — while the dense bf16 einsum stays near one bf16 ulp.

    Metal's ``gather_qmm`` is a different kernel with fp32 simdgroup accumulation, so
    this is the pessimistic end, not a prediction.  But it is measured and it is on
    the wrong side, so the bound is pinned here rather than left to be discovered:
    if a Metal A/B shows the same gap, arm (b)'s bytes are not worth its accuracy.
    """
    _, model = _quantized_model()
    attn = model.layers[0].attn
    attn.set_dtype(mx.bfloat16)          # attention only: no MoE, no gather_mm
    mx.eval(attn.parameters())
    mx.random.seed(3)
    o = (mx.random.normal((1, 140, N_HEADS * HEAD_DIM)) * 0.5).astype(mx.bfloat16)

    attn.install_o_lora_route("cached")
    ref = attn._o_lora(o)
    attn.install_o_lora_route("gather_qmm")
    got = attn._o_lora(o)
    mx.eval(ref, got)
    assert ref.dtype == got.dtype == mx.bfloat16
    a = np.array(ref.astype(mx.float32))
    b = np.array(got.astype(mx.float32))
    rel = float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
    # ~1.3e-2 on this box's CPU kernel — several bf16 ulps, not one.
    assert 1e-3 < rel < 5e-2, (
        f"bf16 gather_qmm gap {rel:.3e} moved; re-derive the arm (b) risk note")


def test_gather_qmm_holds_through_streaming_decode():
    _, model = _quantized_model()
    ids = _tokens(40)
    _set_mode(model, "cached")
    ref = _decode(model, ids, 13)
    _set_mode(model, "gather_qmm")
    got = _decode(model, ids, 13)
    scale = float(np.max(np.abs(ref)))
    rel = float(np.max(np.abs(got - ref))) / (scale + 1e-12)
    assert rel < 2e-3, f"grouped qmm decode rel={rel:.3e}"
    assert np.array_equal(ref[0].argmax(-1), got[0].argmax(-1))


# ---------------------------------------------------------------------------
# the MTP draft block: same attention class, its own instance
# ---------------------------------------------------------------------------
# ``DeepseekV4MTP`` subclasses ``DeepseekV4DecoderLayer``, so the draft head runs
# the *same* ``DeepseekV4Attention`` and inherits every mode above by
# construction.  Inheriting it is not the same as proving it: the gates further
# up build a model with no draft block and drive ``model.layers``, which the
# draft block is not in.  The risk the cache introduces is specifically a
# cross-instance one — one dense ``wo_a`` served to an attention it does not
# belong to — and the draft block is the instance most likely to be missed,
# because it is bound outside the layer list and reached through a different
# entry point (``Model.mtp_forward``).  These gates cover that instance directly.
def _quantized_mtp_model(seed=0):
    """Seeded model **with** a draft block; ``wo_a`` quantised on every attention."""
    args, model = _quantized_model(seed=seed, num_nextn_predict_layers=1)
    assert model.mtp_blocks, "fixture built no draft block — the gates below are vacuous"
    assert isinstance(model.mtp_blocks[0].attn.wo_a, nn.QuantizedLinear)
    return args, model


def _attentions(model):
    return [layer.attn for layer in model.layers] + [
        block.attn for block in model.mtp_blocks
    ]


def _set_mode_everywhere(model, mode):
    for attn in _attentions(model):
        attn.install_o_lora_route(mode)


class _O_LoraArrayMeta:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def reshape(self, *shape):
        shape = tuple(shape)
        if shape.count(-1) > 1:
            raise ValueError("only one inferred reshape dimension is supported")
        if -1 in shape:
            known = int(
                np.prod([dimension for dimension in shape if dimension != -1])
            )
            total = int(np.prod(self.shape))
            shape = tuple(
                total // known if dimension == -1 else dimension
                for dimension in shape
            )
        return _O_LoraArrayMeta(shape, self.dtype)


class _CanonicalEmbedding:
    def __init__(self, output_dtype):
        self.output_dtype = output_dtype

    def __call__(self, input_ids):
        return _O_LoraArrayMeta((*input_ids.shape, 4096), self.output_dtype)


class _CanonicalQuantizedLinear:
    pass


class _CanonicalDenseLinear:
    pass


class _CanonicalBodyWOA(_CanonicalQuantizedLinear):
    def __init__(self):
        self.weight = _O_LoraArrayMeta((8192, 512), mx.uint32)
        self.scales = _O_LoraArrayMeta((8192, 64), mx.bfloat16)
        self.biases = _O_LoraArrayMeta((8192, 64), mx.bfloat16)
        self.bits = 4
        self.group_size = 64
        self.mode = "affine"
        self.bias = None


class _CanonicalBodyWOB(_CanonicalQuantizedLinear):
    def __init__(self):
        self.weight = _O_LoraArrayMeta((4096, 1024), mx.uint32)
        self.scales = _O_LoraArrayMeta((4096, 128), mx.bfloat16)
        self.biases = _O_LoraArrayMeta((4096, 128), mx.bfloat16)
        self.bits = 4
        self.group_size = 64
        self.mode = "affine"
        self.bias = None


class _CanonicalMTPWOA(_CanonicalDenseLinear):
    def __init__(self):
        self.weight = _O_LoraArrayMeta((8192, 4096), mx.bfloat16)
        self.bias = None


class _CanonicalMTPWOB(_CanonicalDenseLinear):
    def __init__(self):
        self.weight = _O_LoraArrayMeta((4096, 8192), mx.bfloat16)
        self.bias = None


class _RouteFakeAttention:
    def __init__(self, wo_a, wo_b):
        self.wo_a = wo_a
        self.wo_b = wo_b
        self.n_groups = 8
        self.o_lora_rank = 1024
        self.n_heads = 64
        self.head_dim = 512
        self.dim = 4096
        self.o_lora_mode = None
        self._o_lora_impl = None


class _RouteBox:
    def __init__(self, wo_a, wo_b):
        self.attn = _RouteFakeAttention(wo_a, wo_b)


def _canonical_route_model(
    *, body_count=43, mtp_count=1, activation_dtype=mx.bfloat16
):
    class FakeModel:
        def __init__(self):
            self.model = type(
                "_CanonicalTrunk",
                (),
                {"embed_tokens": _CanonicalEmbedding(activation_dtype)},
            )()
            self.layers = [
                _RouteBox(_CanonicalBodyWOA(), _CanonicalBodyWOB())
                for _ in range(body_count)
            ]
            self.mtp_blocks = [
                _RouteBox(_CanonicalMTPWOA(), _CanonicalMTPWOB())
                for _ in range(mtp_count)
            ]

    return FakeModel()


class _FakeCachedBodyRoute:
    def __init__(self, attention, quant):
        self.attention = attention
        self.quant = quant
        self.wo_b = attention.wo_b


class _FakeGatherBodyRoute:
    def __init__(self, attention, quant):
        self.attention = attention
        self.quant = quant
        self.wo_b = attention.wo_b


class _FakeDenseMTPRoute:
    def __init__(self, attention, weight):
        self.attention = attention
        self.weight = weight
        self.wo_b = attention.wo_b


_FakeCachedBodyRoute.__name__ = "_DirectCachedOLora"
_FakeGatherBodyRoute.__name__ = "_DirectGatherOLora"
_FakeDenseMTPRoute.__name__ = "_DirectDenseMTPOLora"


class _FakeGatherWideM4BodyRoute:
    def __init__(self, attention, quant, *, activation_dtype):
        self.attention = attention
        self.quant = quant
        self.activation_dtype = activation_dtype
        self.wo_b = attention.wo_b


_FakeGatherWideM4BodyRoute.__name__ = "_DirectGatherOLoraWideM4"


def _patch_canonical_route_types(monkeypatch):
    monkeypatch.setattr(D.nn, "QuantizedLinear", _CanonicalQuantizedLinear)
    monkeypatch.setattr(D.nn, "Linear", _CanonicalDenseLinear)
    monkeypatch.setattr(D, "_DirectCachedOLora", _FakeCachedBodyRoute)
    monkeypatch.setattr(D, "_DirectGatherOLora", _FakeGatherBodyRoute)
    monkeypatch.setattr(D, "_DirectGatherOLoraWideM4", _FakeGatherWideM4BodyRoute)
    monkeypatch.setattr(D, "_DirectDenseMTPOLora", _FakeDenseMTPRoute)
    monkeypatch.setattr(D, "_validate_gather_qmm_wide_m4_body_routes", lambda _body: None)


def test_model_installer_prebinds_only_body_m4_wide_and_keeps_mtp_stock(monkeypatch):
    """The M4-wide candidate is an authenticated body route, never MTP.

    The production callable owns the fixed ``[8, 4, 4096]`` gathered input and
    selects its direct wide entry point only for physical M4.  This construction
    test locks the boundary: all 43 quantized body modules receive that route,
    while the dense MTP module remains the explicitly prebound stock callable.
    """
    _patch_canonical_route_types(monkeypatch)
    model = _canonical_route_model()

    report = D.install_deepseek_v4_o_lora_routes(
        model, mode="gather_qmm", canonical_mixed_route=True
    )

    assert all(
        isinstance(box.attn._o_lora_impl, _FakeGatherWideM4BodyRoute)
        for box in model.layers
    )
    assert isinstance(model.mtp_blocks[0].attn._o_lora_impl, _FakeDenseMTPRoute)
    assert report["callable_census"]["body_route_kind"] == "gather_qmm_m4_wide_direct"
    assert report["callable_census"]["body_callable_class"] == "_DirectGatherOLoraWideM4"
    assert report["callable_census"]["mtp_route_kind"] == "dense_bf16_stock_direct"


def test_model_installer_selfchecks_the_real_weight_sentinels_before_publish(monkeypatch):
    """Layer 0/3/42 parity is a construction gate, not decode instrumentation."""
    _patch_canonical_route_types(monkeypatch)
    model = _canonical_route_model()
    checked = []
    monkeypatch.setattr(
        D,
        "_validate_gather_qmm_wide_m4_body_routes",
        lambda body: checked.extend(body),
        raising=False,
    )

    D.install_deepseek_v4_o_lora_routes(
        model, mode="gather_qmm", canonical_mixed_route=True
    )

    assert len(checked) == 43
    assert all(isinstance(route, _FakeGatherWideM4BodyRoute) for route in checked)


def test_m4_wide_installer_rejects_the_fp32_activation_ab_arm(monkeypatch):
    _patch_canonical_route_types(monkeypatch)
    monkeypatch.setattr(D, "_FP32_ACTIVATIONS", True)
    model = _canonical_route_model()

    with pytest.raises(ValueError, match="BF16 activation storage"):
        D.install_deepseek_v4_o_lora_routes(
            model, mode="gather_qmm", canonical_mixed_route=True
        )

    assert all(box.attn._o_lora_impl is None for box in model.layers + model.mtp_blocks)


def test_m4_wide_installer_rejects_non_bf16_body_activation_output(monkeypatch):
    _patch_canonical_route_types(monkeypatch)
    model = _canonical_route_model(activation_dtype=mx.float16)

    with pytest.raises(ValueError, match="embedding output dtype.*bfloat16"):
        D.install_deepseek_v4_o_lora_routes(
            model, mode="gather_qmm", canonical_mixed_route=True
        )

    assert all(box.attn._o_lora_impl is None for box in model.layers + model.mtp_blocks)


def _canonical_body_quant(attention):
    wo_a = attention.wo_a
    return (
        wo_a.weight,
        wo_a.scales,
        wo_a.biases,
        wo_a.group_size,
        wo_a.bits,
        wo_a.mode,
    )


def test_m4_wide_concrete_route_binds_bf16_kernel_aux_and_output(monkeypatch):
    """The real route fixes one BF16 Metal type at construction, not from input."""
    definition = {}
    launch = {}

    def fake_metal_kernel(**kwargs):
        definition.update(kwargs)

        def run(**kwargs):
            launch.update(kwargs)
            output = _O_LoraArrayMeta(
                kwargs["output_shapes"][0], kwargs["output_dtypes"][0]
            )
            return (output,)

        return run

    monkeypatch.setattr(D.nn, "QuantizedLinear", _CanonicalQuantizedLinear)
    monkeypatch.setattr(D.mx.fast, "metal_kernel", fake_metal_kernel)
    kernel = D._gather_qmm_wide_m4_olora_kernel.__wrapped__()
    monkeypatch.setattr(D, "_gather_qmm_wide_m4_olora_kernel", lambda: kernel)
    monkeypatch.setattr(
        D.mx,
        "arange",
        lambda size, *, dtype: _O_LoraArrayMeta((size,), dtype),
    )
    attention = _RouteFakeAttention(_CanonicalBodyWOA(), _CanonicalBodyWOB())

    route = D._DirectGatherOLoraWideM4(
        attention,
        _canonical_body_quant(attention),
        activation_dtype=mx.bfloat16,
    )
    output = route.m4.grouped(
        _O_LoraArrayMeta((4, 8, 4096), mx.bfloat16),
        _O_LoraArrayMeta((8,), mx.uint32),
    )

    assert type(route) is D._DirectGatherOLoraWideM4
    assert "const device T* srow = scales" in definition["source"]
    assert "const device T* brow = biases" in definition["source"]
    assert launch["template"] == [("T", mx.bfloat16)]
    assert launch["output_dtypes"] == [mx.bfloat16]
    assert route.m4.weight.shape == (8, 1024, 512)
    assert route.m4.scales.shape == (8, 1024, 64)
    assert route.m4.biases.shape == (8, 1024, 64)
    assert output.dtype == mx.bfloat16


@pytest.mark.parametrize("field", ["scales", "biases"])
def test_m4_wide_concrete_route_rejects_non_bf16_aux(monkeypatch, field):
    monkeypatch.setattr(D.nn, "QuantizedLinear", _CanonicalQuantizedLinear)
    attention = _RouteFakeAttention(_CanonicalBodyWOA(), _CanonicalBodyWOB())
    setattr(attention.wo_a, field, _O_LoraArrayMeta((8192, 64), mx.float16))

    with pytest.raises(ValueError, match=f"{field} dtype.*bfloat16"):
        D._DirectGatherOLoraWideM4(
            attention,
            _canonical_body_quant(attention),
            activation_dtype=mx.bfloat16,
        )


def test_m4_wide_concrete_route_rejects_non_bf16_activation_contract(monkeypatch):
    monkeypatch.setattr(D.nn, "QuantizedLinear", _CanonicalQuantizedLinear)
    attention = _RouteFakeAttention(_CanonicalBodyWOA(), _CanonicalBodyWOB())

    with pytest.raises(ValueError, match="activation/output dtype.*bfloat16"):
        D._DirectGatherOLoraWideM4(
            attention,
            _canonical_body_quant(attention),
            activation_dtype=mx.float16,
        )


def test_m4_wide_route_dispatches_only_the_physical_four_row_body_shape():
    """M is runtime routing; AR/M1 and non-M4 verifies retain stock gather."""
    calls = []

    class _Input:
        def __init__(self, batch, sequence):
            self.shape = (batch, sequence, 32768)

    def stock(value):
        calls.append(("stock", value.shape[:2]))
        return "stock"

    def wide(value):
        calls.append(("wide", value.shape[:2]))
        return "wide"

    route = object.__new__(D._DirectGatherOLoraWideM4)
    route.stock = stock
    route.m4 = wide

    assert route(_Input(1, 1)) == "stock"
    assert route(_Input(1, 3)) == "stock"
    assert route(_Input(2, 2)) == "wide"
    assert route(_Input(1, 5)) == "stock"
    assert calls == [
        ("stock", (1, 1)),
        ("stock", (1, 3)),
        ("wide", (2, 2)),
        ("stock", (1, 5)),
    ]


def test_model_installer_prebinds_43_body_gathers_and_explicit_mtp_stock(monkeypatch):
    _patch_canonical_route_types(monkeypatch)
    model = _canonical_route_model()
    original_body_wo_b = [box.attn.wo_b for box in model.layers]
    original_mtp_wo_b = model.mtp_blocks[0].attn.wo_b
    report = D.install_deepseek_v4_o_lora_routes(
        model, mode="gather_qmm", canonical_mixed_route=True
    )
    body = [box.attn for box in model.layers]
    mtp = model.mtp_blocks[0].attn
    assert report["trunk_module_count"] == 43
    assert report["mtp_module_count"] == 1
    assert report["module_count"] == 44
    assert report["body_direct"] == 43
    assert report["mtp_stock"] == 1
    assert all(
        isinstance(attention._o_lora_impl, _FakeGatherWideM4BodyRoute)
        for attention in body
    )
    assert isinstance(mtp._o_lora_impl, _FakeDenseMTPRoute)
    assert all(
        attention._o_lora_impl.wo_b is original
        for attention, original in zip(body, original_body_wo_b)
    )
    assert mtp._o_lora_impl.wo_b is original_mtp_wo_b
    assert report["callable_census"] == {
        "body_route_objects": 43,
        "body_route_kind": "gather_qmm_m4_wide_direct",
        "body_callable_class": "_DirectGatherOLoraWideM4",
        "mtp_route_objects": 1,
        "mtp_route_kind": "dense_bf16_stock_direct",
        "mtp_callable_class": "_DirectDenseMTPOLora",
        "total_route_objects": 44,
        "unique_route_objects": 44,
        "mtp_distinct_type": True,
    }
    assert report["storage_contract"] == D._CANONICAL_O_LORA_STORAGE_CONTRACT
    assert report["storage_contract"]["body"]["wo_b"] == {
        "class": "QuantizedLinear",
        "logical_weight_shape": [4096, 8192],
        "packed_weight": {"shape": [4096, 1024], "dtype": "uint32"},
        "scales": {"shape": [4096, 128], "dtype": "bfloat16"},
        "biases": {"shape": [4096, 128], "dtype": "bfloat16"},
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "additive_bias": None,
    }
    assert report["storage_contract"]["mtp"]["wo_a"]["weight"] == {
        "shape": [8192, 4096],
        "dtype": "bfloat16",
    }
    assert report["storage_contract"]["mtp"]["wo_b"]["weight"] == {
        "shape": [4096, 8192],
        "dtype": "bfloat16",
    }


def _mutate_body_wo_b_logical_output(model):
    wo_b = model.layers[0].attn.wo_b
    wo_b.weight.shape = (4095, 1024)
    wo_b.scales.shape = (4095, 128)
    wo_b.biases.shape = (4095, 128)


def _mutate_body_wo_b_logical_input(model):
    wo_b = model.layers[0].attn.wo_b
    wo_b.weight.shape = (4096, 1016)
    wo_b.scales.shape = (4096, 127)
    wo_b.biases.shape = (4096, 127)


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda model: model.layers.pop(), "expected 43 body"),
        (lambda model: setattr(model.layers[0].attn, "wo_a", _CanonicalMTPWOA()), "body 0"),
        (lambda model: setattr(model.layers[0].attn.wo_a.weight, "dtype", mx.bfloat16), "weight dtype"),
        (lambda model: setattr(model.layers[0].attn.wo_a.weight, "shape", (8192, 513)), "weight shape"),
        (lambda model: setattr(model.layers[0].attn.wo_a.scales, "dtype", mx.float32), "scales dtype"),
        (lambda model: setattr(model.layers[0].attn.wo_a.biases, "shape", (8192, 63)), "biases shape"),
        (lambda model: setattr(model.layers[0].attn.wo_a.biases, "dtype", mx.float32), "biases dtype"),
        (lambda model: setattr(model.layers[0].attn.wo_a, "bits", 8), "bits"),
        (lambda model: setattr(model.layers[0].attn.wo_a, "group_size", 32), "group_size"),
        (lambda model: setattr(model.layers[0].attn.wo_a, "mode", "mxfp4"), "mode"),
        (lambda model: setattr(model.layers[0].attn, "n_groups", 4), "n_groups"),
        (lambda model: setattr(model.layers[0].attn, "o_lora_rank", 512), "o_lora_rank"),
        (lambda model: setattr(model.layers[0].attn, "n_heads", 32), "n_heads"),
        (lambda model: setattr(model.layers[0].attn, "head_dim", 256), "head_dim"),
        (lambda model: setattr(model.layers[0].attn, "dim", 2048), "dim"),
        (lambda model: setattr(model.layers[0].attn, "wo_b", _CanonicalMTPWOB()), "body 0 wo_b"),
        (lambda model: setattr(model.layers[0].attn.wo_b.weight, "shape", (4096, 1023)), "wo_b.*weight.*shape"),
        (lambda model: setattr(model.layers[0].attn.wo_b.weight, "dtype", mx.bfloat16), "wo_b.*weight.*dtype"),
        (lambda model: setattr(model.layers[0].attn.wo_b.scales, "shape", (4096, 127)), "wo_b.*scales.*shape"),
        (lambda model: setattr(model.layers[0].attn.wo_b.scales, "dtype", mx.float32), "wo_b.*scales.*dtype"),
        (lambda model: setattr(model.layers[0].attn.wo_b.biases, "shape", (4096, 127)), "wo_b.*biases.*shape"),
        (lambda model: setattr(model.layers[0].attn.wo_b.biases, "dtype", mx.float32), "wo_b.*biases.*dtype"),
        (lambda model: setattr(model.layers[0].attn.wo_b, "bits", 8), "wo_b.*bits"),
        (lambda model: setattr(model.layers[0].attn.wo_b, "group_size", 32), "wo_b.*group_size"),
        (lambda model: setattr(model.layers[0].attn.wo_b, "mode", "mxfp4"), "wo_b.*mode"),
        (lambda model: setattr(model.layers[0].attn.wo_b, "bias", _O_LoraArrayMeta((4096,), mx.bfloat16)), "wo_b.*additive bias"),
        (_mutate_body_wo_b_logical_output, "wo_b.*logical output"),
        (_mutate_body_wo_b_logical_input, "wo_b.*logical input"),
        (lambda model: setattr(model.mtp_blocks[0].attn, "wo_a", _CanonicalBodyWOA()), "MTP"),
        (lambda model: setattr(model.mtp_blocks[0].attn.wo_a.weight, "dtype", mx.float32), "MTP wo_a weight dtype"),
        (lambda model: setattr(model.mtp_blocks[0].attn.wo_a.weight, "shape", (8192, 4095)), "MTP wo_a weight shape"),
        (lambda model: setattr(model.mtp_blocks[0].attn.wo_a, "bias", _O_LoraArrayMeta((8192,), mx.bfloat16)), "MTP wo_a additive bias"),
        (lambda model: setattr(model.mtp_blocks[0].attn, "wo_b", _CanonicalBodyWOB()), "MTP wo_b"),
        (lambda model: setattr(model.mtp_blocks[0].attn.wo_b.weight, "dtype", mx.float32), "MTP wo_b weight dtype"),
        (lambda model: setattr(model.mtp_blocks[0].attn.wo_b.weight, "shape", (4096, 8191)), "MTP wo_b weight shape"),
        (lambda model: setattr(model.mtp_blocks[0].attn.wo_b, "bias", _O_LoraArrayMeta((4096,), mx.bfloat16)), "MTP wo_b additive bias"),
        (lambda model: model.mtp_blocks.pop(), "expected exactly one MTP"),
    ),
)
def test_model_installer_rejects_noncanonical_mixed_storage(monkeypatch, mutate, match):
    _patch_canonical_route_types(monkeypatch)
    model = _canonical_route_model()
    mutate(model)
    with pytest.raises(ValueError, match=match):
        D.install_deepseek_v4_o_lora_routes(
            model, mode="gather_qmm", canonical_mixed_route=True
        )
    assert all(box.attn._o_lora_impl is None for box in model.layers + model.mtp_blocks)


def test_model_installer_rejects_mtp_quant_metadata(monkeypatch):
    _patch_canonical_route_types(monkeypatch)
    model = _canonical_route_model()
    model.mtp_blocks[0].attn.wo_a.scales = _O_LoraArrayMeta((8192, 64), mx.bfloat16)
    with pytest.raises(ValueError, match="must not expose quantized metadata"):
        D.install_deepseek_v4_o_lora_routes(
            model, mode="gather_qmm", canonical_mixed_route=True
        )
    assert all(box.attn._o_lora_impl is None for box in model.layers + model.mtp_blocks)


def test_geometry_failure_binds_no_partial_route(monkeypatch):
    _patch_canonical_route_types(monkeypatch)
    model = _canonical_route_model()
    model.layers[-1].attn.wo_b.weight.shape = (4096, 1023)
    with pytest.raises(ValueError, match="wo_b.*weight.*shape"):
        D.install_deepseek_v4_o_lora_routes(
            model, mode="gather_qmm", canonical_mixed_route=True
        )
    assert all(box.attn._o_lora_impl is None for box in model.layers + model.mtp_blocks)


def test_model_installer_routes_cached_after_validating_mixed_topology(monkeypatch):
    _patch_canonical_route_types(monkeypatch)
    model = _canonical_route_model()
    report = D.install_deepseek_v4_o_lora_routes(
        model, mode="cached", canonical_mixed_route=True
    )
    assert report["body_direct"] == 0
    assert report["mtp_stock"] == 1
    assert all(
        isinstance(box.attn._o_lora_impl, _FakeCachedBodyRoute)
        for box in model.layers
    )
    assert isinstance(model.mtp_blocks[0].attn._o_lora_impl, _FakeDenseMTPRoute)
    assert report["callable_census"]["body_route_kind"] == "cached_direct"


def test_prebound_body_and_mtp_routes_never_recheck_quant_metadata(monkeypatch):
    _, quantized = _quantized_model()
    body = quantized.layers[0].attn
    quant = body._wo_a_quant()
    body_cached = D._DirectCachedOLora(body, quant)
    body_gather = D._DirectGatherOLora(body, quant)

    _, dense = _seeded_model(num_nextn_predict_layers=1)
    mtp = dense.mtp_blocks[0].attn
    mtp_stock = D._DirectDenseMTPOLora(mtp, mtp.wo_a.weight)
    monkeypatch.setattr(
        body,
        "_wo_a_quant",
        lambda: (_ for _ in ()).throw(AssertionError("body hot metadata check")),
    )
    monkeypatch.setattr(
        mtp,
        "_wo_a_quant",
        lambda: (_ for _ in ()).throw(AssertionError("MTP hot metadata check")),
    )

    body_input = mx.random.normal((1, 3, N_HEADS * HEAD_DIM))
    mtp_input = mx.random.normal((1, 3, N_HEADS * HEAD_DIM))
    body._o_lora_impl = body_cached
    cached_output = body._o_lora(body_input)
    body._o_lora_impl = body_gather
    gather_output = body._o_lora(body_input)
    mtp._o_lora_impl = mtp_stock
    mtp_output = mtp._o_lora(mtp_input)
    outputs = [cached_output, gather_output, mtp_output]
    mx.eval(*outputs)
    assert all(tuple(output.shape) == (1, 3, DIM) for output in outputs)


def _mtp_inputs(args, seq=9, seed=7):
    """``h`` is the trunk's pre-head hyper-connection state; ``ids`` the fused tokens."""
    mx.random.seed(seed)
    h = mx.random.normal((1, seq, args.hc_mult, args.hidden_size)) * 0.5
    return h, mx.random.randint(0, VOCAB, (1, seq))


def test_draft_block_carries_the_o_lora_machinery(monkeypatch):
    """It reads the same env knob the trunk does, and holds its *own* cache object."""
    monkeypatch.setenv("MTPLX_DSV4_O_LORA", "gather_qmm")
    _, model = _quantized_mtp_model()
    draft = model.mtp_blocks[0].attn
    assert draft.o_lora_mode == "gather_qmm"
    assert isinstance(draft._wo_a_cache, D._DerivedCache)
    caches = [a._wo_a_cache for a in _attentions(model)]
    assert len({id(c) for c in caches}) == len(caches), (
        "attention instances share a _DerivedCache — trunk and draft would serve "
        "each other's wo_a")


def test_cached_dequant_is_bit_identical_through_mtp_forward():
    """The bit-identity gate, run through the draft entry point rather than the trunk."""
    args, model = _quantized_mtp_model()
    h, ids = _mtp_inputs(args)

    _set_mode_everywhere(model, "dequant")
    ref = model.mtp_forward(h, ids)
    mx.eval(ref)
    assert model.mtp_blocks[0].attn._wo_a_cache.value is None, (
        "the dequant arm populated the draft cache, so it is not a control")

    _set_mode_everywhere(model, "cached")
    first = model.mtp_forward(h, ids)      # populates
    second = model.mtp_forward(h, ids)     # serves from cache
    mx.eval(first, second)

    assert model.mtp_blocks[0].attn._wo_a_cache.value is not None, (
        "the draft block's cache never populated — the gate would be vacuous")
    assert mx.array_equal(ref, first), "draft cached first call is not bit-identical"
    assert mx.array_equal(ref, second), "draft cache-hit call is not bit-identical"
    assert len(set(np.array(mx.argmax(ref[0], axis=-1)).tolist())) > 1


def test_every_attention_caches_its_own_wo_a_not_a_neighbours():
    """Per-instance keying, measured: five attentions, five distinct dense weights.

    Identity keying gives trunk and draft no shared keyspace to collide in, but
    "no shared keyspace" is an argument; this is the measurement.
    """
    args, model = _quantized_mtp_model()
    _set_mode_everywhere(model, "cached")
    model(_tokens(20))                     # populates the trunk
    model.mtp_forward(*_mtp_inputs(args))  # populates the draft block

    held = []
    for i, attn in enumerate(_attentions(model)):
        value = attn._wo_a_cache.value
        assert value is not None, f"attention {i} cached nothing"
        w, sc, bi, gs, bits, mode = attn._wo_a_quant()
        own = mx.dequantize(
            w, sc, bi, group_size=gs, bits=bits, mode=mode
        ).reshape(value.shape)
        mx.eval(value, own)
        assert mx.array_equal(value, own), f"attention {i} cached another module's wo_a"
        held.append(np.array(value.astype(mx.float32)))

    assert len(held) == len(RATIOS) + 1
    for i in range(len(held)):
        for j in range(i + 1, len(held)):
            assert not np.array_equal(held[i], held[j]), (
                f"attentions {i} and {j} hold identical weights — the check above "
                "cannot tell a cross-served cache from a correct one")


def test_draft_block_cache_never_reaches_the_weight_tree():
    """``mtp.0.*`` is a checkpoint path; a stray derived tensor there breaks strict load."""
    args, model = _quantized_mtp_model()
    before = {k for k, _ in tree_flatten(model.parameters())}
    _set_mode_everywhere(model, "cached")
    model.mtp_forward(*_mtp_inputs(args))
    after = {k for k, _ in tree_flatten(model.parameters())}
    assert before == after
    draft = model.mtp_blocks[0].attn
    assert draft._wo_a_cache.value is not None
    assert not any(k.startswith("_wo_a") for k in dict(draft))
    assert not any(k.startswith("mtp.0.attn._wo_a") for k in before)
    model.load_weights(tree_flatten(model.parameters()), strict=True)
