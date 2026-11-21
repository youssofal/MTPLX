"""Every Inkling pack takes every fused route its stages have: an optimisation never lands on one pack.

The packs differ only in how their matrices are stored: Optimized Quality keeps 16-bit attention, the
head and the embedding; Optimized Speed stores the attention projections and the head as 8-bit affine
codes; Bare Speed stores them and the embedding as 4-bit codes; the FP16 siblings are the same three
with a float16 stream. A stage kernel built for one storage form, with the others left on the
separate MLX launches, is how the recommended pack once ran 230 more launches a step than the native
one (21 November: the attention output fused for 16-bit weights only, 224 launches; the stored
embedding, 3; the stored head's input and scale casts, 3).

1. Routes (no GPU work, no weights): for every shipped pack, stand-ins carry each matrix's real shape
   and the storage its recipe gives it (``forge_inkling_quant.matrix_layout``, the definition the
   packs were built from), and the step's own per-visit rules (``fast.step.stage_fallbacks``, the same
   reasons every stage function asks) must report no visit of any stage leaving its fused route, at
   every width the step serves. The FP16 siblings' known gaps are listed exactly (a ratchet: a new gap
   fails, and so does a closed gap left on the list).
2. Launches (Metal and the native engine, synthetic tensors with the real shapes): the stages whose
   weights differ by pack, stage C, the head and the embedding, take no more launches, and no more of
   MLX's own kernels (stock launches: any kernel that is not one of ours), on the stored forms than on
   the 16-bit form, on a bf16 and on a float16 stream, counted exactly by the engine's recorder. Stage C
   takes no stock launch on any form.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.commands.forge_inkling_quant import EMBED, HEAD, matrix_layout  # noqa: E402
from mtplx.models.inkling.layout import layer_visits  # noqa: E402
from mtplx.models.inkling.precision import AffineWeight, LogicalShape  # noqa: E402

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="the fused routes need a Metal GPU")

# Inkling 2 Mini's text geometry (config.json text_config; tensor shapes from the packs' headers).
CONFIG = {
    "num_hidden_layers": 24,
    "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
    "n_layers_per_attn_res_block": 6,
    "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False},
}
HIDDEN, VOCAB, EXPERTS, MIDDLE = 2304, 201024, 128, 1152
PROJECTIONS = {"wq_du": (3072, HIDDEN), "wk_dv": (1024, HIDDEN), "wv_dv": (1024, HIDDEN), "wr_du": (384, HIDDEN),
               "gate_attn": (3072, HIDDEN), "wo_ud": (HIDDEN, 3072)}
# Every shipped Inkling pack: its recipe and its stream's type.
SHIPPED = {
    "inkling2-mini-bare-speed": ("bare-speed", "bfloat16"),
    "inkling2-mini-optimized-speed": ("speed", "bfloat16"),
    "inkling2-mini-optimized-quality": ("quality", "bfloat16"),
    "inkling2-mini-bare-speed-fp16": ("bare-speed", "float16"),
    "inkling2-mini-optimized-speed-fp16": ("speed", "float16"),
    "inkling2-mini-optimized-quality-fp16": ("quality", "float16"),
}
VISITS = len(layer_visits(CONFIG))
# The FP16 siblings' open gaps (lanes/X1.md: lane L1's stage A and MLP read round to bf16, so a float16
# stream takes the reference for those stages). Offered only behind INKLING_LEGACY_OFFER (off). Stage C,
# the head, the embedding and the expert block serve float16 streams.
_FP16_GAPS = {"fallback.a.stream_dtype": VISITS, "fallback.d.stream_dtype": VISITS}
KNOWN_FP16_GAPS = {"bare-speed": _FP16_GAPS, "speed": _FP16_GAPS, "quality": _FP16_GAPS}
WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32)


class Meta:
    """A tensor's shape and type without data: what the route rules read."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype


def stored(name, rows, cols, recipe, dtype):
    layout = matrix_layout(name, recipe)
    if layout is None:
        return Meta((rows, cols), dtype)
    bits, group = layout
    return AffineWeight(Meta((rows, cols * bits // 32), mx.uint32), Meta((rows, cols // group), dtype),
                        Meta((rows, cols // group), dtype), bits, group)


def experts(index, recipe):
    name = f"model.llm.layers.{index}.mlp.experts.w13_weight"
    mode = "nvfp4" if index < 20 or matrix_layout(name, recipe) == "nvfp4" else "mxfp8"
    per = 8 if mode == "nvfp4" else 4
    return SimpleNamespace(
        gate_up=SimpleNamespace(mode=mode, affine=None, codes=Meta((EXPERTS, 2 * MIDDLE, HIDDEN // per), mx.uint32)),
        down=SimpleNamespace(mode=mode, affine=None, codes=Meta((EXPERTS, HIDDEN, MIDDLE // per), mx.uint32)),
        router=Meta((EXPERTS, HIDDEN), mx.float32), count=8)


def stand_in(recipe, dtype):
    local = set(CONFIG["local_layer_ids"])
    start, end = CONFIG["loop_config"]["core_start"], CONFIG["loop_config"]["core_end"]
    layers = []
    for index in range(CONFIG["num_hidden_layers"]):
        m = {p: stored(f"model.llm.layers.{index}.attn.{p}.weight", *shape, recipe, dtype)
             for p, shape in PROJECTIONS.items()}
        dynamic = index not in local
        convs = [SimpleNamespace(dynamic=Meta((32, HIDDEN), dtype) if dynamic and i < 2 else None, size=4,
                                 weight=Meta((4, width), mx.float32))
                 for i, width in enumerate((1024, 1024, HIDDEN, HIDDEN))]
        controllers = ([SimpleNamespace(down=Meta((64, 2 * HIDDEN), dtype)) for _ in range(4)]
                       if start <= index < end else None)
        layers.append(SimpleNamespace(
            sliding=index in local, boundary=index % CONFIG["n_layers_per_attn_res_block"] == 0,
            q=m["wq_du"], k=m["wk_dv"], v=m["wv_dv"], r=m["wr_du"], gate=m["gate_attn"], out=m["wo_ud"],
            sink=Meta((24, HIDDEN), dtype), convs=convs, controllers=controllers, experts=experts(index, recipe)))
    table = stored(EMBED, VOCAB, HIDDEN, recipe, dtype)
    gain = Meta((HIDDEN,), dtype)
    embed_tokens = (SimpleNamespace(weight=table, packed=None, gain=gain, eps=1e-6)
                    if not isinstance(table, AffineWeight) else
                    SimpleNamespace(weight=LogicalShape((VOCAB, HIDDEN), dtype), packed=table, gain=gain, eps=1e-6))
    return SimpleNamespace(layers=layers, visit_order=layer_visits(CONFIG), start=start, end=end,
                           embed_tokens=embed_tokens, unembed=stored(HEAD, VOCAB, HIDDEN, recipe, dtype))


def test_every_inkling_pack_in_the_catalog_is_covered():
    from mtplx.model_catalog import OFFICIAL_CATALOG

    inkling = {m.id for m in OFFICIAL_CATALOG if m.id.startswith("inkling")}
    assert inkling == set(SHIPPED), "a new Inkling pack needs its recipe and stream here, and its routes proven"


@needs_metal
@pytest.mark.parametrize("pack", sorted(SHIPPED))
def test_every_stage_of_every_pack_takes_its_fused_route(pack):
    from mtplx.models.inkling.fast import step

    recipe, stream = SHIPPED[pack]
    dtype = getattr(mx, stream)
    model = stand_in(recipe, dtype)
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)  # the expert block's rule asks for the GPU; nothing is evaluated
    try:
        for rows in WIDTHS:
            found = step.stage_fallbacks(model, "a,bc,d,e", rows, dtype)
            # The head's rule names a one-row and a panel key; neither may appear.
            gaps = {k: v for k, v in found.items() if not k.startswith("fallback.head.")}
            heads = {k: v for k, v in found.items() if k.startswith("fallback.head.")}
            want = KNOWN_FP16_GAPS[recipe] if stream == "float16" else {}
            if rows > 32:
                continue
            assert gaps == want, (pack, rows, found)
            assert not heads, (pack, rows, heads)
    finally:
        mx.set_default_device(previous)


# --------------------------------------------------------------------------- launches
def _engine():
    import os

    from mtplx.engine import runtime as engine_runtime

    # The suite drops MTPLX_* variables before collection, so the built module's path comes in under
    # INKLING_TEST_ENGINE (as the commit wiring tests take it); without it these launch counts skip.
    path = os.environ.get("INKLING_TEST_ENGINE", "").strip()
    if path and not engine_runtime.available():
        os.environ[engine_runtime.MODULE_ENV] = path
        engine_runtime._REASON = None
        engine_runtime._NATIVE = None
    if not engine_runtime.available():
        pytest.skip(f"the native engine is not loaded: {engine_runtime.unavailable()}")
    return engine_runtime.engine()


def _launches(eng, fn):
    """``(launches, stock launches)`` of one call after its first use (the routes' proofs run there,
    outside the recording). A stock launch is one of MLX's own kernels: any kernel not built from our
    source (``mx.fast.metal_kernel`` names ours ``custom_kernel_mtplx_...``)."""
    out = fn()
    mx.eval(out)
    eng.record_begin(label="parity")
    try:
        out = fn()
        mx.eval(out)
    finally:
        trace = eng.record_end()
    stock = sum(n for name, n in trace.kernel_counts().items() if not name.startswith("custom_kernel_mtplx_"))
    return trace.launches, stock


STREAMS = (mx.bfloat16, mx.float16)


def _affine(rows, cols, bits, seed, dtype=mx.bfloat16):
    """A stored matrix as the packs keep it: its scales and biases in the stream's 16-bit type."""
    w = (mx.random.normal((rows, cols), key=mx.random.key(seed)) * 0.02).astype(dtype)
    codes, scales, biases = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(codes, scales, biases)
    return AffineWeight(codes, scales, biases, bits, 64)


@needs_metal
@pytest.mark.parametrize("dtype", STREAMS, ids=str)
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_stage_c_takes_one_launch_on_every_storage_form(rows, dtype, monkeypatch):
    from mtplx.models.inkling.fast import attend as fast_attend
    from mtplx.models.inkling.fast import step

    eng = _engine()
    hg = (mx.random.normal((1, rows, 3072), key=mx.random.key(1)) * 0.5).astype(dtype)
    x = mx.random.normal((1, rows, HIDDEN), key=mx.random.key(2)).astype(dtype)
    tail = mx.random.normal((1, 3, HIDDEN), key=mx.random.key(3))
    conv_weight = mx.contiguous(mx.random.normal((HIDDEN, 4), key=mx.random.key(4)) * 0.3).T
    mx.eval(hg, x, tail, conv_weight)
    state = SimpleNamespace(window_view=lambda visit: (None, None, 0, 0), tail=lambda visit, slot: tail)
    panel = step.Panel(rows, mx.array(10, mx.uint32), True if rows > 1 else None)
    a = SimpleNamespace(q=None, rel=None, sink=None, gate_logits=None, k_rows=None, v_rows=None)
    monkeypatch.setattr(step, "attention_reason", lambda dtype=None: None)
    monkeypatch.setattr(fast_attend, "attend", lambda *args, **kw: hg)
    dense = (mx.random.normal((HIDDEN, 3072), key=mx.random.key(5)) * 0.02).astype(dtype)
    mx.eval(dense)
    from mtplx.models.inkling.fast.routes import capture_counts, stage_paths

    counts = {}
    for form, out in (("16-bit", dense), ("4-bit", _affine(HIDDEN, 3072, 4, 6, dtype)),
                      ("8-bit", _affine(HIDDEN, 3072, 8, 7, dtype))):
        convs = [SimpleNamespace(dynamic=None, size=4, weight=conv_weight) for _ in range(4)]
        layer = SimpleNamespace(sliding=True, boundary=False, out=out, convs=convs, rel_basis=None)
        with capture_counts() as served:
            counts[form] = _launches(eng, lambda layer=layer: step.fused_attention(layer, a, x, state, 3, panel))
        # The route is read from its own counter (what a request's log shows), not inferred.
        route = "fused.bc_output.panel" if form == "16-bit" else "fused.bc_output.stored"
        assert served.get(route) == 2 and not stage_paths(served).keys() - {route, "fused.bc_attention"}, served
    assert all(n <= counts["16-bit"][0] for n, _ in counts.values()), counts
    assert all(stock == 0 for _, stock in counts.values()), counts


def _head_model(unembed, dtype):
    from mtplx.models.inkling.model import Model

    model = SimpleNamespace(unembed=unembed, embed_tokens=SimpleNamespace(weight=Meta((32768, HIDDEN), dtype)),
                            config={"final_logit_softcapping": 30.0, "unpadded_vocab_size": 32000})
    model.head = lambda h: Model.head(model, h)
    return model


@needs_metal
@pytest.mark.parametrize("dtype", STREAMS, ids=str)
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_a_stored_head_takes_the_launches_of_a_16_bit_one(rows, dtype):
    """A stored head takes the 16-bit head's launches at every width: the 16-bit rows widened inside
    the product launch, no cast and no widened scale copies."""
    from mtplx.models.inkling.fast import step

    eng = _engine()
    hidden = mx.random.normal((1, rows, HIDDEN), key=mx.random.key(8)).astype(dtype)
    dense = (mx.random.normal((32768, HIDDEN), key=mx.random.key(9)) * 0.02).astype(dtype)
    mx.eval(hidden, dense)
    native = _head_model(dense, dtype)
    launches, stock = _launches(eng, lambda: step.row_stable_head(native, hidden))
    for bits in (4, 8):
        model = _head_model(_affine(32768, HIDDEN, bits, 9 + bits, dtype), dtype)
        got = _launches(eng, lambda model=model: step.row_stable_head(model, hidden))
        assert got[0] <= launches and got[1] <= stock, (bits, got, (launches, stock))


@needs_metal
@pytest.mark.parametrize("dtype", STREAMS, ids=str)
@pytest.mark.parametrize("rows", [1, 4])
def test_a_stored_embedding_takes_the_launches_of_a_16_bit_one(rows, dtype):
    from mtplx.models.inkling.fast import step
    from mtplx.models.inkling.model import Embedding

    eng = _engine()
    gain = mx.ones((HIDDEN,), dtype)
    dense = (mx.random.normal((32768, HIDDEN), key=mx.random.key(11)) * 0.02).astype(dtype)
    mx.eval(dense, gain)
    ids = mx.array([[5, 32767, 77, 1024][:rows]], mx.int32)
    native = SimpleNamespace(embed_tokens=Embedding(dense, gain, 1e-6))
    packed = SimpleNamespace(embed_tokens=Embedding(_affine(32768, HIDDEN, 4, 12, dtype), gain, 1e-6))
    got, want = _launches(eng, lambda: step.embed(packed, ids)), _launches(eng, lambda: step.embed(native, ids))
    # Gather, decode and norm in one launch of ours on either table: no stock launch left.
    assert got == (1, 0) and want == (1, 0), (got, want)
