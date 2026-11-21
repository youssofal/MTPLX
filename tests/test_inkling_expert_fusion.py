"""CPU tests for the fused Inkling expert block's admission, proof and counters.

The kernels themselves are compared with the stock body on the GPU at first use;
these tests pin the layer that decides when they run and what every visit reports.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.kernels.first_use import ProofCache
from mtplx.models.inkling import activations, expert_fusion as ef


def block(mode="nvfp4", rows_dtype=mx.bfloat16, affine=None, router_dtype=mx.float32):
    fp4 = mode == "nvfp4"
    pack = lambda o, k: mx.zeros((128, o, k // (8 if fp4 else 4)), mx.uint32)
    gate_up = SimpleNamespace(mode=mode, affine=affine, codes=pack(2304, 2304))
    down = SimpleNamespace(mode=mode, affine=None, codes=pack(2304, 1152))
    return SimpleNamespace(gate_up=gate_up, down=down, router=mx.zeros((128, 2304), router_dtype),
                           count=8, scale=8.0, gate_cap=4.0, up_cap=25.0)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MTPLX_INKLING_EXPERT_FUSION", raising=False)
    monkeypatch.delenv("MTPLX_INKLING_EXPERT_ROUTE", raising=False)
    monkeypatch.setattr(ef, "_fallback_state", None)
    device = mx.default_device()
    mx.set_default_device(mx.cpu)  # these tests never touch the GPU
    yield
    mx.set_default_device(device)


def test_unserved_names_every_stock_reason(monkeypatch):
    flat = mx.zeros((9, 2304), mx.bfloat16)
    assert ef.unserved(block(), flat, gpu=True) is None
    assert ef.unserved(block("mxfp8"), flat, gpu=True) is None
    assert ef.unserved(block(), flat.astype(mx.float16), gpu=True) is None
    assert ef.unserved(block(), flat, gpu=False) == "cpu"
    assert ef.unserved(block(), mx.zeros((33, 2304), mx.bfloat16), gpu=True) == "grouped_rows"
    assert ef.unserved(block(affine=object()), flat, gpu=True) == "affine_experts"
    assert ef.unserved(block(), flat.astype(mx.float32), gpu=True) == "dtype"
    assert ef.unserved(block(router_dtype=mx.bfloat16), flat, gpu=True) == "geometry"
    mixed = block()
    mixed.down.mode = "mxfp8"
    assert ef.unserved(mixed, flat, gpu=True) == "format"
    monkeypatch.setenv("MTPLX_INKLING_EXPERT_FUSION", "off")
    assert ef.unserved(block(), flat, gpu=True) == "disabled"


def test_switches_reject_unknown_values(monkeypatch):
    monkeypatch.setenv("MTPLX_INKLING_EXPERT_FUSION", "sometimes")
    with pytest.raises(ValueError):
        ef.fusion_mode()
    monkeypatch.setenv("MTPLX_INKLING_EXPERT_ROUTE", "approximate")
    with pytest.raises(ValueError):
        ef.route_variant()


def harness(monkeypatch, fused):
    """A served block whose fused path returns `fused(flat)` and whose stock body is fixed."""
    calls = {"stock": 0, "fused": 0}

    def stock(hidden):
        calls["stock"] += 1
        return (hidden.astype(mx.float32) * 2).astype(hidden.dtype)

    def routed(blk, flat, linear, trace=None, variant=None):
        calls["fused"] += 1
        return fused(flat)

    real = ef.unserved
    monkeypatch.setattr(ef, "unserved", lambda blk, flat, gpu=None: real(blk, flat, gpu=True))
    monkeypatch.setattr(ef, "routed", routed)
    return calls, stock


def test_first_use_admits_once_and_then_serves_without_rechecking(monkeypatch, tmp_path):
    calls, stock = harness(monkeypatch, lambda flat: (flat.astype(mx.float32) * 2).astype(flat.dtype))
    blk, model = block(), SimpleNamespace(layers=[SimpleNamespace(experts=None)])
    model.layers[0].experts = blk
    state = ef.bind(model, ProofCache({"test": 1}, tmp_path))
    hidden = mx.ones((1, 9, 2304), mx.bfloat16)
    first = ef.experts_forward(blk, hidden, None, stock)
    second = ef.experts_forward(blk, hidden, None, stock)
    assert mx.array_equal(first, stock(hidden)) and mx.array_equal(second, first)
    assert calls == {"stock": 2, "fused": 2}  # the proof ran stock once; both visits served fused
    assert state.counts == {"nvfp4.r9.fused.fused": 2}
    assert state.proofs.checks == 1
    reloaded = ef.FusionState(ProofCache({"test": 1}, tmp_path))
    plan, reason = reloaded.proofs.decide(ef.NAME, json.loads(json.dumps(_shape_of(state))), lambda: (None, "unused"))
    assert plan is not None and reloaded.proofs.hits == 1 and reloaded.proofs.checks == 0


def _shape_of(state):
    # The record on disk is keyed by the shape the first visit proved.
    files = [p for p in Path(state.proofs.directory).glob(f"{ef.NAME}-*.json")]
    assert len(files) == 1
    return json.loads(files[0].read_text())["shape"]


def test_a_mismatch_keeps_the_stock_body_and_survives_a_reload(monkeypatch, tmp_path):
    calls, stock = harness(monkeypatch, lambda flat: mx.zeros_like(flat))
    blk = block("mxfp8")
    state = ef.bind(SimpleNamespace(layers=[SimpleNamespace(experts=blk)]), ProofCache({"t": 2}, tmp_path))
    hidden = mx.ones((4, 2304), mx.bfloat16)
    for _ in range(3):
        out = ef.experts_forward(blk, hidden, None, stock)
        assert mx.array_equal(out, stock(hidden))
    assert state.counts == {"mxfp8.r4.stock.first_use_mismatch": 3}
    assert calls["fused"] == 2  # each of the two routes was tried once
    again = ef.FusionState(ProofCache({"t": 2}, tmp_path))
    blk.fusion = again
    ef.experts_forward(blk, hidden, None, stock)
    assert again.counts == {"mxfp8.r4.stock.first_use_mismatch": 1} and again.proofs.hits == 2
    assert calls["fused"] == 2  # a refuted kernel never runs again on this identity


def route_harness(monkeypatch):
    """A device whose kernel sigmoid is off by a bit: only the `scores` route gives the stock bits."""
    calls, stock = harness(monkeypatch, None)
    seen = []

    def routed(blk, flat, linear, trace=None, variant=None):
        seen.append(variant)
        return (flat.astype(mx.float32) * (2 if variant == "scores" else 3)).astype(flat.dtype)

    monkeypatch.setattr(ef, "routed", routed)
    return seen, stock


def test_a_refuted_kernel_sigmoid_leaves_the_rest_of_the_block_fused(monkeypatch):
    seen, stock = route_harness(monkeypatch)
    blk, hidden = block(), mx.ones((9, 2304), mx.bfloat16)
    state = ef.bind(SimpleNamespace(layers=[SimpleNamespace(experts=blk)]), ProofCache(None, None))

    def generate(rt):
        for _ in range(3):
            assert mx.array_equal(ef.experts_forward(blk, hidden, None, stock), stock(hidden))
        return SimpleNamespace(stats=SimpleNamespace())

    receipt = ef.observe_expert_request(generate, object()).stats.inkling_experts
    assert receipt["selections"] == {"nvfp4.r9.fused.scores": 3}
    assert receipt["first_use"] == {"declined": 1, "passed": 1}
    assert seen == ["fused", "scores", "scores", "scores"]  # the refuted route ran once, in its proof
    decisions = state.health()["decisions"]
    assert decisions["nvfp4/bfloat16/scores/verify"] == "fused"
    assert str(decisions["nvfp4/bfloat16/fused/verify"]).startswith("mismatch:")


def test_the_switch_pins_one_route(monkeypatch):
    seen, stock = route_harness(monkeypatch)
    monkeypatch.setenv("MTPLX_INKLING_EXPERT_ROUTE", "fused")
    blk, hidden = block(), mx.ones((9, 2304), mx.bfloat16)
    state = ef.bind(SimpleNamespace(layers=[SimpleNamespace(experts=blk)]), ProofCache(None, None))
    assert mx.array_equal(ef.experts_forward(blk, hidden, None, stock), stock(hidden))
    assert state.counts == {"nvfp4.r9.stock.first_use_mismatch": 1} and seen == ["fused"]


def test_a_kernel_error_declines_for_this_load_only(monkeypatch, tmp_path):
    def broken(flat):
        raise RuntimeError("threadgroup larger than the pipeline allows")

    calls, stock = harness(monkeypatch, broken)
    blk = block()
    state = ef.bind(SimpleNamespace(layers=[SimpleNamespace(experts=blk)]), ProofCache({"t": 3}, tmp_path))
    hidden = mx.ones((1, 2304), mx.bfloat16)
    assert mx.array_equal(ef.experts_forward(blk, hidden, None, stock), stock(hidden))
    assert state.counts == {"nvfp4.r1.stock.first_use_error": 1}
    assert not list(Path(state.proofs.directory).glob(f"{ef.NAME}-*.json"))


def test_request_receipt_counts_its_own_visits_and_first_use(monkeypatch):
    calls, stock = harness(monkeypatch, lambda flat: (flat.astype(mx.float32) * 2).astype(flat.dtype))
    blk = block()
    state = ef.bind(SimpleNamespace(layers=[SimpleNamespace(experts=blk)]), ProofCache(None, None))

    def generate(rt):
        ef.experts_forward(blk, mx.ones((9, 2304), mx.bfloat16), None, stock)
        ef.experts_forward(blk, mx.ones((9, 2304), mx.bfloat16), None, stock)
        monkeypatch.setenv("MTPLX_INKLING_EXPERT_FUSION", "off")
        ef.experts_forward(blk, mx.ones((40, 2304), mx.bfloat16), None, stock)
        return SimpleNamespace(stats=SimpleNamespace())

    out = ef.observe_expert_request(generate, object())
    receipt = out.stats.inkling_experts
    assert receipt["selections"] == {"nvfp4.r9.fused.fused": 2, "nvfp4.r>32.stock.disabled": 1}
    assert receipt["first_use"] == {"passed": 1} and receipt["mode"] == "auto"
    assert state.counts == receipt["selections"]
    health = ef.expert_fusion_health(SimpleNamespace(layers=[SimpleNamespace(experts=blk)]))
    assert health["bound"] and health["decisions"] == {"nvfp4/bfloat16/fused/verify": "fused"}


def test_unbound_blocks_share_one_in_memory_state(monkeypatch):
    calls, stock = harness(monkeypatch, lambda flat: (flat.astype(mx.float32) * 2).astype(flat.dtype))
    a, b = block(), block()
    hidden = mx.ones((2, 2304), mx.bfloat16)
    ef.experts_forward(a, hidden, None, stock)
    ef.experts_forward(b, hidden, None, stock)
    assert ef._fallback_state.counts == {"nvfp4.r2.fused.fused": 2}
    assert ef._fallback_state.proofs.directory is None


def test_cpu_rows_take_the_stock_body_with_their_reason():
    blk = block()
    stock = lambda hidden: hidden + 1
    hidden = mx.ones((1, 2304), mx.bfloat16)
    out = ef.experts_forward(blk, hidden, None, stock)
    assert mx.array_equal(out, hidden + 1)
    assert ef._state(blk).counts == {"nvfp4.r1.stock.cpu": 1}


def test_kernel_headers_end_with_a_newline_and_names_are_unique():
    caps = (("GATE_CAP", ef.mlx_literal(4.0)), ("UP_CAP", ef.mlx_literal(25.0)))
    specs = [(stage, fp4, ()) for stage in ("inputs", "combine") for fp4 in (True, False)]
    specs += [("swiglu", fp4, caps) for fp4 in (True, False)]
    specs += [("route", True, (("ROUTE_SCORES", s), ("ROUTE_NORMALIZE", n))) for s in (0, 1) for n in (0, 1)]
    names = set()
    for spec in specs:
        name, inputs, outputs, body, header = ef.kernel_spec(*spec)
        assert header.endswith("\n") and name not in names
        names.add(name)
        assert "static_cast<T>" in body or "out[" not in body


def test_quantiser_matches_the_stock_input_rounding_line_for_line():
    stock = Path(activations.__file__).read_text()
    for line in (
        "float a = min(abs(x), 448.0f);",
        "float unit = ldexp(1.0f, max(-9, ilogb(max(a, 0.015625f)) - 3));",
        "return copysign(rint(a / unit) * unit, x);",
        "float y = a > 5 ? 6 : a >= 3.5f ? 4 : a > 2.5f ? 3 : a >= 1.75f ? 2 :",
        "a > 1.25f ? 1.5f : a >= .75f ? 1 : a > .25f ? .5f : 0;",
        "float s6 = inkling_e4m3(top * (encoder / 6.0f));".replace("inkling_", ""),
        "float ae = abs(a * outer - x), be = abs(b * outer - x);",
        "return simd_shuffle(be, first) < simd_shuffle(ae, first) ? b : a;".replace("return ", ""),
        "float scale = top > 0 ? exp2(ceil(log2(top / 448.0f))) : 1.0f;",
    ):
        assert line.replace("inkling_", "") in stock.replace("inkling_", ""), line
        assert line.replace("inkling_", "") in ef.HEADER.replace("fused_", ""), line


def test_constants_are_written_as_the_compiled_swiglu_writes_them():
    assert [ef.mlx_literal(v) for v in (4.0, 25.0, 0.1, 1e-20)] == ["4", "25", "0.1", "1e-20"]


def test_a_bound_real_expert_block_keeps_its_parameters_and_runs_stock_on_cpu():
    from mlx.utils import tree_flatten
    from mtplx.models.inkling.model import Experts

    weights = {"mlp.gate.weight": mx.zeros((128, 2304), mx.bfloat16), "mlp.gate.bias": mx.zeros((128,), mx.float32)}
    for name, (outputs, inputs) in {"w13_weight": (2304, 2304), "w2_weight": (2304, 1152)}.items():
        stem = "mlp.experts." + name
        weights[stem] = mx.zeros((128, outputs, inputs // 2), mx.uint8)
        weights[stem + "_scale"] = mx.zeros((128, outputs, inputs // 16), mx.uint8)
        weights[stem + "_amax"] = mx.ones((128,), mx.float32)
        weights[stem + "_outer_scale_denominator"] = mx.ones((1,), mx.float32)
    config = {"num_experts_per_tok": 8, "route_scale": 8.0, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0}
    experts = Experts(weights, config, "nvfp4")
    before = [key for key, _ in tree_flatten(experts.parameters())]
    state = ef.bind(SimpleNamespace(layers=[SimpleNamespace(experts=experts)]), ProofCache(None, None))
    assert [key for key, _ in tree_flatten(experts.parameters())] == before
    hidden = mx.zeros((1, 2, 2304), mx.bfloat16)
    assert ef.unserved(experts, hidden.reshape(-1, 2304), gpu=True) is None  # the real pack geometry is served
    assert mx.array_equal(ef.experts_forward(experts, hidden, lambda x, w: x @ w.T, experts.stock),
                          experts.stock(hidden))
    assert state.counts == {"nvfp4.r2.stock.cpu": 1}
