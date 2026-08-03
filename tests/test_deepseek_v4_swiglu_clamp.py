"""Gates for the ``swiglu_limit`` clamp on the DeepSeek-V4 **routed** experts.

The reference applies the clamp inside every expert, routed and shared alike
(``MoE.__init__`` L624 hands ``swiglu_limit=args.swiglu_limit`` to each routed
``Expert``, L627 does the same for the shared one).  ``Expert.forward``,
model.py L596-606, verbatim::

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))

Three things there are easy to get wrong, and each has a mutation below:

  * the branches are clamped **asymmetrically** -- ``up`` (``w3`` = ``up_proj``)
    two-sided to ``[-limit, +limit]``, ``gate`` (``w1`` = ``gate_proj``) only at
    its upper tail, keeping the entire negative range that feeds ``silu``;
  * both cuts are **pre-activation**, on the raw projections;
  * ``limit <= 0`` means *no clamp*, not a clamp at zero.

The shared expert already had this (:class:`DeepseekV4MLP`, pinned by
``test_shared_expert_applies_the_swiglu_clamp`` in test_deepseek_v4_mtp.py); the
routed experts run through mlx-lm's ``SwitchGLU`` and reach it via
:class:`ClampedSwiGLU` on the ``activation`` seam.

Every gate here drives the branches into saturation on all four sides
(gate above/below +/-limit, up above/below +/-limit) and **asserts** it did --
the clamp being a no-op on the test inputs is exactly how this went untested the
first time, so the saturation counts are part of the gate, not a comment.

Both parity goldens were captured at ``swiglu_limit=0``; that path is held
bit-identical to a stock unclamped ``SwitchGLU`` below, which is what keeps them
valid.  NumPy float64 oracle, CPU device so MLX fp32 is bit-exact IEEE rather
than its reduced-precision GPU matmul path.  No torch, no download.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU  # noqa: E402

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
_spec = importlib.util.spec_from_file_location("dsv4_clamp_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_clamp_undertest"] = D
_spec.loader.exec_module(D)


# --------------------------------------------------------------------------- #
# shrunk config.  LIMIT and W_SCALE are chosen together so the pre-activation
# projections straddle +/-LIMIT well on both sides: hidden_size 32 with weights
# ~N(0, W_SCALE^2) and x ~N(0,1) gives branch values of std ~sqrt(32)*0.5 = 2.8
# against a limit of 1.5.  The saturation assert below is what actually holds
# this true, not the arithmetic in this comment.
# --------------------------------------------------------------------------- #
CFG = dict(
    vocab_size=61, hidden_size=32, num_hidden_layers=2, num_hash_layers=1,
    num_attention_heads=4, head_dim=16, qk_rope_head_dim=8,
    q_lora_rank=12, o_lora_rank=6, o_groups=2,
    moe_intermediate_size=10, n_routed_experts=6, num_experts_per_tok=2,
    index_n_heads=4, index_head_dim=16, index_topk=4,
    compress_ratios=[0, 4, 0], sliding_window=5,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6,
    rope_theta=10000.0, routed_scaling_factor=1.5, scoring_func="sqrtsoftplus",
    swiglu_limit=0.0, num_nextn_predict_layers=1,
)
LIMIT = 1.5
W_SCALE = 0.5
HASH_LAYER, SCORE_LAYER = 0, 1     # num_hash_layers=1


def _args(**over):
    c = dict(CFG)
    c.update(over)
    return D.ModelArgs(**c)


def m2n(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


# --------------------------------------------------------------------------- #
# NumPy oracle (float64), transcribed from the reference
# --------------------------------------------------------------------------- #
def np_silu(x):
    return x / (1.0 + np.exp(-x))


def np_sqrt_softplus(z):
    """sqrt(softplus(z)), stable form (reference Gate.forward L563-570)."""
    return np.sqrt(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0))


MODES = ("ref", "none", "flip", "sym_gate", "upper_up", "loose", "post")


def np_expert(row, w1, w2, w3, limit, mode):
    """One reference ``Expert.forward`` (L596-606) minus the routing weight,
    which ``MoE.forward`` folds in before ``w2``; ``w2`` is linear, so applying
    it after (as the MLX side does) is the same number.

    ``mode="ref"`` is the faithful transcription of L600-602; every other mode
    is a deliberate corruption the gate must reject.
    """
    assert mode in MODES, f"unknown mode {mode!r}"
    gate = row @ w1.T
    up = row @ w3.T
    if limit > 0:
        if mode == "ref":                            # L601-602, verbatim
            up = np.clip(up, -limit, limit)
            gate = np.minimum(gate, limit)
        elif mode == "flip":                         # mutation: branches swapped
            up = np.minimum(up, limit)
            gate = np.clip(gate, -limit, limit)
        elif mode == "sym_gate":                     # mutation: gate two-sided
            up = np.clip(up, -limit, limit)
            gate = np.clip(gate, -limit, limit)
        elif mode == "upper_up":                     # mutation: up one-sided
            up = np.minimum(up, limit)
            gate = np.minimum(gate, limit)
        elif mode == "loose":                        # mutation: limit loosened
            up = np.clip(up, -4 * limit, 4 * limit)
            gate = np.minimum(gate, 4 * limit)
        # "none": clamp removed.  "post": handled after the activation, below.
    act = np_silu(gate) * up
    if limit > 0 and mode == "post":                 # mutation: after the silu
        act = np.clip(act, -limit, limit)
    return act @ w2.T


def np_moe(x, P, c, *, limit, routed_mode="ref", ids=None):
    """Reference ``MoE.forward`` L629-644 over ``Gate`` L546-584.

    ``routed_mode`` corrupts only the routed experts; the shared expert always
    runs the faithful clamp, so a mutation failure localizes to the routed path
    that :class:`ClampedSwiGLU` owns.  Also returns how many pre-activation
    values landed in each of the four saturation regions.
    """
    flat = x.reshape(-1, x.shape[-1])
    scores = np_sqrt_softplus(flat @ P["gate.weight"].T)
    if "gate.tid2eid" in P:
        idx = P["gate.tid2eid"].astype(np.int64)[np.asarray(ids).reshape(-1)]
    else:
        biased = scores + P["gate.e_score_correction_bias"]
        idx = np.argsort(-biased, axis=-1, kind="stable")[:, : c["topk"]]
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / w.sum(-1, keepdims=True) * c["route_scale"]

    g1 = P["switch_mlp.gate_proj.weight"]
    g2 = P["switch_mlp.down_proj.weight"]
    g3 = P["switch_mlp.up_proj.weight"]

    sat = dict(gate_hi=0, gate_lo=0, up_hi=0, up_lo=0, total=0)
    y = np.zeros_like(flat)
    for t in range(flat.shape[0]):
        for k in range(c["topk"]):
            e = int(idx[t, k])
            raw_gate = flat[t] @ g1[e].T
            raw_up = flat[t] @ g3[e].T
            sat["gate_hi"] += int((raw_gate > limit).sum())
            sat["gate_lo"] += int((raw_gate < -limit).sum())
            sat["up_hi"] += int((raw_up > limit).sum())
            sat["up_lo"] += int((raw_up < -limit).sum())
            sat["total"] += raw_gate.size
            y[t] += w[t, k] * np_expert(flat[t], g1[e], g2[e], g3[e], limit, routed_mode)

    y = y + np_expert(flat, P["shared_experts.gate_proj.weight"],
                      P["shared_experts.down_proj.weight"],
                      P["shared_experts.up_proj.weight"], limit, "ref")
    return y.reshape(x.shape), sat


# --------------------------------------------------------------------------- #
def _fill(module, args, seed):
    """Seed every parameter and return the oracle's float64 view of them."""
    rng = np.random.default_rng(seed)
    new = {}
    for k, v in tree_flatten(module.parameters()):
        if k.endswith("tid2eid"):
            new[k] = mx.array(rng.integers(0, args.n_routed_experts,
                                           size=v.shape).astype(np.int32))
        else:
            new[k] = mx.array((rng.standard_normal(v.shape) * W_SCALE).astype(np.float32))
    module.update(tree_unflatten(list(new.items())))
    mx.eval(module.parameters())
    return {k: m2n(v) for k, v in new.items()}


def _build_moe(*, limit, layer_id=SCORE_LAYER, seed=0):
    args = _args(swiglu_limit=limit)
    moe = D.DeepseekV4MoE(args, layer_id)
    P = _fill(moe, args, seed)
    c = dict(topk=args.num_experts_per_tok, route_scale=args.routed_scaling_factor)
    return args, moe, P, c


def _inputs(seq, seed=99, vocab=CFG["vocab_size"]):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, seq, CFG["hidden_size"]))
    ids = rng.integers(0, vocab, size=(2, seq)).astype(np.int32)
    return x, ids


def _run(moe, x, ids=None):
    out = moe(mx.array(x.astype(np.float32)),
              None if ids is None else mx.array(ids))
    mx.eval(out)
    return m2n(out)


def _assert_saturated(sat, what):
    """The clamp must actually bind, on every side, or the gate proves nothing.

    ``gate_lo`` is the load-bearing one: it counts values below ``-limit`` on the
    gate branch, which the reference deliberately does *not* clamp.  Without
    those, a symmetric gate clamp is indistinguishable from the reference.
    """
    assert sat["total"] > 0, what
    for region in ("gate_hi", "gate_lo", "up_hi", "up_lo"):
        frac = sat[region] / sat["total"]
        assert frac > 0.05, (
            f"{what}: {region} only {sat[region]}/{sat['total']} ({frac:.3f}) -- "
            "inputs do not drive the clamp, so this gate is vacuous")


# --------------------------------------------------------------------------- #
# 1. numerical parity, clamp ACTIVE
# --------------------------------------------------------------------------- #
# indices.size >= 64 flips SwitchGLU into its gather-sorted expert path, so both
# sides of that branch are covered: 2*6*2=24 (unsorted) and 2*40*2=160 (sorted).
@pytest.mark.parametrize("seq,sorted_path", [(6, False), (40, True)])
def test_routed_experts_match_reference_with_clamp_active(seq, sorted_path):
    args, moe, P, c = _build_moe(limit=LIMIT)
    x, _ = _inputs(seq)
    assert (x.shape[0] * seq * c["topk"] >= 64) is sorted_path, "sorted-path assumption"

    got = _run(moe, x)
    ref, sat = np_moe(x, P, c, limit=LIMIT)
    _assert_saturated(sat, f"seq={seq}")

    mad = float(np.max(np.abs(got - ref)))
    scale = float(np.max(np.abs(ref)))
    assert mad / scale < 1e-5, f"routed MoE diverges: max_abs={mad:.3e} scale={scale:.3e}"


def test_routed_experts_match_reference_on_a_hash_layer():
    """Hash layers route by token id, not score -- a different gate, the same
    experts.  Covers the ``layer_id < num_hash_layers`` construction."""
    args, moe, P, c = _build_moe(limit=LIMIT, layer_id=HASH_LAYER)
    assert moe.gate.hash, "expected a hash-routed layer"
    x, ids = _inputs(6)
    got = _run(moe, x, ids)
    ref, sat = np_moe(x, P, c, limit=LIMIT, ids=ids)
    _assert_saturated(sat, "hash layer")
    mad = float(np.max(np.abs(got - ref)))
    assert mad / float(np.max(np.abs(ref))) < 1e-5, f"hash-layer MoE diverges: {mad:.3e}"


def test_the_clamp_actually_changes_the_result():
    """Guard against a clamp that is wired but inert: same weights, same input,
    limit on vs off must differ, and by a wide margin at this saturation."""
    _, clamped, P, c = _build_moe(limit=LIMIT)
    _, plain, P2, _ = _build_moe(limit=0.0)
    assert set(P) == set(P2) and all(np.array_equal(P[k], P2[k]) for k in P)
    x, _ = _inputs(6)
    a, b = _run(clamped, x), _run(plain, x)
    rel = float(np.max(np.abs(a - b))) / float(np.max(np.abs(b)))
    assert rel > 0.1, f"clamp is inert: max_rel={rel:.3e}"


# --------------------------------------------------------------------------- #
# 2. mutation gate -- each corruption of the clamp must be REJECTED
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["none", "flip", "sym_gate", "upper_up", "loose", "post"])
def test_mutated_clamp_semantics_are_rejected(mode):
    args, moe, P, c = _build_moe(limit=LIMIT)
    x, _ = _inputs(6)
    got = _run(moe, x)
    ref_ok, sat = np_moe(x, P, c, limit=LIMIT)
    _assert_saturated(sat, mode)
    # sanity: the faithful oracle passes the bound the mutation must fail
    assert float(np.max(np.abs(got - ref_ok))) / float(np.max(np.abs(ref_ok))) < 1e-5

    bad, _ = np_moe(x, P, c, limit=LIMIT, routed_mode=mode)
    rel = float(np.max(np.abs(got - bad))) / float(np.max(np.abs(ref_ok)))
    assert rel > 1e-5, f"mutation {mode!r} is not caught (max_rel={rel:.3e})"


def test_a_wrong_limit_value_is_rejected():
    """The magnitude matters, not just the shape of the clamp."""
    args, moe, P, c = _build_moe(limit=LIMIT)
    x, _ = _inputs(6)
    got = _run(moe, x)
    for other in (0.5 * LIMIT, 2.0 * LIMIT):
        bad, _ = np_moe(x, P, c, limit=other)
        rel = float(np.max(np.abs(got - bad))) / float(np.max(np.abs(bad)))
        assert rel > 1e-5, f"limit={other} indistinguishable from {LIMIT} (rel={rel:.3e})"


# --------------------------------------------------------------------------- #
# 3. swiglu_limit=0 is bit-identical to stock -- this is what keeps the goldens
# --------------------------------------------------------------------------- #
def test_clamped_activation_at_zero_is_bit_identical_to_stock_swiglu():
    rng = np.random.default_rng(3)
    # values well outside +/-LIMIT, so a clamp that leaked in would show
    up = mx.array((rng.standard_normal((4, 1, 2, 16)) * 5.0).astype(np.float32))
    gate = mx.array((rng.standard_normal((4, 1, 2, 16)) * 5.0).astype(np.float32))
    stock, off = SwiGLU()(up, gate), D.ClampedSwiGLU(0.0)(up, gate)
    mx.eval(stock, off)
    assert bool(mx.array_equal(stock, off)), "limit=0 diverges from stock SwiGLU"
    # and a negative / None limit is 'disabled', not 'clamp at 0'
    for disabled in (0.0, -1.0, None):
        out = D.ClampedSwiGLU(disabled)(up, gate)
        mx.eval(out)
        assert bool(mx.array_equal(stock, out)), f"limit={disabled} is not the stock path"
    on = D.ClampedSwiGLU(LIMIT)(up, gate)
    mx.eval(on)
    assert not bool(mx.array_equal(stock, on)), "clamp at LIMIT did nothing"


def test_moe_at_limit_zero_is_bit_identical_to_a_stock_switchglu():
    """End-to-end on the module the goldens exercise: a ``DeepseekV4MoE`` built
    at ``swiglu_limit=0`` must equal one whose routed experts are the unmodified
    mlx-lm ``SwitchGLU``, bit for bit."""
    args, moe, P, c = _build_moe(limit=0.0)
    stock = SwitchGLU(args.hidden_size, args.moe_intermediate_size,
                      args.n_routed_experts)
    stock.update(moe.switch_mlp.parameters())
    mx.eval(stock.parameters())
    assert isinstance(stock.activation, SwiGLU)
    assert not isinstance(stock.activation, D.ClampedSwiGLU)

    for seq in (6, 40):                      # unsorted and gather-sorted paths
        x, _ = _inputs(seq)
        xf = mx.array(x.reshape(-1, args.hidden_size).astype(np.float32))
        idx, _ = moe.gate(xf, None)
        a, b = moe.switch_mlp(xf, idx), stock(xf, idx)
        mx.eval(a, b)
        assert bool(mx.array_equal(a, b)), f"seq={seq} not bit-identical to stock"


def test_limit_zero_leaves_the_parameter_tree_untouched():
    """``ClampedSwiGLU`` holds no arrays, so ``sanitize`` -> ``quantize`` ->
    ``load_weights(strict=True)`` sees exactly the keys it saw before."""
    keys = {}
    for limit in (0.0, 10.0):
        model = D.Model(_args(swiglu_limit=limit))
        keys[limit] = {k for k, _ in tree_flatten(model.parameters())}
    assert keys[0.0] == keys[10.0]
    assert not any("activation" in k for k in keys[10.0])


# --------------------------------------------------------------------------- #
# 4. every routed-expert site is covered
# --------------------------------------------------------------------------- #
def test_every_routed_expert_site_carries_the_clamp():
    """Trunk score layers, trunk hash layers and the MTP draft block.  All three
    build their experts through ``DeepseekV4MoE.__init__``, which is the point --
    but assert it, so a future extra construction site cannot slip past."""
    args = _args(swiglu_limit=LIMIT)
    model = D.Model(args)
    sites = {f"layers.{i}": layer.ffn for i, layer in enumerate(model.model.layers)}
    assert model.has_mtp
    sites["mtp.0"] = model.mtp[0].ffn
    assert len(sites) == args.num_hidden_layers + 1

    for name, ffn in sites.items():
        act = ffn.switch_mlp.activation
        assert isinstance(act, D.ClampedSwiGLU), f"{name}: routed experts unclamped ({act})"
        assert act.limit == LIMIT, f"{name}: limit={act.limit}"
        assert ffn.shared_experts.limit == LIMIT, f"{name}: shared expert"
    # both gate kinds are represented, so 'hash layers' is really covered
    assert sites[f"layers.{HASH_LAYER}"].gate.hash
    assert not sites[f"layers.{SCORE_LAYER}"].gate.hash
    assert not sites["mtp.0"].gate.hash


def test_every_routed_expert_site_is_functionally_clamped():
    """The attribute check above would survive an activation that is installed
    but never called.  Drive each site's routed path and require the clamp to
    move the numbers."""
    on, off = D.Model(_args(swiglu_limit=LIMIT)), D.Model(_args(swiglu_limit=0.0))
    rng = np.random.default_rng(7)
    new = {}
    for k, v in tree_flatten(off.parameters()):
        new[k] = (mx.array(rng.integers(0, CFG["n_routed_experts"], size=v.shape).astype(np.int32))
                  if k.endswith("tid2eid")
                  else mx.array((rng.standard_normal(v.shape) * W_SCALE).astype(np.float32)))
    tree = tree_unflatten(list(new.items()))
    on.update(tree)
    off.update(tree)
    mx.eval(on.parameters(), off.parameters())

    x, ids = _inputs(6)
    xf = mx.array(x.reshape(-1, CFG["hidden_size"]).astype(np.float32))
    ids_f = mx.array(ids.reshape(-1))

    pairs = [(f"layers.{i}", a.ffn, b.ffn)
             for i, (a, b) in enumerate(zip(on.model.layers, off.model.layers))]
    pairs.append(("mtp.0", on.mtp[0].ffn, off.mtp[0].ffn))
    for name, ffn_on, ffn_off in pairs:
        idx, _ = ffn_off.gate(xf, ids_f)
        a, b = ffn_on.switch_mlp(xf, idx), ffn_off.switch_mlp(xf, idx)
        mx.eval(a, b)
        rel = float(mx.max(mx.abs(a - b)).item()) / float(mx.max(mx.abs(b)).item())
        assert rel > 0.1, f"{name}: routed experts unaffected by the clamp (rel={rel:.3e})"
