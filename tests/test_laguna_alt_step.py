"""Contracts for the standalone alternative Laguna-S-2.1 runtime.

The alt lane (``mtplx.laguna_alt_step``) is the head-to-head twin of
``LagunaCompiledLane``: same weights, same cache-state plumbing, but a forward
that will route through the ported mlx.fast challenge kernels as
``PORT_LEDGER.md`` is executed. These tests pin the foundation:

* with every kernel off (``STOCK``), the alt lane is digest-identical to the
  reference lane — so any future divergence is a ported kernel's fault, isolated;
* the async-eval ladder (LEDGER S1) is a latency lever only, never a numerics
  change;
* a kernel flag that is turned on before its kernel is wired fails loudly rather
  than silently running the stock span and faking a win.

Toy geometry (window 8, 12-token prompt) on the CPU device, mirroring
``tests/test_laguna_compiled_step.py``.
"""

from __future__ import annotations

from dataclasses import fields

import mlx.core as mx
import pytest

from mtplx.laguna_alt_step import (
    LADDER_EVERY_STEP,
    LADDER_XS21,
    STOCK,
    AltConfig,
    LagunaAltLane,
    alt_prefill_forward,
)
from mtplx.laguna_compiled_step import LagunaCompiledLane
from mtplx.models.laguna import Model, ModelArgs

LAYER_TYPES = [
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]
PROMPT = mx.array([[3, 9, 14, 2, 7, 21, 5, 11, 30, 1, 18, 6]], dtype=mx.uint32)
CAP = 32
STEPS = 12


def _toy_args(**updates):
    config = dict(
        model_type="laguna",
        hidden_size=64,
        num_hidden_layers=len(LAYER_TYPES),
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=256,
        rms_norm_eps=1e-6,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[0],
        gating="per-head",
        sliding_window=8,
        layer_types=list(LAYER_TYPES),
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 500_000.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    config.update(updates)
    return ModelArgs(**config)


@pytest.fixture
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def toy_model(cpu_device):
    mx.random.seed(3)
    model = Model(_toy_args())
    mx.eval(model.parameters())
    return model


def _greedy_token(logits):
    return mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]


def _prefill(model):
    caches = model.make_cache()
    token = _greedy_token(model(PROMPT, cache=caches, logits_keep=1))
    mx.eval(token)
    return caches, token


def _advance_lane(lane_cls, model, **kw):
    caches, token = _prefill(model)
    lane = lane_cls(model, cap=CAP, compiled=False, **kw)
    lane.seed(caches, token)
    return [int(token.item())] + [int(lane.advance().item()) for _ in range(STEPS)]


def test_all_stock_alt_lane_matches_reference(toy_model):
    ref = _advance_lane(LagunaCompiledLane, toy_model)
    alt = _advance_lane(LagunaAltLane, toy_model, config=STOCK)
    assert alt == ref


def test_packed_kv_layout_agrees(toy_model):
    ref = _advance_lane(LagunaCompiledLane, toy_model)
    alt = _advance_lane(LagunaAltLane, toy_model, config=STOCK, packed_kv=True)
    assert alt == ref


def test_async_eval_ladder_is_value_preserving(toy_model):
    ref = _advance_lane(LagunaCompiledLane, toy_model)
    for ladder in (LADDER_EVERY_STEP, LADDER_XS21):
        caches, token = _prefill(toy_model)
        lane = LagunaAltLane(toy_model, cap=CAP, compiled=False, config=STOCK)
        lane.seed(caches, token)
        got = [int(token.item())] + lane.generate(STEPS, ladder=ladder)
        assert got == ref, f"ladder {ladder} changed tokens"


# Kernels wired into the alt lane so far; these must NOT raise when enabled.
_WIRED_DECODE_FLAGS = {
    "d1_residual_router",
    "d6_sdpa_vector",
    "d14_lm_head_prune",
    "d4_input_qkvg",
}

# Decode flags NOT yet wired: enabling one must fail loudly (anti-fake-win).
# Prefill flags (p*) gate a prefill path not built yet, so they legitimately do
# not fire during a decode advance() and are excluded here.
_UNWIRED_DECODE_FLAGS = [
    f.name
    for f in fields(AltConfig)
    if f.name.startswith("d") and f.name not in _WIRED_DECODE_FLAGS
]


@pytest.mark.parametrize("flag", _UNWIRED_DECODE_FLAGS)
def test_unwired_kernel_flag_fails_loudly(toy_model, flag):
    """A flag flipped on before its kernel is wired must raise, never no-op.

    This is the anti-fake-win guard: the whole point of the port is that a
    turned-on kernel actually runs, so a half-wired flag has to fail rather than
    quietly fall through to the stock span and report the reference's numbers as
    the kernel's.
    """

    config = AltConfig(**{flag: True})
    caches, token = _prefill(toy_model)
    lane = LagunaAltLane(toy_model, cap=CAP, compiled=False, config=config)
    lane.seed(caches, token)
    with pytest.raises(NotImplementedError):
        lane.advance()


def test_d1_residual_router_matches_reference(toy_model):
    """D1 wired: the fused residual+norm+router path produces reference tokens.

    On the CPU toy the metal kernel is ineligible so it falls back to the stock
    add+norm+matmul, but the fallback still flows through the full D1 wiring
    (fused_residual_norm_router -> _moe_from_precomputed), so a match confirms the
    integration (residual add across the boundary, precomputed-logits MoE) is
    faithful. The metal kernel itself is proven at the real S-2.1 shape under the
    GPU A/B (allclose + whole-runtime digest-hold).
    """

    ref = _advance_lane(LagunaCompiledLane, toy_model)
    alt = _advance_lane(
        LagunaAltLane, toy_model, config=AltConfig(d1_residual_router=True)
    )
    assert alt == ref


def test_d6_sdpa_vector_matches_reference(toy_model):
    """D6 wired: the group-3 GQA decode SDPA produces reference tokens.

    On the CPU toy (head_dim 8) the kernel is ineligible so it falls back to
    stock SDPA; the fallback path is exercised and the tokens must match. The
    kernel itself is proven at the real (72 heads / gqa 9 sliding, 48/gqa 6 full)
    shape by a standalone allclose check + the GPU A/B digest-hold.
    """

    ref = _advance_lane(LagunaCompiledLane, toy_model)
    alt = _advance_lane(
        LagunaAltLane, toy_model, config=AltConfig(d6_sdpa_vector=True)
    )
    assert alt == ref


def test_d14_lm_head_prune_matches_reference(toy_model):
    """D14 wired: the 8-bit lm-head top-1 path produces reference tokens.

    The toy lm_head is a plain (non-quantized) nn.Linear, so D14 is ineligible and
    falls back to the stock head + argmax — this test guards the fallback + the
    non-tied branch. Whether the real 8-bit top-1 equals the stock argmax bit-for-bit
    is decided by the GPU A/B digest (greedy is unforgiving of any argmax drift).
    """

    ref = _advance_lane(LagunaCompiledLane, toy_model)
    alt = _advance_lane(
        LagunaAltLane, toy_model, config=AltConfig(d14_lm_head_prune=True)
    )
    assert alt == ref


def test_alt_prefill_forward_matches_reference(toy_model):
    """The alt prefill forward reproduces the eager reference's first token.

    STOCK is the eager forward exactly; the D1 variant falls back on the toy
    (ineligible shape) but flows through the fused-residual-router prefill wiring.
    The P5 variants (D1-free and D1-coupled) exercise the MoE-combine tail wiring:
    on the toy the metal combine is ineligible so it falls back to residual +
    stock combine, which must still equal the reference. The real S-2.1 combine's
    bit-exactness is proven by the GPU prefill sweep (digest-exact across ctx).
    Every config must yield the reference's post-prefill argmax token.
    """

    c1 = toy_model.make_cache()
    ref_tok = int(_greedy_token(toy_model(PROMPT, cache=c1, logits_keep=1)).item())

    configs = (
        STOCK,
        AltConfig(d1_residual_router=True),
        AltConfig(p5_prefill_moe_tail=True),  # D1-free P5 path (logits via moe.gate)
        AltConfig(d1_residual_router=True, p5_prefill_moe_tail=True),
    )
    for cfg in configs:
        cache = toy_model.make_cache()
        hidden = alt_prefill_forward(toy_model, PROMPT, cache, config=cfg)
        tok = int(_greedy_token(toy_model.lm_head(hidden[:, -1:, :])).item())
        assert tok == ref_tok, f"alt prefill {cfg} first token {tok} != ref {ref_tok}"


def test_d4_input_qkvg_matches_reference(toy_model):
    """D4 wired: gating the fused q/k/v/g projection produces reference tokens.

    Without ``install_fused_qkvg`` the toy has no ``_qkvg``, so D4 runs the four
    separate projections and must match. The fused path's numerics are checked by
    the GPU A/B digest when FUSED_QKVG is installed.
    """

    ref = _advance_lane(LagunaCompiledLane, toy_model)
    alt = _advance_lane(
        LagunaAltLane, toy_model, config=AltConfig(d4_input_qkvg=True)
    )
    assert alt == ref
