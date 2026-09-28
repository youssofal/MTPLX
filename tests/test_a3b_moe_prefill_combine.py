"""One-kernel MoE combine in the A3B prefill, on the invariant lane.

Pinned on Metal against the stock block (the lane's ``BatchInvariantSwitchGLU``
plus mlx-lm's unsort, weight, sum and shared-expert add):

* the combining block gives the stock block's bits at the A3B routing
  geometry (256 experts, top-8) and on the tiny Qwen3.5-MoE (32 experts,
  top-4), from the lane's sorted minimum up to 4,096 rows;
* narrower forwards, decode and the lone final token keep the stock block
  and the counter proves when the kernel ran;
* a whole prefill (logits, hidden, every cache leaf) is bit-identical with
  and without the route;
* installed only with its switch and only on the lane; the load-time
  self-check validates the kernel at the A3B geometry and a failed check
  turns the route off.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx import nn

pytest.importorskip("mlx_lm.models.qwen3_5_moe")

from mlx_lm.models import base, qwen3_next
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from mtplx import a3b_moe_prefill_combine as combine
from mtplx import batch_invariant_prefill as bip
from mtplx import kernel_selfcheck
from mtplx.attention_context import attention_phase
from tests.a3b_tiny_synth import assert_bit_equal, prompt, tiny_model

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture()
def lane(monkeypatch):
    """Scope the invariant lane's process-wide hooks to one test."""

    monkeypatch.setattr(qwen3_next, "scaled_dot_product_attention", qwen3_next.scaled_dot_product_attention)
    monkeypatch.setattr(base, "scaled_dot_product_attention", base.scaled_dot_product_attention)
    monkeypatch.setitem(bip._STOCK_SDPA, "sdpa", None)
    monkeypatch.setitem(bip._STATE, "installed", False)
    monkeypatch.setitem(bip._STATE, "refusal", None)
    monkeypatch.setenv(combine.ENV, "1")
    kernel_selfcheck._reset_for_tests()
    previous = mx.default_device()
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
    yield
    kernel_selfcheck._reset_for_tests()
    mx.set_default_device(previous)


def _a3b_block(hidden: int = 256, intermediate: int = 64):
    """One MoE block with A3B's routing (256 experts, top-8, shared expert)."""

    args = qwen3_next.ModelArgs(
        model_type="qwen3_next",
        hidden_size=hidden,
        num_hidden_layers=1,
        intermediate_size=intermediate,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=256,
        num_experts_per_tok=8,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=intermediate,
        moe_intermediate_size=intermediate,
        mlp_only_layers=[],
        rms_norm_eps=1e-6,
        vocab_size=256,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        max_position_embeddings=4096,
        norm_topk_prob=True,
    )
    mx.random.seed(3)
    block = Qwen3NextSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    nn.quantize(block, group_size=64, bits=6)
    block.gate = nn.QuantizedLinear.from_linear(nn.Linear(hidden, 256, bias=False), 64, 8)
    block.set_dtype(mx.bfloat16)
    mx.eval(block.parameters())
    return block


def _run(block, x, phase="prefill"):
    with attention_phase(phase):
        out = block(x)
    mx.eval(out)
    return out


def _stock_and_combined(block, x, phase="prefill"):
    bip.install_batch_invariant_prefill(block)
    stock = _run(block, x, phase)
    before = combine.COUNTERS["combined_forwards"]
    assert combine.install_a3b_moe_prefill_combine(block) == 1
    ours = _run(block, x, phase)
    block.__class__ = Qwen3NextSparseMoeBlock
    return stock, ours, combine.COUNTERS["combined_forwards"] - before


@needs_metal
@pytest.mark.parametrize("rows", [128, 331, 2048, 4096])
def test_the_combined_block_equals_the_stock_block_at_the_a3b_geometry(lane, rows):
    block = _a3b_block()
    mx.random.seed(rows)
    x = mx.random.normal((1, rows, 256)).astype(mx.bfloat16)
    stock, ours, combined = _stock_and_combined(block, x)
    assert combined == 1
    assert_bit_equal([ours], [stock])


@needs_metal
@pytest.mark.parametrize("rows, phase", [(127, "prefill"), (9, "prefill"), (1, "decode"), (3, "verify")])
def test_narrow_forwards_and_decode_keep_the_stock_block(lane, rows, phase):
    block = _a3b_block()
    x = mx.random.normal((1, rows, 256)).astype(mx.bfloat16)
    stock, ours, combined = _stock_and_combined(block, x, phase)
    assert combined == 0
    assert_bit_equal([ours], [stock])


@needs_metal
def test_the_lone_final_token_scope_keeps_the_stock_block(lane):
    block = _a3b_block()
    x = mx.random.normal((1, 200, 256)).astype(mx.bfloat16)
    bip.install_batch_invariant_prefill(block)
    combine.install_a3b_moe_prefill_combine(block)
    before = combine.COUNTERS["combined_forwards"]
    with bip.stock_prefill_kernels():
        _run(block, x)
    assert combine.COUNTERS["combined_forwards"] == before


def _prefill(model, tokens, chunk):
    cache = model.make_cache()
    outputs = []
    with attention_phase("prefill"):
        for start in range(0, len(tokens), chunk):
            text = model.language_model
            hidden = text.model(mx.array([tokens[start : start + chunk]]), cache)
            logits = text.lm_head(hidden)
            mx.eval(logits, hidden)
            outputs.extend([logits, hidden])
    leaves = [leaf for entry in cache for leaf in entry.state if leaf is not None]
    return outputs + leaves


@needs_metal
@pytest.mark.parametrize("chunk", [96, 256])
def test_a_whole_prefill_is_bit_identical_with_the_route(lane, chunk):
    model = tiny_model()
    bip.install_batch_invariant_prefill(model)
    tokens = prompt(400, seed=9)
    stock = _prefill(model, tokens, chunk)
    assert combine.install_a3b_moe_prefill_combine(model) == 4
    before = combine.COUNTERS["combined_forwards"]
    ours = _prefill(model, tokens, chunk)
    assert combine.COUNTERS["combined_forwards"] > before
    assert_bit_equal(ours, stock)


def test_the_runtime_installs_the_route_only_with_its_switch_and_the_lane(lane, monkeypatch):
    from mtplx import runtime

    model = tiny_model()
    monkeypatch.setenv(combine.ENV, "0")
    runtime._install_batch_invariant_lane(model)
    assert not any(
        type(module) is combine.CombinedPrefillSparseMoeBlock
        for _name, module in model.named_modules()
    )

    monkeypatch.setenv(combine.ENV, "1")
    model = tiny_model()
    runtime._install_batch_invariant_lane(model)
    assert bip.batch_invariant_prefill_status()["moe_prefill_combine_blocks"] == 4

    # 8 experts, top 2: the lane refuses the model, so no route either.
    refused = tiny_model(num_experts=8, num_experts_per_tok=2)
    monkeypatch.setitem(bip._STATE, "installed", False)
    runtime._install_batch_invariant_lane(refused)
    assert not any(
        type(module) is combine.CombinedPrefillSparseMoeBlock
        for _name, module in refused.named_modules()
    )


def test_the_route_is_off_by_default(monkeypatch):
    monkeypatch.delenv(combine.ENV, raising=False)
    assert combine.switched_on() is False
    assert combine.enabled() is False


@needs_metal
def test_the_selfcheck_validates_the_kernel_at_the_a3b_geometry(lane, monkeypatch):
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "")
    assert kernel_selfcheck.selfcheck_enabled() is True
    report = kernel_selfcheck.run_kernel_selfcheck(mx.bfloat16, 6, 64)
    assert report["lanes"][combine.LANE] == "ok"
    assert report["dmax"][combine.LANE] == 0.0
    assert combine.enabled() is True


@needs_metal
def test_a_failed_selfcheck_turns_the_route_off(lane, monkeypatch):
    monkeypatch.setattr(kernel_selfcheck, "_check_a3b_moe_prefill_combine", lambda _mx: 1.0)
    report = kernel_selfcheck.run_kernel_selfcheck(mx.bfloat16, 6, 64)
    assert report["lanes"][combine.LANE] == "fallback"
    assert combine.enabled() is False
    block = _a3b_block()
    x = mx.random.normal((1, 200, 256)).astype(mx.bfloat16)
    stock, ours, combined = _stock_and_combined(block, x)
    assert combined == 0
    assert_bit_equal([ours], [stock])
