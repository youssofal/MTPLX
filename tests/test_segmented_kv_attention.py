"""Segmented KV through the real attention layer (attention_split ladder), GPU.

Qwen3-Next attention at the 27B head geometry (24 query / 4 KV heads, head_dim 256) with a small
hidden size: a stock KVCache and a SegmentedKVCache see the same turns, the same verify windows and
the same rollbacks and must agree.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_map
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention

from mtplx.attention_split import configure_split_full_attention
from mtplx.kernels import sdpa_segmented as S
from mtplx.nax_verify import nax_available
from mtplx.segmented_kv import SegmentedKVCache

HIDDEN = 128


def _model():
    args = SimpleNamespace(
        hidden_size=HIDDEN, num_attention_heads=24, num_key_value_heads=4, head_dim=256,
        attention_bias=False, rms_norm_eps=1e-6, partial_rotary_factor=0.25,
        rope_theta=10000.0, rope_scaling=None, max_position_embeddings=65536,
    )
    mx.random.seed(0)
    attn = Qwen3NextAttention(args)
    attn.update(tree_map(lambda p: p.astype(mx.bfloat16), attn.parameters()))
    mx.eval(attn.parameters())
    layer = SimpleNamespace(is_linear=False, self_attn=attn)
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    return model, attn


def _x(seed: int, n: int) -> mx.array:
    return (mx.random.normal((1, n, HIDDEN), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)


def _top_ulps(a, b) -> float:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    top = float(mx.maximum(mx.abs(a).max(), mx.abs(b).max()).item())
    return float(mx.abs(a - b).max().item()) / 2.0 ** (math.floor(math.log2(top)) - 7)


def _run_turns(attn, cache, route_env: bool):
    """Turn 1 (two prefill chunks), snapshot at turn end, turn 2 on a restored cache, then verify
    windows with rollback. Returns every output."""
    outs = []
    for chunk, (seed, n) in enumerate([(1, 4608), (2, 4608)]):
        outs.append(attn(_x(seed, n), mask="causal", cache=cache))
    mx.eval(outs)
    if isinstance(cache, SegmentedKVCache):
        state = cache.state  # the snapshot: seals the tail
        cache = SegmentedKVCache()
        cache.state = state
    outs.append(attn(_x(3, 300), mask="causal", cache=cache))  # turn 2 prefill over history
    for seed in (4, 5, 6):
        outs.append(attn(_x(seed, 4), mask="causal", cache=cache))  # verify window
        cache.trim(2)  # accept two, roll back two
    outs.append(attn(_x(7, 1), mask=None, cache=cache))
    mx.eval(outs)
    return outs, cache


@pytest.fixture
def lane(monkeypatch):
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    monkeypatch.setenv("MTPLX_NAX_FLASH_ROUTE", "1")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MAX_SEGMENTS", "1000")
    monkeypatch.delenv("MTPLX_SEGMENTED_KV_PREFILL", raising=False)
    model, attn = _model()
    configure_split_full_attention(model)
    return attn


@pytest.mark.skipif(not mx.metal.is_available() or not nax_available(), reason="TensorOps unavailable")
def test_segmented_attention_matches_the_stock_cache_through_turns_and_rollback(lane) -> None:
    stock, _ = _run_turns(lane, KVCache(), True)
    S.segmented_dispatch_counts.clear()
    seg, cache = _run_turns(lane, SegmentedKVCache(), True)
    assert len(stock) == len(seg)
    for a, b in zip(stock, seg):
        assert bool(mx.all(mx.isfinite(b)).item())
        assert _top_ulps(a, b) <= 3
    # the verify windows over history + tail went through the segment kernel
    assert S.segmented_dispatch_counts.get("dispatched", 0) >= 3
    assert cache.segment_count == 2


@pytest.mark.skipif(not mx.metal.is_available() or not nax_available(), reason="TensorOps unavailable")
def test_single_segment_is_bit_identical_to_the_stock_cache(lane) -> None:
    def run(cache):
        outs = [lane(_x(1, 9000), mask="causal", cache=cache)]
        for seed in (4, 5):
            outs.append(lane(_x(seed, 4), mask="causal", cache=cache))
            cache.trim(2)
        mx.eval(outs)
        return outs

    for a, b in zip(run(KVCache()), run(SegmentedKVCache())):
        assert mx.array_equal(a, b).item()


@pytest.mark.skipif(not mx.metal.is_available() or not nax_available(), reason="TensorOps unavailable")
def test_with_the_route_switched_off_the_gather_fallback_is_the_stock_result(monkeypatch) -> None:
    """Both attention switches set to 0 (the kill switch): segments gather into one array and the
    stock SDPA runs on it. The support check keeps such a model on stock caches; this pins the
    fallback's correctness."""
    for name in ("MTPLX_GQA_PACKED_SDPA", "MTPLX_NAX_FLASH_ROUTE"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_PREFILL", "gather")  # the LSE prefill route is not bitwise stock
    model, attn = _model()
    configure_split_full_attention(model)
    stock, _ = _run_turns(attn, KVCache(), False)
    seg, _ = _run_turns(attn, SegmentedKVCache(), False)
    for a, b in zip(stock, seg):
        assert mx.array_equal(a, b).item()


@pytest.mark.skipif(not mx.metal.is_available() or not nax_available(), reason="TensorOps unavailable")
@pytest.mark.parametrize("profile_env", [{}, {"MTPLX_VLLM_METAL_PAGED_ATTN": "1"}])
def test_without_the_attention_switches_the_segment_kernel_serves(monkeypatch, profile_env) -> None:
    """The sustained profile sets neither MTPLX_GQA_PACKED_SDPA nor MTPLX_NAX_FLASH_ROUTE: a segmented
    cache still takes the segment kernel and never gathers, and agrees with the stock cache."""
    import mtplx.segmented_kv as module

    for name in ("MTPLX_GQA_PACKED_SDPA", "MTPLX_NAX_FLASH_ROUTE", "MTPLX_VLLM_METAL_PAGED_ATTN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in profile_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_PREFILL", "lse")
    model, attn = _model()
    configure_split_full_attention(model)
    stock, _ = _run_turns(attn, KVCache(), False)
    S.segmented_dispatch_counts.clear()
    module.route_counts.clear()
    seg, _ = _run_turns(attn, SegmentedKVCache(), False)
    assert not any(k.startswith("gather") for k in module.route_counts), module.route_counts
    assert S.segmented_dispatch_counts.get("dispatched", 0) >= 3
    for a, b in zip(stock, seg):
        assert bool(mx.all(mx.isfinite(b)).item())
        assert _top_ulps(a, b) <= 8


@pytest.mark.skipif(not mx.metal.is_available() or not nax_available(), reason="TensorOps unavailable")
@pytest.mark.parametrize("head_dim, gqa", [(256, 6), (128, 4), (128, 1)])
def test_the_selfcheck_probe_passes_on_the_real_segment_kernel(head_dim, gqa) -> None:
    from mtplx.kernel_selfcheck import _SDPA_TOLERANCE, _check_segments

    for dtype in (mx.bfloat16, mx.float16):
        assert _check_segments(mx, dtype, d=head_dim, gqa=gqa) <= _SDPA_TOLERANCE


@pytest.mark.skipif(not mx.metal.is_available() or not nax_available(), reason="TensorOps unavailable")
def test_lse_prefill_route_matches_the_gather_route(lane, monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_PREFILL", "gather")
    a, _ = _run_turns(lane, SegmentedKVCache(), True)
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_PREFILL", "lse")
    b, _ = _run_turns(lane, SegmentedKVCache(), True)
    for x, y in zip(a, b):
        assert _top_ulps(x, y) <= 8  # fp32 reference route vs bf16 SDPA on the same rows


@pytest.mark.skipif(not mx.metal.is_available() or not nax_available(), reason="TensorOps unavailable")
@pytest.mark.parametrize("window", [6, 9, 17, 32])
def test_wide_verify_windows_run_on_the_segments_without_a_gather(lane, window) -> None:
    """Context-copy windows (6 to 32 rows): sub-windows over the segments, history never gathered."""
    import mtplx.segmented_kv as module

    def run(cache):
        outs = [lane(_x(1, 9000), mask="causal", cache=cache)]
        if isinstance(cache, SegmentedKVCache):
            state = cache.state
            cache = SegmentedKVCache()
            cache.state = state
        outs.append(lane(_x(2, 300), mask="causal", cache=cache))
        for seed in (4, 5):
            outs.append(lane(_x(seed, window), mask="causal", cache=cache))
            cache.trim(window - 3)
        mx.eval(outs)
        return outs

    stock = run(KVCache())
    module.route_counts.clear()
    seg = run(SegmentedKVCache())
    assert not any(k.startswith("gather_q") and k[8:].isdigit() for k in module.route_counts), module.route_counts
    assert module.route_counts.get(f"chunked_q{window}", 0) + module.route_counts.get(f"fused_q{window}", 0) == 2
    for a, b in zip(stock, seg):
        assert _top_ulps(a, b) <= 4


@pytest.mark.skipif(not mx.metal.is_available() or not nax_available(), reason="TensorOps unavailable")
@pytest.mark.parametrize("cold_rows", [3000, 9000])
def test_no_gather_route_fires_in_a_cold_turn_or_in_follow_up_turns(lane, cold_rows, monkeypatch) -> None:
    """One path with the switch on: a cold prompt (below and above the packed threshold) and the
    follow-up turns after it never gather the history. A single-segment call runs the stock
    route over a zero-copy view (``single_q*``); several segments take the kernels."""
    import mtplx.segmented_kv as module

    monkeypatch.setenv("MTPLX_SEGMENTED_KV_PREFILL", "lse")  # the LSE route also runs (reference) without the MLX kernel
    module.route_counts.clear()
    cache = SegmentedKVCache()
    for seed, n in ((1, cold_rows // 2), (2, cold_rows - cold_rows // 2)):  # cold prefill, two chunks
        lane(_x(seed, n), mask="causal", cache=cache)
    for seed in (3, 4):
        lane(_x(seed, 4), mask="causal", cache=cache)  # cold verify windows
        cache.trim(2)
    lane(_x(5, 1), mask=None, cache=cache)  # cold decode
    mx.eval(cache.keys)
    assert not any(k.startswith("gather") for k in module.route_counts), module.route_counts
    assert any(k.startswith(("single_q", "fused_q")) for k in module.route_counts)
    for turn in range(3):  # follow-up turns on restored caches
        state = cache.state
        cache = SegmentedKVCache()
        cache.state = state
        lane(_x(10 + turn, 300), mask="causal", cache=cache)  # prefill over history
        for seed in (20 + turn, 30 + turn):
            lane(_x(seed, 4), mask="causal", cache=cache)
            cache.trim(2)
        lane(_x(40 + turn, 9), mask="causal", cache=cache)  # context-copy window
        cache.trim(5)
        lane(_x(50 + turn, 1), mask=None, cache=cache)
    mx.eval(cache.keys)
    assert not any(k.startswith("gather") for k in module.route_counts), module.route_counts
    assert module.route_counts.get("prefill_lse", 0) >= 1
