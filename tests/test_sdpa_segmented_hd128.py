"""Segmented verify kernel at head_dim 128 (Qwen3-8B, Llama-like shapes: 32 query / 8 KV heads, GQA 4).

The kernel tests need a TensorOps GPU (pending on a machine on power); the support-contract and the
lse-fallback tests are CPU-safe.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

import mtplx.segmented_kv as module
from mtplx.kernels import sdpa_segmented as S
from mtplx.nax_verify import nax_available

HQ, HK, D = 32, 8, 128
SCALE = 1.0 / math.sqrt(D)

needs_nax = pytest.mark.skipif(
    not mx.metal.is_available() or not nax_available(),
    reason="TensorOps kernel unavailable on this machine",
)


def _data(total, q_len, peaked=False, seed=3):
    mx.random.seed(seed)
    q = (mx.random.normal((1, HQ, q_len, D)) * (2.5 if peaked else 0.5)).astype(mx.bfloat16)
    k = (mx.random.normal((1, HK, total + 64, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HK, total + 64, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


def _split(k, v, lens, spare=0):
    segs, a = [], 0
    for n in lens:
        kk, vv = mx.contiguous(k[:, :, a : a + n + spare]), mx.contiguous(v[:, :, a : a + n + spare])
        mx.eval(kk, vv)
        segs.append((kk, vv, n))
        a += n
    return segs


def _lens(total, nseg):
    lens = [total // nseg] * nseg
    lens[-1] += total - sum(lens)
    return lens


def _ref(q, k, v, total):
    q_len = q.shape[2]
    kf = mx.repeat(k[:, :, :total].astype(mx.float32), HQ // HK, axis=1)
    vf = mx.repeat(v[:, :, :total].astype(mx.float32), HQ // HK, axis=1)
    s = (q.astype(mx.float32) @ kf.transpose(0, 1, 3, 2)) * SCALE
    vis = mx.arange(total)[None, :] <= (total - q_len + mx.arange(q_len))[:, None]
    return mx.softmax(mx.where(vis[None, None], s, -mx.inf), axis=-1) @ vf


def test_the_support_contract_names_head_dim_128_and_nothing_below() -> None:
    assert S.segments_supported(4, 4, 128) and S.segments_supported(1, 4, 256)
    assert not S.segments_supported(4, 4, 64) and not S.segments_supported(4, 4, 96)
    assert S.SUPPORTED_HEAD_DIMS == (128, 256)


@needs_nax
@pytest.mark.parametrize("nseg", [1, 2, 6, 13])
@pytest.mark.parametrize("q_len", [1, 4, 5])
@pytest.mark.parametrize("peaked", [False, True])
def test_segments_match_the_fp32_reference_as_well_as_the_stock_sdpa_does(nseg, q_len, peaked) -> None:
    total = 12000
    q, k, v = _data(total, q_len, peaked)
    out = S.sdpa_nax_flash_dsplit_segments(queries=q, segments=_split(k, v, _lens(total, nseg)), scale=SCALE)
    assert out is not None and bool(mx.all(mx.isfinite(out)).item())
    ref = _ref(q, k, v, total)
    err = float(mx.abs(out.astype(mx.float32) - ref).max())
    mask = (mx.arange(total)[None, :] <= (total - q_len + mx.arange(q_len))[:, None])
    stock = mx.fast.scaled_dot_product_attention(
        q, k[:, :, :total], v[:, :, :total], scale=SCALE, mask=mask[None, None]
    )
    err_stock = float(mx.abs(stock.astype(mx.float32) - ref).max())
    assert err <= max(2.5 * err_stock, 1e-3), (err, err_stock)


@needs_nax
def test_head_dim_128_one_segment_does_not_go_through_the_contiguous_dispatcher() -> None:
    q, k, v = _data(9000, 4)
    before = dict(S._dsplit.nax_flash_dsplit_dispatch_counts)
    out = S.sdpa_nax_flash_dsplit_segments(queries=q, segments=[(k, v, 9000)], scale=SCALE)
    assert out is not None
    assert dict(S._dsplit.nax_flash_dsplit_dispatch_counts) == before  # the contiguous kernel stays 256 only
    assert float(mx.abs(out.astype(mx.float32) - _ref(q, k, v, 9000)).max()) < 5e-3


@needs_nax
def test_a_reference_shorter_than_its_buffer_reads_only_its_rows() -> None:
    total, q_len = 9000, 4
    q, k, v = _data(total, q_len)
    lens = _lens(total, 3)
    segs = _split(k, v, lens)
    junk = mx.ones((1, HK, 300, D), mx.bfloat16) * 9
    segs[0] = (mx.concatenate([segs[0][0], junk], axis=2), mx.concatenate([segs[0][1], junk], axis=2), lens[0])
    out = S.sdpa_nax_flash_dsplit_segments(queries=q, segments=segs, scale=SCALE)
    assert float(mx.abs(out.astype(mx.float32) - _ref(q, k, v, total)).max()) < 5e-3


def test_prefill_lse_falls_back_to_the_reference_route_when_the_kernel_refuses_a_shape(monkeypatch) -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        monkeypatch.setattr(module, "_SDPA_LSE", True)
        module.route_counts.clear()

        def refuses(*a, **k):
            raise ValueError("unsupported head dim for return_lse")

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", refuses)
        monkeypatch.setattr(mx.fast, module.LSE_API_FUNCTION, refuses, raising=False)
        q = mx.random.normal((1, 4, 6, 8)).astype(mx.bfloat16)
        keys = mx.random.normal((1, 2, 40, 8)).astype(mx.bfloat16)
        out, lse = module.segment_sdpa_lse(q, keys, keys, 40, scale=0.35, causal=False)
        assert out.shape == (1, 4, 6, 8) and lse.shape == (1, 4, 6)
        assert module.route_counts.get("sdpa_lse_kernel_unsupported") == 1
        assert module.route_counts.get("sdpa_lse_reference") == 1
    finally:
        mx.set_default_device(previous)


@needs_nax
def test_attention_layer_at_head_dim_128_matches_the_stock_cache_through_turns_and_rollback(monkeypatch) -> None:
    """Qwen3-style attention (32 query / 8 KV heads, head_dim 128) through the attention_split ladder."""
    from types import SimpleNamespace

    from mlx.utils import tree_map
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_next import Qwen3NextAttention

    from mtplx.attention_split import configure_split_full_attention
    from mtplx.segmented_kv import SegmentedKVCache

    for name in ("MTPLX_GQA_PACKED_SDPA", "MTPLX_NAX_FLASH_ROUTE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    hidden = 128
    args = SimpleNamespace(
        hidden_size=hidden, num_attention_heads=HQ, num_key_value_heads=HK, head_dim=D, attention_bias=False,
        rms_norm_eps=1e-6, partial_rotary_factor=0.25, rope_theta=10000.0, rope_scaling=None,
        max_position_embeddings=65536,
    )
    mx.random.seed(0)
    attn = Qwen3NextAttention(args)
    attn.update(tree_map(lambda p: p.astype(mx.bfloat16), attn.parameters()))
    mx.eval(attn.parameters())
    configure_split_full_attention(SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(is_linear=False, self_attn=attn)])))

    def x(seed, n):
        return (mx.random.normal((1, n, hidden), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)

    def run(cache):
        outs = [attn(x(1, 9000), mask="causal", cache=cache)]
        if isinstance(cache, SegmentedKVCache):
            state = cache.state
            cache = SegmentedKVCache()
            cache.state = state
        outs.append(attn(x(2, 300), mask="causal", cache=cache))
        for seed, window in ((4, 4), (5, 4), (6, 9), (7, 1)):
            outs.append(attn(x(seed, window), mask="causal" if window > 1 else None, cache=cache))
            cache.trim(max(1, window - 2))
        mx.eval(outs)
        return outs, cache

    stock, _ = run(KVCache())
    module.route_counts.clear()
    seg, cache = run(SegmentedKVCache())
    assert cache.segment_count >= 2
    assert not any(k.startswith("gather_q") and k[8:].isdigit() for k in module.route_counts), module.route_counts
    for a, b in zip(stock, seg):
        top = float(mx.maximum(mx.abs(a).max(), mx.abs(b).max()).item())
        ulp = 2.0 ** (math.floor(math.log2(top)) - 7)
        assert float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()) / ulp <= 4


@needs_nax
def test_llama_style_attention_at_head_dim_128_matches_the_stock_cache_through_turns_and_rollback(monkeypatch) -> None:
    """Mistral-7B shapes (32 query / 8 KV heads, head_dim 128, no q/k norm) through the attention_split ladder."""
    from types import SimpleNamespace

    from mlx.utils import tree_map
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.llama import Attention, ModelArgs

    from mtplx.attention_split import configure_split_full_attention
    from mtplx.segmented_kv import SegmentedKVCache

    for name in ("MTPLX_GQA_PACKED_SDPA", "MTPLX_NAX_FLASH_ROUTE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    hidden = 256
    args = ModelArgs(
        model_type="mistral", hidden_size=hidden, num_hidden_layers=1, intermediate_size=64, num_attention_heads=HQ,
        rms_norm_eps=1e-5, vocab_size=64, head_dim=D, num_key_value_heads=HK, rope_theta=1e6, tie_word_embeddings=False,
    )
    mx.random.seed(0)
    attn = Attention(args)
    attn.update(tree_map(lambda p: p.astype(mx.bfloat16), attn.parameters()))
    mx.eval(attn.parameters())
    configure_split_full_attention(SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])))

    def x(seed, n):
        return (mx.random.normal((1, n, hidden), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)

    def run(cache):
        outs = [attn(x(1, 9000), mask="causal", cache=cache)]
        if isinstance(cache, SegmentedKVCache):
            state = cache.state
            cache = SegmentedKVCache()
            cache.state = state
        outs.append(attn(x(2, 300), mask="causal", cache=cache))
        for seed, window in ((4, 4), (5, 4), (6, 9), (7, 1)):
            outs.append(attn(x(seed, window), mask="causal" if window > 1 else None, cache=cache))
            cache.trim(max(1, window - 2))
        mx.eval(outs)
        return outs, cache

    stock, _ = run(KVCache())
    module.route_counts.clear()
    seg, cache = run(SegmentedKVCache())
    assert cache.segment_count >= 2
    assert not any(k.startswith("gather_q") and k[8:].isdigit() for k in module.route_counts), module.route_counts
    assert any(k.startswith("fused_q") for k in module.route_counts), module.route_counts
    for a, b in zip(stock, seg):
        top = float(mx.maximum(mx.abs(a).max(), mx.abs(b).max()).item())
        ulp = 2.0 ** (math.floor(math.log2(top)) - 7)
        assert float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()) / ulp <= 4


def _low_gqa_cache(hk: int, d: int, window: int, seed: int = 11):
    """Two sealed segments plus a tail whose last ``window`` rows are the verify window."""
    from mtplx.segmented_kv import SegmentedKVCache

    cache = SegmentedKVCache()
    for i, n in enumerate((5000, 700, 40 + window)):
        k = (mx.random.normal((1, hk, n, d), key=mx.random.key(seed + i)) * 0.5).astype(mx.bfloat16)
        v = (mx.random.normal((1, hk, n, d), key=mx.random.key(seed + 50 + i)) * 0.5).astype(mx.bfloat16)
        cache.append_rows(k, v)
        if i < 2:
            cache.seal()
    return cache


@pytest.mark.parametrize("gqa, expected_sub", [(1, 10), (2, 10), (4, 8), (6, 5), (8, 4)])
@pytest.mark.parametrize("window", [11, 12, 17, 32])
def test_sub_windows_respect_both_kernel_limits_at_every_gqa(monkeypatch, gqa, expected_sub, window) -> None:
    """GQA 1 and 2 used sub-windows of 32 and 16 rows, which the kernel refuses (q_len <= 10): the
    window then fell back to a gather. Every sub-window must be one the kernel takes."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        monkeypatch.setenv("MTPLX_NAX_FLASH_ROUTE", "1")
        hk, d = 2, 128
        cache = _low_gqa_cache(hk, d, window)
        rows: list[int] = []

        def fake_kernel(*, queries, segments, scale):
            q_len = int(queries.shape[2])
            assert S.segments_supported(q_len, gqa, d), (q_len, gqa)
            rows.append(q_len)
            return mx.zeros(queries.shape, queries.dtype)

        monkeypatch.setattr(S, "sdpa_nax_flash_dsplit_segments", fake_kernel)
        monkeypatch.setattr("mtplx.kernel_selfcheck.lane_disabled", lambda name: False)
        q = mx.zeros((1, hk * gqa, window, d), mx.bfloat16)
        out = module.decode_segments_attention(q, cache, scale=0.1, mask="causal", packed_threshold=0)
        assert out is not None and tuple(out.shape) == tuple(q.shape)
        assert sum(rows) == window and max(rows) <= expected_sub
        if window > expected_sub:
            assert max(rows) == expected_sub
    finally:
        mx.set_default_device(previous)


@needs_nax
@pytest.mark.parametrize("gqa", [1, 2])
@pytest.mark.parametrize("window", [12, 32])
def test_low_gqa_wide_windows_run_on_the_segments_and_match_the_reference(monkeypatch, gqa, window) -> None:
    """GQA 1 and 2 (4 and 8 query heads over 4 KV heads, head_dim 128) with a window above the
    kernel's 10 rows: sub-windows over the segments, no gather, exact against fp32."""
    monkeypatch.setenv("MTPLX_NAX_FLASH_ROUTE", "1")
    hk, d = 4, 128
    cache = _low_gqa_cache(hk, d, window)
    hq = hk * gqa
    q = (mx.random.normal((1, hq, window, d), key=mx.random.key(99)) * 0.5).astype(mx.bfloat16)
    module.route_counts.clear()
    out = module.decode_segments_attention(q, cache, scale=1.0 / math.sqrt(d), mask="causal", packed_threshold=0)
    assert out is not None
    assert module.route_counts.get(f"chunked_q{window}") == 1, module.route_counts
    keys, values = cache.gather()
    total = int(keys.shape[2])
    kf = mx.repeat(keys.astype(mx.float32), gqa, axis=1)
    vf = mx.repeat(values.astype(mx.float32), gqa, axis=1)
    s = (q.astype(mx.float32) @ kf.transpose(0, 1, 3, 2)) / math.sqrt(d)
    vis = mx.arange(total)[None, :] <= (total - window + mx.arange(window))[:, None]
    ref = mx.softmax(mx.where(vis[None, None], s, -mx.inf), axis=-1) @ vf
    assert float(mx.abs(out.astype(mx.float32) - ref).max()) < 5e-3
