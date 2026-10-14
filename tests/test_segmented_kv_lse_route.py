"""Segmented KV on Macs without tensor units: the ``lse`` decode route (per segment attention
plus an exact logsumexp merge, ``MTPLX_SEGMENTED_KV_DECODE=lse``).

The route must equal the dense attention over the gathered rows (exactness), run through the real
attention layer like the stock cache (greedy parity: same argmax through a projection, same
routes, no gather) and leave the kernel route alone. Runs on any GPU, including an M5 that
rehearses the other Macs.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_map
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention

import mtplx.segmented_kv as module
from mtplx.attention_split import configure_split_full_attention
from mtplx.segmented_kv import SegmentedKVCache, decode_segments_attention


@pytest.fixture
def cpu():
    """The exactness tests run on the CPU: MLX float32 GEMM on a tensor-unit GPU rounds like TF32."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _lse_route(monkeypatch):
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_DECODE", "lse")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MAX_SEGMENTS", "1000")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_PREFILL", "lse")
    module.route_counts.clear()


def _dense(q, k, v, scale, n_rows):
    """Reference: float32 softmax attention, the new rows (last Q) causal over the whole history."""
    b, hq, q_len, d = q.shape
    hk = k.shape[1]
    k = mx.repeat(k.astype(mx.float32), hq // hk, axis=1)
    v = mx.repeat(v.astype(mx.float32), hq // hk, axis=1)
    scores = (q.astype(mx.float32) @ k.transpose(0, 1, 3, 2)) * scale
    visible = mx.arange(n_rows)[None, :] <= (n_rows - q_len + mx.arange(q_len))[:, None]
    scores = mx.where(visible[None, None], scores, -mx.inf)
    return (mx.softmax(scores, axis=-1) @ v).astype(q.dtype)


def _cache(hk, d, sizes, dtype, seed=0):
    """A segmented cache with sealed segments of ``sizes[:-1]`` rows and a tail of ``sizes[-1]``."""
    mx.random.seed(seed)
    cache = SegmentedKVCache()
    keys, values = [], []
    for index, n in enumerate(sizes):
        k = (mx.random.normal((1, hk, n, d)) * 1.5).astype(dtype)
        v = mx.random.normal((1, hk, n, d)).astype(dtype)
        cache.append_rows(k, v)
        keys.append(k)
        values.append(v)
        if index < len(sizes) - 1:
            cache.seal()
    return cache, mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2)


@pytest.mark.parametrize("dtype, tol", [(mx.float32, 2e-5), (mx.bfloat16, 2e-2)])
@pytest.mark.parametrize("hq, hk, d", [(24, 4, 256), (8, 2, 128), (4, 4, 64)])
@pytest.mark.parametrize("q_len", [1, 4, 9, 32])
@pytest.mark.parametrize("sizes", [[300, 40], [1000, 513, 70, 33]])
def test_the_lse_route_equals_the_dense_attention(cpu, dtype, tol, hq, hk, d, q_len, sizes) -> None:
    cache, k, v = _cache(hk, d, sizes, dtype)
    q = mx.random.normal((1, hq, q_len, d)).astype(dtype)
    scale = 1.0 / math.sqrt(d)
    out = decode_segments_attention(q, cache, scale=scale, mask="causal", packed_threshold=8192)
    assert out is not None and out.shape == q.shape and out.dtype == dtype
    expected = _dense(q, k, v, scale, sum(sizes))
    assert float(mx.abs(out.astype(mx.float32) - expected.astype(mx.float32)).max().item()) <= tol
    assert module.route_counts.get(f"lse_q{q_len}") == 1


def _differing(a, b) -> float:
    """Fraction of elements that differ between two bfloat16 arrays."""
    return float(mx.mean(a.astype(mx.float32) != b.astype(mx.float32)).item())


@pytest.mark.parametrize("q_len", [1, 4, 9])
@pytest.mark.parametrize("sizes", [[3000, 200], [2000, 1024, 600, 40]])
def test_the_lse_route_rounds_once_like_the_stock_attention(cpu, q_len, sizes) -> None:
    """bfloat16 is what the model runs: the route must be the float32 attention rounded once.

    A tolerance of one ulp does not see a second rounding (each segment rounded to bfloat16 before
    the merge), which puts a quarter or more of the elements one ulp off; on real Qwen3.8-27B
    tensors it showed as about one element in three differing from the stock kernel, against one
    in six for the exact result.
    """
    cache, k, v = _cache(4, 256, sizes, mx.bfloat16)
    q = mx.random.normal((1, 24, q_len, 256)).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(256)
    out = decode_segments_attention(q, cache, scale=scale, mask="causal", packed_threshold=8192)
    exact = _dense(q, k, v, scale, sum(sizes))
    assert out.dtype == mx.bfloat16
    assert _differing(out, exact) <= 0.005


@pytest.mark.parametrize("q_len", [1, 4])
def test_the_lse_route_is_as_close_to_stock_sdpa_as_the_exact_result(cpu, q_len) -> None:
    """Against the stock fused SDPA on the gathered rows, the route may differ only where the
    exact result itself differs from it (the stock kernel's own rounding)."""
    sizes = [3000, 1000, 200]
    cache, k, v = _cache(4, 256, sizes, mx.bfloat16)
    q = mx.random.normal((1, 24, q_len, 256)).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(256)
    out = decode_segments_attention(q, cache, scale=scale, mask="causal", packed_threshold=8192)
    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal" if q_len > 1 else None)
    exact = _dense(q, k, v, scale, sum(sizes))
    assert _differing(out, stock) <= _differing(exact, stock) + 0.005


def test_rows_without_a_visible_key_in_a_segment_merge_without_nan(cpu) -> None:
    """The merge ignores segments whose logsumexp is -inf (the shared -inf-safe merge): a tail shorter
    than the window masks its first rows entirely, and those rows take the earlier segments only."""
    cache, k, v = _cache(2, 64, [200, 2], mx.float32)
    q = mx.random.normal((1, 4, 4, 64))
    out = module.attend_segments_lse(q, cache, scale=0.125)
    assert bool(mx.all(mx.isfinite(out)).item())
    expected = mx.softmax((q[:, :, :2] @ mx.repeat(k[:, :, :200], 2, axis=1).transpose(0, 1, 3, 2)) * 0.125, axis=-1) @ mx.repeat(v[:, :, :200], 2, axis=1)
    assert float(mx.abs(out[:, :, :2] - expected).max().item()) <= 2e-5


def test_a_row_that_is_masked_everywhere_merges_to_zeros(cpu) -> None:
    out = module.merge_segment_outputs(
        [mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4))], [mx.full((1, 1, 2), -mx.inf), mx.full((1, 1, 2), -mx.inf)]
    )
    assert bool(mx.all(out == 0).item())


def test_a_single_segment_runs_the_stock_route(cpu) -> None:
    cache, _, _ = _cache(2, 64, [500], mx.float32)
    q = mx.random.normal((1, 4, 4, 64))
    assert decode_segments_attention(q, cache, scale=0.125, mask="causal", packed_threshold=8192) is None


def test_a_mask_or_an_oversized_window_goes_to_the_stock_ladder(cpu) -> None:
    cache, _, _ = _cache(2, 64, [500, 40], mx.float32)
    q = mx.random.normal((1, 4, 4, 64))
    mask = mx.ones((4, 540), dtype=mx.bool_)
    assert decode_segments_attention(q, cache, scale=0.125, mask=mask, packed_threshold=8192) is None
    wide = mx.random.normal((1, 4, 33, 64))
    assert decode_segments_attention(wide, cache, scale=0.125, mask="causal", packed_threshold=8192) is None


def test_more_than_the_fused_kernel_limit_of_segments_stays_on_the_route(cpu) -> None:
    sizes = [64] * 30 + [40]  # the kernel takes 12 per launch and 20 launches; here nothing limits it
    cache, k, v = _cache(2, 64, sizes, mx.float32)
    q = mx.random.normal((1, 4, 4, 64))
    out = decode_segments_attention(q, cache, scale=0.125, mask="causal", packed_threshold=8192)
    assert float(mx.abs(out - _dense(q, k, v, 0.125, sum(sizes))).max().item()) <= 2e-5


def test_the_kernel_is_never_touched_on_the_lse_route(cpu, monkeypatch) -> None:
    def boom(**_kwargs):
        raise AssertionError("segment kernel called")

    monkeypatch.setattr("mtplx.kernels.sdpa_segmented.sdpa_nax_flash_dsplit_segments", boom)
    cache, _, _ = _cache(2, 64, [300, 40], mx.float32)
    q = mx.random.normal((1, 4, 4, 64))
    assert decode_segments_attention(q, cache, scale=0.125, mask=None, packed_threshold=8192) is not None


# ---- through the real attention layer: greedy parity with the stock cache ---------------------

HIDDEN = 128
VOCAB = 512


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
    return SimpleNamespace(model=SimpleNamespace(layers=[layer])), attn


def _x(seed: int, n: int) -> mx.array:
    return (mx.random.normal((1, n, HIDDEN), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)


def _run_turns(attn, cache):
    outs = []
    for seed, n in [(1, 2304), (2, 2304)]:
        outs.append(attn(_x(seed, n), mask="causal", cache=cache))
    if isinstance(cache, SegmentedKVCache):
        state = cache.state
        cache = SegmentedKVCache()
        cache.state = state
    outs.append(attn(_x(3, 300), mask="causal", cache=cache))
    for seed in (4, 5, 6):
        outs.append(attn(_x(seed, 4), mask="causal", cache=cache))
        cache.trim(2)
    outs.append(attn(_x(7, 9), mask="causal", cache=cache))
    cache.trim(5)
    outs.append(attn(_x(8, 1), mask=None, cache=cache))
    mx.eval(outs)
    return outs


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the Metal GPU")
def test_the_attention_layer_agrees_with_the_stock_cache_and_never_gathers() -> None:
    model, attn = _model()
    configure_split_full_attention(model)
    proj = mx.random.normal((attn.o_proj.weight.shape[0], VOCAB), key=mx.random.key(9)).astype(mx.bfloat16)
    stock = _run_turns(attn, KVCache())
    module.route_counts.clear()
    seg = _run_turns(attn, SegmentedKVCache())
    assert not any(k.startswith("gather") for k in module.route_counts), module.route_counts
    assert module.route_counts.get("lse_q4", 0) >= 3 and module.route_counts.get("lse_q9", 0) >= 1
    decided_rows = total_rows = 0
    for a, b in zip(stock, seg):
        assert bool(mx.all(mx.isfinite(b)).item())
        # greedy parity: wherever the top-1 margin exceeds 4x the logit difference the cache can
        # explain, the argmax token through a fixed projection must be the same
        la, lb = a.astype(mx.float32) @ proj.astype(mx.float32), b.astype(mx.float32) @ proj.astype(mx.float32)
        top2 = mx.sort(la, axis=-1)[..., -2:]
        decided = (top2[..., 1] - top2[..., 0]) > 4 * mx.abs(la - lb).max(axis=-1)
        same = mx.argmax(la, axis=-1) == mx.argmax(lb, axis=-1)
        assert bool(mx.all(same | ~decided).item())
        decided_rows += int(decided.sum().item())
        total_rows += int(decided.size)
        top = float(mx.maximum(mx.abs(a).max(), mx.abs(b).max()).item())
        err = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item())
        assert err / 2.0 ** (math.floor(math.log2(top)) - 7) <= 8
    assert decided_rows > 0.5 * total_rows  # the check is not vacuous
