"""QSA block-sparse flash-skip attention parity (GPU: Metal).

Past the indexer engage threshold, MTPLX_QSA_FLASH must reproduce the
dense bool-mask lane's attention output over the identical visible set
(reduction-order bf16 noise only), reading the KV backing arrays in place.
Anti-vacuous kernel counter; dense regime must stay inactive.
"""

import mlx.core as mx
import pytest

from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs


@pytest.fixture()
def attn():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mx.random.seed(23)
    layer = Attention(TextArgs())
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def _run(layer, prefill, decodes):
    cache = QSACache(compress_ratio=layer.indexer.ratio)
    out_p = layer(prefill, cache)
    outs = [layer(d, cache) for d in decodes]
    mx.eval(out_p, *outs)
    return outs


def test_flash_parity_past_engage_threshold(attn, monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_FLASH_MIN_CONTEXT", "0")
    ratio = attn.indexer.ratio
    engage_t = attn.indexer.block_topk * ratio
    T0 = engage_t + 8 * ratio
    mx.random.seed(31)
    prefill = (mx.random.normal((1, T0, 2560)) * 0.3).astype(mx.bfloat16)
    decodes = [
        (mx.random.normal((1, 1, 2560)) * 0.3).astype(mx.bfloat16) for _ in range(3)
    ]

    monkeypatch.setenv("MTPLX_QSA_FLASH", "0")
    monkeypatch.setenv("MTPLX_QSA_GATHER", "0")
    ref = _run(attn, prefill, decodes)

    from mtplx.kernels import qsa_flash_skip as qfs

    calls = {"n": 0}
    orig = qfs.qsa_flash_skip

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(qfs, "qsa_flash_skip", counting)
    monkeypatch.setenv("MTPLX_QSA_FLASH", "1")
    got = _run(attn, prefill, decodes)
    assert calls["n"] == len(decodes), "flash kernel did not run — vacuous"

    for i, (r, g) in enumerate(zip(ref, got)):
        scale = mx.abs(r.astype(mx.float32)).max().item() + 1e-6
        err = (
            mx.abs(g.astype(mx.float32) - r.astype(mx.float32)) / scale
        ).max().item()
        assert err < 2e-2, f"decode {i} rel err {err}"


def test_flash_inactive_below_threshold(attn, monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_FLASH", "1")
    monkeypatch.setenv("MTPLX_QSA_FLASH_MIN_CONTEXT", "0")
    mx.random.seed(37)
    prefill = (mx.random.normal((1, 64, 2560)) * 0.3).astype(mx.bfloat16)
    step = (mx.random.normal((1, 1, 2560)) * 0.3).astype(mx.bfloat16)
    cache = QSACache(compress_ratio=attn.indexer.ratio)
    p = attn(prefill, cache)
    sel = attn.indexer(step, cache.offset, cache)
    assert sel is None
    mx.eval(p)


def test_flash_fence_defaults():
    from mtplx.models.qwen4_exp import _qsa_flash_min_context
    assert _qsa_flash_min_context() == 16384


def test_flash_stays_dense_below_min_context(attn, monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_FLASH", "1")
    monkeypatch.setenv("MTPLX_QSA_FLASH_MIN_CONTEXT", "16384")
    ratio = attn.indexer.ratio
    engage_t = attn.indexer.block_topk * ratio
    T0 = engage_t + 8 * ratio
    mx.random.seed(31)
    prefill = (mx.random.normal((1, T0, 2560)) * 0.3).astype(mx.bfloat16)
    decodes = [
        (mx.random.normal((1, 1, 2560)) * 0.3).astype(mx.bfloat16) for _ in range(3)
    ]
    cache = QSACache(compress_ratio=attn.indexer.ratio)
    p = attn(prefill, cache)
    mx.eval(p)
    sel = attn.indexer(decodes[0], cache.offset, cache)
    assert isinstance(sel, mx.array) and sel.ndim == 4, "must stay dense below 16k context"
