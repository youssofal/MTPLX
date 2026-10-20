"""Correctness gate for the wide-window dsplit flash kernel (GPU required).

``sdpa_nax_flash_dsplit_wide`` tiles the 6 x q_len query rows of a KV head in groups of
16 * TG_M rows inside one launch (grid.y), split in halves or (``MTPLX_NAX_FLASH_DSPLIT4=1``)
quarters of the head dimension. Compared against an fp32 tail-causal reference and the stock
MLX route.
"""

import math
import os

import mlx.core as mx
import pytest

from mtplx.kernels import sdpa_nax_flash_dsplit as module
from mtplx.kernels.sdpa_nax_flash_dsplit import (
    sdpa_nax_flash_dsplit,
    sdpa_nax_flash_dsplit_wide,
)
from mtplx.nax_verify import nax_available

HQ, HKV, D = 24, 4, 256

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or not nax_available() or module._nax_flash_dsplit_kernel() is None,
    reason="TensorOps kernel unavailable on this toolchain",
)


def _ref_tail_causal(q, k, v, offset, scale, q_len):
    qf = q.astype(mx.float32)
    kf = mx.repeat(k[:, :, :offset, :].astype(mx.float32), HQ // HKV, axis=1)
    vf = mx.repeat(v[:, :, :offset, :].astype(mx.float32), HQ // HKV, axis=1)
    s = (qf @ kf.transpose(0, 1, 3, 2)) * scale
    n = mx.arange(offset)[None, None, None, :]
    j = mx.arange(q_len)[None, None, :, None]
    s = mx.where(n <= (offset - q_len + j), s, mx.array(-1e30, dtype=s.dtype))
    return mx.softmax(s, axis=-1) @ vf


def _inputs(ctx, q_len, seed=7):
    mx.random.seed(seed)
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v, 1.0 / math.sqrt(D)


def _stock(q, k, v, offset, scale):
    return mx.fast.scaled_dot_product_attention(
        q, k[:, :, :offset], v[:, :, :offset], scale=scale, mask="causal"
    )


@pytest.mark.parametrize("ctx", [512, 2048, 1000, 4097])
@pytest.mark.parametrize("q_len", [1, 5, 6, 8, 9, 12, 13, 16, 17, 24, 25, 32])
@pytest.mark.parametrize("tgm", ["", "1", "2", "4"])
@pytest.mark.parametrize("dsplit4", ["0", "1"])
def test_wide_matches_reference(monkeypatch, ctx, q_len, tgm, dsplit4):
    monkeypatch.setenv("MTPLX_NAX_FLASH_WIDE_TGM", tgm)
    monkeypatch.setenv("MTPLX_NAX_FLASH_DSPLIT4", dsplit4)
    q, k, v, scale = _inputs(ctx, q_len)
    out = sdpa_nax_flash_dsplit_wide(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    assert out is not None, "wide kernel bailed on a supported shape"
    mx.eval(out)
    ref = _ref_tail_causal(q, k, v, ctx, scale, q_len)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    stock = _stock(q, k, v, ctx, scale)
    stock_err = mx.abs(stock.astype(mx.float32) - ref).max().item()
    # bf16 output rounding bounds both; the kernel must be inside the stock route's envelope.
    assert err <= max(2.5e-3, 1.5 * stock_err), (err, stock_err)


def test_wide_shape(monkeypatch):
    """Halves keep 32-row groups; quarters pair 16-row blocks only when none is left empty."""
    monkeypatch.delenv("MTPLX_NAX_FLASH_WIDE_TGM", raising=False)
    monkeypatch.setenv("MTPLX_NAX_FLASH_DSPLIT4", "0")
    assert [module.wide_shape(6 * q) for q in (6, 9, 13, 17, 25)] == [(2, 2)] * 5
    monkeypatch.setenv("MTPLX_NAX_FLASH_DSPLIT4", "1")
    # 36, 54, 78, 102, 150, 192 rows: 3, 4, 5, 7, 10, 12 blocks.
    assert [module.wide_shape(6 * q)[1] for q in (6, 9, 13, 17, 25, 32)] == [1, 2, 1, 1, 2, 2]
    assert module.wide_shape(6 * 9)[0] == 4
    assert module.wide_shape(6 * 9, quarters=False) == (2, 2)
    monkeypatch.setenv("MTPLX_NAX_FLASH_WIDE_TGM", "4")
    assert module.wide_shape(6 * 32) == (4, 2)  # the score exchange fits 8 simdgroups
    monkeypatch.setenv("MTPLX_NAX_FLASH_DSPLIT4", "0")
    assert module.wide_shape(6 * 32) == (2, 4)


@pytest.mark.parametrize("ctx", [1000, 4097, 20000])
@pytest.mark.parametrize("q_len", [6, 7, 8, 9, 13, 17, 25, 32])
def test_wide_quarters_within_halves_envelope_of_stock(monkeypatch, ctx, q_len):
    """Parity: the quarter split is no further from stock SDPA than the halves route,
    up to one bf16 ulp of the output's magnitude (four partial score sums instead of two)."""
    monkeypatch.delenv("MTPLX_NAX_FLASH_WIDE_TGM", raising=False)
    q, k, v, scale = _inputs(ctx, q_len, seed=ctx + q_len)
    stock = _stock(q, k, v, ctx, scale).astype(mx.float32)
    errs = {}
    for dsplit4 in ("0", "1"):
        monkeypatch.setenv("MTPLX_NAX_FLASH_DSPLIT4", dsplit4)
        out = sdpa_nax_flash_dsplit_wide(queries=q, keys=k, values=v, offset=ctx, scale=scale)
        assert out is not None
        errs[dsplit4] = mx.abs(out.astype(mx.float32) - stock).max().item()
    top = mx.abs(stock).max().item()
    ulp = 2.0 ** (math.floor(math.log2(top)) - 7)
    assert errs["1"] <= errs["0"] + ulp, errs


def test_wide_blocks_and_tensor_offset(monkeypatch):
    """A traced (array) offset and several split-KV block counts give the same answer."""
    ctx, q_len = 3000, 16
    q, k, v, scale = _inputs(ctx, q_len)
    ref = _ref_tail_causal(q, k, v, ctx, scale, q_len)
    for blocks in (32, 64, 128):
        monkeypatch.setenv("MTPLX_NAX_FLASH_DSPLIT_BLOCKS", str(blocks))
        out = sdpa_nax_flash_dsplit_wide(
            queries=q, keys=k, values=v, offset=mx.array([ctx], dtype=mx.int32), scale=scale
        )
        mx.eval(out)
        assert mx.abs(out.astype(mx.float32) - ref).max().item() < 2.5e-3


def test_wide_contract_bails():
    q, k, v, scale = _inputs(512, 33)
    before = dict(module.nax_flash_dsplit_bail_counts)
    assert sdpa_nax_flash_dsplit_wide(queries=q, keys=k, values=v, offset=512, scale=scale) is None
    assert module.nax_flash_dsplit_bail_counts.get("m_rows_gt_wide_max", 0) == before.get(
        "m_rows_gt_wide_max", 0
    ) + 1


def test_shipping_dsplit_contract_unchanged():
    """The <= 32 row contract still bails above it and matches the reference at q_len 5."""
    q, k, v, scale = _inputs(1024, 6)
    assert sdpa_nax_flash_dsplit(queries=q, keys=k, values=v, offset=1024, scale=scale) is None
    q, k, v, scale = _inputs(1024, 5)
    out = sdpa_nax_flash_dsplit(queries=q, keys=k, values=v, offset=1024, scale=scale)
    mx.eval(out)
    ref = _ref_tail_causal(q, k, v, 1024, scale, 5)
    assert mx.abs(out.astype(mx.float32) - ref).max().item() < 2.5e-3
    assert os.environ.get("MTPLX_NAX_FLASH_DSPLIT", "1") != "0"
