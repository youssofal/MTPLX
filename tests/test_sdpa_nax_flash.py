"""Correctness gate for the TensorOps flash-decoding SDPA kernel (GPU required).

Compares sdpa_nax_flash against the production packed kernel and a fp32 reference at
real Qwen3.8-27B decode shapes, across the key-split / block geometries the router can
pick. Tolerances match the packed kernel's own acceptance (bf16 output, fp32 accumulate).
"""

import math
import os

import mlx.core as mx
import pytest

from mtplx.kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail
from mtplx.kernels.sdpa_nax_flash import _nax_flash_kernel, sdpa_nax_flash
from mtplx.kernels.sdpa_nax_flash_dsplit import sdpa_nax_flash_dsplit
from mtplx.nax_verify import nax_available

HQ, HKV, D = 24, 4, 256

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or not nax_available() or _nax_flash_kernel() is None,
    reason="TensorOps kernel unavailable on this toolchain",
)


def _ref_tail_causal(q, k, v, offset, scale, q_len):
    """fp32 reference: row j attends to n <= offset - q_len + j."""
    qf = q.astype(mx.float32)
    kf = k[:, :, :offset, :].astype(mx.float32)
    vf = v[:, :, :offset, :].astype(mx.float32)
    gqa = q.shape[1] // k.shape[1]
    kf = mx.repeat(kf, gqa, axis=1)
    vf = mx.repeat(vf, gqa, axis=1)
    s = (qf @ kf.transpose(0, 1, 3, 2)) * scale
    n = mx.arange(offset)[None, None, None, :]
    j = mx.arange(q_len)[None, None, :, None]
    mask = n <= (offset - q_len + j)
    s = mx.where(mask, s, mx.array(-1e30, dtype=s.dtype))
    p = mx.softmax(s, axis=-1)
    return p @ vf


def _run(ctx, q_len, ks, blocks, qtg=0, seed=7):
    mx.random.seed(seed)
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)
    env = {"MTPLX_NAX_FLASH_KS": str(ks), "MTPLX_NAX_FLASH_BLOCKS": str(blocks),
           "MTPLX_NAX_FLASH_QTG": str(qtg)}
    old = {kk: os.environ.get(kk) for kk in env}
    os.environ.update(env)
    try:
        out = sdpa_nax_flash(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    finally:
        for kk, vv in old.items():
            if vv is None:
                os.environ.pop(kk, None)
            else:
                os.environ[kk] = vv
    return q, k, v, scale, out


@pytest.mark.parametrize("ctx,q_len", [
    (512, 4), (2048, 4), (4096, 4), (2048, 2), (2048, 5),
    (1023, 4),   # non-tile-aligned context
    (1000, 3),   # non-aligned + odd width
    (2048, 6),   # M=36 (3 simdgroups per key split)
    (2048, 8),   # wide-QL M=48
    (2048, 9),   # wide-QL M=54 (4 simdgroups, padded)
])
@pytest.mark.parametrize("ks,blocks", [(4, 32), (1, 64), (2, 32), (8, 32)])
def test_nax_flash_matches_reference(ctx, q_len, ks, blocks):
    q, k, v, scale, out = _run(ctx, q_len, ks, blocks)
    if HQ // HKV * q_len > 64:
        assert out is None, "M>64 must bail to fallback"
        return
    assert out is not None, "kernel bailed on a supported shape"
    mx.eval(out)

    ref = _ref_tail_causal(q, k, v, ctx, scale, q_len)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()

    packed = sdpa_gqa_packed_tail(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    if packed is not None:
        mx.eval(packed)
        perr = mx.abs(packed.astype(mx.float32) - ref).max().item()
        # Must not be materially worse than the shipped kernel's own bf16
        # deviation from the fp32 reference.
        assert err <= max(2.5 * perr, 0.02), (err, perr)
    assert err < 0.05, err


def test_nax_flash_staged_queries_match():
    q, k, v, scale, out0 = _run(3000, 4, 4, 32, qtg=0)
    _, _, _, _, out1 = _run(3000, 4, 4, 32, qtg=1)
    assert out0 is not None and out1 is not None
    mx.eval(out0, out1)
    ref = _ref_tail_causal(q, k, v, 3000, scale, 4)
    e0 = mx.abs(out0.astype(mx.float32) - ref).max().item()
    e1 = mx.abs(out1.astype(mx.float32) - ref).max().item()
    assert e0 < 0.05 and e1 < 0.05, (e0, e1)


def test_nax_flash_array_offset_matches_int_offset():
    """Compiled graphs hand the kernel an int32 offset array; same result."""
    mx.random.seed(11)
    ctx, q_len = 1500, 4
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)
    a = sdpa_nax_flash(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    b = sdpa_nax_flash(queries=q, keys=k, values=v,
                       offset=mx.array([ctx], dtype=mx.int32), scale=scale)
    assert a is not None and b is not None
    mx.eval(a, b)
    assert mx.array_equal(a, b).item()


def test_nax_flash_empty_and_bail_paths():
    mx.random.seed(3)
    q = (mx.random.normal((1, HQ, 4, D))).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, 256, D))).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, 256, D))).astype(mx.bfloat16)
    mx.eval(q, k, v)
    assert sdpa_nax_flash(queries=q, keys=k, values=v, offset=0,
                          scale=0.0625) is None      # offset_range
    assert sdpa_nax_flash(queries=q.astype(mx.float32), keys=k, values=v,
                          offset=128, scale=0.0625) is None  # dtype gate
    os.environ["MTPLX_NAX_FLASH"] = "0"
    try:
        assert sdpa_nax_flash(queries=q, keys=k, values=v, offset=128,
                              scale=0.0625) is None  # kill switch
    finally:
        os.environ.pop("MTPLX_NAX_FLASH", None)


@pytest.mark.parametrize("ctx,q_len", [
    (512, 4), (2048, 4), (4096, 4), (2048, 2), (2048, 5), (1023, 4), (1000, 3),
    (2048, 6),   # M=36 > 32: must bail
])
@pytest.mark.parametrize("blocks", [32, 64, 128])
def test_nax_flash_dsplit_matches_reference(ctx, q_len, blocks):
    mx.random.seed(11)
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)
    os.environ["MTPLX_NAX_FLASH_DSPLIT_BLOCKS"] = str(blocks)
    try:
        out = sdpa_nax_flash_dsplit(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    finally:
        os.environ.pop("MTPLX_NAX_FLASH_DSPLIT_BLOCKS", None)
    if HQ // HKV * q_len > 32:
        assert out is None, "M>32 must bail to the wide kernel"
        return
    assert out is not None, "dsplit kernel bailed on a supported shape"
    mx.eval(out)
    ref = _ref_tail_causal(q, k, v, ctx, scale, q_len)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    packed = sdpa_gqa_packed_tail(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    if packed is not None:
        mx.eval(packed)
        perr = mx.abs(packed.astype(mx.float32) - ref).max().item()
        assert err <= max(2.5 * perr, 0.02), (err, perr)
    assert err < 0.05, err


@pytest.mark.parametrize("ctx,q_len", [
    (512, 2), (2048, 3), (2048, 4), (4096, 4), (1023, 4), (1000, 5), (3000, 2),
])
@pytest.mark.parametrize("blocks", [32, 64, 128])
def test_nax_flash_dsplit_quarter_split_matches_reference(ctx, q_len, blocks):
    """attn4: NDH=4 (MTPLX_NAX_FLASH_DSPLIT4) stays inside the shipping envelope for q_len 2..5."""
    mx.random.seed(11)
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)
    os.environ["MTPLX_NAX_FLASH_DSPLIT_BLOCKS"] = str(blocks)
    try:
        out4 = sdpa_nax_flash_dsplit(queries=q, keys=k, values=v, offset=ctx, scale=scale, ndh=4)
        out2 = sdpa_nax_flash_dsplit(queries=q, keys=k, values=v, offset=ctx, scale=scale, ndh=2)
    finally:
        os.environ.pop("MTPLX_NAX_FLASH_DSPLIT_BLOCKS", None)
    assert out4 is not None and out2 is not None
    mx.eval(out4, out2)
    ref = _ref_tail_causal(q, k, v, ctx, scale, q_len)
    e4 = mx.abs(out4.astype(mx.float32) - ref).max().item()
    e2 = mx.abs(out2.astype(mx.float32) - ref).max().item()
    assert e4 <= max(2.5 * e2, 0.02), (e4, e2)
    assert e4 < 0.05, e4


def test_nax_flash_dsplit_quarter_split_switch():
    """Switch off (default) is the 2-way kernel bit for bit; switch on is the 4-way kernel."""
    mx.random.seed(21)
    ctx, q_len = 3000, 4
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    kw = {"queries": q, "keys": k, "values": v, "offset": ctx, "scale": 1.0 / math.sqrt(D)}
    os.environ.pop("MTPLX_NAX_FLASH_DSPLIT4", None)
    try:
        off = sdpa_nax_flash_dsplit(**kw)
        two = sdpa_nax_flash_dsplit(**kw, ndh=2)
        os.environ["MTPLX_NAX_FLASH_DSPLIT4"] = "1"
        on = sdpa_nax_flash_dsplit(**kw)
        four = sdpa_nax_flash_dsplit(**kw, ndh=4)
    finally:
        os.environ.pop("MTPLX_NAX_FLASH_DSPLIT4", None)
    assert all(x is not None for x in (off, two, on, four))
    mx.eval(off, two, on, four)
    assert mx.array_equal(off, two).item()
    assert mx.array_equal(on, four).item()
    assert sdpa_nax_flash_dsplit(**kw, ndh=3) is None
