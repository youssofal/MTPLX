"""The stock commit's pieces in fewer launches give the stock operations' bytes (lane L7).

``visit_tails``: one launch for a visit's four convolution tails, against the stock commit's
in-place slice copies (kept rows at least the tail) and concatenations (fewer). ``mix``: one
launch for the drafter's four-tap weighted sum, against ``Drafter.mix``'s own operations
(float32 products summed left to right from 0, then MLX's RMS norm), bit for bit, with values
chosen so that a product contracted into the sum would change the bits.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.inkling.fast import commit_launches as cl


def _device():
    try:
        return mx.metal.is_available()
    except Exception:  # noqa: BLE001
        return False


needs_gpu = pytest.mark.skipif(not _device(), reason="needs a Metal GPU")


def _bits(a):
    a = np.asarray(a.view(mx.uint16) if a.dtype in (mx.bfloat16, mx.float16) else a.view(mx.uint32))
    return a


def _stock_tails(tails, rows, kept):
    out = []
    keep = tails[0].shape[1]
    for tail, r in zip(tails, rows):
        if kept >= keep:
            t = mx.array(tail)
            t[:, :, :] = r[:, kept - keep : kept]
            out.append(t)
        else:
            out.append(mx.concatenate([tail[:, kept:], r[:, :kept]], axis=1))
    return out


@needs_gpu
@pytest.mark.parametrize("kept", [1, 2, 3, 4, 8])
def test_visit_tails_equal_the_stock_commit(kept):
    key = mx.random.key(3 + kept)
    widths = (1024, 1024, 2304, 2304)
    rows_n = 8
    ks = mx.random.split(key, 8)
    tails = [mx.random.normal((1, 3, w), key=ks[i]).astype(mx.bfloat16) for i, w in enumerate(widths)]
    rows = [mx.random.normal((1, rows_n, w), key=ks[4 + i]).astype(mx.bfloat16) for i, w in enumerate(widths)]
    mx.eval(tails, rows)
    want = _stock_tails(tails, rows, kept)
    got = cl.visit_tails(tails, rows, kept)
    mx.eval(want, got)
    for a, b in zip(want, got):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(_bits(a), _bits(b))


def _stock_mix(features, blend_row, weight, eps):
    dtype = features[0].dtype
    mixed = sum(h.astype(mx.float32) * blend_row[i].astype(mx.float32) for i, h in enumerate(features))
    return mx.fast.rms_norm(mixed, weight, eps).astype(dtype)


@needs_gpu
@pytest.mark.parametrize("rows", [1, 3, 8])
def test_mix_equals_the_drafters_mix_bit_for_bit(rows):
    key = mx.random.key(11 + rows)
    hidden = 2304
    ks = mx.random.split(key, 6)
    # wide dynamic range, so a fused multiply-add would round differently from multiply then add
    features = [(mx.random.normal((1, rows, hidden), key=ks[i]) * (10.0 ** (i - 1))).astype(mx.bfloat16)
                for i in range(4)]
    features[2] = mx.where(mx.arange(hidden) % 97 == 0, mx.array(-0.0, mx.bfloat16), features[2])
    blend = mx.softmax(mx.random.normal((4, 4), key=ks[4]).astype(mx.float32), axis=-1).astype(mx.bfloat16)
    weight = (1.0 + 0.1 * mx.random.normal((hidden,), key=ks[5])).astype(mx.bfloat16)
    mx.eval(features, blend, weight)
    for row in range(4):
        want = _stock_mix(features, blend[row], weight, 1e-6)
        got = cl.mix(features, blend[row], weight, 1e-6)
        mx.eval(want, got)
        assert np.array_equal(_bits(want), _bits(got)), row
        # and the float32 sum itself, before the norm
        sw = sum(h.astype(mx.float32) * blend[row][i].astype(mx.float32) for i, h in enumerate(features))
        sg = cl._mix_kernel()(inputs=[*features, blend[row]], template=[("N", rows * hidden)],
                              grid=(rows * hidden, 1, 1), threadgroup=(256, 1, 1),
                              output_shapes=[features[0].shape], output_dtypes=[mx.float32])[0]
        mx.eval(sw, sg)
        assert np.array_equal(_bits(sw), _bits(sg)), row


def test_shapes_are_refused():
    with pytest.raises(ValueError):
        cl.visit_tails([mx.zeros((1, 3, 4))] * 3, [mx.zeros((1, 8, 4))] * 3, 1)
    with pytest.raises(ValueError):
        cl.mix([mx.zeros((1, 1, 4))] * 3, mx.zeros((3,)), mx.ones((4,)), 1e-6)
