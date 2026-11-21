"""Fast attention core and output (lane L2): bits against RC6's functions, row stability.

GPU tests on synthetic operands at the model's shapes (24 query heads over 8 key/value
heads of 128, extent 1,024). Each case builds a key/value history, hands the committed
part to the fast kernels in the layout of SIGNATURES.md (window ring in physical order
with ``newest``/``count``, or a global bank) plus the panel's own rows, and compares:

* ``attend`` with RC6's ``stock_attention`` per row on that row's chronological keys
  followed by RC6's sigmoid gate (exactly what RC6 runs for decode and for verify);
* ``attention_output`` with the one-row product per row, ``short_conv.reference`` and the
  residual add (or the boundary form);
* every row of a panel with an independent one-row call on the state with the earlier
  panel rows committed.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the fast attention kernels need a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import short_conv  # noqa: E402
from mtplx.models.inkling.attention import stock_attention  # noqa: E402
from mtplx.models.inkling.fast.attend import attend, reduction_plan  # noqa: E402
from mtplx.models.inkling.fast.attn_output import attention_output  # noqa: E402


def _clipped(shape, limit, key):
    v = mx.random.normal(shape, key=key) * 2.2
    n = mx.linalg.norm(v, axis=-1, keepdims=True)
    return (v * mx.minimum(1.0, limit / (n + 1e-6))).astype(mx.bfloat16)


def _setup(length, rows, seed):
    keys = mx.random.split(mx.random.key(seed), 12)
    k = _clipped((1, 8, length, 128), 22.6, keys[0])
    v = mx.fast.rms_norm(mx.random.normal((1, 8, length, 128), key=keys[1]), None, 1e-6).astype(mx.bfloat16)
    q = _clipped((1, rows, 24, 128), 22.6, keys[2])
    rel = _clipped((1, rows, 24, 16), 8.0, keys[3])
    sink = (mx.random.normal((1, rows, 24), key=keys[4]) * 3.0).astype(mx.bfloat16)
    gate = (mx.random.normal((1, rows, 3072), key=keys[5]) * 2.0).astype(mx.bfloat16)
    x = (mx.random.normal((1, rows, 2304), key=keys[6]) * 1.5).astype(mx.bfloat16)
    tail = (mx.random.normal((1, 3, 2304), key=keys[7]) * 0.8).astype(mx.bfloat16).astype(mx.float32)
    basis = (mx.random.normal((16, 1024), key=keys[8]) * 0.3).astype(mx.bfloat16)
    out = (mx.random.normal((2304, 3072), key=keys[9]) * 0.02).astype(mx.bfloat16)
    conv = (mx.random.normal((2304, 1, 4), key=keys[10]) * 0.5).astype(mx.bfloat16)[:, 0, :].T.astype(mx.float32)
    mx.eval(k, v, q, rel, sink, gate, x, tail, basis, out, conv)
    return k, v, (q, rel, sink, gate, x, tail), (basis, out, conv)


def _committed(k, v, pos, window, rot):
    if window:
        count = min(pos, 1024)
        kr = mx.zeros((1, 8, 1024, 128), mx.bfloat16)
        vr = mx.zeros((1, 8, 1024, 128), mx.bfloat16)
        if count:
            slots = mx.array((np.arange(pos - count, pos) + rot) % 1024, mx.int32)
            kr[:, :, slots, :] = k[:, :, pos - count:pos]
            vr[:, :, slots, :] = v[:, :, pos - count:pos]
        return kr, vr, count, ((pos - 1 + rot) % 1024) if count else 0
    cap = -(-(pos + 32) // 2048) * 2048
    kb = mx.zeros((1, 8, cap, 128), mx.bfloat16)
    vb = mx.zeros((1, 8, cap, 128), mx.bfloat16)
    if pos:
        kb[:, :, :pos] = k[:, :, :pos]
        vb[:, :, :pos] = v[:, :, :pos]
    return kb, vb, pos, 0


def _fast(k, v, pos, inputs, weights, window, boundary, rot):
    q, rel, sink, gate, x, tail = inputs
    rows = q.shape[1]
    kc, vc, count, newest = _committed(k, v, pos, window, rot)
    hg = attend(q, rel, sink, gate, k[:, :, pos:pos + rows], v[:, :, pos:pos + rows], kc, vc,
                count, newest, pos, weights[0], window=window)
    partial, conv_rows = attention_output(hg, weights[1], weights[2], tail, x, boundary=boundary)
    mx.eval(hg, partial, conv_rows)
    return hg, partial, conv_rows


def _rc6(k, v, pos, inputs, weights, window, boundary):
    q, rel, sink, gate, x, tail = inputs
    basis, out, conv = weights
    rows = q.shape[1]
    prof = rel @ basis.astype(rel.dtype)
    heads = []
    for r in range(rows):
        P = pos + r
        begin = max(0, P - 1023) if window else 0
        h = stock_attention(q[:, r:r + 1].transpose(0, 2, 1, 3), k[:, :, begin:P + 1], v[:, :, begin:P + 1],
                            prof[:, r:r + 1], sink[:, r:r + 1], 1024 if window else 0)
        heads.append(h.transpose(0, 2, 1, 3).reshape(1, 1, -1))
    h = mx.concatenate(heads, axis=1)
    hg = (h.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(h.dtype)
    attn = mx.concatenate([hg[:, r:r + 1] @ out.T for r in range(rows)], axis=1)
    conv_out, _, _ = short_conv.reference(attn, tail, conv, None, False)
    partial = conv_out.astype(mx.float32)
    if not boundary:
        partial = partial + x.astype(mx.float32)
    mx.eval(hg, partial, attn)
    return hg, partial, attn.astype(mx.float32)


def _same(a, b):
    word = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
    return bool(mx.array_equal(a.view(word), b.view(word)).item())


CASES = [
    # (window, boundary, history, rows, ring rotation)
    (True, False, 0, 1, 0),        # first token: no history
    (True, False, 300, 3, 0),      # window before fill, single-pass reduction
    (True, True, 1021, 4, 700),    # rows on both sides of the two-pass threshold
    (True, False, 1500, 5, 1023),  # full window, ring wrapped
    (False, False, 300, 2, 0),
    (False, False, 1022, 4, 0),    # global rows crossing the two-pass threshold
    (False, False, 4142, 8, 0),
]


@pytest.mark.parametrize("window,boundary,history,rows,rot", CASES)
def test_bits_against_rc6_and_row_stability(window, boundary, history, rows, rot):
    k, v, inputs, weights = _setup(history + rows, rows, seed=history * 31 + rows)
    hg, partial, conv_rows = _fast(k, v, history, inputs, weights, window, boundary, rot)
    ref_hg, ref_partial, ref_rows = _rc6(k, v, history, inputs, weights, window, boundary)
    assert _same(hg, ref_hg), "attention result differs from RC6"
    assert _same(partial, ref_partial), "partial differs from RC6"
    assert _same(conv_rows, ref_rows), "convolution rows differ from RC6"
    q, rel, sink, gate, x, tail = inputs
    for r in range(rows):
        hist = mx.concatenate([tail, conv_rows[:, :r]], axis=1)[:, -3:] if r else tail
        one = (q[:, r:r + 1], rel[:, r:r + 1], sink[:, r:r + 1], gate[:, r:r + 1], x[:, r:r + 1], hist)
        h1, p1, c1 = _fast(k, v, history + r, one, weights, window, boundary, rot)
        assert _same(h1, hg[:, r:r + 1]) and _same(p1, partial[:, r:r + 1]) and _same(c1, conv_rows[:, r:r + 1]), \
            f"row {r} of a {rows}-row panel differs from its one-row step"


def test_plan_matches_this_gpu_family():
    threshold, blocks = reduction_plan()
    assert (threshold, blocks) in ((1024, 64), (1024, 128), (4096, 32))


def test_rejects_panels_outside_the_served_widths():
    k, v, inputs, weights = _setup(40, 1, seed=1)
    q, rel, sink, gate, x, tail = inputs
    with pytest.raises(ValueError):
        attend(mx.zeros((1, 33, 24, 128), mx.bfloat16), rel, sink, gate, k, v, k, v, 0, 0, 0, weights[0],
               window=True)
