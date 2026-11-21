"""The 16-bit vocabulary head with the soft cap inside its launch (head.head_logits) against the launch
capped by MLX's operations (what Model.head does), value for value.

Device tests: a 16-bit vocabulary matrix of the packs' width (bf16, and float16 for the FP16 siblings), 1
to 8 rows (the singleton launch and the panel), the packs' cap. The route's first-use race against MLX's
product (head.measured_route) is taken as won, so the native launch serves.
"""

import pytest

mx = pytest.importorskip("mlx.core")

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")
WIDTH, ROWS, VOCAB = 2304, 8200, 8137


def _bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)).item())


@needs_gpu
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("panel", ["auto", "0"])
def test_the_capped_launch_equals_the_launch_capped_by_mlx(rows, dtype, panel, monkeypatch):
    from mtplx.models.inkling import head

    monkeypatch.setattr(head, "measured_route", lambda *args, **kw: True)
    monkeypatch.setenv("MTPLX_INKLING_HEAD_PANEL", panel)  # "0": one row a launch; "auto": the panel
    h = (mx.random.normal((1, rows, WIDTH), key=mx.random.key(rows)) * 1.5).astype(dtype)
    w = (mx.random.normal((ROWS, WIDTH), key=mx.random.key(3)) * 0.05).astype(dtype)
    mx.eval(h, w)
    got, route = head.head_logits(h, w, VOCAB, 30.0)
    want = head.vocabulary_logits(h, w, VOCAB)
    want = mx.tanh(want / 30.0) * 30.0
    mx.eval(got, want)
    assert route == "native"
    assert got.dtype == mx.float32 and got.shape == (1, rows, VOCAB)
    assert _bits(got, want), (rows, dtype, panel)


@needs_gpu
def test_no_cap_is_the_plain_launch_and_a_refusal_caps_with_mlx(monkeypatch):
    from mtplx.models.inkling import head

    monkeypatch.setattr(head, "measured_route", lambda *args, **kw: True)
    h = mx.random.normal((1, 2, WIDTH), key=mx.random.key(1)).astype(mx.bfloat16)
    w = (mx.random.normal((ROWS, WIDTH), key=mx.random.key(2)) * 0.05).astype(mx.bfloat16)
    plain, route = head.head_logits(h, w, VOCAB, None)
    assert route == "native" and _bits(plain, head.vocabulary_logits(h, w, VOCAB))
    # A capped launch that refuses (its proof disagreed here) leaves the uncapped launch, capped by MLX.
    monkeypatch.setattr(head, "_capped", lambda *args, **kw: None)
    got, route = head.head_logits(h, w, VOCAB, 30.0)
    want = mx.tanh(head.vocabulary_logits(h, w, VOCAB) / 30.0) * 30.0
    assert route == "capped_by_mlx" and _bits(got, want)
