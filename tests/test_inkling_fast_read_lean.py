"""Lane L1's read kernel scheduled leaner (fast/read_lean.py) gives lane L1's bits, GPU.

``read.read`` with the lean kernel and with lane L1's (``MTPLX_INKLING_READ_LEAN=0``) must write the
same bytes in modes 0 (one and three queries, one and three gains), 1 and 3, at panels of 1-8 rows,
2-5 sources, bf16 and float32 partials, at Inkling's width. Mode 2 (the controllers) is covered by
stage A's goldens (tests/test_inkling_fast_attn_input.py, loop visits 20 and 41).
"""

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

D = 2304


def _bits(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


def test_every_block_is_found_in_lane_l1s_text():
    """An edit of lane L1's read kernel (a comment included) that moves a block the lean kernel replaces
    must fail here, not only leave lane L1's kernel serving, counted (integration 2: lane X1's T16
    rename did exactly that until the blocks followed it)."""
    from mtplx.models.inkling.fast import read_lean
    from mtplx.models.inkling.fast.read import _READ

    text, missing = read_lean.derive(_READ)
    assert missing is None, f"lane L1's read kernel no longer holds the lean block {missing!r}"
    assert "float xs[S][4];" in text and "static_cast<T16>" in text


def _operands(rows, sources, partial_dtype, seed, stream="bfloat16"):
    keys = mx.random.split(mx.random.key(seed), sources + 5)
    s16 = getattr(mx, stream)
    closed = [(mx.random.normal((rows, D), key=keys[i]) * (1 + i)).astype(s16) for i in range(sources - 1)]
    partial = mx.random.normal((rows, D), key=keys[sources - 1]).astype(getattr(mx, partial_dtype))
    queries = (mx.random.normal((3, D), key=keys[sources]) * 0.05).astype(mx.float32)
    gains = [(1 + 0.1 * mx.random.normal((D,), key=keys[sources + 1 + j])).astype(s16) for j in range(3)]
    stand_in = mx.zeros((4, D), mx.float32)
    mx.eval(closed, partial, queries, gains, stand_in)
    return closed, partial, queries, gains, stand_in


def _both(monkeypatch, fn):
    from mtplx.models.inkling.fast import read_lean

    out = {}
    for arm in ("1", "0"):
        monkeypatch.setenv("MTPLX_INKLING_READ_LEAN", arm)
        out[arm] = fn()
        mx.eval(out[arm])
    return out["1"], out["0"], read_lean


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("sources", [2, 3, 4, 5])
@pytest.mark.parametrize("mode,nq", [(0, 1), (0, 3), (1, 1), (3, 1)])
def test_lean_read_equals_lane_l1(rows, sources, mode, nq, monkeypatch):
    from mtplx.models.inkling.fast import read, read_lean

    monkeypatch.setattr(read_lean, "_VERDICT", {})
    closed, partial, queries, gains, _ = _operands(rows, sources, "bfloat16", 13 * rows + sources)
    q = queries[:nq]
    kw = {"gains": gains[:nq]} if mode == 0 else {}
    lean, l1, _ = _both(monkeypatch, lambda: read.read(partial, closed, q, 1e-6, mode=mode, **kw))
    assert len(lean) == len(l1)
    for a, b in zip(lean, l1):
        assert _bits(a, b)
    assert list(read_lean._VERDICT.values()) == [True], "the lean kernel did not prove on this Mac"


@needs
@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("sources", [2, 5])
@pytest.mark.parametrize("mode,nq", [(0, 1), (0, 3), (1, 1), (3, 1)])
def test_lean_read_on_a_float16_stream(rows, sources, mode, nq, monkeypatch):
    """The FP16 packs' stream (lane L1's kernel templated on T16 = float16): the lean blocks carry the
    same type, proven on their own key (the stream's type is part of it), the same bytes."""
    from mtplx.models.inkling.fast import read, read_lean

    monkeypatch.setattr(read_lean, "_VERDICT", {})
    closed, partial, queries, gains, _ = _operands(rows, sources, "float16", 17 * rows + sources, stream="float16")
    q = queries[:nq]
    kw = {"gains": gains[:nq]} if mode == 0 else {}
    lean, l1, _ = _both(monkeypatch, lambda: read.read(partial, closed, q, 1e-6, mode=mode, **kw))
    assert len(lean) == len(l1)
    for a, b in zip(lean, l1):
        assert _bits(a, b)
    assert list(read_lean._VERDICT.values()) == [True], "the lean kernel did not prove on a float16 stream"
    assert all(k[-1] == str(mx.float16) for k in read_lean._VERDICT)


@needs
@pytest.mark.parametrize("rows", [1, 4])
def test_lean_read_with_a_float32_partial_and_layer_zero(rows, monkeypatch):
    from mtplx.models.inkling.fast import read, read_lean

    monkeypatch.setattr(read_lean, "_VERDICT", {})
    closed, partial, queries, gains, stand_in = _operands(rows, 4, "float32", rows)
    lean, l1, _ = _both(monkeypatch, lambda: read.read(partial, closed, queries[:1], 1e-6, mode=1))
    assert _bits(lean[0], l1[0])
    _, partial16, *_ = _operands(rows, 1, "bfloat16", 7)
    lean, l1, _ = _both(monkeypatch, lambda: read.read(partial16, [], None, 1e-6, mode=0, gains=gains[:1],
                                                       f32_stand_in=stand_in))
    assert _bits(lean[0], l1[0])
    assert read_lean._VERDICT and all(read_lean._VERDICT.values())


@needs
def test_switch_off_keeps_lane_l1(monkeypatch):
    from mtplx.models.inkling.fast import read, read_lean

    monkeypatch.setattr(read_lean, "_VERDICT", {})
    monkeypatch.setenv("MTPLX_INKLING_READ_LEAN", "0")
    closed, partial, queries, gains, _ = _operands(1, 3, "bfloat16", 1)
    read.read(partial, closed, queries[:1], 1e-6, mode=1)
    assert not read_lean._VERDICT


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("sources", [2, 3, 4, 5])
@pytest.mark.parametrize("controllers,read_in,separate", [(3, False, False), (1, False, False), (3, True, False),
                                                            (3, False, True)])
def test_lean_controllers_equal_lane_l1(rows, sources, controllers, read_in, separate, monkeypatch):
    """Mode 2: the controllers' down products, sources and messages, with the one-query read in the
    launch or RC6's three-query read handed in, messages as one array or one a row."""
    from mtplx.models.inkling.fast import read, read_lean

    monkeypatch.setattr(read_lean, "_VERDICT", {})
    closed, partial, queries, gains, _ = _operands(rows, sources, "bfloat16", 31 * rows + sources)
    keys = mx.random.split(mx.random.key(rows + 100 * sources), 6)
    messages = mx.random.normal((rows, 4, D), key=keys[0]).astype(mx.float32)
    downs = [(mx.random.normal((64, 2 * D), key=keys[1 + i]) * 0.02).astype(mx.bfloat16) for i in range(3)]
    exact = mx.random.normal((rows, 3, D), key=keys[4]).astype(mx.float32) if read_in else None
    msgs = [messages[r:r + 1].reshape(1, 1, 4, D) for r in range(rows)] if separate else messages
    mx.eval(messages, downs, *( [exact] if read_in else []))
    lean, l1, _ = _both(monkeypatch, lambda: read.read(partial, closed, queries[:1], 1e-6, mode=2, messages=msgs,
                                                       read_in=exact, downs=downs[:controllers], msg_base=0,
                                                       controllers=controllers))
    assert len(lean) == len(l1) == 3
    for a, b in zip(lean, l1):
        assert _bits(a, b)
    assert read_lean._VERDICT and all(read_lean._VERDICT.values())
