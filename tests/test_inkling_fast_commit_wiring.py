"""The stock commit with lane L7's one-launch pieces wired in writes the same state (GPU).

``DecodeState.commit`` takes a visit's four tails in one launch (``commit_launches.visit_tails``) and
``Drafter.mix`` its four-tap sum in one launch (``commit_launches.mix``). With them on and off
(``MTPLX_INKLING_COMMIT_LAUNCHES=0``) every key and value bank, every tail and the mix must be
equal bit for bit, at every kept count of an 8-row panel, across a window wrap.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

WINDOW, HEADS, DIM, KEEP = 16, 2, 8, 3
WIDTHS = (HEADS * DIM, HEADS * DIM, 24, 24)


def _model(local=(True, False, True, False)):
    layers = [SimpleNamespace(sliding=s, convs=[SimpleNamespace(weight=mx.zeros((KEEP + 1, w))) for w in WIDTHS])
              for s in local]
    config = {"num_key_value_heads": HEADS, "head_dim": DIM, "sliding_window_size": WINDOW,
              "sconv_kernel_size": KEEP + 1}
    return SimpleNamespace(config=config, layers=layers, visit_order=tuple(range(len(local))),
                           embed_tokens=SimpleNamespace(weight=mx.zeros((1,), mx.bfloat16)))


def _state(model, length, seed):
    from mlx_lm.models.cache import ArraysCache, KVCache

    from mtplx.models.inkling.cache import WindowCache
    from mtplx.models.inkling.fast.state import DecodeState

    key = mx.random.key(seed)
    cache = []
    for layer in model.layers:
        kv = WindowCache(max_size=WINDOW) if layer.sliding else KVCache()
        key, k1, k2 = mx.random.split(key, 3)
        rows = mx.random.normal((1, HEADS, length, DIM), key=k1).astype(mx.bfloat16)
        kv.update_and_fetch(rows, (rows * 2).astype(mx.bfloat16))
        conv = ArraysCache(4)
        for slot, width in enumerate(WIDTHS):
            key, k3 = mx.random.split(key)
            conv[slot] = mx.random.normal((1, KEEP, width), key=k3)
        cache.extend((kv, conv))
    state = DecodeState(model, cache)
    state.prepare()
    mx.eval([a for a in state.arrays()])
    return state


def _deltas(model, rows, seed):
    from mtplx.models.inkling.fast.state import VisitDelta

    key = mx.random.key(seed)
    out = []
    for _ in model.layers:
        key, k1, k2, *kt = mx.random.split(key, 7)
        keys = mx.random.normal((1, HEADS, rows, DIM), key=k1).astype(mx.bfloat16)
        values = mx.random.normal((1, HEADS, rows, DIM), key=k2).astype(mx.bfloat16)
        taps = tuple(mx.random.normal((1, rows, w), key=k) for w, k in zip(WIDTHS, kt))
        out.append(VisitDelta(keys, values, taps))
    mx.eval([d.arrays() for d in out])
    return out


def _bits(a):
    view = mx.uint16 if a.dtype.size == 2 else mx.uint32
    return mx.contiguous(a).view(view)


@needs
@pytest.mark.parametrize("kept", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("length", [11, 14])  # 14 + 8 crosses the 16-row window
def test_commit_with_one_launch_tails_writes_the_stock_state(kept, length, monkeypatch):
    model = _model()
    states = {}
    for arm in ("1", "0"):
        monkeypatch.setenv("MTPLX_INKLING_COMMIT_LAUNCHES", arm)
        state = _state(model, length, seed=5)
        state.commit(_deltas(model, 8, seed=9), kept)
        mx.eval(state.arrays())
        states[arm] = state
    on, off = states["1"], states["0"]
    assert on.length == off.length == length + kept
    for a, b in zip(on.arrays(), off.arrays()):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert bool(mx.array_equal(_bits(a), _bits(b)).item())


@needs
def test_one_launch_tails_engage_on_the_gpu():
    from mtplx.models.inkling.fast.state import _one_launch_tails

    tails = [mx.zeros((1, KEEP, w)) for w in WIDTHS]
    rows = [mx.zeros((1, 8, w)) for w in WIDTHS]
    assert _one_launch_tails(tails, rows)
    assert not _one_launch_tails(tails[:3], rows[:3])


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_drafter_mix_one_launch_equals_the_stock_mix(rows, monkeypatch):
    from mtplx.models.inkling.draft import Drafter

    key = mx.random.key(3)
    k1, k2, k3 = mx.random.split(key, 3)
    features = [(mx.random.normal((1, rows, 64), key=k) * (10.0 ** (i - 1))).astype(mx.bfloat16)
                for i, k in enumerate(mx.random.split(k1, 4))]
    blend = mx.softmax(mx.random.normal((4, 4), key=k2), axis=-1).astype(mx.bfloat16)
    weight = (1 + 0.1 * mx.random.normal((64,), key=k3)).astype(mx.bfloat16)
    drafter = SimpleNamespace(blend=blend, cfg={"rms_norm_eps": 1e-6})
    got = {}
    for arm in ("1", "0"):
        monkeypatch.setenv("MTPLX_INKLING_COMMIT_LAUNCHES", arm)
        got[arm] = [Drafter.mix(drafter, features, row, weight) for row in (0, 1, 2, -1)]
        mx.eval(got[arm])
    for a, b in zip(got["1"], got["0"]):
        assert bool(mx.array_equal(_bits(a), _bits(b)).item())


@pytest.fixture
def engine_module(monkeypatch):
    """The native engine module named by ``INKLING_TEST_ENGINE`` (the suite drops ``MTPLX_*``
    variables, so the module path comes in under another name), loaded for this test only."""
    import os

    from mtplx.engine import runtime

    path = os.environ.get("INKLING_TEST_ENGINE", "").strip()
    if not path:
        pytest.skip("INKLING_TEST_ENGINE (the native engine module) not given")
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_MODULE", path)
    if not runtime.available():
        monkeypatch.setattr(runtime, "_REASON", None)
        monkeypatch.setattr(runtime, "_NATIVE", None)
    if not runtime.available():
        pytest.skip(f"the native engine module does not load: {runtime.unavailable()}")
    return runtime.native_module()


@needs
@pytest.mark.parametrize("kept", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("length", [11, 14])  # 14 + 8 crosses the 16-row window
def test_commit_with_the_row_writer_writes_the_stock_state(kept, length, monkeypatch, engine_module):
    from mtplx.models.inkling.fast import state as state_mod

    calls = []
    real = state_mod._row_writer

    def counted(sample):
        write = real(sample)
        if write is None:
            return None
        def wrapped(history, rows, offset):
            calls.append(int(rows.shape[2]))
            return write(history, rows, offset)
        return wrapped

    monkeypatch.setattr(state_mod, "_row_writer", counted)
    model = _model()
    states = {}
    for arm in ("1", "0"):
        monkeypatch.setenv("MTPLX_INKLING_ROW_WRITER", arm)
        st = _state(model, length, seed=5)
        st.commit(_deltas(model, 8, seed=9), kept)
        mx.eval(st.arrays())
        states[arm] = st
        if arm == "1":
            assert calls, "the row writer did not engage"
            assert sum(calls) == 2 * kept * len(model.layers)  # keys and values, every visit, every row
    for a, b in zip(states["1"].arrays(), states["0"].arrays()):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert bool(mx.array_equal(_bits(a), _bits(b)).item())


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_whole_drafter_mix_equals_the_stock_mix_at_the_model_width(rows, monkeypatch):
    from mtplx.models.inkling.draft import Drafter
    from mtplx.models.inkling.fast.commit_launches import mix_norm

    width = 2304
    key = mx.random.key(17 + rows)
    k1, k2, k3 = mx.random.split(key, 3)
    features = [(mx.random.normal((1, rows, width), key=k) * (10.0 ** (i - 1))).astype(mx.bfloat16)
                for i, k in enumerate(mx.random.split(k1, 4))]
    blend = mx.softmax(mx.random.normal((4, 4), key=k2), axis=-1).astype(mx.bfloat16)
    weight = (1 + 0.1 * mx.random.normal((width,), key=k3)).astype(mx.bfloat16)
    mx.eval(features, blend, weight)
    drafter = SimpleNamespace(blend=blend, cfg={"rms_norm_eps": 1e-6})
    for row in (0, 1, 2, -1):
        monkeypatch.setenv("MTPLX_INKLING_COMMIT_LAUNCHES", "0")
        want = Drafter.mix(drafter, features, row, weight)
        got = mix_norm(features, blend, row, weight, 1e-6)
        assert got is not None
        assert bool(mx.array_equal(_bits(got), _bits(want)).item()), f"blend row {row}"
        monkeypatch.setenv("MTPLX_INKLING_COMMIT_LAUNCHES", "1")
        assert bool(mx.array_equal(_bits(Drafter.mix(drafter, features, row, weight)), _bits(want)).item())
