"""The three-query depth read and the message reads as our launches (fast/nax_reads.py), GPU.

Every shape must equal the library's products around the softmax, bit for bit: the depth read at
panels of 1-8 and 16 rows with 2-5 sources (queries in float32 and bf16), the message read of twelve
visits at 1-8 and 16 rows with 2-3 prior states. Shapes past what the launch takes, a non-float32
key array and the switch keep the library's launches.
"""

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

D = 2304


def _bits(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


@pytest.fixture(autouse=True)
def nax_reads_on(monkeypatch):
    """These launches are off by default (``read.exact_read`` serves the reads); this file proves them
    with their switch on (a test that turns it off sets it again)."""
    monkeypatch.setenv("MTPLX_INKLING_NAX_READS", "1")


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
def test_the_library_plan_is_read_from_its_dispatch(engine_module, monkeypatch):
    from mtplx.models.inkling.fast import nax_reads

    monkeypatch.setattr(nax_reads, "_PLAN", {})
    found = nax_reads.plan(3, 5, D)
    assert found is not None, "the library's split-K plan was not read from its dispatch"
    partitions, size = found
    assert partitions >= 1 and size % 256 == 0 and partitions * size >= D


def _keys(rows, count, seed):
    raw = mx.random.normal((1, rows, count, D), key=mx.random.key(seed)) * (1 + mx.arange(count).reshape(1, 1, -1, 1))
    keys = mx.fast.rms_norm(raw.astype(mx.bfloat16).astype(mx.float32), None, 1e-6)
    mx.eval(keys)
    return keys


def _library_depth(queries, keys):
    from mtplx.models.inkling.fast import read

    q = queries.reshape(-1, D).astype(mx.float32)
    kw = read.scores(q, keys)
    return read.probs(kw.get("scores"), D ** -0.5, kw.get("joined")) @ keys


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 16])
@pytest.mark.parametrize("count", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_depth_read_equals_the_library(rows, count, dtype, monkeypatch):
    from mtplx.models.inkling.fast import nax_reads

    monkeypatch.setattr(nax_reads, "_VERDICT", {})
    keys = _keys(rows, count, 11 * rows + count)
    queries = (mx.random.normal((3, D), key=mx.random.key(5)) * 0.05).astype(getattr(mx, dtype))
    mx.eval(queries)
    want = _library_depth(queries, keys)
    got = nax_reads.depth_read(queries, keys, lambda: _library_depth(queries, keys))
    assert _bits(got.reshape(want.shape), want)
    verdict = nax_reads._VERDICT[("depth", 3, 3, count, rows, D, str(queries.dtype))]
    assert verdict, "the launch did not prove on this Mac"


def _library_messages(table, stack):
    from mtplx.models.inkling.fast import pass_messages

    rows, count, width = stack.shape[1], stack.shape[2], stack.shape[3]
    visits = table.shape[0]
    scores = table.reshape(visits * 4, width) @ stack.reshape(rows * count, width).T
    weights = pass_messages._kernel()(
        inputs=[scores],
        template=[("W", rows), ("S", count), ("NQ", 4), ("SCALE", pass_messages._f32_bits(width ** -0.5))],
        grid=(visits * rows * 4 * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(visits, rows, 4, count)], output_dtypes=[mx.float32])[0]
    return weights @ stack


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 16])
@pytest.mark.parametrize("prior", [2, 3])
def test_message_read_equals_the_library(rows, prior, monkeypatch):
    from mtplx.models.inkling.fast import nax_reads

    monkeypatch.setattr(nax_reads, "_VERDICT", {})
    stack = _keys(rows, prior, 7 * rows + prior)
    table = (mx.random.normal((12, 4, D), key=mx.random.key(9)) * 0.05).astype(mx.float32)
    mx.eval(table)
    want = _library_messages(table, stack)
    got = nax_reads.message_read(table, stack, lambda: _library_messages(table, stack))
    assert _bits(got, want)
    assert nax_reads._VERDICT[("message", 48, 4, prior, rows, D, str(mx.float32))]


@needs
def test_the_batched_message_rows_take_the_launch(monkeypatch):
    from mtplx.models.inkling.fast import nax_reads, pass_messages

    monkeypatch.setattr(nax_reads, "_VERDICT", {})
    stack = _keys(4, 3, 3)
    table = (mx.random.normal((12, 4, D), key=mx.random.key(4)) * 0.05).astype(mx.float32)
    want = _library_messages(table, stack)
    got = pass_messages.batched_rows(stack, table)
    assert len(got) == 12
    for k in range(12):
        assert _bits(got[k], want[k : k + 1])
    assert nax_reads._VERDICT[("message", 48, 4, 3, 4, D, str(mx.float32))]


@needs
def test_switch_off_keeps_the_library(monkeypatch):
    from mtplx.models.inkling.fast import nax_reads

    monkeypatch.setattr(nax_reads, "_VERDICT", {})
    monkeypatch.setenv("MTPLX_INKLING_NAX_READS", "0")
    keys = _keys(1, 5, 2)
    queries = (mx.random.normal((3, D), key=mx.random.key(1)) * 0.05).astype(mx.float32)
    got = nax_reads.depth_read(queries, keys, lambda: _library_depth(queries, keys))
    assert _bits(got, _library_depth(queries, keys))
    assert not nax_reads._VERDICT


@needs
def test_float16_keys_keep_the_library(monkeypatch):
    from mtplx.models.inkling.fast import nax_reads
    from mtplx.models.inkling.fast.routes import fast_route_counts

    monkeypatch.setattr(nax_reads, "_VERDICT", {})
    keys = _keys(1, 3, 6).astype(mx.float16)
    queries = mx.zeros((3, D), mx.float32)
    before = fast_route_counts().get("fallback.nax_reads.depth_dtype", 0)
    nax_reads.depth_read(queries, keys, lambda: mx.zeros((1, 1, 3, D), mx.float32))
    assert fast_route_counts().get("fallback.nax_reads.depth_dtype", 0) == before + 1
    assert not nax_reads._VERDICT
