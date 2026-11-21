"""A loop pass's twelve cross-pass message reads read at once (fast/pass_messages.py).

Device tests (Metal GPU only), synthetic operands with the model's shapes: a normalised stack of
P prior states of width 2,304 for panels of 1 to 8 rows (and a 16-row copy panel), the twelve loop
layers' (4, 2,304) float32 queries. Every visit's rows from the batched read must carry the bits of
RC6's own message read of that visit (``stock_messages`` for one row, ``cross_pass_message`` for a
verify panel) and of lane L1's ``read.message_rows``; the step must hand each visit exactly those
rows; the switch must give the per-visit reads back.
"""

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

D = 2304
VISITS = 12


def _bits(a, b):
    a, b = mx.contiguous(a), mx.contiguous(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)).item())


def _operands(rows, prior, seed):
    key = mx.random.key(seed)
    k1, k2 = mx.random.split(key)
    raw = mx.random.normal((1, rows, prior, D), key=k1).astype(mx.bfloat16)
    stack = mx.fast.rms_norm(raw.astype(mx.float32), None, 1e-6)
    table = (mx.random.normal((VISITS, 4, D), key=k2) * 0.05).astype(mx.float32)
    mx.eval(stack, table)
    return stack, table


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 16])
@pytest.mark.parametrize("prior", [2, 3])
def test_every_visit_equals_its_own_read(rows, prior):
    from mtplx.models.inkling import model as ref
    from mtplx.models.inkling.fast.pass_messages import batched_rows
    from mtplx.models.inkling.fast.read import message_rows
    from mtplx.models.inkling.projection import stock_messages

    stack, table = _operands(rows, prior, 20261121 + 31 * rows + prior)
    whole = batched_rows(stack, table)
    reference = ref.cross_pass_message if rows > 1 else stock_messages
    for k in range(VISITS):
        want = reference(table[k], stack, D ** -0.5)
        lane = message_rows(stack, table[k])
        if isinstance(lane, (list, tuple)):
            lane = mx.concatenate(list(lane), axis=1)
        assert whole[k].shape == (1, rows, 4, D)
        assert _bits(whole[k], want), f"visit {k}: differs from RC6's read at {rows} rows, {prior} prior states"
        assert _bits(whole[k], lane), f"visit {k}: differs from lane L1's read at {rows} rows, {prior} prior states"


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
def test_rows_are_contiguous_slices_the_read_kernel_takes_whole(engine_module):
    from mtplx.engine import runtime
    from mtplx.models.inkling.fast.pass_messages import batched_rows

    stack, table = _operands(4, 2, 7)
    whole = batched_rows(stack, table)
    mx.eval(whole)
    for k, ref in enumerate(runtime.engine().describe(whole)):
        assert int(ref.flags) & 2, f"visit {k} rows are not row-contiguous: the read kernel would copy them"


@needs
def test_pass_rows_proves_once_and_serves(monkeypatch):
    from mtplx.models.inkling.fast import pass_messages
    from mtplx.models.inkling.fast.routes import fast_route_counts

    monkeypatch.setattr(pass_messages, "_VERDICT", {})
    stack, table = _operands(1, 2, 11)
    got = pass_messages.pass_rows(stack, table, verify=False)
    assert got is not None and len(got) == VISITS
    assert pass_messages._VERDICT[(1, 2, D, VISITS, False)] is True
    before = dict(fast_route_counts())
    again = pass_messages.pass_rows(stack, table, verify=False)
    assert again is not None
    assert {k: v for k, v in fast_route_counts().items() if k.startswith("fallback.pass_messages")} == \
        {k: v for k, v in before.items() if k.startswith("fallback.pass_messages")}


@needs
def test_switch_off_keeps_the_per_visit_reads(monkeypatch):
    from mtplx.models.inkling.fast import pass_messages

    monkeypatch.setenv("MTPLX_INKLING_PASS_MESSAGES", "0")
    stack, table = _operands(1, 2, 13)
    assert pass_messages.pass_rows(stack, table, verify=False) is None


@needs
def test_a_refused_shape_is_counted_and_served_per_visit(monkeypatch):
    from mtplx.models.inkling.fast import pass_messages
    from mtplx.models.inkling.fast.routes import fast_route_counts

    monkeypatch.setattr(pass_messages, "_VERDICT", {})
    monkeypatch.setattr(pass_messages, "_proof", lambda stack, table, verify: "differs_from_reference")
    stack, table = _operands(2, 2, 17)
    before = fast_route_counts().get("fallback.pass_messages.differs_from_reference", 0)
    assert pass_messages.pass_rows(stack, table, verify=True) is None
    assert pass_messages.pass_rows(stack, table, verify=True) is None
    assert fast_route_counts().get("fallback.pass_messages.differs_from_reference", 0) == before + 2


@needs
def test_attach_serves_both_stage_paths():
    from mtplx.models.inkling.fast import read
    from mtplx.models.inkling.fast.pass_messages import attach, batched_rows
    from mtplx.models.inkling.fast.step import Message

    stack, table = _operands(1, 2, 19)
    rows = batched_rows(stack, table)
    message = Message(stack, table[3], False)
    attach(message, rows[3])
    assert read.message_rows_of(message) is rows[3]
    assert message.read is rows[3]
