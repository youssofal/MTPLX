"""CPU tests for the layer that decides how a 2 to 8 row panel is projected.

The panel kernel is compared with the stock one-row products on the GPU at first use;
these tests pin which widths reach it, what it is compared with and what serves when it
is not admitted.
"""
import mlx.core as mx
import pytest

from mtplx.models.inkling import model as im
from mtplx.models.inkling.precision import AffineWeight


def test_projection_receipts_belong_to_the_request():
    from mtplx.models.inkling.head import note_route, observe_projection_routes

    with observe_projection_routes() as outer:
        note_route("projection", "explicit_trial")
        with observe_projection_routes() as inner:
            note_route("projection", "probe_no_fit")
            note_route("fp4", "selected")
        note_route("projection", "selected")
    assert outer == {"panel": 2}
    assert inner == {"probe_no_fit": 1}


@pytest.fixture(autouse=True)
def on_the_cpu():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


def operands(rows, lead=(1,), width=48, outputs=20, dtype=mx.float32):
    mx.random.seed(7)
    x = mx.random.normal((*lead, rows, width)).astype(dtype)
    return x, mx.random.normal((outputs, width)).astype(dtype)


def singletons(x, weight):
    flat = x.reshape(-1, x.shape[-1])
    return mx.concatenate([flat[r : r + 1] @ weight.T for r in range(flat.shape[0])], axis=0)


def gpu_stub(monkeypatch, admitted, kernel=None):
    """Pretend the panel kernel can run here; record what the proof layer was handed."""
    seen = {}

    def measured_route(name, shape, stock, candidate, *, work_bytes, proof, trial, pack):
        seen.update(name=name, shape=shape, stock=stock(), candidate=candidate(), proof=proof,
                    trial=trial, pack=pack)
        return admitted

    monkeypatch.setattr(im, "_panel_device", lambda: True)
    monkeypatch.setattr(im, "measured_route", measured_route)
    monkeypatch.setattr(im, "panel_linear", kernel or (lambda x, weight: (x @ weight.T) + 1))
    return seen


@pytest.mark.parametrize("rows", [1, 9, 512])
def test_one_row_and_wide_panels_take_the_stock_product(monkeypatch, rows):
    gpu_stub(monkeypatch, True)
    x, weight = operands(rows)
    assert mx.array_equal(im.row_stable_linear(x, weight), x @ weight.T)


def test_a_packed_weight_takes_its_own_product(monkeypatch):
    seen = gpu_stub(monkeypatch, True)

    class Packed(AffineWeight):
        def __init__(self):
            pass

        def product(self, x):
            return x * 3

    x, _ = operands(4)
    assert mx.array_equal(im.row_stable_linear(x, Packed()), x * 3) and not seen


@pytest.mark.parametrize("rows,lead", [(2, (1,)), (8, (1,)), (5, ()), (3, (1, 1))])
def test_without_the_kernel_each_row_is_projected_alone(rows, lead):
    x, weight = operands(rows, lead)
    out = im.row_stable_linear(x, weight)
    assert out.shape == (*x.shape[:-1], weight.shape[0])
    assert im._same_bits(out.reshape(rows, -1), singletons(x, weight))


def test_rows_spread_over_a_leading_axis_are_still_one_panel():
    x, weight = operands(1, lead=(6,))  # six rows as (6, 1, width)
    out = im.row_stable_linear(x, weight)
    assert out.shape == (6, 1, weight.shape[0])
    assert im._same_bits(out.reshape(6, -1), singletons(x, weight))


def test_the_kernel_is_proved_against_the_rows_projected_one_at_a_time(monkeypatch):
    seen = gpu_stub(monkeypatch, True)
    im.bind_projection_proofs("the pack's resolver")
    try:
        x, weight = operands(4)
        out = im.row_stable_linear(x, weight)
    finally:
        im.bind_projection_proofs(None)
    assert mx.array_equal(out, (x @ weight.T) + 1)  # admitted: the kernel serves
    assert seen["name"] == "projection" and seen["pack"] == "the pack's resolver"
    assert seen["shape"] == (4, 48, 20, str(mx.float32), str(mx.float32))
    assert im._same_bits(seen["stock"].reshape(4, -1), singletons(x, weight))
    assert mx.array_equal(seen["candidate"], out) and seen["proof"] is im._same_bits
    assert seen["trial"] is True  # the proof admits the kernel; the timing race does not decide


def test_a_proven_kernel_serves_even_when_it_loses_the_timing_race(monkeypatch, tmp_path):
    import time

    from mtplx.models.inkling import head

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x" * 64)
    monkeypatch.setenv("MTPLX_KERNEL_PROOF_DIR", str(tmp_path / "proofs"))
    monkeypatch.setattr(head, "fits_route", lambda extra: True)
    monkeypatch.setattr(im, "_panel_device", lambda: True)
    rows_alone, served = im.row_by_row_linear, []

    def kernel(x, weight):  # the one-row bits, and certain to lose the race
        served.append(1)
        out = rows_alone(x, weight)
        mx.eval(out)
        time.sleep(0.02)
        return out

    monkeypatch.setattr(im, "panel_linear", kernel)
    im.bind_projection_proofs(head.pack_proofs(tmp_path))
    try:
        x, weight = operands(3)
        first = im.row_stable_linear(x, weight)
        proving = len(served)
        second = im.row_stable_linear(x, weight)
    finally:
        im.bind_projection_proofs(None)
    assert im._same_bits(first.reshape(3, -1), singletons(x, weight)) and im._same_bits(first, second)
    assert len(served) == proving + 1  # the kernel served the second call
    stats = head.inkling_first_use_stats()
    decision = [d for d in stats["decisions"] if d["route"] == "projection" and d["shape"][:3] == [3, 48, 20]][-1]
    assert decision["reason"] == "no_gain" and stats["calls"]["projection.explicit_trial"] >= 2


def test_a_kernel_that_is_not_admitted_leaves_the_rows_projected_one_at_a_time(monkeypatch):
    gpu_stub(monkeypatch, False)
    x, weight = operands(7)
    assert im._same_bits(im.row_stable_linear(x, weight).reshape(7, -1), singletons(x, weight))


def test_a_dtype_the_kernel_does_not_take_never_reaches_the_proof(monkeypatch):
    seen = gpu_stub(monkeypatch, True)
    x, weight = operands(3)
    out = im.row_stable_linear(x.astype(mx.int32), weight)
    assert not seen and out.shape == (1, 3, 20)


def test_same_bits_compares_bits_of_equal_dtype_and_shape():
    a = mx.array([[1.5, float("nan"), -0.0]], mx.bfloat16)
    assert im._same_bits(a, a * 1)
    assert not im._same_bits(a, mx.array([[1.5, float("nan"), 0.0]], mx.bfloat16))  # -0.0 is not 0.0
    assert not im._same_bits(a, a.astype(mx.float16))
    assert not im._same_bits(a, a.reshape(3, 1))
    assert im._same_bits(a.astype(mx.float32), a.astype(mx.float32))


def test_the_row_stable_projection_is_what_the_model_uses_with_no_switch():
    import os

    if "MTPLX_INKLING_PROJECTION" not in os.environ:
        assert im.linear is im.row_stable_linear
