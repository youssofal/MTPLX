"""The pass stack and the final read on lane L1's read kernel (fast/frame_reads.py), GPU.

Synthetic bf16 blocks of width 2,304 for panels of 1 to 8 rows (and a 16-row copy panel): the stack
must equal ``rms_norm(stack(prior).astype(float32))`` and the final hidden rows the reference's
``norm(row_stable_read(...), final_norm).astype(bf16)``, bit for bit; a float16 stream and the
switch must keep MLX's launches.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

D = 2304


def _bits(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


def _blocks(rows, count, seed, dtype=mx.bfloat16):
    keys = mx.random.split(mx.random.key(seed), count)
    out = [(mx.random.normal((1, rows, D), key=k) * (1 + i)).astype(dtype) for i, k in enumerate(keys)]
    mx.eval(out)
    return out


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 16])
@pytest.mark.parametrize("prior", [2, 3])
def test_pass_stack_equals_the_mlx_stack(rows, prior, monkeypatch):
    from mtplx.models.inkling.fast import frame_reads

    monkeypatch.setattr(frame_reads, "_VERDICT", {})
    blocks = _blocks(rows, prior, 100 * rows + prior)
    stand_in = mx.zeros((12, 4, D), mx.float32)
    got = frame_reads.pass_stack(blocks, 1e-6, stand_in)
    want = mx.fast.rms_norm(mx.stack(blocks, axis=-2).astype(mx.float32), None, 1e-6)
    assert _bits(got, want)
    assert frame_reads._VERDICT[("stack", rows, prior, D)] is True


def _model(final_dtype=mx.bfloat16):
    k1, k2 = mx.random.split(mx.random.key(5))
    query = (mx.random.normal((D,), key=k1) * 0.05).astype(mx.float32)
    norm = (1 + 0.1 * mx.random.normal((D,), key=k2)).astype(final_dtype)
    mx.eval(query, norm)
    return SimpleNamespace(final_query=query, final_norm=norm, eps=1e-6,
                           config={"attn_res_normalize_values": True})


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 16])
@pytest.mark.parametrize("closed", [3, 4])
def test_final_hidden_equals_the_reference(rows, closed, monkeypatch):
    from mtplx.models.inkling import model as ref
    from mtplx.models.inkling.fast import frame_reads
    from mtplx.models.inkling.fast.step import Panel, row_stable_read

    monkeypatch.setattr(frame_reads, "_VERDICT", {})
    model = _model()
    blocks = _blocks(rows, closed + 1, 7 * rows + closed)
    shut, x = blocks[:-1], blocks[-1]
    panel = Panel(rows, mx.array(4142, mx.uint32), True if rows > 1 else None)

    def reference():
        last = row_stable_read(panel, shut, x, model.final_query, model.eps, True)[..., 0, :]
        return ref.norm(last, model.final_norm, model.eps).astype(x.dtype)

    got = frame_reads.final_hidden(model, panel, shut, x, reference)
    want = reference()
    assert _bits(got, want)
    assert frame_reads._VERDICT[("final", rows, closed, D, str(mx.bfloat16))] is True


@needs
def test_a_float16_stream_keeps_mlx(monkeypatch):
    from mtplx.models.inkling.fast import frame_reads
    from mtplx.models.inkling.fast.routes import fast_route_counts

    monkeypatch.setattr(frame_reads, "_VERDICT", {})
    blocks = _blocks(1, 2, 3, dtype=mx.float16)
    before = fast_route_counts().get("fallback.frame_reads.stack_stream_dtype", 0)
    got = frame_reads.pass_stack(blocks, 1e-6, mx.zeros((12, 4, D), mx.float32))
    want = mx.fast.rms_norm(mx.stack(blocks, axis=-2).astype(mx.float32), None, 1e-6)
    assert _bits(got, want)
    assert fast_route_counts().get("fallback.frame_reads.stack_stream_dtype", 0) == before + 1
    assert not frame_reads._VERDICT


@needs
def test_switch_off_keeps_mlx(monkeypatch):
    from mtplx.models.inkling.fast import frame_reads

    monkeypatch.setattr(frame_reads, "_VERDICT", {})
    monkeypatch.setenv("MTPLX_INKLING_FRAME_READS", "0")
    blocks = _blocks(2, 2, 9)
    frame_reads.pass_stack(blocks, 1e-6, mx.zeros((12, 4, D), mx.float32))
    assert not frame_reads._VERDICT
