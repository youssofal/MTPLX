"""The one-launch depth read (fast.read.exact_read): bit for bit the library's own launches.

On a Mac with matrix units the three- and four-query reads take one launch each; every panel row must
equal what the library's products and RC6's softmax give (the same functions with the one-launch read
off). On other Macs the read must refuse, counted, and give the library's result.
"""

import mlx.core as mx
import pytest

from mtplx.models.inkling.fast import attn_input as A
from mtplx.models.inkling.fast import library_gemm
from mtplx.models.inkling.fast import read as R

D = 2304


def _both(fn):
    outs = []
    for on in (False, True):
        R._EXACT_ON = on
        try:
            out = fn()
            out = mx.concatenate(out, axis=1) if isinstance(out, list) else out
            mx.eval(out)
            outs.append(out)
        finally:
            R._EXACT_ON = True
    return outs


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the GPU")
@pytest.mark.parametrize("queries,sources", [(3, 2), (3, 5), (4, 1), (4, 2)])
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_exact_read_equals_library_launches(queries, sources, rows, monkeypatch):
    # This file proves read.exact_read; the other one-launch read (nax_reads, its own test file) would
    # serve the depth read first.
    monkeypatch.setenv("MTPLX_INKLING_NAX_READS", "0")
    mx.random.seed(17 * queries + 3 * sources + rows)
    q = mx.random.normal((queries, D)).astype(mx.float32)
    keys = mx.random.normal((1, rows, sources, D)).astype(mx.float32)
    if queries == 3:
        want, got = _both(lambda: R.depth_read(q, keys))
    else:
        want, got = _both(lambda: R.message_rows(keys, q))
    assert got.shape == want.shape == (1, rows, queries, D)
    assert bool(mx.array_equal(got.view(mx.uint32), want.view(mx.uint32)).item())
    key = "_".join(map(str, (queries, sources, rows, D)))
    routes = A.route_counts()
    if library_gemm.gpu_generation() >= 17:
        assert routes.get(f"proved:exact_read_{key}", 0) >= 1 or R._exact_verdicts.get((queries, sources, rows, D))
    else:
        assert any(k.startswith("refused:exact_read_") and k.endswith(key) for k in routes)


def test_library_text_is_read_at_run_time():
    text, why = library_gemm.library("matrix_units", "matvec")
    if text is None:
        pytest.skip(f"installed MLX has no readable kernel headers: {why}")
    assert "gemm_loop" in text and "GEMVKernel" in text
    # each header once: two parts that include the same header share it
    assert text.count("steel/utils.h (installed MLX") == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the GPU")
@pytest.mark.parametrize("kind,rank,n", [("bias", 256, 200058), ("bias", 256, 4099), ("update", 256, 768)])
def test_drafter_products_equal_the_library_launches(kind, rank, n):
    """fast/matvec.py: the drafter's folded bias (addmm over the vocabulary) and recurrent update, in
    our launch running the library's mat-vec, bit for bit the library's own launch."""
    from mtplx.models.inkling.fast.matvec import matvec

    mx.random.seed(rank + n)
    if kind == "bias":
        mat = (mx.random.normal((n, rank + 1)) * 0.05).astype(mx.bfloat16)
        vec = mx.tanh(mx.random.normal((1, rank))).astype(mx.bfloat16)
        want, got = mx.addmm(mat[:, -1], vec, mat[:, :-1].T), matvec(vec, mat, cols=rank, bias_col=rank)
    else:
        mat = (mx.random.normal((n, 2 * rank + 2304)) * 0.05).astype(mx.float32)
        vec = mx.random.normal((1, 2 * rank)).astype(mx.float32)
        want, got = vec @ mat[:, : 2 * rank].T, matvec(vec, mat, cols=2 * rank)
    view = mx.uint16 if want.dtype == mx.bfloat16 else mx.uint32
    assert got.shape == want.shape and bool(mx.array_equal(got.view(view), want.view(view)).item())
