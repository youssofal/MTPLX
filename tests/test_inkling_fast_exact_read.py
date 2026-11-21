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
