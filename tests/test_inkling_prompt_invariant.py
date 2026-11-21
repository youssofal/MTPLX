"""The width-independent prompt read (``MTPLX_INKLING_PROMPT_FAST=invariant``, off by default): a row's projections
and its expert block give the same bits whatever rows share its chunk, from 1 row to 2,100, at any place in the
chunk; and the switch's own rules (it implies the MXFP8 gate-up on the tensor units; the width ladder)."""

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.models.inkling.fast import prompt_route as PR  # noqa: E402


def test_invariant_implies_the_fast_experts_and_picks_the_widest_width_that_fits(monkeypatch):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)
    monkeypatch.setenv(PR.PROMPT_FAST_ENV, "invariant")
    assert PR.invariant_sums() and PR.prompt_fast("experts") and not PR.prompt_fast("chunk")
    assert PR.wide_rows(4_000, headroom_bytes=64 * 1024**3) is None  # no chunk part: the default width
    monkeypatch.setenv(PR.PROMPT_FAST_ENV, "invariant,chunk")
    assert PR.wide_rows(4_000, headroom_bytes=64 * 1024**3) == PR.WIDE_ROWS == 2048
    # a long prompt's attention planes push the width down, a small Mac's headroom too, never below 256
    assert PR.wide_rows(200_000, headroom_bytes=64 * 1024**3) < 2048
    assert PR.wide_rows(4_000, headroom_bytes=3 * 1024**3) == 512
    assert PR.wide_rows(4_000, headroom_bytes=0) == 256
    widths = [PR.wide_rows(n, headroom_bytes=24 * 1024**3) for n in (1_000, 16_000, 64_000, 260_000)]
    assert widths == sorted(widths, reverse=True) and set(widths) <= set(PR.INVARIANT_WIDTHS)


if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the prompt launches need a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import expert_fusion as EF  # noqa: E402
from mtplx.models.inkling import model as M  # noqa: E402
from mtplx.models.inkling.fast import prompt_experts as PE  # noqa: E402
from mtplx.models.inkling.fast import prompt_gemm as PG  # noqa: E402

if not (PG.available()[0] and PE.available()[0]):  # pragma: no cover - GPUs without tensor units
    pytest.skip("the width-independent read needs tensor units", allow_module_level=True)


def same(a, b):
    word = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(word), b.view(word)).item())


SLICES = [(0, 1), (0, 7), (5, 38), (0, 512), (700, 1213), (1536, 2100)]


@pytest.fixture
def invariant(monkeypatch):
    monkeypatch.setenv(PR.PROMPT_FAST_ENV, "invariant")
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)


def test_a_rows_projections_do_not_depend_on_its_chunk(invariant):
    k = iter(mx.random.split(mx.random.key(5), 8))
    x = (mx.random.normal((2100, 2304), key=next(k)) * 0.5).astype(mx.bfloat16)
    mats = [(mx.random.normal((w, 2304), key=next(k)) * 0.02).astype(mx.bfloat16) for w in (3072, 384, 24, 1024)]
    whole = PG.project(x, mats, "test")
    router = (mx.random.normal((128, 2304), key=next(k)) * 0.02).astype(mx.float32)
    xf = x.astype(mx.float32)
    (whole_router,) = PG.project(xf, [router], "test_router")
    mx.eval(whole, whole_router)
    for a, b in SLICES:
        part = PG.project(x[a:b], mats, "test")
        (part_router,) = PG.project(xf[a:b], [router], "test_router")
        mx.eval(part, part_router)
        for got, want in zip(part, whole):
            assert same(got, want[a:b]), (a, b, got.shape)
        assert same(part_router, whole_router[a:b]), (a, b)
    assert PG.counts().get("test.ours.invariant", 0) >= len(SLICES)


@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
def test_a_rows_expert_block_does_not_depend_on_its_chunk(invariant, mode):
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_fast_prefill_experts", Path(__file__).with_name("test_inkling_fast_prefill_experts.py"))
    helpers = importlib.util.module_from_spec(spec)
    sys.modules["_fast_prefill_experts"] = helpers
    spec.loader.exec_module(helpers)
    block = helpers.synthetic_block(mode)
    hidden = helpers.hidden_rows(2100, seed=11)
    whole = block(hidden, prompt=True)
    mx.eval(whole)
    for a, b in SLICES:
        part = block(hidden[:, a:b], prompt=True)
        mx.eval(part)
        assert same(part, whole[:, a:b]), (mode, a, b)
    selections = dict(EF._state(block).counts)
    assert any("prompt_invariant." in key for key in selections), selections
