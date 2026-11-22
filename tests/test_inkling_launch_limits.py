"""Issue #706: Inkling launches above this GPU's thread-group limit for their kernel's pipeline.

An M1 Max refused the copy lane's 25-row attention partials (800 threads; that pipeline allows 704 there)
and Inkling Optimized Quality FP16 failed to load. ``launch_limits`` checks the first launch of each pipeline
above 256 threads at the call. A refused launch either takes a form with the same bits in smaller groups
(the attention partials in row groups of 8, the experts' column combine) or raises ``GroupRefused`` at the
call, so its stage serves its reference route. ``MTPLX_INKLING_TEST_THREADGROUP_LIMIT`` treats launches
above a limit as refused, so these paths run on a GPU without such a limit.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.models.inkling import launch_limits as ll  # noqa: E402
from mtplx.models.inkling.fast import routes  # noqa: E402

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

_SOURCE = "uint i = thread_position_in_grid.x; o[i] = a[i] + 1.0f;"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(ll.TEST_LIMIT_ENV, raising=False)
    ll.reset_for_tests()
    yield
    ll.reset_for_tests()


def _kernel(name):
    return ll.kernel(name=name, input_names=["a"], output_names=["o"], source=_SOURCE)


def _args(group, n=4096):
    return dict(inputs=[mx.arange(n, dtype=mx.float32)], grid=(n, 1, 1), threadgroup=(group, 1, 1),
                output_shapes=[(n,)], output_dtypes=[mx.float32])


def test_hook_parses_one_limit_or_kernel_prefixes(monkeypatch):
    assert ll.test_limit() is None
    monkeypatch.setenv(ll.TEST_LIMIT_ENV, "704")
    assert ll.test_limit() == 704 and ll.test_limit("anything") == 704
    monkeypatch.setenv(ll.TEST_LIMIT_ENV, "mtplx_inkling_fast_attend_partials=704,mtplx_inkling_l3_combine_rows=512")
    assert ll.test_limit("mtplx_inkling_fast_attend_partials") == 704
    assert ll.test_limit("mtplx_inkling_l3_combine_rows_nvfp4_conv_form0") == 512
    assert ll.test_limit("mtplx_inkling_fast_attend_merge") is None
    assert ll.test_limit() == 512
    monkeypatch.setenv(ll.TEST_LIMIT_ENV, "640,mtplx_inkling_fast_read=512")
    assert ll.test_limit("mtplx_inkling_fast_read") == 512 and ll.test_limit("other") == 640
    monkeypatch.setenv(ll.TEST_LIMIT_ENV, "seven hundred")
    with pytest.raises(ValueError):
        ll.test_limit("x")


def test_error_text_is_mlx_s_own():
    """MLX 0.32's message (two spaces after "than"), and the one the issue quotes."""
    for text in ("Thread group size (800) is greater than  the maximum allowed threads per threadgroup (704).",
                 "[metal::Device] Thread group size (800) is greater than the maximum allowed threads per "
                 "threadgroup (704)."):
        found = ll._ERROR.search(text)
        assert found and found.groups() == ("800", "704")


@needs_gpu
def test_hook_refuses_at_the_call_counts_and_reports(monkeypatch):
    monkeypatch.setenv(ll.TEST_LIMIT_ENV, "512")
    kernel = _kernel("x_issue706_hook")
    with routes.capture_counts() as box:
        with pytest.raises(ll.GroupRefused) as info:
            kernel(**_args(576))
    assert (info.value.group, info.value.limit, info.value.source) == (576, 512, "test hook")
    assert box == {"threadgroup.refused.x_issue706_hook.576": 1}
    out = kernel(**_args(512))[0]
    assert out[5].item() == 6.0
    report = ll.report()
    assert report["test_limit"] == "512"
    assert [(e["kernel"], e["group"], e["limit"], e["source"], e["served_by"]) for e in report["refused"]] == [
        ("x_issue706_hook", 576, 512, "test hook", ll.STAGE_ROUTE)]


@needs_gpu
def test_runs_names_the_same_bits_form_and_counts_a_stage_path(monkeypatch):
    monkeypatch.setenv(ll.TEST_LIMIT_ENV, "x_issue706_split=704")
    kernel = _kernel("x_issue706_split")
    with routes.capture_counts() as box:
        assert not kernel.runs("row groups serve it", **_args(800))
        assert kernel.runs("unused", **_args(704))
    # A stage path: a replayed program, which runs no Python, counts it again from its record call.
    assert box == routes.stage_paths(box) == {"fused.split.x_issue706_split.800": 1}
    assert ll.report()["refused"][0]["served_by"] == "row groups serve it"


@needs_gpu
def test_first_launch_of_a_pipeline_is_evaluated_at_the_call_once():
    kernel = _kernel("x_issue706_first")
    first = kernel(**_args(512))[0]
    assert ll.report()["probes"] == 1 and ll.report()["pipelines_checked"] == 1
    second = kernel(**_args(512))[0]
    kernel(**_args(256))  # at or below PROBE_ABOVE: never checked
    assert ll.report()["probes"] == 1
    assert bool(mx.array_equal(first, second).item()) and first[9].item() == 10.0


@needs_gpu
def test_a_group_no_apple_gpu_runs_is_read_from_mlx_and_mlx_carries_on():
    """2,048 threads is above every Apple GPU's limit: MLX's own error gives this pipeline's limit, the launch
    raises GroupRefused at the call, and launches and plain operations after it run."""
    kernel = _kernel("x_issue706_real")
    with pytest.raises(ll.GroupRefused) as info:
        kernel(**_args(2048))
    assert info.value.source == "pipeline" and 32 <= info.value.limit <= 1024
    assert ll.report()["refused"][0]["limit"] == info.value.limit
    assert kernel(**_args(256))[0][7].item() == 8.0
    assert kernel(**_args(info.value.limit))[0][11].item() == 12.0
    assert (mx.arange(16) * 2).sum().item() == 240


def test_kernels_are_made_through_the_guard():
    """Every Inkling kernel is made through ``launch_limits.kernel`` (a direct ``mx.fast.metal_kernel`` would
    escape the check)."""
    import pathlib

    root = pathlib.Path(ll.__file__).parent
    direct = [str(path.relative_to(root)) for path in sorted(root.rglob("*.py"))
              if path.name != "launch_limits.py" and "mx.fast.metal_kernel(" in path.read_text()]
    assert direct == []


# ---------------------------------------------------------------- the attention partials in row groups

def test_the_attention_qualification_covers_a_copy_rounds_window_panel():
    """2.13.0's first-use proof had window panels of 1 and 2 rows only, so an M1 Max qualified the fused attention
    and then refused a copy round's 25-row window panel at load. The proof now holds one (whichever form serves)."""
    from mtplx.models.inkling.fast.attend import _qualification_cases

    for plan in ((1024, 64), (1024, 128), (4096, 32)):
        assert any(window and rows >= 25 for window, _, rows, _ in _qualification_cases(plan))


def _clipped(shape, limit, key):
    v = mx.random.normal(shape, key=key) * 2.2
    n = mx.linalg.norm(v, axis=-1, keepdims=True)
    return (v * mx.minimum(1.0, limit / (n + 1e-6))).astype(mx.bfloat16)


def _panel(history, rows, rot, dtype, seed):
    keys = mx.random.split(mx.random.key(seed), 8)
    length = history + rows
    k = _clipped((1, 8, length, 128), 22.6, keys[0]).astype(dtype)
    v = mx.fast.rms_norm(mx.random.normal((1, 8, length, 128), key=keys[1]), None, 1e-6).astype(dtype)
    q = _clipped((1, rows, 24, 128), 22.6, keys[2]).astype(dtype)
    rel = _clipped((1, rows, 24, 16), 8.0, keys[3]).astype(dtype)
    sink = (mx.random.normal((1, rows, 24), key=keys[4]) * 3.0).astype(dtype)
    gate = (mx.random.normal((1, rows, 3072), key=keys[5]) * 2.0).astype(dtype)
    basis = (mx.random.normal((16, 1024), key=keys[6]) * 0.3).astype(mx.bfloat16)
    count = min(history, 1024)
    ring_k = mx.zeros((1, 8, 1024, 128), dtype)
    ring_v = mx.zeros((1, 8, 1024, 128), dtype)
    if count:
        slots = mx.array((np.arange(history - count, history) + rot) % 1024, mx.int32)
        ring_k[:, :, slots, :] = k[:, :, history - count:history]
        ring_v[:, :, slots, :] = v[:, :, history - count:history]
    newest = ((history - 1 + rot) % 1024) if count else 0
    mx.eval(k, v, q, rel, sink, gate, basis, ring_k, ring_v)
    args = (q, rel, sink, gate, k[:, :, history:], v[:, :, history:], ring_k, ring_v, count, newest, history, basis)
    return args, (k, v, q, rel, sink, gate, basis)


def _rc6_window(history, parts):
    from mtplx.models.inkling.attention import stock_attention

    k, v, q, rel, sink, gate, basis = parts
    prof = rel @ basis.astype(rel.dtype)
    heads = []
    for r in range(q.shape[1]):
        end = history + r
        begin = max(0, end - 1023)
        h = stock_attention(q[:, r:r + 1].transpose(0, 2, 1, 3), k[:, :, begin:end + 1], v[:, :, begin:end + 1],
                            prof[:, r:r + 1], sink[:, r:r + 1], 1024)
        heads.append(h.transpose(0, 2, 1, 3).reshape(1, 1, -1))
    h = mx.concatenate(heads, axis=1)
    return (h.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(h.dtype)


def _same(a, b):
    word = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(word), b.view(word)).item())


@needs_gpu
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("history,rows,rot", [(0, 9, 0), (300, 12, 0), (1021, 16, 700), (1500, 22, 1023),
                                              (1500, 23, 5), (5000, 24, 333), (5000, 25, 1000), (2000, 32, 17)])
def test_window_panels_in_row_groups_give_the_one_group_bits(history, rows, rot, dtype, monkeypatch):
    """A copy round's window visits (9 to 32 rows) in one threadgroup of 32 x rows threads, or, where this GPU
    cannot run that group, in row groups of 8: the same bits, and RC6's attention."""
    from mtplx.models.inkling.fast.attend import attend, decline_reason, reduction_plan

    if decline_reason(dtype, reduction_plan()) is not None:
        pytest.skip(f"the fused attention declines {dtype} on this Mac")
    args, parts = _panel(history, rows, rot, dtype, seed=history * 5 + rows)
    one_group = attend(*args, window=True, grouped=False)
    grouped = attend(*args, window=True, grouped=True)
    ref = _rc6_window(history, parts)
    monkeypatch.setenv(ll.TEST_LIMIT_ENV, "mtplx_inkling_fast_attend_partials=704")
    with routes.capture_counts() as box:
        hooked = attend(*args, window=True)
    mx.eval(one_group, grouped, ref, hooked)
    assert _same(grouped, one_group), "row groups differ from the one-threadgroup panel"
    assert _same(one_group, ref), "the panel differs from RC6"
    assert _same(hooked, one_group), "the stepped-down panel differs from the one-threadgroup panel"
    split = {f"fused.split.mtplx_inkling_fast_attend_partials.{32 * rows}": 1} if 32 * rows > 704 else {}
    assert {name: n for name, n in box.items() if name.startswith("fused.split.")} == split


# ---------------------------------------------------------------- the experts' column combine

@needs_gpu
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
def test_wide_panels_combine_by_columns_with_the_rows_combine_bits(mode, dtype, monkeypatch):
    """9 to 32 rows on the rows combine (32 x rows threads), and on the column combine (64 threads) where this
    GPU cannot run that group: the same output and convolution rows."""
    from mtplx.models.inkling.fast import experts as fx
    from test_inkling_fast_experts import synthetic_layer

    monkeypatch.setattr(fx, "GLUE", "rows")
    block, conv, norm_weight = synthetic_layer(mode, dtype=dtype)
    keys = mx.random.split(mx.random.key(706), 3)
    source = mx.random.normal((1, 32, 2304), key=keys[0]) * 3.0
    partial = mx.random.normal((1, 32, 2304), key=keys[1])
    tail = mx.random.normal((1, 3, 2304), key=keys[2])
    for rows in (9, 16, 22, 23, 24, 25, 32):
        args = (block, norm_weight, conv.weight, source[:, :rows], partial[:, :rows], tail)
        monkeypatch.delenv(ll.TEST_LIMIT_ENV, raising=False)
        together = fx.expert_visit_parts(*args, eps=1e-6, dtype=dtype)
        monkeypatch.setenv(ll.TEST_LIMIT_ENV, "mtplx_inkling_l3_combine_rows=256")
        with routes.capture_counts() as box:
            columns = fx.expert_visit_parts(*args, eps=1e-6, dtype=dtype)
        mx.eval(together, columns)
        for i, name in enumerate(("output", "convolution rows")):
            assert _same(columns[i], together[i]), (mode, rows, name)
        split = [name for name in routes.stage_paths(box) if name.startswith("fused.split.")]
        assert len(split) == 1 and split[0].startswith("fused.split.mtplx_inkling_l3_combine_rows_")
        assert split[0].endswith(f".{32 * rows}")
