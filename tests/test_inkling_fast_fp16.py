"""Float16 streams (the FP16 sibling packs) never reach lane L1's bf16 kernels.

CPU tests: each builds the call an FP16 pack makes and checks it is refused with a counted
reason before any kernel is built, instead of failing inside (review 2, finding 9: the fused MLP
read's first pass raised ``IndexError`` on an FP32 partial with FP16 closed states). The
kernels round where RC6 rounds to the stream's dtype, and they round to bf16, so serving a float16
stream would give the wrong bits; the step serves the reference for those visits.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

D = 2304


@pytest.fixture(autouse=True)
def _cpu_and_fresh_counts():
    from mtplx.models.inkling.fast.attn_input import reset_route_counts

    reset_route_counts()
    with mx.stream(mx.cpu):
        yield


def _counted(name):
    from mtplx.models.inkling.fast.attn_input import route_counts

    return route_counts().get(name, 0)


def test_first_pass_read_on_a_float16_stream_is_refused_and_counted():
    from mtplx.models.inkling.fast.read import read

    partial = mx.zeros((1, D), mx.float32)
    closed = [mx.zeros((1, D), mx.float16)]
    with pytest.raises(RuntimeError, match="bf16 streams"):
        read(partial, closed, mx.zeros((1, D), mx.float32), 1e-6, mode=1)
    assert _counted("refused:read_stream_dtype") == 1


def test_fused_mlp_read_first_pass_declines_a_float16_stream():
    from mtplx.models.inkling.fast.read import mlp_source

    layer = SimpleNamespace(mlp_query=mx.zeros((1, D), mx.float32), eps=1e-6, controllers=[])
    partial = mx.zeros((1, 1, D), mx.float32)
    closed = [mx.zeros((1, 1, D), mx.float16), mx.zeros((1, 1, D), mx.float16)]
    with pytest.raises(RuntimeError, match="bf16 streams"):
        mlp_source(layer, partial, closed)
    assert _counted("refused:mlp_stream_dtype") == 1
    assert _counted("fused:mlp_first") == 0


def test_fused_mlp_read_loop_pass_declines_a_float16_stream():
    from mtplx.models.inkling.fast.read import stage_expert_input

    layer = SimpleNamespace(mlp_query=mx.zeros((1, D), mx.float32), eps=1e-6, controllers=[])
    with pytest.raises(RuntimeError, match="bf16 streams"):
        stage_expert_input(layer, mx.zeros((1, 2, D), mx.float32), [mx.zeros((1, 2, D), mx.float16)], None, None)
    assert _counted("refused:mlp_stream_dtype") == 1


def test_fused_stage_a_declines_a_float16_stream():
    from mtplx.models.inkling.fast.attn_input import attention_input

    layer = SimpleNamespace(sliding=True, cfg={"attn_res_normalize_values": True})
    x = mx.zeros((1, 1, D), mx.float16)
    with pytest.raises(RuntimeError, match="bf16 streams"):
        attention_input(layer, x, [x], None, None, mx.array(0, mx.uint32))
    assert _counted("refused:stream_dtype") == 1


def test_step_declines_stage_d_on_a_float16_stream_and_serves_the_reference(monkeypatch):
    from mtplx.models.inkling.fast import step
    from mtplx.models.inkling.fast.routes import fast_route_counts

    served = []
    monkeypatch.setattr(step, "expert_input", lambda *args: served.append(args) or "reference")
    layer = SimpleNamespace(controllers=[None] * 4)
    before = fast_route_counts().get("fallback.d.stream_dtype", 0)
    out = step.fused_expert_input(layer, mx.zeros((1, 1, D), mx.float32), [mx.zeros((1, 1, D), mx.float16)], None, None)
    assert out == "reference" and len(served) == 1
    assert fast_route_counts().get("fallback.d.stream_dtype", 0) == before + 1
    assert step.expert_input_reason(layer, False, mx.float16) == "stream_dtype"
    assert step.expert_input_reason(layer, False, mx.bfloat16) is None
