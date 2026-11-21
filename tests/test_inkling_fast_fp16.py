"""Stream types at lane L1's stage A and MLP read: bf16 and float16 (the FP16 sibling packs) are
served, anything else is refused with a counted reason before a kernel is built.

The kernels round where RC6 rounds to the stream's dtype (``.astype(x.dtype)``), to the stream's
16-bit type, so a float16 stream gets RC6's float16 bits (proven on the FP16 packs' real weights:
``outputs/inkling-fast-20261121/summary/x1/check_fp16_stages.py``). What must still be refused, on
the CPU before any launch (review 2, finding 9: the fused MLP read's first pass once raised
``IndexError`` on an FP32 partial with FP16 closed states): operands of two 16-bit types in one call,
a stream with no 16-bit operand, and the step must decline those visits by its own rules.
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


def test_a_read_mixing_two_16_bit_types_is_refused_and_counted():
    from mtplx.models.inkling.fast.read import read

    partial = mx.zeros((1, D), mx.float32)
    closed = [mx.zeros((1, D), mx.float16), mx.zeros((1, D), mx.bfloat16)]
    with pytest.raises(RuntimeError, match="one 16-bit stream type"):
        read(partial, closed, mx.zeros((1, D), mx.float32), 1e-6, mode=1)
    assert _counted("refused:read_stream_dtype") == 1


def test_a_read_with_no_16_bit_operand_is_refused_and_counted():
    from mtplx.models.inkling.fast.read import read

    with pytest.raises(RuntimeError, match="one 16-bit stream type"):
        read(mx.zeros((1, D), mx.float32), [mx.zeros((1, D), mx.float32)], mx.zeros((1, D), mx.float32), 1e-6,
             mode=1)
    assert _counted("refused:read_stream_dtype") == 1


def test_fused_mlp_read_declines_closed_states_of_two_types():
    from mtplx.models.inkling.fast.read import mlp_source

    layer = SimpleNamespace(mlp_query=mx.zeros((1, D), mx.float32), eps=1e-6, controllers=[])
    partial = mx.zeros((1, 1, D), mx.float32)
    closed = [mx.zeros((1, 1, D), mx.float16), mx.zeros((1, 1, D), mx.bfloat16)]
    with pytest.raises(RuntimeError, match="one 16-bit stream type"):
        mlp_source(layer, partial, closed)
    assert _counted("refused:mlp_stream_dtype") == 1
    assert _counted("fused:mlp_first") == 0


def test_fused_stage_a_declines_a_float32_stream():
    from mtplx.models.inkling.fast.attn_input import attention_input

    layer = SimpleNamespace(sliding=True, cfg={"attn_res_normalize_values": True})
    x = mx.zeros((1, 1, D), mx.float32)
    with pytest.raises(RuntimeError, match="16-bit streams"):
        attention_input(layer, x, [x], None, None, mx.array(0, mx.uint32))
    assert _counted("refused:stream_dtype") == 1


def test_the_step_serves_float16_streams_and_declines_mismatched_weights():
    from mtplx.models.inkling.fast import step

    half, bf16 = mx.zeros((64, 2 * D), mx.float16), mx.zeros((64, 2 * D), mx.bfloat16)
    layer = SimpleNamespace(controllers=[SimpleNamespace(down=half)] * 4)
    assert step.expert_input_reason(layer, True, mx.float16) is None
    assert step.expert_input_reason(layer, False, mx.float16) is None
    assert step.expert_input_reason(layer, True, mx.bfloat16) == "weight_dtype"   # the down weights' type
    assert step.expert_input_reason(layer, False, mx.float32) == "stream_dtype"
    assert step.expert_input_reason(SimpleNamespace(controllers=[SimpleNamespace(down=bf16)] * 4), True,
                                    mx.bfloat16) is None


def test_the_step_declines_stage_d_on_a_float32_stream_and_serves_the_reference(monkeypatch):
    from mtplx.models.inkling.fast import step
    from mtplx.models.inkling.fast.routes import fast_route_counts

    served = []
    monkeypatch.setattr(step, "expert_input", lambda *args: served.append(args) or "reference")
    layer = SimpleNamespace(controllers=[None] * 4)
    before = fast_route_counts().get("fallback.d.stream_dtype", 0)
    out = step.fused_expert_input(layer, mx.zeros((1, 1, D), mx.float32), [mx.zeros((1, 1, D), mx.float32)], None,
                                  None)
    assert out == "reference" and len(served) == 1
    assert fast_route_counts().get("fallback.d.stream_dtype", 0) == before + 1
