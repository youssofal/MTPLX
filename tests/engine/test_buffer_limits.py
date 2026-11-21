"""MLX's command-buffer size rule: MLX's own from a prefill block until decode begins, lifted in decode.

The native module reads and sets MLX's two limits after proving where they
live; ``mtplx/engine/buffer_limits.py`` lifts the size limit for a runtime that
asks and gives MLX's own back for every prefill block.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx import attention_context
from mtplx.engine import buffer_limits, runtime

pytestmark = pytest.mark.skipif(
    not runtime.available(), reason=f"native engine unavailable: {runtime.unavailable()}"
)


def _limits():
    return runtime.native_module().command_buffer_limits()


@pytest.fixture(autouse=True)
def _mlx_rule_restored(monkeypatch):
    for name in ("MLX_MAX_MB_PER_BUFFER", "MTPLX_MLX_COMMAND_BUFFER_MB", buffer_limits.ROLLBACK_ENV):
        monkeypatch.delenv(name, raising=False)
    mx.eval(mx.zeros((4,)))  # MLX's device exists
    before = _limits()
    yield before
    buffer_limits.disable()
    buffer_limits._STATE.update(reason="not_asked", prefill_depth=0, lifted=False)
    assert _limits()[:2] == before[:2]


def test_the_place_of_the_two_limits_is_proven_and_a_write_is_what_mlx_reads(_mlx_rule_restored):
    ops, mb, settable = _mlx_rule_restored
    assert settable and ops > 0 and mb > 0
    native = runtime.native_module().command_buffer_limits
    assert native(0, mb + 7)[:2] == (ops, mb)  # the values before the call
    assert _limits()[:2] == (ops, mb + 7)
    assert native(ops + 3, 0)[:2] == (ops, mb + 7)
    assert _limits()[:2] == (ops + 3, mb + 7)
    native(ops, mb)
    assert _limits()[:2] == (ops, mb)


def test_the_size_limit_is_mlx_own_from_a_prefill_block_until_decode_begins(_mlx_rule_restored):
    ops, mb, _ = _mlx_rule_restored
    report = buffer_limits.enable("test")
    assert report["lifted_outside_prefill"] and report["mlx_mb"] == mb
    assert _limits()[:2] == (ops, mb)  # a runtime's first work is a prefill: nothing is lifted yet
    with attention_context.attention_phase("decode_verify"):
        assert _limits()[:2] == (ops, buffer_limits.LIFTED_MB)  # the count limit is never touched
    assert _limits()[:2] == (ops, buffer_limits.LIFTED_MB)  # and it stays lifted between decode blocks
    with attention_context.attention_phase("prefill"):
        assert _limits()[:2] == (ops, mb)
        with attention_context.attention_phase("prefill"):
            assert _limits()[:2] == (ops, mb)
        assert _limits()[:2] == (ops, mb)  # still inside the outer block
        with attention_context.attention_phase("decode_verify"):
            assert _limits()[:2] == (ops, mb)  # a verify inside a prefill block keeps prefill's rule
    # The chunk a prefill block built is evaluated after the block: MLX's rule must still hold there.
    assert _limits()[:2] == (ops, mb)
    with attention_context.attention_phase("ar_decode"):
        assert _limits()[:2] == (ops, buffer_limits.LIFTED_MB)
    with pytest.raises(RuntimeError):
        with attention_context.attention_phase("prefill"):
            raise RuntimeError("a prefill that fails leaves MLX's rule, and the next decode block lifts it")
    assert _limits()[:2] == (ops, mb)
    with attention_context.attention_phase("decode_verify"):
        assert _limits()[:2] == (ops, buffer_limits.LIFTED_MB)
    assert buffer_limits.report()["lifted_now"]
    buffer_limits.disable()
    assert _limits()[:2] == (ops, mb)


def test_a_runtime_that_loads_inside_a_prefill_block_keeps_mlx_rule_until_decode(_mlx_rule_restored):
    ops, mb, _ = _mlx_rule_restored
    with attention_context.attention_phase("prefill"):
        buffer_limits.enable("test")
        assert _limits()[:2] == (ops, mb)
    assert _limits()[:2] == (ops, mb)
    with attention_context.attention_phase("decode_verify"):
        assert _limits()[:2] == (ops, buffer_limits.LIFTED_MB)


@pytest.mark.parametrize(
    "name, value",
    [("MLX_MAX_MB_PER_BUFFER", "200"), ("MTPLX_MLX_COMMAND_BUFFER_MB", "1024"), (buffer_limits.ROLLBACK_ENV, "mlx")],
)
def test_the_operator_s_own_limit_and_the_rollback_leave_mlx_alone(_mlx_rule_restored, monkeypatch, name, value):
    ops, mb, _ = _mlx_rule_restored
    monkeypatch.setenv(name, value)
    report = buffer_limits.enable("test")
    assert not report["lifted_outside_prefill"]
    assert _limits()[:2] == (ops, mb)
    with attention_context.attention_phase("prefill"):
        assert _limits()[:2] == (ops, mb)
    assert _limits()[:2] == (ops, mb)


def test_values_do_not_depend_on_the_rule(_mlx_rule_restored):
    big = mx.random.normal((1 << 22,), key=mx.random.key(3))  # 4M elements a bind
    small = mx.random.normal((64,), key=mx.random.key(4))

    def step():
        out = small
        for i in range(40):
            out = out + mx.sum(big[i : i + 64]) * 0.001
        mx.eval(out)
        return out

    want = step()
    buffer_limits.enable("test")
    with attention_context.attention_phase("decode_verify"):
        got = step()
    assert mx.array_equal(want, got).item()
