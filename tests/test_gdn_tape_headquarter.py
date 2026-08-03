"""Bit-exactness contract for the headquarter tape-capture kernel.

The headquarter kernel is an execution-layout change only: for every input it
must produce BIT-EQUAL y / final_state / tape versus the incumbent TGY tape
kernel. Runs at the Qwen3.6-27B GDN geometry; skipped without Metal.
"""

from types import SimpleNamespace

import pytest

import mlx.core as mx

from mtplx.gdn_capture import _linear_gated_delta_from_conv_tape_capture
from mtplx.kernels.gdn_tape_headquarter import headquarter_tape_capture

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal"
)


def _gdn():
    gdn = SimpleNamespace(
        conv_dim=10240,
        head_k_dim=128,
        head_v_dim=128,
        num_k_heads=16,
        num_v_heads=48,
        key_dim=2048,
    )
    gdn.A_log = mx.log(mx.random.uniform(low=0.5, high=8.0, shape=(gdn.num_v_heads,)))
    gdn.dt_bias = mx.ones(gdn.num_v_heads) * 0.5
    mx.eval(gdn.A_log, gdn.dt_bias)
    return gdn


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_headquarter_matches_incumbent_bitwise(T, seed, monkeypatch):
    from mlx_lm.models.gated_delta import compute_g

    # The reference arm routes through the env-gated wrapper: a stray
    # MTPLX_LINEAR_GDN_TAPE_IMPL=headquarter in the invoking shell would turn
    # this into headquarter-vs-headquarter and pass vacuously.
    monkeypatch.delenv("MTPLX_LINEAR_GDN_TAPE_IMPL", raising=False)
    mx.random.seed(0)
    gdn = _gdn()
    key = mx.random.key(1000 * T + seed)
    ks = mx.random.split(key, 4)
    conv_out = mx.random.normal((1, T, gdn.conv_dim), key=ks[0]).astype(mx.bfloat16)
    a = (mx.random.normal((1, T, gdn.num_v_heads), key=ks[1]) * 0.5).astype(mx.bfloat16)
    b = (mx.random.normal((1, T, gdn.num_v_heads), key=ks[2]) * 0.5).astype(mx.bfloat16)
    state = (
        mx.random.normal((1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), key=ks[3])
        * 0.5
    ).astype(mx.float32)
    beta = mx.sigmoid(b)
    g = compute_g(gdn.A_log, a, gdn.dt_bias)

    ref = _linear_gated_delta_from_conv_tape_capture(conv_out, g, beta, state, gdn)
    cand = headquarter_tape_capture(conv_out, g, beta, state, gdn)
    assert ref is not None and cand is not None
    for name, r, c in zip(("y", "final_state", "tape"), ref, cand):
        mx.eval(r, c)
        assert bool(mx.array_equal(r, c).item()), f"{name} diverged at T={T} seed={seed}"


def test_headquarter_fail_closed_on_bad_geometry():
    gdn = _gdn()
    gdn.head_k_dim = 100  # not divisible by 32 -> wrapper must decline
    conv_out = mx.zeros((1, 1, gdn.conv_dim), dtype=mx.bfloat16)
    g = mx.zeros((1, 1, gdn.num_v_heads))
    beta = mx.zeros((1, 1, gdn.num_v_heads), dtype=mx.bfloat16)
    state = mx.zeros((1, gdn.num_v_heads, gdn.head_v_dim, 100), dtype=mx.float32)
    assert headquarter_tape_capture(conv_out, g, beta, state, gdn) is None
