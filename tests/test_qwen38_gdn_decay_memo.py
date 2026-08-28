from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mtplx import gdn_capture


def _gdn() -> SimpleNamespace:
    return SimpleNamespace(
        A_log=mx.array([-1.5, -0.5, 0.25], dtype=mx.float32),
        dt_bias=mx.array([0.1, -0.2, 0.3], dtype=mx.float32),
    )


def test_row18_memoized_decay_gate_is_exact_at_fixed_d3_width() -> None:
    from mlx_lm.models.gated_delta import compute_g

    gdn = _gdn()
    a = mx.array(
        [[[0.2, -0.1, 0.4], [0.7, -0.6, 0.5], [-0.2, 0.9, -0.8], [0.0, 0.1, 0.2]]],
        dtype=mx.bfloat16,
    )
    expected = compute_g(gdn.A_log, a, gdn.dt_bias)
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(linear_attn=gdn)])
        )
    )
    gdn_capture.configure_qwen38_row18_gdn_decay_memo(model, active=True)
    actual = gdn._mtplx_compute_g(a)
    mx.eval(expected, actual)

    assert mx.array_equal(expected, actual).item()


def test_row18_configuration_materializes_and_toggles_target_layers() -> None:
    gdns = [_gdn(), _gdn()]
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(linear_attn=gdn) for gdn in gdns]
            )
        )
    )

    active = gdn_capture.configure_qwen38_row18_gdn_decay_memo(model, active=True)
    inactive = gdn_capture.configure_qwen38_row18_gdn_decay_memo(model, active=False)

    assert active == {"configured_modules": 2, "active_modules": 2}
    assert inactive == {"configured_modules": 2, "active_modules": 0}
    assert all(callable(gdn._mtplx_compute_g) for gdn in gdns)


def test_row48_capture_entrypoint_is_prebound_at_installation() -> None:
    runtime = SimpleNamespace()

    active = gdn_capture.configure_qwen38_row48_capture(runtime, active=True)
    candidate_route = runtime._forward_ar_capture_gdn.keywords["boundary_route"]
    inactive = gdn_capture.configure_qwen38_row48_capture(runtime, active=False)
    stock_route = runtime._forward_ar_capture_gdn.keywords["boundary_route"]

    assert active == {"active": 1, "construction_bound": 1}
    assert inactive == {"active": 0, "construction_bound": 1}
    assert candidate_route is gdn_capture._ROW48_BOUNDARY_ROUTE
    assert stock_route is gdn_capture._STOCK_BOUNDARY_ROUTE
