"""Hardware admission must precede tensor access or Metal compilation."""

import importlib

import pytest


@pytest.mark.parametrize("module_name,function_name,counter", [
    ("sdpa_nax_flash", "sdpa_nax_flash", "nax_flash_bail_counts"),
    ("sdpa_nax_flash_dsplit", "sdpa_nax_flash_dsplit", "nax_flash_dsplit_bail_counts"),
    ("sdpa_nax_tile", "sdpa_nax_tile", "nax_tile_bail_counts"),
])
def test_unsupported_nax_attention_declines_before_touching_tensors(
    monkeypatch, module_name, function_name, counter
):
    module = importlib.import_module("mtplx.kernels." + module_name)
    monkeypatch.setattr(module.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(module, "nax_available", lambda: False)
    counts = getattr(module, counter)
    before = counts.get("gpu_family_or_os", 0)
    assert getattr(module, function_name)(
        queries=None, keys=None, values=None, offset=1, scale=.0625
    ) is None
    assert counts["gpu_family_or_os"] == before + 1


@pytest.mark.parametrize("module_name,function_name,counter", [
    ("sdpa_nax_flash", "sdpa_nax_flash", "nax_flash_bail_counts"),
    ("sdpa_nax_flash_dsplit", "sdpa_nax_flash_dsplit", "nax_flash_dsplit_bail_counts"),
    ("sdpa_nax_tile", "sdpa_nax_tile", "nax_tile_bail_counts"),
])
def test_lazy_build_failure_is_caught_once_and_remembered(
    monkeypatch, module_name, function_name, counter
):
    """Issue #461: a Metal library that fails to build only surfaces when the
    lazy outputs are evaluated. The first dispatch of a shape is settled
    inside the guard, and a shape that failed never dispatches again."""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal required")
    module = importlib.import_module("mtplx.kernels." + module_name)
    monkeypatch.setattr(module, "nax_available", lambda: True)
    module._probe_ok.clear()
    module._probe_failed.clear()
    calls = {"kernel": 0, "eval": 0}

    def fake_kernel(**kwargs):
        calls["kernel"] += 1
        shapes = kwargs["output_shapes"]
        dtypes = kwargs["output_dtypes"]
        return tuple(mx.zeros(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes))

    def failing_eval(*_arrays):
        calls["eval"] += 1
        raise RuntimeError("[metal::Device] Unable to build metal library from source")

    kernel_getter = {
        "sdpa_nax_flash": "_nax_flash_kernel",
        "sdpa_nax_flash_dsplit": "_nax_flash_dsplit_kernel",
        "sdpa_nax_tile": "_nax_tile_kernel",
    }[module_name]
    monkeypatch.setattr(module, kernel_getter, lambda: fake_kernel)
    monkeypatch.setattr(module, "_paged_reduce_kernel", lambda: fake_kernel)
    monkeypatch.setattr(module.mx, "eval", failing_eval)
    q = mx.zeros((1, 8, 4, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 8192, 256), dtype=mx.bfloat16)
    v = mx.zeros((1, 2, 8192, 256), dtype=mx.bfloat16)
    counts = getattr(module, counter)
    fn = getattr(module, function_name)
    assert fn(queries=q, keys=k, values=v, offset=4096, scale=.0625) is None
    assert calls == {"kernel": 1, "eval": 1}
    assert any(reason.startswith("dispatch_failed") for reason in counts)
    before = counts.get("build_failed", 0)
    assert fn(queries=q, keys=k, values=v, offset=4096, scale=.0625) is None
    assert calls["kernel"] == 1, "a failed shape must not dispatch again"
    assert counts["build_failed"] == before + 1
    module._probe_failed.clear()
