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
