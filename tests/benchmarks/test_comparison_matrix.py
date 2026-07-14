from __future__ import annotations

import json
from pathlib import Path

from mtplx.benchmarks.runners.comparison_matrix import (
    COMPARISON_LANES,
    SAMPLER_PROFILES,
    lane_specs,
    load_pins,
    run_comparison_matrix,
    sampler_profile,
    validate_comparison_matrix,
    write_comparison_matrix,
)


def test_pins_inventory_documents_nvfp4_modelopt_checkpoint():
    pins = load_pins(Path("benchmarks/inventory/pins.json"))
    assert pins["hardware"]["model"] == "MacBook Pro M5 Pro"
    assert pins["hardware"]["memory_gb"] == 64
    assert "mlx" in pins["forks"]
    assert "mtplx" in pins["forks"]
    model = pins["models"]["nvidia/Qwen3.6-27B-NVFP4"]
    assert model["tensor_count"] == 2194
    assert model["mtp_tensor_count"] == 15
    assert model["format"] == "modelopt"
    assert ".weight_scale_2" in model["tensor_suffixes"]
    assert (
        pins["models"]["Brooooooklyn/Qwen3.6-27B-NVFP4-mlx"]["role"]
        == "reference_mlx_artifact"
    )


def test_lane_specs_cover_affine_and_nvfp4_mtp_depths():
    specs = lane_specs()
    assert len(specs) == len(COMPARISON_LANES)
    names = {spec.lane for spec in specs}
    assert "affine_ar" in names
    assert "nvfp4_mtp_d3" in names


def test_sampler_profiles_include_blog_parity_and_mtplx_optimized():
    blog = sampler_profile("blog-parity")
    mtplx = sampler_profile("mtplx-optimized")
    assert blog["temperature"] == 1.0
    assert blog["presence_penalty"] == 1.5
    assert mtplx["temperature"] == 0.6
    assert mtplx["draft_temperature"] == 0.7
    assert set(SAMPLER_PROFILES) == {"blog-parity", "mtplx-optimized"}


def test_run_comparison_matrix_manifest_is_schema_valid(tmp_path):
    payload = run_comparison_matrix(
        affine_model="models/affine",
        nvfp4_model="models/nvfp4",
        sampler_profile_name="blog-parity",
        dry_run=True,
        pins_path=Path("benchmarks/inventory/pins.json"),
    )
    assert payload["action"] == "bench comparison matrix"
    assert payload["valid"] is True
    assert payload["problems"] == []
    assert len(payload["lanes"]) == len(COMPARISON_LANES)
    assert payload["lanes"][0]["error"] == "manifest_only_no_inference"

    out = tmp_path / "matrix.json"
    write_comparison_matrix(out, payload)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert validate_comparison_matrix(reloaded) == []


def test_validate_comparison_matrix_flags_missing_lane_fields():
    payload = {
        "action": "bench comparison matrix",
        "lanes": [{"lane": "affine_ar"}],
    }
    problems = validate_comparison_matrix(payload)
    assert any("missing quantization_family" in item for item in problems)
    assert any("missing lanes:" in item for item in problems)
