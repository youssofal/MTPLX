"""CPU-only API and source checks for the cached sidecar MLX glue.

The primitive and nanobind surface are intentionally not compiled here.  This
gate checks the exact source contract before the isolated operator build:
three MLX-owned planes, construction-bound CPU stream/permit reuse, copied
fixed handoff inputs, and capsule-owned immutable NumPy outputs.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "native_extensions" / "ple_cpu_rows"
PRIMITIVE = EXT / "cached_sidecar_primitive.cpp"
HEADER = EXT / "cached_sidecar_primitive.h"
BINDINGS = EXT / "bindings.cpp"
CMAKE = EXT / "CMakeLists.txt"
PACKAGE = EXT / "mtplx_native_ple_cpu_rows" / "__init__.py"


def _body(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        char = source[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : position]
    raise AssertionError(f"unterminated body for {marker!r}")


def test_cached_primitive_files_are_staged_and_added_to_cmake():
    assert HEADER.is_file()
    assert PRIMITIVE.is_file()
    cmake = CMAKE.read_text(encoding="utf-8")
    assert "cached_sidecar_primitive.cpp" in cmake
    assert "cached_sidecar_producer.cpp" in cmake
    assert "host_provider.cpp" in cmake


def test_cached_primitive_reuses_cpu_encoder_and_exact_three_plane_contract():
    source = PRIMITIVE.read_text(encoding="utf-8")
    body = _body(source, "void eval_cpu(")
    assert '#include "mlx/backend/cpu/encoder.h"' in source
    assert body.count("set_data(") == 3
    assert body.count("mx::allocator::malloc") == 3
    assert "mx::cpu::get_command_encoder(stream())" in body
    assert "encoder.dispatch" in body
    assert "mx::Shape{64, 20}" in source
    assert source.count("mx::Shape{64, 5}") >= 2
    assert "mx::uint32" in source
    assert source.count("mx::bfloat16") >= 2
    assert "copy_packed_to_planes" in body


def test_cached_primitive_has_no_hidden_event_or_hotpath_sync():
    source = PRIMITIVE.read_text(encoding="utf-8")
    forbidden = (
        r"\bEvent\b",
        r"\bevent\s*\(",
        r"attach_event",
        r"wait_event",
        r"signal_event",
        r"synchronize\s*\(",
        r"mx::eval",
        r"mlx/backend/metal",
        r"objc",
        r"MTL",
        r"std::thread",
        r"condition_variable",
    )
    assert not [pattern for pattern in forbidden if re.search(pattern, source)]
    assert "eval_gpu" in source
    assert "CPU-stream-only" in source


def test_cached_primitive_keeps_job_and_output_descriptors_alive_until_copy():
    source = PRIMITIVE.read_text(encoding="utf-8")
    body = _body(source, "void eval_cpu(")
    assert re.search(r"\[job = std::move\(job\).*weight.*scales.*bias", body, re.DOTALL)
    assert "job->run()" in body
    assert "job->release_permit()" in body
    assert "std::memset" in body
    assert "throw" in body
    assert "array::make_arrays" in source


def test_cached_binding_exposes_exact_additive_api_and_ndarray_contract():
    source = BINDINGS.read_text(encoding="utf-8")
    for name in (
        "CachedSidecarProducer",
        "install_cached_sidecar_provider",
        "compute_cached_row_ids",
        "make_cached_sidecar_rows",
        "drain_cached_completions",
    ):
        assert f'"{name}"' in source
    assert '#include <nanobind/ndarray.h>' in source
    assert "nb::ndarray<const std::uint8_t" in source
    assert "nb::ndarray<const std::uint32_t" in source
    assert "nb::shape<64>" in source
    assert "nb::ndim<2>" in source
    assert "nb::ndim<1>" in source
    assert "nb::capsule" in source
    assert "delete_owned_packed" in source
    assert "std::memcpy(handoff.source.data(), source.data()" in source
    assert "hits.data()" in source
    assert "misses.data()" in source
    assert "submission.ticket.has_value()" in source
    assert "nb::none()" in source
    assert "class_<mtplx_native::ple_cpu_rows::CachedSidecarProducer>" in source
    assert "class_<mtplx_native::ple_cpu_rows::CachedSidecarProducer,\n" not in source


def test_cached_binding_input_validation_is_at_boundary_and_outputs_are_readonly():
    source = BINDINGS.read_text(encoding="utf-8")
    helper = _body(source, "cached_handoff_from_arrays")
    assert "hits.shape(1) != 100" in helper
    assert "hits.shape(0) > 64" in helper
    assert "misses.shape(0) > 64" in helper
    assert "hit_count = static_cast<std::uint8_t>" in helper
    assert "miss_count = static_cast<std::uint8_t>" in helper
    assert "const std::uint8_t" in source
    assert "const std::uint32_t" in source
    assert "owner->data()" in source
    assert "capsule" in source


def test_cached_package_preserves_old_exports_and_adds_new_exports():
    package = PACKAGE.read_text(encoding="utf-8")
    for name in (
        "SidecarProducer",
        "install_sidecar_provider",
        "make_sidecar_rows",
        "make_cpu_rows",
        "CachedSidecarProducer",
        "install_cached_sidecar_provider",
        "compute_cached_row_ids",
        "make_cached_sidecar_rows",
        "drain_cached_completions",
    ):
        assert name in package


def test_cached_primitive_header_keeps_ticket_and_completion_surface_mlxfree_boundary():
    header = HEADER.read_text(encoding="utf-8")
    assert "CachedRowsSubmission" in header
    assert "std::optional<std::uint64_t> ticket" in header
    assert "CachedRowHandoff" in header
    assert "drain_cached_completions" in header
    assert "CachedRowsArrays" in header
