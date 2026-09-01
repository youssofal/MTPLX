// SPDX-License-Identifier: Apache-2.0
//
// Minimal nanobind bindings for MTPLX's vendored Qwen4 QSA sparse-GQA
// primitive. Scope is deliberately one kernel plus the ABI canary; the oMLX
// bindings.cpp this is derived from exposes seven unrelated kernel families
// that MTPLX does not vendor.

#include <nanobind/nanobind.h>
// StreamOrDevice is a std::variant over monostate/Stream/Device; without the
// variant caster the "stream" keyword argument does not bind.
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "qwen4_qsa_sparse_gqa.h"

namespace nb = nanobind;
using namespace nb::literals;

#ifndef MTPLX_QSA_BUILD_MLX_VERSION
#define MTPLX_QSA_BUILD_MLX_VERSION "unknown"
#endif
#ifndef MTPLX_QSA_BUILD_NANOBIND_VERSION
#define MTPLX_QSA_BUILD_NANOBIND_VERSION "unknown"
#endif
#ifndef MTPLX_QSA_METAL_LIBRARY
#define MTPLX_QSA_METAL_LIBRARY "mtplx_qsa_kernels"
#endif

NB_MODULE(_ext, m) {
  m.doc() = "MTPLX Qwen4 QSA sparse-GQA Metal kernel (vendored from oMLX)";

  // ABI canary: when the extension is built with a nanobind whose ABI tag
  // differs from the one the mlx wheel was built with, the NB_DOMAIN is
  // isolated and every mx.array argument is rejected with "incompatible
  // function arguments" (oMLX issue #2139). The Python wrapper calls this
  // probe once at import and disables the lane when it fails.
  m.def(
      "abi_probe",
      [](const mlx::core::array &a) { return static_cast<int64_t>(a.size()); },
      "a"_a);

  // Build receipts. The primitive links MLX's private C++ ABI, so ANY wheel
  // change needs a rebuild — which is why the native build pins mlx exactly.
  // The Python wrapper compares BUILT_AGAINST_MLX against the imported
  // mlx.core.__version__ during its readiness check: a mismatch warns once
  // and disables the direct lane rather than dispatching a .so that will
  // mis-read MLX's structs.
  m.attr("BUILT_AGAINST_MLX") = MTPLX_QSA_BUILD_MLX_VERSION;
  m.attr("BUILT_AGAINST_NANOBIND") = MTPLX_QSA_BUILD_NANOBIND_VERSION;
  m.attr("METAL_LIBRARY") = MTPLX_QSA_METAL_LIBRARY;

  m.def(
      "qwen4_qsa_sparse_gqa_attention",
      &mtplx::qsa_kernels::qwen4_qsa_sparse_gqa_attention,
      "queries"_a,
      "keys"_a,
      "values"_a,
      "selected_blocks"_a,
      "scale"_a,
      "q_offset"_a,
      "key_tile"_a = 64,
      "dimension_tile"_a = 64,
      "stream"_a = nb::none());
}
