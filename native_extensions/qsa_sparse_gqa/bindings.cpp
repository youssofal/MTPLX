// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "sparse_gqa/qsa_sparse_gqa.h"
#include "sparse_gqa/qsa_sparse_gqa_decode.h"

#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "MTPLX native QSA direct-index sparse-GQA attention";

  m.def("qsa_sparse_gqa_attention", &mtplx_native::qsa_sparse_gqa_attention,
        "queries"_a, "keys"_a, "values"_a, "selected_blocks"_a, "scale"_a,
        "q_offset"_a, "key_length"_a = -1, "key_tile"_a = 128,
        "dimension_tile"_a = 32, nb::kw_only(), "stream"_a = nb::none(),
        "Direct-index sparse GQA attention over chronological QSA block ids.");

  m.def("qsa_sparse_gqa_unsupported_reason",
        &mtplx_native::qsa_sparse_gqa_unsupported_reason, "queries"_a, "keys"_a,
        "values"_a, "selected_blocks"_a, "scale"_a, "q_offset"_a,
        "key_length"_a = -1, "key_tile"_a = 128, "dimension_tile"_a = 32,
        nb::kw_only(), "stream"_a = nb::none(),
        "Empty string when the call is on contract; otherwise the reason.");

  m.def(
      "qsa_sparse_gqa_decode", &mtplx_native::qsa_sparse_gqa_decode,
      "queries"_a, "keys"_a, "values"_a, "selected_blocks"_a,
      "query_offset"_a, "scale"_a, "key_length"_a = -1, "key_tile"_a = 128,
      "dimension_tile"_a = 32, "key_splits"_a = 8, nb::kw_only(),
      "stream"_a = nb::none(),
      "Split-K direct-index sparse GQA attention for the decode geometries.");

  m.def(
      "qsa_sparse_gqa_decode_unsupported_reason",
      &mtplx_native::qsa_sparse_gqa_decode_unsupported_reason, "queries"_a,
      "keys"_a, "values"_a, "selected_blocks"_a, "query_offset"_a, "scale"_a,
      "key_length"_a = -1, "key_tile"_a = 128, "dimension_tile"_a = 32,
      "key_splits"_a = 8, nb::kw_only(), "stream"_a = nb::none(),
      "Empty string when the decode call is on contract; else the reason.");

  m.def(
      "qsa_sparse_gqa_decode_split_geometry",
      [](int selected_tokens, int key_tile, int key_splits) {
        int n_tiles = 0, tiles_per_split = 0, n_splits = 0;
        mtplx_native::qsa_sparse_gqa_decode_split_geometry(
            selected_tokens, key_tile, key_splits, &n_tiles, &tiles_per_split,
            &n_splits);
        return std::make_tuple(n_tiles, tiles_per_split, n_splits);
      },
      "selected_tokens"_a, "key_tile"_a, "key_splits"_a,
      "(n_tiles, tiles_per_split, n_splits) for the split-K decode grid.");
}
