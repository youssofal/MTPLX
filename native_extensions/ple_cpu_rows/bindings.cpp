// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

#include "cached_sidecar_primitive.h"
#include "ple_cpu_rows.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using SourceArray = nb::ndarray<const std::uint8_t,
                                nb::numpy,
                                nb::c_contig,
                                nb::shape<64>>;
using HitArray = nb::ndarray<const std::uint8_t,
                             nb::numpy,
                             nb::c_contig,
                             nb::ndim<2>>;
using MissArray = nb::ndarray<const std::uint32_t,
                              nb::numpy,
                              nb::c_contig,
                              nb::ndim<1>>;
using RowIdArray = nb::ndarray<const std::uint32_t,
                               nb::numpy,
                               nb::c_contig,
                               nb::shape<64>>;
using PackedCompletionArray = nb::ndarray<const std::uint8_t,
                                           nb::numpy,
                                           nb::c_contig,
                                           nb::ndim<2>>;

using OwnedRowIds = std::array<std::uint32_t, 64>;

void delete_owned_row_ids(void* pointer) noexcept {
  delete static_cast<OwnedRowIds*>(pointer);
}

void delete_owned_packed(void* pointer) noexcept {
  delete static_cast<std::vector<std::uint8_t>*>(pointer);
}

RowIdArray owned_row_ids(const mtplx_native::ple_cpu_rows::SidecarRowIds& rows) {
  auto* owner = new OwnedRowIds(rows);
  nb::capsule capsule(owner, delete_owned_row_ids);
  return RowIdArray(owner->data(), {64}, capsule);
}

PackedCompletionArray owned_packed_completion(
    const mtplx_native::ple_cpu_rows::CachedCompletion& completion) {
  auto* owner = new std::vector<std::uint8_t>(
      static_cast<std::size_t>(completion.count) * 100);
  for (std::size_t index = 0; index < completion.count; ++index) {
    std::memcpy(owner->data() + index * 100,
                completion.payloads[index].data(),
                100);
  }
  nb::capsule capsule(owner, delete_owned_packed);
  return PackedCompletionArray(
      owner->data(),
      {static_cast<std::size_t>(completion.count), 100},
      capsule);
}

mtplx_native::ple_cpu_rows::CachedRowHandoff cached_handoff_from_arrays(
    const SourceArray& source,
    const HitArray& hits,
    const MissArray& misses) {
  namespace rows = mtplx_native::ple_cpu_rows;
  if (hits.ndim() != 2 || hits.shape(1) != 100 || hits.shape(0) > 64) {
    throw nb::value_error(
        "cached sidecar hits must be contiguous uint8 with shape (H, 100), H<=64");
  }
  if (misses.ndim() != 1 || misses.shape(0) > 64) {
    throw nb::value_error(
        "cached sidecar misses must be contiguous uint32 with shape (M,), M<=64");
  }

  rows::CachedRowHandoff handoff{};
  std::memcpy(handoff.source.data(), source.data(), handoff.source.size());
  handoff.hit_count = static_cast<std::uint8_t>(hits.shape(0));
  handoff.miss_count = static_cast<std::uint8_t>(misses.shape(0));
  if (hits.size() != 0) {
    std::memcpy(handoff.hit_packed.data(),
                hits.data(),
                hits.size() * sizeof(std::uint8_t));
  }
  if (misses.size() != 0) {
    std::memcpy(handoff.miss_ids.data(),
                misses.data(),
                misses.size() * sizeof(std::uint32_t));
  }
  return handoff;
}

mtplx_native::ple_cpu_rows::PackedPayload payload_from_bytes(
    const nb::object& object) {
  if (!nb::isinstance<nb::bytes>(object)) {
    throw nb::type_error("payload must be exactly 6400 bytes");
  }
  // Nanobind's standard-string caster only accepts Python unicode under
  // nanobind 2.15; it rejects bytes before the payload can reach the native
  // primitive.  Keep the checked bytes handle borrowed and copy its binary
  // storage, including embedded NUL bytes, into the immutable request payload.
  const nb::bytes raw = nb::borrow<nb::bytes>(object);
  if (raw.size() != mtplx_native::ple_cpu_rows::kPayloadBytes) {
    throw nb::value_error("payload must contain exactly 6400 bytes");
  }
  mtplx_native::ple_cpu_rows::PackedPayload payload{};
  std::memcpy(payload.data(), raw.c_str(), raw.size());
  return payload;
}

}  // namespace

NB_MODULE(_ext, m) {
  m.doc() =
      "CPU-stream MLX-owned PLE row staging primitive; dequantization stays "
      "in ordinary MLX operations";

  m.def(
      "make_cpu_rows",
      [](const nb::object& payload, int delay_ms, bool fail, bool cancel) {
        return mtplx_native::ple_cpu_rows::make_cpu_rows(
            payload_from_bytes(payload), delay_ms, fail, cancel);
      },
      "payload"_a,
      "delay_ms"_a = 0,
      "fail"_a = false,
      "cancel"_a = false,
      "Create fresh [64,20] U32 and two [64,5] BF16 MLX-owned planes."
      " Submit async_eval(planes) before constructing a GPU consumer.");

  // The provider object has no mutable Python-side state.  Its constructor
  // duplicates/validates the descriptor and binds the complete hash plan;
  // each subsequent call supplies only one copied M4 token snapshot.
  nb::class_<mtplx_native::ple_cpu_rows::SidecarProducer>(m,
                                                           "SidecarProducer")
      .def_prop_ro("row_count",
                   &mtplx_native::ple_cpu_rows::SidecarProducer::row_count);

  nb::class_<mtplx_native::ple_cpu_rows::CachedSidecarProducer>(
      m, "CachedSidecarProducer")
      .def_prop_ro(
          "row_count",
          &mtplx_native::ple_cpu_rows::CachedSidecarProducer::row_count)
      .def_prop_ro(
          "io_workers",
          &mtplx_native::ple_cpu_rows::CachedSidecarProducer::io_workers);

  m.def(
      "install_sidecar_provider",
      [](int descriptor,
         std::uint64_t row_count,
         std::uint64_t weights_offset,
         std::uint64_t weights_length,
         std::uint64_t scales_offset,
         std::uint64_t scales_length,
         std::uint64_t biases_offset,
         std::uint64_t biases_length,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kNgramSize>& multipliers,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kNgramHeads>& sizes,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kNgramHeads>& offsets,
         std::int64_t eos) {
        mtplx_native::host_provider::SidecarLayout layout{};
        layout.row_count = row_count;
        layout.weights = {weights_offset, weights_length, 80};
        layout.scales = {scales_offset, scales_length, 10};
        layout.biases = {biases_offset, biases_length, 10};
        return mtplx_native::ple_cpu_rows::SidecarProducer::install(
            descriptor, layout, multipliers, sizes, offsets, eos);
      },
      "descriptor"_a,
      "row_count"_a,
      "weights_offset"_a,
      "weights_length"_a,
      "scales_offset"_a,
      "scales_length"_a,
      "biases_offset"_a,
      "biases_length"_a,
      "multipliers"_a,
      "sizes"_a,
      "offsets"_a,
        "eos"_a,
        "Install the immutable real-sidecar plan and duplicated reader.");

  m.def(
      "install_cached_sidecar_provider",
      [](int descriptor,
         std::uint64_t row_count,
         std::uint64_t weights_offset,
         std::uint64_t weights_length,
         std::uint64_t scales_offset,
         std::uint64_t scales_length,
         std::uint64_t biases_offset,
         std::uint64_t biases_length,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kNgramSize>& multipliers,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kNgramHeads>& sizes,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kNgramHeads>& offsets,
         std::int64_t eos,
         std::size_t io_workers) {
        mtplx_native::host_provider::SidecarLayout layout{};
        layout.row_count = row_count;
        layout.weights = {weights_offset, weights_length, 80};
        layout.scales = {scales_offset, scales_length, 10};
        layout.biases = {biases_offset, biases_length, 10};
        return mtplx_native::ple_cpu_rows::CachedSidecarProducer::install(
            descriptor,
            layout,
            multipliers,
            sizes,
            offsets,
            eos,
            io_workers);
      },
      "descriptor"_a,
      "row_count"_a,
      "weights_offset"_a,
      "weights_length"_a,
      "scales_offset"_a,
      "scales_length"_a,
      "biases_offset"_a,
      "biases_length"_a,
      "multipliers"_a,
      "sizes"_a,
      "offsets"_a,
      "eos"_a,
      "io_workers"_a =
          mtplx_native::ple_cpu_rows::kCachedDefaultIoWorkers,
      "Install the immutable cache-aware sidecar plan and fixed I/O pool.");

  m.def(
      "compute_cached_row_ids",
      [](const std::shared_ptr<
             mtplx_native::ple_cpu_rows::CachedSidecarProducer>& producer,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kHistoryRows>& previous,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kInputRows>& ids) {
        if (producer == nullptr) {
          throw nb::value_error("cached sidecar producer is null");
        }
        mtplx_native::ple_cpu_rows::SidecarJobInput input{previous, ids};
        return owned_row_ids(producer->compute_row_ids(input));
      },
      "provider"_a,
      "previous"_a,
      "current"_a,
      "Compute one immutable uint32[64] fixed-M4 row-ID snapshot.");

  m.def(
      "make_cached_sidecar_rows",
      [](const std::shared_ptr<
             mtplx_native::ple_cpu_rows::CachedSidecarProducer>& producer,
         const SourceArray& source,
         const HitArray& hits,
         const MissArray& misses) {
        const auto handoff = cached_handoff_from_arrays(source, hits, misses);
        const auto submission =
            mtplx_native::ple_cpu_rows::make_cached_sidecar_rows(
                producer, handoff);
        nb::object ticket = submission.ticket.has_value()
                                ? nb::cast(*submission.ticket)
                                : nb::none();
        return nb::make_tuple(
            ticket,
            nb::make_tuple(std::get<0>(submission.arrays),
                           std::get<1>(submission.arrays),
                           std::get<2>(submission.arrays)));
      },
      "provider"_a,
      "source"_a,
      "hits"_a,
      "misses"_a,
      "Copy the fixed handoff and enqueue MLX-owned U32/BF16 planes.");

  m.def(
      "drain_cached_completions",
      [](const std::shared_ptr<
             mtplx_native::ple_cpu_rows::CachedSidecarProducer>& producer) {
        nb::list output;
        for (const auto& completion :
             mtplx_native::ple_cpu_rows::drain_cached_completions(producer)) {
          output.append(nb::make_tuple(
              completion.ticket, owned_packed_completion(completion)));
        }
        return output;
      },
      "provider"_a,
      "Drain owner-thread miss completions as immutable uint8[M,100] arrays.");

  m.def(
      "make_sidecar_rows",
      [](const std::shared_ptr<mtplx_native::ple_cpu_rows::SidecarProducer>&
             producer,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kHistoryRows>& previous,
         const std::array<std::int64_t,
                          mtplx_native::host_provider::kInputRows>& ids) {
        std::shared_ptr<const mtplx_native::ple_cpu_rows::SidecarProducer>
            installed = producer;
        return mtplx_native::ple_cpu_rows::make_sidecar_rows(
            installed, previous, ids);
      },
      "provider"_a,
      "previous"_a,
      "ids"_a,
      "Create fresh sidecar [64,20] U32 and two [64,5] BF16 planes.");
}
