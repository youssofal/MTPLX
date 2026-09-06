// SPDX-License-Identifier: Apache-2.0
//
// MLX-free cache-aware CPU sidecar handoff.  The stock Python owner remains
// the only persistent row cache.  This component owns only bounded immutable
// handoff/completion bytes and a construction-time fixed I/O pool.

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "host_provider.h"
#include "request_state.h"
#include "sidecar_producer.h"

namespace mtplx_native::ple_cpu_rows {

namespace host = mtplx_native::host_provider;

constexpr std::uint8_t kCachedHitBit = 0x80;
constexpr std::uint8_t kCachedSourceMask = 0x3f;
constexpr std::size_t kCachedDefaultIoWorkers = 8;
constexpr std::size_t kCachedMaxIoWorkers = 16;

// The source byte is either kCachedHitBit | compact hit index or a compact
// miss index.  hit_packed contains only the compact hit rows; miss_ids are
// unique and are read by the native fixed pool.  All arrays are fixed-bounded
// so Python cannot make the queued handoff grow without limit.
struct CachedRowHandoff final {
  std::array<std::uint8_t, host::kRowsPerWindow> source{};
  std::array<host::PackedRow, host::kRowsPerWindow> hit_packed{};
  std::array<std::uint32_t, host::kRowsPerWindow> miss_ids{};
  std::uint8_t hit_count = 0;
  std::uint8_t miss_count = 0;
};

// A completion is tied to the admitted job ticket.  The owner thread drains
// this record and publishes it into the stock Python LRU; worker code never
// touches Python cache objects.
struct CachedCompletion final {
  std::uint64_t ticket = 0;
  std::array<std::uint32_t, host::kRowsPerWindow> miss_ids{};
  std::array<host::PackedRow, host::kRowsPerWindow> payloads{};
  std::uint8_t count = 0;
};

// Test-only reader seam.  Production installation wraps the immutable
// RawSidecarBatchReader; CPU tests inject a counting reader without adding
// counters or branches to the production read path.
class CachedRowReader {
 public:
  virtual ~CachedRowReader() = default;
  virtual host::PackedRow read_one(std::uint32_t row_id) const = 0;
};

class CachedSidecarProducer;

class CachedSidecarJob final {
 public:
  ~CachedSidecarJob();

  CachedSidecarJob(const CachedSidecarJob&) = delete;
  CachedSidecarJob& operator=(const CachedSidecarJob&) = delete;

  // Computes no hash: row IDs and the typed hit/miss scatter map were copied
  // at admission.  The returned packed bytes preserve all 64 source slots.
  host::PackedRows run();

  // Marks MLX/output publication complete.  If misses produced a completion,
  // the permit remains held until CachedSidecarProducer::drain_completed().
  void release_permit() noexcept;

  // Explicitly abandons an admitted job before publication.  Abandonment
  // drops any queued completion ownership and returns its permit once.
  void abandon() noexcept;

  std::uint64_t ticket() const noexcept { return ticket_; }

 private:
  friend class CachedSidecarProducer;

  struct State;

  CachedSidecarJob(
      std::shared_ptr<CachedSidecarProducer> producer,
      CachedRowHandoff handoff,
      std::shared_ptr<PermitPool> permits,
      std::uint64_t ticket);

  const std::shared_ptr<CachedSidecarProducer> producer_;
  const CachedRowHandoff handoff_;
  const std::uint64_t ticket_;
  const std::shared_ptr<State> state_;
};

class CachedSidecarProducer final
    : public std::enable_shared_from_this<CachedSidecarProducer> {
 public:
  ~CachedSidecarProducer();

  static std::shared_ptr<CachedSidecarProducer> install(
      int descriptor,
      host::SidecarLayout layout,
      const std::array<std::int64_t, host::kNgramSize>& multipliers,
      const std::array<std::int64_t, host::kNgramHeads>& sizes,
      const std::array<std::int64_t, host::kNgramHeads>& offsets,
      std::int64_t eos,
      std::size_t io_workers = kCachedDefaultIoWorkers);

  // CPU-test seam: the descriptor/layout still pass the normal construction
  // validation through SidecarProducer; only row reads are instrumented.
  static std::shared_ptr<CachedSidecarProducer> install_for_test(
      int descriptor,
      host::SidecarLayout layout,
      const std::array<std::int64_t, host::kNgramSize>& multipliers,
      const std::array<std::int64_t, host::kNgramHeads>& sizes,
      const std::array<std::int64_t, host::kNgramHeads>& offsets,
      std::int64_t eos,
      std::shared_ptr<const CachedRowReader> reader,
      std::size_t io_workers = kCachedDefaultIoWorkers);

  CachedSidecarProducer(const CachedSidecarProducer&) = delete;
  CachedSidecarProducer& operator=(const CachedSidecarProducer&) = delete;

  // Construction-bound plan reuse for the synchronous owner-thread hash.
  SidecarRowIds compute_row_ids(const SidecarJobInput& input) const;

  std::uint64_t row_count() const noexcept;
  std::size_t io_workers() const noexcept { return io_workers_; }

  // Typed admission validates source/count/index bounds and unique miss IDs
  // once.  No corresponding checks occur in worker read loops.
  std::shared_ptr<CachedSidecarJob> make_job(
      const CachedRowHandoff& handoff,
      const std::shared_ptr<PermitPool>& permits);

  // Draining is the owner-thread completion boundary.  For a miss-bearing
  // job it is what permits the job's two-request slot to be reclaimed after
  // output publication.  Abandoned entries are discarded here.
  std::vector<CachedCompletion> drain_completed();

 private:
  friend class CachedSidecarJob;

  class IoPool;
  struct CompletionEntry;

  CachedSidecarProducer(
      std::shared_ptr<const SidecarProducer> base,
      std::shared_ptr<const CachedRowReader> reader,
      std::size_t io_workers);

  static void validate_handoff(
      const CachedRowHandoff& handoff,
      std::uint64_t row_count);

  void enqueue_completion(
      const std::shared_ptr<CachedSidecarJob::State>& state,
      CachedCompletion completion);

  std::shared_ptr<const SidecarProducer> base_;
  std::shared_ptr<const CachedRowReader> reader_;
  std::unique_ptr<IoPool> io_pool_;
  std::size_t io_workers_ = 0;
  std::atomic<std::uint64_t> next_ticket_{1};
  mutable std::mutex completion_mutex_;
  std::unique_ptr<std::deque<CompletionEntry>> completions_;
};

}  // namespace mtplx_native::ple_cpu_rows
