// SPDX-License-Identifier: Apache-2.0

#include "cached_sidecar_producer.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace mtplx_native::ple_cpu_rows {

namespace {

class RawReaderAdapter final : public CachedRowReader {
 public:
  explicit RawReaderAdapter(std::shared_ptr<const SidecarProducer> base)
      : base_(std::move(base)) {}

  host::PackedRow read_one(std::uint32_t row_id) const override {
    return base_->reader().read_one(row_id);
  }

 private:
  const std::shared_ptr<const SidecarProducer> base_;
};

}  // namespace

struct CachedSidecarJob::State final {
  explicit State(std::shared_ptr<PermitPool> permits)
      : permits(std::move(permits)) {}

  ~State() {
    // A producer may be torn down with undrained completion entries.  The
    // entry owns this state, so its final destruction is the last safe place
    // to return a permit that the owner never drained.
    std::lock_guard<std::mutex> lock(mutex);
    release_locked();
  }

  void start() {
    std::lock_guard<std::mutex> lock(mutex);
    if (abandoned || started) {
      throw std::runtime_error("cached sidecar job was already started or abandoned");
    }
    started = true;
  }

  bool mark_finished(bool has_completion) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    run_finished = true;
    completion_pending = has_completion && !abandoned;
    if (abandoned) {
      release_locked();
    } else {
      release_if_ready_locked();
    }
    return !abandoned;
  }

  void mark_failed() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    run_finished = true;
    failed = true;
    completion_pending = false;
    release_locked();
  }

  void mark_output_published() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    output_published = true;
    release_if_ready_locked();
  }

  void mark_completion_drained() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    completion_drained = true;
    completion_pending = false;
    release_if_ready_locked();
  }

  void abandon() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    abandoned = true;
    // A pre-dispatch job has no worker-owned work.  Once run() has started,
    // keep its permit until mark_finished() observes that every submitted
    // read has been joined; otherwise repeated abandon() calls could admit
    // more than two active I/O jobs.
    if (!started || run_finished) {
      release_locked();
    }
  }

  void on_job_destroyed() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (!started) {
      abandoned = true;
      release_locked();
      return;
    }
    if (!run_finished) {
      abandoned = true;
      return;
    }
    if (abandoned || failed) {
      return;
    }
    if (!output_published) {
      // run() completed, but the caller never published its output.  Any
      // queued completion is stale and will be purged by the owner drain.
      abandoned = true;
      completion_pending = false;
      release_locked();
      return;
    }
    // A published miss completion must remain owned by the producer after
    // the job handle dies.  The owner drain is the only release boundary.
    release_if_ready_locked();
  }

  bool is_abandoned() const noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    return abandoned;
  }

 private:
  void release_if_ready_locked() noexcept {
    if (!run_finished || !output_published || failed) {
      return;
    }
    if (completion_pending && !completion_drained) {
      return;
    }
    release_locked();
  }

  void release_locked() noexcept {
    if (permit_held) {
      permit_held = false;
      permits->release();
    }
  }

  const std::shared_ptr<PermitPool> permits;
  mutable std::mutex mutex;
  bool started = false;
  bool run_finished = false;
  bool output_published = false;
  bool completion_pending = false;
  bool completion_drained = false;
  bool failed = false;
  bool abandoned = false;
  bool permit_held = true;
};

struct CachedSidecarProducer::CompletionEntry final {
  CachedCompletion completion{};
  std::shared_ptr<CachedSidecarJob::State> state;
};

class CachedSidecarProducer::IoPool final {
 public:
  explicit IoPool(std::size_t worker_count) {
    workers_.reserve(worker_count);
    try {
      for (std::size_t index = 0; index < worker_count; ++index) {
        workers_.emplace_back([this] { worker_loop(); });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
      }
      condition_.notify_all();
      for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
      }
      throw;
    }
  }

  ~IoPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
  }

  IoPool(const IoPool&) = delete;
  IoPool& operator=(const IoPool&) = delete;

  template <typename Function>
  auto submit(Function&& function)
      -> std::future<std::invoke_result_t<Function>> {
    using Result = std::invoke_result_t<Function>;
    auto task = std::make_shared<std::packaged_task<Result()>>(
        std::forward<Function>(function));
    auto future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) {
        throw std::runtime_error("cached sidecar I/O pool is stopping");
      }
      tasks_.emplace_back([task] { (*task)(); });
    }
    condition_.notify_one();
    return future;
  }

 private:
  void worker_loop() noexcept {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (tasks_.empty()) {
          if (stopping_) return;
          continue;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
};

CachedSidecarJob::CachedSidecarJob(
    std::shared_ptr<CachedSidecarProducer> producer,
    CachedRowHandoff handoff,
    std::shared_ptr<PermitPool> permits,
    std::uint64_t ticket)
    : producer_(std::move(producer)),
      handoff_(std::move(handoff)),
      ticket_(ticket),
      state_(std::make_shared<State>(std::move(permits))) {}

CachedSidecarJob::~CachedSidecarJob() { state_->on_job_destroyed(); }

void CachedSidecarJob::release_permit() noexcept {
  state_->mark_output_published();
}

void CachedSidecarJob::abandon() noexcept { state_->abandon(); }

host::PackedRows CachedSidecarJob::run() {
  state_->start();

  std::vector<std::future<host::PackedRow>> futures;
  futures.reserve(handoff_.miss_count);
  std::exception_ptr first_error;
  try {
    for (std::size_t index = 0; index < handoff_.miss_count; ++index) {
      const std::uint32_t row_id = handoff_.miss_ids[index];
      const auto reader = producer_->reader_;
      futures.emplace_back(producer_->io_pool_->submit(
          [reader, row_id] { return reader->read_one(row_id); }));
    }
  } catch (...) {
    first_error = std::current_exception();
  }

  if (first_error) {
    // packaged_task futures do not join on destruction.  Drain every task
    // already submitted before releasing this job's permit.
    for (auto& future : futures) {
      try {
        (void)future.get();
      } catch (...) {
        // Preserve the submission failure as the ordinary CPU error.
      }
    }
    state_->mark_failed();
    std::rethrow_exception(first_error);
  }

  std::array<host::PackedRow, host::kRowsPerWindow> miss_payloads{};
  for (std::size_t index = 0; index < futures.size(); ++index) {
    try {
      miss_payloads[index] = futures[index].get();
    } catch (...) {
      if (!first_error) first_error = std::current_exception();
    }
  }
  if (first_error) {
    state_->mark_failed();
    std::rethrow_exception(first_error);
  }

  host::PackedRows output{};
  for (std::size_t row = 0; row < host::kRowsPerWindow; ++row) {
    const std::uint8_t source = handoff_.source[row];
    const std::size_t index = source & kCachedSourceMask;
    const host::PackedRow& payload =
        (source & kCachedHitBit) != 0 ? handoff_.hit_packed[index]
                                      : miss_payloads[index];
    std::copy(
        payload.begin(), payload.end(),
        output.begin() + row * host::kPackedBytesPerRow);
  }

  const bool enqueue_completion =
      state_->mark_finished(handoff_.miss_count != 0);
  if (handoff_.miss_count != 0 && enqueue_completion) {
    CachedCompletion completion{};
    completion.ticket = ticket_;
    completion.count = handoff_.miss_count;
    for (std::size_t index = 0; index < handoff_.miss_count; ++index) {
      completion.miss_ids[index] = handoff_.miss_ids[index];
      completion.payloads[index] = miss_payloads[index];
    }
    try {
      producer_->enqueue_completion(state_, std::move(completion));
    } catch (...) {
      state_->mark_failed();
      throw;
    }
  }
  return output;
}

std::shared_ptr<CachedSidecarProducer> CachedSidecarProducer::install(
    int descriptor,
    host::SidecarLayout layout,
    const std::array<std::int64_t, host::kNgramSize>& multipliers,
    const std::array<std::int64_t, host::kNgramHeads>& sizes,
    const std::array<std::int64_t, host::kNgramHeads>& offsets,
    std::int64_t eos,
    std::size_t io_workers) {
  auto base = SidecarProducer::install(
      descriptor, layout, multipliers, sizes, offsets, eos);
  auto reader = std::make_shared<RawReaderAdapter>(base);
  return std::shared_ptr<CachedSidecarProducer>(new CachedSidecarProducer(
      std::move(base), std::move(reader), io_workers));
}

std::shared_ptr<CachedSidecarProducer>
CachedSidecarProducer::install_for_test(
    int descriptor,
    host::SidecarLayout layout,
    const std::array<std::int64_t, host::kNgramSize>& multipliers,
    const std::array<std::int64_t, host::kNgramHeads>& sizes,
    const std::array<std::int64_t, host::kNgramHeads>& offsets,
    std::int64_t eos,
    std::shared_ptr<const CachedRowReader> reader,
    std::size_t io_workers) {
  if (reader == nullptr) {
    throw std::invalid_argument("cached sidecar test reader is null");
  }
  auto base = SidecarProducer::install(
      descriptor, layout, multipliers, sizes, offsets, eos);
  return std::shared_ptr<CachedSidecarProducer>(new CachedSidecarProducer(
      std::move(base), std::move(reader), io_workers));
}

CachedSidecarProducer::CachedSidecarProducer(
    std::shared_ptr<const SidecarProducer> base,
    std::shared_ptr<const CachedRowReader> reader,
    std::size_t io_workers)
    : base_(std::move(base)),
      reader_(std::move(reader)),
      io_pool_(nullptr),
      io_workers_(io_workers),
      completions_(std::make_unique<std::deque<CompletionEntry>>()) {
  if (base_ == nullptr || reader_ == nullptr) {
    throw std::invalid_argument("cached sidecar producer dependencies are null");
  }
  if (io_workers_ == 0 || io_workers_ > kCachedMaxIoWorkers) {
    throw std::invalid_argument("cached sidecar I/O worker count is out of bounds");
  }
  io_pool_ = std::make_unique<IoPool>(io_workers_);
}

CachedSidecarProducer::~CachedSidecarProducer() = default;

std::uint64_t CachedSidecarProducer::row_count() const noexcept {
  return base_->row_count();
}

SidecarRowIds CachedSidecarProducer::compute_row_ids(
    const SidecarJobInput& input) const {
  const host::NgramRowsResult result =
      base_->plan().compute(input.previous, input.ids);
  SidecarRowIds rows{};
  for (std::size_t index = 0; index < rows.size(); ++index) {
    rows[index] = static_cast<std::uint32_t>(result.rows[index]);
  }
  return rows;
}

void CachedSidecarProducer::validate_handoff(
    const CachedRowHandoff& handoff,
    std::uint64_t row_count) {
  if (handoff.hit_count > host::kRowsPerWindow ||
      handoff.miss_count > host::kRowsPerWindow) {
    throw std::invalid_argument("cached sidecar handoff count exceeds M4 bound");
  }
  for (std::size_t index = 0; index < handoff.source.size(); ++index) {
    const std::uint8_t source = handoff.source[index];
    if ((source & 0x40U) != 0) {
      throw std::invalid_argument(
          "cached sidecar source uses reserved bit 6");
    }
    const std::size_t compact_index = source & kCachedSourceMask;
    const bool hit = (source & kCachedHitBit) != 0;
    const std::size_t limit = hit ? handoff.hit_count : handoff.miss_count;
    if (compact_index >= limit) {
      throw std::invalid_argument("cached sidecar source index is out of bounds");
    }
  }
  for (std::size_t index = 0; index < handoff.miss_count; ++index) {
    if (static_cast<std::uint64_t>(handoff.miss_ids[index]) >= row_count) {
      throw std::out_of_range("cached sidecar miss row exceeds row count");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (handoff.miss_ids[prior] == handoff.miss_ids[index]) {
        throw std::invalid_argument("cached sidecar miss IDs must be unique");
      }
    }
  }
}

std::shared_ptr<CachedSidecarJob> CachedSidecarProducer::make_job(
    const CachedRowHandoff& handoff,
    const std::shared_ptr<PermitPool>& permits) {
  if (permits == nullptr) {
    throw std::invalid_argument("cached sidecar permit pool is null");
  }
  validate_handoff(handoff, row_count());
  if (!permits->try_acquire()) {
    throw std::runtime_error("[mtplx_native_ple_cpu_rows] bounded two-request queue is full");
  }
  PermitLease lease(permits);
  const std::uint64_t ticket = next_ticket_.fetch_add(1, std::memory_order_relaxed);
  auto owned = std::unique_ptr<CachedSidecarJob>(new CachedSidecarJob(
      shared_from_this(), handoff, permits, ticket));
  lease.disarm();
  return std::shared_ptr<CachedSidecarJob>(std::move(owned));
}

void CachedSidecarProducer::enqueue_completion(
    const std::shared_ptr<CachedSidecarJob::State>& state,
    CachedCompletion completion) {
  std::lock_guard<std::mutex> lock(completion_mutex_);
  // An explicit abandon may race the final output copy.  In that case the
  // permit is already released (after the reads have joined), and no stale
  // completion should become visible to the owner.
  if (state->is_abandoned()) return;
  completions_->erase(
      std::remove_if(
          completions_->begin(), completions_->end(),
          [](const CompletionEntry& entry) {
            return entry.state == nullptr || entry.state->is_abandoned();
          }),
      completions_->end());
  if (completions_->size() >= kMaxOutstanding) {
    throw std::runtime_error("cached sidecar completion bound is full");
  }
  if (state->is_abandoned()) return;
  completions_->push_back(CompletionEntry{std::move(completion), state});
}

std::vector<CachedCompletion> CachedSidecarProducer::drain_completed() {
  std::deque<CompletionEntry> pending;
  {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    pending.swap(*completions_);
  }
  std::vector<CachedCompletion> output;
  output.reserve(pending.size());
  for (auto& entry : pending) {
    if (entry.state == nullptr || entry.state->is_abandoned()) continue;
    entry.state->mark_completion_drained();
    output.push_back(std::move(entry.completion));
  }
  return output;
}

}  // namespace mtplx_native::ple_cpu_rows
