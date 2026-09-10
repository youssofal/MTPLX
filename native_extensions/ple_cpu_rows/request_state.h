// SPDX-License-Identifier: Apache-2.0
//
// MLX-free request ownership for the CPU-row primitive.  This header is kept
// independent of the extension ABI so its bounded admission/lifetime rules
// can be checked with an ordinary C++17 compiler before an MLX build.

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace mtplx_native::ple_cpu_rows {

constexpr std::size_t kRows = 64;
constexpr std::size_t kWeightValuesPerRow = 20;
constexpr std::size_t kMetadataValuesPerRow = 5;
constexpr std::size_t kWeightBytes =
    kRows * kWeightValuesPerRow * sizeof(std::uint32_t);
constexpr std::size_t kMetadataBytes =
    kRows * kMetadataValuesPerRow * sizeof(std::uint16_t);
constexpr std::size_t kPayloadBytes = 6400;
constexpr std::size_t kMaxOutstanding = 2;
constexpr std::size_t kPerRequestPlaneBytes =
    kWeightBytes + kMetadataBytes + kMetadataBytes;

static_assert(kWeightBytes == 5120);
static_assert(kMetadataBytes == 640);
static_assert(kWeightBytes + kMetadataBytes + kMetadataBytes == kPayloadBytes);
static_assert(kPayloadBytes == kRows * 100);
static_assert(kPerRequestPlaneBytes == 6400);

using PackedPayload = std::array<std::uint8_t, kPayloadBytes>;

// This pool never waits.  Admission either acquires one of the two bounded
// request slots or fails before an MLX graph is constructed.  Completion and
// abandonment return a slot through RequestState::release_permit().
class PermitPool final {
 public:
  explicit PermitPool(std::size_t capacity = kMaxOutstanding)
      : available_(capacity) {
    if (capacity != kMaxOutstanding) {
      throw std::invalid_argument("PLE CPU-row permit capacity must be two");
    }
  }

  PermitPool(const PermitPool&) = delete;
  PermitPool& operator=(const PermitPool&) = delete;

  bool try_acquire() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (available_ == 0) {
      return false;
    }
    --available_;
    return true;
  }

  void release() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    // An over-release is a lifetime bug.  Do not silently saturate: hiding it
    // would make the two-request bound unreviewable and could over-admit work.
    if (available_ >= kMaxOutstanding) {
      std::terminate();
    }
    ++available_;
  }

  std::size_t outstanding() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return kMaxOutstanding - available_;
  }

 private:
  mutable std::mutex mutex_;
  std::size_t available_;
};

// A construction-only transfer guard closes the allocation-failure window
// between acquiring a permit and putting the RequestState in shared storage.
// Once disarmed, the state destructor owns the same single permit.
class PermitLease final {
 public:
  explicit PermitLease(std::shared_ptr<PermitPool> pool)
      : pool_(std::move(pool)) {}

  PermitLease(const PermitLease&) = delete;
  PermitLease& operator=(const PermitLease&) = delete;
  PermitLease(PermitLease&& other) noexcept
      : pool_(std::move(other.pool_)), held_(other.held_) {
    other.held_ = false;
  }
  PermitLease& operator=(PermitLease&& other) noexcept {
    if (this != &other) {
      release();
      pool_ = std::move(other.pool_);
      held_ = other.held_;
      other.held_ = false;
    }
    return *this;
  }

  ~PermitLease() { release(); }

  void disarm() noexcept { held_ = false; }

 private:
  void release() noexcept {
    if (held_) {
      pool_->release();
      held_ = false;
    }
  }

  std::shared_ptr<PermitPool> pool_;
  bool held_ = true;
};

class RequestState final {
 public:
  // Admission is construction-bound and nonblocking.  The returned state is
  // the sole owner of the permit until its dispatch lambda completes or the
  // primitive is abandoned before evaluation.
  static std::shared_ptr<RequestState> admit(
      const std::shared_ptr<PermitPool>& pool,
      PackedPayload payload,
      int delay_ms,
      bool force_fail,
      bool cancelled) {
    if (pool == nullptr) {
      throw std::invalid_argument("PLE CPU-row permit pool is null");
    }
    if (delay_ms < 0 || delay_ms > 30'000) {
      throw std::invalid_argument(
          "PLE CPU-row delay must be between 0 and 30000 ms");
    }
    if (!pool->try_acquire()) {
      throw std::runtime_error(
          "[mtplx_native_ple_cpu_rows] bounded two-request queue is full");
    }

    // Keep the permit in a move-only guard until a unique owner exists.  This
    // makes both RequestState allocation and shared-control-block failure
    // release exactly once.
    PermitLease lease(pool);
    auto owned = std::unique_ptr<RequestState>(new RequestState(
        pool, std::move(payload), delay_ms, force_fail, cancelled));
    lease.disarm();
    return std::shared_ptr<RequestState>(std::move(owned));
  }

  ~RequestState() { release_permit(); }

  RequestState(const RequestState&) = delete;
  RequestState& operator=(const RequestState&) = delete;

  const PackedPayload& payload() const noexcept { return payload_; }
  int delay_ms() const noexcept { return delay_ms_; }
  bool force_fail() const noexcept { return force_fail_; }
  bool cancelled() const noexcept { return cancelled_; }

  // Idempotent so both the normal and exceptional dispatch paths can call it;
  // the destructor also covers graph-construction abandonment.
  void release_permit() noexcept {
    if (permit_held_.exchange(false, std::memory_order_acq_rel)) {
      pool_->release();
    }
  }

 private:
  RequestState(std::shared_ptr<PermitPool> pool,
               PackedPayload payload,
               int delay_ms,
               bool force_fail,
               bool cancelled) noexcept
      : pool_(std::move(pool)),
        payload_(std::move(payload)),
        delay_ms_(delay_ms),
        force_fail_(force_fail),
        cancelled_(cancelled) {}

  const std::shared_ptr<PermitPool> pool_;
  const PackedPayload payload_;
  const int delay_ms_;
  const bool force_fail_;
  const bool cancelled_;
  std::atomic_bool permit_held_{true};
};

}  // namespace mtplx_native::ple_cpu_rows
