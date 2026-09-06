"""MLX-free TDD coverage for the cache-aware sidecar producer foundation.

This driver links only the authoritative host reader, the existing immutable
plan producer, and the new cache-aware CPU producer.  It never imports MLX,
builds an extension, starts a model, or touches the GPU.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap


ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "native_extensions" / "ple_cpu_rows"
LATE = CPU  # host_provider.{h,cpp} are co-located in ple_cpu_rows in this PR


_DRIVER = r"""
#include "cached_sidecar_producer.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace pp = mtplx_native::ple_cpu_rows;
namespace hp = mtplx_native::host_provider;

namespace {

constexpr std::uint64_t kRowCount = 2000;

struct Config {
  std::array<std::int64_t, hp::kNgramSize> multipliers{
      INT64_MIN, -7, 11};
  std::array<std::int64_t, hp::kNgramHeads> sizes{};
  std::array<std::int64_t, hp::kNgramHeads> offsets{};
  std::int64_t eos = 99;

  Config() {
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      sizes[index] = 500 + static_cast<std::int64_t>(index * 3);
      offsets[index] = 100 + static_cast<std::int64_t>(index * 10);
    }
  }
};

hp::SidecarLayout layout() {
  const std::uint64_t weights_offset = 13;
  const std::uint64_t scales_offset = weights_offset + kRowCount * 80 + 7;
  const std::uint64_t biases_offset = scales_offset + kRowCount * 10 + 11;
  return hp::SidecarLayout{
      kRowCount,
      {weights_offset, kRowCount * 80, 80},
      {scales_offset, kRowCount * 10, 10},
      {biases_offset, kRowCount * 10, 10},
  };
}

hp::PackedRow payload(std::uint32_t row) {
  hp::PackedRow result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::uint8_t>(
        (static_cast<std::uint64_t>(row) * 17 + index * 3 + 5) & 0xff);
  }
  return result;
}

struct ReaderState {
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::unordered_map<std::uint32_t, int> calls;
  std::uint32_t fail_row = UINT32_MAX;
  std::uint32_t block_row = UINT32_MAX;
  bool block_started = false;
  bool unblock = false;
};

class CountingReader final : public pp::CachedRowReader {
 public:
  explicit CountingReader(std::shared_ptr<ReaderState> state)
      : state_(std::move(state)) {}

  hp::PackedRow read_one(std::uint32_t row_id) const override {
    std::unique_lock<std::mutex> lock(state_->mutex);
    ++state_->calls[row_id];
    if (row_id == state_->fail_row) {
      throw std::runtime_error("short read");
    }
    if (row_id == state_->block_row) {
      state_->block_started = true;
      state_->condition.notify_all();
      state_->condition.wait(lock, [this] { return state_->unblock; });
    }
    return payload(row_id);
  }

 private:
  const std::shared_ptr<ReaderState> state_;
};

void create_fixture(const std::string& path) {
  const auto sidecar = layout();
  const std::uint64_t end = sidecar.biases.offset + sidecar.biases.length;
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0600);
  if (fd < 0) throw std::runtime_error("fixture open failed");
  if (::ftruncate(fd, static_cast<off_t>(end)) != 0) {
    ::close(fd);
    throw std::runtime_error("fixture truncate failed");
  }
  ::close(fd);
}

std::shared_ptr<pp::CachedSidecarProducer> install(
    const std::string& path,
    std::shared_ptr<ReaderState> state,
    std::size_t workers = 8) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("fixture reopen failed");
  try {
    auto reader = std::make_shared<CountingReader>(std::move(state));
    auto producer = pp::CachedSidecarProducer::install_for_test(
        fd, layout(), Config{}.multipliers, Config{}.sizes, Config{}.offsets,
        Config{}.eos, std::move(reader), workers);
    ::close(fd);
    return producer;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

pp::CachedRowHandoff handoff(
    std::initializer_list<std::uint8_t> sources,
    std::uint8_t hit_count,
    std::initializer_list<std::uint32_t> miss_ids) {
  pp::CachedRowHandoff result{};
  result.hit_count = hit_count;
  std::size_t index = 0;
  for (const std::uint8_t source : sources) {
    if (index >= result.source.size()) throw std::runtime_error("too many sources");
    result.source[index++] = source;
  }
  while (index < result.source.size()) {
    result.source[index++] = hit_count == 0 ? 0 : 0x80;
  }
  std::size_t miss = 0;
  for (const std::uint32_t row : miss_ids) {
    result.miss_ids[miss++] = row;
  }
  result.miss_count = static_cast<std::uint8_t>(miss);
  for (std::size_t row = 0; row < result.hit_count; ++row) {
    result.hit_packed[row] = payload(static_cast<std::uint32_t>(700 + row));
  }
  return result;
}

void assert_output(const hp::PackedRows& output,
                   const pp::CachedRowHandoff& input) {
  for (std::size_t row = 0; row < hp::kRowsPerWindow; ++row) {
    const std::uint8_t source = input.source[row];
    const bool hit = (source & pp::kCachedHitBit) != 0;
    const std::size_t index = source & pp::kCachedSourceMask;
    const hp::PackedRow expected =
        hit ? input.hit_packed[index] : payload(input.miss_ids[index]);
    for (std::size_t byte = 0; byte < hp::kPackedBytesPerRow; ++byte) {
      if (output[row * hp::kPackedBytesPerRow + byte] != expected[byte]) {
        throw std::runtime_error("scatter output mismatch");
      }
    }
  }
}

int run_all_hits(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  auto producer = install(path, state);
  auto pool = std::make_shared<pp::PermitPool>();
  auto input = handoff({0x80}, 1, {});
  auto job = producer->make_job(input, pool);
  assert_output(job->run(), input);
  job->release_permit();
  if (pool->outstanding() != 0 || !state->calls.empty()) {
    throw std::runtime_error("all-hit job performed a read or leaked permit");
  }
  if (!producer->drain_completed().empty()) {
    throw std::runtime_error("all-hit job emitted a completion");
  }
  std::cout << "all-hits-ok\n";
  return 0;
}

int run_mixed(const std::string& path, bool duplicate) {
  auto state = std::make_shared<ReaderState>();
  auto producer = install(path, state);
  auto pool = std::make_shared<pp::PermitPool>();
  auto input = duplicate
      ? handoff({0x80, 0x00, 0x01, 0x01, 0x02, 0x80}, 1, {11, 12, 13})
      : handoff({0x80, 0x00, 0x01, 0x80, 0x02}, 1, {11, 12, 13});
  auto job = producer->make_job(input, pool);
  assert_output(job->run(), input);
  job->release_permit();
  if (pool->outstanding() != 1) {
    throw std::runtime_error("miss completion released permit early");
  }
  const auto completions = producer->drain_completed();
  if (completions.size() != 1 || completions[0].count != 3) {
    throw std::runtime_error("miss completion shape mismatch");
  }
  if (pool->outstanding() != 0) {
    throw std::runtime_error("completion drain did not release permit");
  }
  if (state->calls.size() != 3) {
    throw std::runtime_error("miss dedup read count mismatch");
  }
  for (const auto& item : state->calls) {
    if (item.second != 1) throw std::runtime_error("miss read repeated");
  }
  std::cout << (duplicate ? "duplicate-ok\n" : "mixed-ok\n");
  return 0;
}

int run_error(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  state->fail_row = 12;
  auto producer = install(path, state);
  auto pool = std::make_shared<pp::PermitPool>();
  auto input = handoff({0x00, 0x01, 0x02}, 0, {11, 12, 13});
  auto job = producer->make_job(input, pool);
  try {
    (void)job->run();
    throw std::runtime_error("read failure unexpectedly succeeded");
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()) != "short read") throw;
  }
  if (pool->outstanding() != 0 || !producer->drain_completed().empty()) {
    throw std::runtime_error("error did not release or emitted completion");
  }
  std::cout << "error-ok\n";
  return 0;
}

int run_abandon(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  auto producer = install(path, state);
  auto pool = std::make_shared<pp::PermitPool>();
  {
    auto job = producer->make_job(handoff({0x00}, 0, {11}), pool);
    job->abandon();
  }
  if (pool->outstanding() != 0 || !producer->drain_completed().empty()) {
    throw std::runtime_error("abandon leaked permit or completion");
  }
  std::cout << "abandon-ok\n";
  return 0;
}

int run_concurrent(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  auto producer = install(path, state, 8);
  auto pool = std::make_shared<pp::PermitPool>();
  auto first = producer->make_job(handoff({0x00}, 0, {11}), pool);
  auto second = producer->make_job(handoff({0x00}, 0, {11}), pool);
  (void)first->run();
  (void)second->run();
  first->release_permit();
  second->release_permit();
  auto completions = producer->drain_completed();
  if (completions.size() != 2 || state->calls[11] != 2) {
    throw std::runtime_error("cross-job dedup was silently claimed");
  }
  if (pool->outstanding() != 0) throw std::runtime_error("permit leaked");
  std::cout << "concurrent-ok\n";
  return 0;
}

int run_pending_bound(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  auto producer = install(path, state, 8);
  auto pool = std::make_shared<pp::PermitPool>();
  auto first = producer->make_job(handoff({0x00}, 0, {11}), pool);
  auto second = producer->make_job(handoff({0x00}, 0, {12}), pool);
  (void)first->run();
  (void)second->run();
  first->release_permit();
  second->release_permit();
  if (pool->outstanding() != 2) {
    throw std::runtime_error("completed jobs released permits before drain");
  }
  bool rejected = false;
  try {
    (void)producer->make_job(handoff({0x00}, 0, {13}), pool);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("third job bypassed pending completion bound");
  }
  const auto completions = producer->drain_completed();
  if (completions.size() != 2 || pool->outstanding() != 0) {
    throw std::runtime_error("pending completion drain did not release both slots");
  }
  std::cout << "pending-bound-ok\n";
  return 0;
}

int run_worker_bounds(const std::string& path) {
  for (const std::size_t workers : {std::size_t{0}, pp::kCachedMaxIoWorkers + 1}) {
    auto state = std::make_shared<ReaderState>();
    bool rejected = false;
    try {
      (void)install(path, state, workers);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error("out-of-bound worker count was accepted");
    }
  }
  std::cout << "worker-bounds-ok\n";
  return 0;
}

int run_active_abandon(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  state->block_row = 11;
  auto producer = install(path, state, 1);
  auto pool = std::make_shared<pp::PermitPool>();
  auto first = producer->make_job(handoff({0x00}, 0, {11}), pool);
  std::exception_ptr run_error;
  std::thread runner([&] {
    try {
      (void)first->run();
    } catch (...) {
      run_error = std::current_exception();
    }
  });
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condition.wait(lock, [&] { return state->block_started; });
  }
  first->abandon();
  auto second = producer->make_job(handoff({0x00}, 0, {12}), pool);
  bool rejected = false;
  try {
    (void)producer->make_job(handoff({0x00}, 0, {13}), pool);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected || pool->outstanding() != 2) {
    throw std::runtime_error("active abandon bypassed two-job bound");
  }
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->unblock = true;
  }
  state->condition.notify_all();
  runner.join();
  if (run_error) std::rethrow_exception(run_error);
  if (pool->outstanding() != 1 || !producer->drain_completed().empty()) {
    throw std::runtime_error("active abandon did not release after worker drain");
  }
  second->abandon();
  if (pool->outstanding() != 0) {
    throw std::runtime_error("active abandon test leaked second permit");
  }
  std::cout << "active-abandon-ok\n";
  return 0;
}

int run_completion_survives_drop(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  auto producer = install(path, state);
  auto pool = std::make_shared<pp::PermitPool>();
  {
    auto job = producer->make_job(handoff({0x00}, 0, {11}), pool);
    (void)job->run();
    job->release_permit();
  }
  if (pool->outstanding() != 1) {
    throw std::runtime_error("job destruction released undrained completion");
  }
  const auto completions = producer->drain_completed();
  if (completions.size() != 1 || completions[0].count != 1 ||
      pool->outstanding() != 0) {
    throw std::runtime_error("dropped job completion was not retained to drain");
  }
  std::cout << "completion-drop-ok\n";
  return 0;
}

int run_provider_drop_releases_completion(const std::string& path) {
  auto pool = std::make_shared<pp::PermitPool>();
  {
    auto producer = install(path, std::make_shared<ReaderState>());
    auto job = producer->make_job(handoff({0x00}, 0, {11}), pool);
    (void)job->run();
    job->release_permit();
    job.reset();
    if (pool->outstanding() != 1) {
      throw std::runtime_error("completion permit was released before teardown");
    }
    producer.reset();
  }
  if (pool->outstanding() != 0) {
    throw std::runtime_error("provider teardown leaked completion permit");
  }
  std::cout << "provider-drop-ok\n";
  return 0;
}

int run_reserved_source(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  auto producer = install(path, state);
  auto pool = std::make_shared<pp::PermitPool>();
  for (const auto& invalid : {
           handoff({0x40}, 0, {11}), handoff({0xc0}, 1, {})}) {
    bool rejected = false;
    try {
      (void)producer->make_job(invalid, pool);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("reserved source bit was accepted");
  }
  if (pool->outstanding() != 0 || !state->calls.empty()) {
    throw std::runtime_error("reserved source reached the reader");
  }
  std::cout << "reserved-source-ok\n";
  return 0;
}

int run_invalid(const std::string& path) {
  auto state = std::make_shared<ReaderState>();
  auto producer = install(path, state);
  auto pool = std::make_shared<pp::PermitPool>();
  auto invalid = handoff({0x80 | 1}, 1, {});
  bool rejected = false;
  try {
    (void)producer->make_job(invalid, pool);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || pool->outstanding() != 0) {
    throw std::runtime_error("invalid typed handoff was accepted");
  }
  std::cout << "invalid-ok\n";
  return 0;
}

int run_host_subset(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("host reopen failed");
  hp::RawSidecarBatchReader reader(fd, layout());
  ::close(fd);
  const auto one = reader.read_one(7);
  if (one != hp::PackedRow{}) throw std::runtime_error("read_one mismatch");
  const auto rows = reader.read_subset({7, 3, 7});
  if (rows.size() != 3 || rows[0] != hp::PackedRow{} ||
      rows[1] != hp::PackedRow{} || rows[2] != hp::PackedRow{}) {
    throw std::runtime_error("read_subset order mismatch");
  }
  bool rejected = false;
  try {
    reader.read_subset(std::vector<std::uint32_t>(65, 1));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("oversized subset accepted");
  std::cout << "host-subset-ok\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::invalid_argument("expected mode and fixture path");
    const std::string mode = argv[1];
    create_fixture(argv[2]);
    if (mode == "all_hits") return run_all_hits(argv[2]);
    if (mode == "mixed") return run_mixed(argv[2], false);
    if (mode == "duplicate") return run_mixed(argv[2], true);
    if (mode == "error") return run_error(argv[2]);
    if (mode == "abandon") return run_abandon(argv[2]);
    if (mode == "concurrent") return run_concurrent(argv[2]);
    if (mode == "pending_bound") return run_pending_bound(argv[2]);
    if (mode == "worker_bounds") return run_worker_bounds(argv[2]);
    if (mode == "active_abandon") return run_active_abandon(argv[2]);
    if (mode == "completion_drop") return run_completion_survives_drop(argv[2]);
    if (mode == "provider_drop") return run_provider_drop_releases_completion(argv[2]);
    if (mode == "reserved_source") return run_reserved_source(argv[2]);
    if (mode == "invalid") return run_invalid(argv[2]);
    if (mode == "host_subset") return run_host_subset(argv[2]);
    throw std::invalid_argument("unknown mode");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
"""


@lru_cache(maxsize=1)
def _driver() -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        raise AssertionError("a C++17 compiler is required")
    temporary = tempfile.TemporaryDirectory(prefix="pr391-cached-sidecar-")
    root = Path(temporary.name)
    source = root / "driver.cpp"
    executable = root / "cached_sidecar_cpu"
    fixture = root / "sidecar.bin"
    source.write_text(textwrap.dedent(_DRIVER), encoding="utf-8")
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wconversion",
        "-Werror",
        "-pthread",
        "-I",
        str(CPU),
        "-I",
        str(LATE),
        str(source),
        str(CPU / "cached_sidecar_producer.cpp"),
        str(CPU / "sidecar_producer.cpp"),
        str(LATE / "host_provider.cpp"),
        "-o",
        str(executable),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            "cached sidecar CPU driver compile failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    fixture.write_bytes(b"\0")
    result = subprocess.run(
        [str(executable), "host_subset", str(fixture)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "cached sidecar fixture setup failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return temporary, executable, fixture


def _run(mode: str) -> str:
    _temporary, executable, fixture = _driver()
    result = subprocess.run(
        [str(executable), mode, str(fixture)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_host_reader_subset_preserves_existing_reader_and_bounds():
    assert _run("host_subset") == "host-subset-ok"


def test_cached_all_hits_perform_no_reads():
    assert _run("all_hits") == "all-hits-ok"


def test_cached_mixed_reads_only_unique_misses_and_releases_on_drain():
    assert _run("mixed") == "mixed-ok"


def test_cached_duplicate_positions_read_each_unique_miss_once():
    assert _run("duplicate") == "duplicate-ok"


def test_cached_read_failure_has_no_partial_completion_and_releases():
    assert _run("error") == "error-ok"


def test_cached_abandon_releases_admission_without_completion():
    assert _run("abandon") == "abandon-ok"


def test_cached_two_jobs_are_bounded_without_claiming_cross_job_dedup():
    assert _run("concurrent") == "concurrent-ok"


def test_cached_completed_jobs_hold_both_slots_until_owner_drains():
    assert _run("pending_bound") == "pending-bound-ok"


def test_cached_io_pool_worker_count_is_construction_bounded():
    assert _run("worker_bounds") == "worker-bounds-ok"


def test_cached_active_abandon_holds_slot_until_worker_drain():
    assert _run("active_abandon") == "active-abandon-ok"


def test_cached_dropped_job_retains_published_completion_until_drain():
    assert _run("completion_drop") == "completion-drop-ok"


def test_cached_provider_teardown_releases_undrained_completion_permit():
    assert _run("provider_drop") == "provider-drop-ok"


def test_cached_handoff_rejects_reserved_source_bit():
    assert _run("reserved_source") == "reserved-source-ok"


def test_cached_typed_handoff_rejects_invalid_source_indices_before_read():
    assert _run("invalid") == "invalid-ok"


def test_cached_sources_and_receipt_contract_are_mlxfree():
    producer = (CPU / "cached_sidecar_producer.cpp").read_text(encoding="utf-8")
    header = (CPU / "cached_sidecar_producer.h").read_text(encoding="utf-8")
    assert "mlx/" not in producer
    assert "mlx/" not in header
    assert "kCachedHitBit" in header
    assert "drain_completed" in header
    assert "read_one" in producer
    assert "read_subset" not in producer
    assert "std::thread" in producer
    assert "std::condition_variable" in producer
    digest = hashlib.sha256(
        (CPU / "cached_sidecar_producer.h").read_bytes()
        + (CPU / "cached_sidecar_producer.cpp").read_bytes()
        + (LATE / "host_provider.h").read_bytes()
        + (LATE / "host_provider.cpp").read_bytes()
    ).hexdigest()
    assert len(digest) == 64
