// A sealed program and its replay.
//
// A program is a flat list of launches with every buffer argument resolved to
// an entry of a small table.  A run fills the table (the caller's inputs,
// fresh output buffers from MLX's allocator, the engine's own working
// memory), encodes the list into one command buffer on the engine's queue,
// commits, and sleeps until the GPU is done.

#pragma once

#import "engine.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "mlx/allocator.h"
#include "mlx/array.h"

namespace mtplx_engine {

namespace mx = mlx::core;

enum class Role : uint8_t { Input = 0, Output = 1, Pinned = 2, Const = 3, Seed = 4, Temp = 5 };

struct BufferSpec {
  Role role = Role::Temp;
  uint64_t nbytes = 0;
  int index = -1;  // input: first declared input in it; pinned: pin slot
  bool clobbered = false;
  std::vector<uint8_t> data;  // const and seed
};

struct ArraySpec {
  int buffer = -1;  // index into the program's buffer list
  int64_t offset = 0;  // recorded byte offset inside that buffer
  mx::Shape shape;
  mx::Dtype dtype = mx::float32;
  mx::Strides strides;
  uint64_t data_size = 0;  // in elements, as MLX keeps it
  uint8_t flags = 0;  // 1 contiguous, 2 row contiguous, 4 column contiguous
  int alias_input = -1;  // outputs: the input whose data this one shares
};

struct StepSpec {
  int kernel = -1;
  std::vector<std::tuple<int, int, uint64_t, bool>> bindings;  // (slot, buffer, offset, can write)
  std::vector<std::pair<int, std::vector<uint8_t>>> inline_bytes;  // (slot, bytes)
  std::vector<std::pair<int, uint64_t>> tg_memory;  // (slot, length)
  bool threads = false;
  int indirect_buffer = -1;
  uint64_t indirect_offset = 0;
  uint64_t grid[3] = {1, 1, 1};
  uint64_t group[3] = {1, 1, 1};
  bool barrier_before = false;
};

struct ProgramSpec {
  std::string name;
  std::vector<BufferSpec> buffers;
  std::vector<StepSpec> steps;
  std::vector<ArraySpec> inputs;
  std::vector<ArraySpec> outputs;
};

struct ExecBind {
  uint8_t slot;
  uint32_t buffer;
  uint64_t offset;
};

struct ExecInline {
  uint8_t slot;
  uint32_t pos;  // in Program::bytes, which is also the contents of the constants buffer
  uint32_t len;
  uint32_t intern;  // equal ids hold equal bytes
};

// Bytes MLX sends inline with a launch are 16-byte aligned in the store, so the
// same offsets are valid when the store is bound as a buffer.
constexpr uint32_t kInlineAlign = 16;

struct ExecStep {
  __unsafe_unretained id<MTLComputePipelineState> pso;
  uint32_t bind_pos;
  uint32_t inline_pos;
  uint32_t tg_pos;
  uint16_t bind_len;
  uint16_t inline_len;
  uint16_t tg_len;
  uint8_t threads;
  int indirect_buffer;
  uint64_t indirect_offset;
  uint8_t barrier;  // as MLX placed it in the recording (pass boundaries included)
  uint8_t barrier_min;  // needed by the buffers this launch touches, whole-pass barriers
  uint8_t barrier_kind;  // with per-buffer barriers: 0 none, 1 named buffers, 2 whole pass
  uint16_t res_len;
  uint32_t res_pos;
  uint32_t chain_run;  // launches from here on that each depend on the one before
  MTLSize grid;
  MTLSize group;
  int kernel;
};

struct RunStats {
  uint64_t launches = 0;
  uint64_t command_buffers = 0;
  uint64_t encode_ns = 0;  // host time from the first encoder call to the commit
  uint64_t gpu_ns = 0;  // the command buffer's own GPU start to end
  uint64_t wall_ns = 0;  // the whole call
  uint64_t cpu_ns = 0;  // CPU time of the calling thread inside the call
  uint64_t wait_ns = 0;  // blocked on completion
  uint64_t prepare_ns = 0;  // drain, table, output buffers, seeds
  uint64_t outputs_ns = 0;  // building the output arrays
  // Where the wait goes besides the GPU's own time (one command buffer, or
  // the first and last of a split run): from the commit to the GPU starting,
  // from the GPU ending to Metal calling the completion handler, and from
  // the handler to this thread running again.
  uint64_t start_ns = 0;
  uint64_t notify_ns = 0;
  uint64_t wake_ns = 0;
  uint64_t binds = 0;  // buffer bindings actually sent
  uint64_t binds_skipped = 0;  // bindings already in place
  uint64_t passes = 0;  // compute passes encoded
  uint64_t barriers = 0;  // barriers encoded
  // Inputs the program writes into that arrived in a buffer shorter than the
  // recorded one and were moved into one of the recorded length for this run
  // (see Engine::can_run), and the bytes copied for it.
  uint64_t grown = 0;
  uint64_t grown_bytes = 0;
  // With the allocation probe on (tests): heap allocations made by the calling
  // thread while encoding our launches, and in the whole call.
  uint64_t mallocs_encode = 0;
  uint64_t mallocs_total = 0;
};

// Test aid: counts heap allocations on the calling thread.  Off by default and
// costs nothing then.  Returns false when the system offers no way to count.
bool alloc_probe_enable(bool on);

struct StepTime {
  int kernel = -1;
  uint64_t gpu_ns = 0;
  uint64_t wall_ns = 0;
};

class Engine;

class Program : public std::enable_shared_from_this<Program> {
 public:
  ~Program();

  std::string name;
  std::vector<ExecStep> steps;
  std::vector<ExecBind> binds;
  std::vector<ExecInline> inlines;
  std::vector<std::pair<uint8_t, uint64_t>> tg;
  std::vector<uint8_t> bytes;
  std::vector<BufferSpec> buffers;
  std::vector<ArraySpec> inputs;
  std::vector<ArraySpec> outputs;
  std::vector<std::vector<int>> members;  // per buffer: declared inputs in it

  // Filled at seal for what never changes, per run for inputs and outputs.
  std::vector<__unsafe_unretained id<MTLBuffer>> table;
  std::vector<int64_t> delta;
  std::vector<mx::allocator::Buffer> owned;  // const and seed buffers
  // Every inline constant of the program in one buffer, written once at seal.
  // A launch binds it at an offset instead of handing Metal bytes to copy.
  __unsafe_unretained id<MTLBuffer> constants = nil;
  std::vector<int> owned_of;  // per buffer: index into owned, or -1
  std::vector<int> temp_slot;  // per buffer: slot in the engine's pool, or -1
  std::vector<int> seed_buffers;
  std::vector<int> input_buffers;
  std::vector<int> output_buffers;
  // Output buffers that start each run with their recorded bytes (a recycled
  // constant the recording's first write to that buffer only partly replaced).
  std::vector<int> seeded_outputs;
  std::vector<int> clobbered_inputs;  // inputs the program writes into
  // Per buffer: the bytes, counted from the start of the recorded buffer, that
  // a run needs to exist in an input's buffer.  The recorded length at seal
  // (a program may write all of a buffer it was recorded in); lowered to the
  // end of the input's own array by Engine::relax_input once a run has shown
  // that the program never touches the room past it (the allocator's slack,
  // which another buffer of the same array may not have).
  std::vector<uint64_t> required;
  // Indirect descriptors read from relocatable inputs. Check their live offsets before encoding.
  std::vector<std::pair<int, uint64_t>> indirect_inputs;
  // Per output buffer (same order as output_buffers): the length a fresh one
  // is allocated at.  At least the recorded length, and at least the recorded
  // length of every buffer the program writes into whose input has the shape
  // and type of an output held here: a state leaf the program rewrites into a
  // fresh buffer every run comes back as that input on the next run.
  std::vector<uint64_t> fresh_bytes;
  std::vector<uint32_t> barrier_buffers;  // buffers named by per-buffer barriers
  uint32_t barriers_recorded = 0;
  uint32_t barriers_min = 0;
  uint32_t barriers_named = 0;
  uint32_t barriers_whole = 0;
  std::vector<mx::array> pinned;  // keeps pinned arrays alive
  uint64_t pool_generation = 0;
  uint64_t temp_bytes = 0;
  uint64_t const_bytes = 0;
  uint64_t runs = 0;
  bool settle_was_futile = false;
  uint64_t best_gpu_ns = 0;  // shortest GPU time of a completed run (0: none yet)
  // How long a replay of this program takes from its commit to Metal's
  // completion handler, smoothed: when the calling thread stops sleeping and
  // starts watching (Engine::set_spin_lead_us).
  uint64_t wait_ema_ns = 0;
  Engine* engine = nullptr;
};

// How the launches are ordered on the GPU.  Every form computes the same
// thing; they differ in what the GPU is allowed to overlap.
enum Dispatch : int {
  kConcurrent = 0,  // one concurrent pass with barriers (what MLX encodes)
  kSerial = 1,  // one serial pass: launches run strictly in order, no barriers
  kHybrid = 2,  // serial passes for runs of dependent launches, concurrent elsewhere
};

enum Barriers : int {
  kRecorded = 0,  // where MLX placed them
  kMinimal = 1,  // recomputed from what each launch reads and writes
  kNamed = 2,  // barriers that name the buffers they protect
  kNone = 3,  // tests only: timing without any barrier (results are wrong)
  kPasses = 4,  // no barrier calls: a new pass wherever one would be needed
};

struct RunOptions {
  int dispatch = kConcurrent;
  int barriers = kRecorded;
  int min_run = 4;  // hybrid: shortest run of dependent launches given a serial pass
  bool copy_inline = false;  // send inline constants as bytes, as MLX does (tests, comparison)
  // Tests: encode into an object that ignores every call.  Nothing reaches the
  // GPU and no outputs are made; what is measured is this engine's own loop.
  bool null_encoder = false;
  bool profile = false;  // one command buffer per launch, GPU time per step
  bool skip_drain = false;  // tests only: do not wait for MLX's work first
  // Encode the launches before waiting for MLX's own queue to empty, and
  // commit after: the encoding then runs under whatever MLX still has on the
  // GPU instead of after it.  The legality check is the same, in two halves.
  bool encode_early = false;
  int split = 0;  // commit every N launches (0: one command buffer)
  // Diagnosis.  fill (0..255): working memory and fresh outputs start as this
  // byte, so bytes the program never writes are the same in two runs (or
  // tellingly different, with two fills).  after_step runs when a launch of a
  // profile run has completed on the GPU; the program's buffers may be read
  // from inside it (Engine::peek).
  int fill = -1;
  std::function<void(size_t)> after_step;
};

class Engine {
 public:
  Engine();
  ~Engine();

  std::string init();

  // Pins: long-lived arrays bound where MLX holds them.
  int pin(const mx::array& array);
  int pin_slot_for(const void* buffer) const;
  size_t pin_count() const {
    return pins_.size();
  }

  std::shared_ptr<Program> seal(ProgramSpec&& spec, std::string* error);

  // "" when `inputs` can run, otherwise a short reason code.
  //
  // MLX keeps a second holder on every array an evaluation just produced
  // until that command buffer's completion handler has been released, some
  // microseconds after the evaluation returns.  An input the program writes
  // into can therefore look shared for a moment.  With `settle`, such an input
  // gets a short bounded wait before the call is refused; a program whose wait
  // was in vain is not made to wait again until it has run once.
  //
  // Order between MLX's queue and ours is settled here too: with `drain`, an
  // input MLX is still computing is waited for before anything else is asked
  // of it.
  //
  // An input the program writes into must sit in a buffer at least as long as
  // the recorded one (the program uses all of it).  With `grow`, a shorter
  // one is not a refusal: its buffer index is appended and the run copies the
  // array into a buffer of the recorded length first (bind_run), up to
  // kGrowLimit bytes.  Without, the answer is "input_buffer_small".
  std::string can_run(
      Program& program,
      const std::vector<const mx::array*>& inputs,
      bool settle = true,
      bool drain = true,
      bool holders = true,
      std::vector<int>* grow = nullptr);

  // Waits until each array's data has a single holder, at most `timeout_us`.
  static bool settle(const std::vector<const mx::array*>& arrays, uint64_t timeout_us);

  // Runs.  `inputs` are the caller's arrays; outputs are appended.  Returns ""
  // or the reason.  A reason from the legality check leaves everything
  // untouched; an error after the commit is prefixed "failed:".
  std::string run(
      Program& program,
      const std::vector<const mx::array*>& inputs,
      std::vector<mx::array>* outputs,
      const RunOptions& options);

  // The same replay in two calls, so the caller can work while the GPU does.
  // `submit` returns once the launches are committed: "" (in flight), or a
  // reason exactly as `run` gives one.  `collect` waits at most `timeout_ns`:
  // "" (finished, outputs appended), "pending" (not finished: nothing has
  // been touched, call again or `give_up`), "failed:..." or "no_flight".
  // One replay is in flight at a time; `run` and `submit` answer
  // "engine_busy" meanwhile.  `give_up` treats the replay as one that passed
  // its bound (the engine is stuck until the GPU has let go of it).
  std::string submit(
      std::shared_ptr<Program> program,
      const std::vector<const mx::array*>& inputs,
      const RunOptions& options);
  std::string collect(std::vector<mx::array>* outputs, uint64_t timeout_ns);
  void give_up();
  bool in_flight() const {
    return flight_.active;
  }

  void synchronize();

  id<MTLDevice> device() const {
    return device_;
  }
  const RunStats& last() const {
    return last_;
  }
  const RunStats& total() const {
    return total_;
  }
  // Where buffer `index` of the program is right now: valid inside a run's
  // after_step, when every buffer of the program is bound.  For an input's
  // buffer `shift` is how far this call's array sits from the recorded one.
  struct Peek {
    const uint8_t* bytes = nullptr;
    size_t length = 0;
    int64_t shift = 0;
  };
  Peek peek(const Program& program, int index) const;

  // The same bytes, shape, strides and offset in a buffer of its own: the
  // whole parent buffer is copied, so a view stays the same view.  The new
  // buffer is at least `min_length` bytes long, and the copy starts `pad`
  // bytes into it (the array's offset grows by as much).
  mx::array clone_layout(const mx::array& source, uint64_t min_length = 0, uint64_t pad = 0);

  // The largest buffer a run copies to give a short input the recorded length.
  static constexpr uint64_t kGrowLimit = 8ull << 20;

  // The room past an array's own end in its buffer (the allocator's slack).
  // mark_tail fills it with a pattern made from `seed` and returns how many
  // bytes it marked: 0 when there is no room, the array is not evaluated, or
  // its data has another holder (a view or a parent shares the buffer, so the
  // room may be someone's data).  tail_intact says whether those bytes still
  // hold the pattern.  Together they show whether a run wrote past the array.
  uint64_t mark_tail(const mx::array& array, uint64_t seed);
  bool tail_intact(const mx::array& array, uint64_t seed, uint64_t marked);
  // After a run on `array` as input `index` left its marked tail intact: the
  // program needs no byte past the array's end in that buffer, so a later
  // call may hand the same array in a buffer with less room (see `required`).
  // Refused (false) for a buffer that holds several inputs, or that a launch
  // binds at an offset past the array's end.
  bool relax_input(Program& program, int index);
  // Why the last legality check answered input_buffer_small: which input, the
  // buffer it arrived in and what the program needs ("" when none yet).
  const std::string& last_short_detail() const {
    return last_short_detail_;
  }
  uint64_t grown_inputs() const {
    return grown_inputs_;
  }
  uint64_t grown_bytes() const {
    return grown_bytes_;
  }

  // No wait of the engine is unbounded.  A replay that has not completed
  // after max(floor, 50 x the program's best GPU time) is abandoned: the run
  // fails with "failed:timeout", and the engine refuses every run
  // ("engine_stuck") until that command buffer has left the GPU, because its
  // buffers are still being written.
  void set_wait_floor_ms(uint64_t ms) {
    wait_floor_ns_ = ms * 1'000'000ull;
  }
  // 0 (the default): the calling thread sleeps until Metal's completion
  // handler wakes it.  N > 0: it sleeps until N microseconds before the
  // replay is due (by the program's own smoothed time) and then watches for
  // the handler's mark without sleeping, at most N microseconds past the due
  // time, so the wake-up of a sleeping thread is not waited for.  Costs up
  // to about 2 N microseconds of one core per replay.
  void set_spin_lead_us(uint64_t us) {
    spin_lead_ns_ = us * 1'000ull;
  }
  // True while the GPU still holds an abandoned replay (asks the command
  // buffer each time, so it turns false by itself).
  bool stuck();

  const std::vector<StepTime>& last_profile() const {
    return profile_;
  }
  uint64_t runs() const {
    return runs_;
  }
  uint64_t settle_waits() const {
    return settle_waits_;
  }
  uint64_t settle_wait_ns() const {
    return settle_wait_ns_;
  }
  uint64_t pool_bytes() const {
    return pool_bytes_.load(std::memory_order_relaxed);
  }
  uint64_t release_idle_temporaries();
  size_t pool_slots() const {
    return pool_slots_.load(std::memory_order_relaxed);
  }
  const std::unordered_map<std::string, uint64_t>& refusals() const {
    return refusals_;
  }
  void count_refusal(const std::string& reason);

  std::string gpu_name;
  std::string gpu_arch;

 private:
  friend class Program;

  struct PoolSlot {
    mx::allocator::Buffer buffer{nullptr};
    uint64_t length = 0;
  };

  bool reserve_temp(size_t slot, uint64_t bytes);

  void encode(
      Program& program,
      id<MTLComputeCommandEncoder> encoder,
      size_t first,
      size_t last,
      const RunOptions& options,
      RunStats* stats);

  // Encodes launches [first, last) into `buffer` as one or more passes,
  // chained by fences after `*fence_index`.
  void encode_passes(
      Program& program,
      id<MTLCommandBuffer> buffer,
      size_t first,
      size_t last,
      const RunOptions& options,
      size_t* fence_index,
      RunStats* stats);

  // A replay that has been committed and not collected yet.
  struct Flight {
    std::shared_ptr<Program> program;
    id<MTLCommandBuffer> first = nil;
    id<MTLCommandBuffer> last = nil;
    NSMutableArray<id<MTLCommandBuffer>>* all = nil;  // split runs only
    dispatch_semaphore_t done = nil;
    // Stamped by the completion handler (which may outlive this struct).
    std::shared_ptr<std::atomic<uint64_t>> handled;
    RunStats stats;
    uint64_t first_committed_at = 0;
    uint64_t committed_at = 0;
    uint64_t woke_at = 0;
    uint64_t bound = 0;
    bool active = false;
  };

  // The parts of a run.  bind_run: the legality check and the buffer table
  // ("" or a reason; "failed:prepare" needs release_run).  commit_run: every
  // launch encoded and committed.  flight_verdict: "" or the command
  // buffer's error, with the GPU time.  build_outputs, release_run,
  // account_run: the arrays, the clean-up, the counters.
  std::string bind_run(
      Program& program,
      const std::vector<const mx::array*>& inputs,
      const RunOptions& options);
  void commit_run(Program& program, const RunOptions& options, Flight* flight, RunStats* stats);
  std::string flight_verdict(const Flight& flight, RunStats* stats);
  std::string build_outputs(Program& program, std::vector<mx::array>* outputs);
  void release_run(Program& program);
  void account_run(Program& program, const RunStats& stats, const std::string& failure);
  // True when the replay in flight (if any) has finished on the GPU.
  bool flight_left_gpu(uint64_t timeout_ns);
  // encode_early: the drain, the second half of the legality check, the commit.
  std::string finish_early(Program& program, Flight* flight);
  std::string settle_run(Program& program);
  std::string sole_holder(Program& program, const mx::array& first, bool settle_inputs);
  // clone_layout without the waits (the caller has drained MLX).
  static mx::array copy_layout(const mx::array& source, uint64_t min_length, uint64_t pad);
  // Waits for a committed replay: true when it completed within `timeout_ns`.
  bool wait_flight(Flight& flight, const Program& program, uint64_t timeout_ns);

  // Commits `buffer` and sleeps until it completes or `bound_ns` passes.
  bool commit_and_wait(id<MTLCommandBuffer> buffer, uint64_t bound_ns);
  uint64_t wait_bound(const Program& program) const;
  void abandon(id<MTLCommandBuffer> buffer, Program& program);
  bool recovered();

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  uint64_t wait_floor_ns_ = 2'000'000'000ull;
  uint64_t spin_lead_ns_ = 0;
  Flight flight_;
  bool stuck_ = false;
  id<MTLCommandBuffer> stuck_buffer_ = nil;
  std::vector<mx::array> parked_held_;
  // The abandoned replay's program: its own buffers and kernels are in use
  // on the GPU until the command buffer has left it.
  std::vector<std::shared_ptr<Program>> parked_programs_;
  std::vector<mx::allocator::Buffer> parked_fresh_;
  std::mutex mutex_;
  std::vector<mx::array> pins_;
  std::unordered_map<const void*, int> pin_by_buffer_;
  std::vector<PoolSlot> pool_;
  // Health may sample these while a different thread runs a program.
  std::atomic<uint64_t> pool_bytes_{0};
  std::atomic<size_t> pool_slots_{0};
  uint64_t pool_generation_ = 1;
  std::vector<mx::array> held_;  // input handles kept for the length of a run
  std::vector<mx::allocator::Buffer> fresh_;  // output buffers of the run in flight
  std::vector<id<MTLFence>> fences_;  // reused by every run; grown on first need
  id null_encoder_ = nil;
  std::vector<StepTime> profile_;
  RunStats last_;
  RunStats total_;
  uint64_t runs_ = 0;
  uint64_t settle_waits_ = 0;
  uint64_t settle_wait_ns_ = 0;
  uint64_t grown_inputs_ = 0;
  uint64_t grown_bytes_ = 0;
  std::vector<int> grow_;  // input buffers of the run being bound that get a longer copy
  uint64_t grown_now_ = 0;  // ... and how many and how many bytes that was
  uint64_t grown_now_bytes_ = 0;
  std::unordered_map<std::string, uint64_t> refusals_;
  std::string last_short_detail_;
};

// Dtype names as Python spells them.
const char* dtype_name(mx::Dtype dtype);
bool dtype_from_name(const std::string& name, mx::Dtype* out);

}  // namespace mtplx_engine
