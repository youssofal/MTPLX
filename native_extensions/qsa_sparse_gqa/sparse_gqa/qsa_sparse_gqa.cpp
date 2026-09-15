// SPDX-License-Identifier: Apache-2.0
//
// Host side of MTPLX's port of oMLX's direct-index sparse-GQA attention
// (jundot/omlx 7467dce8, Jonathan Spangler, Apache-2.0).  See
// steel_qsa_sparse_gqa.h for provenance, the algorithm, and the list of
// MTPLX-side changes.

#include "sparse_gqa/qsa_sparse_gqa.h"

#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include "sparse_gqa/qsa_sparse_gqa_params.h"

namespace mtplx_native {

namespace {

using namespace mlx::core;

constexpr int kBatch = 1;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kGqa = 12;
constexpr int kHeadDim = 256;
constexpr int kTopK = 512;
constexpr int kHeadPad = 16;
constexpr int kWarps = 2;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get mtplx_native_qsa binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool last_dim_contiguous(const array &arr) { return arr.strides(-1) == 1; }

std::string shape_str(const array &arr) {
  std::ostringstream out;
  out << arr.shape();
  return out.str();
}

bool supported_tile(int key_tile, int dimension_tile) {
  return (dimension_tile == 32 && (key_tile == 128 || key_tile == 256)) ||
         (dimension_tile == 64 && (key_tile == 64 || key_tile == 128));
}

// A single reason string, so the Python gate and the raised exception never
// disagree about why a call was refused.
std::string unsupported_reason(const array &q, const array &k, const array &v,
                               const array &selected, float scale, int q_offset,
                               int key_length, int key_tile, int dimension_tile,
                               Stream stream) {
  if (stream.device == Device::cpu) {
    return "the QSA sparse-GQA kernel has no CPU path";
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || selected.ndim() != 4) {
    return "queries, keys, values, and block ids must all be rank four";
  }
  if (q.dtype() != float16 && q.dtype() != bfloat16) {
    return "queries must be float16 or bfloat16";
  }
  if (k.dtype() != q.dtype() || v.dtype() != q.dtype()) {
    return "queries, keys, and values must share one dtype";
  }
  if (selected.dtype() != uint32 && selected.dtype() != int32) {
    return "block ids must be uint32 or int32";
  }
  if (!last_dim_contiguous(q) || !last_dim_contiguous(k) ||
      !last_dim_contiguous(v) || !last_dim_contiguous(selected)) {
    return "every input must be contiguous in its last dimension";
  }
  if (q.shape(0) != kBatch || q.shape(1) != kQHeads || q.shape(3) != kHeadDim) {
    return "queries must have production shape [1, 24, M, 256]; got " +
           shape_str(q);
  }
  const int rows = q.shape(2);
  if (rows <= 0) {
    return "queries must carry at least one row";
  }
  if (k.shape(0) != kBatch || k.shape(1) != kKvHeads || k.shape(3) != kHeadDim) {
    return "keys must have production shape [1, 2, capacity, 256]; got " +
           shape_str(k);
  }
  if (v.shape() != k.shape()) {
    return "values must have the same full-backing shape as keys";
  }
  if (selected.shape(0) != kBatch || selected.shape(1) != 1 ||
      selected.shape(2) != rows || selected.shape(3) != kTopK) {
    return "block ids must have shape [1, 1, M, 512]; got " +
           shape_str(selected);
  }
  const int capacity = k.shape(2);
  if (key_length <= 0 || key_length > capacity) {
    return "key_length must be a positive count within the K/V backing";
  }
  if (q_offset < 0 || q_offset + rows > key_length) {
    return "queries must be a causal suffix inside key_length";
  }
  if (!std::isfinite(scale)) {
    return "scale must be finite";
  }
  if (!supported_tile(key_tile, dimension_tile)) {
    return "(key_tile, dimension_tile) must be one of "
           "(128,32), (256,32), (64,64), (128,64)";
  }
  return std::string();
}

class QsaSparseGqaPrimitive : public Primitive {
public:
  QsaSparseGqaPrimitive(Stream stream, float scale, int q_offset,
                        int key_length, int key_tile, int dimension_tile)
      : Primitive(stream), scale_(scale), q_offset_(q_offset),
        key_length_(key_length), key_tile_(key_tile),
        dimension_tile_(dimension_tile) {}

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error(
        "[mtplx_native_qsa] qsa_sparse_gqa has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &selected = inputs[3];
    auto &out = outputs[0];

    // Authoritative alignment guard.  The builder below validates the
    // NOMINAL strides an unevaluated array reports; only here are the strides
    // final.  Every K/V/Q/O row is read or written with 128-bit `uint4`
    // accesses, so each leading stride must be a whole number of 16-byte
    // words and the last dimension must be unit stride.  Getting this wrong
    // is silent corruption, not a crash.
    auto require_vector_strides = [](const array &a, const char *which) {
      const int per_word = 16 / static_cast<int>(a.itemsize());
      if (a.strides(-1) != 1) {
        throw std::runtime_error(
            std::string("[mtplx_native_qsa] ") + which +
            " must be contiguous in its last dimension.");
      }
      for (int d = 1; d + 1 < static_cast<int>(a.ndim()); ++d) {
        if (a.strides(d) % per_word != 0) {
          throw std::runtime_error(
              std::string("[mtplx_native_qsa] ") + which +
              " strides must be whole 128-bit words for the vector loads.");
        }
      }
    };
    require_vector_strides(q, "queries");
    require_vector_strides(k, "keys");
    require_vector_strides(v, "values");
    require_vector_strides(out, "output");

    out.set_data(allocator::malloc(out.nbytes()));
    MtplxQsaSparseGqaParams params{
        /* B */ kBatch,
        /* q_heads */ kQHeads,
        /* kv_heads */ kKvHeads,
        /* qL */ q.shape(2),
        /* kL */ key_length_,
        /* topk */ selected.shape(3),
        /* gqa_factor */ kGqa,
        /* q_offset */ q_offset_,
        /* scale */ scale_,
        /* _pad */ 0,
        /* Q_strides */ {q.strides(0), q.strides(1), q.strides(2)},
        /* K_strides */ {k.strides(0), k.strides(1), k.strides(2)},
        /* V_strides */ {v.strides(0), v.strides(1), v.strides(2)},
        /* Topk_strides */
        {selected.strides(0), selected.strides(1), selected.strides(2)},
        /* O_strides */ {out.strides(0), out.strides(1), out.strides(2)}};

    std::string kernel_name;
    concatenate(kernel_name, "mtplx_qsa_sparse_gqa_", type_to_name(q), "_",
                type_to_name(selected), "_bk", key_tile_, "_dc",
                dimension_tile_, "_gqa", kGqa, "_hp", kHeadPad, "_d", kHeadDim,
                "_wm", kWarps);

    auto library =
        device.get_library("mtplx_native_qsa_ext", current_binary_dir());
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_input_array(selected, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(params, 5);
    encoder.dispatch_threadgroups(MTL::Size(q.shape(2), kKvHeads, 1),
                                  MTL::Size(32, kWarps, 1));
  }

  DEFINE_NAME(MtplxQsaSparseGqaAttention)
  DEFINE_INPUT_OUTPUT_SHAPE()

  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const QsaSparseGqaPrimitive &>(other);
    return scale_ == rhs.scale_ && q_offset_ == rhs.q_offset_ &&
           key_length_ == rhs.key_length_ && key_tile_ == rhs.key_tile_ &&
           dimension_tile_ == rhs.dimension_tile_;
  }

  auto state() const {
    return std::make_tuple(scale_, q_offset_, key_length_, key_tile_,
                           dimension_tile_);
  }

private:
  float scale_;
  int q_offset_;
  int key_length_;
  int key_tile_;
  int dimension_tile_;
};

} // namespace

std::string qsa_sparse_gqa_unsupported_reason(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_blocks, float scale, int q_offset, int key_length,
    int key_tile, int dimension_tile, mx::StreamOrDevice s) {
  auto stream = to_stream(s);
  const int resolved =
      key_length < 0 ? (keys.ndim() == 4 ? keys.shape(2) : -1) : key_length;
  return unsupported_reason(queries, keys, values, selected_blocks, scale,
                            q_offset, resolved, key_tile, dimension_tile,
                            stream);
}

mx::array qsa_sparse_gqa_attention(const mx::array &queries,
                                   const mx::array &keys,
                                   const mx::array &values,
                                   const mx::array &selected_blocks,
                                   float scale, int q_offset, int key_length,
                                   int key_tile, int dimension_tile,
                                   mx::StreamOrDevice s) {
  auto stream = to_stream(s);
  const int resolved =
      key_length < 0 ? (keys.ndim() == 4 ? keys.shape(2) : -1) : key_length;
  auto reason =
      unsupported_reason(queries, keys, values, selected_blocks, scale,
                         q_offset, resolved, key_tile, dimension_tile, stream);
  if (!reason.empty()) {
    throw std::invalid_argument("[mtplx_native_qsa.qsa_sparse_gqa] " + reason +
                                ".");
  }

  Shape out_shape = queries.shape();
  return array(std::move(out_shape), queries.dtype(),
               std::make_shared<QsaSparseGqaPrimitive>(
                   stream, scale, q_offset, resolved, key_tile,
                   dimension_tile),
               std::vector<mx::array>{queries, keys, values, selected_blocks});
}

} // namespace mtplx_native
