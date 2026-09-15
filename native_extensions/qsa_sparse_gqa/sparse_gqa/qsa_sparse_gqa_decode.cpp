// SPDX-License-Identifier: Apache-2.0
//
// Host side of the SPLIT-K decode variant of MTPLX's direct-index sparse-GQA
// attention.  See steel_qsa_sparse_gqa_decode.h for provenance, the algorithm
// and the list of differences from the phase-1 prefill kernel.

#include "sparse_gqa/qsa_sparse_gqa_decode.h"

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include "sparse_gqa/qsa_sparse_gqa_decode_params.h"

namespace mtplx_native {

namespace {

using namespace mlx::core;

constexpr int kBatch = 1;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kGqa = 12;
constexpr int kHeadDim = 256;
constexpr int kTopK = 512;
constexpr int kCompressRatio = 4;
constexpr int kHeadPad = 16;
constexpr int kWarps = 2;
//: topk*ratio selected tokens plus the at-most ratio-1 causal tail tokens.
constexpr int kSelectedTokens = kTopK * kCompressRatio + (kCompressRatio - 1);
//: Partial rows carry [O(head_dim) | m | l].
constexpr int kPartialLd = kHeadDim + 2;
//: A merge threadgroup is one thread per head dim; keep the cap honest.
constexpr int kMaxKeySplits = 64;

std::string current_binary_dir_decode() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&current_binary_dir_decode), &info)) {
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

std::string unsupported_reason(const array &q, const array &k, const array &v,
                               const array &selected, const array &offset,
                               float scale, int key_length, int key_tile,
                               int dimension_tile, int key_splits,
                               Stream stream) {
  if (stream.device == Device::cpu) {
    return "the QSA sparse-GQA decode kernel has no CPU path";
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
  if (offset.dtype() != int32) {
    return "the query offset must be an int32 array";
  }
  if (offset.size() != 1) {
    return "the query offset must hold exactly one element";
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
  if (rows > key_length) {
    return "the query rows must fit inside key_length";
  }
  if (!std::isfinite(scale)) {
    return "scale must be finite";
  }
  if (!supported_tile(key_tile, dimension_tile)) {
    return "(key_tile, dimension_tile) must be one of "
           "(128,32), (256,32), (64,64), (128,64)";
  }
  if (key_splits < 1 || key_splits > kMaxKeySplits) {
    return "key_splits must be in [1, 64]";
  }
  return std::string();
}

class QsaSparseGqaDecodePrimitive : public Primitive {
public:
  QsaSparseGqaDecodePrimitive(Stream stream, float scale, int key_length,
                              int key_tile, int dimension_tile, int key_splits)
      : Primitive(stream), scale_(scale), key_length_(key_length),
        key_tile_(key_tile), dimension_tile_(dimension_tile),
        key_splits_(key_splits) {}

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error(
        "[mtplx_native_qsa] qsa_sparse_gqa_decode has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &selected = inputs[3];
    const auto &offset = inputs[4];
    auto &out = outputs[0];

    // Authoritative alignment guard: only here are the strides final.  Every
    // K/V/Q row is read with 128-bit `uint4` accesses, so each leading stride
    // must be a whole number of 16-byte words and the last dimension must be
    // unit stride.  Getting this wrong is silent corruption, not a crash.
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

    const int rows = q.shape(2);
    int n_tiles = 0;
    int tiles_per_split = 0;
    int n_splits = 0;
    qsa_sparse_gqa_decode_split_geometry(kSelectedTokens, key_tile_,
                                         key_splits_, &n_tiles,
                                         &tiles_per_split, &n_splits);

    // Partial online-softmax states: [n_splits, q_heads, rows, head_dim + 2].
    // A temporary, so MLX frees it when the command buffer completes; both
    // dispatches live in ONE encoder, so there is no host sync between them
    // and Metal's in-encoder ordering is the barrier.
    Shape partial_shape = {n_splits, kQHeads, rows, kPartialLd};
    array partial(partial_shape, float32, nullptr, std::vector<array>{});
    partial.set_data(allocator::malloc(partial.nbytes()));

    out.set_data(allocator::malloc(out.nbytes()));

    auto library =
        device.get_library("mtplx_native_qsa_ext", current_binary_dir_decode());
    auto &encoder = metal::get_command_encoder(stream);
    encoder.add_temporary(partial);

    MtplxQsaSparseGqaDecodeParams split_params{
        /* q_heads */ kQHeads,
        /* kv_heads */ kKvHeads,
        /* qL */ rows,
        /* kL */ key_length_,
        /* topk */ selected.shape(3),
        /* gqa_factor */ kGqa,
        /* n_tiles */ n_tiles,
        /* tiles_per_split */ tiles_per_split,
        /* n_splits */ n_splits,
        /* partial_ld */ kPartialLd,
        /* scale */ scale_,
        /* _pad */ 0,
        /* Q_strides */ {q.strides(0), q.strides(1), q.strides(2)},
        /* K_strides */ {k.strides(0), k.strides(1), k.strides(2)},
        /* V_strides */ {v.strides(0), v.strides(1), v.strides(2)},
        /* Topk_strides */
        {selected.strides(0), selected.strides(1), selected.strides(2)}};

    std::string split_name;
    concatenate(split_name, "mtplx_qsa_sparse_gqa_decode_split_",
                type_to_name(q), "_", type_to_name(selected), "_bk", key_tile_,
                "_dc", dimension_tile_, "_gqa", kGqa, "_hp", kHeadPad, "_d",
                kHeadDim, "_wm", kWarps);
    auto split_kernel = device.get_kernel(split_name, library);
    encoder.set_compute_pipeline_state(split_kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_input_array(selected, 3);
    encoder.set_input_array(offset, 4);
    encoder.set_output_array(partial, 5);
    encoder.set_bytes(split_params, 6);
    encoder.dispatch_threadgroups(MTL::Size(rows, kKvHeads, n_splits),
                                  MTL::Size(32, kWarps, 1));

    MtplxQsaSparseGqaMergeParams merge_params{
        /* q_heads */ kQHeads,
        /* qL */ rows,
        /* head_dim */ kHeadDim,
        /* n_splits */ n_splits,
        /* partial_ld */ kPartialLd,
        /* _pad */ 0,
        /* O_strides */ {out.strides(0), out.strides(1), out.strides(2)}};

    std::string merge_name;
    concatenate(merge_name, "mtplx_qsa_sparse_gqa_decode_merge_",
                type_to_name(q), "_d", kHeadDim);
    auto merge_kernel = device.get_kernel(merge_name, library);
    encoder.set_compute_pipeline_state(merge_kernel);
    encoder.set_input_array(partial, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(merge_params, 2);
    encoder.dispatch_threadgroups(MTL::Size(kQHeads * rows, 1, 1),
                                  MTL::Size(kHeadDim, 1, 1));
  }

  DEFINE_NAME(MtplxQsaSparseGqaDecode)
  DEFINE_INPUT_OUTPUT_SHAPE()

  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const QsaSparseGqaDecodePrimitive &>(other);
    return scale_ == rhs.scale_ && key_length_ == rhs.key_length_ &&
           key_tile_ == rhs.key_tile_ &&
           dimension_tile_ == rhs.dimension_tile_ &&
           key_splits_ == rhs.key_splits_;
  }

  auto state() const {
    return std::make_tuple(scale_, key_length_, key_tile_, dimension_tile_,
                           key_splits_);
  }

private:
  float scale_;
  int key_length_;
  int key_tile_;
  int dimension_tile_;
  int key_splits_;
};

} // namespace

void qsa_sparse_gqa_decode_split_geometry(int selected_tokens, int key_tile,
                                          int key_splits, int *n_tiles,
                                          int *tiles_per_split, int *n_splits) {
  const int tiles = (selected_tokens + key_tile - 1) / key_tile;
  int splits = std::min(key_splits, tiles);
  if (splits < 1) {
    splits = 1;
  }
  const int per_split = (tiles + splits - 1) / splits;
  // Round the split count back down so the LAST split is never empty: with
  // 17 tiles and 8 requested splits, per_split is 3 and six splits cover the
  // work, so dispatching eight would leave two threadgroups writing nothing.
  const int exact_splits = (tiles + per_split - 1) / per_split;
  *n_tiles = tiles;
  *tiles_per_split = per_split;
  *n_splits = exact_splits;
}

std::string qsa_sparse_gqa_decode_unsupported_reason(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_blocks, const mx::array &query_offset,
    float scale, int key_length, int key_tile, int dimension_tile,
    int key_splits, mx::StreamOrDevice s) {
  auto stream = to_stream(s);
  const int resolved =
      key_length < 0 ? (keys.ndim() == 4 ? keys.shape(2) : -1) : key_length;
  return unsupported_reason(queries, keys, values, selected_blocks,
                            query_offset, scale, resolved, key_tile,
                            dimension_tile, key_splits, stream);
}

mx::array qsa_sparse_gqa_decode(const mx::array &queries, const mx::array &keys,
                                const mx::array &values,
                                const mx::array &selected_blocks,
                                const mx::array &query_offset, float scale,
                                int key_length, int key_tile,
                                int dimension_tile, int key_splits,
                                mx::StreamOrDevice s) {
  auto stream = to_stream(s);
  const int resolved =
      key_length < 0 ? (keys.ndim() == 4 ? keys.shape(2) : -1) : key_length;
  auto reason =
      unsupported_reason(queries, keys, values, selected_blocks, query_offset,
                         scale, resolved, key_tile, dimension_tile, key_splits,
                         stream);
  if (!reason.empty()) {
    throw std::invalid_argument(
        "[mtplx_native_qsa.qsa_sparse_gqa_decode] " + reason + ".");
  }

  Shape out_shape = queries.shape();
  return array(std::move(out_shape), queries.dtype(),
               std::make_shared<QsaSparseGqaDecodePrimitive>(
                   stream, scale, resolved, key_tile, dimension_tile,
                   key_splits),
               std::vector<mx::array>{queries, keys, values, selected_blocks,
                                      query_offset});
}

} // namespace mtplx_native
