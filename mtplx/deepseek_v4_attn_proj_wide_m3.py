"""Exact physical-M3 Q4 query-projection lane for DeepSeek-V4-Flash.

This experiment is married to the canonical target verifier's ``[1,3,1024]``
query-rank tensor.  It keeps MLX 0.31.2's affine-Q4/g64 arithmetic association
and changes only reuse: each packed weight word, scale, and affine offset feeds
the three verifier rows before the kernel advances.  Only the 43 body attention
``wq_b`` projections are installed.  The ratio-4 indexers are shape-eligible but
remain dormant below their 512-row threshold on the exact 328+256 workload, so
their 21 ``wq_b`` projections stay stock with O_LORA, MLA/SDPA, caches, small
projections, and the dense MTP block.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from .attention_context import current_attention_phase, current_model_forward_kind


_ENV = "MTPLX_DSV4_ATTN_PROJ_WIDE_M3"
_BODY_LAYERS = 43
_MTP_BLOCKS = 1
_TARGET_ROWS = 3
_K = 1024
_MAIN_N = 32768
_INDEX_N = 8192
_GROUP_SIZE = 64
_BITS = 4
_MODE = "affine"
_ATTN_PROJECTION_NAMES = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")


def deepseek_v4_attn_proj_wide_m3_enabled() -> bool:
    """Read the experimental opt-in only at the runtime install boundary."""

    return os.environ.get(_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


def _validate_projection(module, *, n: int, label: str) -> None:
    """Prove one original-layout affine-Q4 projection at construction."""

    if getattr(module, "bits", None) != _BITS:
        raise ValueError(f"{label} requires bits=4")
    if getattr(module, "group_size", None) != _GROUP_SIZE:
        raise ValueError(f"{label} requires group_size=64")
    if str(getattr(module, "mode", "")).lower() != _MODE:
        raise ValueError(f"{label} requires mode=affine")
    if getattr(module, "bias", None) is not None:
        raise ValueError(f"{label} must not have additive bias")
    arrays = (
        ("packed weight", getattr(module, "weight", None), (n, _K // 8), mx.uint32),
        ("scales", getattr(module, "scales", None), (n, _K // 64), mx.bfloat16),
        (
            "affine biases",
            getattr(module, "biases", None),
            (n, _K // 64),
            mx.bfloat16,
        ),
    )
    for name, value, expected_shape, expected_dtype in arrays:
        if (
            _shape(value) != expected_shape
            or getattr(value, "dtype", None) != expected_dtype
        ):
            raise ValueError(
                f"{label} {name} requires shape={expected_shape} "
                f"dtype={expected_dtype}; got shape={_shape(value)} "
                f"dtype={getattr(value, 'dtype', None)}"
            )


_METAL_HEADER = r"""
using namespace metal;

constant constexpr int M = 3;
constant constexpr int K = 1024;
constant constexpr int VALUES_PER_THREAD = 8;
constant constexpr int BYTES_PER_PACK = 4;
constant constexpr int BLOCK_SIZE = 256;
constant constexpr int NUM_SIMDGROUPS = 2;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int ROWS_PER_THREADGROUP = 8;

template <typename T>
inline float load_vector4_exact(
    const device T* x, thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

inline float qdot4_exact(
    uint packed_weights,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  const thread uint16_t* ws = (const thread uint16_t*)&packed_weights;
  float accum = 0.0f;
  for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
    accum +=
      (x_thread[4 * i] * float(ws[i] & 0x000f) +
       x_thread[4 * i + 1] * float(ws[i] & 0x00f0) +
       x_thread[4 * i + 2] * float(ws[i] & 0x0f00) +
       x_thread[4 * i + 3] * float(ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""


@lru_cache(maxsize=1)
def _affine_q4_wide_m3_kernel(n: int):
    if n != _MAIN_N:
        raise ValueError(f"attention M3-wide unsupported N={n}")
    source = r"""
constexpr int N = __N__;

uint out_row = threadgroup_position_in_grid.x * ROWS_PER_THREADGROUP
    + simdgroup_index_in_threadgroup * RESULTS_PER_SIMDGROUP;
uint lane = thread_index_in_simdgroup;

thread float x_thread[VALUES_PER_THREAD];
thread float result[M][RESULTS_PER_SIMDGROUP] = {{0.0f}};

for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    int n = int(out_row) + row;
    const device uint8_t* row_weights = (const device uint8_t*)w
        + n * (K / 2) + k_block / 2 + lane * BYTES_PER_PACK;
    uint packed_weights = *(const device uint*)row_weights;
    float scale = float(scales[n * (K / 64) + k_block / 64 + lane / 8]);
    float bias = float(biases[n * (K / 64) + k_block / 64 + lane / 8]);
    for (int m = 0; m < M; ++m) {
      const device T* x_ptr = x + m * K + k_block
          + lane * VALUES_PER_THREAD;
      float sum = load_vector4_exact<T>(x_ptr, x_thread);
      result[m][row] += qdot4_exact(
          packed_weights, x_thread, scale, bias, sum);
    }
  }
}

for (int m = 0; m < M; ++m) {
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    float reduced = simd_sum(result[m][row]);
    if (lane == 0) {
      y[m * N + out_row + row] = static_cast<T>(reduced);
    }
  }
}
""".replace("__N__", str(n))
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_attn_q4_g64_wide_m3_k{_K}_n{n}",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_METAL_HEADER,
        source=source,
    )


class _AffineQ4WideM3Projection:
    """One direct original-layout ``[1,3,1024]`` affine-Q4 projection."""

    __slots__ = ("biases", "kernel", "n", "scales", "stock", "weight")

    def __init__(self, stock, *, n: int) -> None:
        _validate_projection(stock, n=n, label=f"attention M3-wide N={n}")
        self.stock = stock
        self.n = int(n)
        self.weight = stock.weight
        self.scales = stock.scales
        self.biases = stock.biases
        self.kernel = _affine_q4_wide_m3_kernel(self.n)

    def __call__(self, x: mx.array) -> mx.array:
        (out,) = self.kernel(
            inputs=[x, self.weight, self.scales, self.biases],
            template=[("T", mx.bfloat16)],
            output_shapes=[(1, _TARGET_ROWS, self.n)],
            output_dtypes=[mx.bfloat16],
            grid=((self.n // 8) * 64, 1, 1),
            threadgroup=(64, 1, 1),
        )
        return out


class _InstalledAttnProjectionWideM3Route:
    """Candidate route with only phase, target kind, and physical-M selection."""

    __slots__ = ("candidate", "stock")

    def __init__(self, stock, candidate) -> None:
        self.stock = stock
        self.candidate = candidate

    def __call__(self, x: mx.array) -> mx.array:
        if (
            current_attention_phase() == "decode_verify"
            and current_model_forward_kind() == "target_verify"
            and tuple(x.shape) == (1, _TARGET_ROWS, _K)
        ):
            return self.candidate(x)
        return self.stock(x)


class _PreparedProjection:
    __slots__ = ("candidate", "label", "owner", "stock")

    def __init__(self, owner, stock, candidate, label: str) -> None:
        self.owner = owner
        self.stock = stock
        self.candidate = candidate
        self.label = label


class _DeepseekV4AttnProjectionWideM3Selector:
    """Prebuilt complete control and candidate arms selected between generations."""

    __slots__ = ("_active_candidate", "projections", "report")

    def __init__(self, projections: tuple[_PreparedProjection, ...]) -> None:
        if len(projections) != 43:
            raise ValueError("attention M3-wide selector requires exactly 43 routes")
        self.projections = projections
        self._active_candidate = False
        self.report = {
            "route": "target_verify_m3_original_q4_attention_projections",
            "logical_input_shape": [1, 3, 1024],
            "body_wq_b_prepared": 43,
            "body_indexer_wq_b_prepared": 0,
            "body_indexer_wq_b_stock": 21,
            "total_q4_projections_prepared": 43,
            "main_geometry": {"k": 1024, "n": 32768, "layers": 43},
            "indexer_geometry_stock": {"k": 1024, "n": 8192, "layers": 21},
            "indexer_activation_threshold_rows": 512,
            "canonical_max_compressed_rows": 146,
            "quantization": "affine_q4_g64",
            "activation_dtype": "bfloat16",
            "mtp_attention_dense_stock": 1,
            "o_lora_stock": 86,
            "small_attention_projections_stock": True,
            "mla_sdpa_cache_stock": True,
            "other_target_widths_stock": [2, 4],
            "ar_prefill_repair_mtp_stock": True,
            "kernel_selfcheck_exact": True,
            "both_arms_preinstalled": True,
            "arm_selection": "between_generations",
            "in_generation_module_rewrites": False,
        }

    @property
    def candidate_selected(self) -> bool:
        return self._active_candidate

    def _bind(self, candidate: bool) -> None:
        if self._active_candidate is bool(candidate):
            return
        for projection in self.projections:
            projection.owner.wq_b = (
                projection.candidate if candidate else projection.stock
            )
        self._active_candidate = bool(candidate)

    def select_control(self) -> None:
        self._bind(False)

    def select_candidate(self) -> None:
        self._bind(True)


def _require_metal() -> None:
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("attention M3-wide installation requires Metal GPU")


def _require_bf16_activation(model) -> None:
    embedding = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embedding is None or not callable(embedding):
        raise ValueError("attention M3-wide requires the canonical embedding")
    output = embedding(mx.zeros((1, 1), dtype=mx.int32))
    mx.eval(output)
    if _shape(output) != (1, 1, 4096) or output.dtype != mx.bfloat16:
        raise ValueError("attention M3-wide requires BF16 target activations")


def _validate_dense_mtp_projection(module, *, shape: tuple[int, int], label: str) -> None:
    if (
        _shape(getattr(module, "weight", None)) != shape
        or getattr(getattr(module, "weight", None), "dtype", None) != mx.bfloat16
        or getattr(module, "scales", None) is not None
        or getattr(module, "biases", None) is not None
        or getattr(module, "bias", None) is not None
    ):
        raise ValueError(f"attention M3-wide requires dense BF16 MTP {label} stock")


def _validate_topology(model, config: dict):
    if str(getattr(model, "model_type", "")).lower() != "deepseek_v4":
        raise ValueError("attention M3-wide requires loaded model_type=deepseek_v4")
    if str((config or {}).get("model_type", "")).lower() != "deepseek_v4":
        raise ValueError("attention M3-wide config requires model_type=deepseek_v4")
    if "DeepseekV4ForCausalLM" not in {
        str(name) for name in (config or {}).get("architectures", ())
    }:
        raise ValueError("attention M3-wide requires DeepseekV4ForCausalLM")
    expected = {
        "hidden_size": 4096,
        "q_lora_rank": _K,
        "num_attention_heads": 64,
        "head_dim": 512,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "num_hidden_layers": _BODY_LAYERS,
        "num_nextn_predict_layers": _MTP_BLOCKS,
    }
    args = getattr(model, "args", None)
    if args is None:
        raise ValueError("attention M3-wide requires loaded ModelArgs")
    for name, value in expected.items():
        if int(getattr(args, name, -1)) != value or int(config.get(name, -1)) != value:
            raise ValueError(f"attention M3-wide requires {name}={value}")
    quantization = config.get("quantization")
    if not isinstance(quantization, dict) or any(
        quantization.get(name) != value
        for name, value in (
            ("bits", _BITS),
            ("group_size", _GROUP_SIZE),
            ("mode", _MODE),
        )
    ):
        raise ValueError("attention M3-wide config requires affine Q4/g64")
    ratios = config.get("compress_ratios")
    expected_ratios = [0, 0] + [
        4 if layer_id % 2 == 0 else 128 for layer_id in range(2, 43)
    ] + [0]
    if ratios != expected_ratios:
        raise ValueError("attention M3-wide compress-ratio topology is not canonical")
    layers = tuple(getattr(model, "layers", ()))
    mtp_blocks = tuple(getattr(model, "mtp_blocks", ()))
    if len(layers) != _BODY_LAYERS or len(mtp_blocks) != _MTP_BLOCKS:
        raise ValueError("attention M3-wide requires 43 body layers and one MTP block")
    return layers, mtp_blocks


def _validate_one_exact(route: _PreparedProjection, probe: mx.array) -> None:
    stock = route.stock(probe)
    candidate = route.candidate.candidate(probe)
    mx.eval(stock, candidate)
    if (
        _shape(stock) != _shape(candidate)
        or getattr(stock, "dtype", None) != mx.bfloat16
        or getattr(candidate, "dtype", None) != mx.bfloat16
        or not bool(mx.array_equal(stock, candidate).item())
    ):
        maximum = float(
            mx.max(mx.abs(stock.astype(mx.float32) - candidate.astype(mx.float32))).item()
        )
        raise ValueError(
            f"attention M3-wide exact self-check failed at {route.label}: "
            f"max_abs={maximum:g}"
        )


def _validate_real_weight_sentinels(routes: tuple[_PreparedProjection, ...]) -> None:
    """Compile and prove representative real-weight projections before publish."""

    if len(routes) != 43:
        raise ValueError("attention M3-wide self-check requires exactly 43 routes")
    values = mx.arange(_TARGET_ROWS * _K, dtype=mx.float32).reshape(1, 3, _K)
    probes = (
        ((values % 31.0) - 15.0).astype(mx.bfloat16) / 8.0,
        ((values % 17.0) - 8.0).astype(mx.bfloat16) / 4.0,
    )
    by_label = {route.label: route for route in routes}
    labels = (
        "body.0.attn.wq_b",
        "body.22.attn.wq_b",
        "body.42.attn.wq_b",
    )
    for index, label in enumerate(labels):
        _validate_one_exact(by_label[label], probes[index % len(probes)])


def prepare_deepseek_v4_attn_proj_wide_m3_routes(model, config: dict):
    """Construct and authenticate both complete arms without enabling either."""

    layers, mtp_blocks = _validate_topology(model, config)
    _require_metal()
    _require_bf16_activation(model)
    mtp_attn = mtp_blocks[0].attn
    for name, shape in (
        ("wq_a", (1024, 4096)),
        ("wq_b", (32768, 1024)),
        ("wkv", (512, 4096)),
        ("wo_a", (8192, 4096)),
        ("wo_b", (4096, 8192)),
    ):
        _validate_dense_mtp_projection(
            getattr(mtp_attn, name, None), shape=shape, label=f"attn.{name}"
        )
    mtp_identity = tuple(getattr(mtp_attn, name) for name in _ATTN_PROJECTION_NAMES)
    olora_identity = tuple((layer.attn.wo_a, layer.attn.wo_b) for layer in layers)

    routes: list[_PreparedProjection] = []
    for layer_id, layer in enumerate(layers):
        attn = layer.attn
        expected_ratio = config["compress_ratios"][layer_id]
        if int(getattr(attn, "compress_ratio", -1)) != expected_ratio:
            raise ValueError(f"body layer {layer_id} attention ratio is invalid")
        stock = attn.wq_b
        direct = _AffineQ4WideM3Projection(stock, n=_MAIN_N)
        routes.append(
            _PreparedProjection(
                attn,
                stock,
                _InstalledAttnProjectionWideM3Route(stock, direct),
                f"body.{layer_id}.attn.wq_b",
            )
        )
        has_indexer = hasattr(attn, "indexer")
        if has_indexer != (expected_ratio == 4):
            raise ValueError(f"body layer {layer_id} indexer topology is invalid")
        if has_indexer:
            _validate_projection(
                attn.indexer.wq_b,
                n=_INDEX_N,
                label=f"body.{layer_id}.attn.indexer.wq_b stock",
            )

    prepared = tuple(routes)
    _validate_real_weight_sentinels(prepared)
    if tuple(getattr(mtp_attn, name) for name in _ATTN_PROJECTION_NAMES) != mtp_identity:
        raise ValueError("attention M3-wide construction changed the MTP attention")
    if tuple((layer.attn.wo_a, layer.attn.wo_b) for layer in layers) != olora_identity:
        raise ValueError("attention M3-wide construction changed O_LORA")
    return _DeepseekV4AttnProjectionWideM3Selector(prepared)


def install_deepseek_v4_attn_proj_wide_m3(model, config: dict) -> dict:
    """Install the prechecked selector and select the candidate once at load."""

    selector = prepare_deepseek_v4_attn_proj_wide_m3_routes(model, config)
    model._mtplx_dsv4_attn_proj_wide_m3_selector = selector
    selector.select_candidate()
    return dict(selector.report)


def select_deepseek_v4_attn_proj_wide_m3_arm(model, enabled: bool) -> None:
    """Bind one already-built complete arm between benchmark generations."""

    selector = getattr(model, "_mtplx_dsv4_attn_proj_wide_m3_selector", None)
    if type(selector) is not _DeepseekV4AttnProjectionWideM3Selector:
        raise ValueError("attention M3-wide arm selection requires an installed plan")
    if enabled:
        selector.select_candidate()
    else:
        selector.select_control()
