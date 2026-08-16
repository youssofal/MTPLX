"""Fixed-M3 Q6 ``wq_b`` fused with row-owned Q-head norm and RoPE.

One 256-threadgroup owns one of the 64 output heads and all three verifier
rows.  It replays the official-wheel Q6/G128 affine QMV reduction for each
output component, stores the resulting BF16 head in threadgroup memory, then
replays the stock 32-lane per-head RMSNorm reduction and interleaved-RoPE
arithmetic.

This is intentionally only a construction-bound micro candidate.  It has a
small grid (64 threadgroups) and high register pressure, so it is neither a
throughput claim nor a parity claim until its guarded GPU bracket passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np

import mlx.core as mx


_M = 3
_K = 1024
_N = 32768
_HEADS = 64
_HEAD_DIM = 512
_ROPE_DIM = 64
_EPS = 1e-6

# Provenance of the recovered pre-geometry session snapshots. These identify
# the historical inputs, not this module after its staging publisher was added.
RECORDED_PRE_GEOMETRY_SOURCE_SHA256 = (
    "2eb4ce3d5bae9c9b71574d17fedfd37b6755d94299cb4ed6b01d2015c5f8f9a1"
)
RECORDED_PRE_GEOMETRY_TEST_SHA256 = (
    "0c551aa8d7f865454d3a9b6d22f46af5115e00a2fc48c5db82225516c2a77cbb"
)


class M3WQBNormRopeContractError(ValueError):
    """The loaded projection cannot use the fixed 0731 micro candidate."""


@dataclass(frozen=True, slots=True)
class M3WQBNormRopeContract:
    """The immutable Q6 wq_b and post-projection head geometry."""

    bits: int = 6
    group_size: int = 128
    k: int = _K
    n: int = _N
    heads: int = _HEADS
    head_dim: int = _HEAD_DIM
    rope_dim: int = _ROPE_DIM
    eps: float = _EPS
    dtype: mx.Dtype = mx.bfloat16

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.n, self.k * self.bits // 32)

    @property
    def metadata_shape(self) -> tuple[int, int]:
        return (self.n, self.k // self.group_size)

    def valid(self) -> bool:
        return (
            (self.bits, self.group_size, self.k, self.n) == (6, 128, _K, _N)
            and (self.heads, self.head_dim, self.rope_dim)
            == (
                _HEADS,
                _HEAD_DIM,
                _ROPE_DIM,
            )
            and self.eps == _EPS
            and self.dtype == mx.bfloat16
        )


def _shape(value: Any) -> tuple[int, ...] | None:
    try:
        return tuple(int(dimension) for dimension in value.shape)
    except (AttributeError, TypeError, ValueError):
        return None


def validate_0731_m3_wqb_qnorm_rope(
    projection: Any,
    contract: M3WQBNormRopeContract | None = None,
) -> M3WQBNormRopeContract:
    """Validate fixed Q6/G128 storage once before publishing the callable."""

    fixed = M3WQBNormRopeContract() if contract is None else contract
    if not fixed.valid():
        raise M3WQBNormRopeContractError(
            "candidate requires fixed 0731 Q6/G128 geometry"
        )
    if (
        int(getattr(projection, "bits", 0) or 0) != fixed.bits
        or int(getattr(projection, "group_size", 0) or 0) != fixed.group_size
        or str(getattr(projection, "mode", "")).lower() != "affine"
        or getattr(projection, "bias", None) is not None
    ):
        raise M3WQBNormRopeContractError("candidate requires affine Q6/G128 wq_b")
    if (
        _shape(getattr(projection, "weight", None)) != fixed.weight_shape
        or _shape(getattr(projection, "scales", None)) != fixed.metadata_shape
        or _shape(getattr(projection, "biases", None)) != fixed.metadata_shape
        or getattr(getattr(projection, "weight", None), "dtype", None) != mx.uint32
        or getattr(getattr(projection, "scales", None), "dtype", None) != mx.bfloat16
        or getattr(getattr(projection, "biases", None), "dtype", None) != mx.bfloat16
    ):
        raise M3WQBNormRopeContractError(
            "candidate requires exact packed wq_b Q6 storage"
        )
    return fixed


def m3_wqb_qnorm_rope_metal_source(*, capture_projection: bool = False) -> str:
    """Return the production source, or its test-only raw-projection variant."""

    source = r"""
constexpr uint M = 3;
constexpr uint K = 1024;
constexpr uint N = 32768;
constexpr uint HEADS = 64;
constexpr uint HEAD_DIM = 512;
constexpr uint ROPE_DIM = 64;
constexpr float EPS = 1e-6f;
constexpr uint GS = 128;
constexpr uint PACKS_PER_THREAD = 2;
constexpr uint PACK_FACTOR = 4;
constexpr uint BYTES_PER_PACK = 3;
constexpr uint VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;
constexpr uint RESULTS_PER_SIMDGROUP = 4;
constexpr uint SIMDGROUPS = 8;
constexpr uint ROWS_PER_TILE = SIMDGROUPS * RESULTS_PER_SIMDGROUP;
constexpr uint TILES_PER_HEAD = HEAD_DIM / ROWS_PER_TILE;
constexpr uint NORM_LANES = 32;
constexpr uint NORM_READS = 4;
constexpr uint NORM_BLOCKS = HEAD_DIM / (NORM_LANES * NORM_READS);

uint tid = thread_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;
uint simd_gid = simdgroup_index_in_threadgroup;
uint head = threadgroup_position_in_grid.x;
uint row_bytes = K * 6 / 8;
uint group_count = K / GS;
threadgroup T q_shared[M][HEAD_DIM];
threadgroup float norm_scale[M];

for (uint tile = 0; tile < TILES_PER_HEAD; ++tile) {
  uint out_row = head * HEAD_DIM + tile * ROWS_PER_TILE + simd_gid * RESULTS_PER_SIMDGROUP;
  const device uchar* ws = (const device uchar*)w + out_row * row_bytes + simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device T* sc = scales + out_row * group_count + simd_lid / (GS / VALUES_PER_THREAD);
  const device T* bs = biases + out_row * group_count + simd_lid / (GS / VALUES_PER_THREAD);
  const device T* x0 = x + simd_lid * VALUES_PER_THREAD;
  const device T* x1 = x0 + K;
  const device T* x2 = x1 + K;
  float result0[RESULTS_PER_SIMDGROUP] = {0.0f};
  float result1[RESULTS_PER_SIMDGROUP] = {0.0f};
  float result2[RESULTS_PER_SIMDGROUP] = {0.0f};

  for (uint k = 0; k < K; k += BLOCK_SIZE) {
    thread float x0_thread[VALUES_PER_THREAD];
    thread float x1_thread[VALUES_PER_THREAD];
    thread float x2_thread[VALUES_PER_THREAD];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    for (uint i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum0 += x0[i] + x0[i + 1] + x0[i + 2] + x0[i + 3];
      sum1 += x1[i] + x1[i + 1] + x1[i + 2] + x1[i + 3];
      sum2 += x2[i] + x2[i + 1] + x2[i + 2] + x2[i + 3];
      x0_thread[i] = x0[i]; x0_thread[i + 1] = x0[i + 1] / 64.0f;
      x0_thread[i + 2] = x0[i + 2] / 16.0f; x0_thread[i + 3] = x0[i + 3] / 4.0f;
      x1_thread[i] = x1[i]; x1_thread[i + 1] = x1[i + 1] / 64.0f;
      x1_thread[i + 2] = x1[i + 2] / 16.0f; x1_thread[i + 3] = x1[i + 3] / 4.0f;
      x2_thread[i] = x2[i]; x2_thread[i + 1] = x2[i + 1] / 64.0f;
      x2_thread[i + 2] = x2[i + 2] / 16.0f; x2_thread[i + 3] = x2[i + 3] / 4.0f;
    }
    for (uint output_row = 0; output_row < RESULTS_PER_SIMDGROUP; ++output_row) {
      const device uchar* wl = ws + output_row * row_bytes;
      const device T* sl = sc + output_row * group_count;
      const device T* bl = bs + output_row * group_count;
      thread uchar w_thread[PACKS_PER_THREAD * BYTES_PER_PACK];
      for (uint i = 0; i < PACKS_PER_THREAD * BYTES_PER_PACK; ++i) w_thread[i] = wl[i];
      float s = sl[0], b = bl[0];
      float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f;
      const thread uchar* wp = w_thread;
      const thread float* xp0 = x0_thread; const thread float* xp1 = x1_thread; const thread float* xp2 = x2_thread;
      for (uint i = 0; i < VALUES_PER_THREAD / 4; ++i) {
        xp0 += 4 * i; xp1 += 4 * i; xp2 += 4 * i; wp += 3 * i;
        dot0 += (wp[0] & 0x3f) * xp0[0]; dot1 += (wp[0] & 0x3f) * xp1[0]; dot2 += (wp[0] & 0x3f) * xp2[0];
        dot0 += (wp[0] & 0xc0) * xp0[1]; dot1 += (wp[0] & 0xc0) * xp1[1]; dot2 += (wp[0] & 0xc0) * xp2[1];
        dot0 += (wp[1] & 0x0f) * (xp0[1] * 256.0f); dot1 += (wp[1] & 0x0f) * (xp1[1] * 256.0f); dot2 += (wp[1] & 0x0f) * (xp2[1] * 256.0f);
        dot0 += (wp[1] & 0xf0) * xp0[2]; dot1 += (wp[1] & 0xf0) * xp1[2]; dot2 += (wp[1] & 0xf0) * xp2[2];
        dot0 += (wp[2] & 0x03) * (xp0[2] * 256.0f); dot1 += (wp[2] & 0x03) * (xp1[2] * 256.0f); dot2 += (wp[2] & 0x03) * (xp2[2] * 256.0f);
        dot0 += (wp[2] & 0xfc) * xp0[3]; dot1 += (wp[2] & 0xfc) * xp1[3]; dot2 += (wp[2] & 0xfc) * xp2[3];
      }
      result0[output_row] += s * dot0 + sum0 * b;
      result1[output_row] += s * dot1 + sum1 * b;
      result2[output_row] += s * dot2 + sum2 * b;
    }
    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR; sc += BLOCK_SIZE / GS; bs += BLOCK_SIZE / GS;
    x0 += BLOCK_SIZE; x1 += BLOCK_SIZE; x2 += BLOCK_SIZE;
  }
  for (uint output_row = 0; output_row < RESULTS_PER_SIMDGROUP; ++output_row) {
    uint d = tile * ROWS_PER_TILE + simd_gid * RESULTS_PER_SIMDGROUP + output_row;
    float r0 = simd_sum(result0[output_row]); float r1 = simd_sum(result1[output_row]); float r2 = simd_sum(result2[output_row]);
    if (simd_lid == 0) {
      q_shared[0][d] = T(r0); q_shared[1][d] = T(r1); q_shared[2][d] = T(r2);
      /* CAPTURE_PROJECTION */
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint row = 0; row < M; ++row) {
  // Clone the 0.31.2 FP32 row_reduce_simple tree for a 512-element row:
  // 32 lanes, four strided 128-value blocks, and four contiguous reads/lane.
  if (simd_gid == 0) {
    float sum = 0.0f;
    for (uint group = 0; group < NORM_BLOCKS; ++group) {
      uint d = group * NORM_LANES * NORM_READS + simd_lid * NORM_READS;
      sum = float(q_shared[row][d]) * float(q_shared[row][d]) + sum;
      sum = float(q_shared[row][d + 1]) * float(q_shared[row][d + 1]) + sum;
      sum = float(q_shared[row][d + 2]) * float(q_shared[row][d + 2]) + sum;
      sum = float(q_shared[row][d + 3]) * float(q_shared[row][d + 3]) + sum;
    }
    sum = simd_sum(sum);
    if (simd_lid == 0) {
      float mean = sum * (1.0f / float(HEAD_DIM));
      norm_scale[row] = metal::precise::rsqrt(mean + EPS);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = norm_scale[row];
  const device float* c = cos + row * (ROPE_DIM / 2);
  const device float* s = sin + row * (ROPE_DIM / 2);
  device T* out = output + row * N + head * HEAD_DIM;
  for (uint d = tid; d < HEAD_DIM; d += 256) {
    if (d < HEAD_DIM - ROPE_DIM) {
      out[d] = T(float(q_shared[row][d]) * scale);
      continue;
    }
    uint r = d - (HEAD_DIM - ROPE_DIM);
    if ((r & 1) != 0) continue;
    uint pair = r / 2;
    T normalized0 = T(float(q_shared[row][d]) * scale);
    T normalized1 = T(float(q_shared[row][d + 1]) * scale);
    float x0 = float(normalized0); float x1 = float(normalized1);
    // The eager stock graph materializes each FP32 product in a separate
    // binary kernel before its add/subtract.  ``precise`` preserves those
    // intermediate roundings inside this one-launch candidate.
    float rope0_lhs = metal::precise::fma(x0, c[pair], 0.0f);
    float rope0_rhs = metal::precise::fma(x1, s[pair], 0.0f);
    float rope1_lhs = metal::precise::fma(x0, s[pair], 0.0f);
    float rope1_rhs = metal::precise::fma(x1, c[pair], 0.0f);
    out[d] = T(rope0_lhs - rope0_rhs);
    out[d + 1] = T(rope1_lhs + rope1_rhs);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""
    capture = """
      device T* q0 = projection + head * HEAD_DIM + d;
      device T* q1 = q0 + N;
      device T* q2 = q1 + N;
      q0[0] = T(r0); q1[0] = T(r1); q2[0] = T(r2);"""
    return source.replace(
        "/* CAPTURE_PROJECTION */", capture if capture_projection else ""
    )


@lru_cache(maxsize=1)
def _build_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_0731_m3_q6_wqb_qnorm_rope",
        input_names=["x", "w", "scales", "biases", "cos", "sin"],
        output_names=["output"],
        header="using namespace metal;",
        source=m3_wqb_qnorm_rope_metal_source(),
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=1)
def _build_debug_kernel():
    """Build the test-only twin that exposes the raw fused BF16 projection."""

    return mx.fast.metal_kernel(
        name="mtplx_dsv4_0731_m3_q6_wqb_qnorm_rope_debug",
        input_names=["x", "w", "scales", "biases", "cos", "sin"],
        output_names=["projection", "output"],
        header="using namespace metal;",
        source=m3_wqb_qnorm_rope_metal_source(capture_projection=True),
        ensure_row_contiguous=False,
    )


class BoundM3WQBNormRope:
    """Prepared one-launch callable; all geometry checks occurred before binding."""

    __slots__ = (
        "_kernel",
        "biases",
        "contract",
        "grid",
        "input_shape",
        "output_shape",
        "scales",
        "threadgroup",
        "weight",
    )

    def __init__(self, projection: Any, contract: M3WQBNormRopeContract) -> None:
        self.contract = contract
        self.weight = projection.weight
        self.scales = projection.scales
        self.biases = projection.biases
        self._kernel = _build_kernel()
        self.input_shape = (1, _M, _K)
        self.output_shape = (1, _M, _HEADS, _HEAD_DIM)
        self.grid = (_HEADS * 256, 1, 1)
        self.threadgroup = (256, 1, 1)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        (output,) = self._kernel(
            inputs=[
                x.reshape(_M, _K),
                self.weight,
                self.scales,
                self.biases,
                cos.reshape(_M, _ROPE_DIM // 2),
                sin.reshape(_M, _ROPE_DIM // 2),
            ],
            template=[("T", mx.bfloat16)],
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=[(_M, _N)],
            output_dtypes=[mx.bfloat16],
        )
        return output.reshape(self.output_shape)


def build_0731_m3_wqb_qnorm_rope(
    projection: Any,
) -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    """Bind the fixed Q6/G128 M3 candidate without a hot-path fallback."""

    contract = validate_0731_m3_wqb_qnorm_rope(projection)
    return BoundM3WQBNormRope(projection, contract)


class _PreboundWQBQHeadRoute:
    """Construction-selected fixed-M3 route or the captured stock route."""

    __slots__ = ("candidate", "stock")

    def __init__(
        self,
        stock: Callable[[mx.array, mx.array, mx.array], mx.array],
        candidate: Callable[[mx.array, mx.array, mx.array], mx.array],
    ) -> None:
        self.stock = stock
        self.candidate = candidate

    def __call__(self, qr: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        if int(qr.shape[1]) == _M:
            return self.candidate(qr, cos, sin)
        return self.stock(qr, cos, sin)


def prebind_wqb_qhead_route(
    stock: Callable[[mx.array, mx.array, mx.array], mx.array],
    candidate: Callable[[mx.array, mx.array, mx.array], mx.array],
) -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    return _PreboundWQBQHeadRoute(stock, candidate)


@dataclass(frozen=True, slots=True)
class PreparedWQBQHeadM3Routes:
    """Self-checked 43-layer route bank awaiting atomic publication."""

    attentions: tuple[Any, ...]
    stock_routes: tuple[Callable[[mx.array, mx.array, mx.array], mx.array], ...]
    candidate_routes: tuple[Callable[[mx.array, mx.array, mx.array], mx.array], ...]
    published_routes: tuple[Callable[[mx.array, mx.array, mx.array], mx.array], ...]
    q6_count: int
    exact_selfchecked: int

    def publish(self) -> None:
        try:
            for attention, route in zip(self.attentions, self.published_routes):
                attention._q_projection_qhead_route = route
        except Exception as publication_error:
            try:
                self.restore()
            except Exception as restoration_error:
                publication_error.add_note(
                    f"WQB-qhead rollback also raised: {restoration_error}"
                )
            raise

    def restore(self) -> None:
        errors = []
        for attention, stock in zip(self.attentions, self.stock_routes):
            try:
                attention._q_projection_qhead_route = stock
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("WQB-qhead route restoration failed", errors)


def prepare_wqb_qhead_m3(
    layers: Any,
    *,
    exact_selfcheck: Callable[
        [
            Callable[[mx.array, mx.array, mx.array], mx.array],
            Callable[[mx.array, mx.array, mx.array], mx.array],
            int,
        ],
        bool,
    ],
) -> PreparedWQBQHeadM3Routes:
    """Validate, build, and exact-self-check all routes without publishing."""

    layer_tuple = tuple(layers)
    if len(layer_tuple) != 43:
        raise M3WQBNormRopeContractError(
            "fixed M3 wq_b plus q-head preparation requires exactly 43 layers"
        )
    if not callable(exact_selfcheck):
        raise M3WQBNormRopeContractError("exact WQB-qhead self-check is required")

    validated = []
    for layer_index, layer in enumerate(layer_tuple):
        try:
            attention = layer.attn
            projection = attention.wq_b
            stock_route = attention._q_projection_qhead_route
        except AttributeError as exc:
            raise M3WQBNormRopeContractError(
                f"layer {layer_index} lacks the q-projection/post route"
            ) from exc
        if not callable(stock_route):
            raise M3WQBNormRopeContractError(
                f"layer {layer_index} q-projection/post stock route is not callable"
            )
        validate_0731_m3_wqb_qnorm_rope(projection)
        validated.append((attention, projection, stock_route))

    staged = []
    rejected_selfcheck = None
    for layer_index, (attention, projection, stock_route) in enumerate(validated):
        candidate = build_0731_m3_wqb_qnorm_rope(projection)
        try:
            passed = exact_selfcheck(stock_route, candidate, layer_index)
        except Exception as exc:
            raise M3WQBNormRopeContractError(
                f"layer {layer_index} fused M3 exact self-check raised"
            ) from exc
        if not passed and rejected_selfcheck is None:
            rejected_selfcheck = layer_index
        staged.append(
            (
                attention,
                stock_route,
                candidate,
                prebind_wqb_qhead_route(stock_route, candidate),
            )
        )

    if rejected_selfcheck is not None:
        raise M3WQBNormRopeContractError(
            f"layer {rejected_selfcheck} fused M3 exact self-check failed"
        )

    return PreparedWQBQHeadM3Routes(
        attentions=tuple(attention for attention, _, _, _ in staged),
        stock_routes=tuple(stock for _, stock, _, _ in staged),
        candidate_routes=tuple(candidate for _, _, candidate, _ in staged),
        published_routes=tuple(route for _, _, _, route in staged),
        q6_count=len(staged),
        exact_selfchecked=len(staged),
    )


def q_head_norm_rope_cpu_oracle(
    q: np.ndarray, cos: np.ndarray, sin: np.ndarray, *, eps: float, rope_dim: int
) -> np.ndarray:
    """Independent CPU oracle for row-owned per-head norm and RoPE semantics."""

    values = np.asarray(q, dtype=np.float32)
    c = np.asarray(cos, dtype=np.float32)
    s = np.asarray(sin, dtype=np.float32)
    normalized = values * np.reciprocal(
        np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + float(eps))
    )
    out = normalized.copy()
    tail = normalized[..., -int(rope_dim) :].reshape(*normalized.shape[:-1], -1, 2)
    rotated = np.empty_like(tail)
    rotated[..., 0] = (
        tail[..., 0] * c[None, :, None, :] - tail[..., 1] * s[None, :, None, :]
    )
    rotated[..., 1] = (
        tail[..., 0] * s[None, :, None, :] + tail[..., 1] * c[None, :, None, :]
    )
    out[..., -int(rope_dim) :] = rotated.reshape(*normalized.shape[:-1], int(rope_dim))
    return out
