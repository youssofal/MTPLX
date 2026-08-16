"""Construction-bound official-wheel fixed-M3 affine-Q6 0731 ``attn.wo_b``.

The pinned full 0731 artifact stores all 43 body ``wo_b`` projections as one
unbatched affine Q6/group-128 U32 matrix with BF16 scales and biases.  This
module derives the K=8192 loop directly from MLX ``qmv_fast_impl``: each of the
three rows retains its own Q6 load, qdot accumulation, and simd reduction tree.
It neither installs itself nor provides an enabled-path fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import mlx.core as mx


_M = 3
_K = 8192
_N = 4096
_BITS = 6
_GROUP_SIZE = 128


class M3WOBContractError(ValueError):
    """A projection cannot use the fixed physical-M3 0731 wo_b primitive."""


@dataclass(frozen=True, slots=True)
class M3WOBContract:
    """Exact packed affine-Q6 BF16 storage for the 0731 body ``wo_b`` matrix."""

    k: int = _K
    n: int = _N
    bits: int = _BITS
    group_size: int = _GROUP_SIZE
    dtype: mx.Dtype = mx.bfloat16

    def __post_init__(self) -> None:
        if (self.k, self.n, self.bits, self.group_size, self.dtype) != (
            _K,
            _N,
            _BITS,
            _GROUP_SIZE,
            mx.bfloat16,
        ):
            raise M3WOBContractError(
                "fixed M3 wo_b supports only BF16 affine Q6/G128 K=8192 N=4096"
            )

    @property
    def packed_cols(self) -> int:
        return self.k * self.bits // 32

    @property
    def metadata_cols(self) -> int:
        return self.k // self.group_size

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.n, self.packed_cols)

    @property
    def metadata_shape(self) -> tuple[int, int]:
        return (self.n, self.metadata_cols)


def _shape(value: Any) -> tuple[int, ...] | None:
    try:
        return tuple(int(dimension) for dimension in value.shape)
    except (AttributeError, TypeError, ValueError):
        return None


def validate_wob_projection(
    projection: Any, contract: M3WOBContract | None = None
) -> M3WOBContract:
    """Fail at construction unless every immutable stock-storage fact matches."""

    bound_contract = contract or M3WOBContract()
    weight = getattr(projection, "weight", None)
    scales = getattr(projection, "scales", None)
    biases = getattr(projection, "biases", None)
    if _shape(weight) != bound_contract.weight_shape:
        if _shape(weight) and len(_shape(weight) or ()) != 2:
            raise M3WOBContractError("fixed M3 wo_b requires an unbatched RHS")
        raise M3WOBContractError("fixed M3 wo_b packed RHS shape does not match K=8192")
    if (
        int(getattr(projection, "bits", 0) or 0) != _BITS
        or int(getattr(projection, "group_size", 0) or 0) != _GROUP_SIZE
        or str(getattr(projection, "mode", "")).lower() != "affine"
        or getattr(projection, "bias", None) is not None
        or _shape(scales) != bound_contract.metadata_shape
        or _shape(biases) != bound_contract.metadata_shape
        or getattr(weight, "dtype", None) != mx.uint32
        or getattr(scales, "dtype", None) != mx.bfloat16
        or getattr(biases, "dtype", None) != mx.bfloat16
    ):
        raise M3WOBContractError(
            "fixed M3 wo_b requires unbatched affine Q6/G128 U32 weights and "
            "BF16 scales/biases"
        )
    return bound_contract


def m3_wob_metal_source() -> str:
    """Return the fixed K=8192 Q6 body with three independent stock M1 trees."""

    return r"""
constexpr uint M = 3;
constexpr uint K = 8192;
constexpr uint N = 4096;
constexpr uint GS = 128;
constexpr uint PACKS_PER_THREAD = 2;
constexpr uint PACK_FACTOR = 4;
constexpr uint BYTES_PER_PACK = 3;
constexpr uint VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;
constexpr uint RESULTS_PER_SIMDGROUP = 4;

uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;
uint out_row = threadgroup_position_in_grid.x * 8 + simd_gid * RESULTS_PER_SIMDGROUP;
uint group_count = K / GS;
uint row_bytes = K * 6 / 8;
const device uchar* ws = (const device uchar*)w + out_row * row_bytes + simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* sc = scales + out_row * group_count + simd_lid / (GS / VALUES_PER_THREAD);
const device T* bs = biases + out_row * group_count + simd_lid / (GS / VALUES_PER_THREAD);
const device T* x0 = x + simd_lid * VALUES_PER_THREAD;
const device T* x1 = x0 + K;
const device T* x2 = x1 + K;
device T* y0 = y + out_row;
device T* y1 = y0 + N;
device T* y2 = y1 + N;
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
  for (uint row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    const device uchar* wl = ws + row * row_bytes;
    const device T* sl = sc + row * group_count;
    const device T* bl = bs + row * group_count;
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
    result0[row] += s * dot0 + sum0 * b;
    result1[row] += s * dot1 + sum1 * b;
    result2[row] += s * dot2 + sum2 * b;
  }
  ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
  sc += BLOCK_SIZE / GS; bs += BLOCK_SIZE / GS;
  x0 += BLOCK_SIZE; x1 += BLOCK_SIZE; x2 += BLOCK_SIZE;
}
for (uint row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  float r0 = simd_sum(result0[row]);
  float r1 = simd_sum(result1[row]);
  float r2 = simd_sum(result2[row]);
  if (simd_lid == 0) { y0[row] = T(r0); y1[row] = T(r1); y2[row] = T(r2); }
}
"""


@lru_cache(maxsize=1)
def _build_wob_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_0731_official_m3_wob_q6",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header="using namespace metal;",
        source=m3_wob_metal_source(),
        ensure_row_contiguous=False,
    )


class BoundM3WOB:
    """Prebound direct fixed-M3 wo_b callable; no execution-path checks exist."""

    __slots__ = (
        "_kernel",
        "biases",
        "grid",
        "input_shape",
        "output_shape",
        "scales",
        "threadgroup",
        "weight",
    )

    def __init__(self, projection: Any, contract: M3WOBContract):
        validate_wob_projection(projection, contract)
        self.weight = projection.weight
        self.scales = projection.scales
        self.biases = projection.biases
        self._kernel = _build_wob_kernel()
        self.input_shape = (1, _M, _K)
        self.output_shape = (1, _M, _N)
        self.grid = ((_N // 8) * 64, 1, 1)
        self.threadgroup = (64, 1, 1)

    def __call__(self, x: mx.array) -> mx.array:
        (out,) = self._kernel(
            inputs=[x.reshape(_M, _K), self.weight, self.scales, self.biases],
            template=[("T", mx.bfloat16)],
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=[(_M, _N)],
            output_dtypes=[mx.bfloat16],
        )
        return out.reshape(self.output_shape)


def bind_m3_wob(projection: Any) -> Callable[[mx.array], mx.array]:
    """Validate then bind the sole supported physical-M3 wo_b contract."""

    return BoundM3WOB(projection, M3WOBContract())


class _PreboundWOBRoute:
    """The sole hot choice: captured stock phase versus physical M3."""

    __slots__ = ("candidate", "stock")

    def __init__(
        self,
        stock: Callable[[mx.array], mx.array],
        candidate: Callable[[mx.array], mx.array],
    ) -> None:
        self.stock = stock
        self.candidate = candidate

    def __call__(self, x: mx.array) -> mx.array:
        if int(x.shape[1]) == _M:
            return self.candidate(x)
        return self.stock(x)


def prebind_wob_route(
    stock: Callable[[mx.array], mx.array], candidate: Callable[[mx.array], mx.array]
) -> Callable[[mx.array], mx.array]:
    """Capture the M3/stock phase route; execution checks only logical shape."""

    return _PreboundWOBRoute(stock, candidate)


@dataclass(frozen=True, slots=True)
class PreparedWOBM3Routes:
    """Self-checked 43-layer ``wo_b`` bank awaiting atomic publication."""

    attentions: tuple[Any, ...]
    o_lora_impls: tuple[Any, ...]
    stock_routes: tuple[Callable[[mx.array], mx.array], ...]
    candidate_routes: tuple[Callable[[mx.array], mx.array], ...]
    published_routes: tuple[Callable[[mx.array], mx.array], ...]
    layer_count: int
    q6_count: int
    exact_selfchecked: int
    o_lora_sink_count: int

    def publish(self) -> None:
        try:
            for attention, o_lora_impl, route in zip(
                self.attentions,
                self.o_lora_impls,
                self.published_routes,
            ):
                attention.wo_b = route
                o_lora_impl.wo_b = route
        except Exception as publication_error:
            try:
                self.restore()
            except Exception as restoration_error:
                publication_error.add_note(
                    f"WOB rollback also raised: {restoration_error}"
                )
            raise

    def restore(self) -> None:
        errors = []
        for attention, o_lora_impl, stock in zip(
            self.attentions,
            self.o_lora_impls,
            self.stock_routes,
        ):
            try:
                attention.wo_b = stock
            except Exception as exc:
                errors.append(exc)
            try:
                o_lora_impl.wo_b = stock
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("WOB route restoration failed", errors)


def prepare_wob_m3(
    layers: Any,
    *,
    exact_selfcheck: Callable[
        [Callable[[mx.array], mx.array], Callable[[mx.array], mx.array], int], bool
    ],
) -> PreparedWOBM3Routes:
    """Bind and self-check every ``wo_b`` route without publishing.

    The caller supplies an exact real-weight self-check.  Every layer validates
    before any custom candidate exists; every live gather-o-LoRA sink must
    still alias that stock projection. Every candidate is then self-checked
    before either reference changes.
    """

    layer_tuple = tuple(layers)
    if len(layer_tuple) != 43:
        raise M3WOBContractError("fixed M3 wo_b preparation requires exactly 43 layers")
    if not callable(exact_selfcheck):
        raise M3WOBContractError("exact WOB self-check is required")

    validated: list[tuple[Any, Any, Callable[[mx.array], mx.array]]] = []
    for index, layer in enumerate(layer_tuple):
        try:
            attention = layer.attn
            stock = attention.wo_b
        except AttributeError as exc:
            raise M3WOBContractError(f"layer {index} has no attention wo_b") from exc
        if not callable(stock):
            raise M3WOBContractError(f"layer {index} attention wo_b is not callable")
        try:
            o_lora_impl = attention._o_lora_impl
            o_lora_sink = o_lora_impl.wo_b
        except AttributeError as exc:
            raise M3WOBContractError(
                f"layer {index} has no active o-LoRA wo_b sink"
            ) from exc
        if o_lora_sink is not stock:
            raise M3WOBContractError(f"layer {index} active o-LoRA wo_b sink is stale")
        validate_wob_projection(stock, M3WOBContract())
        validated.append((attention, o_lora_impl, stock))

    staged: list[
        tuple[
            Any,
            Any,
            Callable[[mx.array], mx.array],
            Callable[[mx.array], mx.array],
            Callable[[mx.array], mx.array],
        ]
    ] = []
    rejected_selfcheck: int | None = None
    for index, (attention, o_lora_impl, stock) in enumerate(validated):
        candidate = bind_m3_wob(stock)
        if not callable(candidate):
            raise M3WOBContractError(
                f"layer {index} fixed M3 candidate is not callable"
            )
        try:
            passed = exact_selfcheck(stock, candidate, index)
        except Exception as exc:
            raise M3WOBContractError(
                f"layer {index} fixed M3 exact self-check raised"
            ) from exc
        if not passed and rejected_selfcheck is None:
            rejected_selfcheck = index
        staged.append(
            (
                attention,
                o_lora_impl,
                stock,
                candidate,
                prebind_wob_route(stock, candidate),
            )
        )

    if rejected_selfcheck is not None:
        raise M3WOBContractError(
            f"layer {rejected_selfcheck} fixed M3 exact self-check failed"
        )

    return PreparedWOBM3Routes(
        attentions=tuple(attention for attention, _, _, _, _ in staged),
        o_lora_impls=tuple(o_lora_impl for _, o_lora_impl, _, _, _ in staged),
        stock_routes=tuple(stock for _, _, stock, _, _ in staged),
        candidate_routes=tuple(candidate for _, _, _, candidate, _ in staged),
        published_routes=tuple(route for _, _, _, _, route in staged),
        layer_count=len(staged),
        q6_count=len(staged),
        exact_selfchecked=len(staged),
        o_lora_sink_count=len(staged),
    )
