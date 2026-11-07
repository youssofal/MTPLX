"""The hyper-connection read at verify widths, in three dispatches, bit for bit.

Inside a compiled verify body one GatedResidual read at 2 to 8 rows is about
eleven dependent kernels: the pending residual write (multiply, add), the
grouped RMS norm, its weight multiply, three projections (down, inject, up),
the fused silu, the fused inject gate, the fused gate-times-normed product,
the stream sum and the mean's scale. Flash-Next runs 97 of these reads per
verify forward (two per layer plus the final mixer). This module runs one read
as three kernels:

1. ``norm``: the pending write (``hyper + block_out * inject``, when the caller
   hands it over instead of writing it first) and the grouped RMS norm with
   its weight, one threadgroup per (row, stream).
2. ``down``: the down projection with the silu of ``/ hc_count`` on its rows,
   and the inject projection with ``2 * sigmoid(/ hc_count)`` on its rows.
3. ``up``: for each output column, the ``hc_count`` up rows that feed it, the
   sigmoid gate times the normed value, the stream sum and the ``1 / hc``
   scale.

EXACTNESS. Every output bit equals the compiled stock chain's, because each
kernel repeats MLX 0.32.2's own arithmetic in the same order:

* the norm reduces each stream the way MLX's single-row RMS norm does for a
  row of up to 4,096 values: virtual lanes of four consecutive values summed
  in order, a ``simd_sum`` per 32 virtual lanes, one ``simd_sum`` over the 32
  partial slots, IEEE division by the width, precise ``rsqrt``, a cast to the
  element type before the weight multiply;
* the projections ARE the library's wide gemv: the kernels call the
  ``GemvWide`` template of the installed MLX (read from its
  ``mlx/backend/metal/kernels/gemv.h`` when the kernels are built; no copy
  lives in this tree), the code MLX itself runs for 2 to 15 rows on GPU
  generation 15 and newer, with the lane count it picks (32 lanes per row when
  all rows fit one pass or the projection has at most 64 outputs, else 16).
  An install without a readable header, or with a template these kernels do
  not recognise, keeps the stock chain with a printed reason;
* the stock read evaluates its sigmoids two ways, and the kernels read each
  from a table of the stock values at every 16-bit input, made at install:
  the silu and the inject gate fuse into a JIT kernel (fast ``exp``), while
  the mix gate's sigmoid stays a standalone primitive (its reshape blocks
  fusion; precise ``exp``). The two differ at one bfloat16 input, -6.84375;
  the kernels never evaluate ``exp`` themselves;
* the stream sum starts each stream from +0 and adds the streams in order,
  as MLX's small column reduce does (visible only as the sign of a zero).

The read therefore engages only where that is its parent: inside a compiled
step body (the fused elementwise lowering), at 2 to 8 rows, in bfloat16 or
float16, with unquantized mixers, on a single-die GPU of generation 15 or
newer or an M5 Ultra (where MLX serves these projections with its wide gemv;
M1 and M2 run a tiled GEMM and keep the stock chain). An M3 Ultra keeps the
stock chain until it is measured. ``install`` proves every width it enables on this GPU against the
compiled stock chain with the model's own weights before the first verify
trace; a difference, a build failure or a refused thread count turns the read
off for the process with a printed reason, and the stock chain serves.

Rollback: ``MTPLX_QWEN4_HC_VERIFY_READ=0`` keeps the stock chain everywhere.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Iterable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ENV = "MTPLX_QWEN4_HC_VERIFY_READ"

#: Rows the read serves: verify windows. One row keeps its own lanes (the
#: 8-bit v3 read, or the library's one-vector gemv), and above 8 rows is a
#: copy-lane window no compiled body runs.
MIN_ROWS = 2
MAX_ROWS = 8

#: Threads per threadgroup for all three kernels (issue #400: at most 256).
_THREADS = 256
_SIMDGROUPS = _THREADS // 32

#: Stream counts the kernels accept: powers of two, so the mean's 1 / hc is
#: exact in every element type, and at most 8, so MLX's small column reduce
#: gives each stream its own lane.
_HC_COUNTS = (2, 4, 8)

#: Stream widths the single-row RMS norm reduces with one threadgroup.
_MAX_STREAM_WIDTH = 4096

_SUPPORTED_DTYPES = (mx.bfloat16, mx.float16)

_STATE: dict[str, Any] = {
    "installed": False,
    "disabled_reason": None,
    "rows": (),
    "dtype": None,
    "geometry": None,
    "sigmoid": None,
}
_COUNTS: dict[str, int] = {
    "installs": 0,
    "probe_cases": 0,
    "probe_failures": 0,
    "traces": 0,
}
_ENGAGED_ROWS: set[int] = set()


# ---------------------------------------------------------------------------
# Kernel sources
# ---------------------------------------------------------------------------

_COMMON = r"""
// A sigmoid exactly as the stock read computes it at this site: the value at
// every 16-bit input, read from a table the stock lowering itself produced.
template <typename U>
inline U hc_sigmoid(U x, const device U* table) {
    return table[as_type<ushort>(x)];
}
"""

#: The projections run the library's own wide-gemv template (``GemvWide`` in
#: the installed ``mlx/backend/metal/kernels/gemv.h``), read from the installed
#: package when the kernels are built: it is the code the stock chain runs for
#: these projections, so the sums are the stock sums by construction. Its
#: bias/axpby branch (a function constant in the header) is off for these
#: projections and is written as ``false``.
_GEMV_HEADER = "mlx/backend/metal/kernels/gemv.h"
_GEMV_THREADS = 128  # four simdgroups: GemvWide's layout at 32 lanes per row


def _balanced_block(text: str, anchor: str) -> str:
    """The declaration that contains ``anchor``: from the ``template <`` that
    opens it to the ``};`` that closes its braces."""

    at = text.index(anchor)
    start = text.rindex("template <", 0, at)
    depth = 0
    i = text.index("{", at)
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    close = text.index(";", i)
    return text[start : close + 1]


@lru_cache(maxsize=1)
def _gemv_template() -> tuple[Optional[str], Optional[str]]:
    """``(source, None)`` with the installed library's ``DefaultAccT`` and
    ``GemvWide`` templates, or ``(None, reason)`` when the installed package
    has no readable header or the template is not the one these kernels call."""

    from pathlib import Path

    try:
        root = Path(mx.__file__).resolve().parent / "include"
        text = (root / _GEMV_HEADER).read_text()
        acc = _balanced_block(text, "struct DefaultAccT {")
        acc_complex = _balanced_block(text, "struct DefaultAccT<complex64_t> {")
        wide = _balanced_block(text, "struct GemvWide {")
    except (OSError, ValueError) as exc:
        return None, f"the installed MLX gemv header is not readable here ({type(exc).__name__}: {exc})"
    for needle in ("static METAL_FUNC void run(", "uint simd_gid", "uint simd_lid", "gemv_wide_do_axpby"):
        if needle not in wide:
            return None, f"the installed GemvWide template has no {needle!r}"
    wide = wide.replace("gemv_wide_do_axpby", "false")
    return "\n".join((acc, acc_complex, wide)) + "\n", None


# Kernel 1: the pending write and the grouped RMS norm. One threadgroup per
# (row, stream); its 256 threads play the norm's VL virtual lanes, VS virtual
# simdgroups of 32 (virtual lane v = thread + 256 * round, so virtual
# simdgroup v / 32 = simdgroup + 8 * round and its lane is the thread's lane).
_NORM_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint slot = threadgroup_position_in_grid.x;
    const uint row = slot / HC;
    const uint stream = slot % HC;
    const size_t base = (size_t)row * HCD + (size_t)stream * D;
    threadgroup float partial[32];
    threadgroup float scale[1];

    T values[ROUNDS][4];
    float sums[ROUNDS];
    for (uint k = 0; k < ROUNDS; ++k) {
        const uint v = tid + 256u * k;
        float total = 0.0f;
        if (v < VL) {
            for (uint i = 0; i < 4u; ++i) {
                const uint e = v * 4u + i;
                T value = x[base + e];
#if HC_NORM_PENDING
                const T product =
                    block_out[(size_t)row * D + e] * inject[(size_t)row * HC + stream];
                value = value + product;
                written[base + e] = value;
#endif
                values[k][i] = value;
                const float f = float(value);
                total += f * f;
            }
        }
        sums[k] = total;
    }
    if (sg == 0) {
        partial[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < ROUNDS; ++k) {
        const float s = simd_sum(sums[k]);
        const uint vs = sg + 8u * k;
        if (lane == 0 && vs < VS) {
            partial[vs] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float total = simd_sum(partial[lane]);
        if (lane == 0) {
            scale[0] = metal::precise::rsqrt(
                metal::precise::divide(total, float(D)) + NORM_EPS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = scale[0];
    for (uint k = 0; k < ROUNDS; ++k) {
        const uint v = tid + 256u * k;
        if (v < VL) {
            for (uint i = 0; i < 4u; ++i) {
                const uint e = v * 4u + i;
                const T unit = static_cast<T>(float(values[k][i]) * inv);
                normed[base + e] = unit * gamma[(size_t)stream * D + e];
            }
        }
    }
"""

# Kernel 2: the down rows, then (in the last threadgroups) the inject rows.
# GemvWide writes each row's projection for every vector as the element type;
# the lane that stored it then reads it back and applies the row's gate.
_DOWN_SOURCE = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint block = threadgroup_position_in_grid.y;
    const device T* no_bias = nullptr;
    if (block < DOWN_BLOCKS) {
        if (sg < (uint)(KL_DOWN / 8)) {
            GemvWide<T, NV, KL_DOWN>::run(
                w_down, normed, no_bias, down_raw, HCD, LOWRANK, ROWS, HCD, HCD,
                1.0f, 0.0f, 0, 0, uint3(0, block, 0), uint3(1, DOWN_BLOCKS, 1), sg, lane);
            const int out_row = (int)block * 4 + (32 / KL_DOWN) * (int)sg + (int)lane / KL_DOWN;
            if ((int)lane % KL_DOWN == 0 && out_row < LOWRANK) {
                for (int r = 0; r < ROWS; ++r) {
                    const T projected = down_raw[(size_t)r * LOWRANK + out_row];
                    const T scaled = projected / T(HC);
                    mix[(size_t)r * LOWRANK + out_row] = scaled * hc_sigmoid(scaled, sigmoid);
                }
            }
        }
    } else if (INJECT) {
        // The inject projection has hc_count outputs: 32 lanes per row.
        const uint iblock = block - DOWN_BLOCKS;
        GemvWide<T, NV, 32>::run(
            w_inject, normed, no_bias, inject_raw, HCD, HC, ROWS, HCD, HCD,
            1.0f, 0.0f, 0, 0, uint3(0, iblock, 0), uint3(1, INJECT_BLOCKS, 1), sg, lane);
        const int out_row = (int)iblock * 4 + (int)sg;
        if (lane == 0 && out_row < HC) {
            for (int r = 0; r < ROWS; ++r) {
                const T logit = inject_raw[(size_t)r * HC + out_row];
                const T scaled = logit / T(HC);
                inject[(size_t)r * HC + out_row] = T(2) * hc_sigmoid(scaled, sigmoid);
            }
        }
    }
"""

# Kernel 3: one threadgroup per output column c. Its hc_count up rows (c,
# D + c, ...) are a matrix of hc_count rows with row stride D * LOWRANK, which
# GemvWide projects for every vector; then one thread per vector applies the
# gate, the product with the normed value, the stream sum and the 1 / hc scale.
_UP_SOURCE = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint c = threadgroup_position_in_grid.y;
    const device T* no_bias = nullptr;
    device T* raw = up_raw + (size_t)c * (ROWS * HC);
    if (sg < (uint)(KL_UP / 8)) {
        for (int b = 0; b < UP_BLOCKS; ++b) {
            GemvWide<T, NV, KL_UP>::run(
                w_up + (size_t)c * LOWRANK, mix, no_bias, raw, LOWRANK, HC, ROWS, D * LOWRANK, LOWRANK,
                1.0f, 0.0f, 0, 0, uint3(0, b, 0), uint3(1, UP_BLOCKS, 1), sg, lane);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    const uint t = thread_position_in_threadgroup.x;
    if (t < (uint)ROWS) {
        T total = T(0);
        for (int s = 0; s < HC; ++s) {
            const T gate = hc_sigmoid(raw[t * HC + s], sigmoid + 65536);
            const T product = gate * normed[(size_t)t * HCD + (size_t)s * D + c];
            // The small column reduce starts every stream's lane from +0 and
            // then adds the lanes in ascending stream order.
            const T lane_total = product + T(0);
            total = (s == 0) ? lane_total : T(lane_total + total);
        }
        mixed[(size_t)t * D + c] = total * T(INV_HC);
    }
"""


def _passes(rows: int) -> tuple[int, int]:
    """``(passes, vectors per pass)`` of the library's wide gemv for ``rows``."""

    passes = (rows + 4) // 5
    return passes, (rows + passes - 1) // passes


def _k_lanes(rows: int, outputs: int) -> int:
    """Lanes per projection row in the library's wide gemv for this call.

    It takes ceil(rows / 5) passes over the matrix; one pass, or at most 64
    outputs, keeps all 32 lanes on a row, otherwise 16 lanes share it.
    """

    passes = (rows + 4) // 5
    return 32 if passes == 1 or outputs <= 64 else 16


def _eps_literal(eps: float) -> str:
    # Ten significant digits round-trip every float32, so the kernel's
    # constant is the very float32 the library receives.
    return format(float(np.float32(eps)), ".9e") + "f"


def _header(hc: int, width: int, lowrank: int, eps: float) -> str:
    return (
        f"constant constexpr int HC = {hc};\n"
        f"constant constexpr int D = {width};\n"
        f"constant constexpr int HCD = {hc * width};\n"
        f"constant constexpr int LOWRANK = {lowrank};\n"
        f"constant constexpr float NORM_EPS = {_eps_literal(eps)};\n"
        f"constant constexpr float INV_HC = {1.0 / hc!r}f;\n"
        f"constant constexpr uint SIMDGROUPS = {_SIMDGROUPS}u;\n"
    )


@lru_cache(maxsize=None)
def _norm_kernel(hc: int, width: int, lowrank: int, eps: float, pending: bool):
    lanes = width // 4
    header = _header(hc, width, lowrank, eps) + (
        f"constant constexpr uint VL = {lanes}u;\n"
        f"constant constexpr uint VS = {(lanes + 31) // 32}u;\n"
        f"constant constexpr uint ROUNDS = {(lanes + _THREADS - 1) // _THREADS}u;\n"
        f"#define HC_NORM_PENDING {1 if pending else 0}\n"
    )
    outputs = ["written", "normed"] if pending else ["normed"]
    return mx.fast.metal_kernel(
        name=f"mtplx_hc_verify_norm_h{hc}_d{width}_p{int(pending)}",
        input_names=["x", "block_out", "inject", "gamma"],
        output_names=outputs,
        header=header,
        source=_NORM_SOURCE,
    )


def _gemv_source() -> str:
    source, reason = _gemv_template()
    if source is None:
        raise RuntimeError(reason)
    return source


@lru_cache(maxsize=None)
def _down_kernel(hc: int, width: int, lowrank: int, eps: float, rows: int, inject: bool):
    header = _header(hc, width, lowrank, eps) + _COMMON + _gemv_source() + (
        f"constant constexpr int ROWS = {rows};\n"
        f"constant constexpr int NV = {_passes(rows)[1]};\n"
        f"constant constexpr int KL_DOWN = {_k_lanes(rows, lowrank)};\n"
        f"constant constexpr uint DOWN_BLOCKS = {(lowrank + 3) // 4}u;\n"
        f"constant constexpr uint INJECT_BLOCKS = {(hc + 3) // 4}u;\n"
        f"constant constexpr bool INJECT = {'true' if inject else 'false'};\n"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_hc_verify_down_h{hc}_d{width}_l{lowrank}_r{rows}_i{int(inject)}",
        input_names=["normed", "w_down", "w_inject", "sigmoid"],
        output_names=["mix", "inject", "down_raw", "inject_raw"],
        header=header,
        source=_DOWN_SOURCE,
    )


@lru_cache(maxsize=None)
def _up_kernel(hc: int, width: int, lowrank: int, eps: float, rows: int):
    header = _header(hc, width, lowrank, eps) + _COMMON + _gemv_source() + (
        f"constant constexpr int ROWS = {rows};\n"
        f"constant constexpr int NV = {_passes(rows)[1]};\n"
        f"constant constexpr int KL_UP = {_k_lanes(rows, hc * width)};\n"
        f"constant constexpr int UP_BLOCKS = {(hc + 3) // 4};\n"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_hc_verify_up_h{hc}_d{width}_l{lowrank}_r{rows}",
        input_names=["mix", "w_up", "normed", "sigmoid"],
        output_names=["mixed", "up_raw"],
        header=header,
        source=_UP_SOURCE,
    )


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------


def geometry_supported(hc: int, width: int, lowrank: int) -> bool:
    """Shapes whose stock chain the kernels reproduce."""

    return (
        int(hc) in _HC_COUNTS
        and 4 <= int(width) <= _MAX_STREAM_WIDTH
        and int(width) % 4 == 0
        and int(lowrank) >= 4
        and int(lowrank) % 4 == 0
    )


def _sigmoid_times(values: mx.array, scale: mx.array) -> mx.array:
    return mx.sigmoid(values) * scale


@lru_cache(maxsize=4)
def sigmoid_table(dtype) -> mx.array:
    """The stock read's two sigmoids at every 16-bit input of ``dtype``.

    Row 0 is the fused lowering the silu and the inject gate take (made by
    ``mx.compile`` itself, a sigmoid fused with a multiply by one); row 1 is
    the standalone primitive the mix gate takes. Built and evaluated outside
    any trace, flattened to ``[2 * 65536]``.
    """

    values = mx.view(mx.arange(1 << 16, dtype=mx.uint32).astype(mx.uint16), dtype)
    fused = mx.compile(_sigmoid_times)(values, mx.ones((1 << 16,), dtype=dtype))
    standalone = mx.sigmoid(values)
    table = mx.concatenate([fused, standalone])
    mx.eval(table)
    return table


def read_rows(
    x: mx.array,
    gamma: mx.array,
    w_down: mx.array,
    w_up: mx.array,
    w_inject: Optional[mx.array],
    block_out: Optional[mx.array],
    inject: Optional[mx.array],
    *,
    hc: int,
    eps: float,
    sigmoid: mx.array,
):
    """One read of ``x`` ``[rows, hc * width]`` in three dispatches.

    ``block_out`` ``[rows, width]`` and ``inject`` ``[rows, hc]`` are the
    previous block's pending write, or both None; when given, the stream read
    is ``x + block_out * inject`` and it comes back as ``written``. Returns
    ``(mixed [rows, width], written [rows, hc * width], inject [rows, hc] or
    None)``. ``written`` is ``x`` itself when nothing was pending.
    ``sigmoid`` is :func:`sigmoid_table` for ``x``'s dtype.
    """

    rows, hcd = int(x.shape[0]), int(x.shape[1])
    hc = int(hc)
    width = hcd // hc
    lowrank = int(w_down.shape[0])
    dtype = x.dtype
    pending = block_out is not None
    has_inject = w_inject is not None

    norm = _norm_kernel(hc, width, lowrank, float(eps), pending)
    norm_inputs = [x, block_out if pending else x, inject if pending else x, gamma]
    if pending:
        written, normed = norm(
            inputs=norm_inputs,
            template=[("T", dtype)],
            grid=(rows * hc * _THREADS, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(rows, hcd), (rows, hcd)],
            output_dtypes=[dtype, dtype],
        )
    else:
        (normed,) = norm(
            inputs=norm_inputs,
            template=[("T", dtype)],
            grid=(rows * hc * _THREADS, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(rows, hcd)],
            output_dtypes=[dtype],
        )
        written = x

    blocks = (lowrank + 3) // 4 + ((hc + 3) // 4 if has_inject else 0)
    mix, inject_out, _down_raw, _inject_raw = _down_kernel(hc, width, lowrank, float(eps), rows, has_inject)(
        inputs=[normed, w_down, w_inject if has_inject else w_down, sigmoid],
        template=[("T", dtype)],
        grid=(_GEMV_THREADS, blocks, 1),
        threadgroup=(_GEMV_THREADS, 1, 1),
        output_shapes=[(rows, lowrank), (rows, hc), (rows, lowrank), (rows, hc)],
        output_dtypes=[dtype, dtype, dtype, dtype],
    )

    mixed, _up_raw = _up_kernel(hc, width, lowrank, float(eps), rows)(
        inputs=[mix, w_up, normed, sigmoid],
        template=[("T", dtype)],
        grid=(_GEMV_THREADS, width, 1),
        threadgroup=(_GEMV_THREADS, 1, 1),
        output_shapes=[(rows, width), (width, rows, hc)],
        output_dtypes=[dtype, dtype],
    )
    return mixed, written, (inject_out if has_inject else None)


# ---------------------------------------------------------------------------
# Where the read is the parent's arithmetic
# ---------------------------------------------------------------------------


def switched_on() -> bool:
    """The operator's switch (default on)."""

    raw = (os.environ.get(ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def device_route(architecture: Optional[str]) -> tuple[bool, str]:
    """Does MLX serve these projections with its wide gemv on this GPU?

    MLX reads the generation from the last three characters of its
    architecture string (``MLX_METAL_GPU_ARCH`` overrides it, which is what
    the M1-M4 rehearsal switch uses), and takes the wide gemv from generation
    15. Of the Ultra-class GPUs (``d``), generation 17 (the M5 Ultra) has
    measured this read; an earlier Ultra keeps the stock chain until it is
    measured.
    """

    if not isinstance(architecture, str):
        return False, "unknown GPU architecture"
    match = re.search(r"(\d{2})([a-z])$", architecture.strip().lower())
    if match is None:
        return False, f"unparsed GPU architecture {architecture!r}"
    generation, klass = int(match.group(1)), match.group(2)
    if generation < 15:
        return False, (
            f"{architecture}: MLX serves 2-15 row projections with a tiled "
            "GEMM before GPU generation 15, so the stock chain is the parent"
        )
    if klass == "d" and generation < 17:
        return False, (
            f"{architecture}: of the Ultra-class GPUs only generation 17 "
            "(the M5 Ultra) has measured this read"
        )
    if klass not in {"g", "s", "d"}:
        return False, f"{architecture}: no GPU of this class has measured this read"
    return True, ""


def _module_geometry(module: Any) -> Optional[tuple[int, int, int]]:
    """``(hc, width, lowrank)`` when ``module`` is exactly the read the kernels
    reproduce: bias-free unquantized projections and a norm grouped by the
    stream width. Anything else (a quantized, biased or replaced projection)
    keeps the stock chain."""

    linears = [
        getattr(module, "input_mix_weight_down", None),
        getattr(module, "input_mix_weight_up", None),
    ]
    if "block_inject_weight" in module:
        linears.append(module.block_inject_weight)
    if any(type(linear) is not nn.Linear or "bias" in linear for linear in linears):
        return None
    norm = getattr(module, "hc_norm", None)
    width = int(module.hidden_size)
    if getattr(norm, "group_size", None) != width or "weight" not in norm:
        return None
    return (int(module.hc_count), width, int(linears[0].weight.shape[0]))


def _module_dtypes_match(module: Any, dtype) -> bool:
    weights = [
        module.hc_norm.weight,
        module.input_mix_weight_down.weight,
        module.input_mix_weight_up.weight,
    ]
    if "block_inject_weight" in module:
        weights.append(module.block_inject_weight.weight)
    return all(weight.dtype == dtype for weight in weights)


def serves(module: Any, dtype, rows: int) -> bool:
    """True when ``install`` proved this read for ``rows`` rows of ``module``.

    Called while a compiled verify body is traced, once per read.
    """

    if not _STATE["installed"] or rows not in _STATE["rows"]:
        return False
    if dtype != _STATE["dtype"]:
        return False
    if _module_geometry(module) != _STATE["geometry"]:
        return False
    return _module_dtypes_match(module, dtype)


def installed_sigmoid() -> mx.array:
    """The table ``install`` built for the installed dtype."""

    table = _STATE["sigmoid"]
    if table is None:
        raise RuntimeError("the hyper-connection verify read is not installed")
    return table


def note_engaged(rows: int) -> None:
    """Counts traced reads; the first per width leaves one log line."""

    _COUNTS["traces"] += 1
    if rows not in _ENGAGED_ROWS:
        _ENGAGED_ROWS.add(rows)
        print(
            "[mtplx] hyper-connection verify read engaged in a compiled verify "
            f"trace at {rows} rows (3 dispatches per read)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Install: the first-use canary
# ---------------------------------------------------------------------------


def _bits(a: mx.array) -> mx.array:
    return mx.view(a, mx.uint16)


def _silu_stock(values: mx.array) -> mx.array:
    # The read's silu: nn.silu of the down projection over hc, fused.
    return values * mx.sigmoid(values)


def _inject_stock(values: mx.array) -> mx.array:
    # The read's inject gate after its divide, fused.
    return 2.0 * mx.sigmoid(values)


def _gate_stock(values: mx.array, other: mx.array) -> mx.array:
    # The read's mix gate: the sigmoid, the stream reshape, the product.
    gate = mx.sigmoid(values)
    gate = gate.reshape(*gate.shape[:-1], 4, -1)
    return gate * other.reshape(*other.shape[:-1], 4, -1)


def _table_check(dtype) -> Optional[str]:
    """Each table row against the lowering its consumers take, at every input."""

    n = 1 << 16
    values = mx.view(mx.arange(n, dtype=mx.uint32).astype(mx.uint16), dtype)
    table = sigmoid_table(dtype)
    fused, standalone = table[:n], table[n:]
    ones = mx.ones((n,), dtype=dtype)
    checks = {
        "silu": (mx.compile(_silu_stock)(values), values * fused),
        "inject gate": (mx.compile(_inject_stock)(values), 2.0 * fused),
        "mix gate": (
            mx.compile(_gate_stock)(values.reshape(1, 4, -1), ones.reshape(1, 4, -1)),
            (standalone * ones).reshape(1, 4, 4, -1),
        ),
    }
    finite = mx.abs(values.astype(mx.float32)) < float("inf")
    verdicts = {
        name: mx.all(mx.logical_or(mx.logical_not(finite.reshape(want.shape)), mx.equal(mx.view(want, mx.uint16), mx.view(got, mx.uint16))))
        for name, (want, got) in checks.items()
    }
    mx.eval(*verdicts.values())
    for name, verdict in verdicts.items():
        if not bool(verdict.item()):
            return f"the {name} lowering no longer matches its sigmoid table"
    return None


def _probe_case(module: Any, rows: int, pending: bool, seed: int) -> Optional[str]:
    """Fused read against the compiled stock chain on ``module``'s weights."""

    hc, width, lowrank = _module_geometry(module)
    dtype = module.hc_norm.weight.dtype
    hcd = hc * width
    keys = mx.random.split(mx.random.key(seed), 3)
    x = (mx.random.normal((1, rows, hc, width - 8), key=keys[0]) * 3.0).astype(dtype)
    # The first 8 columns of every stream are -0: each of those columns then
    # reaches the stream sum as -0 products only, which the stock reduce turns
    # into a +0 mean (it starts every sum from +0), and so must the kernels.
    x = mx.concatenate([mx.zeros((1, rows, hc, 8), dtype=dtype) * -1.0, x], axis=-1)
    x = x.reshape(1, rows, hcd)
    block = (mx.random.normal((1, rows, width), key=keys[1]) * 2.0).astype(dtype)
    gates = mx.random.uniform(0.0, 2.0, (1, rows, hc), key=keys[2]).astype(dtype)
    combine = "block_inject_weight" in module

    if pending:
        reference = mx.compile(lambda s, b, g: module(s, pending=(b, g)))(x, block, gates)
    else:
        reference = mx.compile(lambda s: module(s))(x)
    if not combine:
        reference = (reference, None, None)
    ref_mixed, ref_written, ref_inject = reference

    mixed, written, inject = read_rows(
        x.reshape(rows, hcd),
        module.hc_norm.weight,
        module.input_mix_weight_down.weight,
        module.input_mix_weight_up.weight,
        module.block_inject_weight.weight if combine else None,
        block.reshape(rows, width) if pending else None,
        gates.reshape(rows, hc) if pending else None,
        hc=hc,
        eps=float(module.hc_norm.eps),
        sigmoid=sigmoid_table(dtype),
    )
    pairs = [("mixed", ref_mixed.reshape(rows, width), mixed)]
    if combine:
        pairs.append(("inject", ref_inject.reshape(rows, hc), inject))
        if pending:
            pairs.append(("written stream", ref_written.reshape(rows, hcd), written))
    checks = [mx.array_equal(_bits(want), _bits(got)) for _name, want, got in pairs]
    mx.eval(*checks)
    for (name, _want, _got), same in zip(pairs, checks):
        if not bool(same.item()):
            return (
                f"{name} differs from the compiled stock chain at {rows} rows "
                f"({'with' if pending else 'without'} a pending write)"
            )
    return None


def install(model: Any, *, rows: Iterable[int] = (4,), architecture: Optional[str] = None) -> dict:
    """Prove the read on this GPU for each width in ``rows``, then enable it.

    ``model`` is the text model (its ``layers`` and ``hyper_connection_mixer``
    supply the weights the probe reads). Runs at model build, outside any
    ``mx.compile`` trace. Any difference, build failure or dispatch refusal
    leaves the read off for the process and the stock chain serving.
    """

    reset_state()
    _COUNTS["installs"] += 1
    widths = tuple(sorted({int(r) for r in rows}))
    report: dict[str, Any] = {"rows": widths}

    def _off(reason: str, *, failure: bool = False) -> dict:
        _STATE["disabled_reason"] = reason
        if failure:
            _COUNTS["probe_failures"] += 1
        print(
            "[mtplx] hyper-connection verify read off: "
            f"{reason}; the stock chain serves",
            flush=True,
        )
        report.update(engagement())
        return report

    if not switched_on():
        return _off(f"{ENV}=0")
    if not widths or widths[0] < MIN_ROWS or widths[-1] > MAX_ROWS:
        return _off(f"widths {widths} outside {MIN_ROWS}..{MAX_ROWS}")
    if not mx.metal.is_available():
        return _off("no Metal GPU")
    if architecture is None:
        from mtplx.nax_detect import gpu_architecture

        architecture = gpu_architecture()
    ok, why = device_route(architecture)
    if not ok:
        return _off(why)

    layers = list(getattr(model, "layers", ()) or ())
    mixer = getattr(model, "hyper_connection_mixer", None)
    if not layers or mixer is None:
        return _off("the model has no hyper-connection layers")
    reader = layers[0].attn_hyper_connection
    geometry = _module_geometry(reader)
    if geometry is None or _module_geometry(mixer) != geometry:
        return _off("quantized or mismatched hyper-connection mixers")
    if not geometry_supported(*geometry):
        return _off(f"geometry {geometry} outside the kernels' contract")
    dtype = reader.hc_norm.weight.dtype
    if dtype not in _SUPPORTED_DTYPES:
        return _off(f"{dtype} mixers (the kernels serve bfloat16 and float16)")
    if not (_module_dtypes_match(reader, dtype) and _module_dtypes_match(mixer, dtype)):
        return _off("mixed-dtype hyper-connection weights")
    _source, why = _gemv_template()
    if _source is None:
        return _off(why)

    cases = 1
    _COUNTS["probe_cases"] += 1
    try:
        reason = _table_check(dtype)
    except Exception as exc:  # a build failure or a refused dispatch
        reason = f"{type(exc).__name__}: {exc}"
    if reason is not None:
        return _off(reason, failure=True)

    seed = 20260929
    for width in widths:
        for module, pending in ((reader, False), (reader, True), (mixer, False)):
            cases += 1
            _COUNTS["probe_cases"] += 1
            seed += 1
            try:
                reason = _probe_case(module, width, pending, seed)
            except Exception as exc:  # a build failure or a refused dispatch
                reason = f"{type(exc).__name__}: {exc}"
            if reason is not None:
                return _off(reason, failure=True)

    _STATE.update(
        installed=True,
        disabled_reason=None,
        rows=widths,
        dtype=dtype,
        geometry=geometry,
        sigmoid=sigmoid_table(dtype),
    )
    print(
        f"[mtplx] hyper-connection verify read on: rows={widths}, dtype={dtype}, "
        f"{cases} probe cases bit-equal to the compiled stock chain",
        flush=True,
    )
    report.update(engagement())
    return report


def engagement() -> dict:
    """The read's verdict and counters, for receipts."""

    return {
        "installed": bool(_STATE["installed"]),
        "disabled_reason": _STATE["disabled_reason"],
        "rows": tuple(_STATE["rows"]),
        "dtype": str(_STATE["dtype"]) if _STATE["dtype"] is not None else None,
        "geometry": _STATE["geometry"],
        "engaged_rows": tuple(sorted(_ENGAGED_ROWS)),
        **dict(_COUNTS),
    }


def reset_state() -> None:
    """Forget the verdict (a new model is being installed)."""

    _STATE.update(
        installed=False,
        disabled_reason=None,
        rows=(),
        dtype=None,
        geometry=None,
        sigmoid=None,
    )
    _ENGAGED_ROWS.clear()


def reset_for_tests() -> None:
    reset_state()
    for key in _COUNTS:
        _COUNTS[key] = 0


__all__ = [
    "ENV",
    "MAX_ROWS",
    "MIN_ROWS",
    "device_route",
    "engagement",
    "geometry_supported",
    "install",
    "installed_sigmoid",
    "note_engaged",
    "read_rows",
    "reset_for_tests",
    "reset_state",
    "serves",
    "sigmoid_table",
    "switched_on",
]
