"""Fused native routed-expert block for Inkling 2 Mini at decode and verify widths.

`model.Experts` records 41 graph primitives per FP4 visit and 28 per FP8 visit at 1 to
32 rows, and only three of them do real work: the router matmul and the two quantized
gathers. This path keeps those calls and MLX's router sigmoid exactly as they are, and
replaces the rest with four kernels, 8 primitives a visit:

    inputs   row -> FP32 router input, native expert input, FP4 outer row scale
    route    + bias, stable top 8, normalise, route scale
    swiglu   FP4 outer scale, round, capped SwiGLU, round, down-input quantisation
    combine  FP4 outer scale, round, probability weight, ordered 8-way sum, round

Exactness: class A. Every value is rounded where `model.Experts` rounds it, with the
same operations in the same order (MLX 0.32.2):
  * runtime-compiled kernels use the default fast single-precision library functions
    (math mode Safe since macOS 15) while MLX's prebuilt ops are built precise, so a
    fused kernel uses a library function only where the stock value also comes from a
    runtime-compiled kernel (the compiled SwiGLU, the input quantiser). The router
    sigmoid is computed in the route kernel with metal::precise::exp where that route
    passes its first-use proof on this device, and stays `mx.sigmoid` where it does not
    (or with MTPLX_INKLING_EXPERT_ROUTE=scores);
  * selection is MLX's stable merge sort of -(sigmoid + bias): highest first, equal
    keys to the lower expert, NaN last;
  * the probability sum runs 0 + p0 + ... + p7 as MLX's small row reduce does;
  * the weighted sum runs u0, u1 + acc, ..., u7 + acc with u = w + 0 as MLX's small
    column reduce does;
  * the input quantiser repeats `activations.expert_inputs` per lane, so each 16 or 32
    block sits on the same simdgroup lanes and reduces in the same order;
  * SwiGLU constants are written as MLX's compiler writes them (7 digits).

Admission, per model: `bind(model, ProofCache.for_pack(path))` at load. Each served
specialisation (format, dtype, route, decode or verify width, model geometry) is proven
once through `ProofCache.decide`: the stock body and the fused path run on that call's
own rows and must agree bit for bit. A proof is kept with the pack, build and GPU
identity it held on; a mismatch keeps the specialisation on the stock body. Everything
the path does not serve (more than 32 rows, affine research experts, CPU, other dtypes)
runs the stock body with its reason. Every visit is counted for its request and for the
model. MTPLX_INKLING_EXPERT_FUSION=off restores the stock body.

Prompt chunks (33 to 4,096 rows): the stock body sorts the pairs by expert and runs the two
quantized gathers with `sorted_indices`; `routed_sorted` keeps the sort and the down gather
exactly as the stock body makes them, takes the gate-up gather in the form the stock body's
measured route proves equal (NVFP4: rows gathered in sorted order, one sorted product; MXFP8:
each pair MLX's one-row product, computed by the panel product that reads each expert once
for all its pairs instead of once a pair), and replaces the glue with the same input, route,
SwiGLU and combine kernels, the combine reading each pair through the inverse of the sort
(`combine_sorted`). Proven per specialisation (width "prompt") like the decode and verify
widths; MTPLX_INKLING_PROMPT_EXPERTS=stock keeps the stock body.
"""

from collections import Counter
from contextvars import ContextVar
from functools import lru_cache
import logging
import os
import re
from threading import RLock

import mlx.core as mx

from mtplx.kernels.first_use import Mismatch, ProofCache

NAME = "inkling_expert_fusion"
THREADS = 384  # one threadgroup per row; 2304 and 1152 are both multiples
MAX_ROWS = 32  # above this model.Experts sorts routes into tiled panels
ROUTES = ("scores", "fused", "ops")

_log = logging.getLogger(__name__)
_request = ContextVar("inkling_expert_request", default=None)

HEADER = """
inline float fused_e4m3(float x) {
    float a = min(abs(x), 448.0f);
    float unit = ldexp(1.0f, max(-9, ilogb(max(a, 0.015625f)) - 3));
    return copysign(rint(a / unit) * unit, x);
}
inline float fused_e2m1(float x) {
    float a = abs(x);
    float y = a > 5 ? 6 : a >= 3.5f ? 4 : a > 2.5f ? 3 : a >= 1.75f ? 2 :
              a > 1.25f ? 1.5f : a >= .75f ? 1 : a > .25f ? .5f : 0;
    return copysign(y, x);
}
// One element of activations.expert_inputs; the element's lane is its index mod 32.
template <bool NVFP4>
inline float fused_quantize(float x, float peak, uint lane) {
    if (NVFP4) {
        float outer = peak / 1536.0f;
        uint first = lane & 16u;
        float top = abs(x);
        for (uint d = 1; d < 16; d *= 2)
            top = max(top, simd_shuffle_xor(top, d));
        float encoder = peak > 0 ? 1536.0f / peak : 0;
        float s6 = fused_e4m3(top * (encoder / 6.0f));
        float s4 = fused_e4m3(top * (encoder / 4.0f));
        float a = s6 == 0 ? 0 : fused_e2m1(x / (s6 * outer)) * s6;
        float b = s4 == 0 ? 0 : fused_e2m1(x / (s4 * outer)) * s4;
        float ae = abs(a * outer - x), be = abs(b * outer - x);
        for (uint d = 1; d < 16; d *= 2) {
            ae += simd_shuffle_xor(ae, d);
            be += simd_shuffle_xor(be, d);
        }
        return simd_shuffle(be, first) < simd_shuffle(ae, first) ? b : a;
    } else {
        float top = simd_max(abs(x));
        float scale = top > 0 ? exp2(ceil(log2(top / 448.0f))) : 1.0f;
        return fused_e4m3(x / scale) * scale;
    }
}
// Row maximum of |x| as MLX's max reduce returns it: any NaN makes it NaN.
inline float fused_row_peak(float local, bool nan_seen, threadgroup float* scratch,
                            uint simd, uint lane, uint simds) {
    float m = simd_max(local);
    bool n = simd_any(nan_seen);
    if (lane == 0) scratch[simd] = n ? NAN : m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float peak = 0;
    bool any = false;
    for (uint s = 0; s < simds; ++s) {
        any = any || isnan(scratch[s]);
        peak = max(peak, scratch[s]);
    }
    return any ? NAN : peak;
}
// MLX sort order: a before b when a < b, a number before NaN, then the lower index.
inline bool fused_before(float a, uint ia, float b, uint ib) {
    bool an = isnan(a), bn = isnan(b);
    bool less = (an || bn) ? (!an && bn) : (a < b);
    bool more = (an || bn) ? (!bn && an) : (b < a);
    return less || (!more && ia < ib);
}
"""

INPUTS = """
    uint row = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    threadgroup float scratch[32];
    float v[E];
    float local = 0;
    bool nan_seen = false;
    for (int c = 0; c < E; ++c) {
        uint i = row * W + c * THREADS + t;
        v[c] = float(x[i]);
        router_in[i] = v[c];
        float a = abs(v[c]);
        nan_seen = nan_seen || isnan(a);
        local = max(local, a);
    }
#if FP4
    float peak = fused_row_peak(local, nan_seen, scratch, simd, lane, THREADS / 32);
    if (t == 0) outer[row] = peak / 1536.0f;
#else
    float peak = 0.0f;
#endif
    for (int c = 0; c < E; ++c)
#ifdef OPERAND
        values[row * W + c * THREADS + t] = static_cast<OPERAND>(fused_quantize<bool(FP4)>(v[c], peak, lane));
#else
        values[row * W + c * THREADS + t] = fused_quantize<bool(FP4)>(v[c], peak, lane);
#endif
"""

ROUTE = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    float s[SLOTS], key[SLOTS];
    bool taken[SLOTS];
    for (int i = 0; i < SLOTS; ++i) {
        uint e = i * 32 + lane;
        float z = router[row * (SLOTS * 32) + e];
#if ROUTE_SCORES
        s[i] = z;
#else
        float y = 1 / (1 + metal::precise::exp(metal::abs(z)));
        s[i] = (z < 0) ? y : 1 - y;
#endif
        float sb = s[i] + float(bias[e]);
        key[i] = -sb;
        taken[i] = false;
    }
    float chosen = 0, total = 0;
    uint expert = 0;
    for (int k = 0; k < K; ++k) {
        float best = NAN;
        uint at = 0xffffffffu;
        for (int i = 0; i < SLOTS; ++i)
            if (!taken[i] && fused_before(key[i], i * 32 + lane, best, at)) {
                best = key[i];
                at = i * 32 + lane;
            }
        for (uint d = 16; d > 0; d /= 2) {
            float ob = simd_shuffle_xor(best, d);
            uint oa = simd_shuffle_xor(at, d);
            if (fused_before(ob, oa, best, at)) {
                best = ob;
                at = oa;
            }
        }
        uint slot = at / 32;
        float p = 0;
        for (int i = 0; i < SLOTS; ++i) {
            if (i == int(slot)) p = s[i];
            if (i == int(slot) && lane == at % 32) taken[i] = true;
        }
        p = simd_shuffle(p, at % 32);
        total = p + total;
        if (lane == uint(k)) {
            chosen = p;
            expert = at;
        }
    }
    if (lane < uint(K)) {
#if ROUTE_NORMALIZE
        float denominator = total + consts[1];
        float share = chosen / denominator;
        float scaled = share * consts[0];
        prob[row * K + lane] = scaled;
#else
        prob[row * K + lane] = chosen;
#endif
        experts[row * K + lane] = expert;
        rows[row * K + lane] = row;
    }
"""

SWIGLU = """
    uint p = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    threadgroup float scratch[32];
#if FP4
    float scale = row_outer[rows[p]] * expert_outer[experts[p]];
#endif
    float h[E];
    float local = 0;
    bool nan_seen = false;
    for (int c = 0; c < E; ++c) {
        uint j = c * THREADS + t;
        float graw = raw[p * (2 * W) + 2 * j];
        float uraw = raw[p * (2 * W) + 2 * j + 1];
#if FP4
        float gs = graw * scale;
        float us = uraw * scale;
        T gq = static_cast<T>(gs);
        T uq = static_cast<T>(us);
#else
        T gq = static_cast<T>(graw);
        T uq = static_cast<T>(uraw);
#endif
        float gate = float(gq);
        float up = float(uq);
        float gcap = static_cast<float>(GATE_CAP);
        float ucap = static_cast<float>(UP_CAP);
        float t1 = gate / gcap;
        float t2 = metal::precise::tanh(t1);
        float t3 = gcap * t2;
        float y = 1 / (1 + metal::exp(metal::abs(gate)));
        float t4 = (gate < 0) ? y : 1 - y;
        float t5 = t3 * t4;
        float t6 = up / ucap;
        float t7 = metal::precise::tanh(t6);
        float t8 = ucap * t7;
        float t9 = t5 * t8;
        T m = static_cast<T>(t9);
        h[c] = float(m);
        float a = abs(h[c]);
        nan_seen = nan_seen || isnan(a);
        local = max(local, a);
    }
#if FP4
    float peak = fused_row_peak(local, nan_seen, scratch, simd, lane, THREADS / 32);
    if (t == 0) mid_outer[p] = peak / 1536.0f;
#else
    float peak = 0.0f;
#endif
    for (int c = 0; c < E; ++c)
#ifdef OPERAND
        values[p * W + c * THREADS + t] = static_cast<OPERAND>(fused_quantize<bool(FP4)>(h[c], peak, lane));
#else
        values[p * W + c * THREADS + t] = fused_quantize<bool(FP4)>(h[c], peak, lane);
#endif
"""

COMBINE = """
    uint c = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    if (c >= uint(H)) return;
    float acc = 0;
    for (int k = 0; k < K; ++k) {
        uint p = n * K + k;
        float v = raw[p * H + c];
#if FP4
        float scale = mid_outer[p] * expert_outer[experts[p]];
        float vs = v * scale;
        T q = static_cast<T>(vs);
#else
        T q = static_cast<T>(v);
#endif
        float w = float(q) * prob[p];
        float u = w + 0.0f;
        acc = u + acc;
    }
    out[n * H + c] = static_cast<T>(acc);
"""


# The combine for pairs held in sorted order: pair p (row-major, as `prob` and `experts`) reads
# its raw product and down-input scale at its sorted position inv[p].
COMBINE_SORTED = (COMBINE
                  .replace("uint p = n * K + k;", "uint p = n * K + k;\n        uint s = inv[p];")
                  .replace("raw[p * H + c]", "raw[s * H + c]")
                  .replace("mid_outer[p]", "mid_outer[s]"))
assert COMBINE_SORTED.count("inv[p]") == 1 and "raw[s * H + c]" in COMBINE_SORTED

# The SwiGLU and the sorted combine reading products the product launch has already scaled and rounded
# (``fast.prompt_experts.product(..., rounded=...)``): the value each pair's stage would compute from the float32
# sum (its outer scale times the sum, rounded to T; or the sum rounded to T) arrives as T, half the bytes, and the
# rest of the stage is the same text.
_SWIGLU_SCALE = """        float graw = raw[p * (2 * W) + 2 * j];
        float uraw = raw[p * (2 * W) + 2 * j + 1];
#if FP4
        float gs = graw * scale;
        float us = uraw * scale;
        T gq = static_cast<T>(gs);
        T uq = static_cast<T>(us);
#else
        T gq = static_cast<T>(graw);
        T uq = static_cast<T>(uraw);
#endif"""
_SWIGLU_ROUNDED = """        T gq = raw[p * (2 * W) + 2 * j];
        T uq = raw[p * (2 * W) + 2 * j + 1];"""
assert SWIGLU.count(_SWIGLU_SCALE) == 1
SWIGLU_ROUNDED = SWIGLU.replace(_SWIGLU_SCALE, _SWIGLU_ROUNDED).replace(
    "    float scale = row_outer[rows[p]] * expert_outer[experts[p]];\n", "")
_COMBINE_SCALE = """        float v = raw[s * H + c];
#if FP4
        float scale = mid_outer[s] * expert_outer[experts[p]];
        float vs = v * scale;
        T q = static_cast<T>(vs);
#else
        T q = static_cast<T>(v);
#endif"""
assert COMBINE_SORTED.count(_COMBINE_SCALE) == 1
COMBINE_SORTED_ROUNDED = COMBINE_SORTED.replace(_COMBINE_SCALE, "        T q = raw[s * H + c];")

PROMPT_ROWS = 4096  # widest prompt chunk the sorted route serves (the prompt read uses 512; a request may ask wider)
SORTED_TILE_PAIRS = 4  # pairs an expert from which MLX's sorted product runs its tile kernel


def mlx_literal(value):
    """The text MLX's compiler writes for a float32 scalar constant (7 significant digits)."""
    return "%.7g" % float(mx.array(value, mx.float32).item())


def kernel_spec(stage, fp4, options=()):
    """Name, input names, output names, body and header handed to mx.fast.metal_kernel.

    `options` are (macro, value) pairs: the SwiGLU caps, or the route variant switches.
    """
    fp4_only = lambda *names: list(names) if fp4 else []
    inputs, outputs, body = {
        "inputs": (["x"], ["router_in", "values", *fp4_only("outer")], INPUTS),
        "route": (["router", "bias", "consts"], ["experts", "rows", "prob"], ROUTE),
        "swiglu": (
            ["raw", *fp4_only("row_outer", "expert_outer"), "experts", "rows"],
            ["values", *fp4_only("mid_outer")],
            SWIGLU,
        ),
        "combine": (
            ["raw", *fp4_only("mid_outer", "expert_outer"), "experts", "prob"],
            ["out"],
            COMBINE,
        ),
        "combine_sorted": (
            ["raw", *fp4_only("mid_outer", "expert_outer"), "experts", "prob", "inv"],
            ["out"],
            COMBINE_SORTED,
        ),
        "swiglu_rounded": (["raw", "experts", "rows"], ["values", *fp4_only("mid_outer")], SWIGLU_ROUNDED),
        "combine_sorted_rounded": (["raw", "experts", "prob", "inv"], ["out"], COMBINE_SORTED_ROUNDED),
    }[stage]
    header = HEADER + f"constexpr constant int THREADS = {THREADS};\n#define FP4 {int(fp4)}\n"
    header += "".join(f"#define {key} {value}\n" for key, value in options)
    name = f"mtplx_inkling_fused_{stage}_{'nvfp4' if fp4 else 'mxfp8'}"
    name += "".join(f"_{key.lower()}_{re.sub(r'[^0-9A-Za-z]', 'x', str(value))}" for key, value in options)
    return name, inputs, outputs, body, header


@lru_cache(maxsize=None)
def _kernel(stage, fp4, options=()):
    name, inputs, outputs, body, header = kernel_spec(stage, fp4, options)
    return mx.fast.metal_kernel(
        name=name, input_names=inputs, output_names=outputs, source=body, header=header
    )


@lru_cache(maxsize=None)
def _route_consts(scale):
    # The same float32 conversions MLX applies to `* scale` and `+ 1e-20`.
    return mx.array([scale, 1e-20], mx.float32)


@lru_cache(maxsize=None)
def _pair_ids(pairs):
    # The down gather's default lhs indices, built once instead of one Arange per visit.
    return mx.arange(pairs, dtype=mx.uint32)


def fusion_mode():
    mode = os.environ.get("MTPLX_INKLING_EXPERT_FUSION", "auto").lower()
    if mode in {"0", "off", "stock"}:
        return "off"
    if mode in {"auto", "on", "1", "candidate"}:
        return "auto"
    raise ValueError("MTPLX_INKLING_EXPERT_FUSION must be auto or off")


def route_variant():
    """`fused` (preferred): the sigmoid in the kernel with metal::precise::exp, one primitive fewer.
    `scores`: stock mx.sigmoid, fused selection and normalisation.
    `ops`: fused selection only; MLX's own ops normalise (diagnostic).
    """
    variant = os.environ.get("MTPLX_INKLING_EXPERT_ROUTE", "fused").lower()
    if variant not in ROUTES:
        raise ValueError(f"MTPLX_INKLING_EXPERT_ROUTE must be one of {ROUTES}")
    return variant


def route_ladder():
    """The route variants a visit may be served by, in order.

    With no switch a visit takes `fused` where its first-use proof passes on this device and
    `scores` where only that one passes, so a GPU whose precise exponential does not give
    `mx.sigmoid`'s bits keeps the rest of the fused block. The switch pins one variant.
    """
    if "MTPLX_INKLING_EXPERT_ROUTE" in os.environ:
        return (route_variant(),)
    return ("fused", "scores")


def unserved(block, flat, gpu=None):
    """Why the fused path cannot serve this call, or None. Cheap and deterministic."""
    if fusion_mode() == "off":
        return "disabled"
    gpu = gpu if gpu is not None else mx.default_device() == mx.gpu and mx.metal.is_available()
    if not gpu:
        return "cpu"
    rows, width = flat.shape
    if rows > MAX_ROWS:
        return "grouped_rows"
    return _block_reason(block, flat)


def prompt_unserved(block, flat, gpu=None):
    """Why the sorted prompt route cannot serve this call (33 to PROMPT_ROWS rows), or None."""
    if fusion_mode() == "off" or os.environ.get("MTPLX_INKLING_PROMPT_EXPERTS", "fused").lower() == "stock":
        return "disabled"
    gpu = gpu if gpu is not None else mx.default_device() == mx.gpu and mx.metal.is_available()
    if not gpu:
        return "cpu"
    if not MAX_ROWS < flat.shape[0] <= PROMPT_ROWS:
        return "prompt_rows"
    return _block_reason(block, flat)


def _invariant_unserved(block, flat):
    """Why the width-independent prompt route cannot serve this call, or None."""
    if not prompt_kernels():
        return "no_prompt_kernels"
    if not 1 <= flat.shape[0] <= PROMPT_ROWS:
        return "prompt_rows"
    return _block_reason(block, flat)


def _block_reason(block, flat):
    rows, width = flat.shape
    if block.gate_up.mode not in {"nvfp4", "mxfp8"} or block.gate_up.mode != block.down.mode:
        return "format"
    if block.gate_up.affine is not None or block.down.affine is not None:
        return "affine_experts"
    if flat.dtype not in {mx.bfloat16, mx.float16}:
        return "dtype"
    experts, outputs = block.router.shape
    middle = block.down.codes.shape[-1] * (8 if block.down.mode == "nvfp4" else 4)
    if (block.router.dtype != mx.float32 or outputs != width or width % THREADS
            or middle % THREADS or experts % 32 or experts > 1024 or not 0 < block.count <= 32
            or block.gate_up.codes.shape[-2] != 2 * middle or block.down.codes.shape[-2] != width):
        return "geometry"
    return None


def route(logits, bias, count, scale, variant=None):
    """Expert ids, row ids and route weights for (rows, experts) FP32 router logits."""
    variant = variant or route_variant()
    rows, total = logits.shape
    pairs = rows * count
    options = (("ROUTE_SCORES", int(variant != "fused")), ("ROUTE_NORMALIZE", int(variant != "ops")))
    experts, row_ids, prob = _kernel("route", True, options)(
        inputs=[logits if variant == "fused" else mx.sigmoid(logits), bias, _route_consts(float(scale))],
        template=[("SLOTS", total // 32), ("K", count)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(pairs,), (pairs,), (pairs,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.float32],
    )
    if variant == "ops":
        chosen = prob.reshape(rows, count)
        prob = (chosen / (chosen.sum(axis=-1, keepdims=True) + 1e-20) * scale).reshape(-1)
    return experts, row_ids, prob


def routed(block, flat, linear, trace=None, variant=None):
    """Mixture of experts for `flat` (rows, hidden): four fused kernels around MLX's own
    router matmul, router sigmoid and two quantized gathers. Call only when `unserved`
    returns None. `trace`, when a dict, receives every intermediate.
    """
    fp4 = block.gate_up.mode == "nvfp4"
    rows, width = flat.shape
    count, pairs = block.count, rows * block.count
    middle = block.down.codes.shape[-1] * (8 if fp4 else 4)
    bits, group = (4, 16) if fp4 else (8, 32)
    scalars = [(rows,)] if fp4 else []

    router_in, values, *outer = _kernel("inputs", fp4)(
        inputs=[flat],
        template=[("T", flat.dtype), ("W", width), ("E", width // THREADS)],
        grid=(rows * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, width), (rows, 1, width), *scalars],
        output_dtypes=[mx.float32] * (3 if fp4 else 2),
    )
    logits = linear(router_in, block.router)
    if logits.dtype != mx.float32:
        raise TypeError("Inkling router logits must be float32 for the fused route")
    experts, row_ids, prob = route(logits, block.bias, count, block.scale, variant)
    raw13 = mx.gather_qmm(
        values, block.gate_up.codes, scales=block.gate_up.scales, lhs_indices=row_ids,
        rhs_indices=experts, group_size=group, bits=bits, mode=block.gate_up.mode,
    )
    caps = (("GATE_CAP", mlx_literal(block.gate_cap)), ("UP_CAP", mlx_literal(block.up_cap)))
    mid, *mid_outer = _kernel("swiglu", fp4, caps)(
        inputs=[raw13.reshape(pairs, 2 * middle),
                *([outer[0], block.gate_up.outer] if fp4 else []), experts, row_ids],
        template=[("T", flat.dtype), ("W", middle), ("E", middle // THREADS)],
        grid=(pairs * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(pairs, 1, middle), *([(pairs,)] if fp4 else [])],
        output_dtypes=[mx.float32] * (2 if fp4 else 1),
    )
    raw2 = mx.gather_qmm(
        mid, block.down.codes, scales=block.down.scales, lhs_indices=_pair_ids(pairs),
        rhs_indices=experts, group_size=group, bits=bits, mode=block.down.mode,
    )
    (out,) = _kernel("combine", fp4)(
        inputs=[raw2.reshape(pairs, width),
                *([mid_outer[0], block.down.outer] if fp4 else []), experts, prob],
        template=[("T", flat.dtype), ("H", width), ("K", count)],
        grid=(width, rows, 1),
        threadgroup=(256 if width % 256 == 0 else 128, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[flat.dtype],
    )
    if trace is not None:
        trace.update(router_in=router_in, values13=values, outer13=outer[0] if fp4 else None,
                     logits=logits, experts=experts, row_ids=row_ids, prob=prob, raw13=raw13,
                     values2=mid, outer2=mid_outer[0] if fp4 else None, raw2=raw2, out=out)
    return out


def prompt_kernels():
    """Whether the prompt route's pair sort and expert products run as our launches
    (``fast.prompt_experts``: tensor-unit GPUs with readable MLX headers), else MLX's argsorts,
    gathers and sorted products. ``MTPLX_INKLING_PROMPT_KERNELS=mlx`` keeps MLX's."""
    if os.environ.get("MTPLX_INKLING_PROMPT_KERNELS", "ours").strip().lower() in ("mlx", "0", "off", "stock"):
        return False
    from .fast import prompt_experts

    return prompt_experts.available()[0]


def routed_sorted(block, flat, linear, trace=None, variant=None, fast=False, invariant=False):
    """Mixture of experts for a prompt chunk (33 to PROMPT_ROWS rows): the stock body's sort and two
    sorted quantized products on the same inputs, around the fused input, route, SwiGLU and combine
    kernels. Call only when `prompt_unserved` returns None.

    On tensor-unit GPUs (`prompt_kernels`) the sort and both products are our launches
    (`fast.prompt_experts`): the token rows are read in place through the sort's row map, and every
    product row has the sums of MLX's sorted product (NVFP4 on bfloat16 operands, which hold the
    rounded expert inputs exactly; an NVFP4 product's sums are exact, so the operand type cannot move
    a bit). Elsewhere MLX's argsorts, gathers and sorted products run, as in the stock body.

    `fast` (the prompt read's fast route, `fast.prompt_route.prompt_fast("experts")`, off by
    default) changes the MXFP8 blocks only: their gate-up takes the sorted product on the
    accelerator, as NVFP4's does, instead of the one-row panel product, and both MXFP8 products
    read bfloat16 operands (the inputs and SwiGLU kernels write the rounded expert inputs in
    bfloat16, which holds them exactly: at most 4 significant bits). An MXFP8 product's bfloat16
    output is its float32 sum rounded once, the rounding the SwiGLU and combine kernels apply to
    a float32 output, so the only change is the gate-up's order of summation: the sorted
    product's (MLX's own grouped product) instead of the one-row product's (class C against the
    default).

    `invariant` (the prompt read's ``invariant`` part, off by default; requires our launches): every
    product is ours at every row count, the MXFP8 gate-up included and both products below 4 pairs an
    expert, so a pair's sums do not depend on how many rows the chunk holds.
    """
    fp4 = block.gate_up.mode == "nvfp4"
    rows, width = flat.shape
    count, pairs = block.count, rows * block.count
    middle = block.down.codes.shape[-1] * (8 if fp4 else 4)
    bits, group = (4, 16) if fp4 else (8, 32)
    scalars = [(rows,)] if fp4 else []
    ours = prompt_kernels()
    # MLX multiplies sorted pairs on its tile kernel only from 4 pairs an expert on; below that each
    # pair is its one-row product. An NVFP4 product's sums are exact, so either is ours; an MXFP8
    # block below that keeps MLX's products (and the fast route's MXFP8 gate-up its panel).
    tiled = fp4 or invariant or pairs >= SORTED_TILE_PAIRS * block.router.shape[0]
    fast = (fast or invariant) and not fp4 and tiled
    # bfloat16 operands: the fast route's MXFP8 products, and NVFP4 products on our launches. MLX's
    # products of an MXFP8 block take float32 operands, and so do ours in their place.
    operand = mx.bfloat16 if fast or (ours and fp4) else mx.float32

    # Kernels writing the rounded expert inputs in bfloat16 (OPERAND); without it the kernel source is
    # the float32 route's, unchanged.
    narrow = (("OPERAND", "bfloat16_t"),) if operand == mx.bfloat16 else ()
    router_in, values, *outer = _kernel("inputs", fp4, narrow)(
        inputs=[flat],
        template=[("T", flat.dtype), ("W", width), ("E", width // THREADS)],
        grid=(rows * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, width), (rows, 1, width), *scalars],
        output_dtypes=[mx.float32, operand] + ([mx.float32] if fp4 else []),
    )
    if ours:
        # The router's product as our launch with MLX's sums where the shape is admitted (fast/prompt_gemm.py).
        from .fast import prompt_gemm

        (logits,) = prompt_gemm.project(router_in, [block.router], "router")
    else:
        logits = linear(router_in, block.router)
    if logits.dtype != mx.float32:
        raise TypeError("Inkling router logits must be float32 for the fused route")
    experts, _, prob = route(logits, block.bias, count, block.scale, variant)
    if ours:
        from .fast import prompt_experts

        rows_a_tile, wide = prompt_experts.tiling(pairs, block.router.shape[0])
        order, sorted_rows, sorted_experts, inverse, tiles, starts = prompt_experts.sort_pairs(
            experts, count, block.router.shape[0], rows_a_tile)
    else:
        # The stock body's sort: pairs ordered by expert, each pair's row its index // count.
        order = mx.argsort(experts)
        sorted_experts = experts[order]
        sorted_rows = order // count
        inverse = None
    # Our products scale and round each sum in their epilogue as the SwiGLU and the combine would (half the
    # bytes, the same values); a traced call keeps the float32 sums it records.
    rounded = ours and trace is None
    rounded13 = False
    if fp4 or fast:
        if ours:
            raw13 = prompt_experts.product(
                values, block.gate_up, tiles, starts, pairs, rowmap=sorted_rows, wide=wide,
                rounded=((flat.dtype, outer[0], block.gate_up.outer) if fp4 else (flat.dtype, None, None))
                if rounded else None)
            rounded13 = rounded
        else:
            # The stock body's grouped form (its rows gathered in sorted order, one sorted matrix
            # product), which its measured route proves equal to the indexed form before using it
            # (NVFP4); the fast route takes it for MXFP8 too.
            raw13 = mx.gather_qmm(
                values[sorted_rows], block.gate_up.codes, scales=block.gate_up.scales,
                rhs_indices=sorted_experts, group_size=group, bits=bits, mode=block.gate_up.mode,
                sorted_indices=True,
            )
    else:
        # The stock body's indexed form computes each pair as MLX's one-row gathered product and
        # re-reads its expert for every pair; the panel product gives each pair the same bits and
        # reads each expert once for all of its pairs.
        from .fast.experts import prompt_gate_up

        raw13 = prompt_gate_up(block, values, sorted_experts, sorted_rows)
    caps = (("GATE_CAP", mlx_literal(block.gate_cap)), ("UP_CAP", mlx_literal(block.up_cap)))
    mid, *mid_outer = _kernel("swiglu_rounded" if rounded13 else "swiglu", fp4, caps + narrow)(
        inputs=[raw13.reshape(pairs, 2 * middle),
                *([outer[0], block.gate_up.outer] if fp4 and not rounded13 else []), sorted_experts, sorted_rows],
        template=[("T", flat.dtype), ("W", middle), ("E", middle // THREADS)],
        grid=(pairs * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(pairs, 1, middle), *([(pairs,)] if fp4 else [])],
        output_dtypes=[operand] + ([mx.float32] if fp4 else []),
    )
    rounded2 = False
    if ours and tiled:
        raw2 = prompt_experts.product(
            mid, block.down, tiles, starts, pairs, wide=wide,
            rounded=((flat.dtype, mid_outer[0], block.down.outer) if fp4 else (flat.dtype, None, None))
            if rounded else None)
        rounded2 = rounded
    else:
        raw2 = mx.gather_qmm(
            mid, block.down.codes, scales=block.down.scales, rhs_indices=sorted_experts,
            group_size=group, bits=bits, mode=block.down.mode, sorted_indices=True,
        )
        inverse = mx.argsort(order) if inverse is None else inverse
    (out,) = _kernel("combine_sorted_rounded" if rounded2 else "combine_sorted", fp4)(
        inputs=[raw2.reshape(pairs, width),
                *([mid_outer[0], block.down.outer] if fp4 and not rounded2 else []), experts, prob, inverse],
        template=[("T", flat.dtype), ("H", width), ("K", count)],
        grid=(width, rows, 1),
        threadgroup=(256 if width % 256 == 0 else 128, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[flat.dtype],
    )
    if trace is not None:
        trace.update(router_in=router_in, values13=values, logits=logits, experts=experts, prob=prob,
                     order=order, sorted_rows=sorted_rows, sorted_experts=sorted_experts, inverse=inverse,
                     raw13=raw13, values2=mid, raw2=raw2, out=out)
    return out


class FusionState:
    """One model's proof cache, its memoised decisions and its visit counters."""

    def __init__(self, proofs):
        self.proofs = proofs
        self.decisions = {}
        self.counts = Counter()
        self.lock = RLock()

    def record(self, key, *proofs):
        with self.lock:
            first = key not in self.counts
            self.counts[key] += 1
            receipt = _request.get()
            if receipt is not None:
                selections = receipt["selections"]
                selections[key] = selections.get(key, 0) + 1
                for proof in proofs:
                    if proof is not None:
                        receipt["first_use"][proof] = receipt["first_use"].get(proof, 0) + 1
        if first:
            _log.info("Inkling experts %s", key)

    def health(self):
        with self.lock:
            decisions = {"/".join(map(str, spec)): ("fused" if plan is not None else reason)
                         for spec, (plan, reason) in self.decisions.items()}
            return {"process_selections": dict(self.counts), "decisions": decisions,
                    "proofs_checked": self.proofs.checks, "proofs_reused": self.proofs.hits,
                    "proofs_persistent": self.proofs.directory is not None}


_fallback_state = None


def bind(model, proofs):
    """Share one FusionState, around the model's ProofCache, across its expert blocks."""
    state = FusionState(proofs)
    for layer in model.layers:
        layer.experts.fusion = state
    return state


def _state(block):
    global _fallback_state
    state = getattr(block, "fusion", None)
    if state is None:
        if _fallback_state is None:
            _fallback_state = FusionState(ProofCache(None, None))
        state = _fallback_state
    return state


@lru_cache(maxsize=None)
def _key(fmt, rows, outcome):
    return f"{fmt}.r{rows if rows <= MAX_ROWS else '>' + str(MAX_ROWS)}.{outcome}"


def _decline(reason):
    """A short counter label for a ProofCache refusal, also after a reload from disk."""
    text = str(reason)
    if text.startswith("mismatch:"):
        return "first_use_mismatch"
    if "first-use check raised" in text:
        return "first_use_error"
    return "first_use_declined"


def _decide(state, block, flat, linear, stock, variant):
    """(plan, reason, output, proof) for this call; output is set only on the proving call."""
    fmt, rows = block.gate_up.mode, flat.shape[0]
    width = "decode" if rows == 1 else "verify" if rows <= MAX_ROWS else "prompt"
    serve = routed_sorted if width == "prompt" else routed
    spec = (fmt, str(flat.dtype).split(".")[-1], variant, width)
    with state.lock:
        if spec in state.decisions:
            return (*state.decisions[spec], None, None)
        proved = []

        def prove():
            want = stock(flat)
            mx.eval(want)
            proved.append(want)
            got = serve(block, flat, linear, variant=variant)
            mx.eval(got)
            if bool(mx.all(got.view(mx.uint16) == want.view(mx.uint16)).item()):
                proved[0] = got
                return {"route": variant, "rows_checked": rows}, None
            return None, Mismatch(f"mismatch: fused {'/'.join(spec)} differs from the stock body")

        shape = {"format": fmt, "dtype": spec[1], "route": variant, "width": spec[3],
                 "experts": block.router.shape[0], "hidden": flat.shape[1],
                 "middle": block.down.codes.shape[-1] * (8 if fmt == "nvfp4" else 4),
                 "top_k": block.count, "route_scale": float(block.scale),
                 "caps": [mlx_literal(block.gate_cap), mlx_literal(block.up_cap)]}
        plan, reason = state.proofs.decide(NAME, shape, prove)
        if plan is None:
            _log.warning("Inkling expert route %s is not used: %s", "/".join(spec), reason)
        state.decisions[spec] = (plan, reason)
        checked = ("passed" if plan is not None else "declined") if proved else None
        return plan, reason, (proved[0] if proved else None), checked


def _prompt_fast(part):
    from .fast.prompt_route import prompt_fast

    return prompt_fast(part)


def experts_forward(block, hidden, linear, stock, prompt=False):
    """Serve `block(hidden)` through the fused path when admitted, else the stock body. `prompt`: the call is a
    prompt chunk's (``model.Layer`` without a verify tape)."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    rows, fmt = flat.shape[0], getattr(block.gate_up, "mode", "unknown")
    state = _state(block)
    if prompt and _prompt_fast("invariant"):
        # The width-independent prompt read (off by default, class C): our sort and products at every row count.
        reason = _invariant_unserved(block, flat)
        if reason is None:
            state.record(_key(fmt, rows, "prompt_invariant." + route_variant()))
            return routed_sorted(block, flat, linear, variant=route_variant(), fast=True,
                                 invariant=True).reshape(hidden.shape)
        state.record(_key(fmt, rows, "prompt_invariant.declined." + reason))
    reason = unserved(block, flat)
    prompt = reason == "grouped_rows"
    if prompt:
        reason = prompt_unserved(block, flat)
        reason = None if reason is None else "grouped_rows" if reason == "prompt_rows" else "prompt_" + reason
    if reason is None:
        proofs, kept = [], None
        for variant in route_ladder():
            plan, refusal, out, proof = _decide(state, block, flat, linear, stock, variant)
            proofs.append(proof)
            if plan is not None:
                if prompt and _prompt_fast("experts"):
                    # The prompt read's fast route (off by default, not bit-identical): the
                    # admitted route's kernels on bfloat16 operands (see `routed_sorted`).
                    state.record(_key(fmt, rows, "prompt_fast." + variant), *proofs)
                    return routed_sorted(block, flat, linear, variant=variant, fast=True).reshape(hidden.shape)
                state.record(_key(fmt, rows, ("prompt_fused." if prompt else "fused.") + variant), *proofs)
                if out is None:
                    out = (routed_sorted if prompt else routed)(block, flat, linear, variant=variant)
                return out.reshape(hidden.shape)
            kept = out if out is not None else kept  # the stock body a failed proof computed
        state.record(_key(fmt, rows, "stock." + _decline(refusal)), *proofs)
        return stock(hidden) if kept is None else kept.reshape(hidden.shape)
    state.record(_key(fmt, rows, "stock." + reason))
    return stock(hidden)


def observe_expert_request(fn, rt, *args, **kwargs):
    """Run one generation request with request-local expert route counters on its stats."""
    if _request.get() is not None:
        return fn(rt, *args, **kwargs)
    receipt = {"mode": fusion_mode(), "route": route_variant(), "selections": {}, "first_use": {}}
    token = _request.set(receipt)
    try:
        output = fn(rt, *args, **kwargs)
        output.stats.inkling_experts = receipt
        return output
    finally:
        _request.reset(token)


def expert_fusion_health(model):
    """Mode, route, the model's decisions and its visit totals for /health."""
    states = {id(s): s for s in (getattr(layer.experts, "fusion", None) for layer in model.layers) if s}
    payload = {"mode": fusion_mode(), "route": route_variant(), "bound": bool(states)}
    for state in states.values():
        payload.update(state.health())
    return payload
