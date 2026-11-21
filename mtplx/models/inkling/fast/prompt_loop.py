"""The loop passes' glue at a prompt chunk, as our launches: the cross-pass controllers and the input norms.

On passes two and three every visit drives each of its sources (three for attention, one for the experts)
through a low-rank controller fed by the visit's cross-pass message (``model.Controller``): two gain-free
norms, their concatenation and cast, a down product, the forget and gain products, and the tanh mix; then
each source is normed with its input gain. The prompt path writes that as about fifteen MLX launches a
controller and two a norm. Here a controller is four launches (its inputs, its down product, its forget and
gain products as one, its mix: the products by ``fast/prompt_gemm.py``) and a visit's norms one:

``controller_inputs``
    MLX's single-row RMS norm of the source and of the message, each four values a thread multiplied and
    added from zero, summed across the simdgroup, then across the simdgroups' totals (zero-padded to 32),
    precise reciprocal square root, ``1 * (x * inv)``; both halves cast to bfloat16 as the concatenation is.
``controller_mix``
    ``(1 - tanh(forget)) * x + tanh(gain) * message``, precise tanh, each operation rounded on its own.
``input_norms``
    the same norm with each source's bfloat16 gain, ``gain * (x * inv)``, cast to the stream's type.

The arithmetic is the decode step's (``fast/read.py``, which serves one row at a time with the same norms
and mix), so every value has the prompt path's bits. The visit's one-query depth read (window visits, and
the expert input) is that file's read kernel; the three-query read of a global visit and the four-query
message read stay MLX's matrix products (their sums are the tensor units'), and their outputs feed these
launches. ``visit_inputs`` and ``expert_input`` are what ``model.Layer`` calls on passes two and three;
``input_norms`` also serves a global visit's three norms on the first pass.

Chunks of more than 32 rows, bfloat16 or float16 streams; anything else keeps the Layer's statements
(counted under its reason).
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import struct
from threading import Lock

import mlx.core as mx

_lock = Lock()
_counts: Counter = Counter()


def counts():
    with _lock:
        return dict(_counts)


def _count(key):
    with _lock:
        _counts[key] += 1


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _threads(width):
    return -(-(width // 4) // 32) * 32


# The single-row RMS of one row's D values held four a thread: this thread's partial, the simdgroup's, the
# threadgroup's (simdgroup totals zero-padded to 32 and summed by the first simdgroup); `slot` picks a row of
# `totals` so several norms can share the scratch.
_RMS = """
inline float mtplx_rms_inverse(const thread float (&v)[4], threadgroup float (*totals)[32], uint slot,
                               threadgroup float* inverse, uint sg, uint lane, uint groups, uint width, float eps) {
    float acc = 0;
    for (uint i = 0; i < 4; ++i)
        acc += v[i] * v[i];
    acc = simd_sum(acc);
    if (lane == 0)
        totals[slot][sg] = acc;
    if (sg == 0 && lane >= groups)
        totals[slot][lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float total = simd_sum(totals[slot][lane]);
        if (lane == 0)
            inverse[slot] = metal::precise::rsqrt(total / float(width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inverse[slot];
}
"""

CONTROLLER_INPUTS = """
    // One threadgroup a row: source SRC of x (rows, NS, D) and message MSG of msg (rows, NM, D), both float32,
    // into mixed (rows, 2D) in the down product's type M.
    const uint r = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint groups = threads_per_threadgroup.x / 32;
    threadgroup float totals[2][32];
    threadgroup float inverse[2];
    const bool held = t * 4 < uint(D);
    float xv[4];
    float mv[4];
    for (uint i = 0; i < 4; ++i) {
        xv[i] = held ? x[(size_t(r) * NS + SRC) * D + t * 4 + i] : 0.0f;
        mv[i] = held ? msg[(size_t(r) * NM + MSG) * D + t * 4 + i] : 0.0f;
    }
    const float eps = as_type<float>(uint(EPS));
    const float inv_x = mtplx_rms_inverse(xv, totals, 0, inverse, sg, lane, groups, uint(D), eps);
    const float inv_m = mtplx_rms_inverse(mv, totals, 1, inverse, sg, lane, groups, uint(D), eps);
    if (held) {
        for (uint i = 0; i < 4; ++i) {
            mixed[size_t(r) * (2 * D) + t * 4 + i] = static_cast<M>(1.0f * (xv[i] * inv_x));
            mixed[size_t(r) * (2 * D) + D + t * 4 + i] = static_cast<M>(1.0f * (mv[i] * inv_m));
        }
    }
"""

CONTROLLER_MIX = """
    // One thread a value: (1 - tanh(forget)) * x + tanh(gain) * message, each operation rounded on its own.
    const uint at = thread_position_in_grid.x;
    const uint total = uint(forget_shape[0]) * uint(D);
    if (at >= total)
        return;
    const uint r = at / uint(D);
    const uint c = at % uint(D);
    const float t1 = metal::precise::tanh(forget[at]);
    const float a = 1.0f - t1;
    const float b = a * x[(size_t(r) * NS + SRC) * D + c];
    const float t2 = metal::precise::tanh(gain[at]);
    const float m = t2 * msg[(size_t(r) * NM + MSG) * D + c];
    h[at] = b + m;
"""

INPUT_NORMS = """
    // One threadgroup a row and source: source s normed with its gain gs into outs (the stream's type T). The
    // sources are h0, h1, h2 ((rows, D) float32 each) or, PACKED, the NS rows of h0 ((rows, NS, D)).
    const uint r = threadgroup_position_in_grid.x / uint(NS);
    const uint s = threadgroup_position_in_grid.x % uint(NS);
    const uint t = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint groups = threads_per_threadgroup.x / 32;
    threadgroup float totals[1][32];
    threadgroup float inverse[1];
    const bool held = t * 4 < uint(D);
    const auto src = PACKED ? h0 + size_t(s) * D : (s == 0 ? h0 : s == 1 ? h1 : h2);
    const size_t stride = PACKED ? size_t(NS) * D : size_t(D);
    float v[4];
    for (uint i = 0; i < 4; ++i)
        v[i] = held ? src[size_t(r) * stride + t * 4 + i] : 0.0f;
    const float inv = mtplx_rms_inverse(v, totals, 0, inverse, sg, lane, groups, uint(D), as_type<float>(uint(EPS)));
    const auto gain = s == 0 ? g0 : s == 1 ? g1 : g2;
    const auto to = s == 0 ? out0 : s == 1 ? out1 : out2;
    if (held) {
        for (uint i = 0; i < 4; ++i)
            to[size_t(r) * D + t * 4 + i] = static_cast<T>(static_cast<float>(gain[t * 4 + i]) * (v[i] * inv));
    }
"""


@lru_cache(maxsize=None)
def _kernel(which):
    spec = {
        "inputs": (["x", "msg"], ["mixed"], CONTROLLER_INPUTS, _RMS),
        "mix": (["forget", "gain", "x", "msg"], ["h"], CONTROLLER_MIX, ""),
        "norms": (["h0", "h1", "h2", "g0", "g1", "g2"], ["out0", "out1", "out2"], INPUT_NORMS, _RMS),
    }[which]
    return mx.fast.metal_kernel(name=f"mtplx_inkling_prompt_loop_{which}", input_names=spec[0], output_names=spec[1],
                                source=spec[2], header=spec[3])


def controller_inputs(x, source, message, index, eps, dtype=mx.bfloat16):
    """``dtype`` ``(rows, 2D)``: the controller's concatenated gain-free norms of ``x[:, source]`` and
    ``message[:, index]`` (``x`` ``(rows, NS, D)``, ``message`` ``(rows, NM, D)``, float32), cast as the
    down product's operand is."""
    rows, ns, width = x.shape
    (mixed,) = _kernel("inputs")(
        inputs=[x, message],
        template=[("M", dtype), ("D", width), ("NS", ns), ("SRC", source), ("NM", message.shape[1]),
                  ("MSG", index), ("EPS", _f32_bits(eps))],
        grid=(rows * _threads(width), 1, 1),
        threadgroup=(_threads(width), 1, 1),
        output_shapes=[(rows, 2 * width)],
        output_dtypes=[dtype],
    )
    return mixed


def controller_mix(forget, gain, x, source, message, index):
    """float32 ``(rows, D)``: ``(1 - tanh(forget)) * x[:, source] + tanh(gain) * message[:, index]``."""
    rows, ns, width = x.shape
    (h,) = _kernel("mix")(
        inputs=[forget, gain, x, message],
        template=[("D", width), ("NS", ns), ("SRC", source), ("NM", message.shape[1]), ("MSG", index)],
        grid=(rows * width, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[mx.float32],
    )
    return h


def input_norms(sources, gains, eps, dtype):
    """Each float32 source normed with its gain (``model.norm(h, gain, eps).astype(dtype)``), one launch for up
    to three: ``sources`` a list of ``(rows, D)`` arrays or one ``(rows, NS, D)`` array. Returns a list."""
    packed = isinstance(sources, mx.array)
    if packed:
        rows, ns, width = sources.shape
        hs = [sources] * 3
    else:
        rows, width = sources[0].shape
        ns = len(sources)
        hs = list(sources) + [sources[0]] * (3 - ns)
    gs = list(gains) + [gains[0]] * (3 - len(gains))
    outs = _kernel("norms")(
        inputs=[*hs, *gs],
        template=[("T", dtype), ("D", width), ("NS", ns), ("PACKED", packed), ("EPS", _f32_bits(eps))],
        grid=(rows * ns * _threads(width), 1, 1),
        threadgroup=(_threads(width), 1, 1),
        output_shapes=[(rows, width)] * ns + [(1,)] * (3 - ns),
        output_dtypes=[dtype] * 3,
    )
    return list(outs[:ns])


_STREAMS = (mx.bfloat16, mx.float16)


def served(rows, width, dtype, *arrays):
    """Why the loop glue cannot serve this call, or None."""
    import os

    if rows < 33:
        return "rows"
    if os.environ.get("MTPLX_INKLING_PROMPT_KERNELS", "ours").strip().lower() in ("mlx", "0", "off", "stock"):
        return "disabled"
    if not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        return "cpu"
    if dtype not in _STREAMS or width % 128 or width > 4096:
        return "stream"
    if any(a.dtype not in (*_STREAMS, mx.float32) for a in arrays):
        return "operand"
    return None


def _operand(weight):
    """The type the controller's concatenation is cast to before its down product (``Controller.__call__``)."""
    dtype = getattr(weight, "dtype", None)
    return dtype if dtype in _STREAMS else None


def controller(ctrl, x, source, message, index, eps):
    """``model.Controller.__call__(x[:, source], message[:, index], eps)`` as four launches (float32
    ``(rows, D)``), the products through ``model.shared_linear`` (our launch where admitted, else MLX's)."""
    from ..model import shared_linear

    mixed = controller_inputs(x, source, message, index, eps, _operand(ctrl.down))
    (state,) = shared_linear(mixed, [ctrl.down], "controller_down")
    forget, gain = shared_linear(state.astype(mx.float32), [ctrl.forget, ctrl.gain], "controller_mix")
    return controller_mix(forget, gain, x, source, message, index)


def _one_query_read(partial, closed, query, gain, eps, where):
    """The one-query depth read ``read_blocks(closed, partial, query, eps, True)`` as ``(rows, 1, D)`` float32:
    the read kernel's launch where it serves (``fast/read.py``), else MLX's chain (counted)."""
    from . import prompt_glue

    rows, width = partial.shape[1], partial.shape[-1]
    reason = prompt_glue._unserved(partial, closed, gain, where)
    if reason is None:
        from .read import read

        (out,) = read(partial, closed, query, eps, mode=1)
        _count(f"{where}.read.ours")
        return out
    from ..model import read_blocks

    _count(f"{where}.read.mlx.{reason}")
    return read_blocks(closed, partial, query, eps, True).reshape(rows, 1, width)


def visit_inputs(layer, x, closed, message):
    """Passes two and three: the visit's attention inputs ``q, k, v`` (``(1, W, D)`` in x's type), each source
    through its controller and its input norm (``model.Layer.__call__``'s statements, same bits), or None when
    this call is not served (counted). ``message``: the visit's ``(1, W, 4, D)`` float32 cross-pass message."""
    rows, width = x.shape[1], x.shape[-1]
    reason = served(rows, width, x.dtype, x, message, *closed)
    if reason is None and (len(layer.controllers) < 4 or layer.attn_query is None
                           or not layer.cfg["attn_res_normalize_values"]):
        reason = "layer"
    if reason is None and any(_operand(c.down) is None for c in layer.controllers[:3]):
        reason = "controller_operand"
    if reason is not None:
        _count(f"attention_sources.mlx.{reason}")
        return None
    ms = message.reshape(rows, message.shape[-2], width)
    if layer.sliding:
        xs = _one_query_read(x, closed, layer.attn_query, layer.qkv_norm[0], layer.eps, "attention_sources")
        picks, gains = (0, 0, 0), [layer.qkv_norm[0]] * 3
    else:
        # the three-query read stays MLX's products (``read_blocks``); its output feeds the controllers
        from ..model import read_blocks

        xs = read_blocks(closed, x, layer.attn_query, layer.eps, True).reshape(rows, 3, width)
        picks, gains = (0, 1, 2), list(layer.qkv_norm)
    hs = [controller(layer.controllers[i], xs, picks[i], ms, i, layer.eps) for i in range(3)]
    outs = input_norms(hs, gains, layer.eps, x.dtype)
    _count("attention_sources.ours." + ("window" if layer.sliding else "global"))
    return [o.reshape(1, rows, width) for o in outs]


def expert_input(layer, residual, closed, message, dtype):
    """Passes two and three: the expert block's input (``(1, W, D)`` in ``dtype``): the MLP read, the MLP
    controller and the norm (``model.Layer.__call__``'s statements, same bits), or None (counted)."""
    rows, width = residual.shape[1], residual.shape[-1]
    reason = served(rows, width, dtype, residual, message, *closed)
    if reason is None and (len(layer.controllers) < 4 or not layer.cfg["attn_res_normalize_values"]):
        reason = "layer"
    if reason is None and _operand(layer.controllers[3].down) is None:
        reason = "controller_operand"
    if reason is not None:
        _count(f"expert_source.mlx.{reason}")
        return None
    ms = message.reshape(rows, message.shape[-2], width)
    xs = _one_query_read(residual, closed, layer.mlp_query, layer.ffn_norm, layer.eps, "expert_source")
    h = controller(layer.controllers[3], xs, 0, ms, 3, layer.eps)
    (out,) = input_norms([h], [layer.ffn_norm], layer.eps, dtype)
    _count("expert_source.ours")
    return out.reshape(1, rows, width)


def global_norms(layer, read, dtype):
    """A global visit's first-pass input norms (``norm(read[..., i, :], qkv_norm[i]).astype(dtype)``) as one
    launch from its ``(1, W, 3, D)`` float32 three-query read, or None (counted)."""
    rows, width = read.shape[1], read.shape[-1]
    reason = served(rows, width, dtype, read, *layer.qkv_norm)
    if reason is None and (read.ndim != 4 or read.shape[2] != 3 or len(layer.qkv_norm) != 3):
        reason = "layer"
    if reason is not None:
        _count(f"global_norms.mlx.{reason}")
        return None
    outs = input_norms(read.reshape(rows, 3, width), list(layer.qkv_norm), layer.eps, dtype)
    _count("global_norms.ours")
    return [o.reshape(1, rows, width) for o in outs]
