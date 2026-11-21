"""The prompt read's glue between the expert blocks and attention, as our launches.

A prompt chunk's layer visit reads its input from the closed residual blocks twice (once with the
attention query, once with the MLP query) and norms what it reads, and the chunk ends with one more
read and the final norm. The prompt path writes each read as MLX operations: a stack of the blocks, a
cast, a norm of every block row, the scores of the query against them, a scale, a softmax, the mix,
and then the input norm and a cast: about twelve MLX launches a read. ``input_norm`` runs it as one
launch of the depth-read kernel the decode step already uses (``fast/read.py``): one threadgroup a row,
every step of it the same operation in the same order as MLX's kernel for that step, so its output
has the prompt path's bits at every chunk width (checked against the MLX chain at 33 to 4,096 rows,
one to four closed blocks, a bfloat16 or float32 partial).

Served: chunks of more than 32 rows (decode and verify panels keep their own routes), one query and
no cross-pass message (the first pass of the loop, and the visits outside it), up to four closed
blocks, a bfloat16 stream (the FP16 sibling packs keep MLX's chain: the kernel rounds to bfloat16),
and ``MTPLX_INKLING_PROMPT_KERNELS`` not ``mlx``. Everything else keeps MLX's chain, counted.
"""

from __future__ import annotations

from collections import Counter
import os
from threading import Lock

import mlx.core as mx

PROMPT_MIN_ROWS = 33
_lock = Lock()
_counts: Counter = Counter()


def counts():
    with _lock:
        return dict(_counts)


def _count(key):
    with _lock:
        _counts[key] += 1


def enabled():
    return os.environ.get("MTPLX_INKLING_PROMPT_KERNELS", "ours").strip().lower() not in ("mlx", "0", "off", "stock")


def min_rows():
    """Rows a call needs before these launches serve it: every row count under the prompt read's width-independent
    part (each row's result is its own, whatever the rows around it), else prompt chunks of more than 32 rows."""
    from .prompt_route import invariant_sums

    return 1 if invariant_sums() else PROMPT_MIN_ROWS


def _unserved(partial, closed, gain, where):
    if partial.ndim != 3 or partial.shape[1] < min_rows():
        return "rows"
    if not enabled():
        return "disabled"
    if not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        return "cpu"
    if len(closed) > 4:
        return "closed_blocks"
    width = partial.shape[-1]
    if width % 128 or width > 4096:
        return "width"
    if gain.dtype != mx.bfloat16 or partial.dtype not in (mx.bfloat16, mx.float32):
        return "stream_dtype"
    if any(c.dtype != mx.bfloat16 for c in closed):
        return "stream_dtype"
    return None


def input_norm(partial, closed, query, gain, eps, where, stand_in=None):
    """``norm(read_blocks(closed, partial, query, eps, True)[..., 0, :], gain, eps)`` in bfloat16 as one
    launch, or None when this call is not served (counted under ``where``). ``query`` None: no depth
    read (the first layer), the partial itself is normed; ``stand_in`` is then any float32 array."""
    reason = _unserved(partial, closed, gain, where)
    if reason is None and query is None and stand_in is None:
        reason = "stand_in"
    if reason is not None:
        _count(f"{where}.mlx.{reason}")
        return None
    from .read import read

    (out,) = read(partial, closed, query, eps, mode=0, gains=[gain], f32_stand_in=stand_in)
    _count(f"{where}.ours")
    return out.reshape(partial.shape)


# --- the visit's head step: query, relative and key clips, the value norm, the position scale ----------

HEADS = """
    // T: the stream's type; QS, RS: the clip scales' types; QH query heads, KH key heads, HD head width, RD
    // relative width; GLOBAL: the position scale applies; VNORM: the values are normed. One simdgroup a row
    // and head slot (query heads, then key heads, then value heads); lane l owns columns 4l .. 4l + 3.
    const uint lane = thread_index_in_simdgroup;
    const uint per_group = threads_per_threadgroup.x / 32;
    const uint unit = threadgroup_position_in_grid.x * per_group + simdgroup_index_in_threadgroup;
    const uint slots = uint(QH + 2 * KH);
    const uint rows = uint(qlin_shape[0]);
    if (unit >= rows * slots)
        return;
    const uint r = unit / slots;
    const uint slot = unit % slots;
    float tau = 1.0f;
    if (GLOBAL) {
        // the prompt path's arange(rows) + offset + 1 in int32, over the floor, at least 1, its log
        const float place = static_cast<float>(int(offset[0]) + int(r) + 1);
        const float over = place / as_type<float>(uint(LOG_FLOOR));
        const float kept = metal::isnan(over) ? over : (over > 1.0f ? over : 1.0f);
        tau = 1.0f + as_type<float>(uint(LOG_ALPHA)) * metal::precise::log(kept);
    }
    if (slot < uint(QH + KH)) {
        // A clipped head: (scaled) values, their squares summed four a lane then across the simdgroup (the
        // library's row reduction over 128), precise square root, the limit over it, at most one.
        const bool query = slot < uint(QH);
        const uint h = query ? slot : slot - uint(QH);
        float v4[4];
        float part = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            const uint d = lane * 4 + j;
            v4[j] = query ? static_cast<float>(qlin[(size_t(r) * QH + h) * HD + d]) * static_cast<float>(qscale[h * HD + d])
                          : static_cast<float>(kc[(size_t(r) * KH + h) * HD + d]);
            const float squared = v4[j] * v4[j];
            part = squared + part;
        }
        const float length = metal::precise::sqrt(simd_sum(part));
        const float ratio = as_type<float>(uint(QK_LIMIT)) / (length + as_type<float>(uint(CLIP_EPS)));
        const float factor = (1.0f < ratio) ? 1.0f : ratio;
        for (uint j = 0; j < 4; ++j) {
            const uint d = lane * 4 + j;
            T y = static_cast<T>(v4[j] * factor);
            if (query) {
                if (GLOBAL)
                    y = static_cast<T>(static_cast<float>(y) * tau);
                q[(size_t(r) * QH + h) * HD + d] = y;
            } else {
                k[(size_t(r) * KH + h) * HD + d] = y;
            }
        }
        if (query && lane == 0) {
            // The relative head (RD wide): one thread sums its squares in order (the library's small row).
            float rv[RD];
            float total = 0.0f;
            for (uint j = 0; j < uint(RD); ++j) {
                rv[j] = static_cast<float>(rlin[(size_t(r) * QH + h) * RD + j]) * static_cast<float>(rscale[h * RD + j]);
                const float squared = rv[j] * rv[j];
                total = squared + total;
            }
            const float rlength = metal::precise::sqrt(total);
            const float rratio = as_type<float>(uint(REL_LIMIT)) / (rlength + as_type<float>(uint(CLIP_EPS)));
            const float rfactor = (1.0f < rratio) ? 1.0f : rratio;
            for (uint j = 0; j < uint(RD); ++j) {
                T y = static_cast<T>(rv[j] * rfactor);
                if (GLOBAL)
                    y = static_cast<T>(static_cast<float>(y) * tau);
                rel[(size_t(r) * QH + h) * RD + j] = y;
            }
        }
    } else {
        // A value head: its root mean square over 128 (four a lane fused-added from zero, then the simdgroup),
        // precise reciprocal square root, or the values as they are.
        const uint h = slot - uint(QH + KH);
        float v4[4];
        float acc = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            v4[j] = static_cast<float>(vc[(size_t(r) * KH + h) * HD + lane * 4 + j]);
            acc += v4[j] * v4[j];
        }
        const float inverse = VNORM
            ? metal::precise::rsqrt(simd_sum(acc) / float(HD) + as_type<float>(uint(NORM_EPS)))
            : 1.0f;
        for (uint j = 0; j < 4; ++j)
            v[(size_t(r) * KH + h) * HD + lane * 4 + j] = VNORM ? static_cast<T>(v4[j] * inverse) : static_cast<T>(v4[j]);
    }
"""


def _f32_bits(value):
    import struct

    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


_heads_kernel_cache = {}


def _heads_kernel():
    kernel = _heads_kernel_cache.get("k")
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mtplx_inkling_prompt_heads",
            input_names=["qlin", "rlin", "kc", "vc", "qscale", "rscale", "offset"],
            output_names=["q", "rel", "k", "v"],
            source=HEADS,
        )
        _heads_kernel_cache["k"] = kernel
    return kernel


def heads(layer, q_lin, r_lin, k_conv, v_conv, offset):
    """The visit's clips, value norm and position scale (``model.Layer.attention`` after its projections
    and convolutions) as one launch, or None when this call is not served (counted). Returns ``q``
    ``(1, W, QH, HD)``, ``rel`` ``(1, W, QH, RD)``, ``k`` and ``v`` ``(1, W, KH, HD)``."""
    cfg = layer.cfg
    rows = q_lin.shape[1]
    q_heads, head_dim = layer.q_scale.shape
    rel_dim = layer.r_scale.shape[1]
    kv_heads = k_conv.shape[-1] // head_dim
    reason = None
    if rows < min_rows():
        reason = "rows"
    elif not enabled():
        reason = "disabled"
    elif not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        reason = "cpu"
    elif q_lin.dtype != mx.bfloat16 or k_conv.dtype != mx.bfloat16 or v_conv.dtype != mx.bfloat16 or r_lin.dtype != mx.bfloat16:
        reason = "stream_dtype"
    elif head_dim != 128 or q_lin.shape[-1] != q_heads * head_dim or r_lin.shape[-1] != q_heads * rel_dim:
        reason = "geometry"
    if reason is not None:
        _count(f"heads.mlx.{reason}")
        return None
    glob = not layer.sliding
    template = [
        ("T", q_lin.dtype), ("QH", q_heads), ("KH", kv_heads), ("HD", head_dim), ("RD", rel_dim),
        ("GLOBAL", glob), ("VNORM", bool(cfg["use_value_norm_no_gamma"])),
        ("QK_LIMIT", _f32_bits(cfg["qk_l2_clip_threshold"])), ("REL_LIMIT", _f32_bits(cfg["rel_l2_clip_threshold"])),
        ("CLIP_EPS", _f32_bits(1e-6)), ("NORM_EPS", _f32_bits(layer.eps)),
        ("LOG_FLOOR", _f32_bits(cfg["log_scaling_n_floor"])), ("LOG_ALPHA", _f32_bits(cfg["log_scaling_alpha"])),
    ]
    slots = q_heads + 2 * kv_heads
    per_group = 8
    groups = -(-rows * slots // per_group)
    q, rel, k, v = _heads_kernel()(
        inputs=[q_lin.reshape(rows, -1), r_lin.reshape(rows, -1), k_conv.reshape(rows, -1), v_conv.reshape(rows, -1),
                layer.q_scale, layer.r_scale, mx.array([int(offset)], dtype=mx.int32)],
        template=template,
        grid=(groups * per_group * 32, 1, 1),
        threadgroup=(per_group * 32, 1, 1),
        output_shapes=[(rows, q_heads, head_dim), (rows, q_heads, rel_dim), (rows, kv_heads, head_dim),
                       (rows, kv_heads, head_dim)],
        output_dtypes=[q_lin.dtype] * 4,
    )
    _count("heads.ours." + ("global" if glob else "window"))
    return (q.reshape(1, rows, q_heads, head_dim), rel.reshape(1, rows, q_heads, rel_dim),
            k.reshape(1, rows, kv_heads, head_dim), v.reshape(1, rows, kv_heads, head_dim))


# --- the output gate ---------------------------------------------------------------------------------

GATE = """
    // x[r, c] = T(float(h[0, head, r, d]) * sigmoid(float(g[0, r, c]))), c = head * HD + d: Layer.attention's
    // `(h.astype(float32) * sigmoid(gate_lin.astype(float32))).astype(h.dtype)` with h read where attend left it
    // (heads first, at its own strides), so neither the transpose's copy nor the casts are written.
    const uint idx = thread_position_in_grid.x;
    const uint rows = uint(g_shape[1]);
    const uint width = uint(QH * HD);
    if (idx >= rows * width)
        return;
    const uint r = idx / width;
    const uint c = idx - r * width;
    const uint head = c / uint(HD);
    const uint d = c - head * uint(HD);
    const float z = float(g[size_t(r) * g_strides[1] + size_t(c) * g_strides[2]]);
    const float y = 1 / (1 + metal::precise::exp(metal::abs(z)));
    const float open = (z < 0) ? y : 1 - y;
    const float value = float(h[size_t(head) * h_strides[1] + size_t(r) * h_strides[2] + size_t(d) * h_strides[3]]);
    x[idx] = static_cast<T>(value * open);
"""

_gate_kernel_cache = {}


def _gate_kernel():
    kernel = _gate_kernel_cache.get("k")
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mtplx_inkling_prompt_output_gate",
            input_names=["h", "g"],
            output_names=["x"],
            source=GATE,
            ensure_row_contiguous=False,
        )
        _gate_kernel_cache["k"] = kernel
    return kernel


def output_gate(h, gate_lin):
    """``(h.transpose(0, 2, 1, 3).reshape(1, rows, -1).astype(float32) * sigmoid(gate_lin.astype(float32)))
    .astype(h.dtype)`` as one launch, ``h`` as attend returns it ``(1, QH, rows, HD)``; or None when this call is
    not served (counted)."""
    reason = None
    if h.ndim != 4 or gate_lin.ndim != 3 or h.shape[0] != 1 or gate_lin.shape[0] != 1:
        reason = "shape"
    elif h.shape[2] < min_rows():
        reason = "rows"
    elif not enabled():
        reason = "disabled"
    elif not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        reason = "cpu"
    elif h.dtype not in (mx.bfloat16, mx.float16) or gate_lin.dtype != h.dtype:
        reason = "stream_dtype"
    elif gate_lin.shape[1] != h.shape[2] or gate_lin.shape[2] != h.shape[1] * h.shape[3]:
        reason = "geometry"
    if reason is not None:
        _count(f"gate.mlx.{reason}")
        return None
    _, heads_, rows, head_dim = h.shape
    total = rows * heads_ * head_dim
    (x,) = _gate_kernel()(
        inputs=[h, gate_lin],
        template=[("T", h.dtype), ("QH", heads_), ("HD", head_dim)],
        grid=(-(-total // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, rows, heads_ * head_dim)],
        output_dtypes=[h.dtype],
    )
    _count("gate.ours")
    return x
