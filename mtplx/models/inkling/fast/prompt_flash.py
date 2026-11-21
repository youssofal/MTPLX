"""Inkling prompt attention in one launch per visit, on the GPU's tensor units.

What one prompt call computes, for query row r of the chunk (absolute position p), query head h:
the logits ``q.k_c / sqrt(128) + bias(p - c)`` over the keys c the row may see (``c <= p``, and
on a window visit ``p - c < 1024``), where ``bias(d)`` is the row's profile entry d for d below
the profile's extent and 0 beyond it, plus one more logit, the sink ``s[r, h]``, whose value is
zero; the output is the softmax of those logits applied to the values.

The default prompt route builds that from the library's attention: a bias tensor written per
call, the bank copied with a zero row appended for the sink, every key of the call computed
whether a row may see it or not, and past about 19,400 keys a decode-shaped kernel that has no
matrix units. Here a launch computes it for a whole visit:

* Blocks. A simdgroup holds 16 consecutive query rows of one head; a threadgroup holds the
  three heads that share a key/value head (times ``TG_M / 3`` row blocks), so every key and
  value piece the threadgroup loads serves all three. All simdgroups of a threadgroup walk the
  same 32-key tiles, held together by a barrier each tile so they share them in the cache.
* Products. The tensor-unit multiply and the register layout of its 16 x 16 pieces are the
  installed MLX's own (``mlx::steel::BaseNAXFrag``: a 16 x 32 x 16 multiply-accumulate with
  float sums), read from the headers MLX ships when the launch is first built
  (``mtplx.kernels.library_headers``): none of the library's text lives in this tree. Scores
  over the head in eight 16-wide steps; the value product over each tile in two 16-key steps,
  the probabilities entering as float. Scores, the running maximum and sum, and the output
  stay in float.
* The softmax is online and opened by the sink: before the first key a row's maximum is its
  sink logit, its sum 1 and its output 0, which is the sink's own term (value zero).
* Tiles. A threadgroup visits only the tiles one of its rows may see: from the first key of
  its earliest row's window (or key 0) to its latest row's own position. Tiles at a distance of
  at least the profile's extent from every row of the threadgroup need no bias and no mask
  (on a global visit at long context, nearly all of them); the rest are masked per element and
  read their bias from the profile.
* Keys, values, queries, profiles and sinks are read in place through their strides: nothing
  is concatenated, no bias or mask tensor exists, the launch writes only its output
  ``(1, rows, heads, 128)`` in the query type.

Class C against the default prompt route: the same sums in another order, all in float. Against
the exact value (float64), per call, its outputs are as often correctly rounded as the default
route's where that uses the library's attention (below about 19,400 keys) and far more often past
it, where the default route rounds per-block partial outputs (lanes/p2/acc-1). Off by default
(``MTPLX_INKLING_PROMPT_FLASH``); :func:`decline` names why a call is not served.
"""

from __future__ import annotations

import os
import struct
from functools import lru_cache

import mlx.core as mx

from mtplx.kernels import library_headers

HEAD_DIM = 128
#: Simdgroups a threadgroup: the heads of one key/value head times row blocks of 16 (3 for
#: Inkling: one block of each head). Measured fastest at 512-row chunks and level at 2,048
#: (lanes/p2/fsweep-2).
TG_M = 3

# Ours, after the library's text: four consecutive elements of one row of a piece. ``step`` 0:
# one 8-byte load (unit stride along the head and every other stride a multiple of four elements,
# so every address formed is a multiple of four elements from the view's first element, which the
# prompt read's views keep at the start of their array); otherwise element by element at ``step``.
# And the store of four into the launch's own output.
_HEADER = """
// Every loop over pieces, rows or columns is unrolled whole (none runs more than 8 times): the
// pieces are 8-wide register vectors and are only indexed with constants; a loop left rolled
// indexes them at run time and gave wrong numbers (lanes/p2 diag-1). The first-use check against
// the definition declines the launch if a build ever does that.
#define INK_EVERY _Pragma("clang loop unroll_count(8)")

template <typename E>
inline vec<E, 4> ink_four(const device E* p, uint step) {
    if (step == 0u) {
        return *reinterpret_cast<const device vec<E, 4>*>(p);
    }
    return vec<E, 4>(p[0], p[step], p[2u * step], p[3u * step]);
}

template <typename E>
inline void ink_put_four(device E* p, vec<E, 4> x) {
    *reinterpret_cast<device vec<E, 4>*>(p) = x;
}
"""

# A lane holds two rows of each 16 x 16 piece (fr and fr + 8) and four consecutive columns of
# each (fc .. fc + 3), as the library places them: element 4 rp + c is row fr + 8 rp, column fc + c.
# INK_FENCE_NOTE: with every loop unrolled the compiler issues all of a tile's key loads (and all of
# its value loads) before the first multiply: 64 registers of loads live at once. A simdgroup fence
# halfway through each loop holds the second half's loads back; the numbers are unchanged and a tile
# runs 1.4 to 1.6x faster at 16K to 128K keys (lanes/p2/fexp3-1.json).
_SOURCE = """
    using Piece = mlx::steel::BaseNAXFrag;
    typedef vec<T, 8> tpiece;
    typedef vec<float, 8> fpiece;
    constexpr int D = 128;
    constexpr int DP = D / 16;                                  // pieces along a head
    constexpr int TG_ROWS = (TG_M / G) * 16;
    const float scale_log2 = as_type<float>(uint(SCALE)) * 1.4426950408889634f;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint tid = thread_position_in_threadgroup.x;
    const int sg = int(tid >> 5);
    const short2 here = Piece::get_coord();
    const short fr = here.y;
    const short fc = here.x;
    const int kvh = int(threadgroup_position_in_grid.y);
    const int rows = int(params[0]);
    const int total = int(params[1]);
    const int hist = total - rows;                              // position of chunk row 0
    const int tg_row0 = int(threadgroup_position_in_grid.x) * TG_ROWS;
    const int tg_row_last = min(tg_row0 + TG_ROWS, rows) - 1;
    const int blk_row0 = tg_row0 + (sg / G) * 16;
    const int head = kvh * G + sg % G;

    // The lane's two rows: query row, position, whether it is a real row.
    int qrow[2];
    int pos[2];
    bool live[2];
    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
        qrow[rp] = blk_row0 + int(fr) + 8 * rp;
        live[rp] = qrow[rp] < rows;
        pos[rp] = hist + qrow[rp];
    }

    // Element strides along a head, and whether rows are packed for 8-byte loads (step 0).
    const uint q_dim = uint(q_strides[3]);
    const uint k_dim = uint(k_strides[3]);
    const uint v_dim = uint(v_strides[3]);
    const uint q_step = (q_dim == 1u && q_strides[2] % 4 == 0 && q_strides[1] % 4 == 0) ? 0u : q_dim;
    const uint k_step = (k_dim == 1u && k_strides[2] % 4 == 0 && k_strides[1] % 4 == 0) ? 0u : k_dim;
    const uint v_step = (v_dim == 1u && v_strides[2] % 4 == 0 && v_strides[1] % 4 == 0) ? 0u : v_dim;

    // The block's queries over the whole head, held for the walk.
    tpiece qp[DP];
    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
        const device T* src = q + (size_t(head) * size_t(q_strides[1]) +
                                   size_t(live[rp] ? qrow[rp] : 0) * size_t(q_strides[2]) + size_t(fc) * size_t(q_dim));
        INK_EVERY for (short p = 0; p < DP; ++p) {
            const vec<T, 4> x = ink_four(src + uint(16 * p) * q_dim, q_step);
            INK_EVERY for (short col = 0; col < 4; ++col) {
                qp[p][rp * 4 + col] = live[rp] ? x[col] : T(0);
            }
        }
    }
    // The output, zero before the first key: eight pieces of 16 dimensions.
    static_assert(DP == 8, "a head of 128 is eight pieces");
    const fpiece zero8 = fpiece(0.0f);
    fpiece acc[DP] = {zero8, zero8, zero8, zero8, zero8, zero8, zero8, zero8};

    // The sink opens each row: maximum its logit, sum 1, output 0.
    float run_max[2];
    float run_sum[2] = {1.0f, 1.0f};
    const device T* prof_row[2];
    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
        const int r = live[rp] ? qrow[rp] : 0;
        run_max[rp] = live[rp]
            ? float(sink[size_t(r) * size_t(sink_strides[1]) + size_t(head) * size_t(sink_strides[2])]) * LOG2E
            : 0.0f;
        prof_row[rp] = prof + (size_t(r) * size_t(prof_strides[1]) + size_t(head) * size_t(prof_strides[2]));
    }
    const uint prof_step = uint(prof_strides[3]);

    // Key and value rows of this lane in a tile: base + 16 j + fr + 8 rp. One 64-bit base a
    // lane, 32-bit element offsets from it.
    const device T* k_lane = k + (size_t(kvh) * size_t(k_strides[1]) + size_t(fc) * size_t(k_dim));
    const device T* v_lane = v + (size_t(kvh) * size_t(v_strides[1]) + size_t(fc) * size_t(v_dim));
    const uint k_row = uint(k_strides[2]);
    const uint v_row = uint(v_strides[2]);

    // Tiles some row of this threadgroup may see; tiles before t_near need no bias and no mask.
    const int pos_lo = hist + tg_row0;
    const int pos_hi = hist + tg_row_last;
    const int t_end = pos_hi / 32 + 1;
    int t_begin = 0;
    int t_near = 0;
    if (WINDOW > 0) {
        t_begin = max(0, pos_lo - (WINDOW - 1)) / 32;
        t_near = t_begin;
    } else {
        const int clear = pos_lo - EXTENT + 1;          // keys below this are far from every row
        t_near = clear > 0 ? min(clear / 32, t_end) : 0;
    }

    for (int t = t_begin; t < t_end; ++t) {
        if (TGBAR) {
            threadgroup_barrier(mem_flags::mem_none);      // keep the blocks on one tile: shared in L1
        }
        const int base = t * 32;
        const bool near = t >= t_near;
        uint koff[2][2];
        uint voff[2][2];
        INK_EVERY for (short j = 0; j < 2; ++j) {
            INK_EVERY for (short rp = 0; rp < 2; ++rp) {
                int key = base + 16 * j + int(fr) + 8 * rp;
                key = key < total ? key : 0;               // past the bank: masked below
                koff[j][rp] = uint(key) * k_row;
                voff[j][rp] = uint(key) * v_row;
            }
        }

        // Scores of the tile: keys base .. base + 15 in s[0], base + 16 .. base + 31 in s[1].
        fpiece s[2];
        s[0] = fpiece(0.0f);
        s[1] = fpiece(0.0f);
        INK_EVERY for (short p = 0; p < DP; ++p) {
            tpiece kp[2];
            INK_EVERY for (short j = 0; j < 2; ++j) {
                INK_EVERY for (short rp = 0; rp < 2; ++rp) {
                    const vec<T, 4> x = ink_four(k_lane + (koff[j][rp] + uint(16 * p) * k_dim), k_step);
                    INK_EVERY for (short col = 0; col < 4; ++col) {
                        kp[j][rp * 4 + col] = x[col];
                    }
                }
            }
            Piece::mma(s[0], s[1], qp[p], metal::false_type{}, kp[0], kp[1], metal::true_type{});
            if (p == DP / 2 - 1) {
                simdgroup_barrier(mem_flags::mem_none);     // see INK_FENCE_NOTE
            }
        }

        // Logits in base-2 units, masked and biased where a row is near.
        INK_EVERY for (short j = 0; j < 2; ++j) {
            INK_EVERY for (short e = 0; e < 8; ++e) {
                const short rp = e >> 2;
                float x = s[j][e] * scale_log2;
                if (near) {
                    const int key = base + 16 * j + int(fc) + (e & 3);
                    const int d = pos[rp] - key;
                    const bool seen = live[rp] && d >= 0 && (WINDOW == 0 || d < WINDOW);
                    if (!seen) {
                        x = -INFINITY;
                    } else if (d < EXTENT) {
                        x = fma(float(prof_row[rp][uint(d) * prof_step]), LOG2E, x);
                    }
                }
                s[j][e] = x;
            }
        }

        // Online softmax per row: the tile's maximum, the correction of what came before, the weights.
        float corr[2];
        INK_EVERY for (short rp = 0; rp < 2; ++rp) {
            float m = -INFINITY;
            INK_EVERY for (short j = 0; j < 2; ++j) {
                INK_EVERY for (short col = 0; col < 4; ++col) {
                    m = max(m, s[j][rp * 4 + col]);
                }
            }
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            const float nm = max(run_max[rp], m);
            corr[rp] = fast::exp2(run_max[rp] - nm);
            float sum = 0.0f;
            INK_EVERY for (short j = 0; j < 2; ++j) {
                INK_EVERY for (short col = 0; col < 4; ++col) {
                    const float w = fast::exp2(s[j][rp * 4 + col] - nm);
                    s[j][rp * 4 + col] = w;
                    sum += w;
                }
            }
            sum += simd_shuffle_xor(sum, ushort(1));
            sum += simd_shuffle_xor(sum, ushort(8));
            run_sum[rp] = run_sum[rp] * corr[rp] + sum;
            run_max[rp] = nm;
        }
        if (simd_any(corr[0] != 1.0f || corr[1] != 1.0f)) {
            INK_EVERY for (short p = 0; p < DP; ++p) {
                INK_EVERY for (short e = 0; e < 8; ++e) {
                    acc[p][e] *= corr[e >> 2];
                }
            }
        }

        // Value sums: each 16-key half of the tile against the values of those keys, 32
        // output dimensions a multiply.
        INK_EVERY for (short j = 0; j < 2; ++j) {
            INK_EVERY for (short g = 0; g < DP / 2; ++g) {
                if (g == DP / 4) {
                    simdgroup_barrier(mem_flags::mem_none);     // see INK_FENCE_NOTE
                }
                tpiece vp[2];
                INK_EVERY for (short h = 0; h < 2; ++h) {
                    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
                        const vec<T, 4> x = ink_four(v_lane + (voff[j][rp] + uint(32 * g + 16 * h) * v_dim), v_step);
                        INK_EVERY for (short col = 0; col < 4; ++col) {
                            vp[h][rp * 4 + col] = x[col];
                        }
                    }
                }
                Piece::mma(acc[2 * g], acc[2 * g + 1], s[j], metal::false_type{}, vp[0], vp[1], metal::false_type{});
            }
        }
    }

    // Store: piece p, element 4 rp + c is dimension 16 p + fc + c of row fr + 8 rp.
    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
        if (!live[rp]) {
            continue;
        }
        const float inv = 1.0f / run_sum[rp];
        device T* o = out + (size_t(qrow[rp]) * size_t(HQ) + size_t(head)) * size_t(D) + size_t(fc);
        INK_GATE_ROW
        INK_EVERY for (short p = 0; p < DP; ++p) {
            vec<T, 4> y;
            INK_EVERY for (short col = 0; col < 4; ++col) {
                INK_RESULT(y[col], acc[p][rp * 4 + col] * inv, 16 * p + col)
            }
            ink_put_four(o + 16 * p, y);
        }
    }
"""

# The result of one element: the attention in the query type; gated, then times the sigmoid of
# the row's gate logit as the layer applies it (``h.astype(float32) * sigmoid(gate.astype(float32))``,
# rounded to the query type), the sigmoid in its stable form, 1 / (1 + e^|x|) mirrored at 0.
_RESULT = "#define INK_RESULT(Y_, V_, DIM_) Y_ = T(V_);\n"
_RESULT_GATED = """
#define INK_RESULT(Y_, V_, DIM_)                                                         \\
    {                                                                                    \\
        const T hv = T(V_);                                                              \\
        const float gx = float(gp[DIM_]);                                                \\
        const float tail = 1.0f / (1.0f + metal::exp(metal::abs(gx)));                   \\
        Y_ = T(float(hv) * (gx < 0.0f ? tail : 1.0f - tail));                            \\
    }
"""


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


SCALE_BITS = _float_bits(HEAD_DIM ** -0.5)


def library_text():
    """``(text, None)`` with the installed MLX's tensor-unit pieces, or ``(None, why)``."""
    return library_headers.text("tensor_tiles")


def runs_here():
    """Whether the stage can read prompt chunks on this Mac and settings at all, before a prompt's length and the
    memory decide each chunk: switched on, the stock attention mode, not forced to the stock prompt attention, a
    GPU with the tensor units, the installed library's headers. The read's width and its memory pricing
    (``prompt_route.flash_attention_on``) assume the stage only where this holds: a Mac without the tensor units
    (M1 to M4) reads on the stock attention, whose planes grow with width times context."""
    from .. import attention as A

    if not A.flash_switch_on():
        return False
    if os.environ.get("MTPLX_INKLING_ATTENTION", "stock") != "stock":
        return False
    if os.environ.get("MTPLX_INKLING_PROMPT_ATTENTION", "fused") == "stock":
        return False
    if mx.default_device() != mx.gpu or not mx.metal.is_available() or not tensor_units():
        return False
    return library_text()[0] is not None


@lru_cache(maxsize=None)
def _kernel(gated=False):
    text, why = library_text()
    if text is None:
        raise RuntimeError(f"library headers: {why}")
    # The gate is the gate product's own output, row-contiguous: a row is HQ * D logits.
    gate_row = ("#define INK_GATE_ROW const device T* gp = gate + (size_t(qrow[rp]) * size_t(HQ) + "
                "size_t(head)) * size_t(D) + size_t(fc);\n") if gated else "#define INK_GATE_ROW\n"
    return mx.fast.metal_kernel(
        name="mtplx_inkling_prompt_flash" + ("_gated" if gated else ""),
        input_names=["q", "k", "v", "prof", "sink", "params"] + (["gate"] if gated else []),
        output_names=["out"],
        header=text + _HEADER + (_RESULT_GATED if gated else _RESULT) + gate_row,
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


def tensor_units():
    """Whether this Mac has the tensor units (and macOS) the launch is written for. Read per call, like
    ``nax_available`` (its hardware probe is cached; the M1 to M4 rehearsal switch is not, so a profile or a test
    that sets it after import is obeyed)."""
    from mtplx.nax_detect import nax_available

    return bool(nax_available())


def settings():
    """The launch's measured choices, each overridable for measurement: ``(tg_m, tgbar)``."""
    env = os.environ.get
    tg_m = int(env("MTPLX_INKLING_PROMPT_FLASH_TGM", TG_M))
    tgbar = env("MTPLX_INKLING_PROMPT_FLASH_TGBAR", "1") not in ("0", "off", "false")
    return tg_m, tgbar


def decline(query, key, value, profiles, sink, window):
    """None when one launch can serve this prompt call, else a short reason."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if not tensor_units():
        return "no_tensor_units"
    if query.dtype not in (mx.bfloat16, mx.float16) or any(
            x.dtype != query.dtype for x in (key, value, profiles, sink)):
        return "dtype"
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or profiles.ndim != 4 or sink.ndim != 3:
        return "geometry"
    batch, heads, rows, width = query.shape
    kv_heads, keys = key.shape[1], key.shape[2]
    if (batch != 1 or width != HEAD_DIM or key.shape[0] != 1 or key.shape[3] != HEAD_DIM
            or tuple(value.shape) != tuple(key.shape) or not kv_heads or heads % kv_heads
            or not 0 < rows <= keys or profiles.shape[:3] != (1, rows, heads)
            or tuple(sink.shape) != (1, rows, heads) or window < 0):
        return "geometry"
    if heads // kv_heads > 16 or TG_M % (heads // kv_heads):
        return "geometry"
    if library_text()[0] is None:
        return "library_headers"
    return None


def flash_attention(query, key, value, profiles, sink, window, *, tg_m=None, tgbar=None, gate=None):
    """The prompt call's attention, ``(1, heads, rows, 128)`` (a view of a ``(1, rows, heads,
    128)`` array), in one launch. With ``gate`` (the gate projection's ``(1, rows, heads * 128)``
    logits) it returns the gated result ``(1, rows, heads * 128)`` the output projection reads.
    Call only where :func:`decline` returned None."""
    d = settings()
    tg_m = d[0] if tg_m is None else tg_m
    tgbar = d[1] if tgbar is None else tgbar
    if gate is not None:
        gate = mx.contiguous(gate)
    _, heads, rows, _ = query.shape
    kv_heads, keys = key.shape[1], key.shape[2]
    group = heads // kv_heads
    if tg_m % group:
        raise ValueError("a threadgroup holds whole sets of the heads a key/value head serves")
    tg_rows = (tg_m // group) * 16
    (out,) = _kernel(gate is not None)(
        inputs=[query, key, value, profiles, sink, mx.array([rows, keys], mx.int32)] + ([gate] if gate is not None else []),
        template=[("T", query.dtype), ("G", group), ("TG_M", tg_m), ("HQ", heads), ("WINDOW", int(window)),
                  ("EXTENT", int(profiles.shape[3])), ("SCALE", SCALE_BITS), ("TGBAR", bool(tgbar))],
        grid=(-(-rows // tg_rows) * tg_m * 32, kv_heads, 1),
        threadgroup=(tg_m * 32, 1, 1),
        output_shapes=[(1, rows, heads, HEAD_DIM)],
        output_dtypes=[query.dtype],
    )
    if gate is not None:
        return out.reshape(1, rows, heads * HEAD_DIM)
    return out.transpose(0, 2, 1, 3)


# ----------------------------------------------------------------------------------------
# The caches a prompt chunk writes.
#
# A global visit's bank grows by the chunk's rows every chunk: the library's cache copies the
# whole bank into a larger one each time (a 512-row chunk at history H copies 2 x H rows a visit;
# over a 1M-token read that is tens of terabytes moved). The prompt's length is known before the
# read (``MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS``, set by the prompt read), so the bank is sized for
# it once, the history it already holds copied once, and every chunk then writes its rows in
# place. The same rows at the same positions: what the attention reads is unchanged.
#
# A window visit's cache keeps the last 1,023 rows and appends the chunk (the library's
# ``RotatingKVCache._update_concat``: a slice and a concatenation a tensor). One launch here
# moves the same values into the same new arrays.
# ----------------------------------------------------------------------------------------

#: A prompt length beyond this is not believed (a stale or wrong value sizes nothing).
MAX_PROMPT = 2 * 1024 * 1024


def _prompt_tokens():
    try:
        return int(os.environ.get("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "0"))
    except ValueError:
        return 0


def reserve_bank(kv, rows_k):
    """Size a global visit's cache for the whole prompt before a chunk would make it grow; returns
    whether it did. ``rows_k`` is the chunk's keys ``(1, heads, rows, dim)``."""
    keys = kv.keys
    need = kv.offset + rows_k.shape[2]
    if keys is not None and need <= keys.shape[2]:
        return False
    total = _prompt_tokens()
    if not need <= total <= MAX_PROMPT:
        return False
    step = getattr(kv, "step", 256)
    cap = -(-total // step) * step
    shape = (1, rows_k.shape[1], cap, rows_k.shape[3])
    new_k, new_v = mx.zeros(shape, rows_k.dtype), mx.zeros(shape, rows_k.dtype)
    if keys is not None and kv.offset:
        new_k[..., : kv.offset, :] = keys[..., : kv.offset, :]
        new_v[..., : kv.offset, :] = kv.values[..., : kv.offset, :]
    kv.keys, kv.values = new_k, new_v
    return True


_WINDOW_SOURCE = """
    const uint d = thread_position_in_grid.x;
    const int r = int(thread_position_in_grid.y);
    const int h = int(thread_position_in_grid.z);
    const int start = int(params[0]);
    const int kept = int(params[1]) - start;
    const int n_out = kept + int(params[2]);
    const size_t at = (size_t(h) * size_t(n_out) + size_t(r)) * size_t(D) + size_t(d);
    if (r < kept) {
        const int s = start + r;
        ko[at] = kold[size_t(h) * size_t(kold_strides[1]) + size_t(s) * size_t(kold_strides[2]) + size_t(d) * size_t(kold_strides[3])];
        vo[at] = vold[size_t(h) * size_t(vold_strides[1]) + size_t(s) * size_t(vold_strides[2]) + size_t(d) * size_t(vold_strides[3])];
    } else {
        const int n = r - kept;
        ko[at] = knew[size_t(h) * size_t(knew_strides[1]) + size_t(n) * size_t(knew_strides[2]) + size_t(d) * size_t(knew_strides[3])];
        vo[at] = vnew[size_t(h) * size_t(vnew_strides[1]) + size_t(n) * size_t(vnew_strides[2]) + size_t(d) * size_t(vnew_strides[3])];
    }
"""


@lru_cache(maxsize=1)
def _window_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_window_rows",
        input_names=["kold", "vold", "knew", "vnew", "params"],
        output_names=["ko", "vo"],
        source=_WINDOW_SOURCE,
        ensure_row_contiguous=False,
    )


def window_update(kv, keys, values):
    """``kv.update_and_fetch(keys, values)`` for a multi-row chunk on a window cache, with the new
    window built by one launch; the library's own update where its window is not in time order
    (a ring rotated by decode) or it keeps leading rows."""
    old = kv.keys
    if (old is None or keys.shape[2] < 2 or getattr(kv, "keep", 0) != 0 or kv._idx != old.shape[2]
            or kv.values.shape != old.shape or old.dtype != keys.dtype):
        return kv.update_and_fetch(keys, values)
    start = max(0, kv._idx - kv.max_size + 1)
    rows = keys.shape[2]
    n_out = old.shape[2] - start + rows
    heads, dim = keys.shape[1], keys.shape[3]
    new_k, new_v = _window_kernel()(
        inputs=[old, kv.values, keys, values, mx.array([start, old.shape[2], rows], mx.int32)],
        template=[("D", dim)],
        grid=(dim, n_out, heads),
        threadgroup=(dim, 1, 1),
        output_shapes=[(1, heads, n_out, dim), (1, heads, n_out, dim)],
        output_dtypes=[keys.dtype, keys.dtype],
    )
    kv.keys, kv.values = new_k, new_v
    kv.offset += rows
    kv._idx = n_out
    return new_k, new_v


# ----------------------------------------------------------------------------------------
# Which prompts the stage reads. It wins where the attention's share of a chunk grows with the
# keys, and is level or behind on short prompts, where a chunk's attention is a small part of it
# and the default route's launches are as fast. So a prompt shorter than SHORT_PROMPT reads on the
# default route, every chunk of it: the length is the whole prompt's (set by the prompt read
# before its first chunk, restored prefix included), never less than the keys this chunk sees,
# so one prompt reads on one route. And a chunk takes the stage only if the Mac's working set has
# room for what the stage holds beyond the default route: its chunk peak is higher by up to
# EXTRA_PEAK_BYTES, and a global visit's first chunk sizes the visit's bank for the whole prompt
# at once (the default route grows it chunk by chunk). Without the room the chunk reads on the
# default route, counted, as it does today.
# ----------------------------------------------------------------------------------------

#: The prompt length from which the stage reads a prompt faster than the default route: for reads of
#: 2,048 rows or more a forward (``SHORT_PROMPT_WIDE``; there the default route computes every masked key
#: of a window visit and writes rows x keys bias planes, so the stage pays from shorter prompts) and for
#: narrower reads (``SHORT_PROMPT``: a Mac without the headroom for the wide width, or a narrowed request).
#: Wide: 4,096. Bare Speed, 4,097 tokens at 2,048 rows, in process through the product's prefill, the stage
#: read in 1.91 s against stock's 2.57 (and 2.79 against 3.52 in an earlier run, both after the same warm-up,
#: the stage read second; lanes/p2/posacc-bare-2.json, posacc-bare-1.json), every prompt position inside stock's
#: own re-cut spread against the exact reference. Narrow: 8,192 until a 512-row read below it is measured
#: (at 1,024 tokens the two tie: 0.66 against 0.65 s).
SHORT_PROMPT = 8192
SHORT_PROMPT_WIDE = 4096
WIDE_READ_ROWS = 2048


def _read_width():
    from mtplx.generation import _prefill_chunk_size

    return _prefill_chunk_size()

#: The most a prompt chunk on the stage holds at its peak beyond the default route's chunk:
#: lanes/p2/decide-1.json, peak growth over the active memory, stage against stock, 0.22 GB at 4K,
#: 0.41 GB at 16K, 0.12 GB at 64K on Optimized Quality.
EXTRA_PEAK_BYTES = 512 * 1024**2


def short_prompt(keys, width=None):
    """Whether the prompt a call belongs to is shorter than the crossover for its read's width (the
    request's rows a forward unless given): its length as the prompt read set it, never less than the
    ``keys`` the call sees."""
    total = _prompt_tokens()
    width = _read_width() if width is None else width
    limit = SHORT_PROMPT_WIDE if width >= WIDE_READ_ROWS else SHORT_PROMPT
    return max(total if total <= MAX_PROMPT else 0, keys) < limit


def _bank_bytes(kv, rows, cfg, dtype):
    """What :func:`reserve_bank` allocates for this chunk of a global visit (0 where it does not)."""
    keys = kv.keys
    need = kv.offset + rows
    if keys is not None and need <= keys.shape[2]:
        return 0
    total = _prompt_tokens()
    if not need <= total <= MAX_PROMPT:
        return 0
    step = getattr(kv, "step", 256)
    return 2 * (-(-total // step) * step) * cfg["num_key_value_heads"] * cfg["head_dim"] * dtype.size


def _select(layer, kv, rows, dtype):
    """``None`` where the stage reads this chunk, else why the default route does."""
    from ..head import fits_route

    if short_prompt(kv.offset + rows):
        return "short_prompt"
    extra = EXTRA_PEAK_BYTES + (0 if layer.sliding else _bank_bytes(kv, rows, layer.cfg, dtype))
    if not fits_route(extra):
        return "memory"
    return None


def select_call(keys):
    """The same choice for one attention call of a prompt (``attention.attend``'s one-launch route,
    which serves prompt calls the stage did not take): it holds no bank of its own."""
    from ..head import fits_route

    if short_prompt(keys):
        return "short_prompt"
    if not fits_route(EXTRA_PEAK_BYTES):
        return "memory"
    return None


def stage(layer, query, k_rows, v_rows, profiles, sink, kv):
    """A prompt chunk's cache write and attention (more than 32 rows, not a verify panel) on our
    launches: the window visit's new window in one launch (:func:`window_update`) or the global
    visit's bank sized once for the prompt (:func:`reserve_bank`), then the attention in one launch
    (:func:`flash_attention`). Everything around it is the layer's own: the projections, the clips,
    value norm and position scale, the profile product, the output gate. Returns the attention
    ``(1, heads, rows, 128)`` as ``attention.attend`` does, or ``None`` (counted) without touching the
    cache where the switch is off or the stage declines (a short prompt, no room, the checks below)."""
    from .. import attention as A

    if not A.flash_switch_on():
        return None
    visit = "window" if layer.sliding else "full"
    dtype = query.dtype
    reason = None
    if os.environ.get("MTPLX_INKLING_ATTENTION", "stock") != "stock":
        reason = "attention_mode"
    elif os.environ.get("MTPLX_INKLING_PROMPT_ATTENTION", "fused") == "stock":
        reason = "forced_stock"
    elif mx.default_device() != mx.gpu or not mx.metal.is_available() or not tensor_units():
        reason = "no_tensor_units"
    elif dtype not in (mx.bfloat16, mx.float16):
        reason = "dtype"
    elif query.ndim != 4 or query.shape[0] != 1 or query.shape[3] != HEAD_DIM or k_rows.shape[3] != HEAD_DIM:
        reason = "geometry"
    if reason is None:
        reason = _select(layer, kv, query.shape[2], dtype)
    if reason is None:
        reason = A._flash_first_use(dtype, layer.sliding)
    A._note_route(f"prompt_stage.{visit}." + ("selected" if reason is None else "stock." + reason))
    if reason is not None:
        return None
    if layer.sliding:
        keys, values = window_update(kv, k_rows, v_rows)
        keys, values = kv._temporal_order(keys), kv._temporal_order(values)
    else:
        reserve_bank(kv, k_rows)
        keys, values = kv.update_and_fetch(k_rows, v_rows)
    window = layer.cfg["sliding_window_size"] if layer.sliding else 0
    return flash_attention(query, keys, values, profiles, sink, window)


def _exact_pieces(x, n):
    """``x`` (float32) as ``n`` bf16-exact float32 pieces whose sum is ``x`` to about 2^-8n."""
    out, r = [], x
    for i in range(n):
        piece = r.astype(mx.bfloat16).astype(mx.float32)
        out.append(piece)
        if i + 1 < n:
            r = r - piece
    return out


def _exact_product(a, b, na, nb):
    """``a @ b`` in float32 with every product exact. The GPU's float32 product rounds its operands
    to about ten mantissa bits (lanes/p2/mmcheck-1: 7.7e-4 relative error against float64), but
    multiplies bf16-exact operands exactly into float32 sums (6.6e-8)."""
    acc = None
    for x in _exact_pieces(a, na):
        for y in _exact_pieces(b, nb):
            t = x @ y
            acc = t if acc is None else acc + t
    return acc


def reference(query, key, value, profiles, sink, window):
    """The definition in float32 with MLX's own ops and exact products (for tests and checks;
    small calls only): 1e-6 relative of the float64 definition (lanes/p2/acc-1)."""
    batch, heads, rows, width = query.shape
    kv_heads, keys = key.shape[1], key.shape[2]
    group = heads // kv_heads
    extent = profiles.shape[3]
    n_in = 1 if query.dtype == mx.bfloat16 else 2
    q = query.astype(mx.float32)
    k = mx.repeat(key.astype(mx.float32), group, axis=1)
    v = mx.repeat(value.astype(mx.float32), group, axis=1)
    scores = _exact_product(q, k.transpose(0, 1, 3, 2), n_in, n_in) * (width ** -0.5)
    p = (keys - rows) + mx.arange(rows)[:, None]
    d = p - mx.arange(keys)[None, :]
    prof = profiles.astype(mx.float32).transpose(0, 2, 1, 3)          # (1, heads, rows, extent)
    bias = mx.take_along_axis(prof, mx.broadcast_to(mx.clip(d, 0, extent - 1)[None, None],
                                                    (batch, heads, rows, keys)), axis=-1)
    bias = mx.where(d >= extent, 0.0, bias)
    seen = (d >= 0) if not window else ((d >= 0) & (d < window))
    logits = mx.where(seen, scores + bias, -mx.inf)
    s = sink.astype(mx.float32).transpose(0, 2, 1)[..., None]           # (1, heads, rows, 1)
    top = mx.maximum(mx.max(logits, axis=-1, keepdims=True), s)
    w = mx.exp(logits - top)
    denom = w.sum(axis=-1, keepdims=True) + mx.exp(s - top)
    return _exact_product(w, v, 3, n_in) / denom
