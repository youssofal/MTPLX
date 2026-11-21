"""The prompt read's expert block in our own launches: the pair sort and both expert products.

At a prompt chunk the expert block sorts its routed pairs by expert, copies every token row once per
pair in that order, and runs MLX's sorted product twice (gate-up, then down), with two argsorts, a
divide, gathers and copies between: about twenty MLX launches a visit besides the fused input, route,
SwiGLU and combine kernels. Here they are three launches of ours:

``sort``
    One threadgroup counts the pairs of every expert, lays the experts' runs out in expert order
    (the first sorted row and the first 64-row tile of each), and places every pair in its expert's
    run: the sorted pair, its token row, its expert, and the inverse (a pair's sorted position, which
    the combine reads). A pair's place inside its run carries no meaning: every product row and every
    later per-pair step depends on that pair's own values only.
``product`` (gate-up, then down)
    Every threadgroup owns one 64-row tile of ONE expert's run (an expert's run is cut into its own
    tiles) and 64 output columns. The token rows are read in place through the sort's row map, so no
    sorted copy of them is written. Per 64-deep block of the reduction the threadgroup decodes the
    expert's weight block (E2M1 codes times E4M3 group scales for NVFP4, E4M3 codes times E8M0 scales
    for MXFP8; every decoded weight is exact in bfloat16) into threadgroup memory, and each of its four
    simdgroups multiplies its 32 by 32 output tile on the tensor units, one 16-deep fragment after the
    other in ascending order, into float32. The output is the float32 sum, as the fused SwiGLU and
    combine kernels read it.

The multiply-accumulate is the installed MLX's own tensor-unit tile and fragment code, read from the
headers MLX ships beside itself when the kernel is first built and compiled into our launch (nothing
of it lives in this tree), fed the same fragments in the same order as MLX's sorted product, so a
product row has the sums MLX's sorted product gives it. Admission is the expert route's own first-use
proof (`expert_fusion._decide`): the prompt route with these launches must give the stock body's bits
on the first call's rows, per format, or the stock body serves.

Tensor-unit GPUs only (``nax_detect``); an install whose headers cannot be read keeps MLX's products,
with the reason counted.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import logging
from threading import Lock

import mlx.core as mx

_log = logging.getLogger(__name__)
_lock = Lock()
_counts: Counter = Counter()

BM = 64  # rows of one expert's run a threadgroup owns
BN = 64  # output columns a threadgroup owns
BK = 64  # reduction depth decoded into threadgroup memory at a time
SORT_THREADS = 1024
PRODUCT_THREADS = 128

#: The installed MLX's tensor-unit tile header (and the headers it includes).

def counts():
    with _lock:
        return dict(_counts)


def _count(key):
    with _lock:
        _counts[key] += 1


@lru_cache(maxsize=1)
def _installed_tile_header():
    """(text, None) with the installed MLX's tensor-unit tile and fragment templates, or (None, why)."""
    from mtplx.kernels.library_headers import text

    header, why = text("tensor_tiles")
    return (None, why) if header is None else (header + "using namespace mlx::steel;\n", None)


@lru_cache(maxsize=1)
def available():
    """(True, None) where these launches can run: tensor units and a readable tile header."""
    if not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        return False, "cpu"
    try:
        from mtplx import nax_detect

        if not nax_detect.nax_available():
            return False, "no_tensor_units"
    except Exception as exc:  # noqa: BLE001
        return False, f"nax_detect: {type(exc).__name__}"
    text, why = _installed_tile_header()
    if text is None:
        _log.warning("Inkling prompt expert products keep MLX's kernels: %s", why)
        return False, "headers"
    return True, None


# --- the pair sort ------------------------------------------------------------------------------

SORT = """
    // Template constants: E experts, TOP pairs a row, BM rows a tile. P pairs from the input's shape.
    threadgroup atomic_int filled[E];
    threadgroup int first[E];
    const uint t = thread_position_in_threadgroup.x;
    const uint n = threads_per_threadgroup.x;
    const uint P = uint(experts_shape[0]);
    for (uint e = t; e < uint(E); e += n)
        atomic_store_explicit(&filled[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = t; p < P; p += n)
        atomic_fetch_add_explicit(&filled[experts[p]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int row = 0;
        int tile = 0;
        for (uint e = 0; e < uint(E); ++e) {
            const int size = atomic_load_explicit(&filled[e], memory_order_relaxed);
            starts[e] = row;
            tiles[e] = tile;
            first[e] = row;
            row += size;
            tile += (size + BM - 1) / BM;
        }
        starts[E] = row;
        tiles[E] = tile;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = t; e < uint(E); e += n)
        atomic_store_explicit(&filled[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = t; p < P; p += n) {
        const uint e = experts[p];
        const uint at = uint(first[e] + atomic_fetch_add_explicit(&filled[e], 1, memory_order_relaxed));
        order[at] = p;
        rows[at] = p / uint(TOP);
        sorted_experts[at] = e;
        inverse[p] = at;
    }
"""


@lru_cache(maxsize=None)
def _sort_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_prompt_pair_sort",
        input_names=["experts"],
        output_names=["order", "rows", "sorted_experts", "inverse", "tiles", "starts"],
        source=SORT,
    )


def sort_pairs(experts, top, n_experts, rows_a_tile=BM):
    """Pairs (row-major, ``experts`` flat uint32) grouped by expert: the sorted pairs, their token rows,
    their experts, each pair's sorted position, and each expert's first tile (``rows_a_tile`` rows a tile)
    and first sorted row (``n_experts + 1`` entries, the last the totals)."""
    pairs = experts.size
    order, rows, sorted_experts, inverse, tiles, starts = _sort_kernel()(
        inputs=[experts.reshape(-1)],
        template=[("E", n_experts), ("TOP", top), ("BM", rows_a_tile)],
        grid=(SORT_THREADS, 1, 1),
        threadgroup=(SORT_THREADS, 1, 1),
        output_shapes=[(pairs,), (pairs,), (pairs,), (pairs,), (n_experts + 1,), (n_experts + 1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32, mx.int32, mx.int32],
    )
    return order, rows, sorted_experts, inverse, tiles, starts


# --- the products ---------------------------------------------------------------------------------

DECODE = """
constant float mtplx_e2m1_magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
// A 4-bit E2M1 code (sign, two exponent bits, one mantissa bit).
inline float mtplx_e2m1(uint code) {
    const float v = mtplx_e2m1_magnitude[code & 7u];
    return (code & 8u) ? -v : v;
}
// An 8-bit E4M3 code (sign, four exponent bits biased by 7, three mantissa bits), decoded as MLX decodes it:
// no infinity and no NaN (0x7F is 480). A subnormal code is its mantissa times 2^-9, computed from an integer
// so that no float32 subnormal (which the GPU may flush) appears on the way.
inline float mtplx_e4m3(uint code) {
    const uint magnitude = code & 0x7Fu;
    const float v = magnitude < 8u
        ? float(magnitude) * 0.001953125f
        : as_type<float>((((magnitude >> 3) + 120u) << 23) | ((magnitude & 7u) << 20));
    return (code & 0x80u) ? -v : v;
}
// An 8-bit E8M0 scale, decoded as MLX decodes it: 2^(code - 127), code 0 the float32 2^-127, 0xFF infinity.
inline float mtplx_e8m0(uint code) {
    return code == 0u ? as_type<float>(0x00400000u) : as_type<float>(code << 23);
}

// Thirty-two consecutive weights of one output column, decoded from their codes and group scales into `to`
// (each decoded weight is exact in W: at most 6 significant bits).
template <typename W, bool NV>
inline void mtplx_decode_run(const device uint8_t* codes, const device uint8_t* scales, threadgroup W* to) {
    if (NV) {
        // 16 bytes of codes, two nibbles a byte (the low one first), a scale per 16 weights
        const uint4 word = *(const device uint4*)codes;
        const float first = mtplx_e4m3(scales[0]);
        const float second = mtplx_e4m3(scales[1]);
        for (short e = 0; e < 32; ++e) {
            const uint lane_word = e < 8 ? word.x : e < 16 ? word.y : e < 24 ? word.z : word.w;
            const uint nib = (lane_word >> ((e & 7) * 4)) & 0xFu;
            to[e] = W((e < 16 ? first : second) * mtplx_e2m1(nib));
        }
    } else {
        // 32 bytes of codes, one scale for all of them
        const float scale = mtplx_e8m0(scales[0]);
        const device uint* words = (const device uint*)codes;
        for (short e = 0; e < 32; ++e)
            to[e] = W(scale * mtplx_e4m3((words[e >> 2] >> ((e & 3) * 8)) & 0xFFu));
    }
}
"""

PRODUCT = """
    // T: the input rows' type; W: the decoded weights' type; FP4; K: the reduction; N: output columns;
    // E: experts; PROMPT_ROWMAP: the input rows are read through `rowmap`. Writes float32 sums in
    // sorted-row order. A threadgroup is 4 simdgroups over a 64 x 64 output block (2 x 2 of 32 x 32);
    // the reduction runs 64 deep a block, 32 deep a multiply (two 16-deep fragments), ascending.
    enum : short { BLOCK = 64, DEEP = 64, PART = 32, FRAGS = 2, STRIDE = DEEP + 16 / short(sizeof(W)) };
    threadgroup W decoded[BLOCK * STRIDE];

    const int tile = int(threadgroup_position_in_grid.y);
    if (tile >= tiles[E])
        return;
    int lo = 0;
    int hi = E - 1;
    while (lo < hi) {                                // the expert whose run holds this tile
        const int probe = (lo + hi + 1) >> 1;
        if (tiles[probe] <= tile)
            lo = probe;
        else
            hi = probe - 1;
    }
    const int expert = lo;
    const int first_row = starts[expert] + (tile - tiles[expert]) * BLOCK;
    const int live = min(int(BLOCK), starts[expert + 1] - first_row);
    const int first_col = int(threadgroup_position_in_grid.x) * BLOCK;
    const ushort group_id = ushort(simdgroup_index_in_threadgroup);
    const short own_row = PART * short(group_id >> 1);
    const short own_col = PART * short(group_id & 1);
    const bool busy = live > own_row;

    // The four input rows this lane feeds to every multiply (two of each 16-row fragment), fixed for the
    // whole reduction; a row past the run feeds zeros.
    const short2 place = BaseNAXFrag::get_coord();
    const device T* feed[4];
    bool real[4];
    for (short q = 0; q < 4; ++q) {
        const short r = own_row + (q >> 1) * 16 + (q & 1) * 8 + place.y;
        real[q] = r < live;
#if PROMPT_ROWMAP
        const uint input_row = rowmap[first_row + (real[q] ? int(r) : 0)];
#else
        const uint input_row = uint(first_row + (real[q] ? int(r) : 0));
#endif
        feed[q] = x + size_t(input_row) * K + place.x;
    }

    // The weights this thread decodes: one column of the block, 32 of every 64 deep.
    const uint me = thread_position_in_threadgroup.x;
    const size_t weight_row = size_t(expert) * N + size_t(first_col) + (me >> 1);
    const int code_bytes = FP4 ? K / 2 : K;
    const int run_codes = FP4 ? 16 : 32;
    const int run_scales = FP4 ? 2 : 1;
    const device uint8_t* my_codes = (const device uint8_t*)w + weight_row * code_bytes + (me & 1u) * run_codes;
    const device uint8_t* my_scales = s + weight_row * (K / (FP4 ? 16 : 32)) + (me & 1u) * run_scales;
    threadgroup W* my_run = decoded + (me >> 1) * STRIDE + (me & 1u) * 32;

    using SumTile = NAXTile<float, FRAGS, FRAGS>;
    using RowTile = NAXTile<T, FRAGS, FRAGS>;
    using WeightTile = NAXTile<W, FRAGS, FRAGS>;
    constexpr metal::bool_constant<false> as_stored{};
    constexpr metal::bool_constant<true> transposed{};
    SumTile total;
    total.clear();
    for (int depth = 0; depth < K; depth += DEEP) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mtplx_decode_run<W, bool(FP4)>(my_codes, my_scales, my_run);
        my_codes += 2 * run_codes;
        my_scales += 2 * run_scales;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!busy)
            continue;
        for (short part = 0; part < DEEP; part += PART) {
            // the simdgroup's 32 output columns, this part's 32 deep, as stored (column-major rows)
            WeightTile weights_in;
            weights_in.template load<W, STRIDE, 1>(decoded + own_col * STRIDE + part);
            // fragment (r, d): this lane's two rows of row fragment r, four values from column
            // 16 d + place.x of this part
            RowTile rows_in;
            for (short f = 0; f < FRAGS * FRAGS; ++f) {
                thread auto& frag = rows_in.frag_at(f >> 1, f & 1);
                const int column = depth + part + (f & 1) * 16;
                for (short e = 0; e < 8; ++e) {
                    const short q = (f >> 1) * 2 + (e >> 2);
                    frag[e] = real[q] ? feed[q][column + (e & 3)] : T(0);
                }
            }
            tile_matmad_nax(total, rows_in, as_stored, weights_in, transposed);
        }
    }
    if (busy) {
        for (short f = 0; f < FRAGS * FRAGS; ++f) {
            const thread auto& frag = total.frag_at(f >> 1, f & 1);
            for (short e = 0; e < 8; ++e) {
                const short r = own_row + (f >> 1) * 16 + (e >> 2) * 8 + place.y;
                if (r < live)
                    y[size_t(first_row + r) * N + first_col + own_col + (f & 1) * 16 + place.x + (e & 3)] = frag[e];
            }
        }
    }
"""


#: The wide tiling (``PRODUCT_WIDE``): the same products in the same order with more rows and columns a
#: threadgroup. ``SGM`` x ``SGN`` simdgroups each own a 32 x 32 output tile, so a threadgroup owns
#: ``32 SGM`` rows of one expert's run and ``32 SGN`` columns; the reduction runs 32 deep a step through two
#: threadgroup buffers, the next step's weights decoded while this step's multiply runs (one barrier a step).
#: Every output element is still its 16-deep fragments in ascending order into float32, so the sums are the
#: narrow kernel's and MLX's sorted product's.
PRODUCT_WIDE = """
    enum : short { BMW = 32 * SGM, BNW = 32 * SGN, DEEPW = 32, STRIDEW = DEEPW + 16 / short(sizeof(W)),
                   PER = 32 / SGM };   // weights a thread decodes a step: one column's PER consecutive weights
    threadgroup W decoded[2][BNW * STRIDEW];

    const int tile = int(threadgroup_position_in_grid.y);
    if (tile >= tiles[E])
        return;
    int lo = 0;
    int hi = E - 1;
    while (lo < hi) {                                // the expert whose run holds this tile
        const int probe = (lo + hi + 1) >> 1;
        if (tiles[probe] <= tile)
            lo = probe;
        else
            hi = probe - 1;
    }
    const int expert = lo;
    const int first_row = starts[expert] + (tile - tiles[expert]) * BMW;
    const int live = min(int(BMW), starts[expert + 1] - first_row);
    const int first_col = int(threadgroup_position_in_grid.x) * BNW;
    const ushort group_id = ushort(simdgroup_index_in_threadgroup);
    const short own_row = 32 * short(group_id / SGN);
    const short own_col = 32 * short(group_id % SGN);
    const bool busy = live > own_row;

    const short2 place = BaseNAXFrag::get_coord();
    const device T* feed[4];
    bool real[4];
    for (short q = 0; q < 4; ++q) {
        const short r = own_row + (q >> 1) * 16 + (q & 1) * 8 + place.y;
        real[q] = r < live;
#if PROMPT_ROWMAP
        const uint input_row = rowmap[first_row + (real[q] ? int(r) : 0)];
#else
        const uint input_row = uint(first_row + (real[q] ? int(r) : 0));
#endif
        feed[q] = x + size_t(input_row) * K + place.x;
    }

    // The weights this thread decodes each step: column `me / SGM` of the block, PER deep from `(me % SGM) PER`.
    const uint me = thread_position_in_threadgroup.x;
    const uint my_col = me / uint(SGM);
    const uint my_part = me % uint(SGM);
    const size_t weight_row = size_t(expert) * N + size_t(first_col) + my_col;
    const device uint8_t* my_codes = (const device uint8_t*)w + weight_row * (FP4 ? K / 2 : K)
                                     + my_part * (FP4 ? PER / 2 : PER);
    const device uint8_t* my_scales = s + weight_row * (K / (FP4 ? 16 : 32));
    threadgroup W* my_run[2] = {decoded[0] + my_col * STRIDEW + my_part * PER,
                                decoded[1] + my_col * STRIDEW + my_part * PER};

    using SumTile = NAXTile<float, 2, 2>;
    using RowTile = NAXTile<T, 2, 2>;
    using WeightTile = NAXTile<W, 2, 2>;
    typedef vec<T, 4> four_t;      // four consecutive values of one input row, one load
    constexpr metal::bool_constant<false> as_stored{};
    constexpr metal::bool_constant<true> transposed{};
    SumTile total;
    total.clear();
    constexpr int steps = K / DEEPW;
    mtplx_decode_part<W, bool(FP4), PER>(my_codes, my_scales[(my_part * PER) / (FP4 ? 16 : 32)], my_run[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) {
            const int next = (step + 1) * DEEPW;
            mtplx_decode_part<W, bool(FP4), PER>(my_codes + (FP4 ? next / 2 : next),
                                                 my_scales[(next + int(my_part) * PER) / (FP4 ? 16 : 32)],
                                                 my_run[(step + 1) & 1]);
        }
        if (busy) {
            WeightTile weights_in;
            weights_in.template load<W, STRIDEW, 1>(decoded[step & 1] + own_col * STRIDEW);
            RowTile rows_in;
            const int column = step * DEEPW;
            for (short f = 0; f < 4; ++f) {
                thread auto& frag = rows_in.frag_at(f >> 1, f & 1);
                for (short pair_row = 0; pair_row < 2; ++pair_row) {
                    const short q = (f >> 1) * 2 + pair_row;
                    const four_t v = real[q] ? reinterpret_cast<const device four_t*>(feed[q] + column + (f & 1) * 16)[0]
                                             : four_t(0);
                    for (short e = 0; e < 4; ++e)
                        frag[pair_row * 4 + e] = v[e];
                }
            }
            tile_matmad_nax(total, rows_in, as_stored, weights_in, transposed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (busy) {
        for (short f = 0; f < 4; ++f) {
            const thread auto& frag = total.frag_at(f >> 1, f & 1);
            for (short e = 0; e < 8; ++e) {
                const short r = own_row + (f >> 1) * 16 + (e >> 2) * 8 + place.y;
                if (r < live)
                    y[size_t(first_row + r) * N + first_col + own_col + (f & 1) * 16 + place.x + (e & 3)] = frag[e];
            }
        }
    }
"""

DECODE_PART = """
// PER consecutive weights of one output column, inside one scale group, decoded from 32-bit words of codes
// (two E2M1 nibbles a byte, the low one first; or one E4M3 code a byte) times the group's scale.
template <typename W, bool NV, short PER>
inline void mtplx_decode_part(const device uint8_t* codes, uint8_t scale_code, threadgroup W* to) {
    const device uint* words = (const device uint*)codes;
    if (NV) {
        const float scale = mtplx_e4m3(scale_code);
        for (short b = 0; b < PER / 8; ++b) {
            const uint word = words[b];
            for (short e = 0; e < 8; ++e)
                to[b * 8 + e] = W(scale * mtplx_e2m1((word >> (4 * e)) & 0xFu));
        }
    } else {
        const float scale = mtplx_e8m0(scale_code);
        for (short b = 0; b < PER / 4; ++b) {
            const uint word = words[b];
            for (short e = 0; e < 4; ++e)
                to[b * 4 + e] = W(scale * mtplx_e4m3((word >> (8 * e)) & 0xFFu));
        }
    }
}
"""

#: The products' epilogue for a caller that would scale and round each float32 sum itself (the prompt route's
#: SwiGLU and combine): the pair's outer scale (its row's or its own, times its expert's) times the sum, rounded
#: once to the stream's type O, as those stages compute it; or the sum rounded to O where there is no outer scale
#: (MXFP8). Half the bytes written and read, the same values.
_STORE = """                if (r < live)
                    y[size_t(first_row + r) * N + first_col + own_col + (f & 1) * 16 + place.x + (e & 3)] = frag[e];"""
_STORE_ROUNDED = """                if (r < live) {
                    const size_t at = size_t(first_row + r) * N + first_col + own_col + (f & 1) * 16 + place.x + (e & 3);
#if ROUNDED_SCALED
#if PROMPT_ROWMAP
                    const float scale = outer[rowmap[first_row + r]] * expert_outer[expert];
#else
                    const float scale = outer[first_row + r] * expert_outer[expert];
#endif
                    y[at] = static_cast<O>(frag[e] * scale);
#else
                    y[at] = static_cast<O>(frag[e]);
#endif
                }"""


#: (SGM, SGN) of the wide tiling by average pairs an expert; None keeps the narrow kernel (64 x 64 a
#: threadgroup, 64 deep a step). Measured on one layer's real weights, gate-up and down together
#: (`lanes/p1/pe-wide-2.json`, provisional): the 2 x 2 tiling (the narrow kernel's 64 x 64 block with the
#: double-buffered 32-deep decode and four-value row loads) is the fastest or within 1.5% at every width, both
#: formats: NVFP4 2.51 -> 2.14 ms at 512 rows, 4.38 -> 3.64 at 1,024, 11.60 -> 9.15 at 2,048, 44.0 -> 34.0 at
#: 4,096; MXFP8 2.78 -> 2.56, 5.51 -> 4.81, 17.13 -> 14.26, 47.9 -> 38.3. Larger blocks (4 x 2, 2 x 4, 4 x 4)
#: are slower.
WIDE_PLAN = ((0, (2, 2)),)


def tiling(pairs, n_experts):
    """``(rows a tile, (SGM, SGN) or None)`` for a prompt chunk of ``pairs`` routed pairs."""
    import os

    forced = os.environ.get("MTPLX_INKLING_PROMPT_EXPERT_TILES", "").strip().lower()
    if forced in ("narrow", "64"):
        return BM, None
    if forced.count("x") == 1:
        sgm, sgn = (int(v) for v in forced.split("x"))
        return 32 * sgm, (sgm, sgn)
    per_expert = pairs / max(1, n_experts)
    for floor, shape in WIDE_PLAN:
        if per_expert >= floor:
            return (BM, None) if shape is None else (32 * shape[0], shape)
    return BM, None


@lru_cache(maxsize=None)
def _product_kernel(mapped, wide=None, rounded=None):
    """``rounded``: None (float32 sums), "cast" (the sums rounded to O) or "scaled" (outer scale times the sum,
    rounded to O)."""
    header, _ = _installed_tile_header()
    source = PRODUCT_WIDE if wide else PRODUCT
    inputs = (["x", "rowmap"] if mapped else ["x"]) + ["w", "s", "tiles", "starts"]
    if rounded is not None:
        source = source.replace(_STORE, _STORE_ROUNDED)
        if rounded == "scaled":
            inputs += ["outer", "expert_outer"]
    return mx.fast.metal_kernel(
        name="mtplx_inkling_prompt_expert_product" + ("_mapped" if mapped else "")
             + (f"_w{wide[0]}x{wide[1]}" if wide else "") + (f"_{rounded}" if rounded else ""),
        input_names=inputs,
        output_names=["y"],
        source=source,
        header=header + DECODE + (DECODE_PART if wide else "") + f"#define PROMPT_ROWMAP {int(mapped)}\n"
               + (f"#define ROUNDED_SCALED {int(rounded == 'scaled')}\n" if rounded else ""),
    )


def product(x, packed, tiles, starts, pairs, rowmap=None, weights=mx.bfloat16, wide=None, rounded=None):
    """float32 ``(pairs, N)``: every sorted pair's row of ``x`` (through ``rowmap`` when given, else the
    pair's own row) times its expert's ``packed`` weight (``model.PackedExperts``, NVFP4 or MXFP8).
    ``wide``: ``(SGM, SGN)`` from ``tiling`` (the sort's tiles must be cut at ``32 SGM`` rows), else the
    narrow kernel (64-row tiles). ``rounded``: ``(dtype, outer, expert_outer)`` returns ``dtype`` values instead
    of the float32 sums: ``outer[i] * expert_outer[e]`` times each sum (``i`` the pair's row through ``rowmap``,
    else its sorted position; ``e`` its expert), rounded once; with ``outer`` None, the sums rounded."""
    fp4 = packed.mode == "nvfp4"
    n_experts, n_out, words = packed.codes.shape
    depth = words * (8 if fp4 else 4)
    cols = 32 * wide[1] if wide else BN
    if depth % BK or n_out % cols or x.shape[-1] != depth:
        raise ValueError(f"prompt expert product: shape {x.shape} against {packed.codes.shape}")
    _count(f"{packed.mode}.{'mapped' if rowmap is not None else 'sorted'}.{str(x.dtype).split('.')[-1]}"
           + (f".w{wide[0]}x{wide[1]}" if wide else ""))
    rows_a_tile = 32 * wide[0] if wide else BM
    threads = 32 * wide[0] * wide[1] if wide else PRODUCT_THREADS
    tiles_bound = -(-pairs // rows_a_tile) + n_experts
    inputs = [x.reshape(-1, depth)] + ([rowmap] if rowmap is not None else [])
    template = [("T", x.dtype), ("W", weights), ("FP4", fp4), ("K", depth), ("N", n_out), ("E", n_experts)]
    if wide:
        template += [("SGM", wide[0]), ("SGN", wide[1])]
    extra, form, out_dtype = [], None, mx.float32
    if rounded is not None:
        out_dtype, outer, expert_outer = rounded
        form = "cast" if outer is None else "scaled"
        template += [("O", out_dtype)]
        if outer is not None:
            extra = [outer.reshape(-1), expert_outer.reshape(-1)]
    (y,) = _product_kernel(rowmap is not None, wide, form)(
        inputs=inputs + [packed.codes.view(mx.uint8), packed.scales, tiles, starts] + extra,
        template=template,
        grid=((n_out // cols) * threads, tiles_bound, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(pairs, n_out)],
        output_dtypes=[out_dtype],
    )
    return y
