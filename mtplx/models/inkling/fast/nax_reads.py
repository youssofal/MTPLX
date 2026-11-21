"""The three-query depth read and the four-query message read in one launch of ours (lane F1).

RC6 reads a global visit's input (three queries) and a loop pass's cross-pass messages (four queries a
visit) with two MLX products on the matrix units around a softmax: the scores ``q @ keys^T`` as a
split-K product whose partitions a second launch adds, the softmax (lane L1's ``read.probs``, or lane
F1's message weights), then ``probs @ keys``. That is four launches, three of them the library's, and
the split-K product runs its longest partition as one simdgroup's chain of dependent
load-then-multiply steps: at Inkling's width (2,304) the library cuts K into 2,048 and 256, so 64
steps, measured 62 to 72 µs a product.

This launch does all of it in one threadgroup. Every partition's chain runs at once in its own
simdgroup, with nothing between one step's multiply and the next step's operand reads; the partitions
are then added in the library's order (from 0.0, partition by partition); the softmax is
``read.probs``'s (the scale rounded on its own, ``share_four``); the mix runs 32 output columns a
multiply. The multiplies themselves are the installed MLX's own matrix-unit operations: its fragment
layout and its multiply-accumulate (``steel/gemm/nax.h``), read from the installed package when the
launch is built and compiled with it, so nothing of that text lives in this tree. They are issued in
the library's order over K (16 values a multiply, rising), with its zero padding (rows past the
queries, columns past the keys, and for the mix the K values past the sources, including the
library's all-zero second multiply), so every output carries the library's bits.

Class A, decided per shape on this Mac on the first call, against the MLX expression it replaces,
bit for bit. The split-K plan (partitions and their size) is read from the library's own dispatch of
that product shape when no recording is running, else the plan the library takes at these shapes on
every recording so far (two partitions, 2,048 rows of K) is tried; either way the proof decides. A
shape that differs, a Mac whose MLX does not put these products on the matrix units, an MLX whose
headers do not read, or a CPU run keeps the library's launches, counted
``fallback.nax_reads.<why>``. ``MTPLX_INKLING_NAX_READS=0`` turns it off.
"""

from __future__ import annotations

import os
import re
import struct
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .routes import count

#: (kind, shape...) -> True when the launch gives the MLX expression's bits on this Mac.
_VERDICT: dict = {}
#: (queries, score columns, width) -> (partitions, partition size) of the library's split-K product.
_PLAN: dict = {}
#: Simdgroups of the launch.
SIMDGROUPS = 8
#: Score-product steps whose operands are read ahead of their multiply. One: a deeper ring is
#: indexed by the loop and leaves registers (f1-naxbench-1: 210 us a read at four, against 13-31 us
#: at one and the library's 57-60 us).
AHEAD = 1
#: Threadgroup memory the launch may take (the partials and the softmax weights), bytes.
_TG_BYTES = 28 * 1024

_QUOTED = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")
_ALWAYS = "mlx/backend/metal/kernels/utils.h"  # MLX puts this (and what it includes) before every kernel
_TILES = "mlx/backend/metal/kernels/steel/gemm/nax.h"
_SPLIT_K = "steel_gemm_splitk_nax_nt_float32_float32_bm64_bn64_bk256_wm2_wn2"


def enabled():
    return os.environ.get("MTPLX_INKLING_NAX_READS", "1").strip().lower() not in ("0", "off", "false", "no")


@lru_cache(maxsize=1)
def library():
    """The installed MLX's matrix-unit fragment and multiply text with what it includes (minus what
    every kernel already has): ``(text, None)``, or ``(None, why)``."""
    root = Path(mx.__file__).resolve().parent / "include"
    have: set = set()
    text: list = []

    def known(rel):
        if rel in have:
            return
        have.add(rel)
        for line in (root / rel).read_text().splitlines():
            m = _QUOTED.match(line)
            if m:
                known(m.group(1))

    def take(rel):
        if rel in have:
            return
        have.add(rel)
        for line in (root / rel).read_text().splitlines():
            m = _QUOTED.match(line)
            if m:
                take(m.group(1))
            elif not _ONCE.match(line):
                text.append(line)

    try:
        known(_ALWAYS)
        take(_TILES)
    except OSError as exc:
        return None, f"headers_{type(exc).__name__}"
    body = "\n".join(text)
    for needle in ("struct BaseNAXFrag", "get_coord(short idx)", "void mma(", "matmul2d_descriptor"):
        if needle not in body:
            return None, "headers_changed"
    return body + "\n", None


_SCORES = """
    // One threadgroup. Template: V groups of G queries (NQ = V * G rows of q), W panel rows of S keys
    // each (N = W * S score columns), width D, P split-K partitions of PS values of K.
    using Frag = mlx::steel::BaseNAXFrag;
    typedef Frag::dtype_frag_t<float> frag_t;
    constexpr uint NQ = V * G;
    constexpr uint N = W * S;
    constexpr uint QB = (NQ + 15) / 16;          // 16-row blocks of the queries
    constexpr uint NB = (N + 31) / 32;           // 32-column blocks of the scores
    constexpr uint CHAINS = P * QB * NB;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float partial[P * NQ * N];
    threadgroup float weight[W * NQ * S];

    // This lane's (column, row) in a 16 x 16 fragment, for each of its 8 values: the library's layout.
    short2 at[8];
    for (short i = 0; i < 8; ++i) at[i] = Frag::get_coord(i);

    // ---- 1. The score product: chain (partition, query block, column block), one simdgroup each.
    for (uint c = sg; c < CHAINS; c += NSG) {
        const uint part = c / (QB * NB);
        const uint qb = (c / NB) % QB;
        const uint nb = c % NB;
        const uint k_lo = part * PS;
        const uint k_hi = min(k_lo + PS, uint(D));
        frag_t lo = frag_t(0.0f), hi = frag_t(0.0f);      // score columns nb*32 + 0..15 and 16..31
        uint qa[8], ka[8], kb[8];
        bool qv[8], av[8], bv[8];
        for (short i = 0; i < 8; ++i) {
            const uint m = qb * 16 + uint(at[i].y);
            const uint n = nb * 32 + uint(at[i].y);      // key rows run down a transposed fragment
            qv[i] = m < NQ;
            av[i] = n < N;
            bv[i] = n + 16 < N;
            qa[i] = (qv[i] ? m : 0) * D + uint(at[i].x);
            ka[i] = (av[i] ? n : 0) * D + uint(at[i].x);
            kb[i] = (bv[i] ? n + 16 : 0) * D + uint(at[i].x);
        }
        // Operands read AHEAD steps before their multiply (a ring of fragments), the multiplies
        // themselves in K order: the same multiplies on the same operands as the library's loop.
        frag_t ra[AHEAD], rb0[AHEAD], rb1[AHEAD];
        for (uint d = 0; d < AHEAD; ++d) {
            const uint k = k_lo + 16 * d;
            for (short i = 0; i < 8; ++i) {
                const bool in_k = k < k_hi;
                ra[d][i] = (in_k && qv[i]) ? static_cast<float>(q[qa[i] + k]) : 0.0f;
                rb0[d][i] = (in_k && av[i]) ? keys[ka[i] + k] : 0.0f;
                rb1[d][i] = (in_k && bv[i]) ? keys[kb[i] + k] : 0.0f;
            }
        }
        for (uint k0 = k_lo; k0 < k_hi; k0 += 16 * AHEAD) {
            for (uint d = 0; d < AHEAD; ++d) {
                const uint k = k0 + 16 * d;
                if (k < k_hi) {
                    Frag::mma(lo, hi, ra[d], metal::bool_constant<false>{}, rb0[d], rb1[d],
                              metal::bool_constant<true>{});
                    const uint kn = k + 16 * AHEAD;
                    const bool in_k = kn < k_hi;
                    for (short i = 0; i < 8; ++i) {
                        ra[d][i] = (in_k && qv[i]) ? static_cast<float>(q[qa[i] + kn]) : 0.0f;
                        rb0[d][i] = (in_k && av[i]) ? keys[ka[i] + kn] : 0.0f;
                        rb1[d][i] = (in_k && bv[i]) ? keys[kb[i] + kn] : 0.0f;
                    }
                }
            }
        }
        for (short i = 0; i < 8; ++i) {
            const uint m = qb * 16 + uint(at[i].y);
            const uint n = nb * 32 + uint(at[i].x);
            if (m < NQ) {
                if (n < N) partial[(part * NQ + m) * N + n] = lo[i];
                if (n + 16 < N) partial[(part * NQ + m) * N + n + 16] = hi[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 2. The partitions added from 0.0 in order, the scale, the softmax (share_four): one
    // simdgroup per (panel row, query).
    for (uint t = sg; t < W * NQ; t += NSG) {
        const uint r = t / NQ, m = t % NQ;
        float slot[4];
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            if (s < S) {
                float total = 0.0f;
                for (uint part = 0; part < P; ++part) total += partial[(part * NQ + m) * N + r * S + s];
                slot[i] = total * as_type<float>(uint(SCALE));
            } else {
                slot[i] = -INFINITY;
            }
        }
        share_four(slot);
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            if (s < S) STORE_WEIGHT;
        }
    }
    TAIL
"""

# How the scores launch ends: the fused launch keeps the weights in threadgroup memory and mixes; the
# other writes them out for the mix launch.
_FUSED_END = ("weight[(r * NQ + m) * S + s] = slot[i]", """threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = sg; item < ITEMS; item += NSG) {
        MIX_ITEM(weight[(r * NQ + v * G + g) * S + s])
    }""")
_WEIGHTS_END = ("weights[(((m / G) * W + r) * G + m % G) * S + s] = slot[i]", "")

# The mix of one (group, row, 16-query block, 32-column tile) item: weights (G x S) @ keys (S x D),
# K = S padded to the library's 32 (two multiplies, the second over zeros). WEIGHT names the weight
# of (r, v, g, s).
_MIX_ITEM = """
#define MIX_ITEM(WEIGHT)                                                                          \\
    {                                                                                             \\
        const uint tile = item % TILES;                                                           \\
        const uint gb = (item / TILES) % GB;                                                      \\
        const uint r = (item / (TILES * GB)) % W;                                                 \\
        const uint v = item / (TILES * GB * W);                                                   \\
        frag_t lo = frag_t(0.0f), hi = frag_t(0.0f);                                              \\
        for (uint kk = 0; kk < 2; ++kk) {                                                         \\
            frag_t a, b0, b1;                                                                     \\
            for (short i = 0; i < 8; ++i) {                                                       \\
                const uint g = gb * 16 + uint(at[i].y);                                           \\
                const uint s = kk * 16 + uint(at[i].x);                                           \\
                a[i] = (g < G && s < S) ? (WEIGHT) : 0.0f;                                        \\
                const uint sr = kk * 16 + uint(at[i].y);                                          \\
                const uint col = tile * 32 + uint(at[i].x);                                       \\
                b0[i] = sr < S ? keys[(r * S + sr) * D + col] : 0.0f;                             \\
                b1[i] = sr < S ? keys[(r * S + sr) * D + col + 16] : 0.0f;                        \\
            }                                                                                     \\
            Frag::mma(lo, hi, a, metal::bool_constant<false>{}, b0, b1, metal::bool_constant<false>{}); \\
        }                                                                                         \\
        for (short i = 0; i < 8; ++i) {                                                           \\
            const uint g = gb * 16 + uint(at[i].y);                                               \\
            if (g < G) {                                                                          \\
                const uint o = ((v * W + r) * G + g) * D + tile * 32 + uint(at[i].x);             \\
                out[o] = lo[i];                                                                   \\
                out[o + 16] = hi[i];                                                              \\
            }                                                                                     \\
        }                                                                                         \\
    }
"""

_MIX = """
    // The mix alone, one item a simdgroup across the grid (weights from the scores launch).
    using Frag = mlx::steel::BaseNAXFrag;
    typedef Frag::dtype_frag_t<float> frag_t;
    const uint sg = simdgroup_index_in_threadgroup;
    short2 at[8];
    for (short i = 0; i < 8; ++i) at[i] = Frag::get_coord(i);
    const uint item = threadgroup_position_in_grid.x * NSG + sg;
    if (item < ITEMS) {
        MIX_ITEM(weights[((v * W + r) * G + g) * S + s])
    }
"""

def _header():
    from .read import _SHARE

    text, _why = library()
    return text + "\nusing namespace mlx::steel;\n" + _SHARE + _MIX_ITEM


@lru_cache(maxsize=None)
def _kernel(kind):
    if kind == "mix":
        return mx.fast.metal_kernel(name="mtplx_inkling_nax_mix", input_names=["weights", "keys"],
                                    output_names=["out"], source=_MIX, header=_header())
    store, tail = _FUSED_END if kind == "fused" else _WEIGHTS_END
    return mx.fast.metal_kernel(
        name="mtplx_inkling_nax_read" if kind == "fused" else "mtplx_inkling_nax_weights",
        input_names=["q", "keys"],
        output_names=["out"] if kind == "fused" else ["weights"],
        source=_SCORES.replace("STORE_WEIGHT", store).replace("TAIL", tail),
        header=_header(),
    )


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


def plan(nq, columns, width):
    """``(partitions, partition size)`` of the library's split-K product ``(nq, width) @ (columns,
    width)^T`` in float32 on this Mac, read from its own dispatch; None when it is not that product
    or cannot be read now (a recording is running, or no engine)."""
    key = (nq, columns, width)
    if key in _PLAN:
        return _PLAN[key]
    try:
        from mtplx.engine import runtime as engine_runtime

        engine = engine_runtime.engine()
    except Exception:  # noqa: BLE001 - no engine: the tried plan and the proof decide
        return None
    if engine is None or engine.recording():
        return None
    a, b = mx.zeros((nq, width), mx.float32), mx.zeros((columns, width), mx.float32)
    mx.eval(a, b)
    engine.record_begin(label="inkling.nax_reads.plan")
    try:
        c = a @ b.T
        mx.eval(c)
    finally:
        trace = engine.record_end()
    found = None
    for step in trace.exported()["steps"]:
        if str(step[1]).startswith(_SPLIT_K):
            for slot, data in step[5]:
                if int(slot) == 3 and len(bytes(data)) >= 52:
                    values = struct.unpack("<13i", bytes(data)[:52])
                    found = (values[8], values[10])
    _PLAN[key] = found
    return found


#: Mix items (16-query block x 32 output columns) the scores launch takes itself; more go to a mix
#: launch across the grid.
FUSED_ITEMS = 320


def _usable(nq, groups, count_, width, dtypes):
    if not enabled():
        return "off"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if dtypes[0] not in (mx.float32, mx.bfloat16) or dtypes[1] != mx.float32:
        return "dtype"
    if width % 32 or count_ > 32 or nq % groups or groups > 16:
        return "shape"
    if library()[0] is None:
        return library()[1]
    return None


def _launch(q, keys, groups, partitions, size):
    nq, width = q.shape[0], q.shape[1]
    rows, count_ = keys.shape[1], keys.shape[2]
    visits = nq // groups
    items = visits * rows * (width // 32)
    template = [("V", visits), ("G", groups), ("S", count_), ("W", rows), ("D", width), ("NSG", SIMDGROUPS),
                ("GB", 1), ("TILES", width // 32), ("ITEMS", items)]
    scores = template + [("P", partitions), ("PS", size), ("SCALE", _f32_bits(width ** -0.5)), ("AHEAD", AHEAD)]
    tg = (SIMDGROUPS * 32, 1, 1)
    if items <= FUSED_ITEMS:
        return _kernel("fused")(inputs=[q, keys], template=scores, grid=tg, threadgroup=tg,
                                output_shapes=[(visits, rows, groups, width)], output_dtypes=[mx.float32])[0]
    weights = _kernel("weights")(inputs=[q, keys], template=scores, grid=tg, threadgroup=tg,
                                 output_shapes=[(visits, rows, groups, count_)], output_dtypes=[mx.float32])[0]
    groups_ = -(-items // SIMDGROUPS)
    return _kernel("mix")(inputs=[weights, keys], template=template, grid=(groups_ * SIMDGROUPS * 32, 1, 1),
                          threadgroup=tg, output_shapes=[(visits, rows, groups, width)],
                          output_dtypes=[mx.float32])[0]


def _read(kind, q, keys, groups, stock):
    """The launch where proven for this shape on this Mac, else ``stock()`` (counted)."""
    nq, width = q.shape
    rows, count_ = keys.shape[1], keys.shape[2]
    why = _usable(nq, groups, count_, width, (q.dtype, keys.dtype))
    if why is not None:
        if why != "off":
            count(f"fallback.nax_reads.{kind}_{why}")
        return stock()
    key = (kind, nq, groups, count_, rows, width, str(q.dtype))
    verdict = _VERDICT.get(key)
    if verdict is None:
        found = plan(nq, rows * count_, width)
        tried = found or (2, 2048)
        verdict = False
        if 4 * (tried[0] * nq * rows * count_ + rows * nq * count_) <= _TG_BYTES:
            try:
                got, want = _launch(q, keys, groups, *tried), stock()
                mx.eval(got, want)
                verdict = tried if _same(got.reshape(want.shape), want) else False
            except (RuntimeError, ValueError):
                verdict = False
        if found is None and not verdict:
            count(f"fallback.nax_reads.{kind}_undecided")
            return stock()  # the library's plan could not be read now: asked again on the next call
        _VERDICT[key] = verdict
        from .attn_input import count_route

        count_route(("proved:" if verdict else "refused:") + "nax_" + "_".join(map(str, key)))
    if not verdict:
        count(f"fallback.nax_reads.{kind}_differs")
        return stock()
    return _launch(q, keys, groups, *verdict)


def depth_read(queries, keys, stock):
    """RC6's three-query depth read from its keys: ``(1, W, Q, D)`` float32 in our launch where
    proven, else ``stock()`` (the library's products around ``read.probs``). ``queries`` ``(Q, D)``
    float32 or bf16 (read as float32 in the launch, as RC6's cast gives them); ``keys`` ``(1, W, S, D)``
    float32."""
    width = keys.shape[-1]
    q = queries.reshape(-1, width)
    return _read("depth", q, keys, q.shape[0], stock)


def message_read(table, stack, stock):
    """A loop pass's message reads: ``table`` ``(V, 4, D)`` float32 (V visits' queries), ``stack``
    ``(1, W, P, D)`` float32 (the pass's normalised prior states) -> ``(V, W, 4, D)`` float32 in our
    launches where proven, else ``stock()``."""
    visits, groups, width = table.shape
    return _read("message", table.reshape(visits * groups, width), stack, groups, stock)


__all__ = ["depth_read", "enabled", "library", "message_read", "plan"]
