"""The round's decision on the GPU: one small packet instead of full logit rows on the host.

A DSpark round verifies W = k + 1 rows (the anchor and k proposals). Today the host
copies every consumed 200,058-entry row to NumPy and draws it in FP64. This module
decides the whole round on the device and returns one int32 packet the host reads once.

Two rules, both the product's own laws (``mtplx/sampling.py``):

* ``ratio`` (class B, the default at temperature > 0): proposal ``x_i`` drawn from the
  draft law ``q_i`` is accepted with probability ``min(1, p_i(x_i) / q_i(x_i))``
  (``acceptance_probability``); at the first rejection the emitted token is drawn from
  ``max(p_i - q_i, 0)`` renormalised (``residual_distribution``), falling back to ``p_i``
  when that is empty, exactly as the product does; if every proposal is accepted the
  bonus token is drawn from the last row ``p_k``. Prompt-copy proposals are point masses:
  accepted with probability ``p(x)``, residual ``p`` without ``x``.
* ``greedy`` (temperature 0): accept while the proposal equals the target's argmax;
  emit the argmax at the first mismatch, or after the last row.

Sampling is an exponential race: ``argmax_j (y_j - log E_j)`` with ``E_j ~ Exp(1)``
independent picks ``j`` with probability ``softmax(y)_j`` exactly in real arithmetic.
Every random number is a pure function of (seed, absolute position, purpose, vocabulary
index) through Philox-4x32-10, so nothing is consumed by rejected or stopped rows and a
request is reproducible from its seed. In float32 the race keys are computed relative to
the row maximum, so the drawn law is within float rounding of ``softmax(y)`` (relative
error of order 1e-7 on the tokens that carry the mass); ``tests/test_inkling_fast_round*``
hold the law gates.

Laws are "law rows": float32 rows ``y`` with ``law = softmax(y)``. Target rows are the
verify logits divided by the temperature (the head already applies the soft cap). Draft
rows are the drafter's scores after its soft cap, divided by the draft temperature. Every
kernel that needs a law reads the same stored law row and the same stored log-sum-exp,
so the proposal draw, the acceptance test and the residual use one law, bit for bit.

Kernels are pure (``lanes/SIGNATURES.md``): they read rows and return fresh arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import secrets

import mlx.core as mx
import numpy as np

THREADS = 256
PER_THREAD = 8
CHUNK = THREADS * PER_THREAD  # vocabulary entries per threadgroup
MAX_ROWS = 8  # anchor + 7 proposals: the trained DSpark block

PURPOSE_DRAFT = 1
PURPOSE_ACCEPT = 2
PURPOSE_TARGET = 3

# Per-row modes of the race and the decision.
MODE_TARGET = 0  # draw from p (bonus row, plain decode, first token)
MODE_RESIDUAL = 1  # draw from max(p - q, 0) with q a draft law row; fall back to p
MODE_POINT = 2  # draw from p without the proposed token (point-mass proposal)

STATUS_OK = 0
STATUS_NON_FINITE = 1

_HEADER = r"""
#define L4_THREADS 256u
#define L4_M0 0xD2511F53u
#define L4_M1 0xCD9E8D57u
#define L4_W0 0x9E3779B9u
#define L4_W1 0xBB67AE85u

inline uint4 l4_philox(uint4 c, uint2 k) {
    for (int r = 0; r < 10; ++r) {
        uint hi0 = metal::mulhi(L4_M0, c.x);
        uint lo0 = L4_M0 * c.x;
        uint hi1 = metal::mulhi(L4_M1, c.z);
        uint lo1 = L4_M1 * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k += uint2(L4_W0, L4_W1);
    }
    return c;
}

// log1p for the small arguments the exponential needs (Kahan's form).
inline float l4_log1p(float x) {
    float u = 1.0f + x;
    return (u == 1.0f) ? x : metal::log(u) * (x / (u - 1.0f));
}

// expm1 by the same device (Kahan); accurate near zero.
inline float l4_expm1(float x) {
    float u = metal::exp(x);
    if (u == 1.0f) return x;
    float um1 = u - 1.0f;
    if (um1 == -1.0f) return -1.0f;
    return um1 * x / metal::log(u);
}

// E = -log(W), W = (hi * 2^32 + lo + 0.5) / 2^64 uniform on (0, 1): an Exp(1) sample,
// accurate to a few ulps everywhere, including W near 1 where E is tiny.
inline float l4_exponential(uint hi, uint lo) {
    if (hi >= 0x80000000u) {
        float om = metal::fma(float(~hi), 0x1p-32f, (float(~lo) + 0.5f) * 0x1p-64f);
        return -l4_log1p(-om);
    }
    float w = metal::fma(float(hi), 0x1p-32f, (float(lo) + 0.5f) * 0x1p-64f);
    return -metal::log(w);
}

inline float l4_log_exponential(uint2 key, uint j, uint position, uint purpose) {
    uint4 r = l4_philox(uint4(j, position, purpose, 0u), key);
    return metal::log(l4_exponential(r.x, r.y));
}

// (m, s) online log-sum-exp pairs: s = sum exp(y - m).
inline float2 l4_lse_merge(float2 a, float2 b) {
    if (a.x == -INFINITY) return b;
    if (b.x == -INFINITY) return a;
    float m = metal::max(a.x, b.x);
    return float2(m, a.y * metal::exp(a.x - m) + b.y * metal::exp(b.x - m));
}

// Race candidates: higher key wins, equal keys go to the lower index; index -1 is empty.
inline bool l4_better(float k1, int i1, float k2, int i2) {
    if (i2 < 0) return i1 >= 0;
    if (i1 < 0) return false;
    if (k1 != k2) return k1 > k2;
    return i1 < i2;
}
"""

# Stats: optional law-row write (y = cap(x) * scale) and per-chunk (max, sumexp).
_STATS = r"""
    uint row = threadgroup_position_in_grid.y;
    uint chunk = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint V = sizes[0];
    uint C = sizes[1];
    float scale = params[2 * row];
    float cap = params[2 * row + 1];
    float2 acc = float2(-INFINITY, 0.0f);
    for (uint e = 0; e < PER_THREAD; ++e) {
        uint j = chunk * (THREADS * PER_THREAD) + e * THREADS + tid;
        if (j < V) {
            float x = float(scores[row * V + j]);
            float y = x;
            if (cap > 0.0f) y = metal::precise::tanh(y / cap) * cap;
            y = y * scale;
            if (WRITE) law[row * V + j] = y;
            if (y == -INFINITY) continue;
            if (!(y <= acc.x)) {          // also takes NaN and +inf into the max
                acc = float2(y, (acc.x == -INFINITY ? 0.0f : acc.y * metal::exp(acc.x - y)) + 1.0f);
            } else {
                acc.y += metal::exp(y - acc.x);
            }
        }
    }
    for (uint d = 16; d > 0; d >>= 1) {
        float2 other = float2(simd_shuffle_down(acc.x, d), simd_shuffle_down(acc.y, d));
        if (lane + d < 32) acc = l4_lse_merge(acc, other);
    }
    threadgroup float2 part[THREADS / 32];
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float2 total = part[0];
        for (uint g = 1; g < THREADS / 32; ++g) total = l4_lse_merge(total, part[g]);
        stats[(row * C + chunk) * 2] = total.x;
        stats[(row * C + chunk) * 2 + 1] = total.y;
    }
"""

# Reduce one row's chunk stats to (max, log-sum), by one fixed tree over a whole threadgroup.
# The log-sum is log(sum exp(y - max)), kept apart from the maximum: a log-probability is
# (y - max) - logsum, never y - (max + logsum). At a large scale (a small temperature) the
# maximum dwarfs the log-sum and float32 would round max + logsum back to max, turning a
# 50% token into a certain one (review 1, finding 4).
_ROW_LSE = r"""
template <typename P>
inline float2 l4_row_lse(P stats, uint row, uint C, uint tid,
                         uint lane, uint sg, threadgroup float2* scratch) {
    float2 acc = float2(-INFINITY, 0.0f);
    for (uint c = tid; c < C; c += L4_THREADS) {
        acc = l4_lse_merge(acc, float2(stats[(row * C + c) * 2], stats[(row * C + c) * 2 + 1]));
    }
    for (uint d = 16; d > 0; d >>= 1) {
        float2 other = float2(simd_shuffle_down(acc.x, d), simd_shuffle_down(acc.y, d));
        if (lane + d < 32) acc = l4_lse_merge(acc, other);
    }
    if (lane == 0) scratch[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 total = scratch[0];
    for (uint g = 1; g < L4_THREADS / 32; ++g) total = l4_lse_merge(total, scratch[g]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (max, log-sum relative to it); a non-finite max propagates through total.y
    return float2(total.x, metal::log(total.y));
}
"""

# Race partials per (row, chunk). Rows of the target law ``yt`` and, for residual rows,
# the draft law ``yd`` (row i of yd belongs to row i of yt). Each threadgroup recomputes
# its row's (max, lse) by the shared fixed tree; chunk 0 publishes them for the decision.
_RACE = r"""
    uint row = threadgroup_position_in_grid.y;
    uint chunk = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint V = sizes[0];
    uint C = sizes[1];
    uint K = sizes[2];               // rows that carry a proposal
    int mode = modes[row];
    uint2 key = uint2(seed[0], seed[1]);
    uint position = positions[row];
    threadgroup float2 scratch[THREADS / 32];
    float2 t = l4_row_lse(tstats, row, C, tid, lane, sg, scratch);
    float2 d = float2(0.0f, 0.0f);
    bool residual = (mode == 1) && (row < K);
    if (residual) d = l4_row_lse(dstats, row, C, tid, lane, sg, scratch);
    if (chunk == 0 && tid == 0) {
        lse_out[row * 4 + 0] = t.x;
        lse_out[row * 4 + 1] = t.y;
        lse_out[row * 4 + 2] = d.x;
        lse_out[row * 4 + 3] = d.y;
    }
    int proposal = (row < K) ? proposals[row] : -1;
    float best_r = -INFINITY, best_p = -INFINITY;
    int idx_r = -1, idx_p = -1;
    for (uint e = 0; e < PER_THREAD; ++e) {
        uint j = chunk * (THREADS * PER_THREAD) + e * THREADS + tid;
        if (j >= V) break;
        float yt = tlaw[row * V + j];
        if (yt == -INFINITY) continue;
        float kp;
        float g = 0.0f;
        if (GREEDY) {
            kp = yt;
        } else {
            g = l4_log_exponential(key, j, position, PURPOSE);
            kp = (yt - t.x) - g;     // relative to the row maximum
        }
        if (l4_better(kp, int(j), best_p, idx_p)) { best_p = kp; idx_p = int(j); }
        if (GREEDY) continue;
        float kr = -INFINITY;
        if (residual) {
            float lp = (yt - t.x) - t.y;
            float lq = (dlaw[row * V + j] - d.x) - d.y;
            if (lq < lp) kr = lp + metal::log(-l4_expm1(lq - lp)) - g;
        } else if (mode == 2 && row < K) {
            if (int(j) != proposal) kr = ((yt - t.x) - t.y) - g;
        }
        if (kr > -INFINITY && l4_better(kr, int(j), best_r, idx_r)) { best_r = kr; idx_r = int(j); }
    }
    // Reduce candidates over the threadgroup: lane tree, then the 8 simdgroups in order.
    for (uint s = 16; s > 0; s >>= 1) {
        float o_r = simd_shuffle_down(best_r, s); int oi_r = simd_shuffle_down(idx_r, s);
        float o_p = simd_shuffle_down(best_p, s); int oi_p = simd_shuffle_down(idx_p, s);
        if (lane + s < 32) {
            if (l4_better(o_r, oi_r, best_r, idx_r)) { best_r = o_r; idx_r = oi_r; }
            if (l4_better(o_p, oi_p, best_p, idx_p)) { best_p = o_p; idx_p = oi_p; }
        }
    }
    threadgroup float keys_r[THREADS / 32];
    threadgroup float keys_p[THREADS / 32];
    threadgroup int ids_r[THREADS / 32];
    threadgroup int ids_p[THREADS / 32];
    if (lane == 0) { keys_r[sg] = best_r; ids_r[sg] = idx_r; keys_p[sg] = best_p; ids_p[sg] = idx_p; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float br = keys_r[0], bp = keys_p[0];
        int ir = ids_r[0], ip = ids_p[0];
        for (uint g2 = 1; g2 < THREADS / 32; ++g2) {
            if (l4_better(keys_r[g2], ids_r[g2], br, ir)) { br = keys_r[g2]; ir = ids_r[g2]; }
            if (l4_better(keys_p[g2], ids_p[g2], bp, ip)) { bp = keys_p[g2]; ip = ids_p[g2]; }
        }
        uint o = (row * C + chunk) * 4;
        race[o + 0] = as_type<int>(br);
        race[o + 1] = ir;
        race[o + 2] = as_type<int>(bp);
        race[o + 3] = ip;
    }
"""

# Reduce one row's race partials (over C chunks) inside one simdgroup.
_ROW_WINNER = r"""
template <typename P>
inline int4 l4_row_winner(P race, uint row, uint C, uint lane) {
    float br = -INFINITY, bp = -INFINITY;
    int ir = -1, ip = -1;
    for (uint c = lane; c < C; c += 32) {
        uint o = (row * C + c) * 4;
        float kr = as_type<float>(race[o]); int i_r = race[o + 1];
        float kp = as_type<float>(race[o + 2]); int i_p = race[o + 3];
        if (l4_better(kr, i_r, br, ir)) { br = kr; ir = i_r; }
        if (l4_better(kp, i_p, bp, ip)) { bp = kp; ip = i_p; }
    }
    for (uint s = 16; s > 0; s >>= 1) {
        float o_r = simd_shuffle_down(br, s); int oi_r = simd_shuffle_down(ir, s);
        float o_p = simd_shuffle_down(bp, s); int oi_p = simd_shuffle_down(ip, s);
        if (lane + s < 32) {
            if (l4_better(o_r, oi_r, br, ir)) { br = o_r; ir = oi_r; }
            if (l4_better(o_p, oi_p, bp, ip)) { bp = o_p; ip = oi_p; }
        }
    }
    return int4(simd_broadcast_first(ir), simd_broadcast_first(ip), 0, 0);
}
"""

# The decision: one threadgroup, simdgroup g reduces row g. Packet layout (int32):
#   [0] accepted proposals n   [1] status   [2 .. 2+W) emitted tokens (-1 after n)
#   [2+W .. 2+W+K) acceptance probability min(1, p/q) of each proposal, float bits
_DECIDE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint V = sizes[0];
    uint C = sizes[1];
    uint K = metal::min(sizes[2], offered[0]);   // proposals this round verifies
    uint W = sizes[3];
    uint2 key = uint2(seed[0], seed[1]);
    threadgroup int win_r[8];
    threadgroup int win_p[8];
    threadgroup int accept[8];
    threadgroup float aprob[8];
    threadgroup int bad[8];
    if (sg < W) {
        int4 w = l4_row_winner(race, sg, C, lane);
        if (lane == 0) {
            win_r[sg] = w.x;
            win_p[sg] = w.y;
            float tmax = lse[sg * 4 + 0], tls = lse[sg * 4 + 1];
            bad[sg] = (metal::isfinite(tmax) && metal::isfinite(tls)) ? 0 : 1;
            accept[sg] = 0;
            aprob[sg] = 0.0f;
            if (sg < K) {
                int x = proposals[sg];
                int mode = modes[sg];
                if (GREEDY) {
                    accept[sg] = (x == w.y) ? 1 : 0;
                    aprob[sg] = float(accept[sg]);
                } else if (MATCH) {
                    // seed-match: the target draws its own token; a proposal survives on equality
                    accept[sg] = (x == w.y) ? 1 : 0;
                    aprob[sg] = (x >= 0 && uint(x) < V) ? metal::exp((tlaw[sg * V + uint(x)] - tmax) - tls) : 0.0f;
                } else if (x >= 0 && uint(x) < V) {
                    float lp = (tlaw[sg * V + uint(x)] - tmax) - tls;
                    float lq = 0.0f;
                    if (mode == 1) {
                        float dmax = lse[sg * 4 + 2], dls = lse[sg * 4 + 3];
                        if (!(metal::isfinite(dmax) && metal::isfinite(dls))) bad[sg] = 1;
                        lq = (dlaw[sg * V + uint(x)] - dmax) - dls;
                    }
                    float diff = lp - lq;          // log(p/q)
                    aprob[sg] = diff >= 0.0f ? 1.0f : metal::exp(diff);
                    if (lq == -INFINITY) {
                        accept[sg] = (lp > -INFINITY) ? 1 : 0;   // q = 0: accept iff p > 0
                    } else if (diff >= 0.0f) {
                        accept[sg] = 1;
                    } else {
                        uint4 r = l4_philox(uint4(0u, positions[sg], 2u, 0u), key);
                        accept[sg] = (l4_exponential(r.x, r.y) > -diff) ? 1 : 0;   // u < p/q
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int status = 0;
        for (uint g = 0; g < W; ++g) if (bad[g]) status = 1;
        uint n = 0;
        while (n < K && accept[n]) ++n;
        int token;
        if (n < K) {
            token = (GREEDY || MATCH) ? win_p[n] : (win_r[n] >= 0 ? win_r[n] : win_p[n]);
        } else {
            token = win_p[n];
        }
        packet[0] = int(n);
        packet[1] = status;
        for (uint g = 0; g < W; ++g) {
            packet[2 + g] = g < n ? proposals[g] : (g == n ? token : -1);
        }
        for (uint g = 0; g < sizes[2]; ++g) packet[2 + W + g] = as_type<int>(g < K ? aprob[g] : 0.0f);
    }
"""


# Winner of each row's p race (one simdgroup per row).
_FINISH = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint C = sizes[1];
    int4 w = l4_row_winner(race, row, C, lane);
    if (lane == 0) winners[row] = w.y;
"""


def _chunks(vocab):
    return -(-vocab // CHUNK)


_CONSTANTS: dict = {}


def _const(values, dtype):
    """A small array of fixed values the round's kernels read (sizes, per-row parameters,
    modes), made once per value and kept for the process.

    A fresh array per call is correct for MLX but not stable for a recording: MLX may hand
    it a recycled buffer longer than the array, whose bytes past the array's end are left
    from earlier work, and a recording snapshots the whole buffer, so two recordings of the
    same round would differ and the native engine would (rightly) refuse the program. A kept
    array lives in one buffer for good, and the replay sites pin it (``constants``)."""
    key = (repr(values), str(dtype))
    array = _CONSTANTS.get(key)
    if array is None:
        array = mx.array(values, dtype)
        mx.eval(array)
        _CONSTANTS[key] = array
    return array


def constants():
    """Every kept constant of this module (pinned by a replay site before each recording)."""
    return list(_CONSTANTS.values())


@lru_cache(maxsize=None)
def _kernel(name):
    if name == "stats":
        return mx.fast.metal_kernel(
            name="mtplx_inkling_l4_stats",
            input_names=["scores", "params", "sizes"],
            output_names=["law", "stats"],
            source=_STATS,
            header=_HEADER,
        )
    if name == "race":
        return mx.fast.metal_kernel(
            name="mtplx_inkling_l4_race",
            input_names=["tlaw", "tstats", "dlaw", "dstats", "proposals", "modes",
                         "positions", "seed", "sizes"],
            output_names=["race", "lse_out"],
            source=_RACE,
            header=_HEADER + _ROW_LSE,
        )
    if name == "finish":
        return mx.fast.metal_kernel(
            name="mtplx_inkling_l4_finish",
            input_names=["race", "sizes"],
            output_names=["winners"],
            source=_FINISH,
            header=_HEADER + _ROW_WINNER,
        )
    if name == "decide":
        return mx.fast.metal_kernel(
            name="mtplx_inkling_l4_decide",
            input_names=["race", "lse", "tlaw", "dlaw", "proposals", "modes", "positions",
                         "seed", "sizes", "offered"],
            output_names=["packet"],
            source=_DECIDE,
            header=_HEADER + _ROW_WINNER,
        )
    raise KeyError(name)


def law_params(rows, temperature=1.0, cap=0.0):
    """The per-row law parameters ``law_rows`` keeps for ``temperature`` and ``cap``, as a
    fresh array: an input of a recorded round, so one program serves every temperature."""
    scale = 1.0 if not temperature or temperature <= 0 else 1.0 / float(temperature)
    return mx.array([v for _ in range(rows) for v in (scale, float(cap or 0.0))], mx.float32)


def law_rows(scores, *, temperature=1.0, cap=0.0, write=None, params=None):
    """Law rows ``y = cap(scores) / temperature`` (float32) and their chunk stats.

    ``scores`` is ``(R, V)`` in any float dtype. When no transform is needed and the input
    is already float32 the input itself is the law row (no copy). ``params`` (from
    ``law_params``) replaces ``temperature`` and ``cap`` with an array read at run time; the
    law is then always written (at scale 1 and no cap the written row equals the input).
    """
    if scores.ndim != 2:
        raise ValueError("law_rows takes (rows, vocabulary)")
    rows, vocab = scores.shape
    if params is not None:
        write = True if write is None else write
    else:
        scale = 1.0 if not temperature or temperature <= 0 else 1.0 / float(temperature)
        if write is None:
            write = not (scale == 1.0 and not cap and scores.dtype == mx.float32)
        params = _const([v for _ in range(rows) for v in (scale, float(cap or 0.0))], mx.float32)
    sizes = _const([vocab, _chunks(vocab)], mx.uint32)
    law, stats = _kernel("stats")(
        inputs=[scores, params, sizes],
        template=[("THREADS", THREADS), ("PER_THREAD", PER_THREAD), ("WRITE", bool(write))],
        grid=(_chunks(vocab) * THREADS, rows, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, vocab) if write else (1,), (rows, _chunks(vocab), 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return (law if write else scores.astype(mx.float32)), stats


def _seed_words(seed):
    if seed is None:
        seed = secrets.randbits(64)
    seed = int(seed) & ((1 << 64) - 1)
    return mx.array([seed & 0xFFFFFFFF, seed >> 32], mx.uint32)


@dataclass(frozen=True)
class DraftLaw:
    """The law the proposals were drawn from: law rows, their stats, and how drawn."""

    rows: mx.array  # (k, V) float32 law rows, or None for point-mass proposals
    stats: mx.array  # (k, C, 2) chunk stats of ``rows``
    point_mass: bool = False
    log_q: mx.array | None = None  # (k,) log q_i(x_i) of each drawn proposal (draft confidence)


def race(tlaw, tstats, *, modes, positions, seed, purpose, greedy=False, dlaw=None,
         dstats=None, proposals=None):
    """Race partials and per-row (max, lse) for a panel of law rows (pure, lazy)."""
    rows, vocab = tlaw.shape
    chunks = _chunks(vocab)
    k = 0 if proposals is None else int(proposals.shape[0])
    if dlaw is None:
        dlaw, dstats = _const([0.0], mx.float32), _const([[[0.0, 0.0]]], mx.float32)
    if proposals is None:
        proposals = _const([0], mx.int32)
    sizes = _const([vocab, chunks, k], mx.uint32)
    return _kernel("race")(
        inputs=[tlaw, tstats, dlaw, dstats, proposals.astype(mx.int32), modes, positions, seed,
                sizes],
        template=[("THREADS", THREADS), ("PER_THREAD", PER_THREAD), ("GREEDY", bool(greedy)),
                  ("PURPOSE", int(purpose))],
        grid=(chunks * THREADS, rows, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, chunks, 4), (rows, 4)],
        output_dtypes=[mx.int32, mx.float32],
    )


def decide(target_logits, proposals, *, position, seed, temperature=1.0, draft=None,
           greedy=None, rule="ratio", offered=None, params=None):
    """Decide one verify panel on the GPU and return the int32 packet (lazy).

    ``target_logits`` (W, V) are the verify rows as the head returns them (soft cap
    applied); row i is the target law after the anchor and the first i proposals.
    ``proposals`` (k,) int32 with k = W - 1, ``draft`` their law (``DraftLaw``), or None
    for point-mass proposals (prompt copy). ``position`` is the absolute position of the
    token row 0 decides (an int or a uint32 array). ``temperature`` 0 selects greedy.
    ``offered`` (an int or a uint32 array, possibly lazy) limits the round to its first
    ``offered`` proposals: rows past it are ignored and the bonus comes from row ``offered``.
    It lets a fixed-width panel carry a depth chosen on the device (``draft.choose_depth``).
    ``rule`` "match" is the seed-match rule on the device: every row draws its own token
    from p and a proposal survives only on equality (the law RC6's target_prefix keeps;
    the random numbers are this module's, not RC6's NumPy stream). ``params`` (from
    ``law_params(W, temperature)``) gives the target law's temperature as an array.
    """
    if rule not in ("ratio", "match"):
        raise ValueError(f"unknown acceptance rule {rule!r}")
    if target_logits.ndim != 2:
        raise ValueError("decide takes (rows, vocabulary) target logits")
    rows, vocab = target_logits.shape
    k = int(proposals.shape[0]) if proposals is not None else 0
    if rows != k + 1 or not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"a panel of {rows} rows needs {rows - 1} proposals (at most {MAX_ROWS})")
    greedy = (not temperature or temperature <= 0) if greedy is None else greedy
    tlaw, tstats = law_rows(target_logits, temperature=None if greedy else temperature,
                            params=None if greedy else params)
    match = rule == "match" and not greedy
    point = draft is None or draft.point_mass or match
    mode = MODE_TARGET if match else (MODE_POINT if point else MODE_RESIDUAL)
    modes = _const([mode] * k + [MODE_TARGET], mx.int32)
    base = position if isinstance(position, mx.array) else mx.array(int(position), mx.uint32)
    positions = base.astype(mx.uint32).reshape(()) + mx.arange(rows, dtype=mx.uint32)
    seed_words = seed if isinstance(seed, mx.array) else _seed_words(seed)
    props = proposals if proposals is not None else _const([], mx.int32)
    dlaw = None if point else draft.rows
    dstats = None if point else draft.stats
    race_parts, lse = race(tlaw, tstats, modes=modes, positions=positions, seed=seed_words,
                           purpose=PURPOSE_TARGET, greedy=greedy, dlaw=dlaw, dstats=dstats,
                           proposals=props if k else None)
    sizes = _const([vocab, _chunks(vocab), k, rows], mx.uint32)
    if offered is None:
        offered = _const([k], mx.uint32)
    elif not isinstance(offered, mx.array):
        offered = _const([int(offered)], mx.uint32)
    offered = offered.astype(mx.uint32).reshape(1)
    (packet,) = _kernel("decide")(
        inputs=[race_parts, lse, tlaw, dlaw if dlaw is not None else _const([0.0], mx.float32),
                props.astype(mx.int32) if k else _const([0], mx.int32), modes, positions,
                seed_words, sizes, offered],
        template=[("GREEDY", bool(greedy)), ("MATCH", bool(match))],
        grid=(THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(2 + rows + k,)],
        output_dtypes=[mx.int32],
    )
    return packet


def commit_count(packet, rows, *, stops, remaining):
    """The generation loop's walk over a packet, as device data (pure, lazy, no host read).

    For a round whose commit runs before the host reads the packet (the whole round as one
    program): ``stops`` the request's stop ids ((S,) int32 array or a list; request constant),
    ``remaining`` the tokens the request may still emit (max_tokens - len(tokens), at least 1;
    an int or a uint32 array). Returns (kept, emitted) as uint32 scalars:
    ``emitted = min(n + 1, remaining, first stop + 1)`` tokens of the packet, with ``n`` the
    accepted proposals, and ``kept = 1 + min(n, emitted)`` verify rows to commit (the anchor
    and the emitted proposals), exactly what the loop computes after reading the packet.
    The next round's anchor is pending when ``emitted == n + 1``. Cancellation and the
    opt-in repetition guard stay host decisions (see ``lanes/L4.md``, "For L7 and L0").
    """
    # Every scalar is a kept constant and every operation int32: a Python scalar here would be
    # a fresh host array per call (a recycled buffer's stale bytes past it make two recordings
    # of the round differ), and argmax's uint32 against int32 would promote to int64.
    one = _const(1, mx.int32)
    n = packet[0].astype(mx.int32)
    tokens = packet[2 : 2 + rows]
    index = mx.arange(rows, dtype=mx.int32)
    if not isinstance(stops, mx.array):
        stops = mx.array([int(t) for t in stops] or [-2], mx.int32)
    hit = mx.any(tokens[:, None] == stops.astype(mx.int32)[None, :], axis=1) & (index <= n)
    first = mx.where(mx.any(hit), mx.argmax(hit.astype(mx.int32)).astype(mx.int32), _const(rows, mx.int32))
    if not isinstance(remaining, mx.array):
        remaining = mx.array(int(remaining), mx.int32)
    emitted = mx.minimum(mx.minimum(n + one, remaining.astype(mx.int32).reshape(())), first + one)
    kept = one + mx.minimum(n, emitted)
    return kept.astype(mx.uint32), emitted.astype(mx.uint32)


def winners(race_parts):
    """The p-race winner of every row: (R,) int32."""
    rows, chunks, _ = race_parts.shape
    (out,) = _kernel("finish")(
        inputs=[race_parts, _const([0, chunks], mx.uint32)],
        template=[("THREADS", THREADS)],
        grid=(32 * rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
    )
    return out


def sample_rows(law, stats, *, positions, seed, purpose, greedy=False):
    """One independent draw per law row (row r at ``positions[r]``): (R,) int32 and the
    per-row (max, log-sum) the draw used, (R, 4) float32 with columns [max, log-sum, 0, 0];
    the log-probability of token j is ``(law[j] - max) - logsum``."""
    rows = law.shape[0]
    modes = _const([0] * rows, mx.int32)
    seed_words = seed if isinstance(seed, mx.array) else _seed_words(seed)
    race_parts, lse = race(law, stats, modes=modes, positions=positions.astype(mx.uint32),
                           seed=seed_words, purpose=purpose, greedy=greedy)
    return winners(race_parts), lse


def draw(target_logits, *, position, seed, temperature=1.0):
    """The tokens of ``draw_checked`` alone, (W,) int32: kept for callers that check the
    rows themselves. The product's loop calls ``draw_checked`` and ``read_draw``, which
    refuse a row that is not a law instead of streaming its meaningless token."""
    return draw_checked(target_logits, position=position, seed=seed, temperature=temperature)[0]


def draw_checked(target_logits, *, position, seed, temperature=1.0):
    """Draw one token per row from the target law (plain decode, the first token after a
    prefill or a restore): (2, W) int32, row 0 the tokens, row 1 each row's status
    (``STATUS_OK``, or ``STATUS_NON_FINITE`` for a row with a NaN or no finite maximum,
    whose token is meaningless). One read gives both; ``read_draw`` raises the product's
    numerical error on a bad row, before the token is counted, streamed or banked
    (review 1, finding 3).

    Row r is the token at absolute position ``position + r``.
    """
    rows, vocab = target_logits.shape
    greedy = not temperature or temperature <= 0
    tlaw, tstats = law_rows(target_logits, temperature=None if greedy else temperature)
    base = position if isinstance(position, mx.array) else mx.array(int(position), mx.uint32)
    positions = base.astype(mx.uint32).reshape(()) + mx.arange(rows, dtype=mx.uint32)
    tokens, stats = sample_rows(tlaw, tstats, positions=positions, seed=seed,
                                purpose=PURPOSE_TARGET, greedy=greedy)
    good = mx.isfinite(stats[:, 0]) & mx.isfinite(stats[:, 1])
    status = mx.where(good, STATUS_OK, STATUS_NON_FINITE).astype(mx.int32)
    return mx.stack([tokens.astype(mx.int32), status])


def read_draw(drawn, rows=None, settle=None):
    """The tokens of a ``draw_checked`` (one read), or the product's ``NonFiniteLogitsError``
    when a row was not a law (``rows`` are the logits it drew from, for the error's census).
    ``settle`` is the owner's eval (keeps the stall watchdog ticking)."""
    if settle is not None:
        settle(drawn)
    values = np.asarray(drawn)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("read_draw takes the (2, W) result of draw_checked")
    if (values[1] != STATUS_OK).any():
        from mtplx.sampling import non_finite_logits_error

        bad = int(np.flatnonzero(values[1] != STATUS_OK)[0])
        row = np.asarray(rows[bad].astype(mx.float32)) if rows is not None else np.array([np.nan])
        raise non_finite_logits_error(row, "device draw")
    return [int(t) for t in values[0]]


def warm(vocab=4096):
    """Build every kernel variant the round uses on a small vocabulary, so the first request
    after a model load does not pay the Metal compiles (call once at load; about a second)."""
    rows = mx.zeros((4, vocab), mx.float32)
    draft_rows = mx.zeros((3, vocab), mx.bfloat16)
    seed = _seed_words(1)
    props = mx.array([1, 2, 3], mx.int32)
    law, stats = law_rows(draft_rows, temperature=1.0, cap=30.0)
    greedy_law, greedy_stats = law_rows(draft_rows, temperature=None, cap=0.0, write=True)
    outs = [
        sample_rows(law, stats, positions=mx.arange(3, dtype=mx.uint32), seed=seed, purpose=PURPOSE_DRAFT)[0],
        sample_rows(greedy_law, greedy_stats, positions=mx.arange(3, dtype=mx.uint32), seed=seed,
                    purpose=PURPOSE_DRAFT, greedy=True)[0],
        decide(rows, props, position=0, seed=seed, draft=DraftLaw(law, stats)),
        decide(rows, props, position=0, seed=seed),
        decide(rows, props, position=0, seed=seed, rule="match"),
        decide(rows, props, position=0, seed=seed, temperature=0),
        draw(rows[:1], position=0, seed=seed),
        draw(rows[:1], position=0, seed=seed, temperature=0),
    ]
    mx.eval(outs)


def supported(sampler, *, constraint=None, draft_sampler=None):
    """(True, None) when the device round serves this request, else (False, reason).

    The device round implements the tempered softmax over the full vocabulary (any
    temperature > 0, or greedy at 0) with no top-p, top-k or penalties, and no grammar
    mask. Everything else keeps the host path, counted under the returned reason:
    those laws depend on the accepted history or need a sort the round does not do.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False, "device"
    if constraint is not None:
        return False, "constraint"
    if getattr(sampler, "presence_penalty", 0) or getattr(sampler, "frequency_penalty", 0):
        return False, "penalties"
    temperature = float(getattr(sampler, "temperature", 1.0) or 0.0)
    if not temperature_served(temperature):
        return False, "temperature_range"
    if temperature > 0 and (float(getattr(sampler, "top_p", 1.0)) < 1.0
                            or int(getattr(sampler, "top_k", 0) or 0) > 0):
        return False, "filtered_sampler"
    if draft_sampler is not None and (
        float(getattr(draft_sampler, "top_p", 1.0)) < 1.0
        or int(getattr(draft_sampler, "top_k", 0) or 0) > 0
        or getattr(draft_sampler, "presence_penalty", 0)
        or getattr(draft_sampler, "frequency_penalty", 0)
    ):
        return False, "filtered_draft_sampler"
    if draft_sampler is not None and not temperature_served(float(getattr(draft_sampler, "temperature", 0.0) or 0.0)):
        return False, "draft_temperature_range"
    return True, None


#: Temperatures the device round serves besides 0 (greedy). Outside it the law rows' scale
#: is so large that float32 spacing at the scaled maximum is a visible share of the gaps
#: between logits, so those requests keep the host's float64 decision, counted.
TEMPERATURE_RANGE = (0.05, 2.0)


def temperature_served(temperature):
    t = float(temperature or 0.0)
    return t <= 0.0 or TEMPERATURE_RANGE[0] <= t <= TEMPERATURE_RANGE[1]


@dataclass(frozen=True)
class Decision:
    """The host's view of one packet."""

    accepted: int
    status: int
    tokens: list
    accept_probability: list


def read_packet(packet, rows, settle=None):
    """The round's only read. ``settle`` is the owner's eval (keeps the watchdog ticking)."""
    if settle is not None:
        settle(packet)
    values = np.asarray(packet)
    n = int(values[0])
    k = rows - 1
    probs = values[2 + rows: 2 + rows + k].view(np.float32).tolist()
    return Decision(n, int(values[1]), [int(t) for t in values[2: 3 + n]], probs)


# ---------------------------------------------------------------- host reference (tests)

_M0, _M1, _W0, _W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85


def philox_np(c0, c1, c2, c3, k0, k1):
    """Philox-4x32-10 on uint32 numpy arrays (broadcast), the kernels' generator."""
    c0, c1, c2, c3 = (np.asarray(v, np.uint64) & 0xFFFFFFFF for v in (c0, c1, c2, c3))
    k0, k1 = np.uint64(k0 & 0xFFFFFFFF), np.uint64(k1 & 0xFFFFFFFF)
    mask = np.uint64(0xFFFFFFFF)
    for _ in range(10):
        p0 = np.uint64(_M0) * c0
        p1 = np.uint64(_M1) * c2
        hi0, lo0 = p0 >> np.uint64(32), p0 & mask
        hi1, lo1 = p1 >> np.uint64(32), p1 & mask
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & mask, lo1, (hi0 ^ c3 ^ k1) & mask, lo0
        k0 = (k0 + np.uint64(_W0)) & mask
        k1 = (k1 + np.uint64(_W1)) & mask
    return c0.astype(np.uint32), c1.astype(np.uint32), c2.astype(np.uint32), c3.astype(np.uint32)


def exponential_np(hi, lo):
    """E = -log(W), W = (hi * 2^32 + lo + 0.5) / 2^64, in float64."""
    hi, lo = np.asarray(hi, np.uint32), np.asarray(lo, np.uint32)
    w = (hi.astype(np.float64) * 2.0**32 + lo.astype(np.float64) + 0.5) / 2.0**64
    om = ((~hi).astype(np.float64) * 2.0**32 + (~lo).astype(np.float64) + 0.5) / 2.0**64
    upper = hi >= 0x80000000
    out = np.empty(np.broadcast(w, om).shape, np.float64)
    out[...] = -np.log(np.where(upper, 0.5, w))
    out[upper] = -np.log1p(-np.broadcast_to(om, out.shape)[upper])
    return out


def seed_pair(seed):
    seed = int(seed) & ((1 << 64) - 1)
    return seed & 0xFFFFFFFF, seed >> 32


def log_exponentials_np(seed, position, purpose, vocab):
    k0, k1 = seed_pair(seed)
    j = np.arange(vocab, dtype=np.uint64)
    r = philox_np(j, position, purpose, 0, k0, k1)
    return np.log(exponential_np(r[0], r[1]))


def reference_decide(target_logits, proposals, *, position, seed, temperature=1.0,
                     draft_rows=None, greedy=None, rule="ratio"):
    """The decision in float64 NumPy with the kernels' random numbers.

    Uses ``mtplx.sampling.acceptance_probability`` and ``residual_distribution`` for the
    law; the race picks the same random numbers as the device, so device and host agree
    except where two race keys tie within float32 rounding.
    """
    from mtplx.sampling import acceptance_probability, residual_distribution

    logits = np.asarray(target_logits, np.float64)
    rows, vocab = logits.shape
    k = rows - 1
    greedy = (not temperature or temperature <= 0) if greedy is None else greedy
    k0, k1 = seed_pair(seed)
    out, probs = [], []
    if greedy:
        for i in range(rows):
            best = int(np.argmax(logits[i]))
            if i < k and int(proposals[i]) == best:
                out.append(best)
                probs.append(1.0)
                continue
            if i < k:
                probs.append(0.0)
            out.append(best)
            break
        return out, probs
    tlaw = logits / float(temperature)

    def softmax(y):
        z = np.exp(y - y.max())
        return z / z.sum()

    if rule == "match":
        for i in range(rows):
            p = softmax(tlaw[i])
            logs = log_exponentials_np(seed, position + i, PURPOSE_TARGET, vocab)
            with np.errstate(divide="ignore"):
                token = int(np.argmax(np.log(p) - logs))
            out.append(token)
            if i < k:
                probs.append(float(p[int(proposals[i])]))
                if token != int(proposals[i]):
                    break
        return out, probs
    for i in range(rows):
        p = softmax(tlaw[i])
        logs = log_exponentials_np(seed, position + i, PURPOSE_TARGET, vocab)
        if i < k:
            x = int(proposals[i])
            if draft_rows is None:
                q = np.zeros(vocab)
                q[x] = 1.0
            else:
                q = softmax(np.asarray(draft_rows[i], np.float64))
            a = acceptance_probability(p, q, x)
            probs.append(a)
            r = philox_np(0, position + i, PURPOSE_ACCEPT, 0, k0, k1)
            e = float(exponential_np(r[0], r[1]))
            u = float(np.exp(-e))
            if u < a or a >= 1.0:
                out.append(x)
                continue
            res = residual_distribution(p, q)
            with np.errstate(divide="ignore"):
                keys = np.log(res) - logs
            out.append(int(np.argmax(keys)))
            return out, probs
        with np.errstate(divide="ignore"):
            keys = np.log(p) - logs
        out.append(int(np.argmax(keys)))
    return out, probs
