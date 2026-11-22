"""Top-p, top-k and min-p for the round decided on the GPU: each law row's kept set, found on the device.

The device round (``round.py``) draws, accepts and resamples from law rows ``y`` with ``law =
softmax(y)``, skipping every entry that is ``-inf``. A client's top-p, top-k or min-p (a "filtered
sampler") is a law of the same kind: the tempered softmax restricted to a kept set and
renormalised. So the round serves it unchanged once each row's excluded entries are ``-inf``:
the race, the ratio test and the residual all read the masked row and its log-sum-exp, and the
ratio rule is exact for any target law (``round.py``'s docstring). This module finds the kept set.

The kept set is the product's own (``mtplx.sampling.apply_top_p_top_k`` on the float64 tempered
softmax, then ``min_p`` on the shaped law, the order ``generation.ClientControlledDraw`` documents).
Order the vocabulary by probability, highest first, exact ties by the lower id
(``deterministic_top_k_order``): top-k keeps the first ``k``; top-p keeps every entry whose mass
before it, in that order, is below ``top_p`` (so the entry that crosses ``top_p`` is kept); both keep
the shorter of the two prefixes; min-p keeps entries at or above ``max + T ln(min_p)``
(``ClientControlledDraw.rows``'s float32 floor). The device orders by score (the rows before the
temperature), which is the host's probability order: the select step refuses a cutoff where two distinct
scores could round to the same float64 probability, including an underflowing tail.

How, per row, with no sort of the vocabulary:

1. ``xmax``: each chunk's largest score.
2. ``bins``: each entry's distance below the row's largest score, in nats of the tempered law, in
   bins of an eighth of a nat (256 bins, 32 nats; farther entries share the last bin), and a
   float32 estimate of its mass as a 40-bit fixed-point integer, summed per chunk and bin with
   integer atomics (associative: the same sums every run).
3. ``cut``: per row, the bins summed and walked from the top: the bin of the k-th entry and the
   bins where the mass can cross ``top_p`` within the estimate's error (2**-17 of the total).
   Earlier bins are kept, later bins dropped; the entries of the boundary bins are the
   candidates (at most ``CAP``).
4. ``gather``: the candidates' scores and ids, and two exact sums per chunk: the mass of the whole
   row and of the entries above the candidates.
5. ``select``: the candidates sorted (score down, id up), their exact masses walked from the
   exact mass above them: the k-th, the top-p crossing, the shorter prefix, and its last entry,
   the boundary (score, id).
6. ``mask``: each law row with every entry after the boundary, or under the min-p floor, set to
   ``-inf``, and its chunk stats (max, sum-exp) as ``round.law_rows`` writes them.

"Exact" masses: ``exp(-(max - x) / T)`` in float-float arithmetic. Each lane sums these masses
before conversion to units of 2**-60. Chunk totals are summed in 21-bit limbs, so the cumulative masses
are within 2**-38 of the total of the true ones. The host's float64 masses are within
``(4 |max| / T + V + 512) * 2**-52`` of the total of the true ones (each entry's argument rounded
at the scale of ``|x| / T``, the running sum rounded once per entry). The select step certifies an
entry kept only when its mass before it is below ``top_p`` by more than both errors together, and
dropped only when above by as much; a crossing within that margin (about 2**-34 of the total on a
200,058-entry row; never met in 600 real rows of this model at top_p 0.9 to 0.99, where a margin of
2**-26 already decides every row), or boundary bins holding more than ``CAP`` entries, gets a status
instead of a decision and that round is decided on the host with the host's own float64 law
(counted ``decision.host.filter_boundary`` and ``decision.host.filter_overflow``). So every row
decided here has the host's kept set; within it the law is the device round's float32 law, as for
every other sampled round. Draft rows (a proposal law, which the ratio rule leaves free) use the
same kept set where it is decided and keep every entry where it is not.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from .. import launch_limits
from .round import CHUNK, PER_THREAD, THREADS, _chunks, _const

#: Bins of the distance below the row's largest score, and bins per nat of the tempered law.
BINS = 256
BINS_PER_NAT = 8
#: Candidates one row's select sorts in threadgroup memory (16 KB of 64-bit keys).
CAP = 2048
#: The float32 estimate's error, as a right shift of the row's total (2**-17 of it): each float32
#: mass is within about (1 + d) * 2**-21.4 of its value at distance d, and the masses weighted by
#: distance sum to at most ln V of the total. It only places the candidates; the select step
#: checks the placement with the exact masses.
COARSE_SHIFT = 17
#: Entries this many nats or more below the maximum weigh 0 in the exact sums (under 2**-62 each).
MASS_NATS = 43
#: The device's own error in the exact cumulative masses, as a share of the total (2**-38).
DEVICE_ERROR = 2.0 ** -38

STATUS_OK = 0
STATUS_BOUNDARY = 2  # the crossing lies within the error margin of top_p: the host decides this round
STATUS_GROUP = 4  # this GPU cannot run the filter pipeline
STATUS_OVERFLOW = 3  # the boundary bins hold more than CAP entries: the host decides this round

REFUSALS = {STATUS_BOUNDARY: "filter_boundary", STATUS_OVERFLOW: "filter_overflow",
            STATUS_GROUP: "filter_threadgroup"}


class FilterRefused(Exception):
    """A single target draw needs the host's filtered law."""

    def __init__(self, status):
        super().__init__(REFUSALS[status])
        self.reason = REFUSALS[status]


#: Filter words (``words``): top-p as a 64-bit fraction (low, high), top-p on, top-k, min-p on,
#: the min-p floor's float32 bits, 1/T as a float-float (high, low float32 bits).
WORDS = 8

_COMMON = (f"#define BINS {BINS}u\n#define BINS_PER_NAT {BINS_PER_NAT}u\n#define CAP {CAP}u\n"
           f"#define COARSE_SHIFT {COARSE_SHIFT}ul\n#define MASS_NATS {MASS_NATS}.0f\n") + r"""
inline float nk_key(float x, float cap) {
    if (cap > 0.0f) x = metal::precise::tanh(x / cap) * cap;
    return x;
}
// Order-preserving bits of a float32 (larger float, larger key; -inf below every finite value;
// -0 and +0 one key, as they are one probability on the host).
inline uint nk_order(float x) {
    uint b = as_type<uint>(x);
    if (b == 0x80000000u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
inline float nk_unorder(uint o) {
    return as_type<float>((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o);
}
template <typename P>
inline float nk_row_max(P part, uint row, uint C) {
    float m = -INFINITY;
    for (uint c = 0; c < C; ++c) m = metal::max(m, part[2 * (row * C + c)]);
    return m;
}
template <typename P>
inline bool nk_underflow_tail(P part, uint row, uint C, float xm, float scale) {
    float low = INFINITY;
    for (uint c = 0; c < C; ++c) low = metal::min(low, part[2 * (row * C + c) + 1]);
    return (xm - low) * scale > 680.0f;
}
inline uint nk_bin(float d) {
    return (d < float(BINS) / float(BINS_PER_NAT)) ? uint(d * float(BINS_PER_NAT)) : uint(BINS - 1);
}
// The float32 estimate of an entry's mass at distance d >= 0 below the row's largest score (2^40 per unit).
inline ulong nk_mass(float d) {
    return (d < 40.0f) ? ulong(metal::precise::exp(-d) * 0x1p40f) : 0ul;
}
// Float-float arithmetic: an unevaluated sum (hi, lo), IEEE float32 (the kernels' math mode is safe).
inline float2 nk_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}
inline float2 nk_fast_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}
inline float2 nk_ff_mul(float2 a, float2 b) {
    float p = a.x * b.x;
    float e = metal::fma(a.x, b.x, -p);
    e = metal::fma(a.x, b.y, e);
    e = metal::fma(a.y, b.x, e);
    return nk_fast_two_sum(p, e);
}
inline uint nk_clz64(ulong x) {
    uint h = uint(x >> 32);
    return h ? metal::clz(h) : 32u + metal::clz(uint(x));
}
inline float2 nk_etab(const device float* etab, uint i) {
    return float2(etab[2 * i], etab[2 * i + 1]);
}
// The exact mass of an entry: exp(-(xm - x) / T) to about 2^-44 of its value; entries MASS_NATS or more below the maximum weigh 0.
// d = n/256 + f with exp(-n/256) = exp(-floor(n/256)) exp(-(n mod 256)/256) from a float-float table
// and exp(-f), |f| < 2^-8, by its series to f^4 (the next term is under 2^-47).
inline float2 nk_exact_mass(float xm, float x, float ih, float il, const device float* etab) {
    float2 diff = nk_two_sum(xm, -x);
    float2 d = nk_ff_mul(diff, float2(ih, il));
    if (!(d.x < MASS_NATS)) return float2(0.0f);
    uint n = uint(metal::max(d.x, 0.0f) * 256.0f);
    float fh = d.x - float(n) * 0.00390625f;
    float2 g = nk_two_sum(fh, d.y);
    float2 one = nk_two_sum(1.0f, -g.x);
    one.y -= g.y;
    float f2h = g.x * g.x;
    float f2l = metal::fma(g.x, g.x, -f2h) + 2.0f * g.x * g.y;
    float delta = g.x * (g.x * (1.0f / 12.0f) - (1.0f / 3.0f));
    float ph = 0.5f * f2h;
    float pl = metal::fma(ph, delta, 0.5f * f2l);
    float2 q = nk_two_sum(one.x, ph);
    q.y += one.y + pl;
    float2 e = nk_ff_mul(nk_etab(etab, n >> 8), nk_etab(etab, 44u + (n & 255u)));
    return nk_ff_mul(e, q);
}
// Sum the eight masses a lane reads before conversion. Rounding every small
// token to an integer here would discard much of its relative precision.
inline float2 nk_add_mass(float2 a, float2 b) {
    float2 s = nk_two_sum(a.x, b.x);
    return nk_fast_two_sum(s.x, (a.y + b.y) + s.y);
}
inline ulong nk_units(float2 mass) {
    // Unsigned high limb: eight maximum masses in a lane total exactly 2^63.
    return ulong(mass.x * 0x1p60f) + ulong(long(mass.y * 0x1p60f));
}
"""

_XMAX = r"""
    uint row = threadgroup_position_in_grid.y;
    uint chunk = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint V = sizes[0];
    uint C = sizes[1];
    float cap = params[2 * row + 1];
    float m = -INFINITY, low = INFINITY;
    for (uint e = 0; e < PER_THREAD; ++e) {
        uint j = chunk * (THREADS * PER_THREAD) + e * THREADS + tid;
        if (j < V) {
            float x = nk_key(float(scores[row * V + j]), cap);
            if (STORE_CAP) capped[row * V + j] = x;
            if (x > m) m = x;
            if (metal::isfinite(x)) low = metal::min(low, x);
        }
    }
    m = simd_max(m);
    low = simd_min(low);
    threadgroup float2 part[THREADS / 32];
    if (lane == 0) part[sg] = float2(m, low);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float2 t = part[0];
        for (uint g = 1; g < THREADS / 32; ++g) {
            t.x = metal::max(t.x, part[g].x);
            t.y = metal::min(t.y, part[g].y);
        }
        xmax[2 * (row * C + chunk)] = t.x;
        xmax[2 * (row * C + chunk) + 1] = t.y;
    }
"""

_BINS = r"""
    uint row = threadgroup_position_in_grid.y;
    uint chunk = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint V = sizes[0];
    uint C = sizes[1];
    float scale = params[2 * row];
    float cap = PRE_CAPPED ? 0.0f : params[2 * row + 1];
    threadgroup atomic_uint h_cnt[BINS];
    threadgroup atomic_uint h_hi[BINS];
    threadgroup atomic_uint h_lo[BINS];
    threadgroup float xms;
    for (uint b = tid; b < BINS; b += THREADS) {
        atomic_store_explicit(&h_cnt[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&h_hi[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&h_lo[b], 0u, memory_order_relaxed);
    }
    if (tid == 0) xms = nk_row_max(xmax, row, C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xm = xms;
    for (uint e = 0; e < PER_THREAD; ++e) {
        uint j = chunk * (THREADS * PER_THREAD) + e * THREADS + tid;
        if (j >= V) break;
        float x = nk_key(float(scores[row * V + j]), cap);
        float d = (xm - x) * scale;                  // >= 0; +inf for a -inf entry
        uint b = nk_bin(d);
        atomic_fetch_add_explicit(&h_cnt[b], 1u, memory_order_relaxed);
        if (!words[2]) continue;       // a pure top-k cut needs counts alone
        ulong f = nk_mass(d);
        uint hi = uint(f >> 20), lo = uint(f & 0xFFFFFul);
        if (hi) atomic_fetch_add_explicit(&h_hi[b], hi, memory_order_relaxed);
        if (lo) atomic_fetch_add_explicit(&h_lo[b], lo, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = tid; b < BINS; b += THREADS) {
        uint o = ((row * C + chunk) * BINS + b) * 3;
        hist[o + 0] = atomic_load_explicit(&h_cnt[b], memory_order_relaxed);
        hist[o + 1] = atomic_load_explicit(&h_hi[b], memory_order_relaxed);
        hist[o + 2] = atomic_load_explicit(&h_lo[b], memory_order_relaxed);
    }
"""

# One threadgroup of BINS threads per row. cut[row * 16 + ...] (uint32 words):
#   0 lo bin  1 hi bin  2 entries above lo  3 candidates  4,5 estimated mass above lo  6,7 estimated total
#   8,9 estimated target  10,11 estimate's bound  12 k  13 status  14 no cut (keep every entry)  15 top_p on
_CUT = r"""
    uint row = threadgroup_position_in_grid.x;
    uint b = thread_position_in_threadgroup.x;
    uint V = sizes[0];
    uint C = sizes[1];
    threadgroup ulong cnt[BINS];
    threadgroup ulong mass[BINS];
    ulong c = 0, mh = 0, ml = 0;
    for (uint k = 0; k < C; ++k) {
        uint o = ((row * C + k) * BINS + b) * 3;
        c += ulong(hist[o]);
        mh += ulong(hist[o + 1]);
        ml += ulong(hist[o + 2]);
    }
    cnt[b] = c;
    mass[b] = (mh << 20) + ml;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (b == 0) {
        ulong total = 0;
        for (uint i = 0; i < BINS; ++i) total += mass[i];
        uint p_on = words[2];
        ulong pf = ulong(words[1]);                 // top_p as a 32-bit fraction (floor)
        ulong target = (total >> 32) * pf + (((total & 0xFFFFFFFFul) * pf) >> 32);
        ulong bnd = (total >> COARSE_SHIFT) + ulong(V) + 2ul;
        uint k = words[3];
        ulong keff = (k > 0u && k < V) ? ulong(k) : ulong(V);
        uint o = row * 16;
        bool none = (p_on == 0u) && keff == ulong(V);
        uint bk = BINS - 1;                         // the bin of the keff-th entry
        ulong run = 0;
        for (uint i = 0; i < BINS; ++i) { run += cnt[i]; if (run >= keff) { bk = i; break; } }
        uint lo = bk, hi = bk;
        if (p_on) {
            // b1: the first bin whose end reaches target - bound (no earlier entry can be dropped);
            // b2: the last bin whose start is below target + bound (every later entry is dropped)
            uint b1 = BINS - 1, b2 = 0;
            bool found = false;
            ulong before = 0;
            for (uint i = 0; i < BINS; ++i) {
                ulong after = before + mass[i];
                if (!found && after + bnd >= target) { b1 = i; found = true; }
                if (before < target + bnd) b2 = i;
                before = after;
            }
            if (b2 < b1) b2 = b1;
            if (bk < b1) { lo = bk; hi = bk; }
            else { lo = b1; hi = metal::min(b2, bk); }
        }
        ulong above_n = 0, above_m = 0;
        for (uint i = 0; i < lo; ++i) { above_n += cnt[i]; above_m += mass[i]; }
        ulong cands = 0;
        for (uint i = lo; i <= hi; ++i) cands += cnt[i];
        cut[o + 0] = lo;
        cut[o + 1] = hi;
        cut[o + 2] = uint(above_n);
        cut[o + 3] = uint(metal::min(cands, ulong(0xFFFFFFFFu)));
        cut[o + 4] = uint(above_m & 0xFFFFFFFFul);
        cut[o + 5] = uint(above_m >> 32);
        cut[o + 6] = uint(total & 0xFFFFFFFFul);
        cut[o + 7] = uint(total >> 32);
        cut[o + 8] = uint(target & 0xFFFFFFFFul);
        cut[o + 9] = uint(target >> 32);
        cut[o + 10] = uint(bnd & 0xFFFFFFFFul);
        cut[o + 11] = uint(bnd >> 32);
        cut[o + 12] = uint(keff);
        cut[o + 13] = (!none && (cands > ulong(CAP) || cands == 0ul)) ? 3u : 0u;
        cut[o + 14] = none ? 1u : 0u;
        cut[o + 15] = p_on;
    }
"""

# Candidates' scores and ids, and per chunk the exact sums (21-bit limbs) of the whole row's mass and of the
# entries above the candidates. Outputs are atomic (the slot counter, and the plain stores beside it).
_GATHER = r"""
    uint row = threadgroup_position_in_grid.y;
    uint chunk = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint V = sizes[0];
    uint C = sizes[1];
    uint o = row * 16;
    bool gather = cut[o + 13] == 0u && cut[o + 14] == 0u;
    bool exact = gather && cut[o + 15] != 0u;
    uint lo = cut[o + 0], hi = cut[o + 1];
    float scale = params[2 * row];
    float cap = PRE_CAPPED ? 0.0f : params[2 * row + 1];
    float ih = as_type<float>(words[6]), il = as_type<float>(words[7]);
    threadgroup float xms;
    threadgroup uint part[6 * (THREADS / 32)];
    if (tid == 0) xms = nk_row_max(xmax, row, C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xm = xms;
    float2 all_mass = float2(0.0f), above_mass = float2(0.0f);
    if (gather) {
        for (uint e = 0; e < PER_THREAD; ++e) {
            uint j = chunk * (THREADS * PER_THREAD) + e * THREADS + tid;
            if (j >= V) break;
            float x = nk_key(float(scores[row * V + j]), cap);
            uint b = nk_bin((xm - x) * scale);
            if (b >= lo && b <= hi) {
                uint slot = atomic_fetch_add_explicit(&counter[row], 1u, memory_order_relaxed);
                if (slot < CAP) {
                    atomic_store_explicit(&cand[(row * CAP + slot) * 2], as_type<uint>(x), memory_order_relaxed);
                    atomic_store_explicit(&cand[(row * CAP + slot) * 2 + 1], j, memory_order_relaxed);
                }
            }
            if (exact) {
                float2 m = nk_exact_mass(xm, x, ih, il, etab);
                all_mass = nk_add_mass(all_mass, m);
                if (b < lo) above_mass = nk_add_mass(above_mass, m);
            }
        }
    }
    ulong s_all = nk_units(all_mass), s_above = nk_units(above_mass);
    uint l0 = uint(s_all & 0x1FFFFFul), l1 = uint((s_all >> 21) & 0x1FFFFFul), l2 = uint(s_all >> 42);
    uint l3 = uint(s_above & 0x1FFFFFul), l4 = uint((s_above >> 21) & 0x1FFFFFul), l5 = uint(s_above >> 42);
    l0 = simd_sum(l0); l1 = simd_sum(l1); l2 = simd_sum(l2);
    l3 = simd_sum(l3); l4 = simd_sum(l4); l5 = simd_sum(l5);
    if (lane == 0) {
        uint G = THREADS / 32;
        part[0 * G + sg] = l0; part[1 * G + sg] = l1; part[2 * G + sg] = l2;
        part[3 * G + sg] = l3; part[4 * G + sg] = l4; part[5 * G + sg] = l5;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) {
        uint t = 0;
        for (uint g = 0; g < THREADS / 32; ++g) t += part[tid * (THREADS / 32) + g];
        atomic_store_explicit(&sums[(row * C + chunk) * 6 + tid], t, memory_order_relaxed);
    }
"""

# One threadgroup per row: the candidates sorted (score down, id up), their exact masses walked from the exact
# mass above them; bound[row * 4 + ...] = boundary score bits, boundary id, status, kept count.
_SELECT = r"""
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint V = sizes[0];
    uint C = sizes[1];
    uint o = row * 16;
    uint status = cut[o + 13];
    uint n = metal::min(cut[o + 3], CAP);
    float scale = params[2 * row];
    float ih = as_type<float>(words[6]), il = as_type<float>(words[7]);
    threadgroup ulong keys[CAP];
    threadgroup ulong part[THREADS];
    threadgroup ulong limb[6];
    threadgroup ulong ex[4];                         // above, target, margin (normalised), the total's shift
    threadgroup atomic_uint i_lo;
    threadgroup atomic_uint i_hi;
    threadgroup float xms;
    if (tid == 0) {
        xms = nk_row_max(xmax, row, C);
        atomic_store_explicit(&i_lo, 0xFFFFFFFFu, memory_order_relaxed);
        atomic_store_explicit(&i_hi, 0xFFFFFFFFu, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xm = xms;
    if (status != 0u || cut[o + 14] != 0u || !metal::isfinite(xm)) {
        // keep every entry: a row the round does not decide (status), a request with no cut (min-p alone),
        // or a row that is no law (the decision's own finiteness check refuses it)
        if (tid == 0) {
            bound[row * 4 + 0] = as_type<uint>(-INFINITY);
            bound[row * 4 + 1] = 0xFFFFFFFFu;
            if (status == 0u && cut[o + 14] && nk_underflow_tail(xmax, row, C, xm, scale)) status = 2u;
            bound[row * 4 + 2] = metal::isfinite(xm) ? status : 0u;
            bound[row * 4 + 3] = V;
        }
        return;
    }
    for (uint i = tid; i < CAP; i += THREADS) {
        ulong key = 0ul;
        if (i < n) {
            float x = as_type<float>(cand[(row * CAP + i) * 2]);
            uint id = cand[(row * CAP + i) * 2 + 1];
            key = (ulong(nk_order(x)) << 32) | ulong(0xFFFFFFFFu - id);
        }
        keys[i] = key;
    }
    uint p_on = cut[o + 15];
    if (p_on && tid < 6u) {
        ulong t = 0;
        for (uint c = 0; c < C; ++c) t += ulong(sums[(row * C + c) * 6 + tid]);
        limb[tid] = t;                               // under 2^36 each
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint size = 1;
    while (size < n) size <<= 1;
    for (uint k = 2; k <= size; k <<= 1) {               // bitonic sort, descending
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < size; i += THREADS) {
                uint l = i ^ j;
                if (l > i) {
                    ulong a = keys[i], c = keys[l];
                    bool down = (i & k) == 0u;
                    if (down ? (a < c) : (a > c)) { keys[i] = c; keys[l] = a; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    uint per = (n + THREADS - 1) / THREADS;
    uint first = metal::min(tid * per, n), last = metal::min(first + per, n);
    if (p_on) {
        if (tid == 0) {
            // the exact total W and mass above A (128 bits from the limbs), shifted so that W < 2^62
            ulong whi = limb[2] >> 22, wlo = limb[2] << 42;
            ulong t1 = limb[1] << 21;
            ulong s1 = wlo + t1; whi += (s1 < wlo) ? 1ul : 0ul;
            ulong s2 = s1 + limb[0]; whi += (s2 < s1) ? 1ul : 0ul; wlo = s2;
            ulong ahi = limb[5] >> 22, alo = limb[5] << 42;
            ulong u1 = alo + (limb[4] << 21); ahi += (u1 < alo) ? 1ul : 0ul;
            ulong u2 = u1 + limb[3]; ahi += (u2 < u1) ? 1ul : 0ul; alo = u2;
            uint bits = (whi != 0ul) ? (128u - nk_clz64(whi)) : (64u - nk_clz64(wlo));
            uint sh = (bits > 62u) ? (bits - 62u) : 0u;
            ulong wn = (sh == 0u) ? wlo : ((wlo >> sh) | (whi << (64u - sh)));
            ulong an = (sh == 0u) ? alo : ((alo >> sh) | (ahi << (64u - sh)));
            // target = floor(wn * p), p the 64-bit fraction (words 0, 1)
            ulong a1 = wn >> 32, a0 = wn & 0xFFFFFFFFul;
            ulong b1 = ulong(words[1]), b0 = ulong(words[0]);
            ulong t = a0 * b0;
            ulong u = a1 * b0 + (t >> 32);
            ulong v = a0 * b1 + (u & 0xFFFFFFFFul);
            ulong target = a1 * b1 + (u >> 32) + (v >> 32);
            // margin: the host's error (4|max|/T + V + 512) 2^-52 and the device's 2^-38, of the total,
            // as q / 2^64 of wn, plus a unit per candidate and per shift
            float qf = (4.0f * metal::abs(xm) * scale + float(V) + 512.0f) * 4096.0f + 67108864.0f;
            qf *= 1.001f;
            ulong margin = 0xFFFFFFFFFFFFFFFFul;
            if (qf < 4294967296.0f) {
                ulong q = ulong(qf) + 1ul;
                ulong m1 = a1 * q, m0 = a0 * q;
                margin = ((m1 + (m0 >> 32)) >> 32) + ulong(n) + 8ul;
            }
            ex[0] = an;
            ex[1] = target;
            ex[2] = margin;
            ex[3] = ulong(sh);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sh = uint(ex[3]);
        ulong local = 0;
        for (uint i = first; i < last; ++i)
            local += nk_units(nk_exact_mass(xm, nk_unorder(uint(keys[i] >> 32)), ih, il, etab)) >> sh;
        part[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            ulong run = 0;
            for (uint t = 0; t < THREADS; ++t) { ulong w = part[t]; part[t] = run; run += w; }
            // position n, the entry after the last candidate
            ulong before = ex[0] + run;
            if (!(before + ex[2] + 2ul <= ex[1])) atomic_fetch_min_explicit(&i_lo, n, memory_order_relaxed);
            if (ex[2] != 0xFFFFFFFFFFFFFFFFul && before >= ex[1] + ex[2] + 2ul)
                atomic_fetch_min_explicit(&i_hi, n, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // i_lo: the first position not surely kept; i_hi: the first surely dropped
        ulong target = ex[1], margin = ex[2];
        bool usable = margin != 0xFFFFFFFFFFFFFFFFul;
        ulong before = ex[0] + part[tid];
        for (uint i = first; i < last; ++i) {
            if (!usable || !(before + margin + 2ul <= target)) atomic_fetch_min_explicit(&i_lo, i, memory_order_relaxed);
            if (usable && before >= target + margin + 2ul) atomic_fetch_min_explicit(&i_hi, i, memory_order_relaxed);
            before += nk_units(nk_exact_mass(xm, nk_unorder(uint(keys[i] >> 32)), ih, il, etab)) >> sh;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        ulong above_n = ulong(cut[o + 2]);
        ulong keff = ulong(cut[o + 12]);
        ulong L = keff;
        uint s = 0u;
        if (p_on) {
            uint lo_i = atomic_load_explicit(&i_lo, memory_order_relaxed);
            uint hi_i = atomic_load_explicit(&i_hi, memory_order_relaxed);
            ulong lo_p = (lo_i == 0xFFFFFFFFu) ? ulong(n) + 1ul : ulong(lo_i);
            ulong hi_p = (hi_i == 0xFFFFFFFFu) ? ulong(n) + 1ul : ulong(hi_i);
            if (ex[2] == 0xFFFFFFFFFFFFFFFFul) s = 2u;  // scores too large for the margin's arithmetic
            else if (lo_p == 0ul) s = 3u;                // the first candidate is not surely kept
            else if (keff > above_n + lo_p) {             // top-p decides: its crossing must be certain
                if (lo_p == hi_p && lo_p <= ulong(n)) L = above_n + lo_p;
                else s = (lo_p < hi_p && hi_p <= ulong(n) + 1ul && lo_p <= ulong(n)) ? 2u : 3u;
            }
        }
        if (s == 0u && (L <= above_n || L > above_n + ulong(n))) s = 3u;
        if (s == 0u) {
            ulong kb = keys[uint(L - above_n - 1ul)];
            float xb = nk_unorder(uint(kb >> 32));
            // Distinct float32 logits must stay distinct through the host's
            // float64 division, subtraction, exp and normalization at the cutoff.
            float distance = (xm - xb) * scale;
            float spacing = metal::abs(xb) * scale * 0x1p-24f;
            float uncertainty = (metal::abs(xm) * scale + distance + 4.0f) * 0x1p-50f;
            if (metal::isfinite(xb) && (distance > 680.0f || spacing <= uncertainty)) s = 2u;
            if (xb == -INFINITY && nk_underflow_tail(xmax, row, C, xm, scale)) s = 2u;
            bound[row * 4 + 0] = as_type<uint>(xb);
            bound[row * 4 + 1] = 0xFFFFFFFFu - uint(kb & 0xFFFFFFFFul);
        }
        if (s != 0u) {
            bound[row * 4 + 0] = as_type<uint>(-INFINITY);
            bound[row * 4 + 1] = 0xFFFFFFFFu;
        }
        bound[row * 4 + 2] = s;
        bound[row * 4 + 3] = uint(L);
    }
"""

# The masked law rows and their chunk stats, as round._STATS writes them: (max, sum exp(y - max)).
_MASK = r"""
    uint row = threadgroup_position_in_grid.y;
    uint chunk = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint V = sizes[0];
    uint C = sizes[1];
    float cap = PRE_CAPPED ? 0.0f : params[2 * row + 1];
    float xb = as_type<float>(bound[row * 4 + 0]);
    uint ib = bound[row * 4 + 1];
    bool decided = bound[row * 4 + 2] == 0u;
    bool minp = words[4] != 0u;
    float floor_ = as_type<float>(words[5]);
    threadgroup float xms;
    if (tid == 0) xms = nk_row_max(xmax, row, C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xm = xms;
    float2 acc = float2(-INFINITY, 0.0f);
    for (uint e = 0; e < PER_THREAD; ++e) {
        uint j = chunk * (THREADS * PER_THREAD) + e * THREADS + tid;
        if (j < V) {
            float x = nk_key(float(scores[row * V + j]), cap);
            float y = HAVE_LAW ? law[row * V + j] : x * params[2 * row];
            if (decided && metal::isfinite(x)) {
                bool keep = (x > xb) || (x == xb && j <= ib);
                if (minp && !(x - xm >= floor_)) keep = false;
                if (!keep) y = -INFINITY;
            }
            out[row * V + j] = y;
            if (y == -INFINITY) continue;
            if (!(y <= acc.x)) {
                acc = float2(y, (acc.x == -INFINITY ? 0.0f : acc.y * metal::exp(acc.x - y)) + 1.0f);
            } else {
                acc.y += metal::exp(y - acc.x);
            }
        }
    }
    for (uint d = 16; d > 0; d >>= 1) {
        float2 other = float2(simd_shuffle_down(acc.x, d), simd_shuffle_down(acc.y, d));
        if (lane + d < 32) acc = nk_lse_merge(acc, other);
    }
    threadgroup float2 part[THREADS / 32];
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float2 total = part[0];
        for (uint g = 1; g < THREADS / 32; ++g) total = nk_lse_merge(total, part[g]);
        stats[(row * C + chunk) * 2] = total.x;
        stats[(row * C + chunk) * 2 + 1] = total.y;
    }
"""

_LSE_MERGE = r"""
inline float2 nk_lse_merge(float2 a, float2 b) {
    if (a.x == -INFINITY) return b;
    if (b.x == -INFINITY) return a;
    float m = metal::max(a.x, b.x);
    return float2(m, a.y * metal::exp(a.x - m) + b.y * metal::exp(b.x - m));
}
"""

# Only a reached row may send a decision to the host. A later speculative
# row depends on today's proposal, so restarting the whole round on its
# refusal would condition the earlier output on that proposal.
_MERGE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= sizes[0]) return;
    int v = packet[i];
    if (packet[1] == 0) {
        uint n = uint(packet[0]);
        for (uint r = 0; r <= n; ++r) {
            uint s = bound[r * 4 + 2];
            if (s) {
                if (i == 0u) v = int(r);
                else if (i == 1u) v = int(s);
                else if (i >= 2u + r && i < 2u + sizes[1]) v = -1;
                break;
            }
        }
    }
    merged[i] = v;
"""

# Both components of the masses used in the chunk sums, for the accuracy gate.
_PROBE = r"""
    uint i = thread_position_in_grid.x;
    uint N = sizes[0];
    if (i >= N) return;
    float2 m = nk_exact_mass(values[0], values[1 + i], as_type<float>(words[6]), as_type<float>(words[7]), etab);
    out[2 * i] = m.x;
    out[2 * i + 1] = m.y;
"""


def _template():
    return [("THREADS", THREADS), ("PER_THREAD", PER_THREAD)]


@lru_cache(maxsize=None)
def _kernel(name):
    specs = {
        "xmax": (["scores", "params", "sizes"], ["xmax", "capped"], _XMAX, False),
        "bins": (["scores", "params", "xmax", "words", "sizes"], ["hist"], _BINS, False),
        "cut": (["hist", "words", "sizes"], ["cut"], _CUT, False),
        "gather": (["scores", "params", "xmax", "cut", "words", "etab", "sizes"], ["counter", "cand", "sums"],
                   _GATHER, True),
        "select": (["cand", "cut", "sums", "params", "xmax", "words", "etab", "sizes"], ["bound"], _SELECT, False),
        "mask": (["scores", "law", "params", "xmax", "bound", "words", "sizes"], ["out", "stats"], _MASK, False),
        "merge": (["packet", "bound", "sizes"], ["merged"], _MERGE, False),
        "probe": (["values", "words", "etab", "sizes"], ["out"], _PROBE, False),
    }
    inputs, outputs, source, atomic = specs[name]
    header = "" if name == "merge" else _COMMON + (_LSE_MERGE if name == "mask" else "")
    return launch_limits.kernel(name=f"mtplx_inkling_nucleus_{name}", input_names=inputs, output_names=outputs,
                                source=source, header=header, atomic_outputs=atomic, probe_group=True)


@lru_cache(maxsize=None)
def _etab_values():
    """exp(-a) for a = 0..43 and exp(-b/256) for b = 0..255, each as a float32 pair (high, low)."""
    v = np.concatenate([np.exp(-np.arange(44, dtype=np.float64)), np.exp(-np.arange(256, dtype=np.float64) / 256.0)])
    hi = v.astype(np.float32)
    lo = (v - hi.astype(np.float64)).astype(np.float32)
    return np.stack([hi, lo], axis=1)


@lru_cache(maxsize=1)
def _etab():
    # The round's recorder pins constants returned by _const.
    return _const(_etab_values().tolist(), mx.float32)


def _bits(value):
    return int(np.array([value], np.float32).view(np.uint32)[0])


def words(*, top_p=1.0, top_k=0, min_p=0.0, temperature=1.0):
    """The request's filter as ``WORDS`` uint32 words (a request constant): ``top_p`` as a 64-bit fraction
    (floor; exact for a float64 ``top_p``) and whether it is on, ``top_k`` (0: off), whether min-p is on and its
    floor (the float32 at or above ``T ln(min_p)``, the test ``ClientControlledDraw.rows`` makes), and 1/T as a
    float-float (two float32), for the exact masses."""
    p_on = 0 < float(top_p) < 1.0
    p64 = int(float(top_p) * 2.0 ** 64) if p_on else 0
    k = min(max(int(top_k or 0), 0), 0xFFFFFFFF)
    t = float(temperature or 0.0)
    m_on = t > 0 and float(min_p or 0.0) > 0
    floor_bits = 0
    if m_on:
        floor = t * float(np.log(float(min_p)))
        floor32 = np.float32(floor)
        if float(floor32) < floor:
            floor32 = np.nextafter(floor32, np.float32(np.inf))
        floor_bits = _bits(floor32)
    inv = 1.0 / t if t > 0 else 1.0
    ih = np.float32(inv)
    il = np.float32(inv - float(ih))
    return [p64 & 0xFFFFFFFF, (p64 >> 32) & 0xFFFFFFFF, int(p_on), k, int(m_on), floor_bits, _bits(ih), _bits(il)]


def active(*, top_p=1.0, top_k=0, min_p=0.0, temperature=1.0):
    """Whether a sampled request's top-p or top-k changes its law (top-p below 1, top-k above 0)."""
    return float(temperature or 0.0) > 0 and (0 < float(top_p) < 1.0 or int(top_k or 0) > 0)


def request_words(sampler, *, min_p=None):
    """One array per sampled request with a vocabulary cut. Min-p alone uses the existing floor."""
    p, k, t = sampler.top_p, sampler.top_k, sampler.temperature
    if not active(top_p=p, top_k=k, temperature=t):
        return None
    return mx.array(words(top_p=p, top_k=k, temperature=t,
                          min_p=sampler.min_p if min_p is None else min_p), mx.uint32)


def filtered(scores, law, params, filter_words, *, store_cap=False):
    """Masked law rows, their chunk stats and each row's boundary (lazy).

    ``scores`` (R, V) the rows ``law`` was made from (any float dtype); ``law`` (R, V) float32 the law rows
    (``round.law_rows`` with ``write``); ``params`` the (2R,) float32 law parameters (scale and cap per row,
    ``round.law_params``); ``filter_words`` a (``WORDS``,) uint32 array (``words``). Returns (law (R, V) float32,
    stats (R, C, 2) float32, bound (R, 4) uint32): ``bound[:, 2]`` is each row's status (``STATUS_OK``,
    ``STATUS_BOUNDARY``, ``STATUS_OVERFLOW``; a row not decided here keeps every entry) and ``bound[:, 3]``
    the entries its top-p and top-k keep.
    """
    try:
        return _filtered(scores, law, params, filter_words, store_cap)
    except launch_limits.GroupRefused:
        from .round import law_rows

        # A drafter can use the whole row as q. A target must take the host.
        unmasked, stats = law_rows(scores, params=params)
        bound = mx.broadcast_to(_const([0, 0, STATUS_GROUP, 0], mx.uint32), (scores.shape[0], 4))
        return unmasked, stats, bound


def _filtered(scores, law, params, filter_words, store_cap):
    rows, vocab = scores.shape
    chunks = _chunks(vocab)
    sizes = _const([vocab, chunks], mx.uint32)
    etab = _etab()
    grid = (chunks * THREADS, rows, 1)
    tg = (THREADS, 1, 1)
    t = _template()
    xmax, capped = _kernel("xmax")(inputs=[scores, params, sizes], template=[*t, ("STORE_CAP", store_cap)],
                                   grid=grid, threadgroup=tg, output_shapes=[(rows, chunks, 2),
                                   (rows, vocab) if store_cap else (1,)], output_dtypes=[mx.float32, mx.float32])
    # The drafter's soft cap is expensive over a full vocabulary. Compute it
    # once when finding the maximum and reuse those same bits in each pass.
    keys = capped if store_cap else scores
    key_template = [*t, ("PRE_CAPPED", store_cap)]
    (hist,) = _kernel("bins")(inputs=[keys, params, xmax, filter_words, sizes], template=key_template,
                              grid=grid, threadgroup=tg, output_shapes=[(rows, chunks, BINS, 3)],
                              output_dtypes=[mx.uint32])
    (cut,) = _kernel("cut")(inputs=[hist, filter_words, sizes], template=t, grid=(BINS * rows, 1, 1),
                            threadgroup=(BINS, 1, 1), output_shapes=[(rows, 16)], output_dtypes=[mx.uint32])
    counter, cand, sums = _kernel("gather")(
        inputs=[keys, params, xmax, cut, filter_words, etab, sizes], template=key_template, grid=grid, threadgroup=tg,
        output_shapes=[(rows,), (rows, CAP, 2), (rows, chunks, 6)], output_dtypes=[mx.uint32] * 3, init_value=0)
    (bound,) = _kernel("select")(inputs=[cand, cut, sums, params, xmax, filter_words, etab, sizes], template=t,
                                 grid=(THREADS * rows, 1, 1), threadgroup=tg, output_shapes=[(rows, 4)],
                                 output_dtypes=[mx.uint32])
    out, stats = _kernel("mask")(inputs=[keys, law if law is not None else keys, params, xmax, bound, filter_words, sizes],
                                 template=[*key_template, ("HAVE_LAW", law is not None)], grid=grid,
                                 threadgroup=tg, output_shapes=[(rows, vocab), (rows, chunks, 2)],
                                 output_dtypes=[mx.float32, mx.float32])
    return out, stats, bound


def merge_status(packet, bound, rows):
    """The packet with the rows' filter status merged (``_MERGE``)."""
    n = int(packet.shape[0])
    try:
        (merged,) = _kernel("merge")(inputs=[packet, bound, _const([n, rows], mx.uint32)], template=[],
                                     grid=(-(-n // 32) * 32, 1, 1), threadgroup=(32, 1, 1),
                                     output_shapes=[(n,)], output_dtypes=[mx.int32])
    except launch_limits.GroupRefused:
        # This is a pipeline refusal, independent of the drafts. The host
        # may start at row zero. Keep the existing invalid-logit status.
        status = mx.where(packet[1:2] == STATUS_OK, _const([STATUS_GROUP], mx.int32), packet[1:2])
        return mx.concatenate([_const([0], mx.int32), status, packet[2:]])
    return merged


def warm(draft_dtype):
    """Build the filtered pipelines on synthetic scores before the first client uses them."""
    from . import round as rnd

    # Enough chunks for the same buffer address spaces as a production row.
    # Nine rows also compile the decision used for wider filtered copy panels.
    scores = mx.zeros((9, 16384), mx.float32)
    filters = mx.array(words(top_p=0.95), mx.uint32)
    proposal, stats, boundary = rnd.sampled_rows(scores[:1].astype(draft_dtype), temperature=1.0,
                                                  cap=30.0, filters=filters)
    mx.eval(proposal, stats, boundary)
    mx.eval(rnd.draw_checked(scores[:1], position=1, seed=29, filters=filters),
            rnd.decide(scores, mx.arange(8, dtype=mx.int32), position=1, seed=29, filters=filters))


# ---------------------------------------------------------------- tests
def exact_masses(xm, values, *, temperature):
    """Device masses before their chunk sum, expressed in units of 2**-60 (tests)."""
    vals = mx.array(np.concatenate([[np.float32(xm)], np.asarray(values, np.float32)]).astype(np.float32))
    n = int(vals.shape[0]) - 1
    w = mx.array(words(temperature=temperature), mx.uint32)
    (out,) = _kernel("probe")(inputs=[vals, w, _etab(), _const([n], mx.uint32)], template=[],
                              grid=(max(n, 1), 1, 1), threadgroup=(min(max(n, 1), 256), 1, 1),
                              output_shapes=[(n, 2)], output_dtypes=[mx.float32])
    mx.eval(out)
    parts = np.asarray(out).astype(np.float64)
    return (parts[:, 0] + parts[:, 1]) * 2.0 ** 60


def kept_reference(scores, *, temperature, top_p=1.0, top_k=0, min_p=0.0, cap=0.0):
    """The product's kept set for one row (bool (V,)): ``mtplx.sampling.apply_top_p_top_k`` on the
    float64 tempered softmax of the float32 scores, then the float32 min-p floor of
    ``ClientControlledDraw.rows``."""
    from mtplx.sampling import apply_top_p_top_k, softmax

    x = np.asarray(scores, np.float32)
    if cap and cap > 0:
        x = (np.tanh(x.astype(np.float64) / cap) * cap).astype(np.float32)
    probs = softmax(x.astype(np.float64), float(temperature))
    kept = apply_top_p_top_k(probs, top_p=float(top_p), top_k=int(top_k or 0)) > 0
    if min_p and min_p > 0:
        w = words(min_p=min_p, temperature=temperature)
        floor32 = np.array([w[5]], np.uint32).view(np.float32)[0]
        kept &= (x - x.max()) >= floor32
    return kept


__all__ = ["BINS", "CAP", "STATUS_BOUNDARY", "STATUS_OK", "STATUS_OVERFLOW", "WORDS", "active", "exact_masses",
           "filtered", "kept_reference", "merge_status", "words"]
