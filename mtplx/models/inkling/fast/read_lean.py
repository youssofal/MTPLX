"""Lane L1's read kernel in fewer dependent steps (lane F1): the same arithmetic, scheduled leaner.

On a dependent chain the read kernel costs about 5 us a read with 2 sources and one query, 14 us with
5 sources and 22 us with 5 sources and three queries (``summary/f1/receipts/readbench-1.json``,
provisional): its cost grows with sources times queries. In its gemv every (query, source) dot
re-reads its operands from device memory and runs its own shuffle tree, one after another; the mix
re-reads every source again; the input norms of a three-query read take two barriers each. This
kernel is lane L1's text with five of its blocks scheduled leaner, every value computed exactly as
there:

1. the first pass keeps each thread's four columns of every source in registers;
2. the keys of a three-query read are written from those registers;
3. the gemv takes every (query, source) dot in one pass over the lane's columns (its own columns
   from registers), each dot still adding its terms in the same column order, then runs the
   shuffle trees side by side (each the same tree);
4. the mix reads the sources from registers;
5. the input norms of every query take their partial sums first, one barrier, each query's total
   in its own simdgroup, one barrier, then the writes.

The kernel text is derived from lane L1's own (``read._READ``) by replacing those five blocks; a
block that is not found (lane L1 changed it) leaves no lean kernel and lane L1's serves. Class A:
decided per launch shape on this Mac on the first call, against lane L1's kernel, every output the
mode writes bit for bit; a shape that differs keeps lane L1's kernel, counted
``fallback.read_lean.differs``. ``MTPLX_INKLING_READ_LEAN=0`` turns it off.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from .routes import count

#: launch key -> True when the lean kernel gives lane L1's bits on this Mac.
_VERDICT: dict = {}

_FIRST_PASS = (
    """    for (uint s = 0; s < S; ++s) {
        float acc = 0;
        for (uint i = 0; i < 4; ++i) {
            const float xval = in_row ? SOURCE(s, r, c4 + i) : 0.0f;
            acc += xval * xval;
        }
        acc = simd_sum(acc);
        if (lane == 0) sums[s][sg] = acc;
    }""",
    """    float xs[S][4];                                    // lane F1: this thread's columns, kept
    for (uint s = 0; s < S; ++s) {
        float acc = 0;
        for (uint i = 0; i < 4; ++i) {
            const float xval = in_row ? SOURCE(s, r, c4 + i) : 0.0f;
            xs[s][i] = xval;
            acc += xval * xval;
        }
        acc = simd_sum(acc);
        if (lane == 0) sums[s][sg] = acc;
    }""",
)

_KEYS = (
    """o0[(size_t(r) * S + s) * D + c4 + i] = static_cast<OUT0>(1.0f * (SOURCE(s, r, c4 + i) * inv[s]));""",
    """o0[(size_t(r) * S + s) * D + c4 + i] = static_cast<OUT0>(1.0f * (xs[s][i] * inv[s]));""",
)

_GEMV_OLD_HEAD = "    if (sg < 8) {\n        for (uint qi = q_first; qi < q_first + q_count; ++qi) {"
_GEMV_OLD_TAIL = "                if (lane == 0) dots[(qi - q_first) * S + s][sg] = p;\n            }\n        }\n    }"
_GEMV_NEW = """    if (sg < 8) {
        // lane F1: every (query, source) dot in one pass over this lane's columns, each adding its
        // terms in the same column order (its own block from registers), then the same trees.
        constexpr uint QC = MODE == 2 ? 1 : NQ;
        float p[QC][S];
        for (uint qi = 0; qi < QC; ++qi)
            for (uint s = 0; s < S; ++s) p[qi][s] = 0.0f;
        for (uint b = 0; b < FULL; ++b) {
            for (uint co = 0; co < 4; ++co) {
                const uint c = b * 1024 + sg * 128 + lane * 4 + co;
                float key[S];
                for (uint s = 0; s < S; ++s)
                    key[s] = b == 0 ? 1.0f * (xs[s][co] * inv[s]) : 1.0f * (SOURCE(s, r, c) * inv[s]);
                for (uint qi = 0; qi < QC; ++qi) {
                    const float qv = queries[qi * D + c];
                    for (uint s = 0; s < S; ++s) p[qi][s] += key[s] * qv;
                }
            }
        }
        if (LEFT > 0) {
            const uint c0_ = FULL * 1024 + sg * 128 + lane * 4;
            for (uint co = 0; co < 4; ++co) {
                const uint c = c0_ + co;
                float key[S];
                for (uint s = 0; s < S; ++s) key[s] = (c < D) ? 1.0f * (SOURCE(s, r, c) * inv[s]) : 0.0f;
                for (uint qi = 0; qi < QC; ++qi) {
                    const float qv = (c < D) ? queries[qi * D + c] : 0.0f;
                    for (uint s = 0; s < S; ++s) p[qi][s] += key[s] * qv;
                }
            }
        }
        for (ushort hop = 16; hop != 0; hop /= 2)
            for (uint qi = 0; qi < QC; ++qi)
                for (uint s = 0; s < S; ++s) p[qi][s] += simd_shuffle_down(p[qi][s], hop);
        if (lane == 0)
            for (uint qi = 0; qi < QC; ++qi)
                for (uint s = 0; s < S; ++s) dots[qi * S + s][sg] = p[qi][s];
    }"""

_MIX_OLD = """                for (uint s = 0; s < S && s < 4; ++s) {
                    const float key = 1.0f * (SOURCE(s, r, c) * inv[s]);
                    v0 += prob[qn][s] * key;
                }
                for (uint s = 4; s < S; ++s) {
                    const float key = 1.0f * (SOURCE(s, r, c) * inv[s]);
                    v1 += prob[qn][s] * key;
                }"""
_MIX_NEW = """                for (uint s = 0; s < S && s < 4; ++s) {
                    const float key = 1.0f * (xs[s][i] * inv[s]);
                    v0 += prob[qn][s] * key;
                }
                for (uint s = 4; s < S; ++s) {
                    const float key = 1.0f * (xs[s][i] * inv[s]);
                    v1 += prob[qn][s] * key;
                }"""

_NORMS_OLD_HEAD = "        // ---- 5. Input norm(s): MLX rms_single_row with the layer's gain, then the bf16 cast.\n"
_NORMS_OLD_TAIL = "                    o0[(size_t(r) * NQ + qn) * D + c4 + i] = static_cast<OUT0>(static_cast<bfloat16_t>(w * (out[qn][i] * iv)));\n                }\n            }\n        }\n"
_NORMS_NEW = """        // ---- 5. (lane F1) Input norm(s): every query's partial sums, one barrier, each query's MLX
        // rms_single_row total in its own simdgroup, one barrier, the gains and the bf16 casts.
        for (uint qn = 0; qn < NQ; ++qn) {
            float acc = 0;
            for (uint i = 0; i < 4; ++i) acc += out[qn][i] * out[qn][i];
            acc = simd_sum(acc);
            if (lane == 0) sums[S + qn][sg] = acc;
            if (sg == 0 && lane >= NSG) sums[S + qn][lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < NQ) {
            const float total = simd_sum(sums[S + sg][lane]);
            if (lane == 0) inv[S + sg] = metal::precise::rsqrt(total / float(D) + as_type<float>(uint(EPS)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint qn = 0; qn < NQ; ++qn) {
            const float iv = inv[S + qn];
            const device bfloat16_t* gain = qn == 0 ? g0 : (qn == 1 ? g1 : g2);
            if (in_row) {
                for (uint i = 0; i < 4; ++i) {
                    const float w = static_cast<float>(gain[c4 + i]);
                    o0[(size_t(r) * NQ + qn) * D + c4 + i] = static_cast<OUT0>(static_cast<bfloat16_t>(w * (out[qn][i] * iv)));
                }
            }
        }
"""

_SIZES = (
    ("threadgroup float sums[(S > 2 ? S : 2) + 2][32];", "threadgroup float sums[(S > 2 ? S : 2) + 4][32];"),
    ("threadgroup float inv[(S > 2 ? S : 2) + 2];", "threadgroup float inv[(S > 2 ? S : 2) + 4];"),
)


def enabled():
    return os.environ.get("MTPLX_INKLING_READ_LEAN", "1").strip().lower() not in ("0", "off", "false", "no")


def _block(text, head, tail):
    """``text`` with the block from ``head`` through ``tail`` located, or None."""
    i = text.find(head)
    if i < 0:
        return None
    j = text.find(tail, i)
    if j < 0:
        return None
    return i, j + len(tail)


@lru_cache(maxsize=1)
def source():
    """The lean kernel text from lane L1's, or None when a block is not found."""
    from .read import _READ

    text = _READ
    for old, new in (_FIRST_PASS, _KEYS, (_MIX_OLD, _MIX_NEW), *_SIZES):
        if text.count(old) != 1:
            return None
        text = text.replace(old, new)
    for head, tail, new in ((_GEMV_OLD_HEAD, _GEMV_OLD_TAIL, _GEMV_NEW),
                            (_NORMS_OLD_HEAD, _NORMS_OLD_TAIL, _NORMS_NEW)):
        span = _block(text, head, tail)
        if span is None:
            return None
        text = text[: span[0]] + new + text[span[1]:]
    return text


@lru_cache(maxsize=1)
def _kernel():
    from .read import _READ_HEADER

    text = source()
    if text is None:
        return None
    return mx.fast.metal_kernel(
        name="mtplx_inkling_f1_read_lean",
        input_names=["c0", "c1", "c2", "c3", "partial", "queries", "g0", "g1", "g2", "rin",
                     "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "d0", "d1", "d2"],
        output_names=["o0", "o1", "o2"],
        header=_READ_HEADER,
        source=text,
    )


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


def launch_read(key, launch, stock_kernel, mode):
    """``launch(kernel)`` with the lean kernel where proven for ``key``, else with lane L1's
    (``stock_kernel()``). The proof compares the outputs the mode writes (o0; mode 2 all three)."""
    if not enabled():
        return launch(stock_kernel())
    lean = _kernel()
    if lean is None:
        count("fallback.read_lean.source")
        return launch(stock_kernel())
    verdict = _VERDICT.get(key)
    if verdict is None:
        try:
            want = launch(stock_kernel())
            got = launch(lean)
            written = 3 if mode == 2 else 1
            mx.eval(want[:written], got[:written])
            verdict = all(_same(g, w) for g, w in zip(got[:written], want[:written]))
        except (RuntimeError, ValueError):
            verdict = False
        _VERDICT[key] = verdict
        from .attn_input import count_route

        count_route(("proved:" if verdict else "refused:") + "read_lean_" + "_".join(map(str, key)))
    if not verdict:
        count("fallback.read_lean.differs")
        return launch(stock_kernel())
    return launch(lean)


__all__ = ["enabled", "launch_read", "source"]
