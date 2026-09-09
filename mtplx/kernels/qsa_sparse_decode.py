"""Split-K native sparse-GQA attention for the Qwen3.8 QSA DECODE lanes.

MTPLX_QSA_SPARSE_DECODE (M=4 fixed verify).  The kernel itself lives in
``native_extensions/qsa_sparse_gqa`` and is reached through
``mtplx.native.qsa_sparse_gqa_decode``; this module is the lane: the gate,
the install probe, the engagement counters and the reference the probe
compares against.

WHAT IT REPLACES, AND WHY THE CENSUS UNDERSTATES IT
---------------------------------------------------
The retained fixed-M4 verify attends 4 query rows per QSA layer over the
indexer's selected top-512 pooled blocks.  Per layer, per verify cycle, the
shipped lane issues six dispatches:

    1  custom_kernel_..._qsa_m4_fused_kv_gather_c17408  [1050624,1,1]
    2  <MLX contiguous copy of k_sel.swapaxes(-1,-2)>   (Copy family)
    3  gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc1       [129,1,96]   scores
    4  block_softmax_float32                            [52224,1,1]
    5  <probs.astype(bf16)>
    6  gemv_t_bfloat16_bm1_bn2_sm8_sn4_tm4_tn4_nc1      [8,1,96]     P@V

Dispatch (1) materialises ``k_sel``/``v_sel`` as ``[1, 2, 4, 2052, 256]``
bf16 -- 8.40 MB each.  So per layer the lane WRITES 16.8 MB of gathered K/V,
MLX then copies 8.4 MB more for the transposed score operand, and dispatches
(3) and (6) read 8.4 MB each back.  Roughly 70 MB per layer, ~840 MB per
verify cycle, to attend 4 rows.

The dispatch census's QSA row (446 MB/cycle, 232 GB/s) does NOT show this:
its cost model prices the gather at a flat 4.19 MB and gives the score,
softmax and P@V dispatches zero bytes, and the transposed copy lands in the
Copy family.  Counted properly the QSA family moves ~1.05 GB in its 1.93
ms/cycle, i.e. about 540 GB/s -- right at this machine's measured 544 GB/s
ceiling.  **The lane is not bandwidth-starved; it is moving three times the
bytes it needs to.**  That is the thing this kernel changes: it reads the
cache rows once, in place, and never materialises them.

WHY SPLIT-K AND NOT THE PHASE-1 KERNEL
---------------------------------------
The phase-1 (prefill) kernel parallelises over query rows: grid
``(qL, kv_heads, 1)``.  At M=4 that is EIGHT threadgroups of 64 threads on a
40-core M5 Max, each walking all 2,051 selected keys.  Phase 1's own design
note priced the fix as its own item, and MTPLX has the general finding
already: a hand-written
metal_kernel SDPA lost to stock at long N precisely because MLX's production
SDPA switches to a KV-split two-pass path there.  So decode gets the KV-split
variant: ``(qL, kv_heads, n_splits)`` threadgroups accumulating independent
online-softmax states, then a merge pass.

NUMERICS -- ROUNDING CLASS, HUMANEVAL-GATED
--------------------------------------------
The visible set is IDENTICAL to the shipped lane's, slot for slot (the
kernel applies the shipped predicate ``block < (pos+1)//4`` to every slot of
``top_idx``, which is what the rows-gather token list does; it makes no
ordering assumption, because the selector hands through
``mx.argpartition``'s raw output unsorted).  The ARITHMETIC is not the same: fp32 online softmax in
exp2, fp32 probabilities into an fp32 P@V, Steel-MMA reassociation of the
256-term score contraction, and one split-K rescale per row.

So this lane is adopted on the same terms as ``MTPLX_QWEN4_HC_M4``: greedy
token agreement plus a full HumanEval run, never on a digest.  The install
probe below is a numerical SANITY gate, not the quality gate -- it exists so
an armed flag that is quietly wrong disables itself instead of shipping.

GATE DISCIPLINE -- REVISED 2026-09-02 AFTER AN ARMED-BUT-INERT WINDOW
---------------------------------------------------------------------
The 2026-09-02 16 K window armed this flag and measured the CONTROL: control
and candidate response texts were byte-identical on both finished seeds, on a
kernel whose arithmetic is rounding class.  Nothing in the run said so,
because every way the lane could decline was silent.  The rule the program
owner set afterwards is: **a flag either works on every request path the
server accepts, or it fails loudly at install.**  Concretely:

* CONTRACT failure RAISES, as before.  An armed flag on a pack the kernel
  cannot serve is a configuration error -- that is how MTPLX_FUSED_HC_V3 came
  to be armed-but-dead at M=4.
* PARITY failure still DISABLES *inside this module* -- :func:`install`
  returns False and records the measured deltas, so the numbers survive for
  the receipt -- but the CALLER
  (``graphbank.TensorOffsetQSACache.__init__``) then RAISES, because an armed
  arm that runs the stock chain is worse than an outage: it is a measurement
  that looks like a result.  Read the deltas off the stderr line and the
  receipt, then unarm the flag deliberately.
* ROUTE narrowing RAISES for a genuinely wrong CONFIGURATION at the width the
  flag arms -- a ratio that is not 4, a top-k that is not 512, a cache whose
  wired row count disagrees with the module.  Those are things an operator
  chose, and they cannot be fixed by declining.
* ROUTE narrowing ROUTES for the shapes a server is entitled to send.  Widths
  the flag does not arm (prefill rows, the S=1 D3 route under a verify-only
  arm), caches the lane never installed on, and -- the case that matters most
  -- a context below :data:`SHORT_CONTEXT_TOKENS`.  All return False; the last
  two are COUNTED in :data:`_ROUTE_DECLINES` so "the flag did nothing" always
  has a readable cause.
* An armed lane that is not in the traced verify graph RAISES from
  :func:`assert_traced`, called inside the compiled verify body -- unless the
  forward declined for short context, which is a legitimate stock lane.

THE LANE ENGAGES ONLY ABOVE 2,052 TOKENS, AND THAT IS THE DESIGN
-----------------------------------------------------------------
The kernel's ABI is a fixed ``[M, 512]`` block selection, and 512 complete
pooled blocks exist only from ``(512 + 1) * 4 = 2,052`` tokens
(:data:`SHORT_CONTEXT_TOKENS`).  Below that there is no smaller kernel to fall
to -- padding the selection would attend keys the shipped lane does not.
Context length is a per-REQUEST shape the server must accept, so a short
request routes to the stock attention, is counted, and prints nothing.  A
HumanEval prompt or a 1 K cell is served, not refused; an armed 16 K decode
still has to bind or the assertions fire.  This is the documented behaviour of
the flag, not a fallback.

The threshold is on the value ATTENTION passes as ``total_tokens``, which on a
fixed-capacity cache is the BANK CAPACITY -- ``update_and_fetch`` returns the
whole backing.  A 1,024-token prompt in a 2,048-token bank therefore has a
full 512-block budget and a context that has NOT crossed the boundary; the two
questions are different, and :func:`context_decline` asks both.  Asking only
the first cost the 1 K cell of the 2026-09-02 served battery.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


def _emit(line: str) -> None:
    """Put the lane's verdict where a benchmark log will actually see it.

    ``logger.info`` alone is INVISIBLE in a driver run -- the 2026-09-02
    window carried no engagement evidence at all, and the same log is missing
    ``[qwen4-fixed-M4-verify]`` and ``[qwen4-compiled-MTP-prepare]`` (both
    ``logger.info``) while it does carry ``[MTPLX_QWEN4_GRAPH_BUILD_OVERLAP]
    armed:`` (a plain ``print``).  The verify-glue lane fixed exactly this
    items; this is the same fix for this lane.
    """

    logger.info("%s", line)
    print(line, file=sys.stderr, flush=True)

#: Production Qwen3.8 Flash-Next QSA geometry -- compiled into the metallib.
Q_HEADS = 24
KV_HEADS = 2
GQA = 12
HEAD_DIM = 256
TOP_K = 512
COMPRESS_RATIO = 4
VERIFY_ROWS = 4
#: The kernel's own selected-token width; see ``mtplx.native`` for why this is
#: one less than the shipped lane's 2,052 and why the visible sets still agree.
SELECTED_TOKENS = TOP_K * COMPRESS_RATIO + (COMPRESS_RATIO - 1)

#: bf16 has an 8-bit significand, so its relative spacing is 2**-8 and its
#: unit roundoff (round to nearest) is u = 2**-9.
_BF16_REL_ULP = 2.0**-8
_BF16_UNIT_ROUNDOFF = 2.0**-9

# ---------------------------------------------------------------------------
# THE GATE, AND WHY IT IS TWO GATES AGAINST TWO REFERENCES
#
# The 2026-09-02 micro measured the kernel against the shipped path at
# max_abs 1.953e-3 (= 2**-9 exactly), rel_l2 4.78e-3, top-1 1.0000 -- and
# reported the SAME four significant figures for all twenty configurations:
# BK 64/128/256, DC 32/64, splits 4..32.  DC changes the fp32 score
# contraction order and the split count changes the online-softmax merge
# tree, so if any of that delta were the kernel's it would move.  It does not
# move at all.  The delta is therefore a property of the REFERENCE.
#
# The shipped path carries two bf16 roundings this kernel does not:
#
#   1. ``mx.matmul(q_view, k_view)`` has bf16 inputs, so its output is bf16 --
#      the SCORES are rounded to bf16 before ``.astype(float32) * scale`` and
#      the softmax.  A relative score error u shifts a softmax logit by
#      u*|x|, and perturbing logits by eps changes each probability by about
#      (eps_i - <eps>).  With scaled logits of order 5 that is ~2*u*5 = 2e-2
#      relative on the probabilities.
#   2. ``probs.astype(bfloat16)`` before P@V: another u = 2e-3 relative.
#
# So (1) should dominate (2) by roughly an order of magnitude, and the
# measured 4.78e-3 sits between the two predictions -- which is why the
# attribution is TESTED by a reference ladder rather than asserted.
#
# The consequence for the gate: a threshold on kernel-vs-shipped is really a
# threshold on how much bf16 rounding the SHIPPED path does, which is not a
# property of this kernel and cannot be tightened by improving it.  So:
#
#   * the DECIDING numerical gate is against the fp32 reference, where the
#     only differences left are fp32 reassociation and one bf16 store.  It is
#     tight, and it FAILS if the attribution above is wrong -- which is the
#     point of stating it this way.
#   * kernel-vs-shipped keeps a loose SANITY bound, an order of magnitude
#     above what the shipped path's own bf16 quantisation implies.
#
# Neither is a quality gate.  This kernel is rounding class; whether the
# difference matters is answered by model-level greedy-token agreement plus a
# full HumanEval run, exactly as for MTPLX_QWEN4_HC_M4.
# ---------------------------------------------------------------------------

#: TIGHT, vs :func:`fp32_reference`.  Derivation: both sides round the same
#: real number to bf16, so they differ by at most one bf16 ulp on any element
#: where the underlying fp32 values straddle a rounding boundary; fp32
#: reassociation over <= 2051 terms contributes ~sqrt(2051)*2**-24 = 2.7e-6
#: relative, three orders below a bf16 ulp, so it can only move an element
#: that already sits within 2.7e-6 of a boundary (about 1 in 4e4).  Hence a
#: couple of ulp at the extreme and a relative L2 far below the 2**-9/sqrt(3)
#: = 1.1e-3 that a uniformly re-rounded output would give.
PARITY_FP32_MAX_ABS_ULPS = 2.0
PARITY_FP32_MAX_REL_L2 = 5.0e-4

#: LOOSE, vs :func:`stock_reference` (the shipped path).  This bounds the
#: shipped path's OWN bf16 score and probability quantisation, derived above
#: at ~2e-2 relative on the probabilities; 5e-2 leaves an order of magnitude
#: over the 4.78e-3 measured on 2026-09-02.  It exists to catch a kernel that
#: is wrong by a factor, not to certify one that is right.
PARITY_SHIPPED_MAX_REL_L2 = 5.0e-2

#: Fraction of (head, row) pairs whose argmax over the head dimension must
#: still agree, against BOTH references.  A coarse discrete statistic,
#: reported for continuity with the "top-1 agreement" the program asks for at
#: the kernel level; the DECIDING top-1 number is model-level greedy-token
#: agreement.  Measured 1.0000 on every configuration.
PARITY_MIN_TOP1 = 0.98

#: Probe geometry: capacity only has to clear the dense/sparse crossover
#: (``total // 4 > 512``), and the split geometry does not depend on context
#: length at all -- it is a function of SELECTED_TOKENS and the tile -- so a
#: 4,096-token probe exercises the same grid a 17,408-token cycle does.
PROBE_CAPACITY = 4096

_COUNTS: Dict[str, int] = {
    "verify_kernel": 0,
    "probe_runs": 0,
    "probe_failures": 0,
    # Every ``TensorOffsetQSACache`` that bound the lane, shadow and parity
    # twins included.  The probe runs on the FIRST one only (the verdict is
    # per process), so a value of ZERO with the flag armed is the whole
    # finding: no cache carried the lane and the kernel could not have run.
    "cache_installs": 0,
    # Every routing decision that came back True, summed over the call sites
    # in ``_ROUTE_SITES``.  Trace-time, like ``verify_kernel``: the Python body
    # of a compiled verify graph runs once per retrace, so read these as "did
    # this lane get into the graph at all", never as a per-cycle count.
    "route_hits": 0,
    # Forwards that routed to the stock chain because of the REQUEST's own
    # shape -- its context length, its row count, its block budget -- rather
    # than the configuration.  Counted separately because the engagement
    # assertions accept these and only these: see :func:`context_decline`.
    "request_declines": 0,
}

#: Smallest context the lane can serve, in tokens.  The kernel's ABI is a
#: fixed ``[M, TOP_K]`` selection, so it needs a FULL budget: ``TOP_K``
#: complete pooled blocks exist only once the context reaches
#: ``(TOP_K + 1) * COMPRESS_RATIO`` = 2,052 tokens.  Below that there is no
#: analogue -- not a smaller kernel, not a padded one -- so the stock chain
#: serves the request and the receipt says how often.
#:
#: This is exactly ``mtplx.native``'s
#: ``total_tokens // _COMPRESS_RATIO <= _TOP_K_BLOCKS`` boundary, restated in
#: tokens.  ``tests/test_qsa_sparse_decode_wiring.py`` pins the two
#: together.
SHORT_CONTEXT_TOKENS = (TOP_K + 1) * COMPRESS_RATIO

#: Largest context the kernel is instantiated for; mirrors
#: ``mtplx.native._MAX_CONTEXT``.
MAX_CONTEXT = 1_048_576

#: Observed extremes for the request-shape declines, so the receipt carries
#: the numbers without giving ``_ROUTE_DECLINES`` an unbounded key space (one
#: key per distinct context length would grow without bound in a server).
_DECLINE_EXTREMES: Dict[str, int] = {}

#: ``site -> hits``.  The 2026-09-02 window failed because the one call site
#: that could reach the verify width asked for a width the flag did not arm;
#: a single total would not have shown that, so the sites are named.
_ROUTE_SITES: Dict[str, int] = {}

#: ``reason -> count`` for the routing narrowings that are NOT failures (a
#: width the flag does not arm, a growable cache the lane never installed on).
#: Recorded rather than merely returned, so "the flag did nothing" always has
#: a readable cause in the receipt.
_ROUTE_DECLINES: Dict[str, int] = {}

#: ``None`` until the probe has run.  ``""`` once it has passed.  A non-empty
#: string is the reason the lane is disabled for this process.
_DISABLED_REASON: Optional[str] = None
_PROBE_REPORT: Dict[str, Any] = {}


class SparseDecodeContractError(RuntimeError):
    """An armed flag met a request path it cannot serve.

    Raised, never swallowed: a lane that quietly declines makes the candidate
    arm measure the control while its receipt claims otherwise, which is the
    exact failure the 2026-09-02 window produced.
    """


def armed() -> bool:
    """True when this process armed the flag."""

    from mtplx.runtime_options import qsa_sparse_decode_enabled

    return bool(qsa_sparse_decode_enabled())


def pending() -> bool:
    """True while the install probe has not run yet in this process."""

    return _DISABLED_REASON is None


def installed() -> bool:
    """True when the probe ran and passed.  False while pending, too."""

    return _DISABLED_REASON == ""


def note_route_hit(site: str) -> None:
    """One routing decision resolved to the kernel, at a named call site."""

    _COUNTS["route_hits"] += 1
    _ROUTE_SITES[site] = _ROUTE_SITES.get(site, 0) + 1


def note_route_decline(reason: str) -> None:
    """One routing narrowing that is not an error, recorded by cause."""

    _ROUTE_DECLINES[reason] = _ROUTE_DECLINES.get(reason, 0) + 1


#: THE MIRROR.  Every branch of
#: ``mtplx.native.qsa_sparse_gqa_decode_unsupported_reason`` that depends on
#: the REQUEST -- its context length, its row count, its block budget -- paired
#: with the stable key :func:`context_decline` reports it under.  Everything
#: else in that function is CONFIGURATION (dtypes, shapes, the tile, the split
#: count, the device, the build) and raises.
#:
#: Two request-shape raises reached production before this list existed. The
#: first was ``k_eff != TOP_K`` (2026-09-02, HTTP 500 on every HumanEval
#: prompt). The second was the one this list is named for: a 1,024-token
#: prompt whose FIXED bank is 2,048 tokens has a full 512-block budget --
#: ``k_eff`` is 512, so the first gate passed -- while the kernel's own
#: boundary is ``total_tokens // 4 > 512``, i.e. 2,052 tokens, so the call
#: died inside ``attention()`` with "the context has not crossed the
#: dense/sparse boundary". A partial mirror is how that happens, so
#: ``tests/test_qsa_sparse_decode_wiring.py`` pins this list against the
#: native source.
REQUEST_SHAPE_DECLINES = (
    ("empty_context", "total_tokens must describe a non-empty context"),
    ("rows_exceed_context", "the query rows must fit inside total_tokens"),
    (
        "context_exceeds_capacity",
        "the logical token count exceeds the full K/V backing capacity",
    ),
    (
        "context_above_limit",
        "the logical token count exceeds the production context limit",
    ),
    ("short_context", "the context has not crossed the dense/sparse boundary"),
    # Not a native branch: the indexer's own budget, which the kernel's fixed
    # [M, TOP_K] ABI requires and which a short context cannot fill.
    ("partial_budget", None),
)

#: The native reason strings the mirror above claims.  A reason in this set
#: reaching :func:`attention` means the routing predicate did not mirror the
#: kernel, which is a wiring bug and says so.
REQUEST_SHAPE_REASONS = frozenset(
    reason for _key, reason in REQUEST_SHAPE_DECLINES if reason is not None
)


def context_decline(
    *, total_tokens: int, rows: int, k_eff: int, capacity: int
) -> Optional[str]:
    """The request-shape verdict, from host ints alone.  ``None`` = servable.

    Mirrors :data:`REQUEST_SHAPE_DECLINES` in the native contract's own order.
    Called by the indexer BEFORE it commits the forward to this lane, because
    that is the last point at which the stock chain is still reachable: once
    the selection returns ``("sparse_blocks", top_idx)`` the rows-gather token
    list was never built and there is nothing to fall back to.

    ``total_tokens`` must be the value the ATTENTION call site will pass, not
    the logical context. On a fixed-capacity cache ``update_and_fetch``
    returns the whole backing, so attention's ``T`` is the bank capacity --
    which is exactly how a 1,024-token prompt (2,048-token bank, 512 complete
    blocks, full budget) reached the kernel and was refused by it.
    """

    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        return "empty_context"
    if int(rows) > total_tokens:
        return "rows_exceed_context"
    if total_tokens > int(capacity):
        return "context_exceeds_capacity"
    if total_tokens > MAX_CONTEXT:
        return "context_above_limit"
    if total_tokens // COMPRESS_RATIO <= TOP_K:
        return "short_context"
    if int(k_eff) != TOP_K:
        return "partial_budget"
    return None


def note_request_decline(
    site: str, reason: str, *, total_tokens: int, blocks: int
) -> None:
    """This forward's own SHAPE is outside the lane.  Routing, not failure.

    A server accepts whatever context length it is sent, and the kernel has no
    analogue below a full budget -- not a smaller kernel, not a padded one.
    Counted (never printed: this runs once per QSA layer per request) and
    accepted by :func:`assert_traced`, so a short prompt runs the stock chain
    instead of returning a 500.

    The numbers ride in :data:`_DECLINE_EXTREMES` as min/max pairs rather than
    in the decline key, which would otherwise grow one key per distinct
    context length.
    """

    _COUNTS["request_declines"] += 1
    note_route_decline(f"{site}: {reason}")
    for name, value in (("blocks", int(blocks)), ("tokens", int(total_tokens))):
        low = _DECLINE_EXTREMES.get(f"{name}_min")
        high = _DECLINE_EXTREMES.get(f"{name}_max")
        _DECLINE_EXTREMES[f"{name}_min"] = value if low is None else min(low, value)
        _DECLINE_EXTREMES[f"{name}_max"] = value if high is None else max(high, value)


def route_snapshot() -> Dict[str, int]:
    """The two counters to sample around a forward, for :func:`assert_traced`.

    A forward proves the armed lane engaged either by ROUTING to the kernel
    (``route_hits``) or by declining for a REQUEST SHAPE the contract cannot
    serve (``request_declines``).  Anything else is an inert flag.
    """

    return {
        "route_hits": int(_COUNTS["route_hits"]),
        "request_declines": int(_COUNTS["request_declines"]),
    }


def route_counters() -> Dict[str, Any]:
    """Per-site hits and per-cause declines, for the receipt."""

    return {
        "route_hits": int(_COUNTS["route_hits"]),
        "route_sites": dict(_ROUTE_SITES),
        "route_declines": dict(_ROUTE_DECLINES),
        "request_declines": int(_COUNTS["request_declines"]),
        "request_decline_extremes": dict(_DECLINE_EXTREMES),
        "short_context_tokens": SHORT_CONTEXT_TOKENS,
    }


def engagement() -> Dict[str, Any]:
    """Snapshot of the lane's engagement counters and install verdict.

    ``verify_kernel`` is the ENGAGEMENT LINE: if an ABBA reports a win and it
    is zero, the win came from somewhere else.
    """

    report = dict(_COUNTS)
    report["installed"] = _DISABLED_REASON == ""
    report["disabled_reason"] = _DISABLED_REASON or None
    report["probe"] = dict(_PROBE_REPORT)
    report["route_sites"] = dict(_ROUTE_SITES)
    report["route_declines"] = dict(_ROUTE_DECLINES)
    return report


def receipt() -> Dict[str, Any]:
    """The compact engagement block a benchmark receipt stores.

    Never raises: it reads ``_DISABLED_REASON`` directly rather than going
    through a helper that treats "pending" as an error, because describing the
    pending state IS what a receipt builder needs to do.
    """

    from mtplx.runtime_options import (
        qsa_sparse_decode_splits,
        qsa_sparse_decode_tile,
    )

    key_tile, dim_tile = qsa_sparse_decode_tile()
    block: Dict[str, Any] = {
        "armed": armed(),
        "installed": installed(),
        "pending": pending(),
        "disabled_reason": _DISABLED_REASON or None,
        "tile": [int(key_tile), int(dim_tile)],
        "splits": int(qsa_sparse_decode_splits()),
        "verify_rows": VERIFY_ROWS,
        "cache_installs": int(_COUNTS["cache_installs"]),
        "probe_runs": int(_COUNTS["probe_runs"]),
        "probe_failures": int(_COUNTS["probe_failures"]),
        "kernel_calls": {
            "verify_kernel": int(_COUNTS["verify_kernel"]),
        },
        "probe": dict(_PROBE_REPORT),
    }
    block.update(route_counters())
    return block


def engagement_line(*, enabled: bool) -> str:
    """The one-line install verdict, in the shape the other Fable lanes use."""

    from mtplx.runtime_options import (
        qsa_sparse_decode_splits,
        qsa_sparse_decode_tile,
    )

    if not enabled:
        reason = _DISABLED_REASON or "install probe has not run"
        return f"[mtplx] qsa_sparse_decode: off ({reason})"
    key_tile, dim_tile = qsa_sparse_decode_tile()
    worst = _PROBE_REPORT.get("worst") or {}
    fp32 = worst.get("vs_fp32") or {}
    shipped = worst.get("vs_shipped") or {}
    widths = [str(VERIFY_ROWS)] if armed() else []
    return (
        "[mtplx] qsa_sparse_decode armed: "
        f"rows={'+'.join(widths) or '-'} "
        f"tile={int(key_tile)}:{int(dim_tile)} "
        f"splits={int(qsa_sparse_decode_splits())} "
        f"caches={int(_COUNTS['cache_installs'])} "
        f"probe cell={worst.get('cell')!r} "
        f"vs_fp32 ulps={fp32.get('max_abs_ulps', float('nan')):.3f} "
        f"rel_l2={fp32.get('rel_l2', float('nan')):.3e} "
        f"top1={fp32.get('top1', float('nan')):.4f} "
        f"vs_shipped rel_l2={shipped.get('rel_l2', float('nan')):.3e} "
        f"probe_runs={int(_COUNTS['probe_runs'])}"
    )


def assert_traced(rows: int, *, before: Dict[str, int], where: str) -> None:
    """The armed verify lane must be IN this graph, not merely armed.

    ``before`` is :func:`route_snapshot` sampled before the traced forward.  A
    trace of the armed width that ends with no additional route hit is an
    inert flag, and replaying that graph a few hundred times produces a delta
    nobody can attribute -- which is what the 2026-09-02 window did.

    A forward that declined for its own REQUEST SHAPE satisfies this -- see
    :func:`context_decline`.  Those are the only declines the assertion
    accepts, and they are why a 1 K prompt does not take the server down with
    the flag armed.
    """

    if not armed() or int(rows) != VERIFY_ROWS:
        return
    now = route_snapshot()
    if now["route_hits"] > int(before["route_hits"]):
        return
    if now["request_declines"] > int(before["request_declines"]):
        return
    raise SparseDecodeContractError(
        "MTPLX_QSA_SPARSE_DECODE is armed but the split-K kernel is not "
        f"in the traced {where} graph at {int(rows)} rows, and this forward's "
        "shape is one the lane can serve, so it should have: the QSA "
        "attention took another path, and this arm would replay the stock "
        f"chain. route_sites={dict(_ROUTE_SITES)} "
        f"declines={dict(_ROUTE_DECLINES)}"
    )


def disabled_reason() -> Optional[str]:
    """The reason the lane is off, or ``None`` while it is usable/pending."""

    return _DISABLED_REASON or None


def reset_for_tests() -> None:
    """Clear the process verdict and counters.  Tests only."""

    global _DISABLED_REASON
    _DISABLED_REASON = None
    _PROBE_REPORT.clear()
    _ROUTE_SITES.clear()
    _ROUTE_DECLINES.clear()
    _DECLINE_EXTREMES.clear()
    for key in _COUNTS:
        _COUNTS[key] = 0


# ---------------------------------------------------------------------------
# The visible set -- one definition, shared by the kernel model, the
# reference and the tests.
# ---------------------------------------------------------------------------
def visible_block_count(q_abs: int) -> int:
    """``visible_blocks`` exactly as ``_SRC_ROW_TOKENS`` computes it."""

    return (int(q_abs) + 1) // COMPRESS_RATIO


def shipped_row_tokens(
    top_idx_row, q_abs: int, *, topk: int = TOP_K
) -> Tuple[list, list]:
    """Host model of the shipped rows-gather token list, for ONE row.

    Integers only.

    Returns ``(token_idx, token_ok)`` of width ``topk*ratio + ratio``, the
    closed form the shipped Metal kernel writes:

        slot < topk*ratio : block = top_idx[slot // ratio]
                            token = block*ratio + slot % ratio
                            ok    = block < visible_blocks
        otherwise         : token = visible_blocks*ratio + (slot - topk*ratio)
                            ok    = token <= q_abs

    Note ``ok`` is evaluated PER SLOT against the block id, not against a
    prefix length: ``top_idx`` is ``mx.argpartition``'s output and is not
    sorted.
    """

    ratio = COMPRESS_RATIO
    visible = visible_block_count(q_abs)
    idx: list = []
    ok: list = []
    for slot in range(topk * ratio):
        block = int(top_idx_row[slot // ratio])
        token = block * ratio + (slot % ratio)
        good = block < visible
        ok.append(good)
        idx.append(token if good else 0)
    for within in range(ratio):
        token = visible * ratio + within
        good = token <= int(q_abs)
        ok.append(good)
        idx.append(token if good else 0)
    return idx, ok


def kernel_row_tokens(
    top_idx_row, q_abs: int, *, key_length: int, topk: int = TOP_K
) -> list:
    """Host model of the SPLIT KERNEL's per-slot selection for ONE row.

    Returns the kernel's ``selected[]`` array: the absolute key position for
    each of the ``topk*ratio + ratio - 1`` slots, or ``-1`` for a masked one.
    Written to be readable against the MSL in
    ``steel_qsa_sparse_gqa_decode.h``; ``tests/test_qsa_sparse_decode.py``
    pins it against :func:`shipped_row_tokens`.
    """

    ratio = COMPRESS_RATIO
    visible = visible_block_count(q_abs)
    out: list = []
    for slot in range(topk * ratio):
        block = int(top_idx_row[slot // ratio])
        pos = -1
        if 0 <= block < visible:
            candidate = block * ratio + (slot % ratio)
            if 0 <= candidate < int(key_length):
                pos = candidate
        out.append(pos)
    for within in range(ratio - 1):
        candidate = visible * ratio + within
        pos = -1
        if 0 <= candidate < int(key_length) and candidate <= int(q_abs):
            pos = candidate
        out.append(pos)
    return out


def visible_sets_agree(
    top_idx_row, q_abs: int, *, key_length: int, topk: int = TOP_K
) -> bool:
    """True when kernel and shipped lane attend the SAME multiset of keys."""

    idx, ok = shipped_row_tokens(top_idx_row, q_abs, topk=topk)
    shipped = sorted(t for t, good in zip(idx, ok) if good)
    kernel = sorted(p for p in kernel_row_tokens(
        top_idx_row, q_abs, key_length=key_length, topk=topk
    ) if p >= 0)
    return shipped == kernel


# ---------------------------------------------------------------------------
# The reference the probe compares against: the shipped rows-gather lane,
# transcribed.
# ---------------------------------------------------------------------------
def _row_tokens_mx(top_idx: mx.array, q_offset, *, topk: int) -> Tuple[mx.array, mx.array]:
    """The rows-gather token list's closed form in plain MLX."""

    ratio = COMPRESS_RATIO
    rows = int(top_idx.shape[0])
    offsets = mx.arange(rows, dtype=mx.int32)
    qpos = (
        q_offset.reshape(1).astype(mx.int32) + offsets
        if isinstance(q_offset, mx.array)
        else mx.array(int(q_offset), dtype=mx.int32) + offsets
    )
    visible = (qpos + 1) // ratio  # [rows]
    within = mx.arange(ratio, dtype=mx.int32)
    blocks = top_idx.astype(mx.int32)  # [rows, topk]
    block_tokens = (blocks[:, :, None] * ratio + within).reshape(rows, topk * ratio)
    block_ok = mx.repeat(blocks < visible[:, None], ratio, axis=1)
    tail_tokens = visible[:, None] * ratio + within
    tail_ok = tail_tokens <= qpos[:, None]
    token_idx = mx.concatenate([block_tokens, tail_tokens], axis=1)
    token_ok = mx.concatenate([block_ok, tail_ok], axis=1)
    token_idx = mx.where(token_ok, token_idx, mx.array(0, dtype=mx.int32))
    return token_idx, token_ok


def reference_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    top_idx: mx.array,
    *,
    query_offset,
    scale: float,
    topk: int = TOP_K,
    fp32_scores: bool,
    fp32_probs: bool,
) -> mx.array:
    """The rows-gather attention over the SAME visible set, one rung at a time.

    Transcribed from ``mtplx/models/qwen4_exp.py::_qsa_rows_gather_attention``
    so the probe compares against ONE definition and does not import the model
    into a kernel module.  ``keys``/``values`` are the full
    ``[1, 2, capacity, 256]`` backing; every gathered index is an absolute row
    inside it, exactly as in the shipped lane.

    The two flags are the reference LADDER, and they exist to attribute the
    kernel-vs-shipped delta rather than assume it:

    ``fp32_scores=False``  reproduces the shipped path exactly -- ``mx.matmul``
        on bf16 operands returns bf16, so the scores are rounded to bf16
        BEFORE the ``astype(float32) * scale`` and the softmax.  Setting it
        True upcasts q and k first (exact, bf16 -> fp32) so only the
        accumulation and output precision change.
    ``fp32_probs=False``  reproduces the shipped ``probs.astype(bfloat16)``
        before P@V.  Setting it True keeps fp32 probabilities and upcasts V
        (also exact).

    Every rung returns ``queries.dtype``, so all three are comparable element
    for element against the kernel's own bf16 output.
    """

    token_idx, token_ok = _row_tokens_mx(top_idx, query_offset, topk=topk)
    rows = int(queries.shape[2])
    width = int(token_idx.shape[1])
    k_sel = mx.take(keys, token_idx.reshape(-1), axis=2).reshape(
        1, KV_HEADS, rows, width, HEAD_DIM
    )
    v_sel = mx.take(values, token_idx.reshape(-1), axis=2).reshape(
        1, KV_HEADS, rows, width, HEAD_DIM
    )
    neg = mx.array(-mx.inf, dtype=mx.float32)
    q_view = queries.reshape(1, KV_HEADS, GQA, rows, 1, HEAD_DIM)
    k_view = k_sel.swapaxes(-1, -2).reshape(1, KV_HEADS, 1, rows, HEAD_DIM, width)
    if fp32_scores:
        # Upcasting bf16 -> fp32 is exact, so this changes the GEMM's output
        # precision and nothing about the operands.
        q_view = q_view.astype(mx.float32)
        k_view = k_view.astype(mx.float32)
    scores = mx.matmul(q_view, k_view).squeeze(-2).astype(mx.float32) * scale
    scores = mx.where(token_ok[None, None, None], scores, neg)
    probs = mx.softmax(scores, axis=-1)
    v_view = v_sel.reshape(1, KV_HEADS, 1, rows, width, HEAD_DIM)
    if fp32_probs:
        v_view = v_view.astype(mx.float32)
    else:
        probs = probs.astype(queries.dtype)
    out = mx.matmul(probs[..., None, :], v_view).squeeze(-2)
    return out.reshape(1, Q_HEADS, rows, HEAD_DIM).astype(queries.dtype)


def stock_reference(queries, keys, values, top_idx, **kwargs) -> mx.array:
    """The shipped path, exactly: bf16 scores AND bf16 probabilities."""

    return reference_attention(
        queries, keys, values, top_idx, fp32_scores=False, fp32_probs=False, **kwargs
    )


def shipped_fp32_probs_reference(queries, keys, values, top_idx, **kwargs) -> mx.array:
    """Shipped, minus only the bf16 probability cast.  Attribution rung."""

    return reference_attention(
        queries, keys, values, top_idx, fp32_scores=False, fp32_probs=True, **kwargs
    )


def fp32_reference(queries, keys, values, top_idx, **kwargs) -> mx.array:
    """What the KERNEL computes: fp32 scores and fp32 probabilities.

    The deciding numerical reference.  Against this the kernel's only
    remaining differences are fp32 reassociation (Steel MMA fragments and the
    split-K merge, both ~1e-6 relative) and the single bf16 store, so a delta
    here of the same size as the kernel-vs-shipped delta would falsify the
    attribution in this module's gate note.
    """

    return reference_attention(
        queries, keys, values, top_idx, fp32_scores=True, fp32_probs=True, **kwargs
    )


# ---------------------------------------------------------------------------
# Contract + install
# ---------------------------------------------------------------------------
def check_cache_contract(keys: mx.array, values: mx.array, ratio: int) -> None:
    """The cache half of the lane's contract.  RAISES; never returns False."""

    if int(ratio) != COMPRESS_RATIO:
        raise RuntimeError(
            "MTPLX_QSA_SPARSE_DECODE is wired for the ratio-4 QSA lane; "
            f"got ratio={ratio}"
        )
    if not mx.metal.is_available():
        raise RuntimeError(
            "MTPLX_QSA_SPARSE_DECODE is a Metal kernel and has no "
            "portable spelling"
        )
    from mtplx.native import native_qsa_available

    if not native_qsa_available():
        raise RuntimeError(
            "MTPLX_QSA_SPARSE_DECODE requires the built native "
            "extension (native_extensions/qsa_sparse_gqa); build it with the "
            "cmake command in mtplx/native/__init__.py's docstring"
        )
    for name, arr in (("keys", keys), ("values", values)):
        if arr is None:
            raise RuntimeError(
                f"MTPLX_QSA_SPARSE_DECODE requires a materialized {name} "
                "bank"
            )
        if arr.ndim != 4 or tuple(int(x) for x in arr.shape)[:2] != (1, KV_HEADS):
            raise RuntimeError(
                "MTPLX_QSA_SPARSE_DECODE requires a "
                f"[1, {KV_HEADS}, capacity, {HEAD_DIM}] {name} bank; got "
                f"{tuple(arr.shape)}"
            )
        if int(arr.shape[3]) != HEAD_DIM:
            raise RuntimeError(
                "MTPLX_QSA_SPARSE_DECODE is wired for head_dim "
                f"{HEAD_DIM}; got {int(arr.shape[3])}"
            )
        if arr.dtype not in (mx.bfloat16, mx.float16):
            raise RuntimeError(
                "MTPLX_QSA_SPARSE_DECODE is wired for bf16/fp16 K/V; "
                f"got {name} dtype {arr.dtype}"
            )
    if keys.dtype != values.dtype:
        raise RuntimeError(
            "MTPLX_QSA_SPARSE_DECODE requires one K/V dtype; got "
            f"{keys.dtype} and {values.dtype}"
        )


def _probe_cell(dtype: mx.Dtype, rows: int, total_tokens: int, q_offset: int, seed: int):
    """One synthetic probe cell on the production geometry."""

    mx.random.seed(seed)
    nb_total = total_tokens // COMPRESS_RATIO
    queries = mx.random.normal((1, Q_HEADS, rows, HEAD_DIM)).astype(dtype)
    keys = mx.random.normal((1, KV_HEADS, PROBE_CAPACITY, HEAD_DIM)).astype(dtype)
    values = mx.random.normal((1, KV_HEADS, PROBE_CAPACITY, HEAD_DIM)).astype(dtype)
    # Deliberately UNSORTED distinct block ids, drawn from the whole logical
    # range so cells with few complete blocks really do carry invisible ids --
    # which is the case a leading-prefix validity cut would get wrong.
    ids = mx.argsort(mx.random.uniform(shape=(rows, nb_total)), axis=-1)
    top_idx = ids[:, :TOP_K].astype(mx.int32)
    return queries, keys, values, top_idx, total_tokens, q_offset


def _compare(reference: mx.array, candidate: mx.array) -> Dict[str, float]:
    """Host-side parity statistics.  One eval, at install, never in the hot path."""

    ref = reference.astype(mx.float32)
    got = candidate.astype(mx.float32)
    diff = mx.abs(ref - got)
    ref_absmax = mx.max(mx.abs(ref))
    l2_diff = mx.sqrt(mx.sum(diff * diff))
    l2_ref = mx.sqrt(mx.sum(ref * ref))
    top1 = mx.mean(
        (mx.argmax(ref, axis=-1) == mx.argmax(got, axis=-1)).astype(mx.float32)
    )
    stats = mx.stack(
        [mx.max(diff), mx.mean(diff), ref_absmax, l2_diff, l2_ref, top1]
    )
    mx.eval(stats)
    max_abs, mean_abs, absmax, l2d, l2r, top1_f = (float(x) for x in stats.tolist())
    scale = max(absmax, 1e-3)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "ref_absmax": absmax,
        "max_abs_ulps": max_abs / (_BF16_REL_ULP * scale),
        "rel_l2": l2d / max(l2r, 1e-12),
        "top1": top1_f,
    }


def probe_cells(rows: int) -> list:
    """The install-time parity-probe cells for a verify window of ``rows``.

    Each cell is ``(name, rows, total_tokens, q_offset, seed)``: a long-context
    cell (every selected block visible) and a just-past-crossover cell (some
    selected ids are NOT visible).

    ``q_offset`` is DERIVED as ``total_tokens - rows``, never a constant,
    because that is the real-serving invariant: the model passes
    ``query_offset = cache.offset`` and ``total_tokens = keys.shape[2]`` (see
    ``mtplx/models/qwen4_exp.py``: ``pos_start = cache.offset``,
    ``T = k.shape[2]``), and the ``rows`` query tokens are the ones just
    written to the cache, so they occupy positions ``[T-rows, T)`` and every
    ``q_abs < total_tokens``. A constant offset sized for M4 (``total - 4``)
    would push extra rows to positions ``>= total_tokens`` where the kernel
    drops out-of-context tail tokens while a plain-MLX reference still attends
    them -- a false parity failure. Deriving keeps M4 byte-identical
    (``4093-4=4089``, ``2052-4=2048``).
    """

    rows = int(rows)
    return [
        ("verify-4096", rows, 4093, 4093 - rows, 20260902),
        ("verify-crossover", rows, 2052, 2052 - rows, 20260903),
    ]


def install(
    keys: mx.array,
    values: mx.array,
    *,
    compress_ratio: int,
    verify: bool,
) -> bool:
    """Contract-check and parity-probe the lane once per process.

    Returns True when the lane is usable.  RAISES on a contract failure;
    DISABLES (returns False, records the reason) on a parity failure.  Called
    from ``TensorOffsetQSACache`` at cache install -- model build time,
    outside any ``mx.compile`` trace.
    """

    global _DISABLED_REASON
    if not verify:
        return False
    # Every cache that binds the lane counts, including the ones that reuse
    # the process verdict: 12 QSA caches on the production pack, and a
    # receipt showing fewer means some layers kept the stock chain.
    _COUNTS["cache_installs"] += 1
    if _DISABLED_REASON is not None:
        return _DISABLED_REASON == ""

    check_cache_contract(keys, values, compress_ratio)

    from mtplx.native import (
        qsa_sparse_gqa_decode,
        qsa_sparse_gqa_decode_unsupported_reason,
    )
    from mtplx.runtime_options import (
        qsa_sparse_decode_splits,
        qsa_sparse_decode_tile,
    )

    key_tile, dim_tile = qsa_sparse_decode_tile()
    key_splits = qsa_sparse_decode_splits()
    scale = float(HEAD_DIM) ** -0.5
    dtype = keys.dtype

    cells = probe_cells(VERIFY_ROWS) if verify else []

    worst: Dict[str, Any] = {}
    for name, rows, total, offset, seed in cells:
        q, k, v, top_idx, total_tokens, q_offset = _probe_cell(
            dtype, rows, total, offset, seed
        )
        reason = qsa_sparse_gqa_decode_unsupported_reason(
            q,
            k,
            v,
            top_idx,
            query_offset=q_offset,
            total_tokens=total_tokens,
            scale=scale,
            key_tile=key_tile,
            dimension_tile=dim_tile,
            key_splits=key_splits,
        )
        if reason is not None:
            # A contract miss on the lane's OWN synthetic production geometry
            # is a configuration error, not a numerical verdict.
            raise RuntimeError(
                f"MTPLX_QSA_SPARSE_DECODE cannot serve its own probe "
                f"cell {name!r}: {reason}"
            )
        _COUNTS["probe_runs"] += 1
        candidate = qsa_sparse_gqa_decode(
            q,
            k,
            v,
            top_idx,
            query_offset=q_offset,
            total_tokens=total_tokens,
            scale=scale,
            key_tile=key_tile,
            dimension_tile=dim_tile,
            key_splits=key_splits,
        )
        # Both rungs: the fp32 reference decides, the shipped one is a sanity
        # bound on the shipped path's own bf16 quantisation.  See the gate
        # note at the top of this module for why they are not one number.
        vs_fp32 = _compare(
            fp32_reference(q, k, v, top_idx, query_offset=q_offset, scale=scale),
            candidate,
        )
        vs_shipped = _compare(
            stock_reference(q, k, v, top_idx, query_offset=q_offset, scale=scale),
            candidate,
        )
        stats = {"cell": name, "vs_fp32": vs_fp32, "vs_shipped": vs_shipped}
        if not worst or vs_fp32["max_abs_ulps"] > worst["vs_fp32"]["max_abs_ulps"]:
            worst = stats
        _PROBE_REPORT[name] = stats

    _PROBE_REPORT["worst"] = worst
    _PROBE_REPORT["tile"] = [key_tile, dim_tile]
    _PROBE_REPORT["key_splits"] = key_splits

    failures = []
    tight = worst.get("vs_fp32", {})
    loose = worst.get("vs_shipped", {})
    if tight.get("max_abs_ulps", math.inf) > PARITY_FP32_MAX_ABS_ULPS:
        failures.append(
            f"vs fp32 reference: max abs diff {tight['max_abs']:.3e} = "
            f"{tight['max_abs_ulps']:.2f} bf16 ulp "
            f"(limit {PARITY_FP32_MAX_ABS_ULPS})"
        )
    if tight.get("rel_l2", math.inf) > PARITY_FP32_MAX_REL_L2:
        failures.append(
            f"vs fp32 reference: relative L2 {tight['rel_l2']:.3e} "
            f"(limit {PARITY_FP32_MAX_REL_L2})"
        )
    if tight.get("top1", 0.0) < PARITY_MIN_TOP1:
        failures.append(
            f"vs fp32 reference: head-dim top-1 {tight['top1']:.4f} "
            f"(limit {PARITY_MIN_TOP1})"
        )
    if loose.get("rel_l2", math.inf) > PARITY_SHIPPED_MAX_REL_L2:
        failures.append(
            f"vs shipped path: relative L2 {loose['rel_l2']:.3e} "
            f"(limit {PARITY_SHIPPED_MAX_REL_L2}) -- this bounds the SHIPPED "
            "path's own bf16 score and probability casts, so exceeding it "
            "means the kernel is wrong by a factor, not merely re-rounded"
        )
    if loose.get("top1", 0.0) < PARITY_MIN_TOP1:
        failures.append(
            f"vs shipped path: head-dim top-1 {loose['top1']:.4f} "
            f"(limit {PARITY_MIN_TOP1})"
        )

    if failures:
        _COUNTS["probe_failures"] += 1
        _DISABLED_REASON = (
            f"parity probe failed on cell {worst.get('cell')!r}: "
            + "; ".join(failures)
        )
        _emit(engagement_line(enabled=False))
        _emit(
            "[mtplx] qsa_sparse_decode install: "
            + json.dumps(receipt(), sort_keys=True)
        )
        return False

    _DISABLED_REASON = ""
    _emit(engagement_line(enabled=True))
    _emit(
        "[mtplx] qsa_sparse_decode install: "
        + json.dumps(receipt(), sort_keys=True)
    )
    return True


# ---------------------------------------------------------------------------
# Hot path
# ---------------------------------------------------------------------------
def attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    top_idx: mx.array,
    *,
    query_offset,
    total_tokens: int,
    scale: float,
) -> mx.array:
    """Run the split-K kernel for one QSA layer.  Raises on a contract miss.

    ``queries`` is the ``[1, 24, M, 256]`` transposed view the attention
    module already holds; ``keys``/``values`` are the FULL cache backing.
    ``top_idx`` is ``mx.argpartition``'s ``[M, 512]`` output in its own
    order.
    """

    from mtplx.native import (
        qsa_sparse_gqa_decode,
        qsa_sparse_gqa_decode_unsupported_reason,
    )
    from mtplx.runtime_options import (
        qsa_sparse_decode_splits,
        qsa_sparse_decode_tile,
    )

    key_tile, dim_tile = qsa_sparse_decode_tile()
    key_splits = qsa_sparse_decode_splits()
    reason = qsa_sparse_gqa_decode_unsupported_reason(
        queries,
        keys,
        values,
        top_idx,
        query_offset=query_offset,
        total_tokens=total_tokens,
        scale=scale,
        key_tile=key_tile,
        dimension_tile=dim_tile,
        key_splits=key_splits,
    )
    if reason is not None:
        if reason in REQUEST_SHAPE_REASONS:
            # Unreachable when the mirror is complete, and that is the point of
            # saying so: the INDEXER owns every request-shape decision, because
            # by the time a forward reaches here the rows-gather token list was
            # never built and there is nothing to fall back to.  A 1,024-token
            # prompt died exactly here on 2026-09-02.
            raise SparseDecodeContractError(
                "MTPLX_QSA_SPARSE_DECODE routing did not mirror the "
                f"kernel contract: the kernel refused this call because {reason!r}, "
                "which is a REQUEST shape that "
                "kernels/qsa_sparse_decode.context_decline must have declined "
                "before the selection committed to this lane. Fix the mirror "
                "in REQUEST_SHAPE_DECLINES; do not make the request fail"
            )
        raise RuntimeError(
            "MTPLX_QSA_SPARSE_DECODE is armed but this call is off "
            f"contract: {reason}"
        )
    _COUNTS["verify_kernel"] += 1
    return qsa_sparse_gqa_decode(
        queries,
        keys,
        values,
        top_idx,
        query_offset=query_offset,
        total_tokens=total_tokens,
        scale=scale,
        key_tile=key_tile,
        dimension_tile=dim_tile,
        key_splits=key_splits,
    )


__all__ = [
    "COMPRESS_RATIO",
    "SparseDecodeContractError",
    "armed",
    "assert_traced",
    "engagement_line",
    "installed",
    "MAX_CONTEXT",
    "REQUEST_SHAPE_DECLINES",
    "REQUEST_SHAPE_REASONS",
    "SHORT_CONTEXT_TOKENS",
    "context_decline",
    "note_request_decline",
    "note_route_decline",
    "note_route_hit",
    "route_snapshot",
    "pending",
    "receipt",
    "route_counters",
    "GQA",
    "HEAD_DIM",
    "KV_HEADS",
    "PARITY_FP32_MAX_ABS_ULPS",
    "PARITY_FP32_MAX_REL_L2",
    "PARITY_MIN_TOP1",
    "PARITY_SHIPPED_MAX_REL_L2",
    "PROBE_CAPACITY",
    "Q_HEADS",
    "SELECTED_TOKENS",
    "TOP_K",
    "VERIFY_ROWS",
    "attention",
    "check_cache_contract",
    "disabled_reason",
    "fp32_reference",
    "engagement",
    "install",
    "kernel_row_tokens",
    "reset_for_tests",
    "reference_attention",
    "shipped_fp32_probs_reference",
    "shipped_row_tokens",
    "stock_reference",
    "visible_block_count",
    "visible_sets_agree",
]
