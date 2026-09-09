"""``MTPLX_QWEN4_DRAFT_K20_PRESCATTER`` -- draft K20 support on the FR-Spec row.

Read ONCE at import, default OFF.  Flag-off, :func:`is_enabled` is False, no
plan is ever built, and the single call site in ``generation.py`` stays behind
a module-level constant, so the retained stock draft lane evaluates exactly the
expressions it evaluated before this module existed.

The problem
-----------
With ``MTPLX_FRSPEC_DRAFT=1`` the draft head is
``frspec_draft._FullVocabDraftHead``: it runs the pruned q8 head over the
65,536 frequency-ranked rows and then *scatters* those logits into a
248,320-wide row padded with ``-1e30``
(``mtplx/frspec_draft.py`` ``_FullVocabDraftHead.__call__``).  Which binding
makes that wrapper the LIVE draft head differs by lane -- see
:func:`_live_draft_route`; on the shipped Qwen4 native-MTP lane
``_mtplx_draft_lm_head`` keeps the unpruned head and says nothing about it.

The stock draft reader then builds its K20 support over that padded row
(``generation._fixed_width_draft_reader`` -> ``_sample_draft_from_logits`` ->
``_sample_from_logits`` -> ``_distribution_from_mlx_logits`` ->
``fast_sampling.sparse_distribution_from_mlx_logits`` ->
``fast_sampling._device_serial_support_arrays``), which costs, per draft step:

* the scatter itself: one ``mx.full`` + one ``put_along_axis`` over 248,320
  float32 lanes (~0.99 MB written, and the ``mx.full`` fill before it),
* one ``argpartition`` to an 80-candidate superset over 248,320 lanes,
* one full-vocabulary ``logsumexp`` over 248,320 lanes,

when 73.6% of those lanes are a constant sentinel that cannot win any of it.

What this module does
---------------------
It reads the draft from the 65,536-row head output and maps the selected LOCAL
rows back to real token ids through the ranked id table.  Nothing evaluates
the scattered array, and because MLX is lazy the scatter graph node is
therefore never executed -- ``put_along_axis`` is built and dropped, not run.

There are TWO reads, because ``generate_mtpk`` has two draft readers, and this
lane serves both:

* **sampled** (``draft_sampler.temperature > 0``) -- :func:`read_draft`
  builds the SAME K20 support the stock reader would and hands its caller the
  same ``(token, SparseDistribution)`` pair.  Costs one host sync per draft
  step, exactly as the stock reader does.
* **greedy** (``temperature <= 0``) -- :func:`greedy_chain_step` returns the
  SAME ``argmax`` token the stock greedy chain would, as an unevaluated device
  array, with the local->real map done by one 1-element ``mx.take``.  It adds
  no sync: the chain's single ``mx.eval`` per cycle still covers it.  Cycles
  that fall out of the greedy chain (a copy streak, a mid-generation steering
  arm, ``cycle_depth == 0``) land in :func:`read_draft`'s own
  ``temperature <= 0`` branch, which is the same argmax on the host.

Greedy is where this matters most: the stock greedy chain's per-depth work is
one ``mx.full`` + ``put_along_axis`` + ``argmax`` over 248,320 lanes, of which
73.6% are a sentinel that cannot win.  A greedy request is not an edge case --
HumanEval, and every ``temperature: 0`` API call, is one.

Exactness argument
------------------
Write ``ids`` for the ranked table (``_mtplx_frspec_ids``), ``sub`` for the
compact head row, ``dense`` for the scattered row, ``T`` for the draft
temperature and ``scaled = row.astype(float32) * (1/T)``.

1. **The table is strictly ascending.**  ``mtplx/data/qwen38_code_ranked_64k.json``
   is row-sorted (``frspec_draft`` module docstring: "row-sorted for efficient
   gathering"); :func:`claim_draft_route` re-proves it per request and refuses
   otherwise.  So ``local -> ids[local]`` is a strictly increasing bijection
   from ``[0, 65536)`` onto the occupied lanes of ``dense``, and it preserves
   BOTH the relative order of the values in the row and every ``id asc``
   tie-break.

2. **No sentinel can enter the support.**  The pad is ``-1e30`` in the head's
   own dtype; the row carries 65,536 real logits and the support is 20 wide,
   so the 80-candidate superset (``m = 4k = 80 <= 65536``) is entirely real on
   both rows, and the multiset of the top-80 VALUES is identical.  The stock
   builder's ``spill`` condition (``nanmin(cand) >= cutoff``) is a function of
   those 80 sorted values alone, so it fires on both rows or neither.

3. **Selection is identical.**  ``argpartition`` leaves the superset
   unordered, and the stock builder then imposes a total order with
   ``np.lexsort((cand_ids, -cand_val_rows))`` -- value desc, id asc.  This
   module maps local rows to real ids BEFORE that lexsort, so it sorts the
   same 80 (value, real id) pairs and takes the same top 20 in the same order.
   The spill fallback re-derives with ``_deterministic_mlx_top_k_support``,
   whose ``higher``/``tied``/``cumsum`` rank is likewise order-preserving under
   a strictly increasing id map, and its ids are mapped before ITS lexsort.
   The greedy branch is ``argmax``, whose lowest-index tie-break maps to the
   lowest real id for the same reason.

4. **The normaliser is value-identical.**  The only quantity the compact row
   cannot reproduce structurally is ``log_total = logsumexp(scaled)`` over the
   full row.  ``logsumexp`` is ``M + log(sum(exp(x - M)))`` with
   ``M = max(scaled)`` a real logit (tens, not -1e30).  A sentinel lane
   contributes ``exp(-1e30/T - M)``, and float32 ``exp`` underflows to
   *exactly* ``+0.0`` below about ``-103.97``; ``-1.67e30`` is 28 orders of
   magnitude past that, in float32 AND after a bfloat16 head output widens to
   float32 (bfloat16 holds ``-1e30`` as a finite ``-9.9964e29``; it is not an
   inf and it is not a NaN).  Since ``x + 0.0 == x`` exactly for every finite
   ``x``, every sentinel lane contributes exactly nothing to the sum, and the
   two rows' logsumexps are equal *as real numbers computed from the same
   float32 terms in the same order*.

   What this does NOT prove is bit-identity across the two reduction SHAPES.
   Floating-point addition is not associative, and a 248,320-lane row reduce
   partitions the 65,536 nonzero terms across threads/blocks differently than a
   65,536-lane one does; a residual of a few ULP in ``log_total`` is admissible
   in principle.  Measured on the CPU stream (which this repo's tests can run)
   the two are bit-identical -- ``tests/test_qwen4_draft_k20_prescatter.py``
   pins that -- and ``the PR #391 harness micro_draft_k20.py`` re-measures it on the
   Metal stream and prints the differing-row and ULP counters rather than
   assuming.  A residual ULP would scale every q entry by ``1 +- 2**-24`` and
   could only change behaviour by flipping a knife-edge top-p ``cumulative_before``
   comparison; it cannot change the SUPPORT (point 3 is exact and independent
   of ``log_total``).

5. **No consumer needs the dense draft row.**  Under this route the only
   readers of ``draft_logits`` in ``generation.generate_mtpk``'s draft loop are
   ``int(draft_logits.shape[-1])`` (a static shape, forces no evaluation) and
   ``_tree_nbytes(draft_logits)`` under ``trace.enabled`` (also static).

The greedy read's own exactness argument -- the argmax tie-break under the
strictly increasing id map, and the traced confidence -- is on
:func:`greedy_chain_step`.

Which request shape gets which path
-----------------------------------
======================================  ==============================
request                                 draft read
======================================  ==============================
temperature 0 (greedy chain running)    pre-scatter, on device, no sync
temperature 0 (chain skipped: ccopy,    pre-scatter, host argmax
mid-generation steer, depth-0 cycle)
temperature > 0 with ``top_k > 0``      pre-scatter K20 support
temperature > 0 with ``top_k <= 0``     stock reader (top-p-only builder;
                                        a different selector, not a
                                        narrower one)
a competing owner of the draft chain    that owner's reader
(PR391 D3, DEVICE_K20, target-prefix,
adapter ensemble, top-k reranker,
adaptive width, correction cache)
penalties / steering overlays           stock reader (they index by real
                                        token id)
``MTPLX_FRSPEC_LEGACY``                 stock reader (already remaps ids)
======================================  ==============================

The rows that do not say "pre-scatter" are ROUTING, decided once at
construction and never mid-decode: those readers are different selectors with
their own contracts, not this one refusing to work.  The lane stands aside
(:mod:`mtplx.qwen4_claim_contract`), the plan is ``None``, the shipped reader
runs, and the receipt records ``declined`` with the reason and a per-process
tally -- ``installed`` stays False, so no receipt ever claims this selector
produced a number it did not.

Separately, an INSTALL-time contract violation -- FR-Spec absent, the head off
the live draft route, a wrong-width or unordered ranked table -- raises
:class:`DraftK20PrescatterIneligible`.  No request in the process could be
served, so failing the first one loudly is the honest report.

``MTPLX_STRICT_CLAIMS=1`` turns the routing declines into raises for a
measured arm that must prove the lane ran.

NO device work happens at import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .qwen4_claim_contract import (
    ClaimDeclined,
    decline as _decline,
    declined_receipt,
)
from .sampling import SamplerConfig, SparseDistribution, sample_from_distribution


_ENV_VAR = "MTPLX_QWEN4_DRAFT_K20_PRESCATTER"

#: The only ranked-table width this route admits.  The built-in artifact
#: ``mtplx/data/qwen38_code_ranked_64k.json`` is exactly this wide; a different
#: width means a different (unproven) artifact, so the claim refuses instead of
#: generalising.
FRSPEC_ROWS = 65_536

#: ``_device_serial_support_arrays``'s candidate-superset multiplier.  Mirrored
#: here only to prove ``m <= FRSPEC_ROWS`` at claim time.
_SUPERSET_MULTIPLIER = 4


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


#: ``None`` = resolve from the environment on each read; a test may force a
#: bool via :func:`_configure_for_test`. Read at USE, never at import: the
#: server's fixed-M4 auto-arm stamps MTPLX_QWEN4_DRAFT_K20_PRESCATTER into the
#: environment AFTER this module is imported (via mtplx.server.openai's
#: generation import), so an import-time read froze the default (off) and the
#: served lane never engaged -- the arming audit, 2026-09-07. The env is frozen
#: once serving starts, so a per-call read returns the same value every time.
_ENABLED = None


def is_enabled() -> bool:
    """True when ``MTPLX_QWEN4_DRAFT_K20_PRESCATTER`` is set for this process.

    Read at use, not frozen at import (a test may force :data:`_ENABLED`).
    """

    if _ENABLED is not None:
        return bool(_ENABLED)
    return _env_truthy(_ENV_VAR)


def _configure_for_test(enabled: bool) -> None:
    """Flip the import-time gate (tests only)."""

    global _ENABLED
    _ENABLED = bool(enabled)


#: First-use engagement latch + the last install receipt, surfaced at /health
#: so the battery can gate on the install verdict instead of the env. The
#: receipt (``{installed, rows, ...}``) is per-request in generation.py; this
#: latches the last one a claim installed. Read-only reporting.
_ENGAGED = [False]
_LAST_RECEIPT: dict[str, object] = {}


def _note_engaged(receipt: dict[str, object] | None = None) -> None:
    """Latch first-use for /health (called by ``claim_draft_route`` on install)."""

    _ENGAGED[0] = True
    if receipt:
        _LAST_RECEIPT.clear()
        _LAST_RECEIPT.update(receipt)


def engagement_report() -> dict:
    """Install/first-use verdict for ``/health qwen4_install_reports.draft_k20_prescatter``.

    ``armed`` is read at use (reflects the served auto-arm stamp, gate-able
    without a request); ``engaged`` latches True the first time a claim installs
    the pre-scatter route, and the last install receipt rides along.
    """

    return {
        "armed": is_enabled(),
        "engaged": bool(_ENGAGED[0]),
        "receipt": dict(_LAST_RECEIPT),
    }


def reset_engagement_for_test() -> None:
    """Clear the first-use latch and receipt (tests only)."""

    _ENGAGED[0] = False
    _LAST_RECEIPT.clear()


class DraftK20PrescatterIneligible(RuntimeError):
    """The armed flag cannot work in THIS PROCESS at all.

    Reserved for install-time contract violations -- FR-Spec absent, the head
    off the live draft route, a ranked table of the wrong width or order.
    Every request would fail identically, so the first one fails loudly.

    A request whose SHAPE this lane does not serve (greedy, penalties, a
    competing owner of the draft chain, ...) does NOT raise: it declines to
    the shipped draft path, which is what :mod:`mtplx.qwen4_claim_contract`
    exists for.  Raising on those turned every greedy request into an HTTP
    500 in serving (composed-decode-stack HumanEval gate, 2026-09-02).
    """


# ---------------------------------------------------------------------------
# Construction-time plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftK20PrescatterPlan:
    """One request's bound pre-scatter draft route."""

    head: Any
    """The live ``frspec_draft._FullVocabDraftHead``."""

    ids_np: np.ndarray
    """The ranked table as int64, strictly ascending, shape ``[rows]``."""

    ids_mx: Any
    """The SAME table as the head's own device array (``head._ids``).

    The greedy route maps local rows to real ids on device with one
    ``mx.take``, so nothing about the greedy chain leaves the GPU until the
    single ``mx.eval`` the chain already does.
    """

    rows: int
    """Ranked-table width (the pruned head's output domain)."""

    vocab_rows: int
    """Full vocabulary width -- the domain every emitted distribution spans."""

    top_k: int
    temperature: float
    top_p: float

    route: str
    """Which binding makes ``head`` live -- see :func:`_live_draft_route`."""

    greedy: bool
    """True when the draft sampler is greedy (``temperature <= 0``).

    A greedy request reads the row with ``argmax`` and never builds a K20
    support, so ``top_k``/``top_p`` are not part of its contract -- see
    :func:`greedy_chain_step` and the ``temperature <= 0`` branch of
    :func:`read_draft`.
    """

    def to_dict(self) -> dict[str, object]:
        """The receipt this route writes."""

        return {
            "installed": True,
            "rows": int(self.rows),
            "vocab_rows": int(self.vocab_rows),
            "top_k": int(self.top_k),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "route": str(self.route),
            "read": "greedy_argmax" if self.greedy else "sampled_k20",
        }


def _refuse(reason: str) -> None:
    """Install-time contract violation: no request here could be served.

    Request-SHAPE ineligibility uses ``_decline`` instead -- see
    :mod:`mtplx.qwen4_claim_contract`.
    """

    raise DraftK20PrescatterIneligible(
        f"{_ENV_VAR}: {reason}"
    )


#: The draft-head hook the Qwen4 native MTP forward projects through.
#: ``models/qwen4_exp.TextModel.mtp_forward`` calls
#: ``self._mtp_draft_head_logits(...)``, and
#: ``TextModel._mtplx_bind_draft_lm_head`` rebinds that attribute to
#: ``head.__call__`` -- so on this route the live head is the ``__self__`` of a
#: bound method, not the value of an attribute.
_NATIVE_MTP_HOOK = "_mtp_draft_head_logits"

#: The draft head the generic ``mtp_patch`` forward projects through.
_CONFIGURED_HEAD = "_mtplx_draft_lm_head"


def _describe(obj: Any) -> str:
    """A short, honest name for whatever a liveness probe actually found."""

    if obj is None:
        return "absent"
    owner = getattr(obj, "__self__", None)
    if owner is not None:
        name = getattr(obj, "__name__", None) or "callable"
        return f"{type(owner).__name__}.{name}"
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if name is not None and not isinstance(obj, type):
        return str(name)
    return type(obj).__name__


def _live_draft_route(text: Any, head: Any) -> str | None:
    """Name the binding that makes ``head`` the live draft head, else ``None``.

    ``frspec_draft.install_frspec_draft_head`` publishes the full-vocabulary
    wrapper at ``text._mtplx_frspec_draft_head`` and then makes it live in
    exactly one of two ways:

    * ``"native_mtp_head"`` -- the Qwen4 native MTP route (the shipped
      ``--full-frspec`` lane, install report ``source: native_mtp_head``,
      ``legacy_swap: False``).  The install calls
      ``text._mtplx_bind_draft_lm_head(full_head)``, which rebinds
      ``text._mtp_draft_head_logits`` to ``full_head.__call__``; that hook is
      what ``TextModel.mtp_forward`` projects through, so the array this
      module reads (``draft_logits``) is exactly the wrapper's own return
      value.  ``text._mtplx_draft_lm_head`` is deliberately LEFT holding the
      unpruned configured head "for legacy consumers" -- it is a
      ``QuantizedLinear`` on this route and says nothing about liveness.
    * ``"legacy_swap"`` -- ``MTPLX_FRSPEC_LEGACY=1``.  The generic
      ``mtp_patch`` draft forward reads ``self._mtplx_draft_lm_head``
      directly, so the install swaps the wrapper in there globally.

    Both bindings hand the wrapper the same argument and return its output
    unchanged, so ``head.take_prescatter_row(draft_logits)`` identity-matches
    on either.  Anything else -- no FR-Spec install, an install whose bind
    hook never fired, or a third party that has since taken the hook over --
    is not a route this module has proven and returns ``None``.
    """

    native = getattr(text, _NATIVE_MTP_HOOK, None)
    if native is not None and getattr(native, "__self__", None) is head:
        return "native_mtp_head"
    if getattr(text, _CONFIGURED_HEAD, None) is head:
        return "legacy_swap"
    return None


def claim_draft_route(
    rt: Any,
    *,
    draft_sampler: SamplerConfig,
    draft_core: str,
    target_prefix_verify: bool,
    a3b_target_prefix_route: Any,
    pr391_route: Any,
    device_k20_route: Any,
    frspec_legacy_ids: Any,
    adaptive_width_policy: Any,
    combine_greedy_draft_read: bool,
    draft_confidence_needed: bool,
    draft_margin_threshold: float | None,
    wants_policy_metrics: bool,
    correction_cache_enabled: bool,
    adapter_ensemble_q: bool,
    mtp_topk_reranker: Any,
    relaxed_draft_ties: bool,
    penalties_active: bool,
    steer_active: bool,
    greedy_chain_enabled: bool = False,
    receipt: dict[str, object] | None = None,
) -> DraftK20PrescatterPlan | None:
    """Bind the pre-scatter draft route to one generation construction.

    Returns ``None`` when the flag is off, and ``None`` again when the flag is
    on but this REQUEST's shape is not one the route serves -- a decline, not
    a failure: the shipped draft path runs and produces the same tokens.  When
    ``receipt`` is passed it is filled in with why (``declined`` /
    ``declined_detail`` / cumulative ``declines``), so a reader can always tell
    a decline from a lane that never armed.

    Raises :class:`DraftK20PrescatterIneligible` only for an INSTALL-time
    contract violation -- something no request in this process could satisfy.
    ``MTPLX_STRICT_CLAIMS=1`` turns declines back into that exception for
    a measured arm that must prove the lane ran.
    """

    if not _ENABLED:
        return None
    try:
        plan = _claim_draft_route(
            rt,
            draft_sampler=draft_sampler,
            draft_core=draft_core,
            target_prefix_verify=target_prefix_verify,
            a3b_target_prefix_route=a3b_target_prefix_route,
            pr391_route=pr391_route,
            device_k20_route=device_k20_route,
            frspec_legacy_ids=frspec_legacy_ids,
            adaptive_width_policy=adaptive_width_policy,
            combine_greedy_draft_read=combine_greedy_draft_read,
            draft_confidence_needed=draft_confidence_needed,
            draft_margin_threshold=draft_margin_threshold,
            wants_policy_metrics=wants_policy_metrics,
            correction_cache_enabled=correction_cache_enabled,
            adapter_ensemble_q=adapter_ensemble_q,
            mtp_topk_reranker=mtp_topk_reranker,
            relaxed_draft_ties=relaxed_draft_ties,
            penalties_active=penalties_active,
            steer_active=steer_active,
            greedy_chain_enabled=greedy_chain_enabled,
        )
    except ClaimDeclined as declined:
        stamped = declined_receipt(
            _ENV_VAR, declined, ineligible=DraftK20PrescatterIneligible
        )
        if receipt is not None:
            receipt.clear()
            receipt.update(stamped)
        return None
    if plan is not None:
        try:
            _note_engaged(plan.to_dict())
        except Exception:
            _note_engaged()
    return plan


def _claim_draft_route(
    rt: Any,
    *,
    draft_sampler: SamplerConfig,
    draft_core: str,
    target_prefix_verify: bool,
    a3b_target_prefix_route: Any,
    pr391_route: Any,
    device_k20_route: Any,
    frspec_legacy_ids: Any,
    adaptive_width_policy: Any,
    combine_greedy_draft_read: bool,
    draft_confidence_needed: bool,
    draft_margin_threshold: float | None,
    wants_policy_metrics: bool,
    correction_cache_enabled: bool,
    adapter_ensemble_q: bool,
    mtp_topk_reranker: Any,
    relaxed_draft_ties: bool,
    penalties_active: bool,
    steer_active: bool,
    greedy_chain_enabled: bool,
) -> DraftK20PrescatterPlan:
    """The claim body.  ``_refuse`` raises; ``_decline`` stands aside."""

    text = getattr(getattr(rt, "model", None), "language_model", None)
    if text is None:
        text = getattr(rt, "model", None)
    head = getattr(text, "_mtplx_frspec_draft_head", None)
    if head is None:
        _refuse("no FR-Spec draft head is installed (need MTPLX_FRSPEC_DRAFT=1)")
    route = _live_draft_route(text, head)
    if route is None:
        _refuse(
            "the FR-Spec full-vocabulary head is installed but is not on the "
            "live draft route "
            f"({_NATIVE_MTP_HOOK}={_describe(getattr(text, _NATIVE_MTP_HOOK, None))}, "
            f"{_CONFIGURED_HEAD}={_describe(getattr(text, _CONFIGURED_HEAD, None))}, "
            f"head={type(head).__name__})"
        )
    ids = getattr(head, "_ids", None)
    if ids is None:
        _refuse("the FR-Spec head carries no ranked id table")
    if not hasattr(head, "arm_prescatter_capture"):
        _refuse(
            "the live draft head has no prescatter capture surface "
            f"({type(head).__name__})"
        )
    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    rows = int(ids_np.shape[0])
    if rows != FRSPEC_ROWS:
        _refuse(
            f"ranked table is {rows} rows; this route is proven only at "
            f"{FRSPEC_ROWS}"
        )
    if not bool(np.all(ids_np[1:] > ids_np[:-1])):
        _refuse(
            "ranked id table is not strictly ascending; the local->real id "
            "map must be monotone for the tie-break contract to hold"
        )
    vocab_rows = int(getattr(head, "_vocab_rows", 0))
    if vocab_rows <= rows or int(ids_np[-1]) >= vocab_rows:
        _refuse(f"full vocabulary width {vocab_rows} does not admit the table")

    # Greedy and sampled are two ROUTES through this lane, not eligible and
    # ineligible.  A greedy request reads the row with `argmax` (device-side
    # inside the greedy chain, host-side in the stock loop) and never builds a
    # K20 support, so the top-k/top-p terms below are not part of its
    # contract; requiring them would have refused every temperature-0 request
    # the server accepts.
    greedy = float(draft_sampler.temperature) <= 0.0
    top_k = int(draft_sampler.top_k)
    if not greedy:
        if top_k <= 0:
            _decline(
                "no_top_k",
                "the sampled route builds a top-k support; this request is "
                f"temperature {float(draft_sampler.temperature)!r} with "
                f"top_k={top_k}",
            )
        superset = min(max(_SUPERSET_MULTIPLIER * top_k, top_k), rows)
        if superset > rows:  # pragma: no cover - unreachable given the min()
            _decline(
                "superset_too_wide",
                f"candidate superset {superset} exceeds the ranked table",
            )
    if (
        float(draft_sampler.presence_penalty) != 0.0
        or float(draft_sampler.frequency_penalty) != 0.0
    ):
        _decline(
            "draft_sampler_penalties",
            "draft sampler penalties index by real token id",
        )
    if penalties_active or steer_active:
        _decline(
            "steer_or_penalties",
            "steering/penalty overlays index by real token id",
        )
    if greedy_chain_enabled and not greedy:  # pragma: no cover - unreachable
        # `generation._greedy_chain_eligible` already requires draft
        # temperature <= 0, so this cannot fire; it is here so a future change
        # to that predicate cannot silently hand the greedy chain a plan whose
        # read is the sampled one.
        _decline(
            "greedy_chain_without_greedy_sampler",
            "the greedy device chain was enabled for a sampled draft sampler",
        )
    if str(draft_core) != "stock":
        _decline(
            "non_stock_draft_core",
            f"this route requires the stock draft selector (got {draft_core!r})",
        )
    if relaxed_draft_ties:
        _decline(
            "relaxed_draft_ties",
            "MTPLX_QWEN4_RELAXED_DRAFT_TIES installs a different builder",
        )
    if frspec_legacy_ids is not None:
        _decline(
            "frspec_legacy",
            "MTPLX_FRSPEC_LEGACY already remaps local draft ids",
        )
    if device_k20_route is not None:
        _decline(
            "device_k20_owns_selector",
            "MTPLX_QWEN4_DEVICE_K20 owns the draft selector",
        )
    if pr391_route is not None:
        _decline(
            "pr391_owns_chain",
            "the PR391 float32 D3 route owns the draft chain",
        )
    if a3b_target_prefix_route is not None or target_prefix_verify:
        _decline(
            "target_prefix_verify",
            "target-prefix verification samples drafts on device",
        )
    if adaptive_width_policy is not None:
        _decline(
            "adaptive_width",
            "adaptive-width readers own the draft read",
        )
    if combine_greedy_draft_read:
        _decline(
            "combined_greedy_read",
            "the joint greedy confidence read materialises the dense row",
        )
    if draft_confidence_needed:
        _decline(
            "draft_confidence",
            "draft-confidence tracing materialises the dense row",
        )
    if draft_margin_threshold is not None or wants_policy_metrics:
        _decline(
            "draft_confidence_metrics",
            "draft confidence metrics materialise the dense row",
        )
    if correction_cache_enabled:
        _decline(
            "correction_cache",
            "the online/prompt correction cache bypasses the draft read",
        )
    if adapter_ensemble_q:
        _decline(
            "adapter_ensemble",
            "the adapter ensemble reads two dense draft rows",
        )
    if mtp_topk_reranker is not None:
        _decline(
            "topk_reranker",
            "the top-k reranker reads the dense draft row",
        )

    head.arm_prescatter_capture(True)
    return DraftK20PrescatterPlan(
        head=head,
        ids_np=ids_np,
        ids_mx=ids,
        rows=rows,
        vocab_rows=vocab_rows,
        top_k=top_k,
        temperature=float(draft_sampler.temperature),
        top_p=float(draft_sampler.top_p),
        route=route,
        greedy=greedy,
    )


def release_draft_route(plan: DraftK20PrescatterPlan | None) -> None:
    """Disarm the head capture and drop the stashed row."""

    if plan is None:
        return
    plan.head.arm_prescatter_capture(False)


# ---------------------------------------------------------------------------
# Per-step read
# ---------------------------------------------------------------------------


def take_compact_row(
    plan: DraftK20PrescatterPlan,
    draft_logits: Any,
) -> Any:
    """Return the 65,536-wide pre-scatter row behind ``draft_logits``.

    Identity-checked against the array the head returned for this very call,
    then consumed, so a stale or mismatched stash raises instead of silently
    scoring the wrong step.
    """

    stashed = plan.head.take_prescatter_row(draft_logits)
    if stashed is None:
        raise DraftK20PrescatterIneligible(
            "the FR-Spec head did not capture a pre-scatter row for this "
            "draft step (the live draft head changed mid-request)"
        )
    row = stashed.reshape(-1)
    if int(row.shape[0]) != plan.rows:
        raise DraftK20PrescatterIneligible(
            f"pre-scatter row is {int(row.shape[0])} wide, expected {plan.rows}"
        )
    return row


def greedy_chain_step(
    plan: DraftK20PrescatterPlan,
    draft_logits: Any,
    *,
    want_confidence: bool,
) -> tuple[Any, Any]:
    """One depth of ``generation``'s greedy draft chain, pre-scatter.

    Returns ``(token_id, confidence_or_None)`` as UNEVALUATED device arrays,
    exactly like the stock chain's ``mx.argmax`` / ``mx.exp(...)`` nodes, so
    the chain's single ``mx.eval`` at the end still costs one sync per cycle
    and the local->real id map is one 1-element ``mx.take`` on device.  The
    248,320-lane ``mx.full`` + ``put_along_axis`` behind ``draft_logits`` is
    built and dropped, never run.

    Exactness of the token
    ----------------------
    Write ``sub`` for the compact row, ``ids`` for the strictly ascending
    ranked table (re-proved at claim time), and ``dense`` for the scattered
    row: ``dense[ids[i]] == sub[i]`` and ``dense[j] == -1e30`` on every lane
    ``j`` no id occupies.

    ``mx.argmax`` returns the LOWEST index attaining the maximum.  Let
    ``M = max(sub)``.  Every occupied lane holds a real logit and every
    unoccupied one holds ``-1e30`` (``-9.9964e29`` after a bfloat16 head
    output widens), so as long as ``M > -1e30`` -- true for ANY finite head
    output; the alternative is a head that emitted no distribution at all, on
    which the stock lane is equally undefined -- ``max(dense) == M`` and the
    lanes attaining it are exactly ``{ids[i] : sub[i] == M}``.  Because
    ``ids`` is strictly increasing it is order-preserving, so

        min {ids[i] : sub[i] == M} == ids[ min {i : sub[i] == M} ]
                                   == ids[argmax(sub)]

    which is ``argmax(dense)``.  The tie-break is the same tie-break, not an
    equivalent one.

    The traced confidence, and why the claim routes it away
    -------------------------------------------------------
    ``want_confidence`` mirrors the stock chain's
    ``exp(max(row) - logsumexp(row))``.  ``max`` is bit-identical (it is the
    same real number, and ``max`` is exact).  ``logsumexp`` is
    ``M + log(sum(exp(x - M)))``, and each sentinel lane contributes
    ``exp(-1e30 - M)``, which underflows to *exactly* ``+0.0`` in float32
    below about ``-103.97`` -- 28 orders of magnitude of headroom -- and
    ``x + 0.0 == x`` exactly for finite ``x``.  So the two rows' logsumexps
    are equal as REAL NUMBERS computed from the same terms.

    They are NOT bit-identical.  Floating-point addition is not associative
    and the two reductions partition their partial sums differently: measured
    at the production shape the pair differs by about 2 ULP
    (``17.241907119750977`` vs ``17.24190902709961``), which is ~2e-6 relative
    on the confidence.  It cannot move a token -- the token is settled by the
    argmax above, and ``max`` agrees to the bit -- but it WOULD move a number
    two arms print, so :func:`claim_draft_route` routes any request that needs
    draft-confidence telemetry (``draft_confidence_needed``, i.e.
    ``MTPLX_DRAFT_CONFIDENCE_TRACE`` or a width threshold) to the stock reader
    rather than shave a step and change a receipt.  With tracing off -- the
    default, and what the server runs -- ``want_confidence`` is False and this
    branch is not taken.  It is kept so that relaxing that routing later
    cannot silently drop the number.
    """

    import mlx.core as mx

    row = take_compact_row(plan, draft_logits)
    local = mx.argmax(row, axis=-1)
    token = mx.take(plan.ids_mx, local)
    if not want_confidence:
        return token, None
    return token, mx.exp(mx.max(row) - mx.logsumexp(row))


def prescatter_serial_support_arrays(
    plan: DraftK20PrescatterPlan,
    compact_row: Any,
    config: SamplerConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``fast_sampling._device_serial_support_arrays`` on the compact row.

    Every arithmetic step is the shipped builder's, in the shipped order and
    the shipped dtypes; the only change is that ``argpartition`` /
    ``logsumexp`` run over ``plan.rows`` lanes instead of ``plan.vocab_rows``,
    and the selected LOCAL rows are mapped to real token ids through the ranked
    table before the deterministic ``lexsort``.  Returns
    ``(token_rows [N,k] int64, prob_rows [N,k] float64, vocab_size)`` with
    ``vocab_size`` the FULL vocabulary, because that is the domain the emitted
    ``SparseDistribution`` spans.
    """

    import mlx.core as mx

    from .fast_sampling import _fixed_top_k_support

    rows = compact_row.reshape(-1, compact_row.shape[-1]).astype(mx.float32)
    row_width = int(rows.shape[-1])
    if row_width != plan.rows:
        raise DraftK20PrescatterIneligible(
            f"pre-scatter row is {row_width} wide, expected {plan.rows}"
        )
    ids_np = plan.ids_np
    vocab_size = int(plan.vocab_rows)
    k = min(int(config.top_k), row_width)
    scaled = rows * (1.0 / float(config.temperature))

    m = min(max(_SUPERSET_MULTIPLIER * k, k), row_width)
    cand_idx = mx.argpartition(-scaled, kth=m - 1, axis=-1)[:, :m]
    cand_vals = mx.take_along_axis(scaled, cand_idx, axis=-1)
    top_p_active = 0.0 < float(config.top_p) < 1.0
    if top_p_active:
        log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
        cand_probs = mx.exp(cand_vals - log_total)
        mx.eval(cand_idx, cand_vals, cand_probs)
        cand_prob_rows = np.asarray(cand_probs, dtype=np.float64)
    else:
        mx.eval(cand_idx, cand_vals)
        cand_prob_rows = None
    # LOCAL rows -> real token ids.  Done here, before the lexsort, so the
    # deterministic (value desc, id asc) order is imposed on the same pairs the
    # full-vocabulary builder would have sorted.
    cand_ids = ids_np[np.asarray(cand_idx, dtype=np.int64)]
    cand_val_rows = np.asarray(cand_vals, dtype=np.float32)

    order = np.lexsort((cand_ids, -cand_val_rows), axis=1)
    cand_ids = np.take_along_axis(cand_ids, order, axis=1)
    cand_val_rows = np.take_along_axis(cand_val_rows, order, axis=1)
    if cand_prob_rows is not None:
        cand_prob_rows = np.take_along_axis(cand_prob_rows, order, axis=1)
    token_rows = cand_ids[:, :k]
    if m > k:
        cutoff = cand_val_rows[:, k - 1]
        spill = np.nanmin(cand_val_rows, axis=1) >= cutoff
    else:
        spill = np.zeros(cand_ids.shape[0], dtype=bool)

    if top_p_active:
        prob_rows = cand_prob_rows[:, :k].copy()
        cumulative_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        prob_rows = np.where(
            cumulative_before < float(config.top_p), prob_rows, 0.0
        )
    else:
        vals64 = cand_val_rows[:, :k].astype(np.float64)
        vals64 -= np.max(vals64, axis=1, keepdims=True)
        prob_rows = np.exp(vals64)
        prob_rows /= np.sum(prob_rows, axis=1, keepdims=True)

    if spill.any():
        _, exact_idx, exact_vals = _fixed_top_k_support(scaled, top_k=k)
        if top_p_active:
            exact_probs = mx.exp(
                exact_vals - mx.logsumexp(scaled, axis=-1, keepdims=True)
            )
        else:
            exact_probs = mx.softmax(exact_vals, axis=-1)
        mx.eval(exact_idx, exact_probs)
        exact_ids = ids_np[np.asarray(exact_idx, dtype=np.int64)]
        exact_prob_rows = np.asarray(exact_probs, dtype=np.float64)
        if top_p_active:
            ex_order = np.lexsort((exact_ids, -exact_prob_rows), axis=1)
            exact_ids = np.take_along_axis(exact_ids, ex_order, axis=1)
            exact_prob_rows = np.take_along_axis(exact_prob_rows, ex_order, axis=1)
            ex_before = np.concatenate(
                (
                    np.zeros((exact_prob_rows.shape[0], 1), dtype=np.float64),
                    np.cumsum(exact_prob_rows[:, :-1], axis=1),
                ),
                axis=1,
            )
            exact_prob_rows = np.where(
                ex_before < float(config.top_p), exact_prob_rows, 0.0
            )
        token_rows = np.where(spill[:, None], exact_ids, token_rows)
        prob_rows = np.where(spill[:, None], exact_prob_rows, prob_rows)

    return token_rows, prob_rows, vocab_size


def prescatter_sparse_distribution(
    plan: DraftK20PrescatterPlan,
    compact_row: Any,
    config: SamplerConfig,
) -> SparseDistribution:
    """``sparse_distribution_from_mlx_logits`` on the compact row."""

    import mlx.core as mx

    from .fast_sampling import _host_sparse_distribution, _serial_row_distribution

    row = compact_row.reshape(-1).astype(mx.float32)
    token_rows, prob_rows, vocab_size = prescatter_serial_support_arrays(
        plan, row, config
    )
    dist = _serial_row_distribution(token_rows[0], prob_rows[0], vocab_size)
    if dist is not None:
        return dist
    # Non-finite mass (NaN/inf logits): the shipped one-hot/dense host
    # reference, run on the compact row and mapped back.  ``apply_top_p_top_k``
    # ranks by probability then id, so the strictly ascending table preserves
    # its answer exactly.
    mx.eval(row)
    local = _host_sparse_distribution(np.asarray(row, dtype=np.float32), config)
    return SparseDistribution(
        plan.ids_np[np.asarray(local.token_ids, dtype=np.int64)],
        local.probs,
        vocab_size,
    )


def read_draft(
    plan: DraftK20PrescatterPlan,
    draft_logits: Any,
    config: SamplerConfig,
    rng: Any,
    *,
    need_distribution: bool,
) -> tuple[int, SparseDistribution | None]:
    """The stock draft read (``_sample_draft_from_logits``), pre-scatter.

    Draws exactly the same one ``rng`` value per draft step, in the same order,
    from the same stream as the stock lane: ``sample_from_distribution`` ->
    ``rng.choice(ids, p=probs)``.
    """

    import mlx.core as mx

    row = take_compact_row(plan, draft_logits)
    if float(config.temperature) <= 0:
        local = int(mx.argmax(row, axis=-1).item())
        token = int(plan.ids_np[local])
        if not need_distribution:
            return token, None
        return token, SparseDistribution.one_hot(token, int(plan.vocab_rows))
    # ``need_distribution`` is deliberately ignored here: the shipped
    # ``_sample_draft_from_logits`` also ignores it on the sampled branch (it
    # tail-calls ``_sample_from_logits``, which always returns the row it
    # sampled from), and the accept loop is fed that same row.
    del need_distribution
    distribution = prescatter_sparse_distribution(plan, row, config)
    return sample_from_distribution(distribution, rng), distribution
