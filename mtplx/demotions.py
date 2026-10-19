"""Nothing goes slow in silence: the process-wide demotion ledger.

Every place the engine leaves the lane it was built to run (a compiled
verifier that runs eager, a sparse prefill that rebuilds the dense mask, a
tensor-unit route that bails on an older GPU) records a count and one plain
line saying why. ``/health`` (``degradation.demotions``), the request log
(``demotions`` on each generation record) and ``mtplx doctor --explain``
read this ledger, so a slow request can be explained from the product,
without a debug switch and without reading source.

Cost contract: ``note`` is a dictionary increment and one reference store.
No device read, no sync, no string formatting (callers on a per-round path
pass a constant reason), no lock (the model owner thread is the only
writer on the decode path, and an int increment under the GIL cannot
corrupt the dictionary). Importing this module imports nothing else from
``mtplx`` and never imports MLX.
"""

from __future__ import annotations

import sys
from typing import Any

# kind -> what it means, in plain English. The keys are a public surface
# (health payload, request log, doctor): add, never rename.
KINDS: dict[str, str] = {
    "qsa_prefill_lane_off": (
        "A Flash-Next prefill chunk past the sparse-attention floor ran the "
        "dense path because the sparse prefill lane is off on this Mac."
    ),
    "qsa_prefill_dense_mask": (
        "A Flash-Next sparse prefill chunk selected its blocks but no sparse "
        "consumer took them, so the dense mask was rebuilt."
    ),
    "qsa_prefill_direct_retired": (
        "The Steel sparse prefill kernel (the M1 to M4 lane) was retired for "
        "this process: it was built against a different MLX or failed its "
        "readiness proof, so Flash-Next prefill runs the pure-MLX path."
    ),
    "qwen4_wide_prefill_chunk_refused": (
        "A Flash-Next prompt prefilled in 2,048-row chunks because the wider "
        "chunk did not fit under the memory line for that request."
    ),
    "fixed_m4_lane_skipped": (
        "A Flash-Next request ran without the compiled verifier (memory gate, "
        "operator ceiling, or a draft depth below 3)."
    ),
    "fixed_m4_uncompiled_round": (
        "A Flash-Next verify round ran eager because its width has no compiled "
        "route (only verify width 4, draft depth 3, is compiled)."
    ),
    "fixed_m4_dispatch_retired": (
        "The Flash-Next compiled verifier failed to dispatch on this GPU and "
        "was retired to the eager verifier for the life of the process."
    ),
    "copy_round_eager": (
        "A copy-block round ran on the eager forward (the Flash-Next batched "
        "lane compiles copy blocks only with MTPLX_FIXED_M4_COPY_WINDOWS=1, "
        "and then only full-length blocks on a bucketed rows-gather bank)."
    ),
    "fixed_m4_copy_windows_retired": (
        "Flash-Next compiled copy windows failed to dispatch on this GPU and "
        "were retired to the eager forward for the life of the process."
    ),
    "compiled_verify_growth_demotion": (
        "A 27B verify round ran eager because the request outgrew the compiled "
        "verifier's growth reserve."
    ),
    "compiled_verify_context_fence": (
        "A 27B verify round ran eager because the context is above the "
        "compiled verifier's context fence."
    ),
    "compiled_verify_other_fallback": (
        "A compiled verify call fell back to eager for another reason (see the "
        "reason line)."
    ),
    "tensor_unit_route_bail": (
        "A tensor-unit attention route bailed because this GPU has no tensor "
        "units (needs GPU generation 17 and macOS 26.2)."
    ),
    "inforward_boundary_capture_missed": (
        "A prefill forward was asked to record a restore boundary inside it "
        "and did not bank one, so a later turn of that session restores from "
        "an earlier boundary and re-prefills more."
    ),
    "gdn_blocked_prefill_not_engaged": (
        "The blocked GDN prefill kernel was requested but a prefill-sized call "
        "ran the stock path."
    ),
    "gdn_fast_prefill_not_engaged": (
        "The MLX fast GDN prefill route was requested but did not install "
        "(MLX before 0.32.4) or a prefill-sized call took the previous path."
    ),
    "vision_request_eager_verify": (
        "An image request ran on the eager verifier. Image requests take the "
        "compiled verifier on Flash-Next and on the dense Qwen3.5 / Qwen3.8 "
        "packs alike; this one was kept off it by its prompt shape, a "
        "diagnostic setting or the kill switch (see the reason line)."
    ),
    "vision_request_eager_draft": (
        "An image request ran the draft head on the stock route: the request "
        "was kept off the compiled verify route, or its prompt ends on an "
        "image row, whose position a compiled draft core cannot express."
    ),
    "vision_mrope_sequential_fallback": (
        "An image request on a dense Qwen pack was roped with sequential "
        "positions because its image position table could not be built or the "
        "model's attention could not carry it."
    ),
    "vision_draft_head_sequential_positions": (
        "An image request's draft head kept sequential positions (its history "
        "cache is windowed, reset or on an explicit position mode); only the "
        "draft acceptance rate is affected, the verifier stays exact."
    ),
    "vision_mrope_tensor_offset_call": (
        "An attention route reached a tensor-offset cache that owns no rotary "
        "origin during an image request and kept the stock positions (a "
        "compiled route the admission never handed the image delta)."
    ),
}

_COUNTS: dict[str, int] = dict.fromkeys(KINDS, 0)
_REASONS: dict[str, str] = {}

# Bail counters that already exist in kernel modules. They are read at
# snapshot time (never imported here, never touched on the hot path):
# (module, attribute, key inside that counter dict).
_EXTERNAL_TENSOR_UNIT_BAILS: tuple[tuple[str, str, str], ...] = (
    ("mtplx.kernels.sdpa_nax_flash", "nax_flash_bail_counts", "gpu_family_or_os"),
    (
        "mtplx.kernels.sdpa_nax_flash_dsplit",
        "nax_flash_dsplit_bail_counts",
        "gpu_family_or_os",
    ),
    ("mtplx.kernels.sdpa_nax_tile", "nax_tile_bail_counts", "gpu_family_or_os"),
)
_TENSOR_UNIT_BAIL_REASON = (
    "no tensor units on this GPU (or macOS below 26.2): the packed attention "
    "kernel serves these verify calls instead of the flash route"
)

# Compiled-verify fallback labels (graphbank) -> ledger kind.
_BANK_REASON_KINDS: dict[str, str] = {
    "growth_budget_exhausted": "compiled_verify_growth_demotion",
    "block_window_capacity": "compiled_verify_growth_demotion",
    "context_above_threshold": "compiled_verify_context_fence",
}


def note(kind: str, reason: str | None = None, count: int = 1) -> None:
    """Record ``count`` demotions of ``kind``. Never raises."""

    try:
        _COUNTS[kind] += count
    except KeyError:
        _COUNTS[kind] = count
    if reason is not None:
        _REASONS[kind] = reason


def note_bank_fallback(reason: str) -> None:
    """Record one compiled-verify fallback by its graphbank reason label."""

    kind = _BANK_REASON_KINDS.get(reason)
    if kind is None:
        note("compiled_verify_other_fallback", reason)
    else:
        note(kind, reason)


def _external_counts() -> dict[str, int]:
    total = 0
    for module_name, attr, key in _EXTERNAL_TENSOR_UNIT_BAILS:
        module = sys.modules.get(module_name)
        counter = getattr(module, attr, None) if module is not None else None
        if isinstance(counter, dict):
            try:
                total += int(counter.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
    return {"tensor_unit_route_bail": total} if total else {}


def counts() -> dict[str, int]:
    """Every kind with its count so far (zeros included)."""

    merged = dict(_COUNTS)
    for kind, value in _external_counts().items():
        merged[kind] = merged.get(kind, 0) + value
    return merged


def mark() -> dict[str, int]:
    """A point to measure a request against (see ``since``)."""

    return counts()


def since(before: dict[str, int] | None) -> dict[str, int]:
    """Demotions recorded after ``before``: only the kinds that moved.

    Attribution is exact on the serial scheduler (one request at a time);
    under a batching scheduler concurrent requests share the ledger, so the
    delta is an upper bound for each of them.
    """

    base = before or {}
    now = counts()
    return {
        kind: value - int(base.get(kind, 0) or 0)
        for kind, value in now.items()
        if value - int(base.get(kind, 0) or 0) > 0
    }


def snapshot() -> dict[str, Any]:
    """The health payload: totals, the last reason per kind, and meanings."""

    now = counts()
    reasons = dict(_REASONS)
    if now.get("tensor_unit_route_bail") and "tensor_unit_route_bail" not in reasons:
        reasons["tensor_unit_route_bail"] = _TENSOR_UNIT_BAIL_REASON
    active = {kind: value for kind, value in now.items() if value > 0}
    return {
        "total": int(sum(active.values())),
        "counts": now,
        "reasons": {kind: reasons[kind] for kind in active if kind in reasons},
        "meanings": {kind: KINDS.get(kind, "") for kind in active},
    }


def explain_lines(payload: dict[str, Any] | None = None) -> list[str]:
    """Plain lines for ``mtplx doctor --explain`` from a snapshot payload."""

    data = payload if isinstance(payload, dict) else snapshot()
    all_counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    active = {k: int(v) for k, v in all_counts.items() if int(v or 0) > 0}
    if not active:
        return ["No demotions recorded: every request stayed on its fast lane."]
    reasons = data.get("reasons") if isinstance(data.get("reasons"), dict) else {}
    lines = []
    for kind in sorted(active, key=lambda item: (-active[item], item)):
        meaning = KINDS.get(kind) or str((data.get("meanings") or {}).get(kind) or "")
        lines.append(f"{active[kind]:>8,}  {kind}: {meaning}".rstrip())
        reason = reasons.get(kind)
        if reason:
            lines.append(f"          last reason: {reason}")
    return lines


def reset() -> None:
    """Tests only."""

    for kind in list(_COUNTS):
        _COUNTS[kind] = 0
    _REASONS.clear()
