"""Pure arithmetic for recorded decode economics; never estimates an AR baseline."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def mtp_economics(receipt: dict, ar_tok_s: float | None = None) -> dict:
    """Compare delivered throughput with a supplied, matched AR measurement.

    A proposal cycle is counted by the first draft position, not verifier
    forwards (which also include copy/bonus work). With G delivered tokens,
    A accepted drafts, P proposals and full decode time T, the observed
    speedup is G / (T * ar_tok_s). The conditional threshold is
    (T * ar_tok_s - (G - A)) / P: hold full cost, non-draft output and the
    depth mix fixed. It does not predict how cost changes with acceptance.
    """
    tokens = _number(receipt.get("completion_tokens"))
    elapsed = _number(receipt.get("decode_elapsed_s"))
    rounds = _number(receipt.get("verify_calls"))
    acc = receipt.get("accepted_by_depth") or []
    drafted = receipt.get("drafted_by_depth") or []
    accepted = sum(float(x) for x in acc)
    proposed = sum(float(x) for x in drafted)
    verify = _number(receipt.get("verify_time_s"))
    draft = _number(receipt.get("draft_time_s"))
    acceptance = accepted / proposed if proposed else None
    cycles = _number(drafted[0]) if drafted else None
    non_draft = tokens - accepted if tokens is not None else None
    drafts_per_cycle = proposed / cycles if proposed and cycles else None
    result: dict[str, Any] = {
        "acceptance_definition": "accepted draft tokens / all proposed draft tokens",
        "acceptance": acceptance,
        "drafts_per_cycle": drafts_per_cycle,
        "fixed_depth": bool(
            cycles and drafted and all(float(n) == cycles for n in drafted)
        ),
        "tokens_per_verify": tokens / rounds if tokens and rounds else None,
        "decode_ms_per_token": 1000 * elapsed / tokens if elapsed and tokens else None,
        "verify_ms_per_round": (
            1000 * verify / rounds if verify is not None and rounds else None
        ),
        "draft_ms_per_round": (
            1000 * draft / rounds if draft is not None and rounds else None
        ),
        "proposal_cycles": cycles,
        "tokens_per_cycle": tokens / cycles if tokens and cycles else None,
        "non_draft_tokens_per_cycle": (
            non_draft / cycles if non_draft is not None and non_draft >= 0 and cycles else None
        ),
        "cycle_ms": 1000 * elapsed / cycles if elapsed and cycles else None,
        "ar_tok_s": ar_tok_s,
        "cycle_cost_ar_steps": None,
        "speedup_vs_ar": None,
        "break_even_acceptance": None,
        "break_even_basis": None,
        "break_even_assumption": (
            "Holds full cost, non-draft output and the proposal depth mix constant; "
            "an estimate, not a measurement at lower acceptance."
        ),
        "acceptance_margin": None,
        "mtp_pays": None,
        "status": "matched_ar_measurement_required",
    }
    if receipt.get("repetition_stop_triggered"):
        result.update(status="guard_stopped_invalid_quality_sample",
                      guard_reason=receipt.get("repetition_stop_reason"))
        return result
    ar = _number(ar_tok_s)
    if ar and tokens and elapsed:
        result["speedup_vs_ar"] = (tokens / elapsed) / ar
        result["status"] = "measured_comparison_requires_matched_workload"
        if cycles:
            result["cycle_cost_ar_steps"] = elapsed / cycles * ar
            result["mtp_pays"] = bool(result["speedup_vs_ar"] > 1)
        if proposed and non_draft is not None and non_draft >= 0 and accepted <= proposed:
            # Keep >1 and <0 values meaningful; never mistake the number of
            # extra verifier calls for extra primary output tokens.
            break_even = (elapsed * ar - non_draft) / proposed
            result["break_even_acceptance"] = break_even
            depth_basis = "fixed_depth" if result["fixed_depth"] else "observed_depth_mix"
            result["break_even_basis"] = depth_basis + (
                "_with_observed_copy_mix"
                if receipt.get("context_copy_rounds") or receipt.get("context_copy_accepted_tokens")
                else "_with_observed_non_draft_output"
            )
            result["acceptance_margin"] = acceptance - break_even
    return result


def sample_intervals(events: list[dict]) -> list[dict]:
    """Difference cumulative counters; leave missing observations missing.

    A gap is an observation gap, not an invented sequence of zero-TPS samples.
    A negative delta is a counter reset and is never rendered as performance.
    """
    samples = sorted((e for e in events if e.get("ev") == "s"), key=lambda e: e.get("ts", 0))
    result = []
    for prev, cur in pairwise(samples):
        duration = float(cur["ts"]) - float(prev["ts"])
        if duration <= 0:
            continue
        row = {"start_s": prev["ts"], "end_s": cur["ts"], "duration_s": duration,
               "observation_gap": duration > 2.5, "context_tokens": cur.get("ctx")}
        for key in ("gen", "vc", "vt", "dt", "at", "ct", "rt", "st", "bt", "cct", "cv", "evc"):
            before, after = _number(prev.get(key)), _number(cur.get(key))
            row[key] = after - before if before is not None and after is not None and after >= before else None
        for key in ("acc", "drf"):
            before, after = prev.get(key), cur.get(key)
            row[key] = ([b-a for a, b in zip(before, after)]
                        if isinstance(before, list) and isinstance(after, list)
                        and len(before) == len(after) and all(b >= a for a, b in zip(before, after)) else None)
        row["tok_s"] = row["gen"] / duration if row["gen"] is not None else None
        denominator = sum(row["drf"]) if row["drf"] is not None else 0
        row["acceptance"] = sum(row["acc"]) / denominator if row["acc"] is not None and denominator else None
        row["active_memory_bytes"] = cur.get("mem_active")
        row["cache_memory_bytes"] = cur.get("mem_cache")
        row["peak_memory_bytes"] = cur.get("mem_peak")
        row["verify_route"] = cur.get("route")
        result.append(row)
    return result
