"""Accounting for the completed passes of a streamed request."""

from __future__ import annotations

from typing import Any, Callable

_STREAM_RECOVERY_STAT_PREFIXES = (
    "inspection_empty_retry_",
    "tool_fed_empty_retry_",
    "reasoning_completion_repair_",
    "read_only_force_answer_retry_",
)
# Key on a recovery pass's result naming the prompt it was generated from,
# when that prompt is not the request's own (the tool-fed empty retry).
_ATTEMPT_PROMPT_IDS_KEY = "_mtplx_attempt_prompt_ids"


def _attempt_receipt(generated: dict[str, Any]) -> dict[str, Any]:
    # Generation results own KV snapshots. Keep only the accounting fields
    # so discarded passes release their caches before a later repair runs.
    return {
        "completion_tokens": generated.get("completion_tokens"),
        "stats": dict(generated.get("stats") or {}),
    }


def _stream_attempt_totals(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over every generation pass of one streamed request.

    Each pass's stats describe only itself. Preserve those fields and add
    totals for completed passes so earlier prefills remain visible.
    """

    per_pass = [dict(attempt.get("stats") or {}) for attempt in attempts]
    totals: dict[str, Any] = {
        "stream_attempts": len(attempts),
        "stream_attempts_prompt_eval_time_s": sum(
            float(stats.get("prompt_eval_time_s") or 0.0) for stats in per_pass
        ),
        "stream_attempts_new_prefill_tokens": sum(
            int(stats.get("new_prefill_tokens") or 0) for stats in per_pass
        ),
        "stream_attempts_completion_tokens": sum(
            int(attempt.get("completion_tokens") or 0) for attempt in attempts
        ),
    }
    first_ttft_s = per_pass[0].get("ttft_s")
    if first_ttft_s is not None:
        totals["stream_attempts_first_ttft_s"] = float(first_ttft_s)
    return totals


def _note_metrics_rows_changed(state: Any) -> None:
    """A dashboard row (state.last_metrics) was added or updated in place:
    metrics streams that send the rows only when they change send them again."""
    state.metrics_revision = int(getattr(state, "metrics_revision", 0) or 0) + 1


DASHBOARD_ROWS_KEPT = 100


def _append_metrics_row(state: Any, row: dict[str, Any]) -> None:
    """A finished request's row joins the dashboard ring: appended, the ring
    trimmed to its last 100 rows, and the change noted for the metrics
    streams that send the rows only when they change. The one way a row is
    added, so no path can add one without the note."""
    state.last_metrics.append(row)
    state.last_metrics = state.last_metrics[-DASHBOARD_ROWS_KEPT:]
    _note_metrics_rows_changed(state)


def _metric_for_request(state: Any, request_id: Any) -> dict[str, Any] | None:
    """Return the last pass of this request, never an unrelated completion.

    Every caller updates the row it gets in place, so finding one counts as
    a change of the dashboard rows."""
    if request_id is None:
        return None
    row = next(
        (row for row in reversed(getattr(state, "last_metrics", ()))
         if row.get("request_id") == request_id),
        None,
    )
    if row is not None:
        _note_metrics_rows_changed(state)
    return row


def _update_recovery_metrics(state: Any, stats: dict[str, Any]) -> None:
    """Attach recovery fields to their request, even after another completion."""
    metric = _metric_for_request(state, stats.get("request_id"))
    if metric is not None:
        metric.update(
            (key, value) for key, value in stats.items()
            if key.startswith((*_STREAM_RECOVERY_STAT_PREFIXES, "stream_attempts"))
        )


def _run_stream_recovery_chain(
    state: Any,
    generated: dict[str, Any],
    steps: list[Callable[[dict[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    """Run the stream worker's recovery passes and account for all of them.

    A step returns its input unchanged or the result of a new pass. When
    more than one pass ran, the final stats gain the ``stream_attempts*``
    totals and keep the recovery fields of earlier passes (a repair after a
    retry used to drop the ``tool_fed_empty_retry_*`` fields). Existing
    fields keep the last pass's values; single-pass envelopes are untouched.
    """

    attempts = [_attempt_receipt(generated)]
    for step in steps:
        result = step(generated)
        if result is not generated:
            attempts.append(_attempt_receipt(result))
            # The worker still holds its initial result until this chain
            # returns. Once replaced, that pass cannot be postcommitted.
            generated.pop("_final_state", None)
        generated = result
    if len(attempts) == 1:
        return generated
    stats = generated.setdefault("stats", {})
    for earlier in attempts[:-1]:
        for key, value in (earlier.get("stats") or {}).items():
            if key.startswith(_STREAM_RECOVERY_STAT_PREFIXES):
                stats.setdefault(key, value)
    totals = _stream_attempt_totals(attempts)
    stats.update(totals)
    _update_recovery_metrics(state, stats)
    return generated
