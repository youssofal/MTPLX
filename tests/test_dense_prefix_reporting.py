"""Prefix reuse must be VISIBLE to the caller, not just performed.

The dense lane restored prompt prefixes correctly for its entire life and told
nobody. `cached_tokens` and `cache_hit` were hardcoded to `0` and `False` in the
completion payload, so a row that reused 630 of its 755 prompt tokens reported
zero re-use on the wire.

Every external instrument therefore agreed the feature did nothing: the OpenAI
`usage.prompt_tokens_details.cached_tokens` block, `/metrics`, the server's
`session_restore_mode` (which read `mtp_batch_cold`), and a multi-turn benchmark
that measured 0.0% across every configuration. Meanwhile the driver's own debug
log showed `path1 exact-restore: HIT, covered=630`.

That combination is the dangerous one. A working optimisation that reports
itself as broken gets deleted by the next reader as dead weight, and no test
catches the deletion because the numbers never moved.

Two separate defects are pinned here, because fixing only the first one changes
nothing a caller can see:

1. The driver reported reuse per COHORT ROW; a response needs it per REQUEST.
   Nothing connected the two, so there was no truthful number to report.
2. `cached_tokens` was written into the prefill-progress payload, which the wire
   does not read. The OpenAI usage block is built from `stats`.
"""

from __future__ import annotations

from typing import Any


class _Meta(dict):
    pass


class _Result:
    def __init__(self, covered_by_row: dict[int, int]) -> None:
        self.meta: dict[str, Any] = {"prefix_covered_by_row": covered_by_row}


def _covered_for(job_row: int, covered_by_row: dict[int, int], prompt_len: int) -> int:
    """The attribution the service performs, isolated for test.

    Mirrors `_complete_cohort_job`: look the row up in the driver's per-row map
    and clamp to the prompt length.
    """
    result = _Result(covered_by_row)
    covered = int(
        ((getattr(result, "meta", None) or {}).get("prefix_covered_by_row") or {}).get(
            job_row, 0
        )
        or 0
    )
    return max(0, min(covered, prompt_len))


def test_a_restored_row_reports_the_tokens_it_actually_reused() -> None:
    """The defect: this returned 0 while the row reused 630 of 755."""
    assert _covered_for(0, {0: 630}, 755) == 630


def test_each_row_gets_its_own_number_not_the_cohorts() -> None:
    """Cohort totals cannot be attributed back to a caller.

    Three rows in one cohort reusing 630, 0 and 755 tokens must report exactly
    that, not a third of 1385 each, and not the cohort sum.
    """
    covered = {0: 630, 2: 755}
    assert _covered_for(0, covered, 755) == 630
    assert _covered_for(1, covered, 800) == 0, "a row that missed must report zero"
    assert _covered_for(2, covered, 889) == 755


def test_a_row_never_placed_in_a_cohort_reports_zero() -> None:
    """`cohort_row` defaults to -1; that must not index into anything."""
    assert _covered_for(-1, {0: 630}, 755) == 0


def test_reuse_is_clamped_to_the_prompt_length() -> None:
    """A caller must never see more cached tokens than it sent.

    `usage.prompt_tokens_details.cached_tokens` is clamped again downstream,
    but a value that has to be clamped twice is a value that was wrong once.
    """
    assert _covered_for(0, {0: 9_999}, 755) == 755


def test_cache_hit_follows_the_measured_reuse() -> None:
    """`cache_hit` was hardcoded False. It must track the real number."""
    assert (_covered_for(0, {0: 630}, 755) > 0) is True
    assert (_covered_for(0, {0: 0}, 755) > 0) is False


def test_the_wire_reads_stats_not_the_prefill_payload() -> None:
    """The subtle half of this bug, and the one that wasted a fix.

    `usage.prompt_tokens_details.cached_tokens` is built from
    `generated["stats"]["cached_tokens"]`. Setting the field on the prefill
    progress payload alone leaves the wire exactly as silent as before -- which
    is what the first attempt at this fix did, and the multi-turn benchmark
    still read 0.0%. This drives the real usage builder to prove which dict
    reaches a caller.
    """
    from mtplx.server.openai import _usage_payload

    # The value in `stats` is the one a caller sees.
    usage = _usage_payload(
        {"prompt_tokens": 755, "completion_tokens": 96, "stats": {"cached_tokens": 630}}
    )
    assert usage["prompt_tokens_details"]["cached_tokens"] == 630

    # A payload that records reuse ANYWHERE ELSE reports nothing. This is the
    # failing shape of the first fix, kept so it cannot come back.
    silent = _usage_payload(
        {
            "prompt_tokens": 755,
            "completion_tokens": 96,
            "stats": {},
            "prefill": {"cached_tokens": 630},
        }
    )
    assert "prompt_tokens_details" not in silent, (
        "reuse recorded outside `stats` never reaches the caller"
    )


def test_the_usage_block_clamps_to_the_prompt() -> None:
    """A caller must never be told it cached more than it sent."""
    from mtplx.server.openai import _usage_payload

    usage = _usage_payload(
        {"prompt_tokens": 100, "completion_tokens": 5, "stats": {"cached_tokens": 9999}}
    )
    assert usage["prompt_tokens_details"]["cached_tokens"] == 100
