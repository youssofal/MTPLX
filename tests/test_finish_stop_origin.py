"""#414 telemetry: finish_reason="stop" must name its commit path.

The report: Flash-Next depth 3 ended a sampled 512-token request at 37
tokens with finish_reason=stop while depth 2 ran to the cap. Adjudicating
that class of report requires knowing WHICH speculative branch emitted
the stop — accepted draft, residual correction, bonus sample, pending
primary, or a context-copy block — without asking the reporter to re-run
under MTPLX_TRACE. These tests pin the helper, the stats field, and the
batched-loop wiring.
"""

from __future__ import annotations

import inspect

import mtplx.generation as gen


STOPS = {151645}
EOS = 151645


class TestStopOriginForCommitted:
    def test_primary_stop(self):
        assert (
            gen._stop_origin_for_committed(
                [EOS, 11, 12], STOPS, has_correction=False
            )
            == "primary"
        )

    def test_accepted_draft_stop(self):
        assert (
            gen._stop_origin_for_committed(
                [10, EOS, 12], STOPS, has_correction=False
            )
            == "accepted_draft"
        )

    def test_correction_tail_stop(self):
        assert (
            gen._stop_origin_for_committed(
                [10, 11, EOS], STOPS, has_correction=True
            )
            == "residual_correction"
        )

    def test_tail_without_correction_is_accepted_draft(self):
        assert (
            gen._stop_origin_for_committed(
                [10, 11, EOS], STOPS, has_correction=False
            )
            == "accepted_draft"
        )

    def test_first_stop_wins(self):
        # An accepted-draft stop before a correction stop attributes to the
        # draft: the response ends at the FIRST stop.
        assert (
            gen._stop_origin_for_committed(
                [10, EOS, EOS], STOPS, has_correction=True
            )
            == "accepted_draft"
        )

    def test_no_stop_is_none(self):
        assert (
            gen._stop_origin_for_committed(
                [10, 11, 12], STOPS, has_correction=True
            )
            is None
        )


class TestStatsField:
    def test_default_none_and_serialized(self):
        stats = gen.GenerationStats(
            mode="mtp",
            generated_tokens=1,
            elapsed_s=0.1,
            tok_s=10.0,
        )
        assert stats.finish_stop_origin is None
        assert "finish_stop_origin" in stats.to_dict()

    def test_value_round_trips(self):
        stats = gen.GenerationStats(
            mode="mtp",
            generated_tokens=1,
            elapsed_s=0.1,
            tok_s=10.0,
            finish_stop_origin="bonus",
        )
        assert stats.to_dict()["finish_stop_origin"] == "bonus"


class TestBatchedLoopWiring:
    def test_every_commit_path_stamps_an_origin(self):
        src = inspect.getsource(gen)
        # Full-accept drafted stop, bonus stop, both context-copy lanes,
        # both partial-accept committed scans, the cycle-start primary
        # backstop, and the final derivation.
        assert src.count('stop_origin = "context_copy"') == 2
        assert src.count('stop_origin = "residual_correction"') == 2
        assert src.count('stop_origin = "accepted_draft"') == 1
        assert src.count('stop_origin = "bonus"') == 1
        assert src.count('stop_origin = "primary"') == 1
        assert src.count('stop_origin = "repetition_stop"') == 1
        assert src.count("stop_origin = _stop_origin_for_committed(") == 2
        assert "finish_stop_origin=stop_origin," in src

    def test_length_finish_clears_origin(self):
        src = inspect.getsource(gen)
        cut = src.index('if finish_reason != "stop":')
        assert "stop_origin = None" in src[cut : cut + 120]


def test_stats_origin_reaches_public_surfaces():
    """#414 followup: the origin reached GenerationStats but neither the SSE
    mtplx_stats payload nor the request-log JSONL row, so the release note's
    "diagnosable from request logs alone" promise was unreadable without a
    local patch."""
    from mtplx.server.openai import _metrics_envelope, _public_mtplx_stats

    stats = {"finish_stop_origin": "residual_correction"}
    # SSE mtplx_stats: quiet-envelope rule — present when a stop named its
    # commit path, absent on length finishes (None in to_dict()).
    assert (
        _public_mtplx_stats({"stats": stats})["finish_stop_origin"]
        == "residual_correction"
    )
    # Request-log JSONL row: nullable like repetition_stop_reason.
    envelope = _metrics_envelope(
        stats=stats,
        prompt_tokens=512,
        completion_tokens=37,
        request_elapsed_s=0.5,
        token_times=[10.0],
        request_started_s=9.5,
        lock_wait_time_s=0.0,
        session_id=None,
        session_cache_hit=False,
        cache_miss_reason=None,
        session_restore_mode="cold",
        mtp_depth=3,
        generation_limits={},
    )
    assert envelope["finish_stop_origin"] == "residual_correction"
