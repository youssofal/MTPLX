"""Inkling must report committed work through the ordinary generation API."""

from unittest.mock import patch

import mlx.core as mx
import pytest

from mtplx import route_tape
from mtplx.generation import generate_mtpk
from mtplx.models.inkling import generation
from test_inkling_customer_controls import CausalRows, GREEDY


@pytest.fixture
def capture(monkeypatch):
    events = []
    monkeypatch.setenv("MTPLX_ROUTE_TAPE", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setattr(route_tape, "_ROUTE_TAPE_SINK", events.append)
    return events


def run(rt, **kw):
    with mx.stream(mx.cpu), patch.object(generation, "restore_or_prefill_prompt_state", rt.prefill):
        return generate_mtpk(rt, [0], sampler=GREEDY, speculative_depth=3, seed=29,
                             max_tokens=12, session_id="private-session", trace_label="private-prompt", **kw)


def test_accepted_stop_leaves_untested_depths_and_standalone_id_opaque(capture):
    rt = CausalRows(lambda token: token + 1, 18)
    out = run(rt, stop_token_ids={2})
    header, event = capture
    assert header["name"] == "header" and header["round"] is None
    assert event["rid"] == header["rid"] and event["rid"].startswith("rt-")
    assert "private-" not in repr(capture)
    attrs = event["attrs"]
    assert route_tape.is_verify_route(attrs["verify_route"])
    assert attrs["proposal_source"] == "dspark" and attrs["commit_route"] == "capture_commit"
    assert [d["accepted"] for d in attrs["drafts"]] == [True, None, None]
    assert (attrs["verify_width"], attrs["committed_rows"], attrs["kv_tail_rows_trimmed"]) == (4, 2, 2)
    assert attrs["rejected_at_depth"] is None and not attrs["bonus_token"]
    assert out.stats.verify_calls == 1 and out.stats.rejected_drafts == 0


def test_emit_only_after_commit_even_if_stream_delivery_fails(capture):
    rt = CausalRows(lambda token: token + 1, 18)
    with patch.object(rt, "commit", side_effect=ValueError("commit failed")), pytest.raises(ValueError, match="commit failed"):
        run(rt, stop_token_ids=set())
    assert [e["name"] for e in capture] == ["header"]
    capture.clear()

    def callback(batch):
        if len(batch) > 1:
            raise ValueError("stream closed")

    with pytest.raises(ValueError, match="stream closed"):
        run(rt, stop_token_ids=set(), token_callback=callback)
    assert [e["name"] for e in capture] == ["header", "round"]
    assert capture[-1]["attrs"]["commit_route"] == "verify_retained"
    assert capture[-1]["attrs"]["bonus_token"]


def test_cancellation_after_verify_records_only_the_anchor(capture):
    rt = CausalRows(lambda token: token + 1, 18)
    with patch.object(rt, "verify", wraps=rt.verify) as verify:
        out = run(rt, stop_token_ids=set(), abort_check=lambda: verify.called)
    attrs = capture[-1]["attrs"]
    assert out.tokens == [1] and out.stats.rejected_drafts == 0
    assert attrs["committed_rows"] == 1 and attrs["target_draws"] == 0
    assert attrs["rejected_at_depth"] is None and all(d["accepted"] is None for d in attrs["drafts"])


def test_off_uses_original_eval_and_commit_without_a_tape_clock_or_id(monkeypatch, capture):
    monkeypatch.setenv("MTPLX_ROUTE_TAPE", "0")
    monkeypatch.setattr(route_tape, "mono_ns", lambda: pytest.fail("disabled clock"))
    with patch.object(route_tape.os, "urandom", side_effect=AssertionError("disabled request ID")):
        out = run(CausalRows(lambda token: token + 1, 18), stop_token_ids={2})
    assert out.tokens == [1, 2] and out.stats.verify_calls == 1
    assert capture == []
