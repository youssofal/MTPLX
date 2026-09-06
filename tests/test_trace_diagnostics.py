import json
from argparse import Namespace

import pytest

from mtplx.commands.trace import _detect_pathologies, _load_receipts, _match_receipt, _source_port
from mtplx.commands.trace_clients import load_pi_session
from mtplx.commands.trace_metrics import mtp_economics, sample_intervals


def test_tps_chart_keeps_missing_samples_and_time_gaps_out_of_the_curve():
    from mtplx.commands.trace_report import _tps_cell

    row = {"sliding": [], "samples": [{"ts": 1}], "decode_tok_s": 25,
           "status": "ok", "turn": 1, "completion_tokens": 100}
    assert _tps_cell(row) is None
    row["samples"] = [{"ts": 1, "tps": 50}, {"ts": 2}, {"ts": 5, "tps": 0}]
    html = _tps_cell(row)
    assert "flight samples n=2" in html
    assert "min 0.0" in html  # A measured zero remains meaningful.
    curve = html.split('<path d="', 1)[1].split('"', 1)[0]
    assert curve.count("M") == 2  # An observation gap is not a connecting line.


def test_economics_distinguishes_aggregate_acceptance_from_conditional_probability():
    receipt = {"completion_tokens": 250, "decode_elapsed_s": 4.0, "verify_calls": 100,
               "accepted_by_depth": [80, 50, 20], "drafted_by_depth": [100, 100, 100],
               "verify_time_s": 3, "draft_time_s": .8}
    result = mtp_economics(receipt, ar_tok_s=50)
    assert result["acceptance"] == .5
    assert result["tokens_per_verify"] == 2.5
    assert result["speedup_vs_ar"] == 1.25
    assert result["break_even_acceptance"] == pytest.approx(1/3)
    assert result["cycle_cost_ar_steps"] == pytest.approx(2.0)
    assert result["drafts_per_cycle"] == 3
    assert result["fixed_depth"] is True
    assert result["acceptance_margin"] == pytest.approx(.5 - 1/3)
    assert result["mtp_pays"] is True
    assert mtp_economics(receipt)["speedup_vs_ar"] is None
    assert mtp_economics(receipt)["break_even_acceptance"] is None
    # Copy output is accounted for by observed non-draft output.
    assert mtp_economics({**receipt, "context_copy_accepted_tokens": 12}, 50)["break_even_acceptance"] == pytest.approx(1 / 3)
    # Adaptive depth holds the observed mix and non-draft output constant;
    # its threshold is conditional, while the throughput comparison is observed.
    adaptive = mtp_economics({**receipt, "drafted_by_depth": [100, 80, 20]}, 50)
    assert adaptive["fixed_depth"] is False
    assert adaptive["drafts_per_cycle"] == 2
    assert adaptive["break_even_acceptance"] == pytest.approx(.5)
    assert adaptive["acceptance"] == pytest.approx(150 / 200)
    assert adaptive["mtp_pays"] is True
    # Real receipts carry a few more verify calls than drafted rows (terminal
    # and bonus boundaries); the threshold must not vanish for them.
    real = mtp_economics({**receipt, "verify_calls": 103}, 50)
    assert real["fixed_depth"] is True
    assert real["break_even_acceptance"] is not None
    # A cycle costlier than perfect acceptance can amortize: threshold > 1.
    slow = mtp_economics({**receipt, "decode_elapsed_s": 9.0}, 50)
    assert slow["break_even_acceptance"] > 1
    assert slow["mtp_pays"] is False
    failed = mtp_economics({**receipt, "repetition_stop_triggered": True}, 50)
    assert failed["status"] == "guard_stopped_invalid_quality_sample"
    assert failed["speedup_vs_ar"] is None


def test_intervals_preserve_zero_and_missing_counters_and_expose_gaps():
    rows = sample_intervals([
        {"ev": "s", "ts": 1, "gen": 10, "vt": 0, "acc": [0], "drf": [0]},
        {"ev": "s", "ts": 2, "gen": 20, "vt": .4, "acc": [3], "drf": [5]},
        {"ev": "s", "ts": 9, "gen": 2, "vt": 0},
    ])
    assert rows[0]["vt"] == .4
    assert rows[0]["acceptance"] == .6
    assert rows[0]["dt"] is None
    assert rows[1]["observation_gap"] is True
    assert rows[1]["gen"] is None
    assert rows[1]["tok_s"] is None


def test_custom_logs_include_rotations_without_silently_using_latest_daemon(tmp_path):
    path = tmp_path / "receipts.jsonl"
    path.write_text('{"request_id":"new"}\n')
    (tmp_path / "receipts.jsonl.1").write_text('{"request_id":"old"}\n')
    assert [r["request_id"] for r in _load_receipts(0, path=str(path))] == ["old", "new"]
    assert _source_port(Namespace(port=None, request_log=str(path))) == 0


def test_pi_reader_follows_active_branch_and_keeps_correlation(tmp_path):
    rows = [{"type": "session", "id": "session", "cwd": str(tmp_path)},
            {"type": "message", "id": "u", "parentId": None,
             "message": {"role": "user", "timestamp": 1000, "content": "Build it"}},
            {"type": "message", "id": "abandoned", "parentId": "u",
             "message": {"role": "assistant", "timestamp": 2000, "content": []}},
            {"type": "message", "id": "current", "parentId": "u",
             "message": {"role": "assistant", "timestamp": 3000,
                         "content": [{"type": "thinking", "thinking": "inspect"}],
                         "usage": {"output": 20, "cacheRead": 12}}}]
    path = tmp_path / "pi.jsonl"
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    session, messages = load_pi_session(path)
    assert session["id"] == "session"
    assert [m["_id"] for m in messages] == ["u", "current"]
    assert messages[-1]["_parent_entry_id"] == "u"
    assert messages[-1]["_parts"] == [{"type": "reasoning", "text": "inspect"}]


def test_exact_pi_parent_beats_nearby_time_match_and_cannot_be_reused():
    message = {"_parent_entry_id": "right", "time": {"created": 1000}}
    receipts = [{"request_client_entry_id": "wrong", "logged_at_s": 1},
                {"request_client_entry_id": "right", "logged_at_s": 5}]
    used = set()
    assert _match_receipt(message, receipts, used) is receipts[1]
    assert used == {1}


def test_new_tool_content_and_unknown_reasoning_are_not_false_regressions():
    turns = [
        {"kind": "assistant", "turn": 1, "message": {"tokens": {}},
         "receipt": {"prompt_tokens": 17513, "completion_tokens": 227}},
        {"kind": "assistant", "turn": 2, "message": {"tokens": {"reasoning": 25000}},
         "receipt": {"prompt_tokens": 25819, "cached_tokens": 17740,
                     "new_prefill_tokens": 8079, "completion_tokens": 26000}},
    ]
    assert _detect_pathologies(turns) == []
    turns[1]["receipt"]["cached_tokens"] = 1024
    assert any("REDUCED PREFIX REUSE" in flag for flag in _detect_pathologies(turns))


def test_economics_counts_real_proposals_and_never_contradicts_throughput():
    receipt = {"completion_tokens": 2400, "decode_elapsed_s": 44.598313789931126,
               "verify_calls": 807, "drafted_by_depth": [801, 801, 801],
               "accepted_by_depth": [654, 526, 409],
               "context_copy_rounds": 6, "context_copy_accepted_tokens": 4}
    result = mtp_economics(receipt, 33.81733719505653)
    assert result["proposal_cycles"] == 801
    assert result["fixed_depth"] is True
    assert result["cycle_ms"] == pytest.approx(55.67829436945209)
    assert result["break_even_acceptance"] == pytest.approx(
        (44.598313789931126 * 33.81733719505653 - 811) / 2403)
    assert result["break_even_basis"] == "fixed_depth_with_observed_copy_mix"
    assert result["speedup_vs_ar"] == pytest.approx(1.591304881230316)
    assert result["mtp_pays"] is True
    extra = mtp_economics({"completion_tokens": 150, "decode_elapsed_s": 3.5,
                          "verify_calls": 200, "drafted_by_depth": [100],
                          "accepted_by_depth": [50]}, 50)
    assert extra["speedup_vs_ar"] < 1
    assert extra["mtp_pays"] is False
    assert extra["break_even_acceptance"] == .75
