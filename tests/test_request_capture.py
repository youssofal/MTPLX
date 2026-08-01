"""Tests for the #196 request-capture ring."""

from __future__ import annotations

import json
import os

from mtplx import request_capture


def _enable(monkeypatch, tmp_path, keep="200"):
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_KEEP", keep)
    request_capture._PATHS_BY_ID.clear()


def test_capture_then_outcome_merge(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request(
        "chatcmpl-abc", {"prompt_token_ids": [1, 2, 3], "max_tokens": 64}
    )
    files = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
    assert len(files) == 1
    rec = json.load(open(tmp_path / files[0]))
    assert rec["phase"] == "dispatched"
    assert rec["prompt_token_ids"] == [1, 2, 3]

    request_capture.capture_outcome(
        "chatcmpl-abc", {"finish_reason": "stop", "completion_tokens": 5}
    )
    rec = json.load(open(tmp_path / files[0]))
    assert rec["phase"] == "completed"
    assert rec["outcome"]["finish_reason"] == "stop"


def test_dispatch_record_survives_missing_outcome(monkeypatch, tmp_path):
    """The whole point: a hung/crashed turn still leaves its envelope."""
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("chatcmpl-hang", {"prompt_token_ids": [7]})
    files = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
    rec = json.load(open(tmp_path / files[0]))
    assert rec["phase"] == "dispatched"
    assert "outcome" not in rec


def test_ring_prunes_to_keep_without_deleting(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, keep="3")
    for i in range(6):
        request_capture.capture_request(f"r{i}", {"i": i})
    live = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
    assert len(live) == 3
    pruned = os.listdir(tmp_path / "pruned")
    assert len(pruned) == 3  # moved, never deleted


def test_disabled_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPLX_REQUEST_CAPTURE_DIR", raising=False)
    request_capture.capture_request("x", {"a": 1})
    request_capture.capture_outcome("x", {"b": 2})
    assert list(tmp_path.iterdir()) == []


def test_clip_text_head_tail():
    small = request_capture.clip_text_head_tail("hello", head=10, tail=10)
    assert small == {"text": "hello", "text_clipped": False}
    big = request_capture.clip_text_head_tail("a" * 100, head=10, tail=10)
    assert big["text_clipped"] and big["text_chars"] == 100
    assert len(big["text_head"]) == 10 and len(big["text_tail"]) == 10


def test_unsafe_request_ids_are_sanitized(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("../../etc/passwd", {"a": 1})
    files = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
    assert len(files) == 1
    assert ".." not in files[0] and "/" not in files[0]
