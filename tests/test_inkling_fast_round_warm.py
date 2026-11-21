"""The load's round warm-up survives a call the program declines (review 2, finding 8).

Since df3782b5 a declined call returns None. The warm-up's timing loop must take that as
"no timing for this width" and carry on; before, it read ``served.packet`` and the model did
not load."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mtplx.models.inkling.fast import round_program as rp


class _Native:
    native = True
    packet = mx.zeros((8,), mx.int32)
    rows = mx.zeros((2, 4), mx.float32)
    features = []
    pending = SimpleNamespace(deltas=[])


def _fakes(monkeypatch, answers):
    calls = []

    def fake_serve(rt, anchor, cache, k, **kwargs):
        calls.append(k)
        return answers(len(calls))

    monkeypatch.setattr(rp, "serve", fake_serve)
    monkeypatch.setattr(rp.engine_runtime, "mode", lambda: "auto")
    monkeypatch.setattr(rp.engine_runtime, "available", lambda: True)
    monkeypatch.setattr(rp.engine_site, "_synthetic_cache", lambda rt, length, seed: [])
    monkeypatch.setattr(rp, "_synthetic_draft_cache", lambda rt, length, seed: [])
    monkeypatch.setattr(rp, "DecodeState", lambda model, cache: SimpleNamespace(prepare=lambda: None))
    monkeypatch.setattr(rp, "site", lambda rt: SimpleNamespace(report=lambda: {}))
    monkeypatch.setattr(rp.fast_round, "read_packet", lambda packet, width: None)
    rt = SimpleNamespace(drafter=object(), trunk_entries=0,
                         model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=mx.zeros((16, 4)))))
    return rt, calls


def test_a_declined_timing_call_leaves_the_width_untimed_and_the_load_going(monkeypatch):
    rt, calls = _fakes(monkeypatch, lambda n: _Native() if n == 1 else None)
    report = rp.warm(rt, (2, 3))
    assert report["counts"][2]["live"] and report["counts"][2]["round_ms"] is None
    assert 2 in report["timing_declined"]
    assert 3 in report["counts"]  # the next width was still warmed


def test_timing_uses_only_native_samples(monkeypatch):
    class _Mlx(_Native):
        native = False

    rt, _ = _fakes(monkeypatch, lambda n: _Native() if n in (1, 3) else _Mlx())
    report = rp.warm(rt, (2,))
    assert report["counts"][2]["round_ms"] is not None  # one native sample of three
