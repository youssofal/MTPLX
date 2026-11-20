"""The Flash-Next install receipts reach /health (2026-09-03 ports, PR #391)."""

from __future__ import annotations

from types import SimpleNamespace

from mtplx.models import qwen4_exp
from mtplx.server.openai import _qwen4_install_reports


class _Sidecar:
    prewarm_at_load = {"mode": "auto", "budget_bytes": 21_368_709_120, "seconds": 1.6}


class _Ple:
    _sidecar = _Sidecar()

    def named_modules(self):
        return [("layers.1.ple", self)]


class _Text:
    def named_modules(self):
        return [("layers.1.ple", _Ple())]


def test_reports_are_collected_from_the_runtime() -> None:
    runtime = SimpleNamespace(
        qwen4_m4_stage3_report={"installed": True, "route_kernel": {"installed": True, "layers": 48}},
        _mtplx_qwen4_verify_glue={"qsa_rope": {"installed": True}, "qsa_rope_idx": {"installed": True}},
        model=_Text(),
    )
    reports = _qwen4_install_reports(SimpleNamespace(runtime=runtime))
    assert reports["m4_stage3"]["route_kernel"]["layers"] == 48
    assert reports["verify_glue"]["qsa_rope"]["installed"] is True
    assert reports["ngram_prewarm"]["mode"] == "auto"


def test_absent_lanes_read_as_absent_and_never_raise(monkeypatch) -> None:
    # The verify head-chunk receipt is process-wide (a server process serves
    # one model). A Flash-Next test that ran earlier on this worker leaves it
    # set, so this read starts from a process where the lane never fired.
    monkeypatch.setattr(qwen4_exp, "_VERIFY_SDPA_HEAD_CHUNK_ENGAGED", None)
    assert _qwen4_install_reports(SimpleNamespace()) == {}
    runtime = SimpleNamespace(model=object())
    assert _qwen4_install_reports(SimpleNamespace(runtime=runtime)) == {}


def test_a_multimodal_wrapper_resolves_the_text_tower() -> None:
    runtime = SimpleNamespace(model=SimpleNamespace(language_model=_Text()))
    reports = _qwen4_install_reports(SimpleNamespace(runtime=runtime))
    assert reports["ngram_prewarm"]["seconds"] == 1.6
