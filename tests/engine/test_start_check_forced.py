"""The engine's start check can be forced to fail (``MTPLX_NATIVE_ENGINE_TEST_FAIL_START``), so a
release gate drives the fallback on an installed wheel without touching the installed files."""

from __future__ import annotations

import mlx.core as mx

from mtplx.engine import runtime


class _GoodModule:
    BUILT_AGAINST_MLX = mx.__version__

    @staticmethod
    def self_check():
        return ""

    @staticmethod
    def abi_probe(array):
        return None


def test_a_good_module_passes_the_start_check(monkeypatch):
    monkeypatch.delenv(runtime.TEST_FAIL_START_ENV, raising=False)
    assert runtime._check_module(_GoodModule) == ""


def test_the_forced_failure_reports_as_a_failed_self_check(monkeypatch):
    monkeypatch.setenv(runtime.TEST_FAIL_START_ENV, "1")
    problem = runtime._check_module(_GoodModule)
    assert "self check failed" in problem and runtime.TEST_FAIL_START_ENV in problem
    monkeypatch.setenv(runtime.TEST_FAIL_START_ENV, "0")
    assert runtime._check_module(_GoodModule) == ""
