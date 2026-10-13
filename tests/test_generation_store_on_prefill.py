"""Env-gate behavior for the store-on-prefill session-cache fix."""

import mtplx.generation as generation
from mtplx.runtime_options import store_on_prefill_min_suffix


def test_store_on_prefill_defaults_on(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_STORE_ON_PREFILL", raising=False)
    assert generation._store_on_prefill_env_enabled() is True


def test_store_on_prefill_kill_switch(monkeypatch):
    for off in ("0", "false", "off", "no"):
        monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL", off)
        assert generation._store_on_prefill_env_enabled() is False
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL", "1")
    assert generation._store_on_prefill_env_enabled() is True


def test_store_on_prefill_min_suffix_parse(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", raising=False)
    assert store_on_prefill_min_suffix() == 1024
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "4096")
    assert store_on_prefill_min_suffix() == 4096
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "garbage")
    assert store_on_prefill_min_suffix() == 1024
