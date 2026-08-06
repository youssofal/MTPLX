"""Tests for sustained prefill chunk defaults.

OpenCode issue #63 froze the chunk policy at 2048 pending a matched workload
benchmark. That benchmark landed 2026-08-05 (temp-gated, M5 Max): dense 4096
gives +41-58% prefill at an 8k prompt and +33-38% at 32k, and beats 8192 at
both rungs. The repage path (>128k contexts) keeps 2048: 4096 there costs
+14.3 GB peak memory (+38%) with no TTFT win
(tools/bench/prefill-chunk-split-128k-2026-05-09.md).

The diagnostic dense/repage env knobs remain supported, and the legacy
single-knob `MTPLX_PREFILL_CHUNK_SIZE` env still overrides BOTH paths.
"""

from __future__ import annotations

import pytest

from mtplx.generation import (
    _prefill_chunk_size,
    _sustained_prefill_layout,
    prefill_chunk_size_override,
)
from mtplx.profiles import SUSTAINED_PREFILL_ENV


# ---------------------------------------------------------------------------
# Profile-level invariants


def test_sustained_profile_keeps_dense_decode_through_128k() -> None:
    assert (
        SUSTAINED_PREFILL_ENV["MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT"]
        == "131072"
    )


def test_sustained_profile_ships_split_chunk_defaults() -> None:
    assert SUSTAINED_PREFILL_ENV["MTPLX_PREFILL_CHUNK_SIZE"] == "auto"
    assert SUSTAINED_PREFILL_ENV["MTPLX_PREFILL_CHUNK_SIZE_DENSE"] == "4096"
    assert SUSTAINED_PREFILL_ENV["MTPLX_PREFILL_CHUNK_SIZE_REPAGE"] == "2048"


# ---------------------------------------------------------------------------
# Product path selection


def _apply_product_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror the sustained-profile env relevant to chunk-size selection."""

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "4096")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")


@pytest.mark.parametrize("context_tokens", [32_768, 65_536, 131_072])
def test_prefill_chunk_dense_uses_4096_through_128k(
    monkeypatch: pytest.MonkeyPatch, context_tokens: int
) -> None:
    _apply_product_env(monkeypatch)
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(context_tokens))

    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 4096


@pytest.mark.parametrize("context_tokens", [150_000, 200_000])
def test_prefill_chunk_repage_keeps_2048_above_128k(
    monkeypatch: pytest.MonkeyPatch, context_tokens: int
) -> None:
    _apply_product_env(monkeypatch)
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(context_tokens))

    assert _sustained_prefill_layout() == "contiguous_then_repage"
    assert _prefill_chunk_size() == 2048


# ---------------------------------------------------------------------------
# Env-var honoring


def test_prefill_chunk_envs_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The diagnostic per-layout envs remain respected on each path."""

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "1024")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "512")

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "32768")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 1024

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "150000")
    assert _sustained_prefill_layout() == "contiguous_then_repage"
    assert _prefill_chunk_size() == 512


def test_prefill_chunk_legacy_env_back_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the legacy single-knob env to a numeric value must override
    BOTH the dense and repage paths so existing deployments keep working."""

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    # Legacy single-knob set to a non-default value; split envs are deliberately
    # left at fresh defaults so we can confirm the legacy knob actually wins.
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "1536")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "32768")
    assert _prefill_chunk_size() == 1536

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "150000")
    assert _prefill_chunk_size() == 1536


def test_prefill_chunk_context_override_is_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")

    assert _prefill_chunk_size() == 2048
    with prefill_chunk_size_override(8192):
        assert _prefill_chunk_size() == 8192
    assert _prefill_chunk_size() == 2048
