"""One reader per session-prefix setting (mtplx.runtime_options)."""

from __future__ import annotations

import pytest

from mtplx import runtime_options as ro

READERS = [
    (
        "MTPLX_SESSION_NEAR_PREFIX_MAX_TOKEN_GAP",
        ro.near_prefix_max_token_gap,
        ro.DEFAULT_NEAR_PREFIX_MAX_TOKEN_GAP,
        8,
        0,
    ),
    (
        "MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS",
        ro.near_prefix_min_match_tokens,
        ro.DEFAULT_NEAR_PREFIX_MIN_MATCH_TOKENS,
        64,
        1,
    ),
    (
        "MTPLX_SESSION_PREFIX_BLOCK_SIZE",
        ro.prefix_block_size,
        ro.DEFAULT_PREFIX_BLOCK_SIZE,
        256,
        1,
    ),
    (
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS",
        ro.block_prefix_min_match_tokens,
        ro.DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS,
        512,
        1,
    ),
    (
        "MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX",
        ro.store_on_prefill_min_suffix,
        ro.DEFAULT_STORE_ON_PREFILL_MIN_SUFFIX,
        1024,
        1,
    ),
]


@pytest.mark.parametrize(("name", "reader", "default", "literal", "floor"), READERS)
def test_reader_default_and_parsing(monkeypatch, name, reader, default, literal, floor):
    # The documented defaults are unchanged.
    assert default == literal
    monkeypatch.delenv(name, raising=False)
    assert reader() == default
    for junk in ("", "   ", "garbage", "1.5"):
        monkeypatch.setenv(name, junk)
        assert reader() == default
    monkeypatch.setenv(name, " 4096 ")
    assert reader() == 4096
    monkeypatch.setenv(name, "-7")
    assert reader() == floor
    monkeypatch.setenv(name, "0")
    assert reader() == floor


def test_cold_tier_and_launch_lane_share_the_constants():
    from mtplx import launch_lane
    from mtplx.cache_bank import cold_tier

    assert ro.DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS == 512
    assert cold_tier.DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS == 512
    assert cold_tier.DEFAULT_BLOCK_SIZE == ro.DEFAULT_PREFIX_BLOCK_SIZE == 256
    assert launch_lane.SSD_SESSION_CACHE_MIN_PREFIX_TOKENS == 512


def test_session_bank_knobs_follow_the_readers(monkeypatch):
    from mtplx import session_bank

    for name, *_ in READERS:
        monkeypatch.delenv(name, raising=False)
    assert session_bank._near_prefix_knobs() == {
        "gap_limit": 8,
        "min_match": 64,
        "block": 256,
        "block_min_match": 512,
    }
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "128")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "64")
    knobs = session_bank._near_prefix_knobs()
    assert knobs["block"] == 128
    # The block-prefix floor never drops below one block.
    assert knobs["block_min_match"] == 128
    assert session_bank.DEFAULT_PREFIX_BLOCK_SIZE == 256
    assert session_bank.DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS == 512


def test_block_prefix_floor_is_rounded_up_to_a_block_on_every_lane(monkeypatch):
    """Every lane combines the two readers as max(block, min_match)."""
    from mtplx import session_bank

    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "256")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "128")
    assert ro.block_prefix_min_match_tokens() == 128
    assert session_bank._near_prefix_knobs()["block_min_match"] == 256
