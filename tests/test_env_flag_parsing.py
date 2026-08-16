"""One parse, one meaning, for the documented MTPLX_* boolean/enum flags.

Each of these four vars had two or three independent readers that disagreed
on at least one spelling (docs/AUDIT_2026-07-18.md, Tier 2). The tests below
pin the agreement rather than any single reader's behaviour: every reader is
asked about the same spelling and must answer the same thing.
"""

from __future__ import annotations

import argparse

import pytest

from mtplx.runtime_options import (
    block_prefix_restore_enabled,
    env_bool,
    normalize_paged_kv_quantization,
)


# ---------------------------------------------------------------------------
# the shared parser


@pytest.mark.parametrize("spelling", ["1", "true", "TRUE", " yes ", "on", "enabled"])
def test_env_bool_accepts_the_true_vocabulary(monkeypatch, spelling: str) -> None:
    monkeypatch.setenv("MTPLX_TEST_FLAG", spelling)
    assert env_bool("MTPLX_TEST_FLAG", default=False) is True


@pytest.mark.parametrize("spelling", ["0", "false", "No", "off", "disabled"])
def test_env_bool_accepts_the_false_vocabulary(monkeypatch, spelling: str) -> None:
    monkeypatch.setenv("MTPLX_TEST_FLAG", spelling)
    assert env_bool("MTPLX_TEST_FLAG", default=True) is False


@pytest.mark.parametrize("default", [True, False])
def test_env_bool_unset_and_empty_take_the_default(monkeypatch, default: bool) -> None:
    monkeypatch.delenv("MTPLX_TEST_FLAG", raising=False)
    assert env_bool("MTPLX_TEST_FLAG", default=default) is default
    monkeypatch.setenv("MTPLX_TEST_FLAG", "   ")
    assert env_bool("MTPLX_TEST_FLAG", default=default) is default


@pytest.mark.parametrize("spelling", ["unlimited", "maybe", "2", "none"])
def test_env_bool_raises_rather_than_guessing(monkeypatch, spelling: str) -> None:
    monkeypatch.setenv("MTPLX_TEST_FLAG", spelling)
    with pytest.raises(ValueError, match="is not a boolean"):
        env_bool("MTPLX_TEST_FLAG", default=False)


# ---------------------------------------------------------------------------
# MTPLX_SESSION_BLOCK_PREFIX_RESTORE — 4 readers that used to hold 3 semantics


def _block_prefix_readers() -> list:
    """Every independent reader of the flag, as zero-arg predicates."""

    from mtplx.engine_session import _block_prefix_restore_enabled
    from mtplx.server.openai import _effective_ram_session_cache_settings

    return [
        block_prefix_restore_enabled,
        _block_prefix_restore_enabled,
        lambda: _effective_ram_session_cache_settings()[
            "ram_session_block_prefix_restore"
        ],
    ]


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        (None, True),  # unset meant OFF in session_bank's cold tier
        ("1", True),
        ("0", False),
        ("off", False),
        ("enabled", True),  # the server's allowlist read this as OFF
        ("on", True),
    ],
)
def test_block_prefix_restore_readers_agree(
    monkeypatch, spelling: str | None, expected: bool
) -> None:
    if spelling is None:
        monkeypatch.delenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", raising=False)
    else:
        monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", spelling)
    for reader in _block_prefix_readers():
        assert bool(reader()) is expected, reader


def test_block_prefix_restore_cold_tier_defaults_on(monkeypatch) -> None:
    """The cold-tier lookup used to bail out whenever the var was unset."""

    from mtplx.session_bank import SessionBank

    monkeypatch.delenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", raising=False)

    seen: dict = {}

    class _ColdTier:
        def lookup_prefix_boundary(self, tokens, **kwargs):
            seen["called"] = True
            return None

    bank = SessionBank.__new__(SessionBank)
    bank.cold_tier = _ColdTier()
    bank._cold_near_prefix_candidate(
        [1, 2, 3],
        max_token_gap=0,
        min_matched_tokens=1,
        block_size=1,
        block_min_matched_tokens=1,
        allow_block_prefix=True,
        model_path="/model",
        mtp_enabled=True,
        hidden_variant=None,
        template_hash=None,
        mtp_history_policy=None,
        draft_head_identity=None,
        policy_fingerprint=None,
    )
    assert seen.get("called") is True


# ---------------------------------------------------------------------------
# MTPLX_PAGED_KV_QUANT — normalize / raise / silently-wrong-layout


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("8", "q8"),
        ("8bit", "q8"),
        ("uint8", "q8"),
        ("int8", "q8"),
        ("q8_0", "q8"),
        ("q8", "q8"),
        ("6", "q6"),
        ("6bit", "q6"),
        ("uint6", "q6"),
        ("int6", "q6"),
        ("q6", "q6"),
        ("4", "q4"),
        ("uint4", "q4"),
        ("q4", "q4"),
        ("off", "off"),
        ("none", "off"),
        ("", "off"),
    ],
)
def test_paged_kv_quant_readers_agree(
    monkeypatch, spelling: str, expected: str
) -> None:
    from mtplx.generation import _sustained_prefill_layout
    from mtplx.kv_quant import config_from_env, paged_kv_quant_mode_from_env
    from mtplx.server.openai import _effective_paged_kv_quantization

    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", spelling)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)

    assert paged_kv_quant_mode_from_env() == expected
    assert normalize_paged_kv_quantization(spelling) == expected
    assert _effective_paged_kv_quantization() == expected

    config = config_from_env()
    if expected == "off":
        assert config is None
    else:
        assert config is not None and config.normalized_mode == expected

    # The layout picker used to test raw membership, so "8"/"8bit"/"uint8"
    # fell through to the dense-decode layout with a quantized cache.
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    layout = _sustained_prefill_layout()
    if expected == "off":
        assert layout in {"contiguous_dense_decode", "contiguous_then_repage"}
    else:
        assert layout == "contiguous_then_repage"


def test_paged_kv_quant_rejects_an_unknown_mode(monkeypatch) -> None:
    from mtplx.kv_quant import config_from_env

    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q3")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    with pytest.raises(ValueError, match="unsupported paged KV quantization"):
        config_from_env()


@pytest.mark.parametrize(
    "var", ["MTPLX_VLLM_METAL_PAGED_KV_QUANT", "MTPLX_PAGED_KV_QUANT"]
)
def test_q6_reaches_the_cache_config_through_either_env_var(monkeypatch, var: str) -> None:
    from mtplx.kv_quant import config_from_env

    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv(var, "q6")

    config = config_from_env()
    assert config is not None
    assert config.normalized_mode == "q6"
    # The bit width has to follow the mode, or q6 would allocate q8 pages.
    assert config.bits == 6


def test_paged_kv_quantization_env_pair_carries_q6() -> None:
    from mtplx.runtime_options import paged_kv_quantization_env

    assert paged_kv_quantization_env("6bit") == {
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": "q6",
        "MTPLX_PAGED_KV_QUANT": "q6",
    }


# ---------------------------------------------------------------------------
# MTPLX_LOOP_GUARD — server answer vs the guard actually built


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [(None, False), ("1", True), ("on", True), ("0", False), ("off", False)],
)
def test_loop_guard_server_answer_matches_the_built_guard(
    monkeypatch, spelling: str | None, expected: bool
) -> None:
    from mtplx.loop_guard import loop_guard_config_from_env
    from mtplx.server.openai import _loop_guard_enabled

    if spelling is None:
        monkeypatch.delenv("MTPLX_LOOP_GUARD", raising=False)
    else:
        monkeypatch.setenv("MTPLX_LOOP_GUARD", spelling)

    reported = _loop_guard_enabled()
    built = loop_guard_config_from_env(reported).enabled
    assert reported is expected
    assert built is expected


@pytest.mark.parametrize("spelling", ["unlimited", "none"])
def test_loop_guard_no_longer_borrows_the_lease_vocabulary(
    monkeypatch, spelling: str
) -> None:
    """These made the server report "off" while the guard was built on."""

    from mtplx.loop_guard import loop_guard_config_from_env
    from mtplx.server.openai import _loop_guard_enabled

    monkeypatch.setenv("MTPLX_LOOP_GUARD", spelling)
    with pytest.raises(ValueError, match="is not a boolean"):
        _loop_guard_enabled()
    with pytest.raises(ValueError, match="is not a boolean"):
        loop_guard_config_from_env(False)


# ---------------------------------------------------------------------------
# MTPLX_SKIP_VERIFY_SNAPSHOT — the strategy list must fail safe


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [(None, False), ("1", True), ("enabled", True), ("0", False), ("off", False)],
)
def test_skip_verify_snapshot_single_parse(
    monkeypatch, spelling: str | None, expected: bool
) -> None:
    from mtplx.generation import _skip_verify_snapshot

    if spelling is None:
        monkeypatch.delenv("MTPLX_SKIP_VERIFY_SNAPSHOT", raising=False)
    else:
        monkeypatch.setenv("MTPLX_SKIP_VERIFY_SNAPSHOT", spelling)
    assert _skip_verify_snapshot() is expected


@pytest.mark.parametrize(
    "strategy", ["trim_commit", "target_prefix", "some_future_strategy"]
)
def test_unknown_verify_strategies_keep_the_snapshot(strategy: str) -> None:
    """Fail safe: a strategy nobody vetted must not inherit the fast-path skip."""

    from mtplx.server.openai import _server_runtime_env_overrides

    args = argparse.Namespace(verify_strategy=strategy, generation_mode="mtp")
    overrides = _server_runtime_env_overrides(args, {"MTPLX_SKIP_VERIFY_SNAPSHOT": "1"})
    assert overrides["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "0"


@pytest.mark.parametrize(
    "strategy", ["batched", "sequential", "capture", "capture_commit", "graphbank"]
)
def test_vetted_verify_strategies_still_skip(strategy: str) -> None:
    from mtplx.server.openai import _server_runtime_env_overrides

    args = argparse.Namespace(verify_strategy=strategy, generation_mode="mtp")
    overrides = _server_runtime_env_overrides(args, {"MTPLX_SKIP_VERIFY_SNAPSHOT": "1"})
    assert overrides["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "1"


# ---------------------------------------------------------------------------
# --chat-template-profile provenance


def _gemma4_args(profile: str) -> argparse.Namespace:
    from mtplx.server.openai import GEMMA4_BACKEND

    return argparse.Namespace(
        backend_id=GEMMA4_BACKEND,
        model=None,
        chat_template_profile=profile,
        reasoning_effort=None,
    )


def test_explicitly_typed_chat_template_profile_is_preserved() -> None:
    from mtplx.server.openai import (
        _CHAT_TEMPLATE_PROFILE_LOCAL,
        _apply_backend_server_defaults,
    )

    args = _gemma4_args(_CHAT_TEMPLATE_PROFILE_LOCAL)
    _apply_backend_server_defaults(args, explicit_flags={"chat-template-profile"})
    assert args.chat_template_profile == _CHAT_TEMPLATE_PROFILE_LOCAL


def test_abbreviated_flags_count_as_explicitly_typed() -> None:
    """``--temp 0.9`` set args.temperature but read as *not typed*.

    ~30 ``cli_flags`` checks key off that signal, so the config file then
    overwrote the value the user had just asked for.
    """

    from mtplx.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["serve", "--temp", "0.9"])
    assert args.temperature == pytest.approx(0.9)
    assert "temperature" in args._cli_flags
    # The raw token is kept too, so checks on either spelling still work.
    assert "temp" in args._cli_flags


def test_flag_canonicalization_does_not_invent_untyped_flags() -> None:
    from mtplx.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["serve", "--host", "1.2.3.4"])
    assert "host" in args._cli_flags
    assert "temperature" not in args._cli_flags
    assert "model" not in args._cli_flags


def test_abbreviations_still_parse() -> None:
    """The fix must not cost users their muscle memory."""

    from mtplx.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["serve", "--temp", "0.5"]).temperature == pytest.approx(
        0.5
    )


def test_untyped_chat_template_profile_still_gets_the_gemma4_default() -> None:
    from mtplx.server.openai import (
        _CHAT_TEMPLATE_PROFILE_LOCAL,
        _CHAT_TEMPLATE_PROFILE_TOKENIZER,
        _apply_backend_server_defaults,
    )

    args = _gemma4_args(_CHAT_TEMPLATE_PROFILE_LOCAL)
    _apply_backend_server_defaults(args, explicit_flags=set())
    assert args.chat_template_profile == _CHAT_TEMPLATE_PROFILE_TOKENIZER
