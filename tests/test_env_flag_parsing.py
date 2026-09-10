"""One parse, one meaning, for the documented MTPLX_* boolean/enum flags.

Each of these four vars had two or three independent readers that disagreed
on at least one spelling (docs/AUDIT_2026-07-18.md, Tier 2). The tests below
pin the agreement rather than any single reader's behaviour: every reader is
asked about the same spelling and must answer the same thing.
"""

from __future__ import annotations

import argparse
import json

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


def _flash_next_fixed_m4_config() -> dict:
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "hc_count": 4,
            "hc_lowrank": 320,
            "indexer_compress_ratio": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "ple_layer_ids": [2],
            "ngram_size": 3,
            "ngram_vocab_size_base": 20_000_000,
            "heads_per_ngram": 8,
            "ple_embed_dim": 2560,
            "ngram_sidecar": True,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "vocab_size": 248_320,
        },
    }


def _flash_next_quantization(*, lm_head_bits: int, stage3: bool) -> dict:
    """Per-module quantization in the catalog packs' config.json shape.

    Optimized-Speed: lm_head Q8/g64 plus the stage-3 MoE layout (router and
    shared expert Q8/g64, routed experts Q4/g32). Bare-Speed: lm_head Q4/g64
    over a flat Q4/g64 MoE (2026-09-02 local config.json receipts).
    """
    head = {"bits": lm_head_bits, "group_size": 64, "mode": "affine"}
    router = {"bits": 8, "group_size": 64, "mode": "affine"}
    shared = {"bits": 8 if stage3 else 4, "group_size": 64, "mode": "affine"}
    routed = {"bits": 4, "group_size": 32 if stage3 else 64, "mode": "affine"}
    quantization: dict = {
        "bits": 4,
        "group_size": 32 if stage3 else 64,
        "mode": "affine",
        "language_model.lm_head": head,
    }
    for index in range(48):
        prefix = f"language_model.model.layers.{index}.mlp."
        quantization[prefix + "gate"] = dict(router)
        quantization[prefix + "shared_expert_gate"] = dict(shared)
        for name in ("gate_proj", "up_proj", "down_proj"):
            quantization[prefix + "shared_expert." + name] = dict(shared)
            quantization[prefix + "switch_mlp." + name] = dict(routed)
    return quantization


_FLASH_NEXT_LANE_KEYS = (
    "MTPLX_QWEN4_BATCHED_TARGET_DISTRIBUTIONS",
    "MTPLX_QWEN4_FIXED_M4_VERIFY",
    "MTPLX_QWEN4_COMPILED_MTP_PREPARE",
    "MTPLX_QWEN4_RELAXED_DRAFT_TIES",
    "MTPLX_QWEN4_M4_STAGE3",
    "MTPLX_QSA_M4_FUSED_KV_GATHER",
    "MTPLX_QSA_GATHER_MAX_ROWS",
    "MTPLX_FRSPEC_DRAFT",
    "MTPLX_FRSPEC_VOCAB",
    "MTPLX_COMPILED_VERIFY",
    "MTPLX_BATCH_TARGET_ARRAYS",
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
)


def test_flash_next_speed_lane_is_default_on_and_pack_gated(
    tmp_path, monkeypatch
) -> None:
    """The Flash-Next speed lane (PR #391 ports by davidtai) is ON by default
    on the one measured fixed-M4 geometry (2026-09-02 A/B/A receipts): the
    load-time gates are pack-checked from config.json, the fused K/V gather
    is derived from the resolved rows-gather lane, and every key yields to
    an explicit operator export, =0 included."""

    from mtplx.profiles import normalize_runtime_env_overrides
    from mtplx.server.openai import _server_runtime_env_overrides

    for key in (*_FLASH_NEXT_LANE_KEYS, "MTPLX_QSA_GATHER", "MTPLX_FUSED_GATE_UP"):
        monkeypatch.delenv(key, raising=False)
    model = tmp_path / "flash-next"
    model.mkdir()

    def write(config: dict) -> None:
        (model / "config.json").write_text(json.dumps(config))

    optimized = _flash_next_fixed_m4_config()
    optimized["quantization"] = _flash_next_quantization(lm_head_bits=8, stage3=True)
    write(optimized)
    args = argparse.Namespace(
        model=str(model),
        verify_strategy="batched",
        generation_mode="mtp",
        scheduler_mode="serial",
    )

    # Optimized-Speed shape: the whole lane, stage 3 and FR-Spec included,
    # and every stamped key survives the boot-time validator.
    overrides = _server_runtime_env_overrides(args, {})
    for key in (
        "MTPLX_QWEN4_BATCHED_TARGET_DISTRIBUTIONS",
        "MTPLX_QWEN4_FIXED_M4_VERIFY",
        "MTPLX_QWEN4_COMPILED_MTP_PREPARE",
        "MTPLX_QWEN4_RELAXED_DRAFT_TIES",
        "MTPLX_QWEN4_M4_STAGE3",
        "MTPLX_QSA_M4_FUSED_KV_GATHER",
        "MTPLX_FRSPEC_DRAFT",
        "MTPLX_COMPILED_VERIFY",
        "MTPLX_BATCH_TARGET_ARRAYS",
    ):
        assert overrides.get(key) == "1", key
    assert overrides["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "0"
    assert overrides["MTPLX_QSA_GATHER_MAX_ROWS"] == "32"
    assert overrides["MTPLX_FRSPEC_VOCAB"] == "builtin:qwen38-code-64k"
    # The 2026-09-03 ports from PR #391 (davidtai): the decode set, the two
    # prefill overlaps, the session-bank pair and the n-gram pre-read, with
    # the load-time children derived from their resolved parents.
    for key in (
        "MTPLX_QWEN4_OPDIET",
        "MTPLX_QWEN4_BLOCK_VERIFY",
        "MTPLX_QWEN4_PLE_PREFILL_LOOKAHEAD",
        "MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY",
        "MTPLX_SESSION_BANK_SHED_BOUNDARIES",
        "MTPLX_SESSION_BANK_PROTECTED_TERMINAL",
        "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE",
        "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL",
        "MTPLX_QWEN4_M4_ROUTED_GLU",
        "MTPLX_QWEN4_ROUTE_KERNEL",
        "MTPLX_QWEN4_VERIFY_GLUE",
        "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
    ):
        assert overrides.get(key) == "1", key
    assert overrides["MTPLX_QWEN4_VERIFY_GLUE_ITEMS"] == "qsa_rope,qsa_rope_idx"
    assert overrides["MTPLX_NGRAM_PREWARM"] == "auto"
    assert normalize_runtime_env_overrides(overrides) == overrides

    # Bare-Speed shape (lm_head Q4/g64 over a flat Q4/g64 MoE): FR-Spec and the
    # M=4 stage-3 optimization now arm here too -- the pruning and the combine
    # read the modules' own bits/group_size. Only the Q8-shared-gate route head
    # stays off (deferred; Bare runs the stock routing head).
    bare = _flash_next_fixed_m4_config()
    bare["quantization"] = _flash_next_quantization(lm_head_bits=4, stage3=False)
    write(bare)
    overrides = _server_runtime_env_overrides(args, {})
    assert overrides["MTPLX_QWEN4_FIXED_M4_VERIFY"] == "1"
    assert overrides["MTPLX_QSA_M4_FUSED_KV_GATHER"] == "1"
    assert overrides["MTPLX_QWEN4_M4_STAGE3"] == "1"
    assert overrides["MTPLX_FRSPEC_DRAFT"] == "1"
    assert overrides["MTPLX_FRSPEC_VOCAB"] == "builtin:qwen38-code-64k"
    assert overrides["MTPLX_QWEN4_DRAFT_K20_PRESCATTER"] == "1"
    # Stage 3's routed children arm with it; the Q8-only route head does not.
    for key in (
        "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE",
        "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL",
        "MTPLX_QWEN4_M4_ROUTED_GLU",
    ):
        assert overrides.get(key) == "1", key
    assert "MTPLX_QWEN4_ROUTE_KERNEL" not in overrides
    assert overrides["MTPLX_QWEN4_VERIFY_GLUE"] == "1"
    assert overrides["MTPLX_QWEN4_BLOCK_VERIFY"] == "1"
    assert normalize_runtime_env_overrides(overrides) == overrides
    # A pack with no per-module entries resolves every module to the pack-wide
    # values: the Q4/g64 lm_head is still FR-Spec-capable, but the pack-wide
    # Q4 router can never satisfy the stage-3 contract (router must be Q8/g64).
    flat = _flash_next_fixed_m4_config()
    flat["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    write(flat)
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_QWEN4_M4_STAGE3" not in overrides
    assert overrides["MTPLX_FRSPEC_DRAFT"] == "1"
    write(optimized)

    # An explicit =0 export wins for every lane key and drops its companions.
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_VERIFY", "0")
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_QWEN4_FIXED_M4_VERIFY" not in overrides
    assert "MTPLX_COMPILED_VERIFY" not in overrides
    assert "MTPLX_QSA_M4_FUSED_KV_GATHER" not in overrides
    assert overrides["MTPLX_QWEN4_M4_STAGE3"] == "1"
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_VERIFY")
    monkeypatch.setenv("MTPLX_QWEN4_BATCHED_TARGET_DISTRIBUTIONS", "0")
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_QWEN4_BATCHED_TARGET_DISTRIBUTIONS" not in overrides
    assert "MTPLX_BATCH_TARGET_ARRAYS" not in overrides
    assert "MTPLX_LAZY_TARGET_DISTRIBUTIONS" not in overrides
    monkeypatch.delenv("MTPLX_QWEN4_BATCHED_TARGET_DISTRIBUTIONS")
    for key in (
        "MTPLX_QWEN4_M4_STAGE3",
        "MTPLX_QWEN4_COMPILED_MTP_PREPARE",
        "MTPLX_QWEN4_RELAXED_DRAFT_TIES",
        "MTPLX_QSA_M4_FUSED_KV_GATHER",
        "MTPLX_QSA_GATHER_MAX_ROWS",
    ):
        monkeypatch.setenv(key, "0")
        overrides = _server_runtime_env_overrides(args, {})
        assert key not in overrides, key
        monkeypatch.delenv(key)
    monkeypatch.setenv("MTPLX_FRSPEC_DRAFT", "0")
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_FRSPEC_DRAFT" not in overrides
    assert "MTPLX_FRSPEC_VOCAB" not in overrides
    assert "MTPLX_QWEN4_DRAFT_K20_PRESCATTER" not in overrides
    monkeypatch.delenv("MTPLX_FRSPEC_DRAFT")
    # The stage-3 children follow their parent's kill switch, and the routing
    # head follows the paired GLU; the verify glue follows the fixed verifier.
    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "0")
    overrides = _server_runtime_env_overrides(args, {})
    for key in (
        "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE",
        "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL",
        "MTPLX_QWEN4_M4_ROUTED_GLU",
        "MTPLX_QWEN4_ROUTE_KERNEL",
    ):
        assert key not in overrides, key
    monkeypatch.delenv("MTPLX_QWEN4_M4_STAGE3")
    monkeypatch.setenv("MTPLX_QWEN4_M4_ROUTED_GLU", "0")
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_QWEN4_ROUTE_KERNEL" not in overrides
    assert overrides["MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE"] == "1"
    monkeypatch.delenv("MTPLX_QWEN4_M4_ROUTED_GLU")
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_VERIFY", "0")
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_QWEN4_VERIFY_GLUE" not in overrides
    assert "MTPLX_QWEN4_VERIFY_GLUE_ITEMS" not in overrides
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_VERIFY")
    for key in (
        "MTPLX_QWEN4_OPDIET",
        "MTPLX_QWEN4_BLOCK_VERIFY",
        "MTPLX_QWEN4_PLE_PREFILL_LOOKAHEAD",
        "MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY",
        "MTPLX_SESSION_BANK_SHED_BOUNDARIES",
        "MTPLX_SESSION_BANK_PROTECTED_TERMINAL",
        "MTPLX_NGRAM_PREWARM",
    ):
        monkeypatch.setenv(key, "0" if key != "MTPLX_NGRAM_PREWARM" else "off")
        overrides = _server_runtime_env_overrides(args, {})
        assert key not in overrides, key
        monkeypatch.delenv(key)
    # An operator's own ranked table or row width beats the stamped values.
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(tmp_path / "ranked.npy"))
    monkeypatch.setenv("MTPLX_QSA_GATHER_MAX_ROWS", "8")
    overrides = _server_runtime_env_overrides(
        args, {"MTPLX_QSA_GATHER_MAX_ROWS": "32"}
    )
    assert overrides["MTPLX_FRSPEC_DRAFT"] == "1"
    assert "MTPLX_FRSPEC_VOCAB" not in overrides
    assert "MTPLX_QSA_GATHER_MAX_ROWS" not in overrides
    monkeypatch.delenv("MTPLX_FRSPEC_VOCAB")
    monkeypatch.delenv("MTPLX_QSA_GATHER_MAX_ROWS")

    # The fused K/V gather follows the resolved rows-gather lane: the kill
    # switch MTPLX_QSA_GATHER=0, from the launch env or a pack, never leaves
    # the fused flag armed (graphbank.from_qsa_cache would refuse the lane).
    monkeypatch.setenv("MTPLX_QSA_GATHER", "0")
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_QSA_M4_FUSED_KV_GATHER" not in overrides
    assert overrides["MTPLX_QWEN4_FIXED_M4_VERIFY"] == "1"
    assert overrides["MTPLX_QSA_GATHER_MAX_ROWS"] == "32"
    monkeypatch.delenv("MTPLX_QSA_GATHER")
    overrides = _server_runtime_env_overrides(args, {"MTPLX_QSA_GATHER": "0"})
    assert overrides["MTPLX_QSA_GATHER"] == "0"
    assert "MTPLX_QSA_M4_FUSED_KV_GATHER" not in overrides
    # Stage 3 needs the fused gate+up owners; that kill switch drops the tail.
    monkeypatch.setenv("MTPLX_FUSED_GATE_UP", "0")
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_QWEN4_M4_STAGE3" not in overrides
    assert overrides["MTPLX_QWEN4_FIXED_M4_VERIFY"] == "1"
    monkeypatch.delenv("MTPLX_FUSED_GATE_UP")

    # A pack contract value is carried unchanged, and an export beats it.
    overrides = _server_runtime_env_overrides(args, {"MTPLX_QWEN4_M4_STAGE3": "0"})
    assert overrides["MTPLX_QWEN4_M4_STAGE3"] == "0"
    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "1")
    overrides = _server_runtime_env_overrides(args, {"MTPLX_QWEN4_M4_STAGE3": "0"})
    assert "MTPLX_QWEN4_M4_STAGE3" not in overrides
    monkeypatch.delenv("MTPLX_QWEN4_M4_STAGE3")

    # An explicit compiled-verify export (the parity gates) wins over the pin.
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")
    overrides = _server_runtime_env_overrides(args, {})
    assert "MTPLX_COMPILED_VERIFY" not in overrides
    assert overrides["MTPLX_QWEN4_FIXED_M4_VERIFY"] == "1"
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY")

    # Any other qwen4_exp layout arms nothing, even with the Q8 head.
    other = _flash_next_fixed_m4_config()
    other["text_config"]["hidden_size"] = 2048
    other["quantization"] = _flash_next_quantization(lm_head_bits=8, stage3=True)
    write(other)
    overrides = _server_runtime_env_overrides(args, {})
    for key in _FLASH_NEXT_LANE_KEYS:
        assert key not in overrides, key


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
