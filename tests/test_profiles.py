from __future__ import annotations

import pytest

from mtplx.profiles import (
    DEFAULT_PROFILE_NAME,
    NATIVE_MTP_60_FAST_PATH_ENV,
    SUSTAINED_PREFILL_ENV,
    apply_profile_env,
    get_profile,
    list_profiles,
    normalize_runtime_env_overrides,
    profile_env_status,
    restore_profile_env,
    resolve_long_context_mtp_depth,
    resolve_profile_name,
    runtime_env_with_contract_overrides,
)


def test_profile_registry_default_is_sustained() -> None:
    profile = get_profile()

    assert DEFAULT_PROFILE_NAME == "sustained"
    assert profile.name == "sustained"
    assert profile.runtime_profile == "native_mtp_sustained"
    assert profile.product_claim_eligible is True


def test_performance_cold_is_explicit_fast_path() -> None:
    profile = get_profile("performance-cold")

    assert profile.runtime_profile == "native_mtp_60_cold"
    assert profile.draft_lm_head is not None
    assert profile.env_dict() == NATIVE_MTP_60_FAST_PATH_ENV
    assert "MTPLX_SUSTAINED_PREFILL_LAYOUT" not in profile.env_dict()


def test_no_profile_requires_or_mentions_an_mlx_fork() -> None:
    # MTPLX runs on stock PyPI MLX. The old required_mlx_fork_commit /
    # required_mlx_fork_fragment metadata was vestigial research residue
    # that kept resurfacing in user bug reports as "MTPLX needs a custom
    # tuned qmm fork" (issue #129). It must never come back: no profile
    # attribute, no payload key, and no caveat may claim a fork is
    # required.
    for payload in list_profiles():
        assert "required_mlx_fork_commit" not in payload
        assert "required_mlx_fork_fragment" not in payload
        for caveat in payload["caveats"]:
            assert "Requires the MLX-MTPLX fork" not in caveat
    for name in ("performance-cold", "sustained", "turbo"):
        profile = get_profile(name)
        assert not hasattr(profile, "required_mlx_fork_commit")
        assert not hasattr(profile, "required_mlx_fork_fragment")


def test_legacy_native_mtp_60_alias_resolves_to_performance_cold() -> None:
    assert get_profile("native-mtp-60").name == "performance-cold"
    assert get_profile("default").name == "sustained"


def test_apply_and_restore_profile_env() -> None:
    environ: dict[str, str] = {}

    previous = apply_profile_env("performance-cold", environ=environ)
    assert previous == {key: None for key in NATIVE_MTP_60_FAST_PATH_ENV}
    assert profile_env_status("performance-cold", environ=environ)[
        "MTPLX_LAZY_VERIFY_LOGITS"
    ]["ok"] is True

    restore_profile_env(previous, environ=environ)
    assert environ == {}


def test_model_runtime_env_overrides_can_disable_fast_path_flags() -> None:
    environ: dict[str, str] = {}
    overrides = {
        "MTPLX_LAZY_VERIFY_LOGITS": False,
        "MTPLX_BATCH_TARGET_ARRAYS": "0",
    }

    previous = apply_profile_env(
        "performance-cold",
        environ=environ,
        runtime_env_overrides=overrides,
    )

    assert previous == {key: None for key in NATIVE_MTP_60_FAST_PATH_ENV}
    assert environ["MTPLX_LAZY_VERIFY_LOGITS"] == "0"
    assert environ["MTPLX_BATCH_TARGET_ARRAYS"] == "0"
    status = profile_env_status(
        "performance-cold",
        environ=environ,
        runtime_env_overrides=overrides,
    )
    assert status["MTPLX_LAZY_VERIFY_LOGITS"]["ok"] is True
    assert status["MTPLX_BATCH_TARGET_ARRAYS"]["ok"] is True
    assert status["MTPLX_LAZY_TARGET_DISTRIBUTIONS"]["ok"] is True

    restore_profile_env(previous, environ=environ)
    assert environ == {}


def test_contract_runtime_env_overrides_are_normalized_and_restricted() -> None:
    contract = {
        "runtime_env_overrides": {
            "MTPLX_LAZY_VERIFY_LOGITS": False,
            "MTPLX_MTP_HISTORY_POLICY": "committed",
            "MTPLX_CLEAR_CACHE_EVERY": 512,
        }
    }

    merged = runtime_env_with_contract_overrides(
        NATIVE_MTP_60_FAST_PATH_ENV,
        contract,
    )

    assert merged["MTPLX_LAZY_VERIFY_LOGITS"] == "0"
    assert merged["MTPLX_MTP_HISTORY_POLICY"] == "committed"
    assert merged["MTPLX_CLEAR_CACHE_EVERY"] == "512"
    try:
        normalize_runtime_env_overrides({"MTPLX_UNSAFE_NEW_FLAG": "1"})
    except ValueError as exc:
        assert "unsupported key" in str(exc)
    else:
        raise AssertionError("unknown runtime env override should fail")


def test_apply_profile_env_preserves_mtp_history_policy_override() -> None:
    environ = {"MTPLX_MTP_HISTORY_POLICY": "committed"}

    previous = apply_profile_env("sustained", environ=environ)

    assert previous["MTPLX_MTP_HISTORY_POLICY"] == "committed"
    assert environ["MTPLX_MTP_HISTORY_POLICY"] == "committed"
    assert profile_env_status("sustained", environ=environ)["MTPLX_MTP_HISTORY_POLICY"][
        "ok"
    ] is True

    restore_profile_env(previous, environ=environ)
    assert environ["MTPLX_MTP_HISTORY_POLICY"] == "committed"


def test_apply_profile_env_preserves_turboquant_overrides() -> None:
    environ = {
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT": "1",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT": "q8_0",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT": "q3_0",
    }

    apply_profile_env("sustained", environ=environ)

    assert environ["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] == "1"
    assert environ["MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT"] == "q8_0"
    assert environ["MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT"] == "q3_0"
    status = profile_env_status("sustained", environ=environ)
    assert status["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"]["ok"] is True


def test_apply_profile_env_preserves_long_context_depth_overrides() -> None:
    environ = {
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY": "auto",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD": "65536",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH": "2",
    }

    apply_profile_env("sustained", environ=environ)

    assert environ["MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"] == "auto"
    assert environ["MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"] == "65536"
    assert environ["MTPLX_LONG_CONTEXT_MTP_DEPTH"] == "2"
    status = profile_env_status("sustained", environ=environ)
    assert status["MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"]["ok"] is True
    assert status["MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"]["ok"] is True
    assert status["MTPLX_LONG_CONTEXT_MTP_DEPTH"]["ok"] is True


def test_list_profiles_includes_all_public_modes() -> None:
    names = [profile["name"] for profile in list_profiles()]

    assert names == ["stable", "performance-cold", "sustained", "turbo", "exact", "max-diagnostic"]


def test_sustained_profile_is_native_mtp_long_context_path() -> None:
    profile = get_profile("sustained")

    assert profile.runtime_profile == "native_mtp_sustained"
    assert profile.draft_lm_head is not None
    assert profile.env_dict() == SUSTAINED_PREFILL_ENV
    assert profile.env_dict()["MTPLX_SUSTAINED_PREFILL_LAYOUT"] == "auto"
    assert profile.env_dict()["MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT"] == "131072"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_SIZE"] == "auto"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_SIZE_DENSE"] == "2048"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_SIZE_REPAGE"] == "2048"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] == "1"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY"] == "auto"
    assert profile.env_dict()["MTPLX_PREFILL_OMLX_EXTERNAL"] == "1"
    assert profile.env_dict()["MTPLX_PREFILL_EXTERNAL_EMIT_LOGITS"] == "0"
    assert profile.env_dict()["MTPLX_CLEAR_CACHE_EVERY"] == "auto"
    assert profile.env_dict()["MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD"] == "16384"
    assert profile.env_dict()["MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT"] == "1024"
    assert profile.env_dict()["MTPLX_LAZY_VERIFY_LOGITS"] == "1"
    assert profile.env_dict()["MTPLX_BATCH_TARGET_ARRAYS"] == "1"
    assert profile.env_dict()["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1"
    assert profile.env_dict()["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] == "1"
    assert profile.env_dict()["MTPLX_VERIFY_HIDDEN_MODE"] == "logits_first_committed_slice"
    assert profile.env_dict()["MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"] == "off"
    assert profile.env_dict()["MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"] == "98304"
    assert profile.env_dict()["MTPLX_LONG_CONTEXT_MTP_DEPTH"] == "3"
    assert profile.env_dict()["MTPLX_MTP_HISTORY_POLICY"] == "committed"
    assert profile.env_dict()["MTPLX_LAZY_MTP_HISTORY_APPEND"] == "1"
    assert profile.env_dict()["MTPLX_DROP_EVENTS"] == "1"
    assert profile.env_dict()["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "1"
    assert profile.env_dict()["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] == "0"
    assert "MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY" not in profile.env_dict()
    assert "MTPLX_EVAL_STATE_ROOTS_ON_COMMIT" not in profile.env_dict()


def test_sustained_long_context_depth_policy_keeps_d3_by_default() -> None:
    env = SUSTAINED_PREFILL_ENV

    depth, details = resolve_long_context_mtp_depth(
        prompt_tokens=65536,
        requested_depth=3,
        env=env,
    )
    assert depth == 3
    assert details["active"] is False
    assert details["reason"] == "disabled"

    depth, details = resolve_long_context_mtp_depth(
        prompt_tokens=131072,
        requested_depth=3,
        env=env,
    )
    assert depth == 3
    assert details["active"] is False
    assert details["reason"] == "disabled"
    assert details["requested_depth"] == 3
    assert details["effective_depth"] == 3

    depth, details = resolve_long_context_mtp_depth(
        prompt_tokens=131072,
        requested_depth=1,
        env=env,
    )
    assert depth == 1
    assert details["active"] is False
    assert details["reason"] == "disabled"


def test_long_context_depth_cap_is_explicit_diagnostic_only() -> None:
    env = {
        **SUSTAINED_PREFILL_ENV,
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY": "auto",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD": "98304",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH": "2",
    }

    depth, details = resolve_long_context_mtp_depth(
        prompt_tokens=131072,
        requested_depth=3,
        env=env,
    )

    assert depth == 2
    assert details["active"] is True
    assert details["reason"] == "long_context_depth_cap"
    assert details["requested_depth"] == 3
    assert details["effective_depth"] == 2


def test_legacy_app_profile_strings_resolve_to_shipping_profiles() -> None:
    # The V1 app Settings picker wrote "auto" and "sustained-max" into
    # persisted configs; both must stay launchable forever.
    assert resolve_profile_name("auto") == "sustained"
    assert resolve_profile_name("sustained-max") == "sustained"
    assert resolve_profile_name("sustained_max") == "sustained"
    assert get_profile("auto").name == "sustained"


def test_unknown_profile_error_lists_canonical_choices() -> None:
    with pytest.raises(ValueError, match="expected one of: stable"):
        resolve_profile_name("banana")
