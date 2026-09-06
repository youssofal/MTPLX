"""Arming, opt-out and shared-disable contract for the stacked auxiliary lanes."""

from __future__ import annotations

import pytest

from mtplx import qwen4_aux_lanes as aux
from mtplx import full_stack_env


def test_lane_names_and_keys_are_the_two_stacked_lanes():
    assert aux.LANES == ("ple_cached_aux", "qsa_pooled_rowsel")
    assert aux.LANE_KEYS == {
        "ple_cached_aux": "MTPLX_FABLE_PLE_CACHED_AUX",
        "qsa_pooled_rowsel": "MTPLX_FABLE_QSA_POOLED_ROWSEL",
    }
    assert aux.STACKED_ENV == {
        "MTPLX_FABLE_PLE_CACHED_AUX": "1",
        "MTPLX_FABLE_QSA_POOLED_ROWSEL": "1",
    }


def test_default_env_arms_both_keys_when_unset():
    armed = aux.default_env({})
    assert armed == aux.STACKED_ENV


def test_operator_export_is_the_off_switch_per_key():
    env = {"MTPLX_FABLE_PLE_CACHED_AUX": "0"}
    armed = aux.default_env(env)
    # The exported key is left alone (its 0 is the off switch); the other arms.
    assert "MTPLX_FABLE_PLE_CACHED_AUX" not in armed
    assert armed == {"MTPLX_FABLE_QSA_POOLED_ROWSEL": "1"}


def test_resolve_disabled_reads_env_and_flag_and_composes():
    off = aux.resolve_disabled(
        {full_stack_env.DISABLE_ENV: "ple_cached_aux"},
        extra=["qsa_pooled_rowsel"],
    )
    assert off == frozenset({"ple_cached_aux", "qsa_pooled_rowsel"})


def test_resolve_disabled_ignores_measured_stack_lane_names():
    # A measured-stack lane is not this registry's concern; it is not returned
    # and does not raise here (full_stack_env validates the whole token list).
    off = aux.resolve_disabled({full_stack_env.DISABLE_ENV: "route_kernel"})
    assert off == frozenset()


def test_resolve_disabled_all_turns_off_every_stacked_lane():
    assert aux.resolve_disabled({full_stack_env.DISABLE_ENV: "all"}) == frozenset(
        aux.LANES
    )


def test_disable_lane_by_name_leaves_the_key_unstamped():
    off = aux.resolve_disabled(extra=["ple_cached_aux"])
    armed = aux.default_env({}, disabled_lanes=off)
    assert armed == {"MTPLX_FABLE_QSA_POOLED_ROWSEL": "1"}


def test_disable_all_leaves_both_unstamped():
    off = aux.resolve_disabled(extra=["all"])
    assert aux.default_env({}, disabled_lanes=off) == {}


def test_measured_resolver_tolerates_stacked_names_without_returning_them():
    # The shared switch must not raise on a stacked lane name, but the measured
    # resolver returns measured lanes only, so its "all" stays == LANES.
    assert full_stack_env.parse_disable_lanes(
        "ple_cached_aux,qsa_pooled_rowsel"
    ) == frozenset()
    assert full_stack_env.parse_disable_lanes(
        "ple_cached_aux,route_kernel"
    ) == frozenset({"route_kernel"})
    assert full_stack_env.parse_disable_lanes("all") == frozenset(full_stack_env.LANES)


def test_measured_resolver_still_raises_on_a_true_typo():
    with pytest.raises(ValueError):
        full_stack_env.parse_disable_lanes("ple_cachd_aux")


def test_registering_a_measured_stack_lane_name_is_rejected():
    with pytest.raises(ValueError):
        full_stack_env.register_extra_lanes({"route_kernel": ("X",)})


def test_re_registering_the_same_lane_is_idempotent():
    # The module already registered these; the same mapping must not raise.
    full_stack_env.register_extra_lanes(
        {lane: (key,) for lane, key in aux.LANE_KEYS.items()}
    )


def test_lane_enabled_reads_the_key_leniently():
    assert aux.ple_cached_aux_enabled({"MTPLX_FABLE_PLE_CACHED_AUX": "1"})
    assert aux.ple_cached_aux_enabled({"MTPLX_FABLE_PLE_CACHED_AUX": "on"})
    assert not aux.ple_cached_aux_enabled({"MTPLX_FABLE_PLE_CACHED_AUX": "0"})
    assert not aux.ple_cached_aux_enabled({})
    assert aux.qsa_pooled_rowsel_enabled({"MTPLX_FABLE_QSA_POOLED_ROWSEL": "yes"})


def test_defaults_report_shape_and_value_source():
    aux.DEFAULTS_APPLIED.clear()
    armed = aux.default_env({})
    aux.record_defaults_applied(armed)
    env = dict(armed)
    report = aux.defaults_report(env)
    assert set(report["armed_by_default"]) == set(aux.STACKED_ENV)
    assert report["lanes"] == list(aux.LANES)
    assert aux.value_source("MTPLX_FABLE_PLE_CACHED_AUX", env) == aux.SOURCE_DEFAULT
    # An operator value that differs from the default reads as operator.
    op_env = {"MTPLX_FABLE_PLE_CACHED_AUX": "0"}
    aux.DEFAULTS_APPLIED.clear()
    aux.record_defaults_applied({"MTPLX_FABLE_QSA_POOLED_ROWSEL": "1"})
    assert aux.value_source("MTPLX_FABLE_PLE_CACHED_AUX", op_env) == aux.SOURCE_OPERATOR
    aux.DEFAULTS_APPLIED.clear()


def test_defaults_report_marks_a_disabled_lane():
    aux.DEFAULTS_APPLIED.clear()
    off = aux.resolve_disabled(extra=["ple_cached_aux"])
    armed = aux.default_env({}, disabled_lanes=off)
    aux.record_defaults_applied(armed)
    report = aux.defaults_report({}, disabled_lanes=off)
    assert report["disabled_lanes"] == ["ple_cached_aux"]
    assert report["armed_by_default"] == ["MTPLX_FABLE_QSA_POOLED_ROWSEL"]
    aux.DEFAULTS_APPLIED.clear()
