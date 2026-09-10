"""CPU default-off audit for the typical-acceptance lane (design ruling 2026-09-06).

The lane is NEVER default-on: it is its own opt-in PR and turns on only when an
operator sets a positive threshold. These pin that contract:

- with a clean environment the generation-side knobs read OFF (threshold 0.0,
  ``_typical_accept_enabled()`` False) and the eps cap defaults to the inert 1.0;
- a positive threshold turns it on, and an explicit 0 is the off switch;
- the served Flash-Next default stack arms NO ``MTPLX_FABLE_TYPICAL_*`` key, so
  serving a Flash-Next pack never turns the lane on by itself.

The served defaults on upstream main (2.11.2) are the auto-arm block
``mtplx.server.openai._server_runtime_env_overrides`` (the fixed-M4 lane keys
``_QWEN4_PORT_KEYS`` / ``_QWEN4_LANE_KEYS``), the runtime profiles in
``mtplx.profiles`` (``PROFILES`` / ``NATIVE_MTP_60_FAST_PATH_ENV``), and the pack
contract allowlist ``MODEL_RUNTIME_ENV_OVERRIDE_KEYS``. 2.11.2 replaced the old
``mtplx.full_stack_env`` pack registry with this block, so the audit runs against
it directly. The only writer of a typical env key anywhere in the package is the
explicit ``--typical-threshold`` CLI handler.

The served fixed-M4 stack this PR sits on also arms the PR #475 aux lanes
(``MTPLX_QWEN4_PLE_CACHED_AUX`` / ``MTPLX_QSA_POOLED_ROWSEL``, each with a
``MTPLX_FABLE_*`` alias) and the PR-391 remainder lanes (``MTPLX_QWEN4_HC_M4`` /
``MTPLX_QWEN4_PREFILL_MASK_FUSE`` / ``MTPLX_QSA_PREFILL_QUERY_TILE``). None of those
keys, aliases, or their arming may ever be, or stamp, a typical key.
"""
from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.profiles as profiles
import mtplx.qwen4_aux_lanes as aux_lanes
import mtplx.server.openai as server
from mtplx.qwen4_fixed_verify import is_qwen4_fixed_verify_config

_TYPICAL_PREFIX = "MTPLX_FABLE_TYPICAL"
_TYPICAL_ENV = (
    "MTPLX_FABLE_TYPICAL_THRESHOLD",
    "MTPLX_FABLE_TYPICAL_EPS",
    "MTPLX_FABLE_TYPICAL_ACCEPT",
    "MTPLX_FABLE_TYPICAL_DELTA",
)

# The one measured Flash-Next fixed-M4 geometry, mirroring the tuple pinned in
# mtplx.qwen4_fixed_verify.is_qwen4_fixed_verify_config. A sanity assertion below
# confirms the predicate still accepts it, so the functional exercise is
# guaranteed to drive the deep fixed-M4 lane (not just the outer family block).
_FIXED_M4_CONFIG = {
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


# The stacked Flash-Next lanes that arm on the fixed-M4 geometry.
# PR #475 aux lanes carry MTPLX_FABLE_* aliases; the PR-391-remainder lanes are
# upstream-native (no alias). None may ever be, or arm, a typical key.
_AUX_LANE_KEYS = ("MTPLX_QWEN4_PLE_CACHED_AUX", "MTPLX_QSA_POOLED_ROWSEL")
_AUX_LANE_ALIASES = ("MTPLX_FABLE_PLE_CACHED_AUX", "MTPLX_FABLE_QSA_POOLED_ROWSEL")
_REMAINDER_LANE_KEYS = (
    "MTPLX_QWEN4_HC_M4",
    "MTPLX_QWEN4_PREFILL_MASK_FUSE",
    "MTPLX_QSA_PREFILL_QUERY_TILE",
)
# The QSA split-K sparse-GQA decode lane (PR-391 remainder). SPARSE_DECODE is
# default-armed on fixed-M4 only when the native mtplx_native_qsa extension is
# built; its _TILE/_SPLITS companions are runtime_options defaults, never stamped
# into the server overrides. All three are in _QWEN4_LANE_KEYS and none is typical.
_SPARSE_DECODE_LANE_KEYS = (
    "MTPLX_QSA_SPARSE_DECODE",
    "MTPLX_QSA_SPARSE_DECODE_TILE",
    "MTPLX_QSA_SPARSE_DECODE_SPLITS",
)
# The verify-path lanes (PR-391 remainder, all resolved at use, not import): the
# FR-Spec-gated draft-K20 prescatter, block verification, opdiet, and the verify
# glue. All are in _QWEN4_LANE_KEYS and none is a typical key. (DRAFT_K20_PRESCATTER
# is conditionally armed -- only under FR-Spec draft -- so it is audited for the
# invariants but not for unconditional arming.)
_VERIFY_LANE_KEYS = (
    "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
    "MTPLX_QWEN4_BLOCK_VERIFY",
    "MTPLX_QWEN4_OPDIET",
    "MTPLX_QWEN4_VERIFY_GLUE",
)
# Keys the served fixed-M4 default stamps unconditionally, regardless of whether
# the native extensions are built (the sparse-decode default arm is native-gated,
# and DRAFT_K20_PRESCATTER is FR-Spec-gated, so they are audited for the invariants
# below but not for unconditional arming).
_DEFAULT_ARMED_LANE_KEYS = _AUX_LANE_KEYS + _REMAINDER_LANE_KEYS
# Every stacked Flash-Next lane key this audit covers.
_STACKED_LANE_KEYS = (
    _DEFAULT_ARMED_LANE_KEYS + _SPARSE_DECODE_LANE_KEYS + _VERIFY_LANE_KEYS
)


def _clean(monkeypatch) -> None:
    for key in _TYPICAL_ENV:
        monkeypatch.delenv(key, raising=False)


def _offenders(keys) -> list[str]:
    return sorted(k for k in keys if str(k).startswith(_TYPICAL_PREFIX))


def _run_auto_arm(monkeypatch, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Run the real auto-arm block over the served fixed-M4 geometry.

    Clears the typical env and every stacked-lane key/alias, applies ``extra_env``,
    then returns the resolved ``overrides`` dict for a fixed-M4 served model.
    """
    _clean(monkeypatch)
    for key in (*_STACKED_LANE_KEYS, *_AUX_LANE_ALIASES):
        monkeypatch.delenv(key, raising=False)
    for key, value in (extra_env or {}).items():
        monkeypatch.setenv(key, value)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(json.dumps(_FIXED_M4_CONFIG))
        args = SimpleNamespace(
            model=tmp,
            generation_mode="mtp",
            verify_strategy="",
            scheduler_mode="serial",
        )
        return server._server_runtime_env_overrides(args, {})


# --- generation-side knobs -------------------------------------------------


def test_generation_defaults_are_off(monkeypatch):
    _clean(monkeypatch)
    gen = pytest.importorskip("mtplx.generation")
    assert gen._typical_accept_threshold() == 0.0
    assert gen._typical_accept_enabled() is False
    assert gen._typical_accept_eps() == 1.0  # advanced cap, inert by default


def test_positive_threshold_turns_it_on_and_zero_is_the_off_switch(monkeypatch):
    _clean(monkeypatch)
    gen = pytest.importorskip("mtplx.generation")
    monkeypatch.setenv("MTPLX_FABLE_TYPICAL_THRESHOLD", "0.09")
    assert gen._typical_accept_threshold() == 0.09
    assert gen._typical_accept_enabled() is True
    monkeypatch.setenv("MTPLX_FABLE_TYPICAL_THRESHOLD", "0")
    assert gen._typical_accept_enabled() is False


# --- served Flash-Next defaults arm no typical key -------------------------


def test_fixed_m4_lane_key_tables_carry_no_typical_key():
    """The auto-arm block's key tables never list a typical key."""
    assert _offenders(server._QWEN4_PORT_KEYS) == []
    assert _offenders(server._QWEN4_LANE_KEYS) == []


def test_auto_arm_block_source_writes_no_typical_key():
    """AST enumeration of every key _server_runtime_env_overrides can stamp.

    Guard-independent: collects the literal key of every ``overrides.setdefault``,
    ``overrides[...] =`` and ``overrides.update({...})`` in the function body,
    regardless of which runtime branch stamps it. None may be a typical key.
    """
    source = Path(server.__file__).read_text("utf-8")
    fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_server_runtime_env_overrides"
    )
    written: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            written.add(node.args[0].value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    for key in arg.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            written.add(key.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    written.add(target.slice.value)
    # Guard against the extractor silently matching nothing (e.g. a rename).
    assert written, "found no env writers in _server_runtime_env_overrides"
    assert _offenders(written) == [], f"auto-arm block stamps typical keys: {written}"


def test_fixed_m4_config_is_still_the_measured_geometry():
    """Keep the functional exercise below honest: the predicate must accept it."""
    assert is_qwen4_fixed_verify_config(_FIXED_M4_CONFIG) is True


def test_auto_arm_block_run_on_fixed_m4_arms_no_typical_key(monkeypatch):
    """Run the real auto-arm block over the served fixed-M4 geometry.

    A clean typical environment plus the shipped config: the block stamps its
    full fixed-M4 lane (dozens of keys, including MTPLX_QWEN4_BLOCK_VERIFY) and
    must not stamp any MTPLX_FABLE_TYPICAL key.
    """
    overrides = _run_auto_arm(monkeypatch)
    assert "MTPLX_QWEN4_BLOCK_VERIFY" in overrides, "fixed-M4 lane did not arm"
    assert _offenders(overrides) == [], f"served stack armed typical keys: {overrides}"


# --- the stacked aux (#475) and remainder (#391) lanes are not the typical lane


def test_stacked_lane_keys_are_served_and_never_a_typical_key():
    """The two PR #475 aux lanes and the three PR-391 remainder lanes are real
    served lane keys (all in _QWEN4_LANE_KEYS) and none is a typical key."""
    for key in _STACKED_LANE_KEYS:
        assert key in server._QWEN4_LANE_KEYS, f"{key} is not a served lane key"
    assert _offenders(_STACKED_LANE_KEYS) == []


def test_aux_lane_aliases_map_only_the_two_aux_lanes_and_are_not_typical():
    """The MTPLX_FABLE_* aliases mirror only the two aux primaries and are not
    the typical key. Both the server-side alias table and the mtplx.qwen4_aux_lanes
    module agree."""
    assert set(server._QWEN4_AUX_LANE_ALIASES) == set(_AUX_LANE_KEYS)
    assert set(server._QWEN4_AUX_LANE_ALIASES.values()) == set(_AUX_LANE_ALIASES)
    assert _offenders(server._QWEN4_AUX_LANE_ALIASES) == []
    assert _offenders(server._QWEN4_AUX_LANE_ALIASES.values()) == []
    assert _offenders(aux_lanes.LANE_KEYS.values()) == []
    assert _offenders(aux_lanes.LANE_ALIASES) == []
    assert _offenders(aux_lanes.LANE_ALIASES.values()) == []


def test_default_fixed_m4_stack_arms_every_default_lane_and_no_typical_key(monkeypatch):
    """The served fixed-M4 default arms every unconditionally-armed stacked lane
    (the aux and remainder keys) and stamps no typical key. The sparse-decode
    default arm is native-gated, so it is not asserted here."""
    overrides = _run_auto_arm(monkeypatch)
    for key in _DEFAULT_ARMED_LANE_KEYS:
        assert overrides.get(key), f"{key} not armed by the served default stack"
    assert _offenders(overrides) == []


def test_arming_each_stacked_lane_never_stamps_a_typical_key(monkeypatch):
    """Arm each lane on its own — primaries, and the aux MTPLX_FABLE_* aliases —
    and the resolved served env never introduces a typical key. An aux alias also
    mirrors onto its primary."""
    for key in _STACKED_LANE_KEYS:
        overrides = _run_auto_arm(monkeypatch, {key: "1"})
        assert _offenders(overrides) == [], f"arming {key} stamped a typical key"
    for primary, alias in server._QWEN4_AUX_LANE_ALIASES.items():
        overrides = _run_auto_arm(monkeypatch, {alias: "1"})
        assert overrides.get(primary) == "1", f"{alias} did not mirror onto {primary}"
        assert _offenders(overrides) == [], f"arming {alias} stamped a typical key"


def test_default_profiles_and_pack_allowlist_carry_no_typical_key():
    """Every runtime profile's env, the base fast-path env, and the pack
    contract allowlist carry no typical key. The allowlist matters most: a pack
    contract that names any key outside it fails normalize_runtime_env_overrides,
    so a pack physically cannot arm the lane."""
    profile_env_keys: set[str] = set(profiles.NATIVE_MTP_60_FAST_PATH_ENV)
    for profile in profiles.PROFILES.values():
        profile_env_keys |= set(profile.env_dict())
    assert _offenders(profile_env_keys) == []
    assert _offenders(server.FAST_PATH_ENV) == []
    assert _offenders(profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS) == []


def test_only_the_cli_flag_handler_writes_the_typical_env_key():
    """Whole-package sweep: the sole environ writer of a typical key is the
    explicit --typical-threshold handler. No profile / pack / driver writes it."""
    pattern = re.compile(
        r"(?:os\.environ\[[^]]*MTPLX_FABLE_TYPICAL"
        r"|environ\.setdefault\([^)]*MTPLX_FABLE_TYPICAL"
        r"|\.setdefault\([^)]*MTPLX_FABLE_TYPICAL)"
    )
    root = Path(server.__file__).resolve().parents[1]  # the mtplx package dir
    writers: list[str] = []
    for py in root.rglob("*.py"):
        if "tests" in py.parts:
            continue
        for lineno, line in enumerate(py.read_text("utf-8").splitlines(), 1):
            if pattern.search(line):
                writers.append(f"{py.relative_to(root.parent)}:{lineno}")
    assert writers == ["mtplx/server/openai.py:{}".format(
        _typical_flag_write_lineno(Path(server.__file__))
    )], f"unexpected typical-key writers: {writers}"


def _typical_flag_write_lineno(server_path: Path) -> int:
    for lineno, line in enumerate(server_path.read_text("utf-8").splitlines(), 1):
        if 'os.environ["MTPLX_FABLE_TYPICAL_THRESHOLD"] = str(float(args.typical_threshold))' in line:
            return lineno
    raise AssertionError("the --typical-threshold flag handler is gone")


def test_health_reports_lane_off_with_no_flag(monkeypatch):
    """/health reports the lane off when no flag / env sets it."""
    _clean(monkeypatch)
    payload = server._typical_acceptance_health_payload()
    assert payload["enabled"] is False
    assert payload["threshold"] == 0.0
    assert payload["distribution_exact"] is True
