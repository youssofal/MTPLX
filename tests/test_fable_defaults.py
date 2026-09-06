"""The retained Flash-Next stack, ON by default, and how to turn it off.

Pure Python: no MLX import, no model load, no GPU. Everything here is a
statement about what a served Qwen3.8 Flash-Next process resolves to and
about how an operator overrides it.

The change these tests pin (2026-09-03): the 26 keys the PR-391 battery
measured stopped being an opt-in. Before, a serve got them only by exporting
``docs/perf/pr391-stack.flags`` and ``docs/perf/pr391-prefill.flags`` by
hand, or by naming ``--profile turbo-full-stack``; now the server defaults
every one of them ON for this model family, by the same ``setdefault``
mechanism it already used for its sixteen M4 keys, and turning one off is a
normal env export.

The last three added were ``MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE``,
``..._DOWN_RESIDUAL_TAIL`` and ``MTPLX_QWEN4_M4_ROUTED_GLU`` were armed on
every measured branch arm out of the benchmark harness's
``BRANCH_BASE_ENV`` rather than out of either flag file, so they were missing
from the served stack -- and their absence made the already-defaulted
``MTPLX_FABLE_ROUTE_KERNEL`` raise at model install. Section (9) pins the
harness and the defaults together so that gap cannot reopen.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mtplx import full_stack_env as fse
from mtplx.full_stack_env import (
    DISABLE_ENV,
    FULL_STACK_PROFILE_ENV,
    IMPORT_TIME_PROFILE_ENV,
    KEY_LANE,
    LANES,
    LANE_KEYS,
    disabled_keys,
    fable_default_env,
    fable_defaults_report,
    is_flash_next_model_dir,
    parse_disable_lanes,
    resolve_disable_lanes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_SOURCE = (REPO_ROOT / "mtplx" / "server" / "openai.py").read_text()


@pytest.fixture(autouse=True)
def _clean_defaults_registry():
    """DEFAULTS_APPLIED is process state; no test may leak into the next."""

    saved = dict(fse.DEFAULTS_APPLIED)
    fse.DEFAULTS_APPLIED.clear()
    try:
        yield
    finally:
        fse.DEFAULTS_APPLIED.clear()
        fse.DEFAULTS_APPLIED.update(saved)


# ---------------------------------------------------------------------------
# (1) what the defaults arm
# ---------------------------------------------------------------------------


def test_a_clean_environment_gets_every_retained_key() -> None:
    armed = fable_default_env({})

    assert armed == FULL_STACK_PROFILE_ENV
    assert len(armed) == 26


def test_the_defaults_are_the_two_committed_files_plus_the_three_opt_ins() -> None:
    from mtplx.full_stack_env import GROUP_FLAG_FILES, parse_flag_file

    union = {
        "MTPLX_FRSPEC_DRAFT": "1",
        "MTPLX_FRSPEC_VOCAB": "builtin:qwen38-code-64k",
        "MTPLX_QWEN4_COMPILED_MTP_PREPARE": "1",
    }
    for relative in GROUP_FLAG_FILES.values():
        union.update(parse_flag_file((REPO_ROOT / relative).read_text()))

    assert fable_default_env({}) == union


@pytest.mark.parametrize("key", sorted(FULL_STACK_PROFILE_ENV))
def test_each_key_can_be_turned_off_on_its_own(key: str) -> None:
    """An export of ANY non-empty value takes the key away from the defaults.

    ``0`` is the ordinary off switch; a different value (a custom FR-Spec
    vocabulary, a bigger session bank) is the same mechanism used to retune
    rather than disable. Either way the defaults must not overwrite it.
    """

    armed = fable_default_env({key: "0"})

    assert key not in armed
    assert set(armed) == set(FULL_STACK_PROFILE_ENV) - {key}


def test_an_empty_export_is_not_an_override() -> None:
    """``FOO=`` is how a shell says "unset"; it must not disarm a lane."""

    assert "MTPLX_FABLE_OPDIET" in fable_default_env({"MTPLX_FABLE_OPDIET": ""})
    assert "MTPLX_FABLE_OPDIET" in fable_default_env({"MTPLX_FABLE_OPDIET": "  "})


# ---------------------------------------------------------------------------
# (2) the lane switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", sorted(LANES))
def test_each_lane_can_be_disabled_by_name(lane: str) -> None:
    armed = fable_default_env({}, disabled_lanes=[lane])

    assert set(armed) == set(FULL_STACK_PROFILE_ENV) - set(LANE_KEYS[lane])
    for key in LANE_KEYS[lane]:
        assert KEY_LANE[key] == lane


def test_disabling_a_lane_leaves_its_keys_UNSET_not_zero() -> None:
    """The five value knobs have no meaningful "0".

    ``MTPLX_FABLE_PREFILL_QSA_QUERY_TILE=0`` happens to mean "whole chunk",
    but ``MTPLX_PREFILL_CHUNK_SIZE=0`` and
    ``MTPLX_SESSION_BANK_MAX_BYTES=0`` are not configurations at all.
    Leaving the key out is what restores each reader's shipped default, and
    it is also the only spelling that does not make the key look armed.
    """

    armed = fable_default_env({}, disabled_lanes=["prefill_chunk", "session_bank_max_bytes"])

    for key in ("MTPLX_PREFILL_CHUNK_SIZE", "MTPLX_QSA_PREFILL_COMPILE_ROWS",
                "MTPLX_SESSION_BANK_MAX_BYTES"):
        assert key not in armed


def test_all_disables_everything_which_is_the_stock_path() -> None:
    assert parse_disable_lanes("all") == frozenset(LANES)
    assert disabled_keys(["all"]) == frozenset(FULL_STACK_PROFILE_ENV)
    assert fable_default_env({}, disabled_lanes=["all"]) == {}


def test_a_lane_list_composes_across_the_env_and_the_flag() -> None:
    lanes = resolve_disable_lanes(
        {DISABLE_ENV: "opdiet, verify_glue"},
        [],
        extra=["qsa_sparse_decode"],
    )

    assert lanes == frozenset({"opdiet", "verify_glue", "qsa_sparse_decode"})


def test_the_flag_is_read_from_argv_before_the_parser_exists() -> None:
    from mtplx.full_stack_env import disable_lanes_from_argv

    argv = [
        "-m", "--model", "/x", "--disable-optimization", "opdiet",
        "--disable-optimization=block_verify",
    ]

    assert disable_lanes_from_argv(argv) == ["opdiet", "block_verify"]


def test_an_unknown_lane_raises_rather_than_disabling_nothing() -> None:
    """A typo that silently left the lane ON would measure the same arm twice."""

    with pytest.raises(ValueError) as excinfo:
        parse_disable_lanes("opdeit")

    message = str(excinfo.value)
    assert "opdeit" in message
    assert "opdiet" in message  # the real names are listed


def test_an_empty_disable_list_disables_nothing() -> None:
    assert parse_disable_lanes("") == frozenset()
    assert parse_disable_lanes(None) == frozenset()
    assert parse_disable_lanes(" , ") == frozenset()


# ---------------------------------------------------------------------------
# (3) who set what: the verdict line has to say
# ---------------------------------------------------------------------------


def test_value_source_says_nothing_in_a_process_that_armed_no_defaults() -> None:
    env = {"MTPLX_FABLE_OPDIET": "1"}

    assert fse.value_source("MTPLX_FABLE_OPDIET", env) == ""


def test_value_source_separates_the_default_from_the_operator() -> None:
    fse.record_defaults_applied({"MTPLX_FABLE_OPDIET": "1"})
    env = {"MTPLX_FABLE_OPDIET": "1", "MTPLX_FABLE_QSA_SPARSE_DECODE": "0"}

    assert fse.value_source("MTPLX_FABLE_OPDIET", env) == fse.SOURCE_DEFAULT
    assert fse.value_source("MTPLX_FABLE_QSA_SPARSE_DECODE", env) == (
        fse.SOURCE_OPERATOR
    )
    assert fse.value_source("MTPLX_TOTALLY_UNRELATED", env) == ""


def test_the_install_verdict_names_the_operator(monkeypatch) -> None:
    """The operator-wins line: ``off (operator: MTPLX_...=0)``."""

    from mtplx import fable_install_receipts as receipts

    monkeypatch.setenv("MTPLX_FABLE_QSA_SPARSE_DECODE", "0")
    monkeypatch.setenv("MTPLX_FABLE_OPDIET", "1")
    fse.record_defaults_applied({"MTPLX_FABLE_OPDIET": "1"})

    assert receipts._env_note("MTPLX_FABLE_QSA_SPARSE_DECODE") == (
        "operator: MTPLX_FABLE_QSA_SPARSE_DECODE=0"
    )
    assert receipts._env_note("MTPLX_FABLE_OPDIET") == (
        "default: MTPLX_FABLE_OPDIET=1"
    )


def test_the_verdict_spelling_is_unchanged_where_no_defaults_were_armed(
    monkeypatch,
) -> None:
    """Every driver/test process keeps the receipt lines it always printed."""

    from mtplx import fable_install_receipts as receipts

    monkeypatch.setenv("MTPLX_FABLE_BLOCK_VERIFY", "1")
    monkeypatch.delenv("MTPLX_FABLE_OPDIET", raising=False)

    assert receipts._env_note("MTPLX_FABLE_BLOCK_VERIFY") == (
        "MTPLX_FABLE_BLOCK_VERIFY='1'"
    )
    assert receipts._env_note("MTPLX_FABLE_OPDIET") == "MTPLX_FABLE_OPDIET unset"


# ---------------------------------------------------------------------------
# (4) the /health report
# ---------------------------------------------------------------------------


def test_the_report_lists_the_defaults_and_the_operator_overrides() -> None:
    env = {
        "MTPLX_FABLE_QSA_SPARSE_DECODE": "0",
        "MTPLX_SESSION_BANK_MAX_BYTES": "16G",
        "MTPLX_FABLE_HC_M4": "1",
    }
    armed = fable_default_env(env, disabled_lanes=["opdiet"])
    fse.record_defaults_applied(armed)

    report = fable_defaults_report(
        env, disabled_lanes=["opdiet"], model_gate="mtp, qwen4_exp fixed-M4 pack"
    )

    assert report["model_gate"] == "mtp, qwen4_exp fixed-M4 pack"
    assert report["disabled_lanes"] == ["opdiet"]
    assert report["disabled_keys"] == ["MTPLX_FABLE_OPDIET"]
    assert "MTPLX_FABLE_OPDIET" not in report["armed_by_default"]
    assert "MTPLX_FABLE_BLOCK_VERIFY" in report["armed_by_default"]
    off = {row["key"]: row for row in report["operator_off"]}
    assert set(off) == {"MTPLX_FABLE_QSA_SPARSE_DECODE", "MTPLX_SESSION_BANK_MAX_BYTES"}
    assert off["MTPLX_SESSION_BANK_MAX_BYTES"]["value"] == "16G"
    assert off["MTPLX_FABLE_QSA_SPARSE_DECODE"]["lane"] == "qsa_sparse_decode"
    pinned = {row["key"] for row in report["operator_pinned"]}
    assert pinned == {"MTPLX_FABLE_HC_M4"}
    assert json.dumps(report)  # /health has to be able to serialize it


def test_the_summary_line_names_the_gate_the_lanes_and_the_overrides() -> None:
    env = {"MTPLX_FABLE_QSA_SPARSE_DECODE": "0"}
    fse.record_defaults_applied(fable_default_env(env, disabled_lanes=["opdiet"]))

    line = fse.defaults_summary_line(
        env, disabled_lanes=["opdiet"], model_gate="mtp, qwen4_exp fixed-M4 pack"
    )

    assert "24/26 retained-stack keys armed by default" in line
    assert "(mtp, qwen4_exp fixed-M4 pack)" in line
    assert "lanes off: opdiet" in line
    assert "operator off: MTPLX_FABLE_QSA_SPARSE_DECODE=0" in line


# ---------------------------------------------------------------------------
# (5) the model gate
# ---------------------------------------------------------------------------


def _model_dir(tmp_path: Path, config: dict) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_the_gate_recognizes_a_flash_next_pack(tmp_path) -> None:
    assert is_flash_next_model_dir(
        str(_model_dir(tmp_path, {"model_type": "qwen4_exp"}))
    )


def test_the_gate_recognizes_a_flash_next_pack_by_text_config(tmp_path) -> None:
    assert is_flash_next_model_dir(
        str(_model_dir(tmp_path, {"text_config": {"model_type": "qwen4_exp_text"}}))
    )


@pytest.mark.parametrize(
    "config", [{"model_type": "qwen3_moe"}, {"model_type": "gemma4"}, {}]
)
def test_another_model_family_sees_no_change(tmp_path, config) -> None:
    assert not is_flash_next_model_dir(str(_model_dir(tmp_path, config)))


def test_the_gate_is_total(tmp_path) -> None:
    """No path, no file, unreadable JSON -- False, never an exception."""

    assert not is_flash_next_model_dir(None)
    assert not is_flash_next_model_dir("")
    assert not is_flash_next_model_dir(str(tmp_path / "nope"))
    (tmp_path / "config.json").write_text("{not json")
    assert not is_flash_next_model_dir(str(tmp_path))


# ---------------------------------------------------------------------------
# (6) the pre-import stamp: eleven keys their readers freeze at import
# ---------------------------------------------------------------------------


def test_the_import_bound_subset_is_the_eleven_the_readers_freeze() -> None:
    # graph_build_overlap.enabled() is lru_cached and its first read precedes
    # the server's default stamp on a real boot (the depth knob then fired
    # its consistency guard), so the pair is stamped at import like the rest.
    assert set(IMPORT_TIME_PROFILE_ENV) == {
        "MTPLX_FABLE_BLOCK_VERIFY",
        "MTPLX_FABLE_DRAFT_K20_PRESCATTER",
        "MTPLX_FABLE_GRAPH_BUILD_OVERLAP",
        "MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS",
        "MTPLX_FABLE_HC_M4",
        "MTPLX_FABLE_OPDIET",
        "MTPLX_FABLE_QSA_SPARSE_DECODE",
        "MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS",
        "MTPLX_FABLE_QSA_SPARSE_DECODE_TILE",
        "MTPLX_FABLE_VERIFY_GLUE",
        "MTPLX_FABLE_VERIFY_GLUE_ITEMS",
    }
    assert set(IMPORT_TIME_PROFILE_ENV) < set(FULL_STACK_PROFILE_ENV)


def test_the_stamp_arms_only_the_import_bound_subset(tmp_path) -> None:
    model = _model_dir(tmp_path, {"model_type": "qwen4_exp"})
    environ: dict[str, str] = {}

    stamped = fse.stamp_import_time_defaults(["-m", "--model", str(model)], environ)

    assert stamped == IMPORT_TIME_PROFILE_ENV
    assert environ == dict(IMPORT_TIME_PROFILE_ENV)


def test_the_stamp_does_nothing_for_another_model_family(tmp_path) -> None:
    model = _model_dir(tmp_path, {"model_type": "qwen3_moe"})
    environ: dict[str, str] = {}

    assert fse.stamp_import_time_defaults(["-m", "--model", str(model)], environ) == {}
    assert environ == {}


def test_the_stamp_fires_on_an_explicit_full_stack_profile() -> None:
    environ: dict[str, str] = {}

    stamped = fse.stamp_import_time_defaults(
        ["-m", "--profile", "turbo-full-stack"], environ
    )

    assert stamped == IMPORT_TIME_PROFILE_ENV


def test_the_stamp_yields_to_an_operator_export(tmp_path) -> None:
    model = _model_dir(tmp_path, {"model_type": "qwen4_exp"})
    environ = {"MTPLX_FABLE_OPDIET": "0"}

    stamped = fse.stamp_import_time_defaults(["-m", "--model", str(model)], environ)

    assert "MTPLX_FABLE_OPDIET" not in stamped
    assert environ["MTPLX_FABLE_OPDIET"] == "0"


def test_the_stamp_honours_a_disabled_lane(tmp_path) -> None:
    model = _model_dir(tmp_path, {"model_type": "qwen4_exp"})
    environ = {DISABLE_ENV: "qsa_sparse_decode"}

    stamped = fse.stamp_import_time_defaults(["-m", "--model", str(model)], environ)

    assert not set(stamped) & set(LANE_KEYS["qsa_sparse_decode"])
    assert "MTPLX_FABLE_OPDIET" in stamped


def test_the_stamp_has_an_off_switch(tmp_path) -> None:
    model = _model_dir(tmp_path, {"model_type": "qwen4_exp"})
    environ = {fse.EARLY_STAMP_ENV: "0"}

    assert fse.stamp_import_time_defaults(["-m", "--model", str(model)], environ) == {}


def test_the_stamp_never_raises(tmp_path) -> None:
    model = _model_dir(tmp_path, {"model_type": "qwen4_exp"})
    environ = {DISABLE_ENV: "not-a-lane"}

    # An unparseable lane list is a loud failure at _load, never a broken
    # import: this hook runs inside mtplx/server/__init__.py.
    assert fse.stamp_import_time_defaults(["-m", "--model", str(model)], environ) == {}


def test_the_stamp_lands_before_the_readers_freeze(tmp_path) -> None:
    """The whole reason this hook exists, proved end to end in a subprocess.

    ``mtplx.runtime_options`` freezes MTPLX_FABLE_OPDIET and friends in
    module constants at ITS import, and ``mtplx.server.openai``'s import
    block pulls it -- so the server's own setdefault at ``_load`` cannot arm
    them. ``import mtplx.server`` (the package, which Python executes first
    under ``python -m mtplx.server.openai``) has to have done it already.
    """

    model = _model_dir(tmp_path, {"model_type": "qwen4_exp"})
    program = (
        "import sys, json;"
        f"sys.argv = ['-m', '--model', {str(model)!r}];"
        "import mtplx.server;"
        "assert 'mtplx.runtime_options' not in sys.modules;"
        "import mtplx.runtime_options as ro;"
        "print(json.dumps({"
        "'opdiet': ro.fable_opdiet_enabled(),"
        "'hc_m4': ro.fable_hc_m4_enabled(),"
        "'glue': ro.fable_verify_glue_enabled(),"
        "'sparse': ro.fable_qsa_sparse_decode_enabled(),"
        "'splits': ro.fable_qsa_sparse_decode_splits(),"
        "'stamped': sorted(mtplx.server.EARLY_STAMPED_ENV),"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)},
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["opdiet"] is True
    assert observed["hc_m4"] is True
    assert observed["glue"] is True
    assert observed["sparse"] is True
    assert observed["splits"] == 17
    assert observed["stamped"] == sorted(IMPORT_TIME_PROFILE_ENV)


def test_without_the_hook_those_readers_stay_off(tmp_path) -> None:
    """The control for the test above: the same env set one import too late."""

    program = (
        "import os, json, mtplx.runtime_options as ro;"
        "os.environ['MTPLX_FABLE_OPDIET'] = '1';"
        "os.environ['MTPLX_FABLE_HC_M4'] = '1';"
        "print(json.dumps({"
        "'opdiet': ro.fable_opdiet_enabled(),"
        "'hc_m4': ro.fable_hc_m4_enabled()}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)},
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "opdiet": False,
        "hc_m4": False,
    }


# ---------------------------------------------------------------------------
# (7) the server wiring, checked at source level (importing it pulls MLX)
# ---------------------------------------------------------------------------


def test_the_defaults_are_armed_inside_the_family_predicate() -> None:
    gate = SERVER_SOURCE.index("    if _served_model_type_is_qwen4_exp(args):")
    end = SERVER_SOURCE.index("def _disabled_optimization_lanes(", gate)
    block = SERVER_SOURCE[gate:end]

    assert "retained_defaults = fable_default_env(" in block
    assert "overrides.setdefault(key, value)" in block
    assert "record_defaults_applied(retained_defaults)" in block


def test_the_defaults_use_setdefault_so_a_model_contract_still_wins() -> None:
    """``overrides`` starts as the model pack's runtime contract.

    setdefault (never ``[]=``) is what keeps a pack's own value on top,
    exactly as the sixteen M4 keys above already behave.
    """

    index = SERVER_SOURCE.index("retained_defaults = fable_default_env(")
    tail = SERVER_SOURCE[index : index + 400]

    assert "overrides.setdefault(key, value)" in tail
    assert "overrides[key] = value" not in tail


def test_the_server_exposes_the_disable_flag() -> None:
    assert '"--disable-optimization",' in SERVER_SOURCE
    assert 'dest="disable_optimization",' in SERVER_SOURCE
    assert 'action="append",' in SERVER_SOURCE
    assert 'choices=(*LANES, *qwen4_aux_lanes.LANES, "all"),' in SERVER_SOURCE


def test_the_serve_wrapper_forwards_the_flag() -> None:
    """`mtplx serve` launches a separate process; a lane has to reach argv."""

    cli = (REPO_ROOT / "mtplx" / "cli.py").read_text()
    public = (REPO_ROOT / "mtplx" / "commands" / "public.py").read_text()

    assert '"--disable-optimization",' in cli
    assert 'cmd.extend(["--disable-optimization", str(lane)])' in public


def test_health_publishes_the_defaults_report() -> None:
    assert 'payload["fable_defaults"] = fable_defaults_report(' in SERVER_SOURCE


def test_the_selfcheck_runs_on_the_family_not_only_the_opt_in_profile() -> None:
    """The stack is the default now, so the receipt cannot be profile-gated."""

    assert "def _full_stack_selfcheck_applies(" in SERVER_SOURCE
    index = SERVER_SOURCE.index("def _full_stack_selfcheck_applies(")
    body = SERVER_SOURCE[index : index + 700]
    assert "_full_stack_profile_selected(args)" in body
    assert "_served_model_type_is_qwen4_exp(args)" in body
    assert "if not _full_stack_selfcheck_applies(args):" in SERVER_SOURCE


# ---------------------------------------------------------------------------
# (8) every key defaulted ON proves itself at install
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(FULL_STACK_PROFILE_ENV))
def test_every_defaulted_key_has_an_install_receipt(key: str) -> None:
    """An arm that cannot prove it ran is unreadable -- and now it is a DEFAULT.

    Two acceptable proofs: an ``mtplx.fable_install_receipts`` lane (the
    per-flag verdict lines), or a named receipt the owning lane already
    prints. A key with neither would be armed on every Flash-Next boot with
    nothing in the log either way.
    """

    from mtplx.fable_install_receipts import REGISTERED_KEYS as RECEIPT_KEYS

    entry = fse.spec(key)
    assert entry is not None
    if key in RECEIPT_KEYS:
        assert not entry.receipt, (
            f"{key} has a lane AND a hand-named receipt; name one"
        )
        return
    assert entry.receipt.strip(), key
    assert "/" in entry.receipt or ":" in entry.receipt, key


def _reset_first_read_caches() -> None:
    """Two of the three defaulted-on lanes cache on FIRST read, not per call.

    graph_build_overlap uses lru_cache and qwen4_m4_stage3 a lazy global, so
    any earlier test in the session can have fixed them before this one sets
    the env. Clearing is a test-only affordance; the runtime never does it,
    which is exactly why the pre-import stamp exists.
    """

    from mtplx import graph_build_overlap, qwen4_m4_stage3

    graph_build_overlap.enabled.cache_clear()
    graph_build_overlap.layers.cache_clear()
    graph_build_overlap.items.cache_clear()
    qwen4_m4_stage3._ROUTE_KERNEL_CACHE = None
    qwen4_m4_stage3._ROUTE_KERNEL_VEC_LANES_CACHE = None


def test_the_three_w93_lanes_render_a_verdict(monkeypatch) -> None:
    """route_kernel, graph_build_overlap and gdn_blocked_prefill had none."""

    from mtplx import fable_install_receipts as receipts

    _reset_first_read_caches()
    for key, value in (
        ("MTPLX_FABLE_ROUTE_KERNEL", "1"),
        ("MTPLX_FABLE_GRAPH_BUILD_OVERLAP", "1"),
        ("MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS", "3"),
        ("MTPLX_GDN_BLOCKED_PREFILL", "1"),
    ):
        monkeypatch.setenv(key, value)

    try:
        for lane in ("route_kernel", "graph_build_overlap", "gdn_blocked_prefill"):
            line = receipts.verdict(lane, {}).line
            assert line.startswith(f"[fable] {lane} armed"), line
    finally:
        _reset_first_read_caches()


def test_an_operator_off_renders_as_off_with_the_operator_named(monkeypatch) -> None:
    from mtplx import fable_install_receipts as receipts

    monkeypatch.setenv("MTPLX_GDN_BLOCKED_PREFILL", "0")
    fse.record_defaults_applied({"MTPLX_FABLE_OPDIET": "1"})

    assert receipts.verdict("gdn_blocked_prefill", {}).line == (
        "[fable] gdn_blocked_prefill off "
        "(operator: MTPLX_GDN_BLOCKED_PREFILL=0)"
    )


def test_an_unparseable_overlap_spelling_is_reported_not_raised(monkeypatch) -> None:
    """The reader raises; the receipt must say so instead of exploding."""

    from mtplx import fable_install_receipts as receipts

    _reset_first_read_caches()
    monkeypatch.setenv("MTPLX_FABLE_GRAPH_BUILD_OVERLAP", "maybe")
    try:
        line = receipts.verdict("graph_build_overlap", {}).line
    finally:
        _reset_first_read_caches()

    assert line.startswith("[fable] graph_build_overlap refused (")
    assert "maybe" in line


# ---------------------------------------------------------------------------
# (9) the benchmark arm and the defaults cannot drift apart
# ---------------------------------------------------------------------------
#
# The whole point of defaulting the stack on is that `mtplx serve` reproduces
# the arm the battery measured. That claim is only as good as its weakest
# key: three MTPLX_QWEN4_M4_ROUTED_* keys were armed on every measured branch
# arm out of the benchmark harness's own base environment -- not out of
# either committed flag file -- and nothing in mtplx set them, so a plain
# serve ran a different MoE tail than the receipts describe.
#
# The keys below are the harness's own controls: what it arms for its own
# reasons and a serve should NOT reproduce. Each exclusion is receipt-backed,
# and the tests refuse an entry that the defaults arm anyway.

#: Keys the benchmark harness sets on a branch arm that are its OWN controls,
#: not part of the stack a serve should reproduce. Each one is excluded for a
#: receipt-backed reason, and ``test_the_harness_exclusions_are_minimal``
#: refuses an entry that is in the defaults anyway.
HARNESS_CONTROL_KEYS: dict[str, str] = {
    # A guard RELAXATION, not an optimization. The harness needs it because
    # it pins MTPLX_QSA_PREFILL_COMPILE_ROWS=4096 while its warm-up ladder
    # clamps a chunk to a width the geometry guard then refused; the harness
    # itself files it under BRANCH_BOOT_FABLE_KEYS ("a boot requirement, not
    # retained tuning"). At the defaults the guard is never reached at all --
    # see test_the_defaults_never_reach_the_geometry_waiver -- and defaulting
    # it ON would permanently disarm the only signal that a retuned chunk
    # width has silently dropped every full chunk onto the eager selector.
    "MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH": "boot guard waiver",
    # In the harness's hard-coded DEFAULT_FABLE_FLAGS, but NOT armed in the
    # measured arm: every battery receipt's fable_flags block records it
    # under "dropped_vs_default", because the run passed --fable-flags-file
    # stack.flags, which does not carry it. mtplx/fable_compiled_draft.py's
    # own docstring says it is "not a member of" the retained stack.
    "MTPLX_FABLE_COMPILED_DRAFT": "measured arm dropped it (dropped_vs_default)",
}












def test_the_harness_exclusions_are_minimal() -> None:
    """No key is excluded that the defaults arm anyway (that would hide a gap)."""

    for key in HARNESS_CONTROL_KEYS:
        assert key not in FULL_STACK_PROFILE_ENV, key


def test_the_defaults_never_reach_the_geometry_waiver() -> None:
    """Why MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH stays out.

    ``assert_prefill_chunk_coherent`` consults the waiver only after two
    earlier returns, and the defaults land on one of them either way:
    a full 4,096-wide chunk equals the compiled row width, and the warm-up
    ladder's narrower chunks are not configured full widths. The waiver is
    reachable only when an operator moves one of the pair without the other,
    which is the misconfiguration the raise exists to report.
    """

    from mtplx.fable_prefill_chunk import (
        COHERENCE_COMPILED,
        COHERENCE_NARROW_EAGER,
        assert_prefill_chunk_coherent,
        configured_full_chunk_widths,
    )

    defaults = dict(FULL_STACK_PROFILE_ENV)
    assert "MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH" not in defaults
    assert configured_full_chunk_widths(defaults) == frozenset({4096})

    assert assert_prefill_chunk_coherent(4096, defaults) == COHERENCE_COMPILED
    for width in (256, 512, 2048, 2560):
        assert assert_prefill_chunk_coherent(width, defaults) == (
            COHERENCE_NARROW_EAGER
        ), width


def test_moving_one_of_the_pair_still_raises_at_the_defaults() -> None:
    """The guard the waiver would have disarmed is still armed and still loud."""

    from mtplx.fable_prefill_chunk import (
        PrefillChunkGeometryError,
        assert_prefill_chunk_coherent,
    )

    retuned = {**FULL_STACK_PROFILE_ENV, "MTPLX_PREFILL_CHUNK_SIZE": "8192"}

    with pytest.raises(PrefillChunkGeometryError) as excinfo:
        assert_prefill_chunk_coherent(8192, retuned)

    assert "MTPLX_QSA_PREFILL_COMPILE_ROWS=4096" in str(excinfo.value)


# ---------------------------------------------------------------------------
# (10) the M4 routed-GLU chain
# ---------------------------------------------------------------------------

ROUTED_CHAIN = (
    "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE",
    "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL",
    "MTPLX_QWEN4_M4_ROUTED_GLU",
)


@pytest.mark.parametrize("key", ROUTED_CHAIN)
def test_the_routed_chain_is_armed_by_default(key: str) -> None:
    armed = fable_default_env({})

    assert armed[key] == "1"


@pytest.mark.parametrize("key", ROUTED_CHAIN)
def test_a_routed_key_yields_to_an_operator_export(key: str) -> None:
    assert key not in fable_default_env({key: "0"})


def test_the_routed_chain_shares_the_route_kernel_lane() -> None:
    """One lane, because the validator refuses any half of the chain.

    reduce -> residual tail -> GLU -> route kernel, each member validating
    the one before it. ``--disable-optimization route_kernel`` therefore
    returns all four keys to their shipped defaults together, which is the
    only lane spelling that still boots.
    """

    assert set(LANE_KEYS["route_kernel"]) == {"MTPLX_FABLE_ROUTE_KERNEL", *ROUTED_CHAIN}
    for key in ROUTED_CHAIN:
        assert KEY_LANE[key] == "route_kernel"

    armed = fable_default_env({}, disabled_lanes=["route_kernel"])

    assert not set(armed) & set(LANE_KEYS["route_kernel"])

    # ... and the all-off shape the lane switch produces still installs:
    # stage 3 stays server-armed, and it is the CHILD routes the validator
    # constrains, so turning the whole tail off is a legal configuration.
    from mtplx.qwen4_m4_stage3 import _validate_feature_combination

    _validate_feature_combination(
        routed_down_reduce_enabled=False,
        routed_down_residual_tail_enabled=False,
        routed_glu_enabled=False,
        route_kernel_enabled=False,
    )


def test_the_defaults_satisfy_the_stage3_validator() -> None:
    """The bug this closes: the defaults used to raise at model install.

    ``MTPLX_FABLE_ROUTE_KERNEL`` was already defaulted ON while the routed
    chain was not, and ``mtplx/runtime.py`` calls
    ``qwen4_m4_stage3_flags()`` unguarded -- so a stock serve against a
    Flash-Next pack could not finish loading the model.
    """

    from mtplx.qwen4_m4_stage3 import _validate_feature_combination

    armed = fable_default_env({})
    on = {key: armed.get(key) == "1" for key in ROUTED_CHAIN}

    # Nothing raises with the whole chain armed alongside the route kernel.
    _validate_feature_combination(
        routed_down_reduce_enabled=on["MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE"],
        routed_down_residual_tail_enabled=on[
            "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL"
        ],
        routed_glu_enabled=on["MTPLX_QWEN4_M4_ROUTED_GLU"],
        route_kernel_enabled=armed.get("MTPLX_FABLE_ROUTE_KERNEL") == "1",
    )

    # ... and the earlier shape is what it refuses.
    with pytest.raises(ValueError, match="MTPLX_QWEN4_M4_ROUTED_GLU"):
        _validate_feature_combination(
            routed_down_reduce_enabled=False,
            routed_down_residual_tail_enabled=False,
            routed_glu_enabled=False,
            route_kernel_enabled=True,
        )


@pytest.mark.parametrize("key", ROUTED_CHAIN)
def test_each_routed_key_names_its_install_receipt(key: str) -> None:
    """No install-receipt lane, so the registry must name where it prints.

    ``install_qwen4_m4_stage3`` reports each of the three as its own field,
    so "armed" is provable per key rather than for the group.
    """

    entry = fse.spec(key)

    assert entry is not None
    assert "install_qwen4_m4_stage3" in entry.receipt
    assert "engagement_reports.qwen4_m4_stage3" in entry.receipt


def test_the_stage3_report_carries_a_field_per_routed_key() -> None:
    """The receipt string above has to name a field that actually exists."""

    from mtplx.qwen4_m4_stage3 import _installation_report

    report = _installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
    )

    assert report["routed_down_reduce"] is True
    assert report["routed_down_residual_tail"] is True
    assert report["paired_routed_glu"] is True
    assert report["boundary"] == (
        "paired_routed_q4g32_glu_reduce_shared_add_mlp_residual"
    )


def test_the_routed_keys_are_not_import_bound() -> None:
    """They are read at model install, so no pre-import stamp is needed."""

    for key in ROUTED_CHAIN:
        assert key not in IMPORT_TIME_PROFILE_ENV
        assert fse.spec(key).binds_at == fse.BIND_CALL


# ---------------------------------------------------------------------------
# (11) a defaulted key must survive the boot-time runtime-env validator
# ---------------------------------------------------------------------------
#
# `_server_runtime_env_overrides` returns the defaults as MODEL RUNTIME
# OVERRIDES, and mtplx/server/openai.py hands that dict to
# `apply_profile_env(runtime_env_overrides=...)`, whose first statement is
# `normalize_runtime_env_overrides` -- which RAISES on any key outside
# `MODEL_RUNTIME_ENV_OVERRIDE_KEYS`. So a defaulted key missing from that
# allowlist is not merely un-pinnable by a model pack: it makes the boot
# validator raise on the server's OWN overrides, inside _load, before a
# weight is read. Eighteen of the twenty-six were missing until 2026-09-03 --
# the fifteen MTPLX_FABLE_* retained keys, both MTPLX_FRSPEC_* and
# MTPLX_SESSION_BANK_MAX_BYTES.


@pytest.mark.parametrize("key", sorted(FULL_STACK_PROFILE_ENV))
def test_every_defaulted_key_is_a_supported_runtime_override(key: str) -> None:
    from mtplx.profiles import MODEL_RUNTIME_ENV_OVERRIDE_KEYS

    assert key in MODEL_RUNTIME_ENV_OVERRIDE_KEYS, (
        f"{key} is armed by default but a model pack cannot pin it, and "
        "normalize_runtime_env_overrides would reject the server's own "
        "overrides at boot"
    )


def test_the_whole_default_set_survives_the_boot_validator() -> None:
    """The class-level statement: normalize round-trips the defaults intact."""

    from mtplx.profiles import normalize_runtime_env_overrides

    defaults = dict(FULL_STACK_PROFILE_ENV)

    assert normalize_runtime_env_overrides(defaults) == defaults


def test_the_real_boot_call_applies_the_defaults_without_raising() -> None:
    """The path that actually failed: apply_profile_env, not normalize alone.

    ``mtplx/server/openai.py:_load`` calls
    ``apply_profile_env(profile, runtime_env_overrides=<the server dict>)``,
    and the defaults are IN that dict. Before this change the call raised
    ``ValueError: runtime_env_overrides has unsupported key: ...`` on the
    first defaulted key the allowlist did not carry.
    """

    from mtplx.profiles import apply_profile_env

    environ: dict[str, str] = {}
    previous = apply_profile_env(
        "turbo", environ=environ, runtime_env_overrides=dict(FULL_STACK_PROFILE_ENV)
    )

    assert previous is not None
    for key, value in FULL_STACK_PROFILE_ENV.items():
        assert environ[key] == value, key


def test_the_allowlist_spreads_the_stack_rather_than_listing_it() -> None:
    """Structural, not incidental: a new stack key cannot reopen the gap.

    Hand-listing the twenty-six would satisfy the subset test above and then
    rot the next time a key joins the stack. The allowlist spreads
    ``_FULL_STACK_PROFILE_KEYS`` (which is ``tuple(FULL_STACK_PROFILE_ENV)``)
    so membership is derived, exactly as ``PROFILE_ENV_USER_OVERRIDE_KEYS``
    already does it.
    """

    source = (REPO_ROOT / "mtplx" / "profiles.py").read_text()
    block = source.split("MODEL_RUNTIME_ENV_OVERRIDE_KEYS", 1)[1].split("\n)\n", 1)[0]

    assert "*_FULL_STACK_PROFILE_KEYS," in block
    assert "_FULL_STACK_PROFILE_KEYS = tuple(FULL_STACK_PROFILE_ENV)" in source


def test_both_override_channels_accept_every_defaulted_key() -> None:
    """A pack contract and an operator export are different allowlists.

    ``MODEL_RUNTIME_ENV_OVERRIDE_KEYS`` is what a pack's
    ``runtime_env_overrides`` may name; ``PROFILE_ENV_USER_OVERRIDE_KEYS`` is
    what beats a profile stamp. A default the operator cannot beat is not a
    default, it is a pin -- so both have to cover the whole set.
    """

    from mtplx.profiles import (
        MODEL_RUNTIME_ENV_OVERRIDE_KEYS,
        PROFILE_ENV_USER_OVERRIDE_KEYS,
    )

    keys = set(FULL_STACK_PROFILE_ENV)

    assert keys <= set(MODEL_RUNTIME_ENV_OVERRIDE_KEYS)
    assert keys <= set(PROFILE_ENV_USER_OVERRIDE_KEYS)
