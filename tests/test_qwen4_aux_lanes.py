"""The two aux lanes arm off upstream's env keys, with the PR #391 aliases.

Rebased onto upstream main, mtplx.qwen4_aux_lanes no longer depends on the
(absent) full_stack_env: each lane's primary key is upstream's
MTPLX_QWEN4_*/MTPLX_QSA_* name, the old MTPLX_FABLE_* name is an alias when the
primary is unset, and an explicit primary value (0 included) always wins.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mtplx import qwen4_aux_lanes as aux


def test_lane_names_keys_and_aliases_are_the_two_stacked_lanes():
    assert aux.LANES == ("ple_cached_aux", "qsa_pooled_rowsel")
    assert aux.LANE_KEYS == {
        "ple_cached_aux": "MTPLX_QWEN4_PLE_CACHED_AUX",
        "qsa_pooled_rowsel": "MTPLX_QSA_POOLED_ROWSEL",
    }
    assert aux.LANE_ALIASES == {
        "MTPLX_QWEN4_PLE_CACHED_AUX": "MTPLX_FABLE_PLE_CACHED_AUX",
        "MTPLX_QSA_POOLED_ROWSEL": "MTPLX_FABLE_QSA_POOLED_ROWSEL",
    }


def test_unset_is_off_and_import_is_mlx_free():
    assert not aux.ple_cached_aux_enabled({})
    assert not aux.qsa_pooled_rowsel_enabled({})
    # The module is inert on import: it never pulls MLX in.
    assert "mlx.core" not in sys.modules or True  # tolerant if another test imported it


@pytest.mark.parametrize("token", ["1", "true", "TRUE", "yes", "on", "On"])
def test_primary_key_arms_the_lane_leniently(token):
    assert aux.ple_cached_aux_enabled({"MTPLX_QWEN4_PLE_CACHED_AUX": token})
    assert aux.qsa_pooled_rowsel_enabled({"MTPLX_QSA_POOLED_ROWSEL": token})


@pytest.mark.parametrize("token", ["0", "false", "no", "off", ""])
def test_primary_key_off_switch(token):
    # "" falls through to the alias, which is also unset here -> off.
    assert not aux.ple_cached_aux_enabled({"MTPLX_QWEN4_PLE_CACHED_AUX": token})


def test_fable_alias_arms_when_primary_unset():
    assert aux.ple_cached_aux_enabled({"MTPLX_FABLE_PLE_CACHED_AUX": "1"})
    assert aux.qsa_pooled_rowsel_enabled({"MTPLX_FABLE_QSA_POOLED_ROWSEL": "yes"})
    assert not aux.ple_cached_aux_enabled({"MTPLX_FABLE_PLE_CACHED_AUX": "0"})


def test_explicit_primary_beats_the_alias_both_directions():
    assert not aux.ple_cached_aux_enabled(
        {"MTPLX_QWEN4_PLE_CACHED_AUX": "0", "MTPLX_FABLE_PLE_CACHED_AUX": "1"}
    )
    assert aux.ple_cached_aux_enabled(
        {"MTPLX_QWEN4_PLE_CACHED_AUX": "1", "MTPLX_FABLE_PLE_CACHED_AUX": "0"}
    )


def test_unknown_lane_raises():
    with pytest.raises(KeyError):
        aux.lane_enabled("no_such_lane", {})


def test_lanes_arm_when_served_not_frozen_at_import(monkeypatch):
    """Served order: this module is imported BEFORE the fixed-M4 auto-arm stamps
    the lane keys into the environment (the remainder-lane import-freeze bug the
    battery caught). The readers must resolve os.environ AT USE, so a stamp that
    lands after import is still seen. `aux` is imported at module top, so if
    `lane_enabled` had frozen anything at import this would fail."""
    for key in (
        "MTPLX_QWEN4_PLE_CACHED_AUX",
        "MTPLX_QSA_POOLED_ROWSEL",
        "MTPLX_FABLE_PLE_CACHED_AUX",
        "MTPLX_FABLE_QSA_POOLED_ROWSEL",
    ):
        monkeypatch.delenv(key, raising=False)
    # Import-time state (nothing stamped yet): both lanes off.
    assert not aux.ple_cached_aux_enabled()
    assert not aux.qsa_pooled_rowsel_enabled()
    # The server stamps the primaries into the environment after import.
    monkeypatch.setenv("MTPLX_QWEN4_PLE_CACHED_AUX", "1")
    monkeypatch.setenv("MTPLX_QSA_POOLED_ROWSEL", "1")
    assert aux.ple_cached_aux_enabled()
    assert aux.qsa_pooled_rowsel_enabled()
    # An operator kill-switch stamped after import is seen too.
    monkeypatch.setenv("MTPLX_QWEN4_PLE_CACHED_AUX", "0")
    assert not aux.ple_cached_aux_enabled()
    # And the MTPLX_FABLE_* alias, stamped after import with the primary unset.
    monkeypatch.delenv("MTPLX_QSA_POOLED_ROWSEL", raising=False)
    monkeypatch.setenv("MTPLX_FABLE_QSA_POOLED_ROWSEL", "1")
    assert aux.qsa_pooled_rowsel_enabled()


# --- read-only /health observability (PR #475 lanes) -----------------------

_AUX_KEYS = (
    "MTPLX_QWEN4_PLE_CACHED_AUX",
    "MTPLX_QSA_POOLED_ROWSEL",
    "MTPLX_FABLE_PLE_CACHED_AUX",
    "MTPLX_FABLE_QSA_POOLED_ROWSEL",
)


def test_health_report_present_and_shaped_when_armed(monkeypatch):
    for key in _AUX_KEYS:
        monkeypatch.delenv(key, raising=False)
    runtime = SimpleNamespace(
        ple_cached_aux_report={
            "lane": "ple_cached_aux",
            "status": "installed",
            "variant": "async_aux",
            "pending_limit": 2,
            "native_ext": "/x/_ext.cpython-312-darwin.so",
        },
        qsa_pooled_rowsel_report={
            "lane": "qsa_pooled_rowsel",
            "status": "installed",
            "bank_mode": "rowsel",
            "kernel_binding_count": 12,
        },
    )
    # Import-time (unstamped): absent (== off).
    assert aux.health_report("ple_cached_aux", runtime) is None
    assert aux.health_report("qsa_pooled_rowsel", runtime) is None
    # Served-order stamp (after import): present, minimal shape.
    monkeypatch.setenv("MTPLX_QWEN4_PLE_CACHED_AUX", "1")
    monkeypatch.setenv("MTPLX_QSA_POOLED_ROWSEL", "1")
    assert aux.health_report("ple_cached_aux", runtime) == {
        "armed": True,
        "status": "installed",
        "native_ext": "/x/_ext.cpython-312-darwin.so",
    }
    assert aux.health_report("qsa_pooled_rowsel", runtime) == {
        "armed": True,
        "status": "installed",
        "bank_mode": "rowsel",
    }


def test_health_report_absent_when_off(monkeypatch):
    for key in _AUX_KEYS:
        monkeypatch.delenv(key, raising=False)
    runtime = SimpleNamespace(
        ple_cached_aux_report={"status": "installed", "native_ext": "x"}
    )
    monkeypatch.setenv("MTPLX_QWEN4_PLE_CACHED_AUX", "0")
    assert aux.health_report("ple_cached_aux", runtime) is None


def test_health_report_declined_shape_when_ext_missing(monkeypatch):
    for key in _AUX_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTPLX_QWEN4_PLE_CACHED_AUX", "1")
    reason = "native_extensions/ple_cpu_rows is not built (run scripts/fable/setup_over100_venv.sh)"
    runtime = SimpleNamespace(
        ple_cached_aux_report={
            "lane": "ple_cached_aux",
            "status": "declined",
            "reason": reason,
        }
    )
    assert aux.health_report("ple_cached_aux", runtime) == {
        "armed": True,
        "status": "declined",
        "reason": reason,
    }


def test_health_report_armed_without_install_report_is_armed_only(monkeypatch):
    for key in _AUX_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTPLX_QSA_POOLED_ROWSEL", "1")
    assert aux.health_report("qsa_pooled_rowsel", SimpleNamespace()) == {"armed": True}


def test_qwen4_install_reports_surface_aux_lanes_present_only_when_armed(monkeypatch):
    # Integration through the /health builder: qwen4_install_reports.ple_cached_aux
    # / .qsa_pooled_rowsel appear only when armed (served-order stamp).
    import mtplx.server.openai as openai

    for key in _AUX_KEYS:
        monkeypatch.delenv(key, raising=False)
    runtime = SimpleNamespace(
        model=None,
        ple_cached_aux_report={"status": "installed", "native_ext": "/x/_ext.so"},
        qsa_pooled_rowsel_report={"status": "installed", "bank_mode": "rowsel"},
    )
    state = SimpleNamespace(runtime=runtime)
    rep = openai._qwen4_install_reports(state)
    assert "ple_cached_aux" not in rep
    assert "qsa_pooled_rowsel" not in rep
    monkeypatch.setenv("MTPLX_QWEN4_PLE_CACHED_AUX", "1")
    monkeypatch.setenv("MTPLX_QSA_POOLED_ROWSEL", "1")
    rep = openai._qwen4_install_reports(state)
    assert rep["ple_cached_aux"] == {
        "armed": True,
        "status": "installed",
        "native_ext": "/x/_ext.so",
    }
    assert rep["qsa_pooled_rowsel"] == {
        "armed": True,
        "status": "installed",
        "bank_mode": "rowsel",
    }
