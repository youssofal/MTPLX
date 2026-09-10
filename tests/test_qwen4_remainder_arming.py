"""The PR 391 remainder lanes must arm when SERVED, not freeze at import.

This reproduces the exact ``mtplx serve`` order that hid two arming failures
the battery caught (2026-09-07): the reader modules are imported BEFORE the
fixed-M4 auto-arm stamps the lane keys into the environment, so an
import-time-frozen reader returns its default (off) forever -- the served lane
is then absent from /health with no verdict line. Every remainder-lane reader
must resolve the environment at USE (the install/route path, which runs after
the overrides are applied), so a stamp that lands after import is still seen.

Nothing here dispatches Metal; it drives the pure-Python auto-arm block and the
env readers.
"""

from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace

# Import the server + runtime modules FIRST, exactly as `mtplx serve` does, so
# any import-time reader cache would be populated with the (unset) defaults
# before the auto-arm runs. If a reader froze here, the asserts below fail.
import mtplx.runtime  # noqa: F401  (import-order fidelity)
import mtplx.server.openai as openai
from mtplx import runtime_options as ro
from mtplx.models import qwen4_exp
from mtplx.native import native_qsa_available
from mtplx.profiles import normalize_runtime_env_overrides
from mtplx.qwen4_prefill_chunk import resolve_query_tile_rows

_LANE_ENV_KEYS = (
    "MTPLX_QWEN4_HC_M4",
    "MTPLX_QWEN4_PREFILL_MASK_FUSE",
    "MTPLX_QSA_PREFILL_QUERY_TILE",
    "MTPLX_QSA_SPARSE_DECODE",
    "MTPLX_QSA_SPARSE_DECODE_TILE",
    "MTPLX_QSA_SPARSE_DECODE_SPLITS",
    "MTPLX_FABLE_HC_M4",
    "MTPLX_FABLE_PREFILL_MASK_FUSE",
    "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE",
    "MTPLX_FABLE_QSA_SPARSE_DECODE",
)


def test_remainder_lanes_arm_when_served_not_frozen_at_import(tmp_path, monkeypatch):
    # The state at a fresh import: reader globals unforced (None -> read env),
    # every lane key unset.
    for name in (
        "_QWEN4_HC_M4",
        "_QSA_SPARSE_DECODE",
        "_QSA_SPARSE_DECODE_TILE",
        "_QSA_SPARSE_DECODE_SPLITS",
    ):
        monkeypatch.setattr(ro, name, None)
    for key in _LANE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    # Before the auto-arm: every reader off, exactly the served import-time
    # state. (A reader frozen True/False at import would already fail here.)
    assert ro.qwen4_hc_m4_enabled() is False
    assert qwen4_exp._prefill_mask_fuse_enabled() is False
    assert resolve_query_tile_rows() == 0
    assert ro.qsa_sparse_decode_enabled() is False

    # Run the fixed-M4 auto-arm and APPLY it to the environment, as the server
    # does via apply_profile_env. Force the fixed-M4 predicate on rather than
    # crafting a full fixed-verify config; the remainder defaults gate on that
    # predicate (and, for the decode lane, the built native extension).
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )
    args = SimpleNamespace(
        generation_mode="mtp",
        verify_strategy="capture_commit",
        model=str(tmp_path),
    )
    monkeypatch.setattr(openai, "_served_model_is_qwen4_fixed_m4", lambda a: True)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        overrides = openai._server_runtime_env_overrides(args, {})
    log = err.getvalue()

    # Every stamped key survives the boot-time validator (the check that only
    # runs inside apply_profile_env), then apply them as the server would.
    assert normalize_runtime_env_overrides(overrides) == overrides
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    # The three unconditional remainder lanes are armed, and each reader --
    # read at USE, after the stamp was applied -- sees it. This is the class
    # that failed when served: an import-frozen reader returns its default.
    assert overrides.get("MTPLX_QWEN4_HC_M4") == "1"
    assert ro.qwen4_hc_m4_enabled() is True
    assert overrides.get("MTPLX_QWEN4_PREFILL_MASK_FUSE") == "1"
    assert qwen4_exp._prefill_mask_fuse_enabled() is True
    assert overrides.get("MTPLX_QSA_PREFILL_QUERY_TILE") == "2048"
    assert resolve_query_tile_rows() == 2048

    # The native-gated decode lane: armed (install attempted) when the
    # extension is built, else an explicit declined-to-stock verdict so a
    # wheel without it still serves -- never a silent absence.
    if native_qsa_available():
        assert overrides.get("MTPLX_QSA_SPARSE_DECODE") == "1"
        assert ro.qsa_sparse_decode_enabled() is True
        # its companions resolve to the measured geometry at use, too
        assert ro.qsa_sparse_decode_tile() == (128, 32)
        assert ro.qsa_sparse_decode_splits() == 17
    else:
        assert "MTPLX_QSA_SPARSE_DECODE" not in overrides
        assert "MTPLX_QSA_SPARSE_DECODE declined to stock" in log
        assert ro.qsa_sparse_decode_enabled() is False


def test_upstream_verify_lanes_read_at_use_not_frozen_at_import(monkeypatch):
    """The four upstream fixed-M4 verify lanes the arming audit found dead when
    served must resolve the environment at USE too.

    ``mtplx.generation`` was imported at module load above (before any stamp),
    so an import-frozen reader would return its default forever. Each lane's
    reader must see a stamp applied after that import -- the served order.
    """

    import mtplx.generation as generation
    import mtplx.qwen4_block_verify as block_verify
    import mtplx.qwen4_draft_k20_prescatter as prescatter

    cases = [
        # (set-unforced, reader, env key)
        (
            lambda: monkeypatch.setattr(ro, "_QWEN4_OPDIET", None),
            ro.qwen4_opdiet_enabled,
            "MTPLX_QWEN4_OPDIET",
        ),
        (
            lambda: monkeypatch.setattr(ro, "_QWEN4_VERIFY_GLUE", None),
            ro.qwen4_verify_glue_enabled,
            "MTPLX_QWEN4_VERIFY_GLUE",
        ),
        (
            lambda: monkeypatch.setattr(prescatter, "_ENABLED", None),
            generation._qwen4_draft_k20_prescatter_enabled,
            "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
        ),
        (
            lambda: monkeypatch.setattr(block_verify, "_ENABLED", None),
            generation._qwen4_block_verify_enabled,
            "MTPLX_QWEN4_BLOCK_VERIFY",
        ),
    ]
    for unforce, reader, env_key in cases:
        unforce()
        monkeypatch.delenv(env_key, raising=False)
        assert reader() is False, env_key
        monkeypatch.setenv(env_key, "1")
        assert reader() is True, env_key  # the stamp lands AFTER import -> seen
        monkeypatch.delenv(env_key, raising=False)


def test_the_fixed_m4_auto_arm_stamps_the_upstream_verify_lanes(tmp_path, monkeypatch):
    """The fixed-M4 auto-arm stamps OPDIET, BLOCK_VERIFY and VERIFY_GLUE, and
    the readers (read at use, after the stamp is applied) arm.

    (DRAFT_K20_PRESCATTER is stamped only on a q8/g64 lm_head pack, gated by the
    FR-Spec draft; its read-at-use is covered above.)
    """

    for name in ("_QWEN4_OPDIET", "_QWEN4_OPDIET_SELECTED", "_QWEN4_VERIFY_GLUE",
                 "_QWEN4_VERIFY_GLUE_SELECTED"):
        monkeypatch.setattr(ro, name, None)
    for key in ("MTPLX_QWEN4_OPDIET", "MTPLX_QWEN4_VERIFY_GLUE",
                "MTPLX_QWEN4_VERIFY_GLUE_ITEMS"):
        monkeypatch.delenv(key, raising=False)

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )
    args = SimpleNamespace(
        generation_mode="mtp", verify_strategy="capture_commit", model=str(tmp_path)
    )
    monkeypatch.setattr(openai, "_served_model_is_qwen4_fixed_m4", lambda a: True)
    overrides = openai._server_runtime_env_overrides(args, {})
    assert normalize_runtime_env_overrides(overrides) == overrides
    assert overrides.get("MTPLX_QWEN4_OPDIET") == "1"
    assert overrides.get("MTPLX_QWEN4_BLOCK_VERIFY") == "1"
    assert overrides.get("MTPLX_QWEN4_VERIFY_GLUE") == "1"
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    assert ro.qwen4_opdiet_enabled() is True
    assert ro.qwen4_verify_glue_enabled() is True


def test_health_surfaces_the_three_no_observable_verify_lanes(monkeypatch):
    """The three verify lanes with no per-window observable appear in
    /health qwen4_install_reports with an at-use ``armed`` the battery can gate
    on (True under a served stamp, False when the key is =0).
    """

    import mtplx.qwen4_block_verify as block_verify
    import mtplx.qwen4_draft_k20_prescatter as prescatter

    state = SimpleNamespace(runtime=SimpleNamespace(model=None))
    # Unforced globals + cleared first-use latches (the served import-time
    # state); the modules were imported at the top of this file, before any stamp.
    monkeypatch.setattr(ro, "_QWEN4_OPDIET", None)
    monkeypatch.setattr(ro, "_QWEN4_OPDIET_SELECTED", None)
    monkeypatch.setattr(block_verify, "_ENABLED", None)
    monkeypatch.setattr(prescatter, "_ENABLED", None)
    ro.reset_qwen4_opdiet_applied_for_test()
    block_verify.reset_engagement_for_test()
    prescatter.reset_engagement_for_test()
    keys = (
        "MTPLX_QWEN4_OPDIET",
        "MTPLX_QWEN4_BLOCK_VERIFY",
        "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
    )
    for k in keys:
        monkeypatch.delenv(k, raising=False)

    # Off -> absent (== off), so an unarmed lane never claims a verdict.
    rep = openai._qwen4_install_reports(state)
    assert "opdiet" not in rep
    assert "block_verify" not in rep
    assert "draft_k20_prescatter" not in rep

    # Served stamp applied AFTER import -> present with armed True at /health
    # (the fix: gate-able from the install verdict, not the env).
    for k in keys:
        monkeypatch.setenv(k, "1")
    rep = openai._qwen4_install_reports(state)
    assert rep["opdiet"]["armed"] is True
    assert rep["block_verify"] == {"armed": True, "engaged": False}
    assert rep["draft_k20_prescatter"]["armed"] is True
    assert rep["draft_k20_prescatter"]["engaged"] is False

    # Per-key opt-out (=0) -> absent again.
    for k in keys:
        monkeypatch.setenv(k, "0")
    rep = openai._qwen4_install_reports(state)
    assert "opdiet" not in rep
    assert "block_verify" not in rep
    assert "draft_k20_prescatter" not in rep


def _bare_speed_config(tmp_path) -> str:
    """A Bare-Speed-shaped config.json: lm_head/shared/routed all Q4/g64,
    router Q8/g64, 48 layers. Geometry matches Optimized-Speed; only the quant
    recipe differs, so the FR-Spec and stage-3 pack predicates read real quant.
    """

    def entry(bits, group):
        return {"bits": bits, "group_size": group, "mode": "affine"}

    quant = {"bits": 4, "group_size": 64}
    quant["language_model.lm_head"] = entry(4, 64)
    per_module = {
        "mlp.gate": (8, 64),
        "mlp.shared_expert_gate": (4, 64),
        "mlp.shared_expert.gate_proj": (4, 64),
        "mlp.shared_expert.up_proj": (4, 64),
        "mlp.shared_expert.down_proj": (4, 64),
        "mlp.switch_mlp.gate_proj": (4, 64),
        "mlp.switch_mlp.up_proj": (4, 64),
        "mlp.switch_mlp.down_proj": (4, 64),
    }
    for index in range(48):
        prefix = f"language_model.model.layers.{index}."
        for module, (bits, group) in per_module.items():
            quant[prefix + module] = entry(bits, group)
    cfg = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 48,
            "tie_word_embeddings": False,
        },
        "quantization": quant,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(tmp_path)


def test_bare_speed_arms_frspec_and_stage3(tmp_path, monkeypatch):
    # The Bare-Speed pack (lm_head Q4/g64, shared + routed experts Q4/g64) must
    # arm FR-Spec and the M=4 stage-3 optimization, exactly as Optimized-Speed's
    # Q8-shared / Q4-g32-routed geometry does. The fixed-M4 predicate is pure
    # geometry (both packs share it); the FR-Spec and stage-3 pack predicates
    # read the real per-module quantization from config.json below.
    import mtplx.qwen4_block_verify as block_verify
    import mtplx.qwen4_draft_k20_prescatter as prescatter

    for key in _LANE_ENV_KEYS + (
        "MTPLX_FRSPEC_DRAFT",
        "MTPLX_FRSPEC_VOCAB",
        "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
        "MTPLX_QWEN4_BLOCK_VERIFY",
        "MTPLX_QWEN4_OPDIET",
        "MTPLX_FUSED_GATE_UP",
        "MTPLX_QWEN4_M4_STAGE3",
        "MTPLX_QWEN4_M4_ROUTED_GLU",
        "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE",
        "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL",
        "MTPLX_QWEN4_PLE_CACHED_AUX",
        "MTPLX_QSA_POOLED_ROWSEL",
        "MTPLX_FABLE_PLE_CACHED_AUX",
        "MTPLX_FABLE_QSA_POOLED_ROWSEL",
    ):
        monkeypatch.delenv(key, raising=False)
    # Unforced readers + cleared first-use latches: the served import-time state.
    monkeypatch.setattr(ro, "_QWEN4_OPDIET", None)
    monkeypatch.setattr(ro, "_QWEN4_OPDIET_SELECTED", None)
    monkeypatch.setattr(block_verify, "_ENABLED", None)
    monkeypatch.setattr(prescatter, "_ENABLED", None)
    ro.reset_qwen4_opdiet_applied_for_test()
    block_verify.reset_engagement_for_test()
    prescatter.reset_engagement_for_test()

    model = _bare_speed_config(tmp_path)
    args = SimpleNamespace(
        generation_mode="mtp",
        verify_strategy="capture_commit",
        model=model,
    )
    monkeypatch.setattr(openai, "_served_model_is_qwen4_fixed_m4", lambda a: True)
    overrides = openai._server_runtime_env_overrides(args, {})

    # FR-Spec + K20 arm on the Q4/g64 head.
    assert openai._served_model_lm_head_is_frspec_capable(args) is True
    assert overrides.get("MTPLX_FRSPEC_DRAFT") == "1"
    assert overrides.get("MTPLX_FRSPEC_VOCAB", "").startswith("builtin:")
    assert overrides.get("MTPLX_QWEN4_DRAFT_K20_PRESCATTER") == "1"

    # The M=4 stage-3 optimization + its children arm on the Q4/g64 experts.
    assert openai._served_model_pack_is_stage3_geometry(args) is True
    assert overrides.get("MTPLX_QWEN4_M4_STAGE3") == "1"
    assert overrides.get("MTPLX_QWEN4_M4_ROUTED_GLU") == "1"
    assert overrides.get("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE") == "1"
    assert overrides.get("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL") == "1"

    # The two #475 aux optimizations arm on the Bare config as well.
    assert overrides.get("MTPLX_QWEN4_PLE_CACHED_AUX") == "1"
    assert overrides.get("MTPLX_QSA_POOLED_ROWSEL") == "1"

    # Applied as the server does, the three no-observable verify lanes and the
    # two #475 aux optimizations surface in /health qwen4_install_reports with
    # armed True for the Bare-Speed config.
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    state = SimpleNamespace(runtime=SimpleNamespace(model=None))
    rep = openai._qwen4_install_reports(state)
    assert rep["opdiet"]["armed"] is True
    assert rep["block_verify"]["armed"] is True
    assert rep["draft_k20_prescatter"]["armed"] is True
    assert rep["ple_cached_aux"]["armed"] is True
    assert rep["qsa_pooled_rowsel"]["armed"] is True
