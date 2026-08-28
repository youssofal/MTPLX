from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from scripts import qwen38_native_mtp_candidates as registry

ROOT = Path(__file__).parents[1]


def test_registry_contains_only_native_mtp_candidate_rows() -> None:
    assert registry.CANDIDATE_ROWS == frozenset({8, 10, 11, 17, 20, 28, 36, 61, 63})
    assert {spec.row for spec in registry.NATIVE_MTP_CANDIDATES.values()} == (
        registry.CANDIDATE_ROWS
    )
    assert set(registry.NATIVE_MTP_CANDIDATES) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r11_position_ema",
        "r17_q4_mtp_block",
        "r20_kv_only_history",
        "r28_q4_mtp_block",
        "r36_qkv_islands",
        "r61_dual_norm_concat",
        "r63_q8_embedding_dual_norm",
    }


def test_registry_is_immutable_and_hermetic() -> None:
    with pytest.raises(TypeError):
        registry.NATIVE_MTP_CANDIDATES["invented"] = object()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        registry.NATIVE_MTP_CANDIDATES["r08_device_draft"].row = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        registry.NATIVE_MTP_CANDIDATES["r17_q4_mtp_block"].callsite_regimes[
            0
        ].phase = "invented"  # type: ignore[misc]
    source_path = Path(registry.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    allowed_imports = {"__future__", "dataclasses", "types", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {alias.name.split(".", 1)[0] for alias in node.names} <= allowed_imports
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".", 1)[0] in allowed_imports
        elif isinstance(node, ast.Call):
            assert not (
                isinstance(node.func, ast.Name)
                and node.func.id in {"open", "eval", "exec", "__import__", "print"}
            )
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"environ", "getenv", "putenv", "unsetenv"}

    isolated = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(ROOT)!r}); "
                "import scripts.qwen38_native_mtp_candidates as r; "
                "assert len(r.NATIVE_MTP_CANDIDATES) == 9"
            ),
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr


def test_candidate_and_frozen_provenance_matches_authoritative_receipt() -> None:
    receipt = json.loads(
        (
            ROOT
            / "docs/perf/receipts/qwen38-challenge-port/"
            "yukon-accepted-2026-08-23.json"
        ).read_text(encoding="utf-8")
    )
    authoritative = {int(row["ordinal"]): row for row in receipt["rows"]}
    specs = [
        *registry.NATIVE_MTP_CANDIDATES.values(),
        *registry.FROZEN_TARGET_SUBSTRATE.values(),
    ]
    for spec in specs:
        row = authoritative[spec.row]
        assert spec.pr_number == int(row["pr_number"])
        assert spec.source_commit == row["source_commit"]
    assert all(len(spec.source_commit) == 40 for spec in registry.NATIVE_MTP_CANDIDATES.values())


def test_candidate_phase_ownership_is_exact() -> None:
    phases = {
        feature: spec.phase for feature, spec in registry.NATIVE_MTP_CANDIDATES.items()
    }
    assert phases == {
        "r08_device_draft": "mtp_decode/proposal",
        "r10_compact_vocab": "mtp_decode/head",
        "r11_position_ema": "mtp_decode/adaptive",
        "r17_q4_mtp_block": "mtp_history+decode/block",
        "r20_kv_only_history": "mtp_history/prefill",
        "r28_q4_mtp_block": "mtp_history+decode/block",
        "r36_qkv_islands": "mtp_history+decode/block",
        "r61_dual_norm_concat": "mtp_history+decode/input",
        "r63_q8_embedding_dual_norm": "mtp_history+decode/input",
    }


def test_real_native_call_site_geometry_is_not_conflated() -> None:
    row8 = registry.NATIVE_MTP_CANDIDATES["r08_device_draft"]
    row10 = registry.NATIVE_MTP_CANDIDATES["r10_compact_vocab"]
    row17 = registry.NATIVE_MTP_CANDIDATES["r17_q4_mtp_block"]
    row20 = registry.NATIVE_MTP_CANDIDATES["r20_kv_only_history"]
    row36 = registry.NATIVE_MTP_CANDIDATES["r36_qkv_islands"]
    row61 = registry.NATIVE_MTP_CANDIDATES["r61_dual_norm_concat"]
    row63 = registry.NATIVE_MTP_CANDIDATES["r63_q8_embedding_dual_norm"]

    assert row8.logical_m == (1,)
    assert row8.native_depths == (1, 2, 3)
    assert row10.logical_m == (1,)
    assert row10.quant_bits == 4 and row10.group_size == 64
    assert "98,330" in row10.callsite_shape and "98,336" in row10.callsite_shape
    assert row10.ownership == "code"

    assert row17.ownership == "artifact"
    assert row17.logical_m == (1,)
    assert row17.dynamic_logical_m == "stock-history chunk length L>=1"
    assert row17.quant_bits == 4 and row17.group_size == 64
    assert any("mtp.fc" in surface for surface in row17.owned_surfaces)
    assert not any("draft_lm_head" in surface for surface in row17.owned_surfaces)
    assert row36.ownership == "artifact"
    assert "bfloat16" in row36.dtypes
    assert any("precision_islands.q.weight" in surface for surface in row36.owned_surfaces)

    assert row20.logical_m == ()
    assert row20.dynamic_logical_m == "prefill chunk length L>=1"
    assert "B=1, L=len(token_ids)" in row20.callsite_shape
    assert row20.dtypes == ("uint32", "bfloat16")
    assert row20.min_context_tokens == 16_384

    assert row61.logical_m == (1,)
    assert row61.dynamic_logical_m == "stock-history chunk length L>=1"
    assert "proposal [1,1,5120] -> [1,1,10240]" in row61.callsite_shape
    assert "stock history [1,L,5120] -> [1,L,10240]" in row61.callsite_shape
    assert row63.quant_bits == 8 and row63.group_size == 64
    assert "Q8 embedding" in row63.callsite_shape


def test_block_artifact_metadata_matches_authoritative_specs() -> None:
    from mtplx import qwen38_mtp_block_artifacts as artifacts

    for feature in ("r17_q4_mtp_block", "r28_q4_mtp_block", "r36_qkv_islands"):
        candidate = registry.NATIVE_MTP_CANDIDATES[feature]
        artifact = artifacts.QWEN38_MTP_BLOCK_ARTIFACTS[candidate.artifact_variant]
        assert candidate.source_commit == artifact.source_commit
        assert candidate.artifact_manifest_sha256 == artifact.manifest_sha256
        assert candidate.artifact_file_sha256 == artifact.file_sha256
        assert candidate.artifact_bytes == artifact.bytes

    island_source = inspect.getsource(artifacts._install_precision_islands)
    packable_source = inspect.getsource(artifacts._PackableDense)
    assert "attn.k_proj = _dense_linear(k_weight)" in island_source
    assert "attn.v_proj = _dense_linear(v_weight)" in island_source
    assert "self._mtplx_pack_linear = layer" in packable_source


def test_row20_threshold_and_dependencies_match_production() -> None:
    from mtplx import mtp_patch
    from mtplx.qwen38_challenge import (
        QWEN38_KV_ONLY_MIN_CONTEXT,
        QWEN38_PACKING,
    )

    row20 = registry.NATIVE_MTP_CANDIDATES["r20_kv_only_history"]
    assert row20.min_context_tokens == QWEN38_KV_ONLY_MIN_CONTEXT
    for feature in (
        "r10_compact_vocab",
        "r17_q4_mtp_block",
        "r28_q4_mtp_block",
        "r36_qkv_islands",
        "r63_q8_embedding_dual_norm",
    ):
        assert registry.NATIVE_MTP_CANDIDATES[feature].packing == QWEN38_PACKING
    source = inspect.getsource(mtp_patch.install_qwen38_kv_only_history_append)
    for dependency in (
        "text.model.embed_tokens",
        "mtp.pre_fc_norm_embedding",
        "mtp.pre_fc_norm_hidden",
        "mtp.fc",
        "layer.input_layernorm",
        "attention.k_proj",
        "attention.v_proj",
        "attention.k_norm",
        "attention.rope",
        'getattr(k_proj, "_mtplx_pack_linear", k_proj)',
        'getattr(v_proj, "_mtplx_pack_linear", v_proj)',
    ):
        assert dependency in source
    for skipped in ("attention.q_proj", "attention.o_proj", "layer.mlp"):
        assert skipped not in source


def test_block_and_input_candidates_freeze_decode_and_stock_history_regimes() -> None:
    shared = {
        "r17_q4_mtp_block",
        "r28_q4_mtp_block",
        "r36_qkv_islands",
        "r61_dual_norm_concat",
        "r63_q8_embedding_dual_norm",
    }
    for feature in shared:
        spec = registry.NATIVE_MTP_CANDIDATES[feature]
        regimes = {regime.route: regime for regime in spec.callsite_regimes}
        assert set(regimes) == {
            "native_proposal",
            "stock_history",
            "row20_kv_only_history",
        }

        proposal = regimes["native_proposal"]
        assert proposal.phase == "mtp_decode"
        assert proposal.logical_m == (1,)
        assert proposal.dynamic_logical_m is None
        assert proposal.native_depths == (1, 2, 3)
        assert proposal.active_when == "native proposal route D1-D3"
        assert proposal.bypassed_by is None

        history = regimes["stock_history"]
        assert history.phase == "mtp_history/prefill"
        assert history.logical_m == ()
        assert history.dynamic_logical_m == "stock-history chunk length L>=1"
        assert history.native_depths == ()
        assert history.active_when == (
            "stock_history selected: prompt_tokens<16384 or row20 not installed"
        )
        assert history.bypassed_by == (
            "r20_kv_only_history selected at prompt_tokens>=16384"
        )
        assert "B=1" in history.callsite_shape
        assert "L>=1" in history.callsite_shape


def test_row20_phase_route_partially_uses_blocks_but_bypasses_input_callables() -> None:
    row20 = registry.NATIVE_MTP_CANDIDATES["r20_kv_only_history"]
    assert row20.min_context_tokens == 16_384
    assert row20.parent_rule == registry.ParentRule()
    assert row20.partially_used_artifact_features == frozenset(
        {"r17_q4_mtp_block", "r28_q4_mtp_block", "r36_qkv_islands"}
    )
    assert row20.bypassed_candidate_callable_features == frozenset(
        {"r61_dual_norm_concat", "r63_q8_embedding_dual_norm"}
    )
    (history,) = row20.callsite_regimes
    assert history.active_when == (
        "row20 installed and prompt_tokens>=16384 selects kv_only_history"
    )
    assert "prompt_tokens<16384 selects stock_history" in row20.callsite_shape

    for feature in row20.partially_used_artifact_features:
        candidate = registry.NATIVE_MTP_CANDIDATES[feature]
        regimes = candidate.callsite_regimes
        partial = next(regime for regime in regimes if regime.route == "row20_kv_only_history")
        assert partial.dynamic_logical_m == "row20 K/V-only chunk length L>=1"
        assert partial.active_candidate_surfaces
        assert partial.bypassed_candidate_surfaces
        assert set(partial.active_candidate_surfaces) < set(candidate.owned_surfaces)
        assert set(partial.bypassed_candidate_surfaces) == (
            set(candidate.owned_surfaces) - set(partial.active_candidate_surfaces)
        )
        assert partial.read_dependencies == row20.unchanged_dependencies

    row17_partial = next(
        regime
        for regime in registry.NATIVE_MTP_CANDIDATES[
            "r17_q4_mtp_block"
        ].callsite_regimes
        if regime.route == "row20_kv_only_history"
    )
    assert {
        "language_model.mtp.layers.0.self_attn.k_proj.weight",
        "language_model.mtp.layers.0.self_attn.k_proj.scales",
        "language_model.mtp.layers.0.self_attn.k_proj.biases",
        "language_model.mtp.layers.0.self_attn.v_proj.weight",
        "language_model.mtp.layers.0.self_attn.v_proj.scales",
        "language_model.mtp.layers.0.self_attn.v_proj.biases",
    } <= set(row17_partial.active_candidate_surfaces)
    row36_partial = next(
        regime
        for regime in registry.NATIVE_MTP_CANDIDATES[
            "r36_qkv_islands"
        ].callsite_regimes
        if regime.route == "row20_kv_only_history"
    )
    assert {
        "language_model.mtp.layers.0.self_attn.k_proj._mtplx_pack_linear.weight <- precision_islands.k.weight sorted by precision_islands.k.indices",
        "language_model.mtp.layers.0.self_attn.v_proj._mtplx_pack_linear.weight <- precision_islands.v.weight sorted by precision_islands.v.indices",
    } <= set(row36_partial.active_candidate_surfaces)

    for feature in row20.bypassed_candidate_callable_features:
        candidate = registry.NATIVE_MTP_CANDIDATES[feature]
        partial = next(
            regime
            for regime in candidate.callsite_regimes
            if regime.route == "row20_kv_only_history"
        )
        assert partial.active_candidate_surfaces == ()
        assert partial.bypassed_candidate_surfaces == candidate.owned_surfaces
        assert partial.read_dependencies == candidate.unchanged_dependencies


def test_artifact_and_input_replacements_are_mutually_exclusive() -> None:
    blocks = {
        "r17_q4_mtp_block",
        "r28_q4_mtp_block",
        "r36_qkv_islands",
    }
    inputs = {"r61_dual_norm_concat", "r63_q8_embedding_dual_norm"}
    for feature in blocks:
        assert registry.NATIVE_MTP_CANDIDATES[feature].incompatible == blocks - {feature}
    for feature in inputs:
        assert registry.NATIVE_MTP_CANDIDATES[feature].incompatible == inputs - {feature}
    assert registry.NATIVE_MTP_CANDIDATES["r10_compact_vocab"].parent_rule == (
        registry.ParentRule(
            required_control_features=frozenset({"r08_device_draft"})
        )
    )
    assert registry.NATIVE_MTP_CANDIDATES["r17_q4_mtp_block"].parent_rule == (
        registry.ParentRule(
            replacement_only=True,
            implicit_replaces="stock_mtp_block",
        )
    )
    assert registry.NATIVE_MTP_CANDIDATES["r28_q4_mtp_block"].parent_rule == (
        registry.ParentRule(
            replacement_only=True,
            replaces=frozenset({"r17_q4_mtp_block"}),
            implicit_replaces="stock_mtp_block",
        )
    )
    assert registry.NATIVE_MTP_CANDIDATES["r36_qkv_islands"].parent_rule == (
        registry.ParentRule(
            replacement_only=True,
            replaces=frozenset({"r17_q4_mtp_block", "r28_q4_mtp_block"}),
            implicit_replaces="stock_mtp_block",
        )
    )
    assert registry.NATIVE_MTP_CANDIDATES[
        "r63_q8_embedding_dual_norm"
    ].parent_rule == registry.ParentRule(
        replacement_only=True,
        replaces=frozenset({"r61_dual_norm_concat"}),
    )


def test_frozen_target_substrate_is_not_a_candidate_delta() -> None:
    assert set(registry.FROZEN_TARGET_SUBSTRATE) == {18, 21, 24, 26, 48, 50, 53}
    assert all(spec.phase == "target/general" for spec in registry.FROZEN_TARGET_SUBSTRATE.values())
    assert all(len(spec.source_commit) == 40 for spec in registry.FROZEN_TARGET_SUBSTRATE.values())
    with pytest.raises(registry.NativeMTPRouteError, match="frozen substrate"):
        registry.validate_native_mtp_route_delta(
            "r08_device_draft+r18_gdn_decay_memo",
            "r08_device_draft+r18_gdn_decay_memo+r21_qk_rms_rope",
        )


def test_unreachable_ineligible_and_removed_families_are_excluded() -> None:
    assert registry.NATIVE_MTP_DEPTH_MAX == 3
    assert registry.NATIVE_D3_VERIFY_LOGICAL_M == 4
    assert {spec.logical_m for spec in registry.UNREACHABLE_NATIVE_ROUTES.values()} == {
        5,
        6,
        7,
        8,
    }
    assert {spec.row for spec in registry.ARGMAX_ONLY_INELIGIBLE.values()} == {
        19,
        42,
        47,
        67,
        69,
    }
    assert all(spec.reason == "temperature1/top-p.95/top-k20 requires full proposal distribution" for spec in registry.ARGMAX_ONLY_INELIGIBLE.values())
    assert set(registry.REMOVED_FAMILIES) == {
        "r70_qmv_sumtable",
        "r78_qmv_active_groups",
        "r80_qmv_m2",
        "source_proposal",
    }
    assert not (
        set(registry.NATIVE_MTP_CANDIDATES)
        & (
            set(registry.UNREACHABLE_NATIVE_ROUTES)
            | set(registry.ARGMAX_ONLY_INELIGIBLE)
            | set(registry.REMOVED_FAMILIES)
        )
    )


def test_public_route_canonicalizer_normalizes_aliases_and_control() -> None:
    assert registry.canonicalize_native_mtp_route("control") == frozenset(
        {"control"}
    )
    assert registry.canonicalize_native_mtp_route("kv_only_history") == frozenset(
        {"r20_kv_only_history"}
    )
    assert registry.canonicalize_native_mtp_route("dual_norm") == frozenset(
        {"r61_dual_norm_concat"}
    )


@pytest.mark.parametrize(
    ("route", "message"),
    [
        ("r08_device_draft+r08_device_draft", "duplicate raw route feature"),
        ("dual_norm+r61_dual_norm_concat", "duplicate canonical route feature"),
        ("kv_only_history+r20_kv_only_history", "duplicate canonical route feature"),
        (
            "dual_norm+r63_q8_embedding_dual_norm",
            "incompatible native-MTP alternatives",
        ),
    ],
)
def test_public_route_canonicalizer_rejects_duplicates_and_alias_conflicts(
    route: str,
    message: str,
) -> None:
    with pytest.raises(registry.NativeMTPRouteError, match=message):
        registry.canonicalize_native_mtp_route(route)


def test_route_delta_accepts_one_candidate_addition_or_explicit_replacement() -> None:
    added = registry.validate_native_mtp_route_delta(
        "r08_device_draft+r18_gdn_decay_memo",
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo",
    )
    assert added.candidate_feature == "r10_compact_vocab"
    assert added.added == frozenset({"r10_compact_vocab"})
    assert added.removed == frozenset()
    assert added.replacement is False
    assert added.implicit_replaced_surface is None

    block = registry.validate_native_mtp_route_delta(
        "r08_device_draft+r17_q4_mtp_block+r18_gdn_decay_memo",
        "r08_device_draft+r36_qkv_islands+r18_gdn_decay_memo",
    )
    assert block.candidate_feature == "r36_qkv_islands"
    assert block.added == frozenset({"r36_qkv_islands"})
    assert block.removed == frozenset({"r17_q4_mtp_block"})
    assert block.replacement is True
    assert block.implicit_replaced_surface is None

    input_route = registry.validate_native_mtp_route_delta(
        "r08_device_draft+r61_dual_norm_concat",
        "r08_device_draft+r63_q8_embedding_dual_norm",
    )
    assert input_route.replacement is True
    assert input_route.implicit_replaced_surface is None


def test_row10_requires_the_device_draft_parent() -> None:
    with pytest.raises(
        registry.NativeMTPRouteError,
        match="requires control feature r08_device_draft",
    ):
        registry.validate_native_mtp_route_delta("control", "r10_compact_vocab")

    delta = registry.validate_native_mtp_route_delta(
        "r08_device_draft", "r08_device_draft+r10_compact_vocab"
    )
    assert delta.candidate_feature == "r10_compact_vocab"
    assert delta.replacement is False
    assert delta.implicit_replaced_surface is None


@pytest.mark.parametrize(
    "feature",
    ["r17_q4_mtp_block", "r28_q4_mtp_block", "r36_qkv_islands"],
)
def test_block_candidate_implicitly_replaces_stock_block(feature: str) -> None:
    delta = registry.validate_native_mtp_route_delta("control", feature)
    assert delta.candidate_feature == feature
    assert delta.added == frozenset({feature})
    assert delta.removed == frozenset()
    assert delta.replacement is True
    assert delta.implicit_replaced_surface == "stock_mtp_block"


@pytest.mark.parametrize(
    ("control", "candidate", "message"),
    [
        (
            "r08_device_draft",
            "r08_device_draft+r63_q8_embedding_dual_norm",
            "replacement-only",
        ),
    ],
)
def test_route_delta_rejects_wrong_candidate_parent(
    control: str,
    candidate: str,
    message: str,
) -> None:
    with pytest.raises(registry.NativeMTPRouteError, match=message):
        registry.validate_native_mtp_route_delta(control, candidate)


def test_row36_can_replace_row28_for_direct_winner_comparison() -> None:
    delta = registry.validate_native_mtp_route_delta(
        "r08_device_draft+r28_q4_mtp_block",
        "r08_device_draft+r36_qkv_islands",
    )
    assert delta.replacement is True
    assert delta.removed == frozenset({"r28_q4_mtp_block"})


def test_rebench_scope_can_isolate_one_frozen_optimized_feature() -> None:
    control = "r08_device_draft+r10_compact_vocab"
    candidate = control + "+r18_gdn_decay_memo"

    with pytest.raises(registry.NativeMTPRouteError, match="frozen substrate"):
        registry.validate_native_mtp_route_delta(control, candidate)

    delta = registry.validate_native_mtp_route_delta(
        control,
        candidate,
        allow_frozen_candidate=True,
    )

    assert delta.candidate_feature == "r18_gdn_decay_memo"
    assert delta.added == frozenset({"r18_gdn_decay_memo"})
    assert delta.removed == frozenset()
    assert delta.replacement is False


@pytest.mark.parametrize(
    ("control", "candidate", "message"),
    [
        ("control", "r08_device_draft+r10_compact_vocab", "exactly one"),
        ("r08_device_draft", "r08_device_draft+r48_boundary_fused", "frozen substrate"),
        ("r08_device_draft", "r08_device_draft+r17_q4_mtp_block+r28_q4_mtp_block", "incompatible"),
        ("r61_dual_norm_concat", "r61_dual_norm_concat+r63_q8_embedding_dual_norm", "incompatible"),
        ("r08_device_draft", "r08_device_draft+dflash_m5", "unreachable"),
        ("r08_device_draft", "r08_device_draft+r42_argmax_shortlist", "correctness-ineligible"),
        ("r08_device_draft", "r08_device_draft+r70_qmv_sumtable", "removed"),
        ("r08_device_draft", "r08_device_draft+invented_route", "unknown route"),
    ],
)
def test_route_delta_rejects_non_isolated_or_ineligible_changes(
    control: str,
    candidate: str,
    message: str,
) -> None:
    with pytest.raises(registry.NativeMTPRouteError, match=message):
        registry.validate_native_mtp_route_delta(control, candidate)
