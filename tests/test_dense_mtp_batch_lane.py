"""Install-gate tests for the dense batched-MTP lane (T-204 item 1).

The gate's job is to fail at STARTUP rather than on a user's first request, and
to route MoE models away from itself. Every refusal below corresponds to a
concrete way the driver would otherwise misbehave.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.dense_mtp_batch_lane import (
    DenseMTPBatchInstallError,
    install_dense_mtp_batch_lane,
    model_is_dense_mtp_batch_capable,
)


DENSE_CONFIG = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "mtp_num_hidden_layers": 1,
    },
    "quantization": {"bits": 4, "group_size": 64},
}

MOE_CONFIG = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_experts": 256,
        "num_experts_per_tok": 8,
    },
    "quantization": {"bits": 4, "group_size": 64},
}


class _FakeRuntime:
    def __init__(self, model_path: Path, *, mtp_enabled: bool = True, drop: str = ""):
        self.model_path = str(model_path)
        self.mtp_enabled = mtp_enabled
        for name in (
            "forward_ar",
            "forward_ar_capture",
            "draft_mtp",
            "make_cache",
            "make_mtp_cache",
            "update_mtp_cache",
        ):
            if name != drop:
                setattr(self, name, lambda *a, **k: None)


def _model_dir(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "model"
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    return path


# --------------------------------------------------------------------------- #
# Model-type routing: the two lanes are mutually exclusive
# --------------------------------------------------------------------------- #
def test_dense_config_is_recognised_and_moe_is_not() -> None:
    assert model_is_dense_mtp_batch_capable(DENSE_CONFIG) is True
    assert model_is_dense_mtp_batch_capable(MOE_CONFIG) is False


def test_install_refuses_the_moe_topology(tmp_path: Path) -> None:
    """A MoE model belongs to the A3B lane and its router receipt, not here."""

    runtime = _FakeRuntime(_model_dir(tmp_path, MOE_CONFIG))
    with pytest.raises(DenseMTPBatchInstallError, match="refuses the MoE topology"):
        install_dense_mtp_batch_lane(runtime)


def test_install_refuses_an_unrelated_model_type(tmp_path: Path) -> None:
    config = dict(DENSE_CONFIG, model_type="llama")
    runtime = _FakeRuntime(_model_dir(tmp_path, config))
    with pytest.raises(DenseMTPBatchInstallError, match="requires model_type"):
        install_dense_mtp_batch_lane(runtime)


# --------------------------------------------------------------------------- #
# Capability gates
# --------------------------------------------------------------------------- #
def test_install_requires_an_mtp_enabled_runtime(tmp_path: Path) -> None:
    runtime = _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG), mtp_enabled=False)
    with pytest.raises(DenseMTPBatchInstallError, match="MTP-enabled runtime"):
        install_dense_mtp_batch_lane(runtime)


def test_install_names_the_missing_runtime_entry_point(tmp_path: Path) -> None:
    runtime = _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG), drop="update_mtp_cache")
    with pytest.raises(DenseMTPBatchInstallError, match="update_mtp_cache"):
        install_dense_mtp_batch_lane(runtime)


def test_install_refuses_a_capture_backend_without_per_step_states(
    tmp_path: Path,
) -> None:
    """A final-only capture cannot express per-row accept lengths.

    Catching it here means a misconfigured server fails at startup instead of
    failing the first cohort a real user lands in.
    """

    runtime = _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG))
    with pytest.raises(DenseMTPBatchInstallError, match="per-step GDN states"):
        install_dense_mtp_batch_lane(runtime, capture_backend="linear-gdn-final")


def test_install_refuses_a_backend_name_that_is_not_a_backend(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG))
    with pytest.raises(DenseMTPBatchInstallError, match="not a capture backend"):
        install_dense_mtp_batch_lane(runtime, capture_backend="tape")


def test_install_refuses_the_compiled_draft_core(tmp_path: Path) -> None:
    """Compiled drafting cannot sample, so it would break every hot request."""

    runtime = _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG))
    with pytest.raises(DenseMTPBatchInstallError, match="draft_core='eager'"):
        install_dense_mtp_batch_lane(runtime, draft_core="compiled")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"cohort_slots": 1}, "cohort_slots must be >= 2"),
        ({"depth": 0}, "depth must be >= 1"),
        ({"max_context_tokens": 0}, "max_context_tokens must be >= 1"),
        ({"head_history": "nonsense"}, "head_history must be"),
        ({"loop_mode": "nonsense"}, "loop_mode must be"),
    ],
)
def test_install_validates_its_settings(tmp_path: Path, kwargs, message) -> None:
    runtime = _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG))
    with pytest.raises(DenseMTPBatchInstallError, match=message):
        install_dense_mtp_batch_lane(runtime, **kwargs)


# --------------------------------------------------------------------------- #
# A successful install
# --------------------------------------------------------------------------- #
def test_successful_install_carries_the_settings_and_an_honest_selfcheck(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG))
    lane = install_dense_mtp_batch_lane(
        runtime, cohort_slots=6, depth=3, max_context_tokens=32768
    )
    assert lane.runtime is runtime
    assert lane.geometry.cohort_slots == 6
    assert lane.geometry.depth == 3
    assert lane.model_type == "qwen3_5"
    assert lane.capture_backend == "stock"
    assert lane.route_id.startswith("dense_mtp_batch/qwen3_5/d3/")
    # The self-check must not claim a numerical result it never computed.
    assert lane.selfcheck["mode"] == "structural"
    assert lane.selfcheck["ran_forward"] is False


def test_route_id_changes_when_the_served_config_changes(tmp_path: Path) -> None:
    """The fingerprint exists to notice a topology swap under a cached id."""

    lane_a = install_dense_mtp_batch_lane(
        _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG))
    )
    changed = dict(DENSE_CONFIG)
    changed["text_config"] = dict(DENSE_CONFIG["text_config"], num_hidden_layers=48)
    lane_b = install_dense_mtp_batch_lane(
        _FakeRuntime(_model_dir(tmp_path / "b", changed))
    )
    assert lane_a.config_fingerprint != lane_b.config_fingerprint
    assert lane_a.route_id != lane_b.route_id


def test_route_id_is_stable_for_the_same_config(tmp_path: Path) -> None:
    lane_a = install_dense_mtp_batch_lane(
        _FakeRuntime(_model_dir(tmp_path, DENSE_CONFIG))
    )
    lane_b = install_dense_mtp_batch_lane(
        _FakeRuntime(_model_dir(tmp_path / "b", DENSE_CONFIG))
    )
    assert lane_a.route_id == lane_b.route_id


# --------------------------------------------------------------------------- #
# The real MoE router-receipt gate: A3B runtime env must not reach a dense model
# --------------------------------------------------------------------------- #
def test_a3b_runtime_env_is_not_applied_to_a_dense_model(tmp_path: Path) -> None:
    """`--scheduler-mode mtp_batch` must not imply the A3B runtime environment.

    Regression test for a real startup failure. Eight A3B-specific env
    overrides were applied whenever the scheduler mode was mtp_batch, before
    any lane was consulted. `MTPLX_QWEN_ROW_OWNED_ROUTER=1` makes
    `prepare_qwen_row_owned_routers` run inside `runtime.load`, and that
    function hard-requires the exact A3B topology, so a dense qwen3_5 server
    died at construction with QwenRowOwnedRouterConfigError.

    This fires strictly earlier than the dense lane's own install gate, which
    is why refusing MoE configs there was necessary but not sufficient.
    """

    import argparse

    from mtplx.server.openai import _server_runtime_env_overrides

    dense_dir = _model_dir(tmp_path / "dense", DENSE_CONFIG)
    args = argparse.Namespace(
        model=str(dense_dir),
        scheduler_mode="mtp_batch",
        generation_mode="mtp",
        verify_strategy="target_prefix",
    )
    overrides = _server_runtime_env_overrides(args, None)
    assert "MTPLX_QWEN_ROW_OWNED_ROUTER" not in overrides
    assert "MTPLX_QWEN_COMBINE_TAIL" not in overrides
    assert "MTPLX_QWEN_MOE_PACK_GATE_UP" not in overrides
    assert not [key for key in overrides if "A3B" in key]


def test_a3b_runtime_env_is_still_applied_to_an_moe_model(tmp_path: Path) -> None:
    """The A3B lane must be completely unaffected by the dense carve-out."""

    import argparse

    from mtplx.server.openai import _server_runtime_env_overrides

    moe_dir = _model_dir(tmp_path / "moe", MOE_CONFIG)
    args = argparse.Namespace(
        model=str(moe_dir),
        scheduler_mode="mtp_batch",
        generation_mode="mtp",
        verify_strategy="target_prefix",
    )
    overrides = _server_runtime_env_overrides(args, None)
    assert overrides["MTPLX_QWEN_ROW_OWNED_ROUTER"] == "1"
    assert overrides["MTPLX_QWEN_COMBINE_TAIL"] == "1"
    assert overrides["MTPLX_A3B_GDN_POSTCONV_IMPL"] == "headquarter"


def test_serial_mode_gets_no_batching_env_on_either_topology(tmp_path: Path) -> None:
    import argparse

    from mtplx.server.openai import _server_runtime_env_overrides

    for name, config in (("dense", DENSE_CONFIG), ("moe", MOE_CONFIG)):
        args = argparse.Namespace(
            model=str(_model_dir(tmp_path / f"serial-{name}", config)),
            scheduler_mode="serial",
            generation_mode="mtp",
            verify_strategy="target_prefix",
        )
        overrides = _server_runtime_env_overrides(args, None)
        assert "MTPLX_QWEN_ROW_OWNED_ROUTER" not in overrides, name


def test_no_shadowed_top_level_functions_in_the_server_module() -> None:
    """A duplicate `def` is legal Python and the later one silently wins.

    This shipped and broke server startup outright: a helper added for the
    dense lane was named `_env_int`, the module already defined
    `_env_int(name, default)` thirty thousand lines further down, and every
    call raised TypeError. The module imported fine, `py_compile` passed, and
    343 CPU tests passed -- because none of them construct a ServerState. It
    took a real server start to find, which is the most expensive place to
    find it.

    Checking the whole module rather than only the names this lane added: the
    defect is a property of the file, and a guard scoped to what I happened to
    touch would miss the next one.
    """

    import ast
    import pathlib

    import mtplx.server.openai as server_module

    source = pathlib.Path(server_module.__file__).read_text()
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in seen:
                duplicates.append(
                    f"{node.name} at lines {seen[node.name]} and {node.lineno}"
                )
            seen[node.name] = node.lineno

    assert not duplicates, (
        "top-level functions defined more than once; the later definition "
        "silently shadows the earlier and every earlier call site breaks:\n  "
        + "\n  ".join(duplicates)
    )


# --------------------------------------------------------------------------- #
# The MoE gate has THREE detection paths; MOE_CONFIG trips all of them
# --------------------------------------------------------------------------- #
# Found by mutation: breaking any single clause of
#   if "moe" in model_type or "moe" in text_type or text.get("num_experts")
# left the suite green, because the shared fixture satisfies all three and the
# survivors caught what the broken one missed. So each path is unverified, and
# a model that is MoE by only ONE of them would install. The soak cannot
# surface this -- it runs the right model, so the gate is never exercised.
#
# One config per clause, each tripping exactly one.
def _moe_by_top_level_model_type() -> dict:
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
        },
        "quantization": {"bits": 4, "group_size": 64},
    }


def _moe_by_text_config_model_type() -> dict:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
        },
        "quantization": {"bits": 4, "group_size": 64},
    }


def _moe_by_num_experts_only() -> dict:
    """The dangerous one: nothing in either model_type says 'moe'.

    If this clause stops working, a mixture-of-experts model whose type strings
    happen not to carry the substring installs on the dense lane, which the
    lane exists to refuse.
    """

    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_experts": 128,
        },
        "quantization": {"bits": 4, "group_size": 64},
    }


@pytest.mark.parametrize(
    "config_fn, clause",
    [
        (_moe_by_top_level_model_type, "top-level model_type"),
        (_moe_by_text_config_model_type, "text_config model_type"),
        (_moe_by_num_experts_only, "num_experts"),
    ],
)
def test_each_moe_detection_path_refuses_on_its_own(
    tmp_path: Path, config_fn, clause: str
) -> None:
    """Every clause must refuse alone, not merely as part of the set."""

    runtime = _FakeRuntime(_model_dir(tmp_path, config_fn()))
    with pytest.raises(DenseMTPBatchInstallError, match="refuses the MoE topology"):
        install_dense_mtp_batch_lane(runtime)
