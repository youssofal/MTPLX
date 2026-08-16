"""CPU construction gates for the retained DSpark native packed-Q3 lane."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa: E402

from mtplx.deepseek_v4_0731_dspark_ffn import (  # noqa: E402
    DSPARK_Q3_GATE_UP_GEOMETRY,
    DeepseekV40731DSparkM5PackedSwitchGLU,
    PreparedDSparkQ3PackedGateUpM5,
    build_dspark_q3_packed_gate_up,
    install_dspark_q3_packed_gate_up_m5,
    prepare_dspark_q3_packed_gate_up_m5,
    validate_dspark_q3_gate_up,
)
from mtplx.models.deepseek_v4 import (  # noqa: E402
    ClampedSwiGLU,
    DeepseekV4DSpark,
    DeepseekV4DSparkStage,
    DeepseekV4MoE,
    Model,
    MoEGate,
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _q3_pair(*, hidden_size: int = 128, width: int = 16, experts: int = 8):
    gate = QuantizedSwitchLinear(
        hidden_size, width, experts, bias=False, group_size=128, bits=3
    )
    up = QuantizedSwitchLinear(
        hidden_size, width, experts, bias=False, group_size=128, bits=3
    )
    for projection in (gate, up):
        projection.scales = projection.scales.astype(mx.bfloat16)
        projection.biases = projection.biases.astype(mx.bfloat16)
    mx.eval(gate.parameters(), up.parameters())
    return gate, up


def _q3_switch(*, hidden_size: int = 128, width: int = 128, experts: int = 8):
    gate, up = _q3_pair(
        hidden_size=hidden_size,
        width=width,
        experts=experts,
    )
    down = QuantizedSwitchLinear(
        width, hidden_size, experts, bias=False, group_size=128, bits=3
    )
    down.scales = down.scales.astype(mx.bfloat16)
    down.biases = down.biases.astype(mx.bfloat16)
    mx.eval(down.parameters())
    return SimpleNamespace(
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
        activation=ClampedSwiGLU(10.0),
    )


def _projection_bytes(projection) -> int:
    return sum(int(projection[name].nbytes) for name in ("weight", "scales", "biases"))


def _stage(stage_id: int, switch=None):
    stage = DeepseekV4DSparkStage.__new__(DeepseekV4DSparkStage)
    ffn = DeepseekV4MoE.__new__(DeepseekV4MoE)
    gate = MoEGate.__new__(MoEGate)
    object.__setattr__(gate, "topk", 6)
    object.__setattr__(ffn, "gate", gate)
    object.__setattr__(ffn, "switch_mlp", object() if switch is None else switch)
    object.__setattr__(stage, "stage_id", stage_id)
    object.__setattr__(stage, "block_size", 5)
    object.__setattr__(stage, "ffn", ffn)
    return stage


def _model_owner(stages=None):
    stages = [_stage(index) for index in range(3)] if stages is None else list(stages)
    dspark = object.__new__(DeepseekV4DSpark)
    object.__setattr__(dspark, "stages", stages)
    object.__setattr__(dspark, "block_size", 5)
    owner = Model.__new__(Model)
    object.__setattr__(owner, "_dspark", dspark)
    object.__setattr__(owner, "mtp", SimpleNamespace(layers=stages))
    return owner


def test_retained_geometry_is_the_physical_five_row_dspark_layout():
    assert DSPARK_Q3_GATE_UP_GEOMETRY == {
        "rows": 5,
        "hidden_size": 4096,
        "width": 2048,
        "experts": 256,
        "top_k": 6,
        "bits": 3,
        "group_size": 128,
        "weight_shape": (256, 2048, 384),
        "metadata_shape": (256, 2048, 32),
    }


def test_loaded_storage_contract_accepts_only_affine_q3_group128_u32_bf16():
    gate, up = _q3_pair()

    contract = validate_dspark_q3_gate_up(
        gate,
        up,
        hidden_size=128,
        width=16,
        experts=8,
        top_k=6,
        rows=5,
    )

    assert contract.bits == 3
    assert contract.group_size == 128
    assert contract.weight_shape == (8, 16, 12)
    assert contract.metadata_shape == (8, 16, 1)


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda gate, up: setattr(up, "bits", 4), "Q3"),
        (lambda gate, up: setattr(up, "group_size", 64), "group-128"),
        (lambda gate, up: setattr(up, "mode", "mxfp4"), "affine"),
        (lambda gate, up: setattr(up, "biases", None), "biases"),
    ],
)
def test_loaded_storage_contract_rejects_nonphysical_q3(mutator, match):
    gate, up = _q3_pair()
    mutator(gate, up)

    with pytest.raises(ValueError, match=match):
        validate_dspark_q3_gate_up(
            gate,
            up,
            hidden_size=128,
            width=16,
            experts=8,
            top_k=6,
            rows=5,
        )


def test_fixed_m5_pack_is_exact_one_dispatch_and_adds_no_resident_weight_bytes():
    switch = _q3_switch()
    original_bytes = _projection_bytes(switch.gate_proj) + _projection_bytes(
        switch.up_proj
    )

    packed = build_dspark_q3_packed_gate_up(
        switch,
        hidden_size=128,
        width=128,
        experts=8,
        top_k=6,
        rows=5,
    )

    assert type(packed) is DeepseekV40731DSparkM5PackedSwitchGLU
    assert packed.down_proj is switch.down_proj
    assert packed.activation is switch.activation
    assert _projection_bytes(packed.gate_up_proj) == original_bytes

    x = (mx.arange(5 * 128).reshape(5, 128) % 17 - 8).astype(mx.bfloat16)
    indices = mx.array([[0, 1, 2, 3, 4, 5]] * 5, dtype=mx.uint32)
    expanded = mx.expand_dims(x, (-2, -3))
    stock = switch.down_proj(
        switch.activation(
            switch.up_proj(expanded, indices),
            switch.gate_proj(expanded, indices),
        ),
        indices,
    ).squeeze(-2)
    candidate = packed(x, indices)
    mx.eval(stock, candidate)
    assert mx.array_equal(candidate, stock)

    class GatherSpy(nn.Module):
        def __init__(self, projection):
            super().__init__()
            self.projection = projection
            self.calls = 0
            self.sorted_indices = []

        def gather(self, x, indices, sorted_indices):
            self.calls += 1
            self.sorted_indices.append(sorted_indices)
            return self.projection.gather(x, indices, sorted_indices)

    spy = GatherSpy(packed.gate_up_proj)
    packed.gate_up_proj = spy
    candidate = packed(x, indices)
    mx.eval(candidate)

    assert spy.calls == 1
    assert spy.sorted_indices == [False]
    assert tuple(candidate.shape) == (5, 6, 128)
    source = inspect.getsource(DeepseekV40731DSparkM5PackedSwitchGLU.__call__)
    assert "indices.size" not in source
    assert "moe_force_unsorted" not in source


@pytest.mark.parametrize("stage_count", [2, 4])
def test_installer_requires_exactly_three_dspark_stages(stage_count):
    owner = _model_owner([_stage(index) for index in range(stage_count)])

    with pytest.raises(ValueError, match="exactly three"):
        prepare_dspark_q3_packed_gate_up_m5(owner)


def test_preparer_rejects_bare_arbitrary_stage_lists():
    with pytest.raises(ValueError, match="model owner"):
        prepare_dspark_q3_packed_gate_up_m5([_stage(index) for index in range(3)])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda owner: owner._dspark.stages.__setitem__(
                1, SimpleNamespace(stage_id=1)
            ),
            "stage identity",
        ),
        (
            lambda owner: setattr(owner._dspark.stages[1], "stage_id", 2),
            "stage order",
        ),
        (
            lambda owner: setattr(owner._dspark.stages[1], "ffn", SimpleNamespace()),
            "FFN identity",
        ),
        (
            lambda owner: setattr(
                owner._dspark.stages[1].ffn, "gate", SimpleNamespace(topk=6)
            ),
            "router identity",
        ),
        (
            lambda owner: setattr(
                owner.mtp, "layers", list(reversed(owner._dspark.stages))
            ),
            "model.mtp",
        ),
        (
            lambda owner: setattr(owner._dspark.stages[1].ffn.gate, "topk", 5),
            "top-k=6",
        ),
        (lambda owner: setattr(owner._dspark, "block_size", 4), "M=5"),
        (
            lambda owner: setattr(owner._dspark.stages[2], "block_size", 4),
            "M=5",
        ),
    ],
)
def test_preparer_rejects_adversarial_dspark_ownership(mutate, match):
    owner = _model_owner()
    mutate(owner)

    with pytest.raises(ValueError, match=match):
        prepare_dspark_q3_packed_gate_up_m5(owner)


def test_installer_validates_and_builds_every_stage_before_atomic_publication(
    monkeypatch,
):
    original = [object(), object(), object()]
    stages = [_stage(index, switch) for index, switch in enumerate(original)]
    owner = _model_owner(stages)
    replacements = [object(), object(), object()]
    calls = []

    def validate(switch, **geometry):
        calls.append(("validate", switch, geometry))

    def build(switch, **geometry):
        calls.append(("build", switch, geometry))
        return replacements[len([call for call in calls if call[0] == "build"]) - 1]

    monkeypatch.setattr(
        "mtplx.deepseek_v4_0731_dspark_ffn._validate_dspark_q3_switch",
        validate,
    )
    monkeypatch.setattr(
        "mtplx.deepseek_v4_0731_dspark_ffn.build_dspark_q3_packed_gate_up",
        build,
    )

    prepared = prepare_dspark_q3_packed_gate_up_m5(owner)

    expected_geometry = {
        "hidden_size": 4096,
        "width": 2048,
        "experts": 256,
        "top_k": 6,
        "rows": 5,
    }
    assert isinstance(prepared, PreparedDSparkQ3PackedGateUpM5)
    assert [stage.ffn.switch_mlp for stage in stages] == original
    assert [call[0] for call in calls] == ["validate"] * 3 + ["build"] * 3
    assert all(call[2] == expected_geometry for call in calls)
    assert prepared.receipt == {
        "candidate": "dspark-native-packed-q3-gate-up-m5",
        "stages": 3,
        "geometry": DSPARK_Q3_GATE_UP_GEOMETRY,
        "gate_up_dispatches_per_stage": 1,
        "stock_gate_up_dispatches_per_stage": 2,
        "explicit_dequantize": False,
        "resident_weight_bytes_added": 0,
    }
    prepared.publish()
    assert [stage.ffn.switch_mlp for stage in stages] == replacements
    prepared.restore()
    assert [stage.ffn.switch_mlp for stage in stages] == original


def test_installer_leaves_every_original_stage_unchanged_when_build_fails(monkeypatch):
    original = [object(), object(), object()]
    stages = [_stage(index, switch) for index, switch in enumerate(original)]
    owner = _model_owner(stages)
    build_calls = 0

    monkeypatch.setattr(
        "mtplx.deepseek_v4_0731_dspark_ffn._validate_dspark_q3_switch",
        lambda *_args, **_kwargs: None,
    )

    def build(*_args, **_kwargs):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 2:
            raise ValueError("stage two failed")
        return object()

    monkeypatch.setattr(
        "mtplx.deepseek_v4_0731_dspark_ffn.build_dspark_q3_packed_gate_up",
        build,
    )

    with pytest.raises(ValueError, match="stage two failed"):
        prepare_dspark_q3_packed_gate_up_m5(owner)

    assert [stage.ffn.switch_mlp for stage in stages] == original


def test_prepared_publication_restores_every_stage_if_one_assignment_fails(
    monkeypatch,
):
    original = [object(), object(), object()]
    replacements = [object(), object(), object()]
    stages = [_stage(index, switch) for index, switch in enumerate(original)]
    owner = _model_owner(stages)
    monkeypatch.setattr(
        "mtplx.deepseek_v4_0731_dspark_ffn._validate_dspark_q3_switch",
        lambda *_args, **_kwargs: None,
    )
    built = iter(replacements)
    monkeypatch.setattr(
        "mtplx.deepseek_v4_0731_dspark_ffn.build_dspark_q3_packed_gate_up",
        lambda *_args, **_kwargs: next(built),
    )
    prepared = prepare_dspark_q3_packed_gate_up_m5(owner)
    original_setattr = DeepseekV4MoE.__setattr__

    def guarded_setattr(self, name, value):
        if self is stages[1].ffn and name == "switch_mlp" and value is replacements[1]:
            raise RuntimeError("publication failed")
        return original_setattr(self, name, value)

    monkeypatch.setattr(DeepseekV4MoE, "__setattr__", guarded_setattr)

    with pytest.raises(RuntimeError, match="publication failed"):
        prepared.publish()

    assert [stage.ffn.switch_mlp for stage in stages] == original


def test_prepared_restore_attempts_every_stage_and_groups_setter_failures(monkeypatch):
    originals = [object(), object(), object()]
    replacements = [object(), object(), object()]
    stages = [
        _stage(index, replacement) for index, replacement in enumerate(replacements)
    ]
    prepared = PreparedDSparkQ3PackedGateUpM5(
        stages=tuple(stages),
        originals=tuple(originals),
        replacements=tuple(replacements),
        receipt={},
    )
    original_setattr = DeepseekV4MoE.__setattr__
    attempts = []

    def guarded_setattr(self, name, value):
        if name == "switch_mlp":
            stage_index = next(
                index for index, stage in enumerate(stages) if stage.ffn is self
            )
            if value is originals[stage_index]:
                attempts.append(stage_index)
                if stage_index in {0, 1}:
                    raise RuntimeError(f"stage {stage_index} restoration failed")
        return original_setattr(self, name, value)

    monkeypatch.setattr(DeepseekV4MoE, "__setattr__", guarded_setattr)

    with pytest.raises(ExceptionGroup, match="DSpark FFN restoration failed") as exc:
        prepared.restore()

    assert attempts == [0, 1, 2]
    assert [str(error) for error in exc.value.exceptions] == [
        "stage 0 restoration failed",
        "stage 1 restoration failed",
    ]
    assert stages[0].ffn.switch_mlp is replacements[0]
    assert stages[1].ffn.switch_mlp is replacements[1]
    assert stages[2].ffn.switch_mlp is originals[2]


def test_publication_keeps_original_error_and_notes_grouped_rollback_failure(
    monkeypatch,
):
    originals = [object(), object(), object()]
    replacements = [object(), object(), object()]
    stages = [_stage(index, original) for index, original in enumerate(originals)]
    prepared = PreparedDSparkQ3PackedGateUpM5(
        stages=tuple(stages),
        originals=tuple(originals),
        replacements=tuple(replacements),
        receipt={},
    )
    original_setattr = DeepseekV4MoE.__setattr__
    publication_error = RuntimeError("stage 1 publication failed")
    restore_attempts = []

    def guarded_setattr(self, name, value):
        if name == "switch_mlp":
            stage_index = next(
                index for index, stage in enumerate(stages) if stage.ffn is self
            )
            if stage_index == 1 and value is replacements[1]:
                raise publication_error
            if value is originals[stage_index]:
                restore_attempts.append(stage_index)
                if stage_index == 0:
                    raise RuntimeError("stage 0 rollback failed")
        return original_setattr(self, name, value)

    monkeypatch.setattr(DeepseekV4MoE, "__setattr__", guarded_setattr)

    with pytest.raises(RuntimeError, match="stage 1 publication failed") as exc:
        prepared.publish()

    assert exc.value is publication_error
    assert restore_attempts == [0, 1, 2]
    assert exc.value.__notes__ == [
        "DSpark FFN publication rollback also failed: "
        "DSpark FFN restoration failed (1 sub-exception)"
    ]


def test_convenience_installer_prepares_then_publishes(monkeypatch):
    owner = _model_owner()
    replacements = [object(), object(), object()]
    monkeypatch.setattr(
        "mtplx.deepseek_v4_0731_dspark_ffn._validate_dspark_q3_switch",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mtplx.deepseek_v4_0731_dspark_ffn.build_dspark_q3_packed_gate_up",
        lambda switch, **_geometry: replacements.pop(0),
    )

    receipt = install_dspark_q3_packed_gate_up_m5(owner)

    assert receipt["candidate"] == "dspark-native-packed-q3-gate-up-m5"
    assert all(stage.ffn.switch_mlp is not None for stage in owner._dspark.stages)
