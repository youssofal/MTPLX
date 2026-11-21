"""The census by block: each launch placed in its layer and block by the weights it binds.

Plain data only: a fabricated program, its pin names, profile and form timings
as the site writes them with ``MTPLX_NATIVE_ENGINE_PROFILE=forms``. No GPU.
"""

from __future__ import annotations

import json

from mtplx.engine.ir import ArrayIR, BufferIR, ProgramIR, StepIR
from mtplx.engine.tools import census

_P = "language_model.model."


def _program() -> ProgramIR:
    # Buffers: 0 the token input, 1 a temporary, 10.. pinned weights (slot = index).
    buffers = [BufferIR(0, "input", 64, index=0), BufferIR(1, "temp", 4096)]
    names = {
        0: [_P + "embed_tokens.weight"],
        1: [_P + "layers.0.input_layernorm.weight"],
        2: [_P + "layers.0.linear_attn.in_proj_qkv.weight"],
        3: [_P + "layers.0.post_attention_layernorm.weight"],
        4: [_P + "layers.0.mlp.switch_mlp.gate_proj.weight"],
        5: [_P + "layers.1.input_layernorm.weight"],
        6: [_P + "layers.1.self_attn.q_proj.weight"],
        7: [_P + "layers.1.mlp.gate.weight"],
        8: [_P + "norm.weight"],
        9: ["language_model.lm_head.weight"],
    }
    for slot in names:
        buffers.append(BufferIR(10 + slot, "pinned", 1024, index=slot))

    def step(name: str, *slots: int) -> StepIR:
        bindings = [(0, 1, 0, "w")] + [(i + 1, 10 + s, 0, "r") for i, s in enumerate(slots)]
        return StepIR(kernel=0, kernel_name=name, bindings=bindings, inline=[],
                      dispatch=("groups", (1, 1, 1), (32, 1, 1)))

    steps = [
        step("arange_int32"),                  # prelude
        step("gather_front", 0),                # embedding
        step("rms_bfloat16", 1),                # layer 0 input norm -> recurrent
        step("affine_qmv_fast", 2),             # recurrent product
        step("gated_delta_step"),               # recurrent, no weight: carried
        step("rms_bfloat16", 3),                # layer 0 post-attention norm -> experts
        step("gather_qmv_fast", 4),             # experts
        step("vv_Add"),                         # experts, carried
        step("rms_bfloat16", 5),                # layer 1 input norm -> attention
        step("affine_qmv_fast", 6),             # attention
        step("sdpa_vector"),                    # attention, carried
        step("affine_qmv_fast", 7),             # layer 1 router -> experts
        step("rms_bfloat16", 8),                # final norm -> head
        step("affine_qmv_fast", 9),             # output head
        step("copy_bfloat16"),                  # a state write after the head -> tail
    ]
    ir = ProgramIR(name="qwen35.verify", buffers=buffers, steps=steps,
                   inputs=[ArrayIR(buffer=0, offset=0, shape=(1, 2), dtype="int32", nbytes=8)])
    ir._names = names  # for the test only
    return ir


def test_attribute_places_every_launch():
    ir = _program()
    pins = {str(k): v for k, v in ir._names.items()}
    places = [(layer, block) for layer, block, _ in census.attribute(ir, pins)]
    assert places == [
        (None, "prelude"), (None, "embedding"),
        (0, "recurrent"), (0, "recurrent"), (0, "recurrent"),
        (0, "experts"), (0, "experts"), (0, "experts"),
        (1, "attention"), (1, "attention"), (1, "attention"),
        (1, "experts"),
        (None, "head"), (None, "head"),
        (None, "tail"),
    ]


def test_blocks_report_reads_the_site_files(tmp_path):
    ir = _program()
    stem = "qwen35_verify-abc123"
    (tmp_path / f"{stem}.ir.json").write_text(ir.to_json())
    (tmp_path / f"{stem}.pins.json").write_text(json.dumps({"slots": {str(k): v for k, v in ir._names.items()}}))
    (tmp_path / f"{stem}.profile.json").write_text(json.dumps({
        "site": "qwen35.verify", "steps": [{"step": i, "gpu_ns": 1000 * (i + 1)} for i in range(len(ir.steps))]}))
    form = lambda d, b, s, g: {"dispatch": d, "barriers": b, "split": s,  # noqa: E731
                               "gpu_ns": {"min": g, "median": g, "max": g}, "wall_ns": {"min": g, "median": g, "max": g}}
    (tmp_path / f"{stem}.forms.json").write_text(json.dumps({"forms": [
        form("hybrid", "minimal", 256, 7_000_000), form("concurrent", "none", 0, 5_500_000)]}))
    report = census.blocks_report(tmp_path)
    (prog,) = report["programs"]
    assert prog["width"] == 2
    assert prog["launches"] == 15
    assert prog["by_block"]["recurrent"]["launches"] == 3
    assert prog["by_block"]["recurrent"]["launches_binding_a_weight"] == 2
    assert prog["by_block"]["experts"]["launches"] == 4
    assert prog["by_block"]["recurrent"]["gpu_ns"] == 3000 + 4000 + 5000
    assert prog["per_layer_mean"]["recurrent_mixer_launches"] == 3
    assert prog["per_layer_mean"]["attention_mixer_launches"] == 3
    assert prog["per_layer_mean"]["expert_block_launches"] == 2  # (3 + 1) / 2 layers
    assert prog["barrier_free_floor"]["most_fusion_could_save_ns"] == 1_500_000
    text = census.render_blocks(report)
    assert "Most that fusing dependent launches could save: 1.500 ms" in text


def test_draft_head_weights_are_their_own_block():
    assert census._param_block("mtp.layers.0.mlp.switch_mlp.up_proj.weight", set()) == (None, "draft")
    assert census._param_block(_P + "layers.3.input_layernorm.weight", {3}) == (3, "recurrent")
    assert census._param_block(_P + "layers.3.input_layernorm.weight", set()) == (3, "attention")


def test_each_setting_is_read_with_the_other_two_held():
    """Barriers, dispatch form, commit split and the zero fill each get their
    own figure (the old floor folded all three into one difference)."""

    def row(d, b, s, g, fill=0):
        return {"dispatch": d, "barriers": b, "split": s, "fill": fill,
                "gpu_ns": {"min": g, "median": g, "max": g}, "wall_ns": {"min": g, "median": g, "max": g}}

    rows = [
        row("hybrid", "minimal", 256, 8_000),
        row("hybrid", "minimal", 0, 8_300),
        row("concurrent", "minimal", 256, 7_700),
        row("concurrent", "minimal", 0, 7_900),
        row("concurrent", "none", 256, 4_400),
        row("concurrent", "none", 0, 4_500),
        row("hybrid", "minimal", 256, 7_950, fill=None),
        {"dispatch": "serial", "barriers": "minimal", "split": 0, "fill": 0, "declined": "x"},
    ]
    effects = census.form_effects(rows)
    assert effects["barriers"] == 3_300
    assert effects["barriers_one_buffer"] == 3_400
    assert effects["dispatch"] == 300
    assert effects["split"] == -300
    assert effects["fill"] == 50
    assert effects["serving_gpu_ns"] == 8_000 and effects["rows_filled"] is True
    # A census from before the fill compares its plain rows.
    plain = [{**r, "fill": None} for r in rows[:6]]
    old = census.form_effects([{k: v for k, v in r.items() if k != "fill"} for r in plain])
    assert old["barriers"] == 3_300 and old["fill"] is None and old["rows_filled"] is False
