"""The release-day pack tools on what needs no checkpoint: what a config is
recognised as, what the census stops on, and the order of the day's steps."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mtplx.backends import geometry, signature
from mtplx.backends.geometry import Geometry
from mtplx.cli import build_parser
from mtplx.commands import forge_census
from mtplx.commands import forge_day_zero as day_zero

_PLAIN = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "full_attention_interval": 4,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "intermediate_size": 128,
}
_TEMPLATE = "<|im_start|>{{ m }}<|im_end|><think></think><tool_call></tool_call><tool_response>"


def _plain(name: str) -> dict:
    return {"model_type": name, "text_config": dict(_PLAIN, model_type=f"{name}_text")}


def _hyper(name: str) -> dict:
    text = dict(geometry._HYPER_TRUNK_DEFAULTS, model_type=f"{name}_text", ple_layer_ids=[2], ple_embed_dim=64)
    return {"model_type": name, "text_config": text}


def _read(config: dict) -> forge_census.Census:
    census = forge_census.Census(source="a test")
    forge_census._read_config(census, config)
    return census


def _day(out, *kinds: str) -> day_zero.Day:
    packs = tuple(
        day_zero.Pack(kind, {"body_bits": 4, **({"body_dtype": "fp16"} if kind.endswith(day_zero.FP16) else {})})
        for kind in kinds
    )
    return day_zero.Day(source="Owner/Model", out=out, model="Model 1B", packs=packs)


# ---------------------------------------------------------------- signature


def test_a_family_is_recognised_by_its_name_and_by_its_contents():
    for name, named in (("qwen4_exp", True), ("qwen4", False)):
        found = signature.read(_hyper(name))
        assert (found.family, found.named) == (signature.QWEN4, named)
        assert signature.HYPER_CONNECTIONS in found.blocks and not found.unread
    found = signature.read(_plain("a_new_name"))
    assert (found.family, found.named) == (signature.QWEN3_5, False)
    assert signature.read({"model_type": "llama", "hidden_size": 8}) is None


def test_only_a_plain_trunk_is_written_under_the_familys_name():
    renamed = signature.under_own_name(_plain("a_new_name"))
    assert renamed["model_type"] == signature.QWEN3_5
    assert "a_new_name" in renamed[signature.UPSTREAM_NAMES_KEY]
    assert signature.renamed_plain_trunk(_plain("qwen3_5")) is None
    assert signature.under_own_name(_hyper("qwen4"))["model_type"] == "qwen4"


def test_geometry_reads_the_sizes_a_config_states():
    plain = Geometry.from_config(_plain("qwen3_5"))
    assert (plain.width, plain.layers, plain.streams) == (64, 4, 1)
    assert plain.experts is None and plain.ngram is None
    hyper = Geometry.from_config(_hyper("qwen4_exp"))
    assert hyper.streams == 4 and hyper.experts is not None and hyper.ngram.layers == (1,)


# -------------------------------------------------------------------- census


def test_the_census_stops_on_a_new_name_this_build_cannot_load():
    assert not _read(_hyper("qwen4_exp")).stops
    stops = _read(_hyper("qwen4")).stops
    assert len(stops) == 1 and "hyper-connection trunk" in stops[0] and "would not load" in stops[0]
    # A plain trunk under a new name is built under the family's own name, which this build loads.
    assert not _read(_plain("a_new_name")).stops


def test_the_census_stops_on_a_key_nothing_reads():
    stops = _read(dict(_plain("qwen3_5"), an_unknown_block={"size": 4})).stops
    assert any("an_unknown_block" in stop for stop in stops)


def test_a_folder_without_a_config_is_a_stop(tmp_path):
    census = forge_census.take(tmp_path)
    assert not census.go and "config.json" in census.stops[0]
    assert "STOP" in census.lines()[-2]
    assert not {"candidate", "differences", "lane_plan"} & set(census.to_dict())


def test_the_census_checks_the_chat_template():
    missing = forge_census.Census(source="a test")
    forge_census._check_template(missing, None, None)
    assert "no chat template ships" in missing.stops[0]
    whole = forge_census.Census(source="a test")
    forge_census._check_template(whole, _TEMPLATE, None)
    assert not whole.stops and "every marker" in whole.template["summary"]
    partial = forge_census.Census(source="a test")
    forge_census._check_template(partial, _TEMPLATE.replace("<tool_response>", ""), None)
    assert any("<tool_response>" in stop for stop in partial.stops)


def test_the_census_command_exits_one_on_a_stop(tmp_path, capsys):
    args = SimpleNamespace(source=str(tmp_path), json=True, reference_template=None)
    assert day_zero.cmd_census(args) == 1
    assert json.loads(capsys.readouterr().out)["go"] is False


# ------------------------------------------------------------------ day zero


def test_a_pack_has_no_gate_and_its_fp16_sibling_is_compared_with_it(tmp_path):
    names = [step.name for step in day_zero.steps(_day(tmp_path, "Optimized Speed", "Optimized Speed FP16"))]
    assert names == [
        "census",
        "download",
        "build/optimized-speed",
        "vision/optimized-speed",
        "heads/optimized-speed",
        "rows/optimized-speed",
        "build/optimized-speed-fp16",
        "vision/optimized-speed-fp16",
        "heads/optimized-speed-fp16",
        "gate/optimized-speed-fp16",
        "rows/optimized-speed-fp16",
        "texts",
        "publish",
    ]


def test_the_plan_names_every_step_and_runs_none(tmp_path):
    day, said = _day(tmp_path / "out", "Optimized Speed"), []
    assert day_zero.execute(day, plan_only=True, say=said.append) == 0
    assert len(said) == len(day_zero.steps(day))
    assert not (tmp_path / "out" / "receipts").exists()


def test_a_census_stop_ends_the_day_at_its_first_step(tmp_path):
    (tmp_path / "empty").mkdir()
    day = day_zero.Day(source=str(tmp_path / "empty"), out=tmp_path / "out", model="Model 1B", packs=_day(tmp_path, "Optimized Speed").packs)
    said = []
    assert day_zero.execute(day, say=said.append) == 1
    assert (tmp_path / "out" / "census.txt").exists() and not day.receipt("census").exists()
    assert any("STOP" in line for line in said)


def test_the_lines_a_person_runs_name_the_packs(tmp_path):
    day = _day(tmp_path, "Optimized Speed", "Optimized Speed FP16")
    parent, sibling = day.packs
    day.results["build/optimized-speed"] = {"path": "/packs/parent", "recommended_profile": "turbo"}
    day.results["build/optimized-speed-fp16"] = {"path": "/packs/sibling"}
    gate = day_zero._gate_line(day, sibling, parent)
    assert "--parent /packs/parent" in gate and "--sibling /packs/sibling" in gate
    assert str(day_zero._parent_check_file(day, sibling)) in gate
    rows = day_zero._rows_line(day, parent)
    assert "served turbo" in rows and "sustained profile" in rows
    assert str(day_zero._handoff_file(day, "rows", parent)) in rows


def test_a_hyper_connection_trunk_gets_three_packs_and_their_siblings():
    packs = day_zero.packs_for(signature.QWEN4, None)
    kinds = [day_zero.OPTIMIZED_SPEED, day_zero.BARE_SPEED, day_zero.OPTIMIZED_QUALITY]
    assert [pack.kind for pack in packs] == kinds + [kind + day_zero.FP16 for kind in kinds]
    assert [pack.half_floats for pack in packs] == [False] * 3 + [True] * 3
    with pytest.raises(day_zero.Stop, match="no recipe of its own"):
        day_zero.packs_for(signature.QWEN3_5, None)


def test_the_day_stops_before_writing_when_the_disk_is_short(tmp_path, monkeypatch):
    day = _day(tmp_path, "Optimized Speed")
    monkeypatch.setattr(day_zero.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 2**30))
    with pytest.raises(day_zero.Stop, match="GiB free"):
        day_zero._require_disk(day, source_bytes=50 * 2**30, to_fetch=50 * 2**30)


def test_the_cli_takes_both_commands():
    parser = build_parser()
    census = parser.parse_args(["forge", "census", "Owner/Model", "--json"])
    assert (census.forge_action, census.source, census.json) == ("census", "Owner/Model", True)
    day = parser.parse_args(["forge", "day-zero", "Owner/Model", "--out", "/o", "--model", "Model 1B", "--plan"])
    assert (day.forge_action, day.model, day.plan) == ("day-zero", "Model 1B", True)
