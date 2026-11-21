"""The Inkling forge fails closed on the vision configuration (Law 1): a source or a pack whose config.json lacks the
publisher's pinned `vision_config`, or carries another one, is never staged, and two missing values never pass."""

import json
from pathlib import Path

import pytest

from mtplx.commands import forge_inkling, forge_inkling_quant
from mtplx.commands.forge_inkling import VISION_CONFIG, stage_native_pack, verify_source, vision_config_problem


def _config(vision="pinned"):
    config = {"model_type": "inkling2_loop", "text_config": {"loop_config": {"is_varloop": False}}}
    if vision == "pinned":
        config["vision_config"] = dict(VISION_CONFIG)
    elif vision is not None:
        config["vision_config"] = vision
    return config


def _source(tmp_path: Path, vision) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_config(vision)))
    return source


def test_the_pinned_vision_configuration_is_the_publishers():
    assert vision_config_problem(_config(), "source") is None
    assert VISION_CONFIG["model_type"] == "inkling2_vision" and VISION_CONFIG["num_hidden_layers"] == 23


@pytest.mark.parametrize("vision", [None, {}, "not a dict"])
def test_a_source_without_the_vision_configuration_is_refused_before_any_weight_is_read(tmp_path, vision):
    # No weight file exists in this source: the refusal must come from config.json alone.
    with pytest.raises(ValueError, match="vision_config"):
        verify_source(_source(tmp_path, vision))


def test_a_changed_vision_configuration_is_refused_and_named(tmp_path):
    changed = dict(VISION_CONFIG, num_hidden_layers=22)
    with pytest.raises(ValueError, match=r"differs .*num_hidden_layers"):
        verify_source(_source(tmp_path, changed))
    extra = dict(VISION_CONFIG, spare=1)
    assert "spare" in vision_config_problem(_config(extra), "pack")


def test_native_quality_staging_refuses_a_source_without_it_and_leaves_nothing(tmp_path):
    destination = tmp_path / "packs" / "quality"
    with pytest.raises(ValueError, match="vision_config"):
        stage_native_pack(_source(tmp_path, None), destination, recipe="quality")
    assert not destination.exists()


def test_the_quantized_forge_checks_the_source_first(tmp_path):
    with pytest.raises(ValueError, match="vision_config"):
        forge_inkling_quant.build_pack(_source(tmp_path, None), tmp_path / "packs" / "bare", "bare-speed")
    assert not (tmp_path / "packs" / "bare").exists()


def test_two_missing_vision_configurations_never_count_as_equal():
    problems = [vision_config_problem(c, w) for c, w in ((_config(None), "source"), (_config(None), "pack"))]
    assert all(problems)
    assert forge_inkling_quant.vision_config_problem is forge_inkling.vision_config_problem
