from importlib.metadata import version
from pathlib import Path
import tomllib


DFLASH_MLX_PIN = (
    "dflash-mlx @ "
    "git+https://github.com/davidtai/dflash-mlx.git@"
    "95a16104c5462793f9f14d2cd0b97b856fb643a1"
)


def test_competitor_extra_pins_immutable_dflash_mlx_source():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["competitors"] == [DFLASH_MLX_PIN]


def test_dflash2_runtime_api_contract():
    from dflash_mlx.draft.dflash2 import DFlash2DraftModel
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    from dflash_mlx.runtime import stream_dflash_generate
    from dflash_mlx.runtime.loading import load_draft_bundle

    assert version("dflash-mlx") == "0.1.10"
    assert DFlash2DraftModel.__name__ == "DFlash2DraftModel"
    assert QwenGdnTargetOps.backend_name == "qwen_gdn"
    assert callable(stream_dflash_generate)
    assert callable(load_draft_bundle)
