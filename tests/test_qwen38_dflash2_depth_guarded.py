import inspect
import json
from pathlib import Path

import pytest


def test_guarded_parser_has_only_closed_benchmark_controls():
    from scripts import qwen38_dflash2_depth_guarded as guarded

    args = guarded.parse_args(
        [
            "--model",
            "speed",
            "--draft-model",
            "draft",
            "--output",
            "result.json",
        ]
    )
    assert args.widths == "1,2,3,4,5,6,7,8"
    assert args.repetitions == 3
    assert args.smoke_tokens is None
    with pytest.raises(SystemExit):
        guarded.parse_args(
            [
                "--model",
                "speed",
                "--draft-model",
                "draft",
                "--smoke-tokens",
                "16",
                "--output",
                "result.json",
            ]
        )


def test_guarded_script_verifies_before_importing_mlx_runner():
    from scripts import qwen38_dflash2_depth_guarded as guarded

    source = inspect.getsource(guarded.main)
    assert source.index("issue_guard_window") < source.index("dflash2_depth_sweep")
    assert "/tmp/mtplx-gpu-exclusive.lock" in source


def test_atomic_json_replaces_destination(tmp_path: Path):
    from scripts import qwen38_dflash2_depth_guarded as guarded

    output = tmp_path / "nested" / "receipt.json"
    guarded.write_atomic_json(output, {"selection": {"best_widths": [8]}})

    assert json.loads(output.read_text()) == {"selection": {"best_widths": [8]}}
    assert list(output.parent.iterdir()) == [output]
