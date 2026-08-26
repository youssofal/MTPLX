import inspect

import pytest


def test_quality_cold_matrix_has_closed_published_controls():
    from scripts import qwen38_dflash2_quality_cold_matrix as matrix

    args = matrix.parse_args(["--output-dir", "receipts"])

    assert args.repetitions == 3
    assert matrix.QUALITY_MODEL == (
        "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"
    )
    assert matrix.QUALITY_REVISION == (
        "09f71b39a75c416be3c974840b53f9fbe9aa1841"
    )
    assert matrix.PHYSICAL_WIDTHS == tuple(range(2, 9))
    assert matrix.COLD_PREFIX_TOKENS == (1024, 16384, 65536)
    assert matrix.TEST_PROMPT_TOKENS == 1024
    assert matrix.OUTPUT_TOKENS == 1024

    with pytest.raises(SystemExit):
        matrix.parse_args(
            ["--output-dir", "receipts", "--repetitions", "2"]
        )


def test_quality_cold_matrix_attests_before_importing_mlx_runner():
    from scripts import qwen38_dflash2_quality_cold_matrix as matrix

    source = inspect.getsource(matrix.main)
    assert source.index("issue_guard_window") < source.index(
        "dflash2_depth_sweep"
    )
    assert "/tmp/mtplx-gpu-exclusive.lock" in source
