"""The one reader of the installed MLX's kernel headers (mtplx/kernels/library_headers.py): the same parts give
the same text, each header once across parts; a package without headers is refused with its reason, counted."""

import pytest

pytest.importorskip("mlx.core")

from mtplx.kernels import library_headers as LH  # noqa: E402


def test_the_same_parts_give_the_same_text_with_each_header_once():
    one, why = LH.text("tensor_tiles", "tensor_gemm")
    if one is None:  # pragma: no cover - an MLX build without the tensor-unit headers
        pytest.skip(why)
    again, _ = LH.read(("tensor_tiles", "tensor_gemm"), LH._root())
    assert again == one
    # the GEMM loop's header includes the tile header: asking for both carries the tiles once
    assert one.count("struct BaseNAXFrag") == 1
    alone, _ = LH.text("tensor_gemm")
    assert alone.count("struct BaseNAXFrag") == 1 and "gemm_loop(" in alone
    # what every custom kernel already starts with (utils.h and its includes) is left out
    assert "inline uint16_t bfloat16_to_uint16(" not in one


def test_a_package_without_headers_is_refused_and_counted(monkeypatch, tmp_path):
    monkeypatch.setattr(LH, "_root", lambda: tmp_path)
    LH.text.cache_clear()
    before = LH.refusals().get("missing:utils.h", 0)
    try:
        body, why = LH.text("tensor_gemm")
    finally:
        LH.text.cache_clear()
    assert body is None and why == "missing:utils.h"
    assert LH.refusals()["missing:utils.h"] == before + 1
    assert LH.read(("no_such_part",), tmp_path) == (None, "unknown_part:no_such_part")


def test_the_gpu_generation_is_a_number():
    assert isinstance(LH.generation(), int) and LH.generation() >= 0
