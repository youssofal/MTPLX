from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from mtplx.edge0.gguf_lite import GGUFError, GGUFFile
from tests import edge0_synth as syn

REAL = os.environ.get("MTPLX_EDGE0_MODEL")
real_only = pytest.mark.skipif(not REAL, reason="MTPLX_EDGE0_MODEL not set")


def _sample(tmp_path: Path, **extra):
    rng = np.random.default_rng(0)
    f32 = rng.standard_normal((3, 4)).astype(np.float32)
    f16 = rng.standard_normal((2, 5)).astype(np.float16)
    bf = np.array([[0x3F80, 0xC000], [0x3F00, 0x0000]], "<u2")
    q = rng.integers(-100, 100, (2, 32), dtype=np.int8)
    d = np.array([0.5, 0.25], np.float16)
    q8 = syn.pack_q8_0(q, d)
    meta = {
        "general.architecture": "toy",
        "toy.n": 7,
        "toy.big": 2**40,
        "toy.f": 1.5,
        "toy.flag": True,
        "toy.name": "hello",
        "toy.list": [1, 2, 3],
        "toy.strs": ["a", "bc"],
        **extra,
    }
    path = tmp_path / "t.gguf"
    syn.write_gguf(
        path,
        meta,
        [
            ("a.f32", f32.shape, 0, f32.tobytes()),
            ("a.f16", f16.shape, 1, f16.tobytes()),
            ("a.bf16", bf.shape, 30, bf.tobytes()),
            ("a.q8", q.shape, 8, q8),
            ("a.q4", (2, 32), 2, bytes(36)),
        ],
    )
    return path, f32, f16, q, d


def test_metadata_and_tensor_table(tmp_path):
    path, f32, *_ = _sample(tmp_path)
    g = GGUFFile(path)
    assert g.version == 3
    assert g.metadata["toy.n"] == 7 and g.metadata["toy.big"] == 2**40
    assert g.metadata["toy.flag"] is True and g.metadata["toy.name"] == "hello"
    assert g.metadata["toy.list"] == [1, 2, 3] and g.metadata["toy.strs"] == ["a", "bc"]
    t = g.tensors["a.f32"]
    assert t.shape == (3, 4) and t.ggml_type == 0 and t.nbytes == 48
    assert t.offset % 32 == 0 and g.tensors["a.q8"].nbytes == 2 * 34
    assert g.tensors["a.q4"].nbytes == 36


def test_raw_and_array(tmp_path):
    path, f32, f16, q, d = _sample(tmp_path)
    g = GGUFFile(path)
    raw = g.raw("a.f32")
    assert isinstance(raw, np.memmap) and raw.dtype == np.uint8 and raw.size == 48
    np.testing.assert_array_equal(g.array("a.f32"), f32)
    np.testing.assert_array_equal(g.array("a.f16"), f16.astype(np.float32))
    assert g.array("a.bf16").tolist() == [[1.0, -2.0], [0.5, 0.0]]
    expect = q.astype(np.float32) * d.astype(np.float32)[:, None]
    np.testing.assert_array_equal(g.array("a.q8"), expect)
    assert g.array("a.q8").shape == (2, 32)


def test_unsupported_type_and_missing_name(tmp_path):
    path, *_ = _sample(tmp_path)
    g = GGUFFile(path)
    with pytest.raises(GGUFError, match="Q8_0"):
        g.array("a.q4")
    with pytest.raises(KeyError):
        g.raw("nope")


def test_custom_alignment(tmp_path):
    path, f32, *_ = _sample(tmp_path, **{"general.alignment": 64})
    g = GGUFFile(path)
    assert g.alignment == 64 and g.tensors["a.f32"].offset % 64 == 0
    np.testing.assert_array_equal(g.array("a.f32"), f32)


def test_hand_built_header(tmp_path):
    # one F32 tensor of dims [3, 2] (innermost first), no metadata, data at offset 32 * ceil(header / 32)
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    head += struct.pack("<Q", 1) + b"x" + struct.pack("<I", 2) + struct.pack("<QQ", 3, 2) + struct.pack("<IQ", 0, 0)
    head += b"\0" * (-len(head) % 32)
    data = np.arange(6, dtype=np.float32)
    p = tmp_path / "h.gguf"
    p.write_bytes(head + data.tobytes())
    g = GGUFFile(p)
    assert g.tensors["x"].shape == (2, 3) and g.tensors["x"].offset == len(head)
    assert g.array("x").tolist() == [[0, 1, 2], [3, 4, 5]]


def test_errors(tmp_path):
    path, *_ = _sample(tmp_path)
    blob = path.read_bytes()
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + blob[4:])
    with pytest.raises(GGUFError, match="magic"):
        GGUFFile(bad)
    bad.write_bytes(blob[:4] + struct.pack("<I", 2) + blob[8:])
    with pytest.raises(GGUFError, match="version"):
        GGUFFile(bad)
    bad.write_bytes(blob[:60])
    with pytest.raises(GGUFError, match="truncated"):
        GGUFFile(bad)
    t = GGUFFile(path).tensors["a.q8"]
    bad.write_bytes(blob[: t.offset + 10])  # header intact, data cut short
    with pytest.raises(GGUFError, match="needs bytes"):
        GGUFFile(bad)


def test_open_is_lazy(tmp_path):
    path, *_ = _sample(tmp_path)
    g = GGUFFile(path)
    path.write_bytes(path.read_bytes()[: g.tensors["a.f32"].offset + 48])  # shrink after open is irrelevant to header
    assert g.tensors["a.f32"].nbytes == 48


@real_only
def test_real_skeleton_and_ple():
    root = Path(REAL)
    g = GGUFFile(root / "model-skeleton.gguf")
    assert len(g.tensors) == 1079 and len(g.metadata) == 57
    assert g.metadata["general.architecture"] == "qwen4exp"
    assert g.metadata["qwen4exp.embedding_length"] == 2560
    assert {t.ggml_type for t in g.tensors.values()} == {0, 1, 8, 30}
    router = g.tensors["blk.0.ffn_gate_inp.weight"]
    assert router.shape == (512, 2560) and router.ggml_type == 30
    arr = g.array("blk.0.ffn_gate_inp.weight")
    assert arr.shape == (512, 2560) and np.isfinite(arr).all() and 0 < arr.std() < 10
    q8 = g.array("blk.0.ffn_up_shexp.weight")
    assert q8.shape == (640, 2560) and np.isfinite(q8).all()
    ple = GGUFFile(root / "ple-fp8.gguf")
    (t,) = ple.tensors.values()
    assert t.ggml_type == 24 and t.shape == (320001536, 160)
    assert ple.metadata["strata.ple.scale"] == pytest.approx(0.00019931793212890625)
    assert ple.raw(t.name)[:16].shape == (16,)
