"""The FP16 sibling of a pack: every packed byte its parent's, every bfloat16 tensor cast as MLX casts it.

Packs here are a few safetensors files written with numpy, so each byte of the parent and the sibling can be
compared; the one MLX call is the cast the sibling must reproduce.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.commands import forge_fp16

ALL_BF16 = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


def _write(path: Path, tensors: dict[str, tuple[str, np.ndarray]], metadata: dict | None = None, pad: int = 0) -> None:
    """A safetensors file: (dtype name, array of that width) per tensor, data in the order given."""

    header: dict = {"__metadata__": metadata} if metadata else {}
    offset, blobs = 0, []
    for name, (dtype, array) in tensors.items():
        blob = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8 + pad)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _tensors(path: Path) -> dict[str, tuple[str, bytes]]:
    length, header = forge_fp16.read_header(path)
    raw = path.read_bytes()[8 + length :]
    return {
        name: (info["dtype"], raw[info["data_offsets"][0] : info["data_offsets"][1]])
        for name, info in header.items()
        if name != "__metadata__"
    }


def _bf16(values) -> np.ndarray:
    """Float values to bfloat16 bit patterns (by truncation: the patterns, not a rounding, are what matter)."""

    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _mlx_cast(bits: np.ndarray, stream) -> np.ndarray:
    with mx.stream(stream):
        cast = mx.array(bits).view(mx.bfloat16).astype(mx.float16).view(mx.uint16)
        mx.eval(cast)
    return np.array(cast)


STREAMS = [
    pytest.param(mx.cpu, id="cpu"),
    pytest.param("gpu", id="gpu", marks=pytest.mark.skipif(not mx.metal.is_available(), reason="no GPU")),
]


@pytest.mark.parametrize("stream", STREAMS)
def test_every_bfloat16_value_is_cast_as_mlx_casts_it(stream):
    """The released siblings were cast by MLX; the sibling's streamed cast must give the same bits for every
    one of the 65,536 bfloat16 patterns (any NaN stays a NaN; its payload is not a value)."""

    ours, stats = forge_fp16.bf16_to_f16(ALL_BF16)
    theirs = _mlx_cast(ALL_BF16, mx.gpu if stream == "gpu" else stream)
    nan = np.isnan(ours.view(np.float16))
    assert np.array_equal(nan, np.isnan(theirs.view(np.float16)))
    assert np.array_equal(ours[~nan], theirs[~nan])
    # What the cast does to the whole range: finite values beyond 65,504 overflow, the smallest round to zero.
    assert stats["elements"] == 1 << 16 and stats["overflow"] > 0 and stats["to_zero"] > 0
    assert stats["exact"] + stats["rounded"] + stats["overflow"] == 1 << 16
    assert stats["not_finite_in_parent"] == 256  # two infinities and 254 NaNs: kept as they are, counted


def test_values_inside_the_float16_range_are_exact_and_the_smallest_are_counted():
    bits = _bf16([0.0, -0.0, 1.0, -2.5, 65280.0, 6.103515625e-05, 1e-6, 1e-9, 3.0e-8])
    half, stats = forge_fp16.bf16_to_f16(bits)

    assert half.view(np.float16)[:6].astype(np.float32).tolist() == [0.0, -0.0, 1.0, -2.5, 65280.0, 6.103515625e-05]
    assert stats["exact"] >= 6 and stats["overflow"] == 0
    assert stats["to_zero"] >= 1  # 1e-9 is below float16's smallest subnormal
    assert 0 < stats["largest_error"] < 6e-8


def _pack(folder: Path, *, big: float = 1.5) -> Path:
    rng = np.random.default_rng(3)
    weights = rng.integers(0, 2**32, size=(16, 4), dtype=np.uint64).astype(np.uint32)
    _write(
        folder / "model-00001-of-00001.safetensors",
        {
            "language_model.model.layers.0.mlp.down_proj.weight": ("U32", weights),
            "language_model.model.layers.0.mlp.down_proj.scales": ("BF16", _bf16(rng.uniform(1e-3, 0.1, (16, 1)))),
            "language_model.model.layers.0.mlp.down_proj.biases": ("BF16", _bf16(rng.uniform(-0.1, 0.1, (16, 1)))),
            "language_model.model.norm.weight": ("BF16", _bf16(np.full(64, big))),
            "language_model.model.layers.0.linear_attn.A_log": ("F32", rng.normal(size=8).astype(np.float32)),
        },
        metadata={"format": "mlx"},
    )
    _write(
        folder / "model-vision.safetensors",
        {"vision_tower.blocks.0.attn.qkv.weight": ("BF16", _bf16(rng.normal(0, 0.02, (8, 8))))},
    )
    _write(
        folder / "ngram_sidecar.safetensors",
        {
            "ngram.weight": ("U32", rng.integers(0, 2**32, size=(40, 4), dtype=np.uint64).astype(np.uint32)),
            "ngram.scales": ("BF16", _bf16(rng.uniform(1e-3, 0.1, (40, 1)))),
            "ngram.biases": ("BF16", _bf16(rng.uniform(-0.1, 0.1, (40, 1)))),
        },
        metadata={"ngram_bits": "4", "ngram_group_size": "32"},
    )
    _write(folder / "mtp" / "weights.safetensors", {"mtp.norm.weight": ("BF16", _bf16(np.ones(64)))})
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen4", "text_config": {"dtype": "bfloat16"}, "vision_config": {}}))
    (folder / "mtplx_runtime.json").write_text(
        json.dumps({"recommended_profile": "turbo", "forge_provenance": {"forge_recipe": {"body_bits": 4}}})
    )
    (folder / "tokenizer.json").write_text("{}")
    return folder


def test_a_sibling_keeps_every_packed_byte_and_casts_every_bfloat16_tensor(tmp_path):
    parent = _pack(tmp_path / "Qwen4-27B-MTPLX-Optimized-Speed")
    sibling = tmp_path / "Qwen4-27B-MTPLX-Optimized-Speed-FP16"

    manifest = forge_fp16.derive(parent, sibling, public_model_id="mtplx-qwen4-27b-optimized-speed-fp16", log=lambda _line: None)

    for relative in ("model-00001-of-00001.safetensors", "model-vision.safetensors", "ngram_sidecar.safetensors", "mtp/weights.safetensors"):
        before, after = _tensors(parent / relative), _tensors(sibling / relative)
        assert sorted(before) == sorted(after)
        for name, (dtype, blob) in before.items():
            new_dtype, new_blob = after[name]
            if dtype == "BF16":
                assert new_dtype == "F16"
                expected = _mlx_cast(np.frombuffer(blob, dtype=np.uint16), mx.cpu)
                assert np.frombuffer(new_blob, dtype=np.uint16).tolist() == expected.tolist(), name
            else:
                assert (new_dtype, new_blob) == (dtype, blob), name  # packed words and float32 state: the parent's bytes
    assert forge_fp16.read_header(sibling / "ngram_sidecar.safetensors")[1]["__metadata__"] == {"ngram_bits": "4", "ngram_group_size": "32"}
    assert forge_fp16.read_header(sibling / "model-vision.safetensors")[0] == forge_fp16.read_header(parent / "model-vision.safetensors")[0]

    assert json.loads((sibling / "config.json").read_text())["text_config"]["dtype"] == "float16"
    stamp = json.loads((sibling / "mtplx_runtime.json").read_text())
    assert stamp["precision_variant"] == "fp16" and stamp["public_model_id"] == "mtplx-qwen4-27b-optimized-speed-fp16"
    assert stamp["forge_provenance"]["forge_recipe"]["body_dtype"] == "fp16"
    assert stamp["recommended_profile"] == "turbo"  # the parent's stamps carry over
    assert (sibling / "tokenizer.json").read_text() == "{}"

    summary = manifest["summary"]
    assert (summary["converted_bf16_to_fp16"], summary["preserved"]) == (7, 3)
    assert summary["values"]["overflow"] == 0
    assert json.loads((sibling / forge_fp16.MANIFEST).read_text())["summary"] == summary
    assert not (tmp_path / "Qwen4-27B-MTPLX-Optimized-Speed-FP16.partial").exists()


def test_a_value_float16_cannot_hold_refuses_the_sibling_and_leaves_no_pack(tmp_path):
    parent = _pack(tmp_path / "parent", big=70000.0)
    sibling = tmp_path / "sibling"

    with pytest.raises(forge_fp16.Fp16SiblingError, match=r"64 value\(s\) of language_model.model.norm.weight are beyond float16's range"):
        forge_fp16.derive(parent, sibling, log=lambda _line: None)

    assert not sibling.exists() and not (tmp_path / "sibling.partial").exists()
    assert _tensors(parent / "model-00001-of-00001.safetensors")["language_model.model.norm.weight"][0] == "BF16"  # the parent untouched


def test_chunks_do_not_change_a_byte(tmp_path):
    parent = _pack(tmp_path / "parent")

    forge_fp16.derive(parent, tmp_path / "whole", log=lambda _line: None)
    forge_fp16.derive(parent, tmp_path / "chunked", chunk=7, log=lambda _line: None)

    for relative in ("model-00001-of-00001.safetensors", "ngram_sidecar.safetensors"):
        assert (tmp_path / "whole" / relative).read_bytes() == (tmp_path / "chunked" / relative).read_bytes()


def test_an_odd_header_and_a_padded_one_keep_their_length(tmp_path):
    """The data start may sit at any offset: the cast reads each tensor at its own byte offset."""

    path = tmp_path / "parent" / "odd.safetensors"
    _write(path, {"a": ("BF16", _bf16([1.0, 2.0, 3.0])), "b": ("U32", np.arange(3, dtype=np.uint32))}, pad=3)
    (tmp_path / "parent" / "config.json").write_text("{}")

    forge_fp16.derive(tmp_path / "parent", tmp_path / "sibling", log=lambda _line: None)

    out = tmp_path / "sibling" / "odd.safetensors"
    assert forge_fp16.read_header(out)[0] == forge_fp16.read_header(path)[0]
    assert _tensors(out)["a"] == ("F16", np.array([1.0, 2.0, 3.0], dtype=np.float16).tobytes())
    assert _tensors(out)["b"] == _tensors(path)["b"]
    with mx.stream(mx.cpu):
        loaded = mx.load(str(out))
    assert loaded["a"].dtype == mx.float16 and loaded["a"].tolist() == [1.0, 2.0, 3.0]


def test_a_destination_that_exists_is_never_written_over(tmp_path):
    parent = _pack(tmp_path / "parent")
    (tmp_path / "sibling").mkdir()

    with pytest.raises(forge_fp16.Fp16SiblingError, match="exists"):
        forge_fp16.derive(parent, tmp_path / "sibling", log=lambda _line: None)


# --------------------------------------------------------------------------- Inkling: Law 1 fails closed
def _inkling_pack(root: Path, *, audited: bool = True, audit_count: int = 3, tower: int = 3) -> Path:
    """A tiny Inkling pack: the publisher's vision_config, `tower` BF16 vision tensors in the real header, packed
    words, a drafter with metadata, its stamp, and (when `audited`) a passing vision audit naming `audit_count`."""
    from mtplx.commands.forge_inkling import REVISION, VISION_CONFIG

    rng = np.random.default_rng(3)
    target = {f"model.visual.blocks.{i}.weight": ("BF16", _bf16(rng.normal(size=(4, 8)))) for i in range(tower)}
    target["model.llm.layers.0.attn.wq_du.weight"] = ("U32", rng.integers(0, 1 << 31, size=(4, 2), dtype=np.uint32))
    _write(root / "model-00001-of-00001.safetensors", target)
    _write(root / "mtp.safetensors", {"blocks.0.w": ("BF16", _bf16(rng.normal(size=(2, 8)))),
                                      "blocks.0.b": ("BF16", _bf16(rng.normal(size=(8,))))}, metadata={"format": "pt"})
    (root / "config.json").write_text(json.dumps({"vision_config": VISION_CONFIG, "text_config": {"dtype": "bfloat16"}}))
    (root / "mtplx-pack.json").write_text(json.dumps({
        "display_name": "Inkling 2 Mini Bare Speed", "recipe": "bare-speed", "source": "thinkingmachines/Inkling-2-Mini",
        "source_revision": REVISION, "weight_bytes": 1_000, "matrix_bit_counts": {"4": 1}, "vision_tensors": tower}))
    if audited:
        (root / "vision-audit.json").write_text(json.dumps({"pack": {"vision_tensors": audit_count}, "problems": [], "pass": True}))
    return root


def test_an_inkling_sibling_carries_the_whole_tower_in_f16_and_says_so(tmp_path):
    from mtplx.commands.forge_inkling import REVISION

    parent = _inkling_pack(tmp_path / "parent")
    forge_fp16.derive(parent, tmp_path / "sibling", log=lambda _line: None)

    sibling = tmp_path / "sibling"
    tower = {name: dtype for name, (dtype, _blob) in _tensors(sibling / "model-00001-of-00001.safetensors").items()
             if name.startswith("model.visual.")}
    assert len(tower) == 3 and set(tower.values()) == {"F16"}
    assert json.loads((sibling / "vision-audit.json").read_text()) == {
        "pass": True, "variant": "fp16", "vision_tensors": 3, "draft_tensors": 2,
        "rule": "every vision and drafter tensor equals the upstream tensor cast BF16 to FP16 (byte for byte where not BF16)",
        "upstream_revision": REVISION}
    stamp = json.loads((sibling / "mtplx-pack.json").read_text())
    assert stamp["precision_variant"] == "fp16" and stamp["display_name"] == "Inkling 2 Mini Bare Speed FP16"


def test_a_sibling_lists_its_own_files_not_its_parents(tmp_path):
    import hashlib

    parent = _inkling_pack(tmp_path / "parent")
    (parent / "SHA256SUMS").write_text("0" * 64 + "  config.json\n")  # the parent's list, stale for any sibling
    forge_fp16.derive(parent, tmp_path / "sibling", log=lambda _line: None)

    sibling = tmp_path / "sibling"
    lines = (sibling / "SHA256SUMS").read_text().splitlines()
    listed = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in lines}
    files = sorted(p.relative_to(sibling).as_posix() for p in sibling.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    assert list(listed) == files
    assert all(hashlib.sha256((sibling / name).read_bytes()).hexdigest() == digest for name, digest in listed.items())


@pytest.mark.parametrize("parent_kwargs,why", [
    ({"audited": False}, "never checked against the upstream checkpoint"),
    ({"audit_count": 3, "tower": 2}, "its audit names 3 in BF16"),
])
def test_an_inkling_parent_without_its_whole_checked_tower_is_refused(tmp_path, parent_kwargs, why):
    parent = _inkling_pack(tmp_path / "parent", **parent_kwargs)
    with pytest.raises(forge_fp16.Fp16SiblingError, match=why):
        forge_fp16.derive(parent, tmp_path / "sibling", log=lambda _line: None)
    assert not (tmp_path / "sibling").exists() and not (tmp_path / "sibling.partial").exists()


def test_mtplx_forge_fp16_is_a_command(tmp_path):
    from mtplx.cli import main

    parent = _inkling_pack(tmp_path / "parent")
    assert main(["forge", "fp16", str(parent), str(tmp_path / "sibling")]) == 0
    assert json.loads((tmp_path / "sibling" / "vision-audit.json").read_text())["variant"] == "fp16"
    assert main(["forge", "fp16", str(_inkling_pack(tmp_path / "blind", audited=False)), str(tmp_path / "blind16")]) != 0
    assert not (tmp_path / "blind16").exists()
