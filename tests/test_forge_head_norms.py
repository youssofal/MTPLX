"""Day zero's draft-head norm audit: every gain the loader hands the model is upstream's stored value plus one.

The failing case is the 35B-A3B Speed and Balance packs' own defect: six head gains shifted, the
``post_attention_layernorm`` stored zero-centred, which a whole-file check passes. The files here are real
safetensors written with numpy (no model, no GPU); the last test reads the shipped pack itself when it is on disk.
"""

from __future__ import annotations

import inspect
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mtplx.commands import forge_head_norms as audit

#: Upstream's stored (zero-centred) means of the 35B-A3B head, from its checkpoint.
UPSTREAM_MEANS = {
    "mtp.layers.0.input_layernorm.weight": -0.095,
    "mtp.layers.0.post_attention_layernorm.weight": -0.1314,
    "mtp.layers.0.self_attn.q_norm.weight": 0.767,
    "mtp.layers.0.self_attn.k_norm.weight": 0.742,
    "mtp.norm.weight": 1.925,
    "mtp.pre_fc_norm_hidden.weight": -0.506,
    "mtp.pre_fc_norm_embedding.weight": -0.727,
}


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 to bfloat16 bits, rounding to nearest with ties to even (MLX's cast)."""

    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def _write(path: Path, tensors: dict[str, np.ndarray], *, dtype: str = "BF16") -> None:
    header, blobs, offset = {}, [], 0
    for name, values in tensors.items():
        raw = (_bf16_bits(values) if dtype == "BF16" else values.astype(np.float16)).tobytes()
        header[name] = {"dtype": dtype, "shape": list(values.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _upstream(tmp_path: Path, channels: int = 64) -> tuple[Path, dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    values = {}
    for key, mean in UPSTREAM_MEANS.items():
        spread = rng.normal(0.0, 0.2, channels).astype(np.float32)
        spread[0] = -1.0  # a channel below -1: zero-centred, the shape the loader's repair looks for
        exact = (mean + spread - spread.mean()).astype(np.float32)
        # As a publisher stores them: bfloat16, so the pack's +1 is taken on the stored values, as a converter does.
        values[key] = (_bf16_bits(exact).astype(np.uint32) << 16).view(np.float32)
    # Upstream keeps the head beside the trunk, under the checkpoint's own prefix.
    _write(tmp_path / "upstream" / "model-00001-of-00001.safetensors",
           {"model.language_model." + key: value for key, value in values.items()})
    return tmp_path / "upstream", values


def _pack(tmp_path: Path, values: dict[str, np.ndarray], shifted, *, dtype: str = "BF16", name: str = "pack") -> Path:
    stored = {key: value + (1.0 if shifted(key) else 0.0) for key, value in values.items()}
    _write(tmp_path / name / "mtp.safetensors", stored, dtype=dtype)
    return tmp_path / name


def test_the_35b_defect_is_found_tensor_by_tensor_and_fails_closed(tmp_path):
    upstream, values = _upstream(tmp_path)
    pack = _pack(tmp_path, values, lambda key: "post_attention" not in key)

    result = audit.audit(pack, upstream, family="qwen3_5_moe")

    assert result["sidecar_read_as"] == "stored"  # q/k read absolute: the set-level verdict shifts nothing
    assert result["tensors"] == 7 and not result["ok"]
    assert [row["tensor"] for row in result["rows"] if not row["ok"]] == ["mtp.layers.0.post_attention_layernorm.weight"]
    assert len(result["problems"]) == 1 and "the +1 is missing" in result["problems"][0]
    with pytest.raises(audit.HeadNormMismatch, match="post_attention_layernorm"):
        audit.check(pack, upstream, family="qwen3_5_moe")


def test_a_pack_with_every_gain_shifted_passes(tmp_path):
    upstream, values = _upstream(tmp_path)
    result = audit.check(_pack(tmp_path, values, lambda key: True), upstream, family="qwen3_5_moe")

    assert result["ok"] and result["tensors"] == 7
    assert max(row["largest_error"] for row in result["rows"]) < 0.02


def test_a_zero_centred_sidecar_is_read_with_the_loaders_shift(tmp_path):
    """The other families' loader adds +1 to all seven gains of a delta sidecar: stored as upstream is right."""

    upstream, values = _upstream(tmp_path)
    result = audit.check(_pack(tmp_path, values, lambda key: False), upstream, family="qwen3_5_moe")

    assert result["ok"] and result["sidecar_read_as"] == "zero-centred (+1 at load)"


def test_the_qwen4_head_follows_its_own_convention_and_a_wrong_shift_stops(tmp_path):
    """Qwen 4's upstream stores only q, k and the stream norms (shifted by the converter) and the pre-fc gains
    (shifted at load) zero-centred; the head's input, post-attention and final norms are absolute."""

    upstream, values = _upstream(tmp_path)
    zero = audit.qwen4_zero_centred()
    by_the_converter = lambda key: key.endswith(zero) and not key.endswith(audit.QWEN4_LOAD_SHIFTED)  # noqa: E731

    right = audit.check(_pack(tmp_path, values, by_the_converter), upstream, family=audit.QWEN4)
    assert right["ok"]
    conventions = {row["tensor"]: row["upstream_convention"] for row in right["rows"]}
    assert conventions["mtp.layers.0.input_layernorm.weight"] == "absolute (w)"
    assert conventions["mtp.layers.0.self_attn.q_norm.weight"] == "zero-centred (1 + w)"

    # Stored the other families' way (every gain shifted): five gains are then one +1 too many.
    wrong = audit.audit(_pack(tmp_path, values, lambda key: True, name="all"), upstream, family=audit.QWEN4)
    assert sorted(row["tensor"] for row in wrong["rows"] if not row["ok"]) == [
        "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight", "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
    ]
    assert all("one +1 too many" in problem for problem in wrong["problems"])


def test_a_missing_or_reshaped_gain_stops(tmp_path):
    upstream, values = _upstream(tmp_path)
    partial = {key: value for key, value in values.items() if "k_norm" not in key}
    partial["mtp.norm.weight"] = values["mtp.norm.weight"][:32]
    pack = _pack(tmp_path, partial, lambda key: True)

    problems = audit.audit(pack, upstream, family="qwen3_5_moe")["problems"]

    assert any("k_norm" in p and "upstream has it and the pack does not" in p for p in problems)
    assert any(p.startswith("mtp.norm.weight: [32] in the pack, [64] upstream") for p in problems)


def test_an_fp16_sibling_is_checked_at_float16_resolution(tmp_path):
    upstream, values = _upstream(tmp_path)
    # The sibling's gains are its parent's bfloat16 values cast to float16 (exact: bfloat16 fits float16).
    parent = {key: (np.frombuffer(_bf16_bits(value + 1.0).tobytes(), dtype=np.uint16).astype(np.uint32) << 16)
              .view(np.float32) for key, value in values.items()}
    _write(tmp_path / "sibling" / "mtp.safetensors", parent, dtype="F16")

    assert audit.check(tmp_path / "sibling", upstream, family="qwen3_5_moe")["ok"]


def test_a_headers_only_source_is_named_as_such(tmp_path):
    def checkpoint(name, metadata):
        folder = tmp_path / name
        folder.mkdir()
        header = json.dumps({"__metadata__": metadata}).encode()
        (folder / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
        return folder

    assert audit.headers_only(checkpoint("synthetic", {audit.SYNTHETIC_MARKER: "1"}))
    assert not audit.headers_only(checkpoint("published", {"format": "pt"}))


def test_the_qwen4_rule_is_the_model_codes():
    """``QWEN4_LOAD_SHIFTED`` is what ``attach_mtp`` shifts at load, and the converter's head shifts are the
    model's own sanitize list: the audit reads the convention the product runs on."""

    from mtplx.commands import forge_qwen4_exp
    from mtplx.models import qwen4_exp

    source = inspect.getsource(qwen4_exp.Model.attach_mtp)
    assert all(f'"{name}"' in source for name in audit.QWEN4_LOAD_SHIFTED)
    assert "+ 1.0" in source
    assert set(forge_qwen4_exp.MTP_NORM_SHIFT_SUFFIXES) == set(qwen4_exp.Model._HF_NORM_SHIFT_SUFFIXES)
    assert not any(name.endswith(audit.QWEN4_LOAD_SHIFTED) for name in qwen4_exp.Model._HF_NORM_SHIFT_SUFFIXES)
