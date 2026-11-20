"""CPU header, precision and corruption checks for the Flash-Next forge lane."""

from __future__ import annotations

import pytest
import json
import struct

from mtplx.commands.forge_qwen4_exp import (
    NGRAM_PRODUCTION_LAYOUT,
    Qwen4ForgeError,
    head_overrides,
    is_qwen4_exp_source,
    mtp_layout_keys,
    mtp_module_rule,
    ngram_sidecar_layout,
    recipe_params,
    named_recipe,
    QUALITY_RECIPE,
    run_lane,
)

# Qwen3.8-Flash-Next: 128 shards x 2,500,012 rows x 160 = the official pack's row count.
ROWS = 128 * 2_500_012
DIM = 160


def test_is_qwen4_exp_source_reads_either_level() -> None:
    assert is_qwen4_exp_source({"model_type": "qwen4_exp"})
    assert is_qwen4_exp_source({"model_type": "", "text_config": {"model_type": "qwen4_exp"}})
    assert not is_qwen4_exp_source({"model_type": "qwen3_next"})


def test_recipe_defaults_match_official_pack() -> None:
    params = recipe_params({"body_bits": 4, "body_group_size": 32, "body_mode": "affine"})
    assert params["body_bits"] == 4 and params["body_group"] == 32
    assert (params["ngram_bits"], params["ngram_group"]) == NGRAM_PRODUCTION_LAYOUT
    # The head follows the trunk unless pinned: measured identical greedy drafts
    # at bf16 and 4-bit, bf16 only costs bandwidth.
    assert params["mtp_bits"] == 4
    assert recipe_params({"body_bits": 4, "qwen4_mtp_bits": 0})["mtp_bits"] == 0


def test_recipe_refuses_non_affine_and_bf16_trunk() -> None:
    with pytest.raises(Qwen4ForgeError):
        recipe_params({"body_bits": 4, "body_mode": "mxfp4"})
    with pytest.raises(Qwen4ForgeError):
        recipe_params({"body_bits": 0})


def test_ngram_layout_matches_attach_sidecar_contract() -> None:
    header = ngram_sidecar_layout(ROWS, DIM, bits=4, group=32)
    assert header["__metadata__"] == {
        "ngram_bits": "4", "ngram_group_size": "32", "rows": str(ROWS), "dim": "160",
    }
    assert header["ngram.weight"]["dtype"] == "U32" and header["ngram.weight"]["shape"] == [ROWS, 20]
    assert header["ngram.scales"]["shape"] == [ROWS, 5] and header["ngram.biases"]["shape"] == [ROWS, 5]
    w, s, b = (header[k]["data_offsets"] for k in ("ngram.weight", "ngram.scales", "ngram.biases"))
    assert w[0] == 0 and w[1] == s[0] and s[1] == b[0]
    assert b[1] == ROWS * (20 * 4 + 5 * 2 + 5 * 2)  # 32,000,153,600 bytes: the 29.8 GiB official table


def test_ngram_layout_other_widths_and_raw() -> None:
    assert ngram_sidecar_layout(ROWS, DIM, bits=3, group=32)["ngram.weight"]["shape"] == [ROWS, 15]
    raw = ngram_sidecar_layout(ROWS, DIM, bits=0, group=32)
    assert set(raw) == {"__metadata__", "ngram.weight"} and raw["ngram.weight"]["dtype"] == "BF16"
    with pytest.raises(Qwen4ForgeError):
        ngram_sidecar_layout(ROWS, DIM, bits=4, group=64)  # 160 % 64


def test_mtp_layout_splits_packed_experts_like_sanitize() -> None:
    keys = mtp_layout_keys([
        "mtp.fc_hidden.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.self_attn.q_norm.weight",
    ])
    assert keys == [
        "fc_hidden.weight",
        "layers.0.mlp.switch_mlp.gate_proj.weight",
        "layers.0.mlp.switch_mlp.up_proj.weight",
        "layers.0.mlp.switch_mlp.down_proj.weight",
        "layers.0.self_attn.q_norm.weight",
    ]


def test_mtp_rules_mirror_trunk_recipe() -> None:
    def rule(key, **kwargs):
        return mtp_module_rule(key, mtp_bits=4, mtp_group=32, **kwargs)
    assert rule("layers.0.mlp.switch_mlp.gate_proj.weight") == (4, 32)
    assert rule("layers.0.mlp.gate.weight") == (8, 64)
    assert rule("layers.0.mlp.shared_expert.down_proj.weight") == (8, 64)
    assert rule("layers.0.self_attn.indexer.index_qk_proj.weight") == (8, 64)
    assert rule("layers.0.self_attn.q_proj.weight") == (4, 32)
    assert rule("layers.0.self_attn.q_proj.weight", qsa_8bit=True) == (8, 64)
    assert rule("layers.0.attn_hyper_connection.input_mix_weight_down.weight") is None
    assert rule("fc_hidden.weight") is None
    assert mtp_module_rule("layers.0.mlp.switch_mlp.up_proj.weight", mtp_bits=0, mtp_group=32) is None


def test_the_head_mirrors_a_recipes_overrides_except_those_that_name_trunk_layers() -> None:
    recipe = {
        "module_overrides": [
            {"suffix": "shared_expert.up_proj", "bits": 4, "group_size": 64},
            {"suffix": "self_attn.o_proj", "quantize": False},
            {"suffix": "mlp.gate", "bits": 4, "group_size": 32, "layers": [46, 47]},
        ]
    }
    overrides = head_overrides(recipe)

    def rule(key):
        return mtp_module_rule(key, mtp_bits=4, mtp_group=32, overrides=overrides)

    assert rule("layers.0.mlp.shared_expert.up_proj.weight") == (4, 64)
    assert rule("layers.0.self_attn.o_proj.weight") is None
    assert rule("layers.0.mlp.gate.weight") == (8, 64)  # the trunk's layers 46 and 47 are not the head's layer 0
    assert rule("layers.0.mlp.shared_expert.down_proj.weight") == (8, 64)
    assert head_overrides({}) == ()


def test_the_head_follows_the_trunks_group_unless_pinned() -> None:
    assert recipe_params({"body_bits": 4, "body_group_size": 64})["mtp_group"] == 64
    assert recipe_params({"body_bits": 4, "body_group_size": 64, "qwen4_mtp_group_size": 32})["mtp_group"] == 32


def test_named_quality_is_the_decided_recipe_and_is_not_mutable_global_state():
    recipe = named_recipe(QUALITY_RECIPE)
    assert recipe_params(recipe) == {"body_bits": 8, "body_group": 64, "ngram_bits": 4,
                                      "ngram_group": 32, "mtp_bits": 8, "mtp_group": 64}
    recipe["ngram"]["bits"] = 8
    assert named_recipe(QUALITY_RECIPE)["ngram"]["bits"] == 4


def test_quality_refuses_q8_table_before_reading_source_or_writing_output(tmp_path):
    recipe = named_recipe(QUALITY_RECIPE)
    recipe["ngram"]["bits"] = 8
    with pytest.raises(Qwen4ForgeError, match="4-bit/g32.*fixed-M4"):
        run_lane(tmp_path / "missing-source", tmp_path / "output", recipe=recipe)
    assert not (tmp_path / "output").exists()


def test_a_custom_recipe_may_still_choose_another_table_layout():
    """The refusal belongs to the published Quality pack. The generic lane keeps
    what it could do before: build the layout it was asked for and warn that
    the fixed-M4 verify lane will decline it."""
    params = recipe_params({"body_bits": 8, "body_group_size": 64, "body_mode": "affine",
                            "ngram": {"bits": 8, "group_size": 64}})
    assert (params["ngram_bits"], params["ngram_group"]) == (8, 64)


def test_quality_refuses_precision_overrides():
    recipe = named_recipe(QUALITY_RECIPE)
    recipe["module_overrides"] = [{"suffix": "gate_proj", "bits": 4}]
    with pytest.raises(Qwen4ForgeError, match="cannot be overridden"):
        recipe_params(recipe)


def _write_tensors(path, tensors):
    """Independent safetensors writer: these audits need no MLX or Metal."""
    offset, header, data = 0, {}, []
    for name, (dtype, array) in tensors.items():
        payload = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
        data.append(payload)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(data))


def _audit_fixture(tmp_path):
    import numpy as np
    from mtplx.commands.forge_qwen4_audit import tensor_class, BF16_CLASSES
    from mtplx.commands.forge_qwen4_exp import NGRAM_KEY

    source, pack = tmp_path / "source", tmp_path / "pack"
    source.mkdir()
    pack.mkdir()
    raw, body, mtp, table = {}, {}, {}, {}
    keys = [
        "language_model.model.embed_tokens.weight", "language_model.lm_head.weight",
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        "language_model.model.layers.0.linear_attn.A_log", "language_model.model.layers.0.linear_attn.dt_bias",
        "language_model.model.layers.0.ple.key_proj.weight", "language_model.model.layers.0.ple.norm_key.weight",
        "language_model.model.layers.0.ple.ple_embedding.layer_multipliers", "vision_tower.patch_embed.proj.weight",
        "mtp.fc_hidden.weight",
    ]
    for prefix in ("language_model.model.", "mtp."):
        keys += [prefix + "layers.0." + suffix for suffix in (
            "mlp.switch_mlp.gate_proj.weight", "mlp.shared_expert.down_proj.weight", "mlp.gate.weight",
            "self_attn.q_proj.weight", "self_attn.indexer.index_qk_proj.weight",
            "attn_hyper_connection.input_mix_weight_down.weight", "self_attn.q_norm.weight")]
    for key in keys:
        cls = tensor_class(key)
        dst = mtp if key.startswith("mtp.") else body
        if cls == "ple_integer_buffers":
            raw[key] = dst[key] = ("I64", np.arange(3, dtype="<i8"))
        elif cls in BF16_CLASSES:
            shifted_norm = key.endswith(("q_norm.weight", "norm_key.weight"))
            shape = (4,) if "norm" in cls or cls in ("gdn_A_log", "gdn_dt_bias") else (2, 3, 1) if cls == "gdn_conv" else (2, 64)
            raw[key] = ("BF16", np.zeros(shape, dtype="<u2"))
            dst[key] = ("BF16", np.full(shape, 0x3f80 if shifted_norm else 0, dtype="<u2"))
        else:
            shape = (2, 3, 64) if "routed" in cls else (3, 64)
            codes = np.broadcast_to(np.arange(64, dtype=np.uint32) % 16, shape)
            values = codes.astype(np.float32) / 16 - 0.5
            raw[key] = ("BF16", (values.view(np.uint32) >> 16).astype("<u2"))
            packed = np.sum(codes.reshape(*shape[:-1], -1, 4) << np.arange(0, 32, 8, dtype=np.uint32), axis=-1).astype("<u4")
            dst[key] = ("U32", packed)
            base = key.removesuffix(".weight")
            dst[base + ".scales"] = ("BF16", np.full((*shape[:-1], 1), 0x3d80, dtype="<u2"))
            dst[base + ".biases"] = ("BF16", np.full((*shape[:-1], 1), 0xbf00, dtype="<u2"))
    for i, rows in enumerate((3, 2)):
        raw[NGRAM_KEY.format(layer=1, index=i)] = ("BF16", np.zeros((rows, 160), dtype="<u2"))
    table["ngram.weight"] = ("U32", np.zeros((5, 20), dtype="<u4"))
    table["ngram.scales"] = ("BF16", np.zeros((5, 5), dtype="<u2"))
    table["ngram.biases"] = ("BF16", np.zeros((5, 5), dtype="<u2"))
    raw = {k.replace("language_model.model.", "model.language_model.", 1): v for k, v in raw.items()}
    _write_tensors(source / "source.safetensors", raw)
    (source / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": {"hidden_size": 64}}))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "source.safetensors" for k in raw}}))
    # Split every affine triple across different body shards.
    weight_map = {}
    for i, suffix in enumerate((".weight", ".scales", ".biases")):
        chunk = {k: v for k, v in body.items() if k.endswith(suffix) or (i == 0 and not k.endswith((".weight", ".scales", ".biases")))}
        filename = f"model-{i}.safetensors"
        _write_tensors(pack / filename, chunk)
        weight_map.update({k: filename for k in chunk})
    _write_tensors(pack / "mtp.safetensors", mtp)
    _write_tensors(pack / "ngram-table.safetensors", table)
    (pack / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return source, pack


def test_streaming_audit_reads_all_classes_and_cross_shard_affine_triples(tmp_path):
    from mtplx.commands.forge_qwen4_audit import audit_pack, checksum

    source, pack = _audit_fixture(tmp_path)
    report = audit_pack(source, pack, named_recipe(QUALITY_RECIPE),
                        expected_checksums={p.name: checksum(p) for p in pack.glob("*.safetensors")})
    assert report["passed"]
    assert len(report["stored_precision"]) == 25
    assert set(report["samples"]) == set(report["stored_precision"])
    assert report["samples"]["ngram_table"][0]["rows"] == [[0], [2], [3], [4]]
    assert report["stored_precision"]["routed_experts"]["stored_precision"].startswith("Q8/g64")
    assert report["stored_precision"]["mtp_fc"]["stored_precision"] == "BF16"


@pytest.mark.parametrize("damage", ["checksum", "payload", "precision", "index"])
def test_streaming_audit_refuses_corrupt_or_mislabelled_artifacts(tmp_path, damage):
    from mtplx.commands.forge_qwen4_audit import audit_pack, checksum, inventory

    source, pack = _audit_fixture(tmp_path)
    hashes = {p.name: checksum(p) for p in pack.glob("*.safetensors")}
    if damage == "checksum":
        hashes["mtp.safetensors"] = "0" * 64
    elif damage in ("payload", "precision"):
        tensors, _ = inventory(pack)
        key = "language_model.model.embed_tokens.weight"
        info = tensors[key]
        path = pack / info["file"]
        if damage == "payload":
            with path.open("r+b") as handle:
                handle.seek(info["start"] + info["data_offsets"][0])
                handle.write(b"\xff" * 4)
        else:
            content = path.read_bytes()
            # Same item width, syntactically valid header, wrong precision.
            path.write_bytes(content.replace(b'"U32"', b'"F32"', 1))
    else:
        index = json.loads((pack / "model.safetensors.index.json").read_text())
        index["weight_map"].pop(next(iter(index["weight_map"])))
        (pack / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(Qwen4ForgeError, match="Checksum|parity|shape|precision|Unindexed"):
        audit_pack(source, pack, named_recipe(QUALITY_RECIPE), expected_checksums=hashes)


def test_streaming_build_never_loads_or_reuses_a_verified_stamp(tmp_path, monkeypatch):
    import shutil
    from mtplx.commands import forge, forge_qwen4_exp
    from mtplx.cli import build_parser

    source, fixture = _audit_fixture(tmp_path)
    config = json.loads((source / "config.json").read_text())
    (fixture / "config.json").write_text(json.dumps(config))
    (source / "mtplx_runtime.json").write_text(json.dumps({"verified_on": {"model": "old"}}))
    monkeypatch.setattr(forge, "probe_source", lambda _: {"forgeable": True, "source_format": "bf16_native", "recommended_backend": "qwen4_exp"})
    monkeypatch.setattr(forge_qwen4_exp, "run_lane", lambda src, dst, **kw: shutil.copytree(fixture, dst))
    monkeypatch.setattr(forge, "_ensure_vision_tower", lambda *a: None)
    monkeypatch.setattr(forge, "_validate_vision_payload", lambda *a: None)
    def forbidden(*a, **kw):
        pytest.fail("Streaming verification tried to load/calibrate or stamp the model")
    for name in ("_run_verify", "_calibrate_sidecar", "_calibrate_mtp_contract", "_stamp_runtime_metadata"):
        monkeypatch.setattr(forge, name, forbidden)
    common = ["forge", "build", "--repo", str(source), "--out", str(tmp_path / "runs"),
              "--run-id", "audit", "--branded-name", "Quality", "--recipe", QUALITY_RECIPE]
    parser = build_parser()
    assert parser.parse_args(common).verification == "full-load"
    args = parser.parse_args([*common, "--verification", "streaming"])
    assert forge._cmd_build(args, model_root=tmp_path / "models") == 0
    meta = json.loads((tmp_path / "models/Quality/mtplx_runtime.json").read_text())
    assert meta["verification"]["status"] == "streaming-audited"
    assert meta["verification"]["full_load_verified"] is False
    assert meta["verified_on"] == {} and "speed_evidence" not in meta
    assert meta["sampler"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    from mtplx.backends.registry import RuntimeContract, _runtime_contract_blocker
    contract = RuntimeContract.from_dict(meta)
    assert contract.arch_id == "qwen4-next"
    assert _runtime_contract_blocker(contract) == "exactness_baseline status is pending_full_load"
    assert meta["served_model_id"] == meta["quality_pack"]["served_id"]
    assert meta["quality_pack"]["min_engine_version"] == "2.12.0"
    assert meta["quality_pack"]["source"]["revision"].startswith("sha256:")
    assert meta["forge_provenance"]["forge_recipe"]["name"] == QUALITY_RECIPE
    # Moving to a bigger machine does not authorize stamping changed bytes.
    from mtplx.commands.forge_qwen4_audit import verify_pack_checksums
    built = tmp_path / "models/Quality"
    verify_pack_checksums(built, meta["quality_pack"]["files"])
    with (built / "mtp.safetensors").open("ab") as handle:
        handle.write(b"corrupt")
    verify_args = parser.parse_args(["forge", "verify", str(built), "--stamp",
                                    "--out", str(tmp_path / "runs"), "--run-id", "promote"])
    with pytest.raises(Qwen4ForgeError, match="Checksum mismatch before full-load"):
        forge._cmd_verify(verify_args)
