"""Edge0 folder -> MTPLX pack (mtplx/edge0/forge.py).

A tiny synthetic Edge0 folder (tests/edge0_synth.make_folder: the shell
skeleton of tests/test_edge0_shell.py plus latent files, an FP8 n-gram table
and a tokenizer pair) is forged; the checks are on the files, the config, the
resume logic and the fail-closed accounting. Running the forged model is in
tests/test_edge0_model.py. The real-pack checks run only with
MTPLX_EDGE0_PACK pointing at a built pack (header reads only, no weights).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from mtplx.edge0 import forge  # noqa: E402
from tests import edge0_synth as syn  # noqa: E402

REAL_PACK = os.environ.get("MTPLX_EDGE0_PACK")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("edge0-forge")
    syn.make_folder(root / "src", root / "base")
    state = forge.build_pack(root / "src", root / "out", tokenizer_from=root / "base")
    return root, state


def _copy(built, tmp_path) -> Path:
    root, _ = built
    shutil.copytree(root / "out", tmp_path / "out")
    return tmp_path / "out"


def test_pack_files_and_config(built):
    root, state = built
    out = root / "out"
    assert state["complete"] is True
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "edge0_latent" and cfg["text_config"]["model_type"] == "edge0_latent_text"
    assert cfg["edge0_latent"] == {
        "format_version": 1, "latent_dim": 64, "expert_ff": 64, "n_groups": 3, "n_experts": 4,
        "dense_format": "q4", "dense_bits": 4, "top_k": 2, "expert_format": "NVFP4", "model_name": "synthetic",
    }
    assert cfg["mtplx_ngram_source"] == {"backend": "edge0_fp8_gguf", "file": str((root / "src" / "ple-fp8.gguf").absolute())}
    assert not (out / "ngram-table.safetensors").exists()  # referenced in place, not copied
    assert sorted(p.name for p in out.glob("model-latent-*.safetensors")) == [f"model-latent-{i:03d}.safetensors" for i in range(4)]
    assert list(out.glob("model-shell-*.safetensors"))
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert "language_model.model.layers.3.mlp.experts.scalars" in index["weight_map"]
    assert index["weight_map"]["language_model.model.embed_tokens.weight"].startswith("model-shell-")
    for name in ("tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"):
        assert (out / name).exists(), name
    gen = json.loads((out / "generation_config.json").read_text())
    assert gen["eos_token_id"] == [94, 92] and gen["bos_token_id"] == 92 and gen["top_k"] == 20
    assert json.loads((out / "tokenizer_config.json").read_text())["chat_template"] == syn.TEMPLATE
    prov = json.loads((out / forge.STATE_NAME).read_text())
    assert prov["settings"]["q8"] == "affine8" and prov["settings"]["dense_format"] == "q4"
    assert prov["source"]["files"]["labels.bin"]["sha256"]
    assert prov["ngram"]["tables"]["0"] == {"rows": 512, "dim": 16, "scale": 0.0625}
    assert prov["accounting"]["parameters"] > 0


def test_rerun_skips_completed_work(built, tmp_path):
    root, _ = built
    out = _copy(built, tmp_path)
    before = {p.name: p.stat().st_mtime_ns for p in out.glob("model-*.safetensors")}
    state = forge.build_pack(root / "src", out, tokenizer_from=root / "base")
    assert state["complete"] and state["steps"]["latent"]["layers_written_this_run"] == []
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("model-*.safetensors")} == before


def test_truncated_layer_is_rewritten(built, tmp_path):
    root, _ = built
    out = _copy(built, tmp_path)
    f = out / "model-latent-002.safetensors"
    data = f.read_bytes()
    f.write_bytes(data[: len(data) // 2])
    state = forge.build_pack(root / "src", out, tokenizer_from=root / "base")
    assert state["steps"]["latent"]["layers_written_this_run"] == [2]
    assert f.read_bytes() == data


def test_partial_build_is_not_marked_complete(built, tmp_path):
    root, _ = built
    state = forge.build_pack(root / "src", tmp_path / "out", tokenizer_from=root / "base", layers=[0, 1])
    assert state["complete"] is False and "accounting" not in state
    assert not (tmp_path / "out" / "model.safetensors.index.json").exists()
    state = forge.build_pack(root / "src", tmp_path / "out", tokenizer_from=root / "base")
    assert state["complete"] is True and state["steps"]["latent"]["layers_written_this_run"] == [2, 3]


def test_changed_settings_refuse_without_force(built, tmp_path):
    root, _ = built
    out = _copy(built, tmp_path)
    with pytest.raises(forge.ForgeError, match="other settings"):
        forge.build_pack(root / "src", out, tokenizer_from=root / "base", dense_format="q8")
    state = forge.build_pack(root / "src", out, tokenizer_from=root / "base", dense_format="q8", force=True)
    assert state["complete"] and json.loads((out / "config.json").read_text())["edge0_latent"]["dense_bits"] == 8
    assert state["steps"]["latent"]["layers_written_this_run"] == [0, 1, 2, 3]


def test_accounting_fails_closed(built, tmp_path):
    import mlx.core as mx

    root, _ = built
    out = _copy(built, tmp_path)
    cfg = json.loads((out / "config.json").read_text())
    forge.account(out, cfg)
    mx.save_safetensors(str(out / "model-stray.safetensors"), {"language_model.model.layers.0.mlp.bogus": mx.zeros(2)})
    with pytest.raises(forge.ForgeError, match="1 unexpected"):
        forge.account(out, cfg)
    (out / "model-stray.safetensors").unlink()
    (out / "model-latent-001.safetensors").unlink()
    with pytest.raises(forge.ForgeError, match="missing"):
        forge.account(out, cfg)


def test_wrong_latent_dtype_fails_accounting(built, tmp_path):
    import mlx.core as mx

    out = _copy(built, tmp_path)
    cfg = json.loads((out / "config.json").read_text())
    f = out / "model-latent-000.safetensors"
    arrays = mx.load(str(f))
    mx.eval(arrays)  # mx.load maps lazily; read everything before overwriting the file
    key = "language_model.model.layers.0.mlp.experts.scalars"
    arrays[key] = arrays[key].astype(mx.bfloat16)  # the ~1e-4 down scalar must stay float32
    mx.save_safetensors(str(f), arrays)
    with pytest.raises(forge.ForgeError, match="dtype"):
        forge.account(out, cfg)


def test_truncated_file_is_caught(built, tmp_path):
    out = _copy(built, tmp_path)
    f = next(out.glob("model-shell-*.safetensors"))
    f.write_bytes(f.read_bytes()[:-4])
    with pytest.raises(forge.ForgeError, match="bytes on disk"):
        forge.read_safetensors_header(f)


def test_mismatched_tokenizer_is_refused(built, tmp_path):
    root, _ = built
    base = tmp_path / "base"
    shutil.copytree(root / "base", base)
    vocab = json.loads((base / "vocab.json").read_text())
    vocab["t7"], vocab["t8"] = vocab["t8"], vocab["t7"]
    (base / "vocab.json").write_text(json.dumps(vocab))
    with pytest.raises(forge.ForgeError, match="vocab"):
        forge.check_tokenizer(root / "src" / "tokenizer", base)


@pytest.mark.skipif(not REAL_PACK, reason="MTPLX_EDGE0_PACK not set")
def test_real_pack_accounts_exactly():
    out = Path(REAL_PACK)
    cfg = json.loads((out / "config.json").read_text())
    acc = forge.account(out, cfg)
    state = json.loads((out / forge.STATE_NAME).read_text())
    assert state["complete"] and acc["parameters"] == state["accounting"]["parameters"]
    assert cfg["edge0_latent"]["latent_dim"] == 1280 and cfg["text_config"]["num_hidden_layers"] == 48
    assert len(list(out.glob("model-latent-*.safetensors"))) == 48
