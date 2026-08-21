from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.backends.dflash2 import (
    DEFAULT_DFLASH2_BLOCK_SIZE,
    DFlash2Runtime,
    DFlash2RuntimeConfig,
    DFlash2Unsupported,
    _load_dflash2_draft,
    load_dflash2_bundle,
    resolve_dflash2_bundle_paths,
)
from mtplx.sampling import SamplerConfig


def _bundle(tmp_path: Path, *, quantization: str = "unquantized") -> Path:
    root = tmp_path / "bundle"
    target = root / "target"
    draft = root / "dflash2"
    target.mkdir(parents=True)
    draft.mkdir()
    target_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "vocab_size": 151936,
    }
    draft_config = {
        "model_type": "qwen3",
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 5120,
        "vocab_size": 151936,
        "num_hidden_layers": 5,
        "num_target_layers": 64,
        "dflash_config": {"target_layer_ids": [1, 15, 30, 45, 60]},
    }
    (target / "config.json").write_text(json.dumps(target_config))
    (draft / "config.json").write_text(json.dumps(draft_config))
    (target / "model.safetensors").write_bytes(b"target")
    (draft / "model.safetensors").write_bytes(b"draft")
    (root / "mtplx_dflash2.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "backend": "dflash2",
                "layout": {"target": "target", "draft": "dflash2"},
                "target": {"repo": "converted/qwen3.8-27b-mlx", "revision": "target-revision"},
                "draft": {
                    "repo": "z-lab/Qwen3.8-27B-DFlash2",
                    "revision": "draft-revision",
                    "precision": quantization,
                },
                "algorithm": {"repo": "z-lab/dflash", "revision": "algorithm-revision"},
                "checksums": {
                    "target_config": {"path": "target/config.json", "sha256": "a" * 64},
                    "draft_config": {"path": "dflash2/config.json", "sha256": "b" * 64},
                    "draft_weights": {"path": "dflash2/model.safetensors", "sha256": "c" * 64},
                },
            }
        )
    )
    return root


def _install_dflash(monkeypatch, *, responses=None):
    calls = []
    module = types.ModuleType("dflash.model_mlx")

    def load(path):
        calls.append(("load", path))
        return "target-model", SimpleNamespace(decode=lambda ids: "".join(map(str, ids)))

    def load_draft(path):
        calls.append(("load_draft", path))
        return SimpleNamespace(parameters=lambda: {"weights": "draft"})

    def stream_generate(*args, **kwargs):
        calls.append(("stream_generate", args, kwargs))
        yield from (responses or [])

    module.load = load
    module.load_draft = load_draft
    module.snapshot_download = lambda *args, **kwargs: "remote-cache"
    module.stream_generate = stream_generate
    package = types.ModuleType("dflash")
    package.model_mlx = module
    monkeypatch.setitem(sys.modules, "dflash", package)
    monkeypatch.setitem(sys.modules, "dflash.model_mlx", module)
    return calls


def test_resolver_uses_manifest_target_and_dflash2_layout(tmp_path):
    root = _bundle(tmp_path, quantization="4-bit")
    resolved = resolve_dflash2_bundle_paths(root)
    assert resolved["target_model"] == str(root / "target")
    assert resolved["draft_model"] == str(root / "dflash2")
    assert resolved["draft_quantization"] == "4bit"
    assert resolved["draft_block_size"] == DEFAULT_DFLASH2_BLOCK_SIZE


def test_resolver_fails_closed_for_missing_sidecar(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "mtplx_dflash2.json").write_text('{"backend":"dflash2"}')
    with pytest.raises(ValueError, match="target/ and dflash2/"):
        resolve_dflash2_bundle_paths(root)


def test_load_is_lazy_and_sets_dflash2_runtime(tmp_path, monkeypatch):
    root = _bundle(tmp_path)
    calls = _install_dflash(monkeypatch)
    runtime = load_dflash2_bundle(root)
    assert runtime.backend_id == "dflash2"
    assert runtime.mtp_enabled is True
    assert runtime.config.draft_block_size == 5
    assert calls[:2] == [
        ("load", str(root / "target")),
        ("load_draft", str(root / "dflash2")),
    ]


def test_local_draft_loader_uses_local_directory_and_restores_on_success(tmp_path):
    module = types.SimpleNamespace()
    original = lambda *args, **kwargs: "remote-cache"
    module.snapshot_download = original
    seen = []
    draft_dir = tmp_path / "dflash2"
    draft_dir.mkdir()

    def loader(draft_id):
        seen.append(module.snapshot_download(draft_id))
        return "draft"

    assert _load_dflash2_draft(module, loader, draft_dir) == "draft"
    assert seen == [str(draft_dir.resolve())]
    assert module.snapshot_download is original


def test_local_draft_loader_restores_on_failure(tmp_path):
    module = types.SimpleNamespace()
    original = lambda *args, **kwargs: "remote-cache"
    module.snapshot_download = original
    draft_dir = tmp_path / "dflash2"
    draft_dir.mkdir()

    def loader(_draft_id):
        assert module.snapshot_download("ignored") == str(draft_dir.resolve())
        raise RuntimeError("draft failure")

    with pytest.raises(RuntimeError, match="draft failure"):
        _load_dflash2_draft(module, loader, draft_dir)
    assert module.snapshot_download is original


def test_remote_draft_loader_keeps_snapshot_download_untouched(tmp_path):
    module = types.SimpleNamespace()
    original = lambda *args, **kwargs: "remote-cache"
    module.snapshot_download = original
    seen = []

    def loader(draft_id):
        seen.append((draft_id, module.snapshot_download))
        return "draft"

    assert _load_dflash2_draft(module, loader, "org/dflash2") == "draft"
    assert seen == [("org/dflash2", original)]
    assert module.snapshot_download is original


def test_dflash2_stream_maps_output_stats_and_callback(tmp_path, monkeypatch):
    responses = [
        SimpleNamespace(tokens=[11], text="a", accepted=None, prompt_tps=20.0, generation_tps=10.0),
        SimpleNamespace(tokens=[12, 13], text="bc", accepted=2, prompt_tps=20.0, generation_tps=10.0, finish_reason="length"),
    ]
    calls = _install_dflash(monkeypatch, responses=responses)
    mlx_calls = _install_mlx(monkeypatch)
    runtime = DFlash2Runtime(
        target_model="target",
        tokenizer=SimpleNamespace(decode=lambda ids: "".join(map(str, ids))),
        draft_model="draft",
        config=DFlash2RuntimeConfig.from_paths(
            target_model_path=tmp_path,
            draft_model_path=tmp_path,
        ),
    )
    from mtplx.backends.dflash2 import generate_dflash2

    chunks = []
    output = generate_dflash2(
        runtime,
        [1, 2],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        token_callback=chunks.append,
    )
    assert output.tokens == [11, 12, 13]
    assert chunks == [[11], [12, 13]]
    assert output.finish_reason == "length"
    assert output.stats.draft_core["backend"] == "dflash2"
    assert output.stats.accepted_drafts == 1
    assert output.stats.drafted_tokens == 3
    stream_call = next(call for call in calls if call[0] == "stream_generate")
    assert stream_call[2]["block_size"] == 5
    assert output.stats.events[-1]["drafted_positions"] == [3, 4, 5]
    assert ("seed", 0) in mlx_calls


def test_malformed_bundle_is_rejected_before_model_loading(tmp_path, monkeypatch):
    root = _bundle(tmp_path)
    config_path = root / "dflash2" / "config.json"
    config = json.loads(config_path.read_text())
    config["num_target_layers"] = 63
    config_path.write_text(json.dumps(config))
    calls = _install_dflash(monkeypatch)
    with pytest.raises(ValueError, match="DFlash2 bundle rejected"):
        load_dflash2_bundle(root)
    assert calls == []


def _install_mlx(monkeypatch):
    from mlx import core, nn

    calls = []
    monkeypatch.setattr(core.random, "seed", lambda value: calls.append(("seed", value)))
    monkeypatch.setattr(core, "eval", lambda value: calls.append(("eval", value)))
    monkeypatch.setattr(nn, "quantize", lambda model, **kwargs: calls.append(("quantize", kwargs)))
    return calls


def test_draft_quantization_is_applied(tmp_path, monkeypatch):
    for quantization, bits in (("4bit", 4), ("8bit", 8)):
        root = _bundle(tmp_path / quantization, quantization=quantization)
        calls = _install_dflash(monkeypatch)
        mlx_calls = _install_mlx(monkeypatch)
        load_dflash2_bundle(root)
        assert ("quantize", {"group_size": 64, "bits": bits}) in mlx_calls
        assert any(call[0] == "eval" for call in mlx_calls)
        assert calls[:2][0][0] == "load"


def test_target_only_ar_seeds_mlx_and_honors_stop(tmp_path, monkeypatch):
    mlx_calls = _install_mlx(monkeypatch)
    generate_module = types.ModuleType("mlx_lm.generate")
    sample_module = types.ModuleType("mlx_lm.sample_utils")

    def stream_generate(*args, **kwargs):
        yield SimpleNamespace(tokens=[4, 9, 8], prompt_tps=10.0, generation_tps=5.0)

    generate_module.stream_generate = stream_generate
    sample_module.make_sampler = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_module)
    runtime = DFlash2Runtime(
        target_model="target",
        tokenizer=SimpleNamespace(decode=lambda ids: "".join(map(str, ids))),
        draft_model="draft",
        config=DFlash2RuntimeConfig.from_paths(
            target_model_path=tmp_path,
            draft_model_path=tmp_path,
        ),
    )
    chunks = []
    from mtplx.backends.dflash2 import generate_dflash2_ar

    output = generate_dflash2_ar(
        runtime,
        [1],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        seed=17,
        stop_token_ids={9},
        token_callback=chunks.append,
    )
    assert output.tokens == [4, 9]
    assert chunks == [[4]]
    assert output.stats.runtime_mtp_enabled is False
    assert ("seed", 17) in mlx_calls


def test_generation_dispatch_and_unsupported_features(tmp_path, monkeypatch):
    responses = [SimpleNamespace(tokens=[7], text="7", finish_reason="stop")]
    _install_dflash(monkeypatch, responses=responses)
    runtime = DFlash2Runtime(
        target_model="target",
        tokenizer=SimpleNamespace(decode=lambda ids: "".join(map(str, ids))),
        draft_model="draft",
        config=DFlash2RuntimeConfig.from_paths(
            target_model_path=tmp_path,
            draft_model_path=tmp_path,
        ),
    )
    from mtplx.generation import generate_ar, generate_mtpk

    sampler = SamplerConfig(temperature=0.0)
    out = generate_mtpk(
        runtime,
        [1],
        max_tokens=2,
        sampler=sampler,
        speculative_depth=5,
        session_id="request-1",
        session_template_hash="template-hash",
        session_draft_head_identity="draft-head",
        session_policy_fingerprint="policy-fingerprint",
        session_restore_mode="cold",
    )
    assert out.tokens == [7]
    with pytest.raises(DFlash2Unsupported, match="constrained"):
        generate_mtpk(runtime, [1], max_tokens=2, sampler=sampler, speculative_depth=5, constraint=object())
    with pytest.raises(DFlash2Unsupported, match="sessions"):
        generate_ar(runtime, [1], max_tokens=2, sampler=sampler, session_bank=object())
