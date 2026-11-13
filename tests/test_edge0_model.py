"""The Edge0 model class (mtplx/models/edge0_latent.py) on a forged pack.

A tiny synthetic Edge0 folder is forged (mtplx.edge0.forge), loaded through
mlx-lm's loader with the model's own classes, and run forward with a cache.
During that forward the MoE half of a layer is captured and checked against a
NumPy composition of mtplx.edge0.reference over the *source* folder's arrays
(skeleton router and shared expert, decoded NVFP4 experts, Q4_0 D/U): the
hyper-connection read, the latent MoE block, and the residual write that the
next layer reads. The real-pack test runs only with MTPLX_EDGE0_PACK set (it
loads ~42 GB).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mtplx.edge0 import forge, reference  # noqa: E402
from mtplx.edge0.pack import Edge0Pack  # noqa: E402
from mtplx.models import edge0_latent  # noqa: E402
from mtplx.models.edge0_latent_moe import LatentMoeBlock  # noqa: E402
from tests import edge0_synth as syn  # noqa: E402

REAL_PACK = os.environ.get("MTPLX_EDGE0_PACK")
TOKENS = [[3, 17, 42, 8, 61, 5, 29, 77, 11, 50]]


@pytest.fixture(scope="module")
def forged(tmp_path_factory):
    root = tmp_path_factory.mktemp("edge0-model")
    syn.make_folder(root / "src", root / "base")
    forge.build_pack(root / "src", root / "out", tokenizer_from=root / "base")
    return root


@pytest.fixture
def model(forged):
    m, _ = edge0_latent.load(forged / "out")
    return m


class _Recorder(nn.Module):
    """Stands in for a child module and logs (args, output) of every call."""

    def __init__(self, inner, log: list):
        super().__init__()
        self.inner = inner
        self._log = log

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        self._log.append((args, out))
        return out


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def _rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def test_every_layer_has_the_latent_block(model):
    assert all(isinstance(layer.mlp, LatentMoeBlock) for layer in model.layers)
    assert not any(".switch_mlp." in k for k, _ in nn.utils.tree_flatten(model.parameters()))
    assert model.layers[0].mlp.gate.weight.dtype == mx.float32  # router held in float32
    assert model.activation_dtype == mx.float32
    table = model.layers[1].ple.ple_embedding.ngram_embedding
    assert table._sidecar is not None and table._sidecar.fp8_scale == 0.0625


def test_forward_with_cache_matches_one_shot(model):
    ids = mx.array(TOKENS)
    full = model(ids, cache=model.make_cache())
    cache = model.make_cache()
    first = model(ids[:, :6], cache=cache)
    steps = [model(ids[:, i : i + 1], cache=cache) for i in range(6, ids.shape[1])]
    inc = mx.concatenate([first] + steps, axis=1)
    assert full.dtype == mx.float32 and full.shape == (1, 10, 96)
    assert bool(mx.isfinite(full).all())
    assert _rel(_np(inc), _np(full).astype(np.float64)) < 5e-3


@pytest.mark.parametrize("layer", [1, 3])
def test_moe_half_matches_the_numpy_reference(model, forged, layer):
    """Hyper-connection read, latent MoE and residual write of one layer against reference.py."""
    lay = model.layers[layer]
    reads, blocks = [], []
    lay.mlp_hyper_connection = _Recorder(lay.mlp_hyper_connection, reads)
    lay.mlp = _Recorder(lay.mlp, blocks)
    nxt = []
    if layer + 1 < len(model.layers):
        model.layers[layer + 1].attn_hyper_connection = _Recorder(model.layers[layer + 1].attn_hyper_connection, nxt)
    else:
        model.language_model.model.hyper_connection_mixer = _Recorder(
            model.language_model.model.hyper_connection_mixer, nxt
        )
    model(mx.array(TOKENS), cache=model.make_cache())

    weights = reference.layer_weights_from_pack(Edge0Pack(forged / "src"), layer)
    (_args, (mixed, written, _inject)), = reads
    (x_in,), y = blocks[0]
    R = _np(written)[0]  # [S, HC * H]: the stream after the attention write
    hc = weights.hc.inject.shape[0]
    x_model = _np(x_in)[0]
    # Measured on an M5 Max: with MLX_ENABLE_TF32=0 all three agree to ~1e-6. MLX runs multi-row float32
    # GEMMs as TF32 by default, which moves the hyper-connection read/write (bf16 weights, float32 rows) by
    # ~4e-4; the latent block computes its router and quantized products without TF32 and stays at ~1e-6.
    hc_tol = 2e-3 if os.environ.get("MLX_ENABLE_TF32", "1") != "0" else 1e-5
    for t in range(R.shape[0]):
        R_new, res, x_ref, _ = reference.moe_half(R[t].reshape(hc, -1), weights, top_k=2, combine="per_group")
        assert _rel(_np(mixed)[0, t], x_ref) < hc_tol, ("gr_read", t)
        # the block on the model's own input, so read and block errors do not compound
        blk = reference.latent_moe(x_model[t], dense_d=weights.dense_d, dense_u=weights.dense_u, labels=weights.labels,
                                   experts=weights.experts, router=weights.router, shared=weights.shared, top_k=2,
                                   combine="per_group")
        assert _rel(_np(y)[0, t], blk.out) < 1e-5, ("latent_moe", t)
        assert _rel(_np(nxt[0][0][0])[0, t], R_new.reshape(-1)) < hc_tol, ("residual write", t)


def test_bfloat16_policy_keeps_the_latent_scales(model):
    ids = mx.array(TOKENS)
    ref = _np(model(ids, cache=model.make_cache()))
    model.set_activation_dtype("bfloat16")
    params = dict(nn.utils.tree_flatten(model.parameters()))
    mlp = "language_model.model.layers.0.mlp."
    for k in ("experts.scalars", "gate.weight"):
        assert params[mlp + k].dtype == mx.float32, k
    # D/U scales stay exact: float32, or float16 when every value round-trips (compact_scales)
    for k in ("dense_d.scales", "dense_d.biases", "dense_u.scales", "dense_u.biases"):
        assert params[mlp + k].dtype in (mx.float32, mx.float16), k
    assert params["language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"].dtype == mx.bfloat16
    assert params["language_model.model.embed_tokens.scales"].dtype == mx.bfloat16
    assert model.layers[0].mlp.latent_dtype == mx.bfloat16
    out = model(ids, cache=model.make_cache())
    assert out.dtype == mx.bfloat16
    got = _np(out).reshape(-1)
    cos = float(got @ ref.reshape(-1) / np.linalg.norm(got) / np.linalg.norm(ref))
    assert cos > 0.99


def test_loader_rejects_a_bad_latent_dtype(forged, tmp_path):
    import shutil

    out = tmp_path / "out"
    shutil.copytree(forged / "out", out)
    f = out / "model-latent-001.safetensors"
    arrays = mx.load(str(f))
    mx.eval(arrays)  # mx.load maps lazily; read everything before overwriting the file
    k = "language_model.model.layers.1.mlp.dense_u.scales"
    arrays[k] = arrays[k].astype(mx.float16)
    mx.save_safetensors(str(f), arrays)
    with pytest.raises(ValueError, match="dense_u.scales"):
        edge0_latent.load(out)


def test_runtime_and_registry_know_the_model_type():
    from mtplx import runtime
    from mtplx.backends import registry

    assert runtime._model_classes_for_config({"model_type": "edge0_latent"}) == (edge0_latent.Model, edge0_latent.ModelArgs)
    assert "edge0_latent" in registry._INTREE_MODEL_TYPES
    assert set(runtime._INTREE_MODEL_CLASS_LOADERS) - {"prism_hadamard_qwen35"} == registry._INTREE_MODEL_TYPES
    from mtplx.models.qwen4_exp import is_qwen4_exp_mtp_config

    cfg = {"model_type": "edge0_latent", "text_config": {"model_type": "edge0_latent_text"}}
    assert not is_qwen4_exp_mtp_config(cfg)  # no qwen4_exp lane keys on this config


_SHELL_LANES = ("MTPLX_COMPILED_GDN", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_FUSED_GDN_STEP", "MTPLX_FUSED_GDN_CONVNORM")


def test_server_stamps_the_shell_lanes(forged, monkeypatch):
    """The Flash-Next shell's decode lanes are stamped for Edge0; no routed-expert or lossy lane is."""
    import argparse

    from mtplx.server.openai import _server_runtime_env_overrides

    for key in (*_SHELL_LANES, "MTPLX_FUSED_HC_V3", "MTPLX_FUSED_GATE_UP"):
        monkeypatch.delenv(key, raising=False)
    args = argparse.Namespace(model=str(forged / "out"), generation_mode="ar", verify_strategy=None)
    got = _server_runtime_env_overrides(args, {})
    assert all(got.get(key) == "1" for key in _SHELL_LANES), got
    assert "MTPLX_FUSED_HC_V3" not in got and "MTPLX_FUSED_GATE_UP" not in got
    # an explicit operator kill switch wins, on either compile alias
    monkeypatch.setenv("MTPLX_COMPILED_GDN", "0")
    got = _server_runtime_env_overrides(args, {})
    assert "MTPLX_COMPILED_GDN" not in got and "MTPLX_QWEN4EXP_COMPILE" not in got
    assert got.get("MTPLX_FUSED_GDN_STEP") == "1"


def test_load_merges_gate_up_and_compacts_exact_scales(model):
    mlp = model.layers[0].mlp
    assert mlp.experts.merged and "gate_proj" not in mlp.experts
    # the forged D/U scales are float16 values (ggml Q4_0), so compaction is exact
    assert mlp.dense_d.scales.dtype == mx.float16 and mlp.dense_u.biases.dtype == mx.float16


def test_runtime_load_autoregressive(forged):
    from mtplx import runtime

    rt = runtime.load(forged / "out", mtp=False)
    assert isinstance(rt.model, edge0_latent.Model) and not rt.mtp_enabled
    logits = rt.model(mx.array(TOKENS), cache=rt.model.make_cache())
    assert bool(mx.isfinite(logits).all())


@pytest.mark.skipif(not REAL_PACK, reason="MTPLX_EDGE0_PACK not set")
def test_real_pack_says_paris():
    from transformers import AutoTokenizer

    path = Path(REAL_PACK)
    tok = AutoTokenizer.from_pretrained(str(path))
    model, _ = edge0_latent.load(path)
    ids = tok.encode("The capital of France is")
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)
    out = []
    for _ in range(4):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)
    assert "Paris" in tok.decode(out)
    assert json.loads((path / "config.json").read_text())["model_type"] == "edge0_latent"
