from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mtplx.edge0.pack import Edge0Pack, Edge0PackError
from tests import edge0_synth as syn

REAL = os.environ.get("MTPLX_EDGE0_MODEL")
real_only = pytest.mark.skipif(not REAL, reason="MTPLX_EDGE0_MODEL not set")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return syn.make_pack(tmp_path_factory.mktemp("edge0") / "pack")


def _copy(synth, tmp_path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(synth.path, dst)
    return dst


def test_geometry(synth):
    p = Edge0Pack(synth.path)
    assert (p.model_name, p.n_layers, p.n_experts) == ("synthetic", syn.N_LAYERS, syn.N_EXPERTS)
    assert (p.hidden, p.expert_ff, p.latent_dim, p.n_groups) == (syn.HIDDEN, syn.EXPERT_FF, syn.LATENT, syn.N_GROUPS)
    assert p.expert_format == "NVFP4" and p.default_dense_format == "q4"
    assert p.labels.dtype == np.int32 and p.labels.shape == (syn.N_LAYERS, syn.N_EXPERTS)
    np.testing.assert_array_equal(p.labels, synth.labels)


def test_expert_blob_and_decode(synth):
    p = Edge0Pack(synth.path)
    for layer, e in [(0, 0), (0, 7), (1, 3)]:
        blob = p.expert_blob(layer, e)
        assert bytes(blob) == synth.blobs[(layer, e)]
        ex = p.expert(layer, e)
        n = syn.EXPERT_FF * syn.LATENT
        raw = synth.blobs[(layer, e)]
        proj = n // 64 * 36
        assert ex.gate.shape == (syn.EXPERT_FF, syn.LATENT) and ex.down.shape == (syn.LATENT, syn.EXPERT_FF)
        assert ex.gate.dtype == np.float32
        np.testing.assert_array_equal(ex.gate.ravel(), np.array(syn.ref_nvfp4(raw[:proj]), np.float32))
        np.testing.assert_array_equal(ex.up.ravel(), np.array(syn.ref_nvfp4(raw[proj : 2 * proj]), np.float32))
        np.testing.assert_array_equal(ex.down.ravel(), np.array(syn.ref_nvfp4(raw[2 * proj : 3 * proj]), np.float32))
        sg, su, sd = synth.tails[(layer, e)]
        assert (ex.s_gate, ex.s_up, ex.s_down) == (sg, su, sd)
        g, u, d = ex.scaled()
        np.testing.assert_allclose(g, ex.gate * np.float32(sg))
        np.testing.assert_allclose(d, ex.down * np.float32(sd))


def test_experts_are_distinct(synth):
    p = Edge0Pack(synth.path)
    assert not np.array_equal(p.expert(0, 0).gate, p.expert(1, 0).gate)
    assert not np.array_equal(p.expert(0, 0).gate, p.expert(0, 1).gate)


def test_index_errors(synth):
    p = Edge0Pack(synth.path)
    with pytest.raises(IndexError):
        p.expert_blob(2, 0)
    with pytest.raises(IndexError):
        p.expert(0, syn.N_EXPERTS)
    with pytest.raises(IndexError):
        p.dense_d(-1)


@pytest.mark.parametrize("fmt, ref", [("q4", syn.ref_q4_0), ("q8", syn.ref_q8_0)])
def test_dense(synth, fmt, ref):
    p = Edge0Pack(synth.path)
    for layer in range(syn.N_LAYERS):
        d, u = p.dense_d(layer, fmt), p.dense_u(layer, fmt)
        assert d.shape == (syn.N_GROUPS, syn.LATENT, syn.HIDDEN) and d.dtype == np.float32
        assert u.shape == (syn.N_GROUPS, syn.HIDDEN, syn.LATENT)
        np.testing.assert_array_equal(d.ravel(), np.array(ref(synth.dense[(fmt, layer, "d")]), np.float32))
        np.testing.assert_array_equal(u.ravel(), np.array(ref(synth.dense[(fmt, layer, "u")]), np.float32))
    # row g * L + i of D is group g, latent i
    flat = np.array(syn.ref_q4_0(synth.dense[("q4", 0, "d")]), np.float32).reshape(-1, syn.HIDDEN)
    np.testing.assert_array_equal(p.dense_d(0)[2, 5], flat[2 * syn.LATENT + 5])


def test_default_dense_follows_model_json(tmp_path):
    pk = syn.make_pack(tmp_path / "q8", dense_fmt="8", both_dense=False)
    p = Edge0Pack(pk.path)
    assert p.default_dense_format == "q8"
    assert p.dense_d(0).shape == (syn.N_GROUPS, syn.LATENT, syn.HIDDEN)
    with pytest.raises(Edge0PackError, match="latent_dense_q4.bin"):
        p.dense_d(0, "q4")


def test_skeleton_and_ple_lazy(synth):
    p = Edge0Pack(synth.path)
    assert p.skeleton.metadata["general.architecture"] == "qwen4exp"
    assert p.skeleton.array("blk.0.ffn_gate_inp.weight").shape == (syn.N_EXPERTS, syn.HIDDEN)
    assert p.skeleton.array("token_embd.weight").shape == (4, syn.HIDDEN)
    (t,) = p.ple.tensors.values()
    assert t.ggml_type == 24 and bytes(p.ple.raw(t.name)[:3]) == bytes([0, 1, 2])
    assert p.ple.metadata["strata.ple.scale"] == 0.25


def test_symlinked_files(synth, tmp_path):
    root = tmp_path / "links"
    root.mkdir()
    for f in synth.path.iterdir():
        (root / f.name).symlink_to(f)
    p = Edge0Pack(root)
    assert bytes(p.expert_blob(1, 1)) == synth.blobs[(1, 1)]


@pytest.mark.parametrize("name", ["experts.bin", "labels.bin", "latent_dense_q4.bin", "latent_dense_q8.bin"])
def test_truncated_files(synth, tmp_path, name):
    dst = _copy(synth, tmp_path)
    f = dst / name
    f.write_bytes(f.read_bytes()[:-1])
    with pytest.raises(Edge0PackError) as exc:
        Edge0Pack(dst)
    msg = str(exc.value)
    assert name in msg and "expected" in msg and "found" in msg


def test_oversized_experts(synth, tmp_path):
    dst = _copy(synth, tmp_path)
    with open(dst / "experts.bin", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(Edge0PackError, match="experts.bin"):
        Edge0Pack(dst)


def test_missing_files(synth, tmp_path):
    for name in ("model.json", "native_experts.txt", "experts.bin", "labels.bin", "model-skeleton.gguf", "latent_dense_q4.bin"):
        dst = tmp_path / ("no_" + name)
        shutil.copytree(synth.path, dst)
        (dst / name).unlink()
        with pytest.raises(Edge0PackError, match=name):
            Edge0Pack(dst)


def test_label_out_of_range(synth, tmp_path):
    dst = _copy(synth, tmp_path)
    bad = synth.labels.copy()
    bad[1, 2] = syn.N_GROUPS
    (dst / "labels.bin").write_bytes(bad.astype("<i4").tobytes())
    with pytest.raises(Edge0PackError, match="out of range"):
        Edge0Pack(dst)
    bad[1, 2] = -1
    (dst / "labels.bin").write_bytes(bad.astype("<i4").tobytes())
    with pytest.raises(Edge0PackError, match="negative"):
        Edge0Pack(dst)


def test_bad_native_experts(synth, tmp_path):
    dst = _copy(synth, tmp_path)
    txt = (dst / "native_experts.txt").read_text()
    (dst / "native_experts.txt").write_text(txt.replace(" 40 40 ", " 40 40 ", 1).replace("\n1 40 40", "\n1 40 40 1"))
    with pytest.raises(Edge0PackError, match="native_experts.txt"):
        Edge0Pack(dst)
    (dst / "native_experts.txt").write_text(txt.replace("1 40 40", "1 40 41"))  # unsupported type pair
    with pytest.raises(NotImplementedError, match="NVFP4"):
        Edge0Pack(dst)


def test_q3k_raises_not_implemented(synth, tmp_path):
    dst = _copy(synth, tmp_path)
    txt = (dst / "native_experts.txt").read_text()
    lines = [ln.replace(" 40 40 ", " 11 11 ") if not ln.startswith("#") else ln for ln in txt.splitlines()]
    (dst / "native_experts.txt").write_text("\n".join(lines) + "\n")
    meta = json.loads((dst / "model.json").read_text())
    meta["build"] = "Q3_K"
    (dst / "model.json").write_text(json.dumps(meta))
    with pytest.raises(NotImplementedError, match="Q3_K"):
        Edge0Pack(dst)


def test_inconsistent_hidden_size(synth, tmp_path):
    dst = _copy(synth, tmp_path)
    meta = json.loads((dst / "model.json").read_text())
    meta["hidden"] = syn.HIDDEN + 64
    (dst / "model.json").write_text(json.dumps(meta))
    with pytest.raises(Edge0PackError):
        Edge0Pack(dst)


# -- real 120B folder ----------------------------------------------------------------


def _rms(a):
    return float(np.sqrt((a.astype(np.float64) ** 2).mean()))


@real_only
def test_real_pack_geometry_and_known_values():
    p = Edge0Pack(Path(REAL))
    assert (p.n_layers, p.n_experts, p.hidden, p.expert_ff, p.latent_dim, p.n_groups) == (48, 512, 2560, 640, 1280, 8)
    assert p.expert_format == "NVFP4" and p.default_dense_format == "q4" and p.model_name == "edge0-120b-fp4"
    assert p.labels.shape == (48, 512) and p.labels.min() == 0 and p.labels.max() == 7
    assert np.bincount(p.labels[0], minlength=8).tolist() != [64] * 8  # groups are not balanced
    ex = p.expert(0, 0)
    assert ex.s_gate == pytest.approx(0.0030459, rel=1e-3) and ex.s_up == pytest.approx(0.0030459, rel=1e-3)
    assert ex.s_down == pytest.approx(8.2107e-5, rel=1e-3)
    g, u, d = ex.scaled()
    assert _rms(g) == pytest.approx(0.1117, rel=2e-3)
    assert _rms(u) == pytest.approx(0.0918, rel=2e-3)
    assert _rms(d) == pytest.approx(0.0314, rel=2e-3)
    assert p.skeleton.metadata["general.architecture"] == "qwen4exp" and len(p.skeleton.tensors) == 1079


@real_only
def test_real_dense_layer0_known_rms_and_q4_q8_agree():
    p = Edge0Pack(Path(REAL))
    d, u = p.dense_d(0), p.dense_u(0)
    assert d.shape == (8, 1280, 2560) and u.shape == (8, 2560, 1280)
    assert _rms(d) == pytest.approx(0.00855, rel=5e-3) and _rms(u) == pytest.approx(0.00997, rel=5e-3)
    d8 = p.dense_d(0, "q8")[:1, :64]
    assert np.linalg.norm(d[:1, :64] - d8) / np.linalg.norm(d8) < 0.2


@real_only
def test_real_blob_and_dense_match_scratch_oracle():
    # independent decoders in the style of the phase0 scratch scripts (row/block reshape, no shared code)
    p = Edge0Pack(Path(REAL))
    blob = np.asarray(p.expert_blob(47, 511))
    kv = np.array(syn.KV, np.float32) * 0.5
    a = blob[:460800].reshape(640, 20, 36)
    sc = a[:, :, :4].astype(np.int32)
    e, m = sc >> 3, sc & 7
    d_ = np.where(e == 0, m * 2.0**-9, (1 + m / 8.0) * 2.0 ** (e - 7)).astype(np.float32)
    d_[(sc == 0) | (sc == 0x7F)] = 0
    qs = a[:, :, 4:].reshape(640, 20, 4, 8)
    want = (np.concatenate([kv[qs & 15], kv[qs >> 4]], -1) * d_[..., None]).reshape(640, 1280)
    np.testing.assert_array_equal(p.expert(47, 511).gate, want)

    layer = 5
    d_bytes = 8 * 1280 * 80 * 18
    off = layer * 2 * d_bytes
    raw = np.memmap(Path(REAL) / "latent_dense_q4.bin", dtype=np.uint8, mode="r", offset=off, shape=(d_bytes,))
    blocks = np.asarray(raw).reshape(-1, 18)
    dd = blocks[:, :2].copy().view("<f2").astype(np.float32)
    q = blocks[:, 2:]
    ref = ((np.concatenate([q & 15, q >> 4], 1).astype(np.float32) - 8) * dd).reshape(8, 1280, 2560)
    np.testing.assert_array_equal(p.dense_d(layer), ref)
