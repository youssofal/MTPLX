"""Lossless repack of Edge0 NVFP4 experts and Q4_0/Q8_0 D/U into MLX containers (mtplx.edge0.latent_convert)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.edge0 import latent_convert as lc
from mtplx.edge0.ggml_quants import decode_nvfp4, decode_q4_0, decode_q8_0
from mtplx.edge0.pack import Edge0Pack
from tests import edge0_synth as syn

REAL = os.environ.get("MTPLX_EDGE0_MODEL")
real_only = pytest.mark.skipif(not REAL, reason="MTPLX_EDGE0_MODEL not set")


def _bits(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _deq_nvfp4(w, s) -> np.ndarray:
    return np.array(mx.dequantize(mx.array(w), mx.array(s), mode="nvfp4", dtype=mx.float32))


def _deq_affine(arrs, bits) -> np.ndarray:
    return np.array(
        mx.dequantize(
            mx.array(arrs["weight"]),
            mx.array(arrs["scales"]),
            mx.array(arrs["biases"]),
            group_size=lc.DENSE_GROUP,
            bits=bits,
            dtype=mx.float32,
        )
    )


# --------------------------------------------------------------------------- MLX's containers, pinned


def test_mlx_nvfp4_layout_and_scale_semantics():
    """What the repack relies on: nibble order, signed E4M3 scales, 0x7F = 480, code 8 = -0."""
    codes = np.tile(np.arange(16, dtype=np.uint8), (256, 1))
    w = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8).view("<u4")
    s = np.arange(256, dtype=np.uint8).reshape(256, 1)
    got = _deq_nvfp4(w, s)
    e2m1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    assert np.array_equal(got[0x38, :8], e2m1)  # 0x38 = 1.0
    assert np.array_equal(got[0x38, 8:], -e2m1)
    assert np.signbit(got[0x38, 8])  # code 8 is -0 in MLX (ggml: +0)
    assert got[0x7F, 2] == 480.0  # 0x7F is 480 in MLX (ggml: 0)
    assert got[0xB8, 2] == -1.0  # bit 7 is a sign in MLX (ggml: unsigned)
    assert np.all(got[0x00] == 0)


def test_mlx_affine_float16_scales_are_not_exact():
    """Why D/U scales are float32: with float16 scales MLX rounds scale * code in float16."""
    d = np.full((1, 1), 0.0123, np.float16)
    codes = np.arange(32, dtype=np.uint8) % 16
    w = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8).view("<u4").reshape(1, 4)
    exact = (codes.astype(np.float32) - 8) * d.astype(np.float32)[0, 0]
    out16 = np.array(mx.dequantize(mx.array(w), mx.array(d), mx.array(-8 * d), group_size=32, bits=4, dtype=mx.float32))
    out32 = _deq_affine({"weight": w, "scales": d.astype(np.float32), "biases": -8 * d.astype(np.float32)}, 4)
    assert not np.array_equal(out16[0], exact)
    assert np.array_equal(_bits(out32[0]), _bits(exact))


# --------------------------------------------------------------------------- NVFP4


def test_nvfp4_repack_bit_exact_all_code_points():
    rng = np.random.default_rng(0)
    rows, cols = 64, 256
    codes = rng.integers(0, 16, rows * cols, dtype=np.uint8)
    codes[:16] = np.arange(16)  # every code, code 8 included
    scales = rng.integers(0, 0x80, rows * cols // 16, dtype=np.uint8)  # every legal byte, 0x00 and 0x7F included
    scales[:128] = np.arange(128)
    raw = syn.pack_nvfp4(codes, scales)
    w, s = lc.nvfp4_to_mlx(raw, rows, cols)
    assert w.dtype == np.uint32 and w.shape == (rows, cols // 8)
    assert s.dtype == np.uint8 and s.shape == (rows, cols // 16)
    ref = decode_nvfp4(raw).reshape(rows, cols)
    np.testing.assert_array_equal(ref.ravel(), np.array(syn.ref_nvfp4(raw), np.float32))
    assert np.array_equal(_bits(_deq_nvfp4(w, s)), _bits(ref))  # sign of zero included


def test_nvfp4_stacked_experts_and_ggml_byte_order():
    """value j and j + 8 share a ggml byte; in MLX values 2i and 2i + 1 do."""
    codes = np.zeros(64, np.uint8)
    codes[0], codes[8], codes[1] = 1, 2, 3
    raw = syn.pack_nvfp4(codes, np.full(4, 0x38, np.uint8))
    assert raw[4] == 0x21 and raw[5] == 0x03  # ggml: byte 0 = (v0, v8), byte 1 = (v1, v9)
    w, _ = lc.nvfp4_to_mlx(raw, 1, 64)
    assert int(w[0, 0]) & 0xFF == 0x31  # MLX: byte 0 = (v0, v1)
    assert (int(w[0, 1]) & 0xF) == 2  # v8 is the low nibble of word 1

    rng = np.random.default_rng(1)
    e, r, c = 3, 16, 128
    codes = rng.integers(0, 16, (e, r * c), dtype=np.uint8)
    scales = rng.integers(1, 0x7F, (e, r * c // 16), dtype=np.uint8)
    w, s = lc.nvfp4_codes_to_mlx(codes, scales, (e, r, c))
    assert w.shape == (e, r, c // 8) and s.shape == (e, r, c // 16)
    for i in range(e):
        raw = syn.pack_nvfp4(codes[i], scales[i])
        assert np.array_equal(_bits(_deq_nvfp4(w[i], s[i])), _bits(decode_nvfp4(raw).reshape(r, c)))


def test_nvfp4_scale_bytes_with_bit7_are_refused():
    codes = np.zeros(64, np.uint8)
    raw = syn.pack_nvfp4(codes, np.array([0x38, 0x80, 0x38, 0x38], np.uint8))
    with pytest.raises(lc.Edge0ConvertError, match="0x80"):
        lc.nvfp4_to_mlx(raw, 1, 64)


# --------------------------------------------------------------------------- Q4_0 / Q8_0


@pytest.mark.parametrize("fmt", ["q4", "q8"])
def test_dense_repack_bit_exact(fmt):
    rng = np.random.default_rng(2)
    rows, cols = 24, 96
    n = rows * cols
    # wide d range: normal and subnormal float16, both signs
    d = (rng.uniform(0.5, 1.0, n // 32) * np.exp2(rng.integers(-24, 4, n // 32)) * rng.choice([-1, 1], n // 32)).astype(np.float16)
    if fmt == "q4":
        raw = syn.pack_q4_0(rng.integers(0, 16, n, dtype=np.uint8), d)
        ref = decode_q4_0(raw)
        np.testing.assert_array_equal(ref, np.array(syn.ref_q4_0(raw), np.float32))
    else:
        raw = syn.pack_q8_0(rng.integers(-128, 128, n).astype(np.int8), d)
        ref = decode_q8_0(raw)
    arrs = lc.dense_to_affine(raw, fmt, rows, cols)
    bits = lc.DENSE_BITS[fmt]
    assert arrs["weight"].shape == (rows, cols * bits // 32) and arrs["weight"].dtype == np.uint32
    assert arrs["scales"].dtype == np.float32 and arrs["biases"].shape == (rows, cols // 32)
    got, ref = _deq_affine(arrs, bits), ref.reshape(rows, cols)
    np.testing.assert_array_equal(got, ref)
    # bit for bit, except that a zero is always +0 in MLX (d * c + (-c * d)) where ggml has -0 for d < 0
    nz = ref != 0
    assert np.array_equal(_bits(got)[nz], _bits(ref)[nz])


def test_regroup_u_layout():
    rng = np.random.default_rng(3)
    g, h, lat = 3, 8, 64
    n = g * h * lat
    raw = syn.pack_q4_0(rng.integers(0, 16, n, dtype=np.uint8), rng.uniform(0.01, 0.02, n // 32).astype(np.float16))
    u = decode_q4_0(raw).reshape(g, h, lat)
    arrs = lc.regroup_u(lc.dense_to_affine(raw, "q4", g * h, lat), g)
    got = _deq_affine(arrs, 4)
    assert got.shape == (h, g * lat)
    np.testing.assert_array_equal(got, u.transpose(1, 0, 2).reshape(h, g * lat))


# --------------------------------------------------------------------------- a whole (synthetic) pack


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return syn.make_pack(tmp_path_factory.mktemp("edge0-latent") / "pack")


def _check_layer(pack: Edge0Pack, layer: int, arrays: dict, experts=None, fmt: str | None = None) -> None:
    g, lat, hid, ff, n_exp = pack.n_groups, pack.latent_dim, pack.hidden, pack.expert_ff, pack.n_experts
    fmt = fmt or pack.default_dense_format
    bits = lc.DENSE_BITS[fmt]
    shapes = {
        "experts.gate_proj.weight": (n_exp, ff, lat // 8),
        "experts.gate_proj.scales": (n_exp, ff, lat // 16),
        "experts.up_proj.weight": (n_exp, ff, lat // 8),
        "experts.up_proj.scales": (n_exp, ff, lat // 16),
        "experts.down_proj.weight": (n_exp, lat, ff // 8),
        "experts.down_proj.scales": (n_exp, lat, ff // 16),
        "experts.scalars": (n_exp, 3),
        "dense_d.weight": (g * lat, hid * bits // 32),
        "dense_d.scales": (g * lat, hid // 32),
        "dense_d.biases": (g * lat, hid // 32),
        "dense_u.weight": (hid, g * lat * bits // 32),
        "dense_u.scales": (hid, g * lat // 32),
        "dense_u.biases": (hid, g * lat // 32),
        "labels": (n_exp,),
    }
    assert {k: tuple(v.shape) for k, v in arrays.items()} == shapes
    np.testing.assert_array_equal(np.asarray(arrays["labels"]), pack.labels[layer])
    for e in experts if experts is not None else range(n_exp):
        ex = pack.expert(layer, e)
        for name, ref in (("gate_proj", ex.gate), ("up_proj", ex.up), ("down_proj", ex.down)):
            got = _deq_nvfp4(np.asarray(arrays[f"experts.{name}.weight"][e]), np.asarray(arrays[f"experts.{name}.scales"][e]))
            assert np.array_equal(_bits(got), _bits(ref)), (layer, e, name)
        sc = np.asarray(arrays["experts.scalars"][e])
        assert (float(sc[0]), float(sc[1]), float(sc[2])) == (ex.s_gate, ex.s_up, ex.s_down)
    d = _deq_affine({k: np.asarray(arrays[f"dense_d.{k}"]) for k in ("weight", "scales", "biases")}, bits)
    np.testing.assert_array_equal(d, pack.dense_d(layer, fmt).reshape(g * lat, hid))
    u = _deq_affine({k: np.asarray(arrays[f"dense_u.{k}"]) for k in ("weight", "scales", "biases")}, bits)
    np.testing.assert_array_equal(u, pack.dense_u(layer, fmt).transpose(1, 0, 2).reshape(hid, g * lat))


@pytest.mark.parametrize("fmt", ["q4", "q8"])
def test_convert_layer_synthetic(synth, fmt):
    pack = Edge0Pack(synth.path)
    for layer in range(pack.n_layers):
        _check_layer(pack, layer, lc.convert_layer(pack, layer, dense_format=fmt), fmt=fmt)


def test_write_latent_safetensors_synthetic(synth, tmp_path):
    pack = Edge0Pack(synth.path)
    index_path = lc.write_latent_safetensors(pack, tmp_path / "out")
    index = json.loads(index_path.read_text())
    assert index["metadata"]["n_groups"] == str(pack.n_groups) and index["metadata"]["dense_bits"] == "4"
    for layer in range(pack.n_layers):
        prefix = lc.DEFAULT_KEY_PREFIX.format(layer=layer)
        fname = index["weight_map"][prefix + "labels"]
        loaded, meta = mx.load(str(tmp_path / "out" / fname), return_metadata=True)
        assert meta["layer"] == str(layer)
        expect = lc.convert_layer(pack, layer)
        assert set(loaded) == {prefix + k for k in expect}
        for k, v in expect.items():
            got = np.array(loaded[prefix + k])
            assert got.dtype == v.dtype and np.array_equal(got, v), k
    assert not list((tmp_path / "out").glob(".*partial*"))


# --------------------------------------------------------------------------- the real folder


@pytest.fixture(scope="module")
def real_pack():
    return Edge0Pack(Path(REAL))


@real_only
@pytest.mark.parametrize("layer", [0, 23, 47])
def test_real_layer_repack_exact(real_pack, layer):
    arrays = lc.convert_layer(real_pack, layer)
    lab = real_pack.labels[layer]
    picks = sorted({0, real_pack.n_experts - 1, int(np.argmin(np.bincount(lab))), 137 % real_pack.n_experts})
    picks = sorted({*picks, *(int(np.flatnonzero(lab == g)[0]) for g in range(real_pack.n_groups))})
    _check_layer(real_pack, layer, arrays, experts=picks)


@real_only
def test_real_code_points_mlx_and_ggml_disagree_on_do_not_occur(real_pack):
    scan = lc.scan_code_points(real_pack, [0, 23, 47])
    for layer, c in scan.items():
        assert c["scale_ge_80"] == 0 and c["scale_7f"] == 0, (layer, c)
        assert c["max_scale"] < 0x7F


@real_only
def test_real_q8_dense_repack_exact(real_pack):
    if not (real_pack.path / "latent_dense_q8.bin").exists():
        pytest.skip("no latent_dense_q8.bin in this folder")
    arrays = lc.convert_dense(real_pack, 23, "q8")
    d = _deq_affine({k: arrays[f"dense_d.{k}"] for k in ("weight", "scales", "biases")}, 8)
    np.testing.assert_array_equal(d, real_pack.dense_d(23, "q8").reshape(d.shape))


@real_only
def test_real_write_two_layers(real_pack, tmp_path):
    out = tmp_path / "latent"
    index_path = lc.write_latent_safetensors(real_pack, out, layers=[0, 47])
    index = json.loads(index_path.read_text())
    assert sorted(set(index["weight_map"].values())) == ["latent-moe-000.safetensors", "latent-moe-047.safetensors"]
    loaded = mx.load(str(out / "latent-moe-047.safetensors"))
    prefix = lc.DEFAULT_KEY_PREFIX.format(layer=47)
    expect = lc.convert_layer(real_pack, 47)
    for k in ("experts.down_proj.weight", "experts.down_proj.scales", "experts.scalars", "dense_u.scales", "labels"):
        assert np.array_equal(np.array(loaded[prefix + k]), expect[k]), k
