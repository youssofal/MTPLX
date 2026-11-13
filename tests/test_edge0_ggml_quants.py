from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mtplx.edge0 import ggml_quants as gq
from tests import edge0_synth as syn

REAL = os.environ.get("MTPLX_EDGE0_MODEL")
real_only = pytest.mark.skipif(not REAL, reason="MTPLX_EDGE0_MODEL not set")


def test_ue4m3_known_values():
    got = gq.ue4m3_to_float(np.array([0x00, 0x7F, 0x01, 0x07, 0x08, 0x38, 0x39, 0x7E], np.uint8))
    # subnormal m*2^-9, then (1+m/8)*2^(e-7): 0x38 -> e=7 -> 1.0, 0x39 -> 1.125, 0x7E -> 1.75*2^8
    assert got.tolist() == [0.0, 0.0, 2.0**-9, 7 * 2.0**-9, 2.0**-6, 1.0, 1.125, 448.0]
    assert gq.ue4m3_to_float(np.arange(256, dtype=np.uint8)).dtype == np.float32


def test_nvfp4_hand_built_block():
    # scale bytes: 0x38 (1.0), 0x40 (2.0), 0x00 (0), 0x30 (0.5); value = kv[code] * scale * 0.5
    d = [0x38, 0x40, 0x00, 0x30]
    qs = bytearray(32)
    # sub-block 0: byte j = low nibble j, high nibble 15 - j  (value j and value j + 8)
    for j in range(8):
        qs[j] = j | ((15 - j) << 4)
    # sub-block 1: all bytes 0x75 -> low code 5 (kv 6), high code 7 (kv 12)
    for j in range(8, 16):
        qs[j] = 0x75
    # sub-block 2 has scale 0; sub-block 3: byte 24 = 0x4D -> low 13 (kv -6), high 4 (kv 4)
    qs[24] = 0x4D
    block = bytes(d) + bytes(qs)
    codes, scales = gq.unpack_nvfp4(block)
    assert scales.tolist() == d
    assert codes.dtype == np.uint8 and codes.size == 64
    assert codes[:8].tolist() == list(range(8))
    assert codes[8:16].tolist() == [15 - j for j in range(8)]
    assert codes[16:24].tolist() == [5] * 8 and codes[24:32].tolist() == [7] * 8
    assert codes[48] == 13 and codes[56] == 4
    out = gq.decode_nvfp4(block)
    assert out.dtype == np.float32
    kv = syn.KV
    assert out[:8].tolist() == [kv[j] * 0.5 for j in range(8)]
    assert out[8:16].tolist() == [kv[15 - j] * 0.5 for j in range(8)]
    assert out[16:24].tolist() == [6 * 2.0 * 0.5] * 8
    assert out[24:32].tolist() == [12 * 2.0 * 0.5] * 8
    assert out[32:48].tolist() == [0.0] * 16
    assert out[48] == -6 * 0.5 * 0.5 and out[56] == 4 * 0.5 * 0.5


def test_nvfp4_matches_scalar_reference_and_roundtrips():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, 64 * 5, dtype=np.uint8)
    scales = rng.integers(0, 256, 64 * 5 // 16, dtype=np.uint8)
    raw = syn.pack_nvfp4(codes, scales)
    assert len(raw) == 5 * 36
    c2, s2 = gq.unpack_nvfp4(raw)
    assert np.array_equal(c2, codes) and np.array_equal(s2, scales)
    np.testing.assert_array_equal(gq.decode_nvfp4(raw), np.array(syn.ref_nvfp4(raw), np.float32))


def test_q4_0_hand_built_block():
    d = np.float16(0.25)
    qs = bytes(j | ((15 - j) << 4) for j in range(16))
    block = d.tobytes() + qs
    codes, scales = gq.unpack_q4_0(block)
    assert codes.tolist() == list(range(16)) + [15 - j for j in range(16)]
    assert scales.dtype == np.float16 and scales.tolist() == [0.25]
    out = gq.decode_q4_0(block)
    assert out[:16].tolist() == [(j - 8) * 0.25 for j in range(16)]
    assert out[16:].tolist() == [(15 - j - 8) * 0.25 for j in range(16)]


def test_q8_0_hand_built_block():
    d = np.float16(0.5)
    q = np.arange(-16, 16, dtype=np.int8)
    block = d.tobytes() + q.tobytes()
    qq, scales = gq.unpack_q8_0(block)
    assert qq.dtype == np.int8 and qq.tolist() == q.tolist() and scales.tolist() == [0.5]
    assert gq.decode_q8_0(block).tolist() == (q.astype(np.float32) * 0.5).tolist()


def test_q4_q8_match_scalar_reference():
    rng = np.random.default_rng(2)
    d = rng.uniform(0.01, 1, 6).astype(np.float16)
    raw4 = syn.pack_q4_0(rng.integers(0, 16, 6 * 32, dtype=np.uint8), d)
    np.testing.assert_array_equal(gq.decode_q4_0(raw4), np.array(syn.ref_q4_0(raw4), np.float32))
    raw8 = syn.pack_q8_0(rng.integers(-128, 128, 6 * 32, dtype=np.int8), d)
    np.testing.assert_array_equal(gq.decode_q8_0(raw8), np.array(syn.ref_q8_0(raw8), np.float32))
    c, s = gq.unpack_q4_0(raw4)
    assert syn.pack_q4_0(c, s) == raw4


def test_bf16_and_fp8():
    bits = np.array([0x3F80, 0xC000, 0x0000, 0x3F00], "<u2")
    assert gq.decode_bf16(bits.tobytes()).tolist() == [1.0, -2.0, 0.0, 0.5]
    # E4M3: 0x38 = 1.0, 0xB8 = -1.0, 0x01 = 2^-9, 0x7E = 448, 0x7F = NaN
    out = gq.decode_fp8_e4m3(bytes([0x38, 0xB8, 0x01, 0x7E, 0x7F, 0x00]), scale=0.5)
    assert out[:4].tolist() == [0.5, -0.5, 2.0**-10, 224.0]
    assert np.isnan(out[4]) and out[5] == 0.0


def test_bad_lengths_raise():
    for fn in (gq.unpack_nvfp4, gq.unpack_q4_0, gq.unpack_q8_0):
        with pytest.raises(ValueError):
            fn(b"\0" * 5)
    with pytest.raises(ValueError):
        gq.decode_bf16(b"\0\0\0")


@real_only
def test_real_nvfp4_blob_matches_phase0_decoder():
    from mtplx.edge0.pack import Edge0Pack

    pack = Edge0Pack(Path(REAL))
    blob = np.asarray(pack.expert_blob(0, 0))
    p = pack._proj_bytes
    ff, lat = pack.expert_ff, pack.latent_dim  # 640 x 1280 on the 120B, 640 x 1792 on the 145B
    got = gq.decode_nvfp4(blob[:p]).reshape(ff, lat)
    kv = np.array(syn.KV, np.float32) * 0.5
    a = blob[:p].reshape(ff, lat // 64, 36)
    d = gq.ue4m3_to_float(a[:, :, :4])
    qs = a[:, :, 4:].reshape(ff, lat // 64, 4, 8)
    want = (np.concatenate([kv[qs & 15], kv[qs >> 4]], -1) * d[..., None]).reshape(ff, lat)
    np.testing.assert_array_equal(got, want)
