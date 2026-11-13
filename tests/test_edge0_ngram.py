"""Edge0 FP8 n-gram table backend for the qwen4_exp sidecar gather."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

from mtplx.edge0 import ngram as edge0_ngram
from mtplx.edge0.ggml_quants import decode_fp8_e4m3
from mtplx.models.qwen4_exp import NGramTable

from tests.edge0_synth import write_gguf

ROWS, DIM = 96, 32
SCALE = 0.00019931793212890625
REAL = os.environ.get("MTPLX_EDGE0_MODEL")
real_only = pytest.mark.skipif(not REAL, reason="MTPLX_EDGE0_MODEL not set")


@pytest.fixture(autouse=True)
def _no_prewarm(monkeypatch):
    monkeypatch.setenv("MTPLX_NGRAM_PREWARM", "off")


def _bytes(seed=0):
    b = np.random.default_rng(seed).integers(0, 256, (ROWS, DIM), dtype=np.uint8)
    b[(b & 0x7F) == 0x7F] = 0x38  # E4M3 NaN pattern
    return b


def _pack(tmp_path: Path, data, *, how="config", fmt="f8_e4m3", shape=None, gtype=24):
    write_gguf(
        tmp_path / "ple-fp8.gguf",
        {"general.architecture": "strata-ple", "strata.ple.format": fmt, "strata.ple.scale": SCALE},
        [(edge0_ngram.TENSOR, shape or data.shape, gtype, data.tobytes() * (2 if gtype == 1 else 1))],
    )
    hdr = {"__metadata__": {}}
    if how == "meta":
        hdr["__metadata__"] = {"ngram_backend": edge0_ngram.BACKEND, "ngram_gguf": "ple-fp8.gguf"}
    else:
        (tmp_path / "config.json").write_text(
            json.dumps({"mtplx_ngram_source": {"backend": edge0_ngram.BACKEND, "file": "ple-fp8.gguf"}})
        )
    blob = json.dumps(hdr).encode()
    blob += b" " * (-len(blob) % 8)
    stub = tmp_path / "ngram-table.safetensors"
    stub.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return stub


def _gather(table, ids):
    out = table(mx.array(np.asarray(ids, dtype=np.int64)))
    mx.eval(out)
    return out


@pytest.mark.parametrize("how", ["config", "meta"])
def test_rows_are_fp8_times_scale(tmp_path, how):
    data = _bytes()
    table = NGramTable(ROWS, DIM, sidecar=True)
    table.attach_sidecar(_pack(tmp_path, data, how=how))
    assert table._sidecar.bits == 0
    ids = np.array([0, 5, 5, ROWS - 1, 17], dtype=np.int64)
    out = _gather(table, ids)
    # float32 and exact: a bfloat16 result rounded most FP8 * scale products (the earlier behaviour)
    assert out.dtype == mx.float32 and out.shape == (5, DIM)
    ref = decode_fp8_e4m3(data[ids], SCALE).reshape(5, DIM)
    assert np.array_equal(np.asarray(out), np.asarray(ref, np.float32))


def test_row_numbering_is_the_gguf_row_and_off_by_one_differs(tmp_path):
    data = _bytes(1)
    table = NGramTable(ROWS, DIM, sidecar=True)
    table.attach_sidecar(_pack(tmp_path, data))
    got = np.asarray(_gather(table, [10]).astype(mx.float32))[0]
    assert not np.array_equal(got, decode_fp8_e4m3(data[11], SCALE))
    assert not np.array_equal(got, decode_fp8_e4m3(data[9], SCALE))


def test_big_gather_and_hot_cache_agree(tmp_path, monkeypatch):
    data = _bytes(2)
    stub = _pack(tmp_path, data)
    ids = np.random.default_rng(3).integers(0, ROWS, 9000)  # above the hot-row threshold
    outs = []
    for hot in ("0", "64"):
        monkeypatch.setenv("MTPLX_NGRAM_HOT_MB", hot)
        table = NGramTable(ROWS, DIM, sidecar=True)
        table.attach_sidecar(stub)
        outs.append(np.asarray(_gather(table, ids).astype(mx.float32)))
        outs.append(np.asarray(_gather(table, ids[:20]).astype(mx.float32)))
    assert np.array_equal(outs[0], outs[2]) and np.array_equal(outs[1], outs[3])
    assert np.array_equal(outs[0][:20], outs[1])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fmt": "q8"}, "strata.ple.format"),
        ({"gtype": 1}, "model expects I8"),
    ],
)
def test_rejects_wrong_file(tmp_path, kwargs, message):
    table = NGramTable(ROWS, DIM, sidecar=True)
    with pytest.raises(edge0_ngram.Edge0NgramError, match=message):
        table.attach_sidecar(_pack(tmp_path, _bytes(), **kwargs))


def test_rejects_wrong_shape(tmp_path):
    table = NGramTable(ROWS + 1, DIM, sidecar=True)
    with pytest.raises(edge0_ngram.Edge0NgramError, match="model expects"):
        table.attach_sidecar(_pack(tmp_path, _bytes()))


def test_stock_sidecar_is_not_redirected(tmp_path):
    """A normal 4-bit sidecar ignores a config.json that asks for the GGUF."""
    w = mx.random.normal((64, 64)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=32, bits=4)
    path = tmp_path / "ngram-table.safetensors"
    mx.save_safetensors(
        str(path), {"ngram.weight": wq, "ngram.scales": s, "ngram.biases": b},
        metadata={"ngram_bits": "4", "ngram_group_size": "32"},
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"mtplx_ngram_source": {"backend": edge0_ngram.BACKEND, "file": "nope.gguf"}})
    )
    table = NGramTable(64, 64, sidecar=True)
    table.attach_sidecar(path)
    assert table._sidecar.bits == 4
    ref = mx.dequantize(wq, s, b, group_size=32, bits=4)
    assert np.array_equal(np.asarray(_gather(table, [3, 9]).astype(mx.float32)),
                          np.asarray(ref.astype(mx.float32))[[3, 9]])


@real_only
def test_real_edge0_file_rows_and_ids():
    from mtplx.edge0.gguf_lite import GGUFFile
    from mtplx.models.qwen4_exp import NGramEmbedding, TextArgs

    model = Path(REAL)
    layout = edge0_ngram.skeleton_layout(model / "model-skeleton.gguf")
    args = TextArgs(
        ple_layer_ids=[layout["layers"][0] + 1], ple_embed_dim=2560, ngram_size=layout["ngram_size"],
        heads_per_ngram=layout["heads_per_ngram"], vocab_size=248320, eos_token_id=layout["eos"],
    )
    emb = NGramEmbedding(args, 0)
    assert [int(v) for v in np.array(emb.layer_multipliers)] == layout["multipliers"]
    assert [int(v) for v in np.array(emb.ngram_heads_offsets)] == layout["offsets"]
    assert [int(v) for v in np.array(emb.ngram_heads_vocab_sizes)] == layout["sizes"]
    table = emb.ngram_embedding
    entries, scale = edge0_ngram.open_table(model / "ple-fp8.gguf", table.rows, table.dim)
    raw = GGUFFile(model / "ple-fp8.gguf").raw(edge0_ngram.TENSOR).reshape(-1, table.dim)
    assert raw.shape[0] == table.rows
    ids = np.array([0, 1, table.rows // 2, layout["offsets"][-1] + layout["sizes"][-1] - 1], dtype=np.int64)
    out = edge0_ngram.decode_rows(mx.array(np.ascontiguousarray(raw[ids])), scale)
    ref = decode_fp8_e4m3(raw[ids], scale).reshape(len(ids), -1)
    assert out.dtype == mx.float32 and np.array_equal(np.asarray(out), np.asarray(ref, np.float32))
