import os

import numpy as np
import pytest
from safetensors.numpy import save_file

from mtplx.ngram_row_layout import convert, header, open_rows


@pytest.mark.parametrize("quantized", [False, True])
def test_interleaved_rows_are_exact_and_stale_cache_is_rejected(tmp_path, quantized):
    rng = np.random.default_rng(449)
    tensors = {"ngram.weight": rng.integers(0, 2**32, (73, 20), dtype=np.uint32)}
    if quantized:
        # F16 values cover the entire raw bit range, including NaN payloads.
        for name in ("scales", "biases"):
            tensors["ngram." + name] = rng.integers(
                0, 65536, (73, 5), dtype=np.uint16
            ).view(np.float16)
    source = tmp_path / "ngram-table.safetensors"
    target = tmp_path / "rows.safetensors"
    save_file(tensors, str(source))
    original = source.read_bytes()
    receipt = convert(source, target, chunk_rows=17)
    assert receipt["all_rows_bit_exact"]
    h, start = header(source)
    entries = {
        name.removeprefix("ngram."): (info, start)
        for name, info in h.items()
        if name != "__metadata__"
    }
    maps, (offset, stride) = open_rows(target, source, entries)
    rows = np.array([72, 0, 4, 4, 17, 1])
    for name, (mapped, _) in maps.items():
        expected = tensors["ngram." + name][rows]
        assert mapped[rows].tobytes() == expected.tobytes()
        assert mapped.strides[0] == stride
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        convert(source, target)
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1))
    with pytest.raises(ValueError, match="stale"):
        open_rows(target, source, entries)


def test_sidecar_uses_one_record_read_and_preserves_gathered_values(
    tmp_path, monkeypatch
):
    from mtplx.models.qwen4_exp import NGramTable

    tensors = {
        "ngram.weight": np.arange(240, dtype=np.uint32).reshape(12, 20),
        "ngram.scales": np.ones((12, 5), dtype=np.float16),
        "ngram.biases": np.zeros((12, 5), dtype=np.float16),
    }
    source = tmp_path / "ngram-table.safetensors"
    save_file(
        tensors, str(source), metadata={"ngram_bits": "4", "ngram_group_size": "32"}
    )
    target = tmp_path / "rows.safetensors"
    convert(source, target)
    monkeypatch.setenv("MTPLX_NGRAM_PREWARM", "off")
    monkeypatch.setenv("MTPLX_NGRAM_HOT_MB", "0")
    monkeypatch.delenv("MTPLX_NGRAM_ROW_FILE", raising=False)
    old = NGramTable(12, 160, sidecar=True)
    old.attach_sidecar(source)
    monkeypatch.setenv("MTPLX_NGRAM_ROW_FILE", str(target))
    new = NGramTable(12, 160, sidecar=True)
    new.attach_sidecar(source)
    assert new._sidecar.row_layout == "interleaved"
    assert len(old._sidecar._row_meta) == 3
    assert len(new._sidecar._row_meta) == 1
    ids = np.array([11, 0, 1, 1, 9])
    names = ("weight", "scales", "biases")
    before = old._sidecar._rows_matrices(ids, names)
    after = new._sidecar._rows_matrices(ids, names)
    for name in names:
        assert before[name].tobytes() == after[name].tobytes()
    for table in (old, new):
        table._sidecar._pool.shutdown(wait=True)
        os.close(table._sidecar._fd)
