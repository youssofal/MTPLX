"""CPU-only contracts for the owner-thread native PLE row handoff."""

from __future__ import annotations

import ast
from collections import OrderedDict
import gc
import importlib
from pathlib import Path
import weakref

import numpy as np
import pytest


MODULE = "mtplx.ple_cached_row_handoff"


class _Matrix:
    def __init__(self, rows: int, width: int, dtype) -> None:
        self.shape = (rows, width)
        self.dtype = np.dtype(dtype)
        self.nbytes = rows * width * self.dtype.itemsize
        self._data = np.arange(rows * width, dtype=self.dtype).reshape(rows, width)

    def __getitem__(self, key):
        return self._data[key]


class _Sidecar:
    _HOT_PATH_MAX_ROWS = 4096

    def __init__(self, *, capacity: int = 8, rows: int = 256) -> None:
        self._maps = {
            "weight": (_Matrix(rows, 20, np.uint32), "U32"),
            "scales": (_Matrix(rows, 5, np.uint16), "BF16"),
            "biases": (_Matrix(rows, 5, np.uint16), "BF16"),
        }
        self._hot = OrderedDict()
        self._hot_row_bytes = 100
        self._hot_cap_rows = capacity
        self._pool = None
        self.hot_hits = 0
        self.hot_misses = 0


def _payload(row: int):
    return (
        np.arange(20, dtype=np.uint32) + row * 1000,
        np.arange(5, dtype=np.uint16) + row * 100,
        np.arange(5, dtype=np.uint16) + row * 10,
    )


def _module():
    return importlib.import_module(MODULE)


def _ids(*values: int) -> np.ndarray:
    values = tuple(values)
    assert len(values) <= 64
    filler = values[-1] if values else 0
    return np.asarray(values + (filler,) * (64 - len(values)), dtype=np.uint32)


def _install_hot(sidecar: _Sidecar, *rows: int) -> None:
    for row in rows:
        sidecar._hot[row] = _payload(row)


def _packed_row(module, payload) -> np.ndarray:
    return module.pack_row_payload(payload)


def test_import_is_cpu_only_and_exposes_minimal_prepared_shape():
    had_mlx = "mlx.core" in __import__("sys").modules
    module = _module()
    if not had_mlx:
        assert "mlx.core" not in __import__("sys").modules
    assert module.MAX_ROW_SLOTS == 64
    assert module.PACKED_ROW_BYTES == 100
    assert tuple(module.PreparedRows.__dataclass_fields__) == (
        "source",
        "hit_packed",
        "miss_ids",
        "touch_order",
        "touch_source",
    )


def test_hit_only_is_compact_immutable_and_touches_sorted_stock_order():
    module = _module()
    sidecar = _Sidecar()
    _install_hot(sidecar, 9, 2)
    handoff = module.bind_stock_cache(sidecar)

    prepared = handoff.prepare(_ids(9, 2, 9, 2))

    assert prepared.miss_ids.tolist() == []
    assert prepared.hit_packed.shape == (2, 100)
    assert prepared.source[:4].tolist() == [0x81, 0x80, 0x81, 0x80]
    assert not prepared.source.flags.writeable
    assert not prepared.hit_packed.flags.writeable
    assert not prepared.miss_ids.flags.writeable
    assert not prepared.touch_order.flags.writeable
    assert not prepared.touch_source.flags.writeable
    assert prepared.touch_order.tolist() == [2, 9]
    empty = np.empty((0, 100), dtype=np.uint8)
    handoff.checked_publish(
        handoff.checked_completion(prepared, prepared.miss_ids, empty)
    )
    # np.unique sorts [2, 9], and the final stock touch loop follows that order.
    assert list(sidecar._hot) == [2, 9]


def test_mixed_duplicates_preserve_np_unique_order_and_compact_scatter():
    module = _module()
    sidecar = _Sidecar()
    _install_hot(sidecar, 9, 2)
    handoff = module.bind_stock_cache(sidecar)

    prepared = handoff.prepare(_ids(9, 2, 7, 9, 7, 2))

    # Sorted unique IDs are [2(hit compact 0), 7(miss compact 0), 9(hit 1)].
    assert prepared.miss_ids.tolist() == [7]
    assert prepared.source[:6].tolist() == [0x81, 0x80, 0x00, 0x81, 0x00, 0x80]
    assert prepared.hit_packed.shape == (2, 100)
    assert prepared.touch_order.tolist() == [2, 7, 9]
    assert prepared.touch_source.tolist() == [0x80, 0x00, 0x81]


def test_hit_payload_bytes_are_exact_weight_then_metadata():
    module = _module()
    sidecar = _Sidecar()
    _install_hot(sidecar, 3)
    prepared = module.bind_stock_cache(sidecar).prepare(_ids(3))

    expected = _packed_row(module, _payload(3))
    np.testing.assert_array_equal(prepared.hit_packed[0], expected)
    assert prepared.hit_packed.flags.c_contiguous


def test_input_is_snapshotted_before_caller_mutation():
    module = _module()
    sidecar = _Sidecar()
    ids = _ids(11, 12)
    prepared = module.bind_stock_cache(sidecar).prepare(ids)
    ids[:] = 99

    assert prepared.miss_ids.tolist() == [11, 12]
    assert prepared.source[:2].tolist() == [0, 1]


def test_checked_completion_rejects_bad_ids_or_shape_before_publish():
    module = _module()
    sidecar = _Sidecar()
    handoff = module.bind_stock_cache(sidecar)
    prepared = handoff.prepare(_ids(4, 5))
    packed = np.vstack((_packed_row(module, _payload(4)), _packed_row(module, _payload(5))))
    before = list(sidecar._hot.items())

    with pytest.raises(ValueError, match="miss IDs"):
        handoff.checked_completion(prepared, np.asarray([5, 4], dtype=np.uint32), packed)
    with pytest.raises(ValueError, match="packed"):
        handoff.checked_completion(prepared, prepared.miss_ids, packed[:1])
    assert list(sidecar._hot.items()) == before


def test_publish_inserts_exact_payload_and_evicts_using_stock_limit():
    module = _module()
    sidecar = _Sidecar(capacity=2)
    _install_hot(sidecar, 1, 2)
    handoff = module.bind_stock_cache(sidecar)
    prepared = handoff.prepare(_ids(2, 3))
    packed = _packed_row(module, _payload(3)).reshape(1, 100)
    ticket = handoff.checked_completion(prepared, prepared.miss_ids, packed)

    handoff.checked_publish(ticket)

    assert list(sidecar._hot) == [2, 3]
    got = sidecar._hot[3]
    for actual, expected in zip(got, _payload(3)):
        np.testing.assert_array_equal(actual, expected)
    assert sidecar._hot_row_bytes == 100
    assert len(sidecar._hot) <= sidecar._hot_cap_rows


def test_publish_restores_prepared_hit_evicted_while_native_batch_was_inflight():
    module = _module()
    sidecar = _Sidecar(capacity=3)
    _install_hot(sidecar, 2, 9)
    handoff = module.bind_stock_cache(sidecar)
    prepared = handoff.prepare(_ids(9, 7, 2))
    packed = _packed_row(module, _payload(7)).reshape(1, 100)

    # A stock/eager/short-tail gather may use the same owner-thread LRU while
    # the native miss batch is outstanding.  Row 2 was a hit at prepare time,
    # but is gone before publication.
    del sidecar._hot[2]
    ticket = handoff.checked_completion(prepared, prepared.miss_ids, packed)
    handoff.checked_publish(ticket)

    assert list(sidecar._hot) == [2, 7, 9]
    for actual, expected in zip(sidecar._hot[2], _payload(2)):
        np.testing.assert_array_equal(actual, expected)
    assert len(sidecar._hot) <= sidecar._hot_cap_rows


def test_cached_row_does_not_retain_full_trusted_completed_batch():
    module = _module()
    sidecar = _Sidecar(capacity=1)
    handoff = module.bind_stock_cache(sidecar)
    prepared = handoff.prepare(_ids(17))
    batch = np.vstack(
        (_packed_row(module, _payload(17)), _packed_row(module, _payload(18)))
    )
    batch_ref = weakref.ref(batch)
    ticket = handoff.trusted_completion(prepared, batch[:1])
    handoff.publish(ticket)
    del ticket, prepared, batch
    gc.collect()

    assert batch_ref() is None
    cached = sidecar._hot[17]
    assert all(array.nbytes == expected for array, expected in zip(cached, (80, 10, 10)))
    for array in cached:
        owner = array
        while isinstance(getattr(owner, "base", None), np.ndarray):
            owner = owner.base
        assert owner.nbytes <= 100


def test_checked_completion_is_immutable_and_binds_to_one_handoff():
    module = _module()
    first = module.bind_stock_cache(_Sidecar())
    second = module.bind_stock_cache(_Sidecar())
    prepared = first.prepare(_ids(21))
    packed = _packed_row(module, _payload(21)).reshape(1, 100)
    ticket = first.checked_completion(prepared, prepared.miss_ids, packed)
    assert not ticket.packed.flags.writeable
    with pytest.raises(ValueError):
        ticket.packed[0, 0] = 1
    with pytest.raises(ValueError, match="handoff"):
        second.checked_publish(ticket)


def test_prepare_rejects_non_fixed64_input_at_boundary():
    module = _module()
    handoff = module.bind_stock_cache(_Sidecar())
    with pytest.raises(ValueError, match="64"):
        handoff.checked_prepare(np.arange(63, dtype=np.uint32))


def _stock_rows_matrices_oracle():
    """Extract the shipped stock implementation, not a copied expectation."""

    source = (Path(__file__).resolve().parents[1] / "mtplx/models/qwen4_exp.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_stack_hot_rows"
    }
    sidecar_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_SidecarGather"
    )
    wanted["_rows_matrices"] = next(
        node
        for node in sidecar_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_rows_matrices"
    )
    assert set(wanted) == {"_stack_hot_rows", "_rows_matrices"}
    namespace = {"np": np}
    module = ast.Module(body=[wanted["_stack_hot_rows"], wanted["_rows_matrices"]], type_ignores=[])
    exec(compile(module, "qwen4_exp.py:stock-cache-oracle", "exec"), namespace)
    return namespace["_rows_matrices"]


def test_mixed_hit_miss_lru_matches_ast_extracted_stock_oracle():
    module = _module()
    oracle_rows_matrices = _stock_rows_matrices_oracle()
    sidecar = _Sidecar(capacity=2)
    _install_hot(sidecar, 2, 9)
    oracle = _Sidecar(capacity=2)
    _install_hot(oracle, 2, 9)
    ids = _ids(9, 7, 2)

    oracle_rows_matrices(oracle, ids, ("weight", "scales", "biases"))
    handoff = module.bind_stock_cache(sidecar)
    prepared = handoff.prepare(ids)
    packed = np.vstack(
        [
            module.pack_row_payload(
                tuple(matrix[7] for matrix, _dtype_name in sidecar._maps.values())
            )
        ]
    )
    handoff.publish(handoff.trusted_completion(prepared, packed))

    assert list(sidecar._hot) == list(oracle._hot)


def test_trusted_methods_do_not_call_checked_boundaries(monkeypatch):
    module = _module()
    sidecar = _Sidecar()
    handoff = module.bind_stock_cache(sidecar)

    def fail(*_args, **_kwargs):
        raise AssertionError("checked boundary called by trusted path")

    monkeypatch.setattr(type(handoff), "checked_prepare", fail)
    monkeypatch.setattr(type(handoff), "checked_publish", fail)
    prepared = handoff.prepare(_ids(31))
    packed = _packed_row(module, _payload(31)).reshape(1, 100)
    handoff.publish(handoff.trusted_completion(prepared, packed))
    assert 31 in sidecar._hot
