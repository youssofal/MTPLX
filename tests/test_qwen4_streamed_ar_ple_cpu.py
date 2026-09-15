"""CPU-only protocol tests for streamed PLE leaves in pipelined AR."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


MODULE = "mtplx.ple_streamed_ar"


class _Plane:
    def __init__(self, name: str) -> None:
        self.name = name


class _DeferredRows:
    def __init__(self) -> None:
        self._planes = (_Plane("weight"), _Plane("scales"), _Plane("biases"))
        self.fills: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def planes(self):
        return self._planes

    def fill(self, weight, scales, biases) -> None:
        if self.fills:
            raise RuntimeError("deferred AR rows already filled")
        self.fills.append(
            tuple(np.array(value, copy=True) for value in (weight, scales, biases))
        )


class _Native:
    def __init__(self) -> None:
        self.handles: list[_DeferredRows] = []

    def make_deferred_ar_rows(self):
        handle = _DeferredRows()
        self.handles.append(handle)
        return handle

    @staticmethod
    def make_deferred_ar_token():
        return object()


class _Embedding:
    shape = (1, 1, 640)

    def reshape(self, *shape):
        assert shape == self.shape
        return self


class _MX:
    int64 = np.int64

    def __init__(self) -> None:
        self.dequantize_calls = []

    @staticmethod
    def full(shape, value, *, dtype):
        return np.full(shape, value, dtype=dtype)

    @staticmethod
    def array(value):
        return np.array(value, copy=True)

    def dequantize(self, weight, scales, biases, *, group_size, bits):
        self.dequantize_calls.append(
            (weight, scales, biases, int(group_size), int(bits))
        )
        return _Embedding()


def _adapter():
    module = importlib.import_module(MODULE)
    native = _Native()
    mx = _MX()
    row_calls = []
    gather_calls = []

    def rows(ids, previous):
        row_calls.append((np.array(ids, copy=True), np.array(previous, copy=True)))
        token = int(ids[0, 0])
        row_ids = np.arange(16, dtype=np.int64).reshape(1, 1, 16) + token * 100
        history = np.array([[int(previous[0, -1]), token]], dtype=np.int64)
        return row_ids, history

    def gather(flat):
        gather_calls.append(np.array(flat, copy=True))
        return (
            np.arange(16 * 20, dtype=np.uint32).reshape(16, 20),
            np.arange(16 * 5, dtype=np.uint16).reshape(16, 5),
            (np.arange(16 * 5, dtype=np.uint16) + 500).reshape(16, 5),
        )

    adapter = module.StreamedArPle(
        native_module=native,
        mx_module=mx,
        rows=rows,
        gather_planes=gather,
        eos_id=99,
        context_len=2,
        output_dim=640,
        bits=4,
        group_size=32,
    )
    return adapter, native, mx, row_calls, gather_calls


def test_build_records_one_pending_leaf_without_advancing_history():
    adapter, native, mx, row_calls, gather_calls = _adapter()
    adapter.set_active(True)
    cache = [None, None, None, np.array([[7, 8]], dtype=np.int64)]

    result = adapter.build(np.array([[9]], dtype=np.int64), cache, 3)

    assert result.shape == (1, 1, 640)
    np.testing.assert_array_equal(cache[3], [[7, 8]])
    assert row_calls == []
    assert gather_calls == []
    assert len(native.handles) == 1
    assert native.handles[0].fills == []
    assert mx.dequantize_calls[0][:3] == native.handles[0].planes()
    assert mx.dequantize_calls[0][3:] == (32, 4)


def test_token_factory_is_bound_to_the_validated_native_module():
    adapter, _native, _mx, _row_calls, _gather_calls = _adapter()
    assert adapter.make_token() is not None


def test_flush_materializes_ids_fills_exact_planes_then_commits_history():
    adapter, native, _mx, row_calls, gather_calls = _adapter()
    adapter.set_active(True)
    cache = [None, None, None, np.array([[7, 8]], dtype=np.int64)]
    adapter.build(np.array([[9]], dtype=np.int64), cache, 3)

    adapter.flush()

    np.testing.assert_array_equal(row_calls[0][0], [[9]])
    np.testing.assert_array_equal(row_calls[0][1], [[7, 8]])
    np.testing.assert_array_equal(gather_calls[0], np.arange(16) + 900)
    assert len(native.handles[0].fills) == 1
    weight, scales, biases = native.handles[0].fills[0]
    assert weight.shape == (16, 20) and weight.dtype == np.uint32
    assert scales.shape == (16, 5) and scales.dtype == np.uint16
    assert biases.shape == (16, 5) and biases.dtype == np.uint16
    np.testing.assert_array_equal(cache[3], [[8, 9]])
    assert adapter.pending is False


def test_discard_does_not_advance_history_and_next_build_gets_fresh_leaf():
    adapter, native, _mx, _rows, _gathers = _adapter()
    adapter.set_active(True)
    cache = [None, None, None, np.array([[7, 8]], dtype=np.int64)]
    adapter.build(np.array([[9]], dtype=np.int64), cache, 3)

    adapter.discard()
    np.testing.assert_array_equal(cache[3], [[7, 8]])
    adapter.build(np.array([[10]], dtype=np.int64), cache, 3)

    assert len(native.handles) == 2
    assert native.handles[0].fills == []
    assert adapter.pending is True


def test_phase_boundary_rejects_disable_with_pending_leaf():
    adapter, _native, _mx, _rows, _gathers = _adapter()
    adapter.set_active(True)
    cache = [None, None, None, np.array([[7, 8]], dtype=np.int64)]
    adapter.build(np.array([[9]], dtype=np.int64), cache, 3)
    with pytest.raises(RuntimeError, match="pending"):
        adapter.set_active(False)


def test_flush_failure_poisoned_route_requires_model_reload():
    module = importlib.import_module(MODULE)
    adapter, _native, _mx, _rows, _gathers = _adapter()
    adapter.set_active(True)
    cache = [None, None, None, np.array([[7, 8]], dtype=np.int64)]
    adapter.build(np.array([[9]], dtype=np.int64), cache, 3)
    adapter._gather_planes = lambda _flat: (_ for _ in ()).throw(OSError("pread"))

    with pytest.raises(OSError, match="pread"):
        adapter.flush()
    adapter.discard()
    adapter.set_active(False)
    with pytest.raises(RuntimeError, match="reload"):
        adapter.set_active(True)
    assert isinstance(adapter.failure, OSError)
    assert module.AR_ROWS == 16
