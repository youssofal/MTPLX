"""CPU-only tests for the construction-bound cached native PLE adapter."""

from __future__ import annotations

import contextlib
from collections import OrderedDict
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mtplx import ple_cached_aux
from mtplx import ple_cached_row_handoff


ROOT = Path(__file__).resolve().parents[1]


class _Matrix:
    def __init__(self, rows: int, width: int, dtype, offset: int):
        self.shape = (rows, width)
        self.dtype = np.dtype(dtype)
        self.offset = int(offset)
        self.nbytes = rows * width * self.dtype.itemsize


class _Sidecar:
    bits = 4
    group_size = 32

    def __init__(self, fd, *, capacity: int = 8, row_count: int = 128):
        weight_bytes = row_count * 80
        metadata_bytes = row_count * 10
        self._fd = fd
        self._maps = {
            "weight": (_Matrix(row_count, 20, np.uint32, 128), "U32"),
            "scales": (
                _Matrix(row_count, 5, np.uint16, 128 + weight_bytes),
                "BF16",
            ),
            "biases": (
                _Matrix(
                    row_count,
                    5,
                    np.uint16,
                    128 + weight_bytes + metadata_bytes,
                ),
                "BF16",
            ),
        }
        self._hot = OrderedDict()
        self._hot_row_bytes = 100
        self._hot_cap_rows = capacity


def _payload(row: int):
    return (
        np.arange(20, dtype=np.uint32) + row * 1000,
        np.arange(5, dtype=np.uint16) + row * 100,
        np.arange(5, dtype=np.uint16) + row * 10,
    )


def _install_hot(sidecar: _Sidecar, *rows: int) -> None:
    for row in rows:
        sidecar._hot[row] = _payload(row)


def _ids(*values: int) -> tuple[int, ...]:
    assert len(values) == 4
    return tuple(values)


class _Planes:
    def __init__(self, name: str):
        self.name = name


class _EmbeddingResult:
    shape = (1, 4, 2560)

    def reshape(self, *shape):
        assert shape == (1, 4, 2560)
        return self


class _FakeMX:
    gpu = object()

    def __init__(self, trace):
        self.trace = trace
        self.new_stream_calls = []
        self.dequantize_calls = []
        self.fail_submit = False

    def new_stream(self, device):
        assert device is self.gpu
        stream = object()
        self.new_stream_calls.append(stream)
        self.trace.append(("new_stream", stream))
        return stream

    @contextlib.contextmanager
    def stream(self, stream):
        self.trace.append(("stream_enter", stream))
        try:
            yield
        finally:
            self.trace.append(("stream_exit", stream))

    def async_eval(self, *arrays):
        self.trace.append(("async_eval", tuple(getattr(a, "name", a) for a in arrays)))
        if self.fail_submit:
            raise RuntimeError("fake plane submit failed")

    def eval(self, *arrays):
        self.trace.append(("eval", tuple(getattr(a, "name", a) for a in arrays)))
        if self.fail_submit:
            raise RuntimeError("fake plane submit failed")

    def dequantize(self, weight, scales, biases, *, group_size, bits):
        self.dequantize_calls.append((weight, scales, biases, group_size, bits))
        self.trace.append(("dequantize", group_size, bits))
        return _EmbeddingResult()


class _Provider:
    def __init__(self):
        self.next_ticket = 1
        self.completions = []
        self.drain_calls = 0
        self.read_count = 0


class _FakeNative:
    def __init__(self, trace, rows_by_current):
        self.trace = trace
        self.rows_by_current = rows_by_current
        self.provider = _Provider()
        self.install_calls = []
        self.compute_calls = []
        self.make_calls = []
        self.drain_error = None

    def install_cached_sidecar_provider(self, fd, *args, io_workers):
        os.fstat(fd)
        assert io_workers == 8
        self.install_calls.append((fd, args, io_workers))
        self.trace.append(("install_provider", io_workers))
        return self.provider

    def compute_cached_row_ids(self, provider, previous, current):
        assert provider is self.provider
        current = tuple(int(value) for value in current)
        self.compute_calls.append((tuple(previous), current))
        result = np.array(self.rows_by_current[current[0]], dtype=np.uint32, copy=True)
        assert result.shape == (64,)
        result.flags.writeable = False
        return result

    def make_cached_sidecar_rows(self, provider, source, hits, misses):
        assert provider is self.provider
        assert source.shape == (64,)
        assert hits.flags.c_contiguous
        assert misses.dtype == np.uint32
        self.make_calls.append(
            {
                "source": np.array(source, copy=True),
                "hits": np.array(hits, copy=True),
                "misses": np.array(misses, copy=True),
            }
        )
        provider.read_count += int(misses.size)
        ticket = None
        if misses.size:
            ticket = provider.next_ticket
            provider.next_ticket += 1
        planes = (_Planes("weight"), _Planes("scales"), _Planes("biases"))
        self.trace.append(("make_rows", ticket, tuple(int(x) for x in misses)))
        return ticket, planes

    def drain_cached_completions(self, provider):
        assert provider is self.provider
        provider.drain_calls += 1
        self.trace.append(("drain",))
        if self.drain_error is not None:
            raise self.drain_error
        completions = list(provider.completions)
        provider.completions.clear()
        return completions

    def enqueue(self, ticket, misses):
        packed = np.vstack(
            [
                ple_cached_row_handoff.pack_row_payload(_payload(int(row)))
                for row in misses
            ]
        )
        packed.flags.writeable = False
        self.provider.completions.append((ticket, packed))


class _StockAux:
    def __init__(self, trace):
        self._pending_warm = ()
        self._prompt_tail = (100, 101)
        self._trace = trace
        self._submit_warm = self._submit
        self._install_owned_rows = self._install
        self.install_calls = []
        self.warm_rows = []

    def _submit(self, rows):
        values = tuple(int(value) for value in np.asarray(rows).reshape(-1))
        self.warm_rows.append(values)
        self._trace.append(("warm_submit", values))
        return ("warm-ticket", values)

    def _install(self, pending):
        self.install_calls.append(pending)
        self._trace.append(("install_warm", pending))

    def prefetch_primary(self, primary, completion_tokens, committed_count):
        del primary, completion_tokens, committed_count
        self._pending_warm = self._submit(np.arange(16, dtype=np.int64))


def _previous_tokens(prompt_tail, completion_tokens, committed_count):
    if committed_count >= 2:
        return (
            int(completion_tokens[committed_count - 2]),
            int(completion_tokens[committed_count - 1]),
        )
    if committed_count == 1:
        return int(prompt_tail[1]), int(completion_tokens[0])
    return prompt_tail


def _fixture(*, capacity: int = 8):
    trace = []
    mx = _FakeMX(trace)
    source = open(os.devnull, "rb")
    sidecar = _Sidecar(source.fileno(), capacity=capacity)
    _install_hot(sidecar, 1, 2)
    embedding = SimpleNamespace(
        context_len=2,
        ngram_size=3,
        heads_per_ngram=8,
        eos_id=99,
        ngram_embedding=SimpleNamespace(_sidecar=sidecar),
        _np_consts=lambda: (
            np.asarray((3, 5, 7), dtype=np.int64),
            np.arange(16, dtype=np.int64) + 32,
            np.arange(16, dtype=np.int64) * 4,
        ),
    )
    inner = SimpleNamespace(
        _ple_stage_idx=0,
        args=SimpleNamespace(ple_embed_dim=2560),
        layers=[SimpleNamespace(ple=SimpleNamespace(ple_embedding=embedding))],
    )
    rows_by_current = {
        10: np.full((64,), 1, dtype=np.uint32),
        20: np.asarray([1] * 32 + [3] * 32, dtype=np.uint32),
        21: np.asarray([2] * 32 + [4] * 32, dtype=np.uint32),
        30: np.asarray([1] * 32 + [5] * 32, dtype=np.uint32),
    }
    native = _FakeNative(trace, rows_by_current)
    stock_builds = []

    def stock_builder(*args, **kwargs):
        stock_builds.append((args, kwargs))
        return _StockAux(trace)

    runtime = SimpleNamespace(
        inner=inner,
        build_fixed_m4_compiled_verify_aux=stock_builder,
    )
    stock_module = SimpleNamespace(
        _inner=lambda value: value.inner,
        _fixed_m4_previous_tokens=_previous_tokens,
        _FixedM4SidecarAux=_StockAux,
    )
    return runtime, stock_module, mx, native, sidecar, trace, stock_builds, source


def _install(fixture, *, sync=False):
    runtime, stock_module, mx, native, sidecar, trace, builds, source = fixture
    installer = (
        ple_cached_aux.install_fixed_m4_sync_cached_aux_builder
        if sync
        else ple_cached_aux.install_fixed_m4_cached_aux_builder
    )
    installation = installer(
        runtime,
        native_module=native,
        mx_module=mx,
        stock_module=stock_module,
    )
    return runtime, installation, mx, native, sidecar, trace, builds, source


def test_module_import_is_cpu_only_and_keeps_native_api_deferred():
    had_mlx = "mlx.core" in sys.modules
    import mtplx.ple_cached_aux as module

    if not had_mlx:
        assert "mlx.core" not in sys.modules
    assert module.PENDING_LIMIT == 2
    for name in (
        "install_cached_sidecar_provider",
        "compute_cached_row_ids",
        "make_cached_sidecar_rows",
        "drain_cached_completions",
    ):
        assert name in module.NATIVE_CACHED_PROVIDER_API


@pytest.mark.parametrize("sync", (False, True))
def test_all_hit_window_has_no_native_completion_and_selects_plane_submitter(sync):
    fixture = _fixture()
    runtime, installation, mx, native, sidecar, trace, _builds, source = _install(
        fixture, sync=sync
    )
    try:
        aux = runtime.build_fixed_m4_compiled_verify_aux("cache")
        aux.prefetch_primary(7, (), 0)
        aux(None, _ids(10, 10, 10, 10), (), 0)

        assert native.provider.read_count == 0
        assert native.make_calls[-1]["misses"].tolist() == []
        assert installation.pending_count == 0
        assert list(sidecar._hot)[-1] == 1
        plane_events = [
            event[0] for event in trace if event[0] in {"eval", "async_eval"}
        ]
        assert plane_events[0] == ("eval" if sync else "async_eval")
        assert any(event[0] == "install_warm" for event in trace)
    finally:
        source.close()


def test_two_windows_out_of_order_completions_and_rebuilt_wrapper_drain_shared_state():
    fixture = _fixture()
    runtime, installation, mx, native, sidecar, trace, builds, source = _install(
        fixture
    )
    del mx
    try:
        aux_first = runtime.build_fixed_m4_compiled_verify_aux("first")
        aux_first(None, _ids(20, 20, 20, 20), (), 0)
        assert native.make_calls[-1]["misses"].tolist() == [3]

        # Building another wrapper must drain through the installation-owned
        # map; the pending ticket is not owned by aux_first.
        aux_second = runtime.build_fixed_m4_compiled_verify_aux("second")
        aux_second(None, _ids(21, 21, 21, 21), (), 0)
        assert installation.pending_count == 2
        native.enqueue(2, [4])
        native.enqueue(1, [3])

        aux_third = runtime.build_fixed_m4_compiled_verify_aux("third")
        assert installation.pending_count == 0
        assert 3 in sidecar._hot and 4 in sidecar._hot
        for actual, expected in zip(sidecar._hot[3], _payload(3)):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(sidecar._hot[4], _payload(4)):
            np.testing.assert_array_equal(actual, expected)
        assert len(builds) == 3
        assert aux_third._state is aux_first._state
        assert any(event[0] == "drain" for event in trace)
    finally:
        source.close()


def test_eviction_interleaving_restores_hit_and_drain_precedes_stock_warmup():
    fixture = _fixture(capacity=2)
    runtime, installation, mx, native, sidecar, trace, _builds, source = _install(
        fixture
    )
    del mx
    try:
        aux = runtime.build_fixed_m4_compiled_verify_aux("cache")
        aux(None, _ids(30, 30, 30, 30), (), 0)
        assert installation.pending_count == 1
        del sidecar._hot[1]
        native.enqueue(1, [5])

        aux.prefetch_primary(7, (), 0)

        assert list(sidecar._hot) == [1, 5]
        for actual, expected in zip(sidecar._hot[1], _payload(1)):
            np.testing.assert_array_equal(actual, expected)
        names = [event[0] for event in trace]
        assert names.index("drain") < names.index("warm_submit")
        assert installation.pending_count == 0
    finally:
        source.close()


def test_submit_failure_stops_before_cache_publish_and_future_calls_fail():
    fixture = _fixture()
    runtime, installation, mx, native, sidecar, _trace, _builds, source = _install(
        fixture
    )
    try:
        aux = runtime.build_fixed_m4_compiled_verify_aux("cache")
        mx.fail_submit = True
        with pytest.raises(RuntimeError, match="plane submit"):
            aux(None, _ids(20, 20, 20, 20), (), 0)
        assert 3 not in sidecar._hot
        assert installation.pending_count == 0
        with pytest.raises(RuntimeError, match="failed"):
            aux(None, _ids(10, 10, 10, 10), (), 0)
        assert 3 not in sidecar._hot
        assert native.provider.drain_calls == 2
    finally:
        source.close()


def test_drain_failure_stops_before_cache_publish():
    fixture = _fixture()
    runtime, installation, _mx, native, sidecar, _trace, _builds, source = _install(
        fixture
    )
    try:
        aux = runtime.build_fixed_m4_compiled_verify_aux("cache")
        aux(None, _ids(20, 20, 20, 20), (), 0)
        native.drain_error = RuntimeError("provider drain failed")
        with pytest.raises(RuntimeError, match="provider drain"):
            aux.prefetch_primary(7, (), 0)
        assert 3 not in sidecar._hot
        assert installation.pending_count == 0
    finally:
        source.close()


def test_pending_warm_install_failure_marks_shared_installation_failed():
    fixture = _fixture()
    runtime, installation, _mx, _native, _sidecar, _trace, _builds, source = _install(
        fixture
    )
    try:
        aux = runtime.build_fixed_m4_compiled_verify_aux("cache")
        aux.prefetch_primary(7, (), 0)

        def fail_install(_pending):
            raise RuntimeError("stock warm install failed")

        aux._stock._install_owned_rows = fail_install
        with pytest.raises(RuntimeError, match="stock warm install"):
            aux(None, _ids(10, 10, 10, 10), (), 0)
        assert installation._state.failure is not None
        assert installation.pending_count == 0
        with pytest.raises(RuntimeError, match="failed"):
            aux(None, _ids(10, 10, 10, 10), (), 0)
    finally:
        source.close()
