"""CPU contract for the construction-bound fixed-M4 pool installer.

The real install is deliberately not imported on this test's MLX path.  Fake
arrays and a fake ``_pool_keys_kernel`` exercise the same scalar-offset,
fixed-bank update ABI without loading a model or compiling Metal.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALL_PATH = ROOT / "mtplx" / "qsa_pooled_rowsel.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeDtype:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other):
        return isinstance(other, FakeDtype) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class FakeArray:
    def __init__(self, data, dtype=None):
        self.data = np.asarray(data)
        self.dtype = dtype or self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def nbytes(self):
        return int(self.data.nbytes)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return FakeArray(self.data.reshape(*shape), self.dtype)

    def astype(self, dtype):
        return FakeArray(self.data, dtype)

    def __int__(self):
        return int(self.data.reshape(-1)[0])

    def _binary(self, other, op):
        rhs = other.data if isinstance(other, FakeArray) else other
        return FakeArray(op(self.data, rhs), self.dtype)

    def __add__(self, other):
        return self._binary(other, np.add)

    def __radd__(self, other):
        return self._binary(other, np.add)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    def __rmul__(self, other):
        return self._binary(other, np.multiply)

    def __floordiv__(self, other):
        return self._binary(other, np.floor_divide)

    def __gt__(self, other):
        rhs = other.data if isinstance(other, FakeArray) else other
        result = self.data > rhs
        return bool(result.reshape(-1)[0]) if result.size == 1 else result

    def __lt__(self, other):
        rhs = other.data if isinstance(other, FakeArray) else other
        result = self.data < rhs
        return bool(result.reshape(-1)[0]) if result.size == 1 else result


class FakeMX:
    int32 = FakeDtype("int32")
    bfloat16 = FakeDtype("bfloat16")
    float32 = FakeDtype("float32")

    @staticmethod
    def array(value, dtype=None):
        return FakeArray(value, dtype)

    @staticmethod
    def minimum(left, right):
        ldata = left.data if isinstance(left, FakeArray) else left
        rdata = right.data if isinstance(right, FakeArray) else right
        dtype = left.dtype if isinstance(left, FakeArray) else right.dtype
        return FakeArray(np.minimum(ldata, rdata), dtype)

    @staticmethod
    def slice(value, start, *, axes, slice_size):
        assert axes == (1,)
        assert tuple(slice_size)[0] == 1
        begin = int(start)
        stop = begin + int(slice_size[1])
        return FakeArray(value.data[:, begin:stop, :], value.dtype)

    @staticmethod
    def mean(value, *, axis):
        return FakeArray(value.data.mean(axis=axis), value.dtype)

    @staticmethod
    def where(condition, left, right):
        cond = condition.data if isinstance(condition, FakeArray) else condition
        ldata = left.data if isinstance(left, FakeArray) else left
        rdata = right.data if isinstance(right, FakeArray) else right
        dtype = left.dtype if isinstance(left, FakeArray) else right.dtype
        return FakeArray(np.where(cond, ldata, rdata), dtype)

    @staticmethod
    def slice_update(value, update, start, *, axes):
        assert axes == (1,)
        result = value.data.copy()
        begin = int(start)
        width = update.data.shape[1]
        result[:, begin : begin + width, :] = update.data
        return FakeArray(result, value.dtype)


class FakeKernelFactory:
    def __init__(self, mx):
        self.mx = mx
        self.bindings = []
        self.calls = []

    def __call__(self, head_dim, rotary_dim, ratio, eps, scale, dtype):
        self.bindings.append((head_dim, rotary_dim, ratio, eps, scale, dtype))

        def kernel(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            raw, norm_weight, inv_freq, block_start = inputs
            self.calls.append(
                {
                    "raw": raw,
                    "norm_weight": norm_weight,
                    "inv_freq": inv_freq,
                    "block_start": block_start,
                    "template": tuple(template),
                    "grid": tuple(grid),
                    "threadgroup": tuple(threadgroup),
                    "output_shapes": tuple(tuple(v) for v in output_shapes),
                    "output_dtypes": tuple(output_dtypes),
                }
            )
            # This is only an ABI oracle: the production helper is the source
            # of numeric truth.  Returning a distinct row makes bank writes
            # and restored-frontier behavior observable.
            result = raw.data[:, :4, :].mean(axis=1, keepdims=True)
            return [self.mx.array(result, dtype=raw.dtype)]

        return kernel


class WeakRuntime:
    pass


@pytest.fixture(autouse=True)
def _construction_environment(monkeypatch):
    graphbank = ModuleType("mtplx.graphbank")
    graphbank._SHARED_VERIFY_STEPS = {}
    graphbank._SHARED_OVERLAP_SPLITS = {}
    monkeypatch.setitem(sys.modules, "mtplx.graphbank", graphbank)

    import mtplx.runtime_options as runtime_options

    monkeypatch.setattr(
        runtime_options,
        "fable_opdiet_enabled",
        lambda item=None: item in {"rope", "bank"} if item is not None else True,
    )
    return graphbank


def _runtime(candidate, mx, *, shared_inv=True, distinct_dtype=False):
    inv_dtype = FakeDtype("float32") if distinct_dtype else mx.float32
    inv = mx.array(np.arange(32, dtype=np.float32), dtype=inv_dtype)
    layers = []
    indexers = []
    for position in range(48):
        if position in candidate.QSA_LAYER_POSITIONS:
            layer_inv = inv if shared_inv else mx.array(inv.data, dtype=mx.float32)
            norm_dtype = FakeDtype("bfloat16") if distinct_dtype else mx.bfloat16
            norm = mx.array(np.ones(128, dtype=np.float32), dtype=norm_dtype)
            indexer = SimpleNamespace(
                ratio=4,
                head_dim=128,
                rms_norm_eps=1e-6,
                _rope_attention_scaling=1.0,
                _inv_freq=layer_inv,
                k_layernorm=SimpleNamespace(weight=norm, eps=1e-6),
            )

            def fixed(self, cache, total):
                return "stock-fixed"

            def nonfixed(self, cache, total):
                return "stock-nonfixed"

            indexer._extend_pooled_fixed = fixed.__get__(indexer)
            indexer._extend_pooled = nonfixed.__get__(indexer)
            indexers.append(indexer)
            layers.append(SimpleNamespace(is_linear=False, self_attn=SimpleNamespace(indexer=indexer)))
        else:
            layers.append(SimpleNamespace(is_linear=True, self_attn=SimpleNamespace(indexer=None)))
    inner = SimpleNamespace(layers=layers, args=SimpleNamespace(num_hidden_layers=48))
    model = SimpleNamespace(language_model=SimpleNamespace(model=inner))
    runtime = WeakRuntime()
    runtime.model = model
    return runtime, indexers, inv


def _cache(mx, *, offset, capacity=8, rows=1, fill=0.0):
    raw = mx.array(np.arange(capacity * 4 * 128, dtype=np.float32).reshape(1, capacity * 4, 128), dtype=mx.bfloat16)
    pooled = mx.array(np.full((1, capacity, 128), fill, dtype=np.float32), dtype=mx.bfloat16)
    return SimpleNamespace(
        offset=mx.array([offset], dtype=mx.int32),
        pooled=pooled,
        raw_keys=raw,
        _last_write_rows=rows,
    )


def test_install_module_is_cpu_importable_and_does_not_import_model_or_mlx():
    before = set(sys.modules)
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_import_cpu")
    assert set(sys.modules) - before <= {"pr391_fixed_m4_pool_install_import_cpu"}
    assert candidate.QSA_LAYER_POSITIONS == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
    tree = ast.parse(INSTALL_PATH.read_text(encoding="utf-8"))
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    assert not any(name == "mlx" or name.startswith("mlx.") for name in top_imports)


def test_install_validates_48_layer_geometry_and_prebinds_actual_weights():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_geometry_cpu")
    mx = FakeMX()
    runtime, indexers, inv = _runtime(candidate, mx)
    factory = FakeKernelFactory(mx)

    report = candidate.install_fixed_m4_pool(
        runtime,
        mx_module=mx,
        kernel_factory=factory,
    )

    assert report["qsa_layer_positions"] == list(candidate.QSA_LAYER_POSITIONS)
    assert report["shared_inv_freq_identity"] is True
    assert report["opdiet"] == {"rope": True, "bank": True}
    assert len(factory.bindings) == 12
    assert all(binding[0:3] == (128, 64, 4) for binding in factory.bindings)
    assert all(binding[3:5] == (1e-6, 1.0) for binding in factory.bindings)
    for indexer in indexers:
        binding = indexer._mtplx_fixed_m4_pool_binding
        assert binding.inv_freq is inv
        assert binding.norm_weight is indexer.k_layernorm.weight
        assert binding.template == (("T", mx.bfloat16),)
        assert binding.grid == (32, 1, 1)
        assert binding.output_shapes == ((1, 1, 128),)


def test_install_rejects_unshared_inv_freq_and_wrong_geometry():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_validation_cpu")
    mx = FakeMX()
    runtime, _indexers, _inv = _runtime(candidate, mx, shared_inv=False)
    with pytest.raises(candidate.FixedM4PoolInstallError, match="inv_freq"):
        candidate.install_fixed_m4_pool(runtime, mx_module=mx, kernel_factory=FakeKernelFactory(mx))

    runtime, _indexers, _inv = _runtime(candidate, mx)
    runtime.model.language_model.model.layers = runtime.model.language_model.model.layers[:-1]
    with pytest.raises(candidate.FixedM4PoolInstallError, match="48"):
        candidate.install_fixed_m4_pool(runtime, mx_module=mx, kernel_factory=FakeKernelFactory(mx))


def test_install_rejects_norm_module_epsilon_mismatch():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_eps_cpu")
    mx = FakeMX()
    runtime, indexers, _inv = _runtime(candidate, mx)
    indexers[0].k_layernorm.eps = 2e-6
    with pytest.raises(candidate.FixedM4PoolInstallError, match="epsilon"):
        candidate.install_fixed_m4_pool(
            runtime, mx_module=mx, kernel_factory=FakeKernelFactory(mx)
        )


def test_install_accepts_equal_but_distinct_dtype_wrappers():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_equal_dtype_cpu")
    mx = FakeMX()
    runtime, _indexers, _inv = _runtime(candidate, mx, distinct_dtype=True)
    report = candidate.install_fixed_m4_pool(
        runtime, mx_module=mx, kernel_factory=FakeKernelFactory(mx)
    )
    assert report["installed"] is True


def test_install_rejects_non_normal_opdiet_at_construction(monkeypatch):
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_opdiet_cpu")
    mx = FakeMX()
    runtime, _indexers, _inv = _runtime(candidate, mx)
    import mtplx.runtime_options as runtime_options

    monkeypatch.setattr(runtime_options, "fable_opdiet_enabled", lambda item=None: False)
    with pytest.raises(candidate.FixedM4PoolInstallError, match="op-diet"):
        candidate.install_fixed_m4_pool(
            runtime, mx_module=mx, kernel_factory=FakeKernelFactory(mx)
        )


def test_install_fails_cold_when_graphbank_has_this_runtime():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_trace_guard_cpu")
    mx = FakeMX()
    runtime, _indexers, _inv = _runtime(candidate, mx)
    graphbank = sys.modules["mtplx.graphbank"]
    graphbank._SHARED_VERIFY_STEPS[(id(runtime), "fixed")] = (
        object(),
        {},
        weakref.ref(runtime),
    )
    with pytest.raises(candidate.FixedM4PoolInstallError, match="cold"):
        candidate.install_fixed_m4_pool(runtime, mx_module=mx, kernel_factory=FakeKernelFactory(mx))


def test_install_ignores_dead_same_id_graphbank_entry():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_stale_graph_cpu")
    mx = FakeMX()
    runtime, _indexers, _inv = _runtime(candidate, mx)
    stale = WeakRuntime()
    stale_ref = weakref.ref(stale)
    del stale
    graphbank = sys.modules["mtplx.graphbank"]
    key = (id(runtime), "stale")
    entry = (object(), {}, stale_ref)
    graphbank._SHARED_OVERLAP_SPLITS[key] = entry
    report = candidate.install_fixed_m4_pool(
        runtime, mx_module=mx, kernel_factory=FakeKernelFactory(mx)
    )
    assert report["installed"] is True
    assert key in graphbank._SHARED_OVERLAP_SPLITS


def test_fixed_method_preserves_nonfixed_method_and_uses_bound_helper():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_method_cpu")
    mx = FakeMX()
    runtime, indexers, _inv = _runtime(candidate, mx)
    factory = FakeKernelFactory(mx)
    candidate.install_fixed_m4_pool(runtime, mx_module=mx, kernel_factory=factory)
    indexer = indexers[0]
    nonfixed = indexer._extend_pooled
    cache = _cache(mx, offset=0, capacity=8, rows=1, fill=-3.0)
    result = indexer._extend_pooled_fixed(cache, mx.array([4], dtype=mx.int32))
    assert result is cache.pooled
    assert nonfixed(cache, mx.array([4], dtype=mx.int32)) == "stock-nonfixed"
    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert int(call["block_start"]) == 0
    assert call["raw"].shape == (1, 4, 128)
    assert call["raw"].data.strides == (16384, 512, 4)
    assert call["norm_weight"] is indexer.k_layernorm.weight
    assert call["inv_freq"] is indexer._inv_freq
    assert call["grid"] == (32, 1, 1)
    assert call["output_shapes"] == ((1, 1, 128),)


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_fixed_method_handles_offset_residues_and_frontier_without_oob(offset):
    candidate = _load(INSTALL_PATH, f"pr391_fixed_m4_pool_install_residue_{offset}")
    mx = FakeMX()
    runtime, indexers, _inv = _runtime(candidate, mx)
    factory = FakeKernelFactory(mx)
    candidate.install_fixed_m4_pool(runtime, mx_module=mx, kernel_factory=factory)
    cache = _cache(mx, offset=offset, capacity=2, rows=1, fill=-7.0)
    before = cache.pooled.data.copy()
    indexers[0]._extend_pooled_fixed(cache, mx.array([offset], dtype=mx.int32))
    assert cache.pooled.shape == (1, 2, 128)
    assert np.array_equal(cache.pooled.data, before)
    assert len(factory.calls) == 1


def test_fixed_method_updates_last_valid_bank_and_restored_frontier():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_boundary_cpu")
    mx = FakeMX()
    runtime, indexers, _inv = _runtime(candidate, mx)
    factory = FakeKernelFactory(mx)
    candidate.install_fixed_m4_pool(runtime, mx_module=mx, kernel_factory=factory)
    indexer = indexers[0]

    boundary = _cache(mx, offset=4, capacity=2, rows=1, fill=-5.0)
    indexer._extend_pooled_fixed(boundary, mx.array([8], dtype=mx.int32))
    assert np.all(boundary.pooled.data[:, 1, :] != -5.0)

    restored = _cache(mx, offset=8, capacity=2, rows=1, fill=-9.0)
    before = restored.pooled.data.copy()
    indexer._extend_pooled_fixed(restored, mx.array([8], dtype=mx.int32))
    assert np.array_equal(restored.pooled.data, before)


@pytest.mark.parametrize(("rows", "expected_calls"), [(1, 1), (4, 1), (5, 2), (8, 2)])
def test_max_new_uses_original_rowcount_formula(rows, expected_calls):
    candidate = _load(INSTALL_PATH, f"pr391_fixed_m4_pool_install_rows_{rows}")
    mx = FakeMX()
    runtime, indexers, _inv = _runtime(candidate, mx)
    factory = FakeKernelFactory(mx)
    candidate.install_fixed_m4_pool(runtime, mx_module=mx, kernel_factory=factory)
    cache = _cache(mx, offset=0, capacity=8, rows=rows, fill=-2.0)
    indexers[0]._extend_pooled_fixed(cache, mx.array([rows], dtype=mx.int32))
    assert len(factory.calls) == expected_calls


def test_rowsel_is_the_only_construction_selected_bank_update():
    candidate = _load(INSTALL_PATH, "pr391_fixed_m4_pool_install_rowsel_cpu")
    source = INSTALL_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in source
    assert not any(isinstance(node, ast.Try) for node in ast.walk(ast.parse(source)))
    assert "rowsel" in source
    assert "fullbankcopy" not in source
    mx = FakeMX()
    runtime, indexers, _inv = _runtime(candidate, mx)
    factory = FakeKernelFactory(mx)
    candidate.install_fixed_m4_pool(
        runtime,
        mx_module=mx,
        kernel_factory=factory,
    )
    cache = _cache(mx, offset=0, capacity=8, rows=1, fill=-2.0)
    indexers[0]._extend_pooled_fixed(cache, mx.array([4], dtype=mx.int32))
    assert len(factory.calls) == 1
