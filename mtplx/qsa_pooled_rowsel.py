"""Construction-bound fixed-M4 pooled-key installation for Qwen4 (rowsel).

The module is inert on import.  :func:`install_fixed_m4_pool` is called only
after the model weights and Fable stack are installed and before graph warmup.
It captures the actual QSA layer weights and the shared RoPE array, then
replaces only the fixed-capacity pooled-key preparation method with a
construction-bound "rowsel" method that binds the existing
``qsa_indexer_pool_keys_metal`` kernel metadata once per indexer and shares one
``inv_freq`` object across the twelve real QSA indexers.  The surrounding cache
state, fixed/non-fixed route, and bank ownership remain stock behavior, so the
output is identical to the stock path.

Stock MLX only: the pool helper is ``mtplx.kernels.qsa_indexer_prepare``'s
``_pool_keys_kernel`` (a ``mx.fast.metal_kernel``); there is no native
extension and no profiler build.
"""

from __future__ import annotations

import importlib
import sys
from types import MethodType
from typing import Any, Callable


QSA_LAYER_POSITIONS = tuple(range(3, 48, 4))
EXPECTED_LAYER_COUNT = 48
EXPECTED_RATIO = 4
EXPECTED_HEAD_DIM = 128
EXPECTED_ROTARY_DIM = 64
EXPECTED_EPS = 1e-6
EXPECTED_SCALE = 1.0
POOL_KERNEL_GRID = (32, 1, 1)
POOL_KERNEL_THREADGROUP = (32, 1, 1)
POOL_KERNEL_OUTPUT_SHAPES = ((1, 1, EXPECTED_HEAD_DIM),)


class FixedM4PoolInstallError(RuntimeError):
    """The construction-time fixed-M4 pool contract was not met."""


class PoolKernelBinding:
    """Immutable metadata and callable for one real QSA indexer."""

    __slots__ = (
        "kernel",
        "norm_weight",
        "inv_freq",
        "head_dim",
        "rotary_dim",
        "ratio",
        "eps",
        "scale",
        "dtype",
        "template",
        "grid",
        "threadgroup",
        "output_shapes",
        "output_dtypes",
        "mx_module",
    )

    def __init__(
        self,
        *,
        kernel: Callable[..., Any],
        norm_weight: Any,
        inv_freq: Any,
        head_dim: int,
        rotary_dim: int,
        ratio: int,
        eps: float,
        scale: float,
        dtype: Any,
    ) -> None:
        self.kernel = kernel
        self.norm_weight = norm_weight
        self.inv_freq = inv_freq
        self.head_dim = int(head_dim)
        self.rotary_dim = int(rotary_dim)
        self.ratio = int(ratio)
        self.eps = float(eps)
        self.scale = float(scale)
        self.dtype = dtype
        self.template = (("T", dtype),)
        self.grid = POOL_KERNEL_GRID
        self.threadgroup = POOL_KERNEL_THREADGROUP
        self.output_shapes = POOL_KERNEL_OUTPUT_SHAPES
        self.output_dtypes = (dtype,)

    def pool(self, raw_keys: Any, block_start: Any) -> Any:
        """Run the already-bound one-block helper with dynamic input leaves."""

        result = self.kernel(
            inputs=[raw_keys, self.norm_weight, self.inv_freq, block_start],
            template=self.template,
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=self.output_shapes,
            output_dtypes=self.output_dtypes,
        )
        return result[0]


def _text_model(runtime: Any) -> Any:
    return getattr(runtime.model, "language_model", runtime.model)


def _inner(runtime: Any) -> Any:
    text = _text_model(runtime)
    inner = getattr(text, "model", None)
    if inner is None:
        raise FixedM4PoolInstallError("runtime has no Qwen4 text model")
    return inner


def _resolve_mx(mx_module: Any | None) -> Any:
    if mx_module is not None:
        return mx_module
    return importlib.import_module("mlx.core")


def _resolve_kernel_factory(kernel_factory: Callable[..., Any] | None) -> Callable[..., Any]:
    if kernel_factory is not None:
        return kernel_factory
    helper = importlib.import_module("mtplx.kernels.qsa_indexer_prepare")
    factory = getattr(helper, "_pool_keys_kernel", None)
    if not callable(factory):
        raise FixedM4PoolInstallError("QSA pool helper lacks _pool_keys_kernel")
    return factory


def _loaded_graphbank() -> Any | None:
    """Return the already-loaded graphbank module without importing it."""

    return sys.modules.get("mtplx.graphbank")


def _graphbank_has_current_runtime_entries(graphbank: Any, runtime: Any) -> bool:
    """Detect fixed graphs already cached for this runtime, cold-only."""

    runtime_id = id(runtime)
    for cache_name in ("_SHARED_VERIFY_STEPS", "_SHARED_OVERLAP_SPLITS"):
        entries = getattr(graphbank, cache_name, None)
        if not isinstance(entries, dict):
            continue
        for key, entry in tuple(entries.items()):
            if not isinstance(key, tuple) or not key or key[0] != runtime_id:
                continue
            if not isinstance(entry, (tuple, list)) or not entry:
                continue
            runtime_ref = entry[-1]
            owner = runtime_ref() if callable(runtime_ref) else None
            if owner is runtime:
                return True
            # A stale entry can have the same first key after Python recycles
            # an object id.  It is not this runtime's trace and must not make a
            # cold installation fail.  Do not mutate the shared graph cache
            # while merely auditing construction state.
    return False


def _validate_normal_opdiet() -> dict[str, bool]:
    """Validate the measured rope+rowsel bank mode once at installation."""

    options = importlib.import_module("mtplx.runtime_options")
    rope = bool(options.fable_opdiet_enabled("rope"))
    bank = bool(options.fable_opdiet_enabled("bank"))
    if not rope or not bank:
        raise FixedM4PoolInstallError(
            "fixed-M4 pool requires normal Fable op-diet rope and bank items"
        )
    return {"rope": rope, "bank": bank}


def _validate_indexers(runtime: Any, mx: Any) -> tuple[Any, ...]:
    inner = _inner(runtime)
    layers = tuple(getattr(inner, "layers", ()))
    if len(layers) != EXPECTED_LAYER_COUNT:
        raise FixedM4PoolInstallError(
            f"fixed-M4 pool requires {EXPECTED_LAYER_COUNT} layers; got {len(layers)}"
        )

    indexers: list[Any] = []
    for position, layer in enumerate(layers):
        expected_qsa = position in QSA_LAYER_POSITIONS
        is_linear = bool(getattr(layer, "is_linear", False))
        if expected_qsa == is_linear:
            role = "QSA" if expected_qsa else "linear"
            raise FixedM4PoolInstallError(
                f"layer {position} is not the expected {role} position"
            )
        if not expected_qsa:
            continue
        attention = getattr(layer, "self_attn", None)
        indexer = getattr(attention, "indexer", None)
        if indexer is None:
            raise FixedM4PoolInstallError(f"QSA layer {position} has no indexer")
        indexers.append(indexer)

    if tuple(position for position in QSA_LAYER_POSITIONS) != tuple(
        position
        for position, layer in enumerate(layers)
        if not bool(getattr(layer, "is_linear", False))
    ):
        raise FixedM4PoolInstallError("QSA layer positions differ from the production 48-layer layout")

    if len(indexers) != len(QSA_LAYER_POSITIONS):
        raise FixedM4PoolInstallError(
            f"expected {len(QSA_LAYER_POSITIONS)} QSA indexers; got {len(indexers)}"
        )
    return tuple(indexers)


def _validate_indexer_contract(indexer: Any, mx: Any, position: int) -> tuple[Any, Any, int, int, float, float, Any]:
    ratio = int(getattr(indexer, "ratio", -1))
    head_dim = int(getattr(indexer, "head_dim", -1))
    eps = float(getattr(indexer, "rms_norm_eps", float("nan")))
    scale = float(getattr(indexer, "_rope_attention_scaling", float("nan")))
    inv_freq = getattr(indexer, "_inv_freq", None)
    norm_module = getattr(indexer, "k_layernorm", None)
    norm_weight = getattr(norm_module, "weight", None)
    if ratio != EXPECTED_RATIO or head_dim != EXPECTED_HEAD_DIM:
        raise FixedM4PoolInstallError(
            f"QSA layer {position} geometry mismatch: ratio={ratio} head_dim={head_dim}"
        )
    if eps != EXPECTED_EPS or scale != EXPECTED_SCALE:
        raise FixedM4PoolInstallError(
            f"QSA layer {position} RoPE metadata mismatch: eps={eps} scale={scale}"
        )
    if inv_freq is None or norm_weight is None:
        raise FixedM4PoolInstallError(f"QSA layer {position} lacks norm or inv_freq")
    if int(getattr(inv_freq, "ndim", -1)) != 1:
        raise FixedM4PoolInstallError(f"QSA layer {position} inv_freq must be rank 1")
    if int(getattr(inv_freq, "shape", (0,))[0]) * 2 != EXPECTED_ROTARY_DIM:
        raise FixedM4PoolInstallError(f"QSA layer {position} rotary dimension is not 64")
    if int(getattr(norm_weight, "ndim", -1)) != 1 or tuple(norm_weight.shape) != (EXPECTED_HEAD_DIM,):
        raise FixedM4PoolInstallError(f"QSA layer {position} norm weight must have shape (128,)")
    if getattr(inv_freq, "dtype", None) != mx.float32:
        raise FixedM4PoolInstallError(f"QSA layer {position} inv_freq must be float32")
    if getattr(norm_weight, "dtype", None) != mx.bfloat16:
        raise FixedM4PoolInstallError(f"QSA layer {position} norm weight must be BF16")
    norm_eps = float(getattr(norm_module, "eps", float("nan")))
    if norm_eps != eps:
        raise FixedM4PoolInstallError(
            f"QSA layer {position} norm epsilon mismatch: module={norm_eps} metadata={eps}"
        )
    return norm_weight, inv_freq, head_dim, EXPECTED_ROTARY_DIM, eps, scale, norm_weight.dtype


def _make_binding(
    indexer: Any,
    *,
    position: int,
    mx: Any,
    kernel_factory: Callable[..., Any],
) -> PoolKernelBinding:
    norm_weight, inv_freq, head_dim, rotary_dim, eps, scale, dtype = _validate_indexer_contract(
        indexer, mx, position
    )
    kernel = kernel_factory(
        head_dim,
        rotary_dim,
        EXPECTED_RATIO,
        eps,
        scale,
        dtype,
    )
    if not callable(kernel):
        raise FixedM4PoolInstallError("QSA pool helper did not return a callable")
    return PoolKernelBinding(
        kernel=kernel,
        norm_weight=norm_weight,
        inv_freq=inv_freq,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        ratio=EXPECTED_RATIO,
        eps=eps,
        scale=scale,
        dtype=dtype,
    )


def _fixed_m4_pool_method(self: Any, cache: Any, total: Any) -> Any:
    """Stock fixed-bank method with only pool arithmetic replaced.

    The Python body is entered at graph trace time.  In particular,
    ``pooled_capacity`` is read from the compiled bank shape here; it is not a
    per-token eligibility check and no capacity is copied into the binding.

    ``max_new`` is derived from the write width (``cache._last_write_rows``),
    not assumed to be one block, so the method stays correct when a
    width-parameterized verify writes 5 or 6 rows in one step
    (MTPLX_QWEN4_FIXED_VERIFY_ROWS): 4 rows fill one pooled block, 5-8 rows fill
    two.  Keep this derivation if that lane lands.
    """

    binding = self._mtplx_fixed_m4_pool_binding
    mx = binding.mx_module
    step_rows = int(getattr(cache, "_last_write_rows", 1))
    nb_old = cache.offset // self.ratio
    nb_total = total // self.ratio
    max_new = max(1, (step_rows + self.ratio - 1) // self.ratio)
    pooled = cache.pooled
    pooled_capacity = int(pooled.shape[1])
    for rel in range(max_new):
        block = nb_old + rel
        safe_block = mx.minimum(
            block, mx.array(pooled_capacity - 1, dtype=block.dtype)
        )
        start = safe_block * self.ratio
        fresh = mx.slice(
            cache.raw_keys,
            start,
            axes=(1,),
            slice_size=(1, self.ratio, self.head_dim),
        )
        block_start = safe_block.reshape(1).astype(mx.int32)
        candidate = binding.pool(fresh, block_start)
        old_row = mx.slice(
            pooled,
            safe_block,
            axes=(1,),
            slice_size=(1, 1, pooled.shape[2]),
        )
        merged = mx.where(
            nb_total > block,
            candidate.astype(pooled.dtype),
            old_row,
        )
        pooled = mx.slice_update(pooled, merged, safe_block, axes=(1,))
    cache.pooled = pooled
    return pooled


def install_fixed_m4_pool(
    runtime: Any,
    *,
    mx_module: Any | None = None,
    kernel_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Cold-install the pooled preparation on all twelve real QSA indexers."""

    if getattr(runtime, "_mtplx_fixed_m4_pool_installed", False):
        raise FixedM4PoolInstallError("fixed-M4 pool installation already completed")
    mx = _resolve_mx(mx_module)
    factory = _resolve_kernel_factory(kernel_factory)
    graphbank = _loaded_graphbank()
    if graphbank is not None and _graphbank_has_current_runtime_entries(graphbank, runtime):
        raise FixedM4PoolInstallError(
            "fixed-M4 pool installation must be cold; graphbank already has this runtime"
        )
    opdiet = _validate_normal_opdiet()
    indexers = _validate_indexers(runtime, mx)

    shared_inv = getattr(indexers[0], "_inv_freq", None)
    if shared_inv is None or any(indexer._inv_freq is not shared_inv for indexer in indexers):
        raise FixedM4PoolInstallError(
            "QSA indexers do not share the exact inv_freq object from TextArgs"
        )

    bindings = tuple(
        _make_binding(
            indexer,
            position=QSA_LAYER_POSITIONS[position],
            mx=mx,
            kernel_factory=factory,
        )
        for position, indexer in enumerate(indexers)
    )
    if any(binding.inv_freq is not shared_inv for binding in bindings):
        raise FixedM4PoolInstallError("pool bindings lost the shared inv_freq object")

    for indexer, binding in zip(indexers, bindings):
        binding.mx_module = mx
        original = indexer._extend_pooled_fixed
        indexer._mtplx_fixed_m4_pool_original = original
        indexer._mtplx_fixed_m4_pool_binding = binding
        indexer._extend_pooled_fixed = MethodType(_fixed_m4_pool_method, indexer)
        indexer._mtplx_fixed_m4_pool_installed = True

    report = {
        "installed": True,
        "qsa_layer_positions": list(QSA_LAYER_POSITIONS),
        "shared_inv_freq_identity": True,
        "shared_inv_freq_object_count": 1,
        "kernel_binding_count": len(bindings),
        "bank_mode": "rowsel",
        "opdiet": opdiet,
        "weights_copied": False,
        "graph_warmup_required_after_install": True,
    }
    runtime._mtplx_fixed_m4_pool_installed = True
    runtime._mtplx_fixed_m4_pool_install_report = dict(report)
    return report


install_qwen4_fixed_m4_pool = install_fixed_m4_pool


__all__ = [
    "EXPECTED_EPS",
    "EXPECTED_HEAD_DIM",
    "EXPECTED_RATIO",
    "EXPECTED_ROTARY_DIM",
    "FixedM4PoolInstallError",
    "PoolKernelBinding",
    "QSA_LAYER_POSITIONS",
    "install_fixed_m4_pool",
    "install_qwen4_fixed_m4_pool",
]
