"""Direct Steel sparse-GQA attention consumer for Qwen4Exp QSA prefill.

This is the third consumer of the indexer's per-query ``block_ids`` /
``block_valid`` contract, sitting between the Metal 4 MPP flash kernel
(:mod:`mtplx.kernels.qsa_prefill_flash`, M4/M5 only) and the portable gather
tier.  It dispatches the vendored oMLX Steel MMA kernel
(``native_extensions/qsa_kernels``, Apache-2.0, oMLX PR #3244) which streams
the selected four-token blocks straight out of the KV cache: no gathered
``[S, selected, Hkv, D]`` tensor and no dense ``[S, T]`` mask.  It is the fast
lane for M3-class GPUs, which have no G17 tensor units and therefore can never
take the MPP path.

Three properties make this module safe to have installed:

* **Optional.** A missing or unloadable native extension leaves the import
  green; :func:`qsa_prefill_direct_supported` simply returns ``False`` and the
  model routes to gather/dense.
* **Fail-closed, not fail-quiet.** Static unsupported geometry returns
  ``False`` before anything is added to the graph.  A call that passed the
  support check and dispatched the primitive is allowed to fail loudly — there
  is no in-kernel dense retry, because a benchmark arm that silently fell back
  is a wrong benchmark.
* **Proven once, then trusted — and retired on failure.** ``abi_probe`` at
  import proves the nanobind type casters match the mlx wheel, and
  ``BUILT_AGAINST_MLX`` is compared against the imported mlx because this
  primitive links MLX's private C++ ABI.  Symbol presence is not pipeline
  readiness, so the *first* dispatch is evaluated eagerly: a missing/stale
  metallib surfaces as a pipeline error at a known point rather than at an
  arbitrary later ``mx.eval``.  The proof is PER QUERY DTYPE, because the
  Metal kernel name embeds it (``qwen4_qsa_sparse_gqa_<type>_bk64_dc64_...``)
  and a metallib can carry a good bfloat16 specialization beside a missing
  float16 one; ``qsa_prefill_direct_ready`` proves both before it says True.
  Any such failure — a receipt mismatch, a failed preflight, a failed first
  evaluation of either dtype — disables the whole lane for the whole process,
  so ``qsa_prefill_direct_ready`` goes False, the M3 producer auto-gate
  disarms, and traffic routes to gather instead of re-hitting the same wall
  on every request.  FAILED is terminal and lock-guarded: a proof that
  succeeds after another has failed cannot reopen the lane.

The Topk ABI seam
-----------------
The native kernel takes only ``[1, 1, S, 512]`` uint32 block ids and no
validity tensor: validity is **positional**.  It reads exactly the first
``min(512, (q_abs + 1) // 4)`` slots of each row.  MTPLX's public block
contract tolerates arbitrary validity holes, so this module owns the
narrowing: the producers must emit a chronological VALID PREFIX (ascending
ids in ``[0, valid_count)``, padding after).  Both production selectors do —
the eager selector sorts then zeroes the invalid tail, and the Metal selector
orders selected blocks by ascending id followed by padded lanes — and
``tests/test_qsa_selector_prefix_contract.py`` is the license for the kernel
to ignore ``block_valid``.  Set ``MTPLX_QSA_PREFILL_DIRECT_VALIDATE=1`` to
re-prove the invariant per call at the cost of a host synchronization.

Environment
-----------
``MTPLX_QSA_PREFILL_DIRECT``
    ``0`` kills only this consumer (gather/dense still serve).  Unset means
    on whenever the native module is loaded and probed.
``MTPLX_QSA_PREFILL_DIRECT_VALIDATE``
    ``1`` runs the full per-call prefix-contract check (synchronizes).
"""

from __future__ import annotations

import logging
import math
import operator
import os
import sys
import threading
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

_BATCH = 1
_Q_HEADS = 24
_KV_HEADS = 2
_GQA = 12
_HEAD_DIM = 256
_MAX_CONTEXT = 1_048_576
_COMPRESS_RATIO = 4
_TOP_K_BLOCKS = 512
_EXPECTED_SCALE = 0.0625  # 1 / sqrt(256), exactly representable
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16)

# The one packaged Steel specialization. oMLX's production glue uses (64, 64);
# the fast.py defaults (128, 32) are not what was measured on M3 and are not
# instantiated in MTPLX's metallib.
_KEY_TILE = 64
_DIMENSION_TILE = 64

__all__ = [
    "qsa_prefill_direct",
    "qsa_prefill_direct_build_info",
    "qsa_prefill_direct_module_ready",
    "qsa_prefill_direct_preflight",
    "qsa_prefill_direct_ready",
    "qsa_prefill_direct_supported",
    "qsa_prefill_direct_topk_buffer",
    "qsa_prefill_direct_unsupported_reason",
]


def _detach(exc: BaseException) -> BaseException:
    """Keep the diagnostic message without retaining import caller frames."""

    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


def _extension_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "native_extensions" / "qsa_kernels"


def _import_extension():
    """Import the optional native module; never raise."""

    native_path = _extension_dir()
    if native_path.is_dir() and str(native_path) not in sys.path:
        # Source-tree builds live beside the sources (the house pattern used
        # by mtplx/kernels/native_gdn_tail.py); an installed wheel is found
        # on the normal path and this insert is a no-op.
        sys.path.insert(0, str(native_path))
    try:
        import mtplx_qsa_kernels  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on a local build
        built = native_path / "mtplx_qsa_kernels"
        if built.is_dir() and any(built.glob("_ext*.so")):
            # A built extension that fails to load is a real defect (usually
            # an unresolved @rpath to the mlx wheel's libmlx.dylib). Leave a
            # trace so the slow path is not silently taken forever.
            logger.warning(
                "%s: native QSA kernel is present but failed to load; the "
                "direct prefill lane is disabled: %s",
                __name__,
                exc,
            )
        return None, _detach(exc)
    return mtplx_qsa_kernels, None


_EXT, _IMPORT_ERROR = _import_extension()


def _verify_abi(ext, import_error):
    """Disable the lane when the extension rejects mlx arrays.

    An extension built with a nanobind whose ABI tag differs from the mlx
    wheel's imports cleanly and lists every symbol, but its type casters live
    in an isolated ``NB_DOMAIN``, so every call raises ``TypeError:
    incompatible function arguments``.  Probe once at import and degrade with
    a single warning instead of failing per layer per token.
    """

    if ext is None:
        return ext, import_error
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        logger.warning(
            "%s: native QSA kernel has no abi_probe symbol; rebuild it from "
            "native_extensions/qsa_kernels. Lane disabled.",
            __name__,
        )
        return None, import_error
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        logger.warning(
            "%s: native QSA kernel disabled — it was built with a nanobind "
            "ABI that does not match this mlx wheel; rebuild it against the "
            "installed mlx (see native_extensions/qsa_kernels/pyproject.toml).",
            __name__,
        )
        return None, _detach(exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("%s: native QSA kernel probe failed: %s", __name__, exc)
        return None, _detach(exc)
    return ext, import_error


_EXT, _IMPORT_ERROR = _verify_abi(_EXT, _IMPORT_ERROR)

# Process-wide lifecycle for the packaged Metal specialization. MLX
# primitives are lazy, so a missing or stale metallib fails at mx.eval, not
# at dispatch: the lane is UNPROVEN until one real call has been evaluated,
# and any failed proof moves it to FAILED for the rest of the process. FAILED
# is terminal and is what makes ``ready()``/``supported()`` answer False
# instead of arming a producer for a consumer that cannot run.
_PIPELINE_UNPROVEN = "unproven"
_PIPELINE_PROVEN = "proven"
_PIPELINE_FAILED = "failed"
_PIPELINE_STATE = _PIPELINE_UNPROVEN

# The native Metal kernel name embeds the QUERY DTYPE
# (``qwen4_qsa_sparse_gqa_<type>_bk64_dc64_...``), so proving the bfloat16
# specialization says nothing about whether the float16 one was packaged into
# the metallib.  Readiness therefore tracks the proved dtypes and only turns
# ``_PIPELINE_STATE`` to PROVEN once every entry of ``_SUPPORTED_DTYPES`` has
# run.  A failure in any one of them retires the WHOLE lane (fail-closed: a
# half-usable lane is not a lane the router can reason about).
_PIPELINE_PROVEN_DTYPES: frozenset[str] = frozenset()

# The transitions above are check-then-set across an ``mx.eval``, so two
# threads racing their first prefill could both observe UNPROVEN and the
# winner could write PROVEN over the loser's FAILED.  The lock serializes the
# proof; ``_record_pipeline_success`` refuses to leave FAILED regardless, so
# FAILED is terminal even if a caller reaches it another way.  Re-entrant
# because ``_prove_pipeline`` proves through ``qsa_prefill_direct``, which
# calls ``_prove_first_dispatch``.
_PIPELINE_LOCK = threading.RLock()

_RECEIPT_WARNED = False


def _dtype_key(dtype: mx.Dtype) -> str:
    return str(dtype)


def _dtype_proven(dtype: mx.Dtype) -> bool:
    return _dtype_key(dtype) in _PIPELINE_PROVEN_DTYPES


def _record_pipeline_failure() -> None:
    """Retire the lane for the process.  FAILED is terminal."""

    global _PIPELINE_STATE, _PIPELINE_PROVEN_DTYPES

    _PIPELINE_STATE = _PIPELINE_FAILED
    # Drop the dtype proofs too: one dead specialization retires the lane, and
    # a stale PROVEN dtype would otherwise let a later dispatch skip its own
    # proof.
    _PIPELINE_PROVEN_DTYPES = frozenset()


def _record_pipeline_success(dtype: mx.Dtype) -> None:
    """Bank one dtype proof; never reopen a lane that already FAILED."""

    global _PIPELINE_STATE, _PIPELINE_PROVEN_DTYPES

    if _PIPELINE_STATE == _PIPELINE_FAILED:
        return
    _PIPELINE_PROVEN_DTYPES = _PIPELINE_PROVEN_DTYPES | {_dtype_key(dtype)}
    if all(_dtype_key(known) in _PIPELINE_PROVEN_DTYPES for known in _SUPPORTED_DTYPES):
        _PIPELINE_STATE = _PIPELINE_PROVEN


def qsa_prefill_direct_build_info() -> dict[str, str]:
    """Build receipts for the loaded extension (empty when not loaded).

    The primitive links MLX's private C++ ABI, so ``mlx`` recording which
    wheel the .so was compiled against is the difference between a clean
    "rebuild me" and a mystery crash after ``pip install -U mlx``.
    """

    if _EXT is None:
        return {}
    return {
        "built_against_mlx": str(getattr(_EXT, "BUILT_AGAINST_MLX", "unknown")),
        "built_against_nanobind": str(
            getattr(_EXT, "BUILT_AGAINST_NANOBIND", "unknown")
        ),
        "metal_library": str(getattr(_EXT, "METAL_LIBRARY", "unknown")),
        "imported_mlx": str(getattr(mx, "__version__", "unknown")),
    }


def qsa_prefill_direct_import_error() -> BaseException | None:
    """Why the native module is unavailable, if it is."""

    return _IMPORT_ERROR


def qsa_prefill_direct_module_ready() -> bool:
    """Whether the native symbol is importable and ABI-compatible."""

    return _EXT is not None and hasattr(_EXT, "qwen4_qsa_sparse_gqa_attention")


def _env_flag(name: str, *, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def qsa_prefill_direct_enabled() -> bool:
    """Kill switch only.  Default on; the module gate is separate."""

    return _env_flag("MTPLX_QSA_PREFILL_DIRECT", default=True)


def _validate_enabled() -> bool:
    return _env_flag("MTPLX_QSA_PREFILL_DIRECT_VALIDATE", default=False)


def _build_receipt_mismatch() -> str | None:
    """Compare the baked mlx receipt against the mlx actually imported.

    The primitive links MLX's private C++ ABI and its Steel headers, so an
    extension built against a different wheel is a crash waiting for the
    first dispatch — a `.so` that imports, probes and lists every symbol, and
    then mis-reads a struct. CMake bakes ``BUILT_AGAINST_MLX`` from the build
    interpreter for exactly this comparison.
    """

    if _EXT is None:
        return None
    built = str(getattr(_EXT, "BUILT_AGAINST_MLX", "") or "").strip()
    runtime = str(getattr(mx, "__version__", "") or "").strip()
    if not built or built == "unknown":
        return "the native QSA extension carries no BUILT_AGAINST_MLX receipt"
    if not runtime:
        return None
    if built != runtime:
        return (
            f"the native QSA extension was built against mlx {built} but "
            f"mlx {runtime} is imported"
        )
    return None


def _lane_unavailable_reason() -> str | None:
    """Why the lane cannot serve this process, ignoring per-call geometry.

    Deliberately does NOT consult :func:`qsa_prefill_direct_ready`: that
    function proves the pipeline through the preflight, which comes back
    through here, and the cycle would be infinite.
    """

    global _RECEIPT_WARNED

    if _PIPELINE_STATE == _PIPELINE_FAILED:
        return (
            "the direct lane was disabled after a failed Metal pipeline "
            "proof in this process"
        )
    if not qsa_prefill_direct_module_ready():
        return "the native QSA sparse-GQA extension is not loaded"
    if not qsa_prefill_direct_enabled():
        return "MTPLX_QSA_PREFILL_DIRECT is off"
    mismatch = _build_receipt_mismatch()
    if mismatch is not None:
        if not _RECEIPT_WARNED:
            _RECEIPT_WARNED = True
            logger.warning(
                "%s: %s; the direct prefill lane is disabled — rebuild the "
                "extension in the venv that serves (see "
                "native_extensions/qsa_kernels/README.md).",
                __name__,
                mismatch,
            )
        return mismatch
    return None


def _lane_eligible() -> bool:
    """Symbol, kill switch, build receipt and process state — no proof."""

    return _lane_unavailable_reason() is None


def qsa_prefill_direct_ready() -> bool:
    """Whether this consumer can serve at all on this process.

    The producer auto-gate in ``qwen4_exp`` calls this: on M3 there is no NAX
    flash consumer, so the large-prefill lane may only arm itself when THIS
    kernel is loaded, probed, receipt-matched, not killed by env, and known
    to have a runnable Metal pipeline.  A present-but-unproven metallib is
    not readiness: symbol presence never proved that ``get_library`` or
    ``get_kernel`` can resolve the packaged specializations, so a real loaded
    extension is proved once, here, before the answer is True — and because
    the native kernel name embeds the query dtype, EVERY packaged dtype
    specialization is proved, not just the bfloat16 default.
    """

    if not _lane_eligible():
        return False
    if _PIPELINE_STATE == _PIPELINE_PROVEN:
        return True
    if _EXT is not None and _on_metal_device():
        return _prove_all_pipelines()
    # No real extension is loaded (unit tests stub the module gate) or MLX is
    # not on a Metal GPU, where nothing can be proved and the device gate in
    # qsa_prefill_lane_auto_supported refuses anyway.
    return True


def _on_metal_device() -> bool:
    """Metal availability is insufficient when MLX currently targets CPU."""

    try:
        return mx.metal.is_available() and mx.default_device() == mx.gpu
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _last_dim_contiguous(array: mx.array) -> bool:
    """Best-effort mirror of the C++ ``strides(-1) == 1`` check.

    The MLX Python array exposes no stride API, so this cannot be decided
    from Python for the views this lane consumes (a transposed Q and a
    length-sliced KV cache — both unit-stride in the feature axis by
    construction, and neither can be made contiguous here without copying the
    whole cache).  Rank/shape are checked above; the authoritative check is
    the C++ ``unsupported()``, which raises rather than mis-indexing.  If a
    future MLX exposes strides, use them.
    """

    strides = getattr(array, "strides", None)
    if strides is None:
        return True
    try:
        return int(strides[-1]) == 1
    except (IndexError, TypeError, ValueError):  # pragma: no cover - defensive
        return True


def qsa_prefill_direct_unsupported_reason(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    compress_ratio: int = _COMPRESS_RATIO,
    block_topk: int = _TOP_K_BLOCKS,
    key_tile: int = _KEY_TILE,
    dimension_tile: int = _DIMENSION_TILE,
) -> str | None:
    """Return why this call cannot use the direct kernel, or ``None``.

    Every clause here has a counterpart in the vendored C++ ``unsupported()``
    (``native_extensions/qsa_kernels/qwen4_qsa_sparse_gqa.cpp``) or in the
    MTPLX block contract the C++ cannot see (model ratio/top-k, the suffix
    relation, the dense/sparse boundary).  Keeping them in lockstep is what
    makes "supported" mean "will dispatch".
    """

    unavailable = _lane_unavailable_reason()
    if unavailable is not None:
        return unavailable
    if not _on_metal_device():
        return "the active MLX device is not an available Metal GPU"

    arrays = (queries, keys, values, block_ids, block_valid)
    if any(not isinstance(array, mx.array) for array in arrays):
        return "all tensor inputs must be MLX arrays"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "Q, K, and V must be rank four"
    if block_ids.ndim != 2 or block_valid.ndim != 2:
        return "block ids and validity must be rank two"

    batch, query_heads, rows, head_dim = (int(x) for x in queries.shape)
    if (batch, query_heads, head_dim) != (_BATCH, _Q_HEADS, _HEAD_DIM):
        return "Q must have production shape [1, 24, S, 256]"
    if rows <= 0:
        return "the prefill kernel requires at least one query row"

    key_batch, kv_heads, key_len, key_dim = (int(x) for x in keys.shape)
    if (key_batch, kv_heads, key_dim) != (_BATCH, _KV_HEADS, _HEAD_DIM):
        return "K must have production shape [1, 2, kL, 256]"
    if tuple(int(x) for x in values.shape) != tuple(int(x) for x in keys.shape):
        return "V must have the same shape as K"

    if queries.dtype not in _SUPPORTED_DTYPES:
        return "Q must be float16 or bfloat16"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "Q, K, and V dtypes must match"
    if block_ids.dtype != mx.int32 or block_valid.dtype != mx.bool_:
        return "block ids must be int32 and validity must be bool"
    if tuple(int(x) for x in block_ids.shape) != (rows, _TOP_K_BLOCKS):
        return "block ids must have shape [S, 512]"
    if tuple(int(x) for x in block_valid.shape) != (rows, _TOP_K_BLOCKS):
        return "block validity must have shape [S, 512]"
    if not (
        _last_dim_contiguous(queries)
        and _last_dim_contiguous(keys)
        and _last_dim_contiguous(values)
    ):
        return "Q, K, and V must be contiguous in their last axis"

    # The model's own QSA configuration. The C++ checks tensor shapes; it
    # cannot see that the indexer was configured with a different block size
    # or budget, and the four-token expansion is compiled into the kernel.
    if int(compress_ratio) != _COMPRESS_RATIO:
        return "the kernel expands exactly four-token blocks"
    if int(block_topk) != _TOP_K_BLOCKS:
        return "the kernel consumes exactly 512 selected blocks"
    if int(key_tile) != _KEY_TILE or int(dimension_tile) != _DIMENSION_TILE:
        return "only the packaged (key_tile, dimension_tile) == (64, 64) ships"

    # Host-static scalars only: comparing a traced scalar here would
    # synchronize the graph on every layer of every chunk.
    if isinstance(pos_start, mx.array) or isinstance(total_tokens, mx.array):
        return "pos_start and total_tokens must be host integers"
    if isinstance(scale, mx.array):
        return "scale must be a host float"
    if isinstance(pos_start, bool) or isinstance(total_tokens, bool):
        return "pos_start and total_tokens cannot be bool"
    try:
        pos_start_i = operator.index(pos_start)
        total_tokens_i = operator.index(total_tokens)
    except TypeError:
        return "pos_start and total_tokens must be exact host integers"
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        return "scale must be a numeric host scalar"
    scale_f = float(scale)

    if pos_start_i < 0 or total_tokens_i <= 0:
        return "positions must describe a non-empty non-negative suffix"
    if pos_start_i + rows != total_tokens_i:
        return "Q must be the suffix ending exactly at total_tokens"
    # The native ABI has no logical-K parameter: params.kL is k.shape(2).
    # Passing the capacity BACKING would make kL mean capacity and hide a
    # logical-frontier mistake, so this lane takes the logical views returned
    # by update_and_fetch and insists they are exactly the live cache.
    if key_len != total_tokens_i:
        return "K/V must be the logical cache prefix with kL == total_tokens"
    if total_tokens_i > _MAX_CONTEXT:
        return "the logical token count exceeds the production context limit"
    if total_tokens_i // _COMPRESS_RATIO <= _TOP_K_BLOCKS:
        return "the context has not crossed the dense/sparse boundary"
    if not math.isfinite(scale_f) or scale_f != _EXPECTED_SCALE:
        return "scale must equal the production 1/sqrt(256) value"
    return None


def qsa_prefill_direct_supported(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    compress_ratio: int = _COMPRESS_RATIO,
    block_topk: int = _TOP_K_BLOCKS,
) -> bool:
    """Whether the exact production-only direct-kernel contract is met."""

    return (
        qsa_prefill_direct_unsupported_reason(
            queries,
            keys,
            values,
            block_ids,
            block_valid,
            pos_start=pos_start,
            total_tokens=total_tokens,
            scale=scale,
            compress_ratio=compress_ratio,
            block_topk=block_topk,
        )
        is None
    )


def _prefix_contract_violation(
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    compress_ratio: int,
) -> str | None:
    """Prove the valid-prefix invariant the native ABI cannot express.

    Synchronizes.  Only the debug/qualification path calls this; the hot path
    relies on the producer contract pinned by
    ``tests/test_qsa_selector_prefix_contract.py``.
    """

    rows = int(block_ids.shape[0])
    slots = int(block_ids.shape[1])
    ratio = int(compress_ratio)
    qpos = mx.arange(pos_start, pos_start + rows, dtype=mx.int32)
    complete = (qpos + 1) // ratio  # blocks fully visible to each row

    valid = block_valid
    counts = valid.astype(mx.int32).sum(axis=-1)
    prefix = mx.arange(slots, dtype=mx.int32)[None, :] < counts[:, None]
    expected_counts = mx.minimum(complete, mx.array(slots, dtype=mx.int32))

    pair_valid = mx.logical_and(valid[:, 1:], valid[:, :-1])
    ascending = mx.logical_or(
        mx.logical_not(pair_valid), block_ids[:, 1:] > block_ids[:, :-1]
    )
    in_range = mx.logical_or(
        mx.logical_not(valid),
        mx.logical_and(block_ids >= 0, block_ids < complete[:, None]),
    )

    checks = {
        "block_valid is not a per-row prefix": mx.all(prefix == valid),
        "valid count != min(512, complete blocks)": mx.all(
            counts == expected_counts
        ),
        "valid block ids are not strictly ascending": mx.all(ascending),
        "a valid block id is negative or not fully visible": mx.all(in_range),
    }
    mx.eval(list(checks.values()))
    for message, ok in checks.items():
        if not bool(ok.item()):
            return message
    return None


def qsa_prefill_direct_topk_buffer(
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    compress_ratio: int = _COMPRESS_RATIO,
    validate: bool | None = None,
) -> mx.array:
    """Adapt MTPLX's ``[S,512] int32`` + ``[S,512] bool`` to the native ABI.

    Returns a contiguous ``[1, 1, S, 512]`` uint32 buffer.  The cast is
    explicit and the contiguity is explicit: a bare reshape could hand the
    kernel a strided view, and a negative int32 id would become a huge
    uint32.  (The kernel would still mask that id via ``candidate < kL``, but
    "would be masked anyway" is not a contract.)
    """

    if block_ids.dtype != mx.int32:
        raise ValueError("block ids must be int32")
    if block_valid.dtype != mx.bool_:
        raise ValueError("block validity must be bool")
    if block_ids.ndim != 2 or tuple(block_valid.shape) != tuple(block_ids.shape):
        raise ValueError("block ids and validity must be matching [S, 512]")
    if int(block_ids.shape[1]) != _TOP_K_BLOCKS:
        raise ValueError("block ids must have exactly 512 slots")

    if validate is None:
        validate = _validate_enabled()
    if validate:
        violation = _prefix_contract_violation(
            block_ids,
            block_valid,
            pos_start=pos_start,
            compress_ratio=compress_ratio,
        )
        if violation is not None:
            raise ValueError(
                "QSA direct kernel prefix contract violated: " + violation
            )

    return mx.contiguous(block_ids.astype(mx.uint32)[None, None])


def qsa_prefill_direct(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    compress_ratio: int = _COMPRESS_RATIO,
    block_topk: int = _TOP_K_BLOCKS,
) -> mx.array:
    """Compute sparse-plus-tail QSA prefill attention as one Metal dispatch.

    ``queries`` is ``[1,24,S,256]``.  ``keys``/``values`` are the LOGICAL
    cache views returned by ``update_and_fetch`` (``kL == total_tokens``), not
    the capacity backings — the native ABI has no separate logical-K length.
    Selection arrays are ``[S,512]`` per query row.  The output is
    ``[1,24,S,256]`` in the Q/K/V dtype.

    Unsupported calls raise.  There is no dense fallback here: once this
    function dispatches, later Metal failures propagate to the caller.
    """

    reason = qsa_prefill_direct_unsupported_reason(
        queries,
        keys,
        values,
        block_ids,
        block_valid,
        pos_start=pos_start,
        total_tokens=total_tokens,
        scale=scale,
        compress_ratio=compress_ratio,
        block_topk=block_topk,
    )
    if reason is not None:
        raise ValueError(f"unsupported QSA prefill direct call: {reason}")

    selected = qsa_prefill_direct_topk_buffer(
        block_ids,
        block_valid,
        pos_start=int(pos_start),
        compress_ratio=int(compress_ratio),
    )
    out = _EXT.qwen4_qsa_sparse_gqa_attention(
        queries,
        keys,
        values,
        selected,
        float(scale),
        int(pos_start),
        key_tile=_KEY_TILE,
        dimension_tile=_DIMENSION_TILE,
    )
    return _prove_first_dispatch(out, dtype=queries.dtype)


def _prove_first_dispatch(out: mx.array, *, dtype: mx.Dtype) -> mx.array:
    """Force the first dispatch OF THIS DTYPE, then trust the kernel.

    MLX is lazy, so a missing or stale metallib (``get_library`` /
    ``get_kernel`` failures are eval-time) would otherwise surface at an
    arbitrary later point in the layer stack.  Evaluating here makes it a
    clean, attributable error AND retires the lane, so the next request
    routes to gather instead of hitting the same wall.  This is not a
    fallback: the error is re-raised, never swallowed.

    Per dtype, because the native kernel name embeds it: a proved bfloat16
    pipeline must not let the first float16 request skip its own proof and
    ride to an unattributed failure deep in the layer stack.
    """

    if _PIPELINE_STATE == _PIPELINE_FAILED or _dtype_proven(dtype):
        return out
    with _PIPELINE_LOCK:
        # Re-read under the lock: a racing caller may have proved or retired
        # this dtype while we waited, and FAILED must stay terminal.
        if _PIPELINE_STATE == _PIPELINE_FAILED or _dtype_proven(dtype):
            return out
        try:
            mx.eval(out)
        except Exception:
            _record_pipeline_failure()
            logger.warning(
                "%s: the QSA direct kernel failed its first %s evaluation; "
                "the lane is disabled for this process.",
                __name__,
                _dtype_key(dtype),
            )
            raise
        _record_pipeline_success(dtype)
    return out


def qsa_prefill_direct_preflight(*, dtype: mx.Dtype | None = None) -> bool:
    """Dispatch and evaluate one tiny real call per dtype; True when it works.

    Symbol presence is not pipeline readiness.  Benchmark harnesses and the
    server preflight call this once, before traffic, so a packaging mistake
    (renamed/absent metallib, missing specialization) cannot be discovered
    mid-request.  ``dtype=None`` proves every packaged specialization, which
    is what a before-traffic preflight wants; pass one dtype to prove just
    that pipeline.  A failure disables the lane process-wide.  Never call
    this on the hot path: it synchronizes.
    """

    if not _lane_eligible() or not _on_metal_device():
        return False
    if dtype is None:
        return _prove_all_pipelines()
    return _prove_pipeline(dtype=dtype)


def _prove_all_pipelines() -> bool:
    """Prove every packaged dtype specialization; the readiness answer.

    The Metal kernel name carries the query dtype, so the metallib can hold a
    valid bfloat16 specialization and a missing or stale float16 one.  Any
    failure retires the whole lane, so the first False is final.
    """

    for dtype in _SUPPORTED_DTYPES:
        if not _prove_pipeline(dtype=dtype):
            return False
    return True


def _prove_pipeline(*, dtype: mx.Dtype = mx.bfloat16) -> bool:
    """The proof itself, for ONE dtype.  Sets the process state; never raises.

    Consults ``_lane_eligible`` rather than ``qsa_prefill_direct_ready``,
    which calls back into here.  Serialized so that two threads racing their
    first prefill cannot both observe UNPROVEN and let a late success
    overwrite the other's FAILED.
    """

    if _PIPELINE_STATE == _PIPELINE_FAILED:
        return False
    if _dtype_proven(dtype):
        return True
    with _PIPELINE_LOCK:
        return _prove_pipeline_locked(dtype)


def _prove_pipeline_locked(dtype: mx.Dtype) -> bool:
    if _PIPELINE_STATE == _PIPELINE_FAILED:
        return False
    if _dtype_proven(dtype):
        return True
    rows = 8
    # Smallest shape that still crosses the dense/sparse boundary the
    # production gate requires (total // 4 > 512).
    total = 2056
    pos_start = total - rows
    queries = mx.zeros((_BATCH, _Q_HEADS, rows, _HEAD_DIM), dtype=dtype)
    keys = mx.zeros((_BATCH, _KV_HEADS, total, _HEAD_DIM), dtype=dtype)
    values = mx.zeros((_BATCH, _KV_HEADS, total, _HEAD_DIM), dtype=dtype)
    ids = mx.broadcast_to(
        mx.arange(_TOP_K_BLOCKS, dtype=mx.int32)[None, :], (rows, _TOP_K_BLOCKS)
    )
    block_ids = mx.contiguous(ids)
    block_valid = mx.ones((rows, _TOP_K_BLOCKS), dtype=mx.bool_)
    try:
        out = qsa_prefill_direct(
            queries,
            keys,
            values,
            block_ids,
            block_valid,
            pos_start=pos_start,
            total_tokens=total,
            scale=_EXPECTED_SCALE,
        )
        mx.eval(out)
    except Exception as exc:
        _record_pipeline_failure()
        logger.warning(
            "%s: QSA direct-kernel %s preflight failed; the lane is disabled "
            "for this process: %s",
            __name__,
            _dtype_key(dtype),
            exc,
        )
        return False
    _record_pipeline_success(dtype)
    return True
