"""Python surface for MTPLX's native (CMake + nanobind) MLX primitives.

Today this is one kernel: :func:`qsa_sparse_gqa`, the direct-index sparse-GQA
attention ported from oMLX (see
``native_extensions/qsa_sparse_gqa/sparse_gqa/steel_qsa_sparse_gqa.h`` for
provenance).  It is Steel MMA, so it cannot live in ``mx.fast.metal_kernel``
(the Laguna full-port verdict: no ``mlx::steel`` MMA reachable from
``metal_kernel``) and has to be a real MLX primitive in a built extension.

Phase 1 is standalone: nothing in ``mtplx/models/qwen4_exp.py`` calls this yet.
The gate order below deliberately mirrors
``mtplx.kernels.qsa_prefill_flash._unsupported_reason`` so the two attention
consumers refuse the same shapes for the same stated reason.

Build (CPU-only; no Metal execution)::

    cd native_extensions/qsa_sparse_gqa
    cmake -S . -B build \\
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/mtplx_native_qsa/ \\
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \\
      -DPython_EXECUTABLE=<venv>/bin/python
    cmake --build build -j 8

``python setup.py build_ext --inplace`` is the same build through setuptools;
it needs ``setuptools`` in the venv, which the current qwen38 venv does not
have (which is also why ``verify_mlp`` has no built artifact on this box).
"""

from __future__ import annotations

import math
import operator
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx

__all__ = [
    "native_qsa_available",
    "qsa_sparse_gqa",
    "qsa_sparse_gqa_decode",
    "qsa_sparse_gqa_decode_split_geometry",
    "qsa_sparse_gqa_decode_supported",
    "qsa_sparse_gqa_decode_unsupported_reason",
    "qsa_sparse_gqa_supported",
    "qsa_sparse_gqa_unsupported_reason",
]

# Production Qwen3.8 Flash-Next QSA geometry.  These are the ONLY shapes the
# kernel is instantiated for; everything else fails closed rather than
# silently changing the attention algorithm.
_BATCH = 1
_Q_HEADS = 24
_KV_HEADS = 2
_GQA = 12
_HEAD_DIM = 256
_COMPRESS_RATIO = 4
_TOP_K_BLOCKS = 512
_MAX_CONTEXT = 1_048_576
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16)
_SUPPORTED_ID_DTYPES = (mx.int32, mx.uint32)
#: (key_tile, dimension_tile) pairs the metallib instantiates.
_SUPPORTED_TILES = ((128, 32), (256, 32), (64, 64), (128, 64))
_DEFAULT_TILE = (128, 32)


def _extension_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "native_extensions"
        / "qsa_sparse_gqa"
    )


#: nanobind writes its ABI tag as one literal, e.g. ``v21_system_libcpp_abi1``.
_NB_ABI_TAG_RE = re.compile(
    rb"v(\d+)(?:[0-9a-zA-Z.\-]*)_[0-9a-zA-Z_]*(?:libcpp|libstdcpp|ms)[0-9a-zA-Z_]*"
)


def _nanobind_internals_version(binary: Path) -> int | None:
    """The nanobind internals version a shared object was built against."""

    try:
        data = binary.read_bytes()
    except OSError:
        return None
    versions = {int(m.group(1)) for m in _NB_ABI_TAG_RE.finditer(data)}
    return versions.pop() if len(versions) == 1 else None


@lru_cache(maxsize=1)
def _nanobind_abi_mismatch() -> str | None:
    """Precise reason when the extension cannot see mlx.core's type registry.

    Two nanobind modules share a type registry only when they agree on the
    capsule key ``__nb_internals_<abi_tag>_<domain>__``.  The domain is
    ``NB_DOMAIN=mlx`` on both sides; the tag carries ``NB_INTERNALS_VERSION``,
    which moves between nanobind releases.  A mismatch does not stop the build
    or the import -- it makes every call that takes an ``mx::array`` raise a
    bare ``TypeError`` whose signature line prints ``mlx::core::array`` instead
    of ``array``.  Diagnosing that from the TypeError alone cost a guarded
    GPU window, so it is named here instead.

    ``None`` when the tags match or cannot be read; a reason string otherwise.
    """

    ours = sorted(_extension_path().glob("mtplx_native_qsa/_ext*.so"))
    if not ours:
        return None
    core = sorted(Path(mx.__file__).parent.glob("core*.so")) if mx.__file__ else []
    if not core:
        return None
    ext_version = _nanobind_internals_version(ours[0])
    mlx_version = _nanobind_internals_version(core[0])
    if ext_version is None or mlx_version is None or ext_version == mlx_version:
        return None
    return (
        f"the built extension uses nanobind internals v{ext_version} but "
        f"mlx.core uses v{mlx_version}, so it cannot resolve mlx::core::array "
        "and every call raises TypeError; rebuild with "
        "-DMTPLX_NANOBIND_DIR=<nanobind whose src/nb_abi.h says "
        f"NB_INTERNALS_VERSION {mlx_version}> (diagnose with "
        "the native-ABI checker)"
    )


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    """Import the built extension, or return the import error.

    A successful import is not enough: the module can import cleanly and still
    be unable to cast a single array (see :func:`_nanobind_abi_mismatch`), so
    that check is folded in here rather than left to fail at the first call.
    """

    native_path = str(_extension_path())
    if native_path not in sys.path:
        sys.path.insert(0, native_path)
    try:
        import mtplx_native_qsa  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on build state
        return exc
    mismatch = _nanobind_abi_mismatch()
    if mismatch is not None:  # pragma: no cover - depends on build state
        return RuntimeError(mismatch)
    return mtplx_native_qsa


def native_qsa_available() -> bool:
    """True when the built extension imports."""

    return not isinstance(_load_extension(), Exception)


def _on_metal_device() -> bool:
    """Metal availability is insufficient when MLX currently targets CPU."""

    try:
        return mx.metal.is_available() and mx.default_device() == mx.gpu
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _normalized_block_ids(block_ids: mx.array, rows: int) -> mx.array | None:
    """Accept the selector's ``[S, 512]`` or the kernel ABI's ``[1,1,S,512]``.

    ``_select_eager`` emits ``[S, 512]``; the kernel wants ``[1, 1, S, 512]``.
    The reshape is a view on the contiguous selector output, not a copy, and
    the int32 dtype is accepted natively (the metallib instantiates both
    int32 and uint32) so the lane never pays an 8 MB astype per layer.
    """

    if block_ids.ndim == 2:
        if tuple(int(x) for x in block_ids.shape) != (rows, _TOP_K_BLOCKS):
            return None
        return block_ids.reshape(1, 1, rows, _TOP_K_BLOCKS)
    if block_ids.ndim == 4:
        if tuple(int(x) for x in block_ids.shape) != (
            _BATCH,
            1,
            rows,
            _TOP_K_BLOCKS,
        ):
            return None
        return block_ids
    return None


def qsa_sparse_gqa_unsupported_reason(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
) -> str | None:
    """``None`` when the call is on contract, else the precise reason."""

    extension = _load_extension()
    if isinstance(extension, Exception):
        return f"the native QSA extension is not built ({extension})"
    if not _on_metal_device():
        return "the active MLX device is not an available Metal GPU"

    arrays = (queries, keys, values, block_ids)
    if any(not isinstance(array, mx.array) for array in arrays):
        return "all tensor inputs must be MLX arrays"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "Q, K, and V must be rank four"
    if block_ids.ndim not in (2, 4):
        return "block ids must be rank two [S, 512] or rank four [1, 1, S, 512]"

    batch, query_heads, rows, head_dim = (int(x) for x in queries.shape)
    if (batch, query_heads, head_dim) != (_BATCH, _Q_HEADS, _HEAD_DIM):
        return "Q must have production shape [1, 24, S, 256]"
    if rows <= 0:
        return "Q must carry at least one query row"

    key_batch, kv_heads, capacity, key_dim = (int(x) for x in keys.shape)
    if (key_batch, kv_heads, key_dim) != (_BATCH, _KV_HEADS, _HEAD_DIM):
        return "K must have production shape [1, 2, capacity, 256]"
    if tuple(int(x) for x in values.shape) != tuple(int(x) for x in keys.shape):
        return "V must have the same full-backing shape as K"

    if queries.dtype not in _SUPPORTED_DTYPES:
        return "Q must be float16 or bfloat16"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "Q, K, and V dtypes must match"
    if block_ids.dtype not in _SUPPORTED_ID_DTYPES:
        return "block ids must be int32 or uint32"
    if _normalized_block_ids(block_ids, rows) is None:
        return "block ids must have shape [S, 512] or [1, 1, S, 512]"

    # Host scalars only: a traced scalar would make these comparisons
    # synchronize the graph.  Same contract as qsa_prefill_flash.
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
    if pos_start_i + rows > total_tokens_i:
        return "Q must be a causal suffix inside total_tokens"
    if total_tokens_i > capacity:
        return "the logical token count exceeds the full K/V backing capacity"
    if total_tokens_i > _MAX_CONTEXT:
        return "the logical token count exceeds the production context limit"
    if total_tokens_i // _COMPRESS_RATIO <= _TOP_K_BLOCKS:
        return "the context has not crossed the dense/sparse boundary"
    if not math.isfinite(scale_f):
        return "scale must be finite"

    if (int(key_tile), int(dimension_tile)) not in _SUPPORTED_TILES:
        return (
            "(key_tile, dimension_tile) must be one of "
            + ", ".join(str(t) for t in _SUPPORTED_TILES)
        )
    return None


def qsa_sparse_gqa_supported(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
) -> bool:
    """Whether the exact production-only kernel contract is met."""

    return (
        qsa_sparse_gqa_unsupported_reason(
            queries,
            keys,
            values,
            block_ids,
            pos_start=pos_start,
            total_tokens=total_tokens,
            scale=scale,
            key_tile=key_tile,
            dimension_tile=dimension_tile,
        )
        is None
    )


def qsa_sparse_gqa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
    stream: Any = None,
) -> mx.array:
    """Direct-index sparse GQA attention over chronological QSA block ids.

    ``queries``  ``[1, 24, S, 256]``  fp16/bf16; the ``[B,H,S,D]`` transposed
                 view the Attention module already builds.
    ``keys``/``values``  ``[1, 2, capacity, 256]`` -- the FULL KV cache
                 backing, read in place at its allocation stride.  Never slice
                 it to ``total_tokens`` first: that copy is the whole context.
    ``block_ids``  ``[S, 512]`` int32 (``_select_eager``'s ``flash_prefill``
                 output) or ``[1, 1, S, 512]``.  Chronological, and the valid
                 entries must occupy the leading
                 ``min(512, (pos + 1) // 4)`` slots of each row -- which is
                 what the selector produces, because it sorts the raw top-k
                 ascending and validity there is the threshold predicate
                 ``id < complete_blocks``.  The kernel derives validity from
                 that invariant instead of reading ``block_valid``; the
                 standalone harness asserts it.
    ``total_tokens``  logical tokens in the cache (NOT ``capacity``).

    Returns ``[1, 24, S, 256]``, same dtype as ``queries``.

    Numerics: fp32 online softmax (exp2) and fp32 P@V over the same visible
    set as the dense lane -- a rounding-class difference, not an exactness
    one.  See the sparse-GQA microbenchmark for the tolerance
    statement and the measured deltas.
    """

    reason = qsa_sparse_gqa_unsupported_reason(
        queries,
        keys,
        values,
        block_ids,
        pos_start=pos_start,
        total_tokens=total_tokens,
        scale=scale,
        key_tile=key_tile,
        dimension_tile=dimension_tile,
    )
    if reason is not None:
        raise ValueError(f"[mtplx.native.qsa_sparse_gqa] {reason}.")

    extension = _load_extension()
    selected = _normalized_block_ids(block_ids, int(queries.shape[2]))
    args = (
        queries,
        keys,
        values,
        selected,
        float(scale),
        int(pos_start),
        int(total_tokens),
        int(key_tile),
        int(dimension_tile),
    )
    # Omit the kwarg entirely when no stream was asked for, rather than
    # passing an explicit None: the binding's default is applied by nanobind
    # without going through the StreamOrDevice caster at all, which is one
    # fewer thing to be wrong about in an extension module.
    if stream is None:
        return extension.qsa_sparse_gqa_attention(*args)
    return extension.qsa_sparse_gqa_attention(*args, stream=stream)


# ---------------------------------------------------------------------------
# Split-K (KV-split) DECODE variant -- M=4 fixed verify and M=1 draft.
#
# Separate entry points, not a rows argument on the prefill one, because the
# two differ in more than the row count:
#
#   * the grid's z axis is the KV SPLIT here, and the kernel is two dispatches
#     (split + merge) rather than one;
#   * validity is decided PER SLOT rather than by a leading-prefix cut,
#     because the decode selector hands ``mx.argpartition``'s raw, UNSORTED
#     output straight through, while the prefill selector sorts;
#   * the query offset is a device buffer, so a tensor-valued cache offset
#     never has to be read on the host.
# ---------------------------------------------------------------------------

#: The kernel's own selected-token width: 512 blocks x 4 tokens plus the at
#: most three causal tail tokens of the incomplete block.  The shipped lane
#: builds 2,052 slots; its 2,052nd is invalid for every query position (see
#: the note in ``qsa_sparse_gqa_decode``), so the two visible sets agree.
_SELECTED_TOKENS = _TOP_K_BLOCKS * _COMPRESS_RATIO + (_COMPRESS_RATIO - 1)
#: Partial rows are [O(head_dim) | m | l] in fp32.
_PARTIAL_LD = _HEAD_DIM + 2
_MAX_KEY_SPLITS = 64
#: Matches mtplx.runtime_options.FABLE_QSA_SPARSE_DECODE_DEFAULT_SPLITS; a
#: test pins the two together so the bench and the lane cannot drift.
_DEFAULT_KEY_SPLITS = 17
_INTEGER_DTYPES = (
    mx.int8,
    mx.int16,
    mx.int32,
    mx.int64,
    mx.uint8,
    mx.uint16,
    mx.uint32,
    mx.uint64,
)


def qsa_sparse_gqa_decode_split_geometry(
    selected_tokens: int = _SELECTED_TOKENS,
    key_tile: int = _DEFAULT_TILE[0],
    key_splits: int = _DEFAULT_KEY_SPLITS,
) -> tuple[int, int, int]:
    """``(n_tiles, tiles_per_split, n_splits)`` for the split-K decode grid.

    A pure-host mirror of the C++ ``qsa_sparse_gqa_decode_split_geometry`` so
    the harness, the tests and the partial-buffer sizing never restate the
    arithmetic.  ``tests/test_fable_qsa_sparse_decode.py`` pins the two
    against each other whenever the extension is built.

    The rounding is deliberate: ``n_splits`` is recomputed from
    ``tiles_per_split`` so the LAST split always has work.  With 17 tiles and
    8 requested splits, ``tiles_per_split`` is 3 and six splits cover the
    range -- dispatching eight would leave two threadgroups writing an empty
    online-softmax state that the merge then has to skip.
    """

    if int(selected_tokens) <= 0:
        raise ValueError(f"selected_tokens must be positive; got {selected_tokens}")
    if int(key_tile) <= 0:
        raise ValueError(f"key_tile must be positive; got {key_tile}")
    tiles = -(-int(selected_tokens) // int(key_tile))
    splits = min(int(key_splits), tiles)
    if splits < 1:
        splits = 1
    per_split = -(-tiles // splits)
    exact_splits = -(-tiles // per_split)
    return tiles, per_split, exact_splits


def qsa_sparse_gqa_decode_partial_shape(
    rows: int,
    key_tile: int = _DEFAULT_TILE[0],
    key_splits: int = _DEFAULT_KEY_SPLITS,
) -> tuple[int, int, int, int]:
    """Shape of the fp32 partial-state buffer the split pass writes."""

    _, _, n_splits = qsa_sparse_gqa_decode_split_geometry(
        _SELECTED_TOKENS, key_tile, key_splits
    )
    return (n_splits, _Q_HEADS, int(rows), _PARTIAL_LD)


def _normalized_query_offset(query_offset: Any) -> mx.array | None:
    """Accept a host int or a one-element int32 array; emit the ABI's ``[1]``.

    A host int becomes a one-element array rather than a params-block scalar
    so the two spellings take the SAME kernel path, and so a tensor-valued
    cache offset (``TensorOffsetKVCache``) never forces a graph sync just to
    read a position the kernel is about to read anyway.
    """

    if isinstance(query_offset, mx.array):
        if query_offset.size != 1:
            return None
        # A ``TensorOffsetKVCache`` offset is a 0-d int32 array; accept every
        # exact integer width and narrow, because a one-element astype costs
        # nothing and a refusal here would raise on a perfectly valid cache.
        if query_offset.dtype not in _INTEGER_DTYPES:
            return None
        return query_offset.reshape(1).astype(mx.int32)
    if isinstance(query_offset, bool):
        return None
    try:
        value = operator.index(query_offset)
    except TypeError:
        return None
    if value < 0:
        return None
    return mx.array([value], dtype=mx.int32)


def qsa_sparse_gqa_decode_unsupported_reason(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    query_offset: Any,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
    key_splits: int = _DEFAULT_KEY_SPLITS,
) -> str | None:
    """``None`` when the decode call is on contract, else the precise reason."""

    extension = _load_extension()
    if isinstance(extension, Exception):
        return f"the native QSA extension is not built ({extension})"
    if not _on_metal_device():
        return "the active MLX device is not an available Metal GPU"

    arrays = (queries, keys, values, block_ids)
    if any(not isinstance(array, mx.array) for array in arrays):
        return "all tensor inputs must be MLX arrays"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "Q, K, and V must be rank four"
    if block_ids.ndim not in (2, 4):
        return "block ids must be rank two [M, 512] or rank four [1, 1, M, 512]"

    batch, query_heads, rows, head_dim = (int(x) for x in queries.shape)
    if (batch, query_heads, head_dim) != (_BATCH, _Q_HEADS, _HEAD_DIM):
        return "Q must have production shape [1, 24, M, 256]"
    if rows <= 0:
        return "Q must carry at least one query row"

    key_batch, kv_heads, capacity, key_dim = (int(x) for x in keys.shape)
    if (key_batch, kv_heads, key_dim) != (_BATCH, _KV_HEADS, _HEAD_DIM):
        return "K must have production shape [1, 2, capacity, 256]"
    if tuple(int(x) for x in values.shape) != tuple(int(x) for x in keys.shape):
        return "V must have the same full-backing shape as K"

    if queries.dtype not in _SUPPORTED_DTYPES:
        return "Q must be float16 or bfloat16"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "Q, K, and V dtypes must match"
    if block_ids.dtype not in _SUPPORTED_ID_DTYPES:
        return "block ids must be int32 or uint32"
    if _normalized_block_ids(block_ids, rows) is None:
        return "block ids must have shape [M, 512] or [1, 1, M, 512]"
    if _normalized_query_offset(query_offset) is None:
        return (
            "query_offset must be a non-negative host int or a one-element "
            "int32 array"
        )

    if isinstance(total_tokens, mx.array):
        return "total_tokens must be a host integer"
    if isinstance(scale, mx.array):
        return "scale must be a host float"
    if isinstance(total_tokens, bool):
        return "total_tokens cannot be bool"
    try:
        total_tokens_i = operator.index(total_tokens)
    except TypeError:
        return "total_tokens must be an exact host integer"
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        return "scale must be a numeric host scalar"
    scale_f = float(scale)

    if total_tokens_i <= 0:
        return "total_tokens must describe a non-empty context"
    if rows > total_tokens_i:
        return "the query rows must fit inside total_tokens"
    if total_tokens_i > capacity:
        return "the logical token count exceeds the full K/V backing capacity"
    if total_tokens_i > _MAX_CONTEXT:
        return "the logical token count exceeds the production context limit"
    if total_tokens_i // _COMPRESS_RATIO <= _TOP_K_BLOCKS:
        return "the context has not crossed the dense/sparse boundary"
    if not math.isfinite(scale_f):
        return "scale must be finite"

    if (int(key_tile), int(dimension_tile)) not in _SUPPORTED_TILES:
        return (
            "(key_tile, dimension_tile) must be one of "
            + ", ".join(str(t) for t in _SUPPORTED_TILES)
        )
    if not 1 <= int(key_splits) <= _MAX_KEY_SPLITS:
        return f"key_splits must be in [1, {_MAX_KEY_SPLITS}]"
    return None


def qsa_sparse_gqa_decode_supported(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    query_offset: Any,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
    key_splits: int = _DEFAULT_KEY_SPLITS,
) -> bool:
    """Whether the exact production-only decode contract is met."""

    return (
        qsa_sparse_gqa_decode_unsupported_reason(
            queries,
            keys,
            values,
            block_ids,
            query_offset=query_offset,
            total_tokens=total_tokens,
            scale=scale,
            key_tile=key_tile,
            dimension_tile=dimension_tile,
            key_splits=key_splits,
        )
        is None
    )


def qsa_sparse_gqa_decode(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    query_offset: Any,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
    key_splits: int = _DEFAULT_KEY_SPLITS,
    stream: Any = None,
) -> mx.array:
    """Split-K direct-index sparse GQA attention for the decode geometries.

    ``queries``  ``[1, 24, M, 256]`` fp16/bf16 -- the ``[B,H,M,D]`` transposed
                 view the Attention module already builds.  M is 4 for the
                 fixed-M4 verify and 1 for a single-row draft/decode step.
    ``keys``/``values``  ``[1, 2, capacity, 256]``, the FULL KV cache backing,
                 read in place at its allocation stride.
    ``block_ids``  ``[M, 512]`` or ``[1, 1, M, 512]``, int32 or uint32 --
                 ``mx.argpartition``'s output IN ITS OWN ORDER.  Unlike the
                 prefill entry point this one makes NO ordering assumption and
                 reads no ``block_valid``: it applies the shipped lane's own
                 per-slot predicate ``block < (pos + 1) // 4`` to every slot.
                 So the visible set is identical whether or not the selector
                 sorts.
    ``query_offset``  absolute position of query row 0, as a host int or a
                 one-element int32 array (a tensor-valued cache offset never
                 has to be read on the host).
    ``total_tokens``  logical tokens in the cache (NOT ``capacity``).
    ``key_splits``  target KV splits; see
                 :func:`qsa_sparse_gqa_decode_split_geometry` for how it is
                 clamped and rounded.

    Returns ``[1, 24, M, 256]``, same dtype as ``queries``.

    NUMERICS -- this is a ROUNDING-CLASS change, HumanEval-gated
    -----------------------------------------------------------
    Against the shipped rows-gather lane, over an IDENTICAL visible set:

      * scores accumulate through Steel MMA fp32 fragments, not MLX's gemv
        tiling, so the 256-term contraction is reassociated;
      * the softmax is an fp32 ONLINE softmax in ``exp2`` with the scale
        pre-multiplied by ``M_LOG2E``, not an fp32 ``exp`` over a
        materialised score row;
      * probabilities stay fp32 instead of being cast to bf16 before P@V,
        and P@V runs fp32 x fp32 instead of bf16 x bf16 with fp32 accumulate;
      * the split-K merge adds one more rescale per query row.

    None of that is bit-exact and none of it can be made so.  Adopt this lane
    on the same terms as ``MTPLX_FABLE_HC_M4``: greedy-token agreement plus a
    full HumanEval gate, never on a digest comparison.

    The shipped lane builds ``topk*ratio + ratio`` = 2,052 token slots; this
    kernel walks ``topk*ratio + ratio - 1`` = 2,051.  The dropped slot is the
    tail's fourth, whose token is ``((pos+1)//4)*4 + 3``; that is ``> pos``
    for every ``pos``, so the shipped lane always masks it.  The visible sets
    are equal.
    """

    reason = qsa_sparse_gqa_decode_unsupported_reason(
        queries,
        keys,
        values,
        block_ids,
        query_offset=query_offset,
        total_tokens=total_tokens,
        scale=scale,
        key_tile=key_tile,
        dimension_tile=dimension_tile,
        key_splits=key_splits,
    )
    if reason is not None:
        raise ValueError(f"[mtplx.native.qsa_sparse_gqa_decode] {reason}.")

    extension = _load_extension()
    selected = _normalized_block_ids(block_ids, int(queries.shape[2]))
    offset = _normalized_query_offset(query_offset)
    args = (
        queries,
        keys,
        values,
        selected,
        offset,
        float(scale),
        int(total_tokens),
        int(key_tile),
        int(dimension_tile),
        int(key_splits),
    )
    # See qsa_sparse_gqa: omit rather than pass an explicit None.
    if stream is None:
        return extension.qsa_sparse_gqa_decode(*args)
    return extension.qsa_sparse_gqa_decode(*args, stream=stream)


# ---------------------------------------------------------------------------
# CPU-stream PLE row staging extension (mtplx_native_ple_cpu_rows)
# ---------------------------------------------------------------------------
#
# The cached async PLE lane (mtplx/ple_cached_aux.py) needs a second built
# extension: the CPU-stream sidecar row producer in
# native_extensions/ple_cpu_rows. It is loaded exactly the way the QSA kernel
# above is -- its package directory is put on sys.path and imported -- and the
# same nanobind/mlx ABI check applies, because it too casts mx::array.

#: The API the cached lane requires from the extension.
_PLE_CPU_ROWS_REQUIRED = (
    "install_cached_sidecar_provider",
    "compute_cached_row_ids",
    "make_cached_sidecar_rows",
    "drain_cached_completions",
)


def _ple_cpu_rows_extension_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "native_extensions"
        / "ple_cpu_rows"
    )


@lru_cache(maxsize=1)
def _ple_cpu_rows_abi_mismatch() -> str | None:
    """nanobind-vs-mlx.core ABI reason for the PLE extension, or ``None``.

    Same failure mode as :func:`_nanobind_abi_mismatch`: a matching build and
    import that still cannot cast an ``mx::array``. Named here so the reason is
    printed rather than surfacing as a bare ``TypeError`` at first call.
    """

    ours = sorted(_ple_cpu_rows_extension_path().glob(
        "mtplx_native_ple_cpu_rows/_ext*.so"
    ))
    if not ours:
        return None
    core = sorted(Path(mx.__file__).parent.glob("core*.so")) if mx.__file__ else []
    if not core:
        return None
    ext_version = _nanobind_internals_version(ours[0])
    mlx_version = _nanobind_internals_version(core[0])
    if ext_version is None or mlx_version is None or ext_version == mlx_version:
        return None
    return (
        f"the built PLE extension uses nanobind internals v{ext_version} but "
        f"mlx.core uses v{mlx_version}, so it cannot resolve mlx::core::array "
        "and every call raises TypeError; rebuild it with "
        "-DMTPLX_NANOBIND_DIR set to a nanobind whose src/nb_abi.h says "
        f"NB_INTERNALS_VERSION {mlx_version}"
    )


@lru_cache(maxsize=1)
def _load_ple_cpu_rows_extension() -> Any:
    """Import the built PLE extension, or return the import/ABI error."""

    native_path = str(_ple_cpu_rows_extension_path())
    if native_path not in sys.path:
        sys.path.insert(0, native_path)
    try:
        import mtplx_native_ple_cpu_rows  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on build state
        return exc
    mismatch = _ple_cpu_rows_abi_mismatch()
    if mismatch is not None:  # pragma: no cover - depends on build state
        return RuntimeError(mismatch)
    missing = [
        name
        for name in _PLE_CPU_ROWS_REQUIRED
        if not callable(getattr(mtplx_native_ple_cpu_rows, name, None))
    ]
    if missing:  # pragma: no cover - depends on build state
        return RuntimeError(
            "the built PLE extension lacks required callables: "
            + ", ".join(missing)
        )
    return mtplx_native_ple_cpu_rows


def native_ple_cpu_rows_available() -> bool:
    """True when the built PLE extension imports and exposes its cached API."""

    return not isinstance(_load_ple_cpu_rows_extension(), Exception)


def ple_cpu_rows_unavailable_reason() -> str | None:
    """The reason the PLE extension cannot be used, or ``None`` when it can.

    A short, printable string for the cached PLE lane's graceful decline when
    the extension has not been built.
    """

    loaded = _load_ple_cpu_rows_extension()
    if not isinstance(loaded, Exception):
        return None
    if isinstance(loaded, ModuleNotFoundError):
        return (
            "native_extensions/ple_cpu_rows is not built "
            "(run scripts/fable/setup_over100_venv.sh)"
        )
    return f"{type(loaded).__name__}: {loaded}"


def load_ple_cpu_rows_extension() -> Any:
    """Return the imported PLE extension module, raising the stored error.

    The cached PLE lane calls :func:`ple_cpu_rows_unavailable_reason` first and
    declines when it is not ``None``; this is the accessor for the armed path.
    """

    loaded = _load_ple_cpu_rows_extension()
    if isinstance(loaded, Exception):
        raise loaded
    return loaded


__all__ += [
    "native_ple_cpu_rows_available",
    "ple_cpu_rows_unavailable_reason",
    "load_ple_cpu_rows_extension",
]
