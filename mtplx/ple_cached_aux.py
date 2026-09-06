"""Construction-bound cached native PLE auxiliary for fixed M4.

This lane reuses the stock fixed-M4 builder and the owner-thread stock sidecar
cache.  A native provider computes the fixed 64 row IDs and reads cold rows;
Python owns only the compact hit/miss handoff and publication.  The auxiliary
embedding plane is produced outside the compiled verifier via ``mx.async_eval``
so PLE-row production overlaps compiled replay.  It does not change the compiled
graph arithmetic, so the output is identical to the stock path.

The module is inert on import and never imports MLX until an explicit installer
is called.  The native provider it needs lives in the ``mtplx_native_ple_cpu_rows``
extension (``native_extensions/ple_cpu_rows``), loaded through
:mod:`mtplx.native`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import numpy as np

from . import ple_cached_row_handoff


PENDING_LIMIT = 2
NATIVE_CACHED_PROVIDER_API = frozenset(
    {
        "install_cached_sidecar_provider",
        "compute_cached_row_ids",
        "make_cached_sidecar_rows",
        "drain_cached_completions",
    }
)
_EMPTY_PACKED_MISSES = np.empty(
    (0, ple_cached_row_handoff.PACKED_ROW_BYTES), dtype=np.uint8
)
_EMPTY_PACKED_MISSES.flags.writeable = False


# --- the exact production fixed-M4 sidecar contract -------------------------
#
# These constants and :func:`_validate_contract` freeze the one sidecar
# geometry the native provider is built for, and are computed once at
# installation.  Nothing here is read again in the per-token call.
_EXPECTED_CONTEXT_LEN = 2
_EXPECTED_NGRAM_SIZE = 3
_EXPECTED_HEADS_PER_NGRAM = 8
_EXPECTED_OUTPUT_DIM = 2560
_EXPECTED_BITS = 4
_EXPECTED_GROUP_SIZE = 32
_EXPECTED_HEAD_COUNT = 16
_EXPECTED_WEIGHT_SHAPE = (20,)
_EXPECTED_METADATA_SHAPE = (5,)
_EXPECTED_WEIGHT_ROW_BYTES = 80
_EXPECTED_METADATA_ROW_BYTES = 10
_PLANE_NAMES = ("weight", "scales", "biases")


def _shape(value: Any) -> tuple[int, ...]:
    """Return a tuple shape without importing MLX or touching model state."""

    return tuple(int(dimension) for dimension in value.shape)


def _validate_contract(inner: Any, embedding: Any, sidecar: Any) -> dict[str, Any]:
    """Validate and freeze the exact production sidecar contract once."""

    observed = (
        int(embedding.context_len),
        int(embedding.ngram_size),
        int(embedding.heads_per_ngram),
        int(inner.args.ple_embed_dim),
        int(sidecar.bits),
        int(sidecar.group_size),
    )
    expected = (
        _EXPECTED_CONTEXT_LEN,
        _EXPECTED_NGRAM_SIZE,
        _EXPECTED_HEADS_PER_NGRAM,
        _EXPECTED_OUTPUT_DIM,
        _EXPECTED_BITS,
        _EXPECTED_GROUP_SIZE,
    )
    if observed != expected:
        raise ValueError(
            "cached native PLE sidecar geometry mismatch: "
            f"observed={observed} expected={expected}"
        )

    # _np_consts is the model's already-derived hash plan.  Compute it once at
    # installation and pass these exact values to the native provider; never
    # reconstruct metadata in the per-token call.
    mult, sizes, offsets = embedding._np_consts()
    constants = (mult, sizes, offsets)
    if tuple(_shape(value) for value in constants) != (
        (3,),
        (_EXPECTED_HEAD_COUNT,),
        (_EXPECTED_HEAD_COUNT,),
    ):
        raise ValueError(
            "cached native PLE sidecar hash constants mismatch: "
            f"shapes={tuple(_shape(value) for value in constants)}"
        )

    maps: dict[str, tuple[int, int]] = {}
    row_count: int | None = None
    expected_planes = {
        "weight": (_EXPECTED_WEIGHT_SHAPE, "U32", _EXPECTED_WEIGHT_ROW_BYTES),
        "scales": (_EXPECTED_METADATA_SHAPE, "BF16", _EXPECTED_METADATA_ROW_BYTES),
        "biases": (_EXPECTED_METADATA_SHAPE, "BF16", _EXPECTED_METADATA_ROW_BYTES),
    }
    for name in _PLANE_NAMES:
        try:
            matrix, dtype_name = sidecar._maps[name]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"cached native PLE sidecar is missing {name} map"
            ) from exc
        shape = _shape(matrix)
        expected_shape, expected_dtype, row_bytes = expected_planes[name]
        if len(shape) != 2 or shape[1:] != expected_shape:
            raise ValueError(
                f"cached native PLE {name} shape mismatch: "
                f"observed={shape} expected=(*,{expected_shape})"
            )
        if str(dtype_name) != expected_dtype:
            raise ValueError(
                f"cached native PLE {name} dtype mismatch: "
                f"observed={dtype_name!r} expected={expected_dtype!r}"
            )
        current_rows = int(shape[0])
        if row_count is None:
            row_count = current_rows
        elif current_rows != row_count:
            raise ValueError(
                "cached native PLE sidecar plane row counts differ: "
                f"{row_count} versus {current_rows} ({name})"
            )
        if current_rows <= 0:
            raise ValueError("cached native PLE sidecar row count must be positive")
        try:
            offset = int(matrix.offset)
            nbytes = int(matrix.nbytes)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cached native PLE {name} map lacks offset/nbytes"
            ) from exc
        if offset < 0 or nbytes != current_rows * row_bytes:
            raise ValueError(
                f"cached native PLE {name} byte span mismatch: "
                f"offset={offset} nbytes={nbytes} "
                f"expected_nbytes={current_rows * row_bytes}"
            )
        maps[name] = (offset, nbytes)

    try:
        source_fd = int(sidecar._fd)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "cached native PLE sidecar has no readable descriptor"
        ) from exc

    # Convert the one-time numpy metadata into the std::array-compatible tuples
    # expected by nanobind.  The values are copied now, not read again from the
    # model during generation.
    native_constants = tuple(
        tuple(int(value) for value in np.asarray(array).reshape(-1))
        for array in constants
    )
    return {
        "embedding": embedding,
        "sidecar": sidecar,
        "row_count": int(row_count),
        "maps": maps,
        "source_fd": source_fd,
        "multipliers": native_constants[0],
        "sizes": native_constants[1],
        "offsets": native_constants[2],
        "eos": int(embedding.eos_id),
        "output_dim": _EXPECTED_OUTPUT_DIM,
    }


@dataclass(frozen=True, slots=True)
class CachedAuxInstallation:
    """Installation-owned resources shared by every built aux wrapper."""

    provider: Any
    auxiliary_stream: Any
    _state: Any

    @property
    def pending_count(self) -> int:
        return self._state.pending_count


class _CachedAuxInstallationState:
    """Bounded provider state and terminal failure boundary."""

    __slots__ = (
        "provider",
        "handoff",
        "compute_cached_row_ids",
        "make_cached_sidecar_rows",
        "drain_cached_completions",
        "pending",
        "failure",
    )

    def __init__(
        self,
        *,
        provider: Any,
        handoff: ple_cached_row_handoff.CachedRowHandoff,
        compute_cached_row_ids: Callable[..., Any],
        make_cached_sidecar_rows: Callable[..., Any],
        drain_cached_completions: Callable[..., Any],
    ) -> None:
        self.provider = provider
        self.handoff = handoff
        self.compute_cached_row_ids = compute_cached_row_ids
        self.make_cached_sidecar_rows = make_cached_sidecar_rows
        self.drain_cached_completions = drain_cached_completions
        self.pending: dict[Any, ple_cached_row_handoff.PreparedRows] = {}
        self.failure: BaseException | None = None

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def ensure_healthy(self) -> None:
        if self.failure is not None:
            raise RuntimeError(
                "cached native PLE installation failed; model reload required"
            ) from self.failure

    def fail(self, error: BaseException) -> None:
        if self.failure is None:
            self.failure = error
        # Native jobs are no longer safe to associate with Python cache state;
        # release the bounded owner references and stop permanently.
        self.pending.clear()

    def register(
        self, ticket: Any, prepared: ple_cached_row_handoff.PreparedRows
    ) -> None:
        self.ensure_healthy()
        try:
            if ticket is None:
                raise RuntimeError("miss-bearing cached PLE request returned no ticket")
            if ticket in self.pending:
                raise RuntimeError("cached native PLE ticket was registered twice")
            if len(self.pending) >= PENDING_LIMIT:
                raise RuntimeError(
                    f"cached native PLE pending limit {PENDING_LIMIT} exceeded"
                )
            # Register before the caller submits the returned planes to MLX.
            self.pending[ticket] = prepared
        except BaseException as error:
            self.fail(error)
            raise

    def publish_all_hit(self, prepared: ple_cached_row_handoff.PreparedRows) -> None:
        self.ensure_healthy()
        try:
            self.handoff.publish(
                self.handoff.trusted_completion(prepared, _EMPTY_PACKED_MISSES)
            )
        except BaseException as error:
            self.fail(error)
            raise

    def drain(self) -> None:
        """Drain and publish all known completions before any new route work."""

        self.ensure_healthy()
        try:
            completions = self.drain_cached_completions(self.provider)
            pairs = []
            seen = set()
            for ticket, packed in completions:
                if ticket is None or ticket in seen or ticket not in self.pending:
                    raise RuntimeError("cached native PLE completion ticket is unknown")
                seen.add(ticket)
                pairs.append((ticket, self.pending[ticket], packed))
            # Validate association for the entire drained batch before any
            # cache publication, so an unknown later ticket cannot partially
            # publish an earlier completion.
            for ticket, prepared, packed in pairs:
                self.pending.pop(ticket)
                self.handoff.publish(self.handoff.trusted_completion(prepared, packed))
        except BaseException as error:
            self.fail(error)
            raise


def _install_cached_provider(native_module: Any, contract: dict[str, Any]) -> Any:
    """Install the cached provider with the original immutable layout/hash."""

    duplicate_fd = os.dup(contract["source_fd"])
    try:
        return native_module.install_cached_sidecar_provider(
            duplicate_fd,
            contract["row_count"],
            contract["maps"]["weight"][0],
            contract["maps"]["weight"][1],
            contract["maps"]["scales"][0],
            contract["maps"]["scales"][1],
            contract["maps"]["biases"][0],
            contract["maps"]["biases"][1],
            contract["multipliers"],
            contract["sizes"],
            contract["offsets"],
            contract["eos"],
            io_workers=8,
        )
    finally:
        os.close(duplicate_fd)


class _CachedFixedM4SidecarAux:
    """Stock warm ownership plus installation-shared cached row handoff."""

    __slots__ = (
        "_auxiliary_stream",
        "_compute_cached_row_ids",
        "_dequantize",
        "_make_cached_sidecar_rows",
        "_previous_tokens",
        "_state",
        "_stock",
        "_stream",
        "_submit_embedding",
        "_submit_planes",
        "_submit_warm",
        "_output_dim",
    )

    def __init__(
        self,
        stock_aux: Any,
        *,
        state: _CachedAuxInstallationState,
        auxiliary_stream: Any,
        compute_cached_row_ids: Callable[..., Any],
        make_cached_sidecar_rows: Callable[..., Any],
        previous_tokens: Callable[..., tuple[int, int]],
        submit_planes: Callable[..., Any],
        submit_embedding: Callable[..., Any],
        mx_module: Any,
        output_dim: int,
    ) -> None:
        self._stock = stock_aux
        self._state = state
        self._auxiliary_stream = auxiliary_stream
        self._compute_cached_row_ids = compute_cached_row_ids
        self._make_cached_sidecar_rows = make_cached_sidecar_rows
        self._previous_tokens = previous_tokens
        self._submit_planes = submit_planes
        self._submit_embedding = submit_embedding
        self._dequantize = mx_module.dequantize
        self._stream = mx_module.stream
        self._output_dim = int(output_dim)
        self._submit_warm = stock_aux._submit_warm

    def prefetch_primary(
        self,
        primary: int,
        completion_tokens: Any,
        committed_count: int,
    ) -> None:
        self._state.ensure_healthy()
        self._state.drain()
        try:
            return self._stock.prefetch_primary(
                primary,
                completion_tokens,
                committed_count,
            )
        except BaseException as error:
            self._state.fail(error)
            raise

    def __call__(
        self,
        _input_ids: Any,
        host_input_ids: Any,
        completion_tokens: Any,
        committed_count: int,
    ) -> Any:
        self._state.ensure_healthy()
        self._state.drain()

        # Preserve stock pending-warm capture -> clear -> install before the
        # cached row lookup.  The stock prefetch callback remains untouched.
        stock = self._stock
        try:
            pending_warm = stock._pending_warm
            stock._pending_warm = ()
            stock._install_owned_rows(pending_warm)
            previous = self._previous_tokens(
                stock._prompt_tail,
                completion_tokens,
                committed_count,
            )
            current_ids = tuple(int(value) for value in host_input_ids)
            row_ids = self._compute_cached_row_ids(
                self._state.provider,
                previous,
                current_ids,
            )
            prepared = self._state.handoff.prepare(row_ids)
            ticket, planes = self._make_cached_sidecar_rows(
                self._state.provider,
                prepared.source,
                prepared.hit_packed,
                prepared.miss_ids,
            )
            if ticket is None:
                if prepared.miss_count:
                    raise RuntimeError(
                        "cached native PLE miss request returned no completion ticket"
                    )
                self._state.publish_all_hit(prepared)
            else:
                if not prepared.miss_count:
                    raise RuntimeError(
                        "cached native PLE all-hit request returned a ticket"
                    )
                self._state.register(ticket, prepared)

            self._submit_planes(*planes)
            with self._stream(self._auxiliary_stream):
                embedding = self._dequantize(
                    planes[0],
                    planes[1],
                    planes[2],
                    group_size=_EXPECTED_GROUP_SIZE,
                    bits=_EXPECTED_BITS,
                ).reshape(1, 4, self._output_dim)
                self._submit_embedding(embedding)
            return embedding
        except BaseException as error:
            self._state.fail(error)
            raise


def _install_cached_builder(
    runtime: Any,
    *,
    native_module: Any,
    mx_module: Any,
    stock_module: Any | None,
    submit_planes: Callable[..., Any],
    submit_embedding: Callable[..., Any],
) -> CachedAuxInstallation:
    if stock_module is None:
        from . import qwen4_fixed_verify as stock_module
    if not callable(submit_planes) or not callable(submit_embedding):
        raise ValueError("cached native PLE requires bound plane/embedding submitters")
    original_builder = getattr(runtime, "build_fixed_m4_compiled_verify_aux", None)
    if not callable(original_builder):
        raise ValueError("cached native PLE requires the stock compiled-aux builder")
    required = (
        "install_cached_sidecar_provider",
        "compute_cached_row_ids",
        "make_cached_sidecar_rows",
        "drain_cached_completions",
    )
    if any(not callable(getattr(native_module, name, None)) for name in required):
        raise ValueError("cached native PLE provider API is incomplete")

    inner = stock_module._inner(runtime)
    layer_index = int(inner._ple_stage_idx)
    ple = inner.layers[layer_index].ple
    embedding = ple.ple_embedding
    sidecar = embedding.ngram_embedding._sidecar
    if sidecar is None:
        raise ValueError("cached native PLE requires the sidecar table")
    contract = _validate_contract(inner, embedding, sidecar)
    provider = _install_cached_provider(native_module, contract)
    handoff = ple_cached_row_handoff.bind_stock_cache(sidecar)
    state = _CachedAuxInstallationState(
        provider=provider,
        handoff=handoff,
        compute_cached_row_ids=native_module.compute_cached_row_ids,
        make_cached_sidecar_rows=native_module.make_cached_sidecar_rows,
        drain_cached_completions=native_module.drain_cached_completions,
    )
    auxiliary_stream = mx_module.new_stream(mx_module.gpu)
    previous_tokens = stock_module._fixed_m4_previous_tokens
    stock_aux_type = stock_module._FixedM4SidecarAux

    def build(*args: Any, **kwargs: Any) -> _CachedFixedM4SidecarAux:
        state.ensure_healthy()
        state.drain()
        try:
            stock_aux = original_builder(*args, **kwargs)
            if not isinstance(stock_aux, stock_aux_type):
                raise ValueError(
                    "cached native PLE requires stock materialized M4 aux"
                )
            return _CachedFixedM4SidecarAux(
                stock_aux,
                state=state,
                auxiliary_stream=auxiliary_stream,
                compute_cached_row_ids=state.compute_cached_row_ids,
                make_cached_sidecar_rows=state.make_cached_sidecar_rows,
                previous_tokens=previous_tokens,
                submit_planes=submit_planes,
                submit_embedding=submit_embedding,
                mx_module=mx_module,
                output_dim=contract["output_dim"],
            )
        except BaseException as error:
            state.fail(error)
            raise

    runtime.build_fixed_m4_compiled_verify_aux = build
    return CachedAuxInstallation(provider, auxiliary_stream, state)


def install_fixed_m4_cached_aux_builder(
    runtime: Any,
    *,
    native_module: Any,
    mx_module: Any | None = None,
    stock_module: Any | None = None,
) -> CachedAuxInstallation:
    """Install the asynchronous cached native-PLE builder explicitly."""

    if mx_module is None:
        import mlx.core as mx_module
    return _install_cached_builder(
        runtime,
        native_module=native_module,
        mx_module=mx_module,
        stock_module=stock_module,
        submit_planes=mx_module.async_eval,
        submit_embedding=mx_module.async_eval,
    )


def install_fixed_m4_sync_cached_aux_builder(
    runtime: Any,
    *,
    native_module: Any,
    mx_module: Any | None = None,
    stock_module: Any | None = None,
) -> CachedAuxInstallation:
    """Install the synchronous cached native-plane diagnostic builder."""

    if mx_module is None:
        import mlx.core as mx_module
    return _install_cached_builder(
        runtime,
        native_module=native_module,
        mx_module=mx_module,
        stock_module=stock_module,
        submit_planes=mx_module.eval,
        submit_embedding=mx_module.async_eval,
    )


install_qwen4_fixed_cached_aux = install_fixed_m4_cached_aux_builder
install_qwen4_fixed_sync_cached_aux = install_fixed_m4_sync_cached_aux_builder


__all__ = [
    "CachedAuxInstallation",
    "PENDING_LIMIT",
    "NATIVE_CACHED_PROVIDER_API",
    "install_fixed_m4_cached_aux_builder",
    "install_fixed_m4_sync_cached_aux_builder",
    "install_qwen4_fixed_cached_aux",
    "install_qwen4_fixed_sync_cached_aux",
]
