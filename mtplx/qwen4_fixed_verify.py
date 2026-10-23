"""Construction-bound fixed-M4 capture surface for Qwen4 Flash-Next."""

from __future__ import annotations

import os
from functools import partial
from types import MethodType
from typing import Any

import mlx.core as mx
import numpy as np

from .models.qwen4_exp import (
    _ngram_rows_np,
    compiled_verify_ple_scope,
    verify_capture_scope,
)

_ROUTE_KERNEL_FP16_ENV = "MTPLX_QWEN4_ROUTE_KERNEL_FP16"
_GDN_ROW_NAMES = ("qkv", "q", "k", "v", "a", "b")
_PLE_ROW_NAMES = ("ple_hidden", "ple_ids")


def is_qwen4_fixed_verify_config(config: dict[str, Any]) -> bool:
    """Identify the one measured Flash-Next fixed-M4 model geometry."""

    text = config.get("text_config")
    if not isinstance(text, dict):
        return False
    observed = (
        config.get("model_type"),
        text.get("model_type"),
        text.get("hidden_size"),
        text.get("num_hidden_layers"),
        text.get("hc_count"),
        text.get("hc_lowrank"),
        text.get("indexer_compress_ratio"),
        text.get("linear_num_key_heads"),
        text.get("linear_num_value_heads"),
        text.get("linear_key_head_dim"),
        text.get("linear_value_head_dim"),
        text.get("ple_layer_ids"),
        text.get("ngram_size"),
        text.get("ngram_vocab_size_base"),
        text.get("heads_per_ngram"),
        text.get("ple_embed_dim"),
        text.get("ngram_sidecar"),
        text.get("num_experts"),
        text.get("num_experts_per_tok"),
        text.get("moe_intermediate_size"),
        text.get("vocab_size"),
    )
    return observed == (
        "qwen4_exp",
        "qwen4_exp_text",
        2560,
        48,
        4,
        320,
        4,
        16,
        48,
        128,
        128,
        [2],
        3,
        20_000_000,
        8,
        2560,
        True,
        512,
        10,
        640,
        248_320,
    )


def qwen4_fixed_verify_enabled() -> bool:
    raw = os.environ.get("MTPLX_QWEN4_FIXED_M4_VERIFY", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _text_model(runtime: Any):
    return getattr(runtime.model, "language_model", runtime.model)


def _inner(runtime: Any):
    return _text_model(runtime).model


def _prepare_compiled_verify_aux(self: Any, input_ids, cache) -> mx.array:
    """Materialize the one host-backed PLE embedding before graph dispatch."""

    inner = _inner(self)
    layer_index = int(inner._ple_stage_idx)
    ple = inner.layers[layer_index].ple
    entry = cache[layer_index]
    previous = entry[ple.NGRAM_IDX]
    embedding = ple.ple_embedding._graph_path(
        input_ids,
        None,
        ple.NGRAM_IDX,
        prev=previous,
    )
    # The SSD sidecar returns a lazy dequantization graph whose inputs are
    # host-acquired row payloads.  It must cross the materialization boundary
    # before becoming an explicit input to mx.compile; otherwise MLX sees the
    # row graph's cache leaves as uncaptured inputs to the verifier graph.
    mx.eval(embedding)
    return embedding


class _FixedM4SidecarAux:
    """Construction-bound host row gather for the physical-M4 verifier."""

    __slots__ = ("_gather", "_output_dim", "_prefetch", "_prompt_tail", "_rows")

    def __init__(self, *, prompt_tail, rows, gather, output_dim, prefetch=None):
        self._prompt_tail = prompt_tail
        self._rows = rows
        self._gather = gather
        self._output_dim = output_dim
        self._prefetch = prefetch

    def prefetch(self, host_input_prefix, completion_tokens, committed_count) -> int:
        """Warm the sidecar's hot rows for the first tokens of the next window.

        A position's rows depend only on that token and the two before it, so
        the rows of a window PREFIX are the first rows of the whole window.
        The draft loop knows the prefix one token at a time while the GPU is
        still drafting; the gather in ``__call__`` then reads warm rows.
        """

        if self._prefetch is None or not host_input_prefix:
            return 0
        ids_np = np.asarray((tuple(host_input_prefix),), dtype=np.int64)
        previous = _fixed_m4_previous_tokens(
            self._prompt_tail,
            completion_tokens,
            committed_count,
        )
        prev_np = np.asarray((previous,), dtype=np.int64)
        rows, _new_history = self._rows(ids_np, prev_np)
        return int(self._prefetch(rows.reshape(-1)))

    def __call__(
        self,
        _input_ids,
        host_input_ids,
        completion_tokens,
        committed_count,
    ) -> mx.array:
        ids_np = np.asarray((host_input_ids,), dtype=np.int64)
        previous = _fixed_m4_previous_tokens(
            self._prompt_tail,
            completion_tokens,
            committed_count,
        )
        prev_np = np.asarray((previous,), dtype=np.int64)
        rows, _new_history = self._rows(ids_np, prev_np)
        # One embedding row per window token: four for a verify round, the
        # block's width for a copy window (graphbank's copy-window replay).
        return self._gather(rows.reshape(-1)).reshape(
            1, int(ids_np.shape[1]), self._output_dim
        )


def _fixed_m4_previous_tokens(
    prompt_tail: tuple[int, int], completion_tokens, committed_count: int
) -> tuple[int, int]:
    """Project the last two cache-owned tokens from the host token ledger."""

    if committed_count >= 2:
        return (
            int(completion_tokens[committed_count - 2]),
            int(completion_tokens[committed_count - 1]),
        )
    if committed_count == 1:
        return int(prompt_tail[1]), int(completion_tokens[0])
    return prompt_tail


def _build_fixed_m4_compiled_verify_aux(self: Any, cache, prompt_ids):
    """Validate and bind the production sidecar gather once after prefill."""

    inner = _inner(self)
    layer_index = int(inner._ple_stage_idx)
    ple = inner.layers[layer_index].ple
    embedding = ple.ple_embedding
    sidecar = embedding.ngram_embedding._sidecar
    entry = cache[layer_index]
    previous = entry[ple.NGRAM_IDX]
    observed = (
        int(embedding.context_len),
        int(embedding.ngram_size),
        int(embedding.heads_per_ngram),
        tuple(previous.shape),
        previous.dtype,
        int(inner.args.ple_embed_dim),
    )
    production = int(inner.args.hidden_size) == 2560
    exact = observed == (2, 3, 8, (1, 2), mx.int64, 2560)
    internally_consistent = (
        observed[0:2] == (2, 3)
        and observed[2] > 0
        and observed[3] == (1, 2)
        and observed[4] == mx.int64
        and observed[5] == int(inner.args.hidden_size)
    )
    if sidecar is None or (not exact if production else not internally_consistent):
        raise ValueError(
            f"qwen4 fixed-M4 sidecar auxiliary geometry mismatch: {observed}"
        )
    mult, sizes, offs = embedding._np_consts()
    const_shapes = tuple(tuple(value.shape) for value in (mult, sizes, offs))
    if const_shapes != ((3,), (2 * observed[2],), (2 * observed[2],)):
        raise ValueError(
            f"qwen4 fixed-M4 sidecar auxiliary constants mismatch: {const_shapes}"
        )
    if production:
        storage = (
            int(sidecar.bits),
            int(sidecar.group_size),
            tuple(
                (name, tuple(sidecar._maps[name][0].shape[1:]), sidecar._maps[name][1])
                for name in ("weight", "scales", "biases")
            ),
        )
        expected_storage = (
            4,
            32,
            (
                ("weight", (20,), "U32"),
                ("scales", (5,), "BF16"),
                ("biases", (5,), "BF16"),
            ),
        )
        if storage != expected_storage:
            raise ValueError(
                f"qwen4 fixed-M4 sidecar auxiliary storage mismatch: {storage}"
            )
    prompt_tail = tuple(
        [int(embedding.eos_id), int(embedding.eos_id)]
        + [int(token) for token in prompt_ids[-2:]]
    )[-2:]
    device_history = tuple(
        int(token)
        for token in np.asarray(previous, dtype=np.int64).reshape(-1)
    )
    if device_history != prompt_tail:
        raise ValueError(
            "qwen4 fixed-M4 prompt history does not match the prefetched cache"
        )
    rows = partial(
        _ngram_rows_np,
        mult=mult,
        sizes=sizes,
        offs=offs,
        eos=int(embedding.eos_id),
        ngram_size=int(embedding.ngram_size),
        heads_per_ngram=int(embedding.heads_per_ngram),
    )
    return _FixedM4SidecarAux(
        prompt_tail=prompt_tail,
        rows=rows,
        gather=sidecar.gather_np,
        output_dim=int(inner.args.ple_embed_dim),
        prefetch=getattr(sidecar, "prefetch_rows_np", None),
    )


def _forward_ar_capture(
    self: Any,
    input_ids,
    *,
    cache=None,
    return_hidden: bool = True,
    hidden_variant: str | None = None,
    capture_backend: str | None = None,
    compiled_aux: mx.array | None = None,
):
    del hidden_variant, capture_backend
    text = _text_model(self)
    if cache is None:
        cache = text.make_cache()
    with verify_capture_scope(), compiled_verify_ple_scope(compiled_aux):
        logits, hidden = text(
            input_ids,
            cache=cache,
            return_hidden=True,
        )

    captures: dict[int, dict[str, mx.array]] = {}
    for index, layer in enumerate(text.model.layers):
        if not layer.is_linear:
            continue
        entry = cache[index]
        rows = getattr(entry, "_mtplx_verify_rows", None)
        if rows is None or len(rows) != len(_GDN_ROW_NAMES):
            raise RuntimeError(
                f"qwen4 fixed-M4 capture missing GDN rows at layer {index}"
            )
        layer_capture = dict(zip(_GDN_ROW_NAMES, rows))
        if getattr(layer, "ple", None) is not None:
            ple_rows = getattr(entry, "_mtplx_verify_ple", None)
            if ple_rows is None or len(ple_rows) != len(_PLE_ROW_NAMES):
                raise RuntimeError(
                    f"qwen4 fixed-M4 capture missing PLE rows at layer {index}"
                )
            layer_capture.update(zip(_PLE_ROW_NAMES, ple_rows))
        captures[index] = layer_capture
    if return_hidden:
        return logits, hidden, captures
    return logits, captures


def _commit_compiled_verify_captures(
    self: Any, cache, captures: dict[int, dict[str, mx.array]]
) -> None:
    """Install materialized capture leaves on the live family cache."""

    for index, layer in enumerate(_inner(self).layers):
        if not layer.is_linear:
            continue
        layer_capture = captures[index]
        cache[index]._mtplx_verify_rows = tuple(
            layer_capture[name] for name in _GDN_ROW_NAMES
        )
        if getattr(layer, "ple", None) is not None:
            cache[index]._mtplx_verify_ple = tuple(
                layer_capture[name] for name in _PLE_ROW_NAMES
            )


def install_qwen4_fixed_verify_route(runtime: Any) -> dict[str, Any]:
    """Validate the Qwen4 state topology once and bind direct fixed-M4 hooks."""

    inner = _inner(runtime)
    layers = tuple(inner.layers)
    linear = tuple(index for index, layer in enumerate(layers) if layer.is_linear)
    qsa = tuple(index for index, layer in enumerate(layers) if not layer.is_linear)
    ple = tuple(
        index for index, layer in enumerate(layers) if getattr(layer, "ple", None)
    )
    if not linear or not qsa or len(ple) != 1 or not layers[ple[0]].is_linear:
        raise ValueError("qwen4 fixed-M4 verifier requires mixed QSA/GDN and one PLE")

    args = inner.args
    if int(args.hidden_size) == 2560:
        observed = (
            len(layers),
            len(linear),
            len(qsa),
            int(args.hc_count),
            int(args.indexer_compress_ratio),
        )
        if observed != (48, 36, 12, 4, 4):
            raise ValueError(f"qwen4 fixed-M4 production geometry mismatch: {observed}")

    extra = []
    for index in linear:
        names = _GDN_ROW_NAMES + (_PLE_ROW_NAMES if index == ple[0] else ())
        extra.append((index, names))

    runtime.forward_ar_capture = MethodType(_forward_ar_capture, runtime)
    runtime.prepare_compiled_verify_aux = MethodType(
        _prepare_compiled_verify_aux, runtime
    )
    ple_embedding = layers[ple[0]].ple.ple_embedding
    if ple_embedding.ngram_embedding._sidecar is not None:
        runtime.build_fixed_m4_compiled_verify_aux = MethodType(
            _build_fixed_m4_compiled_verify_aux, runtime
        )
    runtime.commit_compiled_verify_captures = MethodType(
        _commit_compiled_verify_captures, runtime
    )
    runtime._mtplx_capture_layout = ()
    runtime._mtplx_capture_extra_layout = tuple(extra)
    runtime.qwen4_fixed_m4_compiled_verify = True

    # W70 MTPLX_QWEN4_VERIFY_GLUE: contract-check and bit-exactness-probe the
    # armed glue items here -- model build, outside any mx.compile trace, the
    # same place every other fixed-M4 lane validates itself.  Contract misses
    # raise; exactness misses disable the item and log.  A no-op when the flag
    # is off (the import itself is cheap and has no MLX side effects).
    from mtplx.runtime_options import qwen4_verify_glue_enabled

    if qwen4_verify_glue_enabled():
        from mtplx import qwen4_verify_glue as _glue

        runtime._mtplx_qwen4_verify_glue = _glue.install(
            tuple((index, layers[index].self_attn) for index in qsa),
            rows=4,
        )

    # The hyper-connection read at the verify width: three kernels proved bit
    # for bit against the compiled stock chain on this GPU, here, before the
    # first verify trace. A failed probe leaves the stock chain serving.
    from mtplx.kernels import hc_verify_read

    runtime._mtplx_hc_verify_read = hc_verify_read.install(inner, rows=(4,))

    # The fixed bank's QSA selection around the score GEMM and argpartition:
    # two kernels proved against the compiled stock selector the same way.
    from mtplx.kernels import qsa_verify_select

    runtime._mtplx_qsa_verify_select = qsa_verify_select.install(inner, rows=(4,))

    # The fixed bank's pooled-key row (mean, norm, rotation, select) in one
    # kernel before the stock bank write, proved against the compiled update.
    from mtplx.kernels import qsa_pooled_row

    runtime._mtplx_qsa_pooled_row = qsa_pooled_row.install(inner)

    from mtplx.kernels import qwen4_m4_route
    from mtplx.runtime_options import env_bool

    if env_bool(_ROUTE_KERNEL_FP16_ENV, default=False):
        qwen4_m4_route.install_tail_fp16(inner)

    return {"installed": True, "linear_layers": len(linear), "rows": 4}


__all__ = [
    "is_qwen4_fixed_verify_config",
    "install_qwen4_fixed_verify_route",
    "qwen4_fixed_verify_enabled",
]
