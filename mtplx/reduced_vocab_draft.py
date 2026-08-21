"""FR-Spec (arXiv:2502.14856) reduced-vocabulary draft LM head.

Restricts the draft-only LM head to a frequency-ranked token subset S.
Speculative sampling is exact for any proposal q and the draft head never
feeds the verify/target logits, so the output distribution is unchanged.
The subset is an exact row slice of weight/scales/biases with no
re-quantization, so logits on S are bit-identical to the full head's.

``MTPLX_DRAFT_VOCAB_IDS`` (path to an ``.npy`` integer array of ids) is what
activates the feature.  ``MTPLX_DRAFT_VOCAB_MODE`` selects ``full``
(default: full-vocabulary-shaped row with a sentinel outside S, drop-in) or
``compact`` (``|S|``-shaped rows plus sampler shims).
``MTPLX_DRAFT_VOCAB_NEG`` picks the sentinel, ``MTPLX_DRAFT_VOCAB_KEEP_BASE``
keeps the full head alive for A/B tests instead of freeing it (~0.7 GB), and
``MTPLX_DRAFT_VOCAB_DEBUG`` prints a one-line install report.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

__all__ = [
    "ReducedVocabDraftHead",
    "build_reduced_head",
    "load_subset_ids",
    "slice_head_rows",
    "maybe_wrap_draft_head",
    "install_compact_shims",
    "env_config",
]

_TRUE = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def env_config() -> dict[str, Any]:
    """Resolve the reduced-vocab configuration from the environment."""
    ids_path = (os.environ.get("MTPLX_DRAFT_VOCAB_IDS") or "").strip()
    mode = (os.environ.get("MTPLX_DRAFT_VOCAB_MODE") or "full").strip().lower()
    if mode not in {"full", "compact"}:
        raise ValueError(
            f"MTPLX_DRAFT_VOCAB_MODE must be 'full' or 'compact', got {mode!r}"
        )
    return {
        "enabled": bool(ids_path),
        "ids_path": ids_path,
        "mode": mode,
        "neg": (os.environ.get("MTPLX_DRAFT_VOCAB_NEG") or "finite").strip().lower(),
        "keep_base": _env_flag("MTPLX_DRAFT_VOCAB_KEEP_BASE", False),
        "debug": _env_flag("MTPLX_DRAFT_VOCAB_DEBUG", False),
    }


def load_subset_ids(path: str, *, vocab_size: int | None = None) -> np.ndarray:
    """Load, validate, de-duplicate and sort a token-id subset from ``.npy``."""
    ids = np.load(path)
    ids = np.asarray(ids).reshape(-1)
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{path}: subset ids must be an integer array, got {ids.dtype}")
    ids = np.unique(ids.astype(np.int64))
    if ids.size == 0:
        raise ValueError(f"{path}: empty token-id subset")
    if int(ids[0]) < 0:
        raise ValueError(f"{path}: negative token id {int(ids[0])}")
    if vocab_size is not None and int(ids[-1]) >= int(vocab_size):
        raise ValueError(
            f"{path}: token id {int(ids[-1])} >= vocab_size {int(vocab_size)}"
        )
    return ids.astype(np.int32)


def _neg_sentinel(dtype: Any, spec: str) -> float:
    """Masking value that survives the samplers' ``* (1 / temperature)`` scaling."""
    import mlx.core as mx

    if spec == "inf":
        return -float("inf")
    if spec != "finite":
        try:
            return float(spec)
        except ValueError as exc:  # pragma: no cover - operator error
            raise ValueError(
                "MTPLX_DRAFT_VOCAB_NEG must be 'finite', 'inf' or a float"
            ) from exc
    if dtype == mx.float16:
        return -6.0e4          # fp16 max magnitude is 65504
    return -1.0e30


def _is_array(value: Any) -> bool:
    import mlx.core as mx

    return isinstance(value, mx.array)


def slice_head_rows(base: Any, ids: Any, *, vocab_size: int) -> Any:
    """Copy of ``base`` keeping only output rows ``ids``.  ``base`` is not mutated."""
    import mlx.core as mx
    import mlx.nn as nn

    cls = type(base)
    sub = cls.__new__(cls)
    nn.Module.__init__(sub)

    for key, value in base.items():
        if _is_array(value):
            if value.ndim >= 1 and int(value.shape[0]) == int(vocab_size):
                sub[key] = value[ids]
            else:
                sub[key] = value
        elif isinstance(value, nn.Module):
            sub[key] = slice_head_rows(value, ids, vocab_size=vocab_size)
        else:
            sub[key] = value

    for key, value in vars(base).items():
        if key in {"_no_grad", "_training"}:
            continue
        try:
            object.__setattr__(sub, key, value)
        except Exception:  # pragma: no cover - exotic descriptors
            pass

    try:
        sub.freeze(recurse=True)
    except Exception:  # pragma: no cover - older mlx
        pass
    mx.eval(sub.parameters())
    return sub


def _row_count(module: Any, *, vocab_size: int) -> int:
    """Output-row count of a sliced module, or -1 when no weight is found."""
    import mlx.nn as nn

    weight = None
    if "weight" in module:
        weight = module["weight"]
    else:
        for value in module.values():
            if isinstance(value, nn.Module) and "weight" in value:
                weight = value["weight"]
                break
    if weight is None:
        return -1
    return int(weight.shape[0])


def _make_head_class():
    """Build the head class lazily so importing this file needs no MLX."""
    import mlx.core as mx
    import mlx.nn as nn

    _HAS_PUT_ALONG = hasattr(mx, "put_along_axis")

    class ReducedVocabDraftHead(nn.Module):
        """Draft LM head restricted to a fixed token subset S.

        ``mode="full"`` returns ``(..., V)`` logits with a sentinel outside S,
        a drop-in for every MTPLX draft consumer.  ``mode="compact"`` returns
        ``(..., |S|)`` and requires :func:`install_compact_shims`.
        """

        def __init__(
            self,
            head: Any,
            ids: np.ndarray,
            *,
            vocab_size: int,
            mode: str = "full",
            neg: str = "finite",
            base: Any = None,
        ):
            super().__init__()
            self.head = head
            object.__setattr__(self, "_ids_np", np.asarray(ids, dtype=np.int64))
            object.__setattr__(self, "_ids", mx.array(np.asarray(ids, dtype=np.int32)))
            object.__setattr__(self, "_vocab_size", int(vocab_size))
            object.__setattr__(self, "_subset_size", int(len(ids)))
            object.__setattr__(self, "_mode", str(mode))
            object.__setattr__(self, "_neg_spec", str(neg))
            object.__setattr__(self, "_base", base)
            object.__setattr__(self, "_neg_cache", {})
            object.__setattr__(self, "_idx_cache", {})
            mx.eval(self._ids)

        @property
        def vocab_size(self) -> int:
            return self._vocab_size

        @property
        def subset_size(self) -> int:
            return self._subset_size

        @property
        def mode(self) -> str:
            return self._mode

        @property
        def subset_ids(self) -> np.ndarray:
            return self._ids_np

        # forwarded so server.openai._draft_head_identity fingerprints two
        # different subsets differently
        @property
        def weight(self):
            return self.head.weight

        @property
        def scales(self):
            return self.head.scales

        @property
        def biases(self):
            return self.head.biases

        @property
        def bias(self):
            return self.head.bias

        def _sentinel(self, dtype):
            cache = self._neg_cache
            value = cache.get(dtype)
            if value is None:
                value = _neg_sentinel(dtype, self._neg_spec)
                cache[dtype] = value
            return value

        def _scatter_index(self, shape):
            cache = self._idx_cache
            idx = cache.get(shape)
            if idx is None:
                idx = mx.broadcast_to(
                    self._ids.reshape((1,) * (len(shape) - 1) + (self._subset_size,)),
                    shape,
                )
                cache[shape] = idx
            return idx

        def __call__(self, x):
            y = self.head(x)
            if self._mode == "compact":
                return y
            full_shape = tuple(y.shape[:-1]) + (self._vocab_size,)
            out = mx.full(full_shape, self._sentinel(y.dtype), dtype=y.dtype)
            if _HAS_PUT_ALONG:
                idx = self._scatter_index(tuple(y.shape))
                return mx.put_along_axis(out, idx, y.astype(out.dtype), axis=-1)
            out[..., self._ids] = y.astype(out.dtype)
            return out

        def _extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"subset={self._subset_size}, vocab={self._vocab_size}, "
                f"mode={self._mode}"
            )

    return ReducedVocabDraftHead


_HEAD_CLASS: Any = None


def _head_class():
    global _HEAD_CLASS
    if _HEAD_CLASS is None:
        _HEAD_CLASS = _make_head_class()
    return _HEAD_CLASS


def __getattr__(name: str):
    if name == "ReducedVocabDraftHead":
        return _head_class()
    raise AttributeError(name)


def build_reduced_head(
    base: Any,
    ids: Any,
    *,
    mode: str = "full",
    neg: str = "finite",
    keep_base: bool = False,
) -> Any:
    """Slice ``base`` down to rows ``ids`` and wrap it in a reduced head."""
    import mlx.core as mx

    vocab_size = _base_vocab_size(base)
    if isinstance(ids, str):
        ids_np = load_subset_ids(ids, vocab_size=vocab_size)
    else:
        ids_np = np.unique(np.asarray(ids).reshape(-1).astype(np.int64)).astype(np.int32)
    if int(ids_np[-1]) >= vocab_size:
        raise ValueError(
            f"token id {int(ids_np[-1])} out of range for vocab_size {vocab_size}"
        )
    idx = mx.array(ids_np.astype(np.int32))
    sliced = slice_head_rows(base, idx, vocab_size=vocab_size)
    rows = _row_count(sliced, vocab_size=vocab_size)
    if rows not in (-1, int(ids_np.size)):
        raise RuntimeError(
            f"row slice produced {rows} rows, expected {int(ids_np.size)}"
        )
    head_cls = _head_class()
    return head_cls(
        sliced,
        ids_np,
        vocab_size=vocab_size,
        mode=mode,
        neg=neg,
        base=base if keep_base else None,
    )


def _base_vocab_size(base: Any) -> int:
    import mlx.nn as nn

    if "weight" in base:
        return int(base["weight"].shape[0])
    for value in base.values():
        if isinstance(value, nn.Module) and "weight" in value:
            return int(value["weight"].shape[0])
    raise TypeError(f"cannot determine vocab size of draft head {type(base)!r}")


def maybe_wrap_draft_head(text_model: Any, base: Any) -> Any:
    """Return a reduced head when the env asks for one, else ``base``.

    Safe to call unconditionally: returns ``base`` untouched when
    ``MTPLX_DRAFT_VOCAB_IDS`` is unset or on any recoverable failure.
    """
    cfg = env_config()
    if not cfg["enabled"] or base is None:
        return base
    try:
        vocab_size = _base_vocab_size(base)
        ids = load_subset_ids(cfg["ids_path"], vocab_size=vocab_size)
        head = build_reduced_head(
            base,
            ids,
            mode=cfg["mode"],
            neg=cfg["neg"],
            keep_base=cfg["keep_base"],
        )
    except Exception as exc:
        print(
            f"[reduced-vocab] DISABLED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return base
    if cfg["mode"] == "compact":
        install_compact_shims(head)
    if text_model is not None:
        try:
            object.__setattr__(text_model, "_mtplx_reduced_vocab", {
                "ids_path": cfg["ids_path"],
                "subset_size": int(head.subset_size),
                "vocab_size": int(head.vocab_size),
                "mode": cfg["mode"],
                "neg": cfg["neg"],
            })
        except Exception:
            pass
    print(
        "[reduced-vocab] draft head restricted to "
        f"{head.subset_size}/{head.vocab_size} ids "
        f"({100.0 * head.subset_size / head.vocab_size:.1f}%), mode={cfg['mode']}, "
        f"neg={cfg['neg']}, ids={cfg['ids_path']}",
        file=sys.stderr,
        flush=True,
    )
    return head


_COMPACT_INSTALLED = False


def compact_shims_installed() -> bool:
    """Whether :func:`install_compact_shims` has patched the host draft samplers."""

    return bool(_COMPACT_INSTALLED)


def install_compact_shims(head: Any) -> None:
    """Patch the MTPLX host draft samplers for ``mode="compact"``.

    Compact rows make every sampler output a subset index rather than a
    vocabulary id.  Only the stock host draft path is remapped; the compiled
    device draft cores and the adaptive-width / reranker / a3b-prefix readers
    are hard-disabled because they consume those raw indices directly.
    """
    global _COMPACT_INSTALLED
    if _COMPACT_INSTALLED:
        return

    import mtplx.generation as gen
    from mtplx.sampling import SparseDistribution

    ids_np = np.asarray(head.subset_ids, dtype=np.int64)
    vocab_size = int(head.vocab_size)

    original_sample = gen._sample_draft_from_logits

    def _sample_draft_from_logits(logits, config, rng, *, need_distribution):
        token, dist = original_sample(
            logits, config, rng, need_distribution=need_distribution
        )
        token = int(ids_np[int(token)])
        if isinstance(dist, SparseDistribution):
            dist = SparseDistribution(
                ids_np[np.asarray(dist.token_ids, dtype=np.int64)],
                np.asarray(dist.probs, dtype=np.float64),
                vocab_size,
            )
        elif dist is not None:
            dense = np.zeros(vocab_size, dtype=np.float64)
            dense[ids_np] = np.asarray(dist, dtype=np.float64)
            dist = dense
        return token, dist

    def _unsupported(*_args, **_kwargs):
        raise RuntimeError(
            "MTPLX_DRAFT_VOCAB_MODE=compact supports only the stock host "
            "draft core; use MTPLX_DRAFT_VOCAB_MODE=full for the device "
            "draft cores, adaptive-width readers and a3b prefix route."
        )

    gen._sample_draft_from_logits = _sample_draft_from_logits
    gen._greedy_draft_token_and_top2 = _unsupported
    gen._make_device_draft_core = _unsupported
    gen._make_device_d2_draft_core = _unsupported
    gen.sample_token_ids_from_mlx_logits = _unsupported
    _COMPACT_INSTALLED = True
    print(
        "[reduced-vocab] compact-mode sampler shims installed "
        "(device draft cores disabled)",
        file=sys.stderr,
        flush=True,
    )
