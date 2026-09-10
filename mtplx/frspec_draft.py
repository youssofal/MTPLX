"""FR-Spec: frequency-ranked pruned DRAFT LM head (target verifies the full vocab).

Port of the Y-PC 3090 Session-2 lever (qwen38-int8-maxtps, 2026-08-15): drafting
from the top-65,536 frequency-ranked vocab rows cut step time 12.9% there while
the target kept verifying over all 248,320 rows. Exactness: probability-ratio
acceptance with residual correction is valid for ANY proposal distribution q —
pruning the draft support can only move the acceptance rate, never the emitted
distribution. The correction path already treats tokens outside the stored
draft top-k as q=0, which subsumes the pruned rows.

Coverage receipts for the ranked list (Y-PC runs/draft_vocab.json, generic
corpus deliberately NOT workload-fit — see the rig's LOSSES.md S2-L9):
0.99487 at n=65536 build-time out-of-sample, 0.99728 measured on real traces;
acceptance-length ceiling cost <=2.2% over 8 draft positions.

Env contract (all default-off):
- ``MTPLX_FRSPEC_DRAFT=1`` enables the pruned draft head.
- ``MTPLX_FRSPEC_VOCAB=<path>`` JSON carrying ``{"ids": [...]}`` ranked
  most-frequent-first (the Y-PC artifact loads unchanged — same tokenizer).
- ``MTPLX_FRSPEC_N`` optional cap for external ranked files. The built-in 64K
  artifact is row-sorted for efficient gathering and therefore only accepts
  its full size.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BUILTIN_VOCABS = {
    "qwen38-code-64k": Path(__file__).with_name("data")
    / "qwen38_code_ranked_64k.npy",
}


def _full_vocab_head(head: Any, ids: Any, vocab_rows: int) -> Any:
    """Return the pruned head with a full-vocabulary output domain."""

    import mlx.core as mx
    import mlx.nn as nn

    class _FullVocabDraftHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = head
            object.__setattr__(self, "_ids", ids)
            object.__setattr__(self, "_vocab_rows", int(vocab_rows))
            # MTPLX_QWEN4_DRAFT_K20_PRESCATTER (default off).  Disarmed, the
            # branch below is one attribute read per draft step and the call
            # returns exactly what it returned before; nothing is retained.
            object.__setattr__(self, "_prescatter_capture", False)
            object.__setattr__(self, "_prescatter_last", None)

        def __call__(self, x: Any) -> Any:
            subset = self.head(x)
            output = mx.full(
                (*subset.shape[:-1], self._vocab_rows),
                -1.0e30,
                dtype=subset.dtype,
            )
            index = mx.broadcast_to(
                self._ids.reshape(
                    (1,) * (subset.ndim - 1) + (int(self._ids.shape[0]),)
                ),
                subset.shape,
            )
            dense = mx.put_along_axis(output, index, subset, axis=-1)
            if self._prescatter_capture:
                # MLX is lazy: `dense` is a graph node, not a buffer.  Stashing
                # the compact row lets the draft reader build its K20 support
                # from `subset` and never evaluate `dense`, so the scatter is
                # built and dropped rather than executed.  Keyed by the
                # returned array's identity so a consumer can prove the row
                # belongs to THIS call.
                object.__setattr__(self, "_prescatter_last", (dense, subset))
            return dense

        def arm_prescatter_capture(self, enabled: bool) -> None:
            """Arm/disarm the pre-scatter row stash (construction-bound)."""

            object.__setattr__(self, "_prescatter_capture", bool(enabled))
            object.__setattr__(self, "_prescatter_last", None)

        def take_prescatter_row(self, dense: Any) -> Any:
            """Consume the compact row stashed for ``dense``; None on a miss.

            Consuming clears the stash, so nothing keeps an unevaluated
            scatter graph alive past the step that produced it, and a second
            read of the same step cannot silently succeed.
            """

            stashed = self._prescatter_last
            if stashed is None or stashed[0] is not dense:
                return None
            object.__setattr__(self, "_prescatter_last", None)
            return stashed[1]

    return _FullVocabDraftHead()


def frspec_enabled() -> bool:
    return (os.environ.get("MTPLX_FRSPEC_DRAFT", "").strip().lower()
            in {"1", "true", "yes", "on"})


def _vocab_path() -> Path | None:
    raw = (os.environ.get("MTPLX_FRSPEC_VOCAB") or "").strip()
    if not raw:
        return None
    if raw.startswith("builtin:"):
        return _BUILTIN_VOCABS.get(raw.removeprefix("builtin:"))
    path = Path(raw).expanduser()
    return path if path.exists() else None


def load_frspec_ids() -> list[int] | None:
    path = _vocab_path()
    if path is None:
        logger.warning("[frspec] MTPLX_FRSPEC_DRAFT set but MTPLX_FRSPEC_VOCAB missing/not found")
        return None
    try:
        if path.suffix == ".npy":
            import numpy as np

            ids = np.load(path, allow_pickle=False).reshape(-1).tolist()
        else:
            payload = json.loads(path.read_text())
            ids = payload.get("ids") if isinstance(payload, dict) else payload
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("[frspec] failed to read %s: %s", path, exc)
        return None
    if not isinstance(ids, list) or not ids:
        logger.warning("[frspec] %s carries no ids list", path)
        return None
    raw_n = (os.environ.get("MTPLX_FRSPEC_N") or "").strip()
    if raw_n:
        try:
            n = int(raw_n)
        except ValueError:
            logger.warning("[frspec] MTPLX_FRSPEC_N must be an integer")
            return None
        if n <= 0 or n > len(ids):
            logger.warning("[frspec] MTPLX_FRSPEC_N must be in [1, %d]", len(ids))
            return None
        raw_vocab = (os.environ.get("MTPLX_FRSPEC_VOCAB") or "").strip()
        if raw_vocab.startswith("builtin:") and n != len(ids):
            logger.warning("[frspec] built-in vocabularies do not support truncation")
            return None
        ids = ids[:n]
    resolved = [int(i) for i in ids]
    if len(set(resolved)) != len(resolved):
        logger.warning("[frspec] %s carries duplicate token ids", path)
        return None
    return resolved


def install_frspec_draft_head(text: Any) -> dict[str, Any]:
    """Install a row-pruned proposal head at the model construction boundary.

    Call after the normal draft-head install. A contract miss returns an
    uninstalled report; the explicitly enabled caller turns that into one
    construction-time failure.
    """
    import mlx.core as mx
    import mlx.nn as nn

    started = time.perf_counter()
    native_head = getattr(text, "_mtplx_native_mtp_draft_head", None)
    head = native_head() if native_head is not None else None
    source = "native_mtp_head" if head is not None else "configured_draft_head"
    if head is None:
        head = getattr(text, "_mtplx_draft_lm_head", None)
    if head is None:
        return {"installed": False, "reason": "no_draft_lm_head"}
    if not isinstance(head, nn.QuantizedLinear):
        return {"installed": False, "reason": f"head_type_{type(head).__name__}"}

    bits = int(head.bits)
    group_size = int(head.group_size)
    mode = str(head.mode)
    # The row-pruned head carries the source head's own bits/group_size/mode
    # (the nn.QuantizedLinear built below), so any affine g64 head prunes
    # exactly -- Q8/g64 (Optimized-Speed) and Q4/g64 (Bare-Speed) both install.
    # Reject every other native-head layout so the model LOAD fails loudly
    # rather than serving a mispruned head.
    if source == "native_mtp_head" and (
        bits not in (4, 8) or group_size != 64 or mode != "affine"
    ):
        return {
            "installed": False,
            "reason": "native_head_contract",
            "bits": bits,
            "group_size": group_size,
            "mode": mode,
        }

    ids = load_frspec_ids()
    if not ids:
        return {"installed": False, "reason": "no_ids"}

    vocab_rows = int(head.weight.shape[0])
    if max(ids) >= vocab_rows or min(ids) < 0:
        return {"installed": False, "reason": "ids_out_of_range", "vocab_rows": vocab_rows}
    n = len(ids)
    if n >= vocab_rows:
        return {"installed": False, "reason": "not_actually_pruned", "n": n}

    ids_arr = mx.array(ids, dtype=mx.int32)
    in_dims = int(head.scales.shape[1]) * group_size

    pruned = nn.QuantizedLinear(
        in_dims,
        n,
        bias="bias" in head,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    pruned.weight = mx.take(head.weight, ids_arr, axis=0)
    pruned.scales = mx.take(head.scales, ids_arr, axis=0)
    if "biases" in head:
        pruned.biases = mx.take(head.biases, ids_arr, axis=0)
    if "bias" in head:
        pruned.bias = mx.take(head.bias, ids_arr, axis=0)
    mx.eval(pruned.parameters())

    full_head = _full_vocab_head(pruned, ids_arr, vocab_rows)
    mx.eval(full_head.parameters())

    # Keep the configured draft head unchanged for legacy consumers. Qwen4's
    # native MTP route binds the full-domain wrapper once below and calls it
    # directly without a per-token eligibility branch.
    text._mtplx_frspec_draft_head = full_head
    text._mtplx_frspec_full_vocab = vocab_rows
    text._mtplx_frspec_ids = ids_arr
    legacy = frspec_legacy_enabled()
    bind_draft_head = getattr(text, "_mtplx_bind_draft_lm_head", None)
    if bind_draft_head is not None:
        bind_draft_head(full_head)
    if legacy:
        # Legacy per-step lane (2026-08-25): the mtp forward projects through
        # _mtplx_draft_lm_head directly, so the swap is global here and
        # generate_mtpk remaps sampled local ids -> full ids at its single
        # draft convergence point (width-guarded).
        text._mtplx_frspec_saved_head = head
        text._mtplx_draft_lm_head = full_head
    report = {
        "installed": True,
        "n": n,
        "vocab_rows": vocab_rows,
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
        "bytes_ratio": round(n / vocab_rows, 4),
        "source": source,
        "output_mode": "full",
        "legacy_swap": bool(legacy),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    logger.info("[frspec] pruned draft lm_head installed: %s", report)
    return report


def frspec_legacy_enabled() -> bool:
    return (os.environ.get("MTPLX_FRSPEC_LEGACY", "").strip().lower()
            in {"1", "true", "yes", "on"})
