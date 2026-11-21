"""Machine-aware memory planning: commitments vs caches.

The one place that answers "what fits on this Mac". Issue #305 (2026-08)
showed the cost of not having it: five subsystems (Metal caps, session-bank
budget, per-session snapshot cap, dense-decode ceiling, context window)
each budgeted independently, nothing summed them, and a 48 GB Mac serving
the flagship walked itself to 61.8 GB resident ("129%"), deep swap, and
3-4 tok/s — while a 128 GB Mac with the identical config was fine.

The model:

* **Commitments** cannot be reclaimed without killing a request: model
  weights, and KV for the tokens a session has actually accumulated
  (worst case: the full context window).
* **Caches** are reclaimable in milliseconds: the session-bank RAM tier
  and the MLX allocator pool. They are allowed to take *all* the memory
  commitments don't currently need — precisely because the pressure
  guard (server-side) can seize it back the moment live KV grows.

So the plan is deliberately aggressive-with-a-net, not conservative:
zero throttling and a full-size warm cache while memory is free, and a
dynamic bank ceiling that yields (to the SSD tier, not to oblivion) as a
long-context request's KV actually materializes. Speed on every RAM
tier, no swap-death on any.

Pure arithmetic only. No MLX import, no subprocess, no mtplx imports —
``engine_session`` and the server both import this module, and the
numbers must be computable in tests without a model or a GPU.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

GIB = 1024**3

# The server's Metal memory-limit default (_apply_metal_memory_caps reads
# usable_engine_bytes): min(ram, max(8 GiB, 75% of ram), 192 GiB), and from
# 128 GB up at most ram - 38 GiB. Everything the engine allocates
# through MLX (weights, KV, bank snapshots, transients) lives under that
# allocator bound, so it is the honest "usable" envelope; macOS, other apps,
# and the Python side of the process live in the remaining 25%.
ENGINE_RAM_FRACTION = 0.75
ENGINE_RAM_FLOOR_BYTES = 8 * GIB
ENGINE_RAM_CAP_BYTES = 192 * GIB
# A desktop keeps other apps open beside the engine. On a 128 GB Mac with
# 16 GB of other apps, the 75% rule (a 96 GiB limit) left too little outside
# the engine for a compaction-size prefill: the guard kept the Mac alive by
# refusing it with a 507, while at 90 GiB the same request was served with the
# Mac safe (09-27 final-guard receipts, founder decision G5 on 09-29). From
# 128 GB up the default therefore leaves at least this much outside the
# allocator; at 192 GB and up the 75% rule already leaves more, and Macs under
# 128 GB keep the 75% rule until a desktop receipt says otherwise. A headless
# server chooses ``--memory-limit max`` instead (issue #548).
DESKTOP_HEADROOM_BYTES = 38 * GIB
DESKTOP_HEADROOM_MIN_RAM_BYTES = 128 * GIB

# Decode/prefill working memory that is neither weights, KV, nor bank:
# graphbank compiled buffers, logits_keep tail, draft-head activations,
# chunked-prefill scratch, allocator fragmentation. Anchored on the
# Sustained-mode receipt (27.5 GB peak at 32k on the 128 GB M5 Max:
# ~15.9 GiB weights + ~2.1 GiB KV + a few GiB bank leaves ~3 GiB residual).
RUNTIME_TRANSIENTS_BYTES = 3 * GIB
# Cap on the observed-spike reserve (transient_reserve_bytes): one
# pathological turn must not shrink the bank to its floor forever.
TRANSIENT_RESERVE_CAP_BYTES = 16 * GIB

# Session-bank clamps. The 48 GiB cap is the founder ruling of 2026-07-05
# (after the 55 GB in-flight climb on a 128 GB Mac); the 1 GiB floor is the
# point below which the bank is pure churn. Kept numerically identical to
# engine_session's legacy constants on purpose — the 128 GB resolution must
# not move by a byte.
BANK_FLOOR_BYTES = 1 * GIB
BANK_CAP_BYTES = 48 * GIB

# Margin over RUNTIME_TRANSIENTS_BYTES that the tight-machine admission
# (below) charges instead of the bank floor. Measured 2026-09-21 on the
# Bonsai 2 27B pack (8.23 GiB weights, MTP head and vision weights resident)
# under the 16, 18 and 24 GiB class budgets: the text-decode peak sat 3.056
# to 3.075 GiB above weights plus dense KV in every completed row (4K, 8K
# and 16K prompts, with and without a 1K decode), i.e. the 3 GiB constant
# plus at most 77 MiB. 256 MiB is a bit over three times that excess. One
# pack's text decode is the whole evidence, so the admission is offered only
# to packs no heavier than that one, or to a pack that stamps its own
# measurement (``measured_on_tight_machine``).
TIGHT_MACHINE_MARGIN_BYTES = 256 * 1024 * 1024
# Bonsai 2 27B's resident weights (model + MTP files), the measured pack.
TIGHT_MACHINE_MAX_WEIGHTS_BYTES = 8_834_412_216

# Context windows resolve on 4096-token blocks (the cold tier's restore
# granularity); the floor is one block. Below one block the model simply
# does not fit the machine and the plan says so instead of inventing a
# window.
CONTEXT_ALIGN_TOKENS = 4096
CONTEXT_FLOOR_TOKENS = 4096

# The Qwen3.8-Flash-Next n-gram sidecar (ngram-table.safetensors, ~30 GiB)
# streams from SSD on every Mac: row gathers read numpy memmaps over the
# file, so only touched pages ever become resident, they are clean
# file-backed pages, and macOS reclaims them under pressure. NOT a
# commitment. Counting it as weights anyway was a real shipped-defect chain
# (2026-08-28): a 128G Mac planned ~99G of "weights", printed MODEL DOES NOT
# FIT, and shrank both the context window and the session bank by 30G —
# while the engine actually served fine at ~69G resident.
NGRAM_TABLE_FILENAME = "ngram-table.safetensors"

# Dense-attention KV bytes per token for the flagship geometry
# (16 full-attention layers x K+V x 4 KV heads x head_dim 256 x bf16
# = 65,536). generation._dense_decode_max_context reads the same default
# from MTPLX_DENSE_KV_BYTES_PER_TOKEN; callers pass the resolved value so
# the two never drift.
DEFAULT_DENSE_KV_BYTES_PER_TOKEN = 65536

# Paged-KV quantization shrinks stored KV bytes. Group-64 affine packing:
# q8 = (64 x 1 B + scale/zero) / (64 x 2 B) ~ 0.53, q4 ~ 0.28. Planned a
# shade high so quant metadata and partial-group padding never flip a
# fits/doesn't verdict optimistically.
KV_QUANT_BYTE_FACTOR = {"off": 1.0, "q8": 0.55, "q4": 0.30}


def dense_kv_bytes_per_token_from_config(config: dict | None) -> int | None:
    """Derive dense-attention KV bytes/token from a model config.

    n_full_attention_layers x (K+V) x kv_heads x head_dim x 2 bytes (bf16).
    Reproduces the flagship constant exactly (16 x 2 x 4 x 256 x 2 = 65,536)
    and generalizes to hybrids with different attention geometry — e.g.
    qwen4_exp's 12 x 2 x 2 x 256 x 2 = 24,576, which the flat default would
    over-plan 2.7x. Returns None when the config doesn't say enough.
    """
    if not isinstance(config, dict):
        return None
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    if config.get("model_type") == "inkling2_loop":
        from .models.inkling.layout import cache_footprint

        try:
            return cache_footprint(text).full_bytes_per_token
        except (KeyError, TypeError, ValueError):
            return None
    layer_types = text.get("layer_types")
    n_full: int | None = None
    if isinstance(layer_types, list) and layer_types:
        n_full = sum(1 for t in layer_types if t != "linear_attention")
    else:
        n_layers = text.get("num_hidden_layers")
        interval = text.get("full_attention_interval")
        if isinstance(n_layers, int) and isinstance(interval, int) and interval > 0:
            n_full = n_layers // interval
        elif isinstance(n_layers, int):
            n_full = n_layers  # pure-attention families
    kv_heads = text.get("num_key_value_heads")
    head_dim = text.get("head_dim")
    if head_dim is None and isinstance(text.get("hidden_size"), int) and isinstance(
        text.get("num_attention_heads"), int
    ) and text["num_attention_heads"] > 0:
        head_dim = text["hidden_size"] // text["num_attention_heads"]
    if not (
        isinstance(n_full, int)
        and n_full > 0
        and isinstance(kv_heads, int)
        and kv_heads > 0
        and isinstance(head_dim, int)
        and head_dim > 0
    ):
        return None
    return n_full * 2 * kv_heads * head_dim * 2


# QSA prefill transient model (issue #393): the indexer's dense-mask lane
# materializes ~12.75 bytes per (chunk_row x context_token) per QSA layer —
# fp32 scores [S, H, nb] + relu twin + the [S, nb] fp32 chain + argpartition
# + five [S, T] bool masks (derived at the file level, 2026-08-29 audit).
# Lazy evaluation keeps roughly QSA_TRANSIENT_LIVE_LAYERS layers' worth of
# those intermediates simultaneously live at the peak; 4 reproduces the
# observed #393 blowup (119.2 GB peak on a 262K admit whose resident terms
# sum to ~92 GB -> ~27 GB transient ~= 12.75 x 2048 x 262144 x 4) and is
# deliberately NOT tuned tighter — the estimator must model the mechanism,
# not pad a constant until one repro fits (AGENTS.md: no bug-masking).
QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM = 12.75
QSA_TRANSIENT_LIVE_LAYERS = 4


def _qsa_geometry(config: dict | None) -> tuple[int, int, int, int, int] | None:
    """(n_qsa_layers, indexer_head_dim, compress_ratio, kv_heads, head_dim)
    for QSA hybrids (qwen4_exp), else None."""
    if not isinstance(config, dict):
        return None
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    idx_heads = text.get("indexer_n_heads")
    if not (isinstance(idx_heads, int) and idx_heads > 0):
        return None
    layer_types = text.get("layer_types")
    if not (isinstance(layer_types, list) and layer_types):
        return None
    n_qsa = sum(1 for t in layer_types if t != "linear_attention")
    idx_dim = int(text.get("indexer_head_dim") or 128)
    ratio = max(1, int(text.get("indexer_compress_ratio") or 4))
    kv_heads = int(text.get("num_key_value_heads") or 0)
    head_dim = int(text.get("head_dim") or 0)
    if n_qsa <= 0 or kv_heads <= 0 or head_dim <= 0:
        return None
    return n_qsa, idx_dim, ratio, kv_heads, head_dim


def qsa_aux_bytes_per_token_from_config(config: dict | None) -> int:
    """Per-token QSA bookkeeping the KV term does not cover (issue #393).

    Per QSA layer: raw indexer keys (idx_dim x bf16) + pooled block keys
    (idx_dim x bf16 / ratio) + the fp32-transposed pooled mirror
    (idx_dim x fp32 / ratio). One extra QSA cache serves the MTP head, whose
    full-length KV (2 x kv_heads x head_dim x bf16) the layer_types-derived
    KV term also misses. Zero for non-QSA families.
    """
    geo = _qsa_geometry(config)
    if geo is None:
        return 0
    n_qsa, idx_dim, ratio, kv_heads, head_dim = geo
    per_layer = idx_dim * 2 + (idx_dim * 2) // ratio + (idx_dim * 4) // ratio
    mtp_head_kv = 2 * kv_heads * head_dim * 2
    return (n_qsa + 1) * per_layer + mtp_head_kv


def mtp_history_bytes_per_token_from_config(config: dict | None) -> int:
    """KV the MTP head keeps per committed token under the ``committed``
    MTP-history policy, for families whose aux term does not already count
    it: the Qwen 3.5/3.6/3.8 hybrids' MTP layers are full attention with the
    trunk's KV heads and head size, bf16. 4,096 B a token on the 27B and the
    4B (one layer, 4 KV heads x 256). Zero for QSA hybrids (their aux term
    counts the MTP head) and for configs that do not declare
    ``mtp_num_hidden_layers``."""

    if not isinstance(config, dict) or _qsa_geometry(config) is not None:
        return 0
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    try:
        layers = int(text.get("mtp_num_hidden_layers") or config.get("mtp_num_hidden_layers") or 0)
        kv_heads = int(text.get("num_key_value_heads") or 0)
        head_dim = int(text.get("head_dim") or 0)
    except (TypeError, ValueError):
        return 0
    if layers <= 0 or kv_heads <= 0 or head_dim <= 0:
        return 0
    return layers * 2 * kv_heads * head_dim * 2


def qsa_prefill_transient_bytes_per_token_from_config(
    config: dict | None, *, chunk_size: int = 2048
) -> int:
    """Peak prefill transient per context token for QSA hybrids, else 0.

    Linear in context because the last chunk's indexer intermediates scale
    with the full token count; see QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM.
    """
    if _qsa_geometry(config) is None:
        return 0
    return int(
        QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM
        * max(1, int(chunk_size))
        * QSA_TRANSIENT_LIVE_LAYERS
    )


def detect_total_ram_bytes() -> int | None:
    """Physical RAM, PATH-immune.

    The app-owned daemon launches with a sanitized PATH that broke every
    bare ``sysctl`` subprocess once already (the auto bank budget silently
    fell back to the flat 24G default — mistakes ledger, 2026-07). ctypes
    ``sysctlbyname`` has no PATH to sanitize.
    """
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            value = ctypes.c_uint64(0)
            size = ctypes.c_size_t(ctypes.sizeof(value))
            rc = libc.sysctlbyname(
                b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0
            )
            if rc == 0 and value.value > 0:
                return int(value.value)
        except Exception:
            pass
        return None
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if page > 0 and pages > 0:
            return int(page) * int(pages)
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _ram_share_bytes(total_ram_bytes: int) -> int:
    """The 75% rule: min(ram, max(8 GiB, 75% of ram), 192 GiB)."""
    total = int(total_ram_bytes)
    return min(
        total,
        max(ENGINE_RAM_FLOOR_BYTES, int(total * ENGINE_RAM_FRACTION)),
        ENGINE_RAM_CAP_BYTES,
    )


def usable_engine_bytes(total_ram_bytes: int) -> int:
    """The engine's default allocator envelope for a machine of this size."""
    total = int(total_ram_bytes)
    usable = _ram_share_bytes(total)
    if total >= DESKTOP_HEADROOM_MIN_RAM_BYTES:
        usable = min(usable, total - DESKTOP_HEADROOM_BYTES)
    return usable


def max_engine_bytes(total_ram_bytes: int) -> int:
    """``--memory-limit max``: everything outside macOS's own reserve, and
    never less than the default.

    For a headless server with no desktop to leave room for (issue #548):
    112 GiB on a 128 GB Mac, 56 GiB on 64 GB. On Macs of 32 GB and under the
    default (75% of RAM, at least 8 GiB) already reaches past the reserve
    (8 GiB there), so ``max`` is the default: 8, 12, 18 and 24 GiB on 8, 16,
    24 and 32 GB. The guard's whole-Mac floors still apply; only the
    engine's own budget grows.
    """
    total = int(total_ram_bytes)
    return max(usable_engine_bytes(total), total - system_reserve_bytes(total))


def system_reserve_bytes(total_ram_bytes: int) -> int:
    """Unwired headroom macOS keeps outside the allocator caps.

    16 GiB is right on 128 GB+ machines, but as a flat constant it refused
    the 96 GB class by exactly the release notes' own margin (issue #400:
    77.3 GiB weights + 6 GiB floor margin + 16 GiB = 99.3 > 96, while the
    same pack ships healthy there with about 16 GiB left unwired). Scale to
    RAM/8 below 128 GB, floored at 8 GiB, and keep 128 GB+ unchanged.

    Lives here (pure arithmetic) so the Metal caps, the planner and the
    planner replay in the tests read one rule.
    """
    return int(min(16 * GIB, max(8 * GIB, int(total_ram_bytes) // 8)))


def engine_envelope_bytes(
    total_ram_bytes: int, *, resident_floor_bytes: int | None = None
) -> int:
    """The allocator envelope for this machine AND this model.

    The 75% rule for every model whose weights sit under it. A family that
    declares a resident floor above that rule (Flash-Next on a 96 GB Mac:
    77.3 GiB of weights + margin against a 72 GiB envelope) is admitted by
    the Metal caps on the ground that floor + system reserve fits the
    machine. The same ground defines its envelope: everything outside the
    system reserve. Before this rule the caps raised the limit to the bare
    floor (80.3 GiB) while the planner kept budgeting 72 GiB, so the
    catalog offered the pack, the planner printed MODEL DOES NOT FIT, and
    the fixed-M4 gate (0.97 of a limit that equals the weights) skipped the
    compiled verifier on nearly every request.

    Never below the floor; seats whose 75% envelope already covers the
    floor (128 GB and up for Flash-Next) are unchanged.

    The desktop headroom (``usable_engine_bytes`` from 128 GB up) gives way
    to a model whose floor is above it only as far as the 75% rule, the
    default before that headroom: a 128 GB Mac and a 92 GiB floor get
    96 GiB, as they did, not the whole envelope outside the system reserve
    (112 GiB; the review of 4c9da1ba found the lower default turning that
    floor into 112 GiB).
    """
    total = int(total_ram_bytes)
    base = usable_engine_bytes(total)
    floor = max(0, int(resident_floor_bytes or 0))
    if floor <= base:
        return base
    share = _ram_share_bytes(total)
    if floor <= share:
        return share
    return min(total, max(floor, total - system_reserve_bytes(total)))


def measured_on_tight_machine(contract: Any, model_weights_bytes: int | None) -> bool:
    """Whether a pack's runtime contract carries its own tight-machine peaks.

    ``memory_evidence`` is the measured memory table the pack builder stamps
    (scripts/bonsai_memory_table.py; today only the Bonsai 2 pack has one).
    It counts when it measured these exact weights and holds a row the
    tight-machine rule admitted that completed inside its engine budget.
    """
    evidence = contract.get("memory_evidence") if isinstance(contract, dict) else None
    if not isinstance(evidence, dict):
        return False
    pack = evidence.get("pack")
    if not isinstance(pack, dict) or pack.get("weights_bytes") != model_weights_bytes:
        return False
    return any(
        isinstance(row, dict)
        and row.get("completed") is True
        and row.get("within_engine_budget") is True
        and isinstance(row.get("planner"), dict)
        and row["planner"].get("tight_machine") is True
        for row in evidence.get("rows") or ()
    )


def _align_down(tokens: int) -> int:
    return (int(tokens) // CONTEXT_ALIGN_TOKENS) * CONTEXT_ALIGN_TOKENS


@dataclass(frozen=True)
class MemoryPlan:
    """Resolved memory geometry for one (machine, model, config) triple.

    ``available`` is False when an input could not be detected; callers
    keep their legacy behavior in that case and MUST log the reason (a
    detector whose failure path is "quietly use the default" is how the
    app-daemon PATH bug shipped).
    """

    available: bool
    unavailable_reason: str | None = None

    total_ram_bytes: int = 0
    memory_budget_bytes: int | None = None
    usable_bytes: int = 0
    model_weights_bytes: int = 0
    # Flash-Next n-gram sidecar bytes, streamed from SSD (0 for every other
    # model). Carried so the banner/app can say where the other ~30G of the
    # pack lives instead of the number silently not adding up to the disk size.
    ngram_table_streamed_bytes: int = 0
    kv_bytes_per_token: int = DEFAULT_DENSE_KV_BYTES_PER_TOKEN
    kv_quantization: str = "off"
    kv_bytes_per_token_effective: int = DEFAULT_DENSE_KV_BYTES_PER_TOKEN
    # Family working set the KV term does not cover (QSA raw/pooled streams,
    # MTP-head KV) and the peak prefill transient per token (the #393 terms).
    # Zero for families without them — the fit then matches the legacy solve.
    aux_bytes_per_token: int = 0
    fixed_cache_bytes: int = 0
    prefill_transient_bytes_per_token: int = 0
    # The committed MTP-history cache per token of a non-QSA MTP family
    # (mtp_history_bytes_per_token_from_config). The request admission
    # prices it; the fit and the bank budgets do not count it yet.
    mtp_history_bytes_per_token: int = 0

    model_fits: bool = True
    # Largest window the machine can commit to (weights + full-window KV +
    # transients + bank floor all resident), before the model's own cap.
    context_window_fit: int = 0
    # What the server should actually serve: requested override if given,
    # else min(fit, model max).
    context_window_resolved: int = 0
    # True when the MACHINE (not the model) bound the resolved window.
    context_machine_bound: bool = False
    # True when an explicit request exceeds what fits; the guard becomes
    # the safety net and the caller must warn loudly.
    context_overcommitted: bool = False

    # Steady-state KV projection the bank budget must coexist with: KV at
    # the dense-decode ceiling (sessions past it ride the paged lane and
    # are the dynamic ceiling's job), never more than the resolved window.
    kv_reserve_tokens: int = 0
    kv_reserve_bytes: int = 0

    # Bank budget when live KV is idle (the configured max) and at the
    # steady-state projection (what the banner/app should quote as the
    # under-load value). The dynamic ceiling moves between them at runtime.
    bank_idle_max_bytes: int = BANK_FLOOR_BYTES
    bank_steady_bytes: int = BANK_FLOOR_BYTES

    # The runtime transient and the bank floor this plan was solved with.
    # Both equal the module constants on every machine that funds the
    # bank floor with at least the tight rule's largest window (the 128 GB
    # resolution does not move by a byte); on a tight machine
    # (``tight_machine``) the transient carries the measured margin and
    # the floor is zero: the warm cache yields fully to live KV and
    # restores come from the SSD tier.
    runtime_transients_bytes: int = RUNTIME_TRANSIENTS_BYTES
    bank_floor_bytes: int = BANK_FLOOR_BYTES
    tight_machine: bool = False

    headroom_bytes: int = 0
    # "formula" (the 75% envelope on the planning RAM), "metal_limit" (the
    # Metal cap the server configured decided the engine budget) or
    # "resident_floor" (the family's wired floor sits above the 75% rule, so
    # the envelope is everything outside the system reserve).
    usable_source: str = "formula"
    # One active forward's temporary copy of historical K/V, independent
    # of query width. Decode's persistent KV and idle bank do not own it.
    attention_kv_copy_bytes_per_token: int = field(default=0, kw_only=True)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "total_ram_bytes": int(self.total_ram_bytes),
            "memory_budget_bytes": (
                None
                if self.memory_budget_bytes is None
                else int(self.memory_budget_bytes)
            ),
            "usable_bytes": int(self.usable_bytes),
            "model_weights_bytes": int(self.model_weights_bytes),
            "ngram_table_streamed_bytes": int(self.ngram_table_streamed_bytes),
            "kv_bytes_per_token": int(self.kv_bytes_per_token),
            "kv_quantization": self.kv_quantization,
            "kv_bytes_per_token_effective": int(self.kv_bytes_per_token_effective),
            "aux_bytes_per_token": int(self.aux_bytes_per_token),
            "fixed_cache_bytes": int(self.fixed_cache_bytes),
            "prefill_transient_bytes_per_token": int(
                self.prefill_transient_bytes_per_token
            ),
            **({"attention_kv_copy_bytes_per_token": self.attention_kv_copy_bytes_per_token}
               if self.attention_kv_copy_bytes_per_token else {}),
            "mtp_history_bytes_per_token": int(self.mtp_history_bytes_per_token),
            "model_fits": self.model_fits,
            "context_window_fit": int(self.context_window_fit),
            "context_window_resolved": int(self.context_window_resolved),
            "context_machine_bound": self.context_machine_bound,
            "context_overcommitted": self.context_overcommitted,
            "kv_reserve_tokens": int(self.kv_reserve_tokens),
            "kv_reserve_bytes": int(self.kv_reserve_bytes),
            "bank_idle_max_bytes": int(self.bank_idle_max_bytes),
            "bank_steady_bytes": int(self.bank_steady_bytes),
            "runtime_transients_bytes": int(self.runtime_transients_bytes),
            "bank_floor_bytes": int(self.bank_floor_bytes),
            "tight_machine": bool(self.tight_machine),
            "headroom_bytes": int(self.headroom_bytes),
            "notes": list(self.notes),
        }


def _unavailable(reason: str) -> MemoryPlan:
    return MemoryPlan(available=False, unavailable_reason=reason)


def plan_memory(
    *,
    total_ram_bytes: int | None,
    model_weights_bytes: int | None,
    kv_bytes_per_token: int = DEFAULT_DENSE_KV_BYTES_PER_TOKEN,
    kv_quantization: str = "off",
    model_max_context: int | None = None,
    requested_context: int | None = None,
    dense_decode_ceiling: int | None = None,
    memory_budget_bytes: int | None = None,
    usable_bytes_override: int | None = None,
    ngram_table_streamed_bytes: int = 0,
    aux_bytes_per_token: int = 0,
    fixed_cache_bytes: int = 0,
    prefill_transient_bytes_per_token: int = 0,
    attention_kv_copy_bytes_per_token: int = 0,
    usable_bytes_explicit: bool = False,
    resident_floor_bytes: int | None = None,
    tight_machine_measured: bool = False,
    mtp_history_bytes_per_token: int = 0,
    runtime_transients_bytes: int | None = None,
) -> MemoryPlan:
    """Solve the machine's memory geometry.

    ``memory_budget_bytes`` (--memory-budget / MTPLX_MEMORY_BUDGET)
    substitutes for machine RAM when tighter — the one knob that scales
    the whole stack down, and the way a 128 GB dev box tests a 48 GB
    seat. ``usable_bytes_override`` lets the server pass the Metal memory
    limit it actually configured instead of this module's mirror of the
    default formula; with ``usable_bytes_explicit`` (the operator set
    MTPLX_MEMORY_LIMIT_BYTES) that limit is the engine budget both ways,
    bounded by the machine (#443: a 96 GB seat serving a 69 GiB pack).
    ``resident_floor_bytes`` is the wired floor a family declares (weights
    plus working margin); when it exceeds the 75% rule the envelope is
    everything outside the system reserve (``engine_envelope_bytes``), the
    same number the Metal caps configure, on a real seat and on a seat
    simulated with --memory-budget alike. A machine that cannot fund the
    bank floor still runs a pack no heavier than the measured one
    (``TIGHT_MACHINE_MAX_WEIGHTS_BYTES``), or a pack that stamps its own
    tight-machine peaks (``tight_machine_measured``). A lighter such pack
    whose floor leaves it less context than that rule grants a heavier
    one yields its floor too, so a lighter pack never plans a smaller
    window than a heavier pack of the same geometry.
    ``runtime_transients_bytes`` is the runtime reserve a family measured for
    itself (Inkling 2 Mini: its prefill working set at its own chunk width);
    every other family keeps the flat ``RUNTIME_TRANSIENTS_BYTES``.
    """
    runtime_transients = (
        RUNTIME_TRANSIENTS_BYTES
        if runtime_transients_bytes is None
        else max(0, int(runtime_transients_bytes))
    )
    if total_ram_bytes is None or int(total_ram_bytes) <= 0:
        return _unavailable("total_ram_unknown")
    if model_weights_bytes is None or int(model_weights_bytes) <= 0:
        return _unavailable("model_weights_unknown")
    kv_per_token = max(1, int(kv_bytes_per_token))
    quant = str(kv_quantization or "off").strip().lower() or "off"
    factor = KV_QUANT_BYTE_FACTOR.get(quant, 1.0)
    kv_effective = max(1, int(kv_per_token * factor))

    total_ram = int(total_ram_bytes)
    budget = None if memory_budget_bytes is None else int(memory_budget_bytes)
    if budget is not None and budget > 0:
        planning_ram = min(total_ram, budget)
    else:
        planning_ram = total_ram
        budget = None
    usable_source = "formula"
    if usable_bytes_override is not None and int(usable_bytes_override) > 0:
        # The override is the Metal limit the server actually configured,
        # computed from the REAL machine. Under a tighter --memory-budget
        # the formula on the planning RAM must still win, or a simulated
        # 48G seat would inherit the 128G box's 96G envelope. An operator's
        # explicit limit (MTPLX_MEMORY_LIMIT_BYTES) is the engine budget in
        # both directions, bounded by the machine: issue #443's 96 GB seat
        # runs a 69.2 GiB pack under an 80G limit that the 72G formula
        # refused as "does not fit".
        override = int(usable_bytes_override)
        envelope = engine_envelope_bytes(
            planning_ram, resident_floor_bytes=resident_floor_bytes
        )
        if usable_bytes_explicit and planning_ram >= total_ram:
            usable = min(override, total_ram)
        else:
            usable = min(override, envelope)
        if usable == override and (usable_bytes_explicit or override < envelope):
            usable_source = "metal_limit"
    else:
        usable = engine_envelope_bytes(
            planning_ram, resident_floor_bytes=resident_floor_bytes
        )
    if usable_source == "formula" and usable > usable_engine_bytes(planning_ram):
        usable_source = "resident_floor"

    weights = int(model_weights_bytes)
    notes: list[str] = []

    # --- context window fit -------------------------------------------------
    # Per-token cost = KV + family aux (QSA streams, MTP-head KV) + the peak
    # prefill transient, which for QSA hybrids is LINEAR in context, not the
    # flat legacy constant (#393: 262K admitted with 2.4x phantom headroom
    # because the transient was priced at 3 GiB while the indexer's dense
    # lane peaks at ~27 GB there).
    aux_pt = max(0, int(aux_bytes_per_token))
    transient_pt = max(0, int(prefill_transient_bytes_per_token))
    kv_copy_pt = max(0, int(attention_kv_copy_bytes_per_token))
    per_token = kv_effective + aux_pt + transient_pt + kv_copy_pt
    fixed_cache = max(0, int(fixed_cache_bytes))
    kv_budget = usable - weights - fixed_cache - runtime_transients - BANK_FLOOR_BYTES
    fit_raw = _align_down(max(0, kv_budget) // per_token)
    model_fits = fit_raw >= CONTEXT_FLOOR_TOKENS
    model_cap = int(model_max_context) if model_max_context else None
    transients = runtime_transients
    bank_floor = BANK_FLOOR_BYTES
    tight_machine = False
    if tight_machine_measured or weights <= TIGHT_MACHINE_MAX_WEIGHTS_BYTES:
        # Tight machine: the resident session bank yields. The refusal
        # above charges the 1 GiB bank floor as if it were a physical need;
        # it is a clamp on the warm cache, and a machine that cannot fund
        # it can still run the model. Measured 2026-09-21 (Bonsai 2 27B,
        # 8.23 GiB weights, 16 GiB class at the 12 GiB budget): the peak
        # with no resident bank is weights + dense KV + 3.06 to 3.08 GiB
        # (4K 11.55 GiB, 8K 11.78 to 11.80 GiB under the budget; 16K 12.11
        # GiB over it). The q8 KV setting did not lower that peak (below
        # the paged ceiling the dense lane holds KV at full width: the 16K
        # peaks were byte-identical for off and q8), so this fit counts KV
        # at its dense width. Admit with the bank floor at zero and the
        # measured margin; the SSD tier keeps restores working.
        #
        # A pack the floor refuses is short of one block of KV with it, so
        # the most this rule grants is that block plus the floor net of the
        # margin, at the dense rate (``tight_ceiling``). A lighter pack whose
        # floor leaves it less than that yields its floor as well and takes
        # that window; otherwise a heavier pack plans more context than a
        # lighter one (2026-09-22: a 7.53 GiB MiMo build planned 12,288
        # tokens on 16 GiB while the 8.08 GiB Qwen3.5-9B planned 20,480).
        # Stopping at the ceiling keeps the window monotonic in pack size,
        # and a pack whose floor-funded window reaches it keeps its floor.
        tight_transients = runtime_transients + TIGHT_MACHINE_MARGIN_BYTES
        tight_per_token = kv_per_token + aux_pt + transient_pt + kv_copy_pt
        tight_budget = usable - weights - fixed_cache - tight_transients
        tight_ceiling = _align_down(
            (
                CONTEXT_FLOOR_TOKENS * per_token
                - 1
                + BANK_FLOOR_BYTES
                - TIGHT_MACHINE_MARGIN_BYTES
            )
            // tight_per_token
        )
        tight_own = _align_down(max(0, tight_budget) // tight_per_token)
        tight_fit = min(tight_own, tight_ceiling)
        if (
            tight_fit >= CONTEXT_FLOOR_TOKENS
            and tight_fit > fit_raw
            and (model_cap is None or fit_raw < model_cap)
        ):
            model_fits = True
            tight_machine = True
            fit_raw = tight_fit
            transients = tight_transients
            bank_floor = 0
            # The reserve and the server's admission arithmetic read this
            # rate; under the paged ceiling the measured KV is dense.
            kv_effective = kv_per_token
            notes.append(
                "tight machine: the session bank has no floor (the warm "
                "cache yields fully to live KV; restores come from SSD) and "
                f"KV is counted at its dense width; weights {weights / GIB:.1f}"
                f" GiB + runtime {tight_transients / GIB:.2f} GiB leave "
                f"{tight_budget / GIB:.2f} GiB for KV under the "
                f"{usable / GIB:.1f} GiB engine budget"
                + (
                    f"; the window stops at {tight_ceiling} tokens, the most "
                    "this rule grants a pack that cannot fund the floor"
                    if tight_own > tight_ceiling
                    else ""
                )
            )
    if not model_fits:
        notes.append(
            "model does not fit: weights + minimum runtime need "
            f"{(weights + fixed_cache + runtime_transients + BANK_FLOOR_BYTES + per_token * CONTEXT_FLOOR_TOKENS) / GIB:.1f}"
            f" GiB but the engine budget is {usable / GIB:.1f} GiB"
            + (
                ""
                if usable_source != "formula"
                else "; MTPLX_MEMORY_LIMIT_BYTES sets the engine budget past the"
                " default envelope (swap past the fit is then yours)"
            )
        )
        fit_raw = CONTEXT_FLOOR_TOKENS

    context_fit = fit_raw if model_cap is None else min(fit_raw, model_cap)
    machine_bound = model_cap is not None and fit_raw < model_cap

    if requested_context is not None and int(requested_context) > 0:
        resolved = int(requested_context)
        overcommitted = resolved > fit_raw
    else:
        resolved = context_fit
        overcommitted = False
    if machine_bound and not overcommitted:
        notes.append(
            f"context window {resolved} is machine-bound "
            f"(model supports {model_cap}); larger windows would swap on "
            f"{planning_ram / GIB:.0f} GiB"
        )
    if overcommitted:
        notes.append(
            f"requested context {resolved} exceeds the machine fit "
            f"{fit_raw}: KV at the full window needs "
            f"{resolved * kv_effective / GIB:.1f} GiB; expect cache "
            "shedding and, past the fit, swap"
        )

    # --- bank budgets -------------------------------------------------------
    # Steady-state KV projection: a session may hold the whole committed
    # window, so the bank's advertised under-load budget subtracts the KV
    # and the family's per-token working set (QSA streams, the MTP head's KV)
    # of the resolved window. It used to stop at the dense-decode ceiling
    # and count KV alone (#525: a 262K window on 64 GB reserved 157K tokens,
    # and the committed rest rode the paged lane unpriced).
    # ``dense_decode_ceiling`` no longer narrows it: the paged lane past the
    # ceiling holds the same tokens. The runtime ceiling
    # (bank_dynamic_ceiling) still reads the live working set.
    reserve_tokens = int(resolved)
    kv_reserve = fixed_cache + reserve_tokens * (kv_effective + aux_pt)
    bank_idle = usable - weights - transients
    bank_idle = max(bank_floor, min(BANK_CAP_BYTES, bank_idle))
    bank_steady = usable - weights - transients - kv_reserve
    bank_steady = max(bank_floor, min(BANK_CAP_BYTES, bank_steady))
    headroom = max(0, usable - weights - transients - kv_reserve - bank_steady)

    return MemoryPlan(
        available=True,
        total_ram_bytes=total_ram,
        memory_budget_bytes=budget,
        usable_bytes=usable,
        model_weights_bytes=weights,
        ngram_table_streamed_bytes=max(0, int(ngram_table_streamed_bytes)),
        kv_bytes_per_token=kv_per_token,
        kv_quantization=quant,
        kv_bytes_per_token_effective=kv_effective,
        aux_bytes_per_token=aux_pt,
        fixed_cache_bytes=fixed_cache,
        prefill_transient_bytes_per_token=transient_pt,
        attention_kv_copy_bytes_per_token=kv_copy_pt,
        mtp_history_bytes_per_token=max(0, int(mtp_history_bytes_per_token)),
        model_fits=model_fits,
        context_window_fit=context_fit,
        context_window_resolved=resolved,
        context_machine_bound=machine_bound,
        context_overcommitted=overcommitted,
        kv_reserve_tokens=reserve_tokens,
        kv_reserve_bytes=kv_reserve,
        bank_idle_max_bytes=bank_idle,
        bank_steady_bytes=bank_steady,
        runtime_transients_bytes=transients,
        bank_floor_bytes=bank_floor,
        tight_machine=tight_machine,
        headroom_bytes=headroom,
        usable_source=usable_source,
        notes=tuple(notes),
    )


def transient_reserve_bytes(
    peak_bytes: int, active_bytes: int, *, play_bytes: int | None = None
) -> int:
    """Headroom to hold back from the bank for the next allocation spike.

    The static RUNTIME_TRANSIENTS_BYTES (3 GiB) underestimates a deep
    chunked prefill: measured peak-over-active reached 12.4 GiB at 73k ctx
    (2026-08-29 receipts), so the bank kept entries while the allocator
    peak kissed 0.992-1.001 of the Metal limit — which is what tripped the
    warning banner (allocator trigger 0.97) on every deep coding turn.
    Reserve what this process has actually spiked (clamped: never below
    the static floor, capped so one pathological turn cannot starve the
    bank forever), so entries demote to SSD BEFORE the next spike can
    slam the ceiling.

    ``peak_bytes`` is the allocator's lifetime high-water; ``active`` is
    now. Their difference overstates the instantaneous spike when active
    has since fallen — overstating is the safe direction here (an extra
    SSD restore costs seconds; a ceiling kiss costs the banner plus shed
    churn on every long turn) — but only up to a point: on a model whose
    weights leave little play (Flash-Next: 77 G weights against a 96 GiB
    limit leaves ~18 G), an uncapped lifetime spike would permanently eat
    the whole bank budget after one deep turn. ``play_bytes``
    (usable − weights) caps the reserve at half the play, so the warm
    cache always keeps at least the other half.
    """
    spike = max(0, int(peak_bytes) - max(0, int(active_bytes)))
    cap = TRANSIENT_RESERVE_CAP_BYTES
    if play_bytes is not None and int(play_bytes) > 0:
        cap = min(cap, max(RUNTIME_TRANSIENTS_BYTES, int(play_bytes) // 2))
    return max(RUNTIME_TRANSIENTS_BYTES, min(cap, spike))


def bank_dynamic_ceiling(
    plan: MemoryPlan,
    working_set_bytes: int,
    *,
    transient_bytes: int | None = None,
) -> int:
    """The bank budget right now, given what live requests actually hold.

    ``working_set_bytes`` is live KV + generation transients as measured
    by the server's memory attribution (active - weights - bank). Idle
    machine: the full idle max — the warm cache takes everything free.
    A 150k-token session materializes 10 GiB of KV: the ceiling walks
    down and the guard demotes bank entries to SSD ahead of any swap.
    This is the "guard that turns on": zero effect until commitments
    actually grow, then caches yield in eviction order — never the live
    request.

    ``transient_bytes`` overrides the static spike reserve with an
    observed one (transient_reserve_bytes) so the ceiling anticipates the
    real prefill high-water instead of the 3 GiB guess.
    """
    if not plan.available:
        return BANK_CAP_BYTES
    static = int(getattr(plan, "runtime_transients_bytes", RUNTIME_TRANSIENTS_BYTES))
    reserve = static if transient_bytes is None else max(static, int(transient_bytes))
    ceiling = (
        plan.usable_bytes
        - plan.model_weights_bytes
        - reserve
        - max(0, int(working_set_bytes))
    )
    floor = int(getattr(plan, "bank_floor_bytes", BANK_FLOOR_BYTES))
    return max(floor, min(int(plan.bank_idle_max_bytes), ceiling))


def describe_plan(plan: MemoryPlan) -> str:
    """One human line for the serve banner / doctor / app tooltip."""
    if not plan.available:
        return f"memory plan unavailable ({plan.unavailable_reason})"
    ram = plan.total_ram_bytes / GIB
    budget = (
        ""
        if plan.memory_budget_bytes is None
        else f" (budget {plan.memory_budget_bytes / GIB:.0f}G)"
    )
    line = (
        f"{ram:.0f}G Mac{budget}: engine budget "
        f"{plan.usable_bytes / GIB:.1f}G"
        + (" (Metal limit)" if plan.usable_source == "metal_limit" else "")
        + (
            " (RAM minus the system reserve: the weights sit above the 75% rule)"
            if plan.usable_source == "resident_floor"
            else ""
        )
        + ", weights "
        f"{plan.model_weights_bytes / GIB:.1f}G, context "
        f"{plan.context_window_resolved}"
        + (" (machine-bound)" if plan.context_machine_bound else "")
        + (" (OVERCOMMITTED)" if plan.context_overcommitted else "")
        + f", session bank up to {plan.bank_idle_max_bytes / GIB:.1f}G"
    )
    if plan.bank_steady_bytes < plan.bank_idle_max_bytes:
        line += (
            f" (yields to {plan.bank_steady_bytes / GIB:.1f}G under "
            "long-context load)"
        )
    if plan.ngram_table_streamed_bytes > 0:
        line += (
            f", n-gram table {plan.ngram_table_streamed_bytes / GIB:.1f}G "
            "streamed from SSD (not wired)"
        )
    if plan.tight_machine:
        line += " (tight machine: no bank floor, restores from SSD)"
    if not plan.model_fits:
        line += " — MODEL DOES NOT FIT"
    return line
