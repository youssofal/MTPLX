"""Public MTPLX runtime profiles.

Profiles are product policy, not benchmark folklore.  They are deliberately
kept free of MLX imports so the CLI can explain the available modes in a fresh
environment before the runtime stack is installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

ProfileName = str

DEFAULT_PROFILE_NAME = "sustained"
PROFILE_CHOICES = (
    "stable",
    "performance-cold",
    "sustained",
    "turbo",
    "exact",
    "max-diagnostic",
)
PROFILE_ENV_USER_OVERRIDE_KEYS = frozenset(
    {
        "MTPLX_LONG_CONTEXT_MTP_DEPTH",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT",
        # Prefill chunk sizing is a tuning knob users/operators legitimately
        # sweep; an explicit env must beat the profile default like the
        # depth/history knobs above (before 2026-07-05 the profile applier
        # silently stomped it back to the profile value).
        "MTPLX_PREFILL_CHUNK_SIZE",
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
        "MTPLX_PREFILL_CHUNK_SIZE_REPAGE",
        # Packed-GQA verify attention (speed-war Lane A): operators must be
        # able to force it off/on per launch for A/B work.
        "MTPLX_GQA_PACKED_SDPA",
        "MTPLX_GQA_PACKED_SDPA_THRESHOLD",
        # Dense-decode context ceiling (2026-08-26): past it the auto layout
        # repages decode and the packed lane is structurally excluded — the
        # 147.4k decode cliff. Operators must be able to sweep it per launch.
        "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT",
        # Compiled-verify commit-first donation (speed-war Lane A2): same
        # A/B requirement — an explicit env must beat the profile default.
        "MTPLX_COMPILED_VERIFY_DONATION",
        # Compiled-verify context ceiling: operators sweep it for long-context
        # A/Bs (2026-07-17 the sweep needed a site-packages patch because the
        # profile stomped the env). Same precedent as DONATION above.
        "MTPLX_COMPILED_VERIFY_MAX_CONTEXT",
        # Compiled-verify mode switch: parity/parity2 exactness gates must be
        # launchable against the turbo profile itself (Gate A on the exact
        # config being shipped), not only on profiles that leave the env
        # unset. Same operator-A/B precedent as DONATION/MAX_CONTEXT.
        "MTPLX_COMPILED_VERIFY",
        # Verify target-distribution strategy (PR #314, 2026-08-21): lazy
        # per-row vs batched precompute is a real A/B operators must be able
        # to launch against the shipping profiles — the batched arm is
        # MTPLX_LAZY_TARGET_DISTRIBUTIONS=0 MTPLX_BATCH_TARGET_ARRAYS=1.
        # Before this entry an exported value was silently stomped back to
        # the profile default (the reporter had to patch site-packages to
        # measure). Same operator-A/B precedent as DONATION above.
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
        "MTPLX_BATCH_TARGET_ARRAYS",
        # Background warmup ladder (F6, 2026-08-16): operators sweep the
        # rung list per machine/benchmark; an explicit env must beat the
        # turbo default below, same precedent as the chunk-size knobs.
        "MTPLX_WARMUP_LADDER",
        # Long-generation decay levers (2026-08-28): the uncapped-chat decay
        # investigation sweeps these against the shipping turbo profile, and
        # per-round event capture is how the growth term is attributed.
        # Same operator-A/B precedent as DONATION/LAZY above.
        "MTPLX_CLEAR_CACHE_EVERY",
        "MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD",
        "MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT",
        "MTPLX_MTP_HISTORY_LAST_WINDOW",
        "MTPLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD",
        "MTPLX_DROP_EVENTS",
        "MTPLX_SKIP_VERIFY_SNAPSHOT",
        "MTPLX_FAMILY_CAPTURE_COMMIT",
    }
)

# Operator envs that beat a profile value at apply time (F23c, 2026-08-16).
# Rebuilt in place by every apply_profile_env() call, so the list always
# reflects the most recent application. Stable name for /health readers:
# each entry is {"var", "profile_value", "actual_value"}. Precedence is
# unchanged — operator env SHOULD win — this is visibility only; envs that
# merely pin the profile's own value are not listed (nothing degraded).
profile_env_overridden: list[dict[str, str]] = []

OPTIMIZED_SPEED_V1_HF_MODEL_ID = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
OPTIMIZED_SPEED_V2_HF_MODEL_ID = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2"
DEFAULT_FP16_HF_MODEL_ID = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16"
QUALITY_HF_MODEL_ID = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality"
QUALITY_FP16_HF_MODEL_ID = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16"
LEGACY_OPTIMIZED_HF_MODEL_ID = "Youssofal/Qwen3.6-27B-MTPLX-Optimized"
QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID = (
    "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"
)
QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID = (
    "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16"
)
QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID = (
    "mtplx-qwen35-9b-optimized-speed"
)
QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID = (
    "mtplx-qwen35-9b-optimized-speed-fp16"
)
QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID = (
    "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"
)
QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID = (
    "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"
)
QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID = (
    "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance"
)
QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID = (
    "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16"
)
QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID = (
    "mtplx-qwen36-35b-a3b-optimized-speed"
)
QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID = (
    "mtplx-qwen36-35b-a3b-optimized-speed-fp16"
)
QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID = (
    "mtplx-qwen36-35b-a3b-optimized-balance"
)
QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID = (
    "mtplx-qwen36-35b-a3b-optimized-balance-fp16"
)
QWEN38_BARE_SPEED_HF_MODEL_ID = "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed"
QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID = (
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
)
QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID = (
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"
)
QWEN38_BARE_SPEED_PUBLIC_MODEL_ID = "mtplx-qwen38-27b-bare-speed"
QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID = "mtplx-qwen38-27b-optimized-speed"
QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID = "mtplx-qwen38-27b-optimized-quality"
# FP16 precision siblings of the Qwen 3.8 trio (2026-08-15): byte-identical
# INT packs with the bf16 floats cast to fp16, the M1/M2 routing targets
# (same policy as the 3.6 -FP16 siblings).
QWEN38_BARE_SPEED_FP16_HF_MODEL_ID = "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-FP16"
QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID = (
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"
)
QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID = (
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16"
)
QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID = "mtplx-qwen38-27b-bare-speed-fp16"
QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID = "mtplx-qwen38-27b-optimized-speed-fp16"
QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID = (
    "mtplx-qwen38-27b-optimized-quality-fp16"
)
# Qwen 3.8 Flash-Next serve identities (2026-08-27 turbo-first-class
# promotion). The public ids equal the model_catalog aliases so catalog
# installs, `mtplx pull mtplx-flash-next-*`, and served-dir name inference
# all converge on one identity. Derivative packs (-hc8, -v1a, RC dirs) fall
# through to sanitized names by the exact-equality fence in default_models.
FLASH_NEXT_BARE_SPEED_HF_MODEL_ID = "Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed"
FLASH_NEXT_OPTIMIZED_SPEED_HF_MODEL_ID = (
    "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
)
FLASH_NEXT_BARE_SPEED_PUBLIC_MODEL_ID = "mtplx-flash-next-bare-speed"
FLASH_NEXT_OPTIMIZED_SPEED_PUBLIC_MODEL_ID = "mtplx-flash-next-optimized-speed"
# Public default (2026-08-15, founder ruling on the Qwen3.8 release): the
# Qwen 3.8 Optimized Speed dynamic 4-bit build is the recommended pick and the
# fresh-install default on modern Apple Silicon. Its weights are published on
# the Hub (release-day upload), so quickstart, pull, onboarding, and clean
# machines resolve a downloadable repo. M1/M2 keep the FP16 3.6 sibling and
# <32 GiB Macs keep the 9B route via default_models.
DEFAULT_HF_MODEL_ID = QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID
QUALITY_MODEL_ID = QUALITY_HF_MODEL_ID
DEFAULT_MODEL_ID = DEFAULT_HF_MODEL_ID
OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID = "mtplx-qwen36-27b-optimized-speed"
OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID = "mtplx-qwen36-27b-optimized-speed-v2"
DEFAULT_PUBLIC_MODEL_ID = QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
DEFAULT_FP16_PUBLIC_MODEL_ID = "mtplx-qwen36-27b-optimized-speed-fp16"
QUALITY_PUBLIC_MODEL_ID = "mtplx-qwen36-27b-optimized-quality"
QUALITY_FP16_PUBLIC_MODEL_ID = "mtplx-qwen36-27b-optimized-quality-fp16"
LEGACY_OPTIMIZED_PUBLIC_MODEL_ID = "mtplx-qwen36-27b-optimized"


NATIVE_MTP_60_FAST_PATH_ENV = {
    "MTPLX_LAZY_VERIFY_LOGITS": "1",
    # Batched vs lazy target distributions are mutually exclusive verify
    # strategies: every batched-build site in generation.py is guarded on
    # the lazy flag being OFF, so with LAZY_TARGET_DISTRIBUTIONS=1 a "1"
    # here is dead configuration. The pair shipped contradictory from
    # 1.0.0 through 2.9.0 — BATCH_TARGET_ARRAYS=1 is the May 60-tok/s
    # stack, the lazy strategy was layered on 2026-06-09 ("Recover
    # OpenCode MTP decode speed", eager distribution materialization
    # dominated D3 decode) and wins at runtime. "0" states the active
    # truth: the product strategy is lazy. The batched candidate stays
    # launchable as MTPLX_LAZY_TARGET_DISTRIBUTIONS=0
    # MTPLX_BATCH_TARGET_ARRAYS=1 (both operator-overridable, PR #314);
    # flipping the DEFAULT is ABBA-gated, not a wiring call.
    "MTPLX_BATCH_TARGET_ARRAYS": "0",
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS": "1",
    "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
    "MTPLX_DROP_EVENTS": "1",
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "1",
}

# Known runtime gating relations between fast-path envs: generation.py
# consults the gated flag only inside branches that require the gating flag
# to be OFF, so a launch where both are truthy silently kills the gated
# flag. apply_profile_env() announces every live combination — one loud
# line per dead flag — no matter which layer set it (profile, operator
# env, or an app/CLI lane default). Loud beats silent: this is exactly how
# turbo/sustained shipped a dead MTPLX_BATCH_TARGET_ARRAYS=1 for ten weeks
# (1.0.0 -> 2.9.0) with /health reporting ok:true, and how the coding-agent
# lanes still pin a MTPLX_LAZY_BONUS_VERIFY=1 the same gate disables.
# Entries: (gated key, gating key, why).
RUNTIME_GATED_ENV_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "MTPLX_BATCH_TARGET_ARRAYS",
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
        "batched target-distribution precompute requires the lazy per-row "
        "strategy off",
    ),
    (
        "MTPLX_BATCH_TARGET_DISTS",
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
        "batched target-distribution precompute requires the lazy per-row "
        "strategy off",
    ),
    (
        "MTPLX_LAZY_BONUS_VERIFY",
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
        "lazy bonus verify requires the lazy-distribution strategy off",
    ),
)

# Mirrors generation.py's _env_truthy so announcements judge the same
# values the runtime gates do.
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUTHY_ENV_VALUES


def announce_runtime_gated_env(
    environ: Mapping[str, str] | None = None,
    *,
    profile_name: str | None = None,
) -> list[dict[str, str]]:
    """Print one loud line per env flag that is dead under the current env.

    Returns the gated entries so callers/health surfaces can persist them.
    Runs against the FINAL environment (after profile + overrides), so it
    catches profile-set, operator-set, and launcher-lane-injected combos
    alike.
    """

    target = os.environ if environ is None else environ
    gated: list[dict[str, str]] = []
    for dead_key, gating_key, why in RUNTIME_GATED_ENV_PAIRS:
        if not (_truthy_env(target.get(dead_key)) and _truthy_env(target.get(gating_key))):
            continue
        gated.append(
            {
                "var": dead_key,
                "value": str(target.get(dead_key)),
                "gated_by": gating_key,
                "gated_by_value": str(target.get(gating_key)),
                "reason": why,
            }
        )
        suffix = f"; profile {profile_name}" if profile_name else ""
        try:
            print(
                f"[mtplx] env gated at runtime: {dead_key}="
                f"{target.get(dead_key)} has no effect while "
                f"{gating_key}={target.get(gating_key)} ({why}{suffix})",
                flush=True,
            )
        except Exception:
            pass
    return gated

MODEL_RUNTIME_ENV_OVERRIDE_KEYS = frozenset(
    {
        *NATIVE_MTP_60_FAST_PATH_ENV,
        "MTPLX_AR_PIPELINE",
        "MTPLX_COMPILED_GDN",
        "MTPLX_QWEN4EXP_COMPILE",
        "MTPLX_DEFER_REPAIR_EVAL",
        "MTPLX_FAMILY_CAPTURE_COMMIT",
        "MTPLX_NGRAM_RESIDENT",
        "MTPLX_FUSED_GATE_UP",
        "MTPLX_FUSED_GDN_INPROJ",
        "MTPLX_FUSED_GDN_OUT",
        "MTPLX_FUSED_QSA_QKV",
        # Exact fused QSA score/mask/top-k selector (2026-08-29 candidate).
        # Keep this registered while it is opt-in so model contracts and
        # operator A/B launches can exercise the kill switch without failing
        # the boot-time runtime-env validator.
        "MTPLX_FUSED_QSA_INDEXER",
        # Pure explicit-state mx.compile wrapper around projection/query
        # preparation/cache staging/selection. Independently gated so the
        # Metal selector can be A/B tested without graph capture.
        "MTPLX_COMPILED_QSA_INDEXER",
        # Matrix-shaped QSA prefill pipeline: byte-bounded tiled MLX scoring,
        # dedicated Metal top-k, and direct block-sparse attention.  All
        # shipped profiles keep it off until numeric and production A/B gates.
        "MTPLX_QSA_PREFILL",
        "MTPLX_QSA_PREFILL_MIN_ROWS",
        # Independent measured crossovers: the tiled score/top-k producer can
        # amortize before the direct scattered K/V consumer does.
        "MTPLX_QSA_PREFILL_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_SCORE_MB",
        # Vendored Steel sparse-GQA consumer of the same block selections
        # (native_extensions/qsa_kernels, oMLX PR #3244). Default on wherever
        # the native module is built and ABI-probed; "0" kills only this
        # consumer. MTPLX_QSA_PREFILL=0 still kills the whole lane. It is the
        # only fast consumer M3-class GPUs can reach, so it also participates
        # in the producer auto-gate.
        "MTPLX_QSA_PREFILL_DIRECT",
        "MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_DIRECT_VALIDATE",
        # Portable gathered-attention tier for the flash_prefill contract:
        # the universal (non-NAX) consumer of the same block selections.
        "MTPLX_QSA_PREFILL_GATHER",
        "MTPLX_QSA_PREFILL_GATHER_TILE",
        # Engagement-receipt atexit print for lane A/Bs (counters law).
        "MTPLX_QSA_PREFILL_DEBUG",
        # Capture only one canonical chunk width so arbitrary final/restored
        # suffix sizes cannot grow the mx.compile graph bank without bound.
        "MTPLX_QSA_PREFILL_COMPILE_ROWS",
        # Phase-3 QSA/MTP capacity staging and exact replay reconciliation.
        # Default-off until the fused+compiled model/MTP gates are complete;
        # registered now so that those explicit A/B launches are valid.
        "MTPLX_QSA_MTP_PRECOMPUTE",
        "MTPLX_FUSED_GDN_CONVNORM",
        # One-dispatch GDN decode step (2026-08-27 family default; two
        # boot-triple receipt in the server's family octet comment).
        "MTPLX_FUSED_GDN_STEP",
        # Verify-width conv+silu+l2norm rows kernel (2026-08-27 candidate,
        # A/B pending — registered ahead of any default flip per the
        # boot-time-validator lesson in mistakes/).
        "MTPLX_FUSED_CONVNORM_VERIFY",
        # QSA decode gather lane (2026-08-27 candidate): selected-token
        # gather + maskless SDPA instead of dense-bool-mask over full KV.
        # Long-context A/B pending; registered ahead per the same lesson.
        "MTPLX_QSA_GATHER",
        # Rows-gather routing knobs (2026-08-28, adapting PR #380): S>1
        # engage floor by KV length and the served row-width ceiling.
        # Registered ahead of any default flip per the boot-trap law.
        "MTPLX_QSA_GATHER_MIN_CONTEXT",
        "MTPLX_QSA_GATHER_MAX_ROWS",
        # QSA block-sparse flash-skip attention (2026-08-27 candidate):
        # selected blocks iterated inside the kernel, no staging/copies.
        "MTPLX_QSA_FLASH",
        # n-gram hot-row LRU size in MB for the streamed sidecar
        # (2026-08-28, default 1024; 0 disables) — registered ahead per the
        # boot-trap law so packs/profiles may stamp it.
        "MTPLX_NGRAM_HOT_MB",
        # Family-scoped NAX neutralize (2026-08-27): qwen4_exp holds the 27B
        # NAX verify patch OFF under turbo until it earns a family receipt.
        "MTPLX_NAX_VERIFY",
        "MTPLX_FUSED_HC_V3",
        "MTPLX_FUSED_MOE_DECODE",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_MTP_HISTORY_LAST_WINDOW",
        "MTPLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH",
        "MTPLX_CLEAR_CACHE_EVERY",
        "MTPLX_COMPILED_VERIFY",
        "MTPLX_COMPILED_VERIFY_MAX_LEN",
        "MTPLX_COMPILED_TARGET_PREFIX",
        "MTPLX_FUSE_GDN_POST_CONV",
        "MTPLX_A3B_GDN_POSTCONV_IMPL",
        "MTPLX_LINEAR_GDN_FROM_CONV_TGY",
        "MTPLX_A3B_WHOLE_MOE_FUSION",
        "MTPLX_QWEN_COMBINE_TAIL",
        "MTPLX_QWEN_MOE_PACK_GATE_UP",
        "MTPLX_QWEN_ROW_OWNED_ROUTER",
    }
)


def _runtime_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return str(value)
    text = str(value).strip()
    if text == "":
        raise ValueError("runtime_env_overrides values must not be empty")
    return text


def normalize_runtime_env_overrides(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("runtime_env_overrides must be an object")
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if name not in MODEL_RUNTIME_ENV_OVERRIDE_KEYS:
            raise ValueError(f"runtime_env_overrides has unsupported key: {name}")
        normalized[name] = _runtime_env_value(value)
    return normalized


def runtime_env_overrides_from_contract(contract: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(contract, Mapping):
        return {}
    return normalize_runtime_env_overrides(contract.get("runtime_env_overrides"))


def runtime_env_with_contract_overrides(
    runtime_env: Mapping[str, str],
    contract: Mapping[str, Any] | None,
) -> dict[str, str]:
    merged = dict(runtime_env)
    merged.update(runtime_env_overrides_from_contract(contract))
    return merged

EXACT_PAGED_ATTENTION_ENV = {
    "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
    "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "16",
    "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": "1024",
    "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged",
    "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "1",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "2048",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "512",
}

LONG_RESPONSE_STAGED_ENV = {
    "MTPLX_DROP_EVENTS": "1",
    "MTPLX_EVAL_STATE_ROOTS_ON_COMMIT": "1",
    "MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP": "1",
    "MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE": "1",
    "MTPLX_TARGET_LAYER_EVAL_SCHEDULE": "2048:16,8192:8",
    "MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD": "0",
    "MTPLX_TARGET_LAYER_EVAL_MAX_Q": "8",
}

SUSTAINED_PREFILL_ENV = {
    **NATIVE_MTP_60_FAST_PATH_ENV,
    "MTPLX_SUSTAINED_PREFILL": "1",
    "MTPLX_SUSTAINED_PREFILL_LAYOUT": "auto",
    # "auto" (2.9.3): memory-aware dense-decode ceiling, floored at the old
    # 131072 literal so no machine regresses. The fixed 131072 was a memory
    # guess, not a kernel envelope — past 2^17 tokens the auto layout repaged
    # decode and structurally excluded the packed fast-SDPA lane (the 147.4k
    # decode cliff: 12.0 -> 18.44 tok/s once dense decode holds; walk ladders
    # show both kernels linear through 2^17, MEASUREMENTS 2026-08-26). The
    # resolver budgets 15% of device RAM against the model's KV bytes/token
    # (MTPLX_DENSE_KV_BYTES_PER_TOKEN) and announces its resolution in the
    # serve log.
    "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT": "auto",
    # MTPLX_PREFILL_CHUNK_SIZE is retained as a legacy single-knob fallback:
    # if set to a numeric value it overrides BOTH paths. "auto" resolves to
    # the per-layout defaults below, which intentionally match in product mode.
    "MTPLX_PREFILL_CHUNK_SIZE": "auto",
    "MTPLX_PREFILL_CHUNK_SIZE_DENSE": "2048",
    "MTPLX_PREFILL_CHUNK_SIZE_REPAGE": "2048",
    "MTPLX_PREFILL_CHUNK_CACHE_CLEANUP": "1",
    "MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY": "auto",
    "MTPLX_PREFILL_OMLX_EXTERNAL": "1",
    "MTPLX_PREFILL_EXTERNAL_EMIT_LOGITS": "0",
    "MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS": "0",
    "MTPLX_DEFER_VERIFY_HIDDEN_EVAL": "1",
    "MTPLX_VERIFY_HIDDEN_MODE": "logits_first_committed_slice",
    "MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY": "off",
    "MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD": "98304",
    "MTPLX_LONG_CONTEXT_MTP_DEPTH": "3",
    # Real OpenCode/Pi app runs showed that the 16k auto flip to an 8k
    # MTP-history window can crater acceptance and long-context decode speed.
    # Keep the full committed history in the product path; "auto" remains an
    # explicit diagnostic override.
    "MTPLX_MTP_HISTORY_POLICY": "committed",
    "MTPLX_MTP_HISTORY_LAST_WINDOW": "8192",
    "MTPLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD": "16384",
    "MTPLX_DYNAMIC_PAGED_KV": "1",
    "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
    "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "16",
    "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged",
    "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "1",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "2048",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "512",
    "MTPLX_VLLM_METAL_PAGED_TURBOQUANT": "0",
    "MTPLX_CLEAR_CACHE_EVERY": "auto",
    "MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD": "16384",
    # 256 -> 1024 (2026-07-16): the every-256-step clear measured -3.8%
    # decode on 512-token generations at 33k ctx (42.55 -> 44.15 tok/s with
    # clears off, matched load/acceptance). At 1024, typical agent turns
    # (<1k tokens) never pay the sync, while marathon responses still get
    # periodic allocator bounding. Memory re-validated on a 2k-token 33k-ctx
    # marathon row before shipping.
    "MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT": "1024",
}


def _env_int(
    env: Mapping[str, str] | MutableMapping[str, str] | None,
    key: str,
    *,
    default: int,
) -> int:
    source = os.environ if env is None else env
    try:
        return int(str(source.get(key, "") or default))
    except (TypeError, ValueError):
        return default


def resolve_long_context_mtp_depth(
    *,
    prompt_tokens: int,
    requested_depth: int,
    min_depth: int = 1,
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    """Resolve Sustained's optional context-aware MTP depth cap.

    Product mode currently leaves the cap disabled so a D3 request remains D3 at
    long context. The helper stays in place only as an explicit diagnostic
    switch; it must never silently downgrade the native app/OpenCode path.
    """

    source = os.environ if env is None else env
    requested = max(1, int(requested_depth))
    floor = max(1, int(min_depth))
    policy = (
        str(source.get("MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY", "") or "off")
        .strip()
        .lower()
        .replace("-", "_")
    )
    threshold = max(
        0,
        _env_int(source, "MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD", default=98_304),
    )
    cap_depth = max(
        1,
        _env_int(source, "MTPLX_LONG_CONTEXT_MTP_DEPTH", default=2),
    )
    details: dict[str, object] = {
        "policy": policy,
        "prompt_tokens": int(prompt_tokens),
        "threshold": int(threshold),
        "cap_depth": int(cap_depth),
        "requested_depth": int(requested),
        "min_depth": int(floor),
        "active": False,
        "reason": "disabled",
    }
    if policy in {"", "0", "off", "false", "none"}:
        details["effective_depth"] = int(requested)
        return requested, details
    if policy != "auto":
        details["reason"] = "unknown_policy"
        details["effective_depth"] = int(requested)
        return requested, details
    if int(prompt_tokens) < threshold:
        details["reason"] = "below_threshold"
        details["effective_depth"] = int(requested)
        return requested, details
    effective = min(requested, max(cap_depth, floor))
    details["effective_depth"] = int(effective)
    if effective < requested:
        details["active"] = True
        details["reason"] = "long_context_depth_cap"
    else:
        details["reason"] = "already_within_cap"
    return effective, details


@dataclass(frozen=True)
class SamplerDefaults:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


@dataclass(frozen=True)
class DraftLMHeadRequirement:
    bits: int
    group_size: int
    mode: str


@dataclass(frozen=True)
class RuntimeProfile:
    name: ProfileName
    runtime_profile: str
    summary: str
    env: tuple[tuple[str, str], ...]
    sampler: SamplerDefaults = SamplerDefaults()
    model_id: str = DEFAULT_MODEL_ID
    benchmark_ids: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    draft_lm_head: DraftLMHeadRequirement | None = None
    draft_sampler: SamplerDefaults | None = None
    qa_only: bool = False
    fan_control_allowed: bool = False
    clock_anchor_allowed: bool = False
    product_claim_eligible: bool = True

    def env_dict(self) -> dict[str, str]:
        return dict(self.env)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "runtime_profile": self.runtime_profile,
            "summary": self.summary,
            "env": self.env_dict(),
            "sampler": {
                "temperature": self.sampler.temperature,
                "top_p": self.sampler.top_p,
                "top_k": self.sampler.top_k,
            },
            "model_id": self.model_id,
            "benchmark_ids": list(self.benchmark_ids),
            "caveats": list(self.caveats),
            "draft_lm_head": (
                None
                if self.draft_lm_head is None
                else {
                    "bits": self.draft_lm_head.bits,
                    "group_size": self.draft_lm_head.group_size,
                    "mode": self.draft_lm_head.mode,
                }
            ),
            "draft_sampler": (
                None
                if self.draft_sampler is None
                else {
                    "temperature": self.draft_sampler.temperature,
                    "top_p": self.draft_sampler.top_p,
                    "top_k": self.draft_sampler.top_k,
                }
            ),
            "qa_only": self.qa_only,
            "fan_control_allowed": self.fan_control_allowed,
            "clock_anchor_allowed": self.clock_anchor_allowed,
            "product_claim_eligible": self.product_claim_eligible,
        }


def _items(mapping: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(mapping.items()))


def _merge_env(*mappings: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    env: dict[str, str] = {}
    for mapping in mappings:
        env.update(mapping)
    return _items(env)


STABLE_PROFILE = RuntimeProfile(
    name="stable",
    runtime_profile="long_response_exact_staged",
    summary=(
        "Stable Mode: exact/staged long-reply path with no fan control. Hidden "
        "from first-run onboarding, but available by flag for compatibility."
    ),
    env=_merge_env(EXACT_PAGED_ATTENTION_ENV, LONG_RESPONSE_STAGED_ENV),
    benchmark_ids=(
        "gdn8-flappy-nofan-directhttp-seed42-cli-fix-20260501",
        "phase0j-gdn8-speed4-target-layer-sched-state-root-mlx-vector-paged-flappy-uncapped-modelswap-20260501",
    ),
    caveats=(
        "Lower peak throughput than the fan-backed Burst lane.",
        "Selected for repeatable long replies while the v0.2 decay work continues.",
    ),
)

PERFORMANCE_COLD_PROFILE = RuntimeProfile(
    name="performance-cold",
    runtime_profile="native_mtp_60_cold",
    summary=(
        "Burst engine: native-MTP performance-cold path for the old max-fan "
        "headline lane. Use only for short contexts, recommended max 8K."
    ),
    env=_items(NATIVE_MTP_60_FAST_PATH_ENV),
    benchmark_ids=(
        "mtp-depth-d3-gdn8-speed4-cyankiwi-mtp-draftlmhead4b-gs64-linear-gdn-from-conv-tape-mlx-qmv-unroll4-clean-preflight-batchedtargetarrays-lazymtphistory-dropevents-skipsnapshot-v4-20260429-143701",
    ),
    caveats=(
        "Best fan-backed burst throughput; not recommended for long context.",
        "Runs on stock PyPI MLX; no custom MLX fork or build is required.",
    ),
    draft_lm_head=DraftLMHeadRequirement(bits=4, group_size=64, mode="affine"),
)

SUSTAINED_PROFILE = RuntimeProfile(
    name="sustained",
    runtime_profile="native_mtp_sustained",
    summary=(
        "Sustained Mode: explicit long-context native-MTP path with chunked "
        "contiguous prefill, final-token logits, and repaged decode KV."
    ),
    env=_items(SUSTAINED_PREFILL_ENV),
    caveats=(
        "Default product path for long-context coding and agent use.",
        "Targets long-context memory safety while preserving most Burst TPS.",
        "Does not include v0.2 decode-state eval scheduling flags.",
        "Runs on stock PyPI MLX; no custom MLX fork or build is required.",
    ),
    draft_lm_head=DraftLMHeadRequirement(bits=4, group_size=64, mode="affine"),
)

TURBO_PROFILE = RuntimeProfile(
    name="turbo",
    runtime_profile="native_mtp_turbo",
    summary=(
        "Sustained Mode plus verify-specialized quantized-matmul kernels "
        "(MTPLX_NAX_VERIFY) and the compiled verify step for 4-bit models "
        "(MTPLX_COMPILED_VERIFY, context-routed). Fastest decode profile."
    ),
    env=_merge_env(
        SUSTAINED_PREFILL_ENV,
        {
            "MTPLX_NAX_VERIFY": "1",
            "MTPLX_NAX_M4_IMPL": "vk_k",
            # Compiled verify (W2) promotion, 2026-07-04: default-on for the
            # turbo profile. Measured wins with the growth-demote +
            # shared-traces bank: Speed-4bit +8% bare / +12% @7k / +22% @12k;
            # Quality-q8 +10% bare / flat-to-+6% at 7k contexts (the 07-02
            # sprint's q8 -15/-18% was the since-removed per-request trace
            # tax). parity2 zero divergences on both trunks. The engine
            # self-gates per model (unmeasured quantizations such as the
            # 6-bit 9B stay eager) and contexts above the router fall back
            # per call.
            "MTPLX_COMPILED_VERIFY": "1",
            # 12288 -> 32768 (2026-08-14 dropday verify-wall calibration):
            # the 12288 fence predated donation A2.1 removing the full-attn
            # KV copy tax it guarded against. Gated ABBA on Qwen3.8 Bare:
            # compiled-at-32k beats the eager fallback at both 20k and 30k
            # rungs in every paired epoch (clean window +6.9% @20k), peak
            # memory flat-to-lower (the eager path spikes higher at 30k),
            # and parity2 shows identical comparator behavior to the ship
            # config (warmup ulp-level GDN capture diffs on BOTH, i.e. a
            # comparator artifact, not an extension regression).
            "MTPLX_COMPILED_VERIFY_MAX_CONTEXT": "32768",
            # Packed-GQA verify attention (speed-war Lane A, 2026-07-05):
            # one KV stream per simdgroup with all q=2..4 verify rows in
            # registers + a single float4 shuffle butterfly. Isolated
            # kernel: 207 vs 160 GB/s useful at 128k (+27%). Live gates:
            # 8-arm ABBA @64k warm decode +5.8% mean / verify -3.9ms
            # (fwd -4pt spread), paired 128k cold pair verify 192->152
            # ms/call, decode 14.4->18.6; acceptance-by-depth bands and
            # peak memory byte-identical. Engages only >= threshold and
            # only on dense-cache decode windows; bails to fused SDPA on
            # any contract miss.
            "MTPLX_GQA_PACKED_SDPA": "1",
            "MTPLX_GQA_PACKED_SDPA_THRESHOLD": "8192",
            # Background warmup ladder (F6, 2026-08-16; retrenched 2026-08-17
            # field regression fix). F6 shipped the full pow2 walk up to the
            # 32768 router fence in the PRODUCT profile so benchmark rows
            # would never pay a bucket's first-touch mx.compile. Field
            # fallout on 2.8.0-2.8.2: every desktop/serve boot burned
            # 30-60+ s of max GPU walking rungs users never reach in chat
            # (the reported "idle GPU pin"), rungs preempted by a first chat
            # re-queued and re-burned between turns, and a request landing
            # mid-rung saw its prefill contended (measured 166 vs ~800
            # prefill tok/s). The PRODUCT default is back to the two rungs
            # interactive chat actually touches early (boot cost ~4.5 s on
            # the 27B, 2.7.1-equivalent). Benchmark harnesses that want
            # every bucket pre-warmed set the deep ladder themselves via
            # operator env, which wins (PROFILE_ENV_USER_OVERRIDE_KEYS):
            #   MTPLX_WARMUP_LADDER=512,1024,2048,2560,4096,8192,16384,32768
            "MTPLX_WARMUP_LADDER": "512,2560",
        },
    ),
    caveats=(
        "Speculative-verify matmuls use the MTPLX verify_kernels family. "
        "4-bit (vk_k): argmax- and sampler-distribution-validated, not "
        "bit-exact vs stock. 8-bit (vk q8): ULP-exact vs stock.",
        "Prefill and non-speculative decode remain bit-identical to stock.",
        "4-bit and 8-bit affine models; 6-bit models silently run the "
        "stock path.",
        "Compiled verify engages on 4-bit and 8-bit affine trunks at "
        "contexts <= 32768 (parity2-validated on both; fence raised from "
        "12288 in 2.7.0); other quantizations/contexts run the eager verify "
        "path unchanged.",
        "Measured 2026-07-02/03 on M5 Max chat lane (app-launch flags, "
        "thinking on): 27B Optimized-Speed 44.7 -> 58-60 tok/s (vk_k "
        "within ~2% of the retired dflash-port kernel both directions); "
        "27B Optimized-Quality 31-36 -> 43-44 tok/s (+22-40%, q8 verify "
        "61-64 ms/call vs stock 81-93).",
        "Runs on stock PyPI MLX; the MTPLX verify kernels are shipped "
        "in-package Metal kernels, not an MLX fork or patched qmm build.",
    ),
    draft_lm_head=DraftLMHeadRequirement(bits=4, group_size=64, mode="affine"),
    product_claim_eligible=False,
)

EXACT_PROFILE = RuntimeProfile(
    name="exact",
    runtime_profile="exact",
    summary="QA-only exact paged verifier profile with release gates enabled.",
    env=_items(EXACT_PAGED_ATTENTION_ENV),
    qa_only=True,
    product_claim_eligible=False,
)

MAX_DIAGNOSTIC_PROFILE = RuntimeProfile(
    name="max-diagnostic",
    runtime_profile="max_diagnostic",
    summary=(
        "Diagnostic fan-control profile for QA-only experiments."
    ),
    env=_merge_env(EXACT_PAGED_ATTENTION_ENV, LONG_RESPONSE_STAGED_ENV),
    caveats=(
        "Requires explicit --max before fan control is allowed.",
        "Clock-anchor behavior is separate and experimental.",
    ),
    fan_control_allowed=True,
    clock_anchor_allowed=True,
    product_claim_eligible=False,
)

PROFILES: dict[ProfileName, RuntimeProfile] = {
    STABLE_PROFILE.name: STABLE_PROFILE,
    PERFORMANCE_COLD_PROFILE.name: PERFORMANCE_COLD_PROFILE,
    SUSTAINED_PROFILE.name: SUSTAINED_PROFILE,
    TURBO_PROFILE.name: TURBO_PROFILE,
    EXACT_PROFILE.name: EXACT_PROFILE,
    MAX_DIAGNOSTIC_PROFILE.name: MAX_DIAGNOSTIC_PROFILE,
}

PROFILE_ALIASES = {
    "default": "sustained",
    "safe": "stable",
    "native-mtp-60": "performance-cold",
    "native_mtp_60": "performance-cold",
    "native_mtp_60_cold": "performance-cold",
    "long_response_exact_staged": "stable",
    "max": "max-diagnostic",
    # The V1 app Settings picker briefly offered these as profile values,
    # so configs in the wild persist them. "auto" means the engine
    # default; "sustained-max" was sustained plus max fans, and fan mode
    # travels on --fan-mode, never inside the profile name.
    "auto": DEFAULT_PROFILE_NAME,
    "sustained-max": "sustained",
    "sustained_max": "sustained",
}


def resolve_profile_name(name: str | None) -> ProfileName:
    raw = (name or DEFAULT_PROFILE_NAME).strip()
    resolved = PROFILE_ALIASES.get(raw, raw)
    if resolved not in PROFILES:
        choices = ", ".join(PROFILE_CHOICES)
        raise ValueError(f"unknown MTPLX profile {raw!r}; expected one of: {choices}")
    return resolved


def get_profile(name: str | None = None) -> RuntimeProfile:
    return PROFILES[resolve_profile_name(name)]


def list_profiles() -> list[dict[str, object]]:
    return [PROFILES[name].to_dict() for name in PROFILE_CHOICES]


def apply_profile_env(
    name: str | None,
    *,
    environ: MutableMapping[str, str] | None = None,
    runtime_env_overrides: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    target = os.environ if environ is None else environ
    profile = get_profile(name)
    overrides = normalize_runtime_env_overrides(runtime_env_overrides)
    expected = profile.env_dict()
    expected.update(overrides)
    previous = {key: target.get(key) for key in expected}
    overridden: list[dict[str, str]] = []
    for key, value in profile.env:
        if key in PROFILE_ENV_USER_OVERRIDE_KEYS and str(target.get(key) or "").strip():
            actual = str(target.get(key))
            if actual != value:
                # Operator env beats the profile (by design). Record and
                # announce it (F23c): before this, profile_env_status said
                # ok:true and strict startup passed with zero trace that
                # the launched config was not the profile's.
                overridden.append(
                    {
                        "var": key,
                        "profile_value": value,
                        "actual_value": actual,
                    }
                )
                try:
                    print(
                        f"[mtplx] profile env override: {key}={actual} "
                        f"(profile {profile.name} default {value}; "
                        "operator env wins)",
                        flush=True,
                    )
                except Exception:
                    pass
            continue
        stomped = str(target.get(key) or "").strip()
        if stomped and stomped != value:
            # Profile-owned key: the profile replaces a different pre-set
            # env value. Announce the stomp — the silent version of this
            # is how operator A/Bs die (PR #314 had to patch site-packages
            # to get an env through). Keys meant to beat the profile
            # belong in PROFILE_ENV_USER_OVERRIDE_KEYS.
            try:
                print(
                    f"[mtplx] profile env stomp: {key}={stomped} replaced "
                    f"by profile {profile.name} value {value} "
                    f"({key} is profile-owned, not operator-overridable)",
                    flush=True,
                )
            except Exception:
                pass
        target[key] = value
    profile_env_overridden[:] = overridden
    for key, value in overrides.items():
        target[key] = value
    announce_runtime_gated_env(target, profile_name=profile.name)
    return previous


def restore_profile_env(
    previous: Mapping[str, str | None],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    target = os.environ if environ is None else environ
    for key, value in previous.items():
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value


def profile_env_status(
    name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    runtime_env_overrides: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, object]]:
    target = os.environ if environ is None else environ
    profile = get_profile(name)
    expected_env = profile.env_dict()
    expected_env.update(normalize_runtime_env_overrides(runtime_env_overrides))
    return {
        key: {
            "expected": expected_value,
            "observed": target.get(key),
            "override_allowed": key in PROFILE_ENV_USER_OVERRIDE_KEYS,
            # "ok" deliberately stays permissive for allowed overrides
            # (strict startup must keep passing); "overridden" is the F23c
            # truth bit — the operator env replaced the profile value.
            "overridden": bool(
                key in PROFILE_ENV_USER_OVERRIDE_KEYS
                and str(target.get(key) or "").strip()
                and target.get(key) != expected_value
            ),
            "ok": target.get(key) == expected_value
            or (
                key in PROFILE_ENV_USER_OVERRIDE_KEYS
                and bool(str(target.get(key) or "").strip())
            ),
        }
        for key, expected_value in expected_env.items()
    }
