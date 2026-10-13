"""One resolver, three entry points (PX.1, 2026-09-18).

The macOS app, ``mtplx start`` and bare ``mtplx serve`` each carried their
own copy of the launch lane, and the copies had drifted: Pi got the
``ar_batch`` scheduler from the app and ``serial`` from the CLI, the SSD
session-cache cap was a literal 100 GB on ``serve``, 32 GB on ``start
opencode`` and RAM-tiered from the app, ``start hermes`` forwarded stale
4.8 / 6.0 ms depth-policy priors and exported read-inspection limits the app
had removed, and a saved Adaptive depth switch sent the expected-value
policy to Flash-Next, where only verify width 4 has a compiled route.

This module owns the lane. Given (client, model family, memory, chip class,
entry point) it resolves: scheduler mode, prefill chunk, depth and policy,
SSD session cache and its cap, sampler, reasoning effort, agent env block.
The CLI start tables are generated from it, the server resolves the depth
policy through it, and the app's presets are tested against the matrix it
emits (``lane_matrix``; the Swift test reads the JSON fixture).

Pure data and arithmetic: no MLX import, no server import, no subprocess.
"""

from __future__ import annotations

from typing import Any

from mtplx.backends.family_settings import (
    compiled_verify_depths,
    family_settings,
    resolved_value,
)
from mtplx.runtime_options import DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS

GIB = 1024**3

ENTRY_POINTS: tuple[str, ...] = ("app", "start", "serve")
CLIENTS: tuple[str, ...] = (
    "chat",
    "openwebui",
    "opencode",
    "pi",
    "hermes",
    "other",
    "benchmark",
)
# Families whose sampler contract is their own (the official thinking-mode
# sampler 1.0 / 0.95 / 20): a client table never overrides it.
SAMPLER_OWNING_FAMILIES = frozenset({"qwen3_8", "qwen4_exp"})
SEATS_GIB: tuple[int, ...] = (16, 24, 36, 48, 64, 96, 128)

# --- the constants two owners used to disagree on ---------------------------

# Pi's scheduler lane. The app launched Pi on ar_batch / agent (2 slots, 50 ms
# batch wait), the CLI on serial / latency. Until the Pi cell (P5.0) says
# otherwise BOTH are serial: the OpenCode preset's own receipt measured the
# ar_batch agent lane at 36.8 against serial 51.4 decode tok/s at 8K (31.6
# against 42.4 at 33K) on the same daemon and model. One constant: flip it
# here and both entry points follow (the Swift preset mirrors it and the
# matrix test fails if they part).
PI_SCHEDULER_MODE = "serial"
PI_BATCHING_PRESET = "latency"

# SSD session cache cap: "auto" is the RAM-tiered cap
# (cold_tier.default_cold_tier_max_bytes). Literal sizes are gone from every
# table; a user's explicit size still wins.
SSD_SESSION_CACHE_MAX_SIZE = "auto"
SSD_SESSION_CACHE_MIN_PREFIX_TOKENS = DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS

# Session-bank entries for coding agents: 32 from 96 GiB of RAM, else 6.
AGENT_HIGH_MEMORY_THRESHOLD_BYTES = 96 * GIB
AGENT_BANK_ENTRIES_DEFAULT = "6"
AGENT_BANK_ENTRIES_HIGH_MEMORY = "32"

ADAPTIVE_NO_EFFECT_REASON = "adaptive depth has no effect on this model yet"

# SSD session cache ON/OFF still differs by entry point for two clients, and
# both data sets that read slow on 2026-09-17/18 ran with it off. These are
# DECLARED differences until cell P3.1 / P5.0 measures them; settle each by
# editing one value here.
SSD_SESSION_CACHE_BY_ENTRY: dict[str, dict[str, str]] = {
    "chat": {"app": "off", "start": "on", "serve": "on"},
    "openwebui": {"app": "off", "start": "on", "serve": "on"},
    "pi": {"app": "off", "start": "on", "serve": "on"},
    "benchmark": {"app": "off", "start": "off", "serve": "off"},
}

# (client, choice) pairs allowed to differ between entry points, with why.
DECLARED_ENTRY_POINT_DIFFERENCES: dict[tuple[str, str], str] = {
    ("chat", "ssd_session_cache"): "app chat ships it off, the CLI on: unmeasured, cell P3.1",
    ("openwebui", "ssd_session_cache"): "same owner as chat: unmeasured, cell P3.1",
    ("pi", "ssd_session_cache"): "app Pi ships it off, `mtplx start pi` on: unmeasured, cell P5.0",
}

_AGENT_CLIENTS = frozenset({"opencode", "pi", "hermes"})

# Per-client base lane. Sampler values apply only to families that do not
# own their sampler; None means "the family or the pack decides".
_CLIENT_LANES: dict[str, dict[str, Any]] = {
    "chat": {
        "scheduler_mode": "serial",
        "batching_preset": "solo",
    },
    "openwebui": {
        "scheduler_mode": "serial",
        "batching_preset": "solo",
    },
    "opencode": {
        "scheduler_mode": "serial",
        "batching_preset": "latency",
        "ssd_session_cache": "on",
        "depth": 3,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "tool_prompt_mode": "hybrid",
        "chat_template_profile": "local_qwen36",
    },
    "pi": {
        "scheduler_mode": PI_SCHEDULER_MODE,
        "batching_preset": PI_BATCHING_PRESET,
        "top_p": 0.95,
        "top_k": 20,
        "tool_prompt_mode": "hybrid",
        "chat_template_profile": "local_qwen36",
        "adaptive_policy": "expected_value",
        "preserve_thinking": "auto",
    },
    "hermes": {
        "scheduler_mode": "serial",
        "batching_preset": "latency",
        "ssd_session_cache": "on",
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": 20,
        "tool_prompt_mode": "hybrid",
        "chat_template_profile": "local_qwen36",
        "adaptive_policy": "expected_value",
        "preserve_thinking": "auto",
    },
    "other": {
        "scheduler_mode": "ar_batch",
        "batching_preset": "agent",
        "max_active_requests": 4,
        "decode_batch_max": 4,
        "batch_wait_ms": 50,
        "ssd_session_cache": "on",
    },
    "benchmark": {
        "scheduler_mode": "serial",
        "batching_preset": "latency",
        "ssd_session_cache": "off",
        "top_p": 0.95,
        "top_k": 20,
    },
}


def _choice(value: Any, source: str, reason: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"value": value, "source": source}
    if reason:
        item["reason"] = reason
    return item


def depth_policy_priors(family: str | None) -> dict[str, Any]:
    """The family's expected-value priors (server parser values when absent)."""

    value = resolved_value(family, "depth_policy_priors") or resolved_value(
        "qwen3_8", "depth_policy_priors"
    )
    return dict(value)


def resolve_depth_policy(
    family: str | None,
    *,
    requested_policy: str | None,
    requested_depth: int | None,
) -> dict[str, Any]:
    """Depth and policy the ENGINE serves for this family.

    A family whose verify has a compiled route for some draft depths only
    (``compiled_verify_depths``, Flash-Next: {3}) cannot gain from a depth
    policy: every round the policy stops early runs the eager verifier
    (measured 7 to 8 percent slower than fixed depth 3, and 28 eager rounds
    at the head of every request in the founder's Pi session). The engine
    resolves the request to the static compiled depth and says so. The rule
    is a capability, so it lifts by itself when more widths are compiled.
    """

    policy = str(requested_policy or "none").strip().lower() or "none"
    depth = int(requested_depth) if requested_depth else None
    compiled = compiled_verify_depths(family)
    if policy == "none" or compiled is None:
        return {
            "policy": policy,
            "depth": depth,
            "demoted": False,
            "reason": None,
            "compiled_verify_depths": sorted(compiled) if compiled else None,
        }
    eligible = sorted(d for d in compiled if depth is None or d <= depth)
    static_depth = eligible[-1] if eligible else depth
    return {
        "policy": "none",
        "depth": static_depth,
        "demoted": True,
        "reason": (
            f"{ADAPTIVE_NO_EFFECT_REASON}: only draft depth "
            f"{', '.join(str(d) for d in sorted(compiled))} has a compiled "
            f"verifier, so the engine serves static depth {static_depth}"
        ),
        "compiled_verify_depths": sorted(compiled),
    }


def agent_env_block(client: str, memory_bytes: int | None) -> dict[str, str]:
    """The coding-agent runtime env block, identical for every entry point.

    The read-inspection limits (``MTPLX_ACTIVE_READ_INSPECTION_*``,
    ``MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS``) are NOT here:
    the app removed them because an explicit export re-arms a compactor that
    rewrote agent transcripts (#282); ``mtplx start hermes`` still exported
    them until this resolver.
    """

    if client not in _AGENT_CLIENTS:
        return {}
    high_memory = (
        memory_bytes is not None and int(memory_bytes) >= AGENT_HIGH_MEMORY_THRESHOLD_BYTES
    )
    env = {
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE": "async_per_head",
        "MTPLX_SESSION_BLOCK_PREFIX_RESTORE": "1",
        "MTPLX_SESSION_BANK_MAX_ENTRIES": (
            AGENT_BANK_ENTRIES_HIGH_MEMORY if high_memory else AGENT_BANK_ENTRIES_DEFAULT
        ),
        "MTPLX_SESSION_BANK_MAX_BYTES": "auto",
        "MTPLX_SESSION_BANK_PER_SESSION_BYTES": "auto",
        "MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S": "30.0",
        "MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS": "4096",
        "MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER": "1",
        "MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE": "1",
        "MTPLX_TOOL_PROMPT_MODE": "hybrid",
        "MTPLX_CHAT_TEMPLATE_PROFILE": "local_qwen36",
    }
    if client == "hermes":
        env["MTPLX_CLIENT"] = "hermes"
    return env


def ssd_cap_bytes_for_seat(memory_bytes: int, *, free_disk_bytes: int | None = None) -> int:
    """What "auto" resolves to on a seat (mirror of cold_tier, for the matrix)."""

    ram = int(memory_bytes)
    if ram <= 16 * GIB:
        return 16 * GIB
    if ram <= 32 * GIB:
        return 24 * GIB
    if ram <= 64 * GIB:
        if free_disk_bytes is not None and free_disk_bytes >= 150 * GIB:
            return 100 * GIB
        return 32 * GIB
    return 100 * GIB


def resolve_launch_lane(
    *,
    client: str,
    family: str | None,
    memory_bytes: int | None,
    entry_point: str = "serve",
    chip_class: str | None = None,
    adaptive_depth_switch: bool | None = None,
) -> dict[str, Any]:
    """The launch lane for one (client, family, Mac, entry point).

    ``adaptive_depth_switch`` is the app's saved Adaptive depth switch:
    True asks for the expected-value policy on every target, False for none,
    None leaves the client preset in charge.
    """

    if client not in _CLIENT_LANES:
        raise ValueError(f"unknown client {client!r}")
    if entry_point not in ENTRY_POINTS:
        raise ValueError(f"unknown entry point {entry_point!r}")
    base = _CLIENT_LANES[client]
    family_key = str(family or "unknown")
    choices: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    choices["scheduler_mode"] = _choice(base["scheduler_mode"], "default")
    choices["batching_preset"] = _choice(base["batching_preset"], "default")
    for key in ("max_active_requests", "decode_batch_max", "batch_wait_ms"):
        choices[key] = _choice(base.get(key), "default")
    if client == "pi":
        choices["scheduler_mode"]["reason"] = (
            "serial measured faster than the agent batch lane on single-stream "
            "coding turns (51.4 against 36.8 tok/s at 8K); the Pi cell P5.0 decides"
        )

    # Prefill chunk: never a launch flag. The family block owns it.
    block = family_settings(family_key)
    if block is not None:
        setting = block["prefill_chunk_tokens"]
        choices["prefill_chunk_tokens"] = _choice(
            setting.value, "family", f"{setting.source}; not passed as a launch flag"
        )
    else:
        choices["prefill_chunk_tokens"] = _choice(
            2048, "default", "shared profile value; this family has no block of its own"
        )

    # Depth and policy.
    requested_policy = base.get("adaptive_policy", "none")
    policy_source = "default"
    if adaptive_depth_switch is True:
        requested_policy, policy_source = "expected_value", "app switch"
    elif adaptive_depth_switch is False:
        requested_policy, policy_source = "none", "app switch"
    requested_depth = base.get("depth") or 3
    resolved = resolve_depth_policy(
        family_key, requested_policy=requested_policy, requested_depth=requested_depth
    )
    if resolved["demoted"]:
        choices["adaptive_policy"] = _choice("none", "demoted", resolved["reason"])
        choices["depth"] = _choice(resolved["depth"], "demoted", "the compiled depth")
    else:
        choices["adaptive_policy"] = _choice(resolved["policy"], policy_source)
        choices["depth"] = _choice(
            resolved["depth"], "default" if base.get("depth") else "family"
        )
    if choices["adaptive_policy"]["value"] == "expected_value":
        choices["depth_policy_priors"] = _choice(depth_policy_priors(family_key), "family")

    # SSD session cache.
    by_entry = SSD_SESSION_CACHE_BY_ENTRY.get(client)
    ssd_mode = by_entry[entry_point] if by_entry else base.get("ssd_session_cache", "on")
    choices["ssd_session_cache"] = _choice(ssd_mode, "default")
    choices["ssd_session_cache_max_size"] = _choice(
        SSD_SESSION_CACHE_MAX_SIZE,
        "default",
        "scaled to this Mac's RAM"
        + (
            f": {ssd_cap_bytes_for_seat(memory_bytes) // GIB} GB"
            if memory_bytes
            else ""
        ),
    )
    choices["ssd_session_cache_min_prefix_tokens"] = _choice(
        SSD_SESSION_CACHE_MIN_PREFIX_TOKENS, "default"
    )

    # Sampler: family-owned families ignore the client table.
    family_owns_sampler = family_key in SAMPLER_OWNING_FAMILIES
    for key in ("temperature", "top_p", "top_k"):
        if family_owns_sampler:
            choices[key] = _choice(None, "family", "the model's own sampler contract")
        else:
            choices[key] = _choice(base.get(key), "default")

    # Reasoning effort: the family decides (Flash-Next xhigh, 27B medium).
    choices["reasoning_effort"] = _choice(None, "family")
    for key in ("tool_prompt_mode", "chat_template_profile", "preserve_thinking"):
        if base.get(key) is not None:
            choices[key] = _choice(base[key], "default")

    env = agent_env_block(client, memory_bytes)
    return {
        "client": client,
        "family": family_key,
        "entry_point": entry_point,
        "chip_class": chip_class,
        "memory_gib": (int(memory_bytes) // GIB) if memory_bytes else None,
        "choices": choices,
        "env": env,
        "notes": notes,
    }


def comparable_lane(lane: dict[str, Any]) -> dict[str, Any]:
    """The values of a lane, for equality across entry points."""

    return {
        **{key: item["value"] for key, item in lane["choices"].items()},
        "env": dict(lane["env"]),
    }


def cli_start_defaults(client: str) -> dict[str, Any]:
    """The ``mtplx start <client>`` defaults table, generated from the lane.

    Sampler values are the client table's: the serve-defaults pass replaces
    them with the family sampler for families that own one (see
    ``family_owned_sampler_overrides``).
    """

    base = _CLIENT_LANES[client]
    by_entry = SSD_SESSION_CACHE_BY_ENTRY.get(client)
    table: dict[str, Any] = {
        "scheduler_mode": base["scheduler_mode"],
        "batching_preset": base["batching_preset"],
        "max_active_requests": base.get("max_active_requests"),
        "decode_batch_max": base.get("decode_batch_max"),
        "batch_wait_ms": base.get("batch_wait_ms"),
        "prefill_chunk_tokens": None,
        "ssd_session_cache": by_entry["start"] if by_entry else base.get("ssd_session_cache", "on"),
        "ssd_session_cache_max_size": SSD_SESSION_CACHE_MAX_SIZE,
        "ssd_session_cache_min_prefix_tokens": SSD_SESSION_CACHE_MIN_PREFIX_TOKENS,
    }
    for key in ("temperature", "top_p", "top_k", "tool_prompt_mode", "chat_template_profile"):
        if base.get(key) is not None:
            table[key] = base[key]
    if base.get("adaptive_policy"):
        priors = depth_policy_priors(None)
        table.update(
            {
                "adaptive_policy": base["adaptive_policy"],
                "adaptive_min_depth": priors["min_depth"],
                "adaptive_ev_base_depth": priors["base_depth"],
                "adaptive_ev_warmup_full_depth_cycles": priors["warmup_cycles"],
                "adaptive_ev_exploration_interval": priors["explore_every"],
            }
        )
    if base.get("preserve_thinking"):
        table["reasoning"] = "auto"
        table["preserve_thinking"] = base["preserve_thinking"]
    return table


def family_owned_sampler_overrides(family: str | None) -> bool:
    """True when a client table's sampler must not reach this family."""

    return str(family or "") in SAMPLER_OWNING_FAMILIES


def lane_matrix(
    *,
    families: tuple[str, ...] = ("qwen4_exp", "qwen3_8"),
    seats_gib: tuple[int, ...] = SEATS_GIB,
    clients: tuple[str, ...] = CLIENTS,
) -> list[dict[str, Any]]:
    """Every (client, family, seat, entry point) lane, for the parity tests."""

    rows: list[dict[str, Any]] = []
    for family in families:
        for client in clients:
            for seat in seats_gib:
                for entry_point in ENTRY_POINTS:
                    lane = resolve_launch_lane(
                        client=client,
                        family=family,
                        memory_bytes=seat * GIB,
                        entry_point=entry_point,
                    )
                    rows.append(
                        {
                            "family": family,
                            "client": client,
                            "seat_gib": seat,
                            "entry_point": entry_point,
                            "lane": comparable_lane(lane),
                        }
                    )
    return rows


# Launch values the app's argv is compared on (the Swift parity test). The
# sampler and the reasoning effort are family-owned and the depth-policy
# REQUEST may differ (the engine resolves it), so they are not in this list.
APP_ARGV_KEYS: tuple[str, ...] = (
    "scheduler_mode",
    "batching_preset",
    "max_active_requests",
    "decode_batch_max",
    "batch_wait_ms",
    "ssd_session_cache",
    "ssd_session_cache_max_size",
    "ssd_session_cache_min_prefix_tokens",
)


def app_parity_fixture(
    *,
    families: tuple[str, ...] = ("qwen4_exp", "qwen3_8"),
    seats_gib: tuple[int, ...] = SEATS_GIB,
    clients: tuple[str, ...] = CLIENTS,
) -> dict[str, Any]:
    """The compact matrix the Swift test reads (entry point "app").

    Lane values do not depend on the seat; the agent env block does (6 or 32
    bank entries), so it is carried once per seat.
    """

    rows: list[dict[str, Any]] = []
    for family in families:
        for client in clients:
            for switch in (None, True, False):
                lane = resolve_launch_lane(
                    client=client,
                    family=family,
                    memory_bytes=128 * GIB,
                    entry_point="app",
                    adaptive_depth_switch=switch,
                )
                values = comparable_lane(lane)
                rows.append(
                    {
                        "family": family,
                        "client": client,
                        "adaptive_depth_switch": switch,
                        "argv": {key: values[key] for key in APP_ARGV_KEYS},
                        "effective_adaptive_policy": values["adaptive_policy"],
                        "effective_depth": values["depth"],
                        "prefill_chunk_is_a_launch_flag": False,
                        "agent_env": bool(lane["env"]),
                    }
                )
    return {
        "version": 1,
        "generated_by": "scripts/write_launch_lane_matrix.py",
        "pi_scheduler_mode": PI_SCHEDULER_MODE,
        "families_with_one_compiled_depth": sorted(
            family for family in families if compiled_verify_depths(family)
        ),
        "adaptive_no_effect_reason": ADAPTIVE_NO_EFFECT_REASON,
        "seats_gib": list(seats_gib),
        "agent_env_by_seat_gib": {
            str(seat): {
                client: agent_env_block(client, seat * GIB)
                for client in sorted(_AGENT_CLIENTS)
            }
            for seat in seats_gib
        },
        "declared_differences": [
            {"client": client, "choice": choice, "why": why}
            for (client, choice), why in sorted(DECLARED_ENTRY_POINT_DIFFERENCES.items())
        ],
        "rows": rows,
    }
