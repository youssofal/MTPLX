"""The machine matrix for the memory planner (issue #305).

Two families of pins:

* **No-regression pins** — the 128 GB M5 Max class must resolve exactly
  what ships today (context 262,144; session bank at the 48 GiB founder
  cap, idle AND steady), so the planner cannot slow the machines that
  were already correct.
* **Fit pins** — the 36/48/64 GB tiers must resolve geometry where
  weights + full-window KV + transients + the bank floor actually fit
  the engine's Metal envelope. These are the tiers issue #305 killed:
  a 48 GB Mac walked to 61.8 GB resident and 3-4 tok/s because nothing
  ever summed the budgets.

Byte values are pinned as integers on purpose: the arithmetic is exact,
and a drifted constant should fail loudly, not fuzzily.
"""

from __future__ import annotations

import os

import pytest

from mtplx.memory_plan import (
    BANK_CAP_BYTES,
    BANK_FLOOR_BYTES,
    CONTEXT_FLOOR_TOKENS,
    DEFAULT_DENSE_KV_BYTES_PER_TOKEN,
    GIB,
    bank_dynamic_ceiling,
    describe_plan,
    detect_total_ram_bytes,
    plan_memory,
    usable_engine_bytes,
)

# On-disk safetensors sizes (what engine_session._model_weights_bytes
# measures): the flagship Speed pack and the 8-bit Quality sibling.
SPEED_WEIGHTS = 21_313_949_792
QUALITY_WEIGHTS = 30_600_000_000

RAM = {gib: gib * GIB for gib in (36, 48, 64, 96, 128, 192, 512)}

# The shipped dense-decode ceilings for the machines under test (the
# generation.py auto formula: max(131072, 15% RAM / 65536 B/token)).
DENSE_CEILING = {36: 131_072, 48: 131_072, 64: 157_286, 96: 235_929, 128: 314_572}

FLAGSHIP_MAX_CONTEXT = 262_144


def _plan(ram_gib: int, weights: int = SPEED_WEIGHTS, **kwargs):
    kwargs.setdefault("model_max_context", FLAGSHIP_MAX_CONTEXT)
    kwargs.setdefault("dense_decode_ceiling", DENSE_CEILING.get(ram_gib))
    return plan_memory(
        total_ram_bytes=RAM[ram_gib],
        model_weights_bytes=weights,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# usable envelope


def test_usable_envelope_matches_metal_cap_default() -> None:
    assert usable_engine_bytes(RAM[128]) == 96 * GIB
    assert usable_engine_bytes(RAM[48]) == 36 * GIB
    # 512 GiB boxes hit the 192 GiB allocator cap, not the 75% rule.
    assert usable_engine_bytes(RAM[512]) == 192 * GIB
    # Tiny machines: the 8 GiB floor cannot exceed physical RAM.
    assert usable_engine_bytes(8 * GIB) == 8 * GIB


# ---------------------------------------------------------------------------
# the no-regression pins (128 GB class)


def test_128g_speed_resolves_exactly_todays_geometry() -> None:
    plan = _plan(128)
    assert plan.available and plan.model_fits
    assert plan.context_window_resolved == 262_144
    assert not plan.context_machine_bound
    # Bank at the founder cap, idle AND under full-window load: the
    # planner must not shrink the machine that was already correct.
    assert plan.bank_idle_max_bytes == BANK_CAP_BYTES
    assert plan.bank_steady_bytes == BANK_CAP_BYTES


def test_128g_quality_keeps_the_full_cap_too() -> None:
    plan = _plan(128, weights=QUALITY_WEIGHTS)
    assert plan.context_window_resolved == 262_144
    assert plan.bank_idle_max_bytes == BANK_CAP_BYTES
    assert plan.bank_steady_bytes == BANK_CAP_BYTES


# ---------------------------------------------------------------------------
# the fit pins (the #305 tiers)


def test_48g_speed_fit() -> None:
    plan = _plan(48)
    assert plan.available and plan.model_fits
    # 36G engine budget - 19.85G weights - 3G transients - 1G bank floor
    # leaves 12.15G of KV -> 196,608 tokens. Machine-bound, honestly.
    assert plan.context_window_resolved == 196_608
    assert plan.context_machine_bound
    # Idle bank stays aggressive (~13.1G, same class as the shipped
    # half-surplus 14.05G) ...
    assert plan.bank_idle_max_bytes == 14_119_530_400
    # ... and the advertised under-load budget subtracts KV at the dense
    # ceiling (131,072 x 64 KiB = 8 GiB) so the machine can never be
    # walked into swap by its own warm cache.
    assert plan.bank_steady_bytes == 5_529_595_808
    total = (
        plan.model_weights_bytes
        + plan.kv_reserve_bytes
        + plan.bank_steady_bytes
        + 3 * GIB
    )
    assert total <= plan.usable_bytes


def test_36g_speed_fit() -> None:
    plan = _plan(36)
    assert plan.context_window_resolved == 49_152
    assert plan.context_machine_bound
    assert plan.bank_idle_max_bytes == 4_455_853_984


def test_48g_quality_is_honest_about_the_8bit_pack() -> None:
    plan = _plan(48, weights=QUALITY_WEIGHTS)
    assert plan.model_fits
    assert plan.context_window_resolved == 57_344
    assert plan.context_machine_bound


def test_36g_quality_does_not_fit_and_says_so() -> None:
    plan = _plan(36, weights=QUALITY_WEIGHTS)
    assert not plan.model_fits
    assert plan.context_window_resolved == CONTEXT_FLOOR_TOKENS
    assert any("does not fit" in note for note in plan.notes)


def test_64g_speed_keeps_the_full_model_window() -> None:
    plan = _plan(64)
    assert plan.context_window_resolved == 262_144
    assert not plan.context_machine_bound


def test_q8_unlocks_the_full_window_on_48g() -> None:
    plan = _plan(48, kv_quantization="q8")
    assert plan.context_window_resolved == 262_144
    assert plan.kv_bytes_per_token_effective == 36_044


def test_unknown_quant_value_plans_at_full_bytes() -> None:
    plan = _plan(48, kv_quantization="weird")
    assert plan.kv_bytes_per_token_effective == DEFAULT_DENSE_KV_BYTES_PER_TOKEN


# ---------------------------------------------------------------------------
# invariants across the matrix


@pytest.mark.parametrize("weights", [SPEED_WEIGHTS, QUALITY_WEIGHTS])
def test_fit_is_monotonic_in_ram(weights: int) -> None:
    fits = [
        _plan(gib, weights=weights).context_window_fit for gib in (36, 48, 64, 96, 128)
    ]
    assert fits == sorted(fits)


@pytest.mark.parametrize("ram_gib", [36, 48, 64, 96, 128])
@pytest.mark.parametrize("weights", [SPEED_WEIGHTS, QUALITY_WEIGHTS])
def test_steady_state_always_fits_the_envelope(ram_gib: int, weights: int) -> None:
    plan = _plan(ram_gib, weights=weights)
    if not plan.model_fits:
        return
    committed = (
        plan.model_weights_bytes
        + plan.kv_reserve_bytes
        + plan.bank_steady_bytes
        + 3 * GIB
    )
    assert committed <= plan.usable_bytes
    assert BANK_FLOOR_BYTES <= plan.bank_steady_bytes <= plan.bank_idle_max_bytes


def test_context_is_block_aligned_or_model_capped() -> None:
    for gib in (36, 48, 64, 96, 128):
        plan = _plan(gib)
        resolved = plan.context_window_resolved
        assert resolved == FLAGSHIP_MAX_CONTEXT or resolved % 4096 == 0


# ---------------------------------------------------------------------------
# overrides and the budget knob


def test_explicit_context_override_wins_but_is_flagged() -> None:
    plan = _plan(48, requested_context=262_144)
    assert plan.context_window_resolved == 262_144
    assert plan.context_overcommitted
    assert any("exceeds the machine fit" in note for note in plan.notes)


def test_override_within_fit_is_not_flagged() -> None:
    plan = _plan(48, requested_context=131_072)
    assert plan.context_window_resolved == 131_072
    assert not plan.context_overcommitted


def test_memory_budget_simulates_the_smaller_seat() -> None:
    native = _plan(48)
    simulated = plan_memory(
        total_ram_bytes=RAM[128],
        model_weights_bytes=SPEED_WEIGHTS,
        model_max_context=FLAGSHIP_MAX_CONTEXT,
        dense_decode_ceiling=DENSE_CEILING[48],
        memory_budget_bytes=RAM[48],
    )
    assert simulated.context_window_resolved == native.context_window_resolved
    assert simulated.bank_idle_max_bytes == native.bank_idle_max_bytes
    assert simulated.bank_steady_bytes == native.bank_steady_bytes


def test_usable_override_takes_the_configured_metal_limit() -> None:
    plan = _plan(128, usable_bytes_override=30 * GIB)
    assert plan.usable_bytes == 30 * GIB


def test_budget_beats_the_real_machines_metal_limit() -> None:
    # Simulating a 48G seat on a 128G box: the caps were configured for
    # the real machine (96G); the budgeted formula (36G) must win.
    plan = _plan(
        128, memory_budget_bytes=RAM[48], usable_bytes_override=96 * GIB
    )
    assert plan.usable_bytes == 36 * GIB


def test_explicit_metal_limit_is_the_engine_budget_both_ways() -> None:
    # Issue #443: a 96 GB seat serving a 69.2 GiB pack. The default 75%
    # envelope (72 GiB) refuses it as "does not fit", yet the operator's
    # explicit MTPLX_MEMORY_LIMIT_BYTES=80G runs it. An explicit limit is
    # the engine budget, bounded by the machine.
    weights = int(69.2 * GIB)
    refused = _plan(96, weights=weights, usable_bytes_override=72 * GIB)
    assert not refused.model_fits
    assert refused.usable_source == "formula"
    assert "MTPLX_MEMORY_LIMIT_BYTES" in " ".join(refused.notes)
    plan = _plan(
        96, weights=weights, usable_bytes_override=80 * GIB, usable_bytes_explicit=True
    )
    assert plan.usable_bytes == 80 * GIB
    assert plan.model_fits
    assert plan.usable_source == "metal_limit"
    assert "engine budget 80.0G (Metal limit)" in describe_plan(plan)
    assert "MTPLX_MEMORY_LIMIT_BYTES" not in " ".join(plan.notes)
    capped = _plan(
        96, weights=weights, usable_bytes_override=120 * GIB, usable_bytes_explicit=True
    )
    assert capped.usable_bytes == RAM[96]


def test_default_metal_limit_stays_under_the_formula() -> None:
    # The server passes its default cap as the override too; only an
    # operator-set limit may raise the envelope.
    plan = _plan(96, usable_bytes_override=80 * GIB)
    assert plan.usable_bytes == 72 * GIB
    assert plan.usable_source == "formula"


def test_explicit_limit_yields_to_a_tighter_budget() -> None:
    # A simulated 48G seat on a 128G box keeps the budgeted formula even
    # when the real machine carries an explicit, larger limit.
    plan = _plan(
        128,
        memory_budget_bytes=RAM[48],
        usable_bytes_override=100 * GIB,
        usable_bytes_explicit=True,
    )
    assert plan.usable_bytes == 36 * GIB
    assert plan.usable_source == "formula"


# ---------------------------------------------------------------------------
# the dynamic ceiling (the guard that turns on)


def test_dynamic_ceiling_idle_grants_the_full_idle_max() -> None:
    plan = _plan(48)
    assert bank_dynamic_ceiling(plan, 0) == plan.bank_idle_max_bytes


def test_dynamic_ceiling_yields_as_live_kv_grows() -> None:
    plan = _plan(48)
    assert bank_dynamic_ceiling(plan, 10 * GIB) == 3_382_112_160
    # Extreme live load clamps at the floor, never below.
    assert bank_dynamic_ceiling(plan, 14 * GIB) == BANK_FLOOR_BYTES


def test_dynamic_ceiling_never_exceeds_idle_max() -> None:
    plan = _plan(128)
    assert bank_dynamic_ceiling(plan, 0) <= plan.bank_idle_max_bytes


def test_dynamic_ceiling_without_a_plan_stays_out_of_the_way() -> None:
    unavailable = plan_memory(total_ram_bytes=None, model_weights_bytes=SPEED_WEIGHTS)
    assert not unavailable.available
    assert bank_dynamic_ceiling(unavailable, 10 * GIB) == BANK_CAP_BYTES


def test_transient_reserve_tracks_the_observed_spike() -> None:
    from mtplx.memory_plan import (
        RUNTIME_TRANSIENTS_BYTES,
        TRANSIENT_RESERVE_CAP_BYTES,
        transient_reserve_bytes,
    )

    # Below the static floor: the floor wins (idle process, tiny spike).
    assert (
        transient_reserve_bytes(84 * GIB, 83 * GIB) == RUNTIME_TRANSIENTS_BYTES
    )
    # The measured 2026-08-29 shape: 95.2G peak over 82.8G active = 12.4G —
    # the reserve must carry the real spike, not the 3 GiB guess.
    spike = transient_reserve_bytes(int(95.2 * GIB), int(82.8 * GIB))
    assert spike == int(95.2 * GIB) - int(82.8 * GIB)
    # One pathological turn cannot starve the bank forever: capped.
    assert (
        transient_reserve_bytes(120 * GIB, 80 * GIB)
        == TRANSIENT_RESERVE_CAP_BYTES
    )
    # Tight-play models (Flash-Next: 77G weights on a 96G limit leaves
    # ~18G): the reserve never takes more than half the post-weights play,
    # so a deep turn's lifetime spike cannot permanently eat the bank.
    assert transient_reserve_bytes(
        96 * GIB, 80 * GIB, play_bytes=18 * GIB
    ) == 9 * GIB
    # The play cap never pushes the reserve below the static floor.
    assert transient_reserve_bytes(
        96 * GIB, 80 * GIB, play_bytes=4 * GIB
    ) == RUNTIME_TRANSIENTS_BYTES


def test_dynamic_ceiling_observed_reserve_shrinks_the_bank_first() -> None:
    plan = _plan(48)
    # 9 GiB more reserve (12 observed vs 3 static) comes straight out of
    # the bank's ceiling — identical to 9 GiB more live working set — so
    # entries demote BEFORE the next spike can kiss the Metal limit. The
    # identity form survives the idle-max and floor clamps.
    observed = bank_dynamic_ceiling(plan, 2 * GIB, transient_bytes=12 * GIB)
    assert observed == bank_dynamic_ceiling(plan, 11 * GIB)
    # The shrink actually bites on this plan (not clamped to the floor).
    assert observed < bank_dynamic_ceiling(plan, 2 * GIB)
    # A reserve below the static floor never RAISES the ceiling.
    assert bank_dynamic_ceiling(
        plan, 2 * GIB, transient_bytes=1 * GIB
    ) == bank_dynamic_ceiling(plan, 2 * GIB)


# ---------------------------------------------------------------------------
# unavailability, serialization, description


def test_unavailable_reasons_are_explicit() -> None:
    assert (
        plan_memory(total_ram_bytes=None, model_weights_bytes=1).unavailable_reason
        == "total_ram_unknown"
    )
    assert (
        plan_memory(total_ram_bytes=RAM[48], model_weights_bytes=None)
        .unavailable_reason
        == "model_weights_unknown"
    )


def test_to_dict_round_trips_the_essentials() -> None:
    payload = _plan(48).to_dict()
    assert payload["available"] is True
    assert payload["context_window_resolved"] == 196_608
    assert payload["bank_idle_max_bytes"] == 14_119_530_400
    assert isinstance(payload["notes"], list)


def test_describe_plan_names_the_machine_bound() -> None:
    line = describe_plan(_plan(48))
    assert "machine-bound" in line
    assert "yields to" in line


def test_detect_total_ram_reports_this_machine() -> None:
    detected = detect_total_ram_bytes()
    # Hosted macOS runners expose 7 GiB. Compare with the OS total instead
    # of assuming a minimum machine size unrelated to the probe contract.
    expected = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert detected is not None and detected > 0
    assert detected == expected
