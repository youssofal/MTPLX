"""Every environment switch of the native decode engine, in one place.

Nothing here has to be set.  The engine and its dispatch form are on by
themselves wherever the compiled module loads and passes its start check, from
the app, ``mtplx start`` and bare ``mtplx serve`` alike: no launch surface sets
any of these names, so the three resolve the same lane.

What is left is of three kinds:

``rollback``    turns one default-on part back to the path it replaced, to
                isolate a fault or to measure the part.
``diagnostic``  records, checks, dumps or bounds; never needed to serve.
``lever``       a bit-identical variant that has no row on a real pack yet.
                Off.  It becomes the default by an edit of the code after it
                is measured, or it is deleted; it is never a setting users
                are asked to know.

``set_now()`` is what /health reports, so a process that runs with any of
them set says so.
"""

from __future__ import annotations

import os
from typing import NamedTuple


class Switch(NamedTuple):
    name: str
    kind: str  # "rollback", "diagnostic" or "lever"
    unset: str  # what the process does with the name unset
    what: str


SWITCHES: tuple[Switch, ...] = (
    # ------------------------------------------------------------- rollback
    Switch(
        "MTPLX_NATIVE_ENGINE", "rollback", "on wherever the compiled module loads",
        "0: the MLX path alone, the module is never loaded and the dense-lane bank is the exact one; "
        "shadow: MLX serves and every round is also replayed and compared",
    ),
    Switch(
        "MTPLX_DECODE_BUFFER_RULE", "rollback", "MLX's command-buffer size rule inside prefill only",
        "mlx: MLX's own size rule everywhere, as before (mtplx/engine/buffer_limits.py)",
    ),
    Switch(
        "MTPLX_NATIVE_ENGINE_REPLAY", "rollback", "replays",
        "0: record, build and dump programs but never replay one",
    ),
    Switch(
        "MTPLX_NATIVE_ENGINE_DISPATCH", "rollback", "hybrid",
        "concurrent or serial: how a replay orders its launches (same bits in every form)",
    ),
    Switch(
        "MTPLX_NATIVE_ENGINE_BARRIERS", "rollback", "minimal",
        "recorded, named or passes: where a replay places its barriers (same bits in every form)",
    ),
    Switch(
        "MTPLX_NATIVE_ENGINE_SPLIT", "rollback", "256",
        "launches per command buffer while the rest is encoded; 0: one command buffer",
    ),
    # ----------------------------------------------------------- diagnostic
    Switch("MTPLX_NATIVE_ENGINE_MODULE", "diagnostic", "the packaged module", "path of a built module to load instead"),
    Switch("MTPLX_NATIVE_ENGINE_DUMP", "diagnostic", "nothing is written", "folder for recorded programs and mismatches"),
    Switch("MTPLX_NATIVE_ENGINE_DUMP_DATA", "diagnostic", "off", "1: keep constant bytes in a dumped program"),
    Switch("MTPLX_NATIVE_ENGINE_DIAGNOSE", "diagnostic", "off", "1 with a dump folder: on a mismatch, name the launch"),
    Switch("MTPLX_NATIVE_ENGINE_PROFILE", "diagnostic", "off", "1 with a dump folder: per-launch GPU times, once per program"),
    Switch("MTPLX_NATIVE_ENGINE_MIXED", "diagnostic", "off", "<seed>: serve live rounds from either path at random (QA)"),
    Switch(
        "MTPLX_NATIVE_ENGINE_QUIET_TOKENS", "diagnostic", "96",
        "tokens of an answer before a recording or a check (unset: 8 for a step that has served live)",
    ),
    Switch("MTPLX_NATIVE_ENGINE_WARM_CALLS", "diagnostic", "2", "calls of a new geometry MLX serves before one is recorded"),
    Switch("MTPLX_NATIVE_ENGINE_PROGRAMS", "diagnostic", "96", "most programs a site keeps"),
    Switch("MTPLX_NATIVE_ENGINE_PROGRAM_MB", "diagnostic", "256", "memory a site's programs may own, in MiB"),
    Switch("MTPLX_NATIVE_ENGINE_MIN_RUN", "diagnostic", "4", "shortest run of dependent launches that gets a serial pass"),
    Switch("MTPLX_NATIVE_ENGINE_COPY_INLINE", "diagnostic", "off", "1: send inline constants as bytes on every launch, as MLX does"),
    Switch("MTPLX_NATIVE_ENGINE_TEST_LOSE_REPLAY", "diagnostic", "off", "N[:code]: report the Nth finished replay as lost (tests)"),
    # ---------------------------------------------------------------- lever
    Switch("MTPLX_NATIVE_ENGINE_ENCODE_EARLY", "lever", "off", "1: encode a replay before MLX's queue is empty, commit after"),
    Switch("MTPLX_NATIVE_ENGINE_SPIN_US", "lever", "0", "microseconds before a replay is due from which its wait watches instead of sleeping"),
)

NAMES = frozenset(switch.name for switch in SWITCHES)


def set_now() -> dict[str, str]:
    """The switches of the table that are set in this process, with their values."""

    return {
        switch.name: os.environ[switch.name]
        for switch in SWITCHES
        if os.environ.get(switch.name, "").strip()
    }


__all__ = ["NAMES", "SWITCHES", "Switch", "set_now"]
