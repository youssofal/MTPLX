"""MLX's command-buffer size rule, kept for prefill and lifted for everything else.

MLX closes its open command buffer whenever the arrays the buffer has bound
add up to more than a fixed size (``mtplx/mlx_process_env.py`` has the rule
and its receipts).  The rule bounds the temporaries one buffer can pin, which
is what a prefill needs.  A decode step that runs on MLX's graph binds whole
weight tensors of gigabytes (a routed-expert gather, the draft head's layer,
a commit after a rejection), so the same rule closes a buffer after nearly
every launch: plain decode on Flash-Next made 389 command buffers a token for
828 launches (11 Nov 2026), each one a submission the GPU waits on.

MLX at the pinned version reads the limit once per process.  The native
module can set it while the process runs (``native/engine/limits.mm`` finds
the two numbers inside MLX's device object and proves the place before it
writes).  With that, a runtime that asks for it gets:

* from the start of an ``attention_phase("prefill")`` block until the next
  decode block begins: MLX's own limit, exactly as before, so prefill memory
  does not change.  The block's end is not the prefill's end: a chunk's forward
  is built inside the block and evaluated after it, and MLX encodes launches
  issued with ``async_eval`` later still.  A rule lifted at the block's exit
  reached those launches and a command buffer pinned gigabytes of a
  4,096-row forward's arrays instead of 50 MB (11 Nov 2026: MLX's peak 92.3
  against 87.1 GB on a 16K prompt, 96.7 against 92.0 on a 64K one);
* inside decode blocks (``decode_verify``, ``ar_decode``) and after them: the
  size limit lifted.  The count limit (operations a buffer) is never touched:
  it is what lets the GPU start on the first part of a step while the host
  encodes the rest.

Nothing changes when the native module is missing, when the place could not be
proven, when the operator set MLX's limit himself (``MLX_MAX_MB_PER_BUFFER`` or
``MTPLX_MLX_COMMAND_BUFFER_MB``), or when ``MTPLX_DECODE_BUFFER_RULE=mlx`` asks
for MLX's rule everywhere (the rollback).  Class A: which command buffer a
launch travels in changes no value.
"""

from __future__ import annotations

import os
from typing import Any

#: The size limit outside prefill, in MLX's unit (2^20 elements): no array set
#: a decode step binds reaches it.
LIFTED_MB = 1 << 20
ROLLBACK_ENV = "MTPLX_DECODE_BUFFER_RULE"

_STATE: dict[str, Any] = {
    "on": False,
    "reason": "not_asked",
    "mlx_mb": None,
    "ops": None,
    "prefill_depth": 0,
    "lifted": False,
    "switches": 0,
    "set": None,
}


def _native() -> Any | None:
    try:
        from . import runtime as _runtime

        if not _runtime.available():
            return None
        return getattr(_runtime.native_module(), "command_buffer_limits", None)
    except Exception:  # noqa: BLE001 - a missing module leaves MLX's rule alone
        return None


def _set(mb: int) -> bool:
    call = _STATE["set"]
    if call is None:
        return False
    try:
        call(0, int(mb))
    except Exception:  # noqa: BLE001 - never on the decode path
        _STATE.update(on=False, reason="set_failed")
        return False
    _STATE["switches"] += 1
    return True


def enable(reason: str = "runtime") -> dict[str, Any]:
    """Lift the size limit outside prefill for this process (idempotent)."""

    if _STATE["on"]:
        return report()
    if os.environ.get(ROLLBACK_ENV, "").strip().lower() in {"mlx", "0", "off", "false"}:
        _STATE.update(reason="rollback:" + ROLLBACK_ENV)
        return report()
    try:
        from .. import mlx_process_env

        source = str(mlx_process_env.resolve_command_buffer_mb()[1])
    except Exception:  # noqa: BLE001
        source = ""
    if source.startswith("operator:") or source.startswith("override:"):
        # The operator chose a limit for the whole process: his word stands.
        _STATE.update(reason="operator_set:" + source)
        return report()
    call = _native()
    if call is None:
        _STATE.update(reason="native_module_unavailable")
        return report()
    try:
        ops, mb, settable = call(0, 0)
    except Exception as exc:  # noqa: BLE001
        _STATE.update(reason="read_failed:" + repr(exc)[:80])
        return report()
    if not settable:
        _STATE.update(reason="place_not_proven", mlx_mb=int(mb), ops=int(ops))
        return report()
    # MLX's limit stays until the first decode block: a runtime's first work is a prefill.
    _STATE.update(on=True, reason=str(reason), mlx_mb=int(mb), ops=int(ops), set=call, lifted=False)
    return report()


def disable() -> None:
    """Give MLX its own limit back (tests, and a runtime that unloads)."""

    if _STATE["on"] and _STATE["mlx_mb"] and _STATE["lifted"]:
        _set(_STATE["mlx_mb"])
    _STATE.update(on=False, reason="disabled", set=None, lifted=False)


def prefill_enter() -> None:
    """A prefill block begins: MLX's own limit until a decode block begins after it."""

    _STATE["prefill_depth"] += 1
    if _STATE["on"] and _STATE["lifted"] and _set(_STATE["mlx_mb"]):
        _STATE["lifted"] = False


def prefill_exit() -> None:
    """The block ends; its chunk is evaluated after it, so the limit does not move here."""

    _STATE["prefill_depth"] = max(0, _STATE["prefill_depth"] - 1)


def decode_enter() -> None:
    """A decode block begins outside any prefill block: the size limit is lifted."""

    if _STATE["on"] and not _STATE["lifted"] and _STATE["prefill_depth"] == 0 and _set(LIFTED_MB):
        _STATE["lifted"] = True


def report() -> dict[str, Any]:
    """For /health: whether the rule is lifted outside prefill, and why not."""

    return {
        "lifted_outside_prefill": bool(_STATE["on"]),
        "reason": _STATE["reason"],
        "mlx_mb": _STATE["mlx_mb"],
        "ops": _STATE["ops"],
        "lifted_mb": LIFTED_MB if _STATE["on"] else None,
        "lifted_now": bool(_STATE["lifted"]),
        "switches": int(_STATE["switches"]),
    }
