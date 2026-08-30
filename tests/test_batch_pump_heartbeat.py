"""Every long-lived model-owner pump must tick the progress heartbeat.

The smart-fan activity probe (#201) now reads "busy" as "the owner heartbeat is
advancing", so any owner-thread pump that can run for minutes without ticking is
indistinguishable from a wedge. ``mtplx/generation.py`` already routes every
settled eval through a ticking ``_eval``; the batch lanes did not:

- ``mtplx/batched_decode.py`` settled six raw ``mx.eval`` calls per decode cycle
  with no tick at all,
- ``mtplx/a3b_mtp_batch.py`` ran an entire MTP cohort prefill + verify/commit
  driver on raw ``mx.eval``,
- ``_BatchedARGenerationService`` settled its shared-prefix prefill and its pump
  cycle raw, and its only heartbeat path (``record_batch_step``) fires solely
  when a decode step produced generation responses — never during prefill.

A healthy width-8 cohort prefill therefore read as frozen, and past
``FOREGROUND_STALL_DEADLINE_S`` the stale-lease reconciler would drop the fan
leases out from under live work — a direct R2 violation. These tests pin the
tick at the source: where the pumps settle their evals.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import mlx.core as mx

import mtplx.a3b_mtp_batch as a3b_mod
import mtplx.batched_decode as batched_decode_mod
import mtplx.server.openai as openai_mod
from mtplx import progress_heartbeat


def _raw_eval_lines(tree: ast.AST) -> list[int]:
    """Line numbers of settled ``mx.eval``/``_mx.eval`` calls in ``tree``.

    Calls lexically inside a ticking wrapper (a function named ``_eval`` or
    ``_owner_settled_eval``) are the sanctioned settle point and are exempt.
    ``mx.async_eval`` is not a settle point and is not counted.
    """

    exempt: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_eval",
            "_owner_settled_eval",
        }:
            for inner in ast.walk(node):
                exempt.add(id(inner))
    lines: list[int] = []
    for node in ast.walk(tree):
        if id(node) in exempt:
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "eval"
            and isinstance(func.value, ast.Name)
            and func.value.id in {"mx", "_mx"}
        ):
            lines.append(node.lineno)
    return sorted(lines)


def _module_tree(module) -> ast.AST:
    return ast.parse(Path(inspect.getsourcefile(module)).read_text(encoding="utf-8"))


def _class_tree(module, class_name: str) -> ast.AST:
    tree = _module_tree(module)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in {module.__name__}")


def test_batched_decode_settled_eval_ticks_the_owner_heartbeat():
    before = progress_heartbeat.value()
    batched_decode_mod._eval(mx.array([1]))
    assert progress_heartbeat.value() > before


def test_mtp_cohort_settled_eval_ticks_the_owner_heartbeat():
    before = progress_heartbeat.value()
    a3b_mod._eval(mx.array([1]))
    assert progress_heartbeat.value() > before


def test_ar_batch_settled_eval_ticks_the_owner_heartbeat():
    before = progress_heartbeat.value()
    openai_mod._owner_settled_eval(mx.array([1]))
    assert progress_heartbeat.value() > before


def test_batched_decode_has_no_untracked_settled_eval():
    assert _raw_eval_lines(_module_tree(batched_decode_mod)) == []


def test_mtp_cohort_driver_has_no_untracked_settled_eval():
    assert _raw_eval_lines(_module_tree(a3b_mod)) == []


def test_ar_batch_service_has_no_untracked_settled_eval():
    """Scoped to the AR batch service: other owners of ``mx.eval`` in this
    module (embeddings, reranking) run on request threads, where a tick would
    forge owner liveness and blind the #86 stream stall watchdog."""
    assert _raw_eval_lines(_class_tree(openai_mod, "_BatchedARGenerationService")) == []


class _StubState:
    """Minimal stand-in exposing exactly what the fan activity probe reads."""

    def __init__(self, *, stall_probe):
        self._fan_stall_probe = stall_probe
        self.model_scheduler = None
        self.last_request_started_at = 0.0
        self.last_request_at = 0.0

    def has_foreground(self) -> bool:
        return True


def test_a_long_batch_cohort_that_settles_evals_keeps_its_fan_boost():
    """R2 with the REAL probe and the REAL tick path — no fakes.

    Wall time far past the deadline is not a stall as long as the pump settles
    evals. Before the pumps ticked, this cohort read as wedged and lost its
    leases.
    """

    now = [0.0]
    probe = openai_mod._OwnerStallProbe(deadline_s=100.0, clock=lambda: now[0])
    state = _StubState(stall_probe=probe)
    fan_probe = openai_mod.ServerState._smart_fan_activity_probe

    assert fan_probe(state) is True

    # A genuinely wedged owner: no eval settles for well past the deadline.
    now[0] = 500.0
    assert fan_probe(state) is False

    # A healthy cohort: 8 minutes of pump cycles, each settling one eval.
    for _ in range(8):
        a3b_mod._eval(mx.array([1]))
        now[0] += 60.0
        assert fan_probe(state) is True, "long batch work must keep its fan boost"


# --- Forged progress: an empty pump step is not progress -------------------
#
# ``BatchGenerator.next()`` can return two empty response lists: a transient
# library step, or a pump whose ``_active`` map has desynchronised from the
# generator. Ticking the owner heartbeat there forges liveness — the pump can
# spin forever while continuously resetting BOTH the #86 stream stall watchdog
# and the #201 fan activity probe. Streams get nothing, nothing ever aborts,
# and the fan leases stay pinned: exactly the failure this branch exists to
# contain. Only a settled prompt or generation response proves a step ran.


class _FakeResponse:
    def __init__(self, uid: int = 1):
        self.uid = uid


def test_an_empty_ar_pump_step_does_not_advance_the_owner_heartbeat():
    before = progress_heartbeat.value()
    openai_mod._owner_settled_pump_step([], [])
    assert progress_heartbeat.value() == before


def test_a_prompt_response_proves_a_settled_pump_step():
    before = progress_heartbeat.value()
    openai_mod._owner_settled_pump_step([_FakeResponse()], [])
    assert progress_heartbeat.value() > before


def test_a_generation_response_proves_a_settled_pump_step():
    before = progress_heartbeat.value()
    openai_mod._owner_settled_pump_step([], [_FakeResponse()])
    assert progress_heartbeat.value() > before


def test_the_pump_never_ticks_outside_the_proven_step_gate():
    """Wiring: the pump body must reach the heartbeat only through the gate.

    A bare ``_owner_settled_eval(...)`` anywhere in ``_pump`` ticks
    unconditionally — ``mx.eval([])`` settles nothing, so the tick would be
    pure fabrication.
    """

    pump = _class_tree(openai_mod, "_BatchedARGenerationService")
    pump_body = next(
        node
        for node in ast.walk(pump)
        if isinstance(node, ast.FunctionDef) and node.name == "_pump"
    )
    called = {
        node.func.id
        for node in ast.walk(pump_body)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_owner_settled_eval" not in called
    assert "_owner_settled_pump_step" in called


def test_a_pump_spinning_on_empty_steps_is_eventually_detected_as_stalled():
    """The regression, end to end on the real probe: a pump that produces no
    responses must age into a stall rather than pinning the fans forever."""

    now = [0.0]
    probe = openai_mod._OwnerStallProbe(deadline_s=100.0, clock=lambda: now[0])
    state = _StubState(stall_probe=probe)
    fan_probe = openai_mod.ServerState._smart_fan_activity_probe

    assert fan_probe(state) is True
    for _ in range(8):
        openai_mod._owner_settled_pump_step([], [])
        now[0] += 60.0
    assert fan_probe(state) is False, "an empty-spinning pump must read stalled"
