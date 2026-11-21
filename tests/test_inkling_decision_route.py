"""Who decides an Inkling round, and the first token's guard.

* MTPLX_INKLING_FAST=off restores RC6's whole path: its forward and its host decision.
* A request with client controls (logit_bias, min_p, repetition_penalty) takes the host
  decision, which applies them; target_prefix keeps RC6's seed-match draws.
* The GPU's first-token draw refuses a logits row with NaN or no finite maximum, raising the
  existing numerical error before the id is counted, streamed or banked.
* The depth-one API forwards the verify strategy.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import generation as shared
from mtplx.models.inkling import generation as loop
from mtplx.sampling import NonFiniteLogitsError, SamplerConfig

NATIVE = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0)


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture
def device_round(monkeypatch):
    monkeypatch.setattr(loop.fast_round, "supported", lambda *a, **k: (True, None))
    monkeypatch.delenv("MTPLX_INKLING_FAST", raising=False)


def test_default_request_takes_the_device_round(device_round):
    assert loop.decision_route(NATIVE) == (True, None)


def test_rollback_switch_restores_the_host_decision(device_round, monkeypatch):
    monkeypatch.setenv("MTPLX_INKLING_FAST", "off")
    assert loop.decision_route(NATIVE) == (False, "fast_path_off")


def test_target_prefix_and_client_controls_take_the_host(device_round):
    assert loop.decision_route(NATIVE, verify_strategy="target_prefix") == (False, "target_prefix")
    controlled = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, min_p=0.05)
    controls = shared.ClientControlledDraw(controlled, [1, 2])
    assert loop.decision_route(controlled, controls=controls) == (False, "client_controls")


@pytest.mark.parametrize("bad", ["nan", "-inf", "+inf"])
def test_first_token_refuses_an_invalid_row(monkeypatch, bad):
    monkeypatch.setattr(loop.fast_round, "draw", lambda row, **k: mx.array([7], mx.int32))
    row = mx.zeros((1, 16), mx.float32)
    row = {"nan": mx.where(mx.arange(16) == 3, mx.nan, row),
           "-inf": mx.full((1, 16), -mx.inf),
           "+inf": mx.where(mx.arange(16) == 5, mx.inf, row)}[bad]
    with pytest.raises(NonFiniteLogitsError):
        loop.checked_device_draw(row, position=0, seed=1, temperature=1.0)


def test_first_token_of_a_finite_row_and_a_reported_validity(monkeypatch):
    row = mx.arange(16, dtype=mx.float32).reshape(1, 16)
    monkeypatch.setattr(loop.fast_round, "draw", lambda r, **k: mx.array([7], mx.int32))
    assert loop.checked_device_draw(row, position=0, seed=1, temperature=1.0) == (7, True)
    monkeypatch.setattr(loop.fast_round, "draw", lambda r, **k: (mx.array([7], mx.int32), mx.array([False])))
    with pytest.raises(NonFiniteLogitsError):
        loop.checked_device_draw(row, position=0, seed=1, temperature=1.0)


def test_depth_one_api_forwards_the_verify_strategy(monkeypatch):
    seen = {}
    monkeypatch.setattr(shared, "reject_non_k1_a3b_whole_moe_request", lambda *a, **k: None)
    monkeypatch.setattr(shared, "generate_mtpk", lambda rt, ids, **kw: seen.update(kw) or "out")
    rt = SimpleNamespace(backend_id="inkling2_loop")
    assert shared.generate_mtp1(rt, [1], max_tokens=2, sampler=NATIVE, verify_strategy="target_prefix") == "out"
    assert seen["verify_strategy"] == "target_prefix" and seen["speculative_depth"] == 1


def test_telemetry_reports_the_route_that_served_the_proposals():
    host = SimpleNamespace(sampled=False)
    assert loop.served_proposals(False, 1.0, host, 12) == {"proposal_route": "host_argmax"}
    assert loop.served_proposals(True, 1.0, host, 12) == {
        "proposal_route": "device_sampled", "method": "sampled", "temperature": 1.0, "sampled_tokens": 12}
    assert loop.served_proposals(True, 0.0, host, 12)["method"] == "argmax"


def test_public_stats_carry_the_fast_path_counters():
    from mtplx.server import openai

    assert "inkling_fast" in openai.PUBLIC_MTPLX_STATS_KEYS


def _inkling_state(depth=3):
    from mtplx.backends.descriptors import INKLING_DESCRIPTOR

    return SimpleNamespace(runtime=SimpleNamespace(backend_id="inkling2_loop"), args=SimpleNamespace(depth=depth),
                           backend_descriptor=INKLING_DESCRIPTOR)


@pytest.mark.parametrize("depth,explicit,env,mode,auto", [
    (3, False, None, "mtp", True),       # the launch default: the controller chooses each round
    (3, True, None, "mtp", False),       # a client's depth wins
    (5, False, None, "mtp", False),      # a launch depth other than the default wins
    (3, False, "fixed", "mtp", False),   # pinned by the switch
    (3, False, None, "ar", False),       # no drafts in AR
])
def test_inkling_depth_is_chosen_per_round_unless_pinned(monkeypatch, depth, explicit, env, mode, auto):
    from mtplx.server import openai

    if env:
        monkeypatch.setenv("MTPLX_INKLING_DEPTH", env)
    else:
        monkeypatch.delenv("MTPLX_INKLING_DEPTH", raising=False)
    monkeypatch.setattr(openai, "_backend_descriptor", lambda state: state.backend_descriptor)
    state = _inkling_state(depth)
    assert openai._inkling_auto_depth(state, mode, 0 if mode == "ar" else depth,
                                      {"request_depth_explicit": explicit}) is auto
