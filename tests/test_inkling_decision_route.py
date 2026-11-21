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
    # The draw's own status row (round.draw_checked): a row with NaN or no finite maximum.
    def checked(row, **k):
        finite = bool(mx.logical_not(mx.logical_or(mx.any(mx.isnan(row)), mx.isinf(mx.max(row)))).item())
        return mx.array([[7], [0 if finite else 1]], mx.int32)

    monkeypatch.setattr(loop.fast_round, "draw_checked", checked)
    row = mx.zeros((1, 16), mx.float32)
    row = {"nan": mx.where(mx.arange(16) == 3, mx.nan, row),
           "-inf": mx.full((1, 16), -mx.inf),
           "+inf": mx.where(mx.arange(16) == 5, mx.inf, row)}[bad]
    with pytest.raises(NonFiniteLogitsError):
        loop.checked_device_draw(row, position=0, seed=1, temperature=1.0)


def test_first_token_of_a_finite_row(monkeypatch):
    row = mx.arange(16, dtype=mx.float32).reshape(1, 16)
    monkeypatch.setattr(loop.fast_round, "draw_checked", lambda r, **k: mx.array([[7], [0]], mx.int32))
    assert loop.checked_device_draw(row, position=0, seed=1, temperature=1.0) == (7, True)


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


def test_round_costs_are_fitted_smooth_and_coarse():
    from mtplx.models.inkling.fast.decoder import fit_round_costs

    noisy = (28.0, 38.1, 43.3, 49.6, 57.0, 63.4, 73.7, 77.0)  # increments 10.1 5.2 6.3 7.4 6.4 10.3 3.3
    fit = fit_round_costs(noisy)
    steps = [b - a for a, b in zip(fit["table"], fit["table"][1:])]
    assert fit["model"] == "linear" and len(set(steps)) == 1 and steps[0] == fit["per_row_ms"]
    assert fit["base_ms"] == int(fit["base_ms"]) and fit["per_row_ms"] * 2 == int(fit["per_row_ms"] * 2)
    # Re-timed with noise of a few percent, the controller makes the same choice at every rate,
    # and every proposal count is reachable (the raw timings never let it pick 4 or 6).
    from mtplx.models.inkling.fast.draft import DepthByYield

    def choices(table):
        out = []
        for rate in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            control = DepthByYield(cost_ms=table, max_depth=7, channel_prior={}, prior=rate)
            out.append(control.choose())
        return out

    jitter = tuple(v * f for v, f in zip(noisy, (1.01, 0.99, 1.0, 1.01, 0.99, 1.0, 1.01, 0.99)))
    assert choices(fit_round_costs(jitter)["table"]) == choices(fit["table"])
    picked = set(choices(fit["table"]))
    assert picked >= {2, 3, 7} and choices(fit["table"]) == sorted(choices(fit["table"]))
    # A real step past 5 rows (the 6th to 8th rows twice the price) gets its own slope.
    stepped = tuple(20 + 7 * k + (16 * max(0, k - 4)) for k in range(8))
    bent = fit_round_costs(stepped)
    assert bent["model"] == "linear_with_step" and bent["step_ms_past_4"] == 16.0


def test_round_costs_are_kept_for_the_pack_build_and_mac(tmp_path, monkeypatch):
    from mtplx.models.inkling.fast import decoder

    timed = []
    raw = (28.0, 38.1, 43.3, 49.6, 57.0, 63.4, 73.7, 77.0)
    monkeypatch.setattr(decoder, "measure_round_costs", lambda rt: (timed.append(1), (raw, {"reps": 5}))[1])
    monkeypatch.setattr(decoder.engine_site, "health", lambda rt=None: {"available": True})
    rt = SimpleNamespace(kernel_proofs=SimpleNamespace(directory=tmp_path / "proofs-identity"),
                         drafter=SimpleNamespace(width=8))
    first, detail = decoder.round_costs(rt)
    second, detail2 = decoder.round_costs(rt)  # the next load: read back, not re-timed
    assert first == second and detail["source"] == "timed" and detail2["source"] == "kept" and len(timed) == 1
    monkeypatch.setenv("MTPLX_INKLING_FAST_STAGES", "bc,e")  # another lane set: timed again
    _, detail3 = decoder.round_costs(rt)
    assert detail3["source"] == "timed" and len(timed) == 2 and detail3["replaced"]
