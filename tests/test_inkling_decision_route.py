"""Who decides an Inkling round, and the first token's guard.

* MTPLX_INKLING_FAST=off restores RC6's whole path: its forward and its host decision.
* A request with client controls (logit_bias, min_p, repetition_penalty) takes the device
  round, with the controls applied to the rows it decides from; target_prefix keeps RC6's
  seed-match draws on the host.
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


def test_target_prefix_takes_the_host_and_client_controls_the_device(device_round):
    assert loop.decision_route(NATIVE, verify_strategy="target_prefix") == (False, "target_prefix")
    controlled = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, min_p=0.05,
                               repetition_penalty=1.1, logit_bias=((3, -100.0),))
    assert loop.decision_route(controlled) == (True, None)


def test_wide_copy_decision_is_off_until_switched_on(monkeypatch):
    # Copy panels past 8 rows stay host-decided by default (class B against the host's draw).
    monkeypatch.delenv("MTPLX_INKLING_WIDE_DECISION", raising=False)
    assert not loop.fast_round.wide_decision_enabled()
    for value, on in (("1", True), ("on", True), ("0", False), ("off", False)):
        monkeypatch.setenv("MTPLX_INKLING_WIDE_DECISION", value)
        assert loop.fast_round.wide_decision_enabled() is on


def test_client_controls_shape_the_first_token_row(monkeypatch):
    # The device draw reads the shaped row; the error census would read the model's row.
    seen = {}

    def checked(row, **k):
        seen["row"] = row
        return mx.array([[2], [0]], mx.int32)

    monkeypatch.setattr(loop.fast_round, "draw_checked", checked)
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, logit_bias=((0, -100.0),))
    controls = shared.ClientControlledDraw(config, [])
    row = mx.arange(4, dtype=mx.float32)[::-1].reshape(1, 4)
    assert loop.checked_device_draw(row, position=0, seed=1, temperature=1.0, controls=controls) == (2, True)
    assert seen["row"].tolist() == [[3.0 - 100.0, 2.0, 1.0, 0.0]]


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


def _cost_rt(round_ms, width=8):
    return SimpleNamespace(drafter=SimpleNamespace(width=width), round_warm={"round_ms": round_ms},
                           kernel_proofs=SimpleNamespace(directory=None))


def _pieces(asked, one_row=24.0, per=6.0, commit=1.5):
    """A fake ``measure_round_costs``: k = 0 always, the asked counts (all when None) priced
    from the pieces, the others None; the asked lists recorded."""
    def measure(rt, proposals=None, **_):
        asked.append(None if proposals is None else list(proposals))
        timed = range(1, rt.drafter.width) if proposals is None else proposals
        raw = tuple(one_row + per * k if k == 0 or k in timed else None for k in range(rt.drafter.width))
        return raw, {"commit_observe_ms": commit, "reps": 5}
    return measure


def test_round_costs_come_from_the_programs_own_warm_rounds_this_load(monkeypatch):
    from mtplx.models.inkling.fast import decoder

    asked = []
    monkeypatch.setattr(decoder, "measure_round_costs", _pieces(asked))
    monkeypatch.delenv("MTPLX_INKLING_ROUND_COMMIT", raising=False)  # deciding round: commit added
    program = {1: 27.0, 2: 31.2, 3: 35.1, 4: 40.0, 5: 45.2, 6: 50.6, 7: 56.3}
    table, detail = decoder.round_costs(_cost_rt(program))
    assert asked == [[]]  # only the one-row round from the pieces: the program served every count
    assert detail["source"] == "program_warm" and detail["refused"] == [] and table[0] == 24.0
    fit = detail["fit"]
    assert table[1:] == tuple(decoder._half_ms(fit["table"][k] + 1.5) for k in range(1, 8))
    assert all(b > a for a, b in zip(table[1:], table[2:]))
    # The committing round's program already commits: nothing added.
    monkeypatch.setenv("MTPLX_INKLING_ROUND_COMMIT", "1")
    committed, detail2 = decoder.round_costs(_cost_rt(program))
    assert detail2["commit_added_ms"] == 0.0 and committed[1] == decoder._half_ms(fit["table"][1])
    # Every load times again: nothing is read back from, or written to, the proof directory.
    assert len(asked) == 2


def test_round_costs_price_a_count_the_program_did_not_serve_from_its_pieces(monkeypatch):
    from mtplx.models.inkling.fast import decoder

    asked = []
    monkeypatch.setattr(decoder, "measure_round_costs", _pieces(asked))
    program = {1: 27.0, 2: 31.2, 4: 40.0, 5: 45.2, 6: 50.6, 7: 56.3}  # 3 refused at warm-up
    table, detail = decoder.round_costs(_cost_rt(program))
    assert asked == [[3]] and detail["source"] == "program_warm" and detail["pieces_for"] == [3]
    assert table[3] == 24.0 + 6.0 * 3


def test_a_table_not_plausible_against_the_one_row_round_is_refused(monkeypatch):
    from mtplx.models.inkling.fast import decoder

    # Bare Speed's kept table of 21 November: 105 ms and 3 a proposal beside a one-row round
    # of about 24 ms. Refused, said, and the pieces' line from this load serves instead.
    asked = []
    monkeypatch.setattr(decoder, "measure_round_costs", _pieces(asked))
    table, detail = decoder.round_costs(_cost_rt({k: 105.0 + 3 * k for k in range(1, 8)}))
    assert detail["source"] == "pieces" and asked == [[], None]
    assert detail["refused"][0]["source"] == "program_warm" and "one-row" in detail["refused"][0]["reason"]
    assert decoder.implausible(table) is None
    # Pieces that are not plausible either: the fixed shape from the one-row round (a fifth of
    # it a proposal), with both refusals said.
    monkeypatch.setattr(decoder, "measure_round_costs", _pieces([], one_row=24.0, per=40.0))
    table2, detail2 = decoder.round_costs(_cost_rt({k: 105.0 + 3 * k for k in range(1, 8)}))
    assert detail2["source"] == "fixed_shape" and len(detail2["refused"]) == 2
    assert table2 == decoder.fixed_shape(24.0, 8) and table2[:3] == (24.0, 29.0, 33.5)
    assert decoder.implausible((20.0, 25.0, 30.0)) is None
    assert decoder.implausible((20.0, 70.0, 73.0)) and decoder.implausible((20.0, 25.0, 50.0))


@pytest.mark.parametrize("pack, program, one_row, commit", [
    # L5's gate item 1 on RC2 (21 November, a busy load): least squares through inflated high
    # widths priced a 1-proposal round below the one-row round (native, Bare Speed) or gave a
    # slope a third of the one-row round (Optimized Speed), and every round served 1 proposal.
    ("native", {1: 25, 2: 28, 3: 32, 4: 36, 5: 45, 6: 83, 7: 86}, 30.95, 2.94),
    ("bare", {1: 20, 2: 24, 3: 30, 4: 37, 5: 57, 6: 67, 7: 69}, 20.09, 2.05),
    ("optimized_speed", {1: 23, 2: 27, 3: 36, 4: 50, 5: 59, 6: 58, 7: 61}, 21.05, 2.26),
])
def test_the_degenerate_tables_of_a_busy_load_are_refused_and_the_pieces_serve(monkeypatch, pack, program,
                                                                              one_row, commit):
    from mtplx.models.inkling.fast import decoder

    asked = []
    monkeypatch.setattr(decoder, "measure_round_costs", _pieces(asked, one_row=one_row, per=4.0, commit=commit))
    monkeypatch.delenv("MTPLX_INKLING_ROUND_COMMIT", raising=False)
    table, detail = decoder.round_costs(_cost_rt({k: float(v) for k, v in program.items()}))
    assert detail["refused"][0]["source"] == "program_warm"
    assert detail["source"] == "pieces" and asked == [[], None]
    assert table[1] >= table[0] and decoder.implausible(table) is None
    assert all(b >= a for a, b in zip(table[1:], table[2:]))


def test_a_fresh_loads_table_is_accepted(monkeypatch):
    from mtplx.models.inkling.fast import decoder

    # Bare Speed's fresh load on 93afa7f8 (l0/receipts/depth-93afa7f8.json): served as fitted.
    monkeypatch.setattr(decoder, "measure_round_costs", _pieces([], one_row=16.12, per=4.0, commit=1.87))
    monkeypatch.delenv("MTPLX_INKLING_ROUND_COMMIT", raising=False)
    program = {1: 20.0, 2: 24.0, 3: 28.0, 4: 31.0, 5: 35.0, 6: 40.0, 7: 44.0}
    table, detail = decoder.round_costs(_cost_rt(program))
    assert detail["source"] == "program_warm" and detail["refused"] == []
    assert table == (16.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0)


def test_served_rounds_off_the_table_are_said_once():
    from mtplx.models.inkling.fast import decoder, routes

    table = (21.0, 28.5, 32.0, 35.5, 39.0, 42.5, 46.0, 49.5)
    near = decoder.DepthTableCheck(table)
    for _ in range(decoder.DepthTableCheck.ROUNDS):
        near.observe(3, 0.0337)            # 33.7 ms against 35.5: within a quarter
    assert near.verdict["off"] == {} and near.verdict["by_count"][3]["rounds"] == 300
    far = decoder.DepthTableCheck((31.0, 18.0, 29.0, 40.0, 51.0, 62.0, 73.0, 84.0))
    with routes.capture_counts() as seen:
        for _ in range(decoder.DepthTableCheck.ROUNDS):
            far.observe(1, 0.030)          # every round at 1 proposal, 30 ms against a table's 18
        far.observe(1, 0.030)              # judged once: later rounds change nothing
    assert seen == {"depth_table.off_by": 1}
    assert far.verdict["off"][1]["ratio"] == 1.667 and far.report()["rounds"] == 300


def test_the_committing_round_switch_is_one_for_the_loop_and_the_warm_up(monkeypatch):
    from mtplx.models.inkling.fast.decoder import round_commit_enabled

    monkeypatch.delenv("MTPLX_INKLING_ROUND_COMMIT", raising=False)
    assert not round_commit_enabled()  # off by default: the deciding round and the stock commit
    for off in ("0", "off", "false", "no", " OFF "):
        monkeypatch.setenv("MTPLX_INKLING_ROUND_COMMIT", off)
        assert not round_commit_enabled()
    for on in ("1", "on", "true", " ON "):
        monkeypatch.setenv("MTPLX_INKLING_ROUND_COMMIT", on)
        assert round_commit_enabled()
