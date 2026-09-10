"""The WIRING of ``MTPLX_QSA_SPARSE_DECODE``, not its arithmetic.

``tests/test_qsa_sparse_decode.py`` pins the kernel's selection model,
its split geometry and its parity gates.  This file pins the thing that
actually failed on 2026-09-02: the lane was armed, the cache installed, and
the kernel never ran, because the ONE call site that could reach the verify
width asked about a width the flag never armed.

THE DEFECT, reproduced here on the CPU stream with stub shapes.

A fixed-capacity S=4 verify falls through ``legacy_fused=False`` into
``_select_eager``, whose sparse-decode question named the M=1 width.  It read
a zero row count, returned False, and handed attention the rows-gather lane.

Every test below is host-side.  Nothing evaluates an MLX array, and no test
touches the GPU: the routing decision this file is about is decided entirely
from python ints and cache attributes, which is exactly why it could go wrong
without leaving a mark on a receipt.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.kernels import qsa_sparse_decode as lane
from mtplx.models.qwen4_exp import Attention, QSAIndexer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_lane():
    lane.reset_for_tests()
    yield
    lane.reset_for_tests()


@pytest.fixture
def armed_verify(monkeypatch):
    """Arm the verify width only, exactly as the 2026-09-02 window did."""

    from mtplx import runtime_options

    monkeypatch.setattr(runtime_options, "_QSA_SPARSE_DECODE", True)
    return runtime_options


#: A 16 K prompt reserves 1,024 decode tokens and rounds to ratio: the bank
#: the 2026-09-02 window ran on.  ``17408 // 4 = 4352 > 512`` -- servable.
LONG_BANK = 17408
#: The 1 K cell of the served battery: a 1,024-token prompt plus the same
#: 1,024-token reserve.  ``2048 // 4 == 512``, NOT > 512, so the kernel's own
#: dense/sparse boundary refuses it -- while ``k_eff`` is a full 512.
DENSE_BANK = 2048


def fixed_cache(*, decode_rows: int = 4, bank: int = LONG_BANK) -> SimpleNamespace:
    """The state a ``TensorOffsetQSACache`` hands the routing predicate.

    ``kv.keys`` is the fixed K/V backing.  Its token dimension is what
    ``update_and_fetch`` returns and therefore what the attention call site
    passes as ``total_tokens`` -- the predicate must read the same number, so
    the stub carries it.
    """

    return SimpleNamespace(
        fixed_capacity=True,
        qsa_sparse_decode_rows=int(decode_rows),
        kv=SimpleNamespace(keys=SimpleNamespace(shape=(1, 2, int(bank), 256))),
    )


def indexer(*, ratio: int = 4, block_topk: int = 512) -> SimpleNamespace:
    """The two module attributes the predicate reads, bound to the real method."""

    stub = SimpleNamespace(ratio=int(ratio), block_topk=int(block_topk))
    stub._sparse_decode_route = QSAIndexer._sparse_decode_route.__get__(stub)
    return stub


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------
def test_the_verify_width_routes_on_the_stack_that_measured_the_control(
    armed_verify,
):
    """rows=4, verify armed: the kernel must serve."""

    assert indexer()._sparse_decode_route(
        fixed_cache(), rows=4, k_eff=512, site="select_eager_verify"
    )
    counters = lane.route_counters()
    assert counters["route_hits"] == 1
    assert counters["route_sites"] == {"select_eager_verify": 1}


def test_the_eager_selector_asks_the_verify_width():
    """``_select_eager`` is the selector a fixed-M4 verify actually reaches."""

    source = inspect.getsource(QSAIndexer._select_eager)
    calls = re.findall(r"_sparse_decode_route\((.*?)\)", source, re.S)
    assert len(calls) == 1, calls
    assert "select_eager_verify" in calls[0]


def test_every_route_call_site_names_itself():
    """A single total would not have shown which site declined."""

    tree = ast.parse((ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute) and func.attr == "_sparse_decode_route"
        ):
            continue
        keywords = {kw.arg for kw in node.keywords}
        assert "site" in keywords, ast.dump(node)
        for kw in node.keywords:
            if kw.arg == "site":
                sites.append(kw.value.value)
    assert sorted(sites) == ["select_eager_verify"]


# ---------------------------------------------------------------------------
# Routing vs failure -- the two kinds of "no"
# ---------------------------------------------------------------------------
def test_an_unarmed_width_is_one_cached_bool_test(armed_verify):
    """The OFF path must not do bookkeeping: it runs per QSA layer per forward."""

    assert not indexer()._sparse_decode_route(
        fixed_cache(), rows=1, k_eff=512, site="select_eager_verify"
    )
    assert lane.route_counters()["route_declines"] == {}
    assert lane.route_counters()["route_hits"] == 0


def test_the_off_path_reads_the_flag_before_anything_else():
    """An unarmed process pays one cached-bool test."""

    source = inspect.getsource(QSAIndexer._sparse_decode_route)
    body = source[source.index('"""', source.index('"""') + 3) + 3 :]
    flag = body.index("qsa_sparse_decode_enabled()")
    assert flag < body.index("getattr(cache")
    assert flag < body.index("from mtplx.kernels import qsa_sparse_decode")


def test_a_growable_cache_is_routing_and_is_counted(armed_verify):
    growable = SimpleNamespace(fixed_capacity=False)
    assert not indexer()._sparse_decode_route(
        growable, rows=4, k_eff=512, site="select_eager_verify"
    )
    assert lane.route_counters()["route_declines"] == {
        "select_eager_verify: growable cache": 1
    }


def test_a_width_the_predicate_was_not_asked_about_returns_quietly(armed_verify):
    """A 16 K prefill row count is neither width; no decline, no hit."""

    assert not indexer()._sparse_decode_route(
        fixed_cache(), rows=16384, k_eff=512, site="select_eager_verify"
    )
    assert lane.route_counters()["route_declines"] == {}


def test_a_cache_built_without_the_lane_raises_at_the_armed_width(armed_verify):
    with pytest.raises(RuntimeError, match="qsa_sparse_decode_rows=0"):
        indexer()._sparse_decode_route(
            fixed_cache(decode_rows=0),
            rows=4,
            k_eff=512,
           
            site="select_eager_verify",
        )


def test_a_geometry_the_metallib_is_not_built_for_raises(armed_verify):
    with pytest.raises(RuntimeError, match="ratio-4 top-512"):
        indexer(ratio=2)._sparse_decode_route(
            fixed_cache(), rows=4, k_eff=512, site="select_eager_verify"
        )


# ---------------------------------------------------------------------------
# REQUEST SHAPE: routing, not failure.  Two production 500s live here.
# ---------------------------------------------------------------------------
def test_a_partial_budget_routes_to_stock_instead_of_returning_500(armed_verify):
    """The first regression: HumanEval prompts, a few hundred tokens."""

    assert not indexer()._sparse_decode_route(
        fixed_cache(bank=LONG_BANK), rows=4, k_eff=33,
        site="select_eager_verify",
    )
    counters = lane.route_counters()
    assert counters["route_hits"] == 0
    assert counters["request_declines"] == 1
    assert counters["route_declines"] == {"select_eager_verify: partial_budget": 1}


def test_a_1k_prompt_in_the_dense_regime_routes_to_stock(armed_verify):
    """The second regression, and the one a full budget hides.

    The 1 K cell of the 2026-09-02 served battery: a 1,024-token prompt in a
    2,048-token fixed bank. ``k_eff`` is a FULL 512 -- the budget gate passes
    -- but ``2048 // 4 == 512`` is not ``> 512``, so the kernel refused the
    call from inside ``attention()`` and the server returned 500. The two
    questions are different and the predicate must ask both.
    """

    assert not indexer()._sparse_decode_route(
        fixed_cache(bank=DENSE_BANK), rows=4, k_eff=512,
        site="select_eager_verify",
    )
    counters = lane.route_counters()
    assert counters["route_hits"] == 0
    assert counters["request_declines"] == 1
    assert counters["route_declines"] == {"select_eager_verify: short_context": 1}
    assert counters["request_decline_extremes"]["tokens_min"] == DENSE_BANK


def test_the_dense_regime_boundary_is_the_kernels_own(armed_verify):
    """One token of bank either side of the native contract's boundary."""

    route = indexer()._sparse_decode_route
    assert not route(
        fixed_cache(bank=2048), rows=4, k_eff=512, site="s"
    )
    assert not route(
        fixed_cache(bank=2051), rows=4, k_eff=512, site="s"
    )
    assert route(
        fixed_cache(bank=2052), rows=4, k_eff=512, site="s"
    )
    assert lane.route_counters()["route_hits"] == 1
    assert lane.SHORT_CONTEXT_TOKENS == 2052


def test_the_crossover_inside_one_request_declines_then_binds(armed_verify):
    """Context growing from below the boundary to above it, one cache.

    A fixed bank grows by capacity transition, and each transition re-traces
    and re-asks. Before the boundary the stock chain is correct; after it the
    kernel must bind -- and the earlier declines must not have poisoned it.
    """

    cache = fixed_cache(bank=DENSE_BANK)
    route = indexer()._sparse_decode_route
    assert not route(cache, rows=4, k_eff=512, site="select_eager_verify")
    # The capacity transition: same cache object, a wider bank.
    cache.kv.keys.shape = (1, 2, 4096, 256)
    assert route(cache, rows=4, k_eff=512, site="select_eager_verify")
    counters = lane.route_counters()
    assert counters["route_hits"] == 1
    assert counters["request_declines"] == 1


def test_a_full_budget_forward_still_binds_after_either_decline(armed_verify):
    """Short requests must not poison the long-context arm."""

    route = indexer()._sparse_decode_route
    assert not route(
        fixed_cache(bank=LONG_BANK), rows=4, k_eff=33, site="s"
    )
    assert not route(
        fixed_cache(bank=DENSE_BANK), rows=4, k_eff=512, site="s"
    )
    assert route(
        fixed_cache(bank=LONG_BANK), rows=4, k_eff=512, site="s"
    )
    counters = lane.route_counters()
    assert counters["route_hits"] == 1
    assert counters["request_declines"] == 2


def test_every_request_shape_reason_routes_rather_than_raises(armed_verify):
    """No branch of the mirror may be a raise."""

    route = indexer()._sparse_decode_route
    cases = {
        "short_context": dict(bank=DENSE_BANK, k_eff=512),
        "partial_budget": dict(bank=LONG_BANK, k_eff=33),
        "rows_exceed_context": dict(bank=2, k_eff=512),
        "empty_context": dict(bank=0, k_eff=512),
    }
    for reason, kwargs in cases.items():
        lane.reset_for_tests()
        assert not route(
            fixed_cache(bank=kwargs["bank"]), rows=4, k_eff=kwargs["k_eff"],
            site="s",
        ), reason
        assert lane.route_counters()["route_declines"] == {f"s: {reason}": 1}


def test_the_decline_key_stays_bounded_while_the_numbers_track(armed_verify):
    """One key per (site, reason); the tokens and blocks ride as min/max.

    A server sees every context length there is, so a key carrying the count
    would grow without bound.
    """

    route = indexer()._sparse_decode_route
    for blocks in (7, 33, 511, 33):
        assert not route(
            fixed_cache(bank=LONG_BANK), rows=4, k_eff=blocks,
            site="select_eager_verify",
        )
    counters = lane.route_counters()
    assert counters["route_declines"] == {"select_eager_verify: partial_budget": 4}
    assert counters["request_declines"] == 4
    extremes = counters["request_decline_extremes"]
    assert extremes["blocks_min"] == 7 and extremes["blocks_max"] == 511
    assert extremes["tokens_min"] == extremes["tokens_max"] == LONG_BANK
    assert counters["short_context_tokens"] == 2052


def test_the_threshold_is_the_full_budget_the_abi_needs():
    assert lane.SHORT_CONTEXT_TOKENS == (lane.TOP_K + 1) * lane.COMPRESS_RATIO == 2052


def test_a_request_shape_decline_prints_nothing(armed_verify, capsys):
    """Once per QSA layer per request: a print here is a log flood."""

    indexer()._sparse_decode_route(
        fixed_cache(bank=DENSE_BANK), rows=4, k_eff=512,
        site="select_eager_verify",
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# The mirror against the native contract
# ---------------------------------------------------------------------------
NATIVE = (ROOT / "mtplx" / "native" / "__init__.py").read_text()


def native_decode_reasons():
    """Every string ``qsa_sparse_gqa_decode_unsupported_reason`` can return."""

    tree = ast.parse(NATIVE)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "qsa_sparse_gqa_decode_unsupported_reason"
    )
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out.append(value.value)
    return out


def test_every_request_shape_reason_the_kernel_can_return_is_mirrored():
    """The 1 K regression was a PARTIAL mirror; this is what closes it."""

    native = native_decode_reasons()
    for reason in lane.REQUEST_SHAPE_REASONS:
        assert reason in native, reason


def test_the_context_branches_of_the_native_contract_are_all_claimed():
    """Any native reason mentioning the request's own size must be mirrored.

    A new context-dependent branch upstream fails this test instead of
    reaching a served request as a 500.
    """

    request_words = ("total_tokens", "context", "query rows")
    unclaimed = [
        reason
        for reason in native_decode_reasons()
        if any(word in reason for word in request_words)
        and reason not in lane.REQUEST_SHAPE_REASONS
        and "must be a host integer" not in reason
        and "must be an exact host integer" not in reason
        and "cannot be bool" not in reason
    ]
    assert unclaimed == [], unclaimed


def test_the_kernel_wrapper_names_a_mirror_bug_rather_than_blaming_the_request():
    source = inspect.getsource(lane.attention)
    assert "REQUEST_SHAPE_REASONS" in source
    assert "did not mirror the" in source


def test_context_decline_is_none_only_for_a_servable_shape():
    assert lane.context_decline(
        total_tokens=LONG_BANK, rows=4, k_eff=512, capacity=LONG_BANK
    ) is None
    assert lane.context_decline(
        total_tokens=DENSE_BANK, rows=4, k_eff=512, capacity=DENSE_BANK
    ) == "short_context"
    assert lane.context_decline(
        total_tokens=LONG_BANK, rows=4, k_eff=511, capacity=LONG_BANK
    ) == "partial_budget"
    assert lane.context_decline(
        total_tokens=LONG_BANK + 8, rows=4, k_eff=512, capacity=LONG_BANK
    ) == "context_exceeds_capacity"
    assert lane.context_decline(
        total_tokens=lane.MAX_CONTEXT + 4, rows=4, k_eff=512,
        capacity=lane.MAX_CONTEXT + 4,
    ) == "context_above_limit"


def test_nothing_raises_when_the_flag_is_off():
    """The whole predicate is one host-side compare when nobody armed it."""

    assert not indexer()._sparse_decode_route(
        fixed_cache(decode_rows=0),
        rows=4,
        k_eff=17,
       
        site="select_eager_verify",
    )
    assert lane.route_counters()["route_hits"] == 0


# ---------------------------------------------------------------------------
# The per-layer proof inside the traced verify body
# ---------------------------------------------------------------------------
def required(cache, rows, *, indexer_present=True):
    stub = SimpleNamespace(indexer=object() if indexer_present else None)
    return Attention._sparse_decode_required(stub, cache, rows)


def test_the_armed_verify_width_on_a_fixed_cache_is_required(armed_verify):
    assert required(fixed_cache(), 4)


def test_a_width_the_flag_does_not_arm_is_not_required(armed_verify):
    assert not required(fixed_cache(), 1)


def test_a_growable_cache_is_never_required(armed_verify):
    assert not required(SimpleNamespace(fixed_capacity=False), 4)


def test_a_layer_without_an_indexer_is_never_required(armed_verify):
    assert not required(fixed_cache(), 4, indexer_present=False)


def guard(sel_mask, *, rows=4, before=None):
    stub = SimpleNamespace()
    return Attention._require_sparse_decode_lane(
        stub,
        sel_mask,
        rows=rows,
        before=before or {"route_hits": 0, "request_declines": 0},
    )


def test_the_guard_accepts_the_sparse_lane(armed_verify):
    guard(("sparse_blocks", object()))


def test_the_guard_accepts_a_forward_that_routed_for_short_context(armed_verify):
    """The HumanEval case: the indexer declined, and that is correct."""

    before = lane.route_snapshot()
    lane.note_request_decline(
        "select_eager_verify", "short_context", total_tokens=2048, blocks=512
    )
    guard(("gather_rows", object(), object()), before=before)


def test_the_guard_refuses_a_full_budget_forward_on_another_lane(armed_verify):
    with pytest.raises(RuntimeError, match="gather_rows"):
        guard(("gather_rows", object(), object()))


@pytest.mark.parametrize(
    "sel_mask, name",
    [
        (("flash", object(), 0), "flash"),
        (("flash_prefill", object(), object()), "flash_prefill"),
        (None, "no_selection"),
        (object(), "dense_mask"),
    ],
)
def test_the_guard_names_the_lane_attention_actually_got(armed_verify, sel_mask, name):
    with pytest.raises(RuntimeError, match=name):
        guard(sel_mask)


def test_the_guard_sits_before_every_other_attention_branch():
    source = inspect.getsource(Attention.__call__)
    call = source.index("self._require_sparse_decode_lane(")
    for lane_name in ("flash", "flash_prefill", "sparse_blocks", "gather_rows"):
        assert call < source.index(f'sel_mask[0] == "{lane_name}"')


def test_the_guard_samples_before_the_indexer_runs():
    """The routing happens inside ``self.indexer(...)``, so the baseline must
    be taken above it or the delta would always be zero."""

    source = inspect.getsource(Attention.__call__)
    assert source.index("sparse_before = _sparse_route_snapshot()") < source.index(
        "self.indexer("
    )


def test_the_guard_skips_vision_requests():
    source = inspect.getsource(Attention.__call__)
    assert "if vrope is None and sparse_required:" in source


# ---------------------------------------------------------------------------
# The graph-level proof
# ---------------------------------------------------------------------------
ZERO = {"route_hits": 0, "request_declines": 0}


def test_assert_traced_raises_when_the_armed_lane_missed_the_graph(armed_verify):
    with pytest.raises(lane.SparseDecodeContractError, match="not in the traced"):
        lane.assert_traced(4, before=ZERO, where="compiled verify")


def test_assert_traced_passes_once_a_route_hit_landed(armed_verify):
    lane.note_route_hit("select_eager_verify")
    lane.assert_traced(4, before=ZERO, where="compiled verify")


def test_assert_traced_accepts_a_forward_that_routed_for_short_context(
    armed_verify,
):
    """The stock lane IS correct below 2,052 tokens; the assertion says so."""

    lane.note_request_decline(
        "select_eager_verify", "short_context", total_tokens=2048, blocks=512
    )
    lane.assert_traced(4, before=ZERO, where="compiled verify")


def test_assert_traced_still_raises_on_an_unbound_full_budget_forward(
    armed_verify,
):
    """A short forward earlier in the run must not vouch for this one."""

    lane.note_request_decline(
        "select_eager_verify", "short_context", total_tokens=2048, blocks=512
    )
    before = lane.route_snapshot()
    with pytest.raises(
        lane.SparseDecodeContractError, match="shape is one the lane can serve"
    ):
        lane.assert_traced(4, before=before, where="compiled verify")


def test_assert_traced_measures_the_delta_not_the_total(armed_verify):
    """A hit from a PREVIOUS trace must not vouch for this one."""

    lane.note_route_hit("select_eager_verify")
    with pytest.raises(lane.SparseDecodeContractError):
        lane.assert_traced(
            4, before={"route_hits": 1, "request_declines": 0}, where="compiled verify"
        )


def test_assert_traced_is_silent_off_the_armed_width(armed_verify):
    lane.assert_traced(1, before=ZERO, where="compiled verify")
    lane.assert_traced(16, before=ZERO, where="compiled verify")


def test_assert_traced_is_silent_when_nothing_is_armed():
    lane.assert_traced(4, before=ZERO, where="compiled verify")


def test_the_snapshot_carries_both_ways_a_forward_can_prove_engagement():
    assert lane.route_snapshot() == {"route_hits": 0, "request_declines": 0}
    lane.note_route_hit("x")
    lane.note_request_decline("x", "short_context", total_tokens=2048, blocks=9)
    assert lane.route_snapshot() == {"route_hits": 1, "request_declines": 1}


def test_the_compiled_verify_body_samples_and_asserts_across_the_forward():
    source = (ROOT / "mtplx" / "graphbank.py").read_text()
    body = source[source.index("def verify_step(input_ids, *args):") :]
    body = body[: body.index("        return verify_step")]
    sample = body.index("sparse_route_before = ")
    forward = body.index("result = live._runtime_forward(")
    check = body.index("_qsa_sparse_lane.assert_traced(")
    assert sample < forward < check


# ---------------------------------------------------------------------------
# Construction owns the "cache built without the lane" gate
# ---------------------------------------------------------------------------
def make_cache(monkeypatch, *, decode=False, armed=False):
    from mtplx import graphbank

    monkeypatch.setattr(
        graphbank, "qsa_sparse_decode_enabled", lambda: bool(armed)
    )
    kv = SimpleNamespace(cache=[None, None, None], step=256, keys=None, values=None)
    return graphbank.TensorOffsetQSACache(
        kv,
        None,
        None,
        compress_ratio=4,
        rows_gather_kv_m4=None,
        qsa_sparse_decode=decode,
    )


def test_an_armed_flag_on_a_cache_built_without_the_lane_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="constructed without the lane"):
        make_cache(monkeypatch, decode=False, armed=True)


def test_an_unarmed_process_builds_the_cache_untouched(monkeypatch):
    cache = make_cache(monkeypatch, decode=False, armed=False)
    assert cache.qsa_sparse_decode_rows == 0


def test_a_disabled_probe_now_fails_the_build_rather_than_the_arm(monkeypatch):
    monkeypatch.setattr(lane, "install", lambda *a, **k: False)
    monkeypatch.setattr(lane, "_DISABLED_REASON", "parity probe failed on 'x'")
    with pytest.raises(RuntimeError, match="parity probe failed"):
        make_cache(monkeypatch, decode=True, armed=True)


def test_a_passing_probe_wires_the_verify_rows(monkeypatch):
    monkeypatch.setattr(lane, "install", lambda *a, **k: True)
    cache = make_cache(monkeypatch, decode=True, armed=True)
    assert cache.qsa_sparse_decode_rows == lane.VERIFY_ROWS


def test_the_shadow_twins_carry_the_lane_without_a_defaulting_getattr():
    """A twin that silently defaulted to False would revert to stock.

    Both twin sites build a ``TensorOffsetQSACache`` from an entry that always
    sets this attribute, so reading it directly is what makes a site that
    forgot it loud.
    """

    source = (ROOT / "mtplx" / "graphbank.py").read_text()
    assert 'getattr(\n                        entry, "qsa_sparse_decode"' not in source
    assert source.count("qsa_sparse_decode=entry.qsa_sparse_decode") == 2


# ---------------------------------------------------------------------------
# The evidence a receipt and a log carry
# ---------------------------------------------------------------------------
def test_the_install_verdict_reaches_stderr_not_only_the_logger(capsys):
    lane._emit("[mtplx] qsa_sparse_decode: probe line")
    assert "[mtplx] qsa_sparse_decode: probe line" in capsys.readouterr().err


def test_install_emits_both_the_line_and_the_json(armed_verify):
    source = inspect.getsource(lane.install)
    assert source.count("_emit(engagement_line(enabled=True))") == 1
    assert source.count("_emit(engagement_line(enabled=False))") == 1
    assert source.count('"[mtplx] qsa_sparse_decode install: "') == 2


def test_the_engagement_line_states_rows_tile_splits_and_the_probe(armed_verify):
    lane._PROBE_REPORT["worst"] = {
        "cell": "verify-4096",
        "vs_fp32": {"max_abs_ulps": 0.75, "rel_l2": 1.2e-5, "top1": 1.0},
        "vs_shipped": {"rel_l2": 4.78e-3},
    }
    lane._COUNTS["cache_installs"] = 12
    lane._COUNTS["probe_runs"] = 2
    line = lane.engagement_line(enabled=True)
    assert line.startswith("[mtplx] qsa_sparse_decode armed:")
    for token in ("rows=4", "tile=128:32", "splits=17", "caches=12", "probe_runs=2"):
        assert token in line
    assert "'verify-4096'" in line


def test_the_off_line_carries_the_reason(monkeypatch):
    monkeypatch.setattr(lane, "_DISABLED_REASON", "parity probe failed on 'x'")
    assert lane.engagement_line(enabled=False) == (
        "[mtplx] qsa_sparse_decode: off (parity probe failed on 'x')"
    )


def test_the_receipt_never_raises_while_the_probe_is_pending(armed_verify):
    block = lane.receipt()
    assert block["armed"] is True
    assert block["pending"] is True
    assert block["installed"] is False
    assert block["disabled_reason"] is None


def test_the_receipt_carries_everything_the_owner_asked_for(armed_verify):
    block = lane.receipt()
    for key in (
        "armed",
        "installed",
        "disabled_reason",
        "tile",
        "splits",
        "probe",
        "route_hits",
        "route_sites",
        "route_declines",
        "kernel_calls",
        "cache_installs",
    ):
        assert key in block, key
    assert block["tile"] == [128, 32]
    assert block["splits"] == 17


def test_route_hits_and_kernel_calls_are_separate_counters(armed_verify):
    lane.note_route_hit("select_eager_verify")
    lane._COUNTS["verify_kernel"] += 3
    block = lane.receipt()
    assert block["route_hits"] == 1
    assert block["kernel_calls"] == {"verify_kernel": 3}


def test_reset_clears_the_route_state_too():
    lane.note_route_hit("a")
    lane.note_route_decline("b")
    lane.note_request_decline("a", "short_context", total_tokens=2048, blocks=9)
    lane.reset_for_tests()
    assert lane.route_counters() == {
        "route_hits": 0,
        "route_sites": {},
        "route_declines": {},
        "request_declines": 0,
        "request_decline_extremes": {},
        "short_context_tokens": 2052,
    }


# ---------------------------------------------------------------------------
# The driver's engagement check
# ---------------------------------------------------------------------------


























# ---------------------------------------------------------------------------
# Regression: the served-launch arming ordering (battery, 2026-09-07)
# ---------------------------------------------------------------------------
def test_the_reader_resolves_lazily_not_at_import(monkeypatch):
    """A stamp applied AFTER runtime_options is imported must still arm.

    The served bug: qsa_sparse_decode_enabled() froze the env at IMPORT
    (default False), so openai.py's fixed-M4 auto-arm setdefault -- which runs
    later and stamps MTPLX_QSA_SPARSE_DECODE=1 when the native ext is built --
    landed after the cache and the lane never engaged. The reader must read at
    first USE (install time), which is after the overrides are applied.
    """

    from mtplx import runtime_options as ro

    monkeypatch.delenv("MTPLX_QSA_SPARSE_DECODE", raising=False)
    monkeypatch.delenv("MTPLX_FABLE_QSA_SPARSE_DECODE", raising=False)
    # As at a fresh import, with nothing stamped yet.
    monkeypatch.setattr(ro, "_QSA_SPARSE_DECODE", None)
    assert ro.qsa_sparse_decode_enabled() is False
    # The auto-arm stamps the key AFTER import; a lazily-resolved reader that
    # has not yet been forced to a value must pick it up.
    monkeypatch.setattr(ro, "_QSA_SPARSE_DECODE", None)
    monkeypatch.setenv("MTPLX_QSA_SPARSE_DECODE", "1")
    assert ro.qsa_sparse_decode_enabled() is True
    # The old FABLE name is honoured only when the new key is unset.
    monkeypatch.setattr(ro, "_QSA_SPARSE_DECODE", None)
    monkeypatch.delenv("MTPLX_QSA_SPARSE_DECODE", raising=False)
    monkeypatch.setenv("MTPLX_FABLE_QSA_SPARSE_DECODE", "1")
    assert ro.qsa_sparse_decode_enabled() is True


def test_the_fixed_m4_auto_arm_stamps_or_declines_the_sparse_decode_lane(
    tmp_path, monkeypatch
):
    """Apply the fixed-M4 auto-arm block; the sparse-decode install must be
    ATTEMPTED -- stamped (native ext built) or declined-to-stock (not built) --
    and never silently absent, which is exactly what the served launch showed.
    """

    import io
    import contextlib
    import json as _json
    from types import SimpleNamespace as _NS

    import mtplx.server.openai as openai
    from mtplx import runtime_options as ro
    from mtplx.native import native_qsa_available
    from mtplx.profiles import (
        MODEL_RUNTIME_ENV_OVERRIDE_KEYS,
        normalize_runtime_env_overrides,
    )

    (tmp_path / "config.json").write_text(
        _json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )
    args = _NS(
        generation_mode="mtp",
        verify_strategy="capture_commit",
        model=str(tmp_path),
    )
    # Force the fixed-M4 predicate on rather than crafting a full fixed-verify
    # config; the sparse-decode default gates only on that predicate + the
    # built native extension.
    monkeypatch.setattr(openai, "_served_model_is_qwen4_fixed_m4", lambda a: True)
    for key in ("MTPLX_QSA_SPARSE_DECODE", "MTPLX_FABLE_QSA_SPARSE_DECODE"):
        monkeypatch.delenv(key, raising=False)

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        overrides = openai._server_runtime_env_overrides(args, {})
    log = err.getvalue()

    # The key is registered, so a stamped value survives the boot-time
    # validator (the check that only runs inside apply_profile_env).
    assert "MTPLX_QSA_SPARSE_DECODE" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    assert normalize_runtime_env_overrides(overrides) == overrides

    if native_qsa_available():
        # Ext built: the lane is armed by default -> install attempted.
        assert overrides.get("MTPLX_QSA_SPARSE_DECODE") == "1"
        # And the reader, resolved at install time AFTER the stamp is applied
        # to the environment, arms (the bug returned False here).
        monkeypatch.setattr(ro, "_QSA_SPARSE_DECODE", None)
        monkeypatch.setenv(
            "MTPLX_QSA_SPARSE_DECODE", overrides["MTPLX_QSA_SPARSE_DECODE"]
        )
        assert ro.qsa_sparse_decode_enabled() is True
    else:
        # No ext: decline-to-stock, key NOT stamped, a verdict line printed so
        # a wheel without the extension still serves and says so.
        assert "MTPLX_QSA_SPARSE_DECODE" not in overrides
        assert "MTPLX_QSA_SPARSE_DECODE declined to stock" in log
