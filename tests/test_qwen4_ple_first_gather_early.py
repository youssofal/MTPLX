"""MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY: head start, vectorised gather, exactness.

Pure Python + NumPy.  ``mtplx.ple_row_gather`` and
``mtplx.ple_prefill_lookahead`` import no MLX; the row half is exercised
against a real on-disk memmap built in tmp_path at the production row geometry
(80/10/10 bytes per row), so the gather under test is the shipped expression
over real mapped pages -- no GPU, no MLX array, no model load.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mtplx import ple_prefill_lookahead as lookahead_mod
from mtplx import ple_row_gather as row_gather
from mtplx.ple_prefill_lookahead import EarlyFirstGather, PrefillLookahead


ROOT = Path(__file__).resolve().parents[1]
MODEL_TEXT = (ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text("utf-8")
GENERATION_TEXT = (ROOT / "mtplx" / "generation.py").read_text("utf-8")

#: Production row geometry: 16 ngram heads/token, head_dim 160, q4/g32.
_MAPS = {"weight": (np.uint32, 20), "scales": (np.uint16, 5), "biases": (np.uint16, 5)}
_TABLE_ROWS = 4096


class _Immediate:
    """A future whose work already ran -- keeps the tests single-threaded."""

    def __init__(self, fn, *args):
        self._exc = None
        try:
            self._value = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - re-raised in result()
            self._value, self._exc = None, exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value

    def cancel(self):
        return False

    def add_done_callback(self, fn):
        fn(self)


def _build_table(tmp_path: Path):
    """One real file per map, memmapped read-only, plus an fd for pread."""

    rng = np.random.default_rng(20260902)
    maps: dict[str, np.memmap] = {}
    meta: dict[str, tuple[int, int]] = {}
    fds: dict[str, int] = {}
    for name, (dtype, cols) in _MAPS.items():
        path = tmp_path / f"{name}.bin"
        payload = rng.integers(
            0, np.iinfo(dtype).max, size=(_TABLE_ROWS, cols), dtype=dtype
        )
        path.write_bytes(payload.tobytes())
        maps[name] = np.memmap(
            path, mode="r", dtype=dtype, shape=(_TABLE_ROWS, cols)
        )
        meta[name] = (0, cols * np.dtype(dtype).itemsize)
        fds[name] = os.open(str(path), os.O_RDONLY)
    return maps, meta, fds


def _pread_reference(fds, meta, maps, uniq, inverse, names):
    """The rows the shipped pread path would produce, read syscall by syscall."""

    out = {}
    for name in names:
        base, row_bytes = meta[name]
        dtype = maps[name].dtype
        cols = maps[name].shape[1]
        rows = np.empty((len(uniq), cols), dtype=dtype)
        for i, row in enumerate(uniq):
            raw = os.pread(fds[name], row_bytes, base + int(row) * row_bytes)
            rows[i] = np.frombuffer(raw, dtype=dtype)
        out[name] = rows[inverse]
    return out


# --------------------------------------------------------------------------
# (b) the vectorised gather: byte-identical to the pread path
# --------------------------------------------------------------------------


def test_vectorized_gather_is_byte_identical_to_pread(tmp_path):
    maps, meta, fds = _build_table(tmp_path)
    names = tuple(_MAPS)
    rng = np.random.default_rng(7)
    flat = rng.integers(0, _TABLE_ROWS, size=65_536, dtype=np.int64)
    uniq, inverse = np.unique(flat, return_inverse=True)

    got = row_gather.gather_matrices(maps, uniq, inverse, names)
    want = _pread_reference(fds, meta, maps, uniq, inverse, names)

    for name in names:
        assert got[name].dtype == want[name].dtype
        assert got[name].shape == want[name].shape
        assert got[name].tobytes() == want[name].tobytes(), name
    for fd in fds.values():
        os.close(fd)


def test_gather_matrices_restores_flat_order(tmp_path):
    """The un-unique step is what makes the gather positional, not sorted."""

    maps, _meta, fds = _build_table(tmp_path)
    flat = np.array([9, 3, 9, 1, 3], dtype=np.int64)
    uniq, inverse = np.unique(flat, return_inverse=True)
    got = row_gather.gather_matrices(maps, uniq, inverse, ("weight",))["weight"]
    for position, row in enumerate(flat):
        assert got[position].tobytes() == np.asarray(maps["weight"][row]).tobytes()
    for fd in fds.values():
        os.close(fd)


def test_warm_decision_reads_vectorized_on_a_touched_mapping(tmp_path):
    maps, _meta, fds = _build_table(tmp_path)
    rows = np.arange(_TABLE_ROWS, dtype=np.int64)
    # Fault every page in through the same mapping mincore will be asked about.
    for memmap in maps.values():
        int(np.asarray(memmap).view(np.uint8).sum(dtype=np.int64))
    path, fraction = row_gather.warm_decision(list(maps.values()), rows)
    assert path == "vectorized"
    assert fraction == pytest.approx(1.0)
    for fd in fds.values():
        os.close(fd)


def test_warm_decision_falls_back_to_pread_without_mincore(tmp_path, monkeypatch):
    """An unavailable probe must answer with the shipped path, never guess warm."""

    maps, _meta, fds = _build_table(tmp_path)
    monkeypatch.setattr(row_gather._Libc, "get", classmethod(lambda cls: None))
    path, fraction = row_gather.warm_decision(
        list(maps.values()), np.arange(64, dtype=np.int64)
    )
    assert (path, fraction) == ("pread", None)
    for fd in fds.values():
        os.close(fd)


def test_resident_fraction_sample_is_deterministic(tmp_path):
    """A random sampler would move an A/B's answer run to run."""

    maps, _meta, fds = _build_table(tmp_path)
    rows = np.arange(_TABLE_ROWS, dtype=np.int64)
    first = row_gather.resident_fraction(maps["weight"], rows)
    second = row_gather.resident_fraction(maps["weight"], rows)
    assert first == second
    for fd in fds.values():
        os.close(fd)


# --------------------------------------------------------------------------
# (c) the whole-prompt pre-touch
# --------------------------------------------------------------------------


def test_touch_rows_faults_every_requested_row(tmp_path):
    maps, _meta, fds = _build_table(tmp_path)
    rows = np.unique(
        np.random.default_rng(3).integers(0, _TABLE_ROWS, size=1024, dtype=np.int64)
    )
    touched = row_gather.touch_rows(list(maps.values()), rows, block=97)
    assert touched == int(rows.shape[0])
    path, fraction = row_gather.warm_decision(list(maps.values()), rows)
    assert (path, fraction) == ("vectorized", pytest.approx(1.0))
    for fd in fds.values():
        os.close(fd)


def test_touch_rows_is_a_no_op_on_an_empty_set(tmp_path):
    maps, _meta, fds = _build_table(tmp_path)
    assert row_gather.touch_rows(list(maps.values()), np.array([], dtype=np.int64)) == 0
    for fd in fds.values():
        os.close(fd)


# --------------------------------------------------------------------------
# (a) the head start: adoption, and the single-chunk lane
# --------------------------------------------------------------------------


def _early(ids, span, payload, *, prefetch=None):
    lookahead_mod.reset_counters()
    return EarlyFirstGather(
        ids,
        span,
        prepare=lambda plan, a, b, record: (
            record.update({"path": "vectorized", "rows": 4}) or payload
        ),
        submit=lambda fn, *args: _Immediate(fn, *args),
        prefetch_rest=prefetch,
    )


def test_lookahead_adopts_the_early_first_chunk(tmp_path):
    ids = np.arange(600, dtype=np.int64)
    flat = np.arange(4096, dtype=np.int64)
    payload = (flat, {"weight": np.zeros((4096, 20), dtype=np.uint32)})
    early = _early(ids, (0, 256), payload)
    prepared: list[tuple[int, int]] = []
    lookahead = PrefillLookahead(
        ids,
        [(0, 256), (256, 600)],
        prepare=lambda a, b: prepared.append((a, b)) or ("late", {}),
        submit=lambda fn, *args: _Immediate(fn, *args),
    )
    with lookahead_mod.first_gather_early_scope(early):
        assert lookahead.adopt_early(lookahead_mod.active_early_first_gather())
        assert lookahead.take(0) is payload
    # Span 0 was prepared ONCE, by the early worker.
    assert prepared == []
    assert lookahead.engagement()["hits"] == 1
    counters = lookahead_mod.snapshot_counters()
    assert counters["early_adopted"] == 1
    status = lookahead_mod.last_early_status()
    assert status["outcome"] == "adopted_hit"
    assert status["path"] == "vectorized"
    assert status["rows"] == 4
    assert status["started_at_ms_before_layer2"] >= 0.0


def test_adoption_is_refused_when_the_prefill_chose_another_first_span():
    ids = np.arange(600, dtype=np.int64)
    early = _early(ids, (0, 256), ("early", {}))
    lookahead = PrefillLookahead(
        ids,
        [(0, 300), (300, 600)],
        prepare=lambda a, b: ("late", {}),
        submit=lambda fn, *args: _Immediate(fn, *args),
    )
    with lookahead_mod.first_gather_early_scope(early):
        assert not lookahead.adopt_early(lookahead_mod.active_early_first_gather())
        lookahead.submit(0)
        assert lookahead.take(0) == ("late", {})
    assert lookahead_mod.snapshot_counters()["early_span_mismatch"] == 1
    assert lookahead.engagement()["hits"] == 1


def test_adoption_is_refused_when_the_prompt_is_not_the_one_predicted():
    early = _early(np.arange(600, dtype=np.int64), (0, 256), ("early", {}))
    lookahead = PrefillLookahead(
        np.arange(600, dtype=np.int64) + 1,
        [(0, 256), (256, 600)],
        prepare=lambda a, b: ("late", {}),
        submit=lambda fn, *args: _Immediate(fn, *args),
    )
    with lookahead_mod.first_gather_early_scope(early):
        assert not lookahead.adopt_early(lookahead_mod.active_early_first_gather())
    assert lookahead_mod.snapshot_counters()["early_plan_mismatch"] == 1


def test_single_chunk_prefill_consumes_the_early_gather_directly():
    """The 1K cell: one chunk, so the lookahead is inert and this IS the win."""

    ids = np.arange(1023, dtype=np.int64)
    payload = ("rows", {})
    early = _early(ids, (0, 1023), payload)
    lookahead = PrefillLookahead(
        ids,
        [(0, 1023)],
        prepare=lambda a, b: ("late", {}),
        submit=lambda fn, *args: _Immediate(fn, *args),
    )
    assert lookahead.armed is False
    assert lookahead.inert_reason == "single_span"
    with lookahead_mod.first_gather_early_scope(early) as active:
        assert active.take(ids) is payload
    counters = lookahead_mod.snapshot_counters()
    assert counters["early_hit"] == 1
    assert lookahead_mod.last_early_status()["outcome"] == "hit"


def test_early_take_refuses_a_chunk_that_is_not_its_span():
    ids = np.arange(600, dtype=np.int64)
    early = _early(ids, (0, 256), ("rows", {}))
    with lookahead_mod.first_gather_early_scope(early) as active:
        assert active.take(np.arange(300, dtype=np.int64)) is None
    assert lookahead_mod.snapshot_counters()["early_miss_wrong_span"] == 1
    assert lookahead_mod.last_early_status()["outcome"] == "miss_wrong_span"


def test_early_declines_are_not_reported_as_hits():
    """A span the sidecar routes to its owner-thread LRU returns None."""

    ids = np.arange(600, dtype=np.int64)
    early = _early(ids, (0, 256), None)
    with lookahead_mod.first_gather_early_scope(early) as active:
        assert active.take(ids[:256]) is None
    assert lookahead_mod.snapshot_counters()["early_miss_ineligible"] == 1
    assert lookahead_mod.last_early_status()["outcome"] == "miss_ineligible"


def test_a_never_consumed_early_gather_says_so():
    ids = np.arange(600, dtype=np.int64)
    early = _early(ids, (0, 256), ("rows", {}))
    with lookahead_mod.first_gather_early_scope(early):
        pass
    assert lookahead_mod.last_early_status()["outcome"] == "never_needed"


def test_prefetch_rest_is_queued_behind_the_first_chunk():
    ids = np.arange(600, dtype=np.int64)
    order: list[str] = []

    def _prepare(plan, a, b, record):
        order.append("span0")
        return ("rows", {})

    def _prefetch(plan, a, record):
        order.append("rest")
        record["prefetch_rest_rows"] = 7
        return 7

    lookahead_mod.reset_counters()
    early = EarlyFirstGather(
        ids,
        (0, 256),
        prepare=_prepare,
        submit=lambda fn, *args: _Immediate(fn, *args),
        prefetch_rest=_prefetch,
    )
    with lookahead_mod.first_gather_early_scope(early) as active:
        assert active.take(ids[:256]) == ("rows", {})
    assert order == ["span0", "rest"]
    assert lookahead_mod.last_early_status()["prefetch_rest_rows"] == 7


def test_scope_without_a_gather_records_the_reason():
    with lookahead_mod.first_gather_early_scope(None, "unpredictable_first_span") as e:
        assert e is None
    status = lookahead_mod.last_early_status()
    assert status["armed"] is False
    assert status["reason"] == "unpredictable_first_span"


def test_flag_parses_like_the_lookahead_flag(monkeypatch):
    assert row_gather.ENV_FLAG == "MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY"
    assert lookahead_mod.EARLY_ENV_FLAG == row_gather.ENV_FLAG
    for raw, want in (("", False), ("0", False), ("1", True), ("on", True)):
        row_gather.enabled.cache_clear()
        lookahead_mod.early_enabled.cache_clear()
        monkeypatch.setenv(row_gather.ENV_FLAG, raw)
        assert lookahead_mod.early_enabled() is want
    row_gather.enabled.cache_clear()
    lookahead_mod.early_enabled.cache_clear()
    monkeypatch.setenv(row_gather.ENV_FLAG, "maybe")
    with pytest.raises(ValueError):
        lookahead_mod.early_enabled()
    row_gather.enabled.cache_clear()
    lookahead_mod.early_enabled.cache_clear()


# --------------------------------------------------------------------------
# wiring: the lane has to be reachable from the request, not just importable
# --------------------------------------------------------------------------


def test_request_arrival_is_where_the_gather_starts():
    assert (
        "@_with_ple_first_gather_early\n@_with_vision_rope\n"
        "def restore_or_prefill_prompt_state(" in GENERATION_TEXT
    )
    scope = GENERATION_TEXT.split("def _ple_first_gather_early_scope", 1)[1]
    scope = scope.split("\ndef ", 1)[0]
    assert 'hook = _resolve_ple_lookahead_hook(rt, "ple_first_gather_early")' in scope
    # An unservable architecture is a verdict, not a request failure: strict
    # arms raise, serving records and declines the lane.
    assert 'inertness_verdict(\n            "no_first_gather_hook"' in scope
    # The gather is declined outright when a bank restore will move the
    # prefill past the predicted span (the 2026-09-03 warm-turn storm).
    assert "_bank_may_preempt_first_span(session_bank, prompt_ids, span)" in scope
    assert '"bank_prefix_may_serve_first_span"' in scope


def test_first_span_prediction_declines_rather_than_guessing():
    body = GENERATION_TEXT.split("def _predicted_first_prefill_span", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert "_iter_prefill_chunk_spans(body_len)" in body
    assert "_prefill_spans_with_tail_grid(" in body
    assert "return None" in body


def test_the_sidecar_measures_warmth_instead_of_assuming_it():
    body = MODEL_TEXT.split("def prepare_rows_np", 1)[1].split("\n    def ", 1)[0]
    assert "warm_decision(list(maps.values()), uniq)" in body
    assert 'if path == "vectorized":' in body
    assert "self._warm(uniq, counted=False)" in body
    assert "gather_matrices(maps, uniq, inverse, names)" in body


def test_a_lookahead_less_prefill_still_consumes_the_early_gather():
    body = MODEL_TEXT.split("def _take_prefill_lookahead", 1)[1]
    body = body.split("\n    def ", 1)[0]
    assert "return self._take_first_gather_early(ids_np, flat)" in body


# --------------------------------------------------------------------------
# The prediction itself, run out of the shipped source (generation imports MLX)
# --------------------------------------------------------------------------


def _predict(**stubs):
    """Compile the real prediction, over the real span helpers, with stubs.

    Everything that decides the ANSWER is shipped code -- the two span
    planners and the split -- so the test cannot pass by agreeing with a
    re-implementation of the arithmetic.  Only the environment readers are
    stubbed.
    """

    import ast

    wanted = {
        "_predicted_first_prefill_span",
        "_iter_prefill_chunk_spans",
        "_prefill_spans_with_tail_grid",
        "_geometric_tail_edges",
        "_split_spans_at",
        "_merge_small_remainder",
    }
    tree = ast.parse(GENERATION_TEXT)
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    assert {n.name for n in nodes} == wanted
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {
        "_sustained_prefill_enabled": lambda: True,
        "_prefill_chunk_size": lambda: 4096,
        "_gdn_boundary_capture_enabled": lambda: True,
        "_gdn_boundary_tail_interval": lambda: 256,
        "_gdn_boundary_tail_layout": lambda: "geometric",
        "_gdn_boundary_tail_min_rung": lambda: 1024,
        "_gdn_boundary_tail_backoff": lambda: 64,
        "_prefill_min_chunk_rows": lambda: 0,
    }
    namespace.update(stubs)
    exec(compile(module, "<generation>", "exec"), namespace)
    return namespace["_predicted_first_prefill_span"]


def test_prediction_is_the_plain_chunk_grid_without_a_session_bank():
    """The benchmark lane: no bank, so boundary capture cannot be on."""

    predict = _predict()
    assert predict(list(range(16_384))) == (0, 4096)


def test_prediction_covers_a_single_chunk_prompt_whole():
    """The 1K cell: one chunk, which is exactly where the win is."""

    predict = _predict()
    assert predict(list(range(1024))) == (0, 1023)


def test_prediction_declines_when_the_tail_grid_could_cut_chunk_one():
    """A banked short prompt: the grid cuts 1023 (512 + 256 + 255 under the
    geometric layout, 256s under the dense one), the plain plan does not, and
    the two disagree -- so the lane does not guess."""

    predict = _predict()
    assert predict(list(range(1024)), session_bank=object()) is None
    dense = _predict(_gdn_boundary_tail_layout=lambda: "dense")
    assert dense(list(range(1024)), session_bank=object()) is None


def test_prediction_survives_a_banked_multi_chunk_prompt():
    """The tail grid only refines the LAST span, so chunk 1 is unchanged."""

    predict = _predict()
    assert predict(list(range(16_384)), session_bank=object()) == (0, 4096)


def test_prediction_declines_when_a_stable_prefix_edge_cuts_chunk_one():
    predict = _predict()
    assert (
        predict(list(range(16_384)), session_bank=object(), stable_prefix_len=1000)
        is None
    )


def test_prediction_ignores_the_bank_when_boundary_capture_is_off():
    predict = _predict(_gdn_boundary_capture_enabled=lambda: False)
    assert predict(list(range(1024)), session_bank=object()) == (0, 1023)


def test_prediction_ignores_the_bank_for_a_vision_prompt():
    predict = _predict()
    assert (
        predict(list(range(1024)), session_bank=object(), vision_splice=object())
        == (0, 1023)
    )


def test_prediction_declines_when_the_streaming_loop_is_not_taken():
    predict = _predict(_sustained_prefill_enabled=lambda: False)
    assert predict(list(range(16_384))) is None


@pytest.mark.parametrize("prompt", [[], [7]])
def test_prediction_declines_an_empty_prefill_body(prompt):
    predict = _predict()
    assert predict(prompt) is None


# --------------------------------------------------------------------------
# Mapping advice and the load-time sequential prewarm
# --------------------------------------------------------------------------


def test_madvise_is_random_by_default_and_normal_under_the_flag(monkeypatch):
    monkeypatch.delenv(row_gather.MADVISE_ENV, raising=False)
    monkeypatch.setenv(row_gather.ENV_FLAG, "0")
    row_gather.enabled.cache_clear()
    assert row_gather.madvise_choice() == ("random", row_gather.MADV_RANDOM)
    monkeypatch.setenv(row_gather.ENV_FLAG, "1")
    row_gather.enabled.cache_clear()
    assert row_gather.madvise_choice() == ("normal", row_gather.MADV_NORMAL)
    row_gather.enabled.cache_clear()


def test_madvise_override_wins_either_way(monkeypatch):
    monkeypatch.setenv(row_gather.ENV_FLAG, "1")
    row_gather.enabled.cache_clear()
    monkeypatch.setenv(row_gather.MADVISE_ENV, "random")
    assert row_gather.madvise_choice() == ("random", row_gather.MADV_RANDOM)
    monkeypatch.setenv(row_gather.MADVISE_ENV, "sequential")
    assert row_gather.madvise_choice() == ("sequential", row_gather.MADV_SEQUENTIAL)
    monkeypatch.setenv(row_gather.MADVISE_ENV, "willneed")
    with pytest.raises(ValueError):
        row_gather.madvise_choice()
    row_gather.enabled.cache_clear()


def test_prewarm_is_on_by_default_and_independent_of_the_lane(monkeypatch):
    monkeypatch.delenv(row_gather.PREWARM_ENV, raising=False)
    monkeypatch.delenv(row_gather.PREWARM_AT_LOAD_ENV, raising=False)
    monkeypatch.setenv(row_gather.ENV_FLAG, "0")
    row_gather.enabled.cache_clear()
    assert row_gather.prewarm_source() == (True, "default")
    monkeypatch.setenv(row_gather.PREWARM_ENV, "off")
    assert row_gather.prewarm_source() == (False, "env")
    monkeypatch.setenv(row_gather.PREWARM_ENV, "maybe")
    with pytest.raises(ValueError):
        row_gather.prewarm_at_load_enabled()
    row_gather.enabled.cache_clear()


def test_the_deprecated_alias_still_works_and_the_official_key_beats_it(monkeypatch):
    monkeypatch.delenv(row_gather.PREWARM_ENV, raising=False)
    monkeypatch.setenv(row_gather.PREWARM_AT_LOAD_ENV, "0")
    monkeypatch.setattr(row_gather, "_DEPRECATION_WARNED", False)
    assert row_gather.prewarm_source() == (False, "deprecated_env")
    monkeypatch.setenv(row_gather.PREWARM_ENV, "all")
    assert row_gather.prewarm_source() == (True, "env")


def test_the_cli_choice_beats_a_shell_set_value(monkeypatch):
    monkeypatch.setenv(row_gather.PREWARM_ENV, "all")
    assert row_gather.apply_prewarm_choice("off") == "cli"
    assert os.environ[row_gather.PREWARM_ENV] == "off"
    assert row_gather.prewarm_source() == (False, "env")
    # An unset flag leaves the environment alone and names what decided.
    monkeypatch.setenv(row_gather.PREWARM_ENV, "all")
    assert row_gather.apply_prewarm_choice(None) == "env"
    assert os.environ[row_gather.PREWARM_ENV] == "all"
    monkeypatch.delenv(row_gather.PREWARM_ENV, raising=False)
    monkeypatch.delenv(row_gather.PREWARM_AT_LOAD_ENV, raising=False)
    assert row_gather.apply_prewarm_choice(None) == "default"


def test_the_published_prewarm_receipt_has_one_shape(monkeypatch, tmp_path):
    monkeypatch.delenv(row_gather.PREWARM_ENV, raising=False)
    monkeypatch.delenv(row_gather.PREWARM_AT_LOAD_ENV, raising=False)
    row_gather._LAST_PREWARM.clear()
    keys = {
        "path",
        "bytes",
        "file_bytes",
        "complete",
        "seconds",
        "chunk_bytes",
        "gib_per_s",
        "skipped_reason",
        "enabled",
        "source",
    }
    # Before any model load: same keys, named reason, no invented numbers.
    before = row_gather.last_prewarm()
    assert set(before) == keys
    assert before == {**before, "enabled": True, "source": "default"}
    assert before["skipped_reason"] == "no_model_loaded"
    assert before["bytes"] == 0

    path = tmp_path / "table.bin"
    path.write_bytes(os.urandom(256 * 1024))
    published = row_gather.record_prewarm(
        row_gather.prewarm_file(path, chunk_bytes=64 * 1024),
        enabled=True,
        source="cli",
    )
    assert set(published) == keys
    assert row_gather.last_prewarm() == published
    assert published["skipped_reason"] is None
    assert published["bytes"] == path.stat().st_size

    skipped = row_gather.record_prewarm(
        row_gather.prewarm_skipped("disabled"), enabled=False, source="env"
    )
    assert set(skipped) == keys
    assert skipped["skipped_reason"] == "disabled"
    row_gather._LAST_PREWARM.clear()


def test_the_official_key_is_registered_with_the_other_runtime_keys():
    """A key the CLI stamps must be one packs and profiles may stamp too."""

    profiles = (ROOT / "mtplx" / "profiles.py").read_text("utf-8")
    keys = profiles.split("MODEL_RUNTIME_ENV_OVERRIDE_KEYS", 1)[1]
    keys = keys.split("\n)\n", 1)[0]
    assert '"MTPLX_NGRAM_PREWARM",' in keys
    assert '"MTPLX_NGRAM_HOT_MB",' in keys


def test_prewarm_file_reads_the_whole_file_and_reports_its_rate(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(os.urandom(3 * 1024 * 1024 + 17))
    receipt = row_gather.prewarm_file(path, chunk_bytes=1024 * 1024)
    assert receipt["bytes"] == receipt["file_bytes"] == path.stat().st_size
    assert receipt["complete"] is True
    assert receipt["chunk_bytes"] == 1024 * 1024
    assert receipt["gib_per_s"] > 0


def test_prewarm_makes_the_mapping_read_as_resident(tmp_path):
    """The load-time read has to be visible to the probe, or the lane never
    engages: mincore reports through the mapping, the prewarm reads the fd."""

    path = tmp_path / "table.bin"
    path.write_bytes(os.urandom(2 * 1024 * 1024))
    row_gather.prewarm_file(path, chunk_bytes=64 * 1024)
    memmap = np.memmap(path, mode="r", dtype=np.uint8, shape=(2 * 1024 * 1024,))
    rows = np.arange(0, 2 * 1024 * 1024, 4096, dtype=np.int64)
    path_taken, fraction = row_gather.warm_decision([memmap], rows)
    assert (path_taken, fraction) == ("vectorized", pytest.approx(1.0))


def test_the_sidecar_records_which_gather_path_each_read_took():
    body = MODEL_TEXT.split("def _rows_matrices", 1)[1].split("\n    def ", 1)[0]
    assert "warm_decision(list(maps.values()), uniq)" in body
    assert "self.vectorized_gathers += 1" in body
    assert "self.pread_gathers += 1" in body
    prepare = MODEL_TEXT.split("def prepare_rows_np", 1)[1].split("\n    def ", 1)[0]
    assert "self.vectorized_gathers += 1" in prepare
    assert "self.pread_gathers += 1" in prepare


def test_the_sidecar_decides_its_advice_instead_of_hard_coding_it():
    init = MODEL_TEXT.split("    def __init__(self, path: Path, entries", 1)[1]
    init = init.split("\n    def ", 1)[0]
    # The advice is chosen, never spelled as a constant in the call.
    assert "madvise(_mmap." not in init
    assert "madvise(mmap." not in init
    assert "self.madvise_applied, _advice_value = madvise_choice()" in init
    assert "mm._mmap.madvise(_advice_value)" in init
    assert "self.prewarm_at_load = self._prewarm_table(path)" in init
    prewarm = MODEL_TEXT.split("def _prewarm_table", 1)[1].split("\n    def ", 1)[0]
    assert "run_prewarm(" in prewarm
    assert "record_prewarm(" in prewarm
    assert "format_prewarm_plan(receipt)" in prewarm


