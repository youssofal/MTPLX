"""Recurrent checkpoint retention keeps explicit anchors instead of re-thinning.

History, in order:

* 2026-07-17, Hermes lane (MEASUREMENTS 01:05 §B): the pop(1) policy ("keep
  the oldest plus a dense tail") ate the middle, and a ~22.5k near-miss
  restored at the OLDEST boundary, 2,048: a 20k+ re-prefill.
* Its replacement kept one record per power-of-two distance from the NEWEST
  record and re-thinned after every capture. 2026-09-29, Pi (B-cache 2.6):
  thinning again each time the newest record moved eroded the middle anyway,
  and a 16,371-token match restored at 4,096. B-cache reproduced it with the
  real function: a restore at 4,096 followed by a 42.8K prefill kept
  [4096, 34816, 38912, 40960, 43008, 45056, 46873, 46937].

``mtplx.checkpoint_anchors`` now keeps the prompt-end anchor, the restore
point, the stable edge and the newest checkpoint first, then the first
checkpoint of every 8,192-token cell (coarser cells first when the budget
cannot hold them all), then the newest of the rest. These tests drive the
real capture, inheritance and prefill-plan functions:

* the 16,371 case restores at the best anchor at or below 16,371, 8,192, not
  4,096 (fails on the old code);
* an agent session's churn does not erode deep coverage (fails on the old
  code);
* the 2026-07-17 near-tail case stays tight;
* coverage of a uniform grid stays proportional to the divergence;
* an inherited set keeps its restore point.

The old ``test_random_churn_stays_proportional`` pinned a property of the
retired policy on context-free lists: every divergence within ~4K of the
newest record re-prefills at most the divergence plus a slack, which a
cluster of records at the newest one provided. With the same eight records,
explicit anchors put that budget on the grid instead (the 16,371 case needs
it); near the prompt end they keep the prompt-end anchor, the restore point
(the previous prompt's end) and the newest checkpoint, which is where agent
turns diverge (B-cache 5).
"""

from __future__ import annotations

from types import SimpleNamespace

from mtplx.generation import (
    _capture_gdn_boundary,
    _inherited_gdn_boundaries,
    _prefill_spans_with_tail_grid,
)

CHUNK = 2048
INTERVAL = 256


def _rec(pos: int):
    return (pos, f"snap-{pos}", None)


def _at_or_below(kept_positions, matched):
    usable = [p for p in kept_positions if p <= matched]
    return max(usable) if usable else 0


def _capture_prefill(sink: list, *, start: int, prompt_len: int) -> None:
    """Every checkpoint one prefill of ``[start, prompt_len)`` captures: the
    span ends of the real chunk plan with its tail ladder, in order, through
    the real capture function (an empty cache snapshots to an empty record)."""

    body = prompt_len - start - 1
    for _span_start, end in _prefill_spans_with_tail_grid(
        body, tail_interval=INTERVAL, chunk_size=CHUNK
    ):
        _capture_gdn_boundary(sink, start + end, [])


def _positions(records) -> list[int]:
    return [int(record[0]) for record in records]


def test_the_16371_match_restores_at_8192_not_4096():
    """B-cache 2.6: the retry restored at 4,096 and prefilled to 46,938
    tokens; its next request matched 16,371. The old list's best boundary at
    or below 16,371 was 4,096; the grid keeps 8,192, and nothing past the
    match is ever a candidate.

    Eight checkpoints over 47K tokens only guarantee the 16K grid: 8,192
    survives here because the inherited set held one checkpoint. Had it also
    held 2,048, a production sink (which protects its restore point) would
    keep both in the first 8K cell and give the last slot to a newer cell, so
    this match would restore at 4,096 again; ten checkpoints restore it at
    8,192. The general bound is pinned in tests/test_checkpoint_anchors.py."""

    previous = SimpleNamespace(gdn_boundaries=[_rec(4096)])
    sink = list(_inherited_gdn_boundaries(previous, 4096))
    _capture_prefill(sink, start=4096, prompt_len=46938)
    kept = _positions(sink)
    assert len(kept) <= 8
    restore = _at_or_below(kept, 16371)
    assert restore == 8192, f"restored at {restore}; kept={kept}"
    # The anchors the next agent turn needs are there too.
    assert 46937 - 64 in kept and 46937 in kept


def _agent_session(first_prompt: int, deltas: list[int]) -> tuple[list, int]:
    """An agent session through the real capture and inheritance paths: a
    cold prefill, then each turn diverges five tokens before the previous
    prompt's end (the generation prompt re-rendered as the assistant turn),
    restores at the newest checkpoint at or below that, and prefills the
    rest of its longer prompt."""

    sink: list = []
    _capture_prefill(sink, start=0, prompt_len=first_prompt)
    prompt_len = first_prompt
    for delta in deltas:
        matched = prompt_len - 5
        restore = _at_or_below(_positions(sink), matched)
        assert 0 < restore <= matched
        sink = list(
            _inherited_gdn_boundaries(SimpleNamespace(gdn_boundaries=sink), restore)
        )
        prompt_len += delta
        _capture_prefill(sink, start=restore, prompt_len=prompt_len)
    return sink, prompt_len


# Thirty-six turns of 1,500 to 6,161 tokens after a 20K cold start end at
# 154,530 tokens.
DELTAS = [(1500 + (turn * 2731) % 5000) for turn in range(36)]


def test_agent_session_churn_does_not_erode_deep_coverage():
    """The old policy ended this session holding five records,
    [2048, 29627, 127221, 154428, 154529]: a match anywhere between 29,627
    and 127,221 re-prefilled up to 97,541 tokens. The anchors hold one
    checkpoint per 32,768-token cell through every turn."""

    sink, prompt_len = _agent_session(20000, DELTAS)
    kept = _positions(sink)
    assert prompt_len == 20000 + sum(DELTAS) == 154530
    assert len(kept) <= 8
    # Eight records over 154K: the first checkpoint, the prompt-end anchor,
    # the newest, one grid anchor per 32,768-token cell and the newest
    # 16,384-token cell's. A match anywhere re-prefills at most one 32K cell
    # plus the chunk the next cell's first checkpoint may sit in.
    bound = 32768 + CHUNK
    worst = max(
        (matched - _at_or_below(kept, matched), matched)
        for matched in range(CHUNK, prompt_len - 4, 97)
    )
    re_prefill, matched = worst
    assert re_prefill <= bound, (
        f"matched={matched} restored at {matched - re_prefill} "
        f"(re-prefill {re_prefill} > {bound}); kept={kept}"
    )


def test_retention_keeps_the_cap_and_both_ends():
    sink: list = []
    for pos in range(2048, 22785, 2048):
        _capture_gdn_boundary(sink, pos, [])
    positions = _positions(sink)
    assert len(positions) <= 8
    assert positions[0] == 2048  # the oldest anchor survives
    assert positions[-1] == 22528  # the newest survives
    assert positions == sorted(positions)


def _assert_proportional(kept_positions, all_positions, grid: int) -> None:
    newest = max(all_positions)
    slack = 2 * grid + 2 * 1024
    for matched in all_positions:
        if matched == newest:
            continue
        divergence = newest - matched
        re_prefill = matched - _at_or_below(kept_positions, matched)
        bound = divergence + slack if divergence <= 2 * grid else 8 * divergence + slack
        assert re_prefill <= bound, (
            f"matched={matched} (divergence {divergence}) re-prefills "
            f"{re_prefill} > {bound}; kept={kept_positions}"
        )


def test_coverage_is_proportional_on_a_uniform_grid():
    grid = 512
    sink: list = []
    for pos in range(grid, 64 * grid + 1, grid):
        _capture_gdn_boundary(sink, pos, [])
    _assert_proportional(_positions(sink), list(range(grid, 64 * grid + 1, grid)), grid)


def test_capture_churn_never_reopens_the_2048_cliff():
    """The 2026-07-17 shape: cold-prefill chunk edges, then postcommit-style
    rounds appending the same grid again. matched=22,509 on a 22,784-token
    entry once restored at 2,048 (74x the divergence)."""

    grid = 2048
    sink: list = []
    for pos in range(grid, 22531, grid):
        _capture_gdn_boundary(sink, pos, [])
    _capture_gdn_boundary(sink, 22530, [])
    for _ in range(3):
        for pos in range(grid, 22785, grid):
            _capture_gdn_boundary(sink, pos, [])
        _capture_gdn_boundary(sink, 22784, [])
    kept = _positions(sink)
    matched = 22509
    restore = _at_or_below(kept, matched)
    divergence = 22784 - matched
    assert matched - restore <= 3 * divergence + 2 * grid, (
        f"near-tail miss restored at {restore}; kept={kept}"
    )
    assert restore > 2048


def test_an_inherited_set_keeps_its_restore_point():
    entry = SimpleNamespace(gdn_boundaries=[_rec(p) for p in range(2048, 22785, 2048)])
    kept = _positions(_inherited_gdn_boundaries(entry, restore_point=20480))
    assert all(p <= 20480 for p in kept)
    assert len(kept) <= 8
    assert max(kept) == 20480
    _assert_proportional(kept, list(range(2048, 20481, 2048)), 2048)


def test_a_kept_checkpoint_lets_go_of_the_wider_buffer_it_was_cut_from():
    """A window visit's state during a chunked read is the last window of a wider prefill buffer
    (window + chunk - 1 rows on Inkling 2 Mini). A kept checkpoint is materialized when it is captured,
    so it holds its own rows only: left lazy, its copy kept the whole buffer alive after the read moved
    on (0.25 GB per checkpoint instead of 0.18 on Bare Speed, 1.7 GB of a 4K read, 21 November)."""
    import gc

    import mlx.core as mx

    from mtplx.checkpoint_anchors import record_nbytes
    from mtplx.models.inkling.cache import WindowCache

    window, chunk, heads, dim = 64, 512, 8, 128
    gc.collect()
    base = mx.get_active_memory()
    ring = WindowCache(max_size=window)
    keys = mx.ones((1, heads, chunk, dim), dtype=mx.bfloat16)
    values = mx.ones((1, heads, chunk, dim), dtype=mx.bfloat16) * 2
    ring.update_and_fetch(keys, values)
    mx.eval(ring.keys, ring.values)
    del keys, values
    assert ring.keys.shape[2] == chunk > window
    buffer_bytes = ring.keys.nbytes + ring.values.nbytes
    sink: list = []
    _capture_gdn_boundary(sink, chunk, [ring])
    assert [record[0] for record in sink] == [chunk]
    del ring  # the read moves on: the next chunk replaces the buffer
    gc.collect()
    held = mx.get_active_memory() - base
    kept = record_nbytes(sink[0])
    assert kept == buffer_bytes * window // chunk
    assert held < buffer_bytes // 2, f"the checkpoint holds {held} bytes, its rows are {kept}"
    keys_copy, values_copy = sink[0][1].states[0]
    assert mx.array_equal(keys_copy, mx.ones((1, heads, window, dim), dtype=mx.bfloat16)).item()
    assert mx.array_equal(values_copy, mx.full((1, heads, window, dim), 2, dtype=mx.bfloat16)).item()
