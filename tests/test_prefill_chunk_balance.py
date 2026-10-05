"""Prefill chunk plan: no final chunk below the fused-SDPA row threshold."""

import mtplx.generation as G


def _plan(total, chunk=4096, min_rows=1024):
    spans = [(s, min(total, s + chunk)) for s in range(0, total, chunk)]
    return G._merge_small_remainder(spans, min_rows)


def test_off_is_plain_grid():
    assert _plan(5000, min_rows=0) == [(0, 4096), (4096, 5000)]


def test_small_remainder_is_merged():
    assert _plan(4999) == [(0, 4999)]
    assert _plan(9100) == [(0, 4096), (4096, 9100)]


def test_remainder_wide_enough_untouched():
    assert _plan(6000) == [(0, 4096), (4096, 6000)]
    assert _plan(5120) == [(0, 4096), (4096, 5120)]


def test_single_chunk_untouched():
    assert _plan(839) == [(0, 839)]
    assert _plan(4096) == [(0, 4096)]


def test_merged_chunk_never_exceeds_chunk_plus_threshold():
    for total in range(4097, 4097 + 3 * 4096, 37):
        spans = _plan(total)
        assert all(e - s < 4096 + 1024 for s, e in spans)
        assert spans[0][0] == 0 and spans[-1][1] == total
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        assert len(spans) == 1 or spans[-1][1] - spans[-1][0] >= 1024


def test_iter_spans_env_switch(monkeypatch):
    monkeypatch.delenv("MTPLX_PREFILL_MIN_CHUNK_ROWS", raising=False)
    assert G._iter_prefill_chunk_spans(4999, chunk_size=4096) == [
        (0, 4096),
        (4096, 4999),
    ]
    monkeypatch.setenv("MTPLX_PREFILL_MIN_CHUNK_ROWS", "1024")
    assert G._iter_prefill_chunk_spans(4999, chunk_size=4096) == [(0, 4999)]


def test_ladder_keeps_tail_boundaries_with_merged_plan(monkeypatch):
    monkeypatch.setenv("MTPLX_PREFILL_MIN_CHUNK_ROWS", "1024")
    spans = G._prefill_spans_with_tail_grid(
        4999, tail_interval=256, chunk_size=4096
    )
    assert spans[-1] == (4935, 4999)
    assert min(e - s for s, e in spans[:-1]) >= 1024
    assert max(e - s for s, e in spans) <= 4096
