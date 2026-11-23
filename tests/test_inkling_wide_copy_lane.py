"""Regression tests for the Inkling wide prompt-copy verify lane (Step 1).

Verifies bit-exact Metal kernels across W=1..32 for Stage C attention output,
single-launch vocabulary head up to 32 rows, single-launch row_stable_head,
GPU wide prompt-copy decisions enabled by default, 8-row verify tile snapping,
and warm-up coverage across (1..8, 16, 24).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.models.inkling import head
from mtplx.models.inkling.copy_draft import PromptCopy
from mtplx.models.inkling.fast import attn_output, engine_site, round as fast_round, step
from mtplx.models.inkling import runtime as runtime_module


def _bits_equal(a: mx.array, b: mx.array) -> bool:
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal GPU")
@pytest.mark.parametrize("rows", [1, 4, 8, 9, 13, 16, 17, 24, 25, 32])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("boundary", [False, True], ids=["interior", "boundary"])
def test_stage_c_attn_output_chunked_register_loop_is_bit_exact(rows, dtype, boundary):
    """Chunking Stage C into RCH<=8 register slices must produce 0-ULP bit-identical
    results across all panel widths W=1..32 vs running W=1 row-by-row with chained tail."""
    out_dim = 64
    in_dim = 256
    hg = (mx.random.normal((1, rows, in_dim), key=mx.random.key(1)) * 0.25).astype(dtype)
    out_weight = (mx.random.normal((out_dim, in_dim), key=mx.random.key(2)) * 0.1).astype(dtype)
    conv_weight = (mx.random.normal((4, out_dim), key=mx.random.key(3)) * 0.2).astype(mx.float32)
    tail = (mx.random.normal((1, 3, out_dim), key=mx.random.key(4)) * 0.25).astype(mx.float32)
    x = (mx.random.normal((1, rows, out_dim), key=mx.random.key(5)) * 0.25).astype(mx.float32)
    mx.eval(hg, out_weight, conv_weight, tail, x)

    partial, conv_rows = attn_output.attention_output(
        hg, out_weight, conv_weight, tail, x, boundary=boundary
    )
    mx.eval(partial, conv_rows)

    ref_partials = []
    ref_conv_rows = []
    cur_tail = tail
    for r in range(rows):
        p_r, c_r = attn_output.attention_output(
            hg[:, r : r + 1, :],
            out_weight,
            conv_weight,
            cur_tail,
            x[:, r : r + 1, :],
            boundary=boundary,
        )
        mx.eval(p_r, c_r)
        ref_partials.append(p_r)
        ref_conv_rows.append(c_r)
        cur_tail = mx.concatenate([cur_tail[:, 1:, :], c_r], axis=1)

    want_partial = mx.concatenate(ref_partials, axis=1)
    want_conv = mx.concatenate(ref_conv_rows, axis=1)
    mx.eval(want_partial, want_conv)

    assert _bits_equal(partial, want_partial), f"partial mismatch at rows={rows}, {dtype}, boundary={boundary}"
    assert _bits_equal(conv_rows, want_conv), f"conv_rows mismatch at rows={rows}, {dtype}, boundary={boundary}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal GPU")
@pytest.mark.parametrize("rows", [1, 8, 16, 24, 32])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("cap", [None, 30.0], ids=["uncapped", "capped"])
def test_wide_head_logits_is_native_and_bit_exact(rows, dtype, cap, monkeypatch):
    """head_logits supports panels up to MAX_PANEL_ROWS=32 on the native Metal path with 0 ULP drift."""
    assert head.MAX_PANEL_ROWS == 32
    monkeypatch.setenv("MTPLX_INKLING_HEAD_PANEL", "1")
    # Pin singleton/panel route selection like test_inkling_pack_parity so tiny test matrices
    # deterministically exercise _native/_capped rather than losing a micro-race to MLX matmul.
    monkeypatch.setattr(head, "measured_route", lambda *args, **kw: True)
    width = 256
    vocab = 1024
    h = (mx.random.normal((1, rows, width), key=mx.random.key(10)) * 0.25).astype(dtype)
    weight = (mx.random.normal((vocab, width), key=mx.random.key(11)) * 0.05).astype(dtype)
    mx.eval(h, weight)

    logits, route = head.head_logits(h, weight, vocab, cap)
    mx.eval(logits)
    assert route == "native"

    row_refs = []
    for r in range(rows):
        ref_r, ref_route = head.head_logits(h[:, r : r + 1, :], weight, vocab, cap)
        assert ref_route == "native"
        row_refs.append(ref_r)
    want = mx.concatenate(row_refs, axis=1)
    mx.eval(want)
    assert _bits_equal(logits, want), f"head_logits ULP drift at rows={rows}, {dtype}, cap={cap}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal GPU")
@pytest.mark.parametrize("rows", [16, 24, 32])
def test_row_stable_head_runs_wide_panel_in_single_native_launch(rows, monkeypatch):
    """row_stable_head must not split 9..32 row native panels into 8-row chunks."""
    monkeypatch.setenv("MTPLX_INKLING_HEAD_PANEL", "1")
    monkeypatch.setattr(head, "measured_route", lambda *args, **kw: True)
    dtype = mx.bfloat16
    width = 256
    vocab = 1024
    h = (mx.random.normal((1, rows, width), key=mx.random.key(20)) * 0.25).astype(dtype)
    weight = (mx.random.normal((vocab, width), key=mx.random.key(21)) * 0.05).astype(dtype)
    mx.eval(h, weight)
    model = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=mx.zeros((1,), dtype=dtype)),
        unembed=weight,
        config={"unpadded_vocab_size": vocab, "final_logit_softcapping": 30.0},
    )

    calls = []
    orig_head_logits = head.head_logits

    def tracked_head_logits(*args, **kwargs):
        calls.append(args[0].shape[1])
        return orig_head_logits(*args, **kwargs)

    monkeypatch.setattr(head, "head_logits", tracked_head_logits)
    logits = step.row_stable_head(model, h)
    mx.eval(logits)
    assert calls == [rows], f"expected single launch of {rows} rows, got {calls}"


def test_row_stable_head_stock_fallback_chunks_without_double_counting(monkeypatch):
    """If head_logits falls back to 'stock', row_stable_head must chunk at 8 rows cleanly."""
    dtype = mx.float32
    h = mx.zeros((1, 24, 8), dtype=dtype)
    model = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=mx.zeros((1,), dtype=dtype)),
        unembed=mx.zeros((16, 8), dtype=dtype),
        config={"unpadded_vocab_size": 16, "final_logit_softcapping": None},
    )
    chunk_widths = []
    counted = []

    def fake_head_logits(hidden, *_a, **_kw):
        chunk_widths.append(hidden.shape[1])
        return mx.zeros((1, hidden.shape[1], 16), dtype=mx.float32), "stock"

    monkeypatch.setattr(head, "head_logits", fake_head_logits)
    monkeypatch.setattr(step, "count", lambda name: counted.append(name))

    out = step.row_stable_head(model, h)
    mx.eval(out)
    assert out.shape == (1, 24, 16)
    # 1 probe + three 8-row chunks (counted once per chunk, never double-counted on probe).
    assert chunk_widths == [24, 8, 8, 8]
    assert counted == ["fallback.head.table_stock"] * 3


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal GPU")
@pytest.mark.parametrize("rows", [16, 24])
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_wide_decide_matches_reference_at_copy_widths(rows, temperature):
    """GPU decide() at wide copy widths W=16 and W=24 agrees with reference_decide()."""
    vocab = 256
    logits = (mx.random.normal((rows, vocab), key=mx.random.key(30 + rows)) * 2.0).astype(mx.float32)
    proposals = mx.argmax(logits[:-1], axis=-1).astype(mx.int32)
    mx.eval(logits, proposals)

    pkt = fast_round.decide(
        logits,
        proposals,
        position=100,
        seed=42,
        temperature=temperature,
        draft=None,
    )
    ref_out, _ref_probs = fast_round.reference_decide(
        logits,
        proposals,
        position=100,
        seed=42,
        temperature=temperature,
        draft_rows=None,
    )
    mx.eval(pkt)
    n = int(pkt[0].item())
    status = int(pkt[1].item())
    assert status == 0
    assert list(pkt[2 : 2 + n + 1].tolist()) == ref_out


def test_align_verify_proposals_snaps_unwarmed_copy_widths():
    """Every wide proposal count past 7 aligns to an 8-row tile boundary."""
    # verify_rows in 2..8 stay untouched
    for length in range(1, 8):
        assert PromptCopy.align_verify_proposals(length, 31) == length
    # Unaligned / intermediate widths snap down so (proposals + 1) % 8 == 0
    expected = {
        8: 7,   # w9 -> w8 (probation)
        12: 7,  # w13 -> w8
        15: 15, # w16 exact
        16: 15, # w17 -> w16
        18: 15, # w19 -> w16
        20: 15, # w21 -> w16
        23: 23, # w24 exact
        24: 23, # w25 -> w24
        31: 31, # w32 exact
    }
    for length, want in expected.items():
        assert PromptCopy.align_verify_proposals(length, 31) == want
    # Non-wide engines (max_proposals < 8) are left alone
    assert PromptCopy.align_verify_proposals(7, 7) == 7
    assert PromptCopy.align_verify_proposals(5, 7) == 5


def test_prompt_copy_propose_clips_grammar_and_eof_to_tile_widths():
    """Partial budget / end_of_prompt / grammar clips past 8 rows still land on w8/w16/w24."""
    prompt = list(range(1000, 1400))
    lane = PromptCopy(prompt, session_id=None, verify_rows=32)
    for _ in range(3):
        lane.observe(8, 8, 50, 0.0)
    # end_of_prompt truncates a 24-match down to 18 tokens (w19 -> w16 -> 15 proposals)
    proposals_eof = lane.propose(prompt[-26:-18], 100)
    assert proposals_eof is not None
    assert len(proposals_eof) == 15 and (len(proposals_eof) + 1) % 8 == 0
    # remaining budget = 20 (w21 -> w16 -> 15 proposals)
    proposals_rem = lane.propose(prompt[:8], 20)
    assert proposals_rem is not None
    assert len(proposals_rem) == 15 and (len(proposals_rem) + 1) % 8 == 0


def test_inkling_runtime_warms_all_wide_copy_widths(monkeypatch):
    """InklingRuntime warms both 1..8 and wide copy widths (16, 24) on startup."""
    monkeypatch.setattr(runtime_module, "wide_rows_exact", lambda model, spec: True)
    assert runtime_module.wide_copy_widths(None) == (16, 24)
    assert runtime_module.widest_panel(None) == 24
