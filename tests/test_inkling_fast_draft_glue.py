"""The drafter's glue in launches of ours (fast/draft_glue.py), GPU.

Every launch must give the library expression it replaces bit for bit, at the drafter's shapes
(8 rows of 2,304; heads of 128; an MLP of 9,216; a Markov rank of 256) in bf16 and in the FP16
siblings' float16: the add and norm with both outputs, the gate, the Markov input and update, the
attention with its layout changes (a 1,024-slot ring and banks), the heads normed before their
transpose, and a draw's position offset made in the kernel.
"""

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

D, R, HQ, HK, HD, FF, RANK = 2304, 8, 24, 8, 128, 9216, 256


def _bits(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


@pytest.fixture(autouse=True)
def fresh_verdicts(monkeypatch):
    from mtplx.models.inkling.fast import draft_glue

    monkeypatch.setattr(draft_glue, "_VERDICT", {})
    yield


@needs
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("gain_dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("scale", [0.02, 1.0, 40.0])
def test_add_norm_equals_the_library(dtype, gain_dtype, scale):
    from mtplx.models.inkling.fast import draft_glue as G

    key = mx.random.key(int(scale * 100) + (dtype == mx.float16) * 7 + (gain_dtype == mx.float32) * 3)
    k1, k2, k3 = mx.random.split(key, 3)
    a = (mx.random.normal((1, R, D), key=k1) * scale).astype(dtype)
    b = (mx.random.normal((1, R, D), key=k2) * scale * 0.5).astype(dtype)
    gain = (1 + 0.2 * mx.random.normal((D,), key=k3)).astype(gain_dtype)
    summed = a.astype(mx.float32) + b.astype(mx.float32)
    want = (mx.fast.rms_norm(summed, gain, 1e-6).astype(dtype), summed.astype(dtype))
    got = G.add_norm(a, b, gain, 1e-6)
    assert G._VERDICT, "the launch was not tried"
    assert all(v for v in G._VERDICT.values()), f"refused: {G._VERDICT}"
    assert _bits(got[0], want[0]) and _bits(got[1], want[1])


@needs
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_gate_equals_the_library(dtype):
    import mlx.nn as nn

    from mtplx.models.inkling.fast import draft_glue as G

    k1, k2 = mx.random.split(mx.random.key(5 + (dtype == mx.float16)))
    gate = (mx.random.normal((1, R, FF), key=k1) * 4).astype(dtype)
    up = (mx.random.normal((1, R, FF), key=k2) * 2).astype(dtype)
    want = (nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)).astype(dtype)
    got = G.gate_act(gate, up)
    assert all(v for v in G._VERDICT.values()) and G._VERDICT
    assert _bits(got, want)


@needs
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("zero_state", [True, False])
def test_markov_step_equals_the_library(dtype, zero_state):
    from mtplx.models.inkling.fast import draft_glue as G
    from mtplx.models.inkling.fast import round as fast_round

    keys = mx.random.split(mx.random.key(11 + zero_state + (dtype == mx.float16) * 2), 5)
    state = (mx.zeros((1, RANK)) if zero_state else mx.random.normal((1, RANK), key=keys[0])).astype(dtype)
    table = mx.random.normal((5000, RANK), key=keys[1]).astype(dtype)
    prev = mx.array([4321], mx.int32)
    want_in = mx.concatenate([state, table[prev]], axis=-1).astype(mx.float32)
    assert _bits(G.markov_input(state, table, prev), want_in)
    mv = mx.random.normal((1, 3 * RANK), key=keys[2]) * 3
    cond = mx.random.normal((1, 7, 3 * RANK), key=keys[3]) * 2
    one = fast_round._const(1.0, dtype)
    for i in range(7):
        row = cond[:, i]
        update = (mv + row).astype(dtype)
        g, c, o = mx.split(update, 3, axis=-1)
        g = mx.sigmoid(g.astype(mx.float32)).astype(dtype)
        want_state = g * state + (one - g) * mx.tanh(c.astype(mx.float32)).astype(dtype)
        want_signal = mx.tanh(o.astype(mx.float32)).astype(dtype)
        got_state, got_signal = G.markov_update(mv, row, state, one)
        assert _bits(got_state, want_state) and _bits(got_signal, want_signal), f"step {i}"
        state = got_state
    assert G._VERDICT and all(G._VERDICT.values())


@needs
@pytest.mark.parametrize("ring,slots,count", [(True, 1024, 1024), (True, 1024, 300), (False, 4096, 4142 - 4096 + 4000),
                                              (False, 8192, 5000)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_attention_equals_the_round_programs(ring, slots, count, dtype):
    from mtplx.models.inkling.fast import draft_glue as G
    from mtplx.models.inkling.fast import round_program as rp

    keys_ = mx.random.split(mx.random.key(slots + count + (dtype == mx.float16)), 5)
    q = (mx.random.normal((1, HQ, R, HD), key=keys_[0])).astype(dtype)
    hk = (mx.random.normal((1, HK, slots, HD), key=keys_[1])).astype(dtype)
    hv = (mx.random.normal((1, HK, slots, HD), key=keys_[2])).astype(dtype)
    own_k = (mx.random.normal((1, HK, R, HD), key=keys_[3])).astype(dtype)
    own_v_rows = (mx.random.normal((1, R, HK, HD), key=keys_[4])).astype(dtype)
    cnt = mx.array(min(count, slots), mx.uint32)
    newest = mx.array(slots - 7 if ring else 0, mx.uint32)
    want = rp.draft_attention(q, hk, hv, own_k, own_v_rows.transpose(0, 2, 1, 3), cnt, newest, ring=ring)
    want = want.transpose(0, 2, 1, 3).reshape(1, R, HQ * HD)
    got = G.attention(q, hk, hv, own_k, own_v_rows, cnt, newest, ring=ring)
    assert G._VERDICT and all(G._VERDICT.values()), f"refused: {G._VERDICT}"
    assert _bits(got, want)


@needs
@pytest.mark.parametrize("heads", [HQ, HK])
@pytest.mark.parametrize("pos", [0, 1, 4141, 1 << 20])
def test_heads_normed_before_the_transpose(heads, pos):
    k1, k2 = mx.random.split(mx.random.key(heads + pos))
    x = (mx.random.normal((1, R, heads * HD), key=k1) * 2).astype(mx.bfloat16)
    w = (1 + 0.1 * mx.random.normal((HD,), key=k2)).astype(mx.bfloat16)
    off = mx.array(pos, mx.int32)
    rope = lambda y: mx.fast.rope(y, HD, traditional=False, base=1_000_000.0, scale=1.0, offset=off)
    want = rope(mx.fast.rms_norm(x.reshape(1, R, heads, HD).transpose(0, 2, 1, 3), w, 1e-6))
    got = rope(mx.fast.rms_norm(x.reshape(1, R, heads, HD), w, 1e-6).transpose(0, 2, 1, 3))
    assert _bits(got, want)


@needs
@pytest.mark.parametrize("greedy", [True, False])
def test_a_draws_position_offset_is_the_same_draw(greedy):
    from mtplx.models.inkling.fast import round as fast_round

    scores = mx.random.normal((1, 201024), key=mx.random.key(3)) * 3
    law, st = fast_round.law_rows(scores, temperature=None if greedy else 1.0, cap=0.0, write=True)
    start = mx.array(4141, mx.uint32)
    seed = fast_round._seed_words(0xBEEF)
    for i in range(7):
        want = fast_round.sample_rows(law, st, positions=(start + fast_round._const(i, mx.uint32)).reshape(1),
                                      seed=seed, purpose=fast_round.PURPOSE_DRAFT, greedy=greedy)
        got = fast_round.sample_rows(law, st, positions=start.reshape(1), seed=seed,
                                     purpose=fast_round.PURPOSE_DRAFT, greedy=greedy, offset=i)
        assert _bits(got[0], want[0]) and _bits(got[1], want[1]), f"step {i}"


def test_switch_off_serves_the_round_programs_propose(monkeypatch):
    from mtplx.models.inkling.fast import draft_glue

    monkeypatch.setenv("MTPLX_INKLING_DRAFT_GLUE", "0")
    assert not draft_glue.enabled()
    monkeypatch.delenv("MTPLX_INKLING_DRAFT_GLUE")
    assert draft_glue.enabled()
