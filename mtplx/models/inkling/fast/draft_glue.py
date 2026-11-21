"""The drafter's glue in launches of ours (lane F1): the round program's DSpark drafter between its
products, the same values in fewer launches.

The round program's drafter (``round_program.propose``) spends about 167 of its 206 launches a
round on the library's small kernels between its products: per block an add of two bf16 rows in
float32 with three casts, the RMS norm and a cast back (twice a block, once more after the last
block), copies of the strided query and key heads before their norms, a copy of the value heads and
of the attention output, the two-scalar stack the attention reads, the gate's casts, SiLU and
product; per proposal the gather and joins of the Markov input, its cast, and about fifteen
elementwise launches of the gate arithmetic. Here:

* ``add_norm``: the add, the norm and both casts in one launch, which also writes the sum rounded
  to the drafter's type (the block's carried residual): one threadgroup a row, MLX's single-row
  RMS norm as ``commit_launches.mix_norm`` reproduces it (four values a thread, ``simd_sum``, the
  simdgroup totals summed zero-padded, ``precise::rsqrt``, the gain applied as ``w * (x * inv)``).
* The query and key heads normed in the projection's own layout, then transposed (MLX's rotation
  takes the strided view and writes a row-contiguous result): the same norm of the same rows, no
  copy (``f1/heads_probe.py``: bits equal, the copies gone).
* ``attention``: ``round_program.draft_attention`` derived from its own kernel text with three
  changes that move no value: the row count and the newest slot read as two scalars (no stack), the
  block's own value rows read in the projection's layout (no copy), the output written as
  ``(rows, heads * dim)`` (the output projection reads it without a copy).
* ``gate_act``: ``(silu(gate) * up)`` with its casts, one launch.
* ``markov_input`` and ``markov_update``: a proposal step's Markov input (the state and the table
  row of the previous token, widened) and everything between its two products (the add of the
  conditioned row, the cast, the gate's sigmoid, the candidate's tanh, the state update in the
  drafter's type, the signal's tanh), one launch each.

Every elementwise operation is the library's own: its ``Add``, ``Subtract``, ``Multiply``,
``Sigmoid`` and ``Tanh`` are read from the installed MLX's headers when a launch is built
(``library_gemm.library("elementwise")``) and applied to the same types the library's launches
apply them to, so each value is rounded where the library rounds it.

Class A against ``round_program.propose`` (the drafter's bits, hence the proposals, unchanged):
each launch is decided per shape on this Mac's first use against the library expression it
replaces, bit for bit, and refused (counted ``fallback.draft_glue.<launch>``, the library's
launches kept) otherwise. ``MTPLX_INKLING_DRAFT_GLUE=0`` serves ``round_program.propose``.
"""

from __future__ import annotations

import os
import struct
from functools import lru_cache

import mlx.core as mx

from .routes import count


def enabled() -> bool:
    return os.environ.get("MTPLX_INKLING_DRAFT_GLUE", "1").strip().lower() not in ("0", "off", "false", "no", "")


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


#: (launch, shape...) -> True once the launch has given the library expression's bits on this Mac.
_VERDICT: dict = {}
_TYPES = (mx.bfloat16, mx.float16)


def _same(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


def _decided(key, ours, stock):
    """``ours()`` where it has given ``stock()``'s bits for ``key`` on this Mac (decided on the first
    call, evaluated then only), else ``stock()``; ``ours``/``stock`` return an array or a tuple."""
    verdict = _VERDICT.get(key)
    if verdict is None:
        try:
            got, want = ours(), stock()
            got_t = got if isinstance(got, tuple) else (got,)
            want_t = want if isinstance(want, tuple) else (want,)
            mx.eval(*got_t, *want_t)
            verdict = len(got_t) == len(want_t) and all(_same(a, b) for a, b in zip(got_t, want_t))
        except (RuntimeError, ValueError):
            verdict = False
        _VERDICT[key] = verdict
        from .attn_input import count_route

        count_route(("proved:" if verdict else "refused:") + "draft_glue_" + "_".join(map(str, key)))
    if not verdict:
        count(f"fallback.draft_glue.{key[0]}")
        return stock()
    return ours()


@lru_cache(maxsize=1)
def _library():
    from .library_gemm import library

    return library("elementwise")


# ---------------------------------------------------------------------------------- add + norm
_ADD_NORM = """
    constexpr uint NSG = TPG / 32;
    const uint t = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.x;
    threadgroup float sums[32];
    threadgroup float inv[1];
    const uint c4 = t * 4;
    const bool in_row = c4 < D;
    float x[4];
    for (uint i = 0; i < 4; ++i) {
        const size_t at = size_t(r) * D + c4 + i;
        x[i] = in_row ? Add{}(static_cast<float>(a[at]), static_cast<float>(b[at])) : 0.0f;
    }
    float acc = 0;
    for (uint i = 0; i < 4; ++i) {
        const float xval = in_row ? x[i] : 0.0f;
        acc += xval * xval;
    }
    acc = simd_sum(acc);
    if (lane == 0) sums[sg] = acc;
    if (sg == 0 && lane >= NSG) sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float total = simd_sum(sums[lane]);
        if (lane == 0) inv[0] = metal::precise::rsqrt(total / float(D) + as_type<float>(uint(EPS)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float iv = inv[0];
    if (in_row)
        for (uint i = 0; i < 4; ++i) {
            const size_t at = size_t(r) * D + c4 + i;
            const float w = static_cast<float>(gain[c4 + i]);
            normed[at] = static_cast<T>(w * (x[i] * iv));
            summed[at] = static_cast<T>(x[i]);
        }
"""


@lru_cache(maxsize=None)
def _add_norm_kernel():
    text, _why = _library()
    return mx.fast.metal_kernel(name="mtplx_inkling_f1_add_norm", input_names=["a", "b", "gain"],
                                output_names=["normed", "summed"], source=_ADD_NORM, header=text)


def add_norm(a, b, gain, eps: float):
    """``(rms_norm(a.f32 + b.f32, gain, eps).astype(T), (a.f32 + b.f32).astype(T))`` for two rows of the
    drafter's type ``T`` (bf16 or f16) of one shape ``(1, R, D)``: one launch where proven."""
    width = int(a.shape[-1])

    def stock():
        summed = a.astype(mx.float32) + b.astype(mx.float32)
        return mx.fast.rms_norm(summed, gain, eps).astype(a.dtype), summed.astype(a.dtype)

    if (a.dtype not in _TYPES or b.dtype != a.dtype or a.shape != b.shape or width % 128 or width > 4096
            or gain.size != width or _library()[0] is None or not mx.metal.is_available()
            or mx.default_device() != mx.gpu):
        count("fallback.draft_glue.add_norm_shape")
        return stock()
    rows = a.size // width
    tpg = -(-(width // 4) // 32) * 32

    def ours():
        normed, summed = _add_norm_kernel()(
            inputs=[a, b, gain],
            template=[("T", a.dtype), ("D", width), ("TPG", tpg), ("EPS", _f32_bits(eps))],
            grid=(rows * tpg, 1, 1),
            threadgroup=(tpg, 1, 1),
            output_shapes=[a.shape, a.shape],
            output_dtypes=[a.dtype, a.dtype],
        )
        return normed, summed

    return _decided(("add_norm", rows, width, str(a.dtype), str(gain.dtype), _f32_bits(eps)), ours, stock)


# --------------------------------------------------------------------------- the MLP's gate
_GATE = """
    // silu(g) * u in float32 (the library's Sigmoid and Multiply), rounded once to the drafter's type.
    const uint e = thread_position_in_grid.x;
    if (e >= N) return;
    const float g = static_cast<float>(gate[e]), u = static_cast<float>(up[e]);
    const float act = Multiply{}(Multiply{}(g, Sigmoid{}(g)), u);
    gated[e] = static_cast<T>(act);
"""


@lru_cache(maxsize=None)
def _gate_kernel():
    text, _why = _library()
    return mx.fast.metal_kernel(name="mtplx_inkling_f1_gate_act", input_names=["gate", "up"],
                                output_names=["gated"], source=_GATE, header=text)


def gate_act(gate, up):
    """``(silu(gate.f32) * up.f32).astype(T)`` (``nn.silu`` is ``x * sigmoid(x)``): one launch where
    proven. ``gate``, ``up``: the drafter's type, one shape."""
    import mlx.nn as nn

    def stock():
        return (nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)).astype(gate.dtype)

    if gate.dtype not in _TYPES or up.dtype != gate.dtype or up.shape != gate.shape or _library()[0] is None:
        count("fallback.draft_glue.gate_shape")
        return stock()
    n = gate.size

    def ours():
        return _gate_kernel()(
            inputs=[gate, up],
            template=[("T", gate.dtype), ("N", n)],
            grid=(n, 1, 1),
            threadgroup=(min(256, n), 1, 1),
            output_shapes=[gate.shape],
            output_dtypes=[gate.dtype],
        )[0]

    return _decided(("gate", n, str(gate.dtype)), ours, stock)


# ----------------------------------------------------------------------------- the Markov walk
_MARKOV_INPUT = """
    const uint j = thread_position_in_grid.x;
    if (j >= 2 * R) return;
    out[j] = j < R ? static_cast<float>(state[j]) : static_cast<float>(table[size_t(prev[0]) * R + (j - R)]);
"""

_MARKOV_UPDATE = """
    const uint j = thread_position_in_grid.x;
    if (j >= R) return;
    const T ug = static_cast<T>(Add{}(mv[j], cond[j]));
    const T uc = static_cast<T>(Add{}(mv[R + j], cond[R + j]));
    const T uo = static_cast<T>(Add{}(mv[2 * R + j], cond[2 * R + j]));
    const T gate = static_cast<T>(Sigmoid{}(static_cast<float>(ug)));
    const T cand = static_cast<T>(Tanh{}(static_cast<float>(uc)));
    const T st = state[j];
    const T one = static_cast<T>(1.0f);
    const T keep = Multiply{}(gate, st);
    const T rest = Multiply{}(Subtract{}(one, gate), cand);
    state_out[j] = Add{}(keep, rest);
    signal[j] = static_cast<T>(Tanh{}(static_cast<float>(uo)));
"""


@lru_cache(maxsize=None)
def _markov_kernels():
    text, _why = _library()
    first = mx.fast.metal_kernel(name="mtplx_inkling_f1_markov_input", input_names=["state", "table", "prev"],
                                 output_names=["out"], source=_MARKOV_INPUT, header=text)
    second = mx.fast.metal_kernel(name="mtplx_inkling_f1_markov_update", input_names=["mv", "cond", "state"],
                                  output_names=["state_out", "signal"], source=_MARKOV_UPDATE, header=text)
    return first, second


def markov_input(state, table, prev):
    """``mx.concatenate([state, table[prev]], -1).astype(float32)``: ``state`` ``(1, R)`` and ``table``
    ``(V, R)`` in the drafter's type, ``prev`` ``(1,)`` int32 -> ``(1, 2 R)`` float32, one launch."""
    rank = int(state.shape[-1])

    def stock():
        return mx.concatenate([state, table[prev]], axis=-1).astype(mx.float32)

    if (state.dtype not in _TYPES or table.dtype != state.dtype or state.shape != (1, rank) or table.ndim != 2
            or table.shape[1] != rank or prev.dtype != mx.int32 or prev.size != 1 or _library()[0] is None):
        count("fallback.draft_glue.markov_input_shape")
        return stock()

    def ours():
        return _markov_kernels()[0](
            inputs=[state, table, prev.reshape(1)],
            template=[("R", rank)],
            grid=(2 * rank, 1, 1),
            threadgroup=(min(256, 2 * rank), 1, 1),
            output_shapes=[(1, 2 * rank)],
            output_dtypes=[mx.float32],
        )[0]

    return _decided(("markov_input", rank, str(state.dtype)), ours, stock)


def markov_update(mv, cond, state, one):
    """One proposal step between its two products: ``update = (mv + cond).astype(T)``, split in three;
    ``state' = sigmoid(g) * state + (one - sigmoid(g)) * tanh(c)`` in ``T`` (each factor cast to ``T``
    as the round program casts it), ``signal = tanh(o).astype(T)``. ``mv``, ``cond`` ``(1, 3 R)``
    float32, ``state`` ``(1, R)`` in ``T``, ``one`` the round's kept ``T`` constant 1.0
    (``round._const(1.0, T)``; the launch uses the same value) -> ``(state', signal)``."""
    from . import round as fast_round

    rank = int(state.shape[-1])

    def stock():
        update = (mv + cond).astype(state.dtype)
        gate, candidate, output = mx.split(update, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(state.dtype)
        new = gate * state + (one - gate) * mx.tanh(candidate.astype(mx.float32)).astype(state.dtype)
        signal = mx.tanh(output.astype(mx.float32)).astype(state.dtype)
        return new, signal

    if (state.dtype not in _TYPES or state.shape != (1, rank) or mv.dtype != mx.float32 or cond.dtype != mx.float32
            or mv.size != 3 * rank or cond.size != 3 * rank or one is not fast_round._const(1.0, state.dtype)
            or _library()[0] is None):
        count("fallback.draft_glue.markov_update_shape")
        return stock()

    def ours():
        new, signal = _markov_kernels()[1](
            inputs=[mv.reshape(1, 3 * rank), cond.reshape(1, 3 * rank), state],
            template=[("T", state.dtype), ("R", rank)],
            grid=(rank, 1, 1),
            threadgroup=(min(256, rank), 1, 1),
            output_shapes=[(1, rank), (1, rank)],
            output_dtypes=[state.dtype, state.dtype],
        )
        return new, signal

    return _decided(("markov_update", rank, str(state.dtype)), ours, stock)


# ------------------------------------------------------------------------------ the attention
_REWRITES = {
    "attend": (
        ("const uint count = meta[0], newest = meta[1];", "const uint count = cnt, newest = nw;"),
        ("vt[i] = own ? ov[(h * R + j) * D + d] : hv[(h * S + j) * D + d];",
         "vt[i] = own ? ov[(j * HK + h) * D + d] : hv[(h * S + j) * D + d];"),
    ),
    "merge": (
        ("out[((h * G + qv / R) * R + qv % R) * D + d] = static_cast<T>(o / lsum);",
         "out[((qv % R) * (HK * G) + h * G + qv / R) * D + d] = static_cast<T>(o / lsum);"),
    ),
}


def attend_texts():
    """``({"attend": text, "merge": text}, None)``: ``round_program``'s two drafter attention texts with
    the three layout changes; or ``(None, the line not found)`` when its text no longer holds a line
    they rewrite exactly once (that fails ``tests/test_inkling_fast_draft_glue.py``'s text test; in
    serving the library's attention serves, counted ``fallback.draft_glue.attention_shape``)."""
    from . import round_program as rp

    texts = {"attend": rp._ATTEND, "merge": rp._MERGE}
    for name, pairs in _REWRITES.items():
        for old, new in pairs:
            if texts[name].count(old) != 1:
                return None, f"{name}: {old}"
            texts[name] = texts[name].replace(old, new)
    return texts, None


@lru_cache(maxsize=1)
def _attend_kernels():
    """``round_program``'s two drafter attention launches with the three layout changes, or None when
    its kernel text no longer holds the lines they rewrite."""
    texts, _missing = attend_texts()
    if texts is None:
        return None
    partials = mx.fast.metal_kernel(name="mtplx_inkling_f1_draft_attend", input_names=["q", "hk", "hv", "ok", "ov",
                                                                                      "cnt", "nw"],
                                    output_names=["pm", "pl", "pacc"], source=texts["attend"])
    merge = mx.fast.metal_kernel(name="mtplx_inkling_f1_draft_merge", input_names=["pm", "pl", "pacc"],
                                 output_names=["out"], source=texts["merge"])
    return partials, merge


def attention(q, keys, values, own_k, own_v_rows, count_rows, newest, *, ring: bool):
    """``round_program.draft_attention(q, keys, values, own_k, own_v, count, newest)`` then
    ``.transpose(0, 2, 1, 3).reshape(1, R, -1)``, with ``own_v_rows`` the value heads in the
    projection's layout ``(1, R, Hk, D)`` (``own_v`` is their transpose): ``(1, R, Hq * D)``."""
    from . import round_program as rp

    _, hq, rows, dim = q.shape
    hk, slots = keys.shape[1], keys.shape[2]

    def stock():
        own_v = own_v_rows.transpose(0, 2, 1, 3)
        out = rp.draft_attention(q, keys, values, own_k, own_v, count_rows, newest, ring=ring)
        return out.transpose(0, 2, 1, 3).reshape(1, rows, hq * dim)

    kernels = _attend_kernels()
    group = hq // hk
    if (kernels is None or count_rows.dtype != mx.uint32 or newest.dtype != mx.uint32 or count_rows.size != 1
            or newest.size != 1 or own_v_rows.shape != (1, rows, hk, dim) or own_k.shape != (1, hk, rows, dim)
            or group * rows % 8 or dim % 32 or q.dtype not in _TYPES or keys.dtype != q.dtype
            or values.dtype != q.dtype or own_k.dtype != q.dtype or own_v_rows.dtype != q.dtype):
        count("fallback.draft_glue.attention_shape")
        return stock()
    chunk = rp._chunk(slots, ring)
    chunks = -(-slots // chunk) + 1
    template = [("T", q.dtype), ("G", group), ("R", rows), ("D", dim), ("S", slots), ("CHUNK", chunk),
                ("TK", rp._TILE), ("QPS", group * rows // 8), ("RING", bool(ring)), ("HK", hk)]
    partials, merge = kernels

    def ours():
        pm, pl, pacc = partials(
            inputs=[q, keys, values, own_k, own_v_rows, count_rows.reshape(()), newest.reshape(())],
            template=template,
            grid=(256 * chunks, hk, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(hk * chunks * group * rows,), (hk * chunks * group * rows,),
                           (hk * chunks * group * rows * dim,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        (out,) = merge(
            inputs=[pm, pl, pacc],
            template=template,
            grid=(group * rows * dim, hk, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(1, rows, hq * dim)],
            output_dtypes=[q.dtype],
        )
        return out

    return _decided(("attention", hq, hk, rows, dim, slots, bool(ring), str(q.dtype)), ours, stock)


# ------------------------------------------------------------------------------ block, propose
def block(blk, hidden, residual, view, index: int):
    """``round_program._block`` with the glue above: the same values in fewer launches."""
    from ..model import linear

    dim = blk.cfg["head_dim"]
    eps = blk.cfg["rms_norm_eps"]
    if residual is None:
        x = blk.normalize(hidden, "input_layernorm").astype(hidden.dtype)
        carried = hidden
    else:
        x, carried = add_norm(hidden, residual, blk.w["input_layernorm.weight"], eps)

    def rows_of(name):  # the heads in the projection's own layout (1, R, H, D)
        return linear(x, blk.w["self_attn." + name + "_proj.weight"]).reshape(*x.shape[:2], -1, dim)

    # Normed before the transpose (the same rows; MLX's rotation takes the strided view).
    q = blk.rotate(blk.normalize(rows_of("q"), "self_attn.q_norm").transpose(0, 2, 1, 3), view.offset)
    k = blk.rotate(blk.normalize(rows_of("k"), "self_attn.k_norm").transpose(0, 2, 1, 3), view.offset)
    window = view.windows[index]
    out = attention(q, view.keys[index], view.values[index], k, rows_of("v"),
                    view.count if window else view.length, view.newest, ring=bool(window))
    attn = linear(out, blk.w["self_attn.o_proj.weight"])
    y, carried = add_norm(carried, attn, blk.w["post_attention_layernorm.weight"], eps)
    gate = linear(y, blk.w["mlp.gate_proj.weight"])
    up = linear(y, blk.w["mlp.up_proj.weight"])
    return linear(gate_act(gate, up), blk.w["mlp.down_proj.weight"]), carried


def propose(drafter, target, anchor, view, count_: int, *, position, seed, temperature=1.0, sample=True,
            cap=None, params=None):
    """``round_program.propose`` with the drafter's glue in launches of ours (see the module docstring):
    the same ``(ids, DraftLaw)``, bit for bit."""
    from ..model import linear
    from ..precision import AffineWeight
    from . import round as fast_round
    from .matvec import matvec
    from .step import embed as fast_embed

    if not 1 <= count_ < drafter.width:
        raise ValueError("DSpark proposal count exceeds its trained block")
    greedy = (not sample) or (not temperature or temperature <= 0)
    if cap is None:
        cap = target.config.get("final_logit_softcapping") or 0.0
    anchor_ids = anchor.astype(mx.int32).reshape(1, 1)
    embed = fast_embed(target, anchor_ids)
    dtype = embed.dtype
    masks = mx.broadcast_to(drafter.mask, (1, drafter.width - 1, embed.shape[-1]))
    h, residual = mx.concatenate([embed, masks], axis=1), None
    for i, blk in enumerate(drafter.blocks):
        h, residual = block(blk, h, residual, view, i)
    h = add_norm(h, residual, drafter.output_norm, drafter.cfg["rms_norm_eps"])[0][:, 1 : count_ + 1]

    base = None
    if isinstance(target.unembed, AffineWeight):
        from .qproduct import STREAM_TYPES, affine_product

        try:
            if h.dtype in STREAM_TYPES:
                base = affine_product(h, target.unembed, widen=True)[..., : drafter.vocab]
            else:
                base = affine_product(h.astype(mx.float32), target.unembed)[..., : drafter.vocab]
        except RuntimeError:
            count("fallback.draft_head.affine_launch_refused")
    if base is None:
        base = linear(h, target.unembed)[..., : drafter.vocab]
    rank = drafter.cfg["markov_rank"]
    weights = drafter.markov["rnn_proj.weight"].astype(mx.float32)
    inputs = mx.concatenate([h, mx.broadcast_to(view.last, h.shape)], axis=-1)
    conditioned = linear(inputs.astype(mx.float32), weights[:, 2 * rank :]) + drafter.markov[
        "rnn_proj.bias"
    ].astype(mx.float32)
    folded = drafter.fold(target.unembed)

    start = position.astype(mx.uint32).reshape(())
    state = fast_round._const([[0.0] * rank], dtype)
    one = fast_round._const(1.0, dtype)
    prev = anchor_ids.reshape(1)
    table = drafter.markov["markov_w1.weight"]
    picks, laws, stats, confidence = [], [], [], []
    for i in range(count_):
        step_in = markov_input(state, table, prev)
        update = matvec(step_in, weights, cols=2 * rank)
        state, signal = markov_update(update, conditioned[:, i], state, one)
        bias = matvec(signal, folded, cols=folded.shape[1] - 1, bias_col=folded.shape[1] - 1)
        scores = base[:, i] + bias
        if greedy:
            law, st = fast_round.law_rows(scores, temperature=None, cap=0.0, write=True)
        else:
            law, st = fast_round.law_rows(scores, temperature=temperature, cap=cap, params=params)
        token, lse = fast_round.sample_rows(law, st, positions=start.reshape(1), seed=seed,
                                            purpose=fast_round.PURPOSE_DRAFT, greedy=greedy, offset=i)
        prev = token
        picks.append(token)
        laws.append(law)
        stats.append(st)
        if not greedy:
            confidence.append((mx.take_along_axis(law, token[:, None], axis=1)[:, 0] - lse[:, 0]) - lse[:, 1])
    ids = mx.concatenate(picks)
    law = fast_round.DraftLaw(mx.concatenate(laws), mx.concatenate(stats), point_mass=greedy,
                              log_q=mx.concatenate(confidence) if confidence else None)
    return ids, law


__all__ = ["add_norm", "attend_texts", "attention", "block", "enabled", "gate_act", "markov_input", "markov_update", "propose"]
