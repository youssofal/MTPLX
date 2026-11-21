"""One decode step of Inkling 2 Mini for a panel of 1 to 32 rows, stage by stage.

``decode_panel(model, ids, state)`` returns the logits of every row and the drafter's
captures, and writes every row's key/value and convolution inputs into ``state`` at the
slots of positions ``pos .. pos + rows - 1`` (ahead of the committed position). Accepting
``n`` rows afterwards is ``state.commit(n)``; a rejected row leaves nothing to undo.

A layer visit is five stages (BUILD-ORDER.md):

    A  attention input   depth read, cross-pass controllers, norms, Q/K/V/relative/sink/
                         gate projections, clips, key and value convolutions, value norm,
                         log position scaling, key/value write
    B  attention core    relative-bias sink attention over the window ring or global bank
    C  attention output  gate, output projection, attention convolution, residual or block
    D  expert input      depth read with the MLP query, MLP controller
    E  experts           norm, router, top 8, experts, expert convolution, residual

``REFERENCE`` runs every stage with the operations of the reference model (``model.py``,
``attention.py``, ``short_conv.py``, ``expert_fusion.py``) on the fast state layout, so its
output is bit-identical to the reference forward: plain decode for one row, the reference
verify path for 2 to 8 rows. It is the oracle each fused stage is proven against. Stage B
of the reference reads the keys back in chronological order (host-side shapes); every
other stage is a function of arrays whose shapes depend only on the row count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx

from .. import model as ref
from .. import short_conv
from ..attention import attend
from ..projection import stock_messages


@dataclass(frozen=True)
class Panel:
    """What every stage of one step shares. ``tape`` is the reference verify flag
    (``None`` for one row, as plain decode runs; ``True`` for 2 or more rows)."""

    rows: int
    pos: mx.array
    length: int
    tape: Optional[bool]


# --------------------------------------------------------------------------- helpers
def convolution(conv, x, state, visit, slot, panel, drive=None):
    """The reference short convolution reading its three previous inputs from the tap ring,
    then writing this panel's inputs at ``pos ..``. Same kernel, same proof, same bits."""
    prev = state.tap_history(visit, slot, panel.pos)
    change = None if conv.dynamic is None else ref.linear(drive, conv.dynamic)
    fused = short_conv.convolve(x, prev, conv.weight, change, panel.tape is not None)
    if fused is not None:
        out = fused[0]
    else:
        out = short_conv.reference(x, prev, conv.weight, change, False)[0]
    state.write_taps(visit, slot, x, panel.pos)
    return out


# --------------------------------------------------------------------------- reference stages
def attention_input(layer, x, closed, message, state, visit, panel):
    """Stage A. Returns ``(q (1, 24, W, 128), profiles (1, W, 24, 1024), sink (1, W, 24),
    gate (1, W, 3072) float32)`` and writes the key/value rows and both tap rows."""
    cfg, eps = layer.cfg, layer.eps
    if layer.attn_query is not None:
        read = ref.depth_reader(panel.tape, panel.rows)(
            closed, x, layer.attn_query, eps, cfg["attn_res_normalize_values"]
        )
    else:
        read = x[..., None, :]
    if layer.sliding and message is None:
        q_in = k_in = v_in = ref.norm(read[..., 0, :], layer.qkv_norm[0], eps).astype(x.dtype)
    else:
        sources = [read[..., 0 if layer.sliding else i, :] for i in range(3)]
        if message is not None:
            sources = [
                layer.controllers[i](h, message[..., i, :], eps) for i, h in enumerate(sources)
            ]
        q_in, k_in, v_in = [
            ref.norm(h, layer.qkv_norm[0 if layer.sliding else i], eps).astype(x.dtype)
            for i, h in enumerate(sources)
        ]
    shape = (*q_in.shape[:2], layer.q_scale.shape[0])
    width = layer.q_scale.shape[-1]
    q = ref.clip_heads(
        ref.linear(q_in, layer.q).reshape(*shape, width), cfg["qk_l2_clip_threshold"], layer.q_scale
    )
    rel = ref.clip_heads(
        ref.linear(q_in, layer.r).reshape(*shape, -1), cfg["rel_l2_clip_threshold"], layer.r_scale
    )
    k = convolution(layer.convs[0], ref.linear(k_in, layer.k), state, visit, 0, panel, k_in)
    v = convolution(layer.convs[1], ref.linear(v_in, layer.v), state, visit, 1, panel, v_in)
    k = ref.clip_heads(k.reshape(*k.shape[:2], -1, width), cfg["qk_l2_clip_threshold"])
    v = v.reshape(*v.shape[:2], -1, width)
    if cfg["use_value_norm_no_gamma"]:
        v = ref.norm(v, None, eps).astype(v_in.dtype)
    if not layer.sliding:
        position = mx.arange(panel.rows, dtype=mx.int32) + panel.pos.astype(mx.int32) + 1
        tau = 1 + cfg["log_scaling_alpha"] * mx.log(
            mx.maximum(position / cfg["log_scaling_n_floor"], 1)
        )
        q = (q.astype(mx.float32) * tau[:, None, None]).astype(q.dtype)
        rel = (rel.astype(mx.float32) * tau[:, None, None]).astype(rel.dtype)
    state.write_kv(visit, k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), panel.pos)
    profiles = rel @ layer.rel_basis.astype(rel.dtype)
    sink = ref.linear(q_in, layer.sink)
    gate = mx.sigmoid(ref.linear(q_in, layer.gate).astype(mx.float32))
    return q.transpose(0, 2, 1, 3), profiles, sink, gate


def attention_core(layer, q, profiles, sink, state, visit, panel):
    """Stage B. ``(1, W, 3072)`` in the activation dtype."""
    keys, values = state.history_keys(visit, panel.rows)
    window = layer.cfg["sliding_window_size"] if layer.sliding else 0
    h = attend(q, keys, values, profiles, sink, window,
               **({"verify": True} if panel.tape is not None else {}))
    return h.transpose(0, 2, 1, 3).reshape(q.shape[0], panel.rows, -1)


def attention_output(layer, h, gate, x, closed, state, visit, panel):
    """Stage C. The float32 residual; appends ``x`` to ``closed`` on a block boundary."""
    attn = ref.linear((h.astype(mx.float32) * gate).astype(h.dtype), layer.out)
    residual = convolution(layer.convs[2], attn, state, visit, 2, panel).astype(mx.float32)
    if layer.boundary:
        closed.append(x)
    else:
        residual = residual + x.astype(mx.float32)
    return residual


def expert_input(layer, residual, closed, message, panel):
    """Stage D. The float32 MLP source ``(1, W, 2304)`` after the depth read and controller."""
    ffn = ref.depth_reader(panel.tape, panel.rows)(
        closed, residual, layer.mlp_query, layer.eps, layer.cfg["attn_res_normalize_values"]
    )[..., 0, :]
    if message is not None:
        ffn = layer.controllers[3](ffn, message[..., 3, :], layer.eps)
    return ffn


def experts(layer, ffn, residual, dtype, state, visit, panel):
    """Stage E. The visit's output ``(1, W, 2304)`` in the activation dtype."""
    routed = layer.experts(ref.norm(ffn, layer.ffn_norm, layer.eps).astype(dtype))
    out = convolution(layer.convs[3], routed, state, visit, 3, panel).astype(mx.float32)
    return (residual + out).astype(dtype)


@dataclass(frozen=True)
class Stages:
    """The five stage functions one step runs. Lanes replace entries as kernels land."""

    attention_input: Callable = attention_input
    attention_core: Callable = attention_core
    attention_output: Callable = attention_output
    expert_input: Callable = expert_input
    experts: Callable = experts


REFERENCE = Stages()


# --------------------------------------------------------------------------- one visit, one step
def visit_step(layer, x, closed, message, state, visit, panel, stages=REFERENCE, record=None):
    q, profiles, sink, gate = stages.attention_input(layer, x, closed, message, state, visit, panel)
    if record is not None:
        record(visit, "A", {"x": x, "closed": list(closed), "message": message},
               {"q": q, "profiles": profiles, "sink": sink, "gate": gate})
    h = stages.attention_core(layer, q, profiles, sink, state, visit, panel)
    if record is not None:
        record(visit, "B", {"q": q, "profiles": profiles, "sink": sink}, {"h": h})
    residual = stages.attention_output(layer, h, gate, x, closed, state, visit, panel)
    if record is not None:
        record(visit, "C", {"h": h, "gate": gate, "x": x}, {"residual": residual})
    ffn = stages.expert_input(layer, residual, closed, message, panel)
    if record is not None:
        record(visit, "D", {"residual": residual, "closed": list(closed), "message": message},
               {"ffn": ffn})
    out = stages.experts(layer, ffn, residual, x.dtype, state, visit, panel)
    if record is not None:
        record(visit, "E", {"ffn": ffn, "residual": residual}, {"out": out})
    return out


def decode_hidden(model, ids, state, *, embeddings=None, capture_ids=(), stages=REFERENCE,
                  record=None, tape=None):
    """The final hidden rows ``(1, W, 2304)`` and the captures, writing the panel's state.

    ``tape`` overrides the reference verify flag (default: verify for 2 or more rows).
    """
    rows = ids.shape[-1] if embeddings is None else embeddings.shape[1]
    state.grow(rows)
    panel = Panel(rows, state.pos, state.length, (True if rows > 1 else None) if tape is None else tape)
    closed, captured = [], {}
    x = model.embed_tokens(ids.reshape(1, rows)) if embeddings is None else embeddings
    messages = entry = entry_blocks = None
    for visit, index in enumerate(model.visit_order):
        if index == model.start:
            if visit == model.start:
                entry, entry_blocks = x, list(closed)
                messages = None
            else:
                prior = [*closed[len(entry_blocks) + 1 :], x]
                stacked = mx.stack(prior, axis=-2).astype(mx.float32)
                messages = mx.fast.rms_norm(stacked, None, model.eps)
                x, closed = entry, list(entry_blocks)
        if visit in capture_ids:
            captured[visit] = x
        message = None
        if model.start <= index < model.end and messages is not None:
            read = ref.cross_pass_message if panel.tape is not None else stock_messages
            message = read(model.message_queries[index - model.start], messages, x.shape[-1] ** -0.5)
        x = visit_step(model.layers[index], x, closed, message, state, visit, panel, stages, record)
    last = ref.depth_reader(panel.tape, rows)(
        closed, x, model.final_query, model.eps, model.config["attn_res_normalize_values"]
    )[..., 0, :]
    hidden = ref.norm(last, model.final_norm, model.eps).astype(x.dtype)
    if len(model.visit_order) in capture_ids:
        captured[len(model.visit_order)] = hidden
    return hidden, [captured[i] for i in capture_ids]


def decode_panel(model, ids, state, *, embeddings=None, capture_ids=None, stages=REFERENCE,
                 record=None, tape=None):
    """Logits ``(1, W, vocab)`` float32 and the drafter captures for a panel of W rows.

    The panel's key/value and convolution inputs are written ahead of ``state.pos``;
    ``state.commit(n)`` accepts the first n rows.
    """
    capture_ids = model.capture_ids if capture_ids is None else capture_ids
    hidden, captures = decode_hidden(model, ids, state, embeddings=embeddings,
                                     capture_ids=capture_ids, stages=stages, record=record, tape=tape)
    return model.head(hidden), captures
