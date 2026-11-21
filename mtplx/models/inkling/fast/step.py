"""One decode step of Inkling 2 Mini for a panel of 1 to 32 rows, stage by stage.

``decode_panel(model, ids, state)`` returns the logits of every row, the drafter's
captures and one ``VisitDelta`` per layer visit (the panel's own key/value rows and
convolution inputs). It reads the committed state and writes nothing; after the accept
decision ``state.commit(deltas, kept)`` writes the first ``kept`` rows.

A layer visit is four stages with the edges of ``SIGNATURES.md`` section 3:

    A   attention input   depth read, cross-pass controllers, norms, projections, clips,
                          key and value convolutions, value norm, log position scaling
    BC  attention         relative-bias sink attention over committed history plus the
                          panel, gate, output projection, attention convolution, residual
    D   expert input      depth read with the MLP query, MLP controller
    E   experts           norm, router, top 8, experts, expert convolution, residual

Everything outside them is the frame here: embedding, the loop's entry state and
message stack, closed blocks (a boundary appends the incoming ``x``), captures, the
final read and norm, the head.

``REFERENCE`` runs every stage with the reference model's own operations
(``model.py``, ``attention.py``, ``short_conv.py``, ``expert_fusion.py``), so its output is
bit-identical to the reference forward: plain decode for one row, the reference verify
path for 2 to 8 rows. It is the oracle every fused stage is proven against. Its stage BC
reads the history back in chronological order with host-side shapes; every other stage is
a function of arrays whose shapes depend only on the row count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx

from .. import model as ref
from .. import short_conv
from ..attention import attend
from ..projection import stock_messages
from .state import DecodeState, VisitDelta


@dataclass(frozen=True)
class Panel:
    """What every stage of one step shares. ``tape`` is the reference verify flag
    (``None`` for one row, as plain decode runs; ``True`` for 2 or more rows)."""

    rows: int
    pos: mx.array  # uint32 scalar: committed tokens
    tape: Optional[bool]


class Message:
    """The cross-pass message of one visit on loop passes 2 and 3."""

    def __init__(self, stack, queries, verify):
        self.stack, self.queries, self.verify = stack, queries, verify
        self._read = None

    @property
    def read(self):
        """``(1, W, 4, hidden)`` float32: attention, key, value and MLP controller inputs."""
        if self._read is None:
            fn = ref.cross_pass_message if self.verify else stock_messages
            self._read = fn(self.queries, self.stack, self.stack.shape[-1] ** -0.5)
        return self._read


@dataclass(frozen=True)
class AttentionInput:
    q: mx.array  # (1, W, 24, 128) after the scaled clip and the global position scale
    rel: mx.array  # (1, W, 24, 16) after its scaled clip and the global position scale
    sink: mx.array  # (1, W, 24)
    gate_logits: mx.array  # (1, W, 3072), before the sigmoid
    k_rows: mx.array  # (1, 8, W, 128) as the reference appends them
    v_rows: mx.array
    k_taps: mx.array  # (1, W, 1024) float32: key convolution inputs
    v_taps: mx.array


# --------------------------------------------------------------------------- reference stages
def convolution(conv, x, tail, panel, drive=None):
    """The reference short convolution on the committed tail (same kernel, proof, bits)."""
    change = None if conv.dynamic is None else ref.linear(drive, conv.dynamic)
    fused = short_conv.convolve(x, tail, conv.weight, change, panel.tape is not None)
    if fused is not None:
        return fused[0]
    return short_conv.reference(x, tail, conv.weight, change, False)[0]


def attention_input(layer, x, closed, message, state, visit, panel):
    """Stage A (reference)."""
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
                layer.controllers[i](h, message.read[..., i, :], eps) for i, h in enumerate(sources)
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
    k_taps = ref.linear(k_in, layer.k)
    v_taps = ref.linear(v_in, layer.v)
    k = convolution(layer.convs[0], k_taps, state.tail(visit, 0), panel, k_in)
    v = convolution(layer.convs[1], v_taps, state.tail(visit, 1), panel, v_in)
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
    return AttentionInput(
        q, rel, ref.linear(q_in, layer.sink), ref.linear(q_in, layer.gate),
        k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
        k_taps.astype(mx.float32), v_taps.astype(mx.float32),
    )


def attention(layer, a, x, state, visit, panel):
    """Stage BC (reference). ``(partial (1, W, 2304) float32, attention conv inputs float32)``.
    ``partial`` is the convolved attention result plus ``x``, or alone on a boundary layer."""
    keys, values = state.history(visit, a.k_rows, a.v_rows)
    window = layer.cfg["sliding_window_size"] if layer.sliding else 0
    h = attend(
        a.q.transpose(0, 2, 1, 3), keys, values, a.rel @ layer.rel_basis.astype(a.rel.dtype),
        a.sink, window, **({"verify": True} if panel.tape is not None else {}),
    )
    h = h.transpose(0, 2, 1, 3).reshape(h.shape[0], panel.rows, -1)
    gate = mx.sigmoid(a.gate_logits.astype(mx.float32))
    attn = ref.linear((h.astype(mx.float32) * gate).astype(h.dtype), layer.out)
    partial = convolution(layer.convs[2], attn, state.tail(visit, 2), panel).astype(mx.float32)
    if not layer.boundary:
        partial = partial + x.astype(mx.float32)
    return partial, attn.astype(mx.float32)


def expert_input(layer, partial, closed, message, panel):
    """Stage D (reference). The float32 MLP source, before the MLP norm."""
    source = ref.depth_reader(panel.tape, panel.rows)(
        closed, partial, layer.mlp_query, layer.eps, layer.cfg["attn_res_normalize_values"]
    )[..., 0, :]
    if message is not None:
        source = layer.controllers[3](source, message.read[..., 3, :], layer.eps)
    return source


def experts(layer, source, partial, dtype, state, visit, panel):
    """Stage E (reference). ``(visit output (1, W, 2304) in dtype, MLP conv inputs float32)``."""
    routed = layer.experts(ref.norm(source, layer.ffn_norm, layer.eps).astype(dtype))
    out = convolution(layer.convs[3], routed, state.tail(visit, 3), panel).astype(mx.float32)
    return (partial + out).astype(dtype), routed.astype(mx.float32)


@dataclass(frozen=True)
class Stages:
    """The four stage functions one step runs. Lanes replace entries as kernels land."""

    attention_input: Callable = attention_input
    attention: Callable = attention
    expert_input: Callable = expert_input
    experts: Callable = experts


REFERENCE = Stages()


# --------------------------------------------------------------------------- one visit, one step
def visit_step(layer, x, closed, message, state, visit, panel, stages=REFERENCE, record=None):
    """One layer visit. Returns the visit's output and its delta; appends to ``closed``."""
    a = stages.attention_input(layer, x, closed, message, state, visit, panel)
    partial, attn_taps = stages.attention(layer, a, x, state, visit, panel)
    if record is not None:
        record(visit, "A", {"x": x, "closed": list(closed), "message": message}, a)
        record(visit, "BC", {"a": a, "x": x}, {"partial": partial, "attn_taps": attn_taps})
    if layer.boundary:
        closed.append(x)
    source = stages.expert_input(layer, partial, closed, message, panel)
    out, mlp_taps = stages.experts(layer, source, partial, x.dtype, state, visit, panel)
    if record is not None:
        record(visit, "D", {"partial": partial, "closed": list(closed), "message": message},
               {"source": source})
        record(visit, "E", {"source": source, "partial": partial}, {"out": out, "mlp_taps": mlp_taps})
    return out, VisitDelta(a.k_rows, a.v_rows, (a.k_taps, a.v_taps, attn_taps, mlp_taps))


def decode_hidden(model, ids, state, *, embeddings=None, capture_ids=(), stages=REFERENCE,
                  record=None, tape=None):
    """The final hidden rows ``(1, W, 2304)``, the captures and the per-visit deltas.

    ``tape`` overrides the reference verify flag (default: verify for 2 or more rows).
    """
    rows = ids.shape[-1] if embeddings is None else embeddings.shape[1]
    verify = (rows > 1) if tape is None else bool(tape)
    panel = Panel(rows, state.pos, True if verify else None)
    closed, captured, deltas = [], {}, []
    x = model.embed_tokens(ids.reshape(1, rows)) if embeddings is None else embeddings
    stack = entry = entry_blocks = None
    for visit, index in enumerate(model.visit_order):
        if index == model.start:
            if visit == model.start:
                entry, entry_blocks = x, list(closed)
                stack = None
            else:
                prior = [*closed[len(entry_blocks) + 1 :], x]
                stack = mx.fast.rms_norm(mx.stack(prior, axis=-2).astype(mx.float32), None, model.eps)
                x, closed = entry, list(entry_blocks)
        if visit in capture_ids:
            captured[visit] = x
        message = None
        if model.start <= index < model.end and stack is not None:
            message = Message(stack, model.message_queries[index - model.start], verify)
        x, delta = visit_step(model.layers[index], x, closed, message, state, visit, panel, stages, record)
        deltas.append(delta)
    last = ref.depth_reader(panel.tape, rows)(
        closed, x, model.final_query, model.eps, model.config["attn_res_normalize_values"]
    )[..., 0, :]
    hidden = ref.norm(last, model.final_norm, model.eps).astype(x.dtype)
    if len(model.visit_order) in capture_ids:
        captured[len(model.visit_order)] = hidden
    return hidden, [captured[i] for i in capture_ids], deltas


def decode_panel(model, ids, state, *, embeddings=None, capture_ids=None, stages=REFERENCE,
                 record=None, tape=None):
    """``(logits (1, W, vocab) float32, captures, deltas)`` for a panel of W rows.

    Reads ``state`` and writes nothing; ``state.commit(deltas, n)`` accepts n rows.
    """
    capture_ids = model.capture_ids if capture_ids is None else capture_ids
    hidden, captures, deltas = decode_hidden(
        model, ids, state, embeddings=embeddings, capture_ids=capture_ids, stages=stages,
        record=record, tape=tape,
    )
    return model.head(hidden), captures, deltas


__all__ = ["AttentionInput", "DecodeState", "Message", "Panel", "REFERENCE", "Stages",
           "decode_hidden", "decode_panel", "visit_step"]
