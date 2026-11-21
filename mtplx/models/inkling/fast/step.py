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
from .routes import count
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


# --------------------------------------------------------------------------- exact under compile
# mx.compile fuses chains of elementwise operations into one runtime-compiled kernel. Two of
# the reference's chains change bits when fused (receipts/fusion-bits-1.json): the sigmoid,
# whose prebuilt kernel uses the precise exponential while a runtime-compiled one gets the
# fast one, and the scaled per-head clip. These spellings run the same operations in the same
# order, so eagerly they give the reference's bits launch for launch, and under compile they
# keep them: the exponential is called precisely, and the clip's scaled value and its square
# are materialised before the reduction, exactly where the reference materialises them.
def _materialize(a):
    """A fusion boundary that costs no launch: a flat view and back."""
    return a.reshape(-1).reshape(a.shape)


def exact_sigmoid(x):
    """``mx.sigmoid`` with the prebuilt kernel's arithmetic: 1/(1+exp(|x|)), mirrored for x >= 0."""
    y = 1 / (1 + mx.exp(mx.abs(x)))
    return mx.where(x < 0, y, 1 - y)


def exact_clip_heads(x, limit, scale=None):
    """``model.clip_heads`` with its materialisation points kept under compile."""
    value = x.astype(mx.float32)
    if scale is not None:
        value = _materialize(value * scale.astype(mx.float32))
    norm = mx.sqrt(mx.sum(_materialize(mx.square(value)), axis=-1, keepdims=True))
    factor = mx.minimum(1.0, limit / (norm + 1e-6))
    return (value * factor).astype(x.dtype)


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
    q = exact_clip_heads(
        ref.linear(q_in, layer.q).reshape(*shape, width), cfg["qk_l2_clip_threshold"], layer.q_scale
    )
    rel = exact_clip_heads(
        ref.linear(q_in, layer.r).reshape(*shape, -1), cfg["rel_l2_clip_threshold"], layer.r_scale
    )
    k_taps = ref.linear(k_in, layer.k)
    v_taps = ref.linear(v_in, layer.v)
    k = convolution(layer.convs[0], k_taps, state.tail(visit, 0), panel, k_in)
    v = convolution(layer.convs[1], v_taps, state.tail(visit, 1), panel, v_in)
    k = exact_clip_heads(k.reshape(*k.shape[:2], -1, width), cfg["qk_l2_clip_threshold"])
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


def attention_core(layer, q, profiles, sink, k_rows, v_rows, state, visit, panel):
    """Stage B (reference half): ``(1, W, 3072)`` attention result in the activation dtype.
    ``profiles`` is ``rel @ rel_basis``."""
    keys, values = state.history(visit, k_rows, v_rows)
    window = layer.cfg["sliding_window_size"] if layer.sliding else 0
    h = attend(
        q.transpose(0, 2, 1, 3), keys, values, profiles, sink, window,
        **({"verify": True} if panel.tape is not None else {}),
    )
    return h.transpose(0, 2, 1, 3).reshape(h.shape[0], panel.rows, -1)


def attention_output(layer, h, gate_logits, x, state, visit, panel):
    """Stage C (reference half): ``(partial float32, attention conv inputs float32)``."""
    gate = exact_sigmoid(gate_logits.astype(mx.float32))
    attn = ref.linear((h.astype(mx.float32) * gate).astype(h.dtype), layer.out)
    partial = convolution(layer.convs[2], attn, state.tail(visit, 2), panel).astype(mx.float32)
    if not layer.boundary:
        partial = partial + x.astype(mx.float32)
    return partial, attn.astype(mx.float32)


def relative_profiles(layer, rel):
    return rel @ layer.rel_basis.astype(rel.dtype)


def attention(layer, a, x, state, visit, panel):
    """Stage BC (reference). ``(partial (1, W, 2304) float32, attention conv inputs float32)``.
    ``partial`` is the convolved attention result plus ``x``, or alone on a boundary layer."""
    h = attention_core(layer, a.q, relative_profiles(layer, a.rel), a.sink, a.k_rows, a.v_rows,
                       state, visit, panel)
    return attention_output(layer, h, a.gate_logits, x, state, visit, panel)


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


# --------------------------------------------------------------------------- lane kernels
# Drop-ins with the reference stages' arguments and results (SIGNATURES.md section 3). Each is
# proven against REFERENCE on the goldens and on the model before it is named in a candidate.
def fused_attention(layer, a, x, state, visit, panel):
    """Stage BC on lane L2's kernels: attention over the committed history read in place plus
    the panel's rows, gate, output projection, attention convolution, residual (class A)."""
    from .attend import attend as panel_attend
    from .attn_output import attention_output as panel_output

    if layer.sliding:
        keys, values, newest, valid = state.window_view(visit)
    else:
        keys, values, valid = state.global_view(visit)
        newest = valid
    hg = panel_attend(a.q, a.rel, a.sink, a.gate_logits, a.k_rows, a.v_rows, keys, values, valid,
                      newest, panel.pos, layer.rel_basis, window=layer.sliding)
    return panel_output(hg, layer.out, layer.convs[2].weight, state.tail(visit, 2), x,
                        boundary=layer.boundary)


def fused_attention_input(layer, x, closed, message, state, visit, panel):
    """Stage A on lane L1's kernels (two launches a visit, three on loop passes 2 and 3)."""
    from .attn_input import stage_attention_input

    return stage_attention_input(layer, x, closed, message, state, visit, panel)


def fused_expert_input(layer, partial, closed, message, panel):
    """Stage D on lane L1's read kernel (one launch, two on loop passes 2 and 3)."""
    from .read import stage_expert_input

    return stage_expert_input(layer, partial, closed, message, panel)


def fused_experts(layer, source, partial, dtype, state, visit, panel):
    """Stage E on lane L3's expert block (class A); the reference serves what it declines."""
    from .experts import expert_visit, unserved

    reason = unserved(layer.experts, panel.rows, dtype)
    if reason is not None:
        count(f"fallback.experts.{reason}")
        return experts(layer, source, partial, dtype, state, visit, panel)
    return expert_visit(layer, source, partial, state.tail(visit, 3))


LANES = {
    "a": ("attention_input", fused_attention_input),
    "bc": ("attention", fused_attention),
    "d": ("expert_input", fused_expert_input),
    "e": ("experts", fused_experts),
}


def stages_for(spec):
    """Stages with the named lanes' kernels, from a comma list of a, bc, d, e ('' = reference)."""
    chosen = {}
    for name in filter(None, (part.strip().lower() for part in spec.split(","))):
        field, fn = LANES[name]
        chosen[field] = fn
    return Stages(**chosen)


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


__all__ = ["AttentionInput", "DecodeState", "LANES", "Message", "Panel", "REFERENCE", "Stages",
           "decode_hidden", "decode_panel", "stages_for", "visit_step"]
