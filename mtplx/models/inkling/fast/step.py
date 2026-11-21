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
from ..precision import AffineWeight
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


def chronological_core(layer, q, profiles, sink, k_rows, v_rows, state, visit, panel):
    """Stage B by the one-row reference, row by row: each row is the reference's one-row
    attention call on exactly the keys the one-row step reads at that row's position (the
    window's last 1,023 rows and the row itself, or every earlier key), so every row equals
    the one-row step at any width. Host-side shapes that follow the live history: the step
    takes it only where the fused attention's qualification declines, and nothing records it."""
    keys, values = state.history(visit, k_rows, v_rows)
    window = layer.cfg["sliding_window_size"] if layer.sliding else 0
    rows, total = panel.rows, keys.shape[2]
    qt = q.transpose(0, 2, 1, 3)
    pieces = []
    for row in range(rows):
        stop = total - rows + row + 1
        begin = max(0, stop - window) if window else 0
        pieces.append(attend(qt[:, :, row:row + 1], keys[:, :, begin:stop], values[:, :, begin:stop],
                             profiles[:, row:row + 1], sink[:, row:row + 1], window))
    h = pieces[0] if rows == 1 else mx.concatenate(pieces, axis=2)
    return h.transpose(0, 2, 1, 3).reshape(h.shape[0], rows, -1)


def attention_output(layer, h, gate_logits, x, state, visit, panel):
    """Stage C (reference half): ``(partial float32, attention conv inputs float32)``."""
    gate = exact_sigmoid(gate_logits.astype(mx.float32))
    return output_projection(layer, (h.astype(mx.float32) * gate).astype(h.dtype), x, state, visit, panel)


def output_projection(layer, hg, x, state, visit, panel):
    """Stage C from the gated attention result: the output projection by the pack's own
    product (16-bit) or, for stored codes, ``packed_linear`` (MLX sends a 2 to 24 row affine
    product to a wide kernel whose rows differ from the one-row product on M3 to M5, and the
    reference at every width is the one-row step), the attention convolution and the residual."""
    attn = packed_linear(hg, layer.out) if isinstance(layer.out, AffineWeight) else ref.linear(hg, layer.out)
    partial = convolution(layer.convs[2], attn, state.tail(visit, 2), panel).astype(mx.float32)
    if not layer.boundary:
        partial = partial + x.astype(mx.float32)
    return partial, attn.astype(mx.float32)


def per_row_linear(x, weight):
    """``x @ weight`` with each row projected alone by the stock product, as a one-row step
    projects it (for stored codes, whose multi-row product is not row-stable on M3 and up)."""
    if x.shape[1] == 1:
        return ref.linear(x, weight)
    return mx.concatenate([ref.linear(x[:, row : row + 1], weight) for row in range(x.shape[1])], axis=1)


#: (what, rows, shapes, dtype) -> True when the pack's multi-row product equals its one-row
#: product on this Mac (M1 and M2 keep MLX's one-row kernel for small panels; M3 to M5 do not).
_ROWS_PROVEN: dict = {}


def proven_rows(key, batched, per_row):
    """A panel product whose rows equal the one-row step's: the batched form where its first use
    on this Mac and shape proved it bit for bit against the rows computed one at a time, else the
    rows one at a time. The proof reads both once (first call of a shape: before the engine
    records it); the kernel MLX picks depends on shapes and the GPU, not on values."""
    verdict = _ROWS_PROVEN.get(key)
    if verdict is None:
        whole, rows = batched(), per_row()
        mx.eval(whole, rows)
        verdict = _ROWS_PROVEN[key] = whole.shape == rows.shape and bool(
            mx.array_equal(whole.view(mx.uint16 if whole.dtype.size == 2 else mx.uint32),
                           rows.view(mx.uint16 if rows.dtype.size == 2 else mx.uint32)).item())
        count(f"rows_proof.{key[0]}.{'batched' if verdict else 'one_at_a_time'}")
        return whole if verdict else rows
    return batched() if verdict else per_row()


def launch_serves(weight):
    """Whether lane L1's row-stable affine launch (``qproduct.affine_product``) takes this stored
    form (bits, group, scales); its first use per shape on a Mac still proves the rows."""
    from .qproduct import qmv_constants

    try:
        qmv_constants(weight)
    except RuntimeError:
        return False
    return True


def packed_linear(x, weight):
    """``x @ weight`` for stored codes, every row equal to the one-row product: lane L1's
    row-stable launch (one read of the codes for the whole panel, 2 to 32 rows, proven against
    MLX's one-row product on this Mac's first use of each shape); where it refuses, counted,
    ``proven_rows`` (MLX's batched product where it is row-stable here, else one row at a time)."""
    if x.shape[1] == 1:
        return ref.linear(x, weight)
    from .qproduct import affine_product

    try:
        return affine_product(x, weight)
    except RuntimeError:
        count("fallback.packed.row_stable_launch_refused")
    key = ("linear", x.shape[1], tuple(weight.codes.shape), weight.bits, weight.group_size, str(x.dtype))
    return proven_rows(key, lambda: ref.linear(x, weight), lambda: per_row_linear(x, weight))


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
    """Stage E (reference). ``(visit output (1, W, 2304) in dtype, MLP conv inputs float32)``.

    Past 8 rows the reference block's router product is the stock product, whose rows need not
    equal the one-row router's (Bare Speed's packed experts at 16 rows: rows 5 to 10 differed,
    receipts/integrate-all-bare.json), so a wide panel routes in pieces of at most 8 rows."""
    normed = ref.norm(source, layer.ffn_norm, layer.eps).astype(dtype)
    if panel.rows > ROW_STABLE_ROWS:
        routed = mx.concatenate([layer.experts(normed[:, start:stop]) for start, stop in row_chunks(panel.rows)],
                                axis=1)
    else:
        routed = layer.experts(normed)
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
# Every pack takes the same stages: where a visit's weights are a format a lane kernel does not
# read (stored codes of an affine pack, another 16-bit type), that visit's stage runs the
# pack's own reference product and the step counts ``fallback.<stage>.<reason>``.
def unreadable(dtype, *weights):
    """Why a 16-bit lane kernel cannot read these weights for a ``dtype`` stream, or None."""
    for weight in weights:
        if weight is None:
            continue
        if isinstance(weight, AffineWeight):
            return f"packed_{weight.bits}bit"
        if weight.dtype != dtype:
            return "weight_dtype"
    return None


def attention_input_reason(layer, dtype, message):
    """Why lane L1's stage A cannot serve this visit, or None. Its projection launch reads q, r,
    gate, k and v as 16-bit or as the affine codes it twins MLX's one-row product for (4 and 8
    bits, ``attn_input.affine_route``), and the sink, the dynamic taps and (loop passes 2 and 3)
    the controllers as 16-bit."""
    from .attn_input import affine_route, matrix

    if dtype != mx.bfloat16:
        return "stream_dtype"
    for name, weight in (("q", layer.q), ("r", layer.r), ("gate", layer.gate), ("k", layer.k), ("v", layer.v)):
        if isinstance(weight, AffineWeight):
            why = affine_route(matrix(weight))
            if why:
                return f"packed_{weight.bits}bit_{name}"
        elif weight.dtype != mx.bfloat16:
            return "weight_dtype"
    packed = [layer.sink, layer.convs[0].dynamic, layer.convs[1].dynamic]
    if message:
        packed += [c.down for c in layer.controllers[:3]]
    if not isinstance(layer.sink, AffineWeight) and layer.sink.dtype != mx.bfloat16:
        return "weight_dtype"
    return next((f"packed_{w.bits}bit" for w in packed if isinstance(w, AffineWeight)), None)


def output_reason(layer, dtype):
    """Why lane L2's output projection cannot serve this visit, or None."""
    return unreadable(dtype, layer.out)


def stored_output_reason(layer, dtype, rows):
    """Why the one-launch stage C on stored codes (``qproduct.affine_output``: the output projection
    on the affine codes in the one-row product's order, the attention's short convolution and the
    residual) cannot serve this visit, or None. Its first use per shape on a Mac still proves it bit
    for bit against the step's separate launches, and refuses (counted) on a disagreement."""
    from .qproduct import qmv_constants

    conv = layer.convs[2]
    if not isinstance(layer.out, AffineWeight):
        return "not_stored"
    if dtype != mx.bfloat16:
        return "stream_dtype"
    if conv.dynamic is not None or conv.size != 4:
        return "conv_form"
    if not 1 <= rows <= MAX_STORED_OUTPUT_ROWS:
        return "rows"
    try:
        qmv_constants(layer.out)
    except RuntimeError as err:
        return str(err).rsplit(": ", 1)[-1]
    return None


#: Panel rows the one-launch stored-code stage C takes (its bit proofs cover 1 to 8, 12, 16 and 32).
MAX_STORED_OUTPUT_ROWS = 32


def output_route_reason(layer, dtype, rows):
    """Why neither fused stage C (16-bit weights: lane L2's launch; stored codes: the one-launch
    stored-code form) serves this visit, or None."""
    reason = output_reason(layer, dtype)
    if reason is None or stored_output_reason(layer, dtype, rows) is None:
        return None
    return reason


def stored_output(layer, hg, x, state, visit):
    """Stage C on a stored-code output projection in one launch, or None where it refuses (counted in
    ``attn_input.route_counts`` as ``refused:qoutput_*``)."""
    from .attn_input import _static_form
    from .qproduct import affine_output

    try:
        return affine_output(hg, layer.out, layer.convs[2], state.tail(visit, 2), x, layer.boundary,
                             form=_static_form(layer.out.codes.shape[0], layer.convs[2].size))
    except RuntimeError:
        return None


#: The trunk's activation dtype, set by the decoder at load (one model a process), so the
#: engine site and the round program ask the attention's qualification for the dtype it serves.
_ACTIVATION = {"dtype": mx.bfloat16}


def set_activation_dtype(dtype):
    _ACTIVATION["dtype"] = dtype


def attention_reason(dtype=None):
    """Why lane L2's fused attention cannot serve on this Mac, or None. Its qualification (once
    per dtype and reduction, at first use) declines a reduction it does not serve (for example
    ``MLX_SDPA_BLOCKS`` past 256), bits that differ from MLX's, or a pipeline this Mac refuses.
    The step then takes the chronological one-row reference attention, counted
    ``fallback.bc_attention.<reason>``, and no program is recorded (its shapes follow the
    live history)."""
    from .attend import decline_reason

    return decline_reason(dtype or _ACTIVATION["dtype"])


def expert_input_reason(layer, message, dtype=mx.bfloat16):
    """Why lane L1's MLP read cannot serve this visit, or None. Its read and controller round to
    bf16 where RC6 rounds to the stream's dtype, so a float16 stream (the FP16 sibling packs)
    takes the reference (review 2, finding 9)."""
    if dtype != mx.bfloat16:
        return "stream_dtype"
    if message and isinstance(layer.controllers[3].down, AffineWeight):
        return f"packed_{layer.controllers[3].down.bits}bit"
    return None


def wide_rows_exact(model, spec):
    """Whether panels wider than 8 rows keep every row equal to the one-row step with this lane
    set on this pack: every visit's stage A and MLP read on lane L1's kernels (the reference
    stages read wide panels with the stock product); experts, the output projection, the final
    read and the head are row-stable at any width (lane kernels, or pieces of 8 rows, or one row
    at a time for stored codes)."""
    lanes = {part.strip() for part in spec.split(",") if part.strip()}
    if not {"a", "bc", "d"} <= lanes:
        return False
    found = stage_fallbacks(model, spec, ROW_STABLE_ROWS + 1)
    return not any(key.startswith(("fallback.a.", "fallback.d.")) for key in found)


def stage_fallbacks(model, spec, rows, dtype=None):
    """``{"fallback.<stage>.<reason>": visits}`` one step of ``rows`` takes with lane set
    ``spec``: the same decisions the stages make per visit, for a step the native engine
    replays (its stage functions do not run, so they cannot count)."""
    from .experts import unserved

    dtype = model.embed_tokens.weight.dtype if dtype is None else dtype
    lanes = {part.strip() for part in spec.split(",") if part.strip()}
    out = {}
    for visit, index in enumerate(model.visit_order):
        layer = model.layers[index]
        message = model.start <= index < model.end and visit != index
        reasons = []
        if "a" in lanes:
            reasons.append(("a", attention_input_reason(layer, dtype, message)))
        if "bc" in lanes:
            reasons.append(("bc_output", output_route_reason(layer, dtype, rows)))
        if "d" in lanes:
            reasons.append(("d", expert_input_reason(layer, message, dtype)))
        if "e" in lanes:
            reasons.append(("experts", unserved(layer.experts, rows, dtype)))
        for stage, reason in reasons:
            if reason is not None:
                # A stored output projection the fused output launch cannot read runs as lane
                # L1's row-stable launch beside it: unfused, not the stock product.
                kind = ("unfused" if stage == "bc_output" and isinstance(layer.out, AffineWeight)
                        and launch_serves(layer.out) else "fallback")
                key = f"{kind}.{stage}.{reason}"
                out[key] = out.get(key, 0) + 1
    if embed_reason(model) is not None:
        out[f"fallback.embed.{embed_reason(model)}"] = 1
    unembed = getattr(model, "unembed", None)
    if rows > 1 and isinstance(unembed, AffineWeight) and not launch_serves(unembed):
        out["fallback.head.packed_rows"] = 1
    return out


def fused_attention(layer, a, x, state, visit, panel):
    """Stage BC on lane L2's kernels: attention over the committed history read in place plus
    the panel's rows, gate, output projection, attention convolution, residual (class A)."""
    from .attend import attend as panel_attend
    from .attn_output import attention_output as panel_output

    declined = attention_reason(x.dtype)
    if declined is not None:
        # The fused attention's qualification declined on this Mac: each row by the one-row
        # reference on its own history, gated as the reference gates; the output stays fused.
        count(f"fallback.bc_attention.{declined}")
        h = chronological_core(layer, a.q, relative_profiles(layer, a.rel), a.sink, a.k_rows, a.v_rows,
                               state, visit, panel)
        hg = (h.astype(mx.float32) * exact_sigmoid(a.gate_logits.astype(mx.float32))).astype(h.dtype)
    else:
        if layer.sliding:
            keys, values, newest, valid = state.window_view(visit)
        else:
            keys, values, valid = state.global_view(visit)
            newest = valid
        hg = panel_attend(a.q, a.rel, a.sink, a.gate_logits, a.k_rows, a.v_rows, keys, values, valid,
                          newest, panel.pos, layer.rel_basis, window=layer.sliding)
    reason = output_reason(layer, hg.dtype)
    if reason is None:
        return panel_output(hg, layer.out, layer.convs[2].weight, state.tail(visit, 2), x,
                            boundary=layer.boundary)
    if stored_output_reason(layer, hg.dtype, panel.rows) is None:
        fused = stored_output(layer, hg, x, state, visit)
        if fused is not None:
            return fused
    kind = "unfused" if isinstance(layer.out, AffineWeight) and launch_serves(layer.out) else "fallback"
    count(f"{kind}.bc_output.{reason}")
    return output_projection(layer, hg, x, state, visit, panel)


def fused_attention_input(layer, x, closed, message, state, visit, panel):
    """Stage A on lane L1's kernels (two launches a visit, three on loop passes 2 and 3)."""
    from .attn_input import stage_attention_input

    reason = attention_input_reason(layer, x.dtype, message is not None)
    if reason is not None:
        count(f"fallback.a.{reason}")
        return attention_input(layer, x, closed, message, state, visit, panel)
    return stage_attention_input(layer, x, closed, message, state, visit, panel)


def fused_expert_input(layer, partial, closed, message, panel):
    """Stage D on lane L1's read kernel (one launch, two on loop passes 2 and 3)."""
    from .read import stage_expert_input

    stream = closed[0].dtype if closed else _ACTIVATION["dtype"]
    reason = expert_input_reason(layer, message is not None, stream)
    if reason is not None:
        count(f"fallback.d.{reason}")
        return expert_input(layer, partial, closed, message, panel)
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


#: Rows RC6's own panel products are proven row-stable for (each row equals the one-row step).
ROW_STABLE_ROWS = 8


def row_chunks(rows):
    """``(start, stop)`` pieces of at most ``ROW_STABLE_ROWS`` rows covering a panel."""
    return [(start, min(rows, start + ROW_STABLE_ROWS)) for start in range(0, rows, ROW_STABLE_ROWS)]


def row_stable_read(panel, closed, x, queries, eps, normalize):
    """The final depth read of a panel, every row equal to the one-row step's read.

    Up to 8 rows this is RC6's own reader. Past 8 rows RC6 switches to a batched read whose
    rows need not equal the one-row read; the reference at every width is the one-row step
    (TARGETS, 06:33), so a wide panel (the copy lane) is read in pieces of at most 8 rows,
    each a proven row-stable read (a last single row is the one-row read itself)."""
    rows = panel.rows
    if rows <= ROW_STABLE_ROWS:
        return ref.depth_reader(panel.tape, rows)(closed, x, queries, eps, normalize)
    parts = []
    for start, stop in row_chunks(rows):
        width = stop - start
        reader = ref.depth_reader(panel.tape if width > 1 else None, width)
        parts.append(reader([block[:, start:stop] for block in closed], x[:, start:stop], queries, eps,
                            normalize))
    return mx.concatenate(parts, axis=1)


def affine_head(model, hidden):
    """``Model.head`` on affine codes with the product as lane L1's row-stable launch: the same
    cast (to the model's dtype, then float32, as ``vocabulary_logits`` feeds an affine head), the
    same vocabulary slice and soft cap, so every row equals the one-row head. Raises
    ``RuntimeError`` where the launch refuses (its first use per shape is proven on this Mac)."""
    from .qproduct import affine_product

    cap = model.config["final_logit_softcapping"]
    x = hidden.astype(model.embed_tokens.weight.dtype).astype(mx.float32)
    logits = affine_product(x, model.unembed)[..., : model.config["unpadded_vocab_size"]]
    return mx.tanh(logits / cap) * cap if cap else logits


def row_stable_head(model, hidden):
    """The vocabulary head of a panel, every row equal to the one-row step's (pieces of at
    most 8 rows past 8, as ``row_stable_read``; a head stored as affine codes through lane L1's
    row-stable launch, else one row at a time, counted)."""
    rows = hidden.shape[1]
    if rows > 1 and isinstance(model.unembed, AffineWeight):
        try:
            return affine_head(model, hidden)
        except RuntimeError:
            pass
        count("fallback.head.packed_rows")
        unembed = model.unembed
        key = ("head", rows, tuple(unembed.codes.shape), unembed.bits, unembed.group_size, str(hidden.dtype))
        return proven_rows(key, lambda: model.head(hidden), lambda: mx.concatenate(
            [model.head(hidden[:, row : row + 1]) for row in range(rows)], axis=1))
    if rows <= ROW_STABLE_ROWS:
        return model.head(hidden)
    return mx.concatenate([model.head(hidden[:, start:stop]) for start, stop in row_chunks(rows)], axis=1)


def embed_reason(model):
    """Why the one-launch embedding cannot serve this model's stored table, or None (a 16-bit table
    needs no decode: one gather, as the model does it)."""
    from .qproduct import rows_reason

    packed = getattr(model.embed_tokens, "packed", None)
    return None if packed is None else rows_reason(packed)


def embed(model, ids):
    """``model.embed_tokens(ids)`` with a table stored as affine codes (Bare Speed) gathered and
    decoded in one launch (``qproduct.affine_rows``: MLX gathers codes, scales and biases, then
    decodes, four launches) and normed as ``Embedding`` norms it; every value equal (proven on this
    Mac's first use). A 16-bit table takes the model's own embedding; a stored table the launch does
    not serve, or a refusal, takes it too, counted."""
    table = model.embed_tokens
    packed = getattr(table, "packed", None)
    if packed is not None:
        from .qproduct import affine_rows

        reason = embed_reason(model)
        if reason is None:
            try:
                return mx.fast.rms_norm(affine_rows(ids, packed), table.gain, table.eps)
            except RuntimeError:
                reason = "refused"
        count(f"fallback.embed.{reason}")
    return table(ids)


def decode_hidden(model, ids, state, *, embeddings=None, capture_ids=(), stages=REFERENCE,
                  record=None, tape=None):
    """The final hidden rows ``(1, W, 2304)``, the captures and the per-visit deltas.

    ``tape`` overrides the reference verify flag (default: verify for 2 or more rows).
    """
    rows = ids.shape[-1] if embeddings is None else embeddings.shape[1]
    verify = (rows > 1) if tape is None else bool(tape)
    panel = Panel(rows, state.pos, True if verify else None)
    closed, captured, deltas = [], {}, []
    x = embed(model, ids.reshape(1, rows)) if embeddings is None else embeddings
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
    last = row_stable_read(panel, closed, x, model.final_query, model.eps,
                           model.config["attn_res_normalize_values"])[..., 0, :]
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
    return row_stable_head(model, hidden), captures, deltas


__all__ = ["AttentionInput", "DecodeState", "LANES", "Message", "Panel", "REFERENCE", "Stages",
           "decode_hidden", "decode_panel", "stages_for", "visit_step"]
