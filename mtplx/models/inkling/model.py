"""Text forward for the fixed-depth loop. Each layer owns weights once and state once per visit."""

from pathlib import Path
import json
import math
import os

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache

from .attention import attend
from .cache import WindowCache
from .activations import expert_inputs
from . import expert_fusion
from .fast import prompt_glue
from .head import fits_route, measured_route, note_route, pack_proofs, vocabulary_logits
from .layout import layer_visits
from .precision import AffineWeight, packed_logical_shape, source_modes
from .projection import panel_linear, serial_messages, stock_messages
from .short_conv import convolve


def stock_linear(x, weight):
    return weight.product(x) if isinstance(weight, AffineWeight) else x @ weight.T


_projection_pack = None
_PANEL_DTYPES = (mx.bfloat16, mx.float16, mx.float32)


def bind_projection_proofs(pack):
    """The per-pack proof resolver the row-stable projection records through."""
    global _projection_pack
    _projection_pack = pack


def _panel_device():
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _same_bits(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    word = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return bool(mx.array_equal(a.view(word), b.view(word)).item())


def row_by_row_linear(x, weight):
    """Each row projected alone by the stock product, as a one-row step projects it."""
    flat = x.reshape(-1, x.shape[-1])
    rows = [stock_linear(flat[row : row + 1], weight) for row in range(flat.shape[0])]
    return mx.concatenate(rows, axis=0).reshape(*x.shape[:-1], weight.shape[0])


def row_stable_linear(x, weight):
    """Project a 2 to 8 row panel so that every row gets the bits a one-row step gives it.

    The panel kernel serves a shape once it has reproduced, on this pack and GPU and on the
    panel's own operands, the rows projected one at a time by the stock product. Until then,
    and wherever it has not, the rows are projected one at a time by the stock product. One
    row, wider panels (prompt chunks) and packed weights take the stock product unchanged.

    The proof alone admits the kernel. The first-use timing race is recorded with the decision
    but does not decide: what the kernel replaces here is the row-by-row product, the slowest
    of the three, so a race lost to timing noise must not turn a proven kernel off.
    """
    if (
        isinstance(weight, AffineWeight)
        or weight.ndim != 2
        or not x.ndim
        or not weight.size
        or x.shape[-1] != weight.shape[1]
    ):
        return stock_linear(x, weight)
    rows = x.size // weight.shape[1]
    if not 2 <= rows <= 8:
        return stock_linear(x, weight)
    if not _panel_device() or x.dtype not in _PANEL_DTYPES or weight.dtype not in _PANEL_DTYPES:
        note_route("projection", "rows_no_kernel")
        return row_by_row_linear(x, weight)
    shape = (rows, weight.shape[1], weight.shape[0], str(x.dtype), str(weight.dtype))
    if measured_route(
        "projection",
        shape,
        lambda: row_by_row_linear(x, weight),
        lambda: panel_linear(x, weight),
        work_bytes=16 * rows * weight.shape[0],
        proof=_same_bits,
        trial=True,
        pack=_projection_pack,
    ):
        return panel_linear(x, weight)
    return row_by_row_linear(x, weight)


# MTPLX_INKLING_PROJECTION=stock returns every width to the batched stock product.
linear = (
    stock_linear
    if os.environ.get("MTPLX_INKLING_PROJECTION", "candidate") == "stock"
    else row_stable_linear
)
# A verify panel reads its cross-pass messages one row at a time, as a one-row step does.
cross_pass_message = (
    stock_messages if os.environ.get("MTPLX_INKLING_MESSAGES", "serial") == "stock" else serial_messages
)


def norm(x, weight, eps):
    return mx.fast.rms_norm(x.astype(mx.float32), weight, eps)


def clip_heads(x, limit, scale=None):
    value = x.astype(mx.float32)
    if scale is not None:
        value = value * scale.astype(mx.float32)
    factor = mx.minimum(
        1.0, limit / (mx.linalg.norm(value, axis=-1, keepdims=True) + 1e-6)
    )
    return (value * factor).astype(x.dtype)


def read_blocks(closed, partial, queries, eps, normalize_values):
    values = mx.stack([*closed, partial], axis=-2).astype(mx.float32)
    keys = mx.fast.rms_norm(values, None, eps)
    q = queries.reshape(-1, partial.shape[-1]).astype(mx.float32)
    p = mx.softmax((q @ keys.swapaxes(-1, -2)) * partial.shape[-1] ** -0.5, axis=-1)
    return p @ (keys if normalize_values else values)


def serial_depth_dots(closed, partial, queries, eps, normalize_values):
    values = mx.stack([*closed, partial], axis=-2).astype(mx.float32)
    keys = mx.fast.rms_norm(values, None, eps)
    q = queries.reshape(-1, partial.shape[-1]).astype(mx.float32)
    scores = mx.concatenate(
        [q @ keys[:, row : row + 1].swapaxes(-1, -2) for row in range(partial.shape[1])],
        axis=1,
    )
    probabilities = mx.softmax(scores * partial.shape[-1] ** -0.5, axis=-1)
    return probabilities @ (keys if normalize_values else values)


def serial_depth_read(closed, partial, queries, eps, normalize_values):
    return mx.concatenate([
        read_blocks([v[:, row : row + 1] for v in closed], partial[:, row : row + 1],
                    queries, eps, normalize_values)
        for row in range(partial.shape[1])
    ], axis=1)


# A 2 to 8 row verify takes its within-pass dot products one row at a time, as a one-row step does.
short_verify_depth = {"dots": serial_depth_dots, "rows": serial_depth_read}.get(
    os.environ.get("MTPLX_INKLING_VERIFY_DEPTH_READ", "dots"), read_blocks
)


def depth_reader(tape, width):
    return short_verify_depth if tape is not None and 2 <= width <= 8 else read_blocks


class Controller(nn.Module):
    def __init__(self, weights, stem):
        super().__init__()
        self.down = weights.pop(stem + ".down_weight")
        self.forget = weights.pop(stem + ".forget_weight").astype(mx.float32)
        self.gain = weights.pop(stem + ".gain_weight").astype(mx.float32)

    def __call__(self, x, message, eps):
        x = x.astype(mx.float32)
        message = message.astype(mx.float32)
        mixed = mx.concatenate([norm(x, None, eps), norm(message, None, eps)], axis=-1)
        state = linear(mixed.astype(self.down.dtype), self.down).astype(mx.float32)
        return (1 - mx.tanh(linear(state, self.forget))) * x + mx.tanh(
            linear(state, self.gain)
        ) * message


def use_sorted_expert_inputs():
    return (
        os.environ.get("MTPLX_INKLING_SORTED_INPUTS", "auto").lower()
        in ("auto", "1", "on")
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


class PackedExperts(nn.Module):
    """Keep the native codes and per-group scales. The FP4 outer scale is per expert."""

    def __init__(self, weights, stem, mode):
        super().__init__()
        self.mode = mode
        self.sorted_inputs = use_sorted_expert_inputs()
        self.proof_pack = None
        self.sorted_trial = os.environ.get(
            "MTPLX_INKLING_SORTED_INPUTS", "auto"
        ).lower() in ("1", "on")
        self.sorted_panel_calls = 0
        self.indexed_panel_calls = 0
        weight = weights.pop(stem)
        self.affine = weight if isinstance(weight, AffineWeight) else None
        if self.affine is not None:
            return
        self.codes = weight.view(mx.uint32)
        self.scales = weights.pop(stem + "_scale")
        self.outer = None
        if mode == "nvfp4":
            self.outer = weights.pop(stem + "_amax") / weights.pop(
                stem + "_outer_scale_denominator"
            )

    def __call__(self, rows, experts, row_ids=None, *, sorted_routes=False):
        fp4 = self.mode == "nvfp4"
        inputs, row_scales = expert_inputs(rows, self.mode)
        if self.affine is not None:
            w = self.affine
            result = mx.gather_qmm(
                inputs,
                w.codes,
                w.scales,
                w.biases,
                lhs_indices=row_ids,
                rhs_indices=experts,
                sorted_indices=sorted_routes,
                bits=w.bits,
                group_size=w.group_size,
            )
            if row_scales is not None:
                result = result * (
                    row_scales if row_ids is None else row_scales[row_ids]
                )
            return result.astype(rows.dtype)

        def product(grouped):
            return mx.gather_qmm(
                inputs[row_ids] if grouped else inputs,
                self.codes,
                scales=self.scales,
                rhs_indices=experts,
                lhs_indices=None if grouped else row_ids,
                group_size=16 if fp4 else 32,
                bits=4 if fp4 else 8,
                mode=self.mode,
                sorted_indices=sorted_routes,
            )

        grouped = False
        if sorted_routes and row_ids is not None:
            gather_bytes = experts.size * inputs.shape[-1] * inputs.dtype.size
            name = "experts." + self.mode
            if not self.sorted_inputs or not 64 <= rows.shape[0] <= 2048:
                note_route(name, "disabled_or_shape")
            elif not fp4:
                grouped = self.sorted_trial and fits_route(gather_bytes)
                reason = "run_no_fit" if self.sorted_trial else "stock_pending_class_c"
                note_route(name, "explicit_mxfp8_trial" if grouped else reason)
            else:
                outputs = experts.size * self.codes.shape[-2] * inputs.dtype.size
                shape = (str(rows.dtype), *rows.shape, *self.codes.shape, experts.size)
                grouped = measured_route(
                    name,
                    shape,
                    lambda: product(False),
                    lambda: product(True),
                    work_bytes=gather_bytes + 4 * outputs,
                    extra_bytes=gather_bytes,
                    trial=self.sorted_trial,
                    pack=self.proof_pack,
                )
            self.sorted_panel_calls += int(grouped)
            self.indexed_panel_calls += int(not grouped)
        result = product(grouped)
        if self.outer is not None:
            scale = row_scales if row_ids is None else row_scales[row_ids]
            result = result * (scale * self.outer[experts][..., None, None])
        return result.astype(rows.dtype)


@mx.compile
def activated_experts(pairs, gate_cap, up_cap):
    gate = pairs[..., 0::2].astype(mx.float32)
    up = pairs[..., 1::2].astype(mx.float32)
    return (
        gate_cap
        * mx.tanh(gate / gate_cap)
        * mx.sigmoid(gate)
        * (up_cap * mx.tanh(up / up_cap))
    ).astype(pairs.dtype)


class Experts(nn.Module):
    def __init__(self, weights, config, mode):
        super().__init__()
        self.router = weights.pop("mlp.gate.weight").astype(mx.float32)
        self.bias = weights.pop("mlp.gate.bias")
        self.gate_up = PackedExperts(weights, "mlp.experts.w13_weight", mode)
        self.down = PackedExperts(weights, "mlp.experts.w2_weight", mode)
        self.count = config["num_experts_per_tok"]
        self.scale = config["route_scale"]
        self.gate_cap = config["mlp_gate_softcap"]
        self.up_cap = config["mlp_up_softcap"]

    def __call__(self, hidden):
        # 1-32 rows take four fused kernels around the same router matmul, sigmoid and
        # quantized gathers once proven on this Mac (class A); the rest runs the stock body.
        return expert_fusion.experts_forward(self, hidden, linear, self.stock)

    def stock(self, hidden):
        flat = hidden.reshape(-1, hidden.shape[-1])
        scores = mx.sigmoid(linear(flat.astype(mx.float32), self.router))
        selected = mx.argpartition(-(scores + self.bias), kth=self.count - 1, axis=-1)[
            :, : self.count
        ]
        prob = mx.take_along_axis(scores, selected, axis=-1)
        prob = prob / (prob.sum(axis=-1, keepdims=True) + 1e-20) * self.scale
        experts = selected.flatten()
        group = flat.shape[0] > 32
        order = mx.argsort(experts) if group else mx.arange(experts.size)
        experts = experts[order] if group else experts
        pairs = self.gate_up(
            flat[:, None, :], experts, order // self.count, sorted_routes=group
        )
        mid = activated_experts(pairs, self.gate_cap, self.up_cap)
        out = self.down(mid, experts, sorted_routes=group)[:, 0, :]
        if group:
            out = out[mx.argsort(order)]
        out = out.reshape(*selected.shape, hidden.shape[-1])
        return (
            (out.astype(mx.float32) * prob[..., None])
            .sum(axis=-2)
            .astype(hidden.dtype)
            .reshape(hidden.shape)
        )


class Convolution(nn.Module):
    def __init__(self, weights, stem, is_dynamic=False):
        super().__init__()
        self.dynamic = weights.pop(stem + ".weight_proj") if is_dynamic else None
        self.weight = (
            weights.pop(stem + ".static_w")
            if is_dynamic
            else weights.pop(stem + ".weight")[:, 0, :].T
        ).astype(mx.float32)
        self.size = self.weight.shape[0]

    def __call__(self, x, cache, slot, drive=None, tape=None, add=None, out_dtype=None):
        """The convolution's output; with ``out_dtype`` also the visit's residual step after it:
        ``(out.astype(float32) + add.astype(float32)).astype(out_dtype)`` (``add`` optional)."""
        size = self.size
        change = None if self.dynamic is None else linear(drive, self.dynamic)
        fused = convolve(x, cache[slot], self.weight, change, tape is not None, add, out_dtype)
        if fused is not None:
            out, cache[slot], history = fused
            if tape is not None:
                tape[slot] = history
            return out
        prev = cache[slot]
        xf = x.astype(mx.float32)
        if prev is None:
            prev = mx.zeros((x.shape[0], size - 1, x.shape[-1]), mx.float32)
        history = mx.concatenate([prev, xf], axis=1)
        if x.shape[1] > size:
            # A slice would keep the entire prefill allocation alive in each visit's state.
            cache[slot] = mx.take(
                xf, mx.arange(x.shape[1] - size + 1, x.shape[1]), axis=1
            )
        else:
            cache[slot] = history[:, -(size - 1) :]
        if tape is not None:
            tape[slot] = history
        if self.dynamic is None:
            weighted = mx.conv1d(history, self.weight.T[:, :, None], groups=x.shape[-1])
            divisor = math.sqrt(size + 1)
        else:
            heads = self.dynamic.shape[0] // size
            change = (
                change.astype(mx.float32)
                .reshape(*x.shape[:2], heads, size)
            )
            weighted = mx.zeros((*x.shape[:2], heads, x.shape[-1] // heads), mx.float32)
            for lag in range(size):
                current = history[
                    :, size - 1 - lag : size - 1 - lag + x.shape[1]
                ].reshape(weighted.shape)
                coefficients = (
                    self.weight[lag].reshape(heads, -1) + change[..., lag, None]
                )
                weighted = weighted + current * coefficients
            weighted = weighted.reshape(x.shape)
            divisor = math.sqrt(2 * size + 1)
        out = ((xf + weighted) / divisor).astype(x.dtype)
        if out_dtype is None:
            return out
        joined = out.astype(mx.float32)
        if add is not None:
            joined = joined + add.astype(mx.float32)
        return joined.astype(out_dtype)


class Layer(nn.Module):
    def __init__(self, weights, cfg, index, attn_query, mlp_query, mode):
        super().__init__()
        self.cfg = cfg
        self.sliding = index in cfg["local_layer_ids"]
        self.boundary = index % cfg["n_layers_per_attn_res_block"] == 0
        self.attn_query, self.mlp_query = attn_query, mlp_query
        self.eps = cfg["rms_norm_eps"]
        self.qkv_norm = (
            [weights.pop("attn_norm.weight")]
            if self.sliding
            else [weights.pop(k + "_input_norm.weight") for k in ("q", "k", "v")]
        )
        self.ffn_norm = weights.pop("mlp_norm.weight")
        stem = "attn."
        self.q, self.k, self.v, self.r, self.sink, self.gate, self.out = (
            weights.pop(stem + k + ".weight")
            for k in (
                "wq_du",
                "wk_dv",
                "wv_dv",
                "wr_du",
                "attn_sink_proj",
                "gate_attn",
                "wo_ud",
            )
        )
        self.q_scale = weights.pop(stem + "q_l2_clip.scale")
        self.r_scale = weights.pop(stem + "rel_l2_clip.scale")
        self.rel_basis = weights.pop(stem + "rel_logits_proj.proj")
        self.convs = [
            Convolution(weights, stem + k, not self.sliding)
            for k in ("k_sconv", "v_sconv")
        ]
        self.convs += [
            Convolution(weights, "attn_sconv"),
            Convolution(weights, "mlp_sconv"),
        ]
        self.experts = Experts(weights, cfg, mode)
        self.controllers = (
            [Controller(weights, "controllers." + k) for k in ("attn", "k", "v", "mlp")]
            if "controllers.attn.down_weight" in weights
            else []
        )
        if weights:
            raise ValueError(
                f"Unrecognized Inkling layer {index} tensors: {sorted(weights)}"
            )

    def attention(self, q_input, k_input, v_input, kv, conv, tape):
        cfg = self.cfg
        shape = (*q_input.shape[:2], self.q_scale.shape[0])
        width = self.q_scale.shape[-1]
        q_lin = linear(q_input, self.q)
        r_lin = linear(q_input, self.r)
        k_conv = self.convs[0](linear(k_input, self.k), conv, 0, k_input, tape)
        v_conv = self.convs[1](linear(v_input, self.v), conv, 1, v_input, tape)
        # A prompt chunk's clips, value norm and position scale run as one launch of ours, same bits.
        joined = (
            prompt_glue.heads(self, q_lin, r_lin, k_conv, v_conv, kv.offset)
            if tape is None
            else None
        )
        if joined is not None:
            q, rel, k, v = joined
        else:
            q = clip_heads(q_lin.reshape(*shape, width), cfg["qk_l2_clip_threshold"], self.q_scale)
            rel = clip_heads(r_lin.reshape(*shape, -1), cfg["rel_l2_clip_threshold"], self.r_scale)
            k = clip_heads(k_conv.reshape(*k_conv.shape[:2], -1, width), cfg["qk_l2_clip_threshold"])
            v = v_conv.reshape(*v_conv.shape[:2], -1, width)
            if cfg["use_value_norm_no_gamma"]:
                v = norm(v, None, self.eps).astype(v_input.dtype)
            if not self.sliding:
                pos = mx.arange(q_input.shape[1]) + kv.offset + 1
                tau = 1 + cfg["log_scaling_alpha"] * mx.log(
                    mx.maximum(pos / cfg["log_scaling_n_floor"], 1)
                )
                q = (q.astype(mx.float32) * tau[:, None, None]).astype(q.dtype)
                rel = (rel.astype(mx.float32) * tau[:, None, None]).astype(rel.dtype)
        k, v = kv.update_and_fetch(k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))
        if self.sliding:
            k, v = kv._temporal_order(k), kv._temporal_order(v)
        window = cfg["sliding_window_size"] if self.sliding else 0
        h = (
            attend(
                q.transpose(0, 2, 1, 3),
                k,
                v,
                rel @ self.rel_basis.astype(rel.dtype),
                linear(q_input, self.sink),
                window,
                **({"verify": True} if tape is not None else {}),
            )
            .transpose(0, 2, 1, 3)
            .reshape(*q_input.shape[:2], -1)
        )
        gate = mx.sigmoid(linear(q_input, self.gate).astype(mx.float32))
        return linear((h.astype(mx.float32) * gate).astype(h.dtype), self.out)

    def __call__(self, x, closed, kv, conv, messages=None, tape=None):
        read_depth = depth_reader(tape, x.shape[1])
        # A prompt chunk's one-query read and its norm run as one launch of ours, same bits.
        fused_in = (
            prompt_glue.input_norm(x, closed, self.attn_query, self.qkv_norm[0], self.eps,
                                   "attention_input", stand_in=self.experts.router)
            if tape is None and self.sliding and messages is None and self.cfg["attn_res_normalize_values"]
            else None
        )
        read = (
            None
            if fused_in is not None
            else read_depth(
                closed,
                x,
                self.attn_query,
                self.eps,
                self.cfg["attn_res_normalize_values"],
            )
            if self.attn_query is not None
            else x[..., None, :]
        )
        if fused_in is not None:
            q = k = v = fused_in
        elif self.sliding and messages is None:
            q = k = v = norm(read[..., 0, :], self.qkv_norm[0], self.eps).astype(
                x.dtype
            )
        else:
            sources = [read[..., 0 if self.sliding else i, :] for i in range(3)]
            if messages is not None:
                sources = [
                    self.controllers[i](h, messages[..., i, :], self.eps)
                    for i, h in enumerate(sources)
                ]
            q, k, v = [
                norm(h, self.qkv_norm[0 if self.sliding else i], self.eps).astype(
                    x.dtype
                )
                for i, h in enumerate(sources)
            ]
        attn = self.attention(q, k, v, kv, conv, tape)
        # The attention convolution and the residual join in one launch where the convolution is ours.
        residual = self.convs[2](
            attn, conv, 2, tape=tape, add=None if self.boundary else x, out_dtype=mx.float32
        )
        if self.boundary:
            closed.append(x)
        ffn_in = (
            prompt_glue.input_norm(residual, closed, self.mlp_query, self.ffn_norm, self.eps, "expert_input")
            if tape is None and messages is None and self.cfg["attn_res_normalize_values"]
            else None
        )
        if ffn_in is None:
            ffn = read_depth(
                closed,
                residual,
                self.mlp_query,
                self.eps,
                self.cfg["attn_res_normalize_values"],
            )[..., 0, :]
            if messages is not None:
                ffn = self.controllers[3](ffn, messages[..., 3, :], self.eps)
            ffn_in = norm(ffn, self.ffn_norm, self.eps).astype(x.dtype)
        routed = self.experts(ffn_in)
        return self.convs[3](routed, conv, 3, tape=tape, add=residual, out_dtype=x.dtype)


class Embedding(nn.Module):
    def __init__(self, weight, gain, eps):
        super().__init__()
        self.gain, self.eps = gain, eps
        # A 4-bit embedding (Bare Speed) dequantizes only the rows a step gathers; `weight` then
        # answers .dtype and .shape for the code that sizes buffers from it, and holds no data.
        self.packed = weight if isinstance(weight, AffineWeight) else None
        self.weight = packed_logical_shape(weight) if self.packed is not None else weight

    def __call__(self, ids):
        if self.packed is None:
            return mx.fast.rms_norm(self.weight[ids], self.gain, self.eps)
        w = self.packed
        rows = mx.dequantize(
            w.codes[ids], w.scales[ids], w.biases[ids], bits=w.bits, group_size=w.group_size
        )
        return mx.fast.rms_norm(rows, self.gain, self.eps)


class Model(nn.Module):
    def __init__(self, config, weights, quant_modes):
        super().__init__()
        self.args = self.config = config["text_config"]
        cfg = self.config
        loop = cfg["loop_config"]
        self.start, self.end, self.passes = (
            loop["core_start"],
            loop["core_end"],
            loop["num_passes"],
        )
        count = cfg["num_hidden_layers"]
        self.visit_order = layer_visits(cfg)
        self.eps = cfg["rms_norm_eps"]
        self.embed_tokens = Embedding(
            weights.pop("embed.weight"), weights.pop("embed_norm.weight"), self.eps
        )
        self.final_norm, self.unembed = (
            weights.pop("norm.weight"),
            weights.pop("unembed.weight"),
        )
        self.final_query = weights.pop("attn_res.output_query")
        size = self.end - self.start
        main = weights.pop("middle_cycle_cross_pass_queries").reshape(size, 2, -1)
        kv = weights.pop("middle_cycle_cross_pass_kv_queries").reshape(size, 2, -1)
        self.message_queries = mx.concatenate([main[:, :1], kv, main[:, 1:]], axis=1)
        self.capture_ids = tuple(config["mtp_config"]["target_layer_ids"])
        self.layers = []
        for index in range(count):
            stem = f"layers.{index}."
            params = {
                key[len(stem) :]: weights.pop(key)
                for key in tuple(weights)
                if key.startswith(stem)
            }
            aq = weights.pop(f"attn_res.attn_queries.{index - 1}") if index else None
            mq = weights.pop(f"attn_res.mlp_queries.{index}")
            self.layers.append(Layer(params, cfg, index, aq, mq, quant_modes[index]))
        if weights:
            raise ValueError(f"Unrecognized Inkling target tensors: {sorted(weights)}")

    def make_cache(self):
        cache = []
        for index in self.visit_order:
            kv = (
                WindowCache(max_size=self.config["sliding_window_size"])
                if self.layers[index].sliding
                else KVCache()
            )
            cache.extend((kv, ArraysCache(4)))
        return cache

    def head(self, h):
        cap = self.config["final_logit_softcapping"]
        logits = vocabulary_logits(
            h.astype(self.embed_tokens.weight.dtype),
            self.unembed,
            self.config["unpadded_vocab_size"],
            pack=getattr(self, "proof_pack", None),
        )
        return mx.tanh(logits / cap) * cap if cap else logits

    def forward(self, ids, cache, embeddings=None, capture_ids=(), tapes=None):
        if len(cache) != 2 * len(self.visit_order):
            raise ValueError(
                "Inkling requires separate attention and convolution state for every layer visit"
            )
        closed, captured = [], {}
        x = self.embed_tokens(ids) if embeddings is None else embeddings
        for visit, index in enumerate(self.visit_order):
            if index == self.start:
                if visit == self.start:
                    entry, entry_blocks = x, list(closed)
                    messages = None
                else:
                    prior = [*closed[len(entry_blocks) + 1 :], x]
                    stacked = mx.stack(prior, axis=-2).astype(mx.float32)
                    messages = mx.fast.rms_norm(stacked, None, self.eps)
                    x, closed = entry, list(entry_blocks)
            if visit in capture_ids:
                captured[visit] = x
            message = None
            if self.start <= index < self.end and messages is not None:
                read_message = cross_pass_message if tapes is not None else stock_messages
                message = read_message(
                    self.message_queries[index - self.start], messages, x.shape[-1] ** -0.5
                )
            tape = None if tapes is None else tapes[visit]
            x = self.layers[index](
                x, closed, cache[2 * visit], cache[2 * visit + 1], message, tape
            )
        hidden = (
            prompt_glue.input_norm(x, closed, self.final_query, self.final_norm, self.eps, "final_norm")
            if tapes is None and self.config["attn_res_normalize_values"]
            else None
        )
        if hidden is None:
            last = depth_reader(tapes, x.shape[1])(
                closed,
                x,
                self.final_query,
                self.eps,
                self.config["attn_res_normalize_values"],
            )[..., 0, :]
            hidden = norm(last, self.final_norm, self.eps).astype(x.dtype)
        if len(self.visit_order) in capture_ids:
            captured[len(self.visit_order)] = hidden
        return hidden, [captured[i] for i in capture_ids]

    def __call__(
        self,
        input_ids,
        cache=None,
        *,
        return_hidden=False,
        input_embeddings=None,
        emit_logits=True,
        logits_keep=None,
        hidden_variant=None,
    ):
        hidden, captures = self.forward(
            input_ids,
            self.make_cache() if cache is None else cache,
            input_embeddings,
            self.capture_ids if return_hidden else (),
        )
        tail = hidden if logits_keep is None else hidden[:, -max(1, logits_keep) :]
        logits = self.head(tail) if emit_logits else None
        return (logits, mx.concatenate(captures, axis=-1)) if return_hidden else logits


def load_target(directory):
    path = Path(directory)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "inkling2_loop":
        raise ValueError("Expected an Inkling 2 Mini checkpoint")
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = {}
    for shard in sorted(set(index.values())):
        for key, value in mx.load(str(path / shard)).items():
            if index.get(key) != shard or key in tensors:
                raise ValueError(f"Checkpoint shard index disagrees for {key}")
            tensors[key] = value
    if set(tensors) != set(index):
        raise ValueError("Missing tensors in Inkling checkpoint")
    recipe = config.get("mtplx_inkling_precision", {})
    for key, layout in recipe.items():
        tensors[key] = AffineWeight(
            tensors[key],
            tensors.pop(key + ".mtplx_scales"),
            tensors.pop(key + ".mtplx_biases"),
            layout["bits"],
            layout["group_size"],
        )
    prefix = "model.llm."
    target = {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}
    vision = {
        k.removeprefix("model.visual."): v
        for k, v in tensors.items()
        if k.startswith("model.visual.")
    }
    if not vision or len(vision) + len(target) != len(tensors):
        raise ValueError("Missing vision tower or unknown Inkling weight namespace")
    modes = source_modes(path, config["text_config"]["num_hidden_layers"])
    model = Model(config, target, modes)
    model.proof_pack = pack_proofs(path)
    bind_projection_proofs(model.proof_pack)
    for layer in model.layers:
        layer.experts.gate_up.proof_pack = layer.experts.down.proof_pack = model.proof_pack
    return model, vision, config
