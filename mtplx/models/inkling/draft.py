"""The bundled diffusion drafter attends to committed target features and predicts a token chain."""

import os

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache

from .cache import WindowCache
from .model import linear
from .precision import AffineWeight

DRAFT_HEAD_BITS_ENV = "MTPLX_INKLING_DRAFT_HEAD_BITS"
_DRAFT_HEAD_VALID_BITS = (4, 6, 8)
_DRAFT_HEAD_GROUP_SIZES = (64, 32)
_DRAFT_HEAD_OFF_VALUES = ("", "0", "off", "false", "no", "none")


def draft_head_bits() -> int | None:
    """Configured bit-width for the opt-in draft-only quantized vocabulary head.

    Controlled by ``MTPLX_INKLING_DRAFT_HEAD_BITS`` (default unset/off).
    Accepts ``4``, ``6``, or ``8`` when enabled; leaves the target vocabulary
    head untouched and changes only draft proposal scoring.
    """
    raw = os.environ.get(DRAFT_HEAD_BITS_ENV, "").strip().lower()
    if raw in _DRAFT_HEAD_OFF_VALUES:
        return None
    try:
        bits = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DRAFT_HEAD_BITS_ENV} must be off or one of {_DRAFT_HEAD_VALID_BITS}; got {raw!r}"
        ) from exc
    if bits not in _DRAFT_HEAD_VALID_BITS:
        raise ValueError(
            f"{DRAFT_HEAD_BITS_ENV} must be off or one of {_DRAFT_HEAD_VALID_BITS}; got {bits}"
        )
    return bits


def draft_head_group_size(width: int, preferred: int = 64) -> int | None:
    """Largest valid affine quantization group size dividing ``width``."""
    width = int(width)
    if width <= 0:
        return None
    if preferred in (32, 64, 128) and width % preferred == 0:
        return int(preferred)
    for size in _DRAFT_HEAD_GROUP_SIZES:
        if width % size == 0:
            return size
    return None


def quantize_draft_head(weight, *, bits: int | None = None, group_size: int = 64) -> AffineWeight | None:
    """Build a draft-only affine-quantized copy of a dense 16-bit vocabulary head.

    Returns ``None`` when the opt-in env flag is off, when ``weight`` is already
    quantized (``AffineWeight`` or ``nn.QuantizedLinear``), when its dtype is
    not BF16/FP16, or when its input width has no valid affine group divisor.
    Never mutates ``weight``.
    """
    target_bits = draft_head_bits() if bits is None else int(bits)
    if target_bits is None:
        return None
    if target_bits not in _DRAFT_HEAD_VALID_BITS:
        raise ValueError(
            f"Draft head bits must be one of {_DRAFT_HEAD_VALID_BITS}; got {target_bits}"
        )
    if isinstance(weight, (AffineWeight, nn.QuantizedLinear)):
        return None
    matrix = weight.weight if isinstance(weight, nn.Linear) else weight
    if (
        not isinstance(matrix, mx.array)
        or matrix.ndim != 2
        or matrix.dtype not in (mx.bfloat16, mx.float16)
        or matrix.shape[0] <= 0
        or matrix.shape[1] <= 0
    ):
        return None
    gs = draft_head_group_size(int(matrix.shape[1]), preferred=group_size)
    if gs is None:
        return None
    codes, scales, biases = mx.quantize(matrix, group_size=gs, bits=target_bits)
    head = AffineWeight(codes, scales, biases, bits=target_bits, group_size=gs)
    mx.eval(head.codes, head.scales, head.biases)
    return head


def _one_launch_mix(features):
    """Whether ``Drafter.mix`` can take its four-tap float32 sum in one launch (lane L7's
    ``fast.commit_launches.mix``, the stock products' and adds' bytes): four taps of one shape
    on a Metal device. ``MTPLX_INKLING_COMMIT_LAUNCHES=0`` keeps the stock operations."""
    import os

    if os.environ.get("MTPLX_INKLING_COMMIT_LAUNCHES", "1").strip().lower() in ("0", "off", "false", "no"):
        return False
    if len(features) != 4 or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    shape = features[0].shape
    return all(h.shape == shape for h in features)


def _context_heads(x, weight, name):
    """The context heads a prompt chunk's features feed (the drafter's KV of the prompt): on the width-independent
    prompt read (``fast/prompt_route.py``'s ``invariant`` part) our launch, whose sums do not depend on the rows
    around a row; else the model's product."""
    from .fast.prompt_route import invariant_sums

    if invariant_sums() and isinstance(weight, mx.array) and weight.ndim == 2 and weight.dtype == x.dtype:
        from .fast import prompt_gemm

        flat = x.reshape(-1, x.shape[-1])
        (out,) = prompt_gemm.project(flat, [weight], "drafter_" + name)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    return linear(x, weight)


class DraftBlock(nn.Module):
    def __init__(self, weights, config, index):
        super().__init__()
        self.w = weights
        self.cfg = config
        self.window = (
            config["sliding_window_size"]
            if index in config["local_layer_ids"]
            else None
        )

    def normalize(self, x, name):
        return mx.fast.rms_norm(x, self.w[name + ".weight"], self.cfg["rms_norm_eps"])

    def heads(self, x, name):
        weight = self.w["self_attn." + name + "_proj.weight"]
        projected = (
            _context_heads(x, weight, name)
            if name.endswith("_ctx")
            else linear(x, weight)
        )
        return projected.reshape(*x.shape[:2], -1, self.cfg["head_dim"]).transpose(
            0, 2, 1, 3
        )

    def rotate(self, x, pos):
        return mx.fast.rope(
            x,
            self.cfg["head_dim"],
            traditional=False,
            base=self.cfg["rope_theta"],
            scale=1.0,
            offset=pos,
        )

    def observe(self, features, entry):
        pos = entry.offset
        if self.window and features.shape[1] > self.window:
            skip = features.shape[1] - self.window
            pos += skip
            entry.offset += skip
            features = features[:, -self.window :]
        keys = self.normalize(self.heads(features, "k_ctx"), "self_attn.k_norm_ctx")
        keys, values = self.rotate(keys, pos), self.heads(features, "v_ctx")
        if self.window and features.shape[1] > 1:
            # Lane F1: several rows straight into the ring the next round rebuilds (the same
            # arrays and counters, no host wait there); the generic path where it declines.
            from .fast.draft_ring import observe_rows

            if observe_rows(entry, keys, values):
                return
        entry.update_and_fetch(keys, values)

    def attend(self, x, entry):
        key, value = entry.state
        if self.window:
            key = entry._temporal_order(entry.keys)[:, :, -self.window :]
            value = entry._temporal_order(entry.values)[:, :, -self.window :]
        q = self.rotate(
            self.normalize(self.heads(x, "q"), "self_attn.q_norm"), entry.offset
        )
        k = self.rotate(
            self.normalize(self.heads(x, "k"), "self_attn.k_norm"), entry.offset
        )
        out = mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate([key, k], axis=2),
            mx.concatenate([value, self.heads(x, "v")], axis=2),
            scale=self.cfg["head_dim"] ** -0.5,
        )
        return linear(
            out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1),
            self.w["self_attn.o_proj.weight"],
        )

    def __call__(self, hidden, residual, entry):
        summed = (
            hidden
            if residual is None
            else hidden.astype(mx.float32) + residual.astype(mx.float32)
        )
        attn = self.attend(
            self.normalize(summed, "input_layernorm").astype(hidden.dtype), entry
        )
        summed = summed.astype(hidden.dtype).astype(mx.float32) + attn.astype(
            mx.float32
        )
        x = self.normalize(summed, "post_attention_layernorm").astype(hidden.dtype)
        gate = linear(x, self.w["mlp.gate_proj.weight"]).astype(mx.float32)
        up = linear(x, self.w["mlp.up_proj.weight"]).astype(mx.float32)
        return linear(
            (nn.silu(gate) * up).astype(x.dtype), self.w["mlp.down_proj.weight"]
        ), summed.astype(hidden.dtype)


class Drafter(nn.Module):
    def __init__(self, path, config, target):
        super().__init__()
        self.cfg = config
        weights = mx.load(str(path / "mtp.safetensors"))
        fuse_weight = weights.pop("fuse_weight")
        self.blend = mx.softmax(fuse_weight.astype(mx.float32), axis=-1).astype(
            fuse_weight.dtype
        )
        self.mask = weights.pop("mask_embed")
        self.output_norm = weights.pop("norm.weight")
        self.blocks = []
        for i in range(config["num_hidden_layers"]):
            stem = f"layers.{i}."
            local = {
                k[len(stem) :]: weights.pop(k)
                for k in tuple(weights)
                if k.startswith(stem)
            }
            # Context projections carry their suffix after 'proj' on disk.
            for name in ("k", "v"):
                local[f"self_attn.{name}_ctx_proj.weight"] = local.pop(
                    f"self_attn.{name}_proj_ctx.weight"
                )
            self.blocks.append(DraftBlock(local, config, i))
        self.markov = {
            k.removeprefix("markov_head."): weights.pop(k)
            for k in tuple(weights)
            if k.startswith("markov_head.")
        }
        if weights:
            raise ValueError(f"Unknown DSpark tensors: {sorted(weights)}")
        self.width = config["block_size"]
        self.vocab = target.config["unpadded_vocab_size"]
        self.folded = None
        self.draft_unembed = quantize_draft_head(target.unembed)

    def draft_head(self, target):
        """Vocabulary matrix for draft proposal scoring only; target.unembed stays untouched."""
        head = getattr(self, "draft_unembed", None)
        return target.unembed if head is None else head

    def make_cache(self):
        return [
            WindowCache(max_size=b.window) if b.window else KVCache()
            for b in self.blocks
        ] + [ArraysCache(1)]

    def mix(self, features, row, weight):
        if _one_launch_mix(features):
            # The whole mix in one launch where it applies (lane F1's ``mix_norm``: the sum, the
            # norm, the gain and the cast); else the four-tap float32 sum in one launch, the same
            # bytes as the products and adds below (lane L7's ``mix``), then the same RMS norm.
            from .fast.commit_launches import mix as one_launch_mix, mix_norm

            whole = mix_norm(features, self.blend, row, weight, self.cfg["rms_norm_eps"])
            if whole is not None:
                return whole
            return one_launch_mix(features, self.blend[row], weight, self.cfg["rms_norm_eps"])
        dtype = features[0].dtype
        mixed = sum(
            h.astype(mx.float32) * self.blend[row, i].astype(mx.float32)
            for i, h in enumerate(features)
        )
        return mx.fast.rms_norm(mixed, weight, self.cfg["rms_norm_eps"]).astype(dtype)

    def observe(self, captures, cache):
        for row, block in enumerate(self.blocks):
            features = self.mix(captures, row, block.w["dflare_norm.weight"])
            block.observe(features, cache[row])
        cache[-1][0] = self.mix(
            [h[:, -1:] for h in captures], -1, self.markov["dflare_norm.weight"]
        )

    def fold(self, head):
        if self.folded is None:
            w, b = self.markov["hidden_out.weight"], self.markov["hidden_out.bias"]
            projection = mx.concatenate([w, b[:, None]], axis=1).astype(mx.float32)
            tiles = []
            for first in range(0, self.vocab, 4096):
                end = min(first + 4096, self.vocab)
                rows = (
                    head.rows(first, end)
                    if isinstance(head, AffineWeight)
                    else head[first:end]
                )
                tile = (rows.astype(mx.float32) @ projection).astype(w.dtype)
                mx.eval(tile)
                tiles.append(tile)
            self.folded = mx.concatenate(tiles)
            mx.eval(self.folded)
        return self.folded

    def propose(self, target, anchor, cache, count, *, select_token=None):
        """An explicit sampler selects from each complete Markov logits row."""
        if not 1 <= count < self.width:
            raise ValueError("DSpark proposal count exceeds its trained block")
        embed = target.embed_tokens(mx.array([[anchor]]))
        dtype = embed.dtype
        masks = mx.broadcast_to(self.mask, (1, self.width - 1, embed.shape[-1]))
        h, residual = mx.concatenate([embed, masks], axis=1), None
        for i, block in enumerate(self.blocks):
            h, residual = block(h, residual, cache[i])
        h = mx.fast.rms_norm(
            h.astype(mx.float32) + residual.astype(mx.float32),
            self.output_norm,
            self.cfg["rms_norm_eps"],
        ).astype(dtype)[:, 1 : count + 1]
        draft_head = getattr(self, "draft_head", None)
        head_weight = draft_head(target) if callable(draft_head) else target.unembed
        base = linear(h, head_weight)[..., : self.vocab]
        rank = self.cfg["markov_rank"]
        weights = self.markov["rnn_proj.weight"].astype(mx.float32)
        inputs = mx.concatenate([h, mx.broadcast_to(cache[-1][0], h.shape)], axis=-1)
        conditioned = linear(
            inputs.astype(mx.float32), weights[:, 2 * rank :]
        ) + self.markov["rnn_proj.bias"].astype(mx.float32)
        folded = self.fold(target.unembed)
        state, prev = mx.zeros((1, rank), dtype), mx.array([anchor])
        guesses = []
        for i in range(count):
            inputs = mx.concatenate(
                [state, self.markov["markov_w1.weight"][prev]], axis=-1
            )
            update = (
                linear(inputs.astype(mx.float32), weights[:, : 2 * rank])
                + conditioned[:, i]
            ).astype(dtype)
            gate, candidate, output = mx.split(update, 3, axis=-1)
            gate = mx.sigmoid(gate.astype(mx.float32)).astype(dtype)
            state = gate * state + (1 - gate) * mx.tanh(
                candidate.astype(mx.float32)
            ).astype(dtype)
            signal = mx.tanh(output.astype(mx.float32)).astype(dtype)
            bias = mx.addmm(folded[:, -1], signal, folded[:, :-1].T)
            scores = base[:, i] + bias
            prev = (
                mx.argmax(scores, axis=-1)
                if select_token is None
                else mx.array([select_token(scores[0])])
            )
            guesses.append(prev)
        return mx.stack(guesses, axis=-1)[0]
