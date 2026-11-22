"""Native-resolution perception encoder. Four adjacent 20 px patches make one language embedding."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from .model import linear


def patchify(image: Image.Image, config, processor):
    patch = config["native_patch_size"] * config["adapter_merge_size"]
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.int32)
        y, x = np.ogrid[: image.height, : image.width]
        board = np.where((x // 8 + y // 8) % 2, 153, 204)[..., None]
        alpha = rgba[..., 3:]
        pixels = ((rgba[..., :3] * alpha + board * (255 - alpha) + 127) // 255).astype(
            np.uint8
        )
    else:
        pixels = np.asarray(image.convert("RGB"))
    rows, cols = math.ceil(image.height / patch), math.ceil(image.width / patch)
    padded = np.full((rows * patch, cols * patch, 3), -1, dtype=np.float32)
    padded[: image.height, : image.width] = pixels
    padded = (
        padded * processor["rescale_factor"]
        - np.array(processor["image_mean"], np.float32)
    ) / np.array(processor["image_std"], np.float32)
    return mx.array(padded), (1, rows, cols)


def interpolate_positions(table, height, width):
    """Bilinear interpolation at grid-cell centers, clamped at the learned grid boundary."""
    side = math.isqrt(table.shape[0])
    if side * side != table.shape[0]:
        raise ValueError("Inkling learned position grid must be square")
    table = table.reshape(side, side, -1).astype(mx.float32)

    def axis(length):
        pos = mx.clip(
            (mx.arange(length, dtype=mx.float32) + 0.5) * side / length - 0.5,
            0,
            side - 1,
        )
        lo = pos.astype(mx.int32)
        return lo, mx.minimum(lo + 1, side - 1), pos - lo

    left, right, fx = axis(width)
    top, bottom, fy = axis(height)
    row = table[:, left] * (1 - fx[:, None]) + table[:, right] * fx[:, None]
    return row[top] * (1 - fy[:, None, None]) + row[bottom] * fy[:, None, None]


def rotate(x, cos, sin):
    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    return (
        mx.stack([even * cos - odd * sin, odd * cos + even * sin], axis=-1)
        .reshape(x.shape)
        .astype(x.dtype)
    )


class VisionBlock(nn.Module):
    def __init__(self, w, config):
        super().__init__()
        self.w = w
        self.heads = config["num_attention_heads"]
        self.eps = config["layer_norm_eps"]

    def projection(self, x, name):
        return mx.addmm(self.w[name + ".bias"], x, self.w[name + ".weight"].T)

    def layer_norm(self, x, name):
        return mx.fast.layer_norm(
            x, self.w[name + ".weight"], self.w[name + ".bias"], self.eps
        )

    def __call__(self, x, cos, sin):
        z = self.layer_norm(x, "ln_1")
        heads = [
            self.projection(z, "self_attn." + k + "_proj").reshape(
                len(x), self.heads, -1
            )
            for k in ("q", "k", "v")
        ]
        q, k, v = heads
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=q.shape[-1] ** -0.5,
        )
        x = x + self.w["ls_1.gamma"] * self.projection(
            out[0].transpose(1, 0, 2).reshape(x.shape), "self_attn.out_proj"
        )
        z = nn.gelu(self.projection(self.layer_norm(x, "ln_2"), "mlp.c_fc"))
        return x + self.w["ls_2.gamma"] * self.projection(z, "mlp.c_proj")


class VisionTower(nn.Module):
    def __init__(self, cfg, weights):
        super().__init__()
        self.cfg = cfg
        self.blocks = []
        for i in range(cfg["num_hidden_layers"]):
            stem = f"backbone.layers.{i}."
            block = {
                k[len(stem) :]: weights.pop(k)
                for k in tuple(weights)
                if k.startswith(stem)
            }
            self.blocks.append(VisionBlock(block, cfg))
        self.patch = weights.pop("backbone.embeddings.patch_embedding.weight")
        self.pos = weights.pop("backbone.embeddings.position_embedding")
        self.cls = weights.pop("backbone.embeddings.class_embedding") + weights.pop(
            "backbone.embeddings.cls_position_embedding"
        )
        self.norm_weight, self.norm_bias = (
            weights.pop("backbone.ln_pre.weight"),
            weights.pop("backbone.ln_pre.bias"),
        )
        self.project_in, self.project_out = (
            weights.pop("proj_in.weight"),
            weights.pop("proj_out.weight"),
        )
        if weights:
            raise ValueError(f"Unknown Inkling vision tensors: {sorted(weights)}")
        #: True once an image ran: the pack's mapped tensors are then read into memory (0.59 GiB on the packs).
        self.resident = False

    @property
    def nbytes(self):
        return sum(int(value.nbytes) for _, value in tree_flatten(self.parameters()))

    def encode(self, image, processor):
        self.resident = True
        pixels, grid = patchify(image, self.cfg, processor)
        _, rows, cols = grid
        m, p = self.cfg["adapter_merge_size"], self.cfg["native_patch_size"]
        h, w = rows * m, cols * m

        def merge_order(x):
            return (
                x.reshape(rows, m, cols, m, -1)
                .transpose(0, 2, 1, 3, 4)
                .reshape(h * w, -1)
            )

        raw = pixels.reshape(h, p, w, p, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, -1)
        x = linear(merge_order(raw).astype(self.patch.dtype), self.patch)
        pos = merge_order(interpolate_positions(self.pos, h, w))
        x = mx.concatenate([self.cls[None], x + pos.astype(x.dtype)], axis=0)
        x = mx.fast.layer_norm(
            x, self.norm_weight, self.norm_bias, self.cfg["layer_norm_eps"]
        )
        yy, xx = np.mgrid[:h, :w]
        coords = merge_order(mx.array(np.stack([xx + 1, yy + 1], axis=-1), mx.float32))
        coords = mx.concatenate([mx.zeros((1, 2), mx.float32), coords], axis=0)
        axis_width = (self.cfg["hidden_size"] // self.cfg["num_attention_heads"]) // 2
        inv = self.cfg.get("rope_theta", 10000.0) ** (
            -mx.arange(0, axis_width, 2, dtype=mx.float32) / axis_width
        )
        angle = (coords[:, :, None] * inv).reshape(h * w + 1, 1, -1)
        cos, sin = mx.cos(angle), mx.sin(angle)
        for block in self.blocks:
            x = block(x, cos, sin)
        embeddings = linear(
            nn.gelu(linear(x[1:].reshape(rows * cols, -1), self.project_in)),
            self.project_out,
        )
        return embeddings, rows * cols, grid
