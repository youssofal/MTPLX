# Ported from mlx-vlm (Apache-2.0), Blaizzy/mlx-vlm, adapted to MTPLX checkpoint naming.
"""Qwen3-VL vision tower for qwen3_5 / qwen3_6 checkpoints.

Deepstack note: the mlx-vlm qwen3_vl reference builds one dedicated
``PatchMerger(use_postshuffle_norm=True)`` per entry in
``deepstack_visual_indexes`` and taps the hidden states of those tower
blocks; list element ``i`` is then injected after decoder layer ``i``.
The qwen3_5 family (which mlx-vlm's qwen3_5 model subclasses from
qwen3_vl wholesale) hard-disables deepstack: its VisionConfig raises if
``deepstack_visual_indexes`` is non-empty and forces it to ``[]``. Our
checkpoints ship ``"deepstack_visual_indexes": []`` and no
``deepstack_merger_list`` weights, so this port keeps the faithful
general path (dedicated per-tap mergers) which constructs zero deepstack
mergers for this family and returns an empty deepstack list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

_FUSED_SDPA_DIMS = (64, 80, 128)

# Vision tower weights ship under ``vision_tower.*`` (mlx-vlm qwen3_5/3_6
# layout) or ``model.visual.*`` (HF Qwen3_5MoeForConditionalGeneration).
_VISION_PREFIXES = ("vision_tower.", "model.visual.")


def resolve_vision_prefix(weight_map: dict) -> str | None:
    """Return the checkpoint prefix the vision tower weights live under, or
    ``None`` when the index carries no vision tower tensors (a text-only
    checkpoint whose config.json merely inherits the architecture template).
    """
    for prefix in _VISION_PREFIXES:
        if any(key.startswith(prefix) for key in weight_map):
            return prefix
    return None


@dataclass
class Qwen3VLVisionConfig:
    model_type: str = "qwen3_5"
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    out_hidden_size: int = 5120
    num_heads: int = 16
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, params: dict) -> "Qwen3VLVisionConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in known})


def _fused_sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    # MLX's fused SDPA kernel only supports certain head dims; pad up to the
    # nearest supported size (72 -> 80 for this family) and slice back.
    d = q.shape[-1]
    target = next((t for t in _FUSED_SDPA_DIMS if d <= t), d)
    if target != d:
        pad = [(0, 0)] * (q.ndim - 1) + [(0, target - d)]
        q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)[..., :d]


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb_vision(tensor: mx.array, freqs: mx.array) -> mx.array:
    orig_dtype = tensor.dtype

    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    cos = mx.expand_dims(cos, axis=1)
    cos = mx.tile(cos, (1, 1, 2))
    cos = mx.expand_dims(cos, axis=0)

    sin = mx.expand_dims(sin, axis=1)
    sin = mx.tile(sin, (1, 1, 2))
    sin = mx.expand_dims(sin, axis=0)

    output = (tensor * cos) + (_rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


def _conv_weight_in_mlx_layout(arr: mx.array) -> bool:
    # MLX Conv3d layout is (out, kD, kH, kW, in); PyTorch checkpoints store
    # (out, in, kD, kH, kW) and need a transpose.
    shape = arr.shape
    if len(shape) != 5:
        return False
    _, out_channels, kh, kw, t = shape
    if t == 3:
        return True
    return out_channels >= kh and out_channels >= kw and kh == kw


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        return mx.outer(seq, inv_freq)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)
        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(-1, self.hidden_size)


class PatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(
            x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).reshape(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self, x: mx.array, split_indices: list[int], rotary_pos_emb: mx.array
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)

        q = _apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = _apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Attention is restricted to each frame; split_indices are the interior
        # cumulative sequence boundaries (empty for a single image).
        splits = [mx.split(tensor, split_indices, axis=2) for tensor in (q, k, v)]

        attn_outputs = [
            _fused_sdpa(qs, ks, vs, self.scale) for qs, ks, vs in zip(*splits)
        ]

        output = mx.concatenate(attn_outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.act_fn = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(dim=config.hidden_size, num_heads=config.num_heads)
        self.mlp = MLP(dim=config.hidden_size, hidden_dim=config.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        split_indices: list[int],
        rotary_pos_emb: mx.array,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), split_indices, rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def _normalize_grid(grid_thw) -> list[tuple[int, int, int]]:
    if isinstance(grid_thw, mx.array):
        grid_thw = grid_thw.tolist()
    return [(int(t), int(h), int(w)) for t, h, w in grid_thw]


class Qwen3VLVisionTower(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config=config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = [
            PatchMerger(config=config, use_postshuffle_norm=True)
            for _ in range(len(self.deepstack_visual_indexes))
        ]

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "Qwen3VLVisionTower":
        model_dir = Path(path)
        config = json.loads((model_dir / "config.json").read_text())
        vision_config = config.get("vision_config")
        if not isinstance(vision_config, dict):
            raise ValueError(f"{model_dir} has no vision_config in config.json")

        tower = cls(Qwen3VLVisionConfig.from_dict(vision_config))

        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        weight_map: dict[str, str] = index["weight_map"]
        prefix = resolve_vision_prefix(weight_map)
        if prefix is None:
            raise ValueError(
                f"{model_dir} has no vision tower tensors "
                f"(neither vision_tower.* nor model.visual.*)"
            )
        shards = sorted(
            {shard for key, shard in weight_map.items() if key.startswith(prefix)}
        )

        weights: dict[str, mx.array] = {}
        for shard in shards:
            for key, value in mx.load(str(model_dir / shard)).items():
                if key.startswith(prefix):
                    weights[key[len(prefix) :]] = value

        conv_key = "patch_embed.proj.weight"
        if conv_key in weights and not _conv_weight_in_mlx_layout(weights[conv_key]):
            weights[conv_key] = weights[conv_key].transpose(0, 2, 3, 4, 1)

        tower.load_weights(list(weights.items()), strict=True)
        mx.eval(tower.parameters())
        return tower

    def rot_pos_emb(self, grid: list[tuple[int, int, int]]) -> mx.array:
        merge_size = self.spatial_merge_size

        max_hw = max(max(h, w) for _, h, w in grid)
        freq_table = self.rotary_pos_emb(max_hw)

        pos_ids = []
        for num_frames, height, width in grid:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = mx.arange(merged_h)
            block_cols = mx.arange(merged_w)
            intra_row = mx.arange(merge_size)
            intra_col = mx.arange(merge_size)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )

            row_idx = mx.broadcast_to(
                row_idx, (merged_h, merged_w, merge_size, merge_size)
            ).reshape(-1)
            col_idx = mx.broadcast_to(
                col_idx, (merged_h, merged_w, merge_size, merge_size)
            ).reshape(-1)

            coords = mx.stack([row_idx, col_idx], axis=-1)
            if num_frames > 1:
                coords = mx.tile(coords, (num_frames, 1))
            pos_ids.append(coords)

        pos_ids = mx.concatenate(pos_ids, axis=0)

        h_embeddings = freq_table[pos_ids[:, 0]]
        w_embeddings = freq_table[pos_ids[:, 1]]
        return mx.concatenate([h_embeddings, w_embeddings], axis=-1)

    def fast_pos_embed_interpolate(
        self, grid: list[tuple[int, int, int]]
    ) -> mx.array:
        idx_list: list[list[int]] = [[] for _ in range(4)]
        weight_list: list[list[float]] = [[] for _ in range(4)]

        for _, h, w in grid:
            h_idxs = mx.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = mx.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.astype(mx.int32)
            w_idxs_floor = w_idxs.astype(mx.int32)
            h_idxs_ceil = mx.minimum(h_idxs_floor + 1, self.num_grid_per_side - 1)
            w_idxs_ceil = mx.minimum(w_idxs_floor + 1, self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor.astype(mx.float32)
            dw = w_idxs - w_idxs_floor.astype(mx.float32)

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = mx.array(idx_list, dtype=mx.int32)
        weight_tensor = mx.array(weight_list, dtype=self.pos_embed.weight.dtype)

        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        split_sizes = [h * w for _, h, w in grid]
        if len(split_sizes) > 1:
            split_indices = []
            total = 0
            for size in split_sizes[:-1]:
                total += size
                split_indices.append(total)
            patch_pos_embeds_split = mx.split(patch_pos_embeds, split_indices, axis=0)
        else:
            patch_pos_embeds_split = [patch_pos_embeds]

        merge_size = self.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, (t, h, w) in zip(patch_pos_embeds_split, grid):
            feature_dim = pos_embed.shape[-1]
            pos_embed = mx.tile(pos_embed, (t, 1))
            pos_embed = pos_embed.reshape(t, h, w, feature_dim)
            pos_embed = (
                pos_embed.reshape(
                    t,
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                    feature_dim,
                )
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(-1, feature_dim)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return mx.concatenate(patch_pos_embeds_permute)

    def __call__(
        self, pixel_values: mx.array, grid_thw
    ) -> tuple[mx.array, list[tuple[int, mx.array]]]:
        grid = _normalize_grid(grid_thw)

        hidden_states = self.patch_embed(
            pixel_values.astype(self.patch_embed.proj.weight.dtype)
        )
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid)
        rotary_pos_emb = self.rot_pos_emb(grid)

        split_indices = []
        total = 0
        for t, h, w in grid:
            for _ in range(t):
                total += h * w
                split_indices.append(total)
        split_indices = split_indices[:-1]

        deepstack_features: list[tuple[int, mx.array]] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, split_indices, rotary_pos_emb)
            if layer_num in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ]
                # The reference injects list element i after decoder layer i.
                deepstack_features.append(
                    (len(deepstack_features), merger(hidden_states))
                )

        return self.merger(hidden_states), deepstack_features
