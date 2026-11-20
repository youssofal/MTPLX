"""Synthetic dense Qwen3.5 model and prompt for the image position tests.

A four-layer ``qwen3_5`` text model with random weights (two GatedDeltaNet
layers, two full-attention layers, head_dim 64, 16 rotary dims split
[3, 3, 2] between t, h and w) and one prompt with a 2 x 3 image in it. No
pack is loaded.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mtplx.dense_mrope import DenseMRopeState
from mtplx.vision.mrope import build_mrope_positions

SECTION = [3, 3, 2]
ROPE_THETA = 10000.0
ROTARY_DIMS = 16  # head_dim 64 * partial_rotary_factor 0.25
PAD = 99
VIDEO_PAD = 98
GRID = (1, 4, 6)  # merge 2 -> a 2 x 3 block of image tokens
PROMPT = [5, 6, 7, PAD, PAD, PAD, PAD, PAD, PAD, 8, 9, 10, 11]
DELTA = -3  # the 6 image tokens advance the position by max(1, 2, 3) = 3
# ``text_args`` overrides for the MoE sibling (the 35B-A3B's layout): routed
# and shared experts in every layer. 32 experts, top 4, as tests/a3b_tiny_synth
# has them: with 8 experts MLX picks another expert kernel on short inputs.
MOE_FIELDS = {
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
}


def rope_parameters() -> dict:
    return {
        "type": "default",
        "mrope_section": list(SECTION),
        "mrope_interleaved": True,
        "rope_theta": ROPE_THETA,
        "partial_rotary_factor": 0.25,
    }


def pack_config(**overrides) -> dict:
    """config.json of a vision-capable dense pack, as runtime.load sees it."""
    config = {
        "model_type": "qwen3_5",
        "vision_config": {"spatial_merge_size": 2},
        "text_config": {
            "model_type": "qwen3_5_text",
            "rope_parameters": rope_parameters(),
        },
    }
    config.update(overrides)
    return config


def text_args(*, tie: bool = True, **overrides):
    """``overrides`` replace fields, such as the GDN key head size."""
    from mlx_lm.models.qwen3_5 import TextModelArgs

    fields = dict(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=tie,
        full_attention_interval=2,
        rope_parameters=rope_parameters(),
    )
    fields.update(overrides)
    return TextModelArgs(**fields)


def text_model(seed: int = 7, *, tie: bool = True, **overrides):
    from mlx_lm.models.qwen3_5 import TextModel

    mx.random.seed(seed)
    model = TextModel(text_args(tie=tie, **overrides))
    mx.eval(model.parameters())
    return model


def model_with_draft_head(tmp_path, seed: int = 7, *, tie: bool = True, **overrides):
    """The same model with a one-layer MTP head, injected the product way.

    ``tie=False`` gives the model its own random LM head, so greedy decoding
    produces varied tokens instead of echoing the last one.
    """
    from mlx_lm.models.qwen3_5 import DecoderLayer

    from mtplx.mtp_patch import inject_mtp_support

    model = text_model(seed, tie=tie, **overrides)
    args = model.args
    donor = DecoderLayer(args, layer_idx=args.full_attention_interval - 1)
    tensors = {
        "mtp.fc.weight": mx.random.normal((args.hidden_size, args.hidden_size * 2))
        * 0.02,
        "mtp.norm.weight": mx.ones((args.hidden_size,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((args.hidden_size,)),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((args.hidden_size,)),
    }
    for path, value in tree_flatten(donor.parameters()):
        tensors[f"mtp.layers.0.{path}"] = value
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), tensors)
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mtp_num_hidden_layers": 1,
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "vocab_size": args.vocab_size,
        "linear_num_value_heads": args.linear_num_value_heads,
        "linear_num_key_heads": args.linear_num_key_heads,
        "linear_key_head_dim": args.linear_key_head_dim,
        "linear_value_head_dim": args.linear_value_head_dim,
        "linear_conv_kernel_dim": args.linear_conv_kernel_dim,
        "tie_word_embeddings": bool(tie),
        "full_attention_interval": args.full_attention_interval,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert inject_mtp_support(model, tmp_path, config) is True
    return model


def position_state(prompt=PROMPT, grids=(GRID,)) -> DenseMRopeState:
    """The request state the serve layer builds for ``prompt``."""
    table, delta = build_mrope_positions(
        list(prompt), image_token_id=PAD, image_grids=list(grids), spatial_merge_size=2
    )
    ids = np.asarray(list(prompt))
    return DenseMRopeState(table, delta, pad_positions=np.flatnonzero(ids == PAD))


def full_attention(model):
    return [layer.self_attn for layer in model.model.layers if not layer.is_linear]


class SpyRope:
    """The stock rope with a record of (rows, offset) per call."""

    def __init__(self, rope):
        self.rope = rope
        self.dims, self.base = rope.dims, rope.base
        self.traditional, self.scale = rope.traditional, rope.scale
        self.calls: list[tuple[int, object]] = []

    def __call__(self, x, offset=0):
        self.calls.append((int(x.shape[-2]), offset))
        return self.rope(x, offset=offset)


def spy_on(attn) -> SpyRope:
    """Record every stock rope call an installed adapter makes on ``attn``."""
    spy = SpyRope(attn.rope.inner)
    attn.rope.inner = spy
    return spy
