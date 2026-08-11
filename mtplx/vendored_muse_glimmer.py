# Copyright © 2026 MTPLX contributors.
"""Vendored multimodal wrapper for Muse-Glimmer (``model_type: muse_glimmer``).

Mirrors mlx-lm's ``qwen3_vl`` treatment of Qwen3.6-27B-VL: a thin wrapper that
builds the text backbone from ``text_config``, drops the vision tower's weights
at load, and accepts spliced image ``input_embeddings`` (the vision encoder runs
externally via mlx-vlm; MTPLX's runtime splices its output through the
``input_embeddings`` path). Registered as ``mlx_lm.models.muse_glimmer`` by
``muse_glimmer_patch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.models.base import BaseModelArgs

from . import vendored_muse_glimmer_text as muse_glimmer_text


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = muse_glimmer_text.ModelArgs.from_dict(args.text_config)
        self.language_model = muse_glimmer_text.Model(text_args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        # Drop the vision stack (perception encoder + projector); it is served
        # externally via mlx-vlm and spliced as input_embeddings.
        weights = tree_unflatten(list(weights.items()))
        if isinstance(weights, dict):
            model = weights.get("model")
            if isinstance(model, dict):
                for k in ("vision_tower", "vision_adapter", "vision_projection"):
                    model.pop(k, None)
        weights = dict(tree_flatten(weights))

        # Remap the multimodal checkpoint's language-model weights under the
        # wrapper's ``language_model.`` prefix (matching qwen3_vl).
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model.") :]
            elif key == "lm_head.weight":
                key = "language_model.lm_head.weight"
            elif key.startswith("model.vision"):
                continue
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers
