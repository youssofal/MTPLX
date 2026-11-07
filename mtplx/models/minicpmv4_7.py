"""MiniCPM-V 4.7 35B-A3B autoregressive and prepared-vision adapter."""

from mlx_vlm.models.minicpmv4_7 import Model as UpstreamModel
from mlx_vlm.models.minicpmv4_7 import ModelConfig as ModelConfig
import mlx.core as mx


class Model(UpstreamModel):
    def __init__(self, config):
        text = config.text_config
        vision = config.vision_config
        signature = (
            config.model_type,
            text.model_type,
            text.num_hidden_layers,
            text.hidden_size,
            text.num_experts,
            text.num_experts_per_tok,
            vision.hidden_size,
            vision.num_hidden_layers,
            config.mrope_mode,
        )
        expected = (
            "minicpmv4_7",
            "qwen3_5_moe_text",
            40,
            2048,
            256,
            8,
            1152,
            27,
            "canvas",
        )
        if signature != expected:
            raise ValueError(f"Unqualified MiniCPM architecture: {signature}")
        super().__init__(config)

    def make_cache(self):
        armed = getattr(self, "_armed_vision", None)
        self.language_model._position_ids = armed[1].position_ids if armed else None
        self.language_model._rope_deltas = armed[1].rope_deltas if armed else None
        return self.language_model.make_cache()

    def __call__(
        self,
        input_ids,
        cache=None,
        return_hidden=False,
        emit_logits=True,
        logits_keep=None,
        input_embeddings=None,
        **kwargs,
    ):
        armed = getattr(self, "_armed_vision", None)
        if armed:
            offset = cache[self.language_model.model.fa_idx].offset if cache else 0
            width = armed[0].shape[1]
            if offset < width:
                end = offset + input_ids.shape[1]
                if (
                    end > width
                    or not mx.array_equal(input_ids, armed[0][:, offset:end]).item()
                ):
                    raise ValueError(
                        "Vision prefill IDs must match the prepared prompt"
                    )
                kwargs["inputs_embeds"] = armed[1].inputs_embeds[:, offset:end]
                kwargs["position_ids"] = armed[1].position_ids[:, :, offset:end]
        if input_embeddings is not None:
            kwargs["inputs_embeds"] = input_embeddings
        out = self.language_model(
            input_ids, cache=cache, return_hidden=True, skip_logits=True, **kwargs
        )
        hidden = out.hidden_states[-1]
        logits = None
        if emit_logits:
            rows = hidden[:, -logits_keep:] if logits_keep else hidden
            logits = self.language_model.lm_head(rows)
        return (logits, hidden) if return_hidden else logits

    def prepare_vision(self, inputs):
        ids = mx.array(inputs["input_ids"])
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Prepared MiniCPM vision requires one prompt per request")
        features = self.get_input_embeddings(
            ids,
            pixel_values=inputs.get("pixel_values"),
            tgt_sizes=inputs.get("tgt_sizes"),
            downsample_mode=inputs.get("downsample_mode", "16x"),
        )
        mx.eval(features.inputs_embeds, features.position_ids, features.rope_deltas)
        object.__setattr__(self, "_armed_vision", (ids, features))
        return ids[0].tolist()

    def clear_vision(self):
        object.__setattr__(self, "_armed_vision", None)
        self.language_model._position_ids = None
        self.language_model._rope_deltas = None
