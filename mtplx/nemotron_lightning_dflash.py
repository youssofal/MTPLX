"""Construction-time DFlash capture route for Nemotron 3.5 Lightning."""

from __future__ import annotations

from types import MethodType

from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def install_nemotron_lightning_capture(target, capture_layers: list[int]) -> None:
    """Install a dedicated verifier forward that returns fixed residual taps."""
    if getattr(target.args, "model_type", None) != "nemotron_h":
        raise ValueError("Nemotron Lightning DFlash requires model_type=nemotron_h")
    if not hasattr(target, "backbone"):
        raise ValueError("Nemotron Lightning target is missing its backbone")
    layer_count = len(target.backbone.layers)
    captures = tuple(int(layer) for layer in capture_layers)
    if captures != tuple(sorted(set(captures))):
        raise ValueError(f"DFlash capture layers must be sorted and unique: {captures}")
    if not captures or captures[0] < 0 or captures[-1] >= layer_count:
        raise ValueError(
            f"DFlash capture layers {captures} are outside Lightning's {layer_count} layers"
        )

    # Segments make capture positions construction-time invariants. The inner
    # layer loop has no membership check or metadata validation.
    segments: tuple[tuple[int, int], ...] = tuple(
        (0 if index == 0 else captures[index - 1] + 1, capture + 1)
        for index, capture in enumerate(captures)
    )
    expected_start = 0
    for start, end in segments:
        if start != expected_start or end <= start:
            raise ValueError("invalid Lightning DFlash capture segments")
        expected_start = end

    def forward_capture(self, inputs, cache):
        backbone = self.backbone
        hidden = backbone.embeddings(inputs)
        if cache is None:
            cache = [None] * sum(
                layer.block_type in {"M", "*"} for layer in backbone.layers
            )
        attn_mask = create_attention_mask(hidden, cache[backbone.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[backbone.ssm_idx])
        cache_index = 0
        layer_index = 0
        taps = {}
        for start, end in segments:
            for layer_index in range(start, end):
                layer = backbone.layers[layer_index]
                if layer.block_type in {"M", "*"}:
                    layer_cache = cache[cache_index]
                    cache_index += 1
                else:
                    layer_cache = None
                mask = attn_mask if layer.block_type == "*" else ssm_mask
                hidden = layer(hidden, mask=mask, cache=layer_cache)
            capture = end - 1
            taps[capture] = hidden[0]
            layer_index = end
        for layer_index in range(layer_index, len(backbone.layers)):
            layer = backbone.layers[layer_index]
            if layer.block_type in {"M", "*"}:
                layer_cache = cache[cache_index]
                cache_index += 1
            else:
                layer_cache = None
            mask = attn_mask if layer.block_type == "*" else ssm_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        return self.lm_head(backbone.norm_f(hidden)), taps

    target.dflash_forward_capture = MethodType(forward_capture, target)
