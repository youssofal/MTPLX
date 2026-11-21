"""The native checkpoint contract. Importing this module does not initialize a GPU."""

from dataclasses import dataclass


def draft_keys(layers):
    keys = {"norm.weight", "fuse_weight", "mask_embed"}
    keys.update(
        "markov_head." + name
        for name in (
            "dflare_norm.weight",
            "hidden_out.weight",
            "hidden_out.bias",
            "markov_w1.weight",
            "rnn_proj.weight",
            "rnn_proj.bias",
        )
    )
    names = (
        "input_layernorm",
        "post_attention_layernorm",
        "dflare_norm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "self_attn.k_norm_ctx",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.k_proj_ctx",
        "self_attn.v_proj_ctx",
        "self_attn.o_proj",
    )
    keys.update(
        f"layers.{layer}.{name}.weight" for layer in range(layers) for name in names
    )
    return keys


def layer_visits(config: dict) -> tuple[int, ...]:
    loop = config["loop_config"]
    start, end, passes = loop["core_start"], loop["core_end"], loop["num_passes"]
    layers = config["num_hidden_layers"]
    if loop.get("is_varloop") or not 0 < start < end < layers or passes < 1:
        raise ValueError("Unsupported Inkling loop configuration")
    return (*range(start), *(tuple(range(start, end)) * passes), *range(end, layers))


@dataclass(frozen=True)
class CacheFootprint:
    full_bytes_per_token: int
    sliding_bytes_per_token: int
    window: int
    tail_bytes: int

    def bounded_bytes(self, tokens: int, chunk: int = 1) -> int:
        rows = min(max(0, tokens), self.window + max(0, chunk - 1))
        return self.tail_bytes + rows * self.sliding_bytes_per_token


def cache_footprint(config: dict, *, draft: bool = False) -> CacheFootprint:
    """Each repeat owns its KV state. Only the global visits grow with context."""
    visits = range(config["num_hidden_layers"]) if draft else layer_visits(config)
    local = set(config["local_layer_ids"])
    sliding = sum(layer in local for layer in visits)
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    row_bytes = 2 * kv_width * 2  # Key and value each store BF16.
    if draft:
        tail = (
            2 * config["hidden_size"]
        )  # Last committed feature for the block drafter.
    else:
        tail_channels = 2 * kv_width + 2 * config["hidden_size"]
        tail = len(visits) * (config["sconv_kernel_size"] - 1) * tail_channels * 4
    return CacheFootprint(
        (len(visits) - sliding) * row_bytes,
        sliding * row_bytes,
        config["sliding_window_size"],
        tail,
    )
