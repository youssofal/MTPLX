"""Which family a checkpoint belongs to, read from what its config contains.

A new generation arrives under a new ``model_type`` string before any code
knows the string.  What it is made of is in the config all the same: a trunk
with hyper-connection streams carries ``hc_count``; a trunk with recurrent
(gated delta) layers carries the ``linear_*`` sizes; routed experts, n-gram
rows and selected-row attention each bring their own keys.  This module reads
those, so a checkpoint is recognised by its blocks and a name is only a name.

Recognition is not permission.  A config can match a family and still carry a
key no model class of ours reads, or a value our code does not implement; run
anyway, such a key is dropped in silence and the output is wrong with nothing
to show for it.  ``Signature.unread`` lists both, and a config that is not
under one of the family's own names is built only while that list is empty.

Pure data and dictionaries: importing this never imports MLX or a model
module, so the server may ask it before anything else is set up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: Hyper-connection trunk: Flash-Next and what follows it (``mtplx.models.qwen4_exp``).
QWEN4 = "qwen4_exp"
#: Hybrid trunk with ordinary residuals and a dense MLP (mlx-lm ``qwen3_5``).
QWEN3_5 = "qwen3_5"
#: The same trunk with routed experts (mlx-lm ``qwen3_5_moe``).
QWEN3_5_MOE = "qwen3_5_moe"

HYPER_CONNECTIONS = "hyper_connections"
RECURRENT_LAYERS = "recurrent_layers"
SELECTED_ROW_ATTENTION = "selected_row_attention"
NGRAM_ROWS = "ngram_rows"
ROUTED_EXPERTS = "routed_experts"

_RECURRENT_KEYS = (
    "linear_num_value_heads",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
)

#: Keys that describe training, storage or bookkeeping and change no forward pass.
_INERT_KEYS = frozenset(
    {
        "architectures",
        "attention_dropout",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "initializer_range",
        "output_router_logits",
        "pad_token_id",
        "router_aux_loss_coef",
        "split_ngram_parts",
        "torch_dtype",
        "transformers_version",
        "use_cache",
    }
)

_SHARED_READ_KEYS = frozenset(
    {
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "vocab_size",
        "max_position_embeddings",
        "tie_word_embeddings",
        "attention_bias",
        "layer_types",
        "full_attention_interval",
        "linear_num_value_heads",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "norm_topk_prob",
        "decoder_sparse_step",
        "rope_parameters",
        "partial_rotary_factor",
        "rope_theta",
    }
)


@dataclass(frozen=True)
class _Schema:
    """What one family's model class takes from a text config."""

    #: ``model_type`` strings the family already goes by.
    names: frozenset[str]
    #: Keys the class reads.  A test holds this equal to the class's own fields.
    read: frozenset[str]
    #: Keys the class does not read because it implements these values of them.
    implemented: Mapping[str, tuple[Any, ...]]


_SCHEMAS: dict[str, _Schema] = {
    QWEN4: _Schema(
        names=frozenset({"qwen4_exp", "qwen4_exp_text"}),
        read=_SHARED_READ_KEYS
        | {
            "output_gate_type",
            "hc_count",
            "hc_lowrank",
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
            "ple_layer_ids",
            "ple_embed_dim",
            "ple_conv_kernel_size",
            "ngram_size",
            "heads_per_ngram",
            "ngram_vocab_size_base",
            "make_ngram_vocab_size_divisible_by",
            "seed",
            "eos_token_id",
            "ngram_sidecar",
            "mrope_section",
            "mrope_interleaved",
            "mtp",
            "mtp_num_hidden_layers",
        },
        implemented={
            "decoder_sparse_step": (1,),
            "hidden_act": ("silu",),
            "mamba_ssm_dtype": ("float32",),
            "mtp_use_dedicated_embeddings": (False, None),
            "output_gate_type": ("sigmoid", "swish"),
        },
    ),
}
_SCHEMAS[QWEN3_5] = _Schema(
    names=frozenset({"qwen3_5", "qwen3_5_text"}),
    read=_SHARED_READ_KEYS | {"rope_scaling", "mtp_num_hidden_layers"},
    implemented={
        "attn_output_gate": (True,),
        "decoder_sparse_step": (1,),
        "hidden_act": ("silu",),
        "mamba_ssm_dtype": ("float32",),
        "mtp_use_dedicated_embeddings": (False, None),
        "output_gate_type": ("swish",),
    },
)
_SCHEMAS[QWEN3_5_MOE] = _Schema(
    names=frozenset({"qwen3_5_moe", "qwen3_5_moe_text"}),
    read=_SCHEMAS[QWEN3_5].read,
    implemented=_SCHEMAS[QWEN3_5].implemented,
)

#: Blocks a family has whether or not its config spells their sizes out.
_IMPLIED_BLOCKS: dict[str, frozenset[str]] = {
    QWEN4: frozenset({RECURRENT_LAYERS, HYPER_CONNECTIONS}),
    QWEN3_5: frozenset({RECURRENT_LAYERS}),
    QWEN3_5_MOE: frozenset({RECURRENT_LAYERS, ROUTED_EXPERTS}),
}

#: Sizes a hyper-connection trunk cannot be built without.  The model class
#: has a default for each (Flash-Next's), which is right for Flash-Next's own
#: packs and a different model for anything else.
_SIZES_EVERY_TRUNK_STATES = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
)
_SIZES_A_BLOCK_STATES = {
    ROUTED_EXPERTS: ("num_experts_per_tok", "moe_intermediate_size"),
    SELECTED_ROW_ATTENTION: ("indexer_budget", "indexer_compress_ratio", "indexer_head_dim"),
    NGRAM_ROWS: ("ngram_vocab_size_base", "heads_per_ngram", "ngram_size"),
}

#: The draft head's own sub-config (``text_config.mtp``), under the same rules.
_MTP_READ_KEYS = frozenset({"num_hidden_layers", "layer_types"})
_MTP_INERT_KEYS = frozenset({"hybrid", "rope_theta"})
_MTP_IMPLEMENTED = {"mtp_use_hidden_state_from_layer": (None,)}


@dataclass(frozen=True)
class Signature:
    """A config's family, the blocks its trunk is made of, and what nothing reads."""

    family: str
    blocks: frozenset[str]
    #: ``model_type`` is one of the family's own strings.
    named: bool
    #: Keys no model class reads and values no code implements, as sentences.
    unread: tuple[str, ...] = ()

    @property
    def servable(self) -> bool:
        """A named config is the family's own; an unnamed one must be read in full."""

        return self.named or not self.unread

    def has(self, block: str) -> bool:
        return block in self.blocks


def text_section(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The text model's keys: ``text_config`` when the config nests one."""

    config = config or {}
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


def read(config: Mapping[str, Any] | None) -> Signature | None:
    """The signature of ``config``, or None when it is not one of these families.

    A config under one of a family's own names is that family whatever else
    it carries; any other is the family its blocks make it.
    """

    text = text_section(config)
    named = _named_family(config)
    family = named or _family_of_blocks(text)
    if family is None:
        return None
    return Signature(
        family=family,
        blocks=_blocks(text) | _IMPLIED_BLOCKS[family],
        named=named is not None,
        unread=tuple(_unread(text, family)),
    )


def family(config: Mapping[str, Any] | None) -> str | None:
    """The family a loader may build ``config`` as, or None.

    The family's own names always qualify.  A config under another name
    qualifies when every key in it is read or known inert.
    """

    signature = read(config)
    if signature is None or not signature.servable:
        return None
    return signature.family


def is_qwen4(config: Mapping[str, Any] | None) -> bool:
    """Whether ``config`` loads as the hyper-connection trunk (Flash-Next's family)."""

    return family(config) == QWEN4


def explicit_blocks(text: Mapping[str, Any]) -> dict[str, Any]:
    """A hyper-connection text config with every block it leaves out said to be absent.

    The trunk's model class defaults to Flash-Next's blocks.  Under
    Flash-Next's own names that stands: its packs may leave a default out.
    Under any other name a block the config does not mention is a block the
    model does not have, and Flash-Next's built in its place would be another
    model.  The loader and ``Geometry`` both read a config through this, so
    they cannot disagree about what is there.
    """

    explicit = dict(text)
    name = str(text.get("model_type") or "").lower()
    # A text config that states no name at all is read as it always was.
    if name and name not in _SCHEMAS[QWEN4].names:
        explicit.setdefault("num_experts", 0)
        explicit.setdefault("indexer_n_heads", None)
    return explicit


#: Where a pack written by ``under_own_name`` keeps its publisher's names.
UPSTREAM_NAMES_KEY = "mtplx_upstream_model_type"


def renamed_plain_trunk(config: Mapping[str, Any] | None) -> str | None:
    """The family of a plain trunk published under a name of its own, or None.

    Plain: not the hyper-connection trunk.  Under a name of its own: the
    family is recognised by the config's contents, with nothing unread.
    """

    signature = read(config)
    if signature is None or signature.named or signature.unread or signature.family == QWEN4:
        return None
    return signature.family


def under_own_name(config: Mapping[str, Any]) -> dict[str, Any]:
    """``config`` under the family's own ``model_type`` when it is a renamed plain trunk.

    Every key of such a config is one the family's module reads, so the
    rename loses nothing, and a pack written this way is one that every
    name check in the product, and every released build, already serves.
    The publisher's names are kept under ``UPSTREAM_NAMES_KEY``.  Any other
    config is returned unchanged: a hyper-connection trunk keeps its name,
    because under Flash-Next's a block its config leaves out would be
    Flash-Next's (``explicit_blocks``).
    """

    family = renamed_plain_trunk(config)
    if family is None:
        return dict(config)
    renamed = dict(config, model_type=family)
    renamed[UPSTREAM_NAMES_KEY] = sorted(model_types(config))
    if isinstance(config.get("text_config"), Mapping):
        renamed["text_config"] = dict(config["text_config"], model_type=f"{family}_text")
    return renamed


def model_types(config: Mapping[str, Any] | None) -> set[str]:
    """The lowered ``model_type`` strings of a config and of its text section."""

    config = config or {}
    values = {config.get("model_type"), text_section(config).get("model_type")}
    return {str(value).lower() for value in values if value}


def read_keys(family: str) -> frozenset[str]:
    """The text-config keys ``family``'s model class reads."""

    return _SCHEMAS[family].read


def _named_family(config: Mapping[str, Any] | None) -> str | None:
    names = model_types(config)
    return next((family for family, schema in _SCHEMAS.items() if names & schema.names), None)


def _family_of_blocks(text: Mapping[str, Any]) -> str | None:
    if not all(_positive(text.get(key)) for key in _RECURRENT_KEYS):
        return None
    if _positive(text.get("hc_count")) and _positive(text.get("hc_lowrank")):
        return QWEN4
    if "hc_count" in text or "hc_lowrank" in text:
        return None  # half a hyper-connection declaration is no family of ours
    return QWEN3_5_MOE if _positive(text.get("num_experts")) else QWEN3_5


def _blocks(text: Mapping[str, Any]) -> frozenset[str]:
    present = {RECURRENT_LAYERS}
    if _positive(text.get("hc_count")):
        present.add(HYPER_CONNECTIONS)
    if _positive(text.get("num_experts")):
        present.add(ROUTED_EXPERTS)
    if _positive(text.get("indexer_n_heads")):
        present.add(SELECTED_ROW_ATTENTION)
    if text.get("ple_layer_ids"):
        present.add(NGRAM_ROWS)
    return frozenset(present)


def _positive(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _unread(text: Mapping[str, Any], family: str) -> list[str]:
    schema = _SCHEMAS[family]
    known = schema.read | _INERT_KEYS | frozenset(schema.implemented)
    problems = [f"key {key!r} is not read by any model class" for key in sorted(set(text) - known)]
    problems += _unimplemented(text, schema.implemented, prefix="")
    if family == QWEN4:
        problems += _unstated_sizes(text)
    else:
        problems += _layer_pattern_problems(text)
    head = text.get("mtp")
    if isinstance(head, Mapping):
        problems += _head_problems(head)
    return problems


def _unimplemented(section: Mapping[str, Any], implemented: Mapping[str, tuple], *, prefix: str) -> list[str]:
    return [
        f"{prefix}{key} is {section[key]!r}; implemented: {', '.join(repr(value) for value in values)}"
        for key, values in implemented.items()
        if key in section and section[key] not in values
    ]


def _unstated_sizes(text: Mapping[str, Any]) -> list[str]:
    """Sizes the model class would have to assume for a hyper-connection trunk."""

    wanted = list(_SIZES_EVERY_TRUNK_STATES)
    present = _blocks(text)
    for block, sizes in _SIZES_A_BLOCK_STATES.items():
        if block in present:
            wanted += sizes
    if ROUTED_EXPERTS not in present:
        wanted.append("intermediate_size")
    return [
        f"size {key!r} is not stated; the model class would assume Flash-Next's"
        for key in wanted
        if text.get(key) is None
    ]


def _layer_pattern_problems(text: Mapping[str, Any]) -> list[str]:
    """mlx-lm's hybrid trunk derives its layer kinds from the interval alone."""

    kinds, layers = text.get("layer_types"), text.get("num_hidden_layers")
    interval = text.get("full_attention_interval", 4)
    if not isinstance(kinds, list) or not _positive(layers) or not _positive(interval):
        return []
    derived = ["linear_attention" if (index + 1) % interval else "full_attention" for index in range(layers)]
    if kinds == derived:
        return []
    return [f"layer_types is not one attention layer every {interval}; the trunk is built from the interval"]


def _head_problems(head: Mapping[str, Any]) -> list[str]:
    known = _MTP_READ_KEYS | _MTP_INERT_KEYS | frozenset(_MTP_IMPLEMENTED)
    problems = [f"key 'mtp.{key}' is not read by any model class" for key in sorted(set(head) - known)]
    problems += _unimplemented(head, _MTP_IMPLEMENTED, prefix="mtp.")
    kinds = head.get("layer_types")
    if isinstance(kinds, list) and any(kind != "full_attention" for kind in kinds):
        problems.append(f"mtp.layer_types is {kinds!r}; the draft head is built from attention layers")
    return problems


__all__ = [
    "HYPER_CONNECTIONS",
    "NGRAM_ROWS",
    "QWEN3_5",
    "QWEN3_5_MOE",
    "QWEN4",
    "RECURRENT_LAYERS",
    "ROUTED_EXPERTS",
    "SELECTED_ROW_ATTENTION",
    "Signature",
    "UPSTREAM_NAMES_KEY",
    "explicit_blocks",
    "family",
    "is_qwen4",
    "model_types",
    "read",
    "read_keys",
    "renamed_plain_trunk",
    "text_section",
    "under_own_name",
]
