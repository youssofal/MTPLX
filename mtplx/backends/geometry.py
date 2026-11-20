"""The sizes of a hybrid trunk, read once from its config.

Code that would otherwise compare a checkpoint with Flash-Next's numbers
(2,560 wide, 48 layers, 512 experts, one n-gram layer) asks a ``Geometry``
instead; the census and the forge do.  It holds for both hybrid families
(with and without hyper-connection streams), so a routed-expert model
without n-gram rows and with ordinary attention is described by the same
value.

Like ``signature``, this is data only and imports no model module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import signature
from .signature import text_section

RECURRENT = "linear_attention"
ATTENTION = "full_attention"

#: What the hyper-connection trunk's model class assumes for a size its config
#: leaves out (``mtplx.models.qwen4_exp.TextArgs``; a test holds the two
#: equal).  They are Flash-Next's, except that n-gram rows are absent unless
#: a config lists their layers.
_HYPER_TRUNK_DEFAULTS: dict[str, Any] = {
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 248_320,
    "full_attention_interval": 4,
    "linear_num_value_heads": 48,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "hc_count": 4,
    "hc_lowrank": 320,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 20_000_000,
}


@dataclass(frozen=True)
class Recurrent:
    """A gated-delta layer's heads and the width of its convolution input."""

    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int

    @property
    def key_width(self) -> int:
        return self.key_heads * self.key_dim

    @property
    def value_width(self) -> int:
        return self.value_heads * self.value_dim

    @property
    def conv_width(self) -> int:
        """Query, key and value rows side by side."""

        return 2 * self.key_width + self.value_width


@dataclass(frozen=True)
class Experts:
    """Routed experts: how many, how many a token takes, and their widths."""

    count: int
    per_token: int
    width: int
    shared_width: int


@dataclass(frozen=True)
class SelectedRows:
    """Attention that reads the rows an indexer selects instead of all of them."""

    budget: int
    ratio: int
    heads: int
    kv_heads: int
    head_dim: int

    @property
    def blocks(self) -> int:
        """Key blocks a query row keeps."""

        return self.budget // self.ratio


@dataclass(frozen=True)
class NgramRows:
    """Hashed n-gram rows added to the stream at some layers."""

    layers: tuple[int, ...]  # zero-based
    size: int
    heads: int
    rows_base: int
    dim: int


@dataclass(frozen=True)
class DraftHead:
    """The MTP head a config declares: how many decoder layers it runs."""

    layers: int = 1


@dataclass(frozen=True)
class Geometry:
    """One trunk's sizes.  ``None`` means the block is absent."""

    width: int
    layer_kinds: tuple[str, ...]
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    recurrent: Recurrent
    #: Hyper-connection streams and the rank of their mix; one stream is an
    #: ordinary residual.
    streams: int = 1
    stream_rank: int = 0
    experts: Experts | None = None
    #: Width of the dense MLP of a trunk without routed experts.
    mlp_width: int | None = None
    selected_rows: SelectedRows | None = None
    ngram: NgramRows | None = None
    draft_head: DraftHead = DraftHead()
    #: The pack's 16-bit float type, as its config names it: ``bfloat16`` for every published trunk, ``float16``
    #: for an FP16 sibling (M1 and M2, which have no native bfloat16).
    float_type: str = "bfloat16"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Geometry":
        """The geometry the loader builds from ``config``.

        A hyper-connection trunk takes its model class's defaults for a size
        the config leaves out; any other family must state its own
        (``KeyError`` names the missing one).
        """

        text = text_section(config)
        float_type = _float_type(config, text)
        found = signature.read(config)
        if found is not None and found.family == signature.QWEN4:
            text = _with_hyper_trunk_defaults(text)
        layers = int(text["num_hidden_layers"])
        return cls(
            width=int(text["hidden_size"]),
            layer_kinds=_layer_kinds(text, layers),
            heads=int(text["num_attention_heads"]),
            kv_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            vocab=int(text["vocab_size"]),
            recurrent=Recurrent(
                key_heads=int(text["linear_num_key_heads"]),
                value_heads=int(text["linear_num_value_heads"]),
                key_dim=int(text["linear_key_head_dim"]),
                value_dim=int(text["linear_value_head_dim"]),
                conv_kernel=int(text.get("linear_conv_kernel_dim", 4)),
            ),
            streams=int(text.get("hc_count") or 1),
            stream_rank=int(text.get("hc_lowrank") or 0),
            experts=_experts(text),
            mlp_width=None if text.get("num_experts") else _optional_int(text.get("intermediate_size")),
            selected_rows=_selected_rows(text),
            ngram=_ngram(text),
            draft_head=_draft_head(text),
            float_type=float_type,
        )

    @property
    def layers(self) -> int:
        return len(self.layer_kinds)

    @property
    def recurrent_layers(self) -> tuple[int, ...]:
        return tuple(index for index, kind in enumerate(self.layer_kinds) if kind == RECURRENT)

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(index for index, kind in enumerate(self.layer_kinds) if kind != RECURRENT)

    @property
    def stream_width(self) -> int:
        """The widened residual every block reads: all streams side by side."""

        return self.streams * self.width

    @property
    def attention_state_bytes_per_token(self) -> int:
        """Keys and values of every attention layer, in bfloat16."""

        return len(self.attention_layers) * 2 * self.kv_heads * self.head_dim * 2

    def sizes(self) -> dict[str, Any]:
        """Every size under one flat name.

        An absent block's sizes are None.
        """

        recurrent, experts, selected = self.recurrent, self.experts, self.selected_rows
        return {
            "width": self.width,
            "layers": self.layers,
            "heads": self.heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "vocab": self.vocab,
            "streams": self.streams,
            "stream_rank": self.stream_rank,
            "key_heads": recurrent.key_heads,
            "value_heads": recurrent.value_heads,
            "key_head_dim": recurrent.key_dim,
            "value_head_dim": recurrent.value_dim,
            "conv_kernel": recurrent.conv_kernel,
            "conv_width": recurrent.conv_width,
            "key_width": recurrent.key_width,
            "experts": experts.count if experts else None,
            "experts_per_token": experts.per_token if experts else None,
            "expert_width": experts.width if experts else None,
            "shared_expert_width": experts.shared_width if experts else None,
            "mlp_width": self.mlp_width,
            "selected_blocks": selected.blocks if selected else None,
            "selected_budget": selected.budget if selected else None,
            "selected_ratio": selected.ratio if selected else None,
            "float_type": self.float_type,
        }

    def describe(self) -> str:
        """One line a startup log or a census can print."""

        parts = [
            f"{self.width} wide",
            f"{self.layers} layers ({len(self.recurrent_layers)} recurrent, {len(self.attention_layers)} attention)",
            f"{self.streams} streams" if self.streams > 1 else "ordinary residual",
        ]
        if self.experts is not None:
            parts.append(f"{self.experts.count} experts, {self.experts.per_token} a token")
        else:
            parts.append(f"dense MLP {self.mlp_width}")
        parts.append("selected-row attention" if self.selected_rows else "ordinary attention")
        parts.append(f"n-gram rows at layer {', '.join(map(str, self.ngram.layers))}" if self.ngram else "no n-gram rows")
        parts.append(f"draft head of {self.draft_head.layers} layer{'s' if self.draft_head.layers != 1 else ''}")
        if self.float_type != "bfloat16":
            parts.append(f"in {self.float_type}")
        return ", ".join(parts)


def _float_type(config: Mapping[str, Any], text: Mapping[str, Any]) -> str:
    """The float type a config names (``dtype`` or ``torch_dtype``, in its text section or the model's);
    bfloat16 when it names none."""

    for section in (text, config):
        for key in ("dtype", "torch_dtype"):
            value = section.get(key) if isinstance(section, Mapping) else None
            if isinstance(value, str) and value:
                return value
    return "bfloat16"


def _with_hyper_trunk_defaults(text: Mapping[str, Any]) -> dict[str, Any]:
    return {**_HYPER_TRUNK_DEFAULTS, **signature.explicit_blocks(text)}


def _layer_kinds(text: Mapping[str, Any], layers: int) -> tuple[str, ...]:
    declared = text.get("layer_types")
    if isinstance(declared, list) and declared:
        return tuple(str(kind) for kind in declared)
    interval = int(text.get("full_attention_interval") or 4)
    return tuple(RECURRENT if (index + 1) % interval else ATTENTION for index in range(layers))


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _experts(text: Mapping[str, Any]) -> Experts | None:
    count = _optional_int(text.get("num_experts"))
    if count is None:
        return None
    width = int(text["moe_intermediate_size"])
    return Experts(
        count=count,
        per_token=int(text["num_experts_per_tok"]),
        width=width,
        shared_width=int(text.get("shared_expert_intermediate_size") or width),
    )


def _selected_rows(text: Mapping[str, Any]) -> SelectedRows | None:
    heads = _optional_int(text.get("indexer_n_heads"))
    if heads is None:
        return None
    return SelectedRows(
        budget=int(text["indexer_budget"]),
        ratio=int(text["indexer_compress_ratio"]),
        heads=heads,
        kv_heads=int(text.get("indexer_kv_heads") or 1),
        head_dim=int(text["indexer_head_dim"]),
    )


def _ngram(text: Mapping[str, Any]) -> NgramRows | None:
    one_based = text.get("ple_layer_ids")
    if not one_based:
        return None
    return NgramRows(
        layers=tuple(sorted(int(layer) - 1 for layer in set(one_based))),
        size=int(text.get("ngram_size", 3)),
        heads=int(text.get("heads_per_ngram", 8)),
        rows_base=int(text.get("ngram_vocab_size_base", 20_000_000)),
        dim=int(text.get("ple_embed_dim") or text["hidden_size"]),
    )


def _draft_head(text: Mapping[str, Any]) -> DraftHead:
    declared = text.get("mtp")
    layers = declared.get("num_hidden_layers") if isinstance(declared, Mapping) else None
    return DraftHead(layers=_optional_int(layers) or _optional_int(text.get("mtp_num_hidden_layers")) or 1)


#: Qwen 3.8 Flash-Next, the geometry every fused kernel was first written for.
FLASH_NEXT = Geometry(
    width=2560,
    layer_kinds=tuple(RECURRENT if (index + 1) % 4 else ATTENTION for index in range(48)),
    heads=24,
    kv_heads=2,
    head_dim=256,
    vocab=248_320,
    recurrent=Recurrent(key_heads=16, value_heads=48, key_dim=128, value_dim=128, conv_kernel=4),
    streams=4,
    stream_rank=320,
    experts=Experts(count=512, per_token=10, width=640, shared_width=640),
    selected_rows=SelectedRows(budget=2048, ratio=4, heads=4, kv_heads=1, head_dim=128),
    ngram=NgramRows(layers=(1,), size=3, heads=8, rows_base=20_000_000, dim=2560),
)

__all__ = [
    "ATTENTION",
    "DraftHead",
    "Experts",
    "FLASH_NEXT",
    "Geometry",
    "NgramRows",
    "RECURRENT",
    "Recurrent",
    "SelectedRows",
]
