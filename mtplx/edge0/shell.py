"""Edge0 skeleton -> parameters of the existing ``qwen4_exp`` model ("the shell").

Edge0-120B is Qwen3.8-Flash-Next with the routed experts replaced. Everything
else (embeddings, attention, GatedDeltaNet, router, shared expert,
hyper-connections, PLE projections) ships in ``model-skeleton.gguf`` in
llama.cpp's ``qwen4exp`` layout. This module turns that file into the names,
shapes and conventions :class:`mtplx.models.qwen4_exp.Model` holds after its own
``sanitize`` (no further renaming at load), inverting every llama.cpp layout
transform:

* GatedDeltaNet value heads. llama.cpp stores them tiled (``[r, K]``: head
  ``j * K + g``) so ggml can broadcast the ``K`` key heads; Hugging Face, mlx-lm
  and this model group them by key head (``[K, r]``: head ``g * r + j``). The
  order is undone on the value rows of ``attn_qkv`` and the value channels of
  ``ssm_conv1d``, the rows of ``attn_gate``, ``ssm_alpha`` and ``ssm_beta``, the
  entries of ``ssm_a`` and ``ssm_dt.bias``, and the columns of ``ssm_out``.
* ``ssm_a`` is ``-exp(A_log)``: ``A_log = log(-ssm_a)``, snapped back to the
  bfloat16 value it came from when that value reproduces ``ssm_a``.
* The indexer's fused ``index_qk_proj`` is split in the file into ``q_proj``
  and ``k_proj``; they are concatenated back (q rows first).
* Conv kernels are ``[channels, kernel]``; the model wants ``[channels, kernel, 1]``.
* Norm gains that the HF checkpoint stores zero-centred are stored in the file
  as F32 ``1 + w``: exactly the convention the model holds after ``sanitize``
  (``Model._HF_NORM_SHIFT_SUFFIXES``), so they pass through unchanged. The GDN
  output norm is stored one-centred in both.

Precision is kept lossless. BF16, F32 and F16 tensors keep their values (F32 and
F16 are narrowed to bfloat16 only when that is exact). Q8_0 tensors become MLX
8-bit affine tensors with group size 32: ``scale = d``, ``bias = -128 d`` and
code ``q + 128``, which dequantize to exactly ``d * q`` when scales and biases
are float32 (float16 holds them exactly but ``mx.dequantize`` then rounds the
product in float16; bfloat16 cannot hold ``d``). Row permutations keep Q8_0
blocks whole; the ``ssm_out`` column permutation moves whole 32-column blocks
when the value head width is a multiple of 32 and falls back to float32
otherwise.

Sources: the engine's name table ``edge0-strata``
``include/strata/artifact/gguf_reader.hpp`` (``edge0_tensor_alias``), its GDN
forward ``src/core/layer.cpp`` and ``include/strata/kernels/gdn.hpp``, its
geometry reader ``include/strata/core/geometry.hpp``; the model side is
``mtplx/models/qwen4_exp.py`` (``Model.sanitize``, ``GatedDeltaNet``,
``QSAIndexer``, ``PLELayer``) and mlx-lm's ``qwen3_5.GatedDeltaNet``. Every
transform was checked against the public ``Qwen/Qwen3.8-Flash-Next`` checkpoint.

The module imports no MLX at import time; ``convert_shell`` imports it lazily.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from .ggml_quants import unpack_q8_0
from .gguf_lite import GGML_BF16, GGML_F16, GGML_F32, GGML_Q8_0, GGUFFile

ARCH = "qwen4exp"
LM = "language_model.model."
LM_HEAD = "language_model.lm_head.weight"
Q8_GROUP = 32
Q8_BITS = 8
Q8_OFFSET = 128

Q8_MODES = ("affine8", "float32")
SCALE_DTYPES = ("float32", "float16", "bfloat16")

_FLOAT_TYPES = (GGML_F32, GGML_F16, GGML_BF16)
_SUPPORTED_TYPES = _FLOAT_TYPES + (GGML_Q8_0,)


class ShellError(ValueError):
    """Raised when the skeleton does not map one to one onto the model's shell."""


# --------------------------------------------------------------------------- metadata and geometry

_MISSING = object()


def _skeleton(pack) -> GGUFFile:
    sk = getattr(pack, "skeleton", pack)
    if not hasattr(sk, "tensors") or not hasattr(sk, "metadata"):
        raise TypeError(f"expected an Edge0Pack or a GGUFFile, got {type(pack).__name__}")
    return sk


def _meta(sk, key: str, default=_MISSING):
    """``qwen4exp.<key>``; the Edge0 student build names the same keys ``edge0.<key>`` (``gguf_reader.hpp`` ``get``)."""
    for prefix in (ARCH, "edge0"):
        full = f"{prefix}.{key}"
        if full in sk.metadata:
            return sk.metadata[full]
    if default is _MISSING:
        raise ShellError(f"{getattr(sk, 'path', 'skeleton')}: metadata key {ARCH}.{key} is missing")
    return default


def _f32_repr(v) -> float:
    """The shortest decimal that rounds to the stored float32 (GGUF floats are f32: 1e-6 reads 9.99999997e-07)."""
    return float(str(np.float32(v)))


def _num(v):
    f = _f32_repr(v)
    return int(f) if f.is_integer() else f


@dataclass(frozen=True)
class Geometry:
    hidden: int
    layers: int
    vocab: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    linear_layers: tuple[bool, ...]
    k_heads: int  # GDN key heads
    v_heads: int  # GDN value heads
    k_head_dim: int
    v_head_dim: int
    conv_kernel: int
    hc_count: int
    hc_lowrank: int
    idx_heads: int
    idx_kv_heads: int
    idx_head_dim: int
    ple_layers: tuple[int, ...]  # zero-based
    ple_embed_dim: int
    ple_conv_kernel: int

    @property
    def key_dim(self) -> int:
        return self.k_heads * self.k_head_dim

    @property
    def value_dim(self) -> int:
        return self.v_heads * self.v_head_dim

    @property
    def hc_hidden(self) -> int:
        return self.hc_count * self.hidden


def geometry(pack) -> Geometry:
    sk = _skeleton(pack)
    layers = int(_meta(sk, "block_count"))
    rec = _meta(sk, "attention.recurrent_layers", None)
    if rec is None:
        interval = int(_meta(sk, "full_attention_interval"))
        rec = [(i + 1) % interval != 0 for i in range(layers)]
    if len(rec) != layers:
        raise ShellError(f"attention.recurrent_layers has {len(rec)} entries for {layers} layers")
    v_heads = int(_meta(sk, "ssm.time_step_rank"))
    inner = int(_meta(sk, "ssm.inner_size"))
    if inner % v_heads:
        raise ShellError(f"ssm.inner_size {inner} is not a multiple of ssm.time_step_rank {v_heads}")
    k_heads = int(_meta(sk, "ssm.group_count"))
    if v_heads % k_heads:
        raise ShellError(f"GDN value heads {v_heads} are not a multiple of key heads {k_heads}")
    head_dim = int(_meta(sk, "attention.key_length"))
    if int(_meta(sk, "attention.value_length", head_dim)) != head_dim:
        raise ShellError("attention.key_length and attention.value_length differ; the model has one head_dim")
    idx_heads = int(_meta(sk, "attention.indexer.head_count"))
    idx_head_dim = int(_meta(sk, "attention.indexer.key_length"))
    qsa = [i for i in range(layers) if not rec[i]]
    idx_kv_heads = 1
    if qsa:
        k_rows = sk.tensor(f"blk.{qsa[0]}.indexer.k_proj.weight").shape[0]
        if k_rows % idx_head_dim:
            raise ShellError(f"indexer.k_proj has {k_rows} rows, not a multiple of the indexer head dim {idx_head_dim}")
        idx_kv_heads = k_rows // idx_head_dim
    ple_layers = tuple(int(x) for x in _meta(sk, "ple.layers", []))
    ple_embed_dim = int(_meta(sk, "embedding_length", 0))
    if ple_layers:
        ple_embed_dim = sk.tensor(f"blk.{ple_layers[0]}.ple_value.weight").shape[1]
    tok = sk.tensor("token_embd.weight").shape
    return Geometry(
        hidden=int(_meta(sk, "embedding_length")),
        layers=layers,
        vocab=int(tok[0]),
        n_heads=int(_meta(sk, "attention.head_count")),
        n_kv_heads=int(_meta(sk, "attention.head_count_kv")),
        head_dim=head_dim,
        linear_layers=tuple(bool(x) for x in rec),
        k_heads=k_heads,
        v_heads=v_heads,
        k_head_dim=int(_meta(sk, "ssm.state_size")),
        v_head_dim=inner // v_heads,
        conv_kernel=int(_meta(sk, "ssm.conv_kernel")),
        hc_count=int(_meta(sk, "hyper_connection.count")),
        hc_lowrank=int(_meta(sk, "hyper_connection.low_rank")),
        idx_heads=idx_heads,
        idx_kv_heads=idx_kv_heads,
        idx_head_dim=idx_head_dim,
        ple_layers=ple_layers,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel=int(_meta(sk, "ple.conv_kernel", 4)),
    )


# --------------------------------------------------------------------------- config.json


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _ngram_vocab_base(sizes: list[int]) -> int:
    """The base the model's consecutive-prime head layout starts from: the roundest number above the prime before ``sizes[0]``."""
    lo = sizes[0] - 1
    while lo > 1 and not _is_prime(lo):
        lo -= 1
    best, best_zeros = sizes[0], -1
    for v in range(lo + 1, sizes[0] + 1):
        zeros = len(str(v)) - len(str(v).rstrip("0"))
        if zeros > best_zeros:
            best, best_zeros = v, zeros
    # the model rebuilds sizes as consecutive primes from base; check it reproduces the file's
    p, rebuilt = best - 1, []
    for _ in sizes:
        p += 1
        while not _is_prime(p):
            p += 1
        rebuilt.append(p)
    if rebuilt != list(sizes):
        raise ShellError(f"ple.head_vocab_sizes {sizes[:3]}... are not consecutive primes from one base")
    return best


#: config fields that cannot be read from the GGUF: value used and why.
CONFIG_NOT_IN_GGUF = {
    "output_gate_type": ("sigmoid", "GDN output gate; engine include/strata/kernels/gdn.hpp states SIGMOID for this artifact"),
    "norm_topk_prob": (True, "router renormalises the top-k (engine native_router.cu); HF config omits it, TextArgs default"),
    "attention_bias": (False, "no bias tensors in the skeleton"),
    "make_ngram_vocab_size_divisible_by": (128, "only pads the n-gram table rows; 128, 256 and 512 all fit ple-fp8.gguf's rows"),
    "seed": (1234, "only seeds default layer_multipliers, which the converted buffers overwrite"),
    "decoder_sparse_step": (1, "every layer is MoE (ffn_gate_inp in all blocks)"),
    "mrope_interleaved": (True, "arch qwen4exp uses interleaved M-RoPE (engine include/strata/kernels/mrope.hpp)"),
    "ngram_sidecar": (True, "MTPLX packs stream the table from ngram-table.safetensors"),
}


def shell_text_config(pack) -> dict:
    """The ``text_config`` fields :class:`TextArgs` reads, derived from the skeleton's metadata."""
    sk = _skeleton(pack)
    g = geometry(sk)
    eps = _f32_repr(_meta(sk, "attention.layer_norm_rms_epsilon"))
    sections = [int(x) for x in _meta(sk, "rope.dimension_sections", [])]
    while sections and sections[-1] == 0:
        sections.pop()
    rot = int(_meta(sk, "rope.dimension_count"))
    partial = rot / g.head_dim
    theta = _num(_meta(sk, "rope.freq_base"))
    ratios = [int(x) for x in _meta(sk, "attention.compress_ratios", [])]
    nonzero = sorted({r for r in ratios if r})
    if len(nonzero) > 1:
        raise ShellError(f"attention.compress_ratios has several ratios {nonzero}; the model has one")
    interval = int(_meta(sk, "full_attention_interval", 0)) or None
    layer_types = ["linear_attention" if lin else "full_attention" for lin in g.linear_layers]
    if interval and any((i + 1) % interval == 0 for i, lin in enumerate(g.linear_layers) if lin):
        raise ShellError("attention.recurrent_layers disagrees with full_attention_interval")
    if any(r and lin for r, lin in zip(ratios, g.linear_layers)):
        raise ShellError("attention.compress_ratios marks a recurrent layer")
    sizes = [int(x) for x in _meta(sk, "ple.head_vocab_sizes", [])]
    ple_dim = int(_meta(sk, "embedding_length_per_layer_input", 0))
    ngram = int(_meta(sk, "ple.ngram_size", 3))
    heads_per = int(_meta(sk, "ple.heads_per_ngram", 8))
    if g.ple_layers and ple_dim and ple_dim * (ngram - 1) * heads_per != g.ple_embed_dim:
        raise ShellError(
            f"PLE: {ple_dim} per head x {(ngram - 1) * heads_per} heads != ple_value width {g.ple_embed_dim}"
        )
    tc = {
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        "hidden_size": g.hidden,
        "num_hidden_layers": g.layers,
        "num_attention_heads": g.n_heads,
        "num_key_value_heads": g.n_kv_heads,
        "head_dim": g.head_dim,
        "rms_norm_eps": eps,
        "vocab_size": g.vocab,
        "max_position_embeddings": int(_meta(sk, "context_length")),
        "tie_word_embeddings": "output.weight" not in sk.tensors,
        "layer_types": layer_types,
        "full_attention_interval": interval or 4,
        "linear_num_value_heads": g.v_heads,
        "linear_num_key_heads": g.k_heads,
        "linear_key_head_dim": g.k_head_dim,
        "linear_value_head_dim": g.v_head_dim,
        "linear_conv_kernel_dim": g.conv_kernel,
        "num_experts": int(_meta(sk, "expert_count")),
        "num_experts_per_tok": int(_meta(sk, "expert_used_count")),
        "moe_intermediate_size": int(_meta(sk, "expert_feed_forward_length")),
        "shared_expert_intermediate_size": int(_meta(sk, "expert_shared_feed_forward_length")),
        "hc_count": g.hc_count,
        "hc_lowrank": g.hc_lowrank,
        "indexer_n_heads": g.idx_heads,
        "indexer_kv_heads": g.idx_kv_heads,
        "indexer_head_dim": g.idx_head_dim,
        "indexer_budget": int(_meta(sk, "attention.indexer.top_k")),
        "indexer_compress_ratio": nonzero[0] if nonzero else 4,
        "ple_layer_ids": [i + 1 for i in g.ple_layers],  # the HF config is one-indexed
        "ple_embed_dim": g.ple_embed_dim,
        "ple_conv_kernel_size": g.ple_conv_kernel,
        "ngram_size": ngram,
        "heads_per_ngram": heads_per,
        "partial_rotary_factor": partial,
        "rope_theta": theta,
        "rope_parameters": {
            "mrope_interleaved": CONFIG_NOT_IN_GGUF["mrope_interleaved"][0],
            "mrope_section": sections,
            "partial_rotary_factor": partial,
            "rope_theta": theta,
            "rope_type": "default",
        },
    }
    if sizes:
        tc["ngram_vocab_size_base"] = _ngram_vocab_base(sizes)
    eos = _meta(sk, "ple.eos_token_id", None)
    if eos is not None:
        tc["eos_token_id"] = int(eos)  # the id the n-gram shift treats as a boundary
    bos = sk.metadata.get("tokenizer.ggml.bos_token_id")
    if bos is not None:
        tc["bos_token_id"] = int(bos)
    for key, (value, _why) in CONFIG_NOT_IN_GGUF.items():
        if key != "mrope_interleaved":
            tc[key] = value
    return tc


def shell_config(pack, *, q8: str = "affine8") -> dict:
    """``config.json`` for the existing ``qwen4_exp`` model class, with the per-module ``quantization`` block.

    Modules whose weight is converted to 8-bit affine get ``{"group_size": 32,
    "bits": 8, "mode": "affine"}``; every other Linear/Embedding of the shell
    gets ``False``. The routed experts are not described (their replacement
    block owns them).
    """
    sk = _skeleton(pack)
    tc = shell_text_config(sk)
    cfg = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "tie_word_embeddings": tc["tie_word_embeddings"],
        "text_config": tc,
    }
    image = _meta(sk, "ple.image_token_id", None)
    if image is not None:
        cfg["image_token_id"] = int(image)
    quant = shell_quantization(sk, q8=q8)
    if quant is not None:
        cfg["quantization"] = quant
        cfg["quantization_config"] = dict(quant)
    return cfg


# --------------------------------------------------------------------------- the plan


def _v_head_order(k_heads: int, v_heads: int) -> np.ndarray:
    """``idx`` with ``grouped[h] = tiled[idx[h]]``: grouped head ``g * r + j`` is tiled head ``j * K + g``."""
    r = v_heads // k_heads
    g, j = np.divmod(np.arange(v_heads), r)
    return j * k_heads + g


def _expand(order: np.ndarray, width: int) -> np.ndarray:
    return (order[:, None] * width + np.arange(width)[None, :]).reshape(-1)


@dataclass(frozen=True)
class Entry:
    """One model parameter (or one quantized module) and how it is made."""

    target: str  # parameter name, e.g. language_model.model.layers.0.linear_attn.in_proj_qkv.weight
    sources: tuple[str, ...]
    op: str
    shape: tuple[int, ...]  # the model's (unquantized) parameter shape
    note: str = ""
    linear: bool = False  # a Linear/Embedding weight (may be stored quantized)
    args: dict = field(default_factory=dict, compare=False)


_COPY_NOTES = {
    "norm1": "copy (file holds 1 + w, the model's post-sanitize convention)",
    "copy": "copy",
}


def shell_plan(pack) -> list[Entry]:
    """Every shell parameter, its GGUF source(s) and transform; raises on anything unaccounted for."""
    sk = _skeleton(pack)
    g = geometry(sk)
    entries: list[Entry] = []
    T = sk.tensors

    def add(target, sources, op, shape, note="", linear=False, **args):
        entries.append(Entry(target, tuple(sources), op, tuple(int(x) for x in shape), note, linear, args))

    def lin(target, src, out_dim, in_dim, op="copy", note="copy", **args):
        add(target, [src], op, (out_dim, in_dim), note, True, **args)

    H, HC = g.hidden, g.hc_hidden
    v_rows = _expand(_v_head_order(g.k_heads, g.v_heads), g.v_head_dim)
    v_heads1 = _v_head_order(g.k_heads, g.v_heads)
    qk = 2 * g.key_dim
    qkv_rows = np.concatenate([np.arange(qk), qk + v_rows])

    lin(LM + "embed_tokens.weight", "token_embd.weight", g.vocab, H)
    if "output.weight" in T:
        lin(LM_HEAD, "output.weight", g.vocab, H)
    mix = LM + "hyper_connection_mixer."
    add(mix + "hc_norm.weight", ["output_hc_norm.weight"], "copy", (HC,), _COPY_NOTES["norm1"])
    lin(mix + "input_mix_weight_down.weight", "output_hc_down.weight", g.hc_lowrank, HC)
    lin(mix + "input_mix_weight_up.weight", "output_hc_up.weight", HC, g.hc_lowrank)

    for i in range(g.layers):
        b = f"blk.{i}."
        p = f"{LM}layers.{i}."
        for hc, dst in (("attn", "attn_hyper_connection."), ("ffn", "mlp_hyper_connection.")):
            add(p + dst + "hc_norm.weight", [b + f"hc_{hc}_norm.weight"], "copy", (HC,), _COPY_NOTES["norm1"])
            lin(p + dst + "input_mix_weight_down.weight", b + f"hc_{hc}_down.weight", g.hc_lowrank, HC)
            lin(p + dst + "input_mix_weight_up.weight", b + f"hc_{hc}_up.weight", HC, g.hc_lowrank)
            lin(p + dst + "block_inject_weight.weight", b + f"hc_{hc}_inject.weight", g.hc_count, HC)
        n_exp = int(_meta(sk, "expert_count"))
        ff = int(_meta(sk, "expert_shared_feed_forward_length"))
        lin(p + "mlp.gate.weight", b + "ffn_gate_inp.weight", n_exp, H)
        lin(p + "mlp.shared_expert_gate.weight", b + "ffn_gate_inp_shexp.weight", 1, H)
        lin(p + "mlp.shared_expert.gate_proj.weight", b + "ffn_gate_shexp.weight", ff, H)
        lin(p + "mlp.shared_expert.up_proj.weight", b + "ffn_up_shexp.weight", ff, H)
        lin(p + "mlp.shared_expert.down_proj.weight", b + "ffn_down_shexp.weight", H, ff)
        if g.linear_layers[i]:
            a = p + "linear_attn."
            lin(a + "in_proj_qkv.weight", b + "attn_qkv.weight", qk + g.value_dim, H, "rows",
                "value rows: tiled -> grouped heads", perm=qkv_rows)
            lin(a + "in_proj_z.weight", b + "attn_gate.weight", g.value_dim, H, "rows",
                "rows: tiled -> grouped heads", perm=v_rows)
            lin(a + "in_proj_b.weight", b + "ssm_beta.weight", g.v_heads, H, "rows",
                "rows: tiled -> grouped heads", perm=v_heads1)
            lin(a + "in_proj_a.weight", b + "ssm_alpha.weight", g.v_heads, H, "rows",
                "rows: tiled -> grouped heads", perm=v_heads1)
            lin(a + "out_proj.weight", b + "ssm_out.weight", H, g.value_dim, "cols",
                "columns: tiled -> grouped heads", perm=v_rows, width=g.v_head_dim)
            add(a + "conv1d.weight", [b + "ssm_conv1d.weight"], "conv", (qk + g.value_dim, g.conv_kernel, 1),
                "value channels: tiled -> grouped; [C, K] -> [C, K, 1]", perm=qkv_rows)
            add(a + "dt_bias", [b + "ssm_dt.bias"], "rows", (g.v_heads,), "tiled -> grouped heads", perm=v_heads1)
            add(a + "A_log", [b + "ssm_a"], "a_log", (g.v_heads,),
                "A_log = log(-ssm_a), snapped to its bf16 source; tiled -> grouped", perm=v_heads1)
            add(a + "norm.weight", [b + "ssm_norm.weight"], "copy", (g.v_head_dim,), "copy (one-centred in both)")
        else:
            s = p + "self_attn."
            lin(s + "q_proj.weight", b + "attn_q.weight", 2 * g.n_heads * g.head_dim, H)
            lin(s + "k_proj.weight", b + "attn_k.weight", g.n_kv_heads * g.head_dim, H)
            lin(s + "v_proj.weight", b + "attn_v.weight", g.n_kv_heads * g.head_dim, H)
            lin(s + "o_proj.weight", b + "attn_output.weight", H, g.n_heads * g.head_dim)
            add(s + "q_norm.weight", [b + "attn_q_norm.weight"], "copy", (g.head_dim,), _COPY_NOTES["norm1"])
            add(s + "k_norm.weight", [b + "attn_k_norm.weight"], "copy", (g.head_dim,), _COPY_NOTES["norm1"])
            if g.idx_heads:
                add(s + "indexer.index_qk_proj.weight", [b + "indexer.q_proj.weight", b + "indexer.k_proj.weight"],
                    "concat", ((g.idx_heads + g.idx_kv_heads) * g.idx_head_dim, H),
                    "concat rows [q_proj; k_proj]", True)
                add(s + "indexer.q_layernorm.weight", [b + "indexer.q_norm.weight"], "copy", (g.idx_head_dim,),
                    _COPY_NOTES["norm1"])
                add(s + "indexer.k_layernorm.weight", [b + "indexer.k_norm.weight"], "copy", (g.idx_head_dim,),
                    _COPY_NOTES["norm1"])
        if i in g.ple_layers:
            q = p + "ple."
            lin(q + "key_proj.weight", b + "ple_key.weight", HC, g.ple_embed_dim)
            lin(q + "value_proj.weight", b + "ple_value.weight", H, g.ple_embed_dim)
            for nm in ("norm_key", "norm_query", "norm_conv"):
                add(q + f"{nm}.weight", [b + f"ple_{nm}.weight"], "copy", (HC,), _COPY_NOTES["norm1"])
            add(q + "conv_weight", [b + "ple_conv1d.weight"], "conv", (HC, g.ple_conv_kernel, 1), "[C, K] -> [C, K, 1]")
            e = q + "ple_embedding."
            heads = (int(_meta(sk, "ple.ngram_size", 3)) - 1) * int(_meta(sk, "ple.heads_per_ngram", 8))
            for nm, key, n in (
                ("layer_multipliers", "ple.layer_multipliers", int(_meta(sk, "ple.ngram_size", 3))),
                ("ngram_heads_vocab_sizes", "ple.head_vocab_sizes", heads),
                ("ngram_heads_offsets", "ple.head_offsets", heads),
            ):
                add(e + nm, [], "meta", (n,), f"int64 from metadata {ARCH}.{key}", key=key)

    _check_plan(sk, g, entries)
    return entries


def _logical_shape(sk, e: Entry) -> tuple[int, ...]:
    shapes = [sk.tensor(s).shape for s in e.sources]
    if e.op == "concat":
        if any(len(s) != 2 or s[1] != shapes[0][1] for s in shapes):
            raise ShellError(f"{e.target}: cannot concatenate rows of {shapes}")
        return (sum(s[0] for s in shapes), shapes[0][1])
    if e.op == "conv":
        (c, k), = [tuple(s) for s in shapes]
        return (c, k, 1)
    if e.op == "meta":
        return (len(_meta(sk, e.args["key"])),)
    (s,) = shapes
    if len(e.shape) == 1:
        n = 1
        for d in s:
            n *= d
        return (n,)
    return tuple(s)


def _check_plan(sk, g: Geometry, entries: list[Entry]) -> None:
    used: dict[str, str] = {}
    problems = []
    for e in entries:
        for s in e.sources:
            if s not in sk.tensors:
                problems.append(f"{e.target}: source tensor {s} is missing")
                continue
            if s in used:
                problems.append(f"{s} feeds both {used[s]} and {e.target}")
            used[s] = e.target
            t = sk.tensors[s].ggml_type
            if t not in _SUPPORTED_TYPES:
                problems.append(f"{s}: ggml type {t} is not supported (F32, F16, BF16, Q8_0)")
        if problems:
            continue
        got = _logical_shape(sk, e)
        if got != e.shape:
            problems.append(f"{e.target}: source shape {got} != model shape {e.shape}")
    extra = sorted(set(sk.tensors) - set(used) - set(SKIPPED_TENSORS))
    if extra:
        problems.append(f"{len(extra)} skeleton tensors are not accounted for, e.g. {extra[:6]}")
    names = [e.target for e in entries]
    if len(set(names)) != len(names):
        problems.append("duplicate target names in the plan")
    if problems:
        raise ShellError("; ".join(problems[:12]) + (f" (+{len(problems) - 12} more)" if len(problems) > 12 else ""))


#: GGUF tensors deliberately not converted (name -> reason). Empty for the Edge0-120B skeleton.
SKIPPED_TENSORS: dict[str, str] = {}


def shell_accounting(pack) -> dict[str, str]:
    """GGUF tensor name -> the parameter it becomes (or the reason it is skipped), for every skeleton tensor."""
    sk = _skeleton(pack)
    out = {s: e.target for e in shell_plan(sk) for s in e.sources}
    out.update({k: f"skipped: {v}" for k, v in SKIPPED_TENSORS.items() if k in sk.tensors})
    missing = set(sk.tensors) - set(out)
    if missing:
        raise ShellError(f"unaccounted skeleton tensors: {sorted(missing)[:6]}")
    return out


def expected_shell_parameters(text_config: dict) -> dict[str, tuple[int, ...]]:
    """The model's shell parameters (name -> unquantized shape) for a ``text_config``, written from
    ``qwen4_exp.py``'s module tree: everything except ``mlp.switch_mlp.*`` (routed experts), the
    n-gram table and the MTP head. Tests check it against an instantiated model."""
    c = text_config
    H = c["hidden_size"]
    hc = c.get("hc_count", 4)
    HC = H * hc
    r = c.get("hc_lowrank", 320)
    out: dict[str, tuple[int, ...]] = {}
    out[LM + "embed_tokens.weight"] = (c["vocab_size"], H)
    if not c.get("tie_word_embeddings", False):
        out[LM_HEAD] = (c["vocab_size"], H)
    mix = LM + "hyper_connection_mixer."
    out[mix + "hc_norm.weight"] = (HC,)
    out[mix + "input_mix_weight_down.weight"] = (r, HC)
    out[mix + "input_mix_weight_up.weight"] = (HC, r)
    kd = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    vd = c["linear_num_value_heads"] * c["linear_value_head_dim"]
    nv = c["linear_num_value_heads"]
    hd = c["head_dim"]
    ple_ids = sorted(set(c.get("ple_layer_ids") or []))
    for i, kind in enumerate(c["layer_types"]):
        p = f"{LM}layers.{i}."
        for dst in ("attn_hyper_connection.", "mlp_hyper_connection."):
            out[p + dst + "hc_norm.weight"] = (HC,)
            out[p + dst + "input_mix_weight_down.weight"] = (r, HC)
            out[p + dst + "input_mix_weight_up.weight"] = (HC, r)
            out[p + dst + "block_inject_weight.weight"] = (hc, HC)
        ff = c["shared_expert_intermediate_size"]
        out[p + "mlp.gate.weight"] = (c["num_experts"], H)
        out[p + "mlp.shared_expert_gate.weight"] = (1, H)
        out[p + "mlp.shared_expert.gate_proj.weight"] = (ff, H)
        out[p + "mlp.shared_expert.up_proj.weight"] = (ff, H)
        out[p + "mlp.shared_expert.down_proj.weight"] = (H, ff)
        if kind == "linear_attention":
            a = p + "linear_attn."
            out[a + "in_proj_qkv.weight"] = (2 * kd + vd, H)
            out[a + "in_proj_z.weight"] = (vd, H)
            out[a + "in_proj_b.weight"] = (nv, H)
            out[a + "in_proj_a.weight"] = (nv, H)
            out[a + "out_proj.weight"] = (H, vd)
            out[a + "conv1d.weight"] = (2 * kd + vd, c["linear_conv_kernel_dim"], 1)
            out[a + "dt_bias"] = (nv,)
            out[a + "A_log"] = (nv,)
            out[a + "norm.weight"] = (c["linear_value_head_dim"],)
        else:
            s = p + "self_attn."
            out[s + "q_proj.weight"] = (2 * c["num_attention_heads"] * hd, H)
            out[s + "k_proj.weight"] = (c["num_key_value_heads"] * hd, H)
            out[s + "v_proj.weight"] = (c["num_key_value_heads"] * hd, H)
            out[s + "o_proj.weight"] = (H, c["num_attention_heads"] * hd)
            out[s + "q_norm.weight"] = (hd,)
            out[s + "k_norm.weight"] = (hd,)
            if c.get("indexer_n_heads"):
                ih = c["indexer_head_dim"]
                out[s + "indexer.index_qk_proj.weight"] = ((c["indexer_n_heads"] + c["indexer_kv_heads"]) * ih, H)
                out[s + "indexer.q_layernorm.weight"] = (ih,)
                out[s + "indexer.k_layernorm.weight"] = (ih,)
        if (i + 1) in ple_ids:
            q = p + "ple."
            pd = c.get("ple_embed_dim") or H
            out[q + "key_proj.weight"] = (HC, pd)
            out[q + "value_proj.weight"] = (H, pd)
            for nm in ("norm_key", "norm_query", "norm_conv"):
                out[q + f"{nm}.weight"] = (HC,)
            out[q + "conv_weight"] = (HC, c.get("ple_conv_kernel_size", 4), 1)
            heads = (c.get("ngram_size", 3) - 1) * c.get("heads_per_ngram", 8)
            out[q + "ple_embedding.layer_multipliers"] = (c.get("ngram_size", 3),)
            out[q + "ple_embedding.ngram_heads_vocab_sizes"] = (heads,)
            out[q + "ple_embedding.ngram_heads_offsets"] = (heads,)
    return out


def check_plan_covers_model(pack) -> None:
    """Fail closed: the plan's targets must be exactly the model's shell parameters, with the same shapes."""
    sk = _skeleton(pack)
    plan = {e.target: e.shape for e in shell_plan(sk)}
    want = expected_shell_parameters(shell_text_config(sk))
    missing = sorted(set(want) - set(plan))
    extra = sorted(set(plan) - set(want))
    wrong = sorted(k for k in set(want) & set(plan) if want[k] != plan[k])
    if missing or extra or wrong:
        raise ShellError(
            f"plan vs model: {len(missing)} parameters get no tensor (e.g. {missing[:4]}), "
            f"{len(extra)} targets are not model parameters (e.g. {extra[:4]}), "
            f"{len(wrong)} shapes differ (e.g. {[(k, plan[k], want[k]) for k in wrong[:3]]})"
        )


# --------------------------------------------------------------------------- tensor values


@dataclass
class _Q8:
    codes: np.ndarray  # int8 [R, C]
    d: np.ndarray  # float16 [R, C // 32]

    def floats(self) -> np.ndarray:
        r, c = self.codes.shape
        v = self.codes.reshape(r, c // Q8_GROUP, Q8_GROUP).astype(np.float32) * self.d.astype(np.float32)[..., None]
        return v.reshape(r, c)


@dataclass
class _F:
    values: np.ndarray  # float32, exact copy of the stored values
    src: int  # ggml type it came from


def _load(sk, name: str):
    t = sk.tensor(name)
    if t.ggml_type == GGML_Q8_0:
        if len(t.shape) != 2:
            raise ShellError(f"{name}: Q8_0 tensor of rank {len(t.shape)}")
        q, d = unpack_q8_0(sk.raw(name))
        r, c = t.shape
        return _Q8(q.reshape(r, c), d.reshape(r, c // Q8_GROUP))
    if t.ggml_type in _FLOAT_TYPES:
        return _F(sk.array(name), t.ggml_type)
    raise ShellError(f"{name}: ggml type {t.ggml_type} is not supported")


def _rows(x, perm: np.ndarray):
    if isinstance(x, _Q8):
        return _Q8(np.ascontiguousarray(x.codes[perm]), np.ascontiguousarray(x.d[perm]))
    return _F(np.ascontiguousarray(x.values[perm]), x.src)


def _cols(x, perm: np.ndarray, width: int):
    """Permute columns in runs of ``width``; whole Q8_0 blocks move when ``width % 32 == 0``."""
    if isinstance(x, _Q8):
        if width % Q8_GROUP == 0:
            runs = perm.reshape(-1, width)
            if np.any(runs[:, :1] % Q8_GROUP) or np.any(runs - runs[:, :1] != np.arange(width)):
                raise ShellError("column permutation does not move whole runs")
            bperm = (runs[:, :1] // Q8_GROUP + np.arange(width // Q8_GROUP)[None, :]).reshape(-1)
            return _Q8(np.ascontiguousarray(x.codes[:, perm]), np.ascontiguousarray(x.d[:, bperm]))
        x = _F(x.floats(), GGML_Q8_0)  # a permutation inside a block breaks it: keep the exact values as float
    return _F(np.ascontiguousarray(x.values[:, perm]), x.src)


def _concat(parts):
    if all(isinstance(p, _Q8) for p in parts):
        return _Q8(np.concatenate([p.codes for p in parts]), np.concatenate([p.d for p in parts]))
    vals = [p.floats() if isinstance(p, _Q8) else p.values for p in parts]
    srcs = {GGML_F32 if isinstance(p, _Q8) else p.src for p in parts}
    return _F(np.concatenate(vals), srcs.pop() if len(srcs) == 1 else GGML_F32)


def _bf16_exact(v: np.ndarray) -> bool:
    return not np.any(np.ascontiguousarray(v, np.float32).view(np.uint32) & 0xFFFF)


def _a_log(a: np.ndarray) -> tuple[np.ndarray, bool]:
    """``log(-ssm_a)``; the bf16 value nearest to it when that value maps back onto ``ssm_a`` within 2 f32 ulps."""
    if np.any(a >= 0) or not np.all(np.isfinite(a)):
        raise ShellError("ssm_a must be negative and finite (it is -exp(A_log))")
    lg = np.log(-a.astype(np.float64)).astype(np.float32)
    bits = lg.view(np.uint32).astype(np.uint64)
    snapped = (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)
    back = (-np.exp(snapped.astype(np.float64))).astype(np.float32)
    ulps = np.abs(back.view(np.int32).astype(np.int64) - a.astype(np.float32).view(np.int32).astype(np.int64))
    if np.all(ulps <= 2):
        return snapped, True
    return lg, False


# --------------------------------------------------------------------------- quantization block


def _module(target: str) -> str:
    return target[: -len(".weight")]


def shell_quantization(pack, *, q8: str = "affine8") -> dict | None:
    """mlx-lm's ``config["quantization"]``: defaults plus one entry per shell Linear/Embedding module path."""
    if q8 not in Q8_MODES:
        raise ValueError(f"q8 must be one of {Q8_MODES}")
    sk = _skeleton(pack)
    plan = shell_plan(sk)
    if q8 == "float32":
        return None
    quant: dict = {"group_size": Q8_GROUP, "bits": Q8_BITS, "mode": "affine"}
    for e in plan:
        if not e.linear:
            continue
        stays_q8 = all(sk.tensors[s].ggml_type == GGML_Q8_0 for s in e.sources) and not (
            e.op == "cols" and e.args["width"] % Q8_GROUP
        )
        quant[_module(e.target)] = {"group_size": Q8_GROUP, "bits": Q8_BITS, "mode": "affine"} if stays_q8 else False
    return quant


# --------------------------------------------------------------------------- conversion


def _materialise(sk, e: Entry):
    if e.op == "meta":
        return np.asarray([int(x) for x in _meta(sk, e.args["key"])], dtype=np.int64), {}
    parts = [_load(sk, s) for s in e.sources]
    info: dict = {}
    if e.op == "concat":
        x = _concat(parts)
    else:
        (x,) = parts
    if e.op == "rows":
        x = _rows(x, e.args["perm"])
    elif e.op == "cols":
        x = _cols(x, e.args["perm"], e.args["width"])
        if isinstance(x, _F) and x.src == GGML_Q8_0:
            info["q8_fallback"] = "column permutation inside a Q8_0 block"
    elif e.op == "conv":
        if not isinstance(x, _F):
            raise ShellError(f"{e.sources[0]}: a quantized conv kernel is not supported")
        v = x.values
        if "perm" in e.args:
            v = v[e.args["perm"]]
        x = _F(np.ascontiguousarray(v).reshape(e.shape), x.src)
    elif e.op == "a_log":
        if not isinstance(x, _F):
            raise ShellError(f"{e.sources[0]}: ssm_a must be a float tensor")
        v, snapped = _a_log(x.values.reshape(-1)[e.args["perm"]])
        info["a_log_snapped_to_bf16"] = snapped
        x = _F(v, GGML_BF16 if snapped else GGML_F32)
    if isinstance(x, _F):
        x = _F(x.values.reshape(e.shape), x.src)
    return x, info


def convert_shell(
    pack,
    *,
    q8: str = "affine8",
    q8_scale_dtype: str = "float32",
    narrow_floats: bool = True,
    targets: Iterable[str] | None = None,
    report: dict | None = None,
) -> Iterator[tuple[str, object]]:
    """Yield ``(parameter name, mx.array)`` for every shell parameter.

    ``q8="affine8"`` stores Q8_0 tensors as MLX 8-bit affine ``weight`` (uint32
    words), ``scales`` and ``biases`` (group size 32); ``q8="float32"`` stores
    their exact values as float32. ``q8_scale_dtype`` other than float32 is not
    lossless under ``mx.dequantize`` (see the module docstring). Float tensors
    keep their values; with ``narrow_floats`` F32/F16 tensors whose values are
    all bfloat16 numbers are stored as bfloat16, others stay float32.
    ``targets`` restricts the output to those parameter or module names (for
    spot checks); the plan is still validated in full first. ``report``, when
    given, receives per-target notes (fallbacks, A_log snapping).
    """
    import mlx.core as mx

    if q8 not in Q8_MODES:
        raise ValueError(f"q8 must be one of {Q8_MODES}")
    if q8_scale_dtype not in SCALE_DTYPES:
        raise ValueError(f"q8_scale_dtype must be one of {SCALE_DTYPES}")
    sk = _skeleton(pack)
    plan = shell_plan(sk)
    check_plan_covers_model(sk)
    wanted = None if targets is None else set(targets)
    sdt = getattr(mx, q8_scale_dtype)

    for e in plan:
        if wanted is not None and e.target not in wanted and _module(e.target) not in wanted:
            continue
        x, info = _materialise(sk, e)
        if report is not None:
            report[e.target] = {"op": e.op, "sources": e.sources, "note": e.note, **info}
        if isinstance(x, np.ndarray):  # int64 buffers
            yield e.target, mx.array(x)
            continue
        if isinstance(x, _Q8) and q8 == "affine8":
            scales = x.d.astype(np.float32)
            if q8_scale_dtype == "float16" and np.any(np.abs(scales) * Q8_OFFSET > np.finfo(np.float16).max):
                raise ShellError(f"{e.target}: -128 * d overflows float16")
            words = np.ascontiguousarray(x.codes.view(np.uint8) ^ np.uint8(0x80)).view(np.uint32)
            base = _module(e.target)
            yield base + ".weight", mx.array(words)
            yield base + ".scales", mx.array(scales).astype(sdt)
            yield base + ".biases", mx.array(scales * np.float32(-Q8_OFFSET)).astype(sdt)
            continue
        if isinstance(x, _Q8):
            x = _F(x.floats(), GGML_Q8_0)
        v = x.values
        if x.src == GGML_BF16 or (narrow_floats and x.src != GGML_Q8_0 and _bf16_exact(v)):
            if not _bf16_exact(v):
                raise ShellError(f"{e.target}: BF16 source values are not bfloat16 numbers")
            yield e.target, mx.array(v).astype(mx.bfloat16)
        else:
            yield e.target, mx.array(v)


def q8_affine_parts(codes: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy form of the Q8_0 -> 8-bit affine repack: (uint32 words [R, C/4], scales f32, biases f32)."""
    codes = np.asarray(codes, np.int8)
    words = np.ascontiguousarray(codes.view(np.uint8) ^ np.uint8(0x80)).view(np.uint32)
    s = np.asarray(d, np.float16).astype(np.float32)
    return words, s, s * np.float32(-Q8_OFFSET)
