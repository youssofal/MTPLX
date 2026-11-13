"""Edge0-120B on MTPLX: the Qwen3.8-Flash-Next (``qwen4_exp``) shell with a latent MoE half.

Edge0-120B is Flash Next with only the routed experts replaced (see
:mod:`mtplx.models.edge0_latent_moe`). This module reuses the ``qwen4_exp``
model class unchanged and swaps every decoder layer's ``mlp`` for a
:class:`LatentMoeBlock` right after construction, so ``qwen4_exp.py`` needs no
hook. The pack it loads is written by :mod:`mtplx.edge0.forge`.

Model type. The config says ``model_type: "edge0_latent"`` (text config
``"edge0_latent_text"``), deliberately not ``qwen4_exp``: MTPLX gates about
thirty load-time and serve-time lanes on the string ``qwen4_exp`` (family
settings, MTP injection, Flash-Next prefill kernel self-checks, fixed-M4
verify lanes, graphbank/one-copy, server defaults), several of which assume
the base expert geometry. A new type makes all of them fall through to their
generic behaviour; the lanes that live inside the ``qwen4_exp`` classes
themselves stay env-gated (all off by default) and match on ``switch_mlp`` or
``SparseMoeBlock``, which this model does not have.

Precision. The forged shell keeps Q8_0 projections as 8-bit affine with
float32 scales and its norm gains in float32, so by default the residual
stream, the KV cache and every activation are float32 (``"float32"``). The
``"bfloat16"`` policy casts the shell's floating parameters to bfloat16 after
loading (as the stock Flash Next packs hold them), which makes the stream
bfloat16; it never touches the D/U scales and biases, ``experts.scalars`` or
the router (held in float32, ``LatentMoeBlock.cast_router_float32``). Pick the
policy with ``edge0_activation_dtype`` in ``config.json`` or the
``MTPLX_EDGE0_DTYPE`` environment variable (which wins).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from . import qwen4_exp as _q
from .edge0_latent_moe import LatentGeometry, LatentMoeBlock

MODEL_TYPE = "edge0_latent"
TEXT_MODEL_TYPE = "edge0_latent_text"
ARCHITECTURE = "Edge0LatentForCausalLM"
DTYPE_ENV = "MTPLX_EDGE0_DTYPE"
ACTIVATION_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16}

#: Parameter heads inside ``LatentMoeBlock`` that come from the latent shards
#: (the rest of the block, router and shared expert, comes from the shell).
LATENT_HEADS = ("dense_d", "dense_u", "experts", "labels")


def _keeps_float32(path: str) -> bool:
    """Parameters a 16-bit cast must never touch (see the module docstring)."""
    if ".mlp." in path:
        tail = path.split(".mlp.", 1)[1]
        if tail.split(".", 1)[0] in LATENT_HEADS or tail.startswith("gate."):
            return True
    return path.endswith("A_log") or "ngram" in path


@dataclass
class ModelArgs(_q.ModelArgs):
    model_type: str = MODEL_TYPE
    text_config: dict = field(default_factory=dict)
    edge0_latent: dict = field(default_factory=dict)
    edge0_activation_dtype: str = "float32"
    mtplx_ngram_source: dict | None = None


class Model(_q.Model):
    """``qwen4_exp.Model`` with :class:`LatentMoeBlock` as every layer's MoE half."""

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        lat = dict(args.edge0_latent or {})
        missing = {"latent_dim", "expert_ff", "n_groups", "dense_bits"} - set(lat)
        if missing:
            raise ValueError(f"config.json edge0_latent block lacks {sorted(missing)}")
        targs = self.language_model.args
        geo = LatentGeometry(
            hidden=int(targs.hidden_size),
            latent_dim=int(lat["latent_dim"]),
            expert_ff=int(lat["expert_ff"]),
            n_groups=int(lat["n_groups"]),
            n_experts=int(targs.num_experts),
            dense_bits=int(lat["dense_bits"]),
        )
        if int(lat.get("top_k", targs.num_experts_per_tok)) != int(targs.num_experts_per_tok):
            raise ValueError("edge0_latent.top_k disagrees with text_config.num_experts_per_tok")
        self.edge0_geometry = geo
        # The qwen4_exp constructor built SparseMoeBlock placeholders (lazy,
        # never evaluated); replace them before anything reads the tree.
        for layer in self.layers:
            layer.mlp = LatentMoeBlock.from_model_args(targs, geo)
        self.activation_dtype = mx.float32

    # -- loading -------------------------------------------------------------------

    def sanitize(self, weights):
        out = super().sanitize(weights)
        problems = []
        for i, layer in enumerate(self.layers):
            prefix = f"language_model.model.layers.{i}.mlp."
            own = {
                k: v
                for k, v in tree_flatten(layer.mlp.parameters())
                if k.split(".", 1)[0] in LATENT_HEADS
            }
            for name, ref in own.items():
                got = out.get(prefix + name)
                if got is None:
                    problems.append(f"{prefix + name}: missing")
                elif tuple(got.shape) != tuple(ref.shape) or got.dtype != ref.dtype:
                    problems.append(
                        f"{prefix + name}: {got.dtype}{tuple(got.shape)}, block wants {ref.dtype}{tuple(ref.shape)}"
                    )
            extra = [
                k
                for k in out
                if k.startswith(prefix)
                and k[len(prefix):].split(".", 1)[0] in LATENT_HEADS
                and k[len(prefix):] not in own
            ]
            problems += [f"{k}: not a LatentMoeBlock parameter" for k in extra]
        stray = [k for k in out if ".mlp.switch_mlp." in k]
        problems += [f"{k}: routed-expert weight in a latent pack" for k in stray[:3]]
        if problems:
            raise ValueError(
                f"edge0 latent weights do not match the model ({len(problems)} problems): "
                + "; ".join(problems[:8])
            )
        return out

    def post_weight_load(self, model_path, *, activation_dtype: str | None = None) -> None:
        """After ``load_weights``: the n-gram table, the float32 router and the activation dtype."""
        spec = self.args.mtplx_ngram_source
        if spec:
            self._attach_edge0_ngram(Path(model_path), spec)
        else:
            super().post_weight_load(model_path)
        for layer in self.layers:
            layer.mlp.cast_router_float32()
        name = activation_dtype or os.environ.get(DTYPE_ENV) or self.args.edge0_activation_dtype
        self.set_activation_dtype(name)

    def _attach_edge0_ngram(self, model_path: Path, spec: dict) -> None:
        from mtplx.edge0 import ngram as edge0_ngram

        if spec.get("backend") != edge0_ngram.BACKEND:
            raise ValueError(f"mtplx_ngram_source backend {spec.get('backend')!r} is not {edge0_ngram.BACKEND!r}")
        target = Path(str(spec.get("file") or "ple-fp8.gguf")).expanduser()
        gguf = target if target.is_absolute() else model_path / target
        if not gguf.exists():
            raise FileNotFoundError(f"{gguf}: the Edge0 n-gram table (mtplx_ngram_source.file) is missing")
        for layer in self.layers:
            if "ple" not in layer:
                continue
            table = layer.ple.ple_embedding.ngram_embedding
            table.pop("weight", None)
            entries, scale = edge0_ngram.open_table(gguf, table.rows, table.dim)
            table._sidecar = _q._SidecarGather(gguf, entries, bits=0, group_size=0)
            table._sidecar.fp8_scale = scale
            print(f"[edge0_latent] ngram table: FP8 rows read in place from {gguf}", flush=True)

    def set_activation_dtype(self, name: str) -> None:
        """``"float32"``: leave the loaded precision. ``"bfloat16"``: cast the shell's floating parameters."""
        if name not in ACTIVATION_DTYPES:
            raise ValueError(f"activation dtype must be one of {sorted(ACTIVATION_DTYPES)}, got {name!r}")
        dtype = ACTIVATION_DTYPES[name]
        if dtype != mx.float32:
            cast = [
                (k, v.astype(dtype))
                for k, v in tree_flatten(self.parameters())
                if v.dtype in (mx.float32, mx.float16) and not _keeps_float32(k)
            ]
            if cast:
                self.update(tree_unflatten(cast), strict=False)
                mx.eval([v for _, v in cast])
        for layer in self.layers:
            layer.mlp.latent_dtype = dtype
        self.activation_dtype = dtype


def load(path, *, activation_dtype: str | None = None, lazy: bool = False):
    """``(model, config)`` from a forged pack, outside the MTPLX runtime (scripts, tests)."""
    from mlx_lm.utils import load_model

    path = Path(path)
    model, config = load_model(path, lazy=lazy, get_model_classes=lambda config: (Model, ModelArgs))
    model.post_weight_load(path, activation_dtype=activation_dtype)
    return model, config


def is_edge0_latent_config(config: dict[str, Any] | None) -> bool:
    cfg = config or {}
    tc = cfg.get("text_config") or {}
    return MODEL_TYPE in (str(cfg.get("model_type") or "").lower(), str(tc.get("model_type") or "").lower()) or (
        TEXT_MODEL_TYPE == str(cfg.get("model_type") or "").lower()
    )


__all__ = [
    "ACTIVATION_DTYPES",
    "ARCHITECTURE",
    "DTYPE_ENV",
    "MODEL_TYPE",
    "Model",
    "ModelArgs",
    "TEXT_MODEL_TYPE",
    "is_edge0_latent_config",
    "load",
]
