"""The multi-token-prediction (MTP) draft head on Edge0-120B packs.

Edge0-120B is the Qwen3.8-Flash-Next (``qwen4_exp``) shell with latent MoE
blocks (:mod:`mtplx.models.edge0_latent`). Its trunk is the Flash Next trunk
outside the routed experts, so the Flash Next draft head fits it unchanged:
``qwen4_exp.Model.attach_mtp`` (inherited by ``edge0_latent.Model``) builds a
:class:`~mtplx.models.qwen4_exp.Qwen4ExpMTP` from the pack's ``mtp.safetensors``,
strips the ``mtp.`` prefix, adds +1 to the two ``pre_fc`` norms, infers each
module's quantization from its packed shapes and loads strictly. The head runs
its own full-attention decoder layer with the base model's routed experts
(``switch_mlp``); only the trunk's experts are latent.

Source of the head. The base Flash Next pack's ``mtp.safetensors``; the Edge0
release's own ``mtp/`` folder holds the same head quantized more coarsely.
:mod:`mtplx.edge0.forge` copies the file into the pack after :func:`check_head`
accepts it, and writes ``text_config.mtp_num_hidden_layers = 1``,
``text_config.mtp_use_dedicated_embeddings = false`` and
``mlx_lm_extra_tensors.mtp_file = "mtp.safetensors"`` (the base pack's keys).

The runtime arm (:func:`inject_edge0_latent_mtp_support`) is a separate
predicate from ``is_qwen4_exp_mtp_config``, which stays False for Edge0: about
thirty ``qwen4_exp`` checks gate base-only speed lanes.

No MLX import at module level: the registry imports :func:`head_status`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MODEL_TYPES = frozenset({"edge0_latent", "edge0_latent_text"})
HEAD_FILE = "mtp.safetensors"
#: Keys the forge writes into ``config.json`` when the pack carries a head.
CONFIG_TEXT_KEYS = {"mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False}
#: ``mlx.core.quantize`` widths; anything else cannot be built.
_QUANT_BITS = frozenset({2, 3, 4, 5, 6, 8})
_QUANT_GROUPS = frozenset({32, 64, 128})
_FLOAT_DTYPES = frozenset({"F32", "BF16", "F16"})


class HeadError(ValueError):
    """An ``mtp.safetensors`` that is not a Flash Next draft head for this config."""


# --------------------------------------------------------------------------- predicates


def is_edge0_latent_mtp_config(config: dict | None) -> bool:
    """Is this an Edge0 latent pack config (top-level or text model type)?

    As for the base family, weight presence is decided at inject time: a pack
    without ``mtp.safetensors`` degrades to autoregressive.
    """
    cfg = config or {}
    tc = cfg.get("text_config") or {}
    return bool(MODEL_TYPES & {str(cfg.get("model_type") or "").lower(), str(tc.get("model_type") or "").lower()})


def declares_head(config: dict | None) -> bool:
    """Does ``config.json`` declare a draft head (``text_config.mtp_num_hidden_layers >= 1``)?"""
    cfg = config or {}
    tc = cfg.get("text_config") or cfg
    try:
        return int(tc.get("mtp_num_hidden_layers") or 0) >= 1
    except (TypeError, ValueError):
        return False


def head_path(model_dir, config: dict | None = None) -> Path:
    """Where the pack's head lives. ``attach_mtp`` reads ``<dir>/mtp.safetensors`` only."""
    return Path(model_dir) / HEAD_FILE


# --------------------------------------------------------------------------- header reads


def read_header(path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """``{name: (dtype, shape)}`` from a safetensors header; the file must be exactly header + data."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise HeadError(f"{path}: truncated safetensors header")
            n = int.from_bytes(raw, "little")
            if n <= 0 or n > 100 * 2**20:
                raise HeadError(f"{path}: implausible safetensors header length {n}")
            header = json.loads(f.read(n))
    except OSError as exc:
        raise HeadError(f"{path}: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise HeadError(f"{path}: not a safetensors file ({exc})") from exc
    if not isinstance(header, dict):
        raise HeadError(f"{path}: not a safetensors file")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    end = max((int(v["data_offsets"][1]) for v in tensors.values()), default=0)
    size = os.stat(path).st_size
    if size != 8 + n + end:
        raise HeadError(f"{path}: {size} bytes on disk, header describes {8 + n + end}")
    return {k: (str(v["dtype"]), tuple(int(d) for d in v["shape"])) for k, v in tensors.items()}


def head_status(model_dir, config: dict | None = None) -> tuple[bool, str]:
    """Cheap structural check for the compatibility verdict (header only, no MLX).

    ``(True, "")`` when ``mtp.safetensors`` exists and looks like a Flash Next
    head for this config (every key under ``mtp.``, the two ``pre_fc`` norms
    at the config's widths, the head's attention layer present); else
    ``(False, reason)``. The forge and the loader run the full check.
    """
    model_dir = Path(model_dir)
    if config is None:
        try:
            config = json.loads((model_dir / "config.json").read_text())
        except (OSError, ValueError) as exc:
            return False, f"config.json unreadable: {exc}"
    path = head_path(model_dir, config)
    if not path.exists():
        return False, f"{HEAD_FILE} is missing"
    try:
        header = read_header(path)
    except HeadError as exc:
        return False, str(exc)
    stray = sorted(k for k in header if not k.startswith("mtp."))
    if stray:
        return False, f"{len(stray)} tensors outside mtp.* (e.g. {stray[0]})"
    tc = config.get("text_config") or config
    try:
        hidden = int(tc["hidden_size"])
        width = hidden * int(tc.get("hc_count") or 1)
    except (KeyError, TypeError, ValueError):
        return False, "config.json text_config lacks hidden_size"
    want = {
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (width,),
        "mtp.hyper_connection_mixer.hc_norm.weight": (width,),
    }
    for name, shape in want.items():
        got = header.get(name)
        if got is None:
            return False, f"{name} is missing"
        if got[1] != shape:
            return False, f"{name} has shape {list(got[1])}, this model needs {list(shape)}"
    if "mtp.layers.0.self_attn.q_proj.weight" not in header:
        return False, "mtp.layers.0.self_attn.q_proj.weight is missing"
    return True, ""


# --------------------------------------------------------------------------- the full check


def _text_args(config: dict):
    from mtplx.models.qwen4_exp import TextArgs

    tc = dict(config.get("text_config") or config)
    return TextArgs.from_dict(tc)


def expected_head_parameters(config: dict, header: dict[str, tuple[str, tuple[int, ...]]]) -> tuple[dict, dict]:
    """``({name: shape}, {module: {"bits", "group_size"}})`` the loader will want for this file.

    Mirrors ``qwen4_exp.Model.attach_mtp``: a fresh :class:`Qwen4ExpMTP` for
    this config (parameters stay lazy, nothing is evaluated), quantization per
    module inferred from the file's packed shapes (bits from the uint32 column
    count against the module's input width, group size from the scales
    columns), then quantized the way the loader does.
    """
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mtplx.models.qwen4_exp import Qwen4ExpMTP

    head = Qwen4ExpMTP(_text_args(config))
    stripped = {k[len("mtp."):]: v for k, v in header.items() if k.startswith("mtp.")}
    module_in = {
        p[: -len(".weight")]: int(a.shape[-1])
        for p, a in tree_flatten(head.parameters())
        if p.endswith(".weight") and a.ndim >= 2
    }
    qmap: dict[str, dict[str, int]] = {}
    problems = []
    for mod, in_f in module_in.items():
        w = stripped.get(f"{mod}.weight")
        s = stripped.get(f"{mod}.scales")
        if w is None or s is None or w[0] != "U32":
            continue
        if not s[1] or (int(w[1][-1]) * 32) % in_f or in_f % int(s[1][-1]):
            problems.append(f"{mod}: packed {list(w[1])} / scales {list(s[1])} do not divide input width {in_f}")
            continue
        bits = int(w[1][-1]) * 32 // in_f
        group = in_f // int(s[1][-1])
        if bits not in _QUANT_BITS or group not in _QUANT_GROUPS:
            problems.append(
                f"{mod}: packed {list(w[1])} / scales {list(s[1])} infer {bits}-bit groups of {group} against "
                f"input width {in_f}; MLX builds {sorted(_QUANT_BITS)}-bit, groups {sorted(_QUANT_GROUPS)}"
            )
            continue
        qmap[mod] = {"bits": bits, "group_size": group}
    if problems:
        raise HeadError("; ".join(problems[:6]))

    def predicate(path, _module):
        return qmap.get(path) or False

    try:
        nn.quantize(head, group_size=64, bits=8, class_predicate=predicate)
    except ValueError as exc:
        raise HeadError(f"the head's quantization cannot be built: {exc}") from exc
    return {p: tuple(int(d) for d in a.shape) for p, a in tree_flatten(head.parameters())}, qmap


def check_head(path, config: dict) -> dict[str, Any]:
    """Accept ``path`` only when it is exactly the head ``attach_mtp`` builds for ``config``.

    Every tensor must be under ``mtp.``; after the prefix is stripped the
    names must equal the head's parameters (no missing, no extra) with the
    same shapes; a quantized module's weight must be uint32 and every other
    tensor a float type. Raises :class:`HeadError` naming the first
    differences; returns a short report (tensor count, bytes, quantization).
    """
    path = Path(path)
    header = read_header(path)
    stray = sorted(k for k in header if not k.startswith("mtp."))
    if stray:
        raise HeadError(
            f"{path}: {len(stray)} of {len(header)} tensors are not under mtp.* (e.g. {stray[:3]}); "
            "this is not a Flash Next draft head"
        )
    want, qmap = expected_head_parameters(config, header)
    got = {k[len("mtp."):]: v for k, v in header.items()}
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    wrong = []
    for name in sorted(set(want) & set(got)):
        dtype, shape = got[name]
        if shape != want[name]:
            wrong.append(f"{name}: shape {list(shape)}, the head needs {list(want[name])}")
            continue
        mod, _, leaf = name.rpartition(".")
        if mod in qmap and leaf == "weight":
            if dtype != "U32":
                wrong.append(f"{name}: dtype {dtype}, a packed weight is U32")
        elif dtype not in _FLOAT_DTYPES:
            wrong.append(f"{name}: dtype {dtype}, expected a float type")
    if missing or extra or wrong:
        raise HeadError(
            f"{path} does not fit this model's draft head: {len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(extra)} unexpected (e.g. {extra[:3]}), {len(wrong)} mismatched (e.g. {wrong[:3]})"
        )
    bits = sorted({q["bits"] for q in qmap.values()})
    groups = sorted({q["group_size"] for q in qmap.values()})
    return {
        "tensors": len(header),
        "bytes": os.stat(path).st_size,
        "quantized_modules": len(qmap),
        "bits": bits,
        "group_sizes": groups,
    }


# --------------------------------------------------------------------------- runtime arm


def inject_edge0_latent_mtp_support(model, path=None, config: dict | None = None, contract=None) -> bool:
    """Attach the pack's draft head to a loaded Edge0 model (``runtime.load(mtp=True)``).

    Returns False (serve autoregressive) for a pack without ``mtp.safetensors``.
    A head file that the loader rejects raises :class:`RuntimeError` naming the
    file: a shipped head that cannot attach is a broken pack, not a missing one.
    """
    if not is_edge0_latent_mtp_config(config or {}) or path is None:
        return False
    head = head_path(path, config)
    if not head.exists():
        return False
    attach = getattr(model, "attach_mtp", None)
    if not callable(attach):
        return False
    try:
        return bool(attach(path))
    except Exception as exc:  # strict load: names or shapes do not fit
        raise RuntimeError(
            f"Edge0 MTP head {head} does not fit this model ({type(exc).__name__}: {exc}); "
            "rebuild it with `python -m mtplx.edge0.forge SRC OUT --mtp-head base` or load with --no-mtp"
        ) from exc


__all__ = [
    "CONFIG_TEXT_KEYS",
    "HEAD_FILE",
    "HeadError",
    "check_head",
    "declares_head",
    "expected_head_parameters",
    "head_path",
    "head_status",
    "inject_edge0_latent_mtp_support",
    "is_edge0_latent_mtp_config",
    "read_header",
]
