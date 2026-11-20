"""Every draft-head norm of a built pack against upstream's stored value (plus one where the model is
trained on ``1 + w``), tensor by tensor.

A Qwen draft head multiplies by ``1 + w`` for each gain its upstream stores zero-centred, and by ``w`` for each it
stores absolute. A pack stores each gain so that, after the product's loader has done what it does to a head of
that family, the model multiplies by exactly what upstream's model does. One tensor that missed its shift is
invisible to a whole-file check: the 35B-A3B Speed and Balance packs shipped their head's
``post_attention_layernorm`` zero-centred beside six shifted gains for five months, and every draft step ran the
head's experts on an input scaled by ``w`` instead of ``1 + w`` (a third of the pack's speed, repaired at load by
``compressed_tensors._repair_unshifted_mtp_norms`` since ``01d074b6``).

So day zero checks the pack it built, before anything else is said about it: for every draft-head norm upstream
has, the gain the loader hands the model must equal upstream's stored value, plus one where the family is
zero-centred, channel for channel, at the coarsest stored type's resolution. The load-time repair is not applied:
a pack that needs it is a pack built wrong, and the step stops (fails closed) with the tensor's name.

    python -m mtplx.commands.forge_head_norms <pack folder> <upstream folder> [--family qwen4_exp] [--json]

The conventions, per family, are the model code's own, read from it (``zero_centred``, ``loaded_gain``):
- ``qwen4_exp`` (``models/qwen4_exp.py``): upstream stores zero-centred exactly the gains ``Model``'s sanitize
  shifts (``_HF_NORM_SHIFT_SUFFIXES``: q and k, the stream and indexer norms) and the two pre-fc gains, which
  ``attach_mtp`` shifts at load (``QWEN4_LOAD_SHIFTED``); the head's input, post-attention and final norms are
  absolute. The sidecar is read as stored, plus the pre-fc +1.
- every other family (Qwen 3.5 and 3.6: ``mtp_patch._load_mtp_weights`` ->
  ``compressed_tensors.shift_delta_mtp_norms``): all seven gains are zero-centred upstream; the sidecar's
  convention is decided once from its q/k and low-set means, a delta sidecar gets +1 on all seven at load, an
  absolute one is read as stored.

What this cannot see: a new generation that changes the convention itself. Then our model's code is wrong as
well, the pack agrees with it, and the sign is a low draft keep rate in the rows (step 5), not this step.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: The seven gains of a Qwen 3.x / Qwen 4 draft head, by the end of their name (``mtp.norm.weight`` is the eighth
#: spelling of the final norm's key; matched whole below).
HEAD_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
)
FINAL_NORM = "mtp.norm.weight"
#: The gains the Qwen 4 head's loader shifts itself (``Model.attach_mtp``): they ship raw zero-centred. Kept equal
#: to the loader by ``tests/test_forge_head_norms.py``.
QWEN4_LOAD_SHIFTED = ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")
QWEN4 = "qwen4_exp"


class HeadNormMismatch(RuntimeError):
    """A built pack's draft head would run on gains its upstream was not trained with."""


#: What a synthetic, headers-only checkpoint writes in its safetensors metadata: headers
#: only, no tensor data, so there are no values to compare. Never in a real checkpoint.
SYNTHETIC_MARKER = "mtplx_synthetic_source"


def headers_only(folder: Path) -> bool:
    """True for a synthetic checkpoint written as headers only (no values to read)."""

    for path in sorted(Path(folder).glob("*.safetensors")):
        metadata = _header(path)[1].get("__metadata__") or {}
        if SYNTHETIC_MARKER in metadata:
            return True
    return False


def head_key(name: str) -> str | None:
    """``mtp.<...>`` for a draft-head norm's name (whatever prefix a checkpoint gives its heads), else None."""

    at = name.rfind("mtp.")
    if at < 0:
        return None
    key = name[at:]
    if key == FINAL_NORM or any(key.endswith(suffix) for suffix in HEAD_NORM_SUFFIXES):
        return key
    return None


def _header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        return length, json.loads(stream.read(length))


def _array(path: Path, length: int, info: dict[str, Any]) -> np.ndarray:
    first, last = info["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(8 + length + first)
        raw = stream.read(last - first)
    dtype = info["dtype"]
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(info["shape"])
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy()
    raise HeadNormMismatch(f"a draft-head norm is stored as {dtype}: gains are carried in a float type")


def head_norms(folder: Path) -> dict[str, tuple[np.ndarray, str]]:
    """Every one-dimensional draft-head norm in a folder's safetensors: key -> (float32 values, stored type)."""

    out: dict[str, tuple[np.ndarray, str]] = {}
    for path in sorted(Path(folder).rglob("*.safetensors")):
        length, header = _header(path)
        for name, info in header.items():
            if name == "__metadata__" or len(info.get("shape") or ()) != 1:
                continue
            key = head_key(name)
            if key is None:
                continue
            if key in out:
                raise HeadNormMismatch(f"{key} is stored twice in {folder}")
            out[key] = (_array(path, length, info), info["dtype"])
    return out


def sidecar_is_delta(norms: dict[str, tuple[np.ndarray, str]]) -> bool:
    """The other families' loader verdict on a sidecar (``compressed_tensors.mtp_sidecar_norms_are_delta``)."""

    from mtplx.compressed_tensors import mtp_sidecar_norms_are_delta

    return bool(mtp_sidecar_norms_are_delta({key: values for key, (values, _dtype) in norms.items()}))


def qwen4_zero_centred() -> tuple[str, ...]:
    """The Qwen 4 head's gains its upstream stores zero-centred, from the model's own code."""

    from mtplx.models.qwen4_exp import Model

    return tuple(Model._HF_NORM_SHIFT_SUFFIXES) + QWEN4_LOAD_SHIFTED


def zero_centred(family: str, key: str) -> bool:
    """Whether upstream's model multiplies by ``1 + w`` for this stored gain (else by ``w``)."""

    if family == QWEN4:
        return any(key.endswith(suffix) for suffix in qwen4_zero_centred())
    return True


def loaded_gain(family: str, key: str, stored: np.ndarray, *, delta: bool) -> np.ndarray:
    """The gain the product's loader hands the model for one stored draft-head norm (no load-time repair)."""

    if family == QWEN4:
        return stored + 1.0 if any(key.endswith(suffix) for suffix in QWEN4_LOAD_SHIFTED) else stored
    return stored + 1.0 if delta else stored


_MANTISSA_BITS = {"BF16": 7, "F16": 10, "F32": 23}


def tolerance(expected: np.ndarray, upstream: np.ndarray, *dtypes: str) -> np.ndarray:
    """One unit in the last place of the coarsest of the types the value passed through (upstream's, the
    pack's: an FP16 sibling's gains are its bfloat16 parent's), at the larger of the two operands and at least
    at 1.0: what computing and rounding ``1 + w`` may move it, wherever the sum was taken, and at most 1/128
    below 128 in bfloat16, far below the 1.0 a missed or doubled shift leaves."""

    bits = min(_MANTISSA_BITS.get(dtype, 7) for dtype in dtypes)
    magnitude = np.maximum(np.maximum(np.abs(expected), np.abs(upstream)), 1.0)
    return np.exp2(np.floor(np.log2(magnitude)) - bits)


def audit(pack: Path, upstream: Path, *, family: str) -> dict[str, Any]:
    """Each upstream draft-head norm against the pack's, as the loader reads it: rows and a verdict.
    Raises nothing: ``ok`` is the verdict, ``problems`` its reasons."""

    theirs = head_norms(upstream)
    ours = head_norms(pack)
    delta = sidecar_is_delta(ours) if family != QWEN4 else False
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for key in sorted(set(theirs) | set(ours)):
        if key not in ours:
            problems.append(f"{key}: upstream has it and the pack does not")
            continue
        if key not in theirs:
            problems.append(f"{key}: the pack has it and upstream does not")
            continue
        (upstream_values, upstream_dtype), (stored, dtype) = theirs[key], ours[key]
        if stored.shape != upstream_values.shape:
            problems.append(f"{key}: {list(stored.shape)} in the pack, {list(upstream_values.shape)} upstream")
            continue
        loaded = loaded_gain(family, key, stored, delta=delta)
        shifted = zero_centred(family, key)
        expected = upstream_values + 1.0 if shifted else upstream_values
        error = np.abs(loaded - expected)
        off = int(np.count_nonzero(error > tolerance(expected, upstream_values, dtype, upstream_dtype)))
        row = {
            "tensor": key,
            "channels": int(stored.size),
            "upstream_mean": round(float(upstream_values.mean()), 4),
            "stored_mean": round(float(stored.mean()), 4),
            "loaded_mean": round(float(loaded.mean()), 4),
            "expected_mean": round(float(expected.mean()), 4),
            "upstream_convention": "zero-centred (1 + w)" if shifted else "absolute (w)",
            "largest_error": round(float(error.max()), 6),
            "channels_off": off,
            "ok": off == 0,
        }
        rows.append(row)
        if off:
            shift = float(np.median(loaded - expected))
            hint = (" (the +1 is missing)" if abs(shift + 1.0) < 0.05 else " (one +1 too many)" if abs(shift - 1.0) < 0.05
                    else "")
            problems.append(
                f"{key}: the loader would hand the model mean {row['loaded_mean']}, upstream trained it on "
                f"mean {row['expected_mean']}{hint}; {off} of {stored.size} channels off"
            )
    if not theirs:
        problems.append(f"{upstream} has no draft-head norms: nothing to check the pack's against")
    return {
        "family": family,
        "sidecar_read_as": "zero-centred (+1 at load)" if delta else "stored",
        "tensors": len(rows),
        "rows": rows,
        "problems": problems,
        "ok": not problems,
    }


def check(pack: Path, upstream: Path, *, family: str) -> dict[str, Any]:
    """``audit``, failing closed: a pack whose head would run on gains upstream was not trained with stops."""

    result = audit(pack, upstream, family=family)
    if not result["ok"]:
        raise HeadNormMismatch("draft-head norms against upstream: " + "; ".join(result["problems"]))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pack", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--family", default="", help="qwen4_exp for a hyper-connection trunk; empty for the others")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = audit(args.pack, args.upstream, family=args.family)
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        for row in result["rows"]:
            mark = "ok " if row["ok"] else "OFF"
            print(f"{mark} {row['tensor']}: loaded {row['loaded_mean']}, expected {row['expected_mean']} "
                  f"(upstream {row['upstream_convention']}), largest error {row['largest_error']}")
        print("GO: every draft-head norm is what upstream's model multiplies by" if result["ok"]
              else "STOP: " + "; ".join(result["problems"]))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
