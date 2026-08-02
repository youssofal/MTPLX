#!/usr/bin/env python3
"""Build a self-contained DeepSeek-V4-Flash model dir that ships the MTP block.

The published MLX conversions of DeepSeek-V4-Flash (``mlx-community/
DeepSeek-V4-Flash-2bit-DQ`` and ``-4bit``) DROP the multi-token-prediction block:
their config still declares ``num_nextn_predict_layers: 1`` but no ``mtp.*``
tensor ships, which is the exact case ``mtplx.artifacts.
mtp_weights_present_on_disk`` degrades to autoregressive on.  The MTP weights do
exist upstream: ``deepseek-ai/DeepSeek-V4-Flash`` carries all 1575 of them as
``mtp.0.*``, entirely inside shard ``model-00046-of-00046.safetensors`` (3.59
GiB), FP8/FP4-block quantized.

This script merges the two into ONE stock-served model directory:

  1. hardlink every file of the MLX trunk snapshot (shards + tokenizer + …);
  2. translate ``mtp.0.*`` from the upstream shard onto the MLX module tree and
     write it as one extra shard;
  3. rewrite ``model.safetensors.index.json`` (new entries + total_size) and
     ``config.json`` (per-path ``quantization`` entries where the format needs
     them) so the result loads through the ordinary ``mlx_lm.utils.load_model``
     path with no sidecar, env var or special-case branch.

Source quantization (upstream ``config.json`` ``quantization_config``:
``fmt e4m3``, ``scale_fmt ue8m0``, ``weight_block_size [128, 128]``):

  * **Dense projections** — FP8 ``e4m3`` weight ``[out, in]`` with an ``e8m0``
    scale ``[ceil(out/128), ceil(in/128)]``; real value is
    ``w * scale[n//128, k//128]``.  Reference: ``Linear.__init__`` (model.py
    L138-142) and ``fp8_gemm_kernel`` (kernel.py L242-249), which multiplies the
    accumulator by ``scales_b[n // group_size, k]`` per 128-column block.
  * **Routed experts** — FP4 ``e2m1`` packed two-per-byte along K, stored
    ``[out, in//2]``, with an ``e8m0`` scale ``[out, in//32]``; real value is
    ``fp4(w) * scale[n, k//32]``.  Reference: ``Linear.__init__`` (L131-137) and
    ``fp4_gemm_kernel`` (kernel.py L498-509).  ``e8m0`` is exponent-only, so a
    scale is exactly ``2**(byte - 127)``.

The FP4 nibble order (element ``2j`` in the LOW nibble) is not guessed: the
decode here is asserted **bit-exact** against MLX's own independent ``mxfp4``
dequantizer, which implements the same OCP packing.  Two further invariants of
the reference quantizer are checked as a second, self-contained witness that the
scale semantics are right: ``fast_round_scale`` (kernel.py L36-37) forces the
per-group max magnitude into ``(fp4_max/2, fp4_max]`` = ``(3, 6]`` before
rounding, and into ``(224, 448]`` for FP8.

Precision — ``--bank exact`` (the default, and what the shipped dir holds).  MTP
precision is a standing floor: the draft head must be the most accurate
representation of the source available, because acceptance collapses long before
perplexity notices.  The floor is therefore "no avoidable conversion error at
all", not a bit count — a bit count is the wrong metric here, since re-quantizing
an already-4-bit tensor to affine 8-bit is strictly *worse* than keeping it in its
own format.  So nothing is re-quantized:

  * **routed experts** stay FP4.  MLX's ``mxfp4`` mode is byte-identical to the
    source layout (uint32 words of 8 e2m1 values, low nibble first, one uint8 e8m0
    scale per 32), so the payload is REPACKED — ``w_u8.view(uint32)`` plus the
    scale bytes verbatim — never decoded and re-encoded.  Receipt:
    ``mx.dequantize(..., mode="mxfp4")`` equals the independent LUT decode of the
    source bytes exactly, asserted per tensor.
  * **FP8 e4m3 × e8m0 dense projections** are written as plain **bf16** with no
    quantization entry: every such value has <= 4 significant bits and a
    power-of-two block scale, so bf16's 8 mantissa bits hold it exactly.  Receipt:
    bf16 == the float32 decode, max_abs_diff 0.0, asserted per tensor, with a
    float32 fallback (and a printed warning) if that ever fails.
  * **everything else** keeps its source dtype, round-trip asserted.

``--bank affine-q8`` reproduces the superseded first bank (affine 8-bit,
group_size 64, every stem) and exists only as the A/B arm: the merged dir keeps
that bank beside the live one as ``*.q8-bank.bak`` so a measured acceptance
comparison stays reproducible.  It is lossy by construction — it measures and
reports its own error per stem — and must not be shipped as the default.

Usage:
    python scripts/deepseek_v4_build_mtp_model.py \
        --mtp-shard /path/to/model-00046-of-00046.safetensors \
        --out ~/models/DeepSeek-V4-Flash-2bit-DQ-mtp
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

# --------------------------------------------------------------------------- #
# safetensors reader (the upstream shard uses F8_E4M3 / F8_E8M0, which neither
# numpy nor mx.load can represent, so the header is parsed directly and the
# payload read as raw bytes).
# --------------------------------------------------------------------------- #
_RAW_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1}


class SafeTensorsFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        n = struct.unpack("<Q", self._f.read(8))[0]
        self.header = json.loads(self._f.read(n))
        self.metadata = self.header.pop("__metadata__", None)
        self._base = 8 + n

    def close(self):
        self._f.close()

    def __contains__(self, key):
        return key in self.header

    def dtype(self, key):
        return self.header[key]["dtype"]

    def shape(self, key):
        return tuple(self.header[key]["shape"])

    def bytes_of(self, key) -> np.ndarray:
        """Raw payload of ``key`` as uint8, shaped [*shape[:-1], -1]."""
        meta = self.header[key]
        o0, o1 = meta["data_offsets"]
        self._f.seek(self._base + o0)
        buf = self._f.read(o1 - o0)
        if len(buf) != o1 - o0:
            raise IOError(f"short read for {key}")
        itemsize = _RAW_ITEMSIZE[meta["dtype"]]
        shape = tuple(meta["shape"])
        a = np.frombuffer(buf, dtype=np.uint8)
        if itemsize == 1:
            return a.reshape(shape)
        return a.reshape(*shape[:-1], shape[-1] * itemsize)

    def f32(self, key) -> np.ndarray:
        """Decode an ordinary float tensor (F32 / BF16) to float32."""
        meta = self.header[key]
        raw = self.bytes_of(key)
        if meta["dtype"] == "F32":
            return raw.view(np.float32).reshape(meta["shape"]).copy()
        if meta["dtype"] == "BF16":
            u16 = raw.view(np.uint16).reshape(meta["shape"])
            return (u16.astype(np.uint32) << 16).view(np.float32)
        raise TypeError(f"{key}: not a plain float tensor ({meta['dtype']})")


# --------------------------------------------------------------------------- #
# FP8 / FP4 / E8M0 decode
# --------------------------------------------------------------------------- #
def _e4m3_lut() -> np.ndarray:
    """float8_e4m3fn: 1-4-3, bias 7, no infinities, 0x7f/0xff are NaN."""
    out = np.zeros(256, np.float32)
    for b in range(256):
        sign = -1.0 if b >> 7 else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0:
            val = (man / 8.0) * 2.0**-6          # subnormal
        elif exp == 15 and man == 7:
            val = np.nan
        else:
            val = (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        out[b] = sign * val
    return out


def _e2m1_lut() -> np.ndarray:
    """float4_e2m1fn: 1-2-1, bias 1 -> {0, .5, 1, 1.5, 2, 3, 4, 6} with sign."""
    out = np.zeros(16, np.float32)
    for b in range(16):
        sign = -1.0 if b >> 3 else 1.0
        exp = (b >> 1) & 0x3
        man = b & 0x1
        val = (man * 0.5) if exp == 0 else (1.0 + man * 0.5) * 2.0 ** (exp - 1)
        out[b] = sign * val
    return out


E4M3 = _e4m3_lut()
E2M1 = _e2m1_lut()

FP8_BLOCK = 128
FP4_GROUP = 32


def e8m0(u8: np.ndarray) -> np.ndarray:
    """float8_e8m0fnu is exponent-only: value = 2**(byte - 127)."""
    if np.any(u8 == 255):
        raise ValueError("e8m0 NaN (0xff) in a weight scale")
    return np.exp2(u8.astype(np.float32) - 127.0)


def dequant_fp8_block(w_u8: np.ndarray, s_u8: np.ndarray) -> np.ndarray:
    """FP8 e4m3 [out, in] with an e8m0 block scale [out/128, in/128]."""
    w = E4M3[w_u8]
    if not np.isfinite(w).all():
        raise ValueError("NaN in an FP8 weight payload")
    scale = e8m0(s_u8)
    n, k = w.shape
    scale = np.repeat(np.repeat(scale, FP8_BLOCK, 0), FP8_BLOCK, 1)[:n, :k]
    return w * scale


def dequant_fp4_block(w_u8: np.ndarray, s_u8: np.ndarray) -> np.ndarray:
    """FP4 e2m1 stored [out, in//2] (element 2j low nibble) with e8m0 [out, in/32]."""
    n, half = w_u8.shape
    out = np.empty((n, half * 2), np.float32)
    out[:, 0::2] = E2M1[w_u8 & 0xF]
    out[:, 1::2] = E2M1[w_u8 >> 4]
    scale = np.repeat(e8m0(s_u8), FP4_GROUP, 1)[:, : half * 2]
    return out * scale


def check_fp4_against_mlx(w_u8: np.ndarray, s_u8: np.ndarray, mine: np.ndarray) -> None:
    """Cross-check the nibble order against MLX's independent mxfp4 decoder.

    MLX packs mxfp4 as uint32 words holding 8 fp4 values, low nibble first, with
    uint8 e8m0 scales per 32 elements -- byte-identical to the upstream layout,
    so a bit-exact match pins the packing convention with a second implementation
    rather than an assumption.
    """
    w32 = np.ascontiguousarray(w_u8).view(np.uint32)
    theirs = np.array(
        mx.dequantize(mx.array(w32), mx.array(s_u8), group_size=FP4_GROUP,
                      bits=4, mode="mxfp4", dtype=mx.float32)
    )
    if not np.array_equal(mine, theirs):
        raise AssertionError(
            "FP4 decode disagrees with MLX mxfp4: "
            f"max_abs={float(np.max(np.abs(mine - theirs))):.3e}"
        )


def check_group_max(deq: np.ndarray, s_u8: np.ndarray, group: int, fmt_max: float,
                    axis_scale_repeat: bool, label: str) -> tuple[float, float]:
    """Reference invariant: ``fast_round_scale`` (kernel.py L36-37) picks
    ``s = 2**ceil(log2(amax / fmt_max))``, so the pre-rounding magnitudes land in
    ``(fmt_max/2, fmt_max]``.  After rounding to the format grid the observed max
    can sit one grid step below, so the bound checked here is the rounded one."""
    if axis_scale_repeat:  # FP8: block scale over both axes
        n, k = deq.shape
        s = np.repeat(np.repeat(e8m0(s_u8), FP8_BLOCK, 0), FP8_BLOCK, 1)[:n, :k]
        q = np.abs(deq / s)
        blocks = q.reshape(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
        m = blocks.max((1, 3))
    else:                  # FP4: per-row group of 32 along K
        n, k = deq.shape
        s = e8m0(s_u8)[:, :, None]
        m = np.abs(deq.reshape(n, k // group, group) / s).max(-1)
    lo, hi = float(m.min()), float(m.max())
    if hi > fmt_max * 1.0001:
        raise AssertionError(f"{label}: group max {hi} exceeds format max {fmt_max}")
    if lo <= fmt_max / 4:
        raise AssertionError(
            f"{label}: group max {lo} far below fmt_max/2 -- scale semantics wrong"
        )
    return lo, hi


# --------------------------------------------------------------------------- #
# upstream mtp.0.* -> MLX module tree
# --------------------------------------------------------------------------- #
# Plain (unquantized) renames.  Everything else is handled structurally below.
PLAIN_RENAME = {
    "attn.q_norm.weight": ("attn.q_norm.weight", "bf16"),
    "attn.kv_norm.weight": ("attn.kv_norm.weight", "bf16"),
    "attn.attn_sink": ("attn.attn_sink", "f32"),
    "attn_norm.weight": ("attn_norm.weight", "bf16"),
    "ffn_norm.weight": ("ffn_norm.weight", "bf16"),
    "enorm.weight": ("enorm.weight", "bf16"),
    "hnorm.weight": ("hnorm.weight", "bf16"),
    "norm.weight": ("norm.weight", "bf16"),
    "hc_attn_fn": ("attn_hc.fn", "f32"),
    "hc_attn_base": ("attn_hc.base", "f32"),
    "hc_attn_scale": ("attn_hc.scale", "f32"),
    "hc_ffn_fn": ("ffn_hc.fn", "f32"),
    "hc_ffn_base": ("ffn_hc.base", "f32"),
    "hc_ffn_scale": ("ffn_hc.scale", "f32"),
    "hc_head_fn": ("hc_head.fn", "f32"),
    "hc_head_base": ("hc_head.base", "f32"),
    "hc_head_scale": ("hc_head.scale", "f32"),
    "ffn.gate.weight": ("ffn.gate.weight", "bf16"),
    # reference Gate.bias (model.py L562) is mlx-lm's noaux correction bias
    "ffn.gate.bias": ("ffn.gate.e_score_correction_bias", "f32"),
}

# FP8-block dense projections -> quantized stems on the MLX tree.
FP8_STEMS = {
    "attn.wq_a": "attn.wq_a",
    "attn.wq_b": "attn.wq_b",
    "attn.wkv": "attn.wkv",
    "attn.wo_a": "attn.wo_a",
    "attn.wo_b": "attn.wo_b",
    "e_proj": "e_proj",
    "h_proj": "h_proj",
    "ffn.shared_experts.w1": "ffn.shared_experts.gate_proj",
    "ffn.shared_experts.w3": "ffn.shared_experts.up_proj",
    "ffn.shared_experts.w2": "ffn.shared_experts.down_proj",
}

# FP4-block routed experts -> stacked SwitchGLU stems.
# reference Expert (model.py L587-606): w1 = gate, w3 = up, w2 = down.
EXPERT_STEMS = {
    "w1": "ffn.switch_mlp.gate_proj",
    "w3": "ffn.switch_mlp.up_proj",
    "w2": "ffn.switch_mlp.down_proj",
}


def to_mx(a: np.ndarray, kind: str) -> mx.array:
    arr = mx.array(a)
    return arr.astype(mx.bfloat16) if kind == "bf16" else arr.astype(mx.float32)


MXFP4_SPEC = {"group_size": FP4_GROUP, "bits": 4, "mode": "mxfp4"}


def repack_fp4_to_mxfp4(w_u8: np.ndarray, s_u8: np.ndarray, label: str):
    """Reinterpret one FP4 e2m1 x e8m0 tensor as MLX ``mxfp4``, bit-for-bit.

    Both formats are the OCP microscaling layout: 4-bit elements packed low-nibble
    first, one uint8 e8m0 scale per 32 elements along K.  MLX reads the payload as
    uint32 words, the source stores it as bytes, and little-endian makes those the
    same bytes in the same order — so the translation is a ``view``, not a decode.

    Returns ``(weight_uint32, scales_uint8)`` and asserts the identity that makes
    the view legitimate: MLX's own dequantizer must reproduce the reference LUT
    decode of the source bytes EXACTLY.
    """
    ref = dequant_fp4_block(w_u8, s_u8)
    if np.any(s_u8 == 255):
        raise SystemExit(f"{label}: e8m0 NaN scale byte")
    w32 = np.ascontiguousarray(w_u8).view(np.uint32)
    got = np.array(
        mx.dequantize(
            mx.array(w32), mx.array(s_u8),
            group_size=FP4_GROUP, bits=4, mode="mxfp4", dtype=mx.float32,
        )
    )
    if not np.array_equal(got, ref):
        raise SystemExit(
            f"{label}: mxfp4 repack != source decode "
            f"(max_abs={float(np.max(np.abs(got - ref))):.3e})"
        )
    return w32, s_u8, ref


def dense_exact(dense_f32: np.ndarray, label: str):
    """bf16 if it holds every value of ``dense_f32`` exactly, else float32.

    FP8 e4m3 has 3 mantissa bits and the block scale is a power of two, so bf16's 8
    mantissa bits are strictly more than enough — but that is an argument, and this
    checks it per tensor rather than trusting it.  Returns ``(mx.array, dtype_name,
    max_abs_diff)``.
    """
    bf = mx.array(dense_f32).astype(mx.bfloat16)
    mx.eval(bf)
    diff = float(np.max(np.abs(np.array(bf.astype(mx.float32)) - dense_f32)))
    if diff == 0.0:
        return bf, "bfloat16", diff
    print(f"    !! {label}: bf16 inexact (max_abs_diff={diff:.3e}) -> float32")
    return mx.array(dense_f32), "float32", diff


def quantize_stem(dense_f32: np.ndarray, group_size: int, bits: int):
    """Affine-quantize one [out, in] matrix (or a stack thereof).

    Only reachable under ``--bank affine-q8``; see the module docstring for why the
    default bank does not re-quantize anything.

    bf16 is exact for every FP8/FP4 source value (both hold <= 4 significant bits
    with power-of-two block scales), and the checkpoint convention stores
    scales/biases in bf16, so the cast is made before quantizing rather than
    after.

    Returns ``(weight, scales, biases, max_err_over_absmax, rel_frobenius)``.
    Both errors are normalised by a norm of the tensor, never elementwise: FP4
    holds *exact zeros*, so a per-element relative error is dominated by the
    denominator floor and says nothing about fidelity.
    """
    w = mx.array(dense_f32).astype(mx.bfloat16)
    qw, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    back = mx.dequantize(qw, scales, biases, group_size=group_size, bits=bits)
    mx.eval(qw, scales, biases, back)
    ref = mx.array(dense_f32)
    err = back.astype(mx.float32) - ref
    worst = float(mx.max(mx.abs(err)).item()) / (float(mx.max(mx.abs(ref)).item()) + 1e-12)
    frob = float(mx.sqrt(mx.sum(mx.square(err))).item()) / (
        float(mx.sqrt(mx.sum(mx.square(ref))).item()) + 1e-12
    )
    del back, err, ref, w
    return qw, scales, biases, worst, frob


# --------------------------------------------------------------------------- #
def find_trunk_snapshot(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-2bit-DQ"
        "/snapshots/*/"
    )
    for hit in sorted(glob.glob(pat)):
        if (Path(hit) / "model.safetensors.index.json").exists():
            return Path(hit).resolve()
    raise SystemExit("could not find the mlx-community 2bit-DQ snapshot in the HF cache")


def link_or_clone(src: Path, dst: Path) -> str:
    """Hardlink (free) with an APFS clone fallback; never touches the source."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    real = src.resolve()
    try:
        os.link(real, dst)
        return "link"
    except OSError:
        subprocess.run(["cp", "-c", str(real), str(dst)], check=True)
        return "clone"


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB ({n / 2**30:.2f} GiB)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mtp-shard", required=True,
                    help="upstream model-00046-of-00046.safetensors (mtp.0.* only)")
    ap.add_argument("--source", default=None,
                    help="MLX trunk snapshot dir (default: 2bit-DQ from the HF cache)")
    ap.add_argument("--out", required=True, help="merged model directory to create")
    ap.add_argument(
        "--bank",
        choices=("exact", "affine-q8"),
        default="exact",
        help="exact = mxfp4 expert repack + dense bf16 (ships); affine-q8 = the "
             "superseded lossy bank, kept only as the acceptance A/B arm",
    )
    ap.add_argument("--bits", type=int, default=8,
                    help="affine-q8 bank only")
    ap.add_argument("--group-size", type=int, default=64,
                    help="affine-q8 bank only")
    ap.add_argument("--source-etag", default=None,
                    help="upstream shard etag, recorded in the provenance block")
    ap.add_argument("--source-revision", default="main")
    args = ap.parse_args()

    t0 = time.time()
    trunk = find_trunk_snapshot(args.source)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    shard_path = Path(args.mtp_shard).expanduser().resolve()
    print(f"trunk snapshot : {trunk}")
    print(f"upstream shard : {shard_path} ({human(shard_path.stat().st_size)})")
    print(f"output dir     : {out}")

    st = SafeTensorsFile(shard_path)
    keys = sorted(st.header)
    if not all(k.startswith("mtp.0.") for k in keys):
        raise SystemExit("shard carries non-mtp tensors; refusing to guess")
    print(f"upstream mtp tensors: {len(keys)}")

    n_experts = 1 + max(
        int(k.split(".")[4]) for k in keys if k.startswith("mtp.0.ffn.experts.")
    )
    print(f"routed experts : {n_experts}")

    exact = args.bank == "exact"
    print(f"bank           : {args.bank}"
          + ("" if exact else f" (LOSSY A/B arm, q{args.bits}/gs{args.group_size})"))

    tensors: dict[str, mx.array] = {}
    quant_paths: dict[str, dict] = {}
    errs: list[tuple[str, float, float]] = []
    checked_fp4 = False

    def put(name, arr):
        tensors[f"mtp.0.{name}"] = arr

    # ---- plain tensors -----------------------------------------------------
    for src, (dst, kind) in PLAIN_RENAME.items():
        ref = st.f32(f"mtp.0.{src}")
        arr = to_mx(ref, kind)
        if exact:
            mx.eval(arr)
            if not np.array_equal(np.array(arr.astype(mx.float32)), ref):
                raise SystemExit(f"mtp.0.{src}: dtype-preserving copy is not exact")
        put(dst, arr)

    # ---- FP8-block dense projections --------------------------------------
    print("\nfp8 e4m3 x e8m0 dense projections:")
    for src, dst in FP8_STEMS.items():
        w = st.bytes_of(f"mtp.0.{src}.weight")
        s = st.bytes_of(f"mtp.0.{src}.scale")
        dense = dequant_fp8_block(w, s)
        lo, hi = check_group_max(dense, s, FP8_BLOCK, 448.0, True, src)
        if exact:
            arr, dtype_name, diff = dense_exact(dense, src)
            put(f"{dst}.weight", arr)
            errs.append((dst, diff, diff))
            print(f"  {src:26s} -> {dst:28s} {tuple(dense.shape)!s:16s} "
                  f"blockmax[{lo:6.1f},{hi:6.1f}] {dtype_name} exact")
        else:
            qw, sc, bi, worst, frob = quantize_stem(dense, args.group_size, args.bits)
            put(f"{dst}.weight", qw)
            put(f"{dst}.scales", sc)
            put(f"{dst}.biases", bi)
            quant_paths[f"mtp.0.{dst}"] = {
                "group_size": args.group_size, "bits": args.bits, "mode": "affine"
            }
            errs.append((dst, worst, frob))
            print(f"  {src:26s} -> {dst:28s} {tuple(dense.shape)!s:16s} "
                  f"blockmax[{lo:6.1f},{hi:6.1f}] q{args.bits} max_rel={worst:.2e}")
        del dense

    # ---- FP4-block routed experts -> stacked SwitchGLU ---------------------
    print("\nfp4 e2m1 x e8m0 routed experts:")
    for wsrc, dst in EXPERT_STEMS.items():
        qws, scs, bis = [], [], []
        worst = 0.0
        num = den = 0.0
        lo_all, hi_all = 1e9, 0.0
        for i in range(n_experts):
            stem = f"mtp.0.ffn.experts.{i}.{wsrc}"
            w = st.bytes_of(stem + ".weight")
            s = st.bytes_of(stem + ".scale")
            if exact:
                # the repack asserts mxfp4 == the LUT decode per tensor, which
                # subsumes the one-off check_fp4_against_mlx witness
                w32, s8, dense = repack_fp4_to_mxfp4(w, s, stem)
                lo, hi = check_group_max(dense, s, FP4_GROUP, 6.0, False, stem)
                lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
                qws.append(mx.array(w32))
                scs.append(mx.array(s8))
                del dense, w32
                continue
            dense = dequant_fp4_block(w, s)
            if not checked_fp4:
                check_fp4_against_mlx(w, s, dense)
                checked_fp4 = True
                print("  fp4 decode == MLX mxfp4 (bit-exact)")
            lo, hi = check_group_max(dense, s, FP4_GROUP, 6.0, False, stem)
            lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
            qw, sc, bi, we, fe = quantize_stem(dense, args.group_size, args.bits)
            qws.append(qw); scs.append(sc); bis.append(bi)
            worst = max(worst, we)
            nrm = float((dense.astype("float64") ** 2).sum())
            num += (fe ** 2) * nrm; den += nrm
            del dense
        put(f"{dst}.weight", mx.stack(qws))
        put(f"{dst}.scales", mx.stack(scs))
        if exact:
            # mxfp4 carries no zero point, so the module has no .biases tensor
            quant_paths[f"mtp.0.{dst}"] = dict(MXFP4_SPEC)
            errs.append((dst, 0.0, 0.0))
            print(f"  experts.*.{wsrc} -> {dst:28s} x{n_experts} "
                  f"groupmax[{lo_all:.1f},{hi_all:.1f}] mxfp4/gs{FP4_GROUP} "
                  "ALL EXACT")
        else:
            put(f"{dst}.biases", mx.stack(bis))
            quant_paths[f"mtp.0.{dst}"] = {
                "group_size": args.group_size, "bits": args.bits, "mode": "affine"
            }
            errs.append((dst, worst, (num / den) ** 0.5))
            print(f"  experts.*.{wsrc} -> {dst:28s} x{n_experts} "
                  f"groupmax[{lo_all:.1f},{hi_all:.1f}] q{args.bits} "
                  f"max_rel={worst:.2e}")
        del qws, scs, bis
        mx.clear_cache()

    st.close()

    # ---- write the new shard ----------------------------------------------
    shard_name = "model-00020-of-00020-mtp.safetensors"
    mx.eval(list(tensors.values()))
    mx.save_safetensors(str(out / shard_name), tensors,
                        metadata={"format": "pt"})
    new_size = (out / shard_name).stat().st_size
    print(f"\nwrote {shard_name}: {len(tensors)} tensors, {human(new_size)}")

    # ---- hardlink the trunk ------------------------------------------------
    linked, cloned = 0, 0
    rewritten = {"config.json", "model.safetensors.index.json"}
    for entry in sorted(trunk.iterdir()):
        if entry.name.startswith(".") or entry.name in rewritten:
            continue
        how = link_or_clone(entry, out / entry.name)
        linked += how == "link"
        cloned += how == "clone"
    print(f"trunk files: {linked} hardlinked, {cloned} APFS-cloned")

    # ---- index.json --------------------------------------------------------
    index = json.loads((trunk / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    before = len(wmap)
    for k in tensors:
        wmap[k] = shard_name
    index["metadata"]["total_size"] = int(index["metadata"]["total_size"]) + new_size
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"index: {before} -> {len(wmap)} entries, "
          f"total_size {human(index['metadata']['total_size'])}")

    # ---- config.json -------------------------------------------------------
    cfg = json.loads((trunk / "config.json").read_text())
    cfg["quantization"].update(quant_paths)
    # The trunk conversion left this at 1 while shipping no weights, which is
    # what mtplx's degrade-to-AR guard exists for; here it is finally truthful.
    cfg["num_nextn_predict_layers"] = 1
    cfg["mtp_provenance"] = {
        "source_repo": "deepseek-ai/DeepSeek-V4-Flash",
        "source_revision": args.source_revision,
        "source_shard": shard_path.name,
        "source_shard_sha256_etag": args.source_etag,
        "source_shard_bytes": shard_path.stat().st_size,
        "source_quantization": {
            "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [128, 128],
            "expert_dtype": "fp4", "expert_scale_group": FP4_GROUP,
        },
        "trunk_snapshot": str(trunk),
        "trunk_repo": "mlx-community/DeepSeek-V4-Flash-2bit-DQ",
        "mtp_shard": shard_name,
        "mtp_bank": args.bank,
        "mtp_representation": (
            {
                "routed_experts": {
                    "format": "mxfp4", "group_size": FP4_GROUP, "bits": 4,
                    "how": "byte repack of the source FP4 e2m1 payload + its e8m0 "
                           "scales (format translation, NOT a re-quantization)",
                    "weight_dtype": "uint32", "scales_dtype": "uint8",
                },
                "dense_projections": {
                    "format": "dense bfloat16", "quantization_entry": "removed",
                    "how": "bf16 represents every e4m3 x 2^k value exactly "
                           "(<= 4 significant bits, power-of-two block scale)",
                },
                "other": "source dtype preserved (bf16 norms / f32 sinks, "
                         "hyper-connections, router bias)",
            }
            if exact
            else {
                "all_stems": {
                    "format": "affine", "group_size": args.group_size,
                    "bits": args.bits,
                    "how": "SUPERSEDED lossy re-quantization; A/B arm only",
                },
            }
        ),
        "mtp_tensor_count": len(tensors),
        "mtp_shard_bytes": new_size,
        "built_by": "scripts/deepseek_v4_build_mtp_model.py",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fp4_decode_verified_against": "mlx mxfp4 dequantize (bit-exact)",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"config: quantization {len(quant_paths)} new per-path entries "
          f"({len(cfg['quantization']) - 3} total)")

    label = ("conversion error (relative to the exact FP8/FP4 source) -- all zero "
             "for the exact bank" if exact
             else "re-quantization error (relative to the exact FP8/FP4 source)")
    print(f"\n{label}:")
    for name, worst, frob in errs:
        print(f"  {name:34s} max_err/absmax={worst:.3e}  rel_frobenius={frob:.3e}")

    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
