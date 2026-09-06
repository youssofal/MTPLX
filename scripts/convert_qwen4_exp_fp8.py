#!/usr/bin/env python3
"""Build the Qwen3.8-Flash-Next MTPLX Optimized Speed pack from Qwen's FP8 repo.

Streaming per-tensor: fp8(e4m3, [128,128] block scale_inv) -> bf16 -> mixed
affine quant -> house-shaped pack. Nothing is ever fully resident: the biggest
working set is one stacked expert group (~3.4 GB bf16).

Outputs (house layout, mirrors the 27B Optimized Speed pack):
  model-0000X-of-0000N.safetensors   quantized trunk, house names
  model.safetensors.index.json       trunk index (sidecars NOT registered)
  model-vision.safetensors           bf16 vision tower, source names (preserved)
  mtp.safetensors                    quantized MTP head (attach lands later)
  ngram-table.safetensors            the 51B table, 4-bit/g32, ONE contiguous
                                     tensor streamed shard-by-shard; gathered
                                     lazily at runtime (SSD-resident)
  config.json                        model_type qwen4_exp + quantization dict
                                     keyed by module tree paths + provenance
  tokenizer/template/generation/preprocessor files copied verbatim

Fail-closed accounting (issue #263 pattern): every source tensor must land in
exactly one destination or be explicitly skipped with a reason; the script
refuses to finish otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# ----------------------------------------------------------------------------
# Recipe (Optimized Speed v1). Keys are OUTPUT module paths (house tree).
BASE_BITS, BASE_GROUP = 4, 32
EIGHT = {"bits": 8, "group_size": 64, "mode": "affine"}
NGRAM_BITS, NGRAM_GROUP = 4, 32
# Modules routed to BASE bits explicitly (empty in the speed recipe, where
# the sensitive set lives in EIGHT_SUFFIXES; the bare recipe moves that set
# here so it lands at 4-bit instead of falling through to bf16).
FOUR_SUFFIXES: tuple = ()
MTP_EXPERT_BITS, MTP_EXPERT_GROUP = 4, 32

EIGHT_SUFFIXES = (
    "embed_tokens",
    "lm_head",
    "linear_attn.out_proj",
    "mlp.gate",
    "shared_expert_gate",
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "shared_expert.down_proj",
    "indexer.index_qk_proj",
    # v1: QSA attention projections. Attribution receipt (2026-08-27,
    # attrib-v0.json): full_attention layers carry 7.4x the per-layer output
    # error of GDN at 4-bit (by-type rel_mse 8.7e-3 vs 1.2e-3; worst six
    # layers all QSA, layer 31 at 4.4e-2) while lm_head+mixer contribute
    # KLD 0.0003. These names exist only on full-attention layers (GDN
    # layers use linear_attn.*), so the change is surgical: +299MB/token
    # (~-5% AR) for the dominant share of the quality gap.
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
# Structural, stays bf16 (config carries explicit false entries).
KEEP_BF16_SUFFIXES = (
    "input_mix_weight_down",
    "input_mix_weight_up",
    "block_inject_weight",
    "ple.key_proj",
    "ple.value_proj",
)
# +1.0 shift families (stored mlx-native in the pack)
NORM_SHIFT_SUFFIXES = (
    ".q_norm.weight",
    ".k_norm.weight",
    ".q_layernorm.weight",
    ".k_layernorm.weight",
    ".hc_norm.weight",
    ".norm_key.weight",
    ".norm_query.weight",
    ".norm_conv.weight",
)

_FP8_LUT: np.ndarray | None = None


def fp8_e4m3_lut() -> np.ndarray:
    global _FP8_LUT
    if _FP8_LUT is None:
        vals = np.zeros(256, dtype=np.float32)
        for b in range(256):
            sign = -1.0 if b & 0x80 else 1.0
            exp = (b >> 3) & 0xF
            mant = b & 0x7
            if exp == 0:
                v = (mant / 8.0) * 2.0**-6
            elif exp == 15 and mant == 7:
                v = np.nan
            else:
                v = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
            vals[b] = sign * v
        _FP8_LUT = vals
    return _FP8_LUT


def read_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class SourceReader:
    """Random access to every tensor in the source repo via mmaps."""

    DTYPES = {
        "BF16": (np.uint16, 2),
        "F16": (np.uint16, 2),
        "F32": (np.float32, 4),
        "F8_E4M3": (np.uint8, 1),
        "I64": (np.int64, 8),
        "I32": (np.int32, 4),
        "U8": (np.uint8, 1),
    }

    def __init__(self, src: Path):
        self.src = src
        index = json.loads((src / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._mmaps: dict[str, np.memmap] = {}

    def names(self):
        return list(self.weight_map)

    def info(self, name: str):
        shard = self.weight_map[name]
        if shard not in self._headers:
            self._headers[shard] = read_header(self.src / shard)
        header, data_start = self._headers[shard]
        return header[name], shard, data_start

    def raw(self, name: str) -> tuple[np.ndarray, str]:
        info, shard, data_start = self.info(name)
        if shard not in self._mmaps:
            self._mmaps[shard] = np.memmap(self.src / shard, mode="r", dtype=np.uint8)
        mm = self._mmaps[shard]
        a, b = info["data_offsets"]
        dt, itemsize = self.DTYPES[info["dtype"]]
        arr = mm[data_start + a : data_start + b].view(dt).reshape(info["shape"])
        return arr, info["dtype"]

    def bf16(self, name: str) -> mx.array:
        """Tensor as bf16 mx.array, fp8-dequantized when needed."""
        arr, dtype = self.raw(name)
        if dtype == "F8_E4M3":
            w = mx.array(fp8_e4m3_lut()[np.ascontiguousarray(arr)])
            scale_name = name.replace(".weight", ".weight_scale_inv")
            if scale_name not in self.weight_map and ".ngram_embedding.shard_" in name:
                scale_name = name.split(".ngram_embedding.shard_", 1)[0] + ".ngram_embedding.weight_scale"
            s_arr, s_dtype = self.raw(scale_name)
            scale = mx.array(np.ascontiguousarray(s_arr))
            if s_dtype in ("BF16", "F16"):
                scale = scale.view(mx.bfloat16 if s_dtype == "BF16" else mx.float16)
            scale = scale.astype(mx.float32)
            if tuple(scale.shape) == (1,):
                return (w * scale).astype(mx.bfloat16)
            R, C = w.shape
            br, bc = scale.shape
            scale = mx.repeat(mx.repeat(scale, 128, axis=0)[:R], 128, axis=1)[:, :C]
            return (w * scale).astype(mx.bfloat16)
        out = mx.array(np.ascontiguousarray(arr))
        if dtype == "BF16":
            out = out.view(mx.bfloat16)
        elif dtype == "F16":
            out = out.view(mx.float16).astype(mx.bfloat16)
        return out


def quant_for(dest: str) -> dict | None:
    if any(dest.endswith(s) for s in KEEP_BF16_SUFFIXES):
        return None
    for s in EIGHT_SUFFIXES:
        if dest.endswith(s):
            return dict(EIGHT)
    for s in FOUR_SUFFIXES:
        if dest.endswith(s):
            return {"bits": BASE_BITS, "group_size": BASE_GROUP, "mode": "affine"}
    if dest.endswith(("switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj")):
        return {"bits": BASE_BITS, "group_size": BASE_GROUP, "mode": "affine"}
    if dest.endswith(("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                      "q_proj", "k_proj", "v_proj", "o_proj")):
        return {"bits": BASE_BITS, "group_size": BASE_GROUP, "mode": "affine"}
    return None  # norms, conv, buffers, A_log, dt_bias stay as-is


class ShardWriter:
    def __init__(self, out: Path, max_bytes: int = 4 << 30):
        self.out = out
        self.max_bytes = max_bytes
        self.current: dict[str, mx.array] = {}
        self.current_bytes = 0
        self.flushed: list[tuple[str, list[str]]] = []
        self.weight_map: dict[str, str] = {}
        self.total = 0

    def add(self, name: str, arr: mx.array):
        self.current[name] = arr
        self.current_bytes += arr.nbytes
        self.total += arr.nbytes
        if self.current_bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.current:
            return
        fname = f"model-{len(self.flushed) + 1:05d}.safetensors"
        mx.save_safetensors(str(self.out / fname), self.current)
        self.flushed.append((fname, list(self.current)))
        self.current = {}
        self.current_bytes = 0
        mx.clear_cache()

    def finalize(self):
        self.flush()
        n = len(self.flushed)
        for i, (fname, names) in enumerate(self.flushed):
            final = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            (self.out / fname).rename(self.out / final)
            for name in names:
                self.weight_map[name] = final
        index = {"metadata": {"total_size": self.total}, "weight_map": self.weight_map}
        (self.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


class NgramStreamWriter:
    """Streams the 4-bit table into one safetensors file with three contiguous
    tensors (weight/scales/biases), written shard-by-shard."""

    def __init__(self, path: Path, rows: int, dim: int, bits: int = NGRAM_BITS, group: int = NGRAM_GROUP):
        self.path = path
        self.rows = rows
        self.dim = dim
        self.bits = bits
        header = {
            "__metadata__": {
                "ngram_bits": str(bits),
                "ngram_group_size": str(group),
                "rows": str(rows),
                "dim": str(dim),
            },
        }
        if bits == 0:  # raw bf16 rows (dim not divisible by any mlx group size)
            header["ngram.weight"] = {
                "dtype": "BF16",
                "shape": [rows, dim],
                "data_offsets": [0, rows * dim * 2],
            }
            off = rows * dim * 2
            row_bytes = {"ngram.weight": dim * 2}
        else:
            packed_cols = dim * bits // 32
            groups = dim // group
            header["ngram.weight"] = {
                "dtype": "U32",
                "shape": [rows, packed_cols],
                "data_offsets": [0, rows * packed_cols * 4],
            }
            off = rows * packed_cols * 4
            for name in ("ngram.scales", "ngram.biases"):
                header[name] = {
                    "dtype": "BF16",
                    "shape": [rows, groups],
                    "data_offsets": [off, off + rows * groups * 2],
                }
                off += rows * groups * 2
            row_bytes = {
                "ngram.weight": packed_cols * 4,
                "ngram.scales": groups * 2,
                "ngram.biases": groups * 2,
            }
        blob = json.dumps(header).encode()
        pad = (-(len(blob)) % 8)
        blob += b" " * pad
        self._offsets = {
            name: 8 + len(blob) + header[name]["data_offsets"][0]
            for name in header
            if name != "__metadata__"
        }
        self._row_bytes = row_bytes
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            f.truncate(8 + len(blob) + off)
        self._fh = open(path, "r+b")
        self.rows_written = 0

    def append_rows(self, w: mx.array, s: mx.array = None, b: mx.array = None):
        n = w.shape[0]
        start = self.rows_written
        triples = (
            (("ngram.weight", w, np.uint16),)
            if self.bits == 0
            else (
                ("ngram.weight", w, np.uint32),
                ("ngram.scales", s, np.uint16),
                ("ngram.biases", b, np.uint16),
            )
        )
        for name, arr, np_dt in triples:
            if arr.dtype == mx.bfloat16:
                arr = arr.view(mx.uint16)
            data = np.ascontiguousarray(np.array(arr, copy=False)).astype(np_dt, copy=False)
            self._fh.seek(self._offsets[name] + start * self._row_bytes[name])
            self._fh.write(data.tobytes())
        self.rows_written += n

    def close(self):
        self._fh.close()
        assert self.rows_written == self.rows, (self.rows_written, self.rows)


def rename_trunk(name: str) -> str:
    if name.startswith("model.language_model."):
        name = name.replace("model.language_model.", "language_model.model.", 1)
    elif name == "lm_head.weight":
        name = "language_model.lm_head.weight"
    elif name.startswith("model."):  # flat text-only checkpoints (test harnesses)
        name = "language_model." + name
    return name


def main():
    global BASE_BITS, BASE_GROUP, EIGHT_SUFFIXES, FOUR_SUFFIXES
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--recipe", choices=("speed", "bare"), default="speed",
                    help="speed = Optimized Speed (4/g32 + 8-bit sensitive); "
                         "bare = Bare Speed (flat 4/g64, 8-bit ONLY for the "
                         "router and QSA indexer — routing integrity is "
                         "non-negotiable; ~-33%% weight bytes/token)")
    args = ap.parse_args()
    if args.recipe == "bare":
        BASE_BITS, BASE_GROUP = 4, 64
        EIGHT_SUFFIXES = ("mlp.gate", "indexer.index_qk_proj")
        FOUR_SUFFIXES = (
            "embed_tokens",
            "lm_head",
            "linear_attn.out_proj",
            "shared_expert_gate",
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
            "shared_expert.down_proj",
        )
    src, out = args.src, args.out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    reader = SourceReader(src)
    names = reader.names()
    accounted: dict[str, str] = {}
    quant_config: dict[str, object] = {
        "bits": BASE_BITS,
        "group_size": BASE_GROUP,
        "mode": "affine",
    }
    writer = ShardWriter(out)
    vision: dict[str, mx.array] = {}
    mtp_out: dict[str, mx.array] = {}

    def quantize_into(dest: str, w: mx.array, sink, recipe: dict | None):
        """Quantize w into the sink; returns the EFFECTIVE recipe (group size
        falls back to a divisor of the in-dim, or bf16 when none fits — only
        reachable on tiny test configs; production dims are 32-multiples)."""
        if recipe is not None and w.shape[-1] % recipe["group_size"]:
            g = next((x for x in (128, 64, 32) if x < recipe["group_size"] and w.shape[-1] % x == 0), None)
            recipe = {**recipe, "group_size": g} if g else None
        if recipe is None:
            sink(dest + "", w)
            return None
        wq, s, b = mx.quantize(w, group_size=recipe["group_size"], bits=recipe["bits"])
        mx.eval(wq, s, b)
        base = dest.rsplit(".weight", 1)[0] if dest.endswith(".weight") else dest
        sink(base + ".weight", wq)
        sink(base + ".scales", s)
        sink(base + ".biases", b)
        return recipe

    # ---- 1. n-gram table (streamed, shard order) ----------------------------
    shard_names = sorted(
        (n for n in names if ".ngram_embedding.shard_" in n and n.endswith(".weight")),
        key=lambda n: int(n.rsplit("shard_", 1)[1].split(".")[0]),
    )
    ple_prefix = None
    ng_bits, ng_group = NGRAM_BITS, NGRAM_GROUP
    if shard_names:
        ple_prefix = shard_names[0].split(".ple_embedding.")[0]
        info0, _, _ = reader.info(shard_names[0])
        dim = info0["shape"][1]
        total_rows = 0
        for n in shard_names:
            info, _, _ = reader.info(n)
            total_rows += info["shape"][0]
        ng_group = next((g for g in (NGRAM_GROUP, 64, 128) if dim % g == 0), 0)
        ng_bits = NGRAM_BITS if ng_group else 0
        print(
            f"[ngram] {len(shard_names)} shards, {total_rows} rows x {dim} "
            f"({'raw bf16' if not ng_group else f'{ng_bits}-bit/g{ng_group}'})",
            flush=True,
        )
        ng = NgramStreamWriter(out / "ngram-table.safetensors", total_rows, dim, bits=ng_bits, group=ng_group or NGRAM_GROUP)
        for i, n in enumerate(shard_names):
            w = reader.bf16(n)
            if ng_bits == 0:
                ng.append_rows(w.astype(mx.bfloat16))
                accounted[n] = "ngram"
                scale_name = n.replace(".weight", ".weight_scale_inv")
                if scale_name in reader.weight_map:
                    accounted[scale_name] = "ngram-scale"
                del w
                mx.clear_cache()
                if (i + 1) % 16 == 0:
                    print(f"[ngram] {i + 1}/{len(shard_names)}", flush=True)
                continue
            wq, s, b = mx.quantize(w, group_size=ng_group, bits=ng_bits)
            mx.eval(wq, s, b)
            ng.append_rows(wq, s, b)
            accounted[n] = "ngram"
            scale_name = n.replace(".weight", ".weight_scale_inv")
            if scale_name in reader.weight_map:
                accounted[scale_name] = "ngram-scale"
            del w, wq, s, b
            mx.clear_cache()
            if (i + 1) % 16 == 0:
                print(f"[ngram] {i + 1}/{len(shard_names)}", flush=True)
        ng.close()
        global_scale_name = ple_prefix + ".ple_embedding.ngram_embedding.weight_scale"
        if global_scale_name in reader.weight_map:
            accounted[global_scale_name] = "ngram-scale"

    # ---- 2. numbered experts -> stacked switch_mlp (trunk + mtp) ------------
    expert_re = re.compile(r"^(.*)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    groups: dict[tuple[str, str], int] = {}
    n_experts = 0
    for n in names:
        m = expert_re.match(n)
        if m:
            groups[(m.group(1), m.group(3))] = 1
            n_experts = max(n_experts, int(m.group(2)) + 1)
    print(f"[experts] {len(groups)} groups x {n_experts} experts", flush=True)
    for gi, (prefix, proj) in enumerate(sorted(groups)):
        parts = []
        for e in range(n_experts):
            src_name = f"{prefix}.mlp.experts.{e}.{proj}.weight"
            parts.append(reader.bf16(src_name))
            accounted[src_name] = "experts"
            scale_name = src_name.replace(".weight", ".weight_scale_inv")
            if scale_name in reader.weight_map:
                accounted[scale_name] = "experts-scale"
        stacked = mx.stack(parts)  # [E, out, in] — HF per-expert Linear layout
        del parts
        is_mtp = prefix.startswith("mtp.")
        dest_prefix = prefix if is_mtp else rename_trunk(prefix + ".x")[: -len(".x")]
        dest = f"{dest_prefix}.mlp.switch_mlp.{proj}.weight"
        recipe = {
            "bits": MTP_EXPERT_BITS if is_mtp else BASE_BITS,
            "group_size": MTP_EXPERT_GROUP if is_mtp else BASE_GROUP,
            "mode": "affine",
        }
        sink = (lambda k, v: mtp_out.__setitem__(k, v)) if is_mtp else writer.add
        eff = quantize_into(dest, stacked, sink, recipe)
        if not is_mtp:
            quant_config[dest.rsplit(".weight", 1)[0]] = eff if eff is not None else False
        del stacked
        mx.clear_cache()
        if (gi + 1) % 24 == 0:
            print(f"[experts] {gi + 1}/{len(groups)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2b. packed experts (bf16 hub repo) -> stacked switch_mlp -----------
    # The hub bf16 repo ships one 3D tensor per group: gate_up_proj
    # [E, 2*inter, hidden] (Linear [out, in] halves stacked on the out axis)
    # and down_proj [E, hidden, inter]. transformers save_pretrained instead
    # writes the runtime bmm orientation (gate_up [E, hidden, 2*inter], down
    # [E, inter, hidden]); keyed on which axis equals hidden_size.
    packed_re = re.compile(r"^(.*)\.mlp\.experts\.(gate_up_proj|down_proj)(?:\.weight)?$")
    hid = None
    cfg_path = src / "config.json"
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        hid = c.get("text_config", c).get("hidden_size")
    packed_names = sorted(n for n in names if packed_re.match(n) and n not in accounted)
    if packed_names:
        print(f"[experts-packed] {len(packed_names)} packed tensors", flush=True)
    for gi, n in enumerate(packed_names):
        m = packed_re.match(n)
        prefix, kind = m.group(1), m.group(2)
        v = reader.bf16(n)
        accounted[n] = "experts"
        is_mtp = prefix.startswith("mtp.")
        dest_prefix = prefix if is_mtp else rename_trunk(prefix + ".x")[: -len(".x")]
        recipe = {
            "bits": MTP_EXPERT_BITS if is_mtp else BASE_BITS,
            "group_size": MTP_EXPERT_GROUP if is_mtp else BASE_GROUP,
            "mode": "affine",
        }
        sink = (lambda k, w: mtp_out.__setitem__(k, w)) if is_mtp else writer.add
        if kind == "gate_up_proj":
            if hid is not None and v.shape[1] == hid:
                gate, up = mx.split(v, 2, axis=-1)
                gate = gate.swapaxes(1, 2)
                up = up.swapaxes(1, 2)
            else:
                gate, up = mx.split(v, 2, axis=1)
            halves = (("gate_proj", gate), ("up_proj", up))
        else:
            if hid is not None and v.shape[2] == hid:
                v = v.swapaxes(1, 2)
            halves = (("down_proj", v),)
        for proj, w in halves:
            dest = f"{dest_prefix}.mlp.switch_mlp.{proj}.weight"
            eff = quantize_into(dest, w, sink, recipe)
            if not is_mtp:
                quant_config[dest.rsplit(".weight", 1)[0]] = eff if eff is not None else False
        del v, halves
        mx.clear_cache()
        if (gi + 1) % 24 == 0:
            print(f"[experts-packed] {gi + 1}/{len(packed_names)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 3. everything else -------------------------------------------------
    for n in names:
        if n in accounted:
            continue
        if n.endswith(".weight_scale_inv"):
            continue  # consumed alongside its weight below
        if n.startswith("model.visual."):
            vision[n] = reader.bf16(n)
            accounted[n] = "vision"
            continue
        if ".ple." in n and n.endswith("conv1d.weight") and not n.startswith("mtp."):
            w = reader.bf16(n).moveaxis(2, 1)
            writer.add(rename_trunk(n).replace("ple.conv1d.weight", "ple.conv_weight"), w)
            accounted[n] = "trunk"
            continue

        is_mtp = n.startswith("mtp.")
        dest = n if is_mtp else rename_trunk(n)
        w = reader.bf16(n)
        if n.endswith("linear_attn.conv1d.weight") and w.shape[-1] != 1:
            w = w.moveaxis(2, 1)
        if w.ndim == 1 and any(dest.endswith(s) for s in NORM_SHIFT_SUFFIXES):
            w = w + 1.0
        recipe = quant_for(dest.rsplit(".weight", 1)[0]) if dest.endswith(".weight") and w.ndim >= 2 else None
        if is_mtp:
            quantize_into(dest, w, lambda k, v: mtp_out.__setitem__(k, v), recipe)
        else:
            eff = quantize_into(dest, w, writer.add, recipe)
            if eff is not None:
                quant_config[dest.rsplit(".weight", 1)[0]] = eff
            elif dest.endswith(".weight") and w.ndim >= 2:
                quant_config[dest.rsplit(".weight", 1)[0]] = False
        accounted[n] = "mtp" if is_mtp else "trunk"
        # scale companions of converted fp8 tensors
        scale_name = n.replace(".weight", ".weight_scale_inv")
        if scale_name in reader.weight_map:
            accounted[scale_name] = "scale"
        del w

    # scale_inv stragglers were all consumed with their weights
    for n in names:
        if n not in accounted and n.endswith(".weight_scale_inv"):
            base = n.replace(".weight_scale_inv", ".weight")
            if base in accounted:
                accounted[n] = "scale"

    # ---- 4. fail-closed accounting (issue #263 pattern) ---------------------
    missing = [n for n in names if n not in accounted]
    if missing:
        print(f"FATAL: {len(missing)} unaccounted source tensors, e.g. {missing[:8]}")
        sys.exit(3)

    writer.finalize()
    if vision:
        mx.save_safetensors(str(out / "model-vision.safetensors"), vision)
    if mtp_out:
        mx.save_safetensors(str(out / "mtp.safetensors"), mtp_out)

    # ---- 5. config + sidecar files ------------------------------------------
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)  # the fp8 recipe does not describe this pack
    tcfg = cfg["text_config"] if "text_config" in cfg else cfg  # flat text-only test ckpts
    tcfg["ngram_sidecar"] = True
    cfg["quantization"] = quant_config
    cfg["quantization_config"] = quant_config
    cfg["mlx_lm_extra_tensors"] = {
        "mtp_file": "mtp.safetensors",
        "ngram_file": "ngram-table.safetensors",
        "vision_file": "model-vision.safetensors",
    }
    cfg["mtplx_source_repo"] = f"Qwen/{src.name}" if src.name.startswith("Qwen") else src.name
    cfg["mtplx_recipe"] = {
        "base": {"bits": BASE_BITS, "group_size": BASE_GROUP},
        "eight_bit": list(EIGHT_SUFFIXES),
        "kept_bf16": list(KEEP_BF16_SUFFIXES),
        "ngram": {"bits": ng_bits, "group_size": ng_group},
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "README.md",
        "LICENSE",
    ):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)

    print(f"DONE in {time.time() - t0:.0f}s -> {out}", flush=True)
    for kind in ("trunk", "experts", "ngram", "vision", "mtp"):
        cnt = sum(1 for v in accounted.values() if v == kind)
        print(f"  {kind}: {cnt} tensors")


if __name__ == "__main__":
    main()
