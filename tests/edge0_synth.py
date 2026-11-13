"""Builds tiny synthetic Edge0 pack folders for tests.

Test-side encoders (NVFP4, Q4_0, Q8_0, a minimal GGUF v3 writer) and slow
scalar reference decoders written straight from the format description, so the
vectorised readers are checked against an independent implementation.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_LAYERS = 2
N_EXPERTS = 8
HIDDEN = 128
EXPERT_FF = 64
LATENT = 128
N_GROUPS = 3

KV = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]


def ue4m3_scalar(b: int) -> float:
    if b in (0, 0x7F):
        return 0.0
    e, m = b >> 3, b & 7
    return m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)


# -- encoders ------------------------------------------------------------------


def pack_nvfp4(codes: np.ndarray, scales: np.ndarray) -> bytes:
    """codes uint8[n] (0..15), scales uint8[n // 16] -> 36-byte blocks of 64 values."""
    codes = codes.reshape(-1, 4, 16)
    scales = scales.reshape(-1, 4)
    out = bytearray()
    for blk, sc in zip(codes, scales):
        out += bytes(sc.tolist())
        for s in range(4):
            for j in range(8):
                out.append(int(blk[s, j]) | (int(blk[s, j + 8]) << 4))
    return bytes(out)


def pack_q4_0(codes: np.ndarray, d: np.ndarray) -> bytes:
    codes = codes.reshape(-1, 32)
    out = bytearray()
    for blk, dd in zip(codes, d):
        out += np.float16(dd).tobytes()
        out += bytes(int(blk[j]) | (int(blk[j + 16]) << 4) for j in range(16))
    return bytes(out)


def pack_q8_0(q: np.ndarray, d: np.ndarray) -> bytes:
    q = q.reshape(-1, 32).astype(np.int8)
    out = bytearray()
    for blk, dd in zip(q, d):
        out += np.float16(dd).tobytes() + blk.tobytes()
    return bytes(out)


# -- scalar reference decoders ---------------------------------------------------


def ref_nvfp4(raw: bytes) -> list[float]:
    out = []
    for b in range(0, len(raw), 36):
        d, qs = raw[b : b + 4], raw[b + 4 : b + 36]
        for s in range(4):
            lo = [KV[qs[8 * s + j] & 15] for j in range(8)]
            hi = [KV[qs[8 * s + j] >> 4] for j in range(8)]
            out += [v * ue4m3_scalar(d[s]) * 0.5 for v in lo + hi]
    return out


def ref_q4_0(raw: bytes) -> list[float]:
    out = []
    for b in range(0, len(raw), 18):
        d = float(np.frombuffer(raw[b : b + 2], "<f2")[0])
        qs = raw[b + 2 : b + 18]
        out += [((qs[j] & 15) - 8) * d for j in range(16)] + [((qs[j] >> 4) - 8) * d for j in range(16)]
    return out


def ref_q8_0(raw: bytes) -> list[float]:
    out = []
    for b in range(0, len(raw), 34):
        d = float(np.frombuffer(raw[b : b + 2], "<f2")[0])
        out += [d * v for v in np.frombuffer(raw[b + 2 : b + 34], np.int8).tolist()]
    return out


# -- GGUF writer -----------------------------------------------------------------


def _gstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _gvalue(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return struct.pack("<IQ", 10, v) if v >= 2**31 else struct.pack("<Ii", 5, v)
    if isinstance(v, float):
        return struct.pack("<If", 6, v)
    if isinstance(v, str):
        return struct.pack("<I", 8) + _gstr(v)
    if isinstance(v, list):
        inner = [_gvalue(x) for x in v]
        etype = inner[0][:4] if inner else struct.pack("<I", 5)
        return struct.pack("<I", 9) + etype + struct.pack("<Q", len(v)) + b"".join(x[4:] for x in inner)
    raise TypeError(type(v))


def write_gguf(path: Path, metadata: dict, tensors: list[tuple[str, tuple[int, ...], int, bytes]]) -> None:
    """tensors: (name, numpy row-major shape, ggml type, raw bytes). Dims are written reversed."""
    align = int(metadata.get("general.alignment", 32))
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for k, v in metadata.items():
        head += _gstr(k) + _gvalue(v)
    rel = 0
    blobs = []
    for name, shape, gtype, data in tensors:
        head += _gstr(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in reversed(shape))
        head += struct.pack("<IQ", gtype, rel)
        pad = -len(data) % align
        blobs.append(data + b"\0" * pad)
        rel += len(data) + pad
    head += b"\0" * (-len(head) % align)
    path.write_bytes(bytes(head) + b"".join(blobs))


# -- the synthetic pack ----------------------------------------------------------


@dataclass
class SynthPack:
    path: Path
    blobs: dict = field(default_factory=dict)  # (layer, expert) -> raw blob bytes
    tails: dict = field(default_factory=dict)  # (layer, expert) -> (sg, su, sd)
    dense: dict = field(default_factory=dict)  # (fmt, layer, "d"|"u") -> raw bytes
    labels: np.ndarray | None = None


def make_pack(
    root: Path,
    *,
    dense_fmt: str = "4",
    both_dense: bool = True,
    seed: int = 0,
    build: str = "NVFP4",
    n_layers: int = N_LAYERS,
    n_experts: int = N_EXPERTS,
    hidden: int = HIDDEN,
    expert_ff: int = EXPERT_FF,
    latent: int = LATENT,
    n_groups: int = N_GROUPS,
    skeleton: bool = True,
) -> SynthPack:
    """The latent files of a synthetic folder; ``skeleton=False`` leaves ``model-skeleton.gguf`` to the caller
    (a full shell skeleton, ``tests/test_edge0_shell.build``), whose hidden size must be ``hidden``."""
    N_LAYERS, N_EXPERTS, HIDDEN, EXPERT_FF, LATENT, N_GROUPS = n_layers, n_experts, hidden, expert_ff, latent, n_groups  # noqa: N806
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    pack = SynthPack(root)
    experts = bytearray()
    blob_bytes = 3 * (EXPERT_FF * LATENT // 64 * 36) + 16
    for layer in range(N_LAYERS):
        for e in range(N_EXPERTS):
            parts = b""
            for rows, cols in ((EXPERT_FF, LATENT), (EXPERT_FF, LATENT), (LATENT, EXPERT_FF)):
                n = rows * cols
                # scale bytes avoid 0x7F (NaN) and 0 only partially so decoding is non-trivial
                parts += pack_nvfp4(rng.integers(0, 16, n, dtype=np.uint8), rng.integers(1, 0x7F, n // 16, dtype=np.uint8))
            tail = tuple(float(x) for x in rng.uniform(1e-4, 1e-2, 3).astype(np.float32))
            blob = parts + struct.pack("<4f", *tail, 0.0)
            assert len(blob) == blob_bytes
            pack.blobs[(layer, e)] = blob
            pack.tails[(layer, e)] = tail
            experts += blob
    (root / "experts.bin").write_bytes(bytes(experts))
    lines = [f"# strata native experts v6: latent A-only (expert_embd {LATENT}, expert_ff {EXPERT_FF})"]
    lines += [f"{layer} 40 40 {layer * N_EXPERTS * blob_bytes} {blob_bytes}" for layer in range(N_LAYERS)]
    (root / "native_experts.txt").write_text("\n".join(lines) + "\n")
    pack.labels = rng.integers(0, N_GROUPS, (N_LAYERS, N_EXPERTS), dtype=np.int32)
    pack.labels[0, :N_GROUPS] = np.arange(N_GROUPS)
    (root / "labels.bin").write_bytes(pack.labels.astype("<i4").tobytes())

    formats = ["4", "8"] if both_dense else [dense_fmt]
    for fmt in formats:
        data = bytearray()
        for layer in range(N_LAYERS):
            for key, rows, cols in (("d", N_GROUPS * LATENT, HIDDEN), ("u", N_GROUPS * HIDDEN, LATENT)):
                n = rows * cols
                d = rng.uniform(0.001, 0.02, n // 32).astype(np.float16)
                if fmt == "4":
                    raw = pack_q4_0(rng.integers(0, 16, n, dtype=np.uint8), d)
                else:
                    raw = pack_q8_0(rng.integers(-127, 128, n, dtype=np.int8), d)
                pack.dense[("q" + fmt, layer, key)] = raw
                data += raw
        (root / f"latent_dense_q{fmt}.bin").write_bytes(bytes(data))

    meta = {
        "model_name": "synthetic",
        "layers": N_LAYERS,
        "experts": N_EXPERTS,
        "latent_dim": LATENT,
        "build": build,
        "env": {"STRATA_LATENT_DENSE_FMT": dense_fmt},
    }
    (root / "model.json").write_text(json.dumps(meta))
    if skeleton:
        write_skeleton(root / "model-skeleton.gguf", rng)
    write_gguf(
        root / "ple-fp8.gguf",
        {"general.architecture": "strata-ple", "strata.ple.scale": 0.25},
        [("per_layer_token_embd.weight", (4, 8), 24, bytes(range(32)))],
    )
    return pack


def write_skeleton(path: Path, rng) -> None:
    meta = {"general.architecture": "qwen4exp", "qwen4exp.embedding_length": HIDDEN, "qwen4exp.block_count": N_LAYERS}
    router = rng.standard_normal((N_EXPERTS, HIDDEN)).astype(np.float32)
    tensors = [
        ("blk.0.ffn_gate_inp.weight", router.shape, 0, router.tobytes()),
        ("token_embd.weight", (4, HIDDEN), 30, ((rng.standard_normal(4 * HIDDEN).astype(np.float32).view(np.uint32) >> 16).astype("<u2")).tobytes()),
    ]
    write_gguf(path, meta, tensors)


# -- a complete tiny Edge0 folder (shell + latent + n-gram + tokenizer) ------------

QWEN_PRE_RE = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)
EDGE0_PRE_RE = QWEN_PRE_RE.replace(
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)", "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
)
TEMPLATE = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"


def write_tokenizers(edge0_root: Path, base_root: Path, vocab_size: int) -> None:
    """A Flash-Next-shaped tokenizer pair: Edge0's GGUF-style folder and a Hugging Face folder that matches it."""
    n_bpe = vocab_size - 4
    names = ["a", "b", "ab", "c", "d", "cd"] + [f"t{i}" for i in range(6, n_bpe)]
    bpe = {name: i for i, name in enumerate(names)}
    merges = ["a b", "c d"]
    added = [("<|endoftext|>", n_bpe, True), ("<|im_start|>", n_bpe + 1, True), ("<|im_end|>", n_bpe + 2, True),
             ("<think>", n_bpe + 3, False)]
    tok = edge0_root / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    full = dict(bpe, **{c: i for c, i, _ in added})
    (tok / "vocab.json").write_text(json.dumps(full))
    (tok / "merges.txt").write_text("\n".join(merges) + "\n")
    (tok / "chat_template.jinja").write_text(TEMPLATE)
    (tok / "token_type.json").write_text(json.dumps([1] * n_bpe + [3 if s else 4 for _, _, s in added]))
    (tok / "tokenizer.json").write_text(json.dumps({
        "model": "gpt2", "pre": "qwen35", "vocab_size": vocab_size, "n_merges": len(merges),
        "special_ids": {"tokenizer.ggml.eos_token_id": n_bpe + 2, "tokenizer.ggml.padding_token_id": n_bpe,
                        "tokenizer.ggml.bos_token_id": n_bpe},
        "add_bos_token": False, "pre_pattern": EDGE0_PRE_RE,
    }))
    base_root.mkdir(parents=True, exist_ok=True)
    (base_root / "vocab.json").write_text(json.dumps(bpe))
    (base_root / "merges.txt").write_text("\n".join(merges) + "\n")
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

    hf = Tokenizer(models.BPE(vocab=bpe, merges=[tuple(m.split()) for m in merges]))
    hf.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_PRE_RE), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    for content, _i, special in added:  # ids follow the BPE vocabulary in order
        hf.add_tokens([AddedToken(content, special=special, normalized=False)])
    (base_root / "tokenizer.json").write_text(hf.to_str())
    (base_root / "tokenizer_config.json").write_text(json.dumps(
        {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "chat_template": TEMPLATE}))
    (base_root / "generation_config.json").write_text(json.dumps({"temperature": 1.0, "top_k": 20, "top_p": 0.95}))


def make_folder(root: Path, base_tok: Path, *, seed: int = 0) -> SynthPack:
    """A full tiny Edge0 folder: the shell skeleton of ``tests/test_edge0_shell.build`` (4 layers, hidden 64,
    4 experts, top-2, PLE on layer 1), latent files of that geometry, an FP8 n-gram table of the size the
    model asks for, and a tokenizer pair (Edge0's in ``root/tokenizer``, the matching base one in ``base_tok``)."""
    from tests import test_edge0_shell as shell_tests

    g = shell_tests.G
    pack = make_pack(root, n_layers=g["L"], n_experts=g["E"], hidden=g["H"], expert_ff=64, latent=64, n_groups=3,
                     skeleton=False, seed=seed)
    shell_tests.build(root, seed=seed)
    # n-gram rows: consecutive primes from base 100 for 4 heads (101 + 103 + 107 + 109 = 420), padded to 512;
    # width = ple_embed_dim / heads = 64 / 4
    rng = np.random.default_rng(seed + 1)
    codes = rng.integers(0, 0x7F, 512 * 16, dtype=np.uint8)  # positive FP8 E4M3, no NaN
    codes[::2] |= 0x80
    write_gguf(
        root / "ple-fp8.gguf",
        {"general.architecture": "strata-ple", "strata.ple.format": "f8_e4m3", "strata.ple.scale": 0.0625},
        [("per_layer_token_embd.weight", (512, 16), 24, codes.tobytes())],
    )
    write_tokenizers(root, base_tok, g["V"])
    return pack


# -- a tiny Flash Next MTP draft head for the forged synthetic pack ---------------


def write_mtp_head(path: Path, config: dict, *, seed: int = 0, quantize: bool = True, dtype: str = "bfloat16") -> Path:
    """A ``mtp.safetensors`` shaped like the base Flash Next head for ``config`` (a forged pack's config.json).

    Built from :class:`mtplx.models.qwen4_exp.Qwen4ExpMTP` with seeded random weights, stored the way the
    base packs ship it: every key under ``mtp.``, the two ``pre_fc`` norms zero-centred (the loader adds
    +1), floating tensors in ``dtype``, and (``quantize``) the attention and MoE projections 4-bit g32 as in
    the base recipe while the fc and hyper-connection weights stay unquantized. ``path`` is a folder (the
    file is ``mtp.safetensors`` in it) or a file path.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mtplx.models.qwen4_exp import Qwen4ExpMTP, TextArgs

    path = Path(path)
    if path.suffix != ".safetensors":
        path = path / "mtp.safetensors"
    mx.random.seed(seed)
    head = Qwen4ExpMTP(TextArgs.from_dict(dict(config["text_config"])))
    if quantize:
        def predicate(p, m):
            if not (hasattr(m, "to_quantized") and (".self_attn." in f".{p}." or ".mlp." in f".{p}.")):
                return False
            if p.endswith(("mlp.gate", "shared_expert_gate")) and m.weight.shape[-1] % 64 == 0:
                return {"group_size": 64, "bits": 8}  # the base recipe keeps the routers 8-bit
            return m.weight.shape[-1] % 32 == 0

        nn.quantize(head, group_size=32, bits=4, class_predicate=predicate)
    target = getattr(mx, dtype)
    out = {}
    rng = np.random.default_rng(seed + 7)
    for name, arr in tree_flatten(head.parameters()):
        if name in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
            arr = mx.array(rng.normal(0.0, 0.05, arr.shape).astype(np.float32))  # zero-centred: (1 + w)
        if arr.dtype in (mx.float32, mx.float16, mx.bfloat16):
            arr = arr.astype(target)
        out[f"mtp.{name}"] = arr
    mx.eval(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), out, metadata={"format": "mlx"})
    return path
