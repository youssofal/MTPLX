"""Build a loadable MTPLX pack from an Edge0 latent-MoE model folder.

``build_pack(edge0_dir, out_dir)`` writes::

    model-shell-NNNNN.safetensors   the qwen4_exp shell (mtplx.edge0.shell), ~2 GB shards
    model-latent-LLL.safetensors    one layer's latent MoE arrays (mtplx.edge0.latent_convert)
    model.safetensors.index.json    every key -> its file
    latent-moe.index.json           the latent converter's own index
    config.json                     model_type edge0_latent (mtplx.models.edge0_latent)
    tokenizer files                 copied from a Flash Next tokenizer folder after checking
                                    that its vocabulary, merges and special tokens are Edge0's
    chat_template.jinja, generation_config.json
    edge0-forge.json                provenance: source revision and file hashes, settings, accounting

The n-gram table is not copied: ``config.json`` points at the folder's
``ple-fp8.gguf`` (``mtplx_ngram_source``), which the model reads in place.
Shard names match mlx-lm's ``model*.safetensors`` glob, so the stock loader
reads them.

Resumable: the shell is written as a unit (temporary directory, then
renamed), latent layers one file at a time (temporary name, then renamed), and
a rerun with the same source and settings skips what is complete and
verified. A rerun whose source fingerprint or settings differ refuses unless
``force=True``. The last step instantiates the model class lazily, applies the
config's quantization the way ``mlx_lm.utils.load_model`` does, and checks
that the shard headers hold exactly its parameters with the right shapes (and,
for the latent arrays, dtypes); every file's size must match its header.

``python -m mtplx.edge0.forge SRC OUT [--tokenizer-from DIR]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import latent_convert, shell
from .pack import Edge0Pack

FORGE_VERSION = 1
STATE_NAME = "edge0-forge.json"
SHELL_PATTERN = "model-shell-{index:05d}.safetensors"
LATENT_PATTERN = "model-latent-{layer:03d}.safetensors"
INDEX_NAME = "model.safetensors.index.json"
SHELL_SHARD_BYTES = 2 * 2**30
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
SPECIAL_TYPES = {True: 3, False: 4}  # token_type.json (llama.cpp): 3 control, 4 user-defined
SOURCE_FILES = (
    "model.json",
    "native_experts.txt",
    "experts.bin",
    "labels.bin",
    "latent_dense_q4.bin",
    "latent_dense_q8.bin",
    "model-skeleton.gguf",
    "ple-fp8.gguf",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/token_type.json",
)
_SMALL_HASH_BYTES = 64 * 2**20


class ForgeError(RuntimeError):
    """The pack cannot be built, or what was built does not account for the model exactly."""


# --------------------------------------------------------------------------- helpers


def _log(msg: str) -> None:
    print(f"[edge0-forge] {msg}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict:
    """Size and a hash: the Hugging Face blob name when the file is a cache symlink (an LFS sha256 or git
    blob id, free), else sha256 for files up to 64 MiB, else nothing."""
    rec: dict = {"size": os.stat(path).st_size}
    if path.is_symlink():
        blob = os.readlink(path).rsplit("/", 1)[-1]
        rec["hf_blob"] = blob
        if len(blob) == 64:
            rec["sha256"] = blob
    if "sha256" not in rec and rec["size"] <= _SMALL_HASH_BYTES:
        rec["sha256"] = _sha256(path)
    return rec


def _source_fingerprint(src: Path) -> dict:
    files = {name: _file_record(src / name) for name in SOURCE_FILES if (src / name).exists()}
    rev = None
    parts = src.resolve().parts if not src.is_symlink() else src.parts
    if "snapshots" in src.parts:
        rev = src.parts[src.parts.index("snapshots") + 1]
    elif "snapshots" in parts:
        rev = parts[parts.index("snapshots") + 1]
    return {"path": str(src), "hf_revision": rev, "files": files}


def _mtplx_commit() -> str | None:
    try:
        repo = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """(header dict, data start offset); validates that the file is exactly header + data."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ForgeError(f"{path}: truncated safetensors header")
        n = int.from_bytes(raw, "little")
        header = json.loads(f.read(n))
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    size = os.stat(path).st_size
    if size != 8 + n + end:
        raise ForgeError(f"{path}: {size} bytes on disk, header describes {8 + n + end}")
    return header, 8 + n


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- config and tokenizer


def pack_config(pack: Edge0Pack, *, q8: str, dense_format: str, ngram_file: str) -> dict:
    """``config.json`` for :mod:`mtplx.models.edge0_latent`."""
    from mtplx.models import edge0_latent

    cfg = shell.shell_config(pack, q8=q8)
    tc = cfg["text_config"]
    if int(tc["num_experts"]) != pack.n_experts or int(tc["hidden_size"]) != pack.hidden:
        raise ForgeError(
            f"skeleton says {tc['num_experts']} experts / hidden {tc['hidden_size']}, "
            f"the latent files {pack.n_experts} / {pack.hidden}"
        )
    if int(tc["num_hidden_layers"]) != pack.n_layers:
        raise ForgeError(f"skeleton has {tc['num_hidden_layers']} layers, the latent files {pack.n_layers}")
    tc["model_type"] = edge0_latent.TEXT_MODEL_TYPE
    tc["ngram_sidecar"] = True
    out = {
        "architectures": [edge0_latent.ARCHITECTURE],
        "model_type": edge0_latent.MODEL_TYPE,
        "tie_word_embeddings": cfg["tie_word_embeddings"],
        "text_config": tc,
        "edge0_latent": {
            "format_version": latent_convert.FORMAT_VERSION,
            "latent_dim": pack.latent_dim,
            "expert_ff": pack.expert_ff,
            "n_groups": pack.n_groups,
            "n_experts": pack.n_experts,
            "dense_format": dense_format,
            "dense_bits": latent_convert.DENSE_BITS[dense_format],
            "top_k": int(tc["num_experts_per_tok"]),
            "expert_format": pack.expert_format,
            "model_name": pack.model_name,
        },
        "edge0_activation_dtype": "float32",
        "mtplx_ngram_source": {"backend": "edge0_fp8_gguf", "file": ngram_file},
    }
    for key in ("image_token_id", "quantization", "quantization_config"):
        if key in cfg:
            out[key] = cfg[key]
    return out


def _check_ngram(pack: Edge0Pack, cfg: dict, gguf: Path) -> dict:
    """Open the FP8 table with the row count and width the model will ask for."""
    import math

    from mtplx.models.qwen4_exp import _head_vocab_layout

    from . import ngram

    tc = cfg["text_config"]
    heads = (int(tc["ngram_size"]) - 1) * int(tc["heads_per_ngram"])
    out = {}
    for ple_index, _layer in enumerate(tc.get("ple_layer_ids") or []):
        _sizes, _offsets, total = _head_vocab_layout(int(tc["ngram_vocab_size_base"]), heads, ple_index)
        div = int(tc.get("make_ngram_vocab_size_divisible_by", 128))
        rows = math.ceil(total / div) * div
        dim = int(tc["ple_embed_dim"]) // heads
        _entries, scale = ngram.open_table(gguf, rows, dim)
        out[str(ple_index)] = {"rows": rows, "dim": dim, "scale": scale}
    return out


def _looks_special(text: str) -> bool:
    """llama.cpp's ``does_token_look_special``: such added tokens become CONTROL even when not flagged special."""
    return text in ("<pad>", "<mask>") or (text.startswith("<|") and text.endswith("|>"))


def check_tokenizer(edge0_tok: Path, base: Path) -> dict:
    """Prove that ``base``'s Hugging Face tokenizer is Edge0's: vocabulary, merges, special tokens, regex."""
    problems = []
    ev = json.loads((edge0_tok / "vocab.json").read_text())
    bv = json.loads((base / "vocab.json").read_text())
    tj = json.loads((base / "tokenizer.json").read_text())
    if tj["model"]["vocab"] != bv:
        problems.append("base tokenizer.json vocab differs from its vocab.json")
    if any(ev.get(k) != v for k, v in bv.items()):
        problems.append("BPE vocabulary differs")
    em = (edge0_tok / "merges.txt").read_text().splitlines()
    bm = (base / "merges.txt").read_text().splitlines()
    tm = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
    if em != bm or em != tm:
        problems.append("merges differ")
    meta = json.loads((edge0_tok / "tokenizer.json").read_text())
    types = json.loads((edge0_tok / "token_type.json").read_text())
    added = {a["content"]: a for a in tj["added_tokens"]}
    for content, a in added.items():
        if ev.get(content) != a["id"]:
            problems.append(f"added token {content!r}: id {a['id']} vs Edge0 {ev.get(content)}")
        elif types[a["id"]] != SPECIAL_TYPES[bool(a["special"]) or _looks_special(content)]:
            problems.append(f"added token {content!r}: special={a['special']} but Edge0 type {types[a['id']]}")
    rest = {k: v for k, v in ev.items() if k not in bv and k not in added}
    if any(not (k.startswith("[PAD") and types[v] == 5) for k, v in rest.items()):
        problems.append("Edge0 has non-padding tokens the base tokenizer lacks")
    if len(ev) != int(meta["vocab_size"]):
        problems.append("Edge0 vocab.json size disagrees with its tokenizer.json")
    special = meta.get("special_ids", {})
    ids = {"eos": special.get("tokenizer.ggml.eos_token_id"), "bos": special.get("tokenizer.ggml.bos_token_id"),
           "pad": special.get("tokenizer.ggml.padding_token_id")}
    tc = json.loads((base / "tokenizer_config.json").read_text())
    by_content = {a["content"]: a["id"] for a in tj["added_tokens"]}
    if by_content.get(tc.get("eos_token")) != ids["eos"] or by_content.get(tc.get("pad_token")) != ids["pad"]:
        problems.append(f"eos/pad tokens differ: base {tc.get('eos_token')}/{tc.get('pad_token')}, Edge0 {ids}")
    # Edge0 spells the case-insensitive contractions out; compare after expanding the base's (?i:...)
    base_re = tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    expanded = base_re.replace("(?i:'s|'t|'re|'ve|'m|'ll|'d)", "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])")
    if expanded != meta.get("pre_pattern"):
        problems.append("pre-tokenizer regex differs")
    if problems:
        raise ForgeError(f"{base} is not Edge0's tokenizer: " + "; ".join(problems[:6]))
    return {"source": str(base), "vocab": len(bv), "merges": len(bm), "added_tokens": len(added),
            "edge0_padding_tokens": len(rest), "special_ids": ids}


def find_tokenizer_source() -> Path | None:
    """A local Flash Next tokenizer folder, if one exists (checked by :func:`check_tokenizer` before use)."""
    candidates = sorted(Path("~/.mtplx/models").expanduser().glob("*Qwen3.8-Flash-Next*"))
    hub = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
    candidates += sorted(hub.glob("models--Qwen--Qwen3.8-Flash-Next*/snapshots/*"))
    for c in candidates:
        if all((c / f).exists() for f in TOKENIZER_FILES):
            return c
    return None


def _write_tokenizer(pack: Edge0Pack, base: Path, out: Path) -> dict:
    tok = pack.path / "tokenizer"
    report = check_tokenizer(tok, base)
    for name in TOKENIZER_FILES:
        shutil.copyfile(base / name, out / name)
    template = (tok / "chat_template.jinja").read_text()
    (out / "chat_template.jinja").write_text(template)
    tc = json.loads((out / "tokenizer_config.json").read_text())
    report["inline_template_was_identical"] = tc.get("chat_template") == template
    tc["chat_template"] = template
    _write_json_atomic(out / "tokenizer_config.json", tc)
    ids = report["special_ids"]
    gen = {"bos_token_id": ids["bos"], "eos_token_id": [ids["eos"], ids["pad"]], "pad_token_id": ids["pad"]}
    if (base / "generation_config.json").exists():
        bg = json.loads((base / "generation_config.json").read_text())
        gen.update({k: bg[k] for k in ("do_sample", "temperature", "top_k", "top_p") if k in bg})
    _write_json_atomic(out / "generation_config.json", gen)
    return report


# --------------------------------------------------------------------------- shell and latent shards


def _write_shell(pack: Edge0Pack, out: Path, *, q8: str, shard_bytes: int) -> dict:
    import mlx.core as mx

    tmp = out / ".shell.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    report: dict = {}
    shards: list[dict] = []
    buf: dict = {}
    size = 0

    def flush():
        nonlocal buf, size
        if not buf:
            return
        name = SHELL_PATTERN.format(index=len(shards) + 1)
        mx.save_safetensors(str(tmp / name), buf, metadata={"format": "mlx", "edge0_part": "shell"})
        shards.append({"file": name, "keys": sorted(buf), "bytes": size})
        buf, size = {}, 0

    for name, arr in shell.convert_shell(pack, q8=q8, report=report):
        mx.eval(arr)
        buf[name] = arr
        size += arr.nbytes
        if size >= shard_bytes:
            flush()
    flush()
    for old in out.glob("model-shell-*.safetensors"):
        old.unlink()
    for s in shards:
        os.replace(tmp / s["file"], out / s["file"])
    tmp.rmdir()
    notes = {k: {kk: vv for kk, vv in v.items() if kk not in ("op", "sources", "note")} for k, v in report.items()}
    notes = {k: v for k, v in notes.items() if v}
    return {"shards": [{"file": s["file"], "bytes": s["bytes"], "tensors": len(s["keys"])} for s in shards],
            "notes": notes}


def _latent_layer_ok(path: Path, layer: int, settings: dict) -> bool:
    if not path.exists():
        return False
    try:
        header, _ = read_safetensors_header(path)
    except (ForgeError, ValueError, OSError):
        return False
    meta = header.get("__metadata__", {})
    return (
        meta.get("layer") == str(layer)
        and meta.get("format_version") == str(latent_convert.FORMAT_VERSION)
        and meta.get("dense_format") == settings["dense_format"]
    )


# --------------------------------------------------------------------------- accounting


def account(out: Path, cfg: dict) -> dict:
    """Fail closed unless the shards hold exactly the model's parameters (names, shapes, latent dtypes)."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mtplx.models.edge0_latent import LATENT_HEADS, Model, ModelArgs

    files = sorted(out.glob("model*.safetensors"))
    keys: dict[str, tuple[str, list, str]] = {}
    total = 0
    for f in files:
        header, start = read_safetensors_header(f)
        total += os.stat(f).st_size
        for k, v in header.items():
            if k == "__metadata__":
                continue
            if k in keys:
                raise ForgeError(f"{k} is in both {keys[k][2]} and {f.name}")
            keys[k] = (v["dtype"], v["shape"], f.name)
    model = Model(ModelArgs.from_dict(cfg))
    q = cfg.get("quantization")
    if q is not None:
        def predicate(p, m):
            if p in q:
                return q[p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in keys

        nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"), class_predicate=predicate)
    params = dict(tree_flatten(model.parameters()))
    missing = sorted(set(params) - set(keys))
    extra = sorted(set(keys) - set(params))
    st_dtype = {"F32": "float32", "BF16": "bfloat16", "F16": "float16", "U32": "uint32", "U8": "uint8",
                "I32": "int32", "I64": "int64"}
    wrong = []
    for k, ref in params.items():
        if k not in keys:
            continue
        dt, shape, _ = keys[k]
        if tuple(shape) != tuple(ref.shape):
            wrong.append(f"{k}: shape {shape} vs {tuple(ref.shape)}")
        tail = k.split(".mlp.", 1)[1] if ".mlp." in k else ""
        if tail.split(".", 1)[0] in LATENT_HEADS and st_dtype.get(dt) != str(ref.dtype).split(".")[-1]:
            wrong.append(f"{k}: dtype {dt} vs {ref.dtype}")
    if missing or extra or wrong:
        raise ForgeError(
            f"pack accounting failed: {len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(extra)} unexpected (e.g. {extra[:3]}), {len(wrong)} mismatched (e.g. {wrong[:3]})"
        )
    return {"parameters": len(params), "files": len(files), "bytes": total}


# --------------------------------------------------------------------------- the build


def build_pack(
    edge0_dir,
    out_dir,
    *,
    tokenizer_from=None,
    q8: str = "affine8",
    dense_format: str | None = None,
    shard_bytes: int = SHELL_SHARD_BYTES,
    force: bool = False,
    layers=None,
) -> dict:
    """Build (or finish) the pack; returns the provenance record also written to ``edge0-forge.json``.

    ``layers`` limits the latent layers written (tests, partial builds); the
    final accounting runs only when every layer is present.
    """
    t0 = time.perf_counter()
    src = Path(edge0_dir)
    out = Path(out_dir)
    pack = Edge0Pack(src)
    dense_format = dense_format or pack.default_dense_format
    if q8 not in shell.Q8_MODES:
        raise ValueError(f"q8 must be one of {shell.Q8_MODES}")
    settings = {
        "forge_version": FORGE_VERSION,
        "q8": q8,
        "q8_scale_dtype": "float32",
        "dense_format": dense_format,
        "latent_format_version": latent_convert.FORMAT_VERSION,
        "shell_shard_bytes": int(shard_bytes),
    }
    fingerprint = _source_fingerprint(src)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / STATE_NAME
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and not force:
        if state.get("settings") != settings or state.get("source", {}).get("files") != fingerprint["files"]:
            raise ForgeError(
                f"{out} holds a pack built from another source or with other settings; pass force=True "
                "(--force) to rebuild it"
            )
    if force or state.get("settings") != settings:
        state = {}
    state.update({"settings": settings, "source": fingerprint, "mtplx_commit": _mtplx_commit()})
    state.setdefault("steps", {})
    _write_json_atomic(state_path, state)

    gguf = (src / "ple-fp8.gguf").absolute()
    cfg = pack_config(pack, q8=q8, dense_format=dense_format, ngram_file=str(gguf))
    state["ngram"] = {"file": str(gguf), "tables": _check_ngram(pack, cfg, gguf)}

    base = Path(tokenizer_from) if tokenizer_from else find_tokenizer_source()
    if base is None:
        raise ForgeError("no Flash Next tokenizer folder found; pass tokenizer_from (--tokenizer-from)")
    state["tokenizer"] = _write_tokenizer(pack, base, out)

    shell_step = state["steps"].get("shell")
    shell_ok = bool(shell_step) and all(
        (out / s["file"]).exists() and read_safetensors_header(out / s["file"]) for s in shell_step["shards"]
    )
    if shell_ok:
        _log("shell: complete, skipped")
    else:
        t = time.perf_counter()
        _log("shell: converting the skeleton")
        shell_step = _write_shell(pack, out, q8=q8, shard_bytes=shard_bytes)
        shell_step["seconds"] = round(time.perf_counter() - t, 1)
        state["steps"]["shell"] = shell_step
        _write_json_atomic(state_path, state)
        _log(f"shell: {len(shell_step['shards'])} shards in {shell_step['seconds']} s")

    wanted = list(range(pack.n_layers)) if layers is None else [int(x) for x in layers]
    todo = [i for i in wanted if not _latent_layer_ok(out / LATENT_PATTERN.format(layer=i), i, settings)]
    for i in todo:
        t = time.perf_counter()
        latent_convert.write_latent_safetensors(
            pack,
            out,
            layers=[i],
            dense_format=dense_format,
            key_prefix="language_model.model.layers.{layer}.mlp.",
            file_pattern=LATENT_PATTERN,
        )
        _log(f"latent layer {i}: {time.perf_counter() - t:.1f} s")
    done = [i for i in range(pack.n_layers) if (out / LATENT_PATTERN.format(layer=i)).exists()]
    state["steps"]["latent"] = {"layers_written_this_run": todo, "layers_present": len(done)}

    _write_json_atomic(out / "config.json", cfg)
    complete = len(done) == pack.n_layers
    if complete:
        acc = account(out, cfg)
        index = {"metadata": {"total_size": acc["bytes"]}, "weight_map": {}}
        for f in sorted(out.glob("model*.safetensors")):
            header, _ = read_safetensors_header(f)
            index["weight_map"].update({k: f.name for k in header if k != "__metadata__"})
        _write_json_atomic(out / INDEX_NAME, index)
        state["accounting"] = acc
        _log(f"accounting: {acc['parameters']} parameters in {acc['files']} files, {acc['bytes'] / 2**30:.2f} GiB")
    state["complete"] = complete
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["seconds_this_run"] = round(time.perf_counter() - t0, 1)
    _write_json_atomic(state_path, state)
    return state


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mtplx.edge0.forge", description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="Edge0 model folder")
    ap.add_argument("out", help="output pack folder")
    ap.add_argument("--tokenizer-from", help="Flash Next folder with tokenizer.json (checked against Edge0's)")
    ap.add_argument("--q8", choices=shell.Q8_MODES, default="affine8")
    ap.add_argument("--dense-format", choices=sorted(latent_convert.DENSE_BITS))
    ap.add_argument("--force", action="store_true", help="rebuild even if OUT holds a different build")
    args = ap.parse_args(argv)
    try:
        state = build_pack(args.src, args.out, tokenizer_from=args.tokenizer_from, q8=args.q8,
                           dense_format=args.dense_format, force=args.force)
    except ForgeError as exc:
        print(f"edge0-forge: {exc}", file=sys.stderr)
        return 1
    _log(f"done in {state['seconds_this_run']} s, complete={state['complete']}")
    return 0 if state["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
