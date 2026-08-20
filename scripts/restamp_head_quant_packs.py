#!/usr/bin/env python3
"""Assemble, verify, and stamp the Qwen 3.8 quantized-head ship packs.

For each pack this driver:
  1. Fetches the repo's CURRENT config.json + mtplx_runtime.json from
     Hugging Face (authoritative base — the Bare repos carry an Aug-17
     draft-sampler restamp that only exists remotely; building ship stamps
     from local copies would silently revert it).
  2. Assembles an RC directory: every trunk file symlinked from the local
     base pack, the quantized mtp.safetensors copied from the EXP build,
     a ship config.json (HF config + the mtplx_mtp_quantization block),
     and a provisional ship mtplx_runtime.json (HF runtime + head-quant
     provenance addendum).
  3. Runs the HEAD forge's verification suite on the RC directory
     (`mtplx forge verify --max`) — real model load, max-fan gated by the
     forge itself — and re-mints speed_evidence with the same helpers the
     forge uses, including the artifact fingerprint that binds the rows to
     the exact config+head bytes users will pull.
  4. Gates flat-or-better: the RC acceptance-by-depth must not regress the
     repo's currently published acceptance (old BF16/FP16-cast head rows).
  5. Emits a receipt (shas, sizes, acceptance old vs new) consumed by the
     upload step, which pushes exactly the three owned files per repo in
     one atomic commit.

The fingerprint hashes only config.json + the MTP sidecar, so symlinked
local trunks mint fingerprints valid for HF-pulled packs byte-for-byte.

Never deletes anything; refuses to reuse an existing RC directory name.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODELS = Path.home() / ".mtplx/models"
OWNED_FILES = {"mtp.safetensors", "config.json", "mtplx_runtime.json"}

PACKS = [
    {
        "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
        "base": "Qwen3.8-27B-MTPLX-Optimized-Speed",
        "exp": "Qwen3.8-27B-MTPLX-Optimized-Speed-Q4HEAD-EXP",
        "bits": 4,
    },
    {
        "repo": "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
        "base": "Qwen3.8-27B-MTPLX-Bare-Speed",
        "exp": "Qwen3.8-27B-MTPLX-Bare-Speed-Q4HEAD-EXP",
        "bits": 4,
    },
    {
        "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
        "base": "Qwen3.8-27B-MTPLX-Optimized-Quality",
        "exp": "Qwen3.8-27B-MTPLX-Optimized-Quality-Q8HEAD-EXP",
        "bits": 8,
    },
    {
        "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
        "base": "Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
        "exp": "Qwen3.8-27B-MTPLX-Optimized-Speed-FP16-Q4HEAD-EXP",
        "bits": 4,
    },
    {
        "repo": "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
        "base": "Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
        "exp": "Qwen3.8-27B-MTPLX-Bare-Speed-FP16-Q4HEAD-EXP",
        "bits": 4,
    },
    {
        "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
        "base": "Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
        "exp": "Qwen3.8-27B-MTPLX-Optimized-Quality-FP16-Q8HEAD-EXP",
        "bits": 8,
    },
]

# Real suite runs wobble a little run-to-run; the proven Q4-on-4bit builds
# measured acceptance-POSITIVE and Q8-on-8bit identical, so anything below
# this tolerance is a real regression, not noise.
ACCEPTANCE_TOLERANCE = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_hf_current(repo: str, out_dir: Path) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo_id=repo, files_metadata=False)
    sha = info.sha
    files = {}
    for name in ("config.json", "mtplx_runtime.json"):
        local = hf_hub_download(repo, name, revision=sha)
        target = out_dir / f"hf-current-{name}"
        shutil.copyfile(local, target)
        files[name] = json.loads(target.read_text(encoding="utf-8"))
    return {"sha": sha, **files}


def _quant_block(exp_config: dict, bits: int) -> dict:
    block = dict(exp_config.get("mtplx_mtp_quantization") or {})
    if not block:
        raise SystemExit("EXP config has no mtplx_mtp_quantization block")
    tag = f"INT{bits}/g{block.get('group_size', 64)}"
    block["description"] = (
        f"All 8 MTP draft-head matrices (fc + attention q/k/v/o + MLP "
        f"gate/up/down) packed MLX {tag} affine from the released sidecar; "
        f"head norms keep the pack's float dtype. Verified flat-or-better "
        f"acceptance vs the unquantized head before publishing."
    )
    return block


def assemble(pack: dict, rc_dir: Path, work: Path) -> dict:
    base = MODELS / pack["base"]
    exp = MODELS / pack["exp"]
    if rc_dir.exists():
        raise SystemExit(f"RC dir already exists (pick a new suffix): {rc_dir}")
    for required in (base / "mtp.safetensors", exp / "mtp.safetensors"):
        if not required.exists():
            raise SystemExit(f"missing: {required}")

    hf = _fetch_hf_current(pack["repo"], work)
    exp_config = json.loads((exp / "config.json").read_text(encoding="utf-8"))

    rc_dir.mkdir(parents=True)
    for item in sorted(base.iterdir()):
        if item.name in OWNED_FILES or item.name.startswith("."):
            continue
        if item.name in {"build_report.json", "MTPLX_FP16_CONVERSION_MANIFEST.json"}:
            continue
        os.symlink(item.resolve(), rc_dir / item.name)

    shutil.copyfile(exp / "mtp.safetensors", rc_dir / "mtp.safetensors")

    ship_config = dict(hf["config.json"])
    ship_config["mtplx_mtp_quantization"] = _quant_block(exp_config, pack["bits"])
    (rc_dir / "config.json").write_text(
        json.dumps(ship_config, indent=2) + "\n", encoding="utf-8"
    )

    runtime = dict(hf["mtplx_runtime.json"])
    provenance = dict(runtime.get("forge_provenance") or {})
    provenance["head_quantization"] = {
        "bits": pack["bits"],
        "group_size": 64,
        "mode": "affine",
        "policy": "all",
        "quantized_at": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source_sidecar_bytes": (base / "mtp.safetensors").stat().st_size,
        "quantized_sidecar_bytes": (rc_dir / "mtp.safetensors").stat().st_size,
        "tool": "scripts/build_qwen38_q4head_sidecar.py",
        "note": (
            "Structural head quantization of the released sidecar; no "
            "calibration, no training. Trunk weights unchanged."
        ),
    }
    runtime["forge_provenance"] = provenance
    (rc_dir / "mtplx_runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    return hf


def run_verify(rc_dir: Path, out_dir: Path, run_id: str, max_tokens: int | None) -> list[dict]:
    cmd = [
        sys.executable,
        "-m",
        "mtplx.cli",
        "forge",
        "verify",
        str(rc_dir),
        "--max",
        "--json",
        "--out",
        str(out_dir),
        "--run-id",
        run_id,
    ]
    if max_tokens:
        cmd += ["--max-tokens", str(max_tokens)]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600
    )
    (out_dir / f"{run_id}-stdout.json").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{run_id}-stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(
            f"forge verify failed ({proc.returncode}) for {rc_dir}:\n"
            + proc.stderr[-2000:]
        )
    payload = json.loads(proc.stdout)
    rows = payload.get("rows") or []
    if not rows:
        raise SystemExit(f"forge verify returned no rows for {rc_dir}")
    return rows


def gate_and_stamp(
    pack: dict,
    rc_dir: Path,
    rows: list[dict],
    hf: dict,
    base_rows: list[dict] | None,
) -> dict:
    from mtplx.commands.forge import (
        _annotate_verify_rows,
        _speed_evidence,
        _verification_artifact_fingerprint,
    )
    from mtplx.version import __version__ as engine_version

    evidence = _speed_evidence(_annotate_verify_rows(rows))
    verdict = evidence.get("verdict")
    if verdict != "mtp_depth_wins":
        raise SystemExit(f"{pack['repo']}: verify verdict {verdict!r}, refusing to stamp")
    if evidence.get("failure_reasons"):
        raise SystemExit(f"{pack['repo']}: failure_reasons {evidence['failure_reasons']}")
    if any(row.get("hit_token_budget") for row in rows):
        raise SystemExit(f"{pack['repo']}: verify hit the token budget")

    new_acc = [float(x) for x in evidence.get("acceptance_by_depth") or []]

    # The flat-or-better gate compares SAME-SESSION paired arms: the base
    # pack (current published head) and the RC (quantized head) verified
    # back-to-back under identical suite/version/thermal state. Comparing
    # against the repo's months-old stamp is confounded by forge version,
    # suite drift, and run-to-run noise (the ledger's "single forge-verify
    # tune rows are order/JIT-confounded" scar) — that stamp is recorded
    # for reference only.
    comparison = []
    if base_rows is not None:
        base_evidence = _speed_evidence(_annotate_verify_rows(base_rows))
        base_acc = [float(x) for x in base_evidence.get("acceptance_by_depth") or []]
        for i, old in enumerate(base_acc):
            if i >= len(new_acc):
                break
            delta = new_acc[i] - old
            comparison.append(
                {"depth_pos": i + 1, "base_same_session": old, "rc": new_acc[i], "delta": delta}
            )
            if delta < -ACCEPTANCE_TOLERANCE:
                raise SystemExit(
                    f"{pack['repo']}: acceptance regression vs same-session base "
                    f"at position {i + 1}: {old:.4f} -> {new_acc[i]:.4f} "
                    f"(delta {delta:+.4f}, tolerance -{ACCEPTANCE_TOLERANCE})"
                )

    old_stamp_acc = [
        float(x)
        for x in (hf["mtplx_runtime.json"].get("speed_evidence") or {}).get(
            "acceptance_by_depth"
        )
        or []
    ]

    fingerprint = _verification_artifact_fingerprint(rc_dir)
    if not fingerprint:
        raise SystemExit(f"{pack['repo']}: could not fingerprint RC artifact")
    evidence["artifact_fingerprint"] = fingerprint

    runtime_path = rc_dir / "mtplx_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["speed_evidence"] = evidence
    runtime["mtplx_version"] = engine_version
    runtime["mtp_sidecar"] = f"int{pack['bits']}-g64-prequantized"
    import platform

    runtime["verified_on"] = {
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hardware": platform.platform(),
        "machine_arch": platform.machine(),
        "macos": platform.mac_ver()[0],
        "model": pack["base"],
    }
    tmp = runtime_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, runtime_path)

    return {
        "repo": pack["repo"],
        "rc_dir": str(rc_dir),
        "hf_base_sha": hf["sha"],
        "acceptance": comparison,
        "acceptance_new_full": new_acc,
        "acceptance_hf_stamp_reference": old_stamp_acc,
        "verdict": verdict,
        "artifact_fingerprint": fingerprint,
        "ship_files": {
            name: {
                "sha256": _sha256(rc_dir / name),
                "bytes": (rc_dir / name).stat().st_size,
            }
            for name in sorted(OWNED_FILES)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="RC29-20260820")
    ap.add_argument("--packs", nargs="*", help="pack base names to include")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Verify + stamp already-assembled RC dirs (refuses if the repo's "
            "HF sha moved since assembly)."
        ),
    )
    ap.add_argument(
        "--stamp-only",
        action="store_true",
        help=(
            "Skip the paired base arm: mint + stamp the RC verify rows only. "
            "Use when the flat-or-better acceptance gate already ran on the "
            "multi-seed sweep instrument (head_sweep_gate) — single verify "
            "rows cannot resolve heads and must not gate them."
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "head-restamp-20260820",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    selected = [
        pack
        for pack in PACKS
        if not args.packs or pack["base"] in args.packs
    ]
    receipts = []
    for index, pack in enumerate(selected):
        rc_dir = MODELS / f"{pack['base']}-{args.suffix}"
        work = args.out / pack["base"]
        work.mkdir(parents=True, exist_ok=True)
        print(f"=== {pack['repo']} -> {rc_dir.name}", flush=True)
        if args.verify_existing and rc_dir.exists():
            recorded = json.loads((work / "hf-state.json").read_text(encoding="utf-8"))
            hf = _fetch_hf_current(pack["repo"], work)
            if hf["sha"] != recorded["sha"]:
                raise SystemExit(
                    f"{pack['repo']}: HF moved since assembly "
                    f"({recorded['sha']} -> {hf['sha']}); reassemble first"
                )
            print(f"    reusing assembled RC at HF sha {hf['sha']}", flush=True)
        else:
            hf = assemble(pack, rc_dir, work)
            (work / "hf-state.json").write_text(
                json.dumps({"sha": hf["sha"]}, indent=2), encoding="utf-8"
            )
            print(f"    assembled at HF sha {hf['sha']}", flush=True)
        if args.assemble_only:
            continue
        if args.stamp_only:
            rows = run_verify(rc_dir, work, f"verify-{pack['base']}-rc", args.max_tokens)
            receipt = gate_and_stamp(pack, rc_dir, rows, hf, None)
            receipt["arm_order"] = "stamp-only (gated by head_sweep_gate)"
            receipts.append(receipt)
            (work / "receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            print(f"    STAMPED (sweep-gated) acc={receipt['acceptance_new_full']}", flush=True)
            continue
        # Paired same-session arms; alternate order across packs so a
        # systematic first-run/second-run bias cannot favor one arm
        # fleet-wide.
        base_dir = MODELS / pack["base"]
        rc_first = index % 2 == 1
        if rc_first:
            rows = run_verify(rc_dir, work, f"verify-{pack['base']}-rc", args.max_tokens)
            base_rows = run_verify(
                base_dir, work, f"verify-{pack['base']}-basearm", args.max_tokens
            )
        else:
            base_rows = run_verify(
                base_dir, work, f"verify-{pack['base']}-basearm", args.max_tokens
            )
            rows = run_verify(rc_dir, work, f"verify-{pack['base']}-rc", args.max_tokens)
        receipt = gate_and_stamp(pack, rc_dir, rows, hf, base_rows)
        receipt["arm_order"] = "rc-first" if rc_first else "base-first"
        receipts.append(receipt)
        (work / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        deltas = ", ".join(f"{c['delta']:+.4f}" for c in receipt["acceptance"])
        print(f"    PASS paired deltas [{deltas}] ({receipt['arm_order']})", flush=True)
    summary = args.out / "receipts.json"
    summary.write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    print(f"receipts: {summary}", flush=True)


if __name__ == "__main__":
    main()
