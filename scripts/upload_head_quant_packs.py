#!/usr/bin/env python3
"""Upload the verified quantized-head packs to Hugging Face.

One atomic commit per repo carrying exactly the three owned files
(mtp.safetensors, config.json, mtplx_runtime.json) — NEVER a folder upload:
the local pack dirs hold 81-byte README stubs that would clobber the real
HF model cards, and the trunk shards are already byte-identical upstream.

Refuses to push unless:
  * the RC receipt exists and its stamp carries an artifact fingerprint
    (i.e. the HEAD-forge verification ran and passed its gates), and
  * the repo's HF revision STILL equals the sha the RC was assembled
    against (someone pushing mid-campaign aborts the upload, not the
    other way around), and
  * the bytes on disk still hash to the receipt's shas.

Every push is recorded in upload-receipts.json with the new commit sha —
the input for gen_models_manifest.py and the catalog size re-audit.

FOUNDER APPROVAL REQUIRED: this publishes to public repos. Run only inside
an explicitly approved release campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OWNED_FILES = ("config.json", "mtp.safetensors", "mtplx_runtime.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--receipts",
        type=Path,
        default=REPO_ROOT / "outputs" / "head-restamp-20260820" / "receipts.json",
    )
    ap.add_argument("--repos", nargs="*", help="limit to these repo ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "head-restamp-20260820" / "upload-receipts.json",
    )
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    receipts = json.loads(args.receipts.read_text(encoding="utf-8"))
    api = HfApi()
    results = []
    for receipt in receipts:
        repo = receipt["repo"]
        if args.repos and repo not in args.repos:
            continue
        rc_dir = Path(receipt["rc_dir"])
        if not receipt.get("artifact_fingerprint"):
            raise SystemExit(f"{repo}: receipt has no artifact fingerprint; not verified")

        stamped = json.loads((rc_dir / "mtplx_runtime.json").read_text(encoding="utf-8"))
        stamped_fp = (stamped.get("speed_evidence") or {}).get("artifact_fingerprint")
        if stamped_fp != receipt["artifact_fingerprint"]:
            raise SystemExit(f"{repo}: stamped fingerprint != receipt fingerprint")

        for name in OWNED_FILES:
            expected = receipt["ship_files"][name]["sha256"]
            actual = _sha256(rc_dir / name)
            if actual != expected:
                raise SystemExit(
                    f"{repo}: {name} changed since verification "
                    f"({expected[:12]} -> {actual[:12]}); re-run the restamp"
                )

        info = api.model_info(repo_id=repo)
        if info.sha != receipt["hf_base_sha"]:
            raise SystemExit(
                f"{repo}: HF moved since assembly "
                f"({receipt['hf_base_sha'][:12]} -> {info.sha[:12]}); reassemble"
            )

        head_mb = receipt["ship_files"]["mtp.safetensors"]["bytes"] / 1_000_000
        bits = 8 if "int8" in str(stamped.get("mtp_sidecar")) else 4
        message = (
            f"Quantized MTP draft head (INT{bits}/g64 affine, all 8 head matrices)\n\n"
            f"mtp.safetensors: {head_mb:.0f} MB (was 849 MB). Trunk weights are\n"
            f"unchanged. Acceptance re-verified flat-or-better against the\n"
            f"previous head before publishing; mtplx_runtime.json carries the\n"
            f"fresh fingerprint-bound verification rows and config.json declares\n"
            f"the prequantized head layout (loadable by MTPLX >= 2.0.1;\n"
            f"this model family needs >= 2.7.0)."
        )
        operations = [
            CommitOperationAdd(
                path_in_repo=name, path_or_fileobj=str(rc_dir / name)
            )
            for name in OWNED_FILES
        ]
        if args.dry_run:
            print(f"DRY RUN {repo}: would commit {[op.path_in_repo for op in operations]}")
            continue
        commit = api.create_commit(
            repo_id=repo,
            operations=operations,
            commit_message=message,
            parent_commit=receipt["hf_base_sha"],
        )
        new_sha = getattr(commit, "oid", None) or getattr(commit, "commit_sha", None)
        print(f"pushed {repo}: {receipt['hf_base_sha'][:12]} -> {str(new_sha)[:12]}")
        results.append(
            {
                "repo": repo,
                "previous_sha": receipt["hf_base_sha"],
                "new_sha": new_sha,
                "files": receipt["ship_files"],
            }
        )
    if not args.dry_run:
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"upload receipts: {args.out}")


if __name__ == "__main__":
    main()
