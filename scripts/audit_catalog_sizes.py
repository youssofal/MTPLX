#!/usr/bin/env python3
"""Audit the hand-pinned catalog size_bytes against live Hugging Face repos.

The Python catalog (mtplx/model_catalog.py) and the Swift mirror
(apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXModelOption.swift) pin each
official pack's exact download size. Head re-publishes change repo totals,
so this must run after every pack upload and both pins updated to match.

Prints one line per catalog entry: OK or MISMATCH with the exact new value
to pin. Exits 1 if any mismatch (CI-friendly).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from huggingface_hub import HfApi

    from mtplx.model_catalog import OFFICIAL_CATALOG

    api = HfApi()
    mismatches = 0
    for entry in OFFICIAL_CATALOG:
        repo = entry.hf_model_id
        try:
            info = api.model_info(repo_id=repo, files_metadata=True)
        except Exception as exc:
            print(f"SKIP {repo}: {exc}")
            continue
        total = sum(
            sibling.size
            for sibling in (info.siblings or [])
            if isinstance(getattr(sibling, "size", None), int)
        )
        if total == entry.size_bytes:
            print(f"OK       {repo}  {total:,}")
        else:
            mismatches += 1
            delta = total - entry.size_bytes
            print(
                f"MISMATCH {repo}\n"
                f"         pinned {entry.size_bytes:,}  live {total:,}  "
                f"(delta {delta:+,})\n"
                f"         pin -> size_bytes={total:_}"
            )
    if mismatches:
        print(f"\n{mismatches} catalog pin(s) need updating (Python + Swift sync pair).")
        raise SystemExit(1)
    print("\nall catalog pins match live repo totals")


if __name__ == "__main__":
    main()
