#!/usr/bin/env python3
"""Generate the published models manifest (site payload models.json).

The manifest is the bless-list `mtplx models --check` and the app consult:
for each official pack it pins the exact HF commit users should update
into, the minimum engine version that can load it, and a one-line note
shown next to the update button. Run AFTER the pack uploads so the pinned
revisions are the post-upload commits.

Usage:
  .venv/bin/python scripts/gen_models_manifest.py \
      --out site/releases/models.json \
      --note-39 "Quantized MTP draft head: smaller download, faster decode."
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Every repo the manifest blesses, with the engine floor that can load it.
# The Qwen 3.8 packs carry prequantized MTP heads (loader >= 2.0.1) and need
# the 3.8 family support that landed in 2.7.0.
BLESSED = {
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed": "2.7.0",
    "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed": "2.7.0",
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality": "2.7.0",
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16": "2.7.0",
    "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-FP16": "2.7.0",
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16": "2.7.0",
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed": "2.0.1",
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality": "2.0.1",
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16": "2.0.1",
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16": "2.0.1",
    "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed": "2.0.1",
    "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed": "1.0.0",
    "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16": "1.0.0",
    "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed": "1.0.0",
    "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Quality": "1.0.0",
    "Youssofal/Gemma4-MTPLX-Optimized-Speed": "2.2.0",
}

QWEN38_NOTE_DEFAULT = (
    "Quantized MTP draft head: smaller download, same answers, faster decode."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note-38", default=QWEN38_NOTE_DEFAULT)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    models: dict[str, dict] = {}
    for repo, min_engine in BLESSED.items():
        try:
            info = api.model_info(repo_id=repo)
        except Exception as exc:
            print(f"skip {repo}: {exc}", file=sys.stderr)
            continue
        entry: dict = {
            "revision": info.sha,
            "min_engine_version": min_engine,
        }
        if "/Qwen3.8-" in f"/{repo.split('/', 1)[1]}" or repo.split("/", 1)[1].startswith(
            "Qwen3.8-"
        ):
            entry["note"] = args.note_38
        models[repo] = entry
        print(f"{repo} -> {info.sha[:12]}")

    payload = {
        "schema": 1,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(models)} models)")


if __name__ == "__main__":
    main()
