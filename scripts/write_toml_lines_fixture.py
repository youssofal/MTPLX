#!/usr/bin/env python3
"""Write the TOML line-edit cases the app's Swift twin is tested against.

mtplx/toml_lines.py edits the model folders (model_dir, model_dirs) in the
config.toml the CLI and the app share. The app's TOMLLines must produce the
same bytes, or the two would fight over the file. tests/test_model_folders.py
fails when this file is stale, so after changing toml_lines.py run:

    python scripts/write_toml_lines_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtplx import toml_lines  # noqa: E402

FIXTURE = ROOT / "apps" / "MTPLXApp" / "Tests" / "Fixtures" / "toml_lines_cases.json"

TEXTS = (
    None,
    "",
    "# MTPLX user configuration\n",
    '# MTPLX user configuration\nmodel = "Youssofal/x"\nprofile = "sustained"\n',
    '# mine\nmodel_dir = "/old"\nmodel_dirs = ["/a", "/b"]\nprofile = "sustained"  # keep\n',
    'model_dirs = [\n  "/a",\n  "/b[1]",  # note\n]\n\n[extra]\nmodel_dir = "inner"\n',
    "[only]\nx = 1\n",
    "model_dir = '/literal'\nmodel_dirs = ['/l1', \"/l2\"]\n",
)
UPDATES = (
    {"model_dir": "/Volumes/SSD/models", "model_dirs": ["/Users/me/Models", "/Volumes/Old/llm"]},
    {"model_dir": None, "model_dirs": ["/x \"quoted\" é"]},
    {"model_dirs": None},
    {},
)


def cases() -> list[dict]:
    rows = []
    for text in TEXTS:
        for updates in UPDATES:
            rows.append(
                {
                    "text": text,
                    "updates": updates,
                    "output": toml_lines.set_top_level_values(text, updates),
                }
            )
    return rows


def render() -> str:
    return json.dumps({"version": 1, "cases": cases()}, indent=1, sort_keys=True) + "\n"


def main() -> int:
    FIXTURE.write_text(render(), encoding="utf-8")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
