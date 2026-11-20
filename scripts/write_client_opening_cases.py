#!/usr/bin/env python3
"""Write the client-opening cases the app's Swift test reads.

The cases come from mtplx/client_opening.py, which `mtplx start` uses to
decide where each client opens: the terminal found from an app's Info.plist,
and the opening for every client and saved choice. The Swift test
ClientOpeningTests resolves the same inputs and compares.
tests/test_client_opening.py fails when this file is stale, so after
changing the module run:

    python scripts/write_client_opening_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtplx.client_opening import app_parity_fixture  # noqa: E402

FIXTURE = ROOT / "apps" / "MTPLXApp" / "Tests" / "Fixtures" / "client_opening_cases.json"


def render() -> str:
    return json.dumps(app_parity_fixture(), indent=1, sort_keys=True) + "\n"


def main() -> int:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(render(), encoding="utf-8")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
