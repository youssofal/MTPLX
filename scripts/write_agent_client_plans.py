#!/usr/bin/env python3
"""Write the Claude Code and Codex launch plans the app's Swift test reads.

The plans come from mtplx/agent_clients.py, which `mtplx start
claude-code|codex` and `mtplx connect` use: the Terminal plans for both
clients and the Codex desktop app's isolated-instance plan, config merge
included. The Swift test AgentCLIIntegrationTests builds the app's plan for
the same inputs and compares. tests/test_agent_clients.py fails when this file is stale, so after
changing the plan run:

    python scripts/write_agent_client_plans.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtplx.agent_clients import app_parity_fixture  # noqa: E402

FIXTURE = ROOT / "apps" / "MTPLXApp" / "Tests" / "Fixtures" / "agent_client_plans.json"


def render() -> str:
    return json.dumps(app_parity_fixture(), indent=1, sort_keys=True) + "\n"


def main() -> int:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(render(), encoding="utf-8")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
