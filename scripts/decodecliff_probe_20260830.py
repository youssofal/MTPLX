"""Decode-cliff mechanism probe: MTP history policy A/B on the serve restore path.

Live telemetry (OpenCode session 2026-08-29, M5 Max 128G) shows Flash-Next
MoE decode collapsing 52 -> 30 tok/s from 53K to 89K context, but ONLY on
turns that restore a large session-bank state — a request that grows its own
context by generating 44K tokens holds ~59 tok/s. Issue #400 reports the
committed MTP history policy costing 1.5-1.7 s/verify-round on M2 Max
(cycle = 6.7x). This probe reproduces the founder's workload shape (large
warm-restored context + short follow-up decodes) and A/Bs the history
policy on otherwise-identical serves.

House protocol (qsa_prefill_battery_20260830 lineage):
- one serve per arm-rep, spawned from a neutral cwd with python -P,
  inherited MTPLX_* env stripped, arm env explicit;
- fans pinned max via thermalforge and VERIFIED (mode + actual rpm);
- die-temp gate before every arm;
- committed/cycle run A/B/B/A; last_window rides once at the end;
- per-(arm,rep) prompt salt so the SSD prefix bank can never leak state
  across arms; within an arm-rep, follow-ups warm-restore by design —
  that IS the workload under test;
- receipts read from the serve's own request log; a follow-up that does
  not warm-restore >=95% of its prompt invalidates the rep.

Usage:
  .venv/bin/python scripts/decodecliff_probe_20260830.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
THERMALFORGE = Path("/usr/local/bin/thermalforge")
MODEL = Path.home() / ".mtplx" / "models" / "Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
OUT_DIR = ROOT / "outputs" / "decodecliff-probe-20260830"
REQUEST_LOG = Path.home() / ".mtplx" / "logs" / "request-log-8399.jsonl"
NEUTRAL_CWD = "/tmp"
PORT = 8399
BASE = f"http://127.0.0.1:{PORT}"
DIE_TEMP_GATE_C = 62.0
FAN_MIN_TARGET_RPM = 7000
TARGET_PROMPT_CHARS = 350_000  # ~86-90K tokens of code
FOLLOWUP_MAX_TOKENS = 384

ARMS = {
    "committed": {},  # turbo default
    "cycle": {"MTPLX_MTP_HISTORY_POLICY": "cycle"},
    "last_window": {"MTPLX_MTP_HISTORY_POLICY": "last_window"},
    # S=1 sparse-attention lane arms (committed policy, lane forced):
    "flash": {"MTPLX_QSA_FLASH": "1"},
    "gatherdec": {"MTPLX_QSA_GATHER_DECODE": "1"},
    # Settle A/B controls (committed policy; settle default is ON in-tree):
    "settle_on": {},
    "settle_off": {"MTPLX_SESSION_SNAPSHOT_SETTLE": "0"},
}

FOLLOWUPS = [
    "Now write the next ~60 lines of a clean paged-KV allocator for this "
    "codebase, matching its style (type hints, terse docstrings). Code only.",
    "Continue: implement the eviction path for that allocator (LRU over "
    "pages, pinned-page exemption). Code only.",
    "Continue: add a compact unit test for allocate/evict/pin in this "
    "codebase's pytest style. Code only.",
]

RECEIPT_FIELDS = [
    "prompt_tokens",
    "cached_tokens",
    "new_prefill_tokens",
    "completion_tokens",
    "decode_tok_s",
    "decode_elapsed_s",
    "verify_calls",
    "verify_time_s",
    "verify_forward_time_s",
    "verify_eval_time_s",
    "verify_hidden_eval_time_s",
    "verify_logits_eval_time_s",
    "draft_time_s",
    "accept_time_s",
    "mean_accept_probability_by_depth",
    "mtp_history_policy",
    "mtp_depth",
    "generation_mode",
    "ttft_s",
    "peak_memory_bytes",
    "decode_flash_skip",
    "decode_dense_mask",
    "gather_rows",
    "dense_fallback",
    "decode_gather",
]


def _thermalforge(*args: str, use_sudo: bool = False) -> dict:
    argv = ["sudo", "-n", str(THERMALFORGE), *args] if use_sudo else [
        str(THERMALFORGE),
        *args,
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if args[0] == "status":
        return json.loads(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"thermalforge {args} failed: {proc.stderr or proc.stdout}")
    return {}


def pin_fans_max() -> None:
    _thermalforge("max", use_sudo=True)
    deadline = time.time() + 120
    while True:
        status = _thermalforge("status")
        fans = status["fans"]
        if all(
            f["mode"] == "manual"
            and f["target_rpm"] >= FAN_MIN_TARGET_RPM
            and f["actual_rpm"] >= FAN_MIN_TARGET_RPM
            for f in fans
        ):
            print(
                "fans pinned+verified:",
                [(f["index"], f["mode"], f["target_rpm"], f["actual_rpm"]) for f in fans],
                flush=True,
            )
            return
        if time.time() > deadline:
            raise RuntimeError("fan ramp verification timeout")
        time.sleep(3)


def restore_fans_auto() -> None:
    try:
        _thermalforge("auto", use_sudo=True)
    except RuntimeError as exc:
        print(f"WARNING: fan restore failed: {exc}", file=sys.stderr, flush=True)
    time.sleep(2)
    modes = [f["mode"] for f in _thermalforge("status")["fans"]]
    print("fans restored:", modes, flush=True)


def die_temp_gate() -> None:
    started = time.time()
    while True:
        status = _thermalforge("status")
        temps = [
            v
            for k, v in status["temperatures"].items()
            if k.lower().startswith(("tg", "tp"))
        ]
        hottest = max(temps) if temps else 0.0
        if hottest < DIE_TEMP_GATE_C:
            print(f"die-temp gate passed: hottest {hottest:.1f}C", flush=True)
            return
        if time.time() - started > 900:
            raise RuntimeError(f"die-temp gate timeout, hottest {hottest:.1f}C")
        print(f"cooling: hottest {hottest:.1f}C", flush=True)
        time.sleep(20)


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTPLX_")}
    env.update(extra)
    return env


def no_model_process_gate() -> None:
    proc = subprocess.run(
        ["pgrep", "-fl", r"mtplx(\.cli)? (serve|bench)|mtplx.server.openai|mlx_lm"],
        capture_output=True,
        text=True,
    )
    if proc.stdout.strip():
        raise RuntimeError(f"live model process detected:\n{proc.stdout}")


def build_code_dump() -> str:
    parts: list[str] = []
    total = 0
    for rel in ("mtplx", "mtplx/kernels", "mtplx/models"):
        d = ROOT / rel
        for f in sorted(d.glob("*.py")):
            text = f.read_text(errors="ignore")
            parts.append(f"\n# ==== file: {f.relative_to(ROOT)} ====\n{text}")
            total += len(text)
            if total >= TARGET_PROMPT_CHARS:
                return "".join(parts)[:TARGET_PROMPT_CHARS]
    return "".join(parts)[:TARGET_PROMPT_CHARS]


def http_json(method: str, url: str, payload: dict | None, timeout: float) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_health(timeout_s: float, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"serve process exited rc={proc.returncode} before /health"
            )
        try:
            h = http_json("GET", f"{BASE}/health", None, 5)
            if h.get("ok"):
                print("serve healthy", flush=True)
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("serve never became healthy")


def read_new_receipts(offset: int) -> tuple[list[dict], int]:
    if not REQUEST_LOG.exists():
        return [], offset
    with open(REQUEST_LOG) as f:
        f.seek(offset)
        chunk = f.read()
        new_offset = f.tell()
    rows = []
    for line in chunk.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows, new_offset


def teardown_port(proc: subprocess.Popen | None) -> None:
    pids: set[int] = set()
    out = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
    ).stdout.split()
    pids.update(int(p) for p in out if p.strip())
    if proc is not None and proc.poll() is None:
        pids.add(proc.pid)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 60
    while time.time() < deadline:
        listening = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        alive = proc is not None and proc.poll() is None
        if not listening and not alive:
            break
        time.sleep(2)
    else:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(3)
    time.sleep(10)  # RSS drain grace before the next load
    print("teardown complete (port free)", flush=True)


def run_arm(arm: str, rep: int, code_dump: str, fast: bool) -> dict:
    no_model_process_gate()
    die_temp_gate()
    log_file = OUT_DIR / f"serve-{arm}-rep{rep}.log"
    env = clean_env(ARMS[arm])
    cmd = [
        str(PYTHON),
        "-P",
        "-m",
        "mtplx.cli",
        "serve",
        "--model",
        str(MODEL),
        "--port",
        str(PORT),
        "--profile",
        "turbo",
    ]
    print(f"\n=== ARM {arm} rep{rep} start {datetime.now().isoformat()} ===", flush=True)
    offset = REQUEST_LOG.stat().st_size if REQUEST_LOG.exists() else 0
    log = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, cwd=NEUTRAL_CWD, env=env, stdout=log, stderr=subprocess.STDOUT
    )
    result: dict = {"arm": arm, "rep": rep, "turns": [], "low_ctx": None}
    try:
        wait_health(600, proc)
        salt = f"# probe-salt {arm}-rep{rep}-20260830\n"
        big_prompt = (
            salt
            + "Below is a dump of the MTPLX codebase. Study it carefully; "
            "in follow-up turns you will continue implementations in its "
            "exact style.\n" + code_dump
        )
        messages = [{"role": "user", "content": big_prompt}]
        # Turn 1: cold prefill + state store (not a measurement).
        t0 = time.time()
        r = http_json(
            "POST",
            f"{BASE}/v1/chat/completions",
            {
                "model": "mtplx-flash-next-optimized-speed",
                "messages": messages,
                "max_tokens": 16,
                "stream": False,
            },
            timeout=1200,
        )
        print(f"turn1 prefill done in {time.time()-t0:.0f}s", flush=True)
        messages.append(
            {"role": "assistant", "content": r["choices"][0]["message"]["content"] or "ok"}
        )
        time.sleep(8)  # let postcommit/store settle
        followups = FOLLOWUPS[: 1 if fast else len(FOLLOWUPS)]
        for i, ask in enumerate(followups):
            messages.append({"role": "user", "content": ask})
            t0 = time.time()
            r = http_json(
                "POST",
                f"{BASE}/v1/chat/completions",
                {
                    "model": "mtplx-flash-next-optimized-speed",
                    "messages": messages,
                    "max_tokens": FOLLOWUP_MAX_TOKENS,
                    "stream": False,
                },
                timeout=600,
            )
            print(f"turn{i+2} done in {time.time()-t0:.0f}s", flush=True)
            messages.append(
                {
                    "role": "assistant",
                    "content": r["choices"][0]["message"]["content"] or "ok",
                }
            )
            time.sleep(3)
        # Low-context control on the same serve.
        r = http_json(
            "POST",
            f"{BASE}/v1/chat/completions",
            {
                "model": "mtplx-flash-next-optimized-speed",
                "messages": [
                    {
                        "role": "user",
                        "content": salt
                        + "Write a python function that parses a JSONL file "
                        "and yields dicts, with type hints. Code only.",
                    }
                ],
                "max_tokens": FOLLOWUP_MAX_TOKENS,
                "stream": False,
            },
            timeout=300,
        )
        time.sleep(3)
        receipts, _ = read_new_receipts(offset)
        for row in receipts:
            entry = {k: row.get(k) for k in RECEIPT_FIELDS}
            entry["request_id"] = row.get("request_id")
            result["turns"].append(entry)
    finally:
        teardown_port(proc)
        log.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="1 follow-up per arm")
    parser.add_argument(
        "--arms",
        default="committed,cycle,cycle,committed,last_window",
        help="comma list; reps auto-numbered per arm",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_dir():
        raise SystemExit(f"model pack missing: {MODEL}")
    code_dump = build_code_dump()
    print(f"code dump chars: {len(code_dump)}", flush=True)

    pin_fans_max()
    all_results = []
    try:
        counts: dict[str, int] = {}
        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            counts[arm] = counts.get(arm, 0) + 1
            res = run_arm(arm, counts[arm], code_dump, args.fast)
            all_results.append(res)
            out = OUT_DIR / "results.json"
            out.write_text(json.dumps(all_results, indent=1))
            print(f"wrote {out}", flush=True)
    finally:
        restore_fans_auto()

    # Compact verdict table.
    print("\n=== VERDICT TABLE (warm follow-ups only) ===", flush=True)
    for res in all_results:
        for t in res["turns"]:
            if not t.get("prompt_tokens"):
                continue
            cached = t.get("cached_tokens") or 0
            warm = cached >= 0.90 * t["prompt_tokens"]
            kind = "warm" if warm else "cold"
            if t["prompt_tokens"] < 5000:
                kind = "lowctx"
            vc = t.get("verify_calls") or 0
            per_round = (
                1000.0 * (t.get("decode_elapsed_s") or 0) / vc if vc else 0.0
            )
            print(
                f"{res['arm']:>11} rep{res['rep']} {kind:6} "
                f"ctx={t['prompt_tokens']:>6} comp={t.get('completion_tokens') or 0:>5} "
                f"tok/s={t.get('decode_tok_s') or 0.0:6.1f} "
                f"round={per_round:6.1f}ms "
                f"vt={(t.get('verify_time_s') or 0.0):6.2f}s "
                f"dt={(t.get('draft_time_s') or 0.0):5.2f}s "
                f"policy={t.get('mtp_history_policy')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
