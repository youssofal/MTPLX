#!/usr/bin/env python3
"""Repro: an exact-prefix SSD session restore (``ssd_clone``) runs on the paged decode layout.

Before the fix, ``SessionBank._restore_cold`` built its cache with
``runtime.make_cache()`` and dropped the caller's ``cache_factory``. Under the
turbo profile that is the vLLM-Metal paged layout, so the suffix prefill and the
whole decode of every ``ssd_clone`` turn ran paged, with compiled verify falling
back on every call (``context_above_threshold``).

The harness compares two arms with identical serve flags:

  installed  the MTPLX installed in the Python environment (for example the Mac
             app runtime, or ``pip install mtplx==2.12.2``)
  checkout   the same environment with ``PYTHONPATH`` pointing at this checkout

For each arm it starts a private server (own port, own empty SSD session-cache
folder; ``~/.mtplx/session-bank`` is never touched) and drives one synthetic
multi-turn conversation built from MTPLX's own source files:

  A1  long first turn                  cold prefill, saved to the SSD bank
      -- restart (RAM session cache empty) --
  A2  A1 history + ~6k new tokens      exact-prefix SSD restore -> ssd_clone
  A3  A2 history + ~6k new tokens      RAM restore (control)
      -- restart --
  A4  A3 history + ~6k new tokens      ssd_clone at a larger context
  A5  A4 history + ~6k new tokens      RAM restore (control)

With thinking off, turn N's rendered prompt (and prompt + reply) is an exact
prefix of turn N+1, so the near-prefix lane declines the saved entry
(``matched_out_of_range``) and the exact-prefix SSD lane serves it.

Receipts are read back from ``~/.mtplx/logs/request-log-<port>.jsonl``.

Run (about 25 minutes for both arms on an M5 Pro; one 27B model in memory at a
time, so stop any other MTPLX server first):

    <env>/bin/python benchmarks/repro_ssd_clone_restore_layout.py --dry-run
    <env>/bin/python benchmarks/repro_ssd_clone_restore_layout.py \\
        --out benchmarks/results/ssd-clone-restore-layout-<machine>.json

Before the fix the ``ssd_clone`` turns report ``paged_kv_num_blocks`` > 0,
compiled verify 0/N, and prefill well below the RAM controls. With it they match
the RAM controls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / ".mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"

RECEIPT_FIELDS = [
    "request_id", "prompt_tokens", "cached_tokens", "new_prefill_tokens", "completion_tokens",
    "cache_source", "session_restore_mode", "cache_miss_reason", "ssd_restore_s",
    "cache_restore_time_s", "prompt_eval_time_s", "prefill_compute_tok_s", "ttft_s",
    "decode_tok_s", "decode_elapsed_s", "paged_kv_num_blocks", "paged_kv_capacity_tokens",
    "thermal_pressure_max", "active_memory_bytes", "accepted_by_depth", "drafted_by_depth",
    "context_copy_accepted_tokens", "effective_temperature", "effective_top_p", "effective_top_k",
]

PLAN = [
    # label, expectation, max_tokens, restart_before
    ("A1", "cold", 256, False),
    ("A2", "ssd_clone", 768, True),
    ("A3", "RAM control", 768, False),
    ("A4", "ssd_clone", 768, True),
    ("A5", "RAM control", 768, False),
]


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def serve_flags(args: argparse.Namespace) -> list[str]:
    # The MTPLX app's launch flags for this model, with a private port and SSD folder.
    return [
        "--host", "127.0.0.1",
        "--model", str(args.model),
        "--scheduler-mode", "ar_batch",
        "--batching-preset", "agent",
        "--max-active-requests", "4",
        "--decode-batch-max", "4",
        "--batch-wait-ms", "50.0",
        "--depth", "3",
        "--ssd-session-cache", "on",
        "--ssd-session-cache-max-size", "auto",
        "--ssd-session-cache-min-prefix-tokens", "512",
        "--context-window", "131072",
        "--enable-thermal-poll",
        "--fan-mode", args.fan_mode,
        "--unsafe-force-unverified",
        "--yes",
        "--reasoning", "auto",
        "--temperature", "1.0",
        "--top-p", "0.95",
        "--top-k", "20",
        "--reasoning-effort", args.reasoning_effort,
        "--adaptive-policy", "expected_value",
        "--no-stats-footer",
    ]


# ---------------------------------------------------------------- prompts

def tokenizer(model: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model / "tokenizer.json"))


def source_chunks():
    for f in sorted((CHECKOUT / "mtplx").rglob("*.py")):
        text = f.read_text(encoding="utf-8", errors="replace")
        if len(text) > 2000:
            yield f.relative_to(CHECKOUT).as_posix(), text


def take_tokens(stream, tok, target: int) -> str:
    parts, n = [], 0
    for name, text in stream:
        block = f"\n### File: {name}\n```python\n{text}\n```\n"
        t = len(tok.encode(block).ids)
        if n + t > target:
            keep = max(0, target - n)
            parts.append(tok.decode(tok.encode(block).ids[:keep]) + "\n```\n")
            break
        parts.append(block)
        n += t
    return "".join(parts)


def build_turns(model: Path, first_tokens: int, step_tokens: int):
    tok = tokenizer(model)
    stream = source_chunks()
    first = take_tokens(stream, tok, first_tokens)
    steps = [take_tokens(stream, tok, step_tokens) for _ in range(len(PLAN) - 1)]
    system = "You are a senior engineer reviewing a Python codebase. Answer in plain prose."
    ask1 = "Summarize what these modules do in five short bullet points."
    ask_n = ("Here is another part of the codebase. Explain in detail how it works and how it "
             "fits with the earlier modules. Write at least 700 words.")
    users = [first + "\n" + ask1] + [s + "\n" + ask_n for s in steps]
    return tok, system, users


# ---------------------------------------------------------------- arms and servers

def port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def arm_env(arm: str, checkout: Path) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if arm == "checkout":
        env["PYTHONPATH"] = str(checkout)
    return env


def arm_identity(args: argparse.Namespace, arm: str) -> dict:
    code = ("import json, mtplx.session_bank as s, mtplx.version as v; "
            "print(json.dumps({'session_bank': s.__file__, 'version': v.__version__}))")
    out = subprocess.run([str(args.python), "-P", "-c", code], env=arm_env(arm, args.checkout),
                         capture_output=True, text=True, cwd="/", timeout=120)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": (out.stderr or out.stdout).strip()[-400:]}


def start_server(args, arm: str, bank: Path, logf: Path) -> subprocess.Popen:
    cmd = [str(args.mtplx), "serve", "--port", str(args.port), "--ssd-session-cache-dir", str(bank),
           *serve_flags(args)]
    fh = open(logf, "a")
    fh.write(f"\n===== {dt.datetime.now().isoformat()} start {arm}\n{' '.join(cmd)}\n")
    fh.flush()
    proc = subprocess.Popen(cmd, env=arm_env(arm, args.checkout), stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(logf.parent), start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode}); see {logf}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=3) as r:
                if json.load(r).get("data"):
                    log(f"{arm} server ready on :{args.port} after {time.time() - t0:.0f}s")
                    return proc
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server did not become ready in 600s")


def stop_server(args, proc: subprocess.Popen | None) -> None:
    subprocess.run([str(args.mtplx), "stop", "--port", str(args.port)], capture_output=True, timeout=120)
    t0 = time.time()
    while port_busy(args.port) and time.time() - t0 < 90:
        time.sleep(1)
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if port_busy(args.port):
        raise RuntimeError(f"port {args.port} still busy after stop")


def wait_bank_entry(bank: Path, min_prefix: int, timeout: float = 240) -> int:
    db = bank / "manifest.sqlite"
    t0, best = time.time(), 0
    while time.time() - t0 < timeout:
        if db.exists():
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
                best = int(con.execute("SELECT max(prefix_len) FROM entries").fetchone()[0] or 0)
                con.close()
                if best >= min_prefix:
                    time.sleep(5)
                    return best
            except sqlite3.Error:
                pass
        time.sleep(2)
    raise RuntimeError(f"no SSD entry >= {min_prefix} tokens after {timeout}s (best {best})")


def chat(port: int, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def receipt(port: int, request_id: str, timeout: float = 30) -> dict:
    path = Path.home() / f".mtplx/logs/request-log-{port}.jsonl"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            for line in reversed(path.read_text().splitlines()):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("request_id") == request_id:
                    out = {k: rec.get(k) for k in RECEIPT_FIELDS}
                    cv = rec.get("compiled_verify") or {}
                    out["compiled_verify_calls"] = cv.get("calls")
                    out["compiled_verify_compiled"] = cv.get("compiled_calls")
                    out["compiled_verify_fallback_reasons"] = cv.get("fallback_reasons")
                    return out
        time.sleep(1)
    return {"request_id": request_id, "error": "receipt not found"}


PROFILES: dict[str, str | None] = {}


def server_profile(logf: Path) -> str | None:
    """The runtime profile the server resolved, from its startup banner."""
    text = logf.read_text(errors="replace")
    found = re.findall(r"Runtime contract verified\W+profile:\s*([A-Za-z0-9_-]+)", text) or re.findall(
        r"^\W*Profile\s+([A-Za-z0-9_-]+)", text, flags=re.MULTILINE
    )
    return found[-1] if found else None


def run_arm(args, arm: str, system: str, users: list[str], out_dir: Path, rows: list[dict]) -> None:
    bank = out_dir / f"bank-{arm}"
    bank.mkdir(parents=True, exist_ok=True)
    logf = out_dir / f"server-{arm}.log"
    messages = [{"role": "system", "content": system}]
    proc = start_server(args, arm, bank, logf)
    PROFILES[arm] = server_profile(logf)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=10) as r:
            model_id = json.load(r)["data"][0]["id"]
        for (label, expect, max_tokens, restart_first), user in zip(PLAN, users):
            if restart_first:
                last = [r for r in rows if r["arm"] == arm][-1]["receipt"].get("prompt_tokens") or 0
                saved = wait_bank_entry(bank, int(last * 0.9))
                log(f"{arm}: SSD bank holds a {saved:,}-token entry; restarting to empty RAM")
                stop_server(args, proc)
                proc = start_server(args, arm, bank, logf)
            messages.append({"role": "user", "content": user})
            t0 = time.time()
            resp = chat(args.port, model_id, messages, max_tokens)
            wall = time.time() - t0
            messages.append({"role": "assistant", "content": resp["choices"][0]["message"].get("content") or ""})
            r = receipt(args.port, resp.get("id", ""))
            rows.append({"arm": arm, "turn": label, "expect": expect, "wall_s": round(wall, 1), "receipt": r})
            log(f"{arm} {label} ({expect}): mode={r.get('session_restore_mode')} ctx={r.get('prompt_tokens')} "
                f"new={r.get('new_prefill_tokens')} prefill={_f(r.get('prefill_compute_tok_s'))} tok/s "
                f"decode={_f(r.get('decode_tok_s'))} tok/s paged_blocks={r.get('paged_kv_num_blocks')} "
                f"compiled_verify={r.get('compiled_verify_compiled')}/{r.get('compiled_verify_calls')} "
                f"thermal={r.get('thermal_pressure_max')}")
    finally:
        stop_server(args, proc)


def _f(x) -> str:
    return "-" if x is None else f"{x:.1f}"


# ---------------------------------------------------------------- metadata and summary

def sh(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd).stdout.strip()
    except Exception:
        return ""


def environment(args, identities: dict, turn_tokens: list[int]) -> dict:
    model_cfg = {}
    try:
        model_cfg = json.loads((args.model / "config.json").read_text()).get("quantization", {})
    except Exception:
        pass
    source = {}
    try:
        source = json.loads((args.model / ".mtplx-source.json").read_text())
    except Exception:
        pass
    return {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hardware": {"model": sh(["sysctl", "-n", "hw.model"]), "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "memory_bytes": int(sh(["sysctl", "-n", "hw.memsize"]) or 0)},
        "os": f"macOS {platform.mac_ver()[0]}",
        "model": {"path_name": args.model.name, "repo_id": source.get("repo_id"),
                  "revision": source.get("resolved_sha"),
                  "quantization": {"bits": model_cfg.get("bits"), "group_size": model_cfg.get("group_size")}},
        "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "thinking": "off"},
        "profile": PROFILES,
        "fan_mode": args.fan_mode,
        "reasoning_effort": args.reasoning_effort,
        "user_turn_tokens": turn_tokens,
        "arms": identities,
        "checkout_commit": sh(["git", "rev-parse", "HEAD"], cwd=args.checkout),
        "checkout_describe": sh(["git", "describe", "--tags", "--always", "--dirty"], cwd=args.checkout),
        "serve_flags": [args.model.name if f == str(args.model) else f for f in serve_flags(args)],
    }


def redact_home(text: str) -> str:
    """Keep the user's home folder out of results meant to be shared."""
    return text.replace(str(Path.home()), "~")


def summary(rows: list[dict]) -> str:
    lines = ["| Arm | Turn | Mode | Context | New | Prefill tok/s | Decode tok/s | TTFT s | Paged blocks | Compiled verify |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        r = row["receipt"]
        lines.append(f"| {row['arm']} | {row['turn']} | {r.get('session_restore_mode')} | {r.get('prompt_tokens') or 0:,} | "
                     f"{r.get('new_prefill_tokens') or 0:,} | {_f(r.get('prefill_compute_tok_s'))} | "
                     f"{_f(r.get('decode_tok_s'))} | {_f(r.get('ttft_s'))} | {r.get('paged_kv_num_blocks') or 0} | "
                     f"{r.get('compiled_verify_compiled')}/{r.get('compiled_verify_calls')} |")
    return "\n".join(lines)


def main() -> int:
    env_bin = Path(sys.executable).parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", type=Path, default=Path(sys.executable),
                    help="Python of the environment that has MTPLX installed (default: this one)")
    ap.add_argument("--mtplx", type=Path, default=env_bin / "mtplx", help="mtplx entry point in that environment")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--checkout", type=Path, default=CHECKOUT, help="checkout used by the 'checkout' arm")
    ap.add_argument("--arms", default="installed,checkout")
    ap.add_argument("--port", type=int, default=1240)
    ap.add_argument("--guard-port", type=int, default=1234,
                    help="refuse to run while a server listens here (the MTPLX app default); 0 disables")
    ap.add_argument("--fan-mode", choices=["default", "smart", "max"], default="default")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--first-tokens", type=int, default=56000)
    ap.add_argument("--step-tokens", type=int, default=6000)
    ap.add_argument("--work-dir", type=Path, default=None, help="server logs and private SSD banks")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON result here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.checkout = args.checkout.resolve()

    tok, system, users = build_turns(args.model, args.first_tokens, args.step_tokens)
    turn_tokens = [len(tok.encode(u).ids) for u in users]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    identities = {a: arm_identity(args, a) for a in arms}
    log(f"user turns (tokens): {turn_tokens}")
    for a, ident in identities.items():
        log(f"{a}: {ident}")
    files = [i.get("session_bank") for i in identities.values()]
    if len(arms) > 1 and len(set(files)) < len(files):
        log("both arms import the same mtplx.session_bank; use an environment with a regular (non-editable) install")
        return 2
    if args.dry_run:
        return 0
    if args.guard_port and port_busy(args.guard_port):
        log(f"port {args.guard_port} is busy: stop that MTPLX server first so two 27B models don't share memory")
        return 2
    if port_busy(args.port):
        log(f"port {args.port} is busy")
        return 2

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="mtplx-ssd-clone-repro-"))
    work.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {work}")
    rows: list[dict] = []
    for arm in arms:
        run_arm(args, arm, system, users, work, rows)
    result = {"environment": environment(args, identities, turn_tokens), "turns": rows}
    out = args.out or (work / "result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(redact_home(json.dumps(result, indent=2)) + "\n")
    print(summary(rows))
    log(f"result: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
