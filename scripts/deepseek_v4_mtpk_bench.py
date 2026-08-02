"""Four-arm MTP speculative benchmark for the deepseek_v4 backend.

ONE model load, FOUR arms in-window: an AR control plus ``speculative_depth``
1/2/3.  In-window pairing is the box rule -- cross-window thermal drift on this
machine is 15-20%, which is larger than the effect being measured, so an arm
compared against a number from another window measures the fan, not the depth.

Both arms run through the real ``mtplx.generation`` machine over a real
:class:`~mtplx.runtime.MTPLXRuntime` (``runtime.load(..., mtp=True)`` ->
``inject_deepseek_v4_mtp_support``), not a hand-rolled loop: the point is to
measure the lane that would actually serve, including prefill, draft chain,
batched verify, accept/reject and the rollback repair.

Beyond speed, every K arm is compared token-for-token against the AR arm -- the
shop's standard spec==AR check (tests/test_deepseek_v4_spec.py) run at real dims
on real weights instead of a shrunk seeded model.  Whether a divergence FAILS the
run depends on which activation lane is being measured, and the harness decides
that rather than asking the reader to:

  * ``MTPLX_DSV4_FP32_ACTIVATIONS=1`` -- the diagnostic all-fp32 lane.  Byte
    identity holds there, so any divergence means the rollback is lossy and the
    tok/s number is meaningless.  Hard gate, exit 1.
  * bf16 activation storage (the default, and what serving runs).  Draft and
    verify are batch-shaped forwards, so the committed row's KV is projected
    inside a K+1-wide GEMM rather than alone.  At fp32 the precision headroom
    absorbed the resulting rounding; at bf16 it reaches the argmax on near-tied
    tokens.  The backend's documented invariant is committed-sequence exactness,
    not bitwise-identical logits, so a divergence here measures how often a
    near-tie routes differently -- data the receipt carries (count, first index,
    both tokens), not a verdict this harness is entitled to render on one prompt.
    A task eval is what settles quality; ``--require-exact`` restores the hard
    gate for anyone who wants it on this lane too.

Either way the comparison is always run and always recorded; only the exit status
moves.

``--tiny`` builds the shrunk seeded model the spec gates use and runs the whole
four-arm shape on CPU in seconds.  That is a harness self-test -- it validates
the arm loop, the stats extraction and the receipt writing without spending a
~90 GiB load -- not a performance measurement.

MUST run inside the box's serialized MLX window (bench/laguna/run_guarded.py):
the 2-bit checkpoint plus its MTP bank is ~93 GiB and does not fit beside the
served model.

Usage:
  python scripts/deepseek_v4_mtpk_bench.py \
      --model ~/models/DeepSeek-V4-Flash-2bit-DQ-mtp \
      --prompt-file bench/deepseek-v4/smoke-2bitdq-20260731-prompt2.txt \
      --max-tokens 256 --out bench/deepseek-v4/mtpk-2bitdq-YYYYMMDD
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from deepseek_v4_guard_window import load_verified_guard_window  # noqa: E402


# Peak memory is read per arm, so the ceiling is a per-arm claim.  Kept as a
# guard rather than an assertion: the wired knob is 112 GiB and is never raised
# (that is a box rule), so an arm that would cross it should stop the window
# rather than let the allocator fall off the wired cliff -- over-limit collapses
# throughput ~4x and the number would be garbage anyway.
_PEAK_ABORT_GIB = 108.0


def _gib(n: int) -> float:
    return n / (1024**3)


# ---------------------------------------------------------------------------
# spec-vs-AR: always measured, conditionally gated
# ---------------------------------------------------------------------------
def _fp32_activations_env() -> bool:
    """Whether the diagnostic all-fp32 activation lane is selected.

    Deliberately re-derived from the environment rather than imported from
    ``mtplx.models.deepseek_v4``: the harness must be able to state which lane it
    is gating before a ~90 GiB load, and in ``--tiny`` there is no checkpoint at
    all.  The parsing matches the backend's ``_env_flag``; a test pins the two
    against each other so they cannot drift apart silently.
    """
    return (os.environ.get("MTPLX_DSV4_FP32_ACTIVATIONS") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _exactness_is_enforced(require_exact: bool) -> bool:
    """Whether a spec-vs-AR divergence should fail the run.

    Greedy speculative decode is a pure latency optimisation, so on a lane where
    it *can* be byte-exact any divergence means the rollback is lossy and the
    tok/s number is meaningless.  That lane is fp32 activation storage, and there
    the gate stays hard.

    At the bf16 storage default it is not that lane, and the reason is structural
    rather than a bug: draft and verify are batch-shaped forwards, so the
    committed row's KV is projected inside a K+1-wide GEMM rather than alone.  The
    backend's documented invariant is committed-sequence exactness, not
    bitwise-identical logits (see ``Model.mtp_forward`` and the module docstring);
    at fp32 the precision headroom absorbed that difference, at bf16 it reaches
    the argmax on near-tied tokens.  So a divergence on the bf16 lane is a
    measurement -- how often a near-tie routes differently on this prompt -- and
    a harness that turned it into a verdict would be rendering a quality judgement
    from one 256-token sample, which is not what settles quality here.
    ``--require-exact`` restores the hard gate for anyone who wants it anyway.
    """
    return bool(require_exact) or _fp32_activations_env()


def _divergence(arm_tokens, baseline_tokens) -> dict:
    """Token-for-token comparison of a speculative arm against its AR control.

    Reports the whole shape of the difference, not just the first index: a single
    near-tie that both arms recover from reads very differently from a rollback
    that desyncs and never re-converges, and only the count separates them.
    Positions past the shorter sequence count as divergent, so a truncated arm
    cannot look identical by ending early.
    """
    arm = list(arm_tokens)
    base = list(baseline_tokens)
    overlap = min(len(arm), len(base))
    mismatches = [i for i in range(overlap) if arm[i] != base[i]]
    first = mismatches[0] if mismatches else (overlap if len(arm) != len(base) else None)
    return {
        "pass": arm == base,
        "baseline_tokens": len(base),
        "arm_tokens": len(arm),
        "compared_tokens": overlap,
        "divergent_tokens": len(mismatches) + abs(len(arm) - len(base)),
        "first_divergence_index": first,
        "baseline_at_divergence": (
            None if first is None or first >= len(base) else base[first]
        ),
        "arm_at_divergence": (
            None if first is None or first >= len(arm) else arm[first]
        ),
    }


def _summary_cell(gate) -> str:
    """The ``spec==AR`` column: a verdict when gated, a count when not."""
    if gate is None:
        return "-"
    if gate["pass"]:
        return "PASS"
    return "FAIL" if gate["enforced"] else f"{gate['divergent_tokens']} div"


def _divergence_line(gate: dict) -> str:
    """One-line rendering of :func:`_divergence` for the per-arm log."""
    if gate["pass"]:
        return "spec==AR: PASS (byte-identical)"
    detail = (
        f"{gate['divergent_tokens']} divergent of {gate['compared_tokens']} compared, "
        f"first at index {gate['first_divergence_index']}: "
        f"AR={gate['baseline_at_divergence']} spec={gate['arm_at_divergence']}"
    )
    if gate["enforced"]:
        return f"spec==AR: FAIL ({detail})"
    return f"spec==AR: DIVERGED, reported not gated ({detail})"


def _peak_bytes() -> int:
    fn = getattr(mx, "get_peak_memory", None)
    if callable(fn):
        return int(fn())
    fn = getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    return int(fn()) if callable(fn) else -1


def _active_bytes() -> int:
    fn = getattr(mx, "get_active_memory", None)
    if callable(fn):
        return int(fn())
    fn = getattr(getattr(mx, "metal", None), "get_active_memory", None)
    return int(fn()) if callable(fn) else -1


def _reset_peak() -> None:
    fn = getattr(mx, "reset_peak_memory", None)
    if callable(fn):
        fn()


def _clear_cache() -> None:
    fn = getattr(mx, "clear_cache", None)
    if callable(fn):
        fn()


# The stats surface is huge (every counter every backend ever needed).  Pull the
# ones this measurement is actually about, so the receipt stays readable; the
# full dict is kept alongside under "stats_full".
_STAT_KEYS = (
    "mode",
    "generated_tokens",
    "elapsed_s",
    "tok_s",
    "decode_elapsed_s",
    "decode_tok_s",
    "end_to_end_tok_s",
    "runtime_mtp_enabled",
    "draft_head_installed",
    "speculative_depth",
    "requested_speculative_depth",
    "accepted_by_depth",
    "drafted_by_depth",
    "accepted_drafts",
    "rejected_drafts",
    "drafted_tokens",
    "skipped_drafts",
    "bonus_tokens",
    "correction_tokens",
    "verify_calls",
    "mtp_forward_calls",
    "make_mtp_cache_calls",
    "update_mtp_cache_calls",
    "mtp_history_append_calls",
    "forward_ar_hidden_calls",
    "forward_ar_plain_calls",
    "draft_time_s",
    "verify_time_s",
    "verify_forward_time_s",
    "verify_eval_time_s",
    "target_forward_time_s",
    "snapshot_time_s",
    "prompt_eval_time_s",
    "prompt_tps",
    "mtp_history_policy",
    "reject_path_counts",
    "peak_memory_bytes",
)


def _accept_rates(stats: dict) -> list[dict]:
    """Per-depth accept rate.  ``drafted_by_depth[i]`` is how often depth ``i``
    was even proposed (it is not proposed when a shallower depth was rejected),
    so the rate is conditional on reaching that depth -- which is the number that
    predicts the speedup, unlike accepted/total-drafted."""
    accepted = list(stats.get("accepted_by_depth") or [])
    drafted = list(stats.get("drafted_by_depth") or [])
    rows = []
    for i in range(max(len(accepted), len(drafted))):
        a = int(accepted[i]) if i < len(accepted) else 0
        d = int(drafted[i]) if i < len(drafted) else 0
        rows.append(
            {
                "depth": i + 1,
                "drafted": d,
                "accepted": a,
                "accept_rate": (a / d) if d else None,
            }
        )
    return rows


def _mean_accepted_per_cycle(stats: dict) -> float | None:
    """Committed tokens per verify call: the cycle-anatomy number the projection
    leans on.  1.0 means speculation bought nothing."""
    calls = int(stats.get("verify_calls") or 0)
    if not calls:
        return None
    return float(stats.get("generated_tokens") or 0) / calls


def _run_arm(
    *,
    rt,
    label: str,
    depth: int | None,
    prompt_ids: list[int],
    max_tokens: int,
    verify_strategy: str,
    verify_core: str,
    mtp_history_policy: str,
    baseline_tokens: list[int] | None,
    enforce_exact: bool = True,
) -> dict:
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.sampling import SamplerConfig

    print(f"\n{'#' * 72}\n# ARM {label}\n{'#' * 72}")
    sys.stdout.flush()

    _clear_cache()
    _reset_peak()
    sampler = SamplerConfig(temperature=0.0)
    started = time.perf_counter()
    error = None
    out = None
    try:
        if depth is None:
            out = generate_ar(
                rt,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                # Forced full length in every arm: the arms are compared token
                # for token, so an early stop in one of them would compare
                # different amounts of work as well as different sequences.
                stop_token_ids=set(),
            )
        else:
            out = generate_mtpk(
                rt,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                speculative_depth=depth,
                mtp_history_policy=mtp_history_policy,
                verify_strategy=verify_strategy,
                verify_core=verify_core,
                stop_token_ids=set(),
            )
    except Exception:
        error = traceback.format_exc()
        print(error)
        sys.stdout.flush()
    wall = time.perf_counter() - started
    peak = _peak_bytes()

    arm: dict = {
        "label": label,
        "speculative_depth": depth,
        "verify_strategy": None if depth is None else verify_strategy,
        "verify_core": None if depth is None else verify_core,
        "mtp_history_policy": None if depth is None else mtp_history_policy,
        "wall_seconds": wall,
        "peak_bytes": peak,
        "peak_gib": _gib(peak),
        "active_end_gib": _gib(_active_bytes()),
        "error": error,
    }
    if out is None:
        return arm

    stats = out.stats.to_dict()
    decode_s = float(stats.get("decode_elapsed_s") or 0.0)
    n_new = int(stats.get("generated_tokens") or len(out.tokens))
    arm.update(
        {
            "tokens": list(out.tokens),
            "text": out.text,
            "finish_reason": out.finish_reason,
            "generated_tokens": n_new,
            "decode_seconds": decode_s,
            "decode_tokens_per_second": (n_new / decode_s) if decode_s else 0.0,
            "ms_per_token": (1000.0 * decode_s / n_new) if n_new else 0.0,
            "prefill_seconds": float(stats.get("prompt_eval_time_s") or 0.0),
            "prefill_tokens_per_second": float(stats.get("prompt_tps") or 0.0),
            "accept_rates": _accept_rates(stats),
            # Only meaningful on a speculative arm: AR's "verify calls" are just
            # its forwards, so the ratio there is 1 by construction and would
            # read as if the control were speculating.
            "mean_accepted_per_verify_call": (
                None if depth is None else _mean_accepted_per_cycle(stats)
            ),
            "stats": {k: stats.get(k) for k in _STAT_KEYS},
            "stats_full": stats,
        }
    )
    if baseline_tokens is not None:
        gate = _divergence(out.tokens, baseline_tokens)
        # Which lane this run is gating, recorded beside the comparison so a
        # receipt read later cannot be misread as an ungated run that passed.
        gate["enforced"] = bool(enforce_exact)
        gate["fp32_activations"] = _fp32_activations_env()
        arm["spec_equals_ar"] = gate

    print(f"[arm {label}] generated {n_new} tok  "
          f"decode {decode_s:.2f}s = {arm['decode_tokens_per_second']:.3f} tok/s "
          f"({arm['ms_per_token']:.1f} ms/tok)")
    print(f"[arm {label}] prefill {arm['prefill_seconds']:.2f}s = "
          f"{arm['prefill_tokens_per_second']:.1f} tok/s   "
          f"peak={arm['peak_gib']:.2f} GiB")
    if depth is not None:
        st = arm["stats"]
        print(f"[arm {label}] accepted={st['accepted_drafts']} "
              f"rejected={st['rejected_drafts']} drafted_tokens={st['drafted_tokens']} "
              f"verify_calls={st['verify_calls']} mtp_forward_calls={st['mtp_forward_calls']}")
        print(f"[arm {label}] accepted_by_depth={st['accepted_by_depth']} "
              f"drafted_by_depth={st['drafted_by_depth']}")
        for row in arm["accept_rates"]:
            rate = row["accept_rate"]
            print(f"[arm {label}]   depth {row['depth']}: "
                  f"{row['accepted']}/{row['drafted']} = "
                  f"{'n/a' if rate is None else f'{rate:.3f}'}")
        mac = arm["mean_accepted_per_verify_call"]
        print(f"[arm {label}] committed tokens per verify call: "
              f"{'n/a' if mac is None else f'{mac:.3f}'}")
        print(f"[arm {label}] draft {st['draft_time_s']:.2f}s  "
              f"verify {st['verify_time_s']:.2f}s  "
              f"target_forward {st['target_forward_time_s']:.2f}s  "
              f"snapshot {st['snapshot_time_s']:.2f}s")
        gate = arm.get("spec_equals_ar")
        if gate is not None:
            print(f"[arm {label}] {_divergence_line(gate)}")
    sys.stdout.flush()
    return arm


def _tiny_runtime_and_prompt(n_prompt: int):
    """Reuse the spec gate's shrunk seeded model so --tiny exercises exactly the
    wiring the gates cover.  CPU device, no download, no checkpoint."""
    here = Path(__file__).resolve().parents[1]
    path = here / "tests" / "test_deepseek_v4_spec.py"
    spec = importlib.util.spec_from_file_location("_dsv4_spec_for_bench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dsv4_spec_for_bench"] = module
    spec.loader.exec_module(module)
    return module._runtime(vocab=8), module._prompt(n_prompt, vocab=8)


def _deepseek_v4_moe_tail_install_report(rt, backend) -> dict | None:
    """Construction-time receipt for the fixed body/MTP callable route."""
    body = [layer.ffn._tail_combine for layer in rt.model.layers]
    mtp = [block.ffn._tail_combine for block in rt.model.mtp_blocks]
    installed = [
        route for route in body if isinstance(route, backend._InstalledMoETailRoute)
    ]
    mtp_installed = [
        route for route in mtp if isinstance(route, backend._InstalledMoETailRoute)
    ]
    if not backend._MOE_TAIL:
        if installed or mtp_installed:
            raise RuntimeError("stock arm unexpectedly installed the MoE-tail route")
        if any(route is not backend._stock_moe_tail_combine for route in body + mtp):
            raise RuntimeError("stock arm has a non-stock MoE-tail callable")
        return None
    if len(body) != 43 or len(installed) != 43:
        raise RuntimeError(
            f"MoE-tail candidate installed {len(installed)} of {len(body)} body layers"
        )
    if len(mtp) != 1 or mtp_installed:
        raise RuntimeError("MoE-tail candidate must leave the one MTP block stock")
    if any(route is not backend._stock_moe_tail_combine for route in mtp):
        raise RuntimeError("MoE-tail candidate MTP callable is not stock")
    if not backend._MOE_TAIL_SELF_CHECKED or backend._MOE_TAIL_KERNEL is None:
        raise RuntimeError("MoE-tail Metal construction self-check did not complete")
    return {
        "route": "decode_verify_m4",
        "body_layers_installed": len(installed),
        "mtp_layers_stock": len(mtp),
        "verify_rows": 4,
        "repair_rows": 1,
        "topk": 6,
        "hidden_size": 4096,
        "kernel_selfcheck_exact": True,
    }


def _require_clean_source(repo: Path) -> str:
    """Bind the measurement to one committed source tree before MLX import."""
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    if status.strip():
        raise RuntimeError("worktree is dirty; refusing an unrepeatable benchmark")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"source commit is malformed: {commit!r}")
    return commit


def _capture_moe_tail_routes(rt, backend) -> tuple:
    """Capture the once-selfchecked body callables before the stock primer."""
    routes = tuple(layer.ffn._tail_combine for layer in rt.model.layers)
    if len(routes) != 43 or not all(
        isinstance(route, backend._InstalledMoETailRoute) for route in routes
    ):
        raise RuntimeError("single-load bracket requires 43 prevalidated tail routes")
    mtp = tuple(block.ffn._tail_combine for block in rt.model.mtp_blocks)
    if len(mtp) != 1 or mtp[0] is not backend._stock_moe_tail_combine:
        raise RuntimeError("single-load bracket requires one stock MTP tail")
    if not backend._MOE_TAIL_SELF_CHECKED or backend._MOE_TAIL_KERNEL is None:
        raise RuntimeError("MoE-tail Metal construction self-check did not complete")
    return routes


def _bind_moe_tail_routes(rt, backend, routes: tuple, *, candidate: bool) -> dict | None:
    """Bind one proven callable set between arms, never inside generation."""
    if len(routes) != len(rt.model.layers) or len(routes) != 43:
        raise RuntimeError("MoE-tail route capture does not match the loaded body")
    selected = routes if candidate else (backend._stock_moe_tail_combine,) * len(routes)
    for layer, route in zip(rt.model.layers, selected, strict=True):
        layer.ffn._tail_combine = route
    for block in rt.model.mtp_blocks:
        block.ffn._tail_combine = backend._stock_moe_tail_combine
    observed = tuple(layer.ffn._tail_combine for layer in rt.model.layers)
    if observed != selected:
        raise RuntimeError("MoE-tail callable bind did not take effect exactly")
    if any(
        block.ffn._tail_combine is not backend._stock_moe_tail_combine
        for block in rt.model.mtp_blocks
    ):
        raise RuntimeError("MoE-tail bind changed the stock MTP callable")
    if not candidate:
        return None
    return {
        "route": "decode_verify_m4",
        "body_layers_installed": 43,
        "mtp_layers_stock": 1,
        "verify_rows": 4,
        "repair_rows": 1,
        "topk": 6,
        "hidden_size": 4096,
        "kernel_selfcheck_exact": True,
    }


def _moe_tail_route_census(rt, backend, routes: tuple) -> dict[str, int]:
    """Observe the bound callables between arms, outside measured generation."""
    candidate_ids = {id(route) for route in routes}
    stock = backend._stock_moe_tail_combine
    body = tuple(layer.ffn._tail_combine for layer in rt.model.layers)
    mtp = tuple(block.ffn._tail_combine for block in rt.model.mtp_blocks)
    body_candidate = sum(id(route) in candidate_ids for route in body)
    body_stock = sum(route is stock for route in body)
    mtp_stock = sum(route is stock for route in mtp)
    return {
        "body_candidate": body_candidate,
        "body_stock": body_stock,
        "body_other": len(body) - body_candidate - body_stock,
        "mtp_stock": mtp_stock,
        "mtp_other": len(mtp) - mtp_stock,
    }


def _reset_benchmark_state(rt) -> None:
    """Drop generation-local state while preserving the one loaded model."""
    mx.synchronize()
    counters = getattr(rt, "diagnostic_counters", None)
    if isinstance(counters, dict):
        counters.clear()
    gc.collect()
    _clear_cache()
    _reset_peak()


def _write_pair_receipt(stem: Path, receipt: dict, prompt_ids: list[int], prompt_file: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    blocks = []
    for arm in receipt["arms"]:
        if arm.get("error"):
            blocks.append(
                f"{'=' * 72}\nARM {arm['label']}: ERROR\n{'=' * 72}\n{arm['error']}\n"
            )
        else:
            blocks.append(
                f"{'=' * 72}\nARM {arm['label']} "
                f"({arm['generated_tokens']} tokens, greedy, "
                f"{arm['decode_tokens_per_second']:.3f} tok/s)\n"
                f"{'=' * 72}\n{arm['text']}\n"
            )
    stem.with_suffix(".txt").write_text(
        f"PROMPT ({len(prompt_ids)} tokens) from {prompt_file}\n" + "\n".join(blocks)
    )


def _run_single_process_moe_tail_bracket(
    *,
    rt,
    backend,
    routes: tuple,
    prompt_ids: list[int],
    args,
    common_receipt: dict,
    out_stem: Path,
) -> int:
    """Run primer/C0/B/C1 with one process, model, and construction self-check."""
    order = ("primer", "C0", "candidate", "C1")
    suffixes = ("primer", "before", "candidate", "after")
    bracket_id = hashlib.sha256(
        f"{common_receipt['guard_window']['window_id']}:{os.getpid()}:{time.monotonic_ns()}".encode()
    ).hexdigest()
    status = 0
    for index, (label, suffix) in enumerate(zip(order, suffixes, strict=True)):
        is_candidate = label == "candidate"
        role = "discarded_control_primer" if label == "primer" else "measurement"
        _bind_moe_tail_routes(rt, backend, routes, candidate=False)
        _reset_benchmark_state(rt)
        ar_census = _moe_tail_route_census(rt, backend, routes)
        ar = _run_arm(
            rt=rt,
            label=f"{label} AR stock",
            depth=None,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens,
            verify_strategy=args.verify_strategy,
            verify_core=args.verify_core,
            mtp_history_policy=args.mtp_history_policy,
            baseline_tokens=None,
        )
        _reset_benchmark_state(rt)
        install_report = _bind_moe_tail_routes(
            rt, backend, routes, candidate=is_candidate
        )
        k3_census = _moe_tail_route_census(rt, backend, routes)
        try:
            k3 = _run_arm(
                rt=rt,
                label=f"{label} K=3 {'candidate' if is_candidate else 'stock'}",
                depth=3,
                prompt_ids=prompt_ids,
                max_tokens=args.max_tokens,
                verify_strategy=args.verify_strategy,
                verify_core=args.verify_core,
                mtp_history_policy=args.mtp_history_policy,
                baseline_tokens=ar.get("tokens"),
                enforce_exact=_exactness_is_enforced(args.require_exact),
            )
        finally:
            _bind_moe_tail_routes(rt, backend, routes, candidate=False)
        post_census = _moe_tail_route_census(rt, backend, routes)
        exact_enforced = _exactness_is_enforced(args.require_exact)
        exact_gate = k3.get("spec_equals_ar")
        exact_failed = exact_enforced and (
            not isinstance(exact_gate, dict)
            or exact_gate.get("enforced") is not True
            or exact_gate.get("pass") is not True
        )
        pair_status = int(bool(ar.get("error") or k3.get("error") or exact_failed))
        status = max(status, pair_status)
        receipt = {
            **common_receipt,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "receipt_role": role,
            "performance_eligible": role == "measurement",
            "deepseek_v4_moe_tail": install_report,
            "single_process_bracket": {
                "bracket_id": bracket_id,
                "process_pid": os.getpid(),
                "model_object_id": id(rt.model),
                "model_load_count": 1,
                "execution_order": list(order),
                "arm_index": index,
            },
            "route_binding": {
                "ar": "stock",
                "k3": "candidate" if is_candidate else "stock",
                "post": "stock",
            },
            "route_census": {
                "ar": ar_census,
                "k3": k3_census,
                "post": post_census,
            },
            "arms": [ar, k3],
            "status": pair_status,
        }
        _write_pair_receipt(
            Path(f"{out_stem}-{suffix}"), receipt, prompt_ids, args.prompt_file
        )
        print(f"[single-load bracket] wrote {out_stem}-{suffix}.json")
        sys.stdout.flush()
    return status


def main() -> int:
    # This must precede the first MLX import.  It binds this descendant to the
    # still-live run_guarded process and its still-held canonical GPU lock.
    guard_window = load_verified_guard_window()
    repo = Path(__file__).resolve().parents[1]
    source_commit = _require_clean_source(repo)
    global mx
    import mlx.core as mx

    mlx_core_path = Path(mx.__file__).resolve()
    mlx_lib_path = mlx_core_path.parent / "lib" / "libmlx.dylib"
    mlx_identity = {
        "version": mx.__version__,
        "core_path": str(mlx_core_path),
        "core_sha256": hashlib.sha256(mlx_core_path.read_bytes()).hexdigest(),
        "lib_path": str(mlx_lib_path),
        "lib_sha256": hashlib.sha256(mlx_lib_path.read_bytes()).hexdigest(),
    }
    required_mlx_identity = {
        "version": "0.31.2",
        "core_sha256": "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6",
        "lib_sha256": "2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd",
    }
    if any(mlx_identity[key] != value for key, value in required_mlx_identity.items()):
        raise RuntimeError(
            f"requires official MLX 0.31.2 binary identity: {mlx_identity}"
        )

    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--prompt-file")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="speculative_depth values; one arm each, after the AR control",
    )
    ap.add_argument("--verify-strategy", default="capture_commit")
    ap.add_argument("--verify-core", default="stock")
    ap.add_argument("--mtp-history-policy", default="committed")
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="unrecorded AR warmup before the measured arms (0 to skip)",
    )
    ap.add_argument("--out", help="receipt path stem; writes <stem>.json and <stem>.txt")
    ap.add_argument(
        "--moe-tail-bracket",
        action="store_true",
        help="one-load primer/C0/MoE-tail/C1 K3 bracket",
    )
    ap.add_argument(
        "--receipt-role",
        choices=("measurement", "discarded_control_primer"),
        default="measurement",
    )
    ap.add_argument(
        "--require-exact",
        action="store_true",
        help="fail the run on any spec-vs-AR divergence, on any lane.  Implied by "
             "MTPLX_DSV4_FP32_ACTIVATIONS=1, where byte identity does hold; at the "
             "bf16 storage default divergences are reported as data unless this is "
             "passed (see the module docstring for why)",
    )
    ap.add_argument(
        "--tiny",
        action="store_true",
        help="harness self-test on the spec gate's shrunk seeded model (CPU, "
             "seconds); not a performance measurement",
    )
    args = ap.parse_args()

    launch_mtplx_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("MTPLX_")
        and not key.startswith("MTPLX_GUARD_ATTEST_")
        and not key.startswith("MTPLX_DSV4_GUARD_WINDOW_")
    }

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    load_seconds = 0.0
    artifact_identity = None
    loaded_runtime_identity = None
    prompt_identity = None
    prompt_path = None
    moe_tail_report = None
    config: dict = {}
    quant: dict = {}
    model_path = Path(args.tiny and "." or (args.model or "."))

    if args.tiny:
        rt, prompt_ids = _tiny_runtime_and_prompt(17)
        print(f"[bench] TINY harness self-test: {len(prompt_ids)} prompt tokens")
    else:
        if not args.model:
            sys.exit("no model path; pass --model (or --tiny)")
        model_path = Path(os.path.expanduser(args.model)).resolve()
        from deepseek_v4_moe_tail_gate import (
            _validate_loaded_runtime,
            _validate_model_artifact,
        )
        from mtplx import runtime as mtplx_runtime
        from mtplx.models import deepseek_v4 as deepseek_v4_backend

        try:
            config, artifact_identity = _validate_model_artifact(model_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            sys.exit(f"model identity gate failed: {error}")
        quant = config.get("quantization") or {}
        overrides = [k for k in quant if k not in ("group_size", "bits", "mode")]
        print(f"[bench] model      : {model_path}")
        print(f"[bench] model_type : {config.get('model_type')}  "
              f"layers={config.get('num_hidden_layers')}  "
              f"nextn={config.get('num_nextn_predict_layers')}")
        print(f"[bench] quantization: default bits={quant.get('bits')} "
              f"group_size={quant.get('group_size')} mode={quant.get('mode')} "
              f"per-path overrides={len(overrides)}")
        sys.stdout.flush()

        t0 = time.perf_counter()
        rt = mtplx_runtime.load(model_path, mtp=True)
        mx.eval(rt.model.parameters())
        load_seconds = time.perf_counter() - t0
        try:
            loaded_runtime_identity = _validate_loaded_runtime(rt, config)
            moe_tail_report = _deepseek_v4_moe_tail_install_report(
                rt, deepseek_v4_backend
            )
        except (AttributeError, RuntimeError, ValueError) as error:
            sys.exit(f"loaded runtime identity gate failed: {error}")
        print(f"[bench] loaded in {load_seconds:.1f}s  "
              f"active={_gib(_active_bytes()):.2f} GiB  "
              f"peak={_gib(_peak_bytes()):.2f} GiB  "
              f"mtp_enabled={rt.mtp_enabled}")
        sys.stdout.flush()
        if not rt.mtp_enabled:
            sys.exit(
                "runtime loaded with mtp_enabled=False: the draft head did not "
                "bind, so there is no speculative lane to benchmark"
            )

        if not args.prompt_file:
            sys.exit("no prompt; pass --prompt-file")
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        prompt_bytes = prompt_path.read_bytes()
        prompt_text = prompt_bytes.decode("utf-8")
        prompt_ids = list(rt.tokenizer.encode(prompt_text))
        prompt_identity = {
            "path": str(prompt_path),
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "tokens": len(prompt_ids),
        }
        total_context = len(prompt_ids) + args.max_tokens
        print(f"[bench] prompt tokens: {len(prompt_ids)}  new: {args.max_tokens}  "
              f"total context: {total_context}")
        if total_context > args.max_context:
            sys.exit(
                f"total context {total_context} exceeds --max-context "
                f"({args.max_context}); raise it deliberately, after checking the "
                f"quadratic score-tensor cost against the wired-memory budget"
            )
        sys.stdout.flush()

    after_load_active = _active_bytes()

    if args.moe_tail_bracket:
        if args.tiny:
            sys.exit("--moe-tail-bracket requires the canonical GPU model")
        if list(args.depths) != [3]:
            sys.exit("--moe-tail-bracket requires exactly --depths 3")
        if not args.out:
            sys.exit("--moe-tail-bracket requires --out")
        routes = _capture_moe_tail_routes(rt, deepseek_v4_backend)
        _bind_moe_tail_routes(
            rt, deepseek_v4_backend, routes, candidate=False
        )
        common_receipt = {
            "harness": "scripts/deepseek_v4_mtpk_bench.py",
            "source_commit": source_commit,
            "artifact_identity": artifact_identity,
            "loaded_runtime_identity": loaded_runtime_identity,
            "mlx_identity": mlx_identity,
            "command": ["python", *sys.argv],
            "host": {
                "platform": platform.platform(),
                "mlx_version": mx.__version__,
                "python": sys.version.split()[0],
            },
            "env": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith("MTPLX_")
                or key in ("HF_HUB_OFFLINE", "PYTHONPATH")
            },
            "launch_mtplx_env": launch_mtplx_env,
            "guard_window": guard_window,
            "tiny": False,
            "model_path": str(model_path),
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
            "quantization": {
                "default_bits": quant.get("bits"),
                "default_group_size": quant.get("group_size"),
                "default_mode": quant.get("mode"),
            },
            "sampling": {"greedy": True, "temperature": 0.0, "stop_token_ids": []},
            "prompt_file": str(prompt_path),
            "prompt": prompt_identity,
            "prompt_tokens": len(prompt_ids),
            "max_tokens": args.max_tokens,
            "depths": [3],
            "verify_strategy": args.verify_strategy,
            "verify_core": args.verify_core,
            "mtp_history_policy": args.mtp_history_policy,
            "fp32_activations": _fp32_activations_env(),
            "require_exact": bool(args.require_exact),
            "spec_equals_ar_enforced": _exactness_is_enforced(args.require_exact),
            "load_seconds": load_seconds,
            "active_after_load_gib": _gib(after_load_active),
        }
        return _run_single_process_moe_tail_bracket(
            rt=rt,
            backend=deepseek_v4_backend,
            routes=routes,
            prompt_ids=prompt_ids,
            args=args,
            common_receipt=common_receipt,
            out_stem=Path(args.out),
        )

    # Unrecorded warmup so the AR control is not the arm that pays first-call
    # allocator and kernel-compile cost.  Decode tok/s on this backend is stable
    # across loads (4.513 vs 4.514 in the 20260731 smoke receipts) but prefill is
    # not, and prefill is recorded per arm.
    if args.warmup_tokens > 0:
        print(f"[bench] warmup: AR, {args.warmup_tokens} tokens (not recorded)")
        sys.stdout.flush()
        _run_arm(
            rt=rt,
            label="warmup",
            depth=None,
            prompt_ids=prompt_ids,
            max_tokens=args.warmup_tokens,
            verify_strategy=args.verify_strategy,
            verify_core=args.verify_core,
            mtp_history_policy=args.mtp_history_policy,
            baseline_tokens=None,
        )

    enforce_exact = _exactness_is_enforced(args.require_exact)
    print(f"[bench] activation storage: "
          f"{'fp32 (MTPLX_DSV4_FP32_ACTIVATIONS=1)' if _fp32_activations_env() else 'model dtype (default)'}"
          f"   spec==AR: {'GATED (exit 1 on divergence)' if enforce_exact else 'REPORTED as data'}")
    sys.stdout.flush()

    arms: list[dict] = []
    ar = _run_arm(
        rt=rt,
        label="AR",
        depth=None,
        prompt_ids=prompt_ids,
        max_tokens=args.max_tokens,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        mtp_history_policy=args.mtp_history_policy,
        baseline_tokens=None,
    )
    arms.append(ar)
    baseline_tokens = ar.get("tokens")
    status = 0
    if baseline_tokens is None:
        print("[bench] AR control failed; the K arms have nothing to be gated against")
        status = 1

    for depth in args.depths:
        if not args.tiny and _gib(_peak_bytes()) > _PEAK_ABORT_GIB:
            print(f"[bench] ABORT: peak {_gib(_peak_bytes()):.2f} GiB is over the "
                  f"{_PEAK_ABORT_GIB} GiB per-arm guard; the wired knob is never "
                  f"raised, so the remaining arms are not run")
            status = 1
            break
        arm = _run_arm(
            rt=rt,
            label=f"K={depth}",
            depth=depth,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens,
            verify_strategy=args.verify_strategy,
            verify_core=args.verify_core,
            mtp_history_policy=args.mtp_history_policy,
            baseline_tokens=baseline_tokens,
            enforce_exact=enforce_exact,
        )
        arms.append(arm)
        if arm.get("error"):
            status = 1
        gate = arm.get("spec_equals_ar")
        if gate is not None and not gate["pass"] and gate["enforced"]:
            status = 1

    # ---- summary table ----------------------------------------------------
    ar_tps = float(ar.get("decode_tokens_per_second") or 0.0)
    print(f"\n{'=' * 78}\n=== FOUR-ARM SUMMARY ===\n{'=' * 78}")
    header = (f"{'arm':>6}  {'tok/s':>8}  {'ms/tok':>7}  {'x AR':>6}  "
              f"{'tok/cycle':>9}  {'peak GiB':>8}  {'spec==AR':>9}")
    print(header)
    print("-" * len(header))
    for arm in arms:
        if arm.get("error"):
            print(f"{arm['label']:>6}  {'ERROR':>8}")
            continue
        tps = float(arm.get("decode_tokens_per_second") or 0.0)
        mac = arm.get("mean_accepted_per_verify_call")
        gate = arm.get("spec_equals_ar")
        print(f"{arm['label']:>6}  {tps:8.3f}  {arm['ms_per_token']:7.1f}  "
              f"{(tps / ar_tps if ar_tps else 0.0):6.3f}  "
              f"{('n/a' if mac is None else f'{mac:.3f}'):>9}  "
              f"{arm['peak_gib']:8.2f}  "
              f"{_summary_cell(gate):>9}")
    print(
        f"\nspec==AR column: PASS = byte-identical to the AR arm.  "
        f"{'FAIL = divergence, and this run gates on it.' if enforce_exact else 'N div = divergent token count, reported not gated.'}"
    )
    if not enforce_exact:
        print("  bf16 activation storage is the lane here; the invariant is "
              "committed-sequence exactness, not bitwise-identical logits, so "
              "divergences are near-tie data.  --require-exact (or "
              "MTPLX_DSV4_FP32_ACTIVATIONS=1) restores the hard gate.")
    for arm in arms:
        if arm.get("speculative_depth") is None or arm.get("error"):
            continue
        parts = []
        for r in arm["accept_rates"]:
            rate = r["accept_rate"]
            shown = "n/a" if rate is None else f"{rate:.3f}"
            parts.append(f"d{r['depth']}={shown} ({r['accepted']}/{r['drafted']})")
        print(f"  {arm['label']} accept rates: {', '.join(parts)}")
    sys.stdout.flush()

    receipt = {
        "harness": "scripts/deepseek_v4_mtpk_bench.py",
        "source_commit": source_commit,
        "artifact_identity": artifact_identity,
        "loaded_runtime_identity": loaded_runtime_identity,
        "mlx_identity": mlx_identity,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": ["python", *sys.argv],
        "host": {
            "platform": platform.platform(),
            "mlx_version": mx.__version__,
            "python": sys.version.split()[0],
        },
        "env": {
            k: v for k, v in sorted(os.environ.items())
            if k.startswith("MTPLX_") or k in ("HF_HUB_OFFLINE", "PYTHONPATH")
        },
        "launch_mtplx_env": launch_mtplx_env,
        "guard_window": guard_window,
        "receipt_role": args.receipt_role,
        "performance_eligible": args.receipt_role == "measurement",
        "tiny": bool(args.tiny),
        "model_path": str(model_path),
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
        "quantization": {
            "default_bits": quant.get("bits"),
            "default_group_size": quant.get("group_size"),
            "default_mode": quant.get("mode"),
        },
        "sampling": {"greedy": True, "temperature": 0.0, "stop_token_ids": []},
        "prompt_file": str(prompt_path) if prompt_path is not None else args.prompt_file,
        "prompt": prompt_identity,
        "prompt_tokens": len(prompt_ids),
        "max_tokens": args.max_tokens,
        "depths": list(args.depths),
        "verify_strategy": args.verify_strategy,
        "verify_core": args.verify_core,
        "mtp_history_policy": args.mtp_history_policy,
        # Which lane was measured and whether byte identity was a gate on it, so a
        # status-0 receipt can never be read as "spec==AR held" when it was not asked to.
        "fp32_activations": _fp32_activations_env(),
        "require_exact": bool(args.require_exact),
        "spec_equals_ar_enforced": enforce_exact,
        "load_seconds": load_seconds,
        "deepseek_v4_moe_tail": moe_tail_report,
        "active_after_load_gib": _gib(after_load_active),
        "arms": arms,
        "status": status,
    }

    if args.out:
        stem = Path(args.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
        blocks = []
        for arm in arms:
            if arm.get("error"):
                blocks.append(f"{'=' * 72}\nARM {arm['label']}: ERROR\n"
                              f"{'=' * 72}\n{arm['error']}\n")
                continue
            blocks.append(
                f"{'=' * 72}\nARM {arm['label']}  "
                f"({arm['generated_tokens']} tokens, greedy, "
                f"{arm['decode_tokens_per_second']:.3f} tok/s)\n"
                f"{'=' * 72}\n{arm['text']}\n"
            )
        stem.with_suffix(".txt").write_text(
            f"PROMPT ({len(prompt_ids)} tokens) from {args.prompt_file}\n"
            + "\n".join(blocks)
        )
        print(f"receipts         : {stem.with_suffix('.json')}")
        print(f"                   {stem.with_suffix('.txt')}")
        sys.stdout.flush()

    return status


if __name__ == "__main__":
    raise SystemExit(main())
