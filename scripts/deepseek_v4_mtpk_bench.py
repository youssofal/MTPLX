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

_OFFICIAL_MLX_IDENTITY = {
    "version": "0.31.2",
    "core_sha256": "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6",
    "lib_sha256": "2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd",
}
_CANONICAL_PROMPT_SHA256 = (
    "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33"
)
_ADAPTIVE_WIDTH_STAGE4_ENV = {
    "MTPLX_COMPILED_VERIFY": "off",
    "MTPLX_DSV4_ATTN": "fused",
    "MTPLX_DSV4_FP32_ACTIVATIONS": "0",
    "MTPLX_DSV4_HC_COMPILE": "1",
    "MTPLX_DSV4_MOE_TAIL": "1",
    "MTPLX_DSV4_O_LORA": "gather_qmm",
    "MTPLX_DSV4_SINKHORN_KERNEL": "1",
}
_ADAPTIVE_WIDTH_ARTIFACT_IDENTITY = {
    "config_sha256": "c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f",
    "index_sha256": "c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8",
    "model_type": "deepseek_v4",
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "body_q2_routed_projections": 129,
    "body_q2_manifest_tensors": 387,
    "mtp_manifest_tensors": 35,
    "index_weight_count": 2645,
}
_ADAPTIVE_WIDTH_LOADED_IDENTITY = {
    "runtime_mtp_enabled": True,
    "body_layers_loaded": 43,
    "mtp_blocks_bound": 1,
    "body_q2_routed_projections": 129,
    "body_q2_weight_dtype": "uint32",
    "mtp_mxfp4_routed_projections": 3,
    "mtp_routed_weight_dtype": "uint32",
}
_ADAPTIVE_WIDTH_MOE_TAIL_ROUTE = {
    "route": "decode_verify_m4",
    "body_layers_installed": 43,
    "mtp_layers_stock": 1,
    "verify_rows": 4,
    "repair_rows": 1,
    "topk": 6,
    "hidden_size": 4096,
    "kernel_selfcheck_exact": True,
}
_ADAPTIVE_WIDTH_O_LORA_ROUTE = {
    "mode": "gather_qmm",
    "module_count": 44,
    "trunk_module_count": 43,
    "mtp_module_count": 1,
    "body_direct": 43,
    "mtp_stock": 1,
    "body_all_mode_matches": True,
    "route_plan_matches": True,
}
_ADAPTIVE_WIDTH_O_LORA_CENSUS = {
    "body_route_objects": 43,
    "body_route_kind": "gather_qmm_m4_wide_direct",
    "body_callable_class": "_DirectGatherOLoraWideM4",
    "mtp_route_objects": 1,
    "mtp_route_kind": "dense_bf16_stock_direct",
    "mtp_callable_class": "_DirectDenseMTPOLora",
    "total_route_objects": 44,
    "unique_route_objects": 44,
    "mtp_distinct_type": True,
}
_ADAPTIVE_WIDTH_BRACKET_ARMS = (
    ("K3-PRIMER", False),
    ("K3-C0", False),
    ("ADAPTIVE-B", True),
    ("K3-C1", False),
)
_ATTN_PROJ_WIDE_M3_STAGE4_ENV = {
    **_ADAPTIVE_WIDTH_STAGE4_ENV,
    "MTPLX_DSV4_ATTN_PROJ_WIDE_M3": "1",
}
_ATTN_PROJ_WIDE_M3_BRACKET_ARMS = (
    ("CURRENT-PRIMER", False),
    ("CURRENT-C0", False),
    ("ATTN-PROJ-M3-B", True),
    ("CURRENT-C1", False),
)
_ATTENTION_ISLAND_STAGE4_ENV = {
    **_ATTN_PROJ_WIDE_M3_STAGE4_ENV,
    "MTPLX_DSV4_ATTENTION_ISLAND": "1",
}
_ATTENTION_ISLAND_BRACKET_ARMS = (
    ("ATTENTION-ISLAND-PRIMER", True),
    ("CURRENT-C0", False),
    ("ATTENTION-ISLAND-B", True),
    ("CURRENT-C1", False),
)
_ATTENTION_ISLAND_CONTROL_HISTOGRAM = {
    "K1_M2": 6,
    "K2_M3": 76,
    "K3_M4": 10,
}
_ATTENTION_ISLAND_LAYOUTS = (
    "hash-gs32",
    "score-gs32",
    "score-gs64",
)
_ATTENTION_ISLAND_PAIRED_QUALITY_FILENAME = (
    "hc-olora-51b0f105-20260802T161346Z-quality.json"
)
_ATTENTION_ISLAND_PAIRED_QUALITY_SHA256 = (
    "e8a3c1ed71aa9ac7024a457865c180c3aadafbf654f56066c750cf63e4a4bed2"
)
_ATTENTION_ISLAND_PAIRED_NEAR_TIE = {
    "path": (
        "bench/deepseek-v4/"
        "hc-olora-51b0f105-20260802T161346Z-quality.json"
    ),
    "sha256": _ATTENTION_ISLAND_PAIRED_QUALITY_SHA256,
    "quality_verdict": "ACCEPTED_SINGLE_IDENTICAL_BF16_TOP2_FLIP",
    "continuation_index": 221,
    "absolute_target_position": 549,
    "control_token_id": 14042,
    "candidate_token_id": 12258,
    "control_gap": 0.25,
    "candidate_gap": 0.0,
}
_ATTN_PROJ_WIDE_M3_EXPECTED_HISTOGRAM = {
    "K1_M2": 3,
    "K2_M3": 81,
    "K3_M4": 9,
}
_ATTN_PROJ_WIDE_M3_PROFILER_EVIDENCE = {
    "receipt": "bench/deepseek-v4/semantic-gather-k3-20260802T143027Z-receipt.json",
    "receipt_sha256": "0ebe0048503ec9cd46a6bcef5be6ef82968a43969004fda6673c1d01841c4b90",
    "qmv_wide_operation_count": 72058,
    "q4_bf16_nv3_kernel_count": 690,
    "timing_classification": "OVERLAP_INCLUSIVE_NONEXCLUSIVE_UPPER_BOUND",
    "use": "structural_candidate_selection_only_not_performance_verdict",
}
_ATTN_PROJ_WIDE_M3_ROUTE_RECEIPT = {
    "route": "target_verify_m3_original_q4_attention_projections",
    "logical_input_shape": [1, 3, 1024],
    "body_wq_b_prepared": 43,
    "body_indexer_wq_b_prepared": 0,
    "body_indexer_wq_b_stock": 21,
    "total_q4_projections_prepared": 43,
    "main_geometry": {"k": 1024, "n": 32768, "layers": 43},
    "indexer_geometry_stock": {"k": 1024, "n": 8192, "layers": 21},
    "indexer_activation_threshold_rows": 512,
    "canonical_max_compressed_rows": 146,
    "quantization": "affine_q4_g64",
    "activation_dtype": "bfloat16",
    "mtp_attention_dense_stock": 1,
    "o_lora_stock": 86,
    "small_attention_projections_stock": True,
    "mla_sdpa_cache_stock": True,
    "other_target_widths_stock": [2, 4],
    "ar_prefill_repair_mtp_stock": True,
    "kernel_selfcheck_exact": True,
    "both_arms_preinstalled": True,
    "arm_selection": "between_generations",
    "in_generation_module_rewrites": False,
}
_ADAPTIVE_WIDTH_POLICY_RECEIPT = {
    "kind": "deepseek_v4_preregistered_max_k3",
    "immutable": True,
    "d1_margin_threshold": 0.25,
    "d2_margin_threshold": 10.0,
    "max_speculative_depth": 3,
    "target_routes": {"K1": "M2", "K2": "M3", "K3": "M4"},
    "target_rows": [2, 3, 4],
}
_BEHAVIOR_SCALARS = (
    "generated_tokens",
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
)


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
    adaptive_width_policy=None,
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
                adaptive_width_policy=adaptive_width_policy,
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
        "adaptive_width_policy_enabled": adaptive_width_policy is not None,
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


def _token_sha256(tokens: list[int]) -> str:
    encoded = json.dumps(list(tokens), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _valid_nonnegative_int(value) -> bool:
    return type(value) is int and value >= 0


def _validate_behavior_stats(label: str, stats: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(stats, dict):
        return [f"{label} stats_full is not an object"]
    for key in _BEHAVIOR_SCALARS:
        if not _valid_nonnegative_int(stats.get(key)):
            errors.append(f"{label} has malformed counter {key}")
    for key in ("accepted_by_depth", "drafted_by_depth"):
        values = stats.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(_valid_nonnegative_int(value) for value in values)
        ):
            errors.append(f"{label} has malformed counter {key}")
    drafted_by_depth = stats.get("drafted_by_depth")
    if isinstance(drafted_by_depth, list) and all(
        _valid_nonnegative_int(value) for value in drafted_by_depth
    ):
        if sum(drafted_by_depth) != stats.get("drafted_tokens"):
            errors.append(f"{label} drafted counter sum is inconsistent")
        if not all(
            left >= right
            for left, right in zip(drafted_by_depth, drafted_by_depth[1:])
        ):
            errors.append(f"{label} drafted depth histogram is not monotone")
    if stats.get("generated_tokens") != 256:
        errors.append(f"{label} did not generate the canonical 256 tokens")
    if not isinstance(stats.get("events"), list):
        errors.append(f"{label} events are missing")
    return errors


def _adaptive_width_engagement(arm: dict) -> tuple[dict, list[str]]:
    errors: list[str] = []
    stats = arm.get("stats_full")
    events = stats.get("events", []) if isinstance(stats, dict) else []
    histogram = {"K1_M2": 0, "K2_M3": 0, "K3_M4": 0}
    policy_events = 0
    eligible_events = 0
    context_copy_events = 0
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"ADAPTIVE-B event {index} is malformed")
            continue
        if "context_copy" in event:
            context_copy_events += 1
            if "adaptive_width_policy" in event:
                errors.append("context-copy event carries adaptive policy metadata")
            continue
        policy = event.get("adaptive_width_policy")
        if not isinstance(policy, dict):
            if event.get("drafts"):
                errors.append(f"ADAPTIVE-B event {index} lacks policy engagement")
            continue
        policy_events += 1
        width = policy.get("selected_draft_depth")
        target_rows = policy.get("target_rows")
        margins = policy.get("decision_margins")
        eligible = policy.get("eligible_full_k3")
        if policy.get("kind") != "deepseek_v4_preregistered_max_k3":
            errors.append(f"ADAPTIVE-B event {index} has the wrong policy kind")
        if policy.get("d1_margin_threshold") != 0.25:
            errors.append(f"ADAPTIVE-B event {index} changed D1")
        if policy.get("d2_margin_threshold") != 10.0:
            errors.append(f"ADAPTIVE-B event {index} changed D2")
        if width not in {1, 2, 3} or target_rows != width + 1:
            errors.append(f"ADAPTIVE-B event {index} has an invalid target width")
            continue
        histogram[("K1_M2", "K2_M3", "K3_M4")[width - 1]] += 1
        if eligible is True:
            eligible_events += 1
            if not isinstance(margins, list) or len(margins) != min(width, 2):
                errors.append(f"ADAPTIVE-B event {index} has incomplete margins")
            elif not all(type(value) in {int, float} for value in margins):
                errors.append(f"ADAPTIVE-B event {index} has malformed margins")
            if width == 1 and not (float(margins[0]) < 0.25):
                errors.append(f"ADAPTIVE-B event {index} violates the D1 decision")
            if width >= 2 and not (float(margins[0]) >= 0.25):
                errors.append(f"ADAPTIVE-B event {index} violates the D1 tie rule")
            if width == 2 and not (float(margins[1]) < 10.0):
                errors.append(f"ADAPTIVE-B event {index} violates the D2 decision")
            if width == 3 and not (float(margins[1]) >= 10.0):
                errors.append(f"ADAPTIVE-B event {index} violates the D2 tie rule")
        elif eligible is not False:
            errors.append(f"ADAPTIVE-B event {index} lacks an eligibility receipt")
    if eligible_events <= 0:
        errors.append("ADAPTIVE-B has no eligible full-K3 policy events")
    if isinstance(stats, dict):
        drafted_by_depth = stats.get("drafted_by_depth")
        expected_drafted = [
            sum(histogram.values()),
            histogram["K2_M3"] + histogram["K3_M4"],
            histogram["K3_M4"],
        ]
        if drafted_by_depth != expected_drafted:
            errors.append("event-derived widths do not match drafted_by_depth")
        if stats.get("verify_calls") != policy_events + context_copy_events:
            errors.append("event-derived widths do not cover verify calls")
    return {
        "policy_events": policy_events,
        "eligible_full_k3_events": eligible_events,
        "context_copy_events": context_copy_events,
        "policy_thresholds": {
            "d1_margin_threshold": 0.25,
            "d2_margin_threshold": 10.0,
        },
        "event_derived_width_histogram": histogram,
    }, errors


def _token_quality(arms_by_label: dict[str, dict]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    controls = [
        arms_by_label.get("K3-PRIMER", {}).get("tokens"),
        arms_by_label.get("K3-C0", {}).get("tokens"),
        arms_by_label.get("K3-C1", {}).get("tokens"),
    ]
    candidate = arms_by_label.get("ADAPTIVE-B", {}).get("tokens")
    if not all(isinstance(tokens, list) and len(tokens) == 256 for tokens in controls):
        return {"accepted": False, "mode": "invalid_controls"}, [
            "fixed-K3 controls lack canonical token sequences"
        ]
    if controls[0] != controls[1] or controls[1] != controls[2]:
        return {"accepted": False, "mode": "control_mismatch"}, [
            "fixed-K3 controls are not token-identical"
        ]
    if not isinstance(candidate, list) or len(candidate) != 256:
        return {"accepted": False, "mode": "invalid_candidate"}, [
            "adaptive candidate lacks a canonical token sequence"
        ]
    control = controls[1]
    differing = [index for index, pair in enumerate(zip(control, candidate)) if pair[0] != pair[1]]
    if not differing:
        return {
            "accepted": True,
            "mode": "exact",
            "control_token_sha256": _token_sha256(control),
            "candidate_token_sha256": _token_sha256(candidate),
            "divergent_tokens": 0,
        }, errors
    cause = {
        "continuation_index": 221,
        "absolute_position": 549,
        "control_token_id": 14042,
        "candidate_token_id": 12258,
        "control_target_gap": 0.25,
        "candidate_target_gap": 0.0,
    }
    approved = (
        differing[0] == cause["continuation_index"]
        and control[:221] == candidate[:221]
        and control[221] == cause["control_token_id"]
        and candidate[221] == cause["candidate_token_id"]
    )
    quality = {
        "accepted": approved,
        "mode": "approved_bf16_top2_cause" if approved else "unapproved_divergence",
        "approved_cause": cause if approved else None,
        "divergent_tokens": len(differing),
        "first_divergence_index": differing[0],
        "control_token_sha256": _token_sha256(control),
        "candidate_token_sha256": _token_sha256(candidate),
        "propagated_tail": {
            "documented": approved,
            "start_continuation_index": 222,
            "compared_tokens": len(candidate) - 222,
            "divergent_tokens": sum(
                control[index] != candidate[index] for index in range(222, len(candidate))
            ),
            "control_tail_sha256": _token_sha256(control[222:]),
            "candidate_tail_sha256": _token_sha256(candidate[222:]),
            "human_eval": "deferred",
        },
    }
    if not approved:
        errors.append("adaptive candidate has an unapproved token divergence")
    return quality, errors


def _adaptive_width_common_errors(common: dict) -> list[str]:
    errors: list[str] = []
    source = common.get("source_commit")
    if not isinstance(source, str) or len(source) != 40 or any(
        character not in "0123456789abcdef" for character in source
    ):
        errors.append("source commit is malformed")
    expected = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "prompt_tokens": 328,
        "max_tokens": 256,
        "depths": [3],
        "verify_strategy": "capture_commit",
        "verify_core": "stock",
        "mtp_history_policy": "committed",
        "sampling": {"greedy": True, "temperature": 0.0, "stop_token_ids": []},
        "fp32_activations": False,
    }
    for key, value in expected.items():
        if common.get(key) != value:
            errors.append(f"{key} is not canonical")
    prompt = common.get("prompt")
    if not isinstance(prompt, dict) or prompt.get("sha256") != _CANONICAL_PROMPT_SHA256 or prompt.get("tokens") != 328:
        errors.append("prompt identity is not canonical")
    for key, expected_identity in (
        ("mlx_identity", _OFFICIAL_MLX_IDENTITY),
        ("artifact_identity", _ADAPTIVE_WIDTH_ARTIFACT_IDENTITY),
        ("loaded_runtime_identity", _ADAPTIVE_WIDTH_LOADED_IDENTITY),
        ("launch_mtplx_env", _ADAPTIVE_WIDTH_STAGE4_ENV),
    ):
        observed = common.get(key)
        if not isinstance(observed, dict) or any(
            observed.get(name) != value for name, value in expected_identity.items()
        ) or (key == "launch_mtplx_env" and observed != expected_identity):
            errors.append(f"{key} is not canonical")
    if common.get("deepseek_v4_moe_tail") != _ADAPTIVE_WIDTH_MOE_TAIL_ROUTE:
        errors.append("MoE-tail installed route is not canonical")
    o_lora = common.get("deepseek_v4_o_lora")
    if (
        not isinstance(o_lora, dict)
        or any(
            o_lora.get(name) != value
            for name, value in _ADAPTIVE_WIDTH_O_LORA_ROUTE.items()
        )
        or o_lora.get("callable_census") != _ADAPTIVE_WIDTH_O_LORA_CENSUS
    ):
        errors.append("o-LoRA installed route is not canonical")
    guard = common.get("guard_window")
    if not isinstance(guard, dict) or guard.get("verified") is not True:
        errors.append("guard window is not verified")
    return errors


def _adaptive_width_bracket_receipt(
    *,
    common: dict,
    arms: list[dict],
    process_pid: int,
    model_object_id: int,
    policy_receipt: dict,
) -> dict:
    errors = _adaptive_width_common_errors(common)
    expected_order = [label for label, _enabled in _ADAPTIVE_WIDTH_BRACKET_ARMS]
    observed_order = [arm.get("label") for arm in arms if isinstance(arm, dict)]
    if observed_order != expected_order:
        errors.append("adaptive-width arm order is invalid")
    if type(process_pid) is not int or process_pid <= 0:
        errors.append("process identity is invalid")
    if type(model_object_id) is not int or model_object_id <= 0:
        errors.append("model object identity is invalid")
    if policy_receipt != _ADAPTIVE_WIDTH_POLICY_RECEIPT:
        errors.append("installed adaptive-width policy receipt is invalid")

    arms_by_label = {
        str(arm.get("label")): arm for arm in arms if isinstance(arm, dict)
    }
    for label in expected_order:
        arm = arms_by_label.get(label)
        if arm is None:
            continue
        if arm.get("error") is not None:
            errors.append(f"{label} failed")
        if arm.get("generated_tokens") != 256 or arm.get("finish_reason") != "length":
            errors.append(f"{label} did not complete the canonical workload")
        tokens = arm.get("tokens")
        if not isinstance(tokens, list) or arm.get("token_sha256") != _token_sha256(tokens):
            errors.append(f"{label} token identity is malformed")
        errors.extend(_validate_behavior_stats(label, arm.get("stats_full")))

    for label in ("K3-PRIMER", "K3-C0", "K3-C1"):
        stats = arms_by_label.get(label, {}).get("stats_full", {})
        events = stats.get("events", []) if isinstance(stats, dict) else []
        if any(
            isinstance(event, dict) and "adaptive_width_policy" in event
            for event in events
        ):
            errors.append(f"{label} unexpectedly engaged adaptive width")

    candidate = arms_by_label.get("ADAPTIVE-B", {})
    engagement, engagement_errors = _adaptive_width_engagement(candidate)
    errors.extend(engagement_errors)
    quality, quality_errors = _token_quality(arms_by_label)
    errors.extend(quality_errors)

    def _tps(label: str) -> float:
        value = arms_by_label.get(label, {}).get("decode_tokens_per_second")
        return float(value) if type(value) in {int, float} and value > 0 else 0.0

    c0_tps = _tps("K3-C0")
    c1_tps = _tps("K3-C1")
    candidate_tps = _tps("ADAPTIVE-B")
    if not c0_tps or not c1_tps or not candidate_tps:
        errors.append("performance cells are missing positive throughput")
    control_mean = (c0_tps + c1_tps) / 2.0
    drift_tps = abs(c1_tps - c0_tps)
    promotion_floor = control_mean + drift_tps
    performance = {
        "control_c0_tps": c0_tps,
        "control_c1_tps": c1_tps,
        "control_mean_tps": control_mean,
        "control_drift_tps": drift_tps,
        "candidate_tps": candidate_tps,
        "candidate_minus_control_mean_tps": candidate_tps - control_mean,
        "promotion_floor_tps": promotion_floor,
        "promotion_pass": candidate_tps > promotion_floor,
        "reported_below_40_tps": candidate_tps < 40.0,
    }
    return {
        **common,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "receipt_role": "adaptive_width_performance_bracket",
        "performance_eligible": True,
        "single_process_bracket": {
            "process_pid": process_pid,
            "model_object_id": model_object_id,
            "model_load_count": 1,
            "execution_order": expected_order,
            "discarded_primer": "K3-PRIMER",
        },
        "policy": policy_receipt,
        "policy_engagement": engagement,
        "token_quality": quality,
        "performance": performance,
        "arms": arms,
        "validation_errors": errors,
        "status": int(bool(errors)),
    }


def _installed_policy_receipt(policy) -> dict:
    rows = [int(route.target_rows) for route in policy.target_routes]
    return {
        "kind": "deepseek_v4_preregistered_max_k3",
        "immutable": bool(
            getattr(getattr(type(policy), "__dataclass_params__", None), "frozen", False)
        ),
        "d1_margin_threshold": float(policy.d1_margin_threshold),
        "d2_margin_threshold": float(policy.d2_margin_threshold),
        "max_speculative_depth": int(policy.max_speculative_depth),
        "target_routes": {"K1": "M2", "K2": "M3", "K3": "M4"},
        "target_rows": rows,
    }


def _run_adaptive_width_bracket(
    *,
    rt,
    prompt_ids: list[int],
    args,
    common_receipt: dict,
    out_stem: Path,
) -> int:
    from mtplx.deepseek_v4_adaptive_width import (
        install_deepseek_v4_adaptive_width_policy,
    )
    from mtplx.sampling import SamplerConfig

    sampler = SamplerConfig(temperature=0.0)
    policy = install_deepseek_v4_adaptive_width_policy(
        rt,
        sampler=sampler,
        draft_sampler=None,
        speculative_depth=3,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        mtp_history_policy=args.mtp_history_policy,
    )
    policy_receipt = _installed_policy_receipt(policy)
    arms: list[dict] = []
    for label, enabled in _ADAPTIVE_WIDTH_BRACKET_ARMS:
        _reset_benchmark_state(rt)
        arm = _run_arm(
            rt=rt,
            label=label,
            depth=3,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens,
            verify_strategy=args.verify_strategy,
            verify_core=args.verify_core,
            mtp_history_policy=args.mtp_history_policy,
            baseline_tokens=None,
            adaptive_width_policy=policy if enabled else None,
        )
        if isinstance(arm.get("tokens"), list):
            arm["token_sha256"] = _token_sha256(arm["tokens"])
        arms.append(arm)

    receipt = _adaptive_width_bracket_receipt(
        common=common_receipt,
        arms=arms,
        process_pid=os.getpid(),
        model_object_id=id(rt.model),
        policy_receipt=policy_receipt,
    )
    _write_pair_receipt(out_stem, receipt, prompt_ids, args.prompt_file)
    print(f"[adaptive width] wrote {out_stem.with_suffix('.json')}")
    print(json.dumps(receipt["policy_engagement"], sort_keys=True))
    print(json.dumps(receipt["token_quality"], sort_keys=True))
    print(json.dumps(receipt["performance"], sort_keys=True))
    sys.stdout.flush()
    return int(receipt["status"])


def _attn_proj_wide_m3_arm_binding(rt) -> dict:
    """Read the complete module arm outside measured generation."""

    selector = getattr(rt.model, "_mtplx_dsv4_attn_proj_wide_m3_selector", None)
    projections = tuple(getattr(selector, "projections", ()))
    return {
        "selected": bool(getattr(selector, "candidate_selected", False)),
        "projections": len(projections),
        "original_stock_modules": sum(
            projection.owner.wq_b is projection.stock for projection in projections
        ),
        "candidate_modules": sum(
            projection.owner.wq_b is projection.candidate for projection in projections
        ),
    }


def _attn_proj_wide_m3_bracket_receipt(
    *,
    common: dict,
    arms: list[dict],
    process_pid: int,
    model_object_id: int,
    policy_receipt: dict,
    route_report: dict,
) -> dict:
    """Fail closed on one-load stock/attention-M3 bracket drift."""

    errors = _adaptive_width_common_errors(
        {**common, "launch_mtplx_env": dict(_ADAPTIVE_WIDTH_STAGE4_ENV)}
    )
    if common.get("launch_mtplx_env") != _ATTN_PROJ_WIDE_M3_STAGE4_ENV:
        errors.append("attention M3-wide launch environment is not canonical")
    if common.get("diagnostic_profiler_evidence") != _ATTN_PROJ_WIDE_M3_PROFILER_EVIDENCE:
        errors.append("attention M3-wide diagnostic profiler evidence changed")
    if route_report != common.get("deepseek_v4_attn_proj_wide_m3"):
        errors.append("attention M3-wide construction receipt changed")
    if route_report != _ATTN_PROJ_WIDE_M3_ROUTE_RECEIPT:
        errors.append("attention M3-wide construction receipt is not canonical")
    if policy_receipt != _ADAPTIVE_WIDTH_POLICY_RECEIPT:
        errors.append("installed D2=10/M4-wide policy receipt is invalid")
    if type(process_pid) is not int or process_pid <= 0:
        errors.append("process identity is invalid")
    if type(model_object_id) is not int or model_object_id <= 0:
        errors.append("model object identity is invalid")

    expected_order = [label for label, _selected in _ATTN_PROJ_WIDE_M3_BRACKET_ARMS]
    arms_by_label = {
        str(arm.get("label")): arm for arm in arms if isinstance(arm, dict)
    }
    if list(arms_by_label) != expected_order:
        errors.append("attention M3-wide arm order is invalid")
    engagements: dict[str, dict] = {}
    for label, expected_selected in _ATTN_PROJ_WIDE_M3_BRACKET_ARMS:
        arm = arms_by_label.get(label, {})
        if arm.get("error") is not None:
            errors.append(f"{label} failed")
        if arm.get("generated_tokens") != 256 or arm.get("finish_reason") != "length":
            errors.append(f"{label} did not complete the canonical workload")
        tokens = arm.get("tokens")
        if not isinstance(tokens, list) or arm.get("token_sha256") != _token_sha256(tokens):
            errors.append(f"{label} token identity is malformed")
        errors.extend(_validate_behavior_stats(label, arm.get("stats_full")))
        engagement, engagement_errors = _adaptive_width_engagement(
            {**arm, "label": "ADAPTIVE-B"}
        )
        errors.extend(f"{label}: {error}" for error in engagement_errors)
        binding = arm.get("attn_proj_wide_m3_binding")
        expected_binding = {
            "selected": expected_selected,
            "projections": 43,
            "original_stock_modules": 0 if expected_selected else 43,
            "candidate_modules": 43 if expected_selected else 0,
        }
        if binding != expected_binding:
            errors.append(f"{label} attention M3-wide binding is not the requested arm")
        histogram = engagement.get("event_derived_width_histogram")
        engagement["attn_proj_wide_m3_binding"] = binding
        engagement["eligible_target_m3_projection_calls"] = (
            histogram.get("K2_M3", 0) * 43
            if expected_selected and isinstance(histogram, dict)
            else 0
        )
        engagements[label] = engagement

    for label in expected_order:
        if (
            engagements.get(label, {}).get("event_derived_width_histogram")
            != _ATTN_PROJ_WIDE_M3_EXPECTED_HISTOGRAM
        ):
            errors.append(f"{label} changed the measured D2=10/M4-wide shape mix")
    if (
        engagements.get("ATTN-PROJ-M3-B", {}).get(
            "eligible_target_m3_projection_calls"
        )
        != 3483
    ):
        errors.append("candidate did not expose exactly 81 M3 x 43 projection calls")

    controls = [
        arms_by_label.get(label, {}).get("tokens")
        for label in ("CURRENT-PRIMER", "CURRENT-C0", "CURRENT-C1")
    ]
    controls_equal = (
        all(isinstance(tokens, list) and len(tokens) == 256 for tokens in controls)
        and controls[0] == controls[1] == controls[2]
    )
    if not controls_equal:
        errors.append("current controls are not token-identical")
    candidate_tokens = arms_by_label.get("ATTN-PROJ-M3-B", {}).get("tokens")
    candidate_valid = isinstance(candidate_tokens, list) and len(candidate_tokens) == 256
    if not candidate_valid:
        errors.append("attention M3-wide candidate token sequence is malformed")
    divergent = (
        [
            index
            for index, pair in enumerate(zip(controls[1], candidate_tokens, strict=True))
            if pair[0] != pair[1]
        ]
        if controls_equal and candidate_valid
        else []
    )
    quality = {
        "target_authority_preserved": controls_equal and candidate_valid,
        "mode": "exact" if not divergent else "bf16_near_tie_reported",
        "control_token_sha256": _token_sha256(controls[1]) if controls_equal else None,
        "candidate_token_sha256": (
            _token_sha256(candidate_tokens) if candidate_valid else None
        ),
        "divergent_tokens": len(divergent),
        "first_divergence": None if not divergent else {
            "continuation_index": divergent[0],
            "control_token_id": int(controls[1][divergent[0]]),
            "candidate_token_id": int(candidate_tokens[divergent[0]]),
        },
        "human_eval": "deferred_by_authorized_policy",
    }

    def tps(label: str) -> float:
        value = arms_by_label.get(label, {}).get("decode_tokens_per_second")
        return float(value) if type(value) in {int, float} and value > 0 else 0.0

    c0_tps = tps("CURRENT-C0")
    c1_tps = tps("CURRENT-C1")
    candidate_tps = tps("ATTN-PROJ-M3-B")
    if not c0_tps or not c1_tps or not candidate_tps:
        errors.append("performance cells are missing positive throughput")
    control_mean = (c0_tps + c1_tps) / 2.0
    drift_tps = abs(c1_tps - c0_tps)
    performance = {
        "control_c0_tps": c0_tps,
        "control_c1_tps": c1_tps,
        "control_mean_tps": control_mean,
        "control_drift_tps": drift_tps,
        "candidate_tps": candidate_tps,
        "candidate_minus_control_mean_tps": candidate_tps - control_mean,
        "promotion_floor_tps": control_mean + drift_tps,
        "promotion_pass": candidate_tps > control_mean + drift_tps,
        "reported_below_40_tps": candidate_tps < 40.0,
    }
    return {
        **common,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "receipt_role": "attn_proj_wide_m3_performance_bracket",
        "performance_eligible": True,
        "single_process_bracket": {
            "process_pid": process_pid,
            "model_object_id": model_object_id,
            "model_load_count": 1,
            "execution_order": expected_order,
            "discarded_primer": "CURRENT-PRIMER",
        },
        "policy": policy_receipt,
        "policy_engagement": engagements,
        "token_quality": quality,
        "performance": performance,
        "arms": arms,
        "validation_errors": errors,
        "status": int(bool(errors)),
    }


def _run_attn_proj_wide_m3_bracket(
    *, rt, prompt_ids, args, common_receipt, out_stem
) -> int:
    """Run discarded primer, C0, candidate, and C1 on one loaded model."""

    from mtplx.deepseek_v4_adaptive_width import (
        install_deepseek_v4_adaptive_width_policy,
    )
    from mtplx.deepseek_v4_attn_proj_wide_m3 import (
        select_deepseek_v4_attn_proj_wide_m3_arm,
    )
    from mtplx.sampling import SamplerConfig

    policy = install_deepseek_v4_adaptive_width_policy(
        rt,
        sampler=SamplerConfig(temperature=0.0),
        draft_sampler=None,
        speculative_depth=3,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        mtp_history_policy=args.mtp_history_policy,
    )
    policy_receipt = _installed_policy_receipt(policy)
    route_report = rt.deepseek_v4_attn_proj_wide_m3_report
    arms: list[dict] = []
    try:
        for label, enabled in _ATTN_PROJ_WIDE_M3_BRACKET_ARMS:
            select_deepseek_v4_attn_proj_wide_m3_arm(rt.model, enabled)
            binding = _attn_proj_wide_m3_arm_binding(rt)
            _reset_benchmark_state(rt)
            arm = _run_arm(
                rt=rt,
                label=label,
                depth=3,
                prompt_ids=prompt_ids,
                max_tokens=args.max_tokens,
                verify_strategy=args.verify_strategy,
                verify_core=args.verify_core,
                mtp_history_policy=args.mtp_history_policy,
                baseline_tokens=None,
                adaptive_width_policy=policy,
            )
            if isinstance(arm.get("tokens"), list):
                arm["token_sha256"] = _token_sha256(arm["tokens"])
            arm["attn_proj_wide_m3_binding"] = binding
            arms.append(arm)
    finally:
        select_deepseek_v4_attn_proj_wide_m3_arm(rt.model, False)
    receipt = _attn_proj_wide_m3_bracket_receipt(
        common=common_receipt,
        arms=arms,
        process_pid=os.getpid(),
        model_object_id=id(rt.model),
        policy_receipt=policy_receipt,
        route_report=route_report,
    )
    _write_pair_receipt(out_stem, receipt, prompt_ids, args.prompt_file)
    print(f"[attention M3-wide] wrote {out_stem.with_suffix('.json')}")
    print(json.dumps(receipt["policy_engagement"], sort_keys=True))
    print(json.dumps(receipt["token_quality"], sort_keys=True))
    print(json.dumps(receipt["performance"], sort_keys=True))
    sys.stdout.flush()
    return int(receipt["status"])


def _attention_island_binding(rt) -> dict:
    selector = getattr(
        rt.model, "_mtplx_dsv4_attention_island_selector", None
    )
    route = getattr(rt.model, "_target_hc_hidden_route", None)
    return {
        "selected": bool(getattr(selector, "candidate_selected", False)),
        "selector_present": selector is not None,
        "route_is_stock": route is getattr(selector, "stock", None),
        "route_is_candidate": route is getattr(selector, "candidate", None),
    }


def _attention_island_signatures(engagement: dict) -> list[str]:
    histogram = engagement.get("event_derived_width_histogram")
    if not isinstance(histogram, dict):
        return []
    widths = (
        (2, histogram.get("K1_M2")),
        (3, histogram.get("K2_M3")),
        (4, histogram.get("K3_M4")),
    )
    return sorted(
        f"M{width}:{layout}"
        for width, count in widths
        if type(count) is int and count > 0
        for layout in _ATTENTION_ISLAND_LAYOUTS
    )


def _load_attention_island_paired_near_tie_evidence(
    bench_dir: Path,
    *,
    expected_sha256: str = _ATTENTION_ISLAND_PAIRED_QUALITY_SHA256,
) -> dict:
    """Authenticate the paired teacher-forced evidence before timed arms."""

    path = Path(bench_dir) / _ATTENTION_ISLAND_PAIRED_QUALITY_FILENAME
    encoded = path.read_bytes()
    observed_sha256 = hashlib.sha256(encoded).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "paired near-tie receipt SHA mismatch: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    payload = json.loads(encoded)
    acceptance = payload.get("quality_acceptance") or {}
    flip = acceptance.get("single_flip") or {}
    execution = payload.get("execution_contract") or {}
    expected_schedule = {
        "cached_gap_to_gather_selected": 0.25,
        "gather_gap_to_cached_selected": 0.0,
    }
    checks = {
        "complete": payload.get("status") == "COMPLETE",
        "quality_gate": payload.get("quality_gate_pass") is True,
        "verdict": (
            payload.get("quality_verdict")
            == _ATTENTION_ISLAND_PAIRED_NEAR_TIE["quality_verdict"]
        ),
        "errors": payload.get("errors") == [],
        "strict_errors": payload.get("strict_validation_errors") == [],
        "policy": (
            acceptance.get("policy")
            == "exact_or_single_identical_bf16_top2_flip"
        ),
        "mode": acceptance.get("accepted_mode") == "single_identical_bf16_top2_flip",
        "continuation_index": flip.get("continuation_index") == 221,
        "absolute_target_position": flip.get("absolute_target_position") == 549,
        "control_token": flip.get("cached_selected_id") == 14042,
        "candidate_token": flip.get("gather_selected_id") == 12258,
        "ar_gap": flip.get("AR") == expected_schedule,
        "k3_gap": flip.get("K3_TARGET_ROWS") == expected_schedule,
        "teacher_forced": execution.get("teacher_forced") is True,
        "no_hot_instrumentation": (
            execution.get("production_hot_path_instrumentation") is False
        ),
        "one_model": execution.get("model_objects") == 1,
        "one_load": execution.get("model_load_count") == 1,
        "memory_safe": (
            execution.get("memory_safe_sequential_evaluation") is True
        ),
        "ar_rows": execution.get("ar_rows") == 256,
        "k3_rows": execution.get("k3_target_rows") == 256,
        "k3_physical_m": execution.get("k3_physical_m") == 4,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "paired near-tie receipt failed authenticated contract: "
            + ", ".join(failed)
        )
    return dict(_ATTENTION_ISLAND_PAIRED_NEAR_TIE)


def _attention_island_token_quality(control, candidate, paired_evidence) -> dict:
    valid = (
        isinstance(control, list)
        and isinstance(candidate, list)
        and len(control) == len(candidate) == 256
    )
    divergent = (
        [
            index
            for index, pair in enumerate(zip(control, candidate, strict=True))
            if pair[0] != pair[1]
        ]
        if valid
        else []
    )
    base = {
        "policy": "exact_or_source_bound_paired_bf16_near_tie",
        "valid_complete_sequences": valid,
        "exact": valid and not divergent,
        "divergent_tokens": len(divergent),
        "first_divergence": (
            None
            if not divergent
            else {
                "continuation_index": divergent[0],
                "control_token_id": int(control[divergent[0]]),
                "candidate_token_id": int(candidate[divergent[0]]),
            }
        ),
        "human_eval": "deferred_by_authorized_policy",
    }
    if not valid:
        return {**base, "accepted": False, "mode": "incomplete_sequences"}
    if not divergent:
        return {**base, "accepted": True, "mode": "exact"}

    evidence_valid = paired_evidence == _ATTENTION_ISLAND_PAIRED_NEAR_TIE
    trigger = divergent[0]
    source_bound_flip = (
        evidence_valid
        and trigger == paired_evidence["continuation_index"]
        and control[trigger] == paired_evidence["control_token_id"]
        and candidate[trigger] == paired_evidence["candidate_token_id"]
    )
    if source_bound_flip:
        tail = [index for index in divergent if index > trigger]
        return {
            **base,
            "accepted": True,
            "mode": "source_bound_paired_bf16_near_tie",
            "paired_evidence": paired_evidence,
            "propagated_tail": {
                "start_continuation_index": trigger + 1,
                "divergent_tokens": len(tail),
            },
        }
    return {
        **base,
        "accepted": False,
        "mode": "unapproved_divergence",
        "paired_evidence": paired_evidence if evidence_valid else None,
    }


def _run_attention_island_bracket(
    *, rt, prompt_ids, args, common_receipt, out_stem
) -> int:
    """Candidate primer, current C0, candidate B, current C1 in one load."""

    from mtplx import deepseek_v4_attention_island as island_module
    from mtplx.deepseek_v4_adaptive_width import (
        install_deepseek_v4_adaptive_width_policy,
    )
    from mtplx.deepseek_v4_attention_island import (
        select_deepseek_v4_attention_island_arm,
    )
    from mtplx.sampling import SamplerConfig

    policy = install_deepseek_v4_adaptive_width_policy(
        rt,
        sampler=SamplerConfig(temperature=0.0),
        draft_sampler=None,
        speculative_depth=3,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        mtp_history_policy=args.mtp_history_policy,
    )
    policy_receipt = _installed_policy_receipt(policy)
    route_report = rt.deepseek_v4_attention_island_report
    arms: list[dict] = []
    try:
        for label, enabled in _ATTENTION_ISLAND_BRACKET_ARMS:
            select_deepseek_v4_attention_island_arm(rt.model, enabled)
            binding = _attention_island_binding(rt)
            tape_count_before = len(island_module._TAPES)
            _reset_benchmark_state(rt)
            arm = _run_arm(
                rt=rt,
                label=label,
                depth=3,
                prompt_ids=prompt_ids,
                max_tokens=args.max_tokens,
                verify_strategy=args.verify_strategy,
                verify_core=args.verify_core,
                mtp_history_policy=args.mtp_history_policy,
                baseline_tokens=None,
                adaptive_width_policy=policy,
            )
            if isinstance(arm.get("tokens"), list):
                arm["token_sha256"] = _token_sha256(arm["tokens"])
            arm["attention_island_binding"] = binding
            arm["attention_island_tape_count_before"] = tape_count_before
            arm["attention_island_tape_count_after"] = len(island_module._TAPES)
            arms.append(arm)
    finally:
        select_deepseek_v4_attention_island_arm(rt.model, False)

    errors = _adaptive_width_common_errors(
        {**common_receipt, "launch_mtplx_env": dict(_ADAPTIVE_WIDTH_STAGE4_ENV)}
    )
    if common_receipt.get("launch_mtplx_env") != _ATTENTION_ISLAND_STAGE4_ENV:
        errors.append("attention-island launch environment is not canonical")
    if not isinstance(route_report, dict):
        errors.append("attention-island construction receipt is absent")
    else:
        for key, expected in (
            ("installed", True),
            ("widths", [2, 3, 4]),
            ("body_layers", 43),
            ("bound_layer_routes", 129),
            ("shared_tapes", 9),
            ("expected_shared_tapes", 9),
            ("attention", "eager_exact_logical_cache"),
            ("weight_binding", "explicit_array_inputs"),
            ("runtime_fallback", False),
            ("hot_environment_reads", False),
            ("hot_counters", False),
        ):
            if route_report.get(key) != expected:
                errors.append(
                    f"attention-island construction {key} changed: "
                    f"{route_report.get(key)!r}"
                )
    expected_order = [label for label, _enabled in _ATTENTION_ISLAND_BRACKET_ARMS]
    by_label = {arm.get("label"): arm for arm in arms}
    if list(by_label) != expected_order:
        errors.append("attention-island arm order is invalid")
    engagements: dict[str, dict] = {}
    for label, enabled in _ATTENTION_ISLAND_BRACKET_ARMS:
        arm = by_label.get(label, {})
        if arm.get("error") is not None:
            errors.append(f"{label} failed")
        if arm.get("generated_tokens") != 256 or arm.get("finish_reason") != "length":
            errors.append(f"{label} did not complete the canonical workload")
        tokens = arm.get("tokens")
        if not isinstance(tokens, list) or arm.get("token_sha256") != _token_sha256(tokens):
            errors.append(f"{label} token identity is malformed")
        errors.extend(_validate_behavior_stats(label, arm.get("stats_full")))
        engagement, engagement_errors = _adaptive_width_engagement(arm)
        errors.extend(f"{label}: {error}" for error in engagement_errors)
        engagements[label] = engagement
        expected_binding = {
            "selected": enabled,
            "selector_present": True,
            "route_is_stock": not enabled,
            "route_is_candidate": enabled,
        }
        if arm.get("attention_island_binding") != expected_binding:
            errors.append(f"{label} did not bind its requested complete arm")

    for label in ("CURRENT-C0", "CURRENT-C1"):
        if (
            engagements.get(label, {}).get("event_derived_width_histogram")
            != _ATTENTION_ISLAND_CONTROL_HISTOGRAM
        ):
            errors.append(f"{label} changed the authoritative 6/76/10 control mix")

    complete_signatures = sorted(
        f"M{width}:{layout}"
        for width in (2, 3, 4)
        for layout in _ATTENTION_ISLAND_LAYOUTS
    )
    primer_signatures = _attention_island_signatures(
        engagements.get("ATTENTION-ISLAND-PRIMER", {})
    )
    candidate_signatures = _attention_island_signatures(
        engagements.get("ATTENTION-ISLAND-B", {})
    )
    unprimed = sorted(set(candidate_signatures) - set(primer_signatures))
    if primer_signatures != complete_signatures:
        errors.append("candidate primer did not exercise all nine tape classes")
    if unprimed:
        errors.append(f"candidate B reached unprimed tape classes: {unprimed}")
    primer = by_label.get("ATTENTION-ISLAND-PRIMER", {})
    candidate_arm = by_label.get("ATTENTION-ISLAND-B", {})
    if primer.get("attention_island_tape_count_after") != 9:
        errors.append("candidate primer did not materialize exactly nine tapes")
    if (
        candidate_arm.get("attention_island_tape_count_before") != 9
        or candidate_arm.get("attention_island_tape_count_after") != 9
    ):
        errors.append("candidate B entered a new Python tape compilation class")

    c0_tokens = by_label.get("CURRENT-C0", {}).get("tokens")
    c1_tokens = by_label.get("CURRENT-C1", {}).get("tokens")
    candidate_tokens = candidate_arm.get("tokens")
    if c0_tokens != c1_tokens:
        errors.append("current PR223 control token streams drifted")
    quality = _attention_island_token_quality(
        c0_tokens,
        candidate_tokens,
        common_receipt.get("paired_near_tie_evidence"),
    )
    if not quality["accepted"]:
        errors.append(
            "candidate/control token quality failed: " f"{quality['mode']}"
        )

    def tps(label: str) -> float:
        value = by_label.get(label, {}).get("decode_tokens_per_second")
        return float(value) if type(value) in {int, float} and value > 0 else 0.0

    c0 = tps("CURRENT-C0")
    candidate = tps("ATTENTION-ISLAND-B")
    c1 = tps("CURRENT-C1")
    if not c0 or not candidate or not c1:
        errors.append("performance cells are missing positive throughput")
    control_mean = (c0 + c1) / 2.0
    drift = abs(c1 - c0)
    performance = {
        "control_c0_tps": c0,
        "control_c1_tps": c1,
        "control_mean_tps": control_mean,
        "control_drift_tps": drift,
        "candidate_tps": candidate,
        "candidate_minus_control_mean_tps": candidate - control_mean,
        "promotion_floor_tps": control_mean + drift,
        "promotion_pass": candidate > control_mean + drift,
        "above_40_tps": candidate >= 40.0,
        "above_50_tps": candidate >= 50.0,
    }
    receipt = {
        **common_receipt,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "receipt_role": "attention_island_performance_bracket",
        "performance_eligible": True,
        "single_process_bracket": {
            "process_pid": os.getpid(),
            "model_object_id": id(rt.model),
            "model_load_count": 1,
            "execution_order": expected_order,
            "discarded_primer": "ATTENTION-ISLAND-PRIMER",
        },
        "deepseek_v4_attention_island": route_report,
        "policy": policy_receipt,
        "policy_engagement": engagements,
        "compiled_tape_warmth": {
            "complete": complete_signatures,
            "primer": primer_signatures,
            "candidate": candidate_signatures,
            "unprimed": unprimed,
            "python_tapes_before_b": candidate_arm.get(
                "attention_island_tape_count_before"
            ),
            "python_tapes_after_b": candidate_arm.get(
                "attention_island_tape_count_after"
            ),
        },
        "token_quality": quality,
        "performance": performance,
        "arms": arms,
        "validation_errors": errors,
        "status": int(bool(errors)),
    }
    _write_pair_receipt(out_stem, receipt, prompt_ids, args.prompt_file)
    print(f"[attention island] wrote {out_stem.with_suffix('.json')}")
    print(json.dumps(receipt["compiled_tape_warmth"], sort_keys=True))
    print(json.dumps(quality, sort_keys=True))
    print(json.dumps(performance, sort_keys=True))
    sys.stdout.flush()
    return int(receipt["status"])


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
    required_mlx_identity = _OFFICIAL_MLX_IDENTITY
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
        "--adaptive-width-bracket",
        action="store_true",
        help="one-load fixed-C0/adaptive-B/fixed-C1 max-K3 bracket",
    )
    ap.add_argument(
        "--attn-proj-wide-m3-bracket",
        action="store_true",
        help="one-load stock/attention-projection-M3/stock D2=10 bracket",
    )
    ap.add_argument(
        "--attention-island-bracket",
        action="store_true",
        help="one-load candidate-primer/current/attention-island/current bracket",
    )
    ap.add_argument("--expected-source-commit")
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
    if sum(
        (
            args.moe_tail_bracket,
            args.adaptive_width_bracket,
            args.attn_proj_wide_m3_bracket,
            args.attention_island_bracket,
        )
    ) > 1:
        sys.exit("bracket modes are mutually exclusive")
    if (
        args.adaptive_width_bracket
        and launch_mtplx_env != _ADAPTIVE_WIDTH_STAGE4_ENV
    ):
        sys.exit(
            "--adaptive-width-bracket requires the exact seven-variable "
            f"Stage-4 environment: {launch_mtplx_env}"
        )
    if (
        args.attn_proj_wide_m3_bracket
        and launch_mtplx_env != _ATTN_PROJ_WIDE_M3_STAGE4_ENV
    ):
        sys.exit(
            "--attn-proj-wide-m3-bracket requires the exact Stage-4 environment: "
            f"{launch_mtplx_env}"
        )
    if (
        args.attention_island_bracket
        and launch_mtplx_env != _ATTENTION_ISLAND_STAGE4_ENV
    ):
        sys.exit(
            "--attention-island-bracket requires the exact current "
            f"Stage-4 environment: {launch_mtplx_env}"
        )
    if args.attention_island_bracket and args.expected_source_commit != source_commit:
        sys.exit(
            "--attention-island-bracket source commit attestation failed: "
            f"expected={args.expected_source_commit!r} observed={source_commit!r}"
        )

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

    if args.attention_island_bracket:
        if args.tiny:
            sys.exit("--attention-island-bracket requires the canonical GPU model")
        if list(args.depths) != [3] or args.max_tokens != 256 or not args.out:
            sys.exit(
                "--attention-island-bracket requires --depths 3 "
                "--max-tokens 256 --out"
            )
        if (
            args.verify_strategy != "capture_commit"
            or args.verify_core != "stock"
            or args.mtp_history_policy != "committed"
        ):
            sys.exit(
                "--attention-island-bracket requires "
                "capture_commit/stock/committed"
            )
        if prompt_identity != {
            "path": str(prompt_path),
            "sha256": _CANONICAL_PROMPT_SHA256,
            "tokens": 328,
        }:
            sys.exit(f"canonical prompt identity mismatch: {prompt_identity}")
        route_report = rt.deepseek_v4_attention_island_report
        if not isinstance(route_report, dict) or route_report.get("shared_tapes") != 9:
            sys.exit(
                "attention-island construction gate did not install nine tapes: "
                f"{route_report}"
            )
        attn_report = rt.deepseek_v4_attn_proj_wide_m3_report
        if attn_report != _ATTN_PROJ_WIDE_M3_ROUTE_RECEIPT:
            sys.exit(
                "attention-island bracket requires the current PR223 M3-wide route: "
                f"{attn_report}"
            )
        try:
            paired_near_tie_evidence = (
                _load_attention_island_paired_near_tie_evidence(
                    Path(args.out).parent
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            sys.exit(f"paired near-tie evidence gate failed: {error}")
        common_receipt = {
            "harness": "scripts/deepseek_v4_mtpk_bench.py",
            "source_commit": source_commit,
            "source_commit_attestation": {
                "expected": args.expected_source_commit,
                "observed": source_commit,
                "match": args.expected_source_commit == source_commit,
                "clean": True,
            },
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
            "model_path": str(model_path),
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
            "sampling": {
                "greedy": True,
                "temperature": 0.0,
                "stop_token_ids": [],
            },
            "prompt_file": str(prompt_path),
            "prompt": prompt_identity,
            "prompt_tokens": len(prompt_ids),
            "max_tokens": args.max_tokens,
            "depths": [3],
            "verify_strategy": args.verify_strategy,
            "verify_core": args.verify_core,
            "mtp_history_policy": args.mtp_history_policy,
            "fp32_activations": _fp32_activations_env(),
            "load_seconds": load_seconds,
            "active_after_load_gib": _gib(after_load_active),
            "deepseek_v4_moe_tail": moe_tail_report,
            "deepseek_v4_o_lora": rt.deepseek_v4_o_lora_report,
            "deepseek_v4_attn_proj_wide_m3": attn_report,
            "deepseek_v4_attention_island": route_report,
            "paired_near_tie_evidence": paired_near_tie_evidence,
        }
        return _run_attention_island_bracket(
            rt=rt,
            prompt_ids=prompt_ids,
            args=args,
            common_receipt=common_receipt,
            out_stem=Path(args.out),
        )

    if args.adaptive_width_bracket:
        if args.tiny:
            sys.exit("--adaptive-width-bracket requires the canonical GPU model")
        if args.moe_tail_bracket:
            sys.exit("adaptive-width and MoE-tail bracket modes are exclusive")
        if list(args.depths) != [3] or args.max_tokens != 256:
            sys.exit("--adaptive-width-bracket requires --depths 3 --max-tokens 256")
        if not args.out:
            sys.exit("--adaptive-width-bracket requires --out")
        if (
            args.verify_strategy != "capture_commit"
            or args.verify_core != "stock"
            or args.mtp_history_policy != "committed"
        ):
            sys.exit(
                "--adaptive-width-bracket requires capture_commit/stock/committed"
            )
        if prompt_identity != {
            "path": str(prompt_path),
            "sha256": _CANONICAL_PROMPT_SHA256,
            "tokens": 328,
        }:
            sys.exit(f"canonical prompt identity mismatch: {prompt_identity}")
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
            "launch_mtplx_env": launch_mtplx_env,
            "guard_window": guard_window,
            "model_path": str(model_path),
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
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
            "load_seconds": load_seconds,
            "active_after_load_gib": _gib(after_load_active),
            "deepseek_v4_moe_tail": moe_tail_report,
            "deepseek_v4_o_lora": rt.deepseek_v4_o_lora_report,
        }
        return _run_adaptive_width_bracket(
            rt=rt,
            prompt_ids=prompt_ids,
            args=args,
            common_receipt=common_receipt,
            out_stem=Path(args.out),
        )

    if args.attn_proj_wide_m3_bracket:
        if args.tiny:
            sys.exit("--attn-proj-wide-m3-bracket requires the canonical GPU model")
        if list(args.depths) != [3] or args.max_tokens != 256 or not args.out:
            sys.exit(
                "--attn-proj-wide-m3-bracket requires --depths 3 "
                "--max-tokens 256 --out"
            )
        if (
            args.verify_strategy != "capture_commit"
            or args.verify_core != "stock"
            or args.mtp_history_policy != "committed"
        ):
            sys.exit(
                "--attn-proj-wide-m3-bracket requires capture_commit/stock/committed"
            )
        if prompt_identity != {
            "path": str(prompt_path),
            "sha256": _CANONICAL_PROMPT_SHA256,
            "tokens": 328,
        }:
            sys.exit(f"canonical prompt identity mismatch: {prompt_identity}")
        route_report = rt.deepseek_v4_attn_proj_wide_m3_report
        if route_report != _ATTN_PROJ_WIDE_M3_ROUTE_RECEIPT:
            sys.exit(
                "attention M3-wide construction gate did not install the "
                f"canonical plan: {route_report}"
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
                if key.startswith("MTPLX_") or key in ("HF_HUB_OFFLINE", "PYTHONPATH")
            },
            "launch_mtplx_env": launch_mtplx_env,
            "guard_window": guard_window,
            "model_path": str(model_path),
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
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
            "load_seconds": load_seconds,
            "active_after_load_gib": _gib(after_load_active),
            "deepseek_v4_moe_tail": moe_tail_report,
            "deepseek_v4_o_lora": rt.deepseek_v4_o_lora_report,
            "deepseek_v4_attn_proj_wide_m3": route_report,
            "diagnostic_profiler_evidence": dict(
                _ATTN_PROJ_WIDE_M3_PROFILER_EVIDENCE
            ),
        }
        return _run_attn_proj_wide_m3_bracket(
            rt=rt,
            prompt_ids=prompt_ids,
            args=args,
            common_receipt=common_receipt,
            out_stem=Path(args.out),
        )

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
