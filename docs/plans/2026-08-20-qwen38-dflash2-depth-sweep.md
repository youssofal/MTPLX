# Qwen3.8 DFlash2 Stock Depth Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the fastest correct stock DFlash2 width from 1 through 8 on the Qwen3.8 27B Optimized Speed target with a greedy 1,024/1,024 Python workload and bracketed MTPLX MTP depth-3 controls.

**Architecture:** MTPLX loads the local Optimized Speed artifact once after applying the artifact-resolved `turbo` profile. The existing native MTP generator supplies the control; `dflash-mlx` 0.1.10 receives that same loaded target model object and supplies the candidate, using its existing Qwen GDN cache, rollback, DFlash2 draft backend, and target-prefix verification. Construction takes width authority from the Qwen3.8 DFlash2 checkpoint (`block_size=8`, seven drafted tokens) rather than an unrelated generic five-token runtime policy. Phase A records exact token parity and ranks widths by candidate throughput divided by the mean of adjacent MTP controls. It contains no custom kernel or DFlash2 arithmetic change.

**Tech Stack:** Python 3.12, MTPLX 2.9.0 source, MLX 0.32.x, mlx-lm 0.31.x, dflash-mlx 0.1.10, pytest, Ruff, uv, macOS launchd, and `/tmp/mtplx-gpu-exclusive.lock`.

**Assumptions:**

- Assumes `dflash_mlx.engine.target_qwen_gdn.QwenGdnTargetOps` accepts the model object produced by `mtplx.runtime.load` — Phase A stops at the construction gate and does not load a second stock target if that is false.
- Assumes the local Optimized Speed artifact remains at `/Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` — the benchmark will not substitute another quantization if the artifact is absent or changed.
- Assumes DFlash2 checkpoint block size is exactly 8 and target layer IDs are exactly `[5, 19, 33, 47, 61]` — widths 1-8 execute exactly as requested, while widths above 8 are rejected rather than clamped.
- Assumes one guarded child can retain the loaded target while alternating fresh-cache MTP and DFlash2 arms — this plan will not compare separate machines, target bytes, or process-policy snapshots.
- Assumes the live Qwen service is exactly `mtplx-qwen38-27b-optimized-speed` launched by `com.tea.qwen` — the canonical guard refuses any other observed model or launcher state.
- Assumes Phase A identifies a single winner or measured tie band — no custom kernel is selected, implemented, or benchmarked until a separate Phase B plan is written from that result.

---

## File structure

- Modify `pyproject.toml`: pin the competitor extra to the immutable upstream
  `dflash-mlx` HEAD commit `60803233af4589e18588b9bacbb03880801c828a`
  (package version 0.1.10; that version is not published on PyPI).
- Modify `uv.lock`: record the single dependency update without changing unrelated packages.
- Create `tests/test_dflash2_dependency.py`: pin and public/internal API compatibility contract.
- Create `mtplx/benchmarks/dflash2_contract.py`: width parsing, exact Python prompt IDs, immutable row schema, parity checks, and winner/tie selection.
- Create `tests/test_dflash2_contract.py`: no-MLX contract tests.
- Create `mtplx/benchmarks/dflash2_runtime.py`: construction-only bridge from one `MTPLXRuntime` to the unchanged DFlash2 engine.
- Create `tests/test_dflash2_runtime.py`: fake-runtime proof that no second target loader is called and checkpoint geometry fails closed.
- Create `mtplx/benchmarks/runners/dflash2_depth_sweep.py`: oracle, MTP control, DFlash2 candidate, matched bracket ordering, and JSON receipt assembly.
- Create `tests/test_dflash2_depth_sweep.py`: fake-arm parity, bracketing, drift, and tie-break behavior.
- Modify `mtplx/benchmarks/runners/competitor_baselines.py`: replace the obsolete DFlash import path with the current shared-target runner.
- Modify `mtplx/cli.py`: update the existing DFlash defaults and add the width-sweep command.
- Create `tests/test_dflash2_cli.py`: parser defaults and exit behavior.
- Create `scripts/qwen38_dflash2_depth_guarded.py`: consume the canonical guard attestation before importing MLX, run the complete sweep in one child, and write one atomic receipt.
- Create `tests/test_qwen38_dflash2_depth_guarded.py`: source/import-order and argument contract.
- Update `docs/specs/2026-08-20-qwen38-dflash2-mtp-benchmark-design.md`: record Phase A commit, exact commands, and selected width after the guarded run.
- Create `benchmarks/results/qwen38-dflash2-depth1-8-1024x1024-${run_stamp}.json`: immutable Phase A receipt, where `run_stamp=$(date -u +%Y%m%dT%H%M%SZ)` is set once before the guarded command.
- Operational prerequisite outside the PR: extend `/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py` with the exact Qwen3.8 model tuple and process pattern; do not broaden the match.

### Task 1: Upgrade only dflash-mlx to 0.1.10

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `tests/test_dflash2_dependency.py`

**Security flag:** none

**Does NOT cover:** This task does not migrate the runner, load a model, update any other direct or transitive dependency intentionally, or change DFlash2 behavior.

- [x] **Step 1: Write the failing dependency and API contract**

```python
from pathlib import Path
import tomllib


def test_competitor_extra_pins_dflash_mlx_0_1_10():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["competitors"] == [
        "dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx.git@60803233af4589e18588b9bacbb03880801c828a"
    ]


def test_dflash2_runtime_api_contract():
    from importlib.metadata import version

    from dflash_mlx.draft.dflash2 import DFlash2DraftModel
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    from dflash_mlx.runtime import stream_dflash_generate
    from dflash_mlx.runtime.loading import load_draft_bundle

    assert DFlash2DraftModel.__name__ == "DFlash2DraftModel"
    assert version("dflash-mlx") == "0.1.10"
    assert QwenGdnTargetOps.backend_name == "qwen_gdn"
    assert callable(stream_dflash_generate)
    assert callable(load_draft_bundle)
```

- [x] **Step 2: Run the contract and verify the old pin fails**

Run:

```bash
uv run --extra dev --extra competitors python -m pytest -q tests/test_dflash2_dependency.py
```

Expected: FAIL because `pyproject.toml` still pins `dflash-mlx==0.1.0`; an import failure is also acceptable evidence that the old environment lacks DFlash2.

- [x] **Step 3: Update exactly one dependency and regenerate the lock**

Change the competitor extra to:

```toml
competitors = [
  "dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx.git@60803233af4589e18588b9bacbb03880801c828a",
]
```

Then run:

```bash
uv lock --upgrade-package dflash-mlx
uv sync --extra dev --extra competitors
```

Review `git diff -- pyproject.toml uv.lock`; the intended direct change is
`dflash-mlx 0.1.0 -> 0.1.10` sourced from exact Git commit
`60803233af4589e18588b9bacbb03880801c828a`. If unrelated direct dependencies
move, stop and restore only this task's lockfile edits before retrying with the
existing lock constraints.

- [x] **Step 4: Verify imports and the existing competitor-runner tests**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_dependency.py
.venv/bin/python -c 'import dflash_mlx; from dflash_mlx.draft.dflash2 import DFlash2DraftModel; print(DFlash2DraftModel.__name__)'
```

Expected: tests pass and the smoke prints `DFlash2DraftModel`.

- [x] **Step 5: Commit the isolated dependency migration**

```bash
git add pyproject.toml uv.lock tests/test_dflash2_dependency.py
git commit -m "Upgrade dflash-mlx to 0.1.10"
```

### Task 2: Define the no-MLX sweep and winner contract

**Files:**
- Create: `mtplx/benchmarks/dflash2_contract.py`
- Create: `tests/test_dflash2_contract.py`

**Security flag:** none

**Does NOT cover:** These pure functions do not import MLX, load models, execute benchmarks, or decide a custom optimization.

- [x] **Step 1: Write failing width, prompt, parity, and ranking tests**

```python
import pytest

from mtplx.benchmarks.dflash2_contract import (
    DepthBracket,
    build_exact_python_prompt_ids,
    parse_dflash2_widths,
    select_stock_depth,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert "Python" in messages[0]["content"]
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return list(range(1200))


def test_widths_are_exactly_bounded_by_checkpoint():
    assert parse_dflash2_widths("1,3,8") == (1, 3, 8)
    with pytest.raises(ValueError, match="between 1 and 8"):
        parse_dflash2_widths("8,9")


def test_python_prompt_is_exactly_1024_token_ids():
    prompt = build_exact_python_prompt_ids(FakeTokenizer(), token_count=1024)
    assert prompt.token_ids == tuple(range(1024))
    assert prompt.token_count == 1024
    assert len(prompt.token_sha256) == 64
    assert prompt.enable_thinking is False


def test_selection_uses_mtp_normalized_ratio_and_reports_tie_band():
    rows = [
        DepthBracket(2, 60.0, 59.0, 61.0, True),
        DepthBracket(3, 60.3, 59.0, 61.0, True),
        DepthBracket(4, 55.0, 59.0, 61.0, True),
    ]
    selected = select_stock_depth(rows)
    assert selected.best_widths == (2, 3)
    assert selected.needs_tiebreak is True


def test_mismatched_tokens_cannot_enter_selection():
    with pytest.raises(ValueError, match="parity"):
        select_stock_depth([DepthBracket(8, 70.0, 60.0, 60.0, False)])
```

- [x] **Step 2: Run tests and observe the missing module**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_contract.py
```

Expected: FAIL during collection because `mtplx.benchmarks.dflash2_contract` does not exist.

- [x] **Step 3: Implement immutable contracts and exact-width prompt construction**

```python
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import statistics

PYTHON_BENCHMARK_PROMPT = """You are working in a Python 3.11 repository. Read the supplied modules and tests, then implement the requested production-safe fix. Preserve public behavior, use typed code, add a focused pytest regression test, and return code only.\n\n"""


@dataclass(frozen=True)
class ExactPrompt:
    token_ids: tuple[int, ...]
    token_count: int
    token_sha256: str
    enable_thinking: bool


@dataclass(frozen=True)
class DepthBracket:
    width: int
    candidate_decode_tps: float
    control_before_tps: float
    control_after_tps: float
    parity_passed: bool

    @property
    def control_mean_tps(self) -> float:
        return statistics.mean((self.control_before_tps, self.control_after_tps))

    @property
    def normalized_ratio(self) -> float:
        return self.candidate_decode_tps / self.control_mean_tps

    @property
    def drift_fraction(self) -> float:
        return abs(self.control_before_tps - self.control_after_tps) / self.control_mean_tps


@dataclass(frozen=True)
class DepthSelection:
    best_widths: tuple[int, ...]
    needs_tiebreak: bool
    normalized_ratios: tuple[tuple[int, float], ...]


def parse_dflash2_widths(raw: str) -> tuple[int, ...]:
    widths = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not widths or len(widths) != len(set(widths)) or any(not 1 <= width <= 8 for width in widths):
        raise ValueError("DFlash2 widths must be unique integers between 1 and 8")
    return widths


def build_exact_python_prompt_ids(tokenizer, *, token_count: int = 1024) -> ExactPrompt:
    messages = [{"role": "user", "content": PYTHON_BENCHMARK_PROMPT * 64}]
    encoded = list(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))
    if len(encoded) < token_count:
        raise ValueError(f"Python benchmark prompt encoded to {len(encoded)} tokens, expected at least {token_count}")
    token_ids = tuple(int(token) for token in encoded[:token_count])
    digest = hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()
    return ExactPrompt(token_ids, len(token_ids), digest, False)


def select_stock_depth(rows: list[DepthBracket]) -> DepthSelection:
    if not rows or any(not row.parity_passed for row in rows):
        raise ValueError("every ranked DFlash2 bracket must pass exact token parity")
    grouped: dict[int, list[DepthBracket]] = {}
    for row in rows:
        grouped.setdefault(row.width, []).append(row)
    ratios = sorted(
        (
            width,
            statistics.median(row.normalized_ratio for row in width_rows),
            max(row.drift_fraction for row in width_rows),
        )
        for width, width_rows in grouped.items()
    )
    ratios.sort(key=lambda item: item[1], reverse=True)
    leader = ratios[0]
    tie_band = tuple(
        width
        for width, ratio, drift in ratios
        if leader[1] - ratio <= max(leader[2], drift)
    )
    return DepthSelection(
        tie_band,
        len(tie_band) > 1,
        tuple((width, ratio) for width, ratio, _ in ratios),
    )
```

- [x] **Step 4: Run the pure contract tests**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_contract.py
```

Expected: all tests pass without importing MLX.

- [x] **Step 5: Commit**

```bash
git add mtplx/benchmarks/dflash2_contract.py tests/test_dflash2_contract.py
git commit -m "Define DFlash2 depth sweep contract"
```

### Task 3: Construct stock DFlash2 around one MTPLX-loaded target

**Files:**
- Create: `mtplx/benchmarks/dflash2_runtime.py`
- Create: `tests/test_dflash2_runtime.py`

**Security flag:** none

**Does NOT cover:** Construction does not replace DFlash2 cache arithmetic, add an MTPLX target adapter, add fallback loading, or benchmark performance.

- [x] **Step 1: Write failing single-target and geometry tests**

```python
from types import SimpleNamespace
import pytest

from mtplx.benchmarks import dflash2_runtime


def test_bundle_reuses_runtime_model_without_target_reload(monkeypatch):
    target = object()
    runtime = SimpleNamespace(model=target, tokenizer=object(), model_path="speed")
    draft = SimpleNamespace(
        block_size=8,
        target_layer_ids=[5, 19, 33, 47, 61],
        args=SimpleNamespace(block_size=8),
    )
    monkeypatch.setattr(dflash2_runtime, "load_mtplx_runtime", lambda *_a, **_k: runtime)
    monkeypatch.setattr(dflash2_runtime, "load_draft", lambda *_a, **_k: (draft, {"revision": "pinned"}))
    monkeypatch.setattr(dflash2_runtime, "make_target_ops", lambda: SimpleNamespace(supports_model=lambda model: model is target, family=lambda _model: "hybrid_gdn"))
    monkeypatch.setattr(dflash2_runtime, "bind_draft", lambda draft_model, model, target_ops: None)

    bundle = dflash2_runtime.load_mtplx_dflash2_bundle("speed", "z-lab/Qwen3.8-27B-DFlash2")
    assert bundle.runtime is runtime
    assert bundle.target_model is target
    assert bundle.checkpoint_block_size == 8
    assert bundle.draft_model.capabilities.default_block_tokens == 8
    assert bundle.draft_model.capabilities.max_block_tokens == 8
    assert bundle.target_layer_ids == (5, 19, 33, 47, 61)


def test_bundle_rejects_wrong_checkpoint_geometry(monkeypatch):
    runtime = SimpleNamespace(model=object(), tokenizer=object(), model_path="speed")
    draft = SimpleNamespace(block_size=16, target_layer_ids=[1], args=SimpleNamespace(block_size=16))
    monkeypatch.setattr(dflash2_runtime, "load_mtplx_runtime", lambda *_a, **_k: runtime)
    monkeypatch.setattr(dflash2_runtime, "load_draft", lambda *_a, **_k: (draft, {}))
    monkeypatch.setattr(dflash2_runtime, "make_target_ops", lambda: SimpleNamespace(supports_model=lambda _model: True, family=lambda _model: "hybrid_gdn"))
    with pytest.raises(ValueError, match="block size 8"):
        dflash2_runtime.load_mtplx_dflash2_bundle("speed", "draft")
```

- [x] **Step 2: Verify the construction API is absent**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_runtime.py
```

Expected: FAIL because the runtime bridge does not exist.

- [x] **Step 3: Implement a construction-only bundle with fixed callables**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MTPLXDFlash2Bundle:
    runtime: Any
    target_model: Any
    tokenizer: Any
    target_ops: Any
    draft_model: Any
    draft_backend: Any
    draft_meta: dict[str, Any]
    checkpoint_block_size: int
    target_layer_ids: tuple[int, ...]


def load_mtplx_runtime(model_path: str):
    from mtplx.runtime import load
    return load(model_path, mtp=True)


def load_draft(draft_ref: str, *, draft_quant: str = "w4:gs64"):
    from dflash_mlx.runtime.loading import load_draft_bundle
    return load_draft_bundle(draft_ref, lazy=True, draft_quant=draft_quant)


def make_target_ops():
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    return QwenGdnTargetOps()


def bind_draft(draft_model, target_model, target_ops):
    from dflash_mlx.engine.target_ops import bind_draft_to_target
    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)


def load_mtplx_dflash2_bundle(model_path: str, draft_ref: str) -> MTPLXDFlash2Bundle:
    runtime = load_mtplx_runtime(model_path)
    target_model = runtime.model
    target_ops = make_target_ops()
    if not target_ops.supports_model(target_model) or target_ops.family(target_model) != "hybrid_gdn":
        raise ValueError("DFlash2 requires the MTPLX-loaded Qwen hybrid GDN target")
    draft_model, draft_meta = load_draft(draft_ref)
    block_size = draft_model.block_size
    if type(block_size) is not int:
        raise ValueError("Qwen3.8 DFlash2 checkpoint block size must be an integer")
    raw_layer_ids = draft_model.target_layer_ids
    if not isinstance(raw_layer_ids, (list, tuple)) or any(
        type(value) is not int for value in raw_layer_ids
    ):
        raise ValueError("Qwen3.8 DFlash2 target layer IDs must be integers")
    layer_ids = tuple(raw_layer_ids)
    if block_size != 8:
        raise ValueError(f"Qwen3.8 DFlash2 checkpoint must have block size 8, got {block_size}")
    if layer_ids != (5, 19, 33, 47, 61):
        raise ValueError(f"Qwen3.8 DFlash2 target layer IDs differ: {layer_ids}")
    draft_model.capabilities = replace(
        draft_model.capabilities,
        default_block_tokens=block_size,
        max_block_tokens=block_size,
    )
    bind_draft(draft_model, target_model, target_ops)
    from dflash_mlx.draft_backend import EagerDraftBackend
    return MTPLXDFlash2Bundle(runtime, target_model, runtime.tokenizer, target_ops, draft_model, EagerDraftBackend(), dict(draft_meta), block_size, layer_ids)
```

Keep dependency imports inside construction functions so base MTPLX installs without the competitor extra still import normally. Before `load_mtplx_dflash2_bundle`, the caller resolves model runtime overrides with `inspect_model`, applies `apply_profile_env("turbo", runtime_env_overrides=...)`, and installs the existing q4/group-64 draft lm-head for the MTP control.

- [x] **Step 4: Run focused tests and an import-only smoke**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_runtime.py
.venv/bin/python -c 'from mtplx.benchmarks.dflash2_runtime import MTPLXDFlash2Bundle; print(MTPLXDFlash2Bundle.__name__)'
```

Expected: tests pass and the smoke does not load MLX model weights.

- [x] **Step 5: Commit**

```bash
git add mtplx/benchmarks/dflash2_runtime.py tests/test_dflash2_runtime.py
git commit -m "Connect DFlash2 to an MTPLX target"
```

### Task 4: Implement the greedy matched depth sweep

**Files:**
- Create: `mtplx/benchmarks/runners/dflash2_depth_sweep.py`
- Create: `tests/test_dflash2_depth_sweep.py`

**Security flag:** none

**Does NOT cover:** The runner does not alter DFlash2, tune kernels, sample stochastically, stop on EOS, compare unmatched timing windows, or declare a final MTP win.

- [x] **Step 1: Write failing orchestration tests with fake arms**

```python
from mtplx.benchmarks.runners.dflash2_depth_sweep import run_dflash2_depth_sweep


def test_sweep_brackets_every_width_and_requires_oracle_parity():
    calls = []
    oracle = tuple(range(1024))

    def run_arm(kind, width):
        calls.append((kind, width))
        tps = 60.0 if kind == "mtp" else 62.0 + width
        return {"tokens": oracle, "decode_tps": tps, "generated_tokens": 1024}

    result = run_dflash2_depth_sweep(
        bundle=object(),
        prompt_ids=tuple(range(1024)),
        widths=(1, 2),
        repetitions=1,
        oracle_tokens=oracle,
        arm_runner=run_arm,
    )
    assert calls == [("mtp", 3), ("dflash2", 1), ("mtp", 3), ("mtp", 3), ("dflash2", 2), ("mtp", 3)]
    assert result["workload"] == {"prompt_tokens": 1024, "generated_tokens": 1024, "greedy": True}
    assert all(row["parity_passed"] for row in result["brackets"])


def test_sweep_rejects_short_or_divergent_candidate():
    oracle = tuple(range(1024))

    def run_arm(kind, width):
        tokens = oracle if kind == "mtp" else oracle[:-1]
        return {"tokens": tokens, "decode_tps": 70.0, "generated_tokens": len(tokens)}

    result = run_dflash2_depth_sweep(
        bundle=object(),
        prompt_ids=tuple(range(1024)),
        widths=(8,),
        repetitions=1,
        oracle_tokens=oracle,
        arm_runner=run_arm,
    )
    assert result["selection"] is None
    assert result["brackets"][0]["parity_passed"] is False
```

- [x] **Step 2: Run the tests and observe the missing runner**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_depth_sweep.py
```

Expected: FAIL during collection because the runner does not exist.

- [x] **Step 3: Implement exact oracle, MTP, and DFlash2 arm functions**

The production arm runner uses these fixed contracts:

```python
from dataclasses import asdict
import hashlib

from mtplx.benchmarks.dflash2_contract import DepthBracket, select_stock_depth
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.sampling import SamplerConfig


GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
MTP_DEPTH = 3


def run_target_oracle(bundle, prompt_ids, *, max_tokens=1024):
    output = generate_ar(
        bundle.runtime,
        list(prompt_ids),
        max_tokens=max_tokens,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
    )
    return tuple(int(token) for token in output.tokens)


def run_mtp_control(bundle, prompt_ids, *, max_tokens=1024):
    output = generate_mtpk(
        bundle.runtime,
        list(prompt_ids),
        max_tokens=max_tokens,
        sampler=GREEDY,
        speculative_depth=MTP_DEPTH,
        seed=0,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        mtp_cache_policy="persistent",
        mtp_history_policy="cycle",
    )
    return arm_receipt_from_mtplx(output)


def run_dflash2_candidate(
    bundle,
    prompt_ids,
    width,
    runtime_context,
    *,
    max_tokens=1024,
):
    events = stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt_tokens_override=list(prompt_ids),
        prompt="",
        use_chat_template=False,
        max_new_tokens=max_tokens,
        block_tokens=int(width),
        stop_token_ids=[],
        runtime_context=runtime_context,
    )
    return arm_receipt_from_dflash_events(events, requested_width=width)


def build_fixed_dflash_runtime_context():
    from dflash_mlx.runtime.context import build_offline_runtime_context

    return build_offline_runtime_context(
        quantize_kv_cache=False,
        verify_mode="dflash",
        copyspec_mode="off",
    )


def arm_receipt_from_mtplx(output):
    return {
        "tokens": tuple(int(token) for token in output.tokens),
        "generated_tokens": int(output.stats.generated_tokens),
        "decode_tps": float(output.stats.decode_tok_s),
        "elapsed_s": float(output.stats.elapsed_s),
        "peak_memory_gb": float(output.stats.peak_memory_bytes) / (1024**3),
        "verify_calls": int(output.stats.verify_calls),
        "accepted_by_depth": list(output.stats.accepted_by_depth),
        "engine": "mtplx_mtp",
    }


def arm_receipt_from_dflash_events(events, *, requested_width):
    from dflash_mlx.engine.events import SummaryEvent

    summary = None
    for event in events:
        if isinstance(event, SummaryEvent):
            summary = event
    if summary is None:
        raise RuntimeError("DFlash2 stream ended without a summary")
    prefill_us = float(summary.phase_timings_us.get("prefill", 0.0))
    decode_us = float(summary.elapsed_us) - prefill_us
    effective_width = int(summary.block_tokens or 0)
    if effective_width != int(requested_width):
        raise RuntimeError(
            f"DFlash2 requested width {requested_width} became {effective_width}"
        )
    return {
        "tokens": tuple(int(token) for token in summary.generated_token_ids),
        "generated_tokens": int(summary.generation_tokens),
        "decode_tps": float(summary.generation_tokens) / (decode_us / 1_000_000.0),
        "elapsed_s": float(summary.elapsed_us) / 1_000_000.0,
        "peak_memory_gb": float(summary.peak_memory_gb or 0.0),
        "cycles_completed": int(summary.cycles_completed),
        "accepted_from_draft": int(summary.accepted_from_draft),
        "acceptance_history": list(summary.acceptance_history),
        "requested_width": int(requested_width),
        "effective_width": effective_width,
        "fallback_ar": bool(summary.fallback_ar),
        "fallback_reason": summary.fallback_reason,
        "engine": "dflash_mlx_0_1_10",
    }


def _token_sha256(tokens):
    return hashlib.sha256(
        ",".join(str(int(token)) for token in tokens).encode()
    ).hexdigest()


def _receipt_without_tokens(arm):
    public = dict(arm)
    tokens = tuple(public.pop("tokens"))
    public["token_sha256"] = _token_sha256(tokens)
    return public


def run_dflash2_depth_sweep(
    *,
    bundle,
    prompt_ids,
    widths,
    repetitions,
    max_tokens=1024,
    oracle_tokens=None,
    arm_runner=None,
):
    widths = tuple(int(width) for width in widths)
    if oracle_tokens is None:
        oracle_tokens = run_target_oracle(
            bundle,
            prompt_ids,
            max_tokens=max_tokens,
        )
    oracle_tokens = tuple(int(token) for token in oracle_tokens)
    if len(oracle_tokens) != max_tokens:
        raise RuntimeError("target-only oracle did not produce the forced token count")

    production_runner = arm_runner is None
    runtime_context = build_fixed_dflash_runtime_context() if production_runner else None
    if production_runner:
        def arm_runner(kind, width):
            if kind == "mtp":
                return run_mtp_control(
                    bundle,
                    prompt_ids,
                    max_tokens=max_tokens,
                )
            return run_dflash2_candidate(
                bundle,
                prompt_ids,
                width,
                runtime_context,
                max_tokens=max_tokens,
            )

    brackets = []
    selection_rows = []
    warmed_widths = set()
    for repetition in range(repetitions):
        offset = repetition % len(widths)
        rotated = widths[offset:] + widths[:offset]
        for width in rotated:
            if production_runner and width not in warmed_widths:
                run_dflash2_candidate(
                    bundle,
                    prompt_ids,
                    width,
                    runtime_context,
                    max_tokens=32,
                )
                warmed_widths.add(width)
            control_before = arm_runner("mtp", MTP_DEPTH)
            candidate = arm_runner("dflash2", width)
            control_after = arm_runner("mtp", MTP_DEPTH)
            arms = (control_before, candidate, control_after)
            parity_passed = all(
                tuple(arm["tokens"]) == oracle_tokens
                and int(arm["generated_tokens"]) == max_tokens
                for arm in arms
            )
            if production_runner:
                parity_passed = parity_passed and (
                    candidate["requested_width"] == width
                    and candidate["effective_width"] == width
                    and candidate["fallback_ar"] is False
                )
            selection_rows.append(DepthBracket(
                width=width,
                candidate_decode_tps=float(candidate["decode_tps"]),
                control_before_tps=float(control_before["decode_tps"]),
                control_after_tps=float(control_after["decode_tps"]),
                parity_passed=parity_passed,
            ))
            brackets.append({
                "repetition": repetition,
                "width": width,
                "control_before": _receipt_without_tokens(control_before),
                "candidate": _receipt_without_tokens(candidate),
                "control_after": _receipt_without_tokens(control_after),
                "parity_passed": parity_passed,
            })

    selection = None
    if all(row.parity_passed for row in selection_rows):
        selection = asdict(select_stock_depth(selection_rows))
    return {
        "workload": {
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": max_tokens,
            "greedy": True,
        },
        "oracle_token_sha256": _token_sha256(oracle_tokens),
        "brackets": brackets,
        "selection": selection,
    }
```

Build `runtime_context` once with `verify_mode="dflash"`, `copyspec_mode="off"`, `quantize_kv_cache=False`, and no adaptive width policy. `run_dflash2_depth_sweep(..., max_tokens=1024)` passes the same output budget to the oracle and both measured arms. Assert every DFlash summary reports `block_tokens == requested_width`, `generation_tokens == max_tokens`, and no stop termination. Run an untimed 32-token warmup before each first measured width and one untimed AR oracle before the brackets. The production CLI fixes `max_tokens` at 1,024; only the guarded construction smoke passes 32.

For each repetition, rotate widths by the repetition index, and for every width append `C0 -> B -> C1`. Store raw tokens only as a SHA-256 plus an optional compressed token list in the local receipt; parity compares the full in-memory tuple before aggregation.

- [x] **Step 4: Run focused orchestration and event-adapter tests**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_depth_sweep.py
```

Expected: fake-arm tests pass; no real model loads occur.

- [x] **Step 5: Commit**

```bash
git add mtplx/benchmarks/runners/dflash2_depth_sweep.py tests/test_dflash2_depth_sweep.py
git commit -m "Add greedy DFlash2 depth sweep"
```

### Task 5: Replace the stale DFlash channel and add the guarded entrypoint

**Files:**
- Modify: `mtplx/benchmarks/runners/dflash2_depth_sweep.py`
- Modify: `mtplx/benchmarks/runners/competitor_baselines.py`
- Modify: `mtplx/cli.py`
- Create: `scripts/qwen38_dflash2_depth_guarded.py`
- Create: `tests/test_dflash2_cli.py`
- Create: `tests/test_qwen38_dflash2_depth_guarded.py`

**Security flag:** security

**Does NOT cover:** The command does not own launchd, acquire or steal the GPU lock, accept arbitrary child commands, enable sampling, or run outside the canonical guard attestation.

- [x] **Step 1: Write failing parser and guarded-source tests**

```python
import inspect

from mtplx import cli


def test_dflash_defaults_are_qwen38_greedy():
    parser = cli.build_parser()
    args = parser.parse_args(["dflash-mlx-baseline"])
    assert args.draft_model == "z-lab/Qwen3.8-27B-DFlash2"
    assert args.temperature == 0.0
    assert args.max_tokens == 1024


def test_depth_sweep_parser_is_closed_to_1_through_8():
    parser = cli.build_parser()
    args = parser.parse_args(["dflash2-depth-sweep", "--widths", "1,2,8", "--output", "result.json"])
    assert args.widths == "1,2,8"
    assert args.repetitions == 3
    assert args.prompt_tokens == 1024
    assert args.max_tokens == 1024


def test_guarded_script_verifies_before_importing_mlx():
    import scripts.qwen38_dflash2_depth_guarded as guarded

    source = inspect.getsource(guarded.main)
    assert source.index("issue_guard_window") < source.index("dflash2_depth_sweep")
    assert "/tmp/mtplx-gpu-exclusive.lock" in inspect.getsource(guarded)


def test_in_place_dflash_channel_no_longer_imports_obsolete_module():
    from mtplx.benchmarks.runners.competitor_baselines import (
        run_dflash_mlx_baseline,
    )

    source = inspect.getsource(run_dflash_mlx_baseline)
    assert "dflash.model_mlx" not in source
    assert "load_mtplx_dflash2_bundle" in source
```

- [x] **Step 2: Run tests and verify old defaults and missing command fail**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_cli.py tests/test_qwen38_dflash2_depth_guarded.py
```

Expected: parser assertions fail on the Qwen3.6/temperature-0.6 defaults and the guarded script is absent.

- [x] **Step 3: Update the in-place channel and add the closed sweep command**

Keep `dflash-mlx-baseline` for one-width compatibility, but route it through the new runtime bundle and current `dflash_mlx` API. Remove the source-path import shim from its measured path. Add `dflash2-depth-sweep` with these fixed defaults:

Resolve `z-lab/Qwen3.8-27B-DFlash2` only at immutable revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`; the guarded child uses
`local_files_only=True`, validates the resolved snapshot plus block/layer/quant
metadata, and records the resolved revision in its receipt.

```python
dflash2_p.add_argument("--model", default=default_model)
dflash2_p.add_argument("--draft-model", default="z-lab/Qwen3.8-27B-DFlash2")
dflash2_p.add_argument("--profile", default="turbo", choices=("turbo",))
dflash2_p.add_argument("--widths", default="1,2,3,4,5,6,7,8")
dflash2_p.add_argument("--repetitions", type=int, default=3)
dflash2_p.add_argument("--prompt-tokens", type=int, default=1024, choices=(1024,))
dflash2_p.add_argument("--max-tokens", type=int, default=1024, choices=(1024,))
dflash2_p.add_argument("--output", required=True)
```

The command resolves `inspect_model(model).compatibility.runtime_contract`,
applies `turbo` with `runtime_env_overrides_from_contract`, builds the exact
prompt after tokenizer load, installs the q4/group-64 MTP draft lm-head, and
writes the receipt atomically:

```python
def run_cli_sweep(args, *, token_count=1024):
    inspection = inspect_model(args.model).to_dict()
    runtime_contract = (inspection.get("compatibility") or {}).get(
        "runtime_contract"
    )
    runtime_overrides = runtime_env_overrides_from_contract(runtime_contract)
    apply_profile_env("turbo", runtime_env_overrides=runtime_overrides)
    bundle = load_mtplx_dflash2_bundle(args.model, args.draft_model)
    _install_draft_lm_head(
        bundle.runtime,
        bits=4,
        group_size=64,
        mode="affine",
    )
    prompt = build_exact_python_prompt_ids(
        bundle.tokenizer,
        token_count=token_count,
    )
    return run_dflash2_depth_sweep(
        bundle=bundle,
        prompt_ids=prompt.token_ids,
        widths=parse_dflash2_widths(args.widths),
        repetitions=args.repetitions,
        max_tokens=token_count,
    )
```

- [x] **Step 4: Add the one-child guarded wrapper**

`scripts/qwen38_dflash2_depth_guarded.py` performs this order inside `main()`:

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--widths", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--smoke-tokens", type=int, choices=(32,))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def write_atomic_json(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def main(argv=None):
    args = parse_args(argv)
    from deepseek_v4_guard_window import issue_guard_window

    guard_path, guard_sha256 = issue_guard_window(
        expected_lock=Path("/tmp/mtplx-gpu-exclusive.lock")
    )
    from mtplx.benchmarks.runners.dflash2_depth_sweep import run_cli_sweep

    token_count = args.smoke_tokens or 1024
    receipt = run_cli_sweep(args, token_count=token_count)
    receipt["guard_window"] = {
        "path": str(guard_path),
        "sha256": guard_sha256,
    }
    write_atomic_json(args.output, receipt)
    return 0 if receipt["selection"] is not None else 2
```

The wrapper accepts only model, draft, widths, repetitions, output, and a
wrapper-only `--smoke-tokens` whose only accepted value is 32. It does not
accept a command or plist. It consumes the one-shot guard pipe before importing
the runner and writes with `tempfile.mkstemp`, `fsync`, and `os.replace`.

- [x] **Step 5: Run parser, import-order, and focused runner tests**

```bash
.venv/bin/python -m pytest -q \
  tests/test_dflash2_cli.py \
  tests/test_qwen38_dflash2_depth_guarded.py \
  tests/test_dflash2_depth_sweep.py
```

Expected: all tests pass without model loading.

- [x] **Step 6: Commit**

```bash
git add mtplx/benchmarks/runners/dflash2_depth_sweep.py \
  mtplx/benchmarks/runners/competitor_baselines.py mtplx/cli.py \
  scripts/qwen38_dflash2_depth_guarded.py tests/test_dflash2_cli.py \
  tests/test_qwen38_dflash2_depth_guarded.py
git commit -m "Wire Qwen3.8 DFlash2 benchmark channel"
```

### Task 6: Pass repository verification before GPU work

**Files:**
- Modify only files already owned by Tasks 1-5 when verification exposes a regression.

**Security flag:** none

**Does NOT cover:** Verification fixes do not broaden model support, add optimization code, weaken parity, or change benchmark thresholds.

- [x] **Step 1: Run focused tests with DFlash2 installed**

```bash
.venv/bin/python -m pytest -q \
  tests/test_dflash2_dependency.py \
  tests/test_dflash2_contract.py \
  tests/test_dflash2_runtime.py \
  tests/test_dflash2_depth_sweep.py \
  tests/test_dflash2_cli.py \
  tests/test_qwen38_dflash2_depth_guarded.py
```

Expected: all focused tests pass.

- [x] **Step 2: Run formatting, static checks, and full suite**

```bash
.venv/bin/python -m ruff check mtplx tests scripts/qwen38_dflash2_depth_guarded.py
.venv/bin/python -m pytest -q
git diff --check upstream/main...HEAD
```

Expected: Ruff, the full suite, and whitespace checks pass. If unrelated upstream tests fail, capture the exact failure and prove it also fails at `upstream/main` before classifying it as pre-existing.

Observed on 2026-08-20: the six focused DFlash2 files pass 81 tests and the
changed files pass Ruff. Repository-wide Ruff reports the same 68 pre-existing
errors at `upstream/main`, and the full suite's sole failure,
`test_repeated_stats_polls_reuse_cached_aggregate` (`2 == 1`), reproduces on a
detached `upstream/main` worktree. `git diff --check upstream/main...HEAD`
passes.

- [x] **Step 3: Run a no-model CLI help smoke**

```bash
.venv/bin/mtplx dflash2-depth-sweep --help
```

Expected: help shows fixed greedy 1,024/1,024 defaults, widths 1-8, repetitions 3, and required output.

- [x] **Step 4: Commit only necessary verification fixes**

If Tasks 1-5 require no fixes, do not create an empty commit. Otherwise stage
only the owned files that changed:

```bash
git add pyproject.toml uv.lock \
  mtplx/benchmarks/dflash2_contract.py \
  mtplx/benchmarks/dflash2_runtime.py \
  mtplx/benchmarks/runners/dflash2_depth_sweep.py \
  mtplx/benchmarks/runners/competitor_baselines.py \
  mtplx/cli.py scripts/qwen38_dflash2_depth_guarded.py \
  tests/test_dflash2_dependency.py tests/test_dflash2_contract.py \
  tests/test_dflash2_runtime.py tests/test_dflash2_depth_sweep.py \
  tests/test_dflash2_cli.py tests/test_qwen38_dflash2_depth_guarded.py
git commit -m "Fix DFlash2 sweep verification"
```

### Task 7: Run the guarded stock depth benchmark and freeze the winner

**Files:**
- Operationally modify: `/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py`
- Create: `benchmarks/results/qwen38-dflash2-depth1-8-1024x1024-${run_stamp}.json`
- Modify: `docs/specs/2026-08-20-qwen38-dflash2-mtp-benchmark-design.md`

**Security flag:** security

**Does NOT cover:** This task does not implement or benchmark a custom kernel, change the persistent Qwen launcher, keep Qwen stopped after the window, run widths above 8, or rescue a losing/tied width with unplanned tuning.

- [ ] **Step 1: Extend the canonical guard with only the observed exact model**

Confirm read-only state first:

```bash
curl -fsS --max-time 10 http://127.0.0.1:8080/v1/models | python3 -m json.tool
plutil -p /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist
ps -axo pid,ppid,command | rg 'start-qwen38-27b-mtp|mtplx.server.openai.*Qwen3.8'
```

Require the sole model ID `mtplx-qwen38-27b-optimized-speed` and launcher `/Users/davidtai/projects/qwen36-server/scripts/start-qwen38-27b-mtp.sh`. Add only:

```python
ALLOWED_QWEN_MODEL_SETS = (
    ("mtplx-qwen36-27b-optimized-speed",),
    ("mtplx-qwen36-27b-optimized-quality",),
    ("mtplx-qwen36-27b-optimized-speed-v2",),
    ("mtplx-qwen36-35b-a3b-optimized-speed",),
    ("mtplx-laguna-s21-oq4e",),
    ("mtplx-qwen38-27b-optimized-speed",),
)

PROCESS_PATTERNS = {
    ("mtplx-laguna-s21-oq4e",): "mtplx.server.openai.*Laguna-S-2.1",
    ("mtplx-qwen38-27b-optimized-speed",): "mtplx.server.openai.*Qwen3.8",
}
```

to the existing collections in `/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py`. Do not edit the underlying generic guard constants or add a wildcard model tuple.

- [ ] **Step 2: Download and verify the draft before taking the GPU lock**

```bash
.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("z-lab/Qwen3.8-27B-DFlash2", revision="50307d4c4cde6860d4eee73e2547cd786fe8e8a4"))'
.venv/bin/python -c 'from dflash_mlx.runtime.loading import load_draft_bundle; model, meta = load_draft_bundle("z-lab/Qwen3.8-27B-DFlash2", draft_quant="w4:gs64", lazy=True); print(model.block_size, model.target_layer_ids, meta)'
```

Expected: block size 8, target layers `[5, 19, 33, 47, 61]`, and no target model load.

- [ ] **Step 3: Capture a quiet-machine preflight immediately before the guard**

```bash
.venv/bin/mtplx bench preflight \
  --cpu-threshold 25 \
  --min-free-gib 25 \
  --output /tmp/qwen38-dflash2-preflight.json
.venv/bin/python -c 'import json; value=json.load(open("/tmp/qwen38-dflash2-preflight.json")); assert value["clean"], value'
```

Expected: clean preflight, AC power, no active benchmark, no heavy background process, and no thermal warning.

- [ ] **Step 4: Run a guarded 32/32 construction and parity smoke**

```bash
.venv/bin/python /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --timeout-seconds 300 \
  --lock-timeout-seconds 3600 \
  --child-timeout-seconds 1800 \
  -- \
  .venv/bin/python scripts/qwen38_dflash2_depth_guarded.py \
  --model /Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed \
  --draft-model z-lab/Qwen3.8-27B-DFlash2 \
  --widths 1,8 \
  --repetitions 1 \
  --smoke-tokens 32 \
  --output /tmp/qwen38-dflash2-smoke.json
```

Expected: construction succeeds without a second target load; MTP and both DFlash widths exactly match the greedy oracle for 32 tokens; the guard restores `com.tea.qwen` before returning. The `--smoke-tokens` option exists only on the guarded wrapper and is rejected unless it equals 32; the production CLI remains fixed at 1,024.

- [ ] **Step 5: Verify exact service restoration and lock release after the smoke**

```bash
curl -fsS --max-time 10 http://127.0.0.1:8080/v1/models | python3 -m json.tool
curl -fsS --max-time 60 -H 'Content-Type: application/json' \
  -d '{"model":"mtplx-qwen38-27b-optimized-speed","messages":[{"role":"user","content":"Say READY"}],"max_tokens":8,"temperature":0}' \
  http://127.0.0.1:8080/v1/chat/completions | python3 -m json.tool
python3 -c 'import fcntl; f=open("/tmp/mtplx-gpu-exclusive.lock","rb"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print("lock-free")'
```

Expected: exact model ID, READY response with `finish_reason=stop`, and `lock-free`.

- [ ] **Step 6: Run the full guarded width 1-8 campaign**

Choose one UTC timestamp before starting and use it in the output filename:

```bash
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --timeout-seconds 300 \
  --lock-timeout-seconds 3600 \
  --child-timeout-seconds 43200 \
  -- \
  .venv/bin/python scripts/qwen38_dflash2_depth_guarded.py \
  --model /Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed \
  --draft-model z-lab/Qwen3.8-27B-DFlash2 \
  --widths 1,2,3,4,5,6,7,8 \
  --repetitions 3 \
  --output "benchmarks/results/qwen38-dflash2-depth1-8-1024x1024-${run_stamp}.json"
```

Expected: 24 accepted candidate brackets, each containing `C0/B/C1`, exactly 1,024 prompt IDs, exactly 1,024 generated IDs, requested/effective width equality, and exact oracle token parity. If any individual arm exceeds two minutes, two complete repetitions are acceptable only when both controls are within 5 percent; record the reduced repetition count and reason in the receipt.

- [ ] **Step 7: If needed, run direct tie-break brackets without changing code**

When `selection.needs_tiebreak` is true, set
`tiebreak_stamp=$(date -u +%Y%m%dT%H%M%SZ)`, pass only the reported tied widths,
and run three more alternating brackets with output
`benchmarks/results/qwen38-dflash2-depth-tiebreak-${tiebreak_stamp}.json`. Do not
change block construction, cache settings, warmup, or width order outside the
runner's deterministic rotation.

- [ ] **Step 8: Freeze the Phase A result and stop before optimization**

Append the exact tested commits, dependency version, checkpoint revision, target artifact hashes, profile environment hash, prompt/token hashes, raw bracket table, drift, selected width or tie band, and service postflight to the spec. Then:

```bash
git add benchmarks/results/qwen38-dflash2-depth1-8-1024x1024-*.json \
  docs/specs/2026-08-20-qwen38-dflash2-mtp-benchmark-design.md
git commit -m "Record Qwen3.8 DFlash2 stock depth sweep"
git push mtplx1 perf/qwen38-dflash2
```

Expected: PR #304 contains immutable Phase A evidence and names the fastest correct stock DFlash2 width or measured tie band. Stop here. Write the Phase B profile/custom-optimization plan from this evidence before changing any kernel or DFlash2 arithmetic.
