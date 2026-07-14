"""NVFP4 vs affine comparison-matrix benchmark runner."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

COMPARISON_LANES: tuple[tuple[str, str, str, int], ...] = (
    ("affine_ar", "affine", "ar", 0),
    ("affine_mtp_d1", "affine", "mtp", 1),
    ("affine_mtp_d2", "affine", "mtp", 2),
    ("affine_mtp_d3", "affine", "mtp", 3),
    ("nvfp4_ar", "nvfp4", "ar", 0),
    ("nvfp4_mtp_d1", "nvfp4", "mtp", 1),
    ("nvfp4_mtp_d2", "nvfp4", "mtp", 2),
    ("nvfp4_mtp_d3", "nvfp4", "mtp", 3),
)

SAMPLER_PROFILES: dict[str, dict[str, Any]] = {
    "blog-parity": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "min_p": 0.0,
        "enable_thinking": False,
    },
    "mtplx-optimized": {
        "temperature": 0.6,
        "draft_temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": False,
    },
}

DEFAULT_PINS_PATH = Path("benchmarks/inventory/pins.json")
REQUIRED_LANE_FIELDS = (
    "lane",
    "quantization_family",
    "generation_mode",
    "mtp_depth",
    "prompt_tokens",
    "generated_tokens",
    "decode_tok_s",
    "ttft_s",
    "acceptance_by_depth",
)


@dataclass(frozen=True)
class ComparisonLaneSpec:
    lane: str
    quantization_family: str
    generation_mode: str
    mtp_depth: int


@dataclass
class ComparisonLaneResult:
    lane: str
    quantization_family: str
    generation_mode: str
    mtp_depth: int
    model_path: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    end_to_end_tok_s: float | None = None
    ttft_s: float | None = None
    acceptance_by_depth: list[float | None] | None = None
    correction_count: int | None = None
    peak_memory_bytes: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lane_specs() -> list[ComparisonLaneSpec]:
    return [
        ComparisonLaneSpec(
            lane=lane, quantization_family=quant, generation_mode=mode, mtp_depth=depth
        )
        for lane, quant, mode, depth in COMPARISON_LANES
    ]


def load_pins(path: Path | str | None = None) -> dict[str, Any]:
    pins_path = Path(path or DEFAULT_PINS_PATH)
    payload = json.loads(pins_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {pins_path}")
    return payload


def sampler_profile(name: str) -> dict[str, Any]:
    key = str(name or "mtplx-optimized")
    if key not in SAMPLER_PROFILES:
        raise ValueError(f"unknown sampler profile: {key}")
    return dict(SAMPLER_PROFILES[key])


def validate_comparison_matrix(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if payload.get("action") != "bench comparison matrix":
        problems.append("missing or invalid action")
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        problems.append("lanes must be a non-empty list")
        return problems
    seen = set()
    for index, row in enumerate(lanes):
        if not isinstance(row, dict):
            problems.append(f"lane[{index}] must be an object")
            continue
        lane_name = row.get("lane")
        if not isinstance(lane_name, str) or not lane_name:
            problems.append(f"lane[{index}] missing lane name")
            continue
        if lane_name in seen:
            problems.append(f"duplicate lane: {lane_name}")
        seen.add(lane_name)
        for field in REQUIRED_LANE_FIELDS:
            if field not in row:
                problems.append(f"lane[{lane_name}] missing {field}")
    expected = {spec.lane for spec in lane_specs()}
    missing = sorted(expected - seen)
    if missing:
        problems.append(f"missing lanes: {', '.join(missing)}")
    return problems


def _model_for_lane(
    quantization_family: str,
    *,
    affine_model: str | None,
    nvfp4_model: str | None,
    fallback_model: str | None,
) -> str:
    if quantization_family == "affine":
        return str(affine_model or fallback_model or "")
    if quantization_family == "nvfp4":
        return str(nvfp4_model or fallback_model or "")
    raise ValueError(f"unknown quantization family: {quantization_family}")


def _manifest_lane(
    spec: ComparisonLaneSpec,
    *,
    model_path: str,
    prompt_tokens: int,
    max_tokens: int,
) -> ComparisonLaneResult:
    return ComparisonLaneResult(
        lane=spec.lane,
        quantization_family=spec.quantization_family,
        generation_mode=spec.generation_mode,
        mtp_depth=spec.mtp_depth,
        model_path=model_path,
        prompt_tokens=int(prompt_tokens),
        generated_tokens=0,
        decode_tok_s=None,
        ttft_s=None,
        acceptance_by_depth=[None] * spec.mtp_depth if spec.mtp_depth else [],
        error="manifest_only_no_inference",
    )


def _run_live_lane(
    spec: ComparisonLaneSpec,
    *,
    model_path: str,
    prompt_tokens: list[int],
    max_tokens: int,
    sampler: dict[str, Any],
    seed: int,
) -> ComparisonLaneResult:
    from mtplx.benchmarks.schema import PromptCase, encode_prompt_case
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    if not model_path:
        return ComparisonLaneResult(
            lane=spec.lane,
            quantization_family=spec.quantization_family,
            generation_mode=spec.generation_mode,
            mtp_depth=spec.mtp_depth,
            model_path="",
            error="missing_model_path",
        )

    runtime = load(model_path)
    tokenizer = runtime.tokenizer
    prompt = " ".join(
        ["benchmark"] * max(1, int(prompt_tokens[0] if prompt_tokens else 512))
    )
    encoded = encode_prompt_case(
        tokenizer, PromptCase(id=spec.lane, category="matrix", prompt=prompt)
    )
    sampler_cfg = SamplerConfig(
        temperature=float(sampler.get("temperature", 0.6)),
        top_p=float(sampler.get("top_p", 0.95)),
        top_k=int(sampler.get("top_k", 20)),
        presence_penalty=float(sampler.get("presence_penalty", 0.0)),
        min_p=float(sampler.get("min_p", 0.0)),
    )
    draft_sampler = None
    if spec.generation_mode == "mtp":
        draft_sampler = SamplerConfig(
            temperature=float(
                sampler.get("draft_temperature", sampler_cfg.temperature)
            ),
            top_p=float(sampler.get("top_p", 0.95)),
            top_k=int(sampler.get("top_k", 20)),
        )

    started = time.perf_counter()
    ttft_s: float | None = None
    generated = 0
    acceptance: list[float | None] = []
    correction_count = 0
    try:
        if spec.generation_mode == "ar" or spec.mtp_depth == 0:
            result = generate_ar(
                runtime,
                encoded,
                max_tokens=max_tokens,
                sampler=sampler_cfg,
                seed=seed,
            )
            generated = int(result.get("generated_tokens") or 0)
            ttft_s = float(result.get("ttft_s") or 0.0)
        else:
            result = generate_mtpk(
                runtime,
                encoded,
                depth=spec.mtp_depth,
                max_tokens=max_tokens,
                sampler=sampler_cfg,
                draft_sampler=draft_sampler,
                seed=seed,
            )
            generated = int(result.get("generated_tokens") or 0)
            ttft_s = float(result.get("ttft_s") or 0.0)
            accepted = result.get("accepted_tokens") or []
            drafted = result.get("drafted_tokens") or []
            acceptance = [
                (float(a) / float(d) if d else None)
                for a, d in zip(accepted, drafted, strict=False)
            ]
            correction_count = int(result.get("correction_count") or 0)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return ComparisonLaneResult(
            lane=spec.lane,
            quantization_family=spec.quantization_family,
            generation_mode=spec.generation_mode,
            mtp_depth=spec.mtp_depth,
            model_path=model_path,
            prompt_tokens=len(encoded),
            generated_tokens=generated,
            ttft_s=ttft_s,
            decode_tok_s=(generated / elapsed) if elapsed > 0 and generated else None,
            acceptance_by_depth=acceptance,
            correction_count=correction_count,
            error=str(exc),
        )

    elapsed = time.perf_counter() - started
    decode_elapsed = max(elapsed - (ttft_s or 0.0), 1e-9)
    return ComparisonLaneResult(
        lane=spec.lane,
        quantization_family=spec.quantization_family,
        generation_mode=spec.generation_mode,
        mtp_depth=spec.mtp_depth,
        model_path=model_path,
        prompt_tokens=len(encoded),
        generated_tokens=generated,
        ttft_s=ttft_s,
        decode_tok_s=(max(generated - 1, 0) / decode_elapsed) if generated else 0.0,
        end_to_end_tok_s=(generated / elapsed) if elapsed > 0 else None,
        acceptance_by_depth=acceptance,
        correction_count=correction_count,
    )


def run_comparison_matrix(
    *,
    affine_model: str | None = None,
    nvfp4_model: str | None = None,
    model: str | None = None,
    sampler_profile_name: str = "mtplx-optimized",
    prompt_tokens: int = 512,
    max_tokens: int = 512,
    warmups: int = 2,
    measured_runs: int = 10,
    seed: int = 42,
    pins_path: Path | str | None = None,
    dry_run: bool = False,
    manifest_only: bool = False,
) -> dict[str, Any]:
    """Run or describe the affine/NVFP4 AR/MTP comparison matrix."""

    pins = load_pins(pins_path)
    sampler = sampler_profile(sampler_profile_name)
    specs = lane_specs()
    lane_rows: list[dict[str, Any]] = []

    for spec in specs:
        model_path = _model_for_lane(
            spec.quantization_family,
            affine_model=affine_model,
            nvfp4_model=nvfp4_model,
            fallback_model=model,
        )
        if dry_run or manifest_only:
            lane_rows.append(
                _manifest_lane(
                    spec,
                    model_path=model_path,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                ).to_dict()
            )
            continue

        samples: list[ComparisonLaneResult] = []
        for run_index in range(warmups + measured_runs):
            row = _run_live_lane(
                spec,
                model_path=model_path,
                prompt_tokens=[prompt_tokens],
                max_tokens=max_tokens,
                sampler=sampler,
                seed=seed + run_index,
            )
            if run_index >= warmups:
                samples.append(row)

        if not samples:
            lane_rows.append(
                _manifest_lane(
                    spec,
                    model_path=model_path,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                ).to_dict()
            )
            continue

        decode_rates = [
            item.decode_tok_s for item in samples if item.decode_tok_s is not None
        ]
        ttft_values = [item.ttft_s for item in samples if item.ttft_s is not None]
        aggregate = samples[-1].to_dict()
        aggregate["measured_runs"] = len(samples)
        aggregate["warmups"] = warmups
        aggregate["decode_tok_s_median"] = (
            float(statistics.median(decode_rates)) if decode_rates else None
        )
        aggregate["ttft_s_median"] = (
            float(statistics.median(ttft_values)) if ttft_values else None
        )
        lane_rows.append(aggregate)

    payload: dict[str, Any] = {
        "action": "bench comparison matrix",
        "sampler_profile": sampler_profile_name,
        "sampler": sampler,
        "pins": pins,
        "protocol": {
            "prompt_tokens": int(prompt_tokens),
            "max_tokens": int(max_tokens),
            "warmups": int(warmups),
            "measured_runs": int(measured_runs),
            "seed": int(seed),
            "dry_run": bool(dry_run),
            "manifest_only": bool(manifest_only),
        },
        "lanes": lane_rows,
    }
    problems = validate_comparison_matrix(payload)
    payload["valid"] = not problems
    payload["problems"] = problems
    return payload


def write_comparison_matrix(path: Path | str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
