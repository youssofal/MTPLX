from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from mtplx.generation import generate_ar, generate_mtpk
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig


def _fixed_prompt_ids(tokenizer, length: int = 1024) -> list[int]:
    records = "\n".join(
        f"record {index:04d}: account={index % 17}; region=r{index % 9}; value={index * 13}"
        for index in range(512)
    )
    token_ids = list(tokenizer.encode(records, add_special_tokens=True))
    if len(token_ids) < length:
        raise AssertionError(f"synthetic prompt produced only {len(token_ids)} tokens")
    return [int(token) for token in token_ids[:length]]


def _token_digest(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_real_qwen38_dflash2_greedy_exactness() -> None:
    bundle_value = os.environ.get("MTPLX_DFLASH2_REAL_BUNDLE")
    if not bundle_value:
        pytest.skip("set MTPLX_DFLASH2_REAL_BUNDLE to run the real-model gate")

    bundle = Path(bundle_value).expanduser().resolve()
    max_tokens = int(os.environ.get("MTPLX_DFLASH2_EXACTNESS_TOKENS", "64"))
    evidence_path = os.environ.get("MTPLX_DFLASH2_EVIDENCE")

    runtime = load(bundle, mtp=True)
    prompt_ids = _fixed_prompt_ids(runtime.tokenizer)
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)

    ar_started = time.perf_counter()
    ar = generate_ar(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        seed=0,
        stop_token_ids=set(),
    )
    ar_wall_s = time.perf_counter() - ar_started
    dflash2_started = time.perf_counter()
    dflash2 = generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        speculative_depth=5,
        seed=0,
        stop_token_ids=set(),
    )
    dflash2_wall_s = time.perf_counter() - dflash2_started

    first_mismatch = next(
        (
            index
            for index, (ar_token, draft_token) in enumerate(zip(ar.tokens, dflash2.tokens))
            if ar_token != draft_token
        ),
        None,
    )
    if first_mismatch is None and len(ar.tokens) != len(dflash2.tokens):
        first_mismatch = min(len(ar.tokens), len(dflash2.tokens))

    manifest = json.loads((bundle / "mtplx_dflash2.json").read_text(encoding="utf-8"))
    evidence = {
        "bundle": str(bundle),
        "target_revision": manifest["target"]["revision"],
        "draft_revision": manifest["draft"]["revision"],
        "algorithm_revision": manifest["algorithm"]["revision"],
        "draft_precision": manifest["draft"]["precision"],
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": _token_digest(prompt_ids),
        "requested_tokens": max_tokens,
        "ar_tokens": ar.tokens,
        "dflash2_tokens": dflash2.tokens,
        "ar_sha256": _token_digest(ar.tokens),
        "dflash2_sha256": _token_digest(dflash2.tokens),
        "first_mismatch": first_mismatch,
        "exact": ar.tokens == dflash2.tokens,
        "ar_wall_s": ar_wall_s,
        "dflash2_wall_s": dflash2_wall_s,
        "ar_wall_tok_s": len(ar.tokens) / ar_wall_s,
        "dflash2_wall_tok_s": len(dflash2.tokens) / dflash2_wall_s,
        "ar_stats": ar.stats.to_dict(),
        "dflash2_stats": dflash2.stats.to_dict(),
    }
    if evidence_path:
        output = Path(evidence_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    assert len(ar.tokens) == max_tokens
    assert len(dflash2.tokens) == max_tokens
    assert ar.tokens == dflash2.tokens, (
        f"DFlash2 diverged from target-only AR at output token {first_mismatch}; "
        f"AR={evidence['ar_sha256']} DFlash2={evidence['dflash2_sha256']}"
    )


def test_real_qwen38_native_mtp_baseline() -> None:
    model_value = os.environ.get("MTPLX_DFLASH2_NATIVE_MTP_MODEL")
    if not model_value:
        pytest.skip("set MTPLX_DFLASH2_NATIVE_MTP_MODEL to run the native-MTP baseline")

    model = Path(model_value).expanduser().resolve()
    max_tokens = int(os.environ.get("MTPLX_DFLASH2_EXACTNESS_TOKENS", "64"))
    evidence_path = os.environ.get("MTPLX_DFLASH2_NATIVE_MTP_EVIDENCE")

    runtime = load(model, mtp=True)
    prompt_ids = _fixed_prompt_ids(runtime.tokenizer)
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    ar_started = time.perf_counter()
    ar = generate_ar(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        seed=0,
        stop_token_ids=set(),
    )
    ar_wall_s = time.perf_counter() - ar_started
    native_mtp_started = time.perf_counter()
    native_mtp = generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        speculative_depth=3,
        seed=0,
        stop_token_ids=set(),
    )
    native_mtp_wall_s = time.perf_counter() - native_mtp_started

    first_mismatch = next(
        (
            index
            for index, (ar_token, draft_token) in enumerate(zip(ar.tokens, native_mtp.tokens))
            if ar_token != draft_token
        ),
        None,
    )
    if first_mismatch is None and len(ar.tokens) != len(native_mtp.tokens):
        first_mismatch = min(len(ar.tokens), len(native_mtp.tokens))

    evidence = {
        "model": str(model),
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": _token_digest(prompt_ids),
        "requested_tokens": max_tokens,
        "ar_tokens": ar.tokens,
        "native_mtp_tokens": native_mtp.tokens,
        "ar_sha256": _token_digest(ar.tokens),
        "native_mtp_sha256": _token_digest(native_mtp.tokens),
        "first_mismatch": first_mismatch,
        "exact": ar.tokens == native_mtp.tokens,
        "ar_wall_s": ar_wall_s,
        "native_mtp_wall_s": native_mtp_wall_s,
        "ar_wall_tok_s": len(ar.tokens) / ar_wall_s,
        "native_mtp_wall_tok_s": len(native_mtp.tokens) / native_mtp_wall_s,
        "ar_stats": ar.stats.to_dict(),
        "native_mtp_stats": native_mtp.stats.to_dict(),
    }
    if evidence_path:
        output = Path(evidence_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    assert len(ar.tokens) == max_tokens
    assert len(native_mtp.tokens) == max_tokens
    assert ar.tokens == native_mtp.tokens, (
        f"Native MTP diverged from target-only AR at output token {first_mismatch}; "
        f"AR={evidence['ar_sha256']} MTP={evidence['native_mtp_sha256']}"
    )
