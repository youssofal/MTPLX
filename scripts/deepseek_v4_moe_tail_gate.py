"""One-load, guarded-GPU parity/compile gate for the DeepSeek-V4 MoE tail.

This is deliberately an operator safety receipt, not a throughput benchmark.
It loads the immutable merged 2bit-DQ-MTP checkpoint once through
``mtplx.runtime.load(..., mtp=True)``, proves the bound 43+1 topology and actual
routed storage, then captures the stock score-layer MoE tail after a real
328-token coding prefill.  The exact stock tail is compared with the fused BF16
tail over authentic [4, 6, 4096] K3-verify-shaped tensors; the same installed
route must keep an M1 rejection repair stock.  Every diagnostic sample is
explicitly evaluated and synchronized; it never queues hundreds of independent
outputs and mistakes host enqueue time for GPU work.  The subsequent full
328-token / 256-generated C0->candidate->C1 run is the only performance decision.

Run only through ``bench/laguna/run_guarded.py``.  The gate accepts exactly the
official MLX 0.31.2 serving runtime and the immutable official 328-token prompt
identity below.  A profiler/dev MLX build or altered/copied prompt is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


_REQUIRED_MLX_VERSION = "0.31.2"
_REQUIRED_MLX_CORE_SHA256 = (
    "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6"
)
_REQUIRED_MLX_LIB_SHA256 = (
    "2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd"
)
_REQUIRED_PROMPT_PATH = Path(
    "/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4/"
    "smoke-2bitdq-20260731-prompt2.txt"
)
_REQUIRED_PROMPT_SHA256 = (
    "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33"
)
_REQUIRED_PROMPT_TOKENS = 328
_REQUIRED_MODEL_CONFIG_SHA256 = (
    "c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f"
)
_REQUIRED_MODEL_INDEX_SHA256 = (
    "c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8"
)
_REQUIRED_MODEL_PATH = Path(
    "/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp"
)
_BODY_LAYERS = 43
_MTP_BLOCKS = 1
_TOPK = 6
_HIDDEN_SIZE = 4096
_ROUTED_EXPERTS = 256
_ROUTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _default_model() -> str | None:
    return str(_REQUIRED_MODEL_PATH) if _REQUIRED_MODEL_PATH.is_dir() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_model_contract(config: dict, index: dict) -> dict:
    """Prove the merged 43-body + one-MTP Q2-DQ topology before loading."""
    required_fields = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": _BODY_LAYERS,
        "num_nextn_predict_layers": _MTP_BLOCKS,
        "n_routed_experts": _ROUTED_EXPERTS,
        "num_experts_per_tok": _TOPK,
        "hidden_size": _HIDDEN_SIZE,
        "moe_intermediate_size": 2048,
    }
    mismatches = {
        key: {"required": required, "actual": config.get(key)}
        for key, required in required_fields.items()
        if config.get(key) != required
    }
    ratios = config.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) != _BODY_LAYERS + _MTP_BLOCKS:
        mismatches["compress_ratios"] = {
            "required_length": _BODY_LAYERS + _MTP_BLOCKS,
            "actual_length": len(ratios) if isinstance(ratios, list) else None,
        }
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("model index has no weight_map")
    mtp_keys = sorted(key for key in weight_map if key.startswith("mtp."))
    mtp_indices = {
        key.split(".", 2)[1] for key in mtp_keys if len(key.split(".", 2)) == 3
    }
    if mismatches or not mtp_keys or mtp_indices != {"0"}:
        raise ValueError(
            "requires DeepSeek-V4 43+1 MTP topology; "
            f"field_mismatches={mismatches} mtp_indices={sorted(mtp_indices)} "
            f"mtp_tensors={len(mtp_keys)}"
        )

    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("model config has no quantization map")
    routed_tensors = 0
    group_sizes: set[int] = set()
    for layer in range(_BODY_LAYERS):
        for projection in _ROUTED_PROJECTIONS:
            stem = f"model.layers.{layer}.ffn.switch_mlp.{projection}"
            spec = quantization.get(stem)
            if not isinstance(spec, dict) or (
                int(spec.get("bits", -1)) != 2
                or str(spec.get("mode", "")).lower() != "affine"
                or int(spec.get("group_size", -1)) not in {32, 64}
            ):
                raise ValueError(
                    "requires Q2 affine routed expert storage for every body "
                    f"projection; {stem}={spec!r}"
                )
            group_sizes.add(int(spec["group_size"]))
            required_tensors = {f"{stem}.{part}" for part in ("weight", "scales", "biases")}
            missing = sorted(required_tensors.difference(weight_map))
            if missing:
                raise ValueError(
                    f"Q2 DQ manifest is missing routed storage for {stem}: {missing}"
                )
            routed_tensors += len(required_tensors)
    return {
        **required_fields,
        "compress_ratios": len(ratios),
        "body_q2_routed_projections": _BODY_LAYERS * len(_ROUTED_PROJECTIONS),
        "body_q2_manifest_tensors": routed_tensors,
        "body_q2_group_sizes": sorted(group_sizes),
        "mtp_manifest_tensors": len(mtp_keys),
        "mtp_indices": sorted(mtp_indices),
        "index_weight_count": len(weight_map),
        "index_total_size": (index.get("metadata") or {}).get("total_size"),
    }


def _validate_model_artifact(model_path: Path) -> tuple[dict, dict]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config_sha256 = _sha256(config_path)
    index_sha256 = _sha256(index_path)
    if config_sha256 != _REQUIRED_MODEL_CONFIG_SHA256:
        raise ValueError(
            "DeepSeek-V4 2bit-DQ-MTP config identity mismatch: "
            f"expected {_REQUIRED_MODEL_CONFIG_SHA256}, got {config_sha256}"
        )
    if index_sha256 != _REQUIRED_MODEL_INDEX_SHA256:
        raise ValueError(
            "DeepSeek-V4 2bit-DQ-MTP index identity mismatch: "
            f"expected {_REQUIRED_MODEL_INDEX_SHA256}, got {index_sha256}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    contract = _validate_model_contract(config, index)
    return config, {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "index_path": str(index_path),
        "index_sha256": index_sha256,
        **contract,
    }


def _validate_projection_storage(module, spec: dict, stem: str, *, biases: bool) -> None:
    actual = {
        "bits": getattr(module, "bits", None),
        "group_size": getattr(module, "group_size", None),
        "mode": getattr(module, "mode", None),
    }
    expected = {
        "bits": int(spec["bits"]),
        "group_size": int(spec["group_size"]),
        "mode": str(spec["mode"]),
    }
    if actual != expected:
        raise ValueError(f"loaded quantization mismatch for {stem}: {actual} != {expected}")
    weight_dtype = str(getattr(getattr(module, "weight", None), "dtype", "")).lower()
    if weight_dtype.rsplit(".", 1)[-1] != "uint32":
        raise ValueError(f"loaded quantized weight for {stem} is not uint32 storage")
    if getattr(module, "scales", None) is None:
        raise ValueError(f"loaded quantized weight for {stem} has no scales")
    has_biases = getattr(module, "biases", None) is not None
    if has_biases != biases:
        raise ValueError(
            f"loaded quantized biases contract mismatch for {stem}: "
            f"required={biases} actual={has_biases}"
        )


def _validate_loaded_runtime(runtime, config: dict) -> dict:
    """Construction-time proof of the modules the K3 runtime will execute."""
    if not bool(getattr(runtime, "mtp_enabled", False)):
        raise ValueError("MTP was not bound by mtplx.runtime.load(..., mtp=True)")
    model = runtime.model
    actual_model_type = getattr(model, "model_type", None)
    if str(actual_model_type or "").lower() != "deepseek_v4":
        raise ValueError(
            f"loaded model_type is not deepseek_v4: {actual_model_type!r}"
        )
    layers = list(getattr(model, "layers", []))
    mtp_blocks = list(getattr(model, "mtp_blocks", []))
    if len(layers) != _BODY_LAYERS or len(mtp_blocks) != _MTP_BLOCKS:
        raise ValueError(
            "loaded runtime is not the 43+1 MTP topology: "
            f"body={len(layers)} mtp={len(mtp_blocks)}"
        )
    quantization = config["quantization"]
    body_count = 0
    for layer_id, layer in enumerate(layers):
        switch = layer.ffn.switch_mlp
        for projection in _ROUTED_PROJECTIONS:
            stem = f"model.layers.{layer_id}.ffn.switch_mlp.{projection}"
            _validate_projection_storage(
                getattr(switch, projection), quantization[stem], stem, biases=True
            )
            body_count += 1
    mtp_count = 0
    mtp_switch = mtp_blocks[0].ffn.switch_mlp
    for projection in _ROUTED_PROJECTIONS:
        stem = f"mtp.0.ffn.switch_mlp.{projection}"
        spec = quantization.get(stem)
        if not isinstance(spec, dict):
            raise ValueError(f"MTP routed quantization missing for {stem}")
        _validate_projection_storage(
            getattr(mtp_switch, projection), spec, stem, biases=False
        )
        if (
            int(spec.get("bits", -1)) != 4
            or int(spec.get("group_size", -1)) != 32
            or str(spec.get("mode", "")).lower() != "mxfp4"
        ):
            raise ValueError(f"MTP routed storage is not source-exact MXFP4: {stem}={spec}")
        mtp_count += 1
    return {
        "runtime_mtp_enabled": True,
        "body_layers_loaded": len(layers),
        "mtp_blocks_bound": len(mtp_blocks),
        "body_q2_routed_projections": body_count,
        "body_q2_weight_dtype": "uint32",
        "mtp_mxfp4_routed_projections": mtp_count,
        "mtp_routed_weight_dtype": "uint32",
    }


def _median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def _timed(fn, *, cycles: int) -> list[float]:
    for _ in range(2):
        out = fn()
        mx.eval(out)
        mx.synchronize()
    values = []
    for _ in range(cycles):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        values.append(time.perf_counter() - t0)
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--prompt-file", required=True,
                    help="the exact official coding prompt used by the TPS bracket")
    ap.add_argument("--layer", type=int, default=3,
                    help="score-routed body layer captured (hash layers stay stock)")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--out", required=True, help="JSON receipt path")
    args = ap.parse_args()
    from deepseek_v4_guard_window import load_verified_guard_window

    guard_window = load_verified_guard_window()
    global mx
    import mlx.core as mx

    if not args.model:
        raise SystemExit("no 2-bit DeepSeek-V4 model found; pass --model")
    if args.cycles < 3:
        raise SystemExit("--cycles must be >= 3 for a useful diagnostic median")
    if not mx.metal.is_available():
        raise SystemExit("this gate requires Metal")
    if mx.__version__ != _REQUIRED_MLX_VERSION:
        raise SystemExit(
            f"requires official MLX {_REQUIRED_MLX_VERSION}, got {mx.__version__} "
            f"from {getattr(mx, '__file__', None)}"
        )
    mlx_core_path = Path(mx.__file__).resolve()
    mlx_lib_path = mlx_core_path.parent / "lib" / "libmlx.dylib"
    mlx_core_sha256 = hashlib.sha256(mlx_core_path.read_bytes()).hexdigest()
    mlx_lib_sha256 = hashlib.sha256(mlx_lib_path.read_bytes()).hexdigest()
    if (
        mlx_core_sha256 != _REQUIRED_MLX_CORE_SHA256
        or mlx_lib_sha256 != _REQUIRED_MLX_LIB_SHA256
    ):
        raise SystemExit(
            "MLX 0.31.2 binary identity mismatch: "
            f"core={mlx_core_sha256} lib={mlx_lib_sha256}"
        )
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from mtplx.attention_context import attention_phase
    from mtplx.models import deepseek_v4 as D

    if D._MOE_TAIL:
        raise SystemExit(
            "parity gate must load the stock arm with MTPLX_DSV4_MOE_TAIL=0; "
            "the candidate is installed explicitly after the authentic capture"
        )
    from mtplx import runtime as mtplx_runtime

    model_path = Path(args.model).expanduser().resolve()
    try:
        config, model_identity = _validate_model_artifact(model_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"model identity gate failed: {exc}") from exc
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    if prompt_path != _REQUIRED_PROMPT_PATH:
        raise SystemExit(
            f"requires official prompt path {_REQUIRED_PROMPT_PATH}, got {prompt_path}"
        )
    prompt_bytes = prompt_path.read_bytes()
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    if prompt_sha256 != _REQUIRED_PROMPT_SHA256:
        raise SystemExit(
            f"official prompt SHA mismatch: expected {_REQUIRED_PROMPT_SHA256}, "
            f"got {prompt_sha256}"
        )
    prompt = prompt_bytes.decode("utf-8")
    mx.set_default_device(mx.gpu)
    t0 = time.perf_counter()
    runtime = mtplx_runtime.load(model_path, mtp=True)
    try:
        loaded_identity = _validate_loaded_runtime(runtime, config)
    except ValueError as exc:
        raise SystemExit(f"loaded runtime identity gate failed: {exc}") from exc
    model = runtime.model
    tokenizer = runtime.tokenizer
    mx.eval(runtime.model.parameters())
    load_seconds = time.perf_counter() - t0
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) != _REQUIRED_PROMPT_TOKENS:
        raise SystemExit(
            f"prompt has {len(prompt_ids)} tokens, expected {_REQUIRED_PROMPT_TOKENS}; "
            "pass the exact official 328-token coding prompt"
        )
    if not (0 <= args.layer < len(model.layers)):
        raise SystemExit(f"--layer={args.layer} outside body [0,{len(model.layers)})")
    if args.layer < int(model.args.num_hash_layers):
        raise SystemExit("capture a score layer; hash layers are intentionally stock")

    # A real 328-token prefill establishes authentic cache/routing state.  Then
    # four deterministic next ids exercise the K3 verifier's M=4 body shape.
    cache = runtime.make_cache()
    with attention_phase("prefill"):
        logits, _hidden = runtime.forward_ar(
            mx.array(prompt_ids)[None],
            cache=cache,
            return_hidden=True,
            logits_keep=1,
        )
    next_id = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(next_id)
    verify_ids = mx.broadcast_to(next_id[:, None], (1, 4))

    target = model.layers[args.layer].ffn
    captured: dict[str, mx.array] = {}
    original = target._tail_combine

    def capture(routed, weights, shared):
        captured["routed"] = routed
        captured["weights"] = weights
        captured["shared"] = shared
        return original(routed, weights, shared)

    target._tail_combine = capture
    try:
        with attention_phase("decode_verify"):
            logits, _hidden = runtime.forward_ar(
                verify_ids, cache=cache, return_hidden=True
            )
        mx.eval(logits)
    finally:
        target._tail_combine = original
    if set(captured) != {"routed", "weights", "shared"}:
        raise SystemExit("score-layer tail capture did not engage")
    routed, weights, shared = (captured[k] for k in ("routed", "weights", "shared"))
    mx.eval(routed, weights, shared)
    expected = (4, 6, 4096)
    if tuple(routed.shape) != expected or tuple(weights.shape) != (4, 6):
        raise SystemExit(
            f"capture geometry {tuple(routed.shape)}/{tuple(weights.shape)} != "
            f"{expected}/(4, 6)"
        )
    if routed.dtype != mx.bfloat16 or shared.dtype != mx.bfloat16:
        raise SystemExit(
            f"capture dtype must be BF16/BF16, got {routed.dtype}/{shared.dtype}"
        )

    candidate = D._install_moe_tail_combine(model.args)
    stock = D._stock_moe_tail_combine(routed, weights, shared)
    stock_repair = D._stock_moe_tail_combine(
        routed[:1], weights[:1], shared[:1]
    )
    custom_apply_rows: list[int] = []
    real_apply = D._moe_tail_apply

    def observed_apply(kernel, observed_routed, observed_weights, observed_shared):
        custom_apply_rows.append(int(observed_routed.shape[0]))
        return real_apply(kernel, observed_routed, observed_weights, observed_shared)

    D._moe_tail_apply = observed_apply
    try:
        with attention_phase("decode_verify"):
            fused = candidate(routed, weights, shared)
            repair = candidate(routed[:1], weights[:1], shared[:1])
    finally:
        D._moe_tail_apply = real_apply
    mx.eval(stock, fused, stock_repair, repair)
    exact = bool(mx.array_equal(stock, fused))
    repair_exact = bool(mx.array_equal(stock_repair, repair))
    max_abs = float(mx.max(mx.abs(stock.astype(mx.float32) - fused.astype(mx.float32))).item())
    repair_max_abs = float(
        mx.max(mx.abs(stock_repair.astype(mx.float32) - repair.astype(mx.float32))).item()
    )
    if custom_apply_rows != [4]:
        raise SystemExit(
            "FAIL K3 route: expected only M4 to call custom kernel under "
            f"decode_verify, observed rows={custom_apply_rows}"
        )
    if not exact or not repair_exact:
        raise SystemExit(
            "FAIL exact parity: "
            f"verify_max_abs={max_abs:g} repair_max_abs={repair_max_abs:g}"
        )

    stock_seconds = _timed(lambda: D._stock_moe_tail_combine(routed, weights, shared), cycles=args.cycles)
    def fused_call():
        with attention_phase("decode_verify"):
            return candidate(routed, weights, shared)

    fused_seconds = _timed(fused_call, cycles=args.cycles)
    receipt = {
        "harness": "scripts/deepseek_v4_moe_tail_gate.py",
        "purpose": "one-load real-capture exact-parity and compile safety gate; TPS verdict is external",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "guard_window": guard_window,
        "identity": {
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "required_mlx_version": _REQUIRED_MLX_VERSION,
            "mlx_core_sha256": mlx_core_sha256,
            "mlx_lib_sha256": mlx_lib_sha256,
            "prompt_path": str(prompt_path),
            "prompt_sha256": prompt_sha256,
            "prompt_tokens": len(prompt_ids),
            "model": model_identity,
            "loaded_runtime": loaded_identity,
            "speculative_depth": 3,
            "verify_rows": 4,
        },
        "host": {"platform": platform.platform(),
                 "mlx_required": _REQUIRED_MLX_VERSION,
                 "mlx": mx.__version__, "mlx_file": str(mlx_core_path),
                 "mlx_lib_file": str(mlx_lib_path)},
        "model_path": str(model_path),
        "load_seconds": load_seconds,
        "prompt": {"path": str(prompt_path), "sha256": prompt_sha256,
                   "tokens": len(prompt_ids)},
        "capture": {"body_layer": args.layer, "routed_shape": list(routed.shape),
                    "weights_shape": list(weights.shape), "routed_dtype": str(routed.dtype),
                    "shared_dtype": str(shared.dtype)},
        "route_probe": {
            "attention_phase": "decode_verify",
            "speculative_depth": 3,
            "verify_rows": 4,
            "repair_rows": 1,
            "custom_apply_rows": custom_apply_rows,
            "verify_custom": custom_apply_rows == [4],
            "repair_stock": 1 not in custom_apply_rows,
        },
        "exact_parity": {"verify_m4": exact, "repair_m1": repair_exact},
        "max_abs": {"verify_m4": max_abs, "repair_m1": repair_max_abs},
        "diagnostic_seconds": {"stock": stock_seconds, "fused": fused_seconds,
                               "stock_median": _median(stock_seconds),
                               "fused_median": _median(fused_seconds)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
