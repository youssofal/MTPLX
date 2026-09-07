"""Pinned identity for the public DeepSeek V4 target-only MLX artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEEPSEEK_V4_TARGET_ONLY_REPO_ID = (
    "philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly"
)
# Pin the immutable weight publication rather than a later model-card-only
# commit.  Documentation can evolve without silently changing the runtime
# artifact that MTPLX admits.
DEEPSEEK_V4_TARGET_ONLY_REVISION = "ac33e4f3ca3546e6cec104558d42161e15814e33"
DEEPSEEK_V4_TARGET_ONLY_WEIGHT_SHARDS = tuple(
    [f"model-layer-{idx}.safetensors" for idx in range(43)] + ["model-top.safetensors"]
)


def _shard_sizes() -> dict[str, int]:
    sizes = {
        "model-layer-0.safetensors": 2_232_226_400,
        "model-layer-1.safetensors": 2_232_226_400,
        "model-layer-2.safetensors": 2_262_658_336,
        "model-top.safetensors": 1_125_524_468,
    }
    for layer in range(3, 10):
        sizes[f"model-layer-{layer}.safetensors"] = (
            2_234_674_232 if layer % 2 else 2_256_453_912
        )
    for layer in range(10, 39):
        sizes[f"model-layer-{layer}.safetensors"] = (
            2_234_674_280 if layer % 2 else 2_256_453_968
        )
    for layer in range(39, 43):
        sizes[f"model-layer-{layer}.safetensors"] = (
            3_778_178_160 if layer % 2 else 3_799_957_848
        )
    return sizes


DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES = _shard_sizes()
DEEPSEEK_V4_TARGET_ONLY_SHARD_SHA256 = {
    "model-layer-0.safetensors": "1c7a2069ad82137ed463a0632d6baee4eec8d719eebbf08df79e5d2c61877fd4",
    "model-layer-1.safetensors": "96d7c122914cc6c72290852971a09112411e30b11c16c90a9fc056e9a808b4d4",
    "model-layer-2.safetensors": "c8fd7bca1dacb40d325911110a0871910661dcb2487beaf43f9bb633368c9beb",
    "model-layer-3.safetensors": "feeec5fb91e5c2e52dc7894f37f5f485317c91d4e6c00e3850c32ab2c658788c",
    "model-layer-4.safetensors": "a047c055c20689b648018b64f5905568c116b4edf88a98ba672fc5de46a4f858",
    "model-layer-5.safetensors": "0d32c40401ee99a160cef49069d58f34d09395359fd9b0d54adfdf37f7453b4d",
    "model-layer-6.safetensors": "8974d3acdd6a389bf2f8035e848155d4395d3307556e8b588a298c2aa1bf046d",
    "model-layer-7.safetensors": "732d08c85c6624e54a247da41d1d145928fe78e98eb24b76efd389d65a4893a4",
    "model-layer-8.safetensors": "a9df491f2a54a21c1050a14a7f3f9a731e2231c0ef783de7029e211f46f48628",
    "model-layer-9.safetensors": "1e073b390d0b3666aa3131fc0241275c0d7714b00a6517482703925465561d5a",
    "model-layer-10.safetensors": "afb100ff10ba50581c4f06346439b1f59c97773e6679134e0c00240041382590",
    "model-layer-11.safetensors": "de7c1c783c97ad9dc7589fbfe51823849ea63071ec81a7ca59dcb3d03b1273db",
    "model-layer-12.safetensors": "827302c65f67142dd0c5b6d573d9cadba2a5df68aabbde285bdaf060249b8fc2",
    "model-layer-13.safetensors": "987cba23a581b46954c8dfa2dbcd863dca76f66597cc1a83e14682293e79d487",
    "model-layer-14.safetensors": "9204ed3ba22db5d2b1dc83e47e09cbe2e0afc0edf2ce1b16a003b0efb03c9fa4",
    "model-layer-15.safetensors": "7d848857e65751f56b260b932d72aef99491229d501acdc141214767782809aa",
    "model-layer-16.safetensors": "dd5a3766aaaa8acde59baa03dcbe585ed4a01edcfcaa9d245706207acef29b5b",
    "model-layer-17.safetensors": "77c33de412a75d19d933fa2d9574d6b4f09a8f46d22673cda0377a61f02c375f",
    "model-layer-18.safetensors": "3e3380b3f9d323a0f11d9f599812d9cf0032ddebded5ae06fe0c140cfe545728",
    "model-layer-19.safetensors": "90c4e2a4a6b8a64a9f18b4757c84a0152d67c12510a7e8556e3ef22d0cf23b4d",
    "model-layer-20.safetensors": "a0cc9a6e31332182a4d3f853d377fd6363ec28975071904177158a5bc6495f51",
    "model-layer-21.safetensors": "4ed8c6545bf2cf2c1491c7d2473cbda918aec217eafed78005a2b81dc6a1e243",
    "model-layer-22.safetensors": "c698e641ddcc60fd0759602c4643fb270d65450971c8c704bbfb33907f8ab0c5",
    "model-layer-23.safetensors": "f477b9ae3af9e625403256b41cb0fd74bce2fd65361aea78942bb61c5e5329ed",
    "model-layer-24.safetensors": "321985d1cb6c3ddcc43de2eacd87a1602c6850f3378b2850ae56013d99e09305",
    "model-layer-25.safetensors": "9e8021f7d6e0c6bed50e1c10208454324cbef75c23fe6b4e577d0d1a738eafb3",
    "model-layer-26.safetensors": "59ed379f35e732d78e11810043ca6b87362ac2c926be63abe7aa9f21d1b59f91",
    "model-layer-27.safetensors": "d7e5f5edf015168d093fea227d7708f468079b8bcea532807175ca42951beab4",
    "model-layer-28.safetensors": "00d98eaa48f811b58d09d6ebee6ea31e588a1e84df4ad3f8c216180345f6901a",
    "model-layer-29.safetensors": "47a41736a48f2819c2f79f5316aa9bbe56f668ab61ac6b183d55be2ef0df8bb2",
    "model-layer-30.safetensors": "06105af5adbabfefa412400676f2a7dac4773f360d04e257223fca0d5614a06f",
    "model-layer-31.safetensors": "4051556446455e19585622f7ddfee8adc1dafde2e0c893386b1817f05d54ffe2",
    "model-layer-32.safetensors": "bb55bf7c25105f10f6837badb375172742cf8c696db55a6465ba238cf0a4f20f",
    "model-layer-33.safetensors": "6cf3427ac909a96132782a2e51209da804b60bf6a9bf596448cc521e5067f70f",
    "model-layer-34.safetensors": "b19ce6bc09991cb7598aa17f2cc413e4d3c8ab06a38ab4574edcded7db0d4abb",
    "model-layer-35.safetensors": "4f2632eda38cc96549641e1ec6428879a44c9749bd14eb3471b14734b3e4ee51",
    "model-layer-36.safetensors": "fba407165aef0fd25322f8bec897e0c4bfacefbdce2976a0b194399c115f37a5",
    "model-layer-37.safetensors": "f05c0f5294d910a7f078be9c8c9d424bce536b93c50af301a95d5bf2e80bd69e",
    "model-layer-38.safetensors": "14cef91cb94947fd1e6f82bc7d8798264e336c757a0923bfa6699c8e91bf4aa2",
    "model-layer-39.safetensors": "a0594fe95badf8540efe476ed870339f1449aeea774decedd4a39a5a280441b7",
    "model-layer-40.safetensors": "9752520416dfc125b0ecee74e18bf8678c2ae8d11fc6fcd7ee8ded757f230440",
    "model-layer-41.safetensors": "41216ecd544e1fb02167491cd84a44e2a975d2ad2b58fcf6e7f507801b8606dc",
    "model-layer-42.safetensors": "0755a615f5d49837ab2f7afea89e0601a4f05e4d004664690b58e1957de0c488",
    "model-top.safetensors": "11a33b3911d6723ee60a30822218068d96a4a23e6daef2e36b9bd2551b815e73",
}
DEEPSEEK_V4_TARGET_ONLY_SIDECAR_SHA256 = {
    "config.json": "ab61e3230f196c6eba04bfa81158dd527a7f356b6d926cc4794907a19f35b75d",
    "generation_config.json": "d14db17fca8dc5492af88fd7938475e3a9fdddddd7596a5062d2ecd58abb942b",
    "model.safetensors.index.json": "589b2290d9a24081171d28424e4cd170be6e1820853a286e446cd580d456da8e",
    "tokenizer.json": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    "tokenizer_config.json": "58a9fb7a7e68144c0b6fe4bc8349ee77d818702f95c9c6aa41143d9e27d7c2e6",
}
DEEPSEEK_V4_TARGET_ONLY_REQUIRED_FILES = frozenset(
    (*DEEPSEEK_V4_TARGET_ONLY_SIDECAR_SHA256, *DEEPSEEK_V4_TARGET_ONLY_WEIGHT_SHARDS)
)
DEEPSEEK_V4_TARGET_ONLY_REPO_BYTES = 103_855_774_263


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_deepseek_v4_target_only_config(config: dict[str, Any]) -> bool:
    if not isinstance(config, dict) or "model_file" in config:
        return False
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return False
    try:
        if not (
            config.get("architectures") == ["DeepseekV4ForCausalLM"]
            and config.get("model_type") == "deepseek_v4"
            and int(config.get("num_hidden_layers") or 0) == 43
            and int(config.get("hidden_size") or 0) == 4096
            and int(config.get("num_attention_heads") or 0) == 64
            and int(config.get("num_key_value_heads") or 0) == 1
            and int(config.get("head_dim") or 0) == 512
            and int(config.get("vocab_size") or 0) == 129_280
            and int(config.get("num_nextn_predict_layers") or 0) == 0
            and int(config.get("dspark_block_size") or 0) == 0
            and int(config.get("num_experts_per_tok") or 0) == 6
            and int(config.get("n_routed_experts") or 0) == 256
            and quantization.get("bits") == 8
            and quantization.get("group_size") == 64
            and quantization.get("mode") == "affine"
        ):
            return False
    except (TypeError, ValueError):
        return False

    def affine(name: str, bits: int, group_size: int) -> bool:
        value = quantization.get(name)
        return bool(
            isinstance(value, dict)
            and value.get("bits") == bits
            and value.get("group_size") == group_size
            and value.get("mode") == "affine"
        )

    if not affine("embed", 8, 64) or not affine("head", 8, 64):
        return False
    # The public target-only view has a layer-specific expert recipe; accepting
    # only the global metadata would let a different conversion claim the same
    # family identity. Layers 0-38 use 2/3/2-bit g128 experts; 39-42 use
    # 4/4/4-bit g64 experts.
    for layer in range(43):
        bits = (2, 3, 2) if layer < 39 else (4, 4, 4)
        group_size = 128 if layer < 39 else 64
        for projection, projection_bits in zip(
            ("w1", "w2", "w3"), bits, strict=True
        ):
            if not affine(
                f"layers.{layer}.ffn.experts.{projection}",
                projection_bits,
                group_size,
            ):
                return False
    return True


def deepseek_v4_target_only_artifact_integrity_errors(
    model_path: Path | str,
    *,
    verify_shard_hashes: bool = True,
) -> tuple[str, ...]:
    root = Path(model_path)
    errors: list[str] = []
    for name, expected_size in DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES.items():
        path = root / name
        try:
            if not path.is_file() or path.stat().st_size != expected_size:
                errors.append(name)
            elif (
                verify_shard_hashes
                and _sha256(path) != (DEEPSEEK_V4_TARGET_ONLY_SHARD_SHA256[name])
            ):
                errors.append(name)
        except OSError:
            errors.append(name)
    for name, expected_sha256 in DEEPSEEK_V4_TARGET_ONLY_SIDECAR_SHA256.items():
        path = root / name
        try:
            if not path.is_file() or _sha256(path) != expected_sha256:
                errors.append(name)
        except OSError:
            errors.append(name)
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or set(weight_map.values()) != set(
            DEEPSEEK_V4_TARGET_ONLY_WEIGHT_SHARDS
        ):
            errors.append("model.safetensors.index.json:weight_map")
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("model.safetensors.index.json:parse")
    return tuple(sorted(set(errors)))
