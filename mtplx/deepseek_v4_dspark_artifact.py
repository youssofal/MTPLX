"""Construction-time contract for the DeepSeek V4 Flash DSpark artifact."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


_CONFIG_NAME = "config.json"
_INDEX_NAME = "model.safetensors.index.json"
_STAGE_KEY = re.compile(r"^mtp\.(\d+)\.")

_PHASE1_BLOCK_SIZE = 5
_PHASE1_MARKOV_RANK = 256
_PHASE1_NOISE_TOKEN_ID = 128799
_PHASE1_TARGET_LAYER_IDS = (40, 41, 42)
_PHASE1_STAGE_IDS = (0, 1, 2)

_K64_REQUIRED_WEIGHT_KEYS = (
    "mtp.0.main_proj.weight",
    "mtp.0.attn.wq_a.weight",
    "mtp.1.attn.wq_a.weight",
    "mtp.2.attn.wq_a.weight",
    "mtp.2.hc_head_fn",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.confidence_head.proj.weight",
)


class DSparkArtifactError(ValueError):
    """The selected model directory cannot install the Phase 1 DSpark lane."""


@dataclass(frozen=True)
class DSparkConfig:
    block_size: int
    markov_rank: int
    noise_token_id: int
    target_layer_ids: tuple[int, int, int]
    stage_ids: tuple[int, int, int]


@dataclass(frozen=True)
class VerifiedDSparkArtifact:
    root: Path
    weights_root: Path
    config: DSparkConfig
    config_sha256: str
    index_sha256: str
    weight_map: Mapping[str, str]
    shards: tuple[str, ...]


def _read_json(path: Path, *, label: str) -> tuple[bytes, dict]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DSparkArtifactError(f"cannot read DSpark {label} at {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DSparkArtifactError(f"invalid JSON in DSpark {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DSparkArtifactError(f"DSpark {label} must be a JSON object: {path}")
    return raw, value


def _phase1_config(config: Mapping[str, object], stage_ids: tuple[int, ...]) -> DSparkConfig:
    try:
        block_size = int(config["dspark_block_size"])
        markov_rank = int(config["dspark_markov_rank"])
        noise_token_id = int(config["dspark_noise_token_id"])
        target_layer_ids = tuple(int(v) for v in config["dspark_target_layer_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DSparkArtifactError(f"incomplete DSpark configuration: {exc}") from exc

    observed = (
        block_size,
        markov_rank,
        noise_token_id,
        target_layer_ids,
        stage_ids,
    )
    expected = (
        _PHASE1_BLOCK_SIZE,
        _PHASE1_MARKOV_RANK,
        _PHASE1_NOISE_TOKEN_ID,
        _PHASE1_TARGET_LAYER_IDS,
        _PHASE1_STAGE_IDS,
    )
    if observed != expected:
        raise DSparkArtifactError(
            "unsupported DSpark Phase 1 contract: "
            f"observed={observed!r}, expected={expected!r}"
        )

    return DSparkConfig(
        block_size=block_size,
        markov_rank=markov_rank,
        noise_token_id=noise_token_id,
        target_layer_ids=_PHASE1_TARGET_LAYER_IDS,
        stage_ids=_PHASE1_STAGE_IDS,
    )


def open_verified_dspark_artifact(root: Path) -> VerifiedDSparkArtifact:
    """Open and qualify the fixed-K5 DSpark checkpoint before model execution."""

    try:
        artifact_root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise DSparkArtifactError(f"DSpark artifact directory is unavailable: {root}") from exc
    if not artifact_root.is_dir():
        raise DSparkArtifactError(f"DSpark artifact root is not a directory: {artifact_root}")

    _, config = _read_json(artifact_root / _CONFIG_NAME, label="config")
    hybrid_tail = config.get("hybrid_tr3_tail")
    if not (
        isinstance(hybrid_tail, dict)
        and hybrid_tail.get("format") == "exl3-trellis"
    ):
        raise DSparkArtifactError(
            "the pinned Mia/Sero DSpark artifact requires an exl3-trellis "
            "split target"
        )

    from .deepseek_v4_exl3 import _default_mia_dspark_root
    from .deepseek_v4_mia_engine import validate_pinned_mia_artifacts

    weights_root = _default_mia_dspark_root(artifact_root).resolve()
    try:
        validation = validate_pinned_mia_artifacts(artifact_root, weights_root)
    except (OSError, TypeError, ValueError) as exc:
        raise DSparkArtifactError(
            f"pinned Mia/Sero target and K64 draft validation failed: {exc}"
        ) from exc

    weight_map = dict(validation.draft_weight_map)
    missing = tuple(key for key in _K64_REQUIRED_WEIGHT_KEYS if key not in weight_map)
    if missing:
        raise DSparkArtifactError(f"DSpark artifact is missing required weights: {missing!r}")

    stage_ids = tuple(
        sorted(
            {
                int(match.group(1))
                for key in weight_map
                if (match := _STAGE_KEY.match(key)) is not None
            }
        )
    )
    dspark_config = _phase1_config(validation.target_config, stage_ids)

    shards = tuple(sorted(pin.name for pin in validation.draft_shards))
    target_small_file_sha256 = dict(validation.target_small_file_sha256)
    draft_small_file_sha256 = dict(validation.draft_small_file_sha256)

    return VerifiedDSparkArtifact(
        root=artifact_root,
        weights_root=weights_root,
        config=dspark_config,
        config_sha256=target_small_file_sha256[_CONFIG_NAME],
        index_sha256=draft_small_file_sha256[_INDEX_NAME],
        weight_map=MappingProxyType(weight_map),
        shards=shards,
    )
