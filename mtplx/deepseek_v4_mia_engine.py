"""Construction-owned execution plan for the pinned Mia/Sero DSpark model.

The exact lane is deliberately closed over one artifact and one serving
geometry.  Validation happens here, before a cache or request exists.  The
request path receives already-bound target, draft, cache, and phase callables;
it never re-reads launcher settings or probes whether the exact lane applies.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any

from mtplx.deepseek_v4_paged_indexer import precompute_indexer_rope_table


MIA_CONTEXT_CAPACITY = 384_000
MIA_MAX_BATCH_TOKENS = 8_224
MIA_LONG_PREFILL_CHUNK = 1_024
MIA_MAX_SEQUENCES = 1
MIA_TARGET_LAYERS = 43
MIA_DSPARK_STAGES = 3
MIA_DSPARK_BLOCK = 5
MIA_TARGET_PHYSICAL_CAPACITY = MIA_CONTEXT_CAPACITY + MIA_DSPARK_BLOCK
MIA_TARGET_TAPS = (40, 41, 42)
MIA_TARGET_EXPERTS = 216
MIA_DRAFT_EXPERTS = 64
MIA_TOPK = 6
MIA_MXFP8_MODULES = 390
MIA_INDEX_TOPK = 512
MIA_WINDOW = 128
MIA_HEADS = 64
MIA_HEAD_DIM = 512
MIA_ROPE_DIM = 64
MIA_HIDDEN = 4096
MIA_HC = 4
_MIA_TARGET_COMPRESS_RATIOS = (0, 0) + (4, 128) * 20 + (4,)
MIA_DRAFT_SHARD_BYTES = 3_157_508_012
MIA_DFLASH_COMMIT = "54644e991039110f30140006c892c57734b9311e"
MIA_TARGET_REVISION = "22f28d32b9b29b4352eaa380ff8c2c170b2847ab"
MIA_SOURCE_CONFIG_SHA256 = "b001ec8308044aa11daa0e624f5aea5e5362a63c05879a83a7be046b00eada82"
MIA_SOURCE_INDEX_SHA256 = "61af5c0782a8651ef893004e84369d2281a0fc316c8bcefc0bd8f76244224649"
MIA_SOURCE_EXL3_SHA256 = "1e35cbbc33a977606a950928fba4c6660c7df0134bfab9472dd6d851be894125"
MIA_EXL3_M6_QUAD_DESCRIPTOR_SHA256 = (
    "158d8b220411e42a910b29a47d8af0f045b4eb1feec745cfa39b9997db72efa2"
)


def _require_mia_target_ratio_order(ratios: tuple[int, ...]) -> tuple[int, ...]:
    """Seal the artifact's exact per-layer compressor route at construction."""

    observed = tuple(int(ratio) for ratio in ratios)
    if observed != _MIA_TARGET_COMPRESS_RATIOS:
        raise ValueError(
            "installed Mia target compress-ratio order changed: "
            f"{observed!r} != {_MIA_TARGET_COMPRESS_RATIOS!r}"
        )
    return observed

_TARGET_SMALL_FILE_PINS = {
    "config.json": "39f3a9e158019dc34dd943b64f874cfc43e9e392e6ce9215a56f2e183d661d90",
    "tokenizer.json": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    "tokenizer_config.json": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    "model.safetensors.index.json": "b7a450f88c99aee7f6d44ecb127e91e45ab5ccb1a0dad49ca9eabb90b400c304",
    "EXL3_MANIFEST.json": "1e35cbbc33a977606a950928fba4c6660c7df0134bfab9472dd6d851be894125",
}
_DRAFT_SMALL_FILE_PINS = {
    "config.json": "8dcd2ae923a8e3149454f4db1f1e03109625b19f137995d26f45d357212ba306",
    "model.safetensors.index.json": "c0d0e18e8c84fe6f1b7dc6991a4ba5765d1965f21f8892887aa01169fc2ba2b3",
}
_TARGET_CANONICAL_ARTIFACT_SEAL = "c05e8ecb1d387cc351d9c5733689343ccca9d92c2be663954ce154bd43befd7d"
_DRAFT_CANONICAL_PLAN_SEAL = "b371a65a0f452040109648551eb362bebfa802a15d92659a4cf0b90f23a57cd9"
_TARGET_CANONICAL_SHARD_PINS = {
    "carried-001.safetensors": "172dc4fddaaf003eb4477a5a839358afe31bec041add0f57a7423009cf5fffde",
    "carried-002.safetensors": "9de257123f6e99efe4ddbd9d89ebe159f83ce53e9232fb6f1f08b7adf1692c11",
    "carried-003.safetensors": "e59d5a51203a2ffd91c4ec401302043f30b4ae5140817f5e746e889eaa0541d1",
    "carried-004.safetensors": "43bcc1adbce2a7f2cd4b46178d73a8205b9dc73c529c24507b085a6a2af0490c",
    "carried-005.safetensors": "c902086dcfb6f686145f44568df9144499687f12ae8688dce369ab55c55c9698",
    "exl3-layer-000-tp1-rank0.safetensors": "56e1ee251d028e9e0c8404789d2974bd45c16d136eba149ac2c0fbad1eb39ba0",
    "exl3-layer-001-tp1-rank0.safetensors": "0a1b8c4d61a65745987f8733c6a0a0a8d9375f59bb8d5271fe1bcde25656d739",
    "exl3-layer-002-tp1-rank0.safetensors": "a8e6d9e09e3a8d974bacafec96c38a79907210d2f1464afbccddc2034672fbcf",
    "exl3-layer-003-tp1-rank0.safetensors": "123a4e571ad15461994648b193a6c44e4987076f994fe8137345a4e657f4e196",
    "exl3-layer-004-tp1-rank0.safetensors": "5d4eb35deb25545866ebcb4c22be212c587868f71d5548d79cded1df8514498c",
    "exl3-layer-005-tp1-rank0.safetensors": "5c2912083933c53bac9683629e69b69aff1941135665f8209ba74067c8d3d9c2",
    "exl3-layer-006-tp1-rank0.safetensors": "15201a69a18ce178dbaf30a3a88969ee5e6da89d22c64f6d8bedbe14fabb86bb",
    "exl3-layer-007-tp1-rank0.safetensors": "3e7c9fdf321f793b35449a874273cdfe03f1f79bd7bdf345193eaf505fcb07ef",
    "exl3-layer-008-tp1-rank0.safetensors": "f914659192e27fbe063539e20a1768cf8745f20cb2bfe2b906a5b3eae0a69cbf",
    "exl3-layer-009-tp1-rank0.safetensors": "6aa7d79141a7e017efa160374e438311c073a1ac1ebd6e8990316b5d6169cb4f",
    "exl3-layer-010-tp1-rank0.safetensors": "953a67f1fcb3e9b9b71c7254cdd709fa8a7e86103b733481bd2466db30f2c7f2",
    "exl3-layer-011-tp1-rank0.safetensors": "8044b3604d655ea74b73445db887fdaa309bfd71118571edcef396abb0e3c03c",
    "exl3-layer-012-tp1-rank0.safetensors": "367d899623ddddb245ceba4a4d255b86c50a2dcc250b5bd02ea0482d32809a70",
    "exl3-layer-013-tp1-rank0.safetensors": "e383380b866f5d3462635fd767bb5fd66928ad8407245198c61df44463501fe7",
    "exl3-layer-014-tp1-rank0.safetensors": "bbadbabacbbcd3806dbe2ca1479a35e8da1a9b66db9f045a6d7cc44a57024bdf",
    "exl3-layer-015-tp1-rank0.safetensors": "b3702b5769c2d764cabeb486cd7f1534dad9664d132564a618a655e552784d7d",
    "exl3-layer-016-tp1-rank0.safetensors": "a9b82623f29e73c68b3e13b4129d7bf469cf78bdfd983edf16103b2e89966919",
    "exl3-layer-017-tp1-rank0.safetensors": "d9c57a476f827472c60068c714a5139509e4b0b7f867c2b32c5e02545402e6db",
    "exl3-layer-018-tp1-rank0.safetensors": "4052604ce5c928e1e20045840a76e0d25e594113ba276786dc5058fa9f559bf5",
    "exl3-layer-019-tp1-rank0.safetensors": "9672b3c5d69fc99c453a123a10d4c43b30f7fcb1e7e006b8aec8ed4e5cc3ce10",
    "exl3-layer-020-tp1-rank0.safetensors": "95c5f76a8991d39d83cd68aea6be51459e5e3dbebd4e1c42f9dd92e5b6fe06dd",
    "exl3-layer-021-tp1-rank0.safetensors": "4d44e129f982a52cf8d84a391ca99b2bba7dfa6b6b4e55f5e38f922700993162",
    "exl3-layer-022-tp1-rank0.safetensors": "94c61252e7f75ade0a3a6ad9fb35befb84854ae49bfef7fc2c8e89ec1747b94b",
    "exl3-layer-023-tp1-rank0.safetensors": "3e2cc5dbfbe2a8359c5e3980083d1907c21aab8d28c4b6252f31d24354f1c80d",
    "exl3-layer-024-tp1-rank0.safetensors": "45915ad1d260106ecbda0c23b5a6555cc4c85999c43c17c221849440f11f96f2",
    "exl3-layer-025-tp1-rank0.safetensors": "350004098d1b5758b305c26046bca9dec30a810b972b1c4b018392c480872667",
    "exl3-layer-026-tp1-rank0.safetensors": "f132ccc3ae9ae0bcf5ad781e1193df9e3b51a359d589001adbf7ccb2572eb3bd",
    "exl3-layer-027-tp1-rank0.safetensors": "33b2099ece51752f174e1224d94fc5986b4fc06a883c8b6fa1ef9692e33669d9",
    "exl3-layer-028-tp1-rank0.safetensors": "ad396044a0d202284526f9f99bcf7f799f54aa81bf32b8644c31d4f6e5c80c68",
    "exl3-layer-029-tp1-rank0.safetensors": "e92fddd438b5602344dc753aa571b312571763e5d9a81e1c46d2f5c6614ca207",
    "exl3-layer-030-tp1-rank0.safetensors": "49ae0f675bf5b826c710e888760435547467147f1334fc383803483e953380ae",
    "exl3-layer-031-tp1-rank0.safetensors": "a554f2bb650247b24d0d42151bc2159b458a5d6b635af431fa28ed3e59e2656b",
    "exl3-layer-032-tp1-rank0.safetensors": "dde21dd115b9f1275c917789740b2b707ac8255351ff68c0452a7883e9d0e704",
    "exl3-layer-033-tp1-rank0.safetensors": "ce2b40b1ebdb4412b1a62f18b6ffdb8b66fb26a5e8a5b8be556200acacf0a5c9",
    "exl3-layer-034-tp1-rank0.safetensors": "d74d19f932e9679903c543622490c253cc0fb68ad2e5ff3da303bbff53b8a164",
    "exl3-layer-035-tp1-rank0.safetensors": "e250c12511f34061b2d743dcf14ca49bc49c201ea295577e162e9826a6af6bed",
    "exl3-layer-036-tp1-rank0.safetensors": "ec454bbeb420bc528fdf2e557994764eee46c04a67b8aff4ce02a23c317e02aa",
    "exl3-layer-037-tp1-rank0.safetensors": "97b438c0009dd390dd9db027d51e9491ea31b8b3aa3293406363f3670e0f0fb2",
    "exl3-layer-038-tp1-rank0.safetensors": "464d2577ab866072548e90b7913302c3e5fa2109f63190e95a0cc9bb91b27e79",
    "exl3-layer-039-tp1-rank0.safetensors": "f9437df4893794a69121cbc137de7cd6906cd5426da0b7595ff88b387326b154",
    "exl3-layer-040-tp1-rank0.safetensors": "61e16010ffa32524bc35c92e67e345dfbf176af9b32537f6e7403356cc0a9aea",
    "exl3-layer-041-tp1-rank0.safetensors": "a45233df123226cf1476ef4d02126d79b1ef3e25a65bc28b35a0da64429cd04a",
    "exl3-layer-042-tp1-rank0.safetensors": "5440bca5af2ed689fbfd80b241fb5a69a58c85c1f5b64a17f96b52e67cad1989",
}
_DRAFT_CANONICAL_SHARD_PIN = "5d510d98e9a744aa78724b54250c8f55e319fc4ab8d44db8b68aafc1cbfe6b15"
_TOKENIZER_CONSUMED_SMALL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_MIA_ATTENTION_ROUTE_CONTRACTS = {
    0: (
        "_mia_cached_forward_uncompressed",
        "_mia_cached_attention_ratio0",
        "_mia_uncached_compressed",
        "_run_installed_window_nvfp4_sparse_mla",
        "_run_installed_window_nvfp4_prefill_mla",
    ),
    4: (
        "_mia_cached_forward_ratio4",
        "_mia_cached_attention_ratio4",
        "_mia_uncached_compressed",
        "_run_installed_indexed_paged_nvfp4_sparse_mla",
        "_run_installed_indexed_paged_nvfp4_prefill_mla",
    ),
    128: (
        "_mia_cached_forward_ratio128",
        "_mia_cached_attention_ratio128",
        "_mia_uncached_compressed",
        "_run_installed_sequential_paged_nvfp4_sparse_mla",
        "_run_installed_sequential_paged_nvfp4_prefill_mla",
    ),
}


def _callable_name(value: Any) -> str:
    """Return the installed implementation name, unwrapping bound partials."""

    current = value
    while callable(getattr(current, "func", None)):
        current = current.func
    return str(getattr(current, "__name__", type(current).__name__))


@dataclass(frozen=True, slots=True)
class MiaSmallFilePin:
    name: str
    bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


_SAFETENSORS_CANONICAL_PREFIX = b"mtplx-mia-safetensors-json-order-v1\0"
_TARGET_ARTIFACT_CANONICAL_PREFIX = b"mtplx-mia-target-artifact-v1\0"
_DRAFT_PLAN_CANONICAL_PREFIX = b"mtplx-mia-draft-plan-v1\0"


def _small_file_identity(observed: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError("pinned Mia small file must be a regular file")
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _json_object_without_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant: {value}")


def _canonical_safetensors_header(encoded_header: bytes) -> bytes:
    try:
        header = json.loads(
            encoded_header,
            object_pairs_hook=_json_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid Mia safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("invalid Mia safetensors JSON header")
    return json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(prefix: bytes, value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    return digest.hexdigest()


def _target_artifact_seal_sha256(
    manifest: dict[str, Any],
    shard_seals: dict[str, str | MiaShardPin],
) -> str:
    source = manifest.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"pinned Mia source contract changed: {source!r}")
    sealed_manifest = dict(manifest)
    sealed_manifest["source"] = {
        "repository": "0xSero/deepseek-v4-flash-0731-spark",
        "revision": MIA_TARGET_REVISION,
        "config_sha256": MIA_SOURCE_CONFIG_SHA256,
        "index_sha256": MIA_SOURCE_INDEX_SHA256,
        "exl3_manifest_sha256": MIA_SOURCE_EXL3_SHA256,
    }
    sealed_files = []
    for entry in manifest.get("files", ()):
        sealed_entry = {
            key: value for key, value in entry.items() if key != "sha256"
        }
        name = str(sealed_entry.get("name", ""))
        seal = shard_seals[name]
        canonical_sha256 = (
            seal if isinstance(seal, str) else seal.canonical_sha256
        )
        if len(canonical_sha256) != 64:
            raise ValueError(
                f"pinned Mia TP1 canonical shard checksum is invalid: {name}"
            )
        sealed_entry["canonical_safetensors_sha256"] = canonical_sha256
        sealed_files.append(sealed_entry)
    sealed_manifest["files"] = sealed_files
    sealed_manifest["small_file_sha256"] = {
        name: digest
        for name, digest in _TARGET_SMALL_FILE_PINS.items()
    }
    return _canonical_json_sha256(
        _TARGET_ARTIFACT_CANONICAL_PREFIX,
        sealed_manifest,
    )


def _draft_plan_seal_sha256(plan: dict[str, Any]) -> str:
    sealed_plan = {
        key: value
        for key, value in plan.items()
        if key not in {"source", "source_plan", "sha256"}
    }
    return _canonical_json_sha256(_DRAFT_PLAN_CANONICAL_PREFIX, sealed_plan)


def _read_pinned_small_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[bytes, MiaSmallFilePin]:
    payload, identity = _read_stable_small_file(path, label=label)
    if identity.sha256 != expected_sha256:
        raise ValueError(
            f"pinned {label} file changed: {path.name} "
            f"observed={identity.sha256}, expected={expected_sha256}"
        )
    return payload, identity


def _read_stable_small_file(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, MiaSmallFilePin]:
    try:
        with path.open("rb", buffering=0) as stream:
            before = _small_file_identity(os.fstat(stream.fileno()))
            payload = stream.read()
            after = _small_file_identity(os.fstat(stream.fileno()))
    except OSError as exc:
        raise FileNotFoundError(f"pinned {label} file is absent: {path}") from exc
    if after != before:
        raise ValueError(f"pinned {label} file changed while validating: {path.name}")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    return payload, MiaSmallFilePin(
        name=path.name,
        bytes=before[2],
        sha256=observed_sha256,
        device=before[0],
        inode=before[1],
        mtime_ns=before[3],
        ctime_ns=before[4],
    )


def _validate_small_files(
    root: Path,
    pins: dict[str, str],
    label: str,
) -> tuple[dict[str, bytes], tuple[MiaSmallFilePin, ...]]:
    validated: dict[str, bytes] = {}
    identities: list[MiaSmallFilePin] = []
    for name, expected in pins.items():
        path = root / name
        payload, identity = _read_pinned_small_file(
            path,
            expected_sha256=expected,
            label=label,
        )
        validated[name] = payload
        identities.append(identity)
    return validated, tuple(identities)


@dataclass(frozen=True, slots=True)
class MiaShardPin:
    name: str
    bytes: int
    sha256: str
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class MiaArtifactValidation:
    target_root: Path
    draft_root: Path
    target_config: dict[str, Any]
    target_weight_map: dict[str, str]
    target_shards: tuple[MiaShardPin, ...]
    target_small_files: tuple[MiaSmallFilePin, ...]
    target_small_file_sha256: tuple[tuple[str, str], ...]
    draft_config: dict[str, Any]
    draft_weight_map: dict[str, str]
    draft_shards: tuple[MiaShardPin, ...]
    draft_small_files: tuple[MiaSmallFilePin, ...]
    draft_small_file_sha256: tuple[tuple[str, str], ...]


def validate_pinned_mia_artifacts(
    target_root: Path,
    draft_root: Path,
) -> MiaArtifactValidation:
    """Validate pinned metadata and return shard pins for integrated loading."""

    target_root = Path(target_root).resolve()
    draft_root = Path(draft_root).resolve()
    target_files, target_small_files = _validate_small_files(
        target_root,
        _TARGET_SMALL_FILE_PINS,
        "Mia target",
    )
    draft_files, draft_small_files = _validate_small_files(
        draft_root,
        _DRAFT_SMALL_FILE_PINS,
        "K64 DSpark",
    )
    target_manifest_payload, target_manifest_pin = _read_stable_small_file(
        target_root / "rank-sliced-tp1-manifest.json",
        label="Mia target",
    )
    target_files["rank-sliced-tp1-manifest.json"] = target_manifest_payload
    target_small_files = (*target_small_files, target_manifest_pin)
    draft_plan_payload, draft_plan_pin = _read_stable_small_file(
        draft_root / "DSPARK_DRAFT_PLAN.json",
        label="K64 DSpark",
    )
    draft_files["DSPARK_DRAFT_PLAN.json"] = draft_plan_payload
    draft_small_files = (*draft_small_files, draft_plan_pin)

    target_config = json.loads(target_files["config.json"])
    target_index = json.loads(target_files["model.safetensors.index.json"])
    target_manifest = json.loads(
        target_files["rank-sliced-tp1-manifest.json"]
    )
    manifest_contract = (
        target_manifest.get("format"),
        int(target_manifest.get("source_tp", 0)),
        int(target_manifest.get("target_tp", 0)),
        int(target_manifest.get("tensor_count", 0)),
        int(target_manifest.get("tensor_bytes", 0)),
    )
    if manifest_contract != (
        "rank-sliced-exl3-tp1-v1",
        4,
        1,
        117_005,
        106_084_465_528,
    ):
        raise ValueError(
            "pinned Mia TP1 manifest contract changed: "
            f"{manifest_contract!r}"
        )
    files = target_manifest.get("files")
    if not isinstance(files, list) or len(files) != 48:
        raise ValueError("pinned Mia TP1 manifest must own exactly 48 shards")
    target_shards: list[MiaShardPin] = []
    target_seals: dict[str, str] = {}
    shard_names: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("pinned Mia TP1 manifest has an invalid shard entry")
        name = str(entry.get("name", ""))
        if Path(name).name != name or name in shard_names:
            raise ValueError(f"pinned Mia TP1 shard name changed: {name!r}")
        shard_names.add(name)
        shard = target_root / name
        expected_bytes = int(entry["bytes"])
        if not shard.is_file() or shard.stat().st_size != expected_bytes:
            raise ValueError(
                f"pinned Mia TP1 shard size changed: {shard.name}"
            )
        expected_sha256 = str(entry.get("sha256", ""))
        if len(expected_sha256) != 64:
            raise ValueError(f"pinned Mia TP1 shard checksum is invalid: {name}")
        canonical_sha256 = _TARGET_CANONICAL_SHARD_PINS.get(name, "")
        if len(canonical_sha256) != 64:
            raise ValueError(
                f"pinned Mia TP1 canonical shard checksum is invalid: {name}"
            )
        target_seals[name] = canonical_sha256
        target_shards.append(
            MiaShardPin(
                name,
                expected_bytes,
                expected_sha256,
                canonical_sha256,
            )
        )
    if set(_TARGET_CANONICAL_SHARD_PINS) != shard_names:
        raise ValueError("pinned Mia TP1 canonical shard set changed")
    observed_target_seal = _target_artifact_seal_sha256(
        target_manifest,
        target_seals,
    )
    expected_target_seal = _TARGET_CANONICAL_ARTIFACT_SEAL
    if observed_target_seal != expected_target_seal:
        raise ValueError(
            "pinned Mia TP1 canonical artifact seal changed: "
            f"observed={observed_target_seal}, expected={expected_target_seal}"
        )
    target_weight_map = target_index.get("weight_map")
    if (
        not isinstance(target_weight_map, dict)
        or len(target_weight_map) != 117_005
        or set(target_weight_map.values()) != shard_names
        or int(target_index.get("metadata", {}).get("total_size", 0))
        != 106_084_465_528
    ):
        raise ValueError("pinned Mia TP1 safetensors index changed")

    draft_config = json.loads(draft_files["config.json"])
    draft_plan = json.loads(draft_files["DSPARK_DRAFT_PLAN.json"])
    draft_index = json.loads(draft_files["model.safetensors.index.json"])
    if (
        int(draft_plan.get("draft_experts", 0)) != MIA_DRAFT_EXPERTS
        or int(draft_plan.get("source_experts", 0)) != MIA_TARGET_EXPERTS
        or int(draft_plan.get("tensor_count", 0)) != 1249
        or int(draft_index.get("metadata", {}).get("total_size", 0))
        != int(draft_plan.get("total_size", -1))
    ):
        raise ValueError("pinned K64 DSpark derivation manifest changed")
    observed_draft_plan_seal = _draft_plan_seal_sha256(draft_plan)
    expected_draft_plan_seal = _DRAFT_CANONICAL_PLAN_SEAL
    if observed_draft_plan_seal != expected_draft_plan_seal:
        raise ValueError(
            "pinned K64 DSpark canonical plan seal changed: "
            f"observed={observed_draft_plan_seal}, "
            f"expected={expected_draft_plan_seal}"
        )
    weight_map = draft_index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 1249:
        raise ValueError("pinned K64 DSpark weight index changed")
    filenames = set(weight_map.values())
    if filenames != {"dspark-draft.safetensors"}:
        raise ValueError("pinned K64 DSpark must own one packaged weight shard")
    draft_shard = draft_root / "dspark-draft.safetensors"
    if not draft_shard.is_file():
        raise FileNotFoundError(draft_shard)
    if draft_shard.stat().st_size != MIA_DRAFT_SHARD_BYTES:
        raise ValueError("pinned K64 DSpark shard size changed")
    draft_sha256 = str(draft_plan.get("sha256", {}).get(draft_shard.name, ""))
    if len(draft_sha256) != 64:
        raise ValueError("pinned K64 DSpark shard checksum is invalid")
    if len(_DRAFT_CANONICAL_SHARD_PIN) != 64:
        raise ValueError("pinned K64 DSpark canonical shard checksum is invalid")
    return MiaArtifactValidation(
        target_root=target_root,
        draft_root=draft_root,
        target_config=dict(target_config),
        target_weight_map={
            str(name): str(filename) for name, filename in target_weight_map.items()
        },
        target_shards=tuple(target_shards),
        target_small_files=target_small_files,
        target_small_file_sha256=tuple(
            sorted((pin.name, pin.sha256) for pin in target_small_files)
        ),
        draft_config=dict(draft_config),
        draft_weight_map={
            str(name): str(filename) for name, filename in weight_map.items()
        },
        draft_shards=(
            MiaShardPin(
                draft_shard.name,
                MIA_DRAFT_SHARD_BYTES,
                draft_sha256,
                _DRAFT_CANONICAL_SHARD_PIN,
            ),
        ),
        draft_small_files=draft_small_files,
        draft_small_file_sha256=tuple(
            sorted((pin.name, pin.sha256) for pin in draft_small_files)
        ),
    )


def revalidate_pinned_mia_tokenizer_files(
    validation: MiaArtifactValidation,
) -> None:
    """Prove tokenizer inputs kept the validated identity during construction."""

    pins = {pin.name: pin for pin in validation.target_small_files}
    for name in _TOKENIZER_CONSUMED_SMALL_FILES:
        expected = pins.get(name)
        if expected is None:
            raise ValueError(f"pinned Mia tokenizer file was not validated: {name}")
        _payload, observed = _read_pinned_small_file(
            validation.target_root / name,
            expected_sha256=expected.sha256,
            label="Mia tokenizer",
        )
        if observed != expected:
            raise ValueError(
                f"pinned Mia tokenizer file identity changed: {name}"
            )


@dataclass(frozen=True, slots=True)
class MiaPageGeometry:
    layer_id: int
    compress_ratio: int
    compressed_capacity: int
    attention_record_bytes: int
    index_record_bytes: int


@dataclass(frozen=True, slots=True)
class MiaWorkspaceGeometry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    ownership: str


@dataclass(frozen=True, slots=True)
class MiaPrewarmSignature:
    name: str
    rows: int
    phase: str


@dataclass(frozen=True, slots=True)
class MiaM6RatioSlice:
    """One physical-M6 index schedule shared by every layer of one ratio."""

    ratio: int
    start_offset: int
    prior_rows: int
    first_window: int
    emitted_rows: int
    ape_slots: Any
    journal_slots: Any
    compressed_blocks: Any
    compressed_offsets: Any
    causal_lengths: Any
    batch_slots: Any


@dataclass(frozen=True, slots=True)
class MiaM6Cycle:
    """Request-owned physical-M6 schedule with shared ratio slices."""

    start_offset: int
    stop_offset: int
    ratio4: MiaM6RatioSlice
    ratio128: MiaM6RatioSlice
    by_layer: tuple[MiaM6RatioSlice | None, ...]


@dataclass(frozen=True, slots=True)
class MiaM6RatioAcceptance:
    """One ratio's shared direct-commit frontier after an M6 verdict."""

    stop_offset: int
    compressed_rows: int
    journal_length: int
    current_start: int
    current_rows: int
    completed_rows: int
    previous_start: int


@dataclass(frozen=True, slots=True)
class MiaM6Acceptance:
    """Request-owned direct-commit descriptor shared across target layers."""

    accepted_rows: int
    stop_offset: int
    window_start: int
    ratio4: MiaM6RatioAcceptance
    ratio128: MiaM6RatioAcceptance


@dataclass(frozen=True, slots=True)
class MiaM6RatioTables:
    """Construction-evaluated tables for one exact Mia compressor ratio."""

    ratio: int
    rollback_rows: int
    compressed_block_size: int
    ape_slots: Any
    journal_slots: Any
    compressed_blocks: Any
    compressed_offsets: Any
    causal_lengths: Any
    batch_slots: Any

    @classmethod
    def allocate(
        cls,
        *,
        ratio: int,
        rollback_rows: int,
        capacity_tokens: int,
        compressed_capacity: int,
        compressed_block_size: int,
        block_table: Any,
    ) -> "MiaM6RatioTables":
        import mlx.core as mx

        width = MIA_DSPARK_BLOCK + 1
        ratio = int(ratio)
        rollback_rows = int(rollback_rows)
        compressed_capacity = int(compressed_capacity)
        compressed_block_size = int(compressed_block_size)
        ape_slots = mx.arange(ratio + width - 1, dtype=mx.int32) % ratio
        journal_slots = (
            mx.arange(rollback_rows + width - 1, dtype=mx.int32)
            % rollback_rows
        )
        logical_rows = mx.arange(compressed_capacity, dtype=mx.int32)
        logical_blocks = logical_rows // compressed_block_size
        compressed_offsets = (
            logical_rows - logical_blocks * compressed_block_size
        )
        compressed_blocks = block_table[logical_blocks]
        positions = mx.arange(int(capacity_tokens), dtype=mx.int32)
        causal_lengths = (positions + 1) // ratio
        batch_slots = mx.zeros((width,), dtype=mx.int32)
        mx.eval(
            ape_slots,
            journal_slots,
            compressed_blocks,
            compressed_offsets,
            causal_lengths,
            batch_slots,
        )
        return cls(
            ratio=ratio,
            rollback_rows=rollback_rows,
            compressed_block_size=compressed_block_size,
            ape_slots=ape_slots,
            journal_slots=journal_slots,
            compressed_blocks=compressed_blocks,
            compressed_offsets=compressed_offsets,
            causal_lengths=causal_lengths,
            batch_slots=batch_slots,
        )

    def slice(self, start_offset: int) -> MiaM6RatioSlice:
        width = MIA_DSPARK_BLOCK + 1
        start_offset = int(start_offset)
        prior_rows = start_offset % self.ratio
        first_window = start_offset // self.ratio
        emitted_rows = (prior_rows + width) // self.ratio
        ape_start = prior_rows
        journal_start = start_offset % self.rollback_rows
        return MiaM6RatioSlice(
            ratio=self.ratio,
            start_offset=start_offset,
            prior_rows=prior_rows,
            first_window=first_window,
            emitted_rows=emitted_rows,
            ape_slots=self.ape_slots[ape_start : ape_start + width],
            journal_slots=self.journal_slots[
                journal_start : journal_start + width
            ],
            compressed_blocks=self.compressed_blocks[
                first_window : first_window + emitted_rows
            ],
            compressed_offsets=self.compressed_offsets[
                first_window : first_window + emitted_rows
            ],
            causal_lengths=self.causal_lengths[
                start_offset : start_offset + width
            ],
            batch_slots=self.batch_slots,
        )


class MiaM6CacheSchedule:
    """One active request's physical-M6 schedule owner."""

    def __init__(
        self,
        *,
        authoritative_cache: Any,
        ratio4: MiaM6RatioTables,
        ratio128: MiaM6RatioTables,
    ) -> None:
        self._authoritative_cache = authoritative_cache
        self._ratio4 = ratio4
        self._ratio128 = ratio128
        self.current_cycle: MiaM6Cycle | None = None

    def begin(self) -> MiaM6Cycle:
        # Request admission reserves one M6-minus-one headroom block beyond the
        # 384K logical limit, so the authoritative first cache always owns a
        # complete six-row slice. The sealed target advances all 43 caches in
        # lockstep; repeating either proof here would put validation in the hot path.
        start_offset = int(self._authoritative_cache.offset)
        ratio4 = self._ratio4.slice(start_offset)
        ratio128 = self._ratio128.slice(start_offset)
        by_ratio = {4: ratio4, 128: ratio128}
        cycle = MiaM6Cycle(
            start_offset=int(start_offset),
            stop_offset=int(start_offset) + MIA_DSPARK_BLOCK + 1,
            ratio4=ratio4,
            ratio128=ratio128,
            by_layer=tuple(
                None if ratio == 0 else by_ratio[ratio]
                for ratio in _MIA_TARGET_COMPRESS_RATIOS
            ),
        )
        self.current_cycle = cycle
        return cycle

    def reset(self) -> None:
        self.current_cycle = None

    @staticmethod
    def _ratio_acceptance(
        schedule: MiaM6RatioSlice,
        *,
        accepted_rows: int,
        rollback_rows: int,
    ) -> MiaM6RatioAcceptance:
        total = schedule.prior_rows + int(accepted_rows)
        completed = total // schedule.ratio
        current_start = completed * schedule.ratio
        stop_offset = schedule.start_offset + int(accepted_rows)
        return MiaM6RatioAcceptance(
            stop_offset=stop_offset,
            compressed_rows=schedule.first_window + completed,
            journal_length=min(int(rollback_rows), stop_offset),
            current_start=current_start,
            current_rows=total - current_start,
            completed_rows=completed,
            previous_start=(completed - 1) * schedule.ratio,
        )

    def acceptance(
        self,
        accepted_rows: int,
        *,
        window_keep: int,
    ) -> MiaM6Acceptance:
        """Build the two shared ratio descriptors for one qualified verdict."""
        cycle = self.current_cycle
        stop_offset = cycle.start_offset + int(accepted_rows)
        return MiaM6Acceptance(
            accepted_rows=int(accepted_rows),
            stop_offset=stop_offset,
            window_start=max(0, stop_offset - int(window_keep)),
            ratio4=self._ratio_acceptance(
                cycle.ratio4,
                accepted_rows=accepted_rows,
                rollback_rows=self._ratio4.rollback_rows,
            ),
            ratio128=self._ratio_acceptance(
                cycle.ratio128,
                accepted_rows=accepted_rows,
                rollback_rows=self._ratio128.rollback_rows,
            ),
        )


def _release_prewarm_leases(
    plan: "MiaDeepseekV4EnginePlan",
    model: Any,
    target_cache: list[Any],
    draft_cache: list[Any] | None,
    primary_error: BaseException | None,
) -> None:
    """Release every acquired prewarm lease while preserving body failures."""

    release_errors: list[BaseException] = []
    if draft_cache is not None:
        try:
            model.dspark.release_mia_cache(draft_cache)
        except BaseException as exc:
            release_errors.append(exc)
    try:
        plan.release_target_cache(target_cache)
    except BaseException as exc:
        release_errors.append(exc)

    if not release_errors:
        return
    if primary_error is not None:
        for error in release_errors:
            primary_error.add_note(
                "prewarm cache release also failed: "
                f"{type(error).__name__}: {error}"
            )
        return
    first, *remaining = release_errors
    for error in remaining:
        first.add_note(
            "additional prewarm cache release failed: "
            f"{type(error).__name__}: {error}"
        )
    raise first


class MiaTargetCacheArena:
    """One persistent vLLM-style page lease for the single-sequence model."""

    def __init__(
        self,
        layers: tuple[Any, ...],
        *,
        capacity_tokens: int,
        max_batch_tokens: int,
    ) -> None:
        import mlx.core as mx

        from mtplx.deepseek_v4_nvfp4_kv import FixedMiaNVFP4WindowRecords
        from mtplx.models.deepseek_v4 import DeepseekV4NVFP4Cache

        self._layer_identity = tuple(id(layer) for layer in layers)
        self._caches = tuple(
            DeepseekV4NVFP4Cache(
                window_size=layer.attn.window_size,
                compress_ratio=layer.attn.compress_ratio,
                head_dim=layer.attn.head_dim,
                capacity_tokens=int(capacity_tokens),
                max_batch_tokens=int(max_batch_tokens),
            )
            for layer in layers
        )
        expected_window_capacity = (
            int(max_batch_tokens)
            + int(layers[0].attn.window_size)
            + int(self._caches[0].rollback_capacity)
        )
        if any(
            getattr(cache.window, "mode", None)
            != "nvfp4_stock432_fixed_window"
            or int(getattr(cache.window, "capacity", 0))
            != expected_window_capacity
            for cache in self._caches
        ):
            raise ValueError("Mia target cache arena did not install fixed SWA pages")
        if any(
            not isinstance(
                getattr(cache.window, "_paged_records", None),
                FixedMiaNVFP4WindowRecords,
            )
            or cache.window._paged_records.pages is not cache.window._pages
            or cache.window._paged_records.block_table
            is not cache.window._pool.block_table
            or int(cache.window._paged_records.capacity)
            != expected_window_capacity
            or int(cache.window._paged_records.block_size) != 64
            or int(cache.window._paged_records.physical_rows)
            != ((expected_window_capacity + 63) // 64) * 64
            for cache in self._caches
        ):
            raise ValueError(
                "Mia target cache arena lost its fixed-window paged descriptor"
            )
        if any(
            getattr(getattr(cache, "_write_window_records", None), "keywords", {}).get(
                "owner"
            )
            is not cache.window
            or getattr(cache, "_pack_window_records", object()) is not None
            or _callable_name(getattr(cache, "_update_window_impl", None))
            != "_fixed_window_requires_records"
            or _callable_name(getattr(cache, "_trim_window_impl", None))
            != "_trim_fixed_window"
            for cache in self._caches
        ):
            raise ValueError("Mia target record writers lost their cache owners")
        prefill_settlement_roots = [cache.window._pages for cache in self._caches]
        prefill_frontier_lanes = []
        for layer, cache in zip(layers, self._caches, strict=True):
            ratio = int(layer.attn.compress_ratio)
            if ratio == 0:
                continue
            expected_rows = (2 if ratio == 4 else 1) * ratio + int(
                cache.rollback_capacity
            )
            expected_width = (2 if ratio == 4 else 1) * int(cache.head_dim)
            if (
                getattr(cache.comp, "mode", None) != "mia_fixed_compressor_state"
                or int(getattr(cache.comp, "rollback_rows", 0)) != expected_rows
                or int(getattr(cache.comp, "state_width", 0)) != expected_width
                or getattr(cache.compressed, "mode", None)
                != "nvfp4_stock432_paged"
                or int(getattr(cache.compressed, "capacity", 0))
                != (int(capacity_tokens) + ratio - 1) // ratio
                or int(getattr(cache.compressed, "block_size", 0))
                != max(1, 256 // ratio)
            ):
                raise ValueError(
                    "Mia target cache arena did not install fixed compressor pages"
                )
            prefill_settlement_roots.append(cache.compressed.pages)
            prefill_settlement_roots.extend(cache.comp.journal_buffers)
            prefill_frontier_lanes.append(cache.comp)
            if ratio == 4:
                if (
                    getattr(cache.index_comp, "mode", None)
                    != "mia_fixed_compressor_state"
                    or int(getattr(cache.index_comp, "rollback_rows", 0))
                    != expected_rows
                    or int(getattr(cache.index_comp, "state_width", 0)) != 256
                    or getattr(cache.index_compressed, "mode", None)
                    != "fp8_e4m3_ue8m0_scale132_paged"
                    or int(getattr(cache.index_compressed, "capacity", 0))
                    != (int(capacity_tokens) + ratio - 1) // ratio
                ):
                    raise ValueError(
                        "Mia target cache arena did not install fixed indexer state"
                    )
                prefill_settlement_roots.append(cache.index_compressed.pages)
                prefill_settlement_roots.extend(cache.index_comp.journal_buffers)
                prefill_frontier_lanes.append(cache.index_comp)
        self._prefill_settlement_roots = tuple(prefill_settlement_roots)
        self._prefill_frontier_lanes = tuple(prefill_frontier_lanes)
        ratio4_cache = next(
            cache
            for layer, cache in zip(layers, self._caches, strict=True)
            if int(layer.attn.compress_ratio) == 4
        )
        ratio128_cache = next(
            cache
            for layer, cache in zip(layers, self._caches, strict=True)
            if int(layer.attn.compress_ratio) == 128
        )
        self._m6_schedule = MiaM6CacheSchedule(
            authoritative_cache=self._caches[0],
            ratio4=MiaM6RatioTables.allocate(
                ratio=4,
                rollback_rows=int(ratio4_cache.comp.rollback_rows),
                capacity_tokens=int(capacity_tokens),
                compressed_capacity=int(ratio4_cache.compressed.capacity),
                compressed_block_size=int(ratio4_cache.compressed.block_size),
                block_table=ratio4_cache.compressed.block_table,
            ),
            ratio128=MiaM6RatioTables.allocate(
                ratio=128,
                rollback_rows=int(ratio128_cache.comp.rollback_rows),
                capacity_tokens=int(capacity_tokens),
                compressed_capacity=int(ratio128_cache.compressed.capacity),
                compressed_block_size=int(ratio128_cache.compressed.block_size),
                block_table=ratio128_cache.compressed.block_table,
            ),
        )
        for layer, cache in zip(layers, self._caches, strict=True):
            ratio = int(layer.attn.compress_ratio)
            cache._mia_m6_schedule = self._m6_schedule
            if ratio == 4:
                cache.comp._mia_m6_tables = self._m6_schedule._ratio4
                cache.index_comp._mia_m6_tables = self._m6_schedule._ratio4
            elif ratio == 128:
                cache.comp._mia_m6_tables = self._m6_schedule._ratio128
        commit_routes = {
            0: ("_commit_m6_ratio0", 0),
            4: ("_commit_m6_ratio4", 1),
            128: ("_commit_m6_ratio128", 2),
        }
        self._m6_committers = tuple(
            (getattr(cache, commit_routes[ratio][0]), commit_routes[ratio][1])
            for layer, cache in zip(layers, self._caches, strict=True)
            for ratio in (int(layer.attn.compress_ratio),)
        )
        self._m6_window_keep = int(self._caches[0].window_size) + int(
            self._caches[0].rollback_capacity
        )
        self._eval_prefill_roots = mx.eval
        self._schedule_verify_roots = mx.async_eval
        mx.eval(*self._prefill_settlement_roots)
        self._leased = False

    @property
    def layer_count(self) -> int:
        return len(self._caches)

    @property
    def leased(self) -> bool:
        return self._leased

    @property
    def current_m6_cycle(self) -> MiaM6Cycle | None:
        return self._m6_schedule.current_cycle

    def begin_verify(self) -> MiaM6Cycle:
        return self._m6_schedule.begin()

    def commit_verify(self, accepted_rows: int) -> MiaM6Acceptance:
        acceptance = self._m6_schedule.acceptance(
            accepted_rows,
            window_keep=self._m6_window_keep,
        )
        descriptors = (acceptance, acceptance.ratio4, acceptance.ratio128)
        for commit, descriptor_index in self._m6_committers:
            commit(acceptance, descriptors[descriptor_index])
        self._m6_schedule.reset()
        return acceptance

    def _reset(self) -> None:
        self._m6_schedule.reset()
        for cache in self._caches:
            cache.state = None

    def settle_prefill_chunk(self, *outputs: Any) -> None:
        """Complete one fixed-linear target chunk without gathering cache rows."""

        live_frontiers = tuple(
            value
            for lane in self._prefill_frontier_lanes
            for value in (
                lane.cur_kv,
                lane.cur_score,
                lane.prev_kv,
                lane.prev_score,
            )
            if value is not None
        )
        self._eval_prefill_roots(
            *outputs,
            *self._prefill_settlement_roots,
            *live_frontiers,
        )

    def schedule_verify_chunk(self, *outputs: Any) -> None:
        """Schedule one fixed-linear verify cycle with every retained owner."""

        live_frontiers = tuple(
            value
            for lane in self._prefill_frontier_lanes
            for value in (
                lane.cur_kv,
                lane.cur_score,
                lane.prev_kv,
                lane.prev_score,
            )
            if value is not None
        )
        pending_frontiers = tuple(
            value
            for lane in self._prefill_frontier_lanes
            for value in (
                lane._pending_m6.combined_kv,
                lane._pending_m6.combined_score,
            )
        )
        self._schedule_verify_roots(
            *outputs,
            *self._prefill_settlement_roots,
            *live_frontiers,
            *pending_frontiers,
        )

    def acquire(self, layers: tuple[Any, ...]) -> list[Any]:
        if tuple(id(layer) for layer in layers) != self._layer_identity:
            raise ValueError("Mia target cache arena belongs to a different model")
        if self._leased:
            raise RuntimeError("Mia target cache arena already owns the active request")
        self._reset()
        self._leased = True
        return list(self._caches)

    def release(self, caches: list[Any]) -> None:
        if not self._leased:
            raise RuntimeError("Mia target cache arena has no active request")
        if len(caches) != len(self._caches) or any(
            observed is not expected
            for observed, expected in zip(caches, self._caches, strict=True)
        ):
            raise ValueError("Mia target cache release does not match its page lease")
        self._reset()
        self._leased = False

    def release_active(self) -> None:
        """Release the request-owned lease, if one was acquired."""

        if self._leased:
            self.release(list(self._caches))


@dataclass(frozen=True, slots=True)
class MiaDeepseekV4EnginePlan:
    """Immutable plan installed after all exact callables and weights exist."""

    context_capacity_tokens: int
    target_physical_capacity_tokens: int
    max_batch_tokens: int
    max_sequences: int
    page_geometry: tuple[MiaPageGeometry, ...]
    workspace_geometry: tuple[MiaWorkspaceGeometry, ...]
    indexer_workspace: Any
    indexer_rope_table: Any
    mla_workspace: Any
    target_cache_arena: MiaTargetCacheArena
    prewarm_signatures: tuple[MiaPrewarmSignature, ...]
    installed_routes: tuple[str, ...]
    target_artifact: str
    draft_artifact: str
    artifact_small_file_sha256: tuple[tuple[str, str], ...]
    identity: str

    def make_target_cache(self, layers) -> list[Any]:
        return self.target_cache_arena.acquire(tuple(layers))

    def release_target_cache(self, caches: list[Any]) -> None:
        self.target_cache_arena.release(caches)

    def settle_target_prefill_chunk(self, *outputs: Any) -> None:
        self.target_cache_arena.settle_prefill_chunk(*outputs)

    def schedule_target_verify_chunk(self, *outputs: Any) -> None:
        self.target_cache_arena.schedule_verify_chunk(*outputs)

    def begin_target_verify(self, cache: list[Any]) -> MiaM6Cycle:
        del cache
        return self.target_cache_arena.begin_verify()

    def commit_target_verify(
        self,
        cache: list[Any],
        target_len: int,
    ) -> MiaM6Acceptance:
        del cache
        cycle = self.target_cache_arena.current_m6_cycle
        accepted_rows = int(target_len) - int(cycle.start_offset)
        return self.target_cache_arena.commit_verify(accepted_rows)

    @contextmanager
    def target_cache_lifecycle(self):
        """Own and close one target-only request's persistent cache lease."""

        if self.target_cache_arena.leased:
            raise RuntimeError(
                "Mia target cache arena already owns the active request"
            )
        primary_error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self.target_cache_arena.release_active()
            except BaseException as release_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "target cache release also failed: "
                    f"{type(release_error).__name__}: {release_error}"
                )

    def prewarm(self, model) -> dict[str, Any]:
        """Compile every serving phase with bounded, disposable cache state.

        A 128-row prefill reaches both ratio-4 and ratio-128 compressors and the
        large-M Trellis plan.  A direct M6 layer call compiles the prefill BM8
        Trellis plan, while one physical M6 target verify reaches the direct-QMV
        plan.  The packaged K5 proposal reaches all three K64 draft stages and
        the sequential Markov head.  Custom Metal kernels carry their row counts
        as runtime arguments, so these dispatches compile the finite pipeline set
        without a model-length-sized warmup.
        """

        import mlx.core as mx

        from mtplx.attention_context import attention_phase
        from mtplx.deepseek_v4_paged_indexer import (
            INDEXER_RECORD_BYTES,
            MiaIndexerQueryRecords,
            PagedMiaIndexerRecords,
        )

        target_cache = self.make_target_cache(model.layers)
        draft_cache = None
        primary_error = None
        try:
            prefill_ids = mx.zeros((1, MIA_WINDOW), dtype=mx.uint32)
            with attention_phase("prefill"):
                prefill_logits, target_taps = model(
                    prefill_ids,
                    cache=target_cache,
                    return_hidden=True,
                    logits_keep=1,
                )
            self.settle_target_prefill_chunk(prefill_logits, *target_taps)

            # SparkInfer supplies the BF16 projection only for the FFN
            # connection. Warm that installed large-M component directly so
            # startup covers its signature without a second target-model pass.
            mhc_rows = 384
            first_layer = model.model.layers[0]
            prefill_m6_rows = 6
            with attention_phase("prefill"):
                prefill_m6 = first_layer.ffn(
                    mx.zeros(
                        (1, prefill_m6_rows, MIA_HIDDEN), dtype=mx.bfloat16
                    ),
                    mx.zeros((1, prefill_m6_rows), dtype=mx.uint32),
                )
            mx.eval(prefill_m6)
            mhc_residual, mhc_post, mhc_comb, mhc_y = (
                model.model._mia_mhc.post_pre_ffn(
                    mx.zeros((mhc_rows, MIA_HIDDEN), dtype=mx.bfloat16),
                    mx.zeros(
                        (mhc_rows, MIA_HC, MIA_HIDDEN), dtype=mx.bfloat16
                    ),
                    mx.zeros((mhc_rows, MIA_HC), dtype=mx.float32),
                    mx.zeros((mhc_rows, MIA_HC, MIA_HC), dtype=mx.float32),
                    first_layer.ffn_hc,
                    first_layer.ffn_norm,
                )
            )
            mx.eval(mhc_residual, mhc_post, mhc_comb, mhc_y)

            # Spark's exact B16 WO route has a distinct WO-A epilogue that writes
            # E4M3/UE8M0 output directly for WO-B. Compile that construction-bound
            # plan explicitly; the M128 prefill and M6 verify passes cannot reach
            # this finite logical-M route.
            wo_m16_rows = 16
            _positions, wo_cos, wo_sin = (
                model._mia_base_rope_provider.token_tables(0, wo_m16_rows)
            )
            wo_m16 = model.layers[0].attn._output_projection_impl(
                mx.zeros(
                    (1, wo_m16_rows, MIA_HEADS, MIA_HEAD_DIM),
                    dtype=mx.bfloat16,
                ),
                wo_cos,
                wo_sin,
            )
            mx.eval(wo_m16)

            # The pinned CUDA prologue switches to its reduced one-CTA-per-row
            # topology only at M1024.  Real DFlash prefill emits that exact chunk,
            # while the bounded M128 model warmup above reaches the full grid.
            qkv_prefill_rows = MIA_LONG_PREFILL_CHUNK
            _positions, qkv_cos, qkv_sin = (
                model._mia_base_rope_provider.token_tables(0, qkv_prefill_rows)
            )
            qkv_query, qkv_records = first_layer.attn._mia_qkv_plan.prefill_records(
                mx.zeros(
                    (1, qkv_prefill_rows, MIA_HEADS, MIA_HEAD_DIM),
                    dtype=mx.bfloat16,
                ),
                mx.zeros(
                    (1, qkv_prefill_rows, MIA_HEAD_DIM),
                    dtype=mx.bfloat16,
                ),
                qkv_cos,
                qkv_sin,
            )
            mx.eval(qkv_query, qkv_records)

            # The real sparse selector starts only after 512 ratio-4 rows. Compile
            # its prefill and decode engines against a tiny synthetic paged view so
            # startup does not need a 2K-token model forward merely to reach that
            # phase boundary.
            selector_rows = MIA_INDEX_TOPK + 1
            selector_block = 64
            selector_blocks = (
                selector_rows + selector_block - 1
            ) // selector_block
            selector_view = PagedMiaIndexerRecords(
                records=mx.zeros(
                    (selector_blocks, selector_block, INDEXER_RECORD_BYTES),
                    dtype=mx.uint8,
                ),
                block_table=mx.arange(selector_blocks, dtype=mx.int32),
                length=selector_rows,
                block_size=selector_block,
            )
            selector_q = MiaIndexerQueryRecords(
                mx.zeros(
                    (1, 1, MIA_HEADS, INDEXER_RECORD_BYTES), dtype=mx.uint8
                )
            )
            selector_weights = mx.zeros((1, 1, MIA_HEADS), dtype=mx.float32)
            selector_positions = mx.array(
                [selector_rows * 4 - 1], dtype=mx.int32
            )
            selector = model.layers[2].attn.indexer
            with attention_phase("prefill"):
                prefill_selection = selector._select_rows(
                    selector_q,
                    selector_weights,
                    selector_positions,
                    selector_view,
                )
            with attention_phase("decode_verify"):
                decode_selection = selector._select_rows(
                    selector_q,
                    selector_weights,
                    selector_positions,
                    selector_view,
                )
            mx.eval(
                prefill_selection.indices,
                prefill_selection.lengths,
                decode_selection.indices,
                decode_selection.lengths,
            )

            draft_cache = model.make_dspark_cache()
            model.prefill_dspark(target_taps, draft_cache)
            mx.eval(*(cache.ring.records for cache in draft_cache))
            primary = mx.argmax(prefill_logits[:, -1], axis=-1).astype(mx.uint32)
            proposal = model.propose_dspark_k5(
                primary,
                draft_cache,
                start_pos=MIA_WINDOW,
            )
            mx.eval(proposal.future_tokens, proposal.neural_logits)

            verify_ids = mx.concatenate(
                [primary[:, None], proposal.future_tokens], axis=1
            )
            self.begin_target_verify(target_cache)
            with attention_phase("decode_verify"):
                verify_logits, verify_taps = model.mia_dflash_forward(
                    verify_ids,
                    target_cache,
                    logits_last_only=False,
                )
            self.settle_target_prefill_chunk(verify_logits, *verify_taps)
            return {
                "identity": self.identity,
                "signatures": tuple(
                    signature.name for signature in self.prewarm_signatures
                ),
                "prefill_rows": MIA_WINDOW,
                "prefill_m6_rows": prefill_m6_rows,
                "mhc_prefill_rows": mhc_rows,
                "wo_m16_rows": wo_m16_rows,
                "qkv_reduced_prefill_rows": qkv_prefill_rows,
                "verify_rows": MIA_DSPARK_BLOCK + 1,
                "draft_rows": MIA_DSPARK_BLOCK,
            }
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            _release_prewarm_leases(
                self,
                model,
                target_cache,
                draft_cache,
                primary_error,
            )


def _install_shared_indexer_resources(
    layers: tuple[Any, ...],
    ratios: tuple[int, ...],
    workspace: Any,
    inv_freq: Any,
    *,
    max_positions: int,
) -> Any:
    """Build one ratio-4 RoPE table and bind it to every indexer owner."""

    indexers = tuple(
        layer.attn.indexer
        for layer, ratio in zip(layers, ratios, strict=True)
        if ratio == 4
    )
    if not indexers:
        raise ValueError("the Mia engine requires ratio-4 indexer owners")
    rope_table = precompute_indexer_rope_table(
        inv_freq,
        max_positions=int(max_positions),
    )
    for indexer in indexers:
        indexer.install_mia_paged_topk(workspace, rope_table)
    return rope_table


def _artifact_small_file_sha256(
    target: dict[str, str],
    draft: dict[str, str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                *(
                    (f"target/{name}", digest)
                    for name, digest in target.items()
                ),
                *(
                    (f"draft/{name}", digest)
                    for name, digest in draft.items()
                ),
            )
        )
    )


def _matches_small_file_receipt(
    observed: Any,
    fixed: dict[str, str],
    semantic_name: str,
) -> bool:
    if not isinstance(observed, dict):
        return False
    return (
        set(observed) == {*fixed, semantic_name}
        and all(observed.get(name) == digest for name, digest in fixed.items())
        and len(str(observed.get(semantic_name, ""))) == 64
    )


def _mia_engine_identity(
    context_capacity_tokens: int,
    max_batch_tokens: int,
) -> str:
    identity_source = "|".join(
        (
            *(
                f"{name}:{digest}"
                for name, digest in sorted(
                    (
                        *(
                            (f"target/{name}", digest)
                            for name, digest in _TARGET_SMALL_FILE_PINS.items()
                        ),
                        *(
                            (f"draft/{name}", digest)
                            for name, digest in _DRAFT_SMALL_FILE_PINS.items()
                        ),
                    )
                )
            ),
            f"target/rank-sliced-tp1-manifest.json#canonical:"
            f"{_TARGET_CANONICAL_ARTIFACT_SEAL}",
            f"draft/DSPARK_DRAFT_PLAN.json#canonical:"
            f"{_DRAFT_CANONICAL_PLAN_SEAL}",
            *(
                f"target-shard/{name}:{digest}"
                for name, digest in sorted(
                    _TARGET_CANONICAL_SHARD_PINS.items()
                )
            ),
            f"draft-shard/dspark-draft.safetensors:{_DRAFT_CANONICAL_SHARD_PIN}",
            str(context_capacity_tokens),
            str(max_batch_tokens),
            "stock432",
            "mia132",
            "k5-k64-packed-gate-up-native-mxfp4",
            "bounded-one-shard-raw-canonical-sha256-same-fd-loader",
            "mhc-role-attn-fp32-ffn-bf16-m384-bm64",
            "target-physical-m6-piecewise-compile-eager-attention",
            "draft-physical-k5-fullgraph-compile-live-cache-pages",
            "wo-tp1-b12x-inv-rope-mxfp8-bm8-m16q-bm64",
            "long-prefill-chunk1024",
            "target-exl3-prefill-trellis-bm8-bm64-verify-m6-direct-5stage-staged-fc1-clamp10-bf16tail-u4-stage16b-96x8",
            MIA_EXL3_M6_QUAD_DESCRIPTOR_SHA256,
            "compressor-absolute-state-rings",
            "fixed-target-window-m8224",
            "persistent-target-draft-page-arenas",
            MIA_DFLASH_COMMIT,
        )
    )
    return "mia-dsv4-" + hashlib.sha256(
        identity_source.encode("utf-8")
    ).hexdigest()[:16]


def build_mia_engine_plan(
    model,
    *,
    target_root: Path,
    draft_root: Path,
    context_capacity_tokens: int = MIA_CONTEXT_CAPACITY,
    max_batch_tokens: int = MIA_MAX_BATCH_TOKENS,
) -> MiaDeepseekV4EnginePlan:
    """Validate the installed graph once and seal its exact serving geometry."""

    context_capacity_tokens = int(context_capacity_tokens)
    max_batch_tokens = int(max_batch_tokens)
    if (
        context_capacity_tokens != MIA_CONTEXT_CAPACITY
        or max_batch_tokens != MIA_MAX_BATCH_TOKENS
    ):
        raise ValueError(
            "the pinned Mia launcher requires context=384000 and "
            "max_num_batched_tokens=8224"
        )

    args = model.args
    layers = tuple(model.layers)
    stages = tuple(getattr(model.dspark, "stages", ()))
    ratios = _require_mia_target_ratio_order(
        tuple(int(layer.attn.compress_ratio) for layer in layers)
    )
    from mtplx.deepseek_v4_paged_indexer import MiaIndexerWorkspace
    from mtplx.kernels.deepseek_v4_nvfp4_mla import mia_mla_workspace
    import mlx.core as mx
    from mlx.utils import tree_flatten

    observed = (
        len(layers),
        len(stages),
        int(args.hidden_size),
        int(args.hc_mult),
        int(args.num_attention_heads),
        int(args.head_dim),
        int(args.qk_rope_head_dim),
        int(args.window_size),
        int(args.index_topk),
        int(args.n_routed_experts),
        int(args.num_experts_per_tok),
        int(model.dspark.args.n_routed_experts),
        int(model.dspark.block_size),
        tuple(int(value) for value in model.dspark.target_layer_ids),
        ratios.count(0),
        ratios.count(4),
        ratios.count(128),
    )
    expected = (
        MIA_TARGET_LAYERS,
        MIA_DSPARK_STAGES,
        MIA_HIDDEN,
        MIA_HC,
        MIA_HEADS,
        MIA_HEAD_DIM,
        MIA_ROPE_DIM,
        MIA_WINDOW,
        MIA_INDEX_TOPK,
        MIA_TARGET_EXPERTS,
        MIA_TOPK,
        MIA_DRAFT_EXPERTS,
        MIA_DSPARK_BLOCK,
        MIA_TARGET_TAPS,
        2,
        21,
        20,
    )
    if observed != expected:
        raise ValueError(
            f"installed Mia engine topology changed: {observed!r} != {expected!r}"
        )

    from mtplx.deepseek_v4_exl3 import (
        EXL3_M6_QUAD_DESCRIPTOR_SHA256,
        EXL3_M6_STAGE_VECTOR_BYTES,
        EXL3_M6_STAGE_VECTORS_PER_K_TILE,
        EXL3SwitchGLU,
        _InstalledM6QuadQMVPlan,
        _m6_clamp10_activation_down_kernel,
        _m6_direct_final_tail_kernel,
        _m6_down_inner_kernel,
        _m6_dual_fc1_input_kernel,
        _m6_dual_fc1_inner_kernel,
    )
    from mtplx.models.deepseek_v4 import mia_tp1_wo_projection_receipt

    if EXL3_M6_QUAD_DESCRIPTOR_SHA256 != MIA_EXL3_M6_QUAD_DESCRIPTOR_SHA256:
        raise ValueError("the Mia quad MCG descriptor seal changed")

    wo_projection_receipt = mia_tp1_wo_projection_receipt(model)
    if (
        getattr(model, "_mia_wo_projection_receipt", None)
        != wo_projection_receipt
        or wo_projection_receipt["route"] != "mia_tp1_b12x_wo_mxfp8"
        or wo_projection_receipt["target_attention"] != MIA_TARGET_LAYERS
        or wo_projection_receipt["draft_attention"] != MIA_DSPARK_STAGES
        or wo_projection_receipt["plan_count"]
        != MIA_TARGET_LAYERS + MIA_DSPARK_STAGES
        or wo_projection_receipt["unique_plan_count"]
        != MIA_TARGET_LAYERS + MIA_DSPARK_STAGES
        or wo_projection_receipt["plan_type"] != "MiaTP1WOMXFP8Plan"
        or wo_projection_receipt["max_prefill_rows"] != max_batch_tokens
    ):
        raise ValueError("the Mia TP1 B12X WO projection receipt changed")

    target_physical_capacity_tokens = (
        context_capacity_tokens + MIA_DSPARK_BLOCK
    )
    ratio4_capacity = (target_physical_capacity_tokens + 3) // 4
    indexer_workspace = MiaIndexerWorkspace.allocate(
        max_query_rows=max_batch_tokens,
        topk=MIA_INDEX_TOPK,
        sentinel=ratio4_capacity,
    )
    from mtplx.models.deepseek_v4 import install_mia_target_rope_providers

    base_rope_provider, compress_rope_provider = (
        install_mia_target_rope_providers(
            model,
            physical_max_positions=target_physical_capacity_tokens,
        )
    )
    draft_rope_provider = model._mia_draft_rope_provider
    if (
        int(getattr(base_rope_provider, "max_positions", 0))
        != target_physical_capacity_tokens
        or int(getattr(compress_rope_provider, "max_positions", 0))
        != target_physical_capacity_tokens
        or int(getattr(draft_rope_provider, "max_positions", 0))
        != target_physical_capacity_tokens
    ):
        raise ValueError("the Mia physical M6 RoPE capacity changed")
    draft_fullgraph_route = model.dspark.install_mia_fullgraph_runtime(
        embed_tokens=model.model.embed_tokens,
        lm_head=model.lm_head,
    )
    from mtplx.deepseek_v4_mia_draft_graph import (
        MiaPhysicalK5FullGraphDraftRoute,
    )

    draft_propose_impl = getattr(model.dspark, "_propose_impl", None)
    if (
        getattr(model.dspark, "_mia_fullgraph_route", None)
        is not draft_fullgraph_route
        or not isinstance(
            draft_fullgraph_route,
            MiaPhysicalK5FullGraphDraftRoute,
        )
        or int(getattr(draft_fullgraph_route, "physical_width", 0))
        != MIA_DSPARK_BLOCK
        or not callable(getattr(draft_fullgraph_route, "_compiled", None))
        or getattr(draft_propose_impl, "__self__", None) is not model.dspark
        or getattr(draft_propose_impl, "__func__", None)
        is not type(model.dspark)._mia_fullgraph_propose_k5
    ):
        raise ValueError("the Mia physical-K5 full draft graph was not installed")
    stacked_projection_receipt = getattr(
        model,
        "_mia_stacked_projection_receipt",
        None,
    )
    if stacked_projection_receipt != {
        "target_attention": 43,
        "draft_attention": 3,
        "shared_expert": 46,
        "main_compressor": 41,
        "indexer_compressor": 21,
    }:
        raise ValueError("the Mia stacked projection owner receipt changed")
    from mtplx.models.deepseek_v4 import mia_qkv_prologue_receipt

    qkv_receipt = mia_qkv_prologue_receipt(model)
    if (
        getattr(model, "_mia_qkv_prologue_receipt", None) != qkv_receipt
        or qkv_receipt["route"] != "mia_fused_qkv_stock432"
        or qkv_receipt["target_attention"] != MIA_TARGET_LAYERS
        or qkv_receipt["draft_attention"] != MIA_DSPARK_STAGES
        or qkv_receipt["plan_count"] != MIA_TARGET_LAYERS + MIA_DSPARK_STAGES
        or qkv_receipt["unique_plan_count"]
        != MIA_TARGET_LAYERS + MIA_DSPARK_STAGES
        or qkv_receipt["plan_type"] != "MiaBoundQKVPrologue"
        or qkv_receipt["prefill_cutoff"] != MIA_LONG_PREFILL_CHUNK
        or qkv_receipt["proposal_rows"] != MIA_DSPARK_BLOCK
        or qkv_receipt["context_rows"] != MIA_WINDOW
    ):
        raise ValueError("the Mia fused Q/KV prologue receipt changed")
    indexer_rope_table = _install_shared_indexer_resources(
        layers,
        ratios,
        indexer_workspace,
        compress_rope_provider.inv_freq,
        max_positions=target_physical_capacity_tokens,
    )
    mla_workspace = mia_mla_workspace()

    load_receipt = dict(getattr(model, "_mia_target_load_receipt", {}))
    target_small_file_sha256 = load_receipt.get("small_file_sha256")
    target_parameter_count = sum(
        not name.startswith("mtp.")
        for name, _value in tree_flatten(model.parameters())
    )
    if (
        load_receipt.get("mode") != "bounded_one_shard"
        or load_receipt.get("artifact_identity")
        != "raw_canonical_sha256_same_fd"
        or not _matches_small_file_receipt(
            target_small_file_sha256,
            _TARGET_SMALL_FILE_PINS,
            "rank-sliced-tp1-manifest.json",
        )
        or int(load_receipt.get("source_shards", 0)) != 48
        or int(load_receipt.get("carried_shards", 0)) != 5
        or int(load_receipt.get("exl3_layer_shards", 0)) != MIA_TARGET_LAYERS
        or int(load_receipt.get("installed_parameters", 0))
        != target_parameter_count
    ):
        raise ValueError("the Mia target was not installed by the bounded shard loader")
    draft_load_receipt = dict(getattr(model, "_mia_draft_load_receipt", {}))
    draft_small_file_sha256 = draft_load_receipt.get("small_file_sha256")
    if (
        draft_load_receipt.get("mode") != "single_shard"
        or draft_load_receipt.get("artifact_identity")
        != "raw_canonical_sha256_same_fd"
        or int(draft_load_receipt.get("source_shards", 0)) != 1
        or int(draft_load_receipt.get("source_tensors", 0)) != 1_249
        or not _matches_small_file_receipt(
            draft_small_file_sha256,
            _DRAFT_SMALL_FILE_PINS,
            "DSPARK_DRAFT_PLAN.json",
        )
    ):
        raise ValueError("the Mia draft was not installed from its pinned file")

    # These flags are installed at construction and never checked again by a
    # token path.  They certify that no generic arithmetic owner remains bound.
    router_contract_prefix = (
        "bf16xbf16_fp32_k4096",
        "softplus_threshold20_sqrt",
    )
    for layer_id, layer in enumerate(layers):
        ratio = int(layer.attn.compress_ratio)
        expected_gate = (
            "_mia_hash_route"
            if layer_id < int(args.num_hash_layers)
            else "_mia_score_route"
        )
        (
            expected_forward,
            expected_cached_attention,
            expected_compressed,
            expected_sparse,
            expected_prefill,
        ) = _MIA_ATTENTION_ROUTE_CONTRACTS[ratio]
        m6_fused = getattr(layer.ffn, "_mia_exl3_m6_fused", None)
        trellis_fused = getattr(layer.ffn, "_mia_exl3_trellis_fused", None)
        quad_plan = getattr(layer.ffn.switch_mlp, "_m6_quad_qmv_plan", None)
        shared_projection = getattr(
            layer.ffn.shared_experts,
            "_mia_stacked_projection",
            None,
        )
        installed = (
            bool(getattr(layer.ffn.switch_mlp, "_trellis_installed", False)),
            _callable_name(getattr(layer.ffn, "_forward_impl", None)),
            _callable_name(m6_fused),
            getattr(m6_fused, "__self__", None) is layer.ffn.switch_mlp,
            getattr(m6_fused, "__func__", None)
            is EXL3SwitchGLU.direct_m6_clamp10,
            isinstance(quad_plan, _InstalledM6QuadQMVPlan),
            getattr(quad_plan, "geometry", None)
            == (4096, 2048, 216, 6, 10.0, 256, 36),
            getattr(quad_plan, "descriptor_sha256", None)
            == MIA_EXL3_M6_QUAD_DESCRIPTOR_SHA256,
            getattr(quad_plan, "stage_vector_bytes", None)
            == EXL3_M6_STAGE_VECTOR_BYTES,
            getattr(quad_plan, "stage_vectors_per_k_tile", None)
            == EXL3_M6_STAGE_VECTORS_PER_K_TILE,
            getattr(quad_plan, "dual_fc1_input", None)
            is _m6_dual_fc1_input_kernel(),
            getattr(quad_plan, "dual_fc1_inner", None)
            is _m6_dual_fc1_inner_kernel(),
            getattr(quad_plan, "activation_down", None)
            is _m6_clamp10_activation_down_kernel(),
            getattr(quad_plan, "down_inner", None)
            is _m6_down_inner_kernel(),
            getattr(quad_plan, "direct_final_tail", None)
            is _m6_direct_final_tail_kernel(),
            _callable_name(trellis_fused),
            getattr(trellis_fused, "__self__", None) is layer.ffn.switch_mlp,
            getattr(trellis_fused, "__func__", None) is EXL3SwitchGLU.fused,
            _callable_name(getattr(layer.ffn, "_input_rows_impl", None)),
            type(shared_projection).__name__,
            getattr(layer.ffn.shared_experts, "_gate_up_impl", None)
            is shared_projection,
            _callable_name(getattr(layer.ffn.gate, "_route_impl", None)),
            _callable_name(getattr(layer.attn, "_forward_impl", None)),
            _callable_name(getattr(layer.attn, "_cached_attention_impl", None)),
            _callable_name(getattr(layer.attn, "_uncached_kv_impl", None)),
            _callable_name(
                getattr(layer.attn, "_uncached_compressed_impl", None)
            ),
            _callable_name(getattr(layer.attn, "_cached_mask_impl", None)),
            _callable_name(getattr(layer.attn, "_nvfp4_sparse_mla", None)),
            _callable_name(getattr(layer.attn, "_nvfp4_prefill_mla", None)),
            getattr(layer.attn, "_mia_rope_provider", None)
            is (
                base_rope_provider
                if ratio == 0
                else compress_rope_provider
            ),
            type(getattr(layer.attn, "_mia_input_projection", None)).__name__,
            type(getattr(layer.attn, "_output_projection_impl", None)).__name__,
            getattr(layer.attn, "_mia_mla_workspace", None) is mla_workspace,
            getattr(layer.attn, "_mia_mla_query_layout", None),
            getattr(layer.attn, "_mia_mla_output_layout", None),
            getattr(getattr(layer.attn, "_mia_attn_sink", None), "dtype", None)
            == mx.float32,
            type(getattr(layer.attn, "_mia_qkv_plan", None)).__name__,
            _callable_name(getattr(layer.attn, "_mia_qkv_impl", None)),
            _callable_name(
                getattr(getattr(layer.attn, "_mia_qkv_plan", None), "project_learned", None)
            ),
            _callable_name(
                getattr(getattr(layer.attn, "_mia_qkv_plan", None), "target_records", None)
            ),
            _callable_name(
                getattr(getattr(layer.attn, "_mia_qkv_plan", None), "prefill_records", None)
            ),
            getattr(getattr(layer.attn, "_mia_qkv_plan", None), "q_weight", None)
            is layer.attn.q_norm.weight,
            getattr(getattr(layer.attn, "_mia_qkv_plan", None), "kv_weight", None)
            is layer.attn.kv_norm.weight,
            getattr(layer.ffn.gate, "_mia_router_contract", None),
        )
        expected_router_contract = router_contract_prefix + (
            "hash_tid2eid_int32"
            if layer_id < int(args.num_hash_layers)
            else "bias_selection_fp32",
            "unbiased_normalize_scale1p5",
        )
        required = (
            True,
            "_mia_exl3_forward",
            "direct_m6_clamp10",
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            "fused",
            True,
            True,
            "_required_input_rows",
            "MiaStackedMXFP8Projection",
            True,
            expected_gate,
            expected_forward,
            expected_cached_attention,
            "_mia_uncached_kv",
            expected_compressed,
            "_no_additive_mask",
            expected_sparse,
            expected_prefill,
            True,
            "MiaStackedMXFP8Projection",
            "MiaTP1WOMXFP8Plan",
            True,
            "BMHD",
            "BMHD",
            True,
            "MiaBoundQKVPrologue",
            "_mia_cached_qkv_records",
            "_project_learned_norms",
            "_run_target_qkv_records",
            "_run_prefill_qkv_records",
            True,
            True,
            expected_router_contract,
        )
        if installed != required:
            raise ValueError(
                f"Mia target layer {layer_id} route changed: "
                f"{installed!r} != {required!r}"
            )
        if ratio:
            compressor_route = (
                _callable_name(
                    getattr(layer.attn.compressor, "_mia_record_impl", None)
                ),
                getattr(layer.attn.compressor, "_mia_rope_provider", None)
                is compress_rope_provider,
                type(
                    getattr(
                        layer.attn.compressor,
                        "_mia_stacked_projection",
                        None,
                    )
                ).__name__,
                type(
                    getattr(layer.attn.compressor, "_project_rows_impl", None)
                ).__name__,
            )
            if compressor_route != (
                "_nvfp4_record_impl",
                True,
                "MiaStackedDenseProjection",
                "MiaStackedDenseProjection",
            ):
                raise ValueError(
                    f"Mia target layer {layer_id} compressor route changed"
                )
        if ratio == 4:
            indexer = layer.attn.indexer
            query_record_keywords = getattr(
                getattr(indexer, "_mia_query_records", None),
                "keywords",
                {},
            )
            indexer_routes = (
                getattr(indexer, "_mia_workspace", None) is indexer_workspace,
                getattr(indexer, "_mia_rope_table", None)
                is indexer_rope_table,
                query_record_keywords.get("cos_sin_cache")
                is indexer_rope_table.values,
                _callable_name(getattr(indexer, "_query_components_impl", None)),
                _callable_name(getattr(indexer, "_mia_query_records", None)),
                _callable_name(getattr(indexer, "_prepare_query_rows", None)),
                _callable_name(getattr(indexer, "_select_rows", None)),
                _callable_name(
                    getattr(indexer.compressor, "_mia_record_impl", None)
                ),
                getattr(indexer.compressor, "_mia_rope_provider", None)
                is compress_rope_provider,
                type(
                    getattr(
                        indexer.compressor,
                        "_mia_stacked_projection",
                        None,
                    )
                ).__name__,
            )
            if indexer_routes != (
                True,
                True,
                True,
                "_mia_query_components",
                "_run_installed_indexer_query_records",
                "_native_query_rows",
                "_run_installed_paged_indexer_phase_topk",
                "_indexer_record_impl",
                True,
                "MiaStackedDenseProjection",
            ):
                raise ValueError(
                    f"Mia target layer {layer_id} indexer route changed: "
                    f"{indexer_routes!r}"
                )

    for stage_id, stage in enumerate(stages):
        switch = stage.ffn.switch_mlp
        shared_projection = getattr(
            stage.ffn.shared_experts,
            "_mia_stacked_projection",
            None,
        )
        installed = (
            type(stage.attn).__name__,
            type(switch).__name__,
            _callable_name(getattr(stage.attn, "_forward_impl", None)),
            _callable_name(getattr(stage.attn, "_pack_draft_records", None)),
            _callable_name(getattr(stage.ffn, "_forward_impl", None)),
            _callable_name(getattr(stage.ffn, "_input_rows_impl", None)),
            type(shared_projection).__name__,
            getattr(stage.ffn.shared_experts, "_gate_up_impl", None)
            is shared_projection,
            _callable_name(getattr(stage.ffn.gate, "_route_impl", None)),
            tuple(
                str(getattr(getattr(switch, name, None), "mode", ""))
                for name in ("gate_proj", "up_proj", "down_proj")
            ),
            _callable_name(getattr(stage.attn, "_dspark_k5_mla", None)),
            _callable_name(
                getattr(stage.attn, "_dspark_k5_mla_graph", None)
            ),
            getattr(stage.attn, "_mia_mla_query_layout", None),
            getattr(stage.attn, "_mia_mla_output_layout", None),
            tuple(
                int(value)
                for value in getattr(
                    getattr(
                        stage.attn,
                        "_mia_draft_position_offsets",
                        None,
                    ),
                    "shape",
                    (),
                )
            ),
            getattr(
                getattr(
                    stage.attn,
                    "_mia_draft_position_offsets",
                    None,
                ),
                "dtype",
                None,
            ),
            getattr(getattr(stage.attn, "_mia_attn_sink", None), "dtype", None)
            == mx.float32,
            getattr(stage.attn, "_mia_rope_provider", None)
            is draft_rope_provider,
            getattr(draft_rope_provider, "max_positions", None)
            == target_physical_capacity_tokens,
            type(getattr(stage.attn, "_mia_input_projection", None)).__name__,
            _callable_name(getattr(stage.attn, "_project_kv_impl", None)),
            _callable_name(
                getattr(stage.attn, "_project_context_records_impl", None)
            ),
            _callable_name(
                getattr(stage.attn, "_prefill_context_impl", None)
            ),
            type(getattr(stage.attn, "_mia_qkv_plan", None)).__name__,
            _callable_name(
                getattr(getattr(stage.attn, "_mia_qkv_plan", None), "project_learned", None)
            ),
            _callable_name(
                getattr(getattr(stage.attn, "_mia_qkv_plan", None), "project_kv", None)
            ),
            _callable_name(
                getattr(getattr(stage.attn, "_mia_qkv_plan", None), "proposal_records", None)
            ),
            _callable_name(
                getattr(getattr(stage.attn, "_mia_qkv_plan", None), "context_records", None)
            ),
            getattr(getattr(stage.attn, "_mia_qkv_plan", None), "q_weight", None)
            is stage.attn.q_norm.weight,
            getattr(getattr(stage.attn, "_mia_qkv_plan", None), "kv_weight", None)
            is stage.attn.kv_norm.weight,
            type(getattr(stage.attn, "_output_projection_impl", None)).__name__,
            getattr(stage.ffn.gate, "_mia_router_contract", None),
        )
        required = (
            "DeepseekV4DSparkAttention",
            "MiaPhysicalM5K64SwitchGLU",
            "_run_k5",
            "NoneType",
            "_stock_forward",
            "_required_input_rows",
            "MiaStackedMXFP8Projection",
            True,
            "_mia_score_route",
            ("mxfp4", "mxfp4", "mxfp4"),
            "_run_dspark_k5_nvfp4_mla",
            "_run_dspark_k5_nvfp4_mla_graph",
            "BMHD",
            "BMHD",
            (MIA_DSPARK_BLOCK,),
            mx.int32,
            True,
            True,
            True,
            "MiaStackedMXFP8Projection",
            "NoneType",
            "_mia_context_records",
            "_mia_prefill_context_records",
            "MiaBoundQKVPrologue",
            "_project_learned_norms",
            "_project_kv_norm",
            "_run_k5_proposal_records",
            "_run_context_kv_records",
            True,
            True,
            "MiaTP1WOMXFP8Plan",
            router_contract_prefix
            + ("bias_selection_fp32", "unbiased_normalize_scale1p5"),
        )
        if installed != required:
            raise ValueError(
                f"Mia K64 DSpark stage {stage_id} route changed: "
                f"{installed!r} != {required!r}"
            )
    target_mhc = getattr(model.model, "_mia_mhc", None)
    draft_mhc = getattr(model.dspark, "_mia_mhc", None)
    target_connections = tuple(
        connection
        for layer in layers
        for connection in (layer.attn_hc, layer.ffn_hc)
    )
    draft_connections = tuple(
        connection
        for stage in stages
        for connection in (stage.attn_hc, stage.ffn_hc)
    )
    target_broadcast = getattr(
        getattr(target_connections[0], "_mia_mhc_weight", None),
        "fn_broadcast",
        None,
    )
    draft_broadcast = getattr(
        getattr(draft_connections[0], "_mia_mhc_weight", None),
        "fn_broadcast",
        None,
    )
    mhc_route_contract = (
        "broadcast_fn_fp32",
        "attention_post_pre_fn_fp32",
        "ffn_tiny_post_pre_fn_bf16_split32_fp32",
        "ffn_prefill_post_pre_fn_bf16_mma_bm64_fp32",
        "compact_gram_finalize",
        "head_bf16_then_rmsnorm",
    )
    mhc_installed = (
        target_mhc is not None,
        draft_mhc is not None,
        getattr(model.model.embed_tokens.weight, "dtype", None) == mx.bfloat16,
        getattr(target_mhc, "max_tokens", None),
        getattr(draft_mhc, "max_tokens", None),
        getattr(target_mhc, "prefill_min_rows", None),
        getattr(draft_mhc, "prefill_min_rows", None),
        getattr(target_mhc, "prefill_block_m", None),
        getattr(draft_mhc, "prefill_block_m", None),
        getattr(target_mhc, "route_contract", None),
        getattr(draft_mhc, "route_contract", None),
        getattr(target_mhc, "bound_hyper_connections", None),
        getattr(draft_mhc, "bound_hyper_connections", None),
        all(
            getattr(connection, "_mia_mhc_weight", None) is not None
            and connection._mia_mhc_weight.fn_bf16.dtype == mx.bfloat16
            for connection in target_connections + draft_connections
        ),
        tuple(getattr(target_broadcast, "shape", ())) == (24, MIA_HIDDEN),
        getattr(target_broadcast, "dtype", None) == mx.float32,
        tuple(getattr(draft_broadcast, "shape", ())) == (24, MIA_HIDDEN),
        getattr(draft_broadcast, "dtype", None) == mx.float32,
    )
    expected_mhc = (
        True,
        True,
        True,
        MIA_MAX_BATCH_TOKENS,
        MIA_MAX_BATCH_TOKENS,
        384,
        384,
        64,
        64,
        mhc_route_contract,
        mhc_route_contract,
        MIA_TARGET_LAYERS * 2,
        MIA_DSPARK_STAGES * 2,
        True,
        True,
        True,
        True,
        True,
    )
    if mhc_installed != expected_mhc:
        raise ValueError(
            f"the Mia mHC execution plan changed: {mhc_installed!r} "
            f"!= {expected_mhc!r}"
        )
    if (
        target_mhc is None
        or draft_mhc is None
        or _callable_name(getattr(model.model, "_hc_hidden_impl", None))
        != "_mia_hc_hidden"
        or _callable_name(getattr(model.model, "_collapse_impl", None))
        != "_mia_collapse"
        or _callable_name(
            getattr(model.model, "_run_mia_hc_target_tail_taps", None)
        )
        != "_run_mia_hc_target_tail_taps"
        or _callable_name(getattr(model, "_target_forward_route", None))
        != "_mia_target_forward"
        or _callable_name(getattr(model, "mia_dflash_forward", None))
        != "mia_dflash_forward"
        or getattr(draft_propose_impl, "__self__", None) is not model.dspark
        or getattr(draft_propose_impl, "__func__", None)
        is not type(model.dspark)._mia_fullgraph_propose_k5
        or _callable_name(getattr(model.dspark, "_make_cache_impl", None))
        != "_acquire_mia_cache"
        or _callable_name(getattr(model.dspark, "_commit_main_impl", None))
        != "_mia_commit_main"
        or len(tuple(getattr(model.dspark, "_mia_cache_arena", ())))
        != MIA_DSPARK_STAGES
        or any(
            getattr(getattr(cache, "ring", None), "mode", None)
            != "nvfp4_stock432_fixed_ring"
            or int(getattr(getattr(cache, "ring", None), "_capacity_rows", 0))
            != MIA_WINDOW
            or getattr(
                getattr(cache, "_write_initial_records", None),
                "keywords",
                {},
            ).get("owner")
            is not cache.ring
            or getattr(
                getattr(cache, "_write_commit_records", None),
                "keywords",
                {},
            ).get("owner")
            is not cache.ring
            for cache in tuple(getattr(model.dspark, "_mia_cache_arena", ()))
        )
        or bool(getattr(model.dspark, "_mia_cache_leased", True))
        or _callable_name(getattr(model.dspark, "_draft_input_ids_k5", None))
        != "_draft_input_ids_k5"
        or _callable_name(getattr(model.dspark, "_mia_draft_input_ids_k5", None))
        != "_mia_draft_input_ids_k5"
        or tuple(getattr(getattr(model.dspark, "_mia_noise_tail", None), "shape", ()))
        != (1, MIA_DSPARK_BLOCK - 1)
        or getattr(getattr(model.dspark, "_mia_noise_tail", None), "dtype", None)
        != mx.uint32
        or type(getattr(stages[-1], "markov_head", None)).__name__
        != "DSparkMarkovHead"
    ):
        raise ValueError("the carried Mia target/DSpark state machines are not installed")
    quantized_receipt = dict(getattr(model, "_mia_quantized_modules", {}))
    installed_quantized = {
        path: str(module.mode)
        for path, module in model.named_modules()
        if str(getattr(module, "mode", "")) in {"mxfp4", "mxfp8"}
    }
    if installed_quantized != quantized_receipt:
        raise ValueError(
            "the pinned Mia FP8/FP4 module route is not fully installed"
        )
    if (
        sum(mode == "mxfp4" for mode in quantized_receipt.values()) != 9
        or sum(mode == "mxfp8" for mode in quantized_receipt.values())
        != MIA_MXFP8_MODULES
    ):
        raise ValueError("the pinned K64 draft/general FP8 storage contract changed")

    target_piecewise_route = model.install_mia_piecewise_target_runtime()
    if (
        getattr(model.model, "_mia_piecewise_target_route", None)
        is not target_piecewise_route
        or type(getattr(model, "_mia_dflash_target_route", None)).__name__
        != "MiaDFlashTargetPhaseRoute"
        or int(getattr(target_piecewise_route, "physical_width", 0)) != 6
    ):
        raise ValueError("the Mia physical-M6 piecewise target route was not installed")

    # Physical pages are the final construction allocation. Every artifact,
    # topology, storage, callable, and quantization seal above must pass first.
    target_cache_arena = MiaTargetCacheArena(
        layers,
        capacity_tokens=target_physical_capacity_tokens,
        max_batch_tokens=max_batch_tokens,
    )

    page_geometry = tuple(
        MiaPageGeometry(
            layer_id=layer_id,
            compress_ratio=ratio,
            compressed_capacity=(
                0
                if ratio == 0
                else (target_physical_capacity_tokens + ratio - 1) // ratio
            ),
            attention_record_bytes=432,
            index_record_bytes=132 if ratio == 4 else 0,
        )
        for layer_id, ratio in enumerate(ratios)
    )
    workspace_geometry = (
        MiaWorkspaceGeometry(
            "compressor_c4_state_rings",
            (21, 72, 2, 1024),
            "float32",
            "absolute-position circular KV/score state",
        ),
        MiaWorkspaceGeometry(
            "indexer_c4_state_rings",
            (21, 72, 2, 256),
            "float32",
            "absolute-position circular KV/score state",
        ),
        MiaWorkspaceGeometry(
            "compressor_c128_state_rings",
            (20, 192, 2, 512),
            "float32",
            "absolute-position circular KV/score state",
        ),
        MiaWorkspaceGeometry(
            "indexer_top512_carry",
            (1, max_batch_tokens, MIA_INDEX_TOPK),
            "float32+int32",
            "shared immutable seeds plus bounded functional Metal outputs",
        ),
        MiaWorkspaceGeometry(
            "indexer_rope_table",
            (target_physical_capacity_tokens, 64),
            "float32",
            "one engine-owned immutable table shared by 21 ratio-4 layers",
        ),
        MiaWorkspaceGeometry(
            "nvfp4_prefill_nax_threadgroup",
            (28 * 1024,),
            "uint8",
            "threadgroup local per 16-head query group",
        ),
        MiaWorkspaceGeometry(
            "nvfp4_mla_token_major_output",
            (max_batch_tokens, MIA_HEADS, MIA_HEAD_DIM),
            "bfloat16",
            "functional BMHD output consumed directly by B12X",
        ),
        MiaWorkspaceGeometry(
            "target_swa_stock432_physical_pages",
            (
                MIA_TARGET_LAYERS,
                (max_batch_tokens + MIA_WINDOW + 64 + 63) // 64,
                64,
                432,
            ),
            "uint8",
            "cache-owned pages addressed through the logical 8416-row ring",
        ),
        MiaWorkspaceGeometry(
            "mhc_fp32_partials",
            (max_batch_tokens, 32, 35),
            "float32",
            "initial/head FP32 plus M<384 post-pre split-32 output",
        ),
        MiaWorkspaceGeometry(
            "mhc_prefill_compact",
            (max_batch_tokens, 11 + 24),
            "float32",
            "M>=384 compact Gram plus BF16-MMA FP32 projection outputs",
        ),
        MiaWorkspaceGeometry(
            "exl3_route_arena",
            (max_batch_tokens * MIA_TOPK,),
            "uint32",
            "functional Metal outputs owned by the fused MoE call",
        ),
        MiaWorkspaceGeometry(
            "wo_a_mxfp8_activation_values",
            (8, max_batch_tokens, 4096),
            "uint8",
            "inverse-RoPE group-major E4M3 values",
        ),
        MiaWorkspaceGeometry(
            "wo_a_mxfp8_activation_scales",
            (8, max_batch_tokens, 128),
            "uint8",
            "inverse-RoPE group-32 UE8M0 scales",
        ),
        MiaWorkspaceGeometry(
            "wo_a_bf16_boundary",
            (max_batch_tokens, 8, 1024),
            "bfloat16",
            "grouped WO-A output consumed directly by the WO-B producer",
        ),
        MiaWorkspaceGeometry(
            "wo_b_prefill_mxfp8_values",
            (max_batch_tokens, 8192),
            "uint8",
            "M>8 group-major E4M3 values, produced directly by WO-A at M16",
        ),
        MiaWorkspaceGeometry(
            "wo_b_prefill_mxfp8_scales",
            (max_batch_tokens, 256),
            "uint8",
            "M>8 group-32 UE8M0 scales, produced directly by WO-A at M16",
        ),
        MiaWorkspaceGeometry(
            "qkv_fused_projection_boundary",
            (max_batch_tokens, 1536),
            "bfloat16",
            "single row-adjacent q-rank/KV MXFP8 projection output",
        ),
        MiaWorkspaceGeometry(
            "qkv_learned_norm_boundaries",
            (max_batch_tokens, 1024 + MIA_HEAD_DIM),
            "bfloat16",
            "fused learned Q-rank/KV RMSNorm outputs",
        ),
        MiaWorkspaceGeometry(
            "qkv_finalized_outputs",
            (max_batch_tokens, MIA_HEADS * MIA_HEAD_DIM + 432),
            "bfloat16+uint8",
            "one-BF16-cast Q plus functional stock432 record output",
        ),
    )
    signatures = (
        MiaPrewarmSignature("target_prefill_m128_bm64", MIA_WINDOW, "prefill"),
        MiaPrewarmSignature("target_prefill_m6_bm8", 6, "prefill"),
        MiaPrewarmSignature("mhc_ffn_post_pre_bf16_mma_bm64", 384, "prefill"),
        MiaPrewarmSignature("wo_a_quantized_output_m16", 16, "prefill"),
        MiaPrewarmSignature(
            "qkv_reduced_prefill_m1024", MIA_LONG_PREFILL_CHUNK, "prefill"
        ),
        MiaPrewarmSignature("indexer_sparse_prefill", 1, "prefill"),
        MiaPrewarmSignature("indexer_sparse_decode", 1, "decode_verify"),
        MiaPrewarmSignature(
            "target_verify_m6_direct_5stage_staged_fc1_clamp10",
            6,
            "decode_verify",
        ),
        MiaPrewarmSignature("dspark_k5_bm8", MIA_DSPARK_BLOCK, "decode_verify"),
    )
    installed_routes = (
        "target_bounded_one_shard_raw_canonical_sha256_same_fd_loader",
        "target_mhc_role_bound_attn_fp32_ffn_bf16",
        "target_physical_m6_piecewise_compile_eager_attention",
        "draft_physical_k5_fullgraph_compile_live_cache_pages",
        "compressor_stock432_mia132",
        "compressor_fixed_absolute_state_rings",
        "target_shared_base_compress_rope_graphs",
        "draft_shared_base_rope_graph_through_position_384004",
        "target_draft_stacked_mxfp8_projections",
        "target_draft_fused_qkv_stock432_prologues",
        "target_finalized_record_cache_owner",
        "target_fixed_swa_paged_descriptor_8416",
        "dspark_initial_commit_record_cache_owners",
        "indexer_radix_top512",
        "mla_decode_direct_stock432",
        "mla_prefill_nax_mg16_tile32",
        "mla_token_major_query_output_no_transpose",
        "target_exl3_prefill_trellis_bm8_bm64_verify_m6_direct_5stage_staged_fc1_clamp10_bf16tail_u4_stage16b_96x8",
        "wo_tp1_b12x_inv_rope_mxfp8_bm8_m16q_bm64",
        "nonexpert_native_mxfp8",
        "dspark_k5_direct_stock432_k64_packed_gate_up_native_mxfp4",
        "target_fixed_swa_page_arena_m8224",
        "target_persistent_compressed_page_arena_384k_plus_m6_headroom",
        "dspark_persistent_fixed_ring_arena_128",
        "dflash2_structured_taps_fixed_linear_m6_copyspec_zero_owner",
    )
    artifact_small_file_sha256 = _artifact_small_file_sha256(
        target_small_file_sha256,
        draft_small_file_sha256,
    )
    identity = _mia_engine_identity(context_capacity_tokens, max_batch_tokens)
    return MiaDeepseekV4EnginePlan(
        context_capacity_tokens=context_capacity_tokens,
        target_physical_capacity_tokens=target_physical_capacity_tokens,
        max_batch_tokens=max_batch_tokens,
        max_sequences=MIA_MAX_SEQUENCES,
        page_geometry=page_geometry,
        workspace_geometry=workspace_geometry,
        indexer_workspace=indexer_workspace,
        indexer_rope_table=indexer_rope_table,
        mla_workspace=mla_workspace,
        target_cache_arena=target_cache_arena,
        prewarm_signatures=signatures,
        installed_routes=installed_routes,
        target_artifact=str(Path(target_root).resolve()),
        draft_artifact=str(Path(draft_root).resolve()),
        artifact_small_file_sha256=artifact_small_file_sha256,
        identity=identity,
    )
