"""What a checkpoint is, and whether this build can serve it, before a byte of its weights is read.

The first command on the day a model is published.  It reads the repository's
small files only (``config.json``, the weight index, the JSON header of each
shard, the generation config, the chat template) and answers, in seconds:

* which family the config's contents make it, and under which name;
* its geometry, size by size;
* every config key no model class reads and every value no code implements;
* every tensor outside the prefixes we map;
* a dry load: the model class is built from the config, the upstream tensor
  names and shapes go through the loader's own ``sanitize`` as arrays nobody
  evaluates, and the result is compared with the class's parameters, name
  for name and shape for shape.  The same for the draft head and for the
  vision tower (Law 1: every tower key, against upstream);
* whether the sampler in ``generation_config.json`` is the family's law, and
  whether the chat template is the one our parser was written against.

Anything that would be dropped or guessed is a **stop**, listed one per line.
An empty list means the rest of the day is machine time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mtplx.backends import signature
from mtplx.backends.geometry import Geometry

TRUNK_PREFIXES = ("model.language_model.", "language_model.")
VISION_PREFIXES = ("model.visual.", "vision_tower.")
DRAFT_PREFIX = "mtp."
HEAD_KEY = "lm_head.weight"
NGRAM_SHARD = re.compile(r"\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")

#: No config, weight index or chat template comes near this; a remote file is
#: read up to here and no further.
_SMALL_FILE_BYTES = 16 * 2**20
#: Shard headers read at once.
_HEADER_FETCHES = 8

#: Top-level config keys the loader, the vision splice or the forge read.
_TOP_LEVEL_READ = frozenset(
    {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "tie_word_embeddings",
        "language_model_only",
    }
)
_TOP_LEVEL_INERT = frozenset(
    {
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "initializer_range",
        "library_name",
        "model_file",
        "pad_token_id",
        "torch_dtype",
        "transformers_version",
        "use_cache",
    }
)
#: Keys our own packs add.
_PACK_KEYS = frozenset({"quantization", "quantization_config", "mlx_lm_extra_tensors"})

_VISION_INERT = frozenset({"dtype", "initializer_range"})
#: The tower runs one activation, and the server splices its final rows only:
#: rows tapped from inner blocks would be computed and dropped.
_VISION_IMPLEMENTED = {"hidden_act": ("gelu_pytorch_tanh",), "deepstack_visual_indexes": ([],)}

#: A name of the Qwen 4 generation, and of its preview.  The preview's sampler
#: was decided when it shipped; the generation's own is decided on its day.
_SAMPLER_UNDECIDED = re.compile(r"qwen[._-]?4")
_PREVIEW_NAME = re.compile(r"flash[._-]?next")

#: What the reasoning and tool parsers were written against.
_TEMPLATE_MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "<|im_start|>", "<|im_end|>")


@dataclass(frozen=True)
class Tensor:
    """One tensor as a safetensors header describes it."""

    dtype: str
    shape: tuple[int, ...]
    file: str = ""


class Source:
    """A checkpoint's small files: a local folder or a Hugging Face repository."""

    def __init__(self, ref: str | Path) -> None:
        from mtplx.artifacts import _hf_repo_id_from_ref

        self.ref = str(ref)
        self.repo_id = _hf_repo_id_from_ref(ref)
        self.folder = None if self.repo_id else Path(ref).expanduser()

    def json(self, name: str) -> dict[str, Any] | None:
        text = self.text(name)
        if text is None:
            return None
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None

    def text(self, name: str) -> str | None:
        if self.folder is not None:
            path = self.folder / name
            return path.read_text(encoding="utf-8") if path.is_file() else None
        import urllib.error

        from mtplx.artifacts import _hf_fetch_prefix

        try:
            return _hf_fetch_prefix(self.repo_id, name, end=_SMALL_FILE_BYTES - 1).decode("utf-8")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise

    def shards(self) -> list[str]:
        index = self.json("model.safetensors.index.json")
        if index and isinstance(index.get("weight_map"), dict):
            return sorted(set(index["weight_map"].values()))
        if self.folder is not None:
            return sorted(path.name for path in self.folder.glob("model*.safetensors"))
        from mtplx.artifacts import _hf_list_repo_files

        files, _error = _hf_list_repo_files(self.repo_id)
        return sorted(name for name in files if name.startswith("model") and name.endswith(".safetensors"))

    def tensors(self) -> dict[str, Tensor]:
        """Every tensor's dtype and shape, from the JSON header of each shard."""

        from concurrent.futures import ThreadPoolExecutor

        shards = self.shards()
        # A repository has a shard for every few gigabytes; over the network
        # the headers are fetched a few at a time.
        with ThreadPoolExecutor(max_workers=_HEADER_FETCHES) as pool:
            headers = list(pool.map(self._header, shards))
        found: dict[str, Tensor] = {}
        for shard, header in zip(shards, headers):
            for name, info in header.items():
                if name != "__metadata__":
                    found[name] = Tensor(str(info["dtype"]), tuple(int(dim) for dim in info["shape"]), shard)
        return found

    def _header(self, shard: str) -> dict[str, Any]:
        if self.folder is not None:
            with (self.folder / shard).open("rb") as handle:
                size = int.from_bytes(handle.read(8), "little")
                return json.loads(handle.read(size))
        from mtplx.artifacts import _hf_fetch_prefix

        prefix = _hf_fetch_prefix(self.repo_id, shard, end=65_535)
        size = int.from_bytes(prefix[:8], "little")
        if len(prefix) < 8 + size:
            prefix = _hf_fetch_prefix(self.repo_id, shard, end=8 + size - 1)
        return json.loads(prefix[8 : 8 + size])


@dataclass
class Census:
    """Everything the small files say, and the reasons to stop."""

    source: str
    model_type: str = ""
    family: str | None = None
    named: bool = False
    blocks: tuple[str, ...] = ()
    geometry: Geometry | None = None
    tensors: dict[str, int] = field(default_factory=dict)
    vision: dict[str, Any] = field(default_factory=dict)
    draft_head: dict[str, Any] = field(default_factory=dict)
    sampler: dict[str, Any] = field(default_factory=dict)
    template: dict[str, Any] = field(default_factory=dict)
    stops: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def go(self) -> bool:
        return not self.stops

    def lines(self) -> list[str]:
        """The report a person reads on release day."""

        out = [f"census of {self.source}"]
        if self.family is None:
            out.append(f"  family: not one this build serves (model_type {self.model_type!r})")
        else:
            how = "its own name" if self.named else f"its contents (model_type {self.model_type!r} is new)"
            out.append(f"  family: {self.family}, recognised by {how}")
            out.append(f"  blocks: {', '.join(self.blocks)}")
        if self.geometry is not None:
            out.append(f"  geometry: {self.geometry.describe()}")
        if self.tensors:
            out.append("  tensors: " + ", ".join(f"{count} {group}" for group, count in self.tensors.items()))
        if self.vision:
            out.append(f"  vision tower: {self.vision.get('summary')}")
        if self.draft_head:
            out.append(f"  draft head: {self.draft_head.get('summary')}")
        if self.sampler:
            out.append(f"  sampler: {self.sampler.get('summary')}")
        if self.template:
            out.append(f"  chat template: {self.template.get('summary')}")
        out += [f"  note: {note}" for note in self.notes]
        if self.go:
            out.append("  GO: nothing is dropped or guessed")
        else:
            out.append(f"  STOP: {len(self.stops)} thing(s) would be dropped or guessed")
            out += [f"    - {stop}" for stop in self.stops]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "go": self.go,
            "stops": list(self.stops),
            "notes": list(self.notes),
            "model_type": self.model_type,
            "family": self.family,
            "named": self.named,
            "blocks": list(self.blocks),
            "geometry": self.geometry.describe() if self.geometry else None,
            "sizes": self.geometry.sizes() if self.geometry else None,
            "tensors": dict(self.tensors),
            "vision": dict(self.vision),
            "draft_head": dict(self.draft_head),
            "sampler": dict(self.sampler),
            "template": dict(self.template),
        }


def take(source: str | Path, *, reference_template: str | None = None) -> Census:
    """Read ``source``'s small files and report."""

    opened = source if isinstance(source, Source) else Source(source)
    census = Census(source=opened.ref)
    config = opened.json("config.json")
    if config is None:
        census.stops.append("config.json is missing or is not an object")
        return census
    tensors = opened.tensors()
    _read_config(census, config)
    _count_tensors(census, tensors)
    if "quantization" in config:
        # A pack is the forge's output: packed weights no longer have the
        # shapes the model class declares, and its own load already proved them.
        census.notes.append("a converted pack, not an upstream checkpoint: the dry load is skipped")
    elif census.geometry is not None:
        _dry_load_trunk(census, config, tensors)
        _dry_load_draft_head(census, config, tensors)
    _check_vision(census, config, tensors)
    _check_sampler(census, opened.json("generation_config.json"))
    _check_template(census, _chat_template(opened), reference_template)
    return census


# ------------------------------------------------------------------ config


def _read_config(census: Census, config: Mapping[str, Any]) -> None:
    census.model_type = str(config.get("model_type") or signature.text_section(config).get("model_type") or "")
    found = signature.read(config)
    if found is None:
        census.stops.append(
            f"model_type {census.model_type!r} is not a hybrid trunk this build serves "
            "(its config has no recurrent-layer sizes)"
        )
    else:
        census.family, census.named, census.blocks = found.family, found.named, tuple(sorted(found.blocks))
        census.stops += [f"text config: {problem}" for problem in found.unread]
        try:
            census.geometry = Geometry.from_config(config)
        except KeyError as missing:
            census.stops.append(f"text config: {missing.args[0]} is not stated, and no size can be guessed for it")
        if not found.named and found.family == signature.QWEN4:
            # The loader picks this trunk's class by its model_type, and a
            # pack of it keeps the publisher's name (signature.under_own_name).
            census.stops.append(
                f"model_type {census.model_type!r} is new and its contents are a hyper-connection trunk: "
                "this build loads that trunk under its own model_type only, so a pack of this checkpoint would not load"
            )
    unknown = sorted(
        key
        for key in config
        if key not in _TOP_LEVEL_READ | _TOP_LEVEL_INERT | _PACK_KEYS and not key.startswith("mtplx_")
    )
    census.stops += [f"config key {key!r} is not read by anything (a block we do not build?)" for key in unknown]


# ----------------------------------------------------------------- tensors


def _group(name: str) -> str:
    if NGRAM_SHARD.search(name):
        return "n-gram shards"
    if name.startswith(TRUNK_PREFIXES) or name == HEAD_KEY:
        return "trunk"
    if name.startswith(VISION_PREFIXES):
        return "vision tower"
    if name.startswith(DRAFT_PREFIX):
        return "draft head"
    return "unmapped"


def _count_tensors(census: Census, tensors: Mapping[str, Tensor]) -> None:
    counts: dict[str, int] = {}
    unmapped: dict[str, int] = {}
    for name in tensors:
        group = _group(name)
        counts[group] = counts.get(group, 0) + 1
        if group == "unmapped":
            prefix = name.split(".", 2)
            head = ".".join(prefix[:2]) if len(prefix) > 2 else prefix[0]
            unmapped[head] = unmapped.get(head, 0) + 1
    census.tensors = {group: counts[group] for group in sorted(counts)}
    if not tensors:
        census.stops.append("no safetensors shard was found: nothing to map")
    for head, count in sorted(unmapped.items()):
        census.stops.append(f"{count} tensor(s) under {head}.* belong to no block we map")


def _lazy(tensors: Mapping[str, Tensor]) -> dict[str, Any]:
    """An array of each tensor's dtype and shape that nothing ever evaluates."""

    import mlx.core as mx

    kinds = {
        "BF16": mx.bfloat16,
        "F16": mx.float16,
        "F32": mx.float32,
        "I64": mx.int64,
        "I32": mx.int32,
        "U32": mx.uint32,
        "U8": mx.uint8,
        "BOOL": mx.bool_,
    }
    return {name: mx.zeros(tensor.shape, dtype=kinds.get(tensor.dtype, mx.float32)) for name, tensor in tensors.items()}


def _parameter_shapes(module: Any) -> dict[str, tuple[int, ...]]:
    from mlx.utils import tree_flatten

    return {name: tuple(int(dim) for dim in value.shape) for name, value in tree_flatten(module.parameters())}


def _compare(what: str, expected: Mapping[str, tuple], got: Mapping[str, tuple]) -> list[str]:
    """Stops for a strict load of ``got`` into a module with ``expected`` parameters."""

    stops = []
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    wrong = sorted(name for name in set(expected) & set(got) if tuple(expected[name]) != tuple(got[name]))
    if missing:
        stops.append(f"{what}: {len(missing)} parameter(s) the model class builds have no tensor, first {missing[0]}")
    if extra:
        stops.append(f"{what}: {len(extra)} tensor(s) have no parameter in the model class, first {extra[0]}")
    for name in wrong[:3]:
        stops.append(f"{what}: {name} is {list(got[name])}, the model class builds {list(expected[name])}")
    if len(wrong) > 3:
        stops.append(f"{what}: {len(wrong) - 3} more shape mismatches")
    return stops


def _dry_load_trunk(census: Census, config: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> None:
    trunk = {name: tensor for name, tensor in tensors.items() if _group(name) == "trunk"}
    if not trunk:
        return
    try:
        model = _trunk_class(census.family, config)
        sanitized = model.sanitize(_lazy(trunk))
        got = {name: tuple(int(dim) for dim in value.shape) for name, value in sanitized.items()}
        census.stops += _compare("trunk", _parameter_shapes(model), got)
    except Exception as exc:  # noqa: BLE001 - a census reports, it does not die
        census.stops.append(f"trunk: the model class could not be built or mapped from the config ({type(exc).__name__}: {exc})")
        return
    if census.family == signature.QWEN4:
        _check_ngram_shards(census, model, tensors)


def _trunk_class(family: str, config: Mapping[str, Any]) -> Any:
    """The family's model, built from the config with no weight evaluated."""

    config = copy.deepcopy(dict(config))
    if family == signature.QWEN4:
        from mtplx.models.qwen4_exp import Model, ModelArgs

        if signature.text_section(config).get("ple_layer_ids"):
            # As a pack carries it: the table is a sidecar, never a parameter.
            config["text_config"]["ngram_sidecar"] = True
        return Model(ModelArgs.from_dict(config))
    import importlib

    module = importlib.import_module(f"mlx_lm.models.{family}")
    return module.Model(module.ModelArgs.from_dict(config))


def _check_ngram_shards(census: Census, model: Any, tensors: Mapping[str, Tensor]) -> None:
    shards: dict[int, dict[int, Tensor]] = {}
    for name, tensor in tensors.items():
        match = NGRAM_SHARD.search(name)
        if match:
            shards.setdefault(int(match.group(1)), {})[int(match.group(2))] = tensor
    declared = list(census.geometry.ngram.layers) if census.geometry and census.geometry.ngram else []
    if sorted(shards) != declared:
        census.stops.append(f"n-gram shards at layers {sorted(shards)}, the config declares layers {declared}")
        return
    for layer, parts in shards.items():
        table = model.layers[layer].ple.ple_embedding.ngram_embedding
        if sorted(parts) != list(range(len(parts))):
            census.stops.append(f"n-gram table of layer {layer}: shard numbers are not 0 to {len(parts) - 1}")
            continue
        widths = {tensor.shape[1] for tensor in parts.values() if len(tensor.shape) == 2}
        rows = sum(tensor.shape[0] for tensor in parts.values())
        if widths != {int(table.dim)}:
            census.stops.append(f"n-gram table of layer {layer}: rows are {sorted(widths)} wide, the model class reads {int(table.dim)}")
        if rows > int(table.rows):
            census.stops.append(f"n-gram table of layer {layer}: {rows} rows, the hash addresses {int(table.rows)}")
        census.notes.append(
            f"n-gram table of layer {layer}: {len(parts)} shard(s), {rows:,} rows of {int(table.dim)} "
            f"({rows * int(table.dim) * 2 / 2**30:.1f} GiB in bfloat16, streamed to a sidecar by the forge)"
        )


def _dry_load_draft_head(census: Census, config: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> None:
    head = {name: tensor for name, tensor in tensors.items() if name.startswith(DRAFT_PREFIX)}
    declared = census.geometry.draft_head.layers if census.geometry else 1
    if not head:
        census.draft_head = {"present": False, "summary": "none in the repository"}
        census.notes.append(
            "no MTP head ships with these weights: the pack serves plain decode until one exists "
            "(the runbook's fallback: a drafter, or a head trained on the trunk)"
        )
        return
    layers = sorted({int(match.group(1)) for match in (re.match(r"mtp\.layers\.(\d+)\.", name) for name in head) if match})
    census.draft_head = {
        "present": True,
        "tensors": len(head),
        "layers": len(layers),
        "declared_layers": declared,
        "summary": f"{len(head)} tensors, {len(layers)} layer(s)",
    }
    if len(layers) != declared:
        census.notes.append(f"the config declares a draft head of {declared} layer(s); its tensors describe {len(layers)}, and the tensors decide")
    if census.family != signature.QWEN4:
        return
    try:
        from mtplx.expert_layout import split_fused_experts
        from mtplx.models.qwen4_exp import Qwen4ExpMTP

        model = _trunk_class(census.family, config)
        args = model.language_model.args
        stripped = {name[len(DRAFT_PREFIX) :]: value for name, value in _lazy(head).items()}
        mapped = split_fused_experts(stripped, hidden_size=int(args.hidden_size))
        expected = _parameter_shapes(Qwen4ExpMTP(args, layers=len(layers) or None))
        got = {name: tuple(int(dim) for dim in value.shape) for name, value in mapped.items()}
        census.stops += _compare("draft head", expected, got)
    except Exception as exc:  # noqa: BLE001
        census.stops.append(f"draft head: could not be built or mapped from the config ({type(exc).__name__}: {exc})")


# ------------------------------------------------------------------ vision


def _check_vision(census: Census, config: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> None:
    """Law 1: a declared tower is carried whole, so every key must be one our tower class has."""

    upstream = {name: tensor for name, tensor in tensors.items() if name.startswith(VISION_PREFIXES)}
    declared = config.get("vision_config")
    if not isinstance(declared, Mapping):
        if upstream:
            census.stops.append(f"{len(upstream)} vision tensors and no vision_config: the tower cannot be built")
        census.vision = {"declared": False, "tensors": len(upstream), "summary": "none declared (a text-only checkpoint)"}
        return
    if not upstream:
        census.stops.append("vision_config is declared and no vision tensor ships: a pack from this source would be blind")
        census.vision = {"declared": True, "tensors": 0, "summary": "declared, no tensors"}
        return
    from dataclasses import fields

    from mtplx.vision.qwen3_vl_tower import Qwen3VLVisionConfig, Qwen3VLVisionTower

    known = {item.name for item in fields(Qwen3VLVisionConfig)} | _VISION_INERT | set(_VISION_IMPLEMENTED)
    census.stops += [f"vision config: key {key!r} is not read by the tower class" for key in sorted(set(declared) - known)]
    census.stops += [
        f"vision config: {key} is {declared[key]!r}; implemented: {', '.join(map(repr, values))}"
        for key, values in _VISION_IMPLEMENTED.items()
        if key in declared and declared[key] not in values
    ]
    width = census.geometry.width if census.geometry else None
    if width is not None and declared.get("out_hidden_size") != width:
        census.stops.append(f"vision config: the tower emits rows {declared.get('out_hidden_size')} wide, the trunk is {width}")
    try:
        expected = _parameter_shapes(Qwen3VLVisionTower(Qwen3VLVisionConfig.from_dict(dict(declared))))
    except Exception as exc:  # noqa: BLE001
        census.stops.append(f"vision tower: the tower class could not be built from vision_config ({type(exc).__name__}: {exc})")
        return
    got = {_strip_vision_prefix(name): tensor.shape for name, tensor in upstream.items()}
    # A convolution's weight is stored channels-first upstream and
    # channels-last here: the same elements, compared as a set of sizes.
    conv = "patch_embed.proj.weight"
    if conv in expected and conv in got and sorted(expected[conv]) == sorted(got[conv]):
        got[conv] = expected[conv]
    census.stops += _compare("vision tower", expected, got)
    dtypes = sorted({tensor.dtype for tensor in upstream.values()})
    census.vision = {
        "declared": True,
        "tensors": len(upstream),
        "tower_class_parameters": len(expected),
        "dtypes": dtypes,
        "summary": f"{len(upstream)} tensors upstream, {len(expected)} in the tower class, {'/'.join(dtypes)}",
    }


def _strip_vision_prefix(name: str) -> str:
    for prefix in VISION_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


# ----------------------------------------------------- sampler and template


def sampler_law(census: Census) -> dict[str, Any]:
    """The sampler MTPLX applies to this checkpoint when a request names none."""

    from mtplx.backends import descriptors

    backend = "qwen4_exp" if census.family == signature.QWEN4 else descriptors.QWEN3_NEXT_DESCRIPTOR.backend_id
    return descriptors.sampler_defaults_for_model(census.source, {"recommended_backend": backend}).to_dict()


def _check_sampler(census: Census, generation_config: Mapping[str, Any] | None) -> None:
    law = sampler_law(census)
    if generation_config is None:
        census.sampler = {"declared": None, "law": law, "summary": f"no generation_config.json; the family law applies ({law})"}
        return
    declared = {key: generation_config.get(key) for key in law}
    same = all(declared[key] is None or float(declared[key]) == float(value) for key, value in law.items())
    census.sampler = {
        "declared": declared,
        "law": law,
        "summary": "the family law" if same else f"upstream {declared}; MTPLX applies {law}",
    }
    if same:
        return
    if _sampler_is_undecided(census):
        census.stops.append(
            f"generation_config.json samples at {declared}; MTPLX would apply {law}. "
            "A new generation's sampler is the founder's call before any pack is stamped"
        )
    else:
        census.notes.append(
            f"generation_config.json samples at {declared}; MTPLX applies {law}, "
            "the law already decided for this generation"
        )


def _sampler_is_undecided(census: Census) -> bool:
    """Nobody has decided this generation's sampler: its ``model_type`` is one
    this build has never seen, or its name is Qwen 4's and not the preview's."""

    name = census.source.lower()
    return not census.named or bool(_SAMPLER_UNDECIDED.search(name) and not _PREVIEW_NAME.search(name))


def _chat_template(source: Source) -> str | None:
    template = source.text("chat_template.jinja")
    if template is not None:
        return template
    tokenizer_config = source.json("tokenizer_config.json") or {}
    declared = tokenizer_config.get("chat_template")
    return declared if isinstance(declared, str) else None


def _check_template(census: Census, template: str | None, reference: str | None) -> None:
    if template is None:
        census.template = {"present": False, "summary": "none in the repository"}
        census.stops.append("no chat template ships: the server cannot render a conversation")
        return
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    missing = [marker for marker in _TEMPLATE_MARKERS if marker not in template]
    census.template = {"present": True, "sha256": digest, "bytes": len(template.encode("utf-8")), "missing_markers": missing}
    census.stops += [f"chat template has no {marker}: the qwen3 reasoning and tool parser expects it" for marker in missing]
    if reference is None:
        census.template["summary"] = f"{len(template):,} characters, every marker the parser needs is present" if not missing else "markers missing"
        return
    if reference == template:
        census.template["summary"] = "identical to the reference template"
        return
    import difflib

    diff = [
        line
        for line in difflib.unified_diff(reference.splitlines(), template.splitlines(), "reference", "upstream", lineterm="", n=0)
        if not line.startswith(("---", "+++", "@@"))
    ]
    census.template["diff"] = diff
    census.template["summary"] = f"differs from the reference in {len(diff)} line(s)"
    census.notes.append(
        f"the chat template differs from the reference in {len(diff)} line(s): read the diff (census --json) "
        "and run a tool turn and a thinking turn before a pack is stamped"
    )


__all__ = ["Census", "Source", "Tensor", "sampler_law", "take"]
