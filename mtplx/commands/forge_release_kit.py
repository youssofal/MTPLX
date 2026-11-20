"""What is published about a pack, written from one description of it.

A pack reaches users through three texts that must agree: its entry in the
Python catalog, its entry in the app's catalog (Swift), and its Hugging Face
card.  Typed by hand on release day they drift: a size copied from the wrong
pack, a vision pack labelled text-only, a figure nobody measured.  Here one
``PackFacts`` is filled from the pack on disk and from measured rows, and the
three texts are rendered from it.

Nothing here uploads or edits a catalog file: the entries are text for a
person to place, because where a model sits in the recommendation order is
the founder's call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

VISION_PREFIXES = ("vision_tower.", "model.visual.")
MODERN_TIER = "modern"
LEGACY_TIER = "legacy"


@dataclass(frozen=True)
class Row:
    """One measured figure and where it was recorded."""

    run: str
    value: str
    #: The file and row the figure was read from; a card prints no figure without one.
    source: str


@dataclass(frozen=True)
class PackFacts:
    """Everything the three texts say about one pack."""

    #: ``Qwen 4 27B``: the model as people write it.
    model: str
    #: ``Optimized Speed``.
    kind: str
    #: One sentence a person reads in the picker.
    detail: str
    hf_owner: str
    base_model: str
    license: str
    size_bytes: int
    #: Peak unified memory while serving, GiB, from a measured row.
    peak_memory_gib: float
    vision: bool
    ar_only: bool = False
    tier: str = MODERN_TIER
    rows: tuple[Row, ...] = ()
    measured_on: str = ""
    extra_aliases: tuple[str, ...] = field(default=())

    @property
    def display_name(self) -> str:
        return f"{self.model} {self.kind}"

    @property
    def repo_name(self) -> str:
        return repo_name(self.model, self.kind)

    @property
    def hf_model_id(self) -> str:
        return f"{self.hf_owner}/{self.repo_name}"

    @property
    def id(self) -> str:
        return catalog_id(self.model, self.kind)

    @property
    def aliases(self) -> tuple[str, ...]:
        return (served_id(self.model, self.kind), _versioned(self.display_name), *self.extra_aliases)


def repo_name(model: str, kind: str) -> str:
    """``Qwen 3.8 27B``, ``Optimized Speed`` -> ``Qwen3.8-27B-MTPLX-Optimized-Speed``."""

    return f"{_versioned(model)}-MTPLX-{kind}".replace(" ", "-")


def catalog_id(model: str, kind: str) -> str:
    """``Qwen 3.8 27B``, ``Optimized Speed`` -> ``qwen38-27b-optimized-speed``."""

    return re.sub(r"[^a-z0-9]+", "-", _versioned(f"{model} {kind}").lower().replace(".", "")).strip("-")


def served_id(model: str, kind: str) -> str:
    """The id a server answers to for the pack: ``mtplx-qwen38-27b-optimized-speed``."""

    return f"mtplx-{catalog_id(model, kind)}"


def _versioned(name: str) -> str:
    """``Qwen 3.8 27B`` -> ``Qwen3.8 27B``: the family and its version as one word."""

    return re.sub(r"^(\w+) ([\d.]+)\b", r"\1\2", name)


def facts_from_pack(pack: Path | str, **described: Any) -> PackFacts:
    """Fill the facts a pack states about itself: its bytes and whether it sees."""

    pack = Path(pack)
    files = [path for path in pack.iterdir() if path.is_file() and not path.name.startswith(".")]
    index = json.loads((pack / "model.safetensors.index.json").read_text(encoding="utf-8"))
    config = json.loads((pack / "config.json").read_text(encoding="utf-8"))
    tower = any(str(name).startswith(VISION_PREFIXES) for name in index.get("weight_map", {}))
    if isinstance(config.get("vision_config"), Mapping) and not tower:
        raise ValueError(f"{pack.name} declares a vision tower and carries no tower tensor: it cannot be published")
    return PackFacts(size_bytes=sum(path.stat().st_size for path in files), vision=tower, **described)


# ---------------------------------------------------------------- catalogs


def python_entry(facts: PackFacts) -> str:
    """The ``CatalogModel(...)`` for ``mtplx/model_catalog.py``."""

    tier = "MODERN_TIER" if facts.tier == MODERN_TIER else "LEGACY_TIER"
    lines = [
        "    CatalogModel(",
        f"        id={_py(facts.id)},",
        f"        display_name={_py(facts.display_name)},",
        f"        detail={_py(facts.detail)},",
        f"        hf_model_id={_py(facts.hf_model_id)},",
        f"        size_bytes={_grouped(facts.size_bytes)},",
        f"        peak_memory_gib={facts.peak_memory_gib},",
        f"        recommended_tiers=frozenset({{{tier}}}),",
        "        aliases=(",
        *[f"            {_py(alias)}," for alias in facts.aliases],
        "        ),",
    ]
    if facts.ar_only:
        lines.append("        ar_only=True,")
    lines.append("    ),")
    return "\n".join(lines)


def swift_entry(facts: PackFacts) -> str:
    """The ``MTPLXModelOption(...)`` for ``MTPLXModelOption.swift``."""

    tier = ".modernApple" if facts.tier == MODERN_TIER else ".legacyApple"
    last = [f"            recommendedFor: [{tier}],", "            arOnly: true"] if facts.ar_only else [f"            recommendedFor: [{tier}]"]
    return "\n".join(
        [
            "        MTPLXModelOption(",
            f"            id: {_py(facts.id)},",
            f"            displayName: {_py(facts.display_name)},",
            f"            shortName: {_py(facts.display_name)},",
            f"            localizedDetailKey: {_py(facts.detail)},",
            f"            hfModelID: {_py(facts.hf_model_id)},",
            "            localCandidates: [",
            f"                {_py('~/.mtplx/models/' + facts.hf_model_id.replace('/', '--'))},",
            f"                {_py('~/.mtplx/models/' + facts.repo_name)},",
            "            ],",
            "            aliases: [",
            *[f"                {_py(alias)}," for alias in facts.aliases],
            "            ],",
            f"            sizeBytes: {_grouped(facts.size_bytes)},",
            f"            peakMemoryGiB: {facts.peak_memory_gib},",
            *last,
            "        ),",
        ]
    )


def entry_fields(text: str) -> dict[str, Any]:
    """The values of a rendered entry, either language, for comparing the two."""

    names = {
        "displayName": "display_name",
        "localizedDetailKey": "detail",
        "hfModelID": "hf_model_id",
        "sizeBytes": "size_bytes",
        "peakMemoryGiB": "peak_memory_gib",
    }
    fields: dict[str, Any] = {}
    for key, value in re.findall(r"^\s*(\w+)\s*[:=]\s*(\"[^\"]*\"|[\d_.]+),?$", text, flags=re.MULTILINE):
        fields[names.get(key, key)] = json.loads(value) if value.startswith('"') else float(value.replace("_", ""))
    block = re.search(r"aliases\s*[:=]\s*[\[(](.*?)[\])],", text, flags=re.DOTALL)
    fields["aliases"] = tuple(json.loads(item) for item in re.findall(r"\"[^\"]*\"", block.group(1))) if block else ()
    fields["ar_only"] = bool(re.search(r"ar_only=True|arOnly: true", text))
    fields["legacy"] = bool(re.search(r"LEGACY_TIER|\.legacyApple", text))
    fields.pop("shortName", None)
    return fields


def _py(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _grouped(number: int) -> str:
    return f"{int(number):_}"


# -------------------------------------------------------------------- card


def card(facts: PackFacts, *, mtplx_version: str) -> str:
    """The Hugging Face card: tagline, the app's name for the pack, the pitch,
    the measured table, one line on the base model and its licence."""

    if not facts.rows:
        raise ValueError("a card prints measured figures: no rows were given, so no card is written")
    # The last tag is the model's family as people search for it: "qwen" for "Qwen 4 27B".
    tags = ["mlx", "apple-silicon", "macos", "mtplx", "local-ai", "coding", facts.model.split()[0].lower()]
    if not facts.ar_only:
        tags += ["speculative-decoding", "multi-token-prediction", "mtp"]
    if facts.vision:
        tags.append("vision")
    front = [
        "---",
        f"license: {facts.license}",
        "library_name: mtplx",
        f"pipeline_tag: {'image-text-to-text' if facts.vision else 'text-generation'}",
        f"base_model: {facts.base_model}",
        "base_model_relation: quantized",
        "tags:",
        *[f"- {tag}" for tag in tags],
        "---",
    ]
    how = (
        "It keeps the model's own multi-token-prediction head, so MTPLX drafts several tokens ahead and "
        "checks them in one pass: the same output distribution, more tokens per second."
        if not facts.ar_only
        else "The upstream release has no multi-token-prediction head for this size yet, so this pack "
        "decodes one token at a time. It moves to drafted decoding when a head exists."
    )
    sees = " It reads images: the full vision tower of the base model is in the pack." if facts.vision else ""
    table = ["| Run | Result |", "|---|---|", *[f"| {row.run} | {row.value} |" for row in facts.rows]]
    body = [
        f"**[MTPLX](https://mtplx.com): the fastest way to run {facts.model} on a Mac.**",
        "",
        f"# {facts.display_name}",
        "",
        f"**{facts.detail}**",
        "",
        f"This is {facts.base_model.split('/')[-1]} packed for [MTPLX](https://mtplx.com) on Apple Silicon. {how}{sees}",
        "",
        f"## Measured on MTPLX {mtplx_version}" + (f" ({facts.measured_on})" if facts.measured_on else ""),
        "",
        *table,
        "",
        f"It needs about {facts.peak_memory_gib:g} GiB of unified memory while serving, and "
        f"{facts.size_bytes / 2**30:.1f} GiB on disk.",
        "",
        f"Base model: [{facts.base_model}](https://huggingface.co/{facts.base_model}), licence {facts.license}.",
        "",
    ]
    return "\n".join([*front, "", *body])


def rows_from_file(path: Path | str) -> tuple[Row, ...]:
    """Measured rows as the harness records them: ``[{"run", "value", "source"}, ...]``."""

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = loaded["rows"] if isinstance(loaded, Mapping) else loaded
    return tuple(Row(run=str(row["run"]), value=str(row["value"]), source=str(row["source"])) for row in rows)


def render_all(facts: Sequence[PackFacts], *, mtplx_version: str) -> dict[str, str]:
    """Every text for every pack, by file name."""

    out = {
        "catalog-entries.py.txt": "\n".join(python_entry(pack) for pack in facts) + "\n",
        "catalog-entries.swift.txt": "\n".join(swift_entry(pack) for pack in facts) + "\n",
    }
    for pack in facts:
        out[f"card-{pack.repo_name}.md"] = card(pack, mtplx_version=mtplx_version)
    return out


__all__ = [
    "PackFacts",
    "Row",
    "card",
    "entry_fields",
    "facts_from_pack",
    "python_entry",
    "render_all",
    "repo_name",
    "rows_from_file",
    "swift_entry",
]
