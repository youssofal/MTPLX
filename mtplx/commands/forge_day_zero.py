"""From a repository id to packs that serve: one resumable command.

    mtplx forge day-zero Qwen/<repo> --out ~/day-zero/<name> --model "Qwen 4 27B"

Release day is a sequence nobody should have to remember, on a day when every
hour counts.  The command turns the census of a checkpoint into the steps for
that checkpoint, runs them in order, and writes a receipt after each (what
ran, how long it took, what it found), so a second run resumes where the
first stopped and the timings of a rehearsal are files, not memories.

    census          the small files only; stops on anything dropped or guessed
    download        the whole repository, after a disk check
    build/<pack>    convert, graft the vision tower, the draft head and its
                    depth, verification rows, the runtime stamp; an FP16
                    sibling for M1 and M2 is derived from its parent's pack
                    (its packed weights byte for byte, its bfloat16 cast)
    vision/<pack>   Law 1, key by key: the pack's tower against upstream's
    gate/<pack>     an FP16 sibling against its parent: logits and a greedy
                    continuation
    rows/<pack>     speed, memory and first text through ``mtplx serve``
    texts           the catalog entries (Python and Swift) and the cards
    publish         the upload commands, for the founder's word

``gate`` and ``rows`` load a model and need the machine cool and alone, so
they are not run from here: the command prints what to run and goes on once
the file it writes exists.  ``publish`` is never run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from mtplx.backends import signature
from mtplx.commands import forge_census, forge_release_kit

#: Kept free after everything is written: a serving Mac with less has failed
#: to save its session cache.
DISK_RESERVE_BYTES = 24 * 2**30
#: What a pack adds to its body's bits over a bfloat16 source's sixteen: the
#: packing's scales and biases, the modules kept at 8 bits, the tower and the
#: small tensors in their float type.  (The published Flash-Next packs are
#: 0.29 to 0.32 of their source at 4 bits.)
PACK_OVERHEAD_SHARE = 0.1
#: The new bytes of an FP16 sibling, as a share of its parent's: the bfloat16 tensors it rewrites (the packing's
#: scales and biases are 1 bit a weight beside 4 in a 4-bit pack in groups of 32, plus the norms and the tower);
#: every packed integer weight is a clone of the parent's blocks.
FP16_SIBLING_SHARE_OF_PARENT = 0.35
FP16 = " FP16"

OPTIMIZED_SPEED = "Optimized Speed"
BARE_SPEED = "Bare Speed"
OPTIMIZED_QUALITY = "Optimized Quality"

_NGRAM_TABLE = {"bits": 4, "group_size": 32}
#: Bare Speed is flat: these modules, which the model's own convert-time rule
#: lifts to 8 bits, stay at the pack's base (the router and the indexer
#: projection keep their 8 bits, as in the published pack), and the head's
#: attention is named so that it stays at the base beside group-32 experts.
_BARE_AT_THE_BASE = (
    "embed_tokens",
    "lm_head",
    "linear_attn.out_proj",
    "shared_expert_gate",
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "shared_expert.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)

#: The recipes of a hyper-connection trunk, by pack, in the order the catalog
#: offers them.  Each is the packing of the Flash-Next pack of that name (read
#: from the published packs' own configs; the forge's lane mirrors a recipe
#: into the draft head).  A plain trunk reuses the recipe recorded in one of
#: our published packs (``--pack KIND=<pack folder>``).
#:
#: ``recommended_profile`` is what the pack is stamped to serve under.  The
#: launcher promotes a list of known flagship ids to turbo; a pack of a model
#: with a new name is on no list, and without this line a new Speed pack
#: would serve on the slower default without a word.  The day's scored
#: window carries the other profile as an arm, so the stamp is checked on
#: the pack itself before anything is published.
HYPER_TRUNK_RECIPES: Mapping[str, Mapping[str, Any]] = {
    OPTIMIZED_SPEED: {
        "body_bits": 4,
        "body_group_size": 32,
        "body_mode": "affine",
        "ngram": _NGRAM_TABLE,
        "qwen4_mtp_bits": 4,
        "recommended_profile": "turbo",
    },
    BARE_SPEED: {
        "body_bits": 4,
        "body_group_size": 64,
        "body_mode": "affine",
        "ngram": _NGRAM_TABLE,
        "qwen4_mtp_bits": 4,
        "qwen4_mtp_group_size": 32,
        "module_overrides": [{"suffix": suffix, "bits": 4, "group_size": 64} for suffix in _BARE_AT_THE_BASE],
        "recommended_profile": "turbo",
    },
    OPTIMIZED_QUALITY: {
        "body_bits": 8,
        "body_group_size": 64,
        "body_mode": "affine",
        "ngram": _NGRAM_TABLE,
        "qwen4_mtp_bits": 8,
        "qwen4_qsa_8bit": True,
        "recommended_profile": "sustained",
    },
}
DETAILS = {
    OPTIMIZED_SPEED: "4-bit dynamic quant. Great coding speeds and good quality.",
    BARE_SPEED: "Flat 4-bit quantization. The smallest pack and the quickest speeds.",
    OPTIMIZED_QUALITY: "8-bit dynamic quant. Good coding speeds and the closest to the original model.",
}


class Stop(RuntimeError):
    """A step found a reason not to go on; the message is what a person reads."""


@dataclass(frozen=True)
class Pack:
    kind: str
    recipe: Mapping[str, Any]

    @property
    def slug(self) -> str:
        return self.kind.lower().replace(" ", "-")

    @property
    def half_floats(self) -> bool:
        """Built with float16 structural tensors: the sibling for M1 and M2, which have no native bfloat16."""

        return str(self.recipe.get("body_dtype") or "").lower() in ("fp16", "float16", "f16")

    @property
    def parent_kind(self) -> str:
        """The pack an FP16 sibling is derived from (``Optimized Speed`` for ``Optimized Speed FP16``)."""

        return self.kind.removesuffix(" FP16")

    @property
    def share_of_source(self) -> float:
        """At most this much of a bfloat16 source's bytes, for the disk check."""

        bits = self.recipe.get("body_bits")
        share = min(1.0, (4 if bits is None else int(bits)) / 16 + PACK_OVERHEAD_SHARE)
        # A sibling is a clone of its parent with the bfloat16 tensors rewritten (forge_fp16): the packing's
        # scales and biases, the norms, the tower. That is about a quarter of a 4-bit pack's bytes.
        return share * FP16_SIBLING_SHARE_OF_PARENT if self.half_floats else share

    @property
    def detail(self) -> str:
        """The sentence a person reads in the picker."""

        base = DETAILS.get(self.kind.removesuffix(" FP16"), self.kind)
        return f"{base} For M1 and M2 Macs." if self.half_floats else base


@dataclass
class Day:
    """One checkpoint's release day: where it comes from and where everything goes."""

    source: str
    out: Path
    model: str
    packs: tuple[Pack, ...] = ()
    hf_owner: str = "Youssofal"
    base_model: str = ""
    license: str = "apache-2.0"
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def receipt(self, step: str) -> Path:
        return self.out / "receipts" / (step.replace("/", "--") + ".json")

    def result(self, step: str) -> dict[str, Any]:
        """What an earlier step recorded."""

        if step not in self.results:
            self.results[step] = json.loads(self.receipt(step).read_text(encoding="utf-8"))["result"]
        return self.results[step]

    @property
    def local_source(self) -> Path:
        return Path(self.result("download")["path"])

    def pack_name(self, pack: Pack) -> str:
        return forge_release_kit.repo_name(self.model, pack.kind)

    def pack_path(self, pack: Pack) -> Path:
        return Path(self.result(f"build/{pack.slug}")["path"])


@dataclass(frozen=True)
class Step:
    name: str
    what: str
    #: Runs the step; what it returns is the receipt's result.
    run: Callable[[Day], dict[str, Any]] | None = None
    #: For a step run elsewhere: the line to run, and the file that line writes.
    handoff: Callable[[Day], str] | None = None
    awaits: Callable[[Day], Path] | None = None


# ------------------------------------------------------------------- steps


def steps(day: Day) -> list[Step]:
    """The day's steps, in order."""

    out = [
        Step("census", "read the small files; stop on anything dropped or guessed", run=_census),
        Step("download", "fetch the whole repository, after a disk check", run=_download),
    ]
    for pack in day.packs:
        parent = _parent_of(day, pack)
        out += [
            Step(
                f"build/{pack.slug}",
                f"derive {pack.kind} from {parent.kind}: its packed weights byte for byte, its bfloat16 tensors cast"
                if parent is not None
                else f"build {pack.kind}: convert, vision tower, draft head, verify, stamp",
                run=_bind(_build, pack),
            ),
            Step(f"vision/{pack.slug}", "the pack's vision tower against upstream's, key by key", run=_bind(_vision, pack)),
            Step(
                f"heads/{pack.slug}",
                "every draft-head norm against upstream's stored value plus one, tensor by tensor",
                run=_bind(_head_norms, pack),
            ),
        ]
        if parent is not None:
            out += [
                Step(
                    f"gate/{pack.slug}",
                    f"the same answers as {parent.kind} (logits and a greedy continuation)",
                    handoff=_bind(_gate_line, pack, parent),
                    awaits=_bind(_parent_check_file, pack),
                ),
                Step(
                    f"rows/{pack.slug}",
                    f"the rows of {parent.kind} (the same packed weights: not a separate speed claim)",
                    run=_bind(_rows_of_parent, pack, parent),
                ),
            ]
        else:
            out.append(
                Step(
                    f"rows/{pack.slug}",
                    "speed, memory and first text through mtplx serve",
                    handoff=_bind(_rows_line, pack),
                    awaits=_bind(_handoff_file, "rows", pack),
                )
            )
    out += [
        Step("texts", "catalog entries (Python and Swift) and the cards, from the measured rows", run=_texts),
        Step("publish", "upload: the founder's word", handoff=_publish_lines),
    ]
    return out


def _bind(function: Callable[..., Any], *bound: Any) -> Callable[[Day], Any]:
    return lambda day: function(day, *bound)


def _census(day: Day) -> dict[str, Any]:
    census = forge_census.take(day.source)
    (day.out / "census.txt").write_text("\n".join(census.lines()) + "\n", encoding="utf-8")
    if not census.go:
        raise Stop("\n".join(census.lines()))
    return census.to_dict()


def _download(day: Day) -> dict[str, Any]:
    local = Path(day.source).expanduser()
    if local.is_dir():
        size = _folder_bytes(local)
        _require_disk(day, source_bytes=size, to_fetch=0)
        return {"path": str(local.resolve()), "bytes": size, "fetched": False}
    from mtplx.commands.forge import probe_source
    from mtplx.hf_loader import pull_model

    size = int(probe_source(day.source).get("estimated_size_bytes") or 0)
    _require_disk(day, source_bytes=size, to_fetch=size)
    pulled = pull_model(day.source, cache_dir=day.out / "source")
    return {"path": str(pulled["path"]), "bytes": int(pulled.get("size_bytes") or 0), "fetched": True}


def _require_disk(day: Day, *, source_bytes: int, to_fetch: int) -> None:
    """Stop before anything is written when the day would not leave the reserve free."""

    packs = sum(int(source_bytes * pack.share_of_source) for pack in day.packs)
    need = to_fetch + packs + DISK_RESERVE_BYTES
    free = shutil.disk_usage(day.out).free
    if free < need:
        gib = 2**30
        raise Stop(
            f"the disk has {free / gib:.0f} GiB free; "
            + (f"the source ({to_fetch / gib:.0f} GiB), " if to_fetch else "")
            + f"{len(day.packs)} pack(s) (up to {packs / gib:.0f} GiB) and a {DISK_RESERVE_BYTES / gib:.0f} GiB reserve "
            f"need {need / gib:.0f} GiB"
        )


def _parent_of(day: Day, pack: Pack) -> Pack | None:
    """The day's pack an FP16 sibling derives from, when the day builds it."""

    if not pack.half_floats:
        return None
    return next((other for other in day.packs if other.kind == pack.parent_kind and not other.half_floats), None)


def _build(day: Day, pack: Pack) -> dict[str, Any]:
    census = day.result("census")
    parent = _parent_of(day, pack)
    if parent is not None:
        return _derive_sibling(day, pack, parent)
    if census["family"] == signature.QWEN4 and pack.half_floats:
        # The hyper-connection lane converts in the checkpoint's own float type; its FP16 sibling is derived
        # from the finished pack (forge_fp16), so the parent must be one of the day's packs.
        raise Stop(
            f"{pack.kind}: an FP16 sibling is derived from its parent's pack; name {pack.parent_kind!r} "
            "among the day's packs too"
        )
    if not census["draft_head"].get("present"):
        return _build_without_head(day, pack, family=census["family"])
    log = day.out / "logs" / f"build-{pack.slug}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        done = subprocess.run(
            build_command(day, pack), stdout=subprocess.PIPE, stderr=stream, text=True, check=False, env=_this_code()
        )
    if done.returncode != 0:
        raise Stop(f"forge build failed for {pack.kind} (exit {done.returncode}); its last lines:\n{_tail(log)}")
    # The build's own report is the last thing it prints.
    built = json.loads(done.stdout[done.stdout.rfind("\n{\n") + 1 :])
    stamp = built["runtime_metadata"]
    return {
        "path": built["local_path"],
        "draft_head": True,
        "mtp_depth_max": stamp.get("mtp_depth_max"),
        "mtp_depth_default": stamp.get("mtp_depth_default"),
        "recommended_profile": stamp.get("recommended_profile"),
        "sampler": stamp.get("sampler"),
        "log": str(log),
    }


def build_command(day: Day, pack: Pack) -> list[str]:
    """``mtplx forge build`` for one pack of the day, run by this interpreter."""

    return [
        sys.executable, "-m", "mtplx.cli", "forge", "build",
        "--repo", str(day.local_source),
        "--out", str(day.out / "forge"),
        "--run-id", pack.slug,
        "--recipe", json.dumps(pack.recipe),
        "--branded-name", day.pack_name(pack),
        "--model-root", str(day.out / "packs"),
        "--json",
    ]  # fmt: skip


def _build_without_head(day: Day, pack: Pack, *, family: str) -> dict[str, Any]:
    """A size that ships without MTP heads: the trunk and the tower, decoded
    one token at a time.  No depth is stamped, because nothing drafts."""

    from mtplx.commands import forge

    destination = day.out / "packs" / day.pack_name(pack)
    if destination.exists():
        raise Stop(f"{destination} exists: move it away or remove its receipt's pack first")
    recipe = dict(pack.recipe)
    if family == signature.QWEN4:
        from mtplx.commands.forge_qwen4_exp import run_lane

        recipe = run_lane(day.local_source, destination, recipe=recipe)["recipe"]
    else:
        # A plain trunk is a text model to its converter; a publisher's
        # checkpoint is in its float type, which is what the converter is told.
        run = day.out / "forge" / f"{pack.slug}-without-head"
        run.mkdir(parents=True, exist_ok=True)
        try:
            forge._convert_with_mlx_lm(
                day.local_source,
                destination,
                recipe=recipe,
                source_format=forge.SOURCE_BF16_NATIVE,
                run=run,
                env=_this_code(),
            )
        except forge.ForgeError as error:
            raise Stop(f"{error}\n(the converter's own output is in {run})") from error
    forge._ensure_vision_tower(day.local_source, destination)
    forge._validate_vision_payload(day.local_source, destination)
    return {"path": str(destination), "draft_head": False, "mtp_depth_max": None, "lane": recipe}


def _derive_sibling(day: Day, pack: Pack, parent: Pack) -> dict[str, Any]:
    """The FP16 sibling of a pack the day built: its packed weights byte for byte, its bfloat16 tensors cast
    (``forge_fp16``). Same draft head, depth and profile as the parent; a value float16 cannot hold stops it."""

    from mtplx.commands import forge_fp16

    built = day.result(f"build/{parent.slug}")
    destination = day.out / "packs" / day.pack_name(pack)
    if destination.exists():
        raise Stop(f"{destination} exists: move it away or remove its receipt's pack first")
    try:
        manifest = forge_fp16.derive(
            Path(built["path"]), destination, public_model_id=forge_release_kit.served_id(day.model, pack.kind)
        )
    except forge_fp16.Fp16SiblingError as error:
        raise Stop(f"{pack.kind}: {error}") from error
    return {
        **{key: built.get(key) for key in ("draft_head", "mtp_depth_max", "mtp_depth_default", "recommended_profile", "sampler")},
        "path": str(destination),
        "derived_from": built["path"],
        "fp16": manifest["summary"],
    }


def _vision(day: Day, pack: Pack) -> dict[str, Any]:
    """Law 1: every tower tensor upstream has is in the pack, same shape, never quantized."""

    upstream = _tower(forge_census.Source(day.local_source))
    packed = _tower(forge_census.Source(day.pack_path(pack)))
    if not upstream:
        return {"tower": False}
    dtypes = sorted({tensor.dtype for tensor in packed.values()})
    differences = {
        "missing from the pack": sorted(set(upstream) - set(packed)),
        "upstream does not have": sorted(set(packed) - set(upstream)),
        # A convolution may be stored channels-first or channels-last: the same sizes either way.
        "of another shape": sorted(
            name for name in set(upstream) & set(packed) if sorted(upstream[name].shape) != sorted(packed[name].shape)
        ),
    }
    problems = [f"{len(names)} tower tensor(s) {what}, first {names[0]}" for what, names in differences.items() if names]
    if set(dtypes) - {"BF16", "F16"}:
        problems.append(f"the tower is stored as {dtypes}: it is carried in its float type, never packed")
    elif pack.half_floats and "BF16" in dtypes:
        problems.append("the tower of an FP16 sibling is bfloat16: M1 and M2 would read every image in emulated bfloat16")
    if problems:
        raise Stop("; ".join(problems))
    return {"tower": True, "upstream_tensors": len(upstream), "pack_tensors": len(packed), "dtypes": dtypes}


def _head_norms(day: Day, pack: Pack) -> dict[str, Any]:
    """Every draft-head norm the loader hands the model equals upstream's stored value plus one, channel for
    channel (``forge_head_norms``); one tensor off stops the day. A pack built without a head has none to check;
    a synthetic headers-only source has no values, and the receipt says it was not checked."""

    from mtplx.commands import forge_head_norms

    census = day.result("census")
    if not (census.get("draft_head") or {}).get("present"):
        return {"draft_head": False}
    if forge_head_norms.headers_only(day.local_source):
        return {"draft_head": True, "checked": False, "why": "the source is a synthetic headers-only checkpoint"}
    try:
        result = forge_head_norms.check(day.pack_path(pack), day.local_source, family=census["family"])
    except forge_head_norms.HeadNormMismatch as error:
        raise Stop(f"{pack.kind}: {error}") from error
    return {
        "draft_head": True,
        "checked": True,
        "tensors": result["tensors"],
        "sidecar_read_as": result["sidecar_read_as"],
        "largest_error": max((row["largest_error"] for row in result["rows"]), default=0.0),
    }


def _tower(source: forge_census.Source) -> dict[str, forge_census.Tensor]:
    return {
        forge_census._strip_vision_prefix(name): tensor
        for name, tensor in source.tensors().items()
        if name.startswith(forge_census.VISION_PREFIXES)
    }


def _handoff_file(day: Day, kind: str, pack: Pack) -> Path:
    return day.out / kind / f"{pack.slug}.json"


def _gate_line(day: Day, pack: Pack, parent: Pack) -> str:
    return (
        "compare the sibling with its parent, one model at a time (argmax agreement and top-20 overlap at least "
        "0.90 on the last prompt positions, and a greedy continuation; the casts are exact, so only the "
        "activations' rounding differs):\n"
        f"      <python> scripts/quality_fp16_parent_logitdiff_20260707.py --parent {day.pack_path(parent)} "
        f"--sibling {day.pack_path(pack)} --output {_parent_check_file(day, pack)}"
    )


def _parent_check_file(day: Day, pack: Pack) -> Path:
    return day.out / "gate" / f"{pack.slug}-against-parent.json"


def _rows_of_parent(day: Day, pack: Pack, parent: Pack) -> dict[str, Any]:
    """A sibling's card states its parent's measured rows: the same packed weights, served on the same lanes
    (its stamp says so: not a separate speed claim). They were measured on this Mac, not on an M1 or M2."""

    source = _handoff_file(day, "rows", parent)
    if not source.exists():
        raise Stop(f"{pack.kind} states the rows of {parent.kind}, which are not recorded yet ({source})")
    rows = json.loads(source.read_text(encoding="utf-8"))
    rows["inherited_from"] = parent.kind
    target = _handoff_file(day, "rows", pack)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {"rows": str(target), "inherited_from": str(source)}


def _rows_line(day: Day, pack: Pack) -> str:
    stamped = day.result(f"build/{pack.slug}").get("recommended_profile") or "its stamped profile"
    other = "sustained" if stamped == "turbo" else "turbo"
    served = forge_release_kit.served_id(day.model, pack.kind)
    return (
        f"measure the pack through `mtplx serve` under the id {served} (served {stamped}, as stamped), interleaved "
        f"with itself on the {other} profile and with the best pack users have today, at the family's own "
        "sampler: first text, prompt reading and decode from 4K to 131K and on an image turn, and peak memory. "
        f"Record the rows of the stamped profile as JSON in {_handoff_file(day, 'rows', pack)} "
        "(peak_memory_gib, measured_on, and the rows the card states).\n"
        f"      if the {other} profile finishes the task sooner, set recommended_profile to {other} in the pack's "
        "mtplx_runtime.json and record its rows instead"
    )


def _texts(day: Day) -> dict[str, Any]:
    from mtplx import __version__

    described = []
    for pack in day.packs:
        if _parent_of(day, pack) is not None:
            check = _parent_check_file(day, pack)
            against = json.loads(check.read_text(encoding="utf-8")) if check.exists() else {}
            if against.get("passed") is not True:
                raise Stop(
                    f"{pack.kind}: {'its comparison with its parent did not pass' if against else 'not compared with its parent yet'} "
                    f"({check}); nothing is written about it"
                )
        measured = json.loads(_handoff_file(day, "rows", pack).read_text(encoding="utf-8"))
        described.append(
            forge_release_kit.facts_from_pack(
                day.pack_path(pack),
                model=day.model,
                kind=pack.kind,
                detail=pack.detail,
                tier=forge_release_kit.LEGACY_TIER if pack.half_floats else forge_release_kit.MODERN_TIER,
                hf_owner=day.hf_owner,
                base_model=day.base_model or day.source,
                license=day.license,
                peak_memory_gib=float(measured["peak_memory_gib"]),
                ar_only=not day.result(f"build/{pack.slug}")["draft_head"],
                rows=forge_release_kit.rows_from_file(_handoff_file(day, "rows", pack)),
                measured_on=str(measured.get("measured_on") or ""),
            )
        )
    folder = day.out / "texts"
    folder.mkdir(parents=True, exist_ok=True)
    written = forge_release_kit.render_all(described, mtplx_version=__version__)
    for name, text in written.items():
        (folder / name).write_text(text, encoding="utf-8")
    for pack, facts in zip(day.packs, described):
        shutil.copyfile(folder / f"card-{facts.repo_name}.md", day.pack_path(pack) / "README.md")
    return {"folder": str(folder), "files": sorted(written), "packs": [facts.hf_model_id for facts in described]}


def _publish_lines(day: Day) -> str:
    lines = ["with the founder's word, and not before:"]
    for pack in day.packs:
        name = day.pack_name(pack)
        lines.append(f"      hf upload {day.hf_owner}/{name} {day.out / 'packs' / name} . --repo-type model")
    lines.append(
        f"      then place {day.out / 'texts'}/catalog-entries.py.txt in mtplx/model_catalog.py and "
        "catalog-entries.swift.txt in MTPLXModelOption.swift, in one commit"
    )
    return "\n".join(lines)


# --------------------------------------------------------------- execution


def execute(day: Day, *, plan_only: bool = False, through: str | None = None, say: Callable[[str], None] = print) -> int:
    """Run the day's steps in order, resuming after the last receipt.

    Returns 0 when every step that can run here has run, 1 on a stop, and 3
    when the day is waiting for a line a person runs.
    """

    day.out.mkdir(parents=True, exist_ok=True)
    for step in steps(day):
        if plan_only:
            say(f"  {step.name:<28} {step.what}")
            continue
        code = _run_handoff(day, step, say) if step.run is None else _run_step(day, step, say)
        if code or step.name == through:
            return code
    return 0


def _run_step(day: Day, step: Step, say: Callable[[str], None]) -> int:
    receipt = day.receipt(step.name)
    if receipt.exists():
        say(f"  done     {step.name} ({json.loads(receipt.read_text(encoding='utf-8'))['seconds']:.0f} s, earlier)")
        return 0
    say(f"  running  {step.name}: {step.what}")
    started = time.monotonic()
    try:
        result = step.run(day)
    except Stop as stop:
        say(f"  STOP     {step.name}\n{stop}")
        return 1
    seconds = round(time.monotonic() - started, 1)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {"step": step.name, "seconds": seconds, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "result": result},
            indent=1,
        ),
        encoding="utf-8",
    )
    day.results[step.name] = result
    say(f"  done     {step.name} ({seconds:.0f} s)")
    return 0


def _run_handoff(day: Day, step: Step, say: Callable[[str], None]) -> int:
    awaited = step.awaits(day) if step.awaits is not None else None
    if awaited is not None and awaited.exists():
        say(f"  done     {step.name} ({awaited})")
        return 0
    say(f"  WAITING  {step.name}: {step.handoff(day)}")
    return 3


# --------------------------------------------------------------------- CLI


def packs_for(family: str | None, chosen: list[str] | None) -> tuple[Pack, ...]:
    """``--pack KIND=RECIPE`` (a JSON object, a JSON file, or one of our packs,
    whose recorded recipe is reused), or the family's own three."""

    if chosen:
        return tuple(Pack(kind.strip(), _recipe(source)) for kind, _, source in (item.partition("=") for item in chosen))
    if family == signature.QWEN4:
        # Each pack, then its FP16 sibling for M1 and M2 (every published 3.8 pack has one); a sibling is
        # derived from its parent's pack, so the parents come first.
        parents = tuple(Pack(kind, recipe) for kind, recipe in HYPER_TRUNK_RECIPES.items())
        return parents + tuple(Pack(pack.kind + FP16, {**pack.recipe, "body_dtype": "fp16"}) for pack in parents)
    raise Stop(
        "a plain trunk has no recipe of its own: name the packs, each with the recipe of the published pack it "
        "follows, for example --pack 'Optimized Speed=~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed'"
    )


def _recipe(source: str) -> dict[str, Any]:
    path = Path(source).expanduser()
    if path.is_dir():
        from mtplx.commands.public import resolved_default_profile_name_for_ref

        stamp = json.loads((path / "mtplx_runtime.json").read_text(encoding="utf-8"))
        recipe = dict((stamp.get("forge_provenance") or {}).get("forge_recipe") or {})
        if not recipe:
            # Both Flash-Next packs: an empty recipe would build the lane's
            # defaults and label them with this pack's name.
            raise Stop(f"{path.name} records no recipe (it was converted outside the forge): give the recipe as JSON")
        # The new pack serves under the profile the pack it follows serves under.
        recipe.setdefault("recommended_profile", resolved_default_profile_name_for_ref(path))
        return recipe
    loaded = json.loads(path.read_text(encoding="utf-8") if path.is_file() else source)
    if not isinstance(loaded, dict):
        raise Stop(f"a recipe is a JSON object: {source}")
    return loaded


def cmd_day_zero(args: Any) -> int:
    out = Path(args.out).expanduser().resolve()
    try:
        family = signature.family(forge_census.Source(args.source).json("config.json") or {})
        day = Day(
            source=str(args.source),
            out=out,
            model=str(args.model),
            packs=packs_for(family, getattr(args, "pack", None)),
            hf_owner=str(getattr(args, "owner", None) or "Youssofal"),
            base_model=str(getattr(args, "base_model", None) or ""),
            license=str(getattr(args, "license", None) or "apache-2.0"),
        )
    except Stop as stop:
        print(f"STOP: {stop}", file=sys.stderr)
        return 1
    print(f"day zero for {day.source} -> {out}")
    return execute(day, plan_only=bool(getattr(args, "plan", False)), through=getattr(args, "through", None))


def cmd_census(args: Any) -> int:
    reference = getattr(args, "reference_template", None)
    census = forge_census.take(
        args.source,
        reference_template=Path(reference).expanduser().read_text(encoding="utf-8") if reference else None,
    )
    print(json.dumps(census.to_dict(), indent=1) if getattr(args, "json", False) else "\n".join(census.lines()))
    return 0 if census.go else 1


def _this_code() -> dict[str, str]:
    """An environment in which a child process imports the mtplx this process
    runs: from a source tree, the installed package may be another one."""

    import mtplx

    root = str(Path(mtplx.__file__).resolve().parent.parent)
    return dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))


def _folder_bytes(folder: Path) -> int:
    return sum(os.path.getsize(os.path.join(root, name)) for root, _dirs, names in os.walk(folder) for name in names)


def _tail(path: Path, lines: int = 12) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


__all__ = ["Day", "Pack", "Step", "Stop", "cmd_census", "cmd_day_zero", "execute", "packs_for", "steps"]
