"""Build an Inkling pack from the publisher's pinned vision, target and draft weights."""

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from mtplx.vision_graft import _read_safetensors_header

SOURCE = "thinkingmachines/Inkling-2-Mini"
REVISION = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"
WEIGHTS = {
    "model-00001-of-00001.safetensors": "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1",
    "mtp.safetensors": "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2",
}
SIDECARS = (
    "config.json",
    "model.safetensors.index.json",
    "hf_quant_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tiktoken/tokenizer.model",
    "README.md",
)
PACK_NAMES = {
    "native": "Native",
    "quality": "Optimized Quality",
    "speed": "Optimized Speed",
    "bare-speed": "Bare Speed",
    "bare-speed-mxfp8": "Bare Speed MXFP8 Experts",
    "native-quality": "Native Experts Quality",
}


def _copy_weight(source: Path, destination: Path) -> None:
    try:
        os.link(source.resolve(), destination)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP}:
            raise
        shutil.copyfile(source, destination)


def verify_source(source: Path):
    config = json.loads((source / "config.json").read_text())
    if (
        config.get("model_type") != "inkling2_loop"
        or config["text_config"]["loop_config"]["is_varloop"]
    ):
        raise ValueError("Expected the fixed-depth Inkling 2 Mini release")
    target, draft = WEIGHTS
    header_size, header = _read_safetensors_header(source / target)
    _, draft_header = _read_safetensors_header(source / draft)
    header.pop("__metadata__", None)
    draft_header.pop("__metadata__", None)
    index = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if set(index) != set(header) or set(index.values()) != {target}:
        raise ValueError("Target tensor index disagrees with the checkpoint")
    vision = [entry for key, entry in header.items() if key.startswith("model.visual.")]
    if len(vision) != 422 or any(entry["dtype"] != "BF16" for entry in vision):
        raise ValueError("Incomplete native BF16 vision tower")
    if len(draft_header) != 54 or draft_header.get("fuse_weight", {}).get("shape") != [
        4,
        4,
    ]:
        raise ValueError("Incomplete DSpark sidecar")
    for name, expected in WEIGHTS.items():
        with (source / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(
                    f"Unexpected SHA-256 for {name} at revision {REVISION}"
                )
    return config, header_size, header


def _precision_sentences(manifest: dict) -> list[str]:
    recipe = manifest["recipe"]
    if recipe in {"bare-speed", "bare-speed-mxfp8"}:
        experts = (
            "The first 20 expert layers keep their published NVFP4 weights and the last four are re-coded to "
            "4-bit weights in groups of 64."
            if recipe == "bare-speed"
            else "The experts keep their published NVFP4 and MXFP8 weights."
        )
        return [
            "The attention projections of all 24 layers and the output vocabulary use 4-bit weights in groups "
            "of 64, which cuts the weights read for each token from about 5.5 GB to about 3 GB.",
            experts,
            "The routing, loop-control, normalization and convolution weights keep their published values.",
        ]
    native = [
        "The first 20 expert layers use their published NVFP4 weights and the last four use MXFP8.",
        "The FP32 routing and loop-control weights keep their source values.",
    ]
    if recipe == "speed":
        return [
            "The attention projections of all 24 layers and the output vocabulary use 8-bit weights in groups "
            "of 64, which cuts the weights read for each token from about 5.5 GB to about 3.7 GB.",
            "The first 20 expert layers keep their published NVFP4 weights and the last four are re-coded to "
            "4-bit weights in groups of 64.",
            "The routing, loop-control, normalization and convolution weights keep their published values.",
        ]
    if recipe == "native-quality" and manifest.get("matrix_bit_counts") == {"8": 1}:
        return [*native, "The output vocabulary uses 8-bit weights in groups of 32."]
    return native


def _write_model_card(staged: Path, manifest: dict) -> None:
    fp16 = manifest.get("precision_variant") == "fp16"
    sentences = _precision_sentences(manifest)
    converted = bool(manifest.get("matrix_bit_counts"))
    if fp16:
        sentences.append("BF16 tensors, including the vision tower and drafter, are converted to FP16. "
                         "The conversion manifest records rounding and the largest change.")
    elif converted:
        sentences.append("The vision tower and the drafter keep their published values.")
    else:
        sentences.append("Both weight files are byte-for-byte copies of the published checkpoint.")
    precision = " ".join(sentences)
    size = manifest["weight_bytes"] / 1_000_000_000
    card = f"""---
license: apache-2.0
library_name: mtplx
pipeline_tag: image-text-to-text
base_model: {SOURCE}
tags: [mtplx, mlx, apple-silicon, macos, vision, coding, speculative-decoding]
---
# {manifest["display_name"]}

Inkling 2 Mini is a vision and coding model for local use on Apple silicon. The middle 12 layers run three times per token. This pack carries the vision tower, tool-call template and DSpark block drafter. Its model weights occupy {size:.2f} GB.

{precision}

## Run locally

Select **{manifest["display_name"]}** in the MTPLX model picker. To start Pi with an installed copy, pass the downloaded pack directory.

```sh
mtplx start pi --model /path/to/downloaded/pack
```

The native sampler uses temperature 1.0, top-p 1.0 and an unrestricted vocabulary. The pack manifest records the source revision and hashes. [Original model card](https://huggingface.co/{SOURCE}/blob/{REVISION}/README.md)
"""
    (staged / "README.md").write_text(card)


def write_pack_metadata(staged: Path, recipe: str) -> dict:
    config = json.loads((staged / "config.json").read_text())
    index = json.loads((staged / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    precision = config.get("mtplx_inkling_precision", {})
    display_name = f"Inkling 2 Mini {PACK_NAMES[recipe]}"
    manifest = {
        "display_name": display_name,
        "recipe": recipe,
        "source": SOURCE,
        "source_revision": REVISION,
        "source_weights_sha256": WEIGHTS,
        "target_tensors": len(index),
        "vision_tensors": sum(key.startswith("model.visual.") for key in index),
        "draft_tensors": 54,
        "weight_bytes": sum(
            file.stat().st_size for file in staged.glob("*.safetensors")
        ),
        "matrix_bit_counts": {
            str(bits): sum(layout["bits"] == bits for layout in precision.values())
            for bits in sorted({layout["bits"] for layout in precision.values()})
        },
    }
    _write_model_card(staged, manifest)
    (staged / "mtplx-pack.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def stamp_fp16_variant(staged: Path, manifest: dict) -> None:
    display_name = f"Inkling 2 Mini {PACK_NAMES[manifest['recipe']]} FP16"
    manifest.update(display_name=display_name, precision_variant="fp16")
    (staged / "mtplx-pack.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_model_card(staged, manifest)


def stage_native_pack(
    source: Path, destination: Path, *, recipe: str = "native"
) -> dict:
    if recipe not in {"quality", "native"}:
        raise ValueError(f"Invalid exact-copy recipe {recipe!r}")
    source, destination = source.resolve(), destination.absolute()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    verify_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=destination.parent, prefix=".inkling-stage-"
    ) as scratch:
        staged = Path(scratch) / "pack"
        staged.mkdir()
        for name in WEIGHTS:
            _copy_weight(source / name, staged / name)
        for name in SIDECARS:
            dst = staged / ("UPSTREAM.md" if name == "README.md" else name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, dst)
        manifest = write_pack_metadata(staged, recipe)
        os.rename(staged, destination)
    return manifest
