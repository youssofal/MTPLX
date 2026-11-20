"""Mixed-precision convert driver for Forge ``module_overrides`` recipes.

Runs as a subprocess of ``mtplx forge build`` so the conversion keeps the same
memory isolation as the flat ``mlx_lm convert`` lane. The recipe's
``module_overrides`` entries each carry a module-path ``suffix`` (matched with
``str.endswith``), optional ``layers`` (decoder indices parsed from
``.layers.<n>.`` in the path), and quantization params. First matching
override wins; unmatched quantizable modules fall back to the recipe body
params (``True`` from the predicate).

The predicate is called by ``mlx_lm.utils.quantize_model`` with
``(path, module)``; returning a dict routes those params to ``to_quantized``
and records the per-module entry in ``config["quantization"]``. An override
with ``"quantize": false`` returns ``False`` so sensitive modules remain in
their source precision.

A recipe's ``rounding`` is ``"plain"`` (the default: each group's whole range,
as ``mlx_lm`` rounds) or ``"clip"`` (each group's range chosen for the least
rounding error, ``mtplx.quant_rounding``); both write the same layout. Clipped
rounding evaluates each quantized module as soon as it is built, so its peak is
the quantized model plus one module's source, like plain rounding's.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

from mtplx.quant_rounding import ROUNDINGS, clipping_predicate, evaluate_each_module

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def build_predicate(recipe: dict[str, Any]) -> Callable[[str, Any], bool | dict[str, Any]]:
    body_mode = str(recipe.get("body_mode") or "affine")
    overrides: list[
        tuple[str, frozenset[int] | None, bool | dict[str, Any]]
    ] = []
    for entry in recipe.get("module_overrides") or []:
        if not isinstance(entry, dict):
            raise SystemExit("module_overrides entries must be objects")
        suffix = str(entry.get("suffix") or "").strip()
        if not suffix:
            raise SystemExit("module_overrides entries need a non-empty suffix")
        raw_layers = entry.get("layers")
        layer_set = (
            frozenset(int(index) for index in raw_layers) if raw_layers is not None else None
        )
        params: bool | dict[str, Any]
        if entry.get("quantize") is False:
            params = False
        else:
            params = {
                "bits": int(entry.get("bits") or 8),
                "group_size": int(entry.get("group_size") or 64),
                "mode": str(entry.get("mode") or body_mode),
            }
        overrides.append((suffix, layer_set, params))

    def predicate(path: str, module: Any) -> bool | dict[str, Any]:
        del module
        for suffix, layer_set, params in overrides:
            if not path.endswith(suffix):
                continue
            if layer_set is not None:
                match = _LAYER_INDEX_RE.search(path)
                if match is None or int(match.group(1)) not in layer_set:
                    continue
            return dict(params) if isinstance(params, dict) else params
        return True

    return predicate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mtplx-forge-mixed-convert")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--recipe-json", required=True)
    parser.add_argument("--dtype", default=None, choices=[None, "float16"], nargs="?")
    args = parser.parse_args(argv)

    recipe = json.loads(args.recipe_json)
    if not isinstance(recipe, dict):
        raise SystemExit("--recipe-json must decode to an object")
    raw_body_bits = recipe.get("body_bits")
    body_bits = int(4 if raw_body_bits is None else raw_body_bits)
    if body_bits <= 0:
        raise SystemExit("module_overrides recipes require body_bits > 0")
    body = {
        "bits": body_bits,
        "group_size": int(recipe.get("body_group_size") or 64),
        "mode": str(recipe.get("body_mode") or "affine"),
    }
    rounding = str(recipe.get("rounding") or "plain")
    if rounding not in ROUNDINGS:
        raise SystemExit(f"rounding must be one of {', '.join(ROUNDINGS)}, not {rounding!r}")
    predicate = build_predicate(recipe)
    if rounding == "clip":
        predicate = clipping_predicate(predicate, body=body)

    import importlib

    from mlx_lm.convert import convert

    renamed = _name_the_module_of_a_renamed_trunk(Path(args.source))
    # ``convert`` quantizes through its module's ``quantize_model``; clipped
    # rounding evaluates each module right after it is built (see above).
    stock = importlib.import_module("mlx_lm.convert")
    quantize_model = stock.quantize_model
    if rounding == "clip":
        stock.quantize_model = evaluate_each_module(quantize_model)
    try:
        convert(
            hf_path=args.source,
            mlx_path=args.destination,
            quantize=True,
            q_bits=body["bits"],
            q_group_size=body["group_size"],
            q_mode=body["mode"],
            dtype=args.dtype,
            quant_predicate=predicate,
        )
    finally:
        stock.quantize_model = quantize_model
    if renamed:
        _write_the_pack_under_the_familys_name(Path(args.destination))
    return 0


def _name_the_module_of_a_renamed_trunk(source: Path) -> bool:
    """A plain trunk published under a new ``model_type`` has no module of
    that name in mlx-lm.  Its config's contents say which module reads it
    (``mtplx.backends.signature``); point the new name at that module for
    this process, so the converter builds the right model.  False when the
    source is under a name mlx-lm knows, or is not such a trunk."""

    from mtplx.backends import signature
    from mtplx.runtime import _install_architectures_declared_module_alias

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if signature.renamed_plain_trunk(config) is None:
        return False
    return _install_architectures_declared_module_alias(config)


def _write_the_pack_under_the_familys_name(destination: Path) -> None:
    """The pack carries the family's own ``model_type``: every name check in
    the product, and every released build, already serves it."""

    from mtplx.backends import signature

    path = destination / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(signature.under_own_name(config), indent=4, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
