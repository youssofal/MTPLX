"""From sensitivity rows to a forge recipe: bits for each weight group under a
size budget and a budget of bytes read per token.

The predicted KL of a pack is the sum of what each group adds at its width
(groups were measured one at a time, so this is a first-order model). A
group's KL at a width that was not measured follows its nearest measured
width by the rounding law: one more bit halves the rounding error, so the KL
it adds falls by about four. The search starts from the greedy allocation
(widen where a step removes the most KL per unit of budget) and from every
uniform width that fits, and improves each by local moves (widen one group,
or narrow one to widen another), so it never returns worse than a uniform
pack of the same cost. The widths are chosen by the caller from those the
serving kernels take.

    python -m mtplx.benchmarks.quality.recipe ROWS --size-gb 26 --read-gb 2.4 --name NAME --out RECIPE.json
    python -m mtplx.benchmarks.quality.recipe ROWS --match-uniform 6 --name NAME --out RECIPE.json

The printed table compares the allocation with any uniform width, from the
same model of the rows: a prediction to check against the measured pack.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.quant_rounding import ROUNDINGS

GROUP_SIZE = 64
GIB = 1024**3


def stored_bytes(params: int, bits: int) -> float:
    """Affine packing: ``bits`` per weight plus a 16-bit scale and bias per group."""

    return params * (bits + 32 / GROUP_SIZE) / 8


@dataclass
class Choice:
    name: str
    params: int
    read_params: int
    measured: dict[int, float]  # bits -> mean KL measured
    suffixes: list[str]
    layers: list[int] | None
    bits: int = 0

    def kl(self, bits: int) -> float:
        nearest = min(self.measured, key=lambda m: abs(m - bits))
        return self.measured[nearest] * 4.0 ** (nearest - bits)

    def size(self, bits: int | None = None) -> float:
        return stored_bytes(self.params, self.bits if bits is None else bits)

    def read(self, bits: int | None = None) -> float:
        return stored_bytes(self.read_params, self.bits if bits is None else bits)


def choices_from_rows(rows: list[dict[str, Any]], rounding: str = "plain") -> list[Choice]:
    """One choice per measured group, from the rows measured with ``rounding``
    (rows that do not say were measured with plain rounding)."""

    by_group: dict[str, Choice] = {}
    for row in rows:
        if "bits" not in row or row.get("rounding", "plain") != rounding:
            continue
        choice = by_group.setdefault(row["group"], Choice(
            row["group"], int(row["params"]), int(row["params_read_per_token"]), {}, row["suffixes"], row["layers"]))
        choice.measured[int(row["bits"])] = float(row["kl_mean"])
    return list(by_group.values())


Bits = dict[str, int]


class _Budget:
    def __init__(self, choices: list[Choice], size: float, read: float):
        self.choices, self.size, self.read = choices, size, read

    def fits(self, bits: Bits) -> bool:
        slack = 1 + 1e-9  # a budget taken from a uniform pack must hold that pack despite rounding
        return (sum(c.size(bits[c.name]) for c in self.choices) <= self.size * slack
                and sum(c.read(bits[c.name]) for c in self.choices) <= self.read * slack)

    def kl(self, bits: Bits) -> float:
        return sum(c.kl(bits[c.name]) for c in self.choices)

    def cost(self, choice: Choice, bits: int, wider: int) -> float:
        return ((choice.size(wider) - choice.size(bits)) / self.size
                + (choice.read(wider) - choice.read(bits)) / self.read)


def _greedy(budget: _Budget, widths: list[int]) -> Bits:
    """Widen, one step at a time, wherever a step removes the most KL per unit of budget."""

    bits = {c.name: widths[0] for c in budget.choices}
    while True:
        steps = []
        for c in budget.choices:
            wider = [w for w in widths if w > bits[c.name]]
            if wider and budget.fits({**bits, c.name: wider[0]}):
                gain = c.kl(bits[c.name]) - c.kl(wider[0])
                steps.append((gain / (budget.cost(c, bits[c.name], wider[0]) + 1e-12), c.name, wider[0]))
        if not steps:
            return bits
        _, name, wider = max(steps)
        bits[name] = wider


def _improve(budget: _Budget, widths: list[int], bits: Bits) -> Bits:
    """Local search: widen one group, or narrow one to widen another, while it lowers the KL."""

    while True:
        moves = []
        for c in budget.choices:
            i = widths.index(bits[c.name])
            if i + 1 == len(widths):
                continue
            moves.append({**bits, c.name: widths[i + 1]})
            for d in budget.choices:
                j = widths.index(bits[d.name])
                if d is not c and j > 0:
                    moves.append({**bits, c.name: widths[i + 1], d.name: widths[j - 1]})
        better = [m for m in moves if budget.fits(m) and budget.kl(m) < budget.kl(bits) - 1e-15]
        if not better:
            return bits
        bits = min(better, key=budget.kl)


def allocate(choices: list[Choice], *, widths: list[int], size_budget: float, read_budget: float) -> None:
    """Set each choice's bits: the lowest predicted KL found within both budgets.

    The search starts from every uniform width that fits and from the greedy
    allocation, and improves each by local moves, so the result is never
    worse than a uniform pack of the same cost.
    """

    widths = sorted(widths)
    budget = _Budget(choices, size_budget, read_budget)
    starts = [{c.name: w for c in choices} for w in widths]
    starts = [s for s in starts if budget.fits(s)]
    if not starts:
        raise ValueError("the budgets do not fit even the narrowest width")
    starts.append(_greedy(budget, widths))
    best = min((_improve(budget, widths, s) for s in starts), key=budget.kl)
    for choice in choices:
        choice.bits = best[choice.name]


def summary(choices: list[Choice], bits_of: dict[str, int] | None = None) -> dict[str, float]:
    pick = (lambda c: bits_of[c.name]) if bits_of else (lambda c: c.bits)
    return {
        "predicted_kl": sum(c.kl(pick(c)) for c in choices),
        "size_gib": sum(c.size(pick(c)) for c in choices) / GIB,
        "read_gib_per_token": sum(c.read(pick(c)) for c in choices) / GIB,
    }


def recipe(choices: list[Choice], *, name: str, description: str, rounding: str = "plain") -> dict[str, Any]:
    # The body is the width most of the weights are at, so the pack's config
    # states the trunk's real width (servers read it), and only the groups at
    # another width are overrides, as a shipped mixed pack writes them.
    params_at: dict[int, int] = {}
    for choice in choices:
        params_at[choice.bits] = params_at.get(choice.bits, 0) + choice.params
    body = max(params_at, key=lambda bits: (params_at[bits], bits))
    overrides = []
    for choice in choices:
        if choice.bits == body:
            continue
        entry: dict[str, Any] = {"bits": choice.bits, "group_size": GROUP_SIZE}
        if choice.layers is not None:
            entry["layers"] = choice.layers
        overrides += [{"suffix": suffix, **entry} for suffix in choice.suffixes]
    return {
        "name": name,
        "description": description,
        "body_bits": body,
        "body_group_size": GROUP_SIZE,
        "body_mode": "affine",
        "rounding": rounding,
        "mtp_policy": "keep_bf16",
        "module_overrides": overrides,
        "allocation": {c.name: c.bits for c in choices},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mtplx.benchmarks.quality.recipe", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", help="sensitivity rows (JSON lines)")
    parser.add_argument("--size-gb", type=float, help="GiB for the quantized groups")
    parser.add_argument("--read-gb", type=float, help="GiB read per token by the quantized groups")
    parser.add_argument("--match-uniform", type=int, help="both budgets as a pack with every group at this width")
    parser.add_argument("--except", dest="exceptions", action="append", default=[], metavar="GROUP=BITS",
                        help="with --match-uniform: this group at these bits in the matched pack (a shipped pack's "
                             "8-bit routers, for example)")
    parser.add_argument("--size-scale", type=float, default=1.0,
                        help="with --match-uniform: the size budget times this (1.03 allows a pack 3%% larger)")
    parser.add_argument("--read-scale", type=float, default=1.0,
                        help="with --match-uniform: the read budget times this (1.03 allows 3%% more bytes per token)")
    parser.add_argument("--widths", default="4,6,8")
    parser.add_argument("--rounding", choices=ROUNDINGS, default="clip", help="the rounding the pack will be built with")
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    choices = choices_from_rows(rows, args.rounding)
    if not choices:
        parser.error(f"no rows measured with {args.rounding} rounding in {args.rows}")
    widths = [int(w) for w in args.widths.split(",")]
    if args.match_uniform is not None:
        matched = {c.name: args.match_uniform for c in choices}
        for spec in args.exceptions:
            group, _, bits = spec.rpartition("=")
            if group not in matched:
                parser.error(f"--except {spec!r}: no measured group named {group!r}")
            matched[group] = int(bits)
        like = summary(choices, matched)
        size_budget = like["size_gib"] * GIB * args.size_scale
        read_budget = like["read_gib_per_token"] * GIB * args.read_scale
    elif args.size_gb is not None and args.read_gb is not None:
        size_budget, read_budget = args.size_gb * GIB, args.read_gb * GIB
    else:
        parser.error("give --size-gb and --read-gb, or --match-uniform")
    allocate(choices, widths=widths, size_budget=size_budget, read_budget=read_budget)
    built = recipe(choices, name=args.name, description=args.description, rounding=args.rounding)
    built["predicted"] = summary(choices)
    Path(args.out).write_text(json.dumps(built, indent=2))

    print("| group | bits | params (M) | read per token (M) | predicted KL |")
    print("|---|---:|---:|---:|---:|")
    for c in choices:
        print(f"| {c.name} | {c.bits} | {c.params / 1e6:,.0f} | {c.read_params / 1e6:,.0f} | {c.kl(c.bits):.5f} |")
    print()
    print("| allocation | predicted KL | GiB | GiB read per token |")
    print("|---|---:|---:|---:|")
    compared = [(f"uniform {w}-bit, {args.rounding} rounding", {c.name: w for c in choices}) for w in widths]
    if args.exceptions:
        compared.insert(0, (f"the matched pack: {args.match_uniform}-bit, {', '.join(args.exceptions)}, "
                            f"{args.rounding} rounding", matched))
    for label, bits_of in [(args.name, None), *compared]:
        s = summary(choices, bits_of)
        print(f"| {label} | {s['predicted_kl']:.5f} | {s['size_gib']:.2f} | {s['read_gib_per_token']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
