"""GPU time of a kernel or a block at the real shapes, on random weights.

    python -m mtplx.engine.tools.ktime list
    python -m mtplx.engine.tools.ktime <case> [--repeats 15] [--out FILE.json]

No pack is read.  A case builds its inputs (random values, the real shapes,
the pack's quantization) and one or more arms: functions of those inputs
whose graphs are the launches the product makes.  An arm is recorded once
with the executor's recorder, built into a program and run ``repeats`` times
with one command buffer per launch, the same profile the verify site writes
for a real program.  Per kernel it prints launches and the median GPU
microseconds; per arm the total.  Arms of one case are run interleaved, so a
clock that drifts moves them together.

Each launch alone in a command buffer carries a fixed GPU cost (about 2
microseconds on the M5 Max): arms that differ in launch count are compared on
the same footing the real-pack profile uses, which is the point.

Without the executor module the tool times ``mx.eval`` plus a synchronize on
the wall clock and says so (``mode: wall``): that includes host time.

Machine rules: refuses to start while a hold file exists (``MTPLX_TOOLS_HOLD``,
see ``census.py``); every case stays under about 1 GB.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Callable

CASES: dict[str, Callable[[], dict[str, Any]]] = {}


def case(name: str):
    def wrap(fn):
        CASES[name] = fn
        return fn

    return wrap


# ------------------------------------------------------------------ timing


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def time_arms(arms: dict[str, Callable], inputs: list, *, pinned: list | None = None, repeats: int = 15) -> dict[str, Any]:
    """Per arm: launches, GPU microseconds (median of ``repeats``), and the same per kernel."""

    import mlx.core as mx

    from mtplx.engine import runtime

    mx.eval(*inputs, *(pinned or []))
    for fn in arms.values():  # build every kernel and trace before anything is timed
        mx.eval(*_as_list(fn(*inputs)))
    mx.synchronize()
    result: dict[str, Any] = {"mode": "gpu", "repeats": repeats, "arms": {}}
    programs: dict[str, Any] = {}
    if runtime.available():
        engine = runtime.engine()
        engine.pin(list(pinned or []))
        try:
            for name, fn in arms.items():
                programs[name], _ = engine.record_function(fn, inputs, name=name)
        except Exception as exc:  # noqa: BLE001 - the wall clock still answers; the reason is printed
            result["mode"] = "wall"
            result["engine_refused"] = f"{type(exc).__name__}: {exc}"
            programs = {}
    else:
        result["mode"] = "wall"
        result["engine_refused"] = runtime.unavailable()

    samples: dict[str, list] = {name: [] for name in arms}
    for _ in range(repeats):
        for name, fn in arms.items():
            if programs:
                steps = runtime.engine().profile(programs[name], inputs)
                samples[name].append(steps)
            else:
                mx.synchronize()
                start = time.perf_counter_ns()
                mx.eval(*_as_list(fn(*inputs)))
                mx.synchronize()
                samples[name].append(time.perf_counter_ns() - start)
    for name in arms:
        if not programs:
            result["arms"][name] = {"us": round(statistics.median(samples[name]) / 1e3, 1)}
            continue
        runs = samples[name]
        totals = [sum(step["gpu_ns"] for step in run) for run in runs]
        kernels: dict[str, dict[str, Any]] = {}
        for index, step in enumerate(runs[0]):
            entry = kernels.setdefault(step["kernel_name"], {"launches": 0, "ns": []})
            entry["launches"] += 1
            entry["ns"].append(statistics.median(run[index]["gpu_ns"] for run in runs))
        result["arms"][name] = {
            "launches": len(runs[0]),
            "us": round(statistics.median(totals) / 1e3, 1),
            "us_min": round(min(totals) / 1e3, 1),
            # Under someone else's load the median is theirs; the lower quartile is closer to the kernel's own time.
            "us_q1": round(sorted(totals)[len(totals) // 4] / 1e3, 1),
            # Every repeat's total, in order: arms are interleaved, so repeat i of two arms ran side by side.
            "samples_us": [round(total / 1e3, 1) for total in totals],
            "by_kernel": {
                kernel: {"launches": entry["launches"], "us": round(sum(entry["ns"]) / 1e3, 1)}
                for kernel, entry in sorted(kernels.items(), key=lambda item: -sum(item[1]["ns"]))
            },
        }
    return result


def quiet_ratios(result: dict[str, Any], baseline: str, *, slack: float = 1.25) -> dict[str, Any]:
    """Each arm against ``baseline`` over the repeats in which the machine was quiet.

    A repeat counts as quiet when the baseline arm ran within ``slack`` of its
    best run.  Arms are interleaved, so the other arms of that repeat ran
    beside it.  Returns the number of quiet repeats and, per arm, the median
    of its time in them and of its ratio to the baseline.
    """

    arms = result.get("arms", {})
    base = arms.get(baseline, {}).get("samples_us")
    if not base:
        return {}
    floor = min(base)
    quiet = [index for index, value in enumerate(base) if value <= floor * slack]
    out: dict[str, Any] = {"baseline": baseline, "quiet_repeats": len(quiet), "of": len(base)}
    for name, row in arms.items():
        samples = row.get("samples_us") or []
        picked = [samples[index] for index in quiet if index < len(samples)]
        if not picked:
            continue
        out[name] = {
            "us": round(statistics.median(picked), 1),
            "ratio": round(statistics.median(samples[index] / base[index] for index in quiet if index < len(samples)), 3),
        }
    return out


def render(name: str, result: dict[str, Any]) -> str:
    lines = [f"# ktime {name}: {result['mode']} time, median of {result['repeats']}"]
    if result.get("engine_refused"):
        lines.append(f"(executor not used: {result['engine_refused']})")
    for arm, row in result["arms"].items():
        if "launches" not in row:
            lines.append(f"\n{arm}: {row['us']} us (wall, host time included)")
            continue
        lines.append(
            f"\n{arm}: {row['launches']} launches, {row['us']} us "
            f"(lower quartile {row.get('us_q1', row['us'])} us, best run {row['us_min']} us)"
        )
        for kernel, entry in row["by_kernel"].items():
            lines.append(f"  {entry['launches']:4d}  {entry['us']:9.1f} us  {kernel[:100]}")
    return "\n".join(lines)


# ------------------------------------------------------------------- cases


def _big_pack(key, rows: int, columns: int, *, bits: int, group: int, spread: float, chunk_rows: int = 65_536):
    """A random affine pack built a chunk of rows at a time (no temporary the size of its values)."""

    import mlx.core as mx

    keys = mx.random.split(key, 3)
    words = columns * bits // 32
    parts = []
    for index, start in enumerate(range(0, rows, chunk_rows)):
        count = min(chunk_rows, rows - start)
        part_keys = mx.random.split(mx.random.fold_in(keys[0], index) if hasattr(mx.random, "fold_in") else mx.random.key(1000 + index), 2)
        high = mx.random.randint(0, 1 << 16, (count, words), key=part_keys[0]).astype(mx.uint32)
        low = mx.random.randint(0, 1 << 16, (count, words), key=part_keys[1]).astype(mx.uint32)
        part = (high << 16) | low
        mx.eval(part)
        parts.append(part)
    weight = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
    span = spread * (0.7 + 0.6 * mx.random.uniform(shape=(rows, columns // group), key=keys[1]))
    scales = (2.0 * span / float((1 << bits) - 1)).astype(mx.bfloat16)
    biases = (-span * (0.9 + 0.2 * mx.random.uniform(shape=(rows, columns // group), key=keys[2]))).astype(mx.bfloat16)
    mx.eval(weight, scales, biases)
    return weight, scales, biases


def _expert_block_inputs(experts: int, *, seed: int = 5):
    """Random packs of one expert block at the real shapes (``experts`` routed experts)."""

    import mlx.core as mx

    from mtplx.kernels import moe_shared

    keys = mx.random.split(mx.random.key(seed), 12)
    pack = moe_shared._pack if experts <= 64 else _big_pack
    data = {
        "x": mx.random.normal((4, 2560), key=keys[0]).astype(mx.bfloat16),
        "router": pack(keys[1], 512, 2560, bits=8, group=64, spread=0.05),
        "gate": pack(keys[2], 1, 2560, bits=8, group=64, spread=0.05),
        "gu": pack(keys[3], 1280, 2560, bits=8, group=64, spread=0.03),
        "shared_down": pack(keys[4], 2560, 640, bits=8, group=64, spread=0.05),
        "routed_gu": pack(keys[5], experts * 1280, 2560, bits=4, group=32, spread=0.03),
        "down": pack(keys[6], experts * 2560, 640, bits=4, group=32, spread=0.05),
        "hyper": mx.random.normal((1, 4, 4 * 2560), key=keys[7]).astype(mx.bfloat16),
        "inject": mx.random.normal((1, 4, 4), key=keys[8]).astype(mx.bfloat16),
    }
    return data, keys


@case("moe_shared")
def _case_moe_shared() -> dict[str, Any]:
    """One expert block at 4 rows: the seven-launch form against launch cut moe_shared."""

    import mlx.core as mx

    from mtplx.kernels import moe_shared
    from mtplx.kernels import qwen4_m4_route as route
    from mtplx.kernels import qwen4_m4_routed_down as routed_down
    from mtplx.kernels import qwen4_m4_routed_glu as routed_glu

    experts = 256  # half the real pack's 512: the routed kernels' reads are spread as widely as the memory allows here
    data, keys = _expert_block_inputs(experts)
    # 40 distinct experts, as a forward with no overlap between its rows reads.
    ids = mx.random.permutation(experts, key=keys[9])[:40].reshape(4, 10).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((4, 10), key=keys[10]), axis=-1).astype(mx.bfloat16)
    table = moe_shared.silu_table(True)
    stock_route = route.bind(vec_lanes=route.DEFAULT_VEC_LANES)
    glu = routed_glu.bind()
    stock_tail = routed_down.bind_residual_tail()
    router, gate, gu, shared_down = data["router"], data["gate"], data["gu"], data["shared_down"]
    routed_gu, down = data["routed_gu"], data["down"]
    hyper, inject = data["hyper"], data["inject"]

    def seven(x, ids, scores):
        _ids, _scores, factor = stock_route(x, *router, *gate)
        routed_h = glu(x, *routed_gu, ids)
        shared_h = moe_shared._stock_shared_h(x, gu, False).reshape(1, 4, 640)
        shared = moe_shared._stock_shared_down(shared_h, shared_down)
        return stock_tail(routed_h, *down, ids, scores, shared, factor, hyper, inject), _ids

    def four(x, ids, scores):
        _ids, _scores, factor, shared_h = moe_shared.route_shared(
            x, router, gate, gu, table, vec_lanes=route.DEFAULT_VEC_LANES
        )
        routed_h = glu(x, *routed_gu, ids)
        return moe_shared.routed_down_shared(routed_h, down, ids, scores, shared_h, shared_down, factor, hyper, inject), _ids

    def five(x, ids, scores):
        # The router half of the cut alone: the shared down product stays MLX's.
        _ids, _scores, factor, shared_h = moe_shared.route_shared(
            x, router, gate, gu, table, vec_lanes=route.DEFAULT_VEC_LANES
        )
        routed_h = glu(x, *routed_gu, ids)
        shared = moe_shared._stock_shared_down(shared_h, shared_down)
        return stock_tail(routed_h, *down, ids, scores, shared, factor, hyper, inject), _ids

    return {
        "arms": {
            "seven launches (shipped)": mx.compile(seven),
            "five launches (router half of moe_shared only)": mx.compile(five),
            "four launches (moe_shared)": mx.compile(four),
        },
        "inputs": [data["x"], ids, scores],
        "pinned": [*router, *gate, *gu, *shared_down, *routed_gu, *down, hyper, inject, table],
        "note": f"{experts} routed experts in the pack, 40 distinct ones read; expert ids are fixed inputs so both arms read the same experts",
    }


def _overlap_ids(experts: int, distinct: int, key):
    """Expert ids ``[4, 10]`` with ``distinct`` different experts among the 40 slots.

    Each row's ten are different from one another (a row never picks an expert
    twice); rows share experts once ``distinct`` is under 40.
    """

    import mlx.core as mx

    pool = [int(v) for v in mx.random.permutation(experts, key=key)[:distinct].tolist()]
    rows = []
    for row in range(4):
        # Row r takes ten consecutive pool entries starting where the pool
        # still has unused ones, wrapping onto experts earlier rows used.
        start = (row * 10) % distinct if distinct >= 10 else 0
        rows.append([pool[(start + slot) % distinct] for slot in range(10)])
    return mx.array(rows, dtype=mx.uint32)


@case("expert_overlap")
def _case_expert_overlap() -> dict[str, Any]:
    """The two routed-expert kernels as shipped, at 40, 30, 20 and 10 distinct experts in a forward."""

    import mlx.core as mx

    from mtplx.kernels import qwen4_m4_routed_down as routed_down
    from mtplx.kernels import qwen4_m4_routed_glu as routed_glu

    experts = 128
    data, keys = _expert_block_inputs(experts)
    scores = mx.softmax(mx.random.normal((4, 10), key=keys[10]), axis=-1).astype(mx.bfloat16)
    shared = mx.random.normal((4, 2560), key=keys[11]).astype(mx.bfloat16)
    factor = mx.array([0.5, 0.25, 0.75, 0.6]).astype(mx.bfloat16)
    glu = routed_glu.bind()
    tail = routed_down.bind_residual_tail()
    routed_gu, down, hyper, inject = data["routed_gu"], data["down"], data["hyper"], data["inject"]
    arms = {}
    for distinct in (40, 30, 20, 10):
        ids = _overlap_ids(experts, distinct, keys[9])

        def arm(x, ids=ids):
            routed_h = glu(x, *routed_gu, ids)
            return tail(routed_h, *down, ids, scores, shared, factor, hyper, inject)

        arms[f"{distinct} distinct experts"] = arm
    return {
        "arms": arms,
        "inputs": [data["x"]],
        "pinned": [*routed_gu, *down, hyper, inject, scores, shared, factor],
        "note": f"{experts} experts in the pack; the shipped kernels read each (row, expert) slot separately",
    }


@case("accept_sample")
def _case_accept_sample() -> dict[str, Any]:
    """The accept step's sampler on a verify forward's logits (4 rows x 248,320, native sampler)."""

    import mlx.core as mx

    from mtplx.fast_sampling import sample_token_ids_from_mlx_logits
    from mtplx.sampling import SamplerConfig

    native = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    logits = (mx.random.normal((1, 4, 248_320), key=mx.random.key(9)) * 3.0).astype(mx.bfloat16)

    def shipped(rows):
        mx.random.seed(0)
        return sample_token_ids_from_mlx_logits(rows, native)

    return {
        "arms": {"shipped sampler (full sort of the vocabulary per row)": shipped},
        "inputs": [logits],
        "pinned": [],
        "note": "random logits; the sort's work does not depend on their values",
    }


def _shipped_support(logits, *, k: int = 20, temperature: float = 1.0):
    """The device half of the product's top-k support read, statement for statement
    (``fast_sampling._device_serial_support_arrays_relaxed_ties`` before its host arithmetic)."""

    import mlx.core as mx

    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    scaled = rows * (1.0 / float(temperature))
    cand_idx = mx.argpartition(-scaled, kth=k - 1, axis=-1)[:, :k]
    cand_vals = mx.take_along_axis(scaled, cand_idx, axis=-1)
    log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
    cand_probs = mx.exp(cand_vals - log_total)
    return cand_idx, cand_vals, cand_probs


@case("topk_support")
def _case_topk_support() -> dict[str, Any]:
    """The top-20 support read after a verify forward (4 x 248,320) and after a draft step (1 x 65,536)."""

    import mlx.core as mx

    keys = mx.random.split(mx.random.key(17), 2)
    verify = (mx.random.normal((1, 4, 248_320), key=keys[0]) * 3.0).astype(mx.bfloat16)
    draft = (mx.random.normal((1, 1, 65_536), key=keys[1]) * 3.0).astype(mx.bfloat16)
    arms: dict[str, Any] = {
        "shipped, verify logits 4 x 248,320": lambda v, d: _shipped_support(v),
        "shipped, draft logits 1 x 65,536": lambda v, d: _shipped_support(d),
    }
    try:
        from mtplx.kernels import topk_select
    except ImportError:
        topk_select = None
    if topk_select is not None:
        arms["selection kernels, verify logits 4 x 248,320"] = lambda v, d: topk_select.support(v, k=20, temperature=1.0)
        arms["selection kernels, draft logits 1 x 65,536"] = lambda v, d: topk_select.support(d, k=20, temperature=1.0)
    return {
        "arms": arms,
        "inputs": [verify, draft],
        "pinned": [],
        "note": "random logits; a sort's work does not depend on the values",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU time of a kernel or block at the real shapes.")
    parser.add_argument("case", help="a case name, or 'list'")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if args.case == "list":
        for name, fn in CASES.items():
            print(f"{name}: {(fn.__doc__ or '').strip()}")
        return 0
    if args.case not in CASES:
        parser.error(f"unknown case {args.case!r}; try 'list'")
    from mtplx.engine import runtime
    from mtplx.engine.tools import census

    census.refuse_if_held()
    runtime.available()  # before any kernel is compiled: the executor only knows kernels built after it loads
    built = CASES[args.case]()
    result = time_arms(built["arms"], built["inputs"], pinned=built.get("pinned"), repeats=args.repeats)
    result["case"] = args.case
    result["note"] = built.get("note", "")
    print(render(args.case, result))
    if result["note"]:
        print(f"\n({result['note']})")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
