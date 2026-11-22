"""Issue #706: Inkling's launches above 256 threads at the widths a Mac serves, on synthetic operands, and what this
GPU refuses. For the M1 runner job in .github/workflows/kernel-matrix.yml.

    python tests/inkling_launch_census.py OUT.json
    python tests/inkling_launch_census.py --summary CENSUS.json [CENSUS.json ...]   (Markdown for the job summary)

It runs against any MTPLX tree. On 2.13.0 a launch this GPU refuses raises MLX's thread-group ValueError when it is
evaluated (recorded "refused"): that is the bug, the model does not load. A tree with ``launch_limits`` runs such a
launch in its same-bits form or declines it at the call with ``GroupRefused`` (recorded "declined": its stage serves
its reference route), and ``launch_limits.report()`` lists what this GPU refused and its limit. Exit 1 when a
thread-group error escapes on a tree that has ``launch_limits``; 0 otherwise (a refusal on 2.13.0 is the finding).

Covered (the recorded serve-path launches above 256 threads, Inkling's shapes: hidden 2,304, 24 query heads over 8
key/value heads of 128, window 1,024): the attention of window and global visits at 1 to 32 rows (its partials of
32 x rows threads and its 1,024-thread merge), the one-threadgroup partials of global visits wider than 8 rows, the
expert visit at 1 to 32 rows in both expert formats (its combine of 32 x rows threads, its 384-thread products), the
576-thread read and norm launches of the decode stages, the drafter and the prompt loop; bf16 and float16 streams.
"""

from __future__ import annotations

import json
import sys

ROWS = (1, 8, 9, 16, 22, 23, 24, 25, 32)


def _census():
    import numpy as np
    import mlx.core as mx

    try:
        from mtplx.models.inkling import launch_limits
    except ImportError:
        launch_limits = None
    results = []

    def run(name, make):
        try:
            mx.eval(make())
            results.append({"launch": name, "outcome": "ran"})
        except ValueError as exc:
            if "Thread group size" not in str(exc):
                raise
            results.append({"launch": name, "outcome": "refused", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - recorded: the census is about thread groups, not routes
            if launch_limits is not None and (isinstance(exc, launch_limits.GroupRefused)
                                              or "is above what this GPU runs" in str(exc)):
                # refused at the call (or by the route's own first-use qualification, which it reaches through)
                results.append({"launch": name, "outcome": "declined", "error": f"{type(exc).__name__}: {exc}"})
            else:  # a route this Mac does not serve (its own qualification declines): not a thread-group finding
                results.append({"launch": name, "outcome": "not served", "error": f"{type(exc).__name__}: {exc}"})

    def rnd(shape, key, scale=1.0):
        return mx.random.normal(shape, key=key) * scale

    def clipped(shape, limit, key):
        v = rnd(shape, key, 2.2)
        n = mx.linalg.norm(v, axis=-1, keepdims=True)
        return v * mx.minimum(1.0, limit / (n + 1e-6))

    from mtplx.models.inkling.fast.attend import attend

    for dtype in (mx.bfloat16, mx.float16):
        tag = "bf16" if dtype == mx.bfloat16 else "fp16"
        for rows in ROWS:
            ks = mx.random.split(mx.random.key(706 + rows), 8)
            history = 1500
            k = clipped((1, 8, history + rows, 128), 22.6, ks[0]).astype(dtype)
            v = mx.fast.rms_norm(rnd((1, 8, history + rows, 128), ks[1]), None, 1e-6).astype(dtype)
            q = clipped((1, rows, 24, 128), 22.6, ks[2]).astype(dtype)
            rel = clipped((1, rows, 24, 16), 8.0, ks[3]).astype(dtype)
            sink = rnd((1, rows, 24), ks[4], 3.0).astype(dtype)
            gate = rnd((1, rows, 3072), ks[5], 2.0).astype(dtype)
            basis = rnd((16, 1024), ks[6], 0.3).astype(mx.bfloat16)
            ring_k = mx.zeros((1, 8, 1024, 128), dtype)
            ring_v = mx.zeros((1, 8, 1024, 128), dtype)
            slots = mx.array((np.arange(history - 1024, history) + 77) % 1024, mx.int32)
            ring_k[:, :, slots, :] = k[:, :, history - 1024:history]
            ring_v[:, :, slots, :] = v[:, :, history - 1024:history]
            bank_k = mx.zeros((1, 8, 2048, 128), dtype)
            bank_v = mx.zeros((1, 8, 2048, 128), dtype)
            bank_k[:, :, :history] = k[:, :, :history]
            bank_v[:, :, :history] = v[:, :, :history]
            panel = (q, rel, sink, gate, k[:, :, history:], v[:, :, history:])
            newest = (history - 1 + 77) % 1024
            run(f"attention window {rows} rows {tag}",
                lambda: attend(*panel, ring_k, ring_v, 1024, newest, history, basis, window=True))
            run(f"attention global {rows} rows {tag}",
                lambda: attend(*panel, bank_k, bank_v, history, history, history, basis, window=False))
            if rows > 8:
                run(f"attention global {rows} rows one threadgroup {tag}",
                    lambda: attend(*panel, bank_k, bank_v, history, history, history, basis, window=False,
                                   grouped=False))

    from mtplx.models.inkling import model as M
    from mtplx.models.inkling.fast import experts as fx

    experts = 16

    def synthetic_layer(mode, dtype):
        fp4 = mode == "nvfp4"
        keys = iter(mx.random.split(mx.random.key(7), 16))
        codes = lambda shape, lo, hi: mx.random.randint(lo, hi, shape, key=next(keys)).astype(mx.uint8)  # noqa: E731
        router = (rnd((128, 2304), next(keys)) * 0.02).astype(mx.bfloat16)
        bias = mx.concatenate([rnd((experts,), next(keys), 0.01), mx.full((128 - experts,), -100.0)]).astype(mx.float32)
        w = {"mlp.gate.weight": router, "mlp.gate.bias": bias}
        for stem, (out, kk) in (("w13", (2304, 2304)), ("w2", (2304, 1152))):
            name = f"mlp.experts.{stem}_weight"
            if fp4:
                w[name] = codes((experts, out, kk // 2), 0, 256)
                w[name + "_scale"] = (codes((experts, out, kk // 16), 3, 9) * 8
                                      + codes((experts, out, kk // 16), 0, 8)).astype(mx.uint8)
                w[name + "_amax"] = mx.random.uniform(1, 10, (experts,), key=next(keys))
                w[name + "_outer_scale_denominator"] = mx.array([2688.0])
            else:
                w[name] = codes((experts, out, kk), 0, 256)
                w[name + "_scale"] = codes((experts, out, kk // 32), 118, 126)
        cfg = {"num_experts_per_tok": 8, "route_scale": 8.0, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0}
        block = M.Experts(w, cfg, mode)
        conv = M.Convolution({"mlp_sconv.weight": (rnd((2304, 1, 4), next(keys)) * 0.5).astype(dtype)}, "mlp_sconv")
        norm_weight = (1 + 0.1 * rnd((2304,), next(keys))).astype(dtype)
        return block, conv, norm_weight

    for mode in ("nvfp4", "mxfp8"):
        for dtype in (mx.bfloat16, mx.float16):
            tag = "bf16" if dtype == mx.bfloat16 else "fp16"
            block, conv, norm_weight = synthetic_layer(mode, dtype)
            ks = mx.random.split(mx.random.key(29), 3)
            source = rnd((1, 32, 2304), ks[0], 3.0)
            partial = rnd((1, 32, 2304), ks[1])
            tail = rnd((1, 3, 2304), ks[2])
            for rows in ROWS:
                run(f"experts {mode} {rows} rows {tag}",
                    lambda: fx.expert_visit_parts(block, norm_weight, conv.weight, source[:, :rows],
                                                  partial[:, :rows], tail, eps=1e-6, dtype=dtype))

    from mtplx.models.inkling.fast.read import read
    from mtplx.models.inkling.fast.draft_glue import add_norm
    from mtplx.models.inkling.fast.prompt_loop import input_norms

    for dtype in (mx.bfloat16, mx.float16):
        tag = "bf16" if dtype == mx.bfloat16 else "fp16"
        for rows in (1, 8, 25, 32):
            ks = mx.random.split(mx.random.key(rows), 6)
            partial = rnd((rows, 2304), ks[0], 2.0).astype(dtype)
            closed = [rnd((rows, 2304), ks[1 + i], 1.0 + i).astype(dtype) for i in range(2)]
            queries = rnd((1, 2304), ks[3], 0.05)
            gain = (1 + 0.1 * rnd((2304,), ks[4])).astype(mx.bfloat16)
            run(f"read and norm {rows} rows {tag}",
                lambda: read(partial, closed, queries, 1e-6, mode=0, gains=[gain.astype(dtype)]))
            a = rnd((1, rows, 2304), ks[0]).astype(dtype)
            b = rnd((1, rows, 2304), ks[5]).astype(dtype)
            run(f"drafter add and norm {rows} rows {tag}", lambda: add_norm(a, b, gain.astype(dtype), 1e-6))
        for rows in (33, 512):
            hs = [rnd((rows, 2304), mx.random.key(rows + i)) for i in range(3)]
            gains = [(1 + 0.1 * rnd((2304,), mx.random.key(9 + i))).astype(mx.bfloat16) for i in range(3)]
            run(f"prompt loop norms {rows} rows {tag}", lambda: input_norms(hs, gains, 1e-6, dtype))

    return {
        "device": {k: str(v) for k, v in mx.device_info().items()},
        "mlx": mx.__version__,
        "tree_has_launch_limits": launch_limits is not None,
        "results": results,
        "launch_limits": launch_limits.report() if launch_limits is not None else None,
    }


def _summary(paths):
    lines = ["## Inkling launches above 256 threads on this runner (issue #706)", ""]
    for path in paths:
        try:
            census = json.loads(open(path).read())
        except (OSError, ValueError) as exc:
            lines += [f"**{path}**: no census ({exc})", ""]
            continue
        results = census["results"]
        tree = "this tree" if census["tree_has_launch_limits"] else "2.13.0"
        refused = [r for r in results if r["outcome"] == "refused"]
        declined = [r for r in results if r["outcome"] == "declined"]
        unserved = [r for r in results if r["outcome"] == "not served"]
        ran = len(results) - len(refused) - len(declined) - len(unserved)
        lines.append(f"**{tree}** on `{census['device'].get('device_name', '?')}` "
                     f"({census['device'].get('architecture', '?')}), MLX {census['mlx']}: "
                     f"{len(results)} launches, {ran} ran, {len(refused)} refused by the GPU, "
                     f"{len(declined)} declined at the call, {len(unserved)} not served by that route on this Mac.")
        for r in refused + declined + unserved:
            lines.append(f"- {r['outcome']}: {r['launch']}: {r['error']}")
        report = census.get("launch_limits") or {}
        for entry in report.get("refused") or ():
            lines.append(f"- this GPU refuses `{entry['kernel']}` at {entry['group']} threads (its pipeline allows "
                         f"{entry['limit']}); {entry['served_by']}")
        if not census["tree_has_launch_limits"] and not refused:
            lines.append("- 2.13.0 ran every launch here: this runner does not reproduce issue #706.")
        lines.append("")
    return "\n".join(lines)


def main(argv):
    if argv[:1] == ["--summary"]:
        print(_summary(argv[1:]))
        return 0
    census = _census()
    with open(argv[0], "w") as fh:
        json.dump(census, fh, indent=1)
    print(_summary([argv[0]]))
    escaped = [r for r in census["results"] if r["outcome"] == "refused"]
    return 1 if census["tree_has_launch_limits"] and escaped else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
