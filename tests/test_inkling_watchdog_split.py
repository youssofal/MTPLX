"""CPU-only tests for #422: M1/M2 Inkling cold prefill key-axis splitting & command-buffer flushing.

Every test here runs strictly on ``mx.cpu`` / NumPy and never opens or
dispatches to Metal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.models.inkling.fast import prompt_flash as F  # noqa: E402
from mtplx.models.inkling.fast import prompt_route as PR  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu_device():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert mx.default_device() == mx.cpu
        yield
    finally:
        mx.set_default_device(prev)


# ---- GPU generation detection & split routing ------------------------------


def test_gpu_generation_parses_apple_gpu_families():
    assert F.gpu_generation("applegpu_g13g") == 13
    assert F.gpu_generation("applegpu_g13s") == 13
    assert F.gpu_generation("applegpu_g14g") == 14
    assert F.gpu_generation("applegpu_g14s") == 14
    assert F.gpu_generation("applegpu_g14d") == 14
    assert F.gpu_generation("applegpu_g15s") == 15
    assert F.gpu_generation("applegpu_g16s") == 16
    assert F.gpu_generation("applegpu_g17s") == 17
    assert F.gpu_generation(None) == 0 or isinstance(F.gpu_generation(None), int)
    assert F.gpu_generation("unknown") == 0


def test_key_split_plan_splits_m1_m2_past_budget_and_leaves_m5_untouched(monkeypatch):
    monkeypatch.delenv(F.KEY_SPLIT_ENV, raising=False)
    monkeypatch.delenv(F.WORK_BUDGET_ENV, raising=False)

    # M1 / M2: short/medium full attention stays single-launch up to 512 x 8192 = 4M.
    assert F.key_split_plan(512, 4096, 0, launch_form="simd", architecture="applegpu_g14s") == [(0, 128)]
    assert F.key_split_plan(512, 8192, 0, launch_form="simd", architecture="applegpu_g13s") == [(0, 256)]

    # M2 Max (g14s) cold 30K prefill at 512 rows: splits into 4 slices of <=8192 keys.
    plan_30k_512 = F.key_split_plan(512, 30000, 0, launch_form="simd", architecture="applegpu_g14s")
    assert len(plan_30k_512) == 4
    assert plan_30k_512 == [(0, 256), (256, 512), (512, 768), (768, 938)]

    # M2 Max (g14s) cold 30K prefill at default 2048 rows: 4M / 2048 = 2048 keys (64 tiles) -> 15 slices.
    plan_30k_2048 = F.key_split_plan(2048, 30000, 0, launch_form="simd", architecture="applegpu_g14s")
    assert len(plan_30k_2048) == 15
    assert plan_30k_2048[0] == (0, 64)
    assert plan_30k_2048[-1] == (896, 938)

    # 262K context at 512 rows on M1/M2: contiguous, non-overlapping coverage of all 8192 tiles.
    plan_262k = F.key_split_plan(512, 262144, 0, launch_form="simd", architecture="applegpu_g14d")
    assert len(plan_262k) == 32
    assert plan_262k[0][0] == 0 and plan_262k[-1][1] == 8192
    for prev_s, next_s in zip(plan_262k, plan_262k[1:]):
        assert prev_s[1] == next_s[0]

    # Window visits never split (keys attended <= 1024).
    assert len(F.key_split_plan(2048, 65536, 1024, launch_form="simd", architecture="applegpu_g14s")) == 1

    # M5 Max tensor-unit path stays 100% single-launch regardless of length.
    for rows, keys in ((512, 30000), (2048, 30000), (512, 65536), (2048, 262144)):
        plan_m5 = F.key_split_plan(rows, keys, 0, launch_form="tensor", architecture="applegpu_g17s")
        assert len(plan_m5) == 1

    # M3/M4 uses 16M / 16384 budget: unsplit at 512 x 16384, splits at 512 x 30000.
    assert len(F.key_split_plan(512, 16384, 0, launch_form="simd", architecture="applegpu_g16s")) == 1
    plan_m4_30k = F.key_split_plan(512, 30000, 0, launch_form="simd", architecture="applegpu_g16s")
    assert plan_m4_30k == [(0, 512), (512, 938)]


def test_key_split_plan_honors_env_overrides(monkeypatch):
    monkeypatch.setenv(F.KEY_SPLIT_ENV, "0")
    assert len(F.key_split_plan(2048, 65536, 0, launch_form="simd", architecture="applegpu_g14s")) == 1

    monkeypatch.setenv(F.KEY_SPLIT_ENV, "4096")
    monkeypatch.setenv(F.WORK_BUDGET_ENV, str(8 * 1024 * 1024))
    plan = F.key_split_plan(512, 16384, 0, launch_form="simd", architecture="applegpu_g14s")
    assert plan == [(0, 128), (128, 256), (256, 384), (384, 512)]


def test_visit_handover_policy_flushes_synchronously_on_m1_m2_only(monkeypatch):
    monkeypatch.delenv(PR.SYNC_ENV, raising=False)

    # Small decode / verify batches never hand over inside forward.
    assert PR.visit_handover_policy(1, architecture="applegpu_g14s", is_tensor=False) == (0, False)
    assert PR.visit_handover_policy(32, architecture="applegpu_g14s", is_tensor=False) == (0, False)

    # M5+ preserves existing async HAND_OVER_VISITS behavior unchanged.
    assert PR.visit_handover_policy(512, architecture="applegpu_g17s", is_tensor=True) == (
        PR.HAND_OVER_VISITS,
        False,
    )
    assert PR.visit_handover_policy(2048, architecture="applegpu_g17s", is_tensor=True) == (
        PR.HAND_OVER_VISITS,
        False,
    )
    monkeypatch.setenv(PR.SYNC_ENV, "1")
    assert PR.visit_handover_policy(2048, architecture="applegpu_g17s", is_tensor=True) == (0, False)

    # M1/M2 flushes synchronously every 2 visits at <=512 rows and every visit at >=1024 rows.
    assert PR.visit_handover_policy(512, architecture="applegpu_g14s", is_tensor=False) == (2, True)
    assert PR.visit_handover_policy(2048, architecture="applegpu_g13s", is_tensor=False) == (1, True)

    # M3/M4 keeps async handover (tighter at >=1024 rows) when async is enabled.
    monkeypatch.delenv(PR.SYNC_ENV, raising=False)
    assert PR.visit_handover_policy(512, architecture="applegpu_g16s", is_tensor=False) == (
        PR.HAND_OVER_VISITS,
        False,
    )
    assert PR.visit_handover_policy(2048, architecture="applegpu_g16s", is_tensor=False) == (2, False)


# ---- Online-softmax merge math & CPU kernel simulation ---------------------


def _cpu_partial_slice(q, k, v, prof, sink, window, t_lo, t_hi, include_sink):
    """Exact CPU reference for one key-tile slice [t_lo, t_hi) matching _SOURCE_SIMD_PARTIAL."""
    _, heads, rows, dim = q.shape
    kv_heads, keys = k.shape[1], k.shape[2]
    group = heads // kv_heads
    extent = prof.shape[3]
    hist = keys - rows
    log2e = 1.4426950408889634
    scale_log2 = (dim**-0.5) * log2e

    t_end_all = (keys - 1) // 32 + 1
    t_begin = max(0, t_lo)
    t_end = min(t_end_all, t_hi)
    key_lo = t_begin * 32
    key_hi = min(keys, t_end * 32)

    q_f = q.astype(mx.float32)
    k_rep = mx.repeat(k.astype(mx.float32), group, axis=1)
    v_rep = mx.repeat(v.astype(mx.float32), group, axis=1)
    prof_f = prof.astype(mx.float32).transpose(0, 2, 1, 3) * log2e  # (1, heads, rows, extent)
    sink_log2 = sink.astype(mx.float32).transpose(0, 2, 1) * log2e  # (1, heads, rows)

    if key_lo < key_hi:
        k_sl = k_rep[:, :, key_lo:key_hi, :]
        v_sl = v_rep[:, :, key_lo:key_hi, :]
        scores = (q_f @ k_sl.transpose(0, 1, 3, 2)) * scale_log2
        pos = hist + mx.arange(rows)[:, None]
        d = pos - mx.arange(key_lo, key_hi)[None, :]
        bias = mx.take_along_axis(
            prof_f,
            mx.broadcast_to(mx.clip(d, 0, extent - 1)[None, None], (1, heads, rows, key_hi - key_lo)),
            axis=-1,
        )
        bias = mx.where(d >= extent, 0.0, bias)
        seen = (d >= 0) if not window else ((d >= 0) & (d < window))
        logits = mx.where(seen, scores + bias, -mx.inf)
        row_max = mx.max(logits, axis=-1)
    else:
        logits = None
        v_sl = None
        row_max = mx.full((1, heads, rows), -mx.inf, mx.float32)

    if include_sink:
        part_max = mx.maximum(row_max, sink_log2)
    else:
        part_max = row_max

    finite = mx.isfinite(part_max)
    safe_max = mx.where(finite, part_max, 0.0)
    ln2 = 0.6931471805599453
    if logits is not None:
        weights = mx.where(finite[..., None], mx.exp((logits - safe_max[..., None]) * ln2), 0.0)
        part_sum = weights.sum(axis=-1)
        part_acc = weights @ v_sl
    else:
        part_sum = mx.zeros((1, heads, rows), mx.float32)
        part_acc = mx.zeros((1, heads, rows, dim), mx.float32)

    if include_sink:
        part_sum = part_sum + mx.exp((sink_log2 - safe_max) * ln2)

    return (
        part_acc.transpose(0, 2, 1, 3),
        part_max.transpose(0, 2, 1),
        part_sum.transpose(0, 2, 1),
    )


def test_merge_online_softmax_handles_empty_inf_slices_without_nan():
    dim = 4
    acc_valid = mx.array([[[[2.0, -4.0, 1.0, 0.5]]]], mx.float32)
    max_valid = mx.array([[[3.0]]], mx.float32)
    sum_valid = mx.array([[[2.5]]], mx.float32)

    acc_empty = mx.zeros((1, 1, 1, dim), mx.float32)
    max_empty = mx.array([[[-mx.inf]]], mx.float32)
    sum_empty = mx.array([[[0.0]]], mx.float32)

    for a, b in (
        ((acc_valid, max_valid, sum_valid), (acc_empty, max_empty, sum_empty)),
        ((acc_empty, max_empty, sum_empty), (acc_valid, max_valid, sum_valid)),
    ):
        merged_acc, merged_max, merged_sum = F.merge_online_softmax(*a, *b)
        mx.eval(merged_acc, merged_max, merged_sum)
        assert np.allclose(np.array(merged_acc), np.array(acc_valid))
        assert float(merged_max.item()) == pytest.approx(3.0)
        assert float(merged_sum.item()) == pytest.approx(2.5)

    both_acc, both_max, both_sum = F.merge_online_softmax(
        acc_empty, max_empty, sum_empty, acc_empty, max_empty, sum_empty
    )
    mx.eval(both_acc, both_max, both_sum)
    assert np.all(np.isfinite(np.array(both_acc)))
    assert np.all(np.array(both_acc) == 0.0)
    assert math.isinf(float(both_max.item())) and float(both_max.item()) < 0
    assert float(both_sum.item()) == 0.0


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_multi_slice_online_softmax_matches_unsplit_reference_on_cpu(dtype_name):
    dtype = getattr(mx, dtype_name)
    rng = np.random.default_rng(20261123 + (0 if dtype_name == "float16" else 1))
    rows, keys, heads, kv_heads, dim, extent = 37, 290, 24, 8, 128, 64

    def rand_normed(shape, scale):
        arr = rng.normal(0.0, scale, size=shape).astype(np.float32)
        return mx.array(arr).astype(dtype)

    q = rand_normed((1, heads, rows, dim), 0.3)
    k = rand_normed((1, kv_heads, keys, dim), 0.3)
    v = rand_normed((1, kv_heads, keys, dim), 0.5)
    prof = rand_normed((1, rows, heads, extent), 0.4)
    sink = rand_normed((1, rows, heads), 0.8)

    # Deliberately force 5 slices including an empty slice past t_end_all, causal
    # partial tiles, and profile-extent boundary crossings.
    splits = [(0, 2), (2, 4), (4, 7), (7, 10), (10, 12)]
    acc = run_max = run_sum = None
    for idx, (t_lo, t_hi) in enumerate(splits):
        p_acc, p_max, p_sum = _cpu_partial_slice(
            q, k, v, prof, sink, 0, t_lo, t_hi, include_sink=(idx == 0)
        )
        if acc is None:
            acc, run_max, run_sum = p_acc, p_max, p_sum
        else:
            acc, run_max, run_sum = F.merge_online_softmax(
                acc, run_max, run_sum, p_acc, p_max, p_sum
            )
        mx.eval(acc, run_max, run_sum)

    got_f32 = (acc / run_sum[..., None]).transpose(0, 2, 1, 3)
    got_q = got_f32.astype(dtype).astype(mx.float32)
    ref_f32 = F.reference(q, k, v, prof, sink, 0)
    mx.eval(got_f32, got_q, ref_f32)

    got_np = np.array(got_f32, dtype=np.float64)
    got_q_np = np.array(got_q, dtype=np.float64)
    ref_np = np.array(ref_f32, dtype=np.float64)

    # Independent float64 NumPy reference (unsplit, exact math) cross-check.
    q64 = np.array(q.astype(mx.float32), dtype=np.float64)
    k64 = np.repeat(np.array(k.astype(mx.float32), dtype=np.float64), heads // kv_heads, axis=1)
    v64 = np.repeat(np.array(v.astype(mx.float32), dtype=np.float64), heads // kv_heads, axis=1)
    prof64 = np.array(prof.astype(mx.float32), dtype=np.float64).transpose(0, 2, 1, 3)
    sink64 = np.array(sink.astype(mx.float32), dtype=np.float64).transpose(0, 2, 1)[..., None]
    scores64 = np.matmul(q64, k64.transpose(0, 1, 3, 2)) * (dim**-0.5)
    pos64 = (keys - rows) + np.arange(rows)[:, None]
    d64 = pos64 - np.arange(keys)[None, :]
    bias64 = np.take_along_axis(
        prof64, np.broadcast_to(np.clip(d64, 0, extent - 1)[None, None], (1, heads, rows, keys)), axis=-1
    )
    bias64 = np.where(d64 >= extent, 0.0, bias64)
    logits64 = np.where(d64 >= 0, scores64 + bias64, -np.inf)
    top64 = np.maximum(np.max(logits64, axis=-1, keepdims=True), sink64)
    w64 = np.exp(logits64 - top64)
    denom64 = w64.sum(axis=-1, keepdims=True) + np.exp(sink64 - top64)
    ref64 = np.matmul(w64, v64) / denom64

    assert np.all(np.isfinite(got_np))
    rel_f32 = np.linalg.norm(got_np - ref_np) / np.linalg.norm(ref_np)
    rel_np64 = np.linalg.norm(got_np - ref64) / np.linalg.norm(ref64)
    rel_q = np.linalg.norm(got_q_np - ref64) / np.linalg.norm(ref64)
    assert rel_f32 <= 1e-5
    assert rel_np64 <= 1e-5
    assert rel_q <= (5e-4 if dtype_name == "float16" else 3e-3)


# ---- Kernel source structural check & flash_attention CPU orchestration ----


def test_partial_simd_kernel_source_is_well_formed():
    src = F._SOURCE_SIMD_PARTIAL
    assert src.count("{") == src.count("}")
    assert src.count("(") == src.count(")")
    assert "int(params[2])" in src and "int(params[3])" in src and "int(params[4])" in src
    assert "out_acc" in src and "out_max" in src and "out_sum" in src
    assert "INK_GATE_ROW" not in src and "INK_RESULT" not in src


@pytest.mark.parametrize("gated", [False, True])
def test_flash_attention_routes_split_slices_and_flushes_eval_per_slice(monkeypatch, gated):
    rng = np.random.default_rng(77 if not gated else 88)
    rows, keys, heads, kv_heads, dim, extent = 40, 160, 24, 8, 128, 32
    dtype = mx.float16

    q = mx.array(rng.normal(0.0, 0.3, (1, heads, rows, dim)).astype(np.float32)).astype(dtype)
    k = mx.array(rng.normal(0.0, 0.3, (1, kv_heads, keys, dim)).astype(np.float32)).astype(dtype)
    v = mx.array(rng.normal(0.0, 0.4, (1, kv_heads, keys, dim)).astype(np.float32)).astype(dtype)
    prof = mx.array(rng.normal(0.0, 0.3, (1, rows, heads, extent)).astype(np.float32)).astype(dtype)
    sink = mx.array(rng.normal(0.0, 0.6, (1, rows, heads)).astype(np.float32)).astype(dtype)
    gate = (
        mx.array(rng.normal(0.0, 0.5, (1, rows, heads * dim)).astype(np.float32)).astype(dtype)
        if gated
        else None
    )

    monkeypatch.setattr(F, "form", lambda: "simd")
    monkeypatch.setattr(F, "gpu_generation", lambda arch=None: 14)
    # Trigger splitting past 64 keys at 40 rows: 40 * 160 = 6400 > 2560, slice = 2 tiles (64 keys).
    monkeypatch.setenv(F.WORK_BUDGET_ENV, "2560")
    monkeypatch.setenv(F.KEY_SPLIT_ENV, "64")
    monkeypatch.setattr(F, "MIN_SLICE_KEYS", 64)

    partial_calls: list[tuple[int, int, int]] = []

    def fake_partial_kernel():
        def invoke(*, inputs, template, **_kwargs):
            q_in, k_in, v_in, p_in, s_in, params = inputs
            t_dict = dict(template)
            _, _, t_lo, t_hi, inc_sink = [int(x) for x in params.tolist()]
            partial_calls.append((t_lo, t_hi, inc_sink))
            return _cpu_partial_slice(
                q_in, k_in, v_in, p_in, s_in, t_dict["WINDOW"], t_lo, t_hi, bool(inc_sink)
            )

        return invoke

    eval_calls = {"count": 0}
    real_eval = mx.eval

    def counting_eval(*args, **kwargs):
        eval_calls["count"] += 1
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(F, "_kernel_simd_partial", fake_partial_kernel)
    monkeypatch.setattr(
        F,
        "_kernel_simd",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("should take split path")),
    )
    monkeypatch.setattr(mx, "eval", counting_eval)

    got = F.flash_attention(q, k, v, prof, sink, 0, gate=gate)
    real_eval(got)

    assert partial_calls == [(0, 2, 1), (2, 4, 0), (4, 5, 0)]
    assert eval_calls["count"] >= len(partial_calls)

    ref = F.reference(q, k, v, prof, sink, 0)
    if gated:
        expected = (
            ref.transpose(0, 2, 1, 3).reshape(1, rows, heads * dim)
            * mx.sigmoid(gate.astype(mx.float32))
        ).astype(dtype)
        assert got.shape == (1, rows, heads * dim)
    else:
        expected = ref.astype(dtype)
        assert got.shape == (1, heads, rows, dim)
    real_eval(expected)

    got_np = np.array(got.astype(mx.float32), dtype=np.float64)
    exp_np = np.array(expected.astype(mx.float32), dtype=np.float64)
    rel = np.linalg.norm(got_np - exp_np) / np.linalg.norm(exp_np)
    assert rel <= 5e-4


def test_flash_attention_tensor_form_never_routes_to_split_kernel(monkeypatch):
    monkeypatch.setattr(F, "form", lambda: "tensor")
    called: list[str] = []

    def fake_tensor_kernel(gated=False):
        def invoke(*, output_shapes, output_dtypes, **_kwargs):
            called.append(f"tensor:{gated}")
            return (mx.zeros(output_shapes[0], output_dtypes[0]),)

        return invoke

    monkeypatch.setattr(F, "_kernel", fake_tensor_kernel)
    monkeypatch.setattr(
        F,
        "_kernel_simd_partial",
        lambda: (_ for _ in ()).throw(AssertionError("tensor form must never split")),
    )

    rows, keys, heads, kv_heads, dim = 64, 128, 24, 8, 128
    q = mx.zeros((1, heads, rows, dim), mx.bfloat16)
    k = mx.zeros((1, kv_heads, keys, dim), mx.bfloat16)
    v = mx.zeros((1, kv_heads, keys, dim), mx.bfloat16)
    prof = mx.zeros((1, rows, heads, 16), mx.bfloat16)
    sink = mx.zeros((1, rows, heads), mx.bfloat16)

    out = F.flash_attention(q, k, v, prof, sink, 0)
    assert called == ["tensor:False"]
    assert out.shape == (1, heads, rows, dim)


def test_model_forward_flushes_sync_on_m1_m2_and_async_on_m5(monkeypatch):
    from types import SimpleNamespace

    from mtplx.models.inkling import model as inkling_model

    fake = SimpleNamespace(
        start=99,
        end=99,
        visit_order=[0, 1, 2, 3, 4],
        eps=1e-6,
        config={"attn_res_normalize_values": False},
        final_query=mx.zeros((1, 1, 8), mx.float32),
        final_norm=mx.ones((8,), mx.float32),
        layers=[lambda x, *_a: x + 1.0 for _ in range(5)],
    )

    monkeypatch.setattr(
        inkling_model,
        "depth_reader",
        lambda *_a: (lambda _c, cur, *_r: cur[..., None, :]),
    )

    calls: list[str] = []
    real_eval = mx.eval
    real_async_eval = mx.async_eval

    def rec_eval(*args, **kwargs):
        calls.append("eval")
        return real_eval(*args, **kwargs)

    def rec_async(*args, **kwargs):
        calls.append("async_eval")
        return real_async_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", rec_eval)
    monkeypatch.setattr(mx, "async_eval", rec_async)

    cache = [None] * (2 * len(fake.visit_order))
    emb = mx.zeros((1, 512, 8), mx.float16)

    # M2 Max (g14s, 512 rows): handover every 2 visits synchronously (visits 2 & 4).
    monkeypatch.setattr(
        inkling_model,
        "visit_handover_policy",
        lambda rows: PR.visit_handover_policy(rows, architecture="applegpu_g14s", is_tensor=False),
    )
    calls.clear()
    inkling_model.Model.forward(fake, None, cache, embeddings=emb)
    assert calls == ["eval", "eval"]

    # M5 Max (g17s, 512 rows): handover every 4 visits asynchronously (visit 4).
    monkeypatch.setattr(
        inkling_model,
        "visit_handover_policy",
        lambda rows: PR.visit_handover_policy(rows, architecture="applegpu_g17s", is_tensor=True),
    )
    calls.clear()
    inkling_model.Model.forward(fake, None, cache, embeddings=emb)
    assert calls == ["async_eval"]
