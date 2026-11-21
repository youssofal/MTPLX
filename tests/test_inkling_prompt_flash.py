"""One-launch prompt attention (``fast.prompt_flash``): numbers against the definition, the route switch.

The launch computes a prompt call's attention (the relative bias from the profile below its
extent, the sink as the softmax's first entry, window visits) in another order of the same sums
than the default prompt route, so it is held to the definition computed in float32, within the
rounding the query type itself puts on a result, on the shapes the prompt read makes: the first
chunk, a history crossing the bias extent, window visits that start at key 0 and later, a short
last chunk. The route is off unless ``MTPLX_INKLING_PROMPT_FLASH`` is on; every selection and
decline is counted.
"""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the prompt attention needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import attention as A  # noqa: E402
from mtplx.models.inkling.fast import prompt_flash as F  # noqa: E402

if not F.tensor_units():  # pragma: no cover - Macs without tensor units
    pytest.skip("the launch is written for the tensor units", allow_module_level=True)

# Relative RMS against the float32 definition: the query type's own rounding of a result is
# 1.7e-3 (bf16) and 2.4e-4 (fp16) relative RMS on these distributions (lanes/p2/fcheck-3).
BOUND = {mx.bfloat16: 2.5e-3, mx.float16: 4e-4}


def _clipped(shape, limit, key, dtype):
    v = mx.random.normal(shape, key=key) * 2.2
    n = mx.linalg.norm(v, axis=-1, keepdims=True)
    return (v * mx.minimum(1.0, limit / (n + 1e-6))).astype(dtype)


def _call(rows, keys, seed, dtype=mx.bfloat16):
    ks = mx.random.split(mx.random.key(seed), 5)
    q = _clipped((1, rows, 24, 128), 22.6, ks[0], dtype).transpose(0, 2, 1, 3)
    k = _clipped((1, 8, keys, 128), 22.6, ks[1], dtype)
    v = mx.fast.rms_norm(mx.random.normal((1, 8, keys, 128), key=ks[2]), None, 1e-6).astype(dtype)
    prof = _clipped((1, rows, 24, 1024), 8.0, ks[3], dtype)
    sink = (mx.random.normal((1, rows, 24), key=ks[4]) * 3.0).astype(dtype)
    mx.eval(q, k, v, prof, sink)
    return q, k, v, prof, sink


def _rel_rms(got, ref):
    d = got.astype(mx.float32) - ref
    return (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ref * ref))).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows,keys,window", [
    (64, 64, 0), (64, 64, 1024),          # the first chunk: every row sees only the chunk
    (64, 600, 1024),                      # window starts clamped at key 0
    (200, 2248, 0), (200, 2248, 1024),    # history past the bias extent; windows that slide
    (37, 1100, 0),                        # rows that are not a multiple of a block
    (46, 1069, 1024),                     # a short last chunk on a window visit
    (512, 1535, 1024),                    # the window visit of a 512-row chunk
])
def test_matches_the_definition(rows, keys, window, dtype):
    ops = _call(rows, keys, rows * 7 + keys + window, dtype)
    assert F.decline(*ops, window) is None
    got = F.flash_attention(*ops, window)
    ref = F.reference(*ops, window)
    assert got.shape == (1, 24, rows, 128) and got.dtype == dtype
    assert bool(mx.all(mx.isfinite(got)).item())
    assert _rel_rms(got, ref) <= BOUND[dtype]


def test_every_switch_gives_the_same_numbers_within_rounding():
    ops = _call(200, 2248, 5)
    ref = F.reference(*ops, 0)
    for kw in (dict(tg_m=6), dict(tg_m=12), dict(tgbar=False)):
        assert _rel_rms(F.flash_attention(*ops, 0, **kw), ref) <= BOUND[mx.bfloat16], kw


def test_long_history_global_visit():
    # Past the vector-kernel threshold, where the default route runs the decode-shaped twin.
    ops = _call(48, 20480, 11)
    assert _rel_rms(F.flash_attention(*ops, 0), F.reference(*ops, 0)) <= BOUND[mx.bfloat16]


def test_a_row_gets_the_same_bits_whichever_threadgroup_holds_it():
    # The arithmetic of a row does not depend on which block or threadgroup holds it, nor on the
    # barrier that keeps a threadgroup's blocks on one tile.
    ops = _call(200, 2248, 21)
    a = F.flash_attention(*ops, 0, tg_m=3)
    for kw in (dict(tg_m=6), dict(tg_m=12), dict(tgbar=False)):
        b = F.flash_attention(*ops, 0, **kw)
        assert bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()), kw


def test_the_products_come_from_the_installed_library():
    # The tensor-unit pieces are read from the headers MLX installs; without them the call declines.
    text, why = F.library_text()
    assert why is None and "struct BaseNAXFrag" in text
    assert "matmul2d_descriptor" not in F._SOURCE + F._HEADER     # none of our text declares a product


def test_declines_without_the_library_headers(monkeypatch):
    q, k, v, prof, sink = _call(64, 600, 3)
    monkeypatch.setattr(F, "library_text", lambda: (None, "missing:steel/gemm/nax.h"))
    assert F.decline(q, k, v, prof, sink, 0) == "library_headers"


def test_rows_that_are_not_packed_are_read_element_by_element():
    # Keys and values whose rows are not on 8-byte boundaries, and queries strided along the head.
    q, k, v, prof, sink = _call(96, 1300, 9)
    wide_k = mx.zeros((1, 8, 1300, 130), k.dtype)
    wide_k[..., 1:129] = k
    wide_v = mx.zeros((1, 8, 1300, 130), v.dtype)
    wide_v[..., 1:129] = v
    qs = mx.zeros((1, 24, 96, 256), q.dtype)
    qs[..., ::2] = q
    a = F.flash_attention(q, k, v, prof, sink, 0)
    b = F.flash_attention(qs[..., ::2], wide_k[..., 1:129], wide_v[..., 1:129], prof, sink, 0)
    assert bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def test_declines_what_it_does_not_serve():
    q, k, v, prof, sink = _call(64, 600, 3)
    assert F.decline(q.astype(mx.float32), k, v, prof, sink, 0) == "dtype"
    assert F.decline(q, k[:, :, :32], v[:, :, :32], prof, sink, 0) == "geometry"   # more rows than keys
    assert F.decline(q, k, v, prof[:, :, :, :512], sink[:, :, :12], 0) == "geometry"


def test_route_is_off_by_default_and_counted_when_on(monkeypatch):
    ops = _call(64, 600, 9)
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)
    before = A.attention_route_counts()
    today = A.attend(*ops, 0)
    after = A.attention_route_counts()
    assert not any(k.startswith("prompt_flash.") and after[k] != before.get(k, 0) for k in after)

    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "on")
    before = A.attention_route_counts()
    on = A.attend(*ops, 0)
    on_window = A.attend(*ops, 1024)
    after = A.attention_route_counts()
    assert after.get("prompt_flash.full.selected", 0) == before.get("prompt_flash.full.selected", 0) + 1
    assert after.get("prompt_flash.window.selected", 0) == before.get("prompt_flash.window.selected", 0) + 1
    ref = F.reference(*ops, 0)
    assert _rel_rms(on, ref) <= BOUND[mx.bfloat16]
    assert _rel_rms(today, ref) <= BOUND[mx.bfloat16]
    assert _rel_rms(on_window, F.reference(*ops, 1024)) <= BOUND[mx.bfloat16]


def test_a_decline_runs_the_default_route_and_is_counted(monkeypatch):
    ops = _call(64, 600, 13)
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "on")
    monkeypatch.setattr(F, "decline", lambda *a, **k: "geometry")
    before = A.attention_route_counts()
    out = A.attend(*ops, 0)
    after = A.attention_route_counts()
    assert after.get("prompt_flash.full.stock.geometry", 0) == before.get("prompt_flash.full.stock.geometry", 0) + 1
    assert _rel_rms(out, F.reference(*ops, 0)) <= BOUND[mx.bfloat16]


def test_a_failed_first_use_check_keeps_the_default_route(monkeypatch):
    ops = _call(64, 600, 17, mx.float16)
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "on")
    monkeypatch.setattr(A, "_flash_checks", {})

    def broken(*args, **kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(F, "flash_attention", broken)
    before = A.attention_route_counts()
    out = A.attend(*ops, 1024)
    after = A.attention_route_counts()
    key = "prompt_flash.window.stock.first_use_error"
    assert after.get(key, 0) == before.get(key, 0) + 1
    assert _rel_rms(out, F.reference(*ops, 1024)) <= BOUND[mx.float16]


# ---------------------------------------------------------------------------------------------
# The prompt stage: the attention inputs and the bias profile in one launch, the gate in the
# attention launch's store, against the layer's own path on a synthetic layer.
# ---------------------------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

from mtplx.models.inkling import model as M  # noqa: E402
from mtplx.models.inkling.cache import WindowCache  # noqa: E402

_CFG = {"qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0, "rms_norm_eps": 1e-6,
        "log_scaling_alpha": 0.1, "log_scaling_n_floor": 8192, "use_value_norm_no_gamma": True,
        "sliding_window_size": 1024}


def _layer(sliding, dtype, seed):
    ks = iter(mx.random.split(mx.random.key(seed), 16))

    def w(*shape, s=0.02):
        return (mx.random.normal(shape, key=next(ks)) * s).astype(dtype)

    convs = []
    for stem, dynamic in (("k", not sliding), ("v", not sliding), ("attn", False), ("mlp", False)):
        if dynamic:
            weights = {"c.weight_proj": w(8 * 4, 2304), "c.static_w": w(4, 1024, s=0.3)}
        else:
            channels = 1024 if stem in ("k", "v") else 2304
            weights = {"c.weight": w(channels, 1, 4, s=0.3)}
        convs.append(M.Convolution(weights, "c", dynamic))
    layer = SimpleNamespace(cfg=_CFG, sliding=sliding, eps=1e-6,
                            q=w(3072, 2304, s=0.05), r=w(384, 2304, s=0.05), k=w(1024, 2304, s=0.05),
                            v=w(1024, 2304, s=0.05), sink=w(24, 2304, s=0.05), gate=w(3072, 2304, s=0.05),
                            out=w(2304, 3072), q_scale=(mx.random.uniform(0.5, 1.5, (24, 128), key=next(ks))).astype(dtype),
                            r_scale=(mx.random.uniform(0.5, 1.5, (24, 16), key=next(ks))).astype(dtype),
                            rel_basis=w(16, 1024, s=0.3), convs=convs)
    mx.eval(layer.q, layer.r, layer.k, layer.v, layer.sink, layer.gate, layer.out, layer.q_scale,
            layer.r_scale, layer.rel_basis)
    return layer


def _caches(layer, hist, dtype, rows):
    if layer.sliding:
        kv = WindowCache(max_size=1024)
        if hist:
            k0 = (mx.random.normal((1, 8, 1023 + rows, 128), key=mx.random.key(1)) * 0.3).astype(dtype)
            kv.update_and_fetch(k0, k0)
            kv.offset = hist
    else:
        kv = KVCache()
        if hist:
            kv.keys = (mx.random.normal((1, 8, hist, 128), key=mx.random.key(2)) * 0.3).astype(dtype)
            kv.values = mx.random.normal((1, 8, hist, 128), key=mx.random.key(3)).astype(dtype)
            kv.offset = hist
    return kv, ArraysCache(4)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("sliding,hist", [(False, 0), (False, 9000), (True, 0), (True, 3000)])
def test_the_stage_writes_the_layers_cache_rows_and_its_attention(monkeypatch, sliding, hist, dtype):
    layer = _layer(sliding, dtype, 31 + hist)
    rows = 96
    xs = [(mx.random.normal((1, rows, 2304), key=mx.random.key(40 + i)) * 0.5).astype(dtype) for i in range(3)]
    if sliding:
        xs = [xs[0]] * 3
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)
    kv0, conv0 = _caches(layer, hist, dtype, rows)
    want = M.Layer.attention(layer, *xs, kv0, conv0, None)
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "on")
    kv1, conv1 = _caches(layer, hist, dtype, rows)
    before = A.attention_route_counts()
    got = M.Layer.attention(layer, *xs, kv1, conv1, None)
    after = A.attention_route_counts()
    visit = "window" if sliding else "full"
    assert after.get(f"prompt_stage.{visit}.selected", 0) == before.get(f"prompt_stage.{visit}.selected", 0) + 1
    mx.eval(want, got, kv0.state, kv1.state)
    for a, b in zip(kv1.state, kv0.state):           # the cache rows: the layer's own bits
        assert bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())
    assert got.shape == want.shape and got.dtype == want.dtype
    assert _rel_rms(got, want.astype(mx.float32)) <= 4 * BOUND[dtype]


def test_the_stage_is_off_by_default(monkeypatch):
    layer = _layer(True, mx.bfloat16, 5)
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)
    assert F.stage(layer, *([mx.zeros((1, 64, 2304), mx.bfloat16)] * 3), *_caches(layer, 0, mx.bfloat16, 64), M.linear) is None


def test_the_gated_store_applies_the_layers_gate():
    ops = _call(64, 600, 23)
    gate = (mx.random.normal((1, 64, 3072), key=mx.random.key(9)) * 2).astype(mx.bfloat16)
    h = F.flash_attention(*ops, 1024)
    want = (h.transpose(0, 2, 1, 3).reshape(1, 64, -1).astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(mx.bfloat16)
    got = F.flash_attention(*ops, 1024, gate=gate)
    assert got.shape == (1, 64, 3072)
    assert _rel_rms(got, want.astype(mx.float32)) <= 1e-3


def _rows(rows, seed, dtype=mx.bfloat16):
    k = mx.random.normal((1, rows, 8, 128), key=mx.random.key(seed)).astype(dtype).transpose(0, 2, 1, 3)
    v = mx.random.normal((1, rows, 8, 128), key=mx.random.key(seed + 1)).astype(dtype).transpose(0, 2, 1, 3)
    return k, v


@pytest.mark.parametrize("chunks", [(512, 512, 512), (2048, 46), (300, 700, 1100), (40, 40)])
def test_the_window_launch_moves_the_librarys_values(chunks):
    ours, lib = WindowCache(max_size=1024), WindowCache(max_size=1024)
    for i, rows in enumerate(chunks):
        k, v = _rows(rows, 100 + i)
        a = F.window_update(ours, k, v)
        b = lib.update_and_fetch(k, v)
        mx.eval(a, b)
        for x, y in zip(a, b):
            assert x.shape == y.shape and bool(mx.array_equal(x.view(mx.uint16), y.view(mx.uint16)).item())
        assert (ours.offset, ours._idx) == (lib.offset, lib._idx)
        for x, y in zip(ours.state, lib.state):
            assert bool(mx.array_equal(x.view(mx.uint16), y.view(mx.uint16)).item())


def test_the_bank_sized_for_the_prompt_ends_with_the_librarys_state(monkeypatch):
    chunks = (512, 512, 512, 300)
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(sum(chunks)))
    ours, lib = KVCache(), KVCache()
    for i, rows in enumerate(chunks):
        k, v = _rows(rows, 200 + i)
        grew = F.reserve_bank(ours, k)
        assert grew == (i == 0)                                  # sized once, at the first chunk
        a = ours.update_and_fetch(k, v)
        b = lib.update_and_fetch(k, v)
        mx.eval(a, b)
        for x, y in zip(a, b):
            assert x.shape == y.shape and bool(mx.array_equal(x.view(mx.uint16), y.view(mx.uint16)).item())
    assert ours.keys.shape[2] == 2048                            # the prompt, in whole steps
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "100")   # a length that cannot be right
    assert not F.reserve_bank(KVCache(), _rows(512, 7)[0])


def test_a_restored_prefix_is_copied_once_into_the_sized_bank(monkeypatch):
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "3000")
    kv = KVCache()
    k0, v0 = _rows(1000, 300)
    kv.update_and_fetch(k0, v0)
    k1, v1 = _rows(512, 301)
    assert F.reserve_bank(kv, k1)
    keys, values = kv.update_and_fetch(k1, v1)
    want_k = mx.concatenate([k0, k1], axis=2)
    assert bool(mx.array_equal(keys.view(mx.uint16), want_k.view(mx.uint16)).item())
    assert kv.keys.shape[2] == 3072
