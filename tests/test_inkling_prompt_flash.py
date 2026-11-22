"""One-launch prompt attention (``fast.prompt_flash``): numbers against the definition, the route switch.

The launch computes a prompt call's attention (the relative bias from the profile below its
extent, the sink as the softmax's first entry, window visits) in another order of the same sums
than the default prompt route, so it is held to the definition computed in float32, within the
rounding the query type itself puts on a result, on the shapes the prompt read makes: the first
chunk, a history crossing the bias extent, window visits that start at key 0 and later, a short
last chunk. The route is on unless ``MTPLX_INKLING_PROMPT_FLASH`` is set off; every selection and
decline is counted.
"""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the prompt attention needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import attention as A  # noqa: E402
from mtplx.models.inkling.fast import prompt_flash as F  # noqa: E402

if F.form() is None:  # pragma: no cover - an MLX without the headers either form needs
    pytest.skip("the launch needs the installed MLX's fragment headers", allow_module_level=True)

# Relative RMS against the float32 definition: the query type's own rounding of a result is
# 1.7e-3 (bf16) and 2.4e-4 (fp16) relative RMS on these distributions (lanes/p2/fcheck-3).
BOUND = {mx.bfloat16: 2.5e-3, mx.float16: 4e-4}


def _clipped(shape, limit, key, dtype):
    v = mx.random.normal(shape, key=key) * 2.2
    n = mx.linalg.norm(v, axis=-1, keepdims=True)
    return (v * mx.minimum(1.0, limit / (n + 1e-6))).astype(dtype)


def _no_crossover(monkeypatch):
    for name in ("SHORT_PROMPT", "SHORT_PROMPT_512", "SHORT_PROMPT_WIDE", "SHORT_PROMPT_SIMD", "SHORT_PROMPT_WIDE_SIMD"):
        monkeypatch.setattr(F, name, 0)


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


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows,keys,window", [
    (64, 64, 0), (64, 64, 1024), (64, 600, 1024), (200, 2248, 0), (200, 2248, 1024),
    (37, 1100, 0), (46, 1069, 1024), (512, 1535, 1024), (1, 1, 0), (17, 17, 1024),
])
def test_the_plain_form_for_m1_to_m4_matches_the_definition(rows, keys, window, dtype, monkeypatch):
    """The 8 x 8 simdgroup-matrix form a GPU without tensor units runs (the product's own M1 to M4 rehearsal switch
    selects it here): the same calls, the same bounds."""
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    monkeypatch.setenv(F.SIMD_ENV, "1")
    if F.form() != "simd":  # pragma: no cover - an MLX without the 8 x 8 fragment header
        pytest.skip("no 8 x 8 fragment header")
    ops = _call(rows, keys, rows * 11 + keys + window + 3, dtype)
    assert F.decline(*ops, window) is None
    got = F.flash_attention(*ops, window)
    ref = F.reference(*ops, window)
    assert got.shape == (1, 24, rows, 128) and got.dtype == dtype
    assert bool(mx.all(mx.isfinite(got)).item())
    assert _rel_rms(got, ref) <= BOUND[dtype]


def test_the_plain_form_gates_like_the_tensor_form(monkeypatch):
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    monkeypatch.setenv(F.SIMD_ENV, "1")
    if F.form() != "simd":  # pragma: no cover
        pytest.skip("no 8 x 8 fragment header")
    ops = _call(200, 2248, 9)
    gate = (mx.random.normal((1, 200, 24 * 128), key=mx.random.key(3)) * 2.0).astype(mx.bfloat16)
    plain = F.flash_attention(*ops, 0)
    gated = F.flash_attention(*ops, 0, gate=gate)
    want = (plain.transpose(0, 2, 1, 3).reshape(1, 200, -1).astype(mx.float32)
            * mx.sigmoid(gate.astype(mx.float32))).astype(mx.bfloat16)
    assert gated.shape == (1, 200, 3072)
    assert _rel_rms(gated, want.astype(mx.float32)) <= 1e-3


def test_a_plain_form_the_compiler_refuses_leaves_the_stock_attention_serving(monkeypatch):
    # As F1's F-1 asks of the projections: on a Mac whose compiler refuses the plain form, its first use fails inside
    # the stage's check, and the stage keeps the stock attention for the process (counted
    # prompt_stage.<visit>.stock.first_use_error), with no build attempted again.
    from mtplx.models.inkling import attention as A

    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    monkeypatch.setenv(F.SIMD_ENV, "1")
    if F.form() != "simd":  # pragma: no cover
        pytest.skip("no 8 x 8 fragment header")
    broken = mx.fast.metal_kernel(name="mtplx_test_refused_flash_simd", input_names=["q", "k", "v", "p", "s", "g"],
                                  output_names=["out"], source="this is not Metal;")
    built = []
    monkeypatch.setattr(F, "_kernel_simd", lambda gated: built.append(gated) or broken)
    monkeypatch.setattr(A, "_flash_checks", {})
    for window in (True, False):
        assert A._flash_first_use(mx.bfloat16, window) == "first_use_error"
    tries = len(built)
    assert A._flash_first_use(mx.bfloat16, True) == "first_use_error" and len(built) == tries


def test_an_m5s_selection_and_bits_do_not_move_with_the_plain_forms_default(monkeypatch):
    # The plain form's default reaches only GPUs without the tensor units: on an M5 the form, the counter word, the
    # crossovers and the launch's output bits are the same whatever the plain form's switch says.
    if not F.tensor_units():  # pragma: no cover - runs on the tensor units only
        pytest.skip("an M5 check")
    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", raising=False)
    ops = _call(200, 2248, 17)
    outs = []
    for value in (None, "0", "1"):
        if value is None:
            monkeypatch.delenv(F.SIMD_ENV, raising=False)
        else:
            monkeypatch.setenv(F.SIMD_ENV, value)
        assert F.form() == "tensor" and F.selected() == "selected" and F.runs_here()
        assert (F.SHORT_PROMPT, F.SHORT_PROMPT_WIDE) == (8192, 4096)
        outs.append(F.flash_attention(*ops, 1024))
    mx.eval(outs)
    assert all(_same_bits(outs[0], o) for o in outs[1:])


def test_without_the_tensor_units_the_plain_form_is_on_by_default_and_counted(monkeypatch):
    # M1 to M4 (here: the product's own rehearsal switch): the plain form is chosen from the GPU family unless switched
    # off, its calls are counted simd.selected, and the switch keeps the stock attention, counted plain_form_off.
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    monkeypatch.delenv(F.SIMD_ENV, raising=False)
    if F.simd_text()[0] is None:  # pragma: no cover - an MLX without the 8 x 8 fragment header
        pytest.skip("no 8 x 8 fragment header")
    assert F.form() == "simd" and F.selected() == "simd.selected" and F.runs_here()
    ops = _call(64, 600, 23)
    assert F.decline(*ops, 1024) is None
    monkeypatch.setenv(F.SIMD_ENV, "0")
    assert F.form() is None and not F.runs_here() and F.decline(*ops, 1024) == "plain_form_off"


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
    # Whichever form this Mac runs (X1, 01:03: on an M4 the plain form reads the 8 x 8 header, not the tensor one).
    q, k, v, prof, sink = _call(64, 600, 3)
    monkeypatch.setattr(F, "library_text", lambda: (None, "missing:steel/gemm/nax.h"))
    monkeypatch.setattr(F, "simd_text", lambda: (None, "missing:steel/attn/mma.h"))
    assert F.decline(q, k, v, prof, sink, 0) in ("library_headers", "no_tensor_units")
    assert F.form() is None


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


def test_route_is_on_by_default_off_where_set_off_and_counted(monkeypatch):
    ops = _call(64, 600, 9)
    _no_crossover(monkeypatch)                           # the route's numbers on short calls
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "off")
    before = A.attention_route_counts()
    today = A.attend(*ops, 0)
    after = A.attention_route_counts()
    assert not any(k.startswith("prompt_flash.") and after[k] != before.get(k, 0) for k in after)

    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)
    before = A.attention_route_counts()
    on = A.attend(*ops, 0)
    on_window = A.attend(*ops, 1024)
    after = A.attention_route_counts()
    sel = F.selected()
    assert after.get(f"prompt_flash.full.{sel}", 0) == before.get(f"prompt_flash.full.{sel}", 0) + 1
    assert after.get(f"prompt_flash.window.{sel}", 0) == before.get(f"prompt_flash.window.{sel}", 0) + 1
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
    _no_crossover(monkeypatch)
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
        "sliding_window_size": 1024, "num_key_value_heads": 8, "head_dim": 128}


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
    _no_crossover(monkeypatch)                           # the stage's numbers on short calls
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "off")
    kv0, conv0 = _caches(layer, hist, dtype, rows)
    want = M.Layer.attention(layer, *xs, kv0, conv0, None)
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)   # on by default
    kv1, conv1 = _caches(layer, hist, dtype, rows)
    before = A.attention_route_counts()
    got = M.Layer.attention(layer, *xs, kv1, conv1, None)
    after = A.attention_route_counts()
    visit = "window" if sliding else "full"
    sel = F.selected()
    assert after.get(f"prompt_stage.{visit}.{sel}", 0) == before.get(f"prompt_stage.{visit}.{sel}", 0) + 1
    mx.eval(want, got, kv0.state, kv1.state)
    for a, b in zip(kv1.state, kv0.state):           # the cache rows: the layer's own bits
        assert bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())
    assert got.shape == want.shape and got.dtype == want.dtype
    assert _rel_rms(got, want.astype(mx.float32)) <= 4 * BOUND[dtype]


def test_the_stage_is_off_where_set_off_and_leaves_the_cache_untouched(monkeypatch):
    layer = _layer(True, mx.bfloat16, 5)
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "off")
    query = mx.zeros((1, 24, 64, 128), mx.bfloat16)
    rows = mx.zeros((1, 8, 64, 128), mx.bfloat16)
    kv = _caches(layer, 0, mx.bfloat16, 64)[0]
    assert F.stage(layer, query, rows, rows, mx.zeros((1, 64, 24, 1024), mx.bfloat16),
                   mx.zeros((1, 64, 24), mx.bfloat16), kv) is None
    assert kv.keys is None and kv.offset == 0


def _route(monkeypatch, layer, hist, rows=96, prompt=None, seed=50):
    """The layer's attention for one chunk with the switch on: (counter that moved, output, the
    default route's output on the same inputs)."""
    xs = [(mx.random.normal((1, rows, 2304), key=mx.random.key(seed + i)) * 0.5).astype(mx.bfloat16) for i in range(3)]
    if layer.sliding:
        xs = [xs[0]] * 3
    if prompt is None:
        monkeypatch.delenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", raising=False)
    else:
        monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(prompt))
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "off")
    kv0, conv0 = _caches(layer, hist, mx.bfloat16, rows)
    want = M.Layer.attention(layer, *xs, kv0, conv0, None)
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)   # on by default
    kv1, conv1 = _caches(layer, hist, mx.bfloat16, rows)
    before = A.attention_route_counts()
    got = M.Layer.attention(layer, *xs, kv1, conv1, None)
    after = A.attention_route_counts()
    moved = [k for k in after if k.startswith("prompt_stage.") and after[k] != before.get(k, 0)]
    assert len(moved) == 1 and after[moved[0]] == before.get(moved[0], 0) + 1
    calls = [k for k in after if k.startswith("prompt_flash.") and after[k] != before.get(k, 0)]
    if moved[0].endswith(".selected"):
        assert not calls                                    # the stage's attention is its own
    else:
        assert len(calls) == 1                              # the layer's own path asked attend's route
        moved.append(calls[0])
    mx.eval(want, got)
    return moved if len(moved) > 1 else moved[0], got, want


def _limit():
    wide = F._read_width() >= F.WIDE_READ_ROWS
    if F.form() == "simd":
        return F.SHORT_PROMPT_WIDE_SIMD if wide else F.SHORT_PROMPT_SIMD
    if F._read_width() == 512:
        return F.SHORT_PROMPT_512
    return F.SHORT_PROMPT_WIDE if wide else F.SHORT_PROMPT


@pytest.mark.parametrize("ram_gib", [18, 24, 36, 128])
def test_512_row_tensor_crossover_follows_width_on_every_mac(monkeypatch, ram_gib):
    monkeypatch.setattr(F, "form", lambda: "tensor")
    monkeypatch.setattr(mx, "device_info", lambda *args: {"memory_size": ram_gib * 1024**3})
    for length in (1024, 2048, 4095, 4096, 6144):
        monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(length))
        assert F.short_prompt(512, width=512) == (length < 4096)

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "2048")
    assert F.short_prompt(512, width=1024)
    assert F.short_prompt(512, width=2048)
    monkeypatch.setattr(F, "form", lambda: "simd")
    assert F.short_prompt(512, width=512)


def test_the_crossover_follows_the_reads_width(monkeypatch):
    from mtplx.generation import prefill_chunk_size_override

    monkeypatch.setattr(F, "SHORT_PROMPT_512", 6000)
    for narrow, wide in (("SHORT_PROMPT", "SHORT_PROMPT_WIDE"), ("SHORT_PROMPT_SIMD", "SHORT_PROMPT_WIDE_SIMD")):
        monkeypatch.setattr(F, narrow, 6000)                         # both forms' knobs (X1, 01:03: an M4 reads the
        monkeypatch.setattr(F, wide, 2000)                           # plain form's)
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "4000")
    with prefill_chunk_size_override(512):
        assert F.short_prompt(512)
    with prefill_chunk_size_override(2048):
        assert not F.short_prompt(512)
    assert F.short_prompt(512, width=512) and not F.short_prompt(512, width=4096)


def _same_bits(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


@pytest.mark.parametrize("sliding", [False, True])
def test_a_prompt_shorter_than_the_crossover_reads_on_the_default_route(monkeypatch, sliding):
    layer = _layer(sliding, mx.bfloat16, 61)
    visit = "window" if sliding else "full"
    moved, got, want = _route(monkeypatch, layer, 0, prompt=_limit() - 1)
    assert moved == [f"prompt_stage.{visit}.stock.short_prompt", f"prompt_flash.{visit}.stock.short_prompt"]
    assert _same_bits(got, want)
    moved, got, want = _route(monkeypatch, layer, 0, prompt=_limit())
    assert moved == f"prompt_stage.{visit}.{F.selected()}"
    assert _rel_rms(got, want.astype(mx.float32)) <= 4 * BOUND[mx.bfloat16]


def test_the_prompt_is_never_shorter_than_the_keys_a_chunk_sees(monkeypatch):
    # no prompt length set, or a stale one: the history and the chunk are the length so far
    layer = _layer(False, mx.bfloat16, 67)
    hist = _limit()
    assert _route(monkeypatch, layer, hist, prompt=None)[0] == f"prompt_stage.full.{F.selected()}"
    assert _route(monkeypatch, layer, hist, prompt=100)[0] == f"prompt_stage.full.{F.selected()}"
    assert _route(monkeypatch, layer, 0, prompt=None)[0][0] == "prompt_stage.full.stock.short_prompt"


def test_no_room_for_the_stage_reads_on_the_default_route(monkeypatch):
    from mtplx.models.inkling import head as H

    _no_crossover(monkeypatch)
    prompt = 64 * 1024
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(prompt))
    bank = F._bank_bytes(KVCache(), 96, _CFG, mx.bfloat16)
    assert bank == 2 * prompt * 8 * 128 * 2
    margin = 128 * 1024**2

    def room(n):
        monkeypatch.setattr(H, "_working_set", lambda device: mx.get_active_memory() + margin + n)

    window, full = _layer(True, mx.bfloat16, 71), _layer(False, mx.bfloat16, 73)
    room(F.EXTRA_PEAK_BYTES // 2)                                    # not even the chunk's extra
    for layer, visit in ((window, "window"), (full, "full")):
        moved, got, want = _route(monkeypatch, layer, 0, prompt=prompt)
        assert moved == [f"prompt_stage.{visit}.stock.memory", f"prompt_flash.{visit}.stock.memory"]
        assert _same_bits(got, want)
    room(F.EXTRA_PEAK_BYTES + bank // 2)                             # the chunk, not the whole bank
    assert _route(monkeypatch, window, 0, prompt=prompt)[0] == f"prompt_stage.window.{F.selected()}"
    moved, got, want = _route(monkeypatch, full, 0, prompt=prompt)
    # the stage declines (the bank); attend's one-launch route, which grows no bank, takes the call
    assert moved == ["prompt_stage.full.stock.memory", f"prompt_flash.full.{F.selected()}"]
    assert _rel_rms(got, want.astype(mx.float32)) <= 4 * BOUND[mx.bfloat16]
    room(F.EXTRA_PEAK_BYTES + 2 * bank)
    assert _route(monkeypatch, full, 0, prompt=prompt)[0] == f"prompt_stage.full.{F.selected()}"


def test_the_room_line_is_the_engines_limit_where_the_server_set_it_above_the_working_set(monkeypatch):
    """A 24 GB Mac: macOS recommends 16 GiB to the GPU, the server holds the engine to 18 GiB. 2.13.1 read
    against the working set and declined the one-launch attention on nearly every call past a read's first
    chunk (a simulated 24 GB Mac, `x1-fix-24gb-causes`); the line is the engine's limit there, as the server
    stamps it after the load. Where the working set is the larger (this 128 GB Mac: 120 GiB against a 90 GiB
    limit) it stays the line, and with no limit stamped (a harness, a test) the line is the working set."""
    from mtplx.models.inkling import head as H
    from mtplx.models.inkling.runtime import InklingRuntime

    gib = 1024**3
    monkeypatch.setattr(H, "_engine_limit", {"bytes": None})
    monkeypatch.setattr(H, "_working_set", lambda device: 16 * gib)
    assert H.room_line() == 16 * gib
    InklingRuntime.note_metal_memory_limit(object(), 18 * gib)
    assert H.room_line() == 18 * gib
    monkeypatch.setattr(H, "_working_set", lambda device: 120 * gib)
    H.set_engine_limit(90 * gib)
    assert H.room_line() == 120 * gib
    H.set_engine_limit(0)
    assert H.room_line() == 120 * gib


def test_a_24gb_mac_reads_on_the_stage_up_to_its_engine_limit(monkeypatch):
    """The chunk the working set refused: MLX holds the working set less 0.3 GiB, the engine's limit is 2 GiB
    above the working set. The stage reads it; with no limit above the working set it reads on the default
    route, counted ``memory``, as in 2.13.1."""
    from mtplx.models.inkling import head as H

    _no_crossover(monkeypatch)
    gib = 1024**3
    window = _layer(True, mx.bfloat16, 71)
    monkeypatch.setattr(H, "_engine_limit", {"bytes": None})

    def mac(limit_above):
        ws = mx.get_active_memory() + int(0.3 * gib)
        monkeypatch.setattr(H, "_working_set", lambda device: ws)
        H.set_engine_limit(ws + limit_above if limit_above else 0)

    mac(2 * gib)
    assert _route(monkeypatch, window, 0, prompt=32 * 1024)[0] == f"prompt_stage.window.{F.selected()}"
    mac(0)
    moved, got, want = _route(monkeypatch, window, 0, prompt=32 * 1024)
    assert moved == ["prompt_stage.window.stock.memory", "prompt_flash.window.stock.memory"]
    assert _same_bits(got, want)


def test_the_bank_bytes_are_what_the_bank_takes(monkeypatch):
    for total, hist in ((5000, 0), (5000, 1000), (70000, 4096)):
        monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(total))
        kv = KVCache()
        if hist:
            kv.update_and_fetch(*_rows(hist, 400))
        k, _ = _rows(512, 401)
        want = F._bank_bytes(kv, 512, _CFG, mx.bfloat16)
        assert F.reserve_bank(kv, k)
        assert want == kv.keys.nbytes + kv.values.nbytes
        assert F._bank_bytes(kv, 512, _CFG, mx.bfloat16) == 0          # sized: nothing more to take


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


def test_a_rotated_window_takes_the_librarys_write_and_says_so():
    # F1's F-3: once an answer has wrapped a window ring, the next chunk's write is the library's update (same
    # values), counted in the request's attention receipt instead of passing silently.
    from mtplx.models.inkling import attention as A

    kv, lib = WindowCache(max_size=1024), WindowCache(max_size=1024)
    for cache in (kv, lib):
        cache.update_and_fetch(*_rows(1100, 300))
        for step in range(3):                                      # three decode steps wrap the ring
            cache.update_and_fetch(*_rows(1, 310 + step))
    assert kv._idx != kv.keys.shape[2]
    receipt = {"graph_selections": {}}
    token = A._request.set((receipt, None))
    try:
        k, v = _rows(64, 320)
        got, want = F.window_update(kv, k, v), lib.update_and_fetch(k, v)
    finally:
        A._request.reset(token)
    mx.eval(got, want)
    assert all(bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()) for a, b in zip(got, want))
    assert receipt["graph_selections"] == {"prompt_stage.window.write.library.rotated": 1}
