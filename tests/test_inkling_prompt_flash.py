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
    for kw in (dict(tg_m=2), dict(tg_m=8), dict(qreg=False), dict(tgbar=False), dict(psplit=True),
               dict(ptype=mx.float16)):
        assert _rel_rms(F.flash_attention(*ops, 0, **kw), ref) <= BOUND[mx.bfloat16], kw


def test_long_history_global_visit():
    # Past the vector-kernel threshold, where the default route runs the decode-shaped twin.
    ops = _call(48, 20480, 11)
    assert _rel_rms(F.flash_attention(*ops, 0), F.reference(*ops, 0)) <= BOUND[mx.bfloat16]


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
