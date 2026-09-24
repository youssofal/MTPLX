"""Tests for the vendored ArraysCache advance() leak fix.

Ports the core intent of the upstream (mlx-lm PR #1642) test suite,
runnable with the vendored ``FixedArraysCache`` installed over a STOCK
mlx-lm 0.31.x — the environment the fix exists for. The real proof run is
the stock venv (``.venv-base``, no pytest), so this module is executable
directly (``python tests/test_arrays_cache_patch.py``) as well as under
pytest. No model loads: every check runs on synthetic arrays.

Against an mlx-lm that already carries the fix natively (the pinned
``.venv``), the stock-proof tests skip and only the installer's
``upstream_fixed`` path is asserted — but the patch modules must still
import cleanly there, which the module-level imports below prove.
"""

import gc
import resource
import sys
import traceback

try:
    import pytest
except ImportError:  # stock venv: plain-python runner below
    pytest = None

import mlx.core as mx
import mlx_lm.models.cache as cache_module

# Imported BEFORE any install call so the installer's rebind loop has a
# deterministic already-imported ``from ... import ArraysCache`` holder to
# fix up (mlx_lm.generate imports it too, but package attribute shadowing
# makes the model module the cleaner witness).
import mlx_lm.models.qwen3_next as qwen3_next_module

from mtplx.arrays_cache_patch import (
    ALREADY_INSTALLED,
    UPSTREAM_FIXED,
    VENDORED_INSTALLED,
    install_arrays_cache_fix,
)
from mtplx.vendored_arrays_cache import FixedArraysCache

# Snapshot the pre-install state of the process. On the stock venv this is
# the broken class; on a natively fixed mlx-lm it is the upstream fixed
# class; and if the vendored fix installed before this module imported
# (a3b_mtp_batch installs at import, and pytest collection imports it
# first), recover the genuine stock class the installer preserved so the
# leak-proof negative control keeps measuring real stock behavior.
from mtplx import arrays_cache_patch as _patch_module

_STOCK_CLASS = cache_module.ArraysCache
_VENDORED_PREINSTALLED = _STOCK_CLASS is FixedArraysCache
if _VENDORED_PREINSTALLED:
    _preserved = getattr(_patch_module, "STOCK_ARRAYS_CACHE", None)
    if _preserved is not None and _preserved is not FixedArraysCache:
        # Keep _VENDORED_PREINSTALLED truthful (install-order assertions key
        # on it); only the measurement target swaps to the genuine stock
        # class so the leak negative control stays meaningful.
        _STOCK_CLASS = _preserved
_probe = _STOCK_CLASS(1)
UPSTREAM_NATIVELY_FIXED = (
    not _VENDORED_PREINSTALLED
    and hasattr(_probe, "_lp_advance")
    and hasattr(_probe, "_len_advance")
)
del _probe

_install_results: list[str] = []


class _Skip(Exception):
    """Raised to skip a test when pytest is unavailable."""


_SKIP_EXCEPTIONS = (_Skip,) if pytest is None else (_Skip, pytest.skip.Exception)


def _skip(reason: str):
    if pytest is not None:
        pytest.skip(reason)
    raise _Skip(reason)


def _skip_unless_stock():
    if UPSTREAM_NATIVELY_FIXED:
        _skip("installed mlx-lm already carries the advance() fix natively")


def _ensure_installed() -> str:
    """Install the fix (idempotent) and record every result seen."""
    result = install_arrays_cache_fix()
    _install_results.append(result)
    return result


def _assert_raises(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001 - diagnostic re-raise
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{exc_type.__name__} not raised")


def _rss_bytes() -> int:
    """Peak RSS in bytes (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _armed_cache(cls, left_padding, lengths, slot_shape=(1, 4)):
    cache = cls(1)
    cache[0] = mx.zeros(slot_shape)
    cache.left_padding = mx.array(left_padding)
    cache.lengths = mx.array(lengths)
    return cache


def _warmup(cls):
    """Touch every measured code path once so Metal/stream/lazy-module
    initialization does not bill its RSS to the measured loop."""
    cache = _armed_cache(cls, [1], [2])
    for _ in range(10):
        cache.advance(1)
    mx.eval(cache.make_mask(4), cache.left_padding, cache.lengths)
    mx.synchronize()


def _measure_advance_growth(cls, n):
    """RSS / active-MLX growth across n advance(1) calls on an armed
    1-row cache; returns (rss_delta, active_delta, cache)."""
    cache = _armed_cache(cls, [3], [7])
    mx.eval(cache.left_padding, cache.lengths)
    mx.synchronize()
    rss_before = _rss_bytes()
    active_before = mx.get_active_memory()
    for _ in range(n):
        cache.advance(1)
    rss_delta = _rss_bytes() - rss_before
    active_delta = mx.get_active_memory() - active_before
    return rss_delta, active_delta, cache


# --------------------------------------------------------------------------
# Installer state machine
# --------------------------------------------------------------------------


def test_installer_reports_upstream_fixed():
    """Natively fixed mlx-lm: installer is a no-op, vendored class inert."""
    if not UPSTREAM_NATIVELY_FIXED:
        _skip("stock mlx-lm: covered by the vendored-install tests")
    before = cache_module.ArraysCache
    assert install_arrays_cache_fix() == UPSTREAM_FIXED
    assert cache_module.ArraysCache is before, "upstream class must not be replaced"
    assert install_arrays_cache_fix() == UPSTREAM_FIXED
    # The vendored subclass must still be importable and constructible
    # here even though it is never installed.
    probe = FixedArraysCache(1)
    assert hasattr(probe, "_lp_advance") and hasattr(probe, "_len_advance")
    assert isinstance(probe, before)


def test_installer_state_transitions():
    """stock -> vendored_installed -> already_installed, with rebinding."""
    _skip_unless_stock()
    # "No install yet" must be judged process-wide, not per this module:
    # other test modules (and a3b_mtp_batch's import) install too. The
    # installer preserves the replaced class exactly when a vendored
    # install happened, so that global is the truthful signal.
    install_seen = (
        _VENDORED_PREINSTALLED
        or bool(_install_results)
        or getattr(_patch_module, "STOCK_ARRAYS_CACHE", None) is not None
    )
    if not install_seen:
        assert qwen3_next_module.ArraysCache is _STOCK_CLASS
    result = _ensure_installed()
    assert result == (ALREADY_INSTALLED if install_seen else VENDORED_INSTALLED)
    assert _install_results[0] in (VENDORED_INSTALLED, ALREADY_INSTALLED)
    assert cache_module.ArraysCache is FixedArraysCache
    # Already-imported ``from mlx_lm.models.cache import ArraysCache``
    # holders were rebound (models and the generate loop).
    assert qwen3_next_module.ArraysCache is FixedArraysCache
    generate_module = sys.modules.get("mlx_lm.generate")
    if generate_module is not None:
        assert generate_module.ArraysCache is FixedArraysCache
    # save_prompt_cache/load_prompt_cache name round-trip stays resolvable.
    assert cache_module.FixedArraysCache is FixedArraysCache
    # Idempotent.
    assert install_arrays_cache_fix() == ALREADY_INSTALLED


def test_launcher_gate_probe_and_isinstance():
    """The a3b_mtp_batch launcher gate probe passes; isinstance holds."""
    _skip_unless_stock()
    _ensure_installed()
    # Exact probe from a3b_mtp_batch._require_mlx_lm_arrays_cache_fix.
    cache = cache_module.ArraysCache(1)
    assert hasattr(cache, "_lp_advance") and hasattr(cache, "_len_advance")
    # graphbank builds specs with isinstance(entry, ArraysCache) against
    # whatever name it imported — patched instances must satisfy the
    # stock class too.
    assert isinstance(cache, _STOCK_CLASS)
    assert isinstance(cache, FixedArraysCache)
    # The bookkeeping attributes exist from construction on every path,
    # including _BaseCache.from_state's ``cls.__new__(cls)``.
    bare = FixedArraysCache.__new__(FixedArraysCache)
    assert bare._lp_advance == 0 and bare._len_advance == 0


# --------------------------------------------------------------------------
# (a) 50k advance() calls stay flat in memory
# --------------------------------------------------------------------------


def test_advance_memory_flat_50k():
    """Runs :func:`_advance_memory_flat_50k_body` in a fresh interpreter.

    The measurement reads the process's PEAK RSS, which is a high-water mark:
    once any earlier test in the same process has peaked higher (a tokenizer
    audit, a big synthetic model), every later growth reads 0 bytes. The
    stock class's 257 MiB leak then measured 0 and this test failed, and the
    bound on the fixed class passed without measuring anything. A fresh
    process owns its own high-water mark.
    """
    _skip_unless_stock()
    import os
    import subprocess

    here = os.path.abspath(__file__)
    root = os.path.dirname(os.path.dirname(here))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (root, env.get("PYTHONPATH", "")) if item
    )
    done = subprocess.run(
        [sys.executable, here, "--isolated", "_advance_memory_flat_50k_body"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(done.stdout)
    assert done.returncode == 0, (
        f"isolated measurement failed (rc {done.returncode}):\n"
        f"{done.stdout}\n{done.stderr}"
    )
    assert "[stock]" in done.stdout and "[fixed]" in done.stdout, done.stdout


def _advance_memory_flat_50k_body():
    _skip_unless_stock()
    _ensure_installed()
    n = 50_000
    _warmup(cache_module.ArraysCache)
    _warmup(_STOCK_CLASS)

    rss_fixed, active_fixed, cache = _measure_advance_growth(
        cache_module.ArraysCache, n
    )
    print(
        f"\n[fixed] {n} x advance(1): RSS growth {rss_fixed / 2**20:.2f} MiB, "
        f"active MLX growth {active_fixed / 2**20:.3f} MiB"
    )
    assert rss_fixed < 8 * 2**20, (
        f"fixed class grew RSS by {rss_fixed} bytes over {n} advances "
        "(bound: 8 MiB)"
    )
    assert abs(active_fixed) < 1 * 2**20, (
        f"fixed class grew active MLX memory by {active_fixed} bytes"
    )
    # Deferred arithmetic is exact at scale, and folds per field.
    assert cache._len_advance == n and cache._lp_advance == n
    assert cache.lengths.tolist() == [7 - n]
    assert cache._len_advance == 0 and cache._lp_advance == n
    assert cache.left_padding.tolist() == [3 - n]
    assert cache._lp_advance == 0

    # Stock comparison (cheap: lazy graph construction only). Collapse the
    # dead chain with an eval before dropping it so teardown stays shallow.
    rss_stock, active_stock, stock_cache = _measure_advance_growth(_STOCK_CLASS, n)
    print(
        f"[stock] {n} x advance(1): RSS growth {rss_stock / 2**20:.2f} MiB, "
        f"active MLX growth {active_stock / 2**20:.3f} MiB"
    )
    assert stock_cache.lengths.tolist() == [7 - n]  # same arithmetic, leaked graph
    del stock_cache
    gc.collect()
    assert rss_stock > 32 * 2**20, (
        f"expected the stock class to leak visibly over {n} advances, "
        f"measured only {rss_stock} bytes — comparison harness is broken"
    )
    assert rss_stock > 4 * max(rss_fixed, 1), (rss_stock, rss_fixed)


# --------------------------------------------------------------------------
# (b) per-field lazy fold-on-read
# --------------------------------------------------------------------------


def test_per_field_fold_on_read():
    _skip_unless_stock()
    _ensure_installed()
    cache = _armed_cache(cache_module.ArraysCache, [5], [9])
    cache.advance(3)
    cache.advance(2)
    assert cache._lp_advance == 5 and cache._len_advance == 5
    lengths = cache.lengths  # folds ONLY lengths
    assert cache._len_advance == 0
    assert cache._lp_advance == 5, "reading lengths must not fold left_padding"
    assert lengths.tolist() == [4]
    assert cache._left_padding.tolist() == [5], "stored lp array folded early"
    assert cache.left_padding.tolist() == [0]
    assert cache._lp_advance == 0

    # make_mask folds only the fields the mask consumes: both when both are
    # armed (the mask is their AND since the #420 batched-lane fix), else one.
    cache2 = _armed_cache(cache_module.ArraysCache, [2], [6])
    cache2.advance(2)
    mask = cache2.make_mask(4)
    assert cache2._lp_advance == 0 and cache2._len_advance == 0
    assert mask.tolist() == [[True, True, True, True]]  # pos >= (2 - 2) and pos < (6 - 2)
    cache3 = cache_module.ArraysCache(1)
    cache3[0] = mx.zeros((1, 4))
    cache3.left_padding = mx.array([2])  # lengths stay unarmed
    cache3.advance(1)
    assert cache3.make_mask(3).tolist() == [[False, True, True]] and cache3._lp_advance == 0


# --------------------------------------------------------------------------
# (c) filter/extend churn stays bounded and exact
# --------------------------------------------------------------------------


def test_filter_extend_churn_bounded():
    _skip_unless_stock()
    _ensure_installed()
    cls = cache_module.ArraysCache
    _warmup(cls)
    mx.synchronize()
    rss_before = _rss_bytes()
    active_before = mx.get_active_memory()

    # Filter churn: advance + keep-all filter, 300 rounds.
    cache = cls(1)
    cache[0] = mx.zeros((4, 8))
    cache.left_padding = mx.array([0, 1, 2, 3])
    cache.lengths = mx.array([10, 11, 12, 13])
    keep_all = mx.array([0, 1, 2, 3])
    for _ in range(300):
        cache.advance(1)
        cache.filter(keep_all)
    assert cache.left_padding.tolist() == [v - 300 for v in (0, 1, 2, 3)]
    assert cache.lengths.tolist() == [v - 300 for v in (10, 11, 12, 13)]

    # Extend churn: grow by one row, shrink back, 200 rounds.
    cache2 = cls(1)
    cache2[0] = mx.zeros((2, 8))
    cache2.left_padding = mx.array([1, 2])
    cache2.lengths = mx.array([5, 6])
    keep_two = mx.array([0, 1])
    for _ in range(200):
        cache2.advance(1)
        other = cls(1)
        other[0] = mx.zeros((1, 8))
        other.left_padding = mx.array([9])
        other.lengths = mx.array([9])
        cache2.extend(other)  # folds the pending advance, batch -> 3
        cache2.filter(keep_two)  # batch -> 2
    assert cache2.left_padding.tolist() == [1 - 200, 2 - 200]
    assert cache2.lengths.tolist() == [5 - 200, 6 - 200]
    assert cache2.batch_size == 2

    mx.synchronize()
    rss_delta = _rss_bytes() - rss_before
    active_delta = mx.get_active_memory() - active_before
    print(
        f"\n[churn] 300 filter + 200 extend/filter rounds: RSS growth "
        f"{rss_delta / 2**20:.2f} MiB, active MLX growth "
        f"{active_delta / 2**20:.3f} MiB"
    )
    assert active_delta < 8 * 2**20, f"active MLX grew {active_delta} bytes"
    assert rss_delta < 64 * 2**20, f"RSS grew {rss_delta} bytes"


# --------------------------------------------------------------------------
# (d) meta_state round-trip
# --------------------------------------------------------------------------


def test_meta_state_round_trip():
    _skip_unless_stock()
    _ensure_installed()
    cls = cache_module.ArraysCache

    cache = cls(1)
    cache[0] = mx.zeros((3, 2))
    cache.left_padding = mx.array([1, 2, 3])
    cache.lengths = mx.array([4, 5, 6], dtype=mx.int16)
    cache.advance(2)
    meta = cache.meta_state  # linearizable snapshot: folds both fields
    assert meta == ("int32:-1,0,1", "int16:2,3,4")
    assert cache._lp_advance == 0 and cache._len_advance == 0

    restored = cls(1)
    restored.meta_state = meta
    assert restored.left_padding.tolist() == [-1, 0, 1]
    assert restored.left_padding.dtype == mx.int32
    assert restored.lengths.tolist() == [2, 3, 4]
    assert restored.lengths.dtype == mx.int16

    # Absent metadata stays absent through the round trip.
    assert cls(1).meta_state == ""
    empty = cls(1)
    empty.meta_state = ""
    assert empty.left_padding is None and empty.lengths is None

    one_sided = cls(1)
    one_sided.lengths = mx.array([7, 8])
    lp_entry, len_entry = one_sided.meta_state
    assert lp_entry == "" and len_entry == "int32:7,8"
    back = cls(1)
    back.meta_state = (lp_entry, len_entry)
    assert back.left_padding is None
    assert back.lengths.tolist() == [7, 8]

    # The save_prompt_cache path builds instances via from_state.
    loaded = cls.from_state([mx.zeros((3, 2))], meta)
    assert loaded.left_padding.tolist() == [-1, 0, 1]
    assert loaded.lengths.tolist() == [2, 3, 4]


# --------------------------------------------------------------------------
# (e) advance() offset validation
# --------------------------------------------------------------------------


def test_advance_offset_validation():
    _skip_unless_stock()
    _ensure_installed()
    cls = cache_module.ArraysCache

    unarmed = cls(1)
    _assert_raises(TypeError, unarmed.advance, mx.array(1))  # even with no metadata

    armed = _armed_cache(cls, [1], [2])
    _assert_raises(TypeError, armed.advance, mx.array(1))
    _assert_raises(TypeError, armed.advance, 1.5)

    narrow = cls(1)
    narrow.lengths = mx.array([4], dtype=mx.int16)
    _assert_raises(ValueError, narrow.advance, 40_000)  # out of int16 range
    narrow.advance(True)  # operator.index: integral scalars normalize
    assert narrow.lengths.tolist() == [3]


# --------------------------------------------------------------------------
# (f) behavioral equivalence: stock-with-eager-eval vs fixed
# --------------------------------------------------------------------------


def _mask(cache, n, eager):
    """The mask under test. For the STOCK reference (``eager``) apply the #420 rule the fixed
    class implements: when left_padding and lengths are both armed the mask is their AND
    (stock returned the left-padding mask alone, which let right-padded rows' pad tokens
    reach the recurrent update). Everything else stays stock."""
    mask = cache.make_mask(n)
    if eager and mask is not None and cache.left_padding is not None and cache.lengths is not None:
        mask = mask & (mx.arange(n) < cache.lengths[:, None])
    return mask


def _drive(cls, eager):
    """Run one synthetic cache lifecycle and return a plain-python trace.

    ArraysCache has no update_and_fetch (it is the GDN slot container);
    the equivalent surface is slot __setitem__/__getitem__ plus
    make_mask/advance/filter/extend/merge, all exercised here. ``eager``
    evaluates stock's lazily decremented metadata after every advance so
    its graph stays bounded — values must match the fixed class exactly.
    """
    out = []

    cache = cls(2)
    cache[0] = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    cache[1] = mx.ones((2, 4), dtype=mx.float32)
    cache.left_padding = mx.array([2, 0])
    cache.lengths = mx.array([3, 2])

    def settle():
        if eager:
            mx.eval(cache.left_padding, cache.lengths)

    cache.advance(1)
    settle()
    out.append(("mask_lp", _mask(cache, 4, eager).tolist()))
    cache.advance(1)
    settle()
    out.append(("lp", cache.left_padding.tolist()))
    out.append(("len", cache.lengths.tolist()))

    cache.filter(mx.array([1, 0]))
    out.append(("lp_filtered", cache.left_padding.tolist()))
    out.append(("len_filtered", cache.lengths.tolist()))
    out.append(("slot0_filtered", cache[0].tolist()))

    cache.advance(1)  # left pending across extend for the fixed class
    settle()
    other = cls(2)
    other[0] = mx.full((1, 4), 7.0)
    other[1] = mx.zeros((1, 4))
    other.left_padding = mx.array([2])
    other.lengths = mx.array([5])
    cache.extend(other)
    out.append(("batch_size", cache.batch_size))
    cache.advance(1)
    settle()
    out.append(("mask_extended", _mask(cache, 5, eager).tolist()))
    out.append(("lp_extended", cache.left_padding.tolist()))
    out.append(("len_extended", cache.lengths.tolist()))
    out.append(("slot1_extended", cache[1].tolist()))

    extracted = cache.extract(1)
    out.append(("extract_slot0", extracted[0].tolist()))

    cache.finalize()
    out.append(("final_lp", cache.left_padding))
    out.append(("final_len", cache.lengths))
    out.append(("final_mask", _mask(cache, 3, eager)))

    # lengths-only mask branch
    lengths_only = cls(1)
    lengths_only[0] = mx.zeros((2, 3))
    lengths_only.lengths = mx.array([2, 3])
    lengths_only.advance(1)
    if eager:
        mx.eval(lengths_only.lengths)
    out.append(("mask_len", lengths_only.make_mask(4).tolist()))

    # merge over empty caches arms left_padding with zeros
    merged = cls.merge([cls(1), cls(1), cls(1)])
    out.append(("merge_lp", merged.left_padding.tolist()))
    return out


def test_behavioral_equivalence_stock_vs_fixed():
    _skip_unless_stock()
    _ensure_installed()
    stock_trace = _drive(_STOCK_CLASS, eager=True)
    fixed_trace = _drive(FixedArraysCache, eager=False)
    assert stock_trace == fixed_trace, (
        "stock-with-eager-eval and FixedArraysCache diverged:\n"
        + "\n".join(
            f"  {s} != {f}" for s, f in zip(stock_trace, fixed_trace) if s != f
        )
    )


# --------------------------------------------------------------------------
# Plain-python runner for the pytest-less stock venv
# --------------------------------------------------------------------------


def _main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--isolated":
        # One measured body in this fresh process (see
        # test_advance_memory_flat_50k). A skip is a pass: the parent made
        # the same skip decision before it got here.
        try:
            globals()[sys.argv[2]]()
        except _SKIP_EXCEPTIONS as exc:
            print(f"SKIP {sys.argv[2]}: {exc}")
        return 0
    tests = [
        (name, fn)
        for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except _SKIP_EXCEPTIONS as exc:
            print(f"SKIP {name}: {exc}")
        except Exception:  # noqa: BLE001 - report and continue
            traceback.print_exc()
            print(f"FAIL {name}")
            failures.append(name)
        else:
            print(f"PASS {name}")
    total = len(tests)
    print(
        f"\n{total - len(failures)}/{total} passed-or-skipped"
        + (f"; FAILURES: {', '.join(failures)}" if failures else "")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
