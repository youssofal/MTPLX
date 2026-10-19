"""GDN prefill through MLX's chunked ``mx.fast.gated_delta_update`` (opt-in).

MLX 0.32.4 ships ``mx.fast.gated_delta_update`` with a chunked kernel that
runs 16 tokens at a time on the M5 tensor units (``gated_delta_fused_nax``,
the WY form with a triangular inverse per chunk), next to an 8-token chunked
kernel and a sequential one for older GPUs. On Qwen3.8-27B prefill shapes
(Hk 16, Hv 48, Dk = Dv = 128, 4,096-token chunks) it takes 6.4 ms per layer
call against 12.2 ms for mlx-lm's sequential ``gated_delta_kernel``, measured
on an M5 Pro.

The op casts ``g`` and ``beta`` to the type of ``q``. With bf16 activations
the decay ``g``, which sits close to 1, is then rounded to steps of 1/256
before the kernel takes its log, and on real Qwen3.8 activations the output
moves 4% and the state 6 to 9% (relative L2 against a float64 reference).
This route therefore passes g and beta as float32. On an MLX build that keeps
them float32 (``mlx_keeps_float32_gates``, probed once) q, k and v go in as
they are; on one that rounds them (0.32.4 as released) q, k and v go in as
float32 too and the output is cast back to the input type, which costs about
1.1 ms of the 4.8 ms saved per call. The state is float32 on both sides, as on
the stock path. Against float64 the state then lands at ~1e-3 relative,
inside bf16 rounding (3.9e-3) but not bit-equal to the stock sequential
kernel (~1e-6): the chunked form sums in another order.

That difference has two consequences the stock path does not have:

- A state written by this route and a state written by the stock route are
  not interchangeable bit for bit, so ``gdn_prefill_numerics_identity`` names
  the route and the server adds it to the session policy fingerprint. A
  session bank or SSD entry built under one route then misses under the
  other, instead of continuing a state the current route would not have
  produced. With the switch off the fingerprint is unchanged.
- Chunks start at each call's first token, so a prompt split at a position
  that is not a multiple of 16 rounds differently than the same prompt in
  one call. The stock sequential kernel is split-invariant.

Routed: scalar-gating calls without a mask, of at least
``MTPLX_GDN_FAST_PREFILL_MIN_T`` tokens (default 64), on the GPU, for the
head layouts MLX's kernel is built for. Everything else, and every call when
``mx.fast.gated_delta_update`` is missing (MLX before 0.32.4), stays on the
path that was installed before this one, byte for byte. Enable with
``MTPLX_GDN_FAST_PREFILL=1``.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys

import mlx.core as mx

from mtplx.demotions import note as _note_demotion

GDN_FAST_PREFILL_ENV = "MTPLX_GDN_FAST_PREFILL"
GDN_FAST_PREFILL_MIN_T_ENV = "MTPLX_GDN_FAST_PREFILL_MIN_T"
_DEFAULT_MIN_T = 64

# (Hk, Hv) pairs MLX 0.32.4 instantiates (mlx/backend/metal/gated_delta_update.cpp,
# supported_gated_delta_shape), all with Dk = Dv = 128. Other shapes fall back
# to MLX's per-token op loop, far slower than the stock kernel.
_MLX_KERNEL_HEADS = frozenset(
    {(24, 24), (32, 32), (16, 32), (16, 48), (16, 16), (16, 64)}
)

# Bump when the route's arithmetic changes, so banked states from the old
# version are not continued by the new one.
_ROUTE_VERSION = "mlx_fast_f32_v1"

ROUTE_COUNTS: dict[str, int] = {"routed": 0, "fallback_prefill_sized": 0}
_NOT_ENGAGED_REASON = (
    "a prefill-sized GDN call took the previous path: masked call, vectorized "
    "gating, a head layout MLX's kernel is not built for, or the CPU device"
)

_PATCH_STATE: dict = {"installed": False, "previous": None, "identity": None}


def fast_prefill_env_enabled() -> bool:
    return str(os.environ.get(GDN_FAST_PREFILL_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def fast_prefill_available() -> bool:
    """True when this MLX build has ``mx.fast.gated_delta_update`` (0.32.4+)."""
    return callable(getattr(mx.fast, "gated_delta_update", None))


def _min_route_t() -> int:
    try:
        return max(
            2, int(os.environ.get(GDN_FAST_PREFILL_MIN_T_ENV, str(_DEFAULT_MIN_T)))
        )
    except ValueError:
        return _DEFAULT_MIN_T


def fast_prefill_eligible(q: mx.array, v: mx.array, a: mx.array, mask, state) -> bool:
    """Structural gate: scalar gating, no mask, a head layout MLX instantiates."""
    if mask is not None or q.ndim != 4 or v.ndim != 4 or a.ndim != 3:
        return False
    _, _, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    if Dk != 128 or Dv != 128 or (Hk, Hv) not in _MLX_KERNEL_HEADS:
        return False
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    return state is None or state.dtype == mx.float32


_FLOAT32_GATES: dict = {"probed": False, "value": False}


def mlx_keeps_float32_gates() -> bool:
    """True when this MLX build passes g and beta to the kernel as float32.

    MLX 0.32.4 casts them to the type of q. A fixed build keeps them float32,
    and then 16-bit q, k and v can go in as they are. Probed once on the CPU
    stream (the cast sits in front of every backend): a decay of 0.9995
    rounds to 1.0 in bfloat16, so the two calls below only agree when the
    gates stayed float32.
    """
    if _FLOAT32_GATES["probed"]:
        return _FLOAT32_GATES["value"]
    value = False
    if fast_prefill_available():
        shape = (1, 2, 16, 128)
        q = mx.full(shape, 0.0625, dtype=mx.bfloat16)
        k = mx.full(shape, 0.0625, dtype=mx.bfloat16)
        v = mx.ones((1, 2, 16, 128), dtype=mx.bfloat16)
        g = mx.full((1, 2, 16), 0.9995, dtype=mx.float32)
        beta = mx.full((1, 2, 16), 0.5, dtype=mx.float32)
        state = mx.ones((1, 16, 128, 128), dtype=mx.float32)
        f32 = mx.float32
        with mx.stream(mx.cpu):
            _, s16 = mx.fast.gated_delta_update(q, k, v, g, beta, state)
            _, s32 = mx.fast.gated_delta_update(
                q.astype(f32), k.astype(f32), v.astype(f32), g, beta, state
            )
            value = bool(mx.array_equal(s16, s32).item())
    _FLOAT32_GATES.update(probed=True, value=value)
    return value


def gated_delta_fast_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """``mx.fast.gated_delta_update`` with float32 g and beta; y in the input type.

    q and k are already normalized and scaled (mlx-lm's contract), g is the
    linear decay in (0, 1], state is [B, Hv, Dv, Dk] float32. On an MLX build
    that rounds g and beta to the type of q, q, k and v go in as float32 too.
    """
    f32 = mx.float32
    g = g.astype(f32)
    beta = beta.astype(f32)
    if mlx_keeps_float32_gates():
        return mx.fast.gated_delta_update(q, k, v, g, beta, state)
    in_dtype = q.dtype
    y, state_out = mx.fast.gated_delta_update(
        q.astype(f32), k.astype(f32), v.astype(f32), g, beta, state
    )
    return y.astype(in_dtype), state_out


def _route_identity() -> str:
    # The chunk width (GATED_DELTA_CHUNK, 16 with tensor units and 8 without)
    # and the MLX build both change the arithmetic.
    chunk = os.environ.get("GATED_DELTA_CHUNK", "auto").strip() or "auto"
    # 16-bit and float32 activations give bit-equal results once the gates
    # are float32 (the kernels convert on load), so the build is enough.
    return f"{_ROUTE_VERSION}:mlx{mx.__version__}:chunk{chunk}"


def gdn_prefill_numerics_identity() -> str | None:
    """Name of the active GDN prefill arithmetic when it differs from stock.

    None while this route is not installed: the stock and blocked kernels run
    the same per-token recurrence and need no fingerprint component.
    """
    if not _PATCH_STATE["installed"]:
        return None
    return _PATCH_STATE["identity"]


def install_gdn_fast_prefill_patch() -> dict:
    """Route prefill-sized GDN calls through ``mx.fast.gated_delta_update``.

    Wraps whatever ``mlx_lm.models.gated_delta.gated_delta_update`` is at
    install time (stock, or the blocked prefill kernel when that is on), so
    every call this route declines keeps the previous behaviour. Patches the
    defining module and every loaded mlx-lm module that bound the name at
    import (qwen3_5, qwen3_next). Idempotent.
    """
    if _PATCH_STATE["installed"]:
        return {"installed": True, "already": True}
    if not fast_prefill_available():
        return {
            "installed": False,
            "error": f"mx.fast.gated_delta_update is missing in MLX {mx.__version__} (needs 0.32.4)",
        }
    try:
        from mlx_lm.models import gated_delta as _gd
    except ImportError as exc:  # pragma: no cover - environment without mlx_lm
        return {"installed": False, "error": f"mlx_lm import failed: {exc}"}

    previous = _gd.gated_delta_update
    compute_g = _gd.compute_g
    min_t = _min_route_t()

    def patched(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        prefill_sized = q.ndim == 4 and q.shape[1] >= min_t
        if (
            prefill_sized
            and use_kernel
            and mx.default_device() == mx.gpu
            and fast_prefill_eligible(q, v, a, mask, state)
        ):
            # beta as the stock path computes it (sigmoid in the input type);
            # the float32 cast only keeps the op from rounding it again.
            beta = mx.sigmoid(b)
            g = compute_g(A_log, a, dt_bias)
            ROUTE_COUNTS["routed"] += 1
            return gated_delta_fast_prefill(q, k, v, g, beta, state)
        if prefill_sized:
            ROUTE_COUNTS["fallback_prefill_sized"] += 1
            _note_demotion("gdn_fast_prefill_not_engaged", _NOT_ENGAGED_REASON)
        return previous(
            q, k, v, a, b, A_log, dt_bias, state=state, mask=mask, use_kernel=use_kernel
        )

    _gd.gated_delta_update = patched
    patched_sites = ["mlx_lm.models.gated_delta"]
    # Model modules bind the name at import; import the known ones first so
    # they exist to be swept before the model loads.
    for name in ("mlx_lm.models.qwen3_5", "mlx_lm.models.qwen3_next"):
        with contextlib.suppress(ImportError):
            importlib.import_module(name)
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("mlx_lm"):
            continue
        if getattr(mod, "gated_delta_update", None) is previous:
            mod.gated_delta_update = patched
            patched_sites.append(name)
    identity = _route_identity()
    _PATCH_STATE.update(installed=True, previous=previous, identity=identity)
    report = {
        "installed": True,
        "already": False,
        "sites": patched_sites,
        "min_t": min_t,
        "identity": identity,
    }
    # logger.info at runtime setup is not captured in the serve console.
    print(f"[mtplx] gdn-fast-prefill {report}", flush=True)
    return report


def uninstall_gdn_fast_prefill_patch() -> None:
    if not _PATCH_STATE["installed"]:
        return
    previous = _PATCH_STATE["previous"]
    from mlx_lm.models import gated_delta as _gd

    patched = _gd.gated_delta_update
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("mlx_lm"):
            continue
        if getattr(mod, "gated_delta_update", None) is patched:
            mod.gated_delta_update = previous
    _PATCH_STATE.update(installed=False, previous=None, identity=None)
