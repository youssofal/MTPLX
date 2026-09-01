"""Behavioural gates for the direct Steel QSA prefill consumer.

None of these need the native ``.so``: the extension is stubbed, so they run
on any machine and pin the parts that decide whether a real build is used
correctly — availability, the ABI canary, the kill switch, the geometry
gates, and the Topk adapter that drops ``block_valid`` at the native ABI
boundary.

The numeric parity tests belong on the Studio with a built extension; see
IMPLEMENTATION_NOTES.md.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import pytest

import mtplx.kernels.qsa_prefill_direct as direct

_ROWS = 8
_TOTAL = 2056  # 2056 // 4 == 514 > 512: past the dense/sparse boundary
_POS_START = _TOTAL - _ROWS
_SCALE = 0.0625


class _FakeExt:
    """Stands in for mtplx_qsa_kernels._ext with the same call surface."""

    # A real build bakes this from the build interpreter's wheel, and the
    # wrapper disables the lane when it does not match the mlx that is
    # imported. A healthy stub therefore has to claim the runtime version.
    BUILT_AGAINST_MLX = mx.__version__
    BUILT_AGAINST_NANOBIND = "2.15.0"
    METAL_LIBRARY = "mtplx_qsa_kernels"

    def __init__(self):
        self.calls = []

    def abi_probe(self, a):
        return a.size

    def qwen4_qsa_sparse_gqa_attention(
        self, q, k, v, selected, scale, q_offset, key_tile=64, dimension_tile=64
    ):
        self.calls.append(
            dict(
                q_dtype=q.dtype,
                selected_shape=tuple(selected.shape),
                selected_dtype=selected.dtype,
                scale=scale,
                q_offset=q_offset,
                key_tile=key_tile,
                dimension_tile=dimension_tile,
                k_len=int(k.shape[2]),
            )
        )
        return mx.zeros(tuple(q.shape), dtype=q.dtype)


def _inputs(dtype=mx.bfloat16, *, rows=_ROWS, total=_TOTAL):
    q = mx.zeros((1, 24, rows, 256), dtype=dtype)
    k = mx.zeros((1, 2, total, 256), dtype=dtype)
    v = mx.zeros((1, 2, total, 256), dtype=dtype)
    ids, valid = _selection(total - rows, rows)
    return q, k, v, ids, valid


def _selection(pos_start: int, rows: int):
    """A valid-prefix selection exactly as the production selectors emit."""

    ids = []
    valid = []
    for r in range(rows):
        complete = (pos_start + r + 1) // 4
        take = min(512, complete)
        row = list(range(take)) + [0] * (512 - take)
        ids.append(row)
        valid.append([True] * take + [False] * (512 - take))
    return mx.array(ids, dtype=mx.int32), mx.array(valid, dtype=mx.bool_)


def _kwargs(**overrides):
    base = dict(
        pos_start=_POS_START,
        total_tokens=_TOTAL,
        scale=_SCALE,
        compress_ratio=4,
        block_topk=512,
    )
    base.update(overrides)
    return base


@pytest.fixture()
def ext(monkeypatch):
    """Install the stub extension and pretend we are on a Metal GPU."""

    fake = _FakeExt()
    monkeypatch.setattr(direct, "_EXT", fake)
    monkeypatch.setattr(direct, "_on_metal_device", lambda: True)
    monkeypatch.setattr(direct, "_PIPELINE_STATE", direct._PIPELINE_UNPROVEN)
    monkeypatch.setattr(direct, "_PIPELINE_PROVEN_DTYPES", frozenset())
    monkeypatch.delenv("MTPLX_QSA_PREFILL_DIRECT", raising=False)
    monkeypatch.delenv("MTPLX_QSA_PREFILL_DIRECT_VALIDATE", raising=False)
    return fake


# --------------------------------------------------------------------------
# 1. Missing extension: import stays green, lane reports unsupported
# --------------------------------------------------------------------------


def test_import_without_extension_never_raises(monkeypatch):
    monkeypatch.setattr(direct, "_EXT", None)
    monkeypatch.setattr(direct, "_on_metal_device", lambda: True)
    q, k, v, ids, valid = _inputs()

    assert direct.qsa_prefill_direct_module_ready() is False
    assert direct.qsa_prefill_direct_ready() is False
    assert direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs()) is False
    assert direct.qsa_prefill_direct_build_info() == {}
    # And the producer auto-gate must not arm the lane for a consumer that
    # does not exist.
    assert direct.qsa_prefill_direct_ready() is False


def test_unsupported_call_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(direct, "_EXT", None)
    q, k, v, ids, valid = _inputs()
    with pytest.raises(ValueError, match="not loaded"):
        direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())


# --------------------------------------------------------------------------
# 2. ABI mismatch: one warning, lane disabled, no per-call TypeError storm
# --------------------------------------------------------------------------


def test_abi_mismatch_disables_the_lane_with_one_warning(caplog):
    class _Mismatched(_FakeExt):
        def abi_probe(self, a):
            raise TypeError("incompatible function arguments")

    with caplog.at_level(logging.WARNING, logger=direct.__name__):
        ext, error = direct._verify_abi(_Mismatched(), None)

    assert ext is None
    assert isinstance(error, TypeError)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "nanobind" in warnings[0].getMessage()


def test_extension_without_abi_probe_is_refused(caplog):
    class _NoProbe:
        pass

    with caplog.at_level(logging.WARNING, logger=direct.__name__):
        ext, _ = direct._verify_abi(_NoProbe(), None)
    assert ext is None


def test_healthy_extension_passes_the_probe():
    fake = _FakeExt()
    ext, error = direct._verify_abi(fake, None)
    assert ext is fake and error is None


# --------------------------------------------------------------------------
# 3. Kill switch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_kill_switch_disables_only_this_consumer(ext, monkeypatch, value):
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT", value)
    q, k, v, ids, valid = _inputs()

    assert direct.qsa_prefill_direct_enabled() is False
    assert direct.qsa_prefill_direct_ready() is False
    # The module is still loaded — only the consumer is off.
    assert direct.qsa_prefill_direct_module_ready() is True
    assert direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs()) is False
    with pytest.raises(ValueError, match="MTPLX_QSA_PREFILL_DIRECT is off"):
        direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())
    assert ext.calls == []


def test_default_is_on_when_the_module_is_ready(ext):
    q, k, v, ids, valid = _inputs()
    assert direct.qsa_prefill_direct_ready() is True
    assert direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs()) is True


# --------------------------------------------------------------------------
# 4. Geometry gates: each violation individually fails closed
# --------------------------------------------------------------------------


def test_baseline_geometry_dispatches_once(ext):
    q, k, v, ids, valid = _inputs()
    out = direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())
    assert tuple(out.shape) == (1, 24, _ROWS, 256)
    assert len(ext.calls) == 1
    call = ext.calls[0]
    assert call["selected_shape"] == (1, 1, _ROWS, 512)
    assert call["selected_dtype"] == mx.uint32
    assert call["q_offset"] == _POS_START
    assert call["scale"] == _SCALE
    assert (call["key_tile"], call["dimension_tile"]) == (64, 64)
    # kL is the LOGICAL cache length, never the capacity backing.
    assert call["k_len"] == _TOTAL


def _violations():
    """One malformed argument per case; everything else stays production."""

    q, k, v, ids, valid = _inputs()
    fp16_q = mx.zeros((1, 24, _ROWS, 256), dtype=mx.float16)
    cases = {
        "batch": dict(queries=mx.zeros((2, 24, _ROWS, 256), dtype=mx.bfloat16)),
        "query_heads": dict(queries=mx.zeros((1, 16, _ROWS, 256), dtype=mx.bfloat16)),
        "head_dim": dict(queries=mx.zeros((1, 24, _ROWS, 128), dtype=mx.bfloat16)),
        "rank": dict(queries=mx.zeros((24, _ROWS, 256), dtype=mx.bfloat16)),
        "kv_heads": dict(keys=mx.zeros((1, 4, _TOTAL, 256), dtype=mx.bfloat16)),
        "kv_shape_mismatch": dict(
            values=mx.zeros((1, 2, _TOTAL - 8, 256), dtype=mx.bfloat16)
        ),
        "dtype_fp32": dict(
            queries=mx.zeros((1, 24, _ROWS, 256), dtype=mx.float32),
            keys=mx.zeros((1, 2, _TOTAL, 256), dtype=mx.float32),
            values=mx.zeros((1, 2, _TOTAL, 256), dtype=mx.float32),
        ),
        "dtype_mix": dict(queries=fp16_q),
        "ids_dtype": dict(block_ids=ids.astype(mx.int64)),
        "valid_dtype": dict(block_valid=valid.astype(mx.int32)),
        "ids_width": dict(
            block_ids=mx.zeros((_ROWS, 256), dtype=mx.int32),
            block_valid=mx.zeros((_ROWS, 256), dtype=mx.bool_),
        ),
        "ratio": dict(compress_ratio=2),
        "topk": dict(block_topk=256),
        "scale": dict(scale=0.125),
        "scale_traced": dict(scale=mx.array(0.0625)),
        "pos_traced": dict(pos_start=mx.array(_POS_START)),
        "non_suffix": dict(pos_start=_POS_START - 1),
        # kL must equal the logical frontier: a capacity backing is refused.
        "capacity_backing": dict(
            keys=mx.zeros((1, 2, _TOTAL + 4096, 256), dtype=mx.bfloat16),
            values=mx.zeros((1, 2, _TOTAL + 4096, 256), dtype=mx.bfloat16),
        ),
        "below_boundary": dict(
            keys=mx.zeros((1, 2, 2048, 256), dtype=mx.bfloat16),
            values=mx.zeros((1, 2, 2048, 256), dtype=mx.bfloat16),
            pos_start=2048 - _ROWS,
            total_tokens=2048,
            block_ids=_selection(2048 - _ROWS, _ROWS)[0],
            block_valid=_selection(2048 - _ROWS, _ROWS)[1],
        ),
        "negative_pos": dict(pos_start=-1),
    }
    base = dict(queries=q, keys=k, values=v, block_ids=ids, block_valid=valid)
    for name, override in cases.items():
        merged = dict(base)
        kw = _kwargs()
        for key, value in override.items():
            if key in merged:
                merged[key] = value
            else:
                kw[key] = value
        yield name, merged, kw


@pytest.mark.parametrize("name,args,kwargs", list(_violations()), ids=lambda x: x)
def test_each_geometry_violation_fails_closed(ext, name, args, kwargs):
    ordered = (
        args["queries"],
        args["keys"],
        args["values"],
        args["block_ids"],
        args["block_valid"],
    )
    assert direct.qsa_prefill_direct_supported(*ordered, **kwargs) is False, name
    with pytest.raises(ValueError):
        direct.qsa_prefill_direct(*ordered, **kwargs)
    assert ext.calls == [], "an unsupported call reached the native symbol"


def test_cpu_device_fails_closed(monkeypatch):
    monkeypatch.setattr(direct, "_EXT", _FakeExt())
    monkeypatch.setattr(direct, "_on_metal_device", lambda: False)
    q, k, v, ids, valid = _inputs()
    assert direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs()) is False


def test_float16_is_supported_alongside_bfloat16(ext):
    q, k, v, ids, valid = _inputs(dtype=mx.float16)
    assert direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs()) is True


# --------------------------------------------------------------------------
# Topk adapter: the seam where block_valid is dropped
# --------------------------------------------------------------------------


def test_topk_buffer_shape_dtype_and_values():
    ids, valid = _selection(_POS_START, _ROWS)
    buffer = direct.qsa_prefill_direct_topk_buffer(
        ids, valid, pos_start=_POS_START, validate=True
    )
    mx.eval(buffer)
    assert tuple(buffer.shape) == (1, 1, _ROWS, 512)
    assert buffer.dtype == mx.uint32
    assert bool(mx.array_equal(buffer[0, 0], ids.astype(mx.uint32)).item())


@pytest.mark.parametrize("complete", [0, 1, 511, 512, 513])
def test_topk_buffer_accepts_every_prefix_length(complete):
    """Early rows (complete < 512), the exact boundary, and full rows."""

    pos_start = complete * 4  # q_abs == pos_start -> complete blocks visible
    ids, valid = _selection(pos_start, 1)
    buffer = direct.qsa_prefill_direct_topk_buffer(
        ids, valid, pos_start=pos_start, validate=True
    )
    mx.eval(buffer)
    assert tuple(buffer.shape) == (1, 1, 1, 512)


@pytest.mark.parametrize("tail", [0, 1, 2, 3])
def test_topk_buffer_accepts_every_causal_tail_length(tail):
    pos_start = 4096 + tail
    ids, valid = _selection(pos_start, 4)
    buffer = direct.qsa_prefill_direct_topk_buffer(
        ids, valid, pos_start=pos_start, validate=True
    )
    mx.eval(buffer)
    assert tuple(buffer.shape) == (1, 1, 4, 512)


def _mutate(ids: mx.array, valid: mx.array, row, slot, *, id_=None, ok=None):
    ids_list = ids.tolist()
    valid_list = valid.tolist()
    if id_ is not None:
        ids_list[row][slot] = id_
    if ok is not None:
        valid_list[row][slot] = ok
    return (
        mx.array(ids_list, dtype=mx.int32),
        mx.array(valid_list, dtype=mx.bool_),
    )


def test_validity_hole_is_rejected():
    """The failure the native ABI cannot express: a false inside the prefix
    means the kernel reads that slot anyway and ignores a later valid one."""

    pos_start = 400  # 100 complete blocks: a short, easy-to-poke prefix
    ids, valid = _selection(pos_start, 1)
    ids, valid = _mutate(ids, valid, 0, 3, ok=False)
    with pytest.raises(ValueError, match="prefix"):
        direct.qsa_prefill_direct_topk_buffer(
            ids, valid, pos_start=pos_start, validate=True
        )


def test_unsorted_valid_ids_are_rejected():
    pos_start = 400
    ids, valid = _selection(pos_start, 1)
    swapped = ids.tolist()
    swapped[0][2], swapped[0][5] = swapped[0][5], swapped[0][2]
    with pytest.raises(ValueError, match="ascending"):
        direct.qsa_prefill_direct_topk_buffer(
            mx.array(swapped, dtype=mx.int32),
            valid,
            pos_start=pos_start,
            validate=True,
        )


def test_duplicate_valid_ids_are_rejected():
    pos_start = 400
    ids, valid = _selection(pos_start, 1)
    ids, valid = _mutate(ids, valid, 0, 5, id_=4)  # equal to slot 4: not strict
    with pytest.raises(ValueError, match="ascending"):
        direct.qsa_prefill_direct_topk_buffer(
            ids, valid, pos_start=pos_start, validate=True
        )


def test_negative_valid_id_is_rejected():
    pos_start = 400
    ids, valid = _selection(pos_start, 1)
    ids, valid = _mutate(ids, valid, 0, 0, id_=-1)
    with pytest.raises(ValueError, match="negative"):
        direct.qsa_prefill_direct_topk_buffer(
            ids, valid, pos_start=pos_start, validate=True
        )


def test_out_of_range_valid_id_is_rejected():
    """An id at or past the row's complete-block count is not causally
    visible; the kernel would mask it, but that is luck, not contract."""

    pos_start = 400
    ids, valid = _selection(pos_start, 1)
    # Last valid slot, so the ascending check still passes and only the
    # visibility clause can fire.
    ids, valid = _mutate(ids, valid, 0, 99, id_=9999)
    with pytest.raises(ValueError, match="visible"):
        direct.qsa_prefill_direct_topk_buffer(
            ids, valid, pos_start=pos_start, validate=True
        )


def test_short_valid_count_is_rejected():
    """valid_blocks is positional in the kernel: min(512, complete). A row
    that selected fewer than it should would silently attend padding."""

    pos_start = 400
    ids, valid = _selection(pos_start, 1)
    ids, valid = _mutate(ids, valid, 0, 99, ok=False)
    with pytest.raises(ValueError, match="valid count"):
        direct.qsa_prefill_direct_topk_buffer(
            ids, valid, pos_start=pos_start, validate=True
        )


def test_adapter_rejects_wrong_static_layout():
    ids, valid = _selection(_POS_START, _ROWS)
    with pytest.raises(ValueError, match="int32"):
        direct.qsa_prefill_direct_topk_buffer(
            ids.astype(mx.uint32), valid, pos_start=_POS_START
        )
    with pytest.raises(ValueError, match="bool"):
        direct.qsa_prefill_direct_topk_buffer(
            ids, valid.astype(mx.int32), pos_start=_POS_START
        )
    with pytest.raises(ValueError, match="512 slots"):
        direct.qsa_prefill_direct_topk_buffer(
            ids[:, :256], valid[:, :256], pos_start=_POS_START
        )


def test_hot_path_does_not_validate_by_default(ext, monkeypatch):
    """Per-call validation synchronizes; the producer contract is pinned by
    tests/test_qsa_selector_prefix_contract.py instead."""

    calls = []
    monkeypatch.setattr(
        direct,
        "_prefix_contract_violation",
        lambda *a, **kw: calls.append(1) or None,
    )
    q, k, v, ids, valid = _inputs()
    direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())
    assert calls == []

    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT_VALIDATE", "1")
    direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())
    assert calls == [1]


# --------------------------------------------------------------------------
# Pipeline proof
# --------------------------------------------------------------------------


def test_first_dispatch_is_evaluated_then_later_ones_are_not(ext, monkeypatch):
    evals = []
    real_eval = mx.eval
    monkeypatch.setattr(mx, "eval", lambda *a, **kw: evals.append(a) or real_eval(*a, **kw))
    q, k, v, ids, valid = _inputs()

    direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())
    assert len(evals) == 1, "the first dispatch must prove the Metal pipeline"
    direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())
    assert len(evals) == 1, "later dispatches must stay lazy"


def test_build_info_reports_the_wheel_the_extension_was_built_against(ext):
    info = direct.qsa_prefill_direct_build_info()
    assert info["built_against_mlx"] == mx.__version__
    assert info["built_against_nanobind"] == "2.15.0"
    assert info["metal_library"] == "mtplx_qsa_kernels"
    assert "imported_mlx" in info


# --------------------------------------------------------------------------
# Fail-closed process state: a failed proof and a bad build receipt both
# retire the lane (Codex CHANGES_REQUIRED follow-up)
# --------------------------------------------------------------------------


def test_failed_preflight_disables_the_lane_for_the_process(ext, monkeypatch):
    """Symbol presence is not pipeline readiness.

    A present .so with a missing, stale, or wrongly named metallib fails
    inside the Metal pipeline creation. Before this, the preflight returned
    False and ``ready()`` still said True, so the M3 auto-gate armed a
    producer for a consumer that could not run and every request re-hit the
    same wall.
    """

    def _boom(*args, **kwargs):
        raise RuntimeError("failed to create Metal pipeline state")

    monkeypatch.setattr(ext, "qwen4_qsa_sparse_gqa_attention", _boom)

    assert direct.qsa_prefill_direct_preflight() is False

    monkeypatch.setattr(ext, "qwen4_qsa_sparse_gqa_attention", _FakeExt().qwen4_qsa_sparse_gqa_attention)
    ext.calls.clear()

    assert direct.qsa_prefill_direct_ready() is False
    q, k, v, ids, valid = _inputs()
    assert direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs()) is False
    with pytest.raises(ValueError, match="failed Metal pipeline"):
        direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())
    assert ext.calls == [], "a retired lane must never reach the native symbol"
    # And a second preflight stays refused rather than retrying per request.
    assert direct.qsa_prefill_direct_preflight() is False


def test_failed_preflight_disarms_the_m3_producer_auto_gate(ext, monkeypatch):
    """The gate that decides whether the selector emits the block tuple at
    all. With no NAX consumer, a dead direct lane must leave it off."""

    import mtplx.kernels.qsa_indexer_select as select_module
    import mtplx.models.qwen4_exp as qwen4_exp

    monkeypatch.setattr(
        select_module, "qsa_indexer_select_nax_available", lambda: False
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)

    def _boom(*args, **kwargs):
        raise RuntimeError("failed to create Metal pipeline state")

    monkeypatch.setattr(ext, "qwen4_qsa_sparse_gqa_attention", _boom)
    assert direct.qsa_prefill_direct_preflight() is False
    assert qwen4_exp.qsa_prefill_lane_auto_supported() is False


def test_failed_first_dispatch_eval_also_retires_the_lane(ext, monkeypatch):
    """The proof can also fail on the first REAL request: get_library and
    get_kernel failures are eval-time, not dispatch-time."""

    real_eval = mx.eval

    def _boom_eval(*args, **kwargs):
        raise RuntimeError("failed to load metallib mtplx_qsa_kernels")

    monkeypatch.setattr(mx, "eval", _boom_eval)
    q, k, v, ids, valid = _inputs()
    with pytest.raises(RuntimeError, match="metallib"):
        direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())

    monkeypatch.setattr(mx, "eval", real_eval)
    assert direct.qsa_prefill_direct_ready() is False
    assert direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs()) is False


def test_a_proven_pipeline_is_only_proved_once(ext):
    """ready() proves a real loaded extension the first time it is asked —
    once per packaged dtype specialization, because the Metal kernel name
    embeds the query dtype. Afterwards it is a pure state read, not a
    dispatch per request."""

    assert direct.qsa_prefill_direct_ready() is True
    assert len(ext.calls) == len(direct._SUPPORTED_DTYPES)
    assert {call["q_dtype"] for call in ext.calls} == set(direct._SUPPORTED_DTYPES)
    assert direct.qsa_prefill_direct_ready() is True
    assert len(ext.calls) == len(direct._SUPPORTED_DTYPES)


def _dtype_gated_ext(ext, monkeypatch, *, broken):
    """Let every dtype through except ``broken``, which fails pipeline
    creation the way a metallib missing that specialization does."""

    healthy = ext.qwen4_qsa_sparse_gqa_attention

    def _gated(q, k, v, selected, scale, q_offset, key_tile=64, dimension_tile=64):
        if q.dtype == broken:
            raise RuntimeError(
                "failed to create Metal pipeline state object for "
                f"qwen4_qsa_sparse_gqa_{broken}_bk64_dc64"
            )
        return healthy(
            q,
            k,
            v,
            selected,
            scale,
            q_offset,
            key_tile=key_tile,
            dimension_tile=dimension_tile,
        )

    monkeypatch.setattr(ext, "qwen4_qsa_sparse_gqa_attention", _gated)


def test_a_proved_bfloat16_pipeline_does_not_prove_float16(ext, monkeypatch):
    """The native kernel name carries the query dtype
    (``qwen4_qsa_sparse_gqa_<type>_bk64_dc64_...``), so a metallib can hold a
    good bfloat16 specialization beside a missing or stale float16 one. One
    global PROVEN flag would let that build report ready and supported and
    then die on the first float16 request — with the first-dispatch proof
    skipped, so it would not even retire the lane. This lane fails closed for
    the whole process instead."""

    _dtype_gated_ext(ext, monkeypatch, broken=mx.float16)

    # bfloat16 alone proves cleanly...
    assert direct.qsa_prefill_direct_preflight(dtype=mx.bfloat16) is True
    assert direct._dtype_proven(mx.bfloat16) is True
    # ...and does NOT carry float16, which fails its own proof.
    assert direct.qsa_prefill_direct_preflight(dtype=mx.float16) is False

    # Whole-lane fail-closed: neither dtype is ready, supported, or usable.
    assert direct.qsa_prefill_direct_ready() is False
    assert direct._PIPELINE_STATE == direct._PIPELINE_FAILED
    for dtype in (mx.float16, mx.bfloat16):
        q, k, v, ids, valid = _inputs(dtype=dtype)
        assert (
            direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs())
            is False
        ), dtype
        with pytest.raises(ValueError, match="failed Metal pipeline"):
            direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())


def test_ready_refuses_when_only_one_packaged_dtype_can_load(ext, monkeypatch):
    """ready() is the M3 producer auto-gate. It must prove every packaged
    specialization itself, not just the bfloat16 default."""

    _dtype_gated_ext(ext, monkeypatch, broken=mx.float16)

    assert direct.qsa_prefill_direct_ready() is False
    assert direct.qsa_prefill_direct_ready() is False, "and it stays refused"


def test_failed_stays_terminal_when_a_later_proof_succeeds(ext, monkeypatch):
    """The check-then-set race, made deterministic: FAILED is documented as
    terminal, so a proof that lands after it must not write PROVEN over it
    and reopen readiness."""

    monkeypatch.setattr(direct, "_PIPELINE_STATE", direct._PIPELINE_FAILED)
    monkeypatch.setattr(direct, "_PIPELINE_PROVEN_DTYPES", frozenset())

    # The extension is perfectly healthy; the lane is retired anyway.
    assert direct.qsa_prefill_direct_preflight() is False
    assert direct.qsa_prefill_direct_preflight(dtype=mx.float16) is False
    assert ext.calls == [], "a retired lane must never reach the native symbol"

    # The racing writer: a proof that had already started before the failure
    # and only now reports success.
    direct._record_pipeline_success(mx.bfloat16)
    direct._record_pipeline_success(mx.float16)

    assert direct._PIPELINE_STATE == direct._PIPELINE_FAILED
    assert direct._PIPELINE_PROVEN_DTYPES == frozenset()
    assert direct.qsa_prefill_direct_ready() is False


def test_concurrent_first_proofs_cannot_lose_a_failure(ext, monkeypatch):
    """Extra, non-deterministic cover for the same invariant: many threads
    race their first proof, one dtype cannot load, and the lane ends FAILED
    no matter which thread finishes last."""

    import threading

    _dtype_gated_ext(ext, monkeypatch, broken=mx.float16)

    barrier = threading.Barrier(8)
    results: list[bool] = []
    lock = threading.Lock()

    def _worker(dtype):
        barrier.wait()
        ok = direct.qsa_prefill_direct_preflight(dtype=dtype)
        with lock:
            results.append(ok)

    threads = [
        threading.Thread(target=_worker, args=(mx.bfloat16 if i % 2 else mx.float16,))
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert direct._PIPELINE_STATE == direct._PIPELINE_FAILED
    assert direct.qsa_prefill_direct_ready() is False


def test_mismatched_mlx_build_receipt_disables_the_lane(ext, monkeypatch, caplog):
    """The extension links MLX's private C++ ABI. A .so built against a
    different wheel imports, passes abi_probe, lists every symbol, and then
    mis-reads a struct — so the receipt is a gate, not a note in a dict."""

    monkeypatch.setattr(ext, "BUILT_AGAINST_MLX", "0.31.0", raising=False)
    monkeypatch.setattr(direct, "_RECEIPT_WARNED", False)
    q, k, v, ids, valid = _inputs()

    with caplog.at_level(logging.WARNING, logger=direct.__name__):
        assert direct.qsa_prefill_direct_ready() is False
        assert direct.qsa_prefill_direct_ready() is False
        assert (
            direct.qsa_prefill_direct_supported(q, k, v, ids, valid, **_kwargs())
            is False
        )
        with pytest.raises(ValueError, match="built against mlx 0.31.0"):
            direct.qsa_prefill_direct(q, k, v, ids, valid, **_kwargs())

    assert ext.calls == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "one warning, not one per layer per token"
    assert "0.31.0" in warnings[0].getMessage()
    # The build info still reports both sides so the operator can see why.
    info = direct.qsa_prefill_direct_build_info()
    assert info["built_against_mlx"] == "0.31.0"
    assert info["imported_mlx"] == mx.__version__


def test_missing_build_receipt_is_refused(ext, monkeypatch):
    """An extension with no receipt predates the check; it is not trusted."""

    monkeypatch.delattr(type(ext), "BUILT_AGAINST_MLX", raising=False)
    monkeypatch.setattr(direct, "_RECEIPT_WARNED", False)
    assert direct.qsa_prefill_direct_ready() is False
    assert ext.calls == []


def test_matching_receipt_keeps_the_lane_open(ext):
    assert direct.qsa_prefill_direct_build_info()["built_against_mlx"] == mx.__version__
    assert direct.qsa_prefill_direct_ready() is True


# --------------------------------------------------------------------------
# Native boundary: the C++ unsupported() must refuse what Python refuses.
# These need a built .so; they are the live half of
# test_cpp_unsupported_enforces_the_logical_view_and_scale_contracts.
# --------------------------------------------------------------------------

_NO_EXTENSION = pytest.mark.skipif(
    direct._EXT is None or not hasattr(direct._EXT, "qwen4_qsa_sparse_gqa_attention"),
    reason="no built mtplx_qsa_kernels extension on this machine",
)


def _native_args(*, total=_TOTAL, rows=_ROWS, dtype=mx.bfloat16):
    q = mx.zeros((1, 24, rows, 256), dtype=dtype)
    k = mx.zeros((1, 2, total, 256), dtype=dtype)
    v = mx.zeros((1, 2, total, 256), dtype=dtype)
    selected = mx.zeros((1, 1, rows, 512), dtype=mx.uint32)
    return q, k, v, selected


@_NO_EXTENSION
def test_native_symbol_rejects_a_q_window_that_is_not_the_suffix():
    """params.kL IS k.shape(2). A shorter Q window would launch and attend
    the wrong rows, so the ABI takes equality, not <=."""

    q, k, v, selected = _native_args()
    with pytest.raises(ValueError):
        direct._EXT.qwen4_qsa_sparse_gqa_attention(
            q,
            k,
            v,
            selected,
            _SCALE,
            _POS_START - 1,  # q_offset + qL < kL
            key_tile=64,
            dimension_tile=64,
        )


@_NO_EXTENSION
def test_native_symbol_rejects_a_non_production_scale():
    q, k, v, selected = _native_args()
    with pytest.raises(ValueError):
        direct._EXT.qwen4_qsa_sparse_gqa_attention(
            q,
            k,
            v,
            selected,
            0.125,  # not 1/sqrt(256)
            _POS_START,
            key_tile=64,
            dimension_tile=64,
        )
