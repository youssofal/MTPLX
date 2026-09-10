"""MTPLX_QWEN4_OPDIET - value-identity proofs for the compiled-verifier op diet.

Every rewrite behind ``MTPLX_QWEN4_OPDIET`` claims to be *value identical* to
the expression it replaces, not merely close. These tests hold each rewrite
next to its original on random inputs and compare RAW BITS (through an integer
view, so a sign-flipped zero or a one-ulp drift fails), and they assert the
op-graph shrink that motivates each rewrite by walking the compiled graph
instead of trusting a comment.

They run entirely on the CPU stream with tiny tensors: no Metal, no model, no
kernels. ``mx.export_to_dot`` builds no kernels either -- it prints the graph.
"""

from __future__ import annotations

import re

import mlx.core as mx
import pytest

import mtplx.fast_sampling as fast_sampling
import mtplx.models.qwen4_exp as qwen4_exp
import mtplx.runtime_options as runtime_options


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Confine every op in this module to the CPU stream."""

    with mx.stream(mx.cpu):
        yield


@pytest.fixture
def opdiet(monkeypatch):
    """Arm MTPLX_QWEN4_OPDIET for one test."""

    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    return True


def _bits(value: mx.array) -> mx.array:
    """Raw bit pattern, so ``==`` cannot be satisfied by a near miss."""

    widths = {
        mx.bfloat16: mx.uint16,
        mx.float16: mx.uint16,
        mx.float32: mx.uint32,
    }
    return value.view(widths[value.dtype])


def _identical(left: mx.array, right: mx.array) -> bool:
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    mx.eval(left, right)
    return bool(mx.all(_bits(left) == _bits(right)))


def _primitives(*outputs: mx.array) -> list[str]:
    """Primitive labels of the (compiled) graph producing ``outputs``."""

    import io

    buffer = io.StringIO()
    mx.export_to_dot(buffer, *outputs)
    return re.findall(r'label ="([^"]+)"', buffer.getvalue())


def _kernel_primitives(*outputs: mx.array) -> list[str]:
    """Primitives that actually launch work (views are free)."""

    free = {"Reshape", "ExpandDims", "Squeeze", "Slice", "Transpose", "AsStrided"}
    return [p for p in _primitives(*outputs) if p not in free]


# --------------------------------------------------------------------------
# flag plumbing
# --------------------------------------------------------------------------


def test_opdiet_defaults_off_and_is_read_at_use(monkeypatch):
    # Read at USE, not frozen at import: the served fixed-M4 auto-arm stamps
    # MTPLX_QWEN4_OPDIET AFTER this module is imported, so a frozen import-time
    # read left the compiled verifier without the op diet (arming audit). The
    # env is frozen once serving starts, so two traces of one graph still agree.
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", None)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET_SELECTED", None)
    monkeypatch.delenv("MTPLX_QWEN4_OPDIET", raising=False)
    assert runtime_options.qwen4_opdiet_enabled() is False
    monkeypatch.setenv("MTPLX_QWEN4_OPDIET", "1")
    assert runtime_options.qwen4_opdiet_enabled() is True


def test_every_gated_module_reads_the_same_flag():
    assert qwen4_exp.qwen4_opdiet_enabled is runtime_options.qwen4_opdiet_enabled
    assert fast_sampling.qwen4_opdiet_enabled is runtime_options.qwen4_opdiet_enabled


# --------------------------------------------------------------------------
# item 4 - hyper-connection residual write
# --------------------------------------------------------------------------


def _stock_residual_write(hyper, block_out, inject):
    return hyper + (block_out[..., None, :] * inject[..., :, None]).reshape(
        *hyper.shape
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [1, 4])
def test_residual_write_is_bitwise_identical(monkeypatch, dtype, rows):
    hc, hidden = 4, 6
    hyper = mx.random.normal((1, rows, hc * hidden)).astype(dtype)
    block_out = mx.random.normal((1, rows, hidden)).astype(dtype)
    inject = mx.random.normal((1, rows, hc)).astype(dtype)
    mx.eval(hyper, block_out, inject)

    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock = qwen4_exp._hyper_residual_write(hyper, block_out, inject)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    diet = qwen4_exp._hyper_residual_write(hyper, block_out, inject)

    assert _identical(stock, diet)
    assert _identical(stock, _stock_residual_write(hyper, block_out, inject))


def test_residual_write_off_emits_the_stock_two_kernel_graph(monkeypatch):
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    hc, hidden = 4, 6
    args = (
        mx.zeros((1, 2, hc * hidden)),
        mx.zeros((1, 2, hidden)),
        mx.zeros((1, 2, hc)),
    )
    mx.eval(*args)
    out = mx.compile(qwen4_exp._hyper_residual_write)(*args)
    kernels = _kernel_primitives(out)
    # multiply and add are two separate launches, split by the reshape
    assert kernels == ["CompiledBroadcastBroadcastMultiply", "Add"]


def test_residual_write_on_fuses_multiply_and_add(opdiet):
    hc, hidden = 4, 6
    args = (
        mx.zeros((1, 2, hc * hidden)),
        mx.zeros((1, 2, hidden)),
        mx.zeros((1, 2, hc)),
    )
    mx.eval(*args)
    out = mx.compile(qwen4_exp._hyper_residual_write)(*args)
    kernels = _kernel_primitives(out)
    assert kernels == ["CompiledBroadcastBroadcastMultiplyAdd"]


# --------------------------------------------------------------------------
# item 2 - RoPE tables and the partial rotation
# --------------------------------------------------------------------------


def _inv_freq(rot_half: int) -> mx.array:
    rotary = 2 * rot_half
    value = 1.0 / (
        10000.0 ** (mx.arange(0, rotary, 2, dtype=mx.float32) / rotary)
    )
    mx.eval(value)
    return value


@pytest.mark.parametrize("scaling", [1.0, 1.234])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("pass_through", [0, 8])
def test_partial_rope_half_is_bitwise_identical(scaling, dtype, pass_through):
    rot_half, seq, heads = 16, 3, 2
    inv_freq = _inv_freq(rot_half)
    head_dim = 2 * rot_half + pass_through
    x = mx.random.normal((1, seq, heads, head_dim)).astype(dtype)
    positions = mx.arange(5, 5 + seq, dtype=mx.int32)
    mx.eval(x, positions)

    cos, sin = qwen4_exp._rope_cos_sin(positions, inv_freq, scaling)
    stock = qwen4_exp._apply_partial_rope(x, cos, sin)

    cos_h, sin_h = qwen4_exp._rope_cos_sin_half(positions, inv_freq, scaling)
    diet = qwen4_exp._apply_partial_rope_half(x, cos_h, sin_h)

    assert _identical(stock, diet)


def test_half_table_is_the_first_half_of_the_stock_table():
    inv_freq = _inv_freq(16)
    positions = mx.arange(0, 4, dtype=mx.int32)
    mx.eval(positions)
    cos, sin = qwen4_exp._rope_cos_sin(positions, inv_freq, 1.0)
    cos_h, sin_h = qwen4_exp._rope_cos_sin_half(positions, inv_freq, 1.0)

    assert _identical(cos[:, :16], cos_h)
    assert _identical(sin[:, :16], sin_h)
    # ... and the discarded half was an exact duplicate all along.
    assert _identical(cos[:, 16:], cos_h)
    assert _identical(sin[:, 16:], sin_h)


def test_partial_rope_half_drops_the_negate_and_one_concatenate():
    inv_freq = _inv_freq(16)
    positions = mx.arange(0, 3, dtype=mx.int32)
    x = mx.zeros((1, 3, 2, 40), dtype=mx.bfloat16)
    mx.eval(x, positions)

    def stock(x, positions):
        cos, sin = qwen4_exp._rope_cos_sin(positions, inv_freq, 1.0)
        return qwen4_exp._apply_partial_rope(x, cos, sin)

    def diet(x, positions):
        cos, sin = qwen4_exp._rope_cos_sin_half(positions, inv_freq, 1.0)
        return qwen4_exp._apply_partial_rope_half(x, cos, sin)

    stock_kernels = _kernel_primitives(mx.compile(stock)(x, positions))
    diet_kernels = _kernel_primitives(mx.compile(diet)(x, positions))

    # The doubled cos/sin table costs one concatenate; the rotated buffer
    # costs a standalone negate plus a second concatenate.
    assert stock_kernels.count("Concatenate") == 3
    assert diet_kernels.count("Concatenate") == 1
    assert "Negative" in stock_kernels
    assert "Negative" not in diet_kernels
    assert len(diet_kernels) < len(stock_kernels)


def test_rope_table_memo_shares_one_table_per_scope():
    inv_freq = _inv_freq(8)
    offset = mx.array(7, dtype=mx.int32)
    mx.eval(offset)

    with qwen4_exp._rope_table_scope():
        first = qwen4_exp._shared_rope_cos_sin_half(offset, 4, inv_freq, 1.0)
        second = qwen4_exp._shared_rope_cos_sin_half(offset, 4, inv_freq, 1.0)
        assert first[0] is second[0] and first[1] is second[1]
        # A different width, offset object, or table is never confused for it.
        assert qwen4_exp._shared_rope_cos_sin_half(offset, 3, inv_freq, 1.0)[0] is not first[0]
        other = mx.array(7, dtype=mx.int32)
        mx.eval(other)
        assert qwen4_exp._shared_rope_cos_sin_half(other, 4, inv_freq, 1.0)[0] is not first[0]

    # The memo never outlives the forward that opened it.
    outside = qwen4_exp._shared_rope_cos_sin_half(offset, 4, inv_freq, 1.0)
    assert outside[0] is not first[0]
    assert _identical(outside[0], first[0])


def test_shared_rope_table_matches_the_stock_table():
    inv_freq = _inv_freq(8)
    for pos_start in (0, 11, mx.array(11, dtype=mx.int32)):
        positions = pos_start + mx.arange(4, dtype=mx.int32)
        cos, sin = qwen4_exp._rope_cos_sin(positions, inv_freq, 1.0)
        with qwen4_exp._rope_table_scope():
            cos_h, sin_h = qwen4_exp._shared_rope_cos_sin_half(
                pos_start, 4, inv_freq, 1.0
            )
        assert _identical(cos[:, :8], cos_h)
        assert _identical(sin[:, :8], sin_h)


def test_shared_inv_freq_is_one_object_per_args(monkeypatch):
    args = qwen4_exp.TextArgs()
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    plain_a, scale_a = qwen4_exp._rope_inv_freq_and_scaling_for(args)
    plain_b, _ = qwen4_exp._rope_inv_freq_and_scaling_for(args)
    assert plain_a is not plain_b  # stock: a fresh array per module

    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    shared_a, scale_b = qwen4_exp._rope_inv_freq_and_scaling_for(args)
    shared_b, _ = qwen4_exp._rope_inv_freq_and_scaling_for(args)
    assert shared_a is shared_b
    assert scale_a == scale_b
    assert _identical(plain_a, shared_a)


# --------------------------------------------------------------------------
# item 1 - fixed QSA pooled bank update
# --------------------------------------------------------------------------


def _stock_bank_update(pooled, candidate, safe_block, condition):
    updated = mx.slice_update(pooled, candidate, safe_block, axes=(1,))
    return mx.where(condition, updated, pooled)


def _diet_bank_update(pooled, candidate, safe_block, condition):
    """The shipped rowsel form: resolve the condition on the touched ROW."""

    old_row = mx.slice(
        pooled, safe_block, axes=(1,), slice_size=(1, 1, pooled.shape[2])
    )
    merged = mx.where(condition, candidate.astype(pooled.dtype), old_row)
    return mx.slice_update(pooled, merged, safe_block, axes=(1,))


def _select_bank_update(pooled, candidate, safe_block, condition):
    """The row-id select form: fewer dispatches, but measured SLOWER.

    Kept as a third value-identical spelling so the choice stays visible --
    shipped rowsel form reached -49%, because both of its operands broadcast
    and MLX must emit a general (strided) select over the whole bank.
    """

    row_ids = mx.arange(pooled.shape[1], dtype=safe_block.dtype).reshape(
        1, pooled.shape[1]
    )
    write_row = mx.logical_and(row_ids == safe_block, condition)
    return mx.where(write_row[..., None], candidate.astype(pooled.dtype), pooled)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_bank_update_is_bitwise_identical_for_every_row_and_condition(dtype):
    capacity, head_dim = 12, 8
    pooled = mx.random.normal((1, capacity, head_dim)).astype(dtype)
    candidate = mx.random.normal((1, 1, head_dim)).astype(dtype)
    mx.eval(pooled, candidate)

    for row in range(capacity):
        for condition in (True, False):
            block = mx.array(row, dtype=mx.int32)
            flag = mx.array(condition)
            mx.eval(block, flag)
            stock = _stock_bank_update(pooled, candidate, block, flag)
            assert _identical(
                stock, _diet_bank_update(pooled, candidate, block, flag)
            ), (row, condition)
            assert _identical(
                stock, _select_bank_update(pooled, candidate, block, flag)
            ), (row, condition)


def test_bank_update_drops_one_full_bank_pass_and_keeps_the_rest_contiguous():
    """Dispatch count is not the objective; kernel class is.

    The shipped form issues MORE primitives than the row-id select and still
    measured twice as fast, because its one full-bank pass is a contiguous
    copy while the select's is a general strided kernel over the whole bank.
    This pins the shape of all three graphs so a future "cheaper" rewrite has
    to argue against the measurement, not against the op count.
    """

    capacity, head_dim = 12, 8
    args = (
        mx.zeros((1, capacity, head_dim)),
        mx.zeros((1, 1, head_dim)),
        mx.array(3, dtype=mx.int32),
        mx.array(True),
    )
    mx.eval(*args)
    stock = _kernel_primitives(mx.compile(_stock_bank_update)(*args))
    diet = _kernel_primitives(mx.compile(_diet_bank_update)(*args))
    select = _kernel_primitives(mx.compile(_select_bank_update)(*args))

    # Stock: copy the whole bank (DynamicSliceUpdate cannot donate a leaf the
    # caller still holds) and then select over the whole bank a second time.
    assert stock == ["DynamicSliceUpdate", "CompiledBroadcastSelect"]
    # Shipped: read the row, select head_dim wide, write the row back. The
    # only full-bank pass left is the slice_update's contiguous copy.
    assert diet == ["DynamicSlice", "CompiledBroadcastSelect", "DynamicSliceUpdate"]
    # Rejected: one pass, but both operands broadcast across the whole bank.
    assert select[-1] == "CompiledBroadcastBroadcastSelect"
    assert "DynamicSliceUpdate" not in select


# --------------------------------------------------------------------------
# item 3 - MoE routing scaffold (documented as not applicable)
# --------------------------------------------------------------------------


def test_moe_routing_scaffold_has_no_python_side_constant_to_hoist():
    """The two ``arange`` dispatches in the M4 router are not ours.

    The census attributes an ``arangeuint32[4]`` and an ``arangeuint32[40]``
    to the routing scaffold and proposes hoisting them to construction time.
    The traced graph of that exact scaffold contains no Arange primitive at
    all: both come from inside MLX's own lowering of ArgPartition/GatherAxis,
    below the Python API, so there is nothing here to hoist. Recorded as a
    test so the claim is checked against the installed MLX rather than
    remembered.
    """

    def route(gate_logits):
        gates = mx.softmax(gate_logits, axis=-1, precise=True)
        expert_ids = mx.argpartition(gates, kth=-10, axis=-1)[..., -10:]
        route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
        return expert_ids, route_scores / route_scores.sum(axis=-1, keepdims=True)

    gate_logits = mx.zeros((1, 4, 512))
    mx.eval(gate_logits)
    primitives = _primitives(*mx.compile(route)(gate_logits))
    assert "Arange" not in primitives
    assert "ArgPartition" in primitives and "GatherAxis" in primitives


# --------------------------------------------------------------------------
# item 5 - eager K20 target/draft support
# --------------------------------------------------------------------------


def _stock_ordered_support(rows, top_k):
    idx, values = fast_sampling._deterministic_mlx_top_k_support(rows, top_k)
    return fast_sampling._order_bounded_mlx_top_k_support(idx, values)


@pytest.mark.parametrize("top_k", [1, 3, 20])
def test_ordered_top_k_matches_the_stock_pair_on_random_rows(top_k):
    rows = mx.random.normal((4, 97)).astype(mx.float32)
    mx.eval(rows)
    stock_idx, stock_vals = _stock_ordered_support(rows, top_k)
    diet_idx, diet_vals = fast_sampling._opdiet_ordered_top_k_support(rows, top_k)
    assert stock_idx.dtype == diet_idx.dtype
    mx.eval(stock_idx, diet_idx)
    assert stock_idx.tolist() == diet_idx.tolist()
    assert _identical(stock_vals, diet_vals)


def test_ordered_top_k_matches_the_stock_pair_on_heavy_ties():
    """Rows quantized hard enough that the cutoff is always contested."""

    raw = mx.random.normal((8, 64))
    rows = mx.round(raw * 2.0).astype(mx.float32)
    mx.eval(rows)
    for top_k in (2, 5, 20):
        stock_idx, stock_vals = _stock_ordered_support(rows, top_k)
        diet_idx, diet_vals = fast_sampling._opdiet_ordered_top_k_support(
            rows, top_k
        )
        mx.eval(stock_idx, diet_idx)
        assert stock_idx.tolist() == diet_idx.tolist(), top_k
        assert _identical(stock_vals, diet_vals)


def test_ordered_top_k_owns_cutoff_ties_by_lowest_vocabulary_id():
    """The documented rule, checked directly rather than through the stock pair.

    One value above the cutoff plus a wall of exact ties: the support must be
    that value plus the 19 LOWEST tied ids, ordered value-desc then id-asc.
    (Same fixture as tests/test_fast_sampling.py's tie-break contract.)
    """

    logits = [0.0] * 32
    logits[2] = 2.0
    rows = mx.array([logits], dtype=mx.float32)
    mx.eval(rows)

    idx, values = fast_sampling._opdiet_ordered_top_k_support(rows, 20)
    mx.eval(idx, values)
    assert set(idx.tolist()[0]) == set(range(20))
    assert idx.tolist()[0][0] == 2
    assert idx.tolist()[0][1:] == [i for i in range(20) if i != 2]
    assert values.tolist()[0][0] == 2.0
    assert values.tolist()[0][1:] == [0.0] * 19

    stock_idx, stock_vals = _stock_ordered_support(rows, 20)
    mx.eval(stock_idx)
    assert stock_idx.tolist() == idx.tolist()
    assert _identical(stock_vals, values)


def test_ordered_top_k_matches_when_the_whole_row_is_tied():
    rows = mx.zeros((3, 40), dtype=mx.float32)
    mx.eval(rows)
    idx, values = fast_sampling._opdiet_ordered_top_k_support(rows, 20)
    mx.eval(idx)
    assert idx.tolist() == [list(range(20))] * 3
    stock_idx, stock_vals = _stock_ordered_support(rows, 20)
    mx.eval(stock_idx)
    assert stock_idx.tolist() == idx.tolist()
    assert _identical(stock_vals, values)


def test_ordered_top_k_matches_when_k_is_the_whole_vocabulary():
    rows = mx.random.normal((2, 16)).astype(mx.float32)
    mx.eval(rows)
    stock_idx, stock_vals = _stock_ordered_support(rows, 16)
    diet_idx, diet_vals = fast_sampling._opdiet_ordered_top_k_support(rows, 16)
    mx.eval(stock_idx, diet_idx)
    assert stock_idx.tolist() == diet_idx.tolist()
    assert _identical(stock_vals, diet_vals)


def test_ordered_top_k_preserves_the_prefix_shape():
    rows = mx.random.normal((2, 3, 24)).astype(mx.float32)
    mx.eval(rows)
    idx, values = fast_sampling._opdiet_ordered_top_k_support(rows, 5)
    assert idx.shape == (2, 3, 5)
    assert values.shape == (2, 3, 5)
    stock_idx, stock_vals = _stock_ordered_support(rows, 5)
    mx.eval(stock_idx, idx)
    assert stock_idx.tolist() == idx.tolist()
    assert _identical(stock_vals, values)


def test_ordered_top_k_support_runs_the_stock_pair_when_the_flag_is_off(
    monkeypatch,
):
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    calls: list[str] = []
    stock_deterministic = fast_sampling._deterministic_mlx_top_k_support
    stock_order = fast_sampling._order_bounded_mlx_top_k_support

    def spy_deterministic(rows, top_k):
        calls.append("deterministic")
        return stock_deterministic(rows, top_k)

    def spy_order(idx, values):
        calls.append("order")
        return stock_order(idx, values)

    monkeypatch.setattr(
        fast_sampling, "_deterministic_mlx_top_k_support", spy_deterministic
    )
    monkeypatch.setattr(
        fast_sampling, "_order_bounded_mlx_top_k_support", spy_order
    )
    monkeypatch.setattr(
        fast_sampling,
        "_opdiet_ordered_top_k_support",
        lambda *_a: pytest.fail("op diet ran with the flag off"),
    )

    rows = mx.random.normal((2, 32)).astype(mx.float32)
    mx.eval(rows)
    fast_sampling.ordered_top_k_support(rows, 4)
    assert calls == ["deterministic", "order"]


def test_ordered_top_k_support_runs_the_diet_when_armed(opdiet, monkeypatch):
    monkeypatch.setattr(
        fast_sampling,
        "_deterministic_mlx_top_k_support",
        lambda *_a: pytest.fail("stock pair ran with the flag on"),
    )
    seen: list[int] = []
    stock = fast_sampling._opdiet_ordered_top_k_support

    def spy(rows, top_k):
        seen.append(top_k)
        return stock(rows, top_k)

    monkeypatch.setattr(fast_sampling, "_opdiet_ordered_top_k_support", spy)
    rows = mx.random.normal((2, 32)).astype(mx.float32)
    mx.eval(rows)
    fast_sampling.ordered_top_k_support(rows, 4)
    assert seen == [4]


def test_ordered_top_k_drops_the_full_vocabulary_cumsum_and_widenings():
    """Where the win is: full-VOCABULARY work, not the total primitive count.

    The stock pair runs eight vocabulary-wide passes to resolve the cutoff tie
    (two compares, two bool->int32 widenings, a cumsum, a sum, the chosen-mask
    fold and the -inf select). The diet runs two, and does the rest of the tie
    bookkeeping on 2k-wide candidates.
    """

    rows = mx.zeros((2, 64), dtype=mx.float32)
    mx.eval(rows)
    stock = _kernel_primitives(*mx.compile(_stock_ordered_support)(rows, 8))
    diet = _kernel_primitives(
        *mx.compile(fast_sampling._opdiet_ordered_top_k_support)(rows, 8)
    )

    assert "CumSum" in stock
    assert "CumSum" not in diet
    # The two bool -> int32 widenings of vocabulary-wide masks are gone too.
    assert stock.count("AsType") == 2
    assert diet.count("AsType") == 1
    # Both still pay exactly two vocabulary-wide selections.
    assert stock.count("ArgPartition") == 2
    assert diet.count("ArgPartition") + diet.count("Partition") == 2


# --------------------------------------------------------------------------
# flag-off structural guards for the model-side sites
# --------------------------------------------------------------------------


def test_flag_off_model_sites_keep_the_stock_expressions(monkeypatch):
    """With the flag off the gated sites must build the pre-diet graph."""

    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)

    def guard(*_args, **_kwargs):
        pytest.fail("an op-diet helper ran with the flag off")

    monkeypatch.setattr(qwen4_exp, "_apply_partial_rope_half", guard)
    monkeypatch.setattr(qwen4_exp, "_rope_cos_sin_half", guard)
    monkeypatch.setattr(qwen4_exp, "_shared_rope_cos_sin_half", guard)

    hyper = mx.zeros((1, 2, 24))
    block_out = mx.zeros((1, 2, 6))
    inject = mx.zeros((1, 2, 4))
    mx.eval(hyper, block_out, inject)
    out = qwen4_exp._hyper_residual_write(hyper, block_out, inject)
    assert _identical(out, _stock_residual_write(hyper, block_out, inject))


# --------------------------------------------------------------------------
# items 1 + 2, exercised through the real QSA indexer methods
# --------------------------------------------------------------------------


def _tiny_indexer_args() -> "qwen4_exp.TextArgs":
    return qwen4_exp.TextArgs(
        hidden_size=64,
        head_dim=32,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=32,
        indexer_compress_ratio=4,
        partial_rotary_factor=0.25,
    )


class _FakeFixedQSACache:
    """The three fields ``_extend_pooled_fixed`` reads off a fixed bank."""

    fixed_capacity = True

    def __init__(self, offset, ratio, raw_keys, pooled, rows):
        self.offset = offset
        self.ratio = ratio
        self.raw_keys = raw_keys
        self.pooled = pooled
        self._last_write_rows = rows


def _run_extend_pooled_fixed(indexer, offset, capacity_blocks, head_dim, rows, seed):
    ratio = indexer.ratio
    mx.random.seed(seed)
    raw_keys = mx.random.normal((1, capacity_blocks * ratio, head_dim))
    pooled = mx.random.normal((1, capacity_blocks, head_dim))
    mx.eval(raw_keys, pooled)
    cache = _FakeFixedQSACache(
        mx.array(offset, dtype=mx.int32), ratio, raw_keys, pooled, rows
    )
    total = cache.offset + rows
    out = indexer._extend_pooled_fixed(cache, total)
    mx.eval(out)
    return out


@pytest.mark.parametrize("offset", [0, 4, 8, 44])
def test_extend_pooled_fixed_is_bitwise_identical(monkeypatch, offset):
    """The real method, flag off vs flag on, on the same bank and keys."""

    args = _tiny_indexer_args()
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock_indexer = qwen4_exp.QSAIndexer(args)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    diet_indexer = qwen4_exp.QSAIndexer(args)
    diet_indexer.k_layernorm.weight = stock_indexer.k_layernorm.weight

    capacity_blocks, head_dim, rows = 12, 32, 4
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock = _run_extend_pooled_fixed(
        stock_indexer, offset, capacity_blocks, head_dim, rows, seed=7
    )
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    diet = _run_extend_pooled_fixed(
        diet_indexer, offset, capacity_blocks, head_dim, rows, seed=7
    )
    assert _identical(stock, diet)


def test_extend_pooled_fixed_past_capacity_is_bitwise_identical(monkeypatch):
    """The clamped/no-write branch (block >= capacity) must also match."""

    args = _tiny_indexer_args()
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock_indexer = qwen4_exp.QSAIndexer(args)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    diet_indexer = qwen4_exp.QSAIndexer(args)
    diet_indexer.k_layernorm.weight = stock_indexer.k_layernorm.weight

    capacity_blocks, head_dim, rows = 8, 32, 4
    offset = capacity_blocks * 4  # one block PAST the bank
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock = _run_extend_pooled_fixed(
        stock_indexer, offset, capacity_blocks, head_dim, rows, seed=3
    )
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    diet = _run_extend_pooled_fixed(
        diet_indexer, offset, capacity_blocks, head_dim, rows, seed=3
    )
    assert _identical(stock, diet)


@pytest.mark.parametrize("pos_start", [0, 13])
def test_prepare_queries_eager_is_bitwise_identical(monkeypatch, pos_start):
    args = _tiny_indexer_args()
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock_indexer = qwen4_exp.QSAIndexer(args)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    diet_indexer = qwen4_exp.QSAIndexer(args)
    diet_indexer.q_layernorm.weight = stock_indexer.q_layernorm.weight

    q = mx.random.normal((1, 4, args.indexer_n_heads, args.indexer_head_dim))
    mx.eval(q)

    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock = stock_indexer._prepare_queries_eager(q, pos_start)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    with qwen4_exp._rope_table_scope():
        diet = diet_indexer._prepare_queries_eager(q, pos_start)
    assert _identical(stock, diet)


def _extend_pooled_fixed_primitives(indexer, capacity_blocks=12, head_dim=32):
    """Kernel primitives of one _extend_pooled_fixed call, unevaluated."""

    ratio = indexer.ratio
    raw_keys = mx.zeros((1, capacity_blocks * ratio, head_dim))
    pooled = mx.zeros((1, capacity_blocks, head_dim))
    mx.eval(raw_keys, pooled, indexer.k_layernorm.weight, indexer._inv_freq)
    cache = _FakeFixedQSACache(
        mx.array(8, dtype=mx.int32), ratio, raw_keys, pooled, 4
    )
    return _kernel_primitives(
        indexer._extend_pooled_fixed(cache, cache.offset + 4)
    )


def test_extend_pooled_fixed_uses_the_rowsel_form_when_armed(monkeypatch):
    """Armed, the bank write must read one row and write one row back.

    ``_extend_pooled_fixed`` already dynamic-slices raw_keys, so the tell is
    the SECOND DynamicSlice (the pooled row) next to the slice_update.
    """

    args = _tiny_indexer_args()
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    armed = _extend_pooled_fixed_primitives(qwen4_exp.QSAIndexer(args))
    assert armed.count("DynamicSlice") == 2
    assert armed.count("DynamicSliceUpdate") == 1

    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock = _extend_pooled_fixed_primitives(qwen4_exp.QSAIndexer(args))
    assert stock.count("DynamicSlice") == 1
    assert stock.count("DynamicSliceUpdate") == 1


def test_every_residual_write_site_goes_through_the_shared_helper():
    """No copy of the two-kernel spelling may survive outside the helper."""

    import inspect

    from mtplx import qwen4_m4_stage3

    layer_source = inspect.getsource(qwen4_exp.DecoderLayer.__call__)
    assert layer_source.count("_hyper_residual_write(hyper, block_out, inject)") == 2
    assert "inject[..., :, None]" not in layer_source

    for forward in (
        qwen4_m4_stage3._m4_routed_down_residual_tail_layer_forward,
        qwen4_m4_stage3._m4_paired_routed_glu_residual_tail_layer_forward,
    ):
        source = inspect.getsource(forward)
        assert "_hyper_residual_write(hyper, block_out, inject)" in source
        assert "inject[..., :, None]" not in source


def test_text_trunk_opens_a_rope_scope_only_when_armed():
    import inspect

    source = inspect.getsource(qwen4_exp.Qwen4ExpTextModel.__call__)
    assert 'qwen4_opdiet_enabled("rope")' in source
    assert "_rope_table_scope()" in source
    assert "return self._forward(inputs, cache, input_embeddings)" in source


def test_extend_pooled_fixed_keeps_the_bank_dtype_when_the_norm_widens(
    monkeypatch,
):
    """A wider norm weight must not promote the bank.

    ``mx.slice_update`` casts its update to the destination dtype; the plain
    ``mx.where`` the diet uses would promote both operands instead. Caught by
    fuzzing a bf16 bank behind a float32 norm weight.
    """

    args = _tiny_indexer_args()
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock_indexer = qwen4_exp.QSAIndexer(args)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    diet_indexer = qwen4_exp.QSAIndexer(args)
    diet_indexer.k_layernorm.weight = stock_indexer.k_layernorm.weight
    assert stock_indexer.k_layernorm.weight.dtype == mx.float32

    def run(indexer):
        mx.random.seed(11)
        raw = mx.random.normal((1, 32, 32)).astype(mx.bfloat16)
        pooled = mx.random.normal((1, 8, 32)).astype(mx.bfloat16)
        mx.eval(raw, pooled)
        cache = _FakeFixedQSACache(
            mx.array(8, dtype=mx.int32), 4, raw, pooled, 4
        )
        out = indexer._extend_pooled_fixed(cache, cache.offset + 4)
        mx.eval(out)
        return out

    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    stock = run(stock_indexer)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    diet = run(diet_indexer)

    assert stock.dtype == mx.bfloat16
    assert _identical(stock, diet)


# --------------------------------------------------------------------------
# per-item selection (MTPLX_QWEN4_OPDIET_ITEMS)
# --------------------------------------------------------------------------


def test_opdiet_items_default_to_all_four():
    assert runtime_options.QWEN4_OPDIET_ITEMS == ("bank", "rope", "resid", "k20")
    every = frozenset(runtime_options.QWEN4_OPDIET_ITEMS)
    for raw in (None, "", "   ", "all", ",,"):
        assert runtime_options.parse_opdiet_items(raw) == every, raw


def test_opdiet_items_parse_a_comma_list():
    assert runtime_options.parse_opdiet_items("bank") == {"bank"}
    assert runtime_options.parse_opdiet_items(" ROPE , resid ") == {"rope", "resid"}
    assert runtime_options.parse_opdiet_items("k20,k20") == {"k20"}
    assert runtime_options.parse_opdiet_items("resid,") == {"resid"}


def test_opdiet_items_reject_a_typo_instead_of_dropping_it():
    """A silently dropped item would make an A/B measure the wrong thing."""

    with pytest.raises(ValueError, match="unknown item"):
        runtime_options.parse_opdiet_items("bank,rop")
    with pytest.raises(ValueError, match="unknown item"):
        runtime_options.parse_opdiet_items("nope")


def test_master_switch_gates_every_item(monkeypatch):
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    monkeypatch.setattr(
        runtime_options, "_QWEN4_OPDIET_SELECTED", frozenset({"bank", "rope"})
    )
    for item in runtime_options.QWEN4_OPDIET_ITEMS:
        assert runtime_options.qwen4_opdiet_enabled(item) is False
    assert runtime_options.qwen4_opdiet_enabled() is False


def test_one_item_can_be_armed_alone(monkeypatch):
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET_SELECTED", frozenset({"bank"}))
    assert runtime_options.qwen4_opdiet_enabled() is True
    assert runtime_options.qwen4_opdiet_enabled("bank") is True
    for item in ("rope", "resid", "k20"):
        assert runtime_options.qwen4_opdiet_enabled(item) is False
    with pytest.raises(ValueError):
        runtime_options.qwen4_opdiet_enabled("nope")


def test_deselecting_resid_restores_the_stock_residual_graph(monkeypatch):
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    monkeypatch.setattr(
        runtime_options, "_QWEN4_OPDIET_SELECTED", frozenset({"bank", "rope", "k20"})
    )
    args = (mx.zeros((1, 2, 24)), mx.zeros((1, 2, 6)), mx.zeros((1, 2, 4)))
    mx.eval(*args)
    out = mx.compile(qwen4_exp._hyper_residual_write)(*args)
    assert _kernel_primitives(out) == ["CompiledBroadcastBroadcastMultiply", "Add"]


def test_deselecting_bank_restores_the_stock_bank_write(monkeypatch):
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    monkeypatch.setattr(
        runtime_options, "_QWEN4_OPDIET_SELECTED", frozenset({"rope", "resid", "k20"})
    )
    args = _tiny_indexer_args()
    primitives = _extend_pooled_fixed_primitives(qwen4_exp.QSAIndexer(args))
    # rope stays armed here; only the bank write reverts.
    assert primitives.count("DynamicSlice") == 1
    assert primitives.count("DynamicSliceUpdate") == 1


def test_deselecting_rope_restores_the_stock_tables(monkeypatch):
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    monkeypatch.setattr(
        runtime_options, "_QWEN4_OPDIET_SELECTED", frozenset({"bank", "resid", "k20"})
    )
    monkeypatch.setattr(
        qwen4_exp,
        "_rope_cos_sin_half",
        lambda *_a: pytest.fail("rope rewrite ran while deselected"),
    )
    args = _tiny_indexer_args()
    indexer = qwen4_exp.QSAIndexer(args)
    _run_extend_pooled_fixed(indexer, 8, 12, 32, 4, seed=5)
    indexer._prepare_queries_eager(mx.zeros((1, 4, 2, 32)), 3)


def test_deselecting_k20_restores_the_stock_pair(monkeypatch):
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", True)
    monkeypatch.setattr(
        runtime_options, "_QWEN4_OPDIET_SELECTED", frozenset({"bank", "rope", "resid"})
    )
    monkeypatch.setattr(
        fast_sampling,
        "_opdiet_ordered_top_k_support",
        lambda *_a: pytest.fail("k20 rewrite ran while deselected"),
    )
    rows = mx.random.normal((2, 32)).astype(mx.float32)
    mx.eval(rows)
    fast_sampling.ordered_top_k_support(rows, 4)


def test_every_item_is_reachable_from_a_gated_call_site():
    """No item may be dead: each must name a production gate."""

    import pathlib

    sources = "".join(
        pathlib.Path(name).read_text()
        for name in (
            "mtplx/models/qwen4_exp.py",
            "mtplx/fast_sampling.py",
            "mtplx/generation.py",
        )
    )
    for item in runtime_options.QWEN4_OPDIET_ITEMS:
        assert 'qwen4_opdiet_enabled("' + item + '")' in sources, item
    assert "qwen4_opdiet_enabled()" not in sources


