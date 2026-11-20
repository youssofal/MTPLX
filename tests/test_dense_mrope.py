"""Dense Qwen3.5 / Qwen3.8 path: image tokens rope at their grid positions.

Everything here runs on a four-layer synthetic ``qwen3_5`` text model with
random weights (two GatedDeltaNet layers, two full-attention layers, one MTP
layer) and hand-made tensors. No pack is loaded.

Pinned:

* a text request is bit-identical before and after the adapter is installed,
  and so are the rows of an image request that come before its first image;
* the adapter equals an interleaved M-RoPE written here in numpy from the
  transformers definition (frequency i follows h when i % 3 == 1 below
  3 * section[1], w when i % 3 == 2 below 3 * section[2], else t);
* rows after the prompt rope at ``cache offset + delta`` in decode, in a
  verify window and in the draft head;
* ``MTPLX_DENSE_MROPE=0`` leaves the model and the bank keys as they were;
* every fallback is counted, never silent, and never half applied.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mtplx import demotions
from mtplx.dense_mrope import (
    INSTALL_ATTR,
    ROLE_MTP,
    ROLE_TRUNK,
    DenseMRopeAdapter,
    DenseMRopeState,
    build_request_state,
    configure_dense_mrope,
    dense_mrope_scope,
    dense_mrope_state,
    mrope_axes,
)
from mtplx.vision.mrope import build_mrope_positions
from mtplx.vision.splice import VisionSplice, vision_bank_key_ids
from tests import dense_mrope_synth as synth
from tests.dense_mrope_synth import (
    DELTA,
    GRID,
    PAD,
    PROMPT,
    ROPE_THETA,
    ROTARY_DIMS,
    SECTION,
    VIDEO_PAD,
)

_pack_config = synth.pack_config
_text_model = synth.text_model
_model_with_draft_head = synth.model_with_draft_head
_state = synth.position_state
_full_attention = synth.full_attention
_spy_on = synth.spy_on


@pytest.fixture(autouse=True)
def _clean_ledger(monkeypatch):
    monkeypatch.delenv("MTPLX_DENSE_MROPE", raising=False)
    monkeypatch.delenv("MTPLX_DENSE_MROPE_STRICT", raising=False)
    demotions.reset()
    yield
    demotions.reset()


# -- the reference, written here from the transformers definition ---------------


def _reference_interleaved_mrope(x, positions3, section, base, rotary_dims):
    """apply_interleaved_mrope + rotate_half, float64, [.., L, D] inputs."""
    half = rotary_dims // 2
    inv_freq = base ** (-(np.arange(0, rotary_dims, 2, dtype=np.float64) / rotary_dims))
    freqs = positions3[:, :, None].astype(np.float64) * inv_freq[None, None, :]
    merged = freqs[0].copy()
    for axis, first in ((1, 1), (2, 2)):
        index = slice(first, section[axis] * 3, 3)
        merged[..., index] = freqs[axis][..., index]
    emb = np.concatenate([merged, merged], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)
    x = x.astype(np.float64)
    rot, rest = x[..., :rotary_dims], x[..., rotary_dims:]
    rotated = np.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
    return np.concatenate([rot * cos + rotated * sin, rest], axis=-1)


def test_axis_layout_is_the_reference_interleave():
    assert mrope_axes([11, 11, 10], True, 32) == [
        1 if i % 3 == 1 else 2 if (i % 3 == 2 and i < 30) else 0 for i in range(32)
    ]
    assert mrope_axes(SECTION, True, 8) == [0, 1, 2, 0, 1, 2, 0, 1]
    assert mrope_axes([2, 2, 1], False, 5) == [0, 0, 1, 1, 2]
    with pytest.raises(ValueError):
        mrope_axes([11, 11, 10], True, 16)  # does not tile 16 frequencies


def test_adapter_matches_the_numpy_reference_on_an_image_grid():
    rope = nn.RoPE(ROTARY_DIMS, traditional=False, base=ROPE_THETA)
    adapter = DenseMRopeAdapter(rope, mrope_axes(SECTION, True, 8), ROLE_TRUNK)
    state = _state()
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 2, len(PROMPT), 64)).astype(np.float32)

    with dense_mrope_scope(state):
        whole = adapter(mx.array(x), offset=0)
        # The same rows in three chunks that cut through the image, each one
        # positioned by its cache offset alone.
        chunks = mx.concatenate(
            [
                adapter(mx.array(x[:, :, :5]), offset=0),
                adapter(mx.array(x[:, :, 5:7]), offset=5),
                adapter(mx.array(x[:, :, 7:]), offset=7),
            ],
            axis=2,
        )
    expected = _reference_interleaved_mrope(
        x, state.table, SECTION, ROPE_THETA, ROTARY_DIMS
    )
    np.testing.assert_allclose(np.asarray(whole), expected, atol=2e-5, rtol=0)
    assert mx.array_equal(whole, chunks).item()

    # The test can tell layouts apart: sequential positions, and the chunked
    # (non-interleaved) split of the same sections, are both far away.
    sequential = np.asarray(rope(mx.array(x), offset=0))
    assert np.abs(sequential - expected).max() > 1e-2
    chunked_axes = DenseMRopeAdapter(rope, mrope_axes(SECTION, False, 8), ROLE_TRUNK)
    with dense_mrope_scope(state):
        wrong_layout = np.asarray(chunked_axes(mx.array(x), offset=0))
    assert np.abs(wrong_layout - expected).max() > 1e-2


def test_per_axis_path_with_equal_axes_is_the_stock_rope_bit_for_bit():
    """Why text rows inside an image chunk keep the keys a text request has."""
    rope = nn.RoPE(ROTARY_DIMS, traditional=False, base=ROPE_THETA)
    adapter = DenseMRopeAdapter(rope, mrope_axes(SECTION, True, 8), ROLE_TRUNK)
    run = mx.arange(37, 37 + 9, dtype=mx.int32)
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        for width in (ROTARY_DIMS, 64):  # all rotary, and a partial-rotary head
            x = mx.random.normal((1, 3, 9, width)).astype(dtype)
            per_axis = adapter._rope_axes(x, (run, run, run))
            assert per_axis.dtype == dtype and per_axis.shape == x.shape
            assert mx.array_equal(per_axis, rope(x, offset=37)).item()


def test_positions_past_the_prompt_are_index_plus_delta():
    state = _state()
    assert state.delta == DELTA
    # An independent statement of the rule: the table of the prompt with more
    # text appended is the prompt's table followed by index + delta.
    longer = PROMPT + [12, 13, 14, 15, 16]
    table, delta = build_mrope_positions(
        longer, image_token_id=PAD, image_grids=[GRID], spatial_merge_size=2
    )
    assert delta == DELTA
    np.testing.assert_array_equal(state.positions(0, len(longer)), table)
    np.testing.assert_array_equal(state.positions(11, 7), table[:, 11:18])
    np.testing.assert_array_equal(state.positions(15, 3), table[:, 15:18])
    # Plans: text before the image and rows past the prompt are one stock
    # call; rows that touch the image carry per-axis positions.
    assert state.plan(0, 3) == ("shift", 0)
    assert state.plan(9, 4) == ("shift", 9 + DELTA)
    assert state.plan(len(PROMPT), 1) == ("shift", len(PROMPT) + DELTA)
    assert state.plan(len(PROMPT) + 40, 4) == ("shift", len(PROMPT) + 40 + DELTA)
    kind, axes = state.plan(2, 5)
    assert kind == "axes"
    np.testing.assert_array_equal(np.asarray(axes[1]), state.table[1, 2:7])


# The MoE sibling (the 35B-A3B): the same model with routed and shared experts
# in every layer. Its attention, and so the adapter, is the dense model's.
FAMILIES = [
    pytest.param({}, "qwen3_5", id="dense"),
    pytest.param(synth.MOE_FIELDS, "qwen3_5_moe", id="moe"),
]


@pytest.mark.parametrize("fields, model_type", FAMILIES)
def test_text_request_is_bit_identical_after_install(fields, model_type):
    model = _text_model(**fields)
    ids = mx.array([[(3 * i + 1) % 97 for i in range(24)]])

    def run():
        outs = [model(ids)]
        cache = model.make_cache()
        outs.append(model(ids[:, :10], cache=cache))
        outs.append(model(ids[:, 10:20], cache=cache))
        for step in range(20, 24):
            outs.append(model(ids[:, step : step + 1], cache=cache))
        mx.eval(outs)
        return outs

    before = run()
    install = configure_dense_mrope(model, _pack_config(model_type=model_type))
    assert install is not None and install.installed
    assert install.trunk_layers == 2 and install.mtp_layers == 0
    assert all(isinstance(a.rope, DenseMRopeAdapter) for a in _full_attention(model))
    # The adapter is not a module: the parameter tree is what it was.
    assert not any("rope" in path for path, _ in tree_flatten(model.parameters()))
    after = run()
    assert all(mx.array_equal(a, b).item() for a, b in zip(before, after))

    # Armed with an all-text table the rows are still the stock calls.
    n = int(ids.shape[1])
    text_state = DenseMRopeState(
        np.broadcast_to(np.arange(n, dtype=np.int32), (3, n)), 0
    )
    with dense_mrope_scope(text_state):
        armed = run()
    assert all(mx.array_equal(a, b).item() for a, b in zip(before, armed))
    assert dense_mrope_state() is None


@pytest.mark.parametrize("fields, model_type", FAMILIES)
def test_image_request_changes_only_rows_from_the_first_image_on(fields, model_type):
    model = _text_model(**fields)
    assert configure_dense_mrope(model, _pack_config(model_type=model_type)).installed
    ids = mx.array([PROMPT])

    def chunked_prefill():
        # Two chunks that cut through the image, positioned by the cache.
        cache = model.make_cache()
        return mx.concatenate(
            [model(ids[:, :5], cache=cache), model(ids[:, 5:], cache=cache)], axis=1
        )

    sequential, sequential_chunked = model(ids), chunked_prefill()
    with dense_mrope_scope(_state()):
        grid, grid_chunked = model(ids), chunked_prefill()
    mx.eval(sequential, sequential_chunked, grid, grid_chunked)
    first_image = PROMPT.index(PAD)
    # Rows before the image never see it (causal) and rope at equal axes.
    assert mx.array_equal(grid[:, :first_image], sequential[:, :first_image]).item()
    # The first image row sits at (3, 3, 3) under both schemes: rope is the
    # same there too. Every later row moved, by far more than float noise.
    moved = np.abs(np.asarray(grid - sequential))[0, first_image + 1 :].max(axis=-1)
    assert moved.min() > 1e-2
    if fields:
        # On random weights the MoE's router turns the chunked forward's
        # float noise into other experts, row by row, so the comparison below
        # measures routing, not positions; the positions are the adapter's,
        # which the dense model pins.
        return

    # Chunked prefill agrees with the single forward as closely as it does on
    # a text request (the GatedDeltaNet layers of this random model differ by
    # a few 1e-3 between the two shapes on their own); a row roped at the
    # wrong position would show up at the size of ``moved``.
    def gap(a, b):
        return float(np.abs(np.asarray(a) - np.asarray(b)).max())

    assert gap(grid_chunked, grid) <= 2 * gap(sequential_chunked, sequential) + 1e-4
    assert gap(grid_chunked, grid) < 0.2 * moved.min()


def test_decode_verify_and_draft_head_rope_at_offset_plus_delta(tmp_path):
    model = _model_with_draft_head(tmp_path)
    install = configure_dense_mrope(model, _pack_config())
    assert install.installed and install.mtp_layers == 1
    trunk_spies = [_spy_on(attn) for attn in _full_attention(model)]
    mtp_attn = model.mtp.layers[0].self_attn
    assert mtp_attn.rope.role == ROLE_MTP
    mtp_spy = _spy_on(mtp_attn)

    n = len(PROMPT)
    state = _state()
    cache = model.make_cache()
    mtp_cache = model.make_mtp_cache()
    with dense_mrope_scope(state):
        _logits, hidden = model(mx.array([PROMPT]), cache=cache, return_hidden=True)
        # Committed draft history: row j pairs hidden j with token j + 1.
        model.mtp_update_cache(
            hidden[:, : n - 1], mx.array([PROMPT[1:]]), mtp_cache=mtp_cache
        )
        for spy in (*trunk_spies, mtp_spy):
            spy.calls.clear()

        model(mx.array([[20]]), cache=cache)  # decode, one row at index n
        model(mx.array([[21, 22, 23]]), cache=cache)  # a verify window
        # Draft rows: history row n - 1 is still inside the table, n is past it.
        model.mtp_forward(hidden[:, -1:], mx.array([[20]]), mtp_cache=mtp_cache)
        model.mtp_forward(hidden[:, -1:], mx.array([[21]]), mtp_cache=mtp_cache)

    for spy in trunk_spies:
        assert spy.calls == [
            (1, n + DELTA),
            (1, n + DELTA),
            (3, n + 1 + DELTA),
            (3, n + 1 + DELTA),
        ]
    assert mtp_spy.calls == [
        (1, n - 1 + DELTA),
        (1, n - 1 + DELTA),
        (1, n + DELTA),
        (1, n + DELTA),
    ]

    # A draft history that is not one row per prompt token (cycle caches, a
    # reset or windowed history) keeps the stock rope at its own offset.
    state.mtp_aligned = False
    mtp_spy.calls.clear()
    with dense_mrope_scope(state):
        model.mtp_forward(hidden[:, -1:], mx.array([[22]]), mtp_cache=mtp_cache)
    assert mtp_spy.calls == [(1, n + 1), (1, n + 1)]


def test_explicit_draft_position_offsets_go_through_the_adapter(tmp_path):
    model = _model_with_draft_head(tmp_path)
    assert configure_dense_mrope(model, _pack_config()).installed
    spy = _spy_on(model.mtp.layers[0].self_attn)
    hidden = mx.zeros((1, 1, 64))
    with dense_mrope_scope(_state()):
        model.mtp_forward(
            hidden, mx.array([[20]]), mtp_cache=model.make_mtp_cache(),
            position_offset=len(PROMPT) + 4,
        )
    assert spy.calls == [(1, len(PROMPT) + 4 + DELTA)] * 2


# -- kill switch ----------------------------------------------------------------


def test_kill_switch_leaves_the_model_and_the_request_alone(monkeypatch):
    monkeypatch.setenv("MTPLX_DENSE_MROPE", "0")
    model = _text_model()
    assert configure_dense_mrope(model, _pack_config()) is None
    assert getattr(model, INSTALL_ATTR, None) is None
    assert all(type(a.rope) is nn.RoPE for a in _full_attention(model))
    assert (
        build_request_state(
            model, PROMPT, image_token_id=PAD, image_grids=[GRID], spatial_merge_size=2
        )
        is None
    )
    assert demotions.snapshot()["total"] == 0

    # Flipped after load: the adapters stay but no request is armed.
    monkeypatch.delenv("MTPLX_DENSE_MROPE")
    assert configure_dense_mrope(model, _pack_config()).installed
    monkeypatch.setenv("MTPLX_DENSE_MROPE", "0")
    assert (
        build_request_state(
            model, PROMPT, image_token_id=PAD, image_grids=[GRID], spatial_merge_size=2
        )
        is None
    )
    assert demotions.snapshot()["total"] == 0


def test_other_families_and_text_only_packs_get_no_adapter():
    for config in (
        _pack_config(model_type="qwen4_exp"),
        {k: v for k, v in _pack_config().items() if k != "vision_config"},
    ):
        model = _text_model()
        assert configure_dense_mrope(model, config) is None
        assert all(type(a.rope) is nn.RoPE for a in _full_attention(model))
        assert (
            build_request_state(
                model,
                PROMPT,
                image_token_id=PAD,
                image_grids=[GRID],
                spatial_merge_size=2,
            )
            is None
        )
    assert demotions.snapshot()["total"] == 0


# -- fail closed ----------------------------------------------------------------


def test_request_state_is_built_for_an_installed_model():
    model = _text_model()
    assert configure_dense_mrope(model, _pack_config()).installed
    state = build_request_state(
        model,
        PROMPT,
        image_token_id=PAD,
        image_grids=[GRID],
        spatial_merge_size=2,
        video_token_id=VIDEO_PAD,
    )
    assert isinstance(state, DenseMRopeState)
    assert state.delta == DELTA and state.length == len(PROMPT)
    assert state.mtp_aligned is True
    assert demotions.snapshot()["total"] == 0


def test_unbuildable_table_falls_back_whole_and_is_counted():
    model = _text_model()
    assert configure_dense_mrope(model, _pack_config()).installed
    for prompt, grids in (
        (PROMPT + [VIDEO_PAD], [GRID]),  # a video pad
        (PROMPT, [(1, 4, 8)]),  # the pad run does not match the grid
        (PROMPT + [PAD], [GRID]),  # more pads than images
    ):
        assert (
            build_request_state(
                model,
                prompt,
                image_token_id=PAD,
                image_grids=grids,
                spatial_merge_size=2,
                video_token_id=VIDEO_PAD,
            )
            is None
        )
    snap = demotions.snapshot()
    assert snap["counts"]["vision_mrope_sequential_fallback"] == 3
    assert "could not be built" in snap["reasons"]["vision_mrope_sequential_fallback"]


def test_unrecognised_attention_is_refused_whole_and_counted():
    model = _text_model()
    # One layer with a rope the adapter does not model: nothing is installed,
    # not even on the layers that would have qualified.
    _full_attention(model)[1].rope = nn.RoPE(ROTARY_DIMS, traditional=True, base=ROPE_THETA)
    install = configure_dense_mrope(model, _pack_config())
    assert install is not None and not install.installed
    assert all(type(a.rope) is nn.RoPE for a in _full_attention(model))
    assert (
        build_request_state(
            model, PROMPT, image_token_id=PAD, image_grids=[GRID], spatial_merge_size=2
        )
        is None
    )
    snap = demotions.snapshot()
    assert snap["counts"]["vision_mrope_sequential_fallback"] == 1
    assert "plain half-split rope" in snap["reasons"]["vision_mrope_sequential_fallback"]

    # A section that does not tile the rotary dims is refused the same way.
    other = _text_model()
    config = _pack_config()
    config["text_config"]["rope_parameters"]["mrope_section"] = [11, 11, 10]
    assert not configure_dense_mrope(other, config).installed
    assert all(type(a.rope) is nn.RoPE for a in _full_attention(other))


def test_a_model_that_cannot_be_inspected_is_a_record_not_a_failed_load():
    class Broken:
        @property
        def language_model(self):
            raise RuntimeError("boom")

    model = Broken()
    install = configure_dense_mrope(model, _pack_config())
    assert install is not None and not install.installed
    assert "could not be inspected" in install.reason and "boom" in install.reason
    assert (
        build_request_state(
            model, PROMPT, image_token_id=PAD, image_grids=[GRID], spatial_merge_size=2
        )
        is None
    )
    assert demotions.snapshot()["counts"]["vision_mrope_sequential_fallback"] == 1


def test_tensor_offset_is_the_callers_rotary_origin():
    """A tensor offset comes from a cache that owns its rotary origin (the
    compiled routes hand ``offset + delta``): the stock kernel at it, exactly
    the int-offset call of the eager route past the prompt, nothing counted."""
    from mtplx.dense_mrope import host_positions_for_tensor_offsets

    rope = nn.RoPE(ROTARY_DIMS, traditional=False, base=ROPE_THETA)
    adapter = DenseMRopeAdapter(rope, mrope_axes(SECTION, True, 8), ROLE_TRUNK)
    x = mx.random.normal((1, 2, 3, 64))
    n = len(PROMPT)
    origin = mx.array(n + 4, dtype=mx.int32) + mx.array([DELTA], dtype=mx.int32)
    with dense_mrope_scope(_state()):
        armed = adapter(x, offset=origin)
        eager = adapter(x, offset=n + 4)  # the eager route's shift plan
    assert mx.array_equal(armed, eager).item()
    assert mx.array_equal(armed, rope(x, offset=n + 4 + DELTA)).item()
    # Unarmed, a tensor offset is simply the stock call (compiled text routes).
    assert mx.array_equal(adapter(x, offset=origin), rope(x, offset=origin)).item()
    assert demotions.snapshot()["total"] == 0

    # The instrument's reference leg: a CONCRETE plain offset resolved through
    # the host table, as the eager route resolves a host integer.
    with dense_mrope_scope(_state()), host_positions_for_tensor_offsets():
        reference = adapter(x, offset=mx.array(n + 4, dtype=mx.int32))
        inside = adapter(x, offset=mx.array(2, dtype=mx.int32))
    assert mx.array_equal(reference, eager).item()
    with dense_mrope_scope(_state()):
        assert mx.array_equal(inside, adapter(x, offset=2)).item()
    assert demotions.snapshot()["total"] == 0


def test_a_delta_less_tensor_offset_cache_under_an_armed_request_is_counted(monkeypatch):
    """The routing-hole canary lives where the cache is known: rope_offset_of."""
    from mtplx.graphbank import TensorOffsetKVCache
    from mtplx.rope_origin import (
        cache_owns_rotary_origin,
        rope_offset_of,
        stamp_rope_delta,
    )

    keys = mx.zeros((1, 1, 8, 64))
    cache = TensorOffsetKVCache(keys, keys, 5)
    assert not cache_owns_rotary_origin(cache)
    # Text (unarmed): the plain offset, nothing counted.
    assert rope_offset_of(cache) is cache.offset
    assert demotions.snapshot()["total"] == 0
    # Armed without a delta: a compiled route the admission never handed the
    # delta. The stock positions, counted once per call.
    with dense_mrope_scope(_state()):
        assert rope_offset_of(cache) is cache.offset
    snap = demotions.snapshot()
    assert snap["counts"]["vision_mrope_tensor_offset_call"] == 1
    assert "owns no rotary origin" in snap["reasons"]["vision_mrope_tensor_offset_call"]
    # Stamped: the origin is offset + delta and nothing more is counted.
    assert stamp_rope_delta([cache, None], DELTA) == 1
    assert cache_owns_rotary_origin(cache)
    with dense_mrope_scope(_state()):
        origin = rope_offset_of(cache)
    assert int(origin.item()) == 5 + DELTA
    assert demotions.snapshot()["counts"]["vision_mrope_tensor_offset_call"] == 1
    # The delta is one int32 value, whatever it was given as.
    assert cache.rope_delta.dtype == mx.int32 and cache.rope_delta.shape == (1,)
    assert cache.rope_state == [cache.rope_delta]
    cache.rope_delta = None
    assert cache.rope_state == [] and rope_offset_of(cache) is cache.offset

    monkeypatch.setenv("MTPLX_DENSE_MROPE_STRICT", "1")
    with dense_mrope_scope(_state()), pytest.raises(RuntimeError, match="owns no rotary origin"):
        rope_offset_of(cache)


def test_prompt_whose_image_tokens_moved_is_refused():
    state = _state()
    state.check_prompt(PROMPT, PAD)
    state.check_prompt(PROMPT + [12, 13], PAD)  # text appended after the image
    with pytest.raises(ValueError, match="does not match the prompt"):
        state.check_prompt([4] + PROMPT, PAD)  # a token slipped in before it
    with pytest.raises(ValueError, match="does not match the prompt"):
        state.check_prompt(PROMPT + [PAD], PAD)
    with pytest.raises(ValueError, match="does not match the prompt"):
        state.check_prompt(PROMPT[:-2], PAD)


# -- session bank keys ----------------------------------------------------------


def _splice(*, dense: DenseMRopeState | None, grids=(GRID,), digest: int = 0x1234ABCD):
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.zeros((6, 4)),
        image_digests=(digest,),
        pad_counts=(6,),
        image_grids=tuple(grids),
        dense_mrope=dense,
    )


def test_bank_keys_separate_position_schemes_and_grids():
    import hashlib

    from mtplx.dense_mrope import SCHEME
    from mtplx.vision.splice import (
        _BANK_KEY_FLAG,
        _BANK_KEY_MASK,
        _BANK_KEY_MIX,
        _DENSE_MROPE_KEY_SALT,
    )

    # The salt is the scheme tag, hashed: a new scheme is a new tag and a new
    # salt, and keys written by this one stay readable across processes.
    digest = hashlib.blake2b(SCHEME.encode(), digest_size=8).digest()
    assert _DENSE_MROPE_KEY_SALT == int.from_bytes(digest, "big") & _BANK_KEY_MASK

    sequential = vision_bank_key_ids(PROMPT, _splice(dense=None))
    # Sequential keys are the ones this path always produced.
    legacy = [
        _BANK_KEY_FLAG | ((0x1234ABCD ^ (row * _BANK_KEY_MIX)) & _BANK_KEY_MASK)
        for row in range(6)
    ]
    assert sequential[3:9] == legacy

    grid_roped = vision_bank_key_ids(PROMPT, _splice(dense=_state()))
    pads = [i for i, token in enumerate(PROMPT) if token == PAD]
    # Text ids are shared, so a prefix match can still reach the image...
    assert all(grid_roped[i] == sequential[i] for i in range(len(PROMPT)) if i not in pads)
    # ...but never go past its first row across schemes.
    assert all(grid_roped[i] != sequential[i] for i in pads)
    assert len({grid_roped[i] for i in pads}) == 6

    # Same pixels, same token count, another grid: another position table,
    # so other keys. Same everything: same keys.
    other_grid = vision_bank_key_ids(
        PROMPT, _splice(dense=_state(grids=((1, 6, 4),)), grids=((1, 6, 4),))
    )
    assert all(other_grid[i] != grid_roped[i] for i in pads)
    assert vision_bank_key_ids(PROMPT, _splice(dense=_state())) == grid_roped


# -- the real layout, through runtime.load, on the Bonsai loader -----------------


def test_runtime_load_installs_on_the_bonsai_loader_with_the_pack_layout(tmp_path):
    """Synthetic Prism pack: [11, 11, 10] over 64 rotary dims, float16."""
    from mtplx import runtime
    from tests import prism_hadamard_synth

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)  # the synthetic pack's tests run on the CPU
    try:
        pack = prism_hadamard_synth.build_synthetic_pack(tmp_path / "pack")
        prism_hadamard_synth.write_synthetic_mtp_sidecar(pack)
        rt = runtime.load(pack.path, mtp=True)
        install = getattr(rt.model, INSTALL_ATTR)
        assert install.installed and install.section == (11, 11, 10)
        assert install.interleaved is True and install.mtp_layers == 1
        trunk = [
            layer.self_attn
            for layer in rt.model.language_model.model.layers
            if not layer.is_linear
        ]
        assert install.trunk_layers == len(trunk) > 0
        adapter = trunk[0].rope
        assert isinstance(adapter, DenseMRopeAdapter)
        assert adapter.axes == tuple(mrope_axes([11, 11, 10], True, 32))

        # A 4 x 6 image (24 tokens) between text runs, at the pack's layout.
        pad = prism_hadamard_synth.IMAGE_TOKEN_ID
        prompt = [1, 2, 3, 4, 5] + [pad] * 24 + [6, 7, 8]
        table, delta = build_mrope_positions(
            prompt, image_token_id=pad, image_grids=[(1, 8, 12)], spatial_merge_size=2
        )
        assert delta == 6 - 24
        state = DenseMRopeState(table, delta)
        rng = np.random.default_rng(5)
        x = rng.standard_normal((1, 3, len(prompt), 256)).astype(np.float16)
        with dense_mrope_scope(state):
            roped = adapter(mx.array(x), offset=0)
            tail = adapter(mx.array(x[:, :, :2]), offset=len(prompt) + 100)
        assert roped.dtype == mx.float16
        expected = _reference_interleaved_mrope(
            x, table, [11, 11, 10], float(adapter.inner.base), 64
        )
        np.testing.assert_allclose(
            np.asarray(roped.astype(mx.float32)), expected, atol=4e-3, rtol=0
        )
        stock_tail = adapter.inner(mx.array(x[:, :, :2]), offset=len(prompt) + 100 + delta)
        assert mx.array_equal(tail, stock_tail).item()
    finally:
        mx.set_default_device(previous)
