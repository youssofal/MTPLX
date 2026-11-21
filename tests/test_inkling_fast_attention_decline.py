"""Where the fused attention's qualification declines (a GPU whose bits or pipelines differ, or
``MLX_SDPA_BLOCKS`` past the 256 blocks it serves), decode still serves: the step takes the
chronological one-row reference attention, counted, and no native program is recorded, since
the reference's shapes follow the live history. Review 2, finding 1."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the fused attention's qualification needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling.attention import attend as reference_attend  # noqa: E402
from mtplx.models.inkling.fast import engine_site  # noqa: E402
from mtplx.models.inkling.fast import step as fast_step  # noqa: E402

WINDOW = 1024


def test_an_unserved_reduction_declines_the_fused_attention_and_every_program(monkeypatch):
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "512")
    assert fast_step.attention_reason(mx.bfloat16) == "blocks_512"
    assert engine_site._unfit("a,bc,d,e") == "attention_blocks_512"
    assert engine_site._unfit("a,d,e") == "reference_attention"
    monkeypatch.delenv("MLX_SDPA_BLOCKS")
    assert fast_step.attention_reason(mx.bfloat16) is None      # this Mac qualifies by default
    assert engine_site._unfit("a,bc,d,e") is None


def test_a_failed_check_declines_too(monkeypatch):
    from mtplx.models.inkling.fast import attend as fast_attend

    monkeypatch.setattr(fast_attend, "_same_as_mlx", lambda *args, **kw: False)
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "96")          # a plan no other test qualifies (cache)
    reason = fast_step.attention_reason(mx.bfloat16)
    assert reason is not None and reason.startswith("bits_")
    assert engine_site._unfit("a,bc,d,e") == "attention_" + reason


class _History:
    """``DecodeState.history`` for one visit: the committed rows in chronological order (a
    window keeps its last 1,023) with the panel appended."""

    def __init__(self, keys, values, window):
        self.keys, self.values, self.window = keys, values, window

    def history(self, visit, k_rows, v_rows):
        keys, values = self.keys, self.values
        if self.window:
            keys, values = keys[:, :, -(self.window - 1):], values[:, :, -(self.window - 1):]
        return mx.concatenate([keys, k_rows], axis=2), mx.concatenate([values, v_rows], axis=2)


@pytest.mark.parametrize("window,history,rows", [(WINDOW, 300, 1), (WINDOW, 300, 8), (WINDOW, 1500, 3),
                                                 (WINDOW, 1500, 12), (0, 300, 3), (0, 5000, 8), (0, 5000, 1)])
def test_every_row_is_its_own_one_row_reference_call(window, history, rows):
    keys = iter(mx.random.split(mx.random.key(7 * history + rows + window), 8))
    rnd = lambda shape, scale: (mx.random.normal(shape, key=next(keys)) * scale).astype(mx.bfloat16)
    hist_k, hist_v = rnd((1, 8, history, 128), 0.6), rnd((1, 8, history, 128), 1.0)
    q, k_rows, v_rows = rnd((1, rows, 24, 128), 0.6), rnd((1, 8, rows, 128), 0.6), rnd((1, 8, rows, 128), 1.0)
    profiles, sink = rnd((1, rows, 24, 1024), 0.5), rnd((1, rows, 24), 1.0)
    layer = SimpleNamespace(sliding=bool(window), cfg={"sliding_window_size": WINDOW})
    panel = SimpleNamespace(rows=rows)
    got = fast_step.chronological_core(layer, q, profiles, sink, k_rows, v_rows,
                                       _History(hist_k, hist_v, window), 0, panel)
    assert got.shape == (1, rows, 24 * 128)
    for row in range(rows):
        # The one-row step at this row's position: the earlier panel rows committed, the
        # window's last 1,023 rows kept, the row itself last, formed independently.
        k = mx.concatenate([hist_k, k_rows[:, :, : row + 1]], axis=2)
        v = mx.concatenate([hist_v, v_rows[:, :, : row + 1]], axis=2)
        if window:
            k, v = k[:, :, -WINDOW:], v[:, :, -WINDOW:]
        want = reference_attend(q[:, row:row + 1].transpose(0, 2, 1, 3), k, v, profiles[:, row:row + 1],
                                sink[:, row:row + 1], window)
        want = want.transpose(0, 2, 1, 3).reshape(1, 1, -1)
        assert bool(mx.array_equal(got[:, row:row + 1].view(mx.uint16), want.view(mx.uint16)).item()), row
