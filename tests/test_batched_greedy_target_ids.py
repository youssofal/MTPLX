from __future__ import annotations

import mlx.core as mx
import numpy as np

from mtplx.generation import _eval_verify_outputs


def test_batched_greedy_ids_match_per_row_argmax() -> None:
    rng = np.random.default_rng(7)
    logits_np = rng.normal(size=(1, 5, 32)).astype(np.float32)
    logits = mx.array(logits_np)

    batched = np.asarray(mx.argmax(logits, axis=-1)).reshape(-1).tolist()
    per_row = [
        int(np.asarray(mx.argmax(logits[:, row, :][0], axis=-1)))
        for row in range(logits_np.shape[1])
    ]

    assert batched == per_row


def test_lazy_verify_eval_materializes_batched_greedy_ids(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_LAZY_VERIFY_LOGITS", "1")
    logits = mx.array(
        [[[0.0, 3.0, 1.0], [4.0, 2.0, 0.0], [1.0, 2.0, 5.0]]]
    )
    hidden = mx.array([[[1.0], [2.0], [3.0]]])
    greedy_ids = mx.argmax(logits, axis=-1)

    timings = _eval_verify_outputs(
        logits,
        hidden,
        greedy_target_ids=greedy_ids,
    )

    assert np.asarray(greedy_ids).reshape(-1).tolist() == [1, 0, 2]
    assert timings["verify_hidden_eval_time_s"] >= 0.0
