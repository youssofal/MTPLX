from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mtplx.qwen38_challenge_kernels import (
    configure_qwen38_row21_qk_rms_rope,
    qwen38_dual_rms_norm_concat,
    qwen38_qk_rms_rope,
)


def test_retained_candidate_hot_paths_have_no_dispatch_counters() -> None:
    root = Path(__file__).resolve().parents[1] / "mtplx"
    sources = {
        name: (root / name).read_text(encoding="utf-8")
        for name in (
            "qwen38_challenge_kernels.py",
            "mtp_patch.py",
            "qwen38_mtp_block_artifacts.py",
            "gdn_capture.py",
            "draft_lm_head.py",
        )
    }
    forbidden = (
        "QWEN38_KV_ONLY_HISTORY_COUNTERS",
        "QWEN38_GDN_DECAY_MEMO_COUNTERS",
        "QWEN38_ROW48_BOUNDARY_COUNTERS",
        "qwen38_dual_norm_calls",
        "qwen38_q8_embed_dual_norm_calls",
        "qwen38_qk_rms_rope_calls",
        "qwen38_row36_island_calls",
        "qwen38_row10_compact_head_calls",
        "qwen38_row10_compact_counter_snapshot",
    )
    for token in forbidden:
        assert all(token not in source for source in sources.values()), token


def test_row21_is_bound_per_instance_without_hot_path_fallbacks() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "mtplx"
        / "qwen38_challenge_kernels.py"
    ).read_text(encoding="utf-8")

    assert "Attention.__call__ =" not in source
    assert "_mtplx_qwen38_row21_active" not in source
    assert "return _QWEN38_ATTENTION_ORIGINAL_CALL" not in source


def test_row21_binding_validates_and_allocates_classes_once(
    monkeypatch,
) -> None:
    class Attention:
        def __call__(self, *_args, **_kwargs):
            return "stock"

    def attention(*, heads: int) -> Attention:
        instance = Attention()
        instance.num_attention_heads = heads
        instance.num_key_value_heads = 4
        instance.head_dim = 256
        instance.rope = SimpleNamespace(
            dims=64,
            base=10_000_000.0,
            scale=1.0,
            traditional=False,
        )
        instance.q_norm = SimpleNamespace(
            eps=1e-6,
            weight=mx.zeros((256,), dtype=mx.bfloat16),
        )
        instance.k_norm = SimpleNamespace(
            eps=1e-6,
            weight=mx.zeros((256,), dtype=mx.bfloat16),
        )
        return instance

    eligible = attention(heads=24)
    ineligible = attention(heads=8)
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=eligible),
                SimpleNamespace(self_attn=ineligible),
            ]
        )
    )
    import mtplx.qwen38_challenge_kernels as kernels

    eligibility_calls = 0
    original_eligible = kernels._row21_attention_eligible

    def counted_eligible(candidate):
        nonlocal eligibility_calls
        eligibility_calls += 1
        return original_eligible(candidate)

    monkeypatch.setattr(kernels, "_row21_attention_eligible", counted_eligible)

    report = configure_qwen38_row21_qk_rms_rope(model, active=True)
    fused_class = eligible.__class__

    assert report["active_modules"] == 1
    assert fused_class is not Attention
    assert ineligible.__class__ is Attention
    assert Attention().__class__ is Attention
    assert eligibility_calls == 2

    configure_qwen38_row21_qk_rms_rope(model, active=False)
    assert eligible.__class__ is Attention
    configure_qwen38_row21_qk_rms_rope(model, active=True)
    assert eligible.__class__ is fused_class
    configure_qwen38_row21_qk_rms_rope(model, active=False)
    configure_qwen38_row21_qk_rms_rope(model, active=True)
    assert eligible.__class__ is fused_class
    assert eligibility_calls == 2


def test_retained_candidate_hot_paths_have_no_invariant_flag_dispatch() -> None:
    root = Path(__file__).resolve().parents[1] / "mtplx"
    sources = {
        name: (root / name).read_text(encoding="utf-8")
        for name in ("gdn_capture.py", "mtp_patch.py")
    }
    forbidden = {
        "gdn_capture.py": (
            "_qwen38_compute_g",
            "_mtplx_qwen38_row48_boundary_fused",
        ),
        "mtp_patch.py": (
            "_mtplx_qwen38_row24_eval_ladder",
            "_mtplx_qwen38_row24_prefill_stride",
            "_mtplx_qwen38_dual_norm_concat",
            "_mtplx_qwen38_row63_q8_embedding_dual_norm",
        ),
    }
    for name, tokens in forbidden.items():
        for token in tokens:
            assert token not in sources[name], f"{name}: {token}"


def test_dual_rms_norm_concat_matches_two_stock_norms() -> None:
    a = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    b = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    a_weight = mx.random.normal((5120,)).astype(mx.bfloat16)
    b_weight = mx.random.normal((5120,)).astype(mx.bfloat16)
    expected = mx.concatenate(
        (
            mx.fast.rms_norm(a, a_weight, 1e-6),
            mx.fast.rms_norm(b, b_weight, 1e-6),
        ),
        axis=-1,
    )
    actual = qwen38_dual_rms_norm_concat(a, b, a_weight, b_weight, 1e-6)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()


def test_qk_rms_rope_matches_stock_qwen38_partial_rope_at_fixed_d3_width() -> None:
    queries = mx.random.normal((1, 4, 24, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 4, 4, 256)).astype(mx.bfloat16)
    q_weight = mx.random.normal((256,)).astype(mx.bfloat16)
    k_weight = mx.random.normal((256,)).astype(mx.bfloat16)
    q_norm = mx.fast.rms_norm(queries, q_weight, 1e-6).transpose(0, 2, 1, 3)
    k_norm = mx.fast.rms_norm(keys, k_weight, 1e-6).transpose(0, 2, 1, 3)
    q_expected = mx.fast.rope(
        q_norm,
        64,
        traditional=False,
        base=10_000_000.0,
        scale=1.0,
        offset=37,
    )
    k_expected = mx.fast.rope(
        k_norm,
        64,
        traditional=False,
        base=10_000_000.0,
        scale=1.0,
        offset=37,
    )

    q_actual, k_actual = qwen38_qk_rms_rope(
        queries,
        keys,
        q_weight,
        k_weight,
        1e-6,
        37,
    )
    mx.eval(q_expected, k_expected, q_actual, k_actual)

    assert mx.array_equal(q_actual, q_expected).item()
    assert mx.array_equal(k_actual, k_expected).item()


def test_qk_rms_rope_accepts_tensor_offset_inside_compiled_verify() -> None:
    queries = mx.random.normal((1, 4, 24, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 4, 4, 256)).astype(mx.bfloat16)
    q_weight = mx.random.normal((256,)).astype(mx.bfloat16)
    k_weight = mx.random.normal((256,)).astype(mx.bfloat16)
    mx.eval(queries, keys, q_weight, k_weight)

    compiled = mx.compile(
        lambda q, k, offset: qwen38_qk_rms_rope(
            q,
            k,
            q_weight,
            k_weight,
            1e-6,
            offset,
        )
    )

    q_actual, k_actual = compiled(
        queries,
        keys,
        mx.array(37, dtype=mx.int32),
    )
    mx.eval(q_actual, k_actual)

    q_expected, k_expected = qwen38_qk_rms_rope(
        queries,
        keys,
        q_weight,
        k_weight,
        1e-6,
        37,
    )
    mx.eval(q_expected, k_expected)
    assert mx.array_equal(q_actual, q_expected).item()
    assert mx.array_equal(k_actual, k_expected).item()
