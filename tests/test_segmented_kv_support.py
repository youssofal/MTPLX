"""Models the segment kernel cannot serve keep the stock cache (and say why), never the gather route."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import mtplx.segmented_kv as module
from mtplx.cache_state import configure_tail_owned_attention_kv_cache
from mtplx.segmented_kv import (
    evaluate_model_support,
    model_support,
    reset_model_support,
    segmented_kv_enabled,
    segmented_kv_health,
    segmented_ssd_enabled,
)


@pytest.fixture(autouse=True)
def _cpu_and_clean(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    reset_model_support()
    # The sustained profile's attention switches: neither is set.
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    monkeypatch.delenv("MTPLX_NAX_FLASH_ROUTE", raising=False)
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    monkeypatch.setattr("mtplx.nax_verify.nax_available", lambda: True)
    yield
    reset_model_support()
    mx.set_default_device(previous)


def _attn(head_dim=256, q=24, kv=4, packed=True):
    return SimpleNamespace(
        head_dim=head_dim, num_attention_heads=q, num_key_value_heads=kv, _mtplx_gqa_packed_sdpa_enabled=packed
    )


def test_a_supported_model_stays_enabled() -> None:
    verdict = evaluate_model_support([_attn(), _attn()])
    assert verdict["supported"] and verdict["reasons"] == []
    assert verdict["head_dims"] == [256] and verdict["gqa"] == [6]
    assert segmented_kv_enabled() is True


@pytest.mark.parametrize(
    "layers, hooked, reason",
    [
        ([_attn(head_dim=64, q=16, kv=4)], None, "head_dim_64_unsupported"),
        ([_attn(head_dim=256, q=40, kv=1)], None, "gqa_40_1_unsupported"),
        ([_attn()], [False], "attention_not_hooked"),
        ([], None, "no_full_attention_layers"),
        ([SimpleNamespace(head_dim=None)], None, "attention_shape_unknown"),
    ],
)
def test_an_unsupported_model_switches_the_feature_off_with_a_reason(layers, hooked, reason, capsys) -> None:
    verdict = evaluate_model_support(layers, hooked=hooked)
    assert not verdict["supported"] and reason in verdict["reasons"]
    assert segmented_kv_enabled() is False
    assert segmented_ssd_enabled() is False
    out = capsys.readouterr().out
    assert out.count("MTPLX_SEGMENTED_KV is on but this model cannot use it") == 1 and reason in out


def test_the_log_line_appears_once(capsys) -> None:
    evaluate_model_support([_attn(head_dim=64)])
    evaluate_model_support([_attn(head_dim=64)])
    assert capsys.readouterr().out.count("keeping the stock KV cache") == 1


@pytest.mark.parametrize("value", [None, "1"])
def test_the_attention_switches_unset_or_on_leave_the_verdict_supported(monkeypatch, value) -> None:
    """Segments run their own kernel under any profile: sustained sets neither switch."""
    for name in ("MTPLX_GQA_PACKED_SDPA", "MTPLX_NAX_FLASH_ROUTE"):
        if value is not None:
            monkeypatch.setenv(name, value)
    assert evaluate_model_support([_attn(packed=False)])["supported"] is True
    assert segmented_kv_enabled() is True


@pytest.mark.parametrize(
    "name, reason", [("MTPLX_GQA_PACKED_SDPA", "gqa_packed_sdpa_off"), ("MTPLX_NAX_FLASH_ROUTE", "nax_flash_route_off")]
)
@pytest.mark.parametrize("value", ["0", "off"])
def test_an_attention_switch_set_to_0_is_a_kill_switch(monkeypatch, name, reason, value) -> None:
    monkeypatch.setenv(name, value)
    verdict = evaluate_model_support([_attn()])
    assert not verdict["supported"] and verdict["reasons"] == [reason]
    assert segmented_kv_enabled() is False


def test_missing_nax_is_a_reason(monkeypatch) -> None:
    monkeypatch.setattr("mtplx.nax_verify.nax_available", lambda: False)
    assert "nax_unavailable" in evaluate_model_support([_attn()])["reasons"]


def test_an_unsupported_model_keeps_stock_caches_and_never_builds_a_segmented_one() -> None:
    evaluate_model_support([_attn(head_dim=64)])
    cache = [KVCache(), KVCache()]
    configure_tail_owned_attention_kv_cache(cache)
    assert all(type(c) is KVCache for c in cache)
    for c in cache:
        c.update_and_fetch(mx.zeros((1, 2, 9000, 8)), mx.zeros((1, 2, 9000, 8)))
    configure_tail_owned_attention_kv_cache(cache)  # what the repage after a prefill runs
    assert all(type(c) is KVCache for c in cache)


def test_health_reports_the_reason_and_is_absent_when_not_requested(monkeypatch) -> None:
    evaluate_model_support([_attn(head_dim=64)])
    block = segmented_kv_health([])
    assert block["enabled"] is False and block["requested"] is True
    assert "head_dim_64_unsupported" in block["model_support"]["reasons"]
    monkeypatch.delenv("MTPLX_SEGMENTED_KV")
    assert segmented_kv_health([]) is None


def test_health_of_a_supported_model_carries_the_verdict() -> None:
    evaluate_model_support([_attn()])
    block = segmented_kv_health([])
    assert block["enabled"] is True and block["model_support"]["supported"] is True


def test_without_a_verdict_on_the_model_the_feature_stays_off() -> None:
    """A model that loads without configure_split_full_attention (Laguna, a script) has no hook: with
    segment caches its attention would gather every segment on every call, so no verdict means off."""
    assert model_support() is None and module._MODEL_SUPPORT is None
    assert segmented_kv_enabled() is False and segmented_ssd_enabled() is False


def test_a_load_route_without_a_verdict_builds_stock_caches_and_health_says_why() -> None:
    cache = [KVCache(), KVCache()]
    stats = configure_tail_owned_attention_kv_cache(cache)
    assert all(type(c) is KVCache for c in cache)
    assert stats.get("mode") != "segmented"
    block = segmented_kv_health([])
    assert block["enabled"] is False and block["requested"] is True
    assert block["model_support"]["reasons"] == ["model_not_checked"]
    evaluate_model_support([_attn()])  # the verdict a supported model gets at load
    cache = [KVCache()]
    configure_tail_owned_attention_kv_cache(cache)
    from mtplx.segmented_kv import SegmentedKVCache

    assert isinstance(cache[0], SegmentedKVCache)


def test_every_runtime_load_starts_without_a_verdict() -> None:
    """runtime.load resets the verdict before the model's own check, so a Laguna load after a
    supported model does not inherit that model's verdict."""
    import inspect

    from mtplx import runtime

    evaluate_model_support([_attn()])
    assert segmented_kv_enabled() is True
    source = inspect.getsource(runtime)
    reset_at = source.index("reset_model_support()")
    assert "if not _is_laguna_s_2_1_mlx_4bit_config(config):" in source[reset_at : reset_at + 400]
    reset_model_support()
    assert segmented_kv_enabled() is False


def test_configure_split_full_attention_decides_from_the_hooked_layers(monkeypatch) -> None:
    from mlx_lm.models.qwen3_next import Qwen3NextAttention

    from mtplx.attention_split import configure_split_full_attention

    def model(head_dim):
        args = SimpleNamespace(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim, attention_bias=False,
            rms_norm_eps=1e-6, partial_rotary_factor=0.25, rope_theta=10000.0, rope_scaling=None,
            max_position_embeddings=1024,
        )
        attn = Qwen3NextAttention(args)
        layer = SimpleNamespace(is_linear=False, self_attn=attn)
        return SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    stats = configure_split_full_attention(model(32))
    assert stats["segmented_kv_supported"] is False
    assert "head_dim_32_unsupported" in model_support()["reasons"]
    assert segmented_kv_enabled() is False
    reset_model_support()
    stats = configure_split_full_attention(model(256))
    assert stats["segmented_kv_supported"] is True
    assert segmented_kv_enabled() is True


def test_head_dim_is_read_from_the_scale_when_the_module_has_no_attribute() -> None:
    """mlx-lm's Qwen3 attention (head_dim 128) keeps n_heads, n_kv_heads and scale only."""
    attn = SimpleNamespace(n_heads=32, n_kv_heads=8, scale=128**-0.5)
    verdict = evaluate_model_support([attn])
    assert verdict["supported"], verdict
    assert verdict["head_dims"] == [128] and verdict["gqa"] == [4]


def _qwen3_attention(head_dim=32, q=4, kv=2, hidden=64):
    from mlx_lm.models.qwen3 import Attention, ModelArgs

    args = ModelArgs(
        model_type="qwen3", hidden_size=hidden, num_hidden_layers=1, intermediate_size=64, num_attention_heads=q,
        rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=kv, max_position_embeddings=4096, rope_theta=10000.0,
        head_dim=head_dim, tie_word_embeddings=False,
    )
    mx.random.seed(0)
    return Attention(args)


def test_plain_qwen3_attention_is_hooked_for_segmented_caches_only_and_matches_the_stock_forward() -> None:
    """A plain q/k-norm attention (no gate) goes through the hook body only on a segmented cache; with a
    stock cache it is the unchanged mlx-lm forward, so models without the switch keep their code."""
    from mtplx.attention_split import (
        _attention_has_gated_q_proj,
        _attention_has_plain_q_proj,
        _install_split_attention_hook,
    )
    from mtplx.segmented_kv import SegmentedKVCache

    attn = _qwen3_attention()
    assert not _attention_has_gated_q_proj(attn) and _attention_has_plain_q_proj(attn)

    def run(cache):
        outs = []
        for seed, n in ((1, 40), (2, 9), (3, 1)):
            x = mx.random.normal((1, n, 64), key=mx.random.key(seed))
            outs.append(attn(x, mask="causal" if n > 1 else None, cache=cache))
        mx.eval(outs)
        return outs

    before = run(KVCache())
    _install_split_attention_hook(attn)
    attn._mtplx_split_full_attention_enabled = True
    after_stock = run(KVCache())
    for a, b in zip(before, after_stock):
        assert mx.array_equal(a, b).item()  # stock cache: the unchanged forward
    cache = SegmentedKVCache()
    seg = run(cache)
    cache.seal()
    for a, b in zip(before, seg):
        assert float(mx.abs(a - b).max().item()) < 1e-4  # same math on the gathered rows (CPU: no kernel)
    assert cache.offset == 50


def test_the_hook_covers_plain_attention_in_the_support_check() -> None:
    from types import SimpleNamespace

    from mtplx.attention_split import configure_split_full_attention

    attn = _qwen3_attention(head_dim=128, q=8, kv=2, hidden=128)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(is_linear=False, self_attn=attn)]))
    stats = configure_split_full_attention(model)
    assert stats["segmented_kv_supported"] is True
    assert model_support()["head_dims"] == [128] and model_support()["gqa"] == [4]


def test_without_the_lse_kernel_one_line_says_prefill_gathers_and_the_feature_stays_on(monkeypatch, capsys) -> None:
    monkeypatch.setattr(module, "_SDPA_LSE", False)
    monkeypatch.delenv("MTPLX_SEGMENTED_KV_PREFILL", raising=False)
    verdict = evaluate_model_support([_attn()])
    assert verdict["supported"] and segmented_kv_enabled()
    assert module.prefill_route() == "gather"
    assert capsys.readouterr().out.count("no logsumexp output") == 1
    evaluate_model_support([_attn()])
    assert "no logsumexp output" not in capsys.readouterr().out  # once per process


def test_with_the_lse_kernel_there_is_no_note(monkeypatch, capsys) -> None:
    monkeypatch.setattr(module, "_SDPA_LSE", True)
    evaluate_model_support([_attn()])
    assert "no logsumexp output" not in capsys.readouterr().out


def _llama_attention(head_dim=32, q=4, kv=2, hidden=64, bias=False):
    from mlx_lm.models.llama import Attention, ModelArgs

    args = ModelArgs(
        model_type="mistral", hidden_size=hidden, num_hidden_layers=1, intermediate_size=64, num_attention_heads=q,
        rms_norm_eps=1e-5, vocab_size=64, head_dim=head_dim, num_key_value_heads=kv, rope_theta=1e6,
        attention_bias=bias, tie_word_embeddings=False,
    )
    mx.random.seed(0)
    return Attention(args)


def test_llama_style_attention_is_recognised_and_other_forms_are_not() -> None:
    from mtplx.attention_split import (
        _attention_has_gated_q_proj,
        _attention_has_plain_q_proj,
        _attention_has_unnormed_q_proj,
    )

    llama = _llama_attention()
    assert _attention_has_unnormed_q_proj(llama)
    assert not _attention_has_plain_q_proj(llama) and not _attention_has_gated_q_proj(llama)
    assert not _attention_has_unnormed_q_proj(_qwen3_attention())  # has q/k norm: the plain form
    assert not _attention_has_unnormed_q_proj(SimpleNamespace(q_proj=None, n_heads=4, head_dim=32))


@pytest.mark.parametrize("bias", [False, True])
def test_llama_style_attention_is_hooked_for_segmented_caches_only_and_matches_the_stock_forward(bias) -> None:
    """Llama/Mistral attention (no q/k norm) goes through the hook body only on a segmented cache; with
    a stock cache, switch on or off, it is the unchanged mlx-lm forward bit for bit."""
    from mtplx.attention_split import _install_split_attention_hook
    from mtplx.segmented_kv import SegmentedKVCache

    attn = _llama_attention(bias=bias)

    def run(cache):
        outs = []
        for seed, n in ((1, 40), (2, 9), (3, 1)):
            x = mx.random.normal((1, n, 64), key=mx.random.key(seed))
            outs.append(attn(x, mask="causal" if n > 1 else None, cache=cache))
        mx.eval(outs)
        return outs

    before = run(KVCache())
    _install_split_attention_hook(attn)
    for enabled in (False, True):
        attn._mtplx_split_full_attention_enabled = enabled
        for a, b in zip(before, run(KVCache())):
            assert mx.array_equal(a, b).item()
    attn._mtplx_split_full_attention_enabled = True
    cache = SegmentedKVCache()
    seg = run(cache)
    cache.seal()
    for a, b in zip(before, seg):
        assert float(mx.abs(a - b).max().item()) < 1e-4  # same math on the gathered rows (CPU: no kernel)
    assert cache.offset == 50


def test_the_hook_covers_llama_style_attention_in_the_support_check(monkeypatch) -> None:
    from mtplx.attention_split import configure_split_full_attention

    attn = _llama_attention(head_dim=128, q=8, kv=2, hidden=128)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]))
    stats = configure_split_full_attention(model)
    assert stats["segmented_kv_supported"] is True
    assert model_support()["head_dims"] == [128] and model_support()["gqa"] == [4]


# ---- the kernel self-check decides before the first cache (any profile) --------------------


@pytest.fixture
def selfcheck():
    from mtplx import kernel_selfcheck

    kernel_selfcheck._reset_for_tests()
    yield kernel_selfcheck
    kernel_selfcheck._reset_for_tests()


def test_requesting_segments_runs_the_selfcheck_under_a_profile_without_turbo_lanes(selfcheck, monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_KERNEL_SELFCHECK", raising=False)
    for name in ("MTPLX_NAX_VERIFY", "MTPLX_QWEN_ROW_OWNED_ROUTER", "MTPLX_QWEN_COMBINE_TAIL",
                 "MTPLX_FUSE_GDN_POST_CONV", "MTPLX_A3B_WHOLE_MOE_FUSION"):
        monkeypatch.delenv(name, raising=False)
    assert selfcheck.selfcheck_enabled() is True
    monkeypatch.delenv("MTPLX_SEGMENTED_KV")
    assert selfcheck.selfcheck_enabled() is False  # flag off: unchanged


def test_the_selfcheck_probes_the_segment_kernel_at_the_models_shapes(selfcheck, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(selfcheck, "_check_segments", lambda mx, dtype, *, d, gqa: seen.append((d, gqa)) or 0.0)
    evaluate_model_support([_attn(head_dim=128, q=32, kv=8)])
    report = selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert seen == [(128, 4)]
    assert report["lanes"]["segmented_kv_sdpa"] == "ok"
    assert segmented_kv_enabled() is True


@pytest.mark.parametrize("failure", ["mismatch", "raises", "bails"])
def test_a_failed_segment_kernel_means_stock_caches_with_a_reason(selfcheck, monkeypatch, capsys, failure) -> None:
    from mtplx.segmented_kv import SegmentedKVCache

    def probe(mx, dtype, *, d, gqa):
        if failure == "raises":
            raise RuntimeError("no such kernel")
        return 1.0 if failure == "mismatch" else float("inf")

    monkeypatch.setattr(selfcheck, "_check_segments", probe)
    evaluate_model_support([_attn()])
    assert segmented_kv_enabled() is True
    report = selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["segmented_kv_sdpa"] == "fallback"
    assert segmented_kv_enabled() is False and segmented_ssd_enabled() is False
    assert "segment_kernel_selfcheck_failed" in model_support()["reasons"]
    cache = [KVCache(), KVCache()]
    configure_tail_owned_attention_kv_cache(cache)
    assert all(type(c) is KVCache for c in cache) and not any(isinstance(c, SegmentedKVCache) for c in cache)
    block = segmented_kv_health([])
    assert block["enabled"] is False and "segment_kernel_selfcheck_failed" in block["model_support"]["reasons"]
    selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)  # a second run (a later selfcheck) keeps it off
    assert segmented_kv_enabled() is False
    assert capsys.readouterr().out.count("MTPLX_SEGMENTED_KV is on but this model cannot use it") == 1


def test_a_selfcheck_that_cannot_run_leaves_segments_off(selfcheck, monkeypatch) -> None:
    monkeypatch.setattr(selfcheck, "_model_quant_signature", lambda model: (_ for _ in ()).throw(RuntimeError("x")))
    evaluate_model_support([_attn()])
    assert selfcheck.maybe_run_model_selfcheck(object()) is None
    assert segmented_kv_enabled() is False
    assert "segment_kernel_selfcheck_failed" in model_support()["reasons"]


def test_with_the_flag_off_the_selfcheck_skips_the_segment_lane_and_leaves_no_verdict(selfcheck, monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_SEGMENTED_KV")
    monkeypatch.setattr(selfcheck, "_check_segments", lambda *a, **k: pytest.fail("probed with the flag off"))
    evaluate_model_support([_attn()])
    report = selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["segmented_kv_sdpa"] == "skipped"
    assert segmented_kv_enabled() is False


# ---- the hook routes a segmented cache to the segment route under any profile ----------------


def _qwen3_next_attention():
    from mlx_lm.models.qwen3_next import Qwen3NextAttention

    args = SimpleNamespace(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=32, attention_bias=False,
        rms_norm_eps=1e-6, partial_rotary_factor=0.25, rope_theta=10000.0, rope_scaling=None,
        max_position_embeddings=1024,
    )
    mx.random.seed(0)
    return Qwen3NextAttention(args)


@pytest.mark.parametrize("make", [_qwen3_next_attention, _qwen3_attention, _llama_attention])
@pytest.mark.parametrize("profile_env", [{}, {"MTPLX_VLLM_METAL_PAGED_ATTN": "1"}])
def test_a_segmented_cache_takes_the_segment_route_without_the_attention_switches(monkeypatch, make, profile_env) -> None:
    """Sustained-like: no packed SDPA, no flash route, and either no attention route at all or only
    the paged one. A verify window over several segments goes to decode_segments_attention (never
    the stock forward or a gather); its output matches the stock cache's."""
    import mtplx.attention_split as split
    from mtplx.attention_split import configure_split_full_attention
    from mtplx.segmented_kv import SegmentedKVCache, gathered_view

    for name, value in profile_env.items():
        monkeypatch.setenv(name, value)
    attn = make()
    configure_split_full_attention(SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(is_linear=False, self_attn=attn)])))
    assert attn._mtplx_gqa_packed_sdpa_enabled is False
    assert attn._mtplx_split_full_attention_enabled is bool(profile_env)  # route off, or the paged route only
    calls = []

    def segment_route(queries, cache, *, scale, mask, packed_threshold):
        # CPU stand-in for the kernel: exact attention over the rows, read without a gather count.
        calls.append((int(queries.shape[2]), cache.segment_count))
        view = gathered_view(cache)
        q = int(queries.shape[2])
        causal = mx.arange(view.offset)[None, :] <= (view.offset - q + mx.arange(q)[:, None])
        return mx.fast.scaled_dot_product_attention(queries, view.keys, view.values, scale=scale, mask=causal)

    monkeypatch.setattr(split, "decode_segments_attention", segment_route)

    def run(cache):
        outs = [attn(mx.random.normal((1, 40, 64), key=mx.random.key(1)), mask="causal", cache=cache)]
        if isinstance(cache, SegmentedKVCache):
            state = cache.state  # turn end: the history becomes a sealed segment
            cache = SegmentedKVCache()
            cache.state = state
        outs.append(attn(mx.random.normal((1, 9, 64), key=mx.random.key(2)), mask="causal", cache=cache))
        outs.append(attn(mx.random.normal((1, 4, 64), key=mx.random.key(3)), mask="causal", cache=cache))
        mx.eval(outs)
        return outs

    stock = run(KVCache())
    module.route_counts.clear()
    seg = run(SegmentedKVCache())
    assert (4, 2) in calls and (9, 2) in calls
    assert not any(k.startswith("gather_q") and k[8:].isdigit() for k in module.route_counts), module.route_counts
    for a, b in zip(stock, seg):
        assert float(mx.abs(a - b).max().item()) < 1e-4
