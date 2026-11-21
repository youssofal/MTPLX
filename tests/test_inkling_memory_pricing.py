from types import SimpleNamespace

from mtplx.memory_plan import plan_memory
from mtplx.models.inkling.runtime import InklingRuntime
from mtplx.server.openai import _AdmissionGeometry, _admission_growth


def test_stock_sink_copy_prices_one_visit_and_preserves_steady_kv():
    native = SimpleNamespace(model=SimpleNamespace(config={"num_key_value_heads": 8, "head_dim": 128}),
                             target_footprint=SimpleNamespace(full_bytes_per_token=32768))
    rate = InklingRuntime.attention_kv_copy_bytes_per_token(native)
    assert rate == 4096
    args = dict(total_ram_bytes=32*2**30, model_weights_bytes=20_161_820_584, kv_bytes_per_token=32768,
                aux_bytes_per_token=4096, fixed_cache_bytes=179_999_232, requested_context=8192)
    existing = plan_memory(**args)
    revised = plan_memory(**args, attention_kv_copy_bytes_per_token=rate)
    assert existing.kv_reserve_bytes == revised.kv_reserve_bytes
    assert revised.runtime_transients_bytes == 3*2**30
    assert (existing.context_window_fit, revised.context_window_fit) == (28672, 24576)
    assert "attention_kv_copy_bytes_per_token" not in existing.to_dict()


def test_exact_restore_skips_prefill_copy_but_decode_still_reads_history():
    geometry = _AdmissionGeometry(36864, 36864, 0, 3*2**30, 20_161_820_584,
                                  attention_kv_copy_bytes_per_token=4096)
    args = dict(prompt_tokens=8192, restore_copies_prefix=False, layout="contiguous_dense_decode",
                source_layout="contiguous_dense_decode", output_tokens=48, publish=False, scratch_bytes=0)
    full = _admission_growth(geometry, reused_tokens=8192, **args)
    partial = _admission_growth(geometry, reused_tokens=8191, **args)
    assert (full["prefill_kv_copy_bytes"], full["decode_kv_copy_bytes"]) == (0, 8192*4096)
    assert partial["prefill_kv_copy_bytes"] == partial["decode_kv_copy_bytes"] == 8192*4096
    assert partial["growth_bytes"] == max(partial["prefill_end_bytes"], partial["decode_start_bytes"])


def test_a_chunked_prompt_read_is_512_rows_unless_a_width_is_named(monkeypatch):
    from mtplx.generation import prefill_chunk_size_override, prefill_forward_widths
    from mtplx.server.openai import _admission_narrow_widths, _served_prefill_chunk_auto

    for name in ("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")  # what every profile stamps
    rt = object.__new__(InklingRuntime)
    assert rt.prompt_chunk_rows(30_000) == 512
    assert _served_prefill_chunk_auto(SimpleNamespace(), rt) == 512  # what the app's settings show
    assert prefill_forward_widths(rt, 30_000, None) == [512]
    assert prefill_forward_widths(rt, 30_000, 4096) == [4096, 512]
    assert prefill_forward_widths(rt, 30_000, 256) == [256]
    assert _admission_narrow_widths(rt, [512]) == []
    with prefill_chunk_size_override(256):  # a warming read, or the width the admission chose
        assert rt.prompt_chunk_rows(30_000) == 256
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")  # the user's own width
    assert rt.prompt_chunk_rows(30_000) == 2048
    assert _served_prefill_chunk_auto(SimpleNamespace(), rt) == 2048
    assert prefill_forward_widths(rt, 30_000, None) == [2048]
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL")  # a profile that forwards the prompt whole
    assert rt.prompt_chunk_rows(30_000) == 2048
    assert prefill_forward_widths(rt, 30_000, None) == [None]


def test_the_request_wrapper_installs_the_models_width_and_keeps_a_named_one(monkeypatch):
    from mtplx import generation

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    for name in ("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"):
        monkeypatch.delenv(name, raising=False)
    rt = object.__new__(InklingRuntime)
    rt.backend_id, rt.model_path = "inkling2_loop", None
    seen = []

    @generation._with_inkling_attention_receipt
    def read(rt):
        seen.append((generation._prefill_chunk_size(), generation._iter_prefill_chunk_spans(1100)))
        return SimpleNamespace(stats=SimpleNamespace())

    read(rt)
    with generation.prefill_chunk_size_override(1024):
        read(rt)
    assert seen == [(512, [(0, 512), (512, 1024), (1024, 1100)]), (1024, [(0, 1024), (1024, 1100)])]
    assert generation.current_prefill_chunk_override() is None


def _decode_runtime(pool_bytes=207 << 20):
    """The pieces InklingRuntime's fixed-bytes arithmetic reads, at the catalog geometry."""
    import mlx.core as mx

    from mtplx.models.inkling.layout import CacheFootprint

    rt = object.__new__(InklingRuntime)
    row = 2 * 8 * 128 * 2  # key and value rows of one visit, BF16
    rt.target_footprint = CacheFootprint(full_bytes_per_token=8 * row, sliding_bytes_per_token=40 * row,
                                         window=1024, tail_bytes=0)
    rt.draft_footprint = None
    sliding = [True] * 5 + [False]
    rt.model = SimpleNamespace(
        config={"num_key_value_heads": 8, "head_dim": 128},
        visit_order=list(range(6)) * 8,
        layers=[SimpleNamespace(sliding=s) for s in sliding],
        embed_tokens=SimpleNamespace(weight=SimpleNamespace(dtype=mx.bfloat16)),
    )
    rt.engine_pool_bytes = pool_bytes
    rt.prompt_chunk_rows = lambda prompt_tokens=None: 512
    return rt, row


def test_a_short_prompt_is_priced_at_the_decode_layout_not_its_prompt_windows():
    """A 256-token prompt's rings become 1,024 rows at the first verify and its global banks a
    whole 4,096-row step; the request estimate holds that and the engine's pool (about 240 MiB
    more than the prompt-sized windows at the catalog geometry)."""
    rt, row = _decode_runtime()
    prompt_windows = 256 * 40 * row
    fixed = InklingRuntime.window_cache_bytes(rt)
    assert fixed == 1024 * 40 * row + 8 * 4096 * row + (207 << 20)
    assert InklingRuntime.prefill_window_bytes(rt, 256, 256) == fixed > prompt_windows
    # A long prompt's prefill windows (window + chunk - 1 rows) are the larger figure then.
    assert InklingRuntime.prefill_window_bytes(rt, 100_000, 100_000) == max(
        (1024 + 511) * 40 * row, fixed)


def test_the_engine_pool_is_priced_at_the_widest_copy_panel():
    """Warm-up records programs up to 8 rows; a copy round on the fast step records up to its
    block cap plus the anchor (25 rows by default), and the one pool grows to the widest program
    (81 MB after warm-up, 231 MB after a 25-row copy round on the native pack). The price scales
    the measured pool by rows to the widest panel; never below the measured pool."""
    import mtplx.models.inkling.runtime as runtime_module
    from mtplx.models.inkling.runtime import engine_pool_price, widest_panel

    mb = 1 << 20
    assert engine_pool_price(81 * mb, 25) == 81 * mb * 25 // 8 >= 231 * mb
    assert engine_pool_price(81 * mb, 8) == engine_pool_price(81 * mb, 1) == 81 * mb
    assert engine_pool_price(None, 25) == 0
    rt, row = _decode_runtime(pool_bytes=81 * mb)
    rt.engine_pool_bound = engine_pool_price(rt.engine_pool_bytes, 25)
    assert InklingRuntime.window_cache_bytes(rt) == 1024 * 40 * row + 8 * 4096 * row + 81 * mb * 25 // 8
    # The widest panel: the copy cap plus its anchor where wide panels are exact, else 8.
    original = runtime_module.wide_rows_exact
    try:
        runtime_module.wide_rows_exact = lambda model, spec: True
        assert widest_panel(None) == 25
        runtime_module.wide_rows_exact = lambda model, spec: False
        assert widest_panel(None) == 8
    finally:
        runtime_module.wide_rows_exact = original


def test_load_makes_target_and_drafter_weights_resident_but_not_the_vision_tower(monkeypatch):
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    import mtplx.models.inkling.runtime as runtime_module

    seen = []
    monkeypatch.setattr(runtime_module.mx, "eval", lambda *trees: seen.extend(trees))
    target, drafter = nn.Linear(4, 3), nn.Linear(3, 2)
    runtime_module.make_weights_resident(target, drafter)
    names = [sorted(k for k, _ in tree_flatten(tree)) for tree in seen]
    assert names == [["bias", "weight"], ["bias", "weight"]]
    seen.clear()
    runtime_module.make_weights_resident(target, None)
    assert [sorted(k for k, _ in tree_flatten(tree)) for tree in seen] == [["bias", "weight"], []]


def test_the_family_reserve_replaces_the_flat_three_gib_and_admits_a_tight_mac():
    gib = 2**30
    rt = object.__new__(InklingRuntime)
    assert InklingRuntime.prefill_working_bytes(rt) == 2 * gib  # 512-row chunks, measured
    assert InklingRuntime.prefill_working_bytes(rt, 256) == 2 * gib  # narrower: not measured, same charge
    assert InklingRuntime.prefill_working_bytes(rt, 1024) == 4 * gib  # wider: in proportion
    # Bare Speed with a 4-bit embedding (16.0 GB of weights) on a 24 GB Mac: the flat reserve and the
    # 1 GiB bank floor refuse it; the measured reserve with the tight-machine rule admits it.
    args = dict(total_ram_bytes=24 * gib, model_weights_bytes=15_997_000_000, kv_bytes_per_token=32768,
                aux_bytes_per_token=4096, attention_kv_copy_bytes_per_token=4096, fixed_cache_bytes=314_572_800)
    flat = plan_memory(**args)
    measured = plan_memory(**args, runtime_transients_bytes=2 * gib, tight_machine_measured=True)
    assert not flat.model_fits
    assert measured.model_fits and measured.tight_machine and measured.runtime_transients_bytes == 2 * gib + 256 * 2**20
    assert measured.context_window_fit == 12288
    # Every other family keeps the flat reserve: the default is unchanged.
    assert plan_memory(**{**args, "total_ram_bytes": 48 * gib}).runtime_transients_bytes == 3 * gib


def test_admission_charges_a_family_its_measured_prefill_working_set():
    from mtplx.server.openai import _admission_scratch_bytes

    rt = object.__new__(InklingRuntime)
    state = SimpleNamespace(runtime=rt)
    geometry = _AdmissionGeometry(40960, 40960, 0, 3 * 2**30, 4096)
    assert _admission_scratch_bytes(state, rows=512, prompt_tokens=4142, geometry=geometry) == (2 * 2**30, "runtime_measured")
