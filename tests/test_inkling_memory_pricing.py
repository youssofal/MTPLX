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
    rt.engine_pool_now = lambda: 0  # an engine that cannot say: the whole pool is charged
    prompt_windows = 256 * 40 * row
    fixed = InklingRuntime.window_cache_bytes(rt)
    assert fixed == 1024 * 40 * row + 8 * 4096 * row + (207 << 20)
    assert InklingRuntime.prefill_window_bytes(rt, 256, 256) == fixed > prompt_windows
    # A long prompt's prefill windows (window + chunk - 1 rows) are the larger figure then.
    assert InklingRuntime.prefill_window_bytes(rt, 100_000, 100_000) == max(
        (1024 + 511) * 40 * row, fixed)


def test_the_plan_prices_the_resident_pool_and_load_arrays_and_a_request_only_what_is_new():
    """The engine's pool is allocated through MLX, so it is inside the active memory the admission
    measures (21 November, Bare Speed: a 12,322-token prompt refused at 24 GB because its bill
    charged the 775 MB pool price on top of the 341 MB pool already resident). The plan prices the
    pool at its bound and the arrays the load built beside the weights; a request is charged the
    rings, the banks' rounding and what the pool can still grow by."""
    rt, row = _decode_runtime(pool_bytes=403 << 20)
    rt.engine_pool_bound = 403 << 20
    rt.load_extra_bytes = 172 << 20
    rings, rounding = 1024 * 40 * row, 8 * 4096 * row
    assert InklingRuntime.window_cache_bytes(rt) == rings + rounding + (403 << 20) + (172 << 20)
    rt.engine_pool_now = lambda: 341 << 20
    assert InklingRuntime.admission_fixed_bytes(rt) == rings + rounding + (62 << 20)
    rt.engine_pool_now = lambda: 403 << 20  # at its bound: nothing more
    assert InklingRuntime.admission_fixed_bytes(rt) == rings + rounding
    assert InklingRuntime.prefill_window_bytes(rt, 256, 256) == rings + rounding


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
    # The load recorded the widest panel's program: its pool holds every width inside the first
    # capacity step, and a longer conversation adds a measured part (Bare Speed through `mtplx serve`,
    # 21 November: 322 MB at ready, 410 MB after the first 4K read, 413 MB after a 20K one, where the
    # scaling priced 1,007).
    ready = 322_495_208
    assert engine_pool_price(ready, 25, widest_recorded=True) == ready + (96 << 20) >= 413_000_000
    assert engine_pool_price(0, 25, widest_recorded=True) == 0
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


def test_the_family_reserve_is_its_forward_and_its_checkpoints_and_admits_a_tight_mac():
    gib = 2**30
    rt = _checkpoint_runtime(_CATALOG_TEXT, _CATALOG_DRAFT)
    forward, record = InklingRuntime.FORWARD_WORKING_BYTES, 179_999_232
    assert InklingRuntime.prefill_working_bytes(rt) == forward + 8 * record  # the shared eight checkpoints
    assert InklingRuntime.prefill_working_bytes(rt, few=True) == forward + 2 * record  # where memory sets the window
    assert InklingRuntime.prefill_working_bytes(rt, 256, few=True) == forward + 2 * record  # narrower: same charge
    assert InklingRuntime.prefill_working_bytes(rt, 1024, few=True) == 2 * forward + 2 * record  # wider: in proportion
    # Bare Speed (15.99 GB of weights with its vision tower) on a 24 GB Mac, at the server's fixed term as served
    # (21 November: rings at the full window 180 MB, bank rounding 134 MB, the pool at ready 322 MB plus its
    # growth past the first capacity step, the load's 33 MB and the warm-up's arrays): the flat reserve refuses
    # it, one 2 GiB reserve for both rules leaves the 4,096-token floor, the forward and two checkpoints under the
    # tight rule admit the rule's shared ceiling (20,480; its own room is 24,576 tokens).
    fixed = 179_999_232 + 134_217_728 + 322_495_208 + (96 << 20) + 32_996_786 + InklingRuntime.WARMUP_ARRAYS_BYTES
    args = dict(total_ram_bytes=24 * gib, model_weights_bytes=15_993_374_536, kv_bytes_per_token=32768,
                aux_bytes_per_token=4096, attention_kv_copy_bytes_per_token=4096, fixed_cache_bytes=fixed,
                tight_machine_measured=True)
    assert not plan_memory(**{**args, "tight_machine_measured": False}).model_fits
    one = plan_memory(**args, runtime_transients_bytes=2 * gib)
    assert not one.model_fits and one.context_window_fit == 4096
    split = plan_memory(**args, runtime_transients_bytes=InklingRuntime.prefill_working_bytes(rt),
                        tight_runtime_transients_bytes=InklingRuntime.prefill_working_bytes(rt, few=True))
    assert split.model_fits and split.tight_machine
    assert split.runtime_transients_bytes == forward + 2 * record + 256 * 2**20
    assert split.context_window_fit == 20480
    # Two checkpoints wherever memory sets the window: a tight Mac, or under 16 GiB of engine budget above the
    # weights (every 24 to 36 GB Mac with these packs); from 48 GB the shared eight.
    for ram, few in ((24, True), (32, True), (36, True), (48, False), (128, False)):
        plan = plan_memory(**{**args, "total_ram_bytes": ram * gib}, runtime_transients_bytes=forward + 8 * record,
                           tight_runtime_transients_bytes=forward + 2 * record)
        assert InklingRuntime.keeps_few_checkpoints(rt, plan) is few, ram
    assert InklingRuntime.keeps_few_checkpoints(rt, None) is False
    # The plan itself: a Mac with room plans with the reserve it is given (the tight one plays no part there),
    # and every other family keeps the flat reserve: the defaults are unchanged.
    roomy = plan_memory(**{**args, "total_ram_bytes": 36 * gib}, runtime_transients_bytes=forward + 8 * record,
                        tight_runtime_transients_bytes=forward + 2 * record)
    assert not roomy.tight_machine and roomy.runtime_transients_bytes == forward + 8 * record
    assert roomy.context_window_fit == plan_memory(**{**args, "total_ram_bytes": 36 * gib},
                                                   runtime_transients_bytes=forward + 8 * record).context_window_fit
    assert plan_memory(**{**args, "total_ram_bytes": 48 * gib,
                          "tight_machine_measured": False}).runtime_transients_bytes == 3 * gib


def test_admission_charges_a_family_its_measured_prefill_working_set():
    from mtplx.server.openai import _admission_scratch_bytes

    rt = _checkpoint_runtime(_CATALOG_TEXT, _CATALOG_DRAFT)
    forward, record = InklingRuntime.FORWARD_WORKING_BYTES, 179_999_232
    geometry = _AdmissionGeometry(40960, 40960, 0, 3 * 2**30, 4096)
    weights = 15_993_374_536
    for plan, kept in ((None, 8), (SimpleNamespace(tight_machine=False), 8), (SimpleNamespace(tight_machine=True), 2),
                       (SimpleNamespace(tight_machine=False, usable_bytes=27 * 2**30, model_weights_bytes=weights), 2),
                       (SimpleNamespace(tight_machine=False, usable_bytes=96 * 2**30, model_weights_bytes=weights), 8)):
        state = SimpleNamespace(runtime=rt, memory_plan=plan)
        assert _admission_scratch_bytes(state, rows=512, prompt_tokens=4142, geometry=geometry) == (
            forward + kept * record, "runtime_measured")


_CATALOG_TEXT = {"num_hidden_layers": 24, "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3},
                 "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
                 "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2304, "sconv_kernel_size": 4,
                 "sliding_window_size": 1024}
_CATALOG_DRAFT = {"num_hidden_layers": 3, "local_layer_ids": [0, 2], "num_key_value_heads": 8, "head_dim": 128,
                  "hidden_size": 2304, "sliding_window_size": 1024}


def _checkpoint_runtime(text, draft):
    from mtplx.models.inkling.layout import cache_footprint

    rt = object.__new__(InklingRuntime)
    rt.target_footprint = cache_footprint(text)
    rt.draft_footprint = cache_footprint(draft, draft=True)
    return rt


def test_a_checkpoint_is_priced_at_what_its_snapshot_copies():
    """Real target and drafter caches after a read longer than the window: a checkpoint copies every ring's
    last window and every convolution tail, and none of the global KV (trimmed instead). Its price is exactly
    those bytes; at the catalog geometry 180 MB (40 rings and 48 tails of the target, two drafter rings and
    the drafter's feature)."""
    import mlx.core as mx

    from mtplx.cache_state import snapshot_untrimmable_cache
    from mtplx.checkpoint_anchors import record_nbytes
    from mtplx.models.inkling.draft import Drafter
    from mtplx.models.inkling.layout import layer_visits
    from mtplx.models.inkling.model import Model

    text = {**_CATALOG_TEXT, "num_hidden_layers": 4, "loop_config": {"core_start": 1, "core_end": 3, "num_passes": 2},
            "local_layer_ids": [0, 1, 3], "num_key_value_heads": 2, "head_dim": 8, "hidden_size": 16,
            "sliding_window_size": 8}
    draft = {**_CATALOG_DRAFT, "num_key_value_heads": 2, "head_dim": 8, "hidden_size": 16, "sliding_window_size": 8}
    local = set(text["local_layer_ids"])
    target = Model.make_cache(SimpleNamespace(
        visit_order=layer_visits(text), config=text,
        layers=[SimpleNamespace(sliding=i in local) for i in range(text["num_hidden_layers"])]))
    drafter = Drafter.make_cache(SimpleNamespace(blocks=[
        SimpleNamespace(window=draft["sliding_window_size"] if i in draft["local_layer_ids"] else None)
        for i in range(draft["num_hidden_layers"])]))
    rows, heads, dim, hidden, taps = 20, 2, 8, 16, text["sconv_kernel_size"] - 1

    def kv(n):
        return mx.ones((1, heads, n, dim), dtype=mx.bfloat16)

    for attention, conv in zip(target[0::2], target[1::2]):
        attention.update_and_fetch(kv(rows), kv(rows))  # one chunk wider than the window
        for slot, width in enumerate((heads * dim, heads * dim, hidden, hidden)):
            conv[slot] = mx.zeros((1, taps, width), mx.float32)
    for attention in drafter[:-1]:
        attention.update_and_fetch(kv(8), kv(8))  # the drafter observes at most a window at a time
    drafter[-1][0] = mx.zeros((1, 1, hidden), mx.bfloat16)
    snapshot = snapshot_untrimmable_cache(target + drafter)
    rt = _checkpoint_runtime(text, draft)
    assert record_nbytes((rows, snapshot, None)) == InklingRuntime.checkpoint_record_bytes(rt) == 7712
    assert InklingRuntime.checkpoint_record_bytes(_checkpoint_runtime(_CATALOG_TEXT, _CATALOG_DRAFT)) == 179_999_232


def test_a_mac_whose_window_memory_sets_keeps_two_restore_checkpoints_and_a_roomy_one_eight():
    from mtplx.checkpoint_anchors import AnchorPlan, retain_checkpoints
    from mtplx.engine_session import EngineSessionManager

    rt = _checkpoint_runtime(_CATALOG_TEXT, _CATALOG_DRAFT)
    record = InklingRuntime.checkpoint_record_bytes(rt)
    assert InklingRuntime.checkpoint_budget_bytes(rt) is None
    assert InklingRuntime.checkpoint_budget_bytes(rt, SimpleNamespace(tight_machine=False)) is None
    roomy = SimpleNamespace(tight_machine=False, usable_bytes=96 * 2**30, model_weights_bytes=15_993_374_536)
    assert InklingRuntime.checkpoint_budget_bytes(rt, roomy) is None
    budget = InklingRuntime.checkpoint_budget_bytes(rt, SimpleNamespace(tight_machine=True))
    assert 2 * record < budget < 3 * record
    tier36 = SimpleNamespace(tight_machine=False, usable_bytes=27 * 2**30, model_weights_bytes=15_993_374_536)
    assert InklingRuntime.checkpoint_budget_bytes(rt, tier36) == budget
    # A cold read of a 16,430-token prompt: every 512-row chunk edge, the tail ladder's top rung 64 tokens
    # before the prompt end, and the prompt end.
    records = [(p, SimpleNamespace(nbytes=record), None) for p in (*range(512, 16_129, 512), 16_365, 16_429)]
    kept = lambda plan: [r[0] for r in retain_checkpoints(records, plan)]  # noqa: E731
    tight = AnchorPlan(budget_bytes=budget, prompt_end=16_429)
    assert kept(tight) == [16_365, 16_429]  # the prompt-end anchor and the newest
    # A read that resumed at 4,096 keeps that restore point over the newest.
    assert kept(AnchorPlan(budget_bytes=budget, prompt_end=16_429, restore_point=4096)) == [4096, 16_365]
    assert len(kept(AnchorPlan(prompt_end=16_429))) == 8  # no budget: the shared eight
    # The server hands the budget to the bank every prefill reads it from.
    assert EngineSessionManager(checkpoint_budget_bytes=budget).bank.checkpoint_budget_bytes == budget
    assert EngineSessionManager().bank.checkpoint_budget_bytes is None


def test_the_server_plans_with_the_checkpoints_the_reads_keep():
    """The server's plan inputs for Inkling: the forward plus two checkpoints wherever memory sets the window
    (24, 32 and 36 GB with Bare Speed), plus eight from 48 GB; the tight rule always has the two."""
    from mtplx.server.openai import _family_reserve_inputs

    gib = 2**30
    rt = _checkpoint_runtime(_CATALOG_TEXT, _CATALOG_DRAFT)
    forward, record = InklingRuntime.FORWARD_WORKING_BYTES, 179_999_232
    for ram, kept in ((24, 2), (32, 2), (36, 2), (48, 8), (128, 8)):
        inputs = dict(total_ram_bytes=ram * gib, model_weights_bytes=15_993_374_536, kv_bytes_per_token=32768,
                      aux_bytes_per_token=4096, attention_kv_copy_bytes_per_token=4096,
                      fixed_cache_bytes=1_088_788_347)
        _family_reserve_inputs(rt, inputs, plan_memory)
        assert inputs["runtime_transients_bytes"] == forward + kept * record, ram
        assert inputs["tight_runtime_transients_bytes"] == forward + 2 * record
        assert inputs["tight_machine_measured"] is True
    # A runtime without the hook (every other family) keeps its inputs.
    untouched = {"total_ram_bytes": 24 * gib}
    _family_reserve_inputs(SimpleNamespace(), untouched, plan_memory)
    assert untouched == {"total_ram_bytes": 24 * gib}
