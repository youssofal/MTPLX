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
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FAST", "off")  # the read's own width rule (on by default): its tests
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")  # what every profile stamps
    rt = object.__new__(InklingRuntime)
    assert rt.prompt_chunk_rows(30_000) == 512
    assert _served_prefill_chunk_auto(SimpleNamespace(), rt) == 512  # what the app's settings show
    assert prefill_forward_widths(rt, 30_000, None) == [512]
    assert prefill_forward_widths(rt, 30_000, 4096) == [4096, 512]
    assert prefill_forward_widths(rt, 30_000, 256) == [256]
    assert _admission_narrow_widths(rt, [512]) == []
    # Where the read is wider (2,048 rows on a Mac with the GPU headroom), the admission may narrow it down to the
    # 512 rows the plan reserves at the window's end, not only to 1,024 as for a family that does not price itself.
    assert _admission_narrow_widths(rt, [2048]) == [1024, 512]
    assert _admission_narrow_widths(rt, [1024]) == [512]
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


def test_unset_the_read_width_is_2048_rows_on_a_mac_with_the_headroom_and_512_without(monkeypatch):
    from mtplx.generation import prefill_forward_widths
    from mtplx.models.inkling.fast import prompt_route as PR

    for name in ("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "MTPLX_INKLING_PROMPT_FAST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    rt = object.__new__(InklingRuntime)
    monkeypatch.setattr(PR, "_headroom", {"bytes": 64 * 1024**3})
    assert rt.prompt_chunk_rows(30_000) == 2048
    assert prefill_forward_widths(rt, 30_000, None) == [2048]   # the admission prices the width the read takes
    monkeypatch.setattr(PR, "_headroom", {"bytes": PR.WIDE_HEADROOM_BYTES - 1})
    assert rt.prompt_chunk_rows(30_000) == 512                  # a Mac without the headroom keeps 512 rows


def test_the_request_wrapper_installs_the_models_width_and_keeps_a_named_one(monkeypatch):
    from mtplx import generation

    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FAST", "off")  # the 512-row width, as the model gives it
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



def test_the_plan_leaves_the_banks_rounding_to_the_request():
    """The plan's window is a whole number of bank steps and an answer stops at the window, so a
    request inside it never holds more bank rows than the window has tokens: the window's fixed part
    leaves the rounding out (134 MB, 3,276 tokens of every window). A request below the window still
    rounds up inside it, and the admission charges that."""
    from mtplx.memory_plan import CONTEXT_ALIGN_TOKENS
    from mtplx.models.inkling.fast.state import BANK_STEP

    assert CONTEXT_ALIGN_TOKENS % BANK_STEP == 0
    rt, row = _decode_runtime(pool_bytes=403 << 20)
    rt.engine_pool_bound = 403 << 20
    rt.load_extra_bytes = 172 << 20
    rings, rounding = 1024 * 40 * row, 8 * 4096 * row
    assert InklingRuntime.window_cache_bytes(rt) == rings + rounding + (403 << 20) + (172 << 20)
    assert InklingRuntime.plan_fixed_bytes(rt) == rings + (403 << 20) + (172 << 20)
    rt.engine_pool_now = lambda: 403 << 20
    assert InklingRuntime.admission_fixed_bytes(rt) == rings + rounding  # a request pays its own rounding
    # The window's banks at its end: a prompt and answer that fill it round up to the window itself.
    for window in (20480, 24576, 28672, 32768):
        for used in (window - 4095, window - 1, window):
            assert -(-used // BANK_STEP) * BANK_STEP <= window

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
    # The widest panel: aligned to warmed 8-row tiles where wide panels are exact, else 8.
    original = runtime_module.wide_rows_exact
    try:
        runtime_module.wide_rows_exact = lambda model, spec: True
        assert runtime_module.wide_copy_widths(None) == (16, 24)
        assert widest_panel(None) == 24
        runtime_module.wide_rows_exact = lambda model, spec: False
        assert runtime_module.wide_copy_widths(None) == ()
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
    assert InklingRuntime.prefill_working_bytes(rt, few=True, tight=True) == forward + record  # under the tight rule
    assert InklingRuntime.prefill_working_bytes(rt, 1024, tight=True) == 2 * forward + record
    # Bare Speed (15.99 GB of weights with its vision tower) on a 24 GB Mac, at the server's fixed term as served
    # (21 November: rings at the full window 180 MB, bank rounding 134 MB, the pool at ready 322 MB plus its
    # growth past the first capacity step, the load's 33 MB and the warm-up's arrays): the flat reserve refuses
    # it, one 2 GiB reserve for both rules leaves the 4,096-token floor, the forward and one checkpoint under the
    # tight rule admit the rule's shared ceiling (20,480; its own room is 28,672 tokens).
    fixed = 179_999_232 + 134_217_728 + 322_495_208 + (96 << 20) + 32_996_786 + InklingRuntime.WARMUP_ARRAYS_BYTES
    args = dict(total_ram_bytes=24 * gib, model_weights_bytes=15_993_374_536, kv_bytes_per_token=32768,
                aux_bytes_per_token=4096, attention_kv_copy_bytes_per_token=4096, fixed_cache_bytes=fixed,
                tight_machine_measured=True)
    assert not plan_memory(**{**args, "tight_machine_measured": False}).model_fits
    one = plan_memory(**args, runtime_transients_bytes=2 * gib)
    assert not one.model_fits and one.context_window_fit == 4096
    split = plan_memory(**args, runtime_transients_bytes=InklingRuntime.prefill_working_bytes(rt),
                        tight_runtime_transients_bytes=InklingRuntime.prefill_working_bytes(rt, few=True, tight=True))
    assert split.model_fits and split.tight_machine
    assert split.runtime_transients_bytes == forward + record + 256 * 2**20
    assert split.context_window_fit == 20480
    # As served (the plan's fixed term without the banks' rounding, the family's own cap): the forward and one
    # checkpoint leave 32,768 tokens on a 24 GB Mac; a second checkpoint would leave 28,672.
    served = {**args, "fixed_cache_bytes": fixed - 134_217_728, "tight_window_tokens": InklingRuntime.TIGHT_WINDOW_TOKENS}
    for kept, window in ((1, 32768), (2, 28672)):
        plan = plan_memory(**served, runtime_transients_bytes=forward + 8 * record,
                           tight_runtime_transients_bytes=forward + kept * record)
        assert plan.tight_machine and plan.context_window_fit == window, kept
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



def test_a_measured_family_takes_its_own_tight_window_up_to_its_cap_and_it_only_grows_with_the_budget():
    """The tight rule's shared ceiling (one block plus the waived floor: 20,480 tokens here) stops a measured
    family's window short of what its measured reserve leaves on a 24 GB Mac; ``tight_window_tokens`` lifts
    it to the family's cap, and nowhere else does the plan move."""
    gib = 2**30
    record = 179_999_232
    args = dict(total_ram_bytes=24 * gib, model_weights_bytes=15_993_374_536, kv_bytes_per_token=32768,
                aux_bytes_per_token=4096, attention_kv_copy_bytes_per_token=4096, fixed_cache_bytes=895_000_000,
                tight_machine_measured=True, model_max_context=1_048_576)
    roomy_reserve = gib // 2 + 2 * record  # a forward of 0.5 GiB and two checkpoints
    reserves = dict(runtime_transients_bytes=roomy_reserve, tight_runtime_transients_bytes=roomy_reserve)
    shared = plan_memory(**args, **reserves)
    assert shared.tight_machine and shared.context_window_fit == 20480
    own = plan_memory(**args, **reserves, tight_window_tokens=32768)
    assert own.tight_machine and own.context_window_fit == 28672  # what the reserve leaves, under the cap
    small = gib // 8 + record  # a lighter reserve whose own window passes the cap
    capped = plan_memory(**args, runtime_transients_bytes=small, tight_runtime_transients_bytes=small,
                         tight_window_tokens=32768)
    assert capped.context_window_fit == 32768
    # An unmeasured pack keeps the shared ceiling whatever it is handed.
    assert plan_memory(**{**args, "tight_machine_measured": False, "model_weights_bytes": 8 * gib}, **reserves,
                       tight_window_tokens=32768).context_window_fit == plan_memory(
        **{**args, "tight_machine_measured": False, "model_weights_bytes": 8 * gib}, **reserves).context_window_fit
    # From 32 GB the floor-funded window is far past the cap: the plan is the same with or without it.
    for ram in (32, 36, 48, 128):
        a = plan_memory(**{**args, "total_ram_bytes": ram * gib}, **reserves)
        b = plan_memory(**{**args, "total_ram_bytes": ram * gib}, **reserves, tight_window_tokens=32768)
        assert (a.context_window_fit, a.tight_machine) == (b.context_window_fit, b.tight_machine), ram
        assert not b.tight_machine, ram
    # The window only grows with the budget, through the cap and past it (256 MiB steps, 20 to 40 GB).
    last = 0
    for step in range(80, 161):
        plan = plan_memory(**{**args, "total_ram_bytes": step * gib // 4}, **reserves, tight_window_tokens=32768)
        fit = plan.context_window_fit if plan.model_fits else 0
        assert fit >= last, (step / 4, fit, last)
        last = fit

def test_admission_charges_a_family_its_measured_prefill_working_set():
    from mtplx.server.openai import _admission_scratch_bytes

    rt = _checkpoint_runtime(_CATALOG_TEXT, _CATALOG_DRAFT)
    forward, record = InklingRuntime.FORWARD_WORKING_BYTES, 179_999_232
    geometry = _AdmissionGeometry(40960, 40960, 0, 3 * 2**30, 4096)
    weights = 15_993_374_536
    for plan, kept in ((None, 8), (SimpleNamespace(tight_machine=False), 8), (SimpleNamespace(tight_machine=True), 1),
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


def _catalog_runtime():
    """The catalog geometry with the model fields the banks' rounding reads; no pool, no load arrays."""
    import mlx.core as mx

    from mtplx.models.inkling.layout import layer_visits

    rt = _checkpoint_runtime(_CATALOG_TEXT, _CATALOG_DRAFT)
    local = set(_CATALOG_TEXT["local_layer_ids"])
    rt.model = SimpleNamespace(
        config=_CATALOG_TEXT,
        visit_order=layer_visits(_CATALOG_TEXT),
        layers=[SimpleNamespace(sliding=i in local) for i in range(_CATALOG_TEXT["num_hidden_layers"])],
        embed_tokens=SimpleNamespace(weight=SimpleNamespace(dtype=mx.bfloat16)),
    )
    rt.engine_pool_bytes = 0
    rt.load_extra_bytes = 0
    rt.prompt_chunk_rows = lambda prompt_tokens=None: 512
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


def test_a_tight_mac_keeps_one_restore_checkpoint_one_whose_window_memory_sets_two_and_a_roomy_one_eight():
    from mtplx.checkpoint_anchors import AnchorPlan, retain_checkpoints
    from mtplx.engine_session import EngineSessionManager

    rt = _checkpoint_runtime(_CATALOG_TEXT, _CATALOG_DRAFT)
    record = InklingRuntime.checkpoint_record_bytes(rt)
    assert InklingRuntime.checkpoint_budget_bytes(rt) is None
    assert InklingRuntime.checkpoint_budget_bytes(rt, SimpleNamespace(tight_machine=False)) is None
    roomy = SimpleNamespace(tight_machine=False, usable_bytes=96 * 2**30, model_weights_bytes=15_993_374_536)
    assert InklingRuntime.checkpoint_budget_bytes(rt, roomy) is None
    one = InklingRuntime.checkpoint_budget_bytes(rt, SimpleNamespace(tight_machine=True))
    assert record < one < 2 * record
    tier36 = SimpleNamespace(tight_machine=False, usable_bytes=27 * 2**30, model_weights_bytes=15_993_374_536)
    budget = InklingRuntime.checkpoint_budget_bytes(rt, tier36)
    assert 2 * record < budget < 3 * record
    # A cold read of a 16,430-token prompt: every 512-row chunk edge, the tail ladder's top rung 64 tokens
    # before the prompt end, and the prompt end.
    records = [(p, SimpleNamespace(nbytes=record), None) for p in (*range(512, 16_129, 512), 16_365, 16_429)]
    kept = lambda plan: [r[0] for r in retain_checkpoints(records, plan)]  # noqa: E731
    few = AnchorPlan(budget_bytes=budget, prompt_end=16_429)
    assert kept(few) == [16_365, 16_429]  # the prompt-end anchor and the newest
    # A read that resumed at 4,096 keeps that restore point over the newest.
    assert kept(AnchorPlan(budget_bytes=budget, prompt_end=16_429, restore_point=4096)) == [4096, 16_365]
    # Under the tight rule, the prompt-end anchor alone, whatever else the read passed.
    assert kept(AnchorPlan(budget_bytes=one, prompt_end=16_429)) == [16_365]
    assert kept(AnchorPlan(budget_bytes=one, prompt_end=16_429, restore_point=4096)) == [16_365]
    assert len(kept(AnchorPlan(prompt_end=16_429))) == 8  # no budget: the shared eight
    # The server hands the budget to the bank every prefill reads it from.
    assert EngineSessionManager(checkpoint_budget_bytes=budget).bank.checkpoint_budget_bytes == budget
    assert EngineSessionManager().bank.checkpoint_budget_bytes is None


def test_the_server_plans_with_the_checkpoints_the_reads_keep():
    """The server's plan inputs for Inkling: the forward plus two checkpoints wherever memory sets the window
    (24, 32 and 36 GB with Bare Speed), plus eight from 48 GB; the tight rule always has one. The next turn's
    reserve follows the same checkpoints: what the conversation keeps is what its next turn is billed again."""
    from mtplx.server.openai import _family_reserve_inputs

    gib = 2**30
    rt = _catalog_runtime()
    forward, record = InklingRuntime.FORWARD_WORKING_BYTES, 179_999_232
    held = 179_999_232 + 134_217_728 + 4095 * 36864  # rings, banks' rounding, one bank step less a row
    for ram, kept in ((24, 2), (32, 2), (36, 2), (48, 8), (128, 8)):
        inputs = dict(total_ram_bytes=ram * gib, model_weights_bytes=15_993_374_536, kv_bytes_per_token=32768,
                      aux_bytes_per_token=4096, attention_kv_copy_bytes_per_token=4096,
                      fixed_cache_bytes=1_088_788_347)
        _family_reserve_inputs(rt, inputs, plan_memory)
        assert inputs["runtime_transients_bytes"] == forward + kept * record, ram
        assert inputs["tight_runtime_transients_bytes"] == forward + record
        assert inputs["tight_machine_measured"] is True
        assert inputs["tight_window_tokens"] == InklingRuntime.TIGHT_WINDOW_TOKENS == 32768
        assert inputs["turn_reserve_bytes"] == held + kept * record, ram
        assert inputs["tight_turn_reserve_bytes"] == held + record
    # A runtime without the hook (every other family) keeps its inputs.
    untouched = {"total_ram_bytes": 24 * gib}
    _family_reserve_inputs(SimpleNamespace(), untouched, plan_memory)
    assert untouched == {"total_ram_bytes": 24 * gib}


def test_every_conversation_inside_the_window_can_take_its_next_turn():
    """A turn that continues its conversation in place is billed state the conversation already holds: the
    windows term (rings, the banks' rounding), a second set of checkpoints, and new rows its last bank step may
    already hold. On a simulated 24 GB budget (22 November, `mtplx serve`) a 20-token turn after a 32,233-token
    conversation was 111 MB over the engine limit and refused after its stream opened. The plan now keeps that
    turn's bill (``turn_reserve_bytes``), so the window is one every conversation in it can take its next turn in."""
    from mtplx.server.openai import _family_reserve_inputs

    gib = 2**30
    rt = _catalog_runtime()
    record, live, copy = 179_999_232, 36864, 4096
    geometry = _AdmissionGeometry(live, live, 0, 3 * gib, 15_993_374_536, resident_bytes_per_token=live,
                                  resident_fixed_bytes=InklingRuntime.window_cache_bytes(rt),
                                  prefill_fixed_bytes=InklingRuntime.window_cache_bytes(rt),
                                  prefill_windows=lambda prompt, new: InklingRuntime.prefill_window_bytes(rt, prompt, new),
                                  aux_bytes_per_token=4096, attention_kv_copy_bytes_per_token=copy)
    window = 24576
    for few, tight in ((True, True), (True, False), (False, False)):
        kept = InklingRuntime.checkpoints_kept(rt, few, tight)
        working = InklingRuntime.prefill_working_bytes(rt, few=few, tight=tight)
        # What the plan commits at the window: its fixed part, the per-token price, the read's reserve.
        committed = InklingRuntime.plan_fixed_bytes(rt) + window * (live + copy) + working

        def projected(conversation, new):
            # MLX active once the conversation ends (its banks in whole steps, its rings, its checkpoints), plus
            # the admission's bill for a turn that continues it in place.
            held = -(-conversation // 4096) * 4096 * live + InklingRuntime.ring_bytes(rt) + kept * record
            growth = _admission_growth(geometry, prompt_tokens=conversation + new, reused_tokens=conversation,
                                       restore_copies_prefix=False, layout="contiguous_dense_decode",
                                       source_layout="contiguous_dense_decode", output_tokens=64, publish=True,
                                       scratch_bytes=InklingRuntime.prefill_working_bytes(rt, 512, few=few, tight=tight),
                                       publish_bytes=record)
            return held + growth["growth_bytes"]

        worst = projected(window - 4095, 4094)  # just past a bank step, then a tool result that fills the window
        short = projected(window - 21, 20)
        reserve = InklingRuntime.turn_reserve_bytes(rt, few=few, tight=tight)
        assert short < worst > committed, (few, tight)  # the read's reserve alone does not cover the turn
        assert committed + reserve - worst == live + copy, (few, tight)  # the reserve covers it, to one token
    # The windows this gives the served inputs (the plan's fixed part as served on Bare Speed): 24 GB stops at
    # 24,576 (32,768 before), 48 and 64 GB a few percent lower, 32, 36, 96 and 128 GB unchanged.
    fixed = 179_999_232 + 322_495_208 + (96 << 20) + 32_996_786 + InklingRuntime.WARMUP_ARRAYS_BYTES
    expected = {24: (32768, 24576), 32: (163840, 163840), 36: (245760, 245760), 48: (454656, 434176),
                64: (770048, 749568), 96: (1048576, 1048576), 128: (1048576, 1048576)}
    last = 0
    for ram, (before, after) in expected.items():
        inputs = dict(total_ram_bytes=ram * gib, model_weights_bytes=15_993_374_536, kv_bytes_per_token=32768,
                      aux_bytes_per_token=4096, attention_kv_copy_bytes_per_token=4096, fixed_cache_bytes=fixed,
                      model_max_context=1_048_576)
        _family_reserve_inputs(rt, inputs, plan_memory)
        plan = plan_memory(**inputs)
        old = plan_memory(**{k: v for k, v in inputs.items() if "turn_reserve" not in k})
        assert (old.context_window_fit, plan.context_window_fit) == (before, after), ram
        assert plan.tight_machine is (ram == 24), ram
        assert any("next turn" in note for note in plan.notes) is (after < before), ram
        assert plan.context_window_fit >= last, ram  # the window still grows with the Mac
        last = plan.context_window_fit
    # Every other family plans as it did: no turn reserve, no change.
    for ram in (16, 24, 32, 48, 128):
        args = dict(total_ram_bytes=ram * gib, model_weights_bytes=8 * gib, kv_bytes_per_token=65536,
                    model_max_context=262144)
        assert plan_memory(**args).to_dict() == plan_memory(**args, turn_reserve_bytes=0).to_dict(), ram


def _tiny_vision_pack(path):
    """A pack directory holding a one-block vision tower and one text tensor beside it in the same shard."""
    import json

    import mlx.core as mx

    tensors = {
        "model.visual.backbone.layers.0.ln_1.weight": mx.arange(8, dtype=mx.bfloat16),
        "model.visual.backbone.layers.0.ln_1.bias": mx.ones((8,), dtype=mx.bfloat16),
        "model.visual.backbone.embeddings.patch_embedding.weight": mx.arange(32, dtype=mx.bfloat16).reshape(4, 8),
        "model.visual.backbone.embeddings.position_embedding": mx.ones((4, 8), dtype=mx.bfloat16),
        "model.visual.backbone.embeddings.class_embedding": mx.ones((8,), dtype=mx.bfloat16),
        "model.visual.backbone.embeddings.cls_position_embedding": mx.zeros((8,), dtype=mx.bfloat16),
        "model.visual.backbone.ln_pre.weight": mx.ones((8,), dtype=mx.bfloat16),
        "model.visual.backbone.ln_pre.bias": mx.zeros((8,), dtype=mx.bfloat16),
        "model.visual.proj_in.weight": mx.ones((8, 8), dtype=mx.bfloat16),
        "model.visual.proj_out.weight": mx.ones((8, 8), dtype=mx.bfloat16),
        "model.llm.embed_tokens.weight": mx.ones((16, 8), dtype=mx.bfloat16),
    }
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(path / shard), tensors)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: shard for k in tensors}}))
    vision_config = {"num_hidden_layers": 1, "num_attention_heads": 1, "layer_norm_eps": 1e-6}
    (path / "config.json").write_text(json.dumps({"model_type": "inkling2_loop", "vision_config": vision_config}))
    return tensors, vision_config


def test_an_idle_vision_tower_gives_its_memory_back_and_maps_the_same_weights_again(tmp_path):
    """The memory guard's last step before refusing (`x1-fix-24gb-turns2`: a cold 24K text prompt after an image
    turn, refused 0.20 GiB short beside the idle 0.59 GiB tower). Bit-identical: the new tower maps the pack's
    bytes."""
    import mlx.core as mx

    from mtplx.models.inkling.model import load_vision_weights
    from mtplx.models.inkling.vision import VisionTower

    tensors, vision_config = _tiny_vision_pack(tmp_path)

    class Runtime:
        idle_vision_bytes = InklingRuntime.idle_vision_bytes
        release_idle_vision = InklingRuntime.release_idle_vision

    runtime = Runtime()
    runtime.model_path = tmp_path
    runtime.vision_tower = VisionTower(vision_config, load_vision_weights(tmp_path))
    # Never ran an image: nothing read in, nothing to give back.
    assert runtime.idle_vision_bytes() == 0
    assert runtime.release_idle_vision() == 0
    first = runtime.vision_tower
    first.resident = True  # what encode() sets as an image runs
    # The tower's own arrays: the class and class-position embeddings are one array once it is built.
    held = sum(int(v.nbytes) for k, v in tensors.items() if k.startswith("model.visual.")) - int(
        tensors["model.visual.backbone.embeddings.cls_position_embedding"].nbytes)
    assert first.nbytes == held
    assert runtime.idle_vision_bytes() == held
    assert runtime.release_idle_vision() == held
    second = runtime.vision_tower
    assert second is not first and second.resident is False
    assert runtime.idle_vision_bytes() == 0 and runtime.vision_tower_releases == 1
    for name in ("patch", "pos", "cls", "norm_weight", "norm_bias", "project_in", "project_out"):
        assert mx.array_equal(getattr(first, name), getattr(second, name)).item()
    assert sorted(first.blocks[0].w) == sorted(second.blocks[0].w)
    for key, value in first.blocks[0].w.items():
        assert mx.array_equal(value, second.blocks[0].w[key]).item()


def test_a_vision_tower_stored_below_its_source_precision_is_never_remapped(tmp_path):
    import json

    import pytest

    from mtplx.models.inkling.model import load_vision_weights

    _tiny_vision_pack(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["mtplx_inkling_precision"] = {"model.visual.proj_in.weight": {"bits": 4, "group_size": 64}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_vision_weights(tmp_path)
