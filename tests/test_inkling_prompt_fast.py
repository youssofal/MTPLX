"""The prompt read's fast route switch (`MTPLX_INKLING_PROMPT_FAST`; unset, only the width is on): which parts
it serves, and the chunk width the runtime gives a prompt read with and without it."""

from mtplx.models.inkling.fast.prompt_route import PROMPT_FAST_ENV, prompt_fast
from mtplx.models.inkling.runtime import InklingRuntime


def test_unset_only_the_width_is_on_and_the_switch_serves_only_the_parts_it_names(monkeypatch):
    monkeypatch.delenv(PROMPT_FAST_ENV, raising=False)
    assert prompt_fast("chunk") and not prompt_fast("experts") and not prompt_fast("invariant")
    monkeypatch.setenv(PROMPT_FAST_ENV, "")
    assert prompt_fast("chunk") and not prompt_fast("experts")
    for off in ("0", "off", "no", "false", " Off "):
        monkeypatch.setenv(PROMPT_FAST_ENV, off)
        assert not prompt_fast("chunk") and not prompt_fast("experts")
    for on in ("1", "on", "yes", "true", "all", "ON"):
        monkeypatch.setenv(PROMPT_FAST_ENV, on)
        assert prompt_fast("chunk") and prompt_fast("experts")
    monkeypatch.setenv(PROMPT_FAST_ENV, "experts")
    assert prompt_fast("experts") and not prompt_fast("chunk")
    monkeypatch.setenv(PROMPT_FAST_ENV, " chunk , experts ")
    assert prompt_fast("chunk") and prompt_fast("experts")


def test_the_fast_chunk_is_2048_rows_where_the_prompt_and_the_mac_allow_and_a_named_width_still_wins(monkeypatch):
    from mtplx.generation import prefill_chunk_size_override
    from mtplx.models.inkling.fast import prompt_route as PR

    for name in ("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "off")  # the stock attention's widths (it is on by default)
    monkeypatch.setattr(PR, "_headroom", {"bytes": 64 * 1024**3})
    rt = object.__new__(InklingRuntime)
    monkeypatch.setenv(PROMPT_FAST_ENV, "experts")
    assert rt.prompt_chunk_rows(16_000) == 512
    monkeypatch.setenv(PROMPT_FAST_ENV, "chunk")
    assert rt.prompt_chunk_rows(16_000) == PR.WIDE_ROWS == 2048
    assert rt.prompt_chunk_rows(None) == 2048
    # a prompt past the attention planes' fence keeps 512 rows
    assert rt.prompt_chunk_rows(PR.WIDE_PROMPT_TOKENS + 1) == 512
    # a Mac without the headroom keeps 512 rows
    monkeypatch.setattr(PR, "_headroom", {"bytes": PR.WIDE_HEADROOM_BYTES - 1})
    assert rt.prompt_chunk_rows(16_000) == 512
    monkeypatch.setattr(PR, "_headroom", {"bytes": None})
    assert rt.prompt_chunk_rows(16_000) == 512
    monkeypatch.setattr(PR, "_headroom", {"bytes": 64 * 1024**3})
    with prefill_chunk_size_override(1024):
        assert rt.prompt_chunk_rows(16_000) == 1024
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "4096")  # the user's own width
    assert rt.prompt_chunk_rows(16_000) == 4096


def test_the_headroom_is_read_once_a_process(monkeypatch):
    from mtplx.models.inkling.fast import prompt_route as PR

    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "off")  # the stock attention's widths (it is on by default)
    monkeypatch.setattr(PR, "_headroom", {})
    first = PR.gpu_headroom_bytes()
    assert PR.gpu_headroom_bytes() == first
    monkeypatch.setenv(PROMPT_FAST_ENV, "off")
    assert PR.wide_rows(4_000, headroom_bytes=PR.WIDE_HEADROOM_BYTES) is None  # switch off
    monkeypatch.delenv(PROMPT_FAST_ENV, raising=False)                        # unset: the width is on
    assert PR.wide_rows(4_000, headroom_bytes=PR.WIDE_HEADROOM_BYTES) == 2048
    monkeypatch.setenv(PROMPT_FAST_ENV, "chunk")
    assert PR.wide_rows(4_000, headroom_bytes=PR.WIDE_HEADROOM_BYTES) == 2048
    assert PR.wide_rows(4_000, headroom_bytes=PR.WIDE_HEADROOM_BYTES - 1) is None


def test_generation_reads_the_width_for_this_prompt_as_the_admission_priced_it(monkeypatch):
    """The request wrapper hands the prompt's length to the model's width, so a prompt past the fence reads at
    512 rows in generation exactly as the admission priced it (both ask ``prompt_chunk_rows(len(prompt))``)."""
    from types import SimpleNamespace

    from mtplx import generation
    from mtplx.models.inkling.fast import prompt_route as PR

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    for name in ("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(PROMPT_FAST_ENV, "chunk")
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "off")  # the stock attention's widths (it is on by default)
    monkeypatch.setattr(PR, "_headroom", {"bytes": 64 * 1024**3})
    rt = object.__new__(InklingRuntime)
    rt.backend_id, rt.model_path = "inkling2_loop", None
    seen = []

    @generation._with_inkling_attention_receipt
    def read(rt, prompt_ids):
        seen.append(generation._prefill_chunk_size())
        return SimpleNamespace(stats=SimpleNamespace())

    read(rt, [0] * 4_000)
    read(rt, [0] * (PR.WIDE_PROMPT_TOKENS + 1))
    assert seen == [2048, 512]
    assert generation.prefill_forward_widths(rt, 4_000, None) == [2048]
    assert generation.prefill_forward_widths(rt, PR.WIDE_PROMPT_TOKENS + 1, None) == [512]


def test_with_the_one_launch_attention_the_width_is_4096_and_the_fence_lifts(monkeypatch):
    from mtplx.models.inkling.fast import prompt_flash as F
    from mtplx.models.inkling.fast import prompt_route as PR

    monkeypatch.setenv(PROMPT_FAST_ENV, "chunk")
    monkeypatch.setattr(PR, "_headroom", {"bytes": 64 * 1024**3})
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)   # on by default
    monkeypatch.setattr(F, "tensor_units", lambda: True)              # a Mac it runs on (M5)
    monkeypatch.setattr(F, "library_text", lambda: ("text", None))
    long_prompt = max(F.SHORT_PROMPT_WIDE, 4_000)
    assert PR.flash_attention_on() and PR.flash_attention_on(long_prompt)
    assert PR.wide_rows(long_prompt) == PR.WIDE_ROWS_FLASH
    assert PR.wide_rows(PR.WIDE_PROMPT_TOKENS * 4) == PR.WIDE_ROWS_FLASH  # no fence
    # a prompt below the crossover reads on the stock attention: its width and fence
    if F.SHORT_PROMPT_WIDE > 1:
        assert not PR.flash_attention_on(F.SHORT_PROMPT_WIDE - 1) and PR.wide_rows(F.SHORT_PROMPT_WIDE - 1) == 2048
    monkeypatch.setattr(PR, "_headroom", {"bytes": PR.WIDE_HEADROOM_BYTES - 1})
    assert PR.wide_rows(long_prompt) is None
    # switched off: the stock attention's width and fence
    monkeypatch.setattr(PR, "_headroom", {"bytes": 64 * 1024**3})
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_FLASH", "off")
    assert not PR.flash_attention_on() and PR.wide_rows(long_prompt) == 2048


def test_without_the_tensor_units_a_long_prompt_keeps_the_stock_attentions_width_and_price(monkeypatch):
    """M1 to M4 cannot run the one-launch attention, so a long prompt keeps the stock attention's width rule (its
    bias and mask planes grow with width times context) and is priced with those planes; it does not take the
    2,048 rows the one launch reads at any length."""
    from mtplx.models.inkling.fast import prompt_flash as F
    from mtplx.models.inkling.fast import prompt_route as PR

    monkeypatch.setenv(PROMPT_FAST_ENV, "chunk")
    monkeypatch.setattr(PR, "_headroom", {"bytes": 64 * 1024**3})
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_FLASH", raising=False)
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")       # the product's M1 to M4 rehearsal switch
    long_prompt = PR.WIDE_PROMPT_TOKENS * 4
    assert not F.runs_here() and not PR.flash_attention_on(long_prompt)
    assert PR.wide_rows(long_prompt) is None                          # 512 rows past the stock fence
    assert PR.wide_rows(PR.WIDE_PROMPT_TOKENS) == PR.WIDE_ROWS        # 2,048 inside it
    assert PR.forward_bytes(2048, long_prompt) == PR.forward_bytes(2048, long_prompt, flash=False)


def test_the_stage_counts_as_running_only_where_it_can(monkeypatch):
    import mlx.core as mx

    from mtplx.models.inkling.fast import prompt_flash as F

    for name in ("MTPLX_INKLING_PROMPT_FLASH", "MTPLX_INKLING_ATTENTION", "MTPLX_INKLING_PROMPT_ATTENTION",
                 "MTPLX_FORCE_GPU_FAMILY_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(F, "tensor_units", lambda: True)
    monkeypatch.setattr(F, "library_text", lambda: ("text", None))
    gpu = mx.default_device() == mx.gpu and mx.metal.is_available()
    assert F.runs_here() == gpu
    for name, value in (("MTPLX_INKLING_PROMPT_FLASH", "off"), ("MTPLX_INKLING_ATTENTION", "exact"),
                        ("MTPLX_INKLING_PROMPT_ATTENTION", "stock")):
        monkeypatch.setenv(name, value)
        assert not F.runs_here(), name
        monkeypatch.delenv(name)
    monkeypatch.setattr(F, "tensor_units", lambda: False)
    assert not F.runs_here()
    monkeypatch.setattr(F, "tensor_units", lambda: True)
    monkeypatch.setattr(F, "library_text", lambda: (None, "no headers"))
    assert not F.runs_here()
