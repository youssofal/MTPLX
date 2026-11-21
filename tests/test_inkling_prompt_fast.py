"""The prompt read's fast route switch (`MTPLX_INKLING_PROMPT_FAST`, off by default): which parts it
serves, and the chunk width the runtime gives a prompt read with and without it."""

from mtplx.models.inkling.fast.prompt_route import PROMPT_FAST_ENV, prompt_fast
from mtplx.models.inkling.runtime import InklingRuntime


def test_the_switch_is_off_unless_set_and_serves_only_the_parts_it_names(monkeypatch):
    monkeypatch.delenv(PROMPT_FAST_ENV, raising=False)
    assert not prompt_fast("chunk") and not prompt_fast("experts")
    for off in ("", "0", "off", "no", "false", " Off "):
        monkeypatch.setenv(PROMPT_FAST_ENV, off)
        assert not prompt_fast("chunk") and not prompt_fast("experts")
    for on in ("1", "on", "yes", "true", "all", "ON"):
        monkeypatch.setenv(PROMPT_FAST_ENV, on)
        assert prompt_fast("chunk") and prompt_fast("experts")
    monkeypatch.setenv(PROMPT_FAST_ENV, "experts")
    assert prompt_fast("experts") and not prompt_fast("chunk")
    monkeypatch.setenv(PROMPT_FAST_ENV, " chunk , experts ")
    assert prompt_fast("chunk") and prompt_fast("experts")


def test_the_fast_chunk_is_4096_rows_and_a_named_width_still_wins(monkeypatch):
    from mtplx.generation import prefill_chunk_size_override

    for name in ("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    rt = object.__new__(InklingRuntime)
    monkeypatch.setenv(PROMPT_FAST_ENV, "experts")
    assert rt.prompt_chunk_rows(30_000) == 512
    monkeypatch.setenv(PROMPT_FAST_ENV, "chunk")
    assert rt.prompt_chunk_rows(30_000) == 4096
    with prefill_chunk_size_override(1024):
        assert rt.prompt_chunk_rows(30_000) == 1024
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")  # the user's own width
    assert rt.prompt_chunk_rows(30_000) == 2048
