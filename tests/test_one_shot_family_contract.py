"""#463: one-shot ``mtplx run``/``chat`` on Flash-Next.

The in-process one-shot path applied only the profile defaults, so the
family lanes serve and tune stamp for qwen4_exp never reached it and the
legacy qwen3-next capture walker raised on the hyper-connection layers.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from mtplx.gdn_capture import commit_captured_prefix


def _write(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


FLASH_NEXT = {"model_type": "qwen4_exp", "text_config": {
    "model_type": "qwen4_exp_text", "hidden_size": 2560,
    "num_hidden_layers": 48, "hc_count": 4, "hc_lowrank": 320,
    "indexer_compress_ratio": 4, "linear_num_key_heads": 16,
    "linear_num_value_heads": 48, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "ple_layer_ids": [2], "ngram_size": 3,
    "ngram_vocab_size_base": 20000000, "heads_per_ngram": 8,
    "ple_embed_dim": 2560, "ngram_sidecar": True, "num_experts": 512,
    "num_experts_per_tok": 10, "moe_intermediate_size": 640, "vocab_size": 248320,
}}


def test_legacy_capture_commit_declines_family_native_captures():
    # Flash-Next's fixed-M4 captures are keyed by its own GDN row names and
    # carry no conv tape; the legacy walker used to raise KeyError('conv_states').
    cache = [SimpleNamespace(state=object())]
    captures = {0: {"qkv": object(), "ple_hidden": object(), "capture_start": 0}}
    assert commit_captured_prefix(cache, captures, keep_tokens=2, verified_tokens=4) is False


def test_one_shot_run_resolves_the_serve_contract_for_flash_next(tmp_path, monkeypatch):
    from mtplx.commands import public
    from mtplx.server import openai

    for key in list(os.environ):
        if key.startswith("MTPLX_"):
            monkeypatch.delenv(key)
    model = _write(tmp_path, FLASH_NEXT)
    pack = {"MTPLX_QSA_GATHER_MAX_ROWS": "24"}
    monkeypatch.setattr(openai, "load_runtime_contract", lambda _: (
        SimpleNamespace(runtime_env_overrides=pack), None))

    args = SimpleNamespace(verify_strategy="batched")
    resolved = public._in_process_runtime_env_overrides(args, model, generation_mode="mtp")

    expected = openai._server_runtime_env_overrides(
        SimpleNamespace(model=model, generation_mode="mtp", verify_strategy="batched"), pack)
    assert resolved == expected
    assert resolved["MTPLX_FAMILY_CAPTURE_COMMIT"] == "1"
    assert resolved["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "0"
    assert resolved["MTPLX_QSA_GATHER_MAX_ROWS"] == "24"


def test_one_shot_run_keeps_a_dense_pack_on_its_profile(tmp_path, monkeypatch):
    from mtplx.commands import public
    from mtplx.server import openai

    for key in list(os.environ):
        if key.startswith("MTPLX_"):
            monkeypatch.delenv(key)
    model = _write(tmp_path, {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}})
    monkeypatch.setattr(openai, "load_runtime_contract", lambda _: (None, None))

    resolved = public._in_process_runtime_env_overrides(
        SimpleNamespace(verify_strategy="capture_commit"), model, generation_mode="mtp")

    assert "MTPLX_FAMILY_CAPTURE_COMMIT" not in resolved
    assert "MTPLX_QWEN4_FIXED_M4_VERIFY" not in resolved


def test_missing_contract_is_not_an_error(tmp_path, monkeypatch):
    from mtplx.commands import public
    from mtplx.server import openai

    monkeypatch.setattr(openai, "load_runtime_contract", lambda _: (None, "no contract"))
    resolved = public._in_process_runtime_env_overrides(
        SimpleNamespace(), str(tmp_path), generation_mode="ar")
    assert isinstance(resolved, dict)


def test_interactive_cli_applies_family_contract_before_loading(tmp_path, monkeypatch):
    from mtplx.commands import public
    from mtplx.server import openai
    from mtplx import runtime

    for key in list(os.environ):
        if key.startswith("MTPLX_"):
            monkeypatch.delenv(key)
    model = _write(tmp_path, FLASH_NEXT)
    monkeypatch.setattr(openai, "load_runtime_contract", lambda _: (None, None))
    monkeypatch.setattr(openai, "apply_memory_caps_preflight", lambda **_: None)
    monkeypatch.setattr(public, "_apply_model_default_profile", lambda *_: None)
    monkeypatch.setattr(public, "_model_draft_lm_head_spec", lambda *_: None)
    monkeypatch.setattr(public, "_model_draft_sampler_spec", lambda *_: None)

    class ReachedLoad(Exception):
        pass

    def load(*args, **kwargs):
        assert os.environ["MTPLX_FAMILY_CAPTURE_COMMIT"] == "1"
        assert os.environ["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "0"
        raise ReachedLoad

    monkeypatch.setattr(runtime, "load", load)
    with pytest.raises(ReachedLoad):
        public._quickstart_run_terminal_chat_body(
            SimpleNamespace(profile="turbo", load_mtp=True, generation_mode="mtp",
                            verify_strategy="batched"), runtime_model=model, inspection={})
