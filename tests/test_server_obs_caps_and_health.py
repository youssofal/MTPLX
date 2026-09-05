"""Memory-caps preflight on every entry (F7) + /health degradation truth (F23).

F7 — benchmark/ladder/terminal-chat paths used to load models with NO Metal
allocator caps (the #261 102.6GB-at-262k headline class). Every in-process
entry now runs ``apply_memory_caps_preflight``: the exact serve-path caps
(same function, same values) plus a context-bound refusal with a clear
message instead of silently benchmarking past the model's trained window.

F23 — the benchmark harness gate reads /health; it now carries an ADDITIVE
``degradation`` block (compiled_verify mode/permanent_eager/reason, profile
env overrides, NAX availability + bail counters) built from DEFENSIVE reads
with honest "unknown" defaults, so it never lies or crashes regardless of
which enrichment lane lands first.

Engine always monkeypatched — no model loads, CPU only.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app

from test_server_openai import _fake_state  # noqa: E402 - shared fixtures


def _stub_caps(monkeypatch):
    calls: list[dict] = []

    def fake_caps(**kwargs):
        calls.append(kwargs)
        return {"applied": True, "source": "serve_path_stub"}

    monkeypatch.setattr(openai, "_apply_metal_memory_caps", fake_caps)
    return calls


def _counting_preflight(monkeypatch):
    calls: list[dict] = []

    def fake_preflight(**kwargs):
        calls.append(kwargs)
        return {"entry": kwargs.get("entry"), "stub": True}

    monkeypatch.setattr(openai, "apply_memory_caps_preflight", fake_preflight)
    return calls


def _model_dir_with_window(tmp_path, window: int):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": int(window)})
    )
    return model_dir


# --- the shared preflight itself --------------------------------------------


def test_preflight_applies_the_serve_path_cap_function(monkeypatch):
    calls = _stub_caps(monkeypatch)

    outcome = openai.apply_memory_caps_preflight(entry="unit.test")

    assert len(calls) == 1
    assert outcome["entry"] == "unit.test"
    assert outcome["metal_memory_caps"]["source"] == "serve_path_stub"


def test_preflight_refuses_contexts_beyond_model_window(tmp_path, monkeypatch):
    _stub_caps(monkeypatch)
    model_dir = _model_dir_with_window(tmp_path, 1024)

    with pytest.raises(ValueError) as excinfo:
        openai.apply_memory_caps_preflight(
            entry="bench.prefill_ladder",
            model=str(model_dir),
            contexts=[512, 2048],
        )

    message = str(excinfo.value)
    assert "bench.prefill_ladder" in message
    assert "2,048" in message
    assert "1,024" in message
    assert "exceeds" in message


def test_preflight_allows_contexts_within_model_window(tmp_path, monkeypatch):
    _stub_caps(monkeypatch)
    model_dir = _model_dir_with_window(tmp_path, 1024)

    outcome = openai.apply_memory_caps_preflight(
        entry="bench.prefill_ladder",
        model=str(model_dir),
        contexts=[512, 1024],
    )

    assert outcome["model_context_window"] == 1024
    assert outcome["requested_contexts"] == [512, 1024]


# --- entry: prefill ladder --------------------------------------------------


def _ladder_args(model: str, contexts: str) -> Namespace:
    return Namespace(
        contexts=contexts,
        full=False,
        profile="sustained",
        model=model,
        generation_mode="mtp",
        max_tokens=2,
        dry_run=False,
        prompt_style="coding-agent",
        prompt_format="chat",
        prefill_layout="profile",
        prompt_tail=None,
        prompt_tail_file=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        draft_temperature=None,
        draft_top_p=None,
        draft_top_k=None,
        speculative_depth=3,
        seed=0,
        fanmax=False,
        disable_thinking=True,
        enable_thinking=False,
    )


def test_prefill_ladder_refuses_over_window_contexts_before_load(
    tmp_path, monkeypatch
):
    import mtplx.runtime as runtime
    from mtplx.prefill_bench import run_prefill_ladder

    _stub_caps(monkeypatch)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(
        runtime,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "over-window ladder must refuse BEFORE loading the model"
        ),
    )
    model_dir = _model_dir_with_window(tmp_path, 1024)

    with pytest.raises(ValueError, match="exceeds"):
        run_prefill_ladder(_ladder_args(str(model_dir), "512,2k"))


def test_prefill_ladder_records_preflight_receipt(tmp_path, monkeypatch):
    import mtplx.generation as generation
    import mtplx.runtime as runtime
    from mtplx.prefill_bench import run_prefill_ladder

    preflight_calls = _counting_preflight(monkeypatch)
    monkeypatch.setattr(os, "environ", os.environ.copy())

    class _CharTokenizer:
        def encode(self, text):
            return [ord(ch) for ch in text]

        def decode(self, ids):
            return "".join(chr(int(token)) for token in ids)

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs
        ):
            text = "".join(str(m["content"]) for m in messages)
            if add_generation_prompt:
                text += "<assistant>\n"
            return self.encode(text) if tokenize else text

    monkeypatch.setattr(
        runtime,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(tokenizer=_CharTokenizer()),
    )
    monkeypatch.setattr(
        generation,
        "generate_mtpk",
        lambda *_args, **_kwargs: SimpleNamespace(
            tokens=[1, 2],
            text="ok",
            stats=SimpleNamespace(
                generated_tokens=2,
                tok_s=10.0,
                decode_tok_s=10.0,
                prompt_tps=100.0,
                prompt_eval_time_s=0.01,
                elapsed_s=0.2,
                verify_calls=1,
                verify_time_s=0.01,
                draft_time_s=0.02,
                accepted_drafts=1,
                drafted_tokens=2,
                speculative_depth=2,
                requested_speculative_depth=3,
                peak_memory_bytes=1024**3,
            ),
        ),
    )
    model_dir = _model_dir_with_window(tmp_path, 4096)

    payload = run_prefill_ladder(_ladder_args(str(model_dir), "512"))

    assert len(preflight_calls) == 1
    assert preflight_calls[0]["entry"] == "bench.prefill_ladder"
    assert preflight_calls[0]["contexts"] == [512]
    assert payload["memory_preflight"]["stub"] is True


# --- entry: bench run depth-sweep harness -----------------------------------


def test_depth_sweep_entry_runs_preflight(monkeypatch):
    import mtplx.commands.public as public

    preflight_calls = _counting_preflight(monkeypatch)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    fake_runner = ModuleType("mtplx.benchmarks.runners.mtp_depth_sweep")
    fake_runner.run_mtp_depth_sweep = lambda *_args, **_kwargs: {"depths": []}
    monkeypatch.setitem(
        sys.modules, "mtplx.benchmarks.runners.mtp_depth_sweep", fake_runner
    )

    result = public._depth_sweep_native60(
        model="/tmp/model",
        prompt_suite="/tmp/prompts.jsonl",
        depths="1",
        max_tokens=None,
        limit=1,
        seed=0,
    )

    assert len(preflight_calls) == 1
    assert preflight_calls[0]["entry"] == "bench.depth_sweep"
    assert result["memory_preflight"]["stub"] is True


# --- entry: one-shot run/chat body ------------------------------------------


def test_one_shot_entry_runs_preflight_before_load(monkeypatch):
    import mtplx.commands.public as public

    preflight_calls = _counting_preflight(monkeypatch)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    order: list[str] = []

    fake_runtime = ModuleType("mtplx.runtime")

    def fake_load(*_args, **_kwargs):
        order.append("load")
        return SimpleNamespace(tokenizer=object())

    fake_runtime.load = fake_load
    fake_schema = ModuleType("mtplx.benchmarks.schema")
    fake_schema.PromptCase = lambda **kw: SimpleNamespace(**kw)
    fake_schema.encode_prompt_case = lambda *a, **kw: [1, 2, 3]
    fake_generation = ModuleType("mtplx.generation")
    fake_generation.generate_mtpk = lambda *a, **kw: SimpleNamespace(
        text="ok",
        tokens=[1],
        stats=SimpleNamespace(
            generated_tokens=1, tok_s=1.0, verify_time_s=0.0, verify_calls=0
        ),
    )
    fake_generation.generate_ar = fake_generation.generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "mtplx.benchmarks.schema", fake_schema)
    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: ("/tmp/model", None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda runtime_model, *, unsafe_force_unverified, yes: ({}, None),
    )

    def counting_preflight(**kwargs):
        order.append("preflight")
        preflight_calls.append(kwargs)
        return {"entry": kwargs.get("entry"), "stub": True}

    monkeypatch.setattr(openai, "apply_memory_caps_preflight", counting_preflight)

    args = SimpleNamespace(
        prompt="hello",
        prompt_arg=None,
        model="/tmp/model",
        cache_dir=None,
        unsafe_force_unverified=False,
        yes=True,
        profile="performance-cold",
        max=False,
        system=None,
        max_tokens=8,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        seed=0,
        expect_python=False,
    )

    code, payload, _validations = public._generate_one_shot_public(
        args, command="run"
    )

    assert code == 0
    assert order[:2] == ["preflight", "load"]
    assert payload["memory_preflight"]["entry"] == "cli.run"


# --- entry: quickstart terminal chat ----------------------------------------


def test_quickstart_chat_entry_runs_preflight_before_load(monkeypatch):
    import mtplx.commands.public as public

    preflight_calls = _counting_preflight(monkeypatch)
    monkeypatch.setattr(os, "environ", os.environ.copy())

    class _Boom(RuntimeError):
        pass

    fake_runtime = ModuleType("mtplx.runtime")

    def exploding_load(*_args, **_kwargs):
        raise _Boom("load reached")

    fake_runtime.load = exploding_load
    monkeypatch.setitem(sys.modules, "mtplx.runtime", fake_runtime)

    args = SimpleNamespace(
        profile="sustained",
        _cli_flags={"profile"},
        mtplx_config={},
        model="/tmp/model",
        load_mtp=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        max=False,
        reasoning="on",
    )

    with pytest.raises(_Boom):
        public._quickstart_run_terminal_chat_body(
            args, runtime_model="/tmp/model", inspection={}
        )

    assert len(preflight_calls) == 1
    assert preflight_calls[0]["entry"] == "quickstart.terminal_chat"


# --- F23: /health degradation block -----------------------------------------


def test_health_degradation_block_is_additive_with_defensive_defaults():
    state = _fake_state()
    client = TestClient(create_app(state))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    # Harness-pinned existing fields stay exactly where they were.
    assert body["model"] == "mtplx-test-model"
    assert body["generation_mode"] == state.args.generation_mode
    assert body["profile"]["name"] == state.profile.name
    assert body["context_window"] == 4096
    assert body["metal_memory_caps"] == {"applied": False, "reason": "test"}
    degradation = body["degradation"]
    compiled = degradation["compiled_verify"]
    assert compiled["mode"] in {"off", "on", "parity", "parity2", "unknown"}
    assert compiled["permanent_eager"] in (True, False, "unknown")
    assert isinstance(compiled["reason"], str)
    assert isinstance(degradation["profile_env_overridden"], list)
    nax = degradation["nax"]
    for key in ("env_enabled", "available", "counters", "bail_counters"):
        assert key in nax
    assert nax["env_enabled"] in (True, False, "unknown")
    assert nax["available"] in (True, False, "unknown")


def test_health_degradation_never_crashes_when_probes_blow_up(monkeypatch):
    import mtplx.graphbank as graphbank
    import mtplx.nax_verify as nax_verify
    import mtplx.profiles as profiles

    def boom(*_args, **_kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(graphbank, "compiled_verify_mode", boom)
    monkeypatch.setattr(profiles, "profile_env_status", boom)
    monkeypatch.setattr(nax_verify, "nax_env_enabled", boom)
    monkeypatch.setattr(nax_verify, "nax_available", boom)
    # Kill the module-state lanes too, so every probe is genuinely dead
    # (the runtime lane exposes these as module attrs, not callables).
    monkeypatch.setattr(graphbank, "compiled_verify_status", None, raising=False)
    monkeypatch.setattr(
        nax_verify, "nax_qlinear_fallback_counts", None, raising=False
    )
    import mtplx.attention_split as attention_split
    import mtplx.kernels.sdpa_gqa_packed as sdpa_gqa_packed

    monkeypatch.setattr(
        sdpa_gqa_packed, "gqa_packed_bail_counts", None, raising=False
    )
    monkeypatch.setattr(
        attention_split, "gqa_packed_route_bail_counts", None, raising=False
    )

    client = TestClient(create_app(_fake_state()))
    response = client.get("/health")

    assert response.status_code == 200
    degradation = response.json()["degradation"]
    assert degradation["compiled_verify"]["mode"] == "unknown"
    assert degradation["compiled_verify"]["permanent_eager"] == "unknown"
    assert degradation["profile_env_overridden"] == []
    assert degradation["nax"]["env_enabled"] == "unknown"
    assert degradation["nax"]["available"] == "unknown"


def test_health_degradation_surfaces_parallel_lane_state(monkeypatch):
    import mtplx.graphbank as graphbank

    # The runtime lane's canonical surface is the graphbank module dict;
    # state attrs are the override lane. Fake both and expect the merge:
    # module truth first, state keys override, extra module keys survive.
    monkeypatch.setattr(
        graphbank,
        "compiled_verify_status",
        {
            "mode": "parity2",
            "permanent_eager": True,
            "reason": "bits_gate_unmeasured",
            "flip_count": 2,
            "transient_exception_count": 5,
        },
        raising=False,
    )
    state = _fake_state()
    state.nax_bail_counters = {"m16_bailouts": 3}
    client = TestClient(create_app(state))

    response = client.get("/health")

    assert response.status_code == 200
    degradation = response.json()["degradation"]
    assert degradation["compiled_verify"] == {
        "mode": "parity2",
        "permanent_eager": True,
        "reason": "bits_gate_unmeasured",
        "flip_count": 2,
        "transient_exception_count": 5,
    }
    assert degradation["nax"]["bail_counters"] == {"m16_bailouts": 3}


def test_health_degradation_lists_profile_env_overrides(monkeypatch):
    state = _fake_state()
    profile_env = state.profile.env_dict()
    assert profile_env, "profile must carry env keys for this test"
    key, expected = next(iter(sorted(profile_env.items())))
    monkeypatch.setenv(key, str(expected) + "-overridden")
    client = TestClient(create_app(state))

    response = client.get("/health")

    assert response.status_code == 200
    overridden = response.json()["degradation"]["profile_env_overridden"]
    assert key in overridden


def test_health_nax_block_reports_flash_route_dispatches(monkeypatch):
    # #459: reporters could only see bails. An engaged route now shows its
    # dispatch count so an M5 receipt and an M1-M4 receipt read differently.
    import mtplx.kernels.sdpa_nax_flash as flash
    import mtplx.kernels.sdpa_nax_flash_dsplit as dsplit

    monkeypatch.setattr(flash, "nax_flash_dispatch_counts", {"dispatched": 7})
    monkeypatch.setattr(dsplit, "nax_flash_dsplit_dispatch_counts", {"dispatched": 3})
    state = _fake_state()
    client = TestClient(create_app(state))

    body = client.get("/health").json()

    nax = body["degradation"]["nax"]
    assert nax["flash_dispatch_counters"] == {
        "nax_flash_dispatch_counts": {"dispatched": 7},
        "nax_flash_dsplit_dispatch_counts": {"dispatched": 3},
    }
