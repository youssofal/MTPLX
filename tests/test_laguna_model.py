from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from mtplx.models.laguna_config import LAGUNA_S_2_1_QUANTIZATION


def _tiny_laguna_config(**updates):
    config = {
        "model_type": "laguna",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_attention_heads_per_layer": [2, 4],
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
        "mlp_only_layers": [0],
        "gating": "per-head",
        "sliding_window": 8,
        "layer_types": ["full_attention", "sliding_attention"],
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500_000.0,
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    }
    config.update(updates)
    return config


def _target_laguna_config(**updates):
    layer_types = [
        layer_type
        for _ in range(12)
        for layer_type in (
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        )
    ]
    quantization = copy.deepcopy(LAGUNA_S_2_1_QUANTIZATION)
    config = {
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "hidden_size": 3072,
        "num_hidden_layers": 48,
        "intermediate_size": 12288,
        "num_attention_heads": 48,
        "num_attention_heads_per_layer": [
            48 if layer_type == "full_attention" else 72
            for layer_type in layer_types
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 100352,
        "bos_token_id": 2,
        "eos_token_id": [2, 24],
        "pad_token_id": 9,
        "rms_norm_eps": 1e-6,
        "num_experts": 256,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
        "moe_routed_scaling_factor": 2.5,
        "moe_router_logit_softcapping": 0.0,
        "moe_apply_router_weight_on_input": False,
        "router_aux_loss_coef": 0.0,
        "mlp_only_layers": [0],
        "gating": "per-head",
        "gating_types": ["per_head"] * 48,
        "sliding_window": 512,
        "layer_types": layer_types,
        "mlp_layer_types": ["dense", *("sparse" for _ in range(47))],
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500_000.0,
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.4852030263919618,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        "max_position_embeddings": 1_048_576,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "quantization": copy.deepcopy(quantization),
        "quantization_config": copy.deepcopy(quantization),
    }
    config.update(updates)
    return config


def _pipenetwork_laguna_config(**updates):
    """The superseded uniform-4bit build; kept only as rejection coverage."""

    quantization = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        **{
            f"model.layers.{layer}.mlp.gate": {"bits": 8, "group_size": 64}
            for layer in range(1, 48)
        },
    }
    return _target_laguna_config(
        quantization=copy.deepcopy(quantization),
        quantization_config=copy.deepcopy(quantization),
        **updates,
    )


def test_laguna_model_type_resolves_bundled_classes() -> None:
    from mtplx.runtime import _model_classes_for_config

    model_class, args_class = _model_classes_for_config(_target_laguna_config())

    assert model_class.__module__ == "mtplx.models.laguna"
    assert model_class.__name__ == "Model"
    assert args_class.__module__ == "mtplx.models.laguna"
    assert args_class.__name__ == "ModelArgs"


def test_other_laguna_variants_do_not_resolve_bundled_classes() -> None:
    from mtplx.runtime import _model_classes_for_config

    assert (
        _model_classes_for_config(
            _target_laguna_config(
                quantization_config={
                    "bits": 8,
                    "group_size": 64,
                    "mode": "affine",
                }
            )
        )
        is None
    )
    wrong_rope = _target_laguna_config()
    wrong_rope["rope_parameters"]["full_attention"]["beta_fast"] = 7.0
    assert _model_classes_for_config(wrong_rope) is None

    # The superseded uniform-4bit pipenetwork build is now blocked like any
    # other non-pinned variant: same geometry, wrong quantization map.
    assert _model_classes_for_config(_pipenetwork_laguna_config()) is None

    # Flip the lone non-uniform attention bit (layer 33 o_proj 8 -> 5) in both
    # maps; the exact imatrix quantization map no longer matches.
    mutated_quant = _target_laguna_config()
    for field in ("quantization", "quantization_config"):
        mutated_quant[field][
            "language_model.model.layers.33.self_attn.o_proj"
        ]["bits"] = 5
    assert _model_classes_for_config(mutated_quant) is None

    # Dropping any per-path override also breaks the exact map.
    missing_override = _target_laguna_config()
    for field in ("quantization", "quantization_config"):
        missing_override[field].pop(
            "language_model.model.layers.47.self_attn.q_proj"
        )
    assert _model_classes_for_config(missing_override) is None

    remote_code = _target_laguna_config(model_file="evil.py")
    assert _model_classes_for_config(remote_code) is None

    assert (
        _model_classes_for_config(_target_laguna_config(hidden_size="invalid"))
        is None
    )
    assert (
        _model_classes_for_config(
            _target_laguna_config(num_attention_heads_per_layer=[48] * 48)
        )
        is None
    )
    for token_update in (
        {"bos_token_id": 3},
        {"eos_token_id": [2]},
        {"pad_token_id": 10},
    ):
        assert _model_classes_for_config(_target_laguna_config(**token_update)) is None


def test_laguna_load_bypasses_mlx_lm_registry(monkeypatch, tmp_path: Path) -> None:
    import mlx_lm.utils
    from mtplx import runtime

    expected_model = object()
    expected_tokenizer = object()
    observed: dict[str, object] = {}

    target_config = _target_laguna_config()

    def fake_load_model(path, *, get_model_classes, model_config=None):
        observed["path"] = path
        observed["classes"] = get_model_classes(config=target_config)
        observed["model_config"] = model_config
        return expected_model, target_config

    monkeypatch.setattr(mlx_lm.utils, "load_model", fake_load_model)
    monkeypatch.setattr(
        runtime,
        "_load_tokenizer_resilient",
        lambda path, config: expected_tokenizer,
    )

    model, tokenizer = runtime._load_base_model(
        tmp_path,
        target_config,
    )

    assert model is expected_model
    assert tokenizer is expected_tokenizer
    assert observed["path"] == tmp_path
    assert observed["classes"] == runtime._model_classes_for_config(target_config)

    # Quantization is driven by the checkpoint's own config['quantization'] dict
    # (not a model predicate). For the oQ4e export the runtime must strip the
    # ``language_model.`` key prefix so mlx-lm matches each module by tree path.
    from mtplx.models.laguna_config import laguna_module_quantization

    expected_quant = laguna_module_quantization(target_config)
    assert expected_quant is not None
    model_config = observed["model_config"]
    assert model_config is not None
    assert model_config["quantization"] == expected_quant
    assert model_config["quantization_config"] == expected_quant
    assert all(
        not key.startswith("language_model.")
        for key in model_config["quantization"]
    )
    # Routers (mlp.gate) carry no entry and stay unquantized (BF16).
    assert not any(
        key.endswith(".mlp.gate") for key in model_config["quantization"]
    )


def test_laguna_remote_model_file_is_rejected_before_mlx_loading(
    monkeypatch, tmp_path: Path
) -> None:
    import mlx_lm.utils
    from mtplx import runtime

    def unexpected(*_args, **_kwargs):
        pytest.fail("Laguna model_file reached executable MLX loading")

    monkeypatch.setattr(mlx_lm.utils, "load", unexpected)
    monkeypatch.setattr(mlx_lm.utils, "load_model", unexpected)

    with pytest.raises(ValueError, match="model_file.*not permitted"):
        runtime._load_base_model(
            tmp_path,
            _target_laguna_config(model_file="evil.py"),
        )


def test_laguna_tokenizer_failure_stops_before_weight_loading(
    monkeypatch, tmp_path: Path
) -> None:
    import mlx_lm.utils
    from mtplx import runtime

    monkeypatch.setattr(
        runtime,
        "_load_tokenizer_resilient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("invalid pinned tokenizer")
        ),
    )
    monkeypatch.setattr(
        mlx_lm.utils,
        "load_model",
        lambda *_args, **_kwargs: pytest.fail(
            "weights loaded before tokenizer validation"
        ),
    )

    with pytest.raises(ValueError, match="invalid pinned tokenizer"):
        runtime._load_base_model(tmp_path, _target_laguna_config())


def test_laguna_s_2_1_rejects_mtp_before_loading_weights(
    monkeypatch, tmp_path: Path
) -> None:
    from mtplx import runtime

    (tmp_path / "config.json").write_text(
        json.dumps(_target_laguna_config()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda *_args, **_kwargs: pytest.fail("weights must not load"),
    )

    with pytest.raises(ValueError, match="has no native MTP head.*mtp=False"):
        runtime.load(tmp_path, mtp=True)


def test_laguna_rejects_insufficient_unified_memory_before_loading_weights(
    monkeypatch, tmp_path: Path
) -> None:
    from mtplx import runtime

    (tmp_path / "config.json").write_text(
        json.dumps(_target_laguna_config()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime,
        "_detect_total_system_memory_bytes",
        lambda: 64 * 1024**3,
    )
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda *_args, **_kwargs: pytest.fail("weights must not load"),
    )

    with pytest.raises(RuntimeError, match="requires at least.*unified memory"):
        runtime.load(tmp_path, mtp=False)


def test_laguna_s_2_1_ar_route_skips_qwen_performance_hooks(
    monkeypatch, tmp_path: Path
) -> None:
    from mtplx import (
        attention_split,
        cache_state,
        kernel_selfcheck,
        native_mlp,
        nax_verify,
        runtime,
    )

    (tmp_path / "config.json").write_text(
        json.dumps(_target_laguna_config()),
        encoding="utf-8",
    )
    expected_cache = [object()]
    expected_logits = object()
    calls: list[tuple[object, object, bool, object]] = []

    class FakeModel:
        def make_cache(self):
            return expected_cache

        def __call__(
            self,
            input_ids,
            *,
            cache=None,
            input_embeddings=None,
            emit_logits=True,
            logits_keep=None,
        ):
            calls.append((input_ids, cache, emit_logits, logits_keep))
            return expected_logits if emit_logits else None

    model = FakeModel()
    tokenizer = object()
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda _path, _config: (model, tokenizer),
    )

    def unexpected(*_args, **_kwargs):
        pytest.fail("Qwen performance hook reached the Laguna route")

    monkeypatch.setattr(attention_split, "configure_split_full_attention", unexpected)
    monkeypatch.setattr(native_mlp, "configure_native_mlp", unexpected)
    monkeypatch.setattr(nax_verify, "nax_env_enabled", unexpected)
    monkeypatch.setattr(kernel_selfcheck, "maybe_run_model_selfcheck", unexpected)
    monkeypatch.setattr(
        cache_state,
        "configure_owned_recurrent_state_cache",
        unexpected,
    )
    monkeypatch.setattr(
        cache_state,
        "configure_tail_owned_attention_kv_cache",
        unexpected,
    )

    loaded = runtime.load(tmp_path, mtp=False)

    assert loaded.model is model
    assert loaded.tokenizer is tokenizer
    assert loaded.mtp_enabled is False
    assert loaded.make_cache() is expected_cache
    input_ids = object()
    assert loaded.forward_ar(input_ids, cache=expected_cache) is expected_logits
    assert (
        loaded.forward_ar(
            input_ids,
            cache=expected_cache,
            emit_logits=False,
        )
        is None
    )
    assert calls == [
        (input_ids, expected_cache, True, None),
        (input_ids, expected_cache, False, None),
    ]
    assert loaded.diagnostic_counters == {}

    monkeypatch.setenv(
        "MTPLX_SUSTAINED_PREFILL_LAYOUT",
        "contiguous_then_repage",
    )
    from mtplx.generation import _maybe_repage_target_prefill_cache

    assert _maybe_repage_target_prefill_cache(loaded, expected_cache) == 0.0


def test_laguna_rejects_mismatched_per_layer_geometry() -> None:
    from mtplx.models.laguna import ModelArgs

    with pytest.raises(
        ValueError,
        match="num_attention_heads_per_layer must match num_hidden_layers",
    ):
        ModelArgs.from_dict(_tiny_laguna_config(num_attention_heads_per_layer=[2]))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"layer_types": ["full_attention"]}, "layer_types must match"),
        (
            {
                "num_key_value_heads": 2,
                "num_attention_heads_per_layer": [2, 3],
            },
            "attention head counts must be divisible",
        ),
        ({"num_experts_per_tok": 5}, "num_experts_per_tok"),
        ({"sliding_window": 0}, "sliding_window"),
        ({"decoder_sparse_step": 0}, "decoder_sparse_step"),
    ],
)
def test_laguna_rejects_invalid_installed_geometry(updates, message) -> None:
    from mtplx.models.laguna import ModelArgs

    with pytest.raises(ValueError, match=message):
        ModelArgs.from_dict(_tiny_laguna_config(**updates))


def test_sanitize_maps_oq4e_layout_onto_module_tree() -> None:
    mx = pytest.importorskip("mlx.core")
    from mtplx.models.laguna import Model, ModelArgs

    config = _tiny_laguna_config(architectures=["LagunaForCausalLM"], head_dim=4)
    model = Model(ModelArgs.from_dict(config))

    # Synthetic oQ4e-layout weights: every key wrapped under language_model.,
    # the router stored as gate.proj.weight, and the load-balancing bias parked
    # under gate.e_score_correction_bias. Values are tiny placeholders.
    oq4e = {
        "language_model.model.embed_tokens.weight": mx.zeros((1,)),
        "language_model.lm_head.weight": mx.zeros((1,)),
        "language_model.model.layers.1.mlp.gate.proj.weight": mx.zeros((1,)),
        "language_model.model.layers.1.mlp.gate.e_score_correction_bias": mx.zeros(
            (1,)
        ),
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": mx.zeros(
            (1,)
        ),
        "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
    }
    sanitized = model.sanitize(dict(oq4e))

    assert set(sanitized) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.e_score_correction_bias",
        "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    }
    # The remaps are pure renames — the array objects travel unchanged, and the
    # mapping is total (no source key is dropped or collided).
    assert (
        sanitized["model.layers.1.mlp.gate.weight"]
        is oq4e["language_model.model.layers.1.mlp.gate.proj.weight"]
    )
    assert (
        sanitized["model.layers.1.mlp.e_score_correction_bias"]
        is oq4e["language_model.model.layers.1.mlp.gate.e_score_correction_bias"]
    )
    assert len(sanitized) == len(oq4e)

    # Native-layout weights (no wrapper prefix) pass through untouched.
    native = {"model.norm.weight": mx.zeros((1,))}
    native_result = model.sanitize(dict(native))
    assert set(native_result) == set(native)
    assert native_result["model.norm.weight"] is native["model.norm.weight"]


def test_laguna_memory_floor_tracks_oq4e_weight_bytes() -> None:
    from mtplx.models import laguna_config as lc

    assert lc.LAGUNA_S_2_1_REPO_ID == "mlx-community/Laguna-S-2.1-oQ4e"
    assert lc.LAGUNA_S_2_1_REVISION == "8e3f5cad513746264940c1c4195de48d7ea345a5"
    assert lc.LAGUNA_S_2_1_WEIGHT_BYTES == 64_122_027_323
    assert lc.LAGUNA_S_2_1_REPO_BYTES == 64_129_728_868
    assert lc.LAGUNA_S_2_1_MIN_RESIDENT_BYTES == lc.laguna_s_2_1_required_resident_bytes(
        32_768
    )
    # weights + 8 GiB headroom + rotating KV + per-token KV * default context.
    assert lc.LAGUNA_S_2_1_MIN_RESIDENT_BYTES == (
        64_122_027_323 + 8 * 1024**3 + 75_497_472 + 32_768 * 49_152
    )


def test_tiny_laguna_checkpoint_loads_and_cached_forward_matches_full(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.utils import load_model
    from mlx.utils import tree_flatten

    from mtplx.models.laguna import Model, ModelArgs

    config = _tiny_laguna_config(
        architectures=["LagunaForCausalLM"],
        head_dim=4,
    )
    model = Model(ModelArgs.from_dict(config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        dict(tree_flatten(model.parameters())),
        metadata={"format": "mlx"},
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    loaded, _loaded_config = load_model(
        tmp_path,
        get_model_classes=lambda config: (Model, ModelArgs),
    )
    inputs = mx.array([[1, 2, 3]])
    original_head = loaded.lm_head

    class UnexpectedHead:
        def __call__(self, _hidden):
            pytest.fail("cache-only Laguna prefill reached lm_head")

    loaded.lm_head = UnexpectedHead()
    assert loaded(inputs, cache=loaded.make_cache(), emit_logits=False) is None
    loaded.lm_head = original_head
    full_logits = loaded(inputs)
    final_only_logits = loaded(inputs, logits_keep=1)
    cache = loaded.make_cache()
    cached_logits = None
    for token in (1, 2, 3):
        cached_logits = loaded(mx.array([[token]]), cache=cache)
        mx.eval(cached_logits)

    assert cached_logits is not None
    mx.eval(full_logits, final_only_logits, cached_logits)
    assert final_only_logits.shape[1] == 1
    max_abs_error = float(mx.max(mx.abs(full_logits[:, -1] - cached_logits[:, -1])))
    # Full-sequence and one-token attention use different MLX kernel shapes;
    # keep a tight numerical bound while still catching mask/cache drift.
    assert max_abs_error < 5e-3, f"max cached-logit error: {max_abs_error}"


_TINY_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: "
    "{{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


def _write_tiny_tokenizer(model_dir: Path, *, chat_template: str | None) -> None:
    """Write a minimal but real tokenizer.json + tokenizer_config.json.

    Enough for _load_tokenizer_resilient to construct a working tokenizer with
    apply_chat_template, without any model weights.
    """

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "hello": 2,
        "world": 3,
        "user": 4,
        "assistant": 5,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    tokenizer_config: dict[str, object] = {
        "eos_token": "<eos>",
        "unk_token": "<unk>",
    }
    if chat_template is not None:
        tokenizer_config["chat_template"] = chat_template
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8"
    )


def test_jinja_include_chat_template_detection() -> None:
    from mtplx.runtime import _is_jinja_include_chat_template

    # The pinned oQ4e 35-char redirect stub and whitespace-trim variants.
    assert _is_jinja_include_chat_template("{% include 'chat_template.jinja' %}")
    assert _is_jinja_include_chat_template("{%- include 'chat_template.jinja' -%}")
    # A real, self-contained template must be left alone.
    assert not _is_jinja_include_chat_template(_TINY_CHAT_TEMPLATE)
    assert not _is_jinja_include_chat_template(None)
    assert not _is_jinja_include_chat_template("")


def test_load_tokenizer_resilient_repairs_include_stub_hermetic(
    tmp_path: Path,
) -> None:
    """A fake model dir whose tokenizer_config.json only redirects to the
    sidecar must load with the sidecar contents substituted in memory."""

    from mtplx import runtime

    (tmp_path / "chat_template.jinja").write_text(
        _TINY_CHAT_TEMPLATE, encoding="utf-8"
    )
    _write_tiny_tokenizer(
        tmp_path, chat_template="{% include 'chat_template.jinja' %}"
    )

    tokenizer = runtime._load_tokenizer_resilient(tmp_path, {"eos_token_id": 1})

    # The include stub is gone, replaced by the pinned sidecar's contents.
    assert not runtime._is_jinja_include_chat_template(tokenizer.chat_template)
    assert tokenizer.chat_template == _TINY_CHAT_TEMPLATE
    # The on-disk sidecar is never mutated (artifact hashes are load-bearing).
    assert (tmp_path / "chat_template.jinja").read_text(
        encoding="utf-8"
    ) == _TINY_CHAT_TEMPLATE
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == "user: hello\nassistant:"


def test_pinned_laguna_chat_template_renders_after_include_stub_repair() -> None:
    """The real pinned oQ4e sidecars (when present) must build a tokenizer that
    renders a chat message instead of raising the loader-less-include TypeError.

    Skips cleanly when the model dir is absent so CI stays hermetic.
    """

    import json as _json

    from mtplx import runtime
    from mtplx.hf_loader import DEFAULT_MODEL_CACHE, safe_model_name
    from mtplx.models.laguna_config import LAGUNA_S_2_1_REPO_ID

    # Resolve the real developer-machine cache directly: the suite-wide
    # conftest isolation repoints MTPLX_MODEL_DIR at an empty tmp dir, so
    # cached_model_path() would never see the pinned sidecars.
    model_dir = DEFAULT_MODEL_CACHE / safe_model_name(LAGUNA_S_2_1_REPO_ID)
    required = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    if not model_dir.exists() or not all(
        (model_dir / name).exists() for name in required
    ):
        pytest.skip(f"pinned Laguna sidecars not present under {model_dir}")

    # Confirm the checkpoint really ships the loader-less include stub we fix.
    tokenizer_config = _json.loads(
        (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert runtime._is_jinja_include_chat_template(
        tokenizer_config.get("chat_template")
    )

    config = _json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = runtime._load_tokenizer_resilient(model_dir, config)

    pinned = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    assert not runtime._is_jinja_include_chat_template(tokenizer.chat_template)
    assert tokenizer.chat_template == pinned
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(rendered, str) and rendered


def test_ar_cycle_snapshot_prefill_returns_no_hidden_under_sustained_env(
    monkeypatch,
) -> None:
    """Regression: the AR (cycle-policy) snapshot re-prefill must not demand
    hidden states from a target-only runtime.

    Under the sustained serving profile, an AR runtime's postcommit routes
    through restore_or_prefill_prompt_state with the cycle policy, which
    cold-prefills via _prefill(return_hidden=...). A LagunaARRuntime's
    forward_ar returns logits alone, so requesting hidden unpacked a lone
    logits array as ``(logits, hidden)`` and raised ``ValueError: not enough
    values to unpack (expected 2, got 1)``. hidden must come back None instead.
    """

    pytest.importorskip("mlx.core")
    from mtplx import generation
    from mtplx.models.laguna import Model, ModelArgs
    from mtplx.mtp_patch import MTPContract
    from mtplx.profiles import SUSTAINED_PREFILL_ENV
    from mtplx.runtime import LagunaARRuntime

    # Snapshot the whole environ so the sustained profile's MTPLX_* keys (and
    # anything the code writes) cannot leak into later in-process tests — the
    # idiom from test_laguna_one_shot_uses_target_generation_defaults.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key, value in SUSTAINED_PREFILL_ENV.items():
        os.environ[key] = str(value)

    config = _tiny_laguna_config(architectures=["LagunaForCausalLM"], head_dim=4)
    runtime = LagunaARRuntime(
        Model(ModelArgs.from_dict(config)),
        object(),  # tokenizer: unused by the prefill path
        Path("/tmp/laguna-ar"),
        False,  # mtp_enabled: target-only AR runtime, no draft head
        MTPContract(),
    )
    assert runtime.mtp_enabled is False

    prompt_state = generation.restore_or_prefill_prompt_state(
        runtime,
        [1, 2, 3, 4],
        mtp_history_policy="cycle",
        session_bank=None,
    )

    # Reached the AR cold-prefill path and produced a usable state — no hidden
    # (the trunk cache is all the AR snapshot banks).
    assert prompt_state.mtp_history_policy == "cycle"
    assert prompt_state.hidden is None
    assert prompt_state.trunk_cache is not None
    assert prompt_state.logits is not None


def _target_only_ar_runtime():
    """A real LagunaARRuntime over a tiny CPU model — forward_ar returns logits
    alone, exactly like the served oQ4e target."""

    from mtplx.models.laguna import Model, ModelArgs
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import LagunaARRuntime

    config = _tiny_laguna_config(architectures=["LagunaForCausalLM"], head_dim=4)
    runtime = LagunaARRuntime(
        Model(ModelArgs.from_dict(config)),
        object(),  # tokenizer: unused by the prefill path
        Path("tiny-laguna-ar"),
        False,  # mtp_enabled: target-only AR runtime, no draft head
        MTPContract(),
    )
    assert runtime.mtp_enabled is False
    return runtime


def _warm_prefix_cache(runtime, prefix_ids):
    """Really prefill `prefix_ids` so a restore hands back a live, warm cache."""

    import mlx.core as mx

    cache = runtime.make_cache()
    logits = runtime.forward_ar(mx.array([list(prefix_ids)]), cache=cache)
    mx.eval(logits)
    return cache, logits


def _exact_restore_bank(runtime, prefix_ids):
    """A session bank whose exact restore serves the warm prefix cache."""

    from types import SimpleNamespace

    cache, logits = _warm_prefix_cache(runtime, prefix_ids)

    class Bank:
        last_miss_reason = None

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=len(prefix_ids)),
                cache=cache,
                logits=logits[:, -1, :],
                # An AR turn banks the trunk cache only: no hidden was stored.
                hidden=None,
                mtp_history_cache=None,
                restore_mode="clone",
            )

    return Bank()


@pytest.mark.parametrize(
    ("lane", "fused_max", "suffix_len"),
    [("fused", 64, 3), ("chunked", 0, 5)],
)
def test_warm_restore_suffix_prefill_needs_no_hidden_from_target_only_runtime(
    monkeypatch, lane, fused_max, suffix_len
) -> None:
    """Regression for the live oQ4e serving crash.

    `_prefill_restored_prompt_suffix` asked every runtime for hidden states
    (``rt.forward_ar(..., return_hidden=True)``) on both of its lanes — the
    fused small-suffix forward and the final single-token forward after the
    chunked body. A LagunaARRuntime returns logits alone, so the two-name
    unpack raised ``ValueError: not enough values to unpack (expected 2, got
    1)`` and crash-looped the server on every warm session restore. The cold
    prefill path was already gated on rt.mtp_enabled; the warm one must be too.
    """

    pytest.importorskip("mlx.core")
    from mtplx import generation

    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ["MTPLX_SMALL_SUFFIX_FUSED_MAX"] = str(fused_max)
    os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = "2"

    runtime = _target_only_ar_runtime()
    prefix = [1, 2, 3, 4, 5, 6]
    prompt = prefix + list(range(7, 7 + suffix_len))

    prompt_state = generation.restore_or_prefill_prompt_state(
        runtime,
        prompt,
        mtp_history_policy="cycle",
        session_bank=_exact_restore_bank(runtime, prefix),
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == len(prefix)
    assert prompt_state.suffix_tokens == suffix_len
    # hidden stays None for AR; the trunk cache is all the bank stores.
    assert prompt_state.hidden is None
    assert prompt_state.logits is not None
    # The lane under test really ran.
    counters = runtime.diagnostic_counters
    if lane == "fused":
        assert counters.get("restored_suffix_prefill_fused", 0) == 1
    else:
        assert counters.get("restored_suffix_prefill_chunks", 0) >= 1


def _near_prefix_bank(runtime, prefix_ids, *, entry_len):
    """A bank offering a near-prefix candidate whose warm cache stops at
    `prefix_ids` — the lane that runs the single-token repair forward."""

    from types import SimpleNamespace

    cache, _logits = _warm_prefix_cache(runtime, prefix_ids)
    entry = SimpleNamespace(
        prefix_len=entry_len,
        token_ids=tuple(range(entry_len)),
        session_id="session-1",
        model_path=str(runtime.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="cycle",
        draft_head_identity=None,
        policy_fingerprint=None,
        snapshot_epoch=entry_len,
        mtp_snapshot_epoch=None,
        mtp_history_snapshot=None,
        mtp_history_cache_ref=None,
        gdn_boundaries=(),
        has_recurrent=False,
        live_ref_only=False,
        cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return None

        def near_prefix_candidates(self, _prompt_ids, **_kwargs):
            return [(entry, len(prefix_ids))]

        def restore_entry_prefix_cache(
            self, _rt, _entry, prefix_len, *, mode, cache_factory=None
        ):
            return (cache, None, "clone", int(prefix_len))

        def restore(self, *_args, **_kwargs):
            return None

    return Bank(), entry


@pytest.mark.parametrize("suffix_len", [2, 0])
def test_near_prefix_restore_needs_no_hidden_from_target_only_runtime(
    monkeypatch, suffix_len
) -> None:
    """The near/block-prefix lane's repair forward had the same ungated
    ``return_hidden=True``.

    Nothing about that lane is MTP-gated — it fires for any runtime with a
    session bank — so an AR turn that diverged from the banked transcript hit
    the identical unpack ValueError before reaching the suffix prefill. Both
    the with-suffix and the empty-suffix return must tolerate hidden=None.
    """

    pytest.importorskip("mlx.core")
    from mtplx import generation

    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ["MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS"] = "2"
    os.environ["MTPLX_SESSION_PREFIX_BLOCK_SIZE"] = "2"
    os.environ["MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS"] = "2"

    runtime = _target_only_ar_runtime()
    prefix = [1, 2, 3, 4, 5, 6]
    prompt = prefix + list(range(7, 7 + suffix_len))
    bank, entry = _near_prefix_bank(runtime, prefix, entry_len=len(prefix) + 2)

    prompt_state = generation.restore_or_prefill_prompt_state(
        runtime,
        prompt,
        mtp_history_policy="cycle",
        session_bank=bank,
    )

    assert prompt_state.restore_mode.startswith("near_prefix")
    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == len(prefix)
    assert prompt_state.suffix_tokens == suffix_len
    assert prompt_state.hidden is None
    assert prompt_state.logits is not None
    assert entry.hits == 1


def _batched_greedy_rows(model, prompts, steps: int):
    """Greedy-decode every row of `prompts` in one batched forward per step."""

    import mlx.core as mx

    cache = model.make_cache()
    logits = model(prompts, cache=cache, logits_keep=1)
    token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]
    rows = [token]
    for _ in range(steps):
        logits = model(token, cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]
        rows.append(token)
    stacked = mx.concatenate(rows, axis=1)
    mx.eval(stacked)
    return stacked.tolist()


def test_batched_decode_matches_single_stream_decode() -> None:
    """A batched decode step must not corrupt every row but the first.

    MLX 0.31.2's ``mx.fast.rope`` takes a "single" fast path when the input is
    row-contiguous with sequence length 1 and the offset holds one value — the
    exact shape of a batched decode step, since transposing a length-1 sequence
    dimension leaves the strides row-contiguous.  That path dispatches a
    two-dimensional ``(dims/2, heads)`` grid with no batch term, so rows
    1..B-1 are never written and come back as whatever was in the freshly
    allocated output buffer.  Prefill (T > 1) is unaffected, so the corruption
    only appears once generation starts and only above batch 1.

    Guarding it here rather than in the bench: this is a serving-correctness
    contract, and a mocked test suite cannot see it.
    """

    import mlx.core as mx

    from mtplx.models.laguna import Model, ModelArgs

    mx.random.seed(0)
    args = ModelArgs.from_dict(
        _tiny_laguna_config(
            head_dim=8,  # partial_rotary_factor 0.5 needs an even rotated dim
            num_hidden_layers=4,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_attention_heads_per_layer=None,
        )
    )
    model = Model(args)
    mx.eval(model.parameters())

    # Prompt longer than the sliding window so the rotating cache wraps.
    prompts = mx.array(
        [
            [3, 9, 14, 2, 7, 21, 5, 11, 30, 1, 18, 6],
            [17, 4, 25, 8, 13, 0, 29, 22, 10, 16, 27, 19],
            [1, 1, 2, 3, 5, 8, 13, 21, 2, 3, 5, 8],
        ],
        dtype=mx.uint32,
    )
    steps = 6

    batched = _batched_greedy_rows(model, prompts, steps)
    for index in range(prompts.shape[0]):
        solo = _batched_greedy_rows(model, prompts[index : index + 1], steps)[0]
        assert batched[index] == solo, (
            f"row {index} of a batched decode diverged from the same prompt "
            f"decoded alone: batched={batched[index]} solo={solo}"
        )


def test_batched_decode_rows_are_independent() -> None:
    """Row 0's tokens must not depend on what row 1 contains.

    Held at FIXED batch shape so the two runs use identical kernels; any
    difference is genuine cross-row contamination rather than a batched-matmul
    reduction-order flip.
    """

    import mlx.core as mx

    from mtplx.models.laguna import Model, ModelArgs

    mx.random.seed(1)
    args = ModelArgs.from_dict(
        _tiny_laguna_config(head_dim=8, num_attention_heads_per_layer=None)
    )
    model = Model(args)
    mx.eval(model.parameters())

    row_a = [3, 9, 14, 2, 7, 21, 5, 11, 30, 1]
    row_b = [17, 4, 25, 8, 13, 0, 29, 22, 10, 16]
    mixed = mx.array([row_a, row_b], dtype=mx.uint32)
    duplicated = mx.array([row_a, row_a], dtype=mx.uint32)

    mixed_rows = _batched_greedy_rows(model, mixed, 5)
    duplicated_rows = _batched_greedy_rows(model, duplicated, 5)

    assert mixed_rows[0] == duplicated_rows[0]
    assert duplicated_rows[0] == duplicated_rows[1]


def test_laguna_load_attaches_the_fused_install_report(
    monkeypatch, tmp_path: Path
) -> None:
    """The server's [laguna-fused] startup receipt reads this attribute."""

    from mtplx import runtime
    from mtplx.models import laguna_fused

    target_config = _target_laguna_config()
    fused_report = [{"path": "fused_gate_up", "layers_converted": 47}]

    class _StubLagunaRuntime:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(runtime, "load_config", lambda path: target_config)
    monkeypatch.setattr(
        runtime, "_preflight_laguna_system_memory", lambda config: None
    )
    monkeypatch.setattr(
        runtime, "_load_base_model", lambda path, config: (object(), object())
    )
    monkeypatch.setattr(runtime, "_load_runtime_metadata", lambda path: None)
    monkeypatch.setattr(
        laguna_fused, "install_from_env", lambda model: list(fused_report)
    )
    monkeypatch.setattr(runtime, "LagunaARRuntime", _StubLagunaRuntime)

    loaded = runtime.load(tmp_path, mtp=False)

    assert isinstance(loaded, _StubLagunaRuntime)
    assert loaded.laguna_fused_report == fused_report
