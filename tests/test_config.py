from __future__ import annotations

import argparse

from mtplx.config import apply_user_config, load_user_config
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.profiles import DEFAULT_HF_MODEL_ID, DEFAULT_PROFILE_NAME


def test_load_user_config_reads_runtime_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{tmp_path / "models"}"\n'
        f'model_dirs = ["{tmp_path / "archive"}", "{tmp_path / "external"}"]\n'
        'profile = "exact"\n'
        'thermal_control = "none"\n',
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.exists is True
    assert loaded.model == "mtplx/example"
    assert loaded.model_dir == str(tmp_path / "models")
    assert loaded.model_dirs == (
        str(tmp_path / "archive"),
        str(tmp_path / "external"),
    )
    assert loaded.profile == "exact"
    assert loaded.thermal_control == "none"


def test_load_user_config_reads_mtp_batch_numerics(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.mtp_batch_numerics == "balanced"


def test_apply_user_config_respects_explicit_mtp_batch_numerics(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        mtp_batch_numerics="throughput",
        _cli_flags={"mtp-batch-numerics"},
    )

    apply_user_config(args, config_path=config)

    assert args.mtp_batch_numerics == "throughput"


def test_apply_user_config_fills_mtp_batch_numerics_default(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        mtp_batch_numerics="throughput",
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.mtp_batch_numerics == "balanced"


def test_apply_user_config_fills_runtime_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="run",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags=set(),
    )

    loaded = apply_user_config(args, config_path=config)

    assert loaded.exists is True
    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)
    assert args.profile == "exact"
    assert args.mtplx_config["path"] == str(config)


def test_apply_user_config_fills_and_preserves_model_search_dirs(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model_dirs = ["/models/archive", "/models/external"]\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        model_search_dirs=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)
    assert args.model_search_dirs == ["/models/archive", "/models/external"]

    args.model_search_dirs = ["/models/explicit"]
    args._cli_flags = {"model-search-dir"}
    apply_user_config(args, config_path=config)
    assert args.model_search_dirs == ["/models/explicit"]


def test_apply_user_config_sets_forge_model_root(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model_dir = "/models/primary"\n'
        'model_dirs = ["/models/archive", "/models/external"]\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(command="forge", _cli_flags=set())

    apply_user_config(args, config_path=config)

    assert args.model_root == "/models/primary"


def test_apply_user_config_preserves_explicit_forge_model_root(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('model_dir = "/models/config"\n', encoding="utf-8")
    args = argparse.Namespace(
        command="forge",
        forge_action="build",
        model_root="/models/operation",
        _cli_flags={"model-root"},
    )

    apply_user_config(args, config_path=config)

    assert args.model_root == "/models/operation"


def test_apply_user_config_ignores_legacy_optimized_speed_default(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "models/Qwen3.6-27B-MTPLX-Optimized-Speed"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="quickstart",
        model=DEFAULT_HF_MODEL_ID,
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == DEFAULT_HF_MODEL_ID


def test_apply_user_config_preserves_explicit_runtime_values(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        'model_dir = "/tmp/mtplx-models"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        model="models/local",
        cache_dir="/tmp/explicit-cache",
        profile="performance-cold",
        _cli_flags={"model", "cache-dir", "profile"},
    )

    apply_user_config(args, config_path=config)

    assert args.model == "models/local"
    assert args.cache_dir == "/tmp/explicit-cache"
    assert args.profile == "performance-cold"


def test_apply_user_config_preserves_explicit_default_like_model(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    explicit_model = "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Speed"
    args = argparse.Namespace(
        command="serve",
        model=explicit_model,
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags={"model"},
    )

    apply_user_config(args, config_path=config)

    assert args.model == explicit_model


def test_apply_user_config_fills_tune_model_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="tune",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile="performance-cold",
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)
    assert args.profile == "performance-cold"


def test_apply_user_config_fills_retrieval_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'embedding_models = ["org/embed"]\n'
        'reranker_models = ["org/rank=fast-rank"]\n'
        "retrieval_max_resident = 3\n"
        "retrieval_trust_remote_code = true\n",
        encoding="utf-8",
    )

    loaded = load_user_config(config)
    assert loaded.embedding_models == ("org/embed",)
    assert loaded.retrieval_trust_remote_code is True

    args = argparse.Namespace(
        command="serve",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        embedding_model=[],
        reranker_model=[],
        retrieval_max_resident=2,
        retrieval_trust_remote_code=False,
        _cli_flags=set(),
    )
    apply_user_config(args, config_path=config)

    assert tuple(args.embedding_model) == ("org/embed",)
    assert tuple(args.reranker_model) == ("org/rank=fast-rank",)
    assert args.retrieval_max_resident == 3
    assert args.retrieval_trust_remote_code is True


def test_an_explicit_trust_flag_beats_the_config_file(tmp_path):
    """Trust granted in config must not silently override an explicit CLI no.

    A user who runs with the flag omitted after setting the config key gets
    the config value — but one who explicitly typed the flag spelling into
    _cli_flags keeps their command line.
    """
    config = tmp_path / "config.toml"
    config.write_text("retrieval_trust_remote_code = true\n", encoding="utf-8")
    args = argparse.Namespace(
        command="serve",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        retrieval_trust_remote_code=False,
        _cli_flags={"retrieval-trust-remote-code"},
    )

    apply_user_config(args, config_path=config)

    assert args.retrieval_trust_remote_code is False


def test_apply_user_config_fills_bench_tune_model_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="bench",
        bench_action="tune",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=None,
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)


def test_malformed_user_config_does_not_crash_any_command(tmp_path, monkeypatch, capsys):
    # load_user_config runs on EVERY CLI dispatch. A truncated or hand-mangled
    # config.toml used to raise TOMLDecodeError straight through `mtplx status`,
    # `doctor`, and `stop`, so one bad file bricked the whole CLI.
    config = tmp_path / "config.toml"
    config.write_text('model = "unterminated\nprofile =\n', encoding="utf-8")

    loaded = load_user_config(config)

    assert loaded.exists is False
    assert loaded.path == config
    assert loaded.model is None
    assert loaded.profile is None
    warning = capsys.readouterr().err
    assert str(config) in warning
    assert "mtplx config show" in warning
    assert warning.count("\n") == 1

    # And the same file survives a real command dispatch rather than tracebacking.
    from mtplx.cli import main

    monkeypatch.setenv("MTPLX_CONFIG", str(config))
    assert main(["status", "--json"]) == 0
    assert str(config) in capsys.readouterr().err


def test_malformed_config_value_degrades_to_default(tmp_path, capsys):
    # A bad individual value degrades to that key's default with the same
    # one-line warning; it must not take the rest of the config down with it.
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\npaged_kv_quantization = "not-a-mode"\n',
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.exists is True
    assert loaded.model == "mtplx/example"
    assert loaded.paged_kv_quantization is None
    warning = capsys.readouterr().err
    assert str(config) in warning
    assert "paged_kv_quantization" in warning
