from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.cli import build_parser
from mtplx.commands import public as public_commands
from mtplx.commands.public import (
    DFLASH2_BACKEND_ID,
    _launcher_python,
    _prepare_dflash2_args,
    _serve_dry_run_payload,
    _validate_public_depth,
    resolve_dflash2_bundle_paths,
)
from mtplx.dflash2_bundle import resolve_dflash2_bundle_paths as canonical_resolver
from mtplx.server import openai
from mtplx.server.openai import parse_args


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "qwen38-dflash2"
    target = root / "target"
    draft = root / "dflash2"
    target.mkdir(parents=True)
    draft.mkdir()
    (target / "config.json").write_text('{"model_type":"qwen3_next"}', encoding="utf-8")
    (root / "mtplx_dflash2.json").write_text(
        json.dumps(
            {
                "backend": "dflash2",
                "target_revision": "target-sha",
                "draft_revision": "dflash-sha",
                "layout": {"target": "target", "draft": "dflash2"},
                "provenance": {
                    "target_repo": "Qwen/Qwen3.8-27B",
                    "draft_repo": "z-lab/Qwen3.8-27B-DFlash2",
                    "algorithm_repo": "z-lab/dflash",
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_bundle_resolves_and_preserves_provenance(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    resolved = resolve_dflash2_bundle_paths(root)
    assert resolved is not None
    assert resolved["target_model"] == str(root / "target")
    assert resolved["draft_model"] == str(root / "dflash2")
    assert resolved["metadata"]["draft_revision"] == "dflash-sha"
    assert resolved["metadata"]["provenance"]["algorithm_repo"] == "z-lab/dflash"


def test_resolvers_are_canonical_and_malformed_bundle_fails_closed(tmp_path: Path) -> None:
    assert resolve_dflash2_bundle_paths is canonical_resolver
    assert openai.resolve_dflash2_bundle_paths is canonical_resolver
    root = tmp_path / "broken"
    root.mkdir()
    (root / "mtplx_dflash2.json").write_text('{"backend":"dflash2"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must provide target"):
        canonical_resolver(root)
    with pytest.raises(ValueError, match="must provide target"):
        _prepare_dflash2_args(
            SimpleNamespace(
                model=str(root),
                backend_id=None,
                generation_mode="mtp",
                no_mtp=False,
                _cli_flags=set(),
            )
        )
    with pytest.raises(ValueError, match="must provide target"):
        parse_args(["--model", str(root), "--warmup-tokens", "0"])


@pytest.mark.parametrize(
    "manifest",
    ["{not-json", '{"backend":"other","schemaVersion":1}'],
)
def test_invalid_auto_detect_manifest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: str
) -> None:
    root = tmp_path / "invalid"
    root.mkdir()
    (root / "mtplx_dflash2.json").write_text(manifest, encoding="utf-8")
    monkeypatch.setattr(
        public_commands, "resolve_dflash2_bundle_paths", lambda _root: None
    )

    with pytest.raises(ValueError, match="invalid DFlash2 bundle"):
        _prepare_dflash2_args(
            SimpleNamespace(
                model=str(root),
                backend_id=None,
                generation_mode="mtp",
                no_mtp=False,
                _cli_flags=set(),
            )
        )


def test_public_defaults_and_explicit_sampler(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    args = SimpleNamespace(
        model=str(root),
        backend_id=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        draft_temperature=None,
        draft_top_p=None,
        draft_top_k=None,
        generation_mode="mtp",
        no_mtp=False,
        _cli_flags=set(),
    )
    _prepare_dflash2_args(args)
    assert args.backend_id == DFLASH2_BACKEND_ID
    assert (args.temperature, args.top_p, args.top_k) == (1.0, 0.95, 20)

    explicit_values = vars(args).copy()
    explicit_values.update(temperature=0.7, _cli_flags={"temperature"})
    explicit = SimpleNamespace(**explicit_values)
    _prepare_dflash2_args(explicit)
    assert explicit.temperature == 0.7


def test_server_parser_selects_dflash2_and_defaults(tmp_path: Path) -> None:
    args = parse_args(["--model", str(_bundle(tmp_path)), "--warmup-tokens", "0"])
    assert args.backend_id == DFLASH2_BACKEND_ID
    assert (args.temperature, args.top_p, args.top_k) == (1.0, 0.95, 20)
    assert (args.depth, args.draft_block_size) == (5, 5)


def test_dflash2_explicit_ar_routes_to_target_without_mtp(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    args = parse_args(
        [
            "--model",
            str(root),
            "--generation-mode",
            "ar",
            "--warmup-tokens",
            "0",
        ]
    )
    assert args.model == str(root / "target")
    assert args.backend_id == "qwen3_next"
    assert args.generation_mode == "ar"
    assert args.load_mtp is False


def test_public_ask_dflash2_depth_five_passes_validation(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    args = build_parser().parse_args(
        ["ask", "--model", str(root), "--backend-id", "dflash2", "prompt"]
    )
    _prepare_dflash2_args(args)
    assert args.depth == 5
    assert _validate_public_depth(args, printer=lambda _line: None) is None


def test_dflash2_no_mtp_routes_to_target(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    args = build_parser().parse_args(
        ["serve", "--model", str(root), "--no-mtp"]
    )
    _prepare_dflash2_args(args)
    assert args.model == str(root / "target")
    assert args.backend_id == "qwen3_next"
    assert args.load_mtp is False


def test_serve_dry_run_proves_selected_backend(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model_id="qwen38",
        fan_mode="default",
        generation_mode="mtp",
        backend_id=DFLASH2_BACKEND_ID,
        depth=5,
        draft_block_size=5,
        download=False,
    )
    payload = _serve_dry_run_payload(
        args,
        runtime_model=str(root),
        profile_name="sustained",
        model_id="qwen38",
        generation_mode="mtp",
        cmd=["mtplx", "serve", "--backend-id", DFLASH2_BACKEND_ID],
        env={},
    )
    assert payload["backend_id"] == DFLASH2_BACKEND_ID
    assert payload["draft_block_size"] == 5


def test_public_commands_accept_dflash2_backend() -> None:
    parser = build_parser()
    for command in ("serve", "ask", "start"):
        args = parser.parse_args([command, "--model", "bundle", "--backend-id", "dflash2"])
        assert args.backend_id == DFLASH2_BACKEND_ID


def test_llama_cpp_is_not_an_implicit_fallback() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve", "--model", "bundle", "--backend-id", "llama.cpp"])
    with pytest.raises(ValueError, match="llama.cpp"):
        _prepare_dflash2_args(args)


def test_generic_brew_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("MTPLX_BREW_VENV", str(venv))
    assert _launcher_python() == str(python)


def test_docs_record_upstream_gates() -> None:
    docs = Path(__file__).parents[1] / "docs" / "dflash2.md"
    text = docs.read_text(encoding="utf-8")
    for marker in (
        "dflash2",
        "dflash==0.1.0",
        "dflash generate mlx --model",
        "mtplx dflash-mlx-baseline --model",
        "mtplx mtp-depth-sweep --model",
        "$(brew --prefix mtplx)/bin/mtplx",
        "MTPLX_BREW_VENV",
        "sha256",
        "target-only AR",
        "llama.cpp",
        "#159",
        "#160",
    ):
        assert marker in text
    assert "dflash benchmark" not in text
    assert "/Users/" not in text
