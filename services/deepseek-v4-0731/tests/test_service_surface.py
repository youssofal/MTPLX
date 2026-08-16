"""Static safety contracts for scripts that must not be run in unit tests."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import stat
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_candidate_is_distinct_from_live_service() -> None:
    launch = (ROOT / "launch_candidate.sh").read_text(encoding="utf-8")
    plist = (ROOT / "com.tea.deepseek-v4-0731.candidate.plist").read_text(encoding="utf-8")
    assert "com.tea.deepseek-v4-0731.candidate" in launch + plist
    assert "PORT=8081" in launch
    assert "8080" not in launch
    assert "launchctl" not in launch
    assert "/usr/bin/env -i" in launch
    assert "command environment override rejected" in launch
    assert "MTPLX_DSV4_0731_TEST_FIXTURE" in launch
    assert "candidate_entry.py" in launch
    assert "-m mtplx" not in launch


def test_candidate_config_pins_all_installation_identities() -> None:
    config = json.loads((ROOT / "candidate.json").read_text(encoding="utf-8"))
    assert config["candidate_port"] == 8081
    assert config["candidate_label"] == "com.tea.deepseek-v4-0731.candidate"
    assert config["encoding_source_revision"] == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    for key in ("model_config_sha256", "model_index_sha256", "trusted_python_sha256"):
        assert len(config[key]) == 64
    assert config["reviewed_ref"] == "refs/tags/mtplx-dsv4-0731-reviewed"
    assert config["artifact_validator_commit"] == "bbf02944aab3e17be754ba3c88d6aad3c488d10d"
    assert config["artifact_validator_path"] == "scripts/deepseek_v4_0731_artifact_check.py"
    assert config["artifact_validator_blob_sha256"] == (
        "672e3bafa8381c5264960d065730d9894b12f832eeb358922e0dd703042ac67b"
    )
    validator = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT.parents[1]),
            "cat-file",
            "blob",
            f'{config["artifact_validator_commit"]}:{config["artifact_validator_path"]}',
        ]
    )
    assert hashlib.sha256(validator).hexdigest() == config["artifact_validator_blob_sha256"]
    assert config["model_path"] == "/Users/davidtai/models/DeepSeek-V4-Flash-0731-oQ2e-mtp"
    assert config["model_config_sha256"] == (
        "6d0297a4329d55dccf3cd48fd168efea8044996245195d518a9e8aaa14906d3e"
    )
    assert config["model_index_sha256"] == (
        "9edcd0db7e6b8f0b8e02978d73c30083b2aa64c2e3a8fd77d3b776a4efb4bc91"
    )
    assert len(config["encoding_assets"]) == 9
    for relative, expected in config["encoding_assets"].items():
        assert hashlib.sha256((ROOT / "encoding" / relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT / "encoding/SHA256SUMS").read_bytes()).hexdigest() == config[
        "encoding_manifest_sha256"
    ]
    assert hashlib.sha256((ROOT / "candidate_entry.py").read_bytes()).hexdigest() == config[
        "candidate_entry_sha256"
    ]
    assert hashlib.sha256((ROOT / "com.tea.deepseek-v4-0731.candidate.plist").read_bytes()).hexdigest() == config[
        "candidate_plist_sha256"
    ]


def test_production_profile_pins_coherent_0731_k3_mlx032_and_256k() -> None:
    from production_entry import MODEL, MODEL_ID, serve_argv

    assert MODEL == Path(
        "/Users/davidtai/models/DeepSeek-V4-Flash-0731-2.4bit-mixed"
    )
    assert MODEL_ID == "mtplx-deepseek-v4-flash-0731-2.4bit-k3"
    argv = serve_argv()
    assert argv[:1] == ["serve"]
    assert argv[argv.index("--port") + 1] == "8080"
    assert argv[argv.index("--backend-id") + 1] == "deepseek_mtp"
    assert argv[argv.index("--context-window") + 1] == "262144"
    assert argv[argv.index("--depth") + 1] == "3"
    assert "--deepseek-v4-0731-optimized" in argv
    assert argv[argv.index("--session-cache-mode") + 1] == "off"
    assert argv[argv.index("--ssd-session-cache") + 1] == "off"

    launcher = (ROOT / "launch_production.sh").read_text(encoding="utf-8")
    assert "/tmp/mtplx-gpu-exclusive.lock" in launcher
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in launcher
    assert "mlx==0.32.0" in launcher
    assert "DeepSeek-V4-Flash-0731-2.4bit-mixed" in launcher


def test_command_override_is_rejected_except_for_nonstarting_fixture() -> None:
    launcher = ROOT / "launch_candidate.sh"
    fixture_env = {"PATH": os.environ["PATH"], "MTPLX_DSV4_0731_TEST_FIXTURE": "1"}
    fixture = subprocess.run(
        [str(launcher), "--print-command"], env=fixture_env, check=True, capture_output=True, text=True
    )
    assert "127.0.0.1:8081" in fixture.stdout

    rejected = subprocess.run(
        [str(launcher), "--print-command"],
        env={"PATH": os.environ["PATH"], "MTPLX_DSV4_0731_EXECUTABLE": "/bin/false"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "override rejected" in rejected.stderr


def test_cutover_requires_receipts_lock_identity_and_explicit_promotion() -> None:
    source = (ROOT / "promote_cutover.py").read_text(encoding="utf-8")
    for required in (
        "LOCK_NB",
        "--promote",
        "assert_candidate_receipt",
        "assert_live_identity",
        "/v1/models",
        "finish_reason",
        "SENSITIVE_KEY",
        "finally:",
        "_bootstrap(prior_snapshot.path)",
        "candidate_model_ids",
        "attest_process_identity",
    ):
        assert required in source
    bootstrap = source.index("_bootstrap(target_snapshot.path)")
    new_identity = source.index("_wait_for_process_identity(", bootstrap)
    readiness = source.index("_verify_live_ready(", new_identity)
    assert bootstrap < new_identity < readiness
    assert 'content.strip() != "READY"' in source
    assert "_bootstrap(prior_snapshot.path)" in source
    unchanged = source.index("prior_snapshot.assert_source_unchanged()")
    bootout = source.index('_bootout(current["label"])')
    assert unchanged < bootout


def test_process_attestation_rejects_plist_not_loaded_by_launchd(monkeypatch, tmp_path) -> None:
    import promote_cutover

    plist = tmp_path / "prior.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.tea.qwen",
                "ProgramArguments": ["/bin/false", "--reviewed"],
            }
        )
    )

    def fake_command(*argv: str) -> str:
        if argv[0] == "/bin/launchctl":
            return f"""gui/501/com.tea.qwen = {{
\tpath = {plist}
\tprogram = /bin/true
\targuments = {{
\t\t/bin/true
\t\t--unrelated
\t}}
\tpid = 4242
}}"""
        return "p4242\nn127.0.0.1:8080\n"

    monkeypatch.setattr(promote_cutover, "_command", fake_command)
    with pytest.raises(promote_cutover.PromotionError, match="ProgramArguments"):
        promote_cutover.attest_process_identity(label="com.tea.qwen", plist=plist)


def test_listener_identity_selects_exact_loopback_backend(monkeypatch) -> None:
    import promote_cutover

    monkeypatch.setattr(
        promote_cutover,
        "_command",
        lambda *_argv: (
            "p3098\nn10.8.0.2:8080\n"
            "p14242\nn127.0.0.1:8080\n"
        ),
    )

    assert promote_cutover._listener_pid(8080) == 14242


@pytest.mark.parametrize(
    "listeners",
    [
        "p3098\nn*:8080\np14242\nn127.0.0.1:8080\n",
        "p14242\nn127.0.0.1:8080\np14243\nn127.0.0.1:8080\n",
    ],
)
def test_listener_identity_rejects_wildcard_or_duplicate_loopback_owners(
    monkeypatch, listeners: str
) -> None:
    import promote_cutover

    monkeypatch.setattr(promote_cutover, "_command", lambda *_argv: listeners)
    with pytest.raises(promote_cutover.PromotionError, match="loopback listener"):
        promote_cutover._listener_pid(8080)


def test_prior_plist_snapshot_is_durable_reignorable_and_safely_cleaned(
    monkeypatch, tmp_path
) -> None:
    import promote_cutover
    from promote_cutover import PromotionError, plist_snapshot

    prior = tmp_path / "prior.plist"
    original = plistlib.dumps(
        {
            "Label": "com.tea.qwen",
            "ProgramArguments": ["/bin/false"],
            "EnvironmentVariables": {"SAFE": "reviewed"},
            "KeepAlive": True,
            "StandardOutPath": "/tmp/reviewed.out",
            "StandardErrorPath": "/tmp/reviewed.err",
        }
    )
    prior.write_bytes(original)
    with plist_snapshot(prior) as snapshot:
        snapshot_path = snapshot.path
        assert snapshot_path.stat().st_dev == prior.stat().st_dev
        assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o400
        assert snapshot_path.stat().st_uid == os.getuid()
        assert not snapshot_path.is_symlink()
        assert stat.S_IMODE(snapshot_path.parent.stat().st_mode) == 0o700
        snapshot.assert_source_unchanged()
        replacement = tmp_path / "attacker.plist"
        replacement.write_bytes(
            plistlib.dumps(
                {
                    "Label": "com.tea.qwen",
                    "ProgramArguments": ["/bin/false"],
                    "EnvironmentVariables": {"SAFE": "attacker"},
                    "KeepAlive": False,
                    "StandardOutPath": "/tmp/attacker.out",
                    "StandardErrorPath": "/tmp/attacker.err",
                }
            )
        )
        os.replace(replacement, prior)
        with pytest.raises(PromotionError, match="changed since snapshot"):
            snapshot.assert_source_unchanged()
        snapshot.assert_snapshot_intact()
        assert snapshot.path.read_bytes() == original

    assert snapshot_path.is_file()

    def fake_command(*argv: str) -> str:
        if argv[0] == "/bin/launchctl":
            return f"""gui/501/com.tea.qwen = {{
\tpath = {snapshot_path}
\tprogram = /bin/false
\targuments = {{
\t\t/bin/false
\t}}
\tpid = 4242
}}"""
        return "p4242\nn127.0.0.1:8080\n"

    monkeypatch.setattr(promote_cutover, "_command", fake_command)
    repeated = promote_cutover.attest_process_identity(
        label="com.tea.qwen", plist=snapshot_path
    )
    assert repeated["plist_sha256"] == hashlib.sha256(original).hexdigest()
    monkeypatch.setattr(
        promote_cutover,
        "_launchctl_job_if_loaded",
        lambda _label: {"path": snapshot_path},
    )
    with pytest.raises(PromotionError, match="still loaded"):
        snapshot.cleanup_if_unloaded()
    monkeypatch.setattr(
        promote_cutover, "_launchctl_job_if_loaded", lambda _label: None
    )
    snapshot.cleanup_if_unloaded()
    assert not snapshot_path.exists()


def test_post_commit_snapshot_cleanup_failure_does_not_roll_back_production(
    monkeypatch, tmp_path, capsys
) -> None:
    import promote_cutover

    prior_plist = tmp_path / "prior.plist"
    prior_plist.write_bytes(
        plistlib.dumps(
            {
                "Label": promote_cutover.PRIOR_LIVE_LABEL,
                "ProgramArguments": ["/bin/false", "--prior"],
            }
        )
    )
    target_plist = tmp_path / "production.plist"
    target_plist.write_bytes(
        plistlib.dumps(
            {
                "Label": promote_cutover.PRODUCTION_LABEL,
                "ProgramArguments": ["/bin/false", "--production"],
            }
        )
    )
    reviewed_commit = "c" * 40
    candidate = _passing_candidate_receipt()
    preflight = candidate["candidate_preflight"]
    preflight.update(
        {
            "plist_sha256": promote_cutover.CANDIDATE_PLIST_SHA256,
            "encoding_asset_set_sha256": promote_cutover.ENCODING_ASSET_SET_SHA256,
            "reviewed_commit": reviewed_commit,
            "model_config_sha256": promote_cutover.MODEL_CONFIG_SHA256,
            "model_index_sha256": promote_cutover.MODEL_INDEX_SHA256,
            "promotion_target": {
                "label": promote_cutover.PRODUCTION_LABEL,
                "plist_sha256": hashlib.sha256(target_plist.read_bytes()).hexdigest(),
            },
        }
    )
    candidate_receipt = tmp_path / "candidate.json"
    candidate_receipt.write_text(json.dumps(candidate), encoding="utf-8")
    live = {
        "schema": "mtplx.live-identity.v1",
        "label": promote_cutover.PRIOR_LIVE_LABEL,
        "pid": 14242,
        "listener_port": 8080,
        "plist_sha256": hashlib.sha256(prior_plist.read_bytes()).hexdigest(),
        "model_ids": list(promote_cutover.ALLOWED_PRIOR_MODEL_IDS),
    }
    live_attestation = tmp_path / "live.json"
    live_attestation.write_text(json.dumps(live), encoding="utf-8")
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(promote_cutover, "_verify_candidate_signature", lambda *_args: None)
    monkeypatch.setattr(promote_cutover, "_command", lambda *_args: reviewed_commit)
    monkeypatch.setattr(promote_cutover, "exclusive_gpu_lock", nullcontext)
    monkeypatch.setattr(promote_cutover, "attest_live", lambda **_kwargs: dict(live))
    monkeypatch.setattr(
        promote_cutover,
        "_bootout",
        lambda label: events.append(("bootout", label)),
    )
    monkeypatch.setattr(
        promote_cutover,
        "_bootstrap",
        lambda path: events.append(("bootstrap", Path(path))),
    )
    monkeypatch.setattr(
        promote_cutover,
        "_wait_for_process_identity",
        lambda **kwargs: {
            "plist_sha256": kwargs["plist"].sha256,
        },
    )
    monkeypatch.setattr(
        promote_cutover,
        "_verify_live_ready",
        lambda model_ids: events.append(("ready", tuple(model_ids))),
    )

    def fail_after_unlink(snapshot) -> None:
        events.append(("cleanup", snapshot.label))
        if snapshot.label == promote_cutover.PRIOR_LIVE_LABEL:
            snapshot.path.unlink()
            raise OSError("injected failure immediately after unlink")

    monkeypatch.setattr(promote_cutover.PlistSnapshot, "cleanup_if_unloaded", fail_after_unlink)
    args = SimpleNamespace(
        promote=True,
        candidate_receipt=candidate_receipt,
        candidate_signature=tmp_path / "candidate.json.sig",
        live_attestation=live_attestation,
        live_plist=prior_plist,
        production_plist=target_plist,
        production_label=promote_cutover.PRODUCTION_LABEL,
    )

    promote_cutover.promote(args)

    assert ("ready", tuple(candidate["candidate_smoke"]["candidate_model_ids"])) in events
    assert ("cleanup", promote_cutover.PRIOR_LIVE_LABEL) in events
    assert ("bootout", promote_cutover.PRODUCTION_LABEL) not in events
    assert any(
        event == "bootstrap"
        and path.name.startswith(promote_cutover.PRODUCTION_LABEL)
        for event, path in events
    )
    assert not any(
        event == "bootstrap" and path.name.startswith(promote_cutover.PRIOR_LIVE_LABEL)
        for event, path in events
    )
    assert "promotion committed; prior snapshot cleanup was incomplete" in capsys.readouterr().err


def test_snapshot_cleanup_distinguishes_absent_job_from_probe_failure(
    monkeypatch, tmp_path
) -> None:
    import promote_cutover

    plist = tmp_path / "prior.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.tea.qwen",
                "ProgramArguments": ["/bin/false"],
            }
        )
    )
    with promote_cutover.plist_snapshot(plist) as snapshot:
        snapshot_path = snapshot.path

    monkeypatch.setattr(
        promote_cutover.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=70,
            stdout="",
            stderr="launchd transport failure",
        ),
    )
    with pytest.raises(promote_cutover.PromotionError, match="safely determine"):
        snapshot.cleanup_if_unloaded()
    assert snapshot_path.is_file()

    monkeypatch.setattr(
        promote_cutover.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=113,
            stdout="",
            stderr='Could not find service "com.tea.qwen" in domain for user gui: 501',
        ),
    )
    snapshot.cleanup_if_unloaded()
    assert not snapshot_path.exists()


def test_launcher_invokes_exact_reviewed_artifact_validator() -> None:
    source = (ROOT / "launch_candidate.sh").read_text(encoding="utf-8")
    assert "bbf02944aab3e17be754ba3c88d6aad3c488d10d" in source
    assert "672e3bafa8381c5264960d065730d9894b12f832eeb358922e0dd703042ac67b" in source
    assert "scripts/deepseek_v4_0731_artifact_check.py" in source
    assert "git -C \"$WORKTREE\" cat-file blob" in source
    assert '"$PYTHON" - "$MODEL"' in source


def test_server_construction_installs_verified_0731_encoder() -> None:
    from candidate_entry import install_candidate_surface
    from mtplx.server import openai as openai_server

    def stock(*_args, **_kwargs):
        return [999]

    server = SimpleNamespace(
        _encode_messages=stock,
        _parse_generated_tool_calls_or_content=lambda *_args, **_kwargs: (None, None),
        omlx_extract_tool_calls_with_thinking=lambda *_args, **_kwargs: None,
        _ToolAwareContentStreamTranslator=openai_server._ToolAwareContentStreamTranslator,
        _stream_tool_call_deltas=openai_server._stream_tool_call_deltas,
    )
    receipt = install_candidate_surface(server)
    assert server._encode_messages is not stock
    assert receipt["encoder"] == "deepseek-v4-flash-0731-official"
    assert server._template_hash(None).startswith("deepseek-v4-flash-0731-official:")
    assert server._apply_chat_template_profile(None, None) == {
        "profile": "deepseek-v4-flash-0731-official",
        "source": "official_python_encoder",
        "path": None,
        "applied": True,
        "sha256": receipt["asset_set_sha256"],
    }

    class Tokenizer:
        def __init__(self) -> None:
            self.encoded = ""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            self.encoded = text
            return list(text.encode("utf-8"))

    tokenizer = Tokenizer()
    observability: dict[str, object] = {}
    ids = server._encode_messages(
        tokenizer,
        [SimpleNamespace(role="user", content="hello", tool_calls=None)],
        enable_thinking=True,
        reasoning_effort="high",
        tools=None,
        template_observability=observability,
    )
    assert ids == list(tokenizer.encoded.encode("utf-8"))
    assert "<｜Assistant｜><think>" in tokenizer.encoded
    assert observability == {
        "backend_chat_encoding": "deepseek-v4-flash-0731-official",
        "encoding_source_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    }

    vector = (ROOT / "encoding/tests/test_output_1.txt").read_text(encoding="utf-8")
    marker = "<｜Assistant｜><think>"
    start = vector.find(marker) + len(marker)
    end = vector.find("<｜User｜>", start)
    thinking, regular = vector[start:end].split("</think>", 1)
    extraction = server.omlx_extract_tool_calls_with_thinking(thinking, regular, tokenizer, [])
    assert extraction.parser_source == "deepseek_v4_0731_official"
    assert extraction.tool_calls[0]["function"]["name"] == "get_weather"
    assert extraction.tool_calls[0]["id"].startswith("call_")

    no_tool = server.omlx_extract_tool_calls_with_thinking("", "READY", tokenizer, [])
    assert no_tool.parser_source == "deepseek_v4_0731_official"
    assert no_tool.status == "no_tool"
    assert no_tool.cleaned_text == "READY"
    assert no_tool.tool_calls is None
    assert server._parse_generated_tool_calls_or_content("READY", tools=[]) == (None, None)

    translator = server._ToolAwareContentStreamTranslator(
        tools=[], argument_chunk_chars=16, tokenizer=tokenizer
    )
    midpoint = len(regular) // 2
    assert translator.feed("content", regular[:midpoint]) == []
    assert translator.feed("content", regular[midpoint:]) == []
    deltas = translator.finish()
    assert translator.suppressed_tool_markup is True
    assert translator.tool_calls[0]["function"]["name"] == "get_weather"
    assert any("tool_calls" in delta for delta in deltas)

    preamble = "I will check."
    nonstream = server.omlx_extract_tool_calls_with_thinking(
        "", preamble + regular, tokenizer, []
    )
    translator = server._ToolAwareContentStreamTranslator(
        tools=[], argument_chunk_chars=16, tokenizer=tokenizer
    )
    # The ordinary preamble arrives before there is any evidence of DSML.
    deltas = translator.feed("content", preamble)
    deltas.extend(translator.feed("content", regular[:midpoint]))
    deltas.extend(translator.feed("content", regular[midpoint:]))
    deltas.extend(translator.finish())
    streamed_content = "".join(delta.get("content", "") for delta in deltas)
    assert streamed_content == nonstream.cleaned_text == preamble
    assert translator.tool_calls == nonstream.tool_calls
    assert any("tool_calls" in delta for delta in deltas)
    assert "<｜DSML｜" not in json.dumps(deltas, ensure_ascii=False)

    split_translator = server._ToolAwareContentStreamTranslator(
        tools=[], argument_chunk_chars=16, tokenizer=tokenizer
    )
    split_deltas: list[dict[str, object]] = []
    for character in preamble + regular:
        split_deltas.extend(split_translator.feed("content", character))
    split_deltas.extend(split_translator.finish())
    assert "".join(str(delta.get("content", "")) for delta in split_deltas) == preamble
    assert split_translator.tool_calls == nonstream.tool_calls
    assert "<｜DSML｜" not in json.dumps(split_deltas, ensure_ascii=False)


def test_no_tools_api_stream_split_chunks_never_release_dsml() -> None:
    """The candidate's stream boundary sanitizes no-tools API content too."""
    from candidate_entry import install_candidate_surface
    from mtplx.server import openai as openai_server

    server = SimpleNamespace(
        _encode_messages=lambda *_args, **_kwargs: [],
        _parse_generated_tool_calls_or_content=lambda *_args, **_kwargs: (None, None),
        omlx_extract_tool_calls_with_thinking=lambda *_args, **_kwargs: None,
        _ToolAwareContentStreamTranslator=openai_server._ToolAwareContentStreamTranslator,
        _stream_tool_call_deltas=openai_server._stream_tool_call_deltas,
        _stream_splitter_for_state=lambda *_args, **kwargs: openai_server._ThinkingContentStreamSplitter(
            thinking_enabled=kwargs["thinking_enabled"],
            suppress_orphan_tool_markup=kwargs.get("suppress_orphan_tool_markup", False),
        ),
    )
    install_candidate_surface(server)
    splitter = server._stream_splitter_for_state(
        SimpleNamespace(), thinking_enabled=False, suppress_orphan_tool_markup=True
    )
    wire_chunks: list[tuple[str, str]] = []
    raw = "Preamble.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"x\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    for character in raw:
        wire_chunks.extend(splitter.feed(character))
    wire_chunks.extend(splitter.finish())
    assert "".join(text for field, text in wire_chunks if field == "content") == "Preamble."
    assert "<｜DSML｜" not in json.dumps(wire_chunks, ensure_ascii=False)

    # The real endpoint performs both reads unconditionally during streaming
    # finalization, including when the request declared no tools.
    generated = {"stats": {}}
    state = SimpleNamespace(last_metrics=[{}])
    generated["stats"]["reasoning_reentries"] = splitter.reentry_count
    state.last_metrics[-1]["reasoning_reentries"] = splitter.reentry_count
    assert generated["stats"]["reasoning_reentries"] == 0
    assert state.last_metrics[-1]["reasoning_reentries"] == 0


@pytest.mark.parametrize("malformed", [False, True])
def test_no_tools_nonstream_endpoint_uses_official_dsml_sanitizer(monkeypatch, malformed: bool) -> None:
    from fastapi.testclient import TestClient
    from candidate_entry import install_candidate_surface
    from mtplx.server import openai as openai_server

    tests_root = ROOT.parents[1] / "tests"
    monkeypatch.syspath_prepend(str(tests_root))
    from test_server_openai import _fake_generation, _fake_state  # noqa: PLC0415

    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [ord(character) for character in text]

        def decode(self, tokens, **_kwargs) -> str:
            return "".join(chr(int(token)) for token in tokens)

    state = _fake_state()
    state.args.stats_footer = False
    state.runtime.tokenizer = Tokenizer()
    vector = (ROOT / "encoding/tests/test_output_1.txt").read_text(encoding="utf-8")
    marker = "<｜Assistant｜><think>"
    start = vector.find(marker) + len(marker)
    end = vector.find("<｜User｜>", start)
    _thinking, dsml = vector[start:end].split("</think>", 1)
    raw = "Preamble.\n\n" + ("<｜DSML｜tool_calls><｜DSML｜invoke" if malformed else dsml)

    replaced = (
        "_encode_messages",
        "_parse_generated_tool_calls_or_content",
        "omlx_extract_tool_calls_with_thinking",
        "_ToolAwareContentStreamTranslator",
        "_stream_splitter_for_state",
        "_strip_orphan_tool_markup",
        "_normalize_reasoning_effort",
        "_reasoning_effort_for_state",
        "_apply_chat_template_profile",
        "_template_hash",
        "_template_supports_scoped_reasoning",
        "_DSV4_0731_ENCODER_INSTALLED",
    )
    missing = object()
    originals = {name: getattr(openai_server, name, missing) for name in replaced}
    try:
        install_candidate_surface(openai_server)
        monkeypatch.setattr(
            openai_server,
            "_run_generation",
            lambda *_args, **_kwargs: _fake_generation(raw),
        )
        response = TestClient(openai_server.create_app(state)).post(
            "/v1/chat/completions",
            headers={"x-mtplx-cache-mode": "bypass"},
            json={
                "messages": [{"role": "user", "content": "Do not use tools."}],
                "enable_thinking": False,
                "max_tokens": 64,
            },
        )
    finally:
        for name, original in originals.items():
            if original is missing:
                delattr(openai_server, name)
            else:
                setattr(openai_server, name, original)

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == "Preamble."
    assert "<｜DSML｜" not in response.text


def test_allowed_signers_is_digest_pinned_owned_and_not_writable() -> None:
    import promote_cutover

    assert promote_cutover.ALLOWED_SIGNERS_SHA256 == (
        "003f258613fe308134ef184e52988a082a3376655d6b44f526017d7d71c7f843"
    )
    with tempfile.TemporaryDirectory() as directory:
        allowed = Path(directory) / "allowed-signers"
        allowed.write_text("mtplx-deepseek-v4-0731-candidate ssh-ed25519 AAAA\n", encoding="utf-8")
        allowed.chmod(0o600)
        original_path = promote_cutover.ALLOWED_SIGNERS
        original_digest = promote_cutover.ALLOWED_SIGNERS_SHA256
        try:
            promote_cutover.ALLOWED_SIGNERS = allowed
            promote_cutover.ALLOWED_SIGNERS_SHA256 = hashlib.sha256(allowed.read_bytes()).hexdigest()
            promote_cutover._assert_allowed_signers_trusted()

            allowed.write_text("mutated\n", encoding="utf-8")
            with pytest.raises(promote_cutover.PromotionError, match="digest"):
                promote_cutover._assert_allowed_signers_trusted()

            allowed.write_text("mtplx-deepseek-v4-0731-candidate ssh-ed25519 AAAA\n", encoding="utf-8")
            allowed.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IWGRP)
            with pytest.raises(promote_cutover.PromotionError, match="permissions"):
                promote_cutover._assert_allowed_signers_trusted()

            allowed.chmod(0o600)
            original_getuid = promote_cutover.os.getuid
            try:
                promote_cutover.os.getuid = lambda: original_getuid() + 1
                with pytest.raises(promote_cutover.PromotionError, match="owner"):
                    promote_cutover._assert_allowed_signers_trusted()
            finally:
                promote_cutover.os.getuid = original_getuid
        finally:
            promote_cutover.ALLOWED_SIGNERS = original_path
            promote_cutover.ALLOWED_SIGNERS_SHA256 = original_digest


@pytest.mark.parametrize(
    "model_id",
    [
        "/Users/davidtai/models/private",
        "~/models/private",
        r"C:\\Users\\davidtai\\models\\private",
        "file:///Users/davidtai/models/private",
        "deepseek-v4-0731/../../private",
        "..%2Fprivate",
    ],
)
def test_candidate_receipt_rejects_path_like_model_ids(model_id: str) -> None:
    from promote_cutover import PromotionError, assert_candidate_receipt

    receipt = _passing_candidate_receipt()
    receipt["candidate_smoke"]["candidate_model_ids"] = [model_id]
    with pytest.raises(PromotionError, match="sensitive|model ID"):
        assert_candidate_receipt(receipt)


@pytest.mark.parametrize("path_value", ["~/private", r"C:\\private", "file:///private", "a/../private"])
def test_candidate_receipt_recursively_rejects_path_like_values(path_value: str) -> None:
    from promote_cutover import PromotionError, assert_candidate_receipt

    receipt = _passing_candidate_receipt()
    receipt["candidate_preflight"]["promotion_target"]["label"] = path_value
    with pytest.raises(PromotionError, match="sensitive"):
        assert_candidate_receipt(receipt)


@pytest.mark.parametrize(
    "label",
    [
        "com.tea.prod(/etc/x)",
        "../com.tea.prod",
        "file://com.tea.prod",
        r"C:\\com.tea.prod",
        "~/com.tea.prod",
        "com.tea.other",
    ],
)
def test_candidate_receipt_rejects_nonallowlisted_production_label(label: str) -> None:
    from promote_cutover import PromotionError, assert_candidate_receipt

    receipt = _passing_candidate_receipt()
    receipt["candidate_preflight"]["promotion_target"]["label"] = label
    with pytest.raises(PromotionError, match="sensitive|production label"):
        assert_candidate_receipt(receipt)


def test_live_receipt_rejects_nonallowlisted_label_and_model_id() -> None:
    from promote_cutover import PromotionError, assert_live_identity

    live = {
        "schema": "mtplx.live-identity.v1",
        "label": "com.tea.qwen",
        "pid": 42,
        "listener_port": 8080,
        "plist_sha256": "a" * 64,
        "model_ids": ["mtplx-qwen36-27b-optimized-quality"],
    }
    for field, value in (
        ("label", "com.tea.qwen(/etc/x)"),
        ("model_ids", ["../private-model"]),
    ):
        altered = {**live, field: value}
        with pytest.raises(PromotionError, match="allowlist"):
            assert_live_identity(altered, altered)

    for field, value in (
        ("pid", "42"),
        ("listener_port", 9999),
        ("plist_sha256", "com.tea.prod(/etc/x)"),
    ):
        altered = {**live, field: value}
        with pytest.raises(PromotionError, match="invalid"):
            assert_live_identity(altered, altered)


def _passing_candidate_receipt() -> dict[str, object]:
    return {
        "schema": "mtplx.dsv4-0731-candidate.v1",
        "candidate_preflight": {
            "ok": True,
            "label": "com.tea.deepseek-v4-0731.candidate",
            "port": 8081,
            "plist_sha256": "a" * 64,
            "encoding_source_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "encoding_asset_set_sha256": "b" * 64,
            "reviewed_commit": "c" * 40,
            "model_config_sha256": "d" * 64,
            "model_index_sha256": "e" * 64,
            "promotion_target": {"label": "com.tea.deepseek-v4-0731.production", "plist_sha256": "b" * 64},
        },
        "candidate_smoke": {
            "ok": True,
            "models_ok": True,
            "ready": True,
            "finish_reason": "stop",
            "candidate_model_ids": ["deepseek-v4-0731-candidate"],
        },
    }


@pytest.mark.parametrize("forbidden_key, forbidden_value", [
    ("stdout", "must never be retained"),
    ("prompt", "must never be retained"),
    ("tools", []),
    ("secret", "must never be retained"),
    ("argv", ["must never be retained"]),
    ("env", {"MUST_NEVER": "be retained"}),
    ("model_path", "/Users/davidtai/models/private"),
    ("note", '{"messages":[{"role":"user"}]}'),
    ("note", "Bearer abcdefghijklmnop"),
])
def test_candidate_receipt_rejects_sensitive_capture(forbidden_key: str, forbidden_value: object) -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from promote_cutover import PromotionError, assert_candidate_receipt  # noqa: PLC0415

    receipt = {
        "candidate_preflight": {
            "ok": True,
            "label": "com.tea.deepseek-v4-0731.candidate",
            "port": 8081,
            "promotion_target": {"label": "com.tea.deepseek-v4-0731.production", "plist_sha256": "a" * 64},
        },
        "candidate_smoke": {"ok": True, "models_ok": True, "ready": True, "finish_reason": "stop"},
        forbidden_key: forbidden_value,
    }
    with pytest.raises(PromotionError, match="sensitive"):
        assert_candidate_receipt(receipt)


def test_scrubbed_passing_candidate_receipt_is_accepted() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from promote_cutover import assert_candidate_receipt  # noqa: PLC0415

    assert_candidate_receipt(
        {
            "schema": "mtplx.dsv4-0731-candidate.v1",
            "candidate_preflight": {
                "ok": True,
                "label": "com.tea.deepseek-v4-0731.candidate",
                "port": 8081,
                "plist_sha256": "a" * 64,
                "encoding_source_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
                "encoding_asset_set_sha256": "b" * 64,
                "reviewed_commit": "c" * 40,
                "model_config_sha256": "d" * 64,
                "model_index_sha256": "e" * 64,
                "promotion_target": {"label": "com.tea.deepseek-v4-0731.production", "plist_sha256": "b" * 64},
            },
            "candidate_smoke": {
                "ok": True,
                "models_ok": True,
                "ready": True,
                "finish_reason": "stop",
                "candidate_model_ids": ["deepseek-v4-0731-candidate"],
            },
        }
    )


def test_candidate_receipt_rejects_unknown_nested_fields() -> None:
    from promote_cutover import PromotionError, assert_candidate_receipt

    receipt = {
        "schema": "mtplx.dsv4-0731-candidate.v1",
        "candidate_preflight": {
            "ok": True,
            "label": "com.tea.deepseek-v4-0731.candidate",
            "port": 8081,
            "plist_sha256": "a" * 64,
            "encoding_source_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "encoding_asset_set_sha256": "b" * 64,
            "reviewed_commit": "c" * 40,
            "model_config_sha256": "d" * 64,
            "model_index_sha256": "e" * 64,
            "promotion_target": {
                "label": "com.tea.deepseek-v4-0731.production",
                "plist_sha256": "f" * 64,
                "unexpected": True,
            },
        },
        "candidate_smoke": {
            "ok": True,
            "models_ok": True,
            "ready": True,
            "finish_reason": "stop",
            "candidate_model_ids": ["deepseek-v4-0731-candidate"],
        },
    }
    with pytest.raises(PromotionError, match="strict receipt schema"):
        assert_candidate_receipt(receipt)
