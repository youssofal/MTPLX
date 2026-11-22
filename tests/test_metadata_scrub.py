"""Publish-time scrubbing of machine-identifying artifact metadata.

The fixtures below mirror the shapes found in real local artifacts:
``mtplx_runtime.json``'s ``forge_provenance.forge_inputs`` and a conversion
manifest's ``source``/``target`` paths.
"""

from __future__ import annotations

import copy

from mtplx.metadata_scrub import (
    REDACTED_PATH,
    runtime_metadata_leaks,
    scrub_path_value,
    scrub_runtime_metadata,
)


def _runtime_fixture() -> dict:
    """Mirrors ~/.cache/huggingface/hy3-q4-mlx-mtp/mtplx_runtime.json."""

    return {
        "arch_id": "hy_v3",
        "artifact_role": "forge-local",
        "base_trunk": "tencent/Hy3",
        "mtp_depth_max": 3,
        "mtp_sidecar": "mtp.safetensors",
        "mtp_sidecar_file": "mtp.safetensors",
        "mtplx_version": "2.0.2",
        "forge_provenance": {
            "forge_inputs": {
                "bf16_head_source_path": "/Users/davidtai/.cache/huggingface/hy3-mtp-layer80/layer80-bf16.safetensors",
                "checkpoint_path": "/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp/conversion-checkpoint.jsonl",
                "layout_oracle_path": "/Users/davidtai/.cache/huggingface/hub/models--pipenetwork--Hy3-4bit/snapshots/160619d3",
                "output_path": "/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp",
                "source_path": "/Users/davidtai/.cache/huggingface/hy3-mtp-layer80",
            },
            "forge_recipe": {"mtp_policy": "keep_bf16", "bits": 4},
            "forged_at": "2026-07-11T15:23:33Z",
            "forged_locally": True,
            "mtplx_version": "2.0.2",
            "intended_hf_repo": "davidtai/hy3-q4-mlx-mtp",
            "published_to_hf": None,
            "source_format": "bf16_native",
            "source_repo": "tencent/Hy3",
            "source_sha": "716aa7241bd6d95896be4ebfc761162a9c4d49ef",
            "tool": "mtplx.hy3_native_quantizer v1",
        },
    }


def _conversion_manifest_fixture() -> dict:
    """Mirrors hy3-expert-only-mlx-q2/conversion-manifest.json."""

    return {
        "alignment": 16384,
        "producer": "mtplx.hy3_expert_q2",
        "source": {
            "path": "/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q4",
            "revision": "716aa7241bd6d95896be4ebfc761162a9c4d49ef",
        },
        "journal": [
            {"step": "quantize", "note": "read /Users/davidtai/models/in.safetensors ok"},
        ],
        "target_descriptor": {"bits": 2, "group_size": 32},
    }


def test_forge_inputs_paths_are_redacted():
    scrubbed = scrub_runtime_metadata(_runtime_fixture())
    inputs = scrubbed["forge_provenance"]["forge_inputs"]

    assert inputs["source_path"] == f"{REDACTED_PATH}/hy3-mtp-layer80"
    assert inputs["output_path"] == f"{REDACTED_PATH}/hy3-q4-mlx-mtp"
    assert inputs["bf16_head_source_path"] == f"{REDACTED_PATH}/layer80-bf16.safetensors"
    for value in inputs.values():
        assert "/Users/" not in value


def test_intended_hf_repo_is_dropped():
    scrubbed = scrub_runtime_metadata(_runtime_fixture())

    assert "intended_hf_repo" not in scrubbed["forge_provenance"]


def test_useful_provenance_survives():
    scrubbed = scrub_runtime_metadata(_runtime_fixture())
    provenance = scrubbed["forge_provenance"]

    assert provenance["source_repo"] == "tencent/Hy3"
    assert provenance["source_sha"] == "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
    assert provenance["forge_recipe"] == {"mtp_policy": "keep_bf16", "bits": 4}
    assert provenance["forged_at"] == "2026-07-11T15:23:33Z"
    assert provenance["mtplx_version"] == "2.0.2"
    assert scrubbed["arch_id"] == "hy_v3"
    assert scrubbed["mtp_sidecar_file"] == "mtp.safetensors"
    assert scrubbed["mtp_depth_max"] == 3


def test_input_is_not_mutated():
    original = _runtime_fixture()
    snapshot = copy.deepcopy(original)

    scrub_runtime_metadata(original)

    assert original == snapshot


def test_conversion_manifest_paths_are_redacted():
    scrubbed = scrub_runtime_metadata(_conversion_manifest_fixture())

    assert scrubbed["source"]["path"] == f"{REDACTED_PATH}/hy3-expert-only-mlx-q4"
    assert scrubbed["source"]["revision"] == "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
    assert scrubbed["alignment"] == 16384


def test_paths_embedded_in_free_text_are_redacted():
    scrubbed = scrub_runtime_metadata(_conversion_manifest_fixture())

    note = scrubbed["journal"][0]["note"]
    assert "/Users/davidtai" not in note
    assert note.startswith("read ") and note.endswith(" ok")


def test_leak_detector_finds_and_then_clears():
    fixture = _runtime_fixture()

    assert runtime_metadata_leaks(fixture)
    assert runtime_metadata_leaks(scrub_runtime_metadata(fixture)) == []


def test_scrub_covers_linux_and_temp_dirs():
    assert scrub_path_value("/home/alice/models/x") == f"{REDACTED_PATH}/x"
    assert scrub_path_value("/var/folders/ab/T/run") == f"{REDACTED_PATH}/run"
    assert scrub_path_value("~/models/y") == f"{REDACTED_PATH}/y"


def test_non_path_values_are_left_alone():
    payload = {"repo": "owner/name", "license": "apache-2.0", "n": 3, "ok": True}

    assert scrub_runtime_metadata(payload) == payload


def test_lists_of_paths_are_redacted():
    scrubbed = scrub_runtime_metadata(
        {"inputs": ["/Users/davidtai/a.safetensors", "relative/b.safetensors"]}
    )

    assert scrubbed["inputs"] == [
        f"{REDACTED_PATH}/a.safetensors",
        "relative/b.safetensors",
    ]


def test_routes_and_slash_strings_outside_path_keys_survive():
    payload = {
        "endpoint": "/v1/chat/completions",
        "bos_token": "/",
        "notes": ["/opt/models/x is a shared mount"],
    }

    assert scrub_runtime_metadata(payload) == payload
    assert runtime_metadata_leaks(payload) == []


def test_absolute_paths_under_path_keys_are_redacted_even_outside_home():
    scrubbed = scrub_runtime_metadata({"source_path": "/opt/models/trunk"})

    assert scrubbed["source_path"] == f"{REDACTED_PATH}/trunk"
    assert runtime_metadata_leaks({"output_dir": "/opt/models/out"}) == ["/opt/models/out"]


def test_scrub_json_documents_returns_only_leaking_documents(tmp_path):
    import json

    from mtplx.metadata_scrub import scrub_json_documents

    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    (tmp_path / "mtplx_runtime.json").write_text(
        json.dumps({"base_trunk": "/Users/someone/.mtplx/models/Owner--Trunk"}),
        encoding="utf-8",
    )
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    documents = scrub_json_documents(tmp_path)

    assert [document.name for document in documents] == ["mtplx_runtime.json"]
    assert documents[0].leaks == ("/Users/someone/.mtplx/models/Owner--Trunk",)
    assert documents[0].document == {"base_trunk": f"{REDACTED_PATH}/Owner--Trunk"}
    assert json.loads(documents[0].payload) == documents[0].document
    assert runtime_metadata_leaks(documents[0].document) == []


def _tokenizer_with_tilde_merges() -> dict:
    """A tokenizer.json shaped like the one the scrubber broke on 22 November: seven merges built from ``~``."""

    return {
        "version": "1.0",
        "added_tokens": [{"id": 0, "content": "~", "special": False}],
        "model": {
            "type": "BPE",
            "vocab": {"~": 0, "~~": 1, "~~~~": 2, "~/": 3, "a": 4},
            "merges": ["~ ~", "~~ ~~", "~ ~~", "~~ ~", "~~~ ~", "~ ~~~", "~/ a"],
        },
    }


def test_tilde_tokens_are_text_not_home_paths():
    payload = {"content": "~", "merges": ["~ ~", "~~ ~~", "~~~~"], "home": "~/models/y", "user": "~ann/m"}

    scrubbed = scrub_runtime_metadata(payload)

    assert scrubbed["content"] == "~"
    assert scrubbed["merges"] == ["~ ~", "~~ ~~", "~~~~"]
    assert scrubbed["home"] == f"{REDACTED_PATH}/y"
    assert scrubbed["user"] == f"{REDACTED_PATH}/m"
    assert runtime_metadata_leaks(payload) == ["~/models/y", "~ann/m"]


def test_a_packs_tokenizer_is_never_scanned_or_rewritten(tmp_path):
    import json

    from mtplx.metadata_scrub import scrub_json_documents

    tokenizer = json.dumps(_tokenizer_with_tilde_merges())
    (tmp_path / "tokenizer.json").write_text(tokenizer, encoding="utf-8")
    (tmp_path / "vocab.json").write_text(json.dumps({"/Users/": 7, "~": 8}), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"name_or_path": "/Users/someone/models/Trunk", "added_tokens_decoder": {"0": {"content": "~"}}}),
        encoding="utf-8",
    )

    documents = {document.name: document for document in scrub_json_documents(tmp_path)}

    assert sorted(documents) == ["tokenizer_config.json"]
    assert documents["tokenizer_config.json"].document == {
        "name_or_path": f"{REDACTED_PATH}/Trunk",
        "added_tokens_decoder": {"0": {"content": "~"}},
    }
    assert (tmp_path / "tokenizer.json").read_text(encoding="utf-8") == tokenizer


def test_publish_check_scrub_leaves_the_tokenizer_byte_for_byte(tmp_path, capsys):
    import json

    from mtplx.cli import main

    staging = tmp_path / "staging"
    staging.mkdir()
    tokenizer = json.dumps(_tokenizer_with_tilde_merges())
    (staging / "tokenizer.json").write_text(tokenizer, encoding="utf-8")
    (staging / "mtplx_runtime.json").write_text(
        json.dumps({"base_trunk": "/Users/someone/.mtplx/models/Owner--Trunk"}), encoding="utf-8"
    )

    main(["model", "publish-check", "--staging-dir", str(staging), "--repo-id", "x/y", "--scrub"])
    report = json.loads(capsys.readouterr().out)

    assert report["scrubbed"] == ["mtplx_runtime.json"]
    assert (staging / "tokenizer.json").read_text(encoding="utf-8") == tokenizer
