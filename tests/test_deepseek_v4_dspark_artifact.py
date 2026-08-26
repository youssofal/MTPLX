import json
from types import SimpleNamespace

import pytest

from mtplx import runtime
import mtplx.deepseek_v4_dspark_artifact as dspark_artifact
import mtplx.deepseek_v4_mia_engine as mia_engine


def test_public_artifact_verifier_rejects_generic_before_weight_index_access(
    monkeypatch,
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"}),
        encoding="utf-8",
    )
    original_read_json = dspark_artifact._read_json
    reads: list[str] = []

    def tracked_read_json(path, *, label):
        reads.append(path.name)
        if path.name == "model.safetensors.index.json":
            raise AssertionError("legacy generic weight index was accessed")
        return original_read_json(path, label=label)

    monkeypatch.setattr(dspark_artifact, "_read_json", tracked_read_json)

    with pytest.raises(
        dspark_artifact.DSparkArtifactError,
        match="requires.*exl3-trellis",
    ):
        dspark_artifact.open_verified_dspark_artifact(tmp_path)
    assert reads == ["config.json"]


def test_exact_dspark_base_load_uses_the_pinned_mia_target_and_k64_loader(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[tuple[str, object]] = []
    tokenizer = object()
    model = SimpleNamespace()
    draft_root = tmp_path / "draft"
    config = {
        "model_type": "deepseek_v4",
        "hybrid_tr3_tail": {"format": "exl3-trellis"},
    }
    validation = SimpleNamespace(target_config=config)

    monkeypatch.setattr(
        "mtplx.deepseek_v4_exl3._default_mia_dspark_root",
        lambda path: draft_root,
    )
    monkeypatch.setattr(
        mia_engine,
        "validate_pinned_mia_artifacts",
        lambda target, draft: (
            events.append(("pinned_target_and_k64_validation", (target, draft)))
            or validation
        ),
    )
    monkeypatch.setattr(
        mia_engine,
        "revalidate_pinned_mia_tokenizer_files",
        lambda observed: events.append(("tokenizer_revalidation", observed)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_load_tokenizer_resilient",
        lambda path, observed: events.append(("tokenizer", path)) or tokenizer,
    )
    monkeypatch.setattr(
        "mtplx.deepseek_v4_exl3.load_mia_exl3_dspark_model",
        lambda path, *, draft_root, artifact_validation: (
            events.append(
                (
                    "pinned_mia_k64_loader",
                    (path, draft_root, artifact_validation),
                )
            )
            or model
        ),
    )

    loaded_model, loaded_tokenizer = runtime._load_base_model(tmp_path, config)

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert events == [
        ("pinned_target_and_k64_validation", (tmp_path, draft_root)),
        ("tokenizer", tmp_path),
        ("tokenizer_revalidation", validation),
        (
            "pinned_mia_k64_loader",
            (tmp_path, draft_root, validation),
        ),
    ]


def test_tokenizer_swap_after_validation_rejects_before_model_construction(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[str] = []
    validation = SimpleNamespace(target_config={"model_type": "deepseek_v4"})
    config = {
        "model_type": "deepseek_v4",
        "hybrid_tr3_tail": {"format": "exl3-trellis"},
    }

    monkeypatch.setattr(
        "mtplx.deepseek_v4_exl3._default_mia_dspark_root",
        lambda _path: tmp_path / "draft",
    )
    monkeypatch.setattr(
        mia_engine,
        "validate_pinned_mia_artifacts",
        lambda *_args: events.append("validated") or validation,
    )
    monkeypatch.setattr(
        runtime,
        "_load_tokenizer_resilient",
        lambda *_args: events.append("tokenizer_swapped") or object(),
    )

    def reject_swap(observed) -> None:
        assert observed is validation
        events.append("swap_rejected")
        raise ValueError("pinned Mia tokenizer file identity changed")

    monkeypatch.setattr(
        mia_engine,
        "revalidate_pinned_mia_tokenizer_files",
        reject_swap,
        raising=False,
    )
    monkeypatch.setattr(
        "mtplx.deepseek_v4_exl3.load_mia_exl3_dspark_model",
        lambda *_args, **_kwargs: events.append("model") or object(),
    )

    with pytest.raises(ValueError, match="tokenizer file identity changed"):
        runtime._load_base_model(tmp_path, config)
    assert events == ["validated", "tokenizer_swapped", "swap_rejected"]
