"""Tests for the embedding and reranking endpoints.

The MLX forward passes need real weights, so they are not exercised here.
What is covered is everything that decides *whether the right model runs*:
flag parsing, served-id resolution, the shared-backend guarantee, and the HTTP
contract — including that a chat-only daemon keeps behaving exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.retrieval import (
    RetrievalError,
    RetrievalRegistry,
    RetrievalSpec,
    default_served_id,
    parse_model_flag,
    registry_from_args,
)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mtplx.server.openai import create_app  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_server_openai import _fake_state  # noqa: E402


# ---- flag parsing ---------------------------------------------------------


def test_parse_model_flag_defaults_served_id_to_the_basename():
    spec = parse_model_flag("mlx-community/Qwen3-Embedding-8B-4bit-DWQ", "embedding")
    assert spec.model_ref == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
    assert spec.served_id == "Qwen3-Embedding-8B-4bit-DWQ"
    assert spec.role == "embedding"


def test_parse_model_flag_accepts_an_explicit_alias():
    spec = parse_model_flag("org/name=fast-embed", "embedding")
    assert spec.model_ref == "org/name"
    assert spec.served_id == "fast-embed"


def test_parse_model_flag_keeps_absolute_paths_intact():
    spec = parse_model_flag("/models/Qwen3-Embedding-8B", "embedding")
    assert spec.model_ref == "/models/Qwen3-Embedding-8B"
    assert spec.served_id == "Qwen3-Embedding-8B"


def test_parse_model_flag_rejects_empty_sides():
    with pytest.raises(ValueError):
        parse_model_flag("org/name=", "embedding")
    with pytest.raises(ValueError):
        parse_model_flag("   ", "embedding")


def test_default_served_id_strips_trailing_slash():
    assert default_served_id("org/name/") == "name"


# ---- registry -------------------------------------------------------------


def _registry() -> RetrievalRegistry:
    registry = RetrievalRegistry()
    registry.register(RetrievalSpec("embed-a", "org/embed-a", "embedding"))
    registry.register(RetrievalSpec("rank-a", "org/rank-a", "rerank"))
    return registry


def test_registry_reports_roles_separately():
    registry = _registry()
    assert [spec.served_id for spec in registry.specs_for_role("embedding")] == ["embed-a"]
    assert [spec.served_id for spec in registry.specs_for_role("rerank")] == ["rank-a"]
    assert registry.enabled


def test_registry_resolves_by_served_id_model_ref_and_basename():
    registry = _registry()
    for requested in ("embed-a", "org/embed-a", "other/embed-a"):
        assert registry._spec("embedding", requested).served_id == "embed-a"


def test_registry_defaults_to_the_first_model_for_a_role():
    registry = _registry()
    assert registry._spec("embedding", None).served_id == "embed-a"


def test_registry_rejects_an_unknown_model_by_name():
    registry = _registry()
    with pytest.raises(RetrievalError, match="unknown embedding model"):
        registry._spec("embedding", "nope")


def test_registry_reports_a_missing_role_rather_than_falling_back():
    """A rerank request must never be silently answered by an embedder."""
    registry = RetrievalRegistry()
    registry.register(RetrievalSpec("embed-a", "org/embed-a", "embedding"))
    with pytest.raises(RetrievalError, match="no rerank model is configured"):
        registry._spec("rerank", None)


def test_one_reference_in_both_roles_shares_a_single_backend(monkeypatch):
    """The point of the shared cache: one set of weights, two endpoints."""
    monkeypatch.setattr(
        "mtplx.hf_loader.resolve_model_path",
        lambda ref, cache_dir=None: Path("/models") / str(ref).rsplit("/", 1)[-1],
    )
    registry = RetrievalRegistry()
    registry.register(RetrievalSpec("dual", "org/dual", "embedding"))
    registry.register(RetrievalSpec("dual", "org/dual", "rerank"))

    embedding_backend = registry._backend(registry._spec("embedding", "dual"))
    rerank_backend = registry._backend(registry._spec("rerank", "dual"))
    assert embedding_backend is rerank_backend


def test_resident_models_are_evicted_beyond_the_cap(monkeypatch):
    monkeypatch.setattr(
        "mtplx.hf_loader.resolve_model_path",
        lambda ref, cache_dir=None: Path("/models") / str(ref).rsplit("/", 1)[-1],
    )
    registry = RetrievalRegistry(max_resident=1)
    for index in range(3):
        registry.register(RetrievalSpec(f"e{index}", f"org/e{index}", "embedding"))

    unloaded: list[str] = []
    for index in range(3):
        backend = registry._backend(registry._spec("embedding", f"e{index}"))
        # Pretend the weights loaded so the cap has something to evict.
        backend._model = object()
        backend._tokenizer = object()
        backend.unload = lambda ref=backend.model_ref: unloaded.append(ref)  # type: ignore[method-assign]

    assert unloaded == ["org/e0", "org/e1"]


def test_descriptors_expose_role_and_load_state():
    registry = _registry()
    descriptors = {entry["id"]: entry for entry in registry.descriptors()}
    assert descriptors["embed-a"]["role"] == "embedding"
    assert descriptors["rank-a"]["role"] == "rerank"
    assert descriptors["embed-a"]["loaded"] is False


def test_registry_from_args_reads_repeatable_flags():
    args = SimpleNamespace(
        embedding_model=["org/embed=e1"],
        reranker_model=["org/rank"],
        retrieval_max_resident=3,
        retrieval_max_tokens=1024,
    )
    registry = registry_from_args(args)
    assert registry.max_resident == 3
    embedding = registry.specs_for_role("embedding")[0]
    assert (embedding.served_id, embedding.max_tokens) == ("e1", 1024)
    assert registry.specs_for_role("rerank")[0].served_id == "rank"


def test_registry_from_args_without_any_flags_is_disabled():
    assert not registry_from_args(SimpleNamespace()).enabled


# ---- HTTP contract --------------------------------------------------------


class _StubRegistry:
    """Stands in for the real registry so no weights are needed."""

    enabled = True

    def __init__(self) -> None:
        self.embed_calls: list[dict] = []

    def descriptors(self):
        return [
            {"id": "e1", "role": "embedding", "model_ref": "org/e1", "loaded": True, "max_tokens": 8192},
            {"id": "r1", "role": "rerank", "model_ref": "org/r1", "loaded": False, "max_tokens": 8192},
        ]

    def embed(self, texts, *, model=None, instruction=None):
        self.embed_calls.append({"texts": texts, "model": model, "instruction": instruction})
        return [[0.5, 0.5] for _ in texts], RetrievalSpec("e1", "org/e1", "embedding")

    def rerank(self, query, documents, *, model=None, instruction=None):
        scores = [float(len(document)) for document in documents]
        return scores, RetrievalSpec("r1", "org/r1", "rerank")


def _client(registry=None) -> TestClient:
    state = _fake_state()
    if registry is not None:
        state.retrieval = registry
    return TestClient(create_app(state))


def test_chat_only_daemon_reports_no_embedding_model():
    """Regression guard: adding the routes must not change a chat-only setup."""
    response = _client().post("/v1/embeddings", json={"input": "hello"})
    assert response.status_code == 404
    # MTPLX wraps HTTPException in an OpenAI-style envelope; the retrieval
    # routes inherit that contract rather than inventing their own.
    assert "--embedding-model" in response.json()["error"]["message"]


def test_chat_only_daemon_reports_no_reranking_model():
    response = _client().post("/v1/rerank", json={"query": "q", "documents": ["d"]})
    assert response.status_code == 404
    assert "--reranker-model" in response.json()["error"]["message"]


def test_models_listing_stays_chat_only_without_retrieval():
    payload = _client().get("/v1/models").json()
    assert [entry["id"] for entry in payload["data"]] == ["mtplx-test-model"]
    assert payload["data"][0]["capability"] == "chat"


def test_models_listing_includes_retrieval_models():
    payload = _client(_StubRegistry()).get("/v1/models").json()
    entries = {entry["id"]: entry for entry in payload["data"]}
    assert entries["e1"]["capability"] == "embedding"
    assert entries["r1"]["capability"] == "rerank"
    assert entries["mtplx-test-model"]["capability"] == "chat"


def test_embeddings_returns_openai_shape_in_input_order():
    registry = _StubRegistry()
    response = _client(registry).post(
        "/v1/embeddings", json={"model": "e1", "input": ["a", "b"], "encoding_format": "float"}
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["object"] == "list"
    assert [entry["index"] for entry in payload["data"]] == [0, 1]
    assert payload["data"][0]["embedding"] == [0.5, 0.5]
    assert payload["model"] == "e1"
    assert registry.embed_calls[0]["texts"] == ["a", "b"]


def test_embeddings_accepts_a_bare_string_input():
    registry = _StubRegistry()
    response = _client(registry).post("/v1/embeddings", json={"input": "solo"})
    assert response.status_code == 200
    assert registry.embed_calls[0]["texts"] == ["solo"]


def test_embeddings_forwards_the_query_instruction():
    registry = _StubRegistry()
    _client(registry).post("/v1/embeddings", json={"input": "a", "instruction": "Find docs"})
    assert registry.embed_calls[0]["instruction"] == "Find docs"


def test_embeddings_rejects_a_non_text_input():
    """Pydantic rejects a non-text body before the handler runs."""
    response = _client(_StubRegistry()).post("/v1/embeddings", json={"input": {"bad": 1}})
    assert response.status_code == 422


def test_embeddings_rejects_a_missing_input():
    response = _client(_StubRegistry()).post("/v1/embeddings", json={})
    assert response.status_code == 400
    assert "input must be" in response.json()["error"]["message"]


def test_rerank_sorts_by_score_and_keeps_original_indexes():
    response = _client(_StubRegistry()).post(
        "/v1/rerank", json={"query": "q", "documents": ["short", "much longer document"]}
    )
    payload = response.json()
    assert response.status_code == 200
    assert [entry["index"] for entry in payload["results"]] == [1, 0]
    assert payload["results"][0]["relevance_score"] > payload["results"][1]["relevance_score"]
    assert "document" not in payload["results"][0]


def test_rerank_honours_top_n_and_return_documents():
    response = _client(_StubRegistry()).post(
        "/v1/rerank",
        json={"query": "q", "documents": ["a", "bbb", "cc"], "top_n": 2, "return_documents": True},
    )
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["results"][0]["document"]["text"] == "bbb"


def test_rerank_requires_a_query():
    response = _client(_StubRegistry()).post("/v1/rerank", json={"query": "  ", "documents": ["d"]})
    assert response.status_code == 400


def test_unknown_retrieval_model_is_reported_as_not_found():
    class _Strict(_StubRegistry):
        def embed(self, texts, *, model=None, instruction=None):
            raise RetrievalError("unknown embedding model 'nope'; served: e1")

    response = _client(_Strict()).post("/v1/embeddings", json={"model": "nope", "input": "a"})
    assert response.status_code == 404
    assert "unknown embedding model" in response.json()["error"]["message"]
