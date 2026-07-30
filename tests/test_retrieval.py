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
    BATCH_TOKEN_BUDGET,
    RetrievalError,
    RetrievalStats,
    RetrievalRegistry,
    RetrievalSpec,
    _plan_batches,
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


# ---- batch planning -------------------------------------------------------


def test_plan_batches_keeps_every_sequence_exactly_once():
    lengths = [5, 900, 7, 4, 1200, 6, 8, 3]
    plan = _plan_batches(lengths, max_rows=8)
    assert sorted(index for batch in plan for index in batch) == list(range(len(lengths)))


def test_plan_batches_never_exceeds_the_configured_batch_size():
    plan = _plan_batches([4] * 20, max_rows=8)
    assert [len(batch) for batch in plan] == [8, 8, 4]


def test_plan_batches_isolates_a_long_sequence_from_short_ones():
    """The measured worst case: one long text used to drag seven short ones up.

    Both used to land in a single batch padded to the long one, costing eight
    times the long text alone. They must now be planned apart.
    """
    plan = _plan_batches([900] + [8] * 7, max_rows=8)
    long_batch = next(batch for batch in plan if 0 in batch)

    assert long_batch == [0]
    assert sorted(index for batch in plan if batch is not long_batch for index in batch) == [
        1, 2, 3, 4, 5, 6, 7
    ]


def test_plan_batches_packs_short_sequences_densely():
    """Short texts are where batching actually pays, so they must not be split."""
    plan = _plan_batches([8] * 8, max_rows=8)
    assert len(plan) == 1


def test_plan_batches_respects_the_token_budget():
    lengths = [300] * 8
    plan = _plan_batches(lengths, max_rows=8)
    for batch in plan:
        width = max(lengths[index] for index in batch)
        assert width * len(batch) <= BATCH_TOKEN_BUDGET


def test_plan_batches_still_places_a_sequence_larger_than_the_budget():
    """A single oversized text has nowhere else to go — it must not be dropped."""
    plan = _plan_batches([BATCH_TOKEN_BUDGET * 3, 4], max_rows=8)
    assert sorted(index for batch in plan for index in batch) == [0, 1]
    assert [0] in plan


def test_plan_batches_handles_an_empty_request():
    assert _plan_batches([], max_rows=8) == []


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


def test_a_slot_is_reserved_before_the_weights_finish_loading(monkeypatch):
    """Regression: concurrent first-use requests must not both stay resident.

    Loading happens outside the registry lock. If residency counted only
    finished loads, two requests for different models would each see the other
    as absent, skip eviction, and blow the cap exactly when memory is tightest.
    """
    monkeypatch.setattr(
        "mtplx.hf_loader.resolve_model_path",
        lambda ref, cache_dir=None: Path("/models") / str(ref).rsplit("/", 1)[-1],
    )
    registry = RetrievalRegistry(max_resident=1)
    registry.register(RetrievalSpec("a", "org/a", "embedding"))
    registry.register(RetrievalSpec("b", "org/b", "embedding"))

    # Acquire A but never mark it loaded, as an in-flight load would look.
    registry._backend(registry._spec("embedding", "a"))
    registry._backend(registry._spec("embedding", "b"))

    assert list(registry.status()["resident"]) == ["/models/b"]


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
        backend.unload = lambda ref=str(backend.path): unloaded.append(ref)  # type: ignore[method-assign]

    assert unloaded == ["/models/e0", "/models/e1"]


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


def test_registry_from_args_prefers_the_forwarded_retrieval_cache_dir():
    """The server subprocess carries no cache dir of its own.

    The chat model arrives pre-resolved, so without this the retrieval models
    would silently look in the default cache and miss a custom --cache-dir.
    """
    args = SimpleNamespace(
        embedding_model=["org/embed"],
        retrieval_cache_dir="/custom/retrieval",
        cache_dir="/custom/cli",
        model_dir="/custom/setup",
    )
    assert registry_from_args(args).cache_dir == "/custom/retrieval"


def test_registry_from_args_falls_back_to_the_cli_cache_dir():
    args = SimpleNamespace(embedding_model=["org/embed"], cache_dir="/custom/cli")
    assert registry_from_args(args).cache_dir == "/custom/cli"


# ---- metrics --------------------------------------------------------------


def test_stats_start_empty_so_an_unused_model_is_visible_as_unused():
    stats = RetrievalStats()
    payload = stats.to_dict()
    assert payload["requests"] == 0
    assert payload["itemsPerSecond"] is None
    assert payload["avgLatencyMs"] is None
    assert payload["lastUsedS"] is None


def test_stats_derive_throughput_and_latency():
    stats = RetrievalStats()
    stats.record(items=8, tokens=80, seconds=2.0)
    stats.record(items=2, tokens=20, seconds=2.0)
    payload = stats.to_dict()
    assert payload["requests"] == 2
    assert payload["items"] == 10
    assert payload["itemsPerSecond"] == 2.5      # 10 items / 4.0s
    assert payload["avgLatencyMs"] == 2000.0     # 4.0s over 2 requests
    assert payload["lastUsedS"] is not None


def test_descriptors_report_load_state_and_counters_per_role():
    registry = _registry()
    registry._stats_for(RetrievalSpec("embed-a", "org/embed-a", "embedding")).record(
        items=3, tokens=30, seconds=1.5
    )
    descriptors = {(e["role"], e["id"]): e for e in registry.descriptors()}

    embedding = descriptors[("embedding", "embed-a")]
    assert embedding["requests"] == 1
    assert embedding["items"] == 3
    assert embedding["loaded"] is False
    assert embedding["resident"] is False

    # Counters are per role, so reranking a document must not be attributed to
    # the embedder — even when both roles share one set of weights.
    assert descriptors[("rerank", "rank-a")]["requests"] == 0


def test_load_time_is_not_counted_as_inference_latency(monkeypatch):
    """A cold first request must not make a fast model look ten times slower."""
    monkeypatch.setattr(
        "mtplx.hf_loader.resolve_model_path",
        lambda ref, cache_dir=None: Path("/models") / str(ref).rsplit("/", 1)[-1],
    )
    registry = RetrievalRegistry()
    spec = RetrievalSpec("slow-load", "org/slow-load", "embedding")
    registry.register(spec)
    backend = registry._backend(spec)
    backend.load_seconds = 9.0
    backend._model = object()
    backend._tokenizer = object()

    stats = registry._stats_for(spec)
    stats.record(items=5, tokens=50, seconds=0.4)
    payload = {entry["id"]: entry for entry in registry.descriptors()}["slow-load"]

    assert payload["loadSeconds"] == 9.0
    assert payload["computeSeconds"] == 0.4
    assert payload["avgLatencyMs"] == 400.0


def test_status_reports_the_cap_and_the_resident_set():
    registry = _registry()
    status = registry.status()
    assert status["enabled"] is True
    assert status["max_resident"] == registry.max_resident
    assert status["resident"] == []
    assert {entry["id"] for entry in status["models"]} == {"embed-a", "rank-a"}


def test_status_of_a_chat_only_daemon_is_explicitly_disabled():
    status = RetrievalRegistry().status()
    assert status["enabled"] is False
    assert status["models"] == []


# ---- batching through a real forward pass ---------------------------------


class _IdentityTokenizer:
    """Encodes a text as its own marker id, repeated to the length it asks for.

    Texts are written as ``"<marker>x<count>"``, which makes both the identity
    and the length of every sequence readable straight off the vectors.
    """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        marker, _, count = text.partition("x")
        return [int(marker)] * int(count)

    def convert_tokens_to_ids(self, token: str) -> int:
        return 0


class _EchoModel:
    """Returns hidden states that carry the input ids through unchanged.

    With mean pooling this makes the expected vector computable in plain
    Python, so a batch that reordered its rows — or that pooled over padding
    instead of the true length — cannot pass.
    """

    def model(self, inputs):
        import mlx.core as mx

        values = inputs.astype(mx.float32)
        return mx.stack([values, mx.ones_like(values)], axis=-1)


def _expected_vector(marker: int, count: int) -> tuple[float, float]:
    # _encode_embedding appends the pad token, so the sequence is `count`
    # copies of the marker followed by one zero, and the true length is
    # count + 1. Padding beyond that must not enter the mean.
    mean = marker * count / (count + 1)
    norm = (mean**2 + 1.0) ** 0.5
    return mean / norm, 1.0 / norm


def _echo_registry(monkeypatch) -> tuple[RetrievalRegistry, RetrievalSpec]:
    monkeypatch.setattr(
        "mtplx.hf_loader.resolve_model_path",
        lambda ref, cache_dir=None: Path("/models") / str(ref).rsplit("/", 1)[-1],
    )
    registry = RetrievalRegistry()
    spec = RetrievalSpec("echo", "org/echo", "embedding", pooling="mean")
    registry.register(spec)
    backend = registry._backend(spec)
    backend._model = _EchoModel()
    backend._tokenizer = _IdentityTokenizer()
    return registry, spec


def test_embedding_returns_vectors_in_input_order_despite_reordered_batches(monkeypatch):
    """Sorting by length is an optimisation, so it must be invisible to callers."""
    pytest.importorskip("mlx.core")
    registry, _spec = _echo_registry(monkeypatch)

    # Deliberately interleaved so that planning by length has to reorder, and
    # so the batches do not fall on the input's own boundaries.
    requested = [(1, 900), (2, 4), (3, 700), (4, 6), (5, 5), (6, 850), (7, 3), (8, 7)]
    texts = [f"{marker}x{count}" for marker, count in requested]

    vectors, _used, tokens = registry.embed(texts)

    assert len(vectors) == len(texts)
    assert tokens == sum(count + 1 for _marker, count in requested)
    for vector, (marker, count) in zip(vectors, requested):
        expected = _expected_vector(marker, count)
        assert vector[0] == pytest.approx(expected[0], rel=1e-4)
        assert vector[1] == pytest.approx(expected[1], rel=1e-4)


def test_embedding_of_a_single_text_is_unaffected_by_planning(monkeypatch):
    pytest.importorskip("mlx.core")
    registry, _spec = _echo_registry(monkeypatch)

    vectors, _used, _tokens = registry.embed(["42x10"])

    expected = _expected_vector(42, 10)
    assert vectors[0][0] == pytest.approx(expected[0], rel=1e-4)


def test_a_short_text_embeds_the_same_alone_as_beside_a_long_one(monkeypatch):
    """Padding must not change a vector — that was the bug behind the slowdown."""
    pytest.importorskip("mlx.core")
    registry, _spec = _echo_registry(monkeypatch)

    alone, _used, _tokens = registry.embed(["7x3"])
    beside, _used, _tokens = registry.embed(["9x900", "7x3"])

    assert beside[1][0] == pytest.approx(alone[0][0], rel=1e-4)
    assert beside[1][1] == pytest.approx(alone[0][1], rel=1e-4)


class _MarkerRerankTokenizer:
    """Reads the document marker back out of the assembled rerank prompt."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        marker, _, count = text.rpartition("<Document>: ")[2].partition("x")
        if not count:
            return []  # the fixed prefix and suffix contribute nothing
        return [int(marker)] * int(count)

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"no": 0, "yes": 1}.get(token, 0)


class _EchoReranker:
    """Scores a row from its last real token, so padding cannot go unnoticed."""

    def __call__(self, inputs):
        import mlx.core as mx

        values = inputs.astype(mx.float32)
        return mx.stack([mx.zeros_like(values), values], axis=-1)


def test_reranking_returns_scores_in_document_order_despite_reordered_batches(monkeypatch):
    """The route ranks by index into the caller's list, so a swap is silent."""
    pytest.importorskip("mlx.core")
    import math

    monkeypatch.setattr(
        "mtplx.hf_loader.resolve_model_path",
        lambda ref, cache_dir=None: Path("/models") / str(ref).rsplit("/", 1)[-1],
    )
    registry = RetrievalRegistry()
    spec = RetrievalSpec("echo-rank", "org/echo-rank", "rerank")
    registry.register(spec)
    backend = registry._backend(spec)
    backend._model = _EchoReranker()
    backend._tokenizer = _MarkerRerankTokenizer()

    requested = [(2, 900), (5, 4), (1, 700), (4, 6), (3, 800), (6, 5)]
    documents = [f"{marker}x{count}" for marker, count in requested]

    scores, _used, _tokens = registry.rerank("does it match", documents)

    assert len(scores) == len(documents)
    for score, (marker, _count) in zip(scores, requested):
        assert score == pytest.approx(1.0 / (1.0 + math.exp(-marker)), rel=1e-4)


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

    def status(self):
        return {"enabled": True, "max_resident": 2, "resident": ["org/e1"], "models": self.descriptors()}

    def embed(self, texts, *, model=None, instruction=None):
        self.embed_calls.append({"texts": texts, "model": model, "instruction": instruction})
        return [[0.5, 0.5] for _ in texts], RetrievalSpec("e1", "org/e1", "embedding"), 7 * len(texts)

    def rerank(self, query, documents, *, model=None, instruction=None):
        scores = [float(len(document)) for document in documents]
        return scores, RetrievalSpec("r1", "org/r1", "rerank"), 11 * len(documents)


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


def test_embeddings_honour_base64_encoding_format():
    """A client that asks for base64 must not silently receive float arrays."""
    import base64
    import struct

    response = _client(_StubRegistry()).post(
        "/v1/embeddings", json={"input": "a", "encoding_format": "base64"}
    )
    payload = response.json()
    assert response.status_code == 200
    encoded = payload["data"][0]["embedding"]
    assert isinstance(encoded, str)
    decoded = struct.unpack("<2f", base64.b64decode(encoded))
    assert [round(value, 4) for value in decoded] == [0.5, 0.5]


def test_embeddings_default_to_float_vectors():
    response = _client(_StubRegistry()).post("/v1/embeddings", json={"input": "a"})
    assert response.json()["data"][0]["embedding"] == [0.5, 0.5]


def test_embeddings_reject_an_unknown_encoding_format():
    response = _client(_StubRegistry()).post(
        "/v1/embeddings", json={"input": "a", "encoding_format": "float16"}
    )
    assert response.status_code == 400
    assert "encoding_format" in response.json()["error"]["message"]


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


def test_snapshot_always_carries_a_retrieval_section():
    """The dashboard must distinguish "not configured" from "not supported"."""
    payload = _client().get("/v1/mtplx/snapshot").json()
    assert payload["retrieval"]["enabled"] is False

    configured = _client(_StubRegistry()).get("/v1/mtplx/snapshot").json()
    assert configured["retrieval"]["enabled"] is True
    assert {m["id"] for m in configured["retrieval"]["models"]} == {"e1", "r1"}


# ---- review round 2 -------------------------------------------------------


def _fixed_resolver(monkeypatch, mapping=None):
    """Resolve refs to paths, optionally collapsing several refs onto one."""
    def resolve(ref, cache_dir=None):
        if mapping and str(ref) in mapping:
            return Path(mapping[str(ref)])
        return Path("/models") / str(ref).rsplit("/", 1)[-1]

    monkeypatch.setattr("mtplx.hf_loader.resolve_model_path", resolve)


def test_two_references_to_one_directory_share_a_backend(monkeypatch):
    """A Hugging Face id and its local path must not load the weights twice."""
    _fixed_resolver(
        monkeypatch,
        {"org/model": "/models/model", "/models/model": "/models/model"},
    )
    registry = RetrievalRegistry()
    remote = RetrievalSpec("remote", "org/model", "embedding")
    local = RetrievalSpec("local", "/models/model", "embedding")
    registry.register(remote)
    registry.register(local)

    with registry._acquire(remote) as first, registry._acquire(local) as second:
        assert first is second
    assert list(registry.status()["resident"]) == ["/models/model"]


def test_a_backend_in_use_is_never_evicted(monkeypatch):
    """Unloading weights another request holds frees nothing and forces a reload."""
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry(max_resident=1)
    first = RetrievalSpec("a", "org/a", "embedding")
    second = RetrievalSpec("b", "org/b", "embedding")
    registry.register(first)
    registry.register(second)

    with registry._acquire(first) as pinned:
        pinned._model = object()
        with registry._acquire(second):
            # The cap is briefly exceeded rather than pulling weights out from
            # under the in-flight request.
            assert pinned.loaded is True
            assert set(registry.status()["resident"]) == {"/models/a", "/models/b"}


def test_an_idle_backend_is_evicted_once_it_is_released(monkeypatch):
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry(max_resident=1)
    first = RetrievalSpec("a", "org/a", "embedding")
    second = RetrievalSpec("b", "org/b", "embedding")
    registry.register(first)
    registry.register(second)

    with registry._acquire(first) as backend:
        backend._model = object()
    assert backend.users == 0

    with registry._acquire(second):
        assert backend.loaded is False
    assert list(registry.status()["resident"]) == ["/models/b"]


def test_a_failed_request_is_recorded_so_the_model_is_not_shown_as_idle(monkeypatch):
    """Without this the dashboard's error row could never appear."""
    def explode(ref, cache_dir=None):
        raise FileNotFoundError("Model org/missing is not cached")

    monkeypatch.setattr("mtplx.hf_loader.resolve_model_path", explode)
    registry = RetrievalRegistry()
    registry.register(RetrievalSpec("broken", "org/missing", "embedding"))

    with pytest.raises(FileNotFoundError):
        registry.embed(["text"])

    entry = {e["id"]: e for e in registry.descriptors()}["broken"]
    assert entry["errors"] == 1
    assert "FileNotFoundError" in entry["lastError"]
    assert entry["requests"] == 0


def test_a_successful_request_clears_a_previous_error():
    stats = RetrievalStats()
    stats.record_error(ValueError("boom"))
    assert stats.last_error is not None
    stats.record(items=1, tokens=4, seconds=0.1)
    assert stats.last_error is None
    assert stats.errors == 1


def test_embeddings_report_real_token_usage():
    response = _client(_StubRegistry()).post("/v1/embeddings", json={"input": ["a", "b"]})
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 14
    assert usage["total_tokens"] == 14


def test_rerank_reports_real_token_usage():
    response = _client(_StubRegistry()).post(
        "/v1/rerank", json={"query": "q", "documents": ["a", "b", "c"]}
    )
    assert response.json()["usage"]["total_tokens"] == 33


def test_the_cap_is_restored_once_overlapping_requests_finish(monkeypatch):
    """Surplus from concurrent pins must not stay loaded until the next request."""
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry(max_resident=1)
    first = RetrievalSpec("a", "org/a", "embedding")
    second = RetrievalSpec("b", "org/b", "embedding")
    registry.register(first)
    registry.register(second)

    with registry._acquire(first) as a:
        a._model = object()
        with registry._acquire(second) as b:
            b._model = object()
            # Both pinned: exceeding the cap here is the deliberate trade-off.
            assert len(registry.status()["resident"]) == 2

    # Once the pins are gone the cap must hold again without further traffic.
    resident = registry.status()["resident"]
    assert len(resident) == 1
    assert resident == ["/models/b"]
    assert a.loaded is False


def test_resident_bytes_counts_a_shared_backend_once(monkeypatch):
    """One reference in both roles occupies its weights once, not twice."""
    _fixed_resolver(monkeypatch, {"org/dual": "/models/dual"})
    registry = RetrievalRegistry()
    registry.register(RetrievalSpec("dual", "org/dual", "embedding"))
    registry.register(RetrievalSpec("dual", "org/dual", "rerank"))

    with registry._acquire(RetrievalSpec("dual", "org/dual", "embedding")) as backend:
        backend._model = object()
        backend.weight_bytes = 4_000_000_000

    status = registry.status()
    assert status["resident_bytes"] == 4_000_000_000
    assert len(status["models"]) == 2


def test_resident_bytes_excludes_unloaded_models(monkeypatch):
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry()
    registry.register(RetrievalSpec("a", "org/a", "embedding"))
    assert registry.status()["resident_bytes"] == 0


# ---- idle standby ---------------------------------------------------------


def _loaded_backend(registry, spec, *, idle_for=0.0, weight_bytes=1_000_000_000):
    import time as _time
    with registry._acquire(spec) as backend:
        backend._model = object()
        backend.weight_bytes = weight_bytes
    backend.last_used_s = _time.time() - idle_for
    return backend


def test_idle_release_is_off_by_default(monkeypatch):
    """A daemon without a configured timeout must behave exactly as before."""
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry()
    spec = RetrievalSpec("a", "org/a", "embedding")
    registry.register(spec)
    backend = _loaded_backend(registry, spec, idle_for=99_999)

    assert registry.unload_idle() == {"unloaded": [], "freed_bytes": 0}
    assert backend.loaded is True


def test_idle_models_are_unloaded_and_freed_bytes_reported(monkeypatch):
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry(idle_timeout_s=60)
    spec = RetrievalSpec("a", "org/a", "embedding")
    registry.register(spec)
    backend = _loaded_backend(registry, spec, idle_for=120, weight_bytes=4_000_000_000)

    released = registry.unload_idle()
    assert released["freed_bytes"] == 4_000_000_000
    assert released["unloaded"] == ["/models/a"]
    assert backend.loaded is False
    assert registry.status()["resident"] == []


def test_a_recently_used_model_survives(monkeypatch):
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry(idle_timeout_s=600)
    spec = RetrievalSpec("a", "org/a", "embedding")
    registry.register(spec)
    backend = _loaded_backend(registry, spec, idle_for=5)

    assert registry.unload_idle()["unloaded"] == []
    assert backend.loaded is True


def test_a_model_in_use_is_never_released_by_the_watcher(monkeypatch):
    """The watcher must not pull weights out of an in-flight request."""
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry(idle_timeout_s=1)
    spec = RetrievalSpec("a", "org/a", "embedding")
    registry.register(spec)
    with registry._acquire(spec) as backend:
        backend._model = object()
        backend.last_used_s = 0  # ancient, but pinned
        assert registry.unload_idle()["unloaded"] == []
        assert backend.loaded is True


def test_status_and_descriptors_expose_the_idle_state(monkeypatch):
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry(idle_timeout_s=300)
    spec = RetrievalSpec("a", "org/a", "embedding")
    registry.register(spec)
    _loaded_backend(registry, spec, idle_for=42)

    status = registry.status()
    assert status["idle_timeout_s"] == 300
    entry = {e["id"]: e for e in status["models"]}["a"]
    assert entry["idleSeconds"] >= 42


def test_idle_seconds_is_absent_for_an_unloaded_model():
    entry = {e["id"]: e for e in _registry().descriptors()}["embed-a"]
    assert entry["idleSeconds"] is None


def test_pressure_release_ignores_the_idle_timeout_but_not_pinning(monkeypatch):
    """Under memory pressure any unpinned model is fair game, idle or not."""
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry(idle_timeout_s=0)  # idle release disabled
    spec = RetrievalSpec("a", "org/a", "embedding")
    registry.register(spec)
    backend = _loaded_backend(registry, spec, idle_for=0, weight_bytes=3_000_000_000)

    # threshold 0 is what the pressure guard passes: just-used but unpinned.
    released = registry.unload_idle(0)
    assert released["freed_bytes"] == 3_000_000_000
    assert backend.loaded is False


def test_pressure_release_still_spares_a_pinned_model(monkeypatch):
    _fixed_resolver(monkeypatch)
    registry = RetrievalRegistry()
    spec = RetrievalSpec("a", "org/a", "embedding")
    registry.register(spec)
    with registry._acquire(spec) as backend:
        backend._model = object()
        assert registry.unload_idle(0)["unloaded"] == []
        assert backend.loaded is True
