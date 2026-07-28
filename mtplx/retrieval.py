"""Embedding and reranking models served alongside the MTPLX chat runtime.

MTPLX's generation path is a multi-token-prediction decoder: it exists to make
*next-token* prediction cheaper, which is meaningless for a model that emits a
vector instead of a token stream. Retrieval models therefore do not go through
the MTP runtime at all — they run the transformer stack directly:

* **Embedding** — take the final hidden state of the last real token and
  L2-normalise it. Qwen3-Embedding and its relatives append ``<|endoftext|>``
  and pool that position, which is what ``pooling="last"`` reproduces.
* **Reranking** — build the yes/no judging prompt the Qwen3-Reranker family was
  trained on and take a softmax over the ``yes``/``no`` logits at the last real
  position.

Batches are padded on the **right**. With causal attention a real token never
attends to a later pad token, so right padding leaves every real position
bit-identical to the unpadded run — left padding would corrupt it.

A single set of weights can back several served ids, and one model can serve
both roles at once: backends are cached by resolved filesystem path, so
registering the same reference as an embedder and as a reranker loads it once.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Role = Literal["embedding", "rerank"]

DEFAULT_MAX_TOKENS = 8192
DEFAULT_EMBEDDING_BATCH = 8
DEFAULT_RERANK_BATCH = 4
DEFAULT_MAX_RESIDENT = 2

EOD_TOKEN = "<|endoftext|>"
QUERY_INSTRUCTION_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"

RERANK_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
RERANK_PREFIX = (
    f"<|im_start|>system\n{RERANK_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RERANK_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class RetrievalError(RuntimeError):
    """Raised when a retrieval request cannot be served."""


@dataclass
class RetrievalStats:
    """Live counters for one served retrieval model.

    Configuration without observability is a guess: these make it visible
    whether a configured model ever loaded, whether it is being used, and what
    it costs per item.
    """

    requests: int = 0
    items: int = 0
    compute_seconds: float = 0.0
    load_seconds: float = 0.0
    last_used_s: float | None = None
    last_error: str | None = None

    def record(self, *, items: int, seconds: float) -> None:
        self.requests += 1
        self.items += items
        self.compute_seconds += seconds
        self.last_used_s = time.time()
        self.last_error = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "items": self.items,
            "computeSeconds": round(self.compute_seconds, 4),
            "loadSeconds": round(self.load_seconds, 3),
            "lastUsedS": self.last_used_s,
            "lastError": self.last_error,
            "itemsPerSecond": (
                round(self.items / self.compute_seconds, 2)
                if self.compute_seconds > 0
                else None
            ),
            "avgLatencyMs": (
                round(1000.0 * self.compute_seconds / self.requests, 1)
                if self.requests
                else None
            ),
        }


@dataclass(frozen=True)
class RetrievalSpec:
    """One served retrieval model."""

    served_id: str
    model_ref: str
    role: Role
    pooling: str = "last"
    max_tokens: int = DEFAULT_MAX_TOKENS
    batch_size: int = 0
    instruction: str | None = None

    def effective_batch_size(self) -> int:
        if self.batch_size > 0:
            return self.batch_size
        return DEFAULT_EMBEDDING_BATCH if self.role == "embedding" else DEFAULT_RERANK_BATCH


def parse_model_flag(value: str, role: Role) -> RetrievalSpec:
    """Parse a ``REF`` or ``REF=SERVED_ID`` command line value.

    Splitting on the last ``=`` keeps Hugging Face ids (``org/name``) and
    filesystem paths intact while still allowing an explicit alias.
    """
    text = str(value).strip()
    if not text:
        raise ValueError("model reference must not be empty")
    if "=" in text:
        model_ref, _, served_id = text.rpartition("=")
        model_ref = model_ref.strip()
        served_id = served_id.strip()
        if not model_ref or not served_id:
            raise ValueError(f"invalid model flag {value!r}; expected REF or REF=SERVED_ID")
    else:
        model_ref = text
        served_id = default_served_id(text)
    return RetrievalSpec(served_id=served_id, model_ref=model_ref, role=role)


def default_served_id(model_ref: str) -> str:
    """Return the short id clients use for a model reference."""
    return str(model_ref).rstrip("/").rsplit("/", 1)[-1]


def _right_padded(sequences: list[list[int]], pad_id: int) -> tuple[Any, list[int]]:
    """Pad token sequences on the right and report their true lengths."""
    import mlx.core as mx

    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths)
    padded = [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences]
    return mx.array(padded), lengths


class _Backend:
    """One set of weights, loaded lazily and shared across served ids."""

    def __init__(self, model_ref: str, path: Path) -> None:
        self.model_ref = model_ref
        self.path = path
        self.lock = threading.RLock()
        self.load_seconds = 0.0
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> tuple[Any, Any]:
        with self.lock:
            if self._model is None:
                from mlx_lm import load

                started = time.time()
                self._model, self._tokenizer = load(str(self.path))
                self.load_seconds = time.time() - started
            return self._model, self._tokenizer

    def unload(self) -> None:
        with self.lock:
            self._model = None
            self._tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def pad_id(self, tokenizer: Any) -> int:
        token = tokenizer.convert_tokens_to_ids(EOD_TOKEN)
        if token is None:
            token = getattr(tokenizer, "eos_token_id", 0) or 0
        return int(token)


class RetrievalRegistry:
    """Serves every configured embedding and reranking model.

    Backends are keyed by resolved path so one model registered under several
    served ids — or under both roles — occupies memory once.
    """

    def __init__(self, *, max_resident: int = DEFAULT_MAX_RESIDENT, cache_dir: str | Path | None = None) -> None:
        self.max_resident = max(1, int(max_resident))
        self.cache_dir = cache_dir
        self._specs: dict[tuple[Role, str], RetrievalSpec] = {}
        self._backends: dict[str, _Backend] = {}
        self._resident: OrderedDict[str, None] = OrderedDict()
        self._stats: dict[tuple[Role, str], RetrievalStats] = {}
        self._lock = threading.RLock()

    # ── registration ────────────────────────────────────────────────

    def register(self, spec: RetrievalSpec) -> None:
        """Add a served model. Re-registering the same id replaces it."""
        with self._lock:
            self._specs[(spec.role, spec.served_id)] = spec

    def register_all(self, specs: Iterable[RetrievalSpec]) -> None:
        for spec in specs:
            self.register(spec)

    @property
    def enabled(self) -> bool:
        return bool(self._specs)

    def _stats_for(self, spec: RetrievalSpec) -> RetrievalStats:
        with self._lock:
            return self._stats.setdefault((spec.role, spec.served_id), RetrievalStats())

    def specs_for_role(self, role: Role) -> list[RetrievalSpec]:
        with self._lock:
            return [spec for (spec_role, _), spec in sorted(self._specs.items()) if spec_role == role]

    def descriptors(self) -> list[dict[str, Any]]:
        """Describe every served retrieval model for ``/v1/models``."""
        entries: list[dict[str, Any]] = []
        with self._lock:
            for (role, served_id), spec in sorted(self._specs.items()):
                backend = self._backends.get(spec.model_ref)
                stats = self._stats.get((role, served_id)) or RetrievalStats()
                if backend is not None and backend.load_seconds:
                    stats.load_seconds = backend.load_seconds
                entries.append(
                    {
                        "id": served_id,
                        "role": role,
                        "model_ref": spec.model_ref,
                        "loaded": bool(backend is not None and backend.loaded),
                        "resident": spec.model_ref in self._resident,
                        "max_tokens": spec.max_tokens,
                        "batch_size": spec.effective_batch_size(),
                        **stats.to_dict(),
                    }
                )
        return entries

    def status(self) -> dict[str, Any]:
        """Return a snapshot for diagnostics and the dashboard."""
        with self._lock:
            resident = list(self._resident.keys())
        return {
            "enabled": self.enabled,
            "max_resident": self.max_resident,
            "resident": resident,
            "models": self.descriptors(),
        }

    # ── resolution ──────────────────────────────────────────────────

    def _spec(self, role: Role, requested: str | None) -> RetrievalSpec:
        candidates = self.specs_for_role(role)
        if not candidates:
            raise RetrievalError(f"no {role} model is configured")
        if not requested:
            return candidates[0]
        wanted = str(requested).strip()
        for spec in candidates:
            if wanted in {spec.served_id, spec.model_ref} or default_served_id(wanted) == spec.served_id:
                return spec
        served = ", ".join(spec.served_id for spec in candidates)
        raise RetrievalError(f"unknown {role} model {requested!r}; served: {served}")

    def _backend(self, spec: RetrievalSpec) -> _Backend:
        with self._lock:
            backend = self._backends.get(spec.model_ref)
            if backend is None:
                from .hf_loader import resolve_model_path

                path = resolve_model_path(spec.model_ref, cache_dir=self.cache_dir)
                backend = _Backend(spec.model_ref, path)
                self._backends[spec.model_ref] = backend
            # A slot is reserved on acquisition, not on completed load. Two
            # first-use requests for different models run concurrently, and
            # loading happens outside this lock: if residency counted only
            # finished loads, each request would see the other as absent, skip
            # eviction, and both models would end up resident — the cap would
            # be silently exceeded exactly when memory is tightest.
            self._resident[spec.model_ref] = None
            self._resident.move_to_end(spec.model_ref)
            while len(self._resident) > self.max_resident:
                oldest = next(iter(self._resident))
                if oldest == spec.model_ref:
                    break
                self._backends[oldest].unload()
                self._resident.pop(oldest, None)
        return backend

    def unload_all(self) -> None:
        """Drop every resident retrieval model."""
        with self._lock:
            backends = list(self._backends.values())
            self._resident.clear()
        for backend in backends:
            backend.unload()

    # ── inference ───────────────────────────────────────────────────

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        instruction: str | None = None,
    ) -> tuple[list[list[float]], RetrievalSpec]:
        """Embed texts in input order, returning vectors and the spec used."""
        spec = self._spec("embedding", model)
        if not texts:
            return [], spec
        stats = self._stats_for(spec)
        backend = self._backend(spec)
        model_obj, tokenizer = backend.ensure_loaded()
        # Started after the load so a cold first request does not report the
        # weight load as inference latency — that would make a fast model look
        # ten times slower than it is. Load cost is reported separately.
        started = time.time()
        effective_instruction = instruction if instruction is not None else spec.instruction
        prepared = [
            QUERY_INSTRUCTION_TEMPLATE.format(instruction=effective_instruction, text=text)
            if effective_instruction
            else text
            for text in texts
        ]
        pad_id = backend.pad_id(tokenizer)
        vectors: list[list[float]] = []
        batch = spec.effective_batch_size()
        import mlx.core as mx

        with backend.lock:
            for start in range(0, len(prepared), batch):
                chunk = prepared[start : start + batch]
                sequences = [self._encode_embedding(tokenizer, text, spec, pad_id) for text in chunk]
                inputs, lengths = _right_padded(sequences, pad_id)
                hidden = model_obj.model(inputs)
                if spec.pooling == "mean":
                    pooled = mx.stack(
                        [hidden[row, :length, :].mean(axis=0) for row, length in enumerate(lengths)]
                    )
                else:
                    pooled = mx.stack(
                        [hidden[row, length - 1, :] for row, length in enumerate(lengths)]
                    )
                pooled = pooled.astype(mx.float32)
                normalised = pooled / mx.linalg.norm(pooled, axis=-1, keepdims=True)
                mx.eval(normalised)
                vectors.extend(normalised.tolist())
                del hidden, pooled, normalised
                mx.clear_cache()
        stats.record(items=len(texts), seconds=time.time() - started)
        return vectors, spec

    def _encode_embedding(
        self, tokenizer: Any, text: str, spec: RetrievalSpec, pad_id: int
    ) -> list[int]:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        ids = ids[: max(1, spec.max_tokens - 1)]
        ids.append(pad_id)
        return ids

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        instruction: str | None = None,
    ) -> tuple[list[float], RetrievalSpec]:
        """Score documents against a query, returning scores in input order."""
        spec = self._spec("rerank", model)
        if not documents:
            return [], spec
        stats = self._stats_for(spec)
        backend = self._backend(spec)
        model_obj, tokenizer = backend.ensure_loaded()
        # Started after the load so a cold first request does not report the
        # weight load as inference latency — that would make a fast model look
        # ten times slower than it is. Load cost is reported separately.
        started = time.time()
        effective_instruction = (
            instruction or spec.instruction or RERANK_DEFAULT_INSTRUCTION
        )
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
        if yes_id is None or no_id is None:
            raise RetrievalError(
                f"{spec.served_id} has no yes/no tokens; it is not a Qwen-style reranker"
            )
        pad_id = backend.pad_id(tokenizer)
        scores: list[float] = []
        batch = spec.effective_batch_size()
        import mlx.core as mx

        with backend.lock:
            for start in range(0, len(documents), batch):
                chunk = documents[start : start + batch]
                sequences = [
                    self._encode_rerank(tokenizer, query, document, effective_instruction, spec)
                    for document in chunk
                ]
                inputs, lengths = _right_padded(sequences, pad_id)
                logits = model_obj(inputs)
                pairs = mx.stack(
                    [
                        mx.stack(
                            [logits[row, length - 1, no_id], logits[row, length - 1, yes_id]]
                        )
                        for row, length in enumerate(lengths)
                    ]
                )
                probabilities = mx.softmax(pairs.astype(mx.float32), axis=-1)[:, 1]
                mx.eval(probabilities)
                scores.extend(float(value) for value in probabilities.tolist())
                del logits, pairs, probabilities
                mx.clear_cache()
        stats.record(items=len(documents), seconds=time.time() - started)
        return scores, spec

    def _encode_rerank(
        self,
        tokenizer: Any,
        query: str,
        document: str,
        instruction: str,
        spec: RetrievalSpec,
    ) -> list[int]:
        prefix = list(tokenizer.encode(RERANK_PREFIX, add_special_tokens=False))
        suffix = list(tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False))
        body = list(
            tokenizer.encode(
                f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
                add_special_tokens=False,
            )
        )
        budget = max(1, spec.max_tokens - len(prefix) - len(suffix))
        return prefix + body[:budget] + suffix


def registry_from_args(args: Any) -> RetrievalRegistry:
    """Build a registry from parsed CLI arguments.

    Unknown attributes are tolerated so callers that construct a bare namespace
    (tests, embedded use) do not have to populate every retrieval flag.
    """
    # The chat model is resolved to an absolute path before the server
    # subprocess starts, so that process carries no cache directory of its own.
    # Retrieval references stay symbolic and are resolved here, which means the
    # directory has to be threaded through explicitly or a model pulled into a
    # custom --cache-dir is invisible despite being on disk.
    cache_dir = (
        getattr(args, "retrieval_cache_dir", None)
        or getattr(args, "cache_dir", None)
        or getattr(args, "model_dir", None)
    )
    registry = RetrievalRegistry(
        max_resident=int(getattr(args, "retrieval_max_resident", DEFAULT_MAX_RESIDENT) or DEFAULT_MAX_RESIDENT),
        cache_dir=cache_dir,
    )
    for role, attribute in (("embedding", "embedding_model"), ("rerank", "reranker_model")):
        for value in getattr(args, attribute, None) or []:
            spec = parse_model_flag(value, role)  # type: ignore[arg-type]
            max_tokens = int(getattr(args, "retrieval_max_tokens", 0) or 0)
            if max_tokens > 0:
                spec = RetrievalSpec(
                    served_id=spec.served_id,
                    model_ref=spec.model_ref,
                    role=spec.role,
                    pooling=spec.pooling,
                    max_tokens=max_tokens,
                    batch_size=spec.batch_size,
                    instruction=spec.instruction,
                )
            registry.register(spec)
    return registry
