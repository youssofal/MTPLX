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
from contextlib import contextmanager
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Role = Literal["embedding", "rerank"]

DEFAULT_MAX_TOKENS = 8192
DEFAULT_EMBEDDING_BATCH = 8
DEFAULT_RERANK_BATCH = 4
DEFAULT_MAX_RESIDENT = 2

# A padded batch costs rows * longest-row token positions no matter how short
# its other rows are, so the batch size alone is the wrong unit: it is generous
# where batching does not pay and stingy where it does. Measured on a 4-bit
# Qwen3-Embedding-8B, long sequences gain nothing from batching (~950 ms each at
# 853 tokens, whether run alone or eight at a time) while short ones gain about
# threefold. Capping the product instead lets short texts pack densely and drops
# long ones into batches of their own; 2048 keeps a batch under a second.
BATCH_TOKEN_BUDGET = 2048

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
    tokens: int = 0
    errors: int = 0
    compute_seconds: float = 0.0
    load_seconds: float = 0.0
    last_used_s: float | None = None
    last_error: str | None = None

    def record(self, *, items: int, tokens: int, seconds: float) -> None:
        self.requests += 1
        self.items += items
        self.tokens += tokens
        self.compute_seconds += seconds
        self.last_used_s = time.time()
        self.last_error = None

    def record_error(self, error: BaseException) -> None:
        """Remember a failure so a broken model is not displayed as merely idle."""
        self.errors += 1
        self.last_used_s = time.time()
        self.last_error = f"{type(error).__name__}: {error}"[:400]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "items": self.items,
            "tokens": self.tokens,
            "errors": self.errors,
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


def _weight_bytes(path: Path) -> int:
    """Sum the shard sizes of a model directory.

    MLX holds weights in unified memory that never appears in the process RSS,
    so file size is the only honest attribution available — the same basis
    MTPLX already uses for the chat model.
    """
    try:
        return sum(
            item.stat().st_size
            for item in Path(path).glob("*.safetensors")
            if item.is_file()
        )
    except OSError:
        return 0


def _right_padded(sequences: list[list[int]], pad_id: int) -> tuple[Any, list[int]]:
    """Pad token sequences on the right and report their true lengths."""
    import mlx.core as mx

    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths)
    padded = [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences]
    return mx.array(padded), lengths


def _plan_batches(lengths: list[int], max_rows: int) -> list[list[int]]:
    """Group sequence indices into batches that are cheap to pad.

    Every row of a batch is padded out to the longest row in it, so the model
    pays ``rows * width`` token positions no matter how short the other rows
    are. Grouping similar lengths together is what keeps that product close to
    the work actually asked for.

    Args:
        lengths: token count of each sequence, in caller order.
        max_rows: the configured batch size — a batch may never exceed it.

    Returns:
        Batches of indices into ``lengths``. Every index appears exactly once;
        the caller scatters results back into input order, so the grouping here
        is free to reorder.
    """
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        # Ascending order means the incoming sequence is always the widest, so
        # it alone decides what the batch will be padded to.
        width = lengths[index]
        if current and (
            len(current) >= max_rows or width * (len(current) + 1) > BATCH_TOKEN_BUDGET
        ):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def _load_sibling_module(directory: Path, filename: str, name: str) -> Any:
    """Import a module that ships beside a checkpoint, without touching sys.path.

    jina distributes its MLX inference code inside the model repository rather
    than as an installable package. Loading it by file location keeps it out
    of the global module namespace, so two checkpoints that both ship a
    ``model.py`` cannot shadow each other.
    """
    import importlib.util

    path = Path(directory) / filename
    if not path.is_file():
        raise FileNotFoundError(f"{filename} not found in {directory}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_jina_embedding_checkpoint(path: Path) -> bool:
    """jina-embeddings-v5 ships its own model.py/utils.py; mlx_lm cannot load it."""
    return (path / "utils.py").is_file() and (path / "model.py").is_file()


def _is_jina_reranker_checkpoint(path: Path) -> bool:
    """jina-reranker-v3.5 scores through a projector head, not yes/no logits."""
    return (path / "rerank.py").is_file() and (path / "projector.safetensors").is_file()


class _JinaEmbedBackend:
    """jina-embeddings-v5 backend using the model's own asymmetric task types.

    Qwen3-Embedding expresses the query/passage asymmetry through an
    instruction prefix; jina expresses it by swapping a LoRA adapter. The
    ``embed`` signature matches that of the Qwen3 path — an instruction means
    "this is a question" — so ``RetrievalRegistry`` does not need to know
    which model answered.
    """

    def __init__(self, model_ref: str, path: Path) -> None:
        self.model_ref = model_ref
        self.path = path
        self.lock = threading.RLock()
        self.load_seconds = 0.0
        self.weight_bytes = 0
        self.last_used_s = time.time()
        self.users = 0
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> Any:
        with self.lock:
            if self._model is None:
                started = time.time()
                utils = _load_sibling_module(self.path, "utils.py", "jina_v5_utils")
                model = utils.load_model(str(self.path))
                model.switch_task("retrieval")
                self._model = model
                self.load_seconds = time.time() - started
                self.weight_bytes = _weight_bytes(self.path)
            return self._model

    def unload(self) -> None:
        with self.lock:
            self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def embed(self, texts: list[str], *, instruction: str | None) -> tuple[list[list[float]], int]:
        """Embed texts in input order, returning vectors and the true token count."""
        import mlx.core as mx

        model = self.ensure_loaded()
        task_type = "retrieval.query" if instruction else "retrieval.passage"
        tokens = sum(len(model.tokenizer.encode(text, add_special_tokens=False).ids) for text in texts)
        vectors: list[list[float]] = []
        with self.lock:
            encoded = model.encode(texts, task_type=task_type)
            stacked = encoded if isinstance(encoded, mx.array) else mx.stack(list(encoded))
            pooled = stacked.astype(mx.float32)
            normalised = pooled / mx.linalg.norm(pooled, axis=-1, keepdims=True)
            mx.eval(normalised)
            vectors = normalised.tolist()
            del encoded, stacked, pooled, normalised
            mx.clear_cache()
        return vectors, tokens


class _JinaRerankBackend:
    """jina-reranker-v3.5 backend, scoring a whole candidate list in one pass.

    Qwen3-Reranker judges one query/document pair per row, so N candidates
    cost N sequences. jina is listwise: the query and up to ``block_size``
    documents share a single forward pass — measured on real recalls, 50
    candidates took ~1.0 s here against ~18.7 s for the Qwen3 4B.
    """

    def __init__(self, model_ref: str, path: Path) -> None:
        self.model_ref = model_ref
        self.path = path
        self.lock = threading.RLock()
        self.load_seconds = 0.0
        self.weight_bytes = 0
        self.last_used_s = time.time()
        self.users = 0
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> Any:
        with self.lock:
            if self._model is None:
                import sys

                started = time.time()
                # rerank.py does `import modeling`, a bare name that only
                # resolves if the checkpoint directory is importable.
                # Registering the module under that name first satisfies it
                # without putting the directory on sys.path, where it would
                # shadow anything else named `modeling`.
                modeling = _load_sibling_module(self.path, "modeling.py", "modeling")
                previous = sys.modules.get("modeling")
                sys.modules["modeling"] = modeling
                try:
                    rerank_module = _load_sibling_module(
                        self.path, "rerank.py", "jina_v35_rerank"
                    )
                finally:
                    if previous is None:
                        sys.modules.pop("modeling", None)
                    else:
                        sys.modules["modeling"] = previous
                self._model = rerank_module.MLXReranker(str(self.path))
                self.load_seconds = time.time() - started
                self.weight_bytes = _weight_bytes(self.path)
            return self._model

    def unload(self) -> None:
        with self.lock:
            self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def score(self, query: str, documents: list[str], *, instruction: str) -> tuple[list[float], int]:
        """Return a relevance probability per document, in input order, plus token count.

        ``instruction`` is accepted for interface parity and ignored: jina
        encodes the ranking task in its projector head rather than in a prompt.
        """
        model = self.ensure_loaded()
        tokens = len(model.tokenizer.encode(query, add_special_tokens=False).ids)
        tokens += sum(
            len(model.tokenizer.encode(document, add_special_tokens=False).ids)
            for document in documents
        )
        scores = [0.0] * len(documents)
        with self.lock:
            results = model.rerank(query, documents)
        for entry in results:
            scores[int(entry["index"])] = float(entry["relevance_score"])
        return scores, tokens


class _Backend:
    """One set of weights, loaded lazily and shared across served ids."""

    def __init__(self, model_ref: str, path: Path) -> None:
        self.model_ref = model_ref
        self.path = path
        self.lock = threading.RLock()
        self.load_seconds = 0.0
        self.weight_bytes = 0
        self.last_used_s = time.time()
        # Requests currently holding this backend. Evicting a pinned backend
        # would unload weights another thread is about to use, which then
        # reloads them — leaving two copies live while only one is counted.
        self.users = 0
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
                self.weight_bytes = _weight_bytes(self.path)
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

    Backends are keyed by resolved filesystem path, so one model registered
    under several served ids, under both roles, or under both a Hugging Face id
    and the local path it resolves to occupies memory once.
    """

    def __init__(
        self,
        *,
        max_resident: int = DEFAULT_MAX_RESIDENT,
        cache_dir: str | Path | None = None,
        idle_timeout_s: float = 0.0,
    ) -> None:
        self.max_resident = max(1, int(max_resident))
        self.cache_dir = cache_dir
        # 0 disables idle release entirely, which keeps a daemon that never
        # configured a timeout behaving exactly as before.
        self.idle_timeout_s = max(0.0, float(idle_timeout_s))
        self._specs: dict[tuple[Role, str], RetrievalSpec] = {}
        self._backends: dict[str, _Backend] = {}
        self._resident: OrderedDict[str, None] = OrderedDict()
        self._stats: dict[tuple[Role, str], RetrievalStats] = {}
        self._keys: dict[str, str] = {}
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
        now = time.time()
        with self._lock:
            for (role, served_id), spec in sorted(self._specs.items()):
                # Look the backend up by its resolved key, the same one used
                # for residency — the raw reference would miss a backend shared
                # with another alias.
                key = self._keys.get(spec.model_ref)
                backend = self._backends.get(key) if key else None
                stats = self._stats.get((role, served_id)) or RetrievalStats()
                if backend is not None and backend.load_seconds:
                    stats.load_seconds = backend.load_seconds
                entries.append(
                    {
                        "id": served_id,
                        "role": role,
                        "model_ref": spec.model_ref,
                        "loaded": bool(backend is not None and backend.loaded),
                        "weightBytes": backend.weight_bytes if backend is not None else 0,
                        "idleSeconds": (
                            round(now - backend.last_used_s, 1)
                            if backend is not None and backend.loaded
                            else None
                        ),
                        "resident": bool(key and key in self._resident),
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
        models = self.descriptors()
        with self._lock:
            # Sum per backend, not per served id: one model serving both roles
            # occupies its weights once and must not be counted twice.
            resident_bytes = sum(
                backend.weight_bytes
                for key, backend in self._backends.items()
                if key in self._resident and backend.loaded
            )
        return {
            "enabled": self.enabled,
            "max_resident": self.max_resident,
            "idle_timeout_s": self.idle_timeout_s,
            "resident": resident,
            "resident_bytes": resident_bytes,
            "models": models,
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

    def _backend_key(self, spec: RetrievalSpec) -> str:
        """Return the canonical residency key for a spec: its resolved path.

        Keying by the raw reference would give a Hugging Face id and the local
        path it resolves to two separate backends, loading the same weights
        twice and counting them twice against the cap — the opposite of the
        one-model-two-roles guarantee.
        """
        cached = self._keys.get(spec.model_ref)
        if cached is not None:
            return cached
        from .hf_loader import resolve_model_path

        path = resolve_model_path(spec.model_ref, cache_dir=self.cache_dir)
        key = str(Path(path).resolve())
        with self._lock:
            self._keys[spec.model_ref] = key
        return key

    @contextmanager
    def _acquire(self, spec: RetrievalSpec):
        """Yield a backend pinned for the whole load-and-inference lifetime.

        Reserving a slot is not enough on its own: a request that has left this
        method but not yet finished still holds its weights, so evicting them
        frees nothing and forces a reload. Only unpinned backends are evicted;
        when every resident backend is busy the cap is briefly exceeded rather
        than unloading a model that is in use.
        """
        key = self._backend_key(spec)
        with self._lock:
            backend = self._backends.get(key)
            if backend is None:
                # Which loader a checkpoint needs is read from the checkpoint
                # itself, so pointing a spec at a jina repository is all that
                # is required — no separate flag or served-id convention.
                path = Path(key)
                if spec.role == "embedding" and _is_jina_embedding_checkpoint(path):
                    backend = _JinaEmbedBackend(spec.model_ref, path)
                elif spec.role == "rerank" and _is_jina_reranker_checkpoint(path):
                    backend = _JinaRerankBackend(spec.model_ref, path)
                else:
                    backend = _Backend(spec.model_ref, path)
                self._backends[key] = backend
            backend.users += 1
            self._resident[key] = None
            self._resident.move_to_end(key)
            self._evict_locked()
        try:
            yield backend
        finally:
            with self._lock:
                backend.users = max(0, backend.users - 1)
                backend.last_used_s = time.time()
                # Overlapping requests can legitimately push past the cap while
                # every backend is pinned. Without retrying here the surplus
                # would stay loaded until some later request happened to
                # trigger eviction — several GB held for no reason.
                self._evict_locked()

    def _evict_locked(self) -> None:
        """Unload unpinned backends until the cap holds. Caller holds the lock.

        Oldest first, and never the most recently used entry: that is the one
        just acquired or just released, and dropping it would evict the newest
        model whenever an older one happens to be pinned — the opposite of LRU.
        """
        candidates = list(self._resident)[:-1]
        for candidate in candidates:
            if len(self._resident) <= self.max_resident:
                break
            victim = self._backends[candidate]
            if victim.users:
                continue
            victim.unload()
            self._resident.pop(candidate, None)

    def _backend(self, spec: RetrievalSpec) -> _Backend:
        """Acquire without pinning — for tests and introspection only."""
        with self._acquire(spec) as backend:
            return backend

    def unload_idle(self, older_than_s: float | None = None) -> dict[str, Any]:
        """Unload loaded, unpinned backends idle for longer than the threshold.

        Returns what was freed so the caller can log it and the dashboard can
        show it. A backend inside an in-flight request is never a candidate:
        its pin count is checked under the same lock that hands it out.
        """
        # An omitted threshold means "use the configured timeout", where 0
        # disables idle release. An explicit threshold is always honoured,
        # including 0 — that is how the memory-pressure guard asks for every
        # unpinned model regardless of how recently it was used.
        if older_than_s is None:
            if self.idle_timeout_s <= 0:
                return {"unloaded": [], "freed_bytes": 0}
            threshold = self.idle_timeout_s
        else:
            threshold = max(0.0, float(older_than_s))
        now = time.time()
        victims: list[_Backend] = []
        with self._lock:
            for key in list(self._resident):
                backend = self._backends.get(key)
                if backend is None or not backend.loaded or backend.users:
                    continue
                if threshold > 0 and now - backend.last_used_s < threshold:
                    continue
                victims.append(backend)
                self._resident.pop(key, None)
        freed = 0
        unloaded: list[str] = []
        for backend in victims:
            freed += backend.weight_bytes
            unloaded.append(str(backend.path))
            backend.unload()
        return {"unloaded": unloaded, "freed_bytes": freed}

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
    ) -> tuple[list[list[float]], RetrievalSpec, int]:
        """Embed texts in order, returning vectors, the spec used and token count."""
        spec = self._spec("embedding", model)
        if not texts:
            return [], spec, 0
        stats = self._stats_for(spec)
        started = time.time()
        try:
            with self._acquire(spec) as backend:
                if isinstance(backend, _JinaEmbedBackend):
                    # jina expresses the query/passage asymmetry through an
                    # adapter switch rather than a prompt, so it skips the
                    # tokenize/pad/pool path entirely — that path assumes a
                    # tokenizer object and yes/no-style logits that jina does
                    # not expose.
                    backend.ensure_loaded()
                    started = time.time()  # after load, same accounting as the Qwen path
                    effective_instruction = (
                        instruction if instruction is not None else spec.instruction
                    )
                    vectors, tokens = backend.embed(texts, instruction=effective_instruction)
                    stats.record(items=len(texts), tokens=tokens, seconds=time.time() - started)
                    return vectors, spec, tokens
                model_obj, tokenizer = backend.ensure_loaded()
                effective_instruction = instruction if instruction is not None else spec.instruction
                prepared = [
                    QUERY_INSTRUCTION_TEMPLATE.format(instruction=effective_instruction, text=text)
                    if effective_instruction
                    else text
                    for text in texts
                ]
                pad_id = backend.pad_id(tokenizer)
                # Started after the load so a cold first request does not report
                # the weight load as inference latency — that would make a fast
                # model look ten times slower than it is. Load cost is reported
                # separately.
                started = time.time()
                import mlx.core as mx

                with backend.lock:
                    # Tokenise everything before batching. A batch costs
                    # rows * longest-row token positions, so the lengths have to
                    # be known before the batch boundaries can be drawn — a
                    # single long text dragged into a batch of short ones used to
                    # cost as much as eight long ones.
                    sequences = [
                        self._encode_embedding(tokenizer, text, spec, pad_id) for text in prepared
                    ]
                    tokens = sum(len(sequence) for sequence in sequences)
                    vectors: list[list[float]] = [[] for _ in sequences]
                    plan = _plan_batches(
                        [len(sequence) for sequence in sequences], spec.effective_batch_size()
                    )
                    for group in plan:
                        inputs, lengths = _right_padded(
                            [sequences[index] for index in group], pad_id
                        )
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
                        # Scatter, not extend: the plan is free to reorder, so
                        # the caller's order is restored here.
                        for index, vector in zip(group, normalised.tolist()):
                            vectors[index] = vector
                        del hidden, pooled, normalised
                        mx.clear_cache()
        except BaseException as error:
            stats.record_error(error)
            raise
        stats.record(items=len(texts), tokens=tokens, seconds=time.time() - started)
        return vectors, spec, tokens

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
    ) -> tuple[list[float], RetrievalSpec, int]:
        """Score documents in order, returning scores, the spec used and tokens."""
        spec = self._spec("rerank", model)
        if not documents:
            return [], spec, 0
        stats = self._stats_for(spec)
        try:
            with self._acquire(spec) as backend:
                if isinstance(backend, _JinaRerankBackend):
                    # jina scores the whole candidate list in one listwise
                    # pass through a projector head, so it has no yes/no
                    # logits to read and skips the tokenize/pad loop entirely.
                    backend.ensure_loaded()
                    started = time.time()  # after load, same accounting as the Qwen path
                    effective_instruction = (
                        instruction or spec.instruction or RERANK_DEFAULT_INSTRUCTION
                    )
                    scores, tokens = backend.score(
                        query, documents, instruction=effective_instruction
                    )
                    stats.record(items=len(documents), tokens=tokens, seconds=time.time() - started)
                    return scores, spec, tokens
                model_obj, tokenizer = backend.ensure_loaded()
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
                started = time.time()
                import mlx.core as mx

                with backend.lock:
                    # Same padding trap as embed(), and tighter here: every row
                    # carries the query as well as its document, so one long
                    # document inflates the whole batch.
                    sequences = [
                        self._encode_rerank(tokenizer, query, document, effective_instruction, spec)
                        for document in documents
                    ]
                    tokens = sum(len(sequence) for sequence in sequences)
                    scores: list[float] = [0.0] * len(sequences)
                    plan = _plan_batches(
                        [len(sequence) for sequence in sequences], spec.effective_batch_size()
                    )
                    for group in plan:
                        inputs, lengths = _right_padded(
                            [sequences[index] for index in group], pad_id
                        )
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
                        # Scatter, not extend: the plan reorders by length, and
                        # callers rank by index against their own document list.
                        for index, value in zip(group, probabilities.tolist()):
                            scores[index] = float(value)
                        del logits, pairs, probabilities
                        mx.clear_cache()
        except BaseException as error:
            stats.record_error(error)
            raise
        stats.record(items=len(documents), tokens=tokens, seconds=time.time() - started)
        return scores, spec, tokens

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
        idle_timeout_s=float(getattr(args, "retrieval_idle_timeout", 0) or 0),
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
