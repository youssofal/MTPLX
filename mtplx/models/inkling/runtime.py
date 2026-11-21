"""Target and speculative state share the conversation cache and live on the generation stream."""

from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime, _load_tokenizer_resilient

from . import expert_fusion
from .draft import Drafter
from .fast import engine_site
from .fast import round as fast_round
from .fast import round_program
from .fast import draft as fast_draft
from .fast.decoder import (DECLINED, DepthTableCheck, FastDecoder, FastPending, fast_enabled,
                           round_commit_enabled, round_costs, stage_spec)
from .fast.state import MAX_PANEL
from .fast.step import wide_rows_exact
from .layout import cache_footprint
from .model import load_target
from .vision import VisionTower


def make_weights_resident(model, drafter):
    """Read the target's and the drafter's weights into memory now, at load.

    The loader maps the pack lazily, so without this the first forward after the server reports
    ready (its own warm-up request) read some 15 GB of weights, and the memory guard counted that
    load as the request's growth (``prefill_grew_past_price`` on every start). The vision tower
    stays lazy: a text-only session never reads it."""
    mx.eval(model.parameters(), drafter.parameters() if drafter is not None else {})


def widest_panel(model):
    """The widest verify panel the fast step will be asked for: a copy round's block (the copy
    lane's cap, 24 tokens by default) plus its anchor, up to ``MAX_PANEL``, where wide panels are
    exact on this pack; else warm-up's widest."""
    if not wide_rows_exact(model, stage_spec()):
        return engine_site.MAX_WARM_WIDTH
    from mtplx import context_copy

    return max(engine_site.MAX_WARM_WIDTH, min(MAX_PANEL, context_copy.context_copy_block_k() + 1))


def engine_pool_price(measured, widest):
    """The engine pool a request is priced at. One pool serves every program's temporaries and
    grows to the widest program served: warm-up's widest is ``MAX_WARM_WIDTH`` rows, and a copy
    round records up to ``widest`` (measured on the native pack: 81 MB after warm-up, 231 MB
    once a 25-row copy round had recorded), so the pool is priced at the widest panel, scaled by
    rows from the measured one."""
    measured = int(measured or 0)
    return measured * max(int(widest), engine_site.MAX_WARM_WIDTH) // engine_site.MAX_WARM_WIDTH


class InklingRuntime(MTPLXRuntime):
    backend_id = "inkling2_loop"

    def __init__(self, path: Path, *, mtp: bool, contract=None):
        model, vision, config = load_target(path)
        # One proof cache for this pack: the vocabulary panel, the sorted expert read and
        # expert fusion all record through the cache the model's resolver owns.
        self.kernel_proofs = model.proof_pack(str(mx.default_device()))[0]
        expert_fusion.bind(model, self.kernel_proofs)
        tokenizer = _load_tokenizer_resilient(path, config)
        tokenizer.inkling_protocol = True
        super().__init__(model, tokenizer, path, mtp, contract or MTPContract())
        self.vision_tower = VisionTower(config["vision_config"], vision)
        self.drafter = Drafter(path, config["mtp_config"], model) if mtp else None
        self.trunk_entries = len(model.visit_order) * 2
        self.target_footprint = cache_footprint(config["text_config"])
        self.draft_footprint = (
            cache_footprint(config["mtp_config"], draft=True) if mtp else None
        )
        make_weights_resident(model, self.drafter)
        # Verify panels, commits and one-token forwards go through the fast step first.
        self.fast = FastDecoder(self)
        # Before the server reports ready: build the round's kernels, then record and check
        # the step's program for widths 1 to 8 on a synthetic state (no request touched), so
        # the first request neither compiles nor records.
        self.engine_warm, self.engine_pool_bytes = {"skipped": "fast path off"}, 0
        self.round_warm = {"skipped": "fast path off"}
        # The depth controller's round costs on this Mac and pack (None: RC6's table).
        self.round_cost_ms, self.round_cost_detail = None, "fast path off"
        if fast_enabled() and mx.default_device() == mx.gpu and mx.metal.is_available():
            # The round's kernels serve speculative rounds only: warm them where they run.
            if self.drafter is not None:
                fast_round.warm()
            self.engine_warm = engine_site.warm(self)
            # The DSpark round as one program for 1 to 7 proposals (lane L7), recorded and checked
            # before ready, so no request records; the pool below is read after it. The variant the
            # served loop runs: the committing round unless MTPLX_INKLING_ROUND_COMMIT=0.
            self.round_warm = (round_program.warm(self, commit=round_commit_enabled())
                               if self.drafter is not None else {"skipped": "no drafter"})
            # The executor's pool after warm-up (one pool serves every width's temporaries).
            executor = engine_site.health().get("executor") or {}
            self.engine_pool_bytes = int(executor.get("pool_bytes") or 0)
            self.engine_pool_bound = engine_pool_price(self.engine_pool_bytes, widest_panel(model))
            if self.drafter is not None:
                try:
                    self.round_cost_ms, self.round_cost_detail = round_costs(self)
                except Exception as exc:  # noqa: BLE001 - a timing pass must not stop the load
                    # Said, not hidden: /health and every auto-depth request carry the reason,
                    # and the controller prices rounds with RC6's table meanwhile.
                    self.round_cost_ms = None
                    self.round_cost_detail = f"round cost timing failed: {type(exc).__name__}: {exc}"
                    print(f"[mtplx] inkling: {self.round_cost_detail}", flush=True)
        # The table against the rounds it prices, whatever made it (DepthTableCheck).
        self.depth_table_check = DepthTableCheck(self.round_cost_ms or fast_draft.RC6_COST_MS_2K)

    def resident_kv_bytes_per_token(self):
        return self.target_footprint.full_bytes_per_token

    def prefill_kv_bytes_per_token(self):
        return self.resident_kv_bytes_per_token()

    def attention_kv_copy_bytes_per_token(self):
        # Stock attention appends a sink slot to chronological K and V.
        # One global visit's two BF16 arrays grow with the attended history.
        cfg = self.model.config
        if not self.target_footprint.full_bytes_per_token:
            return 0
        return 4 * cfg["num_key_value_heads"] * cfg["head_dim"]

    def prefill_cache_layout(self, context_tokens):
        # Global KV, local rings and convolution state stay on the same
        # cache across prefill and decode. Admission must use the owned windows.
        return "contiguous_dense_decode"

    def window_cache_bytes(self, width=None):
        """The fixed part of what decode keeps besides per-token KV: the sliding rings, the
        global banks' rounding to whole capacity steps, and the native engine's working pool."""
        rows = max(1, width or 1)
        return sum(
            part.bounded_bytes(part.window + rows - 1, rows)
            for part in (self.target_footprint, self.draft_footprint)
            if part is not None
        ) + self.fast_fixed_bytes()

    def fast_fixed_bytes(self):
        from .fast.state import BANK_STEP

        cfg = self.model.config
        row = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * self.model.embed_tokens.weight.dtype.size
        globals_ = sum(1 for i in self.model.visit_order if not self.model.layers[i].sliding)
        pool = getattr(self, "engine_pool_bound", None) or getattr(self, "engine_pool_bytes", 0)
        return globals_ * BANK_STEP * row + pool

    #: Rows a chunked prompt forward reads when nobody names a width.
    PROMPT_CHUNK_ROWS = 512

    def prompt_chunk_rows(self, prompt_tokens=None):
        """512 rows a prompt forward where the prompt read is chunked, unless a request
        carries its own width (a caller's chunk, ``--prefill-chunk-tokens``, the width the
        memory admission narrowed to) or the user set ``MTPLX_PREFILL_CHUNK_SIZE``
        themselves. A profile that forwards the prompt whole keeps the shared answer."""
        from mtplx.generation import (
            _prefill_chunk_env_is_pinned,
            _prefill_chunk_size,
            _sustained_prefill_enabled,
            current_prefill_chunk_override,
        )

        if (
            current_prefill_chunk_override() is not None
            or _prefill_chunk_env_is_pinned()
            or not _sustained_prefill_enabled()
        ):
            return _prefill_chunk_size(prompt_tokens)
        return self.PROMPT_CHUNK_ROWS

    #: The prefill's working memory beyond the request's own KV, measured through `mtplx serve` on
    #: Bare Speed at 512-row chunks (each request's growth less its KV, from the request log):
    #: 1.76 GiB at a 4K prompt, 1.53 at 16K, 1.47 at 32K, 1.83 at a fresh 48K. Rounded up to 2 GiB.
    #: It replaces the flat 3 GiB runtime reserve in this family's memory plan, and it is what the
    #: admission charges a forward (the generic share charged 0.75 GiB at 512 rows, 0.8 GB short).
    PREFILL_WORKING_BYTES = 2 * 1024**3

    def prefill_working_bytes(self, rows=None):
        """The working memory of one prompt forward of ``rows`` rows (default: the chunk width).
        Only the 512-row width is measured: a narrower forward is charged the same, a wider one
        in proportion to its rows."""
        rows = self.PROMPT_CHUNK_ROWS if rows is None else max(1, int(rows))
        return self.PREFILL_WORKING_BYTES * max(rows, self.PROMPT_CHUNK_ROWS) // self.PROMPT_CHUNK_ROWS

    def prefill_window_bytes(self, prompt_tokens, new_tokens, width=None):
        if not new_tokens:
            return 0
        chunk = min(new_tokens, width or self.prompt_chunk_rows(prompt_tokens))
        prefill = sum(
            part.bounded_bytes(prompt_tokens, chunk)
            for part in (self.target_footprint, self.draft_footprint)
            if part is not None
        )
        # From the first verify the fast path holds every ring at the full window and every
        # global bank in whole capacity steps, whatever the prompt's length (prepare), plus
        # the engine's pool: a short prompt's request holds that, not its prompt-sized windows.
        return max(prefill, self.window_cache_bytes())

    def embed_tokens(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def make_cache(self):
        target = self.model.make_cache()
        return target + (self.drafter.make_cache() if self.drafter else [])

    def can_restore_session_boundary(self, entry, snapshot):
        count = self.trunk_entries + (
            len(self.drafter.blocks) + 1 if self.drafter else 0
        )
        if (
            snapshot is None
            or len(snapshot.states) != count
            or len(snapshot.meta_states) != count
        ):
            return False
        if any(snapshot.states[i] is None for i in range(1, self.trunk_entries, 2)):
            return False
        if self.drafter and snapshot.states[-1] is None:
            return False
        # Older snapshots omitted local KV at boundaries taken before the window filled.
        # Trimming can recover those rows only while the stored conversation still has them.
        windows = [
            (2 * i, self.model.config["sliding_window_size"])
            for i, layer in enumerate(self.model.visit_order)
            if self.model.layers[layer].sliding
        ]
        if self.drafter:
            windows.extend(
                (self.trunk_entries + i, b.window)
                for i, b in enumerate(self.drafter.blocks)
                if b.window
            )
        for i, limit in windows:
            if snapshot.states[i] is None:
                live = entry.cache_ref
                if max(entry.prefix_len, live[i].offset if live else 0) > limit:
                    return False
            elif snapshot.meta_states[i] is None:
                return False
        return True

    def repage_target_prefill_cache(self, cache):
        return False

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden=False,
        hidden_variant=None,
        emit_logits=True,
        logits_keep=None,
        input_embeddings=None,
    ):
        cache = self.make_cache() if cache is None else cache
        self._count(
            "forward_ar_hidden_calls" if return_hidden else "forward_ar_plain_calls"
        )
        served = self.fast.forward_one(
            input_ids, cache, return_hidden, emit_logits, input_embeddings
        )
        if served is not DECLINED:
            return served
        captures = (
            self.model.capture_ids if return_hidden or self.drafter is not None else ()
        )
        h, features = self.model.forward(
            input_ids, cache[: self.trunk_entries], input_embeddings, captures
        )
        if self.drafter:
            self.drafter.observe(features, cache[self.trunk_entries :])
        logits = (
            self.model.head(h if logits_keep is None else h[:, -max(1, logits_keep) :])
            if emit_logits
            else None
        )
        if return_hidden:
            return logits, mx.concatenate(features, axis=-1)
        return logits

    def verify(self, block, cache):
        served = self.fast.verify(block, cache)
        if served is not None:
            return served
        tapes = [{} for _ in self.model.visit_order]
        ids = block if isinstance(block, mx.array) else mx.array(block)
        h, features = self.model.forward(
            ids[None],
            cache[: self.trunk_entries],
            capture_ids=self.model.capture_ids,
            tapes=tapes,
        )
        return self.model.head(h)[0], features, tapes

    def commit(self, cache, features, tapes, kept, total):
        if isinstance(tapes, FastPending):
            return self.fast.commit(cache, features, tapes, kept, total)
        if not 1 <= kept <= total:
            raise ValueError(
                "Inkling commits the anchor and a prefix of the verified block"
            )
        if kept < total:
            for visit, tape in enumerate(tapes):
                kv, conv = cache[2 * visit : 2 * visit + 2]
                if isinstance(kv, RotatingKVCache):
                    # Multi-row verify preserved the full pre-anchor window in chronological order.
                    # Physically discard the rejected tail before ring-buffer indexing resumes.
                    end = kv._idx - (total - kept)
                    start = max(0, end - kv.max_size)
                    kv.keys, kv.values = (
                        kv.keys[:, :, start:end, :],
                        kv.values[:, :, start:end, :],
                    )
                    kv.offset -= total - kept
                    kv._idx = end - start
                else:
                    kv.trim(total - kept)
                for slot, history in tape.items():
                    prefix = history.shape[1] - total
                    conv[slot] = history[:, kept : kept + prefix]
        self.drafter.observe(
            [h[:, :kept] for h in features], cache[self.trunk_entries :]
        )
