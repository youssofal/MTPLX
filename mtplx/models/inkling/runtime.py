"""Target and speculative state share the conversation cache and live on the generation stream."""

from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime, _load_tokenizer_resilient

from . import expert_fusion
from .draft import Drafter
from .layout import cache_footprint
from .model import load_target
from .vision import VisionTower


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
        rows = max(1, width or 1)
        return sum(
            part.bounded_bytes(part.window + rows - 1, rows)
            for part in (self.target_footprint, self.draft_footprint)
            if part is not None
        )

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

    def prefill_window_bytes(self, prompt_tokens, new_tokens, width=None):
        if not new_tokens:
            return 0
        chunk = min(new_tokens, width or self.prompt_chunk_rows(prompt_tokens))
        return sum(
            part.bounded_bytes(prompt_tokens, chunk)
            for part in (self.target_footprint, self.draft_footprint)
            if part is not None
        )

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
