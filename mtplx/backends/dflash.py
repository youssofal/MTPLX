"""Generic dflash speculative-decoding backend for MTPLX.

Drives a :class:`~mtplx.models.dflash.DFlashDrafter` (a config-driven block-
diffusion drafter, validated bit-exact against bstnxbt/dflash-mlx) against any
MTPLX target that exposes the residual-stream tap hook (``model._tap_layers`` /
``model._taps``). One round = propose a block, verify it in a single target
forward, accept a prefix by **target-prefix** semantics, and roll the target KV
cache back to the accepted length.

Adding a future dflash drafter is code-free: drop a pair bundle (``target/`` +
``drafter/`` + ``dflash_pair.json``) and this backend loads and runs it — the
tap layers, block size, and mask token all come from the drafter's config.

The decode loop mirrors the acceptance-validated reference flow:
  * ``staged_first`` = the last *forwarded* token; its tap + all prior committed
    taps form the drafter's context (accumulated across rounds, like the
    reference ``ContextOnlyDraftKVCache``).
  * verify forwards the k-1 draft tokens once; ``target[j]`` is the argmax of the
    logit *before* draft position j (the first uses the carried ``prev_logit``).
  * accept the leading run of matches; the first miss becomes the correction,
    an all-match becomes the bonus; both are the next ``staged_first``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mtplx.models.dflash import DFlashDrafter, load_dflash


@dataclass(frozen=True)
class DFlashRuntimeConfig:
    target_model_path: str
    drafter_model_path: str
    block_size: int
    # Target-model layer outputs to tap after applying the checkpoint-specific
    # offset once during bundle construction.
    capture_layers: list[int]
    mask_token_id: int
    embed_scale: float = 1.0
    max_context: int = 1088  # sink(64)+window(1024); cap accumulated ctx taps

    @property
    def assistant_model_path(self) -> str:
        return self.drafter_model_path

    @property
    def draft_block_size(self) -> int:
        return self.block_size

    target_distribution_mode: str = "target_prefix"


class DFlashRuntime:
    """Self-contained dflash runtime. ``backend_id`` marks it for the
    ``generate_mtpk`` dispatch branch (mirrors ``gemma4_assistant``)."""

    backend_id = "dflash"
    mtp_enabled = True

    def __init__(self, target, tokenizer, drafter: DFlashDrafter,
                 config: DFlashRuntimeConfig):
        self.target = target
        self.model = target
        self.args = target.args
        self.tokenizer = tokenizer
        self.drafter = drafter
        self.config = config
        if getattr(target.args, "model_type", None) == "nemotron_h":
            from mtplx.nemotron_lightning_dflash import (
                install_nemotron_lightning_capture,
            )

            install_nemotron_lightning_capture(target, config.capture_layers)
            self._tm = target.backbone
            self._target_embedding = target.backbone.embeddings
            self._target_lm_head = target.lm_head
            self._forward_capture_impl = target.dflash_forward_capture
            self._round_impl = self._round_hybrid
        else:
            self._tm = target.model
            self._tm._tap_layers = set(config.capture_layers)
            self._target_embedding = self._tm.embed_tokens
            self._target_lm_head = None
            self._forward_capture_impl = self._forward_capture_tapped
            self._round_impl = self._round_kv
        self.model_path = config.target_model_path
        self.path = config.target_model_path

    # ---- target token-embedding / lm-head as callables for the drafter ----
    def _tok_embd(self, ids: mx.array) -> mx.array:
        return self._target_embedding(ids)

    def _lm_head(self, x: mx.array) -> mx.array:
        if self._target_lm_head is not None:
            return self._target_lm_head(x)
        args = self.target.args
        logits = (self._tm.embed_tokens.as_linear(x)
                  if args.tie_word_embeddings else self.target.lm_head(x))
        logits = logits * args.logit_scale
        cap = args.final_logit_softcapping
        return mx.tanh(logits / cap) * cap if cap else logits

    def _forward_capture_tapped(self, ids: mx.array, cache) -> tuple[mx.array, dict]:
        """Forward the target on ids [1,T] with tap capture on. Returns
        (logits [1,T,vocab], taps {L: [T, hidden]})."""
        logits = self.target(ids, cache=cache)
        taps = {L: self._tm._taps[L][0] for L in self.config.capture_layers}
        return logits, taps

    def _forward_capture(self, ids: mx.array, cache) -> tuple[mx.array, dict]:
        return self._forward_capture_impl(ids, cache)

    # ---- drop grown-buffer zero-garbage so temporal_order stays clean ------
    @staticmethod
    def _normalize(cache) -> None:
        """A single-token forward (`_update_in_place`) grows the KV buffer with
        zeros past `offset`; the next batched forward's `_temporal_order` would
        fold that garbage into committed context (the exactness bug). Slice each
        non-rotated cache down to its valid offset before every batched forward."""
        for c in cache:
            if c.keys is None:
                continue
            max_size = getattr(c, "max_size", None)
            if max_size is not None and c.offset >= max_size:
                continue  # rotated sliding cache: leave the rotating buffer alone
            if c.keys.shape[2] > c.offset:
                c.keys = c.keys[..., : c.offset, :]
                c.values = c.values[..., : c.offset, :]
                if hasattr(c, "_idx"):
                    c._idx = c.offset

    # ---- roll every layer cache back to `keep_len` committed positions -----
    @staticmethod
    def _rollback(cache, keep_len: int) -> None:
        """Drop speculative KV beyond `keep_len`. For a non-rotated cache we
        SLICE the buffer (not just move `offset`) so the next batched forward's
        temporal-order/trim never re-mixes rejected drafts into context — the
        cause of the exactness divergence. A genuinely rotated sliding cache
        falls back to `trim` bookkeeping."""
        for c in cache:
            if c.keys is None or c.offset <= keep_len:
                continue
            max_size = getattr(c, "max_size", None)
            if max_size is not None and c.offset >= max_size:   # rotated sliding
                c.trim(c.offset - keep_len)
            else:                                               # global / unrotated
                c.keys = c.keys[..., :keep_len, :]
                c.values = c.values[..., :keep_len, :]
                c.offset = keep_len
                if hasattr(c, "_idx"):
                    c._idx = keep_len

    # ---- one propose+verify+accept round (cached ctx, one target forward) --
    @staticmethod
    def _accepted_prefix(draft_ids: list[int], vlog: mx.array) -> tuple[list[int], int]:
        targ = [int(x) for x in mx.argmax(vlog, axis=-1).tolist()]
        accepted: list[int] = []
        nxt: int | None = None
        for j, draft in enumerate(draft_ids):
            if draft == targ[j]:
                accepted.append(draft)
            else:
                nxt = targ[j]
                break
        if nxt is None:
            nxt = targ[len(draft_ids)]
        return accepted, nxt

    def _round_kv(self, ctx_cache: list, ctx_len: int, primary: int, cache):
        cfg = self.config
        drafts = self.drafter.propose_block_cached(
            ctx_cache, ctx_len, self._tok_embd, self._lm_head,
            primary_token_id=primary, mask_token_id=cfg.mask_token_id,
            block_size=cfg.block_size, embed_scale=cfg.embed_scale,
        )
        draft_ids = [int(x) for x in drafts.tolist()]        # k-1 tokens
        self._normalize(cache)                               # clean grown-buffer garbage
        base = cache[0].offset

        # ONE target forward over [primary, *drafts] (folds the old separate
        # nxt decode): vlog[j] = logits after position j predicts token j+1.
        vlogits, vtaps = self._forward_capture(
            mx.array([primary] + draft_ids, dtype=mx.int32)[None], cache)  # appends k
        vlog = vlogits[0]                                     # [k, vocab]

        # target-prefix accept walk. Compute ALL k target argmaxes in ONE op +
        # ONE host sync (not k `.item()` calls — that was k GPU stalls/round).
        accepted, nxt = self._accepted_prefix(draft_ids, vlog)
        A = len(accepted)

        # commit [primary + A accepted]; the correction/bonus `nxt` becomes the
        # next round's primary (seated by that round's verify position 0).
        self._rollback(cache, base + 1 + A)
        new_taps = [vtaps[L][: 1 + A] for L in cfg.capture_layers]  # primary+accepted taps
        self.drafter.extend_context(ctx_cache, new_taps, ctx_len)
        committed = [primary] + accepted
        return ctx_len + 1 + A, nxt, committed

    def _round_hybrid(self, ctx_cache: list, ctx_len: int, primary: int, cache):
        from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache

        cfg = self.config
        drafts = self.drafter.propose_block_cached(
            ctx_cache,
            ctx_len,
            self._tok_embd,
            self._lm_head,
            primary_token_id=primary,
            mask_token_id=cfg.mask_token_id,
            block_size=cfg.block_size,
            embed_scale=cfg.embed_scale,
        )
        draft_ids = [int(x) for x in drafts.tolist()]
        verify_ids = [primary] + draft_ids
        before_verify = snapshot_untrimmable_cache(cache)
        vlogits, verify_taps = self._forward_capture(
            mx.array(verify_ids, dtype=mx.int32)[None], cache
        )
        accepted, nxt = self._accepted_prefix(draft_ids, vlogits[0])
        committed = [primary] + accepted
        if len(accepted) == len(draft_ids):
            committed_taps = verify_taps
        else:
            rollback_after_verify(cache, before_verify, verified_tokens=len(verify_ids))
            _repair_logits, committed_taps = self._forward_capture(
                mx.array(committed, dtype=mx.int32)[None], cache
            )
        new_taps = [committed_taps[layer] for layer in cfg.capture_layers]
        self.drafter.extend_context(ctx_cache, new_taps, ctx_len)
        return ctx_len + len(committed), nxt, committed

    # ---- greedy generate --------------------------------------------------
    def generate(self, prompt, max_tokens: int = 128, *,
                 stop_token_ids: set | None = None, token_callback=None) -> dict:
        """Greedy speculative generate. `prompt` is a string or a list of token
        ids. `token_callback([id])` is called per committed token (streaming);
        generation stops at `max_tokens` or when a stop id is committed. Output
        is token-exact vs greedy AR (up to fp near-tie non-determinism)."""
        prompt_ids = (self.tokenizer.encode(prompt) if isinstance(prompt, str)
                      else list(prompt))
        ids = mx.array(prompt_ids)[None]
        cache = self.target.make_cache()
        logits, taps = self._forward_capture(ids, cache)
        P = int(ids.shape[1])
        ctx_cache = self.drafter.init_context_cache()
        self.drafter.extend_context(
            ctx_cache, [taps[L] for L in self.config.capture_layers], 0)  # prompt @0..P-1
        ctx_len = P
        primary = int(mx.argmax(logits[0, -1]).item())           # first token to commit
        stops = stop_token_ids or set()

        out: list[int] = []
        rounds = 0
        accepts = 0
        drafted = 0
        stopped = False
        while len(out) < max_tokens and not stopped:
            ctx_len, primary, committed = self._round_impl(
                ctx_cache, ctx_len, primary, cache
            )
            rounds += 1
            drafted += self.config.block_size - 1
            accepts += len(committed) - 1                        # accepted drafts only
            for tid in committed:
                out.append(tid)
                if token_callback is not None:
                    token_callback([tid])
                if tid in stops or len(out) >= max_tokens:
                    stopped = True
                    break
        return {
            "text": self.tokenizer.decode(out),
            "tokens": out,
            "rounds": rounds,
            "accepted": accepts,
            "drafted": drafted,
            "rejected": drafted - accepts,
            "mean_accept": accepts / max(1, rounds),      # drafts accepted / round
            "tokens_per_target_step": (accepts + rounds) / max(1, rounds),
        }


def generate_dflash(runtime: DFlashRuntime, prompt_ids, *, max_tokens: int,
                    sampler=None, speculative_depth: int | None = None,
                    stop_token_ids=None, token_callback=None, seed: int = 0,
                    **_ignored):
    """`generate_mtpk` dispatch entry for the dflash backend (mirrors
    `generate_gemma4_assistant`). Greedy target-prefix decode — token-exact vs
    greedy AR. `sampler`/session/trace kwargs are accepted for signature
    compatibility; sampling (temperature>0) is not yet wired (the argmax drafter
    has no q-distribution for a p/q accept), so decode is greedy."""
    import time as _time

    from mtplx.generation import GenerationOutput, GenerationStats

    # The released assistant was trained for the block geometry encoded in its
    # pair manifest. Keep that construction-time invariant: the server's
    # generic native-MTP depth argument does not resize an installed DFlash
    # lane at request time.
    del speculative_depth
    stops = {int(t) for t in (stop_token_ids or ())}
    t0 = _time.perf_counter()
    result = runtime.generate(list(prompt_ids), max_tokens=int(max_tokens),
                              stop_token_ids=stops, token_callback=token_callback)
    elapsed = _time.perf_counter() - t0
    toks = result["tokens"]
    rate = len(toks) / max(1e-9, elapsed)
    stats = GenerationStats(
        mode="mtpk", generated_tokens=len(toks), elapsed_s=elapsed, tok_s=rate,
        decode_elapsed_s=elapsed, decode_tok_s=rate, end_to_end_tok_s=rate,
        runtime_mtp_enabled=True, mtp_forward_calls=result["rounds"],
        accepted_drafts=result["accepted"], rejected_drafts=result["rejected"],
        drafted_tokens=result["drafted"], verify_calls=result["rounds"],
        speculative_depth=runtime.config.block_size - 1,
        requested_speculative_depth=runtime.config.block_size - 1,
    )
    finish = "stop" if (toks and toks[-1] in stops) else "length"
    return GenerationOutput(tokens=toks, text=result["text"], stats=stats,
                            final_state=None, finish_reason=finish)


def load_dflash_runtime(bundle_root: str) -> DFlashRuntime:
    """Load a dflash pair bundle (target/ + drafter/ + dflash_pair.json) into a
    ready `DFlashRuntime`. Installs the target's model shim if needed (the
    drafter borrows the target's tok_embd/lm_head, so the target must load)."""
    import json
    import os

    from mlx_lm import load as _load

    from mtplx.dflash_pair import (
        dflash_pair_block_size,
        resolve_dflash_pair_paths,
    )

    pair = resolve_dflash_pair_paths(bundle_root)
    if pair is None:
        raise ValueError(f"{bundle_root} is not a dflash pair bundle")
    # install the target arch shim before loading (e.g. Muse-Glimmer)
    try:
        tcfg = json.load(open(os.path.join(pair["target_model"], "config.json")))
        from mtplx.muse_glimmer_patch import (
            install_muse_glimmer_model_shim,
            is_muse_glimmer_config,
        )
        if is_muse_glimmer_config(tcfg):
            install_muse_glimmer_model_shim()
    except (OSError, ValueError, ImportError):
        pass
    target, tokenizer = _load(pair["target_model"])
    drafter, dcfg = load_dflash(pair["drafter_model"])
    block = dflash_pair_block_size(pair["metadata"], dcfg.block_size)
    cfg = DFlashRuntimeConfig(
        target_model_path=pair["target_model"],
        drafter_model_path=pair["drafter_model"],
        block_size=block,
        capture_layers=[t + dcfg.target_layer_offset for t in dcfg.target_layers],
        mask_token_id=dcfg.mask_token_id if dcfg.mask_token_id is not None else 201818,
    )
    return DFlashRuntime(target, tokenizer, drafter, cfg)
