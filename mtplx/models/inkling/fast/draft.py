"""DSpark proposals drawn on the GPU from the drafter's own law, and the per-round depth.

``propose`` runs RC6's drafter arithmetic unchanged (the three backbone blocks at the
trained width of 8 slots including the anchor, the vocabulary head over the proposal
rows, the low-rank Markov walk with the folded table) and replaces the proposal choice:

* temperature 0 or the seed-match rule: the argmax of each Markov row, as RC6 does
  (lowest index on ties, the same token ``mx.argmax`` picks);
* the ratio rule at temperature T > 0: a draw from ``q = softmax(cap(scores) / T)``, the
  drafter's soft-capped law at the request temperature, by the same exponential race as
  the target side (``fast.round``), keyed by (seed, absolute position, draft purpose).

It returns the proposal ids as a device array (no host read) and the exact law rows the
draws used (``round.DraftLaw``), which ``round.decide`` needs for min(1, p/q) and the
residual. The drafter's caches are only read: observing accepted rows stays the commit's
job (``lanes/SIGNATURES.md``).

``DepthByYield`` chooses how many proposals to verify each round from the yield the
recent rounds actually had, per text channel, against a table of round cost by verify
width. It reads nothing from the device: its inputs are the accepted counts the host
already holds after each packet. Because every input is a function of the token history
and the seed, the same request with the same seed makes the same choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from . import round as rnd


def _anchor_ids(anchor):
    if isinstance(anchor, mx.array):
        return anchor.astype(mx.int32).reshape(1, 1)
    return mx.array([[int(anchor)]], mx.int32)


def propose(drafter, target, anchor, cache, count, *, position, seed, temperature=1.0,
            sample=True, cap=None):
    """``count`` proposals (1 to 7) after ``anchor``; returns (ids (count,) int32, DraftLaw).

    ``position`` is the absolute position of the first proposal (the token after the
    anchor), an int or a uint32 array. ``sample`` False keeps RC6's argmax proposals (the
    seed-match rule). ``cap`` defaults to the target's final soft cap.
    """
    if not 1 <= count < drafter.width:
        raise ValueError("DSpark proposal count exceeds its trained block")
    greedy = (not sample) or (not temperature or temperature <= 0)
    if cap is None:
        cap = target.config.get("final_logit_softcapping") or 0.0
    embed = target.embed_tokens(_anchor_ids(anchor))
    dtype = embed.dtype
    masks = mx.broadcast_to(drafter.mask, (1, drafter.width - 1, embed.shape[-1]))
    h, residual = mx.concatenate([embed, masks], axis=1), None
    for i, block in enumerate(drafter.blocks):
        h, residual = block(h, residual, cache[i])
    h = mx.fast.rms_norm(
        h.astype(mx.float32) + residual.astype(mx.float32),
        drafter.output_norm,
        drafter.cfg["rms_norm_eps"],
    ).astype(dtype)[:, 1 : count + 1]
    from ..model import linear

    base = linear(h, target.unembed)[..., : drafter.vocab]
    rank = drafter.cfg["markov_rank"]
    weights = drafter.markov["rnn_proj.weight"].astype(mx.float32)
    inputs = mx.concatenate([h, mx.broadcast_to(cache[-1][0], h.shape)], axis=-1)
    conditioned = linear(inputs.astype(mx.float32), weights[:, 2 * rank :]) + drafter.markov[
        "rnn_proj.bias"
    ].astype(mx.float32)
    folded = drafter.fold(target.unembed)
    start = position if isinstance(position, mx.array) else mx.array(int(position), mx.uint32)
    start = start.astype(mx.uint32).reshape(())
    seed_words = seed if isinstance(seed, mx.array) else rnd._seed_words(seed)
    state = mx.zeros((1, rank), dtype)
    prev = _anchor_ids(anchor).reshape(1)
    picks, laws, stats = [], [], []
    for i in range(count):
        step_in = mx.concatenate([state, drafter.markov["markov_w1.weight"][prev]], axis=-1)
        update = (
            linear(step_in.astype(mx.float32), weights[:, : 2 * rank]) + conditioned[:, i]
        ).astype(dtype)
        gate, candidate, output = mx.split(update, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(dtype)
        state = gate * state + (1 - gate) * mx.tanh(candidate.astype(mx.float32)).astype(dtype)
        signal = mx.tanh(output.astype(mx.float32)).astype(dtype)
        bias = mx.addmm(folded[:, -1], signal, folded[:, :-1].T)
        scores = base[:, i] + bias
        if greedy:
            law, st = rnd.law_rows(scores, temperature=None, cap=0.0, write=True)
        else:
            law, st = rnd.law_rows(scores, temperature=temperature, cap=cap)
        token, _ = rnd.sample_rows(law, st, positions=(start + i).reshape(1), seed=seed_words,
                                   purpose=rnd.PURPOSE_DRAFT, greedy=greedy)
        prev = token
        picks.append(token)
        laws.append(law)
        stats.append(st)
    ids = mx.concatenate(picks)
    law = rnd.DraftLaw(mx.concatenate(laws), mx.concatenate(stats), point_mass=greedy)
    return ids, law


# ------------------------------------------------------------------ depth by yield

#: Protocol channel tokens of Inkling 2 Mini (tokenizer added tokens).
CHANNELS = {200008: "reasoning", 200004: "text", 200049: "tool", 200057: "tool", 200024: "tool"}


@dataclass
class DepthByYield:
    """Per-round proposal count from measured yield, per text channel.

    ``cost_ms[k]`` is the round cost with k proposals (verify width k + 1, k = 0 is a
    plain step that still keeps the draft history). The expected tokens of a round with
    k proposals are 1 + sum_{i<=k} prod_{j<=i} a_j, where a_j is the conditional
    acceptance at position j, kept as a decayed count per channel. The choice maximises
    expected tokens per ms. Counts start from a prior worth ``prior_rounds`` rounds.
    """

    cost_ms: tuple
    max_depth: int = 7
    prior: tuple = (0.70, 0.62, 0.58, 0.55, 0.52, 0.50, 0.48)
    prior_rounds: float = 4.0
    decay: float = 0.92
    min_depth: int = 1
    channel: str = "text"
    _hits: dict = field(default_factory=dict)
    _tries: dict = field(default_factory=dict)
    choices: list = field(default_factory=list)

    def _rates(self, channel):
        hits = self._hits.setdefault(channel, [p * self.prior_rounds for p in self.prior])
        tries = self._tries.setdefault(channel, [self.prior_rounds] * len(self.prior))
        return [h / t if t > 0 else 0.0 for h, t in zip(hits, tries)]

    def expected_tokens(self, k, rates=None):
        rates = self._rates(self.channel) if rates is None else rates
        total, chain = 1.0, 1.0
        for j in range(k):
            chain *= rates[j]
            total += chain
        return total

    def choose(self, remaining=None):
        """The proposal count for the next round (deterministic in the history)."""
        rates = self._rates(self.channel)
        limit = min(self.max_depth, len(self.cost_ms) - 1)
        if remaining is not None:
            limit = min(limit, max(0, int(remaining) - 1))
        best, best_rate = self.min_depth if limit >= self.min_depth else limit, -1.0
        for k in range(min(self.min_depth, limit), limit + 1):
            rate = self.expected_tokens(k, rates) / self.cost_ms[k]
            if rate > best_rate * (1.0 + 1e-9):
                best, best_rate = k, rate
        self.choices.append(best)
        return best

    def observe(self, offered, accepted, tokens=()):
        """Record a round: ``offered`` proposals, the first ``accepted`` kept; then follow
        the channel of the emitted tokens for the next choice."""
        hits = self._hits.setdefault(self.channel, [p * self.prior_rounds for p in self.prior])
        tries = self._tries.setdefault(self.channel, [self.prior_rounds] * len(self.prior))
        for j in range(len(hits)):
            hits[j] *= self.decay
            tries[j] *= self.decay
        # position j (0-based) was tried when every earlier proposal was accepted
        for j in range(min(offered, accepted + 1)):
            tries[j] += 1.0
            if j < accepted:
                hits[j] += 1.0
        for token in tokens:
            self.channel = CHANNELS.get(int(token), self.channel)
