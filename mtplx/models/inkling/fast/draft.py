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
    picks, laws, stats, confidence = [], [], [], []
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
        token, lse = rnd.sample_rows(law, st, positions=(start + i).reshape(1), seed=seed_words,
                                     purpose=rnd.PURPOSE_DRAFT, greedy=greedy)
        prev = token
        picks.append(token)
        laws.append(law)
        stats.append(st)
        if not greedy:
            confidence.append(mx.take_along_axis(law, token[:, None], axis=1)[:, 0] - lse[:, 1])
    ids = mx.concatenate(picks)
    law = rnd.DraftLaw(mx.concatenate(laws), mx.concatenate(stats), point_mass=greedy,
                       log_q=mx.concatenate(confidence) if confidence else None)
    return ids, law


# ------------------------------------------------------------------ depth by yield

#: Protocol channel tokens of Inkling 2 Mini (tokenizer added tokens).
CHANNELS = {200008: "reasoning", 200004: "text", 200049: "tool", 200057: "tool", 200024: "tool"}


@dataclass
class DepthByYield:
    """Per-round proposal count from the yield recent rounds actually had, per text channel.

    ``cost_ms[k]`` is the round cost with k proposals (verify width k + 1; k = 0 is a plain
    step that still keeps the draft history). Measured acceptance of DSpark proposals is
    nearly flat by position (study: 0.72, 0.68, 0.77, 0.70, 0.76, 0.80, 0.62), so each
    channel keeps one decayed acceptance rate ``a`` over every proposal it verified, and a
    round with k proposals is expected to yield 1 + a + ... + a^k tokens. The choice
    maximises expected tokens per ms; ties go to the shallower depth (fewer wasted rows).
    The state starts from ``prior`` worth ``prior_weight`` proposals. Every input is a
    function of the token history, so a seeded request makes the same choices every time.
    """

    cost_ms: tuple
    max_depth: int = 7
    min_depth: int = 1
    prior: float = 0.70
    prior_weight: float = 6.0
    decay: float = 0.85
    channel: str = "text"
    _hits: dict = field(default_factory=dict)
    _tries: dict = field(default_factory=dict)
    choices: list = field(default_factory=list)

    def rate(self, channel=None):
        channel = self.channel if channel is None else channel
        hits = self._hits.get(channel, self.prior * self.prior_weight)
        tries = self._tries.get(channel, self.prior_weight)
        return hits / tries if tries > 0 else self.prior

    def expected_tokens(self, k, rate=None):
        a = self.rate() if rate is None else rate
        total, chain = 1.0, 1.0
        for _ in range(k):
            chain *= a
            total += chain
        return total

    def choose(self, remaining=None):
        """The proposal count for the next round."""
        a = self.rate()
        limit = min(self.max_depth, len(self.cost_ms) - 1)
        if remaining is not None:
            limit = min(limit, max(0, int(remaining) - 1))
        low = min(self.min_depth, limit)
        best, best_rate = low, -1.0
        for k in range(low, limit + 1):
            value = self.expected_tokens(k, a) / self.cost_ms[k]
            if value > best_rate * (1.0 + 1e-6):
                best, best_rate = k, value
        self.choices.append(best)
        return best

    def observe(self, offered, accepted, tokens=()):
        """Record a round (``offered`` proposals, the first ``accepted`` kept), then follow
        the channel of the emitted tokens for the next choice."""
        channel = self.channel
        hits = self._hits.get(channel, self.prior * self.prior_weight) * self.decay
        tries = self._tries.get(channel, self.prior_weight) * self.decay
        # proposals actually tested: every accepted one plus the first rejected one
        tested = min(offered, accepted + 1)
        self._hits[channel] = hits + accepted
        self._tries[channel] = tries + tested
        for token in tokens:
            self.channel = CHANNELS.get(int(token), self.channel)
