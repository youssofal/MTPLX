"""Adaptive-depth policies for native MTP experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class AdaptiveDepthPolicy:
    max_depth: int
    min_depth: int = 1
    start_depth: int = 1
    increase_after: int = 4
    decrease_after: int = 1

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must be <= max_depth")
        self.current_depth = min(max(self.start_depth, self.min_depth), self.max_depth)
        self._full_accept_streak = 0
        self._early_reject_streak = 0

    def observe(self, *, attempted_depth: int, accepted_depths: int) -> dict[str, int | str]:
        """Update depth from one cycle outcome and return a loggable decision."""
        attempted_depth = max(1, min(int(attempted_depth), self.max_depth))
        accepted_depths = max(0, min(int(accepted_depths), attempted_depth))
        previous_depth = self.current_depth
        action = "hold"

        if accepted_depths == attempted_depth:
            self._full_accept_streak += 1
            self._early_reject_streak = 0
            if self._full_accept_streak >= self.increase_after and self.current_depth < self.max_depth:
                self.current_depth += 1
                self._full_accept_streak = 0
                action = "increase"
        else:
            self._full_accept_streak = 0
            rejected_at = accepted_depths + 1
            if rejected_at <= max(1, previous_depth // 2):
                self._early_reject_streak += 1
            else:
                self._early_reject_streak = 0

            if self._early_reject_streak >= self.decrease_after and self.current_depth > self.min_depth:
                self.current_depth -= 1
                self._early_reject_streak = 0
                action = "decrease"

        return {
            "previous_depth": previous_depth,
            "attempted_depth": attempted_depth,
            "accepted_depths": accepted_depths,
            "next_depth": self.current_depth,
            "action": action,
        }


@dataclass
class ExpectedValueDepthPolicy:
    """Cost-aware D2/D3 controller.

    The policy starts each cycle at ``max_depth`` so the generation loop can
    draft sequentially, then it may stop after ``base_depth`` if the next
    depth does not clear a measured expected-value gate.
    """

    max_depth: int
    base_depth: int = 2
    min_depth: int = 1
    accept_priors: tuple[float, ...] = (0.92, 0.64, 0.32)
    ewma_alpha: float = 0.12
    draft_cost_s: float = 0.0048
    extra_verify_cost_s: float = 0.0060
    baseline_tok_s: float = 40.0
    safety_margin: float = 0.10
    margin_center: float = 1.0
    margin_scale: float = 2.0
    confidence_weight: float = 0.35
    min_extra_accept_probability: float = 0.18
    warmup_full_depth_cycles: int = 4
    exploration_interval: int = 32

    wants_draft_metrics: bool = True

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must be <= max_depth")
        if self.base_depth < self.min_depth:
            raise ValueError("base_depth must be >= min_depth")
        if self.base_depth > self.max_depth:
            raise ValueError("base_depth must be <= max_depth")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.draft_cost_s < 0.0 or self.extra_verify_cost_s < 0.0:
            raise ValueError("cost estimates must be non-negative")
        if self.baseline_tok_s <= 0.0:
            raise ValueError("baseline_tok_s must be > 0")
        if self.warmup_full_depth_cycles < 0:
            raise ValueError("warmup_full_depth_cycles must be >= 0")
        if self.exploration_interval < 0:
            raise ValueError("exploration_interval must be >= 0")
        self.current_depth = self.max_depth
        priors = list(self.accept_priors)
        if len(priors) < self.max_depth:
            priors.extend([priors[-1] if priors else 0.5] * (self.max_depth - len(priors)))
        self._accept_ewma = [_clamp(float(value), 0.0, 1.0) for value in priors[: self.max_depth]]
        self._last_continue_decision: dict[str, int | float | bool | str] | None = None
        self._cycles_observed = 0
        self._attempt_counts = [0 for _ in range(self.max_depth)]

    def should_continue_after_draft(
        self,
        *,
        drafted_depth: int,
        max_depth: int,
        draft_metrics: dict,
    ) -> dict[str, int | float | bool | str]:
        """Return whether generation should draft the next depth."""
        drafted_depth = int(drafted_depth)
        max_depth = int(max_depth)
        if drafted_depth < self.base_depth or drafted_depth >= max_depth:
            decision = {
                "continue": True,
                "action": "continue",
                "reason": "below_base_or_at_max",
                "drafted_depth": drafted_depth,
            }
            self._last_continue_decision = decision
            return decision

        next_depth = drafted_depth + 1
        if next_depth > self.max_depth:
            decision = {
                "continue": False,
                "action": "stop",
                "reason": "beyond_max_depth",
                "drafted_depth": drafted_depth,
            }
            self._last_continue_decision = decision
            return decision

        exploration_reason = self._exploration_reason(next_depth)
        if exploration_reason is not None:
            decision = {
                "continue": True,
                "action": "continue",
                "reason": exploration_reason,
                "drafted_depth": drafted_depth,
                "next_depth": next_depth,
                "cycles_observed": self._cycles_observed,
                "attempted_next_depth": self._attempt_counts[next_depth - 1],
            }
            self._last_continue_decision = decision
            return decision

        prefix_probability = 1.0
        for index in range(drafted_depth):
            prefix_probability *= self._accept_ewma[index]
        next_probability = self._accept_ewma[next_depth - 1]
        confidence_factor = self._confidence_factor(draft_metrics)
        expected_extra_accept = _clamp(
            prefix_probability * next_probability * confidence_factor,
            0.0,
            0.999,
        )
        extra_cost_s = self.draft_cost_s + self.extra_verify_cost_s
        required_extra_accept = max(
            self.min_extra_accept_probability,
            extra_cost_s * self.baseline_tok_s * (1.0 + self.safety_margin),
        )
        should_continue = expected_extra_accept >= required_extra_accept
        decision = {
            "continue": bool(should_continue),
            "action": "continue" if should_continue else "stop",
            "reason": "ev_pass" if should_continue else "ev_fail",
            "drafted_depth": drafted_depth,
            "next_depth": next_depth,
            "prefix_accept_estimate": float(prefix_probability),
            "next_accept_estimate": float(next_probability),
            "confidence_factor": float(confidence_factor),
            "expected_extra_accept": float(expected_extra_accept),
            "required_extra_accept": float(required_extra_accept),
            "extra_cost_s": float(extra_cost_s),
            "baseline_tok_s": float(self.baseline_tok_s),
        }
        self._last_continue_decision = decision
        return decision

    def observe(self, *, attempted_depth: int, accepted_depths: int) -> dict[str, int | float | bool | str | list[float] | dict | None]:
        """Update EWMAs from one cycle outcome and return a loggable decision."""
        attempted_depth = max(1, min(int(attempted_depth), self.max_depth))
        accepted_depths = max(0, min(int(accepted_depths), attempted_depth))
        self._cycles_observed += 1
        for index in range(attempted_depth):
            self._attempt_counts[index] += 1
            accepted = 1.0 if accepted_depths > index else 0.0
            self._accept_ewma[index] = (
                (1.0 - self.ewma_alpha) * self._accept_ewma[index]
                + self.ewma_alpha * accepted
            )
        return {
            "kind": "expected_value",
            "attempted_depth": attempted_depth,
            "accepted_depths": accepted_depths,
            "next_depth": self.current_depth,
            "action": "update_ewma",
            "accept_ewma": [float(v) for v in self._accept_ewma],
            "last_continue_decision": self._last_continue_decision,
            "cycles_observed": self._cycles_observed,
            "attempt_counts": list(self._attempt_counts),
        }

    def _exploration_reason(self, next_depth: int) -> str | None:
        """Avoid self-locking before the policy has measured deeper drafts."""
        if self._cycles_observed < self.warmup_full_depth_cycles:
            return "warmup_full_depth"
        if self._attempt_counts[next_depth - 1] == 0:
            return "unobserved_depth"
        if (
            self.exploration_interval > 0
            and self._cycles_observed > 0
            and self._cycles_observed % self.exploration_interval == 0
        ):
            return "exploration_interval"
        return None

    def _confidence_factor(self, draft_metrics: dict) -> float:
        margin = draft_metrics.get("top2_margin")
        if margin is None:
            return 1.0
        scaled = (float(margin) - self.margin_center) / max(self.margin_scale, 1e-6)
        margin_term = math.tanh(scaled)
        top1_prob = draft_metrics.get("top1_prob_topk")
        prob_term = 0.0
        if top1_prob is not None:
            prob_term = 2.0 * _clamp(float(top1_prob), 0.0, 1.0) - 1.0
        raw = 1.0 + self.confidence_weight * (0.75 * margin_term + 0.25 * prob_term)
        return _clamp(raw, 0.25, 1.75)


@dataclass
class PositionEMADepthPolicy:
    """Per-position marginal-value schedule from challenge row 11.

    ``position_accept_ema[i]`` estimates the conditional probability that
    draft position ``i`` is accepted after every earlier position was
    accepted. The chosen depth maximizes expected committed tokens per unit
    cost using the source's greedy marginal test. Unlike MTPLX's older
    adaptive controllers, this policy deliberately permits depth zero: when
    even the first proposal cannot repay one head step, generation takes the
    target-only serial path for that cycle.
    """

    max_depth: int
    depth_cap: int = 4
    min_depth: int = 0
    accept_ema_alpha: float = 0.15
    head_step_cost_ratio: float = 0.20
    prior: float = 0.85
    prior_decay: float = 0.98
    serial_probe_interval: int = 8

    allows_depth_zero: bool = True

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.depth_cap < 1:
            raise ValueError("depth_cap must be >= 1")
        if self.min_depth < 0:
            raise ValueError("min_depth must be >= 0")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must be <= max_depth")
        if not 0.0 < self.accept_ema_alpha <= 1.0:
            raise ValueError("accept_ema_alpha must be in (0, 1]")
        if self.head_step_cost_ratio <= 0.0:
            raise ValueError("head_step_cost_ratio must be > 0")
        if self.serial_probe_interval < 1:
            raise ValueError("serial_probe_interval must be >= 1")
        self.depth_cap = min(int(self.depth_cap), int(self.max_depth))
        self.min_depth = int(self.min_depth)
        self.allows_depth_zero = self.min_depth == 0
        self.position_accept_ema = [
            float(self.prior) * (float(self.prior_decay) ** index)
            for index in range(self.max_depth)
        ]
        self._serial_skip_cycles = 0
        self.current_depth = 0
        self.recompute_depth()

    def recompute_depth(self) -> int:
        reach = 1.0
        expected = 0.0
        depth = 0
        h = float(self.head_step_cost_ratio)
        while depth < self.depth_cap:
            reach *= self.position_accept_ema[depth]
            threshold = h * (1.0 + expected) / (1.0 + depth * h)
            if reach <= threshold:
                break
            expected += reach
            depth += 1
        self.current_depth = max(self.min_depth, depth)
        return self.current_depth

    def observe_serial_skip(self) -> dict:
        """Schedule a bounded D1 probe so a true D0 decision is not absorbing."""

        previous_depth = self.current_depth
        self._serial_skip_cycles += 1
        action = "hold"
        if self._serial_skip_cycles >= self.serial_probe_interval:
            self._serial_skip_cycles = 0
            self.current_depth = 1
            action = "probe"
        return {
            "kind": "position_ema",
            "previous_depth": previous_depth,
            "attempted_depth": 0,
            "accepted_depths": 0,
            "next_depth": self.current_depth,
            "action": action,
            "accept_ema": [float(value) for value in self.position_accept_ema],
        }

    def observe(self, *, attempted_depth: int, accepted_depths: int) -> dict:
        attempted = max(1, min(int(attempted_depth), self.max_depth))
        accepted = max(0, min(int(accepted_depths), attempted))
        previous_depth = self.current_depth
        self._serial_skip_cycles = 0
        alpha = float(self.accept_ema_alpha)

        for index in range(accepted):
            value = self.position_accept_ema[index]
            self.position_accept_ema[index] = value + alpha * (1.0 - value)

        if accepted < attempted:
            value = self.position_accept_ema[accepted]
            self.position_accept_ema[accepted] = value + alpha * (0.0 - value)
        elif accepted < self.max_depth:
            # A full round is positive evidence about the first unoffered
            # position, allowing a hot chain to widen beyond its current cap.
            value = self.position_accept_ema[accepted]
            self.position_accept_ema[accepted] = value + alpha * (1.0 - value)

        next_depth = self.recompute_depth()
        return {
            "kind": "position_ema",
            "previous_depth": previous_depth,
            "attempted_depth": attempted,
            "accepted_depths": accepted,
            "next_depth": next_depth,
            "action": (
                "increase"
                if next_depth > previous_depth
                else "decrease"
                if next_depth < previous_depth
                else "hold"
            ),
            "accept_ema": [float(value) for value in self.position_accept_ema],
        }


class CostModelDepthPolicy:
    """Cost-model adaptive depth: maximize expected committed tokens per
    wall-clock cycle, not acceptance streaks.

    Ported from omlx's ``_DepthController`` (jundot/omlx v0.5.4rc1,
    omlx/patches/mlx_lm_mtp/batch_generator.py) with the depth-0
    park/exit machinery deliberately left out for now (our 27B lanes
    always profit from speculation; the escape hatch matters for
    head_dim-512/MoE models and can come later). Their measured design
    decisions preserved verbatim:

    - ``score(d) = (1 + p1 + p1 p2 + ...) / t_est(d)``.
    - Acceptance is a token-domain EMA (a property of model/content);
      cost is a wall-clock-horizon EMA (tracks context growth, thermal
      state, and external GPU load at constant real-time responsiveness)
      with a one-off-spike damp.
    - The marginal cost of an extra verify row is the measured slope
      between the cheapest and priciest measured depths, not a constant.
    - Probes are bidirectional and staleness-directed, duty-bounded to
      ~15% of cycles: re-measuring a SHALLOWER rival is what breaks the
      depth-2 lock omlx measured (stale-high t[1] hides depth 1 forever).
    - The cost EMA is per-cycle, NOT staleness-age-weighted: omlx
      measured age-weighting worse (probe-burst noise injected straight
      into the decision; prose re-over-drafted 1.6%).

    Drop-in for the ``AdaptiveDepthPolicy`` interface: ``current_depth``
    plus ``observe(attempted_depth=, accepted_depths=)``. Cycle cost is
    self-timed as the wall interval between observe calls (one observe
    per verify cycle), which deliberately includes the loop's host
    bookkeeping — that tax is part of the real cost of running a cycle.
    """

    ALPHA = 0.08
    TAU_MS = 400.0
    PROBE_PERIOD_MS = 1000.0
    PROBE_PERIOD_MAX_MS = 5000.0
    PROBE_LEN = 4
    PROBE_DUTY = 0.15
    PROBE_MARGIN = 1.15
    SPIKE_RATIO = 2.0
    SPIKE_DAMP = 0.25
    MARGINAL_MS = 7.0
    HYSTERESIS = 1.03
    # Ignore absurd inter-observe gaps (queue waits, tool round-trips in
    # agent serving): a "cycle" above this is not a cycle measurement.
    MAX_CYCLE_MS = 5000.0
    # The generation loop passes its own measured cycle wall-time when it
    # sees this flag; the self-timed inter-observe fallback stays for
    # callers that do not.
    accepts_cycle_ms = True

    def __init__(
        self,
        max_depth: int,
        min_depth: int = 1,
        marginal_ms: float | None = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.max_depth = int(max_depth)
        self.min_depth = max(1, min(int(min_depth), self.max_depth))
        if marginal_ms:
            self.MARGINAL_MS = float(marginal_ms)
        self.current_depth = self.max_depth
        self.p = [0.6] * self.max_depth
        self.t: dict[int, float] = {}
        self.t_age: dict[int, float] = {}
        self.cycles = 0
        self.probe_left = 0
        self._ms_probe = 0.0
        self._ms_explore = 0.0
        self._warmup = list(range(self.max_depth, self.min_depth - 1, -1))
        self._last_observe_s: float | None = None

    # -- cost bookkeeping --------------------------------------------------

    def _time_alpha(self, cycle_ms: float) -> float:
        return 1.0 - math.exp(-max(0.0, float(cycle_ms)) / self.TAU_MS)

    def _update_time(self, used: int, cycle_ms: float) -> None:
        prev = self.t.get(used)
        if prev is None:
            self.t[used] = cycle_ms
            return
        if self._warmup:
            self.t[used] = min(prev, cycle_ms)
            return
        a = self._time_alpha(cycle_ms)
        if cycle_ms > self.SPIKE_RATIO * prev:
            a *= self.SPIKE_DAMP
        self.t[used] = (1.0 - a) * prev + a * cycle_ms

    def _marginal_est(self) -> float:
        if len(self.t) >= 2:
            depths = sorted(self.t)
            lo, hi = depths[0], depths[-1]
            if hi > lo:
                slope = (self.t[hi] - self.t[lo]) / (hi - lo)
                if slope > 0.0:
                    return slope
        return self.MARGINAL_MS

    def _t_est(self, d: int) -> float:
        if d in self.t:
            return self.t[d]
        if not self.t:
            return 30.0 + self.MARGINAL_MS * d
        ref = min(self.t, key=lambda x: abs(x - d))
        return max(1e-3, self.t[ref] + self._marginal_est() * (d - ref))

    def _score(self, d: int) -> float:
        expected = 1.0
        run = 1.0
        for j in range(d):
            run *= self.p[j]
            expected += run
        return expected / max(1e-6, self._t_est(d))

    # -- selection ---------------------------------------------------------

    def _depths(self) -> list[int]:
        return list(range(self.min_depth, self.max_depth + 1))

    def _best(self) -> int:
        cur_score = self._score(self.current_depth)
        best_d, best_score = self.current_depth, cur_score
        for d in self._depths():
            s = self._score(d)
            if s > best_score:
                best_d, best_score = d, s
        if best_d != self.current_depth and best_score < cur_score * self.HYSTERESIS:
            return self.current_depth
        return best_d

    def _best_rival(self) -> int | None:
        best = self._score(self.current_depth)
        if best <= 0.0:
            return self._most_stale()
        rival, rival_score = None, 0.0
        for d in self._depths():
            if d == self.current_depth:
                continue
            s = self._score(d)
            if s > rival_score:
                rival, rival_score = d, s
        if rival is not None and rival_score * self.PROBE_MARGIN >= best:
            return rival
        return None

    def _most_stale(self) -> int | None:
        candidates = [d for d in self._depths() if d != self.current_depth]
        if not candidates:
            return None
        never = [d for d in candidates if d not in self.t]
        if never:
            return never[0]
        return max(candidates, key=lambda d: self.t_age.get(d, 0.0))

    # -- the drop-in interface ---------------------------------------------

    def observe(
        self,
        *,
        attempted_depth: int,
        accepted_depths: int,
        cycle_ms: float | None = None,
    ) -> dict:
        import time as _time

        now = _time.perf_counter()
        if cycle_ms is None and self._last_observe_s is not None:
            cycle_ms = (now - self._last_observe_s) * 1000.0
        if cycle_ms is not None and (cycle_ms > self.MAX_CYCLE_MS or cycle_ms <= 0.0):
            cycle_ms = None
        self._last_observe_s = now

        self.cycles += 1
        used = max(1, min(int(attempted_depth), self.max_depth))
        accepted = max(0, min(int(accepted_depths), used))
        previous_depth = self.current_depth

        a = self.ALPHA
        for j in range(used):
            hit = 1.0 if j < accepted else 0.0
            self.p[j] = (1.0 - a) * self.p[j] + a * hit
            if j >= accepted:
                break

        if cycle_ms is not None:
            self._update_time(used, cycle_ms)
            for d in list(self.t_age):
                self.t_age[d] += cycle_ms
            self.t_age[used] = 0.0
            self._ms_probe += cycle_ms
            self._ms_explore += cycle_ms

        action = "hold"
        if self._warmup:
            # A warmup slot is consumed only once its depth has a real cost
            # sample; the very first observe has no prior timestamp to diff
            # against, so that cycle repeats its depth instead of advancing.
            if cycle_ms is not None and used == self._warmup[0]:
                self._warmup.pop(0)
            if self._warmup:
                self.current_depth = self._warmup[0]
                action = "warmup"
            else:
                self.current_depth = self._best()
                self._ms_probe = 0.0
                action = "warmup_done"
        elif self.probe_left > 0:
            self.probe_left -= 1
            if self.probe_left == 0:
                self.current_depth = self._best()
                self._ms_probe = 0.0
                action = "probe_done"
            else:
                action = "probing"
        else:
            self.current_depth = self._best()
            if self.current_depth != previous_depth:
                action = (
                    "increase" if self.current_depth > previous_depth else "decrease"
                )
            if self.max_depth > self.min_depth and cycle_ms is not None:
                period = max(
                    self.PROBE_PERIOD_MS,
                    self.PROBE_LEN * cycle_ms / self.PROBE_DUTY,
                )
                if self._ms_probe >= period:
                    explore_due = self._ms_explore >= max(
                        self.PROBE_PERIOD_MAX_MS, 2.0 * period
                    )
                    target = self._most_stale() if explore_due else self._best_rival()
                    if target is not None:
                        self.current_depth = target
                        self.probe_left = self.PROBE_LEN
                        self._ms_probe = 0.0
                        if explore_due:
                            self._ms_explore = 0.0
                        action = "probe"

        return {
            "previous_depth": previous_depth,
            "attempted_depth": used,
            "accepted_depths": accepted,
            "next_depth": self.current_depth,
            "action": action,
        }
