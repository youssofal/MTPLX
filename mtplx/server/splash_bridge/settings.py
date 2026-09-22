"""The live generation defaults a Splash-backed server applies.

The MLX server keeps sampling defaults that the app's parameter panel and the
browser chat both read and write through ``/v1/mtplx/settings``, and fills
them into any request that leaves a field out. The app's own chat sends no
sampling fields at all, so without this every Splash turn ran on the engine's
built-in defaults whatever the panel showed.

Splash's sampler is narrower than MTPLX's: temperature in [0, 2], top_p in
(0, 1], top_k in [1, 32], and no presence or frequency penalty. A setting
outside that is moved to the nearest value Splash runs, and the reply to the
write says what moved, so a panel always shows what the engine will use.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

# Splash's own defaults (its server's frontend), in force until a client
# changes them.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
TEMPERATURE_MAX = 2.0
TOP_P_MIN = 0.01
TOP_K_MIN = 1
TOP_K_MAX = 32
REASONING_MODES = ("auto", "on", "off")
# "auto" is MTPLX's "the model's default level", not a level Splash knows.
EFFORT_AUTO = "auto"
# What Splash itself folds into its template's levels.
EFFORT_ALIASES = {"high": "xhigh", "max": "xhigh", "minimal": "low"}

# The package's effort levels and the template's default, read per call so a
# model switch is followed: ``((), None)`` when the template has none.
EffortPolicy = Callable[[], "tuple[tuple[str, ...], str | None]"]

# Settings MTPLX clients send that describe the MLX engine. Accepted so a
# client syncing its whole panel is not refused, and named in the reply.
ENGINE_FIXED = {
    "generation_mode": "Splash always speculates with its DFlash 2 draft.",
    "depth": "Splash's draft length is fixed by the package.",
    "adaptive_policy": "Splash's draft length is fixed by the package.",
    "draft_temperature": "Splash's draft sampler is fixed by the package.",
    "draft_top_p": "Splash's draft sampler is fixed by the package.",
    "draft_top_k": "Splash's draft sampler is fixed by the package.",
    "prefill_chunk_tokens": "Splash sizes prefill from its compiled geometry.",
    "stream_interval": "Splash streams every token.",
    "reasoning_parser": "Splash parses Qwen thinking itself.",
}
PENALTIES = ("presence_penalty", "frequency_penalty")

# Generation routes whose sampling fields share OpenAI's names.
SAMPLED_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class SamplingSettings:
    """Live defaults, shared by every client of one bridge process."""

    def __init__(self, efforts: EffortPolicy | None = None) -> None:
        self._lock = threading.Lock()
        self._efforts = efforts or (lambda: ((), None))
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.top_k = DEFAULT_TOP_K
        self.max_response_tokens: int | None = None
        self.reasoning = "auto"
        self.reasoning_effort = EFFORT_AUTO

    def reasoning_policy(self) -> dict[str, Any]:
        """The reasoning block of MTPLX's model controls, for this package.

        The app's parameter panel shows an effort picker when it lists
        levels, and the browser chat builds its Thinking selector from it.
        """
        levels, default = self._efforts()
        return {
            "supported": True,
            "parser": "qwen3",
            "display_name": "Qwen think tags",
            "modes": list(REASONING_MODES),
            "default_mode": "auto",
            "default": "auto",
            "effort_levels": list(levels),
            "default_effort": default,
        }

    def payload(self) -> dict[str, Any]:
        """The fields of the MLX server's settings reply that apply to Splash."""
        with self._lock:
            return {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "max_response_tokens": self.max_response_tokens,
                "reasoning": self.reasoning,
                "reasoning_effort": self.reasoning_effort,
                "reasoning_policy": self.reasoning_policy(),
                "enable_thinking": self.reasoning != "off",
                "sampling_defaults": {
                    "temperature": DEFAULT_TEMPERATURE,
                    "top_p": DEFAULT_TOP_P,
                    "top_k": DEFAULT_TOP_K,
                    "family_default_reason": "Splash engine defaults",
                },
            }

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a settings write; report what changed, moved or was ignored."""
        applied: dict[str, Any] = {}
        adjusted: dict[str, str] = {}
        ignored: dict[str, str] = {}
        with self._lock:
            for key, value in patch.items():
                if value is None:
                    continue
                if key in ENGINE_FIXED:
                    ignored[key] = ENGINE_FIXED[key]
                elif key in PENALTIES:
                    number = _number(value)
                    if number is None:
                        ignored[key] = "Not a number."
                    elif number != 0:
                        adjusted[key] = "Splash samples without penalties; kept at 0."
                elif key == "temperature":
                    number = _number(value)
                    if number is None:
                        ignored[key] = "Not a number."
                        continue
                    clamped = min(TEMPERATURE_MAX, max(0.0, number))
                    if clamped != number:
                        adjusted[key] = f"Splash runs temperature 0 to {TEMPERATURE_MAX:g}."
                    self.temperature = applied[key] = clamped
                elif key == "top_p":
                    number = _number(value)
                    if number is None:
                        ignored[key] = "Not a number."
                        continue
                    clamped = min(1.0, max(TOP_P_MIN, number))
                    if clamped != number:
                        adjusted[key] = f"Splash runs top_p {TOP_P_MIN:g} to 1."
                    self.top_p = applied[key] = clamped
                elif key == "top_k":
                    number = _number(value)
                    if number is None:
                        ignored[key] = "Not a number."
                        continue
                    if number == 0:
                        # MTPLX's 0 means "no top-k filter". Splash always
                        # samples from a top-k set, and the nearest thing to
                        # no filter is its widest one, not a greedy top-1.
                        clamped = TOP_K_MAX
                    else:
                        clamped = int(min(TOP_K_MAX, max(TOP_K_MIN, int(number))))
                    if clamped != number:
                        adjusted[key] = f"Splash runs top_k {TOP_K_MIN} to {TOP_K_MAX}."
                    self.top_k = applied[key] = clamped
                elif key == "max_response_tokens":
                    number = _number(value)
                    if number is None or number < 1:
                        ignored[key] = "Must be a positive whole number."
                        continue
                    self.max_response_tokens = applied[key] = int(number)
                elif key == "reasoning":
                    if value not in REASONING_MODES:
                        ignored[key] = "Must be auto, on or off."
                        continue
                    self.reasoning = applied[key] = value
                elif key == "reasoning_effort":
                    effort = EFFORT_ALIASES.get(str(value).lower(), str(value).lower())
                    levels, _default = self._efforts()
                    if effort != EFFORT_AUTO and effort not in levels:
                        ignored[key] = (
                            "This model's levels are "
                            + (", ".join(levels) if levels else "none")
                            + "."
                        )
                        continue
                    self.reasoning_effort = applied[key] = effort
                elif key == "enable_thinking":
                    if patch.get("reasoning") is not None:
                        continue  # the three-way setting says it exactly
                    if not isinstance(value, bool):
                        ignored[key] = "Must be true or false."
                        continue
                    self.reasoning = applied["reasoning"] = "on" if value else "off"
                else:
                    ignored[key] = "Not a setting the Splash engine has."
        result: dict[str, Any] = {}
        if applied:
            result["applied"] = applied
        if adjusted:
            result["adjusted"] = adjusted
        if ignored:
            result["ignored"] = ignored
        return result

    def apply(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """``body`` with every field the client left out filled from here."""
        if path not in SAMPLED_PATHS:
            return body
        out = dict(body)
        with self._lock:
            defaults = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            }
            max_tokens = self.max_response_tokens
            reasoning = self.reasoning
            effort = self.reasoning_effort
        for key, value in defaults.items():
            if out.get(key) is None:
                out[key] = value
        if path != "/v1/chat/completions":
            return out
        if max_tokens and all(
            out.get(key) is None for key in ("max_tokens", "max_completion_tokens")
        ):
            out["max_tokens"] = max_tokens
        if out.get("reasoning_effort") == EFFORT_AUTO:
            # MTPLX clients may say "auto" for "the default level"; Splash
            # would refuse it, and leaving it out means exactly that.
            del out["reasoning_effort"]
        # Splash turns thinking off with reasoning_effort "none" and ignores
        # MTPLX's enable_thinking, so "Hide thinking" has to be translated.
        if out.get("reasoning_effort") is None:
            thinking = out.get("enable_thinking")
            kwargs = out.get("chat_template_kwargs")
            if thinking is None and isinstance(kwargs, dict):
                thinking = kwargs.get("enable_thinking")
            if thinking is None:
                thinking = {"on": True, "off": False}.get(reasoning)
            if thinking is False:
                out["reasoning_effort"] = "none"
            elif effort != EFFORT_AUTO and effort in self._efforts()[0]:
                out["reasoning_effort"] = effort
        return out
