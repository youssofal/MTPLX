"""Pure-Python contracts for the stock DFlash2 width sweep."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import statistics


PYTHON_BENCHMARK_PROMPT = """You are working in a Python 3.11 repository. Read the supplied modules and tests, then implement the requested production-safe fix. Preserve public behavior, use typed code, add a focused pytest regression test, and return code only.

"""


@dataclass(frozen=True)
class ExactPrompt:
    """An exact encoded prompt suitable for reuse by every benchmark arm."""

    token_ids: tuple[int, ...]
    token_count: int
    token_sha256: str
    enable_thinking: bool


@dataclass(frozen=True)
class ColdPrefillPrompt:
    """One cold synthetic prefix followed by one fixed test input."""

    cold_prefix_ids: tuple[int, ...]
    test_prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    cold_prefix_tokens: int
    test_prompt_tokens: int
    total_prompt_tokens: int
    cold_prefix_sha256: str
    test_prompt_sha256: str
    token_sha256: str
    enable_thinking: bool


@dataclass(frozen=True)
class DepthBracket:
    """One validated DFlash2 measurement and its adjacent MTP controls."""

    width: int
    candidate_decode_tps: float
    control_before_tps: float
    control_after_tps: float
    validation_passed: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or not 1 <= self.width <= 8
        ):
            raise ValueError("DFlash2 bracket width must be between 1 and 8")
        for field_name in (
            "candidate_decode_tps",
            "control_before_tps",
            "control_after_tps",
        ):
            value = getattr(self, field_name)
            try:
                valid = math.isfinite(value) and value > 0
            except TypeError:
                valid = False
            if not valid:
                raise ValueError(f"{field_name} must be finite and positive")

    @property
    def control_mean_tps(self) -> float:
        return statistics.mean((self.control_before_tps, self.control_after_tps))

    @property
    def normalized_ratio(self) -> float:
        return self.candidate_decode_tps / self.control_mean_tps

    @property
    def drift_fraction(self) -> float:
        return (
            abs(self.control_before_tps - self.control_after_tps)
            / self.control_mean_tps
        )


@dataclass(frozen=True)
class DepthSelection:
    """Normalized stock-width ranking and any unresolved leading tie band."""

    best_widths: tuple[int, ...]
    needs_tiebreak: bool
    normalized_ratios: tuple[tuple[int, float], ...]


def parse_dflash2_widths(raw: str) -> tuple[int, ...]:
    """Parse unique DFlash2 widths supported by the released block-size-8 model."""

    message = "DFlash2 widths must be unique integers between 1 and 8"
    try:
        parts = tuple(part.strip() for part in raw.split(","))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(message) from error
    if not parts or any(not part for part in parts):
        raise ValueError(message)
    try:
        widths = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError(message) from error
    if (
        not widths
        or len(widths) != len(set(widths))
        or any(not 1 <= width <= 8 for width in widths)
    ):
        raise ValueError(message)
    return widths


def build_exact_python_prompt_ids(
    tokenizer,
    *,
    token_count: int = 1024,
) -> ExactPrompt:
    """Build one production-templated prompt and retain its exact ID prefix."""

    if token_count < 1:
        raise ValueError("token_count must be a positive integer")
    messages = [{"role": "user", "content": PYTHON_BENCHMARK_PROMPT * 64}]
    encoded = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if len(encoded) < token_count:
        raise ValueError(
            f"Python benchmark prompt encoded to {len(encoded)} tokens, "
            f"expected at least {token_count}"
        )
    token_ids = tuple(int(token) for token in encoded[:token_count])
    return ExactPrompt(
        token_ids=token_ids,
        token_count=len(token_ids),
        token_sha256=_token_ids_sha256(token_ids),
        enable_thinking=False,
    )


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()


def _coding_agent_prefill_text() -> str:
    from mtplx.prefill_bench import _model_prompt_text

    return _model_prompt_text()


def build_cold_prefill_python_prompt(
    tokenizer,
    *,
    cold_prefix_tokens: int,
    test_prompt_tokens: int = 1024,
) -> ColdPrefillPrompt:
    """Compose an exact cold coding prefix and a separately counted test input."""

    if type(cold_prefix_tokens) is not int or cold_prefix_tokens <= 0:
        raise ValueError("cold_prefix_tokens must be a positive integer")
    if type(test_prompt_tokens) is not int or test_prompt_tokens <= 0:
        raise ValueError("test_prompt_tokens must be a positive integer")

    filler = _coding_agent_prefill_text()
    raw_prefix_ids = [int(token) for token in tokenizer.encode(filler)]
    if not raw_prefix_ids:
        raise ValueError("coding-agent prefill text encoded to no tokens")
    repeated_prefix_ids = raw_prefix_ids * (
        (cold_prefix_tokens + len(raw_prefix_ids) - 1) // len(raw_prefix_ids)
    )
    prefix_ids = tuple(repeated_prefix_ids[:cold_prefix_tokens])
    test_prompt = build_exact_python_prompt_ids(
        tokenizer,
        token_count=test_prompt_tokens,
    )
    test_ids = test_prompt.token_ids
    combined_ids = prefix_ids + test_ids
    return ColdPrefillPrompt(
        cold_prefix_ids=prefix_ids,
        test_prompt_ids=test_ids,
        token_ids=combined_ids,
        cold_prefix_tokens=len(prefix_ids),
        test_prompt_tokens=len(test_ids),
        total_prompt_tokens=len(combined_ids),
        cold_prefix_sha256=_token_ids_sha256(prefix_ids),
        test_prompt_sha256=_token_ids_sha256(test_ids),
        token_sha256=_token_ids_sha256(combined_ids),
        enable_thinking=False,
    )


def select_stock_depth(rows: list[DepthBracket]) -> DepthSelection:
    """Rank valid widths by median bracket-normalized decode throughput."""

    if not rows or any(not row.validation_passed for row in rows):
        raise ValueError("every ranked DFlash2 bracket must pass benchmark validation")

    grouped: dict[int, list[DepthBracket]] = {}
    for row in rows:
        grouped.setdefault(row.width, []).append(row)

    ranking = [
        (
            width,
            statistics.median(row.normalized_ratio for row in width_rows),
            max(row.drift_fraction for row in width_rows),
        )
        for width, width_rows in grouped.items()
    ]
    ranking.sort(key=lambda item: (-item[1], item[0]))
    leader = ranking[0]
    tie_band = tuple(
        sorted(
            width
            for width, ratio, drift in ranking
            if leader[1] - ratio <= max(leader[2], drift)
        )
    )
    return DepthSelection(
        best_widths=tie_band,
        needs_tiebreak=len(tie_band) > 1,
        normalized_ratios=tuple((width, ratio) for width, ratio, _ in ranking),
    )
