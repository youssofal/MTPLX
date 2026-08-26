from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from mtplx.benchmarks import dflash2_contract as contract
from mtplx.benchmarks.dflash2_contract import (
    DepthBracket,
    ExactPrompt,
    build_exact_python_prompt_ids,
    parse_dflash2_widths,
    select_stock_depth,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert "Python" in messages[0]["content"]
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return list(range(1200))


class PrefixTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return [101, 102, 103, 104]


def test_widths_are_unique_integers_bounded_by_checkpoint():
    assert parse_dflash2_widths("1,3,8") == (1, 3, 8)

    for raw in (
        "0,1",
        "8,9",
        "1,1",
        "",
        "1,two",
        "1,,2",
        ",1",
        "1,",
    ):
        with pytest.raises(
            ValueError,
            match="unique integers between 1 and 8",
        ):
            parse_dflash2_widths(raw)


def test_python_prompt_is_exactly_1024_token_ids():
    prompt = build_exact_python_prompt_ids(FakeTokenizer(), token_count=1024)

    assert prompt.token_ids == tuple(range(1024))
    assert prompt.token_count == 1024
    expected_hash = hashlib.sha256(
        ",".join(map(str, range(1024))).encode()
    ).hexdigest()
    assert prompt.token_sha256 == expected_hash
    assert prompt.enable_thinking is False


def test_python_prompt_rejects_an_encoded_prefix_that_is_too_short():
    with pytest.raises(ValueError, match="expected at least 1201"):
        build_exact_python_prompt_ids(FakeTokenizer(), token_count=1201)


def test_cold_prefill_prompt_records_prefix_and_test_input_separately(monkeypatch):
    monkeypatch.setattr(
        contract,
        "_coding_agent_prefill_text",
        lambda: "abcdef",
        raising=False,
    )

    prompt = contract.build_cold_prefill_python_prompt(
        PrefixTokenizer(),
        cold_prefix_tokens=3,
        test_prompt_tokens=4,
    )

    assert prompt.cold_prefix_ids == (97, 98, 99)
    assert prompt.test_prompt_ids == (101, 102, 103, 104)
    assert prompt.token_ids == (97, 98, 99, 101, 102, 103, 104)
    assert prompt.cold_prefix_tokens == 3
    assert prompt.test_prompt_tokens == 4
    assert prompt.total_prompt_tokens == 7
    assert len(prompt.cold_prefix_sha256) == 64
    assert len(prompt.test_prompt_sha256) == 64
    assert len(prompt.token_sha256) == 64


def test_contract_records_are_immutable():
    prompt = ExactPrompt((1, 2), 2, "hash", False)
    row = DepthBracket(2, 60.0, 59.0, 61.0, True)
    selection = select_stock_depth([row])

    with pytest.raises(FrozenInstanceError):
        prompt.token_count = 3
    with pytest.raises(FrozenInstanceError):
        row.width = 3
    with pytest.raises(FrozenInstanceError):
        selection.needs_tiebreak = True


def test_depth_bracket_rejects_unsupported_width():
    with pytest.raises(ValueError, match="width must be between 1 and 8"):
        DepthBracket(9, 60.0, 59.0, 61.0, True)


@pytest.mark.parametrize(
    "field",
    ("candidate_decode_tps", "control_before_tps", "control_after_tps"),
)
@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_depth_bracket_rejects_non_positive_or_non_finite_tps(field, value):
    values = {
        "width": 2,
        "candidate_decode_tps": 60.0,
        "control_before_tps": 59.0,
        "control_after_tps": 61.0,
        "validation_passed": True,
    }
    values[field] = value

    with pytest.raises(ValueError, match=rf"{field} must be finite and positive"):
        DepthBracket(**values)


def test_selection_ranks_median_mtp_normalized_ratio():
    rows = [
        # width 2 median: median(1.00, 1.20, 1.10) = 1.10
        DepthBracket(2, 100.0, 100.0, 100.0, True),
        DepthBracket(2, 120.0, 100.0, 100.0, True),
        DepthBracket(2, 110.0, 100.0, 100.0, True),
        # width 3 raw TPS is lower, but normalized median is higher: 1.15.
        DepthBracket(3, 57.5, 50.0, 50.0, True),
        DepthBracket(3, 56.0, 50.0, 50.0, True),
        DepthBracket(3, 58.0, 50.0, 50.0, True),
    ]

    selected = select_stock_depth(rows)

    assert selected.best_widths == (3,)
    assert selected.needs_tiebreak is False
    assert selected.normalized_ratios == ((3, 1.15), (2, 1.1))


def test_selection_reports_leaders_inside_full_control_drift_as_tie_band():
    rows = [
        DepthBracket(2, 60.0, 59.0, 61.0, True),
        DepthBracket(3, 60.3, 59.0, 61.0, True),
        DepthBracket(4, 55.0, 59.0, 61.0, True),
    ]

    selected = select_stock_depth(rows)

    assert selected.best_widths == (2, 3)
    assert selected.needs_tiebreak is True


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [DepthBracket(8, 70.0, 60.0, 60.0, False)],
    ],
)
def test_empty_or_invalid_brackets_cannot_enter_selection(rows):
    with pytest.raises(ValueError, match="validation"):
        select_stock_depth(rows)
