"""The family-serve verify lane produces AR + one speculative row; a build must
not demand the tune lane's D1..Dn sweep from it."""

from __future__ import annotations

from pathlib import Path

import pytest

from mtplx.commands import forge


FAMILY_SERVE_ROWS = [
    {"depth": 0, "tok_s": 45.4, "multiplier_vs_ar": 1.0, "lane": "family-serve"},
    {"depth": 3, "tok_s": 81.7, "multiplier_vs_ar": 1.8, "lane": "family-serve"},
]


def test_family_serve_build_does_not_require_every_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forge, "_verify_rows_lane", lambda path: "family-serve")
    assert forge._build_requires_all_depths(Path("/pack")) is False
    forge._require_verify_rows(FAMILY_SERVE_ROWS, require_all_depths=False, verify_depths=(1, 2, 3))


def test_tune_lane_build_still_requires_every_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forge, "_verify_rows_lane", lambda path: "tune")
    assert forge._build_requires_all_depths(Path("/pack")) is True
    with pytest.raises(forge.ForgeError, match="D1, D2"):
        forge._require_verify_rows(FAMILY_SERVE_ROWS, require_all_depths=True, verify_depths=(1, 2, 3))
