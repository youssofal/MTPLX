"""A chip newer than this build knows takes the newest known path, and is told so (#695).

Before, the name was matched against M1 to M5: an M6 was "unknown", lost the
tensor-unit report and fell through every policy keyed on generation.
"""

from __future__ import annotations

import pytest

import mtplx.hardware as hardware
from mtplx.default_models import select_default_model


@pytest.mark.parametrize(
    "chip, expected",
    [
        ("Apple M5 Max", "m5"),
        ("Apple M6", "m5"),
        ("Apple M6 Max", "m5"),
        ("Apple M7 Ultra", "m5"),
        ("Apple M12 Pro", "m5"),
        ("Apple M10", "m5"),  # two digits: never read as an M1
        ("Apple M4 Pro", "m4"),
        ("Apple M1 Ultra", "m1"),
    ],
)
def test_a_newer_chip_is_classified_as_the_newest_known(chip, expected):
    assert hardware.classify_apple_silicon_generation(chip, system="Darwin", machine="arm64") == expected


def test_a_name_without_a_generation_stays_unknown():
    for chip in ("", "Apple processor", "Apple M Max", "Apple M0"):
        assert hardware.classify_apple_silicon_generation(chip, system="Darwin", machine="arm64") == "unknown"


def test_only_a_newer_chip_gets_the_note():
    assert hardware.newer_chip_note("Apple M5 Max") is None
    assert hardware.newer_chip_note("Apple M2") is None
    assert hardware.newer_chip_note("") is None
    note = hardware.newer_chip_note("Apple M6 Max")
    assert note is not None and "Apple M6 Max is newer than this MTPLX build knows" in note
    assert f"M{hardware.NEWEST_KNOWN_GENERATION}" in note


def _pretend(monkeypatch, chip: str) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "_hardware_json", lambda: {"SPHardwareDataType": [{"chip_type": chip}]})
    monkeypatch.setattr(hardware, "_display_json", lambda: {})

    def run_text(*cmd, **_kwargs):
        if "machdep.cpu.brand_string" in cmd:
            return chip
        if "hw.memsize" in cmd:
            return str(128 * 1024**3)
        if cmd and cmd[0] == "sw_vers":
            return "27.0"
        return ""

    monkeypatch.setattr(hardware, "_run_text", run_text)
    monkeypatch.setattr(hardware, "_dist_version", lambda name: "0.32.2" if name == "mlx" else "0.31.3")


def test_detection_reports_an_m6_as_apple_silicon_with_the_note(monkeypatch):
    _pretend(monkeypatch, "Apple M6 Max")
    detected = hardware.detect_apple_silicon()
    assert detected["chip"] == "Apple M6 Max"
    assert detected["apple_silicon_generation"] == "m5"
    assert detected["is_apple_silicon"] is True
    assert "newer than this MTPLX build knows" in detected["newer_chip_note"]


def test_detection_of_a_known_chip_is_unchanged(monkeypatch):
    _pretend(monkeypatch, "Apple M5 Max")
    detected = hardware.detect_apple_silicon()
    assert detected["apple_silicon_generation"] == "m5" and detected["newer_chip_note"] is None


def test_inspection_gives_an_m6_the_tensor_unit_report_and_says_why(monkeypatch):
    _pretend(monkeypatch, "Apple M6 Max")
    report = hardware.inspect_hardware()
    assert report["apple_silicon_generation"] == "m5"
    assert report["m5_neural_accelerator_eligible"] is True
    assert any("newer than this MTPLX build knows" in warning for warning in report["warnings"])
    # A measured figure belongs to the chip it was measured on.
    assert report["memory_bandwidth_class_gb_s"] is None


def test_an_m6_gets_the_default_pack_an_m5_gets():
    def pick(chip: str):
        generation = hardware.classify_apple_silicon_generation(chip, system="Darwin", machine="arm64")
        return select_default_model(
            hardware={"chip": chip, "apple_silicon_generation": generation, "memory_gib": 128.0}
        )

    newer, known = pick("Apple M6 Max"), pick("Apple M5 Max")
    assert (newer.model, newer.variant) == (known.model, known.variant)
    assert newer.variant != "fp16"
