"""Static checks for the macOS app localization resources."""

from __future__ import annotations

import re
from pathlib import Path


_ROOT = Path(__file__).parents[1]
_LOCALIZABLE = (
    _ROOT
    / "apps"
    / "MTPLXApp"
    / "Sources"
    / "MTPLXAppHost"
    / "Resources"
    / "Localization"
    / "zh-Hans.lproj"
    / "Localizable.strings"
)
_BUILD_SCRIPT = _ROOT / "apps" / "MTPLXApp" / "script" / "build_and_run.sh"
_ENTRY_RE = re.compile(r'^"((?:\\.|[^"\\])*)"\s*=\s*"((?:\\.|[^"\\])*)";$')
_FORMAT_RE = re.compile(
    r"%(?:(\d+)\$)?[-+#0']*(?:\d+|\*)?(?:\.(?:\d+|\*))?"
    r"(?:hh|h|ll|l|q|L|z|t|j)?([@diufFeEgG%])"
)


def _entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        _LOCALIZABLE.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("/*"):
            continue
        match = _ENTRY_RE.fullmatch(line)
        assert match is not None, f"invalid .strings entry on line {line_number}"
        entries.append((match.group(1), match.group(2)))
    return entries


def _format_signature(value: str) -> list[tuple[int, str]]:
    signature: list[tuple[int, str]] = []
    next_position = 1
    for match in _FORMAT_RE.finditer(value):
        conversion = match.group(2)
        if conversion == "%":
            continue
        if match.group(1) is None:
            position = next_position
            next_position += 1
        else:
            position = int(match.group(1))
        signature.append((position, conversion))
    return sorted(signature)


def test_simplified_chinese_strings_are_unique_and_format_safe():
    entries = _entries()
    keys = [key for key, _ in entries]

    assert len(keys) == len(set(keys)), "duplicate localization keys"
    assert len(entries) >= 600, "the Simplified Chinese catalog looks truncated"

    mismatches = {
        key: (_format_signature(key), _format_signature(value))
        for key, value in entries
        if _format_signature(key) != _format_signature(value)
    }
    assert not mismatches, f"format placeholder mismatch: {mismatches}"


def test_app_bundle_copies_language_directories_generically():
    build_script = _BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "Resources/Localization" in build_script
    assert "-name '*.lproj'" in build_script
    assert "CFBundleDevelopmentRegion" in build_script
