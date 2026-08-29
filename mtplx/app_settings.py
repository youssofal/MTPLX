"""Read-only access to the MTPLX macOS app's persisted configuration.

The app stores its configuration at
``~/Library/Application Support/MTPLX/settings.json`` with snake_case keys
(see ``AppConfiguration.swift`` ``CodingKeys`` — the sync source for the
field names read here). The CLI never writes this file; it only reads it so
``mtplx start`` can offer "same as the MTPLX app" and reuse the app's model,
model directory, port, and API key instead of walking a returning user through
onboarding the app already completed.

Dates in the file are Apple-epoch (seconds since 2001-01-01); convert with
``APPLE_EPOCH_OFFSET_S`` when a Unix timestamp is needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

APPLE_EPOCH_OFFSET_S = 978_307_200
APP_SETTINGS_PATH_ENV = "MTPLX_APP_SETTINGS_PATH"
DEFAULT_APP_SETTINGS_PATH = Path(
    "~/Library/Application Support/MTPLX/settings.json"
)


@dataclass(frozen=True)
class AppSettings:
    """The subset of the app configuration the CLI acts on."""

    path: Path
    model: str | None
    model_dir: str | None
    host: str | None
    port: int | None
    api_key: str | None
    last_launch_target: str | None
    onboarding_completed_at: float | None  # Unix seconds
    raw: Mapping[str, Any]

    @property
    def onboarding_completed(self) -> bool:
        return self.onboarding_completed_at is not None


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _clean_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _apple_date_to_unix(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) + APPLE_EPOCH_OFFSET_S


def app_settings_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get(APP_SETTINGS_PATH_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_APP_SETTINGS_PATH.expanduser()


def read_app_settings(path: str | Path | None = None) -> AppSettings | None:
    """Parse the app's settings.json; ``None`` when absent or unreadable.

    Every field is optional: the reader degrades to ``None`` values rather
    than raising, because a settings file written by any app version (or a
    partially-written one) must never break CLI startup.
    """

    settings_file = app_settings_path(path)
    try:
        data = json.loads(settings_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return AppSettings(
        path=settings_file,
        model=_clean_str(data.get("model")),
        model_dir=_clean_str(data.get("model_dir")),
        host=_clean_str(data.get("host")),
        port=_clean_int(data.get("port")),
        api_key=_clean_str(data.get("api_key")),
        last_launch_target=_clean_str(data.get("last_launch_target")),
        onboarding_completed_at=_apple_date_to_unix(
            data.get("onboarding_completed_at")
        ),
        raw=data,
    )
