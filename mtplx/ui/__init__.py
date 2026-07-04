"""MTPLX terminal UI helpers.

This package is import-on-demand from command handlers. It is intentionally
NOT imported at the top of ``mtplx/__init__.py`` so the package can survive in
fresh venvs that do not have ``rich`` installed.

Each module gracefully degrades to plain stdlib ``print`` when ``rich`` is not
available, so the CLI never hard-fails on a missing dependency.
"""

from __future__ import annotations

from .banner import banner_text, render_banner
from .chat_printer import ChatPrinter
from .panels import render_startup_panel
from .progress import ModelLoadProgress

__all__ = [
    "banner_text",
    "render_banner",
    "render_startup_panel",
    "ChatPrinter",
    "ModelLoadProgress",
    "pretty_path",
]


def __getattr__(name: str):
    # ``pretty_path`` lives in ``onboarding`` as ``_pretty_path``. Re-export it
    # lazily so command handlers can ``from mtplx.ui import pretty_path``
    # without eagerly importing the onboarding dependency chain on every
    # ``mtplx.ui`` import (this package must stay import-cheap for fresh venvs).
    if name == "pretty_path":
        from .onboarding import _pretty_path

        return _pretty_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
